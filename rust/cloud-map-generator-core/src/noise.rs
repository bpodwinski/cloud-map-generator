//! Port of `cloudgen/noise.py`: seeded 3D gradient noise (improved Perlin), cellular noise (Worley), fBm and
//! the octave helpers, evaluated on 3D points (unit directions times a frequency) so the field has no seam on
//! the sphere.
//!
//! Precision follows Python. `Perlin3` runs in the dtype of its coordinates (generic over `f32`/`f64`, as numpy
//! does); `Worley3` takes the cell index and the fraction in the dtype of its points and then does every
//! distance in `f32`; `fbm` and `cell_edges` cast their points to `f32` first (the caller does that cast: they
//! take `[f32; 3]`) and accumulate in `f32` with the amplitudes, frequencies and the normalization computed in
//! `f64` and rounded per use.
//!
//! Random draws are arguments (README rule 3): [`Perlin3::new`], [`Worley3::new`], [`FbmPlan`] and
//! [`CellEdgesPlan`] hold the tables Python drew with numpy; each has a `draw` constructor that draws the same
//! plan with [`crate::rng::Rng`] (statistically equivalent, never numpy's bit stream). The chunking of fbm and
//! cell_edges in Python (`1 << 20` points) is not part of the semantics and is not ported: no float32 sum
//! crosses a point.

use crate::rng::Rng;
use crate::sphere::Vec3;
use num_traits::Float;

fn c<F: Float>(x: f64) -> F {
    F::from(x).expect("float conversion")
}

/// numpy's `np.minimum` on two floats: NaN propagates (Rust's `f32::min` would ignore it).
fn np_min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else if b < a {
        b
    } else {
        a
    }
}

/// `np.floor(x).astype(np.int64)`: a NaN or out-of-range value becomes `i64::MIN`, as numpy does on x86-64.
fn floor_to_i64<F: Float>(x: F) -> i64 {
    x.floor().to_i64().unwrap_or(i64::MIN)
}

// ---------------------------------------------------------------------------------------------------------
// Constants

/// Standard deviation of one Perlin3 octave over uniformly random points (measured over 2M points). fBm divides
/// by it so its output has unit variance by construction.
pub const PERLIN3_STD: f64 = 0.2692;
/// Same measurement for |noise| (billow octaves: rounded puffs separated by sharp creases): the mean.
pub const PERLIN3_ABS_MEAN: f64 = 0.2171;
/// Same measurement for |noise|: the standard deviation.
pub const PERLIN3_ABS_STD: f64 = 0.1542;

/// Python default of `lacunarity` in `octaves_below`, `octave_ladder` and `fbm`.
pub const DEFAULT_LACUNARITY: f64 = 2.03;
/// Python default of `top` in `octave_ladder`.
pub const DEFAULT_TOP: f64 = 0.95;
/// Python default of `min_texels` in `map_max_frequency`.
pub const DEFAULT_MIN_TEXELS: f64 = 4.0;
/// Python default of `gain` in `fbm`.
pub const DEFAULT_GAIN: f64 = 0.5;

// ---------------------------------------------------------------------------------------------------------
// Perlin

/// Port of `_fade`: the Perlin fade curve `6t^5 - 15t^4 + 10t^3` as `t*t*t*(t*(t*6 - 15) + 10)`, in the
/// precision of `t`, not clamped.
pub fn fade<F: Float>(t: F) -> F {
    t * t * t * (t * (t * c(6.0) - c(15.0)) + c(10.0))
}

/// Port of `_grad`: the gradient selected by `h & 15` dotted with `(x, y, z)`: `u = x` if `h < 8` else `y`;
/// `v = y` if `h < 4`, `x` if `h` is 12 or 14, else `z`; bit 0 of `h` negates `u` and bit 1 negates `v`.
pub fn grad<F: Float>(h: i32, x: F, y: F, z: F) -> F {
    let h = h & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 != 0 { -u } else { u }) + (if h & 2 != 0 { -v } else { v })
}

fn lerp<F: Float>(t: F, a: F, b: F) -> F {
    a + t * (b - a)
}

/// Port of `Perlin3`: improved Perlin noise over a doubled permutation table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Perlin3 {
    /// The DOUBLED table `concat(perm, perm)`, 512 entries (the kernel reads `p[A + 1]` up to index 511).
    pub perm: Vec<i32>,
}

impl Perlin3 {
    /// `Perlin3.__init__` with the permutation given (the 256-entry draw as recorded by the golden plans).
    pub fn new(perm: &[i32]) -> Perlin3 {
        assert_eq!(perm.len(), 256, "a Perlin3 permutation has 256 entries");
        Perlin3 { perm: perm.iter().chain(perm.iter()).copied().collect() }
    }

    /// `Perlin3.__init__` with the permutation drawn from `rng` (Python: `default_rng(seed).permutation(256)`).
    pub fn draw(rng: &mut Rng) -> Perlin3 {
        let p: Vec<i32> = rng.permutation(256).into_iter().map(|v| v as i32).collect();
        Perlin3::new(&p)
    }

    /// `Perlin3(seed)`: the permutation of the generator seeded with `seed`.
    pub fn from_seed(seed: u64) -> Perlin3 {
        Perlin3::draw(&mut Rng::new(seed))
    }

    /// The noise at one point, in the precision of the coordinates. The lattice index wraps by `& 255`
    /// (two's complement for negative cells).
    pub fn at<F: Float>(&self, x: F, y: F, z: F) -> F {
        let (xi, yi, zi) = (x.floor(), y.floor(), z.floor());
        let (xf, yf, zf) = (x - xi, y - yi, z - zi);
        let cell = |v: F| (floor_to_i64(v) & 255) as usize;
        let (xc, yc, zc) = (cell(xi), cell(yi), cell(zi));
        let (u, v, w) = (fade(xf), fade(yf), fade(zf));
        let p = &self.perm;
        let ix = |i: i64| p[i as usize] as i64;
        let a = ix(xc as i64) + yc as i64;
        let aa = ix(a) + zc as i64;
        let ab = ix(a + 1) + zc as i64;
        let b = ix(xc as i64 + 1) + yc as i64;
        let ba = ix(b) + zc as i64;
        let bb = ix(b + 1) + zc as i64;
        let one = F::one();
        let (x1, y1, z1) = (xf - one, yf - one, zf - one);
        let g = |h: i64, gx: F, gy: F, gz: F| grad(p[h as usize], gx, gy, gz);
        lerp(
            w,
            lerp(
                v,
                lerp(u, g(aa, xf, yf, zf), g(ba, x1, yf, zf)),
                lerp(u, g(ab, xf, y1, zf), g(bb, x1, y1, zf)),
            ),
            lerp(
                v,
                lerp(u, g(aa + 1, xf, yf, z1), g(ba + 1, x1, yf, z1)),
                lerp(u, g(ab + 1, xf, y1, z1), g(bb + 1, x1, y1, z1)),
            ),
        )
    }

    /// `Perlin3.__call__`: the noise at the points `(x[i], y[i], z[i])`.
    pub fn call<F: Float + Send + Sync>(&self, x: &[F], y: &[F], z: &[F]) -> Vec<F> {
        assert!(x.len() == y.len() && y.len() == z.len(), "x, y, z must have the same length");
        crate::par::map_index(x.len(), |i| self.at(x[i], y[i], z[i]))
    }
}

// ---------------------------------------------------------------------------------------------------------
// Rotations

type Mat3 = [[f64; 3]; 3];

fn det3(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// numpy's `np.sign` on a float: -1, 0 or 1, NaN stays NaN.
fn np_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x * 0.0 // 0 for +-0, NaN for NaN
    }
}

/// Householder QR of a 3x3 matrix in LAPACK's convention (`dgeqr2` + `dorg2r`): returns Q and the diagonal of R
/// (a diagonal entry is `-sign(alpha) |x|`, so it may be negative). A reflector with nothing below the diagonal
/// is the identity and leaves the diagonal entry as it is.
fn householder_qr(a: &Mat3) -> (Mat3, [f64; 3]) {
    let mut r = *a;
    let mut q: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut diag = [0.0; 3];
    for j in 0..3 {
        let alpha = r[j][j];
        let xnorm = ((j + 1)..3).map(|i| r[i][j] * r[i][j]).sum::<f64>().sqrt();
        if xnorm == 0.0 {
            diag[j] = alpha;
            continue;
        }
        let beta = -alpha.hypot(xnorm).copysign(alpha);
        let tau = (beta - alpha) / beta;
        // The reflector H = I - tau v v^T, with v = (1, r[j+1..][j] / (alpha - beta)) on rows j..3.
        let mut v = [0.0; 3];
        v[j] = 1.0;
        for i in (j + 1)..3 {
            v[i] = r[i][j] / (alpha - beta);
        }
        // R <- H R on the columns j.. (column j becomes (beta, 0, 0)).
        for col in (j + 1)..3 {
            let s: f64 = (j..3).map(|i| v[i] * r[i][col]).sum();
            for i in j..3 {
                r[i][col] -= tau * v[i] * s;
            }
        }
        r[j][j] = beta;
        for i in (j + 1)..3 {
            r[i][j] = 0.0;
        }
        diag[j] = beta;
        // Q <- Q H.
        for row in q.iter_mut() {
            let s: f64 = (j..3).map(|k| row[k] * v[k]).sum();
            for k in j..3 {
                row[k] -= tau * s * v[k];
            }
        }
    }
    (q, diag)
}

/// The deterministic half of `_random_rotation`: QR of the 3x3 matrix `g` (the draw), the columns of Q
/// multiplied by `sign(diag(R))` (which makes the factorization unique, so any QR algorithm gives this Q:
/// Householder here), then column 0 negated when `det(Q) < 0`, so the result is a proper rotation. Row-major,
/// `q[row][col]`. Scale free in `g`; `np.sign(0) = 0` would zero a column (never happens for normal draws).
pub fn rotation_from_normals(g: &Mat3) -> Mat3 {
    let (mut q, diag) = householder_qr(g);
    for j in 0..3 {
        let s = np_sign(diag[j]);
        for row in q.iter_mut() {
            row[j] *= s;
        }
    }
    if det3(&q) < 0.0 {
        for row in q.iter_mut() {
            row[0] = -row[0];
        }
    }
    q
}

/// Port of `_random_rotation`: a rotation from a 3x3 matrix of standard normals drawn from `rng` in C order.
pub fn random_rotation(rng: &mut Rng) -> Mat3 {
    let mut g = [[0.0; 3]; 3];
    for row in g.iter_mut() {
        for v in row.iter_mut() {
            *v = rng.normal();
        }
    }
    rotation_from_normals(&g)
}

/// `rot.astype(np.float32)`.
pub fn rotation_to_f32(q: &Mat3) -> [[f32; 3]; 3] {
    let mut o = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = q[i][j] as f32;
        }
    }
    o
}

/// One row of `pts @ rot.T`: `sum_j p[j] rot[k][j]` in `f32`, summed in order (numpy runs a BLAS sgemm whose
/// summation order and fused multiply-add use are unspecified: the golden tolerances cover both).
fn rotate_f32(p: Vec3<f32>, rot: &[[f32; 3]; 3]) -> Vec3<f32> {
    let row = |r: &[f32; 3]| p[0] * r[0] + p[1] * r[1] + p[2] * r[2];
    [row(&rot[0]), row(&rot[1]), row(&rot[2])]
}

// ---------------------------------------------------------------------------------------------------------
// Worley

/// Port of `Worley3`: seeded 3D cellular noise. `F1` and `F2` are the distances to the nearest and second
/// nearest feature point, one jittered point per unit cell, in cell units. A 1024-entry permutation keeps the
/// lattice from repeating within a unit sphere scaled up to ~500 cells across.
#[derive(Clone, Debug, PartialEq)]
pub struct Worley3 {
    /// The DOUBLED permutation `concat(perm, perm)`, 2048 entries (the hash indexes up to 2046).
    pub perm: Vec<i64>,
    /// One feature point per hash value, in [0, 1)^3 cell coordinates, 1024 entries.
    pub jitter: Vec<Vec3<f32>>,
}

impl Worley3 {
    /// `Worley3.SIZE`.
    pub const SIZE: usize = 1024;

    /// `Worley3.__init__` with the tables given: the 1024-entry permutation and the 1024 jitter rows, as drawn.
    pub fn new(perm: &[i64], jitter: &[Vec3<f32>]) -> Worley3 {
        assert_eq!(perm.len(), Self::SIZE, "a Worley3 permutation has {} entries", Self::SIZE);
        assert_eq!(jitter.len(), Self::SIZE, "a Worley3 jitter table has {} rows", Self::SIZE);
        Worley3 { perm: perm.iter().chain(perm.iter()).copied().collect(), jitter: jitter.to_vec() }
    }

    /// `Worley3.__init__` with the tables drawn from `rng`, in Python's order: the permutation, then the
    /// `(1024, 3)` uniform jitter in C order rounded to `f32`.
    pub fn draw(rng: &mut Rng) -> Worley3 {
        let perm: Vec<i64> = rng.permutation(Self::SIZE).into_iter().map(|v| v as i64).collect();
        let jitter: Vec<Vec3<f32>> = (0..Self::SIZE)
            .map(|_| {
                let a = rng.uniform() as f32;
                let b = rng.uniform() as f32;
                let d = rng.uniform() as f32;
                [a, b, d]
            })
            .collect();
        Worley3::new(&perm, &jitter)
    }

    /// `Worley3(seed)`: the tables of the generator seeded with `seed`.
    pub fn from_seed(seed: u64) -> Worley3 {
        Worley3::draw(&mut Rng::new(seed))
    }

    /// `(F1, F2)` at one point. The cell index is `floor` in the dtype of the point and the fraction is rounded
    /// to `f32` only after the subtraction; the 27 neighbor cells are searched with `f32` squared distances and
    /// the square root is taken at the end. An exact tie gives `F2 == F1`. Cells wrap by `& 1023`.
    pub fn at<F: Float>(&self, p: Vec3<F>) -> (f32, f32) {
        let m = (Self::SIZE - 1) as i64;
        let cell = [p[0].floor(), p[1].floor(), p[2].floor()];
        let f: Vec3<f32> = [
            (p[0] - cell[0]).to_f32().unwrap_or(f32::NAN),
            (p[1] - cell[1]).to_f32().unwrap_or(f32::NAN),
            (p[2] - cell[2]).to_f32().unwrap_or(f32::NAN),
        ];
        let ci = [floor_to_i64(cell[0]), floor_to_i64(cell[1]), floor_to_i64(cell[2])];
        let perm = &self.perm;
        let (mut f1, mut f2) = (f32::INFINITY, f32::INFINITY);
        for dx in -1i64..=1 {
            let x = ci[0].wrapping_add(dx) & m;
            for dy in -1i64..=1 {
                let xy = perm[x as usize] + (ci[1].wrapping_add(dy) & m);
                for dz in -1i64..=1 {
                    let h = perm[(perm[xy as usize] + (ci[2].wrapping_add(dz) & m)) as usize];
                    let j = self.jitter[h as usize];
                    let ex = j[0] + dx as f32 - f[0];
                    let ey = j[1] + dy as f32 - f[1];
                    let ez = j[2] + dz as f32 - f[2];
                    let d2 = ex * ex + ey * ey + ez * ez;
                    f2 = if d2 < f1 { f1 } else { np_min(f2, d2) };
                    f1 = np_min(f1, d2);
                }
            }
        }
        (f1.sqrt(), f2.sqrt())
    }

    /// `Worley3.__call__`: `(F1, F2)` at every point.
    pub fn call<F: Float + Send + Sync>(&self, pts: &[Vec3<F>]) -> (Vec<f32>, Vec<f32>) {
        crate::par::map(pts, |&p| self.at(p)).into_iter().unzip()
    }
}

// ---------------------------------------------------------------------------------------------------------
// cell_edges

/// What `cell_edges` draws: the Worley tables and the rotation of the points, plus the frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct CellEdgesPlan {
    pub worley: Worley3,
    /// `_random_rotation(rng).astype(np.float32)`, `rot[row][col]`.
    pub rot: [[f32; 3]; 3],
    /// Cycles per unit length (the sphere's radius is 1), cast to `f32` at use. May be zero or negative.
    pub frequency: f64,
}

impl CellEdgesPlan {
    /// The draws of `cell_edges(points, frequency, seed)`, in Python's order: a Worley seed
    /// (`rng.integers(1 << 31)`), the tables of `Worley3(that seed)`, then the rotation from the same `rng`.
    pub fn draw(rng: &mut Rng, frequency: f64) -> CellEdgesPlan {
        let worley_seed = rng.below(1 << 31);
        let worley = Worley3::from_seed(worley_seed);
        let rot = rotation_to_f32(&random_rotation(rng));
        CellEdgesPlan { worley, rot, frequency }
    }

    /// The plan of `cell_edges(points, frequency, seed)`: `draw` on the generator seeded with `seed`.
    pub fn from_seed(seed: u64, frequency: f64) -> CellEdgesPlan {
        CellEdgesPlan::draw(&mut Rng::new(seed), frequency)
    }
}

/// Port of `cell_edges`: `F2 - F1` of Worley3 at `points @ rot^T * frequency`: 0 on a cell border, rising into
/// each cell's interior. The points are `f32` (Python casts them first); the matmul is done first and then the
/// scale by `float32(frequency)`; `F2 - F1` is an `f32` subtraction. Frequency 0 gives one constant for the
/// whole array and a negative frequency mirrors the points.
pub fn cell_edges(points: &[Vec3<f32>], plan: &CellEdgesPlan) -> Vec<f32> {
    let f = plan.frequency as f32;
    crate::par::map(points, |&p| {
        let r = rotate_f32(p, &plan.rot);
        let (a, b) = plan.worley.at([r[0] * f, r[1] * f, r[2] * f]);
        b - a
    })
}

/// `cell_edges(points, frequency, seed)` end to end: the plan drawn from `seed`, then the field.
pub fn cell_edges_seeded(points: &[Vec3<f32>], frequency: f64, seed: u64) -> Vec<f32> {
    cell_edges(points, &CellEdgesPlan::from_seed(seed, frequency))
}

// ---------------------------------------------------------------------------------------------------------
// Octave helpers

/// Port of `octaves_below`: how many octaves `frequency * lacunarity^k` (k = 0, 1, ...) start strictly below
/// `max_frequency` (at least 1): an octave past the map's Nyquist limit aliases into stair-steps instead of
/// adding detail. The power is libm's `pow` (as Python's `**`), not repeated multiplication: on an exact
/// boundary the two can round to different counts.
///
/// Python loops forever when `frequency < max_frequency` and `frequency <= 0` or `lacunarity <= 1`; this
/// panics instead.
pub fn octaves_below(frequency: f64, max_frequency: f64, lacunarity: f64) -> usize {
    assert!(
        !(frequency < max_frequency) || (frequency > 0.0 && lacunarity > 1.0),
        "octaves_below never terminates for frequency {frequency}, max {max_frequency}, lacunarity {lacunarity}"
    );
    let mut n = 0usize;
    while frequency * lacunarity.powf(n as f64) < max_frequency {
        n += 1;
    }
    n.max(1)
}

/// Port of `octave_ladder`: `(start frequency, octave count)` of a ladder anchored at the TOP: its last octave
/// sits at `top` x the map's limit, then steps down by the lacunarity until `low_frequency`. When
/// `hi = top * max_frequency` is not above `low_frequency` the result is `(hi, 1)`; otherwise
/// `n = floor(ln(hi / low) / ln(lacunarity)) + 1` and the start is `hi / lacunarity^(n - 1)`. A ratio that is
/// an exact power of the lacunarity lands on either side by the rounding of the logarithms (golden case
/// `octave_ladder.fragile`).
///
/// Python fails (division by zero, a log of a non-positive number) for `low_frequency <= 0` or
/// `lacunarity <= 1` when the ladder has more than one octave; this panics.
pub fn octave_ladder(low_frequency: f64, max_frequency: f64, lacunarity: f64, top: f64) -> (f64, usize) {
    let hi = top * max_frequency;
    if hi <= low_frequency {
        return (hi, 1);
    }
    assert!(low_frequency > 0.0 && lacunarity > 1.0, "octave_ladder needs low_frequency > 0 and lacunarity > 1");
    let n = ((hi / low_frequency).ln() / lacunarity.ln()).floor() as i64 + 1;
    (hi / lacunarity.powf((n - 1) as f64), n as usize)
}

/// Port of `map_max_frequency`: the highest noise frequency (cycles per radian of arc) a width-`W`
/// equirectangular map resolves with at least `min_texels` texels per wavelength at the equator:
/// `width / (2 pi min_texels)`.
pub fn map_max_frequency(width: f64, min_texels: f64) -> f64 {
    width / (2.0 * std::f64::consts::PI * min_texels)
}

// ---------------------------------------------------------------------------------------------------------
// fBm

/// One octave of an fBm: its amplitude and frequency (`float64`), and the rotation and offset of its lattice
/// (`float32`), so octaves do not share a lattice.
#[derive(Clone, Debug, PartialEq)]
pub struct FbmOctave {
    pub amp: f64,
    pub freq: f64,
    /// `rot[row][col]`: the point is multiplied by `rot^T`.
    pub rot: [[f32; 3]; 3],
    pub off: Vec3<f32>,
}

/// `(amp, freq)` of each octave in Python's order: `amp` starts at 1 and `freq` at `frequency`, then
/// `amp *= gain; freq *= lacunarity` per octave (repeated float64 multiplication, not a power).
pub fn octave_amp_freq(frequency: f64, octaves: usize, gain: f64, lacunarity: f64) -> Vec<(f64, f64)> {
    let (mut amp, mut freq) = (1.0, frequency);
    (0..octaves)
        .map(|_| {
            let af = (amp, freq);
            amp *= gain;
            freq *= lacunarity;
            af
        })
        .collect()
}

/// Everything `fbm` draws or derives: the Perlin permutation, the octaves, the billow flag and the
/// normalization `norm`, so the output has unit variance by construction ([`PERLIN3_STD`]) whatever points
/// (or chunk of a 16K map) are evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct FbmPlan {
    pub perlin: Perlin3,
    pub octaves: Vec<FbmOctave>,
    /// Billow octaves use `|noise| - PERLIN3_ABS_MEAN`: rounded puffs separated by sharp creases.
    pub billow: bool,
    /// `1 / (std1 sqrt(sum amp^2))` in `f64`, `std1` = [`PERLIN3_ABS_STD`] for billow else [`PERLIN3_STD`].
    pub norm: f64,
}

impl FbmPlan {
    /// Assembles a plan from drawn tables and computes `norm` as Python does. At least one octave.
    pub fn new(perlin: Perlin3, octaves: Vec<FbmOctave>, billow: bool) -> FbmPlan {
        assert!(!octaves.is_empty(), "fbm needs at least one octave");
        let std1 = if billow { PERLIN3_ABS_STD } else { PERLIN3_STD };
        let mut sum_sq = 0.0;
        for o in &octaves {
            sum_sq += o.amp * o.amp;
        }
        let norm = 1.0 / (std1 * sum_sq.sqrt());
        FbmPlan { perlin, octaves, billow, norm }
    }

    /// The draws of `fbm(points, frequency, octaves, seed, gain, lacunarity, billow)`, in Python's order: a
    /// Perlin seed (`rng.integers(1 << 31)`) whose `Perlin3` permutation comes from its own generator, then per
    /// octave a rotation (`_random_rotation`) and an offset (`uniform(0, 256, size=3)` as `f32`).
    pub fn draw(rng: &mut Rng, frequency: f64, octaves: usize, gain: f64, lacunarity: f64, billow: bool) -> FbmPlan {
        assert!(octaves >= 1, "fbm needs at least one octave");
        let perlin = Perlin3::from_seed(rng.below(1 << 31));
        let octs = octave_amp_freq(frequency, octaves, gain, lacunarity)
            .into_iter()
            .map(|(amp, freq)| {
                let rot = rotation_to_f32(&random_rotation(rng));
                let off = [
                    rng.uniform_range(0.0, 256.0) as f32,
                    rng.uniform_range(0.0, 256.0) as f32,
                    rng.uniform_range(0.0, 256.0) as f32,
                ];
                FbmOctave { amp, freq, rot, off }
            })
            .collect();
        FbmPlan::new(perlin, octs, billow)
    }

    /// The plan of `fbm(...)`: `draw` on the generator seeded with `seed`.
    pub fn from_seed(seed: u64, frequency: f64, octaves: usize, gain: f64, lacunarity: f64, billow: bool) -> FbmPlan {
        FbmPlan::draw(&mut Rng::new(seed), frequency, octaves, gain, lacunarity, billow)
    }
}

/// Port of `fbm`: fBm of Perlin noise at 3D points; frequency in cycles per unit length (the sphere's radius
/// is 1). Per point and per octave in order, all in `f32`: `q = (p @ rot^T) * float32(freq) + off`,
/// `v = perlin(q)`, for billow `v = |v| - float32(PERLIN3_ABS_MEAN)`, `total += float32(amp) * v`; the result
/// is `total * float32(norm)`. The points are `f32` (Python casts them first).
pub fn fbm(points: &[Vec3<f32>], plan: &FbmPlan) -> Vec<f32> {
    let mean1 = if plan.billow { PERLIN3_ABS_MEAN as f32 } else { 0.0 };
    let norm = plan.norm as f32;
    crate::par::map(points, |&p| {
        let mut total = 0.0f32;
        for o in &plan.octaves {
            let f = o.freq as f32;
            let r = rotate_f32(p, &o.rot);
            let q = [r[0] * f + o.off[0], r[1] * f + o.off[1], r[2] * f + o.off[2]];
            let mut v = plan.perlin.at(q[0], q[1], q[2]);
            if plan.billow {
                v = v.abs() - mean1;
            }
            total += o.amp as f32 * v;
        }
        total * norm
    })
}

/// `fbm(points, frequency, octaves, seed, gain, lacunarity, billow)` end to end: the plan drawn from `seed`,
/// then the field.
pub fn fbm_seeded(
    points: &[Vec3<f32>],
    frequency: f64,
    octaves: usize,
    seed: u64,
    gain: f64,
    lacunarity: f64,
    billow: bool,
) -> Vec<f32> {
    fbm(points, &FbmPlan::from_seed(seed, frequency, octaves, gain, lacunarity, billow))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_permutation<T: Copy + Into<i64>>(p: &[T], n: usize) -> bool {
        let mut seen = vec![false; n];
        p.iter().all(|&v| {
            let v: i64 = v.into();
            (0..n as i64).contains(&v) && !std::mem::replace(&mut seen[v as usize], true)
        })
    }

    fn matmul_t(a: &Mat3) -> Mat3 {
        // a a^T
        let mut o = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                o[i][j] = (0..3).map(|k| a[i][k] * a[j][k]).sum();
            }
        }
        o
    }

    fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
        let mut o = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                o[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
            }
        }
        o
    }

    fn sphere_points(rng: &mut Rng, n: usize) -> Vec<Vec3<f32>> {
        (0..n)
            .map(|_| {
                let (x, y, z) = (rng.normal(), rng.normal(), rng.normal());
                let r = (x * x + y * y + z * z).sqrt();
                [(x / r) as f32, (y / r) as f32, (z / r) as f32]
            })
            .collect()
    }

    fn mean_std(v: &[f32]) -> (f64, f64) {
        let n = v.len() as f64;
        let m = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let var = v.iter().map(|&x| (f64::from(x) - m).powi(2)).sum::<f64>() / n;
        (m, var.sqrt())
    }

    #[test]
    fn fade_endpoints_and_grad_table() {
        assert_eq!(fade(0.0f64), 0.0);
        assert_eq!(fade(1.0f64), 1.0);
        assert_eq!(fade(0.5f64), 0.5);
        assert_eq!(fade(2.0f64), 8.0 * (2.0 * (2.0 * 6.0 - 15.0) + 10.0)); // not clamped: 2^3 (24 - 30 + 10) = 32
        assert_eq!(fade(2.0f32), 32.0);
        // The 16 gradients at (x, y, z) = (1, 10, 100): u is x below 8 else y; v is y below 4, x for 12 and 14,
        // else z; bit 0 negates u and bit 1 negates v.
        let (ub, vb): ([f64; 16], [f64; 16]) = (
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1.0, 100.0, 1.0, 100.0],
        );
        for h in 0..16usize {
            let u = if h & 1 != 0 { -ub[h] } else { ub[h] };
            let v = if h & 2 != 0 { -vb[h] } else { vb[h] };
            let got = grad(h as i32, 1.0f64, 10.0, 100.0);
            assert_eq!(got, u + v, "h = {h}");
            assert_eq!(grad(h as i32 + 16, 1.0f64, 10.0, 100.0), got, "h & 15 for h = {}", h + 16);
            assert_eq!(grad(h as i32 + 240, 1.0f64, 10.0, 100.0), got, "h & 15 for h = {}", h + 240);
        }
    }

    #[test]
    fn perlin_table_is_doubled_and_vanishes_on_the_lattice() {
        let p = Perlin3::from_seed(3);
        assert_eq!(p.perm.len(), 512);
        assert_eq!(&p.perm[..256], &p.perm[256..]);
        assert!(is_permutation(&p.perm[..256], 256));
        for &(x, y, z) in &[(0.0f32, 0.0, 0.0), (3.0, -7.0, 12.0), (-256.0, 300.0, 5.0)] {
            assert_eq!(p.at(x, y, z), 0.0);
        }
        // Period 256 on each axis, in both signs; floor, not truncation, for negative coordinates.
        for &(x, y, z) in &[(0.25f64, 0.5, 0.75), (-3.375, 8.125, -0.40625), (100.875, -100.9375, 55.5)] {
            let v = p.at(x, y, z);
            assert_eq!(p.at(x + 256.0, y - 256.0, z + 512.0), v);
            assert!(v.abs() < 1.0);
        }
        assert!(p.at(-0.25f64, 0.5, 0.5) != p.at(0.25f64, 0.5, 0.5));
    }

    #[test]
    fn perlin_draw_depends_on_the_seed_only() {
        assert_eq!(Perlin3::from_seed(5), Perlin3::from_seed(5));
        assert_ne!(Perlin3::from_seed(5), Perlin3::from_seed(6));
        let mut rng = Rng::new(9);
        let a = Perlin3::draw(&mut rng);
        let b = Perlin3::draw(&mut rng);
        assert_ne!(a, b, "two draws from one generator differ");
    }

    #[test]
    fn perlin_statistics_match_the_pinned_std() {
        // PERLIN3_STD is the measured std of one octave over random points: the draw must reproduce it.
        let p = Perlin3::from_seed(11);
        let mut rng = Rng::new(12);
        let v: Vec<f32> = (0..100_000)
            .map(|_| p.at(rng.uniform_range(-50.0, 50.0) as f32, rng.uniform_range(-50.0, 50.0) as f32, rng.uniform_range(-50.0, 50.0) as f32))
            .collect();
        let (m, s) = mean_std(&v);
        assert!(m.abs() < 0.005, "{m}");
        assert!((s - PERLIN3_STD).abs() < 0.006, "{s}");
        let abs: Vec<f32> = v.iter().map(|x| x.abs()).collect();
        let (am, asd) = mean_std(&abs);
        assert!((am - PERLIN3_ABS_MEAN).abs() < 0.006, "{am}");
        assert!((asd - PERLIN3_ABS_STD).abs() < 0.006, "{asd}");
    }

    #[test]
    fn nan_propagates_unguarded() {
        let p = Perlin3::from_seed(1);
        assert!(p.at(f32::NAN, 0.5, 0.5).is_nan());
        let w = Worley3::from_seed(1);
        let (a, b) = w.at([f32::NAN, 0.5, 0.5]);
        assert!(a.is_nan() && b.is_nan());
        assert!(np_min(1.0, f32::NAN).is_nan() && np_min(f32::NAN, 1.0).is_nan());
    }

    #[test]
    fn worley_tables_and_ranges() {
        let w = Worley3::from_seed(21);
        assert_eq!(w.perm.len(), 2048);
        assert_eq!(&w.perm[..1024], &w.perm[1024..]);
        assert!(is_permutation(&w.perm[..1024], 1024));
        assert_eq!(w.jitter.len(), 1024);
        assert!(w.jitter.iter().flatten().all(|&j| (0.0..=1.0).contains(&j)));
        assert_eq!(Worley3::from_seed(21), w);
        assert_ne!(Worley3::from_seed(22), w);
        // F1 <= F2, both bounded, continuous, and periodic with period 1024 cells.
        let mut rng = Rng::new(4);
        for _ in 0..2000 {
            let p = [rng.uniform_range(-6.0, 6.0), rng.uniform_range(-6.0, 6.0), rng.uniform_range(-6.0, 6.0)];
            let (f1, f2) = w.at(p);
            assert!(0.0 <= f1 && f1 <= f2 && f2 < 1.8, "{f1} {f2}");
            let q = [p[0] + 1024.0, p[1] - 2048.0, p[2]];
            assert_eq!(w.at(q), (f1, f2));
        }
    }

    #[test]
    fn worley_ties_on_a_centered_lattice() {
        // Identity permutation, every jitter 0.5: features at the cell centers.
        let perm: Vec<i64> = (0..1024).collect();
        let w = Worley3::new(&perm, &vec![[0.5; 3]; 1024]);
        let near = |a: f32, b: f32| (a - b).abs() < 1e-6;
        let (f1, f2) = w.at([0.5f32, 0.5, 0.5]);
        assert!(near(f1, 0.0) && near(f2, 1.0));
        let (f1, f2) = w.at([1.0f32, 0.5, 0.5]); // on a face: two features at 0.5
        assert!(near(f1, 0.5) && f2 == f1);
        let (f1, f2) = w.at([1.0f32, 1.0, 1.0]); // at a corner: eight features at sqrt(3)/2
        assert!(near(f1, 0.75f32.sqrt()) && f2 == f1);
    }

    #[test]
    fn worley_floors_in_the_precision_of_the_points() {
        // A float64 point far from the origin keeps its fraction: rounding it to f32 first would not.
        let w = Worley3::from_seed(8);
        let p = [1500.0 + 1.0 / 3.0, -1700.25 + 1e-5, 321.123456789];
        let exact = w.at(p);
        let coarse = w.at([f64::from(p[0] as f32), f64::from(p[1] as f32), f64::from(p[2] as f32)]);
        assert!((exact.0 - coarse.0).abs() > 1e-6 || (exact.1 - coarse.1).abs() > 1e-6);
    }

    #[test]
    fn rotation_from_normals_is_the_unique_positive_diagonal_factor() {
        // q0 a proper rotation, r0 upper triangular with a positive diagonal: g = q0 r0 gives back q0.
        let (s, c) = (0.5f64.sin(), 0.5f64.cos());
        let (s2, c2) = (1.1f64.sin(), 1.1f64.cos());
        let rz: Mat3 = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let rx: Mat3 = [[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, s2, c2]];
        let q0 = mul(&rz, &rx);
        let r0: Mat3 = [[2.0, -1.5, 0.3], [0.0, 0.7, 4.0], [0.0, 0.0, 1.9]];
        let g = mul(&q0, &r0);
        let q = rotation_from_normals(&g);
        for i in 0..3 {
            for j in 0..3 {
                assert!((q[i][j] - q0[i][j]).abs() < 1e-12, "{q:?} vs {q0:?}");
            }
        }
        // Scale free.
        let g2: Mat3 = g.map(|r| r.map(|v| v * 1e-3));
        let q2 = rotation_from_normals(&g2);
        for i in 0..3 {
            for j in 0..3 {
                assert!((q2[i][j] - q0[i][j]).abs() < 1e-12);
            }
        }
        // A negative diagonal entry of R (here the second) is absorbed by the sign fix: the unique factor with a
        // positive diagonal is q0 diag(1, -1, 1), a reflection, so column 0 is then negated: columns
        // (-c0, -c1, c2) of q0.
        let r1: Mat3 = [[2.0, -1.5, 0.3], [0.0, -0.7, 4.0], [0.0, 0.0, 1.9]];
        let q = rotation_from_normals(&mul(&q0, &r1));
        for i in 0..3 {
            let want = [-q0[i][0], -q0[i][1], q0[i][2]];
            for j in 0..3 {
                assert!((q[i][j] - want[j]).abs() < 1e-12, "{q:?} vs {q0:?}");
            }
        }
        // A reflection input: diag(1, 1, -1) has Q = I with the third column negated by the sign fix... then
        // det = -1, so column 0 is negated: diag(-1, 1, -1).
        let q = rotation_from_normals(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
        assert_eq!(q, [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
    }

    #[test]
    fn drawn_rotations_are_proper_and_depend_on_the_generator() {
        let mut rng = Rng::new(31);
        let mut flipped = 0;
        for _ in 0..500 {
            let q = random_rotation(&mut rng);
            let gram = matmul_t(&q);
            for i in 0..3 {
                for j in 0..3 {
                    assert!((gram[i][j] - f64::from(i == j)).abs() < 1e-12);
                }
            }
            assert!((det3(&q) - 1.0).abs() < 1e-12);
            flipped += usize::from(q[0][0] < 0.0);
        }
        assert!(flipped > 100 && flipped < 400, "column 0 of Haar rotations is not biased in sign: {flipped}");
        assert_ne!(random_rotation(&mut Rng::new(1)), random_rotation(&mut Rng::new(2)));
    }

    #[test]
    fn octave_helpers() {
        assert_eq!(octaves_below(3.0, 24.0, 2.0), 3); // 3 * 2^3 = 24 is not below 24
        assert_eq!(octaves_below(3.0, 24.000001, 2.0), 4);
        assert_eq!(octaves_below(10.0, 5.0, 2.0), 1); // frequency above the limit: at least 1
        assert_eq!(octaves_below(1.0, 1.0 + 1e-12, DEFAULT_LACUNARITY), 1);
        let (start, n) = octave_ladder(48.0, 100.0, 2.0, 1.0);
        assert_eq!(n, 2);
        assert_eq!(start, 50.0); // hi = 100, steps down by 2 once
        assert_eq!(octave_ladder(10.0, 5.0, 2.03, 0.95), (4.75, 1));
        let (s, n) = octave_ladder(0.01, 1e4, DEFAULT_LACUNARITY, DEFAULT_TOP);
        assert_eq!(n, 20);
        assert!((s * DEFAULT_LACUNARITY.powi(19) - 0.95e4).abs() < 1e-6);
        assert_eq!(map_max_frequency(0.0, DEFAULT_MIN_TEXELS), 0.0);
        assert!((map_max_frequency(4096.0, DEFAULT_MIN_TEXELS) - 4096.0 / (8.0 * std::f64::consts::PI)).abs() < 1e-12);
        assert!(map_max_frequency(100.0, -2.0) < 0.0);
    }

    #[test]
    #[should_panic(expected = "never terminates")]
    fn octaves_below_rejects_what_loops_forever() {
        octaves_below(1.0, 10.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "never terminates")]
    fn octaves_below_rejects_a_zero_frequency() {
        octaves_below(0.0, 10.0, 2.0);
    }

    #[test]
    #[should_panic(expected = "lacunarity > 1")]
    fn octave_ladder_rejects_a_flat_lacunarity() {
        octave_ladder(1.0, 100.0, 1.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "at least one octave")]
    fn fbm_needs_an_octave() {
        FbmPlan::from_seed(1, 3.0, 0, 0.5, 2.03, false);
    }

    #[test]
    fn fbm_plan_schedule_and_norm() {
        let s = octave_amp_freq(3.0, 4, 0.5, 2.0);
        assert_eq!(s, vec![(1.0, 3.0), (0.5, 6.0), (0.25, 12.0), (0.125, 24.0)]);
        let plan = FbmPlan::from_seed(7, 3.0, 4, 0.5, 2.0, false);
        let want = 1.0 / (PERLIN3_STD * (1.0f64 + 0.25 + 0.0625 + 0.015625).sqrt());
        assert!((plan.norm - want).abs() < 1e-12);
        assert_eq!(plan.octaves.len(), 4);
        for o in &plan.octaves {
            let gram = matmul_t(&o.rot.map(|r| r.map(f64::from)));
            for i in 0..3 {
                for j in 0..3 {
                    assert!((gram[i][j] - f64::from(i == j)).abs() < 1e-6);
                }
            }
            assert!(o.off.iter().all(|&v| (0.0..=256.0).contains(&v)));
        }
        let billow = FbmPlan::from_seed(7, 3.0, 4, 0.5, 2.0, true);
        assert!((billow.norm - want * PERLIN3_STD / PERLIN3_ABS_STD).abs() < 1e-12);
        assert_eq!(billow.perlin, plan.perlin, "the same seed draws the same permutation, billow or not");
        assert_eq!(FbmPlan::from_seed(7, 3.0, 4, 0.5, 2.0, false), plan);
    }

    #[test]
    fn fbm_drawn_with_a_seed_has_unit_variance() {
        // The slow golden cases fbm.stat.* check this over 16 seeds and the exact bands; a quick version here.
        let pts = sphere_points(&mut Rng::new(100), 60_000);
        for &(oct, billow) in &[(1usize, false), (3, false), (3, true)] {
            let v = fbm_seeded(&pts, 40.0, oct, 7000, 0.5, DEFAULT_LACUNARITY, billow);
            let (m, s) = mean_std(&v);
            assert!((s - 1.0).abs() < 0.06, "{oct} octaves, billow {billow}: std {s}");
            assert!(m.abs() < if billow { 0.12 } else { 0.05 }, "{oct} octaves, billow {billow}: mean {m}");
        }
    }

    #[test]
    fn fbm_and_cell_edges_are_deterministic_and_seed_dependent() {
        let pts = sphere_points(&mut Rng::new(5), 300);
        let a = fbm_seeded(&pts, 6.0, 3, 1, 0.5, 2.03, false);
        assert_eq!(a, fbm_seeded(&pts, 6.0, 3, 1, 0.5, 2.03, false));
        assert_ne!(a, fbm_seeded(&pts, 6.0, 3, 2, 0.5, 2.03, false));
        let e = cell_edges_seeded(&pts, 8.0, 1);
        assert_eq!(e, cell_edges_seeded(&pts, 8.0, 1));
        assert_ne!(e, cell_edges_seeded(&pts, 8.0, 2));
        // The cell-edge plan is drawn from its own Worley seed, then the rotation.
        let plan = CellEdgesPlan::from_seed(1, 8.0);
        assert_eq!(plan.frequency, 8.0);
        assert_eq!(cell_edges(&pts, &plan), e);
    }

    #[test]
    fn drawn_offsets_cover_the_whole_lattice_period() {
        // Offsets are uniform in [0, 256) on every axis (the Perlin lattice period), 24 octaves per plan.
        let mut off = Vec::new();
        for seed in 0..20 {
            for o in &FbmPlan::from_seed(seed, 3.0, 24, 0.5, 2.03, false).octaves {
                off.extend_from_slice(&o.off);
            }
        }
        let axis = |k: usize| -> Vec<f32> { off.iter().skip(k).step_by(3).copied().collect() };
        for k in 0..3 {
            let a = axis(k);
            let (m, s) = mean_std(&a);
            assert!(a.iter().all(|&v| (0.0..256.0).contains(&v)));
            assert!(a.iter().copied().fold(0.0, f32::max) > 250.0 && a.iter().copied().fold(256.0, f32::min) < 6.0);
            assert!((m - 128.0).abs() < 8.0 && (s - 256.0 / 12f64.sqrt()).abs() < 5.0, "axis {k}: {m} {s}");
        }
    }

    #[test]
    fn empty_inputs_give_empty_outputs() {
        let plan = FbmPlan::from_seed(1, 3.0, 2, 0.5, 2.03, false);
        assert!(fbm(&[], &plan).is_empty());
        assert!(cell_edges(&[], &CellEdgesPlan::from_seed(1, 4.0)).is_empty());
        let (a, b) = Worley3::from_seed(1).call::<f32>(&[]);
        assert!(a.is_empty() && b.is_empty());
        assert!(Perlin3::from_seed(1).call::<f32>(&[], &[], &[]).is_empty());
    }

    #[test]
    fn cell_edges_field_statistics() {
        // F2 - F1 of a jittered lattice: non-negative, 0 on borders. Over 20000 unit-sphere points at frequency 8
        // the Python generator gives, over 64 seeds, a mean of 0.1966 (sd over seeds 0.0028) and a std of 0.1583
        // (sd 0.0023): the port's own draws must land within 4 sd of that.
        let pts = sphere_points(&mut Rng::new(77), 20_000);
        for seed in 0..16 {
            let e = cell_edges_seeded(&pts, 8.0, 5000 + seed);
            assert!(e.iter().all(|&v| v >= 0.0));
            let (m, s) = mean_std(&e);
            assert!((m - 0.1966).abs() < 4.0 * 0.0028, "seed {seed}: mean {m}");
            assert!((s - 0.1583).abs() < 4.0 * 0.0023, "seed {seed}: std {s}");
        }
        // Frequency 0 collapses to one constant.
        let e = cell_edges_seeded(&pts, 0.0, 3);
        assert!(e.iter().all(|&v| v == e[0]));
    }
}
