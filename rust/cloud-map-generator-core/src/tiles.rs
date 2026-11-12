//! Port of `cloudgen/tiles.py`: the cell-field tiles of the cloud deck (P4, product B). Four regimes of
//! sub-map cloud structure in one tileable RGBA tile (R heaps, G closed cells, B open cells, A sheet), each
//! channel written as a rank byte, baked at dev time.
//!
//! Everything here is float64 (as in Python) except the rank encoding, which is generic over the dtype of the
//! field it ranks. Fields are square `size x size`, row-major, row 0 first (the tile has no north); point sets
//! are `[x, y]` pairs in tile units with x the column.
//!
//! Random draws are inputs of the kernels (`../README.md` rule 3). Python's draws are recorded in the golden
//! cases as `plan<i>.<field>` arrays; here each kernel takes them as a plan struct ([`WorleyPlan`],
//! [`FbmPlan`], [`PoissonPlan`], [`LumpsPlan`], and the composites [`WarpedVoronoiPlan`] and
//! [`RawChannelsPlan`]), and each plan has a `draw` constructor that draws it with [`crate::rng::Rng`]. A tile
//! built from such a plan is statistically equivalent to Python's for the same seed, not identical.
//!
//! Python functions that are private there (`_smoothstep`, `_inverted`, `_uniform`, `_sample`,
//! `_texel_centers`) lose their underscore here, so that the golden tests can reach them.
//!
//! Third-party numerics replaced by local code (the only dependencies are `num-traits` and `rustfft`):
//! * `scipy.special.ndtr` -> [`ndtr`], [`erf`] and [`erfc`]: the Cephes routines scipy runs (rational
//!   approximations, same branches and same operation order), ported literally;
//! * `scipy.spatial.cKDTree(points, boxsize=1.0).query(q, k)` -> [`PeriodicPoints`], an exact periodic
//!   k-nearest search on a uniform grid;
//! * `np.fft.fft2` / `ifft2` -> `rustfft`, same conventions (unnormalized forward, `1 / N^2` on the inverse).
//!
//! numpy details that are part of the semantics although the golden tolerances cannot see them, each pinned
//! to the bit by a unit test against values numpy or scipy produced: `array ** scalar` takes the reciprocal,
//! square root and square fast paths (`np_power`, behind the spectrum of `periodic_fbm`); `sum` is pairwise
//! and added to the identity 0.0, not to the first element (`pairwise_sum`, behind the mean and std of
//! `periodic_fbm`); `fftfreq(n) * n` is not an exact integer for every `n` (`fft_k`); Cephes `erf`, `erfc`
//! and `ndtr` are scipy's own, not a more accurate function; the float32 smoothstep forms `b - a` in float64
//! before rounding it.

use crate::rng::Rng;
use num_traits::Float;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

// ---------------------------------------------------------------------------------------------------------
// Constants (cloudgen/tiles.py)

/// Side of the baked tile, texels.
pub const TILE_SIZE: usize = 1024;
/// Earth's reference tile, km a side.
pub const REFERENCE_TILE_KM: f64 = 240.0;
/// Envelope frequency of the heaps: clusters a tile (cloud_cell_bake.ts CLUSTERS).
pub const HEAP_CLUSTERS: usize = 8;
/// Envelope frequency of the heaps: cells a tile (CELLS).
pub const HEAP_CELLS: usize = 32;
/// Puffs a tile (CLOUD_CELL_PUFFS).
pub const PUFFS: usize = 96;
/// Weight of the puffs where the envelope is 0.
pub const HEAP_BASE: f64 = 0.2;
/// Weight of the envelope itself in the heaps.
pub const ENVELOPE_FLOOR: f64 = 0.2;
/// Closed-cell diameters on Earth's reference tile, km (log-uniform between the bounds).
pub const CLOSED_CELL_KM: (f64, f64) = (8.0, 32.0);
/// Open-cell diameters, km.
pub const OPEN_CELL_KM: (f64, f64) = (12.0, 48.0);
/// Cumulus lumps of the open-cell walls: spacing, km.
pub const LUMP_KM: f64 = 4.0;
/// Lump radius as a fraction of the spacing.
pub const LUMP_RADIUS: (f64, f64) = (0.35, 0.7);
/// Border warp, in units of the local cell size.
pub const BORDER_WARP: f64 = 0.12;
/// Sheet fBm: lowest frequency, cycles a tile.
pub const SHEET_LOW: f64 = 2.0;
/// Sheet fBm: power-spectrum slope.
pub const SHEET_SLOPE: f64 = 3.0;
/// The four channels in tile order.
pub const CHANNELS: [&str; 4] = ["heaps", "closed", "open", "sheet"];
/// Candidates a Poisson dart throwing draws per unit of `1 / s_min^2` (the literal 12.0 of `poisson_points`).
pub const POISSON_OVERSAMPLE: f64 = 12.0;
/// Default `spacing` of `poisson_points`.
pub const POISSON_SPACING: f64 = 0.85;
/// Lowest frequency of the `_uniform` field `cell_size_field` maps to a size (the literal 1.0).
pub const CELL_SIZE_LOW: f64 = 1.0;
/// Power-spectrum slope of that field (the literal 3.0).
pub const CELL_SIZE_SLOPE: f64 = 3.0;
/// Lowest frequency (cycles a tile) of the two warp fBms of `warped_voronoi` (the literal 3.0).
pub const WARP_LOW: f64 = 3.0;
/// Power-spectrum slope of the two warp fBms of `warped_voronoi` (the literal 3.0).
pub const WARP_SLOPE: f64 = 3.0;
/// Lumps `lumps` takes the max over: the `k` of its KD-tree query (the literal 4).
pub const LUMP_NEAREST: usize = 4;
/// Exponent of a lump's dome (the literal 0.6).
pub const LUMP_EXPONENT: f64 = 0.6;
/// Default coverages of `coverage_masks`: what the deck draws of a pure regime.
pub const DEFAULT_COVERAGES: [f64; 3] = [0.3, 0.5, 0.7];

// ---------------------------------------------------------------------------------------------------------
// numpy helpers

/// `np.maximum` for two scalars: NaN propagates (`f64::max` would drop it).
fn np_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `np.clip(v, lo, hi)`: NaN propagates, infinities go to the bounds.
fn np_clip<F: Float>(v: F, lo: F, hi: F) -> F {
    if v.is_nan() {
        v
    } else {
        v.max(lo).min(hi)
    }
}

/// `x ** e` for a float64 array `x` and a Python float `e`, the way numpy runs it: `array ** scalar` does not
/// call `pow` for the exponents `1` (a copy), `-1` (`np.reciprocal`, `1 / x`), `0` (ones), `0.5` (`np.sqrt`)
/// and `2` (`np.square`, `x * x`); every other exponent is the libm `pow`. The results differ from `pow` by an
/// ulp now and then (`pow(x, -1.0)` is not always `1 / x`), so the fast paths are part of the semantics.
fn np_power(x: f64, e: f64) -> f64 {
    if e == 1.0 {
        x
    } else if e == -1.0 {
        1.0 / x
    } else if e == 0.0 {
        1.0
    } else if e == 0.5 {
        x.sqrt()
    } else if e == 2.0 {
        x * x
    } else {
        x.powf(e)
    }
}

/// `a.sum()` of a contiguous float64 array, numpy's pairwise summation: blocks of 8 accumulators up to 128
/// elements, halves above (split on a multiple of 8), added to the identity 0.0 of the reduction (NOT to the
/// first element: that order differs by an ulp or more, and `sum([-0.0])` is 0.0).
fn pairwise_sum(a: &[f64]) -> f64 {
    fn pw(a: &[f64]) -> f64 {
        let n = a.len();
        if n < 8 {
            let mut res = 0.0;
            for &v in a {
                res += v;
            }
            res
        } else if n <= 128 {
            let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
            let body = n - n % 8;
            let mut i = 8;
            while i < body {
                for j in 0..8 {
                    r[j] += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for &v in &a[body..] {
                res += v;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pw(&a[..n2]) + pw(&a[n2..])
        }
    }
    let identity = 0.0;
    identity + pw(a)
}

/// `field.mean()` and the population `field.std()` (ddof = 0), numpy's own operation order.
fn mean_std(a: &[f64]) -> (f64, f64) {
    let n = a.len() as f64;
    let mean = pairwise_sum(a) / n;
    let sq: Vec<f64> = a.iter().map(|&v| (v - mean) * (v - mean)).collect();
    (mean, (pairwise_sum(&sq) / n).sqrt())
}

// ---------------------------------------------------------------------------------------------------------
// Smoothstep, inverted distance, Worley

/// `_smoothstep(a, b, x)`: `k^2 (3 - 2k)` with `k = clip((x - a) / (b - a), 0, 1)`, in the dtype of `x` (the
/// Python float edges stay weak, so `b - a` is formed in float64 and then rounded to `F`). `a == b` divides by
/// zero (0, NaN, 1 for a value below, at, above); NaN propagates.
pub fn smoothstep<F: Float>(a: f64, b: f64, x: &[F]) -> Vec<F> {
    let conv = |v: f64| F::from(v).expect("float conversion");
    let (af, den) = (conv(a), conv(b - a));
    let (zero, one, two, three) = (conv(0.0), conv(1.0), conv(2.0), conv(3.0));
    x.iter()
        .map(|&v| {
            let k = np_clip((v - af) / den, zero, one);
            k * k * (three - two * k)
        })
        .collect()
}

/// `_inverted(f1)`: `max(0, 1 - f1)`, 1 at a feature point and 0 a full cell away (cloud_cell_bake.ts
/// tileableWorley2). No upper clamp: a negative distance gives more than 1.
pub fn inverted(f1: &[f64]) -> Vec<f64> {
    f1.iter().map(|&v| np_max(0.0, 1.0 - v)).collect()
}

/// The draw of `tileable_worley`: the jitter of a `freq x freq` lattice, `[row][col][x, y]` in cell units
/// (`rng.random((freq, freq, 2))`, plan kind `tile.worley`).
#[derive(Clone, Debug, PartialEq)]
pub struct WorleyPlan {
    pub freq: usize,
    /// `freq * freq * 2` values in [0, 1).
    pub jitter: Vec<f64>,
}

impl WorleyPlan {
    /// Draws the jitter with `rng` (row-major, x then y per cell).
    pub fn draw(freq: usize, rng: &mut Rng) -> WorleyPlan {
        WorleyPlan { freq, jitter: (0..freq * freq * 2).map(|_| rng.uniform()).collect() }
    }
}

/// `tileable_worley(size, freq, seed)`: F1 and F2 distances, in cell units, to a jittered grid of
/// `freq x freq` points that wraps on the tile, at texel centers. The jitter is looked up at the wrapped cell
/// and the distance taken to the unwrapped one, so the field repeats exactly with the tile. Returns `(f1, f2)`,
/// `size * size` each.
pub fn tileable_worley(size: usize, plan: &WorleyPlan) -> (Vec<f64>, Vec<f64>) {
    let freq = plan.freq;
    assert!(freq > 0, "tileable_worley: freq must be positive");
    assert_eq!(plan.jitter.len(), freq * freq * 2, "tileable_worley: the jitter must be freq x freq x 2");
    let c: Vec<f64> = (0..size).map(|i| ((i as f64 + 0.5) / size as f64) * freq as f64).collect();
    let ic: Vec<i64> = c.iter().map(|v| v.floor() as i64).collect();
    let fq = freq as i64;
    let mut f1 = vec![f64::INFINITY; size * size];
    let mut f2 = vec![f64::INFINITY; size * size];
    for dy in -1_i64..=1 {
        for dx in -1_i64..=1 {
            for row in 0..size {
                let y = c[row];
                let ny = ic[row] + dy;
                let jr = ny.rem_euclid(fq) as usize;
                for col in 0..size {
                    let x = c[col];
                    let nx = ic[col] + dx;
                    let jc = nx.rem_euclid(fq) as usize;
                    let j = (jr * freq + jc) * 2;
                    let d = ((nx as f64 + plan.jitter[j]) - x).hypot((ny as f64 + plan.jitter[j + 1]) - y);
                    let i = row * size + col;
                    f2[i] = if d < f1[i] { f1[i] } else { f2[i].min(d) };
                    f1[i] = f1[i].min(d);
                }
            }
        }
    }
    (f1, f2)
}

// ---------------------------------------------------------------------------------------------------------
// Spectral synthesis, the normal CDF

/// The draw of `periodic_fbm`: a `size x size` field of standard normals (`rng.standard_normal((size, size))`,
/// plan kind `tile.fbm`).
#[derive(Clone, Debug, PartialEq)]
pub struct FbmPlan {
    pub white: Vec<f64>,
}

impl FbmPlan {
    /// Draws the white noise with `rng` (row-major).
    pub fn draw(size: usize, rng: &mut Rng) -> FbmPlan {
        FbmPlan { white: (0..size * size).map(|_| rng.normal()).collect() }
    }
}

/// `np.fft.fftfreq(n) * n`, element by element: `(i * (1 / n)) * n` for the first `(n - 1) / 2 + 1` indices
/// and `((i - n) * (1 / n)) * n` for the rest. NOT an exact integer for a size that is not a power of two
/// (n = 14: k = 5 is 4.999999999999999), which changes which modes pass `kk >= low`; kept as numpy has it.
fn fft_k(n: usize) -> Vec<f64> {
    let val = 1.0 / (n as f64);
    let positive = (n - 1) / 2 + 1;
    (0..n)
        .map(|i| {
            let signed = if i < positive { i as f64 } else { i as f64 - n as f64 };
            (signed * val) * n as f64
        })
        .collect()
}

fn transpose(src: &[Complex<f64>], n: usize) -> Vec<Complex<f64>> {
    let mut out = vec![Complex::new(0.0, 0.0); n * n];
    for r in 0..n {
        for c in 0..n {
            out[c * n + r] = src[r * n + c];
        }
    }
    out
}

/// `real(ifft2(fft2(white) * amp))`: unnormalized forward transform, `1 / N^2` on the inverse, `amp` indexed
/// `[row][col]` like the field.
fn filter_fft2(white: &[f64], n: usize, amp: &[f64]) -> Vec<f64> {
    let mut planner = FftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let mut buf: Vec<Complex<f64>> = white.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fwd.process(&mut buf); // along the columns index, one row at a time
    let mut t = transpose(&buf, n); // t[col][row]
    fwd.process(&mut t); // along the rows index
    for c in 0..n {
        for r in 0..n {
            t[c * n + r] *= amp[r * n + c];
        }
    }
    inv.process(&mut t);
    let mut b = transpose(&t, n);
    inv.process(&mut b);
    let scale = (n * n) as f64;
    b.iter().map(|z| z.re / scale).collect()
}

/// The floor of the radius in the spectrum (`np.maximum(kk, 1e-9)`): the DC mode of a spectrum that keeps it
/// (`low <= 0`) gets this radius instead of 0.
pub const FBM_RADIUS_FLOOR: f64 = 1e-9;
/// The floor of the standard deviation `periodic_fbm` divides by (`max(std, 1e-12)`): an all-zero field stays 0.
pub const FBM_STD_FLOOR: f64 = 1e-12;

/// The amplitude table of `periodic_fbm`, `size x size` indexed `[row][col]` like the field:
/// `np.where(kk >= low, np.maximum(kk, 1e-9) ** (-slope / 2.0), 0.0)` with `kk = hypot(k_col, k_row)` and
/// `k = fftfreq(size) * size`. The power is numpy's `array ** scalar` (see `np_power`), so slope 2 is an exact
/// reciprocal and slope -4 an exact square.
pub fn fbm_amplitude(size: usize, low: f64, slope: f64) -> Vec<f64> {
    let k = fft_k(size);
    let exponent = -slope / 2.0;
    let mut amp = vec![0.0; size * size];
    for r in 0..size {
        for c in 0..size {
            let kk = k[c].hypot(k[r]);
            amp[r * size + c] = if kk >= low { np_power(np_max(kk, FBM_RADIUS_FLOOR), exponent) } else { 0.0 };
        }
    }
    amp
}

/// `periodic_fbm(size, low, slope, seed)`: spectral synthesis. White noise filtered to a power spectrum
/// `k^-slope` above `low` cycles a tile (amplitude `max(kk, 1e-9)^(-slope / 2)` for `kk >= low`, 0 below, see
/// [`fbm_amplitude`]), periodic by construction, then mean 0 and population std 1 (the std floored at 1e-12,
/// so an all-zero field stays 0).
pub fn periodic_fbm(size: usize, low: f64, slope: f64, plan: &FbmPlan) -> Vec<f64> {
    assert_eq!(plan.white.len(), size * size, "periodic_fbm: the white noise must be size x size");
    let amp = fbm_amplitude(size, low, slope);
    let field = filter_fft2(&plan.white, size, &amp);
    let (mean, std) = mean_std(&field);
    let den = np_max(std, FBM_STD_FLOOR);
    field.iter().map(|&v| (v - mean) / den).collect()
}

const SQRT_1_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// `ln(f64::MAX)`, Cephes' `MAXLOG`: `exp(z)` underflows below `-MAXLOG`.
const MAXLOG: f64 = 7.09782712893383996843E2;

/// Cephes `polevl`: the polynomial with coefficients `coef` (highest degree first), Horner's rule.
fn polevl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

/// Cephes `p1evl`: like `polevl` for a polynomial whose leading coefficient is 1 and not stored.
fn p1evl(x: f64, coef: &[f64]) -> f64 {
    let mut ans = x + coef[0];
    for &c in &coef[1..] {
        ans = ans * x + c;
    }
    ans
}

// The rational approximations of Cephes ndtr.c, which scipy.special.{erf, erfc, ndtr} run:
// erf(x) = x T(x^2) / U(x^2) for |x| <= 1; erfc(x) = exp(-x^2) P(x) / Q(x) for 1 <= x < 8 and
// exp(-x^2) R(x) / S(x) for x >= 8.
// The literals keep all the digits of the Cephes source: a 16-digit rounding of a coefficient can land on the
// neighboring float64 and moves the result by an ulp or more.
const ERF_T: [f64; 5] = [
    9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4,
];
const ERF_U: [f64; 5] = [
    3.35617141647503099647E1,
    5.21357949780152679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4,
];
const ERFC_P: [f64; 9] = [
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2,
];
const ERFC_Q: [f64; 8] = [
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2,
];
const ERFC_R: [f64; 6] = [
    5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0,
];
const ERFC_S: [f64; 6] = [
    2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0,
];

/// `scipy.special.erf(x)` (Cephes `erf`): odd, `x T(x^2) / U(x^2)` for `|x| <= 1`, `1 - erfc(x)` beyond; NaN
/// propagates.
pub fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return -erf(-x);
    }
    if x.abs() > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &ERF_T) / p1evl(z, &ERF_U)
}

/// `scipy.special.erfc(a)` (Cephes `erfc`): `1 - erf(a)` for `|a| < 1`, otherwise `exp(-a^2)` times a rational
/// function of `|a|` (two ranges, the switch at 8), `2 - erfc(-a)` for negative `a`. `exp(-a^2)` is formed
/// as written in Cephes (no split of the square), so the relative error grows to a few 1e-14 near the
/// underflow at `|a| = 26.6`, where the result becomes 0 (or 2 for a negative `a`).
pub fn erfc(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a.abs();
    if x < 1.0 {
        return 1.0 - erf(a);
    }
    let under = if a < 0.0 { 2.0 } else { 0.0 };
    let z = -a * a;
    if z < -MAXLOG {
        return under;
    }
    let z = z.exp();
    let (p, q) = if x < 8.0 { (polevl(x, &ERFC_P), p1evl(x, &ERFC_Q)) } else { (polevl(x, &ERFC_R), p1evl(x, &ERFC_S)) };
    let mut y = (z * p) / q;
    if a < 0.0 {
        y = 2.0 - y;
    }
    if y == 0.0 {
        return under;
    }
    y
}

/// `scipy.special.ndtr(a)` (Cephes `ndtr`), the standard normal CDF, with `x = a / sqrt 2`: `0.5 + 0.5 erf(x)`
/// for `|x| < 1 / sqrt 2` (that is `|a| < 1`), else `0.5 erfc(|x|)`, taken as `1 - ` that for positive `x`.
/// The erfc form keeps the whole lower tail (`ndtr(-37) = 5.7e-300`). NaN propagates.
pub fn ndtr(a: f64) -> f64 {
    if a.is_nan() {
        return f64::NAN;
    }
    let x = a * SQRT_1_2;
    let z = x.abs();
    if z < SQRT_1_2 {
        0.5 + 0.5 * erf(x)
    } else {
        let y = 0.5 * erfc(z);
        if x > 0.0 {
            1.0 - y
        } else {
            y
        }
    }
}

/// `_uniform(size, low, seed, slope)`: a smooth periodic field, roughly uniform on [0, 1] (the normal CDF of a
/// unit fBm).
pub fn uniform(size: usize, low: f64, slope: f64, plan: &FbmPlan) -> Vec<f64> {
    periodic_fbm(size, low, slope, plan).into_iter().map(ndtr).collect()
}

/// `cell_size_field(size, km, seed)`: local cell diameter in tile units, log-uniform between the bounds
/// (`km` converted through [`REFERENCE_TILE_KM`]), smooth over a tile (`_uniform` with [`CELL_SIZE_LOW`],
/// [`CELL_SIZE_SLOPE`]).
pub fn cell_size_field(size: usize, km: (f64, f64), plan: &FbmPlan) -> Vec<f64> {
    let (lo, hi) = ((km.0 / REFERENCE_TILE_KM).ln(), (km.1 / REFERENCE_TILE_KM).ln());
    uniform(size, CELL_SIZE_LOW, CELL_SIZE_SLOPE, plan).into_iter().map(|u| (lo + (hi - lo) * u).exp()).collect()
}

// ---------------------------------------------------------------------------------------------------------
// Sampling, Poisson points

/// `_sample(field, p)`: nearest-texel lookup of a tile field at points in [0, 1)^2 (x = column, y = row):
/// `field[trunc(y m) % m, trunc(x m) % m]` with `m = rows`. Like Python it indexes BOTH axes with the number of
/// rows, so it is only defined for square fields (`cols >= rows` is required here; a wider field never reads
/// its extra columns). The cast truncates toward zero, the wrap is a floor modulo.
pub fn sample(field: &[f64], rows: usize, cols: usize, p: &[[f64; 2]]) -> Vec<f64> {
    assert_eq!(field.len(), rows * cols, "sample: field size does not match rows x cols");
    assert!(rows > 0 && cols >= rows, "sample: only defined for square fields (columns indexed by rows)");
    let m = rows as i64;
    p.iter()
        .map(|q| {
            let row = ((q[1] * rows as f64) as i64).rem_euclid(m) as usize;
            let col = ((q[0] * rows as f64) as i64).rem_euclid(m) as usize;
            field[row * cols + col]
        })
        .collect()
}

/// The draw of `poisson_points`: the candidates (`rng.random((n, 2))`, `n = int(12 / s_min^2)`) and the
/// thinning mask (`rng.random(n) < (s_min / size at the candidate)^2`, plan kind `tile.poisson`).
#[derive(Clone, Debug, PartialEq)]
pub struct PoissonPlan {
    /// Candidate points `[x, y]` in [0, 1)^2.
    pub cand: Vec<[f64; 2]>,
    /// Which candidates survive the thinning.
    pub keep: Vec<bool>,
}

/// `int(12.0 / s_min**2)`: the candidate count (truncation: 0.2 gives 299, not 300).
fn candidate_count(s_min: f64) -> usize {
    (POISSON_OVERSAMPLE / (s_min * s_min)) as usize
}

/// `size_field.min()`: NaN propagates.
fn field_min(f: &[f64]) -> f64 {
    f.iter().copied().fold(f64::INFINITY, |a, b| if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) })
}

impl PoissonPlan {
    /// The thinning mask from the uniforms `u` it compares: `u < (s_min / size at the candidate)^2`, STRICT,
    /// the square one multiplication, the size the nearest-texel sample of the `m x m` field.
    pub fn from_uniforms(size_field: &[f64], m: usize, cand: Vec<[f64; 2]>, u: &[f64]) -> PoissonPlan {
        assert_eq!(cand.len(), u.len(), "PoissonPlan: one uniform per candidate");
        let s_min = field_min(size_field);
        let s = sample(size_field, m, m, &cand);
        let keep = (0..cand.len())
            .map(|i| {
                let r = s_min / s[i];
                u[i] < r * r
            })
            .collect();
        PoissonPlan { cand, keep }
    }

    /// Draws the candidates, then the thinning uniforms, with `rng` (the two draws and their order are
    /// Python's).
    pub fn draw(size_field: &[f64], m: usize, rng: &mut Rng) -> PoissonPlan {
        let n = candidate_count(field_min(size_field));
        let cand: Vec<[f64; 2]> = (0..n)
            .map(|_| {
                let x = rng.uniform();
                let y = rng.uniform();
                [x, y]
            })
            .collect();
        let u: Vec<f64> = (0..n).map(|_| rng.uniform()).collect();
        PoissonPlan::from_uniforms(size_field, m, cand, &u)
    }
}

/// `poisson_points(size_field, seed, spacing)` with the indices of the accepted points into `cand[keep]`.
/// Variable-radius Poisson-disc points on the periodic unit tile: a point at `p` keeps every other point at
/// least `spacing x size(p)` away, so a Voronoi cell there is about `size(p)` across. Dart throwing in the
/// order of the kept candidates: a candidate is rejected when the PERIODIC distance to any accepted point is
/// STRICTLY below `spacing * max(its size, that point's size)`, the accepted point's size being the one
/// recorded when it was accepted. `size_field` is `m x m`.
pub fn poisson_points_indexed(
    size_field: &[f64],
    m: usize,
    plan: &PoissonPlan,
    spacing: f64,
) -> (Vec<[f64; 2]>, Vec<usize>) {
    assert_eq!(plan.cand.len(), plan.keep.len(), "poisson_points: one keep flag per candidate");
    let s_all = sample(size_field, m, m, &plan.cand);
    let kept: Vec<usize> = (0..plan.cand.len()).filter(|&i| plan.keep[i]).collect();
    let mut pts: Vec<[f64; 2]> = Vec::new();
    let mut radius: Vec<f64> = Vec::new();
    let mut index: Vec<usize> = Vec::new();
    for (pos, &ci) in kept.iter().enumerate() {
        let (p, sp) = (plan.cand[ci], s_all[ci]);
        let rejected = pts.iter().zip(&radius).any(|(q, &rq)| {
            let dx = (q[0] - p[0]).abs();
            let dy = (q[1] - p[1]).abs();
            let dx = dx.min(1.0 - dx);
            let dy = dy.min(1.0 - dy);
            dx.hypot(dy) < spacing * np_max(sp, rq)
        });
        if rejected {
            continue;
        }
        pts.push(p);
        radius.push(sp);
        index.push(pos);
    }
    (pts, index)
}

/// `poisson_points(size_field, seed, spacing)`: the accepted points (see [`poisson_points_indexed`]).
pub fn poisson_points(size_field: &[f64], m: usize, plan: &PoissonPlan, spacing: f64) -> Vec<[f64; 2]> {
    poisson_points_indexed(size_field, m, plan, spacing).0
}

/// `_texel_centers(size)`: `(size^2, 2)` rows `[x, y] = [(col + 0.5) / size, (row + 0.5) / size]` in row-major
/// texel order (numpy `meshgrid` default `xy` indexing: x is the fast index).
pub fn texel_centers(size: usize) -> Vec<[f64; 2]> {
    let c: Vec<f64> = (0..size).map(|i| (i as f64 + 0.5) / size as f64).collect();
    let mut out = Vec::with_capacity(size * size);
    for row in 0..size {
        for col in 0..size {
            out.push([c[col], c[row]]);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------
// Periodic nearest-neighbour search (scipy.spatial.cKDTree with boxsize = 1.0)

/// A static set of points of the periodic unit square with an exact k-nearest-neighbour query: the stand-in
/// for `cKDTree(points, boxsize=1.0).query(q, k)`. Distances are Euclidean on the minimum image per axis;
/// the points must lie in [0, 1) (as scipy requires), queries are wrapped. A uniform grid of about two
/// points a cell is searched ring by ring until no unseen cell can hold a closer point, so the answer is
/// exact. Equal distances are ordered by point index (scipy's order is unspecified).
#[derive(Clone, Debug)]
pub struct PeriodicPoints {
    pts: Vec<[f64; 2]>,
    g: usize,
    start: Vec<u32>,
    items: Vec<u32>,
}

impl PeriodicPoints {
    /// Builds the grid.
    pub fn new(pts: &[[f64; 2]]) -> PeriodicPoints {
        for p in pts {
            assert!((0.0..1.0).contains(&p[0]) && (0.0..1.0).contains(&p[1]), "PeriodicPoints: points must be in [0, 1)");
        }
        let g = (((pts.len() as f64) / 2.0).sqrt().round() as usize).max(1);
        let cell = |p: &[f64; 2]| ((p[1] * g as f64) as usize).min(g - 1) * g + ((p[0] * g as f64) as usize).min(g - 1);
        let mut start = vec![0u32; g * g + 1];
        for p in pts {
            start[cell(p) + 1] += 1;
        }
        for i in 0..g * g {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut items = vec![0u32; pts.len()];
        for (i, p) in pts.iter().enumerate() {
            let c = cell(p);
            items[fill[c] as usize] = i as u32;
            fill[c] += 1;
        }
        PeriodicPoints { pts: pts.to_vec(), g, start, items }
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.pts.len()
    }

    /// True when there is no point.
    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }

    /// The `K` nearest points to `q` as `(distance, index)`, nearest first. Fewer than `K` points leaves the
    /// scipy fill: distance infinity and index `len()`.
    pub fn nearest<const K: usize>(&self, q: [f64; 2]) -> [(f64, usize); K] {
        assert!(K >= 1, "nearest: K must be at least 1");
        let n = self.pts.len();
        // (squared distance, index), sorted; missing entries are (inf, n).
        let mut best = [(f64::INFINITY, n); K];
        let q = [q[0] - q[0].floor(), q[1] - q[1].floor()];
        let g = self.g as i64;
        let cs = 1.0 / self.g as f64;
        let cx = ((q[0] * self.g as f64) as i64).min(g - 1);
        let cy = ((q[1] * self.g as f64) as i64).min(g - 1);
        let lo = -((g - 1) / 2);
        let hi = g / 2;
        let visit = |dx: i64, dy: i64, best: &mut [(f64, usize); K]| {
            if dx < lo || dx > hi {
                return;
            }
            let cell = (cy + dy).rem_euclid(g) as usize * self.g + (cx + dx).rem_euclid(g) as usize;
            for &it in &self.items[self.start[cell] as usize..self.start[cell + 1] as usize] {
                let p = self.pts[it as usize];
                let ax = (p[0] - q[0]).abs();
                let ay = (p[1] - q[1]).abs();
                let ax = ax.min(1.0 - ax);
                let ay = ay.min(1.0 - ay);
                let cand = (ax * ax + ay * ay, it as usize);
                if cand >= best[K - 1] {
                    continue;
                }
                let mut pos = K - 1;
                while pos > 0 && cand < best[pos - 1] {
                    best[pos] = best[pos - 1];
                    pos -= 1;
                }
                best[pos] = cand;
            }
        };
        // Rings of cells around the query cell, each cell once: the offsets are the canonical ones
        // (-(g - 1) / 2 ..= g / 2), so a small grid does not visit a cell twice.
        let mut r = 0_i64;
        while r <= hi {
            for dy in -r..=r {
                if dy < lo || dy > hi {
                    continue;
                }
                if dy.abs() == r {
                    for dx in -r..=r {
                        visit(dx, dy, &mut best);
                    }
                } else {
                    visit(-r, dy, &mut best);
                    visit(r, dy, &mut best);
                }
            }
            // Every unseen cell is more than r cells away along an axis: nothing there is nearer than r cells.
            let reach = r as f64 * cs;
            if best[K - 1].0 <= reach * reach {
                break;
            }
            r += 1;
        }
        let mut out = [(f64::INFINITY, n); K];
        for k in 0..K {
            out[k] = (best[k].0.sqrt(), best[k].1);
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------------------
// Warped variable Voronoi cells

/// The draws of `warped_voronoi`, in call order: the Poisson candidates and thinning, then the white noise of
/// the two warp fBms (x, then y; low [`WARP_LOW`], slope [`WARP_SLOPE`]).
#[derive(Clone, Debug, PartialEq)]
pub struct WarpedVoronoiPlan {
    pub poisson: PoissonPlan,
    pub warp_x: FbmPlan,
    pub warp_y: FbmPlan,
}

impl WarpedVoronoiPlan {
    /// Draws the three parts with `rng`, in Python's call order. `size_field` is the `size x size` cell size
    /// field the Poisson set is thinned by.
    pub fn draw(size: usize, size_field: &[f64], rng: &mut Rng) -> WarpedVoronoiPlan {
        let poisson = PoissonPlan::draw(size_field, size, rng);
        let warp_x = FbmPlan::draw(size, rng);
        let warp_y = FbmPlan::draw(size, rng);
        WarpedVoronoiPlan { poisson, warp_x, warp_y }
    }
}

/// What `warped_voronoi` computes, with the Poisson points it used (an intermediate in Python).
#[derive(Clone, Debug)]
pub struct VoronoiCells {
    /// `(F2 - F1) / size of the nearest cell`: 0 on a border, about 0.5 to 1 at a center. Infinite everywhere
    /// when there is a single point (no second neighbor).
    pub border: Vec<f64>,
    /// The size field sampled at the nearest cell's point (a copy of a `size_field` value).
    pub s_cell: Vec<f64>,
    /// The accepted Poisson points.
    pub points: Vec<[f64; 2]>,
}

/// `warped_voronoi(size, size_field, seed)` with its intermediate points: the variable cells at texel
/// centers. The query point is the texel center plus `BORDER_WARP x (size at the texel) x [fbm_x, fbm_y]`
/// wrapped on the tile, so the borders bend instead of running straight; the two nearest of the Poisson
/// points come from a periodic search. `size_field` is `size x size` and the Poisson set must not be empty.
pub fn warped_voronoi_cells(size: usize, size_field: &[f64], plan: &WarpedVoronoiPlan) -> VoronoiCells {
    assert_eq!(size_field.len(), size * size, "warped_voronoi: the size field must be size x size");
    let points = poisson_points(size_field, size, &plan.poisson, POISSON_SPACING);
    assert!(!points.is_empty(), "warped_voronoi: no Poisson point (no cell to find)");
    let fx = periodic_fbm(size, WARP_LOW, WARP_SLOPE, &plan.warp_x);
    let fy = periodic_fbm(size, WARP_LOW, WARP_SLOPE, &plan.warp_y);
    let q = texel_centers(size);
    let tree = PeriodicPoints::new(&points);
    let s_pts = sample(size_field, size, size, &points);
    let mut border = Vec::with_capacity(size * size);
    let mut s_cell = Vec::with_capacity(size * size);
    for i in 0..size * size {
        let w = BORDER_WARP * size_field[i];
        let (wx, wy) = (w * fx[i], w * fy[i]);
        let (qx, qy) = (q[i][0] + wx, q[i][1] + wy);
        let query = [qx - qx.floor(), qy - qy.floor()];
        let nn = tree.nearest::<2>(query);
        let s = s_pts[nn[0].1];
        border.push((nn[1].0 - nn[0].0) / s);
        s_cell.push(s);
    }
    VoronoiCells { border, s_cell, points }
}

/// `warped_voronoi(size, size_field, seed)`: `(border, cell size)` per texel, see [`warped_voronoi_cells`].
pub fn warped_voronoi(size: usize, size_field: &[f64], plan: &WarpedVoronoiPlan) -> (Vec<f64>, Vec<f64>) {
    let v = warped_voronoi_cells(size, size_field, plan);
    (v.border, v.s_cell)
}

// ---------------------------------------------------------------------------------------------------------
// Lumps

/// The draw of `lumps`: the jittered `freq x freq` lump grid (`freq = round(240 / 4) = 60`, centers `g` as
/// `(grid + rng.random((freq^2, 2))) / freq`, `[x, y]`, row-major with x the column) and the lump radii `r`
/// (`rng.uniform(0.35, 0.7, freq^2) / freq`), plan kind `tile.lumps`.
#[derive(Clone, Debug, PartialEq)]
pub struct LumpsPlan {
    pub g: Vec<[f64; 2]>,
    pub r: Vec<f64>,
}

/// `int(round(REFERENCE_TILE_KM / LUMP_KM))`: lumps along a tile side, 60 (Python `round`: half to even). A
/// lump plan has `lump_freq()^2` centers and radii.
pub fn lump_freq() -> usize {
    (REFERENCE_TILE_KM / LUMP_KM).round_ties_even() as usize
}

impl LumpsPlan {
    /// Draws the jitter of every lump center, then every radius, with `rng` (Python's two draws, in order).
    pub fn draw(rng: &mut Rng) -> LumpsPlan {
        let freq = lump_freq();
        let n = freq * freq;
        let jitter: Vec<[f64; 2]> = (0..n)
            .map(|_| {
                let x = rng.uniform();
                let y = rng.uniform();
                [x, y]
            })
            .collect();
        let g = (0..n)
            .map(|k| {
                let (i, j) = (k / freq, k % freq);
                [(j as f64 + jitter[k][0]) / freq as f64, (i as f64 + jitter[k][1]) / freq as f64]
            })
            .collect();
        let r = (0..n).map(|_| rng.uniform_range(LUMP_RADIUS.0, LUMP_RADIUS.1) / freq as f64).collect();
        LumpsPlan { g, r }
    }
}

/// `lumps(size, seed)`: separate cumulus domes on a jittered grid about `LUMP_KM` apart, each with its own
/// radius: 1 at a lump's center, 0 outside every lump. The dome at a texel is the max over its FOUR NEAREST
/// lumps (by distance, periodic; a fifth lump covering the texel is ignored, as in Python) of
/// `clip(1 - dist / r, 0, 1)^0.6` ([`LUMP_NEAREST`], [`LUMP_EXPONENT`]).
pub fn lumps(size: usize, plan: &LumpsPlan) -> Vec<f64> {
    assert_eq!(plan.g.len(), plan.r.len(), "lumps: one radius per center");
    let tree = PeriodicPoints::new(&plan.g);
    texel_centers(size)
        .into_iter()
        .map(|q| {
            // Every value below is at least 0, so 0 is the identity of the max (and the answer when fewer
            // than four lumps exist, where scipy reports index n and Python would fail).
            let mut dome = 0.0;
            for (dist, idx) in tree.nearest::<LUMP_NEAREST>(q) {
                if idx < plan.r.len() {
                    dome = np_max(dome, np_clip(1.0 - dist / plan.r[idx], 0.0, 1.0).powf(LUMP_EXPONENT));
                }
            }
            dome
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------
// The four channels

/// The draws of the closed-cells channel, in call order.
///
/// The crack width, `_uniform(3.0)`, is not a draw of its own: Python seeds it with the seed of the y-warp fBm
/// (`seed + 105 = (seed + 103) + 2`, same low 3 and slope 3), so it is the SAME white noise as
/// `voronoi.warp_y` (width and warp are one correlated field). `raw_channels` reads `voronoi.warp_y` for
/// both, which makes the sharing a property of the kernel instead of a promise of the constructor.
#[derive(Clone, Debug, PartialEq)]
pub struct ClosedPlan {
    /// `cell_size_field` (low 1, slope 3).
    pub cell_size: FbmPlan,
    pub voronoi: WarpedVoronoiPlan,
    /// `_uniform(4.0)`: where the cracks open.
    pub open_crack: FbmPlan,
    /// `_uniform(6.0)`: crack depth.
    pub depth: FbmPlan,
    /// The lumpy tops' own 64-cell Worley lattice.
    pub tops: WorleyPlan,
    /// `_uniform(12.0, slope = 2)`: top texture.
    pub texture: FbmPlan,
}

/// The draws of the open-cells channel, in call order.
///
/// The wall width, `_uniform(3.0)`, is the SAME white noise as `voronoi.warp_y` (Python's seed + 155 is the
/// seed of the y-warp, `(seed + 153) + 2`): see [`ClosedPlan`].
#[derive(Clone, Debug, PartialEq)]
pub struct OpenPlan {
    pub cell_size: FbmPlan,
    pub voronoi: WarpedVoronoiPlan,
    /// `_uniform(5.0)`: the walls that are missing.
    pub wall_gaps: FbmPlan,
    pub lumps: LumpsPlan,
}

/// Every draw of `raw_channels`, in Python's evaluation order (heaps; closed; open; sheet).
#[derive(Clone, Debug, PartialEq)]
pub struct RawChannelsPlan {
    /// Worley lattice of the heaps' clusters (`HEAP_CLUSTERS` cells a side).
    pub clusters: WorleyPlan,
    /// Heaps' cells (`HEAP_CELLS`).
    pub cells: WorleyPlan,
    /// Heaps' puffs (`PUFFS`).
    pub puffs: WorleyPlan,
    pub closed: ClosedPlan,
    pub open: OpenPlan,
    pub sheet: FbmPlan,
}

impl RawChannelsPlan {
    /// Draws every table of a `size` tile with `rng`, in Python's order (the crack width and the wall width are
    /// not drawn: they reuse the y-warp noise of their channel, see [`ClosedPlan`]).
    pub fn draw(size: usize, rng: &mut Rng) -> RawChannelsPlan {
        let clusters = WorleyPlan::draw(HEAP_CLUSTERS, rng);
        let cells = WorleyPlan::draw(HEAP_CELLS, rng);
        let puffs = WorleyPlan::draw(PUFFS, rng);

        let cell_size = FbmPlan::draw(size, rng);
        let field = cell_size_field(size, CLOSED_CELL_KM, &cell_size);
        let voronoi = WarpedVoronoiPlan::draw(size, &field, rng);
        let open_crack = FbmPlan::draw(size, rng);
        let depth = FbmPlan::draw(size, rng);
        let tops = WorleyPlan::draw(64, rng);
        let texture = FbmPlan::draw(size, rng);
        let closed = ClosedPlan { cell_size, voronoi, open_crack, depth, tops, texture };

        let cell_size = FbmPlan::draw(size, rng);
        let field = cell_size_field(size, OPEN_CELL_KM, &cell_size);
        let voronoi = WarpedVoronoiPlan::draw(size, &field, rng);
        let wall_gaps = FbmPlan::draw(size, rng);
        let lumps = LumpsPlan::draw(rng);
        let open = OpenPlan { cell_size, voronoi, wall_gaps, lumps };

        let sheet = FbmPlan::draw(size, rng);
        RawChannelsPlan { clusters, cells, puffs, closed, open, sheet }
    }
}

/// The four fields before rank encoding, `size * size` each; higher = more cloud.
#[derive(Clone, Debug)]
pub struct RawChannels {
    pub heaps: Vec<f64>,
    pub closed: Vec<f64>,
    pub open: Vec<f64>,
    pub sheet: Vec<f64>,
}

impl RawChannels {
    /// The channels in [`CHANNELS`] order.
    pub fn in_order(&self) -> [&Vec<f64>; 4] {
        [&self.heaps, &self.closed, &self.open, &self.sheet]
    }
}

/// `raw_channels(size, seed)`: the four fields before rank encoding.
///
/// * HEAPS: puffs (96 a tile) under an envelope of cells (32) and clusters (8).
/// * CLOSED: cracks of a warped Voronoi whose half-width wanders along them and closes where `open_crack`
///   drops to 0, domes sagging toward an open crack, lumpy tops on their own lattice.
/// * OPEN: a wall band around each clear center (width wandering, some walls missing) drawn by separate lumps,
///   plus a faint wall term and a distance term that keep the centers off one plateau.
/// * SHEET: a periodic fBm (low 2, slope 3).
pub fn raw_channels(size: usize, plan: &RawChannelsPlan) -> RawChannels {
    let n = size * size;
    // HEAPS
    let clusters = inverted(&tileable_worley(size, &plan.clusters).0);
    let cells = inverted(&tileable_worley(size, &plan.cells).0);
    let puffs = inverted(&tileable_worley(size, &plan.puffs).0);
    let mut heaps = Vec::with_capacity(n);
    for i in 0..n {
        let envelope = 0.5 * clusters[i] + 0.5 * cells[i];
        heaps.push(ENVELOPE_FLOOR * envelope + puffs[i] * (HEAP_BASE + (1.0 - HEAP_BASE - ENVELOPE_FLOOR) * envelope));
    }

    // CLOSED
    let cp = &plan.closed;
    let field = cell_size_field(size, CLOSED_CELL_KM, &cp.cell_size);
    let (border, _) = warped_voronoi(size, &field, &cp.voronoi);
    // The crack width is drawn with the seed of the y-warp: the same white noise (see `ClosedPlan`).
    let width: Vec<f64> = uniform(size, 3.0, 3.0, &cp.voronoi.warp_y).into_iter().map(|u| 0.03 + 0.22 * u).collect();
    let open_crack = smoothstep(0.25, 0.65, &uniform(size, 4.0, 3.0, &cp.open_crack));
    let ratio: Vec<f64> = (0..n).map(|i| border[i] / width[i]).collect();
    let crack_ramp = smoothstep(0.0, 1.0, &ratio);
    let depth: Vec<f64> = uniform(size, 6.0, 3.0, &cp.depth).into_iter().map(|u| 0.55 + 0.4 * u).collect();
    let sag_ramp = smoothstep(0.0, 0.6, &border);
    let tops = inverted(&tileable_worley(size, &cp.tops).0);
    let tex_u = uniform(size, 12.0, 2.0, &cp.texture);
    let mut closed = Vec::with_capacity(n);
    for i in 0..n {
        let crack = open_crack[i] * (1.0 - crack_ramp[i]);
        let sag = 1.0 - 0.35 * open_crack[i] * (1.0 - sag_ramp[i]);
        let texture = 0.3 * tops[i] + 0.7 * tex_u[i];
        closed.push(sag * (0.7 + 0.3 * texture) * (1.0 - depth[i] * crack));
    }

    // OPEN
    let op = &plan.open;
    let field = cell_size_field(size, OPEN_CELL_KM, &op.cell_size);
    let (border_o, _) = warped_voronoi(size, &field, &op.voronoi);
    // The wall width is drawn with the seed of the y-warp: the same white noise (see `OpenPlan`).
    let wall_width: Vec<f64> = uniform(size, 3.0, 3.0, &op.voronoi.warp_y).into_iter().map(|u| 0.15 + 0.3 * u).collect();
    let ratio: Vec<f64> = (0..n).map(|i| border_o[i] / wall_width[i]).collect();
    let wall_ramp = smoothstep(0.0, 1.0, &ratio);
    let gaps = smoothstep(0.2, 0.6, &uniform(size, 5.0, 3.0, &op.wall_gaps));
    let domes = lumps(size, &op.lumps);
    let mut open = Vec::with_capacity(n);
    for i in 0..n {
        let mut wall = 1.0 - wall_ramp[i];
        wall *= 0.25 + 0.75 * gaps[i];
        let inward = 1.0 - np_clip(border_o[i], 0.0, 1.0);
        open.push(wall * domes[i] + 0.03 * wall + 0.02 * inward);
    }

    // SHEET
    let sheet = periodic_fbm(size, SHEET_LOW, SHEET_SLOPE, &plan.sheet);
    RawChannels { heaps, closed, open, sheet }
}

// ---------------------------------------------------------------------------------------------------------
// Rank encoding

/// `rank_bytes(field)`: the field's rank as a byte, `floor(256 rank / n)`, a uniform histogram (4096 texels a
/// level at 1024^2), in the order of the flattened field. Equal values (decided in the dtype of the field,
/// `-0.0 == 0.0`, all NaNs one value sorted last) take their MEAN rank `first + (count - 1) / 2`: a plateau
/// stays one value. The product is `rank * 256` THEN divided by `n` (the other order is off by one at some
/// ranks), truncated, and clamped to 255 (a dead bound).
pub fn rank_bytes<F: Float>(field: &[F]) -> Vec<u8> {
    let n = field.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        let (x, y) = (field[a as usize], field[b as usize]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => x.partial_cmp(&y).expect("non-NaN values compare"),
        }
    });
    let same = |a: F, b: F| a == b || (a.is_nan() && b.is_nan());
    let mut out = vec![0u8; n];
    let mut i = 0;
    while i < n {
        let first = field[order[i] as usize];
        let mut j = i + 1;
        while j < n && same(first, field[order[j] as usize]) {
            j += 1;
        }
        let mean_rank = i as f64 + (j - i - 1) as f64 / 2.0;
        let byte = ((mean_rank * 256.0 / n as f64) as i64).min(255) as u8;
        for &k in &order[i..j] {
            out[k as usize] = byte;
        }
        i = j;
    }
    out
}

/// `coverage_masks(rank, coverages)`: what the deck draws of a pure regime at each coverage, the top
/// `coverage` fraction of texels (a rank byte at or above `round(256 (1 - coverage))`, Python `round`: half to
/// even). The threshold may fall below 0 or above 255; the comparison does not wrap. One mask per coverage.
pub fn coverage_masks(rank: &[u8], coverages: &[f64]) -> Vec<Vec<bool>> {
    coverages
        .iter()
        .map(|&c| {
            let threshold = (256.0 * (1.0 - c)).round_ties_even() as i64;
            rank.iter().map(|&r| i64::from(r) >= threshold).collect()
        })
        .collect()
}

/// `build_tile(size, seed)`: the tile, `size * size * 4` bytes RGBA in [`CHANNELS`] order (channel last, rows
/// in scan order, no flip), every channel a rank.
pub fn build_tile(size: usize, plan: &RawChannelsPlan) -> Vec<u8> {
    let raw = raw_channels(size, plan);
    let ranks: Vec<Vec<u8>> = raw.in_order().iter().map(|f| rank_bytes(f)).collect();
    let mut tile = Vec::with_capacity(size * size * 4);
    for i in 0..size * size {
        for r in &ranks {
            tile.push(r[i]);
        }
    }
    tile
}

/// A tile of a seed: draws a [`RawChannelsPlan`] with `Rng::new(seed)` and builds it. Statistically equivalent
/// to Python's `build_tile(size, seed)`, not identical (the draws are this crate's own).
pub fn build_tile_seeded(size: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    build_tile(size, &RawChannelsPlan::draw(size, &mut rng))
}

// ---------------------------------------------------------------------------------------------------------
// What bake_tiles.py pins: flags, file names, the RGBA bytes, the preview, the sidecar

/// Default `--version` of the bake.
pub const DEFAULT_BAKE_VERSION: &str = "v2";
/// Default `--seed` of the bake.
pub const DEFAULT_BAKE_SEED: i64 = 1;
/// The `generatedBy` of the sidecar (the project's former name, frozen in the golden data), kept as the Python script writes it so that the sidecars compare equal.
pub const BAKE_GENERATED_BY: &str = "planet-cloudgen bake_tiles.py";

/// The flags of `bake_tiles.py`: `--version` (default `v2`), `--seed` (default 1), `--size` (default
/// [`TILE_SIZE`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BakeArgs {
    pub version: String,
    pub seed: i64,
    pub size: usize,
}

impl Default for BakeArgs {
    fn default() -> BakeArgs {
        BakeArgs { version: DEFAULT_BAKE_VERSION.to_string(), seed: DEFAULT_BAKE_SEED, size: TILE_SIZE }
    }
}

/// What argparse does when it stops the script: `parser.exit(code, message)`. Code 0 is `-h/--help` (the
/// message is the help text, printed to stdout); code 2 is an error (the message is the usage and the
/// `bake_tiles.py: error: ...` line, printed to stderr).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgsExit {
    pub code: i32,
    pub message: String,
}

/// The `usage:` line argparse prints for the parser of `bake_tiles.py` (`prog` = `bake_tiles.py`).
pub const BAKE_USAGE: &str = "usage: bake_tiles.py [-h] [--version VERSION] [--seed SEED] [--size SIZE]\n";

/// The help argparse (Python 3.12) prints for `-h`.
pub const BAKE_HELP: &str = "usage: bake_tiles.py [-h] [--version VERSION] [--seed SEED] [--size SIZE]\n\noptions:\n  -h, --help         show this help message and exit\n  --version VERSION\n  --seed SEED\n  --size SIZE\n";

/// The actions of the parser.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BakeOpt {
    Help,
    Version,
    Seed,
    Size,
}

/// The option strings, in the order argparse registered them (`-h`, `--help`, then the three flags): the order
/// of the matches an ambiguous abbreviation lists.
const BAKE_OPTION_STRINGS: [(&str, BakeOpt); 5] =
    [("-h", BakeOpt::Help), ("--help", BakeOpt::Help), ("--version", BakeOpt::Version), ("--seed", BakeOpt::Seed), ("--size", BakeOpt::Size)];

/// How an option string carries an argument of its own: `--seed=3` (`Sep`) or the tail of a single-dash
/// short option, `-hx` (`Short`).
#[derive(Clone, Debug, PartialEq, Eq)]
enum Explicit {
    None,
    Sep(String),
    Short(String),
}

/// argparse's verdict on one argument string (`_parse_optional`): a value ("A"), or an option ("O"): one this
/// parser knows, one it does not (`opt: None`), or an abbreviation that fits several (the error is raised
/// when the parse reaches it, not before).
#[derive(Clone, Debug, PartialEq, Eq)]
enum ArgKind {
    Positional,
    Option { opt: Option<BakeOpt>, name: String, explicit: Explicit },
    Ambiguous(String),
}

/// `repr(str)` of Python, for the error messages: single quotes unless the text holds one and no double
/// quote, backslash, `\n`, `\r`, `\t`, and other control and non-space whitespace characters as `\xNN`,
/// `\uNNNN` or `\UNNNNNNNN`. (`str.isprintable` also escapes format, unassigned and private-use characters,
/// which need Unicode tables this crate does not carry.)
fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c == ' ' || (!c.is_control() && !c.is_whitespace()) => out.push(c),
            c if (c as u32) < 0x100 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (c as u32) < 0x10000 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push_str(&format!("\\U{:08x}", c as u32)),
        }
    }
    out.push(quote);
    out
}

/// Why `int(text)` of Python did not give a value this port can hold.
#[derive(Debug, PartialEq, Eq)]
enum IntError {
    /// Python's `ValueError`: not an integer literal.
    Invalid,
    /// A valid integer beyond 64 bits (Python's ints do not overflow, this port's do).
    OutOfRange,
}

/// `int(text)` of Python (base 10): surrounding whitespace, an optional sign, ASCII digits with single
/// underscores between digits (`1_000`). Non-ASCII decimal digits (`int('\u{663}')` is 3 in Python) are not
/// accepted here.
fn py_int(text: &str) -> Result<i64, IntError> {
    let t = text.trim();
    let (negative, digits) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    let well_formed = !digits.is_empty()
        && !digits.starts_with('_')
        && !digits.ends_with('_')
        && !digits.contains("__")
        && digits.chars().all(|c| c.is_ascii_digit() || c == '_');
    if !well_formed {
        return Err(IntError::Invalid);
    }
    let mut value: i128 = 0;
    for c in digits.chars().filter(|&c| c != '_') {
        value = value * 10 + i128::from(c.to_digit(10).expect("an ASCII digit"));
        if value > i128::from(u64::MAX) * 2 {
            return Err(IntError::OutOfRange);
        }
    }
    i64::try_from(if negative { -value } else { value }).map_err(|_| IntError::OutOfRange)
}

/// argparse's `_negative_number_matcher` (`^-\d+$|^-\d*\.\d+$`, ASCII digits): a "-3" that is a value, not
/// an option, because the parser has no option that looks like a negative number.
fn looks_like_negative_number(arg: &str) -> bool {
    let Some(rest) = arg.strip_prefix('-') else { return false };
    let digits = |t: &str| t.chars().all(|c| c.is_ascii_digit());
    match rest.split_once('.') {
        Some((int, frac)) => digits(int) && !frac.is_empty() && digits(frac),
        None => !rest.is_empty() && digits(rest),
    }
}

/// argparse's `_get_option_tuples`: the options an abbreviation could mean (long prefixes, `--se` and
/// `--se=3`; the tail of a short option, `-hx`).
fn option_tuples(arg: &str) -> Vec<(BakeOpt, &'static str, Explicit)> {
    let mut found = Vec::new();
    if arg.starts_with("--") {
        let (prefix, explicit) = match arg.split_once('=') {
            Some((p, e)) => (p, Explicit::Sep(e.to_string())),
            None => (arg, Explicit::None),
        };
        for &(name, opt) in &BAKE_OPTION_STRINGS {
            if name.starts_with(prefix) {
                found.push((opt, name, explicit.clone()));
            }
        }
    } else {
        // A single dash and a second character that is not a dash: split after the first two characters.
        let cut = arg.char_indices().nth(2).map_or(arg.len(), |(i, _)| i);
        let (short, tail) = arg.split_at(cut);
        for &(name, opt) in &BAKE_OPTION_STRINGS {
            if name == short {
                found.push((opt, name, Explicit::Short(tail.to_string())));
            } else if name.starts_with(arg) {
                found.push((opt, name, Explicit::None));
            }
        }
    }
    found
}

/// argparse's `_parse_optional` for this parser (`allow_abbrev` on).
fn classify_arg(arg: &str) -> ArgKind {
    if !arg.starts_with('-') {
        return ArgKind::Positional;
    }
    if let Some(&(name, opt)) = BAKE_OPTION_STRINGS.iter().find(|(n, _)| *n == arg) {
        return ArgKind::Option { opt: Some(opt), name: name.to_string(), explicit: Explicit::None };
    }
    if arg.len() == 1 {
        return ArgKind::Positional; // a lone "-"
    }
    if let Some((head, value)) = arg.split_once('=') {
        if let Some(&(name, opt)) = BAKE_OPTION_STRINGS.iter().find(|(n, _)| *n == head) {
            return ArgKind::Option { opt: Some(opt), name: name.to_string(), explicit: Explicit::Sep(value.to_string()) };
        }
    }
    let mut tuples = option_tuples(arg);
    if tuples.len() > 1 {
        let names: Vec<&str> = tuples.iter().map(|t| t.1).collect();
        return ArgKind::Ambiguous(format!("ambiguous option: {arg} could match {}", names.join(", ")));
    }
    if let Some((opt, name, explicit)) = tuples.pop() {
        return ArgKind::Option { opt: Some(opt), name: name.to_string(), explicit };
    }
    if looks_like_negative_number(arg) || arg.contains(' ') {
        return ArgKind::Positional;
    }
    ArgKind::Option { opt: None, name: arg.to_string(), explicit: Explicit::None }
}

/// What `consume_optional` of argparse does with the tail of `-h<tail>` (no `=`): it reads the tail as more
/// single-dash flags, and the only one there is, `-h`, takes no argument. `Ok` when the help runs (`-hx`,
/// `-hh`: an unknown flag in the tail goes to the extras, but the help exits first), `Err` with the text of
/// `ignored explicit argument` when the tail starts with a dash (`-h-x`) or a chained `-h` is followed by
/// one (`-hh-x`) or by `=` (`-hh=3`).
fn help_tail(tail: &str) -> Result<(), String> {
    let (mut text, mut sep) = (tail, false);
    loop {
        if text.is_empty() || sep || text.starts_with('-') {
            return Err(text.to_string());
        }
        let mut chars = text.chars();
        if chars.next() != Some('h') {
            return Ok(());
        }
        let rest = chars.as_str();
        if rest.is_empty() {
            return Ok(());
        }
        match rest.strip_prefix('=') {
            Some(after) => (text, sep) = (after, true),
            None => (text, sep) = (rest, false),
        }
    }
}

fn bake_error(message: &str) -> ArgsExit {
    ArgsExit { code: 2, message: format!("{BAKE_USAGE}bake_tiles.py: error: {message}\n") }
}

impl BakeArgs {
    /// The argparse of `bake_tiles.py` (`--version`, `--seed`, `--size`, all optional, Python 3.12), with its
    /// own behaviors: `--flag value` or `--flag=value`, any order, the last one wins; a unique prefix of a long
    /// flag stands for it (`--ver`, `--se=5`; `--s` is ambiguous); a value that starts with a dash is a value
    /// only when it is a negative number or has a space (`--seed -3`), otherwise the flag is missing its
    /// argument; `int()` forms (`1_000`, ` 7 `, `+5`) are accepted by `--seed` and `--size`; `-h` and
    /// `--help` (and `-hx`, `--he`) end the parse with the help; an unknown argument is reported at the end
    /// (`unrecognized arguments`), and everything from a `--` on is one. Errors come in order, the first
    /// wins, so `-h --s` is the help and `--s -h` is the ambiguity (`--s` could be `--seed` or `--size`); an
    /// ambiguous abbreviation in the place of a value is a missing value. `argv` does not include the
    /// program name.
    ///
    /// Two deviations, both beyond what a build script is fed: Python's `int` also reads non-ASCII decimal
    /// digits and integers beyond 64 bits (`--seed` takes an `i64` here, `--size` a `usize`: a value outside
    /// them is an error with its own message, reported last: after any help, any error and any unrecognized
    /// argument that Python would have reported).
    pub fn parse(argv: &[String]) -> Result<BakeArgs, ArgsExit> {
        // argparse classifies every argument up to a `--` before it consumes any (and none after it).
        let separator = argv.iter().position(|a| a == "--");
        let kinds: Vec<ArgKind> = argv[..separator.unwrap_or(argv.len())].iter().map(|a| classify_arg(a)).collect();
        let mut args = BakeArgs::default();
        let mut extras: Vec<&str> = Vec::new();
        // A value Python accepts and this port cannot hold (beyond 64 bits, a negative size): the parse goes on
        // as Python's would (a later flag may replace it, an earlier exit still comes first) and it is an
        // error only at the end.
        let (mut seed_limit, mut size_limit): (Option<String>, Option<String>) = (None, None);
        let mut i = 0;
        while i < kinds.len() {
            match &kinds[i] {
                ArgKind::Ambiguous(message) => return Err(bake_error(message)),
                ArgKind::Option { opt: Some(BakeOpt::Help), explicit, .. } => {
                    let ignored = match explicit {
                        Explicit::None => None,
                        Explicit::Sep(text) => Some(text.clone()),
                        Explicit::Short(tail) => help_tail(tail).err(),
                    };
                    return Err(match ignored {
                        Some(text) => bake_error(&format!("argument -h/--help: ignored explicit argument {}", py_repr(&text))),
                        None => ArgsExit { code: 0, message: BAKE_HELP.to_string() },
                    });
                }
                ArgKind::Option { opt: Some(opt), name, explicit } => {
                    let value: &str = match explicit {
                        Explicit::Sep(v) | Explicit::Short(v) => v,
                        Explicit::None => match kinds.get(i + 1) {
                            Some(ArgKind::Positional) => {
                                i += 1;
                                &argv[i]
                            }
                            _ => return Err(bake_error(&format!("argument {name}: expected one argument"))),
                        },
                    };
                    let invalid = || bake_error(&format!("argument {name}: invalid int value: {}", py_repr(value)));
                    match opt {
                        BakeOpt::Version => args.version = value.to_string(),
                        BakeOpt::Seed => match py_int(value) {
                            Ok(v) => (args.seed, seed_limit) = (v, None),
                            Err(IntError::Invalid) => return Err(invalid()),
                            Err(IntError::OutOfRange) => {
                                seed_limit = Some(format!("argument {name}: {} does not fit this port's 64-bit seed", py_repr(value)))
                            }
                        },
                        BakeOpt::Size => match py_int(value).map(usize::try_from) {
                            Ok(Ok(v)) => (args.size, size_limit) = (v, None),
                            Err(IntError::Invalid) => return Err(invalid()),
                            _ => {
                                size_limit = Some(format!("argument {name}: {} is not a size this port can build", py_repr(value)))
                            }
                        },
                        BakeOpt::Help => unreachable!("handled above"),
                    }
                }
                ArgKind::Positional | ArgKind::Option { opt: None, .. } => extras.push(&argv[i]),
            }
            i += 1;
        }
        if let Some(at) = separator {
            extras.extend(argv[at..].iter().map(String::as_str));
        }
        if !extras.is_empty() {
            return Err(bake_error(&format!("unrecognized arguments: {}", extras.join(" "))));
        }
        match seed_limit.or(size_limit) {
            Some(message) => Err(bake_error(&message)),
            None => Ok(args),
        }
    }
}

/// The stem of the baked files, `cloud_cell_tiles_<size>_<version>`.
pub fn bake_stem(size: usize, version: &str) -> String {
    format!("cloud_cell_tiles_{size}_{version}")
}

/// The files a bake writes, relative to the output root, sorted, `/`-separated:
/// `output/tiles/<version>/<stem>.json`, `.rgba.gz` and `_preview.png`.
pub fn bake_files(version: &str, size: usize) -> Vec<String> {
    let stem = bake_stem(size, version);
    let dir = format!("output/tiles/{version}");
    let mut files = vec![format!("{dir}/{stem}.rgba.gz"), format!("{dir}/{stem}_preview.png"), format!("{dir}/{stem}.json")];
    files.sort();
    files
}

/// The bytes of the `.rgba.gz` before compression: the tile itself, RGBA8, rows in order (the engine uploads
/// them as they are; the tile has no north). The gzip (level 9) is a later CLI job.
pub fn bake_rgba(tile: &[u8]) -> Vec<u8> {
    tile.to_vec()
}

/// The preview image of a `size` tile, 8-bit grey, `(4 size) x (4 size)` row-major: one band of `size` rows a
/// channel in [`CHANNELS`] order, each band `[the channel's rank | the coverage mask at 0.3 | 0.5 | 0.7, as 0
/// or 255]` side by side (judge the tile through that threshold, not as grey).
pub fn bake_preview(tile: &[u8], size: usize) -> Vec<u8> {
    assert_eq!(tile.len(), size * size * 4, "bake_preview: the tile must be size x size x 4");
    let w = 4 * size;
    let mut out = vec![0u8; 4 * size * w];
    for c in 0..4 {
        let ch: Vec<u8> = (0..size * size).map(|i| tile[i * 4 + c]).collect();
        let masks = coverage_masks(&ch, &DEFAULT_COVERAGES);
        for r in 0..size {
            let row = &mut out[(c * size + r) * w..(c * size + r + 1) * w];
            row[..size].copy_from_slice(&ch[r * size..(r + 1) * size]);
            for (k, m) in masks.iter().enumerate() {
                for col in 0..size {
                    row[(k + 1) * size + col] = if m[r * size + col] { 255 } else { 0 };
                }
            }
        }
    }
    out
}

fn json_string(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The sidecar `.json` of a bake, as the script writes it (`json.dumps(meta, indent=2)`, keys in insertion
/// order, `\n` line ends): the channels, the sizes, the seed, the encoding and where it lives on the CDN.
pub fn bake_sidecar_json(args: &BakeArgs) -> String {
    let stem = bake_stem(args.size, &args.version);
    let list = |name: &str, items: Vec<String>| {
        let body: Vec<String> = items.iter().map(|i| format!("    {i}")).collect();
        format!("  \"{name}\": [\n{}\n  ]", body.join(",\n"))
    };
    let pair = |name: &str, p: (f64, f64)| list(name, vec![format!("{:?}", p.0), format!("{:?}", p.1)]);
    let fields = [
        format!("  \"generatedBy\": {}", json_string(BAKE_GENERATED_BY)),
        format!("  \"version\": {}", json_string(&args.version)),
        format!("  \"size\": {}", args.size),
        format!("  \"seed\": {}", args.seed),
        list("channels", CHANNELS.iter().map(|c| json_string(c)).collect()),
        format!(
            "  \"encoding\": {}",
            json_string(
                "RGBA8, each channel a rank: byte = floor(256 rank / texels); a renderer maps it to its normal score to mix \
                 regimes without changing the distribution"
            )
        ),
        format!("  \"referenceTileKm\": {}", REFERENCE_TILE_KM as i64),
        format!(
            "  \"heaps\": {{\n    \"clusters\": {HEAP_CLUSTERS},\n    \"cells\": {HEAP_CELLS},\n    \"puffs\": {PUFFS}\n  }}"
        ),
        pair("closedCellKm", CLOSED_CELL_KM),
        pair("openCellKm", OPEN_CELL_KM),
        format!("  \"openWallLumpKm\": {:?}", LUMP_KM),
        format!("  \"sheet\": {{\n    \"low\": {SHEET_LOW:?},\n    \"slope\": {SHEET_SLOPE:?}\n  }}"),
        format!("  \"cdn\": {}", json_string(&format!("clouds/tiles/{}/{stem}.rgba.gz", args.version))),
    ];
    format!("{{\n{}\n}}", fields.join(",\n"))
}

// ---------------------------------------------------------------------------------------------------------
// Unit tests: what the golden cases do not pin (the golden cases themselves are tests/golden_tiles.rs)

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    // The reference tables below come from the Python of the project (numpy 2.5.4, scipy 1.18.1, Python 3.12):
    // the golden cases pin these functions only to a tolerance, these tests pin them to the bit.

    /// `(x, scipy.special.ndtr(x), scipy.special.erf(x), scipy.special.erfc(x))`: every branch of the Cephes
    /// routines (`|x| < 1`, `1 <= |x| < 8`, `|x| >= 8`, the underflow at 26.6, both signs).
    const SCIPY: &[(f64, f64, f64, f64)] = &[
        (0.0, 0.5, 0.0, 1.0),
        (1e-17, 0.5, 1.1283791670955127e-17, 1.0),
        (0.5, 0.6914624612740131, 0.5204998778130465, 0.4795001221869535),
        (-0.5, 0.3085375387259869, -0.5204998778130465, 1.5204998778130465),
        (0.99999, 0.8413423263491993, 0.8426966418332298, 0.1573033581667702),
        (-0.99999, 0.1586576736508008, -0.8426966418332298, 1.8426966418332298),
        (1.0, 0.8413447460685429, 0.8427007929497148, 0.15729920705028516),
        (-1.0, 0.15865525393145707, -0.8427007929497148, 1.8427007929497148),
        (1.0000001, 0.8413447702656143, 0.8427008344604605, 0.15729916553953954),
        (1.5, 0.9331927987311419, 0.9661051464753108, 0.03389485352468927),
        (-1.5, 0.06680720126885807, -0.9661051464753108, 1.9661051464753108),
        (2.0, 0.9772498680518208, 0.9953222650189527, 0.004677734981047266),
        (3.0, 0.9986501019683699, 0.9999779095030014, 2.2090496998585445e-05),
        (-3.0, 0.0013498980316300933, -0.9999779095030014, 1.9999779095030015),
        (5.9, 0.9999999981824922, 0.9999999999999999, 7.190409783550493e-17),
        (-5.9, 1.8175078630994235e-09, -0.9999999999999999, 2.0),
        (7.99999, 0.9999999999999993, 1.0, 1.1226107024563904e-29),
        (8.0, 0.9999999999999993, 1.0, 1.1224297172982928e-29),
        (8.0000001, 0.9999999999999993, 1.0, 1.1224279075929497e-29),
        (10.0, 1.0, 1.0, 2.0884875837625446e-45),
        (-10.0, 7.61985302416047e-24, -1.0, 2.0),
        (11.3, 1.0, 1.0, 1.7442148479967262e-57),
        (20.0, 1.0, 1.0, 5.395865611607901e-176),
        (-20.0, 2.7536241186061556e-89, -1.0, 2.0),
        (26.5, 1.0, 1.0, 2.2109076642637343e-307),
        (26.6, 1.0, 1.0, 1.088512588544247e-309),
        (26.64, 1.0, 1.0, 1.2921361031683e-310),
        (26.65, 1.0, 1.0, 0.0),
        (27.0, 1.0, 1.0, 0.0),
        (36.0, 1.0, 1.0, 0.0),
        (-36.0, 4.182624065796808e-284, -1.0, 2.0),
        (37.5, 1.0, 1.0, 0.0),
        (-37.5, 4.605353009581954e-308, -1.0, 2.0),
        (38.0, 1.0, 1.0, 0.0),
        (-38.0, 0.0, -1.0, 2.0),
        (0.3, 0.6179114221889526, 0.3286267594591274, 0.6713732405408726),
        (1.7359714287628147, 0.9587155567141258, 0.9859128868698684, 0.014087113130131518),
        (-1.541933002560139, 0.06154493757377204, -0.9707886552470426, 1.9707886552470426),
        (0.7071067811865475, 0.7602499389065233, 0.6826894921370859, 0.31731050786291415),
        (0.7071067811865476, 0.7602499389065234, 0.6826894921370859, 0.31731050786291415),
        (-0.7071067811865476, 0.23975006109347669, -0.6826894921370859, 1.6826894921370859),
        (2.5, 0.9937903346742238, 0.999593047982555, 0.00040695201744495886),
        (4.4, 0.9999945874560923, 0.999999999510829, 4.891710270605874e-10),
        (6.6, 0.9999999999794421, 1.0, 1.0213251678575794e-20),
        (7.7, 0.9999999999999932, 1.0, 1.29427400677171e-27),
        (9.9, 1.0, 1.0, 1.5431200214053102e-44),
        (12.5, 1.0, 1.0, 6.231942781979911e-70),
        (15.0, 1.0, 1.0, 7.212994172451209e-100),
        (18.2, 1.0, 1.0, 4.315154869899458e-146),
        (22.2, 1.0, 1.0, 2.3277703713110373e-216),
    ];

    #[test]
    fn cephes_erf_erfc_ndtr_are_scipys_to_the_bit() {
        for &(x, want_ndtr, want_erf, want_erfc) in SCIPY {
            assert_eq!(ndtr(x).to_bits(), want_ndtr.to_bits(), "ndtr({x:e}) = {:e}, scipy {want_ndtr:e}", ndtr(x));
            assert_eq!(erf(x).to_bits(), want_erf.to_bits(), "erf({x:e}) = {:e}, scipy {want_erf:e}", erf(x));
            assert_eq!(erfc(x).to_bits(), want_erfc.to_bits(), "erfc({x:e}) = {:e}, scipy {want_erfc:e}", erfc(x));
        }
    }

    /// The wrapping sum of the bit patterns of scipy's `ndtr`, `erf` and `erfc` over the grid
    /// `x_k = (k - 3000) * 0.0137`, `k = 0..=6000` (-41.1 to 41.1: every branch, both tails, the underflow): one
    /// float that differs in one bit anywhere changes the sum, which pins every coefficient of the rational
    /// approximations, not only the 50 points of the table above.
    const SCIPY_GRID_CHECKSUMS: [u64; 3] = [10225367035144702481, 8872995275754321520, 13627052907383879379];

    #[test]
    fn cephes_routines_match_scipy_over_a_dense_grid() {
        let mut sums = [0_u64; 3];
        for k in 0..=6000 {
            let x = (k as f64 - 3000.0) * 0.0137;
            for (sum, v) in sums.iter_mut().zip([ndtr(x), erf(x), erfc(x)]) {
                *sum = sum.wrapping_add(v.to_bits());
            }
        }
        assert_eq!(sums, SCIPY_GRID_CHECKSUMS, "ndtr, erf, erfc over the grid");
    }

    #[test]
    fn erf_erfc_ndtr_edges() {
        // Reference values of the true functions (high precision), to 1e-13 relative: the table above pins
        // scipy's own rounding, this pins that the routines are the functions they claim to be.
        let erfc_ref = [
            (0.5, 0.479_500_122_186_953_5),
            (1.0, 0.157_299_207_050_285_13),
            (2.0, 0.004_677_734_981_047_266),
            (5.0, 1.537_459_794_428_035e-12),
            (10.0, 2.088_487_583_762_545e-45),
        ];
        for (x, want) in erfc_ref {
            assert!(close(erfc(x), want, 1e-13), "erfc({x}) = {:e}, want {want:e}", erfc(x));
            assert!(close(erfc(-x), 2.0 - want, 1e-15));
            assert!(close(erf(x), 1.0 - want, 1e-15));
        }
        assert_eq!(erfc(30.0), 0.0);
        assert_eq!(erfc(-30.0), 2.0);
        assert_eq!((erf(0.0), erf(f64::INFINITY), erf(f64::NEG_INFINITY)), (0.0, 1.0, -1.0));
        assert_eq!((erfc(f64::INFINITY), erfc(f64::NEG_INFINITY)), (0.0, 2.0));
        assert!(erf(f64::NAN).is_nan() && erfc(f64::NAN).is_nan() && ndtr(f64::NAN).is_nan());
        assert_eq!((ndtr(f64::INFINITY), ndtr(f64::NEG_INFINITY)), (1.0, 0.0));
    }

    #[test]
    fn ndtr_is_a_cdf() {
        assert_eq!(ndtr(0.0), 0.5);
        assert!(close(ndtr(1.959_963_984_540_054), 0.975, 1e-12));
        assert!(close(ndtr(-1.959_963_984_540_054), 0.025, 1e-12));
        for x in [0.1, 0.7, 1.3, 2.9, 5.5] {
            assert!((ndtr(x) + ndtr(-x) - 1.0).abs() < 2e-16, "symmetry at {x}");
        }
        assert!(ndtr(-37.0) > 5e-300 && ndtr(-37.0) < 6e-300);
        assert_eq!(ndtr(40.0), 1.0);
        // Monotone across every switch of the Cephes branches (a = 1, x = 1, x = 8).
        for (start, step, count) in [(0.5, 0.001, 1000), (1.0, 0.001, 1000), (11.0, 0.003, 400)] {
            let mut prev = ndtr(start);
            for i in 1..count {
                let v = ndtr(start + i as f64 * step);
                assert!(v >= prev, "ndtr decreases at {}", start + i as f64 * step);
                prev = v;
            }
        }
    }

    /// `(x, e, np.array([x]) ** e)` for exponents where numpy's fast path (reciprocal, sqrt, square) and the
    /// libm `pow` give different floats.
    const NP_POWER: &[(f64, f64, f64)] = &[
        (91.84772494597141, -1.0, 0.010887585953688465),
        (1.6041097352226956e-07, -1.0, 6233987.476306737),
        (0.00014417110005178311, -1.0, 6936.202884217585),
        (4943.495498484727, -1.0, 0.00020228601407779547),
        (2.1028376530663416e-07, -1.0, 4755478.857541893),
        (3.7175357983729573, 0.5, 1.9280912318593633),
        (0.0001304971971902304, 0.5, 0.01142353698248622),
        (5.681457971879326, 0.5, 2.3835809136421875),
        (3.3269848300132297e-07, 0.5, 0.000576800210646046),
        (598766.1892016464, 0.5, 773.7998379436676),
        (2.094752966939616e-08, 2.0, 4.3879899925023246e-16),
        (43144.423910673584, 2.0, 1861441314.5839024),
        (3.531159648033824e-08, 2.0, 1.2469088459902358e-15),
        (4598.941798898748, 2.0, 21150265.669658054),
        (0.03794225468138954, 2.0, 0.0014396146903074263),
    ];

    #[test]
    fn np_power_takes_numpys_fast_paths() {
        for &(x, e, want) in NP_POWER {
            assert_eq!(np_power(x, e).to_bits(), want.to_bits(), "{x:e} ** {e}");
        }
        let mut rng = Rng::new(4);
        for _ in 0..20_000 {
            let x = rng.uniform_range(-20.0, 20.0).exp();
            assert_eq!(np_power(x, -1.0).to_bits(), (1.0 / x).to_bits());
            assert_eq!(np_power(x, 0.5).to_bits(), x.sqrt().to_bits());
            assert_eq!(np_power(x, 2.0).to_bits(), (x * x).to_bits());
            assert_eq!(np_power(x, 1.0).to_bits(), x.to_bits());
            assert_eq!(np_power(x, 0.0), 1.0);
            assert_eq!(np_power(x, -0.0), 1.0);
            // Any other exponent is the libm pow.
            assert_eq!(np_power(x, -1.5).to_bits(), x.powf(-1.5).to_bits());
            assert_eq!(np_power(x, 3.0).to_bits(), x.powf(3.0).to_bits());
        }
        assert_eq!(np_power(f64::NAN, 0.0), 1.0); // numpy's ones_like
        assert!(np_power(f64::NAN, 1.0).is_nan() && np_power(f64::NAN, -1.0).is_nan());
    }

    #[test]
    fn fbm_amplitude_is_the_filtered_spectrum() {
        let n = 16;
        let k = fft_k(n);
        let radius = |r: usize, c: usize| k[c].hypot(k[r]);
        // low = 5 on size 16: the ring of radius exactly 5 is inside (`kk >= low`), radius 4.47 is out.
        let amp = fbm_amplitude(n, 5.0, 3.0);
        for r in 0..n {
            for c in 0..n {
                let kk = radius(r, c);
                let want = if kk >= 5.0 { kk.powf(-1.5) } else { 0.0 };
                assert_eq!(amp[r * n + c].to_bits(), want.to_bits(), "mode ({r}, {c}) radius {kk}");
            }
        }
        assert!(amp[5] > 0.0 && amp[3 * n + 4] > 0.0 && amp[4 * n + 3] > 0.0, "radius 5 passes");
        assert_eq!(amp[4], 0.0);
        // numpy's fast paths of `kk ** (-slope / 2)`: slope 2 is a reciprocal, -1 a square root, -4 a square.
        for (slope, f) in [(2.0, (|kk: f64| 1.0 / kk) as fn(f64) -> f64), (-1.0, f64::sqrt), (-4.0, |kk| kk * kk), (-2.0, |kk| kk), (0.0, |_| 1.0)] {
            let amp = fbm_amplitude(n, 1.0, slope);
            for r in 0..n {
                for c in 0..n {
                    let kk = radius(r, c);
                    if kk >= 1.0 {
                        assert_eq!(amp[r * n + c].to_bits(), f(kk).to_bits(), "slope {slope} radius {kk}");
                    }
                }
            }
        }
        // The DC mode is kept by low <= 0, with the radius floored at 1e-9 (a huge amplitude for a falling spectrum).
        let dc = fbm_amplitude(n, 0.0, 3.0)[0];
        assert!(close(dc, (1e-9_f64).powf(-1.5), 1e-14), "{dc:e}");
        assert_eq!(fbm_amplitude(n, 0.0, 0.0)[0], 1.0);
        assert_eq!(fbm_amplitude(n, 0.0, -2.0)[0], 1e-9);
        assert_eq!(fbm_amplitude(n, 0.5, 3.0)[0], 0.0, "the DC mode is out when low > 0");
    }

    /// `(n, sum, mean, std)` of `a[i] = ((i * 7919 + 13) % 1009) / 7 - 60` as numpy computes them (its pairwise
    /// summation, which differs from a left-to-right sum from n = 8 on).
    const NP_SUMS: &[(usize, f64, f64, f64)] = &[
        (1, -58.142857142857146, -58.142857142857146, 0.0),
        (2, 5.999999999999993, 2.9999999999999964, 61.14285714285714),
        (3, 48.285714285714285, 16.095238095238095, 53.24725195979452),
        (4, 68.71428571428572, 17.17857142857143, 46.15163284528202),
        (5, 67.28571428571429, 13.457142857142859, 41.94490068980437),
        (6, 44.00000000000001, 7.333333333333335, 40.66510535644527),
        (7, -1.1428571428571317, -0.16326530612244738, 41.88805215037978),
        (8, 76.0, 9.5, 46.78604143833273),
        (9, 131.28571428571428, 14.587301587301587, 46.39788702498173),
        (10, 164.71428571428572, 16.47142857142857, 44.37833862117317),
        (11, 176.28571428571428, 16.025974025974026, 42.33652695473859),
        (15, 148.14285714285714, 9.876190476190477, 44.56916559171767),
        (16, 194.5714285714286, 12.160714285714288, 44.05162425484213),
        (17, 219.14285714285717, 12.89075630252101, 42.83600401396224),
        (31, 430.2857142857142, 13.880184331797233, 41.69049723785612),
        (64, 790.9999999999999, 12.359374999999998, 41.65192250159364),
        (100, 1018.8571428571429, 10.188571428571429, 41.98444259970785),
        (127, 1374.857142857143, 10.825646794150732, 41.95144195048025),
        (128, 1423.7142857142858, 11.122767857142858, 41.92118506357955),
        (129, 1450.7142857142856, 11.245847176079733, 41.78159429857523),
        (130, 1455.8571428571427, 11.198901098901098, 41.624000956435),
        (135, 1586.1428571428569, 11.749206349206347, 41.90837093645785),
        (255, 3089.285714285714, 12.11484593837535, 41.651088112348894),
        (256, 3079.1428571428573, 12.027901785714286, 41.59283766419095),
        (257, 3047.1428571428573, 11.856586992773764, 41.602236106771166),
        (300, 3624.7142857142853, 12.082380952380952, 41.72525723024012),
        (1000, 12080.142857142857, 12.080142857142857, 41.62670852753956),
        (4096, 49066.71428571428, 11.979178292410714, 41.61581569550322),
        (10007, 120053.57142857142, 11.996959271367185, 41.614596579728506),
    ];

    #[test]
    fn pairwise_sum_mean_and_std_are_numpys_to_the_bit() {
        let mut differs_from_naive = 0;
        for &(n, sum, mean, std) in NP_SUMS {
            let a: Vec<f64> = (0..n).map(|i| ((i * 7919 + 13) % 1009) as f64 / 7.0 - 60.0).collect();
            assert_eq!(pairwise_sum(&a).to_bits(), sum.to_bits(), "sum of {n}");
            let (m, s) = mean_std(&a);
            assert_eq!(m.to_bits(), mean.to_bits(), "mean of {n}");
            assert_eq!(s.to_bits(), std.to_bits(), "std of {n}");
            let naive = a.iter().fold(0.0, |acc, &v| acc + v);
            differs_from_naive += usize::from(naive.to_bits() != sum.to_bits());
        }
        // The table is a test of the summation order only if the order matters in it.
        assert!(differs_from_naive >= 5, "{differs_from_naive}");
    }

    /// `(a, b, x, _smoothstep(a, b, x))` on float32 arrays: the float32 denominator is `float32(b - a)` with
    /// `b - a` formed in float64 first (`(0.1, 0.3)` gives 0.2, not `float32(0.3) - float32(0.1)`).
    const SMOOTHSTEP_F32: &[(f64, f64, f32, f32)] = &[
        (0.1, 0.3, 0.1284836381673813_f32, 0.05507149547338486_f32),
        (0.1, 0.3, 0.13954734802246094_f32, 0.10183649510145187_f32),
        (0.1, 0.3, 0.2942677140235901_f32, 0.9975826740264893_f32),
        (0.1, 0.3, 0.07757478207349777_f32, 0.0_f32),
        (0.1, 0.3, 0.23003016412258148_f32, 0.7184557914733887_f32),
        (0.1, 0.3, 0.26856815814971924_f32, 0.9336663484573364_f32),
        (0.7, 0.9, 0.7063702940940857_f32, 0.0029789318796247244_f32),
        (0.7, 0.9, 0.6665439605712891_f32, 0.0_f32),
        (0.7, 0.9, 0.7324908375740051_f32, 0.07059936225414276_f32),
        (0.7, 0.9, 0.8472298979759216_f32, 0.8278857469558716_f32),
        (0.7, 0.9, 0.818679690361023_f32, 0.638468325138092_f32),
        (0.7, 0.9, 0.6950187087059021_f32, 0.0_f32),
        (0.25, 0.65, 0.4163154065608978_f32, 0.37487712502479553_f32),
        (0.25, 0.65, 0.5346486568450928_f32, 0.7984780669212341_f32),
        (0.25, 0.65, 0.41139233112335205_f32, 0.3570196032524109_f32),
        (0.25, 0.65, 0.5165922045707703_f32, 0.7404924631118774_f32),
        (0.25, 0.65, 0.6837179660797119_f32, 1.0_f32),
        (0.25, 0.65, 0.5415323972702026_f32, 0.8192816376686096_f32),
    ];

    #[test]
    fn smoothstep_float32_forms_the_denominator_in_float64() {
        for &(a, b, x, want) in SMOOTHSTEP_F32 {
            let got = smoothstep::<f32>(a, b, &[x])[0];
            assert_eq!(got.to_bits(), want.to_bits(), "smoothstep({a}, {b}, {x})");
        }
    }

    fn brute<const K: usize>(pts: &[[f64; 2]], q: [f64; 2]) -> Vec<(f64, usize)> {
        let mut all: Vec<(f64, usize)> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let dx = (p[0] - q[0]).abs();
                let dy = (p[1] - q[1]).abs();
                let (dx, dy) = (dx.min(1.0 - dx), dy.min(1.0 - dy));
                ((dx * dx + dy * dy).sqrt(), i)
            })
            .collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        all.truncate(K);
        all
    }

    #[test]
    fn periodic_search_equals_brute_force() {
        let mut rng = Rng::new(5);
        for n in [1usize, 2, 3, 4, 5, 8, 17, 100, 600] {
            let pts: Vec<[f64; 2]> = (0..n).map(|_| [rng.uniform(), rng.uniform()]).collect();
            let tree = PeriodicPoints::new(&pts);
            assert_eq!(tree.len(), n);
            for _ in 0..300 {
                // Queries anywhere, including on and beyond the tile edges.
                let q = [rng.uniform_range(-0.2, 1.2), rng.uniform_range(-0.2, 1.2)];
                let qw = [q[0] - q[0].floor(), q[1] - q[1].floor()];
                let got = tree.nearest::<2>(q);
                let want = brute::<2>(&pts, qw);
                for k in 0..2 {
                    if k < want.len() {
                        assert_eq!(got[k].1, want[k].1, "n {n} k {k} q {q:?}");
                        assert!((got[k].0 - want[k].0).abs() < 1e-15);
                    } else {
                        assert!(got[k].0.is_infinite() && got[k].1 == n, "missing neighbors are (inf, n)");
                    }
                }
                let got4 = tree.nearest::<4>(q);
                let want4 = brute::<4>(&pts, qw);
                for k in 0..want4.len() {
                    assert_eq!(got4[k].1, want4[k].1);
                }
            }
        }
    }

    #[test]
    fn periodic_search_wraps_across_the_edge_and_orders_ties_by_index() {
        let pts = [[0.02, 0.5], [0.98, 0.5], [0.5, 0.5]];
        let tree = PeriodicPoints::new(&pts);
        let nn = tree.nearest::<2>([0.0, 0.5]);
        assert_eq!(nn[0].1, 0); // 0.02 away, the image of 0.98 is also 0.02 away: equal, lower index first
        assert_eq!(nn[1].1, 1);
        assert!((nn[0].0 - 0.02).abs() < 1e-15 && (nn[1].0 - 0.02).abs() < 1e-15);
        // A query at exactly 1.0 (np.mod(-1e-17, 1.0) is 1.0) is the same point as 0.0.
        assert_eq!(tree.nearest::<1>([1.0, 0.5])[0].1, 0);
    }

    #[test]
    fn sample_truncates_and_wraps() {
        let field: Vec<f64> = (0..9).map(|v| v as f64).collect(); // 3 x 3, value row * 3 + col
        let at = |x: f64, y: f64| sample(&field, 3, 3, &[[x, y]])[0];
        assert_eq!(at(0.0, 0.0), 0.0);
        assert_eq!(at(0.999, 0.999), 8.0);
        assert_eq!(at(1.0, 0.0), 0.0); // wraps
        assert_eq!(at(-0.1, 0.0), 0.0); // trunc(-0.3) = 0, not -1
        assert_eq!(at(-0.4, 0.0), 2.0); // trunc(-1.2) = -1 -> 2
        assert_eq!(at(0.4, 0.7), 3.0 * 2.0 + 1.0);
    }

    #[test]
    fn plans_have_the_right_shape_and_range() {
        let mut rng = Rng::new(3);
        let w = WorleyPlan::draw(20, &mut rng);
        assert_eq!(w.jitter.len(), 800);
        // The jitter spans [0, 1): a lattice point can sit anywhere in its cell.
        let (lo, hi) = w.jitter.iter().fold((1.0_f64, 0.0_f64), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(lo < 0.01 && hi > 0.99 && hi < 1.0, "{lo} {hi}");
        assert!((mean_std(&w.jitter).0 - 0.5).abs() < 0.03);
        let f = FbmPlan::draw(7, &mut rng);
        assert_eq!(f.white.len(), 49);
        let big = FbmPlan::draw(40, &mut rng);
        let (bm, bs) = mean_std(&big.white);
        assert!(bm.abs() < 0.1 && (bs - 1.0).abs() < 0.1, "standard normals: {bm} {bs}");
        let l = LumpsPlan::draw(&mut rng);
        assert_eq!((l.g.len(), l.r.len()), (3600, 3600));
        assert_eq!(lump_freq(), 60);
        assert!(l.g.iter().all(|p| (0.0..1.0).contains(&p[0]) && (0.0..1.0).contains(&p[1])));
        assert!(l.r.iter().all(|&r| (LUMP_RADIUS.0 / 60.0..LUMP_RADIUS.1 / 60.0).contains(&r)));
        // The radii span the whole range (3600 draws: the extremes are within 0.002 of the bounds).
        let (rmin, rmax) = l.r.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), &r| (lo.min(r), hi.max(r)));
        assert!(rmin * 60.0 < LUMP_RADIUS.0 + 0.002 && rmax * 60.0 > LUMP_RADIUS.1 - 0.002, "{rmin} {rmax}");
        // Lump k sits in its own grid cell (row k / 60 is y, column k % 60 is x).
        assert!(l.g.iter().enumerate().all(|(k, p)| (p[0] * 60.0) as usize == k % 60 && (p[1] * 60.0) as usize == k / 60));
        let size = 16;
        let plan = RawChannelsPlan::draw(size, &mut rng);
        assert_ne!(plan.closed.voronoi.warp_x, plan.closed.voronoi.warp_y);
        assert_eq!(plan.closed.tops.freq, 64);
        assert_eq!((plan.clusters.freq, plan.cells.freq, plan.puffs.freq), (HEAP_CLUSTERS, HEAP_CELLS, PUFFS));
        // The Poisson candidate count is int(12 / s_min^2) of the size field the plan thinned by.
        let field = cell_size_field(size, CLOSED_CELL_KM, &plan.closed.cell_size);
        assert_eq!(plan.closed.voronoi.poisson.cand.len(), candidate_count(field_min(&field)));
        let field = cell_size_field(size, OPEN_CELL_KM, &plan.open.cell_size);
        assert_eq!(plan.open.voronoi.poisson.cand.len(), candidate_count(field_min(&field)));
        assert_eq!(plan.open.lumps.g.len(), 3600);
    }

    #[test]
    fn candidate_count_truncates() {
        assert_eq!(candidate_count(0.2), 299);
        assert_eq!(candidate_count(0.25), 192);
        assert_eq!(candidate_count(4.0), 0);
    }

    #[test]
    fn poisson_draw_counts_and_thins_like_python() {
        let mut rng = Rng::new(9);
        // 0.2 * 0.2 = 0.04000000000000001 and 12 / that = 299.99999999999994: 299 candidates, not 300.
        let field = vec![0.2; 16];
        let plan = PoissonPlan::draw(&field, 4, &mut rng);
        assert_eq!((plan.cand.len(), plan.keep.len()), (299, 299));
        assert!(plan.cand.iter().all(|p| (0.0..1.0).contains(&p[0]) && (0.0..1.0).contains(&p[1])));
        assert!(plan.keep.iter().all(|&k| k), "a constant field keeps every candidate");
        // Half of the field is twice as coarse: the thinning keeps (s_min / s)^2 = 1 or 0.25 of its candidates.
        let field = [0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5, 0.25, 0.5];
        let plan = PoissonPlan::draw(&field, 4, &mut rng);
        assert_eq!(plan.cand.len(), 192);
        let kept = plan.keep.iter().filter(|&&k| k).count() as f64;
        assert!((kept - 192.0 * 0.625).abs() < 25.0, "{kept} kept of 192, 0.625 expected");
        // Candidates in a fine column are all kept.
        for (p, &k) in plan.cand.iter().zip(&plan.keep) {
            if ((p[0] * 4.0) as usize) % 2 == 0 {
                assert!(k);
            }
        }
    }

    #[test]
    fn poisson_with_a_constant_field_is_a_poisson_disc() {
        let field = vec![0.25; 16];
        let mut rng = Rng::new(9);
        let plan = PoissonPlan::draw(&field, 4, &mut rng);
        assert_eq!(plan.cand.len(), 192);
        assert!(plan.keep.iter().all(|&k| k)); // s_min / s = 1: every candidate survives the thinning
        let (pts, index) = poisson_points_indexed(&field, 4, &plan, POISSON_SPACING);
        assert_eq!(pts.len(), index.len());
        assert!(pts.len() > 8);
        for (i, p) in pts.iter().enumerate() {
            for q in &pts[..i] {
                let dx = (p[0] - q[0]).abs();
                let dy = (p[1] - q[1]).abs();
                assert!(dx.min(1.0 - dx).hypot(dy.min(1.0 - dy)) >= POISSON_SPACING * 0.25);
            }
        }
    }

    #[test]
    fn periodic_fbm_has_unit_variance_and_is_periodic_in_spectrum() {
        let mut rng = Rng::new(11);
        let plan = FbmPlan::draw(32, &mut rng);
        let f = periodic_fbm(32, 2.0, 3.0, &plan);
        let (m, s) = mean_std(&f);
        assert!(m.abs() < 1e-12 && (s - 1.0).abs() < 1e-12, "{m} {s}");
        // Shifting the white noise by a whole row shifts the field by the same row (translation invariance).
        let mut shifted = vec![0.0; 32 * 32];
        for r in 0..32 {
            for c in 0..32 {
                shifted[((r + 1) % 32) * 32 + c] = plan.white[r * 32 + c];
            }
        }
        let g = periodic_fbm(32, 2.0, 3.0, &FbmPlan { white: shifted });
        for r in 0..32 {
            for c in 0..32 {
                assert!((g[((r + 1) % 32) * 32 + c] - f[r * 32 + c]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn worley_f1_is_zero_at_a_point_and_f2_not_below_f1() {
        let mut rng = Rng::new(2);
        let plan = WorleyPlan::draw(4, &mut rng);
        let (f1, f2) = tileable_worley(32, &plan);
        assert!(f1.iter().zip(&f2).all(|(a, b)| a <= b && *a >= 0.0));
        assert!(f1.iter().all(|&v| v < 1.5));
    }

    #[test]
    fn rank_bytes_of_nan_and_empty() {
        assert!(rank_bytes::<f64>(&[]).is_empty());
        // NaNs are one value sorted last: ranks 0, 1, then the three NaNs share mean rank 3.
        let r = rank_bytes(&[f64::NAN, 1.0, f64::NAN, 0.0, f64::NAN]);
        assert_eq!(r, vec![153, 51, 153, 0, 153]);
        // Infinities sort at the ends, -0.0 and 0.0 are one value (mean rank 1.5 of 4 -> 96).
        let r = rank_bytes(&[f64::INFINITY, 0.0, -0.0, f64::NEG_INFINITY]);
        assert_eq!(r, vec![192, 96, 96, 0]);
        let r = rank_bytes(&[f32::NAN, f32::NEG_INFINITY, 2.5, f32::NAN]);
        assert_eq!(r, vec![160, 0, 64, 160]); // np.unique collapses the NaNs: ranks 0, 1 and 2.5 for the pair
    }

    #[test]
    fn seeded_tile_is_deterministic_and_every_channel_a_uniform_rank() {
        let a = build_tile_seeded(32, 7);
        assert_eq!(a, build_tile_seeded(32, 7));
        assert_ne!(a, build_tile_seeded(32, 8));
        assert_eq!(a.len(), 32 * 32 * 4);
        for c in 0..4 {
            let mut hist = [0usize; 256];
            for i in 0..32 * 32 {
                hist[a[i * 4 + c] as usize] += 1;
            }
            // 1024 texels over 256 levels: 4 a level, ties (a plateau) aside.
            let spread = hist.iter().filter(|&&h| h > 0).count();
            assert!(spread >= 128, "channel {c} uses {spread} levels");
            assert!(a.chunks(4).map(|t| t[c] as usize).max().unwrap() >= 250, "channel {c} reaches the top");
        }
    }

    #[test]
    fn bake_args_parse_like_argparse() {
        let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(BakeArgs::parse(&[]).unwrap(), BakeArgs { version: "v2".into(), seed: 1, size: 1024 });
        let a = BakeArgs::parse(&v(&["--size=16", "--seed", "-3", "--version", "x", "--size", "32"])).unwrap();
        assert_eq!(a, BakeArgs { version: "x".into(), seed: -3, size: 32 });
        assert!(BakeArgs::parse(&v(&["--bogus", "1"])).is_err());
        assert!(BakeArgs::parse(&v(&["--size"])).is_err());
        assert!(BakeArgs::parse(&v(&["--size", "big"])).is_err());
    }

    /// argv -> what `argparse` of `bake_tiles.py` did with it (Python 3.12): `Ok((version, seed, size))`, or
    /// `Err((exit code, the text after `bake_tiles.py: error: `))`, `None` for the help.
    #[allow(clippy::type_complexity)]
    const ARGPARSE: &[(&[&str], Result<(&str, i64, usize), (i32, Option<&str>)>)] = &[
        (&[], Ok(("v2", 1, 1024))),
        (&["--size", "16"], Ok(("v2", 1, 16))),
        (&["--size=16"], Ok(("v2", 1, 16))),
        (&["--ver", "v3"], Ok(("v3", 1, 1024))),
        (&["--ver=v3"], Ok(("v3", 1, 1024))),
        (&["--v", "x"], Ok(("x", 1, 1024))),
        (&["--se", "5"], Ok(("v2", 5, 1024))),
        (&["--seed", "-3"], Ok(("v2", -3, 1024))),
        (&["--seed=-3"], Ok(("v2", -3, 1024))),
        (&["--s", "5"], Err((2, Some("ambiguous option: --s could match --seed, --size")))),
        (&["--s=5"], Err((2, Some("ambiguous option: --s=5 could match --seed, --size")))),
        (&["--si", "7"], Ok(("v2", 1, 7))),
        (&["--siz=7"], Ok(("v2", 1, 7))),
        (&["--size", "1_000"], Ok(("v2", 1, 1000))),
        (&["--size", "1__0"], Err((2, Some("argument --size: invalid int value: '1__0'")))),
        (&["--size", "_1"], Err((2, Some("argument --size: invalid int value: '_1'")))),
        (&["--size", "1_"], Err((2, Some("argument --size: invalid int value: '1_'")))),
        (&["--size", " 7 "], Ok(("v2", 1, 7))),
        (&["--size", "+5"], Ok(("v2", 1, 5))),
        (&["--size", "007"], Ok(("v2", 1, 7))),
        (&["--size", "0x10"], Err((2, Some("argument --size: invalid int value: '0x10'")))),
        (&["--size", "1.5"], Err((2, Some("argument --size: invalid int value: '1.5'")))),
        (&["--size", ""], Err((2, Some("argument --size: invalid int value: ''")))),
        (&["--size="], Err((2, Some("argument --size: invalid int value: ''")))),
        (&["--size", "-x"], Err((2, Some("argument --size: expected one argument")))),
        (&["--size", "--seed"], Err((2, Some("argument --size: expected one argument")))),
        (&["--size"], Err((2, Some("argument --size: expected one argument")))),
        (&["--bogus", "1"], Err((2, Some("unrecognized arguments: --bogus 1")))),
        (&["--bogus=1"], Err((2, Some("unrecognized arguments: --bogus=1")))),
        (&["x"], Err((2, Some("unrecognized arguments: x")))),
        (&["--size", "4", "x"], Err((2, Some("unrecognized arguments: x")))),
        (&["-h"], Err((0, None))),
        (&["--help"], Err((0, None))),
        (&["--h"], Err((0, None))),
        (&["--he"], Err((0, None))),
        (&["--help=1"], Err((2, Some("argument -h/--help: ignored explicit argument '1'")))),
        (&["--help="], Err((2, Some("argument -h/--help: ignored explicit argument ''")))),
        (&["-h="], Err((2, Some("argument -h/--help: ignored explicit argument ''")))),
        (&["-h=3"], Err((2, Some("argument -h/--help: ignored explicit argument '3'")))),
        (&["-hx"], Err((0, None))),
        (&["-hh"], Err((0, None))),
        (&["-hx=3"], Err((0, None))),
        (&["-h", "--bogus"], Err((0, None))),
        (&["--bogus", "-h"], Err((0, None))),
        (&["--size", "x", "-h"], Err((2, Some("argument --size: invalid int value: 'x'")))),
        (&["--version", "-h"], Err((2, Some("argument --version: expected one argument")))),
        (&["-h", "--s"], Err((0, None))),
        (&["--seed", "1", "--seed", "2"], Ok(("v2", 2, 1024))),
        (&["--version", "a b"], Ok(("a b", 1, 1024))),
        (&["--version", "-a b"], Ok(("-a b", 1, 1024))),
        (&["--version", "--"], Err((2, Some("argument --version: expected one argument")))),
        (&["--", "--size", "4"], Err((2, Some("unrecognized arguments: -- --size 4")))),
        (&["--size", "4", "--"], Err((2, Some("unrecognized arguments: --")))),
        (&["--seed", "5", "--", "6"], Err((2, Some("unrecognized arguments: -- 6")))),
        (&["--seed", "--", "5"], Err((2, Some("argument --seed: expected one argument")))),
        (&["--version=a=b"], Ok(("a=b", 1, 1024))),
        (&["--version=", "x"], Err((2, Some("unrecognized arguments: x")))),
        (&["--version", ""], Ok(("", 1, 1024))),
        (&["-s"], Err((2, Some("unrecognized arguments: -s")))),
        (&["-"], Err((2, Some("unrecognized arguments: -")))),
        (&["--size", "-"], Err((2, Some("argument --size: invalid int value: '-'")))),
        (&["--version", "-"], Ok(("-", 1, 1024))),
        (&["--seed", "-5", "--size", "-0"], Ok(("v2", -5, 0))),
        (&["--seed", " -7_0 "], Ok(("v2", -70, 1024))),
        (&["--seed", "-_7"], Err((2, Some("argument --seed: expected one argument")))),
        (&["--seed", "-5_5"], Err((2, Some("argument --seed: expected one argument")))),
        (&["--seed", "+-7"], Err((2, Some("argument --seed: invalid int value: '+-7'")))),
        (&["--seed", "\t12\n"], Ok(("v2", 12, 1024))),
        (&["--seed", "\u{b}5\u{c}"], Ok(("v2", 5, 1024))),
        (&["--seed", "\u{1c}5"], Err((2, Some("argument --seed: invalid int value: '\\x1c5'")))),
        (&["--seed", "5\u{1f}"], Err((2, Some("argument --seed: invalid int value: '5\\x1f'")))),
        (&["--seed", "1\u{3000}"], Ok(("v2", 1, 1024))),
        (&["--seed", "- 7"], Err((2, Some("argument --seed: invalid int value: '- 7'")))),
        (&["--seed", "1e3"], Err((2, Some("argument --seed: invalid int value: '1e3'")))),
        (&["--size", "4", "--size"], Err((2, Some("argument --size: expected one argument")))),
        (&["--versio=q"], Ok(("q", 1, 1024))),
        (&["--versions", "q"], Err((2, Some("unrecognized arguments: --versions q")))),
        (&["--version", "v", "--v=w"], Ok(("w", 1, 1024))),
        (&["--seed=1", "--seed=2", "--size=3"], Ok(("v2", 2, 3))),
        (&["--seed", "-1.5"], Err((2, Some("argument --seed: invalid int value: '-1.5'")))),
        (&["--seed", "-.5"], Err((2, Some("argument --seed: invalid int value: '-.5'")))),
        (&["--seed", "--5"], Err((2, Some("argument --seed: expected one argument")))),
        (&["---size", "3"], Err((2, Some("unrecognized arguments: ---size 3")))),
        (&["--size=3", "junk"], Err((2, Some("unrecognized arguments: junk")))),
        (&["--h=3"], Err((2, Some("argument -h/--help: ignored explicit argument '3'")))),
        (&["--sizes=3"], Err((2, Some("unrecognized arguments: --sizes=3")))),
        (&["--si=3", "-h"], Err((0, None))),
        (&["--seed", "0b11"], Err((2, Some("argument --seed: invalid int value: '0b11'")))),
        (&["--seed", "00"], Ok(("v2", 0, 1024))),
        (&["--seed", "-0"], Ok(("v2", 0, 1024))),
        (&["--seed=-1_0"], Ok(("v2", -10, 1024))),
        (&["--si", "3", "--se"], Err((2, Some("argument --seed: expected one argument")))),
        (&["--size=3", "--seed=4", "--version=v7", "--ver=v8"], Ok(("v8", 4, 3))),
        (&["--size", " 3"], Ok(("v2", 1, 3))),
        (&["--seed", "--size", "3"], Err((2, Some("argument --seed: expected one argument")))),
        (&["--seed=--size", "3"], Err((2, Some("argument --seed: invalid int value: '--size'")))),
        (&["--version", "x", "--", "--size"], Err((2, Some("unrecognized arguments: -- --size")))),
        (&["x", "-h"], Err((0, None))),
        (&["--bogus", "-3"], Err((2, Some("unrecognized arguments: --bogus -3")))),
        (&["-3", "--bogus"], Err((2, Some("unrecognized arguments: -3 --bogus")))),
        (&["--seed", "-3", "-4"], Err((2, Some("unrecognized arguments: -4")))),
        (&["--SEED", "3"], Err((2, Some("unrecognized arguments: --SEED 3")))),
        (&["-H"], Err((2, Some("unrecognized arguments: -H")))),
        (&["--version", "it's"], Ok(("it's", 1, 1024))),
        (&["--seed", "it's"], Err((2, Some("argument --seed: invalid int value: \"it's\"")))),
        (&["--seed", "a\"b"], Err((2, Some("argument --seed: invalid int value: 'a\"b'")))),
        (&["--seed", "a'b\"c"], Err((2, Some("argument --seed: invalid int value: 'a\\'b\"c'")))),
        (&["--seed", "caf\u{e9}"], Err((2, Some("argument --seed: invalid int value: 'caf\u{e9}'")))),
        (&["--seed", "tab\there"], Err((2, Some("argument --seed: invalid int value: 'tab\\there'")))),
        (&["--seed", "x\u{a0}y"], Err((2, Some("argument --seed: invalid int value: 'x\\xa0y'")))),
    ];

    #[test]
    fn bake_args_are_argparses_on_a_table_of_argument_lists() {
        for (argv, want) in ARGPARSE.iter() {
            let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
            let got = BakeArgs::parse(&argv);
            match (got, *want) {
                (Ok(a), Ok((version, seed, size))) => {
                    assert_eq!((a.version.as_str(), a.seed, a.size), (version, seed, size), "{argv:?}")
                }
                (Err(e), Err((0, None))) => assert_eq!((e.code, e.message.as_str()), (0, BAKE_HELP), "{argv:?}"),
                (Err(e), Err((2, Some(text)))) => {
                    assert_eq!(e.code, 2, "{argv:?}");
                    assert_eq!(e.message, format!("{BAKE_USAGE}bake_tiles.py: error: {text}\n"), "{argv:?}");
                }
                (got, want) => panic!("{argv:?}: got {got:?}, Python gave {want:?}"),
            }
        }
    }

    #[test]
    fn python_int_and_repr_forms() {
        for (text, want) in [("12", 12), ("  -7 ", -7), ("+5", 5), ("1_000", 1000), ("0_0", 0), ("007", 7), ("-0", 0), ("\t3\n", 3)] {
            assert_eq!(py_int(text), Ok(want), "{text:?}");
        }
        for text in ["", " ", "-", "+", "1__0", "_1", "1_", "- 7", "+-7", "1.5", "1e3", "0x10", "0b1", "٣", "\x1c5", "5\x1f"] {
            assert_eq!(py_int(text), Err(IntError::Invalid), "{text:?}");
        }
        assert_eq!(py_int("9223372036854775807"), Ok(i64::MAX));
        assert_eq!(py_int("-9223372036854775808"), Ok(i64::MIN));
        assert_eq!(py_int("9223372036854775808"), Err(IntError::OutOfRange));
        assert_eq!(py_int("99999999999999999999999999999999999999999999"), Err(IntError::OutOfRange));
        assert_eq!(py_repr("abc"), "'abc'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(py_repr("tab\there\n\\"), "'tab\\there\\n\\\\'");
        assert_eq!(py_repr("\x1c5\u{a0}\u{3000}"), "'\\x1c5\\xa0\\u3000'");
        assert_eq!(py_repr("caf\u{e9}"), "'caf\u{e9}'");
        assert_eq!(py_repr("\u{1f600}"), "'\u{1f600}'");
    }

    #[test]
    fn what_python_accepts_and_the_port_cannot_hold_is_an_error_at_the_end() {
        let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let err = |a: &[&str]| BakeArgs::parse(&v(a)).unwrap_err();
        // Alone: the port's own error (Python builds a tile of -1 texels, or seeds with a 20-digit integer).
        assert!(err(&["--size=-1"]).message.ends_with("is not a size this port can build\n"));
        assert!(err(&["--seed", "99999999999999999999"]).message.contains("does not fit this port's 64-bit seed"));
        // A later flag replaces it, as in Python.
        assert_eq!(BakeArgs::parse(&v(&["--size=-1", "--size", "8"])).unwrap().size, 8);
        assert_eq!(BakeArgs::parse(&v(&["--seed=-99999999999999999999", "--seed=3"])).unwrap().seed, 3);
        // Everything Python reports first still comes first.
        assert_eq!(err(&["--size=-1", "-h"]).code, 0);
        assert!(err(&["--size=-1", "--seed", "x"]).message.ends_with("invalid int value: 'x'\n"));
        assert!(err(&["--size=-1", "extra"]).message.ends_with("unrecognized arguments: extra\n"));
        // The limits of the types themselves.
        assert_eq!(BakeArgs::parse(&v(&["--seed", "-9223372036854775808"])).unwrap().seed, i64::MIN);
        assert_eq!(BakeArgs::parse(&v(&["--size", "0"])).unwrap().size, 0);
        assert_eq!(BakeArgs::parse(&v(&["--size", "-0"])).unwrap().size, 0);
    }

    #[test]
    fn help_tail_follows_consume_optional() {
        for tail in ["x", "h", "hh", "hx", "x=3", "xyz", "h3"] {
            assert_eq!(help_tail(tail), Ok(()), "-h{tail}");
        }
        assert_eq!(help_tail("-x"), Err("-x".to_string()));
        assert_eq!(help_tail("h-x"), Err("-x".to_string()));
        assert_eq!(help_tail("h="), Err(String::new()));
        assert_eq!(help_tail("h=3"), Err("3".to_string()));
        assert_eq!(help_tail("hh=3"), Err("3".to_string()));
    }

    #[test]
    fn sidecar_text_is_the_scripts_indent_2_layout() {
        let args = BakeArgs { version: "v2".into(), seed: 1, size: 1024 };
        let want = "{\n  \"generatedBy\": \"planet-cloudgen bake_tiles.py\",\n  \"version\": \"v2\",\n  \"size\": 1024,\n  \"seed\": 1,\n  \"channels\": [\n    \"heaps\",\n    \"closed\",\n    \"open\",\n    \"sheet\"\n  ],\n  \"encoding\": \"RGBA8, each channel a rank: byte = floor(256 rank / texels); a renderer maps it to its normal score to mix regimes without changing the distribution\",\n  \"referenceTileKm\": 240,\n  \"heaps\": {\n    \"clusters\": 8,\n    \"cells\": 32,\n    \"puffs\": 96\n  },\n  \"closedCellKm\": [\n    8.0,\n    32.0\n  ],\n  \"openCellKm\": [\n    12.0,\n    48.0\n  ],\n  \"openWallLumpKm\": 4.0,\n  \"sheet\": {\n    \"low\": 2.0,\n    \"slope\": 3.0\n  },\n  \"cdn\": \"clouds/tiles/v2/cloud_cell_tiles_1024_v2.rgba.gz\"\n}";
        assert_eq!(bake_sidecar_json(&args), want);
        assert_eq!(json_string("a\"b\\c\u{e9}"), "\"a\\\"b\\\\c\\u00e9\"");
    }

    #[test]
    fn bake_files_are_sorted_and_the_preview_has_four_bands() {
        assert_eq!(
            bake_files("v2", 16),
            vec![
                "output/tiles/v2/cloud_cell_tiles_16_v2.json",
                "output/tiles/v2/cloud_cell_tiles_16_v2.rgba.gz",
                "output/tiles/v2/cloud_cell_tiles_16_v2_preview.png"
            ]
        );
        let tile: Vec<u8> = (0..4 * 4 * 4).map(|i| (i * 5 % 256) as u8).collect();
        let p = bake_preview(&tile, 4);
        assert_eq!(p.len(), 16 * 16);
        // Band 2 (open), row 1: the channel's own bytes first.
        let row = &p[(2 * 4 + 1) * 16..(2 * 4 + 2) * 16];
        let want: Vec<u8> = (0..4).map(|c| tile[(4 + c) * 4 + 2]).collect();
        assert_eq!(&row[..4], &want[..]);
        assert!(row[4..].iter().all(|&v| v == 0 || v == 255));
        assert_eq!(bake_rgba(&tile), tile);
    }
}
