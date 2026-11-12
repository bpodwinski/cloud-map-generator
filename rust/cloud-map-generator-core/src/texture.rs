//! Port of `cloudgen/texture.py`: the regime-driven fine texture, the small-scale cloud pattern chosen by the
//! local weather regime instead of one noise everywhere.
//!
//! The decision records (why the small scale is organized by regime, the mesoscale cell sizes, the
//! concessions) are in the Python module docstring and are not repeated here. In short: closed cells under
//! the subtropical subsidence inversion, open cells in cold-air outbreaks, cloud streets along the low-level
//! wind, an fBm elsewhere; cell sizes are physical (km), not a fraction of the planet, and a size finer than
//! the map resolves is dropped rather than aliased.
//!
//! Precision follows numpy 2 (NEP 50): a Python float argument is a WEAK scalar, so an array expression runs
//! in the dtype of its arrays. The functions that numpy runs in the dtype of their input are generic over
//! `f32`/`f64` and run in that precision; where two arrays of different precision meet (a float64 latitude
//! and a float32 back-traced one) the result is the wider one and the narrower operand is widened first
//! (exactly), which [`regime_weights`] spells out with its `L`, `B` and `P` parameters.
//!
//! Random draws are inputs (README rule 3): [`regime_texture`] takes the noise plans in a
//! [`RegimeTexturePlans`] (the two fBm plans and one cell-edge plan per USED cell diameter) and
//! [`RegimeTexturePlans::from_seed`] draws them with the port's own generator. Python's `log` callback (it
//! prints the cell diameters used) is not part of the port: [`used_cells`] returns them.
//!
//! Fields are flat row-major slices (a `(h, w, 3)` array of directions is `h * w` [`Vec3`]s).

use crate::climate::convective_tendency;
use crate::matching::ndimage;
use crate::noise::{
    cell_edges, fbm, octave_ladder, CellEdgesPlan, FbmPlan, DEFAULT_LACUNARITY, DEFAULT_TOP,
};
use crate::sphere::{normalize, Vec3};
use num_traits::Float;
use std::f64::consts::PI;

/// Mesoscale cell diameters (km): more small cells than large ones (`CELL_DIAMETERS_KM`).
pub const CELL_DIAMETERS_KM: [f64; 4] = [15.0, 30.0, 60.0, 120.0];
/// Their weights (`CELL_WEIGHTS`).
pub const CELL_WEIGHTS: [f64; 4] = [0.4, 0.3, 0.2, 0.1];

/// Seed offset of the generic fBm: `fbm(..., seed + 3, ...)`.
pub const GENERIC_SEED_OFFSET: u64 = 3;
/// Seed offset of the street fBm: `fbm(..., seed + 5, ...)`.
pub const STREET_SEED_OFFSET: u64 = 5;

// ---------------------------------------------------------------------------------------------------------
// numpy semantics, in one place

/// A Python float literal meeting an array of precision `F`: rounded to `F` (NEP 50 weak scalar).
fn lit<F: Float>(x: f64) -> F {
    F::from(x).expect("float conversion")
}

/// An element converted between precisions (`astype`, or the widening of numpy's type promotion).
fn conv<A: Float, B: Float>(x: A) -> B {
    B::from(x).expect("float conversion")
}

/// `np.degrees(x)` in the precision of `x`: `x * (180 / pi)` with the constant formed IN that precision
/// (numpy's `npy_degrees` is `x * (180.0f / NPY_PIf)` for a float32, which differs from the float32 rounding of
/// the exact `180 / pi`, as Rust's `f32::to_degrees` uses).
fn degrees<F: Float>(x: F) -> F {
    x * (lit::<F>(180.0) / lit::<F>(PI))
}

/// `np.clip(x, lo, hi)`: numpy's `min(max(x, lo), hi)` with a NaN passing through both steps.
fn np_clip<F: Float>(x: F, lo: F, hi: F) -> F {
    let m = if x.is_nan() || x > lo { x } else { lo };
    if m.is_nan() || m < hi {
        m
    } else {
        hi
    }
}

/// `np.maximum(a, b)`: `a` when `a >= b` or `a` is NaN, else `b` (so a NaN in `b` propagates).
fn np_maximum<F: Float>(a: F, b: F) -> F {
    if a >= b || a.is_nan() {
        a
    } else {
        b
    }
}

// ---------------------------------------------------------------------------------------------------------
// Scalar building blocks

/// Port of `_smoothstep(a, b, x)`: the cubic `k^2 (3 - 2k)` with `k = clip((x - a) / (b - a), 0, 1)`, in the
/// precision of `x` (`a`, `b` and `b - a` are Python floats, rounded to that precision where they meet `x`).
/// Twin of `climate._smoothstep`.
pub fn smoothstep<F: Float>(a: f64, b: f64, x: F) -> F {
    let k = np_clip((x - lit::<F>(a)) / lit::<F>(b - a), F::zero(), F::one());
    k * k * (lit::<F>(3.0) - lit::<F>(2.0) * k)
}

/// Port of `_gauss(x, mu, s)`: `exp(-((x - mu) / s)^2)` (no 1/2 in the exponent), in the precision of `x`.
/// Twin of `climate._gauss`.
pub fn gauss<F: Float>(x: F, mu: f64, s: f64) -> F {
    let d = (x - lit::<F>(mu)) / lit::<F>(s);
    (-(d * d)).exp()
}

/// Port of `_standardize(v)`: `(v - mean) / sd`, a zero-mean unit-variance field, over the WHOLE array.
///
/// The statistics are numpy's, in the precision of `v` (float32 for a float32 field): `mean` is a pairwise sum
/// over the count and `std` (ddof 0) the square root of the pairwise sum of the squared deviations over the
/// count. A zero standard deviation falls back to a divisor of 1 (`sd > 0`, strict, no epsilon), so a constant
/// field gives `v - mean`, not NaN. A CONSTANT float32 field whose float32 mean is not exactly the constant has
/// a tiny nonzero std and standardizes to -1 or +1, as in numpy (a numpy float32 artifact: only dyadic
/// constants such as 0 and 0.5, and some lengths, give 0).
pub fn standardize<F: Float + Send + Sync>(v: &[F]) -> Vec<F> {
    let sd = ndimage::std(v, 0);
    let mean = ndimage::mean(v);
    let divisor = if sd > F::zero() { sd } else { F::one() };
    crate::par::map(v, |&x| (x - mean) / divisor)
}

/// The mean and standard deviation a field is standardized with, when they are known from elsewhere (the banded
/// engine's pass A measures them over the whole map in float64 blocks and hands them to every band).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Norm {
    pub mean: f32,
    pub std: f32,
}

/// [`standardize`] with given statistics: a zero or negative standard deviation falls back to a divisor of 1.
pub fn standardize_with<F: Float + Send + Sync>(v: &[F], mean: F, std: F) -> Vec<F> {
    let divisor = if std > F::zero() { std } else { F::one() };
    crate::par::map(v, |&x| (x - mean) / divisor)
}

/// The four standardizations of [`regime_texture`], given: the generic and street fBm, and the closed- and
/// open-cell sums (`None` when no cell diameter is used).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureNorm {
    pub generic: Norm,
    pub street: Norm,
    pub cells: Option<(Norm, Norm)>,
}

fn standardize_opt(v: &[f32], norm: Option<Norm>) -> Vec<f32> {
    match norm {
        Some(n) => standardize_with(v, n.mean, n.std),
        None => standardize(v),
    }
}

// ---------------------------------------------------------------------------------------------------------
// regime_weights

/// The weights (sum <= 1, the rest is fBm) of the closed-cell, open-cell and street textures, one value per
/// texel, in the precision `P` of [`regime_weights`].
#[derive(Clone, Debug, PartialEq)]
pub struct RegimeWeights<P> {
    pub closed: Vec<P>,
    pub open: Vec<P>,
    pub street: Vec<P>,
}

/// [`regime_weights`] together with the intermediates the golden cases check, there to localize a mismatch:
/// `convective` (`convective_tendency` of the latitude in degrees, precision `L`), `cold` (the cold-air ramp
/// `smoothstep(0, 8, |back lat| - |lat|)`, in degrees, precision `P`) and `total_raw` (`closed + open + street`
/// before the normalization, precision `P`).
#[derive(Clone, Debug, PartialEq)]
pub struct RegimeTerms<L, P> {
    pub weights: RegimeWeights<P>,
    pub convective: Vec<L>,
    pub cold: Vec<P>,
    pub total_raw: Vec<P>,
}

struct TermsAt<L, P> {
    closed: P,
    open: P,
    street: P,
    convective: L,
    cold: P,
    total_raw: P,
}

/// One texel of `regime_weights`. `L` is the precision of the latitude, `B` of the back-traced latitude and
/// `P` of the result, the numpy promotion of the two (`P` at least as wide as both).
///
/// The factors that depend on the latitude alone (`1 - c`, the ramps and the gaussian of `x`) are computed in
/// `L`; they meet the cold-air ramp (`P`, the difference of the two latitudes) in the products and are widened
/// there.
fn terms_at<L: Float, B: Float, P: Float>(lat: L, lat_back: B, hadley_deg: f64) -> TermsAt<L, P> {
    let latd = degrees(lat);
    let x = latd.abs() / lit::<L>(hadley_deg);
    let c = convective_tendency(latd, hadley_deg);
    // Equatorward displacement of the air over the advection time: a cold-air outbreak.
    let cold: P = smoothstep(0.0, 8.0, conv::<B, P>(degrees(lat_back).abs()) - conv::<L, P>(latd.abs()));
    let one_l = L::one();
    let one = P::one();
    let closed_l = (one_l - c) * smoothstep(0.4, 0.7, x) * (one_l - smoothstep(1.1, 1.5, x));
    let closed: P = conv::<L, P>(closed_l) * (one - cold);
    let open: P = cold * conv::<L, P>(smoothstep(0.8, 1.2, x));
    let street: P = np_maximum(conv::<L, P>(gauss(x, 0.55, 0.3)) * (one - closed), lit::<P>(0.6) * cold) * (one - open);
    let total = closed + open + street;
    // np.where evaluates both branches; only the selected one is kept.
    let scale = if total > one { one / np_maximum(total, lit::<P>(1e-9)) } else { one };
    TermsAt { closed: closed * scale, open: open * scale, street: street * scale, convective: c, cold, total_raw: total }
}

/// Port of `regime_weights(lat, lat_back, hadley_deg)`: per-texel weights (sum <= 1, the rest is fBm) of the
/// four textures, as `closed`, `open` and `street`. `lat` and `lat_back` are the latitude and the
/// back-traced latitude in radians (equal lengths); `hadley_deg` is the Hadley edge in degrees.
///
/// With `x = |lat| / hadley` and `cold = smoothstep(0, 8, |back lat| - |lat|)` (degrees: the air came from at
/// least 8 degrees poleward):
///
/// ```text
/// closed = (1 - c) smooth(0.4, 0.7, x) (1 - smooth(1.1, 1.5, x)) (1 - cold)    c = convective_tendency
/// open   = cold smooth(0.8, 1.2, x)
/// street = max(gauss(x, 0.55, 0.3) (1 - closed), 0.6 cold) (1 - open)
/// ```
///
/// and all three are divided by their sum where it exceeds 1 (`total > 1`, strict; the scale is
/// `1 / max(total, 1e-9)`).
///
/// Precision: `L` and `B` are the dtypes of `lat` and `lat_back` and `P` is numpy's promotion of the two
/// (`f64` unless both are `f32`); the arrays are widened to `P` where they meet, and the closed, open and
/// street weights are `P`. The generator passes float64 latitudes and float32 back-traced ones, so `P = f64`.
/// Panics if `P` is narrower than `L` or `B`.
pub fn regime_weights<L: Float + Send + Sync, B: Float + Send + Sync, P: Float + Send + Sync>(
    lat: &[L],
    lat_back: &[B],
    hadley_deg: f64,
) -> RegimeWeights<P> {
    check_promotion::<L, B, P>(lat, lat_back);
    // One texel does not depend on another: the three weights of every texel, in parallel and in order.
    let terms = crate::par::map_index(lat.len(), |i| {
        let t = terms_at::<L, B, P>(lat[i], lat_back[i], hadley_deg);
        [t.closed, t.open, t.street]
    });
    let mut w = RegimeWeights { closed: Vec::with_capacity(lat.len()), open: Vec::with_capacity(lat.len()), street: Vec::with_capacity(lat.len()) };
    for t in terms {
        w.closed.push(t[0]);
        w.open.push(t[1]);
        w.street.push(t[2]);
    }
    w
}

/// [`regime_weights`] with its intermediates ([`RegimeTerms`]): the same single implementation, for the golden
/// cases and for diagnosis.
pub fn regime_weight_terms<L: Float, B: Float, P: Float>(lat: &[L], lat_back: &[B], hadley_deg: f64) -> RegimeTerms<L, P> {
    check_promotion::<L, B, P>(lat, lat_back);
    let n = lat.len();
    let mut out = RegimeTerms {
        weights: RegimeWeights { closed: Vec::with_capacity(n), open: Vec::with_capacity(n), street: Vec::with_capacity(n) },
        convective: Vec::with_capacity(n),
        cold: Vec::with_capacity(n),
        total_raw: Vec::with_capacity(n),
    };
    for (&la, &lb) in lat.iter().zip(lat_back) {
        let t = terms_at::<L, B, P>(la, lb, hadley_deg);
        out.weights.closed.push(t.closed);
        out.weights.open.push(t.open);
        out.weights.street.push(t.street);
        out.convective.push(t.convective);
        out.cold.push(t.cold);
        out.total_raw.push(t.total_raw);
    }
    out
}

fn check_promotion<L: Float, B: Float, P: Float>(lat: &[L], lat_back: &[B]) {
    assert_eq!(lat.len(), lat_back.len(), "lat and lat_back must have the same length");
    assert!(
        std::mem::size_of::<P>() >= std::mem::size_of::<L>().max(std::mem::size_of::<B>()),
        "the result precision must be at least as wide as both latitudes (numpy's promotion)"
    );
}

// ---------------------------------------------------------------------------------------------------------
// regime_texture

/// One cell diameter the map resolves: `frequency` is `radius_km / diameter_km`, the cells per radian of arc.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellScale {
    pub diameter_km: f64,
    pub weight: f64,
    pub frequency: f64,
}

/// The cell diameters [`regime_texture`] uses on a body of `radius_km` mapped up to `fmax` cycles per radian,
/// in [`CELL_DIAMETERS_KM`] order. A diameter whose `f = radius_km / d_km` is past `fmax` is finer than the map
/// resolves and is skipped (`f > fmax`, strict: `f == fmax` is USED; `f` is a DIVISION, a multiplication by the
/// reciprocal can land one ulp off and break the tie).
pub fn used_cells(radius_km: f64, fmax: f64) -> Vec<CellScale> {
    CELL_DIAMETERS_KM
        .iter()
        .zip(CELL_WEIGHTS)
        .filter_map(|(&d_km, w)| {
            let f = radius_km / d_km; // cells per radian of arc
            if f > fmax {
                None
            } else {
                Some(CellScale { diameter_km: d_km, weight: w, frequency: f })
            }
        })
        .collect()
}

/// `seed + int(d_km)`: the seed of the cell-edge noise of one diameter.
pub fn cell_seed(seed: u64, d_km: f64) -> u64 {
    seed + d_km as u64
}

/// Everything [`regime_texture`] draws: the generic and street fBm plans and one cell-edge plan per USED
/// diameter ([`used_cells`], in order; none when `fmax` is too low for every diameter).
#[derive(Clone, Debug, PartialEq)]
pub struct RegimeTexturePlans {
    /// `fbm(.., seed + 3, ..)`: the fine noise, read between the point and the back-traced direction.
    pub generic: FbmPlan,
    /// `fbm(.., seed + 5, ..)`: the same noise read far back along the flow, stretched along the wind.
    pub street: FbmPlan,
    /// `cell_edges(q, f, seed + int(d_km))` for each used diameter, `frequency` = `f`.
    pub cells: Vec<CellEdgesPlan>,
}

impl RegimeTexturePlans {
    /// The plans of `regime_texture(p, back, weights, radius_km, fmax, seed, fine_frequency, fine_gain)`, each
    /// drawn from its own generator seeded as Python seeds it: the fBm ladder is
    /// `octave_ladder(fine_frequency, fmax)` (a start frequency and an octave count anchored at the top), the
    /// generic fBm uses `seed + 3`, the street fBm `seed + 5`, and the cells of diameter `d_km` `seed + int(d_km)`.
    pub fn from_seed(radius_km: f64, fmax: f64, seed: u64, fine_frequency: f64, fine_gain: f64) -> RegimeTexturePlans {
        let (f_fine, n_fine) = octave_ladder(fine_frequency, fmax, DEFAULT_LACUNARITY, DEFAULT_TOP);
        let fbm_plan = |s: u64| FbmPlan::from_seed(s, f_fine, n_fine, fine_gain, DEFAULT_LACUNARITY, false);
        RegimeTexturePlans {
            generic: fbm_plan(seed + GENERIC_SEED_OFFSET),
            street: fbm_plan(seed + STREET_SEED_OFFSET),
            cells: used_cells(radius_km, fmax)
                .iter()
                .map(|c| CellEdgesPlan::from_seed(cell_seed(seed, c.diameter_km), c.frequency))
                .collect(),
        }
    }
}

/// The result of [`regime_texture_terms`]: the texture and its intermediates, all `f32`.
#[derive(Clone, Debug, PartialEq)]
pub struct RegimeTextureTerms {
    /// Unit-variance fine texture (the result of [`regime_texture`]).
    pub texture: Vec<f32>,
    /// The standardized generic fBm.
    pub generic_std: Vec<f32>,
    /// The standardized street fBm.
    pub street_std: Vec<f32>,
    /// The cell fields, when a cell diameter is used.
    pub cells: Option<CellFields>,
}

/// The cell intermediates of [`regime_texture`]: the weighted sums of the cell ramps (accumulated in `f32`,
/// before standardization) and their standardizations over the whole array.
#[derive(Clone, Debug, PartialEq)]
pub struct CellFields {
    pub closed_sum: Vec<f32>,
    pub open_sum: Vec<f32>,
    pub closed_std: Vec<f32>,
    pub open_std: Vec<f32>,
}

/// `(a * ka + b * kb)` per component and normalized, in the precision of the directions, then as `f32` (what
/// `fbm` and `cell_edges` cast to first): `normalize(p * ka + back * kb).astype(np.float32)`.
pub(crate) fn mixed_dirs<F: Float + Send + Sync>(p: &[Vec3<F>], back: &[Vec3<F>], ka: f64, kb: f64) -> Vec<Vec3<f32>> {
    let (ka, kb) = (lit::<F>(ka), lit::<F>(kb));
    crate::par::map_index(p.len().min(back.len()), |i| {
        let (a, b) = (&p[i], &back[i]);
        let q = normalize([a[0] * ka + b[0] * kb, a[1] * ka + b[1] * kb, a[2] * ka + b[2] * kb]);
        [conv::<F, f32>(q[0]), conv::<F, f32>(q[1]), conv::<F, f32>(q[2])]
    })
}

/// Port of `regime_texture` with its intermediates: unit-variance fine texture, the regime-weighted blend of
/// cells, streets and fBm; see [`regime_texture`] for the arguments.
pub fn regime_texture_terms<F: Float + Send + Sync, W: Float + Send + Sync>(
    p: &[Vec3<F>],
    back: &[Vec3<F>],
    weights: &RegimeWeights<W>,
    radius_km: f64,
    fmax: f64,
    plans: &RegimeTexturePlans,
) -> RegimeTextureTerms {
    regime_texture_terms_norm(p, back, weights, radius_km, fmax, plans, None)
}

/// [`regime_texture_terms`] with the four standardizations given (`norm`) instead of measured over `p`: the
/// non-banded reference of the banded engine.
#[allow(clippy::too_many_arguments)]
pub fn regime_texture_terms_norm<F: Float + Send + Sync, W: Float + Send + Sync>(
    p: &[Vec3<F>],
    back: &[Vec3<F>],
    weights: &RegimeWeights<W>,
    radius_km: f64,
    fmax: f64,
    plans: &RegimeTexturePlans,
    norm: Option<&TextureNorm>,
) -> RegimeTextureTerms {
    let n = p.len();
    assert_eq!(back.len(), n, "p and back must have the same length");
    assert!(
        weights.closed.len() == n && weights.open.len() == n && weights.street.len() == n,
        "the weights must have one value per direction"
    );
    let used = used_cells(radius_km, fmax);
    assert_eq!(plans.cells.len(), used.len(), "one cell plan per used diameter ({} used)", used.len());

    let generic = standardize_opt(&fbm(&mixed_dirs(p, back, 0.75, 0.25), &plans.generic), norm.map(|n| n.generic));
    // Streets: the same noise read far back along the flow, so it is stretched along the wind.
    let street = standardize_opt(&fbm(&mixed_dirs(p, back, 0.1, 0.9), &plans.street), norm.map(|n| n.street));
    // Cells drift little over the advection time; read them a short way back.
    let q = mixed_dirs(p, back, 0.85, 0.15);
    let mut closed = vec![0.0f32; n];
    let mut open = vec![0.0f32; n];
    for (cell, plan) in used.iter().zip(&plans.cells) {
        assert_eq!(plan.frequency, cell.frequency, "the cell plan of {} km was drawn for another frequency", cell.diameter_km);
        let edge = cell_edges(&q, plan);
        let w = cell.weight as f32; // np.float32(w)
        crate::par::for_each_chunk2_mut(&mut closed, &mut open, crate::par::BLOCK, |b, cl, op| {
            let edge = &edge[b * crate::par::BLOCK..];
            for (j, (c, o)) in cl.iter_mut().zip(op.iter_mut()).enumerate() {
                *c += w * smoothstep(0.03, 0.30, edge[j]);
                *o += w * (1.0 - smoothstep(0.05, 0.25, edge[j]));
            }
        });
    }

    // The blend runs in the precision of the weights (a float32 field meeting float64 weights is widened).
    let one = W::one();
    let mut tex: Vec<W> = crate::par::map(&generic, |&v| conv::<f32, W>(v));
    let cells = if used.is_empty() {
        None
    } else {
        let cell_norm = norm.and_then(|n| n.cells);
        let closed_std = standardize_opt(&closed, cell_norm.map(|c| c.0));
        let open_std = standardize_opt(&open, cell_norm.map(|c| c.1));
        crate::par::for_each_chunk_mut(&mut tex, crate::par::BLOCK, |b, chunk| {
            let base = b * crate::par::BLOCK;
            for (j, t) in chunk.iter_mut().enumerate() {
                let i = base + j;
                let (wc, wo) = (weights.closed[i], weights.open[i]);
                *t = *t * (one - wc - wo) + wc * conv::<f32, W>(closed_std[i]) + wo * conv::<f32, W>(open_std[i]);
            }
        });
        Some(CellFields { closed_sum: closed, open_sum: open, closed_std, open_std })
    };
    crate::par::for_each_chunk_mut(&mut tex, crate::par::BLOCK, |b, chunk| {
        let base = b * crate::par::BLOCK;
        for (j, t) in chunk.iter_mut().enumerate() {
            let ws = weights.street[base + j];
            *t = *t * (one - ws) + ws * conv::<f32, W>(street[base + j]);
        }
    });
    RegimeTextureTerms {
        texture: crate::par::map(&tex, |&v| conv::<W, f32>(v)),
        generic_std: generic,
        street_std: street,
        cells,
    }
}

/// Port of `regime_texture(p, back, weights, radius_km, fmax, seed, fine_frequency, fine_gain)`: the
/// unit-variance fine texture, the regime-weighted blend of cells, streets and fBm, as `f32`.
///
/// * `p` and `back` are the unit directions of the texels and their back-traced directions (`F` = `f32` or
///   `f64`: the mixes `p * 0.75 + back * 0.25` and the normalization run in that precision, then the noise
///   casts to `f32`).
/// * `weights` are those of [`regime_weights`] (`W` = their precision; the final blend runs in `W`).
/// * The noise comes in `plans` ([`RegimeTexturePlans`], the seeds' draws): a generic fBm of `0.75 p +
///   0.25 back`, a street fBm of `0.1 p + 0.9 back`, and for each USED cell diameter ([`used_cells`]) the cell
///   edges `F2 - F1` of `0.85 p + 0.15 back` at `f = radius_km / d_km` cells per radian, ramped into a
///   closed-cell field (`smoothstep(0.03, 0.30, edge)`) and an open-cell field
///   (`1 - smoothstep(0.05, 0.25, edge)`), each weighted by [`CELL_WEIGHTS`] and accumulated in `f32`.
/// * Each of the four fields (generic, street, closed cells, open cells) is standardized over the WHOLE array
///   (a global mean and std, [`standardize`]), then
///   `tex = generic (1 - wc - wo) + wc closed + wo open` when a cell diameter is used (otherwise the closed and
///   open weights are ignored), and `tex = tex (1 - ws) + ws street`.
pub fn regime_texture<F: Float + Send + Sync, W: Float + Send + Sync>(
    p: &[Vec3<F>],
    back: &[Vec3<F>],
    weights: &RegimeWeights<W>,
    radius_km: f64,
    fmax: f64,
    plans: &RegimeTexturePlans,
) -> Vec<f32> {
    regime_texture_terms(p, back, weights, radius_km, fmax, plans).texture
}

/// `regime_texture` end to end: the plans drawn from `seed` ([`RegimeTexturePlans::from_seed`]), then the field.
#[allow(clippy::too_many_arguments)]
pub fn regime_texture_seeded<F: Float + Send + Sync, W: Float + Send + Sync>(
    p: &[Vec3<F>],
    back: &[Vec3<F>],
    weights: &RegimeWeights<W>,
    radius_km: f64,
    fmax: f64,
    seed: u64,
    fine_frequency: f64,
    fine_gain: f64,
) -> Vec<f32> {
    let plans = RegimeTexturePlans::from_seed(radius_km, fmax, seed, fine_frequency, fine_gain);
    regime_texture(p, back, weights, radius_km, fmax, &plans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::sphere::grid_dirs;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn the_constants_are_the_python_ones() {
        assert_eq!(CELL_DIAMETERS_KM, [15.0, 30.0, 60.0, 120.0]);
        assert_eq!(CELL_WEIGHTS, [0.4, 0.3, 0.2, 0.1]);
        assert!(close(CELL_WEIGHTS.iter().sum::<f64>(), 1.0, 1e-15));
        assert_eq!((GENERIC_SEED_OFFSET, STREET_SEED_OFFSET), (3, 5));
    }

    #[test]
    fn smoothstep_knees_midpoint_and_clamp() {
        for (a, b) in [(0.4, 0.7), (0.0, 8.0), (1.1, 1.5), (0.03, 0.30)] {
            assert_eq!(smoothstep(a, b, a), 0.0);
            assert_eq!(smoothstep(a, b, b), 1.0);
            assert!(close(smoothstep(a, b, 0.5 * (a + b)), 0.5, 1e-15));
            assert_eq!(smoothstep(a, b, a - 100.0), 0.0);
            assert_eq!(smoothstep(a, b, 1e30), 1.0);
            // the cubic: 3k^2 - 2k^3 at a quarter of the way
            let x = a + 0.25 * (b - a);
            assert!(close(smoothstep(a, b, x), 3.0 / 16.0 - 2.0 / 64.0, 1e-14));
        }
        assert!(smoothstep(0.0, 1.0, f64::NAN).is_nan());
    }

    #[test]
    fn smoothstep_and_gauss_run_in_the_precision_of_x() {
        // 0.3f32 - 0.03f32 differs from the float64 ramp by more than a float64 evaluation would
        let x = 0.1234567_f32;
        let s32: f32 = smoothstep(0.03, 0.30, x);
        assert_ne!(f64::from(s32), smoothstep(0.03, 0.30, f64::from(x)), "float32 chain, not float64 rounded");
        assert!(close(f64::from(s32), smoothstep(0.03, 0.30, f64::from(x)), 2e-7));
        // the weak scalar b - a is formed in float64 then rounded to float32
        let k32 = (x - 0.03_f32) / (0.30_f64 - 0.03_f64) as f32;
        assert_eq!(s32, k32 * k32 * (3.0 - 2.0 * k32));
        let g32: f32 = gauss(0.2_f32, 0.55, 0.3);
        assert!(close(f64::from(g32), gauss(f64::from(0.2_f32), 0.55, 0.3), 2e-7));
        assert_ne!(f64::from(g32), gauss(f64::from(0.2_f32), 0.55, 0.3));
    }

    #[test]
    fn gauss_has_no_half_in_the_exponent() {
        assert_eq!(gauss(0.55, 0.55, 0.3), 1.0);
        assert!(close(gauss(0.85, 0.55, 0.3), (-1.0_f64).exp(), 1e-15)); // one s away: e^-1, not e^-0.5
        assert!(close(gauss(0.25, 0.55, 0.3), gauss(0.85, 0.55, 0.3), 1e-15)); // symmetric
        assert_eq!(gauss(1e3, 0.55, 0.3), 0.0); // underflows
        assert_eq!(gauss(-1e3, 0.55, 0.3), 0.0);
    }

    fn sample(seed: u64, n: usize) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        (0..n).map(|_| 3.0 * rng.normal() - 7.0).collect()
    }

    #[test]
    fn standardize_gives_zero_mean_unit_variance() {
        let v = sample(1, 1000);
        let y = standardize(&v);
        let n = y.len() as f64;
        let m = y.iter().sum::<f64>() / n;
        let var = y.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / n;
        assert!(close(m, 0.0, 1e-12) && close(var, 1.0, 1e-12), "mean {m}, var {var}");
        // float32: the statistics are float32 ones, still unit variance to float32 accuracy
        let v32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let y32 = standardize(&v32);
        let m32 = y32.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let var32 = y32.iter().map(|&x| (f64::from(x) - m32).powi(2)).sum::<f64>() / n;
        assert!(close(m32, 0.0, 1e-5) && close(var32, 1.0, 1e-5), "mean {m32}, var {var32}");
    }

    #[test]
    fn standardize_is_scale_and_shift_free_down_to_a_tiny_std() {
        // sd > 0 is strict with no epsilon: a std of 1e-15 still gives unit variance
        let base = sample(2, 64);
        for scale in [1.0, 1e-4, 1e-15] {
            let v: Vec<f64> = base.iter().map(|&x| x * scale).collect();
            let a = standardize(&base);
            let b = standardize(&v);
            assert!(a.iter().zip(&b).all(|(x, y)| close(*x, *y, 1e-9)), "scale {scale}");
        }
    }

    #[test]
    fn standardize_of_a_constant_divides_by_one() {
        assert_eq!(standardize(&[0.0_f32; 32]), vec![0.0_f32; 32]);
        assert_eq!(standardize(&[0.5_f32; 64]), vec![0.0_f32; 64]); // dyadic: the float32 mean is exact
        assert_eq!(standardize(&[3.25_f64; 5]), vec![0.0_f64; 5]);
        assert_eq!(standardize(&[7.0_f32]), vec![0.0_f32]);
        assert!(standardize::<f32>(&[]).is_empty());
        // a field with a NaN has a NaN std: `sd > 0` is false, the divisor is 1, and the NaN spreads through the mean
        assert!(standardize(&[1.0_f64, f64::NAN, 3.0]).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn standardize_of_a_float32_constant_is_numpys_artifact() {
        // 7 values of 0.1f: the float32 sequential mean is one ulp above 0.1f, so std = 7.45e-9 > 0 and the
        // result is -1 everywhere; with 8 or 16 values the pairwise sum is exact and the result is 0
        assert_eq!(standardize(&[0.1_f32; 7]), vec![-1.0_f32; 7]);
        assert_eq!(standardize(&[0.1_f32; 8]), vec![0.0_f32; 8]);
        assert_eq!(standardize(&[0.1_f32; 16]), vec![0.0_f32; 16]);
        assert_eq!(standardize(&[0.1_f32; 64]), vec![-1.0_f32; 64]);
        // the same artifact in float64 (numpy gives std 1.39e-17 for 7 and 64 values, so +1: the float64 mean is
        // one ulp BELOW 0.1), and again exactly 0 for 8 and 16 values
        assert_eq!(standardize(&[0.1_f64; 7]), vec![1.0_f64; 7]);
        assert_eq!(standardize(&[0.1_f64; 64]), vec![1.0_f64; 64]);
        assert_eq!(standardize(&[0.1_f64; 8]), vec![0.0_f64; 8]);
    }

    fn weights_f64(lat_deg: f64, back_deg: f64, hadley: f64) -> (f64, f64, f64) {
        let w: RegimeWeights<f64> = regime_weights(&[lat_deg.to_radians()], &[back_deg.to_radians()], hadley);
        (w.closed[0], w.open[0], w.street[0])
    }

    #[test]
    fn regime_weights_at_the_equator_is_streets_only_and_tiny() {
        let (c, o, s) = weights_f64(0.0, 0.0, 30.0);
        assert_eq!((c, o), (0.0, 0.0));
        assert!(close(s, (-(0.55_f64 / 0.3).powi(2)).exp(), 1e-15), "the street gaussian at x = 0, got {s}");
    }

    #[test]
    fn regime_weights_hand_checked_points() {
        // a closed deck: 25 degrees, no displacement, Hadley 30: x = 5/6
        let x = 25.0_f64.to_radians().to_degrees().abs() / 30.0;
        let g = |x: f64, mu: f64, s: f64| (-((x - mu) / s).powi(2)).exp();
        let ss = |a: f64, b: f64, x: f64| {
            let k = ((x - a) / (b - a)).clamp(0.0, 1.0);
            k * k * (3.0 - 2.0 * k)
        };
        let cv = (0.15 + 0.85 * g(x, 0.0, 0.3) + 0.4 * g(x, 1.7, 0.35) - 0.05 * ss(2.0, 2.6, x)).clamp(0.0, 1.0);
        let closed = (1.0 - cv) * ss(0.4, 0.7, x) * (1.0 - ss(1.1, 1.5, x));
        let street = (g(x, 0.55, 0.3) * (1.0 - closed)).max(0.0);
        let (c, o, s) = weights_f64(25.0, 25.0, 30.0);
        assert!(close(c, closed, 1e-12) && o == 0.0 && close(s, street, 1e-12), "{c} {o} {s}");
        assert!(c + o + s < 1.0);
        // a full cold-air outbreak at x = 1.33 (40 degrees), 8+ degrees of poleward back-trace: open cells only
        let (c, o, s) = weights_f64(40.0, 55.0, 30.0);
        assert_eq!((c, o), (0.0, 1.0));
        assert_eq!(s, 0.0); // street's (1 - open) is zero
        // the poles
        let (c, o, s) = weights_f64(90.0, 90.0, 30.0);
        assert_eq!((c, o), (0.0, 0.0), "x = 3: past the closed-cell fall and no cold air");
        assert!(s < 1e-20, "the street gaussian at x = 3 is exp(-67), got {s}");
    }

    #[test]
    fn regime_weights_normalizes_where_the_total_exceeds_one() {
        // 34.75 degrees with 5 degrees of poleward back-trace: the raw total exceeds 1 (the doc of the golden case)
        let t = regime_weight_terms::<f64, f64, f64>(&[34.75_f64.to_radians()], &[39.75_f64.to_radians()], 30.0);
        assert!(t.total_raw[0] > 1.0);
        let (c, o, s) = (t.weights.closed[0], t.weights.open[0], t.weights.street[0]);
        assert!(close(c + o + s, 1.0, 1e-14), "normalized to 1, got {}", c + o + s);
        assert!(c > 0.0 && o > 0.0 && s > 0.0);
        // proportions are kept
        let raw_open = t.cold[0] * smoothstep(0.8, 1.2, 34.75_f64.to_radians().to_degrees() / 30.0);
        assert!(close(o, raw_open / t.total_raw[0], 1e-14));
    }

    #[test]
    fn regime_weights_total_threshold_is_one() {
        // sweep: wherever the raw total is <= 1 the weights are the raw values, above 1 they sum to 1
        let lat: Vec<f64> = (0..=180).map(|i| (-90.0 + f64::from(i) * 1.0).to_radians()).collect();
        for disp in [-4.0_f64, 0.0, 5.0, 12.0, 20.0] {
            let back: Vec<f64> = lat.iter().map(|&l| l + disp.to_radians() * l.signum()).collect();
            let t = regime_weight_terms::<f64, f64, f64>(&lat, &back, 30.0);
            let mut over = 0;
            for i in 0..lat.len() {
                let (c, o, s) = (t.weights.closed[i], t.weights.open[i], t.weights.street[i]);
                let tot = t.total_raw[i];
                assert!(c >= 0.0 && o >= 0.0 && s >= 0.0 && c + o + s <= 1.0 + 1e-14);
                if tot > 1.0 {
                    over += 1;
                    assert!(close(c + o + s, 1.0, 1e-14), "lat index {i}");
                } else {
                    assert!(close(c + o + s, tot, 1e-15), "lat index {i}");
                }
            }
            if disp == 5.0 {
                assert!(over > 0, "the 5 degree back-trace engages the normalization");
            }
        }
    }

    #[test]
    fn regime_weights_precisions() {
        let lat64: Vec<f64> = (-18..=18).map(|i| (f64::from(i) * 5.0).to_radians()).collect();
        let back64: Vec<f64> = lat64.iter().map(|&l| l + 7.0_f64.to_radians() * l.signum()).collect();
        let lat32: Vec<f32> = lat64.iter().map(|&v| v as f32).collect();
        let back32: Vec<f32> = back64.iter().map(|&v| v as f32).collect();
        let w64: RegimeWeights<f64> = regime_weights(&lat64, &back64, 30.0);
        let w32: RegimeWeights<f32> = regime_weights(&lat32, &back32, 30.0);
        let wmix: RegimeWeights<f64> = regime_weights(&lat64, &back32, 30.0);
        for i in 0..lat64.len() {
            for (a, b, m) in [(w64.closed[i], w32.closed[i], wmix.closed[i]), (w64.open[i], w32.open[i], wmix.open[i]), (w64.street[i], w32.street[i], wmix.street[i])] {
                assert!(close(a, f64::from(b), 1.5e-5 + 1e-4 * a.abs()), "float32 chain: {a} vs {b}");
                assert!(close(a, m, 1.5e-5 + 1e-4 * a.abs()), "mixed: {a} vs {m}");
            }
        }
        // the float32 chain is not the float64 one rounded: it runs in float32
        assert!((0..lat64.len()).any(|i| f64::from(w32.street[i]) != f64::from(w64.street[i] as f32)));
    }

    #[test]
    fn degrees_uses_the_constant_of_its_own_precision() {
        // numpy's float32 degrees multiplies by float32(180) / float32(pi) = 57.295776, not by 57.29578
        assert_eq!(degrees(1.0_f32), 180.0_f32 / std::f32::consts::PI);
        assert_ne!(degrees(1.0_f32), 1.0_f32.to_degrees());
        assert_eq!(degrees(1.0_f64), 180.0 / PI);
    }

    #[test]
    #[should_panic(expected = "result precision")]
    fn regime_weights_refuses_a_narrower_result() {
        let _: RegimeWeights<f32> = regime_weights(&[0.5_f64], &[0.5_f32], 30.0);
    }

    #[test]
    fn used_cells_cut_at_the_resolution_of_the_map() {
        let sizes = |r: f64, fmax: f64| used_cells(r, fmax).iter().map(|c| c.diameter_km).collect::<Vec<_>>();
        assert_eq!(sizes(6371.0, 326.0), vec![30.0, 60.0, 120.0]); // 15 km cells: f = 424 > 326
        assert_eq!(sizes(6371.0, 20.0), Vec::<f64>::new());
        assert_eq!(sizes(600.0, 50.0), vec![15.0, 30.0, 60.0, 120.0]);
        // the tie f = 1200 / 120 = 10 exactly is USED; a hair above is not, a hair below is
        assert_eq!(sizes(1200.0, 10.0), vec![120.0]);
        assert_eq!(sizes(1200.12, 10.0), Vec::<f64>::new());
        assert_eq!(sizes(1199.88, 10.0), vec![120.0]);
        let c = used_cells(6371.0, 326.0);
        assert_eq!(c[0], CellScale { diameter_km: 30.0, weight: 0.3, frequency: 6371.0 / 30.0 });
        // a NaN fmax never skips (`f > nan` is false)
        assert_eq!(sizes(6371.0, f64::NAN).len(), 4);
        assert_eq!(cell_seed(7, 30.0), 37);
        assert_eq!(cell_seed(3, 15.0), 18);
    }

    #[test]
    fn the_cutoff_divides_instead_of_multiplying_by_the_reciprocal() {
        // 13066.700157182982 / 15 and 13066.700157182982 * (1 / 15) are adjacent doubles (the product is one ulp
        // below): with fmax just under the quotient the 15 km cells are finer than the map resolves and are
        // skipped; with the reciprocal they would sit exactly on the tie and be used
        let radius = 13066.700157182982_f64;
        let f = radius / 15.0;
        assert_ne!(f, radius * (1.0 / 15.0), "the premise of the test");
        let fmax = f64::from_bits(f.to_bits() - 1);
        assert!(!used_cells(radius, fmax).iter().any(|c| c.diameter_km == 15.0));
        assert!(used_cells(radius, f).iter().any(|c| c.diameter_km == 15.0), "f == fmax is used");
    }

    #[test]
    fn plans_are_drawn_with_python_seeds() {
        let (radius, fmax, seed) = (6371.0, 326.0, 7);
        let plans = RegimeTexturePlans::from_seed(radius, fmax, seed, 48.0, 0.65);
        let (f, n) = octave_ladder(48.0, fmax, DEFAULT_LACUNARITY, DEFAULT_TOP);
        assert_eq!(plans.generic, FbmPlan::from_seed(10, f, n, 0.65, DEFAULT_LACUNARITY, false));
        assert_eq!(plans.street, FbmPlan::from_seed(12, f, n, 0.65, DEFAULT_LACUNARITY, false));
        assert_ne!(plans.generic, plans.street);
        assert_eq!(plans.cells.len(), 3);
        for (plan, (d, k)) in plans.cells.iter().zip([(30.0_f64, 37_u64), (60.0, 67), (120.0, 127)]) {
            assert_eq!(*plan, CellEdgesPlan::from_seed(k, radius / d));
        }
        assert_eq!(plans, RegimeTexturePlans::from_seed(radius, fmax, seed, 48.0, 0.65));
        assert_ne!(plans, RegimeTexturePlans::from_seed(radius, fmax, seed + 1, 48.0, 0.65));
    }

    /// A small grid of directions and a back-traced copy rotated a little about the z axis.
    fn grid(w: usize, h: usize) -> (Vec<Vec3<f32>>, Vec<Vec3<f32>>) {
        let p = grid_dirs::<f32>(w, h);
        let (c, s) = (0.08_f32.cos(), 0.08_f32.sin());
        let back = p.iter().map(|d| [c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]]).collect();
        (p, back)
    }

    fn uniform_weights<W: Float>(n: usize, closed: f64, open: f64, street: f64) -> RegimeWeights<W> {
        let v = |x: f64| vec![lit::<W>(x); n];
        RegimeWeights { closed: v(closed), open: v(open), street: v(street) }
    }

    fn mean_var(v: &[f32]) -> (f64, f64) {
        let n = v.len() as f64;
        let m = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        (m, v.iter().map(|&x| (f64::from(x) - m).powi(2)).sum::<f64>() / n)
    }

    #[test]
    fn the_blend_selects_each_field_by_its_weight() {
        let (p, back) = grid(64, 32);
        let n = p.len();
        let (radius, fmax) = (600.0, 50.0); // all four diameters
        let plans = RegimeTexturePlans::from_seed(radius, fmax, 3, 8.0, 0.6);
        assert_eq!(plans.cells.len(), 4);
        let run = |wc: f64, wo: f64, ws: f64| regime_texture_terms(&p, &back, &uniform_weights::<f32>(n, wc, wo, ws), radius, fmax, &plans);

        let none = run(0.0, 0.0, 0.0);
        assert_eq!(none.texture, none.generic_std, "no weight: the standardized generic fBm");
        let street = run(0.0, 0.0, 1.0);
        assert_eq!(street.texture, street.street_std, "street weight 1: the standardized street fBm");
        let closed = run(1.0, 0.0, 0.0);
        let cells = closed.cells.as_ref().expect("four cell diameters are used");
        assert_eq!(closed.texture, cells.closed_std);
        let open = run(0.0, 1.0, 0.0);
        assert_eq!(open.texture, open.cells.as_ref().unwrap().open_std);
        // a mixture is the weighted sum (float32, in the code's order)
        let mix = run(0.3, 0.2, 0.25);
        let c = mix.cells.as_ref().unwrap();
        for i in (0..n).step_by(37) {
            let t = mix.generic_std[i] * (1.0 - 0.3 - 0.2) + 0.3 * c.closed_std[i] + 0.2 * c.open_std[i];
            let t = t * (1.0 - 0.25) + 0.25 * mix.street_std[i];
            assert!((mix.texture[i] - t).abs() <= 1e-6 * (1.0 + t.abs()), "texel {i}: {} vs {t}", mix.texture[i]);
        }
        // every standardized field has zero mean and unit variance over the whole array
        for f in [&none.generic_std, &none.street_std, &cells.closed_std, &cells.open_std] {
            let (m, v) = mean_var(f);
            assert!(close(m, 0.0, 1e-5) && close(v, 1.0, 1e-4), "mean {m}, var {v}");
        }
        // the cell sums: closed ramps and open ramps of the same edges sum to the weights (0.4 + 0.3 + 0.2 + 0.1)
        assert!(cells.closed_sum.iter().zip(&cells.open_sum).all(|(a, b)| (0.0..=1.0 + 1e-6).contains(a) && (0.0..=1.0 + 1e-6).contains(b)));
    }

    #[test]
    fn without_cells_the_closed_and_open_weights_are_ignored() {
        let (p, back) = grid(32, 16);
        let n = p.len();
        let (radius, fmax) = (6371.0, 20.0); // even the 120 km cells (f = 53) are finer than the map resolves
        let plans = RegimeTexturePlans::from_seed(radius, fmax, 11, 4.0, 0.65);
        assert!(plans.cells.is_empty());
        let a = regime_texture_terms(&p, &back, &uniform_weights::<f64>(n, 0.7, 0.2, 0.3), radius, fmax, &plans);
        let b = regime_texture_terms(&p, &back, &uniform_weights::<f64>(n, 0.0, 0.0, 0.3), radius, fmax, &plans);
        assert!(a.cells.is_none());
        assert_eq!(a.texture, b.texture);
        for i in (0..n).step_by(11) {
            let t = f64::from(a.generic_std[i]) * 0.7 + 0.3 * f64::from(a.street_std[i]);
            assert!(close(f64::from(a.texture[i]), t, 1e-6 * (1.0 + t.abs())), "texel {i}");
        }
    }

    #[test]
    fn the_texture_is_finite_deterministic_and_directions_may_be_float64() {
        let (p, back) = grid(48, 24);
        let n = p.len();
        let weights = regime_weights::<f64, f32, f64>(
            &(0..n).map(|i| (i % 24) as f64 * 0.06 - 0.7).collect::<Vec<_>>(),
            &(0..n).map(|i| (i % 24) as f32 * 0.06 - 0.8).collect::<Vec<_>>(),
            30.0,
        );
        let (radius, fmax) = (1500.0, 60.0);
        let t1 = regime_texture_seeded(&p, &back, &weights, radius, fmax, 21, 6.0, 0.55);
        let t2 = regime_texture_seeded(&p, &back, &weights, radius, fmax, 21, 6.0, 0.55);
        assert_eq!(t1, t2);
        assert!(t1.iter().all(|v| v.is_finite()));
        assert_ne!(t1, regime_texture_seeded(&p, &back, &weights, radius, fmax, 22, 6.0, 0.55));
        // float64 directions are mixed and normalized in float64, then cast: a float32 input widened is the same
        let p64: Vec<Vec3<f64>> = p.iter().map(|d| d.map(f64::from)).collect();
        let b64: Vec<Vec3<f64>> = back.iter().map(|d| d.map(f64::from)).collect();
        let t64 = regime_texture_seeded(&p64, &b64, &weights, radius, fmax, 21, 6.0, 0.55);
        let worst = t1.iter().zip(&t64).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(worst < 1e-2, "float32 and float64 direction math agree to float32 noise, worst {worst}");
        let (m, v) = mean_var(&t1);
        assert!(m.abs() < 0.5 && v > 0.1 && v < 3.0, "mean {m}, var {v}");
    }

    #[test]
    #[should_panic(expected = "one cell plan per used diameter")]
    fn plans_for_another_resolution_are_refused() {
        let (p, back) = grid(16, 8);
        let plans = RegimeTexturePlans::from_seed(6371.0, 20.0, 1, 4.0, 0.65); // no cells
        let _ = regime_texture(&p, &back, &uniform_weights::<f32>(p.len(), 0.1, 0.1, 0.1), 600.0, 50.0, &plans);
    }

    #[test]
    #[should_panic(expected = "another frequency")]
    fn a_cell_plan_drawn_for_another_frequency_is_refused() {
        let (p, back) = grid(16, 8);
        let mut plans = RegimeTexturePlans::from_seed(1200.0, 10.0, 1, 2.0, 0.6);
        plans.cells[0].frequency = 9.0;
        let _ = regime_texture(&p, &back, &uniform_weights::<f32>(p.len(), 0.1, 0.1, 0.1), 1200.0, 10.0, &plans);
    }
}
