//! Port of `cloudgen/generate.py`: the STAGE FUNCTIONS (the coverage composition `compose`, the per-row
//! threshold solve and its sensitivity, the moisture profile, the area mean, the advection blend and the
//! surface regime weights) and the pipeline that chains them, [`generate`] / [`generate_with_plans`].
//!
//! The random draws of the pipeline are data ([`GeneratePlans`]): the golden cases hand it numpy's recorded
//! draws, [`GeneratePlans::from_seed`] draws it with the port's own `Rng` (rule 3 of `rust/README.md`).
//!
//! Layout of the fields: row-major, `height` rows of `width` (the Python arrays of shape `(height, width)`),
//! north-up. A "column" is a `(height, 1)` array, one value per row.
//!
//! numpy's dtype promotion is the part of this module that is easy to get wrong, so it is spelled out once:
//!
//! - a Python float is WEAK: it takes the dtype of the array it meets (a `float32` array times `0.6` stays
//!   `float32`, with `0.6` rounded to `float32` first);
//! - a numpy scalar (`np.clip` of a Python float returns `np.float64`) and any array are STRONG:
//!   `float32` meets `float64` in `float64`;
//! - so `compose` runs in `float32` only when the threshold `t` is a Python float AND the streak is an
//!   array; a `float64` threshold array (what the pipeline always passes) or a scalar streak (an
//!   `np.float64`) promotes the chain to `float64` at the point where it meets that operand.
//!
//! The kernels are therefore generic over four precisions, chosen by [`compose`]'s dispatch from the inputs
//! exactly as numpy would promote them: `F` the fields (`s`, `fine`, `streak`, `subtexel`), `U` the
//! threshold-relative chain (`u = s - t`, shield ramp, cumulus density), `S` the streak factor and `P` the
//! result.

use crate::climate::{hadley_edge_deg, is_bakeable, target_zonal_coverage, BodyInputs};
use crate::flow::{build_flow, Flow, VortexKind};
use crate::matching::ndimage::{self, Mode};
use crate::noise::{fbm, map_max_frequency, octave_ladder, FbmPlan, DEFAULT_GAIN, DEFAULT_LACUNARITY};
use crate::products::{high_share, type_map, wind_map};
use crate::rng::Rng;
use crate::sphere::{grid_dirs, grid_lonlat, normalize, upsample_dirs, Vec3};
use crate::surface::{deck_window, surface_terms, Surface, SurfaceTerms, WindGrid};
use crate::texture::{regime_texture_terms_norm, regime_weights, RegimeTexturePlans, TextureNorm, RegimeWeights as TexWeights, GENERIC_SEED_OFFSET};
use num_traits::Float;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------------------------------------
// numpy primitives

/// A Python float converted to the dtype `X` of the array it meets (round to nearest, overflow to infinity).
fn k<X: Float>(x: f64) -> X {
    X::from(x).expect("float conversion")
}

/// A value widened to `float64` (exact for `float32`).
fn wide<X: Float>(x: X) -> f64 {
    x.to_f64().expect("float conversion")
}

/// A value converted between precisions (widening is exact, narrowing rounds to nearest).
fn cast<A: Float, B: Float>(x: A) -> B {
    k::<B>(wide(x))
}

/// `np.clip(x, lo, hi)` for floats: `minimum(maximum(x, lo), hi)` with NaN propagating (and `x` itself
/// returned for `-0.0`/`+0.0` ties as numpy's `_NPY_CLIP` macro does).
pub(crate) fn clip<X: Float>(x: X, lo: X, hi: X) -> X {
    let m = if x.is_nan() || x > lo { x } else { lo };
    if m.is_nan() || m < hi {
        m
    } else {
        hi
    }
}

/// `np.maximum(a, b)`: NaN in either operand wins (Rust's `max` would swallow it).
fn np_maximum<X: Float>(a: X, b: X) -> X {
    if a >= b || a.is_nan() {
        a
    } else {
        b
    }
}

/// `np.minimum(a, b)`: NaN in either operand wins.
fn np_minimum<X: Float>(a: X, b: X) -> X {
    if a <= b || a.is_nan() {
        a
    } else {
        b
    }
}

/// numpy's pairwise summation of a contiguous run (`pairwise_sum` of `loops_utils.h`): below 8 elements a
/// plain loop, up to 128 eight accumulators combined as a tree, above that a recursive halving that keeps
/// the split a multiple of 8. `a.sum()` and `a.mean(axis=-1)` of a C-contiguous array both use it, in the
/// dtype of the array (`float32` rows are summed in `float32`).
pub(crate) fn pairwise_sum<X: Float>(a: &[X]) -> X {
    let n = a.len();
    if n < 8 {
        let mut res = -X::zero();
        for &v in a {
            res = res + v;
        }
        res
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let end = n - (n % 8);
        let mut i = 8;
        while i < end {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc = *acc + a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res = res + a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

// ---------------------------------------------------------------------------------------------------------
// Parameters

/// Port of `GenParams` (a frozen dataclass): the generator's knobs with the Python defaults. The golden case
/// `genparams.defaults` pins every field and its value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenParams {
    /// Back-trace grid width (the flow is smooth at this scale).
    pub structure_width: usize,
    pub a_moisture: f64,
    pub a_ascent: f64,
    pub a_mid: f64,
    /// Contrast of the cumulus field's texture.
    pub a_fine: f64,
    pub moisture_noise: f64,
    pub advect_mid: f64,
    pub advect_fine: f64,
    pub mid_frequency: f64,
    /// Wavelength 1/48 rad = 130 km on Earth.
    pub fine_frequency: f64,
    /// Size of one fair-weather cumulus [measured: ~1-2 km]; 0 turns the sub-texel term off.
    pub cumulus_cell_km: f64,
    /// Where the fine fBm ladder stops, as a fraction of the map limit.
    pub fine_top: f64,
    /// Flatter than 0.5: Earth's spectrum keeps energy down to the texel.
    pub fine_gain: f64,
    pub fine_billow: bool,
    pub fine_min_texels: f64,
    /// Synoptic cloud: crisp edges (units of the synoptic field's standard deviation).
    pub shield_ramp: f64,
    /// How far into clear air the cumulus field reaches.
    pub cumulus_depth: f64,
    /// Cumulus fraction just outside a shield.
    pub cumulus_max: f64,
    /// How far a shield's interior dips below overcast (fibrous streaks).
    pub shield_texture: f64,
    pub threshold_smooth_deg: f64,
    /// The fine texture picked by the weather regime (texture.py) instead of one fBm.
    pub regime_texture: bool,
    /// Baseline for the metrics: the same calibration with no flow.
    pub isotropic: bool,
    /// Generate for a body whose clouds are not water (an experiment only, never a bake).
    pub allow_any_regime: bool,
    /// Use the body's surface when one is given; false keeps the blind generator (A/B).
    pub surface_terms: bool,
    /// Finite-difference step of the row sensitivity d coverage / d t, in units of s's std.
    pub sensitivity_step: f64,
}

impl Default for GenParams {
    fn default() -> GenParams {
        GenParams {
            structure_width: 512,
            a_moisture: 1.0,
            a_ascent: 0.5,
            a_mid: 0.3,
            a_fine: 0.6,
            moisture_noise: 0.45,
            advect_mid: 0.6,
            advect_fine: 0.25,
            mid_frequency: 6.0,
            fine_frequency: 48.0,
            cumulus_cell_km: 1.4,
            fine_top: 0.95,
            fine_gain: 0.65,
            fine_billow: false,
            fine_min_texels: 2.0,
            shield_ramp: 0.35,
            cumulus_depth: 1.6,
            cumulus_max: 0.5,
            shield_texture: 0.3,
            threshold_smooth_deg: 2.0,
            regime_texture: true,
            isotropic: false,
            allow_any_regime: false,
            surface_terms: true,
            sensitivity_step: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Operands that numpy broadcasts and promotes

/// The threshold `t` of [`compose`]: a Python float, or a float64 array broadcast against the fields.
#[derive(Clone, Copy, Debug)]
pub enum Threshold<'a> {
    /// A Python float: weak, so it takes the dtype of the fields (`float32` fields keep a `float32` chain).
    Scalar(f64),
    /// A `(height, 1)` float64 array (`thr[:, None]` of `generate`): one value per row.
    Column(&'a [f64]),
    /// A `(height, width)` float64 array (`thr[:, None] + offset`): one value per texel.
    Full(&'a [f64]),
}

/// The `streak` of [`compose`] and its callers: a Python float (default `0.0`), or an array of the fields'
/// dtype and shape.
#[derive(Clone, Copy, Debug)]
pub enum Streak<'a, F> {
    /// A Python float. `np.clip` turns it into an `np.float64` scalar, which is STRONG: the shield
    /// factor is a `float64` and promotes the chain it multiplies.
    Scalar(f64),
    /// A `(height, width)` array, same dtype as `s` (the factor is computed in that dtype).
    Array(&'a [F]),
}

/// The `offset` of [`solve_thresholds`]: a Python float, or an array added to the row threshold
/// (`mid[:, None] + offset` is float64 whatever the dtype of `offset`, which is widened exactly).
#[derive(Clone, Copy, Debug)]
pub enum Offset<'a, O> {
    /// A Python float (`0.0` by default): the same value in every texel.
    Scalar(f64),
    /// A `(height, 1)` array: one value per row.
    Column(&'a [O]),
    /// A `(height, width)` array: the surface terms' per-texel anomaly over the row sensitivity.
    Full(&'a [O]),
}

/// The result of [`compose`], in the dtype numpy gives it by promotion: `float32` only for a Python-float
/// threshold with an array streak on `float32` fields, `float64` otherwise.
#[derive(Clone, Debug, PartialEq)]
pub enum Composed {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Composed {
    /// `.astype(np.float32)` (round to nearest even).
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Composed::F32(v) => v.clone(),
            Composed::F64(v) => crate::par::map(v, |&x| x as f32),
        }
    }

    /// `.astype(np.float64)` (exact).
    pub fn to_f64(&self) -> Vec<f64> {
        match self {
            Composed::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Composed::F64(v) => v.clone(),
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// moisture_profile, _smoothstep

/// Port of `moisture_profile(lat_rad, hadley_deg)`: relative humidity of the source air by latitude, in
/// units of the Hadley edge: moist through the tropics, falling across the storm track, dry over the poles.
///
/// Runs in the dtype of `lat_rad` (`np.degrees` of a `float32` array is `x * (f32(180) / f32(pi))`, and the
/// Python floats are rounded to that dtype), `hadley_deg` being a Python float.
pub fn moisture_profile<F: Float + Send + Sync>(lat_rad: &[F], hadley_deg: f64) -> Vec<F> {
    let rad2deg = k::<F>(180.0) / k::<F>(PI);
    let hadley = k::<F>(hadley_deg);
    crate::par::map(lat_rad, |&lat| {
        let x = (lat * rad2deg).abs() / hadley;
        let t = clip((x - k::<F>(0.6)) / k::<F>(1.6), F::zero(), F::one());
        k::<F>(1.0) - k::<F>(0.8) * t * t * (k::<F>(3.0) - k::<F>(2.0) * t)
    })
}

/// `k * k * (3 - 2 k)` with `k = clip((x - a) / ba, 0, 1)`, the window constants already in the dtype `X`.
fn smoothstep_c<X: Float>(a: X, ba: X, x: X) -> X {
    let kk = clip((x - a) / ba, X::zero(), X::one());
    kk * kk * (k::<X>(3.0) - k::<X>(2.0) * kk)
}

/// Port of `_smoothstep(a, b, x)`: the Hermite step `3 k^2 - 2 k^3` with `k = clip((x - a) / (b - a), 0, 1)`,
/// elementwise, in the dtype of `x` (`a`, `b` are Python floats and `b - a` is formed in float64 first).
///
/// Unguarded, as in Python: `a == b` divides by zero (`-inf`/`+inf` clip to 0/1, `x == a` gives NaN, which
/// the clip keeps), and NaN propagates.
pub fn smoothstep<F: Float>(a: f64, b: f64, x: F) -> F {
    smoothstep_c(k::<F>(a), k::<F>(b - a), x)
}

// ---------------------------------------------------------------------------------------------------------
// compose

/// One row of [`compose`]'s threshold: the same value for the whole row (a scalar or a column entry), or one
/// per texel.
#[derive(Clone, Copy)]
enum RowT<'a> {
    Const(f64),
    Cols(&'a [f64]),
}

/// `1 - shield_texture * clip(0.5 + 0.5 streak, 0, 1)` in the dtype `S` of the streak.
fn streak_factor<S: Float>(streak: S, texture: S) -> S {
    let half = k::<S>(0.5);
    k::<S>(1.0) - texture * clip(half + half * streak, S::zero(), S::one())
}

/// One row of `compose`, element by element, with the precision of every intermediate pinned by the type
/// parameters (see the module docs): `F` fields, `U` for `u = s - t` and everything derived from it before
/// it meets the streak or the fine texture, `S` for the streak factor, `P` for the shield and the result.
#[allow(clippy::too_many_arguments)]
fn compose_row<F: Float, U: Float, S: Float, P: Float>(
    s: &[F],
    fine: &[F],
    t: RowT<'_>,
    prm: &GenParams,
    streak: Streak<'_, F>,
    subtexel: Option<&[F]>,
    out: &mut [P],
) {
    let n = out.len();
    // Python floats, in the dtype of the array each one meets.
    let ramp = k::<U>(prm.shield_ramp);
    let (u_zero, u_half, u_one) = (U::zero(), k::<U>(0.5), U::one());
    let cumulus_max = k::<U>(prm.cumulus_max);
    // _smoothstep(-cumulus_depth, 0.0, u): a = -depth, b - a = 0.0 - (-depth) formed in float64.
    let (ss_a, ss_ba) = (k::<U>(-prm.cumulus_depth), k::<U>(0.0 - (-prm.cumulus_depth)));
    let (f_one, a_fine) = (F::one(), k::<F>(prm.a_fine));
    let texture = k::<S>(prm.shield_texture);
    let p_one = P::one();
    // A scalar streak gives one np.float64 factor for the whole map.
    let scalar_factor = match streak {
        Streak::Scalar(v) => Some(streak_factor(k::<S>(v), texture)),
        Streak::Array(_) => None,
    };
    for i in 0..n {
        let t_i = match t {
            RowT::Const(v) => v,
            RowT::Cols(c) => c[i],
        };
        let u: U = cast::<F, U>(s[i]) - k::<U>(t_i);
        let shield0 = clip(u_half + u / ramp, u_zero, u_one);
        // A shield is not uniformly overcast at 10 km: thinner streaks along the flow let its fraction dip.
        let factor = match (&streak, scalar_factor) {
            (_, Some(f)) => f,
            (Streak::Array(a), None) => streak_factor(cast::<F, S>(a[i]), texture),
            (Streak::Scalar(_), None) => unreachable!("a scalar streak has its factor"),
        };
        let shield: P = cast::<U, P>(shield0) * cast::<S, P>(factor);
        let density = cumulus_max * smoothstep_c(ss_a, ss_ba, u);
        let texture_gain = f_one + a_fine * fine[i];
        let mut cumulus = clip(density * cast::<F, U>(texture_gain), u_zero, u_one);
        if let Some(sub) = subtexel {
            // Sub-texel cumulus: a texel holding N independent clouds has a fraction of binomial variance
            // f (1 - f) / N around the field's value (subtexel = unit white noise / sqrt(N)).
            cumulus = clip(cumulus + (cumulus * (u_one - cumulus)).sqrt() * cast::<F, U>(sub[i]), u_zero, u_one);
        }
        out[i] = shield + (p_one - shield) * cast::<U, P>(cumulus);
    }
}

fn check_shapes<F>(s: &[F], fine: &[F], height: usize, width: usize, subtexel: Option<&[F]>, streak: &Streak<'_, F>) {
    let n = height * width;
    assert_eq!(s.len(), n, "s must hold height x width values");
    assert_eq!(fine.len(), n, "fine must hold height x width values");
    if let Some(sub) = subtexel {
        assert_eq!(sub.len(), n, "subtexel must hold height x width values");
    }
    if let Streak::Array(a) = streak {
        assert_eq!(a.len(), n, "streak must hold height x width values");
    }
}

/// Checks the shapes of one `compose` call (the arrays hold `height * width` values, a column `height`).
fn check_threshold<F>(
    s: &[F],
    fine: &[F],
    height: usize,
    width: usize,
    t: &Threshold<'_>,
    streak: &Streak<'_, F>,
    subtexel: Option<&[F]>,
) {
    check_shapes(s, fine, height, width, subtexel, streak);
    match t {
        Threshold::Column(c) => assert_eq!(c.len(), height, "a threshold column holds one value per row"),
        Threshold::Full(f) => assert_eq!(f.len(), height * width, "a full threshold holds height x width values"),
        Threshold::Scalar(_) => {}
    }
}

/// `compose_row` of row `r` of the fields, into `out` (one row). Rows are independent of each other, which is what
/// lets the callers run them on every core: a row is always computed by one thread, left to right.
#[allow(clippy::too_many_arguments)]
fn compose_row_of<F: Float, U: Float, S: Float, P: Float>(
    r: usize,
    s: &[F],
    fine: &[F],
    width: usize,
    t: Threshold<'_>,
    prm: &GenParams,
    streak: Streak<'_, F>,
    subtexel: Option<&[F]>,
    out: &mut [P],
) {
    let span = r * width..(r + 1) * width;
    let t_row = match t {
        Threshold::Scalar(v) => RowT::Const(v),
        Threshold::Column(c) => RowT::Const(c[r]),
        Threshold::Full(f) => RowT::Cols(&f[span.clone()]),
    };
    let streak_row = match streak {
        Streak::Scalar(v) => Streak::Scalar(v),
        Streak::Array(a) => Streak::Array(&a[span.clone()]),
    };
    compose_row::<F, U, S, P>(&s[span.clone()], &fine[span.clone()], t_row, prm, streak_row, subtexel.map(|x| &x[span.clone()]), out);
}

/// Port of `compose(s, fine, t, prm, streak, subtexel)`: coverage from the synoptic field `s` (unit
/// variance), the fine texture and a threshold `t`.
///
/// Two cloud populations, the split the satellite cloud regimes make (ISCCP weather states): synoptic
/// cloud (frontal shields, comma heads, convective clusters), where `s` exceeds `t` with a crisp edge
/// (`shield_ramp`), thinned by `streak`; and boundary-layer cumulus in the clear air around them, a partial
/// fraction rising toward the shields' edges, textured by `fine` and, when `subtexel` is given, by the
/// binomial noise of the clouds a texel holds. Monotone decreasing in `t`, so the per-row calibration
/// stays a bisection.
///
/// `s`, `fine`, `streak` (when an array) and `subtexel` are `(height, width)` arrays of the same dtype `F`.
/// The result's dtype follows numpy's promotion (see [`Composed`] and the module docs): the threshold
/// arrays and a scalar streak are float64 and promote the chain.
///
/// # Panics
/// If an array does not hold `height * width` values (`t` as a column: `height`).
#[allow(clippy::too_many_arguments)]
pub fn compose<F: Float + Send + Sync>(
    s: &[F],
    fine: &[F],
    height: usize,
    width: usize,
    t: Threshold<'_>,
    prm: &GenParams,
    streak: Streak<'_, F>,
    subtexel: Option<&[F]>,
) -> Composed {
    fn run<F: Float + Send + Sync, U: Float, S: Float, P: Float + Send + Sync>(
        s: &[F],
        fine: &[F],
        height: usize,
        width: usize,
        t: Threshold<'_>,
        prm: &GenParams,
        streak: Streak<'_, F>,
        subtexel: Option<&[F]>,
    ) -> Vec<P> {
        check_threshold(s, fine, height, width, &t, &streak, subtexel);
        let mut out = vec![P::zero(); height * width];
        crate::par::for_each_chunk_mut(&mut out, width.max(1), |r, row| {
            compose_row_of::<F, U, S, P>(r, s, fine, width, t, prm, streak, subtexel, row);
        });
        out
    }
    let wide_fields = std::mem::size_of::<F>() == 8;
    // numpy's promotion: u = s - t is float64 for a float64 array t (weak python float otherwise); the
    // streak factor is float64 for a scalar streak (an np.float64), the fields' dtype for an array; the
    // shield, which multiplies them, and the result take the wider of the two.
    match (wide_fields, t, streak) {
        (true, _, _) => Composed::F64(run::<F, f64, f64, f64>(s, fine, height, width, t, prm, streak, subtexel)),
        (false, Threshold::Scalar(_), Streak::Array(_)) => {
            Composed::F32(run::<F, f32, f32, f32>(s, fine, height, width, t, prm, streak, subtexel))
        }
        (false, Threshold::Scalar(_), Streak::Scalar(_)) => {
            Composed::F64(run::<F, f32, f64, f64>(s, fine, height, width, t, prm, streak, subtexel))
        }
        (false, _, Streak::Array(_)) => Composed::F64(run::<F, f64, f32, f64>(s, fine, height, width, t, prm, streak, subtexel)),
        (false, _, Streak::Scalar(_)) => Composed::F64(run::<F, f64, f64, f64>(s, fine, height, width, t, prm, streak, subtexel)),
    }
}

/// `compose(s, fine, t, ...).mean(axis=1)` for a float64 threshold ARRAY (`t` is never `Scalar` here): the
/// result is float64, and each row mean is numpy's pairwise float64 sum over the row divided by the width.
#[allow(clippy::too_many_arguments)]
fn mean_rows<F: Float + Send + Sync>(
    s: &[F],
    fine: &[F],
    height: usize,
    width: usize,
    t: Threshold<'_>,
    prm: &GenParams,
    streak: Streak<'_, F>,
    subtexel: Option<&[F]>,
) -> Vec<f64> {
    assert!(!matches!(t, Threshold::Scalar(_)), "mean_rows is for float64 threshold arrays");
    check_threshold(s, fine, height, width, &t, &streak, subtexel);
    // The rows run in parallel, each summed by one thread with numpy's pairwise sum: bit-identical to a serial loop.
    crate::par::map_rows(height, |r| {
        let mut row = vec![0.0_f64; width];
        match streak {
            Streak::Array(_) => compose_row_of::<F, f64, F, f64>(r, s, fine, width, t, prm, streak, subtexel, &mut row),
            Streak::Scalar(_) => compose_row_of::<F, f64, f64, f64>(r, s, fine, width, t, prm, streak, subtexel, &mut row),
        }
        pairwise_sum(&row) / width as f64
    })
}

// ---------------------------------------------------------------------------------------------------------
// solve_thresholds, row_sensitivity

/// Port of `solve_thresholds(s, fine, target, prm, streak, subtexel, offset)`: per row, the threshold `t`
/// with `mean(compose(s, fine, t + offset)) == target`, by bisection.
///
/// 36 steps on `[-10, 10]` (literals of the source), returning the midpoint of the final bracket
/// (`20 / 2^36` wide, never exactly +-10). The decision is the strict `mean > target` on float64 row means
/// (numpy's pairwise sum), so an unreachable target runs to the lower bound, and a row exactly at the target
/// (a row of exactly 1.0 at target 1.0) never triggers either. `mid[:, None] + offset` is float64.
///
/// Returns one threshold per row. `target` holds `height` values.
#[allow(clippy::too_many_arguments)]
pub fn solve_thresholds<F: Float + Send + Sync, O: Float + Send + Sync>(
    s: &[F],
    fine: &[F],
    height: usize,
    width: usize,
    target: &[f64],
    prm: &GenParams,
    streak: Streak<'_, F>,
    subtexel: Option<&[F]>,
    offset: Offset<'_, O>,
) -> Vec<f64> {
    assert_eq!(target.len(), height, "target holds one value per row");
    match offset {
        Offset::Column(c) => assert_eq!(c.len(), height, "an offset column holds one value per row"),
        Offset::Full(f) => assert_eq!(f.len(), height * width, "a full offset holds height x width values"),
        Offset::Scalar(_) => {}
    }
    let mut lo = vec![-10.0_f64; height];
    let mut hi = vec![10.0_f64; height];
    let mut t_buf: Vec<f64> = Vec::new();
    for _ in 0..36 {
        let mid: Vec<f64> = (0..height).map(|r| 0.5 * (lo[r] + hi[r])).collect();
        t_buf.clear();
        let t = match offset {
            Offset::Scalar(o) => {
                t_buf.extend(mid.iter().map(|&m| m + o));
                Threshold::Column(&t_buf)
            }
            Offset::Column(c) => {
                t_buf.extend((0..height).map(|r| mid[r] + wide(c[r])));
                Threshold::Column(&t_buf)
            }
            Offset::Full(f) => {
                t_buf.resize(height * width, 0.0);
                let mid = &mid;
                crate::par::for_each_chunk_mut(&mut t_buf, width.max(1), |r, row| {
                    for (slot, &o) in row.iter_mut().zip(&f[r * width..(r + 1) * width]) {
                        *slot = mid[r] + wide(o);
                    }
                });
                Threshold::Full(&t_buf)
            }
        };
        let mean = mean_rows(s, fine, height, width, t, prm, streak, subtexel);
        for r in 0..height {
            if mean[r] > target[r] {
                lo[r] = mid[r];
            } else {
                hi[r] = mid[r];
            }
        }
    }
    (0..height).map(|r| 0.5 * (lo[r] + hi[r])).collect()
}

/// Port of `row_sensitivity(s, fine, thr, prm, streak, subtexel)`: `-d mean(compose) / d t` per row at the
/// thresholds `thr` (`> 0`: coverage falls as `t` rises), by a central difference of half-width
/// `prm.sensitivity_step`, floored at `0.05` (applied AFTER the division) so a saturated row does not turn a
/// small anomaly into a huge offset. `thr` holds `height` values.
#[allow(clippy::too_many_arguments)]
pub fn row_sensitivity<F: Float + Send + Sync>(
    s: &[F],
    fine: &[F],
    height: usize,
    width: usize,
    thr: &[f64],
    prm: &GenParams,
    streak: Streak<'_, F>,
    subtexel: Option<&[F]>,
) -> Vec<f64> {
    assert_eq!(thr.len(), height, "thr holds one value per row");
    let e = prm.sensitivity_step;
    let below: Vec<f64> = thr.iter().map(|&t| t - e).collect();
    let above: Vec<f64> = thr.iter().map(|&t| t + e).collect();
    let lo = mean_rows(s, fine, height, width, Threshold::Column(&below), prm, streak, subtexel);
    let hi = mean_rows(s, fine, height, width, Threshold::Column(&above), prm, streak, subtexel);
    (0..height).map(|r| np_maximum((lo[r] - hi[r]) / (2.0 * e), 0.05)).collect()
}

// ---------------------------------------------------------------------------------------------------------
// area_mean, along

/// Port of `area_mean(a)`: the area-weighted mean of a north-up equirectangular field (`height` rows of
/// `width`), with the weights `cos(lat)` at the texel-center latitudes of `grid_lonlat(width, height)`.
///
/// Each row is averaged in the dtype of `a` (numpy's pairwise sum, divided by the width: a `float32` row
/// mean is a `float32`), then weighted and summed in float64 (pairwise again): `sum_i(rowmean_i c_i) /
/// sum_i c_i`.
pub fn area_mean<F: Float>(a: &[F], height: usize, width: usize) -> f64 {
    assert_eq!(a.len(), height * width, "a must hold height x width values");
    let (_, lat) = grid_lonlat(width, height);
    let c: Vec<f64> = lat.iter().map(|l| l.cos()).collect();
    let weighted: Vec<f64> = (0..height)
        .map(|r| {
            let row_mean = pairwise_sum(&a[r * width..(r + 1) * width]) / k::<F>(width as f64);
            wide(row_mean) * c[r]
        })
        .collect();
    pairwise_sum(&weighted) / pairwise_sum(&c)
}

/// Port of `along(p, back, f)`: the point `f` of the way back along the same great circle,
/// `normalize(p (1 - f) + back f)` per direction, in the dtype of the directions (`1 - f` is formed in
/// float64, then rounded to it). `f` is not clipped (`f > 1` extrapolates past `back`), and an exactly
/// antipodal pair at `f = 0.5` gives a zero vector and NaN (`sphere::normalize` is unguarded).
///
/// # Panics
/// If `p` and `back` differ in length.
pub fn along<F: Float + Send + Sync>(p: &[Vec3<F>], back: &[Vec3<F>], f: f64) -> Vec<Vec3<F>> {
    assert_eq!(p.len(), back.len(), "p and back hold the same number of directions");
    let (keep, move_) = (k::<F>(1.0 - f), k::<F>(f));
    crate::par::map_index(p.len(), |i| {
        let (a, b) = (&p[i], &back[i]);
        normalize([a[0] * keep + b[0] * move_, a[1] * keep + b[1] * move_, a[2] * keep + b[2] * move_])
    })
}

// ---------------------------------------------------------------------------------------------------------
// surface_regime_weights

/// The weather-regime shares returned by [`surface_regime_weights`] (the dict `closed`, `open`, `street`
/// of float32 `(height, width)` fields).
#[derive(Clone, Debug, PartialEq)]
pub struct RegimeWeights {
    pub closed: Vec<f32>,
    pub open: Vec<f32>,
    pub street: Vec<f32>,
}

/// Port of `surface_regime_weights(weights, terms, lat, hadley_deg)`: closed cells are a marine regime,
/// thinned over land and filled in over the decks; the other regimes give up the share the decks take.
///
/// `closed`, `open`, `street` are the regime weights (the entries of the Python dict), `W` their dtype:
/// `f64` as the pipeline holds them (`regime_weights` of a float64 latitude grid), `f32` as the golden
/// cases of this function feed them. `continentality` and `deck_strength` are the two `SurfaceTerms` fields
/// the function reads (float32); `lat_rows` is the latitude of each row in radians (`generate` broadcasts one
/// latitude per row over the columns, so the window is evaluated once per row). All `(height, width)`.
///
/// Dtype chain, reproduced (for `W = f32` it leaves slightly negative shares where the float32 rounding of
/// `1 - closed` meets the float64 deck, down to about -2.3e-8; the source does not clamp them):
/// `sea = 1 - continentality` is float32 and `closed * sea` takes the dtype of the weights; `deck =
/// deck_window(x) * deck_strength` is float64; in `closed + (1 - closed) deck` the factor `1 - closed` has
/// the dtype of the weights BEFORE it meets the deck; the guard `rest > 1e-9` and the floor `max(rest, 1e-9)`
/// use the Python float rounded to that dtype; the scale `min(1, (1 - closed) / max(rest, 1e-9))` is float64;
/// the results are cast to float32.
///
/// # Panics
/// If an array does not hold `lat_rows.len() * width` values.
#[allow(clippy::too_many_arguments)]
pub fn surface_regime_weights<W: Float + Send + Sync>(
    closed: &[W],
    open: &[W],
    street: &[W],
    continentality: &[f32],
    deck_strength: &[f32],
    lat_rows: &[f64],
    width: usize,
    hadley_deg: f64,
) -> RegimeWeights {
    let height = lat_rows.len();
    let n = height * width;
    for a in [closed, open, street] {
        assert_eq!(a.len(), n, "every weight field holds height x width values");
    }
    for a in [continentality, deck_strength] {
        assert_eq!(a.len(), n, "every weight field holds height x width values");
    }
    let rad2deg = 180.0 / PI;
    let guard = k::<W>(1e-9);
    let mut out = RegimeWeights { closed: vec![0.0; n], open: vec![0.0; n], street: vec![0.0; n] };
    let RegimeWeights { closed: o_closed, open: o_open, street: o_street } = &mut out;
    // A texel is independent of the others: rows of the three outputs are filled in parallel.
    let rows = crate::par::map_rows(height, |r| {
        let lat = lat_rows[r];
        let x = (lat * rad2deg).abs() / hadley_deg;
        // surface.py's `deck_window` (the band of the marine decks, in units of the Hadley edge), in float64.
        let window = deck_window::<f64>(x);
        let mut row = vec![[0.0_f32; 3]; width];
        for (j, slot) in row.iter_mut().enumerate() {
            let i = r * width + j;
            let deck = window * f64::from(deck_strength[i]);
            let sea = 1.0_f32 - continentality[i];
            let thinned: W = closed[i] * cast::<f32, W>(sea);
            let room: W = W::one() - thinned;
            let filled = wide(thinned) + wide(room) * deck;
            let rest: W = open[i] + street[i];
            let scale = if rest > guard { np_minimum(1.0, (1.0 - filled) / wide(np_maximum(rest, guard))) } else { 1.0 };
            *slot = [filled as f32, (wide(open[i]) * scale) as f32, (wide(street[i]) * scale) as f32];
        }
        row
    });
    for (r, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            o_closed[r * width + j] = v[0];
            o_open[r * width + j] = v[1];
            o_street[r * width + j] = v[2];
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------
// The pipeline: generate()

/// The seed offsets of `generate()`: every random field draws from its own generator seeded `seed + offset`
/// (the moisture texture, the mid-scale texture, the fine texture, the streak texture, the sub-texel noise;
/// `regime_texture` adds its own, `texture::STREET_SEED_OFFSET` and the cell diameters).
pub const SEED_MOISTURE: u64 = 1;
pub const SEED_MID: u64 = 2;
pub const SEED_FINE: u64 = GENERIC_SEED_OFFSET;
pub const SEED_STREAK: u64 = 4;
pub const SEED_SUBTEXEL: u64 = 7;
/// Frequency (cycles per radian) and octave count of the moisture texture, `fbm(back, 1.5, 3, seed + 1)`.
pub const MOISTURE_FREQUENCY: f64 = 1.5;
pub const MOISTURE_OCTAVES: usize = 3;
/// Octave count of the mid-scale texture, `fbm(.., mid_frequency, 2, seed + 2)`.
pub const MID_OCTAVES: usize = 2;
/// Gain of the streak texture, `fbm(back, f_fine, n_fine, seed + 4, gain=0.7)`.
pub const STREAK_GAIN: f64 = 0.7;
/// The `term_width=1024` default of `surface.surface_terms`, which `generate()` leaves alone.
pub const SURFACE_TERM_WIDTH: usize = 1024;
/// `meta['orientation']`.
pub const ORIENTATION: &str = "north-up, column 0 at longitude -180, east to the right";

/// What `generate()` raises (`ValueError`) instead of making a map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GenerateError {
    /// The body's clouds are not water ([`is_bakeable`]) and `GenParams::allow_any_regime` is off.
    NotBakeable { cloud_regime: String },
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerateError::NotBakeable { cloud_regime } => write!(
                f,
                "cloud regime '{cloud_regime}': this generator's cloud amount is Earth's water clouds \
                 (climate::is_bakeable); set GenParams::allow_any_regime only for an experiment"
            ),
        }
    }
}

impl std::error::Error for GenerateError {}

fn check_bakeable(body: &BodyInputs, params: &GenParams) -> Result<(), GenerateError> {
    if is_bakeable(body) || params.allow_any_regime {
        Ok(())
    } else {
        Err(GenerateError::NotBakeable { cloud_regime: body.cloud_regime.to_string() })
    }
}

/// The octave ladder of the fine texture of a `width`-wide map: `fmax = map_max_frequency(width,
/// fine_min_texels)` and `octave_ladder(fine_frequency, fmax, top=fine_top)` (the default lacunarity). The
/// streak texture and the plain fine texture use it, and `meta.fine_octaves` reports it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FineLadder {
    /// The map's own Nyquist limit in cycles per radian.
    pub fmax: f64,
    /// Frequency of the first octave.
    pub frequency: f64,
    /// Number of octaves.
    pub octaves: usize,
}

/// [`FineLadder`] of a map `width` wide.
pub fn fine_ladder(width: usize, params: &GenParams) -> FineLadder {
    let fmax = map_max_frequency(width as f64, params.fine_min_texels);
    let (frequency, octaves) = octave_ladder(params.fine_frequency, fmax, DEFAULT_LACUNARITY, params.fine_top);
    FineLadder { fmax, frequency, octaves }
}

/// The floor and the ceiling of the local target, `np.clip(target[:, None] + terms.anomaly, 0.01, 0.99)`: a texel
/// is asked for at least 1 % and at most 99 % of cloud, whatever its anomaly. The floor binds (dry ridges); no
/// golden case reaches the ceiling (measured: target plus anomaly peaks near 0.95 on synthetic ridges up to 20 km
/// high), so only the unit tests of [`local_target`] pin it.
pub const LOCAL_TARGET_MIN: f64 = 0.01;
pub const LOCAL_TARGET_MAX: f64 = 0.99;

/// The cloud amount the surface asks of one texel: the zonal target of its row plus its anomaly (`float32`,
/// widened), clipped to [`LOCAL_TARGET_MIN`, `LOCAL_TARGET_MAX`].
pub fn local_target(target_zonal: f64, anomaly: f32) -> f64 {
    clip(target_zonal + f64::from(anomaly), LOCAL_TARGET_MIN, LOCAL_TARGET_MAX)
}

/// The `fmax` argument `generate()` hands to `regime_texture`: `fmax * fine_top / 0.95` (that function
/// anchors its own ladder at the default `top = 0.95`).
pub fn texture_fmax(fmax: f64, params: &GenParams) -> f64 {
    fmax * params.fine_top / 0.95
}

/// The divisor of the sub-texel white noise, `float32(sqrt(n_clouds))` with `n_clouds = max(1, (texel_km /
/// cumulus_cell_km)^2)` and `texel_km = 2 pi radius_km / width`: a texel holding `n` independent clouds has
/// a fraction of binomial variance `f (1 - f) / n`. `None` when `cumulus_cell_km` is not above 0 (the term
/// is off).
pub fn subtexel_scale(body: &BodyInputs, width: usize, params: &GenParams) -> Option<f32> {
    // Python `if cumulus_cell_km > 0`: a NaN size is off too.
    if params.cumulus_cell_km.is_nan() || params.cumulus_cell_km <= 0.0 {
        return None;
    }
    let texel_km = 2.0 * PI * body.radius_km / width as f64;
    let ratio = texel_km / params.cumulus_cell_km;
    let squared = ratio * ratio;
    // Python's max(1.0, x) keeps 1.0 unless x is strictly greater (a NaN loses).
    let n_clouds = if squared > 1.0 { squared } else { 1.0 };
    Some(n_clouds.sqrt() as f32)
}

/// The sub-texel field from a field of unit white noise: `white / float32(sqrt(n_clouds))` in `float32`.
pub fn subtexel_from_white(white: &[f32], scale: f32) -> Vec<f32> {
    white.iter().map(|&v| v / scale).collect()
}

/// The flow `GenParams::isotropic` swaps in: no jet, no trades, no polar easterlies and no vortex.
pub fn isotropic_flow(body: &BodyInputs) -> Flow {
    Flow { hadley_deg: hadley_edge_deg(body), vortices: Vec::new(), jet: 0.0, trades: 0.0, polar_easterlies: 0.0 }
}

/// The fine texture's plan: the regime texture's draws (generic and street fBm, one Worley cell plan per cell
/// diameter the map resolves) or, with `regime_texture` off, the one fBm that replaces it.
#[derive(Clone, Debug, PartialEq)]
pub enum TexturePlans {
    Regime(RegimeTexturePlans),
    Plain(FbmPlan),
}

/// Every random draw of `generate()`, as pure data: the golden cases hand it numpy's recorded draws, and
/// [`GeneratePlans::from_seed`] draws it with the port's own [`Rng`] (so the maps equal Python's
/// statistically, not bit for bit).
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratePlans {
    /// `fbm(back, 1.5, 3, seed + 1)`: the moisture texture read at the back-traced position.
    pub moisture: FbmPlan,
    /// `fbm(along(p, back, advect_mid), mid_frequency, 2, seed + 2)`.
    pub mid: FbmPlan,
    /// The fine texture: [`regime_texture`]'s plans, or the plain fBm of `regime_texture = false`.
    pub texture: TexturePlans,
    /// `fbm(back, f_fine, n_fine, seed + 4, gain=0.7)`.
    pub streak: FbmPlan,
    /// The sub-texel field, already divided by [`subtexel_scale`] (`None`: the term is off).
    pub subtexel: Option<Vec<f32>>,
    /// The flow: the body's vortex population, or [`isotropic_flow`].
    pub flow: Flow,
}

impl GeneratePlans {
    /// Draws every random table `generate(body, seed, width, params)` would: each field from its own generator
    /// seeded as the Python seeds it (`seed + 1` ... `seed + 7`, `seed ^ 0x5F10` for the flow), with the
    /// port's [`Rng`] in place of numpy's PCG64. Seed offsets wrap on overflow.
    pub fn from_seed(body: &BodyInputs, seed: u64, width: usize, params: &GenParams) -> GeneratePlans {
        let height = width / 2;
        let ladder = fine_ladder(width, params);
        let plan = |offset: u64, frequency: f64, octaves: usize, gain: f64, billow: bool| {
            FbmPlan::from_seed(seed.wrapping_add(offset), frequency, octaves, gain, DEFAULT_LACUNARITY, billow)
        };
        let moisture = plan(SEED_MOISTURE, MOISTURE_FREQUENCY, MOISTURE_OCTAVES, DEFAULT_GAIN, false);
        let mid = plan(SEED_MID, params.mid_frequency, MID_OCTAVES, DEFAULT_GAIN, false);
        let texture = if params.regime_texture {
            TexturePlans::Regime(RegimeTexturePlans::from_seed(
                body.radius_km,
                texture_fmax(ladder.fmax, params),
                seed,
                params.fine_frequency,
                params.fine_gain,
            ))
        } else {
            TexturePlans::Plain(plan(SEED_FINE, ladder.frequency, ladder.octaves, params.fine_gain, params.fine_billow))
        };
        let streak = plan(SEED_STREAK, ladder.frequency, ladder.octaves, STREAK_GAIN, false);
        let subtexel = subtexel_scale(body, width, params).map(|scale| {
            let mut rng = Rng::new(seed.wrapping_add(SEED_SUBTEXEL));
            (0..height * width).map(|_| rng.normal() as f32 / scale).collect()
        });
        let flow = if params.isotropic { isotropic_flow(body) } else { build_flow(body, seed) };
        GeneratePlans { moisture, mid, texture, streak, subtexel, flow }
    }
}

/// The regime weights as the pipeline holds them: `float64` straight from `regime_weights` (a `float64`
/// latitude grid meets the `float32` back-traced latitude), `float32` once `surface_regime_weights` has
/// reshaped them over land and decks.
#[derive(Clone, Debug, PartialEq)]
pub enum TraceWeights {
    F64(TexWeights<f64>),
    F32(TexWeights<f32>),
}

impl TraceWeights {
    /// The three fields as `float32` (`astype(np.float32)`: exact for `F32`, rounded to nearest for `F64`).
    pub fn to_f32(&self) -> TexWeights<f32> {
        match self {
            TraceWeights::F32(w) => w.clone(),
            TraceWeights::F64(w) => TexWeights {
                closed: w.closed.iter().map(|&v| v as f32).collect(),
                open: w.open.iter().map(|&v| v as f32).collect(),
                street: w.street.iter().map(|&v| v as f32).collect(),
            },
        }
    }
}

/// The back-trace on its own grid (`trace['back_coarse']`): `height` x `width` unit vectors, before the
/// upsampling to the map's grid.
#[derive(Clone, Debug, PartialEq)]
pub struct CoarseBack {
    pub height: usize,
    pub width: usize,
    pub data: Vec<Vec3<f32>>,
}

/// Every intermediate field of one `generate()` run: the Python `trace` dict, under the same names (and in
/// the same dtypes). Row-major, north-up; fields of the map are `height x width`, the per-row ones `height`.
/// The terms of the surface, the per-texel target and the threshold offset exist only when the surface terms
/// ran; the weights only with `regime_texture`; the shield, `kind` (Python's `type`), `high_share` and `wind`
/// only with extras.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    /// `back_coarse`: the back-trace grid (absent when `isotropic`).
    pub back_coarse: Option<CoarseBack>,
    /// `back`: where the air at each texel came from, `height x width` unit vectors.
    pub back: Vec<Vec3<f32>>,
    /// `lat_back`: `arcsin(clip(back.z, -1, 1))`, radians.
    pub lat_back: Vec<f32>,
    pub moisture_noise: Vec<f32>,
    /// `ascent`: [`Flow::ascent`] at the texel's own direction.
    pub ascent: Vec<f32>,
    pub mid_noise: Vec<f32>,
    /// `s_raw`: the synoptic field before its standardization.
    pub s_raw: Vec<f32>,
    /// `s`: `(s_raw - mean) / std` over the whole map.
    pub s: Vec<f32>,
    /// `weights.*`: the regime weights (with `regime_texture`).
    pub weights: Option<TraceWeights>,
    pub fine: Vec<f32>,
    pub streak: Vec<f32>,
    /// `subtexel`: the sub-texel field (with `cumulus_cell_km > 0`).
    pub subtexel: Option<Vec<f32>>,
    /// `target_zonal`: the zonal target per row (with the ITCZ shift of the surface terms).
    pub target_zonal: Vec<f64>,
    /// `terms.*`: the surface terms (`anomaly`, `land`, `deck`, `orographic`, `continentality`,
    /// `deck_strength`, `itcz_shift_deg`).
    pub terms: Option<SurfaceTerms>,
    /// `target_local`: `clip(target_zonal + terms.anomaly, 0.01, 0.99)`, `height x width`, `float64`.
    pub target_local: Option<Vec<f64>>,
    /// `thr0`: the bisection of the rows without the offset (surface terms only).
    pub thr0: Option<Vec<f64>>,
    /// `sensitivity`: the row sensitivity at `thr0` (surface terms only).
    pub sensitivity: Option<Vec<f64>>,
    /// `target`: the per-row target of the bisection (the row mean of `target_local` with surface terms).
    pub target: Vec<f64>,
    /// `offset`: the per-texel threshold offset, `float32` (surface terms only; the constant 0 otherwise).
    pub offset: Option<Vec<f32>>,
    /// `thr_raw`: the bisected row thresholds before smoothing.
    pub thr_raw: Vec<f64>,
    /// `thr`: the row thresholds after the smoothing in latitude.
    pub thr: Vec<f64>,
    pub coverage: Vec<f32>,
    /// `shield`: `clip(0.5 + (s - t_px) / shield_ramp, 0, 1)`, `float64` (with extras).
    pub shield: Option<Vec<f64>>,
    /// `type` (a Rust keyword): the cloud type map (with extras).
    pub kind: Option<Vec<f32>>,
    pub high_share: Option<Vec<f32>>,
    /// `wind`: `wind_height x wind_width x 2` (east, north) m/s (with extras).
    pub wind: Option<Vec<f32>>,
}

/// The derived maps of `generate()` (Python's `extras` dict, minus the surface terms, which only the trace
/// keeps).
#[derive(Clone, Debug, PartialEq)]
pub struct Extras {
    /// `type`: the cloud type, `height x width`, 0 stratiform .. 1 convective.
    pub kind: Vec<f32>,
    /// `high_share`: the conditional share of the cloud above 440 hPa, `height x width`.
    pub high_share: Vec<f32>,
    /// `wind`: east and north m/s, interleaved, on the back-trace grid (`wind_height x wind_width x 2`).
    pub wind: Vec<f32>,
    pub wind_height: usize,
    pub wind_width: usize,
}

/// One vortex of `meta['flow']['vortices']`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VortexMeta {
    pub kind: VortexKind,
    pub radius: f64,
    pub omega: f64,
    pub ascent: f64,
    pub age: f64,
    pub lon_deg: f64,
    pub lat_deg: f64,
}

/// `meta['flow']`.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowMeta {
    pub jet: f64,
    pub trades: f64,
    pub polar_easterlies: f64,
    pub vortices: Vec<VortexMeta>,
}

/// `meta['surface']`: the three numbers rounded as Python's `round` does (4, 2 and 3 decimals).
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceMeta {
    pub source: String,
    pub land_fraction: f64,
    pub itcz_shift_deg: f64,
    pub anomaly_range: [f64; 2],
}

/// The sidecar dict of `generate()` (`body`, `params` and the wall-clock `seconds` are the caller's: it has
/// them already, or measures them).
#[derive(Clone, Debug, PartialEq)]
pub struct GenMeta {
    /// The seed `generate()` was called with; [`generate_with_plans`] cannot know it (the plans carry none)
    /// and reports 0.
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    pub orientation: &'static str,
    pub hadley_edge_deg: f64,
    pub fine_octaves: usize,
    pub flow: FlowMeta,
    pub surface: Option<SurfaceMeta>,
}

/// What `generate()` returns: the coverage and its sidecar, plus the derived maps and the trace on request.
#[derive(Clone, Debug, PartialEq)]
pub struct GenOutput {
    /// Coverage in `[0, 1]`, `height x width`, north-up, column 0 at longitude -180.
    pub coverage: Vec<f32>,
    pub height: usize,
    pub width: usize,
    pub meta: GenMeta,
    pub extras: Option<Extras>,
    pub trace: Option<Trace>,
}

/// Python's `round(x, ndigits)` of a float: the correctly rounded decimal of the binary value (ties to
/// even), read back. Infinities and NaN pass through.
pub fn py_round(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.ndigits$}").parse().expect("a decimal string parses")
}

/// `math.copysign(1.0, period or 1.0)`: +1 for no spin or a zero period, -1 for a retrograde one (a NaN
/// period takes its sign bit: a Python NaN is truthy). The same rule as `products.rs`, which keeps its copy
/// private.
pub(crate) fn spin_sign(body: &BodyInputs) -> f64 {
    let period = match body.rotation_period_days {
        Some(p) if p != 0.0 => p,
        _ => 1.0,
    };
    1.0_f64.copysign(period)
}

/// `np.min` / `np.max` of a float32 field (NaN propagates).
fn extremum(a: &[f32], pick_min: bool) -> f64 {
    let mut best = f32::NAN;
    let mut seen = false;
    for &v in a {
        if v.is_nan() {
            return f64::NAN;
        }
        if !seen || (pick_min && v < best) || (!pick_min && v > best) {
            best = v;
            seen = true;
        }
    }
    f64::from(best)
}

/// The regime weights of every texel for a map `width` wide: `regime_weights(lat_grid, lat_back, hadley)`
/// with the grid latitude of each row repeated over the columns, then `surface_regime_weights` when the
/// surface terms ran.
pub(crate) fn pipeline_weights(
    lat_rows: &[f64],
    width: usize,
    lat_back: &[f32],
    hadley_deg: f64,
    terms: Option<&SurfaceTerms>,
) -> TraceWeights {
    let mut lat_grid = vec![0.0_f64; lat_rows.len() * width];
    crate::par::for_each_chunk_mut(&mut lat_grid, width.max(1), |r, row| row.fill(lat_rows[r]));
    let w = regime_weights::<f64, f32, f64>(&lat_grid, lat_back, hadley_deg);
    match terms {
        None => TraceWeights::F64(w),
        Some(t) => {
            let r = surface_regime_weights(&w.closed, &w.open, &w.street, &t.continentality, &t.deck_strength, lat_rows, width, hadley_deg);
            TraceWeights::F32(TexWeights { closed: r.closed, open: r.open, street: r.street })
        }
    }
}

/// Port of `generate(body, seed, width, params, flow, log, extras, surface, trace)` with its random draws
/// made apart: the coverage map of `body`, `width` x `width / 2`, north-up, from the plans of
/// [`GeneratePlans`].
///
/// ```text
/// s(p)        = a_moisture (moisture(back(p)) + moisture_noise fbm(back)) + a_ascent ascent(p) + a_mid fbm(along(p, f_mid))
/// coverage(p) = compose(s(p), fine(along(p, f_fine)), t(lat))                        (two populations)
/// ```
///
/// `back(p)` is where the air at `p` came from through the model flow ([`Flow::backtrace`], on a
/// `min(structure_width, width)` grid and upsampled); `t(lat)` the per-row threshold that makes the zonal mean
/// equal `climate::target_zonal_coverage` (a bisection, then smoothed in latitude); with a `surface` (and
/// `surface_terms`, and not `isotropic`) every texel's threshold is also offset by its climatological
/// anomaly over the row's sensitivity. `want_extras` adds the type, high-share and wind maps, `want_trace`
/// every intermediate field ([`Trace`]).
///
/// # Errors
/// [`GenerateError::NotBakeable`] when the clouds of `body` are not water and `params.allow_any_regime` is off.
///
/// # Panics
/// If `width < 2` (no row), `structure_width < 2` on an anisotropic run, or a plan does not fit the map
/// (`subtexel` not `height x width`, a texture plan of the other kind than `params.regime_texture`).
#[allow(clippy::too_many_arguments)]
pub fn generate_with_plans(
    body: &BodyInputs,
    plans: &GeneratePlans,
    width: usize,
    params: &GenParams,
    surface: Option<&Surface>,
    want_extras: bool,
    want_trace: bool,
) -> Result<GenOutput, GenerateError> {
    generate_with_plans_norm(body, plans, width, params, surface, want_extras, want_trace, None)
}

/// The statistics the synoptic field and the fine texture are standardized with, when they are given instead of
/// measured over the whole array (see [`generate_with_plans_norm`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormStats {
    /// Mean and standard deviation of the raw synoptic field `s` (the divisor is used as is, as in `generate`).
    pub s: crate::texture::Norm,
    /// The four standardizations of the regime texture (`None` without `regime_texture`).
    pub texture: Option<TextureNorm>,
}

/// [`generate_with_plans`] with the standardization statistics given: the non-banded reference of the banded
/// engine (`crate::banded`), which measures them in float64 blocks that do not depend on its band height.
/// `None` is `generate_with_plans` exactly.
#[allow(clippy::too_many_arguments)]
pub fn generate_with_plans_norm(
    body: &BodyInputs,
    plans: &GeneratePlans,
    width: usize,
    params: &GenParams,
    surface: Option<&Surface>,
    want_extras: bool,
    want_trace: bool,
    norm: Option<&NormStats>,
) -> Result<GenOutput, GenerateError> {
    check_bakeable(body, params)?;
    assert!(width >= 2, "a map needs at least 2 columns (a 1 x 2 grid), got {width}");
    let height = width / 2;
    let n = height * width;
    let phi_h = hadley_edge_deg(body);
    let flow = &plans.flow;
    if let Some(sub) = &plans.subtexel {
        assert_eq!(sub.len(), n, "the sub-texel plan must hold height x width values");
    }
    assert_eq!(
        matches!(plans.texture, TexturePlans::Regime(_)),
        params.regime_texture,
        "the texture plan must be the regime texture's when params.regime_texture is on, the plain fBm's otherwise"
    );

    // Where the air came from: the back-trace on a coarse grid (the flow is smooth at that scale), upsampled.
    let p = grid_dirs::<f32>(width, height);
    let (back, back_coarse): (Vec<Vec3<f32>>, Option<CoarseBack>) = if params.isotropic {
        (p.clone(), None)
    } else {
        let sw = params.structure_width.min(width);
        assert!(sw >= 2, "structure_width must be at least 2, got {}", params.structure_width);
        let coarse = grid_dirs::<f64>(sw, sw / 2);
        let back_c = flow.backtrace_default(&coarse);
        if sw == width {
            let trace_copy = want_trace.then(|| CoarseBack { height: sw / 2, width: sw, data: back_c.clone() });
            (back_c, trace_copy)
        } else {
            let widened: Vec<Vec3<f64>> = back_c.iter().map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]).collect();
            let back = upsample_dirs(&widened, sw / 2, sw, width, height);
            (back, want_trace.then_some(CoarseBack { height: sw / 2, width: sw, data: back_c }))
        }
    };

    // The surface terms need the wind (orography), which the extras return too.
    let use_surface = surface.is_some() && params.surface_terms && !params.isotropic;
    let wind_width = (params.structure_width * 2).min(width);
    let wind_height = wind_width / 2;
    let wind: Option<Vec<f32>> = (use_surface || want_extras).then(|| wind_map(flow, body, wind_width));
    let terms: Option<SurfaceTerms> = match (use_surface, surface) {
        (true, Some(surf)) => {
            let uv = wind.as_deref().expect("the wind is made when the surface terms run");
            let grid = WindGrid { uv, h: wind_height, w: wind_width };
            Some(surface_terms(surf, width, phi_h, body.radius_km, spin_sign(body), Some(&grid), SURFACE_TERM_WIDTH))
        }
        _ => None,
    };

    // The synoptic field.
    let ladder = fine_ladder(width, params);
    let lat_back: Vec<f32> = crate::par::map(&back, |b| clip(b[2], -1.0, 1.0).asin());
    let moisture_noise = fbm(&back, &plans.moisture);
    let profile = moisture_profile(&lat_back, phi_h);
    let (a_moisture, k_noise) = (k::<f32>(params.a_moisture), k::<f32>(params.moisture_noise));
    let mut s: Vec<f32> = crate::par::map_index(n, |i| a_moisture * (profile[i] + k_noise * moisture_noise[i]));
    let ascent = flow.ascent(&p);
    let a_ascent = k::<f32>(params.a_ascent);
    crate::par::for_each_chunk_mut(&mut s, crate::par::BLOCK, |b, chunk| {
        for (v, a) in chunk.iter_mut().zip(&ascent[b * crate::par::BLOCK..]) {
            *v += a_ascent * *a;
        }
    });
    let mid_noise = fbm(&along(&p, &back, params.advect_mid), &plans.mid);
    let a_mid = k::<f32>(params.a_mid);
    crate::par::for_each_chunk_mut(&mut s, crate::par::BLOCK, |b, chunk| {
        for (v, m) in chunk.iter_mut().zip(&mid_noise[b * crate::par::BLOCK..]) {
            *v += a_mid * *m;
        }
    });
    let s_raw = want_trace.then(|| s.clone());
    let (mean, std) = match norm {
        Some(n) => (n.s.mean, n.s.std),
        None => (ndimage::mean(&s), ndimage::std(&s, 0)),
    };
    crate::par::for_each_chunk_mut(&mut s, crate::par::BLOCK, |_, chunk| {
        for v in chunk.iter_mut() {
            *v = (*v - mean) / std;
        }
    });

    // The fine texture and the streaks.
    let lat_rows: Vec<f64> = grid_lonlat(width, height).1;
    let weights: Option<TraceWeights> =
        (params.regime_texture || want_extras).then(|| pipeline_weights(&lat_rows, width, &lat_back, phi_h, terms.as_ref()));
    let fine: Vec<f32> = match &plans.texture {
        TexturePlans::Regime(tp) => {
            let fmax = texture_fmax(ladder.fmax, params);
            let tn = norm.and_then(|n| n.texture);
            match weights.as_ref().expect("the regime weights are made with the regime texture") {
                TraceWeights::F64(w) => regime_texture_terms_norm(&p, &back, w, body.radius_km, fmax, tp, tn.as_ref()).texture,
                TraceWeights::F32(w) => regime_texture_terms_norm(&p, &back, w, body.radius_km, fmax, tp, tn.as_ref()).texture,
            }
        }
        TexturePlans::Plain(plan) => fbm(&along(&p, &back, params.advect_fine), plan),
    };
    let streak = fbm(&back, &plans.streak);

    // The calibration: per row, the threshold that makes the zonal mean the target.
    let itcz = terms.as_ref().map_or(0.0, |t| t.itcz_shift_deg);
    let rad2deg = 180.0 / PI;
    let target_zonal: Vec<f64> = lat_rows.iter().map(|&la| target_zonal_coverage(la * rad2deg, phi_h, itcz)).collect();
    let subtexel = plans.subtexel.as_deref();
    let streak_arg = Streak::Array(&streak[..]);
    let no_offset = Offset::<f32>::Scalar(0.0);
    let (target, offset, local, thr0, sens, thr_raw): CalibrationStages = match &terms {
        Some(t) => {
            let local: Vec<f64> = crate::par::map_index(n, |i| local_target(target_zonal[i / width], t.anomaly[i]));
            let target: Vec<f64> = (0..height).map(|r| pairwise_sum(&local[r * width..(r + 1) * width]) / width as f64).collect();
            let thr0 = solve_thresholds(&s, &fine, height, width, &target, params, streak_arg, subtexel, no_offset);
            let sens = row_sensitivity(&s, &fine, height, width, &thr0, params, streak_arg, subtexel);
            let offset: Vec<f32> = crate::par::map_index(n, |i| (-(local[i] - target[i / width]) / sens[i / width]) as f32);
            let thr_raw = solve_thresholds(&s, &fine, height, width, &target, params, streak_arg, subtexel, Offset::Full(&offset[..]));
            (target, Some(offset), Some(local), Some(thr0), Some(sens), thr_raw)
        }
        None => {
            let thr_raw = solve_thresholds(&s, &fine, height, width, &target_zonal, params, streak_arg, subtexel, no_offset);
            (target_zonal.clone(), None, None, None, None, thr_raw)
        }
    };
    let sigma = params.threshold_smooth_deg / (180.0 / height as f64);
    let thr = ndimage::gaussian_filter1d(&thr_raw, height, 1, 0, sigma, Mode::Nearest, 4.0);
    // t_px = thr[:, None] + offset: a column without the surface terms, a field with them.
    let t_px: Option<Vec<f64>> = offset.as_ref().map(|o| crate::par::map_index(n, |i| thr[i / width] + f64::from(o[i])));
    let threshold = match &t_px {
        Some(t) => Threshold::Full(&t[..]),
        None => Threshold::Column(&thr[..]),
    };
    let coverage: Vec<f32> = compose(&s, &fine, height, width, threshold, params, streak_arg, subtexel).to_f32();

    // The derived maps.
    let mut shield_trace = None;
    let extras = if want_extras {
        let t_at = |i: usize| t_px.as_ref().map_or(thr[i / width], |t| t[i]);
        let shield: Vec<f64> = crate::par::map_index(n, |i| clip(0.5 + (f64::from(s[i]) - t_at(i)) / params.shield_ramp, 0.0, 1.0));
        // The convergence zone's convective tendency moves with the ITCZ.
        let itcz_rad = itcz.to_radians();
        let lat_conv: Vec<f64> = lat_rows.iter().map(|&la| la - itcz_rad).collect();
        let w = weights.as_ref().expect("the regime weights are made for the extras");
        let kind = match w {
            TraceWeights::F64(w) => type_map::<f64, f64>(&shield, &w.closed, &w.open, &w.street, &lat_conv, width, phi_h),
            TraceWeights::F32(w) => type_map::<f64, f32>(&shield, &w.closed, &w.open, &w.street, &lat_conv, width, phi_h),
        };
        let share = high_share::<f64, f32, f32>(&shield, &ascent, &lat_conv, width, phi_h, &coverage);
        shield_trace = want_trace.then_some(shield);
        Some(Extras { kind, high_share: share, wind: wind.clone().expect("the wind is made for the extras"), wind_height, wind_width })
    } else {
        None
    };

    let meta = GenMeta {
        seed: 0,
        width,
        height,
        orientation: ORIENTATION,
        hadley_edge_deg: phi_h,
        fine_octaves: ladder.octaves,
        flow: FlowMeta {
            jet: flow.jet,
            trades: flow.trades,
            polar_easterlies: flow.polar_easterlies,
            vortices: flow
                .vortices
                .iter()
                .map(|v| VortexMeta {
                    kind: v.kind,
                    radius: v.radius,
                    omega: v.omega,
                    ascent: v.ascent,
                    age: v.age,
                    lon_deg: v.lon.to_degrees(),
                    lat_deg: v.lat.to_degrees(),
                })
                .collect(),
        },
        surface: match (&terms, surface) {
            (Some(t), Some(surf)) => Some(SurfaceMeta {
                source: surf.source.clone(),
                land_fraction: py_round(area_mean(&surf.land, surf.h, surf.w), 4),
                itcz_shift_deg: py_round(t.itcz_shift_deg, 2),
                anomaly_range: [py_round(extremum(&t.anomaly, true), 3), py_round(extremum(&t.anomaly, false), 3)],
            }),
            _ => None,
        },
    };

    let trace = want_trace.then(|| Trace {
        back_coarse,
        back,
        lat_back,
        moisture_noise,
        ascent,
        mid_noise,
        s_raw: s_raw.expect("s_raw is kept for the trace"),
        s,
        weights: if params.regime_texture { weights } else { None },
        fine,
        streak,
        subtexel: plans.subtexel.clone(),
        target_zonal,
        terms,
        target_local: local,
        thr0,
        sensitivity: sens,
        target,
        offset,
        thr_raw,
        thr,
        coverage: coverage.clone(),
        shield: shield_trace,
        kind: extras.as_ref().map(|e| e.kind.clone()),
        high_share: extras.as_ref().map(|e| e.high_share.clone()),
        wind: extras.as_ref().map(|e| e.wind.clone()),
    });
    Ok(GenOutput { coverage, height, width, meta, extras, trace })
}

/// `(target, offset, target_local, thr0, sensitivity, thr_raw)`: the stages of the calibration (the middle
/// four exist only with the surface terms).
type CalibrationStages = (Vec<f64>, Option<Vec<f32>>, Option<Vec<f64>>, Option<Vec<f64>>, Option<Vec<f64>>, Vec<f64>);

/// Port of `generate(body, seed, width, params, extras=..., surface=...)`: the plans drawn from `seed`
/// ([`GeneratePlans::from_seed`]), then [`generate_with_plans`] with `want_extras` and no trace. The
/// returned meta carries `seed`. The body is checked BEFORE anything is drawn.
///
/// # Errors
/// [`GenerateError::NotBakeable`] as [`generate_with_plans`].
pub fn generate(
    body: &BodyInputs,
    seed: u64,
    width: usize,
    params: &GenParams,
    surface: Option<&Surface>,
    want_extras: bool,
) -> Result<GenOutput, GenerateError> {
    check_bakeable(body, params)?;
    let plans = GeneratePlans::from_seed(body, seed, width, params);
    let mut out = generate_with_plans(body, &plans, width, params, surface, want_extras, false)?;
    out.meta.seed = seed;
    Ok(out)
}


// ---------------------------------------------------------------------------------------------------------
// Unit tests: the helpers the golden cases reach only through the stage functions, and the behaviors
// the cases do not pin.

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random field in [-1, 1) (LCG; no dependency on the crate's rng).
    fn field(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((x >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    fn all_f64(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| f64::from(x)).collect()
    }

    #[test]
    fn clip_and_extrema_propagate_nan_like_numpy() {
        assert!(clip(f64::NAN, 0.0, 1.0).is_nan());
        assert_eq!(clip(f64::INFINITY, 0.0, 1.0), 1.0);
        assert_eq!(clip(f64::NEG_INFINITY, 0.0, 1.0), 0.0);
        assert_eq!(clip(0.25_f32, 0.0, 1.0), 0.25);
        assert!(np_maximum(f64::NAN, 0.05).is_nan());
        assert!(np_maximum(0.0, f64::NAN).is_nan());
        assert_eq!(np_maximum(0.01, 0.05), 0.05);
        assert_eq!(np_maximum(0.2, 0.05), 0.2);
        assert!(np_minimum(1.0, f64::NAN).is_nan());
        assert!(np_minimum(f64::NAN, 1.0).is_nan());
        assert_eq!(np_minimum(1.0, 7.0), 1.0);
        assert_eq!(np_minimum(1.0, -7.0), -7.0);
    }

    /// Reference values from numpy 2.5 (`np.sum` of float32 arrays), which differ from a sequential sum:
    /// they pin the 8-accumulator tree and the recursive split above 128 elements.
    #[test]
    fn pairwise_sum_follows_numpys_block_structure() {
        let a: Vec<f32> = (0..1000).map(|i| 0.1_f32 * ((i % 7) + 1) as f32).collect();
        let sequential = a.iter().fold(0.0_f32, |acc, &v| acc + v);
        let pairwise = pairwise_sum(&a);
        // numpy: np.sum(np.float32(0.1) * ((np.arange(1000) % 7) + 1).astype(np.float32)) = 399.69998
        // (a sequential float32 loop gives 399.701).
        assert_eq!(pairwise, 399.69998_f32, "pairwise f32 sum of the 1000-element ramp");
        assert_ne!(sequential, pairwise, "a sequential sum would round differently here");
        let b: Vec<f32> = (0..130).map(|i| 0.1_f32 * ((i % 7) + 1) as f32).collect();
        assert_eq!(pairwise_sum(&b), 51.399998_f32, "numpy gives 51.399998 for 130 elements (sequential: 51.4)");
        let c: Vec<f32> = (0..9).map(|i| 0.1_f32 * (i + 1) as f32).collect();
        assert_eq!(pairwise_sum(&c), 4.5_f32);
        assert_eq!(pairwise_sum::<f64>(&[]), 0.0);
    }

    #[test]
    fn genparams_defaults_match_the_python_dataclass() {
        let p = GenParams::default();
        assert_eq!((p.structure_width, p.a_fine, p.shield_ramp, p.cumulus_depth), (512, 0.6, 0.35, 1.6));
        assert_eq!((p.cumulus_max, p.shield_texture, p.sensitivity_step), (0.5, 0.3, 0.05));
        assert!(p.regime_texture && p.surface_terms && !p.isotropic && !p.allow_any_regime && !p.fine_billow);
    }

    #[test]
    fn compose_result_dtype_follows_numpys_promotion() {
        let (h, w) = (3, 5);
        let s = field(h * w, 1);
        let fine = field(h * w, 2);
        let streak = field(h * w, 3);
        let col = [0.1, -0.2, 0.3];
        let prm = GenParams::default();
        let dtype = |c: Composed| matches!(c, Composed::F32(_));
        // Python-float t + array streak on float32 fields: float32 all the way.
        assert!(dtype(compose(&s, &fine, h, w, Threshold::Scalar(0.1), &prm, Streak::Array(&streak), None)));
        // A scalar streak is an np.float64: the result is float64.
        assert!(!dtype(compose(&s, &fine, h, w, Threshold::Scalar(0.1), &prm, Streak::Scalar(0.0), None)));
        // A float64 threshold array promotes everything it touches.
        assert!(!dtype(compose(&s, &fine, h, w, Threshold::Column(&col), &prm, Streak::Array(&streak), None)));
        // Float64 fields stay float64.
        let (s64, fine64) = (all_f64(&s), all_f64(&fine));
        assert!(!dtype(compose(&s64, &fine64, h, w, Threshold::Scalar(0.1), &prm, Streak::Scalar(0.0), None)));
    }

    #[test]
    fn a_float64_chain_does_not_depend_on_how_the_threshold_is_broadcast() {
        let (h, w) = (4, 9);
        let s = all_f64(&field(h * w, 4));
        let fine = all_f64(&field(h * w, 5));
        let sub = all_f64(&field(h * w, 6));
        let col = [0.3, -0.1, 0.0, 0.7];
        let full: Vec<f64> = col.iter().flat_map(|&t| vec![t; w]).collect();
        let prm = GenParams::default();
        let a = compose(&s, &fine, h, w, Threshold::Column(&col), &prm, Streak::Scalar(0.4), Some(&sub));
        let b = compose(&s, &fine, h, w, Threshold::Full(&full), &prm, Streak::Scalar(0.4), Some(&sub));
        assert_eq!(a, b);
        // An array streak holding the scalar everywhere is the scalar streak.
        let streak = vec![0.4_f64; h * w];
        let c = compose(&s, &fine, h, w, Threshold::Column(&col), &prm, Streak::Array(&streak), Some(&sub));
        assert_eq!(a, c);
    }

    #[test]
    fn compose_is_monotone_decreasing_in_the_threshold() {
        let (h, w) = (1, 64);
        let s = field(w, 7);
        let fine = field(w, 8);
        let prm = GenParams::default();
        let mean = |t: f64| {
            let c = compose(&s, &fine, h, w, Threshold::Scalar(t), &prm, Streak::Scalar(0.0), None).to_f64();
            c.iter().sum::<f64>() / w as f64
        };
        let mut previous = f64::INFINITY;
        for step in 0..30 {
            let m = mean(-3.0 + 0.2 * f64::from(step));
            assert!(m <= previous + 1e-12, "coverage must not rise with the threshold");
            previous = m;
        }
    }

    #[test]
    fn solve_thresholds_reaches_a_reachable_target() {
        let (h, w) = (6, 96);
        let s = field(h * w, 9);
        let fine = field(h * w, 10);
        let sub = field(h * w, 11);
        let streak = field(h * w, 12);
        let prm = GenParams::default();
        let target = [0.1, 0.3, 0.5, 0.6, 0.75, 0.9];
        let thr =
            solve_thresholds(&s, &fine, h, w, &target, &prm, Streak::Array(&streak), Some(&sub), Offset::<f32>::Scalar(0.0));
        let got = mean_rows(&s, &fine, h, w, Threshold::Column(&thr), &prm, Streak::Array(&streak), Some(&sub));
        for r in 0..h {
            assert!((got[r] - target[r]).abs() < 1e-6, "row {r}: {} vs {}", got[r], target[r]);
        }
        // The result is the midpoint of the final bracket: (k + 0.5) cells of 20 / 2^36 above -10.
        for t in &thr {
            let cells = (t + 10.0) / (20.0 / 2.0_f64.powi(36));
            assert!((cells - cells.floor() - 0.5).abs() < 1e-3, "{cells} is not a half-integer");
        }
    }

    #[test]
    fn solve_thresholds_offset_variants_agree() {
        let (h, w) = (3, 32);
        let s = field(h * w, 13);
        let fine = field(h * w, 14);
        let prm = GenParams::default();
        let target = [0.2, 0.5, 0.8];
        let streak = Streak::Scalar(0.0);
        let by_scalar = solve_thresholds(&s, &fine, h, w, &target, &prm, streak, None, Offset::<f32>::Scalar(0.25));
        let column = [0.25_f32; 3];
        let by_column = solve_thresholds(&s, &fine, h, w, &target, &prm, streak, None, Offset::Column(&column));
        let full = vec![0.25_f32; h * w];
        let by_full = solve_thresholds(&s, &fine, h, w, &target, &prm, streak, None, Offset::Full(&full));
        assert_eq!(by_scalar, by_column);
        assert_eq!(by_scalar, by_full);
        // A positive offset moves the solved threshold down by the same amount: compose sees t + offset.
        let plain = solve_thresholds(&s, &fine, h, w, &target, &prm, streak, None, Offset::<f32>::Scalar(0.0));
        for r in 0..h {
            assert!((by_scalar[r] + 0.25 - plain[r]).abs() < 1e-8);
        }
    }

    #[test]
    fn solve_thresholds_unreachable_targets_hit_the_bounds() {
        let (h, w) = (2, 16);
        let s = field(h * w, 15);
        let fine = vec![0.0_f32; h * w];
        let prm = GenParams::default();
        let thr = solve_thresholds(&s, &fine, h, w, &[1.2, -0.2], &prm, Streak::Scalar(0.0), None, Offset::<f32>::Scalar(0.0));
        let width = 20.0 / 2.0_f64.powi(36);
        assert!((thr[0] + 10.0 - width / 2.0).abs() < 1e-12, "above any coverage: the lower bound");
        assert!((thr[1] - 10.0 + width / 2.0).abs() < 1e-12, "below any coverage: the upper bound");
    }

    #[test]
    fn row_sensitivity_is_floored_and_positive() {
        let (h, w) = (4, 48);
        let s = field(h * w, 16);
        let fine = field(h * w, 17);
        let prm = GenParams::default();
        // Rows pinned at the extremes have no slope: the floor.
        let sens = row_sensitivity(&s, &fine, h, w, &[-10.0, 10.0, 0.0, 0.3], &prm, Streak::Scalar(-3.0), None);
        assert_eq!(sens[0], 0.05);
        assert_eq!(sens[1], 0.05);
        assert!(sens[2] > 0.05 && sens[3] > 0.05);
    }

    #[test]
    fn moisture_profile_is_one_in_the_tropics_and_a_fifth_at_the_pole() {
        let lat = [0.0_f64, 0.1, 90.0_f64.to_radians(), -90.0_f64.to_radians()];
        let m = moisture_profile(&lat, 30.0);
        assert_eq!(m[0], 1.0);
        assert_eq!(m[1], 1.0);
        assert!((m[2] - 0.2).abs() < 1e-12 && (m[3] - 0.2).abs() < 1e-12);
    }

    #[test]
    fn smoothstep_edges_and_the_unguarded_degenerate_window() {
        assert_eq!(smoothstep(0.0, 1.0, -5.0_f64), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 0.5_f64), 0.5);
        assert_eq!(smoothstep(0.0, 1.0, 5.0_f64), 1.0);
        assert!(smoothstep(0.5, 0.5, 0.5_f64).is_nan());
        assert_eq!(smoothstep(0.5, 0.5, 0.25_f64), 0.0);
        assert_eq!(smoothstep(0.5, 0.5, 0.75_f64), 1.0);
        assert_eq!(smoothstep(0.0, 1.0, 0.5_f32), 0.5_f32);
    }

    #[test]
    fn area_mean_of_a_constant_field_is_the_constant_in_both_precisions() {
        let a32 = vec![0.25_f32; 12 * 20];
        let a64 = vec![0.25_f64; 12 * 20];
        assert!((area_mean(&a32, 12, 20) - 0.25).abs() < 1e-15);
        assert!((area_mean(&a64, 12, 20) - 0.25).abs() < 1e-15);
    }

    #[test]
    fn area_mean_weights_rows_by_the_cosine_of_their_latitude() {
        // Two rows at +-45 degrees (equal weights): the plain mean of the row values.
        assert!((area_mean(&[1.0_f64, 1.0, 3.0, 3.0], 2, 2) - 2.0).abs() < 1e-15);
        // Three rows: the equator (weight 1) outweighs the two rows at +-60 degrees (0.5 each).
        let m = area_mean(&[0.0_f64, 1.0, 0.0], 3, 1);
        assert!((m - 0.5).abs() < 1e-12);
    }

    #[test]
    fn along_ends_and_the_antipodal_degenerate_case() {
        let p = [[1.0_f64, 0.0, 0.0]];
        let q = [[0.0_f64, 1.0, 0.0]];
        assert_eq!(along(&p, &q, 0.0), vec![[1.0, 0.0, 0.0]]);
        assert_eq!(along(&p, &q, 1.0), vec![[0.0, 1.0, 0.0]]);
        let half = along(&p, &q, 0.5)[0];
        assert!((half[0] - half[1]).abs() < 1e-15 && (half[0] * half[0] + half[1] * half[1] - 1.0).abs() < 1e-15);
        let anti = along(&p, &[[-1.0, 0.0, 0.0]], 0.5);
        assert!(anti[0].iter().all(|v| v.is_nan()), "an exactly antipodal back is a zero vector, normalized to NaN");
    }

    #[test]
    fn surface_regime_weights_branches() {
        // Four rows (21, 0, -21, 60 degrees; Hadley 30): the deck window is 1 at +-21, 0 at 0 and 60.
        let lat: Vec<f64> = [21.0_f64, 0.0, -21.0, 60.0].iter().map(|d| d.to_radians()).collect();
        let width = 1;
        // closed 0.5, open 0.2, street 0.1 over open sea with a full deck strength.
        let (c, o, s) = (vec![0.5_f32; 4], vec![0.2_f32; 4], vec![0.1_f32; 4]);
        let (cont, deck) = (vec![0.0_f32; 4], vec![1.0_f32; 4]);
        let w = surface_regime_weights(&c, &o, &s, &cont, &deck, &lat, width, 30.0);
        // On the decks (rows 0 and 2) the closed share fills to ~1 and the others give way to ~0.
        for r in [0, 2] {
            assert!(w.closed[r] > 0.999, "closed {}", w.closed[r]);
            assert!(w.open[r].abs() < 1e-3 && w.street[r].abs() < 1e-3);
        }
        // Off the decks the weights are untouched (no land, no deck).
        for r in [1, 3] {
            assert_eq!((w.closed[r], w.open[r], w.street[r]), (0.5, 0.2, 0.1));
        }
        // Over full land the closed regime vanishes off the decks.
        let land = vec![1.0_f32; 4];
        let l = surface_regime_weights(&c, &o, &s, &land, &vec![0.0_f32; 4], &lat, width, 30.0);
        assert!(l.closed.iter().all(|&v| v == 0.0));
        assert_eq!(l.open, vec![0.2_f32; 4]);
        // A remainder under the 1e-9 guard is left alone even with no room left.
        let tiny = surface_regime_weights(&[1.0], &[5e-10], &[0.0], &[0.0], &[0.0], &[0.0], 1, 30.0);
        assert_eq!(tiny.open[0], 5e-10_f32);
        // A remainder over it with no room left is cut to zero.
        let cut = surface_regime_weights(&[1.0], &[0.5], &[0.0], &[0.0], &[0.0], &[0.0], 1, 30.0);
        assert_eq!(cut.open[0], 0.0);
    }

    // -----------------------------------------------------------------------------------------------
    // Dtype chains pinned against the Python generator (numpy 2.5): the golden tolerances (1e-5 for
    // float32 kernels) cannot tell a float32 intermediate from a float64 one, these sums (on the LCG fields
    // above, computed by `cloudgen.generate` on the same arrays) can: every wrong promotion moves them by
    // 1e-8 or more, the correct chain reproduces them to the rounding of the float64 sum.

    fn scaled(v: Vec<f32>, by: f32) -> Vec<f32> {
        v.into_iter().map(|x| x * by).collect()
    }

    fn sum_f64(v: impl IntoIterator<Item = f64>) -> f64 {
        v.into_iter().sum()
    }

    fn near(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() <= tol, "got {got:?}, numpy gives {want:?} (tolerance {tol:e})");
    }

    #[test]
    fn compose_dtype_chains_reproduce_numpys_promotion() {
        let (h, w) = (3, 20);
        let s = scaled(field(h * w, 21), 1.5);
        let fine = field(h * w, 22);
        let streak = field(h * w, 23);
        let sub = scaled(field(h * w, 24), 0.3);
        let col = [0.31, -0.17, 0.05];
        let prm = GenParams::default();
        let total = |c: Composed| sum_f64(c.to_f64());
        let (arr, sc) = (Streak::Array(&streak[..]), Streak::Scalar(0.1));
        // float32 fields, float64 threshold column, array streak, sub-texel: the pipeline call.
        let a = compose(&s, &fine, h, w, Threshold::Column(&col), &prm, arr, Some(&sub));
        assert!(matches!(a, Composed::F64(_)));
        near(total(a), 35.04811418116271, 1e-11);
        // Python-float threshold + array streak: float32 end to end.
        let b = compose(&s, &fine, h, w, Threshold::Scalar(0.07), &prm, arr, Some(&sub));
        assert!(matches!(b, Composed::F32(_)));
        near(total(b), 34.48243869654834, 1e-11);
        // Python-float threshold + scalar streak (an np.float64): float32 until the streak factor, then float64.
        let c = compose(&s, &fine, h, w, Threshold::Scalar(0.07), &prm, sc, Some(&sub));
        assert!(matches!(c, Composed::F64(_)));
        near(total(c), 34.17846483078792, 1e-11);
        // float64 threshold column + scalar streak.
        let d = compose(&s, &fine, h, w, Threshold::Column(&col), &prm, sc, Some(&sub));
        near(total(d), 34.683541467505016, 1e-11);
        // No sub-texel term.
        let e = compose(&s, &fine, h, w, Threshold::Scalar(0.07), &prm, arr, None);
        assert!(matches!(e, Composed::F32(_)));
        near(total(e), 35.22827692236751, 1e-11);
        // Everything float64: no rounding left (35.0481143237 against the pipeline's 35.0481141812).
        let (s6, f6, k6, u6) = (all_f64(&s), all_f64(&fine), all_f64(&streak), all_f64(&sub));
        let f = compose(&s6, &f6, h, w, Threshold::Column(&col), &prm, Streak::Array(&k6), Some(&u6));
        near(total(f), 35.048114323702166, 1e-11);
    }

    #[test]
    fn moisture_profile_in_float32_uses_float32_degrees() {
        // np.degrees(float32) multiplies by f32(180) / f32(pi) = 57.295776, not f32(180 / pi) = 57.29578.
        let lat = scaled(field(40, 31), 1.5);
        let m = moisture_profile(&lat, 30.0);
        near(sum_f64(m.iter().map(|&v| f64::from(v))), 23.3172025680542, 1e-11);
    }

    #[test]
    fn area_mean_averages_float32_rows_in_float32() {
        let a: Vec<f32> = field(7 * 50, 41).into_iter().map(|x| x * 3.0 + 0.7).collect();
        near(area_mean(&a, 7, 50), 0.6139810067436233, 1e-13);
        near(area_mean(&all_f64(&a), 7, 50), 0.6139810184185467, 1e-13);
    }

    #[test]
    fn along_in_float32_rounds_the_weights_to_float32() {
        let vectors = |seed| -> Vec<Vec3<f32>> { field(90, seed).chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect() };
        let (p, b) = (vectors(51), vectors(52));
        for (f, want) in [(0.25, -8.905290919356048), (0.6, -0.8377106646075845), (1.5, 1.1652999520301819)] {
            let q = along(&p, &b, f);
            near(sum_f64(q.iter().flatten().map(|&v| f64::from(v))), want, 1e-11);
        }
    }

    #[test]
    fn solver_and_sensitivity_reproduce_numpy_on_float32_fields() {
        let (h, w) = (6, 96);
        let (s, fine) = (field(h * w, 9), field(h * w, 10));
        let (sub, streak) = (field(h * w, 11), field(h * w, 12));
        let prm = GenParams::default();
        let target = [0.1, 0.3, 0.5, 0.6, 0.75, 0.9];
        let arr = Streak::Array(&streak[..]);
        let thr = solve_thresholds(&s, &fine, h, w, &target, &prm, arr, Some(&sub), Offset::<f32>::Scalar(0.0));
        let want = [
            1.3857535466377158,
            0.7226520394033287,
            0.39443256115191616,
            0.07066146485158242,
            -0.4119728885416407,
            -0.7875809331017081,
        ];
        for (g, w) in thr.iter().zip(want) {
            near(*g, w, 1e-9);
        }
        let sens = row_sensitivity(&s, &fine, h, w, &thr, &prm, arr, Some(&sub));
        let want = [
            0.16845433877030427,
            0.44324030494655,
            0.4752186677730169,
            0.3984543667292628,
            0.33618606988854993,
            0.11568510232262463,
        ];
        for (g, w) in sens.iter().zip(want) {
            near(*g, w, 1e-9);
        }
        // A per-texel float32 offset (the surface terms' anomaly over the sensitivity).
        let off = scaled(field(h * w, 13), 0.4);
        let thr = solve_thresholds(&s, &fine, h, w, &target, &prm, arr, Some(&sub), Offset::Full(&off));
        let want = [
            1.5174565654888283,
            0.712361986952601,
            0.3863919743162114,
            0.029542681731982157,
            -0.429896578862099,
            -0.8456917219155002,
        ];
        for (g, w) in thr.iter().zip(want) {
            near(*g, w, 1e-9);
        }
    }

    #[test]
    fn surface_regime_weights_round_one_minus_closed_to_float32() {
        let (h, w) = (4, 100);
        let unit = |seed| -> Vec<f32> { field(h * w, seed).into_iter().map(|x| (x + 1.0) * 0.5).collect() };
        let closed = unit(61);
        let open: Vec<f32> = unit(62).iter().zip(&closed).map(|(&u, &c)| u * (1.0 - c) * 0.7).collect();
        let street: Vec<f32> =
            unit(63).iter().zip(closed.iter().zip(&open)).map(|(&u, (&c, &o))| u * (1.0 - c - o) * 0.9).collect();
        let (cont, deck) = (unit(64), unit(65));
        let lat: Vec<f64> = [0.0_f64, 21.0, -21.0, 14.0].iter().map(|d| d * (PI / 180.0)).collect();
        let r = surface_regime_weights(&closed, &open, &street, &cont, &deck, &lat, w, 30.0);
        let total = |v: &[f32]| sum_f64(v.iter().map(|&x| f64::from(x)));
        // 39 of the 400 closed shares differ by one float32 ulp if 1 - closed is kept in float64.
        near(total(&r.closed), 208.33992726786528, 1e-9);
        near(total(&r.open), 53.362559364235494, 1e-9);
        near(total(&r.street), 49.503835602667095, 1e-9);
    }

    #[test]
    fn pairwise_sum_split_above_128_rounds_down_to_a_multiple_of_eight() {
        // numpy: np.sum(float32 field) = 10.861397 for 1000 elements (splitting at a multiple of 4
        // instead of 8 gives 10.861395) and 1.7498927 for 777 (1.7498908).
        assert_eq!(pairwise_sum(&field(1000, 81)), 10.861397_f32);
        assert_eq!(pairwise_sum(&field(777, 82)), 1.7498927_f32);
    }

    #[test]
    fn row_means_are_numpys_pairwise_float64_sums() {
        // At 1000 columns a sequential float64 sum lands a few last bits away from numpy's pairwise one
        // (0.6912443980825637 against 0.6912443980825627 on the second row).
        let (h, w) = (3, 1000);
        let s = scaled(field(h * w, 71), 1.5);
        let (fine, streak) = (field(h * w, 72), field(h * w, 73));
        let sub = scaled(field(h * w, 74), 0.3);
        let col = [0.2, -0.3, 0.0];
        let means =
            mean_rows(&s, &fine, h, w, Threshold::Column(&col), &GenParams::default(), Streak::Array(&streak[..]), Some(&sub));
        assert_eq!(means, vec![0.5458349015250588, 0.6912443980825627, 0.5951318769647173]);
    }

    #[test]
    fn surface_regime_weights_keep_the_slightly_negative_shares_the_source_leaves() {
        // closed 0.2 + (1 - 0.2) deck with (1 - closed) rounded to float32 (0.8000000119) reaches
        // 1.0000000149 on a full deck: the scale (1 - closed) / rest is slightly negative and so are the
        // shares, which the source does not clamp (numpy: open -9.934108e-09, street -4.967054e-09).
        let lat = [21.0_f64 * (PI / 180.0)];
        let r = surface_regime_weights(&[0.2_f32], &[0.5_f32], &[0.25_f32], &[0.0], &[1.0], &lat, 1, 30.0);
        assert_eq!(r.closed, vec![1.0_f32]);
        assert_eq!(r.open, vec![-9.934108e-9_f32]);
        assert_eq!(r.street, vec![-4.967054e-9_f32]);
    }

    #[test]
    fn compose_rounds_every_parameter_to_the_dtype_of_the_operand_it_meets() {
        // None of these is exactly representable in float32, so a parameter held in the wrong precision
        // moves the sums (numpy: A 29.81360345931972 with a float64 threshold column, B 29.200988456606865
        // in float32 with a Python-float threshold).
        let prm = GenParams {
            shield_ramp: 0.1,
            shield_texture: 0.6,
            cumulus_max: 0.8,
            cumulus_depth: 0.5,
            a_fine: 1.4,
            ..GenParams::default()
        };
        let (h, w) = (3, 20);
        let s = scaled(field(h * w, 21), 1.5);
        let fine = field(h * w, 22);
        let streak = field(h * w, 23);
        let sub = scaled(field(h * w, 24), 0.3);
        let col = [0.31, -0.17, 0.05];
        let arr = Streak::Array(&streak[..]);
        let a = compose(&s, &fine, h, w, Threshold::Column(&col), &prm, arr, Some(&sub));
        near(sum_f64(a.to_f64()), 29.81360345931972, 1e-11);
        let b = compose(&s, &fine, h, w, Threshold::Scalar(0.07), &prm, arr, Some(&sub));
        near(sum_f64(b.to_f64()), 29.200988456606865, 1e-11);
    }
}

// ---------------------------------------------------------------------------------------------------------
// Unit tests of the pipeline: what the golden cases cannot say because they hand it numpy's draws (the port's
// own plans, the entry points, the error path) or freeze only a few scenarios.

#[cfg(test)]
mod pipeline_tests {
    use super::*;
    use crate::texture::regime_texture;
    use crate::climate::EARTH;
    use crate::texture::used_cells;

    const W: usize = 128;

    fn earth(seed: u64) -> GenOutput {
        generate(&EARTH, seed, W, &GenParams::default(), None, true).expect("Earth is bakeable")
    }

    fn ocean(h: usize, w: usize) -> Surface {
        Surface::from_elevation(&vec![-1.0_f32; h * w], h, w, 0.0, "test ocean")
    }

    /// A world with a continent in the northern middle latitudes (0.8 km high) and sea elsewhere.
    fn continent(h: usize, w: usize) -> Surface {
        let elev: Vec<f32> = (0..h * w)
            .map(|i| {
                let (r, c) = (i / w, i % w);
                if (h / 8..h / 2).contains(&r) && (w / 4..w / 2).contains(&c) {
                    0.8
                } else {
                    -1.0
                }
            })
            .collect();
        Surface::from_elevation(&elev, h, w, 0.0, "test continent")
    }

    #[test]
    fn a_body_whose_clouds_are_not_water_is_refused_by_every_entry_point() {
        let mars = BodyInputs::new(Some(1.026), 3389.5, 3.721, 11.1).with_regime("thin water ice");
        let prm = GenParams::default();
        let want = GenerateError::NotBakeable { cloud_regime: "thin water ice".to_string() };
        assert_eq!(generate(&mars, 1, 16, &prm, None, false).unwrap_err(), want);
        let plans = GeneratePlans::from_seed(&mars, 1, 16, &prm);
        assert_eq!(generate_with_plans(&mars, &plans, 16, &prm, None, false, false).unwrap_err(), want);
        // The message names the regime and the escape hatch.
        let msg = want.to_string();
        assert!(msg.contains("thin water ice") && msg.contains("allow_any_regime"), "{msg}");
        // The experiment flag lets it through; "Water" and " water" are not "water".
        let allow = GenParams { allow_any_regime: true, ..prm };
        assert!(generate(&mars, 1, 16, &allow, None, false).is_ok());
        for regime in ["Water", " water", "water ", ""] {
            let b = BodyInputs::new(Some(1.0), 6371.0, 9.81, 8.0).with_regime(regime.to_string());
            assert!(generate(&b, 1, 16, &prm, None, false).is_err(), "regime {regime:?}");
        }
    }

    #[test]
    fn the_refusal_comes_before_any_draw_and_before_the_width_is_looked_at() {
        // Width 1 would panic in the pipeline, the regime check does not get that far.
        let mars = BodyInputs::new(Some(1.026), 3389.5, 3.721, 11.1).with_regime("thin water ice");
        assert!(generate(&mars, 1, 1, &GenParams::default(), None, false).is_err());
    }

    #[test]
    #[should_panic(expected = "at least 2 columns")]
    fn a_map_needs_a_row() {
        let _ = generate(&EARTH, 1, 1, &GenParams::default(), None, false);
    }

    #[test]
    fn the_same_seed_gives_the_same_map_and_another_seed_another() {
        let (a, b, c) = (earth(7), earth(7), earth(8));
        assert_eq!(a.coverage, b.coverage);
        assert_eq!(a.extras, b.extras);
        assert_ne!(a.coverage, c.coverage);
        assert_eq!(a.coverage.len(), (W / 2) * W);
        assert!(a.coverage.iter().all(|v| (0.0..=1.0).contains(v)), "coverage is a fraction");
    }

    #[test]
    fn generate_is_the_plans_of_its_seed_through_generate_with_plans() {
        let prm = GenParams::default();
        let direct = generate(&EARTH, 5, W, &prm, None, true).unwrap();
        let plans = GeneratePlans::from_seed(&EARTH, 5, W, &prm);
        let via = generate_with_plans(&EARTH, &plans, W, &prm, None, true, false).unwrap();
        assert_eq!(direct.coverage, via.coverage);
        assert_eq!(direct.extras, via.extras);
        // The plans do not know the seed: only generate() reports it.
        assert_eq!((direct.meta.seed, via.meta.seed), (5, 0));
        assert_eq!(GenMeta { seed: 0, ..direct.meta.clone() }, via.meta);
    }

    #[test]
    fn the_plans_follow_the_parameters() {
        let prm = GenParams::default();
        let p = GeneratePlans::from_seed(&EARTH, 3, W, &prm);
        assert!(matches!(p.texture, TexturePlans::Regime(_)));
        assert_eq!(p.moisture.octaves.len(), MOISTURE_OCTAVES);
        assert_eq!(p.mid.octaves.len(), MID_OCTAVES);
        assert_eq!(p.moisture.octaves[0].freq, 1.5);
        assert_eq!(p.mid.octaves[0].freq, 6.0);
        let ladder = fine_ladder(W, &prm);
        assert_eq!(p.streak.octaves.len(), ladder.octaves);
        assert_eq!(p.streak.octaves[0].freq, ladder.frequency);
        assert_eq!(p.subtexel.as_ref().map(Vec::len), Some((W / 2) * W));
        assert!(!p.flow.vortices.is_empty(), "Earth has a storm track");

        // regime_texture off: one plain fBm with the fine gain and the billow flag.
        let off = GenParams { regime_texture: false, fine_billow: true, fine_gain: 0.4, ..prm };
        let q = GeneratePlans::from_seed(&EARTH, 3, 4096, &off);
        match &q.texture {
            TexturePlans::Plain(f) => {
                assert!(f.billow);
                assert_eq!(f.octaves.len(), fine_ladder(4096, &off).octaves);
                assert_eq!(f.octaves.len(), 3);
                assert_eq!(f.octaves[1].amp, 0.4);
            }
            other => panic!("{other:?}"),
        }
        // cumulus_cell_km = 0 turns the sub-texel field off; isotropic swaps the flow.
        let flat = GenParams { cumulus_cell_km: 0.0, isotropic: true, ..prm };
        let r = GeneratePlans::from_seed(&EARTH, 3, W, &flat);
        assert!(r.subtexel.is_none());
        assert_eq!((r.flow.jet, r.flow.trades, r.flow.polar_easterlies, r.flow.vortices.len()), (0.0, 0.0, 0.0, 0));
        assert_eq!(r.flow, isotropic_flow(&EARTH));
    }

    #[test]
    fn the_subtexel_noise_is_unit_white_noise_over_the_cloud_count() {
        let prm = GenParams::default();
        // sqrt(n_clouds) = texel / cell: a 1024 map has 39.1 km texels, over clouds of 1.4 km.
        let scale = subtexel_scale(&EARTH, 1024, &prm).unwrap();
        let texel = 2.0 * PI * 6371.0 / 1024.0;
        assert!((f64::from(scale) - texel / 1.4).abs() < 1e-5 * texel / 1.4, "{scale}");
        let plans = GeneratePlans::from_seed(&EARTH, 9, 128, &prm);
        let sub = plans.subtexel.unwrap();
        let s128 = subtexel_scale(&EARTH, 128, &prm).unwrap();
        let (n, scaled) = (sub.len() as f64, |v: f32| f64::from(v * s128));
        let mean = sub.iter().map(|&v| scaled(v)).sum::<f64>() / n;
        let std = (sub.iter().map(|&v| (scaled(v) - mean).powi(2)).sum::<f64>() / n).sqrt();
        assert!((std - 1.0).abs() < 0.05 && mean.abs() < 0.05, "unit normal before the scale: mean {mean}, std {std}");
        // A texel smaller than a cloud still holds one: the divisor floors at 1 (the field IS the white noise).
        assert_eq!(subtexel_scale(&EARTH, 128, &GenParams { cumulus_cell_km: 4000.0, ..prm }), Some(1.0));
        // 0, a negative size and NaN switch the term off.
        for off in [0.0, -1.4, f64::NAN] {
            assert_eq!(subtexel_scale(&EARTH, 128, &GenParams { cumulus_cell_km: off, ..prm }), None);
        }
        assert_eq!(subtexel_from_white(&[2.0, -3.0, 0.5], 4.0), vec![0.5, -0.75, 0.125]);
    }

    #[test]
    fn the_fine_ladder_is_anchored_at_the_top_of_the_maps_limit() {
        let prm = GenParams::default();
        let l = fine_ladder(1024, &prm);
        assert!((l.fmax - 1024.0 / (4.0 * PI)).abs() < 1e-12);
        assert_eq!(l.octaves, 1, "0.95 x 81.5 = 77.4 cycles per radian does not reach a second octave above 48");
        // A 4096 map carries more detail than a 1024 one: three octaves, the last at 0.95 of the limit.
        let big = fine_ladder(4096, &prm);
        assert_eq!(big.octaves, 3);
        assert!((big.frequency * 2.03_f64.powi(2) - 0.95 * big.fmax).abs() < 1e-9 * big.fmax);
        // The regime texture is handed fmax * fine_top / 0.95: its own 0.95 anchor then lands on fine_top.
        let t = texture_fmax(big.fmax, &GenParams { fine_top: 0.8, ..prm });
        assert!((0.95 * t - 0.8 * big.fmax).abs() < 1e-9 * big.fmax);
    }

    /// A small water world: its convective cells (15 to 120 km, so 67 to 8 per radian of arc) are resolved by
    /// maps a few hundred texels wide, which Earth's (a 6371 km radius) are not.
    fn small_world() -> BodyInputs {
        BodyInputs::new(Some(1.0), 1000.0, 9.81, 8.0).with_regime("water")
    }

    /// `GeneratePlans::from_seed` against `generate()` of `generate.py`, spelled with the literals of that file
    /// (`fbm(back, 1.5, 3, seed + 1)`, `fbm(.., mid_frequency, 2, seed + 2)`, the fine texture at `seed + 3`, the
    /// streak `fbm(back, f_fine, n_fine, seed + 4, gain=0.7)`, the sub-texel noise at `seed + 7`, the regime
    /// texture handed `fmax * fine_top / 0.95`) and NOT with the constants of this module. Every plan must be
    /// the draw of its own seeded generator, so a plan is compared whole (octave schedule, norm, lattice
    /// tables). Returns the octave count of the fine ladder, so a caller can check the case is sensitive to the
    /// gains (a one-octave ladder has no gain to get wrong).
    fn assert_plans_follow_generate_py(body: &BodyInputs, seed: u64, width: usize, prm: &GenParams) -> usize {
        let got = GeneratePlans::from_seed(body, seed, width, prm);
        let lac = DEFAULT_LACUNARITY;
        let fmax = width as f64 / (2.0 * PI * prm.fine_min_texels);
        let (f_fine, n_fine) = octave_ladder(prm.fine_frequency, fmax, lac, prm.fine_top);
        assert_eq!(got.moisture, FbmPlan::from_seed(seed + 1, 1.5, 3, DEFAULT_GAIN, lac, false), "moisture");
        assert_eq!(got.mid, FbmPlan::from_seed(seed + 2, prm.mid_frequency, 2, DEFAULT_GAIN, lac, false), "mid");
        match &got.texture {
            TexturePlans::Regime(t) => {
                assert!(prm.regime_texture);
                let anchored = fmax * prm.fine_top / 0.95;
                let want = RegimeTexturePlans::from_seed(body.radius_km, anchored, seed, prm.fine_frequency, prm.fine_gain);
                assert_eq!(*t, want, "regime texture");
                assert_eq!(t.cells.len(), used_cells(body.radius_km, anchored).len(), "one cell plan per resolved diameter");
            }
            TexturePlans::Plain(f) => {
                assert!(!prm.regime_texture);
                assert_eq!(*f, FbmPlan::from_seed(seed + 3, f_fine, n_fine, prm.fine_gain, lac, prm.fine_billow), "plain fine");
            }
        }
        assert_eq!(got.streak, FbmPlan::from_seed(seed + 4, f_fine, n_fine, 0.7, lac, false), "streak");
        let want_sub = (prm.cumulus_cell_km > 0.0).then(|| {
            let texel_km = 2.0 * PI * body.radius_km / width as f64;
            let n_clouds = (texel_km / prm.cumulus_cell_km).powi(2).max(1.0);
            let scale = n_clouds.sqrt() as f32;
            let mut rng = Rng::new(seed + 7);
            (0..width / 2 * width).map(|_| rng.normal() as f32 / scale).collect::<Vec<f32>>()
        });
        assert_eq!(got.subtexel, want_sub, "sub-texel noise");
        let want_flow = if prm.isotropic {
            Flow { hadley_deg: hadley_edge_deg(body), vortices: Vec::new(), jet: 0.0, trades: 0.0, polar_easterlies: 0.0 }
        } else {
            build_flow(body, seed)
        };
        assert_eq!(got.flow, want_flow, "flow");
        n_fine
    }

    #[test]
    fn from_seed_draws_every_plan_as_generate_py_seeds_it() {
        let dflt = GenParams::default();
        let mut ladders = Vec::new();
        // Earth, as the stat cases run it (a single fine octave there) and as a 2048 map does (two).
        for (seed, width) in [(1, 128), (9, 1024), (4, 2048)] {
            let prm = GenParams { cumulus_cell_km: if width > 1024 { 0.0 } else { dflt.cumulus_cell_km }, ..dflt };
            ladders.push(assert_plans_follow_generate_py(&EARTH, seed, width, &prm));
        }
        // Every field of the parameters that reaches a plan, off the defaults, on a world whose cells are resolved,
        // with a ladder of three octaves (a gain, a frequency or an anchor that is ignored changes the plans).
        let odd = GenParams { fine_top: 0.7, fine_gain: 0.4, fine_frequency: 8.0, mid_frequency: 9.0, fine_min_texels: 1.5, ..dflt };
        ladders.push(assert_plans_follow_generate_py(&small_world(), 5, 512, &odd));
        // The plain fBm of regime_texture = false (its own gain and billow flag), a retrograde body, no sub-texel term.
        let plain = GenParams { regime_texture: false, fine_billow: true, fine_gain: 0.4, fine_frequency: 8.0, cumulus_cell_km: 0.0, ..dflt };
        let retro = BodyInputs::new(Some(-2.5), 3000.0, 6.0, 20.0).with_regime("water");
        ladders.push(assert_plans_follow_generate_py(&retro, 11, 512, &plain));
        // The isotropic flow.
        ladders.push(assert_plans_follow_generate_py(&small_world(), 2, 256, &GenParams { isotropic: true, ..odd }));
        // The gains show only where a ladder has a second octave: the stat cases' 1024 map has one, the others more.
        assert_eq!(ladders, [1, 1, 2, 3, 3, 2]);
    }

    #[test]
    fn the_regime_texture_is_anchored_at_fine_top_inside_the_pipeline() {
        // regime_texture anchors its own ladder at 0.95, so generate() hands it fmax * fine_top / 0.95: the
        // cell diameters it resolves follow fine_top. fmax of a 128 map is 10.19 cycles per radian, and the cells
        // of a 1000 km world are 67, 33, 17 and 8.3 per radian (15, 30, 60 and 120 km).
        let (width, body) = (128, small_world());
        let p = grid_dirs::<f32>(width, width / 2);
        for (fine_top, want_cells) in [(0.5, 0), (0.95, 1), (1.7, 2)] {
            let prm = GenParams { fine_top, ..GenParams::default() };
            let plans = GeneratePlans::from_seed(&body, 3, width, &prm);
            let TexturePlans::Regime(tp) = &plans.texture else { panic!("regime texture plans") };
            assert_eq!(tp.cells.len(), want_cells, "fine_top {fine_top}");
            let out = generate_with_plans(&body, &plans, width, &prm, None, false, true).expect("a water world");
            let t = out.trace.expect("traced");
            // The texture of the trace is regime_texture at the independently derived fmax and nothing else.
            let fmax = width as f64 / (2.0 * PI * 2.0) * fine_top / 0.95;
            let w = match t.weights.as_ref().expect("the regime weights") {
                TraceWeights::F64(w) => w,
                TraceWeights::F32(_) => panic!("no surface: the weights are float64"),
            };
            assert_eq!(t.fine, regime_texture(&p, &t.back, w, body.radius_km, fmax, tp), "fine_top {fine_top}");
        }
    }

    #[test]
    fn the_local_target_is_clipped_to_one_and_ninety_nine_percent() {
        // np.clip(target + anomaly, 0.01, 0.99), the anomaly a float32 widened.
        assert_eq!((LOCAL_TARGET_MIN, LOCAL_TARGET_MAX), (0.01, 0.99));
        assert_eq!(local_target(0.97, 0.05), 0.99);
        assert_eq!(local_target(0.99, 0.0), 0.99);
        assert_eq!(local_target(1.5, -0.1), 0.99);
        assert_eq!(local_target(0.02, -0.05), 0.01);
        assert_eq!(local_target(0.0, 0.0), 0.01);
        // Inside the band it is the plain sum in float64 (0.3f32 is not 0.3).
        assert_eq!(local_target(0.5, 0.3), 0.5 + f64::from(0.3_f32));
        assert_eq!(local_target(0.5, -0.25), 0.25);
        assert!(local_target(0.5, f32::NAN).is_nan(), "NaN propagates like np.clip");
    }

    #[test]
    fn the_pipelines_local_target_is_the_clipped_sum_and_reaches_the_floor() {
        // Ridges across a world: a strong anomaly where the air is lifted or sheltered.
        let (h, w) = (32, 64);
        let elev: Vec<f32> = (0..h * w).map(|i| if (i % w) % 4 < 2 { 6.0 } else { -1.0 }).collect();
        let surf = Surface::from_elevation(&elev, h, w, 0.0, "test ridges");
        let prm = GenParams::default();
        let plans = GeneratePlans::from_seed(&EARTH, 1, W, &prm);
        let t = generate_with_plans(&EARTH, &plans, W, &prm, Some(&surf), false, true).unwrap().trace.unwrap();
        let (local, terms) = (t.target_local.as_ref().expect("surface terms ran"), t.terms.as_ref().unwrap());
        for (i, &v) in local.iter().enumerate() {
            assert_eq!(v, local_target(t.target_zonal[i / W], terms.anomaly[i]), "texel {i}");
        }
        assert!(local.iter().all(|v| (LOCAL_TARGET_MIN..=LOCAL_TARGET_MAX).contains(v)));
        // Cloud-free climates are asked for 1 %, not for nothing: the floor binds.
        assert_eq!(local.iter().copied().fold(f64::INFINITY, f64::min), LOCAL_TARGET_MIN);
        // The row target is the row mean of the local target (the offsets are measured from it).
        for r in 0..W / 2 {
            let mean = local[r * W..(r + 1) * W].iter().sum::<f64>() / W as f64;
            assert!((t.target[r] - mean).abs() < 1e-12, "row {r}");
        }
    }

    #[test]
    fn the_surface_terms_are_computed_at_1024_columns_at_most() {
        // surface_terms(surface, width, ...) keeps its default term_width=1024: the smooth terms are climatological
        // (~40 km on Earth), computed at min(1024, width) and upsampled, so a 2048 map does not pay for them twice
        // over. A cheap 2048 run (plain fine texture, no sub-texel noise, a coarse back-trace) shows the terms
        // are those of a 1024 grid, not of 512 or 2048.
        let (h, w) = (32, 64);
        let elev: Vec<f32> = (0..h * w).map(|i| if (i % w) % 5 < 2 { 3.0 } else { -1.0 }).collect();
        let surf = Surface::from_elevation(&elev, h, w, 0.0, "test ridges");
        let (width, structure) = (2048, 64);
        let prm = GenParams { structure_width: structure, regime_texture: false, cumulus_cell_km: 0.0, ..GenParams::default() };
        let plans = GeneratePlans::from_seed(&EARTH, 1, width, &prm);
        let got = generate_with_plans(&EARTH, &plans, width, &prm, Some(&surf), false, true).unwrap().trace.unwrap().terms.unwrap();

        // The wind the pipeline gives them: the map of min(2 structure_width, width) columns.
        let wind = wind_map(&plans.flow, &EARTH, 2 * structure);
        let grid = WindGrid { uv: &wind, h: structure, w: 2 * structure };
        let spin = 1.0; // Earth turns prograde
        let at = |term_width: usize| surface_terms(&surf, width, 30.0, EARTH.radius_km, spin, Some(&grid), term_width);
        assert_eq!(got, at(1024));
        assert_ne!(got.anomaly, at(512).anomaly, "a coarser grid draws other terms");
        assert_ne!(got.anomaly, at(2048).anomaly, "so does a finer one");
        assert_eq!(SURFACE_TERM_WIDTH, 1024);
    }

    #[test]
    fn extras_are_returned_only_when_asked_and_sit_on_the_right_grids() {
        let prm = GenParams::default();
        let bare = generate(&EARTH, 2, W, &prm, None, false).unwrap();
        assert!(bare.extras.is_none() && bare.trace.is_none());
        let full = earth(2);
        assert_eq!(bare.coverage, full.coverage, "asking for the extras does not change the coverage");
        let e = full.extras.unwrap();
        let n = (W / 2) * W;
        assert_eq!((e.kind.len(), e.high_share.len()), (n, n));
        // The wind grid is min(2 structure_width, width) wide: here the map itself.
        assert_eq!((e.wind_width, e.wind_height, e.wind.len()), (W, W / 2, W * W / 2 * 2));
        assert!(e.kind.iter().chain(&e.high_share).all(|v| (0.0..=1.0).contains(v)));
        let narrow = generate(&EARTH, 2, W, &GenParams { structure_width: 32, ..prm }, None, true).unwrap().extras.unwrap();
        assert_eq!((narrow.wind_width, narrow.wind_height), (64, 32), "2 structure_width = 64 < 128");
    }

    #[test]
    fn the_trace_holds_every_stage_and_agrees_with_the_result() {
        let prm = GenParams::default();
        let plans = GeneratePlans::from_seed(&EARTH, 11, W, &prm);
        let out = generate_with_plans(&EARTH, &plans, W, &prm, None, true, true).unwrap();
        let t = out.trace.as_ref().unwrap();
        let n = (W / 2) * W;
        assert_eq!(t.coverage, out.coverage);
        assert_eq!((t.back.len(), t.lat_back.len(), t.s.len(), t.s_raw.len(), t.fine.len(), t.streak.len()), (n, n, n, n, n, n));
        // structure_width 512 >= 128: the back-trace grid is the map's own.
        let bc = t.back_coarse.as_ref().unwrap();
        assert_eq!((bc.height, bc.width), (W / 2, W));
        assert_eq!(bc.data, t.back);
        // s is standardized over the whole map; the weights are float64 without a surface.
        let mean = t.s.iter().map(|&v| f64::from(v)).sum::<f64>() / n as f64;
        let var = t.s.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 1e-5 && (var - 1.0).abs() < 1e-4, "s: mean {mean}, variance {var}");
        assert!(matches!(t.weights, Some(TraceWeights::F64(_))));
        assert_eq!(t.subtexel, plans.subtexel);
        // Without a surface there is no terms stage.
        assert!(t.terms.is_none() && t.target_local.is_none() && t.thr0.is_none() && t.sensitivity.is_none() && t.offset.is_none());
        assert_eq!(t.target, t.target_zonal, "the bisection's target is the zonal one");
        assert_eq!((t.thr_raw.len(), t.thr.len(), t.target.len()), (W / 2, W / 2, W / 2));
        assert_eq!(t.shield.as_ref().map(Vec::len), Some(n));
        assert_eq!(t.kind, out.extras.as_ref().map(|e| e.kind.clone()));
        assert_eq!(t.wind, out.extras.as_ref().map(|e| e.wind.clone()));
        // Without extras the derived stages are absent from the trace too.
        let lean = generate_with_plans(&EARTH, &plans, W, &prm, None, false, true).unwrap();
        let lt = lean.trace.unwrap();
        assert!(lt.shield.is_none() && lt.kind.is_none() && lt.high_share.is_none() && lt.wind.is_none());
        assert_eq!(lt.coverage, out.coverage);
        // The weights of the plain fine texture (regime_texture = false) stay out of the trace even when the
        // extras need them: Python only traces them from the regime branch.
        let plain = GenParams { regime_texture: false, ..prm };
        let plain_plans = GeneratePlans::from_seed(&EARTH, 11, W, &plain);
        let p = generate_with_plans(&EARTH, &plain_plans, W, &plain, None, true, true).unwrap();
        assert!(p.trace.as_ref().unwrap().weights.is_none() && p.extras.is_some());
    }

    #[test]
    #[should_panic(expected = "texture plan")]
    fn a_texture_plan_of_the_other_kind_is_refused() {
        let prm = GenParams::default();
        let plain = GenParams { regime_texture: false, ..prm };
        let plans = GeneratePlans::from_seed(&EARTH, 1, 16, &plain);
        let _ = generate_with_plans(&EARTH, &plans, 16, &prm, None, false, false);
    }

    #[test]
    #[should_panic(expected = "sub-texel plan")]
    fn a_subtexel_plan_of_the_wrong_size_is_refused() {
        let prm = GenParams::default();
        let mut plans = GeneratePlans::from_seed(&EARTH, 1, 16, &prm);
        plans.subtexel = Some(vec![0.0; 3]);
        let _ = generate_with_plans(&EARTH, &plans, 16, &prm, None, false, false);
    }

    #[test]
    fn the_calibration_puts_every_rows_mean_on_its_target() {
        let out = earth(1);
        let lat = grid_lonlat(W, W / 2).1;
        let t_rows: Vec<f64> = lat.iter().map(|&la| target_zonal_coverage(la * (180.0 / PI), 30.0, 0.0)).collect();
        // The 2-degree smoothing of the thresholds, the sub-texel noise and 128 texels per row: within a few
        // percent away from the poles (a pole row is one point).
        let mut worst = 0.0_f64;
        for r in 0..W / 2 {
            if (lat[r] * 180.0 / PI).abs() > 70.0 {
                continue;
            }
            let mean = out.coverage[r * W..(r + 1) * W].iter().map(|&v| f64::from(v)).sum::<f64>() / W as f64;
            worst = worst.max((mean - t_rows[r]).abs());
        }
        assert!(worst < 0.08, "worst row residual {worst}");
    }

    #[test]
    fn isotropic_runs_without_a_back_trace_and_ignores_the_surface() {
        let prm = GenParams { isotropic: true, ..GenParams::default() };
        let plans = GeneratePlans::from_seed(&EARTH, 4, W, &prm);
        let surf = continent(16, 32);
        let out = generate_with_plans(&EARTH, &plans, W, &prm, Some(&surf), true, true).unwrap();
        let t = out.trace.as_ref().unwrap();
        assert!(t.back_coarse.is_none() && t.terms.is_none() && out.meta.surface.is_none());
        assert_eq!(t.back, grid_dirs::<f32>(W, W / 2), "isotropic: the air came from where it is");
        assert_eq!((out.meta.flow.jet, out.meta.flow.vortices.len()), (0.0, 0));
        // The wind map is still produced (the climatological zonal wind alone: no vortex).
        assert!(out.extras.is_some());
    }

    #[test]
    fn a_surface_moves_the_cloud_amount_where_it_acts_and_is_reported_in_the_meta() {
        let prm = GenParams::default();
        let surf = continent(16, 32);
        let blind = generate(&EARTH, 6, W, &prm, None, true).unwrap();
        let withs = generate(&EARTH, 6, W, &prm, Some(&surf), true).unwrap();
        assert_ne!(blind.coverage, withs.coverage);
        let m = withs.meta.surface.as_ref().expect("the surface terms ran");
        assert_eq!(m.source, "test continent");
        assert!(m.land_fraction > 0.0 && m.land_fraction < 0.2, "{}", m.land_fraction);
        assert_eq!(m.land_fraction, py_round(m.land_fraction, 4), "the sidecar value is rounded to 4 decimals");
        assert_eq!(m.itcz_shift_deg, py_round(m.itcz_shift_deg, 2));
        assert!(m.anomaly_range[0] < 0.0 && m.anomaly_range[1] > 0.0, "{:?}", m.anomaly_range);
        // The same surface with the terms off (A/B) is ignored: the blind generator.
        let off = generate(&EARTH, 6, W, &GenParams { surface_terms: false, ..prm }, Some(&surf), true).unwrap();
        assert_eq!(off.coverage, blind.coverage, "surface_terms = false is the blind generator");
        assert!(off.meta.surface.is_none());
        // The trace of the surface run has the whole terms stage, and the float32 weights of the reshaped regimes.
        let plans = GeneratePlans::from_seed(&EARTH, 6, W, &prm);
        let traced = generate_with_plans(&EARTH, &plans, W, &prm, Some(&surf), true, true).unwrap();
        let t = traced.trace.unwrap();
        assert!(t.terms.is_some() && t.target_local.is_some() && t.thr0.is_some() && t.sensitivity.is_some() && t.offset.is_some());
        assert!(matches!(t.weights, Some(TraceWeights::F32(_))));
        assert!(t.target != t.target_zonal, "the target is the row mean of the local target");
        assert_eq!(traced.coverage, withs.coverage);
    }

    #[test]
    fn a_world_with_no_land_has_no_relief_term_and_no_itcz_shift() {
        let prm = GenParams::default();
        let plans = GeneratePlans::from_seed(&EARTH, 1, 64, &prm);
        let out = generate_with_plans(&EARTH, &plans, 64, &prm, Some(&ocean(16, 32)), false, true).unwrap();
        let m = out.meta.surface.as_ref().expect("terms ran");
        assert_eq!((m.land_fraction, m.itcz_shift_deg), (0.0, 0.0));
        let terms = out.trace.unwrap().terms.unwrap();
        assert!(terms.orographic.iter().all(|&v| v == 0.0), "flat ground lifts nothing");
        assert!(terms.continentality.iter().all(|&v| v == 0.0) && terms.deck_strength.iter().all(|&v| v == 0.0));
        // The anomaly is a departure from Earth's own mixture: a world with no land is drier than Earth's
        // land belts, so it is not zero (the range sits around the departure of the land contrast).
        assert!(m.anomaly_range[0] < 0.0 && m.anomaly_range[1] > 0.0, "{:?}", m.anomaly_range);
    }

    #[test]
    fn the_smallest_map_is_one_row_of_two_texels() {
        let out = generate(&EARTH, 1, 2, &GenParams::default(), Some(&ocean(2, 4)), true).unwrap();
        assert_eq!((out.height, out.width, out.coverage.len()), (1, 2, 2));
        assert_eq!(out.extras.as_ref().unwrap().kind.len(), 2);
    }

    #[test]
    fn the_flow_summary_lists_the_vortices_in_degrees() {
        let out = earth(3);
        let plans = GeneratePlans::from_seed(&EARTH, 3, W, &GenParams::default());
        assert_eq!(out.meta.flow.vortices.len(), plans.flow.vortices.len());
        for (m, v) in out.meta.flow.vortices.iter().zip(&plans.flow.vortices) {
            assert_eq!((m.kind, m.radius, m.omega, m.ascent, m.age), (v.kind, v.radius, v.omega, v.ascent, v.age));
            assert!((m.lon_deg - v.lon * 180.0 / PI).abs() < 1e-9 && (m.lat_deg - v.lat * 180.0 / PI).abs() < 1e-9);
        }
        assert_eq!((out.meta.hadley_edge_deg, out.meta.orientation), (30.0, ORIENTATION));
        assert_eq!(out.meta.fine_octaves, 1);
    }

    #[test]
    fn python_rounding_is_decimal_and_ties_to_even() {
        // Python: round(0.125, 2) = 0.12 (an exact tie), round(0.375, 2) = 0.38, round(2.5) = 2.0,
        // round(1.005, 2) = 1.0 (the binary value is below the tie), round(-0.0004, 3) = -0.0.
        assert_eq!(py_round(0.125, 2), 0.12);
        assert_eq!(py_round(0.375, 2), 0.38);
        assert_eq!(py_round(2.5, 0), 2.0);
        assert_eq!(py_round(3.5, 0), 4.0);
        assert_eq!(py_round(1.005, 2), 1.0);
        assert_eq!(py_round(0.123456, 4), 0.1235);
        assert!(py_round(-0.0004, 3) == 0.0 && py_round(-0.0004, 3).is_sign_negative());
        assert!(py_round(f64::NAN, 2).is_nan());
        assert_eq!(py_round(f64::INFINITY, 2), f64::INFINITY);
    }

    #[test]
    fn extrema_propagate_nan_like_numpy() {
        assert_eq!(extremum(&[0.5, -2.0, 3.0], true), -2.0);
        assert_eq!(extremum(&[0.5, -2.0, 3.0], false), 3.0);
        assert!(extremum(&[0.5, f32::NAN, 3.0], true).is_nan());
        assert!(extremum(&[], false).is_nan());
    }

    #[test]
    fn the_surface_regime_weights_of_float64_regimes_do_not_leak_a_float32_rounding() {
        // The pipeline feeds float64 weights: closed 0.2 + (1 - 0.2) x a full deck is exactly 1, so the
        // other regimes get a scale of exactly 0 (the float32 chain of the golden cases leaves -9.9e-9).
        let lat = [21.0_f64 * (PI / 180.0)];
        let r64 = surface_regime_weights(&[0.2_f64], &[0.5], &[0.25], &[0.0], &[1.0], &lat, 1, 30.0);
        assert_eq!((r64.closed[0], r64.open[0].abs(), r64.street[0].abs()), (1.0, 0.0, 0.0));
        let r32 = surface_regime_weights(&[0.2_f32], &[0.5], &[0.25], &[0.0], &[1.0], &lat, 1, 30.0);
        assert!(r32.open[0] < 0.0, "the float32 chain keeps the slightly negative share");
        // Away from that edge the two precisions agree to float32 rounding.
        let a = surface_regime_weights(&[0.3_f64], &[0.4], &[0.2], &[0.25], &[0.6], &lat, 1, 30.0);
        let b = surface_regime_weights(&[0.3_f32], &[0.4], &[0.2], &[0.25], &[0.6], &lat, 1, 30.0);
        for (x, y) in [(a.closed[0], b.closed[0]), (a.open[0], b.open[0]), (a.street[0], b.street[0])] {
            assert!((x - y).abs() < 1e-6, "{x} vs {y}");
        }
    }
}
