//! Port of `cloudgen/climate.py`: the climate bands of one body (Hadley edge, storm track, deformation
//! radius, the convective tendency and the target zonal cloud fraction, the map width).
//!
//! Python's `float32`/`float64` numpy arrays are `f32`/`f64` here. A numpy function that runs in the dtype of
//! its input (`convective_tendency`, `target_zonal_coverage`) is generic over `f32`/`f64`, takes ONE latitude
//! (call it per element) and runs in that precision; a Python float argument (the Hadley edge, the ITCZ shift)
//! is "weak" in numpy 2 (NEP 50), so it is rounded to the precision of the array it meets, which the code
//! below does literally with [`lit`]. Python's `max`/`min` keep their first argument when the comparison is
//! false (so a NaN operand loses), which [`py_max`]/[`py_min`] copy; `np.clip` propagates NaN, which
//! [`np_clip`] copies.
//!
//! Where Python raises (a division by an exact zero in `hadley_edge_raw_deg`, `math.ceil` of a NaN or an
//! infinity in `map_width_for_texel`) the port returns the IEEE value instead and says so in the doc.

use num_traits::Float;
use std::borrow::Cow;

pub const EARTH_ROTATION_DAYS: f64 = 0.99727;
pub const EARTH_RADIUS_KM: f64 = 6371.0;
pub const EARTH_GRAVITY: f64 = 9.81;
pub const EARTH_SCALE_HEIGHT_KM: f64 = 8.0;
pub const EARTH_HADLEY_EDGE_DEG: f64 = 30.0;
pub const HADLEY_MIN_DEG: f64 = 10.0;
pub const HADLEY_MAX_DEG: f64 = 90.0;

/// Earth's first baroclinic Rossby deformation radius as a fraction of `phi_H * a` (see climate.py).
pub const DEFORMATION_FRACTION_OF_HADLEY: f64 = 0.30;
pub const EARTH_STORM_TRACK_DEG: f64 = 51.0;
pub const STORM_TRACK_IN_HADLEY: f64 = 1.7;
pub const STORM_TRACK_MAX_DEG: f64 = 80.0;

/// Target texel of a map, km (`MAP_TEXEL_KM`).
pub const MAP_TEXEL_KM: f64 = 3.0;
/// Widest map, texels (`MAP_MAX_WIDTH`).
pub const MAP_MAX_WIDTH: i64 = 16384;

/// The cloud regime that [`is_bakeable`] accepts, and the default of [`BodyInputs`].
pub const WATER_REGIME: &str = "water";

// ---------------------------------------------------------------------------------------------------------
// numpy and Python semantics, in one place

/// A Python float literal meeting an array of precision `F`: rounded to `F` (NEP 50 weak scalar).
pub(crate) fn lit<F: Float>(x: f64) -> F {
    F::from(x).expect("float conversion")
}

/// Python's `max(a, b)`: `b` only when `b > a`, so a NaN `b` (or NaN `a` vs a number) returns `a`.
pub(crate) fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's `min(a, b)`: `b` only when `b < a`.
pub(crate) fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `np.clip(x, lo, hi)` for finite bounds: NaN propagates, `lo` is taken below it and `hi` above it.
pub(crate) fn np_clip<F: Float>(x: F, lo: F, hi: F) -> F {
    if x.is_nan() {
        x
    } else if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// `np.degrees(x)` in the precision of `F`: `x * (180 / pi)` with both constants rounded to `F` BEFORE the
/// division, as numpy's `npy_degrees`/`npy_degreesf` do. NOT `Float::to_degrees`: for `f32` Rust multiplies
/// by the correctly rounded `180 / pi` (57.29578) where numpy multiplies by `180f / pi_f32` (57.295776), and
/// the two products differ by one ulp for about 78 per cent of the inputs. For `f64` the two agree.
pub(crate) fn np_degrees<F: Float>(x: F) -> F {
    x * (lit::<F>(180.0) / lit::<F>(std::f64::consts::PI))
}

// ---------------------------------------------------------------------------------------------------------
// The body record

/// Port of `BodyInputs` (frozen dataclass): the physical inputs of a body. `rotation_period_days` is `None`
/// for a body with no spin (Python `None`); a NaN, infinite or zero period means the same thing to every
/// function here. A negative period is a retrograde spin. `cloud_regime` is the condensate that makes the
/// clouds (`"water"` for Earth-like clouds; [`is_bakeable`] compares the exact string).
#[derive(Clone, Debug, PartialEq)]
pub struct BodyInputs {
    pub rotation_period_days: Option<f64>,
    pub radius_km: f64,
    pub gravity_mps2: f64,
    pub scale_height_km: f64,
    pub cloud_regime: Cow<'static, str>,
}

impl BodyInputs {
    /// `BodyInputs(period, radius, gravity, scale_height)`: the default `cloud_regime` of `"water"`.
    pub const fn new(
        rotation_period_days: Option<f64>,
        radius_km: f64,
        gravity_mps2: f64,
        scale_height_km: f64,
    ) -> BodyInputs {
        BodyInputs {
            rotation_period_days,
            radius_km,
            gravity_mps2,
            scale_height_km,
            cloud_regime: Cow::Borrowed(WATER_REGIME),
        }
    }

    /// The same body with another `cloud_regime`.
    pub fn with_regime(mut self, cloud_regime: impl Into<Cow<'static, str>>) -> BodyInputs {
        self.cloud_regime = cloud_regime.into();
        self
    }
}

/// Port of `EARTH`.
pub const EARTH: BodyInputs =
    BodyInputs::new(Some(EARTH_ROTATION_DAYS), EARTH_RADIUS_KM, EARTH_GRAVITY, EARTH_SCALE_HEIGHT_KM);

/// Whether this generator's cloud amount applies to the body: Earth-like water clouds only (the exact string
/// `"water"`: case and spaces matter).
pub fn is_bakeable(b: &BodyInputs) -> bool {
    b.cloud_regime == WATER_REGIME
}

// ---------------------------------------------------------------------------------------------------------
// Hadley edge, storm track, deformation radius

/// Held-Hou scaling anchored on Earth, NOT clamped, in degrees; `inf` for no spin (a `None`, NaN, infinite or
/// zero period).
///
/// Operation order kept: `30 sqrt(gh_ratio) / (omega_ratio radius_ratio)` is IEEE-exact for Earth's own radius,
/// gravity and scale height, which puts the body of period 1.6621166666666667 d at exactly 50. The product
/// `gravity scale_height` goes through `max(0.0, g H)`, so a NaN product is 0 (a fast rotator). Python raises
/// ZeroDivisionError when `omega_ratio radius_ratio` is exactly 0 (a zero radius); here it is the IEEE
/// quotient (inf or NaN).
pub fn hadley_edge_raw_deg(b: &BodyInputs) -> f64 {
    let p = match b.rotation_period_days {
        Some(p) if p.is_finite() && p != 0.0 => p,
        _ => return f64::INFINITY,
    };
    let omega_ratio = (EARTH_ROTATION_DAYS / p).abs();
    let radius_ratio = b.radius_km / EARTH_RADIUS_KM;
    let gh = py_max(0.0, b.gravity_mps2 * b.scale_height_km);
    let gh_ratio = gh / (EARTH_GRAVITY * EARTH_SCALE_HEIGHT_KM);
    EARTH_HADLEY_EDGE_DEG * gh_ratio.sqrt() / (omega_ratio * radius_ratio)
}

/// The Hadley edge clamped to `[10, 90]` degrees (twin of `hadleyEdgeDeg` of cloud_climate_type.ts). A
/// non-finite raw edge (no spin, an infinite gravity, a NaN radius) gives 90, a negative one 10.
pub fn hadley_edge_deg(b: &BodyInputs) -> f64 {
    let phi = hadley_edge_raw_deg(b);
    if !phi.is_finite() {
        return HADLEY_MAX_DEG;
    }
    py_min(HADLEY_MAX_DEG, py_max(HADLEY_MIN_DEG, phi))
}

/// Latitude of the storm track, degrees: `min(80, 1.7 phi_H)`.
pub fn storm_track_deg(b: &BodyInputs) -> f64 {
    py_min(STORM_TRACK_MAX_DEG, STORM_TRACK_IN_HADLEY * hadley_edge_deg(b))
}

/// Rossby deformation radius as an angle on the sphere, radians; `inf` with no spin, negative for a negative
/// radius. `0.3 radians(raw) sin(51 deg) / sin(track)`.
pub fn deformation_radius_rad(b: &BodyInputs) -> f64 {
    let raw = hadley_edge_raw_deg(b);
    let latitude = EARTH_STORM_TRACK_DEG.to_radians().sin() / storm_track_deg(b).to_radians().sin();
    DEFORMATION_FRACTION_OF_HADLEY * raw.to_radians() * latitude
}

/// Whether the body has a storm track for cyclones to ride: the raw Hadley edge STRICTLY below 50 degrees (a
/// NaN or infinite raw edge is false).
pub fn has_baroclinic_eddies(b: &BodyInputs) -> bool {
    hadley_edge_raw_deg(b) < 50.0
}

// ---------------------------------------------------------------------------------------------------------
// Latitude profiles

/// `_gauss(x, mu, sigma)`: `exp(-((x - mu) / sigma)^2)` (no 1/2 in the exponent), in the precision of `x`.
pub fn gauss<F: Float>(x: F, mu: f64, sigma: f64) -> F {
    let d = (x - lit::<F>(mu)) / lit::<F>(sigma);
    (-d * d).exp()
}

/// `_smoothstep(a, b, x)`: `t^2 (3 - 2t)` with `t = clip((x - a) / (b - a), 0, 1)`, in the precision of `x`
/// (`b - a` is a Python float subtraction, rounded to the precision of `x` afterwards).
pub fn smoothstep<F: Float>(a: f64, b: f64, x: F) -> F {
    let t = np_clip((x - lit::<F>(a)) / lit::<F>(b - a), F::zero(), F::one());
    t * t * (lit::<F>(3.0) - lit::<F>(2.0) * t)
}

/// `convective_tendency(lat_deg, hadley_deg)` for one latitude (twin of `convectiveTendency`): the whole chain
/// runs in the precision of `lat_deg`. The Hadley edge floor is `max(1e-3, hadley_deg)` (a NaN edge is the
/// floor, `+inf` makes `x` 0).
pub fn convective_tendency<F: Float>(lat_deg: F, hadley_deg: f64) -> F {
    let x = lat_deg.abs() / lit::<F>(py_max(1e-3, hadley_deg));
    let c = lit::<F>(0.15) + lit::<F>(0.85) * gauss(x, 0.0, 0.3) + lit::<F>(0.4) * gauss(x, 1.7, 0.35)
        - lit::<F>(0.05) * smoothstep(2.0, 2.6, x);
    np_clip(c, F::zero(), F::one())
}

/// `target_zonal_coverage(lat_deg, hadley_deg, itcz_shift_deg)` for one latitude: the zonal-mean cloud
/// fraction in units of the Hadley edge `x = |lat| / phi_H`.
///
/// The dtype chain of numpy 2 is kept: for an `f32` latitude `x` and the two smoothsteps are `f32`, while the
/// latitude is widened (`np.asarray(lat, float64)`) for the ITCZ shift fade, `x_itcz` and the ITCZ gaussian,
/// which are `f64` (the shift is a Python float and stays `f64`), and the sum is `f64`. The result is always
/// `f64`. The floor `max(1e-3, hadley_deg)` appears three times (`x`, the fade scale `0.5 floor`, `x_itcz`).
pub fn target_zonal_coverage<F: Float>(lat_deg: F, hadley_deg: f64, itcz_shift_deg: f64) -> f64 {
    let floor = py_max(1e-3, hadley_deg);
    let x = lat_deg.abs() / lit::<F>(floor);
    // The crest moves, the flanks under the subtropical highs much less: the shift fades over half the Hadley edge.
    let lat = lat_deg.to_f64().expect("float conversion");
    let q = lat / (0.5 * floor);
    let shift = itcz_shift_deg * (-(q * q)).exp();
    let x_itcz = (lat - shift).abs() / floor;
    let s1 = lit::<F>(0.45) * smoothstep(1.0, 2.0, x);
    let s2 = lit::<F>(0.05) * smoothstep(2.0, 2.6, x);
    0.20 + 0.17 * gauss(x_itcz, 0.1, 0.25)
        + s1.to_f64().expect("float conversion")
        + s2.to_f64().expect("float conversion")
}

// ---------------------------------------------------------------------------------------------------------
// Map width

/// `map_width_for_texel(radius_km, texel_km, max_width)`: circumference over the target texel, rounded UP to a
/// power of two (never below 2), capped by `max_width` (returned as is, even when it is not a power of two,
/// 0 or 1).
///
/// `w = ((2 pi) radius) / texel` is a true division, then `2^ceil(log2(max(w, 2)))` with the libm `log2`, as
/// Python: at `w` one ulp above `2^k` `log2` returns exactly `k` and the width is `2^k`. `2^62` is the
/// largest power of two an `i64` holds, so an exponent of 63 or more is wider than any `i64` cap and
/// saturates to the cap (Python compares big integers: `min(2**63 - 1, 2**63)` is the cap, but
/// `min(2**63 - 1, 2**62)` is `2**62`). Python raises (`math.ceil` of a NaN or an infinity, a zero texel)
/// where `w` is NaN or infinite; here such a `w` takes the cap when it is `+inf` and 2 when it is NaN or
/// `-inf` (`max(w, 2)` keeps the NaN in Python, and the width of a NaN has no meaning).
pub fn map_width_for_texel(radius_km: f64, texel_km: f64, max_width: i64) -> i64 {
    let w = 2.0 * std::f64::consts::PI * radius_km / texel_km;
    let k = py_max(w, 2.0).log2().ceil();
    let width: i64 = if k.is_nan() {
        2
    } else if k >= 63.0 {
        i64::MAX
    } else {
        1i64 << (k as u32)
    };
    width.min(max_width)
}

/// `map_width_for_texel(radius_km)` with both defaults ([`MAP_TEXEL_KM`], [`MAP_MAX_WIDTH`]), as the callers use it.
pub fn map_width_for_radius(radius_km: f64) -> i64 {
    map_width_for_texel(radius_km, MAP_TEXEL_KM, MAP_MAX_WIDTH)
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: f64 = EARTH_ROTATION_DAYS;

    fn body(p: Option<f64>, r: f64, g: f64, h: f64) -> BodyInputs {
        BodyInputs::new(p, r, g, h)
    }

    #[test]
    fn python_max_min_and_clip_semantics() {
        assert_eq!(py_max(0.0, f64::NAN), 0.0);
        assert_eq!(py_max(1e-3, f64::NAN), 1e-3);
        assert!(py_max(f64::NAN, 1.0).is_nan()); // 1.0 > NaN is false: the first argument stays
        assert_eq!(py_max(1.0, 2.0), 2.0);
        assert_eq!(py_min(80.0, 90.0), 80.0);
        assert_eq!(py_min(80.0, 70.0), 70.0);
        assert!(py_min(f64::NAN, 1.0).is_nan());
        assert_eq!(py_min(80.0, f64::NAN), 80.0);
        assert!(np_clip(f64::NAN, 0.0, 1.0).is_nan());
        assert_eq!(np_clip(-3.0, 0.0, 1.0), 0.0);
        assert_eq!(np_clip(3.0f32, 0.0, 1.0), 1.0);
        assert_eq!(np_clip(0.25, 0.0, 1.0), 0.25);
    }

    #[test]
    fn np_degrees_is_numpys_not_rusts_for_f32() {
        // Oracle: numpy 2.5.4 `np.degrees(np.float32(...))` as (input bits, output bits). Rust's own
        // `f32::to_degrees` gives output bits + 1 on every one of these (57.29578 against numpy's 57.295776).
        let pins: [(u32, u32); 8] = [
            (1067571966, 1116792456),
            (1067610678, 1116827113),
            (1055251176, 1104004840),
            (1034780317, 1083920903),
            (1058679146, 1107952454),
            (1059340735, 1108544739),
            (1032955487, 1082287230),
            (1070130546, 1119083016),
        ];
        for (x, want) in pins {
            let got = np_degrees(f32::from_bits(x)).to_bits();
            assert_eq!(got, want, "np.degrees({})", f32::from_bits(x));
        }
        // float64: `x * (180 / pi)`, the same as Rust's.
        let pins64: [(f64, u64); 4] = [
            (0.3, 4625531416143348014),
            (1.0, 4633260481411531256),
            (-2.5, 13862615843482786107),
            (3.0, 4640251764640764282),
        ];
        for (x, want) in pins64 {
            assert_eq!(np_degrees(x).to_bits(), want, "np.degrees({x})");
        }
        assert!(np_degrees(f32::NAN).is_nan());
        assert_eq!(np_degrees(0.0f32), 0.0);
    }

    #[test]
    fn earth_and_defaults() {
        assert_eq!(EARTH.rotation_period_days, Some(0.99727));
        assert_eq!(EARTH.radius_km, 6371.0);
        assert_eq!(EARTH.gravity_mps2, 9.81);
        assert_eq!(EARTH.scale_height_km, 8.0);
        assert_eq!(EARTH.cloud_regime, "water");
        assert!(is_bakeable(&EARTH));
        assert!(!is_bakeable(&EARTH.clone().with_regime("methane")));
        assert!(!is_bakeable(&EARTH.clone().with_regime(String::from("Water"))));
        assert_eq!(
            body(Some(1.0), 1.0, 1.0, 1.0),
            BodyInputs { cloud_regime: "water".into(), ..body(Some(1.0), 1.0, 1.0, 1.0) }
        );
    }

    #[test]
    fn earth_hadley_edge_is_the_anchor_and_the_derived_values_follow() {
        assert_eq!(hadley_edge_raw_deg(&EARTH), 30.0);
        assert_eq!(hadley_edge_deg(&EARTH), 30.0);
        assert!((storm_track_deg(&EARTH) - 51.0).abs() < 1e-12);
        // At Earth's own storm track the latitude factor is 1: L_R = 0.3 radians(30).
        assert!((deformation_radius_rad(&EARTH) - 0.3 * 30.0_f64.to_radians()).abs() < 1e-12);
        assert!(has_baroclinic_eddies(&EARTH));
    }

    #[test]
    fn no_spin_in_every_spelling_is_an_infinite_raw_edge() {
        for p in [None, Some(f64::NAN), Some(f64::INFINITY), Some(f64::NEG_INFINITY), Some(0.0), Some(-0.0)] {
            let b = body(p, 6371.0, 9.81, 8.0);
            assert_eq!(hadley_edge_raw_deg(&b), f64::INFINITY, "{p:?}");
            assert_eq!(hadley_edge_deg(&b), 90.0);
            assert_eq!(storm_track_deg(&b), 80.0);
            assert_eq!(deformation_radius_rad(&b), f64::INFINITY);
            assert!(!has_baroclinic_eddies(&b));
        }
    }

    #[test]
    fn the_sign_of_the_period_is_lost_in_the_hadley_edge() {
        let (a, b) = (body(Some(2.5), 5000.0, 7.0, 9.0), body(Some(-2.5), 5000.0, 7.0, 9.0));
        assert_eq!(hadley_edge_raw_deg(&a), hadley_edge_raw_deg(&b));
    }

    #[test]
    fn a_nan_gravity_height_product_is_a_fast_rotator_and_a_nan_radius_a_nan_edge() {
        assert_eq!(hadley_edge_raw_deg(&body(Some(P), 6371.0, f64::NAN, 8.0)), 0.0);
        assert_eq!(hadley_edge_raw_deg(&body(Some(P), 6371.0, f64::INFINITY, 0.0)), 0.0); // inf * 0 = NaN -> 0
        assert_eq!(hadley_edge_deg(&body(Some(P), 6371.0, f64::NAN, 8.0)), 10.0);
        let nan_radius = body(Some(P), f64::NAN, 9.81, 8.0);
        assert!(hadley_edge_raw_deg(&nan_radius).is_nan());
        assert_eq!(hadley_edge_deg(&nan_radius), 90.0);
        assert!(!has_baroclinic_eddies(&nan_radius));
        assert_eq!(hadley_edge_deg(&body(Some(P), -6371.0, 9.81, 8.0)), 10.0); // a negative edge clamps to 10
        assert_eq!(hadley_edge_deg(&body(Some(P), 6371.0, f64::INFINITY, 8.0)), 90.0);
    }

    #[test]
    fn a_minus_infinite_raw_edge_is_not_finite_and_gives_the_maximum() {
        // A negative radius and an infinite gravity: 30 inf / (omega * -1) = -inf. Python tests `isfinite`, so
        // the clamp to the minimum (10) is not taken: the edge is 90.
        let b = body(Some(P), -6371.0, f64::INFINITY, 8.0);
        assert_eq!(hadley_edge_raw_deg(&b), f64::NEG_INFINITY);
        assert_eq!(hadley_edge_deg(&b), 90.0);
    }

    #[test]
    fn the_exact_threshold_body_is_not_baroclinic_and_its_neighbors_are_on_both_sides() {
        let at = |p: f64| body(Some(p), 6371.0, 9.81, 8.0);
        assert_eq!(hadley_edge_raw_deg(&at(1.6621166666666667)), 50.0);
        assert!(!has_baroclinic_eddies(&at(1.6621166666666667)));
        assert!(has_baroclinic_eddies(&at(1.6621166666666665)));
        assert!(!has_baroclinic_eddies(&at(1.662116666666667)));
    }

    #[test]
    fn a_zero_radius_is_an_ieee_quotient_where_python_raises() {
        let b = body(Some(P), 0.0, 9.81, 8.0);
        assert_eq!(hadley_edge_raw_deg(&b), f64::INFINITY);
        assert_eq!(hadley_edge_deg(&b), 90.0);
    }

    #[test]
    fn convective_tendency_is_the_documented_profile() {
        // x = 0: 0.15 + 0.85 + 0.4 exp(-(1.7/0.35)^2) - 0.05 * 0 is clipped to 1.
        assert_eq!(convective_tendency(0.0f64, 30.0), 1.0);
        // x = 1.7 (lat 51 at Hadley 30): 0.15 + 0.85 exp(-(1.7/0.3)^2) + 0.4 - 0.05 smoothstep(2, 2.6, 1.7) = 0.55 + tiny.
        let c = convective_tendency(51.0f64, 30.0);
        assert!((c - (0.55 + 0.85 * (-(1.7f64 / 0.3).powi(2)).exp())).abs() < 1e-15);
        // Past x = 2.6 the smoothstep is 1: 0.10 for a huge latitude, and NaN stays NaN.
        assert!((convective_tendency(1e300f64, 30.0) - 0.10).abs() < 1e-15);
        assert!(convective_tendency(f64::NAN, 30.0).is_nan());
        // Symmetric in the latitude, f32 runs in f32.
        assert_eq!(convective_tendency(-25.0f64, 30.0), convective_tendency(25.0f64, 30.0));
        let a: f32 = convective_tendency(25.0f32, 30.0);
        assert!((f64::from(a) - convective_tendency(25.0f64, 30.0)).abs() < 1e-6);
    }

    #[test]
    fn the_hadley_floor_is_a_python_max() {
        // An edge of 0, -5 and NaN all use the floor 1e-3; +inf makes x 0.
        let at = |h: f64| convective_tendency(0.002f64, h);
        assert_eq!(at(0.0), at(1e-3));
        assert_eq!(at(-5.0), at(1e-3));
        assert_eq!(at(f64::NAN), at(1e-3));
        assert_eq!(at(f64::INFINITY), convective_tendency(0.0f64, 30.0));
    }

    #[test]
    fn target_zonal_coverage_anchor_values_and_the_itcz_shift() {
        // The equator at Hadley 30: 0.20 + 0.17 exp(-(0.1/0.25)^2) (the smoothsteps are 0).
        let eq = target_zonal_coverage(0.0f64, 30.0, 0.0);
        assert!((eq - (0.20 + 0.17 * (-(0.4f64).powi(2)).exp())).abs() < 1e-15);
        // Far from everything: 0.20 + 0.45 + 0.05 = 0.70 (the ITCZ bump is gone).
        assert!((target_zonal_coverage(90.0f64, 30.0, 0.0) - 0.70).abs() < 1e-12);
        // A northward shift moves the crest north: more cloud at +3 than at -3, symmetric without it.
        assert_eq!(target_zonal_coverage(3.0f64, 30.0, 0.0), target_zonal_coverage(-3.0f64, 30.0, 0.0));
        assert!(target_zonal_coverage(3.0f64, 30.0, 6.0) > target_zonal_coverage(-3.0f64, 30.0, 6.0));
        // A NaN shift makes everything NaN; an infinite one removes the bump.
        assert!(target_zonal_coverage(10.0f64, 30.0, f64::NAN).is_nan());
        assert!((target_zonal_coverage(90.0f64, 30.0, f64::INFINITY) - 0.70).abs() < 1e-12);
        // The shift stays a float64 for an f32 latitude: the result is the f64 formula to f32 rounding.
        let a = target_zonal_coverage(10.0f32, 30.0, 3.7);
        assert!((a - target_zonal_coverage(10.0f64, 30.0, 3.7)).abs() < 1e-6);
    }

    #[test]
    fn map_width_divides_in_the_order_of_the_python() {
        // Radii (oracle: Python's map_width_for_texel(r, 3.0, 1 << 20)) where w = ((2 pi) r) / 3 sits a few ulps
        // above a power of two, so that a multiplication by 1/3 (widths 256, 512, 1024) or 2 pi (r / 3)
        // (widths 64, 128, 256) rounds the other way.
        let at = |r: f64| map_width_for_texel(r, 3.0, 1 << 20);
        assert_eq!([122.23099629457569, 244.46199258915138, 488.92398517830276].map(at), [512, 1024, 2048]);
        assert_eq!([15.278874536821958, 30.557749073643915, 61.11549814728783].map(at), [32, 64, 128]);
    }

    #[test]
    fn map_width_is_a_ceiled_power_of_two_with_the_cap_as_is() {
        let w = |r: f64| map_width_for_radius(r);
        assert_eq!((w(6371.0), w(6051.8), w(3389.5), w(2574.7), w(1737.4)), (16384, 16384, 8192, 8192, 4096));
        assert_eq!(map_width_for_texel(6371.0, 3.0, 100), 100);
        assert_eq!(map_width_for_texel(6371.0, 3.0, 1), 1);
        assert_eq!(map_width_for_texel(6371.0, 3.0, 0), 0);
        assert_eq!(map_width_for_texel(1024.0, 2.0 * std::f64::consts::PI, 1 << 20), 1024);
        assert_eq!((w(0.0), w(-5.0), w(0.1)), (2, 2, 2));
        assert_eq!(map_width_for_texel(6371.0, -3.0, 16384), 2);
        assert_eq!(map_width_for_texel(1e300, 3.0, 16384), 16384);
        assert_eq!(map_width_for_texel(1e300, 3.0, i64::MAX), i64::MAX);
        assert_eq!(map_width_for_texel(f64::INFINITY, 3.0, 16384), 16384);
    }

    #[test]
    fn map_width_saturates_only_past_the_largest_power_of_two_an_i64_holds() {
        // Oracle: Python's `map_width_for_texel(r, 2 pi, 2**63 - 1)`, with a texel of 2 pi so that `w` is `r`
        // (`(2 pi r) / (2 pi)` is exact for a power of two, and `log2` of a power of two is exact). A width of
        // `2^62` is a real width, `2^63` is not an `i64` and takes the cap.
        let two_pi = 2.0 * std::f64::consts::PI;
        let at = |r: f64, cap: i64| map_width_for_texel(r, two_pi, cap);
        let (p60, p61, p62) = (2f64.powi(60), 2f64.powi(61), 2f64.powi(62));
        assert_eq!(at(p60, i64::MAX), 1 << 60);
        assert_eq!(at(1.5 * p60, i64::MAX), 1 << 61);
        assert_eq!(at(p61, i64::MAX), 1 << 61);
        assert_eq!(at(1.5 * p61, i64::MAX), 1 << 62); // w in (2^61, 2^62]: 2^62, NOT the cap
        assert_eq!(at(p62, i64::MAX), 1 << 62);
        assert_eq!(at(p62 + 1024.0, i64::MAX), 1 << 62);
        assert_eq!(at(1.5 * p62, i64::MAX), i64::MAX); // w in (2^62, 2^63]: 2^63 does not fit, the cap
        assert_eq!(at(2.0 * p62, i64::MAX), i64::MAX);
        assert_eq!(at(1e300, i64::MAX), i64::MAX);
        // A cap below the width wins whatever the exponent.
        assert_eq!(at(1.5 * p62, 1 << 62), 1 << 62);
        assert_eq!(at(1.5 * p61, 1 << 61), 1 << 61);
        assert_eq!(at(p60, 1 << 61), 1 << 60);
        // A radius whose `w` is 2^62 up to rounding, with a texel of 3 km (oracle: 4611686018427387904).
        assert_eq!(map_width_for_texel(p62 * 3.0 / two_pi, 3.0, i64::MAX), 1 << 62);
    }
}
