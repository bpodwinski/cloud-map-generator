//! Port of `cloudgen/products.py`: the maps derived from the generator's own fields, for the deck. CLOUD TYPE
//! ([`type_map`]), HIGH-CLOUD SHARE ([`high_share`]) and WIND ([`wind_map`], with the climatological zonal
//! profile [`climatological_zonal_mps`] and its scale [`zonal_wind_scale`]), plus the wind ENCODING
//! ([`encode_wind`]) that `export.rs` writes into the products' PNGs.
//!
//! Dtypes, as in Python (numpy 2, NEP 50). The maps take their fields in the precision the generator hands
//! them and run each step in the dtype numpy gives that step: a Python float is a "weak" scalar and takes the
//! precision of the array it meets (`lit`), two arrays meet in the wider of their precisions, `np.interp`
//! always answers in float64, and the result of `type_map` and `high_share` is cast to float32 at the end.
//! The generator's shield and latitudes are float64 and its weights, ascent and coverage float32 (or float64
//! when it has no surface), so the maps are generic over the precision of each group of arrays; the group of
//! the shield and the latitudes must be at least as wide as the others (the arrangement of the pipeline), which
//! the functions assert.
//!
//! `type_map` and `high_share` take ONE latitude per row (`lat_rows`) and the row width instead of the
//! `(height, width)` latitude array that Python broadcasts from them (`generate` passes
//! `np.broadcast_to(lat[:, None], shape)`): every texel of a row has the same latitude, so the tendency, the
//! table lookup and the power are evaluated once per row and give the very same values.

use crate::climate::{
    convective_tendency, has_baroclinic_eddies, BodyInputs, EARTH_GRAVITY, EARTH_RADIUS_KM, EARTH_ROTATION_DAYS,
    EARTH_SCALE_HEIGHT_KM,
};
use crate::export::round_ties_even;
use crate::flow::Flow;
use crate::matching::ndimage::interp;
use crate::sphere::{grid_dirs, grid_lonlat, Vec3};
use num_traits::Float;
use std::f64::consts::PI;

/// `products.WIND_STEP_MPS`: one byte step of the wind encoding, m/s (128 = calm, range +-64 m/s). Exactly
/// representable in `f32`, so the arithmetic below is the same in either precision.
pub const WIND_STEP_MPS: f64 = 0.5;

/// `products.ADVECTION_DAYS`: the model's advection time, days, that turns the vortices' rotation angles into
/// m/s (a cyclone's winding of ~3 rad over 3 days is ~20 m/s at 1000 km) [model choice].
pub const ADVECTION_DAYS: f64 = 3.0;

/// `products.TYPE_CLOSED`: the cloud type (0 stratiform .. 1 convective) of closed cells (stratocumulus).
pub const TYPE_CLOSED: f64 = 0.45;
/// `products.TYPE_OPEN`: the cloud type of open cells (cumulus).
pub const TYPE_OPEN: f64 = 0.85;
/// `products.TYPE_STREET`: the cloud type of cloud streets.
pub const TYPE_STREET: f64 = 0.8;
/// `products.TYPE_BROKEN`: the cloud type of the rest of the cumulus population (broken fields).
pub const TYPE_BROKEN: f64 = 0.7;

/// `products.HIGH_BASE_X`: the abscissae of the MODIS zonal high-cloud-share table, in units of the Hadley
/// edge (`|lat| = 30 x` on Earth).
pub const HIGH_BASE_X: [f64; 8] = [0.0, 0.33, 0.67, 1.0, 1.5, 1.83, 2.17, 3.0];
/// `products.HIGH_BASE_SHARE`: the zonal means of `earth_high_cloud_share_1440.png` (MODIS Aqua, both
/// hemispheres) at the abscissae of [`HIGH_BASE_X`], lowered by the mean of the vortex term.
pub const HIGH_BASE_SHARE: [f64; 8] = [0.33, 0.39, 0.25, 0.29, 0.27, 0.22, 0.22, 0.2];
/// `products.HIGH_COMMA_ASCENT`: cirrus over a low's comma head.
pub const HIGH_COMMA_ASCENT: f64 = 0.25;
/// `products.HIGH_DEEP_CORE`: anvils over the convergence zone's clusters.
pub const HIGH_DEEP_CORE: f64 = 0.15;

// ---------------------------------------------------------------------------------------------------------
// numpy semantics, in one place
//
// TODO(dedupe): the same small helpers live in `climate.rs` (`lit`, `np_clip`, `py_max`, both `pub(crate)`),
// which another agent owns; they are kept private here so that this module does not depend on their names.

/// A Python float literal meeting an array of precision `F`: rounded to `F` (NEP 50 weak scalar).
fn lit<F: Float>(x: f64) -> F {
    F::from(x).expect("float conversion")
}

/// A value of one float precision in another: exact when widening, round to nearest even when narrowing
/// (numpy's promotion between two arrays and `astype`).
fn conv<A: Float, B: Float>(x: A) -> B {
    B::from(x).expect("float conversion")
}

/// `np.clip(x, lo, hi)`: NaN propagates, `lo` is taken below it and `hi` above it.
fn np_clip<F: Float>(x: F, lo: F, hi: F) -> F {
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

/// `np.maximum(a, b)`: NaN propagates (either operand), a tie gives `a`.
fn np_maximum<F: Float>(a: F, b: F) -> F {
    if a >= b || a.is_nan() {
        a
    } else {
        b
    }
}

/// Python's `max(a, b)`: `b` only when `b > a`, so a NaN `b` (or a NaN `a` against a number) gives `a`.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// The factor of `np.degrees` in the precision `F`: numpy computes `x * (180 / pi)` with the constant divided
/// in `F` (for float32 that is `180f / PIf`, one ulp away from the correctly rounded constant that
/// `f32::to_degrees` multiplies by).
fn degrees_factor<F: Float>() -> F {
    lit::<F>(180.0) / lit::<F>(PI)
}

/// The final `astype(np.float32)` of a value already clipped to `[0, 1]` (or NaN).
fn to_f32<F: Float>(x: F) -> f32 {
    x.to_f32().expect("float conversion")
}

/// The promotion this module supports: the first group of arrays (shield, latitudes) is at least as wide as
/// the second, so that every mixed operation of the Python runs in the first one's precision. numpy promotes
/// to the wider of the two, which is the other way round only for arrangements the generator never makes.
fn assert_at_least_as_wide<A: Float, B: Float>(what: &str) {
    assert!(
        std::mem::size_of::<A>() >= std::mem::size_of::<B>(),
        "{what}: the shield and latitude arrays must be at least as precise as the others (float64 over float32, as \
         the generator passes them)"
    );
}

/// The `n` values a `(height, width)` field holds, with `height = lat_rows.len()`.
fn field_len(lat_rows: usize, width: usize) -> usize {
    lat_rows.checked_mul(width).expect("field size overflows")
}

// ---------------------------------------------------------------------------------------------------------
// type_map, high_share

/// `products.type_map(shield, weights, lat, hadley_deg)`: the weather-map B ladder, 0 = stratiform bottom type
/// (stratus) .. 1 = convective top type (cumulus). Synoptic shields take the convective tendency `c` of their
/// latitude, the cumulus population the type of its regime (the weights times the four type levels, the rest
/// of the weight being the broken fields): `clip(shield c + (1 - shield) cumulus, 0, 1)`, cast to float32.
///
/// `shield` and the `(height, width)` weight fields `closed`, `open`, `street` (the entries of Python's
/// `weights` dict) are row-major with `height = lat_rows.len()`; `lat_rows` is the CONVECTION latitude of each
/// row in radians (`generate` passes the grid latitude less the ITCZ shift). Pipeline dtypes: `S = f64` for
/// the shield and latitudes, `W = f32` for the weights (`S` must be at least as wide as `W`); the cumulus
/// level is evaluated in the precision of the weights, the blend in the precision of the shield. Out-of-contract
/// weights are not clamped: only the final clip bounds the result. NaN propagates.
///
/// # Panics
/// If a field does not hold `lat_rows.len() * width` values, or if `W` is wider than `S`.
#[allow(clippy::too_many_arguments)]
pub fn type_map<S: Float + Send + Sync, W: Float + Send + Sync>(
    shield: &[S],
    closed: &[W],
    open: &[W],
    street: &[W],
    lat_rows: &[S],
    width: usize,
    hadley_deg: f64,
) -> Vec<f32> {
    assert_at_least_as_wide::<S, W>("type_map");
    let n = field_len(lat_rows.len(), width);
    for len in [shield.len(), closed.len(), open.len(), street.len()] {
        assert_eq!(len, n, "type_map: every field holds height x width values");
    }
    let degrees = degrees_factor::<S>();
    let (k_closed, k_open, k_street, k_broken) =
        (lit::<W>(TYPE_CLOSED), lit::<W>(TYPE_OPEN), lit::<W>(TYPE_STREET), lit::<W>(TYPE_BROKEN));
    // A texel does not depend on another: the rows run in parallel.
    let mut out = vec![0.0_f32; n];
    crate::par::for_each_chunk_mut(&mut out, width.max(1), |r, row| {
        let c = convective_tendency(lat_rows[r] * degrees, hadley_deg);
        for (j, slot) in row.iter_mut().enumerate() {
            let i = r * width + j;
            let (wc, wo, ws) = (closed[i], open[i], street[i]);
            let cumulus = wc * k_closed + wo * k_open + ws * k_street + (W::one() - wc - wo - ws) * k_broken;
            let s = shield[i];
            let blend = s * c + (S::one() - s) * conv::<W, S>(cumulus);
            *slot = to_f32(np_clip(blend, S::zero(), S::one()));
        }
    });
    out
}

/// `products.high_share(shield, ascent, lat, hadley_deg, coverage)`: the CONDITIONAL share of the cloud present
/// in a texel whose tops lie above 440 hPa (MODIS's definition; the deck draws its top band as fraction x
/// share): the MODIS zonal table placed in units of the Hadley edge (`np.interp` of `x = |lat| / phi_H`, which
/// clamps to the end values) plus, for the part of the texel under a synoptic shield (`clip(shield /
/// max(coverage, 0.05), 0, 1)`), a cirrus term over the lows' comma heads (`0.25 clip(ascent, 0, 1)`) and the
/// deep cores of the convergence zone (`0.15 c^4`): `clip(base + (1 - base) extra, 0, 1)`, cast to float32.
///
/// Fields are row-major with `height = lat_rows.len()`, `lat_rows` the CONVECTION latitude of each row in
/// radians. Pipeline dtypes: `S = f64` for the shield and latitudes, `A = C = f32` for the ascent and coverage
/// (`S` must be at least as wide as both); the coverage floor `0.05` and the ascent clip run in the precision
/// of their array, the rest in the precision of the shield, the table lookup and the final blend in float64.
/// NaN propagates.
///
/// # Panics
/// If a field does not hold `lat_rows.len() * width` values, or if `A` or `C` is wider than `S`.
#[allow(clippy::too_many_arguments)]
pub fn high_share<S: Float + Send + Sync, A: Float + Send + Sync, C: Float + Send + Sync>(
    shield: &[S],
    ascent: &[A],
    lat_rows: &[S],
    width: usize,
    hadley_deg: f64,
    coverage: &[C],
) -> Vec<f32> {
    assert_at_least_as_wide::<S, A>("high_share");
    assert_at_least_as_wide::<S, C>("high_share");
    let n = field_len(lat_rows.len(), width);
    assert_eq!(shield.len(), n, "high_share: every field holds height x width values");
    assert_eq!(ascent.len(), n, "high_share: every field holds height x width values");
    assert_eq!(coverage.len(), n, "high_share: every field holds height x width values");
    let degrees = degrees_factor::<S>();
    let (comma, floor) = (lit::<A>(HIGH_COMMA_ASCENT), lit::<C>(0.05));
    let deep = lit::<S>(HIGH_DEEP_CORE);
    let mut out = vec![0.0_f32; n];
    crate::par::for_each_chunk_mut(&mut out, width.max(1), |r, row| {
        let lat_deg = lat_rows[r] * degrees;
        let c = convective_tendency(lat_deg, hadley_deg);
        let x = lat_deg.abs() / lit::<S>(hadley_deg);
        // The zonal base is MODIS's own profile (np.interp answers in float64 whatever the dtype of x).
        let base = interp(conv::<S, f64>(x), &HIGH_BASE_X, &HIGH_BASE_SHARE);
        // `c ** 4` is np.power (libm pow), not repeated products.
        let deep_core = deep * c.powf(lit::<S>(4.0));
        for (j, slot) in row.iter_mut().enumerate() {
            let i = r * width + j;
            let shield_share = np_clip(shield[i] / conv::<C, S>(np_maximum(coverage[i], floor)), S::zero(), S::one());
            let comma_ascent = conv::<A, S>(comma * np_clip(ascent[i], A::zero(), A::one()));
            let extra = shield_share * (comma_ascent + deep_core);
            let share = np_clip(base + (1.0 - base) * conv::<S, f64>(extra), 0.0, 1.0);
            *slot = share as f32;
        }
    });
    out
}

// ---------------------------------------------------------------------------------------------------------
// The wind

/// `products.zonal_wind_scale(body)`: the wind scale relative to Earth, Held and Hou's angular-momentum wind
/// at the cell edge `U ~ Omega a phi_H^2 ~ g H / (Omega a)`: `max(0, g H) / (g H)_Earth` over `Omega a /
/// (Omega a)_Earth`. 0 for a body with no storm track (a global Hadley cell, [`has_baroclinic_eddies`] false):
/// there the real circulation is a super-rotation (Venus ~100 m/s retrograde at the cloud tops) that this
/// model does not describe, so it bakes no wind rather than a wrong one. A negative gravity or scale height
/// floors `g H` at 0 (Python's `max(0.0, ..)`, which a NaN product loses to).
pub fn zonal_wind_scale(body: &BodyInputs) -> f64 {
    if !has_baroclinic_eddies(body) {
        return 0.0;
    }
    // A storm track needs a finite, non-zero rotation period (hadley_edge_raw_deg is infinite otherwise).
    let period = body.rotation_period_days.expect("a body with a storm track has a rotation period");
    let omega_a = 2.0 * PI / (period.abs() * 86400.0) * body.radius_km * 1000.0;
    let earth_omega_a = 2.0 * PI / (EARTH_ROTATION_DAYS * 86400.0) * EARTH_RADIUS_KM * 1000.0;
    let gh = py_max(0.0, body.gravity_mps2 * body.scale_height_km) / (EARTH_GRAVITY * EARTH_SCALE_HEIGHT_KM);
    gh / (omega_a / earth_omega_a)
}

/// The direction of the spin of `body`: `math.copysign(1.0, period or 1.0)`. No period, or a period of 0, is
/// `+1`; a negative period (retrograde) `-1`; a NaN period takes the sign bit of the NaN (a Python NaN is
/// truthy), as `math.copysign` does.
fn spin_sign(body: &BodyInputs) -> f64 {
    let period = match body.rotation_period_days {
        Some(p) if p != 0.0 => p,
        _ => 1.0,
    };
    1.0_f64.copysign(period)
}

/// `products.climatological_zonal_mps(lat, hadley_deg, body)`: the climatological zonal wind of a latitude
/// row near the cloud base, m/s, positive eastward: westerlies ~12 m/s on the storm track (`x = |lat| /
/// phi_H = 1.5`), trades ~7 m/s easterly (`x = 0.45`), weak polar easterlies (a smoothstep to -4 beyond
/// `x = 2.1`) [measured, Earth's low-level circulation], times [`zonal_wind_scale`] and the spin sign
/// (a retrograde body turns the whole profile round). `lat` in radians, symmetric in `|lat|`.
///
/// Runs in the precision `F` of `lat` (the scale and the spin are Python floats: weak scalars).
pub fn climatological_zonal_mps<F: Float>(lat: &[F], hadley_deg: f64, body: &BodyInputs) -> Vec<F> {
    let degrees = degrees_factor::<F>();
    let hadley = lit::<F>(hadley_deg);
    let scale = lit::<F>(zonal_wind_scale(body));
    let spin = lit::<F>(spin_sign(body));
    lat.iter()
        .map(|&la| {
            let x = (la * degrees).abs() / hadley;
            // `-((x - mu) / s) ** 2`: the minus applies after the square.
            let g = |mu: f64, s: f64| {
                let d = (x - lit::<F>(mu)) / lit::<F>(s);
                (-(d * d)).exp()
            };
            let sm = np_clip((x - lit::<F>(2.1)) / lit::<F>(0.6), F::zero(), F::one());
            let sm = sm * sm * (lit::<F>(3.0) - lit::<F>(2.0) * sm);
            let u = lit::<F>(12.0) * g(1.5, 0.45) - lit::<F>(7.0) * g(0.45, 0.35) - lit::<F>(4.0) * sm;
            u * scale * spin
        })
        .collect()
}

/// `a x b`, in float64 (`np.cross` of 3-vectors: `a1 b2 - a2 b1`, `a2 b0 - a0 b2`, `a0 b1 - a1 b0`).
fn cross(a: Vec3<f64>, b: Vec3<f64>) -> Vec3<f64> {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// The unit eastward direction at the point `p` of the sphere, as `wind_map` builds it: `z x p` over its norm
/// (`np.linalg.norm`: the square root of the sum of the squares in order, floored at 1e-12 in the quotient),
/// or the fixed `+y` where that cross product is shorter than 1e-9 (the poles). The fallback is unreachable
/// from `wind_map`, whose cell-centered grid never has a row nearer a pole than `90 / h` degrees.
fn east_unit(p: Vec3<f64>) -> Vec3<f64> {
    let e = cross([0.0, 0.0, 1.0], p);
    let n = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
    if n > 1e-9 {
        let d = np_maximum(n, 1e-12);
        [e[0] / d, e[1] / d, e[2] / d]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// The float64 intermediates of [`wind_map`], row-major: what its two components are made of.
#[derive(Clone, Debug, PartialEq)]
pub struct WindComponents {
    /// `height = width // 2` rows.
    pub height: usize,
    pub width: usize,
    /// `(height,)`: the climatological zonal profile at the latitude of each row, m/s.
    pub zonal: Vec<f64>,
    /// `(height, width)`: the vortices' own flow projected on the local east, m/s (before the zonal profile
    /// is added to it).
    pub vortex_east: Vec<f64>,
    /// `(height, width)`: the vortices' own flow projected on the local north, m/s.
    pub vortex_north: Vec<f64>,
}

/// The float64 body of `products.wind_map(flow, body, width)`: the flow with its zonal terms (`jet`,
/// `trades`, `polar_easterlies`) switched off (the zonal wind is the BODY's climatology, not the flow's),
/// its velocity at the cell centers of a `width x width/2` grid in rad per advection time, turned into m/s
/// by `radius_km 1000 / (ADVECTION_DAYS 86400)` and projected on the local east and north.
///
/// Python has no such function (it exposes no intermediates); the golden cases rebuild them from the same
/// expressions. `wind_map` is this plus the zonal profile added to the east component.
pub fn wind_components(flow: &Flow, body: &BodyInputs, width: usize) -> WindComponents {
    let height = width / 2;
    let p = grid_dirs::<f64>(width, height);
    let still = Flow { jet: 0.0, trades: 0.0, polar_easterlies: 0.0, ..flow.clone() };
    let v = still.velocity(&p);
    let (metres, seconds) = (body.radius_km * 1000.0, ADVECTION_DAYS * 86400.0);
    let (_, lat) = grid_lonlat(width, height);
    let zonal = climatological_zonal_mps(&lat, flow.hadley_deg, body);
    let both = crate::par::map_index(p.len(), |i| {
        let (pt, vel) = (p[i], &v[i]);
        let vel = [vel[0] * metres / seconds, vel[1] * metres / seconds, vel[2] * metres / seconds];
        let east = east_unit(pt);
        let north = cross(pt, east);
        [vel[0] * east[0] + vel[1] * east[1] + vel[2] * east[2], vel[0] * north[0] + vel[1] * north[1] + vel[2] * north[2]]
    });
    let vortex_east: Vec<f64> = both.iter().map(|b| b[0]).collect();
    let vortex_north: Vec<f64> = both.iter().map(|b| b[1]).collect();
    WindComponents { height, width, zonal, vortex_east, vortex_north }
}

/// `products.wind_map(flow, body, width)`: the east and north wind in m/s near the cloud base on a `width x
/// width/2` grid (`height = width // 2`, north-up, column 0 at -180 degrees), interleaved `(height, width, 2)`
/// row-major float32: the vortices' own flow ([`wind_components`]) plus the climatological zonal profile of
/// the body ([`climatological_zonal_mps`]) on the east component. Computed in float64 and rounded to float32
/// once at the end. A global Hadley cell has no zonal wind (profile 0), a retrograde body turns the profile
/// round but not the vortices' flow; the flow's own `jet`, `trades` and `polar_easterlies` are ignored.
pub fn wind_map(flow: &Flow, body: &BodyInputs, width: usize) -> Vec<f32> {
    let parts = wind_components(flow, body, width);
    let mut out = Vec::with_capacity(parts.vortex_east.len() * 2);
    for r in 0..parts.height {
        for i in r * width..(r + 1) * width {
            out.push((parts.vortex_east[i] + parts.zonal[r]) as f32);
            out.push(parts.vortex_north[i] as f32);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------
// The wind encoding

/// `products.encode_wind`: the `(height, width, 2)` east/north wind (interleaved, row-major, m/s) as the
/// `(height, width, 3)` RGB bytes of the wind PNG: R east, G north, `clip(round(128 + v / WIND_STEP_MPS), 0,
/// 255)`, B = 0.
///
/// numpy semantics kept: the whole computation runs in the precision `F` of the wind (a Python float is a
/// weak scalar, so a float32 wind gives a float32 `128.0 + wind / 0.5`); `np.round` is half-to-even, so
/// `128.5 -> 128` but `129.5 -> 130`; the clip saturates at 0 and 255; a NaN component, whose `astype(uint8)`
/// numpy leaves to the platform (0 on x86-64 and arm64), gives 0.
///
/// Panics when the number of values is odd (a wind has two components per texel).
pub fn encode_wind<F: Float>(wind: &[F]) -> Vec<u8> {
    assert!(wind.len() % 2 == 0, "a wind has two components (east, north) per texel");
    let step = F::from(WIND_STEP_MPS).expect("float conversion");
    let bias = F::from(128.0).expect("float conversion");
    let (lo, hi) = (F::zero(), F::from(255.0).expect("float conversion"));
    let mut out = Vec::with_capacity(wind.len() / 2 * 3);
    for texel in wind.chunks_exact(2) {
        for &v in texel {
            let q = round_ties_even(bias + v / step);
            // np.clip then astype(uint8): the clip engages at both ends; NaN is the only value left outside.
            let q = if q < lo {
                lo
            } else if q > hi {
                hi
            } else {
                q
            };
            out.push(q.to_u8().unwrap_or(0));
        }
        out.push(0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calm_is_128_and_blue_is_zero() {
        assert_eq!(encode_wind(&[0.0_f32, -0.0, 0.0, 0.0]), vec![128, 128, 0, 128, 128, 0]);
    }

    #[test]
    fn halves_round_to_even_and_the_clip_saturates() {
        // v = 0.25 m/s -> 128.5 (tie, even: 128); v = 0.75 -> 129.5 (tie, even: 130); -0.25 -> 127.5 -> 128.
        let got = encode_wind(&[0.25_f64, 0.75, -0.25, -0.75]);
        assert_eq!(got, vec![128, 130, 0, 128, 126, 0]);
        // +-63.75 m/s -> 255.5 -> 256 -> 255 and 0.5 -> 0; the edges +-1000 saturate.
        let got = encode_wind(&[63.75_f32, -63.75, 1000.0, -1000.0, f32::INFINITY, f32::NEG_INFINITY]);
        assert_eq!(got, vec![255, 0, 0, 255, 0, 0, 255, 0, 0]);
    }

    #[test]
    fn the_precision_of_the_wind_decides_the_tie() {
        // 0.25 + 2^-30 m/s: in float64 the sum 128.5 + 2^-29 is just above the tie and rounds up to 129; in
        // float32 the wind itself is 0.25 (2^-30 is under half an ulp), a tie that goes to the even 128.
        let v = 0.25_f64 + 2.0_f64.powi(-30);
        assert_eq!(encode_wind(&[v, 0.0])[0], 129);
        assert_eq!(encode_wind(&[v as f32, 0.0])[0], 128);
    }

    #[test]
    fn nan_gives_zero() {
        assert_eq!(encode_wind(&[f32::NAN, 1.0]), vec![0, 130, 0]);
    }

    #[test]
    #[should_panic]
    fn an_odd_length_is_rejected() {
        encode_wind(&[1.0_f32, 2.0, 3.0]);
    }
}

/// Unit tests of the maps: the behaviors the golden cases reach only through whole fields (the ones their
/// docs name as unreachable included) and the dtype contract of the generic signatures.
#[cfg(test)]
mod map_tests {
    use super::*;
    use crate::climate::EARTH;
    use crate::flow::{Vortex, VortexKind};

    fn rows(degrees: &[f64]) -> Vec<f64> {
        degrees.iter().map(|d| d.to_radians()).collect()
    }

    fn body(period: Option<f64>) -> BodyInputs {
        BodyInputs::new(period, EARTH_RADIUS_KM, EARTH_GRAVITY, EARTH_SCALE_HEIGHT_KM)
    }

    // ---- type_map ---------------------------------------------------------------------------------------

    #[test]
    fn type_levels_without_a_shield_are_the_four_cumulus_levels() {
        // Columns: all closed, all open, all street, none of the three (the broken remainder).
        let (closed, open, street) = ([1.0_f32, 0.0, 0.0, 0.0], [0.0_f32, 1.0, 0.0, 0.0], [0.0_f32, 0.0, 1.0, 0.0]);
        let lat = rows(&[0.0, 45.0, -80.0]);
        let tile = |f: &[f32]| f.iter().cycle().take(12).copied().collect::<Vec<f32>>();
        let got = type_map(&[0.0_f64; 12], &tile(&closed), &tile(&open), &tile(&street), &lat, 4, 30.0);
        for r in 0..3 {
            assert_eq!(&got[r * 4..r * 4 + 4], &[0.45_f32, 0.85, 0.8, 0.7], "row {r}");
        }
    }

    #[test]
    fn a_full_shield_is_the_convective_tendency_of_its_row() {
        let lat = rows(&[0.0, 17.0, 45.0, 90.0, -30.0]);
        let w = vec![0.2_f32; 5];
        let got = type_map(&[1.0_f64; 5], &w, &w, &w, &lat, 1, 30.0);
        for (r, &la) in lat.iter().enumerate() {
            let c = convective_tendency(la * (180.0 / PI), 30.0);
            assert_eq!(got[r], c as f32, "row {r}");
        }
        assert_eq!(got[0], 1.0, "the equator is clipped to exactly 1");
    }

    #[test]
    fn every_column_of_a_row_shares_its_latitude() {
        // Two rows, three columns, identical fields in every column: the columns of a row are identical.
        let lat = rows(&[10.0, 60.0]);
        let shield = [0.3_f64; 6];
        let w = [0.1_f32; 6];
        let got = type_map(&shield, &w, &w, &w, &lat, 3, 30.0);
        assert_eq!(got[0], got[1]);
        assert_eq!(got[1], got[2]);
        assert_eq!(got[3], got[5]);
        assert_ne!(got[0], got[3]);
    }

    #[test]
    fn out_of_contract_weights_are_only_bounded_by_the_final_clip() {
        // closed = 5 pushes the cumulus level to 5 * 0.45 + (1 - 5) * 0.7 = -0.55: clipped to 0. open = 3 -> 3 * 0.85
        // + (1 - 3) * 0.7 = 1.15: clipped to 1.
        let lat = rows(&[20.0]);
        let got = type_map(&[0.0_f64], &[5.0_f32], &[0.0], &[0.0], &lat, 1, 30.0);
        assert_eq!(got, vec![0.0]);
        let got = type_map(&[0.0_f64], &[0.0_f32], &[3.0], &[0.0], &lat, 1, 30.0);
        assert_eq!(got, vec![1.0]);
    }

    #[test]
    fn type_map_precisions() {
        let lat = rows(&[20.0]);
        // The cumulus level runs in the precision of the weights (the four levels rounded to float32 first), the
        // blend in the precision of the shield: spelled out for float32 weights 0.25, 0.5, 0.125.
        let cumulus32 = 0.25_f32 * 0.45_f32 + 0.5_f32 * 0.85_f32 + 0.125_f32 * 0.8_f32 + (1.0_f32 - 0.25 - 0.5 - 0.125) * 0.7_f32;
        let c = convective_tendency(lat[0] * (180.0 / PI), 30.0);
        let want = (0.5 * c + (1.0 - 0.5) * f64::from(cumulus32)).clamp(0.0, 1.0) as f32;
        let a = type_map(&[0.5_f64], &[0.25_f32], &[0.5], &[0.125], &lat, 1, 30.0);
        assert_eq!(a, vec![want]);
        // The same weights as float64 differ only by the float32 rounding of the levels.
        let b = type_map(&[0.5_f64], &[0.25_f64], &[0.5], &[0.125], &lat, 1, 30.0);
        assert!((a[0] - b[0]).abs() < 1e-6, "{} vs {}", a[0], b[0]);
        // An all-float32 map runs in float32: close to the float64 one, within a few float32 ulp.
        let d = type_map(&[0.5_f32], &[0.25_f32], &[0.5], &[0.125], &[lat[0] as f32], 1, 30.0);
        assert!((d[0] - b[0]).abs() < 1e-6, "{} vs {}", d[0], b[0]);
    }

    #[test]
    fn type_map_propagates_nan() {
        let lat = rows(&[20.0]);
        let w = [0.1_f32];
        assert!(type_map(&[f64::NAN], &w, &w, &w, &lat, 1, 30.0)[0].is_nan());
        assert!(type_map(&[0.5_f64], &[f32::NAN], &w, &w, &lat, 1, 30.0)[0].is_nan());
        assert!(type_map(&[0.5_f64], &w, &w, &w, &[f64::NAN], 1, 30.0)[0].is_nan());
    }

    #[test]
    #[should_panic(expected = "at least as precise")]
    fn type_map_rejects_weights_wider_than_the_shield() {
        type_map(&[0.5_f32], &[0.1_f64], &[0.1], &[0.1], &[0.0_f32], 1, 30.0);
    }

    #[test]
    #[should_panic(expected = "height x width")]
    fn type_map_rejects_a_field_of_the_wrong_size() {
        type_map(&[0.5_f64; 3], &[0.1_f32; 4], &[0.1; 4], &[0.1; 4], &[0.0, 0.1], 2, 30.0);
    }

    // ---- high_share -------------------------------------------------------------------------------------

    /// The share of a lone texel at latitude `lat_deg`.
    fn share(shield: f64, ascent: f32, lat_deg: f64, hadley: f64, coverage: f32) -> f32 {
        high_share(&[shield], &[ascent], &[lat_deg.to_radians()], 1, hadley, &[coverage])[0]
    }

    #[test]
    fn without_a_shield_the_share_is_the_modis_table_in_units_of_the_hadley_edge() {
        // Knots: x = 0 (equator), 1.5 (45 degrees at Hadley 30), 3 (90 degrees).
        assert_eq!(share(0.0, 0.5, 0.0, 30.0, 0.6), 0.33_f32);
        assert_eq!(share(0.0, 0.5, 45.0, 30.0, 0.6), 0.27_f32);
        assert_eq!(share(0.0, 0.5, 90.0, 30.0, 0.6), 0.2_f32);
        // Between knots: x = 0.5 is half way from 0.33 (0.39) to 0.67 (0.25) at (0.5 - 0.33) / 0.34 = 0.5.
        let want = 0.39 + (0.25 - 0.39) * ((0.5 - 0.33) / (0.67 - 0.33));
        assert!((f64::from(share(0.0, 0.5, 15.0, 30.0, 0.6)) - want).abs() < 1e-6);
        // Past the last knot the table clamps (Hadley 10: x = 9), and so below the first one (south: |lat|).
        assert_eq!(share(0.0, 0.5, 90.0, 10.0, 0.6), 0.2_f32);
        assert_eq!(share(0.0, 0.5, -45.0, 30.0, 0.6), 0.27_f32);
    }

    #[test]
    fn the_shield_adds_the_comma_and_deep_core_terms() {
        // Equator, Hadley 30: c = 1, so the deep core is 0.15; a full shield (shield / coverage clipped to 1) with a
        // full ascent adds 0.25 + 0.15 = 0.4 to extra: 0.33 + 0.67 * 0.4.
        let want = 0.33 + (1.0 - 0.33) * 0.4;
        assert!((f64::from(share(5.0, 2.0, 0.0, 30.0, 1.0)) - want).abs() < 1e-6);
        // A negative ascent is clipped to 0: only the deep core remains.
        let want = 0.33 + (1.0 - 0.33) * 0.15;
        assert!((f64::from(share(5.0, -3.0, 0.0, 30.0, 1.0)) - want).abs() < 1e-6);
        // Coverage is floored at 0.05: shield 0.025 over coverage 0 is half a shield.
        let want = 0.33 + (1.0 - 0.33) * (0.5 * 0.15);
        assert!((f64::from(share(0.025, 0.0, 0.0, 30.0, 0.0)) - want).abs() < 1e-6);
        // A negative shield is clipped to 0: the bare table.
        assert_eq!(share(-0.5, 1.0, 0.0, 30.0, 0.6), 0.33_f32);
    }

    #[test]
    fn high_share_precisions_and_nan() {
        let lat = rows(&[10.0, 40.0]);
        let (shield, ascent, coverage) = ([0.4_f64, 0.2], [0.5_f32, 0.9], [0.6_f32, 0.3]);
        let f32s = high_share(&shield, &ascent, &lat, 1, 30.0, &coverage);
        let (ascent64, coverage64) = ([0.5_f64, f64::from(0.9_f32)], [0.6_f64, f64::from(0.3_f32)]);
        let f64s = high_share(&shield, &ascent64, &lat, 1, 30.0, &coverage64);
        for (a, b) in f32s.iter().zip(&f64s) {
            // The float32 coverage floor (0.05 rounded to float32) and the float32 clip move it by float32 ulps.
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        assert!(share(0.4, 0.5, f64::NAN, 30.0, 0.6).is_nan());
        assert!(share(f64::NAN, 0.5, 20.0, 30.0, 0.6).is_nan());
        assert!(share(0.4, f32::NAN, 20.0, 30.0, 0.6).is_nan());
        assert!(share(0.4, 0.5, 20.0, 30.0, f32::NAN).is_nan());
    }

    #[test]
    #[should_panic(expected = "at least as precise")]
    fn high_share_rejects_an_ascent_wider_than_the_shield() {
        high_share(&[0.5_f32], &[0.5_f64], &[0.0_f32], 1, 30.0, &[0.5_f32]);
    }

    // ---- the zonal wind ---------------------------------------------------------------------------------

    #[test]
    fn earth_is_the_unit_of_the_zonal_wind_scale() {
        assert_eq!(zonal_wind_scale(&EARTH), 1.0);
        // The scale reads |period|: a retrograde Earth has the same scale (and the opposite spin).
        assert_eq!(zonal_wind_scale(&body(Some(-EARTH_ROTATION_DAYS))), 1.0);
        // No spin, a slow spin (global Hadley cell): no storm track, no wind.
        for p in [None, Some(0.0), Some(f64::NAN), Some(f64::INFINITY), Some(243.0), Some(-243.0)] {
            assert_eq!(zonal_wind_scale(&body(p)), 0.0, "{p:?}");
        }
        // g H floors at 0.
        let mut b = EARTH;
        b.gravity_mps2 = -3.0;
        assert_eq!(zonal_wind_scale(&b), 0.0);
        // A 0.4-day Earth turns 2.5 times as fast: g H / (Omega a) = 0.4.
        assert!((zonal_wind_scale(&body(Some(0.4))) - 0.4 / EARTH_ROTATION_DAYS).abs() < 1e-12);
    }

    #[test]
    fn the_zonal_profile_has_its_three_belts_and_the_spin_sign() {
        // Westerlies on the storm track (x = 1.5, 45 degrees), trades at x = 0.45 (13.5 degrees), polar easterlies.
        let lat = rows(&[45.0, 13.5, 90.0, -45.0, 0.0]);
        let u = climatological_zonal_mps(&lat, 30.0, &EARTH);
        assert!((u[0] - 12.0).abs() < 1e-2, "{}", u[0]);
        assert!((u[1] + 7.0).abs() < 0.1, "{}", u[1]);
        assert!((u[2] + 4.0).abs() < 1e-3, "{}", u[2]);
        assert_eq!(u[0], u[3], "symmetric in |lat|");
        assert!(u[4] < 0.0 && u[4] > -7.0, "the equator is on the trades' flank: {}", u[4]);
        // Retrograde: the whole profile turns round.
        let r = climatological_zonal_mps(&lat, 30.0, &body(Some(-EARTH_ROTATION_DAYS)));
        for (a, b) in u.iter().zip(&r) {
            assert_eq!(*a, -*b);
        }
        // No storm track: exactly zero, times the spin -1 of a retrograde body: a negative zero under the positive
        // profile (45 degrees), a positive one under the negative profile (the trades at 13.5 degrees).
        let none = climatological_zonal_mps(&lat, 30.0, &body(Some(-243.0)));
        assert!(none.iter().all(|&v| v == 0.0));
        assert!(none[0].is_sign_negative() && none[1].is_sign_positive());
    }

    #[test]
    fn the_zonal_profile_runs_in_the_precision_of_the_latitude() {
        let lat64 = rows(&[0.0, 20.0, 45.0, 70.0, 90.0]);
        let lat32: Vec<f32> = lat64.iter().map(|&v| v as f32).collect();
        let u64s = climatological_zonal_mps(&lat64, 30.0, &EARTH);
        let u32s = climatological_zonal_mps(&lat32, 30.0, &EARTH);
        for (a, b) in u64s.iter().zip(&u32s) {
            assert!((a - f64::from(*b)).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn the_spin_sign_follows_python_copysign_of_period_or_one() {
        assert_eq!(spin_sign(&body(None)), 1.0);
        assert_eq!(spin_sign(&body(Some(0.0))), 1.0);
        assert_eq!(spin_sign(&body(Some(-0.0))), 1.0, "-0.0 is falsy too");
        assert_eq!(spin_sign(&body(Some(-5.0))), -1.0);
        assert_eq!(spin_sign(&body(Some(f64::NEG_INFINITY))), -1.0);
        assert_eq!(spin_sign(&body(Some(f64::NAN))), 1.0, "a NaN is truthy: the sign of its own bit");
        assert_eq!(spin_sign(&body(Some(-f64::NAN))), -1.0);
    }

    // ---- wind_map ---------------------------------------------------------------------------------------

    #[test]
    fn the_wind_grid_has_width_over_two_rows() {
        let flow = Flow::new(30.0);
        for (width, height) in [(64_usize, 32_usize), (33, 16), (2, 1), (1, 0), (0, 0)] {
            let w = wind_map(&flow, &EARTH, width);
            assert_eq!(w.len(), height * width * 2, "width {width}");
        }
    }

    #[test]
    fn without_vortices_the_wind_is_the_zonal_profile_alone() {
        let flow = Flow::new(30.0);
        let (width, height) = (16, 8);
        let w = wind_map(&flow, &EARTH, width);
        let (_, lat) = grid_lonlat(width, height);
        let zonal = climatological_zonal_mps(&lat, 30.0, &EARTH);
        for r in 0..height {
            for c in 0..width {
                let i = (r * width + c) * 2;
                assert_eq!(w[i], zonal[r] as f32, "east, row {r} column {c}");
                assert_eq!(w[i + 1], 0.0, "north, row {r} column {c}");
            }
        }
    }

    #[test]
    fn the_flow_s_own_zonal_terms_are_ignored() {
        let mut flow = Flow::new(30.0);
        flow.vortices.push(Vortex::new(VortexKind::Cyclone, 0.5, 0.9, 0.2, 3.0, 1.0));
        let plain = wind_map(&flow, &EARTH, 32);
        flow.jet = 0.9;
        flow.trades = 0.7;
        flow.polar_easterlies = 0.5;
        assert_eq!(wind_map(&flow, &EARTH, 32), plain);
    }

    #[test]
    fn a_counterclockwise_vortex_blows_north_on_its_east_flank_in_metres_per_second() {
        // No storm track (a slow body), so only the vortex blows. A cyclone (omega > 0) on the equator at lon 0.
        let slow = BodyInputs::new(Some(243.0), EARTH_RADIUS_KM, EARTH_GRAVITY, EARTH_SCALE_HEIGHT_KM);
        let mut flow = Flow::new(90.0);
        flow.vortices.push(Vortex::new(VortexKind::Cyclone, 0.0, 0.0, 0.5, 3.0, 1.0));
        let (width, height) = (64, 32);
        let w = wind_map(&flow, &slow, width);
        let north = |row: usize, col: usize| w[(row * width + col) * 2 + 1];
        // Row 16 is 1.4 degrees south of the equator; columns 34 / 29 are 14 degrees east / west of the center.
        let (east_flank, west_flank) = (north(height / 2, 34), north(height / 2, 29));
        assert!(east_flank > 5.0 && east_flank < 25.0, "{east_flank}");
        assert!(west_flank < -5.0 && west_flank > -25.0, "{west_flank}");
        // The flow of a vortex dies out far away (the antipode is ~1.2e-2 of e-folding radii off).
        assert!(north(height / 2, 0).abs() < 1.0);
    }

    #[test]
    fn a_retrograde_body_flips_only_the_zonal_part_of_the_wind() {
        let mut flow = Flow::new(30.0);
        flow.vortices.push(Vortex::new(VortexKind::Cyclone, 0.5, 0.9, 0.2, 3.0, 1.0));
        let prograde = wind_components(&flow, &EARTH, 32);
        let retrograde = wind_components(&flow, &body(Some(-EARTH_ROTATION_DAYS)), 32);
        assert_eq!(prograde.vortex_east, retrograde.vortex_east);
        assert_eq!(prograde.vortex_north, retrograde.vortex_north);
        for (a, b) in prograde.zonal.iter().zip(&retrograde.zonal) {
            assert_eq!(*a, -*b);
        }
    }

    #[test]
    fn wind_map_is_the_components_added_up_in_float32() {
        let mut flow = Flow::new(30.0);
        flow.vortices.push(Vortex::new(VortexKind::Tropical, -3.0, 0.3, 0.1, 9.0, 1.6));
        let parts = wind_components(&flow, &EARTH, 20);
        let w = wind_map(&flow, &EARTH, 20);
        assert_eq!((parts.height, parts.width), (10, 20));
        for r in 0..10 {
            for c in 0..20 {
                let i = r * 20 + c;
                assert_eq!(w[2 * i], (parts.vortex_east[i] + parts.zonal[r]) as f32);
                assert_eq!(w[2 * i + 1], parts.vortex_north[i] as f32);
            }
        }
    }

    #[test]
    fn the_east_frame_falls_back_to_plus_y_at_the_poles_and_is_a_unit_vector_elsewhere() {
        // The pole: z x z = 0, shorter than 1e-9. (Unreachable from wind_map, whose rows are cell-centered.)
        assert_eq!(east_unit([0.0, 0.0, 1.0]), [0.0, 1.0, 0.0]);
        assert_eq!(east_unit([0.0, 0.0, -1.0]), [0.0, 1.0, 0.0]);
        // Just off the pole (cross product 5e-10 long) it still falls back; at 2e-9 it is the true direction.
        assert_eq!(east_unit([5e-10, 0.0, 1.0]), [0.0, 1.0, 0.0]);
        let e = east_unit([2e-9, 0.0, 1.0]);
        assert!((e[1] - 1.0).abs() < 1e-12 && e[0].abs() < 1e-12);
        // At lon 0, lat 0 east is +y; at lon 90 it is -x.
        assert_eq!(east_unit([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
        let e = east_unit([0.0, 1.0, 0.0]);
        assert!((e[0] + 1.0).abs() < 1e-15 && e[1].abs() < 1e-15 && e[2] == 0.0);
    }

    #[test]
    fn constants_keep_their_values() {
        assert_eq!(ADVECTION_DAYS, 3.0);
        assert_eq!([TYPE_CLOSED, TYPE_OPEN, TYPE_STREET, TYPE_BROKEN], [0.45, 0.85, 0.8, 0.7]);
        assert_eq!((HIGH_COMMA_ASCENT, HIGH_DEEP_CORE), (0.25, 0.15));
        assert!(HIGH_BASE_X.windows(2).all(|w| w[0] < w[1]), "np.interp needs increasing abscissae");
        assert_eq!(HIGH_BASE_X.len(), HIGH_BASE_SHARE.len());
    }
}
