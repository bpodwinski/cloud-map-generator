//! Golden cases of cloudgen/products.py (group products_generate, functions cloudgen.products.*): the
//! constants (a pin case), type_map, high_share, zonal_wind_scale, climatological_zonal_mps, wind_map and
//! encode_wind (which products.rs took over from the export module and the cases call through here).
//!
//! The cases that need a word more:
//!
//! * `type_map` and `high_share` take their latitude as one value per row (`lat_rows`), exactly as the cases
//!   freeze it ("broadcast over the columns as generate() does"); the port evaluates the tendency once per row.
//!   The dtype of each group of arrays is the case's own (float64 shield and latitudes with float32 weights, ascent
//!   and coverage in the pipeline; all float32 or all float64 in the dtype cases): the port is generic over them.
//! * `wind_map` has three float64 intermediates next to the float32 wind (the zonal profile of the rows and the
//!   vortices' own east and north flow). The port exposes them as `wind_components`; the glue returns them from
//!   there and the wind from `wind_map`, and checks that the wind is the intermediates added up and rounded.
//! * `zonal_wind_scale.nonfinite_rotation` holds the rotation periods in an input (JSON cannot carry
//!   infinities and NaN); a NaN is `Some(NaN)` there, a period that is a number and not "no spin".
//! * A rotation period of `null` in a body record is `None`.
//!
//! What the cases cannot see is pinned by the tests after the runner's.

use cloud_map_generator_core::climate::{BodyInputs, EARTH_GRAVITY, EARTH_RADIUS_KM, EARTH_ROTATION_DAYS, EARTH_SCALE_HEIGHT_KM};
use cloud_map_generator_core::flow::{Flow, Vortex, VortexKind};
use cloud_map_generator_core::products::*;
use cloud_map_generator_golden::{run, Array, Case, Config, DType, Out};
use num_traits::Float;
use serde_json::Value;

const EXPECTED_FAILURES: &[(&str, &str)] = &[];

// ---------------------------------------------------------------------------------------------------------
// Reading cases

/// A float dtype the cases freeze (f32 or f64): reads an array and writes an output of that dtype.
trait Elem: Float + Send + Sync {
    fn read(a: &Array) -> Vec<Self>;
    fn put(out: Out, name: &str, shape: &[usize], data: &[Self]) -> Out;
}
impl Elem for f32 {
    fn read(a: &Array) -> Vec<f32> {
        a.as_f32()
    }
    fn put(out: Out, name: &str, shape: &[usize], data: &[f32]) -> Out {
        out.f32(name, shape, data)
    }
}
impl Elem for f64 {
    fn read(a: &Array) -> Vec<f64> {
        a.as_f64()
    }
    fn put(out: Out, name: &str, shape: &[usize], data: &[f64]) -> Out {
        out.f64(name, shape, data)
    }
}

/// The `(height, width)` of a 2-D field input.
fn dims(c: &Case, name: &str) -> (usize, usize) {
    let a = c.input(name);
    assert_eq!(a.shape.len(), 2, "case {}: {name} is (height, width)", c.id);
    (a.shape[0], a.shape[1])
}

/// The row latitudes of a field case: one per row, in the dtype of the shield.
fn lat_rows_of<S: Elem>(c: &Case, height: usize) -> Vec<S> {
    let lat = c.input("lat_rows");
    assert_eq!(lat.shape, [height], "case {}: lat_rows is one latitude per row", c.id);
    S::read(lat)
}

/// A body record of `params` (`name` is only a label; a `null` period is `None`).
fn body_of(v: &Value) -> BodyInputs {
    let f = |k: &str| v[k].as_f64().unwrap_or_else(|| panic!("body field {k} is not a number in {v}"));
    BodyInputs::new(v["rotation_period_days"].as_f64(), f("radius_km"), f("gravity_mps2"), f("scale_height_km"))
        .with_regime(v["cloud_regime"].as_str().expect("cloud_regime is a string").to_string())
}

fn kind_of(name: &str) -> VortexKind {
    match name {
        "cyclone" => VortexKind::Cyclone,
        "anticyclone" => VortexKind::Anticyclone,
        "tropical" => VortexKind::Tropical,
        other => panic!("unknown vortex kind {other:?}"),
    }
}

/// The hand-built flow of a wind_map case: `hadley_deg`, the three shear scalars and the vortex list.
fn flow_of(c: &Case) -> Flow {
    let vortices = c.param("vortices").as_array().unwrap_or_else(|| panic!("case {}: vortices is not a list", c.id));
    let vortex = |v: &Value| {
        let f = |k: &str| v[k].as_f64().unwrap_or_else(|| panic!("vortex field {k} is not a number in {v}"));
        Vortex {
            kind: kind_of(v["kind"].as_str().expect("vortex kind")),
            lon: f("lon"),
            lat: f("lat"),
            radius: f("radius"),
            omega: f("omega"),
            ascent: f("ascent"),
            age: f("age"),
        }
    };
    Flow {
        hadley_deg: c.p_f64("hadley_deg"),
        vortices: vortices.iter().map(vortex).collect(),
        jet: c.p_f64("jet"),
        trades: c.p_f64("trades"),
        polar_easterlies: c.p_f64("polar_easterlies"),
    }
}

// ---------------------------------------------------------------------------------------------------------
// Cases

/// The pin case: the port's own constants must be the reference table (compared bit for bit by the runner).
fn constants_case() -> Out {
    Out::new()
        .f64("high_base_x", &[HIGH_BASE_X.len()], &HIGH_BASE_X)
        .f64("high_base_share", &[HIGH_BASE_SHARE.len()], &HIGH_BASE_SHARE)
        .f64("type_levels", &[4], &[TYPE_CLOSED, TYPE_OPEN, TYPE_STREET, TYPE_BROKEN])
        .f64("scalars", &[4], &[ADVECTION_DAYS, WIND_STEP_MPS, HIGH_COMMA_ASCENT, HIGH_DEEP_CORE])
        .f64("earth", &[4], &[EARTH_ROTATION_DAYS, EARTH_RADIUS_KM, EARTH_GRAVITY, EARTH_SCALE_HEIGHT_KM])
}

fn encode_case<F: Elem>(c: &Case) -> Out {
    let a = c.input("wind");
    assert!(a.shape.len() == 3 && a.shape[2] == 2, "case {}: wind is (height, width, 2)", c.id);
    let rgb = encode_wind(&F::read(a));
    Out::new().u8("rgb", &[a.shape[0], a.shape[1], 3], &rgb)
}

fn type_case<S: Elem, W: Elem>(c: &Case) -> Out {
    let (h, w) = dims(c, "shield");
    assert_eq!(c.input("lat_rows").dtype, c.input("shield").dtype, "case {}: shield and latitudes share a dtype", c.id);
    let weight = |name: &str| {
        assert_eq!(dims(c, name), (h, w), "case {}: {name} has the shape of the shield", c.id);
        W::read(c.input(name))
    };
    let got = type_map(
        &S::read(c.input("shield")),
        &weight("w_closed"),
        &weight("w_open"),
        &weight("w_street"),
        &lat_rows_of::<S>(c, h),
        w,
        c.p_f64("hadley_deg"),
    );
    Out::new().f32("type", &[h, w], &got)
}

fn high_case<S: Elem, A: Elem, C: Elem>(c: &Case) -> Out {
    let (h, w) = dims(c, "shield");
    assert_eq!(c.input("lat_rows").dtype, c.input("shield").dtype, "case {}: shield and latitudes share a dtype", c.id);
    for name in ["ascent", "coverage"] {
        assert_eq!(dims(c, name), (h, w), "case {}: {name} has the shape of the shield", c.id);
    }
    let got = high_share(
        &S::read(c.input("shield")),
        &A::read(c.input("ascent")),
        &lat_rows_of::<S>(c, h),
        w,
        c.p_f64("hadley_deg"),
        &C::read(c.input("coverage")),
    );
    Out::new().f32("high_share", &[h, w], &got)
}

fn scale_case(c: &Case) -> Out {
    let scales: Vec<f64> = if c.has_input("rotation") {
        // One body per rotation period of the input; the rest of the record is shared.
        let shared = |k: &str| c.p_f64(k);
        c.input("rotation")
            .as_f64()
            .into_iter()
            .map(|p| {
                let b = BodyInputs::new(Some(p), shared("radius_km"), shared("gravity_mps2"), shared("scale_height_km"))
                    .with_regime(c.p_str("cloud_regime").to_string());
                zonal_wind_scale(&b)
            })
            .collect()
    } else {
        let bodies = c.param("bodies").as_array().expect("params.bodies is a list");
        bodies.iter().map(|b| zonal_wind_scale(&body_of(b))).collect()
    };
    Out::new().f64("scale", &[scales.len()], &scales)
}

fn zonal_case<F: Elem>(c: &Case) -> Out {
    let lat = c.input("lat");
    assert_eq!(lat.shape.len(), 1, "case {}: lat is a vector", c.id);
    let u = climatological_zonal_mps(&F::read(lat), c.p_f64("hadley_deg"), &body_of(c.param("body")));
    F::put(Out::new(), "u", &lat.shape, &u)
}

fn wind_case(c: &Case) -> Out {
    let (flow, body, width) = (flow_of(c), body_of(c.param("body")), c.p_usize("width"));
    let parts = wind_components(&flow, &body, width);
    let (h, w) = (parts.height, parts.width);
    let wind = wind_map(&flow, &body, width);
    // The wind is the intermediates added up (east: the zonal row profile on top) and rounded to float32 once.
    for r in 0..h {
        for col in 0..w {
            let i = r * w + col;
            assert_eq!(wind[2 * i], (parts.vortex_east[i] + parts.zonal[r]) as f32, "case {}: east {r},{col}", c.id);
            assert_eq!(wind[2 * i + 1], parts.vortex_north[i] as f32, "case {}: north {r},{col}", c.id);
        }
    }
    Out::new()
        .f32("wind", &[h, w, 2], &wind)
        .f64("zonal", &[h], &parts.zonal)
        .f64("vortex_east", &[h, w], &parts.vortex_east)
        .f64("vortex_north", &[h, w], &parts.vortex_north)
}

fn port(c: &Case) -> Option<Out> {
    use DType::{F32, F64};
    let dtype = |name: &str| c.input(name).dtype;
    Some(match c.function.as_str() {
        "cloudgen.products.constants" => constants_case(),
        "cloudgen.products.encode_wind" => match dtype("wind") {
            F32 => encode_case::<f32>(c),
            F64 => encode_case::<f64>(c),
            other => panic!("case {}: wind is {}", c.id, other.name()),
        },
        "cloudgen.products.type_map" => match (dtype("shield"), dtype("w_closed")) {
            (F64, F32) => type_case::<f64, f32>(c),
            (F32, F32) => type_case::<f32, f32>(c),
            (s, w) => panic!("case {}: no glue for a {} shield with {} weights", c.id, s.name(), w.name()),
        },
        "cloudgen.products.high_share" => match (dtype("shield"), dtype("ascent"), dtype("coverage")) {
            (F64, F32, F32) => high_case::<f64, f32, f32>(c),
            (F64, F64, F64) => high_case::<f64, f64, f64>(c),
            (s, a, v) => panic!("case {}: no glue for {} shield, {} ascent, {} coverage", c.id, s.name(), a.name(), v.name()),
        },
        "cloudgen.products.zonal_wind_scale" => scale_case(c),
        "cloudgen.products.climatological_zonal_mps" => match dtype("lat") {
            F64 => zonal_case::<f64>(c),
            F32 => zonal_case::<f32>(c),
            other => panic!("case {}: lat is {}", c.id, other.name()),
        },
        "cloudgen.products.wind_map" => wind_case(c),
        _ => return None,
    })
}

#[test]
fn products_functions() {
    run(&Config { group: "products_generate", prefixes: &["cloudgen.products."], expected_failures: EXPECTED_FAILURES }, port)
        .assert_clean();
}

// ---------------------------------------------------------------------------------------------------------
// What the cases cannot see

/// A cyclone exactly on the antimeridian (lon = +pi or -pi, the same point up to 1.2e-16): the field is
/// mirror-symmetric about the seam, the east component of column `i` equal to that of column `w - 1 - i` and
/// the north components opposite, and the two signs of pi give the same map (the case doc says Python checked
/// this exactly; the port, with other libm trigonometry, within 1e-9 m/s).
#[test]
fn a_vortex_on_the_antimeridian_blows_the_same_either_side_of_the_seam() {
    let slow = BodyInputs::new(Some(-243.0), 6051.8, 8.87, 15.9);
    let map = |lon: f64| {
        let mut flow = Flow::new(90.0);
        flow.vortices.push(Vortex::new(VortexKind::Cyclone, lon, 0.2, 0.2, 3.0, 1.0));
        wind_components(&flow, &slow, 64)
    };
    let (plus, minus) = (map(std::f64::consts::PI), map(-std::f64::consts::PI));
    let (h, w) = (plus.height, plus.width);
    for r in 0..h {
        for i in 0..w {
            let (a, m) = (r * w + i, r * w + (w - 1 - i));
            assert!((plus.vortex_east[a] - plus.vortex_east[m]).abs() < 1e-9, "east {r},{i}");
            assert!((plus.vortex_north[a] + plus.vortex_north[m]).abs() < 1e-9, "north {r},{i}");
            assert!((plus.vortex_east[a] - minus.vortex_east[a]).abs() < 1e-9, "east +/-pi {r},{i}");
            assert!((plus.vortex_north[a] - minus.vortex_north[a]).abs() < 1e-9, "north +/-pi {r},{i}");
        }
    }
}
