//! Golden cases of the stage functions of cloudgen/generate.py (group products_generate, functions
//! cloudgen.generate.*): compose, solve_thresholds, row_sensitivity, moisture_profile, _smoothstep,
//! area_mean, along, surface_regime_weights and the GenParams defaults. generate() itself is not covered
//! here (the pipeline group freezes it).

use cloud_map_generator_core::generate::*;
use cloud_map_generator_core::sphere::Vec3;
use cloud_map_generator_golden::{run, Array, Case, Config, DType, Out};
use num_traits::Float;
use serde_json::json;

const EXPECTED_FAILURES: &[(&str, &str)] = &[];

/// A float dtype the cases freeze (f32 or f64): reads an array and writes an output of that dtype.
trait Fld: Float + Send + Sync {
    fn read(a: &Array) -> Vec<Self>;
    fn put(out: Out, name: &str, shape: &[usize], data: &[Self]) -> Out;
}
impl Fld for f32 {
    fn read(a: &Array) -> Vec<f32> {
        a.as_f32()
    }
    fn put(out: Out, name: &str, shape: &[usize], data: &[f32]) -> Out {
        out.f32(name, shape, data)
    }
}
impl Fld for f64 {
    fn read(a: &Array) -> Vec<f64> {
        a.as_f64()
    }
    fn put(out: Out, name: &str, shape: &[usize], data: &[f64]) -> Out {
        out.f64(name, shape, data)
    }
}

/// Runs `$body::<F>($c)` with F the dtype of the array input `$name` (the case's field dtype).
macro_rules! by_dtype {
    ($c:expr, $name:expr, $f:ident) => {
        match $c.input($name).dtype {
            DType::F32 => $f::<f32>($c),
            DType::F64 => $f::<f64>($c),
            other => panic!("case {}: input {} has dtype {}", $c.id, $name, other.name()),
        }
    };
}

/// The GenParams of a case: the defaults, with the fields of `params.prm` overridden.
fn prm_of(c: &Case) -> GenParams {
    let mut p = GenParams::default();
    for (name, v) in c.param("prm").as_object().expect("prm is an object") {
        let x = v.as_f64().unwrap_or_else(|| panic!("case {}: prm.{name} is not a number", c.id));
        match name.as_str() {
            "shield_ramp" => p.shield_ramp = x,
            "shield_texture" => p.shield_texture = x,
            "cumulus_max" => p.cumulus_max = x,
            "cumulus_depth" => p.cumulus_depth = x,
            "a_fine" => p.a_fine = x,
            "sensitivity_step" => p.sensitivity_step = x,
            other => panic!("case {}: unknown prm field {other}", c.id),
        }
    }
    p
}

/// The fields every compose-family case carries: `s`, `fine`, the optional streak array (or the scalar
/// `params.streak`, default 0.0), the optional subtexel array and the GenParams.
struct Fields<F> {
    s: Vec<F>,
    fine: Vec<F>,
    streak_array: Option<Vec<F>>,
    streak_scalar: f64,
    subtexel: Option<Vec<F>>,
    height: usize,
    width: usize,
    prm: GenParams,
}

impl<F: Fld> Fields<F> {
    fn load(c: &Case) -> Fields<F> {
        let s = c.input("s");
        assert_eq!(s.shape.len(), 2, "case {}: s is (height, width)", c.id);
        let optional = |name: &str| c.has_input(name).then(|| F::read(c.input(name)));
        Fields {
            s: F::read(s),
            fine: F::read(c.input("fine")),
            streak_array: optional("streak"),
            streak_scalar: if c.has_param("streak") { c.p_f64("streak") } else { 0.0 },
            subtexel: optional("subtexel"),
            height: s.shape[0],
            width: s.shape[1],
            prm: prm_of(c),
        }
    }
    fn streak(&self) -> Streak<'_, F> {
        match &self.streak_array {
            Some(a) => Streak::Array(a),
            None => Streak::Scalar(self.streak_scalar),
        }
    }
    fn subtexel(&self) -> Option<&[F]> {
        self.subtexel.as_deref()
    }
}

fn compose_case<F: Fld>(c: &Case) -> Out {
    let f = Fields::<F>::load(c);
    let t_data: Vec<f64> = if c.has_input("t") { c.input("t").as_f64() } else { Vec::new() };
    let t = if c.has_input("t") {
        if c.input("t").shape == [f.height, 1] {
            Threshold::Column(&t_data)
        } else {
            Threshold::Full(&t_data)
        }
    } else {
        Threshold::Scalar(c.p_f64("t"))
    };
    let result = compose(&f.s, &f.fine, f.height, f.width, t, &f.prm, f.streak(), f.subtexel());
    // The dtype numpy promotes to is part of the contract: the case says which one it froze.
    assert_eq!(matches!(result, Composed::F32(_)), c.p_str("result_dtype") == "f32", "result dtype of {}", c.id);
    let shape = [f.height, f.width];
    match result {
        Composed::F32(v) => Out::new().f32("coverage", &shape, &v),
        Composed::F64(v) => Out::new().f64("coverage", &shape, &v),
    }
}

fn solve_offset<F: Fld, O: Float + Send + Sync>(f: &Fields<F>, target: &[f64], offset: Offset<'_, O>) -> Vec<f64> {
    solve_thresholds(&f.s, &f.fine, f.height, f.width, target, &f.prm, f.streak(), f.subtexel(), offset)
}

fn solve_case<F: Fld>(c: &Case) -> Out {
    let f = Fields::<F>::load(c);
    let target = c.input("target").as_f64();
    let thr = if c.has_input("offset") {
        // A per-row column (height, 1) or a per-texel field (height, width), float32 as the pipeline holds it.
        let o = c.input("offset");
        let column = o.shape == [f.height, 1];
        match o.dtype {
            DType::F32 => {
                let d = o.as_f32();
                solve_offset(&f, &target, if column { Offset::Column(&d) } else { Offset::Full(&d) })
            }
            DType::F64 => {
                let d = o.as_f64();
                solve_offset(&f, &target, if column { Offset::Column(&d) } else { Offset::Full(&d) })
            }
            other => panic!("case {}: offset dtype {}", c.id, other.name()),
        }
    } else {
        let o = if c.has_param("offset") { c.p_f64("offset") } else { 0.0 };
        solve_offset(&f, &target, Offset::<f32>::Scalar(o))
    };
    Out::new().f64("thr", &[f.height], &thr)
}

fn sensitivity_case<F: Fld>(c: &Case) -> Out {
    let f = Fields::<F>::load(c);
    let thr = c.input("thr").as_f64();
    let sens = row_sensitivity(&f.s, &f.fine, f.height, f.width, &thr, &f.prm, f.streak(), f.subtexel());
    Out::new().f64("sens", &[f.height], &sens)
}

fn moisture_case<F: Fld>(c: &Case) -> Out {
    let lat = F::read(c.input("lat"));
    let profile = moisture_profile(&lat, c.p_f64("hadley_deg"));
    F::put(Out::new(), "profile", &c.input("lat").shape, &profile)
}

fn smoothstep_case<F: Fld>(c: &Case) -> Out {
    let x = F::read(c.input("x"));
    let y: Vec<F> = x.iter().map(|&v| smoothstep(c.p_f64("a"), c.p_f64("b"), v)).collect();
    F::put(Out::new(), "y", &c.input("x").shape, &y)
}

fn area_case<F: Fld>(c: &Case) -> Out {
    let a = c.input("a");
    let mean = area_mean(&F::read(a), a.shape[0], a.shape[1]);
    Out::new().f64("mean", &[1], &[mean])
}

fn vec3s<F: Fld>(a: &Array) -> Vec<Vec3<F>> {
    F::read(a).chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect()
}

fn along_case<F: Fld>(c: &Case) -> Out {
    let (p, back) = (vec3s::<F>(c.input("p")), vec3s::<F>(c.input("back")));
    let shape = &c.input("p").shape;
    let flat = |v: Vec<Vec3<F>>| -> Vec<F> { v.into_iter().flatten().collect() };
    let mut out = F::put(Out::new(), "q", shape, &flat(along(&p, &back, c.p_f64("f"))));
    if c.has_param("f_neg") {
        out = F::put(out, "q_neg", shape, &flat(along(&p, &back, c.p_f64("f_neg"))));
    }
    out
}

fn regime_case(c: &Case) -> Out {
    let f32s = |name: &str| c.input(name).as_f32();
    let shape = c.input("w_closed").shape.clone();
    let lat_rows = c.input("lat_rows").as_f64();
    let w = surface_regime_weights(
        &f32s("w_closed"),
        &f32s("w_open"),
        &f32s("w_street"),
        &f32s("continentality"),
        &f32s("deck_strength"),
        &lat_rows,
        shape[1],
        c.p_f64("hadley_deg"),
    );
    Out::new().f32("closed", &shape, &w.closed).f32("open", &shape, &w.open).f32("street", &shape, &w.street)
}

/// The GenParams pin: every field of the Rust struct against the Python dataclass's `asdict`. The
/// exhaustive destructuring stops compiling when a field is added or removed on the Rust side; the count of
/// the case's params catches one added or removed on the Python side; a changed default fails the equality.
fn genparams_case(c: &Case) -> Out {
    let GenParams {
        structure_width,
        a_moisture,
        a_ascent,
        a_mid,
        a_fine,
        moisture_noise,
        advect_mid,
        advect_fine,
        mid_frequency,
        fine_frequency,
        cumulus_cell_km,
        fine_top,
        fine_gain,
        fine_billow,
        fine_min_texels,
        shield_ramp,
        cumulus_depth,
        cumulus_max,
        shield_texture,
        threshold_smooth_deg,
        regime_texture,
        isotropic,
        allow_any_regime,
        surface_terms,
        sensitivity_step,
    } = GenParams::default();
    let fields = [
        ("structure_width", json!(structure_width)),
        ("a_moisture", json!(a_moisture)),
        ("a_ascent", json!(a_ascent)),
        ("a_mid", json!(a_mid)),
        ("a_fine", json!(a_fine)),
        ("moisture_noise", json!(moisture_noise)),
        ("advect_mid", json!(advect_mid)),
        ("advect_fine", json!(advect_fine)),
        ("mid_frequency", json!(mid_frequency)),
        ("fine_frequency", json!(fine_frequency)),
        ("cumulus_cell_km", json!(cumulus_cell_km)),
        ("fine_top", json!(fine_top)),
        ("fine_gain", json!(fine_gain)),
        ("fine_billow", json!(fine_billow)),
        ("fine_min_texels", json!(fine_min_texels)),
        ("shield_ramp", json!(shield_ramp)),
        ("cumulus_depth", json!(cumulus_depth)),
        ("cumulus_max", json!(cumulus_max)),
        ("shield_texture", json!(shield_texture)),
        ("threshold_smooth_deg", json!(threshold_smooth_deg)),
        ("regime_texture", json!(regime_texture)),
        ("isotropic", json!(isotropic)),
        ("allow_any_regime", json!(allow_any_regime)),
        ("surface_terms", json!(surface_terms)),
        ("sensitivity_step", json!(sensitivity_step)),
    ];
    for (name, value) in &fields {
        assert_eq!(c.params.get(*name), Some(value), "GenParams.{name} differs from the Python default");
    }
    assert_eq!(c.params.as_object().map(|o| o.len()), Some(fields.len()), "the Python dataclass has other fields");
    Out::new().i32("n_fields", &[1], &[fields.len() as i32])
}

fn port(c: &Case) -> Option<Out> {
    Some(match c.function.as_str() {
        "cloudgen.generate.compose" => by_dtype!(c, "s", compose_case),
        "cloudgen.generate.solve_thresholds" => by_dtype!(c, "s", solve_case),
        "cloudgen.generate.row_sensitivity" => by_dtype!(c, "s", sensitivity_case),
        "cloudgen.generate.moisture_profile" => by_dtype!(c, "lat", moisture_case),
        "cloudgen.generate._smoothstep" => {
            if c.has_input("x") {
                by_dtype!(c, "x", smoothstep_case)
            } else {
                // Called with Python floats: float64 scalars.
                let xs = c.p_f64s("x");
                let y: Vec<f64> = xs.iter().map(|&x| smoothstep(c.p_f64("a"), c.p_f64("b"), x)).collect();
                Out::new().f64("y", &[y.len()], &y)
            }
        }
        "cloudgen.generate.area_mean" => by_dtype!(c, "a", area_case),
        "cloudgen.generate.along" => by_dtype!(c, "p", along_case),
        "cloudgen.generate.surface_regime_weights" => regime_case(c),
        "cloudgen.generate.GenParams" => genparams_case(c),
        _ => return None,
    })
}

#[test]
fn generate_stage_functions() {
    run(&Config { group: "products_generate", prefixes: &["cloudgen.generate."], expected_failures: EXPECTED_FAILURES }, port)
        .assert_clean();
}
