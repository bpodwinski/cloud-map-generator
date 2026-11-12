//! Golden cases of cloudgen/texture.py (group texture_surface, functions cloudgen.texture.*).
//!
//! The scalar helpers (`_smoothstep`, `_gauss`, `_standardize`) and `regime_weights` are plain numeric
//! comparisons, run in the precision of the case's arrays (float32 or float64, and the float64 latitude with a
//! float32 back-traced one that `generate()` really passes). `regime_weights` also returns its intermediates
//! (`convective`, `cold`, `total_raw`), which the port exposes through `regime_weight_terms`.
//!
//! `regime_texture` consumes numpy's draws, recorded as `plan<i>.*` inputs in call order: the generic fBm
//! (plans 0 and 1: the Perlin permutation, the octaves), the street fBm (2 and 3), then a (worley3, cell_edges)
//! pair for each USED cell diameter. The glue builds the port's plan structs from them. What the port derives
//! from its own inputs is checked against the recorded plan scalars instead of being papered over by the
//! recording: the octave ladder (start frequency and octave count), the seeds (`seed + 3`, `seed + 5`,
//! `seed + int(d_km)`), the cell diameters the resolution cutoff keeps and their frequencies
//! (`radius_km / d_km`), the octave amplitude schedule and the fBm normalization. The case's intermediates
//! (`generic_std`, `street_std`, `closed_sum`, `open_sum`, `closed_std`, `open_std`) come from
//! `regime_texture_terms`, the same implementation as `regime_texture`.

use cloud_map_generator_core::noise::{octave_amp_freq, octave_ladder, CellEdgesPlan, FbmOctave, FbmPlan, Perlin3, Worley3, DEFAULT_LACUNARITY, DEFAULT_TOP};
use cloud_map_generator_core::sphere::Vec3;
use cloud_map_generator_core::texture::*;
use cloud_map_generator_golden::{run, Array, Case, Config, DType, Out};
use num_traits::Float;

const EXPECTED_FAILURES: &[(&str, &str)] = &[];

// ---------------------------------------------------------------------------------------------------------
// Reading cases

trait Elem: Float + Send + Sync + std::fmt::Debug {
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

fn points<F: Elem>(a: &Array) -> Vec<Vec3<F>> {
    F::read(a).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Shape of an array without its trailing axis of 3.
fn lead(a: &Array) -> Vec<usize> {
    a.shape[..a.shape.len() - 1].to_vec()
}

fn mat3_f32(v: &[f32]) -> [[f32; 3]; 3] {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

fn float_dtype(c: &Case, a: &Array) -> DType {
    assert!(a.dtype.is_float(), "case {}: expected a float array, got {}", c.id, a.dtype.name());
    a.dtype
}

// ---------------------------------------------------------------------------------------------------------
// _smoothstep, _gauss, _standardize

fn elementwise<F: Elem>(c: &Case, input: &str, output: &str, f: impl Fn(F) -> F) -> Out {
    let x = c.input(input);
    let y: Vec<F> = F::read(x).into_iter().map(f).collect();
    F::put(Out::new(), output, &x.shape, &y)
}

fn smoothstep_case<F: Elem>(c: &Case) -> Out {
    let (a, b) = (c.p_f64("a"), c.p_f64("b"));
    elementwise::<F>(c, "x", "y", |x| smoothstep(a, b, x))
}

fn gauss_case<F: Elem>(c: &Case) -> Out {
    let (mu, s) = (c.p_f64("mu"), c.p_f64("s"));
    elementwise::<F>(c, "x", "y", |x| gauss(x, mu, s))
}

fn standardize_case<F: Elem>(c: &Case) -> Out {
    let v = c.input("v");
    F::put(Out::new(), "y", &v.shape, &standardize(&F::read(v)))
}

// ---------------------------------------------------------------------------------------------------------
// regime_weights

fn regime_weights_typed<L: Elem, B: Elem, P: Elem>(c: &Case) -> Out {
    let (lat, back) = (c.input("lat"), c.input("lat_back"));
    let (l, b, hadley) = (L::read(lat), B::read(back), c.p_f64("hadley_deg"));
    let t = regime_weight_terms::<L, B, P>(&l, &b, hadley);
    // the plain function is the same implementation, not a second formula
    assert_eq!(regime_weights::<L, B, P>(&l, &b, hadley), t.weights, "case {}", c.id);
    let shape = &lat.shape;
    let out = P::put(Out::new(), "closed", shape, &t.weights.closed);
    let out = P::put(out, "open", shape, &t.weights.open);
    let out = P::put(out, "street", shape, &t.weights.street);
    let out = L::put(out, "convective", shape, &t.convective);
    let out = P::put(out, "cold", shape, &t.cold);
    P::put(out, "total_raw", shape, &t.total_raw)
}

/// numpy's promotion of the two latitude dtypes is the result precision: float32 only for two float32 arrays.
fn regime_weights_case(c: &Case) -> Out {
    let (lat, back) = (float_dtype(c, c.input("lat")), float_dtype(c, c.input("lat_back")));
    match (lat, back) {
        (DType::F64, DType::F64) => regime_weights_typed::<f64, f64, f64>(c),
        (DType::F32, DType::F32) => regime_weights_typed::<f32, f32, f32>(c),
        (DType::F64, DType::F32) => regime_weights_typed::<f64, f32, f64>(c),
        (DType::F32, DType::F64) => regime_weights_typed::<f32, f64, f64>(c),
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------------------------------------
// regime_texture

/// The recorded plan scalars of plan `i` (`params.plans`, by its `index`).
fn plan_record(c: &Case, i: usize) -> &serde_json::Value {
    let rec = c.param("plans").as_array().expect("plans is a list").iter().find(|r| r["index"].as_u64() == Some(i as u64));
    rec.unwrap_or_else(|| panic!("case {}: no recorded plan {i}", c.id))
}

fn rec_f64(rec: &serde_json::Value, key: &str) -> f64 {
    rec[key].as_f64().unwrap_or_else(|| panic!("recorded plan has no number {key}: {rec}"))
}

/// The fBm plan recorded as plans `perm_idx` (the Perlin permutation) and `fbm_idx` (the octaves): the tables
/// are the recorded draws, everything the port derives (seed, ladder, amplitudes, norm) is checked against the
/// record.
fn fbm_plan_of(c: &Case, perm_idx: usize, fbm_idx: usize, seed: u64, frequency: f64, octaves: usize, gain: f64) -> FbmPlan {
    assert_eq!(plan_record(c, perm_idx)["kind"], "perlin3");
    let rec = plan_record(c, fbm_idx);
    assert_eq!(rec["kind"], "fbm");
    assert_eq!(rec["seed"].as_u64(), Some(seed), "case {}: the fbm seed", c.id);
    assert_eq!(rec_f64(rec, "frequency"), frequency, "case {}: the ladder's start frequency", c.id);
    assert_eq!(rec["octaves"].as_u64(), Some(octaves as u64), "case {}: the ladder's octave count", c.id);
    assert_eq!(rec_f64(rec, "gain"), gain);
    assert_eq!(rec_f64(rec, "lacunarity"), DEFAULT_LACUNARITY);
    assert_eq!(rec["billow"].as_bool(), Some(false));
    let key = |i: usize, name: &str| format!("plan{i}.{name}");
    let amp = c.input(&key(fbm_idx, "amp")).as_f64();
    let freq = c.input(&key(fbm_idx, "freq")).as_f64();
    let rot = c.input(&key(fbm_idx, "rot")).as_f32();
    let off = c.input(&key(fbm_idx, "off")).as_f32();
    assert_eq!(amp.len(), octaves);
    assert_eq!(
        octave_amp_freq(frequency, octaves, gain, DEFAULT_LACUNARITY),
        amp.iter().copied().zip(freq.iter().copied()).collect::<Vec<_>>(),
        "case {}: the amplitude/frequency schedule is not the recorded one",
        c.id
    );
    let octs = (0..octaves)
        .map(|k| FbmOctave { amp: amp[k], freq: freq[k], rot: mat3_f32(&rot[9 * k..9 * k + 9]), off: [off[3 * k], off[3 * k + 1], off[3 * k + 2]] })
        .collect();
    let plan = FbmPlan::new(Perlin3::new(&c.input(&key(perm_idx, "perm")).as_i32()), octs, false);
    let norm = rec_f64(rec, "norm");
    assert!((plan.norm - norm).abs() <= 1e-13 * norm.abs(), "case {}: norm {} vs recorded {norm}", c.id, plan.norm);
    plan
}

/// The cell-edge plan recorded as plans `worley_idx` (the Worley tables) and `edges_idx` (the rotation).
fn cell_plan_of(c: &Case, worley_idx: usize, edges_idx: usize, seed: u64, frequency: f64) -> CellEdgesPlan {
    assert_eq!(plan_record(c, worley_idx)["kind"], "worley3");
    let rec = plan_record(c, edges_idx);
    assert_eq!(rec["kind"], "cell_edges");
    assert_eq!(rec["seed"].as_u64(), Some(seed), "case {}: the cell seed", c.id);
    assert_eq!(rec_f64(rec, "frequency"), frequency, "case {}: cells per radian", c.id);
    let perm = c.input(&format!("plan{worley_idx}.perm")).as_i64();
    let jitter: Vec<Vec3<f32>> = c.input(&format!("plan{worley_idx}.jitter")).as_f32().chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
    CellEdgesPlan {
        worley: Worley3::new(&perm, &jitter),
        rot: mat3_f32(&c.input(&format!("plan{edges_idx}.rot")).as_f32()),
        frequency,
    }
}

/// The plans of a `regime_texture` case, with the port's own derivations checked against the record.
fn plans_of(c: &Case) -> RegimeTexturePlans {
    let (radius, fmax, seed) = (c.p_f64("radius_km"), c.p_f64("fmax"), c.p_i64("seed") as u64);
    // the constants of the case are the constants of the port
    assert_eq!(c.p_f64s("cell_diameters_km"), CELL_DIAMETERS_KM);
    assert_eq!(c.p_f64s("cell_weights"), CELL_WEIGHTS);
    assert_eq!(c.p_f64s("cells_per_radian"), CELL_DIAMETERS_KM.map(|d| radius / d), "case {}: f = radius_km / d_km", c.id);

    let (f_fine, n_fine) = octave_ladder(c.p_f64("fine_frequency"), fmax, DEFAULT_LACUNARITY, DEFAULT_TOP);
    let gain = c.p_f64("fine_gain");
    let generic = fbm_plan_of(c, 0, 1, seed + GENERIC_SEED_OFFSET, f_fine, n_fine, gain);
    let street = fbm_plan_of(c, 2, 3, seed + STREET_SEED_OFFSET, f_fine, n_fine, gain);

    let used = used_cells(radius, fmax);
    assert_eq!(used.iter().map(|u| u.diameter_km).collect::<Vec<_>>(), c.p_f64s("cells_used_km"), "case {}: the cutoff keeps other diameters", c.id);
    assert_eq!(c.param("plans").as_array().unwrap().len(), 4 + 2 * used.len(), "case {}: one (worley3, cell_edges) pair per used diameter", c.id);
    let cells = used
        .iter()
        .enumerate()
        .map(|(k, u)| cell_plan_of(c, 4 + 2 * k, 5 + 2 * k, cell_seed(seed, u.diameter_km), u.frequency))
        .collect();
    RegimeTexturePlans { generic, street, cells }
}

fn regime_texture_typed<F: Elem, W: Elem>(c: &Case) -> Out {
    let pts = c.input("p");
    let (p, back) = (points::<F>(pts), points::<F>(c.input("back")));
    let weights = RegimeWeights {
        closed: W::read(c.input("weights.closed")),
        open: W::read(c.input("weights.open")),
        street: W::read(c.input("weights.street")),
    };
    let (radius, fmax) = (c.p_f64("radius_km"), c.p_f64("fmax"));
    let plans = plans_of(c);
    let t = regime_texture_terms(&p, &back, &weights, radius, fmax, &plans);
    // the plain function is the same implementation, not a second formula
    assert_eq!(regime_texture(&p, &back, &weights, radius, fmax, &plans), t.texture, "case {}", c.id);
    let shape = lead(pts);
    let mut out = Out::new().f32("texture", &shape, &t.texture).f32("generic_std", &shape, &t.generic_std).f32("street_std", &shape, &t.street_std);
    if let Some(cf) = &t.cells {
        out = out
            .f32("closed_sum", &shape, &cf.closed_sum)
            .f32("open_sum", &shape, &cf.open_sum)
            .f32("closed_std", &shape, &cf.closed_std)
            .f32("open_std", &shape, &cf.open_std);
    }
    out
}

/// Directions and weights come in either precision: `p` and `back` float32 or float64, the weights float32 or
/// float64 (the blend runs in the weights' precision).
fn regime_texture_case(c: &Case) -> Out {
    let (dirs, weights) = (float_dtype(c, c.input("p")), float_dtype(c, c.input("weights.closed")));
    assert_eq!(c.input("back").dtype, dirs, "case {}: p and back share a dtype", c.id);
    for k in ["open", "street"] {
        assert_eq!(c.input(&format!("weights.{k}")).dtype, weights, "case {}: the three weights share a dtype", c.id);
    }
    match (dirs, weights) {
        (DType::F32, DType::F32) => regime_texture_typed::<f32, f32>(c),
        (DType::F32, DType::F64) => regime_texture_typed::<f32, f64>(c),
        (DType::F64, DType::F32) => regime_texture_typed::<f64, f32>(c),
        (DType::F64, DType::F64) => regime_texture_typed::<f64, f64>(c),
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------------------------------------

fn by_dtype<R>(c: &Case, input: &str, f32_case: impl Fn(&Case) -> R, f64_case: impl Fn(&Case) -> R) -> R {
    match float_dtype(c, c.input(input)) {
        DType::F32 => f32_case(c),
        DType::F64 => f64_case(c),
        _ => unreachable!(),
    }
}

fn port(c: &Case) -> Option<Out> {
    Some(match c.function.as_str() {
        "cloudgen.texture._smoothstep" => by_dtype(c, "x", smoothstep_case::<f32>, smoothstep_case::<f64>),
        "cloudgen.texture._gauss" => by_dtype(c, "x", gauss_case::<f32>, gauss_case::<f64>),
        "cloudgen.texture._standardize" => by_dtype(c, "v", standardize_case::<f32>, standardize_case::<f64>),
        "cloudgen.texture.regime_weights" => regime_weights_case(c),
        "cloudgen.texture.regime_texture" => regime_texture_case(c),
        _ => return None,
    })
}

#[test]
fn texture_cases() {
    let cfg = Config { group: "texture_surface", prefixes: &["cloudgen.texture."], expected_failures: EXPECTED_FAILURES };
    run(&cfg, port).assert_clean();
}
