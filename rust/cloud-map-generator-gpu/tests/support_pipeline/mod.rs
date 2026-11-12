//! The glue that turns a golden `cloudgen.generate.generate` case into the inputs of the pipeline: the body, the
//! `GenParams`, the flow, the surface and the plans of numpy's recorded draws. Copied from
//! `cloud-map-generator-core/tests/golden_pipeline.rs` (where the same glue feeds the CPU port) so that the GPU pipeline
//! is fed the very same inputs; every derived quantity (seeds, ladders, norms, cell frequencies) is checked
//! against the record there too.
#![allow(dead_code)]

use cloud_map_generator_core::climate::BodyInputs;
use cloud_map_generator_core::flow::{Flow, Vortex, VortexKind};
use cloud_map_generator_core::generate::*;
use cloud_map_generator_core::noise::{
    octave_amp_freq, octave_ladder, CellEdgesPlan, FbmOctave, FbmPlan, Perlin3, Worley3, DEFAULT_GAIN, DEFAULT_LACUNARITY, DEFAULT_TOP,
};
use cloud_map_generator_core::sphere::Vec3;
use cloud_map_generator_core::surface::Surface;
use cloud_map_generator_core::texture::{cell_seed, used_cells, RegimeTexturePlans, CELL_DIAMETERS_KM, CELL_WEIGHTS, STREET_SEED_OFFSET};
use cloud_map_generator_golden::Case;
use serde_json::Value;

pub fn num(v: &Value, what: &str) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("{what} is not a number: {v}"))
}

pub fn body_of(v: &Value) -> BodyInputs {
    let period = if v["rotation_period_days"].is_null() { None } else { Some(num(&v["rotation_period_days"], "rotation_period_days")) };
    BodyInputs::new(period, num(&v["radius_km"], "radius_km"), num(&v["gravity_mps2"], "gravity_mps2"), num(&v["scale_height_km"], "scale_height_km"))
        .with_regime(v["cloud_regime"].as_str().expect("cloud_regime is a string").to_string())
}

/// The GenParams of a case's `gen_params` (`asdict`): every field, none unknown.
pub fn gen_params_of(v: &Value) -> GenParams {
    let mut p = GenParams::default();
    let obj = v.as_object().expect("gen_params is an object");
    assert_eq!(obj.len(), 25, "the Python GenParams has another number of fields");
    for (name, x) in obj {
        let f = || num(x, name);
        let b = || x.as_bool().unwrap_or_else(|| panic!("{name} is not a bool"));
        match name.as_str() {
            "structure_width" => p.structure_width = f() as usize,
            "a_moisture" => p.a_moisture = f(),
            "a_ascent" => p.a_ascent = f(),
            "a_mid" => p.a_mid = f(),
            "a_fine" => p.a_fine = f(),
            "moisture_noise" => p.moisture_noise = f(),
            "advect_mid" => p.advect_mid = f(),
            "advect_fine" => p.advect_fine = f(),
            "mid_frequency" => p.mid_frequency = f(),
            "fine_frequency" => p.fine_frequency = f(),
            "cumulus_cell_km" => p.cumulus_cell_km = f(),
            "fine_top" => p.fine_top = f(),
            "fine_gain" => p.fine_gain = f(),
            "fine_billow" => p.fine_billow = b(),
            "fine_min_texels" => p.fine_min_texels = f(),
            "shield_ramp" => p.shield_ramp = f(),
            "cumulus_depth" => p.cumulus_depth = f(),
            "cumulus_max" => p.cumulus_max = f(),
            "shield_texture" => p.shield_texture = f(),
            "threshold_smooth_deg" => p.threshold_smooth_deg = f(),
            "regime_texture" => p.regime_texture = b(),
            "isotropic" => p.isotropic = b(),
            "allow_any_regime" => p.allow_any_regime = b(),
            "surface_terms" => p.surface_terms = b(),
            "sensitivity_step" => p.sensitivity_step = f(),
            other => panic!("unknown GenParams field {other}"),
        }
    }
    p
}

/// The flow the case ran with: its scalars and the vortex arrays (kind codes of `vortex_kind_codes`).
pub fn flow_of(c: &Case) -> Flow {
    let f = c.param("flow");
    let codes = c.param("vortex_kind_codes");
    let kinds = c.input("vortex.kind").as_i32();
    let col = |name: &str| c.input(&format!("vortex.{name}")).as_f64();
    let (lon, lat, radius, omega, ascent, age) = (col("lon"), col("lat"), col("radius"), col("omega"), col("ascent"), col("age"));
    let kind_of = |code: i32| {
        [("cyclone", VortexKind::Cyclone), ("anticyclone", VortexKind::Anticyclone), ("tropical", VortexKind::Tropical)]
            .into_iter()
            .find(|(name, _)| codes[*name].as_i64() == Some(i64::from(code)))
            .unwrap_or_else(|| panic!("case {}: vortex kind code {code}", c.id))
            .1
    };
    let vortices = (0..kinds.len())
        .map(|i| Vortex { kind: kind_of(kinds[i]), lon: lon[i], lat: lat[i], radius: radius[i], omega: omega[i], ascent: ascent[i], age: age[i] })
        .collect();
    Flow {
        hadley_deg: num(&f["hadley_deg"], "flow.hadley_deg"),
        vortices,
        jet: num(&f["jet"], "flow.jet"),
        trades: num(&f["trades"], "flow.trades"),
        polar_easterlies: num(&f["polar_easterlies"], "flow.polar_easterlies"),
    }
}

pub fn surface_of(c: &Case) -> Option<Surface> {
    c.param("has_surface").as_bool().unwrap().then(|| {
        let s = c.param("surface");
        let (h, w) = (s["height"].as_u64().unwrap() as usize, s["width"].as_u64().unwrap() as usize);
        Surface::new(c.input("surface.land").as_f32(), c.input("surface.elevation_km").as_f32(), h, w, s["source"].as_str().unwrap())
    })
}

// ---------------------------------------------------------------------------------------------------------
// The recorded plans, read in call order

fn mat3_f32(v: &[f32]) -> [[f32; 3]; 3] {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

/// Two doubles equal to a few ulps. The plan scalars come out of `case.json` through serde_json, which parses
/// a decimal like `9.676620539987237` without correct rounding (one ulp off here); the arrays in `data.bin`
/// are exact, and they are what the schedule check below compares bit for bit.
fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-14 * b.abs()
}

fn rec_f64(rec: &Value, key: &str) -> f64 {
    rec[key].as_f64().unwrap_or_else(|| panic!("recorded plan has no number {key}: {rec}"))
}

/// A cursor over the case's recorded draws. Every plan the port needs is read in the order Python made them,
/// and checked against what the port itself derives (seed, frequency, octave count, gain, schedule, norm).
struct Plans<'a> {
    c: &'a Case,
    at: usize,
}

impl Plans<'_> {
    fn record(&self, i: usize) -> &Value {
        let rec = self.c.param("plans").as_array().expect("plans is a list").iter().find(|r| r["index"].as_u64() == Some(i as u64));
        rec.unwrap_or_else(|| panic!("case {}: no recorded plan {i}", self.c.id))
    }

    /// The next fBm plan: its Perlin permutation (plan `at`) and its octaves (plan `at + 1`).
    fn fbm(&mut self, seed: u64, frequency: f64, octaves: usize, gain: f64, billow: bool, what: &str) -> FbmPlan {
        let (perm_idx, fbm_idx) = (self.at, self.at + 1);
        self.at += 2;
        let c = self.c;
        assert_eq!(self.record(perm_idx)["kind"], "perlin3", "case {}: {what}: plan {perm_idx} is not a Perlin permutation", c.id);
        let rec = self.record(fbm_idx);
        assert_eq!(rec["kind"], "fbm", "case {}: {what}: plan {fbm_idx} is not an fBm", c.id);
        assert_eq!(rec["seed"].as_u64(), Some(seed), "case {}: {what}: the fbm seed", c.id);
        assert!(same(rec_f64(rec, "frequency"), frequency), "case {}: {what}: the start frequency {} vs recorded {}", c.id, frequency, rec["frequency"]);
        assert_eq!(rec["octaves"].as_u64(), Some(octaves as u64), "case {}: {what}: the octave count", c.id);
        assert!(same(rec_f64(rec, "gain"), gain), "case {}: {what}: the gain", c.id);
        assert!(same(rec_f64(rec, "lacunarity"), DEFAULT_LACUNARITY), "case {}: {what}: the lacunarity", c.id);
        assert_eq!(rec["billow"].as_bool(), Some(billow), "case {}: {what}: the billow flag", c.id);
        let key = |i: usize, name: &str| format!("plan{i}.{name}");
        let amp = c.input(&key(fbm_idx, "amp")).as_f64();
        let freq = c.input(&key(fbm_idx, "freq")).as_f64();
        let rot = c.input(&key(fbm_idx, "rot")).as_f32();
        let off = c.input(&key(fbm_idx, "off")).as_f32();
        assert_eq!(
            octave_amp_freq(frequency, octaves, gain, DEFAULT_LACUNARITY),
            amp.iter().copied().zip(freq.iter().copied()).collect::<Vec<_>>(),
            "case {}: {what}: the amplitude/frequency schedule is not the recorded one",
            c.id
        );
        let octs = (0..octaves)
            .map(|k| FbmOctave { amp: amp[k], freq: freq[k], rot: mat3_f32(&rot[9 * k..9 * k + 9]), off: [off[3 * k], off[3 * k + 1], off[3 * k + 2]] })
            .collect();
        let plan = FbmPlan::new(Perlin3::new(&c.input(&key(perm_idx, "perm")).as_i32()), octs, billow);
        let norm = rec_f64(rec, "norm");
        assert!((plan.norm - norm).abs() <= 1e-13 * norm.abs(), "case {}: {what}: norm {} vs recorded {norm}", c.id, plan.norm);
        plan
    }

    /// The next cell-edge plan: its Worley tables (plan `at`) and its rotation (plan `at + 1`).
    fn cells(&mut self, seed: u64, frequency: f64) -> CellEdgesPlan {
        let (worley_idx, edges_idx) = (self.at, self.at + 1);
        self.at += 2;
        let c = self.c;
        assert_eq!(self.record(worley_idx)["kind"], "worley3", "case {}: plan {worley_idx} is not a Worley table", c.id);
        let rec = self.record(edges_idx);
        assert_eq!(rec["kind"], "cell_edges", "case {}: plan {edges_idx} is not a cell-edge plan", c.id);
        assert_eq!(rec["seed"].as_u64(), Some(seed), "case {}: the cell seed", c.id);
        assert!(same(rec_f64(rec, "frequency"), frequency), "case {}: cells per radian", c.id);
        let perm = c.input(&format!("plan{worley_idx}.perm")).as_i64();
        let jitter: Vec<Vec3<f32>> =
            c.input(&format!("plan{worley_idx}.jitter")).as_f32().chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
        CellEdgesPlan { worley: Worley3::new(&perm, &jitter), rot: mat3_f32(&c.input(&format!("plan{edges_idx}.rot")).as_f32()), frequency }
    }
}

/// [`GeneratePlans`] from the recorded draws of a case, every derived quantity checked against the record.
pub fn plans_of(c: &Case, body: &BodyInputs, seed: u64, width: usize, prm: &GenParams) -> GeneratePlans {
    let height = width / 2;
    let ladder = fine_ladder(width, prm);
    let mut r = Plans { c, at: 0 };
    let moisture = r.fbm(seed + SEED_MOISTURE, MOISTURE_FREQUENCY, MOISTURE_OCTAVES, DEFAULT_GAIN, false, "moisture");
    let mid = r.fbm(seed + SEED_MID, prm.mid_frequency, MID_OCTAVES, DEFAULT_GAIN, false, "mid");
    let texture = if prm.regime_texture {
        // regime_texture anchors its own ladder (default top 0.95) at fmax * fine_top / 0.95.
        let fmax = texture_fmax(ladder.fmax, prm);
        let (f_fine, n_fine) = octave_ladder(prm.fine_frequency, fmax, DEFAULT_LACUNARITY, DEFAULT_TOP);
        let generic = r.fbm(seed + SEED_FINE, f_fine, n_fine, prm.fine_gain, false, "generic fine texture");
        let street = r.fbm(seed + STREET_SEED_OFFSET, f_fine, n_fine, prm.fine_gain, false, "street texture");
        let used = used_cells(body.radius_km, fmax);
        assert_eq!(CELL_DIAMETERS_KM.len(), CELL_WEIGHTS.len());
        let cells: Vec<CellEdgesPlan> = used.iter().map(|u| r.cells(cell_seed(seed, u.diameter_km), u.frequency)).collect();
        TexturePlans::Regime(RegimeTexturePlans { generic, street, cells })
    } else {
        TexturePlans::Plain(r.fbm(seed + SEED_FINE, ladder.frequency, ladder.octaves, prm.fine_gain, prm.fine_billow, "plain fine texture"))
    };
    let streak = r.fbm(seed + SEED_STREAK, ladder.frequency, ladder.octaves, STREAK_GAIN, false, "streak");
    assert_eq!(r.at, c.param("plans").as_array().unwrap().len(), "case {}: the pipeline consumed another number of plans", c.id);

    let scale = subtexel_scale(body, width, prm);
    assert_eq!(scale.is_some(), c.has_input("subtexel_white"), "case {}: the sub-texel term is on iff the white noise is recorded", c.id);
    let subtexel = scale.map(|s| {
        let white = c.input("subtexel_white");
        assert_eq!(white.shape, [height, width], "case {}: shape of the sub-texel white noise", c.id);
        subtexel_from_white(&white.as_f32(), s)
    });
    GeneratePlans { moisture, mid, texture, streak, subtexel, flow: flow_of(c) }
}
