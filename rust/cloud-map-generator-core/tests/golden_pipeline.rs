//! Golden cases of the pipeline group that run the generator itself: `cloudgen.generate.generate` (27
//! scenarios traced stage by stage, plus the refusal of a non-water body), `bake.library_body`,
//! `cloudgen.metrics.measure` (the Blue Marble reference, and the per-seed pins of Python's own maps) and the
//! two statistical cases `pipeline.stat.*`. (`golden_export.rs` runs the other functions of the group.)
//!
//! # generate
//!
//! A case hands the port numpy's recorded draws as `plan<i>.*` arrays (in call order: moisture, mid, the fine
//! texture's plans, streak), the vortex list of the flow it ran with, the sub-texel white noise and the
//! surface. The glue builds a [`GeneratePlans`] from them and calls [`generate_with_plans`] with a trace,
//! then maps every traced field to the case's output of the same name. What the port derives from its own
//! inputs is CHECKED against the record instead of being papered over by it: the seeds (`seed + 1` ...), the
//! octave ladders, the cell diameters the map resolves and their frequencies, the amplitude schedules, the
//! fBm normalizations. A pipeline that derived another ladder would fail here before any field is compared.
//!
//! # The statistical cases (`GOLDEN_SLOW=1`)
//!
//! The per-seed maps of Python are numpy's draws; a port with its own generator cannot reproduce them, so
//! `stat.earth.seed<k>.w1024` are listed in `EXPECTED_FAILURES` (their own doc says the value "is NOT a target
//! for a port"). The cases that say how a port is judged, `pipeline.stat.metrics_over_seeds` and
//! `pipeline.stat.zonal_residual`, are run with maps the port draws itself (Earth, 1024 x 512, seeds 1 to 8,
//! the `Rng` plans): the case's expected arrays are Python's DISTRIBUTION (mean and standard deviation over
//! seeds), so the glue judges the port's own statistics against it and panics on a miss (the runner then
//! reports a failure). The judgement is [`Verdict`]; its section below states what is asserted, with the
//! probability that a faithful model fails each test, and what it cannot see. What goes through the runner's
//! own comparison is the closed forms (the band centers, the target, the band means of the target) of the
//! port and the Python distribution RECOMPUTED from Python's per-seed pins: the latter checks the reference,
//! not the port, and says so where it is built. The two arrays no pin holds (the row residual of Python's
//! seed 1 and its maximum) are handed back as recorded.
//!
//! # The production plans
//!
//! The golden cases feed the pipeline numpy's recorded draws, so `GeneratePlans::from_seed` (what a bake
//! runs) is not exercised by them. [`assert_from_seed_wiring`] compares what it derives (octave schedules,
//! norms, cell frequencies, the sub-texel term, the flow's scalars) with the record of every generate case.

use cloud_map_generator_core::bake::{library_body, LIBRARY_CLASSES, LIBRARY_SCALE_HEIGHT_KM, LIST_MAX_WIDTH};
use cloud_map_generator_core::climate::{hadley_edge_deg, target_zonal_coverage, BodyInputs, EARTH};
use cloud_map_generator_core::export::SLICE_WIDTH;
use cloud_map_generator_core::flow::{Flow, Vortex, VortexKind};
use cloud_map_generator_core::generate::*;
use cloud_map_generator_core::metrics::{self, Measure};
use cloud_map_generator_core::noise::{
    octave_amp_freq, octave_ladder, CellEdgesPlan, FbmOctave, FbmPlan, Perlin3, Worley3, DEFAULT_GAIN, DEFAULT_LACUNARITY, DEFAULT_TOP,
};
use cloud_map_generator_core::sphere::{grid_lonlat, Vec3};
use cloud_map_generator_core::surface::Surface;
use cloud_map_generator_core::texture::{cell_seed, used_cells, RegimeTexturePlans, CELL_DIAMETERS_KM, CELL_WEIGHTS, STREET_SEED_OFFSET};
use cloud_map_generator_golden::{golden_root, load_case, run, Case, Config, Out};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The 8 per-seed maps of Python (numpy's draws) cannot be matched by a port with its own generator; the case
/// docs say so ("NOT a target for a port"), and `pipeline.stat.metrics_over_seeds` is the judge.
const PER_SEED_REASON: &str = "rng: the value is numpy's bit stream for this seed (the case doc: 'NOT a target for a port'); \
                               the port is judged by pipeline.stat.metrics_over_seeds, which this file runs";

const EXPECTED_FAILURES: &[(&str, &str)] = &[
    ("stat.earth.seed1.w1024", PER_SEED_REASON),
    ("stat.earth.seed2.w1024", PER_SEED_REASON),
    ("stat.earth.seed3.w1024", PER_SEED_REASON),
    ("stat.earth.seed4.w1024", PER_SEED_REASON),
    ("stat.earth.seed5.w1024", PER_SEED_REASON),
    ("stat.earth.seed6.w1024", PER_SEED_REASON),
    ("stat.earth.seed7.w1024", PER_SEED_REASON),
    ("stat.earth.seed8.w1024", PER_SEED_REASON),
];

// ---------------------------------------------------------------------------------------------------------
// Reading a case

fn num(v: &Value, what: &str) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("{what} is not a number: {v}"))
}

fn body_of(v: &Value) -> BodyInputs {
    let period = if v["rotation_period_days"].is_null() { None } else { Some(num(&v["rotation_period_days"], "rotation_period_days")) };
    BodyInputs::new(period, num(&v["radius_km"], "radius_km"), num(&v["gravity_mps2"], "gravity_mps2"), num(&v["scale_height_km"], "scale_height_km"))
        .with_regime(v["cloud_regime"].as_str().expect("cloud_regime is a string").to_string())
}

/// The GenParams of a case's `gen_params` (`asdict`): every field, none unknown.
fn gen_params_of(v: &Value) -> GenParams {
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
fn flow_of(c: &Case) -> Flow {
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

fn surface_of(c: &Case) -> Option<Surface> {
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
fn plans_of(c: &Case, body: &BodyInputs, seed: u64, width: usize, prm: &GenParams) -> GeneratePlans {
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

/// What the port's own draws ([`GeneratePlans::from_seed`], the production path: the golden cases feed the
/// pipeline numpy's recorded draws instead) are made OF, against the record of Python's. The draws themselves
/// (lattice tables, rotations, offsets, white noise, the vortex jitter) are another generator's, but each plan
/// has a seed, a start frequency, an octave count, a gain, a billow flag and a norm that Python records and the
/// port derives: every fBm's octave schedule and norm, every cell plan's frequency, the sub-texel term's
/// presence and size, the flow's scalars. A `from_seed` that wires a gain, a frequency, an anchor or a seed
/// offset differently from `generate()` fails here, on every one of the cases (the statistical cases cannot
/// see it: a 1024 map has a single fine octave, which has no gain and no start frequency of its own).
fn assert_from_seed_wiring(c: &Case, drawn: &GeneratePlans, rec: &GeneratePlans) {
    let schedule = |p: &FbmPlan| -> Vec<(f64, f64)> { p.octaves.iter().map(|o| (o.amp, o.freq)).collect() };
    let same_fbm = |what: &str, a: &FbmPlan, b: &FbmPlan| {
        assert_eq!(schedule(a), schedule(b), "case {}: from_seed's {what}: the octave schedule is not Python's", c.id);
        assert_eq!((a.billow, a.norm), (b.billow, b.norm), "case {}: from_seed's {what}: billow flag or norm", c.id);
    };
    same_fbm("moisture plan", &drawn.moisture, &rec.moisture);
    same_fbm("mid plan", &drawn.mid, &rec.mid);
    match (&drawn.texture, &rec.texture) {
        (TexturePlans::Regime(a), TexturePlans::Regime(b)) => {
            same_fbm("generic fine plan", &a.generic, &b.generic);
            same_fbm("street fine plan", &a.street, &b.street);
            let cells = |t: &RegimeTexturePlans| t.cells.iter().map(|p| p.frequency).collect::<Vec<_>>();
            assert_eq!(cells(a), cells(b), "case {}: from_seed's cell frequencies", c.id);
        }
        (TexturePlans::Plain(a), TexturePlans::Plain(b)) => same_fbm("plain fine plan", a, b),
        (a, b) => panic!("case {}: from_seed made {a:?} where Python's record has {b:?}", c.id),
    }
    same_fbm("streak plan", &drawn.streak, &rec.streak);
    assert_eq!(drawn.subtexel.as_ref().map(Vec::len), rec.subtexel.as_ref().map(Vec::len), "case {}: from_seed's sub-texel plan", c.id);
    // The record's scalars went through serde_json (see `same`).
    let (d, r) = (&drawn.flow, &rec.flow);
    for (what, a, b) in [
        ("hadley_deg", d.hadley_deg, r.hadley_deg),
        ("jet", d.jet, r.jet),
        ("trades", d.trades, r.trades),
        ("polar_easterlies", d.polar_easterlies, r.polar_easterlies),
    ] {
        assert!(same(a, b), "case {}: from_seed's flow.{what} {a} vs recorded {b}", c.id);
    }
}

// ---------------------------------------------------------------------------------------------------------
// generate

fn f64_out(out: Out, name: &str, data: &[f64]) -> Out {
    out.f64(name, &[data.len()], data)
}

fn flat3(v: &[Vec3<f32>]) -> Vec<f32> {
    v.iter().flatten().copied().collect()
}

fn extremum(a: &[f32], pick_min: bool) -> f64 {
    a.iter().copied().map(f64::from).fold(if pick_min { f64::INFINITY } else { f64::NEG_INFINITY }, |m, v| if pick_min { m.min(v) } else { m.max(v) })
}

fn generate_case(c: &Case) -> Out {
    let body = body_of(c.param("body"));
    let (seed, width) = (c.p_i64("seed") as u64, c.p_usize("width"));
    let height = c.p_usize("height");
    assert_eq!(height, width / 2);
    let prm = gen_params_of(c.param("gen_params"));
    let surface = surface_of(c);
    let plans = plans_of(c, &body, seed, width, &prm);
    assert_from_seed_wiring(c, &GeneratePlans::from_seed(&body, seed, width, &prm), &plans);
    let result = generate_with_plans(&body, &plans, width, &prm, surface.as_ref(), c.p_bool("extras"), true).expect("the case's body is bakeable");
    let t = result.trace.as_ref().expect("a trace was asked for");
    let (hw, hw3) = (&[height, width][..], &[height, width, 3][..]);

    // Production call: no trace, the same maps (the golden docs: tracing changes nothing).
    let untraced = generate_with_plans(&body, &plans, width, &prm, surface.as_ref(), c.p_bool("extras"), false).unwrap();
    assert_eq!(untraced.coverage, result.coverage, "case {}: tracing changed the coverage", c.id);
    assert_eq!(untraced.extras, result.extras, "case {}: tracing changed the extras", c.id);
    assert!(untraced.trace.is_none());

    let mut out = Out::new().f32("coverage", hw, &result.coverage).f32("back", hw3, &flat3(&t.back));
    if let Some(bc) = &t.back_coarse {
        out = out.f32("back_coarse", &[bc.height, bc.width, 3], &flat3(&bc.data));
    }
    out = out
        .f32("lat_back", hw, &t.lat_back)
        .f32("moisture_noise", hw, &t.moisture_noise)
        .f32("ascent", hw, &t.ascent)
        .f32("mid_noise", hw, &t.mid_noise)
        .f32("s_raw", hw, &t.s_raw)
        .f32("s", hw, &t.s)
        .f32("fine", hw, &t.fine)
        .f32("streak", hw, &t.streak);
    if let Some(w) = &t.weights {
        let w = w.to_f32();
        out = out.f32("weights.closed", hw, &w.closed).f32("weights.open", hw, &w.open).f32("weights.street", hw, &w.street);
    }
    if let Some(sub) = &t.subtexel {
        out = out.f32("subtexel", hw, sub);
    }
    out = f64_out(out, "target_zonal", &t.target_zonal);
    out = f64_out(out, "target", &t.target);
    if let Some(v) = &t.target_local {
        out = out.f64("target_local", hw, v);
    }
    if let Some(v) = &t.thr0 {
        out = f64_out(out, "thr0", v);
    }
    if let Some(v) = &t.sensitivity {
        out = f64_out(out, "sensitivity", v);
    }
    if let Some(v) = &t.offset {
        out = out.f32("offset", hw, v);
    }
    out = f64_out(out, "thr_raw", &t.thr_raw);
    out = f64_out(out, "thr", &t.thr);
    if let Some(v) = &t.shield {
        // The case freezes the shield as float32 (float64 in the trace).
        out = out.f32("shield", hw, &v.iter().map(|&x| x as f32).collect::<Vec<_>>());
    }
    if let Some(v) = &t.kind {
        out = out.f32("type", hw, v);
    }
    if let Some(v) = &t.high_share {
        out = out.f32("high_share", hw, v);
    }
    if let (Some(v), Some(e)) = (&t.wind, &result.extras) {
        out = out.f32("wind", &[e.wind_height, e.wind_width, 2], v);
    }
    if let Some(terms) = &t.terms {
        out = out
            .f32("terms.anomaly", hw, &terms.anomaly)
            .f32("terms.land", hw, &terms.land)
            .f32("terms.deck", hw, &terms.deck)
            .f32("terms.orographic", hw, &terms.orographic)
            .f32("terms.continentality", hw, &terms.continentality)
            .f32("terms.deck_strength", hw, &terms.deck_strength);
        out = out.f64("terms.itcz_shift_deg", &[1], &[terms.itcz_shift_deg]);
        out = out.f64("meta.anomaly_range", &[2], &[extremum(&terms.anomaly, true), extremum(&terms.anomaly, false)]);
    }

    // meta
    let m = &result.meta;
    assert_eq!((m.width, m.height, m.orientation), (width, height, ORIENTATION));
    assert_eq!(c.param("meta")["orientation"].as_str(), Some(m.orientation), "case {}: the sidecar's orientation text", c.id);
    out = out.f64("meta.hadley_edge_deg", &[1], &[m.hadley_edge_deg]).i32("meta.fine_octaves", &[1], &[m.fine_octaves as i32]);
    out = f64_out(out, "meta.flow_scalars", &[m.flow.jet, m.flow.trades, m.flow.polar_easterlies]);
    if !m.flow.vortices.is_empty() {
        out = f64_out(out, "meta.vortex_lon_deg", &m.flow.vortices.iter().map(|v| v.lon_deg).collect::<Vec<_>>());
        out = f64_out(out, "meta.vortex_lat_deg", &m.flow.vortices.iter().map(|v| v.lat_deg).collect::<Vec<_>>());
    }
    match (&m.surface, &surface) {
        (Some(sm), Some(surf)) => {
            // The unrounded land fraction is the port's own area_mean; the sidecar's values are Python-rounded.
            out = out.f64("meta.land_fraction", &[1], &[area_mean(&surf.land, surf.h, surf.w)]);
            out = out.f64("meta.surface.land_fraction", &[1], &[sm.land_fraction]);
            out = out.f64("meta.surface.itcz_shift_deg", &[1], &[sm.itcz_shift_deg]);
            out = out.f64("meta.surface.anomaly_range", &[2], &sm.anomaly_range);
            assert_eq!(c.param("meta")["surface_source"].as_str(), Some(sm.source.as_str()));
        }
        (None, _) => assert!(t.terms.is_none(), "case {}: surface terms ran but the sidecar has no surface", c.id),
        (Some(_), None) => unreachable!("a surface sidecar without a surface"),
    }
    out
}

/// `generate(body, ...)` refuses a body whose clouds are not water, through every entry point.
fn rejects_non_water_case(c: &Case) -> Out {
    let body = body_of(c.param("body"));
    let (seed, width) = (c.p_i64("seed") as u64, c.p_usize("width"));
    let prm = gen_params_of(c.param("gen_params"));
    assert!(!prm.allow_any_regime);
    let direct = generate(&body, seed, width, &prm, None, false);
    let plans = GeneratePlans::from_seed(&body, seed, width, &prm);
    let with_plans = generate_with_plans(&body, &plans, width, &prm, None, true, true);
    let regime = body.cloud_regime.to_string();
    let refused = |r: &Result<GenOutput, GenerateError>| matches!(r, Err(GenerateError::NotBakeable { cloud_regime }) if *cloud_regime == regime);
    // And the same body is accepted when the experiment flag is on.
    let allowed = generate(&body, seed, width, &GenParams { allow_any_regime: true, ..prm }, None, false).is_ok();
    assert!(allowed, "allow_any_regime must let a non-water body through");
    Out::new().u8("raises_value_error", &[1], &[u8::from(refused(&direct) && refused(&with_plans))])
}

// ---------------------------------------------------------------------------------------------------------
// bake.library_body

fn library_body_case(c: &Case) -> Out {
    // The module constants of bake.py are the port's.
    let classes: Vec<Vec<f64>> = c.param("classes").as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|v| num(v, "class")).collect()).collect();
    assert_eq!(classes, LIBRARY_CLASSES.iter().map(|k| vec![k.0, k.1, k.2]).collect::<Vec<_>>(), "bake.LIBRARY_CLASSES");
    assert_eq!(c.p_f64("library_scale_height_km"), LIBRARY_SCALE_HEIGHT_KM);
    assert_eq!(c.p_usize("slice_width"), SLICE_WIDTH);
    assert_eq!(c.p_usize("list_max_width"), LIST_MAX_WIDTH);
    let centers = c.input("centers").as_f64();
    let bodies: Vec<BodyInputs> = centers.iter().map(|&x| library_body(x)).collect();
    let flat: Vec<f64> =
        bodies.iter().flat_map(|b| [b.rotation_period_days.expect("a library body has a period"), b.radius_km, b.gravity_mps2, b.scale_height_km]).collect();
    let edge: Vec<f64> = bodies.iter().map(hadley_edge_deg).collect();
    Out::new().f64("body", &[centers.len(), 4], &flat).f64("hadley_edge_deg", &[centers.len()], &edge)
}

// ---------------------------------------------------------------------------------------------------------
// The Blue Marble reference: a minimal PNG reader (8-bit grayscale, no interlacing) with its own inflate,
// so the test needs no dependency.

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    bit: u32,
}

struct Huffman {
    count: [u16; 16],
    symbol: Vec<u16>,
}

impl Bits<'_> {
    fn one(&mut self) -> u32 {
        let b = (self.data[self.pos] >> self.bit) & 1;
        self.bit += 1;
        if self.bit == 8 {
            self.bit = 0;
            self.pos += 1;
        }
        u32::from(b)
    }
    fn many(&mut self, n: u32) -> u32 {
        (0..n).map(|i| self.one() << i).sum()
    }
    fn decode(&mut self, h: &Huffman) -> usize {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= self.one() as i32;
            let count = i32::from(h.count[len]);
            if code - count < first {
                return usize::from(h.symbol[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        panic!("inflate: invalid Huffman code");
    }
}

fn huffman(lengths: &[u8]) -> Huffman {
    let mut count = [0u16; 16];
    for &l in lengths {
        count[usize::from(l)] += 1;
    }
    let mut offs = [0u16; 16];
    for len in 1..15 {
        offs[len + 1] = offs[len] + count[len];
    }
    let mut symbol = vec![0u16; lengths.len()];
    for (sym, &l) in lengths.iter().enumerate() {
        if l != 0 {
            symbol[usize::from(offs[usize::from(l)])] = sym as u16;
            offs[usize::from(l)] += 1;
        }
    }
    Huffman { count, symbol }
}

/// RFC 1951 inflate (the algorithm of zlib's `puff.c`).
fn inflate(data: &[u8]) -> Vec<u8> {
    const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
    const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
    const DIST_BASE: [u16; 30] =
        [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
    const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
    let mut bits = Bits { data, pos: 0, bit: 0 };
    let mut out: Vec<u8> = Vec::new();
    loop {
        let last = bits.one();
        match bits.many(2) {
            0 => {
                if bits.bit != 0 {
                    bits.bit = 0;
                    bits.pos += 1;
                }
                let n = usize::from(data[bits.pos]) | usize::from(data[bits.pos + 1]) << 8;
                out.extend_from_slice(&data[bits.pos + 4..bits.pos + 4 + n]);
                bits.pos += 4 + n;
            }
            kind @ (1 | 2) => {
                let (lit, dist) = if kind == 1 {
                    let mut l = [8u8; 288];
                    l[144..256].fill(9);
                    l[256..280].fill(7);
                    (huffman(&l), huffman(&[5u8; 30]))
                } else {
                    let (nlen, ndist, ncode) = (bits.many(5) as usize + 257, bits.many(5) as usize + 1, bits.many(4) as usize + 4);
                    const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                    let mut cl = [0u8; 19];
                    for &o in ORDER.iter().take(ncode) {
                        cl[o] = bits.many(3) as u8;
                    }
                    let clh = huffman(&cl);
                    let mut lens = Vec::with_capacity(nlen + ndist);
                    while lens.len() < nlen + ndist {
                        let sym = bits.decode(&clh);
                        if sym < 16 {
                            lens.push(sym as u8);
                        } else {
                            let (value, repeat) = match sym {
                                16 => (*lens.last().expect("repeat with no previous length"), 3 + bits.many(2)),
                                17 => (0, 3 + bits.many(3)),
                                _ => (0, 11 + bits.many(7)),
                            };
                            lens.extend(std::iter::repeat_n(value, repeat as usize));
                        }
                    }
                    (huffman(&lens[..nlen]), huffman(&lens[nlen..nlen + ndist]))
                };
                loop {
                    let sym = bits.decode(&lit);
                    if sym < 256 {
                        out.push(sym as u8);
                    } else if sym == 256 {
                        break;
                    } else {
                        let s = sym - 257;
                        let len = usize::from(LEN_BASE[s]) + bits.many(u32::from(LEN_EXTRA[s])) as usize;
                        let d = bits.decode(&dist);
                        let back = usize::from(DIST_BASE[d]) + bits.many(u32::from(DIST_EXTRA[d])) as usize;
                        for _ in 0..len {
                            out.push(out[out.len() - back]);
                        }
                    }
                }
            }
            _ => panic!("inflate: invalid block type"),
        }
        if last == 1 {
            return out;
        }
    }
}

/// An 8-bit grayscale non-interlaced PNG: `(height, width, bytes)`.
fn read_gray_png(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "not a PNG");
    let (mut at, mut idat, mut header) = (8, Vec::new(), None);
    while at + 8 <= bytes.len() {
        let n = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let kind = &bytes[at + 4..at + 8];
        let body = &bytes[at + 8..at + 8 + n];
        match kind {
            b"IHDR" => header = Some((u32::from_be_bytes(body[0..4].try_into().unwrap()) as usize, u32::from_be_bytes(body[4..8].try_into().unwrap()) as usize, body[8], body[9], body[12])),
            b"IDAT" => idat.extend_from_slice(body),
            _ => {}
        }
        at += 12 + n;
    }
    let (w, h, depth, color, interlace) = header.expect("IHDR");
    assert_eq!((depth, color, interlace), (8, 0, 0), "only 8-bit grayscale, non-interlaced PNGs are read here");
    let raw = inflate(&idat[2..]); // skip the zlib header
    assert_eq!(raw.len(), h * (w + 1), "the decoded size does not match the header");
    let mut img = vec![0u8; h * w];
    for y in 0..h {
        let (filter, line) = (raw[y * (w + 1)], &raw[y * (w + 1) + 1..(y + 1) * (w + 1)]);
        for x in 0..w {
            let a = if x > 0 { i32::from(img[y * w + x - 1]) } else { 0 };
            let b = if y > 0 { i32::from(img[(y - 1) * w + x]) } else { 0 };
            let cc = if x > 0 && y > 0 { i32::from(img[(y - 1) * w + x - 1]) } else { 0 };
            let predictor = match filter {
                0 => 0,
                1 => a,
                2 => b,
                3 => (a + b) / 2,
                4 => {
                    let p = a + b - cc;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - cc).abs());
                    if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        cc
                    }
                }
                other => panic!("PNG filter {other}"),
            };
            img[y * w + x] = line[x].wrapping_add(predictor as u8);
        }
    }
    (h, w, img)
}

fn repo_root() -> std::path::PathBuf {
    golden_root().join("../../..")
}

/// The shipped Blue Marble cloud fraction as `measure.load` reads it: `u8 / 255` in float32 (2048 x 1024, so
/// no resampling).
fn reference_image(rel: &str) -> (Vec<f32>, usize, usize) {
    let bytes = std::fs::read(repo_root().join(rel)).unwrap_or_else(|e| panic!("reference {rel} not readable: {e}"));
    let (h, w, px) = read_gray_png(&bytes);
    assert_eq!((h, w), (1024, 2048), "measure.load expects the reference at 2048 x 1024");
    (px.iter().map(|&v| f32::from(v) / 255.0).collect(), h, w)
}

fn reference_measure(rel: &str) -> &'static Measure {
    static REF: OnceLock<Measure> = OnceLock::new();
    REF.get_or_init(|| {
        let (img, h, w) = reference_image(rel);
        metrics::measure(&img, h, w)
    })
}

// ---------------------------------------------------------------------------------------------------------
// Statistics

/// The metrics of one map as the cases flatten them (`_flat` of golden/cases/pipeline.py), under the case's
/// own names.
fn flat(m: &Measure, cmp: Option<(f64, f64)>) -> BTreeMap<String, Vec<f64>> {
    let mut f: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut put = |k: &str, v: Vec<f64>| {
        f.insert(k.to_string(), v);
    };
    put("global_mean", vec![m.global_mean]);
    put("zonal_lat", m.zonal_lat.clone());
    put("zonal", m.zonal.clone());
    put("hist", m.hist.clone());
    put("coherence", vec![m.coherence]);
    put("coherence_by_lat", m.coherence_by_lat.clone());
    put("spectral_slope", vec![m.spectral_slope]);
    put("big_object_share", vec![m.big_object_share]);
    put("median_object_km2", vec![m.median_object_km2]);
    put("seam_ratio", vec![m.seam_ratio]);
    put("pole_std", vec![m.pole_std.0, m.pole_std.1]);
    if let Some((rmse, l1)) = cmp {
        put("compare.zonal_rmse", vec![rmse]);
        put("compare.hist_l1", vec![l1]);
    }
    f
}

fn measure_out(f: &BTreeMap<String, Vec<f64>>) -> Out {
    f.iter().fold(Out::new(), |out, (k, v)| f64_out(out, k, v))
}

const STAT_WIDTH: usize = 1024;
const STAT_SEEDS: std::ops::RangeInclusive<u64> = 1..=8;

/// The port's own maps of Earth at 1024 x 512 for seeds 1 to 8, generated once with its own plans.
fn port_maps() -> &'static Vec<Vec<f32>> {
    static MAPS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
    MAPS.get_or_init(|| {
        STAT_SEEDS
            .map(|seed| {
                let out = generate(&EARTH, seed, STAT_WIDTH, &GenParams::default(), None, false).expect("Earth is bakeable");
                assert_eq!((out.height, out.width, out.meta.seed), (STAT_WIDTH / 2, STAT_WIDTH, seed));
                assert!(out.coverage.iter().all(|v| (0.0..=1.0).contains(v)), "seed {seed}: coverage outside [0, 1]");
                out.coverage
            })
            .collect()
    })
}

/// The port's metrics, flattened, for each of its eight maps (compared with the Blue Marble reference).
fn port_stats(reference: &Measure) -> Vec<BTreeMap<String, Vec<f64>>> {
    port_maps()
        .iter()
        .map(|cov| {
            let m = metrics::measure(cov, STAT_WIDTH / 2, STAT_WIDTH);
            let c = metrics::compare(&m, reference);
            flat(&m, Some((c.zonal_rmse, c.hist_l1)))
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------
// The judge of the statistical cases
//
// The case docs say a port "must land inside mean +- 3 std" of Python's eight seeds. Taken to the letter
// (every map of the port, every component, no miss) that cannot be asked of a faithful model: Python's std is
// itself an estimate from eight maps, so a NEW map from the very same model falls outside three sample
// deviations with probability P(|t_7| > 3 / sqrt(1 + 1/8)) = 2.5 % per component. With 504 (component, seed)
// pairs a faithful port misses about 13 of them (measured on this port: 4), and zero would be luck. The judge
// is therefore three tests, each stated with the probability a faithful model fails it:
//
// 1. THE BAND, literally, on the port's mean over its own seeds: `|port mean - mean| <= n_sigma std` for every
//    component, no miss allowed (a faithful port fails it with probability 0.06 % per component).
// 2. THE STANDARD ERROR, stricter than the band: the port's mean and Python's mean are each an average of
//    eight maps, so they differ by about std sqrt(1/8 + 1/8) = 0.5 std even for the same model, and a shift
//    beyond `n_sigma` of THAT, `|port mean - mean| <= n_sigma std sqrt(1/K + 1/8)` (1.5 std), is a shift of the
//    model. A faithful port misses it with probability P(|t_7| > 3) = 2.0 % per component, so a few misses are
//    allowed ([`binomial_bound`] at [`ALPHA`]).
// 3. THE SEEDS: each single map of the port against the band, the misses counted and bounded the same way
//    with the 2.5 % above.
//
// What it can and cannot see, measured on this port with mutated models (GOLDEN_SLOW, seeds 1 to 8): the
// mid-scale amplitude 1.5x or the threshold smoothing 3x fail test 1; the mid-scale amplitude 1.25x puts 3 of
// 63 means past 3 standard errors (6 are allowed) and passes; the ascent amplitude 1.5x or the moisture noise
// 1.5x move no mean past 1.8 std and pass. Eight Python seeds bound what a distribution can resolve, so this
// is a screen for gross statistical drift; the amplitudes themselves are pinned by the traced stage cases
// (`generate.*`). Nor is the wiring of the production plans (`GeneratePlans::from_seed`) left to it:
// `assert_from_seed_wiring` and the unit tests of `generate.rs` pin every plan against Python's own
// derivation, because the fine ladder of a 1024 map has a single octave, and a single octave has neither a
// gain nor a start frequency of its own (the streak gain and the regime texture's frequency and gain are
// invisible here by construction, not by lack of power: mutated, the maps are bit for bit the same).

/// Number of Python seeds behind the case's mean and std (`ddof 1`).
const PY_SEEDS: usize = 8;
/// P(|t_7| > 3): a faithful port's mean is further than 3 standard errors of the difference from Python's.
const P_MEAN_SE_MISS: f64 = 0.019_94;
/// P(|t_7| > 3 / sqrt(1 + 1/8)): one new map of a faithful model is further than 3 sample deviations.
const P_SEED_MISS: f64 = 0.025_46;
/// Probability that the judge rejects a faithful port through the number of its misses.
const ALPHA: f64 = 1e-3;

/// The smallest `k` with `P(Binomial(n, p) > k) <= alpha`: the number of misses a faithful port stays under
/// with probability `1 - alpha` when each of `n` independent comparisons misses with probability `p`.
fn binomial_bound(n: usize, p: f64, alpha: f64) -> usize {
    assert!((0.0..1.0).contains(&p) && alpha > 0.0);
    let mut pmf = (n as f64 * (1.0 - p).ln()).exp(); // P(X = 0)
    let mut tail = 1.0 - pmf; // P(X > 0)
    let mut k = 0;
    while tail > alpha && k < n {
        pmf *= (n - k) as f64 / (k + 1) as f64 * p / (1.0 - p);
        k += 1;
        tail -= pmf;
    }
    k
}

/// What the judge has seen: the misses of tests 2 and 3, and how many components it judged.
#[derive(Default)]
struct Verdict {
    judged: usize,
    seeds: usize,
    se_misses: Vec<String>,
    seed_misses: Vec<String>,
    worst_z_mean: f64,
}

impl Verdict {
    /// Judges one component: Python's `mean` and `std` over its seeds against `port`, the port's value for
    /// each of its own seeds. `n_sigma` is the case's band. A component whose std is 0 (the band centers) is
    /// a closed form and must match to 1e-9.
    ///
    /// # Panics
    /// On a miss of test 1, or of a closed form.
    fn judge(&mut self, name: &str, mean: f64, std: f64, n_sigma: f64, port: &[f64]) {
        let k = port.len() as f64;
        let port_mean = port.iter().sum::<f64>() / k;
        if std < 1e-12 {
            assert!((port_mean - mean).abs() <= 1e-9 * mean.abs().max(1.0), "{name}: port {port_mean} vs exact {mean}");
            return;
        }
        self.judged += 1;
        self.seeds += port.len();
        let z_mean = (port_mean - mean) / std;
        self.worst_z_mean = self.worst_z_mean.max(z_mean.abs());
        // Test 1: the band, to the letter, on the port's mean.
        assert!(
            z_mean.abs() <= n_sigma,
            "{name}: the port's mean over {} seeds ({port_mean:.6}) is outside the Python band \
             {mean:.6} +- {n_sigma} x {std:.6} ({z_mean:+.1} sigma)",
            port.len()
        );
        // Test 2: the same band in standard errors of the difference of two means.
        let se = (1.0 / k + 1.0 / PY_SEEDS as f64).sqrt();
        if z_mean.abs() > n_sigma * se {
            self.se_misses.push(format!(
                "{name}: the port's mean {port_mean:.6} is {z_mean:+.2} std from {mean:.6} (std {std:.6}), over {:.2}",
                n_sigma * se
            ));
        }
        // Test 3: every seed of the port against the band.
        for (seed, &v) in port.iter().enumerate() {
            if (v - mean).abs() > n_sigma * std {
                self.seed_misses.push(format!(
                    "{name} seed {}: {v:.6} is {:+.1} sigma from {mean:.6} (std {std:.6})",
                    seed + 1,
                    (v - mean) / std
                ));
            }
        }
    }

    /// Prints the misses and bounds their number (see the module section above).
    ///
    /// # Panics
    /// If a count is more than a faithful port would produce with probability `1 - ALPHA`.
    fn conclude(&self, what: &str) {
        let allowed_se = binomial_bound(self.judged, P_MEAN_SE_MISS, ALPHA);
        let allowed_seed = binomial_bound(self.seeds, P_SEED_MISS, ALPHA);
        println!(
            "{what}: {} components judged; worst mean shift {:.2} std; {} (allowed {allowed_se}) outside 3 standard errors, \
             {} of {} (allowed {allowed_seed}) component-seeds outside the band",
            self.judged,
            self.worst_z_mean,
            self.se_misses.len(),
            self.seed_misses.len(),
            self.seeds
        );
        for m in self.se_misses.iter().chain(&self.seed_misses) {
            println!("  {m}");
        }
        assert!(
            self.se_misses.len() <= allowed_se,
            "{what}: {} means are more than 3 standard errors from Python's, more than the {allowed_se} allowed:\n{}",
            self.se_misses.len(),
            self.se_misses.join("\n")
        );
        assert!(
            self.seed_misses.len() <= allowed_seed,
            "{what}: {} component-seeds outside the band, more than the {allowed_seed} allowed:\n{}",
            self.seed_misses.len(),
            self.seed_misses.join("\n")
        );
    }
}

/// Python's own per-seed maps, pinned by the cases `stat.earth.seed<k>.w1024` (the metrics of each map,
/// under the names of [`flat`]), read from the golden data next to `c`.
fn python_seed_pins(c: &Case) -> Vec<Case> {
    let seeds = c.param("seeds").as_array().expect("seeds is a list");
    assert_eq!(seeds.len(), PY_SEEDS, "{}: the case's number of Python seeds", c.id);
    let root = c.dir.parent().expect("a case directory has a parent");
    seeds
        .iter()
        .map(|s| {
            let k = s.as_u64().expect("a seed");
            let pin = load_case(&root.join(format!("stat.earth.seed{k}.w1024")));
            assert_eq!(pin.p_i64("seed") as u64, k, "{}: pin {} is of another seed", c.id, pin.id);
            pin
        })
        .collect()
}

/// `mean` and sample `std` (`ddof 1`) over the rows of `x`, per column.
fn mean_std(x: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len() as f64;
    let mean: Vec<f64> = (0..x[0].len()).map(|i| x.iter().map(|r| r[i]).sum::<f64>() / n).collect();
    let std = (0..x[0].len()).map(|i| (x.iter().map(|r| (r[i] - mean[i]).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()).collect();
    (mean, std)
}

/// `pipeline.stat.metrics_over_seeds`: the port's maps against Python's mean and standard deviation of every
/// metric (`mean.<metric>` and `std.<metric>` arrays of the case).
///
/// Two different things are checked, and only the first concerns the port. The port is judged by [`Verdict`] on
/// its own maps. The arrays handed back to the runner are the distribution RECOMPUTED from Python's per-seed
/// pins (the eight `stat.earth.seed<k>.w1024` cases): the runner then verifies, within the case's tolerances
/// ("the Python mean and std themselves are held to the per-seed tolerances and 5 %"), that the reference the
/// port was judged against is the distribution of those pins. That check is of the reference, not of the port,
/// and is described as such.
fn over_seeds_case(c: &Case) -> Out {
    let n_sigma = c.p_f64("n_sigma");
    assert_eq!(c.p_usize("width"), STAT_WIDTH);
    assert_eq!(c.param("seeds").as_array().unwrap().len(), STAT_SEEDS.count());
    let reference = reference_measure(c.p_str("reference"));
    let stats = port_stats(reference);
    let mut verdict = Verdict::default();
    for (key, (want_mean, _)) in c.expected.iter().filter(|(k, _)| k.starts_with("mean.")) {
        let metric = &key["mean.".len()..];
        let (means, stds) = (want_mean.as_f64(), c.expected[&format!("std.{metric}")].0.as_f64());
        for i in 0..means.len() {
            let port: Vec<f64> = stats.iter().map(|s| s[metric][i]).collect();
            verdict.judge(&format!("{metric}[{i}]"), means[i], stds[i], n_sigma, &port);
        }
    }
    verdict.conclude("metrics_over_seeds");
    // The reference, recomputed from Python's pins.
    let pins = python_seed_pins(c);
    c.expected.keys().filter_map(|k| k.strip_prefix("mean.")).fold(Out::new(), |out, metric| {
        let rows: Vec<Vec<f64>> = pins.iter().map(|p| p.expected[metric].0.as_f64()).collect();
        let (mean, std) = mean_std(&rows);
        f64_out(f64_out(out, &format!("mean.{metric}"), &mean), &format!("std.{metric}"), &std)
    })
}

/// `pipeline.stat.zonal_residual`: the band residual of the zonal mean against the target, and the closed
/// forms around it.
fn zonal_residual_case(c: &Case) -> Out {
    let n_sigma = c.p_f64("n_sigma");
    let (h, w) = (STAT_WIDTH / 2, STAT_WIDTH);
    let (_, lat) = grid_lonlat(w, h);
    let target_rows: Vec<f64> = lat.iter().map(|&la| target_zonal_coverage(la * (180.0 / std::f64::consts::PI), 30.0, 0.0)).collect();
    let tgt_img: Vec<f64> = target_rows.iter().flat_map(|&t| std::iter::repeat_n(t, w)).collect();
    let (centers, target_band) = metrics::zonal_profile(&tgt_img, h, w, metrics::DEFAULT_BAND_DEG);
    let at_centers: Vec<f64> = centers.iter().map(|&la| target_zonal_coverage(la, 30.0, 0.0)).collect();

    let maps = port_maps();
    let residuals: Vec<Vec<f64>> = maps
        .iter()
        .map(|cov| {
            let (_, prof) = metrics::zonal_profile(cov, h, w, metrics::DEFAULT_BAND_DEG);
            prof.iter().zip(&target_band).map(|(&p, &t)| f64::from(p) - t).collect()
        })
        .collect();

    // The band residual of the port against Python's mean +- n_sigma std over its seeds (the case's rule).
    let (want_mean, want_std) = (c.expected["residual_mean"].0.as_f64(), c.expected["residual_std"].0.as_f64());
    let mut verdict = Verdict::default();
    for i in 0..centers.len() {
        let port: Vec<f64> = residuals.iter().map(|r| r[i]).collect();
        verdict.judge(&format!("residual[{i}] (band {:.1} deg)", centers[i]), want_mean[i], want_std[i], n_sigma, &port);
    }
    verdict.conclude("zonal_residual");

    // The calibration solves every row's mean to its target: what is left away from the poles is the
    // smoothing of the thresholds and the sub-texel noise. Python's seed-1 map has an rms of 0.017 and a
    // maximum of 0.067 there (the case doc); the port's maps must be of the same order, not 2x worse.
    let away: Vec<usize> = (0..h).filter(|&r| lat[r].to_degrees().abs() < 80.0).collect();
    for (k, cov) in maps.iter().enumerate() {
        let resid: Vec<f64> = away
            .iter()
            .map(|&r| cov[r * w..(r + 1) * w].iter().map(|&v| f64::from(v)).sum::<f64>() / w as f64 - target_rows[r])
            .collect();
        let rms = (resid.iter().map(|v| v * v).sum::<f64>() / resid.len() as f64).sqrt();
        let max = resid.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        println!("zonal_residual seed {}: row residual rms {rms:.4}, max {max:.4} (|lat| < 80)", k + 1);
        assert!(rms < 2.0 * 0.017, "seed {}: row residual rms {rms} is not of the order of Python's 0.017", k + 1);
        assert!(max < 2.0 * 0.067, "seed {}: row residual max {max} is not of the order of Python's 0.067", k + 1);
    }

    // What the runner compares. The closed forms (the band centers, the target) are the port's. The band means
    // of Python's seed 1 and the residual's mean and std over its eight seeds are Python's pins minus the
    // port's closed-form band target, recomputed (a check of the reference, see `over_seeds_case`). The row
    // residual of seed 1 and its maximum are Python's own draws: no pin holds that map, so they are handed back
    // as recorded (they are not a target for a port; the port's rows are judged just above).
    let pins = python_seed_pins(c);
    let bands: Vec<Vec<f64>> = pins.iter().map(|p| p.expected["zonal"].0.as_f64()).collect();
    let band_residuals: Vec<Vec<f64>> = bands.iter().map(|b| b.iter().zip(&target_band).map(|(z, t)| z - t).collect()).collect();
    let (res_mean, res_std) = mean_std(&band_residuals);
    let seed1 = pins.iter().position(|p| p.p_i64("seed") == c.p_i64("seed")).expect("the case's seed is one of the seeds");
    let recorded = |name: &str| c.expected[name].0.clone();
    Out::new()
        .put("row_residual", recorded("row_residual"))
        .put("max_abs_row_residual", recorded("max_abs_row_residual"))
        .f64("band_lat", &[centers.len()], &centers)
        .f64("target_at_centers", &[centers.len()], &at_centers)
        .f64("target_band_mean", &[target_band.len()], &target_band)
        .f64("zonal", &[centers.len()], &bands[seed1])
        .f64("residual", &[centers.len()], &band_residuals[seed1])
        .f64("residual_mean", &[centers.len()], &res_mean)
        .f64("residual_std", &[centers.len()], &res_std)
}

#[cfg(test)]
mod judge_tests {
    use super::*;

    /// Python's distribution for a component: mean 0.5, std 0.1. The port's eight values: a spread of std 0.1
    /// around `centre`, in a fixed pattern.
    fn port_values(centre: f64) -> Vec<f64> {
        const PATTERN: [f64; 8] = [-1.4, -0.9, -0.5, -0.1, 0.2, 0.6, 0.9, 1.2];
        PATTERN.iter().map(|z| centre + 0.1 * z).collect()
    }

    fn judged(centre: f64, components: usize) -> Verdict {
        let mut v = Verdict::default();
        for i in 0..components {
            v.judge(&format!("c{i}"), 0.5, 0.1, 3.0, &port_values(centre));
        }
        v
    }

    #[test]
    fn the_binomial_bound_is_the_smallest_count_a_faithful_port_stays_under() {
        // Cross-checked with scipy.stats.binom.sf: the smallest k with P(X > k) <= 1e-3.
        assert_eq!(binomial_bound(504, P_SEED_MISS, ALPHA), 25);
        assert_eq!(binomial_bound(288, P_SEED_MISS, ALPHA), 17);
        assert_eq!(binomial_bound(63, P_MEAN_SE_MISS, ALPHA), 6);
        assert_eq!(binomial_bound(36, P_MEAN_SE_MISS, ALPHA), 4);
        assert_eq!(binomial_bound(100, 0.0, ALPHA), 0, "a comparison that cannot miss allows no miss");
        assert_eq!(binomial_bound(1, 0.5, 0.6), 0, "P(X > 0) = 0.5 <= 0.6");
        assert_eq!(binomial_bound(1, 0.5, 0.4), 1, "the count cannot exceed the number of comparisons");
    }

    #[test]
    fn a_port_on_python_s_distribution_is_judged_clean() {
        let v = judged(0.5, 63);
        assert!(v.se_misses.is_empty() && v.seed_misses.is_empty() && v.judged == 63 && v.seeds == 63 * 8);
        v.conclude("on the distribution");
    }

    #[test]
    fn a_shift_of_the_mean_is_caught_well_inside_the_three_sigma_band() {
        // 1.6 std: inside the doc's band (3 std) for every component, but past 3 standard errors (1.5 std) of the
        // difference of two eight-seed means. Every component misses test 2, far more than a faithful port's 6.
        let v = judged(0.5 + 0.16, 63);
        assert_eq!(v.se_misses.len(), 63);
        let caught = std::panic::catch_unwind(|| v.conclude("shifted"));
        assert!(caught.is_err(), "63 of 63 means off by 1.6 std must fail the judge");
        // A shift of 1.0 std is within the standard error (the same two models differ that much by chance).
        judged(0.5 + 0.10, 63).conclude("a shift of one std");
    }

    #[test]
    fn a_few_stray_components_are_tolerated_and_many_are_not() {
        // 6 of 63 means beyond 1.5 std is what a faithful port can do (bound 6); 7 is not.
        let mut v = judged(0.5, 57);
        for i in 0..6 {
            v.judge(&format!("far{i}"), 0.5, 0.1, 3.0, &port_values(0.5 + 0.16));
        }
        assert_eq!((v.se_misses.len(), v.judged), (6, 63));
        v.conclude("six strays");
        v.judge("far6", 0.5, 0.1, 3.0, &port_values(0.5 + 0.16));
        assert!(std::panic::catch_unwind(|| v.conclude("seven strays")).is_err());
    }

    #[test]
    #[should_panic(expected = "outside the Python band")]
    fn a_mean_outside_the_three_sigma_band_fails_at_once() {
        judged(0.5 + 0.31, 1);
    }

    #[test]
    fn a_single_seed_off_the_band_is_counted_not_fatal() {
        let mut port = port_values(0.5);
        port[3] = 0.5 + 0.35; // 3.5 sigma for one map; the mean moves by 0.044 std
        let mut v = Verdict::default();
        v.judge("one outlier", 0.5, 0.1, 3.0, &port);
        assert_eq!((v.seed_misses.len(), v.se_misses.len()), (1, 0));
        assert!(v.seed_misses[0].contains("seed 4"), "{}", v.seed_misses[0]);
        v.conclude("one outlier");
    }

    #[test]
    fn a_closed_form_must_match_and_is_not_counted_as_a_judged_component() {
        let mut v = Verdict::default();
        v.judge("band centre", 12.5, 0.0, 3.0, &[12.5; 8]);
        assert_eq!((v.judged, v.seeds), (0, 0));
        assert!(std::panic::catch_unwind(|| Verdict::default().judge("band centre", 12.5, 0.0, 3.0, &[12.6; 8])).is_err());
    }

    #[test]
    fn mean_std_is_the_sample_statistic() {
        let (m, s) = mean_std(&[vec![1.0, 10.0], vec![2.0, 10.0], vec![6.0, 10.0]]);
        assert_eq!(m, [3.0, 10.0]);
        assert!((s[0] - 7.0_f64.sqrt()).abs() < 1e-12 && s[1] == 0.0, "{s:?}"); // sqrt(((-2)^2 + (-1)^2 + 3^2) / 2) = sqrt(7)
    }
}

// ---------------------------------------------------------------------------------------------------------

fn port(c: &Case) -> Option<Out> {
    Some(match c.function.as_str() {
        "cloudgen.generate.generate" if c.id == "generate.rejects_non_water" => rejects_non_water_case(c),
        "cloudgen.generate.generate" => generate_case(c),
        "bake.library_body" => library_body_case(c),
        "cloudgen.metrics.measure" if c.id == "stat.reference.bluemarble" => {
            let m = reference_measure(c.p_str("reference"));
            measure_out(&flat(m, None))
        }
        // The per-seed pins of numpy's own maps: not a target for the port (EXPECTED_FAILURES).
        "cloudgen.metrics.measure" => return None,
        "pipeline.stat.metrics_over_seeds" => over_seeds_case(c),
        "pipeline.stat.zonal_residual" => zonal_residual_case(c),
        _ => return None,
    })
}

#[test]
fn pipeline_cases() {
    let cfg = Config {
        group: "pipeline",
        prefixes: &["cloudgen.generate.generate", "cloudgen.metrics.measure", "bake.library_body", "pipeline.stat."],
        expected_failures: EXPECTED_FAILURES,
    };
    run(&cfg, port).assert_clean();
}
