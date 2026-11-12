//! Golden cases of cloudgen/climate.py and cloudgen/flow.py (group climate_flow, functions cloudgen.climate.*
//! and cloudgen.flow.*).
//!
//! The climate cases and the flow kernels (zonal_rate, velocity, backtrace, ascent, comma ascent) are plain
//! numeric comparisons. The ones that need a word more:
//!
//! * A rotation period of `None` is stored as NaN in the case arrays. Every body table is run twice, once with
//!   NaN read as `None` and once as `Some(NaN)`, and the two must agree bit for bit (both mean "no spin").
//! * `build_flow` draws from numpy's generator (`default_rng(seed ^ 0x5F10)`), which the port does not and
//!   must not reproduce (README rule 3). Two kinds of case are therefore EXPECTED FAILURES, each with its
//!   reason, and the glue returns the port's OWN numbers for them: nothing is echoed from the case.
//!   - `build_flow.<body>.seed<k>` (tag `rng`) holds numpy's vortex list; the port's list differs in every
//!     lon, lat, radius, omega, age and in the number of tropical storms. (`slow_rotator_4d` has no storm
//!     track, so no vortex and no draw: it is a plain passing case.)
//!   - `build_flow.stats.<body>` (tags `stat`, `rng`, `slow`) holds the sample of numpy's population over 300
//!     seeds (ranges, tropical-count mean, std and histogram); the port's sample is another one.
//!
//!   The runner counts a panic inside an expected failure as that failure, so the checks that bind the port
//!   are NOT inside the cases: they are the separate tests at the end of this file, which fail on their own.
//!   `build_flow_statistics_follow_the_case_docs` runs the checks the stat cases document for a port with
//!   another generator (counts and the deterministic outputs equal to the case, hard bounds, sample ranges
//!   within 10 per cent of each bound, Poisson(1.5) by pooled mean and chi-square) on the port's population;
//!   `build_flow_seeds_agree_with_numpy_where_the_draws_do_not_decide` compares the port's `build_flow(body,
//!   seed)` with the case on flow scalars, the number of lows and highs per hemisphere and every winding sign;
//!   `build_flow_replays_numpys_recorded_vortices` inverts numpy's recorded vortices into the draws that made
//!   them (each must lie in the range Python draws it from) and runs them through the port's deterministic
//!   half, which must give the recorded numbers back within the case tolerance.
//! * Three more tests at the end pin what no case reaches (the saturation of `map_width_for_texel`, the
//!   half-to-even rounding of the wave count, the ranges and the Poisson mean of the vortex sampler); they
//!   repeat unit tests of `climate.rs` and `flow.rs` so that this file alone binds them.

use cloud_map_generator_core::climate::*;
use cloud_map_generator_core::flow::*;
use cloud_map_generator_core::sphere::Vec3;
use cloud_map_generator_golden::{compare_array, load_group, run, Array, Case, Config, DType, Out};
use num_traits::Float;
use serde_json::Value;
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// Known failures. Every entry is a case whose frozen output is numpy's random stream (see the module doc);
/// the checks that bind the port in its place are the tests at the end of the file.
const EXPECTED_FAILURES: &[(&str, &str)] = &[
    ("build_flow.earth.seed0", SEED_REASON),
    ("build_flow.earth.seed12345", SEED_REASON),
    ("build_flow.earth.seed3", SEED_REASON),
    ("build_flow.earth.seed4294967301", SEED_REASON),
    ("build_flow.earth.seed7", SEED_REASON),
    ("build_flow.fast_rotator_0.4d.seed7", SEED_REASON),
    ("build_flow.mars.seed7", SEED_REASON),
    ("build_flow.retrograde_earth.seed7", SEED_REASON),
    ("build_flow.super_earth_2r.seed7", SEED_REASON),
    ("build_flow.threshold_below_50.seed7", SEED_REASON),
    ("build_flow.very_fast_0.05d.seed7", SEED_REASON),
    ("build_flow.stats.earth", STATS_REASON),
    ("build_flow.stats.fast_rotator_0.4d", STATS_REASON),
    ("build_flow.stats.mars", STATS_REASON),
    ("build_flow.stats.retrograde_earth", STATS_REASON),
    ("build_flow.stats.super_earth_2r", STATS_REASON),
    ("build_flow.stats.threshold_below_50", STATS_REASON),
];

const SEED_REASON: &str = "rng: lon, lat, radius, omega, age and the tropical count are numpy's default_rng(seed ^ \
0x5F10) bit stream, which the port's generator does not reproduce; flow scalars, the lows/highs per hemisphere and \
every winding sign are asserted by build_flow_seeds_agree_with_numpy_where_the_draws_do_not_decide, and the \
deterministic half by build_flow_replays_numpys_recorded_vortices";

const STATS_REASON: &str = "rng: the sample outputs (*.range, tropical.count.mean_by_hemisphere/pooled_mean/\
pooled_std/histogram) are numpy's sample over 300 seeds; the bounds, counts, pmf and sign fractions are compared \
and the sample checks the case documents are run by build_flow_statistics_follow_the_case_docs";

// ---------------------------------------------------------------------------------------------------------
// Reading cases

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

/// Runs the generic handler `$f::<f32>` or `$f::<f64>` by the dtype of the input array `$input`.
macro_rules! by_dtype {
    ($c:expr, $input:expr, $f:ident) => {
        match $c.input($input).dtype {
            DType::F32 => $f::<f32>($c),
            DType::F64 => $f::<f64>($c),
            d => panic!("case {}: input {} is {}, not a float", $c.id, $input, d.name()),
        }
    };
}

fn points<F: Elem>(a: &Array) -> Vec<Vec3<F>> {
    F::read(a).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}
fn flat<F: Copy>(v: &[Vec3<F>]) -> Vec<F> {
    v.iter().flatten().copied().collect()
}

fn kind_of(name: &str) -> VortexKind {
    match name {
        "cyclone" => VortexKind::Cyclone,
        "anticyclone" => VortexKind::Anticyclone,
        "tropical" => VortexKind::Tropical,
        other => panic!("unknown vortex kind {other:?}"),
    }
}

fn kind_name(k: VortexKind) -> &'static str {
    match k {
        VortexKind::Cyclone => "cyclone",
        VortexKind::Anticyclone => "anticyclone",
        VortexKind::Tropical => "tropical",
    }
}

/// A vortex of `params`: a missing kind is a cyclone (the comma cases), a missing age the default.
fn vortex_of(v: &Value) -> Vortex {
    let f = |k: &str| v[k].as_f64().unwrap_or_else(|| panic!("vortex field {k} is not a number in {v}"));
    Vortex {
        kind: v.get("kind").and_then(Value::as_str).map_or(VortexKind::Cyclone, kind_of),
        lon: f("lon"),
        lat: f("lat"),
        radius: f("radius"),
        omega: f("omega"),
        ascent: f("ascent"),
        age: v.get("age").and_then(Value::as_f64).unwrap_or(DEFAULT_AGE),
    }
}

/// The hand-built flow of a case: `hadley_deg`, the three shear scalars and the vortex list of its params.
fn flow_of(c: &Case) -> Flow {
    let vortices = c.param("vortices").as_array().unwrap_or_else(|| panic!("case {}: vortices is not a list", c.id));
    Flow {
        hadley_deg: c.p_f64("hadley_deg"),
        vortices: vortices.iter().map(vortex_of).collect(),
        jet: c.p_f64("jet"),
        trades: c.p_f64("trades"),
        polar_easterlies: c.p_f64("polar_easterlies"),
    }
}

fn regimes_of(c: &Case, n: usize) -> Vec<String> {
    let list =
        c.param("cloud_regime").as_array().unwrap_or_else(|| panic!("case {}: cloud_regime is not a list", c.id));
    assert_eq!(list.len(), n);
    list.iter().map(|r| r.as_str().expect("regime string").to_string()).collect()
}

/// The rows of a body table. A NaN rotation period is `None` (`nan_is_none`) or `Some(NaN)`.
fn bodies_of(c: &Case, nan_is_none: bool) -> Vec<BodyInputs> {
    let period = c.input("rotation_period_days").as_f64();
    let (radius, gravity, height) =
        (c.input("radius_km").as_f64(), c.input("gravity_mps2").as_f64(), c.input("scale_height_km").as_f64());
    let regimes = regimes_of(c, period.len());
    (0..period.len())
        .map(|i| {
            let p = if period[i].is_nan() && nan_is_none { None } else { Some(period[i]) };
            BodyInputs::new(p, radius[i], gravity[i], height[i]).with_regime(regimes[i].clone())
        })
        .collect()
}

/// One body of a `[period, radius, gravity, scale height]` vector (a NaN period is `None`).
fn body_of_vec(v: &Array, regime: &str) -> BodyInputs {
    let b = v.as_f64();
    BodyInputs::new(if b[0].is_nan() { None } else { Some(b[0]) }, b[1], b[2], b[3]).with_regime(regime.to_string())
}

/// `f` over a body table, with the NaN periods read both ways, which must agree.
fn per_body<T: PartialEq + std::fmt::Debug>(c: &Case, f: impl Fn(&BodyInputs) -> T) -> Vec<T> {
    let (none, some_nan) = (bodies_of(c, true), bodies_of(c, false));
    let a: Vec<T> = none.iter().map(&f).collect();
    let b: Vec<T> = some_nan.iter().map(&f).collect();
    for i in 0..a.len() {
        let eq = a[i] == b[i] || format!("{:?}", a[i]) == format!("{:?}", b[i]); // NaN = NaN
        assert!(eq, "case {}: row {i}: a NaN period gives {:?} as None and {:?} as Some(NaN)", c.id, a[i], b[i]);
    }
    a
}

// ---------------------------------------------------------------------------------------------------------
// climate

fn constants(c: &Case) -> Out {
    let names: Vec<String> =
        c.param("names").as_array().expect("names").iter().map(|n| n.as_str().expect("name").to_string()).collect();
    let values: Vec<f64> = names
        .iter()
        .map(|n| match n.as_str() {
            "EARTH_ROTATION_DAYS" => EARTH_ROTATION_DAYS,
            "EARTH_RADIUS_KM" => EARTH_RADIUS_KM,
            "EARTH_GRAVITY" => EARTH_GRAVITY,
            "EARTH_SCALE_HEIGHT_KM" => EARTH_SCALE_HEIGHT_KM,
            "EARTH_HADLEY_EDGE_DEG" => EARTH_HADLEY_EDGE_DEG,
            "HADLEY_MIN_DEG" => HADLEY_MIN_DEG,
            "HADLEY_MAX_DEG" => HADLEY_MAX_DEG,
            "DEFORMATION_FRACTION_OF_HADLEY" => DEFORMATION_FRACTION_OF_HADLEY,
            "EARTH_STORM_TRACK_DEG" => EARTH_STORM_TRACK_DEG,
            "STORM_TRACK_IN_HADLEY" => STORM_TRACK_IN_HADLEY,
            "STORM_TRACK_MAX_DEG" => STORM_TRACK_MAX_DEG,
            "MAP_TEXEL_KM" => MAP_TEXEL_KM,
            "MAP_MAX_WIDTH" => MAP_MAX_WIDTH as f64,
            "Flow.jet" => Flow::new(30.0).jet,
            "Flow.trades" => Flow::new(30.0).trades,
            "Flow.polar_easterlies" => Flow::new(30.0).polar_easterlies,
            "Vortex.age" => Vortex::new(VortexKind::Cyclone, 0.0, 0.0, 0.1, 1.0, 1.0).age,
            other => panic!("constant {other} is not ported"),
        })
        .collect();
    Out::new().f64("values", &[values.len()], &values)
}

fn body_inputs(c: &Case) -> Out {
    let default = BodyInputs::new(Some(1.0), 1.0, 1.0, 1.0);
    assert_eq!(default.cloud_regime, c.p_str("default_cloud_regime"));
    assert_eq!(EARTH.cloud_regime, c.p_str("earth_cloud_regime"));
    let e =
        [EARTH.rotation_period_days.expect("Earth spins"), EARTH.radius_km, EARTH.gravity_mps2, EARTH.scale_height_km];
    Out::new().f64("earth", &[4], &e).bool("default_regime_bakeable", &[1], &[is_bakeable(&default)])
}

/// The Hadley edges of a case: its `hadley_deg` input (one row each) or its scalar param.
fn rows(c: &Case, name: &str) -> (Vec<f64>, bool) {
    if c.has_input(name) {
        (c.input(name).as_f64(), true)
    } else {
        (vec![c.p_f64(name)], false)
    }
}

fn convective<F: Elem>(c: &Case) -> Out {
    let lat: Vec<F> = F::read(c.input("lat_deg"));
    let (hadley, as_rows) = rows(c, "hadley_deg");
    let shape = if as_rows { vec![hadley.len(), lat.len()] } else { vec![lat.len()] };
    let out: Vec<F> = hadley.iter().flat_map(|&h| lat.iter().map(move |&l| convective_tendency(l, h))).collect();
    F::put(Out::new(), "c", &shape, &out)
}

fn target<F: Elem>(c: &Case) -> Out {
    let lat: Vec<F> = F::read(c.input("lat_deg"));
    let (hadley, h_rows) = rows(c, "hadley_deg");
    let (shift, s_rows) = rows(c, "itcz_shift_deg");
    let mut shape = Vec::new();
    if h_rows {
        shape.push(hadley.len());
    }
    if s_rows {
        shape.push(shift.len());
    }
    shape.push(lat.len());
    let mut out = Vec::new();
    for &h in &hadley {
        for &s in &shift {
            out.extend(lat.iter().map(|&l| target_zonal_coverage(l, h, s)));
        }
    }
    Out::new().f64("coverage", &shape, &out)
}

fn map_width(c: &Case) -> Out {
    let r = c.input("radius_km").as_f64();
    let width: Vec<i64> = if c.has_input("texel_km") && c.has_input("max_width") {
        let (t, m) = (c.input("texel_km").as_f64(), c.input("max_width").as_i64());
        (0..r.len()).map(|i| map_width_for_texel(r[i], t[i], m[i])).collect()
    } else if c.has_input("texel_km") {
        // default cap
        assert_eq!(c.p_i64("max_width"), MAP_MAX_WIDTH);
        let t = c.input("texel_km").as_f64();
        (0..r.len()).map(|i| map_width_for_texel(r[i], t[i], MAP_MAX_WIDTH)).collect()
    } else if c.has_input("max_width") {
        // default texel
        assert_eq!(c.p_f64("texel_km"), MAP_TEXEL_KM);
        let m = c.input("max_width").as_i64();
        (0..r.len()).map(|i| map_width_for_texel(r[i], MAP_TEXEL_KM, m[i])).collect()
    } else if c.p_f64("texel_km") == MAP_TEXEL_KM && c.p_i64("max_width") == MAP_MAX_WIDTH {
        // both defaults, as the callers use it
        r.iter().map(|&r| map_width_for_radius(r)).collect()
    } else {
        // explicit texel and cap in the params
        let (t, m) = (c.p_f64("texel_km"), c.p_i64("max_width"));
        r.iter().map(|&r| map_width_for_texel(r, t, m)).collect()
    };
    Out::new().i64("width", &[width.len()], &width)
}

fn gauss_sweep(c: &Case) -> Out {
    let (x, mu, sigma) = (c.input("x").as_f64(), c.input("mu").as_f64(), c.input("sigma").as_f64());
    let g: Vec<f64> =
        (0..mu.len()).flat_map(|i| x.iter().map(|&x| gauss(x, mu[i], sigma[i])).collect::<Vec<_>>()).collect();
    Out::new().f64("g", &[mu.len(), x.len()], &g)
}

fn smoothstep_sweep(c: &Case) -> Out {
    let (x, a, b) = (c.input("x").as_f64(), c.input("a").as_f64(), c.input("b").as_f64());
    let s: Vec<f64> =
        (0..a.len()).flat_map(|i| x.iter().map(|&x| smoothstep(a[i], b[i], x)).collect::<Vec<_>>()).collect();
    Out::new().f64("s", &[a.len(), x.len()], &s)
}

// ---------------------------------------------------------------------------------------------------------
// flow kernels

fn flow_defaults(c: &Case) -> Out {
    assert_eq!(c.p_usize("n_vortices"), Flow::new(30.0).vortices.len());
    let f = Flow::new(30.0);
    Out::new().f64("shear", &[3], &[f.jet, f.trades, f.polar_easterlies]).f64(
        "default_age",
        &[1],
        &[Vortex::new(VortexKind::Cyclone, 0.0, 0.0, 0.1, 1.0, 1.0).age],
    )
}

fn vortex_center(c: &Case) -> Out {
    let (lon, lat) = (c.input("lon").as_f64(), c.input("lat").as_f64());
    let center: Vec<Vec3<f64>> = lon
        .iter()
        .zip(&lat)
        .map(|(&lon, &lat)| Vortex::new(VortexKind::Cyclone, lon, lat, 0.1, 1.0, 1.0).center())
        .collect();
    Out::new().f64("center", &[lon.len(), 3], &flat(&center))
}

fn zonal_rate<F: Elem>(c: &Case) -> Out {
    let lat: Vec<F> = F::read(c.input("lat"));
    if c.has_param("flows") {
        // One row per (hadley, jet, trades, polar easterlies) of params.flows.
        let flows = c.param("flows").as_array().expect("flows");
        let mut rate = Vec::new();
        for row in flows {
            let r: Vec<f64> = row.as_array().expect("flow row").iter().map(|v| v.as_f64().expect("number")).collect();
            let f = Flow { hadley_deg: r[0], jet: r[1], trades: r[2], polar_easterlies: r[3], vortices: vec![] };
            rate.extend(lat.iter().map(|&l| f.zonal_rate(l)));
        }
        Out::new().f64("rate", &[flows.len(), lat.len()], &rate)
    } else {
        let f = flow_of(c);
        let rate: Vec<f64> = lat.iter().map(|&l| f.zonal_rate(l)).collect();
        Out::new().f64("rate", &[lat.len()], &rate)
    }
}

fn velocity<F: Elem>(c: &Case) -> Out {
    let p = c.input("p");
    let v = flow_of(c).velocity(&points::<F>(p));
    F::put(Out::new(), "v", &p.shape, &flat(&v))
}

fn backtrace<F: Elem>(c: &Case) -> Out {
    let p = c.input("p");
    let (flow, pts, time, steps) = (flow_of(c), points::<F>(p), c.p_f64("time"), c.p_usize("steps"));
    // The default call (Python: time=1.0, steps=48) goes through the defaults of the port, which pins them.
    let q = if time == 1.0 && steps == 48 { flow.backtrace_default(&pts) } else { flow.backtrace(&pts, time, steps) };
    Out::new().f32("q", &p.shape, &flat(&q))
}

fn ascent<F: Elem>(c: &Case) -> Out {
    let p = c.input("p");
    let a = flow_of(c).ascent(&points::<F>(p));
    Out::new().f32("a", &[a.len()], &a)
}

fn comma_ascent<F: Elem>(c: &Case) -> Out {
    let p = points::<F>(c.input("p"));
    let vx = vortex_of(c.param("vortex"));
    if c.has_input("ages") {
        // Output row i is ages[i]; the vortex of the params has no age of its own.
        let ages = c.input("ages").as_f64();
        let g: Vec<F> = ages.iter().flat_map(|&age| Flow::comma_ascent(&Vortex { age, ..vx }, &p)).collect();
        F::put(Out::new(), "g", &[ages.len(), p.len()], &g)
    } else {
        assert!(c.param("vortex").get("age").is_some(), "case {}: the vortex has no age", c.id);
        F::put(Out::new(), "g", &[p.len()], &Flow::comma_ascent(&vx, &p))
    }
}

// ---------------------------------------------------------------------------------------------------------
// build_flow

fn hemisphere(v: &Vortex) -> f64 {
    1.0_f64.copysign(v.lat)
}

/// `build_flow.structure.bodies`: what does not depend on the draws, asserted equal over 20 seeds as the case
/// builder does.
fn build_flow_structure(c: &Case) -> Out {
    let ascent_by_kind = c.param("ascent_by_kind");
    let bodies = bodies_of(c, true);
    let n = bodies.len();
    let (mut scalars, mut cyc, mut anti, mut has, mut wavenumber_x) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (row, body) in bodies.iter().enumerate() {
        let mut first: Option<([i64; 2], [i64; 2], [f64; 4])> = None;
        for seed in 0..20u64 {
            let flow = build_flow(body, seed);
            for v in &flow.vortices {
                let w = ascent_by_kind[kind_name(v.kind)].as_f64().expect("ascent weight");
                assert_eq!(v.ascent, w, "row {row} seed {seed}: ascent weight of {v:?}");
                assert!(v.kind == VortexKind::Cyclone || v.age == 1.0, "row {row}: age of {v:?}");
            }
            let count = |kind: VortexKind, s: f64| {
                flow.vortices.iter().filter(|v| v.kind == kind && hemisphere(v) == s).count() as i64
            };
            let this = (
                [count(VortexKind::Cyclone, 1.0), count(VortexKind::Cyclone, -1.0)],
                [count(VortexKind::Anticyclone, 1.0), count(VortexKind::Anticyclone, -1.0)],
                [flow.hadley_deg, flow.jet, flow.trades, flow.polar_easterlies],
            );
            match &first {
                None => first = Some(this),
                Some(f) => assert!(
                    f.0 == this.0 && f.1 == this.1 && f.2 == this.2,
                    "row {row}: the structure depends on the seed"
                ),
            }
        }
        let (c0, a0, s0) = first.expect("at least one seed");
        if let Some(m) = wave_count(body) {
            assert_eq!(c0, [m as i64; 2], "row {row}");
        }
        cyc.extend(c0);
        anti.extend(a0);
        scalars.extend(s0);
        has.push(has_baroclinic_eddies(body));
        wavenumber_x.push(wavenumber(body).unwrap_or(f64::NAN));
        assert_eq!(wave_count(body).is_some(), *has.last().unwrap(), "row {row}");
    }
    Out::new()
        .f64("flow_scalars", &[n, 4], &scalars)
        .i64("cyclones_per_hemisphere", &[n, 2], &cyc)
        .i64("anticyclones_per_hemisphere", &[n, 2], &anti)
        .bool("has_eddies", &[n], &has)
        .f64("wavenumber_x", &[n], &wavenumber_x)
}

fn within(what: &str, v: f64, lo: f64, hi: f64) {
    assert!(v >= lo - 1e-9 && v <= hi + 1e-9, "{what}: {v} is outside the range [{lo}, {hi}] its draw comes from");
}

/// The random draws that make the recorded `vortices` of `body`, inverted from the formulas of Python's
/// `build_flow` (each draw is checked against the range the Python draws it from). The one number that is not
/// recoverable is the split of a hemisphere's first longitude between `lon0` and the first low's jitter, which
/// only enters their sum: `lon0` is the first low's longitude and its jitter 0, and what is checked is that
/// some split exists (the first longitude lies within `[-pi, pi]` widened by the largest jitter, and the other
/// lows' jitters differ from the first by at most twice the largest one).
fn plan_from_vortices(what: &str, body: &BodyInputs, vs: &[Vortex]) -> FlowPlan {
    let Some(m) = wave_count(body) else {
        assert!(vs.is_empty(), "{what}: a body without a storm track has no vortex");
        let empty = HemispherePlan { lon0: 0.0, lows: vec![], tropicals: vec![] };
        return FlowPlan { north: empty.clone(), south: empty };
    };
    let (phi, lr, track) = (hadley_edge_deg(body), deformation_radius_rad(body), storm_track_deg(body).to_radians());
    let spin = 1.0_f64.copysign(body.rotation_period_days.expect("a period"));
    let mf = m as f64;
    let mut at = 0;
    let mut hemispheres = Vec::new();
    for lat_sign in [1.0_f64, -1.0] {
        let hemi = lat_sign * spin;
        let mut lon0 = 0.0;
        let mut lows = Vec::new();
        for i in 0..m {
            let (low, high) = (&vs[at], &vs[at + 1]);
            at += 2;
            assert_eq!((low.kind, high.kind), (VortexKind::Cyclone, VortexKind::Anticyclone), "{what}: vortex order");
            assert!(hemisphere(low) == lat_sign && hemisphere(high) == lat_sign, "{what}: hemisphere of the pair {i}");
            if i == 0 {
                lon0 = low.lon;
                let reach = 0.3 * 2.0 * PI / mf;
                within(&format!("{what}: first low longitude"), lon0, -PI - reach, PI + reach);
            }
            let d = LowDraws {
                lon_jitter: (low.lon - lon0) * mf / (2.0 * PI) - i as f64,
                lat_offset: (lat_sign * low.lat - track).to_degrees() / phi,
                age: low.age,
                radius_factor: low.radius / (lr * (0.7 + 0.3 * low.age)),
                omega_factor: low.omega / (hemi * 3.0 * low.age),
                high_lon_factor: (high.lon - low.lon) / (PI / mf),
                high_lat_factor: (lat_sign * high.lat).to_degrees() / phi,
                high_radius_factor: high.radius / lr,
                high_omega_factor: high.omega / (-hemi * 1.2),
            };
            within(&format!("{what}: low {i} longitude jitter from the first"), d.lon_jitter, -0.6, 0.6);
            within(&format!("{what}: low {i} latitude offset"), d.lat_offset, -0.25, 0.25);
            within(&format!("{what}: low {i} age"), d.age, 0.25, 1.45);
            within(&format!("{what}: low {i} radius factor"), d.radius_factor, 0.8, 1.4);
            within(&format!("{what}: low {i} winding factor"), d.omega_factor, 0.8, 1.2);
            within(&format!("{what}: high {i} longitude factor"), d.high_lon_factor, 0.7, 1.3);
            within(&format!("{what}: high {i} latitude factor"), d.high_lat_factor, 0.9, 1.3);
            within(&format!("{what}: high {i} radius factor"), d.high_radius_factor, 1.3, 2.0);
            within(&format!("{what}: high {i} winding factor"), d.high_omega_factor, 0.7, 1.3);
            assert_eq!((low.ascent, high.ascent, high.age), (1.0, -0.45, 1.0), "{what}: constants of pair {i}");
            lows.push(d);
        }
        let mut tropicals = Vec::new();
        while at < vs.len() && vs[at].kind == VortexKind::Tropical && hemisphere(&vs[at]) == lat_sign {
            let t = &vs[at];
            at += 1;
            let d = TropicalDraws {
                lat_factor: (lat_sign * t.lat).to_degrees() / phi,
                lon: t.lon,
                radius_factor: t.radius / (0.35 * lr),
                omega_factor: t.omega / (hemi * 12.0),
            };
            let k = tropicals.len();
            within(&format!("{what}: tropical {k} latitude factor"), d.lat_factor, 0.35, 0.65);
            within(&format!("{what}: tropical {k} longitude"), d.lon, -PI, PI);
            within(&format!("{what}: tropical {k} radius factor"), d.radius_factor, 0.8, 1.2);
            within(&format!("{what}: tropical {k} winding factor"), d.omega_factor, 0.8, 1.2);
            assert_eq!((t.ascent, t.age), (1.6, 1.0), "{what}: constants of tropical {k}");
            tropicals.push(d);
        }
        hemispheres.push(HemispherePlan { lon0, lows, tropicals });
    }
    assert_eq!(at, vs.len(), "{what}: vortices left over after both hemispheres");
    let south = hemispheres.pop().unwrap();
    let north = hemispheres.pop().unwrap();
    FlowPlan { north, south }
}

// --- build_flow.<body>.seed<k>

/// The body and the seed of a `build_flow.<body>.seed<k>` or `build_flow.stats.<body>` case (the stat cases have
/// no seed of their own: 0).
fn body_and_seed(c: &Case) -> (BodyInputs, u64) {
    let seed = if c.has_param("seed") { c.p_i64("seed") as u64 } else { 0 };
    (body_of_vec(c.input("body"), c.p_str("cloud_regime")), seed)
}

/// The arrays of a seed case (`flow_scalars`, then `kind`, `lon`, `lat`, `radius`, `omega`, `ascent`, `age` per
/// vortex) for a flow.
fn flow_out(c: &Case, flow: &Flow) -> Out {
    let codes = c.param("kind_codes");
    let code_of = |k: VortexKind| codes[kind_name(k)].as_i64().expect("kind code") as i32;
    let n = flow.vortices.len();
    let col = |f: fn(&Vortex) -> f64| flow.vortices.iter().map(f).collect::<Vec<f64>>();
    Out::new()
        .f64("flow_scalars", &[4], &[flow.hadley_deg, flow.jet, flow.trades, flow.polar_easterlies])
        .i32("kind", &[n], &flow.vortices.iter().map(|v| code_of(v.kind)).collect::<Vec<_>>())
        .f64("lon", &[n], &col(|v| v.lon))
        .f64("lat", &[n], &col(|v| v.lat))
        .f64("radius", &[n], &col(|v| v.radius))
        .f64("omega", &[n], &col(|v| v.omega))
        .f64("ascent", &[n], &col(|v| v.ascent))
        .f64("age", &[n], &col(|v| v.age))
}

/// `build_flow.<body>.seed<k>` as the port computes it: its own population for the body and seed. For a body
/// with a storm track it differs from numpy's in every drawn number (an expected failure, see the module doc).
fn build_flow_seed(c: &Case) -> Out {
    let (body, seed) = body_and_seed(c);
    flow_out(c, &build_flow(&body, seed))
}

/// The flow numpy recorded in a seed case, rebuilt from its arrays. Used by the tests that compare the port
/// with the recorded numbers, never by the port's own output.
fn recorded_flow(c: &Case) -> Flow {
    let want = |name: &str| &c.expected.get(name).unwrap_or_else(|| panic!("case {}: no output {name}", c.id)).0;
    let codes = c.param("kind_codes");
    let code_of = |k: VortexKind| codes[kind_name(k)].as_i64().expect("kind code") as i32;
    let (kind, lon, lat, radius, omega, ascent, age) = (
        want("kind").as_i32(),
        want("lon").as_f64(),
        want("lat").as_f64(),
        want("radius").as_f64(),
        want("omega").as_f64(),
        want("ascent").as_f64(),
        want("age").as_f64(),
    );
    let vortices = (0..kind.len())
        .map(|i| {
            let k = [VortexKind::Cyclone, VortexKind::Anticyclone, VortexKind::Tropical]
                .into_iter()
                .find(|&k| code_of(k) == kind[i])
                .unwrap_or_else(|| panic!("case {}: kind code {}", c.id, kind[i]));
            Vortex { kind: k, lon: lon[i], lat: lat[i], radius: radius[i], omega: omega[i], ascent: ascent[i], age: age[i] }
        })
        .collect();
    let s = want("flow_scalars").as_f64();
    Flow { hadley_deg: s[0], jet: s[1], trades: s[2], polar_easterlies: s[3], vortices }
}

// --- build_flow.stats.<body>

/// The 13 drawn quantities of the stat cases, normalized to their uniform draws (the case doc's list).
const STAT_KEYS: [&str; 13] = [
    "cyclone.lat_offset",
    "cyclone.radius",
    "cyclone.omega",
    "cyclone.age",
    "cyclone.lon_gap",
    "anticyclone.lat",
    "anticyclone.radius",
    "anticyclone.omega",
    "anticyclone.lon_offset",
    "tropical.lat",
    "tropical.radius",
    "tropical.omega",
    "tropical.lon",
];

/// `rng.poisson(1.5)` of Python's `build_flow`: the mean number of tropical storms per hemisphere.
const TROPICAL_LAMBDA: f64 = 1.5;

/// The theoretical bound of every normalized quantity, from the uniform ranges of Python's `build_flow`
/// (the radius of a low is the product of two draws; the longitude gap of consecutive lows is the difference
/// of two jitters).
fn stat_bound(key: &str) -> (f64, f64) {
    match key {
        "cyclone.lat_offset" => (-0.25, 0.25),
        "cyclone.radius" => (0.8 * (0.7 + 0.3 * 0.25), 1.4 * (0.7 + 0.3 * 1.45)),
        "cyclone.omega" => (0.8, 1.2),
        "cyclone.age" => (0.25, 1.45),
        "cyclone.lon_gap" => (-0.6, 0.6),
        "anticyclone.lat" => (0.9, 1.3),
        "anticyclone.radius" => (1.3, 2.0),
        "anticyclone.omega" => (0.7, 1.3),
        "anticyclone.lon_offset" => (0.7, 1.3),
        "tropical.lat" => (0.35, 0.65),
        "tropical.radius" => (0.8, 1.2),
        "tropical.omega" => (0.8, 1.2),
        "tropical.lon" => (-PI, PI),
        other => panic!("no bound for {other}"),
    }
}

/// The outputs of a stat case that are numpy's sample (and so differ for another generator): the sample
/// ranges and the statistics of the tropical counts. Every other output is deterministic.
fn is_sample_output(name: &str) -> bool {
    name.ends_with(".range")
        || [
            "tropical.count.mean_by_hemisphere",
            "tropical.count.pooled_mean",
            "tropical.count.pooled_std",
            "tropical.count.histogram",
        ]
        .contains(&name)
}

/// The port's own population of a body over seeds `0..n_seeds`, summarized the way the stat cases are.
struct Population {
    /// The normalized samples of [`STAT_KEYS`].
    samples: BTreeMap<&'static str, Vec<f64>>,
    /// Cyclones and anticyclones per hemisphere, for every seed and hemisphere.
    cyclones: Vec<usize>,
    anticyclones: Vec<usize>,
    /// Tropical storms per hemisphere: `[north, south]` of every seed.
    tropicals: Vec<[usize; 2]>,
    /// `omega > 0` counts and totals by `[cyclone, anticyclone, tropical][north, south]`.
    positive: [[f64; 2]; 3],
    total: [[f64; 2]; 3],
}

fn population_of(body: &BodyInputs, n_seeds: usize) -> Population {
    let (phi, lr, track) = (hadley_edge_deg(body), deformation_radius_rad(body), storm_track_deg(body).to_radians());
    let spin = 1.0_f64.copysign(body.rotation_period_days.expect("a period"));
    let mut samples: BTreeMap<&'static str, Vec<f64>> = STAT_KEYS.iter().map(|&k| (k, Vec::new())).collect();
    let mut pop = Population {
        samples: BTreeMap::new(),
        cyclones: Vec::new(),
        anticyclones: Vec::new(),
        tropicals: Vec::new(),
        positive: [[0.0; 2]; 3],
        total: [[0.0; 2]; 3],
    };
    for seed in 0..n_seeds {
        let flow = build_flow(body, seed as u64);
        let mut tropical_here = [0usize; 2];
        for (hi, lat_sign) in [1.0_f64, -1.0].into_iter().enumerate() {
            let hemi = lat_sign * spin;
            let of = |k: VortexKind| {
                flow.vortices.iter().filter(|v| v.kind == k && hemisphere(v) == lat_sign).copied().collect::<Vec<_>>()
            };
            let (cyc, anti, trop) = (of(VortexKind::Cyclone), of(VortexKind::Anticyclone), of(VortexKind::Tropical));
            let m = cyc.len();
            pop.cyclones.push(m);
            pop.anticyclones.push(anti.len());
            tropical_here[hi] = trop.len();
            let mut push = |k: &str, v: f64| samples.get_mut(k).expect("stat key").push(v);
            for (i, v) in cyc.iter().enumerate() {
                push("cyclone.lat_offset", (v.lat.abs() - track).to_degrees() / phi);
                push("cyclone.radius", v.radius / lr);
                push("cyclone.omega", v.omega / (hemi * 3.0 * v.age));
                push("cyclone.age", v.age);
                if i + 1 < m {
                    push("cyclone.lon_gap", (cyc[i + 1].lon - v.lon) / (2.0 * PI / m as f64) - 1.0);
                }
            }
            for (cy, a) in cyc.iter().zip(&anti) {
                push("anticyclone.lon_offset", (a.lon - cy.lon) / (PI / m as f64));
            }
            for v in &anti {
                push("anticyclone.lat", v.lat.abs().to_degrees() / phi);
                push("anticyclone.radius", v.radius / lr);
                push("anticyclone.omega", v.omega / (-hemi * 1.2));
            }
            for v in &trop {
                push("tropical.lat", v.lat.abs().to_degrees() / phi);
                push("tropical.radius", v.radius / (0.35 * lr));
                push("tropical.omega", v.omega / (hemi * 12.0));
                push("tropical.lon", v.lon);
            }
            for (ki, group) in [&cyc, &anti, &trop].into_iter().enumerate() {
                for v in group {
                    pop.total[ki][hi] += 1.0;
                    pop.positive[ki][hi] += f64::from(v.omega > 0.0);
                }
            }
        }
        pop.tropicals.push(tropical_here);
    }
    pop.samples = samples;
    pop
}

fn min_max(v: &[f64]) -> (f64, f64) {
    (v.iter().copied().fold(f64::INFINITY, f64::min), v.iter().copied().fold(f64::NEG_INFINITY, f64::max))
}

/// `Poisson(lambda)` probabilities of 0..=8 and the tail `1 - sum` at index 9.
fn poisson_pmf(lambda: f64) -> Vec<f64> {
    let mut p: Vec<f64> = (0..9u32)
        .map(|k| (-lambda).exp() * lambda.powi(k as i32) / (1..=k).map(f64::from).product::<f64>())
        .collect();
    let tail = 1.0 - p.iter().sum::<f64>();
    p.push(tail);
    p
}

/// Every output of a stat case, computed from the port's own population.
fn stats_out(pop: &Population) -> Out {
    let mut out = Out::new();
    for k in STAT_KEYS {
        let (lo, hi) = stat_bound(k);
        let (min, max) = min_max(&pop.samples[k]);
        out = out.f64(&format!("{k}.bound"), &[2], &[lo, hi]).f64(&format!("{k}.range"), &[2], &[min, max]);
    }
    let pooled: Vec<f64> = pop.tropicals.iter().flatten().map(|&n| n as f64).collect();
    let n = pooled.len() as f64;
    let mean = pooled.iter().sum::<f64>() / n;
    let std = (pooled.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)).sqrt(); // ddof = 1
    let by_hemisphere: Vec<f64> =
        (0..2).map(|h| pop.tropicals.iter().map(|t| t[h] as f64).sum::<f64>() / pop.tropicals.len() as f64).collect();
    let mut histogram = [0i64; 10];
    for &x in &pooled {
        histogram[(x as usize).min(9)] += 1;
    }
    let fraction: Vec<f64> = (0..3)
        .flat_map(|ki| (0..2).map(move |hi| (ki, hi)))
        .map(|(ki, hi)| pop.positive[ki][hi] / pop.total[ki][hi].max(1.0))
        .collect();
    out.i64("count.cyclones_per_hemisphere", &[2], &[pop.cyclones[0] as i64; 2])
        .i64("count.anticyclones_per_hemisphere", &[2], &[pop.anticyclones[0] as i64; 2])
        .f64("tropical.count.mean_by_hemisphere", &[2], &by_hemisphere)
        .f64("tropical.count.pooled_mean", &[1], &[mean])
        .f64("tropical.count.pooled_std", &[1], &[std])
        .i64("tropical.count.n_samples", &[1], &[pooled.len() as i64])
        .f64("tropical.count.poisson_lambda", &[1], &[TROPICAL_LAMBDA])
        .i64("tropical.count.histogram", &[10], &histogram)
        .f64("tropical.count.poisson_pmf", &[10], &poisson_pmf(TROPICAL_LAMBDA))
        .f64("omega_positive_fraction", &[3, 2], &fraction)
}

/// `build_flow.stats.<body>` as the port computes it, from its own population (an expected failure: the
/// sample outputs are another generator's; see the module doc).
fn build_flow_stats(c: &Case) -> Out {
    let (body, _) = body_and_seed(c);
    stats_out(&population_of(&body, c.p_usize("n_seeds")))
}

// ---------------------------------------------------------------------------------------------------------
// Tests that bind the port where the cases cannot (they fail on their own, outside the runner's expected
// failures)

fn build_flow_cases(pick: impl Fn(&Case) -> bool) -> Vec<Case> {
    load_group("climate_flow").into_iter().filter(|c| c.function == "cloudgen.flow.build_flow" && pick(c)).collect()
}

fn seed_cases() -> Vec<Case> {
    build_flow_cases(|c| c.has_input("body") && !c.has_tag("stat"))
}

/// The checks the stat cases document for a port with another random generator, on the port's population of
/// every stat case body over the case's seeds (300): the deterministic outputs equal the case's under its own
/// tolerance; every drawn quantity lies within its theoretical bound and its sample range reaches within 10 per
/// cent of the bound width of each bound; the tropical count is Poisson(1.5) (pooled mean within 4 standard
/// errors of 1.5, chi-square of the histogram over the bins 0, 1, 2, 3, 4, >= 5 below the case's limit).
#[test]
fn build_flow_statistics_follow_the_case_docs() {
    let cases = build_flow_cases(|c| c.has_tag("stat"));
    assert_eq!(cases.len(), 6, "the six stat cases of the frozen data");
    for c in &cases {
        let (body, _) = body_and_seed(c);
        let n_seeds = c.p_usize("n_seeds");
        let slack = c.p_f64("range_slack_fraction_of_width");
        let sigma = c.p_f64("tropical_sigma_limit");
        let pop = population_of(&body, n_seeds);
        let out = stats_out(&pop);
        // Deterministic outputs: bounds, counts, n, lambda, pmf, sign fractions (exact 0 or 1).
        let mut problems = Vec::new();
        let mut compared = 0;
        for (name, (want, tol)) in &c.expected {
            if is_sample_output(name) {
                continue;
            }
            compared += 1;
            problems.extend(compare_array(name, &out.arrays[name], want, tol));
        }
        assert!(compared >= 13 + 5, "case {}: only {compared} deterministic outputs compared", c.id);
        assert!(problems.is_empty(), "case {}: {problems:?}", c.id);
        // (1) The number of lows and highs per hemisphere is exactly m, for every seed.
        let m = wave_count(&body).expect("a stat case body has a storm track");
        assert!(pop.cyclones.iter().all(|&k| k == m), "case {}: cyclones per hemisphere {:?}", c.id, pop.cyclones);
        assert!(pop.anticyclones.iter().all(|&k| k == m), "case {}: anticyclones per hemisphere", c.id);
        // (2) Hard bound, and the sample reaches within `slack` of the bound width of each bound.
        for k in STAT_KEYS {
            let (lo, hi) = stat_bound(k);
            let (min, max) = min_max(&pop.samples[k]);
            assert!(
                min >= lo - 1e-9 && max <= hi + 1e-9,
                "case {}: {k} range [{min}, {max}] leaves its bound [{lo}, {hi}]",
                c.id
            );
            let gap = slack * (hi - lo);
            assert!(
                min <= lo + gap && max >= hi - gap,
                "case {}: {k} sample range [{min}, {max}] misses its bound [{lo}, {hi}] by more than {slack} of the width",
                c.id
            );
        }
        // (3) Poisson(1.5) tropical count: pooled mean and each hemisphere's mean within the sigma limit,
        // chi-square of the histogram against the pmf over the case's bins.
        let pooled: Vec<f64> = pop.tropicals.iter().flatten().map(|&n| n as f64).collect();
        assert_eq!(pooled.len(), 2 * n_seeds);
        let mean_of = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        for (what, sample) in [
            ("pooled", pooled.clone()),
            ("north", pop.tropicals.iter().map(|t| t[0] as f64).collect()),
            ("south", pop.tropicals.iter().map(|t| t[1] as f64).collect()),
        ] {
            let (mean, se) = (mean_of(&sample), (TROPICAL_LAMBDA / sample.len() as f64).sqrt());
            assert!(
                (mean - TROPICAL_LAMBDA).abs() <= sigma * se,
                "case {}: {what} tropical mean {mean} is not within {sigma} standard errors ({se}) of {TROPICAL_LAMBDA}",
                c.id
            );
        }
        let top = c.param("tropical_chi2_bins").as_array().expect("bins").len() - 1; // the last bin is ">= top"
        let (mut observed, mut expected) = (vec![0.0_f64; top + 1], vec![0.0_f64; top + 1]);
        for &n in &pooled {
            observed[(n as usize).min(top)] += 1.0;
        }
        for (k, p) in poisson_pmf(TROPICAL_LAMBDA).iter().enumerate() {
            expected[k.min(top)] += p * pooled.len() as f64;
        }
        let chi2: f64 = observed.iter().zip(&expected).map(|(o, e)| (o - e) * (o - e) / e).sum();
        assert!(
            chi2 < c.p_f64("tropical_chi2_limit"),
            "case {}: tropical count chi-square {chi2} (observed {observed:?}, expected {expected:?})",
            c.id
        );
    }
}

/// The seed cases, against the port's own `build_flow(body, seed)`: what does not depend on the draws must
/// equal numpy's (the flow scalars under the case's tolerance; the cyclone and anticyclone of every pair,
/// hemisphere by hemisphere, in order; the sign of every winding, which is hemisphere times spin; the constant
/// ascent weight of each kind and the age of a high or a tropical storm), for the port's population and for
/// numpy's recorded one alike.
#[test]
fn build_flow_seeds_agree_with_numpy_where_the_draws_do_not_decide() {
    let cases = seed_cases();
    assert_eq!(cases.len(), 12, "the twelve seed cases of the frozen data");
    for c in &cases {
        let (body, seed) = body_and_seed(c);
        let (own, numpy) = (build_flow(&body, seed), recorded_flow(c));
        let (want, tol) = &c.expected["flow_scalars"];
        let got = Array::from_f64(&[4], &[own.hadley_deg, own.jet, own.trades, own.polar_easterlies]);
        let problems = compare_array("flow_scalars", &got, want, tol);
        assert!(problems.is_empty(), "case {}: {problems:?}", c.id);
        let signature = |f: &Flow| {
            f.vortices
                .iter()
                .filter(|v| v.kind != VortexKind::Tropical)
                .map(|v| (v.kind, hemisphere(v) > 0.0, v.omega > 0.0))
                .collect::<Vec<_>>()
        };
        assert_eq!(signature(&own), signature(&numpy), "case {}: lows and highs, hemispheres and windings", c.id);
        if let Some(m) = wave_count(&body) {
            // A body with a storm track has m lows and m highs in each hemisphere, which the signatures just
            // proved equal to numpy's; pin the count itself against the formula.
            assert_eq!(signature(&own).len(), 4 * m, "case {}", c.id);
        } else {
            assert!(own.vortices.is_empty() && numpy.vortices.is_empty(), "case {}", c.id);
        }
        let spin = 1.0_f64.copysign(body.rotation_period_days.expect("a period"));
        for (who, flow) in [("port", &own), ("numpy", &numpy)] {
            for v in &flow.vortices {
                let ascent = match v.kind {
                    VortexKind::Cyclone => 1.0,
                    VortexKind::Anticyclone => -0.45,
                    VortexKind::Tropical => 1.6,
                };
                assert_eq!(v.ascent, ascent, "case {}: {who} {v:?}", c.id);
                assert!(v.kind == VortexKind::Cyclone || v.age == 1.0, "case {}: {who} {v:?}", c.id);
                if v.kind == VortexKind::Tropical {
                    // Poisson-many, but each one winds with the sign of hemisphere times spin.
                    assert_eq!(v.omega > 0.0, hemisphere(v) * spin > 0.0, "case {}: {who} {v:?}", c.id);
                }
            }
        }
    }
}

/// numpy's recorded vortices, inverted into the draws that made them ([`plan_from_vortices`]: each draw must
/// lie in the range Python draws it from) and run forward through the port's deterministic half
/// ([`build_flow_from_plan`]): every recorded array must come back under the case's own tolerance. The
/// inversion is written here from the Python source, so a constant, a sign or an index of the port that
/// differs from Python's breaks the round trip, and a constant of this test that differs from Python's puts a
/// recorded draw out of its range. (It does not test the port's sampler: the unit tests of `flow.rs` do.)
#[test]
fn build_flow_replays_numpys_recorded_vortices() {
    let cases = seed_cases();
    assert_eq!(cases.len(), 12);
    let mut vortices = 0;
    for c in &cases {
        let (body, _) = body_and_seed(c);
        let numpy = recorded_flow(c);
        vortices += numpy.vortices.len();
        let plan = plan_from_vortices(&c.id, &body, &numpy.vortices);
        let flow = build_flow_from_plan(&body, &plan);
        let problems = cloud_map_generator_golden::compare(c, &flow_out(c, &flow));
        assert!(problems.is_empty(), "case {}: {problems:?}", c.id);
    }
    assert!(vortices > 100, "the replay covered only {vortices} recorded vortices");
}

// ---------------------------------------------------------------------------------------------------------

fn port(c: &Case) -> Option<Out> {
    Some(match c.function.as_str() {
        "cloudgen.climate.BodyInputs" => body_inputs(c),
        "cloudgen.climate.constants" => constants(c),
        "cloudgen.climate.convective_tendency" => by_dtype!(c, "lat_deg", convective),
        "cloudgen.climate.target_zonal_coverage" => by_dtype!(c, "lat_deg", target),
        "cloudgen.climate._gauss" => gauss_sweep(c),
        "cloudgen.climate._smoothstep" => smoothstep_sweep(c),
        "cloudgen.climate.map_width_for_texel" => map_width(c),
        "cloudgen.climate.hadley_edge_raw_deg" => {
            let v = per_body(c, hadley_edge_raw_deg);
            Out::new().f64("raw_deg", &[v.len()], &v)
        }
        "cloudgen.climate.hadley_edge_deg" => {
            let v = per_body(c, hadley_edge_deg);
            Out::new().f64("hadley_deg", &[v.len()], &v)
        }
        "cloudgen.climate.storm_track_deg" => {
            let v = per_body(c, storm_track_deg);
            Out::new().f64("track_deg", &[v.len()], &v)
        }
        "cloudgen.climate.deformation_radius_rad" => {
            let v = per_body(c, deformation_radius_rad);
            Out::new().f64("radius_rad", &[v.len()], &v)
        }
        "cloudgen.climate.has_baroclinic_eddies" => {
            let v = per_body(c, has_baroclinic_eddies);
            Out::new().bool("has_eddies", &[v.len()], &v)
        }
        "cloudgen.climate.is_bakeable" => {
            let v: Vec<bool> = if c.has_input("rotation_period_days") {
                per_body(c, is_bakeable)
            } else {
                // One body, only the regime varies.
                c.param("regimes")
                    .as_array()
                    .expect("regimes")
                    .iter()
                    .map(|r| is_bakeable(&EARTH.clone().with_regime(r.as_str().expect("regime").to_string())))
                    .collect()
            };
            Out::new().bool("bakeable", &[v.len()], &v)
        }
        "cloudgen.flow.Flow" => flow_defaults(c),
        "cloudgen.flow.Vortex.center" => vortex_center(c),
        "cloudgen.flow.Flow.zonal_rate" => by_dtype!(c, "lat", zonal_rate),
        "cloudgen.flow.Flow.velocity" => by_dtype!(c, "p", velocity),
        "cloudgen.flow.Flow.backtrace" => by_dtype!(c, "p", backtrace),
        "cloudgen.flow.Flow.ascent" => by_dtype!(c, "p", ascent),
        "cloudgen.flow.Flow._comma_ascent" => by_dtype!(c, "p", comma_ascent),
        "cloudgen.flow.build_flow" => {
            if c.has_input("rotation_period_days") {
                build_flow_structure(c)
            } else if c.has_tag("stat") {
                build_flow_stats(c)
            } else {
                build_flow_seed(c)
            }
        }
        _ => return None,
    })
}

#[test]
fn climate_flow_cases() {
    let cfg = Config {
        group: "climate_flow",
        prefixes: &["cloudgen.climate.", "cloudgen.flow."],
        expected_failures: EXPECTED_FAILURES,
    };
    run(&cfg, port).assert_clean();
}

// ---------------------------------------------------------------------------------------------------------
// Pins for the gaps no case reaches. Each is also a unit test of its module (`climate.rs`, `flow.rs`), where
// the oracle values come from; they are repeated here, on the public API, so that this file alone binds them.

/// `map_width_for_texel` saturates to the cap only for an exponent of 63 or more: `2^62` is a real width and
/// `min(2**63 - 1, 2**62)` is `2**62`. Oracle: Python's `map_width_for_texel(r, 2 pi, 2**63 - 1)`, where `w` is
/// `r` exactly for a power of two `r`.
#[test]
fn map_width_saturates_only_past_two_to_the_62() {
    let two_pi = 2.0 * PI;
    let at = |r: f64, cap: i64| map_width_for_texel(r, two_pi, cap);
    let p = |k: i32| 2f64.powi(k);
    assert_eq!(at(p(60), i64::MAX), 1 << 60);
    assert_eq!(at(p(61), i64::MAX), 1 << 61);
    assert_eq!(at(1.5 * p(61), i64::MAX), 1 << 62);
    assert_eq!(at(p(62), i64::MAX), 1 << 62);
    assert_eq!(at(1.5 * p(62), i64::MAX), i64::MAX);
    assert_eq!(at(p(63), i64::MAX), i64::MAX);
    assert_eq!(at(1.5 * p(62), 1 << 62), 1 << 62);
    assert_eq!(map_width_for_texel(p(62) * 3.0 / two_pi, 3.0, i64::MAX), 1 << 62);
}

/// `m = max(2, round(x))` of `build_flow` is Python's round-half-to-even: no body of the cases has an exact
/// half-integer wavenumber, so the rounding is pinned on the function itself (`round` would give 5 for 4.5).
#[test]
fn the_wave_count_rounds_half_to_even_with_a_floor_of_two() {
    let m = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.49, 6.51, 20.5].map(waves_from_wavenumber);
    assert_eq!(m, [2, 2, 2, 4, 4, 6, 6, 7, 20]);
    assert_eq!(waves_from_wavenumber(-3.0), 2);
}

/// The sampler of the vortex population, against the `rng.uniform(lo, hi)` and `rng.poisson(1.5)` calls of the
/// Python `build_flow` (the ranges are copied from that source): every field lies in its range and reaches
/// both ends of it (within 0.5 per cent of the width), and the tropical count has the Poisson(1.5) mean. The
/// stat cases accept a sample range within 10 per cent of each bound and a pooled mean within 0.2 of 1.5, which
/// cannot tell an upper age bound of 1.40 from 1.45 or a Poisson mean of 1.4 from 1.5; this test can.
#[test]
fn the_flow_plan_draws_the_ranges_and_the_poisson_mean_of_the_python() {
    use cloud_map_generator_core::rng::Rng;
    let ranges: [(&str, f64, f64); 14] = [
        ("lon0", -PI, PI),
        ("low.lon_jitter", -0.3, 0.3),
        ("low.lat_offset", -0.25, 0.25),
        ("low.age", 0.25, 1.45),
        ("low.radius_factor", 0.8, 1.4),
        ("low.omega_factor", 0.8, 1.2),
        ("high.lon_factor", 0.7, 1.3),
        ("high.lat_factor", 0.9, 1.3),
        ("high.radius_factor", 1.3, 2.0),
        ("high.omega_factor", 0.7, 1.3),
        ("tropical.lat_factor", 0.35, 0.65),
        ("tropical.lon", -PI, PI),
        ("tropical.radius_factor", 0.8, 1.2),
        ("tropical.omega_factor", 0.8, 1.2),
    ];
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); ranges.len()];
    let mut counts: Vec<f64> = Vec::new();
    let mut rng = Rng::new(31);
    for _ in 0..6_000 {
        let plan = FlowPlan::draw(2, &mut rng);
        for h in [&plan.north, &plan.south] {
            samples[0].push(h.lon0);
            for l in &h.lows {
                let row = [
                    l.lon_jitter,
                    l.lat_offset,
                    l.age,
                    l.radius_factor,
                    l.omega_factor,
                    l.high_lon_factor,
                    l.high_lat_factor,
                    l.high_radius_factor,
                    l.high_omega_factor,
                ];
                for (k, v) in row.into_iter().enumerate() {
                    samples[1 + k].push(v);
                }
            }
            counts.push(h.tropicals.len() as f64);
            for t in &h.tropicals {
                for (k, v) in [t.lat_factor, t.lon, t.radius_factor, t.omega_factor].into_iter().enumerate() {
                    samples[10 + k].push(v);
                }
            }
        }
    }
    for ((name, lo, hi), s) in ranges.iter().zip(&samples) {
        let (min, max) = (
            s.iter().copied().fold(f64::INFINITY, f64::min),
            s.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
        assert!(s.len() >= 10_000 && min >= *lo && max <= *hi, "{name}: [{min}, {max}] of {} leaves [{lo}, {hi}]", s.len());
        let gap = 0.005 * (hi - lo);
        assert!(min - lo < gap && hi - max < gap, "{name}: the sample [{min}, {max}] misses an end of [{lo}, {hi}]");
    }
    let mean = counts.iter().sum::<f64>() / counts.len() as f64;
    assert!((mean - 1.5).abs() < 5.0 * (1.5 / counts.len() as f64).sqrt(), "tropical count mean {mean}");
}
