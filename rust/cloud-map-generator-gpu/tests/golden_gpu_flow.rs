//! Golden cases of cloudgen/flow.py run through the GPU kernels (group climate_flow, functions
//! cloudgen.flow.Flow.velocity, .backtrace, .ascent, ._comma_ascent and .zonal_rate: the hand-built Flow cases).
//!
//! 53 of the 63 cases pass at their own frozen tolerance, the back-traces included (the 48 steps are
//! integrated with the compensated f32 position of `flow::Precision::Compensated`, see
//! `tests/gpu_flow_accuracy.rs`). The other 10 freeze a FLOAT64 oracle (rtol 1e-9, atol 1e-12), which float32
//! arithmetic cannot meet; they are EXPECTED_FAILURES with the measured error, and
//! `float64_oracle_cases_hold_at_the_float32_tier` binds the GPU result on them at the tolerance the same
//! function's float32 cases carry in the frozen data, so they are not left unchecked.
//!
//! No case needed a widened tolerance. Without an adapter the tests print why and skip; with `REQUIRE_GPU` set
//! they fail instead.

use cloud_map_generator_golden::{compare_array, load_group, run, Array, Case, Config, Out, Tol};
use cloud_map_generator_core::flow::{Flow, Vortex, VortexKind};
use cloud_map_generator_gpu::flow::{FlowKernels, GpuFlow};
use cloud_map_generator_gpu::GpuContext;

/// The adapter, or `None` (with the reason printed) when there is none; with `REQUIRE_GPU` set a missing
/// adapter fails the test instead.
fn gpu_or_skip() -> Option<GpuContext> {
    let g = GpuContext::new();
    match &g {
        Some(g) => eprintln!("adapter: {} ({:?})", g.adapter_info.name, g.adapter_info.backend),
        None => {
            assert!(std::env::var_os("REQUIRE_GPU").is_none(), "REQUIRE_GPU is set and there is no GPU adapter");
            eprintln!("no GPU adapter: the GPU golden cases are skipped (set REQUIRE_GPU=1 to fail instead)");
        }
    }
    g
}

/// Points of any float dtype as f32: the GPU cannot represent a float64 input.
fn points_f32(a: &Array) -> Vec<[f32; 3]> {
    assert!(a.dtype.is_float(), "points are floats");
    a.to_f64().chunks_exact(3).map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect()
}

fn lats_f32(a: &Array) -> Vec<f32> {
    a.to_f64().into_iter().map(|v| v as f32).collect()
}

fn flat(v: &[[f32; 3]]) -> Vec<f32> {
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

/// A vortex of `params` (a JSON object, read through `get`, so that this file does not need serde_json): a
/// missing kind is a cyclone (the comma cases), a missing age the default of 1.
macro_rules! vortex_of {
    ($v:expr) => {{
        let v = $v;
        let num = |k: &str| v[k].as_f64().unwrap_or_else(|| panic!("vortex field {k} is not a number in {v}"));
        Vortex {
            kind: v.get("kind").and_then(|k| k.as_str()).map_or(VortexKind::Cyclone, kind_of),
            lon: num("lon"),
            lat: num("lat"),
            radius: num("radius"),
            omega: num("omega"),
            ascent: num("ascent"),
            age: v.get("age").and_then(|a| a.as_f64()).unwrap_or(1.0),
        }
    }};
}

/// The hand-built flow of a case: `hadley_deg`, the three shear scalars and the vortex list of its params.
fn flow_of(c: &Case) -> Flow {
    let vortices = c.param("vortices").as_array().unwrap_or_else(|| panic!("case {}: vortices is not a list", c.id));
    Flow {
        hadley_deg: c.p_f64("hadley_deg"),
        vortices: vortices.iter().map(|v| vortex_of!(v)).collect(),
        jet: c.p_f64("jet"),
        trades: c.p_f64("trades"),
        polar_easterlies: c.p_f64("polar_easterlies"),
    }
}

struct Gpu {
    ctx: GpuContext,
    kernels: FlowKernels,
}

impl Gpu {
    fn new(ctx: GpuContext) -> Gpu {
        Gpu { ctx, kernels: FlowKernels::new() }
    }

    fn upload(&self, flow: &Flow) -> GpuFlow {
        GpuFlow::new(&self.ctx, flow)
    }

    fn zonal_rate(&self, c: &Case) -> Out {
        let lats = lats_f32(c.input("lat"));
        if c.has_param("flows") {
            // One row per (hadley, jet, trades, polar easterlies) of params.flows.
            let flows = c.param("flows").as_array().expect("flows");
            let mut rate = Vec::new();
            for row in flows {
                let r: Vec<f64> = row.as_array().expect("flow row").iter().map(|v| v.as_f64().expect("number")).collect();
                let spec = Flow { hadley_deg: r[0], jet: r[1], trades: r[2], polar_easterlies: r[3], vortices: vec![] };
                rate.extend(self.kernels.zonal_rate(&self.ctx, &self.upload(&spec), &lats));
            }
            Out::new().f32("rate", &[flows.len(), lats.len()], &rate)
        } else {
            let rate = self.kernels.zonal_rate(&self.ctx, &self.upload(&flow_of(c)), &lats);
            Out::new().f32("rate", &[lats.len()], &rate)
        }
    }

    fn velocity(&self, c: &Case) -> Out {
        let p = c.input("p");
        let v = self.kernels.velocity(&self.ctx, &self.upload(&flow_of(c)), &points_f32(p));
        Out::new().f32("v", &p.shape, &flat(&v))
    }

    fn backtrace(&self, c: &Case) -> Out {
        let p = c.input("p");
        let (time, steps) = (c.p_f64("time"), c.p_usize("steps"));
        let q = self.kernels.backtrace(&self.ctx, &self.upload(&flow_of(c)), &points_f32(p), time, steps);
        Out::new().f32("q", &p.shape, &flat(&q))
    }

    fn ascent(&self, c: &Case) -> Out {
        let p = c.input("p");
        let a = self.kernels.ascent(&self.ctx, &self.upload(&flow_of(c)), &points_f32(p));
        Out::new().f32("a", &[a.len()], &a)
    }

    /// `Flow._comma_ascent(vortex, p)`: the comma of one vortex, unweighted, as the ascent kernel of a flow of
    /// that cyclone with weight 1.
    fn comma_ascent(&self, c: &Case) -> Out {
        let p = points_f32(c.input("p"));
        let vx = vortex_of!(c.param("vortex"));
        let comma = |vx: Vortex| {
            let one = Vortex { kind: VortexKind::Cyclone, ascent: 1.0, ..vx };
            let flow = Flow { vortices: vec![one], ..Flow::new(30.0) };
            self.kernels.ascent(&self.ctx, &self.upload(&flow), &p)
        };
        if c.has_input("ages") {
            // Output row i is ages[i]; the vortex of the params has no age of its own.
            let ages = c.input("ages").as_f64();
            let g: Vec<f32> = ages.iter().flat_map(|&age| comma(Vortex { age, ..vx })).collect();
            Out::new().f32("g", &[ages.len(), p.len()], &g)
        } else {
            assert!(c.param("vortex").get("age").is_some(), "case {}: the vortex has no age", c.id);
            Out::new().f32("g", &[p.len()], &comma(vx))
        }
    }

    fn port(&self, c: &Case) -> Option<Out> {
        Some(match c.function.as_str() {
            "cloudgen.flow.Flow.zonal_rate" => self.zonal_rate(c),
            "cloudgen.flow.Flow.velocity" => self.velocity(c),
            "cloudgen.flow.Flow.backtrace" => self.backtrace(c),
            "cloudgen.flow.Flow.ascent" => self.ascent(c),
            "cloudgen.flow.Flow._comma_ascent" => self.comma_ascent(c),
            _ => return None,
        })
    }
}

const PREFIXES: &[&str] = &[
    "cloudgen.flow.Flow.velocity",
    "cloudgen.flow.Flow.backtrace",
    "cloudgen.flow.Flow.ascent",
    "cloudgen.flow.Flow._comma_ascent",
    "cloudgen.flow.Flow.zonal_rate",
];

/// Cases whose frozen oracle is float64 (rtol 1e-9, atol 1e-12): the GPU is float32 (and takes float64 inputs
/// rounded to float32), so they cannot pass at that tolerance. Each reason quotes the largest |GPU - frozen| of
/// the run on the RTX 5080 (Vulkan). None of these is a bug: `float64_oracle_cases_hold_at_the_float32_tier`
/// checks them at the float32 tier of their function.
const EXPECTED_FAILURES: &[(&str, &str)] = &[
    ("velocity.earth.float64", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 2.3e-7 (f64 inputs rounded to f32 too)"),
    ("velocity.non_unit_input", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 5.9e-8 (f64 inputs rounded to f32 too)"),
    ("velocity.still_earth", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.9e-7 (f64 inputs rounded to f32 too)"),
    ("velocity.stress.float64", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 2.2e-7 (f64 inputs rounded to f32 too)"),
    ("velocity.zonal_only", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.8e-7 (f64 inputs rounded to f32 too)"),
    ("zonal_rate.default", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 2.0e-7 (f64 latitudes rounded to f32 too)"),
    ("zonal_rate.hadley_and_shear", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 3.6e-7 (f64 latitudes rounded to f32 too)"),
    ("zonal_rate.out_of_range_lat", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.0e-7 (f64 latitudes rounded to f32 too)"),
    ("comma_ascent.age_sweep", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.5e-6 (f64 inputs rounded to f32; the occluded ages subtract a second Gaussian)"),
    ("comma_ascent.south_float64", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.6e-7 (f64 inputs rounded to f32 too)"),
];

#[test]
fn flow_gpu_cases() {
    let Some(ctx) = gpu_or_skip() else { return };
    let gpu = Gpu::new(ctx);
    let cfg = Config { group: "climate_flow", prefixes: PREFIXES, expected_failures: EXPECTED_FAILURES };
    let report = run(&cfg, |c| gpu.port(c));
    assert!(report.widened.is_empty(), "no tolerance is widened in this file");
    report.assert_clean();
}

/// The tolerance a function's float32 cases carry in the frozen data: (function, sibling float32 case, output).
const F32_TIER: &[(&str, &str, &str)] = &[
    ("cloudgen.flow.Flow.velocity", "velocity.earth.float32", "v"),
    ("cloudgen.flow.Flow.zonal_rate", "zonal_rate.float32_lat", "rate"),
    ("cloudgen.flow.Flow._comma_ascent", "comma_ascent.float32", "g"),
];

/// The 10 float64-oracle cases, on the GPU, against the same frozen outputs at the tolerance the oracle itself
/// froze for float32 arithmetic of that function (velocity rtol 1e-4 / atol 1e-5, zonal rate and comma ascent
/// rtol 2e-5 / atol 2e-6): the GPU error is two to three orders of magnitude inside it for the velocity and
/// the zonal rate, and 1.5e-6 of 2e-6 for the comma ascent of the age sweep.
#[test]
fn float64_oracle_cases_hold_at_the_float32_tier() {
    let Some(ctx) = gpu_or_skip() else { return };
    let gpu = Gpu::new(ctx);
    let cases = load_group("climate_flow");
    let by_id = |id: &str| cases.iter().find(|c| c.id == id).unwrap_or_else(|| panic!("no golden case {id}"));
    for (id, _) in EXPECTED_FAILURES {
        let c = by_id(id);
        let (_, sibling, output) = F32_TIER
            .iter()
            .find(|(function, _, _)| *function == c.function)
            .unwrap_or_else(|| panic!("case {id}: no float32 tier known for {}", c.function));
        let tier = by_id(sibling).expected[*output].1.clone();
        assert!(matches!(tier, Tol::Close { .. }), "the float32 tier of {} is a close tolerance", c.function);
        let out = gpu.port(c).expect("the case is ported");
        let (want, frozen) = &c.expected[*output];
        // The case's own tolerance is the float64 one, tighter than the tier it is held to here.
        match (frozen, &tier) {
            (Tol::Close { rtol: own, .. }, Tol::Close { rtol: loose, .. }) => assert!(own < loose, "case {id}"),
            _ => panic!("case {id}: close tolerances expected"),
        }
        let problems = compare_array(output, &out.arrays[*output], want, &tier);
        assert!(problems.is_empty(), "case {id} at the float32 tier {tier:?}: {problems:?}");
    }
}
