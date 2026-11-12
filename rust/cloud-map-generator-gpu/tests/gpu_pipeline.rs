//! The GPU pipeline (`cloud_map_generator_gpu::pipeline`) against the golden cases and against the CPU port.
//!
//! * `golden_generate_cases_through_the_gpu_pipeline`: the 26 scenarios of `cloudgen.generate.generate` (group
//!   pipeline, injected plans at 2 to 256 texels wide), run through `generate_gpu` with the very plans the CPU
//!   port is fed, and the OUTPUTS compared with the frozen ones: coverage, type, high_share (the fields of the
//!   pipeline stay on the GPU, so no intermediate can be compared). The tolerance TIERS are below.
//! * `cpu_and_gpu_make_the_same_map_at_1024x512`: the same plans through `generate_with_plans` and
//!   `generate_gpu` at 1024 x 512 for two seeds: the maximum and the mean absolute difference and the fraction
//!   of texels that differ by more than 0.01.
//! * `bands_do_not_change_the_map` and `the_seeded_gpu_map_is_calibrated`: the banding (small bands forced) and
//!   the counter-based sub-texel noise.
//!
//! Without a GPU adapter the tests print why and skip, unless REQUIRE_GPU is set (then they fail).
//!
//! # The tiers
//!
//! The frozen tolerances were measured for float32 numpy against a float64 shadow pipeline. The GPU is float32
//! all the way (its transcendental functions, its sums in a different order, a fused multiply-add), and the
//! pipeline is CHAOTIC at a threshold: the coverage of a texel is a steep function of `s - t` (the shield's ramp
//! is 0.35 of a standard deviation wide) and `t` comes out of a bisection on a row mean, so a 1e-6 change of a
//! field moves a threshold by about 1e-6 and a texel on the edge of a shield by 1e-5; only a decision that is
//! discrete (a regime weight crossing a clip, a back-trace sample landing on the other side of a Worley cell
//! border) changes a texel by more. The cases are therefore judged on three numbers, each against the tier of
//! its map (see `tier`):
//!
//! * `max`: the largest |GPU - frozen| over the map;
//! * `frac`: the fraction of texels that differ by more than 0.01;
//! * `mean`: the mean |GPU - frozen|.
//!
//! A case that exceeds its tier is listed in `EXPECTED_OUTLIERS` with the measured number and the cause; the
//! runner fails on a case that exceeds its tier unlisted, and on a listed case that now fits its tier.

use cloud_map_generator_core::climate::EARTH;
use cloud_map_generator_core::generate::{generate_with_plans, GenParams, GeneratePlans};
use cloud_map_generator_core::surface::Surface;
use cloud_map_generator_golden::{load_group, Case};
use cloud_map_generator_gpu::pipeline::{generate_gpu, generate_gpu_seeded, GpuPipeline};
use cloud_map_generator_gpu::GpuContext;

mod support_pipeline;
use support_pipeline::{body_of, gen_params_of, plans_of, surface_of};

fn gpu_or_skip(test: &str) -> Option<GpuContext> {
    match GpuContext::new() {
        Some(g) => {
            println!("{test}: adapter {} ({:?})", g.adapter_info.name, g.adapter_info.backend);
            Some(g)
        }
        None if std::env::var_os("REQUIRE_GPU").is_some() => panic!("{test}: no GPU adapter and REQUIRE_GPU is set"),
        None => {
            println!("{test}: no GPU adapter (wgpu found none): skipped; set REQUIRE_GPU to fail instead");
            None
        }
    }
}

/// How far two maps are apart.
#[derive(Clone, Copy, Debug)]
struct Diff {
    max: f64,
    mean: f64,
    /// Fraction of texels with |a - b| > 0.01.
    frac: f64,
}

fn diff(a: &[f32], b: &[f32]) -> Diff {
    assert_eq!(a.len(), b.len());
    let (mut max, mut sum, mut over) = (0.0_f64, 0.0_f64, 0usize);
    for (x, y) in a.iter().zip(b) {
        let d = f64::from(*x - *y).abs();
        // A NaN on either side is infinitely far.
        let d = if d.is_nan() { f64::INFINITY } else { d };
        max = max.max(d);
        sum += d;
        if d > 0.01 {
            over += 1;
        }
    }
    let n = a.len().max(1) as f64;
    Diff { max, mean: sum / n, frac: over as f64 / n }
}

/// The tier of a map: what a float32 pipeline may differ from the frozen map by. MEASURED on the 26 cases (the
/// table the test prints is the evidence): the largest difference of any map is 1.3e-4, the largest mean 2.4e-6,
/// and no texel of any case is off by 0.01. The tier is about four times that: a wrong constant, a missing term
/// or a stage on the wrong side of a clip moves the mean of a map by 1e-3 and more, so it cannot hide in it.
fn tier(_texels: usize) -> Diff {
    Diff { max: 5e-4, mean: 1e-5, frac: 5e-4 }
}

/// Cases that do not fit their tier, with the measured cause. Empty when every case fits.
const EXPECTED_OUTLIERS: &[(&str, &str)] = &[];

fn generate_cases() -> Vec<Case> {
    load_group("pipeline")
        .into_iter()
        .filter(|c| c.function == "cloudgen.generate.generate" && c.id != "generate.rejects_non_water")
        .collect()
}

#[test]
fn golden_generate_cases_through_the_gpu_pipeline() {
    let Some(gpu) = gpu_or_skip("golden_generate_cases_through_the_gpu_pipeline") else { return };
    let pipeline = GpuPipeline::new(&gpu);
    let cases = generate_cases();
    assert!(cases.len() >= 20, "the pipeline group has {} generate cases", cases.len());
    println!("{:<38} {:>6} {:<10} {:>10} {:>10} {:>9}", "case", "width", "field", "max", "mean", "frac>.01");
    let mut failures: Vec<String> = Vec::new();
    let mut stale: Vec<String> = Vec::new();
    for c in &cases {
        let body = body_of(c.param("body"));
        let (seed, width) = (c.p_i64("seed") as u64, c.p_usize("width"));
        let prm = gen_params_of(c.param("gen_params"));
        let surface = surface_of(c);
        let plans = plans_of(c, &body, seed, width, &prm);
        let extras = c.p_bool("extras");
        let (out, _) = pipeline
            .generate(&body, &plans, width, &prm, surface.as_ref(), extras, &mut |_, _| {})
            .unwrap_or_else(|e| panic!("case {}: {e}", c.id));
        let mut worst: Option<String> = None;
        for (name, got) in [("coverage", Some(&out.coverage)), ("type", out.extras.as_ref().map(|e| &e.kind)), ("high_share", out.extras.as_ref().map(|e| &e.high_share))] {
            let Some((want, _)) = c.expected.get(name) else { continue };
            let got = got.unwrap_or_else(|| panic!("case {}: the GPU made no {name}", c.id));
            let want32: Vec<f32> = want.to_f64().iter().map(|&v| v as f32).collect();
            let d = diff(got, &want32);
            println!("{:<38} {:>6} {:<10} {:>10.3e} {:>10.3e} {:>9.5}", c.id, width, name, d.max, d.mean, d.frac);
            let t = tier(out.coverage.len());
            if d.max > t.max || d.mean > t.mean || d.frac > t.frac {
                worst = Some(format!("{name}: max {:.3e} mean {:.3e} frac {:.5} (tier: max {} mean {} frac {})", d.max, d.mean, d.frac, t.max, t.mean, t.frac));
            }
        }
        let listed = EXPECTED_OUTLIERS.iter().any(|(id, _)| *id == c.id);
        match (&worst, listed) {
            (Some(w), false) => failures.push(format!("{}: {w}", c.id)),
            (None, true) => stale.push(c.id.clone()),
            _ => {}
        }
    }
    assert!(stale.is_empty(), "listed outliers that now fit their tier: {stale:?}");
    assert!(failures.is_empty(), "cases outside their tier:\n{}", failures.join("\n"));
}

fn print_diff(what: &str, d: Diff) {
    println!("  {what:<11} max |GPU - CPU| {:>10.3e}   mean {:>10.3e}   texels over 0.01: {:.6}", d.max, d.mean, d.frac);
}

/// The CPU port and the GPU pipeline on the same plans at 1024 x 512.
#[test]
fn cpu_and_gpu_make_the_same_map_at_1024x512() {
    let Some(gpu) = gpu_or_skip("cpu_and_gpu_make_the_same_map_at_1024x512") else { return };
    let pipeline = GpuPipeline::new(&gpu);
    let width = 1024;
    let params = GenParams::default();
    for seed in [1_u64, 2] {
        let plans = GeneratePlans::from_seed(&EARTH, seed, width, &params);
        let cpu = generate_with_plans(&EARTH, &plans, width, &params, None, true, false).expect("Earth is bakeable");
        let (gpu_out, timings) = pipeline.generate(&EARTH, &plans, width, &params, None, true, &mut |_, _| {}).expect("Earth is bakeable");
        println!("seed {seed}: CPU stages of the GPU run {:.3} s, the rest {:.3} s", timings.cpu_seconds, timings.gpu_seconds);
        let (ce, ge) = (cpu.extras.as_ref().expect("extras"), gpu_out.extras.as_ref().expect("extras"));
        let cov = diff(&gpu_out.coverage, &cpu.coverage);
        let kind = diff(&ge.kind, &ce.kind);
        let high = diff(&ge.high_share, &ce.high_share);
        print_diff("coverage", cov);
        print_diff("type", kind);
        print_diff("high_share", high);
        assert_eq!(ge.wind, ce.wind, "the wind is the CPU's");
        assert_eq!(gpu_out.meta, {
            let mut m = cpu.meta.clone();
            m.seed = 0;
            m
        });
        // The mean of a map is what a wrong constant moves first.
        assert!(cov.mean < 1e-3 && cov.frac < 0.01 && cov.max < 0.3, "coverage {cov:?}");
        assert!(kind.mean < 1e-3 && kind.frac < 0.01, "type {kind:?}");
        assert!(high.mean < 1e-3 && high.frac < 0.01, "high share {high:?}");
    }
}

/// A smooth synthetic world: a continent band and a ridge, `h x w`, north-up.
fn synthetic_surface(h: usize, w: usize) -> Surface {
    let mut land = Vec::with_capacity(h * w);
    let mut elevation = Vec::with_capacity(h * w);
    for i in 0..h {
        for j in 0..w {
            let (y, x) = (i as f32 / h as f32, j as f32 / w as f32);
            let continent = ((x * 6.0).sin() * (y * 5.0).cos() + 0.35).clamp(0.0, 1.0);
            land.push(continent);
            elevation.push(continent * (2.0 + 4.0 * ((x * 17.0).sin() * (y * 9.0).sin()).abs()));
        }
    }
    Surface::new(land, elevation, h, w, "test world")
}

/// A map made in bands of 100 rows is the map made in one band, and the surface terms run through the GPU.
#[test]
fn bands_do_not_change_the_map() {
    let Some(gpu) = gpu_or_skip("bands_do_not_change_the_map") else { return };
    let width = 768; // wider than the 512 back-trace grid: the upsampling runs, per band
    let params = GenParams::default();
    let surface = synthetic_surface(64, 128);
    let plans = GeneratePlans::from_seed(&EARTH, 5, width, &params);
    let whole = GpuPipeline::new(&gpu);
    let banded = GpuPipeline::new(&gpu).with_band_rows(100);
    for (label, surf) in [("without a surface", None), ("with the synthetic surface", Some(&surface))] {
        let (a, _) = whole.generate(&EARTH, &plans, width, &params, surf, true, &mut |_, _| {}).unwrap();
        let (b, _) = banded.generate(&EARTH, &plans, width, &params, surf, true, &mut |_, _| {}).unwrap();
        let (ae, be) = (a.extras.as_ref().unwrap(), b.extras.as_ref().unwrap());
        let (cov, kind, high) = (diff(&a.coverage, &b.coverage), diff(&ae.kind, &be.kind), diff(&ae.high_share, &be.high_share));
        println!("{label}: whole vs 100-row bands");
        print_diff("coverage", cov);
        print_diff("type", kind);
        print_diff("high_share", high);
        // The bands merge their statistics in another order: the standardized fields differ by f32 rounding.
        assert!(cov.mean < 1e-5 && cov.frac < 1e-3, "{label}: {cov:?}");
        assert!(kind.mean < 1e-5 && kind.frac < 1e-3, "{label}: {kind:?}");
        assert!(high.mean < 1e-5 && high.frac < 1e-3, "{label}: {high:?}");
        assert_eq!(a.meta, b.meta);
    }
    // The surface changes the map (a pipeline that ignored it would not).
    let (flat, _) = whole.generate(&EARTH, &plans, width, &params, None, false, &mut |_, _| {}).unwrap();
    let (land, _) = whole.generate(&EARTH, &plans, width, &params, Some(&surface), false, &mut |_, _| {}).unwrap();
    let d = diff(&flat.coverage, &land.coverage);
    println!("surface vs none: mean |diff| {:.4}", d.mean);
    assert!(d.mean > 0.005, "the surface terms did not change the map: {d:?}");
    assert!(land.meta.surface.is_some() && flat.meta.surface.is_none());
}

/// The seeded GPU map (counter-based sub-texel noise): calibrated like the CPU's, and the same between runs.
#[test]
fn the_seeded_gpu_map_is_calibrated() {
    let Some(gpu) = gpu_or_skip("the_seeded_gpu_map_is_calibrated") else { return };
    let width = 1024;
    let params = GenParams::default();
    let mut ignore = |_: f32, _: &str| {};
    let a = generate_gpu_seeded(&gpu, &EARTH, 11, width, &params, None, true, &mut ignore).expect("Earth is bakeable");
    let b = generate_gpu_seeded(&gpu, &EARTH, 11, width, &params, None, false, &mut ignore).expect("Earth is bakeable");
    assert_eq!(a.coverage, b.coverage, "the same seed, the same map");
    assert_eq!(a.meta.seed, 11);
    let cpu = cloud_map_generator_core::generate::generate(&EARTH, 11, width, &params, None, false).expect("Earth is bakeable");
    // The sub-texel noise differs (another generator); the rest of the map is the same plans.
    let d = diff(&a.coverage, &cpu.coverage);
    print_diff("coverage", d);
    let (mean_gpu, mean_cpu) = (mean(&a.coverage), mean(&cpu.coverage));
    println!("  global mean: GPU {mean_gpu:.5}, CPU {mean_cpu:.5}");
    assert!((mean_gpu - mean_cpu).abs() < 2e-3, "{mean_gpu} vs {mean_cpu}");
    assert!(d.mean < 0.02, "{d:?}");
    assert!(a.coverage.iter().all(|v| (0.0..=1.0).contains(v)), "coverage leaves [0, 1]");
    // A body whose clouds are not water is refused, as on the CPU.
    let methane = EARTH.with_regime("methane");
    assert!(generate_gpu_seeded(&gpu, &methane, 1, 64, &params, None, false, &mut ignore).is_err());
    let plans = GeneratePlans::from_seed(&EARTH, 1, 64, &params);
    assert!(generate_gpu(&gpu, &methane, &plans, 64, &params, None, false, &mut ignore).is_err());
}

fn mean(a: &[f32]) -> f64 {
    a.iter().map(|&v| f64::from(v)).sum::<f64>() / a.len() as f64
}
