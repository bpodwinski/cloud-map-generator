//! Golden cases of the compose stage run through the GPU kernels (group products_generate): the functions
//! `cloudgen.generate.compose`, `solve_thresholds` and `row_sensitivity` (the shield with its texture and
//! streak, the cumulus population with the fine texture and the sub-texel term, the 36-step per-row bisection,
//! the row sensitivity) and `cloudgen.products.type_map` and `high_share`.
//!
//! The frozen tolerances were measured for float32 numpy. A case that passes at its own tolerance keeps it; the
//! cases whose frozen oracle is FLOAT64 (rtol 1e-9) cannot be met by float32 arithmetic and are listed as
//! EXPECTED_FAILURES with the measured error; `float64_oracle_cases_hold_at_the_float32_tier` binds them at the
//! tolerance the same function's float32 cases carry. No tolerance is widened in this file.
//!
//! Beyond the cases: `gpu_pipeline_agrees_with_the_cpu_port_at_one_million_texels` runs the CPU port's own
//! pipeline on a 1448 x 724 map (without and with the surface terms), feeds its fields to the GPU kernels and
//! reports, stage by stage, how far the GPU is from the CPU (thresholds, sensitivity, offset, coverage, type,
//! high share); `compose_speed_table` (ignored: run it with --ignored --nocapture) measures the kernels on an
//! 8.4M-texel map against the CPU port, one thread.
//!
//! Without a GPU adapter the tests print why and skip, unless the environment variable REQUIRE_GPU is set, in
//! which case they fail.

use cloud_map_generator_core::climate::{hadley_edge_deg, EARTH};
use cloud_map_generator_core::generate::{
    generate_with_plans, solve_thresholds as cpu_solve, row_sensitivity as cpu_sens, compose as cpu_compose, GenParams, GeneratePlans, Offset, Streak, Threshold, Trace,
};
use cloud_map_generator_core::products::{high_share as cpu_high_share, type_map as cpu_type_map};
use cloud_map_generator_core::sphere::grid_lonlat;
use cloud_map_generator_core::surface::Surface;
use cloud_map_generator_golden::{compare_array, load_group, run, Array, Case, Config, Out, Tol};
use cloud_map_generator_gpu::compose::{counter_subtexel_field, subtexel_noise_scale, Band, ComposeKernels, HostFields, HostOffset, HostStreak, OffsetSrc, StreakSrc};
use cloud_map_generator_gpu::products::ProductKernels;
use cloud_map_generator_gpu::GpuContext;
use std::sync::OnceLock;
use std::time::Instant;

fn gpu_or_skip(test: &str) -> Option<GpuContext> {
    match GpuContext::new() {
        Some(g) => {
            println!("{test}: adapter {} ({:?}), max storage binding {} bytes", g.adapter_info.name, g.adapter_info.backend, g.limits.max_storage_buffer_binding_size);
            Some(g)
        }
        None if std::env::var_os("REQUIRE_GPU").is_some() => panic!("{test}: no GPU adapter and REQUIRE_GPU is set"),
        None => {
            println!("{test}: no GPU adapter (wgpu found none): skipped; set REQUIRE_GPU to fail instead");
            None
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// The golden cases

/// Known failures: the cases whose oracle is float64 (rtol 1e-9). Each reason quotes the largest |GPU - frozen|
/// of the run on the RTX 5080 (Vulkan). None of these is a bug: `float64_oracle_cases_hold_at_the_float32_tier`
/// checks them at the float32 tier of their function. (`high_share.all_f64` freezes a float32 output and passes
/// at its own tolerance.)
const EXPECTED_FAILURES: &[(&str, &str)] = &[
    ("compose.all_f64", "float64 oracle (rtol 1e-9, atol 1e-12): the f32 GPU differs by at most 1.5e-7 (f64 inputs rounded to f32 too; 1524 of 2048 texels are outside)"),
    ("row_sensitivity.all_f64", "float64 oracle (rtol 1e-9, atol 1e-9): the f32 GPU differs by at most 2.3e-7 (the f32 compose roundings, divided by the 0.1 of the central difference)"),
    ("solve_thresholds.all_f64", "float64 oracle (rtol 1e-9, atol 1e-9 = 3.5 final-bracket widths): the f32 GPU differs by at most 2.7e-7 (f32 spacing near the root is 6e-8, and the f32 row mean moves the root by that over the slope)"),
];

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

/// A float array of any dtype as f32: what the GPU holds.
fn f32s(a: &Array) -> Vec<f32> {
    a.to_f64().into_iter().map(|v| v as f32).collect()
}

/// The fields of a compose-family case, as host slices.
struct CaseFields {
    s: Vec<f32>,
    fine: Vec<f32>,
    streak: Option<Vec<f32>>,
    streak_scalar: f64,
    subtexel: Option<Vec<f32>>,
    height: usize,
    width: usize,
    prm: GenParams,
}

impl CaseFields {
    fn load(c: &Case) -> CaseFields {
        let s = c.input("s");
        assert_eq!(s.shape.len(), 2, "case {}: s is (height, width)", c.id);
        let optional = |name: &str| c.has_input(name).then(|| f32s(c.input(name)));
        CaseFields {
            s: f32s(s),
            fine: f32s(c.input("fine")),
            streak: optional("streak"),
            streak_scalar: if c.has_param("streak") { c.p_f64("streak") } else { 0.0 },
            subtexel: optional("subtexel"),
            height: s.shape[0],
            width: s.shape[1],
            prm: prm_of(c),
        }
    }

    fn host(&self) -> HostFields<'_> {
        HostFields {
            s: &self.s,
            fine: &self.fine,
            streak: match &self.streak {
                Some(a) => HostStreak::Array(a),
                None => HostStreak::Scalar(self.streak_scalar),
            },
            subtexel: self.subtexel.as_deref(),
            height: self.height,
            width: self.width,
        }
    }
}

struct Golden<'a> {
    kernels: &'a ComposeKernels<'a>,
    products: &'a ProductKernels<'a>,
    /// Forces bands of at most this many rows (the banding itself is under test), or `None`.
    band_rows: Option<usize>,
}

impl Golden<'_> {
    fn compose(&self, c: &Case) -> Out {
        let f = CaseFields::load(c);
        let (h, w) = (f.height, f.width);
        // The threshold is a python float, a (h, 1) column or a (h, w) field: a field is a zero row threshold plus
        // a per-texel offset (u = s - (0 + t)).
        let (thr, offset_full): (Vec<f64>, Option<Vec<f32>>) = if c.has_input("t") {
            let t = c.input("t");
            if t.shape == [h, 1] {
                (t.to_f64(), None)
            } else {
                (vec![0.0; h], Some(f32s(t)))
            }
        } else {
            (vec![c.p_f64("t"); h], None)
        };
        let offset = offset_full.as_deref().map_or(HostOffset::None, HostOffset::Full);
        let cov = self.kernels.compose_host(&f.host(), &f.prm, &thr, offset, self.band_rows);
        Out::new().f32("coverage", &[h, w], &cov)
    }

    fn solve(&self, c: &Case) -> Out {
        let f = CaseFields::load(c);
        let target = c.input("target").to_f64();
        let thr = if c.has_input("offset") {
            let o = c.input("offset");
            let data = f32s(o);
            let offset = if o.shape == [f.height, 1] { HostOffset::Column(&data) } else { HostOffset::Full(&data) };
            self.kernels.solve_thresholds_host(&f.host(), &f.prm, &target, offset, self.band_rows)
        } else {
            let scalar = if c.has_param("offset") { c.p_f64("offset") } else { 0.0 };
            self.kernels.solve_thresholds_host(&f.host(), &f.prm, &target, HostOffset::Scalar(scalar), self.band_rows)
        };
        let thr: Vec<f64> = thr.into_iter().map(f64::from).collect();
        Out::new().f64("thr", &[f.height], &thr)
    }

    fn sensitivity(&self, c: &Case) -> Out {
        let f = CaseFields::load(c);
        let sens = self.kernels.row_sensitivity_host(&f.host(), &f.prm, &c.input("thr").to_f64(), self.band_rows);
        let sens: Vec<f64> = sens.into_iter().map(f64::from).collect();
        Out::new().f64("sens", &[f.height], &sens)
    }

    fn type_map(&self, c: &Case) -> Out {
        let shield = c.input("shield");
        let (h, w) = (shield.shape[0], shield.shape[1]);
        let got = self.products.type_map_host(
            &f32s(shield),
            &f32s(c.input("w_closed")),
            &f32s(c.input("w_open")),
            &f32s(c.input("w_street")),
            &c.input("lat_rows").to_f64(),
            w,
            c.p_f64("hadley_deg"),
            self.band_rows,
        );
        Out::new().f32("type", &[h, w], &got)
    }

    fn high_share(&self, c: &Case) -> Out {
        let shield = c.input("shield");
        let (h, w) = (shield.shape[0], shield.shape[1]);
        let got = self.products.high_share_host(&f32s(shield), &f32s(c.input("ascent")), &f32s(c.input("coverage")), &c.input("lat_rows").to_f64(), w, c.p_f64("hadley_deg"), self.band_rows);
        Out::new().f32("high_share", &[h, w], &got)
    }

    fn port(&self, c: &Case) -> Option<Out> {
        Some(match c.function.as_str() {
            "cloudgen.generate.compose" => self.compose(c),
            "cloudgen.generate.solve_thresholds" => self.solve(c),
            "cloudgen.generate.row_sensitivity" => self.sensitivity(c),
            "cloudgen.products.type_map" => self.type_map(c),
            "cloudgen.products.high_share" => self.high_share(c),
            _ => return None,
        })
    }
}

const PREFIXES: &[&str] = &[
    "cloudgen.generate.compose",
    "cloudgen.generate.solve_thresholds",
    "cloudgen.generate.row_sensitivity",
    "cloudgen.products.type_map",
    "cloudgen.products.high_share",
];

/// `(use, max |diff|)`: how much of its frozen tolerance the GPU result uses, the max over the elements of
/// `|got - want| / (atol + rtol |want|)`, and the largest absolute difference. A use above 1 means the case fails
/// at its own tolerance; it is the factor a widening would need.
fn tolerance_use(c: &Case, out: &Out) -> (f64, f64) {
    let (mut worst, mut max_abs) = (0.0f64, 0.0f64);
    for (name, (want, tol)) in &c.expected {
        let (Some(got), Tol::Close { rtol, atol, .. }) = (out.arrays.get(name), tol) else { continue };
        if want.dtype.is_float() && got.shape == want.shape {
            let (g, w) = (got.to_f64(), want.to_f64());
            for i in 0..g.len() {
                let e = (g[i] - w[i]).abs();
                if e.is_finite() {
                    worst = worst.max(e / (atol + rtol * w[i].abs()));
                    max_abs = max_abs.max(e);
                }
            }
        }
    }
    (worst, max_abs)
}

fn run_cases(gpu: &GpuContext, band_rows: Option<usize>) -> cloud_map_generator_golden::Report {
    let (kernels, products) = (ComposeKernels::new(gpu), ProductKernels::new(gpu));
    let golden = Golden { kernels: &kernels, products: &products, band_rows };
    let uses: std::cell::RefCell<std::collections::BTreeMap<String, (f64, f64)>> = Default::default();
    let cfg = Config { group: "products_generate", prefixes: PREFIXES, expected_failures: EXPECTED_FAILURES };
    let report = run(&cfg, |c| {
        let out = golden.port(c)?;
        uses.borrow_mut().insert(c.id.clone(), tolerance_use(c, &out));
        Some(out)
    });
    for (id, (u, abs)) in uses.borrow().iter() {
        println!("  tolerance use {u:10.3}  max |diff| {abs:10.3e}  {id}");
    }
    report
}

#[test]
fn compose_cases_on_the_gpu() {
    let Some(gpu) = gpu_or_skip("compose_cases_on_the_gpu") else { return };
    run_cases(&gpu, None).assert_clean();
}

/// The same cases with bands of 3 rows: the banding changes no result (every row is independent).
#[test]
fn compose_cases_on_the_gpu_with_bands_of_three_rows() {
    let Some(gpu) = gpu_or_skip("compose_cases_on_the_gpu_with_bands_of_three_rows") else { return };
    run_cases(&gpu, Some(3)).assert_clean();
}

/// A device requested with the WebGPU DEFAULT limits (128 MiB storage bindings, 256 MiB buffers, 65535 workgroups
/// per dimension, 8 storage buffers per stage, 16 KiB of workgroup memory), whatever the adapter offers: what a
/// browser gives. The kernels must run on it unchanged.
fn portable_context() -> Option<GpuContext> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok()?;
    let limits = wgpu::Limits::default();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("cloudgen portable"), required_features: wgpu::Features::empty(), required_limits: limits.clone(), ..Default::default() })).ok()?;
    Some(GpuContext { device, queue, adapter_info: adapter.get_info(), limits })
}

fn portable_or_skip(test: &str) -> Option<GpuContext> {
    match portable_context() {
        Some(g) => {
            println!("{test}: adapter {} ({:?}) with the WebGPU default limits: binding {} bytes, buffer {} bytes", g.adapter_info.name, g.adapter_info.backend, g.limits.max_storage_buffer_binding_size, g.limits.max_buffer_size);
            Some(g)
        }
        None if std::env::var_os("REQUIRE_GPU").is_some() => panic!("{test}: no GPU adapter and REQUIRE_GPU is set"),
        None => {
            println!("{test}: no GPU adapter: skipped; set REQUIRE_GPU to fail instead");
            None
        }
    }
}

/// The same cases on a device with the WebGPU default limits (what a browser gives).
#[test]
fn compose_cases_on_a_device_with_the_webgpu_default_limits() {
    let Some(gpu) = portable_or_skip("compose_cases_on_a_device_with_the_webgpu_default_limits") else { return };
    assert_eq!(gpu.limits.max_storage_buffer_binding_size, 128 << 20);
    run_cases(&gpu, None).assert_clean();
}

/// An adapter of one native backend (the default context picks one for the machine).
fn context_for(backends: wgpu::Backends) -> Option<GpuContext> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok()?;
    let (adapter_info, limits) = (adapter.get_info(), adapter.limits());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("cloudgen"), required_features: wgpu::Features::empty(), required_limits: limits.clone(), ..Default::default() })).ok()?;
    Some(GpuContext { device, queue, adapter_info, limits })
}

/// The golden cases and the counter-based normal (bit for bit against the Rust twin) on every native backend the
/// machine has: the WGSL goes through a different translator on each (SPIR-V, HLSL, MSL), so a backend that fuses
/// the barrier-protected arithmetic or reassociates a sum would show here. Backends without an adapter are
/// reported and skipped.
#[test]
fn compose_cases_and_the_counter_normal_on_every_native_backend() {
    let mut ran = 0;
    for (name, backends) in [("Vulkan", wgpu::Backends::VULKAN), ("DX12", wgpu::Backends::DX12), ("Metal", wgpu::Backends::METAL)] {
        match context_for(backends) {
            Some(gpu) => {
                println!("backend {name}: adapter {} ({:?}, {:?})", gpu.adapter_info.name, gpu.adapter_info.backend, gpu.adapter_info.device_type);
                run_cases(&gpu, None).assert_clean();
                let kernels = ComposeKernels::new(&gpu);
                let (height, width, seed) = (256usize, 1024usize, 0xABCD_EF01_2345_6789u64);
                let field = kernels.subtexel_host(height, width, seed, 1.0, None);
                let twin = counter_subtexel_field(seed, height, width, 1.0);
                let bad = field.iter().zip(&twin).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                println!("backend {name}: counter-based normal differs from the Rust twin in {bad} of {} texels", field.len());
                assert_eq!(bad, 0, "backend {name}: the counter-based normal is not the twin's");
                ran += 1;
            }
            None => println!("backend {name}: no adapter, skipped"),
        }
    }
    if ran == 0 && std::env::var_os("REQUIRE_GPU").is_some() {
        panic!("no native backend has an adapter and REQUIRE_GPU is set");
    }
}

/// A map larger than one default storage binding, 16384 x 8192 = 134M texels (a binding holds 33.5M f32, so a
/// band is 2048 rows and the map four bands), on a device with the WebGPU default limits. The fields are filled on
/// the GPU (counter-based normals, band-invariant, so a different band size gives the same fields); the
/// thresholds must not depend on the band size, bit for bit, and must match the CPU port on sampled rows.
#[test]
fn a_map_larger_than_one_default_binding_is_banded_and_matches_on_sampled_rows() {
    let Some(gpu) = portable_or_skip("a_map_larger_than_one_default_binding_is_banded_and_matches_on_sampled_rows") else { return };
    let (width, height) = (16384usize, 8192usize);
    let prm = GenParams::default();
    let kernels = ComposeKernels::new(&gpu);
    let hadley = hadley_edge_deg(&EARTH);
    let lat_rows: Vec<f64> = grid_lonlat(width, height).1;
    let target: Vec<f64> = lat_rows.iter().map(|&la| cloud_map_generator_core::climate::target_zonal_coverage(la.to_degrees(), hadley, 0.0)).collect();
    let max_rows = cloud_map_generator_gpu::compose::max_band_rows(&gpu, width);
    assert_eq!(max_rows, 2048, "a 128 MiB binding holds 2048 rows of 16384 f32");
    // fields of the generator's statistics, per band: (seed, scale) of the counter-based normal
    let field = |rows: usize, r0: usize, seed: u64, scale: f32| kernels.fill_subtexel(rows, width, r0, seed, scale);
    let solve_all = |band_rows: usize, sample: &mut Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>| -> Vec<f32> {
        let mut thr_all = Vec::with_capacity(height);
        for r0 in (0..height).step_by(band_rows) {
            let rows = band_rows.min(height - r0);
            let (s, fine, streak, sub) = (field(rows, r0, 1, 1.0), field(rows, r0, 2, 0.35), field(rows, r0, 3, 0.5), field(rows, r0, 4, 0.3));
            let band = Band { rows, width, s: &s, fine: &fine, streak: StreakSrc::Buffer(&streak), subtexel: Some(&sub) };
            let tz: Vec<f32> = target[r0..r0 + rows].iter().map(|&v| v as f32).collect();
            let thr = kernels.solve_thresholds(&band, &prm, &gpu.storage_with("target", &tz), OffsetSrc::None);
            thr_all.extend(gpu.read_back::<f32>(&thr, rows));
            if band_rows == max_rows {
                // the first two rows of every band, for the CPU check
                let two = |b: &wgpu::Buffer| gpu.read_back::<f32>(b, 2 * width);
                sample.push((r0, two(&s), two(&fine), two(&streak), two(&sub)));
            }
        }
        thr_all
    };
    let mut sampled = Vec::new();
    let t0 = Instant::now();
    let banded = solve_all(max_rows, &mut sampled);
    println!("16384 x 8192 in {} bands of {} rows: {:.0} ms (fields filled, thresholds solved and read back)", height / max_rows, max_rows, t0.elapsed().as_secs_f64() * 1e3);
    let smaller = solve_all(1000, &mut Vec::new());
    assert!(banded.iter().zip(&smaller).all(|(a, b)| a.to_bits() == b.to_bits()), "the thresholds depend on the band size");
    assert!(banded.iter().all(|t| t.is_finite() && *t > -10.0 && *t < 10.0));
    assert_eq!(sampled.len(), 4);
    for (r0, s, fine, streak, sub) in &sampled {
        let tz = &target[*r0..*r0 + 2];
        let cpu = cpu_solve(s, fine, 2, width, tz, &prm, Streak::Array(&streak[..]), Some(&sub[..]), Offset::<f32>::Scalar(0.0));
        let err = (0..2).map(|i| (f64::from(banded[r0 + i]) - cpu[i]).abs()).fold(0.0, f64::max);
        println!("  rows {r0}, {}: thr GPU {:?} CPU {:?}, max |diff| {err:.2e}", r0 + 1, &banded[*r0..*r0 + 2], cpu);
        assert!(err < 5e-6, "rows {r0}..: {err}");
    }
}

/// The banding of a real map changes nothing, bit for bit: thresholds and coverage with bands of 100 and 61 rows
/// equal the one-band results.
#[test]
fn bands_do_not_change_the_result_on_a_real_map() {
    let Some(gpu) = gpu_or_skip("bands_do_not_change_the_result_on_a_real_map") else { return };
    let r = reference(true);
    let t = &r.trace;
    let kernels = ComposeKernels::new(&gpu);
    let host = HostFields { s: &t.s, fine: &t.fine, streak: HostStreak::Array(&t.streak), subtexel: t.subtexel.as_deref(), height: r.height, width: r.width };
    let offset = HostOffset::Full(t.offset.as_ref().expect("surface offset"));
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    let whole = kernels.solve_thresholds_host(&host, &r.prm, &t.target, offset, None);
    let banded = kernels.solve_thresholds_host(&host, &r.prm, &t.target, offset, Some(100));
    assert!(same(&whole, &banded), "solve_thresholds depends on the banding");
    let sens_whole = kernels.row_sensitivity_host(&host, &r.prm, &t.thr0.clone().unwrap(), None);
    let sens_banded = kernels.row_sensitivity_host(&host, &r.prm, &t.thr0.clone().unwrap(), Some(61));
    assert!(same(&sens_whole, &sens_banded), "row_sensitivity depends on the banding");
    let cov_whole = kernels.compose_host(&host, &r.prm, &t.thr, offset, None);
    let cov_banded = kernels.compose_host(&host, &r.prm, &t.thr, offset, Some(61));
    assert!(same(&cov_whole, &cov_banded), "compose depends on the banding");
}

/// The tolerance a function's float32 cases carry in the frozen data: (function, sibling float32 case, output).
const F32_TIER: &[(&str, &str, &str)] = &[
    ("cloudgen.generate.compose", "compose.pipeline_dtypes", "coverage"),
    ("cloudgen.generate.solve_thresholds", "solve_thresholds.typical.128x64", "thr"),
    ("cloudgen.generate.row_sensitivity", "row_sensitivity.typical", "sens"),
];

/// The float64-oracle cases, on the GPU, against the same frozen outputs at the tolerance the oracle itself froze
/// for float32 arithmetic of that function.
#[test]
fn float64_oracle_cases_hold_at_the_float32_tier() {
    let Some(gpu) = gpu_or_skip("float64_oracle_cases_hold_at_the_float32_tier") else { return };
    let (kernels, products) = (ComposeKernels::new(&gpu), ProductKernels::new(&gpu));
    let golden = Golden { kernels: &kernels, products: &products, band_rows: None };
    let cases = load_group("products_generate");
    let by_id = |id: &str| cases.iter().find(|c| c.id == id).unwrap_or_else(|| panic!("no golden case {id}"));
    for (id, _) in EXPECTED_FAILURES {
        let c = by_id(id);
        let (_, sibling, output) = F32_TIER.iter().find(|(function, _, _)| *function == c.function).unwrap_or_else(|| panic!("case {id}: no float32 tier known for {}", c.function));
        let tier = by_id(sibling).expected[*output].1.clone();
        let out = golden.port(c).expect("the case is ported");
        let (want, frozen) = &c.expected[*output];
        match (frozen, &tier) {
            (Tol::Close { rtol: own, .. }, Tol::Close { rtol: loose, .. }) => assert!(own < loose, "case {id}: its own tolerance is the tighter one"),
            _ => panic!("case {id}: close tolerances expected"),
        }
        let problems = compare_array(output, &out.arrays[*output], want, &tier);
        assert!(problems.is_empty(), "case {id} at the float32 tier {tier:?}: {problems:?}");
    }
}

// ---------------------------------------------------------------------------------------------------------
// The GPU against the CPU port on a whole map

/// The CPU port's pipeline on a 1448 x 724 map (1.05M texels) with its trace, once per surface variant.
struct Reference {
    width: usize,
    height: usize,
    prm: GenParams,
    trace: Trace,
    coverage: Vec<f32>,
    kind: Vec<f32>,
    high_share: Vec<f32>,
    /// The convection latitude of each row (grid latitude less the ITCZ shift), radians.
    lat_conv: Vec<f64>,
    hadley: f64,
}

const ACCURACY_WIDTH: usize = 1448;

/// A synthetic Earth-like surface: three continents with relief.
fn synthetic_surface(h: usize, w: usize) -> Surface {
    let elev: Vec<f32> = (0..h * w)
        .map(|i| {
            let (r, c) = ((i / w) as f64 / h as f64, (i % w) as f64 / w as f64);
            let (lat, lon) = ((0.5 - r) * std::f64::consts::PI, (c - 0.5) * 2.0 * std::f64::consts::PI);
            let continents = (3.0 * lon).sin() * (2.0 * lat + 0.4).cos() + 0.55 * (5.0 * lat).sin() * (2.0 * lon + 1.0).cos();
            (1.2 * continents - 0.15) as f32
        })
        .collect();
    Surface::from_elevation(&elev, h, w, 0.0, "synthetic continents")
}

fn reference(with_surface: bool) -> &'static Reference {
    static PLAIN: OnceLock<Reference> = OnceLock::new();
    static SURFACE: OnceLock<Reference> = OnceLock::new();
    let cell = if with_surface { &SURFACE } else { &PLAIN };
    cell.get_or_init(|| {
        let (width, prm) = (ACCURACY_WIDTH, GenParams::default());
        let height = width / 2;
        let t0 = Instant::now();
        let plans = GeneratePlans::from_seed(&EARTH, 11, width, &prm);
        let surface = with_surface.then(|| synthetic_surface(360, 720));
        let out = generate_with_plans(&EARTH, &plans, width, &prm, surface.as_ref(), true, true).expect("Earth is bakeable");
        println!("CPU reference ({}): {width} x {height} in {:.1} s", if with_surface { "with surface terms" } else { "no surface" }, t0.elapsed().as_secs_f64());
        let trace = out.trace.expect("a trace was asked for");
        let extras = out.extras.expect("extras were asked for");
        let itcz = trace.terms.as_ref().map_or(0.0, |t| t.itcz_shift_deg).to_radians();
        let lat_conv: Vec<f64> = grid_lonlat(width, height).1.iter().map(|l| l - itcz).collect();
        Reference { width, height, hadley: hadley_edge_deg(&EARTH), prm, coverage: out.coverage, kind: extras.kind, high_share: extras.high_share, lat_conv, trace }
    })
}

/// `(max |a - b|, mean |a - b|, count of |a - b| > tol)`.
fn diff32(a: &[f32], b: &[f32], tol: f32) -> (f32, f64, usize) {
    assert_eq!(a.len(), b.len());
    let (mut max, mut sum, mut over) = (0.0f32, 0.0f64, 0usize);
    for (x, y) in a.iter().zip(b) {
        let e = (x - y).abs();
        max = max.max(e);
        sum += f64::from(e);
        over += usize::from(e > tol);
    }
    (max, sum / a.len().max(1) as f64, over)
}

fn diff_f64(a: &[f32], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (f64::from(*x) - y).abs()).fold(0.0, f64::max)
}

/// What the smoothing scale is: the largest threshold change between two neighboring rows. A threshold error far
/// below it cannot show in the smoothed field.
fn row_step(thr: &[f64]) -> f64 {
    thr.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f64::max)
}

fn upload(gpu: &GpuContext, label: &str, data: &[f32]) -> wgpu::Buffer {
    gpu.storage_with(label, data)
}

/// Runs the whole chain on the GPU on the CPU reference's fields and prints/asserts the agreement stage by stage.
fn compare_with_reference(gpu: &GpuContext, with_surface: bool) {
    let r = reference(with_surface);
    let t = &r.trace;
    let (height, width) = (r.height, r.width);
    let kernels = ComposeKernels::new(gpu);
    let products = ProductKernels::new(gpu);
    let tag = if with_surface { "surface" } else { "plain" };
    println!("--- {tag}: {width} x {height} = {} texels", width * height);

    let (s, fine, streak) = (upload(gpu, "s", &t.s), upload(gpu, "fine", &t.fine), upload(gpu, "streak", &t.streak));
    let subtexel = t.subtexel.as_ref().map(|v| upload(gpu, "subtexel", v));
    let band = Band { rows: height, width, s: &s, fine: &fine, streak: StreakSrc::Buffer(&streak), subtexel: subtexel.as_ref() };
    let anomaly = with_surface.then(|| upload(gpu, "anomaly", &t.terms.as_ref().expect("surface terms").anomaly));

    // The calibration, stage by stage.
    let cal = kernels.calibrate(&band, &r.prm, &t.target_zonal, anomaly.as_ref());
    let read = |b: &wgpu::Buffer, n: usize| -> Vec<f32> { gpu.read_back::<f32>(b, n) };
    let target = read(&cal.target, height);
    let target_err = diff_f64(&target, &t.target);
    println!("  target          max |diff| {target_err:.2e}");
    assert!(target_err < 2e-7, "target {target_err}");
    if with_surface {
        let thr0 = read(cal.thr0.as_ref().unwrap(), height);
        let sens = read(cal.sensitivity.as_ref().unwrap(), height);
        let offset = read(cal.offset.as_ref().unwrap(), width * height);
        let (cpu_thr0, cpu_sens, cpu_off) = (t.thr0.as_ref().unwrap(), t.sensitivity.as_ref().unwrap(), t.offset.as_ref().unwrap());
        let (e_thr0, e_sens) = (diff_f64(&thr0, cpu_thr0), diff_f64(&sens, cpu_sens));
        let (e_off, e_off_mean, _) = diff32(&offset, cpu_off, 0.0);
        println!("  thr0            max |diff| {e_thr0:.2e}   (one row step: {:.2e})", row_step(cpu_thr0));
        println!("  sensitivity     max |diff| {e_sens:.2e}   (range {:.3} .. {:.3})", cpu_sens.iter().cloned().fold(f64::INFINITY, f64::min), cpu_sens.iter().cloned().fold(0.0, f64::max));
        println!("  offset          max |diff| {e_off:.2e}, mean {e_off_mean:.2e}   (range {:.3} .. {:.3})", cpu_off.iter().cloned().fold(f32::INFINITY, f32::min), cpu_off.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        assert!(e_thr0 < 5e-5, "thr0 {e_thr0}");
        assert!(e_sens < 5e-4, "sensitivity {e_sens}");
        assert!(e_off < 1e-3, "offset {e_off}");
    }
    let thr_raw = read(&cal.thr_raw, height);
    let e_raw = diff_f64(&thr_raw, &t.thr_raw);
    let step = row_step(&t.thr_raw);
    println!("  thr_raw         max |diff| {e_raw:.2e}   (the largest step between two rows: {step:.2e}; the smoothing sigma is {:.1} rows)", r.prm.threshold_smooth_deg / (180.0 / height as f64));
    assert!(e_raw < 5e-5, "thr_raw {e_raw}");
    assert!(e_raw < 0.05 * step.max(1e-9), "the threshold error {e_raw} is not well below the row step {step}");

    // The smoothed thresholds and the coverage, the whole chain.
    let out = kernels.coverage(&band, &r.prm, &t.target_zonal, anomaly.as_ref());
    let e_thr = diff_f64(&out.thr, &t.thr);
    let coverage = read(&out.coverage, width * height);
    let (c_max, c_mean, c_over) = diff32(&coverage, &r.coverage, 1e-3);
    println!("  thr (smoothed)  max |diff| {e_thr:.2e}");
    println!("  coverage        max |diff| {c_max:.2e}, mean {c_mean:.2e}, {c_over} of {} texels differ by more than 1e-3", coverage.len());
    assert!(e_thr < 5e-5, "thr {e_thr}");
    assert!(c_max < 5e-3 && c_mean < 2e-5, "coverage: max {c_max}, mean {c_mean}");

    // The compose kernel alone, on the CPU's own thresholds and offsets: what the arithmetic costs, without the
    // bisection.
    let cpu_thr: Vec<f32> = t.thr.iter().map(|&v| v as f32).collect();
    let thr_buffer = upload(gpu, "thr", &cpu_thr);
    let offset_buffer = t.offset.as_ref().map(|o| upload(gpu, "offset", o));
    let kernel_only = kernels.compose(&band, &r.prm, &thr_buffer, offset_buffer.as_ref().map_or(OffsetSrc::None, OffsetSrc::Full));
    let (k_max, k_mean, k_over) = diff32(&read(&kernel_only, width * height), &r.coverage, 1e-5);
    println!("  compose alone (CPU thresholds)  max |diff| {k_max:.2e}, mean {k_mean:.2e}, {k_over} texels above 1e-5");
    assert!(k_max < 5e-5, "compose with the CPU thresholds: {k_max}");

    // The products: the shield, the type and the high share, from the CPU's thresholds and fields.
    let weights = t.weights.as_ref().expect("regime weights").to_f32();
    let (closed, open, street) = (upload(gpu, "closed", &weights.closed), upload(gpu, "open", &weights.open), upload(gpu, "street", &weights.street));
    let (ascent, cpu_cov) = (upload(gpu, "ascent", &t.ascent), upload(gpu, "coverage", &r.coverage));
    let inputs = cloud_map_generator_gpu::products::ProductInputs {
        width,
        lat_rows: &r.lat_conv,
        hadley_deg: r.hadley,
        s: &s,
        thr: &thr_buffer,
        offset: offset_buffer.as_ref().map_or(OffsetSrc::None, OffsetSrc::Full),
        closed: &closed,
        open: &open,
        street: &street,
        ascent: &ascent,
        coverage: &cpu_cov,
    };
    let (kind_buffer, high_buffer) = products.products(&inputs, &r.prm);
    let (kind, high) = (read(&kind_buffer, width * height), read(&high_buffer, width * height));
    let (ty_max, ty_mean, _) = diff32(&kind, &r.kind, 1e-4);
    let (hi_max, hi_mean, _) = diff32(&high, &r.high_share, 1e-4);
    println!("  type            max |diff| {ty_max:.2e}, mean {ty_mean:.2e}");
    println!("  high share      max |diff| {hi_max:.2e}, mean {hi_mean:.2e}");
    assert!(ty_max < 2e-5 && hi_max < 5e-5, "products: type {ty_max}, high share {hi_max}");
    let shield = products.shield_host(&t.s, &cpu_thr.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(), t.offset.as_deref().map_or(HostOffset::None, HostOffset::Full), width, &r.prm, Some(100));
    let cpu_shield = t.shield.as_ref().expect("the shield of the extras");
    let sh_err = diff_f64(&shield, cpu_shield);
    println!("  shield          max |diff| {sh_err:.2e} (banded by 100 rows)");
    assert!(sh_err < 5e-5, "shield {sh_err}");
}

#[test]
fn gpu_pipeline_agrees_with_the_cpu_port_at_one_million_texels() {
    let Some(gpu) = gpu_or_skip("gpu_pipeline_agrees_with_the_cpu_port_at_one_million_texels") else { return };
    compare_with_reference(&gpu, false);
    compare_with_reference(&gpu, true);
}

/// The counter-based sub-texel noise used by both engines: the CPU port's compose on the twin's field and the
/// GPU's chain on the GPU's field give one coverage (the field is bit for bit the same).
#[test]
fn a_counter_based_subtexel_field_gives_the_same_map_on_both_engines() {
    let Some(gpu) = gpu_or_skip("a_counter_based_subtexel_field_gives_the_same_map_on_both_engines") else { return };
    let r = reference(false);
    let t = &r.trace;
    let (height, width, seed) = (r.height, r.width, 77u64);
    let scale = subtexel_noise_scale(10.0);
    let kernels = ComposeKernels::new(&gpu);
    let gpu_field = kernels.subtexel_host(height, width, seed, scale, None);
    let cpu_field = counter_subtexel_field(seed, height, width, scale);
    assert!(gpu_field.iter().zip(&cpu_field).all(|(a, b)| a.to_bits() == b.to_bits()), "the two fields are the same bits");
    // CPU: the port's own solve and compose on the twin's field; GPU: the chain on its own field.
    let streak = Streak::Array(&t.streak[..]);
    let thr_raw = cpu_solve(&t.s, &t.fine, height, width, &t.target_zonal, &r.prm, streak, Some(&cpu_field[..]), Offset::<f32>::Scalar(0.0));
    let host = HostFields { s: &t.s, fine: &t.fine, streak: HostStreak::Array(&t.streak), subtexel: Some(&gpu_field), height, width };
    let gpu_raw = kernels.solve_thresholds_host(&host, &r.prm, &t.target_zonal, HostOffset::None, None);
    let e = diff_f64(&gpu_raw, &thr_raw);
    let step = row_step(&thr_raw);
    println!("counter-based sub-texel field: thr_raw max |diff| GPU vs CPU port {e:.2e} (row step {step:.2e})");
    assert!(e < 5e-5, "{e}");
    let cpu_cov = cpu_compose(&t.s, &t.fine, height, width, Threshold::Column(&thr_raw), &r.prm, streak, Some(&cpu_field[..])).to_f32();
    let gpu_cov = kernels.compose_host(&host, &r.prm, &thr_raw, HostOffset::None, None);
    let (max, mean, _) = diff32(&gpu_cov, &cpu_cov, 1e-4);
    println!("counter-based sub-texel field: compose at the same thresholds, max |diff| {max:.2e}, mean {mean:.2e}");
    assert!(max < 5e-5, "{max}");
}

// ---------------------------------------------------------------------------------------------------------
// Speed

fn best_of<F: FnMut()>(runs: usize, mut f: F) -> f64 {
    (0..runs)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .fold(f64::INFINITY, f64::min)
}

fn wait_idle(gpu: &GpuContext) {
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
}

/// A deterministic standard-normal-ish stream (the counter-based generator: it is also the sub-texel field).
fn normals(seed: u64, n: usize) -> Vec<f32> {
    (0..n).map(|i| cloud_map_generator_gpu::compose::counter_normal(seed, i as u32)).collect()
}

/// The kernels on a 4096 x 2048 map (8.4M texels), the GPU against the CPU port on one thread. Run:
/// `cargo test -p cloud-map-generator-gpu --test golden_gpu_compose --release -- --ignored --nocapture compose_speed_table`.
/// Per kernel: `cpu` = the CPU port, one thread; `gpu` = the kernel alone on data already in GPU buffers (wall
/// time from submit until the queue is idle); `gpu+io` = the data uploaded from slices and the result read back.
/// Each figure is the best of several runs (other processes share the CPU and the GPU). The accuracy of the
/// result against the CPU is printed too, so that a fast wrong kernel cannot pass.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn compose_speed_table() {
    let Some(gpu) = gpu_or_skip("compose_speed_table") else { return };
    let (width, height) = (4096usize, 2048usize);
    let n = width * height;
    let prm = GenParams::default();
    let kernels = ComposeKernels::new(&gpu);
    let products = ProductKernels::new(&gpu);
    println!("adapter: {} ({:?}), driver {} {}", gpu.adapter_info.name, gpu.adapter_info.backend, gpu.adapter_info.driver, gpu.adapter_info.driver_info);

    // Fields with the statistics of the generator's: s of unit variance, fine and streak in about [-1, 1], the
    // sub-texel noise of a 3 km texel holding a few clouds; the zonal target of Earth.
    let s = normals(1, n);
    let fine: Vec<f32> = normals(2, n).iter().map(|v| 0.35 * v).collect();
    let streak: Vec<f32> = normals(3, n).iter().map(|v| 0.5 * v).collect();
    let subtexel: Vec<f32> = normals(4, n).iter().map(|v| 0.3 * v).collect();
    let anomaly: Vec<f32> = normals(5, n).iter().map(|v| 0.08 * v).collect();
    let lat_rows: Vec<f64> = grid_lonlat(width, height).1;
    let hadley = hadley_edge_deg(&EARTH);
    let target: Vec<f64> = lat_rows.iter().map(|&la| cloud_map_generator_core::climate::target_zonal_coverage(la.to_degrees(), hadley, 0.0)).collect();
    let host = HostFields { s: &s, fine: &fine, streak: HostStreak::Array(&streak), subtexel: Some(&subtexel), height, width };

    // Resident copies of the inputs.
    let (b_s, b_fine, b_streak, b_sub, b_anomaly) = (upload(&gpu, "s", &s), upload(&gpu, "fine", &fine), upload(&gpu, "streak", &streak), upload(&gpu, "sub", &subtexel), upload(&gpu, "anomaly", &anomaly));
    let band = Band { rows: height, width, s: &b_s, fine: &b_fine, streak: StreakSrc::Buffer(&b_streak), subtexel: Some(&b_sub) };
    let b_target = upload(&gpu, "target", &target.iter().map(|&v| v as f32).collect::<Vec<_>>());
    // warm-up: the pipelines compile on first use
    let warm = kernels.solve_thresholds(&band, &prm, &b_target, OffsetSrc::None);
    let _ = gpu.read_back::<f32>(&warm, height);

    println!("{:<46} {:>14} {:>14} {:>14}   {}", "kernel (8388608 texels)", "cpu 1 thread", "gpu kernel", "gpu + io", "accuracy vs the CPU port");

    // -- solve_thresholds: 36 bisection steps ----------------------------------------------------------------
    let t0 = Instant::now();
    let cpu_thr = cpu_solve(&s, &fine, height, width, &target, &prm, Streak::Array(&streak), Some(&subtexel[..]), Offset::<f32>::Scalar(0.0));
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    let kernel_ms = best_of(7, || {
        let thr = kernels.solve_thresholds(&band, &prm, &b_target, OffsetSrc::None);
        wait_idle(&gpu);
        drop(thr);
    });
    let mut gpu_thr = Vec::new();
    let io_ms = best_of(3, || gpu_thr = kernels.solve_thresholds_host(&host, &prm, &target, HostOffset::None, None));
    println!("{:<46} {:>11.1} ms {:>11.2} ms {:>11.1} ms   thr max |diff| {:.2e}", "solve_thresholds (36 steps)", cpu_ms, kernel_ms, io_ms, diff_f64(&gpu_thr, &cpu_thr));

    // -- row_sensitivity --------------------------------------------------------------------------------------
    let thr_buffer = upload(&gpu, "thr", &cpu_thr.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let t0 = Instant::now();
    let cpu_sens_v = cpu_sens(&s, &fine, height, width, &cpu_thr, &prm, Streak::Array(&streak), Some(&subtexel[..]));
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    let kernel_ms = best_of(7, || {
        let v = kernels.row_sensitivity(&band, &prm, &thr_buffer);
        wait_idle(&gpu);
        drop(v);
    });
    let mut gpu_sens = Vec::new();
    let io_ms = best_of(3, || gpu_sens = kernels.row_sensitivity_host(&host, &prm, &cpu_thr, None));
    println!("{:<46} {:>11.1} ms {:>11.2} ms {:>11.1} ms   sens max |diff| {:.2e}", "row_sensitivity (2 evaluations)", cpu_ms, kernel_ms, io_ms, diff_f64(&gpu_sens, &cpu_sens_v));

    // -- compose (the final coverage) -------------------------------------------------------------------------
    let t0 = Instant::now();
    let cpu_cov = cpu_compose(&s, &fine, height, width, Threshold::Column(&cpu_thr), &prm, Streak::Array(&streak), Some(&subtexel[..])).to_f32();
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    let kernel_ms = best_of(7, || {
        let v = kernels.compose(&band, &prm, &thr_buffer, OffsetSrc::None);
        wait_idle(&gpu);
        drop(v);
    });
    let mut gpu_cov = Vec::new();
    let io_ms = best_of(3, || gpu_cov = kernels.compose_host(&host, &prm, &cpu_thr, HostOffset::None, None));
    println!("{:<46} {:>11.1} ms {:>11.2} ms {:>11.1} ms   coverage max |diff| {:.2e}", "compose (one evaluation)", cpu_ms, kernel_ms, io_ms, diff32(&gpu_cov, &cpu_cov, 1.0).0);

    // -- the sub-texel noise --------------------------------------------------------------------------------------
    let scale = subtexel_noise_scale(3.0);
    let t0 = Instant::now();
    let cpu_noise = counter_subtexel_field(5, height, width, scale);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    let kernel_ms = best_of(7, || {
        let v = kernels.fill_subtexel(height, width, 0, 5, scale);
        wait_idle(&gpu);
        drop(v);
    });
    let mut gpu_noise = Vec::new();
    let io_ms = best_of(3, || gpu_noise = kernels.subtexel_host(height, width, 5, scale, None));
    let same = gpu_noise.iter().zip(&cpu_noise).all(|(a, b)| a.to_bits() == b.to_bits());
    println!("{:<46} {:>11.1} ms {:>11.2} ms {:>11.1} ms   bit for bit: {same}", "fill_subtexel (counter-based normal)", cpu_ms, kernel_ms, io_ms);

    // -- the surface calibration: targets, two bisections, a sensitivity, the offsets ----------------------------------
    let kernel_ms = best_of(5, || {
        let cal = kernels.calibrate(&band, &prm, &target, Some(&b_anomaly));
        wait_idle(&gpu);
        drop(cal);
    });
    println!("{:<46} {:>14} {:>11.2} ms {:>14}   (2 bisections + 1 sensitivity + targets + offsets)", "calibrate with surface terms", "-", kernel_ms, "-");

    // -- the products -----------------------------------------------------------------------------------------------
    let shield64: Vec<f64> = s.iter().map(|&v| (0.5 + (f64::from(v) - 0.1) / prm.shield_ramp).clamp(0.0, 1.0)).collect();
    let (closed, open, street): (Vec<f32>, Vec<f32>, Vec<f32>) = (
        normals(6, n).iter().map(|v| 0.25 + 0.1 * v.clamp(-2.0, 2.0)).collect(),
        normals(7, n).iter().map(|v| 0.15 + 0.05 * v.clamp(-2.0, 2.0)).collect(),
        normals(8, n).iter().map(|v| 0.1 + 0.03 * v.clamp(-2.0, 2.0)).collect(),
    );
    let ascent: Vec<f32> = normals(9, n).iter().map(|v| 0.4 + 0.3 * v).collect();
    let t0 = Instant::now();
    let cpu_type = cpu_type_map::<f64, f32>(&shield64, &closed, &open, &street, &lat_rows, width, hadley);
    let cpu_high = cpu_high_share::<f64, f32, f32>(&shield64, &ascent, &lat_rows, width, hadley, &cpu_cov);
    let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;
    let b_thr0 = upload(&gpu, "thr0", &vec![0.1f32; height]);
    let (b_closed, b_open, b_street, b_ascent, b_cov) = (upload(&gpu, "closed", &closed), upload(&gpu, "open", &open), upload(&gpu, "street", &street), upload(&gpu, "ascent", &ascent), upload(&gpu, "cov", &cpu_cov));
    let inputs = cloud_map_generator_gpu::products::ProductInputs { width, lat_rows: &lat_rows, hadley_deg: hadley, s: &b_s, thr: &b_thr0, offset: OffsetSrc::None, closed: &b_closed, open: &b_open, street: &b_street, ascent: &b_ascent, coverage: &b_cov };
    let kernel_ms = best_of(7, || {
        let v = products.products(&inputs, &prm);
        wait_idle(&gpu);
        drop(v);
    });
    let (mut gpu_type, mut gpu_high) = (Vec::new(), Vec::new());
    let io_ms = best_of(3, || {
        // the products from slices: shield, type and high share, each band uploaded and read back
        let shield = products.shield_host(&s, &vec![0.1f64; height], HostOffset::None, width, &prm, None);
        gpu_type = products.type_map_host(&shield, &closed, &open, &street, &lat_rows, width, hadley, None);
        gpu_high = products.high_share_host(&shield, &ascent, &cpu_cov, &lat_rows, width, hadley, None);
    });
    println!(
        "{:<46} {:>11.1} ms {:>11.2} ms {:>11.1} ms   type max |diff| {:.2e}, high share {:.2e}",
        "products (shield + type + high share)",
        cpu_ms,
        kernel_ms,
        io_ms,
        diff32(&gpu_type, &cpu_type, 1.0).0,
        diff32(&gpu_high, &cpu_high, 1.0).0
    );
}

/// Where the GPU starts to pay for the threshold solve, the dominant cost of the pipeline: the whole call from
/// slices (upload of the four fields, 36 bisection steps, read-back of the thresholds) against the CPU port on one
/// thread, for maps from 32 to 4096 wide. Run like `compose_speed_table`.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn compose_crossover_table() {
    let Some(gpu) = gpu_or_skip("compose_crossover_table") else { return };
    let prm = GenParams::default();
    let kernels = ComposeKernels::new(&gpu);
    let hadley = hadley_edge_deg(&EARTH);
    println!("{:>6} {:>10} {:>14} {:>14} {:>10}", "width", "texels", "cpu 1 thread", "gpu + io", "speed-up");
    for width in [32usize, 64, 128, 256, 512, 1024, 2048, 4096] {
        let height = width / 2;
        let n = width * height;
        let s = normals(1, n);
        let fine: Vec<f32> = normals(2, n).iter().map(|v| 0.35 * v).collect();
        let streak: Vec<f32> = normals(3, n).iter().map(|v| 0.5 * v).collect();
        let subtexel: Vec<f32> = normals(4, n).iter().map(|v| 0.3 * v).collect();
        let target: Vec<f64> = grid_lonlat(width, height).1.iter().map(|&la| cloud_map_generator_core::climate::target_zonal_coverage(la.to_degrees(), hadley, 0.0)).collect();
        let host = HostFields { s: &s, fine: &fine, streak: HostStreak::Array(&streak), subtexel: Some(&subtexel), height, width };
        let _ = kernels.solve_thresholds_host(&host, &prm, &target, HostOffset::None, None); // warm-up
        let runs = if n <= 1 << 19 { 5 } else { 2 };
        let cpu_ms = best_of(runs, || {
            let _ = cpu_solve(&s, &fine, height, width, &target, &prm, Streak::Array(&streak), Some(&subtexel[..]), Offset::<f32>::Scalar(0.0));
        });
        let gpu_ms = best_of(7, || {
            let _ = kernels.solve_thresholds_host(&host, &prm, &target, HostOffset::None, None);
        });
        println!("{width:>6} {n:>10} {cpu_ms:>11.2} ms {gpu_ms:>11.2} ms {:>9.1}x", cpu_ms / gpu_ms);
    }
}
