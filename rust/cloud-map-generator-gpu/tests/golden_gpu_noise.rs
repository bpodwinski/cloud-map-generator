//! Golden cases of cloudgen/noise.py (group sphere_noise) run through the GPU kernels: functions
//! `cloudgen.noise.fbm`, `cell_edges`, `Perlin3` and `Worley3`. The scalar helpers (`_fade`, `_grad`, the octave
//! schedule, the rotation) stay on the CPU and are covered by `cloud-map-generator-core/tests/golden_noise.rs`.
//!
//! The frozen tolerances were measured for float32 numpy. A case that passes at its own tolerance keeps it; the
//! measured GPU-versus-golden ratio of every case is printed (`tolerance use`), and a widening is declared only
//! where the GPU needs it, with the measurement as its reason. Without a GPU adapter the tests print why and
//! skip, unless the environment variable REQUIRE_GPU is set, in which case they fail.

use cloud_map_generator_core::noise::{CellEdgesPlan, FbmOctave, FbmPlan, Perlin3, Worley3};
use cloud_map_generator_core::rng::Rng;
use cloud_map_generator_golden::{load_group, run_widened, Array, Case, Config, DType, Out, Tol, Widen};
use cloud_map_generator_gpu::noise::{self, NoiseKernels};
use cloud_map_generator_gpu::GpuContext;
use std::cell::RefCell;
use std::collections::BTreeMap;

/// The six `fbm.stat.*` cases freeze Python's per-seed mean and std over numpy's draws: the output cannot match
/// a port that draws its own plans. They are judged by `fbm_stat_cases_stay_in_their_bands_on_the_gpu`
/// (GOLDEN_SLOW=1), which draws the plans with the CPU port's own Rng and evaluates them on the GPU.
const STAT_REASON: &str = "rng+stat: Python's 16 per-seed means and stds over numpy's bit stream; band test fbm_stat_cases_stay_in_their_bands_on_the_gpu";

/// Known failures: the stat cases (not judged by value) and the cases that need float64 arithmetic.
const EXPECTED_FAILURES: &[(&str, &str)] = &[
    ("fbm.stat.1oct.billow", STAT_REASON),
    ("fbm.stat.1oct.plain", STAT_REASON),
    ("fbm.stat.3oct.billow", STAT_REASON),
    ("fbm.stat.3oct.plain", STAT_REASON),
    ("fbm.stat.6oct.billow", STAT_REASON),
    ("fbm.stat.6oct.plain", STAT_REASON),
    (
        "perlin3.f64.random",
        "f64: every intermediate of the case is float64 (rtol 1e-9, atol 1e-12) and WGSL has no f64; the GPU kernel on the coordinates rounded to f32 (they reach +-300, float32 spacing 3e-5) is off by 3.0e-5 at most, 6.9e6 x the tolerance",
    ),
    (
        "worley3.f64_points.large",
        "f64: floor() and the fraction must be taken in float64 for coordinates of magnitude 1024..2048 (float32 spacing 1.2e-4); the kernel takes f32 points, so F1 and F2 move by up to 8.0e-5, outside the tolerance for 646 (F1) and 569 (F2) of 800 points, the failure the case doc describes for any float32 port",
    ),
];

/// Widenings of the frozen tolerances, per function prefix: none needed so far.
const WIDEN: &[Widen] = &[];

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

fn vec3_f32(a: &Array) -> Vec<[f32; 3]> {
    a.as_f32().chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}
/// Points of any float dtype as float32: what `fbm` and `cell_edges` do first (`astype(np.float32)`), and all the
/// GPU can hold.
fn points_as_f32(a: &Array) -> Vec<[f32; 3]> {
    a.to_f64().chunks_exact(3).map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect()
}
/// Shape of an array without its trailing axis of 3.
fn lead(a: &Array) -> Vec<usize> {
    a.shape[..a.shape.len() - 1].to_vec()
}
fn mat3_f32(v: &[f32]) -> [[f32; 3]; 3] {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

fn perlin_of(c: &Case) -> Perlin3 {
    Perlin3::new(&c.input("plan0.perm").as_i32())
}

fn worley_of(c: &Case) -> Worley3 {
    Worley3::new(&c.input("plan0.perm").as_i64(), &vec3_f32(c.input("plan0.jitter")))
}

/// The fBm plan of an fbm case from the recorded draws (plan0 = the Perlin permutation, plan1 = the octaves);
/// the recorded `norm` is checked against the recomputed one.
fn fbm_plan_of(c: &Case) -> FbmPlan {
    let amp = c.input("plan1.amp").as_f64();
    let freq = c.input("plan1.freq").as_f64();
    let rot = c.input("plan1.rot").as_f32();
    let off = c.input("plan1.off").as_f32();
    let n = amp.len();
    let octaves = (0..n)
        .map(|k| FbmOctave { amp: amp[k], freq: freq[k], rot: mat3_f32(&rot[9 * k..9 * k + 9]), off: [off[3 * k], off[3 * k + 1], off[3 * k + 2]] })
        .collect();
    let plan = FbmPlan::new(perlin_of(c), octaves, c.p_bool("billow"));
    let recorded = c.param("plans")[1]["norm"].as_f64().expect("recorded norm");
    assert!((plan.norm - recorded).abs() <= 1e-13 * recorded.abs(), "norm {} vs recorded {recorded}", plan.norm);
    plan
}

fn cell_edges_plan_of(c: &Case) -> CellEdgesPlan {
    CellEdgesPlan { worley: worley_of(c), rot: mat3_f32(&c.input("plan1.rot").as_f32()), frequency: c.p_f64("frequency") }
}

/// Reads a table back from the GPU after uploading it: the golden `__init__` cases pin the doubled tables the
/// kernels index, so the check goes through the GPU buffer the kernels would bind.
fn through_the_gpu<T: bytemuck::Pod>(gpu: &GpuContext, label: &str, table: &[T]) -> Vec<T> {
    gpu.read_back::<T>(&gpu.storage_with(label, table), table.len())
}

/// The GPU result of one case, as the arrays the case freezes.
fn port(gpu: &GpuContext, kernels: &NoiseKernels, c: &Case) -> Option<Out> {
    Some(match c.function.as_str() {
        "cloudgen.noise.Perlin3.__init__" => {
            let table = through_the_gpu(gpu, "perlin perm", &noise::perlin_table(&perlin_of(c)));
            Out::new().i32("table", &[table.len()], &table)
        }
        "cloudgen.noise.Perlin3.__call__" => {
            let (x, y, z) = (c.input("x"), c.input("y"), c.input("z"));
            let (xs, ys, zs) = (x.to_f64(), y.to_f64(), z.to_f64());
            // The kernel takes f32 points: a float64 case is rounded to f32 (and is listed as a known failure).
            let pts: Vec<[f32; 3]> = (0..xs.len()).map(|i| [xs[i] as f32, ys[i] as f32, zs[i] as f32]).collect();
            let v = kernels.perlin3(&perlin_of(c), &pts);
            Out::new().f32("value", &x.shape, &v)
        }
        "cloudgen.noise.Worley3.__init__" => {
            let table = through_the_gpu(gpu, "worley perm", &noise::worley_perm_table(&worley_of(c)));
            let wide: Vec<i64> = table.iter().map(|&v| i64::from(v)).collect();
            Out::new().i64("perm", &[wide.len()], &wide)
        }
        "cloudgen.noise.Worley3.__call__" => {
            let pts = c.input("pts");
            // The kernel takes f32 points: for float64 points the cell and the fraction are taken after the
            // rounding to f32 (the CPU port splits them in f64).
            let (f1, f2) = kernels.worley3(&worley_of(c), &points_as_f32(pts));
            Out::new().f32("f1", &lead(pts), &f1).f32("f2", &lead(pts), &f2)
        }
        "cloudgen.noise.cell_edges" => {
            let pts = c.input("points");
            let v = kernels.cell_edges(&points_as_f32(pts), &cell_edges_plan_of(c));
            Out::new().f32("value", &lead(pts), &v)
        }
        "cloudgen.noise.fbm" => {
            if c.has_tag("stat") {
                return None; // judged by fbm_stat_cases_stay_in_their_bands_on_the_gpu, see EXPECTED_FAILURES
            }
            let pts = c.input("points");
            let v = kernels.fbm(&points_as_f32(pts), &fbm_plan_of(c));
            Out::new().f32("value", &lead(pts), &v)
        }
        _ => return None,
    })
}

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

/// Runs the golden cases of the four functions through `gpu` and prints the tolerance use of every case.
fn run_cases(gpu: &GpuContext) -> cloud_map_generator_golden::Report {
    let kernels = NoiseKernels::new(gpu);
    let cfg = Config {
        group: "sphere_noise",
        prefixes: &["cloudgen.noise.fbm", "cloudgen.noise.cell_edges", "cloudgen.noise.Perlin3", "cloudgen.noise.Worley3"],
        expected_failures: EXPECTED_FAILURES,
    };
    let uses: RefCell<BTreeMap<String, (f64, f64)>> = RefCell::new(BTreeMap::new());
    let report = run_widened(&cfg, WIDEN, |c| {
        let out = port(gpu, &kernels, c)?;
        uses.borrow_mut().insert(c.id.clone(), tolerance_use(c, &out));
        Some(out)
    });
    // One line per case: the share of its tolerance the GPU uses (1.00 = at the limit).
    let uses = uses.into_inner();
    for (id, (u, abs)) in &uses {
        println!("  tolerance use {u:10.3}  max |diff| {abs:10.3e}  {id}");
    }
    let worst = |p: &str| uses.iter().filter(|(id, _)| id.starts_with(p)).map(|(_, (u, _))| *u).fold(0.0, f64::max);
    println!(
        "tolerance use, worst per function: fbm {:.3}, cell_edges {:.3}, perlin3 {:.3}, worley3 {:.3}",
        worst("fbm"),
        worst("cell_edges"),
        worst("perlin3"),
        worst("worley3")
    );
    report
}

#[test]
fn noise_cases_on_the_gpu() {
    let Some(gpu) = gpu_or_skip("noise_cases_on_the_gpu") else { return };
    run_cases(&gpu).assert_clean();
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
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cloudgen"),
        required_features: wgpu::Features::empty(),
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .ok()?;
    Some(GpuContext { device, queue, adapter_info, limits })
}

/// The same cases on every native backend the machine has (Vulkan, DX12, Metal): the WGSL goes through a
/// different translator on each (SPIR-V, HLSL, MSL), so a backend-specific miscompile or a different
/// fused-multiply-add choice would show here. Backends without an adapter are reported and skipped.
#[test]
fn noise_cases_on_every_native_backend() {
    let mut ran = 0;
    for (name, backends) in [("Vulkan", wgpu::Backends::VULKAN), ("DX12", wgpu::Backends::DX12), ("Metal", wgpu::Backends::METAL)] {
        match context_for(backends) {
            Some(gpu) => {
                println!("backend {name}: adapter {} ({:?}, {:?})", gpu.adapter_info.name, gpu.adapter_info.backend, gpu.adapter_info.device_type);
                run_cases(&gpu).assert_clean();
                ran += 1;
            }
            None => println!("backend {name}: no adapter, skipped"),
        }
    }
    if ran == 0 && std::env::var_os("REQUIRE_GPU").is_some() {
        panic!("no native backend has an adapter and REQUIRE_GPU is set");
    }
}

/// Points uniform on the unit sphere (normalized normal draws), as float32.
fn sphere_points(rng: &mut Rng, n: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|_| {
            let (x, y, z) = (rng.normal(), rng.normal(), rng.normal());
            let r = (x * x + y * y + z * z).sqrt();
            [(x / r) as f32, (y / r) as f32, (z / r) as f32]
        })
        .collect()
}

/// The slow `fbm.stat.*` cases for a port with its own generator, evaluated on the GPU: per seed, 200000
/// unit-sphere points and a plan drawn by the CPU port's Rng; the mean and the (population) std of the field must
/// each lie within `k_sigma` standard deviations of the Python distribution (`params.expect`). Runs with
/// GOLDEN_SLOW=1, like the runner's slow cases.
#[test]
fn fbm_stat_cases_stay_in_their_bands_on_the_gpu() {
    if std::env::var_os("GOLDEN_SLOW").is_none() {
        println!("fbm_stat_cases_stay_in_their_bands_on_the_gpu: skipped (set GOLDEN_SLOW=1)");
        return;
    }
    let Some(gpu) = gpu_or_skip("fbm_stat_cases_stay_in_their_bands_on_the_gpu") else { return };
    let kernels = NoiseKernels::new(&gpu);
    let cases: Vec<Case> = load_group("sphere_noise").into_iter().filter(|c| c.function == "cloudgen.noise.fbm" && c.has_tag("stat")).collect();
    assert_eq!(cases.len(), 6, "the six stat cases of fbm");
    for c in &cases {
        let expect = c.param("expect");
        let (em, esm) = (expect["mean"].as_f64().unwrap(), expect["mean_sd"].as_f64().unwrap());
        let (es, ess) = (expect["std"].as_f64().unwrap(), expect["std_sd"].as_f64().unwrap());
        let k = expect["k_sigma"].as_f64().unwrap();
        let (n, seeds) = (c.p_usize("n_points"), c.p_usize("seeds"));
        let (pseed, fseed) = (c.p_i64("point_seed_base") as u64, c.p_i64("fbm_seed_base") as u64);
        for s in 0..seeds as u64 {
            let pts = sphere_points(&mut Rng::new(pseed + s), n);
            let plan = FbmPlan::from_seed(fseed + s, c.p_f64("frequency"), c.p_usize("octaves"), c.p_f64("gain"), c.p_f64("lacunarity"), c.p_bool("billow"));
            let v = kernels.fbm(&pts, &plan);
            let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
            let std = (v.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            assert!((mean - em).abs() <= k * esm, "{} seed {s}: mean {mean} outside {em} +- {k} x {esm}", c.id);
            assert!((std - es).abs() <= k * ess, "{} seed {s}: std {std} outside {es} +- {k} x {ess}", c.id);
        }
        println!("{}: {seeds} seeds inside the band on the GPU (mean {em} +- {}, std {es} +- {})", c.id, k * esm, k * ess);
    }
}

/// The GPU result matches the CPU port's on the golden inputs of the cases that have a CPU result to compare
/// with: prints the maximum difference of every case and fails past 10x the case's own tolerance (the frozen
/// tolerance covers numpy's own variation; the GPU and the CPU port differ by the same kind of rounding).
#[test]
fn gpu_agrees_with_the_cpu_port_on_the_golden_inputs() {
    let Some(gpu) = gpu_or_skip("gpu_agrees_with_the_cpu_port_on_the_golden_inputs") else { return };
    let kernels = NoiseKernels::new(&gpu);
    let mut worst_overall = 0.0f64;
    for c in load_group("sphere_noise") {
        let (gpu_out, cpu_out, tol): (Vec<f32>, Vec<f32>, (f64, f64)) = match c.function.as_str() {
            "cloudgen.noise.fbm" if !c.has_tag("stat") => {
                let (pts, plan) = (points_as_f32(c.input("points")), fbm_plan_of(&c));
                (kernels.fbm(&pts, &plan), cloud_map_generator_core::noise::fbm(&pts, &plan), tol_of(&c, "value"))
            }
            "cloudgen.noise.cell_edges" => {
                let (pts, plan) = (points_as_f32(c.input("points")), cell_edges_plan_of(&c));
                (kernels.cell_edges(&pts, &plan), cloud_map_generator_core::noise::cell_edges(&pts, &plan), tol_of(&c, "value"))
            }
            "cloudgen.noise.Perlin3.__call__" if c.input("x").dtype == DType::F32 => {
                let p = perlin_of(&c);
                let (x, y, z) = (c.input("x").as_f32(), c.input("y").as_f32(), c.input("z").as_f32());
                let pts: Vec<[f32; 3]> = (0..x.len()).map(|i| [x[i], y[i], z[i]]).collect();
                (kernels.perlin3(&p, &pts), p.call(&x, &y, &z), tol_of(&c, "value"))
            }
            "cloudgen.noise.Worley3.__call__" if c.input("pts").dtype == DType::F32 => {
                let (w, pts) = (worley_of(&c), vec3_f32(c.input("pts")));
                let (g1, g2) = kernels.worley3(&w, &pts);
                let (c1, c2) = w.call(&pts);
                ([g1, g2].concat(), [c1, c2].concat(), tol_of(&c, "f1"))
            }
            _ => continue,
        };
        let (rtol, atol) = tol;
        let (mut max, mut ratio) = (0.0f32, 0.0f64);
        for (g, w) in gpu_out.iter().zip(&cpu_out) {
            let e = (g - w).abs();
            max = max.max(e);
            ratio = ratio.max(f64::from(e) / (atol + rtol * f64::from(w.abs())));
        }
        println!("  GPU vs CPU port  max |diff| {max:10.3e}  tolerance use {ratio:7.3}  {}", c.id);
        worst_overall = worst_overall.max(ratio);
        assert!(ratio < 10.0, "{}: the GPU differs from the CPU port by {ratio} x the case's tolerance (max {max})", c.id);
    }
    println!("worst GPU vs CPU port tolerance use over the golden inputs: {worst_overall:.3}");
}

fn tol_of(c: &Case, output: &str) -> (f64, f64) {
    match &c.expected[output].1 {
        Tol::Close { rtol, atol, .. } => (*rtol, *atol),
        Tol::Exact => (0.0, 0.0),
    }
}
