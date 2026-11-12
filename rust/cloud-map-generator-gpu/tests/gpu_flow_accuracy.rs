//! Accuracy and speed of the GPU flow kernels (cloud-map-generator-gpu/src/flow.rs) against the CPU port
//! (cloud-map-generator-core/src/flow.rs), beyond the frozen golden cases (`golden_gpu_flow.rs`).
//!
//! The light tests run with every `cargo test`: they bind the error of every kernel against the CPU port over
//! a few ten thousand points. The `#[ignore]`d measurements print the numbers of the delivery report and
//! assert the bounds they support:
//!
//! ```text
//! cargo test --release -p cloud-map-generator-gpu --test gpu_flow_accuracy -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The back-trace is the delicate kernel: Python and the CPU port integrate its 48 midpoint steps in f64 and
//! round the result to f32, while WGSL has no f64. The GPU keeps the position either in one f32
//! (`Precision::Plain`) or as a pair of f32 updated with error-free sums (`Precision::Compensated`, the
//! default); the accumulated error of both against the f64 CPU result is measured over the whole sphere and
//! over the strongest vortices the generator draws (a tropical storm of omega up to 14.4).

use cloud_map_generator_core::climate::{BodyInputs, EARTH};
use cloud_map_generator_core::flow::{self as cpu, Flow, Vortex, VortexKind};
use cloud_map_generator_core::rng::Rng;
use cloud_map_generator_gpu::flow::{FlowKernels, GpuFlow, Precision};
use cloud_map_generator_gpu::GpuContext;
use std::time::Instant;

/// The adapter, or `None` (with the reason printed) when there is none; with `REQUIRE_GPU` set a missing
/// adapter fails the test instead.
fn gpu_or_skip() -> Option<GpuContext> {
    let g = GpuContext::new();
    if g.is_none() {
        assert!(std::env::var_os("REQUIRE_GPU").is_none(), "REQUIRE_GPU is set and there is no GPU adapter");
        eprintln!("no GPU adapter: the GPU accuracy tests are skipped (set REQUIRE_GPU=1 to fail instead)");
    }
    g
}

fn to_f64(p: &[[f32; 3]]) -> Vec<[f64; 3]> {
    p.iter().map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]).collect()
}

/// Uniform unit directions as f32 (normalized normal draws).
fn sphere_points(rng: &mut Rng, n: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|_| {
            let (x, y, z) = (rng.normal(), rng.normal(), rng.normal());
            let r = (x * x + y * y + z * z).sqrt();
            [(x / r) as f32, (y / r) as f32, (z / r) as f32]
        })
        .collect()
}

/// The texel-center directions of a `width` x `height` map (north-up, column 0 at longitude -180), row-major.
fn map_points(width: usize, height: usize) -> Vec<[f32; 3]> {
    let pi = std::f64::consts::PI;
    let mut out = Vec::with_capacity(width * height);
    for row in 0..height {
        let lat = pi / 2.0 - (row as f64 + 0.5) * (pi / height as f64);
        let (sl, cl) = lat.sin_cos();
        for col in 0..width {
            let lon = -pi + (col as f64 + 0.5) * (2.0 * pi / width as f64);
            out.push([(cl * lon.cos()) as f32, (cl * lon.sin()) as f32, sl as f32]);
        }
    }
    out
}

/// Unit directions uniform in area inside the cap of angular radius `max_angle` about the direction of
/// (`lon`, `lat`).
fn cap_points(rng: &mut Rng, lon: f64, lat: f64, max_angle: f64, n: usize) -> Vec<[f32; 3]> {
    let c = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
    let east = {
        let e = [-c[1], c[0], 0.0];
        let k = (e[0] * e[0] + e[1] * e[1]).sqrt();
        [e[0] / k, e[1] / k, 0.0]
    };
    let north = [c[1] * east[2] - c[2] * east[1], c[2] * east[0] - c[0] * east[2], c[0] * east[1] - c[1] * east[0]];
    (0..n)
        .map(|_| {
            let cos_t = 1.0 - rng.uniform() * (1.0 - max_angle.cos());
            let sin_t = (1.0 - cos_t * cos_t).sqrt();
            let phi = rng.uniform() * 2.0 * std::f64::consts::PI;
            let (a, b) = (sin_t * phi.cos(), sin_t * phi.sin());
            [0, 1, 2].map(|k| (cos_t * c[k] + a * east[k] + b * north[k]) as f32)
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Stat {
    max: f64,
    mean: f64,
    p999: f64,
}

fn stat(mut e: Vec<f64>) -> Stat {
    assert!(!e.is_empty());
    let mean = e.iter().sum::<f64>() / e.len() as f64;
    e.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in an error"));
    Stat { max: *e.last().unwrap(), mean, p999: e[((e.len() - 1) as f64 * 0.999) as usize] }
}

fn show(what: &str, s: Stat) {
    println!("MEASURE {what:<64} max {:.3e}  mean {:.3e}  p99.9 {:.3e}", s.max, s.mean, s.p999);
}

/// Every component error of two vector fields (not only the largest of each point).
fn all_comp(a: &[[f32; 3]], b: &[[f32; 3]]) -> Vec<f64> {
    a.iter().zip(b).flat_map(|(x, y)| (0..3).map(move |k| f64::from((x[k] - y[k]).abs()))).collect()
}

fn all_comp_f64(a: &[[f32; 3]], b: &[[f64; 3]]) -> Vec<f64> {
    a.iter().zip(b).flat_map(|(x, y)| (0..3).map(move |k| (f64::from(x[k]) - y[k]).abs())).collect()
}

/// Chord length between each pair of directions: the angle in radians, for the small angles that matter here.
fn per_point_chord(a: &[[f32; 3]], b: &[[f32; 3]]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| (0..3).map(|k| (f64::from(x[k]) - f64::from(y[k])).powi(2)).sum::<f64>().sqrt()).collect()
}

/// The largest component difference of each pair of vectors.
fn per_point_comp(a: &[[f32; 3]], b: &[[f32; 3]]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| (0..3).map(|k| f64::from((x[k] - y[k]).abs())).fold(0.0, f64::max)).collect()
}

fn earth_flow(seed: u64) -> Flow {
    cpu::build_flow(&EARTH, seed)
}

/// A single vortex in the default zonal flow.
fn one_vortex(kind: VortexKind, lat: f64, radius: f64, omega: f64, ascent: f64, age: f64) -> Flow {
    Flow { vortices: vec![Vortex { kind, lon: 0.4, lat, radius, omega, ascent, age }], ..Flow::new(30.0) }
}

/// `Flow::zonal_rate` over f32 latitudes.
fn cpu_zonal(flow: &Flow, lats: &[f32]) -> Vec<f64> {
    lats.iter().map(|&l| flow.zonal_rate(l)).collect()
}

// ---------------------------------------------------------------------------------------------------------
// The light tests

#[test]
fn every_kernel_agrees_with_the_cpu_port_on_earth_like_flows() {
    let Some(g) = gpu_or_skip() else { return };
    let k = FlowKernels::new();
    let mut rng = Rng::new(2026);
    let mut worst = [0.0_f64; 4]; // velocity, ascent, backtrace chord, zonal rate
    for seed in 0..3 {
        let flow = earth_flow(seed);
        let points = sphere_points(&mut rng, 8000);
        let gf = GpuFlow::new(&g, &flow);
        worst[0] = worst[0].max(all_comp(&k.velocity(&g, &gf, &points), &flow.velocity(&points)).into_iter().fold(0.0, f64::max));
        let (ga, ca) = (k.ascent(&g, &gf, &points), flow.ascent(&points));
        worst[1] = worst[1].max(ga.iter().zip(&ca).map(|(a, b)| f64::from((a - b).abs())).fold(0.0, f64::max));
        let chord = per_point_chord(&k.backtrace(&g, &gf, &points, 1.0, 48), &flow.backtrace(&points, 1.0, 48));
        worst[2] = worst[2].max(chord.into_iter().fold(0.0, f64::max));
        let lats: Vec<f32> = points.iter().map(|p| p[2].asin()).collect();
        let (gz, cz) = (k.zonal_rate(&g, &gf, &lats), cpu_zonal(&flow, &lats));
        worst[3] = worst[3].max(gz.iter().zip(&cz).map(|(a, b)| (f64::from(*a) - b).abs()).fold(0.0, f64::max));
    }
    println!(
        "worst errors over 3 Earth flows x 8000 points: velocity {:.2e} ascent {:.2e} backtrace {:.2e} zonal {:.2e}",
        worst[0], worst[1], worst[2], worst[3]
    );
    assert!(worst[0] < 2e-6, "velocity {}", worst[0]);
    assert!(worst[1] < 2e-6, "ascent {}", worst[1]);
    assert!(worst[2] < 2e-6, "back-trace {}", worst[2]);
    assert!(worst[3] < 1e-6, "zonal rate {}", worst[3]);
}

/// The compensated back-trace is several times more accurate than the plain one where the error is largest
/// (around a tropical storm of omega 14.4), and both stay inside the tolerance of the golden cases.
#[test]
fn the_compensated_backtrace_beats_the_plain_one_around_a_strong_vortex() {
    let Some(g) = gpu_or_skip() else { return };
    let spec = one_vortex(VortexKind::Tropical, 0.26, 0.05, 14.4, 1.6, 1.0);
    let mut rng = Rng::new(5);
    let points = cap_points(&mut rng, 0.4, 0.26, 0.2, 30_000);
    let want = spec.backtrace(&points, 1.0, 48);
    let gf = GpuFlow::new(&g, &spec);
    let err = |p: Precision| {
        let got = FlowKernels::with_precision(p).backtrace(&g, &gf, &points, 1.0, 48);
        stat(per_point_chord(&got, &want))
    };
    let (compensated, plain) = (err(Precision::Compensated), err(Precision::Plain));
    println!("tropical core, 30000 points: compensated {compensated:?} plain {plain:?}");
    assert!(compensated.mean * 2.0 < plain.mean, "mean {} against {}", compensated.mean, plain.mean);
    assert!(compensated.max * 4.0 < plain.max, "max {} against {}", compensated.max, plain.max);
    assert!(compensated.max < 2e-6 && plain.max < 2e-5, "{compensated:?} {plain:?}");
}

// ---------------------------------------------------------------------------------------------------------
// Measurements for the delivery report

#[test]
#[ignore = "measurement: see the module doc"]
fn measure_accuracy_at_one_million_points() {
    let Some(g) = gpu_or_skip() else { return };
    println!("MEASURE adapter {} ({:?})", g.adapter_info.name, g.adapter_info.backend);
    let (k, kp) = (FlowKernels::new(), FlowKernels::with_precision(Precision::Plain));
    let mut rng = Rng::new(7);
    // 8 Earth flows (different vortex populations) x 125000 points = 1M.
    let (mut vel32, mut vel64, mut cpu_vs64, mut asc, mut asc64, mut zon) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    let (mut back, mut back_comp, mut plain, mut plain_comp) = (vec![], vec![], vec![], vec![]);
    let t0 = Instant::now();
    for seed in 0..8u64 {
        let flow = earth_flow(seed);
        let gf = GpuFlow::new(&g, &flow);
        let points = sphere_points(&mut rng, 125_000);
        let p64 = to_f64(&points);
        let gv = k.velocity(&g, &gf, &points);
        let (v32, v64) = (flow.velocity(&points), flow.velocity(&p64));
        vel32.extend(all_comp(&gv, &v32));
        vel64.extend(all_comp_f64(&gv, &v64));
        cpu_vs64.extend(all_comp_f64(&v32, &v64));
        let ga = k.ascent(&g, &gf, &points);
        let (a32, a64) = (flow.ascent(&points), flow.ascent(&p64));
        asc.extend(ga.iter().zip(&a32).map(|(a, b)| f64::from((a - b).abs())));
        asc64.extend(ga.iter().zip(&a64).map(|(a, b)| f64::from((a - b).abs())));
        let lats: Vec<f32> = points.iter().map(|p| p[2].asin()).collect();
        let gz = k.zonal_rate(&g, &gf, &lats);
        zon.extend(gz.iter().zip(lats.iter()).map(|(a, &l)| (f64::from(*a) - flow.zonal_rate(l)).abs()));
        let cb = flow.backtrace(&points, 1.0, 48);
        let gb = k.backtrace(&g, &gf, &points, 1.0, 48);
        let gp = kp.backtrace(&g, &gf, &points, 1.0, 48);
        back.extend(per_point_chord(&gb, &cb));
        back_comp.extend(per_point_comp(&gb, &cb));
        plain.extend(per_point_chord(&gp, &cb));
        plain_comp.extend(per_point_comp(&gp, &cb));
        println!("seed {seed}: {} vortices, {:.0} s so far", flow.vortices.len(), t0.elapsed().as_secs_f64());
    }
    show("zonal_rate: GPU f32 vs CPU (f32 lat), 1M", stat(zon));
    show("velocity: GPU vs CPU f32, 1M points x 3 components", stat(vel32));
    show("velocity: GPU vs CPU f64 (truth), 1M x 3", stat(vel64));
    show("velocity: CPU f32 vs CPU f64 (the f32 floor), 1M x 3", stat(cpu_vs64));
    show("ascent: GPU vs CPU f32, 1M", stat(asc));
    show("ascent: GPU vs CPU f64 points (f32 rounded), 1M", stat(asc64));
    let (c, p) = (stat(back), stat(plain));
    show("backtrace 48 steps COMPENSATED vs CPU f64, chord rad, 1M", c);
    show("backtrace 48 steps PLAIN f32 vs CPU f64, chord rad, 1M", p);
    let (cc, pc) = (stat(back_comp), stat(plain_comp));
    show("backtrace 48 steps COMPENSATED vs CPU f64, max component, 1M", cc);
    show("backtrace 48 steps PLAIN f32 vs CPU f64, max component, 1M", pc);
    // What the measurements support.
    assert!(c.max < 2e-6 && cc.max < 2e-6, "compensated back-trace {c:?} {cc:?}");
    assert!(p.max < 2e-5, "plain back-trace {p:?}");
}

/// The back-trace error where it can be largest: points around the core of the strongest vortices.
#[test]
#[ignore = "measurement: see the module doc"]
fn measure_backtrace_error_around_the_strongest_vortices() {
    let Some(g) = gpu_or_skip() else { return };
    let (k, kp) = (FlowKernels::new(), FlowKernels::with_precision(Precision::Plain));
    let mut rng = Rng::new(11);
    let cases: Vec<(&str, Flow, f64, f64)> = vec![
        // (label, flow, center lat, cap radius)
        ("tropical omega 14.4 radius 0.05, cap 4 radii", one_vortex(VortexKind::Tropical, 0.26, 0.05, 14.4, 1.6, 1.0), 0.26, 4.0 * 0.05),
        ("tropical omega -14.4 radius 0.035 (small), cap 4 radii", one_vortex(VortexKind::Tropical, -0.3, 0.035, -14.4, 1.6, 1.0), -0.3, 4.0 * 0.035),
        ("tropical omega 12 radius 0.056 (Earth lr), cap 6 radii", one_vortex(VortexKind::Tropical, 0.26, 0.056, 12.0, 1.6, 1.0), 0.26, 6.0 * 0.056),
        ("cyclone omega 5.2 age 1.45 radius 0.3, cap 3 radii", one_vortex(VortexKind::Cyclone, 0.89, 0.3, 5.22, 1.0, 1.45), 0.89, 3.0 * 0.3),
        ("polar cyclone omega 5.2 on the pole, cap 3 radii", one_vortex(VortexKind::Cyclone, std::f64::consts::FRAC_PI_2, 0.3, 5.22, 1.0, 1.45), 1.45, 3.0 * 0.3),
    ];
    let mut worst = (0.0_f64, 0.0_f64);
    let mut report = |label: &str, flow: &Flow, points: &[[f32; 3]]| {
        let gf = GpuFlow::new(&g, flow);
        let cb = flow.backtrace(points, 1.0, 48);
        let (c, p) = (stat(per_point_chord(&k.backtrace(&g, &gf, points, 1.0, 48), &cb)), stat(per_point_chord(&kp.backtrace(&g, &gf, points, 1.0, 48), &cb)));
        show(&format!("COMPENSATED chord rad: {label}"), c);
        show(&format!("PLAIN       chord rad: {label}"), p);
        worst = (worst.0.max(c.max), worst.1.max(p.max));
    };
    for (label, spec, clat, cap) in &cases {
        let points = cap_points(&mut rng, 0.4, *clat, *cap, 200_000);
        report(label, spec, &points);
    }
    // The strongest flows the generator draws: Earth populations plus fast rotators (many vortices).
    for (label, body, seed, n) in [
        ("Earth seed 3", EARTH, 3u64, 200_000usize),
        ("fast rotator 0.4 d (many vortices)", BodyInputs::new(Some(0.4), 6371.0, 9.81, 8.0), 7, 100_000),
        ("very fast rotator 0.05 d", BodyInputs::new(Some(0.05), 6371.0, 9.81, 8.0), 7, 20_000),
    ] {
        let flow = cpu::build_flow(&body, seed);
        let points = sphere_points(&mut rng, n);
        report(&format!("{label}, {} vortices", flow.vortices.len()), &flow, &points);
    }
    assert!(worst.0 < 2e-6, "compensated back-trace worst {}", worst.0);
    assert!(worst.1 < 2e-5, "plain back-trace worst {}", worst.1);
}

fn best_of<T>(runs: usize, mut f: impl FnMut() -> T) -> f64 {
    (0..runs)
        .map(|_| {
            let t = Instant::now();
            let r = f();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            drop(r);
            ms
        })
        .fold(f64::INFINITY, f64::min)
}

/// GPU time (kernel alone, then with upload and read-back) against the CPU port on one thread, for the maps of
/// 4096 x 2048 (8.4M points) and 8192 x 4096 (33M) with the 21 to 24 vortices of an Earth flow and 48 steps.
#[test]
#[ignore = "measurement: see the module doc"]
fn measure_speed() {
    let Some(g) = gpu_or_skip() else { return };
    println!("SPEED adapter {} ({:?}), {:?}", g.adapter_info.name, g.adapter_info.backend, g.adapter_info.device_type);
    let flow = earth_flow(3);
    println!("SPEED flow: Earth seed 3, {} vortices, back-trace 48 steps", flow.vortices.len());
    let (k, kp) = (FlowKernels::new(), FlowKernels::with_precision(Precision::Plain));
    let gf = GpuFlow::new(&g, &flow);
    let wait = || g.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    // The CPU back-trace takes minutes per map: its time per point is measured on a subsample (every point is
    // independent, so the time is linear in the number of points; `measure_cpu_backtrace_directly` checks it).
    // The velocity and the ascent are timed on the whole map.
    let small = map_points(256, 64);
    let cpu_backtrace = best_of(5, || flow.backtrace(&small, 1.0, 48)) / small.len() as f64;
    println!("SPEED cpu single thread back-trace: {:.2} us per point (best of 5 on {} points)", cpu_backtrace * 1e3, small.len());
    for (width, height) in [(4096usize, 2048usize), (8192, 4096)] {
        let points = map_points(width, height);
        let n = points.len();
        let (cpu_velocity, cpu_ascent) = (best_of(2, || flow.velocity(&points)), best_of(2, || flow.ascent(&points)));
        let input = g.storage_with("points", &points);
        let out3 = g.storage_empty::<[f32; 3]>("out3", n);
        let out1 = g.storage_empty::<f32>("out1", n);
        let label = format!("{width}x{height} = {n} points");
        // Kernel alone: buffers resident, submit + wait.
        let kernel = |name: &str, run: &dyn Fn()| {
            run();
            wait(); // warm-up (pipeline compile)
            let ms = best_of(5, || {
                run();
                wait();
            });
            println!("SPEED {label}: {name:<22} GPU kernel {ms:9.2} ms");
            ms
        };
        let gpu_vel = kernel("velocity", &|| k.velocity_buffers(&g, &gf, &input, &out3, n));
        let gpu_asc = kernel("ascent", &|| k.ascent_buffers(&g, &gf, &input, &out1, n));
        let gpu_bt = kernel("backtrace compensated", &|| k.backtrace_buffers(&g, &gf, &input, &out3, n, 1.0, 48));
        let gpu_btp = kernel("backtrace plain f32", &|| kp.backtrace_buffers(&g, &gf, &input, &out3, n, 1.0, 48));
        // With transfer: upload the points, run, read the result back to a Vec.
        let with = |name: &str, run: &dyn Fn() -> usize| {
            let ms = best_of(3, run);
            println!("SPEED {label}: {name:<22} GPU with upload+readback {ms:9.2} ms");
            ms
        };
        let tv = with("velocity", &|| k.velocity(&g, &gf, &points).len());
        let ta = with("ascent", &|| k.ascent(&g, &gf, &points).len());
        let tb = with("backtrace compensated", &|| k.backtrace(&g, &gf, &points, 1.0, 48).len());
        let tbp = with("backtrace plain f32", &|| kp.backtrace(&g, &gf, &points, 1.0, 48).len());
        let nf = n as f64;
        for (name, kernel_ms, total_ms, cpu_ms, how) in [
            ("velocity", gpu_vel, tv, cpu_velocity, "measured"),
            ("ascent", gpu_asc, ta, cpu_ascent, "measured"),
            ("backtrace compensated", gpu_bt, tb, cpu_backtrace * nf, "extrapolated"),
            ("backtrace plain f32", gpu_btp, tbp, cpu_backtrace * nf, "extrapolated"),
        ] {
            println!(
                "SPEED {label}: {name:<22} kernel {kernel_ms:9.2} ms | with transfer {total_ms:9.2} ms | CPU 1 thread {cpu_ms:11.0} ms ({how}) | speedup {:8.0}x kernel, {:8.0}x total",
                cpu_ms / kernel_ms,
                cpu_ms / total_ms
            );
        }
    }
}

/// One direct run of the CPU back-trace on a 4096 x 2048 map (minutes), to check the linear extrapolation that
/// `measure_speed` makes from a small sample.
#[test]
#[ignore = "measurement, several minutes: see the module doc"]
fn measure_cpu_backtrace_directly() {
    let flow = earth_flow(3);
    let points = map_points(4096, 2048);
    let ms = best_of(1, || flow.backtrace(&points, 1.0, 48));
    println!("SPEED cpu single thread back-trace of 4096x2048 = {} points, measured directly: {ms:.0} ms", points.len());
}
