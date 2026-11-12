//! Speed of the GPU noise kernels against the CPU port on the maps the generator makes: a 4096x2048 map (8.4M
//! points) and an 8192x4096 map (33.6M points), at the octave counts of the generator's fine ladder
//! (`octave_ladder(48, width / (4 pi), top 0.95)` is 3 octaves from 75 on a 4096 map and 4 octaves from 75 on an
//! 8192 map, gain 0.65) and at the `cell_edges` frequencies of the regime texture (cells per radian from 424 down).
//!
//! Run: `cargo test -p cloud-map-generator-gpu --test bench_gpu_noise -- --ignored --nocapture`. Each figure is the best of
//! several runs (other processes share the CPU and the GPU). Three times are printed per case:
//! `cpu` = the CPU port, one thread; `gpu+io` = points uploaded and the result read back, from slices to a Vec;
//! `gpu` = the kernel alone on points already in a GPU buffer, wall time from submit until the queue is idle.
//! Also checks that the GPU and CPU results agree, so a fast wrong kernel cannot pass.

use cloud_map_generator_core::noise::{cell_edges, fbm, CellEdgesPlan, FbmPlan, DEFAULT_LACUNARITY};
use cloud_map_generator_gpu::noise::{upload_points, NoiseKernels};
use cloud_map_generator_gpu::GpuContext;
use std::time::Instant;

/// Directions of the texel centers of a north-up equirectangular `w` x `h` map, row-major.
fn map_dirs(w: usize, h: usize) -> Vec<[f32; 3]> {
    let pi = std::f64::consts::PI;
    let lon: Vec<(f64, f64)> = (0..w).map(|c| (-pi + (c as f64 + 0.5) / w as f64 * 2.0 * pi).sin_cos()).collect();
    let mut out = Vec::with_capacity(w * h);
    for r in 0..h {
        let (sl, cl) = (pi / 2.0 - (r as f64 + 0.5) / h as f64 * pi).sin_cos();
        for &(so, co) in &lon {
            out.push([(cl * co) as f32, (cl * so) as f32, sl as f32]);
        }
    }
    out
}

type Points = [[f32; 3]];
type WithIo = Box<dyn Fn(&NoiseKernels, &Points) -> Vec<f32>>;
type KernelOnly = Box<dyn Fn(&NoiseKernels, &wgpu::Buffer) -> wgpu::Buffer>;
type CpuRun = Box<dyn Fn(&Points) -> Vec<f32>>;

/// One measured kernel: its name and the three ways to run it.
struct BenchCase {
    name: String,
    /// GPU, points from a slice and the result read back to a `Vec`.
    with_io: WithIo,
    /// GPU, points already in a buffer, the result left in a buffer.
    kernel_only: KernelOnly,
    /// The CPU port, one thread.
    cpu: CpuRun,
}

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

fn max_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn gpu_against_the_cpu_port_on_full_maps() {
    let Some(gpu) = GpuContext::new() else {
        println!("no GPU adapter: nothing to measure");
        return;
    };
    println!("adapter: {} ({:?}), driver {} {}", gpu.adapter_info.name, gpu.adapter_info.backend, gpu.adapter_info.driver, gpu.adapter_info.driver_info);
    let kernels = NoiseKernels::new(&gpu);
    {
        // The fixed cost of a dispatch (buffers, bind group, submit, wait): the same kernel on 1024 points.
        let small = map_dirs(64, 16);
        let small_buf = upload_points(&gpu, &small);
        let plan = FbmPlan::from_seed(7, 75.0, 3, 0.65, DEFAULT_LACUNARITY, false);
        let _ = kernels.fbm_to_buffer(&small_buf, small.len(), &plan);
        wait_idle(&gpu);
        let t = best_of(20, || {
            let _ = kernels.fbm_to_buffer(&small_buf, small.len(), &plan);
            wait_idle(&gpu);
        });
        println!("fixed cost of one dispatch ({} points, fbm 3 octaves): {t:.2} ms", small.len());
    }
    println!("{:<34} {:>11} {:>11} {:>11} {:>9} {:>9} {:>11}", "kernel", "points", "cpu ms", "gpu+io ms", "gpu ms", "cpu/gpu", "max |diff|");
    // NOISE_BENCH_WIDTHS="4096" limits the maps (default: 4096 and 8192).
    let widths: Vec<usize> = std::env::var("NOISE_BENCH_WIDTHS")
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|_| vec![4096, 8192]);
    for &(w, h, cpu_runs) in &widths.iter().map(|&w| (w, w / 2, if w <= 4096 { 3usize } else { 2 })).collect::<Vec<_>>() {
        let pts = map_dirs(w, h);
        let n = pts.len();
        let buf = upload_points(&gpu, &pts);
        let mut cases: Vec<BenchCase> = Vec::new();
        for &octaves in &[3usize, 4] {
            let plan = FbmPlan::from_seed(7, 75.0, octaves, 0.65, DEFAULT_LACUNARITY, false);
            let (p1, p2, p3) = (plan.clone(), plan.clone(), plan);
            cases.push(BenchCase {
                name: format!("fbm {octaves} octaves, 75 cycles"),
                with_io: Box::new(move |k, p| k.fbm(p, &p1)),
                kernel_only: Box::new(move |k, b| k.fbm_to_buffer(b, n, &p2)),
                cpu: Box::new(move |p| fbm(p, &p3)),
            });
        }
        for &freq in &[106.0f64, 424.0] {
            let plan = CellEdgesPlan::from_seed(8, freq);
            let (p1, p2, p3) = (plan.clone(), plan.clone(), plan);
            cases.push(BenchCase {
                name: format!("cell_edges, {freq} cells/rad"),
                with_io: Box::new(move |k, p| k.cell_edges(p, &p1)),
                kernel_only: Box::new(move |k, b| k.cell_edges_to_buffer(b, n, &p2)),
                cpu: Box::new(move |p| cell_edges(p, &p3)),
            });
        }
        // Where the transfer time goes, for the first kernel: upload alone, readback alone.
        let up = best_of(5, || {
            let b = upload_points(&gpu, &pts);
            wait_idle(&gpu);
            drop(b);
        });
        let out = kernels.fbm_to_buffer(&buf, n, &FbmPlan::from_seed(7, 75.0, 3, 0.65, DEFAULT_LACUNARITY, false));
        wait_idle(&gpu);
        let down = best_of(5, || {
            let _ = gpu.read_back::<f32>(&out, n);
        });
        println!("    transfers at {n} points: upload {up:.1} ms ({:.1} GB/s), GpuContext::read_back {down:.1} ms ({:.1} GB/s)", 12.0 * n as f64 / up / 1e6, 4.0 * n as f64 / down / 1e6);
        for BenchCase { name, with_io, kernel_only, cpu } in &cases {
            let got = with_io(&kernels, &pts); // warm-up: compiles the pipeline
            let mut cpu_out = Vec::new();
            let t_cpu = best_of(cpu_runs, || cpu_out = cpu(&pts));
            let diff = max_error(&got, &cpu_out);
            let t_io = best_of(8, || {
                let _ = with_io(&kernels, &pts);
            });
            wait_idle(&gpu);
            let t_gpu = best_of(15, || {
                let _out = kernel_only(&kernels, &buf);
                wait_idle(&gpu);
            });
            println!("{name:<34} {n:>11} {t_cpu:>11.1} {t_io:>11.1} {t_gpu:>9.2} {:>8.0}x {diff:>11.2e}", t_cpu / t_gpu);
        }
    }
}
