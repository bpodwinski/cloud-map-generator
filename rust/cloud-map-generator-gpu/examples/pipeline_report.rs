//! Timing table and map dumps of the GPU pipeline.
//!
//! ```text
//! cargo run --release -p cloud-map-generator-gpu --example pipeline_report -- timing 512 1024 2048 4096
//! cargo run --release -p cloud-map-generator-gpu --example pipeline_report -- dump <dir> [width] [seed...]
//! ```
//!
//! `timing`: for every width the CPU port on one thread (`generate`, with the derived maps) against the GPU
//! pipeline end to end (`GpuPipeline::generate_seeded`: the plans drawn on the CPU, the uploads, the kernels,
//! the readback of coverage, type and high share). Both repeat and report the best time (other processes use
//! the machine). Prints the table and the width from which the GPU wins by a margin.
//!
//! `dump`: Earth maps of the given seeds at one width, made from the same plans by the CPU port and the GPU
//! pipeline, written as raw little-endian f32 files (`cpu_<seed>.f32`, `gpu_<seed>.f32`), for a contact sheet.

use cloud_map_generator_core::climate::EARTH;
use cloud_map_generator_core::generate::{generate, generate_with_plans, GenParams, GeneratePlans};
use cloud_map_generator_gpu::pipeline::GpuPipeline;
use cloud_map_generator_gpu::GpuContext;
use std::time::Instant;

fn best_of<T>(repeats: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut last = None;
    for _ in 0..repeats {
        let t = Instant::now();
        let out = f();
        best = best.min(t.elapsed().as_secs_f64());
        last = Some(out);
    }
    (best, last.expect("at least one repeat"))
}

fn timing(gpu: &GpuContext, widths: &[usize]) {
    let params = GenParams::default();
    let pipeline = GpuPipeline::new(gpu);
    let warm = pipeline.warm_up();
    println!("adapter {} ({:?}); kernels compiled by a warm-up map in {warm:.2} s", gpu.adapter_info.name, gpu.adapter_info.backend);
    println!("{:>6} {:>10} {:>10} {:>9} {:>14} {:>14}", "width", "CPU s", "GPU s", "speedup", "GPU cpu-part s", "GPU rest s");
    let mut rows: Vec<(usize, f64, f64)> = Vec::new();
    for &width in widths {
        // The CPU is slow on a wide map: three tries up to 2048, one above.
        let cpu_repeats = if width <= 2048 { 3 } else { 1 };
        let (cpu, _) = best_of(cpu_repeats, || generate(&EARTH, 1, width, &params, None, true).expect("Earth is bakeable"));
        let mut parts = (0.0, 0.0);
        let (gpu_s, _) = best_of(5, || {
            let (out, t) = pipeline.generate_seeded(&EARTH, 1, width, &params, None, true, &mut |_, _| {}).expect("Earth is bakeable");
            parts = (t.cpu_seconds, t.gpu_seconds);
            out
        });
        println!("{width:>6} {cpu:>10.3} {gpu_s:>10.3} {:>8.1}x {:>14.3} {:>14.3}", cpu / gpu_s, parts.0, parts.1);
        rows.push((width, cpu, gpu_s));
    }
    // The crossover: the narrowest width from which the GPU is at least 1.25 times faster (a margin: the CPU
    // number is a single thread of a loaded machine, and a map near the crossover gains nothing by moving).
    match rows.iter().find(|&&(_, c, g)| c >= 1.25 * g) {
        Some((w, c, g)) => println!("the GPU wins by 1.25x or more from width {w} ({c:.3} s vs {g:.3} s)"),
        None => println!("the GPU never wins by 1.25x in this range"),
    }
}

fn stats(name: &str, a: &[f32], b: &[f32]) {
    let (mut max, mut sum, mut over) = (0.0_f64, 0.0_f64, 0usize);
    for (x, y) in a.iter().zip(b) {
        let d = f64::from(*x - *y).abs();
        max = max.max(d);
        sum += d;
        over += usize::from(d > 0.01);
    }
    let n = a.len() as f64;
    println!("  {name:<11} max {max:.3e}  mean {:.3e}  texels over 0.01: {:.6}", sum / n, over as f64 / n);
}

fn dump(gpu: &GpuContext, dir: &str, width: usize, seeds: &[u64]) {
    std::fs::create_dir_all(dir).expect("the output directory");
    let params = GenParams::default();
    let pipeline = GpuPipeline::new(gpu);
    for &seed in seeds {
        let plans = GeneratePlans::from_seed(&EARTH, seed, width, &params);
        let cpu = generate_with_plans(&EARTH, &plans, width, &params, None, true, false).expect("Earth is bakeable");
        let (g, _) = pipeline.generate(&EARTH, &plans, width, &params, None, true, &mut |_, _| {}).expect("Earth is bakeable");
        for (name, map) in [("cpu", &cpu.coverage), ("gpu", &g.coverage)] {
            let bytes: Vec<u8> = map.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/{name}_{seed}.f32"), bytes).expect("write the map");
        }
        println!("seed {seed}: {width} x {} written to {dir}; GPU vs CPU on the same plans:", width / 2);
        let (ce, ge) = (cpu.extras.as_ref().expect("extras"), g.extras.as_ref().expect("extras"));
        stats("coverage", &g.coverage, &cpu.coverage);
        stats("type", &ge.kind, &ce.kind);
        stats("high_share", &ge.high_share, &ce.high_share);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(gpu) = GpuContext::new() else {
        eprintln!("no GPU adapter");
        std::process::exit(2);
    };
    match args.first().map(String::as_str) {
        Some("timing") => {
            let widths: Vec<usize> = args[1..].iter().map(|a| a.parse().expect("a width")).collect();
            timing(&gpu, if widths.is_empty() { &[512, 1024, 2048, 4096] } else { &widths });
        }
        Some("dump") => {
            let dir = args.get(1).expect("a directory");
            let width = args.get(2).map_or(1024, |a| a.parse().expect("a width"));
            let seeds: Vec<u64> = if args.len() > 3 { args[3..].iter().map(|a| a.parse().expect("a seed")).collect() } else { vec![1, 2] };
            dump(&gpu, dir, width, &seeds);
        }
        _ => {
            eprintln!("usage: pipeline_report timing [widths...] | dump <dir> [width] [seeds...]");
            std::process::exit(1);
        }
    }
}
