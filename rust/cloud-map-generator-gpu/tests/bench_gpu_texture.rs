//! Speed of the GPU stage and texture kernels against the CPU port on a 4096 x 2048 Earth map (8,388,608 texels,
//! the size the pipeline's crossover is about), and the texture of an 8192 x 4096 map (33.6M texels) driven in bands
//! on the GPU alone.
//!
//! Run: `cargo test -p cloud-map-generator-gpu --test bench_gpu_texture -- --ignored --nocapture`. Each figure is
//! the best of several runs (other processes share the CPU and the GPU). Three times are printed per kernel:
//! `cpu` = the CPU port, one thread; `gpu+io` = the inputs uploaded from host slices and the result read back to a
//! `Vec`, wall time; `gpu` = the kernel alone on buffers already on the GPU, wall time from submit until the queue is
//! idle (for the moments the readback of the few thousand partial sums, which the kernel needs, is in it).
//! Also checks that the GPU result agrees with the CPU's, so a fast wrong kernel cannot pass.

mod support_texture;

use cloud_map_generator_core::generate::{along as cpu_along, moisture_profile as cpu_moisture};
use cloud_map_generator_core::matching::ndimage;
use cloud_map_generator_core::sphere::{grid_dirs as cpu_grid_dirs, upsample_dirs as cpu_upsample};
use cloud_map_generator_core::texture::{regime_texture, regime_weights, standardize as cpu_standardize};
use cloud_map_generator_gpu::stages::{StageKernels, SynopticParams};
use cloud_map_generator_gpu::texture::{LatInput, TextureKernels, WeightBuffers};
use cloud_map_generator_gpu::GpuContext;
use std::time::Instant;
use support_texture::{diff_stats, dirs_diff_stats, fixture, gpu_or_skip};

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

fn line(name: &str, n: usize, cpu: f64, io: f64, kernel: f64, diff: f32) {
    println!("{name:<34} {n:>10} {cpu:>10.1} {io:>10.1} {kernel:>9.2} {:>8.0}x {diff:>11.2e}", cpu / kernel);
}

#[test]
#[ignore = "a measurement: run with --release --ignored --nocapture"]
fn gpu_against_the_cpu_port_on_a_4096_map() {
    let Some(gpu) = gpu_or_skip("gpu_against_the_cpu_port_on_a_4096_map") else { return };
    let f = fixture(4096);
    let (w, h, n) = (f.width, f.height, f.n);
    let tk = TextureKernels::new(&gpu);
    let sk: &StageKernels = tk.stages();
    {
        // The fixed cost of a dispatch (buffers, bind group, submit, wait): the same kernel on 1024 texels.
        let small = sk.upload_field(&[0.5f32; 1024]);
        let _ = sk.moisture_profile_to_buffer(&small, 1024, 30.0);
        wait_idle(&gpu);
        let t = best_of(30, || {
            let _ = sk.moisture_profile_to_buffer(&small, 1024, 30.0);
            wait_idle(&gpu);
        });
        println!("fixed cost of one dispatch (1024 texels, moisture_profile): {t:.2} ms");
    }
    println!("{:<34} {:>10} {:>10} {:>10} {:>9} {:>9} {:>11}", "kernel", "texels", "cpu ms", "gpu+io ms", "gpu ms", "cpu/gpu", "max |diff|");

    // Inputs on the GPU for the kernel-only runs.
    let (p_buf, back_buf) = (sk.upload_dirs(&f.p), sk.upload_dirs(&f.back));
    let lat_cpu: Vec<f32> = f.back.iter().map(|b| b[2].clamp(-1.0, 1.0).asin()).collect();
    let lat_buf = sk.upload_field(&lat_cpu);
    let coarse_buf = sk.upload_dirs(&f.coarse);

    // grid_dirs
    let cpu_dirs = cpu_grid_dirs::<f32>(w, h);
    let t_cpu = best_of(2, || {
        let _ = cpu_grid_dirs::<f32>(w, h);
    });
    let got = sk.grid_dirs_to_vec(w, h, 0, h);
    let t_io = best_of(3, || {
        let _ = sk.grid_dirs_to_vec(w, h, 0, h);
    });
    let _ = sk.grid_dirs_to_buffer(w, h, 0, h);
    wait_idle(&gpu);
    let t_k = best_of(10, || {
        let _ = sk.grid_dirs_to_buffer(w, h, 0, h);
        wait_idle(&gpu);
    });
    line("grid_dirs", n, t_cpu, t_io, t_k, dirs_diff_stats(&got, &cpu_dirs).0);

    // upsample_dirs
    let widened: Vec<[f64; 3]> = f.coarse.iter().map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]).collect();
    let cpu_up = cpu_upsample(&widened, f.coarse_h, f.coarse_w, w, h);
    let t_cpu = best_of(2, || {
        let _ = cpu_upsample(&widened, f.coarse_h, f.coarse_w, w, h);
    });
    let got = sk.upsample_dirs_to_vec(&f.coarse, f.coarse_w, f.coarse_h, w, h);
    let t_io = best_of(3, || {
        let _ = sk.upsample_dirs_to_vec(&f.coarse, f.coarse_w, f.coarse_h, w, h);
    });
    let t_k = best_of(10, || {
        let _ = sk.upsample_dirs_to_buffer(&coarse_buf, f.coarse_w, f.coarse_h, w, h, 0, h);
        wait_idle(&gpu);
    });
    line("upsample_dirs (512x256 -> map)", n, t_cpu, t_io, t_k, dirs_diff_stats(&got, &cpu_up).0);

    // along(p, back, 0.6)
    let cpu_a = cpu_along(&f.p, &f.back, 0.6);
    let t_cpu = best_of(2, || {
        let _ = cpu_along(&f.p, &f.back, 0.6);
    });
    let got = sk.along_to_vec(&f.p, &f.back, 0.6);
    let t_io = best_of(3, || {
        let _ = sk.along_to_vec(&f.p, &f.back, 0.6);
    });
    let t_k = best_of(10, || {
        let _ = sk.along_to_buffer(&p_buf, &back_buf, n, 0.6);
        wait_idle(&gpu);
    });
    line("along(p, back, f)", n, t_cpu, t_io, t_k, dirs_diff_stats(&got, &cpu_a).0);

    // lat_of_dirs
    let t_cpu = best_of(2, || {
        let _: Vec<f32> = f.back.iter().map(|b| b[2].clamp(-1.0, 1.0).asin()).collect();
    });
    let flat_back: Vec<f32> = f.back.iter().flatten().copied().collect();
    let t_io = best_of(3, || {
        let b = sk.upload_field(&flat_back);
        let l = sk.lat_of_dirs_to_buffer(&b, n);
        let _ = sk.field_to_vec(&l, n);
    });
    let got = sk.field_to_vec(&sk.lat_of_dirs_to_buffer(&back_buf, n), n);
    let t_k = best_of(10, || {
        let _ = sk.lat_of_dirs_to_buffer(&back_buf, n);
        wait_idle(&gpu);
    });
    line("lat_of_dirs (asin)", n, t_cpu, t_io, t_k, diff_stats(&got, &lat_cpu).0);

    // moisture_profile
    let cpu_m = cpu_moisture(&lat_cpu, f.hadley_deg);
    let t_cpu = best_of(2, || {
        let _ = cpu_moisture(&lat_cpu, f.hadley_deg);
    });
    let got = sk.moisture_profile_to_vec(&lat_cpu, f.hadley_deg);
    let t_io = best_of(3, || {
        let _ = sk.moisture_profile_to_vec(&lat_cpu, f.hadley_deg);
    });
    let t_k = best_of(10, || {
        let _ = sk.moisture_profile_to_buffer(&lat_buf, n, f.hadley_deg);
        wait_idle(&gpu);
    });
    line("moisture_profile", n, t_cpu, t_io, t_k, diff_stats(&got, &cpu_m).0);

    // synoptic s (the four inputs are fields of the map)
    let fields: Vec<Vec<f32>> = (0..3).map(|k| (0..n).map(|i| ((i * (7 + k) % 1009) as f32 / 1009.0 - 0.5) * 2.0).collect()).collect();
    let g = &f.params;
    let (a_m, k_n, a_a, a_mid) = (g.a_moisture as f32, g.moisture_noise as f32, g.a_ascent as f32, g.a_mid as f32);
    let cpu_s = |prof: &[f32]| -> Vec<f32> {
        let mut s: Vec<f32> = (0..n).map(|i| a_m * (prof[i] + k_n * fields[0][i])).collect();
        for (v, a) in s.iter_mut().zip(&fields[1]) {
            *v += a_a * *a;
        }
        for (v, m) in s.iter_mut().zip(&fields[2]) {
            *v += a_mid * *m;
        }
        s
    };
    let s_ref = cpu_s(&cpu_m);
    let t_cpu = best_of(2, || {
        let _ = cpu_s(&cpu_m);
    });
    let sp = SynopticParams { hadley_deg: f.hadley_deg, a_moisture: g.a_moisture, moisture_noise: g.moisture_noise, a_ascent: g.a_ascent, a_mid: g.a_mid };
    let fb: Vec<_> = fields.iter().map(|v| sk.upload_field(v)).collect();
    let run_io = || {
        let (l, m, a, d) = (sk.upload_field(&lat_cpu), sk.upload_field(&fields[0]), sk.upload_field(&fields[1]), sk.upload_field(&fields[2]));
        let s = sk.synoptic_to_buffer(&l, &m, &a, &d, n, &sp);
        sk.field_to_vec(&s, n)
    };
    let got = run_io();
    let t_io = best_of(3, || {
        let _ = run_io();
    });
    let t_k = best_of(10, || {
        let _ = sk.synoptic_to_buffer(&lat_buf, &fb[0], &fb[1], &fb[2], n, &sp);
        wait_idle(&gpu);
    });
    line("synoptic s (assembly)", n, t_cpu, t_io, t_k, diff_stats(&got, &s_ref).0);

    // mean and std of a whole field (numpy's pairwise f32 mean and std on the CPU)
    let s_buf = sk.upload_field(&s_ref);
    let t_cpu = best_of(2, || {
        let _ = (ndimage::mean(&s_ref), ndimage::std(&s_ref, 0));
    });
    let m = sk.moments(&s_buf, n);
    let (cm, cs) = (ndimage::mean(&s_ref), ndimage::std(&s_ref, 0));
    let t_io = best_of(3, || {
        let b = sk.upload_field(&s_ref);
        let _ = sk.moments(&b, n);
    });
    let t_k = best_of(10, || {
        let _ = sk.moments(&s_buf, n);
    });
    line("mean + std (moments)", n, t_cpu, t_io, t_k, ((m.mean - f64::from(cm)).abs().max((m.std() - f64::from(cs)).abs())) as f32);

    // standardize (the statistics and the division)
    let cpu_std = cpu_standardize(&s_ref);
    let t_cpu = best_of(2, || {
        let _ = cpu_standardize(&s_ref);
    });
    let got = sk.standardize_to_vec(&s_ref);
    let t_io = best_of(3, || {
        let _ = sk.standardize_to_vec(&s_ref);
    });
    let t_k = best_of(10, || {
        let (b, _) = sk.standardize_field(&s_buf, n);
        drop(b);
        wait_idle(&gpu);
    });
    line("standardize (moments + divide)", n, t_cpu, t_io, t_k, diff_stats(&got, &cpu_std).0);
    let t_div = best_of(10, || {
        let _ = sk.standardize_to_buffer(&s_buf, n, &m);
        wait_idle(&gpu);
    });
    println!("    of which the divide kernel alone: {t_div:.2} ms");

    // regime_weights: the generator's float64 latitude grid and float32 back-traced latitude on the CPU, the f32
    // chain with the per-row latitude table on the GPU
    let lat_grid: Vec<f64> = f.lat_rows.iter().flat_map(|&l| std::iter::repeat_n(l, w)).collect();
    let cpu_w = regime_weights::<f64, f32, f64>(&lat_grid, &lat_cpu, f.hadley_deg);
    let t_cpu = best_of(2, || {
        let _ = regime_weights::<f64, f32, f64>(&lat_grid, &lat_cpu, f.hadley_deg);
    });
    let rows_buf = sk.upload_field(&sk.grid_lat_rows(w, h, 0, h));
    let run_w_io = || {
        let (r, l) = (sk.upload_field(&sk.grid_lat_rows(w, h, 0, h)), sk.upload_field(&lat_cpu));
        let wb = tk.regime_weights_to_buffers(LatInput::Rows { lat: &r, width: w }, &l, n, f.hadley_deg);
        (sk.field_to_vec(&wb.closed, n), sk.field_to_vec(&wb.open, n), sk.field_to_vec(&wb.street, n))
    };
    let (gc, go, gs) = run_w_io();
    let t_io = best_of(3, || {
        let _ = run_w_io();
    });
    let t_k = best_of(10, || {
        let _ = tk.regime_weights_to_buffers(LatInput::Rows { lat: &rows_buf, width: w }, &lat_buf, n, f.hadley_deg);
        wait_idle(&gpu);
    });
    let f32s = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    let diff = diff_stats(&gc, &f32s(&cpu_w.closed)).0.max(diff_stats(&go, &f32s(&cpu_w.open)).0).max(diff_stats(&gs, &f32s(&cpu_w.street)).0);
    line("regime_weights (3 fields)", n, t_cpu, t_io, t_k, diff);

    // regime_texture end to end: 2 fBm, the cell edges of every used diameter, the ramps, 4 reductions, the blend
    let cpu_tex = regime_texture(&f.p, &f.back, &cpu_w, f.radius_km, f.fmax, &f.texture_plans);
    let t_cpu = best_of(1, || {
        let _ = regime_texture(&f.p, &f.back, &cpu_w, f.radius_km, f.fmax, &f.texture_plans);
    });
    let weights32 = WeightBuffers { closed: sk.upload_field(&f32s(&cpu_w.closed)), open: sk.upload_field(&f32s(&cpu_w.open)), street: sk.upload_field(&f32s(&cpu_w.street)) };
    let run_tex_io = || {
        let (p, b) = (sk.upload_dirs(&f.p), sk.upload_dirs(&f.back));
        let wts = WeightBuffers { closed: sk.upload_field(&f32s(&cpu_w.closed)), open: sk.upload_field(&f32s(&cpu_w.open)), street: sk.upload_field(&f32s(&cpu_w.street)) };
        let t = tk.regime_texture_to_buffer(&p, &b, &wts, n, f.radius_km, f.fmax, &f.texture_plans);
        sk.field_to_vec(&t, n)
    };
    let got = run_tex_io();
    let t_io = best_of(3, || {
        let _ = run_tex_io();
    });
    let t_k = best_of(5, || {
        let _ = tk.regime_texture_to_buffer(&p_buf, &back_buf, &weights32, n, f.radius_km, f.fmax, &f.texture_plans);
        wait_idle(&gpu);
    });
    line("regime_texture (all stages)", n, t_cpu, t_io, t_k, diff_stats(&got, &cpu_tex).0);
    println!("  ({} fine octaves, {} cell diameters: {:?} km)", f.texture_plans.generic.octaves.len(), f.texture_plans.cells.len(), cloud_map_generator_core::texture::used_cells(f.radius_km, f.fmax).iter().map(|c| c.diameter_km).collect::<Vec<_>>());
}

/// The texture of an 8192 x 4096 map made band by band on the GPU (no CPU reference at this size), with the
/// bands the WebGPU default limits require: directions, back-traced directions, latitude, weights, the raw
/// fields, their moments, the blend; the time is the wall time of the whole map until the queue is idle, without
/// the readback of the result.
#[test]
#[ignore = "a measurement: run with --ignored --nocapture"]
fn gpu_texture_of_an_8192_map_in_bands() {
    let Some(gpu) = gpu_or_skip("gpu_texture_of_an_8192_map_in_bands") else { return };
    let f = support_texture::fixture_plans_only(8192);
    let (w, h) = (f.width, f.height);
    let tk = TextureKernels::new(&gpu);
    let sk = tk.stages();
    let coarse = sk.upload_dirs(&f.coarse);
    for band_rows in [512usize, 1024, sk.rows_per_chunk(w, 3)] {
        let run = || {
            let mut bands = Vec::new();
            let mut moments = cloud_map_generator_gpu::texture::TextureMoments::EMPTY;
            for row0 in (0..h).step_by(band_rows) {
                let rows = band_rows.min(h - row0);
                let n = w * rows;
                let p = sk.grid_dirs_to_buffer(w, h, row0, rows);
                let back = sk.upsample_dirs_to_buffer(&coarse, f.coarse_w, f.coarse_h, w, h, row0, rows);
                let lat_back = sk.lat_of_dirs_to_buffer(&back, n);
                let lat_rows = sk.upload_field(&sk.grid_lat_rows(w, h, row0, rows));
                let weights = tk.regime_weights_to_buffers(LatInput::Rows { lat: &lat_rows, width: w }, &lat_back, n, f.hadley_deg);
                let fields = tk.texture_fields(&p, &back, n, f.radius_km, f.fmax, &f.texture_plans);
                moments = moments.merge(tk.texture_moments(&fields));
                bands.push((weights, fields));
            }
            let out: Vec<_> = bands.iter().map(|(wts, fields)| tk.texture_blend(fields, wts, &moments)).collect();
            wait_idle(&gpu);
            out
        };
        let _ = run();
        let t = best_of(3, || {
            let _ = run();
        });
        println!("8192 x 4096 texture in bands of {band_rows:>4} rows: {t:8.1} ms ({} texels)", w * h);
    }
}
