//! Accuracy of the GPU stage and texture kernels (`stages`, `texture`) against the CPU port on the data of a real
//! Earth map, and the properties the pipeline relies on: the mean and standard deviation reduction against f64
//! at 8.4M and 33.6M texels, chunking forced small, and a map driven in bands under the WebGPU default limits.
//!
//! Every comparison prints the largest and the mean difference. Without a GPU adapter the tests print why and
//! skip, unless the environment variable REQUIRE_GPU is set, in which case they fail.

mod support_texture;

use cloud_map_generator_core::generate::{along as cpu_along, moisture_profile as cpu_moisture};
use cloud_map_generator_core::noise::fbm as cpu_fbm;
use cloud_map_generator_core::rng::Rng;
use cloud_map_generator_core::texture::{regime_texture, regime_weights, standardize as cpu_standardize, RegimeWeights};
use cloud_map_generator_gpu::stages::{Moments, StageKernels, SynopticParams};
use cloud_map_generator_gpu::texture::{LatInput, TextureKernels, TextureMoments, WeightBuffers};
use support_texture::{diff_stats, dirs_diff_stats, fixture, fixture_plans_only, gpu_or_skip, webgpu_default_context};

/// Prints a comparison line and fails when the largest difference exceeds `bound`.
fn check(name: &str, (max, mean): (f32, f64), bound: f32) {
    println!("  {name:<46} max |diff| {max:10.3e}   mean |diff| {mean:10.3e}   (bound {bound:.1e})");
    assert!(max <= bound, "{name}: max |diff| {max} exceeds {bound}");
}

#[test]
fn kernels_match_the_cpu_port_at_1m_texels() {
    let Some(gpu) = gpu_or_skip("kernels_match_the_cpu_port_at_1m_texels") else { return };
    let f = fixture(1448); // 1448 x 724 = 1,048,352 texels
    let (w, h, n) = (f.width, f.height, f.n);
    let tk = TextureKernels::new(&gpu);
    let sk = tk.stages();
    println!("{n} texels, {w} x {h}");

    // grid_dirs
    check("grid_dirs", dirs_diff_stats(&sk.grid_dirs_to_vec(w, h, 0, h), &f.p), 1.5e-7);

    // upsample_dirs of the coarse back-trace grid (512 x 256 -> 1448 x 724)
    let coarse = sk.upload_dirs(&f.coarse);
    let back_buf = sk.upsample_dirs_to_buffer(&coarse, f.coarse_w, f.coarse_h, w, h, 0, h);
    check("upsample_dirs", dirs_diff_stats(&sk.dirs_to_vec(&back_buf, n), &f.back), 5e-7);

    // along(p, back, advect_mid)
    let (p_buf, back_up) = (sk.upload_dirs(&f.p), sk.upload_dirs(&f.back));
    let mid = f.params.advect_mid;
    let mid_gpu = sk.dirs_to_vec(&sk.along_to_buffer(&p_buf, &back_up, n, mid), n);
    check("along(p, back, advect_mid)", dirs_diff_stats(&mid_gpu, &cpu_along(&f.p, &f.back, mid)), 3e-7);

    // lat_of_dirs: asin(clip(z, -1, 1)) of the back-traced direction
    let lat_cpu: Vec<f32> = f.back.iter().map(|b| b[2].clamp(-1.0, 1.0).asin()).collect();
    let lat_buf = sk.lat_of_dirs_to_buffer(&back_up, n);
    check("lat_of_dirs (polynomial asin)", diff_stats(&sk.field_to_vec(&lat_buf, n), &lat_cpu), 5e-7);

    // moisture_profile of the CPU latitude (the same input on both sides)
    let profile_cpu = cpu_moisture(&lat_cpu, f.hadley_deg);
    let lat_in = sk.upload_field(&lat_cpu);
    let profile_gpu = sk.field_to_vec(&sk.moisture_profile_to_buffer(&lat_in, n, f.hadley_deg), n);
    check("moisture_profile", diff_stats(&profile_gpu, &profile_cpu), 5e-7);

    // synoptic field s before standardization, from the CPU's own fBm and ascent fields on both sides
    let moisture_noise = cpu_fbm(&f.back, &f.plans.moisture);
    let ascent = f.plans.flow.ascent(&f.p);
    let mid_noise = cpu_fbm(&mid_gpu_cpu(&f), &f.plans.mid);
    let g = &f.params;
    let (a_m, k_n, a_a, a_mid) = (g.a_moisture as f32, g.moisture_noise as f32, g.a_ascent as f32, g.a_mid as f32);
    let mut s_cpu: Vec<f32> = (0..n).map(|i| a_m * (profile_cpu[i] + k_n * moisture_noise[i])).collect();
    for (v, a) in s_cpu.iter_mut().zip(&ascent) {
        *v += a_a * *a;
    }
    for (v, m) in s_cpu.iter_mut().zip(&mid_noise) {
        *v += a_mid * *m;
    }
    let sp = SynopticParams { hadley_deg: f.hadley_deg, a_moisture: g.a_moisture, moisture_noise: g.moisture_noise, a_ascent: g.a_ascent, a_mid: g.a_mid };
    let (m_in, a_in, mid_in) = (sk.upload_field(&moisture_noise), sk.upload_field(&ascent), sk.upload_field(&mid_noise));
    let s_buf = sk.synoptic_to_buffer(&lat_in, &m_in, &a_in, &mid_in, n, &sp);
    check("synoptic s (before standardization)", diff_stats(&sk.field_to_vec(&s_buf, n), &s_cpu), 5e-6);

    // mean / std and standardize: the GPU's f64-combined statistics against the exact ones and against numpy's
    let moments = sk.moments(&s_buf, n);
    let exact = Moments::of_slice(&s_cpu);
    let (cpu_mean, cpu_std) = (cloud_map_generator_core::matching::ndimage::mean(&s_cpu), cloud_map_generator_core::matching::ndimage::std(&s_cpu, 0));
    println!("  s: GPU mean {:.9} std {:.9}   exact f64 mean {:.9} std {:.9}   CPU port (numpy f32) mean {cpu_mean:.9} std {cpu_std:.9}", moments.mean, moments.std(), exact.mean, exact.std());
    assert!((moments.std() / exact.std() - 1.0).abs() < 1e-6 && (moments.mean - exact.mean).abs() < 1e-6 * exact.std());
    let s_std = sk.field_to_vec(&sk.standardize_to_buffer(&s_buf, n, &moments), n);
    check("standardized s", diff_stats(&s_std, &cpu_standardize(&s_cpu)), 5e-6);

    // regime_weights: the CPU's float64 latitude grid with the float32 back-traced latitude (what generate() calls)
    let lat_grid: Vec<f64> = f.lat_rows.iter().flat_map(|&l| std::iter::repeat_n(l, w)).collect();
    let cpu_w = regime_weights::<f64, f32, f64>(&lat_grid, &lat_cpu, f.hadley_deg);
    let rows = sk.upload_field(&sk.grid_lat_rows(w, h, 0, h));
    let wb = tk.regime_weights_to_buffers(LatInput::Rows { lat: &rows, width: w }, &lat_in, n, f.hadley_deg);
    let get = |b: &wgpu::Buffer| sk.field_to_vec(b, n);
    let f32s = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    check("regime_weights closed (rows, f32 chain)", diff_stats(&get(&wb.closed), &f32s(&cpu_w.closed)), 5e-6);
    check("regime_weights open", diff_stats(&get(&wb.open), &f32s(&cpu_w.open)), 5e-6);
    check("regime_weights street", diff_stats(&get(&wb.street), &f32s(&cpu_w.street)), 5e-6);

    // regime_texture: the CPU's weights (f64) on both sides (f32 for the GPU), the directions of the map
    let cpu_tex = regime_texture(&f.p, &f.back, &cpu_w, f.radius_km, f.fmax, &f.texture_plans);
    let weights32 = WeightBuffers { closed: sk.upload_field(&f32s(&cpu_w.closed)), open: sk.upload_field(&f32s(&cpu_w.open)), street: sk.upload_field(&f32s(&cpu_w.street)) };
    let tex_buf = tk.regime_texture_to_buffer(&p_buf, &back_up, &weights32, n, f.radius_km, f.fmax, &f.texture_plans);
    let tex_gpu = sk.field_to_vec(&tex_buf, n);
    check("regime_texture (same weights)", diff_stats(&tex_gpu, &cpu_tex), 2e-3);
    let tex_moments = sk.moments(&tex_buf, n);
    println!("  texture: GPU mean {:.4} std {:.4}   CPU mean {:.4} std {:.4}", tex_moments.mean, tex_moments.std(), Moments::of_slice(&cpu_tex).mean, Moments::of_slice(&cpu_tex).std());
    // the pipeline as it runs on the GPU: the GPU's own directions and weights (no CPU input at all)
    let tex_end = tk.regime_texture_to_buffer(&sk.grid_dirs_to_buffer(w, h, 0, h), &back_buf, &wb, n, f.radius_km, f.fmax, &f.texture_plans);
    check("regime_texture (GPU directions and weights)", diff_stats(&sk.field_to_vec(&tex_end, n), &cpu_tex), 3e-3);
    let _: RegimeWeights<f64> = cpu_w;
}

/// `along(p, back, advect_mid)` on the CPU (the directions the mid-scale fBm reads).
fn mid_gpu_cpu(f: &support_texture::Fixture) -> Vec<[f32; 3]> {
    cpu_along(&f.p, &f.back, f.params.advect_mid)
}

/// A field with a mean and a spread, as the pipeline's fields are, made with the CPU port's generator: a smooth
/// part (a few low-frequency waves, like the synoptic field) plus white noise, then scaled and shifted.
fn test_field(n: usize, mean: f32, std: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let x = i as f32 * 1.7e-5;
            mean + std * (0.8 * (x.sin() + 0.5 * (3.1 * x).cos()) + 0.6 * rng.normal() as f32)
        })
        .collect()
}

/// The reduction against the exact f64 statistics of the same f32 values, at the sizes of 4096 x 2048 (8,388,608
/// texels) and 8192 x 4096 (33,554,432 texels, a 128 MiB field: the whole WebGPU binding limit), for fields of
/// different mean/std ratios. Relative error: `|std_gpu / std_f64 - 1|`, and for the mean `|mean_gpu - mean_f64|`
/// in units of the std (a mean near 0 has no meaningful relative error) and relative to `|mean|` where it is not
/// tiny. Required below 1e-6.
#[test]
fn reduction_error_against_f64_at_8m_and_33m_texels() {
    let Some(gpu) = gpu_or_skip("reduction_error_against_f64_at_8m_and_33m_texels") else { return };
    let sk = StageKernels::new(&gpu);
    let mut worst = 0.0f64;
    for (label, n) in [("4096 x 2048", 4096usize * 2048), ("8192 x 4096", 8192usize * 4096)] {
        for (mean, std, seed) in [(0.0f32, 1.0f32, 1u64), (0.37, 1.3, 2), (5.0, 0.5, 3), (-40.0, 3.0, 4)] {
            let v = test_field(n, mean, std, seed);
            let buf = sk.upload_field(&v);
            let t = std::time::Instant::now();
            let got = sk.moments(&buf, n);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let exact = Moments::of_slice(&v);
            let std_rel = (got.std() / exact.std() - 1.0).abs();
            let mean_in_std = (got.mean - exact.mean).abs() / exact.std();
            let mean_rel = if exact.mean.abs() > 0.05 * exact.std() { (got.mean / exact.mean - 1.0).abs() } else { 0.0 };
            println!(
                "  {label:<12} {n:>10} texels  field mean {mean:6.2} std {std:4.1}: std rel err {std_rel:9.2e}   mean err {mean_in_std:9.2e} of std, {mean_rel:9.2e} relative   ({ms:.1} ms with readback)"
            );
            worst = worst.max(std_rel).max(mean_in_std).max(mean_rel);
            assert!(std_rel < 1e-6, "{label} mean {mean}: std relative error {std_rel}");
            assert!(mean_in_std < 1e-6, "{label} mean {mean}: mean error {mean_in_std} of the std");
            assert!(mean_rel < 1e-6, "{label} mean {mean}: mean relative error {mean_rel}");
        }
    }
    println!("worst relative error of the reduction: {worst:.2e} (required < 1e-6)");
}

/// The same kernels with the chunk limit forced to 64 KiB (instead of 128 MiB) give the same maps: bit for bit
/// for the elementwise kernels (a chunk boundary changes no arithmetic) and for the moments (the chunks are
/// whole tiles).
#[test]
fn chunking_forced_small_changes_nothing_at_1m_texels() {
    let Some(gpu) = gpu_or_skip("chunking_forced_small_changes_nothing_at_1m_texels") else { return };
    let f = fixture(1448);
    let (w, h, n) = (f.width, f.height, f.n);
    let big = TextureKernels::new(&gpu);
    let small = TextureKernels::with_stages(&gpu, StageKernels::with_binding_bytes(&gpu, 64 * 1024));
    let (bk, sk) = (big.stages(), small.stages());
    assert!(sk.rows_per_chunk(w, 3) < 6, "64 KiB holds fewer than 6 rows of vec3 at this width");
    assert_eq!(bk.grid_dirs_to_vec(w, h, 0, h), sk.grid_dirs_to_vec(w, h, 0, h));
    // the coarse grid is bound whole (the pipeline's is 512 x 256 = 1.5 MB): a 64 x 32 grid fits the 64 KiB limit
    let coarse = f.plans.flow.backtrace_default(&cloud_map_generator_core::sphere::grid_dirs::<f64>(64, 32));
    assert_eq!(bk.upsample_dirs_to_vec(&coarse, 64, 32, w, h), sk.upsample_dirs_to_vec(&coarse, 64, 32, w, h));
    assert_eq!(bk.along_to_vec(&f.p, &f.back, 0.6), sk.along_to_vec(&f.p, &f.back, 0.6));
    let lat: Vec<f32> = f.back.iter().map(|b| b[2].clamp(-1.0, 1.0).asin()).collect();
    assert_eq!(bk.moisture_profile_to_vec(&lat, f.hadley_deg), sk.moisture_profile_to_vec(&lat, f.hadley_deg));
    let s: Vec<f32> = test_field(n, 0.4, 1.2, 9);
    assert_eq!(bk.moments_of_slice(&s), sk.moments_of_slice(&s));
    assert_eq!(bk.standardize_to_vec(&s), sk.standardize_to_vec(&s));
    let lat_in = [bk.upload_field(&lat), sk.upload_field(&lat)];
    let (wb, ws) = (
        big.regime_weights_to_buffers(LatInput::Texels(&lat_in[0]), &lat_in[0], n, f.hadley_deg),
        small.regime_weights_to_buffers(LatInput::Texels(&lat_in[1]), &lat_in[1], n, f.hadley_deg),
    );
    assert_eq!(bk.field_to_vec(&wb.closed, n), sk.field_to_vec(&ws.closed, n));
    assert_eq!(bk.field_to_vec(&wb.street, n), sk.field_to_vec(&ws.street, n));
    // the texture end to end (the noise kernels chunk on their own limit; the stage kernels around them on 64 KiB)
    let p = [bk.upload_dirs(&f.p), sk.upload_dirs(&f.p)];
    let b = [bk.upload_dirs(&f.back), sk.upload_dirs(&f.back)];
    let tb = big.regime_texture_to_buffer(&p[0], &b[0], &wb, n, f.radius_km, f.fmax, &f.texture_plans);
    let ts = small.regime_texture_to_buffer(&p[1], &b[1], &ws, n, f.radius_km, f.fmax, &f.texture_plans);
    assert_eq!(bk.field_to_vec(&tb, n), sk.field_to_vec(&ts, n));
    println!("chunks of 64 KiB (1024 items): every kernel gives the bits of the unchunked run");
}

/// The map of one band, on the GPU, under the WebGPU default limits (a device created with them): the texture of a
/// `width x height` Earth map driven in bands of `band_rows` rows, the way the pipeline does it. Returns the
/// texture of every band read back.
fn texture_in_bands(gpu: &cloud_map_generator_gpu::GpuContext, f: &support_texture::Fixture, band_rows: usize) -> Vec<f32> {
    let tk = TextureKernels::new(gpu);
    let sk = tk.stages();
    let (w, h) = (f.width, f.height);
    let coarse = sk.upload_dirs(&f.coarse);
    let band_of = |row0: usize| (row0, band_rows.min(h - row0));
    // pass 1 of every band: directions, back-traced directions, latitude, weights, the raw texture fields and their moments
    struct Band {
        n: usize,
        weights: WeightBuffers,
        fields: cloud_map_generator_gpu::texture::TextureFields,
    }
    let mut bands = Vec::new();
    let mut moments = TextureMoments::EMPTY;
    for row0 in (0..h).step_by(band_rows) {
        let (row0, rows) = band_of(row0);
        let n = w * rows;
        let p = sk.grid_dirs_to_buffer(w, h, row0, rows);
        let back = sk.upsample_dirs_to_buffer(&coarse, f.coarse_w, f.coarse_h, w, h, row0, rows);
        let lat_back = sk.lat_of_dirs_to_buffer(&back, n);
        let lat_rows = sk.upload_field(&sk.grid_lat_rows(w, h, row0, rows));
        let weights = tk.regime_weights_to_buffers(LatInput::Rows { lat: &lat_rows, width: w }, &lat_back, n, f.hadley_deg);
        let fields = tk.texture_fields(&p, &back, n, f.radius_km, f.fmax, &f.texture_plans);
        moments = moments.merge(tk.texture_moments(&fields));
        bands.push(Band { n, weights, fields });
    }
    // pass 2: the blend with the statistics of the whole map
    let mut out = Vec::with_capacity(w * h);
    for b in &bands {
        out.extend(sk.field_to_vec(&tk.texture_blend(&b.fields, &b.weights, &moments), b.n));
    }
    out
}

/// A 4096 x 2048 map in bands of 512 and of 700 rows on a device with exactly the WebGPU default limits, against
/// the whole map as one band (100 MB of directions: within the limits): banding changes no texel beyond the f32
/// noise of the merged statistics. A buffer or a binding past a default limit would be a validation error.
#[test]
fn a_4096_map_in_bands_under_the_webgpu_default_limits() {
    let Some(native) = gpu_or_skip("a_4096_map_in_bands_under_the_webgpu_default_limits") else { return };
    drop(native);
    let Some(gpu) = webgpu_default_context() else { panic!("a device with the WebGPU default limits could not be created") };
    assert_eq!(gpu.limits.max_storage_buffer_binding_size, 128 << 20);
    assert_eq!(gpu.limits.max_buffer_size, 256 << 20);
    let f = fixture(4096);
    let whole = texture_in_bands(&gpu, &f, f.height);
    for band_rows in [512usize, 700] {
        let banded = texture_in_bands(&gpu, &f, band_rows);
        let (max, mean) = diff_stats(&whole, &banded);
        println!("  4096 x 2048 in bands of {band_rows} rows vs one band: max |diff| {max:.3e}, mean {mean:.3e}");
        assert!(max < 2e-5, "bands of {band_rows} rows change the texture by {max}");
    }
    let m = Moments::of_slice(&whole);
    println!("  texture of the 4096 map: mean {:.4}, std {:.4}", m.mean, m.std());
    assert!(m.mean.abs() < 0.2 && m.std() > 0.5 && m.std() < 1.5, "a unit-variance texture, got mean {} std {}", m.mean, m.std());
}

/// An 8192 x 4096 map (33.5M texels: a direction field of 402 MB, past the 256 MiB of `maxBufferSize`) driven in
/// bands under the WebGPU default limits: the stage kernels, the weights and the texture. The CPU is not asked
/// for a reference at this size; the bands (1024 and 1365 rows) must agree with each other.
#[test]
fn an_8192_map_in_bands_under_the_webgpu_default_limits() {
    let Some(native) = gpu_or_skip("an_8192_map_in_bands_under_the_webgpu_default_limits") else { return };
    drop(native);
    let Some(gpu) = webgpu_default_context() else { panic!("a device with the WebGPU default limits could not be created") };
    let stages = StageKernels::new(&gpu);
    assert_eq!(stages.rows_per_chunk(8192, 3), 1365, "a vec3 band of 8192 columns holds 1365 rows");
    // The whole map as one band is refused by the limits; the bands the API proposes work.
    let f = fixture_plans_only(8192);
    let a = texture_in_bands(&gpu, &f, 1024);
    let b = texture_in_bands(&gpu, &f, stages.rows_per_chunk(8192, 3));
    let (max, mean) = diff_stats(&a, &b);
    println!("  8192 x 4096 in bands of 1024 and of 1365 rows: max |diff| {max:.3e}, mean {mean:.3e}");
    assert!(max < 2e-5, "{max}");
    let m = Moments::of_slice(&a);
    println!("  texture of the 8192 map: mean {:.4}, std {:.4}", m.mean, m.std());
    assert!(m.mean.abs() < 0.2 && m.std() > 0.5 && m.std() < 1.5, "mean {} std {}", m.mean, m.std());
}
