//! Shared fixture of `gpu_texture_accuracy.rs` and `bench_gpu_texture.rs`: the inputs the pipeline hands to the
//! stage and texture kernels for an Earth map of a given width, made by the CPU port (the directions of the grid,
//! the coarse back-trace of the flow and its upsampling, the plans of `generate()`).

#![allow(dead_code)]

use cloud_map_generator_core::climate::{hadley_edge_deg, EARTH};
use cloud_map_generator_core::generate::{fine_ladder, texture_fmax, GenParams, GeneratePlans, TexturePlans};
use cloud_map_generator_core::sphere::{grid_dirs, grid_lonlat, upsample_dirs};
use cloud_map_generator_core::texture::RegimeTexturePlans;
use cloud_map_generator_gpu::GpuContext;
use std::time::Instant;

/// The adapter, or `None` (with the reason printed) when there is none; with `REQUIRE_GPU` set a missing adapter
/// fails the test instead.
pub fn gpu_or_skip(test: &str) -> Option<GpuContext> {
    match GpuContext::new() {
        Some(g) => {
            println!("{test}: adapter {} ({:?}), driver {} {}", g.adapter_info.name, g.adapter_info.backend, g.adapter_info.driver, g.adapter_info.driver_info);
            Some(g)
        }
        None if std::env::var_os("REQUIRE_GPU").is_some() => panic!("{test}: no GPU adapter and REQUIRE_GPU is set"),
        None => {
            println!("{test}: no GPU adapter (wgpu found none): skipped; set REQUIRE_GPU to fail instead");
            None
        }
    }
}

/// An adapter with exactly the WebGPU default limits requested (not the adapter's own): what a browser gives
/// by default. A binding or a buffer past them is a validation error (a panic in a test).
pub fn webgpu_default_context() -> Option<GpuContext> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok()?;
    let limits = wgpu::Limits::default();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cloudgen webgpu defaults"),
        required_features: wgpu::Features::empty(),
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .ok()?;
    Some(GpuContext { device, queue, adapter_info: adapter.get_info(), limits })
}

/// Everything a pipeline run of an Earth map of `width x width / 2` gives the kernels, from the CPU port.
pub struct Fixture {
    pub width: usize,
    pub height: usize,
    pub n: usize,
    pub params: GenParams,
    pub plans: GeneratePlans,
    pub texture_plans: RegimeTexturePlans,
    pub hadley_deg: f64,
    pub radius_km: f64,
    /// The `fmax` handed to `regime_texture`.
    pub fmax: f64,
    /// Latitude of every row (f64, radians).
    pub lat_rows: Vec<f64>,
    /// Texel directions (`grid_dirs::<f32>`).
    pub p: Vec<[f32; 3]>,
    /// The coarse back-trace grid (f32 directions) and its size.
    pub coarse: Vec<[f32; 3]>,
    pub coarse_w: usize,
    pub coarse_h: usize,
    /// The coarse grid upsampled to the map (`upsample_dirs`): where the air at each texel came from.
    pub back: Vec<[f32; 3]>,
}

/// The fixture with the texel directions and the upsampled back-trace made by the CPU port.
pub fn fixture(width: usize) -> Fixture {
    build(width, true)
}

/// The fixture without `p`, `back` and `lat_rows` (empty): for a run whose GPU makes them, where the CPU's
/// upsampling of a very large map is not wanted.
pub fn fixture_plans_only(width: usize) -> Fixture {
    build(width, false)
}

fn build(width: usize, with_maps: bool) -> Fixture {
    let t = Instant::now();
    let params = GenParams::default();
    let height = width / 2;
    let plans = GeneratePlans::from_seed(&EARTH, 7, width, &params);
    let texture_plans = match &plans.texture {
        TexturePlans::Regime(tp) => tp.clone(),
        TexturePlans::Plain(_) => panic!("the default parameters use the regime texture"),
    };
    let ladder = fine_ladder(width, &params);
    let fmax = texture_fmax(ladder.fmax, &params);
    let cw = params.structure_width.min(width);
    let coarse_f64 = grid_dirs::<f64>(cw, cw / 2);
    let coarse = plans.flow.backtrace_default(&coarse_f64);
    let (p, back) = if with_maps {
        let widened: Vec<[f64; 3]> = coarse.iter().map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]).collect();
        (grid_dirs::<f32>(width, height), upsample_dirs(&widened, cw / 2, cw, width, height))
    } else {
        (Vec::new(), Vec::new())
    };
    println!("fixture {width} x {height}: made by the CPU port in {:.1} s ({} octaves of fine fBm, {} cell diameters)", t.elapsed().as_secs_f64(), texture_plans.generic.octaves.len(), texture_plans.cells.len());
    Fixture {
        width,
        height,
        n: width * height,
        hadley_deg: hadley_edge_deg(&EARTH),
        radius_km: EARTH.radius_km,
        fmax,
        lat_rows: if with_maps { grid_lonlat(width, height).1 } else { Vec::new() },
        p,
        coarse,
        coarse_w: cw,
        coarse_h: cw / 2,
        back,
        params,
        plans,
        texture_plans,
    }
}

/// `(max, mean)` of `|a - b|` over finite pairs.
pub fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f64) {
    assert_eq!(a.len(), b.len());
    let (mut max, mut sum) = (0.0f32, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let e = (x - y).abs();
        max = max.max(e);
        sum += f64::from(e);
    }
    (max, sum / a.len().max(1) as f64)
}

/// [`diff_stats`] of two direction fields (over the 3 components).
pub fn dirs_diff_stats(a: &[[f32; 3]], b: &[[f32; 3]]) -> (f32, f64) {
    let (fa, fb): (Vec<f32>, Vec<f32>) = (a.iter().flatten().copied().collect(), b.iter().flatten().copied().collect());
    diff_stats(&fa, &fb)
}
