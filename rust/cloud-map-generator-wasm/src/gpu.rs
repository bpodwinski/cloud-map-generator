//! The GPU engine in the browser: `cloud_map_generator_gpu::pipeline` over wgpu's WebGPU backend, awaited from a Web Worker.
//!
//! The same code as the native engine (`cloud-map-generator-service/src/gpu_engine.rs`): the per-texel stages as WGSL compute
//! shaders in float32, the plans, the coarse grid of the back-trace, the wind, the smoothing of the row thresholds and
//! the sidecar on the CPU. The pipeline reads the GPU back at a handful of points (the moments of the fields, the row
//! thresholds, the finished rows); a browser cannot block on the GPU, so those are awaited (`cloud-map-generator-gpu` is written
//! once with `maybe-async`, plain blocking calls natively and futures here).
//!
//! Compiled for `wasm32` only. The context is opened once per instance by [`gpu_probe`] and kept for the instance's life.

use crate::finalize::{finish, now_ms, params_of, Run};
use crate::{parse, parse_caps, Finished};
use cloud_map_generator_core::generate::GenParams;
use cloud_map_generator_gpu::df64;
use cloud_map_generator_gpu::pipeline::{GpuBandState, GpuPipeline};
use cloud_map_generator_gpu::GpuContext;
use cloud_map_generator_service::engine::{ComputeUse, DeviceKind, DeviceUse, SurfaceChoice, Used};
use cloud_map_generator_service::pyjson::J;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

/// What the answer says about the GPU's sub-texel grain (the one place a GPU map and the CPU map of a seed differ by more
/// than float32 rounding): the same sentence as the native engine's.
const GRAIN_NOTE: &str = "The sub-texel grain of the GPU's cumulus comes from the GPU's own counter-based random numbers: this map is the CPU map of the seed up to float32 rounding, except for that grain.";

/// The pipeline of this instance, its kernels compiled once.
struct Gpu {
    name: String,
    pipeline: GpuPipeline<'static>,
}

thread_local! {
    static GPU: RefCell<Option<Rc<Gpu>>> = const { RefCell::new(None) };
}

fn df64_label(config: Option<df64::Df64Config>) -> Option<String> {
    config.map(|c| format!("{}{}", if c.fma { "fma" } else { "split" }, if c.barrier { "+barrier" } else { "" }))
}

fn backend_label(debug_name: &str) -> String {
    match debug_name {
        "BrowserWebGpu" => "WebGPU",
        other => other,
    }
    .to_string()
}

/// The adapter's name: the description when the browser gives one, else the page's hint (the vendor and architecture the
/// browser's `GPUAdapterInfo` reveals, which is all a browser tells), else its driver text, else a generic label.
fn adapter_name(ctx: &GpuContext, hint: Option<&str>) -> String {
    let info = &ctx.adapter_info;
    if !info.name.trim().is_empty() {
        info.name.trim().to_string()
    } else if let Some(h) = hint.map(str::trim).filter(|h| !h.is_empty()) {
        h.to_string()
    } else {
        let driver = [info.driver.trim(), info.driver_info.trim()].into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join(" ");
        if driver.is_empty() {
            "WebGPU adapter".to_string()
        } else {
            driver
        }
    }
}

/// Opens the WebGPU adapter and device, runs the double-single probe on it, compiles the pipeline and checks it by making a
/// small map (the contract: a mode is available only when the engine has just verified it can run it here).
///
/// Returns `{"name", "backend", "df64", "df64_configs", "limits", "probe_seconds"}` as JSON, or `Err(reason)`: a sentence a
/// visitor can act on.
#[wasm_bindgen]
pub async fn gpu_probe(name_hint: Option<String>) -> Result<String, String> {
    let t0 = now_ms();
    let Some(ctx) = GpuContext::new_async().await else {
        return Err("WebGPU is present, but the browser gave no adapter (the GPU may be blocklisted, or hardware acceleration is off). Use the CPU, or the native engine (cloud-map-generator serve).".to_string());
    };
    let name = adapter_name(&ctx, name_hint.as_deref());
    // The double-single probe: which configuration of the library is exact on this adapter (informational: the kernels run in f32).
    let probe = df64::probe(&ctx).await;
    let best = probe.best();
    let limits = ctx.limits.clone();
    let backend = backend_label(&format!("{:?}", ctx.adapter_info.backend));
    // The kernels borrow the context for the life of the instance, as the native engine's slot does.
    let ctx: &'static GpuContext = Box::leak(Box::new(ctx));
    let pipeline = GpuPipeline::new(ctx).await;
    pipeline.warm_up().await;
    GPU.with(|g| *g.borrow_mut() = Some(Rc::new(Gpu { name: name.clone(), pipeline })));
    let configs = probe
        .results
        .iter()
        .map(|r| {
            J::obj([
                ("config", J::s(df64_label(Some(r.config)).unwrap_or_default())),
                ("two_sum_exact", J::Bool(r.two_sum_exact)),
                ("two_prod_exact", J::Bool(r.two_prod_exact)),
            ])
        })
        .collect();
    let answer = J::obj([
        ("name", J::s(name)),
        ("backend", J::s(backend)),
        ("df64", df64_label(best).map_or(J::Null, J::Str)),
        ("df64_configs", J::Arr(configs)),
        (
            "limits",
            J::obj([
                ("max_buffer_size", J::Float(limits.max_buffer_size as f64)),
                ("max_storage_buffer_binding_size", J::Float(f64::from(limits.max_storage_buffer_binding_size))),
                ("max_compute_workgroup_size_x", J::Int(i64::from(limits.max_compute_workgroup_size_x))),
                ("max_compute_invocations_per_workgroup", J::Int(i64::from(limits.max_compute_invocations_per_workgroup))),
            ]),
        ),
        ("probe_seconds", J::Float((now_ms() - t0) / 1000.0)),
    ]);
    Ok(cloud_map_generator_service::json::compact(&answer))
}

/// A whole generation on the GPU, then the answer, in this instance (a Web Worker). Needs [`gpu_probe`] to have succeeded.
/// `note` is the sentence of an `auto` choice; the sentence about the sub-texel grain is added to it.
#[wasm_bindgen]
pub async fn run_gpu(request_json: String, caps_json: String, id: String, note: Option<String>, elapsed_before: f64) -> Result<Finished, String> {
    let caps = parse_caps(&caps_json)?;
    let req = parse(&request_json, &caps).map_err(|e| e.message)?;
    let gpu = GPU.with(|g| g.borrow().clone()).ok_or("the GPU was not probed in this worker: call gpu_probe first")?;
    let params: GenParams = params_of(&req.options);
    let surface = (req.options.surface == SurfaceChoice::Synthetic).then(|| cloud_map_generator_service::synthetic::synthetic_surface(req.width.min(1024)));
    let t0 = now_ms();
    let (output, timings) = gpu
        .pipeline
        .generate_seeded(&req.body, req.seed, req.width, &params, surface.as_ref(), true, &mut |_, _| {})
        .await
        .map_err(|e| format!("the GPU pipeline refused the request: {e}"))?;
    let seconds = (now_ms() - t0) / 1000.0;
    // The time of the call splits into what the CPU stages took (plans, coarse grid, wind, smoothing, sidecar) and the rest
    // (uploads, kernels, the wait and the readbacks); the shares are those of the engine's time.
    let (cpu, gpu_seconds) = (timings.cpu_seconds.min(seconds), (seconds - timings.cpu_seconds).max(0.0));
    let total = (cpu + gpu_seconds).max(f64::MIN_POSITIVE);
    let compute = ComputeUse {
        used: Used::Gpu,
        devices: vec![
            DeviceUse { kind: DeviceKind::Gpu, name: Some(gpu.name.clone()), seconds: gpu_seconds, share: gpu_seconds / total },
            DeviceUse { kind: DeviceKind::Cpu, name: None, seconds: cpu, share: cpu / total },
        ],
        note: Some(match note {
            Some(chose) => format!("{chose} {GRAIN_NOTE}"),
            None => GRAIN_NOTE.to_string(),
        }),
    };
    let inner = finish(Run { req: &req, id: &id, compute, engine_seconds: seconds, elapsed_before: elapsed_before + seconds }, output)?;
    Ok(Finished { inner })
}

// ---------------------------------------------------------------------------------------------------------
// The GPU as a unit of the banded engine (see `crate::large`): the native GPU band worker's passes, awaited

thread_local! {
    /// What the banded run keeps on the device between bands (the vortex table, the coarse back-trace).
    static BAND_STATE: RefCell<Option<Rc<GpuBandState>>> = const { RefCell::new(None) };
}

fn band_unit(band: usize) -> Result<(Rc<Gpu>, Rc<cloud_map_generator_core::banded::Banded<'static>>, Rc<GpuBandState>, (usize, usize)), String> {
    let gpu = GPU.with(|g| g.borrow().clone()).ok_or("the GPU was not probed in this worker: call gpu_probe first")?;
    let banded = crate::large::banded()?;
    let state = BAND_STATE.with(|s| s.borrow().clone()).ok_or("the GPU's banded run is not open: call large_gpu_open")?;
    let range = crate::large::check_band(&banded, band)?;
    Ok((gpu, banded, state, range))
}

/// Opens the GPU side of the banded run this worker opened with `large_open`.
#[wasm_bindgen]
pub fn large_gpu_open() -> Result<(), String> {
    let gpu = GPU.with(|g| g.borrow().clone()).ok_or("the GPU was not probed in this worker: call gpu_probe first")?;
    let banded = crate::large::banded()?;
    let state = GpuBandState::new(&gpu.pipeline, &banded)?;
    BAND_STATE.with(|s| *s.borrow_mut() = Some(Rc::new(state)));
    Ok(())
}

#[wasm_bindgen]
pub fn large_gpu_close() {
    BAND_STATE.with(|s| *s.borrow_mut() = None);
}

/// Pass A of a band on the GPU (the same flat form as `large_pass_a`).
#[wasm_bindgen]
pub async fn large_gpu_pass_a(band: usize) -> Result<Vec<f64>, String> {
    let (gpu, banded, state, (r0, r1)) = band_unit(band)?;
    let m = state.pass_a(&gpu.pipeline, &banded, r0, r1).await;
    Ok(crate::large::moments_to_flat(&m))
}

#[wasm_bindgen]
pub async fn large_gpu_pass_b(band: usize) -> Result<Vec<f64>, String> {
    let (gpu, banded, state, (r0, r1)) = band_unit(band)?;
    let norm = crate::large::norm()?;
    let t = state.pass_b(&gpu.pipeline, &banded, &norm, r0, r1).await;
    Ok(crate::large::thresholds_to_flat(&t))
}

#[wasm_bindgen]
pub async fn large_gpu_pass_c(band: usize) -> Result<Vec<f32>, String> {
    let (gpu, banded, state, (r0, r1)) = band_unit(band)?;
    let (norm, thr) = (crate::large::norm()?, crate::large::thresholds()?);
    let r = state.pass_c(&gpu.pipeline, &banded, &norm, &thr.0, &thr.1, r0, r1).await;
    Ok(crate::large::result_to_flat(r))
}
