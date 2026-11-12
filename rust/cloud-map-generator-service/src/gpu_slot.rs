//! THE GPU SLOT: where the GPU pipeline plugs into `cloud-map-generator serve`.
//!
//! `cloud-map-generator serve` probes the GPU at start-up ([`super::gpu_probe`]) and calls [`build`] with the device it found.
//! [`build`] makes the [`GpuEngine`] (the pipeline of `cloud_map_generator_gpu::pipeline` behind the [`Engine`] trait) and
//! checks it by running a small map: an engine that cannot make that map is reported unavailable, with the
//! reason, instead of failing the first request. While `build` returns `Err`, the `gpu` compute mode is
//! unavailable with that message as its reason, `hybrid` is too, `auto` runs on the CPU, and a request for `gpu`
//! is a 400 carrying the reason: no silent fallback.

use crate::engine::Engine;
use crate::gpu_engine::GpuEngine;
use cloud_map_generator_gpu::df64::Df64Config;
use cloud_map_generator_gpu::GpuContext;
use std::sync::Arc;

/// The GPU engine for `ctx` (the device the probe opened) and the double-single configuration the probe found
/// exact on it (`None`: the kernels stay in f32; none of them needs it today), or why there is none.
///
/// The context lives as long as the process: the kernels borrow it, and a server keeps them for its whole life,
/// so it is leaked once, here (the probe runs once per process).
pub fn build(ctx: GpuContext, df64: Option<Df64Config>) -> Result<Arc<dyn Engine>, String> {
    let _ = df64;
    let ctx: &'static GpuContext = Box::leak(Box::new(ctx));
    GpuEngine::new(ctx).map(|engine| Arc::new(engine) as Arc<dyn Engine>)
}
