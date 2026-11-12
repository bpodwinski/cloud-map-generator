//! The GPU engine of `cloud-map-generator serve`: [`cloud_map_generator_gpu::pipeline`] behind the [`Engine`] trait.
//!
//! What runs where, as the compute block reports it: the per-texel stages (noise, flow back-trace and ascent,
//! regime blend, compose, per-row threshold solve, products) on the GPU in float32; the plans, the coarse grid of
//! the back-trace, the wind, the surface terms, the smoothing of the row thresholds and the sidecar on the CPU
//! (the match to Earth, the encodings and the metrics the server runs afterwards are CPU work too: the server adds
//! them to the CPU's seconds). `devices` lists both with the wall seconds each worked and the share of the engine's
//! time it took, the shares adding to 1.

use crate::engine::{cpu_name, ComputeUse, DeviceKind, DeviceUse, Engine, EngineError, GenerateRequest, Generated, Progress, Used};
use crate::bands::BandCompute;
use cloud_map_generator_core::banded::{BandMoments, BandResult, BandThresholds};
use cloud_map_generator_core::generate::NormStats;
use cloud_map_generator_gpu::pipeline::{GpuBandState, GpuPipeline};
use cloud_map_generator_gpu::GpuContext;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

/// The narrowest map, in texels, for which `auto` prefers the GPU to one CPU thread of the Rust engine.
///
/// MEASURED with `cargo run --release -p cloud-map-generator-gpu --example pipeline_report -- timing 32 64 128 256 384 512 1024
/// 2048 4096` (a whole map with its derived maps: the CPU port on one thread against the GPU pipeline end to end,
/// plans and readback included; the best of three to five runs; an RTX 5080 on Vulkan and a Ryzen other jobs were
/// using, 2026-11-11):
///
/// ```text
/// width      32     64    128    256    384    512   1024   2048   4096
/// CPU s   0.010  0.039  0.163  0.655  1.816  2.935  3.972  7.218 24.087
/// GPU s   0.007  0.008  0.010  0.018  0.033  0.045  0.159  0.176  0.338
/// ```
///
/// The GPU's fixed cost is about 8 ms (the kernels' submissions and the readback), so it already wins by 4.6 times
/// at 64 px and by more than 1.25 times (the margin used) from 32 px, below the narrowest map the API accepts.
/// `auto` therefore chooses the GPU for every width the API offers; the constant is that narrowest width, so that
/// it is the one place to raise if a slower adapter (an integrated GPU) moves the crossover. A GPU engine that
/// is slower than the CPU at some width must have a larger constant.
pub const MIN_AUTO_WIDTH: usize = 64;

/// What the answer says about the GPU's sub-texel grain: the one place where a GPU map and the CPU map of the same
/// seed are not the same map to float32 rounding (the GPU draws that noise itself, from a counter-based generator).
const GRAIN_NOTE: &str = "The sub-texel grain of the GPU's cumulus comes from the GPU's own counter-based random numbers: this map is the CPU map of the seed up to float32 rounding, except for that grain.";

/// The GPU pipeline, with its kernels compiled once and kept.
pub struct GpuEngine {
    /// The adapter's name, for the `devices` block.
    name: String,
    pipeline: Mutex<GpuPipeline<'static>>,
}

impl GpuEngine {
    /// The engine for `ctx`, after a self-test: a small map is made through the whole pipeline (which also
    /// compiles every shader, so that the first request of a client does not pay for them). `Err` says why the
    /// device cannot run the pipeline.
    pub fn new(ctx: &'static GpuContext) -> Result<GpuEngine, String> {
        let name = ctx.adapter_info.name.clone();
        let built = catch_unwind(AssertUnwindSafe(|| {
            let pipeline = GpuPipeline::new(ctx);
            pipeline.warm_up();
            pipeline
        }));
        match built {
            Ok(pipeline) => Ok(GpuEngine { name, pipeline: Mutex::new(pipeline) }),
            Err(panic) => Err(format!("The GPU pipeline failed its self-test on {name}: {}", panic_text(panic.as_ref()))),
        }
    }

    fn lock(&self) -> MutexGuard<'_, GpuPipeline<'static>> {
        // A map that panicked poisoned the lock; the pipeline holds compiled kernels only, which stay valid.
        self.pipeline.lock().unwrap_or_else(|p| p.into_inner())
    }
}

fn panic_text(panic: &(dyn std::any::Any + Send)) -> String {
    panic.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| panic.downcast_ref::<String>().cloned()).unwrap_or_else(|| "unknown error".to_string())
}

impl Engine for GpuEngine {
    fn generate(&self, req: &GenerateRequest, progress: Progress<'_>) -> Result<Generated, EngineError> {
        progress(0.0, &format!("generating a {} px map on the GPU ({})", req.width, self.name));
        let t0 = Instant::now();
        let pipeline = self.lock();
        let result = catch_unwind(AssertUnwindSafe(|| {
            pipeline.generate_seeded(&req.body, req.seed, req.width, &req.params(), req.surface.as_ref(), true, &mut |fraction, text| {
                progress(f64::from(fraction), text)
            })
        }));
        drop(pipeline);
        let (output, timings) = match result {
            Ok(Ok(done)) => done,
            Ok(Err(refused)) => return Err(refused.into()),
            Err(panic) => return Err(EngineError::Failed(format!("the GPU pipeline failed on {}: {}", self.name, panic_text(panic.as_ref())))),
        };
        let seconds = t0.elapsed().as_secs_f64();
        progress(1.0, &format!("generated in {seconds:.2} s"));
        // The time of the call splits into what the CPU stages took and the rest (the uploads, the kernels, the wait
        // and the readback). The shares are those of the engine's time.
        let (cpu, gpu) = (timings.cpu_seconds.min(seconds), (seconds - timings.cpu_seconds).max(0.0));
        let total = (cpu + gpu).max(f64::MIN_POSITIVE);
        Ok(Generated {
            output,
            compute: ComputeUse {
                used: Used::Gpu,
                devices: vec![
                    DeviceUse { kind: DeviceKind::Gpu, name: Some(self.name.clone()), seconds: gpu, share: gpu / total },
                    DeviceUse { kind: DeviceKind::Cpu, name: cpu_name(), seconds: cpu, share: cpu / total },
                ],
                note: Some(GRAIN_NOTE.to_string()),
            },
        })
    }

    fn min_auto_width(&self) -> usize {
        MIN_AUTO_WIDTH
    }

    fn band_worker<'a>(&'a self, banded: &'a cloud_map_generator_core::banded::Banded<'a>) -> Result<Box<dyn BandCompute + 'a>, String> {
        let state = {
            let pipe = self.lock();
            catch_unwind(AssertUnwindSafe(|| GpuBandState::new(&pipe, banded)))
                .map_err(|p| format!("the GPU failed to start the banded run on {}: {}", self.name, panic_text(p.as_ref())))??
        };
        Ok(Box::new(GpuBands { engine: self, banded, state }))
    }

    fn threads(&self) -> usize {
        1
    }
}

// ---------------------------------------------------------------------------------------------------------
// The banded worker

/// The GPU as a worker of the band scheduler ([`crate::bands`]): each pass of a band runs the kernels of
/// `cloud_map_generator_gpu::pipeline::bands`, on the pipeline this engine holds (locked for the length of the band: one GPU
/// queue is one unit of the scheduler). A panic of the driver is a device failure: the scheduler hands the band back.
struct GpuBands<'a> {
    engine: &'a GpuEngine,
    banded: &'a cloud_map_generator_core::banded::Banded<'a>,
    state: GpuBandState,
}

impl GpuBands<'_> {
    fn guarded<T>(&self, what: &str, f: impl FnOnce(&GpuPipeline<'static>) -> T) -> Result<T, String> {
        let pipe = self.engine.lock();
        catch_unwind(AssertUnwindSafe(|| f(&pipe))).map_err(|p| format!("the GPU failed in {what} on {}: {}", self.engine.name, panic_text(p.as_ref())))
    }
}

impl BandCompute for GpuBands<'_> {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Gpu
    }

    fn name(&self) -> Option<String> {
        Some(self.engine.name.clone())
    }

    fn pass_a(&self, band: usize) -> Result<BandMoments, String> {
        let (r0, r1) = self.banded.band_range(band);
        self.guarded("pass A", |pipe| self.state.pass_a(pipe, self.banded, r0, r1))
    }

    fn pass_b(&self, band: usize, norm: &NormStats) -> Result<BandThresholds, String> {
        let (r0, r1) = self.banded.band_range(band);
        self.guarded("pass B", |pipe| self.state.pass_b(pipe, self.banded, norm, r0, r1))
    }

    fn pass_c(&self, band: usize, norm: &NormStats, thr: &[f64], sens: &[f64]) -> Result<BandResult, String> {
        let (r0, r1) = self.banded.band_range(band);
        self.guarded("pass C", |pipe| self.state.pass_c(pipe, self.banded, norm, thr, sens, r0, r1))
    }
}
