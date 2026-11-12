//! GPU kernels of `cloudgen/flow.py` (`rust/cloud-map-generator-core/src/flow.rs` is the CPU port they are checked
//! against): the zonal rate profile, the tangent velocity of the flow, the 48-step midpoint back-trace on the
//! sphere, and the local ascent of every vortex kind (the comma ascent of a cyclone and the clear eye of a
//! tropical storm included). WGSL source: `shaders/flow.wgsl`.
//!
//! Everything runs in f32 (WGSL has no f64): a result matches the CPU port within f32 tolerances, not bit for
//! bit. The back-trace is the delicate one (Python and the CPU port integrate it in f64): its position is kept
//! as an unevaluated sum of two f32 and updated with error-free sums ([`Precision::Compensated`], the default),
//! which is about ten times more accurate than one f32 position ([`Precision::Plain`]); both errors are
//! measured in `tests/gpu_flow_accuracy.rs`. Two more portability measures: `asin` is a polynomial in the
//! shader (the WGSL builtin is only specified to a loose bound that differs between adapters), and the
//! error-free transforms are hidden from the shader compiler, which otherwise folds them to nothing.
//!
//! The flow is the CPU port's own [`Flow`] (and its [`Vortex`]es, [`VortexKind`]): the kernels take it as is, and a
//! [`GpuFlow`] uploads its vortex table once for every kernel and call.
//!
//! Everything per vortex that does not depend on the point (unit center, squared radii, comma head and its
//! life-stage constants) is computed here in f64, exactly as the CPU port does, and uploaded rounded to f32.
//!
//! Three levels of API, from cheap to convenient: the `*_buffers` methods of [`FlowKernels`] take and fill GPU
//! buffers and wait for nothing (the kernels of the other stages chain on them); the `*_to_buffer` methods
//! upload a slice and leave the result on the GPU; the Vec-returning methods and the free functions of this
//! module upload, run and read back. Large inputs are split: into dispatches of at most
//! [`MAX_POINTS_PER_DISPATCH`] points, each bound to a sub-range of the buffers and tiled in two dimensions past
//! 65535 workgroups (the adapter's `max_storage_buffer_binding_size` and `max_buffer_size` are respected), and,
//! for the Vec-returning functions, into upload segments of [`segment_len`] points.

use crate::GpuContext;
use cloud_map_generator_core::flow::{Flow, Vortex, VortexKind};
use bytemuck::{Pod, Zeroable};
use std::sync::OnceLock;

const SHADER: &str = include_str!("../shaders/flow.wgsl");

/// Threads per workgroup of every kernel (`@workgroup_size` in the shader).
const WORKGROUP: usize = 64;
/// The most points one dispatch handles (2^22: more than 65535 workgroups of 64, so the 2-D tiling is the
/// normal path; also a bound on how long one dispatch runs, for the watchdog of a slow adapter).
pub const MAX_POINTS_PER_DISPATCH: usize = 1 << 22;
/// Workgroups per dimension of a dispatch.
const MAX_GROUPS_PER_DIM: usize = 65535;

// ---------------------------------------------------------------------------------------------------------
// Host side of the vortex table (f64, as the CPU port), packed for the shader

/// Layout of `struct Vortex` in `flow.wgsl` (64 bytes, array stride 64).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuVortex {
    pub center: [f32; 3],
    pub omega: f32,
    pub rr_vel: f32,
    pub rr_asc: f32,
    pub eye2: f32,
    pub weight: f32,
    pub head: [f32; 3],
    pub dry_weight: f32,
    pub kind: u32,
    pub dry_slot: u32,
    pub pad: [u32; 2],
}

/// Layout of `struct Params` in `flow.wgsl` (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct GpuParams {
    pub hadley_deg: f32,
    pub jet: f32,
    pub trades: f32,
    pub polar_easterlies: f32,
    pub prof_ref: f32,
    pub dt: f32,
    pub half_dt: f32,
    pub n_vortices: u32,
    pub steps: u32,
    pub zero: u32,
    pub pad: [u32; 2],
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// The comma head of a cyclone (`Flow._comma_ascent`), f64: the center shifted by the east and poleward
/// vectors according to the life stage. The east vector falls back to `(0, 1, 0)` when `|cross(z, c)| < 1e-9`
/// (a vortex on a pole) and the poleward shift takes the SIGN BIT of the latitude (`copysign(1, lat)`).
fn comma_head(v: &Vortex, c: [f64; 3]) -> [f64; 3] {
    let east = cross([0.0, 0.0, 1.0], c);
    let n = (east[0] * east[0] + east[1] * east[1] + east[2] * east[2]).sqrt();
    let east = if n < 1e-9 { [0.0, 1.0, 0.0] } else { [east[0] / n, east[1] / n, east[2] / n] };
    let sign = 1.0_f64.copysign(v.lat);
    let pole = cross(c, east).map(|x| x * sign);
    // Python `min(age, 1.0)`: `age` unless `1.0 < age`.
    let grown = if 1.0 < v.age { 1.0 } else { v.age };
    let (e_w, p_w) = (0.8 - 0.5 * grown, 0.5);
    let shift: [f64; 3] = [0, 1, 2].map(|k| east[k] * e_w + pole[k] * p_w * grown);
    normalize([0, 1, 2].map(|k| c[k] + v.radius * shift[k]))
}

pub(crate) fn pack_vortex(v: &Vortex) -> GpuVortex {
    let c = v.center();
    let f3 = |a: [f64; 3]| [a[0] as f32, a[1] as f32, a[2] as f32];
    let sq = |r: f64| (r * r) as f32;
    let (rr_asc, eye2, head, dry_weight, dry_slot) = match v.kind {
        VortexKind::Cyclone => {
            // The comma: Gaussian of radius r (1 + 0.3 age) about the head, minus (age - 1) 2 times a Gaussian of
            // radius 0.45 r about the center when age > 1.
            (sq(v.radius * (1.0 + 0.3 * v.age)), sq(0.45 * v.radius), f3(comma_head(v, c)), ((v.age - 1.0) * 2.0) as f32, v.age > 1.0)
        }
        VortexKind::Anticyclone => (sq(v.radius * 1.3), 0.0, [0.0; 3], 0.0, false),
        // The ring has radius 0.6 r; the eye uses the VORTEX radius (0.12 r), not the ring's.
        VortexKind::Tropical => (sq(v.radius * 0.6), sq(0.12 * v.radius), [0.0; 3], 0.0, false),
    };
    GpuVortex {
        center: f3(c),
        omega: v.omega as f32,
        rr_vel: sq(v.radius),
        rr_asc,
        eye2,
        weight: v.ascent as f32,
        head,
        dry_weight,
        kind: match v.kind {
            VortexKind::Cyclone => 0,
            VortexKind::Anticyclone => 1,
            VortexKind::Tropical => 2,
        },
        dry_slot: u32::from(dry_slot),
        pad: [0; 2],
    }
}

/// The zonal profile `prof(x)` of the Python in f64 (used for the constant `prof(1.7)` only).
fn prof_f64(flow: &Flow, x: f64) -> f64 {
    let g = |mu: f64, s: f64| {
        let d = (x - mu) / s;
        (-(d * d)).exp()
    };
    let t = ((x - 2.0) / 0.8).clamp(0.0, 1.0);
    let sm = t * t * (3.0 - 2.0 * t);
    flow.jet * g(1.5, 0.4) - flow.trades * g(0.4, 0.4) - flow.polar_easterlies * sm
}

pub(crate) fn pack_params(flow: &Flow, time: f64, steps: usize) -> GpuParams {
    let dt = time / steps.max(1) as f64;
    GpuParams {
        hadley_deg: flow.hadley_deg as f32,
        jet: flow.jet as f32,
        trades: flow.trades as f32,
        polar_easterlies: flow.polar_easterlies as f32,
        prof_ref: prof_f64(flow, 1.7) as f32,
        dt: dt as f32,
        half_dt: (0.5 * dt) as f32,
        n_vortices: flow.vortices.len() as u32,
        steps: steps as u32,
        zero: 0,
        pad: [0; 2],
    }
}

// ---------------------------------------------------------------------------------------------------------
// The kernels

/// A flow uploaded to the GPU: the vortex table (packed once, reusable by every kernel and every call) and the
/// scalars of the flow.
pub struct GpuFlow {
    flow: Flow,
    vortices: wgpu::Buffer,
}

impl GpuFlow {
    pub fn new(gpu: &GpuContext, flow: &Flow) -> GpuFlow {
        let mut table: Vec<GpuVortex> = flow.vortices.iter().map(pack_vortex).collect();
        if table.is_empty() {
            table.push(Zeroable::zeroed()); // a zero-size binding is invalid: one unused entry
        }
        GpuFlow { flow: flow.clone(), vortices: gpu.storage_with("flow vortices", &table) }
    }

    pub fn flow(&self) -> &Flow {
        &self.flow
    }
}

/// How the back-trace keeps its position over the 48 steps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// The position is an unevaluated sum of two f32 (about 48 bits) updated with error-free sums: no step
    /// rounds it. About ten times more accurate than [`Precision::Plain`] (measured against the f64 CPU
    /// result), for a few per cent of the time.
    #[default]
    Compensated,
    /// The position is one f32, rounded at every step: the largest error is about 1e-5 rad (a twentieth of a
    /// texel of a 16K map) over the whole sphere.
    Plain,
}

/// The compute pipelines of the flow kernels, each built on first use. Building a pipeline compiles the
/// shader: keep one `FlowKernels` for repeated calls (the free functions of this module build a fresh one per
/// call).
#[derive(Default)]
pub struct FlowKernels {
    precision: Precision,
    zonal: OnceLock<wgpu::ComputePipeline>,
    velocity: OnceLock<wgpu::ComputePipeline>,
    backtrace: OnceLock<wgpu::ComputePipeline>,
    backtrace_f32: OnceLock<wgpu::ComputePipeline>,
    ascent: OnceLock<wgpu::ComputePipeline>,
    two_sum_probe: OnceLock<wgpu::ComputePipeline>,
}

/// The size of a chunk (in items) for buffers of `stride_floats` f32 per item: what the adapter's binding and
/// buffer limits allow, at most `max_points`, a multiple of the storage offset alignment so that every chunk
/// starts on a bindable offset.
pub(crate) fn chunk_items(limits: &wgpu::Limits, stride_floats: usize, max_points: usize) -> usize {
    let bytes = u64::from(limits.max_storage_buffer_binding_size).min(limits.max_buffer_size) as usize;
    let by_limits = bytes / (4 * stride_floats);
    let align = (limits.min_storage_buffer_offset_alignment as usize).max(WORKGROUP);
    let chunk = by_limits.min(max_points) / align * align;
    assert!(chunk > 0, "the adapter's storage buffer limit ({bytes} bytes) is too small for one chunk of points");
    chunk
}

/// `(x, y)` workgroups covering `items` threads, at most 65535 per dimension.
pub(crate) fn tile_groups(items: usize) -> (u32, u32) {
    let groups = items.div_ceil(WORKGROUP);
    let x = groups.min(MAX_GROUPS_PER_DIM);
    (x as u32, groups.div_ceil(x).max(1) as u32)
}

/// One buffer of a bind group: the whole buffer.
fn whole(buffer: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    buffer.as_entire_binding()
}

/// A sub-range of a buffer, `offset` bytes from its start, `size` bytes long.
fn range(buffer: &wgpu::Buffer, offset: u64, size: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset, size: wgpu::BufferSize::new(size) })
}

impl FlowKernels {
    /// The kernels with the default back-trace precision ([`Precision::Compensated`]).
    pub fn new() -> FlowKernels {
        FlowKernels::default()
    }

    /// The kernels with an explicit back-trace precision.
    pub fn with_precision(precision: Precision) -> FlowKernels {
        FlowKernels { precision, ..FlowKernels::default() }
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Whether the error-free sum of the compensated back-trace is exact on this adapter: a one-thread-per-pair
    /// self-test of TwoSum on pairs whose rounding error is known. `false` means the shader compiler folded
    /// the transform away, and [`Precision::Compensated`] is then no better than [`Precision::Plain`] (it never
    /// is worse): use `Plain` and save the extra work. Call it once per adapter.
    #[maybe_async::maybe_async]
    pub async fn compensation_works(&self, gpu: &GpuContext) -> bool {
        // (a, b): the rounding error of a + b is not zero and is exactly representable.
        let pairs: [(f32, f32); 6] = [(1.0, 1e-9), (1e8, 1.0001), (0.1, 0.2), (3.141_592_7, -2.718_28e-8), (-0.999_999_9, 3.7e-9), (65_537.0, 0.013)];
        let input: Vec<f32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        let (input, out) = (gpu.storage_with("two_sum pairs", &input), gpu.storage_empty::<f32>("two_sum results", 2 * pairs.len()));
        let params = gpu.uniform_with("flow params", &pack_params(&Flow::new(30.0), 0.0, 1));
        let p = Self::pipeline(&self.two_sum_probe, gpu, "two_sum_probe_main");
        self.run(gpu, p, &params, None, (&input, 2), (&out, 2), pairs.len(), MAX_POINTS_PER_DISPATCH);
        let got: Vec<f32> = gpu.read_back_maybe(&out, 2 * pairs.len()).await;
        pairs.iter().enumerate().all(|(i, &(a, b))| {
            let (hi, lo) = (got[2 * i], got[2 * i + 1]);
            let exact = f64::from(a) + f64::from(b);
            // hi is the rounded sum and lo what the rounding lost: hi + lo is the exact sum, lo is not zero.
            lo != 0.0 && f64::from(hi) + f64::from(lo) == exact
        })
    }

    fn pipeline<'a>(
        slot: &'a OnceLock<wgpu::ComputePipeline>,
        gpu: &GpuContext,
        entry: &str,
    ) -> &'a wgpu::ComputePipeline {
        slot.get_or_init(|| gpu.pipeline(entry, SHADER, entry))
    }

    /// Runs `pipeline` over `n` items in chunks: the shared bindings (0 = params, 1 = vortices where the kernel
    /// reads them) whole, binding 2 = `input` and binding 3 = `output` as sub-ranges of `chunk` items. One
    /// submission per chunk. Does not wait for the result.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        gpu: &GpuContext,
        pipeline: &wgpu::ComputePipeline,
        params: &wgpu::Buffer,
        vortices: Option<&wgpu::Buffer>,
        input: (&wgpu::Buffer, usize),
        output: (&wgpu::Buffer, usize),
        n: usize,
        max_points: usize,
    ) {
        let (in_stride, out_stride) = (input.1, output.1);
        let chunk = chunk_items(&gpu.limits, in_stride.max(out_stride), max_points);
        let layout = pipeline.get_bind_group_layout(0);
        let mut start = 0;
        while start < n {
            let len = chunk.min(n - start);
            let in_range = range(input.0, (start * in_stride * 4) as u64, (len * in_stride * 4) as u64);
            let out_range = range(output.0, (start * out_stride * 4) as u64, (len * out_stride * 4) as u64);
            let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: whole(params) }];
            if let Some(v) = vortices {
                entries.push(wgpu::BindGroupEntry { binding: 1, resource: whole(v) });
            }
            entries.push(wgpu::BindGroupEntry { binding: 2, resource: in_range });
            entries.push(wgpu::BindGroupEntry { binding: 3, resource: out_range });
            let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flow"),
                layout: &layout,
                entries: &entries,
            });
            let (gx, gy) = tile_groups(len);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            gpu.queue.submit([enc.finish()]);
            start += len;
        }
    }

    // --- buffer level: inputs and results stay on the GPU ---------------------------------------------------

    /// `Flow.zonal_rate` of `n` latitudes (radians) in `lats` into `out` (one f32 each).
    pub fn zonal_rate_buffers(&self, gpu: &GpuContext, flow: &GpuFlow, lats: &wgpu::Buffer, out: &wgpu::Buffer, n: usize) {
        if n == 0 {
            return;
        }
        let params = gpu.uniform_with("flow params", &pack_params(&flow.flow, 0.0, 1));
        let p = Self::pipeline(&self.zonal, gpu, "zonal_main");
        self.run(gpu, p, &params, None, (lats, 1), (out, 1), n, MAX_POINTS_PER_DISPATCH);
    }

    /// `Flow.velocity` of `n` points (3 f32 each) in `points` into `out` (3 f32 each).
    pub fn velocity_buffers(&self, gpu: &GpuContext, flow: &GpuFlow, points: &wgpu::Buffer, out: &wgpu::Buffer, n: usize) {
        if n == 0 {
            return;
        }
        let params = gpu.uniform_with("flow params", &pack_params(&flow.flow, 0.0, 1));
        let p = Self::pipeline(&self.velocity, gpu, "velocity_main");
        self.run(gpu, p, &params, Some(&flow.vortices), (points, 3), (out, 3), n, MAX_POINTS_PER_DISPATCH);
    }

    /// `Flow.backtrace` of `n` points (3 f32 each) in `points` into `out` (3 f32 each): `steps` midpoint steps
    /// over `time`. Panics for `steps = 0` (Python: ZeroDivisionError).
    #[allow(clippy::too_many_arguments)]
    pub fn backtrace_buffers(
        &self,
        gpu: &GpuContext,
        flow: &GpuFlow,
        points: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n: usize,
        time: f64,
        steps: usize,
    ) {
        assert!(steps > 0, "backtrace needs at least one step (Python: ZeroDivisionError)");
        if n == 0 {
            return;
        }
        let params = gpu.uniform_with("flow params", &pack_params(&flow.flow, time, steps));
        let p = match self.precision {
            Precision::Compensated => Self::pipeline(&self.backtrace, gpu, "backtrace_main"),
            Precision::Plain => Self::pipeline(&self.backtrace_f32, gpu, "backtrace_f32_main"),
        };
        self.run(gpu, p, &params, Some(&flow.vortices), (points, 3), (out, 3), n, MAX_POINTS_PER_DISPATCH);
    }

    /// `Flow.ascent` of `n` points (3 f32 each) in `points` into `out` (one f32 each).
    pub fn ascent_buffers(&self, gpu: &GpuContext, flow: &GpuFlow, points: &wgpu::Buffer, out: &wgpu::Buffer, n: usize) {
        if n == 0 {
            return;
        }
        let params = gpu.uniform_with("flow params", &pack_params(&flow.flow, 0.0, 1));
        let p = Self::pipeline(&self.ascent, gpu, "ascent_main");
        self.run(gpu, p, &params, Some(&flow.vortices), (points, 3), (out, 1), n, MAX_POINTS_PER_DISPATCH);
    }

    /// The same kernels with an explicit chunk size, for the tests that exercise the chunking.
    #[cfg(test)]
    pub(crate) fn velocity_chunked(&self, gpu: &GpuContext, flow: &GpuFlow, points: &[[f32; 3]], max_points: usize) -> Vec<[f32; 3]> {
        let input = gpu.storage_with("points", points);
        let out = gpu.storage_empty::<[f32; 3]>("velocity", points.len());
        let params = gpu.uniform_with("flow params", &pack_params(&flow.flow, 0.0, 1));
        let p = Self::pipeline(&self.velocity, gpu, "velocity_main");
        self.run(gpu, p, &params, Some(&flow.vortices), (&input, 3), (&out, 3), points.len(), max_points);
        gpu.read_back(&out, points.len())
    }

    // --- slice level: upload, run, keep on the GPU or read back ---------------------------------------------

    /// Uploads `points` (3 f32 each) as one storage buffer; it must fit the adapter's `max_buffer_size` (see
    /// [`max_points_per_buffer`]).
    fn upload_points(gpu: &GpuContext, points: &[[f32; 3]]) -> wgpu::Buffer {
        assert!(
            points.len() <= max_points_per_buffer(&gpu.limits),
            "{} points do not fit one buffer of this adapter (at most {}): split them, or use the Vec-returning functions, which do",
            points.len(),
            max_points_per_buffer(&gpu.limits)
        );
        // A zero-size binding is invalid: an empty input is one unused point.
        if points.is_empty() {
            gpu.storage_with("points", &[[0.0_f32; 3]])
        } else {
            gpu.storage_with("points", points)
        }
    }

    /// `Flow.backtrace` of `points`; the result stays on the GPU (3 f32 per point), the work is queued, not
    /// waited for. The points must fit one buffer ([`max_points_per_buffer`]).
    pub fn backtrace_to_buffer(&self, gpu: &GpuContext, flow: &GpuFlow, points: &[[f32; 3]], time: f64, steps: usize) -> wgpu::Buffer {
        let input = Self::upload_points(gpu, points);
        let out = gpu.storage_empty::<[f32; 3]>("backtrace", points.len().max(1));
        self.backtrace_buffers(gpu, flow, &input, &out, points.len(), time, steps);
        out
    }

    /// `Flow.ascent` of `points`; the result stays on the GPU (one f32 per point).
    pub fn ascent_to_buffer(&self, gpu: &GpuContext, flow: &GpuFlow, points: &[[f32; 3]]) -> wgpu::Buffer {
        let input = Self::upload_points(gpu, points);
        let out = gpu.storage_empty::<f32>("ascent", points.len().max(1));
        self.ascent_buffers(gpu, flow, &input, &out, points.len());
        out
    }

    /// `Flow.velocity` of `points`; the result stays on the GPU (3 f32 per point).
    pub fn velocity_to_buffer(&self, gpu: &GpuContext, flow: &GpuFlow, points: &[[f32; 3]]) -> wgpu::Buffer {
        let input = Self::upload_points(gpu, points);
        let out = gpu.storage_empty::<[f32; 3]>("velocity", points.len().max(1));
        self.velocity_buffers(gpu, flow, &input, &out, points.len());
        out
    }

    /// `Flow.backtrace` of `points`, read back to the CPU. Any number of points: they are uploaded, run and
    /// read back in segments of at most [`segment_len`] points, so no buffer exceeds the adapter's limits.
    pub fn backtrace(&self, gpu: &GpuContext, flow: &GpuFlow, points: &[[f32; 3]], time: f64, steps: usize) -> Vec<[f32; 3]> {
        assert!(steps > 0, "backtrace needs at least one step (Python: ZeroDivisionError)");
        segmented(gpu, points, segment_len(&gpu.limits), |part| self.backtrace_to_buffer(gpu, flow, part, time, steps))
    }

    /// `Flow.ascent` of `points`, read back to the CPU (segmented like [`FlowKernels::backtrace`]).
    pub fn ascent(&self, gpu: &GpuContext, flow: &GpuFlow, points: &[[f32; 3]]) -> Vec<f32> {
        segmented(gpu, points, segment_len(&gpu.limits), |part| self.ascent_to_buffer(gpu, flow, part))
    }

    /// `Flow.velocity` of `points`, read back to the CPU (segmented like [`FlowKernels::backtrace`]).
    pub fn velocity(&self, gpu: &GpuContext, flow: &GpuFlow, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
        self.velocity_in_segments(gpu, flow, points, segment_len(&gpu.limits))
    }

    pub(crate) fn velocity_in_segments(&self, gpu: &GpuContext, flow: &GpuFlow, points: &[[f32; 3]], segment: usize) -> Vec<[f32; 3]> {
        segmented(gpu, points, segment, |part| self.velocity_to_buffer(gpu, flow, part))
    }

    /// `Flow.zonal_rate` of latitudes in radians, as f32 (the CPU port returns f64 holding f32-rounded values
    /// for an f32 latitude).
    pub fn zonal_rate(&self, gpu: &GpuContext, flow: &GpuFlow, lats: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(lats.len());
        for part in lats.chunks(segment_len(&gpu.limits)) {
            let input = gpu.storage_with("lats", part);
            let rate = gpu.storage_empty::<f32>("rate", part.len());
            self.zonal_rate_buffers(gpu, flow, &input, &rate, part.len());
            out.extend(gpu.read_back::<f32>(&rate, part.len()));
        }
        out
    }
}

/// The most points a buffer of 3 f32 per point can hold on this adapter (its `max_buffer_size`), a multiple of
/// the storage offset alignment. 22 million points on a browser's default 256 MiB, 357 million at 4 GiB.
pub fn max_points_per_buffer(limits: &wgpu::Limits) -> usize {
    let align = (limits.min_storage_buffer_offset_alignment as usize).max(WORKGROUP);
    (limits.max_buffer_size / 12) as usize / align * align
}

/// Points per upload segment of the Vec-returning functions: what fits one buffer, at most 2^24 (a 200 MB
/// upload, result and read-back each).
pub fn segment_len(limits: &wgpu::Limits) -> usize {
    max_points_per_buffer(limits).min(1 << 24)
}

/// Runs `to_buffer` (upload, dispatch, result left on the GPU) over `points` in segments of at most `segment`
/// points and reads every result back, in order.
fn segmented<T: Pod>(
    gpu: &GpuContext,
    points: &[[f32; 3]],
    segment: usize,
    mut to_buffer: impl FnMut(&[[f32; 3]]) -> wgpu::Buffer,
) -> Vec<T> {
    let mut out = Vec::with_capacity(points.len());
    for part in points.chunks(segment.max(1)) {
        let buffer = to_buffer(part);
        out.extend(gpu.read_back::<T>(&buffer, part.len()));
    }
    out
}

// ---------------------------------------------------------------------------------------------------------
// Free functions: one call, a fresh `FlowKernels` (shader compiled each time)

/// `Flow.backtrace(points, time, steps)` on the GPU: where the air at each point was `time` ago.
pub fn backtrace(gpu: &GpuContext, flow: &Flow, points: &[[f32; 3]], time: f64, steps: usize) -> Vec<[f32; 3]> {
    FlowKernels::new().backtrace(gpu, &GpuFlow::new(gpu, flow), points, time, steps)
}

/// `Flow.backtrace` with the result left on the GPU (3 f32 per point).
pub fn backtrace_to_buffer(gpu: &GpuContext, flow: &Flow, points: &[[f32; 3]], time: f64, steps: usize) -> wgpu::Buffer {
    FlowKernels::new().backtrace_to_buffer(gpu, &GpuFlow::new(gpu, flow), points, time, steps)
}

/// `Flow.ascent(points)` on the GPU.
pub fn ascent(gpu: &GpuContext, flow: &Flow, points: &[[f32; 3]]) -> Vec<f32> {
    FlowKernels::new().ascent(gpu, &GpuFlow::new(gpu, flow), points)
}

/// `Flow.ascent` with the result left on the GPU (one f32 per point).
pub fn ascent_to_buffer(gpu: &GpuContext, flow: &Flow, points: &[[f32; 3]]) -> wgpu::Buffer {
    FlowKernels::new().ascent_to_buffer(gpu, &GpuFlow::new(gpu, flow), points)
}

/// `Flow.velocity(points)` on the GPU.
pub fn velocity(gpu: &GpuContext, flow: &Flow, points: &[[f32; 3]]) -> Vec<[f32; 3]> {
    FlowKernels::new().velocity(gpu, &GpuFlow::new(gpu, flow), points)
}

/// `Flow.velocity` with the result left on the GPU (3 f32 per point).
pub fn velocity_to_buffer(gpu: &GpuContext, flow: &Flow, points: &[[f32; 3]]) -> wgpu::Buffer {
    FlowKernels::new().velocity_to_buffer(gpu, &GpuFlow::new(gpu, flow), points)
}

/// `Flow.zonal_rate(lat)` on the GPU for latitudes in radians.
pub fn zonal_rate(gpu: &GpuContext, flow: &Flow, lats: &[f32]) -> Vec<f32> {
    FlowKernels::new().zonal_rate(gpu, &GpuFlow::new(gpu, flow), lats)
}

/// `Flow._comma_ascent(vortex, points)` on the GPU: the comma ascent of one vortex, unweighted, whatever the
/// kind of `vortex` (the Python method does not read it).
pub fn comma_ascent(gpu: &GpuContext, vortex: &Vortex, points: &[[f32; 3]]) -> Vec<f32> {
    let one = Vortex { kind: VortexKind::Cyclone, ascent: 1.0, ..*vortex };
    let flow = Flow { vortices: vec![one], ..Flow::new(30.0) };
    ascent(gpu, &flow, points)
}

// ---------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests need an adapter; without one they print why and pass, unless `REQUIRE_GPU` is set.
    fn gpu() -> Option<GpuContext> {
        let g = GpuContext::new();
        if g.is_none() {
            assert!(std::env::var_os("REQUIRE_GPU").is_none(), "REQUIRE_GPU is set and there is no GPU adapter");
            eprintln!("no GPU adapter: test skipped");
        }
        g
    }

    fn vortex(kind: VortexKind, lon: f64, lat: f64, radius: f64, omega: f64, ascent: f64, age: f64) -> Vortex {
        Vortex { kind, lon, lat, radius, omega, ascent, age }
    }

    fn earth_like() -> Flow {
        Flow {
            vortices: vec![
                vortex(VortexKind::Cyclone, 0.35, 0.87, 0.2, 3.0, 1.0, 0.8),
                vortex(VortexKind::Anticyclone, -1.0, 0.5, 0.3, -1.2, -0.45, 1.0),
                vortex(VortexKind::Tropical, 2.0, -0.25, 0.06, -12.0, 1.6, 1.0),
                vortex(VortexKind::Cyclone, -2.0, -0.9, 0.25, -3.9, 1.0, 1.4),
            ],
            ..Flow::new(30.0)
        }
    }

    /// A deterministic spread of unit directions (golden-angle spiral: poles and every longitude).
    fn spiral(n: usize) -> Vec<[f32; 3]> {
        let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        (0..n)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                let r = (1.0 - z * z).sqrt();
                let t = golden * i as f64;
                [(r * t.cos()) as f32, (r * t.sin()) as f32, z as f32]
            })
            .collect()
    }

    // --- host side: no adapter needed ---------------------------------------------------------------------

    #[test]
    fn the_packed_structs_have_the_layout_of_the_shader() {
        assert_eq!(std::mem::size_of::<GpuVortex>(), 64);
        assert_eq!(std::mem::size_of::<GpuParams>(), 48);
    }

    /// Evaluates the ascent of one packed vortex the way the shader does (f32), to check the packing against
    /// the CPU port without a GPU.
    fn packed_ascent(v: &GpuVortex, p: [f32; 3]) -> f32 {
        let d2 = |a: [f32; 3], b: [f32; 3]| (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]);
        let g = match v.kind {
            0 => {
                let g = (-d2(p, v.head) / v.rr_asc).exp();
                if v.dry_slot != 0 {
                    g - v.dry_weight * (-d2(p, v.center) / v.eye2).exp()
                } else {
                    g
                }
            }
            1 => (-d2(p, v.center) / v.rr_asc).exp(),
            _ => (-d2(p, v.center) / v.rr_asc).exp() - 1.6 * (-d2(p, v.center) / v.eye2).exp(),
        };
        v.weight * g
    }

    #[test]
    fn a_packed_vortex_reproduces_the_cpu_ascent_of_every_kind_and_life_stage() {
        let mut vortices = earth_like().vortices;
        for age in [0.0, 0.3, 1.0, 1.0000001, 1.4] {
            vortices.push(vortex(VortexKind::Cyclone, 0.4, 0.9, 0.2, 3.0, 1.0, age));
            vortices.push(vortex(VortexKind::Cyclone, 0.4, -0.9, 0.2, -3.0, 1.0, age));
        }
        // A pole (the east vector falls back) and a signed zero latitude (the sign bit decides the hemisphere).
        vortices.push(vortex(VortexKind::Cyclone, 0.3, std::f64::consts::FRAC_PI_2, 0.2, 3.0, 1.0, 0.6));
        vortices.push(vortex(VortexKind::Cyclone, 0.3, -0.0, 0.2, 3.0, 1.0, 1.0));
        vortices.push(vortex(VortexKind::Cyclone, 0.3, 0.0, 0.2, 3.0, 1.0, 1.0));
        let points = spiral(500);
        for vx in &vortices {
            let flow = Flow { vortices: vec![*vx], ..Flow::new(30.0) };
            let want = flow.ascent(&points);
            let packed = pack_vortex(vx);
            for (p, w) in points.iter().zip(&want) {
                let got = packed_ascent(&packed, *p);
                assert!((got - w).abs() <= 1e-6 + 1e-5 * w.abs(), "{vx:?}: {p:?} packed {got} cpu {w}");
            }
        }
    }

    #[test]
    fn the_comma_head_of_a_pole_vortex_falls_back_on_east_and_the_sign_bit_picks_the_hemisphere() {
        let pole = vortex(VortexKind::Cyclone, 0.3, std::f64::consts::FRAC_PI_2, 0.2, 3.0, 1.0, 1.0);
        let head = comma_head(&pole, pole.center());
        assert!(head.iter().all(|x| x.is_finite()));
        // Poleward shift of an equatorial low: north for +0.0, south for -0.0.
        let (pz, nz) = (vortex(VortexKind::Cyclone, 0.0, 0.0, 0.2, 3.0, 1.0, 1.0), vortex(VortexKind::Cyclone, 0.0, -0.0, 0.2, 3.0, 1.0, 1.0));
        assert!(comma_head(&pz, pz.center())[2] > 0.0);
        assert!(comma_head(&nz, nz.center())[2] < 0.0);
    }

    #[test]
    fn the_params_hold_the_f64_constants_rounded_to_f32() {
        let spec = earth_like();
        let p = pack_params(&spec, 1.0, 48);
        assert_eq!((p.n_vortices, p.steps), (4, 48));
        assert_eq!(p.dt, (1.0_f64 / 48.0) as f32);
        assert_eq!(p.half_dt, (0.5 / 48.0) as f32);
        // prof(1.7): the jet and the trades at x = 1.7, no polar easterlies yet (smoothstep 0 below x = 2).
        let want = 0.35 * (-(0.2_f64 / 0.4).powi(2)).exp() - 0.05 * (-(1.3_f64 / 0.4).powi(2)).exp();
        assert!((f64::from(p.prof_ref) - want).abs() < 1e-7);
        // A negative time advects forward, a zero time only renormalizes.
        assert!(pack_params(&spec, -1.0, 4).dt < 0.0);
        assert_eq!(pack_params(&spec, 0.0, 4).dt, 0.0);
    }

    #[test]
    fn chunks_respect_the_limits_and_the_offset_alignment() {
        let mut limits = wgpu::Limits {
            max_storage_buffer_binding_size: 128 << 20,
            max_buffer_size: 256 << 20,
            min_storage_buffer_offset_alignment: 256,
            ..wgpu::Limits::default()
        };
        // 3 f32 per point: 128 MiB / 12 B = 11184810 points, capped by the dispatch bound, a multiple of 256.
        let c = chunk_items(&limits, 3, MAX_POINTS_PER_DISPATCH);
        assert_eq!(c, MAX_POINTS_PER_DISPATCH);
        let tiny = chunk_items(&limits, 3, 1000);
        assert_eq!(tiny, 768);
        assert_eq!(tiny * 3 * 4 % 256, 0);
        limits.max_storage_buffer_binding_size = 1 << 20;
        let small = chunk_items(&limits, 3, MAX_POINTS_PER_DISPATCH);
        assert!(small * 12 <= 1 << 20 && small.is_multiple_of(256) && small > 0);
    }

    #[test]
    fn dispatches_are_tiled_in_two_dimensions_past_65535_workgroups() {
        assert_eq!(tile_groups(1), (1, 1));
        assert_eq!(tile_groups(64), (1, 1));
        assert_eq!(tile_groups(65), (2, 1));
        assert_eq!(tile_groups(65535 * 64), (65535, 1));
        assert_eq!(tile_groups(65535 * 64 + 1), (65535, 2));
        let (x, y) = tile_groups(MAX_POINTS_PER_DISPATCH);
        assert!(u64::from(x) * u64::from(y) * 64 >= MAX_POINTS_PER_DISPATCH as u64 && y == 2);
    }

    // --- kernels: need an adapter -------------------------------------------------------------------------

    #[test]
    fn velocity_matches_the_cpu_in_f32() {
        let Some(g) = gpu() else { return };
        let spec = earth_like();
        let points = spiral(3000);
        let got = velocity(&g, &spec, &points);
        let want = spec.velocity(&points);
        for (a, b) in got.iter().zip(&want) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() <= 1e-5 + 1e-4 * b[k].abs(), "{a:?} {b:?}");
            }
        }
    }

    #[test]
    fn zonal_rate_is_zero_at_the_storm_track_and_symmetric() {
        let Some(g) = gpu() else { return };
        let spec = Flow::new(30.0);
        let lats = [51.0_f32.to_radians(), 0.4, -0.4, 0.0, std::f32::consts::FRAC_PI_2, 7.0];
        let r = zonal_rate(&g, &spec, &lats);
        assert!(r[0].abs() < 1e-6, "{}", r[0]);
        assert_eq!(r[1], r[2]);
        let want = spec.zonal_rate(0.4_f32);
        assert!((f64::from(r[1]) - want).abs() < 2e-6);
        assert!(r[4] < 0.0); // polar easterlies
        assert!(r.iter().all(|x| x.is_finite())); // a latitude beyond pi/2 is accepted
    }

    #[test]
    fn backtrace_matches_the_f64_cpu_and_renormalizes() {
        let Some(g) = gpu() else { return };
        let spec = earth_like();
        let points = spiral(2000);
        let got = backtrace(&g, &spec, &points, 1.0, 48);
        let want = spec.backtrace(&points, 1.0, 48);
        let mut worst = 0.0_f32;
        for (a, b) in got.iter().zip(&want) {
            for k in 0..3 {
                worst = worst.max((a[k] - b[k]).abs());
            }
            let n = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
            assert!((n - 1.0).abs() < 1e-6, "{a:?}");
        }
        assert!(worst < 4e-6, "worst component error {worst}");
        // time 0 only renormalizes: a non-unit input comes back unit, a unit one unchanged up to rounding.
        let id = backtrace(&g, &spec, &[[0.0, 0.0, 3.0], [1.0, 0.0, 0.0]], 0.0, 3);
        assert!(id[0][0] == 0.0 && id[0][1] == 0.0 && (id[0][2] - 1.0).abs() < 2e-7, "{:?}", id[0]);
        assert!((id[1][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn the_error_free_sum_of_the_compensated_backtrace_is_exact_on_this_adapter() {
        let Some(g) = gpu() else { return };
        assert!(FlowKernels::new().compensation_works(&g), "the shader compiler folded TwoSum: the barrier does not hold here");
    }

    #[test]
    fn backtrace_forward_then_back_is_nearly_the_identity() {
        let Some(g) = gpu() else { return };
        let spec = earth_like();
        let points = spiral(1000);
        let there = backtrace(&g, &spec, &points, -1.0, 96);
        let back = backtrace(&g, &spec, &there, 1.0, 96);
        for (a, b) in points.iter().zip(&back) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 5e-3, "{a:?} {b:?}");
            }
        }
    }

    #[test]
    fn backtrace_needs_a_step() {
        let Some(g) = gpu() else { return };
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backtrace(&g, &Flow::new(30.0), &[[1.0, 0.0, 0.0]], 1.0, 0)
        }));
        assert!(refused.is_err(), "Python raises ZeroDivisionError for steps = 0");
    }

    #[test]
    fn ascent_matches_the_cpu_in_f32_for_every_kind() {
        let Some(g) = gpu() else { return };
        let spec = earth_like();
        let points = spiral(3000);
        let got = ascent(&g, &spec, &points);
        let want = spec.ascent(&points);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() <= 8e-6 + 8e-5 * b.abs(), "{a} {b}");
        }
        // No vortex: exactly zero.
        assert!(ascent(&g, &Flow::new(30.0), &points).iter().all(|&a| a == 0.0));
    }

    #[test]
    fn comma_ascent_alone_is_the_unweighted_comma_of_any_kind() {
        let Some(g) = gpu() else { return };
        let vx = vortex(VortexKind::Anticyclone, 0.4, 0.9, 0.2, 3.0, -0.45, 1.4);
        let points = spiral(1000);
        let got = comma_ascent(&g, &vx, &points);
        let want = Flow::comma_ascent(&Vortex { kind: VortexKind::Anticyclone, lon: 0.4, lat: 0.9, radius: 0.2, omega: 3.0, ascent: -0.45, age: 1.4 }, &points);
        for (a, b) in got.iter().zip(&want) {
            assert!((a - b).abs() <= 2e-6 + 2e-5 * b.abs(), "{a} {b}");
        }
    }

    #[test]
    fn an_empty_input_and_an_empty_flow_are_fine() {
        let Some(g) = gpu() else { return };
        let spec = Flow::new(30.0);
        assert!(velocity(&g, &spec, &[]).is_empty() && ascent(&g, &spec, &[]).is_empty());
        assert!(backtrace(&g, &spec, &[], 1.0, 48).is_empty() && zonal_rate(&g, &spec, &[]).is_empty());
        // The zonal flow alone: purely east-west, zero on the axis.
        let v = velocity(&g, &spec, &[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]);
        assert_eq!(v, vec![[0.0, 0.0, 0.0]; 2]);
    }

    #[test]
    fn chunked_and_tiled_dispatches_give_the_same_result_as_one_dispatch() {
        let Some(g) = gpu() else { return };
        let spec = earth_like();
        let k = FlowKernels::new();
        let flow = GpuFlow::new(&g, &spec);
        let points = spiral(5000);
        let whole = k.velocity(&g, &flow, &points);
        // Chunks of 768 points (the smallest the alignment allows): 7 dispatches, the last one partial.
        let chunked = k.velocity_chunked(&g, &flow, &points, 768);
        assert_eq!(whole, chunked);
        // Uploaded and read back in segments of 1000 points (5 buffers): the same again.
        assert_eq!(whole, k.velocity_in_segments(&g, &flow, &points, 1000));
    }

    #[test]
    fn a_buffer_holds_what_the_adapter_allows() {
        let mut limits = wgpu::Limits {
            min_storage_buffer_offset_alignment: 256,
            max_buffer_size: 256 << 20, // a browser's default
            ..wgpu::Limits::default()
        };
        assert_eq!(max_points_per_buffer(&limits), 22_369_536);
        assert_eq!(segment_len(&limits), 1 << 24);
        limits.max_buffer_size = 64 << 20;
        assert_eq!(max_points_per_buffer(&limits), 5_592_320);
        assert_eq!(segment_len(&limits), 5_592_320);
        limits.max_buffer_size = 4 << 30;
        assert_eq!(segment_len(&limits), 1 << 24);
        assert!(max_points_per_buffer(&limits) * 12 <= 4 << 30 && max_points_per_buffer(&limits).is_multiple_of(256));
    }

    #[test]
    fn more_points_than_one_dispatch_holds_are_split_and_tiled() {
        let Some(g) = gpu() else { return };
        // 2^22 + 100 points: one full dispatch of 65536 workgroups (tiled 65535 x 2) and a partial one.
        let n = MAX_POINTS_PER_DISPATCH + 100;
        let base = spiral(997);
        let points: Vec<[f32; 3]> = (0..n).map(|i| base[i % base.len()]).collect();
        let spec = earth_like();
        let got = velocity(&g, &spec, &points);
        assert_eq!(got.len(), n);
        let want = spec.velocity(&base);
        // The last thread of the tiled dispatch (an index past 65535 x 64) and the partial chunk.
        for i in [0, 1, 65535 * 64, 65535 * 64 + 1, MAX_POINTS_PER_DISPATCH - 1, MAX_POINTS_PER_DISPATCH, n - 1] {
            let (a, b) = (got[i], want[i % base.len()]);
            for c in 0..3 {
                assert!((a[c] - b[c]).abs() <= 1e-5 + 1e-4 * b[c].abs(), "point {i}: {a:?} {b:?}");
            }
        }
    }

    #[test]
    fn results_can_stay_on_the_gpu_and_feed_another_kernel() {
        let Some(g) = gpu() else { return };
        let spec = earth_like();
        let k = FlowKernels::new();
        let flow = GpuFlow::new(&g, &spec);
        let points = spiral(2000);
        // Back-trace to a buffer, then the ascent at the back-traced points straight from that buffer.
        let traced = k.backtrace_to_buffer(&g, &flow, &points, 1.0, 48);
        let out = g.storage_empty::<f32>("ascent", points.len());
        k.ascent_buffers(&g, &flow, &traced, &out, points.len());
        let on_gpu: Vec<f32> = g.read_back(&out, points.len());
        let via_host = k.ascent(&g, &flow, &k.backtrace(&g, &flow, &points, 1.0, 48));
        assert_eq!(on_gpu, via_host);
    }
}
