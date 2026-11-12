//! GPU kernels of `cloudgen/noise.py`, ported from the CPU port `cloud_map_generator_core::noise`: Perlin fBm over the
//! doubled permutation table (octave rotation and offset, billow, norm), Worley F1/F2 and `cell_edges`
//! (F2 - F1 of a rotated Worley lattice). The WGSL is `shaders/noise.wgsl`.
//!
//! The wrappers take the CPU port's plan structs ([`FbmPlan`], [`CellEdgesPlan`], [`Perlin3`], [`Worley3`]) and
//! plain slices of points, and return `Vec`s. Everything is `f32` (WGSL has no `f64`): the frequency, the
//! amplitudes and the norm are rounded once on the CPU, exactly where the CPU port rounds them per use, and the
//! kernels then repeat the CPU port's order of `f32` operations (`(p @ rot^T) * f + off`, floor, fraction, hash).
//! A GPU may fuse a multiply and an add, as numpy's BLAS matmul does; the golden tolerances were measured with
//! that freedom, and `tests/golden_gpu_noise.rs` reports what the GPU needs beyond them.
//!
//! Large inputs are chunked to fit the adapter: a chunk is at most [`max_chunk_points`] points (the
//! storage-binding limit and a cap of 16M points, a multiple of 1024 so that chunk offsets stay aligned) and is
//! dispatched over a 2-D grid, so neither the 65535-groups-per-dimension limit nor the binding size limit is
//! hit. The `*_to_buffer` variants leave the result on the GPU, so a later kernel can chain without a readback.
//!
//! Not supported on the GPU: NaN and infinite points (WGSL lets a compiler assume they do not occur), and
//! `f64` arithmetic (the `Perlin3` golden case on `float64` coordinates, which needs every intermediate in
//! `f64`).

use crate::GpuContext;
use cloud_map_generator_core::noise::{CellEdgesPlan, FbmPlan, Perlin3, Worley3, PERLIN3_ABS_MEAN};
use std::cell::OnceCell;
use std::num::NonZeroU64;

const SHADER: &str = include_str!("../shaders/noise.wgsl");

/// Threads per workgroup (`@workgroup_size` of the kernels).
const WORKGROUP: usize = 64;
/// Workgroups along x of a dispatch: a dispatch is `groups_x * groups_y` workgroups over a 2-D grid.
const MAX_GROUPS_X: usize = 4096;
/// Largest chunk, in points: bounds the staging memory of one readback (16M points = 192 MB in, 64 MB out).
const MAX_CHUNK_POINTS: usize = 1 << 24;
/// Chunk sizes are multiples of this, so that `chunk * 12` and `chunk * 4` bytes are multiples of the 256-byte
/// storage-buffer offset alignment of every adapter.
const CHUNK_ALIGN: usize = 1024;
/// f32 per octave record in the `octaves` buffer (see the WGSL header).
const OCT_STRIDE: usize = 16;

/// The `Params` uniform of the kernels (32 bytes, the layout of the WGSL struct).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    row_stride: u32,
    octaves: u32,
    billow: u32,
    norm: f32,
    mean1: f32,
    pad0: u32,
    pad1: u32,
}

// ---------------------------------------------------------------------------------------------------------
// Packing of the plans into what the kernels read

/// The doubled Perlin permutation table the kernel indexes: 512 `i32`.
pub fn perlin_table(perlin: &Perlin3) -> Vec<i32> {
    assert_eq!(perlin.perm.len(), 512, "a Perlin3 table is the doubled 256-entry permutation");
    perlin.perm.clone()
}

/// The doubled Worley permutation the kernel indexes: 2048 `i32` (WGSL has no 64-bit integers; the values are
/// below 1024).
pub fn worley_perm_table(worley: &Worley3) -> Vec<i32> {
    assert_eq!(worley.perm.len(), 2 * Worley3::SIZE, "a Worley3 table is the doubled 1024-entry permutation");
    worley.perm.iter().map(|&v| i32::try_from(v).expect("a Worley3 permutation entry fits an i32")).collect()
}

/// The Worley jitter table the kernel indexes: 1024 points, 3 `f32` each, back to back.
pub fn worley_jitter_table(worley: &Worley3) -> Vec<f32> {
    assert_eq!(worley.jitter.len(), Worley3::SIZE, "a Worley3 jitter table has 1024 rows");
    worley.jitter.iter().flatten().copied().collect()
}

/// One 16-`f32` record per octave: `rot[row][col]` row-major (9), the offset (3), `float32(freq)`,
/// `float32(amp)`, two zeros: `OCT_*` of the WGSL header.
pub fn pack_fbm_octaves(plan: &FbmPlan) -> Vec<f32> {
    let mut out = Vec::with_capacity(OCT_STRIDE * plan.octaves.len());
    for o in &plan.octaves {
        out.extend(o.rot.iter().flatten());
        out.extend(o.off);
        out.push(o.freq as f32);
        out.push(o.amp as f32);
        out.extend([0.0, 0.0]);
    }
    out
}

/// The single record of `cell_edges`: the rotation, no offset, `float32(frequency)`.
fn pack_cell_edges_record(plan: &CellEdgesPlan) -> Vec<f32> {
    let mut out = Vec::with_capacity(OCT_STRIDE);
    out.extend(plan.rot.iter().flatten());
    out.extend([0.0, 0.0, 0.0]);
    out.push(plan.frequency as f32);
    out.extend([1.0, 0.0, 0.0]);
    out
}

// ---------------------------------------------------------------------------------------------------------
// Chunks and dispatch geometry

/// Points per chunk for this adapter: what fits one storage binding (the 12-byte input is the larger side),
/// capped at 16M, rounded down to a multiple of 1024.
pub fn max_chunk_points(gpu: &GpuContext) -> usize {
    let bytes = u64::from(gpu.limits.max_storage_buffer_binding_size).min(gpu.limits.max_buffer_size);
    let by_limit = usize::try_from(bytes / 12).unwrap_or(usize::MAX);
    let chunk = by_limit.min(MAX_CHUNK_POINTS) / CHUNK_ALIGN * CHUNK_ALIGN;
    assert!(chunk >= CHUNK_ALIGN, "the adapter's storage binding limit ({bytes} bytes) is too small for a chunk");
    chunk
}

/// `(groups_x, groups_y, row_stride)` for `n` points: a 2-D grid of workgroups.
fn grid(n: usize) -> (u32, u32, u32) {
    let groups = n.div_ceil(WORKGROUP).max(1);
    let gx = groups.min(MAX_GROUPS_X);
    let gy = groups.div_ceil(gx);
    assert!(gy <= 65535, "{n} points need more than 65535 rows of workgroups: chunk the input");
    (gx as u32, gy as u32, (gx * WORKGROUP) as u32)
}

fn chunks(n: usize, chunk: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n).step_by(chunk).map(move |start| (start, chunk.min(n - start)))
}

/// A byte range of a buffer, bound whole or in part.
#[derive(Clone, Copy)]
struct BufRange<'a> {
    buffer: &'a wgpu::Buffer,
    offset: u64,
    size: u64,
}

impl<'a> BufRange<'a> {
    fn whole(buffer: &'a wgpu::Buffer) -> BufRange<'a> {
        BufRange { buffer, offset: 0, size: buffer.size() }
    }
    fn binding(&self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.buffer,
            offset: self.offset,
            size: NonZeroU64::new(self.size),
        })
    }
}

/// The kernels of this module, compiled once per `GpuContext` and on first use. The free functions of the module
/// build a fresh set per call; a caller that runs several maps builds one set and reuses it.
pub struct NoiseKernels<'a> {
    gpu: &'a GpuContext,
    module: wgpu::ShaderModule,
    fbm: OnceCell<wgpu::ComputePipeline>,
    perlin: OnceCell<wgpu::ComputePipeline>,
    worley: OnceCell<wgpu::ComputePipeline>,
    cell_edges: OnceCell<wgpu::ComputePipeline>,
}

/// Which kernel, and the static tables and extra inputs it binds besides params, points and out.
struct Job<'a> {
    pipeline: &'a wgpu::ComputePipeline,
    /// `(binding, buffer)` of the tables (Perlin permutation, octaves, Worley tables).
    statics: Vec<(u32, &'a wgpu::Buffer)>,
    params: Params,
    /// The second output (`out2`, binding 7) of the Worley kernel.
    has_second_output: bool,
}

impl<'a> NoiseKernels<'a> {
    pub fn new(gpu: &'a GpuContext) -> NoiseKernels<'a> {
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloudgen noise"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });
        NoiseKernels {
            gpu,
            module,
            fbm: OnceCell::new(),
            perlin: OnceCell::new(),
            worley: OnceCell::new(),
            cell_edges: OnceCell::new(),
        }
    }

    /// The pipeline of an entry point, built on first use.
    fn pipeline<'s>(&'s self, cell: &'s OnceCell<wgpu::ComputePipeline>, entry: &str) -> &'s wgpu::ComputePipeline {
        cell.get_or_init(|| {
            self.gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &self.module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        })
    }

    /// Records one chunk: a bind group of the job's bindings and the chunk's ranges, and a dispatch of the grid.
    fn record_chunk(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        job: &Job<'_>,
        n: usize,
        points: BufRange<'_>,
        out: BufRange<'_>,
        out2: Option<BufRange<'_>>,
    ) {
        let (gx, gy, row_stride) = grid(n);
        let params = Params { n: n as u32, row_stride, ..job.params };
        let params_buffer = self.gpu.uniform_with("noise params", &params);
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: points.binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out.binding() },
        ];
        if let Some(o2) = out2 {
            entries.push(wgpu::BindGroupEntry { binding: 7, resource: o2.binding() });
        }
        for (binding, buffer) in &job.statics {
            entries.push(wgpu::BindGroupEntry { binding: *binding, resource: buffer.as_entire_binding() });
        }
        let bind = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("noise"),
            layout: &job.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        pass.set_pipeline(job.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }

    /// Runs a job over host points: per chunk, upload, dispatch, copy to staging in the same submission, then map
    /// and append the staging memory straight to the result (one host copy). Returns one `Vec` per output.
    fn run_slice(&self, job: &Job<'_>, points: &[[f32; 3]], chunk: usize) -> Vec<Vec<f32>> {
        let n_out = if job.has_second_output { 2 } else { 1 };
        let mut outs: Vec<Vec<f32>> = (0..n_out).map(|_| Vec::with_capacity(points.len())).collect();
        for (start, len) in chunks(points.len(), chunk) {
            let bytes = 4 * len as u64;
            let input = self.gpu.storage_with("noise points", &points[start..start + len]);
            let out = self.gpu.storage_empty::<f32>("noise out", len);
            let out2 = job.has_second_output.then(|| self.gpu.storage_empty::<f32>("noise out2", len));
            let stage = self.staging(bytes);
            let stage2 = job.has_second_output.then(|| self.staging(bytes));
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                self.record_chunk(&mut pass, job, len, BufRange::whole(&input), BufRange::whole(&out), out2.as_ref().map(BufRange::whole));
            }
            enc.copy_buffer_to_buffer(&out, 0, &stage, 0, bytes);
            if let (Some(o2), Some(s2)) = (&out2, &stage2) {
                enc.copy_buffer_to_buffer(o2, 0, s2, 0, bytes);
            }
            self.gpu.queue.submit([enc.finish()]);
            self.append_mapped(&stage, &mut outs[0]);
            if let Some(s2) = &stage2 {
                self.append_mapped(s2, &mut outs[1]);
            }
        }
        outs
    }

    /// A buffer the CPU can map for reading.
    fn staging(&self, bytes: u64) -> wgpu::Buffer {
        self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("noise readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Waits for the submitted work, maps a staging buffer and appends its `f32` to `dst`.
    fn append_mapped(&self, stage: &wgpu::Buffer, dst: &mut Vec<f32>) {
        let slice = stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).expect("the receiver outlives the mapping"));
        self.gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        rx.recv().expect("map callback").expect("map");
        dst.extend_from_slice(bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()));
        stage.unmap();
    }

    /// Runs a job over a points buffer already on the GPU and leaves the result on the GPU: one pass, one
    /// dispatch per chunk, bound with offsets.
    fn run_buffer(&self, job: &Job<'_>, points: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
        assert!(points.usage().contains(wgpu::BufferUsages::STORAGE), "the points buffer needs STORAGE usage");
        assert!(points.size() >= 12 * n as u64, "the points buffer holds {} bytes, {n} points need {}", points.size(), 12 * n);
        let out = self.gpu.storage_empty::<f32>("noise out", n.max(1));
        if n > 0 {
            let chunk = max_chunk_points(self.gpu);
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for (start, len) in chunks(n, chunk) {
                    let p = BufRange { buffer: points, offset: 12 * start as u64, size: 12 * len as u64 };
                    let o = BufRange { buffer: &out, offset: 4 * start as u64, size: 4 * len as u64 };
                    self.record_chunk(&mut pass, job, len, p, o, None);
                }
            }
            self.gpu.queue.submit([enc.finish()]);
        }
        out
    }

    // -- fBm ------------------------------------------------------------------------------------------------

    fn fbm_inputs(&self, plan: &FbmPlan) -> (wgpu::Buffer, wgpu::Buffer, Params) {
        assert!(!plan.octaves.is_empty(), "fbm needs at least one octave");
        let perm = self.gpu.storage_with("perlin perm", &perlin_table(&plan.perlin));
        let octaves = self.gpu.storage_with("fbm octaves", &pack_fbm_octaves(plan));
        let params = Params {
            n: 0,
            row_stride: 0,
            octaves: plan.octaves.len() as u32,
            billow: u32::from(plan.billow),
            norm: plan.norm as f32,
            mean1: if plan.billow { PERLIN3_ABS_MEAN as f32 } else { 0.0 },
            pad0: 0,
            pad1: 0,
        };
        (perm, octaves, params)
    }

    /// `fbm(points, plan)` of the CPU port, on the GPU: the points go up, the field comes back.
    pub fn fbm(&self, points: &[[f32; 3]], plan: &FbmPlan) -> Vec<f32> {
        self.fbm_with_chunk(points, plan, max_chunk_points(self.gpu))
    }

    /// [`NoiseKernels::fbm`] with an explicit chunk size in points (a test of the chunking, or a smaller staging
    /// footprint). Not rounded: any chunk size that fits the adapter works for slices.
    pub fn fbm_with_chunk(&self, points: &[[f32; 3]], plan: &FbmPlan, chunk: usize) -> Vec<f32> {
        assert!(chunk > 0);
        if points.is_empty() {
            return Vec::new();
        }
        let (perm, octaves, params) = self.fbm_inputs(plan);
        let job = Job { pipeline: self.pipeline(&self.fbm, "fbm_main"), statics: vec![(3, &perm), (4, &octaves)], params, has_second_output: false };
        self.run_slice(&job, points, chunk).swap_remove(0)
    }

    /// `fbm` on a points buffer already on the GPU (`n` points, 3 `f32` each, `STORAGE` usage); the field stays
    /// on the GPU, in a buffer of `n` `f32`, for the next kernel.
    pub fn fbm_to_buffer(&self, points: &wgpu::Buffer, n: usize, plan: &FbmPlan) -> wgpu::Buffer {
        let (perm, octaves, params) = self.fbm_inputs(plan);
        let job = Job { pipeline: self.pipeline(&self.fbm, "fbm_main"), statics: vec![(3, &perm), (4, &octaves)], params, has_second_output: false };
        self.run_buffer(&job, points, n)
    }

    // -- cell_edges -----------------------------------------------------------------------------------------

    fn cell_edges_inputs(&self, plan: &CellEdgesPlan) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let perm = self.gpu.storage_with("worley perm", &worley_perm_table(&plan.worley));
        let jitter = self.gpu.storage_with("worley jitter", &worley_jitter_table(&plan.worley));
        let record = self.gpu.storage_with("cell_edges record", &pack_cell_edges_record(plan));
        (perm, jitter, record)
    }

    /// `cell_edges(points, plan)` of the CPU port, on the GPU.
    pub fn cell_edges(&self, points: &[[f32; 3]], plan: &CellEdgesPlan) -> Vec<f32> {
        self.cell_edges_with_chunk(points, plan, max_chunk_points(self.gpu))
    }

    /// [`NoiseKernels::cell_edges`] with an explicit chunk size in points.
    pub fn cell_edges_with_chunk(&self, points: &[[f32; 3]], plan: &CellEdgesPlan, chunk: usize) -> Vec<f32> {
        assert!(chunk > 0);
        if points.is_empty() {
            return Vec::new();
        }
        let (perm, jitter, record) = self.cell_edges_inputs(plan);
        let params = Params { n: 0, row_stride: 0, octaves: 1, billow: 0, norm: 1.0, mean1: 0.0, pad0: 0, pad1: 0 };
        let job = Job {
            pipeline: self.pipeline(&self.cell_edges, "cell_edges_main"),
            statics: vec![(4, &record), (5, &perm), (6, &jitter)],
            params,
            has_second_output: false,
        };
        self.run_slice(&job, points, chunk).swap_remove(0)
    }

    /// `cell_edges` on a points buffer already on the GPU; the field stays on the GPU.
    pub fn cell_edges_to_buffer(&self, points: &wgpu::Buffer, n: usize, plan: &CellEdgesPlan) -> wgpu::Buffer {
        let (perm, jitter, record) = self.cell_edges_inputs(plan);
        let params = Params { n: 0, row_stride: 0, octaves: 1, billow: 0, norm: 1.0, mean1: 0.0, pad0: 0, pad1: 0 };
        let job = Job {
            pipeline: self.pipeline(&self.cell_edges, "cell_edges_main"),
            statics: vec![(4, &record), (5, &perm), (6, &jitter)],
            params,
            has_second_output: false,
        };
        self.run_buffer(&job, points, n)
    }

    // -- raw Perlin and Worley ------------------------------------------------------------------------------

    /// `Perlin3.__call__` on `f32` points: the noise at each point, with no rotation and no octaves.
    pub fn perlin3(&self, perlin: &Perlin3, points: &[[f32; 3]]) -> Vec<f32> {
        let perm = self.gpu.storage_with("perlin perm", &perlin_table(perlin));
        if points.is_empty() {
            return Vec::new();
        }
        let params = Params { n: 0, row_stride: 0, octaves: 0, billow: 0, norm: 1.0, mean1: 0.0, pad0: 0, pad1: 0 };
        let job = Job { pipeline: self.pipeline(&self.perlin, "perlin_main"), statics: vec![(3, &perm)], params, has_second_output: false };
        self.run_slice(&job, points, max_chunk_points(self.gpu)).swap_remove(0)
    }

    /// `Worley3.__call__` on `f32` points: `(F1, F2)` at each point (the CPU port also takes `f64` points and
    /// splits the cell from the fraction in `f64`; here the points are `f32`).
    pub fn worley3(&self, worley: &Worley3, points: &[[f32; 3]]) -> (Vec<f32>, Vec<f32>) {
        let perm = self.gpu.storage_with("worley perm", &worley_perm_table(worley));
        let jitter = self.gpu.storage_with("worley jitter", &worley_jitter_table(worley));
        if points.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let params = Params { n: 0, row_stride: 0, octaves: 0, billow: 0, norm: 1.0, mean1: 0.0, pad0: 0, pad1: 0 };
        let job = Job { pipeline: self.pipeline(&self.worley, "worley_main"), statics: vec![(5, &perm), (6, &jitter)], params, has_second_output: true };
        let mut outs = self.run_slice(&job, points, max_chunk_points(self.gpu));
        let f2 = outs.swap_remove(1);
        (outs.swap_remove(0), f2)
    }
}

// ---------------------------------------------------------------------------------------------------------
// Free functions: one set of kernels per call

/// `fbm(points, plan)` on the GPU. Chunked internally to fit the adapter.
pub fn fbm(gpu: &GpuContext, points: &[[f32; 3]], plan: &FbmPlan) -> Vec<f32> {
    NoiseKernels::new(gpu).fbm(points, plan)
}

/// `fbm` on a points buffer already on the GPU (`n` points of 3 `f32`, `STORAGE` usage), leaving the field on the
/// GPU in a buffer of `n` `f32` (`STORAGE | COPY_SRC`), so that a later kernel chains without a readback. Large
/// `n` is chunked with offset bindings. See [`upload_points`].
pub fn fbm_to_buffer(gpu: &GpuContext, points_buffer: &wgpu::Buffer, n: usize, plan: &FbmPlan) -> wgpu::Buffer {
    NoiseKernels::new(gpu).fbm_to_buffer(points_buffer, n, plan)
}

/// `cell_edges(points, plan)` on the GPU.
pub fn cell_edges(gpu: &GpuContext, points: &[[f32; 3]], plan: &CellEdgesPlan) -> Vec<f32> {
    NoiseKernels::new(gpu).cell_edges(points, plan)
}

/// `cell_edges` on a points buffer already on the GPU; the field stays on the GPU.
pub fn cell_edges_to_buffer(gpu: &GpuContext, points_buffer: &wgpu::Buffer, n: usize, plan: &CellEdgesPlan) -> wgpu::Buffer {
    NoiseKernels::new(gpu).cell_edges_to_buffer(points_buffer, n, plan)
}

/// `Perlin3.__call__` on `f32` points.
pub fn perlin3(gpu: &GpuContext, perlin: &Perlin3, points: &[[f32; 3]]) -> Vec<f32> {
    NoiseKernels::new(gpu).perlin3(perlin, points)
}

/// `Worley3.__call__` on `f32` points: `(F1, F2)`.
pub fn worley3(gpu: &GpuContext, worley: &Worley3, points: &[[f32; 3]]) -> (Vec<f32>, Vec<f32>) {
    NoiseKernels::new(gpu).worley3(worley, points)
}

/// Uploads points (3 `f32` each, no padding) to a storage buffer for the `*_to_buffer` kernels.
pub fn upload_points(gpu: &GpuContext, points: &[[f32; 3]]) -> wgpu::Buffer {
    gpu.storage_with("noise points", points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cloud_map_generator_core::noise::{cell_edges as cpu_cell_edges, fbm as cpu_fbm, FbmOctave, DEFAULT_LACUNARITY};
    use cloud_map_generator_core::rng::Rng;

    /// The adapter, or `None` (with the reason printed) when there is none; with `REQUIRE_GPU` set a missing
    /// adapter is a failure.
    fn test_gpu() -> Option<GpuContext> {
        let gpu = GpuContext::new();
        match &gpu {
            Some(g) => eprintln!("adapter: {} ({:?})", g.adapter_info.name, g.adapter_info.backend),
            None if std::env::var_os("REQUIRE_GPU").is_some() => panic!("no GPU adapter and REQUIRE_GPU is set"),
            None => eprintln!("no GPU adapter: test skipped (set REQUIRE_GPU to fail instead)"),
        }
        gpu
    }

    fn sphere_points(rng: &mut Rng, n: usize) -> Vec<[f32; 3]> {
        (0..n)
            .map(|_| {
                let (x, y, z) = (rng.normal(), rng.normal(), rng.normal());
                let r = (x * x + y * y + z * z).sqrt();
                [(x / r) as f32, (y / r) as f32, (z / r) as f32]
            })
            .collect()
    }

    fn box_points(rng: &mut Rng, n: usize, half: f64) -> Vec<[f32; 3]> {
        (0..n).map(|_| [0, 1, 2].map(|_| rng.uniform_range(-half, half) as f32)).collect()
    }

    /// (max, mean) of |a - b| and the count of elements whose error exceeds `tol`.
    fn errors(a: &[f32], b: &[f32], tol: f32) -> (f32, f64, usize) {
        assert_eq!(a.len(), b.len());
        let mut max = 0.0f32;
        let mut sum = 0.0f64;
        let mut over = 0;
        for (x, y) in a.iter().zip(b) {
            let e = (x - y).abs();
            max = max.max(e);
            sum += f64::from(e);
            over += usize::from(e > tol);
        }
        (max, sum / a.len().max(1) as f64, over)
    }

    #[test]
    fn octave_records_pack_the_plan_in_the_layout_of_the_shader() {
        let plan = FbmPlan::from_seed(3, 6.0, 3, 0.5, 2.03, true);
        let packed = pack_fbm_octaves(&plan);
        assert_eq!(packed.len(), 3 * OCT_STRIDE);
        for (k, o) in plan.octaves.iter().enumerate() {
            let rec = &packed[k * OCT_STRIDE..(k + 1) * OCT_STRIDE];
            for r in 0..3 {
                for c in 0..3 {
                    assert_eq!(rec[3 * r + c], o.rot[r][c]);
                }
            }
            assert_eq!(&rec[9..12], &o.off);
            assert_eq!(rec[12], o.freq as f32);
            assert_eq!(rec[13], o.amp as f32);
            assert_eq!(&rec[14..], &[0.0, 0.0]);
        }
        let ce = CellEdgesPlan::from_seed(5, 12.5);
        let rec = pack_cell_edges_record(&ce);
        assert_eq!(rec.len(), OCT_STRIDE);
        assert_eq!(rec[12], 12.5);
        assert_eq!(&rec[9..12], &[0.0, 0.0, 0.0]);
        assert_eq!(rec[3 * 2 + 1], ce.rot[2][1]);
    }

    #[test]
    fn tables_are_the_doubled_ones_and_narrow_without_loss() {
        let p = Perlin3::from_seed(4);
        let t = perlin_table(&p);
        assert_eq!(t.len(), 512);
        assert_eq!(&t[..256], &t[256..]);
        let w = Worley3::from_seed(4);
        let wp = worley_perm_table(&w);
        assert_eq!(wp.len(), 2048);
        assert!(wp.iter().zip(&w.perm).all(|(&a, &b)| i64::from(a) == b));
        let wj = worley_jitter_table(&w);
        assert_eq!(wj.len(), 3 * 1024);
        assert_eq!(wj[3 * 17 + 2], w.jitter[17][2]);
    }

    #[test]
    fn grid_covers_every_point_in_two_dimensions() {
        for &n in &[1usize, 63, 64, 65, 4096 * 64, 4096 * 64 + 1, 1 << 20, (1 << 24) + 17] {
            let (gx, gy, stride) = grid(n);
            assert_eq!(stride as usize, gx as usize * WORKGROUP);
            assert!(gx <= 4096 && gy <= 65535);
            assert!(gx as usize * gy as usize * WORKGROUP >= n, "{n}");
            assert!((gx as usize * (gy as usize - 1)) * WORKGROUP < n.max(1), "no whole row of idle workgroups: {n}");
        }
        assert_eq!(chunks(10, 4).collect::<Vec<_>>(), vec![(0, 4), (4, 4), (8, 2)]);
        assert!(chunks(0, 4).next().is_none());
    }

    #[test]
    fn the_chunk_size_fits_the_adapter() {
        let Some(gpu) = test_gpu() else { return };
        let c = max_chunk_points(&gpu);
        assert!(c >= CHUNK_ALIGN && c.is_multiple_of(CHUNK_ALIGN) && c <= MAX_CHUNK_POINTS);
        assert!(12 * c as u64 <= u64::from(gpu.limits.max_storage_buffer_binding_size));
        assert!(((12 * CHUNK_ALIGN) as u32).is_multiple_of(gpu.limits.min_storage_buffer_offset_alignment));
        assert!(((4 * CHUNK_ALIGN) as u32).is_multiple_of(gpu.limits.min_storage_buffer_offset_alignment));
    }

    #[test]
    fn fbm_matches_the_cpu_port_plain_and_billow() {
        let Some(gpu) = test_gpu() else { return };
        let pts = sphere_points(&mut Rng::new(1), 20_000);
        for &(octaves, billow) in &[(1usize, false), (3, false), (4, true), (7, false)] {
            let plan = FbmPlan::from_seed(40 + octaves as u64, 48.0, octaves, 0.65, DEFAULT_LACUNARITY, billow);
            let want = cpu_fbm(&pts, &plan);
            let got = fbm(&gpu, &pts, &plan);
            let (max, mean, over) = errors(&got, &want, 5e-3);
            eprintln!("fbm {octaves} octaves billow {billow}: max {max:e} mean {mean:e} over 5e-3: {over}");
            assert!(over == 0, "{octaves} octaves, billow {billow}: {over} points off by more than 5e-3 (max {max})");
            assert!(mean < 2e-4, "mean error {mean}");
        }
    }

    #[test]
    fn fbm_handles_the_corner_cases_of_the_plan() {
        let Some(gpu) = test_gpu() else { return };
        let pts = sphere_points(&mut Rng::new(2), 5000);
        // Zero, negative and tiny frequencies, a negative gain, a flat lacunarity.
        for &(freq, gain, lac) in &[(0.0f64, 0.5f64, 2.03f64), (-8.0, 0.5, 2.03), (0.5, 0.5, 2.03), (6.0, -0.5, 2.03), (6.0, 1.0, 1.0)] {
            let plan = FbmPlan::from_seed(9, freq, 3, gain, lac, false);
            let (max, _, over) = errors(&fbm(&gpu, &pts, &plan), &cpu_fbm(&pts, &plan), 2e-3);
            assert!(over == 0, "frequency {freq} gain {gain} lacunarity {lac}: max error {max}");
        }
        // Off the unit sphere and at the origin: only the 3-D position matters.
        let mut rng = Rng::new(3);
        let mut off = box_points(&mut rng, 4000, 8.0);
        off[0] = [0.0; 3];
        let plan = FbmPlan::from_seed(10, 5.0, 4, 0.5, 2.03, false);
        let (max, _, over) = errors(&fbm(&gpu, &off, &plan), &cpu_fbm(&off, &plan), 2e-3);
        assert!(over == 0, "off-sphere points: max error {max}");
    }

    #[test]
    fn a_chunked_run_is_the_unchunked_run() {
        let Some(gpu) = test_gpu() else { return };
        let k = NoiseKernels::new(&gpu);
        let pts = sphere_points(&mut Rng::new(4), 10_000);
        let plan = FbmPlan::from_seed(11, 30.0, 3, 0.5, 2.03, false);
        let whole = k.fbm(&pts, &plan);
        // Points are independent, so a prefix of the whole run is the whole run of the prefix: the tiny chunks
        // (one dispatch each) only get a prefix.
        for (chunk, len) in [(1usize, 150usize), (7, 700), (64, 10_000), (1000, 10_000), (4096, 10_000), (9999, 10_000), (10_000, 10_000), (50_000, 10_000)] {
            assert_eq!(k.fbm_with_chunk(&pts[..len], &plan, chunk), whole[..len], "chunk {chunk}");
        }
        let ce = CellEdgesPlan::from_seed(12, 20.0);
        let whole = k.cell_edges(&pts, &ce);
        for (chunk, len) in [(1usize, 150usize), (100, 10_000), (4097, 10_000), (10_000, 10_000)] {
            assert_eq!(k.cell_edges_with_chunk(&pts[..len], &ce, chunk), whole[..len], "chunk {chunk}");
        }
    }

    #[test]
    fn a_big_input_uses_rows_of_workgroups_and_stays_correct_to_the_last_point() {
        let Some(gpu) = test_gpu() else { return };
        // More than 4096 * 64 = 262144 points: a second row of workgroups, and a ragged last workgroup.
        let n = 4096 * 64 * 2 + 77;
        let pts = sphere_points(&mut Rng::new(5), n);
        let plan = FbmPlan::from_seed(13, 12.0, 2, 0.5, 2.03, false);
        let got = fbm(&gpu, &pts, &plan);
        assert_eq!(got.len(), n);
        for i in [0, 1, 4096 * 64 - 1, 4096 * 64, 4096 * 64 + 1, 2 * 4096 * 64, n - 1] {
            let want = cpu_fbm(&pts[i..=i], &plan)[0];
            assert!((got[i] - want).abs() < 2e-3, "point {i}: {} vs {want}", got[i]);
        }
        let (max, _, over) = errors(&got, &cpu_fbm(&pts, &plan), 2e-3);
        assert!(over == 0, "max error {max}");
    }

    #[test]
    fn the_result_can_stay_on_the_gpu_and_chunks_use_offset_bindings() {
        let Some(gpu) = test_gpu() else { return };
        let pts = sphere_points(&mut Rng::new(6), 3000);
        let plan = FbmPlan::from_seed(14, 25.0, 3, 0.5, 2.03, true);
        let direct = fbm(&gpu, &pts, &plan);
        let buf = upload_points(&gpu, &pts);
        let out = fbm_to_buffer(&gpu, &buf, pts.len(), &plan);
        assert_eq!(gpu.read_back::<f32>(&out, pts.len()), direct);
        // The first 1000 points of a larger buffer, and the cell_edges kernel.
        let out = fbm_to_buffer(&gpu, &buf, 1000, &plan);
        assert_eq!(gpu.read_back::<f32>(&out, 1000), direct[..1000]);
        let ce = CellEdgesPlan::from_seed(15, 9.0);
        let out = cell_edges_to_buffer(&gpu, &buf, pts.len(), &ce);
        assert_eq!(gpu.read_back::<f32>(&out, pts.len()), cell_edges(&gpu, &pts, &ce));
        // Chained: a following kernel reads the field straight from the buffer (the smoke kernel adds one).
        let add = gpu.pipeline("add", include_str!("../shaders/smoke.wgsl"), "main");
        let chained = gpu.storage_empty::<f32>("chained", pts.len());
        gpu.dispatch(&add, &[&out, &chained], (pts.len().div_ceil(64) as u32, 1, 1));
        let chained = gpu.read_back::<f32>(&chained, pts.len());
        let ce_cpu = cpu_cell_edges(&pts, &ce);
        assert!(chained.iter().zip(&ce_cpu).all(|(c, w)| (c - (w + 1.0)).abs() < 2e-3));
    }

    #[test]
    fn empty_inputs_give_empty_outputs() {
        let Some(gpu) = test_gpu() else { return };
        let plan = FbmPlan::from_seed(1, 3.0, 2, 0.5, 2.03, false);
        assert!(fbm(&gpu, &[], &plan).is_empty());
        assert!(cell_edges(&gpu, &[], &CellEdgesPlan::from_seed(1, 4.0)).is_empty());
        assert!(perlin3(&gpu, &Perlin3::from_seed(1), &[]).is_empty());
        let (a, b) = worley3(&gpu, &Worley3::from_seed(1), &[]);
        assert!(a.is_empty() && b.is_empty());
        let out = fbm_to_buffer(&gpu, &upload_points(&gpu, &[]), 0, &plan);
        assert!(out.size() >= 4);
    }

    #[test]
    fn perlin_matches_the_cpu_port_and_vanishes_on_the_lattice() {
        let Some(gpu) = test_gpu() else { return };
        let perlin = Perlin3::from_seed(21);
        // Random points of both signs: cells wrap by & 255 whatever the sign.
        let pts = box_points(&mut Rng::new(7), 50_000, 300.0);
        let got = perlin3(&gpu, &perlin, &pts);
        let want: Vec<f32> = pts.iter().map(|p| perlin.at(p[0], p[1], p[2])).collect();
        let (max, _, over) = errors(&got, &want, 2e-5);
        assert!(over == 0, "max error {max}");
        // Integer points, -0.0, huge integers: the noise is exactly 0 on the lattice (the cell index of a huge
        // f32 is 0 modulo 256 and must not overflow a conversion).
        let lattice = [[0.0f32, 0.0, 0.0], [3.0, -7.0, 12.0], [-256.0, 300.0, 5.0], [-0.0, -0.0, -0.0], [1e10, -3e9, 2.5e9]];
        assert!(perlin3(&gpu, &perlin, &lattice).iter().all(|&v| v == 0.0));
        // Period 256 on each axis in both signs, for points whose fractions are exact in f32.
        let base = [[0.25f32, 0.5, 0.75], [-3.375, 8.125, -0.40625], [100.875, -100.9375, 55.5]];
        let shifted: Vec<[f32; 3]> = base.iter().map(|p| [p[0] + 256.0, p[1] - 256.0, p[2] + 512.0]).collect();
        assert_eq!(perlin3(&gpu, &perlin, &base), perlin3(&gpu, &perlin, &shifted));
    }

    #[test]
    fn worley_matches_the_cpu_port_and_wraps_every_1024_cells() {
        let Some(gpu) = test_gpu() else { return };
        let w = Worley3::from_seed(22);
        let pts = box_points(&mut Rng::new(8), 50_000, 40.0);
        let (f1, f2) = worley3(&gpu, &w, &pts);
        let (c1, c2): (Vec<f32>, Vec<f32>) = pts.iter().map(|&p| w.at(p)).unzip();
        let (m1, _, o1) = errors(&f1, &c1, 2e-6);
        let (m2, _, o2) = errors(&f2, &c2, 2e-6);
        assert!(o1 == 0 && o2 == 0, "F1 max {m1}, F2 max {m2}");
        assert!(f1.iter().zip(&f2).all(|(a, b)| 0.0 <= *a && a <= b && *b < 1.8));
        // Period 1024 on each axis: points whose fractions are exact in f32, shifted by whole periods.
        let base = [[0.25f32, 0.5, 0.75], [-3.375, 8.125, -0.40625], [100.875, -100.9375, 55.5]];
        let shifted: Vec<[f32; 3]> = base.iter().map(|p| [p[0] + 1024.0, p[1] - 2048.0, p[2] + 3072.0]).collect();
        assert_eq!(worley3(&gpu, &w, &base), worley3(&gpu, &w, &shifted));
    }

    #[test]
    fn huge_coordinates_wrap_like_the_cpu_port() {
        let Some(gpu) = test_gpu() else { return };
        // f32 values past 2^24 are integers, past 2^31 multiples of 256: the lattice index must be taken
        // modulo 256 (Perlin) or 1024 (Worley) exactly, with no integer conversion that saturates.
        let w = Worley3::from_seed(23);
        let p = Perlin3::from_seed(24);
        let mut pts: Vec<[f32; 3]> = Vec::new();
        for &v in &[1.0e7f32, 16_777_217.0, 3.0e9, -2.5e9, 2.147_483_6e9, -2.147_483_6e9, 6.0e9, 1.0e10, -1.7e10, 1.0e20, -3.0e30] {
            pts.push([v, 0.0, 0.0]);
            pts.push([0.5, v, -0.25]);
            pts.push([v, -v, v]);
        }
        let (f1, f2) = worley3(&gpu, &w, &pts);
        let (c1, c2): (Vec<f32>, Vec<f32>) = pts.iter().map(|&q| w.at(q)).unzip();
        // The same cells and features (a wrong modulus would pick other jitter rows and move F1 by tenths); the
        // last bit of a squared distance may differ by a fused multiply-add.
        let (m1, _, o1) = errors(&f1, &c1, 1e-6);
        let (m2, _, o2) = errors(&f2, &c2, 1e-6);
        assert!(o1 == 0 && o2 == 0, "Worley on huge coordinates: F1 off by {m1}, F2 off by {m2}");
        let got = perlin3(&gpu, &p, &pts);
        let want: Vec<f32> = pts.iter().map(|q| p.at(q[0], q[1], q[2])).collect();
        assert_eq!(got, want, "Perlin on huge coordinates");
    }

    #[test]
    fn worley_ties_on_a_centered_lattice_give_f2_equal_f1() {
        let Some(gpu) = test_gpu() else { return };
        // Identity permutation, every jitter 0.5: features at the cell centers.
        let perm: Vec<i64> = (0..1024).collect();
        let w = Worley3::new(&perm, &vec![[0.5; 3]; 1024]);
        let (f1, f2) = worley3(&gpu, &w, &[[0.5, 0.5, 0.5], [1.0, 0.5, 0.5], [1.0, 1.0, 1.0]]);
        let near = |a: f32, b: f32| (a - b).abs() < 1e-6;
        assert!(near(f1[0], 0.0) && near(f2[0], 1.0));
        assert!(near(f1[1], 0.5) && f2[1] == f1[1], "on a face: two features at 0.5");
        assert!(near(f1[2], 0.75f32.sqrt()) && f2[2] == f1[2], "at a corner: eight features at sqrt(3)/2");
    }

    #[test]
    fn cell_edges_matches_the_cpu_port_and_collapses_at_frequency_zero() {
        let Some(gpu) = test_gpu() else { return };
        let pts = sphere_points(&mut Rng::new(9), 30_000);
        for &freq in &[0.5f64, 8.0, 50.0, -8.0, 200.0] {
            let plan = CellEdgesPlan::from_seed(30, freq);
            let want = cpu_cell_edges(&pts, &plan);
            let got = cell_edges(&gpu, &pts, &plan);
            // A rounding difference at q ~ 200 moves a point by 1.5e-5 of a cell; F2 - F1 has slope up to 2.
            let tol = 1e-4 + 5e-6 * freq.abs() as f32;
            let (max, mean, over) = errors(&got, &want, tol);
            eprintln!("cell_edges f {freq}: max {max:e} mean {mean:e} over {tol:e}: {over}");
            // A point sitting within an ulp of a hash-cell border can resolve to the neighbor feature on one side
            // (F2 - F1 is continuous, so the error stays of the size of the rounding): no bounded outlier count.
            assert!(over == 0, "frequency {freq}: {over} points off by more than {tol} (max {max})");
            assert!(got.iter().all(|&v| v >= 0.0));
        }
        let flat = cell_edges(&gpu, &pts, &CellEdgesPlan::from_seed(31, 0.0));
        assert!(flat.iter().all(|&v| v == flat[0]));
    }

    #[test]
    fn plans_with_unusual_octave_counts_run() {
        let Some(gpu) = test_gpu() else { return };
        let pts = sphere_points(&mut Rng::new(10), 2000);
        // 24 octaves (the most a test draws) and a hand-built plan with one octave at the identity rotation.
        let plan = FbmPlan::from_seed(40, 3.0, 24, 0.5, 1.3, false);
        let (max, _, over) = errors(&fbm(&gpu, &pts, &plan), &cpu_fbm(&pts, &plan), 2e-3);
        assert!(over == 0, "max {max}");
        let id = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let plan = FbmPlan::new(Perlin3::from_seed(5), vec![FbmOctave { amp: 1.0, freq: 5.0, rot: id, off: [0.5, 0.25, 0.125] }], false);
        let (max, _, over) = errors(&fbm(&gpu, &pts, &plan), &cpu_fbm(&pts, &plan), 1e-4);
        assert!(over == 0, "max {max}");
    }

    /// The accuracy of every kernel against the CPU port at one million points: prints the max and mean error.
    #[test]
    fn accuracy_against_the_cpu_port_at_one_million_points() {
        let Some(gpu) = test_gpu() else { return };
        let k = NoiseKernels::new(&gpu);
        let pts = sphere_points(&mut Rng::new(100), 1_000_000);
        let mut report = String::new();
        // The fine ladder of a 4096 map (48 .. 310, 3 octaves, gain 0.65) and of an 8192 map (4 octaves).
        for &(freq, octaves) in &[(75.0f64, 3usize), (75.0, 4)] {
            let plan = FbmPlan::from_seed(1, freq, octaves, 0.65, DEFAULT_LACUNARITY, false);
            let (max, mean, over) = errors(&k.fbm(&pts, &plan), &cpu_fbm(&pts, &plan), 1e-3);
            report += &format!("fbm f {freq} {octaves} oct: max {max:e} mean {mean:e} over 1e-3: {over}\n");
            assert!(max < 2e-2 && mean < 1e-4, "{report}");
        }
        let plan = FbmPlan::from_seed(2, 75.0, 4, 0.65, DEFAULT_LACUNARITY, true);
        let (max, mean, _) = errors(&k.fbm(&pts, &plan), &cpu_fbm(&pts, &plan), 1e-3);
        report += &format!("fbm billow f 75 4 oct: max {max:e} mean {mean:e}\n");
        assert!(max < 2e-2 && mean < 1e-4, "{report}");
        for &freq in &[106.0f64, 424.0] {
            let plan = CellEdgesPlan::from_seed(3, freq);
            let (max, mean, _) = errors(&k.cell_edges(&pts, &plan), &cpu_cell_edges(&pts, &plan), 1e-3);
            report += &format!("cell_edges f {freq}: max {max:e} mean {mean:e}\n");
            assert!(max < 2e-2 && mean < 1e-4, "{report}");
        }
        let perlin = Perlin3::from_seed(4);
        let spread = box_points(&mut Rng::new(101), 1_000_000, 50.0);
        let want: Vec<f32> = spread.iter().map(|p| perlin.at(p[0], p[1], p[2])).collect();
        let (max, mean, _) = errors(&k.perlin3(&perlin, &spread), &want, 1e-3);
        report += &format!("perlin3 in [-50, 50]^3: max {max:e} mean {mean:e}\n");
        assert!(max < 1e-4 && mean < 1e-7, "{report}");
        let worley = Worley3::from_seed(5);
        let (g1, g2) = k.worley3(&worley, &spread);
        let (c1, c2): (Vec<f32>, Vec<f32>) = spread.iter().map(|&p| worley.at(p)).unzip();
        let (m1, a1, _) = errors(&g1, &c1, 1e-3);
        let (m2, a2, _) = errors(&g2, &c2, 1e-3);
        report += &format!("worley3 F1 in [-50, 50]^3: max {m1:e} mean {a1:e}\nworley3 F2 in [-50, 50]^3: max {m2:e} mean {a2:e}\n");
        assert!(m1 < 1e-4 && m2 < 1e-4, "{report}");
        eprintln!("accuracy at 1M points against the CPU port:\n{report}");
    }
}
