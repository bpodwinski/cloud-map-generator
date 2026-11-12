//! GPU kernels of the per-texel stages of `generate()` that every map needs, ported from the CPU port
//! (`cloud_map_generator_core::sphere`, `generate`, `texture::standardize`); the WGSL is `shaders/stages.wgsl`.
//!
//! * [`StageKernels::grid_dirs_to_buffer`]: `grid_dirs` of the whole map or of a band of rows.
//! * [`StageKernels::upsample_dirs_to_buffer`]: `upsample_dirs`, the coarse back-trace grid sampled bilinearly on
//!   the 3D vectors and renormalized.
//! * [`StageKernels::mix_dirs_to_buffer`] / [`StageKernels::along_to_buffer`]: `normalize(a ka + b kb)` and
//!   `along(p, back, f)`.
//! * [`StageKernels::lat_of_dirs_to_buffer`]: `asin(clip(z, -1, 1))`, the latitude of the back-traced direction.
//! * [`StageKernels::moisture_profile_to_buffer`] and [`StageKernels::synoptic_to_buffer`]: `moisture_profile`
//!   (`np.degrees` in f32 as the CPU port does) and the assembly of the synoptic field before standardization.
//! * [`StageKernels::moments`] and [`StageKernels::standardize_to_buffer`]: the mean and standard deviation of
//!   a whole field (a hierarchical f32 reduction on the GPU, the final combination in f64 on the CPU) and
//!   `(x - mean) / std`.
//!
//! # Data model
//!
//! Everything is f32 (WGSL has no f64). A direction field is `array<f32>` with a stride of 3 (a `vec3` would pad
//! every texel to 16 bytes). A buffer holds one BAND of the map, a run of `rows` whole rows starting at map row
//! `row0` (`item = row_in_band * width + column`), so the pipeline can drive a map that does not fit in one
//! buffer: a vec3 field of an 8192 x 4096 map is 402 MB, past the 256 MiB of `maxBufferSize` of the WebGPU
//! defaults. [`StageKernels::rows_per_chunk`] says how many rows a band may have; the whole map is one band
//! whenever it fits. Statistics over the whole map ([`Moments`]) are accumulated across bands with
//! [`Moments::merge`].
//!
//! # Limits
//!
//! The kernels respect the WebGPU default limits (`maxStorageBufferBindingSize` 128 MiB, `maxBufferSize`
//! 256 MiB, 65535 workgroups per dimension, 8 storage buffers per stage, 4 bind groups, 16 KiB of workgroup
//! memory) whatever the adapter offers: a buffer bound to a kernel is cut into chunks of at most
//! [`StageKernels::binding_bytes`] bytes (default: the adapter's limit capped at 128 MiB) and bound with
//! offsets, and a dispatch is tiled in 2-D. [`StageKernels::with_binding_bytes`] forces smaller chunks (tests).
//! The two exceptions are tables bound whole: the cos/sin tables of `grid_dirs` (`W + rows` pairs) and the coarse
//! grid of `upsample_dirs` (the pipeline's is `structure_width` = 512 columns, 1.5 MB; at most one binding, 128 MiB,
//! so up to 2896 x 1448 coarse cells).
//!
//! # Host tables
//!
//! `grid_dirs` takes the cosine and sine of the longitude of every column and of the latitude of every row of
//! the band from tables the host computes in f64 and rounds to f32 (`W + rows` trigonometric pairs for a map of
//! `W x rows` texels; the per-texel work is two products on the GPU). That is what makes the GPU direction as
//! close to the f64 direction the CPU port casts as two roundings allow.

use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use cloud_map_generator_core::sphere::grid_lonlat;
use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroU64;

const SHADER: &str = include_str!("../shaders/stages.wgsl");

/// Threads per workgroup of the elementwise kernels (`@workgroup_size` of `stages.wgsl`).
pub(crate) const WORKGROUP: usize = 64;
/// Workgroups along x of a dispatch: a dispatch is `groups_x * groups_y` workgroups over a 2-D grid.
const MAX_GROUPS_X: usize = 4096;
/// Chunk sizes (in items) are multiples of this: a chunk offset of `items * 4 * stride` bytes is then a multiple
/// of the 256-byte storage-buffer offset alignment of every adapter, whatever the stride (1 to 4 f32).
const CHUNK_ALIGN: usize = 1024;
/// The WebGPU default `maxStorageBufferBindingSize`: no binding of this crate's stage kernels exceeds it.
pub const PORTABLE_BINDING_BYTES: u64 = 128 << 20;
/// Values per workgroup of the moments kernel (256 threads, 16 values each).
pub const MOMENTS_TILE: usize = 4096;

/// The `Params` uniform of the stage kernels (and, with the same layout, of the texture kernels): 96 bytes.
/// `u` and `k` are the kernel-specific slots, named by each entry point of the WGSL.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(crate) struct Params {
    pub n: u32,
    pub row_stride: u32,
    pub base: u32,
    pub width: u32,
    pub u: [u32; 4],
    pub k: [f32; 16],
}

impl Params {
    pub(crate) fn with_k(k: &[f32]) -> Params {
        assert!(k.len() <= 16, "at most 16 kernel scalars");
        let mut p = Params::default();
        p.k[..k.len()].copy_from_slice(k);
        p
    }
}

/// One buffer of a dispatch.
#[derive(Clone, Copy)]
pub(crate) enum Arg<'a> {
    /// A per-item buffer of `stride` f32 per item: bound as the sub-range of the current chunk.
    Items(&'a wgpu::Buffer, usize),
    /// A table (or any buffer the kernel indexes freely), bound whole.
    Whole(&'a wgpu::Buffer),
}

/// `(groups_x, groups_y, row_stride)` for `n` threads: a 2-D grid of workgroups.
pub(crate) fn grid(n: usize) -> (u32, u32, u32) {
    let groups = n.div_ceil(WORKGROUP).max(1);
    let gx = groups.min(MAX_GROUPS_X);
    let gy = groups.div_ceil(gx);
    assert!(gy <= 65535, "{n} threads need more than 65535 rows of workgroups: chunk the input");
    (gx as u32, gy as u32, (gx * WORKGROUP) as u32)
}

/// The largest storage binding the kernels of `gpu` make by default: the adapter's own limits, capped at the
/// WebGPU default of 128 MiB so that the same WGSL and the same chunking run in a browser.
pub fn portable_binding_bytes(gpu: &GpuContext) -> u64 {
    u64::from(gpu.limits.max_storage_buffer_binding_size).min(gpu.limits.max_buffer_size).min(PORTABLE_BINDING_BYTES)
}

/// How many rows of a `width`-texel map fit one buffer of `floats_per_texel` f32 per texel under the portable
/// limits of `gpu` (at least 1). The band size of a pipeline that drives a map too large for one buffer.
pub fn rows_per_chunk(gpu: &GpuContext, width: usize, floats_per_texel: usize) -> usize {
    rows_per_binding(portable_binding_bytes(gpu), width, floats_per_texel)
}

fn rows_per_binding(binding_bytes: u64, width: usize, floats_per_texel: usize) -> usize {
    assert!(width > 0 && floats_per_texel > 0);
    let row_bytes = (width * floats_per_texel * 4) as u64;
    usize::try_from(binding_bytes / row_bytes).unwrap_or(usize::MAX).max(1)
}

// ---------------------------------------------------------------------------------------------------------
// Moments

/// The count, mean and sum of squared deviations (M2) of a field: what is needed to standardize it. Statistics
/// of the pieces (bands, workgroups) are combined with [`Moments::merge`] (Chan et al.), in f64.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moments {
    pub count: u64,
    pub mean: f64,
    /// Sum of `(x - mean)^2`.
    pub m2: f64,
}

impl Moments {
    /// No values.
    pub const EMPTY: Moments = Moments { count: 0, mean: 0.0, m2: 0.0 };

    /// The statistics of the union of the two sets (Chan's parallel update).
    pub fn merge(self, other: Moments) -> Moments {
        if self.count == 0 {
            return other;
        }
        if other.count == 0 {
            return self;
        }
        let (na, nb) = (self.count as f64, other.count as f64);
        let n = na + nb;
        let delta = other.mean - self.mean;
        Moments { count: self.count + other.count, mean: self.mean + delta * nb / n, m2: self.m2 + other.m2 + delta * delta * na * nb / n }
    }

    /// `ndarray.var()` (ddof 0): `M2 / count` (NaN for no values).
    pub fn variance(&self) -> f64 {
        self.m2 / self.count as f64
    }

    /// `ndarray.std()` (ddof 0).
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// The divisor of `standardize`: the std, or 1 where it is not strictly positive (numpy's rule of the CPU
    /// port: no epsilon, so a constant field gives `x - mean`, not NaN).
    pub fn divisor(&self) -> f64 {
        let sd = self.std();
        if sd > 0.0 {
            sd
        } else {
            1.0
        }
    }

    /// The exact statistics of host values, in f64 (two passes, summed in blocks): the reference the GPU
    /// reduction is measured against.
    pub fn of_slice(values: &[f32]) -> Moments {
        let n = values.len();
        if n == 0 {
            return Moments::EMPTY;
        }
        let block_sum = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>();
        let total: f64 = values.chunks(4096).map(block_sum).sum();
        let mean = total / n as f64;
        let m2: f64 = values
            .chunks(4096)
            .map(|b| {
                let s: f64 = b.iter().map(|&x| (f64::from(x) - mean) * (f64::from(x) - mean)).sum();
                s
            })
            .sum();
        Moments { count: n as u64, mean, m2 }
    }
}

/// `(hi, lo)` of an f64 as two f32 whose sum is the f64 to 48 bits.
pub(crate) fn split(x: f64) -> (f32, f32) {
    let hi = x as f32;
    (hi, (x - f64::from(hi)) as f32)
}

// ---------------------------------------------------------------------------------------------------------
// The kernels

/// The inputs of [`StageKernels::synoptic_to_buffer`]: the Hadley edge and the four weights of the synoptic
/// field `s = a_moisture (moisture_profile(lat_back) + moisture_noise fbm) + a_ascent ascent + a_mid fbm_mid`
/// (the `GenParams` fields of the same names).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SynopticParams {
    pub hadley_deg: f64,
    pub a_moisture: f64,
    pub moisture_noise: f64,
    pub a_ascent: f64,
    pub a_mid: f64,
}

/// The compute pipelines of the stage kernels, each built on first use (building one compiles the shader: keep
/// one `StageKernels` for repeated calls). Also holds the chunk limit of the kernels.
pub struct StageKernels<'a> {
    gpu: &'a GpuContext,
    module: wgpu::ShaderModule,
    pipelines: RefCell<HashMap<&'static str, wgpu::ComputePipeline>>,
    binding_bytes: u64,
}

impl<'a> StageKernels<'a> {
    /// The kernels with the portable chunk limit ([`portable_binding_bytes`]).
    pub fn new(gpu: &'a GpuContext) -> StageKernels<'a> {
        StageKernels::with_binding_bytes(gpu, portable_binding_bytes(gpu))
    }

    /// The kernels with an explicit largest storage binding, in bytes (a test of the chunking forces it small).
    /// At least 16 KiB, so that a chunk holds one aligned group of items of every stride.
    pub fn with_binding_bytes(gpu: &'a GpuContext, binding_bytes: u64) -> StageKernels<'a> {
        assert!(binding_bytes >= (CHUNK_ALIGN * 16) as u64, "a chunk of {CHUNK_ALIGN} items of 4 f32 needs {} bytes", CHUNK_ALIGN * 16);
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloudgen stages"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });
        StageKernels { gpu, module, pipelines: RefCell::new(HashMap::new()), binding_bytes }
    }

    pub fn gpu(&self) -> &'a GpuContext {
        self.gpu
    }

    /// The largest storage binding these kernels make (their chunk limit).
    pub fn binding_bytes(&self) -> u64 {
        self.binding_bytes
    }

    /// Rows of a `width`-texel map per band of `floats_per_texel` f32 per texel under this kernel set's limit.
    pub fn rows_per_chunk(&self, width: usize, floats_per_texel: usize) -> usize {
        rows_per_binding(self.binding_bytes, width, floats_per_texel)
    }

    fn pipeline(&self, entry: &'static str) -> wgpu::ComputePipeline {
        self.pipelines
            .borrow_mut()
            .entry(entry)
            .or_insert_with(|| {
                self.gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &self.module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .clone()
    }

    /// Items per chunk for these arguments: the largest multiple of [`CHUNK_ALIGN`] whose widest per-item buffer
    /// fits the binding limit.
    pub(crate) fn chunk_items(&self, args: &[(u32, Arg<'_>)]) -> usize {
        let stride = args.iter().filter_map(|(_, a)| if let Arg::Items(_, s) = a { Some(*s) } else { None }).max().unwrap_or(1);
        let items = usize::try_from(self.binding_bytes / (4 * stride as u64)).unwrap_or(usize::MAX);
        let chunk = items / CHUNK_ALIGN * CHUNK_ALIGN;
        assert!(chunk >= CHUNK_ALIGN, "the binding limit of {} bytes holds no chunk of {stride}-float items", self.binding_bytes);
        chunk
    }

    /// Records the dispatches of `pipeline` over `n` items, one per chunk, in one compute pass. `args` are
    /// `(binding, buffer)`; binding 0 is the `Params` uniform, filled per chunk (`n`, `row_stride`, `base`).
    /// Returns without waiting for the GPU.
    pub(crate) fn run_items(&self, pipeline: &wgpu::ComputePipeline, params: &Params, args: &[(u32, Arg<'_>)], n: usize) {
        if n == 0 {
            return;
        }
        for (b, a) in args {
            if let Arg::Items(buf, stride) = a {
                assert!(buf.size() >= (n * stride * 4) as u64, "binding {b}: the buffer holds {} bytes, {n} items of {stride} f32 need {}", buf.size(), n * stride * 4);
            }
        }
        let chunk = self.chunk_items(args);
        let layout = pipeline.get_bind_group_layout(0);
        let mut uniforms = Vec::new();
        let mut binds = Vec::new();
        for start in (0..n).step_by(chunk) {
            let len = chunk.min(n - start);
            let (gx, gy, row_stride) = grid(len);
            let p = Params { n: len as u32, row_stride, base: start as u32, ..*params };
            let uniform = self.gpu.uniform_with("stage params", &p);
            let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }];
            for (binding, a) in args {
                let resource = match a {
                    Arg::Whole(buf) => buf.as_entire_binding(),
                    Arg::Items(buf, stride) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf,
                        offset: (start * stride * 4) as u64,
                        size: NonZeroU64::new((len * stride * 4) as u64),
                    }),
                };
                entries.push(wgpu::BindGroupEntry { binding: *binding, resource });
            }
            let bind = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("stage"), layout: &layout, entries: &entries });
            uniforms.push(uniform);
            binds.push((bind, gx, gy));
        }
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            for (bind, gx, gy) in &binds {
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(*gx, *gy, 1);
            }
        }
        self.gpu.queue.submit([enc.finish()]);
    }

    fn run(&self, entry: &'static str, params: &Params, args: &[(u32, Arg<'_>)], n: usize) {
        self.run_items(&self.pipeline(entry), params, args, n);
    }

    // -- uploads -----------------------------------------------------------------------------------------------

    /// Uploads directions (3 f32 each) to a storage buffer.
    pub fn upload_dirs(&self, dirs: &[[f32; 3]]) -> wgpu::Buffer {
        self.gpu.storage_with("stage dirs", &dirs.iter().flatten().copied().collect::<Vec<f32>>())
    }

    /// Uploads a field (1 f32 per item) to a storage buffer.
    pub fn upload_field(&self, values: &[f32]) -> wgpu::Buffer {
        self.gpu.storage_with("stage field", if values.is_empty() { &[0.0] } else { values })
    }

    fn output(&self, label: &str, n: usize, stride: usize) -> wgpu::Buffer {
        self.gpu.storage_empty::<f32>(label, (n * stride).max(1))
    }

    /// A buffer of `n` directions read back to the host.
    pub fn dirs_to_vec(&self, buffer: &wgpu::Buffer, n: usize) -> Vec<[f32; 3]> {
        if n == 0 {
            return Vec::new();
        }
        self.gpu.read_back::<f32>(buffer, 3 * n).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    }

    /// A buffer of `n` f32 read back to the host.
    pub fn field_to_vec(&self, buffer: &wgpu::Buffer, n: usize) -> Vec<f32> {
        if n == 0 {
            return Vec::new();
        }
        self.gpu.read_back::<f32>(buffer, n)
    }

    // -- grid_dirs ---------------------------------------------------------------------------------------------

    /// The latitude (radians, f32) of the `rows` rows of the band starting at map row `row0`, for a map of
    /// `height` rows: `pi/2 - (r + 0.5) pi / height` in f64, rounded to f32 (the per-row latitude the weights
    /// kernels read). The f64 table has one entry per row of the band, not per texel.
    pub fn grid_lat_rows(&self, width: usize, height: usize, row0: usize, rows: usize) -> Vec<f32> {
        assert!(row0 + rows <= height, "rows {row0}..{} are outside the {height} rows of the map", row0 + rows);
        let (_, lat) = grid_lonlat(width, height);
        lat[row0..row0 + rows].iter().map(|&l| l as f32).collect()
    }

    /// [`StageKernels::grid_lat_rows`] uploaded: the per-row latitude buffer of the weights kernel
    /// ([`crate::texture::LatInput::Rows`]).
    pub fn grid_lat_rows_to_buffer(&self, width: usize, height: usize, row0: usize, rows: usize) -> wgpu::Buffer {
        self.upload_field(&self.grid_lat_rows(width, height, row0, rows))
    }

    /// `grid_dirs(width, height)` of the CPU port for the band of `rows` rows from map row `row0`, on the GPU.
    pub fn grid_dirs_to_buffer(&self, width: usize, height: usize, row0: usize, rows: usize) -> wgpu::Buffer {
        assert!(row0 + rows <= height, "rows {row0}..{} are outside the {height} rows of the map", row0 + rows);
        let n = width * rows;
        let out = self.output("grid dirs", n, 3);
        if n == 0 {
            return out;
        }
        let (lon, lat) = grid_lonlat(width, height);
        let cols: Vec<f32> = lon.iter().flat_map(|&l| [l.cos() as f32, l.sin() as f32]).collect();
        let rws: Vec<f32> = lat[row0..row0 + rows].iter().flat_map(|&l| [l.cos() as f32, l.sin() as f32]).collect();
        let (cols, rws) = (self.upload_field(&cols), self.upload_field(&rws));
        let params = Params { width: width as u32, ..Params::default() };
        self.run("grid_dirs_main", &params, &[(1, Arg::Whole(&cols)), (2, Arg::Whole(&rws)), (5, Arg::Items(&out, 3))], n);
        out
    }

    /// [`StageKernels::grid_dirs_to_buffer`] read back to the host.
    pub fn grid_dirs_to_vec(&self, width: usize, height: usize, row0: usize, rows: usize) -> Vec<[f32; 3]> {
        self.dirs_to_vec(&self.grid_dirs_to_buffer(width, height, row0, rows), width * rows)
    }

    // -- upsample_dirs -----------------------------------------------------------------------------------------

    /// `upsample_dirs(coarse, src_h, src_w, width, height)` of the CPU port for the band of `rows` rows from map
    /// row `row0` of the `width x height` target, on the GPU: `coarse` is the `src_w x src_h` grid of unit vectors
    /// (3 f32 each, row-major, north-up: the back-trace on the coarse grid), the result the band's directions.
    #[allow(clippy::too_many_arguments)]
    pub fn upsample_dirs_to_buffer(
        &self,
        coarse: &wgpu::Buffer,
        src_w: usize,
        src_h: usize,
        width: usize,
        height: usize,
        row0: usize,
        rows: usize,
    ) -> wgpu::Buffer {
        assert!(src_w > 0 && src_h > 0, "the coarse grid is empty");
        assert!(row0 + rows <= height, "rows {row0}..{} are outside the {height} rows of the map", row0 + rows);
        assert!(coarse.size() >= (12 * src_w * src_h) as u64, "the coarse buffer holds {} bytes, {src_w} x {src_h} directions need {}", coarse.size(), 12 * src_w * src_h);
        assert!(coarse.size() <= self.binding_bytes, "the coarse grid is bound whole and exceeds the {} byte binding limit", self.binding_bytes);
        assert!(2 * width * src_w.max(src_h) < i32::MAX as usize && 2 * height * src_w.max(src_h) < i32::MAX as usize, "the sampling positions overflow i32");
        let n = width * rows;
        let out = self.output("upsampled dirs", n, 3);
        let params = Params { width: width as u32, u: [row0 as u32, height as u32, src_w as u32, src_h as u32], ..Params::default() };
        self.run("upsample_dirs_main", &params, &[(1, Arg::Whole(coarse)), (5, Arg::Items(&out, 3))], n);
        out
    }

    /// [`StageKernels::upsample_dirs_to_buffer`] of host directions, the whole target, read back.
    pub fn upsample_dirs_to_vec(&self, coarse: &[[f32; 3]], src_w: usize, src_h: usize, width: usize, height: usize) -> Vec<[f32; 3]> {
        assert_eq!(coarse.len(), src_w * src_h);
        let c = self.upload_dirs(coarse);
        self.dirs_to_vec(&self.upsample_dirs_to_buffer(&c, src_w, src_h, width, height, 0, height), width * height)
    }

    // -- mix_dirs, along ---------------------------------------------------------------------------------------

    /// `normalize(a ka + b kb)` per direction (`ka`, `kb` rounded to f32 once): the mixes of the regime texture.
    pub fn mix_dirs_to_buffer(&self, a: &wgpu::Buffer, b: &wgpu::Buffer, n: usize, ka: f32, kb: f32) -> wgpu::Buffer {
        let out = self.output("mixed dirs", n, 3);
        let params = Params::with_k(&[ka, kb]);
        self.run("mix_dirs_main", &params, &[(1, Arg::Items(a, 3)), (2, Arg::Items(b, 3)), (5, Arg::Items(&out, 3))], n);
        out
    }

    /// `along(p, back, f)` of the CPU port: `normalize(p (1 - f) + back f)` per direction, `1 - f` formed in f64
    /// and both weights rounded to f32. `f` is not clipped.
    pub fn along_to_buffer(&self, p: &wgpu::Buffer, back: &wgpu::Buffer, n: usize, f: f64) -> wgpu::Buffer {
        self.mix_dirs_to_buffer(p, back, n, (1.0 - f) as f32, f as f32)
    }

    /// [`StageKernels::along_to_buffer`] of host directions, read back.
    pub fn along_to_vec(&self, p: &[[f32; 3]], back: &[[f32; 3]], f: f64) -> Vec<[f32; 3]> {
        assert_eq!(p.len(), back.len(), "p and back hold the same number of directions");
        let (pb, bb) = (self.upload_dirs(p), self.upload_dirs(back));
        self.dirs_to_vec(&self.along_to_buffer(&pb, &bb, p.len(), f), p.len())
    }

    // -- lat_of_dirs -------------------------------------------------------------------------------------------

    /// `asin(clip(z, -1, 1))` of `n` directions (radians): the latitude of the back-traced direction. `asin` is a
    /// polynomial of the shader (about 1e-7 absolute), not the adapter's builtin.
    pub fn lat_of_dirs_to_buffer(&self, dirs: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
        let out = self.output("latitude", n, 1);
        self.run("lat_of_dirs_main", &Params::default(), &[(1, Arg::Items(dirs, 3)), (5, Arg::Items(&out, 1))], n);
        out
    }

    // -- moisture_profile, synoptic ----------------------------------------------------------------------------

    /// The two scalars of `np.degrees` and of the Hadley edge as the CPU port rounds them: `f32(180) / f32(pi)`
    /// (formed in f32: numpy's `npy_degrees` of a float32) and `f32(hadley_deg)`.
    fn degrees_and_hadley(hadley_deg: f64) -> [f32; 2] {
        [180.0_f32 / std::f64::consts::PI as f32, hadley_deg as f32]
    }

    /// `moisture_profile(lat, hadley_deg)` of the CPU port in f32, on the GPU: `lat` is `n` latitudes in radians.
    pub fn moisture_profile_to_buffer(&self, lat: &wgpu::Buffer, n: usize, hadley_deg: f64) -> wgpu::Buffer {
        let out = self.output("moisture profile", n, 1);
        let params = Params::with_k(&Self::degrees_and_hadley(hadley_deg));
        self.run("moisture_profile_main", &params, &[(1, Arg::Items(lat, 1)), (5, Arg::Items(&out, 1))], n);
        out
    }

    /// [`StageKernels::moisture_profile_to_buffer`] of host latitudes, read back.
    pub fn moisture_profile_to_vec(&self, lat: &[f32], hadley_deg: f64) -> Vec<f32> {
        let l = self.upload_field(lat);
        self.field_to_vec(&self.moisture_profile_to_buffer(&l, lat.len(), hadley_deg), lat.len())
    }

    /// The synoptic field `s` BEFORE its standardization, in the CPU port's order of operations: the moisture
    /// profile of `lat_back`, then `a_moisture (profile + moisture_noise fbm_moisture)`, `+ a_ascent ascent`,
    /// `+ a_mid fbm_mid`. The four inputs are fields of `n` f32 (the fBm of the moisture texture read at the back
    /// traced position, the flow's ascent, the fBm of the mid-scale texture read along the flow). Standardize
    /// the result with [`StageKernels::moments`] and [`StageKernels::standardize_to_buffer`].
    pub fn synoptic_to_buffer(
        &self,
        lat_back: &wgpu::Buffer,
        moisture_noise: &wgpu::Buffer,
        ascent: &wgpu::Buffer,
        mid_noise: &wgpu::Buffer,
        n: usize,
        p: &SynopticParams,
    ) -> wgpu::Buffer {
        let out = self.output("synoptic", n, 1);
        let [deg, had] = Self::degrees_and_hadley(p.hadley_deg);
        let params = Params::with_k(&[deg, had, p.a_moisture as f32, p.moisture_noise as f32, p.a_ascent as f32, p.a_mid as f32]);
        let args = [
            (1, Arg::Items(lat_back, 1)),
            (2, Arg::Items(moisture_noise, 1)),
            (3, Arg::Items(ascent, 1)),
            (4, Arg::Items(mid_noise, 1)),
            (5, Arg::Items(&out, 1)),
        ];
        self.run("synoptic_main", &params, &args, n);
        out
    }

    // -- moments, standardize ----------------------------------------------------------------------------------

    /// The mean and the sum of squared deviations of the first `n` f32 of a field, in f64 from the partial
    /// results of a hierarchical f32 reduction (see the header of `stages.wgsl`): one pass over the data, one
    /// small readback (3 f32 per 4096 values). A field bound past the chunk limit is reduced chunk by chunk.
    #[maybe_async::maybe_async]
    pub async fn moments(&self, field: &wgpu::Buffer, n: usize) -> Moments {
        if n == 0 {
            return Moments::EMPTY;
        }
        assert!(field.size() >= 4 * n as u64, "the field buffer holds {} bytes, {n} values need {}", field.size(), 4 * n);
        let pipeline = self.pipeline("moments_main");
        let layout = pipeline.get_bind_group_layout(0);
        // A chunk is a whole number of tiles, at least one aligned group of items.
        let chunk = (self.binding_bytes / 4) as usize / MOMENTS_TILE * MOMENTS_TILE;
        assert!(chunk >= MOMENTS_TILE.max(CHUNK_ALIGN), "the binding limit of {} bytes holds no tile of {MOMENTS_TILE} values", self.binding_bytes);
        let mut parts = Vec::new(); // (start, len, partials buffer)
        let mut keep = Vec::new();
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for start in (0..n).step_by(chunk) {
                let len = chunk.min(n - start);
                let groups = len.div_ceil(MOMENTS_TILE);
                assert!(groups <= 65535, "{groups} workgroups in one dispatch");
                let partials = self.gpu.storage_empty::<f32>("moments partials", 3 * groups);
                let uniform = self.gpu.uniform_with("moments params", &Params { n: len as u32, ..Params::default() });
                let entries = [
                    wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: field,
                            offset: 4 * start as u64,
                            size: NonZeroU64::new(4 * len as u64),
                        }),
                    },
                    wgpu::BindGroupEntry { binding: 5, resource: partials.as_entire_binding() },
                ];
                let bind = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("moments"), layout: &layout, entries: &entries });
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(groups as u32, 1, 1);
                keep.push((uniform, bind));
                parts.push((len, groups, partials));
            }
        }
        self.gpu.queue.submit([enc.finish()]);
        let mut total = Moments::EMPTY;
        for (len, groups, partials) in &parts {
            let flat = self.gpu.read_back_maybe::<f32>(partials, 3 * groups).await;
            for (g, tile) in flat.chunks_exact(3).enumerate() {
                let count = MOMENTS_TILE.min(len - g * MOMENTS_TILE) as f64;
                let (c, s1, s2) = (f64::from(tile[0]), f64::from(tile[1]), f64::from(tile[2]));
                // Corrected two-pass statistics of the tile about the reference c.
                let m = Moments { count: count as u64, mean: c + s1 / count, m2: (s2 - s1 * s1 / count).max(0.0) };
                total = total.merge(m);
            }
        }
        total
    }

    /// The statistics of host values through the GPU reduction.
    #[maybe_async::maybe_async]
    pub async fn moments_of_slice(&self, values: &[f32]) -> Moments {
        self.moments(&self.upload_field(values), values.len()).await
    }

    /// The scalars of the standardize kernel for these statistics: the mean as (hi, lo) and the divisor.
    fn standardize_params(m: &Moments) -> Params {
        let (hi, lo) = split(m.mean);
        Params::with_k(&[hi, lo, m.divisor() as f32])
    }

    /// `(x - mean) / std` of the first `n` values of a field with the statistics `m` (computed over the whole
    /// array the field is part of: see [`StageKernels::moments`]); the divisor is 1 where the std is not
    /// positive, as the CPU port's `standardize`. The result is a new buffer.
    pub fn standardize_to_buffer(&self, field: &wgpu::Buffer, n: usize, m: &Moments) -> wgpu::Buffer {
        let out = self.output("standardized", n, 1);
        self.run("standardize_main", &Self::standardize_params(m), &[(1, Arg::Items(field, 1)), (5, Arg::Items(&out, 1))], n);
        out
    }

    /// [`StageKernels::standardize_to_buffer`] in place.
    pub fn standardize_in_place(&self, field: &wgpu::Buffer, n: usize, m: &Moments) {
        self.run("standardize_in_place_main", &Self::standardize_params(m), &[(5, Arg::Items(field, 1))], n);
    }

    /// The whole of `_standardize(v)`: the statistics of the field over its `n` values, then the new buffer.
    #[maybe_async::maybe_async]
    pub async fn standardize_field(&self, field: &wgpu::Buffer, n: usize) -> (wgpu::Buffer, Moments) {
        let m = self.moments(field, n).await;
        (self.standardize_to_buffer(field, n, &m), m)
    }

    /// `_standardize(v)` of host values, read back.
    #[maybe_async::maybe_async]
    pub async fn standardize_to_vec(&self, values: &[f32]) -> Vec<f32> {
        let f = self.upload_field(values);
        let (out, _) = self.standardize_field(&f, values.len()).await;
        self.field_to_vec(&out, values.len())
    }
}

// ---------------------------------------------------------------------------------------------------------
// Free functions: one set of kernels per call (each compiles its shader: use StageKernels for repeated calls)

/// `grid_dirs(width, height)` on the GPU, read back.
pub fn grid_dirs(gpu: &GpuContext, width: usize, height: usize) -> Vec<[f32; 3]> {
    StageKernels::new(gpu).grid_dirs_to_vec(width, height, 0, height)
}

/// `upsample_dirs(coarse, src_h, src_w, width, height)` on the GPU, read back.
pub fn upsample_dirs(gpu: &GpuContext, coarse: &[[f32; 3]], src_w: usize, src_h: usize, width: usize, height: usize) -> Vec<[f32; 3]> {
    StageKernels::new(gpu).upsample_dirs_to_vec(coarse, src_w, src_h, width, height)
}

/// `along(p, back, f)` on the GPU, read back.
pub fn along(gpu: &GpuContext, p: &[[f32; 3]], back: &[[f32; 3]], f: f64) -> Vec<[f32; 3]> {
    StageKernels::new(gpu).along_to_vec(p, back, f)
}

/// `moisture_profile(lat, hadley_deg)` on the GPU, read back.
pub fn moisture_profile(gpu: &GpuContext, lat: &[f32], hadley_deg: f64) -> Vec<f32> {
    StageKernels::new(gpu).moisture_profile_to_vec(lat, hadley_deg)
}

/// `_standardize(v)` on the GPU, read back.
#[maybe_async::maybe_async]
pub async fn standardize(gpu: &GpuContext, values: &[f32]) -> Vec<f32> {
    StageKernels::new(gpu).standardize_to_vec(values).await
}

/// `grid_dirs` of the band of `rows` rows from map row `row0`, left on the GPU (3 f32 per texel).
pub fn grid_dirs_to_buffer(gpu: &GpuContext, width: usize, height: usize, row0: usize, rows: usize) -> wgpu::Buffer {
    StageKernels::new(gpu).grid_dirs_to_buffer(width, height, row0, rows)
}

/// `upsample_dirs` of a coarse grid already on the GPU (`src_w x src_h` directions, 3 f32 each) for the band of
/// `rows` rows from map row `row0` of the `width x height` target, left on the GPU.
#[allow(clippy::too_many_arguments)]
pub fn upsample_dirs_to_buffer(
    gpu: &GpuContext,
    coarse: &wgpu::Buffer,
    src_w: usize,
    src_h: usize,
    width: usize,
    height: usize,
    row0: usize,
    rows: usize,
) -> wgpu::Buffer {
    StageKernels::new(gpu).upsample_dirs_to_buffer(coarse, src_w, src_h, width, height, row0, rows)
}

/// `along(p, back, f)` of two direction buffers (`n` directions each), left on the GPU.
pub fn along_to_buffer(gpu: &GpuContext, p: &wgpu::Buffer, back: &wgpu::Buffer, n: usize, f: f64) -> wgpu::Buffer {
    StageKernels::new(gpu).along_to_buffer(p, back, n, f)
}

/// The latitude of `n` directions already on the GPU, left on the GPU (1 f32 per texel).
pub fn lat_of_dirs_to_buffer(gpu: &GpuContext, dirs: &wgpu::Buffer, n: usize) -> wgpu::Buffer {
    StageKernels::new(gpu).lat_of_dirs_to_buffer(dirs, n)
}

/// `moisture_profile` of `n` latitudes already on the GPU, left on the GPU.
pub fn moisture_profile_to_buffer(gpu: &GpuContext, lat: &wgpu::Buffer, n: usize, hadley_deg: f64) -> wgpu::Buffer {
    StageKernels::new(gpu).moisture_profile_to_buffer(lat, n, hadley_deg)
}

/// The synoptic field before standardization (see [`StageKernels::synoptic_to_buffer`]), left on the GPU.
pub fn synoptic_to_buffer(
    gpu: &GpuContext,
    lat_back: &wgpu::Buffer,
    moisture_noise: &wgpu::Buffer,
    ascent: &wgpu::Buffer,
    mid_noise: &wgpu::Buffer,
    n: usize,
    p: &SynopticParams,
) -> wgpu::Buffer {
    StageKernels::new(gpu).synoptic_to_buffer(lat_back, moisture_noise, ascent, mid_noise, n, p)
}

/// The mean and the sum of squared deviations of the first `n` f32 of a field on the GPU.
#[maybe_async::maybe_async]
pub async fn moments(gpu: &GpuContext, field: &wgpu::Buffer, n: usize) -> Moments {
    StageKernels::new(gpu).moments(field, n).await
}

/// `(x - mean) / std` of a field on the GPU with the statistics `m`, as a new buffer.
pub fn standardize_to_buffer(gpu: &GpuContext, field: &wgpu::Buffer, n: usize, m: &Moments) -> wgpu::Buffer {
    StageKernels::new(gpu).standardize_to_buffer(field, n, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cloud_map_generator_core::generate::{along as cpu_along, moisture_profile as cpu_moisture};
    use cloud_map_generator_core::sphere::{grid_dirs as cpu_grid_dirs, upsample_dirs as cpu_upsample};

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

    fn max_diff(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs())).fold(0.0, f32::max)
    }

    #[test]
    fn rows_per_chunk_is_whole_rows_under_the_limit() {
        // 4096 x 2048: 12 B per texel is 49152 B per row, 128 MiB holds 2730 rows (the whole map of 2048 rows)
        assert_eq!(rows_per_binding(PORTABLE_BINDING_BYTES, 4096, 3), 2730);
        assert_eq!(rows_per_binding(PORTABLE_BINDING_BYTES, 8192, 3), 1365); // 8192 x 4096 needs 3 bands
        assert_eq!(rows_per_binding(PORTABLE_BINDING_BYTES, 8192, 1), 4096); // a scalar field of the 8192 map fits whole
        assert_eq!(rows_per_binding(10, 8192, 3), 1, "never fewer than one row");
    }

    #[test]
    fn moments_merge_is_the_union_of_the_sets() {
        let v: Vec<f32> = (0..1000).map(|i| ((i * 37 % 101) as f32 - 40.0) * 0.25).collect();
        let whole = Moments::of_slice(&v);
        let merged = v.chunks(137).map(Moments::of_slice).fold(Moments::EMPTY, Moments::merge);
        assert_eq!(merged.count, whole.count);
        assert!((merged.mean - whole.mean).abs() < 1e-12 && (merged.m2 / whole.m2 - 1.0).abs() < 1e-12);
        assert_eq!(Moments::EMPTY.merge(whole), whole);
        assert_eq!(whole.merge(Moments::EMPTY), whole);
        assert_eq!(Moments { count: 3, mean: 1.0, m2: 0.0 }.divisor(), 1.0, "a constant field divides by 1");
        assert!(Moments { count: 3, mean: 1.0, m2: f64::NAN }.divisor() == 1.0, "a NaN std is not positive");
    }

    #[test]
    fn split_pairs_hold_48_bits() {
        let x = 0.123_456_789_012_345_67_f64;
        let (hi, lo) = split(x);
        assert!((f64::from(hi) + f64::from(lo) - x).abs() < 1e-15);
    }

    #[test]
    fn the_params_layout_is_the_96_bytes_of_the_shader() {
        assert_eq!(std::mem::size_of::<Params>(), 96);
        assert_eq!(std::mem::offset_of!(Params, u), 16);
        assert_eq!(std::mem::offset_of!(Params, k), 32);
    }

    #[test]
    fn grid_dirs_match_the_cpu_port_and_bands_do_not_matter() {
        let Some(gpu) = test_gpu() else { return };
        let k = StageKernels::new(&gpu);
        for (w, h) in [(64usize, 32usize), (37, 19), (1, 1), (2, 1), (7, 5)] {
            let want = cpu_grid_dirs::<f32>(w, h);
            let got = k.grid_dirs_to_vec(w, h, 0, h);
            assert!(max_diff(&got, &want) <= 1.5e-7, "{w}x{h}: {}", max_diff(&got, &want));
            // a band in the middle
            if h >= 4 {
                let band = k.grid_dirs_to_vec(w, h, 1, h - 2);
                assert_eq!(band, got[w..w * (h - 1)].to_vec(), "{w}x{h}: a band is the same rows");
            }
        }
    }

    #[test]
    fn chunking_forced_small_gives_the_same_bits() {
        let Some(gpu) = test_gpu() else { return };
        let (w, h) = (211usize, 97usize); // 20467 texels: many chunks of 1024
        let whole = StageKernels::new(&gpu);
        let small = StageKernels::with_binding_bytes(&gpu, 16 * 1024);
        assert_eq!(small.rows_per_chunk(w, 3), 6, "211 x 12 B = 2532 B per row: 6 rows fit 16 KiB");
        let (a, b) = (whole.grid_dirs_to_vec(w, h, 0, h), small.grid_dirs_to_vec(w, h, 0, h));
        assert_eq!(a, b);
        let coarse: Vec<[f32; 3]> = cpu_grid_dirs::<f32>(16, 8);
        assert_eq!(whole.upsample_dirs_to_vec(&coarse, 16, 8, w, h), small.upsample_dirs_to_vec(&coarse, 16, 8, w, h));
        let back: Vec<[f32; 3]> = a.iter().map(|d| [d[1], -d[0], d[2]]).collect();
        assert_eq!(whole.along_to_vec(&a, &back, 0.25), small.along_to_vec(&a, &back, 0.25));
        let lat: Vec<f32> = (0..w * h).map(|i| (i % 181) as f32 * 0.0174 - 1.57).collect();
        assert_eq!(whole.moisture_profile_to_vec(&lat, 30.0), small.moisture_profile_to_vec(&lat, 30.0));
        let (ma, mb) = (whole.moments_of_slice(&lat), small.moments_of_slice(&lat));
        assert_eq!(ma, mb, "the tiles are the same whatever the chunking (chunks are whole tiles)");
        assert_eq!(whole.standardize_to_vec(&lat), small.standardize_to_vec(&lat));
    }

    #[test]
    fn upsample_matches_the_cpu_port_on_odd_sizes_and_bands() {
        let Some(gpu) = test_gpu() else { return };
        let k = StageKernels::new(&gpu);
        // a perturbed coarse grid, as the back-trace makes
        let coarse_w = [(8usize, 4usize, 32usize, 16usize), (5, 3, 11, 7), (1, 1, 4, 2), (16, 8, 64, 4), (8, 4, 8, 4), (16, 512, 8, 2048)];
        for (sw, sh, w, h) in coarse_w {
            let base = cpu_grid_dirs::<f64>(sw, sh);
            let coarse: Vec<[f64; 3]> = base
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    let e = 0.05 * ((i * 7 % 13) as f64 / 13.0 - 0.5);
                    let v = [d[0] + e, d[1] - 0.5 * e, d[2] + 0.3 * e];
                    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    [v[0] / n, v[1] / n, v[2] / n]
                })
                .collect();
            let want = cpu_upsample(&coarse, sh, sw, w, h);
            let c32: Vec<[f32; 3]> = coarse.iter().map(|d| [d[0] as f32, d[1] as f32, d[2] as f32]).collect();
            let got = k.upsample_dirs_to_vec(&c32, sw, sh, w, h);
            assert!(max_diff(&got, &want) < 1e-6, "{sw}x{sh} -> {w}x{h}: {}", max_diff(&got, &want));
            // in two bands
            let cb = k.upload_dirs(&c32);
            if h >= 2 {
                let top = h / 2;
                let a = k.dirs_to_vec(&k.upsample_dirs_to_buffer(&cb, sw, sh, w, h, 0, top), w * top);
                let b = k.dirs_to_vec(&k.upsample_dirs_to_buffer(&cb, sw, sh, w, h, top, h - top), w * (h - top));
                assert_eq!([a, b].concat(), got, "{sw}x{sh} -> {w}x{h}: banding must not change a texel");
            }
        }
    }

    #[test]
    fn along_and_the_moisture_profile_match_the_cpu_port() {
        let Some(gpu) = test_gpu() else { return };
        let k = StageKernels::new(&gpu);
        let p = cpu_grid_dirs::<f32>(120, 60);
        let back: Vec<[f32; 3]> = p
            .iter()
            .map(|d| {
                let (c, s) = (0.1_f32.cos(), 0.1_f32.sin());
                [c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]]
            })
            .collect();
        for f in [0.0, 0.25, 0.6, 1.0, 1.5, -0.5] {
            assert!(max_diff(&k.along_to_vec(&p, &back, f), &cpu_along(&p, &back, f)) < 2e-7, "f = {f}");
        }
        let lat: Vec<f32> = (0..=180).map(|d| (d as f32 - 90.0).to_radians()).collect();
        for hadley in [10.0, 30.0, 47.5, 90.0] {
            let got = k.moisture_profile_to_vec(&lat, hadley);
            let want = cpu_moisture(&lat, hadley);
            let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(worst < 5e-7, "hadley {hadley}: {worst}");
        }
    }

    #[test]
    fn standardize_constants_and_the_reduction_edges() {
        let Some(gpu) = test_gpu() else { return };
        let k = StageKernels::new(&gpu);
        assert_eq!(k.standardize_to_vec(&[0.5; 64]), vec![0.0; 64]);
        assert_eq!(k.standardize_to_vec(&[0.0; 32]), vec![0.0; 32]);
        assert_eq!(k.standardize_to_vec(&[7.0]), vec![0.0]);
        // sizes around the tile: the tail tile counts only its own values
        for n in [1usize, 2, 255, 256, 257, 4095, 4096, 4097, 8192, 12_345] {
            let v: Vec<f32> = (0..n).map(|i| ((i * 2_654_435_761usize % 1000) as f32) / 100.0 - 3.0).collect();
            let (m, want) = (k.moments_of_slice(&v), Moments::of_slice(&v));
            assert_eq!(m.count, want.count);
            assert!((m.mean - want.mean).abs() < 1e-6, "n = {n}: mean {} vs {}", m.mean, want.mean);
            assert!(n == 1 || (m.std() / want.std() - 1.0).abs() < 1e-6, "n = {n}: std {} vs {}", m.std(), want.std());
        }
        // a field whose mean is far above its std keeps its std (the deviations are taken about the tile mean)
        let v: Vec<f32> = (0..10_000).map(|i| 1000.0 + (i % 7) as f32 * 0.01).collect();
        let (m, want) = (k.moments_of_slice(&v), Moments::of_slice(&v));
        assert!((m.std() / want.std() - 1.0).abs() < 1e-3, "{} vs {}", m.std(), want.std());
    }
}
