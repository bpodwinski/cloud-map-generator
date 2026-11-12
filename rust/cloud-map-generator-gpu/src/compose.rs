//! GPU kernels of `cloudgen/generate.py`, ported from the CPU port `cloud_map_generator_core::generate`: `compose` (the shield
//! with its texture and streak, the cumulus population with the fine texture, the sub-texel term), the per-row
//! threshold bisection `solve_thresholds` (36 steps; ONE WORKGROUP PER ROW evaluates the whole row at every step
//! and a tree reduction forms the row mean), `row_sensitivity`, the surface terms' per-row target and per-texel
//! offset, and a counter-based sub-texel noise. The WGSL is `shaders/compose.wgsl`.
//!
//! Everything runs in f32, so a result equals the CPU port's to float32 rounding, not bit for bit: the CPU
//! solves in f64 (pairwise row means), the GPU forms the f32 row sum with a tree reduction and compares it with
//! `target * width`. The thresholds therefore differ by the conditioning of the bisection times the f32 error
//! of a mean (about 1e-6 on a row with a slope of 0.3); `tests/golden_gpu_compose.rs` measures it on whole maps.
//!
//! # Bands
//!
//! A [`Band`] is a slice of the map's rows held in GPU buffers (`rows x width` texels, row-major). Every row is
//! independent (the bisection decides on one row's mean), so a map larger than a storage binding is processed
//! band by band: [`max_band_rows`] bounds a band by the WebGPU default limits (128 MiB per binding, 65535
//! workgroups per dimension) whatever the native adapter would allow, so the same WGSL runs in a browser. Only
//! the smoothing of the thresholds over latitude ([`smooth_thresholds`], on the CPU: one value per row) couples
//! the rows: calibrate every band, smooth the concatenated thresholds, then compose every band.
//!
//! # API
//!
//! Three levels, as in the flow kernels: the buffer methods of [`ComposeKernels`] take and fill GPU buffers and
//! wait for nothing (the next kernel chains on the result); [`ComposeKernels::calibrate`] and
//! [`ComposeKernels::coverage`] chain the whole calibration of one band; the `*_host` methods take slices, band
//! the map (an explicit `max_band_rows` forces small bands, to test the banding) and read the result back.
//!
//! # Counter-based sub-texel noise
//!
//! The CPU pipeline draws the unit white noise of the sub-texel term from its own generator in one go
//! ([`cloud_map_generator_core::generate::GeneratePlans`]); the GPU fills it with [`ComposeKernels::fill_subtexel`] from a
//! COUNTER-BASED generator, `normal(seed, texel_index)`, whose Rust twin [`counter_normal`] returns the very same
//! bits: a CPU engine that fills its sub-texel field with the twin produces the field of the GPU.

use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use cloud_map_generator_core::generate::GenParams;
use cloud_map_generator_core::matching::ndimage::{self, Mode};
use std::sync::OnceLock;

const SHADER: &str = include_str!("../shaders/compose.wgsl");

/// Threads along x of a texel workgroup.
const TEXEL_GROUP_X: usize = 64;
/// The bisection of `solve_thresholds`: 36 steps on [-10, 10] (literals of the Python source).
pub const BISECTION_STEPS: u32 = 36;
pub const BISECTION_LO: f32 = -10.0;
pub const BISECTION_HI: f32 = 10.0;
/// The floor of the row sensitivity (a literal of the Python source).
pub const SENSITIVITY_FLOOR: f32 = 0.05;
/// WebGPU's default `maxStorageBufferBindingSize`: a band never exceeds it, whatever the adapter offers.
pub const PORTABLE_BINDING_BYTES: u64 = 128 << 20;
/// WebGPU's default `maxComputeWorkgroupsPerDimension`: one workgroup per row means at most this many rows.
pub const MAX_BAND_ROWS: usize = 65535;

const FLAG_SUB: u32 = 1;
const FLAG_STREAK_ARRAY: u32 = 2;
const OFFSET_SHIFT: u32 = 2;

/// The `Params` uniform of `compose.wgsl` (96 bytes, the layout of the WGSL struct).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    rows: u32,
    row0: u32,
    flags: u32,
    ramp: f32,
    inv_ramp: f32,
    texture: f32,
    cumulus_max: f32,
    depth: f32,
    inv_depth: f32,
    a_fine: f32,
    streak_factor: f32,
    offset_scalar: f32,
    step: f32,
    sens_floor: f32,
    inv_width: f32,
    steps: u32,
    lo0: f32,
    hi0: f32,
    noise_scale: f32,
    seed_lo: u32,
    seed_hi: u32,
    pad0: u32,
    pad1: u32,
}

/// `streak` of `compose`: a Python float (the shield texture is then one constant), or one value per texel.
#[derive(Clone, Copy)]
pub enum StreakSrc<'a> {
    Scalar(f64),
    Buffer(&'a wgpu::Buffer),
}

/// The per-texel offset added to the row threshold (`solve_thresholds(..., offset)` and the final `t_px`).
#[derive(Clone, Copy)]
pub enum OffsetSrc<'a> {
    /// No offset (0).
    None,
    /// A Python float.
    Scalar(f64),
    /// One value per row of the band (a `(height, 1)` array).
    Column(&'a wgpu::Buffer),
    /// One value per texel (the surface terms' anomaly over the row sensitivity).
    Full(&'a wgpu::Buffer),
}

/// The fields of `compose` for one band of rows, in GPU buffers of `rows * width` f32 each (`STORAGE` usage).
#[derive(Clone, Copy)]
pub struct Band<'a> {
    pub rows: usize,
    pub width: usize,
    /// The synoptic field `s` (unit variance).
    pub s: &'a wgpu::Buffer,
    /// The fine texture.
    pub fine: &'a wgpu::Buffer,
    pub streak: StreakSrc<'a>,
    /// The sub-texel field (unit white noise over `sqrt(n_clouds)`); `None` switches the binomial term off.
    pub subtexel: Option<&'a wgpu::Buffer>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Entry {
    Solve,
    Sensitivity,
    Compose,
    RowTarget,
    SurfaceOffset,
    FillSubtexel,
}

impl Entry {
    const ALL: [Entry; 6] = [Entry::Solve, Entry::Sensitivity, Entry::Compose, Entry::RowTarget, Entry::SurfaceOffset, Entry::FillSubtexel];

    fn name(self) -> &'static str {
        match self {
            Entry::Solve => "solve_main",
            Entry::Sensitivity => "sensitivity_main",
            Entry::Compose => "compose_main",
            Entry::RowTarget => "row_target_main",
            Entry::SurfaceOffset => "surface_offset_main",
            Entry::FillSubtexel => "fill_subtexel_main",
        }
    }

    /// The bindings of group 0 (besides the params at 0) the entry point statically uses.
    fn bindings(self) -> &'static [u32] {
        match self {
            Entry::Solve | Entry::Compose => &[1, 2, 3, 4, 5, 6, 7],
            Entry::Sensitivity => &[1, 2, 3, 4, 6, 7],
            Entry::RowTarget => &[1, 2, 7],
            Entry::SurfaceOffset => &[1, 2, 3, 4, 7],
            Entry::FillSubtexel => &[7],
        }
    }
}

/// The kernels of this module, compiled on first use. Building a pipeline compiles the shader: keep one
/// `ComposeKernels` for repeated calls.
pub struct ComposeKernels<'a> {
    gpu: &'a GpuContext,
    module: wgpu::ShaderModule,
    pipelines: [OnceLock<wgpu::ComputePipeline>; 6],
    /// Bound to the slots of a kernel that its configuration does not read (no streak array, no sub-texel...).
    dummy: wgpu::Buffer,
    /// Four zero words: the opaque zero of the counter-based generator's barrier (group 3).
    zero: wgpu::Buffer,
}

/// Rows a band may hold for `width` columns: the WebGPU default binding size and workgroup count, and the
/// adapter's own limits when they are lower.
pub fn max_band_rows(gpu: &GpuContext, width: usize) -> usize {
    assert!(width > 0, "a map has at least one column");
    let bytes = u64::from(gpu.limits.max_storage_buffer_binding_size).min(gpu.limits.max_buffer_size).min(PORTABLE_BINDING_BYTES);
    let row_bytes = 4 * width as u64;
    assert!(row_bytes <= bytes, "one row of {width} texels ({row_bytes} bytes) exceeds the binding limit ({bytes} bytes)");
    ((bytes / row_bytes) as usize).clamp(1, MAX_BAND_ROWS)
}

/// The constant streak factor `1 - shield_texture * clip(0.5 + 0.5 streak, 0, 1)` of a scalar streak, in f64
/// as the CPU port forms it.
fn scalar_streak_factor(prm: &GenParams, streak: f64) -> f32 {
    let k = (0.5 + 0.5 * streak).clamp(0.0, 1.0);
    (1.0 - prm.shield_texture * k) as f32
}

fn offset_flags(offset: OffsetSrc<'_>) -> (u32, f32) {
    match offset {
        OffsetSrc::None => (0, 0.0),
        OffsetSrc::Scalar(v) => (0, v as f32),
        OffsetSrc::Column(_) => (1 << OFFSET_SHIFT, 0.0),
        OffsetSrc::Full(_) => (2 << OFFSET_SHIFT, 0.0),
    }
}

impl<'a> ComposeKernels<'a> {
    pub fn new(gpu: &'a GpuContext) -> ComposeKernels<'a> {
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloudgen compose"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });
        ComposeKernels {
            gpu,
            module,
            pipelines: std::array::from_fn(|_| OnceLock::new()),
            dummy: gpu.storage_with("compose unused slot", &[0.0f32; 4]),
            zero: gpu.uniform_with("compose zero", &[0u32; 4]),
        }
    }

    pub fn gpu(&self) -> &GpuContext {
        self.gpu
    }

    fn pipeline(&self, entry: Entry) -> &wgpu::ComputePipeline {
        let slot = Entry::ALL.iter().position(|&e| e == entry).expect("a known entry point");
        self.pipelines[slot].get_or_init(|| {
            self.gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry.name()),
                layout: None,
                module: &self.module,
                entry_point: Some(entry.name()),
                compilation_options: Default::default(),
                cache: None,
            })
        })
    }

    /// Dispatches `entry` with the params and the buffers of `slots` (binding -> buffer); a binding the entry
    /// uses and `slots` does not name gets the dummy buffer. Does not wait for the result.
    fn launch(&self, entry: Entry, params: &Params, slots: &[(u32, &wgpu::Buffer)], groups: (u32, u32, u32)) {
        let pipeline = self.pipeline(entry);
        let params_buffer = self.gpu.uniform_with("compose params", params);
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() }];
        for &binding in entry.bindings() {
            let buffer = slots.iter().find(|(b, _)| *b == binding).map_or(&self.dummy, |(_, buf)| *buf);
            entries.push(wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() });
        }
        let device = &self.gpu.device;
        let group0 = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("compose"), layout: &pipeline.get_bind_group_layout(0), entries: &entries });
        // The counter-based generator reads the zero of group 3; wgpu wants every group below it set too.
        let mut extra: Vec<(u32, wgpu::BindGroup)> = Vec::new();
        if entry == Entry::FillSubtexel {
            for g in [1u32, 2] {
                extra.push((g, device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(g), entries: &[] })));
            }
            let zero_entry = [wgpu::BindGroupEntry { binding: 0, resource: self.zero.as_entire_binding() }];
            extra.push((3, device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(3), entries: &zero_entry })));
        }
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group0, &[]);
            for (g, bg) in &extra {
                pass.set_bind_group(*g, bg, &[]);
            }
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        self.gpu.queue.submit([enc.finish()]);
    }

    fn check_band(&self, band: &Band<'_>) {
        assert!(band.rows >= 1 && band.rows <= MAX_BAND_ROWS, "a band holds 1..={MAX_BAND_ROWS} rows (one workgroup per row), got {}", band.rows);
        let bytes = 4 * (band.rows * band.width) as u64;
        assert!(bytes <= PORTABLE_BINDING_BYTES, "a band of {} x {} texels exceeds the portable binding size: split it (max_band_rows)", band.rows, band.width);
        assert!(band.s.size() >= bytes && band.fine.size() >= bytes, "s and fine must hold rows x width f32");
        if let StreakSrc::Buffer(b) = band.streak {
            assert!(b.size() >= bytes, "the streak buffer must hold rows x width f32");
        }
        if let Some(b) = band.subtexel {
            assert!(b.size() >= bytes, "the subtexel buffer must hold rows x width f32");
        }
    }

    fn check_offset(&self, band: &Band<'_>, offset: OffsetSrc<'_>) {
        match offset {
            OffsetSrc::Column(b) => assert!(b.size() >= 4 * band.rows as u64, "an offset column holds one f32 per row"),
            OffsetSrc::Full(b) => assert!(b.size() >= 4 * (band.rows * band.width) as u64, "a full offset holds rows x width f32"),
            OffsetSrc::None | OffsetSrc::Scalar(_) => {}
        }
    }

    /// The params of a band, with the GenParams constants rounded once to f32.
    fn params(&self, band: &Band<'_>, prm: &GenParams, offset: OffsetSrc<'_>) -> Params {
        let (offset_flags, offset_scalar) = offset_flags(offset);
        let (streak_flag, streak_factor) = match band.streak {
            StreakSrc::Scalar(v) => (0, scalar_streak_factor(prm, v)),
            StreakSrc::Buffer(_) => (FLAG_STREAK_ARRAY, 1.0),
        };
        Params {
            width: band.width as u32,
            rows: band.rows as u32,
            row0: 0,
            flags: offset_flags | streak_flag | if band.subtexel.is_some() { FLAG_SUB } else { 0 },
            ramp: prm.shield_ramp as f32,
            inv_ramp: (1.0 / prm.shield_ramp) as f32,
            texture: prm.shield_texture as f32,
            cumulus_max: prm.cumulus_max as f32,
            depth: prm.cumulus_depth as f32,
            inv_depth: (1.0 / prm.cumulus_depth) as f32,
            a_fine: prm.a_fine as f32,
            streak_factor,
            offset_scalar,
            step: prm.sensitivity_step as f32,
            sens_floor: SENSITIVITY_FLOOR,
            inv_width: (1.0 / band.width as f64) as f32,
            steps: BISECTION_STEPS,
            lo0: BISECTION_LO,
            hi0: BISECTION_HI,
            noise_scale: 1.0,
            seed_lo: 0,
            seed_hi: 0,
            pad0: 0,
            pad1: 0,
        }
    }

    /// The buffers of a band's fields as the slots 1 (s), 2 (fine), 3 (streak), 4 (subtexel) of the shader.
    fn field_slots<'b>(&'b self, band: &'b Band<'b>) -> Vec<(u32, &'b wgpu::Buffer)> {
        let mut slots: Vec<(u32, &wgpu::Buffer)> = vec![(1, band.s), (2, band.fine)];
        if let StreakSrc::Buffer(b) = band.streak {
            slots.push((3, b));
        }
        if let Some(b) = band.subtexel {
            slots.push((4, b));
        }
        slots
    }

    // -- the kernels, on buffers -------------------------------------------------------------------------------

    /// `solve_thresholds(s, fine, target, prm, streak, subtexel, offset)` of the CPU port for the rows of `band`:
    /// per row, the threshold `t` with `mean(compose(s, fine, t + offset)) == target`, by 36 bisection steps on
    /// [-10, 10] (the midpoint of the final bracket). `target` is `rows` f32. Returns `rows` f32.
    pub fn solve_thresholds(&self, band: &Band<'_>, prm: &GenParams, target: &wgpu::Buffer, offset: OffsetSrc<'_>) -> wgpu::Buffer {
        self.check_band(band);
        assert!(target.size() >= 4 * band.rows as u64, "target holds one f32 per row");
        self.check_offset(band, offset);
        let out = self.gpu.storage_empty::<f32>("thresholds", band.rows);
        let mut slots = self.field_slots(band);
        slots.push((6, target));
        slots.push((7, &out));
        match offset {
            OffsetSrc::Column(b) | OffsetSrc::Full(b) => slots.push((5, b)),
            _ => {}
        }
        self.launch(Entry::Solve, &self.params(band, prm, offset), &slots, (band.rows as u32, 1, 1));
        out
    }

    /// `row_sensitivity(s, fine, thr, prm, streak, subtexel)`: `-d mean(compose) / d t` per row at the thresholds
    /// `thr` (`rows` f32) by a central difference of half-width `sensitivity_step`, floored at 0.05. The offset of
    /// the surface terms plays no part in it (as on the CPU). Returns `rows` f32.
    pub fn row_sensitivity(&self, band: &Band<'_>, prm: &GenParams, thr: &wgpu::Buffer) -> wgpu::Buffer {
        self.check_band(band);
        assert!(thr.size() >= 4 * band.rows as u64, "thr holds one f32 per row");
        let out = self.gpu.storage_empty::<f32>("sensitivity", band.rows);
        let mut slots = self.field_slots(band);
        slots.push((6, thr));
        slots.push((7, &out));
        self.launch(Entry::Sensitivity, &self.params(band, prm, OffsetSrc::None), &slots, (band.rows as u32, 1, 1));
        out
    }

    /// `compose(s, fine, thr + offset, prm, streak, subtexel)`: the coverage of every texel of the band, `rows *
    /// width` f32. `thr` is one threshold per row (`rows` f32); a threshold per texel is `thr = 0` with
    /// `OffsetSrc::Full`.
    pub fn compose(&self, band: &Band<'_>, prm: &GenParams, thr: &wgpu::Buffer, offset: OffsetSrc<'_>) -> wgpu::Buffer {
        self.check_band(band);
        assert!(thr.size() >= 4 * band.rows as u64, "thr holds one f32 per row");
        self.check_offset(band, offset);
        let out = self.gpu.storage_empty::<f32>("coverage", band.rows * band.width);
        let mut slots = self.field_slots(band);
        slots.push((6, thr));
        slots.push((7, &out));
        match offset {
            OffsetSrc::Column(b) | OffsetSrc::Full(b) => slots.push((5, b)),
            _ => {}
        }
        let groups = (band.width.div_ceil(TEXEL_GROUP_X) as u32, band.rows as u32, 1);
        self.launch(Entry::Compose, &self.params(band, prm, offset), &slots, groups);
        out
    }

    /// The row targets of the surface terms: the mean of `clip(target_zonal + anomaly, 0.01, 0.99)` over each
    /// row. `anomaly` is `rows * width` f32, `target_zonal` `rows` f32; returns `rows` f32.
    pub fn row_targets(&self, anomaly: &wgpu::Buffer, target_zonal: &wgpu::Buffer, rows: usize, width: usize) -> wgpu::Buffer {
        assert!((1..=MAX_BAND_ROWS).contains(&rows), "a band holds 1..={MAX_BAND_ROWS} rows");
        let out = self.gpu.storage_empty::<f32>("row targets", rows);
        let params = Params { width: width as u32, rows: rows as u32, ..Zeroable::zeroed() };
        self.launch(Entry::RowTarget, &params, &[(1, anomaly), (2, target_zonal), (7, &out)], (rows as u32, 1, 1));
        out
    }

    /// The per-texel threshold offset of the surface terms: `-(clip(target_zonal + anomaly, 0.01, 0.99) - target)
    /// / sensitivity`. `target` and `sens` are `rows` f32; returns `rows * width` f32.
    pub fn surface_offsets(&self, anomaly: &wgpu::Buffer, target_zonal: &wgpu::Buffer, target: &wgpu::Buffer, sens: &wgpu::Buffer, rows: usize, width: usize) -> wgpu::Buffer {
        assert!((1..=MAX_BAND_ROWS).contains(&rows), "a band holds 1..={MAX_BAND_ROWS} rows");
        let out = self.gpu.storage_empty::<f32>("surface offsets", rows * width);
        let params = Params { width: width as u32, rows: rows as u32, ..Zeroable::zeroed() };
        let groups = (width.div_ceil(TEXEL_GROUP_X) as u32, rows as u32, 1);
        self.launch(Entry::SurfaceOffset, &params, &[(1, anomaly), (2, target_zonal), (3, target), (4, sens), (7, &out)], groups);
        out
    }

    /// The sub-texel field of a band from the counter-based generator: `counter_normal(seed, index) * scale` for
    /// the texels of rows `row0 .. row0 + rows` of a map `width` wide (the counter is the texel's index in the
    /// whole map, so bands of one map get the field of the whole map). `scale` is `float32(1 / sqrt(n_clouds))`,
    /// see [`subtexel_noise_scale`]. Returns `rows * width` f32.
    pub fn fill_subtexel(&self, rows: usize, width: usize, row0: usize, seed: u64, scale: f32) -> wgpu::Buffer {
        assert!((1..=MAX_BAND_ROWS).contains(&rows), "a band holds 1..={MAX_BAND_ROWS} rows");
        assert!(((row0 + rows) as u64) * (width as u64) <= u64::from(u32::MAX), "texel indices of the map must fit a u32");
        let out = self.gpu.storage_empty::<f32>("subtexel", rows * width);
        let params = Params { width: width as u32, rows: rows as u32, row0: row0 as u32, noise_scale: scale, seed_lo: seed as u32, seed_hi: (seed >> 32) as u32, ..Zeroable::zeroed() };
        let groups = (width.div_ceil(TEXEL_GROUP_X) as u32, rows as u32, 1);
        self.launch(Entry::FillSubtexel, &params, &[(7, &out)], groups);
        out
    }

    // -- the calibration of a band --------------------------------------------------------------------------------

    /// The calibration of the rows of `band` as `generate()` runs it: without the surface terms the row targets
    /// are `target_zonal` and one bisection gives the thresholds; with them (`anomaly`, `rows * width` f32) the
    /// target of a row is the mean of its local target, a first bisection gives `thr0`, the row sensitivity at
    /// `thr0` gives the per-texel offset, and a second bisection with that offset gives the thresholds.
    /// `target_zonal` holds one value per row of the band. Nothing waits: the result is buffers.
    pub fn calibrate(&self, band: &Band<'_>, prm: &GenParams, target_zonal: &[f64], anomaly: Option<&wgpu::Buffer>) -> Calibration {
        assert_eq!(target_zonal.len(), band.rows, "target_zonal holds one value per row of the band");
        let zonal: Vec<f32> = target_zonal.iter().map(|&v| v as f32).collect();
        let zonal_buffer = self.gpu.storage_with("target zonal", &zonal);
        match anomaly {
            None => {
                let thr_raw = self.solve_thresholds(band, prm, &zonal_buffer, OffsetSrc::None);
                Calibration { target: zonal_buffer, thr0: None, sensitivity: None, offset: None, thr_raw }
            }
            Some(anomaly) => {
                let target = self.row_targets(anomaly, &zonal_buffer, band.rows, band.width);
                let thr0 = self.solve_thresholds(band, prm, &target, OffsetSrc::None);
                let sensitivity = self.row_sensitivity(band, prm, &thr0);
                let offset = self.surface_offsets(anomaly, &zonal_buffer, &target, &sensitivity, band.rows, band.width);
                let thr_raw = self.solve_thresholds(band, prm, &target, OffsetSrc::Full(&offset));
                Calibration { target, thr0: Some(thr0), sensitivity: Some(sensitivity), offset: Some(offset), thr_raw }
            }
        }
    }

    /// The whole chain for a band that IS the whole map (`band.rows` = the map's height): [`calibrate`], the
    /// smoothing of the thresholds in latitude on the CPU ([`smooth_thresholds`]: one value per row), and the
    /// final compose. Waits for the thresholds (the smoothing needs them on the CPU); the coverage stays on the GPU.
    ///
    /// [`calibrate`]: ComposeKernels::calibrate
    pub fn coverage(&self, band: &Band<'_>, prm: &GenParams, target_zonal: &[f64], anomaly: Option<&wgpu::Buffer>) -> CoverageOut {
        let cal = self.calibrate(band, prm, target_zonal, anomaly);
        let thr_raw: Vec<f32> = self.gpu.read_back(&cal.thr_raw, band.rows);
        let thr = smooth_thresholds(&thr_raw, prm, band.rows);
        let thr_buffer = self.gpu.storage_with("thresholds smoothed", &thr);
        let offset = cal.offset.as_ref().map_or(OffsetSrc::None, OffsetSrc::Full);
        let coverage = self.compose(band, prm, &thr_buffer, offset);
        CoverageOut { coverage, thr, thr_raw, offset: cal.offset }
    }

    // -- the same on slices ----------------------------------------------------------------------------------------

    /// Bands of at most `forced` rows (when given) within the limits of the adapter and of WebGPU.
    fn bands(&self, height: usize, width: usize, forced: Option<usize>) -> Vec<(usize, usize)> {
        let rows = max_band_rows(self.gpu, width).min(forced.unwrap_or(usize::MAX)).max(1);
        (0..height).step_by(rows).map(|r0| (r0, rows.min(height - r0))).collect()
    }

    /// Uploads the rows `r0 .. r0 + rows` of the host fields and runs `f` on them as a [`Band`].
    fn with_band<T>(&self, f: &HostFields<'_>, r0: usize, rows: usize, run: impl FnOnce(&Band<'_>) -> T) -> T {
        let span = r0 * f.width..(r0 + rows) * f.width;
        let s = self.gpu.storage_with("s", &f.s[span.clone()]);
        let fine = self.gpu.storage_with("fine", &f.fine[span.clone()]);
        let streak_buffer = match f.streak {
            HostStreak::Array(a) => Some(self.gpu.storage_with("streak", &a[span.clone()])),
            HostStreak::Scalar(_) => None,
        };
        let sub_buffer = f.subtexel.map(|a| self.gpu.storage_with("subtexel", &a[span.clone()]));
        let band = Band {
            rows,
            width: f.width,
            s: &s,
            fine: &fine,
            streak: match (&streak_buffer, f.streak) {
                (Some(b), _) => StreakSrc::Buffer(b),
                (None, HostStreak::Scalar(v)) => StreakSrc::Scalar(v),
                (None, HostStreak::Array(_)) => unreachable!("an array streak was uploaded"),
            },
            subtexel: sub_buffer.as_ref(),
        };
        run(&band)
    }

    /// `solve_thresholds` on slices: the thresholds, one f32 per row. `target` holds `height` values;
    /// `max_band_rows` forces bands of at most that many rows (a test of the banding).
    pub fn solve_thresholds_host(&self, f: &HostFields<'_>, prm: &GenParams, target: &[f64], offset: HostOffset<'_>, max_band_rows: Option<usize>) -> Vec<f32> {
        f.check();
        assert_eq!(target.len(), f.height, "target holds one value per row");
        let mut out = Vec::with_capacity(f.height);
        for (r0, rows) in self.bands(f.height, f.width, max_band_rows) {
            let target_f32: Vec<f32> = target[r0..r0 + rows].iter().map(|&v| v as f32).collect();
            let target_buffer = self.gpu.storage_with("target", &target_f32);
            let thr = self.with_band(f, r0, rows, |band| {
                let offset_buffer = offset.upload(self.gpu, r0, rows, f.width);
                self.solve_thresholds(band, prm, &target_buffer, offset_buffer.as_src(offset))
            });
            out.extend(self.gpu.read_back::<f32>(&thr, rows));
        }
        out
    }

    /// `row_sensitivity` on slices: one f32 per row; `thr` holds `height` values.
    pub fn row_sensitivity_host(&self, f: &HostFields<'_>, prm: &GenParams, thr: &[f64], max_band_rows: Option<usize>) -> Vec<f32> {
        f.check();
        assert_eq!(thr.len(), f.height, "thr holds one value per row");
        let mut out = Vec::with_capacity(f.height);
        for (r0, rows) in self.bands(f.height, f.width, max_band_rows) {
            let thr_f32: Vec<f32> = thr[r0..r0 + rows].iter().map(|&v| v as f32).collect();
            let thr_buffer = self.gpu.storage_with("thr", &thr_f32);
            let sens = self.with_band(f, r0, rows, |band| self.row_sensitivity(band, prm, &thr_buffer));
            out.extend(self.gpu.read_back::<f32>(&sens, rows));
        }
        out
    }

    /// `compose` on slices: `height * width` f32; `thr` holds `height` values (the row thresholds).
    pub fn compose_host(&self, f: &HostFields<'_>, prm: &GenParams, thr: &[f64], offset: HostOffset<'_>, max_band_rows: Option<usize>) -> Vec<f32> {
        f.check();
        assert_eq!(thr.len(), f.height, "thr holds one value per row");
        let mut out = Vec::with_capacity(f.height * f.width);
        for (r0, rows) in self.bands(f.height, f.width, max_band_rows) {
            let thr_f32: Vec<f32> = thr[r0..r0 + rows].iter().map(|&v| v as f32).collect();
            let thr_buffer = self.gpu.storage_with("thr", &thr_f32);
            let coverage = self.with_band(f, r0, rows, |band| {
                let offset_buffer = offset.upload(self.gpu, r0, rows, f.width);
                self.compose(band, prm, &thr_buffer, offset_buffer.as_src(offset))
            });
            out.extend(self.gpu.read_back::<f32>(&coverage, rows * f.width));
        }
        out
    }

    /// The counter-based sub-texel field of a whole map on the host: the GPU fills it band by band.
    pub fn subtexel_host(&self, height: usize, width: usize, seed: u64, scale: f32, max_band_rows: Option<usize>) -> Vec<f32> {
        let mut out = Vec::with_capacity(height * width);
        for (r0, rows) in self.bands(height, width, max_band_rows) {
            let buffer = self.fill_subtexel(rows, width, r0, seed, scale);
            out.extend(self.gpu.read_back::<f32>(&buffer, rows * width));
        }
        out
    }
}

/// What [`ComposeKernels::calibrate`] leaves on the GPU for one band.
pub struct Calibration {
    /// The per-row target of the bisection (`rows` f32): `target_zonal`, or the row mean of the local target.
    pub target: wgpu::Buffer,
    /// With the surface terms: the first bisection, without the offset (`rows` f32).
    pub thr0: Option<wgpu::Buffer>,
    /// With the surface terms: the row sensitivity at `thr0` (`rows` f32).
    pub sensitivity: Option<wgpu::Buffer>,
    /// With the surface terms: the per-texel threshold offset (`rows * width` f32).
    pub offset: Option<wgpu::Buffer>,
    /// The bisected row thresholds before the smoothing (`rows` f32).
    pub thr_raw: wgpu::Buffer,
}

/// What [`ComposeKernels::coverage`] returns.
pub struct CoverageOut {
    /// The coverage of the map (`rows * width` f32), on the GPU.
    pub coverage: wgpu::Buffer,
    /// The smoothed row thresholds.
    pub thr: Vec<f32>,
    /// The thresholds before the smoothing.
    pub thr_raw: Vec<f32>,
    /// The per-texel offset (with the surface terms), on the GPU.
    pub offset: Option<wgpu::Buffer>,
}

/// The smoothing of the row thresholds over latitude: `gaussian_filter1d(thr, threshold_smooth_deg / (180 /
/// height), mode='nearest')`, on the CPU (a vector of `height` values: microseconds). `height` is the HEIGHT OF
/// THE WHOLE MAP, not of a band.
pub fn smooth_thresholds(thr_raw: &[f32], prm: &GenParams, height: usize) -> Vec<f32> {
    assert_eq!(thr_raw.len(), height, "one threshold per row of the whole map");
    let sigma = prm.threshold_smooth_deg / (180.0 / height as f64);
    let wide: Vec<f64> = thr_raw.iter().map(|&v| f64::from(v)).collect();
    ndimage::gaussian_filter1d(&wide, height, 1, 0, sigma, Mode::Nearest, 4.0).into_iter().map(|v| v as f32).collect()
}

/// `float32(1 / sqrt(n_clouds))`: the factor [`ComposeKernels::fill_subtexel`] applies to the unit normal, from
/// the divisor `float32(sqrt(n_clouds))` of `cloud_map_generator_core::generate::subtexel_scale`.
pub fn subtexel_noise_scale(divisor: f32) -> f32 {
    1.0 / divisor
}

// ---------------------------------------------------------------------------------------------------------
// Host-side fields

/// `streak` of a host call: a Python float, or one value per texel.
#[derive(Clone, Copy)]
pub enum HostStreak<'a> {
    Scalar(f64),
    Array(&'a [f32]),
}

/// The offset of a host call.
#[derive(Clone, Copy)]
pub enum HostOffset<'a> {
    None,
    Scalar(f64),
    /// `height` values.
    Column(&'a [f32]),
    /// `height * width` values.
    Full(&'a [f32]),
}

/// The upload of the band's rows of a [`HostOffset`], kept alive while the kernel runs.
struct UploadedOffset {
    buffer: Option<wgpu::Buffer>,
}

impl UploadedOffset {
    fn as_src<'a>(&'a self, offset: HostOffset<'_>) -> OffsetSrc<'a> {
        match (offset, &self.buffer) {
            (HostOffset::None, _) => OffsetSrc::None,
            (HostOffset::Scalar(v), _) => OffsetSrc::Scalar(v),
            (HostOffset::Column(_), Some(b)) => OffsetSrc::Column(b),
            (HostOffset::Full(_), Some(b)) => OffsetSrc::Full(b),
            _ => unreachable!("an array offset was uploaded"),
        }
    }
}

impl HostOffset<'_> {
    fn upload(&self, gpu: &GpuContext, r0: usize, rows: usize, width: usize) -> UploadedOffset {
        let buffer = match self {
            HostOffset::None | HostOffset::Scalar(_) => None,
            HostOffset::Column(c) => Some(gpu.storage_with("offset column", &c[r0..r0 + rows])),
            HostOffset::Full(f) => Some(gpu.storage_with("offset", &f[r0 * width..(r0 + rows) * width])),
        };
        UploadedOffset { buffer }
    }
}

/// The fields of `compose` on the host: `height x width` f32 slices, row-major.
#[derive(Clone, Copy)]
pub struct HostFields<'a> {
    pub s: &'a [f32],
    pub fine: &'a [f32],
    pub streak: HostStreak<'a>,
    pub subtexel: Option<&'a [f32]>,
    pub height: usize,
    pub width: usize,
}

impl HostFields<'_> {
    fn check(&self) {
        let n = self.height * self.width;
        assert!(n > 0, "an empty map");
        assert_eq!(self.s.len(), n, "s must hold height x width values");
        assert_eq!(self.fine.len(), n, "fine must hold height x width values");
        if let HostStreak::Array(a) = self.streak {
            assert_eq!(a.len(), n, "streak must hold height x width values");
        }
        if let Some(a) = self.subtexel {
            assert_eq!(a.len(), n, "subtexel must hold height x width values");
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// The counter-based normal: the Rust twin of `counter_normal` in compose.wgsl

/// `(pi / 2) / 2^22` as an f32: the angle step of the folded 22-bit position inside a quadrant.
pub const CN_PSI_STEP: f32 = 3.7450704e-7;
/// `2^-24`, exact.
#[allow(clippy::excessive_precision)]
const CN_TWO_M24: f32 = 5.9604644775390625e-8;

/// The PCG hash the generator is built on (the shader's `pcg32`).
pub fn pcg32(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// `ln(u)` for `u` in `(0, 1]`: the Cephes `logf` polynomial, additions and multiplications in the shader's order.
// The constants are the published Cephes decimals, kept textually identical to the shader's: the two literals must
// parse to the same f32.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
fn cn_ln(u: f32) -> f32 {
    let bits = u.to_bits();
    let mut e = ((bits >> 23) & 255) as i32 - 126;
    let mut x = f32::from_bits((bits & 0x007F_FFFF) | 0x3F00_0000);
    if x < 0.70710678 {
        e -= 1;
        x = (x + x) - 1.0;
    } else {
        x -= 1.0;
    }
    let z = x * x;
    let mut y: f32 = 7.0376836292e-2;
    y = y * x + -1.1514610310e-1;
    y = y * x + 1.1676998740e-1;
    y = y * x + -1.2420140846e-1;
    y = y * x + 1.4249322787e-1;
    y = y * x + -1.6668057665e-1;
    y = y * x + 2.0000714765e-1;
    y = y * x + -2.4999993993e-1;
    y = y * x + 3.3333331174e-1;
    y = (y * x) * z;
    let ef = e as f32;
    if e != 0 {
        y += ef * -2.12194440e-4;
    }
    y -= 0.5 * z;
    let mut r = x + y;
    if e != 0 {
        r += ef * 0.693359375;
    }
    r
}

/// `sqrt(w)` for `w >= 0`: the inverse square root from the integer estimate, three Newton steps, times `w`.
fn cn_sqrt(w: f32) -> f32 {
    if w <= 0.0 {
        return 0.0;
    }
    let xh = 0.5 * w;
    let mut y = f32::from_bits(0x5F37_59DF - (w.to_bits() >> 1));
    for _ in 0..3 {
        let y2 = y * y;
        let t = xh * y2;
        y *= 1.5 - t;
    }
    w * y
}

/// `(sin psi, cos psi)` for `psi` in `[0, pi/4]`: the Cephes `sinf` and `cosf` polynomials.
#[allow(clippy::excessive_precision)]
fn cn_sincos(psi: f32) -> (f32, f32) {
    let z = psi * psi;
    let mut ys: f32 = -1.9515295891e-4;
    ys = ys * z + 8.3321608736e-3;
    ys = ys * z + -1.6666654611e-1;
    ys = ((ys * z) * psi) + psi;
    let mut yc: f32 = 2.443315711809948e-5;
    yc = yc * z + -1.388731625493765e-3;
    yc = yc * z + 4.166664568298827e-2;
    yc = (yc * z) * z;
    yc -= 0.5 * z;
    yc += 1.0;
    (ys, yc)
}

/// The counter-based standard normal of texel `index` for `seed`: a pure function of its arguments, the same
/// bits as `counter_normal` in `compose.wgsl` (a PCG hash of the index and the seed, then Box-Muller with
/// polynomial `ln`, `sqrt` and `cos`; about 1e-6 from the exact transform, mean 0 and standard deviation 1).
///
/// `index` is the texel's index in the whole map, `row * width + col`.
pub fn counter_normal(seed: u64, index: u32) -> f32 {
    let (seed_lo, seed_hi) = (seed as u32, (seed >> 32) as u32);
    let k0 = pcg32(seed_lo ^ 0x9E37_79B9);
    let k1 = pcg32(seed_hi ^ k0 ^ 0x85EB_CA6B);
    let x = pcg32(index);
    let a = pcg32(x ^ k0);
    let b = pcg32(x.wrapping_add(0x9E37_79B9) ^ k1);
    let u1 = ((a >> 8) as f32 + 1.0) * CN_TWO_M24;
    let radius = cn_sqrt(-2.0 * cn_ln(u1));
    let quadrant = b >> 30;
    let r22 = (b >> 8) & 0x003F_FFFF;
    let upper = r22 >= 0x0020_0000;
    let folded = if upper { 0x0040_0000 - r22 } else { r22 };
    let (sin_psi, cos_psi) = cn_sincos(folded as f32 * CN_PSI_STEP);
    let (cos_phi, sin_phi) = if upper { (sin_psi, cos_psi) } else { (cos_psi, sin_psi) };
    let c = match quadrant {
        0 => cos_phi,
        1 => -sin_phi,
        2 => -cos_phi,
        _ => sin_phi,
    };
    radius * c
}

/// The sub-texel field of a map of `height x width` texels from the counter-based generator on the CPU: the
/// field [`ComposeKernels::subtexel_host`] makes on the GPU, bit for bit. `scale` is [`subtexel_noise_scale`].
pub fn counter_subtexel_field(seed: u64, height: usize, width: usize, scale: f32) -> Vec<f32> {
    assert!(((height as u64) * (width as u64)) <= u64::from(u32::MAX), "texel indices of the map must fit a u32");
    (0..height * width).map(|i| counter_normal(seed, i as u32) * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_gpu() -> Option<GpuContext> {
        let gpu = GpuContext::new();
        match &gpu {
            Some(g) => eprintln!("adapter: {} ({:?})", g.adapter_info.name, g.adapter_info.backend),
            None if std::env::var_os("REQUIRE_GPU").is_some() => panic!("no GPU adapter and REQUIRE_GPU is set"),
            None => eprintln!("no GPU adapter: test skipped (set REQUIRE_GPU to fail instead)"),
        }
        gpu
    }

    /// The decimal literal of `const NAME: f32 = ...;` in the shader.
    fn shader_const(name: &str) -> f32 {
        let line = SHADER.lines().find(|l| l.starts_with(&format!("const {name}:"))).unwrap_or_else(|| panic!("no const {name} in compose.wgsl"));
        let value = line.split('=').nth(1).expect("a value").trim().trim_end_matches(';');
        value.parse::<f32>().unwrap_or_else(|e| panic!("const {name} = {value}: {e}"))
    }

    #[test]
    fn the_shader_constants_are_the_ones_of_the_twin() {
        assert_eq!(shader_const("CN_PSI_STEP"), CN_PSI_STEP);
        assert_eq!(CN_PSI_STEP, (std::f64::consts::FRAC_PI_2 / 4_194_304.0) as f32);
        assert_eq!(shader_const("CN_TWO_M24"), CN_TWO_M24);
        assert_eq!(CN_TWO_M24, 2f32.powi(-24));
    }

    #[test]
    fn the_params_uniform_has_the_size_of_the_wgsl_struct() {
        assert_eq!(std::mem::size_of::<Params>(), 96);
        assert_eq!(std::mem::size_of::<Params>() % 16, 0);
    }

    #[test]
    fn the_twin_matches_the_exact_box_muller_transform() {
        // The same two words through f64 ln, sqrt and cos: the polynomials are within 1e-5 of it everywhere.
        let mut worst = 0.0f64;
        for seed in [0u64, 7, 0xDEAD_BEEF_1234_5678] {
            for i in 0..200_000u32 {
                let (seed_lo, seed_hi) = (seed as u32, (seed >> 32) as u32);
                let k0 = pcg32(seed_lo ^ 0x9E37_79B9);
                let k1 = pcg32(seed_hi ^ k0 ^ 0x85EB_CA6B);
                let x = pcg32(i);
                let a = pcg32(x ^ k0);
                let b = pcg32(x.wrapping_add(0x9E37_79B9) ^ k1);
                let u1 = f64::from((a >> 8) + 1) / 16_777_216.0;
                let quadrant = f64::from(b >> 30);
                let frac = f64::from((b >> 8) & 0x003F_FFFF) / 4_194_304.0;
                let theta = std::f64::consts::FRAC_PI_2 * (quadrant + frac);
                let exact = (-2.0 * u1.ln()).sqrt() * theta.cos();
                let got = f64::from(counter_normal(seed, i));
                worst = worst.max((got - exact).abs());
            }
        }
        eprintln!("counter_normal against the f64 Box-Muller transform of the same words: worst |diff| {worst:.2e}");
        assert!(worst < 2e-6, "{worst}");
    }

    #[test]
    fn the_normals_have_mean_0_std_1_and_the_gaussian_tails() {
        // 1M draws per seed. The sampling error of the mean is 1e-3 and of the std 7e-4: the 3e-3 bound is 3 sigma
        // of the mean, the seeds below were not chosen (0, 1, 2, 12345 and a 64-bit one).
        let n = 1_000_000u32;
        for seed in [0u64, 1, 2, 12345, 0xDEAD_BEEF_1234_5678] {
            let z: Vec<f64> = (0..n).map(|i| f64::from(counter_normal(seed, i))).collect();
            let mean = z.iter().sum::<f64>() / f64::from(n);
            let var = z.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / f64::from(n);
            let frac = |k: f64| z.iter().filter(|v| v.abs() > k).count() as f64 / f64::from(n);
            let (f1, f2, f3) = (frac(1.0), frac(2.0), frac(3.0));
            eprintln!("seed {seed:#x}: mean {mean:+.2e}, std {:.5}, P(|z|>1) {f1:.4} (0.3173), P(|z|>2) {f2:.4} (0.0455), P(|z|>3) {f3:.5} (0.0027), max |z| {:.2}", var.sqrt(), z.iter().fold(0.0f64, |m, v| m.max(v.abs())));
            assert!(mean.abs() < 3e-3, "seed {seed}: mean {mean}");
            assert!((var.sqrt() - 1.0).abs() < 3e-3, "seed {seed}: std {}", var.sqrt());
            assert!((f1 - 0.3173).abs() < 3e-3 && (f2 - 0.0455).abs() < 1.5e-3 && (f3 - 0.0027).abs() < 6e-4, "seed {seed}: tails {f1} {f2} {f3}");
        }
    }

    #[test]
    fn neighbors_and_seeds_are_uncorrelated() {
        let n = 1_000_000u32;
        let z0: Vec<f64> = (0..n).map(|i| f64::from(counter_normal(5, i))).collect();
        let z1: Vec<f64> = (0..n).map(|i| f64::from(counter_normal(6, i))).collect();
        let corr = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / a.len() as f64;
        let (lag1, lag2, lagw) = (corr(&z0[1..], &z0[..z0.len() - 1]), corr(&z0[2..], &z0[..z0.len() - 2]), corr(&z0[1024..], &z0[..z0.len() - 1024]));
        let across = corr(&z0, &z1);
        eprintln!("correlation: lag 1 {lag1:+.2e}, lag 2 {lag2:+.2e}, lag 1024 {lagw:+.2e}, seed 5 vs 6 {across:+.2e} (sampling error 1e-3)");
        for c in [lag1, lag2, lagw, across] {
            assert!(c.abs() < 4e-3, "{c}");
        }
    }

    #[test]
    fn the_gpu_agrees_with_the_twin_bit_for_bit_on_a_million_indices() {
        let Some(gpu) = test_gpu() else { return };
        let kernels = ComposeKernels::new(&gpu);
        // 1M texels as a 2048 x 512 map, with a seed that uses both 32-bit halves
        let (height, width, seed) = (512usize, 2048usize, 0x0123_4567_89AB_CDEFu64);
        let scale = subtexel_noise_scale(3.0);
        let field = kernels.subtexel_host(height, width, seed, scale, None);
        let want = counter_subtexel_field(seed, height, width, scale);
        let bad: Vec<usize> = (0..field.len()).filter(|&i| field[i].to_bits() != want[i].to_bits()).collect();
        assert!(bad.is_empty(), "{} of {} differ from the Rust twin, first at {}: gpu {:e} ({:#x}), twin {:e} ({:#x})", bad.len(), field.len(), bad[0], field[bad[0]], field[bad[0]].to_bits(), want[bad[0]], want[bad[0]].to_bits());
        // banded: the counter is the index in the whole map, so small bands give the same field
        let banded = kernels.subtexel_host(height, width, seed, scale, Some(37));
        assert!(banded.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "banded fill differs");
    }

    #[test]
    fn the_gpu_twin_agreement_holds_for_other_seeds_and_a_second_band_offset() {
        let Some(gpu) = test_gpu() else { return };
        let kernels = ComposeKernels::new(&gpu);
        for seed in [0u64, 1, 0xFFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF] {
            let (height, width) = (64usize, 4096usize);
            let field = kernels.subtexel_host(height, width, seed, 1.0, None);
            let want = counter_subtexel_field(seed, height, width, 1.0);
            assert!(field.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "seed {seed:#x}");
        }
    }

    #[test]
    fn band_rows_respect_the_portable_limits() {
        let Some(gpu) = test_gpu() else { return };
        for width in [1usize, 64, 4096, 16384, 65536] {
            let rows = max_band_rows(&gpu, width);
            assert!((1..=MAX_BAND_ROWS).contains(&rows));
            assert!((rows * width * 4) as u64 <= PORTABLE_BINDING_BYTES, "{rows} x {width}");
        }
        assert_eq!(max_band_rows(&gpu, 4096), 8192);
    }

    #[test]
    fn an_undersized_offset_is_refused_before_the_kernel_runs() {
        let Some(gpu) = test_gpu() else { return };
        let kernels = ComposeKernels::new(&gpu);
        let (s, fine, small) = (gpu.storage_with("s", &[0.0f32; 8]), gpu.storage_with("fine", &[0.0f32; 8]), gpu.storage_with("offset", &[0.0f32; 4]));
        let band = Band { rows: 2, width: 4, s: &s, fine: &fine, streak: StreakSrc::Scalar(0.0), subtexel: None };
        let thr = gpu.storage_with("thr", &[0.0f32; 2]);
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| kernels.compose(&band, &GenParams::default(), &thr, OffsetSrc::Full(&small))));
        let message = refused.err().map(|p| p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|m| m.to_string())).unwrap_or_default()).unwrap_or_default();
        assert!(message.contains("a full offset holds rows x width f32"), "{message:?}");
    }

    #[test]
    fn smoothing_keeps_a_constant_and_flattens_a_spike() {
        let prm = GenParams::default();
        let flat = smooth_thresholds(&[0.25; 90], &prm, 90);
        assert!(flat.iter().all(|&v| (v - 0.25).abs() < 1e-6));
        let mut spike = vec![0.0f32; 90];
        spike[45] = 1.0;
        let smooth = smooth_thresholds(&spike, &prm, 90);
        assert!(smooth[45] < 1.0 && smooth[45] > 0.3 && (smooth.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }
}
