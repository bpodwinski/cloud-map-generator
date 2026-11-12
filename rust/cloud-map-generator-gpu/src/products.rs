//! GPU kernels of `cloudgen/products.py` (the cloud TYPE map and the HIGH-CLOUD SHARE map) and of the shield of
//! `generate()`, ported from the CPU port `cloud_map_generator_core::products`. The WGSL is `shaders/products.wgsl`. The wind
//! map is coarse (a back-trace grid) and stays on the CPU.
//!
//! The per-texel arithmetic runs on the GPU in f32 (the CPU port holds the shield and the latitude terms in f64:
//! the difference is a few 1e-8, far inside the golden tolerances). What depends on the latitude alone (the
//! convective tendency `c` of the row, the zonal base of the high share, the deep-core term `0.15 c^4`) is a
//! function of the row: it is evaluated once per row in f64 on the host, as the CPU port does, and uploaded as
//! one or two values per row.
//!
//! Like the compose kernels these work on a band of rows held in GPU buffers (see [`crate::compose`]); the
//! `*_host` methods band a map held in slices within the WebGPU default limits.

use crate::compose::{max_band_rows, HostOffset, MAX_BAND_ROWS};
use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use cloud_map_generator_core::climate::convective_tendency;
use cloud_map_generator_core::generate::GenParams;
use cloud_map_generator_core::products::{HIGH_BASE_SHARE, HIGH_BASE_X, HIGH_DEEP_CORE};
use std::sync::OnceLock;

const SHADER: &str = include_str!("../shaders/products.wgsl");
const TEXEL_GROUP_X: usize = 64;

/// The `Params` uniform of `products.wgsl` (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    rows: u32,
    flags: u32,
    pad0: u32,
    ramp: f32,
    inv_ramp: f32,
    offset_scalar: f32,
    pad1: f32,
}

/// The offset of the shield kernel: the same enum as the compose kernels' (it is the offset of `t_px`).
pub use crate::compose::OffsetSrc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Entry {
    Shield,
    TypeMap,
    HighShare,
}

impl Entry {
    const ALL: [Entry; 3] = [Entry::Shield, Entry::TypeMap, Entry::HighShare];

    fn name(self) -> &'static str {
        match self {
            Entry::Shield => "shield_main",
            Entry::TypeMap => "type_map_main",
            Entry::HighShare => "high_share_main",
        }
    }

    /// The bindings of group 0 (besides the params at 0) the entry point statically uses.
    fn bindings(self) -> &'static [u32] {
        match self {
            Entry::Shield => &[1, 2, 3, 7],
            Entry::TypeMap => &[1, 2, 3, 4, 5, 7],
            Entry::HighShare => &[1, 2, 3, 5, 7],
        }
    }
}

/// The kernels of this module, compiled on first use.
pub struct ProductKernels<'a> {
    gpu: &'a GpuContext,
    module: wgpu::ShaderModule,
    pipelines: [OnceLock<wgpu::ComputePipeline>; 3],
    dummy: wgpu::Buffer,
}

/// `np.interp(x, xp, fp)` for increasing `xp`: the end values outside, numpy's slope formula inside.
fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x.is_nan() {
        return x;
    }
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// `convective_tendency(degrees(lat), hadley_deg)` of every row, f64 as the CPU port evaluates it, rounded to f32.
pub fn type_rows(lat_rows: &[f64], hadley_deg: f64) -> Vec<f32> {
    let degrees = 180.0 / std::f64::consts::PI;
    lat_rows.iter().map(|&lat| convective_tendency(lat * degrees, hadley_deg) as f32).collect()
}

/// `(base, deep)` of every row for the high share: the MODIS zonal profile at `x = |lat| / phi_H` and the deep
/// cores `0.15 c^4`, f64 as the CPU port evaluates them, rounded to f32, interleaved.
pub fn high_share_rows(lat_rows: &[f64], hadley_deg: f64) -> Vec<f32> {
    let degrees = 180.0 / std::f64::consts::PI;
    let mut out = Vec::with_capacity(2 * lat_rows.len());
    for &lat in lat_rows {
        let lat_deg = lat * degrees;
        let c = convective_tendency(lat_deg, hadley_deg);
        let x = lat_deg.abs() / hadley_deg;
        out.push(np_interp(x, &HIGH_BASE_X, &HIGH_BASE_SHARE) as f32);
        out.push((HIGH_DEEP_CORE * c.powf(4.0)) as f32);
    }
    out
}

impl<'a> ProductKernels<'a> {
    pub fn new(gpu: &'a GpuContext) -> ProductKernels<'a> {
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloudgen products"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });
        ProductKernels { gpu, module, pipelines: std::array::from_fn(|_| OnceLock::new()), dummy: gpu.storage_with("products unused slot", &[0.0f32; 4]) }
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

    /// Dispatches `entry` over `rows x width` texels with the buffers of `slots` (binding -> buffer); a binding the
    /// entry uses and `slots` does not name gets the dummy buffer. Does not wait for the result.
    fn launch(&self, entry: Entry, params: &Params, slots: &[(u32, &wgpu::Buffer)]) {
        assert!((1..=MAX_BAND_ROWS).contains(&(params.rows as usize)), "a band holds 1..={MAX_BAND_ROWS} rows");
        let pipeline = self.pipeline(entry);
        let params_buffer = self.gpu.uniform_with("products params", params);
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() }];
        for &binding in entry.bindings() {
            let buffer = slots.iter().find(|(b, _)| *b == binding).map_or(&self.dummy, |(_, buf)| *buf);
            entries.push(wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() });
        }
        let group = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("products"), layout: &pipeline.get_bind_group_layout(0), entries: &entries });
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((params.width as usize).div_ceil(TEXEL_GROUP_X) as u32, params.rows, 1);
        }
        self.gpu.queue.submit([enc.finish()]);
    }

    fn params(rows: usize, width: usize) -> Params {
        Params { width: width as u32, rows: rows as u32, ..Zeroable::zeroed() }
    }

    fn check_texels(&self, what: &str, buffer: &wgpu::Buffer, rows: usize, width: usize) {
        assert!(buffer.size() >= 4 * (rows * width) as u64, "{what} must hold rows x width f32");
    }

    // -- the kernels, on buffers -------------------------------------------------------------------------------

    /// `shield = clip(0.5 + (s - t_px) / shield_ramp, 0, 1)` of `generate()` with `t_px = thr + offset`: `s` is
    /// `rows * width` f32, `thr` one f32 per row. Returns `rows * width` f32. The shield is what `type_map` and
    /// `high_share` take (it carries no streak texture: that belongs to the coverage alone).
    pub fn shield(&self, s: &wgpu::Buffer, thr: &wgpu::Buffer, offset: OffsetSrc<'_>, rows: usize, width: usize, prm: &GenParams) -> wgpu::Buffer {
        self.check_texels("s", s, rows, width);
        assert!(thr.size() >= 4 * rows as u64, "thr holds one f32 per row");
        let out = self.gpu.storage_empty::<f32>("shield", rows * width);
        let (mode, scalar) = match offset {
            OffsetSrc::None => (0, 0.0),
            OffsetSrc::Scalar(v) => (0, v as f32),
            OffsetSrc::Column(_) => (1, 0.0),
            OffsetSrc::Full(_) => (2, 0.0),
        };
        let params = Params { flags: mode, ramp: prm.shield_ramp as f32, inv_ramp: (1.0 / prm.shield_ramp) as f32, offset_scalar: scalar, ..Self::params(rows, width) };
        let mut slots: Vec<(u32, &wgpu::Buffer)> = vec![(1, s), (2, thr), (7, &out)];
        if let OffsetSrc::Column(b) | OffsetSrc::Full(b) = offset {
            slots.push((3, b));
        }
        self.launch(Entry::Shield, &params, &slots);
        out
    }

    /// `products.type_map(shield, weights, lat, hadley_deg)`: `rows * width` f32. `shield`, `closed`, `open` and
    /// `street` are `rows * width` f32; `lat_rows` is the CONVECTION latitude of each row in radians (the grid
    /// latitude less the ITCZ shift, as `generate()` passes it).
    #[allow(clippy::too_many_arguments)]
    pub fn type_map(&self, shield: &wgpu::Buffer, closed: &wgpu::Buffer, open: &wgpu::Buffer, street: &wgpu::Buffer, lat_rows: &[f64], width: usize, hadley_deg: f64) -> wgpu::Buffer {
        let rows = lat_rows.len();
        for (what, b) in [("shield", shield), ("closed", closed), ("open", open), ("street", street)] {
            self.check_texels(what, b, rows, width);
        }
        let row_c = self.gpu.storage_with("type rows", &type_rows(lat_rows, hadley_deg));
        let out = self.gpu.storage_empty::<f32>("type", rows * width);
        self.launch(Entry::TypeMap, &Self::params(rows, width), &[(1, shield), (2, closed), (3, open), (4, street), (5, &row_c), (7, &out)]);
        out
    }

    /// `products.high_share(shield, ascent, lat, hadley_deg, coverage)`: `rows * width` f32. `shield`, `ascent` and
    /// `coverage` are `rows * width` f32; `lat_rows` as in [`ProductKernels::type_map`].
    #[allow(clippy::too_many_arguments)]
    pub fn high_share(&self, shield: &wgpu::Buffer, ascent: &wgpu::Buffer, coverage: &wgpu::Buffer, lat_rows: &[f64], width: usize, hadley_deg: f64) -> wgpu::Buffer {
        let rows = lat_rows.len();
        for (what, b) in [("shield", shield), ("ascent", ascent), ("coverage", coverage)] {
            self.check_texels(what, b, rows, width);
        }
        let row_hi = self.gpu.storage_with("high share rows", &high_share_rows(lat_rows, hadley_deg));
        let out = self.gpu.storage_empty::<f32>("high share", rows * width);
        self.launch(Entry::HighShare, &Self::params(rows, width), &[(1, shield), (2, ascent), (3, coverage), (5, &row_hi), (7, &out)]);
        out
    }

    /// The products of `generate()` for a band, in one call: the shield from `s` and the thresholds, then `type`
    /// and `high_share`. Returns `(type, high_share)`, `rows * width` f32 each; the shield buffer is dropped when
    /// the kernels that read it have run.
    #[allow(clippy::too_many_arguments)]
    pub fn products(&self, inputs: &ProductInputs<'_>, prm: &GenParams) -> (wgpu::Buffer, wgpu::Buffer) {
        let (rows, width) = (inputs.lat_rows.len(), inputs.width);
        let shield = self.shield(inputs.s, inputs.thr, inputs.offset, rows, width, prm);
        let kind = self.type_map(&shield, inputs.closed, inputs.open, inputs.street, inputs.lat_rows, width, inputs.hadley_deg);
        let high = self.high_share(&shield, inputs.ascent, inputs.coverage, inputs.lat_rows, width, inputs.hadley_deg);
        (kind, high)
    }

    // -- the same on slices ----------------------------------------------------------------------------------------

    fn bands(&self, height: usize, width: usize, forced: Option<usize>) -> Vec<(usize, usize)> {
        let rows = max_band_rows(self.gpu, width).min(forced.unwrap_or(usize::MAX)).max(1);
        (0..height).step_by(rows).map(|r0| (r0, rows.min(height - r0))).collect()
    }

    /// `shield` on slices: `height * width` f32. `s` is `height x width`, `thr` one value per row.
    pub fn shield_host(&self, s: &[f32], thr: &[f64], offset: HostOffset<'_>, width: usize, prm: &GenParams, max_band_rows: Option<usize>) -> Vec<f32> {
        let height = thr.len();
        assert_eq!(s.len(), height * width, "s must hold height x width values");
        let mut out = Vec::with_capacity(s.len());
        for (r0, rows) in self.bands(height, width, max_band_rows) {
            let span = r0 * width..(r0 + rows) * width;
            let s_buffer = self.gpu.storage_with("s", &s[span.clone()]);
            let thr_f32: Vec<f32> = thr[r0..r0 + rows].iter().map(|&v| v as f32).collect();
            let thr_buffer = self.gpu.storage_with("thr", &thr_f32);
            let offset_buffer = match offset {
                HostOffset::None | HostOffset::Scalar(_) => None,
                HostOffset::Column(c) => Some(self.gpu.storage_with("offset column", &c[r0..r0 + rows])),
                HostOffset::Full(f) => Some(self.gpu.storage_with("offset", &f[span.clone()])),
            };
            let src = match (offset, &offset_buffer) {
                (HostOffset::Scalar(v), _) => OffsetSrc::Scalar(v),
                (HostOffset::Column(_), Some(b)) => OffsetSrc::Column(b),
                (HostOffset::Full(_), Some(b)) => OffsetSrc::Full(b),
                _ => OffsetSrc::None,
            };
            let shield = self.shield(&s_buffer, &thr_buffer, src, rows, width, prm);
            out.extend(self.gpu.read_back::<f32>(&shield, rows * width));
        }
        out
    }

    /// `type_map` on slices (the shield and the weights `height x width`, `lat_rows` one value per row).
    #[allow(clippy::too_many_arguments)]
    pub fn type_map_host(&self, shield: &[f32], closed: &[f32], open: &[f32], street: &[f32], lat_rows: &[f64], width: usize, hadley_deg: f64, max_band_rows: Option<usize>) -> Vec<f32> {
        let height = lat_rows.len();
        for a in [shield, closed, open, street] {
            assert_eq!(a.len(), height * width, "every field must hold height x width values");
        }
        let mut out = Vec::with_capacity(shield.len());
        for (r0, rows) in self.bands(height, width, max_band_rows) {
            let span = r0 * width..(r0 + rows) * width;
            let up = |name: &str, a: &[f32]| self.gpu.storage_with(name, &a[span.clone()]);
            let (b_shield, b_closed, b_open, b_street) = (up("shield", shield), up("closed", closed), up("open", open), up("street", street));
            let kind = self.type_map(&b_shield, &b_closed, &b_open, &b_street, &lat_rows[r0..r0 + rows], width, hadley_deg);
            out.extend(self.gpu.read_back::<f32>(&kind, rows * width));
        }
        out
    }

    /// `high_share` on slices.
    #[allow(clippy::too_many_arguments)]
    pub fn high_share_host(&self, shield: &[f32], ascent: &[f32], coverage: &[f32], lat_rows: &[f64], width: usize, hadley_deg: f64, max_band_rows: Option<usize>) -> Vec<f32> {
        let height = lat_rows.len();
        for a in [shield, ascent, coverage] {
            assert_eq!(a.len(), height * width, "every field must hold height x width values");
        }
        let mut out = Vec::with_capacity(shield.len());
        for (r0, rows) in self.bands(height, width, max_band_rows) {
            let span = r0 * width..(r0 + rows) * width;
            let up = |name: &str, a: &[f32]| self.gpu.storage_with(name, &a[span.clone()]);
            let (b_shield, b_ascent, b_coverage) = (up("shield", shield), up("ascent", ascent), up("coverage", coverage));
            let share = self.high_share(&b_shield, &b_ascent, &b_coverage, &lat_rows[r0..r0 + rows], width, hadley_deg);
            out.extend(self.gpu.read_back::<f32>(&share, rows * width));
        }
        out
    }
}

/// The buffers [`ProductKernels::products`] reads, for one band of `lat_rows.len()` rows of `width` texels.
#[derive(Clone, Copy)]
pub struct ProductInputs<'a> {
    pub width: usize,
    /// The CONVECTION latitude of each row, radians.
    pub lat_rows: &'a [f64],
    pub hadley_deg: f64,
    /// The synoptic field and the row thresholds (the smoothed ones) with the per-texel offset: the shield.
    pub s: &'a wgpu::Buffer,
    pub thr: &'a wgpu::Buffer,
    pub offset: OffsetSrc<'a>,
    /// The regime weights, `rows * width` f32 each.
    pub closed: &'a wgpu::Buffer,
    pub open: &'a wgpu::Buffer,
    pub street: &'a wgpu::Buffer,
    /// The vortices' local ascent and the coverage.
    pub ascent: &'a wgpu::Buffer,
    pub coverage: &'a wgpu::Buffer,
}

#[cfg(test)]
mod tests {
    use super::*;
    use cloud_map_generator_core::products::{high_share as cpu_high_share, type_map as cpu_type_map};

    fn test_gpu() -> Option<GpuContext> {
        let gpu = GpuContext::new();
        match &gpu {
            Some(g) => eprintln!("adapter: {} ({:?})", g.adapter_info.name, g.adapter_info.backend),
            None if std::env::var_os("REQUIRE_GPU").is_some() => panic!("no GPU adapter and REQUIRE_GPU is set"),
            None => eprintln!("no GPU adapter: test skipped (set REQUIRE_GPU to fail instead)"),
        }
        gpu
    }

    #[test]
    fn the_params_uniform_has_the_size_of_the_wgsl_struct() {
        assert_eq!(std::mem::size_of::<Params>(), 32);
    }

    #[test]
    fn np_interp_clamps_and_interpolates_like_numpy() {
        let (xp, fp) = ([0.0, 1.0, 3.0], [10.0, 20.0, 0.0]);
        assert_eq!(np_interp(-5.0, &xp, &fp), 10.0);
        assert_eq!(np_interp(9.0, &xp, &fp), 0.0);
        assert_eq!(np_interp(1.0, &xp, &fp), 20.0);
        assert!((np_interp(0.5, &xp, &fp) - 15.0).abs() < 1e-12);
        assert!((np_interp(2.0, &xp, &fp) - 10.0).abs() < 1e-12);
        // the table of the module: its end values and a knot
        assert_eq!(np_interp(0.0, &HIGH_BASE_X, &HIGH_BASE_SHARE), HIGH_BASE_SHARE[0]);
        assert_eq!(np_interp(99.0, &HIGH_BASE_X, &HIGH_BASE_SHARE), HIGH_BASE_SHARE[7]);
        assert_eq!(np_interp(1.0, &HIGH_BASE_X, &HIGH_BASE_SHARE), HIGH_BASE_SHARE[3]);
    }

    #[test]
    fn the_row_tables_are_the_cpu_ports_f64_values() {
        let lat = [-1.2f64, -0.3, 0.0, 0.5, std::f64::consts::FRAC_PI_2];
        let c = type_rows(&lat, 30.0);
        let hi = high_share_rows(&lat, 30.0);
        assert_eq!(c.len(), 5);
        assert_eq!(hi.len(), 10);
        assert_eq!(c[2], 1.0, "c is 1 on the equator");
        assert!((0.0..=1.0).contains(&hi[0]) && hi[1] >= 0.0);
        // base at the equator is the first table entry; the deep core there is 0.15 (c = 1)
        assert!((f64::from(hi[4]) - HIGH_BASE_SHARE[0]).abs() < 1e-7 && (f64::from(hi[5]) - 0.15).abs() < 1e-7);
    }

    /// `(shield, [closed, open, street], ascent, coverage, convection latitudes)`.
    type Fields = (Vec<f32>, [Vec<f32>; 3], Vec<f32>, Vec<f32>, Vec<f64>);
    /// The offset of a texel `(row, col)` in the shield test.
    type OffsetFn<'a> = Box<dyn Fn(usize, usize) -> f64 + 'a>;

    /// A synthetic band: shield, weights, ascent and coverage that exercise every branch (overcast, clear, both
    /// clips of the weights), on rows from pole to pole.
    fn fields(height: usize, width: usize) -> Fields {
        let n = height * width;
        let h = |i: usize, k: u32| ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(k.wrapping_mul(97_531)) >> 8) as f32 / 16_777_216.0;
        let shield: Vec<f32> = (0..n).map(|i| if i % 11 == 0 { 0.0 } else if i % 13 == 0 { 1.0 } else { h(i, 1) }).collect();
        let closed: Vec<f32> = (0..n).map(|i| 0.6 * h(i, 2)).collect();
        let open: Vec<f32> = (0..n).map(|i| 0.3 * h(i, 3)).collect();
        let street: Vec<f32> = (0..n).map(|i| 0.1 * h(i, 4)).collect();
        let ascent: Vec<f32> = (0..n).map(|i| 1.4 * h(i, 5) - 0.2).collect();
        let coverage: Vec<f32> = (0..n).map(|i| if i % 7 == 0 { 0.01 } else { h(i, 6) }).collect();
        let lat: Vec<f64> = (0..height).map(|r| std::f64::consts::FRAC_PI_2 - (r as f64 + 0.5) / height as f64 * std::f64::consts::PI - 0.07).collect();
        (shield, [closed, open, street], ascent, coverage, lat)
    }

    #[test]
    fn type_map_and_high_share_match_the_cpu_port_on_the_gpu() {
        let Some(gpu) = test_gpu() else { return };
        let kernels = ProductKernels::new(&gpu);
        let (height, width, hadley) = (90usize, 160usize, 30.0);
        let (shield, [closed, open, street], ascent, coverage, lat) = fields(height, width);
        let shield64: Vec<f64> = shield.iter().map(|&v| f64::from(v)).collect();
        let want_type = cpu_type_map::<f64, f32>(&shield64, &closed, &open, &street, &lat, width, hadley);
        let want_high = cpu_high_share::<f64, f32, f32>(&shield64, &ascent, &lat, width, hadley, &coverage);
        for forced in [None, Some(7)] {
            let got_type = kernels.type_map_host(&shield, &closed, &open, &street, &lat, width, hadley, forced);
            let got_high = kernels.high_share_host(&shield, &ascent, &coverage, &lat, width, hadley, forced);
            let max = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            let (e_type, e_high) = (max(&got_type, &want_type), max(&got_high, &want_high));
            eprintln!("bands {forced:?}: type max |diff| {e_type:.2e}, high share max |diff| {e_high:.2e}");
            assert!(e_type < 2e-6, "type {e_type}");
            assert!(e_high < 2e-6, "high share {e_high}");
        }
    }

    #[test]
    fn the_shield_is_the_clipped_ramp_with_any_offset_kind() {
        let Some(gpu) = test_gpu() else { return };
        let kernels = ProductKernels::new(&gpu);
        let prm = GenParams::default();
        let (height, width) = (6usize, 40usize);
        let s: Vec<f32> = (0..height * width).map(|i| (i as f32 - 120.0) * 0.01).collect();
        let thr: Vec<f64> = (0..height).map(|r| 0.1 * r as f64 - 0.2).collect();
        let column: Vec<f32> = (0..height).map(|r| 0.05 * r as f32).collect();
        let full: Vec<f32> = (0..height * width).map(|i| ((i * 7919) % 100) as f32 * 0.004 - 0.2).collect();
        let cases: [(HostOffset<'_>, OffsetFn<'_>); 4] = [
            (HostOffset::None, Box::new(|_, _| 0.0)),
            (HostOffset::Scalar(0.3), Box::new(|_, _| 0.3)),
            (HostOffset::Column(&column), Box::new(|r, _| f64::from(0.05 * r as f32))),
            (HostOffset::Full(&full), Box::new(|r, c| f64::from(full[r * width + c]))),
        ];
        for (offset, off_of) in cases {
            let got = kernels.shield_host(&s, &thr, offset, width, &prm, Some(4));
            for r in 0..height {
                for c in 0..width {
                    let t = thr[r] + off_of(r, c);
                    let want = (0.5 + (f64::from(s[r * width + c]) - t) / prm.shield_ramp).clamp(0.0, 1.0);
                    assert!((f64::from(got[r * width + c]) - want).abs() < 3e-6, "({r}, {c}): {} vs {want}", got[r * width + c]);
                }
            }
        }
    }
}
