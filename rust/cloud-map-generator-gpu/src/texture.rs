//! GPU kernels of `cloudgen/texture.py`, ported from the CPU port `cloud_map_generator_core::texture`: the regime weights of
//! every texel ([`TextureKernels::regime_weights_to_buffers`]) and the regime texture ([`TextureKernels::regime_texture_to_buffer`]),
//! the small-scale cloud pattern chosen by the local weather regime. The WGSL is `shaders/texture.wgsl`.
//!
//! # What runs where
//!
//! On the GPU, per texel: the weights (latitude and back-traced latitude to the closed-cell, open-cell and street
//! shares), the mixes of the direction fields (`stages::StageKernels::mix_dirs_to_buffer`), the generic and street
//! fBm and the cell edges of every used diameter (the kernels of [`crate::noise`], chained on GPU buffers with no
//! readback), the weighted cell ramps, the mean and standard deviation of the four fields over the whole map
//! (`stages::StageKernels::moments`) and the standardized blend. On the CPU: the plans (random draws), which
//! diameters the resolution of the map keeps ([`used_cells`]), the final f64 combination of the reduction (a few
//! thousand partial sums).
//!
//! # Bands
//!
//! A buffer holds one band of whole map rows (see [`crate::stages`]). The texture needs statistics over the WHOLE
//! array, so it is done in two steps when the map is driven in several bands: [`TextureKernels::texture_fields`]
//! makes the four raw fields of a band, [`TextureKernels::texture_moments`] their moments; merge the moments of
//! all the bands ([`TextureMoments::merge`]) and [`TextureKernels::texture_blend`] gives the band's texture.
//! [`TextureKernels::regime_texture_to_buffer`] does the three for one band that is the whole array.
//!
//! # Precision
//!
//! Everything is f32. The CPU port runs `regime_weights` in f64 for the latitude chain of the generator
//! (`regime_weights::<f64, f32, f64>`) and the blend in the precision of the weights; here the weights chain is
//! the f32 one of the golden cases of float32 latitudes (it differs from the f64 chain by about 1e-6 of weight),
//! and the directions are mixed and normalized in f32 where the CPU port does it in f64 for float64 directions
//! (then rounds to f32 as the fBm does). A multiply and an add may be fused by the driver.

use crate::noise::NoiseKernels;
use crate::stages::{split, Arg, Moments, Params, StageKernels};
use crate::GpuContext;
use cloud_map_generator_core::texture::{used_cells, CellFields, RegimeTexturePlans, RegimeTextureTerms};
use std::cell::RefCell;
use std::collections::HashMap;

const SHADER: &str = include_str!("../shaders/texture.wgsl");

/// Bindings of `texture.wgsl`: inputs 1..8, outputs 9..14.
const IN: u32 = 1;
const OUT: u32 = 9;

/// The regime weights of a band: one f32 per texel each, `closed + open + street <= 1` (the rest is fBm).
pub struct WeightBuffers {
    pub closed: wgpu::Buffer,
    pub open: wgpu::Buffer,
    pub street: wgpu::Buffer,
}

/// The intermediates of `regime_weights` the golden cases check: the convective tendency of the latitude, the
/// cold-air ramp and the raw total before the normalization.
pub struct WeightTerms {
    pub convective: wgpu::Buffer,
    pub cold: wgpu::Buffer,
    pub total_raw: wgpu::Buffer,
}

/// Where the latitude of the weights kernel comes from.
#[derive(Clone, Copy)]
pub enum LatInput<'a> {
    /// One latitude (radians, f32) per texel of the band.
    Texels(&'a wgpu::Buffer),
    /// One latitude per ROW of the band (the latitude of a grid row is the same for every column; see
    /// [`StageKernels::grid_lat_rows`]), the map being `width` texels wide: `width / rows` times less to read.
    Rows { lat: &'a wgpu::Buffer, width: usize },
}

/// The raw fields of the regime texture of a band, before their standardization over the whole array.
pub struct TextureFields {
    /// Texels of the band.
    pub n: usize,
    /// The generic fBm of `0.75 p + 0.25 back`.
    pub generic: wgpu::Buffer,
    /// The street fBm of `0.1 p + 0.9 back`.
    pub street: wgpu::Buffer,
    /// The weighted sums of the closed-cell and open-cell ramps over the used diameters (`None` when the map
    /// resolves no cell diameter).
    pub cells: Option<(wgpu::Buffer, wgpu::Buffer)>,
}

/// The statistics of the four fields over a whole array (or the part of it seen so far).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureMoments {
    pub generic: Moments,
    pub street: Moments,
    /// [`Moments::EMPTY`] when no cell diameter is used.
    pub closed: Moments,
    pub open: Moments,
}

impl TextureMoments {
    pub const EMPTY: TextureMoments = TextureMoments { generic: Moments::EMPTY, street: Moments::EMPTY, closed: Moments::EMPTY, open: Moments::EMPTY };

    /// The statistics of the union of two sets of bands.
    pub fn merge(self, other: TextureMoments) -> TextureMoments {
        TextureMoments {
            generic: self.generic.merge(other.generic),
            street: self.street.merge(other.street),
            closed: self.closed.merge(other.closed),
            open: self.open.merge(other.open),
        }
    }
}

/// The kernels of the texture, each pipeline built on first use. Owns the [`StageKernels`] and the
/// [`NoiseKernels`] it chains.
pub struct TextureKernels<'a> {
    gpu: &'a GpuContext,
    stages: StageKernels<'a>,
    noise: NoiseKernels<'a>,
    module: wgpu::ShaderModule,
    pipelines: RefCell<HashMap<&'static str, wgpu::ComputePipeline>>,
}

impl<'a> TextureKernels<'a> {
    pub fn new(gpu: &'a GpuContext) -> TextureKernels<'a> {
        TextureKernels::with_stages(gpu, StageKernels::new(gpu))
    }

    /// With explicit stage kernels (a test of the chunking builds them with a small binding limit).
    pub fn with_stages(gpu: &'a GpuContext, stages: StageKernels<'a>) -> TextureKernels<'a> {
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloudgen texture"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });
        TextureKernels { gpu, stages, noise: NoiseKernels::new(gpu), module, pipelines: RefCell::new(HashMap::new()) }
    }

    pub fn stages(&self) -> &StageKernels<'a> {
        &self.stages
    }

    pub fn noise(&self) -> &NoiseKernels<'a> {
        &self.noise
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

    fn field(&self, label: &str, n: usize) -> wgpu::Buffer {
        self.gpu.storage_empty::<f32>(label, n.max(1))
    }

    // -- regime_weights ----------------------------------------------------------------------------------------

    /// The scalars of the weights kernel as the CPU port rounds them: `np.degrees` of a float32
    /// (`f32(180) / f32(pi)`), `f32(hadley_deg)` and `f32(max(1e-3, hadley_deg))` (Python's `max`: a NaN edge
    /// takes the floor).
    fn weights_params(lat: LatInput<'_>, hadley_deg: f64) -> Params {
        let floor = if hadley_deg > 1e-3 { hadley_deg } else { 1e-3 };
        let mut p = Params::with_k(&[180.0_f32 / std::f64::consts::PI as f32, hadley_deg as f32, floor as f32]);
        match lat {
            LatInput::Texels(_) => p.width = 1,
            LatInput::Rows { width, .. } => {
                p.width = width as u32;
                p.u[0] = 1;
            }
        }
        p
    }

    fn lat_arg(lat: LatInput<'_>) -> Arg<'_> {
        match lat {
            LatInput::Texels(b) => Arg::Items(b, 1),
            LatInput::Rows { lat, .. } => Arg::Whole(lat),
        }
    }

    /// `regime_weights(lat, lat_back, hadley_deg)` of the CPU port for the `n` texels of a band, on the GPU:
    /// `closed`, `open` and `street` shares (sum <= 1, the rest is fBm). `lat_back` is the latitude of the
    /// back-traced direction ([`StageKernels::lat_of_dirs_to_buffer`]), one f32 per texel.
    pub fn regime_weights_to_buffers(&self, lat: LatInput<'_>, lat_back: &wgpu::Buffer, n: usize, hadley_deg: f64) -> WeightBuffers {
        let w = WeightBuffers { closed: self.field("weights closed", n), open: self.field("weights open", n), street: self.field("weights street", n) };
        let lat_arg = Self::lat_arg(lat);
        let args = [
            (IN, lat_arg),
            (IN + 1, Arg::Items(lat_back, 1)),
            (OUT, Arg::Items(&w.closed, 1)),
            (OUT + 1, Arg::Items(&w.open, 1)),
            (OUT + 2, Arg::Items(&w.street, 1)),
        ];
        self.stages.run_items(&self.pipeline("regime_weights_main"), &Self::weights_params(lat, hadley_deg), &args, n);
        w
    }

    /// [`TextureKernels::regime_weights_to_buffers`] with the intermediates of the golden cases.
    pub fn regime_weight_terms_to_buffers(&self, lat: LatInput<'_>, lat_back: &wgpu::Buffer, n: usize, hadley_deg: f64) -> (WeightBuffers, WeightTerms) {
        let w = WeightBuffers { closed: self.field("weights closed", n), open: self.field("weights open", n), street: self.field("weights street", n) };
        let t = WeightTerms { convective: self.field("convective", n), cold: self.field("cold", n), total_raw: self.field("total raw", n) };
        let lat_arg = Self::lat_arg(lat);
        let args = [
            (IN, lat_arg),
            (IN + 1, Arg::Items(lat_back, 1)),
            (OUT, Arg::Items(&w.closed, 1)),
            (OUT + 1, Arg::Items(&w.open, 1)),
            (OUT + 2, Arg::Items(&w.street, 1)),
            (OUT + 3, Arg::Items(&t.convective, 1)),
            (OUT + 4, Arg::Items(&t.cold, 1)),
            (OUT + 5, Arg::Items(&t.total_raw, 1)),
        ];
        self.stages.run_items(&self.pipeline("regime_weight_terms_main"), &Self::weights_params(lat, hadley_deg), &args, n);
        (w, t)
    }

    /// `regime_weights` of host latitudes (one per texel), read back: `(closed, open, street)`.
    pub fn regime_weights_to_vec(&self, lat: &[f32], lat_back: &[f32], hadley_deg: f64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(lat.len(), lat_back.len(), "lat and lat_back must have the same length");
        let n = lat.len();
        let (l, b) = (self.stages.upload_field(lat), self.stages.upload_field(lat_back));
        let w = self.regime_weights_to_buffers(LatInput::Texels(&l), &b, n, hadley_deg);
        (self.stages.field_to_vec(&w.closed, n), self.stages.field_to_vec(&w.open, n), self.stages.field_to_vec(&w.street, n))
    }

    /// [`TextureKernels::regime_weight_terms_to_buffers`] of host latitudes, read back: the weights and the
    /// intermediates, as `([closed, open, street], [convective, cold, total_raw])`.
    pub fn regime_weight_terms_to_vec(&self, lat: &[f32], lat_back: &[f32], hadley_deg: f64) -> ([Vec<f32>; 3], [Vec<f32>; 3]) {
        assert_eq!(lat.len(), lat_back.len(), "lat and lat_back must have the same length");
        let n = lat.len();
        let (l, b) = (self.stages.upload_field(lat), self.stages.upload_field(lat_back));
        let (w, t) = self.regime_weight_terms_to_buffers(LatInput::Texels(&l), &b, n, hadley_deg);
        let get = |buf: &wgpu::Buffer| self.stages.field_to_vec(buf, n);
        ([get(&w.closed), get(&w.open), get(&w.street)], [get(&t.convective), get(&t.cold), get(&t.total_raw)])
    }

    // -- regime_texture ----------------------------------------------------------------------------------------

    /// The raw fields of `regime_texture` for the `n` texels of a band (`p` and `back` are direction buffers,
    /// 3 f32 per texel: the texel directions and where the air there came from): the generic and street fBm of
    /// the mixed directions and, for each cell diameter the map resolves ([`used_cells`] of `radius_km` and
    /// `fmax`), the cell edges ramped into the weighted closed-cell and open-cell sums, accumulated in f32 in
    /// the order of the diameters. Nothing is standardized yet.
    ///
    /// # Panics
    /// If `plans` does not hold one cell plan per used diameter, drawn for that diameter's frequency.
    pub fn texture_fields(&self, p: &wgpu::Buffer, back: &wgpu::Buffer, n: usize, radius_km: f64, fmax: f64, plans: &RegimeTexturePlans) -> TextureFields {
        let used = used_cells(radius_km, fmax);
        assert_eq!(plans.cells.len(), used.len(), "one cell plan per used diameter ({} used)", used.len());
        let mix = |ka: f64, kb: f64| self.stages.mix_dirs_to_buffer(p, back, n, ka as f32, kb as f32);
        let generic = {
            let dirs = mix(0.75, 0.25);
            self.noise.fbm_to_buffer(&dirs, n, &plans.generic)
        };
        // Streets: the same noise read far back along the flow, so it is stretched along the wind.
        let street = {
            let dirs = mix(0.1, 0.9);
            self.noise.fbm_to_buffer(&dirs, n, &plans.street)
        };
        let cells = if used.is_empty() {
            None
        } else {
            // Cells drift little over the advection time; read them a short way back.
            let q = mix(0.85, 0.15);
            let (closed, open) = (self.field("closed cells", n), self.field("open cells", n));
            let ramps = self.pipeline("cell_ramps_main");
            for (i, (cell, plan)) in used.iter().zip(&plans.cells).enumerate() {
                assert_eq!(plan.frequency, cell.frequency, "the cell plan of {} km was drawn for another frequency", cell.diameter_km);
                let edge = self.noise.cell_edges_to_buffer(&q, n, plan);
                let mut params = Params::with_k(&[cell.weight as f32]);
                params.u[0] = u32::from(i == 0);
                let args = [(IN, Arg::Items(&edge, 1)), (OUT, Arg::Items(&closed, 1)), (OUT + 1, Arg::Items(&open, 1))];
                self.stages.run_items(&ramps, &params, &args, n);
            }
            Some((closed, open))
        };
        TextureFields { n, generic, street, cells }
    }

    /// The moments of the four fields of a band: one pass over each, one small readback each.
    #[maybe_async::maybe_async]
    pub async fn texture_moments(&self, fields: &TextureFields) -> TextureMoments {
        let n = fields.n;
        let (closed, open) = match &fields.cells {
            Some((c, o)) => (self.stages.moments(c, n).await, self.stages.moments(o, n).await),
            None => (Moments::EMPTY, Moments::EMPTY),
        };
        TextureMoments { generic: self.stages.moments(&fields.generic, n).await, street: self.stages.moments(&fields.street, n).await, closed, open }
    }

    /// The regime texture of a band: the four fields standardized with `moments` (the statistics over the WHOLE
    /// array: merge the moments of every band), then
    /// `tex = generic (1 - wc - wo) + wc closed + wo open` when a cell diameter is used (otherwise the closed
    /// and open weights are ignored) and `tex = tex (1 - ws) + ws street`. One f32 per texel.
    pub fn texture_blend(&self, fields: &TextureFields, weights: &WeightBuffers, moments: &TextureMoments) -> wgpu::Buffer {
        let n = fields.n;
        let out = self.field("regime texture", n);
        let (g, s, c, o) = (split(moments.generic.mean), split(moments.street.mean), split(moments.closed.mean), split(moments.open.mean));
        let mut params = Params::with_k(&[
            g.0,
            g.1,
            moments.generic.divisor() as f32,
            s.0,
            s.1,
            moments.street.divisor() as f32,
            c.0,
            c.1,
            moments.closed.divisor() as f32,
            o.0,
            o.1,
            moments.open.divisor() as f32,
        ]);
        params.u[0] = u32::from(fields.cells.is_some());
        let dummy = self.field("unused", 1);
        let (cs, os) = match &fields.cells {
            Some((c, o)) => (Arg::Items(c, 1), Arg::Items(o, 1)),
            None => (Arg::Whole(&dummy), Arg::Whole(&dummy)),
        };
        let args = [
            (IN, Arg::Items(&fields.generic, 1)),
            (IN + 1, Arg::Items(&fields.street, 1)),
            (IN + 2, cs),
            (IN + 3, os),
            (IN + 4, Arg::Items(&weights.closed, 1)),
            (IN + 5, Arg::Items(&weights.open, 1)),
            (IN + 6, Arg::Items(&weights.street, 1)),
            (OUT, Arg::Items(&out, 1)),
        ];
        self.stages.run_items(&self.pipeline("regime_blend_main"), &params, &args, n);
        out
    }

    /// `regime_texture(p, back, weights, radius_km, fmax, plans)` of the CPU port for one band that is the whole
    /// array: the unit-variance fine texture, the regime-weighted blend of cells, streets and fBm, one f32 per
    /// texel, left on the GPU.
    #[allow(clippy::too_many_arguments)]
    #[maybe_async::maybe_async]
    pub async fn regime_texture_to_buffer(&self, p: &wgpu::Buffer, back: &wgpu::Buffer, weights: &WeightBuffers, n: usize, radius_km: f64, fmax: f64, plans: &RegimeTexturePlans) -> wgpu::Buffer {
        let fields = self.texture_fields(p, back, n, radius_km, fmax, plans);
        let moments = self.texture_moments(&fields).await;
        self.texture_blend(&fields, weights, &moments)
    }

    /// [`TextureKernels::regime_texture_to_buffer`] of host directions and weights, with the intermediates the
    /// golden cases check ([`RegimeTextureTerms`]: the texture, the standardized generic and street fBm and the
    /// cell sums and their standardizations), read back.
    #[allow(clippy::too_many_arguments)]
    #[maybe_async::maybe_async]
    pub async fn regime_texture_terms_to_vec(
        &self,
        p: &[[f32; 3]],
        back: &[[f32; 3]],
        weights: [&[f32]; 3],
        radius_km: f64,
        fmax: f64,
        plans: &RegimeTexturePlans,
    ) -> RegimeTextureTerms {
        let n = p.len();
        assert_eq!(back.len(), n, "p and back must have the same length");
        assert!(weights.iter().all(|w| w.len() == n), "the weights must have one value per direction");
        let (pb, bb) = (self.stages.upload_dirs(p), self.stages.upload_dirs(back));
        let w = WeightBuffers { closed: self.stages.upload_field(weights[0]), open: self.stages.upload_field(weights[1]), street: self.stages.upload_field(weights[2]) };
        let fields = self.texture_fields(&pb, &bb, n, radius_km, fmax, plans);
        let m = self.texture_moments(&fields).await;
        let texture = self.stages.field_to_vec(&self.texture_blend(&fields, &w, &m), n);
        let standardized = |buf: &wgpu::Buffer, m: &Moments| self.stages.field_to_vec(&self.stages.standardize_to_buffer(buf, n, m), n);
        let cells = fields.cells.as_ref().map(|(c, o)| CellFields {
            closed_sum: self.stages.field_to_vec(c, n),
            open_sum: self.stages.field_to_vec(o, n),
            closed_std: standardized(c, &m.closed),
            open_std: standardized(o, &m.open),
        });
        RegimeTextureTerms { texture, generic_std: standardized(&fields.generic, &m.generic), street_std: standardized(&fields.street, &m.street), cells }
    }
}

/// `regime_weights` of host latitudes on the GPU, read back (one set of kernels per call: use [`TextureKernels`]
/// for repeated calls).
pub fn regime_weights(gpu: &GpuContext, lat: &[f32], lat_back: &[f32], hadley_deg: f64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    TextureKernels::new(gpu).regime_weights_to_vec(lat, lat_back, hadley_deg)
}

/// `regime_weights` of a band already on the GPU, left on the GPU (see [`TextureKernels::regime_weights_to_buffers`]).
pub fn regime_weights_to_buffers(gpu: &GpuContext, lat: LatInput<'_>, lat_back: &wgpu::Buffer, n: usize, hadley_deg: f64) -> WeightBuffers {
    TextureKernels::new(gpu).regime_weights_to_buffers(lat, lat_back, n, hadley_deg)
}

/// `regime_texture` of one band that is the whole array, left on the GPU (see
/// [`TextureKernels::regime_texture_to_buffer`]).
#[allow(clippy::too_many_arguments)]
#[maybe_async::maybe_async]
pub async fn regime_texture_to_buffer(
    gpu: &GpuContext,
    p: &wgpu::Buffer,
    back: &wgpu::Buffer,
    weights: &WeightBuffers,
    n: usize,
    radius_km: f64,
    fmax: f64,
    plans: &RegimeTexturePlans,
) -> wgpu::Buffer {
    TextureKernels::new(gpu).regime_texture_to_buffer(p, back, weights, n, radius_km, fmax, plans).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use cloud_map_generator_core::sphere::grid_dirs;
    use cloud_map_generator_core::texture::{regime_texture_terms, regime_weight_terms, RegimeWeights};

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

    /// Reads the f32 literal of every `ramp(a, ba, ...)` call of the shader and checks it against the Python
    /// floats' rounding: `a as f32` and `(b - a) as f32` (the difference formed in f64, as `lit::<f32>(b - a)`).
    #[test]
    fn the_ramp_constants_are_the_f32_roundings_of_the_python_floats() {
        let knees: [(f64, f64); 7] = [(0.4, 0.7), (1.1, 1.5), (0.8, 1.2), (2.0, 2.6), (0.0, 8.0), (0.03, 0.30), (0.05, 0.25)];
        let calls: Vec<(f32, f32)> = SHADER
            .match_indices("ramp(")
            .filter(|(i, _)| !SHADER[..*i].ends_with("fn "))
            .filter_map(|(i, _)| {
                let args = &SHADER[i + 5..];
                let end = args.find(')')?;
                let mut it = args[..end].split(',').map(|s| s.trim().parse::<f32>());
                match (it.next(), it.next()) {
                    (Some(Ok(a)), Some(Ok(ba))) => Some((a, ba)),
                    _ => None, // the call passes a variable
                }
            })
            .collect();
        for (a, b) in knees {
            let want = (a as f32, (b - a) as f32);
            assert!(calls.contains(&want), "no ramp({}, {}, ..) in texture.wgsl: the knees ({a}, {b}) are not written as f32({a}) and f32({b} - {a})", want.0, want.1);
        }
        // and the gaussians' constants are plain f32 roundings of the Python floats: nothing to subtract
        assert_eq!(0.55_f32, 0.55_f64 as f32);
    }

    fn sweep(n: usize) -> (Vec<f32>, Vec<f32>) {
        let lat: Vec<f32> = (0..n).map(|i| ((i as f32 / n as f32) - 0.5) * std::f32::consts::PI).collect();
        let back: Vec<f32> = lat.iter().enumerate().map(|(i, &l)| l + l.signum() * (((i * 7919) % 97) as f32 / 97.0 * 0.45 - 0.05)).collect();
        (lat, back)
    }

    #[test]
    fn regime_weights_match_the_cpu_port_chain_of_float32_latitudes() {
        let Some(gpu) = test_gpu() else { return };
        let k = TextureKernels::new(&gpu);
        let (lat, back) = sweep(20_000);
        for hadley in [10.0, 30.0, 47.5, 90.0] {
            let ([closed, open, street], [convective, cold, total]) = k.regime_weight_terms_to_vec(&lat, &back, hadley);
            let cpu = regime_weight_terms::<f32, f32, f32>(&lat, &back, hadley);
            let worst = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
            assert!(worst(&closed, &cpu.weights.closed) < 2e-6, "hadley {hadley} closed {}", worst(&closed, &cpu.weights.closed));
            assert!(worst(&open, &cpu.weights.open) < 2e-6, "hadley {hadley} open {}", worst(&open, &cpu.weights.open));
            assert!(worst(&street, &cpu.weights.street) < 2e-6, "hadley {hadley} street {}", worst(&street, &cpu.weights.street));
            assert!(worst(&convective, &cpu.convective) < 2e-6, "hadley {hadley} convective {}", worst(&convective, &cpu.convective));
            assert!(worst(&cold, &cpu.cold) < 2e-6, "hadley {hadley} cold {}", worst(&cold, &cpu.cold));
            assert!(worst(&total, &cpu.total_raw) < 4e-6, "hadley {hadley} total {}", worst(&total, &cpu.total_raw));
            // the plain entry point is the same arithmetic
            let (c2, o2, s2) = k.regime_weights_to_vec(&lat, &back, hadley);
            assert_eq!((c2, o2, s2), (closed, open, street));
        }
    }

    #[test]
    fn the_row_latitude_table_is_the_per_texel_latitude() {
        let Some(gpu) = test_gpu() else { return };
        let k = TextureKernels::new(&gpu);
        let (w, h) = (97usize, 50usize);
        let lat_rows = k.stages().grid_lat_rows(w, h, 0, h);
        let lat: Vec<f32> = lat_rows.iter().flat_map(|&l| std::iter::repeat_n(l, w)).collect();
        let back: Vec<f32> = lat.iter().enumerate().map(|(i, &l)| l + l.signum() * (i % 11) as f32 * 0.01).collect();
        let (per_texel, _, _) = k.regime_weights_to_vec(&lat, &back, 30.0);
        let (rows_buf, back_buf) = (k.stages().upload_field(&lat_rows), k.stages().upload_field(&back));
        let wb = k.regime_weights_to_buffers(LatInput::Rows { lat: &rows_buf, width: w }, &back_buf, w * h, 30.0);
        assert_eq!(k.stages().field_to_vec(&wb.closed, w * h), per_texel);
        // chunks forced to 1024 texels: the row of a texel is found from the chunk's base
        let small = TextureKernels::with_stages(&gpu, StageKernels::with_binding_bytes(&gpu, 16 * 1024));
        let (rows_small, back_small) = (small.stages().grid_lat_rows_to_buffer(w, h, 0, h), small.stages().upload_field(&back));
        let ws = small.regime_weights_to_buffers(LatInput::Rows { lat: &rows_small, width: w }, &back_small, w * h, 30.0);
        assert_eq!(small.stages().field_to_vec(&ws.closed, w * h), per_texel);
        assert_eq!(small.stages().field_to_vec(&ws.street, w * h), k.stages().field_to_vec(&wb.street, w * h));
        // a band of rows 10..30 reads its own rows of the table
        let band_rows = k.stages().upload_field(&lat_rows[10..30]);
        let band_back = k.stages().upload_field(&back[10 * w..30 * w]);
        let wb2 = k.regime_weights_to_buffers(LatInput::Rows { lat: &band_rows, width: w }, &band_back, 20 * w, 30.0);
        assert_eq!(k.stages().field_to_vec(&wb2.closed, 20 * w), per_texel[10 * w..30 * w].to_vec());
    }

    /// A small grid of directions and a back-traced copy rotated a little about the z axis.
    fn grid(w: usize, h: usize) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let p = grid_dirs::<f32>(w, h);
        let (c, s) = (0.08_f32.cos(), 0.08_f32.sin());
        let back = p.iter().map(|d| [c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]]).collect();
        (p, back)
    }

    fn weights_of(n: usize, closed: f32, open: f32, street: f32) -> [Vec<f32>; 3] {
        [vec![closed; n], vec![open; n], vec![street; n]]
    }

    #[test]
    fn the_regime_texture_matches_the_cpu_port_on_a_grid() {
        let Some(gpu) = test_gpu() else { return };
        let k = TextureKernels::new(&gpu);
        let (p, back) = grid(128, 64);
        let n = p.len();
        for (radius, fmax, wts) in [(600.0, 50.0, (0.3_f32, 0.2_f32, 0.25_f32)), (6371.0, 326.0, (0.4, 0.1, 0.3)), (6371.0, 20.0, (0.7, 0.2, 0.3))] {
            let plans = RegimeTexturePlans::from_seed(radius, fmax, 5, 8.0, 0.6);
            let w = weights_of(n, wts.0, wts.1, wts.2);
            let gpu_terms = k.regime_texture_terms_to_vec(&p, &back, [&w[0], &w[1], &w[2]], radius, fmax, &plans);
            let cpu_w = RegimeWeights { closed: w[0].clone(), open: w[1].clone(), street: w[2].clone() };
            let cpu = regime_texture_terms(&p, &back, &cpu_w, radius, fmax, &plans);
            let worst = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
            assert!(worst(&gpu_terms.generic_std, &cpu.generic_std) < 2e-3, "radius {radius}: generic {}", worst(&gpu_terms.generic_std, &cpu.generic_std));
            assert!(worst(&gpu_terms.street_std, &cpu.street_std) < 2e-3, "radius {radius}: street {}", worst(&gpu_terms.street_std, &cpu.street_std));
            assert!(worst(&gpu_terms.texture, &cpu.texture) < 4e-3, "radius {radius}: texture {}", worst(&gpu_terms.texture, &cpu.texture));
            assert_eq!(gpu_terms.cells.is_some(), cpu.cells.is_some(), "radius {radius}: the cells exist when a diameter is used");
            if let (Some(g), Some(c)) = (&gpu_terms.cells, &cpu.cells) {
                assert!(worst(&g.closed_sum, &c.closed_sum) < 2e-3 && worst(&g.open_sum, &c.open_sum) < 2e-3, "radius {radius}: cell sums");
                assert!(worst(&g.closed_std, &c.closed_std) < 1e-2 && worst(&g.open_std, &c.open_std) < 1e-2, "radius {radius}: cell standardizations");
            }
        }
    }

    #[test]
    fn bands_with_merged_moments_equal_the_whole_array() {
        let Some(gpu) = test_gpu() else { return };
        let k = TextureKernels::new(&gpu);
        let (w, h) = (96usize, 48usize);
        let (p, back) = grid(w, h);
        let n = p.len();
        let (radius, fmax) = (600.0, 50.0);
        let plans = RegimeTexturePlans::from_seed(radius, fmax, 9, 8.0, 0.6);
        let wts = weights_of(n, 0.3, 0.2, 0.25);
        let whole = k.regime_texture_terms_to_vec(&p, &back, [&wts[0], &wts[1], &wts[2]], radius, fmax, &plans).texture;
        // three bands of 16 rows each
        let band = 16usize;
        let mut fields = Vec::new();
        let mut moments = TextureMoments::EMPTY;
        for b in 0..3 {
            let r = b * band * w..(b + 1) * band * w;
            let (pb, bb) = (k.stages().upload_dirs(&p[r.clone()]), k.stages().upload_dirs(&back[r.clone()]));
            let f = k.texture_fields(&pb, &bb, band * w, radius, fmax, &plans);
            moments = moments.merge(k.texture_moments(&f));
            fields.push((f, r));
        }
        let mut banded = Vec::new();
        for (f, r) in &fields {
            let weights = WeightBuffers {
                closed: k.stages().upload_field(&wts[0][r.clone()]),
                open: k.stages().upload_field(&wts[1][r.clone()]),
                street: k.stages().upload_field(&wts[2][r.clone()]),
            };
            banded.extend(k.stages().field_to_vec(&k.texture_blend(f, &weights, &moments), band * w));
        }
        let worst = whole.iter().zip(&banded).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(worst < 1e-5, "merging the bands' moments is the whole array's statistics, worst {worst}");
    }

    #[test]
    fn plans_for_another_resolution_are_refused() {
        let Some(gpu) = test_gpu() else { return };
        let k = TextureKernels::new(&gpu);
        let (p, back) = grid(16, 8);
        let plans = RegimeTexturePlans::from_seed(6371.0, 20.0, 1, 4.0, 0.65); // no cells
        let w = weights_of(p.len(), 0.1, 0.1, 0.1);
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            k.regime_texture_terms_to_vec(&p, &back, [&w[0], &w[1], &w[2]], 600.0, 50.0, &plans)
        }));
        let message = refused.expect_err("plans drawn for another resolution must be refused");
        let message = message.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(message.contains("one cell plan per used diameter"), "{message}");
    }
}
