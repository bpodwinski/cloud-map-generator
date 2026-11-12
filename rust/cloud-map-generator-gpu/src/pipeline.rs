//! The GPU pipeline: `generate()` of the CPU port with every per-texel stage on the GPU. It chains the kernels of
//! this crate (flow, noise, stages, texture, compose, products) on GPU buffers: the fields stay on the device
//! between stages and the maps come back in one readback at the end.
//!
//! [`generate_gpu`] has the inputs and the output of [`cloud_map_generator_core::generate::generate_with_plans`] (minus the
//! trace: the intermediate fields never leave the GPU); [`generate_gpu_seeded`] draws the plans from a seed as
//! [`cloud_map_generator_core::generate::generate`] does, except the sub-texel noise, which the GPU makes itself from a
//! counter-based generator (see [`crate::compose`]) instead of receiving 8 million normals from the CPU.
//! [`GpuPipeline`] holds the compiled kernels: build it once and run many maps (a server does); the two functions
//! build a fresh one per call, which recompiles every shader.
//!
//! # What runs where
//!
//! On the GPU, per texel: the grid directions, the back-trace (the coarse grid), its upsampling, the ascent, the
//! synoptic field and its standardization, the regime weights (and their reshaping by the surface), the fine
//! texture, the streak, the sub-texel noise when the counter-based form is asked, the per-row threshold
//! bisections and the sensitivity, the surface offset, the coverage, and the type and high-share maps.
//!
//! On the CPU: the plans (random draws, in [`generate_gpu_seeded`]), the coarse grid of the back-trace (512 x
//! 256 directions), the wind map (a coarse grid), the surface terms, the zonal targets, the smoothing of the
//! row thresholds (one value per row) and the sidecar numbers. No per-texel stage of the map runs on the CPU.
//!
//! # Bands
//!
//! A buffer holds one band of whole rows, as in [`crate::stages`] and [`crate::compose`]: the pipeline cuts the
//! map into the fewest equal bands that respect the WebGPU default limits (128 MiB per binding, 65535 workgroups
//! per dimension) whatever the adapter offers, so the same WGSL and the same chunking run in a browser. A map up
//! to 4096 x 2048 is one band; 8192 x 4096 is four. The statistics over the whole map (the synoptic field, the
//! four fields of the regime texture) are merged across the bands before any band is standardized, and the
//! smoothing of the thresholds sees every row.
//!
//! # Precision
//!
//! f32, so a map equals the CPU map to float32 rounding and statistically, not bit for bit (WGSL has no f64).
//! The tests of `tests/gpu_pipeline.rs` measure how far, per case and on whole maps.

use crate::compose::{max_band_rows, smooth_thresholds, subtexel_noise_scale, Band, ComposeKernels, OffsetSrc, StreakSrc};
use crate::flow::{FlowKernels, GpuFlow};
use crate::products::{ProductInputs, ProductKernels};
use crate::stages::{Moments, SynopticParams};
use crate::texture::{LatInput, TextureFields, TextureKernels, TextureMoments, WeightBuffers};
use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use cloud_map_generator_core::climate::{hadley_edge_deg, is_bakeable, target_zonal_coverage, BodyInputs};
use cloud_map_generator_core::flow::{build_flow, Flow};
use cloud_map_generator_core::generate::{
    area_mean, fine_ladder, isotropic_flow, py_round, subtexel_scale, texture_fmax, Extras, FlowMeta, GenMeta, GenOutput, GenParams, GeneratePlans,
    GenerateError, SurfaceMeta, TexturePlans, VortexMeta, MID_OCTAVES, MOISTURE_FREQUENCY, MOISTURE_OCTAVES, ORIENTATION, SEED_FINE, SEED_MID,
    SEED_MOISTURE, SEED_STREAK, SEED_SUBTEXEL, STREAK_GAIN, SURFACE_TERM_WIDTH,
};
use cloud_map_generator_core::noise::{FbmPlan, DEFAULT_GAIN, DEFAULT_LACUNARITY};
use cloud_map_generator_core::sphere::{grid_dirs, grid_lonlat};
use cloud_map_generator_core::surface::{deck_window, surface_terms, Surface, SurfaceTerms, WindGrid};
use cloud_map_generator_core::texture::RegimeTexturePlans;
use cloud_map_generator_core::products::wind_map;
use std::sync::OnceLock;
use crate::Instant;

mod bands;
pub use bands::GpuBandState;

const SHADER: &str = include_str!("../shaders/pipeline.wgsl");

/// Texels along x of a workgroup of `surface_weights_main`.
const TEXEL_GROUP_X: usize = 64;

/// The `Params` uniform of `pipeline.wgsl` (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SurfaceParams {
    width: u32,
    rows: u32,
    pad0: u32,
    pad1: u32,
}

/// Where the time of a map went: what the CPU did (plans excluded: the caller draws them) and what the rest of
/// the wall time was (the uploads, the kernels, the waiting and the readback: the GPU side).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuTimings {
    /// Seconds spent in the CPU stages of the pipeline (the coarse grid, the wind, the surface terms, the
    /// targets, the smoothing of the thresholds, the sidecar).
    pub cpu_seconds: f64,
    /// The wall time of the call less `cpu_seconds`.
    pub gpu_seconds: f64,
}

/// The sub-texel field of a run.
#[derive(Clone, Copy)]
enum Subtexel<'a> {
    /// The term is off.
    Off,
    /// The CPU's field, already divided by `float32(sqrt(n_clouds))` ([`GeneratePlans::subtexel`]).
    Array(&'a [f32]),
    /// The GPU's counter-based field: `counter_normal(seed, texel) * scale`.
    Counter { seed: u64, scale: f32 },
}

/// The fine texture's state between its two passes.
enum FineStage {
    /// The regime texture: the four raw fields, waiting for the statistics of the whole map.
    Regime(TextureFields),
    /// The plain fBm: the texture itself.
    Ready(wgpu::Buffer),
}

/// The compiled kernels of the pipeline.
pub struct GpuPipeline<'a> {
    gpu: &'a GpuContext,
    flow: FlowKernels,
    texture: TextureKernels<'a>,
    compose: ComposeKernels<'a>,
    products: ProductKernels<'a>,
    surface_weights: OnceLock<wgpu::ComputePipeline>,
    /// Bands of at most this many rows (tests force small bands); `None`: as many as the limits allow.
    forced_band_rows: Option<usize>,
}

impl<'a> GpuPipeline<'a> {
    /// The kernels of `gpu`. The back-trace uses the compensated position when the adapter's error-free sums are
    /// exact, the plain f32 one when they are not (it is then no worse and cheaper).
    #[maybe_async::maybe_async]
    pub async fn new(gpu: &'a GpuContext) -> GpuPipeline<'a> {
        let compensated = FlowKernels::new();
        let flow = if compensated.compensation_works(gpu).await { compensated } else { FlowKernels::with_precision(crate::flow::Precision::Plain) };
        GpuPipeline {
            gpu,
            flow,
            texture: TextureKernels::new(gpu),
            compose: ComposeKernels::new(gpu),
            products: ProductKernels::new(gpu),
            surface_weights: OnceLock::new(),
            forced_band_rows: None,
        }
    }

    /// The same kernels cutting the map into bands of at most `rows` rows (a test of the banding).
    pub fn with_band_rows(mut self, rows: usize) -> GpuPipeline<'a> {
        assert!(rows >= 1, "a band holds at least one row");
        self.forced_band_rows = Some(rows);
        self
    }

    pub fn gpu(&self) -> &GpuContext {
        self.gpu
    }

    /// Compiles every pipeline the default parameters use, by making a small map of Earth: the first map of a
    /// server then does not pay for the shaders. Returns the time it took.
    #[maybe_async::maybe_async]
    pub async fn warm_up(&self) -> f64 {
        let t0 = Instant::now();
        let body = cloud_map_generator_core::climate::EARTH;
        let params = GenParams::default();
        // 640 wide: past the 512 of the back-trace grid, so the upsampling kernel is built too.
        let width = 640;
        let plans = plans_without_subtexel(&body, 1, width, &params);
        let sub = Subtexel::Counter { seed: 1u64.wrapping_add(SEED_SUBTEXEL), scale: 1.0 };
        self.run(&body, &plans, sub, width, &params, None, true, &mut |_, _| {})
            .await
            .expect("the warm-up map is Earth's");
        t0.elapsed().as_secs_f64()
    }

    /// `generate_with_plans` on the GPU (see [`generate_gpu`]), with the seconds the CPU stages took.
    #[allow(clippy::too_many_arguments)]
    #[maybe_async::maybe_async]
    pub async fn generate(
        &self,
        body: &BodyInputs,
        plans: &GeneratePlans,
        width: usize,
        params: &GenParams,
        surface: Option<&Surface>,
        want_extras: bool,
        progress: &mut dyn FnMut(f32, &str),
    ) -> Result<(GenOutput, GpuTimings), GenerateError> {
        let sub = match &plans.subtexel {
            Some(array) => Subtexel::Array(array),
            None => Subtexel::Off,
        };
        self.run(body, plans, sub, width, params, surface, want_extras, progress).await
    }

    /// `generate(body, seed, width, ...)` on the GPU (see [`generate_gpu_seeded`]), with the seconds the CPU
    /// stages took (drawing the plans included).
    #[allow(clippy::too_many_arguments)]
    #[maybe_async::maybe_async]
    pub async fn generate_seeded(
        &self,
        body: &BodyInputs,
        seed: u64,
        width: usize,
        params: &GenParams,
        surface: Option<&Surface>,
        want_extras: bool,
        progress: &mut dyn FnMut(f32, &str),
    ) -> Result<(GenOutput, GpuTimings), GenerateError> {
        if !is_bakeable(body) && !params.allow_any_regime {
            return Err(GenerateError::NotBakeable { cloud_regime: body.cloud_regime.to_string() });
        }
        let t0 = Instant::now();
        let plans = plans_without_subtexel(body, seed, width, params);
        let drawn = t0.elapsed().as_secs_f64();
        let sub = match subtexel_scale(body, width, params) {
            Some(divisor) => Subtexel::Counter { seed: seed.wrapping_add(SEED_SUBTEXEL), scale: subtexel_noise_scale(divisor) },
            None => Subtexel::Off,
        };
        let (mut out, mut timings) = self.run(body, &plans, sub, width, params, surface, want_extras, progress).await?;
        out.meta.seed = seed;
        timings.cpu_seconds += drawn;
        Ok((out, timings))
    }

    // -- helpers -----------------------------------------------------------------------------------------------

    /// The bands of a map: `(first row, rows)`, the fewest equal ones that fit the limits.
    fn bands(&self, width: usize, height: usize) -> Vec<(usize, usize)> {
        let by_limits = self.texture.stages().rows_per_chunk(width, 3).min(max_band_rows(self.gpu, width)).max(1);
        let rows = by_limits.min(self.forced_band_rows.unwrap_or(usize::MAX)).max(1);
        let count = height.div_ceil(rows);
        let rows = height.div_ceil(count);
        (0..height).step_by(rows).map(|r0| (r0, rows.min(height - r0))).collect()
    }

    fn surface_pipeline(&self) -> &wgpu::ComputePipeline {
        self.surface_weights.get_or_init(|| self.gpu.pipeline("surface_weights", SHADER, "surface_weights_main"))
    }

    /// `surface_regime_weights` of a band, in place: `continentality` and `strength` are `rows * width` f32, the
    /// window one f32 per row.
    fn reshape_weights(&self, weights: &WeightBuffers, continentality: &[f32], strength: &[f32], window: &[f32], width: usize) {
        let rows = window.len();
        let (cont, deck) = (self.gpu.storage_with("continentality", continentality), self.gpu.storage_with("deck strength", strength));
        let window = self.gpu.storage_with("deck window", window);
        let params = self.gpu.uniform_with("surface weights params", &SurfaceParams { width: width as u32, rows: rows as u32, pad0: 0, pad1: 0 });
        let groups = (width.div_ceil(TEXEL_GROUP_X) as u32, rows as u32, 1);
        self.gpu.dispatch(self.surface_pipeline(), &[&params, &cont, &deck, &window, &weights.closed, &weights.open, &weights.street], groups);
    }

    /// The whole pipeline.
    #[allow(clippy::too_many_arguments)]
    #[maybe_async::maybe_async]
    async fn run(
        &self,
        body: &BodyInputs,
        plans: &GeneratePlans,
        subtexel: Subtexel<'_>,
        width: usize,
        params: &GenParams,
        surface: Option<&Surface>,
        want_extras: bool,
        progress: &mut dyn FnMut(f32, &str),
    ) -> Result<(GenOutput, GpuTimings), GenerateError> {
        let start = Instant::now();
        if !is_bakeable(body) && !params.allow_any_regime {
            return Err(GenerateError::NotBakeable { cloud_regime: body.cloud_regime.to_string() });
        }
        assert!(width >= 2, "a map needs at least 2 columns (a 1 x 2 grid), got {width}");
        let height = width / 2;
        let n_total = height * width;
        if let Subtexel::Array(sub) = subtexel {
            assert_eq!(sub.len(), n_total, "the sub-texel plan must hold height x width values");
        }
        assert_eq!(
            matches!(plans.texture, TexturePlans::Regime(_)),
            params.regime_texture,
            "the texture plan must be the regime texture's when params.regime_texture is on, the plain fBm's otherwise"
        );
        let mut cpu = 0.0_f64;
        // Runs a CPU stage and adds its time to the CPU's.
        let mut on_cpu = |f: &mut dyn FnMut()| {
            let t = Instant::now();
            f();
            cpu += t.elapsed().as_secs_f64();
        };

        let phi_h = hadley_edge_deg(body);
        let flow = &plans.flow;
        let stages = self.texture.stages();
        let noise = self.texture.noise();
        let bands = self.bands(width, height);
        let band_n = |rows: usize| rows * width;
        progress(0.0, &format!("GPU: {} x {} map in {} band(s)", width, height, bands.len()));

        // CPU: the coarse grid of the back-trace, the wind, the surface terms, the latitude of every row.
        let use_surface = surface.is_some() && params.surface_terms && !params.isotropic;
        let sw = params.structure_width.min(width);
        let mut coarse_points: Vec<[f32; 3]> = Vec::new();
        let mut lat_rows: Vec<f64> = Vec::new();
        let mut wind: Option<Vec<f32>> = None;
        let mut terms: Option<SurfaceTerms> = None;
        let wind_width = (params.structure_width * 2).min(width);
        let wind_height = wind_width / 2;
        on_cpu(&mut || {
            if !params.isotropic {
                assert!(sw >= 2, "structure_width must be at least 2, got {}", params.structure_width);
                coarse_points = grid_dirs::<f64>(sw, sw / 2).iter().map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]).collect();
            }
            lat_rows = grid_lonlat(width, height).1;
            if use_surface || want_extras {
                wind = Some(wind_map(flow, body, wind_width));
            }
            if let (true, Some(surf)) = (use_surface, surface) {
                let uv = wind.as_deref().expect("the wind is made when the surface terms run");
                let grid = WindGrid { uv, h: wind_height, w: wind_width };
                terms = Some(surface_terms(surf, width, phi_h, body.radius_km, spin_sign(body), Some(&grid), SURFACE_TERM_WIDTH));
            }
        });
        if let Some(t) = &terms {
            assert!(t.anomaly.len() == n_total && t.continentality.len() == n_total && t.deck_strength.len() == n_total, "the surface terms are not height x width");
        }
        let itcz = terms.as_ref().map_or(0.0, |t| t.itcz_shift_deg);

        // GPU: where the air came from, on the coarse grid.
        let gpu_flow = GpuFlow::new(self.gpu, flow);
        let coarse_back = (!params.isotropic).then(|| self.flow.backtrace_to_buffer(self.gpu, &gpu_flow, &coarse_points, params.backtrace_time, params.backtrace_steps()));
        progress(0.05, "GPU: back-trace and the synoptic field");

        // Phase A, per band: the directions, the synoptic field (raw), its statistics.
        let synoptic = SynopticParams {
            hadley_deg: phi_h,
            a_moisture: params.a_moisture,
            moisture_noise: params.moisture_noise,
            a_ascent: params.a_ascent,
            a_mid: params.a_mid,
        };
        let mut dirs: Vec<(wgpu::Buffer, wgpu::Buffer)> = Vec::new(); // (p, back)
        let mut lat_backs: Vec<wgpu::Buffer> = Vec::new();
        let mut ascents: Vec<wgpu::Buffer> = Vec::new();
        let mut s_raw: Vec<wgpu::Buffer> = Vec::new();
        let mut s_moments = Moments::EMPTY;
        for &(row0, rows) in &bands {
            let n = band_n(rows);
            let p = stages.grid_dirs_to_buffer(width, height, row0, rows);
            let back = match &coarse_back {
                None => p.clone(),
                Some(coarse) if sw == width => {
                    assert_eq!(bands.len(), 1, "a map as narrow as the back-trace grid is one band");
                    coarse.clone()
                }
                Some(coarse) => stages.upsample_dirs_to_buffer(coarse, sw, sw / 2, width, height, row0, rows),
            };
            let lat_back = stages.lat_of_dirs_to_buffer(&back, n);
            let moisture_noise = noise.fbm_to_buffer(&back, n, &plans.moisture);
            let ascent = self.gpu.storage_empty::<f32>("ascent", n);
            self.flow.ascent_buffers(self.gpu, &gpu_flow, &p, &ascent, n);
            let mid_dirs = stages.along_to_buffer(&p, &back, n, params.advect_mid);
            let mid_noise = noise.fbm_to_buffer(&mid_dirs, n, &plans.mid);
            let raw = stages.synoptic_to_buffer(&lat_back, &moisture_noise, &ascent, &mid_noise, n, &synoptic);
            s_moments = s_moments.merge(stages.moments(&raw, n).await);
            dirs.push((p, back));
            lat_backs.push(lat_back);
            ascents.push(ascent);
            s_raw.push(raw);
        }
        let s_fields: Vec<wgpu::Buffer> = bands.iter().zip(&s_raw).map(|(&(_, rows), raw)| stages.standardize_to_buffer(raw, band_n(rows), &s_moments)).collect();
        drop(s_raw);
        progress(0.35, "GPU: regime weights and fine texture");

        // Phase B, per band: the regime weights (reshaped by the surface), the fine texture's raw fields, the streak
        // and the sub-texel field.
        let need_weights = params.regime_texture || want_extras;
        let rad2deg = 180.0 / std::f64::consts::PI;
        let fmax = texture_fmax(fine_ladder(width, params).fmax, params);
        let mut weights: Vec<WeightBuffers> = Vec::new();
        let mut fine_stages: Vec<FineStage> = Vec::new();
        let mut streaks: Vec<wgpu::Buffer> = Vec::new();
        let mut subtexels: Vec<wgpu::Buffer> = Vec::new();
        let mut tex_moments = TextureMoments::EMPTY;
        for (b, &(row0, rows)) in bands.iter().enumerate() {
            let n = band_n(rows);
            let (p, back) = &dirs[b];
            if need_weights {
                let lat_buffer = stages.grid_lat_rows_to_buffer(width, height, row0, rows);
                let w = self.texture.regime_weights_to_buffers(LatInput::Rows { lat: &lat_buffer, width }, &lat_backs[b], n, phi_h);
                if let Some(t) = &terms {
                    let window: Vec<f32> = lat_rows[row0..row0 + rows].iter().map(|&lat| deck_window::<f64>((lat * rad2deg).abs() / phi_h) as f32).collect();
                    let span = row0 * width..(row0 + rows) * width;
                    self.reshape_weights(&w, &t.continentality[span.clone()], &t.deck_strength[span], &window, width);
                }
                weights.push(w);
            }
            fine_stages.push(match &plans.texture {
                TexturePlans::Regime(tp) => {
                    let fields = self.texture.texture_fields(p, back, n, body.radius_km, fmax, tp);
                    tex_moments = tex_moments.merge(self.texture.texture_moments(&fields).await);
                    FineStage::Regime(fields)
                }
                TexturePlans::Plain(plan) => {
                    let along = stages.along_to_buffer(p, back, n, params.advect_fine);
                    FineStage::Ready(noise.fbm_to_buffer(&along, n, plan))
                }
            });
            streaks.push(noise.fbm_to_buffer(back, n, &plans.streak));
            match subtexel {
                Subtexel::Off => {}
                Subtexel::Array(all) => subtexels.push(self.gpu.storage_with("subtexel", &all[row0 * width..(row0 + rows) * width])),
                Subtexel::Counter { seed, scale } => subtexels.push(self.compose.fill_subtexel(rows, width, row0, seed, scale)),
            }
        }
        // The fine texture, now that its statistics cover the whole map.
        let mut fines: Vec<wgpu::Buffer> = Vec::new();
        for (b, stage) in fine_stages.into_iter().enumerate() {
            fines.push(match stage {
                FineStage::Ready(buffer) => buffer,
                FineStage::Regime(fields) => self.texture.texture_blend(&fields, &weights[b], &tex_moments),
            });
        }
        drop(dirs);
        drop(lat_backs);
        progress(0.6, "GPU: threshold bisection");

        // Phase C: the calibration. Every band's rows are independent; the smoothing of the thresholds is not.
        let target_zonal: Vec<f64> = lat_rows.iter().map(|&la| target_zonal_coverage(la * rad2deg, phi_h, itcz)).collect();
        let anomalies: Vec<wgpu::Buffer> = match &terms {
            Some(t) => bands.iter().map(|&(row0, rows)| self.gpu.storage_with("anomaly", &t.anomaly[row0 * width..(row0 + rows) * width])).collect(),
            None => Vec::new(),
        };
        let mut calibrations = Vec::new();
        let mut thr_raw: Vec<f32> = Vec::with_capacity(height);
        for (b, &(row0, rows)) in bands.iter().enumerate() {
            let band = Band {
                rows,
                width,
                s: &s_fields[b],
                fine: &fines[b],
                streak: StreakSrc::Buffer(&streaks[b]),
                subtexel: subtexels.get(b),
            };
            let cal = self.compose.calibrate(&band, params, &target_zonal[row0..row0 + rows], anomalies.get(b));
            thr_raw.extend(self.gpu.read_back_maybe::<f32>(&cal.thr_raw, rows).await);
            calibrations.push(cal);
        }
        let mut thr: Vec<f32> = Vec::new();
        on_cpu(&mut || thr = smooth_thresholds(&thr_raw, params, height));
        progress(0.8, "GPU: coverage and products");

        // Phase D: the coverage, then the products, band by band.
        let lat_conv: Vec<f64> = {
            let itcz_rad = itcz.to_radians();
            lat_rows.iter().map(|&la| la - itcz_rad).collect()
        };
        let mut coverage: Vec<f32> = Vec::with_capacity(n_total);
        let mut kind: Vec<f32> = Vec::new();
        let mut high_share: Vec<f32> = Vec::new();
        if want_extras {
            kind.reserve(n_total);
            high_share.reserve(n_total);
        }
        for (b, &(row0, rows)) in bands.iter().enumerate() {
            let band = Band {
                rows,
                width,
                s: &s_fields[b],
                fine: &fines[b],
                streak: StreakSrc::Buffer(&streaks[b]),
                subtexel: subtexels.get(b),
            };
            let thr_buffer = self.gpu.storage_with("thresholds smoothed", &thr[row0..row0 + rows]);
            let offset = calibrations[b].offset.as_ref().map_or(OffsetSrc::None, OffsetSrc::Full);
            let cover = self.compose.compose(&band, params, &thr_buffer, offset);
            let products = want_extras.then(|| {
                self.products.products(
                    &ProductInputs {
                        width,
                        lat_rows: &lat_conv[row0..row0 + rows],
                        hadley_deg: phi_h,
                        s: &s_fields[b],
                        thr: &thr_buffer,
                        offset,
                        closed: &weights[b].closed,
                        open: &weights[b].open,
                        street: &weights[b].street,
                        ascent: &ascents[b],
                        coverage: &cover,
                    },
                    params,
                )
            });
            // The readback of the band: the one synchronization of the phase.
            coverage.extend(self.gpu.read_back_maybe::<f32>(&cover, rows * width).await);
            if let Some((k, h)) = &products {
                kind.extend(self.gpu.read_back_maybe::<f32>(k, rows * width).await);
                high_share.extend(self.gpu.read_back_maybe::<f32>(h, rows * width).await);
            }
        }

        let extras = if want_extras {
            Some(Extras { kind, high_share, wind: wind.take().expect("the wind is made for the extras"), wind_height, wind_width })
        } else {
            None
        };
        let mut meta: Option<GenMeta> = None;
        on_cpu(&mut || meta = Some(gen_meta(flow, width, height, fine_ladder(width, params).octaves, phi_h, terms.as_ref(), surface)));
        progress(1.0, "GPU: done");
        let total = start.elapsed().as_secs_f64();
        let out = GenOutput { coverage, height, width, meta: meta.expect("the sidecar was made"), extras, trace: None };
        Ok((out, GpuTimings { cpu_seconds: cpu, gpu_seconds: (total - cpu).max(0.0) }))
    }
}

/// `math.copysign(1.0, period or 1.0)`: +1 for no spin or a zero period, -1 for a retrograde one (the rule of
/// `cloud_map_generator_core::generate`, which keeps its copy private).
fn spin_sign(body: &BodyInputs) -> f64 {
    let period = match body.rotation_period_days {
        Some(p) if p != 0.0 => p,
        _ => 1.0,
    };
    1.0_f64.copysign(period)
}

/// `np.min` / `np.max` of a float32 field (NaN propagates).
fn extremum(a: &[f32], pick_min: bool) -> f64 {
    let mut best = f32::NAN;
    let mut seen = false;
    for &v in a {
        if v.is_nan() {
            return f64::NAN;
        }
        if !seen || (pick_min && v < best) || (!pick_min && v > best) {
            best = v;
            seen = true;
        }
    }
    f64::from(best)
}

/// The sidecar numbers of a map, as `generate_with_plans` builds them.
fn gen_meta(
    flow: &Flow,
    width: usize,
    height: usize,
    fine_octaves: usize,
    hadley_edge: f64,
    terms: Option<&SurfaceTerms>,
    surface: Option<&Surface>,
) -> GenMeta {
    GenMeta {
        seed: 0,
        width,
        height,
        orientation: ORIENTATION,
        hadley_edge_deg: hadley_edge,
        fine_octaves,
        flow: FlowMeta {
            jet: flow.jet,
            trades: flow.trades,
            polar_easterlies: flow.polar_easterlies,
            vortices: flow
                .vortices
                .iter()
                .map(|v| VortexMeta {
                    kind: v.kind,
                    radius: v.radius,
                    omega: v.omega,
                    ascent: v.ascent,
                    age: v.age,
                    lon_deg: v.lon.to_degrees(),
                    lat_deg: v.lat.to_degrees(),
                })
                .collect(),
        },
        surface: match (terms, surface) {
            (Some(t), Some(surf)) => Some(SurfaceMeta {
                source: surf.source.clone(),
                land_fraction: py_round(area_mean(&surf.land, surf.h, surf.w), 4),
                itcz_shift_deg: py_round(t.itcz_shift_deg, 2),
                anomaly_range: [py_round(extremum(&t.anomaly, true), 3), py_round(extremum(&t.anomaly, false), 3)],
            }),
            _ => None,
        },
    }
}

/// The plans of `generate(body, seed, width, params)` except the sub-texel field (`None`): the GPU makes that one
/// itself. Every other field is drawn exactly as [`GeneratePlans::from_seed`] draws it (a test holds the two to
/// it).
pub fn plans_without_subtexel(body: &BodyInputs, seed: u64, width: usize, params: &GenParams) -> GeneratePlans {
    let ladder = fine_ladder(width, params);
    let plan = |offset: u64, frequency: f64, octaves: usize, gain: f64, billow: bool| {
        FbmPlan::from_seed(seed.wrapping_add(offset), frequency, octaves, gain, DEFAULT_LACUNARITY, billow)
    };
    let moisture = plan(SEED_MOISTURE, MOISTURE_FREQUENCY, MOISTURE_OCTAVES, DEFAULT_GAIN, false);
    let mid = plan(SEED_MID, params.mid_frequency, MID_OCTAVES, DEFAULT_GAIN, false);
    let texture = if params.regime_texture {
        TexturePlans::Regime(RegimeTexturePlans::from_seed(body.radius_km, texture_fmax(ladder.fmax, params), seed, params.fine_frequency, params.fine_gain))
    } else {
        TexturePlans::Plain(plan(SEED_FINE, ladder.frequency, ladder.octaves, params.fine_gain, params.fine_billow))
    };
    let streak = plan(SEED_STREAK, ladder.frequency, ladder.octaves, STREAK_GAIN, false);
    let flow = if params.isotropic { isotropic_flow(body) } else { build_flow(body, seed) };
    GeneratePlans { moisture, mid, texture, streak, subtexel: None, flow }
}

/// `cloud_map_generator_core::generate::generate_with_plans` on the GPU: the same inputs and the same output (without a
/// trace). The coverage, and with `want_extras` the type, high-share and wind maps, equal the CPU's to float32
/// rounding and statistically; `plans.subtexel` is uploaded as it is. `progress(fraction, text)` is called at the
/// stage boundaries. Builds the kernels for this call: see [`GpuPipeline`] to keep them.
///
/// # Errors
/// [`GenerateError::NotBakeable`] when the clouds of `body` are not water and `params.allow_any_regime` is off.
#[allow(clippy::too_many_arguments)]
#[maybe_async::maybe_async]
pub async fn generate_gpu(
    gpu: &GpuContext,
    body: &BodyInputs,
    plans: &GeneratePlans,
    width: usize,
    params: &GenParams,
    surface: Option<&Surface>,
    want_extras: bool,
    progress: &mut dyn FnMut(f32, &str),
) -> Result<GenOutput, GenerateError> {
    GpuPipeline::new(gpu).await.generate(body, plans, width, params, surface, want_extras, progress).await.map(|(out, _)| out)
}

/// `cloud_map_generator_core::generate::generate` on the GPU: the plans drawn from `seed` with the port's `Rng` (the
/// sub-texel noise excepted: the GPU draws it from the counter-based generator of [`crate::compose`], so the map
/// of a seed is not the CPU's map of that seed, only statistically the same). The returned meta carries `seed`.
///
/// # Errors
/// [`GenerateError::NotBakeable`] as [`generate_gpu`].
#[allow(clippy::too_many_arguments)]
#[maybe_async::maybe_async]
pub async fn generate_gpu_seeded(
    gpu: &GpuContext,
    body: &BodyInputs,
    seed: u64,
    width: usize,
    params: &GenParams,
    surface: Option<&Surface>,
    want_extras: bool,
    progress: &mut dyn FnMut(f32, &str),
) -> Result<GenOutput, GenerateError> {
    GpuPipeline::new(gpu).await.generate_seeded(body, seed, width, params, surface, want_extras, progress).await.map(|(out, _)| out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cloud_map_generator_core::climate::EARTH;
    use cloud_map_generator_core::generate::surface_regime_weights;

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
        assert_eq!(std::mem::size_of::<SurfaceParams>(), 16);
    }

    #[test]
    fn the_seeded_plans_are_the_ports_plans_but_for_the_subtexel_field() {
        for params in [GenParams::default(), GenParams { regime_texture: false, ..GenParams::default() }, GenParams { isotropic: true, ..GenParams::default() }] {
            for width in [64, 600] {
                let mut drawn = GeneratePlans::from_seed(&EARTH, 7, width, &params);
                assert!(drawn.subtexel.is_some(), "the default parameters switch the sub-texel term on");
                drawn.subtexel = None;
                assert_eq!(plans_without_subtexel(&EARTH, 7, width, &params), drawn, "width {width}, {params:?}");
            }
        }
    }

    #[test]
    fn the_bands_are_equal_and_cover_the_map() {
        let Some(gpu) = test_gpu() else { return };
        let pipeline = GpuPipeline::new(&gpu);
        assert_eq!(pipeline.bands(4096, 2048), vec![(0, 2048)], "a 4096 map is one band");
        let eight_k = pipeline.bands(8192, 4096);
        assert!(eight_k.len() >= 3 && eight_k.iter().all(|&(_, rows)| rows == eight_k[0].1 || rows < eight_k[0].1), "{eight_k:?}");
        assert_eq!(eight_k.iter().map(|b| b.1).sum::<usize>(), 4096);
        assert!(eight_k.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0));
        let forced = GpuPipeline::new(&gpu).with_band_rows(100);
        let bands = forced.bands(512, 256);
        assert_eq!(bands, vec![(0, 86), (86, 86), (172, 84)]);
    }

    #[test]
    fn the_surface_weights_kernel_matches_the_cpu_port() {
        let Some(gpu) = test_gpu() else { return };
        let pipeline = GpuPipeline::new(&gpu);
        let (width, rows) = (200, 37);
        let n = width * rows;
        // Deterministic fields: shares that sum to at most 1, a land fraction and a deck strength in [0, 1].
        let field = |salt: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|i| (((i * 2654435761 + salt * 40503) % 1000) as f32 / 1000.0) * scale).collect()
        };
        let (closed, open, street) = (field(1, 0.5), field(2, 0.3), field(3, 0.2));
        let (cont, strength) = (field(4, 1.0), field(5, 1.0));
        let window_f64: Vec<f64> = (0..rows).map(|r| deck_window::<f64>(r as f64 / 20.0)).collect();
        let w = WeightBuffers { closed: gpu.storage_with("c", &closed), open: gpu.storage_with("o", &open), street: gpu.storage_with("s", &street) };
        let window: Vec<f32> = window_f64.iter().map(|&v| v as f32).collect();
        pipeline.reshape_weights(&w, &cont, &strength, &window, width);
        let got = [gpu.read_back::<f32>(&w.closed, n), gpu.read_back::<f32>(&w.open, n), gpu.read_back::<f32>(&w.street, n)];
        // The CPU port with the same window: hadley 30 and latitudes chosen so that x = r / 20.
        let hadley = 30.0;
        let lat_for_x: Vec<f64> = (0..rows).map(|r| (r as f64 / 20.0 * hadley).to_radians()).collect();
        let want = surface_regime_weights::<f32>(&closed, &open, &street, &cont, &strength, &lat_for_x, width, hadley);
        let worst = |g: &[f32], w: &[f32]| g.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        for (name, g, w) in [("closed", &got[0], &want.closed), ("open", &got[1], &want.open), ("street", &got[2], &want.street)] {
            let d = worst(g, w);
            eprintln!("surface weights {name}: max |GPU - CPU| = {d:e}");
            assert!(d < 2e-6, "{name}: {d:e}");
        }
    }
}
