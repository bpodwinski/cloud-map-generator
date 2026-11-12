//! The GPU worker of the banded engine (`cloud_map_generator_core::banded`): the three passes of one band on the GPU.
//!
//! The monolithic [`GpuPipeline::generate`] keeps every field of every band resident until the map is done. A map of
//! 32768 x 16384 does not fit, so here each pass recomputes the fields of ITS band from the position (the same stages,
//! the same kernels) and keeps nothing between passes but the small vectors of the engine:
//!
//! * pass A: the raw synoptic field and the four raw fields of the regime texture of the band, their moments (merged
//!   across the sub-bands the buffer limits impose) handed back as the `(sum, sum of squares)` of float64 the core's
//!   statistics are built from;
//! * pass B: the fields standardized with the statistics of the whole map, the row thresholds of the band by the
//!   GPU bisection (with the surface terms: the first solve, the sensitivity, the offset, the second solve), and the
//!   row sensitivity handed back for pass C;
//! * pass C: the fields again, the offset from the sensitivity of pass B, the coverage with the smoothed thresholds of
//!   the whole map, the type and high-share maps, and one readback of the band.
//!
//! The surface terms of a band, the targets of its rows and the sub-texel noise's description come from the core's
//! [`Banded`], so the CPU worker and this one read the same numbers. Everything is f32 on the device: a band equals the
//! CPU's band to float32 rounding, not bit for bit.

use super::*;
use cloud_map_generator_core::banded::{BandMoments, BandResult, BandThresholds, Banded, SubNoise};
use cloud_map_generator_core::generate::NormStats;
use cloud_map_generator_core::texture::Norm;

/// What a run keeps on the device between bands: the vortex table and the coarse back-trace.
pub struct GpuBandState {
    gpu_flow: GpuFlow,
    coarse_back: Option<wgpu::Buffer>,
    sw: usize,
}

/// The fields of one sub-band on the device.
struct Geo {
    p: wgpu::Buffer,
    back: wgpu::Buffer,
    lat_back: wgpu::Buffer,
    ascent: wgpu::Buffer,
    s_raw: wgpu::Buffer,
}

/// The standardized fields of one sub-band, ready for the solve and the composition.
struct SubFields {
    s: wgpu::Buffer,
    fine: wgpu::Buffer,
    streak: wgpu::Buffer,
    sub: Option<wgpu::Buffer>,
    ascent: wgpu::Buffer,
    weights: Option<WeightBuffers>,
    anomaly: Option<wgpu::Buffer>,
    anomaly_host: Vec<f32>,
}

/// `(sum, sum of squares)` of the values a [`Moments`] summarizes.
fn as_sums(m: Moments) -> Vec<(f64, f64)> {
    if m.count == 0 {
        return Vec::new();
    }
    let n = m.count as f64;
    vec![(m.mean * n, m.m2 + m.mean * m.mean * n)]
}

/// The statistics of the whole map, as the device's [`Moments`] (`count` texels).
fn moments_of(norm: Norm, count: u64) -> Moments {
    let std = f64::from(norm.std);
    Moments { count, mean: f64::from(norm.mean), m2: std * std * count as f64 }
}

impl GpuBandState {
    /// The device state of a run: uploads the flow and runs the coarse back-trace.
    ///
    /// # Errors
    /// A map as narrow as the back-trace grid (it is one band of the monolithic pipeline), which the banded GPU path
    /// does not cut.
    pub fn new(pipe: &GpuPipeline<'_>, banded: &Banded<'_>) -> Result<GpuBandState, String> {
        let params = &banded.params;
        let sw = params.structure_width.min(banded.width);
        if !params.isotropic && sw == banded.width {
            return Err(format!("a {}-px map is as narrow as the back-trace grid: the GPU bands need a wider one", banded.width));
        }
        let gpu_flow = GpuFlow::new(pipe.gpu, &banded.plans().flow);
        let coarse_back = (!params.isotropic).then(|| {
            let points: Vec<[f32; 3]> = grid_dirs::<f64>(sw, sw / 2).iter().map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]).collect();
            pipe.flow.backtrace_to_buffer(pipe.gpu, &gpu_flow, &points, DEFAULT_BACKTRACE_TIME, DEFAULT_BACKTRACE_STEPS)
        });
        Ok(GpuBandState { gpu_flow, coarse_back, sw })
    }

    /// The sub-bands of rows `r0 .. r1` that fit the buffer limits.
    fn sub_bands(pipe: &GpuPipeline<'_>, width: usize, r0: usize, r1: usize) -> Vec<(usize, usize)> {
        let limit = pipe.texture.stages().rows_per_chunk(width, 3).min(max_band_rows(pipe.gpu, width)).max(1);
        (r0..r1).step_by(limit).map(|a| (a, limit.min(r1 - a))).collect()
    }

    fn geometry(&self, pipe: &GpuPipeline<'_>, banded: &Banded<'_>, row0: usize, rows: usize) -> Geo {
        let (width, height) = (banded.width, banded.height);
        let prm = &banded.params;
        let plans = banded.plans();
        let stages = pipe.texture.stages();
        let noise = pipe.texture.noise();
        let n = rows * width;
        let p = stages.grid_dirs_to_buffer(width, height, row0, rows);
        let back = match &self.coarse_back {
            None => p.clone(),
            Some(coarse) => stages.upsample_dirs_to_buffer(coarse, self.sw, self.sw / 2, width, height, row0, rows),
        };
        let lat_back = stages.lat_of_dirs_to_buffer(&back, n);
        let moisture_noise = noise.fbm_to_buffer(&back, n, &plans.moisture);
        let ascent = pipe.gpu.storage_empty::<f32>("ascent", n);
        pipe.flow.ascent_buffers(pipe.gpu, &self.gpu_flow, &p, &ascent, n);
        let mid_dirs = stages.along_to_buffer(&p, &back, n, prm.advect_mid);
        let mid_noise = noise.fbm_to_buffer(&mid_dirs, n, &plans.mid);
        let synoptic = SynopticParams {
            hadley_deg: banded.hadley_deg(),
            a_moisture: prm.a_moisture,
            moisture_noise: prm.moisture_noise,
            a_ascent: prm.a_ascent,
            a_mid: prm.a_mid,
        };
        let s_raw = stages.synoptic_to_buffer(&lat_back, &moisture_noise, &ascent, &mid_noise, n, &synoptic);
        Geo { p, back, lat_back, ascent, s_raw }
    }

    /// Pass A of rows `r0 .. r1`.
    #[maybe_async::maybe_async]
    pub async fn pass_a(&self, pipe: &GpuPipeline<'_>, banded: &Banded<'_>, r0: usize, r1: usize) -> BandMoments {
        let width = banded.width;
        let stages = pipe.texture.stages();
        let (mut s, mut tex) = (Moments::EMPTY, TextureMoments::EMPTY);
        for (row0, rows) in Self::sub_bands(pipe, width, r0, r1) {
            let n = rows * width;
            let g = self.geometry(pipe, banded, row0, rows);
            s = s.merge(stages.moments(&g.s_raw, n).await);
            if let TexturePlans::Regime(tp) = &banded.plans().texture {
                let fields = pipe.texture.texture_fields(&g.p, &g.back, n, banded.body.radius_km, banded.texture_fmax(), tp);
                tex = tex.merge(pipe.texture.texture_moments(&fields).await);
            }
        }
        BandMoments { s: as_sums(s), generic: as_sums(tex.generic), street: as_sums(tex.street), closed: as_sums(tex.closed), open: as_sums(tex.open) }
    }

    /// The fields of a sub-band with the statistics of the map applied.
    fn fields(&self, pipe: &GpuPipeline<'_>, banded: &Banded<'_>, norm: &NormStats, row0: usize, rows: usize) -> SubFields {
        let (width, height) = (banded.width, banded.height);
        let prm = &banded.params;
        let stages = pipe.texture.stages();
        let n = rows * width;
        let total = (width * height) as u64;
        let g = self.geometry(pipe, banded, row0, rows);
        let s = stages.standardize_to_buffer(&g.s_raw, n, &moments_of(norm.s, total));
        let phi_h = banded.hadley_deg();
        let terms = banded.band_terms(row0, row0 + rows);
        let need_weights = prm.regime_texture || banded.want_extras;
        let weights = need_weights.then(|| {
            let lat_buffer = stages.grid_lat_rows_to_buffer(width, height, row0, rows);
            let w = pipe.texture.regime_weights_to_buffers(LatInput::Rows { lat: &lat_buffer, width }, &g.lat_back, n, phi_h);
            if let Some(t) = &terms {
                let rad2deg = 180.0 / std::f64::consts::PI;
                let window: Vec<f32> =
                    banded.lat_rows()[row0..row0 + rows].iter().map(|&lat| deck_window::<f64>((lat * rad2deg).abs() / phi_h) as f32).collect();
                pipe.reshape_weights(&w, &t.continentality, &t.deck_strength, &window, width);
            }
            w
        });
        let fine = match &banded.plans().texture {
            TexturePlans::Regime(tp) => {
                let fields = pipe.texture.texture_fields(&g.p, &g.back, n, banded.body.radius_km, banded.texture_fmax(), tp);
                let tn = norm.texture.expect("the texture statistics exist with the regime texture");
                let (closed, open) = tn.cells.map_or((Moments::EMPTY, Moments::EMPTY), |(c, o)| (moments_of(c, total), moments_of(o, total)));
                let moments = TextureMoments { generic: moments_of(tn.generic, total), street: moments_of(tn.street, total), closed, open };
                pipe.texture.texture_blend(&fields, weights.as_ref().expect("the regime weights"), &moments)
            }
            TexturePlans::Plain(plan) => {
                let along = stages.along_to_buffer(&g.p, &g.back, n, prm.advect_fine);
                pipe.texture.noise().fbm_to_buffer(&along, n, plan)
            }
        };
        let streak = pipe.texture.noise().fbm_to_buffer(&g.back, n, &banded.plans().streak);
        let (kind, seed, scale) = banded.noise();
        let sub = match kind {
            SubNoise::Off => None,
            SubNoise::Counter => Some(pipe.compose.fill_subtexel(rows, width, row0, seed, scale)),
            SubNoise::Array(all) => Some(pipe.gpu.storage_with("subtexel", &all[row0 * width..(row0 + rows) * width])),
        };
        let (anomaly, anomaly_host) = match terms {
            Some(t) => (Some(pipe.gpu.storage_with("anomaly", &t.anomaly)), t.anomaly),
            None => (None, Vec::new()),
        };
        SubFields { s, fine, streak, sub, ascent: g.ascent, weights, anomaly, anomaly_host }
    }

    /// Pass B of rows `r0 .. r1`.
    #[maybe_async::maybe_async]
    pub async fn pass_b(&self, pipe: &GpuPipeline<'_>, banded: &Banded<'_>, norm: &NormStats, r0: usize, r1: usize) -> BandThresholds {
        let width = banded.width;
        let (mut thr_raw, mut sens): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for (row0, rows) in Self::sub_bands(pipe, width, r0, r1) {
            let f = self.fields(pipe, banded, norm, row0, rows);
            let band = Band { rows, width, s: &f.s, fine: &f.fine, streak: StreakSrc::Buffer(&f.streak), subtexel: f.sub.as_ref() };
            let cal = pipe.compose.calibrate(&band, &banded.params, &banded.target_zonal()[row0..row0 + rows], f.anomaly.as_ref());
            thr_raw.extend(pipe.gpu.read_back_maybe::<f32>(&cal.thr_raw, rows).await.into_iter().map(f64::from));
            if let Some(sb) = &cal.sensitivity {
                sens.extend(pipe.gpu.read_back_maybe::<f32>(sb, rows).await.into_iter().map(f64::from));
            }
            for &v in &f.anomaly_host {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        BandThresholds { thr_raw, sensitivity: sens, anomaly_min: if lo.is_finite() { lo } else { f32::NAN }, anomaly_max: if hi.is_finite() { hi } else { f32::NAN } }
    }

    /// Pass C of rows `r0 .. r1`: `thr` and `sens` hold every row of the map.
    #[allow(clippy::too_many_arguments)]
    #[maybe_async::maybe_async]
    pub async fn pass_c(&self, pipe: &GpuPipeline<'_>, banded: &Banded<'_>, norm: &NormStats, thr: &[f64], sens: &[f64], r0: usize, r1: usize) -> BandResult {
        let width = banded.width;
        let prm = &banded.params;
        let phi_h = banded.hadley_deg();
        let itcz_rad = banded.itcz_deg().to_radians();
        let (mut coverage, mut kind, mut high) = (Vec::with_capacity((r1 - r0) * width), Vec::new(), Vec::new());
        for (row0, rows) in Self::sub_bands(pipe, width, r0, r1) {
            let f = self.fields(pipe, banded, norm, row0, rows);
            let band = Band { rows, width, s: &f.s, fine: &f.fine, streak: StreakSrc::Buffer(&f.streak), subtexel: f.sub.as_ref() };
            let thr32: Vec<f32> = thr[row0..row0 + rows].iter().map(|&v| v as f32).collect();
            let thr_buffer = pipe.gpu.storage_with("thresholds smoothed", &thr32);
            let offset_buffer = f.anomaly.as_ref().map(|anomaly| {
                let zonal: Vec<f32> = banded.target_zonal()[row0..row0 + rows].iter().map(|&v| v as f32).collect();
                let zonal_buffer = pipe.gpu.storage_with("target zonal", &zonal);
                let target = pipe.compose.row_targets(anomaly, &zonal_buffer, rows, width);
                let sens32: Vec<f32> = sens[row0..row0 + rows].iter().map(|&v| v as f32).collect();
                let sens_buffer = pipe.gpu.storage_with("sensitivity", &sens32);
                pipe.compose.surface_offsets(anomaly, &zonal_buffer, &target, &sens_buffer, rows, width)
            });
            let offset = offset_buffer.as_ref().map_or(OffsetSrc::None, OffsetSrc::Full);
            let cover = pipe.compose.compose(&band, prm, &thr_buffer, offset);
            let products = banded.want_extras.then(|| {
                let lat_conv: Vec<f64> = banded.lat_rows()[row0..row0 + rows].iter().map(|&la| la - itcz_rad).collect();
                let w = f.weights.as_ref().expect("the regime weights are made for the extras");
                pipe.products.products(
                    &ProductInputs {
                        width,
                        lat_rows: &lat_conv,
                        hadley_deg: phi_h,
                        s: &f.s,
                        thr: &thr_buffer,
                        offset,
                        closed: &w.closed,
                        open: &w.open,
                        street: &w.street,
                        ascent: &f.ascent,
                        coverage: &cover,
                    },
                    prm,
                )
            });
            coverage.extend(pipe.gpu.read_back_maybe::<f32>(&cover, rows * width).await);
            if let Some((k, h)) = &products {
                kind.extend(pipe.gpu.read_back_maybe::<f32>(k, rows * width).await);
                high.extend(pipe.gpu.read_back_maybe::<f32>(h, rows * width).await);
            }
        }
        BandResult {
            index: r0 / banded.band_rows,
            r0,
            rows: r1 - r0,
            width,
            coverage,
            kind: banded.want_extras.then_some(kind),
            high_share: banded.want_extras.then_some(high),
        }
    }
}
