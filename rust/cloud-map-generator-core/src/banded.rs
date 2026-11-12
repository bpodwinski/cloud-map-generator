//! The banded (out-of-core) engine: the whole generator in three passes over horizontal bands of rows, so that a
//! 32768 x 16384 map needs a few bands of memory instead of fifteen 2 GiB fields. The design is `docs/large-maps.html`.
//!
//! Every stage of [`crate::generate::generate_with_plans_norm`] is per texel except a few global couplings, each of
//! them a small vector. The fields are therefore recomputed band by band instead of stored:
//!
//! * pass A: each band computes its raw synoptic field `s` and the four raw fields of the regime texture (generic and
//!   street fBm, closed- and open-cell sums), and returns one `(sum, sum of squares)` per row in float64. The rows are
//!   the fixed blocks: the engine combines them in row order, so the statistics (and everything after them) do not
//!   depend on the band height. [`NormStats`] is the result: the mean and standard deviation of the five fields;
//! * pass B: each band recomputes its fields with the statistics and solves the thresholds of its rows (with the
//!   surface terms: the first solve, the row sensitivity and the second solve, all on the same band);
//! * the threshold vector (one value per row) is smoothed once ([`smooth_thresholds`]);
//! * pass C: each band recomputes its fields, composes with the final thresholds, derives the type and high-share
//!   maps and hands the band to a [`BandSink`].
//!
//! The sub-texel white noise is counter based ([`crate::counter`]): `counter_normal(seed, texel_index)`, so a band
//! draws its own texels. The array form of the golden cases lives behind the same [`SubNoise`] type.
//!
//! Equality is the test of the design: the output equals [`generate_with_plans_norm`] given the same statistics and
//! the same noise (bit for bit on the CPU), whatever the band height (`tests/banded.rs`).
//!
//! No `std::fs`, no threads outside the `par` helpers: this module builds for wasm32.

use crate::climate::{hadley_edge_deg, BodyInputs};
use crate::counter::{fill_subtexel, subtexel_noise_scale};
use crate::flow::{build_flow, Flow};
use crate::generate::{
    area_mean, along, clip, fine_ladder, isotropic_flow, local_target, moisture_profile, pairwise_sum, pipeline_weights, row_sensitivity,
    solve_thresholds, spin_sign, subtexel_scale, texture_fmax, compose, FlowMeta, GenMeta, GenParams, GeneratePlans, GenerateError,
    NormStats, Offset, Streak, Threshold, TraceWeights, VortexMeta, MID_OCTAVES, MOISTURE_FREQUENCY, MOISTURE_OCTAVES, ORIENTATION,
    SEED_FINE, SEED_MID, SEED_MOISTURE, SEED_STREAK, SEED_SUBTEXEL, STREAK_GAIN, SURFACE_TERM_WIDTH,
};
use crate::matching::ndimage::{self, Mode};
use crate::noise::{fbm, cell_edges, FbmPlan, DEFAULT_GAIN, DEFAULT_LACUNARITY};
use crate::products::{high_share, type_map, wind_map};
use crate::sphere::{grid_lonlat, lonlat_to_dir, normalize, sample_equirect, Vec3};
use crate::surface::{surface_terms, Surface, SurfaceTerms, WindGrid};
use crate::texture::{
    mixed_dirs, regime_texture_terms_norm, smoothstep, Norm, RegimeTexturePlans, TextureNorm,
};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------------------------------------
// Types

/// Where the sub-texel white noise comes from.
#[derive(Clone, Debug)]
pub enum SubNoise {
    /// The cumulus' sub-texel term is off (`cumulus_cell_km <= 0`).
    Off,
    /// The counter-based generator: `normal(seed, texel_index) / sqrt(n_clouds)`. The banded path's own source.
    Counter,
    /// A stored field of `height x width` values, already divided by `sqrt(n_clouds)` (the golden cases' form).
    Array(Vec<f32>),
}

/// The three passes, for the progress callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pass {
    A,
    B,
    C,
}

impl Pass {
    pub fn letter(self) -> char {
        match self {
            Pass::A => 'A',
            Pass::B => 'B',
            Pass::C => 'C',
        }
    }
}

/// One progress event: `band` of `bands` (1-based) of `pass` has just finished (`band == 0`: the pass starts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    pub pass: Pass,
    pub band: usize,
    pub bands: usize,
}

/// Why a banded run stopped.
#[derive(Debug)]
pub enum BandedError {
    /// The progress callback asked to stop.
    Cancelled,
    /// A sink refused a band.
    Sink(String),
    Generate(GenerateError),
}

impl std::fmt::Display for BandedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BandedError::Cancelled => write!(f, "cancelled"),
            BandedError::Sink(e) => write!(f, "{e}"),
            BandedError::Generate(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BandedError {}

impl From<GenerateError> for BandedError {
    fn from(e: GenerateError) -> Self {
        BandedError::Generate(e)
    }
}

/// What a sink is told before the first band.
#[derive(Clone, Debug)]
pub struct BandedInfo {
    pub width: usize,
    pub height: usize,
    pub band_rows: usize,
    pub bands: usize,
    pub want_extras: bool,
    /// The wind product (east, north m/s interleaved, `wind_height x wind_width x 2`), small whatever the map's width.
    pub wind: Option<(Vec<f32>, usize, usize)>,
}

/// A finished band of pass C: rows `r0 .. r0 + rows`, north to south, north-up.
pub struct BandOut<'a> {
    pub index: usize,
    pub r0: usize,
    pub rows: usize,
    pub width: usize,
    pub coverage: &'a [f32],
    pub kind: Option<&'a [f32]>,
    pub high_share: Option<&'a [f32]>,
}

/// What the run produced besides the bands.
#[derive(Clone, Debug)]
pub struct BandedDone {
    pub meta: GenMeta,
    /// The smoothed per-row thresholds.
    pub thresholds: Vec<f64>,
    pub norm: NormStats,
}

/// A receiver of the finished bands (a file writer, a downsampler, a collector).
pub trait BandSink: Send {
    fn begin(&mut self, _info: &BandedInfo) -> Result<(), String> {
        Ok(())
    }
    /// Bands arrive north to south.
    fn band(&mut self, out: &BandOut<'_>) -> Result<(), String>;
    fn finish(&mut self, _done: &BandedDone) -> Result<(), String> {
        Ok(())
    }
}

/// Per-row float64 moments of the five fields of pass A, one band: `(sum, sum of squares)` per row for `s`, the
/// generic and street fBm and the closed- and open-cell sums (the last three only with the regime texture).
#[derive(Clone, Debug, Default)]
pub struct BandMoments {
    pub s: Vec<(f64, f64)>,
    pub generic: Vec<(f64, f64)>,
    pub street: Vec<(f64, f64)>,
    pub closed: Vec<(f64, f64)>,
    pub open: Vec<(f64, f64)>,
}

/// The result of pass B for one band.
#[derive(Clone, Debug)]
pub struct BandThresholds {
    /// The row thresholds before the smoothing.
    pub thr_raw: Vec<f64>,
    /// The row sensitivity (with the surface terms; empty otherwise).
    pub sensitivity: Vec<f64>,
    /// Range of the surface anomaly over the band (NaN without the surface terms).
    pub anomaly_min: f32,
    pub anomaly_max: f32,
}

// ---------------------------------------------------------------------------------------------------------
// The setup

/// The immutable state of one banded run: the plans, the coarse back-trace and surface terms, the wind, the
/// per-row tables. Everything a band needs, nothing of size `width x height` (the sub-texel array form excepted).
pub struct Banded<'a> {
    pub body: &'a BodyInputs,
    pub params: GenParams,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub band_rows: usize,
    pub want_extras: bool,
    surface: Option<&'a Surface>,
    plans: GeneratePlans,
    noise: SubNoise,
    noise_scale: f32,
    noise_seed: u64,
    phi_h: f64,
    lon: Vec<f64>,
    lat: Vec<f64>,
    target_zonal: Vec<f64>,
    itcz: f64,
    /// The back-trace on its coarse grid: as f32 when it is the map's own grid, widened (flat) to upsample otherwise.
    back_same_grid: Option<Vec<Vec3<f32>>>,
    back_coarse_flat: Vec<f64>,
    back_coarse_dims: (usize, usize),
    /// The surface terms on their own 1024-wide grid (or the map's, when narrower): anomaly, continentality, deck.
    coarse_terms: Option<CoarseTerms>,
    wind: Option<(Vec<f32>, usize, usize)>,
    fmax: f64,
}

struct CoarseTerms {
    h: usize,
    w: usize,
    /// The three fields the pipeline reads, widened once for the bilinear sampler (`None` when the grid is the map's).
    anomaly: Vec<f64>,
    continentality: Vec<f64>,
    deck_strength: Vec<f64>,
    same: Option<SurfaceTerms>,
    itcz_shift_deg: f64,
}

/// The plans of `GeneratePlans::from_seed` without the sub-texel array (the banded engine draws it per band).
pub fn plans_without_subtexel(body: &BodyInputs, seed: u64, width: usize, params: &GenParams) -> GeneratePlans {
    let ladder = fine_ladder(width, params);
    let plan = |offset: u64, frequency: f64, octaves: usize, gain: f64, billow: bool| {
        FbmPlan::from_seed(seed.wrapping_add(offset), frequency, octaves, gain, DEFAULT_LACUNARITY, billow)
    };
    let moisture = plan(SEED_MOISTURE, MOISTURE_FREQUENCY, MOISTURE_OCTAVES, DEFAULT_GAIN, false);
    let mid = plan(SEED_MID, params.mid_frequency, MID_OCTAVES, DEFAULT_GAIN, false);
    let texture = if params.regime_texture {
        crate::generate::TexturePlans::Regime(RegimeTexturePlans::from_seed(
            body.radius_km,
            texture_fmax(ladder.fmax, params),
            seed,
            params.fine_frequency,
            params.fine_gain,
        ))
    } else {
        crate::generate::TexturePlans::Plain(plan(SEED_FINE, ladder.frequency, ladder.octaves, params.fine_gain, params.fine_billow))
    };
    let streak = plan(SEED_STREAK, ladder.frequency, ladder.octaves, STREAK_GAIN, false);
    let flow = if params.isotropic { isotropic_flow(body) } else { build_flow(body, seed) };
    GeneratePlans { moisture, mid, texture, streak, subtexel: None, flow }
}

/// Texels per band of at most `budget` texels (at least one row, at most the map).
pub fn auto_band_rows(width: usize, height: usize, budget_texels: usize) -> usize {
    (budget_texels / width.max(1)).clamp(1, height.max(1))
}

impl<'a> Banded<'a> {
    /// Sets one run up: the plans from `seed` (the noise per `noise`: [`SubNoise::Counter`] draws it per band),
    /// the coarse back-trace, the wind and the coarse surface terms.
    ///
    /// # Errors
    /// [`GenerateError::NotBakeable`] when the body's clouds are not water and `params.allow_any_regime` is off.
    ///
    /// # Panics
    /// If `width < 2`, the map has more than `u32::MAX` texels, an array noise has the wrong length, or
    /// `band_rows == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        body: &'a BodyInputs,
        seed: u64,
        width: usize,
        params: &GenParams,
        surface: Option<&'a Surface>,
        noise: SubNoise,
        band_rows: usize,
        want_extras: bool,
    ) -> Result<Banded<'a>, GenerateError> {
        if !crate::climate::is_bakeable(body) && !params.allow_any_regime {
            return Err(GenerateError::NotBakeable { cloud_regime: body.cloud_regime.to_string() });
        }
        assert!(width >= 2, "a map needs at least 2 columns, got {width}");
        assert!(band_rows >= 1, "a band holds at least one row");
        let height = width / 2;
        assert!((height as u64) * (width as u64) <= u64::from(u32::MAX), "texel indices of the map must fit a u32");
        let phi_h = hadley_edge_deg(body);
        let plans = plans_without_subtexel(body, seed, width, params);
        let divisor = subtexel_scale(body, width, params);
        let noise = match (divisor, noise) {
            (None, _) => SubNoise::Off,
            (Some(_), n) => n,
        };
        if let SubNoise::Array(a) = &noise {
            assert_eq!(a.len(), height * width, "the sub-texel plan must hold height x width values");
        }
        let (lon, lat) = grid_lonlat(width, height);
        let flow = &plans.flow;

        // Where the air came from, on the coarse grid.
        let (mut back_same_grid, mut back_coarse_flat, mut back_coarse_dims) = (None, Vec::new(), (0, 0));
        if !params.isotropic {
            let sw = params.structure_width.min(width);
            assert!(sw >= 2, "structure_width must be at least 2, got {}", params.structure_width);
            let coarse = crate::sphere::grid_dirs::<f64>(sw, sw / 2);
            let back_c = flow.backtrace_default(&coarse);
            if sw == width {
                back_same_grid = Some(back_c);
            } else {
                back_coarse_flat = back_c.iter().flat_map(|v| [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]).collect();
                back_coarse_dims = (sw / 2, sw);
            }
        }

        // The wind and the surface terms (on their own coarse grid).
        let use_surface = surface.is_some() && params.surface_terms && !params.isotropic;
        let wind_width = (params.structure_width * 2).min(width);
        let wind_height = wind_width / 2;
        let wind_field: Option<Vec<f32>> = (use_surface || want_extras).then(|| wind_map(flow, body, wind_width));
        let coarse_terms = match (use_surface, surface) {
            (true, Some(surf)) => {
                let uv = wind_field.as_deref().expect("the wind is made when the surface terms run");
                let grid = WindGrid { uv, h: wind_height, w: wind_width };
                let tw = SURFACE_TERM_WIDTH.min(width);
                // At the coarse width the call returns the coarse arrays themselves (no upsampling).
                let terms = surface_terms(surf, tw, phi_h, body.radius_km, spin_sign(body), Some(&grid), SURFACE_TERM_WIDTH);
                let itcz_shift_deg = terms.itcz_shift_deg;
                if tw == width {
                    Some(CoarseTerms { h: terms.h, w: terms.w, anomaly: vec![], continentality: vec![], deck_strength: vec![], same: Some(terms), itcz_shift_deg })
                } else {
                    let widen = |a: &[f32]| -> Vec<f64> { a.iter().map(|&v| f64::from(v)).collect() };
                    Some(CoarseTerms {
                        h: terms.h,
                        w: terms.w,
                        anomaly: widen(&terms.anomaly),
                        continentality: widen(&terms.continentality),
                        deck_strength: widen(&terms.deck_strength),
                        same: None,
                        itcz_shift_deg,
                    })
                }
            }
            _ => None,
        };
        let itcz = coarse_terms.as_ref().map_or(0.0, |t| t.itcz_shift_deg);
        let rad2deg = 180.0 / PI;
        let target_zonal: Vec<f64> = lat.iter().map(|&la| crate::climate::target_zonal_coverage(la * rad2deg, phi_h, itcz)).collect();
        let ladder = fine_ladder(width, params);
        let wind = want_extras.then(|| (wind_field.clone().expect("the wind is made for the extras"), wind_height, wind_width));
        Ok(Banded {
            body,
            params: *params,
            width,
            height,
            seed,
            band_rows: band_rows.min(height),
            want_extras,
            surface,
            plans,
            noise,
            noise_scale: divisor.map_or(0.0, subtexel_noise_scale),
            noise_seed: seed.wrapping_add(SEED_SUBTEXEL),
            phi_h,
            lon,
            lat,
            target_zonal,
            itcz,
            back_same_grid,
            back_coarse_flat,
            back_coarse_dims,
            coarse_terms,
            wind,
            fmax: ladder.fmax,
        })
    }

    /// Number of bands.
    pub fn bands(&self) -> usize {
        self.height.div_ceil(self.band_rows)
    }

    /// Rows `r0 .. r1` of band `i`.
    pub fn band_range(&self, i: usize) -> (usize, usize) {
        let r0 = i * self.band_rows;
        (r0, (r0 + self.band_rows).min(self.height))
    }

    /// What a sink is told first.
    pub fn info(&self) -> BandedInfo {
        BandedInfo {
            width: self.width,
            height: self.height,
            band_rows: self.band_rows,
            bands: self.bands(),
            want_extras: self.want_extras,
            wind: self.wind.clone(),
        }
    }

    pub fn uses_surface(&self) -> bool {
        self.coarse_terms.is_some()
    }

    /// The plans of the run (the sub-texel field is not among them: see [`Banded::noise`]).
    pub fn plans(&self) -> &GeneratePlans {
        &self.plans
    }

    /// The Hadley edge of the body, degrees.
    pub fn hadley_deg(&self) -> f64 {
        self.phi_h
    }

    /// The latitude of every row, radians (north-up).
    pub fn lat_rows(&self) -> &[f64] {
        &self.lat
    }

    /// The zonal target of every row (the ITCZ shift of the surface terms included).
    pub fn target_zonal(&self) -> &[f64] {
        &self.target_zonal
    }

    /// The ITCZ shift of the surface terms, degrees (0 without them).
    pub fn itcz_deg(&self) -> f64 {
        self.itcz
    }

    /// Where the sub-texel noise comes from, with the seed and the scale of the counter-based form.
    pub fn noise(&self) -> (&SubNoise, u64, f32) {
        (&self.noise, self.noise_seed, self.noise_scale)
    }

    /// `fmax` the regime texture is anchored at.
    pub fn texture_fmax(&self) -> f64 {
        texture_fmax(self.fmax, &self.params)
    }

    // -----------------------------------------------------------------------------------------------------
    // Per-band fields

    /// Unit directions of the texel centers of rows `r0 .. r1` in float64.
    fn dirs64(&self, r0: usize, r1: usize) -> Vec<Vec3<f64>> {
        let width = self.width;
        let mut out = vec![[0.0_f64; 3]; (r1 - r0) * width];
        crate::par::for_each_chunk_mut(&mut out, width, |r, row| {
            let la = self.lat[r0 + r];
            for (slot, &lo) in row.iter_mut().zip(&self.lon) {
                *slot = lonlat_to_dir::<f64>(lo, la);
            }
        });
        out
    }

    /// `p` and `back` of the band (f32).
    fn geometry(&self, r0: usize, r1: usize) -> (Vec<Vec3<f32>>, Vec<Vec3<f32>>) {
        let width = self.width;
        let dirs = self.dirs64(r0, r1);
        let p: Vec<Vec3<f32>> = crate::par::map(&dirs, |d| [d[0] as f32, d[1] as f32, d[2] as f32]);
        let back = if self.params.isotropic {
            p.clone()
        } else if let Some(same) = &self.back_same_grid {
            same[r0 * width..r1 * width].to_vec()
        } else {
            let (hc, wc) = self.back_coarse_dims;
            let sampled = sample_equirect(&self.back_coarse_flat, hc, wc, 3, &dirs);
            crate::par::map_index(sampled.len() / 3, |i| {
                let s = &sampled[3 * i..3 * i + 3];
                let n = normalize([s[0], s[1], s[2]]);
                [n[0] as f32, n[1] as f32, n[2] as f32]
            })
        };
        (p, back)
    }

    /// The raw synoptic field of the band, with what the later stages reuse.
    fn core(&self, r0: usize, r1: usize) -> Core {
        let prm = &self.params;
        let n = (r1 - r0) * self.width;
        let (p, back) = self.geometry(r0, r1);
        let lat_back: Vec<f32> = crate::par::map(&back, |b| clip(b[2], -1.0, 1.0).asin());
        let moisture_noise = fbm(&back, &self.plans.moisture);
        let profile = moisture_profile(&lat_back, self.phi_h);
        let (a_moisture, k_noise) = (prm.a_moisture as f32, prm.moisture_noise as f32);
        let mut s: Vec<f32> = crate::par::map_index(n, |i| a_moisture * (profile[i] + k_noise * moisture_noise[i]));
        drop(moisture_noise);
        drop(profile);
        let ascent = self.plans.flow.ascent(&p);
        let a_ascent = prm.a_ascent as f32;
        crate::par::for_each_chunk_mut(&mut s, crate::par::BLOCK, |b, chunk| {
            for (v, a) in chunk.iter_mut().zip(&ascent[b * crate::par::BLOCK..]) {
                *v += a_ascent * *a;
            }
        });
        let mid_noise = fbm(&along(&p, &back, prm.advect_mid), &self.plans.mid);
        let a_mid = prm.a_mid as f32;
        crate::par::for_each_chunk_mut(&mut s, crate::par::BLOCK, |b, chunk| {
            for (v, m) in chunk.iter_mut().zip(&mid_noise[b * crate::par::BLOCK..]) {
                *v += a_mid * *m;
            }
        });
        Core { p, back, lat_back, ascent, s }
    }

    /// The surface terms of the band's rows (`anomaly`, `continentality`, `deck_strength`; the other fields empty).
    pub fn band_terms(&self, r0: usize, r1: usize) -> Option<SurfaceTerms> {
        let ct = self.coarse_terms.as_ref()?;
        let (width, rows) = (self.width, r1 - r0);
        if let Some(t) = &ct.same {
            let span = r0 * width..r1 * width;
            return Some(SurfaceTerms {
                anomaly: t.anomaly[span.clone()].to_vec(),
                land: vec![],
                deck: vec![],
                orographic: vec![],
                continentality: t.continentality[span.clone()].to_vec(),
                deck_strength: t.deck_strength[span].to_vec(),
                itcz_shift_deg: ct.itcz_shift_deg,
                h: rows,
                w: width,
            });
        }
        let dirs = self.dirs64(r0, r1);
        let sample = |img: &[f64]| -> Vec<f32> { sample_equirect(img, ct.h, ct.w, 1, &dirs).into_iter().map(|v| v as f32).collect() };
        Some(SurfaceTerms {
            anomaly: sample(&ct.anomaly),
            land: vec![],
            deck: vec![],
            orographic: vec![],
            continentality: sample(&ct.continentality),
            deck_strength: sample(&ct.deck_strength),
            itcz_shift_deg: ct.itcz_shift_deg,
            h: rows,
            w: width,
        })
    }

    /// The sub-texel field of the band.
    fn band_noise(&self, r0: usize, r1: usize) -> Option<Vec<f32>> {
        let (width, n) = (self.width, (r1 - r0) * self.width);
        match &self.noise {
            SubNoise::Off => None,
            SubNoise::Array(a) => Some(a[r0 * width..r1 * width].to_vec()),
            SubNoise::Counter => {
                let mut out = vec![0.0_f32; n];
                fill_subtexel(self.noise_seed, (r0 * width) as u32, self.noise_scale, &mut out);
                Some(out)
            }
        }
    }

    // -----------------------------------------------------------------------------------------------------
    // Pass A

    /// Pass A of band `i`: the per-row moments of the raw fields.
    pub fn pass_a(&self, i: usize) -> BandMoments {
        let (r0, r1) = self.band_range(i);
        let width = self.width;
        let core = self.core(r0, r1);
        let mut m = BandMoments { s: row_moments(&core.s, width), ..BandMoments::default() };
        if let crate::generate::TexturePlans::Regime(tp) = &self.plans.texture {
            let fmax = texture_fmax(self.fmax, &self.params);
            let used = crate::texture::used_cells(self.body.radius_km, fmax);
            let generic = fbm(&mixed_dirs(&core.p, &core.back, 0.75, 0.25), &tp.generic);
            m.generic = row_moments(&generic, width);
            drop(generic);
            let street = fbm(&mixed_dirs(&core.p, &core.back, 0.1, 0.9), &tp.street);
            m.street = row_moments(&street, width);
            drop(street);
            if !used.is_empty() {
                let q = mixed_dirs(&core.p, &core.back, 0.85, 0.15);
                let n = q.len();
                let (mut closed, mut open) = (vec![0.0_f32; n], vec![0.0_f32; n]);
                for (cell, plan) in used.iter().zip(&tp.cells) {
                    let edge = cell_edges(&q, plan);
                    let w = cell.weight as f32;
                    crate::par::for_each_chunk2_mut(&mut closed, &mut open, crate::par::BLOCK, |b, cl, op| {
                        let edge = &edge[b * crate::par::BLOCK..];
                        for (j, (c, o)) in cl.iter_mut().zip(op.iter_mut()).enumerate() {
                            *c += w * smoothstep(0.03, 0.30, edge[j]);
                            *o += w * (1.0 - smoothstep(0.05, 0.25, edge[j]));
                        }
                    });
                }
                m.closed = row_moments(&closed, width);
                m.open = row_moments(&open, width);
            }
        }
        m
    }

    /// The statistics of the whole map from the moments of every band, in band order (rows in row order: the sums
    /// are float64 and sequential over the rows, so they do not depend on the band height).
    pub fn norm_from(&self, bands: &[BandMoments]) -> NormStats {
        let n = (self.height * self.width) as f64;
        let norm = |pick: fn(&BandMoments) -> &Vec<(f64, f64)>| -> Option<Norm> {
            let (mut sum, mut sq, mut rows) = (0.0_f64, 0.0_f64, 0usize);
            for b in bands {
                for &(s, q) in pick(b) {
                    sum += s;
                    sq += q;
                    rows += 1;
                }
            }
            if rows == 0 {
                return None;
            }
            let mean = sum / n;
            let var = (sq / n - mean * mean).max(0.0);
            Some(Norm { mean: mean as f32, std: var.sqrt() as f32 })
        };
        let s = norm(|b| &b.s).expect("pass A ran over every band");
        let texture = match &self.plans.texture {
            crate::generate::TexturePlans::Regime(_) => {
                let cells = match (norm(|b| &b.closed), norm(|b| &b.open)) {
                    (Some(c), Some(o)) => Some((c, o)),
                    _ => None,
                };
                Some(TextureNorm { generic: norm(|b| &b.generic).expect("generic"), street: norm(|b| &b.street).expect("street"), cells })
            }
            crate::generate::TexturePlans::Plain(_) => None,
        };
        NormStats { s, texture }
    }

    // -----------------------------------------------------------------------------------------------------
    // Passes B and C

    /// The fields of band `i` with the statistics applied: what pass B solves and pass C composes.
    pub fn fields(&self, i: usize, norm: &NormStats) -> Fields {
        let (r0, r1) = self.band_range(i);
        let width = self.width;
        let prm = &self.params;
        let terms = self.band_terms(r0, r1);
        let Core { p, back, lat_back, ascent, mut s } = self.core(r0, r1);
        let (mean, std) = (norm.s.mean, norm.s.std);
        crate::par::for_each_chunk_mut(&mut s, crate::par::BLOCK, |_, chunk| {
            for v in chunk.iter_mut() {
                *v = (*v - mean) / std;
            }
        });
        let weights: Option<TraceWeights> = (prm.regime_texture || self.want_extras)
            .then(|| pipeline_weights(&self.lat[r0..r1], width, &lat_back, self.phi_h, terms.as_ref()));
        let fine: Vec<f32> = match &self.plans.texture {
            crate::generate::TexturePlans::Regime(tp) => {
                let fmax = texture_fmax(self.fmax, prm);
                let tn = norm.texture.as_ref();
                match weights.as_ref().expect("the regime weights are made with the regime texture") {
                    TraceWeights::F64(w) => regime_texture_terms_norm(&p, &back, w, self.body.radius_km, fmax, tp, tn).texture,
                    TraceWeights::F32(w) => regime_texture_terms_norm(&p, &back, w, self.body.radius_km, fmax, tp, tn).texture,
                }
            }
            crate::generate::TexturePlans::Plain(plan) => fbm(&along(&p, &back, prm.advect_fine), plan),
        };
        let streak = fbm(&back, &self.plans.streak);
        drop((p, back, lat_back));
        let sub = self.band_noise(r0, r1);
        Fields { r0, rows: r1 - r0, s, fine, streak, sub, ascent, weights, terms }
    }

    /// The local target, the per-row target and the offset base of a band with the surface terms.
    fn local_and_target(&self, f: &Fields) -> Option<(Vec<f64>, Vec<f64>)> {
        let t = f.terms.as_ref()?;
        let (width, rows) = (self.width, f.rows);
        let tz = &self.target_zonal[f.r0..f.r0 + rows];
        let local: Vec<f64> = crate::par::map_index(rows * width, |i| local_target(tz[i / width], t.anomaly[i]));
        let target: Vec<f64> = (0..rows).map(|r| pairwise_sum(&local[r * width..(r + 1) * width]) / width as f64).collect();
        Some((local, target))
    }

    /// Pass B of band `i`: the thresholds of its rows.
    pub fn pass_b(&self, i: usize, norm: &NormStats) -> BandThresholds {
        let f = self.fields(i, norm);
        self.solve(&f)
    }

    /// The solve of pass B on the fields of a band.
    pub fn solve(&self, f: &Fields) -> BandThresholds {
        let (width, rows) = (self.width, f.rows);
        let streak = Streak::Array(&f.streak[..]);
        let sub = f.sub.as_deref();
        let none = Offset::<f32>::Scalar(0.0);
        match self.local_and_target(f) {
            Some((local, target)) => {
                let thr0 = solve_thresholds(&f.s, &f.fine, rows, width, &target, &self.params, streak, sub, none);
                let sens = row_sensitivity(&f.s, &f.fine, rows, width, &thr0, &self.params, streak, sub);
                let offset: Vec<f32> = crate::par::map_index(rows * width, |i| (-(local[i] - target[i / width]) / sens[i / width]) as f32);
                let thr_raw = solve_thresholds(&f.s, &f.fine, rows, width, &target, &self.params, streak, sub, Offset::Full(&offset[..]));
                let an = &f.terms.as_ref().expect("terms").anomaly;
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for &v in an {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                BandThresholds { thr_raw, sensitivity: sens, anomaly_min: lo, anomaly_max: hi }
            }
            None => {
                let tz = &self.target_zonal[f.r0..f.r0 + rows];
                let thr_raw = solve_thresholds(&f.s, &f.fine, rows, width, tz, &self.params, streak, sub, none);
                BandThresholds { thr_raw, sensitivity: vec![], anomaly_min: f32::NAN, anomaly_max: f32::NAN }
            }
        }
    }

    /// Pass C of band `i`: compose with the final thresholds `thr` (the smoothed vector of all rows) and the row
    /// sensitivities `sens` of pass B (all rows; empty without the surface terms).
    pub fn pass_c(&self, i: usize, norm: &NormStats, thr: &[f64], sens: &[f64]) -> BandResult {
        let f = self.fields(i, norm);
        self.compose_band(&f, thr, sens)
    }

    /// The composition of pass C on the fields of a band.
    pub fn compose_band(&self, f: &Fields, thr: &[f64], sens: &[f64]) -> BandResult {
        let (width, rows, r0) = (self.width, f.rows, f.r0);
        let prm = &self.params;
        let streak = Streak::Array(&f.streak[..]);
        let sub = f.sub.as_deref();
        let thr_rows = &thr[r0..r0 + rows];
        // t_px = thr[:, None] + offset: a column without the surface terms, a field with them.
        let t_px: Option<Vec<f64>> = self.local_and_target(f).map(|(local, target)| {
            let sens = &sens[r0..r0 + rows];
            crate::par::map_index(rows * width, |i| {
                let offset = (-(local[i] - target[i / width]) / sens[i / width]) as f32;
                thr_rows[i / width] + f64::from(offset)
            })
        });
        let threshold = match &t_px {
            Some(t) => Threshold::Full(&t[..]),
            None => Threshold::Column(thr_rows),
        };
        let coverage: Vec<f32> = compose(&f.s, &f.fine, rows, width, threshold, prm, streak, sub).to_f32();
        let (kind, high) = if self.want_extras {
            let t_at = |i: usize| t_px.as_ref().map_or(thr_rows[i / width], |t| t[i]);
            let shield: Vec<f64> =
                crate::par::map_index(rows * width, |i| clip(0.5 + (f64::from(f.s[i]) - t_at(i)) / prm.shield_ramp, 0.0, 1.0));
            let itcz_rad = self.itcz.to_radians();
            let lat_conv: Vec<f64> = self.lat[r0..r0 + rows].iter().map(|&la| la - itcz_rad).collect();
            let w = f.weights.as_ref().expect("the regime weights are made for the extras");
            let kind = match w {
                TraceWeights::F64(w) => type_map::<f64, f64>(&shield, &w.closed, &w.open, &w.street, &lat_conv, width, self.phi_h),
                TraceWeights::F32(w) => type_map::<f64, f32>(&shield, &w.closed, &w.open, &w.street, &lat_conv, width, self.phi_h),
            };
            let share = high_share::<f64, f32, f32>(&shield, &f.ascent, &lat_conv, width, self.phi_h, &coverage);
            (Some(kind), Some(share))
        } else {
            (None, None)
        };
        BandResult { index: r0 / self.band_rows, r0, rows, width, coverage, kind, high_share: high }
    }

    /// `gaussian_filter1d` of the row thresholds over `threshold_smooth_deg` degrees of latitude.
    pub fn smooth(&self, thr_raw: &[f64]) -> Vec<f64> {
        smooth_thresholds(thr_raw, self.height, self.params.threshold_smooth_deg)
    }

    /// The sidecar, from the anomaly range of pass B.
    pub fn meta(&self, anomaly_range: Option<[f32; 2]>) -> GenMeta {
        let flow: &Flow = &self.plans.flow;
        let ladder = fine_ladder(self.width, &self.params);
        GenMeta {
            seed: self.seed,
            width: self.width,
            height: self.height,
            orientation: ORIENTATION,
            hadley_edge_deg: self.phi_h,
            fine_octaves: ladder.octaves,
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
            surface: match (self.uses_surface(), self.surface, anomaly_range) {
                (true, Some(surf), Some(r)) => Some(crate::generate::SurfaceMeta {
                    source: surf.source.clone(),
                    land_fraction: crate::generate::py_round(area_mean(&surf.land, surf.h, surf.w), 4),
                    itcz_shift_deg: crate::generate::py_round(self.itcz, 2),
                    anomaly_range: [crate::generate::py_round(f64::from(r[0]), 3), crate::generate::py_round(f64::from(r[1]), 3)],
                }),
                _ => None,
            },
        }
    }

    /// The whole run on the calling thread (each band is computed with every core of the `par` pool): the three
    /// passes, the smoothing, and every finished band handed to `sink`. `progress` is called when a pass starts
    /// (`band == 0`) and after each band; returning `false` cancels the run.
    pub fn run(&self, sink: &mut dyn BandSink, progress: &mut dyn FnMut(Progress) -> bool) -> Result<BandedDone, BandedError> {
        let bands = self.bands();
        let mut tick = |pass: Pass, band: usize| -> Result<(), BandedError> {
            if progress(Progress { pass, band, bands }) {
                Ok(())
            } else {
                Err(BandedError::Cancelled)
            }
        };
        tick(Pass::A, 0)?;
        let mut moments = Vec::with_capacity(bands);
        for i in 0..bands {
            moments.push(self.pass_a(i));
            tick(Pass::A, i + 1)?;
        }
        let norm = self.norm_from(&moments);
        drop(moments);
        tick(Pass::B, 0)?;
        let (mut thr_raw, mut sens) = (Vec::with_capacity(self.height), Vec::with_capacity(self.height));
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for i in 0..bands {
            let b = self.pass_b(i, &norm);
            thr_raw.extend_from_slice(&b.thr_raw);
            sens.extend_from_slice(&b.sensitivity);
            lo = lo.min(b.anomaly_min);
            hi = hi.max(b.anomaly_max);
            tick(Pass::B, i + 1)?;
        }
        let thr = self.smooth(&thr_raw);
        sink.begin(&self.info()).map_err(BandedError::Sink)?;
        tick(Pass::C, 0)?;
        for i in 0..bands {
            let r = self.pass_c(i, &norm, &thr, &sens);
            sink.band(&r.as_out()).map_err(BandedError::Sink)?;
            tick(Pass::C, i + 1)?;
        }
        let meta = self.meta(self.uses_surface().then_some([lo, hi]));
        let done = BandedDone { meta, thresholds: thr, norm };
        sink.finish(&done).map_err(BandedError::Sink)?;
        Ok(done)
    }
}


/// Smooths the row thresholds as `generate` does: `gaussian_filter1d(thr_raw, sigma = threshold_smooth_deg / (180 /
/// height), mode='nearest', truncate=4)`.
pub fn smooth_thresholds(thr_raw: &[f64], height: usize, threshold_smooth_deg: f64) -> Vec<f64> {
    let sigma = threshold_smooth_deg / (180.0 / height as f64);
    ndimage::gaussian_filter1d(thr_raw, height, 1, 0, sigma, Mode::Nearest, 4.0)
}

/// What one band is computed from.
struct Core {
    p: Vec<Vec3<f32>>,
    back: Vec<Vec3<f32>>,
    lat_back: Vec<f32>,
    ascent: Vec<f32>,
    /// The raw synoptic field (standardized by the caller).
    s: Vec<f32>,
}

/// The fields of one band with the statistics applied: the input of the threshold solve and of the composition.
pub struct Fields {
    pub r0: usize,
    pub rows: usize,
    pub s: Vec<f32>,
    pub fine: Vec<f32>,
    pub streak: Vec<f32>,
    pub sub: Option<Vec<f32>>,
    ascent: Vec<f32>,
    weights: Option<TraceWeights>,
    terms: Option<SurfaceTerms>,
}

/// A band composed (the owned form of [`BandOut`]).
pub struct BandResult {
    pub index: usize,
    pub r0: usize,
    pub rows: usize,
    pub width: usize,
    pub coverage: Vec<f32>,
    pub kind: Option<Vec<f32>>,
    pub high_share: Option<Vec<f32>>,
}

impl BandResult {
    pub fn as_out(&self) -> BandOut<'_> {
        BandOut {
            index: self.index,
            r0: self.r0,
            rows: self.rows,
            width: self.width,
            coverage: &self.coverage,
            kind: self.kind.as_deref(),
            high_share: self.high_share.as_deref(),
        }
    }
}

/// `(sum, sum of squares)` of every row of a band in float64: a row is one fixed block, summed left to right.
pub fn row_moments(field: &[f32], width: usize) -> Vec<(f64, f64)> {
    crate::par::map_rows(field.len() / width, |r| {
        let (mut s, mut q) = (0.0_f64, 0.0_f64);
        for &v in &field[r * width..(r + 1) * width] {
            let v = f64::from(v);
            s += v;
            q += v * v;
        }
        (s, q)
    })
}

