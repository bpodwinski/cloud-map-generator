//! The banded CPU engine: one map made as bands of whole rows, each band by a different WebAssembly instance.
//!
//! `cloud_map_generator_core::generate::generate_with_plans` makes the map in one piece, because the Python it ports does.
//! A page without `SharedArrayBuffer` cannot share one instance between threads, so the map is cut in bands of
//! rows, one per Web Worker (each worker has its own instance and its own copy of everything below). Every stage
//! of the generator is a pure function of the texel's direction on the sphere and of the request's random plans,
//! except five global couplings, which are the barriers of this module (`docs/large-maps.html`, the three-pass
//! design, specialised to what a map of at most 4096 texels wide needs):
//!
//! ```text
//! pass A  prepare()   every band: back-trace, the synoptic field s (raw), the four raw regime-texture fields
//!                     -> the mean and the squared deviations of s and of the four fields (a [`Moments`] each)
//! barrier 1 (caller)  merge the moments (Chan's formula), in band order: the standardization constants
//! pass B  solve()     every band: standardize s and the texture, the per-row threshold bisection
//!                     -> thr_raw of its rows
//! barrier 2 (caller)  concatenate thr_raw, smooth it over 2 degrees of latitude ([`smooth_thresholds`])
//! pass C  compose()   every band: coverage, shield, type and high-share maps with the smoothed thresholds
//! ```
//!
//! A band keeps its fields between the passes (a worker never gives them away), so the only things that cross
//! between instances are the moments, the threshold vector and the finished rows. The map equals the one-piece
//! engine to float rounding: the standardization constants are merged in `f64` here and summed pairwise in `f32`
//! there, which moves `s` by about 1e-7 relative; the tests below measure it.
//!
//! Not banded: the surface terms (`options.surface = synthetic`) couple the whole map twice more (the row
//! sensitivity and a second solve) and need the whole wind grid, so that case runs in one instance
//! ([`generate_whole`]); it is limited to 1024 wide, where that costs a few seconds.

use cloud_map_generator_core::climate::{hadley_edge_deg, target_zonal_coverage, BodyInputs};
use cloud_map_generator_core::flow::{build_flow, Flow};
use cloud_map_generator_core::generate::{
    along, compose, fine_ladder, isotropic_flow, moisture_profile, solve_thresholds, subtexel_scale, texture_fmax, GeneratePlans, GenParams, Offset,
    Streak, TexturePlans, Threshold, MID_OCTAVES, MOISTURE_FREQUENCY, MOISTURE_OCTAVES, SEED_FINE, SEED_MID, SEED_MOISTURE, SEED_STREAK,
    SEED_SUBTEXEL, STREAK_GAIN,
};
use cloud_map_generator_core::matching::ndimage::{self, Mode};
use cloud_map_generator_core::noise::{cell_edges, fbm, FbmPlan, DEFAULT_GAIN, DEFAULT_LACUNARITY};
use cloud_map_generator_core::products::{high_share, type_map};
use cloud_map_generator_core::rng::Rng;
use cloud_map_generator_core::sphere::{grid_lonlat, lonlat_to_dir, normalize, sample_equirect, Vec3};
use cloud_map_generator_core::texture::{regime_weights, smoothstep, used_cells, RegimeWeights, RegimeTexturePlans};
use std::f64::consts::PI;
use std::rc::Rc;

/// How many fields have moments: `s`, then the generic fBm, the street fBm, the closed and the open cell fields.
pub const FIELDS: usize = 5;
/// Numbers per field in the flat moments array: count, mean, sum of squared deviations.
pub const MOMENT_STRIDE: usize = 3;

/// What the worker of one band derives for itself, deterministically, from the request alone (so the workers never
/// exchange plans): the random plans (everything `GeneratePlans::from_seed` draws except the sub-texel noise, which is a
/// map-sized array of which a band draws its own rows) and the coarse back-trace of the rows its band reads.
///
/// The back-trace is the one stage that dominates a map's cost in WebAssembly (about 6 million steps of trigonometry and
/// exponentials at the 512-wide coarse grid, which `wasm32-unknown-unknown` computes with a software libm), and every
/// coarse point is independent of the others, so a band traces only the coarse rows its own rows interpolate between.
pub struct Setup {
    pub body: BodyInputs,
    pub params: GenParams,
    pub seed: u64,
    pub width: usize,
    pub height: usize,
    pub phi_h: f64,
    /// The rows of the band this setup is for.
    pub r0: usize,
    pub r1: usize,
    pub plans: GeneratePlans,
    /// The coarse back-trace when it is the map's own grid (width at most `structure_width`): the rows `r0..r1` only.
    coarse_f32: Option<Vec<Vec3<f32>>>,
    /// Otherwise the whole coarse grid widened to `float64`, with the rows no band row reads left at zero.
    coarse_f64: Option<Vec<Vec3<f64>>>,
    coarse_width: usize,
}

impl Setup {
    /// Draws the plans exactly as [`GeneratePlans::from_seed`] does (the seeds and their order are the same), minus the
    /// sub-texel array, and back-traces the coarse grid.
    pub fn new(body: &BodyInputs, seed: u64, width: usize, params: &GenParams, r0: usize, r1: usize) -> Setup {
        assert!(width >= 2, "a map needs at least 2 columns");
        assert!(r0 < r1 && r1 <= width / 2, "a band is a non-empty run of rows of the map");
        let height = width / 2;
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
        let plans = GeneratePlans { moisture, mid, texture, streak, subtexel: None, flow };

        // Where the air came from, on the coarse grid (the flow is smooth at that scale), for the rows the band reads.
        let coarse_width = params.structure_width.min(width);
        let (mut coarse_f32, mut coarse_f64) = (None, None);
        if !params.isotropic {
            assert!(coarse_width >= 2, "structure_width must be at least 2");
            let coarse_height = coarse_width / 2;
            if coarse_width == width {
                // The coarse grid is the map's own: the band's rows are exactly the rows it reads.
                let grid = band_dirs::<f64>(width, height, r0, r1);
                coarse_f32 = Some(plans.flow.backtrace(&grid, params.backtrace_time, params.backtrace_steps()));
            } else {
                // `sample_equirect` reads rows floor(fy) and floor(fy) + 1 of the coarse grid at fy = (pi/2 - lat) / pi * h - 0.5;
                // one row more on each side is margin against rounding at the edges (a test compares with the one-piece map).
                let (_, lat) = grid_lonlat(width, height);
                let fy = |row: usize| (PI / 2.0 - lat[row]) / PI * coarse_height as f64 - 0.5;
                let last = coarse_height as i64 - 1;
                let lo = (fy(r0).floor() as i64 - 1).clamp(0, last) as usize;
                let hi = (fy(r1 - 1).floor() as i64 + 2).clamp(0, last) as usize;
                let grid = band_dirs::<f64>(coarse_width, coarse_height, lo, hi + 1);
                let traced = plans.flow.backtrace(&grid, params.backtrace_time, params.backtrace_steps());
                let mut full = vec![[0.0f64; 3]; coarse_width * coarse_height];
                for (i, v) in traced.iter().enumerate() {
                    full[lo * coarse_width + i] = [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])];
                }
                coarse_f64 = Some(full);
            }
        }
        Setup { body: body.clone(), params: *params, seed, width, height, phi_h: hadley_edge_deg(body), r0, r1, plans, coarse_f32, coarse_f64, coarse_width }
    }

    /// The flow the map is made with (the page's meta block and the wind map read it).
    pub fn flow(&self) -> &Flow {
        &self.plans.flow
    }
}

/// `grid_dirs` of a `width` x `height` map restricted to rows `r0..r1`: the same `float64` formula, so the same bits.
fn band_dirs<F: num_traits::Float>(width: usize, height: usize, r0: usize, r1: usize) -> Vec<Vec3<F>> {
    let (lon, lat) = grid_lonlat(width, height);
    let c = |x: f64| F::from(x).expect("float conversion");
    let mut out = Vec::with_capacity((r1 - r0) * width);
    for &la in &lat[r0..r1] {
        for &lo in &lon {
            let d = lonlat_to_dir::<f64>(lo, la);
            out.push([c(d[0]), c(d[1]), c(d[2])]);
        }
    }
    out
}

/// `normalize(a ka + b kb)` per direction, as `float32` (`texture::mixed_dirs`, which is private to the core).
fn mixed_dirs(p: &[Vec3<f32>], back: &[Vec3<f32>], ka: f64, kb: f64) -> Vec<Vec3<f32>> {
    let (ka, kb) = (ka as f32, kb as f32);
    p.iter().zip(back).map(|(a, b)| normalize([a[0] * ka + b[0] * kb, a[1] * ka + b[1] * kb, a[2] * ka + b[2] * kb])).collect()
}

/// Count, mean and sum of squared deviations of a field, in `f64`: what barrier 1 merges.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moments {
    pub n: f64,
    pub mean: f64,
    pub m2: f64,
}

impl Moments {
    pub fn of(values: &[f32]) -> Moments {
        if values.is_empty() {
            return Moments::default();
        }
        let n = values.len() as f64;
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let m2 = values.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>();
        Moments { n, mean, m2 }
    }

    /// Chan's parallel merge of two sets of moments.
    pub fn merge(self, other: Moments) -> Moments {
        if self.n == 0.0 {
            return other;
        }
        if other.n == 0.0 {
            return self;
        }
        let n = self.n + other.n;
        let delta = other.mean - self.mean;
        Moments { n, mean: self.mean + delta * other.n / n, m2: self.m2 + other.m2 + delta * delta * self.n * other.n / n }
    }

    /// Population standard deviation (`ddof = 0`).
    pub fn std(&self) -> f64 {
        (self.m2 / self.n).sqrt()
    }
}

/// The flat array one band returns from pass A: [`FIELDS`] x (count, mean, m2).
pub fn moments_to_flat(m: &[Moments; FIELDS]) -> Vec<f64> {
    m.iter().flat_map(|x| [x.n, x.mean, x.m2]).collect()
}

/// Barrier 1: merges the flat moments of every band, in band order, and returns for each field `(mean, std)` flat
/// (`2 x FIELDS` numbers), the constants pass B standardizes with.
pub fn merge_flat(bands: &[Vec<f64>]) -> Vec<f64> {
    let mut total = [Moments::default(); FIELDS];
    for flat in bands {
        for (k, t) in total.iter_mut().enumerate() {
            let m = Moments { n: flat[k * MOMENT_STRIDE], mean: flat[k * MOMENT_STRIDE + 1], m2: flat[k * MOMENT_STRIDE + 2] };
            *t = t.merge(m);
        }
    }
    total.iter().flat_map(|t| if t.n > 0.0 { [t.mean, t.std()] } else { [0.0, 1.0] }).collect()
}

/// numpy's `(x - mean) / divisor` of `texture::standardize`, with the constants given (`float32`).
fn standardize_with(v: &[f32], mean: f64, std: f64) -> Vec<f32> {
    let (mean, sd) = (mean as f32, std as f32);
    let divisor = if sd > 0.0 { sd } else { 1.0 };
    v.iter().map(|&x| (x - mean) / divisor).collect()
}

/// Barrier 2: the row thresholds smoothed in latitude (`generate`'s `gaussian_filter1d` of `thr_raw`).
pub fn smooth_thresholds(thr_raw: &[f64], params: &GenParams) -> Vec<f64> {
    let height = thr_raw.len();
    let sigma = params.threshold_smooth_deg / (180.0 / height as f64);
    ndimage::gaussian_filter1d(thr_raw, height, 1, 0, sigma, Mode::Nearest, 4.0)
}

/// The raw regime-texture fields of a band, before their standardization.
struct RawTexture {
    generic: Vec<f32>,
    street: Vec<f32>,
    /// The weighted cell sums; empty when no cell diameter is used at this width.
    closed: Vec<f32>,
    open: Vec<f32>,
}

/// One band of a map: rows `r0..r1`, and every field of those rows.
pub struct Band {
    setup: Rc<Setup>,
    pub r0: usize,
    pub r1: usize,
    lat_rows: Vec<f64>,
    ascent: Vec<f32>,
    weights: RegimeWeights<f64>,
    s: Vec<f32>,
    raw: Option<RawTexture>,
    fine: Vec<f32>,
    streak: Vec<f32>,
    subtexel: Option<Vec<f32>>,
    thr: Vec<f64>,
}

/// What pass C returns: the rows of the three maps the engine makes.
pub struct BandOutput {
    pub coverage: Vec<f32>,
    pub kind: Vec<f32>,
    pub high_share: Vec<f32>,
}

impl Band {
    /// Pass A: the raw fields of rows `r0..r1`, and their moments.
    pub fn prepare(setup: &Rc<Setup>, r0: usize, r1: usize) -> (Band, [Moments; FIELDS]) {
        assert!(r0 == setup.r0 && r1 == setup.r1, "a band reads the rows its setup traced");
        let (width, height, params) = (setup.width, setup.height, &setup.params);
        let rows = r1 - r0;
        let n = rows * width;
        let p = band_dirs::<f32>(width, height, r0, r1);

        // Where the air at each texel came from.
        let back: Vec<Vec3<f32>> = if params.isotropic {
            p.clone()
        } else if let Some(rows) = &setup.coarse_f32 {
            rows.clone()
        } else {
            let coarse = setup.coarse_f64.as_ref().expect("the coarse back-trace of an anisotropic run");
            let (cw, ch) = (setup.coarse_width, setup.coarse_width / 2);
            let flat: Vec<f64> = coarse.iter().flatten().copied().collect();
            let target = band_dirs::<f64>(width, height, r0, r1);
            sample_equirect::<f64>(&flat, ch, cw, 3, &target)
                .chunks_exact(3)
                .map(|s| {
                    let v = normalize([s[0], s[1], s[2]]);
                    [v[0] as f32, v[1] as f32, v[2] as f32]
                })
                .collect()
        };

        // The synoptic field, before its standardization.
        let lat_back: Vec<f32> = back.iter().map(|b| b[2].max(-1.0).min(1.0).asin()).collect();
        let moisture_noise = fbm(&back, &setup.plans.moisture);
        let profile = moisture_profile(&lat_back, setup.phi_h);
        let (a_moisture, k_noise) = (params.a_moisture as f32, params.moisture_noise as f32);
        let mut s: Vec<f32> = (0..n).map(|i| a_moisture * (profile[i] + k_noise * moisture_noise[i])).collect();
        let ascent = setup.plans.flow.ascent(&p);
        let a_ascent = params.a_ascent as f32;
        for (v, a) in s.iter_mut().zip(&ascent) {
            *v += a_ascent * *a;
        }
        let mid_noise = fbm(&along(&p, &back, params.advect_mid), &setup.plans.mid);
        let a_mid = params.a_mid as f32;
        for (v, m) in s.iter_mut().zip(&mid_noise) {
            *v += a_mid * *m;
        }

        // The regime weights of the rows (the type map reads them even when the texture does not).
        let (_, lat_all) = grid_lonlat(width, height);
        let lat_rows: Vec<f64> = lat_all[r0..r1].to_vec();
        let lat_grid: Vec<f64> = lat_rows.iter().flat_map(|&l| std::iter::repeat_n(l, width)).collect();
        let weights = regime_weights::<f64, f32, f64>(&lat_grid, &lat_back, setup.phi_h);

        // The fine texture: the raw fields of the regime texture (standardized in pass B) or the plain fBm.
        let mut moments = [Moments::default(); FIELDS];
        moments[0] = Moments::of(&s);
        let (raw, fine) = match &setup.plans.texture {
            TexturePlans::Regime(tp) => {
                let fmax = texture_fmax(fine_ladder(width, params).fmax, params);
                let used = used_cells(setup.body.radius_km, fmax);
                assert_eq!(tp.cells.len(), used.len(), "one cell plan per used diameter");
                let generic = fbm(&mixed_dirs(&p, &back, 0.75, 0.25), &tp.generic);
                let street = fbm(&mixed_dirs(&p, &back, 0.1, 0.9), &tp.street);
                let (mut closed, mut open) = (Vec::new(), Vec::new());
                if !used.is_empty() {
                    closed = vec![0.0f32; n];
                    open = vec![0.0f32; n];
                    let q = mixed_dirs(&p, &back, 0.85, 0.15);
                    for (cell, plan) in used.iter().zip(&tp.cells) {
                        let edge = cell_edges(&q, plan);
                        let w = cell.weight as f32;
                        for i in 0..n {
                            closed[i] += w * smoothstep(0.03, 0.30, edge[i]);
                            open[i] += w * (1.0 - smoothstep(0.05, 0.25, edge[i]));
                        }
                    }
                }
                moments[1] = Moments::of(&generic);
                moments[2] = Moments::of(&street);
                moments[3] = Moments::of(&closed);
                moments[4] = Moments::of(&open);
                (Some(RawTexture { generic, street, closed, open }), Vec::new())
            }
            TexturePlans::Plain(plan) => (None, fbm(&along(&p, &back, params.advect_fine), plan)),
        };
        let streak = fbm(&back, &setup.plans.streak);
        let subtexel = subtexel_rows(setup, r0, r1);
        let band = Band { setup: setup.clone(), r0, r1, lat_rows, ascent, weights, s, raw, fine, streak, subtexel, thr: Vec::new() };
        (band, moments)
    }

    /// Pass B: standardizes the fields with the merged constants (`2 x FIELDS` numbers from [`merge_flat`]) and solves
    /// the threshold of each row. Returns `thr_raw` of this band's rows.
    pub fn solve(&mut self, constants: &[f64]) -> Vec<f64> {
        let setup = self.setup.clone();
        let (width, rows) = (setup.width, self.r1 - self.r0);
        let c = |k: usize| (constants[2 * k], constants[2 * k + 1]);
        self.s = standardize_with(&self.s, c(0).0, c(0).1);
        if let Some(raw) = self.raw.take() {
            let generic = standardize_with(&raw.generic, c(1).0, c(1).1);
            let street = standardize_with(&raw.street, c(2).0, c(2).1);
            let cells = (!raw.closed.is_empty()).then(|| (standardize_with(&raw.closed, c(3).0, c(3).1), standardize_with(&raw.open, c(4).0, c(4).1)));
            // The blend runs in the precision of the weights (float64), as `regime_texture_terms` does.
            let w = &self.weights;
            let mut tex: Vec<f64> = generic.iter().map(|&v| f64::from(v)).collect();
            if let Some((closed, open)) = &cells {
                for i in 0..tex.len() {
                    tex[i] = tex[i] * (1.0 - w.closed[i] - w.open[i]) + w.closed[i] * f64::from(closed[i]) + w.open[i] * f64::from(open[i]);
                }
            }
            for i in 0..tex.len() {
                tex[i] = tex[i] * (1.0 - w.street[i]) + w.street[i] * f64::from(street[i]);
            }
            self.fine = tex.iter().map(|&v| v as f32).collect();
        }
        let rad2deg = 180.0 / PI;
        let target: Vec<f64> = self.lat_rows.iter().map(|&la| target_zonal_coverage(la * rad2deg, setup.phi_h, 0.0)).collect();
        solve_thresholds(
            &self.s,
            &self.fine,
            rows,
            width,
            &target,
            &setup.params,
            Streak::Array(&self.streak[..]),
            self.subtexel.as_deref(),
            Offset::<f32>::Scalar(0.0),
        )
    }

    /// Pass C: the coverage, type and high-share rows from the smoothed thresholds of the WHOLE map (`thr`, `height` values).
    pub fn compose(&mut self, thr: &[f64]) -> BandOutput {
        let setup = self.setup.clone();
        let (width, rows) = (setup.width, self.r1 - self.r0);
        let params = &setup.params;
        self.thr = thr[self.r0..self.r1].to_vec();
        let coverage = compose(
            &self.s,
            &self.fine,
            rows,
            width,
            Threshold::Column(&self.thr),
            params,
            Streak::Array(&self.streak[..]),
            self.subtexel.as_deref(),
        )
        .to_f32();
        // The shield (`clip(0.5 + (s - t) / ramp, 0, 1)`), then the products that read it.
        let shield: Vec<f64> =
            (0..rows * width).map(|i| (0.5 + (f64::from(self.s[i]) - self.thr[i / width]) / params.shield_ramp).clamp(0.0, 1.0)).collect();
        let w = &self.weights;
        let kind = type_map::<f64, f64>(&shield, &w.closed, &w.open, &w.street, &self.lat_rows, width, setup.phi_h);
        let share = high_share::<f64, f32, f32>(&shield, &self.ascent, &self.lat_rows, width, setup.phi_h, &coverage);
        BandOutput { coverage, kind, high_share: share }
    }
}

/// Rows `r0..r1` of the sub-texel white noise, `normal / sqrt(n_clouds)`: the generator's stream is sequential, so a
/// band draws and drops the rows above it (a few milliseconds per million texels).
fn subtexel_rows(setup: &Setup, r0: usize, r1: usize) -> Option<Vec<f32>> {
    let scale = subtexel_scale(&setup.body, setup.width, &setup.params)?;
    let mut rng = Rng::new(setup.seed.wrapping_add(SEED_SUBTEXEL));
    for _ in 0..r0 * setup.width {
        rng.normal();
    }
    Some((0..(r1 - r0) * setup.width).map(|_| rng.normal() as f32 / scale).collect())
}

/// The cuts of a map `height` rows tall into `bands` runs of whole rows (the first `height % bands` get one more).
pub fn row_cuts(height: usize, bands: usize) -> Vec<(usize, usize)> {
    let bands = bands.clamp(1, height);
    let (base, extra) = (height / bands, height % bands);
    let mut cuts = Vec::with_capacity(bands);
    let mut r = 0;
    for i in 0..bands {
        let rows = base + usize::from(i < extra);
        cuts.push((r, r + rows));
        r += rows;
    }
    cuts
}

/// The whole banded pipeline in one instance (the passes and barriers run in a loop): what the native tests compare
/// with the one-piece engine, and what a one-worker pool amounts to.
pub fn generate_banded(body: &BodyInputs, seed: u64, width: usize, params: &GenParams, bands: usize) -> BandOutput {
    let cuts = row_cuts(width / 2, bands);
    let (mut parts, mut flats) = (Vec::new(), Vec::new());
    for &(r0, r1) in &cuts {
        let setup = Rc::new(Setup::new(body, seed, width, params, r0, r1));
        let (band, moments) = Band::prepare(&setup, r0, r1);
        flats.push(moments_to_flat(&moments));
        parts.push(band);
    }
    let constants = merge_flat(&flats);
    let thr_raw: Vec<f64> = parts.iter_mut().flat_map(|b| b.solve(&constants)).collect();
    let thr = smooth_thresholds(&thr_raw, params);
    let mut out = BandOutput { coverage: Vec::new(), kind: Vec::new(), high_share: Vec::new() };
    for b in parts.iter_mut() {
        let o = b.compose(&thr);
        out.coverage.extend(o.coverage);
        out.kind.extend(o.kind);
        out.high_share.extend(o.high_share);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cloud_map_generator_core::generate::generate;

    fn max_diff(a: &[f32], b: &[f32]) -> (f64, f64) {
        assert_eq!(a.len(), b.len());
        let (mut max, mut sum) = (0.0f64, 0.0f64);
        for (x, y) in a.iter().zip(b) {
            let d = f64::from((x - y).abs());
            max = max.max(d);
            sum += d;
        }
        (max, sum / a.len() as f64)
    }

    #[test]
    fn row_cuts_cover_the_height_exactly() {
        for (h, b) in [(32, 1), (32, 3), (32, 7), (5, 9), (1024, 6)] {
            let cuts = row_cuts(h, b);
            assert_eq!(cuts.first().unwrap().0, 0);
            assert_eq!(cuts.last().unwrap().1, h);
            assert!(cuts.windows(2).all(|w| w[0].1 == w[1].0 && w[0].0 < w[0].1));
        }
    }

    #[test]
    fn moments_merge_like_one_pass() {
        let v: Vec<f32> = (0..1000).map(|i| ((i * 37 % 101) as f32) * 0.1 - 3.0).collect();
        let whole = Moments::of(&v);
        let merged = v.chunks(137).map(Moments::of).fold(Moments::default(), Moments::merge);
        assert!((whole.mean - merged.mean).abs() < 1e-12 && (whole.m2 - merged.m2).abs() < 1e-9, "{whole:?} {merged:?}");
    }

    /// The gate of the banding: every band count gives the map of the one-piece engine to float rounding. Width 512 is
    /// the back-trace grid itself (no upsampling); 1024 takes the upsampling and the cells of the regime texture.
    #[test]
    fn banded_equals_one_piece_to_float_rounding() {
        let body = BodyInputs::new(Some(0.99727), 6371.0, 9.81, 8.0);
        let params = GenParams::default();
        for width in [512usize, 1024] {
            let reference = generate(&body, 42, width, &params, None, true).unwrap();
            let extras = reference.extras.as_ref().unwrap();
            for bands in [1usize, 3, 8] {
                let out = generate_banded(&body, 42, width, &params, bands);
                let (cmax, cmean) = max_diff(&out.coverage, &reference.coverage);
                let (tmax, _) = max_diff(&out.kind, &extras.kind);
                let (hmax, _) = max_diff(&out.high_share, &extras.high_share);
                eprintln!("width {width} bands {bands}: coverage max {cmax:.2e} mean {cmean:.2e}; type max {tmax:.2e}; high share max {hmax:.2e}");
                assert!(cmax < 1e-3 && cmean < 1e-5, "coverage differs: max {cmax}, mean {cmean} (width {width}, {bands} bands)");
                assert!(tmax < 1e-3 && hmax < 1e-3, "products differ (width {width}, {bands} bands)");
            }
        }
    }

    #[test]
    fn banding_changes_nothing_but_rounding_between_band_counts() {
        let body = BodyInputs::new(Some(0.4), 6371.0, 9.81, 8.0);
        let params = GenParams { isotropic: true, ..GenParams::default() };
        let one = generate_banded(&body, 5, 256, &params, 1);
        let many = generate_banded(&body, 5, 256, &params, 5);
        let (max, _) = max_diff(&one.coverage, &many.coverage);
        assert!(max < 1e-4, "isotropic 256: {max}");
    }
}

