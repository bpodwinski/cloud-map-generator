//! Port of `cloudgen/match.py`: multi-scale statistics match of a generated coverage map to Earth's map
//! (Heeger-Bergen style), plus [`ndimage`], the numpy / scipy primitives this module and `metrics` share.
//!
//! The Python name `match` is a Rust keyword, hence `matching`. Maps are row-major `h x w` slices, north-up
//! (row 0 at the north pole edge), as everywhere in the port. Functions numpy runs in the dtype of their input
//! are generic over `f32` / `f64` and run in that precision; where numpy promotes (a float32 array against a
//! float64 table), the port promotes at the same place.
//!
//! The doc of each function names the Python function it ports. Python's private helpers lose their leading
//! underscore (`_blur` is [`blur`], `_match_by_zone` is [`match_by_zone`]).

use crate::sphere::{grid_lonlat, linspace};
use num_traits::Float;
use std::sync::OnceLock;

/// Width of a latitude zone of Earth's map, degrees (`match.ZONE_DEG`).
pub const ZONE_DEG: f64 = 5.0;
/// Earth rows poleward of this |latitude| are not references: their cloud fraction is contaminated by ice
/// (`match.EARTH_CAP_DEG`).
pub const EARTH_CAP_DEG: f64 = 67.5;
/// Default number of scales of [`match_to_earth`] (the Python default `levels=6`).
pub const DEFAULT_LEVELS: usize = 6;
/// Default blend of [`match_to_earth`] (the Python default `strength=1.0`).
pub const DEFAULT_STRENGTH: f64 = 1.0;
/// Number of quantiles of the match tables.
pub const QUANTILE_COUNT: usize = 1024;

/// `match.QUANTILES = np.linspace(0.0005, 0.9995, 1024)`.
pub fn quantiles() -> &'static [f64] {
    static Q: OnceLock<Vec<f64>> = OnceLock::new();
    Q.get_or_init(|| linspace(0.0005, 0.9995, QUANTILE_COUNT))
}

fn c<F: Float>(x: f64) -> F {
    F::from(x).expect("float conversion")
}

fn f64_of<F: Float>(x: F) -> f64 {
    x.to_f64().expect("float conversion")
}

/// `match._blur`: `gaussian_filter(img, sigma, mode=('nearest', 'wrap'))`. Longitude (columns) wraps, latitude
/// (rows) clamps at the poles; truncate 4.0; the output keeps the dtype of the input.
pub fn blur<F: Float>(img: &[F], h: usize, w: usize, sigma: f64) -> Vec<F> {
    ndimage::gaussian_filter(img, h, w, sigma, [ndimage::Mode::Nearest, ndimage::Mode::Wrap])
}

/// `match.band_layers`: the band-pass layers `G(s_{k-1}) - G(s_k)` (`G(s_-1)` = the image, `G` = [`blur`]) and the
/// low-pass residual `G(s_last)` (the image itself for no sigma). The layers and the residual sum back to the image.
pub fn band_layers<F: Float>(img: &[F], h: usize, w: usize, sigmas: &[f64]) -> (Vec<Vec<F>>, Vec<F>) {
    let mut layers = Vec::with_capacity(sigmas.len());
    let mut prev = img.to_vec();
    for &s in sigmas {
        let cur = blur(img, h, w, s);
        layers.push(prev.iter().zip(&cur).map(|(&a, &b)| a - b).collect());
        prev = cur;
    }
    (layers, prev)
}

/// `match._earth_lat`: the Earth-equivalent |latitude| in degrees of each row, `min(|degrees(lat)| / hadley * 30,
/// EARTH_CAP_DEG - ZONE_DEG / 2)`. All the arithmetic runs in the precision of `F` (`np.degrees` of a float32 array
/// stays float32).
pub fn earth_lat<F: Float>(lat_rad: &[F], hadley_deg: f64) -> Vec<F> {
    let deg_per_rad = c::<F>(180.0) / c::<F>(std::f64::consts::PI);
    let (hadley, thirty, cap) = (c::<F>(hadley_deg), c::<F>(30.0), c::<F>(EARTH_CAP_DEG - ZONE_DEG / 2.0));
    lat_rad
        .iter()
        .map(|&l| {
            let x = (l * deg_per_rad).abs() / hadley;
            let v = x * thirty;
            // np.minimum propagates NaN.
            if v.is_nan() || v < cap {
                v
            } else {
                cap
            }
        })
        .collect()
}

/// `match._zone_index`: `floor(_earth_lat / ZONE_DEG).astype(int)`, the zone number of each row (truncating cast).
pub fn zone_index<F: Float>(lat_rad: &[F], hadley_deg: f64) -> Vec<i64> {
    let zone = c::<F>(ZONE_DEG);
    earth_lat(lat_rad, hadley_deg).into_iter().map(|e| (e / zone).floor().to_i64().unwrap_or(0)).collect()
}

/// `match._match_by_zone`: quantile-map each row of `gen` (`h x w`, dtype `G`) onto Earth's distribution (`earth`,
/// `he x we`, dtype `E`) at its latitude zone, the map itself interpolated linearly between neighbouring zone
/// centers. `lat_g` is the latitude of each row of `gen` (radians, float64), `ze` the zone of each Earth row (`-1`
/// for a row that is not a reference). Returns an `h x w` array of the dtype of `gen`.
///
/// Per zone present in both: `src = quantile(gen rows of the zone, QUANTILES) + k * 1e-9` (strictly increasing) and
/// `dst = quantile(earth rows of the zone, QUANTILES)`; per row, the two nearest zone centers `(z + 0.5) * 5` around
/// the row's Earth-equivalent latitude, `t = clip((el - c_i) / (c_j - c_i), 0, 1)`, `a = interp(row, src_i, dst_i)`,
/// `b` likewise for zone `j` (skipped for `t == 0`), `out = a + t (b - a)`. Panics when gen and Earth share no zone
/// (an IndexError in Python).
#[allow(clippy::too_many_arguments)]
pub fn match_by_zone<G: Float, E: Float>(
    gen: &[G],
    h: usize,
    w: usize,
    earth: &[E],
    he: usize,
    we: usize,
    lat_g: &[f64],
    hadley_deg: f64,
    ze: &[i64],
) -> Vec<G> {
    assert_eq!(gen.len(), h * w, "gen size does not match h x w");
    assert_eq!(earth.len(), he * we, "earth size does not match he x we");
    assert_eq!(lat_g.len(), h);
    assert_eq!(ze.len(), he);
    let zg = zone_index::<f64>(lat_g, hadley_deg);
    // np.unique(zg): the sorted distinct zones.
    let mut distinct = zg.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let zones: Vec<i64> = distinct.into_iter().filter(|z| ze.contains(z)).collect();
    assert!(!zones.is_empty(), "match_by_zone: gen and Earth share no zone");

    let q = quantiles();
    let mut tables: Vec<(Vec<f64>, Vec<f64>)> = Vec::with_capacity(zones.len());
    for &z in &zones {
        let mut gen_vals: Vec<G> = Vec::new();
        for r in (0..h).filter(|&r| zg[r] == z) {
            gen_vals.extend_from_slice(&gen[r * w..(r + 1) * w]);
        }
        let mut earth_vals: Vec<E> = Vec::new();
        for r in (0..he).filter(|&r| ze[r] == z) {
            earth_vals.extend_from_slice(&earth[r * we..(r + 1) * we]);
        }
        let src = ndimage::quantile(&gen_vals, q);
        let dst = ndimage::quantile(&earth_vals, q);
        // Strictly increasing source for interp (flat runs of clipped 0 or 1).
        let src: Vec<f64> = src.iter().enumerate().map(|(k, &s)| s + k as f64 * 1e-9).collect();
        tables.push((src, dst));
    }
    let centers: Vec<f64> = zones.iter().map(|&z| (z as f64 + 0.5) * ZONE_DEG).collect();
    let el = earth_lat::<f64>(lat_g, hadley_deg);
    let nz = zones.len();
    let mut out = vec![G::zero(); h * w];
    for r in 0..h {
        let i = (ndimage::searchsorted_left(&centers, el[r]) as i64 - 1).clamp(0, nz as i64 - 1) as usize;
        let j = (i + 1).min(nz - 1);
        let t = if i == j {
            0.0
        } else {
            let raw = (el[r] - centers[i]) / (centers[j] - centers[i]);
            // np.clip(x, 0, 1) propagates NaN.
            if raw.is_nan() {
                raw
            } else {
                raw.max(0.0).min(1.0)
            }
        };
        let row: Vec<f64> = gen[r * w..(r + 1) * w].iter().map(|&v| f64_of(v)).collect();
        let a: Vec<f64> = row.iter().map(|&x| ndimage::interp(x, &tables[i].0, &tables[i].1)).collect();
        let b: Vec<f64> = if t > 0.0 {
            row.iter().map(|&x| ndimage::interp(x, &tables[j].0, &tables[j].1)).collect()
        } else {
            a.clone()
        };
        for col in 0..w {
            out[r * w + col] = c::<G>(a[col] + t * (b[col] - a[col]));
        }
    }
    out
}

/// `match.match_to_earth`: `cov` (`h x w`) with Earth's per-scale, per-zone statistics. Sigmas `2^k` (`k < levels`) in
/// Earth texels, scaled by the width ratio for the generated map; [`band_layers`] of both (float32), each generated
/// layer quantile-matched per latitude zone to the same Earth layer ([`match_by_zone`], Earth zones from the
/// latitude at hadley 30 with the rows past [`EARTH_CAP_DEG`] excluded), summed with `strength * matched + (1 -
/// strength) * original`, plus the matched residual, clipped to `[0, 1]`, matched once more as a whole to Earth's own
/// map and blended again. `strength` in `[0, 1]` blends the matched layers with the original ones. The result is
/// float32 whatever the input dtypes: both maps are cast to float32 first.
#[allow(clippy::too_many_arguments)]
pub fn match_to_earth<C: Float, E: Float>(
    cov: &[C],
    h: usize,
    w: usize,
    earth: &[E],
    he: usize,
    we: usize,
    hadley_deg: f64,
    levels: usize,
    strength: f64,
) -> Vec<f32> {
    assert_eq!(cov.len(), h * w);
    assert_eq!(earth.len(), he * we);
    let ratio = w as f64 / we as f64;
    let sig_e: Vec<f64> = (0..levels).map(|k| 2.0_f64.powi(k as i32)).collect();
    let sig_g: Vec<f64> = sig_e.iter().map(|s| s * ratio).collect();
    let (_, lat_g) = grid_lonlat(w, h);
    let (_, lat_e) = grid_lonlat(we, he);
    let to_deg = 180.0 / std::f64::consts::PI;
    let ze: Vec<i64> = zone_index::<f64>(&lat_e, 30.0)
        .into_iter()
        .zip(&lat_e)
        .map(|(z, &l)| if (l * to_deg).abs() < EARTH_CAP_DEG { z } else { -1 })
        .collect();

    let cov32: Vec<f32> = cov.iter().map(|&v| v.to_f32().expect("float conversion")).collect();
    let earth32: Vec<f32> = earth.iter().map(|&v| v.to_f32().expect("float conversion")).collect();
    let (layers_g, low_g) = band_layers(&cov32, h, w, &sig_g);
    let (layers_e, low_e) = band_layers(&earth32, he, we, &sig_e);

    // Python floats are weak: `strength * m` multiplies in float32 by float32(strength).
    let s = strength as f32;
    let one_minus_s = (1.0 - strength) as f32;
    let mut out = vec![0.0_f32; h * w];
    for (lg, le) in layers_g.iter().zip(&layers_e) {
        let m = match_by_zone(lg, h, w, le, he, we, &lat_g, hadley_deg, &ze);
        for k in 0..h * w {
            out[k] += s * m[k] + one_minus_s * lg[k];
        }
    }
    let m = match_by_zone(&low_g, h, w, &low_e, he, we, &lat_g, hadley_deg, &ze);
    for k in 0..h * w {
        out[k] += m[k] * s + low_g[k] * one_minus_s;
    }
    for v in out.iter_mut() {
        // np.clip(out, 0.0, 1.0): NaN stays NaN.
        if *v < 0.0 {
            *v = 0.0;
        } else if *v > 1.0 {
            *v = 1.0;
        }
    }
    let fin = match_by_zone(&out, h, w, &earth32, he, we, &lat_g, hadley_deg, &ze);
    (0..h * w).map(|k| s * fin[k] + one_minus_s * out[k]).collect()
}

/// Gains of the finest scales of [`refine_to_earth`]: the band-pass layers of `sigma = 1` and `2` Earth texels (about 20 and 40 km), for
/// a map as wide as Earth's reference (2048) and for one twice as wide (4096; beyond, the 4096 values hold).
///
/// At 2048 wide Earth's spectrum falls faster over 40 to 400 km than the generator's (slope -2.90 against -2.65): the grain at those
/// scales is damped by 20 % and 10 %. At 4096 the quantile match of the layers itself steepens the spectrum (the generated layers hold
/// scales Earth's 2048-wide map cannot, and are squeezed to its amplitudes: slope -3.14 matched alone), so the same layers are
/// sharpened by 20 % and 10 %. Chosen by sweeps (seeds 1 to 4 at 2048, 5 and 6 at 4096): slope -2.87 and -2.97, coherence 0.483 and
/// 0.489 (Earth 0.487), histogram L1 0.022 and 0.035; stronger gains overshoot the slope and the coherence.
pub const REFINE_GAINS_2048: [f32; 2] = [0.8, 0.9];
/// See [`REFINE_GAINS_2048`].
pub const REFINE_GAINS_4096: [f32; 2] = [1.2, 1.1];

/// The gains of [`refine_to_earth`] for a generated map `ratio` times as wide as Earth's reference: [`REFINE_GAINS_2048`] up to 1,
/// [`REFINE_GAINS_4096`] from 2, in between linear in the log of the width.
pub fn refine_gains(ratio: f64) -> [f32; 2] {
    let t = ratio.log2().clamp(0.0, 1.0) as f32;
    [0, 1].map(|k| REFINE_GAINS_2048[k] + t * (REFINE_GAINS_4096[k] - REFINE_GAINS_2048[k]))
}

/// [`match_to_earth`], then the spectral tilt Earth's statistics ask for: the finest band-pass layers of the matched map are scaled by
/// [`refine_gains`] (a linear step, which sets the spectrum; the quantile match sets the marginals but not the slope), and the map is
/// matched to Earth's histograms once more, per latitude zone, because the tilt moved them. Not a port of `match.py`: the product
/// option `match_earth` runs this. The map must be whole (a quantile table per zone needs every row).
#[allow(clippy::too_many_arguments)]
pub fn refine_to_earth<C: Float, E: Float>(
    cov: &[C],
    h: usize,
    w: usize,
    earth: &[E],
    he: usize,
    we: usize,
    hadley_deg: f64,
    levels: usize,
    strength: f64,
) -> Vec<f32> {
    let matched = match_to_earth(cov, h, w, earth, he, we, hadley_deg, levels, strength);
    let ratio = w as f64 / we as f64;
    let gains = refine_gains(ratio);
    let sigmas: Vec<f64> = (0..gains.len()).map(|k| 2.0_f64.powi(k as i32) * ratio).collect();
    let (layers, mut out) = band_layers(&matched, h, w, &sigmas);
    for (layer, &gain) in layers.iter().zip(&gains) {
        for (o, &v) in out.iter_mut().zip(layer) {
            *o += gain * v;
        }
    }
    for v in out.iter_mut() {
        // Clamp like `match_to_earth` (NaN stays NaN).
        if *v < 0.0 {
            *v = 0.0;
        } else if *v > 1.0 {
            *v = 1.0;
        }
    }
    // No layer: the map as a whole, matched, clipped and matched again.
    match_to_earth(&out, h, w, earth, he, we, hadley_deg, 0, 1.0)
}

/// numpy and scipy primitives shared by `metrics` and `matching`, ported with their exact semantics (summation
/// order, dtype of every step, boundary modes). Generic and independent of the cloud model: later modules import
/// them from here (`cloud_map_generator_core::matching::ndimage`).
///
/// 2-D arrays are row-major `h x w` slices.
pub mod ndimage {
    use num_traits::Float;
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    fn c<F: Float>(x: f64) -> F {
        F::from(x).expect("float conversion")
    }

    fn f64_of<F: Float>(x: F) -> f64 {
        x.to_f64().expect("float conversion")
    }

    // -----------------------------------------------------------------------------------------------------
    // numpy reductions

    /// numpy's pairwise summation of a contiguous array (`np.add.reduce` of a 1-D array, or along a contiguous
    /// axis), in the precision of `F`: a plain loop under 8 elements, 8 interleaved accumulators up to 128, a
    /// recursive halving (on a multiple of 8) above.
    pub fn pairwise_sum<F: Float>(a: &[F]) -> F {
        let n = a.len();
        if n < 8 {
            let mut res = F::zero();
            for &x in a {
                res = res + x;
            }
            res
        } else if n <= 128 {
            let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
            let mut i = 8;
            while i < n - (n % 8) {
                for j in 0..8 {
                    r[j] = r[j] + a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            while i < n {
                res = res + a[i];
                i += 1;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
        }
    }

    /// `sum / n` as numpy does it: the sum in `F`, the division in float64 (the count is an `intp`), rounded
    /// back to `F`.
    fn div_count<F: Float>(sum: F, n: usize) -> F {
        c::<F>(f64_of(sum) / n as f64)
    }

    /// `ndarray.mean()` of a contiguous array: [`pairwise_sum`] over the count, in the dtype of the array (NaN for
    /// an empty array).
    pub fn mean<F: Float>(a: &[F]) -> F {
        div_count(pairwise_sum(a), a.len())
    }

    /// `ndarray.var(ddof=ddof)` of a contiguous array, in the dtype of the array: mean, squared deviations
    /// (pairwise), division by `n - ddof`.
    pub fn var<F: Float>(a: &[F], ddof: usize) -> F {
        let m = mean(a);
        let sq: Vec<F> = a
            .iter()
            .map(|&v| {
                let d = v - m;
                d * d
            })
            .collect();
        div_count(pairwise_sum(&sq), a.len().saturating_sub(ddof))
    }

    /// `ndarray.std(ddof=ddof)`: `sqrt(var)` in the dtype of the array.
    pub fn std<F: Float>(a: &[F], ddof: usize) -> F {
        var(a, ddof).sqrt()
    }

    /// `np.nanmean` of float64 values: NaN replaced by 0, summed pairwise, divided by the count of the others
    /// (NaN when all are NaN).
    pub fn nanmean(a: &[f64]) -> f64 {
        let cnt = a.iter().filter(|v| !v.is_nan()).count();
        let z: Vec<f64> = a.iter().map(|&v| if v.is_nan() { 0.0 } else { v }).collect();
        pairwise_sum(&z) / cnt as f64
    }

    /// `np.median` of float64 values: the middle one, or the mean of the middle two for an even count (NaN
    /// propagates).
    pub fn median(a: &[f64]) -> f64 {
        if a.iter().any(|v| v.is_nan()) {
            return f64::NAN;
        }
        let mut s = a.to_vec();
        s.sort_by(|x, y| x.partial_cmp(y).expect("no NaN"));
        let n = s.len();
        if n == 0 {
            return f64::NAN;
        }
        if n % 2 == 1 {
            s[n / 2]
        } else {
            pairwise_sum(&s[n / 2 - 1..n / 2 + 1]) / 2.0
        }
    }

    /// `np.max` of an array: NaN propagates (unlike `f64::max`).
    pub fn max_nan<F: Float>(a: &[F]) -> F {
        let mut m = a[0];
        for &x in a {
            if x.is_nan() {
                return x;
            }
            if x > m {
                m = x;
            }
        }
        m
    }

    // -----------------------------------------------------------------------------------------------------
    // numpy: gradient, arange, searchsorted, quantile, interp, polyfit, hanning

    /// `np.gradient(f)` of a 2-D array with unit spacing: central differences `(f[i+1] - f[i-1]) / 2` inside, one-sided
    /// first differences at both borders (edge order 1), in the dtype of the array. Returns `(d/drow, d/dcol)`.
    /// Needs at least 2 samples per axis.
    pub fn gradient<F: Float>(f: &[F], h: usize, w: usize) -> (Vec<F>, Vec<F>) {
        assert_eq!(f.len(), h * w);
        assert!(h >= 2 && w >= 2, "np.gradient needs at least 2 samples along each axis");
        let two = c::<F>(2.0);
        let mut gy = vec![F::zero(); h * w];
        let mut gx = vec![F::zero(); h * w];
        for i in 0..h {
            for j in 0..w {
                let k = i * w + j;
                gy[k] = if i == 0 {
                    f[w + j] - f[j]
                } else if i == h - 1 {
                    f[k] - f[k - w]
                } else {
                    (f[k + w] - f[k - w]) / two
                };
                gx[k] = if j == 0 {
                    f[k + 1] - f[k]
                } else if j == w - 1 {
                    f[k] - f[k - 1]
                } else {
                    (f[k + 1] - f[k - 1]) / two
                };
            }
        }
        (gy, gx)
    }

    /// `np.arange(start, stop, step)` for float arguments: `ceil((stop - start) / step)` values, `start`,
    /// `start + step`, then `start + i * delta` with `delta = (start + step) - start` (how numpy fills it).
    pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
        let len = ((stop - start) / step).ceil();
        if !(len > 0.0) {
            return vec![];
        }
        let len = len as usize;
        let delta = (start + step) - start;
        (0..len)
            .map(|i| match i {
                0 => start,
                1 => start + step,
                _ => start + i as f64 * delta,
            })
            .collect()
    }

    /// `np.searchsorted(a, v)` (side `left`) of a sorted array: the number of elements strictly below `v` (NaN
    /// sorts last).
    pub fn searchsorted_left(a: &[f64], v: f64) -> usize {
        if v.is_nan() {
            return a.len();
        }
        a.partition_point(|&x| x < v)
    }

    /// `np.quantile(data, q)` (method `linear`) of a 1-D array, for float64 `q` in `[0, 1]`: virtual index
    /// `(n - 1) q`, lerp between the two neighbours with `t` = the fractional part; the result is float64 whatever
    /// the dtype of the data, and the neighbour difference is computed in the dtype of the data. Any NaN gives NaN.
    pub fn quantile<F: Float>(data: &[F], q: &[f64]) -> Vec<f64> {
        let n = data.len();
        assert!(n > 0, "quantile of an empty array");
        if data.iter().any(|v| v.is_nan()) {
            return vec![f64::NAN; q.len()];
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|x, y| x.partial_cmp(y).expect("no NaN"));
        let last = (n - 1) as f64;
        q.iter()
            .map(|&qq| {
                let virt = last * qq;
                let mut prev = virt.floor();
                let mut next = prev + 1.0;
                if virt >= last {
                    // above the last index: the maximum.
                    prev = -1.0;
                    next = -1.0;
                }
                if virt < 0.0 {
                    prev = 0.0;
                    next = 0.0;
                }
                let gamma = virt - prev;
                let at = |i: f64| sorted[if i < 0.0 { n - 1 } else { i as usize }];
                let (a, b) = (at(prev), at(next));
                let diff = f64_of(b - a);
                let (af, bf) = (f64_of(a), f64_of(b));
                let mut r = af + diff * gamma;
                if gamma >= 0.5 {
                    r = bf - diff * (1.0 - gamma);
                }
                r
            })
            .collect()
    }

    /// `np.interp(x, xp, fp)` for one float64 `x`: clamps to `fp[0]` / `fp[last]` outside `[xp[0], xp[last]]`,
    /// `fp[j]` exactly at a node, `slope * (x - xp[j]) + fp[j]` between (with numpy's fallbacks when that is NaN).
    /// `xp` must be increasing.
    pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
        let n = xp.len();
        assert!(n > 0 && fp.len() == n);
        if x.is_nan() {
            return x;
        }
        if n == 1 {
            // numpy: left value below, right value above, fp at the node: all fp[0] for one point.
            return fp[0];
        }
        if x < xp[0] {
            return fp[0];
        }
        if x >= xp[n - 1] {
            return fp[n - 1];
        }
        // The largest j with xp[j] <= x (j <= n - 2 here).
        let j = xp.partition_point(|&v| v <= x) - 1;
        if xp[j] == x {
            return fp[j];
        }
        let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
        let mut r = slope * (x - xp[j]) + fp[j];
        if r.is_nan() {
            r = slope * (x - xp[j + 1]) + fp[j + 1];
            if r.is_nan() && fp[j] == fp[j + 1] {
                r = fp[j];
            }
        }
        r
    }

    /// The degree-1 `np.polyfit(x, y, 1)`: `(slope, intercept)` of the least-squares line, in closed form on
    /// mean-centered data (agrees with numpy's SVD solve to about 4e-14).
    pub fn polyfit1(x: &[f64], y: &[f64]) -> (f64, f64) {
        assert_eq!(x.len(), y.len());
        let (xm, ym) = (mean(x), mean(y));
        let dx: Vec<f64> = x.iter().map(|v| v - xm).collect();
        let sxx = pairwise_sum(&dx.iter().map(|v| v * v).collect::<Vec<_>>());
        let sxy = pairwise_sum(&dx.iter().zip(y).map(|(a, b)| a * (b - ym)).collect::<Vec<_>>());
        let slope = sxy / sxx;
        (slope, ym - slope * xm)
    }

    /// `np.hanning(m)`: the symmetric Hann window `0.5 + 0.5 cos(pi n / (m - 1))` for `n = -(m-1), -(m-3), ..., m-1`
    /// (`[1.0]` for `m = 1`, empty for `m = 0`).
    pub fn hanning(m: usize) -> Vec<f64> {
        match m {
            0 => vec![],
            1 => vec![1.0],
            _ => {
                let mf = m as f64;
                (0..m)
                    .map(|i| {
                        let n = (1.0 - mf) + 2.0 * i as f64;
                        0.5 + 0.5 * (std::f64::consts::PI * n / (mf - 1.0)).cos()
                    })
                    .collect()
            }
        }
    }

    // -----------------------------------------------------------------------------------------------------
    // numpy.fft

    /// `np.fft.fft2` of a real `h x w` array: the unnormalized forward transform, complex128 (last axis first, as
    /// numpy does).
    pub fn fft2(data: &[f64], h: usize, w: usize) -> Vec<Complex<f64>> {
        assert_eq!(data.len(), h * w);
        let mut planner = FftPlanner::<f64>::new();
        let mut buf: Vec<Complex<f64>> = data.iter().map(|&v| Complex::new(v, 0.0)).collect();
        planner.plan_fft_forward(w).process(&mut buf);
        let mut t = vec![Complex::new(0.0, 0.0); h * w];
        for i in 0..h {
            for j in 0..w {
                t[j * h + i] = buf[i * w + j];
            }
        }
        planner.plan_fft_forward(h).process(&mut t);
        for i in 0..h {
            for j in 0..w {
                buf[i * w + j] = t[j * h + i];
            }
        }
        buf
    }

    /// `np.fft.fftshift` of a 2-D array (all axes): the zero frequency moves to `n // 2` on each axis, also for odd
    /// `n` (a roll by `n // 2`).
    pub fn fftshift2<T: Copy>(x: &[T], h: usize, w: usize) -> Vec<T> {
        assert_eq!(x.len(), h * w);
        let mut out = x.to_vec();
        for i in 0..h {
            for j in 0..w {
                out[((i + h / 2) % h) * w + (j + w / 2) % w] = x[i * w + j];
            }
        }
        out
    }

    // -----------------------------------------------------------------------------------------------------
    // scipy.ndimage: correlate1d, gaussian_filter1d, gaussian_filter, label

    /// How a line is extended past its ends (`scipy.ndimage` boundary modes).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Mode {
        /// `a a a a | a b c d | d d d d`: the edge value repeats.
        Nearest,
        /// `a b c d | a b c d | a b c d`: periodic, folded as many times as the extension needs.
        Wrap,
        /// `d c b a | a b c d | d c b a`: the scipy default; the edge texel repeats.
        Reflect,
        /// `d c b | a b c d | c b a`: reflection about the edge texel.
        Mirror,
    }

    /// The index in `0..n` a (possibly out of range) line index maps to under `mode`.
    fn fold_index(i: isize, n: usize, mode: Mode) -> usize {
        let ni = n as isize;
        if (0..ni).contains(&i) {
            return i as usize;
        }
        match mode {
            Mode::Nearest => i.clamp(0, ni - 1) as usize,
            Mode::Wrap => i.rem_euclid(ni) as usize,
            Mode::Reflect => {
                let p = i.rem_euclid(2 * ni);
                (if p >= ni { 2 * ni - 1 - p } else { p }) as usize
            }
            Mode::Mirror => {
                if n == 1 {
                    0
                } else {
                    let m = 2 * ni - 2;
                    let p = i.rem_euclid(m);
                    (if p >= ni { m - p } else { p }) as usize
                }
            }
        }
    }

    /// `scipy.ndimage.correlate1d` of a 2-D array along `axis` (0: rows, 1: columns) with float64 `weights` (origin
    /// 0): every line is extended by `mode`, accumulated in float64 in scipy's order (center first, then the
    /// outermost pair inwards for a symmetric kernel) and rounded to the dtype of the input.
    ///
    /// The symmetric branch (every Gaussian) matches scipy bit for bit (checked on random inputs, all four modes);
    /// the antisymmetric and general branches follow scipy's loop order but agree with it to an ulp or two, scipy's
    /// build fusing the multiply-add there.
    pub fn correlate1d<F: Float>(data: &[F], h: usize, w: usize, axis: usize, weights: &[f64], mode: Mode) -> Vec<F> {
        assert_eq!(data.len(), h * w);
        assert!(axis < 2 && !weights.is_empty());
        let (n, lines) = if axis == 1 { (w, h) } else { (h, w) };
        let at = |line: usize, k: usize| if axis == 1 { line * w + k } else { k * w + line };
        let size1 = weights.len() / 2;
        let size2 = weights.len() - size1 - 1;
        let fw = |jj: isize| weights[(size1 as isize + jj) as usize];
        // scipy detects a symmetric (1) or antisymmetric (-1) odd kernel and sums the pairs.
        let mut symmetric = 0_i8;
        if weights.len() % 2 == 1 {
            symmetric = 1;
            for ii in 1..=weights.len() / 2 {
                if (weights[size1 + ii] - weights[size1 - ii]).abs() > f64::EPSILON {
                    symmetric = 0;
                    break;
                }
            }
            if symmetric == 0 {
                symmetric = -1;
                for ii in 1..=weights.len() / 2 {
                    if (weights[size1 + ii] + weights[size1 - ii]).abs() > f64::EPSILON {
                        symmetric = 0;
                        break;
                    }
                }
            }
        }
        let mut out = vec![F::zero(); h * w];
        let mut ext = vec![0.0_f64; n + size1 + size2];
        for line in 0..lines {
            for (k, e) in ext.iter_mut().enumerate() {
                let i = k as isize - size1 as isize;
                *e = f64_of(data[at(line, fold_index(i, n, mode))]);
            }
            for i in 0..n {
                let ctr = (i + size1) as isize;
                let e = |d: isize| ext[(ctr + d) as usize];
                let v = if symmetric > 0 {
                    let mut o = e(0) * fw(0);
                    for jj in -(size1 as isize)..0 {
                        o += (e(jj) + e(-jj)) * fw(jj);
                    }
                    o
                } else if symmetric < 0 {
                    let mut o = 0.0;
                    for jj in -(size1 as isize)..0 {
                        o += (e(jj) - e(-jj)) * fw(jj);
                    }
                    o
                } else {
                    let mut o = e(size2 as isize) * fw(size2 as isize);
                    for jj in -(size1 as isize)..size2 as isize {
                        o += e(jj) * fw(jj);
                    }
                    o
                };
                out[at(line, i)] = c::<F>(v);
            }
        }
        out
    }

    /// `scipy.ndimage._gaussian_kernel1d(sigma, 0, radius)`: `exp(-0.5 x^2 / sigma^2)` for `x = -radius..radius`,
    /// normalized to sum 1 (float64, numpy's pairwise sum).
    pub fn gaussian_kernel1d(sigma: f64, radius: usize) -> Vec<f64> {
        let sigma2 = sigma * sigma;
        let r = radius as i64;
        let phi: Vec<f64> = (-r..=r).map(|k| ((-0.5 / sigma2) * (k * k) as f64).exp()).collect();
        let s = pairwise_sum(&phi);
        phi.iter().map(|v| v / s).collect()
    }

    /// `scipy.ndimage.gaussian_filter1d(data, sigma, axis, mode=mode, truncate=truncate)` of a 2-D array: radius
    /// `int(truncate * sigma + 0.5)`, the kernel of [`gaussian_kernel1d`], [`correlate1d`] semantics (float64
    /// accumulation, output in the dtype of the input).
    pub fn gaussian_filter1d<F: Float>(
        data: &[F],
        h: usize,
        w: usize,
        axis: usize,
        sigma: f64,
        mode: Mode,
        truncate: f64,
    ) -> Vec<F> {
        let radius = (truncate * sigma + 0.5) as usize;
        // scipy reverses the kernel (it calls correlate, not convolve): the Gaussian is symmetric.
        let weights = gaussian_kernel1d(sigma, radius);
        correlate1d(data, h, w, axis, &weights, mode)
    }

    /// `scipy.ndimage.gaussian_filter(data, sigma, mode=modes, truncate=truncate)` of a 2-D array with one sigma and
    /// one mode per axis `[rows, columns]`: axis 0 then axis 1, an axis with `sigma <= 1e-15` skipped (a copy when
    /// both are), each pass rounded to the dtype of the input.
    pub fn gaussian_filter_axes<F: Float>(
        data: &[F],
        h: usize,
        w: usize,
        sigmas: [f64; 2],
        modes: [Mode; 2],
        truncate: f64,
    ) -> Vec<F> {
        let mut cur = data.to_vec();
        for axis in 0..2 {
            if sigmas[axis] > 1e-15 {
                cur = gaussian_filter1d(&cur, h, w, axis, sigmas[axis], modes[axis], truncate);
            }
        }
        cur
    }

    /// `scipy.ndimage.gaussian_filter(data, sigma, mode=modes)` with the default `truncate = 4.0` and the same sigma
    /// on both axes; `modes` is `[rows, columns]` (the scipy default is `[Mode::Reflect, Mode::Reflect]`).
    pub fn gaussian_filter<F: Float>(data: &[F], h: usize, w: usize, sigma: f64, modes: [Mode; 2]) -> Vec<F> {
        gaussian_filter_axes(data, h, w, [sigma, sigma], modes, 4.0)
    }

    /// `scipy.ndimage.label(mask)` with the default structure of a 2-D array, 4-connectivity (edge neighbours only,
    /// no wraparound): returns the labels (`0` = background, objects `1..=n` numbered by the raster position of their
    /// first texel) and the number of objects.
    pub fn label(mask: &[bool], h: usize, w: usize) -> (Vec<u32>, usize) {
        assert_eq!(mask.len(), h * w);
        let mut labels = vec![0_u32; h * w];
        let mut n = 0_u32;
        let mut stack: Vec<usize> = Vec::new();
        for start in 0..h * w {
            if !mask[start] || labels[start] != 0 {
                continue;
            }
            n += 1;
            labels[start] = n;
            stack.push(start);
            while let Some(k) = stack.pop() {
                let (r, col) = (k / w, k % w);
                let mut visit = |q: usize| {
                    if mask[q] && labels[q] == 0 {
                        labels[q] = n;
                        stack.push(q);
                    }
                };
                if r > 0 {
                    visit(k - w);
                }
                if r + 1 < h {
                    visit(k + w);
                }
                if col > 0 {
                    visit(k - 1);
                }
                if col + 1 < w {
                    visit(k + 1);
                }
            }
        }
        (labels, n as usize)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn pairwise_matches_a_plain_sum_on_exact_values_and_splits_long_arrays() {
            let a: Vec<f64> = (1..=1000).map(|v| v as f64).collect();
            assert_eq!(pairwise_sum(&a), 500500.0);
            assert_eq!(pairwise_sum::<f64>(&[]), 0.0);
            assert_eq!(pairwise_sum(&[1.5_f32, 2.0]), 3.5);
            // The 8-way unrolling reorders: 1e16 + 1 + (-1e16) in a block of 8 loses the 1 or not by position.
            let mut b = vec![0.0_f64; 9];
            b[0] = 1e16;
            b[8] = 1.0;
            assert_eq!(pairwise_sum(&b), 1e16);
        }

        #[test]
        fn pairwise_sum_follows_numpys_block_order_in_float32() {
            // 2^24 followed by ones: a sequential float32 sum stays at 2^24 (each + 1 rounds back to even), numpy's
            // interleaved accumulators keep most of the ones. Expected values computed with numpy 2.x.
            let sum = |n: usize| {
                let mut v = vec![1.0_f32; n + 1];
                v[0] = 16777216.0;
                pairwise_sum(&v)
            };
            assert_eq!(sum(100), 16777300.0);
            assert_eq!(sum(300), 16777508.0);
            assert_eq!(sum(1000), 16778204.0);
            // mean / std of a float32 array keep float32 (numpy: 0.05000000074505806, 0.04999999701976776).
            let mut x = vec![0.1_f32; 50];
            x.extend(vec![1e-8_f32; 50]);
            assert_eq!(f64::from(pairwise_sum(&x)), 5.0);
            assert_eq!(f64::from(mean(&x)), 0.05000000074505806);
            assert_eq!(f64::from(std(&x, 0)), 0.04999999701976776);
        }

        #[test]
        fn mean_var_std_follow_numpy() {
            let a = [1.0_f64, 2.0, 3.0, 4.0];
            assert_eq!(mean(&a), 2.5);
            assert_eq!(var(&a, 0), 1.25);
            assert!((std(&a, 1) - (5.0_f64 / 3.0).sqrt()).abs() < 1e-15);
            assert!(mean::<f32>(&[]).is_nan());
            assert!((nanmean(&[1.0, f64::NAN, 3.0]) - 2.0).abs() < 1e-15);
            assert!(nanmean(&[f64::NAN]).is_nan());
        }

        #[test]
        fn median_and_max_nan() {
            assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
            assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
            assert!(median(&[1.0, f64::NAN]).is_nan());
            assert!(max_nan(&[1.0_f32, f32::NAN, 3.0]).is_nan());
            assert_eq!(max_nan(&[1.0_f32, 5.0, 3.0]), 5.0);
        }

        #[test]
        fn gradient_is_central_inside_and_one_sided_at_the_borders() {
            // f = [[0, 1, 4], [1, 3, 7]] (h = 2, w = 3).
            let f = [0.0_f64, 1.0, 4.0, 1.0, 3.0, 7.0];
            let (gy, gx) = gradient(&f, 2, 3);
            assert_eq!(gy, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
            assert_eq!(gx, vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn arange_fills_like_numpy() {
            assert_eq!(arange(-90.0, 95.0, 5.0).len(), 37);
            assert_eq!(arange(-90.0, 97.0, 7.0).len(), 27);
            assert_eq!(*arange(-90.0, 97.0, 7.0).last().unwrap(), 92.0);
            assert_eq!(arange(-90.0, 450.0, 360.0), vec![-90.0, 270.0]);
            assert!(arange(1.0, 0.0, 1.0).is_empty());
            // An inexact step: numpy fills start + i * ((start + step) - start). Values from numpy 2.x.
            let r = arange(-90.0, 90.0 + 3.3, 3.3);
            assert_eq!(r.len(), 56);
            assert_eq!(r[2], -83.4);
            assert_eq!(r[30], 8.999999999999915);
            assert_eq!(r[55], 91.49999999999983);
        }

        #[test]
        fn searchsorted_left_counts_strictly_smaller() {
            let a = [1.0, 2.0, 2.0, 5.0];
            assert_eq!(searchsorted_left(&a, 2.0), 1);
            assert_eq!(searchsorted_left(&a, 0.0), 0);
            assert_eq!(searchsorted_left(&a, 9.0), 4);
            assert_eq!(searchsorted_left(&a, f64::NAN), 4);
        }

        #[test]
        fn quantile_is_numpys_linear_method() {
            let d = [4.0_f64, 1.0, 3.0, 2.0];
            assert_eq!(quantile(&d, &[0.0, 0.5, 1.0, 1.0 / 3.0]), vec![1.0, 2.5, 4.0, 2.0]);
            // t >= 0.5 uses the other form: b - (b - a)(1 - t).
            let q = quantile(&d, &[0.9])[0];
            assert!((q - 3.7).abs() < 1e-15);
            // A single value: every quantile is that value.
            assert_eq!(quantile(&[7.0_f32], &[0.0005, 0.9995]), vec![7.0, 7.0]);
            assert!(quantile(&[1.0_f32, f32::NAN], &[0.5])[0].is_nan());
            // float32 data, float64 q: the result is float64 but the neighbour difference is computed in float32
            // (1.0 - 1e-8 rounds to 1.0). Expected values computed with numpy 2.x: np.quantile(f32 array, f64 array).
            let r = quantile(&[1e-8_f32, 1.0], &[0.25, 0.75, 0.1]);
            assert_eq!(r, vec![0.25000000999999994, 0.75, 0.10000000999999994]);
        }

        #[test]
        fn interp_clamps_hits_nodes_and_interpolates() {
            let xp = [0.0, 1.0, 3.0];
            let fp = [10.0, 20.0, 40.0];
            assert_eq!(interp(-5.0, &xp, &fp), 10.0);
            assert_eq!(interp(9.0, &xp, &fp), 40.0);
            assert_eq!(interp(1.0, &xp, &fp), 20.0);
            assert_eq!(interp(3.0, &xp, &fp), 40.0);
            assert_eq!(interp(0.5, &xp, &fp), 15.0);
            assert_eq!(interp(2.0, &xp, &fp), 30.0);
            assert!(interp(f64::NAN, &xp, &fp).is_nan());
            assert_eq!(interp(4.0, &[1.0], &[7.0]), 7.0);
        }

        #[test]
        fn polyfit1_recovers_a_line() {
            let x: Vec<f64> = (0..10).map(|v| v as f64).collect();
            let y: Vec<f64> = x.iter().map(|v| 3.0 - 0.5 * v).collect();
            let (s, i) = polyfit1(&x, &y);
            assert!((s + 0.5).abs() < 1e-14 && (i - 3.0).abs() < 1e-13);
        }

        #[test]
        fn hanning_is_the_symmetric_window() {
            assert!(hanning(0).is_empty());
            assert_eq!(hanning(1), vec![1.0]);
            let w = hanning(5);
            let want = [0.0, 0.5, 1.0, 0.5, 0.0];
            for (a, b) in w.iter().zip(want) {
                assert!((a - b).abs() < 1e-15, "{w:?}");
            }
            let w6 = hanning(6);
            assert!((w6[1] - w6[4]).abs() < 1e-15 && (w6[0]).abs() < 1e-15);
        }

        #[test]
        fn fft2_of_an_impulse_is_flat_and_of_a_constant_is_a_dc_spike() {
            let mut a = vec![0.0; 3 * 5];
            a[0] = 1.0;
            assert!(fft2(&a, 3, 5).iter().all(|z| (z.re - 1.0).abs() < 1e-14 && z.im.abs() < 1e-14));
            let f = fft2(&vec![2.0; 4 * 6], 4, 6);
            assert!((f[0].re - 48.0).abs() < 1e-12);
            assert!(f[1..].iter().all(|z| z.norm() < 1e-12));
        }

        #[test]
        fn fft2_matches_the_naive_dft_with_the_numpy_sign_convention() {
            let (h, w) = (3, 4);
            let x: Vec<f64> = (0..h * w).map(|k| ((k * 7 + 3) % 11) as f64 - 4.5).collect();
            let f = fft2(&x, h, w);
            for u in 0..h {
                for v in 0..w {
                    let (mut re, mut im) = (0.0_f64, 0.0_f64);
                    for i in 0..h {
                        for j in 0..w {
                            let ang = -2.0
                                * std::f64::consts::PI
                                * ((u * i) as f64 / h as f64 + (v * j) as f64 / w as f64);
                            re += x[i * w + j] * ang.cos();
                            im += x[i * w + j] * ang.sin();
                        }
                    }
                    let z = f[u * w + v];
                    assert!((z.re - re).abs() < 1e-12 && (z.im - im).abs() < 1e-12, "({u},{v}): {z} vs {re} {im}");
                }
            }
        }

        #[test]
        fn fftshift2_rolls_by_half_for_even_and_odd_sizes() {
            let x: Vec<usize> = (0..5).collect();
            assert_eq!(fftshift2(&x, 1, 5), vec![3, 4, 0, 1, 2]);
            let y: Vec<usize> = (0..4).collect();
            assert_eq!(fftshift2(&y, 1, 4), vec![2, 3, 0, 1]);
            let z: Vec<usize> = (0..6).collect();
            // Both axes roll: rows by 3 // 2 = 1 and columns by 2 // 2 = 1.
            assert_eq!(fftshift2(&z, 3, 2), vec![5, 4, 1, 0, 3, 2]);
        }

        #[test]
        fn boundary_modes_extend_lines_as_scipy_documents() {
            // Line a b c d = 0 1 2 3, index -1 and n.
            let idx = |m| (0..4).map(|k| fold_index(k - 3, 4, m)).collect::<Vec<_>>();
            assert_eq!(idx(Mode::Nearest), vec![0, 0, 0, 0]);
            assert_eq!(idx(Mode::Wrap), vec![1, 2, 3, 0]);
            assert_eq!(idx(Mode::Reflect), vec![2, 1, 0, 0]);
            assert_eq!(idx(Mode::Mirror), vec![3, 2, 1, 0]);
            let after = |m| (4..8).map(|k| fold_index(k, 4, m)).collect::<Vec<_>>();
            assert_eq!(after(Mode::Nearest), vec![3, 3, 3, 3]);
            assert_eq!(after(Mode::Wrap), vec![0, 1, 2, 3]);
            assert_eq!(after(Mode::Reflect), vec![3, 2, 1, 0]);
            assert_eq!(after(Mode::Mirror), vec![2, 1, 0, 1]);
            // Far outside, and a single-element line.
            assert_eq!(fold_index(-13, 4, Mode::Reflect), 3);
            assert_eq!(fold_index(-13, 4, Mode::Mirror), 1);
            assert_eq!(fold_index(7, 1, Mode::Mirror), 0);
            assert_eq!(fold_index(-7, 1, Mode::Reflect), 0);
        }

        #[test]
        fn correlate1d_handles_symmetric_antisymmetric_and_general_kernels() {
            let a = [1.0_f64, 4.0, 9.0, 16.0, 25.0];
            // Symmetric box: a moving sum, reflect extension.
            let s = correlate1d(&a, 1, 5, 1, &[1.0, 1.0, 1.0], Mode::Reflect);
            assert_eq!(s, vec![6.0, 14.0, 29.0, 50.0, 66.0]);
            // Antisymmetric (central difference): w = [-1, 0, 1].
            let d = correlate1d(&a, 1, 5, 1, &[-1.0, 0.0, 1.0], Mode::Nearest);
            assert_eq!(d, vec![3.0, 8.0, 12.0, 16.0, 9.0]);
            // General (not symmetric): out[i] = sum_j w[j] a[i + j - size1], size1 = 1.
            let g = correlate1d(&a, 1, 5, 1, &[1.0, 2.0, 4.0], Mode::Nearest);
            assert_eq!(g[2], 1.0 * 4.0 + 2.0 * 9.0 + 4.0 * 16.0);
            assert_eq!(g[0], 1.0 * 1.0 + 2.0 * 1.0 + 4.0 * 4.0);
            // Even length: size1 = 2, size2 = 1.
            let e = correlate1d(&a, 1, 5, 1, &[1.0, 0.0, 0.0, 10.0], Mode::Nearest);
            assert_eq!(e[2], 1.0 * 1.0 + 10.0 * 16.0);
            // A kernel that is symmetric up to 1e-9 is NOT symmetric for scipy (tolerance DBL_EPSILON): general branch.
            let ns = correlate1d(&a, 1, 5, 1, &[1.0, 2.0, 1.0 + 1e-9], Mode::Nearest);
            let want = 1.0 * 4.0 + 2.0 * 9.0 + (1.0 + 1e-9) * 16.0;
            assert!((ns[2] - want).abs() < 1e-12, "{} vs {want}", ns[2]);
            // The same kernel along axis 0 of a column.
            let col = correlate1d(&a, 5, 1, 0, &[1.0, 1.0, 1.0], Mode::Reflect);
            assert_eq!(col, s);
        }

        #[test]
        fn correlate1d_rounds_a_float32_input_to_float32() {
            let a = [0.1_f32, 0.7, 0.3];
            let out = correlate1d(&a, 1, 3, 1, &[0.25, 0.5, 0.25], Mode::Wrap);
            let want = (f64::from(0.7_f32) * 0.5 + (f64::from(0.1_f32) + f64::from(0.3_f32)) * 0.25) as f32;
            assert_eq!(out[0], (f64::from(0.1_f32) * 0.5 + (f64::from(0.3_f32) + f64::from(0.7_f32)) * 0.25) as f32);
            assert_eq!(out[1], want);
        }

        #[test]
        fn gaussian_kernel_is_normalized_and_sized_by_truncate() {
            let k = gaussian_kernel1d(2.0, 8);
            assert_eq!(k.len(), 17);
            assert!((k.iter().sum::<f64>() - 1.0).abs() < 1e-15);
            assert!(k[8] > k[7] && k[7] == k[9]);
            assert_eq!(gaussian_kernel1d(0.3, 0), vec![1.0]);
        }

        #[test]
        fn gaussian_filter_keeps_constants_copies_on_tiny_sigma_and_wraps() {
            let flat = vec![0.25_f32; 6 * 7];
            for modes in [[Mode::Reflect; 2], [Mode::Nearest, Mode::Wrap], [Mode::Mirror; 2]] {
                for out in gaussian_filter(&flat, 6, 7, 2.0, modes) {
                    assert!((out - 0.25).abs() < 1e-7);
                }
            }
            let noise: Vec<f64> = (0..20).map(|k| ((k * 37) % 11) as f64).collect();
            assert_eq!(gaussian_filter(&noise, 4, 5, 0.0, [Mode::Reflect; 2]), noise);
            assert_eq!(gaussian_filter(&noise, 4, 5, 0.1, [Mode::Reflect; 2]), noise); // radius 0
            // sigma 0.15 already has radius int(4 * 0.15 + 0.5) = 1: it filters.
            assert_ne!(gaussian_filter(&noise, 4, 5, 0.15, [Mode::Reflect; 2]), noise);
            // Wrap: an impulse in column 0 leaks to the last column, with a clamp it does not (one row).
            let mut imp = vec![0.0_f64; 16];
            imp[0] = 1.0;
            let wrap = gaussian_filter(&imp, 1, 16, 1.0, [Mode::Nearest, Mode::Wrap]);
            let near = gaussian_filter(&imp, 1, 16, 1.0, [Mode::Nearest, Mode::Nearest]);
            assert!((wrap[15] - wrap[1]).abs() < 1e-15);
            assert!(near[15] < 1e-6);
            // Mass is conserved under wrap.
            assert!((wrap.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        }

        #[test]
        fn gaussian_filter_matches_scipy_values_in_every_mode() {
            // Reference values computed with scipy.ndimage.gaussian_filter (3 x 5 map, mode = (rows, columns)).
            let x = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 9.0, 1.0, 7.0, 2.0, 8.0, 6.0, 4.0, 0.0];
            let cases: [([Mode; 2], f64, [f64; 15]); 3] = [
                (
                    [Mode::Nearest, Mode::Wrap],
                    1.0,
                    [
                        2.4552600495506853, 2.4406325830473783, 2.9888406913753265, 3.2537521423124183, 3.076654415216707,
                        3.443744540843852, 4.015910140286307, 4.327086467472121, 3.7965023743385897, 3.40619117062011,
                        3.6193651620600127, 5.049135121611934, 5.178058700867812, 3.8464384475721314, 2.931291599517613,
                    ],
                ),
                (
                    [Mode::Mirror, Mode::Reflect],
                    0.8,
                    [
                        2.4037743081083653, 2.979971381923998, 3.8443150614432584, 3.640519499166742, 4.432169827546873,
                        3.3778951980847696, 4.214650436084774, 4.854594171321159, 3.863845578939879, 4.113985419061469,
                        3.902690963046955, 5.221342880132697, 5.562073269234861, 3.8978449135448754, 3.2653562888673022,
                    ],
                ),
                (
                    [Mode::Reflect, Mode::Mirror],
                    2.0,
                    [
                        3.6526330349214664, 3.6666937484548283, 3.688453095613913, 3.6930031891654203, 3.689921196093751,
                        3.864066103252438, 3.861901843267068, 3.843060160169611, 3.8050062734482784, 3.7836882645909737,
                        4.075107145311398, 4.056719896657986, 3.9972826129900025, 3.9166312354801476, 3.8770801453354773,
                    ],
                ),
            ];
            for (modes, sigma, want) in cases {
                let got = gaussian_filter(&x, 3, 5, sigma, modes);
                for k in 0..15 {
                    assert!((got[k] - want[k]).abs() < 1e-13, "{modes:?} sigma {sigma} at {k}: {} vs {}", got[k], want[k]);
                }
            }
            // float32: each pass is rounded to float32 (scipy: 0.4002234637737274 ... for x / 7, sigma 1.5).
            let x32: Vec<f32> = x.iter().map(|&v| v as f32 / 7.0).collect();
            let y = gaussian_filter(&x32, 3, 5, 1.5, [Mode::Nearest, Mode::Wrap]);
            assert_eq!(f64::from(y[0]), 0.4002234637737274);
            assert_eq!(f64::from(y[7]), 0.5346977114677429);
            assert_eq!(f64::from(y[14]), 0.5018824338912964);
            // Axis 0 THEN axis 1 matters in float32 (a float32 rounding between the passes): on this 4 x 6 map,
            // sigma 1.3, these four texels differ in the last bits when the order is reversed. scipy's values:
            let m: Vec<f32> = (0..24).map(|k| ((k * 37) % 101) as f32 / 100.0).collect();
            let g = gaussian_filter(&m, 4, 6, 1.3, [Mode::Nearest, Mode::Wrap]);
            assert_eq!(f64::from(g[0]), 0.40164613723754883);
            assert_eq!(f64::from(g[6]), 0.42519262433052063);
            assert_eq!(f64::from(g[18]), 0.5248386859893799);
            assert_eq!(f64::from(g[22]), 0.4347754120826721);
        }

        #[test]
        fn label_is_four_connected_numbered_in_raster_order() {
            // 1 . 1
            // . 1 .   -> the diagonal texels are separate objects.
            // . . 1
            let m = [true, false, true, false, true, false, false, false, true];
            let (lab, n) = label(&m, 3, 3);
            assert_eq!(n, 4);
            assert_eq!(lab, vec![1, 0, 2, 0, 3, 0, 0, 0, 4]);
            // A U shape is one object although its arms are numbered first by the raster scan.
            let u = [true, false, true, true, true, true];
            let (lab, n) = label(&u, 2, 3);
            assert_eq!((n, lab), (1, vec![1, 0, 1, 1, 1, 1]));
            let (lab, n) = label(&[false; 4], 2, 2);
            assert_eq!((n, lab), (0, vec![0; 4]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_constant() {
        let q = quantiles();
        assert_eq!(q.len(), 1024);
        assert_eq!(q[0], 0.0005);
        assert_eq!(q[1023], 0.9995);
    }

    #[test]
    fn earth_lat_caps_and_scales_by_the_hadley_edge() {
        let lat = [0.0_f64, 30.0_f64.to_radians(), -30.0_f64.to_radians(), 1.5];
        let e = earth_lat(&lat, 30.0);
        assert_eq!(e[0], 0.0);
        assert!((e[1] - 30.0).abs() < 1e-12 && (e[2] - 30.0).abs() < 1e-12);
        assert_eq!(e[3], 65.0);
        // Edge at 15 degrees: 15 degrees of latitude is Earth's 30.
        assert!((earth_lat(&[15.0_f64.to_radians()], 15.0)[0] - 30.0).abs() < 1e-12);
        // NaN propagates through the minimum.
        assert!(earth_lat(&[f64::NAN], 30.0)[0].is_nan());
        // float32 stays float32.
        let e32 = earth_lat(&[0.5_f32], 30.0);
        assert!((e32[0] - 28.6479).abs() < 1e-3);
    }

    #[test]
    fn zone_index_floors_into_five_degree_zones() {
        let lat: Vec<f64> = [0.0, 4.9, 5.1, 29.0, 64.9, 80.0].iter().map(|d: &f64| d.to_radians()).collect();
        assert_eq!(zone_index(&lat, 30.0), vec![0, 0, 1, 5, 12, 13]);
    }

    #[test]
    fn band_layers_telescope_to_the_image() {
        let h = 8;
        let w = 16;
        let img: Vec<f64> = (0..h * w).map(|k| ((k * 7919) % 101) as f64 / 100.0).collect();
        let (layers, low) = band_layers(&img, h, w, &[1.0, 2.0, 4.0]);
        assert_eq!(layers.len(), 3);
        for k in 0..h * w {
            let s: f64 = layers.iter().map(|l| l[k]).sum::<f64>() + low[k];
            assert!((s - img[k]).abs() < 1e-12);
        }
        let (none, same) = band_layers(&img, h, w, &[]);
        assert!(none.is_empty());
        assert_eq!(same, img);
    }

    #[test]
    fn matching_a_map_to_itself_keeps_it_close() {
        // Same map, same grid: the per-zone quantile maps are (nearly) the identity.
        let (h, w) = (32, 64);
        let (_, lat) = grid_lonlat(w, h);
        let img: Vec<f32> = (0..h * w).map(|k| (((k * 2654435761_usize) >> 7) % 1000) as f32 / 1000.0).collect();
        let ze = zone_index::<f64>(&lat, 30.0);
        let ze: Vec<i64> = ze.iter().zip(&lat).map(|(&z, &l)| if l.to_degrees().abs() < 67.5 { z } else { -1 }).collect();
        let out = match_by_zone(&img, h, w, &img, h, w, &lat, 30.0, &ze);
        let worst = img.iter().zip(&out).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(worst < 0.05, "worst {worst}");
    }

    #[test]
    #[should_panic(expected = "share no zone")]
    fn match_by_zone_without_a_shared_zone_panics() {
        let (_, lat) = grid_lonlat(4, 2);
        let img = vec![0.5_f32; 8];
        let _ = match_by_zone(&img, 2, 4, &img, 2, 4, &lat, 30.0, &[-1, -1]);
    }

    #[test]
    fn refine_gains_follow_the_width_between_the_two_fits() {
        assert_eq!(refine_gains(0.25), REFINE_GAINS_2048);
        assert_eq!(refine_gains(1.0), REFINE_GAINS_2048);
        assert_eq!(refine_gains(2.0), REFINE_GAINS_4096);
        assert_eq!(refine_gains(4.0), REFINE_GAINS_4096);
        let mid = refine_gains(2.0_f64.sqrt());
        assert!((mid[0] - 1.0).abs() < 1e-6 && (mid[1] - 1.0).abs() < 1e-6, "{mid:?}");
    }

    #[test]
    fn refine_damps_the_finest_scale_and_stays_in_range() {
        let (h, w, he, we) = (32, 64, 64, 128);
        // A smooth Earth, a noisy generated map: the refined one has less energy in the finest layer than the matched one.
        let earth: Vec<f32> = (0..he * we).map(|k| 0.5 + 0.4 * ((k % we) as f32 / 9.0).sin() * ((k / we) as f32 / 7.0).cos()).collect();
        let cov: Vec<f32> = (0..h * w).map(|k| (((k as u64).wrapping_mul(2654435761) >> 9) % 1000) as f32 / 1000.0).collect();
        let fine = |m: &[f32]| -> f64 {
            let (layers, _) = band_layers(m, h, w, &[1.0 * w as f64 / we as f64]);
            layers[0].iter().map(|&v| (v as f64).powi(2)).sum::<f64>()
        };
        let matched = match_to_earth(&cov, h, w, &earth, he, we, 30.0, DEFAULT_LEVELS, DEFAULT_STRENGTH);
        let refined = refine_to_earth(&cov, h, w, &earth, he, we, 30.0, DEFAULT_LEVELS, DEFAULT_STRENGTH);
        assert_eq!(refined.len(), h * w);
        assert!(refined.iter().all(|&v| v.is_finite() && (0.0..=1.0).contains(&v)));
        assert!(fine(&refined) < fine(&matched) * 1.05, "{} vs {}", fine(&refined), fine(&matched));
    }

    #[test]
    fn strength_zero_returns_the_map_and_a_flat_map_stays_in_range() {
        let (h, w, he, we) = (16, 32, 32, 64);
        let cov: Vec<f32> = (0..h * w).map(|k| ((k * 31) % 17) as f32 / 16.0).collect();
        let earth: Vec<f32> = (0..he * we).map(|k| ((k * 13) % 23) as f32 / 22.0).collect();
        let out = match_to_earth(&cov, h, w, &earth, he, we, 30.0, 3, 0.0);
        for k in 0..h * w {
            assert!((out[k] - cov[k]).abs() < 1e-5, "{k}: {} vs {}", out[k], cov[k]);
        }
        let flat = vec![0.4_f64; h * w];
        let o = match_to_earth(&flat, h, w, &earth, he, we, 30.0, DEFAULT_LEVELS, DEFAULT_STRENGTH);
        assert!(o.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }
}
