//! Port of `cloudgen/metrics.py`: numbers that say whether a coverage map looks like Earth's weather, measured the
//! same way on both.
//!
//! Every texture statistic is taken on gnomonic patches ([`patches`]), never on the equirectangular grid, whose
//! east-west stretch grows as 1/cos(lat) and would read as anisotropy.
//!
//! Maps are row-major `h x w` slices, north-up. Functions numpy runs in the dtype of the map are generic over
//! `f32` / `f64` and keep that precision (`ndarray.mean` of a float32 row accumulates in float32, a float32 value
//! compared with a Python float is compared in float32); everything the Python code promotes to float64 is float64
//! here too. Python dicts become structs with the keys as field names.

use crate::matching::ndimage::{self, Mode};
use crate::sphere::{gnomonic_patch, grid_lonlat, sample_equirect};
use num_traits::Float;
use std::collections::BTreeMap;

/// Latitudes (degrees) of the patch grid (`PATCH_LATS`).
pub const PATCH_LATS: [i32; 7] = [-55, -40, -20, 0, 20, 40, 55];
/// Longitudes (degrees) of the patch grid (`PATCH_LONS = range(-180, 180, 45)`).
pub const PATCH_LONS: [i32; 8] = [-180, -135, -90, -45, 0, 45, 90, 135];
/// Half-angle of a gnomonic patch, degrees (`PATCH_HALF_DEG`).
pub const PATCH_HALF_DEG: f64 = 10.0;
/// Texels per side of a patch (`PATCH_N`).
pub const PATCH_N: usize = 256;
/// Earth's radius, km (`EARTH_RADIUS_KM`).
pub const EARTH_RADIUS_KM: f64 = 6371.0;
/// Past this |latitude| the reference map's cloud fraction is contaminated by ice (`SURFACE_LAT_MAX`).
pub const SURFACE_LAT_MAX: f64 = 60.0;
/// A metric passes within this many spreads of one map across seeds (`SURFACE_SPREAD_FACTOR`).
pub const SURFACE_SPREAD_FACTOR: f64 = 2.2;
/// Smallest tolerance of `itcz_lat`: the crest is read on 2.5-degree bands, one band (`SURFACE_FLOOR['itcz_lat']`).
pub const SURFACE_FLOOR_ITCZ_LAT: f64 = 2.5;
/// Smallest tolerance of every other judged metric (the `SURFACE_FLOOR.get(k, 0.01)` default).
pub const SURFACE_FLOOR_DEFAULT: f64 = 0.01;
/// What `surface_contrasts` is judged on against Earth, each with the reference key of its season term
/// (`SURFACE_JUDGED`, in order).
pub const SURFACE_JUDGED: [(&str, Option<&str>); 6] = [
    ("land_minus_ocean", None),
    ("subtropics_land_minus_ocean", None),
    ("north_minus_south", Some("ocean_north_minus_south")),
    ("midlat_north_minus_south", Some("ocean_midlat_north_minus_south")),
    ("itcz_lat", None),
    ("deck_minus_open", None),
];

/// Default width of the zonal bands of [`zonal_profile`], degrees.
pub const DEFAULT_BAND_DEG: f64 = 5.0;
/// Default number of bins of [`area_histogram`].
pub const DEFAULT_BINS: usize = 10;
/// Default window sigma of [`coherence`], texels.
pub const DEFAULT_SIGMA: f64 = 4.0;
/// Default shortest wavelength of [`spectral_slope`], km.
pub const DEFAULT_LO_KM: f64 = 40.0;
/// Default longest wavelength of [`spectral_slope`], km.
pub const DEFAULT_HI_KM: f64 = 400.0;

fn c<F: Float>(x: f64) -> F {
    F::from(x).expect("float conversion")
}

fn f64_of<F: Float>(x: F) -> f64 {
    x.to_f64().expect("float conversion")
}

/// `np.degrees` of a float64: `x * (180 / pi)`.
fn degrees(x: f64) -> f64 {
    x * (180.0 / std::f64::consts::PI)
}

/// Python's `max(a, b)`: `a` unless `b > a` (so a NaN `a` stays NaN, a NaN `b` is ignored).
fn py_max<F: Float>(a: F, b: F) -> F {
    if b > a {
        b
    } else {
        a
    }
}

/// `zonal_profile(img, band_deg)`: the band centers (degrees, float64) and the mean of the row means in each band
/// `[a, b)` of `arange(-90, 90 + band_deg, band_deg)` (the row latitude is `degrees(lat)`). The profile has the dtype
/// of the image (float32 row means accumulate in float32); a band holding no row is NaN.
pub fn zonal_profile<F: Float>(img: &[F], h: usize, w: usize, band_deg: f64) -> (Vec<f64>, Vec<F>) {
    assert_eq!(img.len(), h * w);
    let (_, lat) = grid_lonlat(w, h);
    let latd: Vec<f64> = lat.iter().map(|&l| degrees(l)).collect();
    let edges = ndimage::arange(-90.0, 90.0 + band_deg, band_deg);
    let centers: Vec<f64> = edges.windows(2).map(|e| 0.5 * (e[0] + e[1])).collect();
    let rows: Vec<F> = (0..h).map(|r| ndimage::mean(&img[r * w..(r + 1) * w])).collect();
    let prof: Vec<F> = edges
        .windows(2)
        .map(|e| {
            let sel: Vec<F> = (0..h).filter(|&r| latd[r] >= e[0] && latd[r] < e[1]).map(|r| rows[r]).collect();
            ndimage::mean(&sel)
        })
        .collect();
    (centers, prof)
}

/// `area_histogram(img, bins)`: the 10-bin (by default) histogram of the coverage on `[0, 1]`, weighted by `cos(lat)`
/// and normalized to sum 1.
///
/// `np.histogram` semantics: the bin edges are `k * (1 / bins)` rounded to the dtype of the image (float32 edges for
/// a float32 map), bin `k` is `[edge_k, edge_k+1)` with the last bin closed, a texel equal to an edge goes to the
/// upper bin, values outside `[0, 1]` and NaN are dropped (all dropped: 0 / 0 = NaN in every bin). The weights are
/// summed in float64 per block of 65536 texels, the block sums added in order.
pub fn area_histogram<F: Float>(img: &[F], h: usize, w: usize, bins: usize) -> Vec<f64> {
    assert_eq!(img.len(), h * w);
    assert!(bins >= 1);
    let (_, lat) = grid_lonlat(w, h);
    let wrow: Vec<f64> = lat.iter().map(|l| l.cos()).collect();
    // np.linspace(0, 1, bins + 1, dtype=F): float64 k * step, rounded to F, the last edge exactly 1.
    let step = 1.0 / bins as f64;
    let mut edges: Vec<F> = (0..=bins).map(|k| c::<F>(k as f64 * step)).collect();
    edges[bins] = F::one();
    let (zero, one) = (F::zero(), F::one());
    const BLOCK: usize = 65536;
    let mut hist = vec![0.0_f64; bins];
    let mut start = 0;
    while start < h * w {
        let end = (start + BLOCK).min(h * w);
        let mut block = vec![0.0_f64; bins];
        for (i, &a) in img.iter().enumerate().take(end).skip(start) {
            if !(a >= zero && a <= one) {
                continue;
            }
            // Index estimate in float64 (a float32 texel is promoted by the uint64 divisor), then numpy's two fixes.
            let mut idx = (f64_of(a) * bins as f64) as i64;
            if idx == bins as i64 {
                idx -= 1;
            }
            if a < edges[idx as usize] {
                idx -= 1;
            }
            if a >= edges[idx as usize + 1] && idx != bins as i64 - 1 {
                idx += 1;
            }
            block[idx as usize] += wrow[i / w];
        }
        for k in 0..bins {
            hist[k] += block[k];
        }
        start = end;
    }
    let total = ndimage::pairwise_sum(&hist);
    hist.iter().map(|v| v / total).collect()
}

/// `global_mean(img)`: `sum(row_mean * cos(lat)) / sum(cos(lat))`. The row means keep the dtype of the image
/// (float32 pairwise accumulation for a float32 map); the weighting is float64.
pub fn global_mean<F: Float>(img: &[F], h: usize, w: usize) -> f64 {
    assert_eq!(img.len(), h * w);
    let (_, lat) = grid_lonlat(w, h);
    let cl: Vec<f64> = lat.iter().map(|l| l.cos()).collect();
    let weighted: Vec<f64> = (0..h).map(|r| f64_of(ndimage::mean(&img[r * w..(r + 1) * w])) * cl[r]).collect();
    ndimage::pairwise_sum(&weighted) / ndimage::pairwise_sum(&cl)
}

/// One gnomonic patch of [`patches`]: its center in whole degrees and its `PATCH_N x PATCH_N` float64 texels
/// (row 0 north, column 0 west).
#[derive(Clone, Debug)]
pub struct Patch {
    pub lat: i32,
    pub lon: i32,
    pub data: Vec<f64>,
}

/// `patches(img)`: the 56 gnomonic patches (`PATCH_LATS` x `PATCH_LONS`, latitude-major) sampled bilinearly from the
/// map CAST TO FLOAT32 first (a float64 map is rounded to float32), returned as float64 (the lerp weights are
/// float64).
pub fn patches<F: Float>(img: &[F], h: usize, w: usize) -> Vec<Patch> {
    assert_eq!(img.len(), h * w);
    let half = PATCH_HALF_DEG.to_radians();
    let img32: Vec<f64> = img.iter().map(|&v| f64::from(v.to_f32().expect("float conversion"))).collect();
    // The 56 patches are independent: one task each, joined in latitude-major order.
    crate::par::map_rows(PATCH_LATS.len() * PATCH_LONS.len(), |k| {
        let (la, lo) = (PATCH_LATS[k / PATCH_LONS.len()], PATCH_LONS[k % PATCH_LONS.len()]);
        let dirs = gnomonic_patch(f64::from(lo).to_radians(), f64::from(la).to_radians(), half, PATCH_N);
        Patch { lat: la, lon: lo, data: sample_equirect::<f64>(&img32, h, w, 1, &dirs) }
    })
}

/// `patch_px_km()`: kilometres per texel of a gnomonic patch, `2 tan(10 deg) 6371 / 256`.
pub fn patch_px_km() -> f64 {
    2.0 * PATCH_HALF_DEG.to_radians().tan() * EARTH_RADIUS_KM / PATCH_N as f64
}

/// `coherence(patch, sigma)`: the structure-tensor coherence `(lambda1 - lambda2) / (lambda1 + lambda2)` of an `h x w`
/// patch, weighted by the gradient energy: ~0 for an isotropic field, toward 1 for parallel bands.
/// `gaussian_filter(patch, 1.0)` (default `reflect` mode), `np.gradient`, the products smoothed by
/// `gaussian_filter(sigma)`; the whole chain keeps the dtype of the patch.
pub fn coherence<F: Float>(patch: &[F], h: usize, w: usize, sigma: f64) -> f64 {
    let modes = [Mode::Reflect; 2];
    let (gy, gx) = ndimage::gradient(&ndimage::gaussian_filter(patch, h, w, 1.0, modes), h, w);
    let prod = |a: &[F], b: &[F]| -> Vec<F> { a.iter().zip(b).map(|(&x, &y)| x * y).collect() };
    let jxx = ndimage::gaussian_filter(&prod(&gx, &gx), h, w, sigma, modes);
    let jyy = ndimage::gaussian_filter(&prod(&gy, &gy), h, w, sigma, modes);
    let jxy = ndimage::gaussian_filter(&prod(&gx, &gy), h, w, sigma, modes);
    let tiny = c::<F>(1e-12);
    let four = c::<F>(4.0);
    let tr: Vec<F> = jxx.iter().zip(&jyy).map(|(&a, &b)| a + b).collect();
    let weighted: Vec<F> = (0..h * w)
        .map(|k| {
            let d = jxx[k] - jyy[k];
            let disc = (d * d + four * jxy[k] * jxy[k]).sqrt();
            // np.maximum propagates NaN.
            let t = if tr[k].is_nan() || tr[k] > tiny { tr[k] } else { tiny };
            disc / t * tr[k]
        })
        .collect();
    f64_of(ndimage::pairwise_sum(&weighted) / py_max(ndimage::pairwise_sum(&tr), tiny))
}

/// Integer square root, rounded down (`int(np.hypot(x, y))` of integers is exactly this).
fn isqrt(s: u64) -> u64 {
    let mut r = (s as f64).sqrt() as u64;
    while r * r > s {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= s {
        r += 1;
    }
    r
}

/// `spectral_slope(patch, lo_km, hi_km)`: the log-log slope of the radially averaged power spectrum of a square
/// `n x n` patch over the wavelengths `lo_km..=hi_km` (NaN with fewer than 4 usable bins).
///
/// `(patch - mean) * Hann` (`np.hanning` outer itself, float64 whatever the dtype of the patch; the mean and the
/// subtraction keep the dtype), `fft2` (unnormalized), `fftshift`, `|F|^2`, the integer radius `int(hypot(x, y))`
/// counted from `n // 2`, `radial[r]` = the mean power at radius `r`; bin `k` has the wavelength `n px / k`, `px =`
/// [`patch_px_km`], and enters the fit when `lo <= wl <= hi` (inclusive) and `radial > 0`; the slope is the one of
/// the degree-1 `polyfit(log k, log radial)`.
pub fn spectral_slope<F: Float>(patch: &[F], n: usize, lo_km: f64, hi_km: f64) -> f64 {
    assert_eq!(patch.len(), n * n, "spectral_slope needs a square patch");
    let hann = ndimage::hanning(n);
    let m = ndimage::mean(patch);
    let windowed: Vec<f64> = (0..n * n).map(|k| f64_of(patch[k] - m) * (hann[k / n] * hann[k % n])).collect();
    let spec = ndimage::fftshift2(&ndimage::fft2(&windowed, n, n), n, n);
    let half = (n / 2) as i64;
    let mut sum = Vec::<f64>::new();
    let mut count = Vec::<u64>::new();
    for y in 0..n {
        for x in 0..n {
            let (dy, dx) = (y as i64 - half, x as i64 - half);
            let r = isqrt((dx * dx + dy * dy) as u64) as usize;
            if r >= sum.len() {
                sum.resize(r + 1, 0.0);
                count.resize(r + 1, 0);
            }
            // np.abs(complex) is hypot, then ** 2.
            let a = spec[y * n + x].re.hypot(spec[y * n + x].im);
            sum[r] += a * a;
            count[r] += 1;
        }
    }
    let px = patch_px_km();
    let (mut lk, mut lr) = (Vec::new(), Vec::new());
    for k in 1..sum.len() {
        let radial = sum[k] / count[k].max(1) as f64;
        let wl = n as f64 * px / k as f64;
        if wl >= lo_km && wl <= hi_km && radial > 0.0 {
            lk.push((k as f64).ln());
            lr.push(radial.ln());
        }
    }
    if lk.len() < 4 {
        return f64::NAN;
    }
    ndimage::polyfit1(&lk, &lr).0
}

/// The area (km^2) of each cloud object of a patch in `scipy.ndimage.label` order (by the raster position of the
/// object's first texel): `patch > 0.5` (strict, compared in the dtype of the patch), 4-connected, texel count times
/// the texel area `patch_px_km^2`.
pub fn object_areas_km2<F: Float>(patch: &[F], h: usize, w: usize) -> Vec<f64> {
    let half = c::<F>(0.5);
    let mask: Vec<bool> = patch.iter().map(|&v| v > half).collect();
    let (lab, n) = ndimage::label(&mask, h, w);
    let mut counts = vec![0_u64; n];
    for &l in &lab {
        if l > 0 {
            counts[l as usize - 1] += 1;
        }
    }
    let px = patch_px_km();
    let px_km2 = px * px;
    counts.iter().map(|&k| k as f64 * px_km2).collect()
}

/// `object_stats(patch)`: `(big, median)` of the cloud objects of a patch ([`object_areas_km2`]): the share of the
/// cloudy area held by objects larger than 1e5 km^2 (organized systems vs speckle) and the median object area in
/// km^2; `(0.0, 0.0)` when there is no object.
pub fn object_stats<F: Float>(patch: &[F], h: usize, w: usize) -> (f64, f64) {
    let areas = object_areas_km2(patch, h, w);
    if areas.is_empty() {
        return (0.0, 0.0);
    }
    let big: Vec<f64> = areas.iter().copied().filter(|&a| a > 1e5).collect();
    let share = ndimage::pairwise_sum(&big) / py_max(ndimage::pairwise_sum(&areas), 1e-9);
    (share, ndimage::median(&areas))
}

/// `detail_lost_2x(img)`: the share of the map's variance lost by a 2x2 box average brought back bilinearly (rows
/// clamp, columns wrap): how much lives at its own texel scale. Scale-free (a `u8` map gives the same number as the
/// same map in `0..1`); the map is converted to float64 first and every step is float64. `h` and `w` must be even
/// (a ValueError in Python). A flat map gives 0.
pub fn detail_lost_2x<T: Copy + Into<f64>>(img: &[T], h: usize, w: usize) -> f64 {
    assert_eq!(img.len(), h * w);
    assert!(h % 2 == 0 && w % 2 == 0, "detail_lost_2x needs even sizes");
    let a: Vec<f64> = img.iter().map(|&v| v.into()).collect();
    let (hh, hw) = (h / 2, w / 2);
    let mut half = vec![0.0_f64; hh * hw];
    for i in 0..hh {
        for j in 0..hw {
            let (r0, r1) = (2 * i * w + 2 * j, (2 * i + 1) * w + 2 * j);
            half[i * hw + j] = ((a[r0] + a[r0 + 1]) + (a[r1] + a[r1 + 1])) / 4.0;
        }
    }
    let fy: Vec<f64> = (0..h).map(|i| ((i as f64 + 0.5) / 2.0 - 0.5).max(0.0).min((hh - 1) as f64)).collect();
    let fx: Vec<f64> = (0..w).map(|j| (j as f64 + 0.5) / 2.0 - 0.5).collect();
    let mut sq = Vec::with_capacity(h * w);
    for i in 0..h {
        let y0f = fy[i].floor();
        let y0 = y0f as usize;
        let y1 = (hh - 1).min(y0 + 1);
        let ty = fy[i] - y0f;
        for j in 0..w {
            let x0f = fx[j].floor();
            let tx = fx[j] - x0f;
            let x0 = (x0f as i64).rem_euclid(hw as i64) as usize;
            let x1 = (x0 + 1) % hw;
            let up = (half[y0 * hw + x0] * (1.0 - tx) + half[y0 * hw + x1] * tx) * (1.0 - ty)
                + (half[y1 * hw + x0] * (1.0 - tx) + half[y1 * hw + x1] * tx) * ty;
            let d = a[i * w + j] - up;
            sq.push(d * d);
        }
    }
    let var = ndimage::var(&a, 0);
    if var > 0.0 {
        ndimage::mean(&sq) / var
    } else {
        0.0
    }
}

/// `seam_ratio(img)`: mean `|first column - last column|` over the mean `|neighbour column difference|` (the seam is
/// not wrapped in the inner term; 1 = invisible). Means of rows in the dtype of the map, so the quotient has it too.
pub fn seam_ratio<F: Float>(img: &[F], h: usize, w: usize) -> f64 {
    assert_eq!(img.len(), h * w);
    let edge: Vec<F> = (0..h).map(|r| (img[r * w] - img[r * w + w - 1]).abs()).collect();
    let mut inner = Vec::with_capacity(h * (w - 1));
    for r in 0..h {
        for k in 0..w - 1 {
            inner.push((img[r * w + k + 1] - img[r * w + k]).abs());
        }
    }
    let (edge, inner) = (ndimage::mean(&edge), ndimage::mean(&inner));
    f64_of(edge / py_max(inner, c::<F>(1e-12)))
}

/// `pole_std(img)`: the population standard deviation across longitude of the first and of the last row (a pole is
/// one point), in the dtype of the map.
pub fn pole_std<F: Float>(img: &[F], h: usize, w: usize) -> (f64, f64) {
    assert_eq!(img.len(), h * w);
    (f64_of(ndimage::std(&img[..w], 0)), f64_of(ndimage::std(&img[(h - 1) * w..], 0)))
}

/// The statistics of one patch in [`measure`] (the entries of the lists it averages).
#[derive(Clone, Debug)]
pub struct PatchMeasure {
    pub lat: i32,
    pub lon: i32,
    /// [`coherence`] with the default sigma.
    pub coherence: f64,
    /// [`spectral_slope`] with the default band (NaN for a flat patch).
    pub spectral_slope: f64,
    /// First of [`object_stats`].
    pub big_object_share: f64,
    /// Second of [`object_stats`].
    pub median_object_km2: f64,
}

/// What `measure(img)` returns (the dict keys are the field names).
#[derive(Clone, Debug, Default)]
pub struct Measure {
    pub global_mean: f64,
    /// Centers of the 5-degree zonal bands.
    pub zonal_lat: Vec<f64>,
    /// The zonal profile, widened from the dtype of the map (Python's `.tolist()`).
    pub zonal: Vec<f64>,
    pub hist: Vec<f64>,
    /// Mean of the per-patch coherences.
    pub coherence: f64,
    /// Mean over the 8 longitudes of each latitude of [`PATCH_LATS`], in that order.
    pub coherence_by_lat: Vec<f64>,
    /// `np.nanmean` of the per-patch slopes (a flat patch is NaN and skipped).
    pub spectral_slope: f64,
    pub big_object_share: f64,
    pub median_object_km2: f64,
    pub seam_ratio: f64,
    pub pole_std: (f64, f64),
}

/// `measure(img)` and the per-patch values it averages (patches order: latitude-major), to localize a difference.
pub fn measure_patches<F: Float>(img: &[F], h: usize, w: usize) -> (Measure, Vec<PatchMeasure>) {
    // Each patch is measured on its own (an FFT, a structure tensor, object labeling): one task per patch, in order;
    // the means over the patches below are the serial, numpy-ordered ones.
    let all = patches(img, h, w);
    let per: Vec<PatchMeasure> = crate::par::map_rows(all.len(), |k| {
        let p = &all[k];
        let (big, median) = object_stats(&p.data, PATCH_N, PATCH_N);
        PatchMeasure {
            lat: p.lat,
            lon: p.lon,
            coherence: coherence(&p.data, PATCH_N, PATCH_N, DEFAULT_SIGMA),
            spectral_slope: spectral_slope(&p.data, PATCH_N, DEFAULT_LO_KM, DEFAULT_HI_KM),
            big_object_share: big,
            median_object_km2: median,
        }
    });
    let coh: Vec<f64> = per.iter().map(|p| p.coherence).collect();
    let slp: Vec<f64> = per.iter().map(|p| p.spectral_slope).collect();
    let big: Vec<f64> = per.iter().map(|p| p.big_object_share).collect();
    let med: Vec<f64> = per.iter().map(|p| p.median_object_km2).collect();
    let (lat_c, prof) = zonal_profile(img, h, w, DEFAULT_BAND_DEG);
    let coherence_by_lat = PATCH_LATS
        .iter()
        .map(|&la| ndimage::mean(&per.iter().filter(|p| p.lat == la).map(|p| p.coherence).collect::<Vec<_>>()))
        .collect();
    let m = Measure {
        global_mean: global_mean(img, h, w),
        zonal_lat: lat_c,
        zonal: prof.iter().map(|&v| f64_of(v)).collect(),
        hist: area_histogram(img, h, w, DEFAULT_BINS),
        coherence: ndimage::mean(&coh),
        coherence_by_lat,
        spectral_slope: ndimage::nanmean(&slp),
        big_object_share: ndimage::mean(&big),
        median_object_km2: ndimage::median(&med),
        seam_ratio: seam_ratio(img, h, w),
        pole_std: pole_std(img, h, w),
    };
    (m, per)
}

/// `measure(img)`: the numbers that say whether a coverage map looks like Earth's weather.
pub fn measure<F: Float>(img: &[F], h: usize, w: usize) -> Measure {
    measure_patches(img, h, w).0
}

/// What `compare(gen, ref)` returns.
#[derive(Clone, Copy, Debug)]
pub struct Comparison {
    /// RMSE of the zonal means over the bands with `|lat| < 70`.
    pub zonal_rmse: f64,
    /// L1 distance of the two histograms.
    pub hist_l1: f64,
}

/// `compare(gen, ref)`: the zonal RMSE over the bands with `|gen.zonal_lat| < 70` (strict; the latitudes come from the
/// GENERATED measure, the reference's own `zonal_lat` is never read) and the L1 distance of the histograms. A NaN band
/// inside the selection makes the RMSE NaN (`np.mean` does not skip it).
pub fn compare(gen: &Measure, reference: &Measure) -> Comparison {
    let sq: Vec<f64> = gen
        .zonal_lat
        .iter()
        .enumerate()
        .filter(|(_, l)| l.abs() < 70.0)
        .map(|(k, _)| {
            let d = gen.zonal[k] - reference.zonal[k];
            d * d
        })
        .collect();
    let l1: Vec<f64> = gen.hist.iter().zip(&reference.hist).map(|(a, b)| (a - b).abs()).collect();
    Comparison { zonal_rmse: ndimage::mean(&sq).sqrt(), hist_l1: ndimage::pairwise_sum(&l1) }
}

/// The deck terms of [`SurfaceContrasts`] (present when a deck strength map is given).
#[derive(Clone, Copy, Debug)]
pub struct DeckContrasts {
    /// Ocean 8..35 degrees under a deck (`deck > 0.3`).
    pub deck: f64,
    /// Ocean 8..35 degrees with no coast downstream (`deck < 0.05`).
    pub deck_open: f64,
    pub deck_minus_open: f64,
}

/// What `surface_contrasts` returns (the dict keys are the field names, in the dict order).
#[derive(Clone, Copy, Debug)]
pub struct SurfaceContrasts {
    pub land: f64,
    pub ocean: f64,
    pub subtropics_land: f64,
    pub subtropics_ocean: f64,
    pub north: f64,
    pub south: f64,
    pub midlat_north: f64,
    pub midlat_south: f64,
    pub ocean_north_minus_south: f64,
    pub ocean_midlat_north_minus_south: f64,
    pub land_minus_ocean: f64,
    pub subtropics_land_minus_ocean: f64,
    pub north_minus_south: f64,
    pub midlat_north_minus_south: f64,
    pub itcz_lat: f64,
    pub deck: Option<DeckContrasts>,
}

impl SurfaceContrasts {
    /// The `(key, value)` pairs in the order of the Python dict (the three deck keys last, only with a deck).
    pub fn entries(&self) -> Vec<(&'static str, f64)> {
        let mut v = vec![
            ("land", self.land),
            ("ocean", self.ocean),
            ("subtropics_land", self.subtropics_land),
            ("subtropics_ocean", self.subtropics_ocean),
            ("north", self.north),
            ("south", self.south),
            ("midlat_north", self.midlat_north),
            ("midlat_south", self.midlat_south),
            ("ocean_north_minus_south", self.ocean_north_minus_south),
            ("ocean_midlat_north_minus_south", self.ocean_midlat_north_minus_south),
            ("land_minus_ocean", self.land_minus_ocean),
            ("subtropics_land_minus_ocean", self.subtropics_land_minus_ocean),
            ("north_minus_south", self.north_minus_south),
            ("midlat_north_minus_south", self.midlat_north_minus_south),
            ("itcz_lat", self.itcz_lat),
        ];
        if let Some(d) = &self.deck {
            v.extend([("deck", d.deck), ("deck_open", d.deck_open), ("deck_minus_open", d.deck_minus_open)]);
        }
        v
    }

    /// [`entries`](Self::entries) as a map, the form [`judge_surface`] reads.
    pub fn to_map(&self) -> MetricMap {
        self.entries().into_iter().map(|(k, v)| (k.to_string(), v)).collect()
    }
}

/// A dict of metric name to value (what `judge_surface` reads from the reference and from each map).
pub type MetricMap = BTreeMap<String, f64>;

/// `surface_contrasts(img, land, deck)`: land/sea and north/south contrasts of a map given a land fraction on the
/// same grid (and optionally the windowed deck strength). Area-weighted (`cos lat`, float64), `|lat| < 60` unless
/// named:
/// - `land_minus_ocean`, `subtropics_land_minus_ocean` (15-35 degrees, both hemispheres);
/// - `north_minus_south` (0-60), `midlat_north_minus_south` (40-60);
/// - `itcz_lat`: the crest of the 2.5-degree zonal mean within 20 degrees of the equator (the mean latitude of the
///   bands within 0.02 of the largest; NaN when a band is empty);
/// - `deck_minus_open`: ocean at 8..35 degrees under a deck (`strength > 0.3`) minus ocean at the same latitudes with
///   no coast downstream (`strength < 0.05`).
///
/// `land > 0.5` and the deck thresholds are compared in the dtype of each map (a float32 deck against `0.3` compares
/// `float32(0.3)`); a mask with no texel gives 0 (the 1e-12 floor).
pub fn surface_contrasts<F: Float, L: Float, D: Float>(
    img: &[F],
    land: &[L],
    deck: Option<&[D]>,
    h: usize,
    w: usize,
) -> SurfaceContrasts {
    assert_eq!(img.len(), h * w);
    assert_eq!(land.len(), h * w);
    let (_, lat) = grid_lonlat(w, h);
    let latd: Vec<f64> = lat.iter().map(|&l| degrees(l)).collect();
    let wts: Vec<f64> = latd.iter().map(|&d| (d * (std::f64::consts::PI / 180.0)).cos()).collect();
    let half = c::<L>(0.5);
    let is_land: Vec<bool> = land.iter().map(|&v| v > half).collect();
    let weighted_img: Vec<f64> = (0..h * w).map(|k| f64_of(img[k]) * wts[k / w]).collect();

    // mean(mask) over the texels of the mask inside |lat| < SURFACE_LAT_MAX; the mask reads (texel, row latitude).
    let mean = |mask: &dyn Fn(usize, f64) -> bool| -> f64 {
        let mut num = Vec::with_capacity(h * w);
        let mut den = Vec::with_capacity(h * w);
        for r in 0..h {
            let in_band = latd[r].abs() < SURFACE_LAT_MAX;
            for k in r * w..(r + 1) * w {
                let m = if in_band && mask(k, latd[r]) { 1.0 } else { 0.0 };
                num.push(weighted_img[k] * m);
                den.push(wts[r] * m);
            }
        }
        ndimage::pairwise_sum(&num) / py_max(ndimage::pairwise_sum(&den), 1e-12)
    };
    let sub = |lat: f64| lat.abs() >= 15.0 && lat.abs() < 35.0;
    let mid_n = |lat: f64| lat >= 40.0 && lat < 60.0;
    let mid_s = |lat: f64| lat <= -40.0 && lat > -60.0;

    let land_m = mean(&|k, _| is_land[k]);
    let ocean_m = mean(&|k, _| !is_land[k]);
    let subtropics_land = mean(&|k, l| sub(l) && is_land[k]);
    let subtropics_ocean = mean(&|k, l| sub(l) && !is_land[k]);
    let north = mean(&|_, l| l > 0.0);
    let south = mean(&|_, l| l < 0.0);
    let midlat_north = mean(&|_, l| mid_n(l));
    let midlat_south = mean(&|_, l| mid_s(l));
    let ocean_north_minus_south = mean(&|k, l| l > 0.0 && !is_land[k]) - mean(&|k, l| l < 0.0 && !is_land[k]);
    let ocean_midlat_north_minus_south = mean(&|k, l| mid_n(l) && !is_land[k]) - mean(&|k, l| mid_s(l) && !is_land[k]);

    // The crest of the 2.5-degree zonal mean: the threshold is taken in the dtype of the map (np.float32 - 0.02).
    let (cen, prof) = zonal_profile(img, h, w, 2.5);
    let sel: Vec<usize> = (0..cen.len()).filter(|&k| cen[k].abs() < 20.0).collect();
    let ps: Vec<F> = sel.iter().map(|&k| prof[k]).collect();
    let thr = ndimage::max_nan(&ps) - c::<F>(0.02);
    let crest: Vec<f64> = sel.iter().zip(&ps).filter(|(_, &p)| p >= thr).map(|(&k, _)| cen[k]).collect();
    let itcz_lat = ndimage::mean(&crest);

    let deck_terms = deck.map(|dk| {
        assert_eq!(dk.len(), h * w);
        let (strong, weak) = (c::<D>(0.3), c::<D>(0.05));
        let band = |k: usize, l: f64| l.abs() >= 8.0 && l.abs() < 35.0 && !is_land[k];
        let under = mean(&|k, l| band(k, l) && dk[k] > strong);
        let open = mean(&|k, l| band(k, l) && dk[k] < weak);
        DeckContrasts { deck: under, deck_open: open, deck_minus_open: under - open }
    });

    SurfaceContrasts {
        land: land_m,
        ocean: ocean_m,
        subtropics_land,
        subtropics_ocean,
        north,
        south,
        midlat_north,
        midlat_south,
        ocean_north_minus_south,
        ocean_midlat_north_minus_south,
        land_minus_ocean: land_m - ocean_m,
        subtropics_land_minus_ocean: subtropics_land - subtropics_ocean,
        north_minus_south: north - south,
        midlat_north_minus_south: midlat_north - midlat_south,
        itcz_lat,
        deck: deck_terms,
    }
}

/// The smallest tolerance of a judged metric (`SURFACE_FLOOR.get(key, 0.01)`).
pub fn surface_floor(key: &str) -> f64 {
    if key == "itcz_lat" {
        SURFACE_FLOOR_ITCZ_LAT
    } else {
        SURFACE_FLOOR_DEFAULT
    }
}

/// One row of [`judge_surface`].
#[derive(Clone, Debug)]
pub struct JudgeRow {
    pub metric: &'static str,
    pub reference: f64,
    /// Mean of the metric over the maps.
    pub mean: f64,
    /// Sample standard deviation over the maps (`ddof = 1`; NaN for one map).
    pub spread: f64,
    /// `|reference[season term]|`, 0 without one.
    pub season: f64,
    pub tolerance: f64,
    pub diff: f64,
    pub pass: bool,
}

/// `judge_surface(ref, maps)`: one row per judged metric ([`SURFACE_JUDGED`] order): reference, generated mean and
/// spread, tolerance, verdict. A metric passes when `|mean - reference| <= max(2.2 spread, floor) + season`, the
/// season being the absolute value of the reference's own season term. With a single map the spread is NaN, the
/// tolerance NaN, and nothing passes (Python's `max(nan, floor)` keeps the NaN: the first argument wins when a
/// comparison fails). Panics when a key is missing from a map.
pub fn judge_surface(reference: &MetricMap, maps: &[MetricMap]) -> Vec<JudgeRow> {
    let get = |m: &MetricMap, k: &str| *m.get(k).unwrap_or_else(|| panic!("judge_surface: no metric {k:?}"));
    SURFACE_JUDGED
        .iter()
        .map(|&(k, season)| {
            let v: Vec<f64> = maps.iter().map(|m| get(m, k)).collect();
            let sd = if v.len() > 1 { ndimage::std(&v, 1) } else { f64::NAN };
            let allowance = season.map_or(0.0, |s| get(reference, s).abs());
            let tol = py_max(SURFACE_SPREAD_FACTOR * sd, surface_floor(k)) + allowance;
            let m = ndimage::mean(&v);
            let diff = m - get(reference, k);
            JudgeRow {
                metric: k,
                reference: get(reference, k),
                mean: m,
                spread: sd,
                season: allowance,
                tolerance: tol,
                diff,
                pass: tol.is_finite() && diff.abs() <= tol,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_the_documented_ones() {
        assert_eq!(PATCH_LONS.len(), 8);
        assert_eq!(PATCH_LONS[0], -180);
        assert_eq!(PATCH_LONS[7], 135);
        assert_eq!(PATCH_LATS.len(), 7);
        assert!((patch_px_km() - 8.7764).abs() < 1e-3);
        assert_eq!(surface_floor("itcz_lat"), 2.5);
        assert_eq!(surface_floor("land_minus_ocean"), 0.01);
    }

    #[test]
    fn defaults_are_the_python_defaults() {
        // zonal_profile(band_deg=5.0), area_histogram(bins=10), coherence(sigma=4.0), spectral_slope(40.0, 400.0).
        assert_eq!(DEFAULT_BAND_DEG, 5.0);
        assert_eq!(DEFAULT_BINS, 10);
        assert_eq!(DEFAULT_SIGMA, 4.0);
        assert_eq!(DEFAULT_LO_KM, 40.0);
        assert_eq!(DEFAULT_HI_KM, 400.0);
    }

    #[test]
    fn isqrt_is_exact() {
        for s in [0_u64, 1, 2, 3, 4, 8, 9, 15, 16, 17, 24, 25, 1_000_000, 999_999, 4_194_304, 4_194_303] {
            let r = isqrt(s);
            assert!(r * r <= s && (r + 1) * (r + 1) > s, "{s}");
        }
    }

    #[test]
    fn zonal_profile_groups_rows_into_bands() {
        // 4 rows at latitudes 67.5, 22.5, -22.5, -67.5 degrees, 90-degree bands: [-90, 0) and [0, 90).
        let img = [1.0_f64, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let (cen, prof) = zonal_profile(&img, 4, 2, 90.0);
        assert_eq!(cen, vec![-45.0, 45.0]);
        assert_eq!(prof, vec![3.5, 1.5]);
    }

    #[test]
    fn area_histogram_sends_edge_texels_to_the_upper_bin_and_closes_the_last() {
        let img = [0.0_f64, 0.25, 0.5, 1.0];
        let hist = area_histogram(&img, 1, 4, 4);
        assert_eq!(hist, vec![0.25, 0.25, 0.25, 0.25]);
        // 0.7 as float64 is below the float64 edge 7 * 0.1: bin 6 of 10.
        let h2 = area_histogram(&[0.7_f64], 1, 1, 10);
        assert_eq!(h2[6], 1.0);
        // Out of range and NaN are dropped, the rest renormalized.
        let h3 = area_histogram(&[0.05_f32, -1.0, f32::NAN, 2.0], 1, 4, 10);
        assert_eq!(h3[0], 1.0);
        assert!(area_histogram(&[2.0_f32], 1, 1, 10).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn global_mean_of_a_constant_is_the_constant() {
        let img = vec![0.375_f64; 12 * 5];
        assert!((global_mean(&img, 12, 5) - 0.375).abs() < 1e-15);
    }

    #[test]
    fn patch_stats_of_trivial_maps() {
        let flat = vec![0.5_f64; 64];
        assert_eq!(coherence(&flat, 8, 8, 4.0), 0.0);
        assert!(spectral_slope(&flat, 8, 40.0, 400.0).is_nan());
        assert_eq!(object_stats(&flat, 8, 8), (0.0, 0.0));
        assert_eq!(object_areas_km2(&[0.9_f32, 0.1, 0.1, 0.9], 2, 2).len(), 2);
    }

    #[test]
    fn detail_lost_is_zero_for_flat_and_one_for_a_checkerboard() {
        assert_eq!(detail_lost_2x(&[0.25_f32; 16], 4, 4), 0.0);
        // A texel checkerboard: the box average is flat 0.5, all the variance is lost.
        let cb: Vec<f64> = (0..64).map(|k| ((k / 8 + k % 8) % 2) as f64).collect();
        assert!((detail_lost_2x(&cb, 8, 8) - 1.0).abs() < 1e-12);
        // Scale-free: bytes give the same number as 0..1.
        let u: Vec<u8> = (0..64).map(|k| ((k * 37) % 256) as u8).collect();
        let f: Vec<f64> = u.iter().map(|&v| f64::from(v) / 255.0).collect();
        assert!((detail_lost_2x(&u, 8, 8) - detail_lost_2x(&f, 8, 8)).abs() < 1e-12);
    }

    #[test]
    fn seam_and_pole_statistics() {
        // 0 .25 .5 .75 .. a ramp: seam 0.75 over a neighbour step of 0.25.
        let ramp = [0.0_f64, 0.25, 0.5, 0.75];
        assert!((seam_ratio(&ramp, 1, 4) - 3.0).abs() < 1e-12);
        // Flat: 0 over the 1e-12 floor.
        assert_eq!(seam_ratio(&[0.5_f32; 8], 2, 4), 0.0);
        let (n, s) = pole_std(&[0.0_f32, 1.0, 0.0, 1.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 1.0, 1.0], 3, 4);
        assert!((n - 0.5).abs() < 1e-7 && (s - 0.5).abs() < 1e-7, "{n} {s}");
        // One row: the two are the same row.
        let (a, b) = pole_std(&[1.0_f64, 3.0], 1, 2);
        assert_eq!((a, b), (1.0, 1.0));
    }

    #[test]
    fn compare_reads_the_generated_latitudes_only() {
        let gen = Measure {
            zonal_lat: vec![-80.0, 0.0, 69.999, 70.0],
            zonal: vec![0.0, 0.5, 0.5, 9.0],
            hist: vec![0.5, 0.5],
            ..Measure::default()
        };
        let mut reference = Measure {
            zonal_lat: vec![0.0; 4],
            zonal: vec![0.0, 0.3, 0.7, 0.0],
            hist: vec![0.25, 0.75],
            ..Measure::default()
        };
        let c1 = compare(&gen, &reference);
        reference.zonal_lat = vec![100.0; 4];
        let c2 = compare(&gen, &reference);
        assert_eq!(c1.zonal_rmse, c2.zonal_rmse);
        assert!((c1.zonal_rmse - ((0.2_f64 * 0.2 + 0.2 * 0.2) / 2.0).sqrt()).abs() < 1e-12);
        assert_eq!(c1.hist_l1, 0.5);
        // A NaN band inside the selection poisons the RMSE; no band selected too.
        let nan = Measure { zonal_lat: vec![0.0], zonal: vec![f64::NAN], hist: vec![1.0], ..Measure::default() };
        assert!(compare(&nan, &nan).zonal_rmse.is_nan());
    }

    #[test]
    fn surface_contrasts_of_a_uniform_land_sea_map() {
        let (h, w) = (72, 8);
        let img = vec![0.5_f32; h * w];
        // Land in the northern hemisphere only.
        let land: Vec<f32> = (0..h * w).map(|k| if k / w < h / 2 { 1.0 } else { 0.0 }).collect();
        let sc = surface_contrasts(&img, &land, None::<&[f32]>, h, w);
        assert!((sc.land - 0.5).abs() < 1e-12 && (sc.ocean - 0.5).abs() < 1e-12);
        assert_eq!(sc.land_minus_ocean, sc.land - sc.ocean);
        assert!(sc.deck.is_none() && sc.entries().len() == 15);
        // A deck: texels under 0.3 / over 0.05 split the ocean band.
        let deck: Vec<f64> = (0..h * w).map(|k| if k % 2 == 0 { 0.9 } else { 0.0 }).collect();
        let with = surface_contrasts(&img, &land, Some(&deck[..]), h, w);
        let d = with.deck.expect("deck terms");
        assert!((d.deck - 0.5).abs() < 1e-12 && (d.deck_open - 0.5).abs() < 1e-12);
        assert_eq!(with.entries().len(), 18);
        assert_eq!(with.to_map()["deck_minus_open"], d.deck_minus_open);
        // A map with no ocean reads 0 through the floor.
        let all_land = vec![1.0_f32; h * w];
        assert_eq!(surface_contrasts(&img, &all_land, None::<&[f32]>, h, w).ocean, 0.0);
    }

    fn map(vals: [f64; 8]) -> MetricMap {
        let keys = [
            "land_minus_ocean",
            "subtropics_land_minus_ocean",
            "north_minus_south",
            "midlat_north_minus_south",
            "itcz_lat",
            "deck_minus_open",
            "ocean_north_minus_south",
            "ocean_midlat_north_minus_south",
        ];
        keys.iter().zip(vals).map(|(k, v)| (k.to_string(), v)).collect()
    }

    #[test]
    fn judge_surface_applies_floors_seasons_and_the_nan_rule() {
        let reference = map([0.0, 0.0, 0.0, 0.0, 8.0, 0.0, -0.2, 0.05]);
        let a = map([0.1, 0.0, 0.15, 0.04, 8.0, 0.0, 0.0, 0.0]);
        let rows = judge_surface(&reference, &[a.clone(), a.clone()]);
        assert_eq!(rows.len(), 6);
        assert_eq!(rows[0].tolerance, 0.01); // no season, spread 0: the floor
        assert!(!rows[0].pass);
        assert!((rows[2].tolerance - (0.01 + 0.2)).abs() < 1e-15 && rows[2].pass); // season allowance
        assert!((rows[3].tolerance - (0.01 + 0.05)).abs() < 1e-15 && rows[3].pass);
        assert_eq!(rows[4].tolerance, 2.5);
        // One map: the spread is NaN, nothing passes.
        let one = judge_surface(&reference, &[a]);
        assert!(one.iter().all(|r| r.spread.is_nan() && r.tolerance.is_nan() && !r.pass));
    }

    #[test]
    fn judge_surface_never_passes_an_infinite_tolerance() {
        // Two maps at +-1e308: the spread overflows to inf, so tolerance = inf and |diff| = 0 <= inf, but
        // np.isfinite(tol) is False: no verdict.
        let reference = map([0.0; 8]);
        let (a, b) = (map([1e308; 8]), map([-1e308; 8]));
        let rows = judge_surface(&reference, &[a, b]);
        assert!(rows.iter().all(|r| r.tolerance.is_infinite() && r.diff == 0.0 && !r.pass));
    }

    #[test]
    fn measure_runs_on_a_small_map_with_flat_patches() {
        let (h, w) = (16, 32);
        let img: Vec<f32> = (0..h * w).map(|k| if k % w < w / 2 { 0.0 } else { ((k * 13) % 17) as f32 / 16.0 }).collect();
        let (m, per) = measure_patches(&img, h, w);
        assert_eq!(per.len(), 56);
        assert_eq!(m.zonal_lat.len(), 36);
        assert_eq!(m.hist.len(), 10);
        assert_eq!(m.coherence_by_lat.len(), 7);
        assert!(per.iter().any(|p| p.spectral_slope.is_nan()) || per.iter().all(|p| p.spectral_slope.is_finite()));
        assert!((m.hist.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
