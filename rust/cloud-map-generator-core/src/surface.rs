//! Port of `cloudgen/surface.py`: what the body's surface (land and sea, relief) does to its clouds.
//!
//! Four terms change the CLIMATOLOGICAL cloud amount the per-row calibration aims at (the decision records
//! are in the Python module docstring and are not repeated here): land is drier ([`continentality`],
//! [`land_contrast`]), marine stratocumulus east of the subtropical highs ([`deck_strength`],
//! [`deck_window`]), orographic uplift and the rain shadow ([`orographic_anomaly`]), and the ITCZ in the
//! hemisphere with more land ([`itcz_shift_deg`]). [`surface_terms`] assembles them.
//!
//! Maps are NORTH-UP row-major `f32` arrays with explicit dims (`h` rows of `w` columns); the wind is
//! interleaved east/north (`h * w * 2`). Python's dtypes are kept at every step: the surface fields are `f32`,
//! the grids (`lat`, `x = |lat| / phi_H`, the Earth tables) are `f64`, and a numpy expression that mixes the
//! two is evaluated here in the precision numpy promotes it to (each such spot says so). Python float scalars
//! are WEAK in numpy (a `float32` array times `0.25` stays `float32`): the generic scalar helpers round the
//! constant to the dtype of the data, as numpy does.
//!
//! numpy summation order is reproduced where a float32 sum is taken ([`pairwise_sum`]: `mean`, `sum`), and the
//! scipy Gaussian filter in [`scipy_filters`] keeps scipy's kernel, boundary modes and accumulation order.

use crate::sphere::{grid_dirs, grid_lonlat, sample_equirect};
use num_traits::Float;
use std::borrow::Cow;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------------------------------------
// Constants (the decision-record numbers)

// 1. Land contrast (ocean minus land) as a function of x: negative in the convergence zone, a plateau through
// the subtropics and the storm tracks, a deeper dip under the subtropical highs, fading toward the poles.
pub const LAND_CONTRAST_CONVERGENCE: f64 = -0.08;
pub const LAND_CONTRAST_EXTRATROPICS: f64 = 0.12;
pub const LAND_CONTRAST_DESERT: f64 = 0.04;
pub const CONTINENTALITY_KM: f64 = 300.0;

// 2. Marine stratocumulus.
pub const DECK_AMPLITUDE: f64 = 0.35;
pub const DECK_EFOLD_KM: f64 = 1200.0;
/// Smoothed land fraction that counts as a continent's coast.
pub const DECK_COAST_LAND: f64 = 0.5;
pub const DECK_MAX_SEARCH_KM: f64 = 4000.0;

// 3. Relief.
pub const OROGRAPHIC_AMPLITUDE: f64 = 0.25;
pub const RAIN_SHADOW_AMPLITUDE: f64 = 0.20;
pub const OROGRAPHIC_W0_MPS: f64 = 0.05;
pub const RAIN_SHADOW_KM: f64 = 800.0;
/// The air must have come down this far (km) from the upstream ground to be in the shadow.
pub const SHADOW_DESCENT_KM: (f64, f64) = (0.2, 1.0);

// 4. ITCZ.
pub const EARTH_ITCZ_SHIFT_X: f64 = 5.0 / 30.0;
pub const EARTH_LAND_ASYMMETRY: f64 = 0.20;
pub const ITCZ_SHIFT_MAX_X: f64 = 0.35;

/// Earth's zonal-mean table nodes: the centers of 5-degree bands (x steps of 1/6, phi_H = 30), out to 75 deg
/// (`np.arange(15) / 6.0 + 1.0 / 12.0`). Past 75 deg the tables hold their last value (`np.interp` clamps).
pub const EARTH_TABLE_X: [f64; 15] = {
    let mut a = [0.0; 15];
    let mut i = 0;
    while i < 15 {
        a[i] = i as f64 / 6.0 + 1.0 / 12.0;
        i += 1;
    }
    a
};
/// The edges of the 15 half-open bands `[k/6, (k+1)/6)` of `earth_tables` (`np.arange(16) / 6.0`).
pub const EARTH_BAND_EDGES_X: [f64; 16] = {
    let mut a = [0.0; 16];
    let mut i = 0;
    while i < 16 {
        a[i] = i as f64 / 6.0;
        i += 1;
    }
    a
};
/// Earth's zonal mean of term 1 (continentality), both hemispheres averaged, measured on Earth's DEM.
pub const EARTH_CONTINENTALITY: [f64; 15] =
    [0.231, 0.232, 0.232, 0.263, 0.294, 0.306, 0.286, 0.253, 0.258, 0.283, 0.297, 0.296, 0.359, 0.451, 0.486];
/// Earth's zonal mean of term 2 (windowed deck strength), both hemispheres averaged, measured on Earth's DEM.
pub const EARTH_DECK: [f64; 15] = [0.0, 0.0002, 0.057, 0.1228, 0.1141, 0.1064, 0.0695, 0.0029, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

// Numbers that sit inline in the Python source, named here so the golden glue can check them.
/// Floor of `cos(lat)` in every km-per-texel conversion and in the back-trace longitude (Python: `1e-3`).
pub const COS_FLOOR: f64 = 1e-3;
/// A row is smoothed along longitude only when its sigma exceeds this many pixels (Python: `s > 0.3`).
pub const ROW_MIN_SIGMA_PX: f64 = 0.3;
/// Cap of a row's sigma, as a fraction of the width: a quarter turn (Python: `w / 4.0`).
pub const ROW_MAX_SIGMA_WIDTH_FRACTION: f64 = 0.25;
/// Floor of the wind speed in the back-trace (m/s, Python: `1e-3`).
pub const SPEED_FLOOR_MPS: f64 = 1e-3;
/// Fractions of [`RAIN_SHADOW_KM`] at which the upwind uplift is sampled.
pub const SHADOW_FRACTIONS: [f64; 4] = [0.15, 0.35, 0.6, 1.0];
/// The sample at `frac` is weighted by `1 - SHADOW_WEIGHT_PER_FRACTION * frac`.
pub const SHADOW_WEIGHT_PER_FRACTION: f64 = 0.5;
/// `land_contrast`: the convergence-zone Gaussian `_gauss(x, mu, s)`.
pub const CONVERGENCE_GAUSS: (f64, f64) = (0.0, 0.3);
/// `land_contrast`: the extratropical plateau rises over this x range ...
pub const EXTRATROPICS_RISE: (f64, f64) = (0.35, 0.7);
/// ... and fades over this one.
pub const EXTRATROPICS_FADE: (f64, f64) = (2.2, 2.8);
/// `land_contrast`: the dip under the subtropical highs, `_gauss(x, mu, s)`.
pub const DESERT_GAUSS: (f64, f64) = (0.8, 0.25);
/// `deck_window`: the window rises over this x range ...
pub const WINDOW_RISE: (f64, f64) = (0.3, 0.5);
/// ... and falls over this one.
pub const WINDOW_FALL: (f64, f64) = (0.95, 1.25);

// ---------------------------------------------------------------------------------------------------------
// numpy semantics

/// The constant `x` rounded to the dtype `F` (a Python float is a weak scalar: numpy rounds it to the dtype of
/// the array it meets).
fn c<F: Float>(x: f64) -> F {
    F::from(x).expect("float conversion")
}

/// Python `min(a, b)`: `b` only when `b < a` (so a NaN `b` loses, a NaN `a` stays).
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Python `max(a, b)`: `b` only when `b > a`.
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// `np.maximum(a, b)`: NaN-propagating, `a` on a tie (`(a >= b || isnan(a)) ? a : b`).
fn np_maximum<F: Float>(a: F, b: F) -> F {
    if a >= b || a.is_nan() {
        a
    } else {
        b
    }
}

/// `np.clip(x, lo, hi)`: NaN propagates.
fn np_clip<F: Float>(x: F, lo: F, hi: F) -> F {
    if x.is_nan() {
        x
    } else {
        x.max(lo).min(hi)
    }
}

/// numpy's pairwise summation (`pairwise_sum` of `loops_utils.h`, used by `add.reduce` over a contiguous
/// run): fewer than 8 elements are summed in order, up to 128 with eight interleaved accumulators, more by
/// halving (the split rounded down to a multiple of 8). Accumulates in `F` (`f32` for a float32 array).
pub fn pairwise_sum<F: Float>(a: &[F]) -> F {
    let n = a.len();
    if n < 8 {
        let mut res = F::zero();
        for &v in a {
            res = res + v;
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

/// `a.mean()` of a contiguous float32 array: the float32 pairwise sum added to the reduction's initial 0,
/// divided by the count in float64 (numpy divides by an `intp`) and stored as float32. An empty array gives NaN.
fn mean_f32(a: &[f32]) -> f32 {
    let sum = 0.0_f32 + pairwise_sum(a);
    (f64::from(sum) / a.len() as f64) as f32
}

/// `a.mean()` of a contiguous float64 array (empty: NaN).
fn mean_f64(a: &[f64]) -> f64 {
    (0.0_f64 + pairwise_sum(a)) / a.len() as f64
}

/// `np.interp(x, xp, fp)` for one point, `xp` increasing: clamps outside the table, a node returns its value.
fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x.is_nan() {
        return x;
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    if x < xp[0] {
        return fp[0];
    }
    // The last j with xp[j] <= x (j = n - 1 when x is the last node).
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == n - 1 || x == xp[j] {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut r = slope * (x - xp[j]) + fp[j];
    if r.is_nan() {
        // numpy: if we get NaN in one direction, try the other.
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if r.is_nan() && fp[j] == fp[j + 1] {
            r = fp[j];
        }
    }
    r
}

// ---------------------------------------------------------------------------------------------------------
// scipy.ndimage.gaussian_filter1d

/// TODO(dedupe): matching::ndimage. A private copy of `scipy.ndimage.gaussian_filter1d` with the semantics
/// the surface code needs: `truncate = 4.0`, radius `int(4 sigma + 0.5)`, weights `exp(-0.5 x^2 / sigma^2)`
/// normalized to sum 1 (numpy's pairwise sum), the line extended by the boundary mode, then correlated in
/// float64 with scipy's symmetric-kernel loop (`NI_Correlate1D`: the center first, then the pairs from the
/// outermost in).
mod scipy_filters {
    use super::pairwise_sum;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Mode {
        /// scipy `'wrap'`: periodic, even when the radius exceeds the line.
        Wrap,
        /// scipy `'nearest'`: the edge value repeats.
        Nearest,
    }

    /// The `2 r + 1` weights of `gaussian_filter1d(.., sigma)` (`r = int(4 sigma + 0.5)`); `sigma > 0`.
    pub fn gaussian_weights(sigma: f64) -> Vec<f64> {
        // Python int(): truncation toward zero.
        let lw = (4.0 * sigma + 0.5) as i64;
        let sigma2 = sigma * sigma;
        let k = -0.5 / sigma2;
        let phi: Vec<f64> = (-lw..=lw).map(|x| (k * (x * x) as f64).exp()).collect();
        let sum = pairwise_sum(&phi);
        phi.into_iter().map(|v| v / sum).collect()
    }

    /// Filters one line (float64 values) with symmetric odd-length `weights`; `ext` is scratch space.
    pub fn filter_line(line: &[f64], weights: &[f64], mode: Mode, ext: &mut Vec<f64>, out: &mut [f64]) {
        let n = line.len();
        let lw = weights.len() / 2;
        ext.clear();
        for k in 0..n + 2 * lw {
            let i = k as i64 - lw as i64;
            let src = match mode {
                Mode::Wrap => i.rem_euclid(n as i64),
                Mode::Nearest => i.clamp(0, n as i64 - 1),
            };
            ext.push(line[src as usize]);
        }
        for (l, o) in out.iter_mut().enumerate() {
            let ctr = lw + l;
            let mut acc = ext[ctr] * weights[lw];
            for m in (1..=lw).rev() {
                acc += (ext[ctr - m] + ext[ctr + m]) * weights[lw - m];
            }
            *o = acc;
        }
    }
}

use scipy_filters::Mode;

// ---------------------------------------------------------------------------------------------------------
// Surface

/// A body's surface on a north-up equirectangular grid (column 0 at longitude -180, like every map here):
/// land fraction 0..1 and elevation above the sea in km (0 over the sea). Port of the frozen dataclass
/// `Surface`: `h` rows of `w` columns, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Surface {
    pub land: Vec<f32>,
    pub elevation_km: Vec<f32>,
    pub source: String,
    pub h: usize,
    pub w: usize,
}

/// A bodies-file record of a surface (`Surface.load`'s `record`): the grid size, the names of the raw land and
/// elevation files (relative to the record's base directory), and the optional `source`.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceRecord {
    pub width: usize,
    pub height: usize,
    /// Raw u8 land fraction 0..255, north-up.
    pub land: String,
    /// Raw little-endian int16 elevation, m above the sea, north-up.
    pub elevation: String,
    /// `None`: the source is the NAME of the land file.
    pub source: Option<String>,
}

/// Why a surface record could not be loaded.
#[derive(Clone, Debug, PartialEq)]
pub enum SurfaceError {
    /// Python `FileNotFoundError(f'{land_path}: surface file not found')`: it names the LAND file even when
    /// the elevation file is the missing one.
    FileNotFound(String),
    /// numpy's `reshape` `ValueError`: a file whose size does not match `width x height`.
    SizeMismatch { what: &'static str, expected: usize, got: usize },
}

impl std::fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SurfaceError::FileNotFound(p) => write!(f, "{p}: surface file not found"),
            SurfaceError::SizeMismatch { what, expected, got } => {
                write!(f, "surface {what} file holds {got} bytes, expected {expected}")
            }
        }
    }
}

impl std::error::Error for SurfaceError {}

impl Surface {
    /// A surface from its fields (panics when a field's length is not `h * w`).
    pub fn new(land: Vec<f32>, elevation_km: Vec<f32>, h: usize, w: usize, source: &str) -> Surface {
        assert_eq!(land.len(), h * w, "land size does not match h x w");
        assert_eq!(elevation_km.len(), h * w, "elevation size does not match h x w");
        Surface { land, elevation_km, source: source.to_string(), h, w }
    }

    /// Port of `Surface.from_elevation`: land = `e > 0` and elevation = `max(e, 0)` after `e = elevation -
    /// sea_level`. The elevation is cast to float32 FIRST and the sea level is `np.float32(sea_level_km)`, so
    /// the subtraction is a float32 one (a float64 subtraction would call `float32(0.3)` land at sea level 0.3).
    /// `T` is the dtype of the input array (`f32`, `f64`, integers...).
    pub fn from_elevation<T: Copy + Into<f64>>(elevation_km: &[T], h: usize, w: usize, sea_level_km: f64, source: &str) -> Surface {
        assert_eq!(elevation_km.len(), h * w, "elevation size does not match h x w");
        let sea = sea_level_km as f32;
        let mut land = Vec::with_capacity(elevation_km.len());
        let mut elev = Vec::with_capacity(elevation_km.len());
        for &v in elevation_km {
            let e = v.into() as f32 - sea;
            land.push(if e > 0.0 { 1.0 } else { 0.0 });
            elev.push(np_maximum(e, 0.0));
        }
        Surface { land, elevation_km: elev, source: source.to_string(), h, w }
    }

    /// Port of the decoding in `Surface.load`, from the raw bytes of the two files: land = `u8 / 255.0` and
    /// elevation = `max(int16 / 1000.0, 0)` (little-endian), both float32 divisions. A count of values that
    /// does not match `width x height` is rejected (numpy's `reshape` raises `ValueError`).
    ///
    /// `np.fromfile(path, dtype='<i2')` reads `len / 2` values and silently DROPS a trailing odd byte, so an
    /// elevation file of `2 n + 1` bytes still loads (the extra byte is ignored); the land file is bytes, so
    /// it must hold exactly `n`.
    pub fn from_raw(width: usize, height: usize, land_u8: &[u8], elevation_i16_le: &[u8], source: &str) -> Result<Surface, SurfaceError> {
        let n = width * height;
        if land_u8.len() != n {
            return Err(SurfaceError::SizeMismatch { what: "land", expected: n, got: land_u8.len() });
        }
        if elevation_i16_le.len() / 2 != n {
            return Err(SurfaceError::SizeMismatch { what: "elevation", expected: 2 * n, got: elevation_i16_le.len() });
        }
        let land = land_u8.iter().map(|&v| f32::from(v) / 255.0).collect();
        // chunks_exact(2) skips the trailing odd byte, as np.fromfile does.
        let elevation = elevation_i16_le
            .chunks_exact(2)
            .map(|b| {
                let e = f32::from(i16::from_le_bytes([b[0], b[1]])) / 1000.0;
                np_maximum(e, 0.0)
            })
            .collect();
        Ok(Surface { land, elevation_km: elevation, source: source.to_string(), h: height, w: width })
    }

    /// Port of `Surface.load`: a body's surface from its bodies-file record, `None` when the body has no
    /// surface (Python: a `None` or empty record), an error when the record names files that do not exist.
    /// File I/O is not part of the core: `read_file` maps a name of the record to the bytes of that file
    /// (`None` when it does not exist); the caller joins it to the record's base directory. The sizes are
    /// checked as [`Surface::from_raw`] does (a trailing odd byte of the elevation file is ignored).
    pub fn load(record: Option<&SurfaceRecord>, read_file: impl Fn(&str) -> Option<Vec<u8>>) -> Result<Option<Surface>, SurfaceError> {
        let Some(rec) = record else {
            return Ok(None);
        };
        let (land, elev) = (read_file(&rec.land), read_file(&rec.elevation));
        let (Some(land), Some(elev)) = (land, elev) else {
            return Err(SurfaceError::FileNotFound(rec.land.clone()));
        };
        let default_source = std::path::Path::new(&rec.land)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let source = rec.source.as_deref().unwrap_or(&default_source);
        Surface::from_raw(rec.width, rec.height, &land, &elev, source).map(Some)
    }

    /// Port of `Surface.at`: resampled to `width` x `width / 2`, box average down, bilinear up. The same width
    /// returns the surface itself (no copy). The box branch needs `w % width == 0` AND `h % (width / 2) == 0`
    /// (and the blocks to tile the map: numpy's `reshape` raises otherwise); every other case samples
    /// bilinearly through `sample_equirect` at the texel-center directions (longitude wraps, latitude clamps, no
    /// prefiltering: a downsample aliases). Panics for `width < 2` when the width differs (Python:
    /// `ZeroDivisionError`).
    pub fn at(&self, width: usize) -> Cow<'_, Surface> {
        let (h, w) = (self.h, self.w);
        if w == width {
            return Cow::Borrowed(self);
        }
        if w % width == 0 && h % (width / 2) == 0 {
            let f = w / width;
            assert!(h % f == 0, "cannot reshape the {h} x {w} surface into {f} x {f} blocks");
            // box(a) = a.reshape(h // f, f, w // f, f).mean(axis=(1, 3)) in float32: numpy sums each block
            // row pairwise, accumulates the rows from a 0 start, and divides by f * f in float64.
            let boxed = |a: &[f32]| -> Vec<f32> {
                let (oh, ow) = (h / f, w / f);
                let mut out = Vec::with_capacity(oh * ow);
                for bi in 0..oh {
                    for bj in 0..ow {
                        let mut acc = 0.0_f32;
                        for r in 0..f {
                            let row = (bi * f + r) * w + bj * f;
                            acc += pairwise_sum(&a[row..row + f]);
                        }
                        out.push((f64::from(acc) / (f * f) as f64) as f32);
                    }
                }
                out
            };
            return Cow::Owned(Surface {
                land: boxed(&self.land),
                elevation_km: boxed(&self.elevation_km),
                source: self.source.clone(),
                h: h / f,
                w: w / f,
            });
        }
        let hh = width / 2;
        let dirs = grid_dirs::<f64>(width, hh);
        let sample = |a: &[f32]| -> Vec<f32> {
            let img: Vec<f64> = a.iter().map(|&v| f64::from(v)).collect();
            sample_equirect::<f64>(&img, h, w, 1, &dirs).into_iter().map(|v| v as f32).collect()
        };
        Cow::Owned(Surface { land: sample(&self.land), elevation_km: sample(&self.elevation_km), source: self.source.clone(), h: hh, w: width })
    }
}

// ---------------------------------------------------------------------------------------------------------
// Scalar shapes

/// Port of `_gauss(x, mu, s) = exp(-(((x - mu) / s) ** 2))`, run in the dtype of `x` (`mu` and `s` are weak
/// scalars rounded to it).
pub fn gauss<F: Float>(x: F, mu: f64, s: f64) -> F {
    let t = (x - c::<F>(mu)) / c::<F>(s);
    (-(t * t)).exp()
}

/// Port of `_smoothstep(a, b, x)`: `k = clip((x - a) / (b - a), 0, 1)`, `k k (3 - 2 k)`, in the dtype of `x`
/// (`b - a` is a Python float subtraction, rounded to that dtype after).
pub fn smoothstep<F: Float>(a: f64, b: f64, x: F) -> F {
    let k = np_clip((x - c::<F>(a)) / c::<F>(b - a), F::zero(), F::one());
    k * k * (c::<F>(3.0) - c::<F>(2.0) * k)
}

/// Port of `land_contrast`: ocean minus land cloud amount at `x = |lat| / phi_H` (decision record 1).
pub fn land_contrast<F: Float>(x: F) -> F {
    c::<F>(LAND_CONTRAST_CONVERGENCE) * gauss(x, CONVERGENCE_GAUSS.0, CONVERGENCE_GAUSS.1)
        + c::<F>(LAND_CONTRAST_EXTRATROPICS)
            * smoothstep(EXTRATROPICS_RISE.0, EXTRATROPICS_RISE.1, x)
            * (F::one() - smoothstep(EXTRATROPICS_FADE.0, EXTRATROPICS_FADE.1, x))
        + c::<F>(LAND_CONTRAST_DESERT) * gauss(x, DESERT_GAUSS.0, DESERT_GAUSS.1)
}

/// Port of `deck_window`: the latitude window of the marine stratocumulus.
pub fn deck_window<F: Float>(x: F) -> F {
    smoothstep(WINDOW_RISE.0, WINDOW_RISE.1, x) * (F::one() - smoothstep(WINDOW_FALL.0, WINDOW_FALL.1, x))
}

/// Port of `earth_tables_at` for one `x`: Earth's zonal means of continentality and of the windowed deck
/// strength at `x`, `np.interp` into the two tables (clamped outside them). Map it over an array of `x`.
pub fn earth_tables_at(x: f64) -> (f64, f64) {
    (np_interp(x, &EARTH_TABLE_X, &EARTH_CONTINENTALITY), np_interp(x, &EARTH_TABLE_X, &EARTH_DECK))
}

/// `x = np.abs(np.degrees(lat)) / hadley_deg` of every row of a `width x height` grid (the expression sits
/// inline in `surface_terms` and `earth_tables`): the latitude of a row in units of the Hadley edge.
pub fn hadley_x(width: usize, height: usize, hadley_deg: f64) -> Vec<f64> {
    let (_, lat) = grid_lonlat(width, height);
    lat.iter().map(|&la| la.to_degrees().abs() / hadley_deg).collect()
}

// ---------------------------------------------------------------------------------------------------------
// Smoothing and the coast

/// Port of `smooth_on_sphere`: Gaussian smoothing of a north-up equirectangular field at a physical scale,
/// along latitude with the row's own sigma (`sigma / cos(lat)`, wrapped, capped at a quarter turn), then along
/// longitude lines (mode `nearest`). Not exact near the poles. Each row is cast to float32 first (as Python's
/// `a[i].astype(np.float32)`), and the result is float32. `T` is the dtype of `a`.
pub fn smooth_on_sphere<T: Copy + Into<f64>>(a: &[T], h: usize, w: usize, sigma_km: f64, radius_km: f64) -> Vec<f32> {
    assert_eq!(a.len(), h * w, "field size does not match h x w");
    let (_, lat) = grid_lonlat(w, h);
    let px_km = 2.0 * PI * radius_km / w as f64;
    let mut out = vec![0.0_f32; h * w];
    let (mut ext, mut line, mut filtered) = (Vec::new(), vec![0.0_f64; w], vec![0.0_f64; w]);
    for i in 0..h {
        let s = py_min(w as f64 * ROW_MAX_SIGMA_WIDTH_FRACTION, sigma_km / (px_km * py_max(lat[i].cos(), COS_FLOOR)));
        let row = &a[i * w..(i + 1) * w];
        if s > ROW_MIN_SIGMA_PX {
            for (l, &v) in line.iter_mut().zip(row) {
                *l = f64::from(v.into() as f32);
            }
            let weights = scipy_filters::gaussian_weights(s);
            scipy_filters::filter_line(&line, &weights, Mode::Wrap, &mut ext, &mut filtered);
            for (o, &v) in out[i * w..(i + 1) * w].iter_mut().zip(&filtered) {
                *o = v as f32;
            }
        } else {
            for (o, &v) in out[i * w..(i + 1) * w].iter_mut().zip(row) {
                *o = v.into() as f32;
            }
        }
    }
    // gaussian_filter1d(out, sigma_km / px_km, axis=0, mode='nearest')
    let weights = scipy_filters::gaussian_weights(sigma_km / px_km);
    let (mut col, mut colf) = (vec![0.0_f64; h], vec![0.0_f64; h]);
    for j in 0..w {
        for i in 0..h {
            col[i] = f64::from(out[i * w + j]);
        }
        scipy_filters::filter_line(&col, &weights, Mode::Nearest, &mut ext, &mut colf);
        for i in 0..h {
            out[i * w + j] = colf[i] as f32;
        }
    }
    out
}

/// Port of `continentality`: the land fraction smoothed at [`CONTINENTALITY_KM`] and clipped to [0, 1]
/// (float32, `surface.h` x `surface.w`).
pub fn continentality(surface: &Surface, radius_km: f64) -> Vec<f32> {
    let smooth = smooth_on_sphere(&surface.land, surface.h, surface.w, CONTINENTALITY_KM, radius_km);
    smooth.into_iter().map(|v| np_clip(v, 0.0, 1.0)).collect()
}

/// Port of `deck_strength`: the 0..1 marine stratocumulus strength from the coast geometry alone (no latitude
/// window) of the smoothed land `cont` (`h` rows of `w`, float32): an ocean texel with a continent's coast
/// (`cont >= 0.5`, tested on the float32 values) DOWNSTREAM along its row (to the east on a prograde body, the
/// row mirrored for `spin < 0`; a spin of 0 counts as prograde), `exp(-distance / DECK_EFOLD_KM)`; a distance
/// above [`DECK_MAX_SEARCH_KM`] (strict) counts as infinite. The exp is float64 rounded to float32, then a
/// float32 product with `1 - cont`. Rows with no coast or only coast are zeros.
pub fn deck_strength(cont: &[f32], h: usize, w: usize, radius_km: f64, spin: f64) -> Vec<f32> {
    assert_eq!(cont.len(), h * w, "field size does not match h x w");
    let (_, lat) = grid_lonlat(w, h);
    let coast_threshold = DECK_COAST_LAND as f32; // float32 array against a weak Python float
    let px_km = 2.0 * PI * radius_km / w as f64;
    // Python `spin >= 0` is False for a NaN spin, which then mirrors: keep the negated comparison.
    let mirrored = !(spin >= 0.0);
    let mut out = vec![0.0_f32; h * w];
    let mut ext: Vec<i64> = Vec::with_capacity(2 * w);
    for i in 0..h {
        let src = &cont[i * w..(i + 1) * w];
        let row: Vec<bool> = if mirrored { src.iter().rev().map(|&v| v >= coast_threshold).collect() } else { src.iter().map(|&v| v >= coast_threshold).collect() };
        if !row.iter().any(|&b| b) || row.iter().all(|&b| b) {
            continue;
        }
        // Columns to the next coast cell eastward (westward when mirrored), wrapped: ext = c ++ (c + w).
        let coast_cols: Vec<i64> = (0..w).filter(|&j| row[j]).map(|j| j as i64).collect();
        ext.clear();
        ext.extend(coast_cols.iter().copied());
        ext.extend(coast_cols.iter().map(|&j| j + w as i64));
        let dx = px_km * py_max(lat[i].cos(), COS_FLOOR);
        let mut v = vec![0.0_f32; w];
        for (j, vj) in v.iter_mut().enumerate() {
            // np.searchsorted(ext, j), side = 'left'
            let k = ext.partition_point(|&e| e < j as i64);
            let mut dist = (ext[k] - j as i64) as f64;
            if dist * dx > DECK_MAX_SEARCH_KM {
                dist = f64::INFINITY;
            }
            *vj = ((-dist * dx / DECK_EFOLD_KM).exp()) as f32;
        }
        if mirrored {
            v.reverse();
        }
        out[i * w..(i + 1) * w].copy_from_slice(&v);
    }
    for (o, &cv) in out.iter_mut().zip(cont) {
        *o *= 1.0_f32 - cv;
    }
    out
}

// ---------------------------------------------------------------------------------------------------------
// ITCZ, Earth tables

/// Port of `hemispheric_land_asymmetry`: the area-weighted (`cos lat`) land fraction north minus south. The
/// row means are float32 pairwise means, the weighted sums float64; the equator row of an odd height belongs
/// to neither hemisphere. A one-row map has no hemisphere: 0/0 gives NaN (numpy warns).
pub fn hemispheric_land_asymmetry(surface: &Surface) -> f64 {
    let (h, w) = (surface.h, surface.w);
    let (_, lat) = grid_lonlat(w, h);
    let cosl: Vec<f64> = lat.iter().map(|l| l.cos()).collect();
    let rows: Vec<f32> = (0..h).map(|i| mean_f32(&surface.land[i * w..(i + 1) * w])).collect();
    // (rows[sel] * c[sel]).sum() / c[sel].sum()
    let side = |sel: &dyn Fn(f64) -> bool| -> f64 {
        let (prod, wts): (Vec<f64>, Vec<f64>) =
            (0..h).filter(|&i| sel(lat[i])).map(|i| (f64::from(rows[i]) * cosl[i], cosl[i])).unzip();
        pairwise_sum(&prod) / pairwise_sum(&wts)
    };
    side(&|l| l > 0.0) - side(&|l| l < 0.0)
}

/// Port of `itcz_shift_deg`: the northward displacement of the convergence zone in degrees (decision record
/// 4); 0 with no surface. The x shift is clipped to +-[`ITCZ_SHIFT_MAX_X`] before the Hadley edge multiplies.
pub fn itcz_shift_deg(surface: Option<&Surface>, hadley_deg: f64) -> f64 {
    let Some(surface) = surface else {
        return 0.0;
    };
    let x = EARTH_ITCZ_SHIFT_X * hemispheric_land_asymmetry(surface) / EARTH_LAND_ASYMMETRY;
    np_clip(x, -ITCZ_SHIFT_MAX_X, ITCZ_SHIFT_MAX_X) * hadley_deg
}

/// Port of `earth_tables`: Earth's zonal means of continentality and of the windowed deck strength, both
/// hemispheres averaged, in the 15 half-open bands `[k/6, (k+1)/6)` of `x = |lat| / hadley_deg` (a row with
/// `x >= 15/6` is in no band; an empty band is NaN): the centering tables. Run on Earth's DEM only. The
/// continentality means are float32 means of float32 values; the deck is float64 (`deck_window` is float64).
pub fn earth_tables(surface: &Surface, radius_km: f64, spin: f64, hadley_deg: f64, width: usize) -> ([f64; 15], [f64; 15]) {
    let s = surface.at(width);
    let h = width / 2;
    assert!(s.h == h && s.w == width, "the surface is not width x width/2 after at()");
    let x = hadley_x(width, h, hadley_deg);
    let cont = continentality(&s, radius_km);
    let strength = deck_strength(&cont, h, width, radius_km, spin);
    // deck = deck_window(x[:, None]) * strength: float64 times float32 is float64.
    let mut deck = vec![0.0_f64; h * width];
    for i in 0..h {
        let dw = deck_window(x[i]);
        for j in 0..width {
            deck[i * width + j] = dw * f64::from(strength[i * width + j]);
        }
    }
    let (mut rc, mut rd) = ([0.0_f64; 15], [0.0_f64; 15]);
    for k in 0..15 {
        let (a, b) = (EARTH_BAND_EDGES_X[k], EARTH_BAND_EDGES_X[k + 1]);
        let mut sel_cont: Vec<f32> = Vec::new();
        let mut sel_deck: Vec<f64> = Vec::new();
        for i in (0..h).filter(|&i| x[i] >= a && x[i] < b) {
            sel_cont.extend_from_slice(&cont[i * width..(i + 1) * width]);
            sel_deck.extend_from_slice(&deck[i * width..(i + 1) * width]);
        }
        rc[k] = f64::from(mean_f32(&sel_cont));
        rd[k] = mean_f64(&sel_deck);
    }
    (rc, rd)
}

// ---------------------------------------------------------------------------------------------------------
// Relief

/// A wind grid: `h` rows of `w` columns of (east, north) m/s, interleaved (`h * w * 2` values), north-up.
#[derive(Clone, Copy, Debug)]
pub struct WindGrid<'a> {
    pub uv: &'a [f32],
    pub h: usize,
    pub w: usize,
}

/// Port of `orographic_anomaly`: the coverage anomaly of decision record 3 at the wind grid. `wind` is `h x w`
/// (east, north) m/s; the surface is resampled to the wind's width by [`Surface::at`] (and must then have the
/// wind's `h` rows).
///
/// The float32/float64 mix of the Python chain is kept: `dh/dx` and the east product are float64, `dh/dy` is
/// float32 (`float32 / weak float`) and so is the north product, `lift`, `up` and the wind speed are float32,
/// the back-trace offsets are float32 (a weak `d` times a float32 wind over a float32 speed) subtracted from
/// the float64 grid, and the shadow is float64 (the first `np.maximum` with a float64 sample promotes it).
pub fn orographic_anomaly(surface: &Surface, wind: &[f32], h: usize, w: usize, radius_km: f64) -> Vec<f32> {
    assert_eq!(wind.len(), h * w * 2, "wind size does not match h x w x 2");
    let s = surface.at(w);
    assert!(s.h == h && s.w == w, "the surface at the wind width is {} x {}, the wind {h} x {w}", s.h, s.w);
    let (lon, lat) = grid_lonlat(w, h);
    let dy_km = PI * radius_km / h as f64;
    let two_dy = (2.0 * dy_km) as f32;
    let e = &s.elevation_km;
    let n = h * w;
    // dx_km, per row: (2 pi R / w) * max(cos lat, 1e-3).
    let cos_max: Vec<f64> = lat.iter().map(|l| py_max(l.cos(), COS_FLOOR)).collect();
    let two_dx: Vec<f64> = cos_max.iter().map(|&cm| 2.0 * ((2.0 * PI * radius_km / w as f64) * cm)).collect();

    let mut lift = vec![0.0_f32; n];
    let mut up = vec![0.0_f32; n];
    for i in 0..h {
        for j in 0..w {
            let k = i * w + j;
            let (u, v) = (wind[2 * k], wind[2 * k + 1]);
            // np.roll(e, -1, axis=1) - np.roll(e, 1, axis=1): float32, then / float64 (2 dx).
            let dhdx = f64::from(e[i * w + (j + 1) % w] - e[i * w + (j + w - 1) % w]) / two_dx[i];
            // dhdy is zero at the two pole rows; row - 1 is north. float32 / weak float stays float32.
            let dhdy = if i >= 1 && i + 1 < h { (e[(i - 1) * w + j] - e[(i + 1) * w + j]) / two_dy } else { 0.0_f32 };
            // wind_u * dhdx is float64, wind_v * dhdy float32; the sum is float64 (km/km x m/s = m/s).
            let w_mps = f64::from(u) * dhdx + f64::from(v * dhdy);
            lift[k] = (w_mps / OROGRAPHIC_W0_MPS).tanh() as f32;
            up[k] = np_maximum(lift[k], 0.0_f32);
        }
    }
    // The shadow: the strongest uplift the air crossed within RAIN_SHADOW_KM upstream, where the air has come
    // DOWN from higher ground since.
    let speed: Vec<f32> = (0..n).map(|k| np_maximum(wind[2 * k].hypot(wind[2 * k + 1]), SPEED_FLOOR_MPS as f32)).collect();
    let e64: Vec<f64> = e.iter().map(|&v| f64::from(v)).collect();
    let up64: Vec<f64> = up.iter().map(|&v| f64::from(v)).collect();
    let mut shadow = vec![0.0_f64; n];
    let mut dirs: Vec<[f64; 3]> = vec![[0.0; 3]; n];
    for frac in SHADOW_FRACTIONS {
        let d = RAIN_SHADOW_KM * frac / radius_km; // rad
        let d32 = d as f32; // weak scalar times a float32 wind
        for i in 0..h {
            for j in 0..w {
                let k = i * w + j;
                let (u, v) = (wind[2 * k], wind[2 * k + 1]);
                let la2 = lat[i] - f64::from(d32 * v / speed[k]);
                let lo2 = lon[j] - f64::from(d32 * u / speed[k]) / cos_max[i];
                dirs[k] = [la2.cos() * lo2.cos(), la2.cos() * lo2.sin(), la2.sin()];
            }
        }
        let ground = sample_equirect::<f64>(&e64, h, w, 1, &dirs);
        let uplift = sample_equirect::<f64>(&up64, h, w, 1, &dirs);
        for k in 0..n {
            let descent = smoothstep(SHADOW_DESCENT_KM.0, SHADOW_DESCENT_KM.1, ground[k] - e64[k]);
            shadow[k] = np_maximum(shadow[k], uplift[k] * descent * (1.0 - SHADOW_WEIGHT_PER_FRACTION * frac));
        }
    }
    (0..n)
        .map(|k| {
            let shade = shadow[k] * f64::from(1.0_f32 - up[k]); // (1.0 - up) is float32
            (f64::from(OROGRAPHIC_AMPLITUDE as f32 * lift[k]) - RAIN_SHADOW_AMPLITUDE * shade) as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------
// The terms

/// Port of the dataclass `SurfaceTerms`: the climatological coverage anomaly of one body at one grid
/// (`h` x `w`, north-up, same as the map), its parts for the tests and the sidecar, and the ITCZ shift.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceTerms {
    pub anomaly: Vec<f32>,
    pub land: Vec<f32>,
    pub deck: Vec<f32>,
    pub orographic: Vec<f32>,
    pub continentality: Vec<f32>,
    pub deck_strength: Vec<f32>,
    pub itcz_shift_deg: f64,
    pub h: usize,
    pub w: usize,
}

/// Port of `_resample`: an `h x w` array resampled to `width` x `width / 2` by `sample_equirect` at the texel
/// centers (bilinear, longitude wraps, latitude clamps, no prefiltering). The same width returns the array
/// unchanged (Python returns the object itself, whatever its dtype); otherwise the values are float32 (Python
/// casts to float32 before and after), which is exact for `T = f32`.
pub fn resample<T: Float>(a: &[T], h: usize, w: usize, width: usize) -> Vec<T> {
    assert_eq!(a.len(), h * w, "array size does not match h x w");
    if w == width {
        return a.to_vec();
    }
    let dirs = grid_dirs::<f64>(width, width / 2);
    let img: Vec<f64> = a.iter().map(|&v| f64::from(v.to_f32().expect("float conversion"))).collect();
    sample_equirect::<f64>(&img, h, w, 1, &dirs).into_iter().map(|v| T::from(v as f32).expect("float conversion")).collect()
}

/// Port of `surface_terms`: the four terms at `width` x `width / 2`. The smooth terms are computed at
/// `min(term_width, width)` (~40 km on Earth: they are climatological) and upsampled. `wind` is the
/// generator's own wind product (east, north m/s) on its own grid; without it the orographic term is zero.
///
/// numpy promotions kept: `land = land_contrast(x) * (EARTH_cont(x) - cont)` and `deck = 0.35 (window(x)
/// strength - EARTH_deck(x))` are float64 (a float64 column against a float32 map), `total` too; the parts are
/// stored as float32.
pub fn surface_terms(surface: &Surface, width: usize, hadley_deg: f64, radius_km: f64, spin: f64, wind: Option<&WindGrid>, term_width: usize) -> SurfaceTerms {
    let tw = term_width.min(width);
    let s = surface.at(tw);
    let h = tw / 2;
    assert!(s.h == h && s.w == tw, "the surface is not tw x tw/2 after at()");
    let x = hadley_x(tw, h, hadley_deg);
    let cont = continentality(&s, radius_km);
    let strength = deck_strength(&cont, h, tw, radius_km, spin);
    let n = h * tw;
    let (mut land, mut deck) = (vec![0.0_f64; n], vec![0.0_f64; n]);
    for i in 0..h {
        let (ref_cont, ref_deck) = earth_tables_at(x[i]);
        let (lc, dw) = (land_contrast(x[i]), deck_window(x[i]));
        for j in 0..tw {
            let k = i * tw + j;
            land[k] = lc * (ref_cont - f64::from(cont[k]));
            deck[k] = DECK_AMPLITUDE * (dw * f64::from(strength[k]) - ref_deck);
        }
    }
    let oro: Vec<f32> = match wind {
        None => vec![0.0; n],
        Some(wg) => {
            let o = orographic_anomaly(&s, wg.uv, wg.h, wg.w, radius_km);
            let r = resample(&o, wg.h, wg.w, tw);
            assert_eq!(r.len(), n, "the wind grid does not match the term grid");
            r
        }
    };
    let total: Vec<f64> = (0..n).map(|k| land[k] + deck[k] + f64::from(oro[k])).collect();
    let shift = itcz_shift_deg(Some(surface), hadley_deg);
    let f32s = |a: &[f64]| -> Vec<f32> { a.iter().map(|&v| v as f32).collect() };
    if tw != width {
        let up = |a: &[f32]| resample(a, h, tw, width);
        return SurfaceTerms {
            anomaly: up(&f32s(&total)),
            land: up(&f32s(&land)),
            deck: up(&f32s(&deck)),
            orographic: up(&oro),
            continentality: up(&cont),
            deck_strength: up(&strength),
            itcz_shift_deg: shift,
            h: width / 2,
            w: width,
        };
    }
    SurfaceTerms {
        anomaly: f32s(&total),
        land: f32s(&land),
        deck: f32s(&deck),
        orographic: oro,
        continentality: cont,
        deck_strength: strength,
        itcz_shift_deg: shift,
        h,
        w: tw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(h: usize, w: usize, f: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        (0..h).flat_map(|i| (0..w).map(move |j| (i, j))).map(|(i, j)| f(i, j)).collect()
    }

    // The references below were computed with numpy 2.5 and scipy 1.18 (float32 arrays built from the same
    // expressions), the ones the goldens cannot pin because they sit inside the golden tolerances.

    #[test]
    fn pairwise_sum_matches_the_numpy_bits() {
        // np.arange(200).astype(f32) * f32(0.1): .sum() and .mean().
        let a: Vec<f32> = (0..200).map(|i| i as f32 * 0.1_f32).collect();
        assert_eq!(pairwise_sum(&a).to_bits(), 1157152768);
        assert_eq!(mean_f32(&a).to_bits(), 1092563763);
        // np.arange(37).astype(f32) * f32(0.37) + f32(0.01): fewer than 128 elements, with a remainder.
        let b: Vec<f32> = (0..37).map(|i| i as f32 * 0.37_f32 + 0.01_f32).collect();
        assert_eq!(pairwise_sum(&b).to_bits(), 1131858495);
        // np.full(1000, 0.1, f32).sum() halves the run: 100.00001525878906; a sequential float32 loop gives
        // 99.99905.
        let c = vec![0.1_f32; 1000];
        assert_eq!(pairwise_sum(&c), 100.000_015_f32);
        assert_ne!(c.iter().fold(0.0_f32, |s, &x| s + x), pairwise_sum(&c));
        // Eight elements already use the eight accumulators (a sequential sum would give 16777216).
        let d = [16777216.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(pairwise_sum(&d), 16777222.0);
        // Seven elements are summed in order.
        let e = [16777216.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(pairwise_sum(&e), 16777216.0);
        assert_eq!(pairwise_sum::<f32>(&[]), 0.0);
        // Any run of exactly representable values sums exactly, whatever the size class.
        for n in [1usize, 7, 8, 9, 15, 16, 127, 128, 129, 300, 1000] {
            assert_eq!(pairwise_sum(&vec![1.0_f32; n]), n as f32);
        }
    }

    /// `((i * 7919) % 1009) * 0.37 + 0.01`, times 1000 on every fifth element (a wide dynamic range, so that the
    /// association order of the sum shows in the last bits), in float32.
    fn sum_data_f32(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = ((i * 7919) % 1009) as f32 * 0.37_f32 + 0.01_f32;
                if i % 5 == 0 {
                    v * 1000.0_f32
                } else {
                    v
                }
            })
            .collect()
    }

    /// The same data in float64.
    fn sum_data_f64(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let v = ((i * 7919) % 1009) as f64 * 0.37 + 0.01;
                if i % 5 == 0 {
                    v * 1000.0
                } else {
                    v
                }
            })
            .collect()
    }

    #[test]
    fn pairwise_sum_switches_algorithm_exactly_at_the_numpy_sizes() {
        // Bits of numpy 2.5.4 `a.sum()` (a contiguous array of the data above) around every boundary of
        // numpy's pairwise_sum: 127 and 128 elements use the eight accumulators (a 128-element run is NOT
        // split in halves: the limit is `n <= 128`), 129 is the first run that halves (64 + 65), 136 and 257
        // halve with a split rounded down to a multiple of 8 (64, 128), and 255, 256 and 300 recurse deeper.
        // For these data a plain sequential float32 sum differs from numpy's by 1 to 2 ulp at every size (127:
        // 1250955154 against 1250955156; 128: 1250955708 against 1250955710), so a wrong split or limit shows.
        let f32_sums: [(usize, u32); 9] = [
            (127, 1250955156),
            (128, 1250955710),
            (129, 1250956150),
            (136, 1251568244),
            (255, 1259395782),
            (256, 1259520111),
            (257, 1259520179),
            (300, 1260785346),
            (1000, 1276077892),
        ];
        for (n, bits) in f32_sums {
            assert_eq!(pairwise_sum(&sum_data_f32(n)).to_bits(), bits, "float32, n = {n}");
        }
        // mean = (0 + sum) / n in float64, stored as float32.
        assert_eq!(mean_f32(&sum_data_f32(128)).to_bits(), 1192235454);
        assert_eq!(mean_f32(&sum_data_f32(1000)).to_bits(), 1192417366);
        let f64_sums: [(usize, u64); 9] = [
            (127, 4706826701156552539),
            (128, 4706826998733361643),
            (129, 4706827235525646090),
            (136, 4707155851043342910),
            (255, 4711358227702811197),
            (256, 4711424976863300157),
            (257, 4711425013220198318),
            (300, 4712104244559845785),
            (1000, 4720314368029318185),
        ];
        for (n, bits) in f64_sums {
            assert_eq!(pairwise_sum(&sum_data_f64(n)).to_bits(), bits, "float64, n = {n}");
        }
    }

    #[test]
    fn a_128_element_run_uses_the_eight_accumulators_not_a_halving() {
        // numpy's limit is `n <= 128`: a 128-element float32 run is summed with eight interleaved accumulators,
        // NOT split into 64 + 64. These three data sets (`((i * k) % m) * scale + 0.01`, float32, 128 elements)
        // were picked so that the two algorithms give different bits (they differ by 1 or 2 ulp); the bits are
        // numpy 2.5.4's `a.sum()`, which is the eight-accumulator one.
        let sets: [(usize, usize, f32, u32); 3] = [(7919, 257, 0.37, 1169885779), (977, 1009, 0.37, 1186490721), (31, 4099, 0.013, 1162661332)];
        for (k, m, scale, bits) in sets {
            let a: Vec<f32> = (0..128).map(|i| ((i * k) % m) as f32 * scale + 0.01_f32).collect();
            assert_eq!(pairwise_sum(&a).to_bits(), bits, "k = {k}, m = {m}, scale = {scale}");
            // The halving, written out: 64 + 64 gives a different float32 (the algorithm the limit must not pick).
            let halved = pairwise_sum(&a[..64]) + pairwise_sum(&a[64..]);
            assert_ne!(halved.to_bits(), bits, "the data must tell the two algorithms apart");
        }
    }

    #[test]
    fn means_of_an_empty_slice_are_nan_and_the_division_is_the_float32_one() {
        assert!(mean_f32(&[]).is_nan());
        assert!(mean_f64(&[]).is_nan());
        assert_eq!(mean_f32(&[1.0, 2.0, 4.0]), 7.0_f32 / 3.0);
        assert_eq!(mean_f64(&[1.0, 2.0, 4.0]), 7.0 / 3.0);
    }

    #[test]
    fn the_float32_block_mean_and_row_mean_match_numpy() {
        // blk = np.arange(128).astype(f32).reshape(8, 16) * f32(0.013): blk.reshape(2, 4, 4, 4).mean(axis=(1, 3))
        // and blk.mean(axis=1).
        let land = grid(8, 16, |i, j| (i * 16 + j) as f32 * 0.013_f32);
        let s = Surface::new(land.clone(), land.clone(), 8, 16, "");
        let half = s.at(4);
        assert_eq!((half.h, half.w), (2, 4));
        let bits: Vec<u32> = half.land.iter().map(|v| v.to_bits()).collect();
        assert_eq!(bits, vec![1051310686, 1053055517, 1054800347, 1056545178, 1066724754, 1067160961, 1067597169, 1068033376]);
        assert_eq!(half.elevation_km, half.land);
        let rows: Vec<u32> = (0..8).map(|i| mean_f32(&land[i * 16..(i + 1) * 16]).to_bits()).collect();
        assert_eq!(rows, vec![1036496405, 1050438271, 1057191100, 1060680762, 1064170422, 1066506650, 1068251480, 1069996310]);
    }

    #[test]
    fn np_interp_clamps_hits_nodes_and_interpolates() {
        let (xp, fp) = ([0.0, 1.0, 3.0], [10.0, 20.0, 40.0]);
        assert_eq!(np_interp(-5.0, &xp, &fp), 10.0);
        assert_eq!(np_interp(7.0, &xp, &fp), 40.0);
        assert_eq!(np_interp(3.0, &xp, &fp), 40.0);
        assert_eq!(np_interp(1.0, &xp, &fp), 20.0);
        assert_eq!(np_interp(2.0, &xp, &fp), 30.0);
        assert!(np_interp(f64::NAN, &xp, &fp).is_nan());
        // The Earth tables hold their end values past the first and last node.
        assert_eq!(earth_tables_at(0.0), (EARTH_CONTINENTALITY[0], EARTH_DECK[0]));
        assert_eq!(earth_tables_at(99.0), (EARTH_CONTINENTALITY[14], EARTH_DECK[14]));
    }

    // -- scipy gaussian_filter1d

    #[test]
    fn gaussian_weights_match_scipy() {
        // scipy.ndimage.gaussian_filter1d(unit impulse, 1.0): radius 4, the kernel renormalized over its 9 taps.
        let w = scipy_filters::gaussian_weights(1.0);
        assert_eq!(w.len(), 9);
        let want = [0.0001338306174147874_f64, 0.004431861452758312, 0.053991127759218216, 0.2419714480638504, 0.3989434838294983];
        for (k, &x) in want.iter().enumerate() {
            assert_eq!(w[k] as f32, x as f32, "tap {k}");
            assert_eq!(w[k], w[8 - k]);
        }
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        // A sigma under 0.125 truncates to radius 0: the identity.
        assert_eq!(scipy_filters::gaussian_weights(0.1), vec![1.0]);
    }

    #[test]
    fn filter_line_matches_scipy_in_both_modes() {
        // gaussian_filter1d(np.array([0.5, 0, 0.25, 1, 0.75, 0.125], f32), sigma, mode=...): radius 9 > the line.
        let line: Vec<f64> = [0.5_f32, 0.0, 0.25, 1.0, 0.75, 0.125].iter().map(|&v| f64::from(v)).collect();
        let (mut ext, mut out) = (Vec::new(), vec![0.0; 6]);
        let w = scipy_filters::gaussian_weights(2.3);
        scipy_filters::filter_line(&line, &w, Mode::Wrap, &mut ext, &mut out);
        let wrap = [0.42031699419021606_f64, 0.4203108847141266, 0.4374924600124359, 0.4546981453895569, 0.45468607544898987, 0.4374954402446747];
        assert_eq!(out.iter().map(|&v| v as f32).collect::<Vec<_>>(), wrap.iter().map(|&v| v as f32).collect::<Vec<_>>());
        scipy_filters::filter_line(&line, &w, Mode::Nearest, &mut ext, &mut out);
        let near = [0.42887604236602783_f64, 0.42832139134407043, 0.43512094020843506, 0.43150797486305237, 0.4005379378795624, 0.34121742844581604];
        assert_eq!(out.iter().map(|&v| v as f32).collect::<Vec<_>>(), near.iter().map(|&v| v as f32).collect::<Vec<_>>());
        // A small sigma (radius 1) and a sub-0.125 one (the identity).
        let w = scipy_filters::gaussian_weights(0.31);
        scipy_filters::filter_line(&line, &w, Mode::Wrap, &mut ext, &mut out);
        let small = [0.49523940682411194_f64, 0.00408049626275897, 0.25272032618522644, 0.9945593476295471, 0.7479597330093384, 0.13044066727161407];
        assert_eq!(out.iter().map(|&v| v as f32).collect::<Vec<_>>(), small.iter().map(|&v| v as f32).collect::<Vec<_>>());
        scipy_filters::filter_line(&line, &scipy_filters::gaussian_weights(0.1), Mode::Nearest, &mut ext, &mut out);
        assert_eq!(out, line);
    }

    #[test]
    fn filter_line_extends_by_hand() {
        let w = [0.25, 0.5, 0.25];
        let line = [1.0, 0.0, 0.0, 0.0];
        let (mut ext, mut out) = (Vec::new(), [0.0; 4]);
        scipy_filters::filter_line(&line, &w, Mode::Wrap, &mut ext, &mut out);
        assert_eq!(out, [0.5, 0.25, 0.0, 0.25]);
        scipy_filters::filter_line(&line, &w, Mode::Nearest, &mut ext, &mut out);
        assert_eq!(out, [0.75, 0.25, 0.0, 0.0]);
    }

    // -- Surface

    #[test]
    fn from_elevation_casts_to_f32_first_and_keeps_the_float32_sea_level() {
        // 1e-50 is a positive float64 that is 0 in float32: sea. -0 is sea too.
        let s = Surface::from_elevation(&[1e-50_f64, -0.0, 0.5, -1.0], 1, 4, 0.0, "t");
        assert_eq!(s.land, vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(s.elevation_km, vec![0.0, 0.0, 0.5, 0.0]);
        // sea level 0.3 is float32(0.3): the float32 elevation 0.3 is sea, the next float32 above is land.
        let up = f32::from_bits(0.3_f32.to_bits() + 1);
        let s = Surface::from_elevation(&[0.3_f32, up], 1, 2, 0.3, "");
        assert_eq!(s.land, vec![0.0, 1.0]);
        assert_eq!(s.source, "");
        // A NaN elevation is not land and stays NaN (np.maximum propagates it).
        let s = Surface::from_elevation(&[f32::NAN], 1, 1, 0.0, "");
        assert_eq!(s.land, vec![0.0]);
        assert!(s.elevation_km[0].is_nan());
    }

    #[test]
    fn load_decodes_and_rejects_a_wrong_size() {
        let land = [0u8, 255, 51, 128];
        let elev: Vec<u8> = [-5000i16, 0, 1234, 32767].iter().flat_map(|v| v.to_le_bytes()).collect();
        let s = Surface::from_raw(2, 2, &land, &elev, "x").unwrap();
        assert_eq!(s.land, vec![0.0, 1.0, 51.0 / 255.0, 128.0 / 255.0]);
        assert_eq!(s.elevation_km, vec![0.0, 0.0, 1.234, 32.767]);
        // numpy's reshape ValueError: a file whose size does not match width x height.
        assert!(matches!(Surface::from_raw(2, 2, &land[..3], &elev, ""), Err(SurfaceError::SizeMismatch { what: "land", .. })));
        assert!(matches!(Surface::from_raw(2, 2, &land, &elev[..7], ""), Err(SurfaceError::SizeMismatch { what: "elevation", .. })));
        assert!(matches!(Surface::from_raw(3, 2, &land, &elev, ""), Err(SurfaceError::SizeMismatch { .. })));
        // Too many values is as wrong as too few, for both files (a whole extra int16 is 2 bytes).
        let mut long_elev = elev.clone();
        long_elev.extend_from_slice(&[0, 0]);
        assert!(matches!(Surface::from_raw(2, 2, &land, &long_elev, ""), Err(SurfaceError::SizeMismatch { what: "elevation", .. })));
        assert!(matches!(Surface::from_raw(2, 2, &[land.as_slice(), &[0]].concat(), &elev, ""), Err(SurfaceError::SizeMismatch { what: "land", .. })));
    }

    #[test]
    fn a_trailing_odd_byte_of_the_elevation_file_is_ignored_like_np_fromfile() {
        // np.fromfile(path, dtype='<i2') on 5 bytes [1, 0, 2, 0, 9] gives [1, 2] (numpy 2.5): the odd byte is
        // dropped, so a 2 n + 1 byte elevation file loads. A land file is bytes: an extra byte is a size error.
        let land = [0u8, 255, 51, 128];
        let mut elev: Vec<u8> = [-5000i16, 0, 1234, 32767].iter().flat_map(|v| v.to_le_bytes()).collect();
        let even = Surface::from_raw(2, 2, &land, &elev, "x").unwrap();
        elev.push(9);
        let odd = Surface::from_raw(2, 2, &land, &elev, "x").unwrap();
        assert_eq!(odd, even);
        assert_eq!(odd.elevation_km, vec![0.0, 0.0, 1.234, 32.767]);
        // Through load, with a record: the same file with the stray byte still loads.
        let rec = SurfaceRecord { width: 2, height: 2, land: "l".into(), elevation: "e".into(), source: Some("x".into()) };
        let files = |name: &str| if name == "l" { Some(land.to_vec()) } else { Some(elev.clone()) };
        assert_eq!(Surface::load(Some(&rec), files).unwrap().unwrap(), even);
        // Two extra bytes are a whole extra value, and 2 n - 1 bytes are one value short: both rejected.
        elev.push(9);
        assert!(matches!(Surface::from_raw(2, 2, &land, &elev, ""), Err(SurfaceError::SizeMismatch { .. })));
        elev.truncate(7);
        assert!(matches!(Surface::from_raw(2, 2, &land, &elev, ""), Err(SurfaceError::SizeMismatch { .. })));
    }

    #[test]
    fn load_follows_the_record() {
        let rec = SurfaceRecord { width: 2, height: 1, land: "bodies/mars/land.u8".into(), elevation: "bodies/mars/elev.i16".into(), source: None };
        let files = |name: &str| match name {
            "bodies/mars/land.u8" => Some(vec![255u8, 0]),
            "bodies/mars/elev.i16" => Some(vec![0xe8, 0x03, 0, 0]),
            _ => None,
        };
        // No record: no surface.
        assert_eq!(Surface::load(None, files), Ok(None));
        // The default source is the NAME of the land file, a given one is kept (even an empty one).
        let s = Surface::load(Some(&rec), files).unwrap().unwrap();
        assert_eq!((s.source.as_str(), s.land.clone(), s.elevation_km.clone()), ("land.u8", vec![1.0, 0.0], vec![1.0, 0.0]));
        let given = SurfaceRecord { source: Some(String::new()), ..rec.clone() };
        assert_eq!(Surface::load(Some(&given), files).unwrap().unwrap().source, "");
        // A missing file is an error naming the LAND file, even when the elevation is the missing one.
        let no_elev = |name: &str| name.ends_with("land.u8").then(|| vec![0u8, 0]);
        assert_eq!(Surface::load(Some(&rec), no_elev), Err(SurfaceError::FileNotFound("bodies/mars/land.u8".into())));
        assert_eq!(Surface::load(Some(&rec), |_: &str| None), Err(SurfaceError::FileNotFound("bodies/mars/land.u8".into())));
        // A file of the wrong size is rejected.
        let short = |name: &str| name.ends_with("land.u8").then(|| vec![0u8]).or_else(|| Some(vec![0u8; 4]));
        assert!(matches!(Surface::load(Some(&rec), short), Err(SurfaceError::SizeMismatch { .. })));
    }

    #[test]
    fn at_returns_the_same_surface_for_the_same_width_and_averages_boxes() {
        let s = Surface::new(grid(2, 4, |i, j| (i * 4 + j) as f32), vec![0.0; 8], 2, 4, "s");
        assert!(matches!(s.at(4), Cow::Borrowed(_)));
        let half = s.at(2);
        assert_eq!((half.h, half.w), (1, 2));
        // 2x2 blocks: means of (0, 1, 4, 5) and (2, 3, 6, 7).
        assert_eq!(half.land, vec![2.5, 4.5]);
        assert_eq!(half.source, "s");
    }

    #[test]
    fn at_needs_both_divisibility_tests_and_blocks_follow_the_width_ratio() {
        // 12 x 40, width 24: h % 12 == 0 but w % 24 != 0, so the box branch is refused by its FIRST test and the
        // surface is sampled bilinearly (width x width/2 = 24 x 12).
        let s = Surface::new(vec![0.5; 12 * 40], vec![0.0; 12 * 40], 12, 40, "");
        let r = s.at(24);
        assert_eq!((r.h, r.w), (12, 24));
        assert!(r.land.iter().all(|&v| (v - 0.5).abs() < 1e-6));
        // A square 16 x 16 surface at width 4: w % 4 == 0 and h % 2 == 0, so the box branch averages
        // f = w / width = 4 blocks and the result has h / f = 4 rows (not width / 2 = 2).
        let sq = Surface::new(grid(16, 16, |i, j| (i + j) as f32), vec![0.0; 256], 16, 16, "");
        let b = sq.at(4);
        assert_eq!((b.h, b.w), (4, 4));
        assert_eq!(b.land[0], (0.0 + 1.0 + 2.0 + 3.0 + 1.0 + 2.0 + 3.0 + 4.0 + 2.0 + 3.0 + 4.0 + 5.0 + 3.0 + 4.0 + 5.0 + 6.0) / 16.0);
    }


    #[test]
    fn the_strict_thresholds_hold_at_an_exact_tie() {
        // A body whose texel is exactly 1e6 km wide at w = 8: at the polar row of a 2000-row map cos(lat) is
        // under the 1e-3 floor, so dx = px_km * 1e-3 = 1000 km exactly.
        let radius = 1273239.5447351628;
        assert_eq!(2.0 * PI * radius / 8.0, 1e6);
        // deck_strength: a coast 4 columns away is at dist * dx == 4000 km, which is NOT above the limit
        // (strict): its strength is exp(-4000 / 1200); 5 columns (5000 km) are out of range.
        let mut cont = vec![0.0_f32; 2000 * 8];
        cont[4] = 1.0;
        let d = deck_strength(&cont, 2000, 8, radius, 1.0);
        assert!((f64::from(d[0]) - (-4000.0_f64 / DECK_EFOLD_KM).exp()).abs() < 1e-7);
        assert!((f64::from(d[1]) - (-3000.0_f64 / DECK_EFOLD_KM).exp()).abs() < 1e-7);
        assert_eq!(d[7], 0.0);
        // smooth_on_sphere: a row sigma of exactly 0.3 px is NOT above the threshold: the polar row is copied
        // (sigma_km 300 / 1000 km per texel), and the column pass sigma 3e-4 px has radius 0.
        let mut a = vec![0.0_f32; 2000 * 8];
        a[3] = 1.0;
        let s = smooth_on_sphere(&a, 2000, 8, 300.0, radius);
        assert_eq!(s[..8], a[..8]);
    }

    // -- The terms

    #[test]
    fn hemispheric_asymmetry_of_a_single_row_has_no_hemisphere() {
        // h = 1: the only row is at latitude 0, in neither hemisphere: 0/0 - 0/0 = NaN.
        let s = Surface::new(vec![1.0; 4], vec![0.0; 4], 1, 4, "");
        assert!(hemispheric_land_asymmetry(&s).is_nan());
    }

    #[test]
    fn itcz_shift_is_zero_without_a_surface_and_clipped_with_one() {
        assert_eq!(itcz_shift_deg(None, 30.0), 0.0);
        let north = Surface::new(grid(4, 4, |i, _| if i < 2 { 1.0 } else { 0.0 }), vec![0.0; 16], 4, 4, "");
        assert!((itcz_shift_deg(Some(&north), 30.0) - 0.35 * 30.0).abs() < 1e-12);
        let south = Surface::new(grid(4, 4, |i, _| if i < 2 { 0.0 } else { 1.0 }), vec![0.0; 16], 4, 4, "");
        assert!((itcz_shift_deg(Some(&south), 30.0) + 0.35 * 30.0).abs() < 1e-12);
    }

    #[test]
    fn smoothstep_and_gauss_run_in_the_dtype_of_x() {
        assert_eq!(smoothstep(0.0, 1.0, 0.5_f64), 0.5);
        assert_eq!(smoothstep(0.0, 1.0, -3.0_f32), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 9.0_f32), 1.0);
        assert!(smoothstep(0.0, 1.0, f64::NAN).is_nan());
        assert_eq!(gauss(2.0_f64, 2.0, 0.5), 1.0);
        let a: f32 = gauss(0.123_f32, 0.8, 0.25);
        assert!((f64::from(a) - gauss(0.123_f64, 0.8, 0.25)).abs() < 1e-6);
    }

    #[test]
    fn the_land_contrast_at_the_equator_is_the_pinned_value() {
        // -0.08 + 0.12 * 0 * 1 + 0.04 * gauss(0, 0.8, 0.25)
        let v = land_contrast(0.0_f64);
        assert!((v - (-0.08 + 0.04 * (-(0.8_f64 / 0.25).powi(2)).exp())).abs() < 1e-15);
        // Past the fade the extratropical plateau is gone and both gaussians are 0.
        assert!(land_contrast(10.0_f64).abs() < 1e-12);
        // In the middle of the plateau (x = 1.5): 0.12 minus the two gaussian tails.
        let plateau = land_contrast(1.5_f64);
        assert!((plateau - (0.12 + 0.04 * (-(0.7_f64 / 0.25).powi(2)).exp() - 0.08 * (-(1.5_f64 / 0.3).powi(2)).exp())).abs() < 1e-12);
    }

    #[test]
    fn deck_strength_decays_east_of_a_coast_and_mirrors_for_a_retrograde_body() {
        // A 2 x 16 map whose texel is 100 km wide at the equator (2 pi R / 16): at the +-45 degree rows of a
        // two-row grid a texel is 100 cos 45 km. The only coast is column 12 of both rows.
        let radius = 100.0 * 16.0 / (2.0 * PI);
        let mut cont = vec![0.0_f32; 2 * 16];
        cont[12] = 1.0;
        cont[16 + 12] = 1.0;
        let dx = 100.0 * grid_lonlat(16, 2).1[0].cos();
        let decay = |cols: f64| (-cols * dx / DECK_EFOLD_KM).exp();
        let east = deck_strength(&cont, 2, 16, radius, 1.0);
        // Two columns west of the coast the next coast is 2 columns away; right after it the next one wraps.
        assert!((f64::from(east[10]) - decay(2.0)).abs() < 1e-6);
        assert!((f64::from(east[13]) - decay(15.0)).abs() < 1e-6);
        assert_eq!(east[12], 0.0); // the coast cell itself: 1 - cont = 0
        let west = deck_strength(&cont, 2, 16, radius, -1.0);
        // Mirrored: the coast is DOWNSTREAM to the west, so the strength sits east of it.
        assert!((f64::from(west[14]) - decay(2.0)).abs() < 1e-6);
        assert!((f64::from(west[10]) - decay(14.0)).abs() < 1e-6);
        // Beyond 4000 km the strength is 0: with 10x wider texels a column is 707 km, so 5 columns (3536 km) are
        // inside the search range and 6 (4243 km) are outside.
        let big = deck_strength(&cont, 2, 16, radius * 10.0, 1.0);
        assert!((f64::from(big[7]) - (-5.0 * 10.0 * dx / DECK_EFOLD_KM).exp()).abs() < 1e-6);
        assert_eq!(big[6], 0.0);
        assert_eq!(big[13], 0.0);
        // A row without any coast, or made only of coast, is zero.
        assert!(deck_strength(&[0.0; 32], 2, 16, radius, 1.0).iter().all(|&v| v == 0.0));
        assert!(deck_strength(&[1.0; 32], 2, 16, radius, 1.0).iter().all(|&v| v == 0.0));
        // A spin of 0 is prograde (`spin >= 0`), and a NaN spin fails that test, so it mirrors like a negative
        // one (Python: `coast[i] if spin >= 0 else coast[i][::-1]`).
        assert_eq!(deck_strength(&cont, 2, 16, radius, 0.0), east);
        assert_eq!(deck_strength(&cont, 2, 16, radius, -0.0), east);
        assert_eq!(deck_strength(&cont, 2, 16, radius, f64::NAN), west);
        assert_ne!(east, west);
    }

    #[test]
    fn surface_terms_without_wind_has_a_zero_orographic_part_and_the_asked_size() {
        let land = grid(8, 16, |i, j| if (4..9).contains(&j) && (2..6).contains(&i) { 1.0 } else { 0.0 });
        let s = Surface::new(land, vec![0.0; 8 * 16], 8, 16, "");
        let t = surface_terms(&s, 16, 30.0, 6371.0, 1.0, None, 1024);
        assert_eq!((t.h, t.w, t.anomaly.len()), (8, 16, 128));
        assert!(t.orographic.iter().all(|&v| v == 0.0));
        // Upsampled: term_width 8 computed at 8x4 and resampled to 16x8.
        let u = surface_terms(&s, 16, 30.0, 6371.0, 1.0, None, 8);
        assert_eq!((u.h, u.w, u.anomaly.len()), (8, 16, 128));
        assert_eq!(u.itcz_shift_deg, t.itcz_shift_deg);
    }

    #[test]
    fn resample_keeps_the_values_of_a_constant_field_and_the_same_width_is_untouched() {
        let a = vec![0.25_f32; 8 * 16];
        let r = resample(&a, 8, 16, 24);
        assert_eq!(r.len(), 12 * 24);
        assert!(r.iter().all(|&v| (v - 0.25).abs() < 1e-7));
        // The same width returns the array itself, whatever its dtype or shape: no float32 rounding of
        // 0.1_f64, and a 3 x 4 array keeps its 3 rows (a resample would give 2).
        let b = vec![0.1_f64; 12];
        assert_eq!(resample(&b, 3, 4, 4), b);
    }

    #[test]
    fn flat_ground_gives_no_orographic_anomaly_and_a_ridge_lifts_its_windward_side() {
        let s = Surface::new(vec![0.0; 8 * 16], vec![1.0; 8 * 16], 8, 16, "");
        let wind: Vec<f32> = (0..8 * 16).flat_map(|k| [3.0 + (k % 5) as f32, -2.0]).collect();
        assert!(orographic_anomaly(&s, &wind, 8, 16, 6371.0).iter().all(|&v| v == 0.0));
        // A 3 km ridge at column 8 under a 10 m/s westerly: air rises on the west side (column 7) and sinks on
        // the lee side (column 9).
        let elev = grid(8, 16, |_, j| if j == 8 { 3.0 } else { 0.0 });
        let ridge = Surface::new(vec![0.0; 128], elev, 8, 16, "");
        let westerly: Vec<f32> = (0..128).flat_map(|_| [10.0_f32, 0.0]).collect();
        let a = orographic_anomaly(&ridge, &westerly, 8, 16, 6371.0);
        assert!(a[4 * 16 + 7] > 0.02, "{}", a[4 * 16 + 7]);
        assert!(a[4 * 16 + 9] < -0.02, "{}", a[4 * 16 + 9]);
    }

    /// FNV-1a over the float32 bit patterns, one 32-bit word at a time.
    fn fnv_bits(a: &[f32]) -> u64 {
        a.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, v| (h ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3))
    }

    #[test]
    fn orographic_anomaly_matches_the_numpy_bits_and_keeps_the_float32_north_product() {
        // A 16 x 32 map of closed-form float32 values (numpy 2.5.4 builds the same arrays with the same float32
        // operations), the output of `orographic_anomaly(Surface(zeros, elev), wind, 6371.0)` taken bit for bit:
        // a hash of the 512 float32 words and a few words in the clear. The north wind component times dh/dy is
        // a FLOAT32 product (float32 wind times a float32 slope) added to the float64 east product: computing it
        // in float64 moves the anomaly by about one float32 ulp in many cells, far under any golden tolerance.
        let (h, w) = (16usize, 32usize);
        let cell = |f: &dyn Fn(usize, usize) -> f32| -> Vec<f32> { (0..h * w).map(|k| f(k / w, k % w)).collect() };
        let elev = cell(&|i, j| ((i * 13 + j * 29 + ((i * j) % 7) * 11) % 41) as f32 * 0.07_f32);
        let east = cell(&|i, j| ((i * 5 + j * 3) % 23) as f32 * 0.9_f32 - 8.0_f32);
        let north = cell(&|i, j| ((i * 11 + j * 7) % 19) as f32 * 0.8_f32 - 7.0_f32);
        let calm = vec![0.0_f32; h * w];
        let surface = Surface::new(vec![0.0; h * w], elev, h, w, "");
        let wind_of = |u: &[f32], v: &[f32]| -> Vec<f32> { u.iter().zip(v).flat_map(|(&a, &b)| [a, b]).collect() };
        // (name, wind, hash, [(cell, bits)])
        let runs: [(&str, Vec<f32>, u64, [(usize, u32); 6]); 3] = [
            (
                "east and north",
                wind_of(&east, &north),
                0x7030086e3ca8e6c6,
                [(5, 3190491061), (37, 3154451841), (100, 3164644854), (201, 996530781), (300, 3164344101), (450, 3154580821)],
            ),
            (
                "east only",
                wind_of(&east, &calm),
                0xbd301fc73fb2d92b,
                [(5, 3192789910), (37, 3156072063), (100, 3149848208), (201, 3132491742), (300, 3154954174), (450, 3160667129)],
            ),
            (
                "north only",
                wind_of(&calm, &north),
                0x70afbaa298f5d9a8,
                [(5, 0), (37, 985033470), (100, 3139681819), (201, 1000369139), (300, 3157334224), (450, 1002050743)],
            ),
        ];
        for (name, wind, hash, words) in runs {
            let a = orographic_anomaly(&surface, &wind, h, w, 6371.0);
            assert_eq!(a.len(), h * w);
            for (k, bits) in words {
                assert_eq!(a[k].to_bits(), bits, "{name}: cell {k}");
            }
            assert_eq!(fnv_bits(&a), hash, "{name}: hash of the 512 words");
        }
    }

    /// A 64 x 128 surface of closed-form values (blocks of land, a few coast fractions, elevation on every
    /// land texel) that numpy builds with the same integer and float32 operations.
    fn closed_form_surface() -> Surface {
        let (h, w) = (64usize, 128usize);
        let land: Vec<f32> = (0..h * w)
            .map(|k| {
                let (i, j) = (k / w, k % w);
                match (i / 6 * 5 + j / 9 * 3 + (i / 6) * (j / 9) + i / 20) % 7 {
                    0..=2 => 1.0,
                    3 => 0.3,
                    _ => 0.0,
                }
            })
            .collect();
        let elev: Vec<f32> = (0..h * w).map(|k| land[k] * (((k / w * 13 + k % w * 29) % 41) as f32 * 0.07_f32)).collect();
        Surface::new(land, elev, h, w, "src")
    }

    /// Asserts the FNV-1a hash of the float32 words of `a` (numpy's bits, hashed by the same function).
    fn assert_bits(name: &str, a: &[f32], want: u64) {
        assert_eq!(fnv_bits(a), want, "{name}");
    }

    /// Asserts float64 values equal numpy's to a relative 1e-12 (NaN against NaN): far under any dtype mix
    /// (a float32 step moves them by 1e-8 or more), far over a libm difference of one ulp.
    fn assert_f64s(name: &str, got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "{name}");
        for (k, (&g, &x)) in got.iter().zip(want).enumerate() {
            let ok = (g.is_nan() && x.is_nan()) || (g - x).abs() <= 1e-12 * (1.0 + x.abs());
            assert!(ok, "{name}[{k}]: {g:?} against numpy's {x:?}");
        }
    }

    #[test]
    fn the_smoothing_and_the_coast_functions_match_the_numpy_bits() {
        // numpy 2.5.4 and scipy 1.18.1 on the closed-form surface (a hash of every float32 word of the result).
        let s = closed_form_surface();
        let r = 6371.0;
        assert_bits("smooth_on_sphere land 300 km", &smooth_on_sphere(&s.land, 64, 128, 300.0, r), 0x51cfe61125b1e4f3);
        assert_bits("smooth_on_sphere elevation 900 km", &smooth_on_sphere(&s.elevation_km, 64, 128, 900.0, r), 0xd7efa6bcc2dc3de0);
        let cont = continentality(&s, r);
        assert_bits("continentality", &cont, 0x51cfe61125b1e4f3);
        assert_bits("deck_strength prograde", &deck_strength(&cont, 64, 128, r, 1.0), 0x4d229ee958af7615);
        assert_bits("deck_strength retrograde", &deck_strength(&cont, 64, 128, r, -1.0), 0x2ac174df16add870);
        // Surface::at: the box average (2 x 2 blocks), and bilinear down (96, aliasing) and up (256).
        let box64 = s.at(64);
        assert_bits("at(64) land", &box64.land, 0x03c6ca2dd09cb873);
        assert_bits("at(64) elevation", &box64.elevation_km, 0xa6e49e6d6edba20a);
        // Blocks of 8 x 8 and 16 x 16: each block row is summed pairwise (eight accumulators), not in order.
        let (box16, box8) = (s.at(16), s.at(8));
        assert_eq!((box16.h, box16.w, box8.h, box8.w), (8, 16, 4, 8));
        assert_bits("at(16) land", &box16.land, 0xc29c27feca8ad54e);
        assert_bits("at(16) elevation", &box16.elevation_km, 0x5c7b4d00909c93cd);
        assert_bits("at(8) land", &box8.land, 0x8478c91aebe567e5);
        assert_bits("at(8) elevation", &box8.elevation_km, 0x8709bb58c9739b0a);
        let down = s.at(96);
        assert_bits("at(96) land", &down.land, 0xd4c55a164df5a131);
        assert_bits("at(96) elevation", &down.elevation_km, 0x7a058d63c062bce3);
        let up = s.at(256);
        assert_bits("at(256) land", &up.land, 0x4df55e144a07cfd1);
        assert_bits("at(256) elevation", &up.elevation_km, 0x566aea298c7229ad);
    }

    #[test]
    fn smooth_on_sphere_rounds_a_float64_row_to_float32_before_filtering_or_copying_it() {
        // A 16 x 32 float64 field of values that float32 cannot hold (`k / 19`), smoothed at 300 km on a 6371 km
        // body: the texel is 1251 km wide, so the rows near the equator have a sigma under 0.3 px and are copied
        // (rounded to float32 by the assignment into the float32 output), the others are filtered after
        // `a[i].astype(np.float32)`. numpy 2.5.4 / scipy 1.18.1 bits.
        let (h, w) = (16usize, 32usize);
        let a: Vec<f64> = (0..h * w).map(|k| ((k / w * 7 + k % w * 13) % 19) as f64 / 19.0).collect();
        let o = smooth_on_sphere(&a, h, w, 300.0, 6371.0);
        assert_bits("smooth_on_sphere float64 input", &o, 0x1bcedc6c37d88220);
    }

    #[test]
    fn continentality_clips_a_land_field_outside_zero_one_and_keeps_nan() {
        // np.clip(smooth, 0, 1): a land field is 0..1 by contract, but a constant 3 (or -2) smooths to itself
        // and is clipped; a NaN texel spreads through the smoothing and stays NaN (np.clip propagates it).
        let hot = Surface::new(vec![3.0; 8 * 16], vec![0.0; 8 * 16], 8, 16, "");
        assert!(continentality(&hot, 6371.0).iter().all(|&v| v == 1.0));
        let cold = Surface::new(vec![-2.0; 8 * 16], vec![0.0; 8 * 16], 8, 16, "");
        assert!(continentality(&cold, 6371.0).iter().all(|&v| v == 0.0));
        let mut land = vec![0.5_f32; 8 * 16];
        land[3 * 16 + 5] = f32::NAN;
        let holed = Surface::new(land, vec![0.0; 8 * 16], 8, 16, "");
        let c = continentality(&holed, 6371.0);
        assert!(c[3 * 16 + 5].is_nan());
        assert!(c.iter().all(|&v| v.is_nan() || (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn the_itcz_inputs_and_the_earth_tables_match_numpy() {
        let s = closed_form_surface();
        let r = 6371.0;
        assert_f64s("hemispheric_land_asymmetry", &[hemispheric_land_asymmetry(&s)], &[0.15079911791064604]);
        // Rows of 64 different non-dyadic float32 fractions: each row mean is a float32 pairwise mean (an
        // exact float64 mean rounded to float32 would move some of them by an ulp, and the asymmetry by 1e-9).
        let (rh, rw) = (32usize, 64usize);
        let rich_land: Vec<f32> = (0..rh * rw).map(|k| ((k / rw * 7 + k % rw * 13 + (k / rw) * (k / rw) * (k / rw)) % 19) as f32 / 19.0_f32).collect();
        let rich = Surface::new(rich_land, vec![0.0; rh * rw], rh, rw, "");
        assert_f64s("hemispheric_land_asymmetry rich rows", &[hemispheric_land_asymmetry(&rich)], &[-0.0031222003857880876]);
        assert_f64s("itcz_shift_deg rich rows", &[itcz_shift_deg(Some(&rich), 30.0)], &[-0.07805500964470218]);
        assert_f64s("itcz_shift_deg", &[itcz_shift_deg(Some(&s), 30.0)], &[3.7699779477661504]);
        let nan = f64::NAN;
        // earth_tables(surface, 6371, spin, 30, width): at 128 every band holds rows, at 64 two bands are empty.
        let (rc, rd) = earth_tables(&s, r, 1.0, 30.0, 128);
        let rc128 = [0.5104044079780579, 0.6865323781967163, 0.7328155040740967, 0.727252721786499, 0.6033347249031067, 0.48283135890960693, 0.4792936444282532, 0.478738009929657, 0.47515547275543213, 0.4742238521575928, 0.4738045334815979, 0.4721648395061493, 0.46803197264671326, 0.46448224782943726, 0.46796903014183044];
        let rd128 = [0.0, 0.001288436848665102, 0.05152493899045832, 0.0608743206539657, 0.10398504127806518, 0.14304399550428112, 0.1018606587002632, 0.012852358754231446, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_f64s("earth_tables continentality 128", &rc, &rc128);
        assert_f64s("earth_tables deck 128", &rd, &rd128);
        let (rc, rd) = earth_tables(&s, r, 1.0, 30.0, 64);
        let rc64 = [0.488968163728714, 0.707379937171936, 0.7340850830078125, 0.7100144624710083, nan, 0.4969612956047058, 0.47899752855300903, 0.47917887568473816, 0.47472625970840454, 0.4742191731929779, 0.47400015592575073, 0.47136741876602173, 0.46478894352912903, nan, 0.46478861570358276];
        let rd64 = [0.0, 0.0, 0.05727260430566263, 0.06204264872940257, nan, 0.09960076946299523, 0.1054073683400472, 0.004261908826063836, 0.0, 0.0, 0.0, 0.0, 0.0, nan, 0.0];
        assert_f64s("earth_tables continentality 64", &rc, &rc64);
        assert_f64s("earth_tables deck 64", &rd, &rd64);
        // A retrograde body mirrors the deck, not the continentality.
        let (rc, rd) = earth_tables(&s, r, -1.0, 30.0, 64);
        let rd64_retro = [0.0, 0.0, 0.0531844501547454, 0.056502651277696714, nan, 0.10002659316523932, 0.10615775731318855, 0.004258900625810332, 0.0, 0.0, 0.0, 0.0, 0.0, nan, 0.0];
        assert_f64s("earth_tables continentality 64 retrograde", &rc, &rc64);
        assert_f64s("earth_tables deck 64 retrograde", &rd, &rd64_retro);
    }

    #[test]
    fn surface_terms_match_the_numpy_bits_with_their_float32_and_float64_mix() {
        // numpy 2.5.4 on the closed-form surface (see `closed_form_surface`), the six float32 maps of each call
        // hashed: [anomaly, land, deck, orographic, continentality, deck_strength]. `land` and `deck` are float64
        // (a float64 column against a float32 map) rounded to float32 once, at the end; computing a step in
        // float32 moves some words by an ulp, far under every golden tolerance.
        let s = closed_form_surface();
        let r = 6371.0;
        let check = |name: &str, t: &SurfaceTerms, h: usize, w: usize, want: [u64; 6]| {
            assert_eq!((t.h, t.w), (h, w), "{name}");
            let maps = [&t.anomaly, &t.land, &t.deck, &t.orographic, &t.continentality, &t.deck_strength];
            let names = ["anomaly", "land", "deck", "orographic", "continentality", "deck_strength"];
            for ((m, n), hash) in maps.iter().zip(names).zip(want) {
                assert_eq!(m.len(), h * w, "{name} {n}");
                assert_bits(&format!("{name} {n}"), m, hash);
            }
        };
        // No wind, the smooth terms at the full width (term_width 1024 >= 128).
        let t = surface_terms(&s, 128, 30.0, r, 1.0, None, 1024);
        check("128 no wind", &t, 64, 128, [0x4e350f026d7361fa, 0x705576192187e625, 0xde0ed2e52c69a3f7, 0xb9d103fd6854a325, 0x51cfe61125b1e4f3, 0x4d229ee958af7615]);
        assert_f64s("itcz", &[t.itcz_shift_deg], &[3.7699779477661504]);
        // A 16 x 32 wind, a retrograde body, the terms computed at 64 and upsampled to 128.
        let (hw, ww) = (16usize, 32usize);
        let uv: Vec<f32> = (0..hw * ww)
            .flat_map(|k| {
                let (i, j) = (k / ww, k % ww);
                [((i * 5 + j * 3) % 23) as f32 * 0.9_f32 - 8.0_f32, ((i * 11 + j * 7) % 19) as f32 * 0.8_f32 - 7.0_f32]
            })
            .collect();
        let wind = WindGrid { uv: &uv, h: hw, w: ww };
        let t = surface_terms(&s, 128, 30.0, r, -1.0, Some(&wind), 64);
        check("128 wind tw 64", &t, 64, 128, [0x3dcb33e917be7207, 0xf6a226a1f4de95a9, 0xab834ed9dca1e12d, 0x94184345fc88e75a, 0x3b5efcc3046a930a, 0x5ce1c281944da19d]);
        // The smooth terms at the asked width itself (term_width 1024 >= 64), a prograde body.
        let t = surface_terms(&s, 64, 30.0, r, 1.0, Some(&wind), 1024);
        assert_eq!((t.h, t.w), (32, 64));
        assert_bits("64 wind anomaly", &t.anomaly, 0x80674adbc55ae989);
        assert_bits("64 wind orographic", &t.orographic, 0x424bb23192660003);
    }
}
