//! Port of `cloudgen/sphere.py`: equirectangular grids, directions, gnomonic patches, bilinear lookups.
//!
//! Conventions (as in Python): maps are NORTH-UP (row 0 at the north pole edge); column `i` covers longitude
//! `-pi + (i + 0.5) 2 pi / W`; directions are right-handed with z north. Functions that numpy runs in the
//! dtype of their input are generic over `f32`/`f64` here, and run in that precision.

use num_traits::Float;

pub type Vec3<F> = [F; 3];

fn c<F: Float>(x: f64) -> F {
    F::from(x).expect("float conversion")
}

/// Longitudes (width) and latitudes (height) of the texel centers, radians, float64.
pub fn grid_lonlat(width: usize, height: usize) -> (Vec<f64>, Vec<f64>) {
    let pi = std::f64::consts::PI;
    let lon = (0..width).map(|i| -pi + (i as f64 + 0.5) * (2.0 * pi / width as f64)).collect();
    let lat = (0..height).map(|i| pi / 2.0 - (i as f64 + 0.5) * (pi / height as f64)).collect();
    (lon, lat)
}

/// `[cos lat cos lon, cos lat sin lon, sin lat]`, all the trigonometry in the precision of `F`.
pub fn lonlat_to_dir<F: Float>(lon: F, lat: F) -> Vec3<F> {
    let cl = lat.cos();
    [cl * lon.cos(), cl * lon.sin(), lat.sin()]
}

/// `(atan2(y, x), asin(clip(z, -1, 1)))`.
pub fn dir_to_lonlat<F: Float>(d: Vec3<F>) -> (F, F) {
    let lon = d[1].atan2(d[0]);
    let lat = d[2].max(-F::one()).min(F::one()).asin();
    (lon, lat)
}

/// Unit directions of every texel center, row-major (height rows of width), computed in float64 and
/// cast to `F` (Python: `lonlat_to_dir(...).astype(dtype)`).
pub fn grid_dirs<F: Float + Send + Sync>(width: usize, height: usize) -> Vec<Vec3<F>> {
    let (lon, lat) = grid_lonlat(width, height);
    let mut out = vec![[F::zero(); 3]; width * height];
    crate::par::for_each_chunk_mut(&mut out, width.max(1), |r, row| {
        let la = lat[r];
        for (slot, &lo) in row.iter_mut().zip(&lon) {
            let d = lonlat_to_dir::<f64>(lo, la);
            *slot = [c(d[0]), c(d[1]), c(d[2])];
        }
    });
    out
}

/// `v / |v|`, unguarded: the zero vector gives NaN, as numpy does.
pub fn normalize<F: Float>(v: Vec3<F>) -> Vec3<F> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

fn cross(a: Vec3<f64>, b: Vec3<f64>) -> Vec3<f64> {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// East and north unit vectors at a direction (float64; the poles get the fixed frame east = +y).
pub fn tangent_frame(center: Vec3<f64>) -> (Vec3<f64>, Vec3<f64>) {
    let east = cross([0.0, 0.0, 1.0], center);
    let n = (east[0] * east[0] + east[1] * east[1] + east[2] * east[2]).sqrt();
    let east = if n < 1e-9 { [0.0, 1.0, 0.0] } else { [east[0] / n, east[1] / n, east[2] / n] };
    let north = cross(center, east);
    (east, north)
}

/// numpy's `linspace(start, stop, n)` (endpoint included): `start + i step`, the last element exactly `stop`.
pub fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    match n {
        0 => vec![],
        1 => vec![start],
        _ => {
            let step = (stop - start) / (n - 1) as f64;
            let mut v: Vec<f64> = (0..n).map(|i| i as f64 * step + start).collect();
            v[n - 1] = stop;
            v
        }
    }
}

/// `n x n` directions of a gnomonic (tangent-plane) patch, row-major, row 0 to the north, column 0 to the west.
pub fn gnomonic_patch(lon0: f64, lat0: f64, half_angle: f64, n: usize) -> Vec<Vec3<f64>> {
    let c0 = lonlat_to_dir::<f64>(lon0, lat0);
    let (east, north) = tangent_frame(c0);
    let t: Vec<f64> = linspace(-1.0, 1.0, n).into_iter().map(|v| half_angle.tan() * v).collect();
    let mut out = Vec::with_capacity(n * n);
    for i in 0..n {
        let ty = -t[i];
        for j in 0..n {
            let tx = t[j];
            let p = [
                c0[0] + tx * east[0] + ty * north[0],
                c0[1] + tx * east[1] + ty * north[1],
                c0[2] + tx * east[2] + ty * north[2],
            ];
            out.push(normalize(p));
        }
    }
    out
}

/// Bilinear lookup of a north-up equirectangular image `img` (`h` rows, `w` columns, `ch` channels, row-major,
/// channel fastest) at directions: longitude wraps, latitude clamps. Returns `dirs.len() * ch` values.
///
/// numpy semantics kept: the pixel position `fx, fy` is computed in the precision `D` of the directions
/// (f32 directions give an f32 position), but the weights `fx - x0` mix it with an int64 index, so everything
/// after is float64, and so is the result whatever the image dtype was (callers pass an f32 image widened).
pub fn sample_equirect<D: Float + Send + Sync>(img: &[f64], h: usize, w: usize, ch: usize, dirs: &[Vec3<D>]) -> Vec<f64> {
    assert_eq!(img.len(), h * w * ch, "image size does not match h x w x channels");
    let pi: D = c(std::f64::consts::PI);
    let two_pi: D = c(2.0 * std::f64::consts::PI);
    let half: D = c(0.5);
    let (wf, hf): (D, D) = (c(w as f64), c(h as f64));
    // Each direction is looked up on its own: blocks of directions run in parallel and join in order.
    crate::par::map_chunks(dirs, crate::par::BLOCK, |block| {
        let mut out = Vec::with_capacity(block.len() * ch);
        for d in block {
            let (lon, lat) = dir_to_lonlat(*d);
            let fx = (lon + pi) / two_pi * wf - half;
            let fy = (pi / c(2.0) - lat) / pi * hf - half;
            let x0 = fx.floor().to_i64().unwrap_or(i64::MIN);
            let y0 = fy.floor().to_i64().unwrap_or(i64::MIN);
            let tx = fx.to_f64().unwrap() - x0 as f64;
            let ty = fy.to_f64().unwrap() - y0 as f64;
            let wi = w as i64;
            let x0w = x0.rem_euclid(wi) as usize;
            let x1w = (x0.wrapping_add(1)).rem_euclid(wi) as usize;
            let y0c = y0.clamp(0, h as i64 - 1) as usize;
            let y1c = (y0.saturating_add(1)).clamp(0, h as i64 - 1) as usize;
            for k in 0..ch {
                let at = |y: usize, x: usize| img[(y * w + x) * ch + k];
                let a = at(y0c, x0w) * (1.0 - tx) + at(y0c, x1w) * tx;
                let b = at(y1c, x0w) * (1.0 - tx) + at(y1c, x1w) * tx;
                out.push(a * (1.0 - ty) + b * ty);
            }
        }
        out
    })
}

/// Upsample a coarse north-up grid of unit vectors (`h_c` rows of `w_c`) to `height` x `width` by bilinear
/// interpolation of the 3D vectors, renormalized: seam-free. Works in float64, returns float32 (as Python).
pub fn upsample_dirs(dirs: &[Vec3<f64>], h_c: usize, w_c: usize, width: usize, height: usize) -> Vec<Vec3<f32>> {
    assert_eq!(dirs.len(), h_c * w_c);
    let target = grid_dirs::<f64>(width, height);
    let flat: Vec<f64> = dirs.iter().flatten().copied().collect();
    let sampled = sample_equirect(&flat, h_c, w_c, 3, &target);
    crate::par::map_index(sampled.len() / 3, |i| {
        let s = &sampled[3 * i..3 * i + 3];
        let n = normalize([s[0], s[1], s[2]]);
        [n[0] as f32, n[1] as f32, n[2] as f32]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_centers_and_round_trip() {
        let (lon, lat) = grid_lonlat(4, 2);
        assert!((lon[0] + std::f64::consts::PI * 3.0 / 4.0).abs() < 1e-12);
        assert!((lat[0] - std::f64::consts::PI / 4.0).abs() < 1e-12);
        let d = lonlat_to_dir(0.3_f64, -0.4);
        let (lo, la) = dir_to_lonlat(d);
        assert!((lo - 0.3).abs() < 1e-12 && (la + 0.4).abs() < 1e-12);
    }

    #[test]
    fn linspace_matches_numpy() {
        assert_eq!(linspace(-1.0, 1.0, 1), vec![-1.0]);
        assert_eq!(linspace(-1.0, 1.0, 3), vec![-1.0, 0.0, 1.0]);
        let v = linspace(-1.0, 1.0, 5);
        assert_eq!(v[4], 1.0);
    }

    #[test]
    fn sampling_a_constant_image_returns_the_constant() {
        let img = vec![0.25_f64; 8 * 16];
        let dirs = grid_dirs::<f64>(16, 8);
        assert!(sample_equirect::<f64>(&img, 8, 16, 1, &dirs).iter().all(|&v| (v - 0.25).abs() < 1e-15));
    }
}
