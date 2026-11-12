//! The poles of an equirectangular map. A pole is one point, yet its row holds `width` texels: the circle of latitude
//! `phi` has a circumference of only `cos(phi)` of the equator's, so along longitude a map of a sphere is oversampled by
//! `1 / cos(phi)` there. A field made texel by texel (the fine texture, the sub-texel grain) keeps a texel's worth of
//! independent detail in every one of them, so the rows near a pole read as streaks of noise with an std of 0.09 where a
//! smooth sphere has none (the top row is one point: its values must agree).
//!
//! [`polar_filter`] averages each row along longitude over the `1 / cos(phi)` texels it oversamples, which is what rendering
//! the sphere at the map's resolution gives. The mean of a row is kept; rows nearer the equator than the threshold are not
//! touched, so a map is unchanged below about 70 degrees.

/// Rows where one texel stands for fewer than this many texels of the equator's width are left alone.
const MIN_WINDOW: usize = 3;

/// The window, texels, of the row `row` of `height` rows: `round(1 / cos(latitude))` of the row's centre, at most `width`.
pub fn window(row: usize, height: usize, width: usize) -> usize {
    let lat = (0.5 - (row as f64 + 0.5) / height as f64) * std::f64::consts::PI;
    let n = (1.0 / lat.cos().max(1e-9)).round();
    if n >= width as f64 {
        width
    } else {
        n as usize
    }
}

/// Filters `nrows` rows (`rows`, row-major, `width` wide) that are the rows `r0 ..` of a map `height` rows tall, in place.
/// A row with a non-finite value (a mask) is left as it is.
pub fn polar_filter(rows: &mut [f32], r0: usize, nrows: usize, width: usize, height: usize) {
    assert_eq!(rows.len(), nrows * width, "{nrows} rows of {width}");
    let mut padded: Vec<f64> = Vec::new();
    for i in 0..nrows {
        let n = window(r0 + i, height, width);
        if n < MIN_WINDOW {
            continue;
        }
        let row = &mut rows[i * width..(i + 1) * width];
        if row.iter().any(|v| !v.is_finite()) {
            continue;
        }
        // A circular box average of `n` texels, centred: prefix sums over the row repeated three times.
        padded.clear();
        padded.reserve(3 * width + 1);
        padded.push(0.0);
        let mut acc = 0.0;
        for k in 0..3 * width {
            acc += f64::from(row[k % width]);
            padded.push(acc);
        }
        let half = n / 2;
        let left = n - half - 1; // texels before the centre, so the window holds exactly n
        for (c, out) in row.iter_mut().enumerate() {
            let centre = c + width;
            let (a, b) = (centre - left, centre + half + 1);
            *out = ((padded[b] - padded[a]) / n as f64) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_grows_toward_the_pole_and_is_one_at_the_equator() {
        let (h, w) = (1024, 2048);
        assert_eq!(window(h / 2, h, w), 1);
        assert!(window(0, h, w) > 200, "{}", window(0, h, w));
        assert!(window(100, h, w) < window(10, h, w));
        assert_eq!(window(h / 2 - 100, h, w), window(h / 2 + 99, h, w), "symmetric about the equator");
    }

    #[test]
    fn equatorial_rows_are_untouched_and_polar_rows_lose_their_grain_but_keep_their_mean() {
        let (h, w) = (512, 1024);
        // Row noise that is white along longitude at every latitude.
        let mut state = 12345_u64;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / (1u64 << 24) as f32
        };
        let map: Vec<f32> = (0..h * w).map(|_| next()).collect();
        let mut out = map.clone();
        polar_filter(&mut out, 0, h, w, h);
        let std = |row: &[f32]| {
            let m = row.iter().sum::<f32>() / row.len() as f32;
            (row.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / row.len() as f32).sqrt()
        };
        let mean = |row: &[f32]| row.iter().sum::<f32>() / row.len() as f32;
        // Row 0 (the pole): grain gone, mean kept.
        assert!(std(&out[..w]) < 0.2 * std(&map[..w]), "{} vs {}", std(&out[..w]), std(&map[..w]));
        assert!((mean(&out[..w]) - mean(&map[..w])).abs() < 1e-4);
        // The equator: bit for bit.
        let eq = (h / 2) * w;
        assert_eq!(&out[eq..eq + w], &map[eq..eq + w]);
        // The same rows cut in bands give the same result.
        let mut banded = map.clone();
        for (r0, rows) in [(0usize, 100usize), (100, 300), (400, 112)] {
            polar_filter(&mut banded[r0 * w..(r0 + rows) * w], r0, rows, w, h);
        }
        assert_eq!(banded, out);
    }

    #[test]
    fn a_row_with_a_mask_is_left_alone() {
        let (h, w) = (64, 128);
        let mut map = vec![0.5_f32; h * w];
        map[3] = f32::NAN;
        let before = map.clone();
        polar_filter(&mut map[..w], 0, 1, w, h);
        assert!(map[3].is_nan());
        assert_eq!(&map[..3], &before[..3]);
    }
}
