//! `measure.load`: a north-up grayscale coverage PNG as a float32 map, box-resampled to the reference's grid.
//!
//! Every map is measured on the reference's 2048 x 1024 grid so that spectra and object sizes compare like
//! for like: a wider map is averaged down by whole blocks (numpy's `reshape(..).mean(axis=(1, 3))`, which is
//! `export::box_downsample`).
//!
//! Python's `load` has no answer for a map NARROWER than the reference: its integer block factor is 0, the
//! reshape of an empty array succeeds, and the "map" it returns is 1024 x 2048 of NaN, so every metric is NaN.
//! Here such a map is measured at its own resolution (the statistics sample gnomonic patches by bilinear
//! lookup, so they exist at any resolution; the finest scales are simply not in the map) and the caller is told.

use crate::fail::{Fail, Result};
use crate::imageio::{read_gray, GrayImage};
use cloud_map_generator_core::export::box_downsample;
use std::path::Path;

/// The reference's size, `measure.REF_W` x `REF_H`.
pub const REF_W: usize = 2048;
pub const REF_H: usize = 1024;

/// A map as measured: float32 samples in 0..1, north-up, `h x w`, and whether it is the resampled one.
#[derive(Clone, Debug)]
pub struct LoadedMap {
    pub data: Vec<f32>,
    pub h: usize,
    pub w: usize,
    /// The map was narrower than the reference and is kept at its own size (see the module comment).
    pub kept_small: bool,
}

/// The decoded samples as float32: `a.astype(np.float32) / scale` (a float32 division by 255 or 65535).
fn to_f32(img: &GrayImage) -> Vec<f32> {
    img.data.iter().map(|&v| f32::from(v) / img.max).collect()
}

/// `load(path)`.
pub fn load_map(path: &Path) -> Result<LoadedMap> {
    let img = read_gray(path)?;
    let a = to_f32(&img);
    let (h, w) = (img.height, img.width);
    if (w, h) == (REF_W, REF_H) {
        return Ok(LoadedMap { data: a, h, w, kept_small: false });
    }
    if w < REF_W || h < REF_H {
        if w != 2 * h {
            return Err(Fail::new(format!("{}: {w} x {h} is not a 2:1 equirectangular map", path.display())));
        }
        return Ok(LoadedMap { data: a, h, w, kept_small: true });
    }
    // The block factors of Python's `load` are `h // 1024` and `w // 2048`; `box_downsample` uses the one of the
    // width, which is the same for a 2:1 map. Anything else is refused rather than cut to a wrong shape.
    if w != 2 * h || w % REF_W != 0 {
        return Err(Fail::new(format!(
            "{}: {w} x {h} cannot be box-resampled to {REF_W} x {REF_H} (needs a 2:1 map whose width is a multiple of {REF_W})",
            path.display()
        )));
    }
    let (data, oh, ow) = box_downsample(&a, h, w, REF_W);
    debug_assert_eq!((oh, ow), (REF_H, REF_W));
    Ok(LoadedMap { data, h: oh, w: ow, kept_small: false })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imageio::{write_gray16, write_gray8};

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("cloud-map-generator-cli-refmap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn the_reference_grid_is_read_as_is_with_the_right_scale() {
        let p = scratch("ref8.png");
        let d: Vec<u8> = (0..REF_W * REF_H).map(|i| (i % 256) as u8).collect();
        write_gray8(&p, REF_W, REF_H, &d).unwrap();
        let m = load_map(&p).unwrap();
        assert_eq!((m.h, m.w, m.kept_small), (REF_H, REF_W, false));
        assert_eq!(m.data[255], 1.0);
        assert_eq!(m.data[1], 1.0 / 255.0);

        let p = scratch("ref16.png");
        let d: Vec<u16> = (0..REF_W * REF_H).map(|i| (i % 65536) as u16).collect();
        write_gray16(&p, REF_W, REF_H, &d).unwrap();
        let m = load_map(&p).unwrap();
        assert_eq!(m.data[65535], 1.0);
        assert_eq!(m.data[32768], 32768.0_f32 / 65535.0);
    }

    #[test]
    fn a_wider_map_is_averaged_down_by_blocks() {
        let p = scratch("wide.png");
        // 4096 x 2048: blocks of 2 x 2; the left half is 0, the right half 200, a 4-texel column pair of 100.
        let (w, h) = (2 * REF_W, 2 * REF_H);
        let mut d = vec![0u8; w * h];
        for row in 0..h {
            for col in 0..w {
                d[row * w + col] = if col >= w / 2 { 200 } else { 0 };
            }
        }
        d[0] = 100; // one texel of the first block: the block mean is 25 / 255
        write_gray8(&p, w, h, &d).unwrap();
        let m = load_map(&p).unwrap();
        assert_eq!((m.h, m.w), (REF_H, REF_W));
        assert!((m.data[0] - 25.0 / 255.0).abs() < 1e-7);
        assert!((m.data[REF_W - 1] - 200.0 / 255.0).abs() < 1e-7);
    }

    #[test]
    fn a_narrower_map_is_kept_at_its_size_and_a_crooked_one_is_refused() {
        let p = scratch("small.png");
        write_gray8(&p, 64, 32, &[7u8; 64 * 32]).unwrap();
        let m = load_map(&p).unwrap();
        assert_eq!((m.h, m.w, m.kept_small), (32, 64, true));
        let p = scratch("crooked.png");
        write_gray8(&p, 64, 40, &[7u8; 64 * 40]).unwrap();
        assert!(load_map(&p).unwrap_err().0.contains("2:1"));
        let p = scratch("odd.png");
        write_gray8(&p, 3 * REF_W, 3 * REF_H / 2 + 1, &vec![0u8; 3 * REF_W * (3 * REF_H / 2 + 1)]).unwrap();
        assert!(load_map(&p).is_err());
    }
}
