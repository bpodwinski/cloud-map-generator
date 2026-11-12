//! Earth's reference map and its measures, computed once (`earth_reference` of `ui/server.py`).

use crate::images::decode_gray;
use cloud_map_generator_core::metrics::{measure, Measure};

/// Earth's cloud fraction (0..1, north-up, `h x w`: 1024 x 2048 for the shipped image) and its measures.
pub struct Reference {
    pub data: Vec<f32>,
    pub h: usize,
    pub w: usize,
    pub measures: Measure,
}

impl Reference {
    /// Decodes the PNG (8 or 16 bit; a color image reads its first channel) as `a.astype(np.float32) / 255` (or
    /// 65535) and measures it (a second or so at 2048 x 1024).
    pub fn from_png(png: &[u8]) -> Result<Reference, String> {
        let img = decode_gray(png).map_err(|e| format!("the Earth reference map: {e}"))?;
        let data: Vec<f32> = img.data.iter().map(|&v| f32::from(v) / img.max).collect();
        let measures = measure(&data, img.height, img.width);
        Ok(Reference { data, h: img.height, w: img.width, measures })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_reference_is_earth_at_2048() {
        let r = Reference::from_png(crate::REFERENCE_PNG).unwrap();
        assert_eq!((r.h, r.w), (1024, 2048));
        assert!((r.measures.global_mean - 0.40).abs() < 0.1, "mean cloud fraction {}", r.measures.global_mean);
        assert_eq!(r.measures.zonal.len(), 36);
        assert!(r.data.iter().all(|v| (0.0..=1.0).contains(v)));
    }
}
