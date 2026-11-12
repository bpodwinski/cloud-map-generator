//! The made-up world of `options.surface = "synthetic"`: `cloudgen/synthetic.py` / `golden/util.py`'s
//! `synthetic_surface`, ported.
//!
//! Five continents and a meridional range, analytic (no data file). Sea at 0 km, the ocean floor below it, the
//! land above, a 3 km range along the west coast of the big northern continent. `width` x `width / 2`, north-up,
//! column 0 at -180 degrees. The golden cases use the same world: the test below checks this function against
//! the frozen arrays of those cases.

use cloud_map_generator_core::sphere::grid_lonlat;
use cloud_map_generator_core::surface::Surface;
use std::f64::consts::PI;

/// The source name the Python world carries (`Surface.source`).
pub const SOURCE: &str = "golden synthetic world";

/// One Gaussian blob of the field: `exp(-((dlon / rx)^2 + ((lat - lat0) / ry)^2))`, the longitude difference
/// wrapped to `[-180, 180)`.
fn blob(lon: f64, lat: f64, lon0: f64, lat0: f64, rx: f64, ry: f64) -> f64 {
    let dlon = (lon - lon0 + 180.0).rem_euclid(360.0) - 180.0;
    let (a, b) = (dlon / rx, (lat - lat0) / ry);
    (-(a * a + b * b)).exp()
}

/// `synthetic_surface(width)`: the world on a `width` x `width / 2` grid.
pub fn synthetic_surface(width: usize) -> Surface {
    let h = width / 2;
    let (lon, lat) = grid_lonlat(width, h);
    // np.degrees: x * (180 / pi).
    let to_deg = 180.0 / PI;
    let (lon_deg, lat_deg): (Vec<f64>, Vec<f64>) = (lon.iter().map(|x| x * to_deg).collect(), lat.iter().map(|x| x * to_deg).collect());
    let mut elevation = Vec::with_capacity(h * width);
    for &la in &lat_deg {
        for &lo in &lon_deg {
            let field = [
                blob(lo, la, -100.0, 40.0, 35.0, 25.0),
                blob(lo, la, 20.0, 5.0, 28.0, 35.0),
                blob(lo, la, 100.0, 50.0, 45.0, 20.0),
                blob(lo, la, 135.0, -25.0, 22.0, 14.0),
                blob(lo, la, -60.0, -20.0, 18.0, 30.0),
            ]
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max);
            let land = field > 0.35;
            let base = if land { 1.2 * (field - 0.35) } else { -2.5 * (0.35 - field) };
            // `3.0 * blob(..) * (field > 0.35)`: the range only stands on land.
            let range = 3.0 * blob(lo, la, -112.0, 40.0, 5.0, 16.0) * if land { 1.0 } else { 0.0 };
            elevation.push((base + range) as f32);
        }
    }
    Surface::from_elevation(&elevation, h, width, 0.0, SOURCE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cloud_map_generator_golden::load_group;

    /// The frozen golden arrays hold this very world at 128 x 64: the elevation as the surface keeps it (clamped
    /// at the sea), and its land mask. The analytic field is float64 (`exp` of this platform's libm against
    /// numpy's): after the cast to float32 the arrays agree to a float32 rounding at worst.
    #[test]
    fn the_world_is_the_frozen_golden_one() {
        let cases = load_group("texture_surface");
        let ridge = cases.iter().find(|c| c.id == "orographic_anomaly.westerly_ridge128").expect("the westerly ridge case");
        let want_elevation = ridge.input("elevation_km").to_f64();
        let asym = cases.iter().find(|c| c.id == "hemispheric_land_asymmetry.synthetic128").expect("the land asymmetry case");
        let want_land = asym.input("land").to_f64();

        let world = synthetic_surface(128);
        assert_eq!((world.h, world.w), (64, 128));
        assert_eq!(world.source, SOURCE);
        assert_eq!(want_elevation.len(), 64 * 128);
        let mut worst = 0.0_f64;
        let mut inexact = 0;
        for (got, want) in world.elevation_km.iter().zip(&want_elevation) {
            let d = (f64::from(*got) - want).abs();
            worst = worst.max(d);
            inexact += usize::from(d > 0.0);
            assert!(d <= 1e-6 + 1e-6 * want.abs(), "elevation {got} against {want}");
        }
        assert!(inexact * 100 < want_elevation.len(), "{inexact} of {} texels differ from numpy's by an ulp or more (worst {worst:e})", want_elevation.len());
        assert_eq!(world.land.iter().map(|&l| f64::from(l)).collect::<Vec<_>>(), want_land, "the land mask is exact");
    }

    #[test]
    fn the_world_has_the_land_and_range_the_python_docstring_describes() {
        let world = synthetic_surface(256);
        let land = world.land.iter().filter(|&&l| l > 0.5).count() as f64 / world.land.len() as f64;
        assert!((0.1..0.5).contains(&land), "land fraction {land}");
        // The range stands 3 km above the west coast of the big northern continent: the highest ground is there.
        let (mut best, mut at) = (0.0_f32, (0, 0));
        for r in 0..world.h {
            for c in 0..world.w {
                if world.elevation_km[r * world.w + c] > best {
                    best = world.elevation_km[r * world.w + c];
                    at = (r, c);
                }
            }
        }
        assert!(best > 3.0, "the range is higher than 3 km: {best}");
        let (lat, lon) = (90.0 - (at.0 as f64 + 0.5) * 180.0 / world.h as f64, -180.0 + (at.1 as f64 + 0.5) * 360.0 / world.w as f64);
        assert!((lat - 40.0).abs() < 10.0 && (lon + 112.0).abs() < 10.0, "the peak is at {lat} N {lon} E");
        // Sea is 0 km (clamped) and no texel is below it.
        assert!(world.elevation_km.iter().all(|&e| e >= 0.0));
    }
}
