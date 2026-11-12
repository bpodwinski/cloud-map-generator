//! Port of the pure helpers of `bake.py` that are neither an encoding nor I/O: the climate classes of the
//! library, [`library_body`] (the Earth-sized water world each class is baked on), the naming and seeding
//! rules of the library and list entries, and [`bake_one`], the generation and encoding of one entry without
//! the files.
//!
//! The encodings of `bake.py` (`box_downsample`, `key_seed`, `slug`, `write_entry`) live in
//! `export.rs` ([`crate::export`]) and are re-exported here under their Python names where a caller wants
//! `bake::key_seed`. What stays out is the file system, the JSON and the manifest: the command-line wrapper
//! writes the bytes this module and `export.rs` return.

use crate::climate::{map_width_for_radius, BodyInputs, EARTH_GRAVITY, EARTH_RADIUS_KM, EARTH_ROTATION_DAYS};
use crate::export::{encode_entry, EncodedEntry, EntryExtras, Plane};
use crate::generate::{generate, GenOutput, GenParams, GenerateError};
use crate::surface::Surface;

pub use crate::export::{key_seed, slug, SLICE_WIDTH};

/// `bake.LIST_MAX_WIDTH`: the widest map of the list product (the texel rule capped here).
pub const LIST_MAX_WIDTH: usize = 8192;

/// `bake.LIBRARY_SCALE_HEIGHT_KM`: the scale height of the `earth` profile's air, which every procedural water
/// world uses.
pub const LIBRARY_SCALE_HEIGHT_KM: f64 = 11.1;

/// `bake.LIBRARY_CLASSES`: `(min, max, center)` of each climate class, in degrees of Hadley edge: the class
/// holds the edges in `[min, max)` and is baked at `center`. The last class is the global cell (no storm track,
/// `has_baroclinic_eddies` false).
pub const LIBRARY_CLASSES: [(f64, f64, f64); 4] = [(0.0, 25.0, 20.0), (25.0, 35.0, 30.0), (35.0, 50.0, 42.5), (50.0, 91.0, 70.0)];

/// The `--library-width` default of `bake.py`: the library is baked at 4096.
pub const DEFAULT_LIBRARY_WIDTH: usize = 4096;

/// The `--k` default of `bake.py`: seeds per library class.
pub const DEFAULT_LIBRARY_SEEDS: usize = 4;

/// `bake.library_body(center_deg)`: an Earth-sized water world whose Hadley edge is `center_deg` (Held-Hou,
/// `climate.rs`): the rotation period that puts the edge at the center for the library's scale height,
/// `EARTH_ROTATION_DAYS center / (30 sqrt(gh_ratio))`.
///
/// Unguarded, as in Python: a center below the Hadley minimum (10 degrees) or above the maximum (90) is
/// clamped by `hadley_edge_deg`, a center of 0 gives a zero period (no spin: edge 90), a negative center a
/// negative period (a retrograde spin; the edge uses its magnitude). The power is `gh_ratio ** 0.5`, a libm
/// `pow`, as Python spells it.
pub fn library_body(center_deg: f64) -> BodyInputs {
    let gh_ratio = (EARTH_GRAVITY * LIBRARY_SCALE_HEIGHT_KM) / (EARTH_GRAVITY * 8.0);
    let edge_at_earth_spin = 30.0 * gh_ratio.powf(0.5);
    BodyInputs::new(Some(EARTH_ROTATION_DAYS * center_deg / edge_at_earth_spin), EARTH_RADIUS_KM, EARTH_GRAVITY, LIBRARY_SCALE_HEIGHT_KM)
}

/// `int(center * 10)`: the center in tenths of a degree, truncated toward zero, which names the files and
/// seeds of a library class.
fn tenths(center_deg: f64) -> i64 {
    (center_deg * 10.0) as i64
}

/// One entry of the library: seed number `index` of the class centered at `center_deg`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LibraryEntry {
    /// Position of the class in [`LIBRARY_CLASSES`].
    pub class_index: usize,
    pub center_deg: f64,
    /// Which of the class's seeds (`0..k`).
    pub index: usize,
}

impl LibraryEntry {
    /// The generator seed of the entry, `1000 int(center 10) + index` (`bake.main`).
    ///
    /// # Panics
    /// For a negative center (the library has none).
    pub fn seed(&self) -> u64 {
        assert!(self.center_deg >= 0.0, "a library center is a Hadley edge, not negative: {}", self.center_deg);
        1000 * tenths(self.center_deg) as u64 + self.index as u64
    }

    /// The file stem, `cloud_library_h<tenths, 3 digits>_s<index>_<version>`.
    pub fn stem(&self, version: &str) -> String {
        format!("cloud_library_h{:03}_s{}_{version}", tenths(self.center_deg), self.index)
    }

    /// The label of the entry in the log and the sidecar, `library <center, 1 decimal> deg #<index>`.
    pub fn label(&self) -> String {
        format!("library {:.1} deg #{}", self.center_deg, self.index)
    }

    /// The body the entry is baked on ([`library_body`] of its center).
    pub fn body(&self) -> BodyInputs {
        library_body(self.center_deg)
    }
}

/// Every entry of the library for `k` seeds per class, in `bake.main`'s order (class by class).
pub fn library_entries(k: usize) -> Vec<LibraryEntry> {
    LIBRARY_CLASSES
        .iter()
        .enumerate()
        .flat_map(|(class_index, &(_, _, center_deg))| (0..k).map(move |index| LibraryEntry { class_index, center_deg, index }))
        .collect()
}

/// The file stem of a list entry, `cloud_<slug(key)>_<version>`.
pub fn list_stem(key: &str, version: &str) -> String {
    format!("cloud_{}_{version}", slug(key))
}

/// The width a list body is baked at: the `--list-width` override (0 counts as absent, as Python's `or`), else
/// the texel rule of `climate::map_width_for_radius` capped at [`LIST_MAX_WIDTH`].
pub fn list_width(radius_km: f64, override_width: Option<usize>) -> usize {
    match override_width {
        Some(w) if w != 0 => w,
        _ => map_width_for_radius(radius_km).min(LIST_MAX_WIDTH as i64) as usize,
    }
}

/// One baked entry: what `bake_one` produces before it touches the disk.
#[derive(Clone, Debug, PartialEq)]
pub struct BakedEntry {
    /// The label of the entry (`meta['label']`).
    pub label: String,
    /// The generated maps and the sidecar dict.
    pub generated: GenOutput,
    /// The decoded files of the entry and its manifest record.
    pub encoded: EncodedEntry,
}

/// `bake.bake_one(body, seed, width, out, stem, prefix, label, surface)` without the files: the coverage and
/// derived maps of the body at the default [`GenParams`] (its `surface` enters when given), encoded as the
/// entry's images and manifest record (paths under `prefix`, the slice at [`SLICE_WIDTH`]).
///
/// # Errors
/// [`GenerateError::NotBakeable`] for a body whose clouds are not water.
pub fn bake_one(
    body: &BodyInputs,
    seed: u64,
    width: usize,
    stem: &str,
    prefix: &str,
    label: &str,
    surface: Option<&Surface>,
) -> Result<BakedEntry, GenerateError> {
    let generated = generate(body, seed, width, &GenParams::default(), surface, true)?;
    let extras = generated.extras.as_ref().expect("generate returns the extras it was asked for");
    let (h, w) = (generated.height, generated.width);
    let encoded = encode_entry(
        Plane::new(&generated.coverage, h, w),
        &EntryExtras {
            high_share: Plane::new(&extras.high_share, h, w),
            type_: Plane::new(&extras.kind, h, w),
            wind: Plane::new(&extras.wind, extras.wind_height, extras.wind_width),
        },
        stem,
        prefix,
        seed,
        SLICE_WIDTH,
    );
    Ok(BakedEntry { label: label.to_string(), generated, encoded })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::climate::{hadley_edge_deg, has_baroclinic_eddies, EARTH};

    #[test]
    fn the_constants_are_the_python_ones() {
        assert_eq!(LIST_MAX_WIDTH, 8192);
        assert_eq!(LIBRARY_SCALE_HEIGHT_KM, 11.1);
        assert_eq!(SLICE_WIDTH, 2048);
        assert_eq!(LIBRARY_CLASSES.map(|c| c.2), [20.0, 30.0, 42.5, 70.0]);
        // The classes tile [0, 91): each starts where the previous one ends.
        assert_eq!(LIBRARY_CLASSES[0].0, 0.0);
        assert!(LIBRARY_CLASSES.windows(2).all(|w| w[0].1 == w[1].0));
        assert!(LIBRARY_CLASSES.iter().all(|c| c.0 <= c.2 && c.2 < c.1), "a class is baked inside its own range");
        assert_eq!((DEFAULT_LIBRARY_WIDTH, DEFAULT_LIBRARY_SEEDS), (4096, 4));
    }

    #[test]
    fn a_library_body_lands_its_hadley_edge_on_the_center() {
        for &(_, _, center) in &LIBRARY_CLASSES {
            let b = library_body(center);
            assert!((hadley_edge_deg(&b) - center).abs() < 1e-9 * center, "{center}: {}", hadley_edge_deg(&b));
            assert_eq!((b.radius_km, b.gravity_mps2, b.scale_height_km), (6371.0, 9.81, 11.1));
            assert!(crate::climate::is_bakeable(&b), "a library world makes water clouds");
        }
        // The global cell has no storm track; the three others do.
        let eddies: Vec<bool> = LIBRARY_CLASSES.iter().map(|c| has_baroclinic_eddies(&library_body(c.2))).collect();
        assert_eq!(eddies, [true, true, true, false]);
    }

    #[test]
    fn the_period_is_earths_scaled_by_the_center_over_the_edge_at_earth_spin() {
        // gh_ratio = 11.1 / 8 = 1.3875, the edge at Earth's spin 30 sqrt(1.3875) = 35.3...: a center of that
        // edge spins like Earth.
        let edge_at_earth = 30.0 * (11.1_f64 / 8.0).sqrt();
        let p = library_body(edge_at_earth).rotation_period_days.unwrap();
        assert!((p - EARTH.rotation_period_days.unwrap()).abs() < 1e-12, "{p}");
        // The period is linear in the center.
        let (a, b) = (library_body(20.0).rotation_period_days.unwrap(), library_body(40.0).rotation_period_days.unwrap());
        assert!((b - 2.0 * a).abs() < 1e-12);
    }

    #[test]
    fn centers_outside_the_edge_range_clamp_and_the_degenerate_ones_keep_python_behavior() {
        assert_eq!(hadley_edge_deg(&library_body(5.0)), 10.0);
        assert_eq!(hadley_edge_deg(&library_body(120.0)), 90.0);
        // 0: a zero period is no spin (edge 90, no storm track).
        let zero = library_body(0.0);
        assert_eq!(zero.rotation_period_days, Some(0.0));
        assert_eq!(hadley_edge_deg(&zero), 90.0);
        // Negative: retrograde, and the edge uses the magnitude of the period.
        let retro = library_body(-30.0);
        assert!(retro.rotation_period_days.unwrap() < 0.0);
        assert!((hadley_edge_deg(&retro) - 30.0).abs() < 1e-9);
        // The period of -c is minus the period of c.
        assert_eq!(retro.rotation_period_days.unwrap(), -library_body(30.0).rotation_period_days.unwrap());
    }

    #[test]
    fn library_entries_are_named_and_seeded_as_bake_main_does() {
        let all = library_entries(4);
        assert_eq!(all.len(), 16);
        assert_eq!((all[0].class_index, all[0].index, all[15].class_index, all[15].index), (0, 0, 3, 3));
        // Python: 1000 * int(center * 10) + i, `cloud_library_h{int(center*10):03d}_s{i}_{version}`,
        // `library {center:.1f} deg #{i}`.
        let seeds: Vec<u64> = all.iter().map(LibraryEntry::seed).collect();
        assert_eq!(&seeds[..4], &[200_000, 200_001, 200_002, 200_003]);
        assert_eq!(&seeds[4..8], &[300_000, 300_001, 300_002, 300_003]);
        assert_eq!(&seeds[8..12], &[425_000, 425_001, 425_002, 425_003]);
        assert_eq!(&seeds[12..], &[700_000, 700_001, 700_002, 700_003]);
        assert_eq!(all[0].stem("v1"), "cloud_library_h200_s0_v1");
        assert_eq!(all[9].stem("v2"), "cloud_library_h425_s1_v2");
        assert_eq!(all[15].stem("v1"), "cloud_library_h700_s3_v1");
        assert_eq!(all[8].label(), "library 42.5 deg #0");
        assert_eq!(all[0].label(), "library 20.0 deg #0");
        // A center under 10 degrees pads to three digits; a fractional tenth truncates.
        let small = LibraryEntry { class_index: 0, center_deg: 5.0, index: 2 };
        assert_eq!(small.stem("v1"), "cloud_library_h050_s2_v1");
        let odd = LibraryEntry { class_index: 0, center_deg: 12.99, index: 0 };
        assert_eq!((odd.seed(), odd.stem("v1").as_str()), (129_000, "cloud_library_h129_s0_v1"));
        assert!(library_entries(0).is_empty());
    }

    #[test]
    #[should_panic(expected = "not negative")]
    fn a_negative_center_has_no_seed() {
        let _ = LibraryEntry { class_index: 0, center_deg: -5.0, index: 0 }.seed();
    }

    #[test]
    fn list_stems_and_widths() {
        assert_eq!(list_stem("Dev:Gaia", "v1"), "cloud_dev_gaia_v1");
        assert_eq!(list_stem("Sol:Earth Moon", "v3"), "cloud_sol_earth_moon_v3");
        // Earth: 2 pi 6371 / 3 = 13343 texels, rounded up to 16384, capped at 8192. A small body is not capped.
        assert_eq!(list_width(6371.0, None), 8192);
        assert_eq!(list_width(6371.0, Some(0)), 8192, "0 is absent, as Python's `or`");
        assert_eq!(list_width(6371.0, Some(512)), 512);
        assert_eq!(list_width(1000.0, None), 4096, "2 pi 1000 / 3 = 2094 texels, rounded up to a power of two");
    }

    #[test]
    fn bake_one_encodes_an_entry_with_its_manifest_record() {
        let b = bake_one(&library_body(30.0), 300_000, 64, "cloud_library_h300_s0_v1", "baked/v1", "library 30.0 deg #0", None).unwrap();
        assert_eq!(b.label, "library 30.0 deg #0");
        assert_eq!((b.generated.width, b.generated.height, b.generated.meta.seed), (64, 32, 300_000));
        // The map is narrower than the 2048 slice: the images are the maps themselves.
        assert_eq!((b.encoded.slice.width, b.encoded.slice.height), (64, 32));
        assert_eq!(b.encoded.hires_south_up.len(), 64 * 32);
        let wind = b.generated.extras.as_ref().unwrap();
        assert_eq!((b.encoded.wind.width, b.encoded.wind.height, b.encoded.wind.channels), (wind.wind_width, wind.wind_height, 3));
        let rec = &b.encoded.record;
        assert_eq!(rec.seed, 300_000);
        assert_eq!((rec.hires.width, rec.hires.height), (64, 32));
        assert_eq!(rec.hires.url, "baked/v1/cloud_library_h300_s0_v1_coverage_64x32.r8.gz");
        assert_eq!(rec.slice, "baked/v1/cloud_library_h300_s0_v1_coverage_2048.png");
        assert_eq!(b.encoded.files.wind, format!("cloud_library_h300_s0_v1_wind_{}x{}.png", wind.wind_width, wind.wind_height));
        // The hires bytes are the coverage's, south-up.
        let first_row_north: Vec<u8> = b.generated.coverage[..64].iter().map(|&v| (v * 255.0).round_ties_even() as u8).collect();
        assert_eq!(&b.encoded.hires_south_up[31 * 64..], &first_row_north[..]);
    }

    #[test]
    fn bake_one_refuses_a_body_whose_clouds_are_not_water() {
        let mars = BodyInputs::new(Some(1.026), 3389.5, 3.721, 11.1).with_regime("thin water ice");
        let err = bake_one(&mars, 1, 16, "s", "p", "mars", None).unwrap_err();
        assert!(matches!(&err, GenerateError::NotBakeable { cloud_regime } if cloud_regime == "thin water ice"));
        assert!(err.to_string().contains("thin water ice"));
    }
}
