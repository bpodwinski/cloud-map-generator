//! End-to-end tests of the `cloud-map-generator` binary: each subcommand is run as a person would run it, on small sizes,
//! and what it writes is read back with the PNG and gzip decoders.
//!
//! The project root (the shipped reference map and the example bodies) is `$CLOUD_MAP_GENERATOR_ROOT`, else the repository
//! this crate sits in (`../..`). A root without `data/earth_cloud_fraction_bluemarble_2048.png` fails the tests
//! that need it, loudly: they do not skip.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn root() -> PathBuf {
    std::env::var_os("CLOUD_MAP_GENERATOR_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(".."))
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cloud-map-generator-cli-e2e-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cloudgen(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cloud-map-generator")).args(args).env("CLOUD_MAP_GENERATOR_ROOT", root()).output().expect("the binary runs")
}

fn ok(args: &[&str]) -> String {
    let out = cloudgen(args);
    assert!(out.status.success(), "cloudgen {args:?}: {}\n{}", out.status, String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

/// A PNG decoded to its info and its bytes (no transformation: 16-bit samples stay big-endian).
fn png(path: &Path) -> (png::OutputInfo, Vec<u8>) {
    let mut reader = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap())).read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    (info, buf)
}

fn gunzip(path: &Path) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap()).read_to_end(&mut out).unwrap();
    out
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn gen_writes_the_files_gen_py_writes() {
    let out = scratch("gen");
    let said = ok(&["gen", "--seed", "3", "--width", "64", "--out", out.to_str().unwrap()]);
    let lines: Vec<&str> = said.lines().collect();
    assert_eq!(lines[0], "earth seed 3 width 64");
    // gen.py prints the products first (type, high_share, wind), then master, preview, r8gz, meta.
    let keys: Vec<&str> = lines[1..].iter().map(|l| l.trim_start().split(':').next().unwrap()).collect();
    assert_eq!(keys, ["type", "high_share", "wind", "master", "preview", "r8gz", "meta"]);
    let mut names: Vec<String> = std::fs::read_dir(&out).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    assert_eq!(
        names,
        [
            "coverage_earth_s3.json",
            "coverage_earth_s3_64x32.png",
            "coverage_earth_s3_64x32.r8.gz",
            "coverage_earth_s3_64x32_16bit.png",
            "coverage_earth_s3_high_share_64x32.png",
            "coverage_earth_s3_type_64x32.png",
            "coverage_earth_s3_wind_64x32.png"
        ]
    );

    // The preview is 8-bit gray, the master 16-bit gray; the master's high byte is within one level of the
    // preview (the same coverage, quantized twice); the raw R8 is the preview with its rows reversed.
    let (pi, preview) = png(&out.join("coverage_earth_s3_64x32.png"));
    assert_eq!((pi.width, pi.height, pi.color_type, pi.bit_depth), (64, 32, png::ColorType::Grayscale, png::BitDepth::Eight));
    let (mi, master) = png(&out.join("coverage_earth_s3_64x32_16bit.png"));
    assert_eq!((mi.width, mi.height, mi.color_type, mi.bit_depth), (64, 32, png::ColorType::Grayscale, png::BitDepth::Sixteen));
    for (k, p) in preview.iter().enumerate() {
        let m = u16::from_be_bytes([master[2 * k], master[2 * k + 1]]);
        assert!((f64::from(m) / 257.0 - f64::from(*p)).abs() <= 0.5 + 1e-9, "texel {k}: master {m}, preview {p}");
    }
    let r8 = gunzip(&out.join("coverage_earth_s3_64x32.r8.gz"));
    let flipped: Vec<u8> = preview.chunks(64).rev().flatten().copied().collect();
    assert_eq!(r8, flipped, "rows south-up");
    let (ti, _) = png(&out.join("coverage_earth_s3_type_64x32.png"));
    assert_eq!((ti.width, ti.height, ti.color_type), (64, 32, png::ColorType::Grayscale));
    // The wind is RGB on its own grid (the back-trace grid, here as wide as the map).
    let (wi, _) = png(&out.join("coverage_earth_s3_wind_64x32.png"));
    assert_eq!((wi.width, wi.height, wi.color_type), (64, 32, png::ColorType::Rgb));

    let meta = json(&out.join("coverage_earth_s3.json"));
    assert_eq!((meta["seed"].as_u64(), meta["width"].as_u64(), meta["height"].as_u64()), (Some(3), Some(64), Some(32)));
    assert_eq!(meta["body_name"], "earth");
    assert_eq!(meta["body"]["cloud_regime"], "water");
    assert_eq!(meta["products"]["files"]["type"], "coverage_earth_s3_type_64x32.png");
    // The key order is generate()'s, then gen.py's additions, then write_products's.
    let text = std::fs::read_to_string(out.join("coverage_earth_s3.json")).unwrap();
    let at = |key: &str| text.find(&format!("\n  \"{key}\":")).unwrap_or_else(|| panic!("no top-level key {key}"));
    assert!(at("seed") < at("body") && at("body") < at("params") && at("flow") < at("seconds") && at("seconds") < at("body_name") && at("body_name") < at("products"));
}

#[test]
fn gen_refuses_a_cloud_regime_that_is_not_water_unless_it_is_an_experiment() {
    let out = scratch("mars");
    let refused = cloudgen(&["gen", "--body", "mars", "--width", "32", "--out", out.to_str().unwrap()]);
    assert_eq!(refused.status.code(), Some(1));
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(err.contains("mars: clouds of thin water ice, not water") && err.contains("--experiment"), "{err}");
    assert!(!out.join("coverage_mars_s42_32x16.png").exists(), "a refusal writes nothing");
    ok(&["gen", "--body", "mars", "--width", "32", "--experiment", "--out", out.to_str().unwrap()]);
    assert!(out.join("coverage_mars_s42_32x16.png").exists());
}

#[test]
fn gen_isotropic_has_no_products_and_the_stem_carries_the_tag() {
    let out = scratch("iso");
    let said = ok(&["gen", "--width", "32", "--isotropic", "--out", out.to_str().unwrap()]);
    assert!(!said.contains("type:") && !said.contains("wind:"), "{said}");
    assert!(out.join("coverage_earth_s42_iso_32x16.png").exists());
    assert!(out.join("coverage_earth_s42_iso.json").exists());
    let meta = json(&out.join("coverage_earth_s42_iso.json"));
    assert!(meta.get("products").is_none());
    assert_eq!(meta["params"]["isotropic"], true);
}

#[test]
fn measure_of_the_reference_reads_like_measure_py() {
    // The reference measured against itself: the numbers measure.py prints for its own reference row.
    let reference = root().join("data").join("earth_cloud_fraction_bluemarble_2048.png");
    assert!(reference.is_file(), "the reference map is missing: {}", reference.display());
    let said = ok(&["measure", reference.to_str().unwrap()]);
    let lines: Vec<&str> = said.lines().collect();
    assert_eq!(lines[0], format!("{:<40} {:>5} {:>7} {:>6} {:>5} {:>6} {:>5} {:>7} {:>5} {:>5} {:>5}", "map", "mean", "zonRMSE", "histL1", "coher", "slope", "big%", "medKm2", "seam", "poleN", "poleS"));
    let want = "0.404   0.000  0.000 0.487  -2.90  80.3     308  1.39 0.000 0.029";
    assert_eq!(lines[1], format!("{:<40} {want}", "earth (Blue Marble)"));
    assert!(lines[2].starts_with("earth_cloud_fraction_bluemarble_2048"), "{}", lines[2]);
    assert!(lines[2].ends_with(want), "a map compared with itself: zero zonal RMSE and histogram distance\n{}", lines[2]);
    assert_eq!(lines[3], format!("  {:<30} hist 0.31 0.08 0.07 0.06 0.06 0.06 0.07 0.08 0.09 0.10", "earth (Blue Marble)"));
    assert!(lines[4].ends_with("coherence by lat -55:0.51 -40:0.50 -20:0.47 0:0.46 20:0.45 40:0.52 55:0.50"), "{}", lines[4]);
}

#[test]
fn measure_writes_its_json_and_its_sheet() {
    let out = scratch("measure");
    let reference = root().join("data").join("earth_cloud_fraction_bluemarble_2048.png");
    let (j, s) = (out.join("m.json"), out.join("sheet.png"));
    let said = ok(&["measure", reference.to_str().unwrap(), "--json", j.to_str().unwrap(), "--sheet", s.to_str().unwrap()]);
    assert!(said.contains("sheet: "), "{said}");
    let v = json(&j);
    let m = &v["earth (Blue Marble)"];
    assert!(m["vs_earth"].is_null());
    assert!((m["metrics"]["global_mean"].as_f64().unwrap() - 0.404).abs() < 5e-4);
    // The keys are the patch latitudes as Python's json writes int keys (strings), in PATCH_LATS order in the file
    // (serde_json sorts them on reading, so the order is checked on the text).
    let mut keys: Vec<&String> = m["metrics"]["coherence_by_lat"].as_object().unwrap().keys().collect();
    keys.sort();
    assert_eq!(keys, ["-20", "-40", "-55", "0", "20", "40", "55"]);
    let text = std::fs::read_to_string(&j).unwrap();
    let pos: Vec<usize> = ["\"-55\":", "\"-40\":", "\"-20\":", "\"0\":", "\"20\":", "\"40\":", "\"55\":"].iter().map(|k| text.find(k).unwrap()).collect();
    assert!(pos.windows(2).all(|p| p[0] < p[1]), "{pos:?}");
    assert!(v["earth_cloud_fraction_bluemarble_2048"]["vs_earth"]["zonal_rmse"].as_f64().unwrap() < 1e-12);
    // The sheet has a row per map, the reference first and then the one given: two rows of 256 pixels and a 6-pixel
    // separator, 512 + 8 x 256 wide.
    let (info, _) = png(&s);
    assert_eq!((info.width, info.height), (512 + 8 * 256, 2 * 256 + 6));
}

#[test]
fn tiles_bake_a_rank_tile() {
    let out = scratch("tiles");
    let said = ok(&["tiles", "--version", "t", "--seed", "2", "--size", "64", "--out", out.to_str().unwrap()]);
    assert!(said.starts_with("cloud_cell_tiles_64_t: "), "{said}");
    let mut names: Vec<String> = std::fs::read_dir(&out).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    assert_eq!(names, ["cloud_cell_tiles_64_t.json", "cloud_cell_tiles_64_t.rgba.gz", "cloud_cell_tiles_64_t_preview.png"]);
    let tile = gunzip(&out.join("cloud_cell_tiles_64_t.rgba.gz"));
    assert_eq!(tile.len(), 64 * 64 * 4);
    // Three of the four channels are a rank of distinct values: every byte level appears 64 * 64 / 256 times. (The
    // open-cell channel has plateaus, whose texels share a mean rank.)
    for c in [0, 1, 3] {
        let mut hist = [0usize; 256];
        tile.chunks(4).for_each(|px| hist[px[c] as usize] += 1);
        assert!(hist.iter().all(|&n| n == 16), "channel {c}: {hist:?}");
    }
    let (info, preview) = png(&out.join("cloud_cell_tiles_64_t_preview.png"));
    assert_eq!((info.width, info.height, info.color_type), (256, 256, png::ColorType::Grayscale));
    // The first column of each band is the channel's rank itself.
    for c in 0..4 {
        for r in 0..64 {
            for x in 0..64 {
                assert_eq!(preview[(c * 64 + r) * 256 + x], tile[(r * 64 + x) * 4 + c]);
            }
        }
    }
    let meta = json(&out.join("cloud_cell_tiles_64_t.json"));
    assert_eq!((meta["size"].as_u64(), meta["seed"].as_u64(), meta["version"].as_str()), (Some(64), Some(2), Some("t")));
    assert_eq!(meta["cdn"], "clouds/tiles/t/cloud_cell_tiles_64_t.rgba.gz");
    // The same seed gives the same tile; another seed another.
    let again = scratch("tiles2");
    ok(&["tiles", "--version", "t", "--seed", "2", "--size", "64", "--out", again.to_str().unwrap()]);
    assert_eq!(gunzip(&again.join("cloud_cell_tiles_64_t.rgba.gz")), tile);
    ok(&["tiles", "--version", "t", "--seed", "3", "--size", "64", "--out", again.to_str().unwrap()]);
    assert_ne!(gunzip(&again.join("cloud_cell_tiles_64_t.rgba.gz")), tile);
}

#[test]
fn bake_a_small_bodies_file_names_files_and_manifest_as_bake_py_does() {
    let dir = scratch("bake");
    let bodies = dir.join("bodies.json");
    std::fs::write(
        &bodies,
        r#"{"bodies": [
            {"key": "Dev:Gaia", "rotationPeriodDays": 0.99727, "radiusKm": 6371, "gravityMps2": 9.81, "scaleHeightKm": 8.4, "cloudRegime": "water"},
            {"key": "Sol:Mars", "rotationPeriodDays": 1.02596, "radiusKm": 3389.5, "gravityMps2": 3.721, "scaleHeightKm": 11.1, "cloudRegime": "thin water ice"},
            {"key": "Sol:Earth", "rotationPeriodDays": 0.99727, "radiusKm": 6371, "gravityMps2": 9.81, "scaleHeightKm": 8.4, "cloudRegime": "water", "measured": true}
        ]}"#,
    )
    .unwrap();
    let out = dir.join("out");
    let said = ok(&[
        "bake", "--version", "v9", "--bodies", bodies.to_str().unwrap(), "--list-width", "64", "--library-width", "64", "--k", "1", "--out", out.to_str().unwrap(),
    ]);
    assert!(said.contains("Sol:Mars                     skipped: clouds of thin water ice"), "{said}");
    assert!(said.contains("Sol:Earth                    skipped: measured map"), "{said}");
    assert!(said.contains("library 42.5 deg #0"), "{said}");
    assert!(said.contains("manifest: "), "{said}");

    let manifest = json(&out.join("manifest.json"));
    assert_eq!(manifest["version"], "v9");
    assert_eq!(manifest["generatedBy"], "planet-cloudgen bake.py");
    // The seed of a list entry is crc32(key) & 0x7fffffff (Python: zlib.crc32(b'Dev:Gaia') & 0x7fffffff).
    let gaia = &manifest["list"]["Dev:Gaia"];
    assert_eq!(gaia["seed"].as_u64(), Some(u64::from(cloudgen_key_seed("Dev:Gaia"))));
    assert_eq!(gaia["hires"]["url"], "baked/v9/cloud_dev_gaia_v9_coverage_64x32.r8.gz");
    assert_eq!((gaia["hires"]["width"].as_u64(), gaia["hires"]["height"].as_u64()), (Some(64), Some(32)));
    assert_eq!(gaia["slice"], "baked/v9/cloud_dev_gaia_v9_coverage_2048.png");
    assert_eq!(gaia["highShare"], "baked/v9/cloud_dev_gaia_v9_high_share_2048.png");
    assert_eq!(gaia["type"], "baked/v9/cloud_dev_gaia_v9_type_2048.png");
    assert_eq!(gaia["wind"], "baked/v9/cloud_dev_gaia_v9_wind_64x32.png");
    assert_eq!(manifest["list"].as_object().unwrap().len(), 1, "the non-water and the measured bodies are not baked");
    let classes = manifest["library"].as_array().unwrap();
    assert_eq!(classes.len(), 4);
    assert_eq!(classes[2]["bakedAtDeg"].as_f64(), Some(42.5));
    assert_eq!((classes[0]["hadleyMinDeg"].as_f64(), classes[0]["hadleyMaxDeg"].as_f64()), (Some(0.0), Some(25.0)));
    assert_eq!(classes[2]["entries"][0]["seed"].as_u64(), Some(425_000));
    assert_eq!(classes[3]["entries"][0]["hires"]["url"], "baked/v9/cloud_library_h700_s0_v9_coverage_64x32.r8.gz");

    // Every file the manifest names exists, and the sidecar lists them under `files`.
    for url in [gaia["hires"]["url"].as_str().unwrap(), gaia["slice"].as_str().unwrap(), gaia["wind"].as_str().unwrap()] {
        assert!(out.join(url.trim_start_matches("baked/v9/")).is_file(), "{url}");
    }
    let side = json(&out.join("cloud_dev_gaia_v9.json"));
    assert_eq!(side["label"], "Dev:Gaia");
    assert_eq!(side["files"]["wind"], "cloud_dev_gaia_v9_wind_64x32.png");
    assert_eq!(side["encoding"]["hires"], "R8 cloud fraction 0..255, rows SOUTH-up, gzip");
    // The slice of a map narrower than 2048 is the map itself (the engine resamples).
    let (si, _) = png(&out.join("cloud_dev_gaia_v9_coverage_2048.png"));
    assert_eq!((si.width, si.height), (64, 32));

    // `--only` writes no manifest.
    let part = dir.join("part");
    let said = ok(&["bake", "--only", "library", "--version", "v9", "--library-width", "64", "--k", "1", "--out", part.to_str().unwrap()]);
    assert!(said.contains("partial run: manifest not written"), "{said}");
    assert!(!part.join("manifest.json").exists());
}

/// `zlib.crc32(key) & 0x7fffffff`, the bake seed of a key (here from the core: the test is that the CLI uses it).
fn cloudgen_key_seed(key: &str) -> u32 {
    cloud_map_generator_core::bake::key_seed(key)
}
