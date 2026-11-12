//! The real engine behind the server (`ServiceBackend` over `cloud-map-generator-service`), called directly: the files a map
//! writes, the rows and the zonal profile it reports, cancellation, measuring a PNG, the presets, the bake. The
//! machine under test has no GPU probe (`cpu_only`): the GPU is exercised by the service crate's own tests and by
//! the end-to-end run of the binary.

mod common;

use std::sync::Arc;

use cloud_map_generator_mcp::backend::{
    Backend, BackendError, BakeOnly, BakeRequest, GenerateOptions, GenerateRequest, Physics, Surface,
};
use cloud_map_generator_mcp::service_backend::ServiceBackend;
use common::TempDir;
use serde_json::Value;

fn backend() -> Arc<ServiceBackend> {
    ServiceBackend::cpu_only("No GPU in this test.")
}

fn earth() -> Physics {
    Physics::new(0.99727, 6371.0, 9.81, 8.0)
}

fn request(out: Option<&std::path::Path>, seed: u32, width: u32, options: GenerateOptions) -> GenerateRequest {
    GenerateRequest { physics: earth(), seed, width, options, out_dir: out.map(|p| p.to_path_buf()) }
}

fn cpu() -> GenerateOptions {
    GenerateOptions { compute: "cpu".into(), ..GenerateOptions::default() }
}

#[test]
fn a_map_is_made_measured_and_written_under_the_python_names() {
    let t = TempDir::new("service-gen");
    let out = t.join("out");
    let b = backend();
    let mut stages = Vec::new();
    let r = b
        .generate(&request(Some(&out), 7, 256, GenerateOptions { isotropic: true, ..cpu() }), &mut |f, m| stages.push((f, m.to_string())), &|| false)
        .unwrap();
    assert!(stages.windows(2).all(|w| w[0].0 <= w[1].0), "progress never decreases: {stages:?}");
    assert_eq!(stages.last().map(|s| s.0), Some(1.0));
    assert_eq!((r.width, r.height, r.seed), (256, 128, 7));
    assert_eq!(r.metrics.len(), 10);
    assert_eq!(r.metrics[0].key, "mean");
    assert!(r.metrics[0].generated.is_some() && r.metrics[0].earth.is_some());
    assert_eq!(r.zonal.lat_deg.len(), 36);
    assert_eq!(r.zonal.target.as_ref().map(Vec::len), Some(36));
    assert_eq!(r.coverage.nbytes(), 256 * 128);
    assert!(["type", "high_share", "wind"].iter().all(|k| r.extra_planes.contains_key(*k)));
    let compute = r.compute.as_ref().unwrap();
    assert_eq!((compute["requested"].as_str(), compute["used"].as_str()), (Some("cpu"), Some("cpu")));
    assert!(compute["devices"][0]["seconds"].as_f64().is_some());

    let names: Vec<(&str, String)> = r.files.iter().map(|(k, p)| (k.as_str(), p.file_name().unwrap().to_string_lossy().into_owned())).collect();
    for (key, name) in [
        ("master", "cloudgen_s7_256x128_16bit.png"),
        ("preview", "cloudgen_s7_256x128.png"),
        ("r8gz", "cloudgen_s7_256x128.r8.gz"),
        ("type", "cloudgen_s7_type_256x128.png"),
        ("high_share", "cloudgen_s7_high_share_256x128.png"),
        ("meta", "cloudgen_s7.json"),
        ("metrics", "cloudgen_s7_256x128_metrics.json"),
    ] {
        assert!(names.contains(&(key, name.to_string())), "{key} {name} in {names:?}");
    }
    assert!(names.iter().any(|(k, n)| *k == "wind" && n.starts_with("cloudgen_s7_wind_")));
    assert!(r.files.values().all(|p| p.is_file() && p.starts_with(&out)));

    // The sidecar has no wall time and records where the map ran; the metrics file has the table and the zonal profile.
    let meta: Value = serde_json::from_slice(&std::fs::read(&r.files["meta"]).unwrap()).unwrap();
    assert!(meta.get("seconds").is_none());
    assert_eq!(meta["options"]["isotropic"], true);
    assert_eq!(meta["compute"]["used"], "cpu");
    let metrics: Value = serde_json::from_slice(&std::fs::read(&r.files["metrics"]).unwrap()).unwrap();
    assert_eq!(metrics["metrics"].as_array().unwrap().len(), 10);
    assert_eq!(metrics["zonal"]["target"].as_array().unwrap().len(), 36);
    assert!(metrics["earth_reference"].as_str().unwrap().starts_with("Earth"));

    // The same inputs make the same bytes.
    let out2 = t.join("out2");
    let r2 = b.generate(&request(Some(&out2), 7, 256, GenerateOptions { isotropic: true, ..cpu() }), &mut |_, _| {}, &|| false).unwrap();
    for key in ["master", "meta", "r8gz"] {
        assert_eq!(std::fs::read(&r.files[key]).unwrap(), std::fs::read(&r2.files[key]).unwrap(), "{key}");
    }

    // Measuring the saved 16-bit master gives the same table the generation reported.
    let measured = b.measure_png(&r.files["master"]).unwrap();
    assert_eq!((measured.width, measured.height), (256, 128));
    let (a, m) = (r.metrics[0].generated.unwrap(), measured.metrics[0].generated.unwrap());
    assert!((a - m).abs() < 1e-3, "mean {a} against {m}");
    assert!(measured.zonal.target.is_none());
}

#[test]
fn a_cancelled_run_writes_nothing() {
    let t = TempDir::new("service-cancel");
    let out = t.join("out");
    let e = backend().generate(&request(Some(&out), 1, 256, cpu()), &mut |_, _| {}, &|| true).unwrap_err();
    assert_eq!(e, BackendError::Cancelled);
    assert!(!out.exists(), "nothing is written before the stages succeed");
}

#[test]
fn bad_requests_are_refused_with_the_services_words() {
    let b = backend();
    let mars = Physics::new(1.02596, 3389.5, 3.721, 11.1).with_regime("thin water ice");
    let e = b.generate(&GenerateRequest { physics: mars, ..request(None, 1, 256, cpu()) }, &mut |_, _| {}, &|| false).unwrap_err().to_string();
    assert!(e.starts_with("cloud_regime: clouds of thin water ice, not water"), "{e}");
    let fast = Physics::new(0.02, 6371.0, 9.81, 8.0);
    let e = b.generate(&GenerateRequest { physics: fast, ..request(None, 1, 256, cpu()) }, &mut |_, _| {}, &|| false).unwrap_err().to_string();
    assert!(e.starts_with("rotation_period_days: 0.02 days gives ") && e.contains("Use a rotation period of at least"), "{e}");
    let e = b.generate(&request(None, 1, 256, GenerateOptions { compute: "gpu".into(), ..cpu() }), &mut |_, _| {}, &|| false).unwrap_err().to_string();
    assert!(e.contains("'gpu' is not available: No GPU in this test."), "{e}");
    let e = b.measure_png(std::path::Path::new("no/such/file.png")).unwrap_err().to_string();
    assert!(e.starts_with("path: ") && e.contains("is not a file"), "{e}");
}

#[test]
fn the_presets_and_what_they_imply() {
    let b = backend();
    let presets = b.presets();
    let keys: Vec<&str> = presets.iter().map(|p| p.key.as_str()).collect();
    assert_eq!(keys, ["earth", "earth-retrograde", "fast-rotator", "slow-rotator", "super-earth", "mars", "titan", "venus"]);
    assert_eq!(presets.iter().map(|p| p.experiment_required).collect::<Vec<_>>(), [false, false, false, false, false, true, true, true]);
    let d = b.derive(&earth()).unwrap();
    assert_eq!((d.hadley_edge_deg, d.storm_track_deg, d.has_eddies, d.cyclones_per_hemisphere), (30.0, 51.0, true, Some(5)));
    assert_eq!((d.cyclone_spin_north.as_deref(), d.cyclone_spin_south.as_deref()), (Some("counterclockwise"), Some("clockwise")));
    assert!(d.warnings.is_empty() && d.bakeable && d.texel_km.len() == 5);
    let retro = b.derive(&Physics::new(-0.99727, 6371.0, 9.81, 8.0)).unwrap();
    assert_eq!(retro.cyclone_spin_north.as_deref(), Some("clockwise"));
    assert!(retro.warnings[0].starts_with("Retrograde spin"));
    let caps = b.capabilities();
    assert_eq!(caps["engine"], "rust");
    assert_eq!(caps["modes"].as_array().unwrap().len(), 4);
    assert_eq!(caps["modes"][2]["available"], false);
    assert!(b.explain_model().contains("the Rust one"));
}

#[test]
fn a_synthetic_surface_and_match_earth_are_honored() {
    let b = backend();
    let r = b
        .generate(&request(None, 3, 256, GenerateOptions { surface: Surface::Synthetic, match_earth: true, ..cpu() }), &mut |_, _| {}, &|| false)
        .unwrap();
    assert!(r.files.is_empty(), "no out_dir, no file");
    assert_eq!(r.options["surface"], "synthetic");
    assert!(r.warnings.iter().any(|w| w.starts_with("match_earth is built for maps of 2048 texels or more")), "{:?}", r.warnings);
}

#[test]
fn a_library_bake_writes_entries_and_a_manifest() {
    let t = TempDir::new("service-bake");
    let bodies = t.join("bodies.json");
    std::fs::write(
        &bodies,
        r#"{"bodies": [{"key": "earth", "rotationPeriodDays": 0.99727, "radiusKm": 6371, "gravityMps2": 9.81, "scaleHeightKm": 8.4, "cloudRegime": "water"}]}"#,
    )
    .unwrap();
    let out = t.join("bake").join("v1");
    let req = |only| BakeRequest {
        bodies_file: bodies.clone(),
        out_dir: out.clone(),
        version: "v1".into(),
        url_prefix: None,
        list_width: 256,
        library_width: 256,
        k: 1,
        only,
    };
    let mut stages = Vec::new();
    let r = backend().bake(&req(None), &mut |f, m| stages.push((f, m.to_string())), &|| false).unwrap();
    assert_eq!(r.manifest["version"], "v1");
    assert_eq!(r.manifest["list"]["earth"]["hires"]["width"], 256);
    assert_eq!(r.manifest["library"].as_array().unwrap().len(), 4);
    let manifest_path = r.manifest_path.clone().expect("a full bake writes the manifest");
    assert_eq!(manifest_path, out.join("manifest.json"));
    assert!(r.files.contains(&manifest_path) && r.files.iter().all(|p| p.is_file()));
    assert_eq!(r.files.len(), 5 * 6 + 1, "5 entries of 6 files and the manifest");
    assert!(stages.iter().any(|(_, m)| m.starts_with("baking earth (1 of 5)")), "{stages:?}");
    // A partial bake writes no manifest.
    let out2 = t.join("bake2");
    let r = backend().bake(&BakeRequest { out_dir: out2.clone(), ..req(Some(BakeOnly::List)) }, &mut |_, _| {}, &|| false).unwrap();
    assert!(r.manifest_path.is_none() && !out2.join("manifest.json").exists());
    // Cancelled before the first entry: nothing baked.
    let out3 = t.join("bake3");
    let e = backend().bake(&BakeRequest { out_dir: out3, ..req(None) }, &mut |_, _| {}, &|| true).unwrap_err();
    assert_eq!(e, BackendError::Cancelled);
}
