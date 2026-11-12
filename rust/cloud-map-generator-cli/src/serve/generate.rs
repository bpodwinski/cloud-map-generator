//! One generation of `POST /api/generate` (`run_generation` of `ui/server.py`): the service's run
//! ([`cloud_map_generator_service::generate`]: the map from an engine, the match to Earth, the encodings, the measures) kept
//! in the result store and answered as the API's JSON.

use super::json::opt_str;
use super::request::{bad, ApiError, GenRequest};
use super::store::StoredResult;
use super::AppState;
use crate::pyjson::J;
use cloud_map_generator_core::flow::VortexKind;
use cloud_map_generator_core::generate::{py_round, GenMeta};
use cloud_map_generator_service::generate::{compute_json, metric_table, round5, run_generation as run, Job, RunError};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// The message of the 409 a stopped generation answers (`POST /api/generate/cancel`); the page tells it from a failure by the status.
pub const CANCELLED: &str = "the generation was cancelled";

/// `wrap_lon_deg`: a longitude in [-180, 180) (the flow places its lows at lon0 + k 2 pi / m, past 180 for the last ones).
pub fn wrap_lon_deg(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

fn kind_name(k: VortexKind) -> &'static str {
    match k {
        VortexKind::Cyclone => "cyclone",
        VortexKind::Anticyclone => "anticyclone",
        VortexKind::Tropical => "tropical",
    }
}

/// The `meta` block of an answer: the Hadley edge, the octaves, the flow and its vortices, the surface.
fn meta_json(m: &GenMeta) -> J {
    J::obj([
        ("hadley_edge_deg", J::Float(m.hadley_edge_deg)),
        ("fine_octaves", J::Int(m.fine_octaves as i64)),
        (
            "flow",
            J::obj([
                ("jet", J::Float(m.flow.jet)),
                ("trades", J::Float(m.flow.trades)),
                ("polar_easterlies", J::Float(m.flow.polar_easterlies)),
                (
                    "vortices",
                    J::Arr(
                        m.flow
                            .vortices
                            .iter()
                            .map(|v| {
                                J::obj([
                                    ("kind", J::s(kind_name(v.kind))),
                                    ("lon_deg", J::Float(py_round(wrap_lon_deg(v.lon_deg), 2))),
                                    ("lat_deg", J::Float(py_round(v.lat_deg, 2))),
                                    ("radius", J::Float(py_round(v.radius, 4))),
                                    ("omega", J::Float(py_round(v.omega, 4))),
                                    ("ascent", J::Float(py_round(v.ascent, 4))),
                                    ("age", J::Float(py_round(v.age, 4))),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
        ),
        (
            "surface",
            m.surface.as_ref().map_or(J::Null, |s| J::obj([("land_fraction", J::Float(s.land_fraction)), ("itcz_shift_deg", J::Float(s.itcz_shift_deg))])),
        ),
    ])
}

/// The whole job, as `gen.py` then `measure.py` do it. The caller holds the generation lock.
pub fn run_generation(state: &AppState, req: &GenRequest) -> Result<J, ApiError> {
    let t0 = Instant::now();
    let reference = state.reference()?;
    let rid = state.results.new_id();
    if req.width >= cloud_map_generator_service::large::LARGE_FROM_WIDTH || req.compute == cloud_map_generator_service::engine::ComputeMode::Hybrid {
        return run_large_generation(state, req, reference, rid, t0);
    }
    let job = Job {
        body: req.body.clone(),
        seed: req.seed,
        width: req.width,
        options: req.options,
        compute: req.compute,
        stem: "coverage".to_string(),
        sidecar_seconds: true,
    };
    let log = |fraction: f64, msg: &str| state.log(&format!("[gen] {:3.0} % {msg}", 100.0 * fraction));
    let cancelled = || state.cancel.load(Ordering::Relaxed);
    let g = run(&state.runtime, reference, &job, &log, &cancelled).map_err(|e| match e {
        RunError::Refused(m) => bad(m),
        RunError::Failed(m) => ApiError::new(500, m),
        RunError::Cancelled => ApiError::new(409, CANCELLED),
    })?;
    let (height, width) = (g.height, g.width);
    let stem = format!("cloudgen_s{}_{width}x{height}_{rid}", req.seed);
    let files = HashMap::from([
        ("coverage.png", g.files.coverage_png.clone()),
        ("type.png", g.files.type_png.clone()),
        ("high_share.png", g.files.high_share_png.clone()),
        ("wind.png", g.files.wind_png.clone()),
        ("master16.png", g.files.master16_png.clone()),
        ("coverage.r8.gz", g.files.r8_gz.clone()),
        ("meta.json", g.files.meta_json.clone()),
    ]);

    // Stored last: a result exists only when the whole answer could be built.
    state.results.add(
        &rid,
        StoredResult {
            files,
            names: HashMap::from([
                ("master16.png", format!("{stem}_16bit.png")),
                ("coverage.r8.gz", format!("{stem}.r8.gz")),
                ("meta.json", format!("{stem}.json")),
            ]),
        },
    );
    let total_seconds = t0.elapsed().as_secs_f64();
    let seconds = py_round(total_seconds, 2);
    let href = |name: &str| format!("/api/result/{rid}/{name}");
    let floats = |v: &[f64]| J::Arr(v.iter().map(|&x| J::Float(round5(x))).collect());
    let m = &g.measured;
    Ok(J::obj([
        ("id", J::s(rid.as_str())),
        ("seconds", J::Float(seconds)),
        ("width", J::Int(width as i64)),
        ("height", J::Int(height as i64)),
        ("compute", compute_json(req.compute.key(), &g.compute, Some(total_seconds))),
        ("meta", meta_json(&g.output.meta)),
        ("images", J::obj(["coverage", "type", "high_share", "wind"].map(|k| (k, J::s(href(&format!("{k}.png"))))))),
        (
            "downloads",
            J::Arr(vec![
                J::obj([("label", J::s("Cloud fraction, 16-bit PNG")), ("href", J::s(href("master16.png")))]),
                J::obj([("label", J::s("Raw R8 for the shader (gzip, rows south-up)")), ("href", J::s(href("coverage.r8.gz")))]),
                J::obj([("label", J::s("Sidecar JSON")), ("href", J::s(href("meta.json")))]),
            ]),
        ),
        (
            "metrics",
            J::obj([
                ("table", metric_table(&m.measures, &m.comparison, &reference.measures)),
                ("earth_label", J::s("Earth (Blue Marble)")),
                ("note", opt_str(g.metrics_note.as_deref())),
            ]),
        ),
        (
            "zonal",
            J::obj([("lat_deg", J::floats(&m.zonal.lat_deg)), ("generated", floats(&m.zonal.generated)), ("earth", floats(&m.zonal.earth)), ("target", floats(&m.zonal.target))]),
        ),
        ("warnings", J::Arr(g.warnings.into_iter().map(J::Str).collect())),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longitudes_are_wrapped_for_the_page() {
        assert!((wrap_lon_deg(197.71) - -162.29).abs() < 1e-9);
        assert_eq!(wrap_lon_deg(-180.0), -180.0);
        assert!((wrap_lon_deg(12.3) - 12.3).abs() < 1e-12);
        assert_eq!(wrap_lon_deg(540.0), -180.0);
    }
}

/// A large map (from [`cloud_map_generator_service::large::LARGE_FROM_WIDTH`]): the banded engine writes its files to
/// `<output>/serve/<id>/`, the page gets the 2048-wide slices from memory and the paths of the big files. The full image
/// is never sent: a 32768-wide map is 1 GiB.
fn run_large_generation(state: &AppState, req: &GenRequest, reference: &cloud_map_generator_service::reference::Reference, rid: String, t0: Instant) -> Result<J, ApiError> {
    use cloud_map_generator_service::large::{run_large, LargeJob};
    let dir = state.project.output().join("serve").join(&rid);
    let job = LargeJob {
        body: req.body.clone(),
        seed: req.seed,
        width: req.width,
        options: req.options,
        compute: req.compute,
        stem: "coverage".to_string(),
        out_dir: dir.clone(),
        surface: None,
        band_rows: None,
        sidecar_seconds: true,
    };
    let log = |fraction: f64, msg: &str, _: Option<cloud_map_generator_core::banded::Progress>| state.log(&format!("[gen] {:3.0} % {msg}", 100.0 * fraction));
    let cancelled = || state.cancel.load(Ordering::Relaxed);
    let g = run_large(&state.runtime, reference, &job, &log, &cancelled).map_err(|e| match e {
        RunError::Refused(m) => bad(m),
        RunError::Failed(m) => ApiError::new(500, m),
        RunError::Cancelled => ApiError::new(409, CANCELLED),
    })?;
    let (height, width) = (g.height, g.width);
    let read = |key: &str| -> Vec<u8> { g.files.iter().find(|f| f.key == key).and_then(|f| std::fs::read(&f.path).ok()).unwrap_or_default() };
    let files = HashMap::from([
        ("coverage.png", read("preview")),
        ("type.png", read("type")),
        ("high_share.png", read("high_share")),
        ("wind.png", read("wind")),
        ("meta.json", read("meta")),
    ]);
    let stem = format!("cloudgen_s{}_{width}x{height}_{rid}", req.seed);
    state.results.add(&rid, StoredResult { files, names: HashMap::from([("meta.json", format!("{stem}.json"))]) });
    let total_seconds = t0.elapsed().as_secs_f64();
    let href = |name: &str| format!("/api/result/{rid}/{name}");
    let floats = |v: &[f64]| J::Arr(v.iter().map(|&x| J::Float(round5(x))).collect());
    let m = &g.measured;
    let file_json = |label: &str, key: &str| {
        let f = g.files.iter().find(|f| f.key == key).expect("a file of the run");
        J::obj([("label", J::s(label)), ("key", J::s(key)), ("path", J::s(f.path.display().to_string())), ("bytes", J::Int(f.bytes as i64))])
    };
    Ok(J::obj([
        ("id", J::s(rid.as_str())),
        ("seconds", J::Float(py_round(total_seconds, 2))),
        ("width", J::Int(width as i64)),
        ("height", J::Int(height as i64)),
        ("compute", compute_json(req.compute.key(), &g.compute, Some(total_seconds))),
        ("meta", meta_json(&g.meta)),
        ("images", J::obj(["coverage", "type", "high_share", "wind"].map(|k| (k, J::s(href(&format!("{k}.png"))))))),
        ("image_note", J::s(format!("The images are the {}-px downsample of the {width}-px map: the full map is in the files below, never sent to the page.", g.preview.width))),
        ("downloads", J::Arr(vec![J::obj([("label", J::s("Sidecar JSON")), ("href", J::s(href("meta.json")))])])),
        ("files", J::Arr(vec![file_json("Cloud fraction, 16-bit PNG", "master"), file_json("Raw R8 for the shader (gzip, rows south-up)", "r8gz")])),
        ("banded", J::obj([("band_rows", J::Int(g.band_rows as i64)), ("bands", J::Int(g.bands as i64)), ("dir", J::s(dir.display().to_string()))])),
        (
            "metrics",
            J::obj([
                ("table", metric_table(&m.measures, &m.comparison, &reference.measures)),
                ("earth_label", J::s("Earth (Blue Marble)")),
                ("note", opt_str(g.metrics_note.as_deref())),
            ]),
        ),
        (
            "zonal",
            J::obj([("lat_deg", J::floats(&m.zonal.lat_deg)), ("generated", floats(&m.zonal.generated)), ("earth", floats(&m.zonal.earth)), ("target", floats(&m.zonal.target))]),
        ),
        ("warnings", J::Arr(g.warnings.into_iter().map(J::Str).collect())),
    ]))
}
