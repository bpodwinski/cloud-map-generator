//! Tests of the routes, called as functions (no socket): the contract of the API, deterministic and fast. The socket
//! itself (keep-alive, HEAD, `Content-Length`) is tested against the real binary in `tests/serve.rs`.

use super::*;
use crate::project::Project;
use crate::serve::engine::{ComputeUse, CpuEngine, DeviceKind, DeviceUse, Engine, EngineError, GenerateRequest, Generated, GpuInfo, GpuState, Progress, Runtime, Used};
use cloud_map_generator_service::images::decode_gray;
use crate::serve::{Config, DEFAULT_KEEP};
use serde_json::json;
use std::io::Read;
use std::sync::{Mutex, OnceLock};

/// The tests that generate share one state (the reference is measured once) and take turns.
static SERIAL: Mutex<()> = Mutex::new(());

fn config(widths: &[usize], keep: usize, static_dir: Option<PathBuf>) -> Config {
    Config { widths: widths.to_vec(), keep, static_dir, reference: None, quiet: true }
}

fn state_with(runtime: Runtime, widths: &[usize], keep: usize, static_dir: Option<PathBuf>) -> AppState {
    AppState::new(config(widths, keep, static_dir), runtime, Project::discover(Some(PathBuf::from("no/such/root"))), 8765).unwrap()
}

fn shared() -> &'static AppState {
    static STATE: OnceLock<AppState> = OnceLock::new();
    STATE.get_or_init(|| state_with(Runtime::cpu_only("no GPU in this test"), &[64, 128], 100, None))
}

fn request(method: &str, target: &str, headers: &[(&str, &str)], body: Body) -> HttpRequest {
    HttpRequest { method: method.to_string(), target: target.to_string(), headers: headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), body }
}

fn get(state: &AppState, target: &str) -> HttpResponse {
    route(state, &request("GET", target, &[], Body::Bytes(Vec::new())))
}

fn post(state: &AppState, target: &str, payload: &Value) -> HttpResponse {
    route(state, &request("POST", target, &[], Body::Bytes(serde_json::to_vec(payload).unwrap())))
}

fn json_of(r: &HttpResponse) -> Value {
    assert_eq!(r.header("Content-Type"), Some(JSON_TYPE), "{}", String::from_utf8_lossy(&r.body));
    serde_json::from_slice(&r.body).unwrap()
}

fn earth() -> Value {
    json!({"rotation_period_days": 0.99727, "radius_km": 6371, "gravity_mps2": 9.81, "scale_height_km": 8.0, "cloud_regime": "water"})
}

fn gen_request(width: usize, options: Value) -> Value {
    json!({"body": earth(), "seed": 42, "width": width, "options": options})
}

fn assert_error(r: &HttpResponse, status: u16, contains: &str) {
    assert_eq!(r.status, status, "{}", String::from_utf8_lossy(&r.body));
    let j = json_of(r);
    assert_eq!(j.as_object().unwrap().len(), 1);
    let msg = j["error"].as_str().unwrap();
    assert!(msg.contains(contains), "{msg:?} does not contain {contains:?}");
}

fn fetch(state: &AppState, href: &str) -> HttpResponse {
    let r = get(state, href);
    assert_eq!(r.status, 200, "{href}");
    r
}

// ---- defaults, derive

#[test]
fn defaults_have_the_contracts_shape() {
    let r = get(shared(), "/api/defaults");
    assert_eq!(r.status, 200);
    let d = json_of(&r);
    assert_eq!(d["backend"], "rust");
    let version = d["version"].as_str().unwrap();
    assert!(version.split('.').count() == 3 && version.split('.').all(|p| p.parse::<u32>().is_ok()), "{version}");
    assert_eq!(d["widths"], json!([64, 128]));
    assert_eq!(d["limits"], json!({"seed": [0, 2147483647], "rotation_period_days": [-400, 400], "radius_km": [500, 30000], "gravity_mps2": [0.1, 100], "scale_height_km": [0.5, 100]}));
    assert_eq!(d["max_cyclones_per_hemisphere"], 32);
    let est = d["estimate_seconds"].as_object().unwrap();
    assert_eq!(est.keys().map(String::as_str).collect::<Vec<_>>(), ["128", "64"], "serde sorts, the wire does not");
    assert!(est.values().all(|v| v.as_f64().unwrap() > 0.0));
    assert_eq!(d["compute"]["engine"], "rust");
    assert_eq!(d["compute"]["default"], "auto");
    assert!(d["presets"].as_array().unwrap().len() >= 8);
    // The ordered writer keeps the contract's key order (serde_json would sort it).
    let text = String::from_utf8(r.body.clone()).unwrap();
    assert!(text.starts_with("{\"version\":\"") && text.contains("\"backend\":\"rust\",\"presets\":["), "{}", &text[..80]);
    assert!(text.contains("\"estimate_seconds\":{\"64\":") && text.contains("\"widths\":[64,128],"));
}

#[test]
fn derive_answers_and_refuses() {
    let r = post(shared(), "/api/derive", &json!({"body": earth()}));
    assert_eq!(r.status, 200);
    let d = json_of(&r);
    assert_eq!((d["hadley_edge_deg"].as_f64().unwrap(), d["cyclones_per_hemisphere"].as_i64().unwrap()), (30.0, 5));
    assert_eq!(d["texel_km"].as_object().unwrap().len(), 2);
    assert_error(&post(shared(), "/api/derive", &json!({"body": {"radius_km": 1}})), 400, "body.rotation_period_days: missing");
    assert_error(&post(shared(), "/api/derive", &json!({"body": 1})), 400, "body: expected an object");
    assert_error(&post(shared(), "/api/derive", &json!([1])), 400, "request: expected an object");
}

#[test]
fn the_body_of_a_post_is_checked_before_anything_else_in_it() {
    let s = shared();
    let r = |body| route(s, &request("POST", "/api/derive", &[], body));
    assert_error(&r(Body::Missing), 400, "Content-Length is required");
    assert_error(&r(Body::BadLength), 400, "Content-Length is not a number");
    assert_error(&r(Body::TooLarge), 400, "at most 65536 bytes");
    assert_error(&r(Body::Bytes(b"not json".to_vec())), 400, "invalid JSON");
    assert_error(&r(Body::Bytes(Vec::new())), 400, "invalid JSON");
    assert_error(&r(Body::Bytes(b"{\"body\": NaN}".to_vec())), 400, "invalid JSON");
    assert_error(&r(Body::Bytes(vec![0xff, 0xfe])), 400, "invalid JSON");
    // A wrong route or method is answered before the body is looked at.
    assert_eq!(route(s, &request("POST", "/api/nothing", &[], Body::Missing)).status, 404);
    assert_eq!(route(s, &request("POST", "/api/defaults", &[], Body::Missing)).status, 405);
}

// ---- generate

#[test]
fn a_generation_answers_with_the_contracts_json_and_serves_its_files() {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let s = shared();
    let r = post(s, "/api/generate", &gen_request(128, json!({})));
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let out = json_of(&r);
    let (id, w, h) = (out["id"].as_str().unwrap().to_string(), 128usize, 64usize);
    assert_eq!((out["width"].as_u64().unwrap() as usize, out["height"].as_u64().unwrap() as usize), (w, h));
    assert!(out["seconds"].as_f64().unwrap() >= 0.0);
    // The keys come in the order of ui/README.md.
    let text = String::from_utf8(r.body.clone()).unwrap();
    let positions: Vec<usize> = ["\"id\":", "\"seconds\":", "\"width\":", "\"height\":", "\"compute\":", "\"meta\":", "\"images\":", "\"downloads\":", "\"metrics\":", "\"zonal\":", "\"warnings\":"]
        .iter()
        .map(|k| text.find(k).unwrap_or_else(|| panic!("{k} missing")))
        .collect();
    assert!(positions.windows(2).all(|p| p[0] < p[1]), "{positions:?}");
    assert_eq!(out["images"], json!({"coverage": format!("/api/result/{id}/coverage.png"), "type": format!("/api/result/{id}/type.png"), "high_share": format!("/api/result/{id}/high_share.png"), "wind": format!("/api/result/{id}/wind.png")}));
    assert_eq!(out["downloads"][0]["href"], format!("/api/result/{id}/master16.png"));
    assert_eq!(out["downloads"][0]["label"], "Cloud fraction, 16-bit PNG");
    assert_eq!(out["downloads"][1]["label"], "Raw R8 for the shader (gzip, rows south-up)");
    assert_eq!(out["downloads"][2]["label"], "Sidecar JSON");
    assert_eq!(out["meta"]["hadley_edge_deg"], 30.0);
    assert!(out["meta"]["surface"].is_null());
    let vortices = out["meta"]["flow"]["vortices"].as_array().unwrap();
    assert!(!vortices.is_empty());
    for v in vortices {
        let lon = v["lon_deg"].as_f64().unwrap();
        assert!((-180.0..=180.0).contains(&lon), "{lon}");
        assert_eq!(v.as_object().unwrap().len(), 7);
    }
    // The compute block: auto on a machine with no GPU engine ran on the CPU, alone, for the whole wall time.
    let c = &out["compute"];
    assert_eq!((c["requested"].as_str().unwrap(), c["used"].as_str().unwrap()), ("auto", "cpu"));
    assert!(c["note"].is_null());
    assert_eq!(c["devices"].as_array().unwrap().len(), 1);
    assert_eq!(c["devices"][0]["share"], 1.0);
    let device_seconds = c["devices"][0]["seconds"].as_f64().unwrap();
    assert_eq!(device_seconds, out["seconds"].as_f64().unwrap(), "the CPU device carries the whole wall time");
    // The metrics table and the zonal profile.
    assert_eq!(out["metrics"]["earth_label"], "Earth (Blue Marble)");
    assert!(out["metrics"]["note"].as_str().unwrap().contains("interpolated"));
    let rows = out["metrics"]["table"].as_array().unwrap();
    assert_eq!(rows.len(), 10);
    for row in rows {
        assert_eq!(row.as_object().unwrap().len(), 6);
        assert!(row["generated"].as_f64().unwrap().is_finite() && row["earth"].as_f64().unwrap().is_finite());
    }
    assert_eq!(rows[1]["earth"], 0.0);
    assert_eq!(out["zonal"]["lat_deg"].as_array().unwrap().len(), 36);
    assert_eq!(out["zonal"]["lat_deg"][0], -87.5);
    for k in ["generated", "earth", "target"] {
        let z = out["zonal"][k].as_array().unwrap();
        assert_eq!(z.len(), 36, "{k}");
        assert!(z.iter().all(|v| (0.0..=1.0).contains(&v.as_f64().unwrap())), "{k}");
    }
    // The files: sizes, bit depths and the south-up R8.
    let cov = fetch(s, &format!("/api/result/{id}/coverage.png"));
    assert_eq!((cov.header("Content-Type"), cov.header("Content-Disposition")), (Some("image/png"), None));
    let cov8 = decode_gray(&cov.body).unwrap();
    assert_eq!((cov8.width, cov8.height, cov8.max), (w, h, 255.0));
    let master = decode_gray(&fetch(s, &format!("/api/result/{id}/master16.png")).body).unwrap();
    assert_eq!((master.width, master.height, master.max), (w, h, 65535.0));
    for (m, c) in master.data.iter().zip(&cov8.data) {
        assert!((f64::from(*m) / 257.0 - f64::from(*c)).abs() <= 1.0);
    }
    assert!(master.data.iter().any(|&v| v > 255), "16 bits of range");
    for key in ["type", "high_share", "wind"] {
        let img = decode_gray(&fetch(s, &format!("/api/result/{id}/{key}.png")).body).unwrap();
        assert_eq!((img.width, img.height), (w, h), "{key} (the wind grid is min(1024, width) wide)");
    }
    let r8 = fetch(s, &format!("/api/result/{id}/coverage.r8.gz"));
    assert_eq!(r8.header("Content-Type"), Some("application/gzip"));
    let disp = r8.header("Content-Disposition").unwrap().to_string();
    assert!(disp.starts_with("attachment; filename=\"cloudgen_s42_128x64_") && disp.contains(&id) && disp.ends_with(".r8.gz\""), "{disp}");
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(&r8.body[..]).read_to_end(&mut raw).unwrap();
    assert_eq!(raw.len(), w * h);
    let north_up: Vec<u8> = cov8.data.iter().map(|&v| v as u8).collect();
    let mut flipped = Vec::new();
    for row in (0..h).rev() {
        flipped.extend_from_slice(&north_up[row * w..(row + 1) * w]);
    }
    assert_eq!(raw, flipped, "rows are south-up");
    // The sidecar.
    let meta_resp = fetch(s, &format!("/api/result/{id}/meta.json"));
    assert_eq!(meta_resp.header("Content-Type"), Some(JSON_TYPE));
    assert!(meta_resp.header("Content-Disposition").unwrap().ends_with(".json\""));
    let meta: Value = serde_json::from_slice(&meta_resp.body).unwrap();
    assert_eq!((meta["seed"].as_u64(), meta["width"].as_u64(), meta["height"].as_u64()), (Some(42), Some(128), Some(64)));
    assert_eq!(meta["body"]["radius_km"], 6371.0);
    assert_eq!(meta["options"], json!({"match_earth": false, "isotropic": false, "experiment": false, "surface": "none"}));
    assert_eq!(meta["compute"], json!({"requested": "auto", "used": "cpu", "devices": [{"kind": "cpu", "name": s.runtime.cpu.name, "share": 1.0}], "note": null}));
    assert_eq!(meta["products"]["files"].as_object().unwrap().keys().collect::<Vec<_>>(), ["high_share", "type", "wind"]);
    assert!(meta.get("matched_to").is_none());
    // The same request makes the same map (the engine is deterministic per seed) under a new id.
    let again = json_of(&post(s, "/api/generate", &gen_request(128, json!({"isotropic": false}))));
    assert_ne!(again["id"], out["id"]);
    let again_id = again["id"].as_str().unwrap();
    assert_eq!(fetch(s, &format!("/api/result/{again_id}/coverage.png")).body, cov.body);
}

#[test]
fn the_options_change_the_map_and_are_recorded() {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let s = shared();
    let run = |options: Value| {
        let r = post(s, "/api/generate", &gen_request(64, options));
        assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
        json_of(&r)
    };
    let sidecar = |out: &Value| -> Value {
        let id = out["id"].as_str().unwrap();
        serde_json::from_slice(&fetch(s, &format!("/api/result/{id}/meta.json")).body).unwrap()
    };
    let image = |out: &Value| fetch(s, out["images"]["coverage"].as_str().unwrap()).body;
    let plain = run(json!({}));
    let iso = run(json!({"isotropic": true}));
    assert!(iso["meta"]["flow"]["vortices"].as_array().unwrap().is_empty());
    assert!(iso["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().starts_with("Isotropic baseline")));
    assert_eq!(sidecar(&iso)["options"]["isotropic"], true);
    assert_ne!(image(&iso), image(&plain));
    let matched = run(json!({"match_earth": true}));
    assert_eq!(sidecar(&matched)["matched_to"], "data/earth_cloud_fraction_bluemarble_2048.png");
    assert_ne!(image(&matched), image(&plain));
    let surf = run(json!({"surface": "synthetic"}));
    let land = surf["meta"]["surface"]["land_fraction"].as_f64().unwrap();
    assert!((0.05..0.6).contains(&land), "{land}");
    assert!(surf["meta"]["surface"]["itcz_shift_deg"].as_f64().unwrap().abs() < 10.0);
    assert_eq!(sidecar(&surf)["options"]["surface"], "synthetic");
    // A non-water body needs the experiment option.
    let mars = json!({"rotation_period_days": 1.02596, "radius_km": 3389.5, "gravity_mps2": 3.721, "scale_height_km": 11.1, "cloud_regime": "thin water ice"});
    let refused = post(s, "/api/generate", &json!({"body": mars, "seed": 1, "width": 64}));
    assert_error(&refused, 400, "experiment");
    let ok = post(s, "/api/generate", &json!({"body": mars, "seed": 1, "width": 64, "options": {"experiment": true}}));
    assert_eq!(ok.status, 200);
    assert!(json_of(&ok)["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("experiment")));
    // No storm track: no vortices. Retrograde: the northern lows turn the other way.
    let slow = json!({"body": {"rotation_period_days": 4.0, "radius_km": 6371, "gravity_mps2": 9.81, "scale_height_km": 8.0}, "seed": 1, "width": 64});
    assert!(json_of(&post(s, "/api/generate", &slow))["meta"]["flow"]["vortices"].as_array().unwrap().is_empty());
    let retro = json!({"body": {"rotation_period_days": -0.99727, "radius_km": 6371, "gravity_mps2": 9.81, "scale_height_km": 8.0}, "seed": 1, "width": 64});
    let out = json_of(&post(s, "/api/generate", &retro));
    let north: Vec<&Value> = out["meta"]["flow"]["vortices"].as_array().unwrap().iter().filter(|v| v["kind"] == "cyclone" && v["lat_deg"].as_f64().unwrap() > 0.0).collect();
    assert!(!north.is_empty() && north.iter().all(|v| v["omega"].as_f64().unwrap() < 0.0));
}

#[test]
fn a_bad_generate_request_is_a_400_before_anything_runs() {
    let s = shared();
    assert_error(&post(s, "/api/generate", &gen_request(300, json!({}))), 400, "width: must be one of [64, 128]");
    assert_error(&post(s, "/api/generate", &gen_request(64, json!({"turbo": true}))), 400, "options: unknown field 'turbo'");
    assert_error(&post(s, "/api/generate", &json!({"seed": 1, "width": 64})), 400, "body: missing");
    let fast = json!({"body": {"rotation_period_days": 0.02, "radius_km": 6371, "gravity_mps2": 9.81, "scale_height_km": 8.0}, "seed": 1, "width": 64});
    assert_error(&post(s, "/api/generate", &fast), 400, "rotation too fast for this tool");
}

#[test]
fn compute_modes_are_checked_reported_and_never_fall_back() {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let s = shared();
    for (mode, reason) in [("gpu", "is not available: no GPU in this test."), ("hybrid", "there is no GPU to share the work with.")] {
        let r = post(s, "/api/generate", &gen_request(64, json!({"compute": mode})));
        assert_error(&r, 400, reason);
        let msg = json_of(&r)["error"].as_str().unwrap().to_string();
        assert!(msg.starts_with("options.compute:") && msg.contains("'auto'") && msg.contains("'cpu'"), "{msg}");
    }
    for bad in [json!("turbo"), json!("GPU"), json!(""), Value::Null, json!(5), json!(true), json!(["cpu"])] {
        let r = post(s, "/api/generate", &gen_request(64, json!({ "compute": bad })));
        assert_error(&r, 400, "options.compute: expected one of 'auto', 'cpu', 'gpu', 'hybrid'");
    }
    let out = json_of(&post(s, "/api/generate", &gen_request(64, json!({"compute": "cpu"}))));
    assert_eq!((out["compute"]["requested"].as_str(), out["compute"]["used"].as_str()), (Some("cpu"), Some("cpu")));
    let id = out["id"].as_str().unwrap();
    let meta: Value = serde_json::from_slice(&fetch(s, &format!("/api/result/{id}/meta.json")).body).unwrap();
    assert_eq!(meta["compute"]["requested"], "cpu");
    assert!(meta["compute"]["devices"][0].get("seconds").is_none(), "no wall time in a file");
    assert_eq!(meta["options"].as_object().unwrap().len(), 4, "compute is not an option of the map");
}

// ---- an engine that is not the CPU's: the plumbing a GPU engine plugs into

/// A test double: it makes the map with the CPU engine and reports a GPU and a CPU device, to check what the server
/// does with an engine that is not the CPU's. It is not a GPU path.
struct Double {
    mode: &'static str,
}

impl Engine for Double {
    fn generate(&self, req: &GenerateRequest, progress: Progress<'_>) -> Result<Generated, EngineError> {
        match self.mode {
            "panic" => panic!("the device was lost"),
            "refuse" => Err(EngineError::Refused("this map does not fit the device".to_string())),
            "fail" => Err(EngineError::Failed("the kernel failed".to_string())),
            "gpu_only" => {
                let mut g = CpuEngine { name: None, workers: None }.generate(req, progress)?;
                g.compute = ComputeUse { used: Used::Gpu, devices: vec![DeviceUse { kind: DeviceKind::Gpu, name: Some("Test GPU".to_string()), seconds: 0.25, share: 1.0 }], note: Some("a band was retried".to_string()) };
                Ok(g)
            }
            _ => {
                let mut g = CpuEngine { name: None, workers: None }.generate(req, progress)?;
                g.compute = ComputeUse {
                    used: Used::Gpu,
                    devices: vec![
                        DeviceUse { kind: DeviceKind::Cpu, name: Some("Test CPU".to_string()), seconds: 0.01, share: 0.25 },
                        DeviceUse { kind: DeviceKind::Gpu, name: Some("Test GPU".to_string()), seconds: 0.25, share: 0.75 },
                    ],
                    note: None,
                };
                Ok(g)
            }
        }
    }

    fn min_auto_width(&self) -> usize {
        128
    }
}

fn state_with_double(mode: &'static str) -> AppState {
    let info = GpuInfo { name: "Test GPU".into(), backend: "Vulkan".into(), float64: false, df64: Some("fma+barrier".into()) };
    let runtime = Runtime { cpu: CpuEngine { name: Some("Test CPU".into()), workers: None }, threads: 4, gpu: GpuState::Found { info, engine: Ok(Arc::new(Double { mode })) } };
    state_with(runtime, &[64, 128], DEFAULT_KEEP, None)
}

#[test]
fn a_plugged_in_gpu_engine_is_offered_chosen_by_auto_from_its_width_and_reported() {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let s = state_with_double("ok");
    let caps = json_of(&get(&s, "/api/defaults"))["compute"].clone();
    let gpu = &caps["modes"][2];
    assert_eq!((gpu["available"].clone(), gpu["reason"].clone(), gpu["detail"].clone()), (json!(true), Value::Null, json!("Test GPU")));
    assert_eq!(caps["modes"][3]["available"], cloud_map_generator_service::engine::cpu_threads() > 1, "hybrid needs a GPU and more than one CPU thread");
    assert_eq!(caps["devices"]["gpu"], json!({"name": "Test GPU", "backend": "Vulkan", "float64": false, "df64": "fma+barrier"}));
    // Explicit gpu.
    let out = json_of(&post(&s, "/api/generate", &gen_request(64, json!({"compute": "gpu"}))));
    let c = &out["compute"];
    assert_eq!((c["requested"].as_str(), c["used"].as_str()), (Some("gpu"), Some("gpu")));
    assert_eq!(c["devices"][0], json!({"kind": "gpu", "name": "Test GPU", "seconds": 0.25, "share": 0.75}));
    let cpu_seconds = c["devices"][1]["seconds"].as_f64().unwrap();
    assert!(cpu_seconds >= 0.01, "the CPU device carries the server's own work: {cpu_seconds}");
    assert_eq!(c["devices"][1]["share"], 0.25);
    // Auto: the GPU from its crossover width, the CPU below it with a sentence on why.
    let wide = json_of(&post(&s, "/api/generate", &gen_request(128, json!({}))));
    assert_eq!((wide["compute"]["requested"].as_str(), wide["compute"]["used"].as_str()), (Some("auto"), Some("gpu")));
    assert!(wide["compute"]["note"].as_str().unwrap().starts_with("auto chose the GPU: a 128-wide map"));
    let narrow = json_of(&post(&s, "/api/generate", &gen_request(64, json!({}))));
    assert_eq!(narrow["compute"]["used"], "cpu");
    assert!(narrow["compute"]["note"].as_str().unwrap().contains("auto chose the CPU: a 64-wide map is below the width where the GPU pays"));
    assert_error(&post(&s, "/api/generate", &gen_request(64, json!({"compute": "hybrid"}))), 400, "cannot run bands");
}

#[test]
fn an_engine_that_refuses_fails_or_panics_is_a_400_or_a_500_and_frees_the_lock() {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for (mode, status, text) in [("refuse", 400, "does not fit the device"), ("fail", 500, "the kernel failed"), ("panic", 500, "the device was lost")] {
        let s = state_with_double(mode);
        assert_error(&post(&s, "/api/generate", &gen_request(64, json!({"compute": "gpu"}))), status, text);
        // The lock is free again (a poisoned one is recovered): the CPU still works, and nothing was stored before.
        assert_eq!(s.results.len(), 0);
        let r = post(&s, "/api/generate", &gen_request(64, json!({"compute": "cpu"})));
        assert_eq!(r.status, 200, "after {mode}: {}", String::from_utf8_lossy(&r.body));
        assert_eq!(s.results.len(), 1);
    }
}

// ---- busy, results

#[test]
fn a_generation_while_one_runs_is_a_429_and_the_rest_of_the_api_still_answers() {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let s = shared();
    let held = s.gen_lock.lock().unwrap_or_else(|e| e.into_inner());
    let r = post(s, "/api/generate", &gen_request(64, json!({})));
    assert_error(&r, 429, "a generation is already running");
    assert_eq!(json_of(&r), json!({"error": "a generation is already running"}));
    assert_eq!(get(s, "/api/defaults").status, 200);
    assert_eq!(post(s, "/api/derive", &json!({"body": earth()})).status, 200);
    // A bad request is a 400 first, busy or not.
    assert_eq!(post(s, "/api/generate", &json!({})).status, 400);
    drop(held);
}

#[test]
fn results_are_evicted_oldest_first_and_unknown_ids_are_404() {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let s = state_with(Runtime::cpu_only("none"), &[64], 2, None);
    let ids: Vec<String> = (1..=3)
        .map(|seed| {
            let r = post(&s, "/api/generate", &json!({"body": earth(), "seed": seed, "width": 64, "options": {"isotropic": true}}));
            assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
            json_of(&r)["id"].as_str().unwrap().to_string()
        })
        .collect();
    assert_eq!(s.results.len(), 2);
    for name in ["coverage.png", "master16.png"] {
        assert_error(&get(&s, &format!("/api/result/{}/{name}", ids[0])), 404, "no such result (results are kept in memory, the last 2): generate again");
    }
    assert_eq!(get(&s, &format!("/api/result/{}/coverage.png", ids[1])).status, 200);
    assert_eq!(get(&s, &format!("/api/result/{}/coverage.png", ids[2])).status, 200);
    for path in [
        "/api/result/000000000000/coverage.png".to_string(),
        "/api/result/abc/coverage.png".to_string(),
        format!("/api/result/{}/nothing.png", ids[2]),
        format!("/api/result/{}/../defaults", ids[2]),
        format!("/api/result/{}/", ids[2]),
    ] {
        assert_eq!(get(&s, &path).status, 404, "{path}");
    }
}

#[test]
fn the_reference_map_is_served() {
    let r = get(shared(), "/api/reference/coverage.png");
    assert_eq!((r.status, r.header("Content-Type")), (200, Some("image/png")));
    let img = decode_gray(&r.body).unwrap();
    assert_eq!((img.width, img.height), (2048, 1024));
}

// ---- static files

#[test]
fn the_embedded_page_is_served_as_the_index_and_under_static() {
    let s = shared();
    let r = get(s, "/");
    assert_eq!((r.status, r.header("Content-Type"), r.header("X-Content-Type-Options"), r.header("Cache-Control")), (200, Some("text/html; charset=utf-8"), Some("nosniff"), Some("no-store")));
    assert_eq!(get(s, "/static/index.html").body, r.body);
    let js = get(s, "/static/app.js");
    assert_eq!((js.status, js.header("Content-Type")), (200, Some("text/javascript; charset=utf-8")));
    assert_error(&get(s, "/static/nope.js"), 404, "not found: /static/nope.js");
    assert_eq!(get(s, "/static/").status, 404);
    assert_eq!(get(s, "/static/sub").status, 404);
    assert_eq!(get(s, "/?x=1").status, 200, "the query is not part of the path");
    let fav = get(s, "/favicon.ico");
    assert_eq!((fav.status, fav.body.len()), (204, 0));
}

#[test]
fn the_openapi_document_and_its_swagger_page_are_served_and_match_the_routes() {
    let s = shared();
    let spec = get(s, "/openapi.json");
    assert_eq!((spec.status, spec.header("Content-Type")), (200, Some("application/json; charset=utf-8")));
    assert_eq!(get(s, "/static/openapi.json").body, spec.body);
    let docs = get(s, "/docs");
    assert_eq!((docs.status, docs.header("Content-Type")), (200, Some("text/html; charset=utf-8")));
    assert!(String::from_utf8_lossy(&docs.body).contains("SwaggerUIBundle"));
    assert_eq!(route(s, &request("POST", "/docs", &[], Body::Bytes(b"{}".to_vec()))).status, 405);

    let doc: Value = serde_json::from_slice(&spec.body).unwrap();
    assert_eq!(doc["openapi"], "3.1.0");
    // release-please bumps both (release-please-config.json): the document never describes another version.
    assert_eq!(doc["info"]["version"], env!("CARGO_PKG_VERSION"));
    let mut documented: Vec<(String, String)> = Vec::new();
    for (path, item) in doc["paths"].as_object().unwrap() {
        for method in item.as_object().unwrap().keys() {
            documented.push((method.to_uppercase(), path.clone()));
        }
    }
    documented.sort();
    let mut expected: Vec<(String, String)> = [
        ("GET", "/api/defaults"),
        ("GET", "/api/reference/coverage.png"),
        ("GET", "/api/result/{id}/{name}"),
        ("POST", "/api/derive"),
        ("POST", "/api/generate"),
    ]
    .iter()
    .map(|(m, p)| (m.to_string(), p.to_string()))
    .collect();
    expected.sort();
    assert_eq!(documented, expected, "the document lists exactly the API's routes");
    // Each documented route answers (not a 404 of an unknown route) with the method the document gives it.
    assert_eq!(get(s, "/api/defaults").status, 200);
    assert_eq!(get(s, "/api/reference/coverage.png").status, 200);
    assert_error(&get(s, "/api/result/000000000000/coverage.png"), 404, "no such result");
    assert_eq!(post(s, "/api/derive", &json!({"body": earth()})).status, 200);
    assert_eq!(route(s, &request("GET", "/api/generate", &[], Body::Bytes(Vec::new()))).status, 405);
    // The result file names the document allows are the ones the server keeps.
    let names: Vec<&str> = doc["paths"]["/api/result/{id}/{name}"]["get"]["parameters"][1]["schema"]["enum"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(names, crate::serve::store::RESULT_FILES.iter().map(|(n, _, _)| *n).collect::<Vec<_>>());
}

fn static_dir(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("cloudgen-serve-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("static").join("sub")).unwrap();
    std::fs::write(root.join("static").join("index.html"), "<!doctype html><title>t</title>").unwrap();
    std::fs::write(root.join("static").join("app.js"), "export const x = 1;").unwrap();
    std::fs::write(root.join("static").join("style.css"), "body{}").unwrap();
    std::fs::write(root.join("static").join("sub").join("icon.svg"), "<svg xmlns=\"http://www.w3.org/2000/svg\"/>").unwrap();
    std::fs::write(root.join("secret.txt"), "TOP SECRET").unwrap();
    root
}

#[test]
fn a_static_dir_replaces_the_embedded_page_and_cannot_be_left() {
    let root = static_dir("dir");
    let s = state_with(Runtime::cpu_only("none"), &[64], 8, Some(root.join("static")));
    let r = get(&s, "/");
    assert_eq!(r.status, 200);
    assert!(String::from_utf8_lossy(&r.body).contains("<title>t</title>"));
    for (path, ctype) in [("/static/app.js", "text/javascript; charset=utf-8"), ("/static/style.css", "text/css; charset=utf-8"), ("/static/sub/icon.svg", "image/svg+xml"), ("/static/index.html", "text/html; charset=utf-8")] {
        let r = get(&s, path);
        assert_eq!((r.status, r.header("Content-Type")), (200, Some(ctype)), "{path}");
        assert_eq!(r.header("X-Content-Type-Options"), Some("nosniff"));
    }
    for path in ["/static/nope.js", "/static/sub", "/static/"] {
        assert_eq!(get(&s, path).status, 404, "{path}");
    }
    for path in [
        "/static/../secret.txt",
        "/static/%2e%2e/secret.txt",
        "/static/%2E%2E%2Fsecret.txt",
        "/static/sub/../../secret.txt",
        "/static/..%5csecret.txt",
        "/static/..\\secret.txt",
        "/static//secret.txt",
        "/static/C:/Windows/win.ini",
        "/static/%00/secret.txt",
        "/static/sub/%2e%2e/%2e%2e/secret.txt",
        "/static/index.html::$DATA",
    ] {
        let r = get(&s, path);
        assert!(r.status == 400 || r.status == 404, "{path}: {}", r.status);
        assert!(!String::from_utf8_lossy(&r.body).contains("TOP SECRET"), "{path}");
    }
    // A static folder with no index is a JSON 404, and the favicon is a 204.
    let empty = std::env::temp_dir().join(format!("cloudgen-serve-empty-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    let s = state_with(Runtime::cpu_only("none"), &[64], 8, Some(empty));
    assert_error(&get(&s, "/"), 404, "ui/static/index.html not found");
    assert_eq!(get(&s, "/favicon.ico").status, 204);
}

#[test]
fn a_symlink_out_of_the_static_folder_is_not_served() {
    let root = static_dir("link");
    let link = root.join("static").join("leak.txt");
    #[cfg(unix)]
    let made = std::os::unix::fs::symlink(root.join("secret.txt"), &link).is_ok();
    #[cfg(windows)]
    let made = std::os::windows::fs::symlink_file(root.join("secret.txt"), &link).is_ok();
    #[cfg(not(any(unix, windows)))]
    let made = false;
    if !made {
        eprintln!("cannot create a symlink here: test skipped");
        return;
    }
    let s = state_with(Runtime::cpu_only("none"), &[64], 8, Some(root.join("static")));
    let r = get(&s, "/static/leak.txt");
    assert_eq!(r.status, 404);
    assert!(!String::from_utf8_lossy(&r.body).contains("TOP SECRET"));
}

// ---- routing and safety

#[test]
fn unknown_routes_wrong_methods_and_unsupported_methods() {
    let s = shared();
    assert_error(&get(s, "/api/nothing"), 404, "not found: /api/nothing");
    assert_eq!(get(s, "/elsewhere").status, 404);
    let with = |method: &str, path: &str| route(s, &request(method, path, &[], Body::Bytes(b"{}".to_vec())));
    assert_error(&with("POST", "/api/defaults"), 405, "POST not allowed on /api/defaults");
    assert_eq!(with("GET", "/api/generate").status, 405);
    assert_eq!(with("GET", "/api/derive").status, 405);
    assert_eq!(with("POST", "/").status, 405);
    assert_eq!(with("POST", "/static/app.js").status, 405);
    assert_error(&with("DELETE", "/api/defaults"), 501, "Unsupported method ('DELETE')");
    assert_eq!(with("OPTIONS", "/api/defaults").status, 501);
    assert_eq!(with("PUT", "/").status, 501);
    // HEAD is GET (the transport drops the body).
    assert_eq!(with("HEAD", "/api/defaults").status, 200);
    assert_eq!(with("HEAD", "/api/nothing").status, 404);
}

#[test]
fn a_foreign_host_or_origin_is_a_403_and_the_pages_own_is_fine() {
    let s = shared();
    let call = |method: &str, path: &str, headers: &[(&str, &str)]| route(s, &request(method, path, headers, Body::Bytes(serde_json::to_vec(&json!({"body": earth()})).unwrap())));
    assert_error(&call("POST", "/api/derive", &[("origin", "http://evil.example")]), 403, "foreign Origin");
    assert_eq!(call("POST", "/api/derive", &[("origin", "null")]).status, 403);
    assert_eq!(call("POST", "/api/derive", &[("origin", "https://127.0.0.1:8765")]).status, 403);
    assert_eq!(call("POST", "/api/derive", &[("origin", "http://user@127.0.0.1:8765")]).status, 403);
    assert_error(&call("GET", "/api/defaults", &[("host", "evil.example")]), 403, "unexpected Host header");
    assert_eq!(call("GET", "/api/defaults", &[("host", "evil.example:8765")]).status, 403);
    assert_eq!(call("GET", "/api/defaults", &[("host", "127.0.0.1:9999")]).status, 403, "another port is another origin");
    for origin in ["http://127.0.0.1:8765", "http://localhost:8765", "http://[::1]:8765", "HTTP://LOCALHOST:8765", "http://localhost:8765/"] {
        assert_eq!(call("POST", "/api/derive", &[("origin", origin)]).status, 200, "{origin}");
    }
    for host in ["localhost:8765", "127.0.0.1:8765", "[::1]:8765", "LocalHost:8765"] {
        assert_eq!(call("GET", "/api/defaults", &[("host", host)]).status, 200, "{host}");
    }
    // A request with neither header (curl) passes.
    assert_eq!(call("GET", "/api/defaults", &[]).status, 200);
}

#[test]
fn no_cors_headers_and_the_security_headers_on_every_answer() {
    let s = shared();
    for r in [get(s, "/api/defaults"), get(s, "/api/nothing"), get(s, "/"), get(s, "/api/reference/coverage.png")] {
        assert!(r.headers.iter().all(|(k, _)| !k.to_ascii_lowercase().starts_with("access-control-")));
        assert_eq!(r.header("Cache-Control"), Some("no-store"));
        assert_eq!(r.header("X-Content-Type-Options"), Some("nosniff"));
    }
    assert_eq!(get(s, "/api/defaults").header("Content-Type"), Some("application/json; charset=utf-8"));
}

#[test]
fn targets_are_split_at_the_query_and_origins_at_the_slash() {
    assert_eq!(path_of("/api/defaults?x=1#f"), "/api/defaults");
    assert_eq!(path_of("/static/a.js"), "/static/a.js");
    assert_eq!(origin_netloc("http://localhost:1/x").as_deref(), Some("localhost:1"));
    assert_eq!(origin_netloc("https://localhost:1"), None);
    assert_eq!(origin_netloc("null"), None);
    assert_eq!(result_route("/api/result/0123456789ab/meta.json"), Some(("0123456789ab", "meta.json")));
    assert_eq!(result_route("/api/result/0123456789AB/meta.json"), None);
    assert_eq!(result_route("/api/result/0123456789ab/a/b"), None);
    assert_eq!(content_type("x.JS"), "text/javascript; charset=utf-8");
    assert_eq!(content_type("x.unknown"), "application/octet-stream");
}

#[test]
fn the_reply_lists_the_cpu_too_orders_devices_by_share_and_keeps_the_engines_note() {
    let _turn = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // An engine that reports only the GPU: the CPU is listed with share 0.0 and the seconds the server spent on its own work.
    let s = state_with_double("gpu_only");
    let out = json_of(&post(&s, "/api/generate", &gen_request(128, json!({"compute": "gpu"}))));
    let devices = out["compute"]["devices"].as_array().unwrap();
    assert_eq!(devices.iter().map(|d| d["kind"].as_str().unwrap()).collect::<Vec<_>>(), ["gpu", "cpu"], "larger share first");
    assert_eq!((devices[0]["share"].as_f64(), devices[1]["share"].as_f64()), (Some(1.0), Some(0.0)));
    assert_eq!(devices[1]["name"], "Test CPU");
    assert!(devices[1]["seconds"].as_f64().unwrap() > 0.0, "the CPU spent time on the plans, the metrics and the encodings");
    assert_eq!(out["compute"]["note"], "a band was retried", "an explicit mode has no choice note, the engine's own is kept");
    let id = out["id"].as_str().unwrap();
    let meta: Value = serde_json::from_slice(&fetch(&s, &format!("/api/result/{id}/meta.json")).body).unwrap();
    assert_eq!(meta["compute"]["devices"], json!([{"kind": "gpu", "name": "Test GPU", "share": 1.0}, {"kind": "cpu", "name": "Test CPU", "share": 0.0}]));
    // Auto chose it: the note of the choice comes first, then the engine's.
    let auto = json_of(&post(&s, "/api/generate", &gen_request(128, json!({}))));
    assert!(auto["compute"]["note"].as_str().unwrap().starts_with("auto chose the GPU: ") && auto["compute"]["note"].as_str().unwrap().ends_with(" a band was retried"));
    // The engine listed the CPU first with the smaller share: the reply puts the larger share first.
    let s = state_with_double("ok");
    let out = json_of(&post(&s, "/api/generate", &gen_request(64, json!({"compute": "gpu"}))));
    assert_eq!(out["compute"]["devices"].as_array().unwrap().iter().map(|d| d["kind"].as_str().unwrap()).collect::<Vec<_>>(), ["gpu", "cpu"]);
}

#[test]
fn the_date_header_follows_rfc_9110() {
    // The example of the RFC, a leap day, and the epoch.
    assert_eq!(http_date_of(784_111_777), "Sun, 06 Nov 1994 08:49:37 GMT");
    assert_eq!(http_date_of(951_782_400), "Tue, 29 Feb 2000 00:00:00 GMT");
    assert_eq!(http_date_of(0), "Thu, 01 Jan 1970 00:00:00 GMT");
    assert_eq!(http_date_of(1_794_398_400), "Wed, 11 Nov 2026 12:00:00 GMT");
}

#[test]
fn every_form_of_a_request_target_names_its_path() {
    for (target, path) in [
        ("/api/defaults", "/api/defaults"),
        ("/api/defaults?x=1", "/api/defaults"),
        ("//api/defaults", "/api/defaults"),
        ("///api/defaults?x=1", "/api/defaults"),
        ("http://127.0.0.1:8765/api/defaults", "/api/defaults"),
        ("HTTP://127.0.0.1:8765/api/defaults#top", "/api/defaults"),
        ("http://127.0.0.1:8765", "/"),
        ("http://127.0.0.1:8765?x=1", "/"),
        ("/", "/"),
        ("", "/"),
        ("/static/a://b", "/static/a://b"),
    ] {
        assert_eq!(path_of(target), path, "{target:?}");
    }
    assert_eq!(origin_netloc("Http://127.0.0.1:8765").as_deref(), Some("127.0.0.1:8765"));
    assert_eq!(origin_netloc("http://LOCALHOST:8765/x").as_deref(), Some("localhost:8765"));
    assert_eq!(origin_netloc("https://127.0.0.1:8765"), None);
    assert_eq!(origin_netloc("null"), None);
}
