//! The MCP server end to end, through the official rmcp client over an in-memory pipe, with the mock engine at
//! small widths: tools, resources, prompts, the sandbox seen from the outside, the job lifecycle, the compute
//! modes. Ports of `tests/test_mcp_server.py` of the Python server.

mod common;

use std::path::Path;
use std::time::Duration;

use cloud_map_generator_mcp::backend::{Backend, GenerateOptions};
use cloud_map_generator_mcp::mock::{Gate, MockBackend, MockFailure};
use cloud_map_generator_mcp::sandbox::MAX_READ_BYTES;
use common::*;
use rmcp::model::{GetPromptRequestParams, ProgressNotificationParam, Role};
use serde_json::{json, Value};

const TOOLS: [&str; 10] = [
    "list_presets",
    "derive_physics",
    "generate_map",
    "start_generation",
    "get_generation",
    "cancel_generation",
    "view_map",
    "measure_map",
    "bake_bodies",
    "explain_model",
];
const READ_ONLY: [&str; 6] = ["list_presets", "derive_physics", "get_generation", "view_map", "measure_map", "explain_model"];
const RESOURCES: [&str; 5] = [
    "cloud-map-generator://docs/model",
    "cloud-map-generator://docs/bodies-format",
    "cloud-map-generator://presets",
    "cloud-map-generator://capabilities",
    "cloud-map-generator://reference/earth-coverage.png",
];
const RESULT_FILES: [&str; 5] = ["coverage.png", "type.png", "high_share.png", "wind.png", "meta.json"];

async fn harness() -> (Harness, Dirs) {
    let dirs = Dirs::new();
    let h = connect(dirs.config(), MockBackend::new()).await;
    (h, dirs)
}

fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap_or_else(|| panic!("{k} is not a string in {v}"))
}

fn lats(rows: &Value) -> Vec<f64> {
    rows.as_array().unwrap().iter().map(|r| r["lat_deg"].as_f64().unwrap()).collect()
}

fn fifteen() -> Vec<f64> {
    (-6..=6).rev().map(|i| f64::from(i) * 15.0).collect()
}

// ------------------------------------------------------------------------------------------------ catalog

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tools_have_descriptions_schemas_and_honest_hints() {
    let (h, _d) = harness().await;
    let mut names: Vec<String> = h.tools.iter().map(|t| t.name.to_string()).collect();
    names.sort();
    let mut want: Vec<String> = TOOLS.iter().map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(names, want);
    for t in &h.tools {
        let name = t.name.as_ref();
        assert!(t.description.as_deref().unwrap_or("").len() > 150 && t.title.is_some(), "{name}");
        assert_eq!(t.input_schema["type"], "object", "{name}");
        assert_eq!(t.output_schema.as_ref().unwrap()["type"], "object", "{name}");
        let a = t.annotations.as_ref().unwrap_or_else(|| panic!("{name}: annotations"));
        assert_eq!(a.open_world_hint, Some(false), "{name}");
        assert!(a.idempotent_hint.is_some(), "{name}");
        assert_eq!(a.read_only_hint, Some(READ_ONLY.contains(&name)), "{name}");
        if READ_ONLY.contains(&name) {
            assert!(a.destructive_hint != Some(true), "{name}");
        } else {
            assert_eq!(a.destructive_hint, Some(name == "bake_bodies"), "{name}: only a rerun of a bake overwrites");
        }
    }
    assert_eq!(h.tool("view_map").input_schema["properties"]["view"]["enum"], json!(["map", "globe"]));
    let props = h.tool("generate_map").input_schema["properties"].as_object().unwrap().clone();
    for k in ["preset", "overrides", "seed", "width", "match_earth", "isotropic", "experiment", "surface", "save", "compute"] {
        assert!(props.contains_key(k), "{k}");
    }
    assert!(!props.contains_key("ctx") && props["seed"]["maximum"] == 2147483647);
    let ov = &h.tool("derive_physics").input_schema["$defs"]["PhysicsOverrides"];
    assert_eq!(ov["additionalProperties"], false);
    let rot = ov["properties"]["rotation_period_days"]["description"].as_str().unwrap();
    assert!(rot.to_lowercase().contains("negative") && rot.contains("Allowed range -400 to 400"), "{rot}");
    for name in ["start_generation", "bake_bodies", "generate_map"] {
        let d = h.tool(name).description.as_deref().unwrap();
        assert!(d.contains("Next") || d.to_lowercase().contains("poll"), "{name}: what to call next is in the description");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_readme_documents_what_the_server_exposes_and_stays_generic() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let readme = std::fs::read_to_string(root.join("README.md")).expect("README.md");
    for name in TOOLS.iter().chain(["design_planet_clouds", "bake_for_engine"].iter()).chain(RESOURCES.iter()) {
        assert!(readme.contains(name), "{name}");
    }
    for file in RESULT_FILES {
        assert!(readme.contains(file), "{file}");
    }
    for flag in ["--out-dir", "--max-width", "--read-dir", "--http", "--port", "--version", "PLANET_CLOUDGEN_OUT"] {
        assert!(readme.contains(flag), "{flag}");
    }
    for word in ["compute", "auto", "hybrid", "GPU-capable", "silent fallback", "compute.used", "mock"] {
        assert!(readme.contains(word), "{word}");
    }
    for client in [
        "Claude Code", "Claude Desktop", "Codex", "Cursor", "VS Code", "Zed", "claude_desktop_config.json", ".mcp.json",
        "config.toml", ".cursor/mcp.json", ".vscode/mcp.json", "cargo install --path rust/cloud-map-generator-mcp", "cloud-map-generator-mcp",
    ] {
        assert!(readme.contains(client), "{client}");
    }
    assert!(!readme.contains("uvx") && !readme.contains("pip install"), "no Python at run time");
    // the server is for any developer: no project of ours leaks into it
    let mut texts = vec![readme];
    for dir in [root.join("src"), root.join("src").join("text"), root.join("tests")] {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_file() {
                texts.push(std::fs::read_to_string(e.path()).unwrap_or_default());
            }
        }
    }
    let needle = ["exo", "kosm"].concat();
    for t in texts {
        assert!(!t.to_lowercase().contains(&needle));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resources_templates_and_prompts_are_listed() {
    let (h, _d) = harness().await;
    let res = h.client.list_all_resources().await.unwrap();
    let mut uris: Vec<String> = res.iter().map(|r| r.uri.clone()).collect();
    uris.sort();
    let mut want: Vec<String> = RESOURCES.iter().map(|s| s.to_string()).collect();
    want.sort();
    assert_eq!(uris, want);
    assert!(res.iter().all(|r| r.description.as_deref().is_some_and(|d| !d.is_empty()) && !r.name.is_empty()));
    let tpl = h.client.list_all_resource_templates().await.unwrap();
    let mut t: Vec<String> = tpl.iter().map(|t| t.uri_template.clone()).collect();
    t.sort();
    let mut want: Vec<String> = RESULT_FILES.iter().map(|n| format!("cloud-map-generator://result/{{id}}/{n}")).collect();
    want.sort();
    assert_eq!(t, want);
    let meta = tpl.iter().find(|t| t.uri_template.ends_with("meta.json")).unwrap();
    assert_eq!(meta.mime_type.as_deref(), Some("application/json"));
    let prompts = h.client.list_all_prompts().await.unwrap();
    let mut names: Vec<&str> = prompts.iter().map(|p| p.name.as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["bake_for_engine", "design_planet_clouds"]);
    let arg = |name: &str| -> Vec<String> {
        prompts.iter().find(|p| p.name == name).unwrap().arguments.as_ref().unwrap().iter().map(|a| a.name.clone()).collect()
    };
    assert_eq!(arg("design_planet_clouds"), vec!["planet_description"]);
    assert_eq!(arg("bake_for_engine"), vec!["bodies_file"]);
    assert!(prompts.iter().all(|p| p.description.as_deref().is_some_and(|d| !d.is_empty())));
    let info = h.client.peer_info().expect("server info");
    let server = info.server_info.as_ref().expect("an implementation identity");
    assert_eq!((server.name.as_str(), server.version.as_str()), ("cloud-map-generator", env!("CARGO_PKG_VERSION")));
    let instructions = info.instructions.clone().unwrap_or_default();
    assert!(instructions.contains("view_map") && instructions.contains("compute") && instructions.contains("cloud-map-generator://capabilities"));
}

// ------------------------------------------------------------------------------------------------ physics tools

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn list_presets_gives_the_ranges_and_marks_experiments() {
    let (h, d) = harness().await;
    let r = h.good("list_presets", json!({})).await;
    let st = r.structured_content.clone().unwrap();
    let by = |k: &str| st["presets"].as_array().unwrap().iter().find(|p| p["key"] == k).cloned().unwrap_or_else(|| panic!("{k}"));
    for k in ["earth", "earth-retrograde", "fast-rotator", "slow-rotator", "super-earth", "mars", "titan", "venus"] {
        by(k);
    }
    assert_eq!((by("earth")["experiment_required"].clone(), by("titan")["experiment_required"].clone()), (json!(false), json!(true)));
    assert!(by("earth-retrograde")["rotation_period_days"].as_f64().unwrap() < 0.0);
    assert_eq!(st["limits"]["rotation_period_days"], json!([-400.0, 400.0]));
    assert_eq!(st["widths"], json!([256, 512, 1024, 2048, 4096]));
    assert_eq!(st["max_width"], 1024);
    assert!(st["estimate_seconds"]["512"].as_f64().unwrap() > 0.0);
    let folder = std::path::PathBuf::from(s(&st, "output_folder"));
    assert_eq!(folder, cloud_map_generator_mcp::sandbox::resolve_path(&d.out).unwrap());
    assert_eq!(st["read_dirs"].as_array().unwrap().len(), 1);
    assert!(text_of(&r).contains(s(&st, "output_folder")), "a text-only client finds the folder too");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn derive_physics_checks_a_planet_before_generating() {
    let (h, _d) = harness().await;
    let e = h.good_json("derive_physics", json!({"preset": "earth"})).await;
    let dv = &e["derived"];
    let hadley = dv["hadley_edge_deg"].as_f64().unwrap();
    assert!(25.0 < hadley && hadley < 35.0 && dv["has_eddies"] == true && dv["cyclones_per_hemisphere"].as_i64().unwrap() >= 3);
    let mut texel: Vec<&String> = dv["texel_km"].as_object().unwrap().keys().collect();
    texel.sort();
    assert_eq!(dv["bakeable"], true);
    assert_eq!(texel, vec!["1024", "2048", "256", "4096", "512"]);
    let slow = h.good_json("derive_physics", json!({"preset": "earth", "overrides": {"rotation_period_days": 4}})).await;
    assert_eq!(slow["physics"]["rotation_period_days"], 4.0);
    assert!(slow["derived"]["cyclones_per_hemisphere"].is_null());
    assert!(slow["derived"]["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().to_lowercase().contains("storm track")));
    let own = h
        .good_json("derive_physics", json!({"overrides": {"rotation_period_days": 1.5, "radius_km": 3000, "gravity_mps2": 4, "scale_height_km": 10}}))
        .await;
    assert!(own["preset"].is_null());
    assert_eq!(own["physics"]["cloud_regime"], "water");
    let titan = h.good_json("derive_physics", json!({"preset": "titan"})).await;
    assert_eq!(titan["derived"]["bakeable"], false);
    assert!(!titan["derived"]["warnings"].as_array().unwrap().is_empty());
    assert!(titan["derived"]["cyclone_spin_north"].is_null() && slow["derived"]["cyclone_spin_south"].is_null());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn derive_physics_says_which_way_the_lows_turn_and_a_retrograde_spin_reverses_it() {
    let (h, _d) = harness().await;
    let fwd = h.good("derive_physics", json!({"preset": "earth"})).await;
    let d = &fwd.structured_content.as_ref().unwrap()["derived"];
    assert_eq!((s(d, "cyclone_spin_north"), s(d, "cyclone_spin_south")), ("counterclockwise", "clockwise"));
    assert!(text_of(&fwd).contains("counterclockwise in the north"));
    let back = h.good_json("derive_physics", json!({"preset": "earth", "overrides": {"rotation_period_days": -1.5}})).await;
    let d = &back["derived"];
    assert_eq!((s(d, "cyclone_spin_north"), s(d, "cyclone_spin_south")), ("clockwise", "counterclockwise"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn physics_errors_say_what_to_change() {
    let (h, _d) = harness().await;
    let m = h.bad("derive_physics", json!({"preset": "nowhere"})).await;
    assert!(m.contains("preset") && m.contains("earth"), "{m}");
    assert!(h.bad("derive_physics", json!({"preset": "eart"})).await.contains("Did you mean"));
    let m = h.bad("derive_physics", json!({"preset": "earth", "overrides": {"radius_km": 10}})).await;
    assert!(m.contains("radius_km") && m.contains("500") && m.contains("30000"), "{m}");
    assert!(h.bad("derive_physics", json!({"overrides": {"radius_km": 6000}})).await.contains("rotation_period_days"), "the rest is missing");
    let m = h.bad("derive_physics", json!({"preset": "earth", "overrides": {"rotation": 1}})).await;
    assert!(m.contains("rotation") && m.contains("Extra inputs"), "{m}");
    assert!(h.bad("derive_physics", json!({"preset": "earth", "overrides": {"radius_km": "nan"}})).await.contains("finite"));
}

// ------------------------------------------------------------------------------------------------ generate_map

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn generate_map_summary_image_and_saved_files() {
    let (h, d) = harness().await;
    let r = h.good("generate_map", json!({"preset": "slow-rotator", "width": 256, "seed": 7})).await;
    let st = r.structured_content.clone().unwrap();
    assert_eq!(r.content.len(), 2);
    let (_, dims) = decode_image(&r.content[1]);
    assert_eq!(dims, (256, 128));
    assert_eq!((st["width"].clone(), st["height"].clone(), st["seed"].clone()), (json!(256), json!(128), json!(7)));
    assert!(st["seconds"].as_f64().unwrap() > 0.0);
    assert_eq!(st["physics"]["rotation_period_days"], 4.0);
    assert_eq!(st["options"]["surface"], "none");
    let keys: Vec<&str> = st["metrics"].as_array().unwrap().iter().map(|m| m["key"].as_str().unwrap()).collect();
    for k in ["mean", "coherence", "zonal_rmse"] {
        assert!(keys.contains(&k), "{k}");
    }
    assert!(st["metrics"].as_array().unwrap().iter().all(|m| m["hint"].as_str().is_some_and(|h| !h.is_empty())));
    assert_eq!(lats(&st["zonal"]), fifteen());
    assert!(st["zonal"].as_array().unwrap().iter().all(|z| !z["earth"].is_null()));
    assert_eq!(st["saved"], true);
    let id = s(&st, "id");
    let folder = std::path::PathBuf::from(s(&st, "folder"));
    assert_eq!(folder, cloud_map_generator_mcp::sandbox::resolve_path(&d.out.join(id)).unwrap());
    for k in ["master", "preview", "r8gz", "type", "high_share", "wind", "meta", "metrics"] {
        assert!(st["files"].get(k).is_some(), "{k}");
    }
    for (name, p) in st["files"].as_object().unwrap() {
        let p = std::path::PathBuf::from(p.as_str().unwrap());
        assert!(p.is_file() && p.parent().unwrap() == folder, "{name}");
    }
    let mut res: Vec<&String> = st["resources"].as_object().unwrap().keys().collect();
    res.sort();
    assert_eq!(res, vec!["coverage.png", "high_share.png", "meta.json", "type.png", "wind.png"]);
    assert!(st["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("256")), "the width note travels with the result");
    let text = text_of(&r);
    assert!(text.contains("Reliable at this width") && !text.contains("Largest gaps"), "no chasing metrics a 256 map cannot meet");
    assert!(!d.out.join(".pending").exists(), "the scratch folder is gone");
    assert_eq!(st["compute"]["used"], "cpu");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn generate_map_not_saved_writes_nothing_and_reports_progress() {
    let (h, d) = harness().await;
    std::fs::create_dir_all(&d.out).unwrap();
    let before: Vec<_> = std::fs::read_dir(&d.out).unwrap().map(|e| e.unwrap().file_name()).collect();
    let r = h.call_with_progress("generate_map", json!({"preset": "earth", "width": 256, "isotropic": true, "save": false})).await;
    assert!(r.is_error != Some(true), "{}", text_of(&r));
    let st = r.structured_content.clone().unwrap();
    assert!(st["saved"] == false && st["folder"].is_null() && st["files"] == json!({}) && st["resources"] == json!({}));
    assert!(text_of(&r).contains("Not saved") && decode_image(&r.content[1]).1 == (256, 128));
    let after: Vec<_> = std::fs::read_dir(&d.out).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(before, after);
    assert!(eventually(5, || h.progress.lock().unwrap().len() >= 3).await);
    let seen: Vec<ProgressNotificationParam> = h.progress.lock().unwrap().clone();
    // The rmcp client runs every notification handler in its own task, so the ARRIVAL order seen here is not the
    // order the server sent them in (tests/binary.rs checks the wire order over raw stdio): compare as a set.
    let mut values: Vec<f64> = seen.iter().map(|p| p.progress).collect();
    values.sort_by(f64::total_cmp);
    assert!(values.len() >= 3 && values.iter().all(|v| (0.0..=1.0).contains(v)), "{values:?}");
    assert_eq!(*values.last().unwrap(), 1.0);
    assert!(seen.iter().all(|p| p.total == Some(1.0) && p.message.as_deref().is_some_and(|m| !m.is_empty())));
    assert!(st["metrics"].as_array().unwrap().iter().any(|m| !m["relative_gap"].is_null()), "the rows name what differs from Earth");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn generate_map_refuses_what_it_should_with_a_way_out() {
    let (h, d) = harness().await;
    let m = h.bad("generate_map", json!({"preset": "earth", "width": 2048})).await; // this server's max width is 1024
    assert!(m.contains("--max-width 2048") && m.contains("start_generation"), "{m}");
    assert!(h.bad("generate_map", json!({"preset": "earth", "width": 300})).await.contains("256, 512, 1024, 2048, 4096"));
    let m = h.bad("generate_map", json!({"preset": "mars", "width": 256})).await;
    assert!(m.contains("experiment") && m.contains("not water"), "{m}");
    let m = h.bad("generate_map", json!({"preset": "earth", "overrides": {"rotation_period_days": 0.05}, "width": 256})).await;
    assert!(m.contains("cyclones per hemisphere") && m.contains("at least"), "{m}");
    assert!(h.bad("generate_map", json!({"preset": "earth", "seed": 1_099_511_627_776u64, "width": 256})).await.contains("less than or equal"));
    assert!(h.bad("generate_map", json!({"preset": "earth", "surface": "ocean", "width": 256})).await.contains("surface"));
    assert!(h.bad("generate_map", json!({"width": 256})).await.contains("preset"), "neither a preset nor physics");
    assert!(!d.out.join(".pending").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_widths_a_server_accepts_follow_its_max_width() {
    let d = Dirs::new();
    let h = connect(cloud_map_generator_mcp::config::ServerConfig::new(&d.out), MockBackend::new()).await; // the default: 2048
    let m = h.bad("generate_map", json!({"preset": "earth", "width": 2048})).await;
    assert!(m.contains("start_generation") && !m.contains("--max-width"), "{m}");
    let m = h.bad("generate_map", json!({"preset": "earth", "width": 4096})).await;
    assert!(m.contains("--max-width 4096") && m.contains("about"), "{m}");
    assert!(h.bad("start_generation", json!({"preset": "earth", "width": 4096})).await.contains("--max-width 4096"));
    let big = connect(cloud_map_generator_mcp::config::ServerConfig::new(&d.out).with_max_width(4096), MockBackend::new()).await;
    assert!(big.bad("generate_map", json!({"preset": "earth", "width": 4096})).await.contains("start_generation"));
    assert_eq!(big.good_json("list_presets", json!({})).await["max_width"], 4096);
}

// ------------------------------------------------------------------------------------------------ jobs

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_job_runs_to_completion_through_the_tools() {
    let (h, _d) = harness().await;
    let st = h.good_json("start_generation", json!({"preset": "earth", "width": 256, "isotropic": true, "save": false})).await;
    assert!(!s(&st, "job_id").is_empty() && st["width"] == 256 && st["estimate_seconds"].as_f64().unwrap() > 0.0);
    assert!(s(&st, "next").contains("get_generation"));
    let job = h.good("get_generation", json!({"job_id": st["job_id"]})).await;
    let j = job.structured_content.clone().unwrap();
    assert_eq!(j["kind"], "generate");
    let done = h.wait_job(s(&st, "job_id"), Duration::from_secs(60)).await;
    assert!(done["status"] == "done" && done["progress"] == 1.0 && done["result"]["width"] == 256 && done["error"].is_null(), "{done}");
    let again = h.good("get_generation", json!({"job_id": st["job_id"]})).await;
    assert_eq!(decode_image(&again.content[1]).1, (256, 128), "the same image block as generate_map");
    assert!(h.bad("get_generation", json!({"job_id": "nope"})).await.to_lowercase().contains("unknown"));
    assert!(h.bad("cancel_generation", json!({"job_id": "ffffffffffff"})).await.contains("Unknown job_id"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn job_states_queue_and_cancel_with_a_gated_engine() {
    let d = Dirs::new();
    let gate = Gate::new();
    let h = connect(d.config(), MockBackend::new().with_gate(gate.clone())).await;
    let start = |seed: u32| json!({"preset": "earth", "width": 256, "seed": seed, "save": false});
    let mut ids = vec![s(&h.good_json("start_generation", start(0)).await, "job_id").to_string()];
    assert!(gate.wait_entered(Duration::from_secs(10)));
    let running = h.good_json("get_generation", json!({"job_id": ids[0]})).await;
    let p = running["progress"].as_f64().unwrap();
    assert!(running["status"] == "running" && 0.2 < p && p < 0.3 && s(&running, "message").contains("gate"), "{running}");
    for i in 1..5u32 {
        let started = h.good_json("start_generation", start(i)).await;
        assert!(started["status"] == "queued" && started["queued_ahead"] == i, "{started}");
        ids.push(s(&started, "job_id").to_string());
    }
    let full = h.bad("start_generation", start(9)).await;
    assert!(full.contains("queue is full") && full.contains("cancel_generation"), "{full}");
    assert!(h.bad("generate_map", json!({"preset": "earth", "width": 256, "save": false})).await.contains("queue is full"));
    let queued = h.good_json("get_generation", json!({"job_id": ids[2]})).await;
    assert!(queued["status"] == "queued" && queued["progress"] == 0.0 && queued["elapsed_seconds"] == 0.0, "{queued}");
    let c = h.good_json("cancel_generation", json!({"job_id": ids[2]})).await; // a waiting job goes at once
    assert_eq!(c["status"], "cancelled");
    let c = h.good_json("cancel_generation", json!({"job_id": ids[0]})).await; // a running job is only asked
    assert!(matches!(s(&c, "status"), "running" | "cancelled") && (s(&c, "message").contains("stage boundary") || s(&c, "message").contains("already")), "{c}");
    let end = tokio::time::Instant::now() + Duration::from_secs(10);
    while h.good_json("get_generation", json!({"job_id": ids[0]})).await["status"] != "cancelled" {
        assert!(tokio::time::Instant::now() < end);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let backend = h.backend.clone();
    assert!(eventually(10, || backend.generate_calls() >= 2).await, "the next waiting job took over");
    gate.open();
    for jid in [&ids[1], &ids[3], &ids[4]] {
        let st = h.wait_job(jid, Duration::from_secs(30)).await;
        assert_eq!(st["status"], "done", "{st}");
    }
    let again = h.good_json("cancel_generation", json!({"job_id": ids[1]})).await;
    assert!(again["status"] == "done" && s(&again, "message").contains("already done"), "{again}");
    assert_eq!(h.good_json("get_generation", json!({"job_id": ids[2]})).await["status"], "cancelled");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropping_a_generate_map_call_cancels_its_job() {
    let d = Dirs::new();
    let gate = Gate::new();
    let h = connect(d.config(), MockBackend::new().with_gate(gate.clone())).await;
    let rt = h.server.rt.clone();
    let physics = cloud_map_generator_mcp::rules::resolve_physics(&rt.backend.presets(), Some("earth"), &Default::default()).unwrap();
    let job = rt.jobs.submit("generate", "x", rt.generation_job(physics, 1, 256, GenerateOptions::default(), false), None).unwrap();
    let waiter = tokio::spawn({
        let (rt, job) = (rt.clone(), job.clone());
        async move { rt.wait_for(&job, None).await }
    });
    assert!(gate.wait_entered(Duration::from_secs(10)));
    waiter.abort(); // the call is dropped: the job must not keep running for nobody
    let _ = waiter.await;
    assert!(eventually(10, || job.is_final()).await, "the job was cancelled");
    assert!(job.cancel_requested());
    assert_eq!(job.status().as_str(), "cancelled");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_client_that_gives_up_on_a_tool_call_cancels_the_generation_behind_it() {
    let d = Dirs::new();
    let gate = Gate::new();
    let h = connect(d.config(), MockBackend::new().with_gate(gate.clone())).await;
    let call = h.call("generate_map", json!({"preset": "earth", "width": 256, "save": false}));
    assert!(tokio::time::timeout(Duration::from_millis(600), call).await.is_err(), "the gate holds the call");
    assert!(gate.wait_entered(Duration::from_secs(10)));
    // the abandoned call cancelled its job: the engine is free for the next one
    let st = h.good_json("start_generation", json!({"preset": "earth", "width": 256, "save": false})).await;
    assert_eq!(st["queued_ahead"], 0, "{st}");
    assert!(eventually(10, || gate.entered() >= 2).await, "the next job runs");
    gate.open();
    assert_eq!(h.wait_job(s(&st, "job_id"), Duration::from_secs(30)).await["status"], "done");
}

// ------------------------------------------------------------------------------------------------ view_map

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn view_map_draws_the_map_and_the_globe_from_a_result() {
    let (h, _d) = harness().await;
    let made = h.made().await;
    let rid = s(&made, "id").to_string();
    let m = h.good("view_map", json!({"id": rid, "view": "map", "size": 256})).await;
    let st = m.structured_content.clone().unwrap();
    assert_eq!(decode_image(&m.content[1]).1, (256, 128));
    assert!(st["view"] == "map" && st["source"] == format!("result {rid}").as_str());
    let g = h.good("view_map", json!({"id": rid, "view": "globe", "lon_deg": 190, "lat_deg": -35, "size": 200})).await;
    assert_eq!(decode_image(&g.content[1]).1, (200, 200));
    let gs = g.structured_content.unwrap();
    assert!(gs["lon_deg"] == -170.0 && gs["lat_deg"] == -35.0, "wrapped: {gs}");
    let pic = |extra: Value| {
        let mut a = json!({"id": rid, "view": "globe", "size": 128});
        a.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        a
    };
    let a = decode_image(&h.good("view_map", pic(json!({}))).await.content[1]).0;
    let b = decode_image(&h.good("view_map", pic(json!({}))).await.content[1]).0;
    let c = decode_image(&h.good("view_map", pic(json!({"lon_deg": 90}))).await.content[1]).0;
    assert!(a == b && a != c, "deterministic, and the viewpoint matters");
    for product in ["type", "high_share", "wind"] {
        let r = h.good("view_map", pic(json!({"product": product}))).await;
        assert_eq!(decode_image(&r.content[1]).1, (128, 128), "{product}");
    }
    assert!(text_of(&h.good("view_map", json!({"id": rid})).await).contains("coverage"));
    let pole = h.good("view_map", json!({"id": rid, "view": "globe", "lat_deg": 90, "size": 128})).await;
    assert!(text_of(&pole).contains("directly above the north pole"), "{}", text_of(&pole));
    let south = h.good("view_map", json!({"id": rid, "view": "globe", "lat_deg": -90, "size": 128})).await;
    assert!(text_of(&south).contains("south pole") && text_of(&south).contains("east runs clockwise"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn view_map_reads_a_saved_coverage_png_from_the_sandbox() {
    let (h, d) = harness().await;
    let made = h.made().await;
    let preview = s(&made["files"], "preview").to_string();
    let r = h.good("view_map", json!({"path": preview, "view": "globe", "size": 128})).await;
    assert_eq!(decode_image(&r.content[1]).1, (128, 128));
    assert!(s(&r.structured_content.unwrap(), "source").ends_with(".png"));
    let rel = Path::new(&preview).strip_prefix(cloud_map_generator_mcp::sandbox::resolve_path(&d.out).unwrap()).unwrap().to_string_lossy().to_string();
    assert_eq!(decode_image(&h.good("view_map", json!({"path": rel, "size": 300})).await.content[1]).1, (256, 128));
    assert!(h.bad("view_map", json!({"path": preview, "product": "wind"})).await.to_lowercase().contains("only"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn view_map_argument_errors() {
    let (h, _d) = harness().await;
    let made = h.made().await;
    let rid = s(&made, "id").to_string();
    assert!(h.bad("view_map", json!({})).await.contains("exactly one"));
    assert!(h.bad("view_map", json!({"id": rid, "path": "x.png"})).await.contains("exactly one"));
    assert!(h.bad("view_map", json!({"id": "../../etc/passwd"})).await.contains("not a result id"));
    assert!(h.bad("view_map", json!({"id": "ABCDEF123456"})).await.contains("not a result id"));
    assert!(h.bad("view_map", json!({"id": "0123456789ab"})).await.contains("Unknown result id"));
    assert!(h.bad("view_map", json!({"id": rid, "size": 5000})).await.contains("less than or equal to 1024"));
    assert!(h.bad("view_map", json!({"id": rid, "size": 10})).await.contains("greater than or equal to 64"));
    assert!(h.bad("view_map", json!({"id": rid, "view": "globe", "lat_deg": 120})).await.contains("less than or equal to 90"));
    assert!(h.bad("view_map", json!({"id": rid, "view": "cube"})).await.contains("'map' or 'globe'"));
}

// ------------------------------------------------------------------------------------------------ measure_map

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn measure_map_reads_a_saved_map_and_a_16_bit_master() {
    let (h, d) = harness().await;
    let made = h.made().await;
    let m = h.good_json("measure_map", json!({"path": made["files"]["master"]})).await;
    assert!(m["width"] == 256 && m["height"] == 128);
    let keys: Vec<&str> = m["metrics"].as_array().unwrap().iter().map(|x| x["key"].as_str().unwrap()).collect();
    assert!(keys.contains(&"mean") && keys.contains(&"seam_ratio"));
    assert_eq!(lats(&m["zonal"]), fifteen());
    assert!(m["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("256")));
    let mean = |v: &Value| v["metrics"].as_array().unwrap().iter().find(|x| x["key"] == "mean").unwrap()["generated"].as_f64().unwrap();
    assert!((mean(&m) - mean(&made)).abs() < 2e-3, "same map, same numbers");
    let rel = Path::new(s(&made["files"], "preview")).strip_prefix(cloud_map_generator_mcp::sandbox::resolve_path(&d.out).unwrap()).unwrap().to_string_lossy().to_string();
    assert_eq!(h.good_json("measure_map", json!({"path": rel})).await["width"], 256, "an 8-bit PNG, relative path");
    let mine = d.read.join("mine.png"); // a map from somewhere else, in a --read-dir directory
    std::fs::copy(s(&made["files"], "preview"), &mine).unwrap();
    assert!(!h.good_json("measure_map", json!({"path": "mine.png"})).await["metrics"].as_array().unwrap().is_empty(), "found through the read dir");
    let v = h.good("view_map", json!({"path": mine.to_str().unwrap(), "size": 128})).await; // and by full path
    assert_eq!(decode_image(&v.content[1]).1, (128, 64));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn measure_map_cannot_read_outside_the_sandbox() {
    let (h, d) = harness().await;
    let secret = d.outside.join("secret.png");
    std::fs::write(&secret, gray_png(256, 128, 0)).unwrap();
    for bad in [secret.to_str().unwrap(), "../outside/secret.png", "..\\outside\\secret.png", "../../../../../../Windows/win.ini", "/etc/passwd", "C:\\Windows\\win.ini"] {
        let m = h.bad("measure_map", json!({"path": bad})).await;
        assert!(m.contains("outside the readable") || m.contains("not an existing file"), "{bad}: {m}");
        assert!(!m.contains("Traceback"));
        if !Path::new(bad).is_absolute() {
            assert!(!m.contains(d.outside.to_str().unwrap()), "where the file really was is not revealed: {m}");
        }
    }
    for bad in ["\\\\server\\share\\a.png", "//server/share/a.png", "\\\\?\\C:\\a.png"] {
        assert!(h.bad("measure_map", json!({"path": bad})).await.contains("network (UNC) or device path"), "{bad}");
    }
    assert!(h.bad("measure_map", json!({"path": "C:a.png"})).await.contains("drive-relative"));
    assert!(h.bad("measure_map", json!({"path": "a\u{0}.png"})).await.contains("control characters"));
    assert!(h.bad("measure_map", json!({"path": "  "})).await.contains("empty"));
    std::fs::create_dir_all(&d.out).unwrap();
    if link_dir(&d.out.join("escape"), &d.outside) {
        assert!(h.bad("measure_map", json!({"path": "escape/secret.png"})).await.contains("outside the readable"));
        let full = d.out.join("escape").join("secret.png");
        assert!(h.bad("view_map", json!({"path": full.to_str().unwrap()})).await.contains("outside the readable"));
        unlink_dir(&d.out.join("escape"));
    }
    let big = std::fs::File::create(d.out.join("huge.png")).unwrap();
    big.set_len(MAX_READ_BYTES + 1).unwrap();
    drop(big);
    assert!(h.bad("measure_map", json!({"path": "huge.png"})).await.contains("above the 64 MiB limit"));
    std::fs::remove_file(d.out.join("huge.png")).unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn measure_map_refuses_files_it_should_not_decode() {
    let (h, d) = harness().await;
    std::fs::create_dir_all(&d.out).unwrap();
    let put = |name: &str, bytes: &[u8]| std::fs::write(d.out.join(name), bytes).unwrap();
    put("text.png", b"not an image");
    assert!(h.bad("measure_map", json!({"path": "text.png"})).await.contains("not a readable PNG"));
    put("wide.png", &gray_png(300, 100, 0));
    assert!(h.bad("measure_map", json!({"path": "wide.png"})).await.contains("2:1"));
    put("tiny.png", &gray_png(18, 9, 0));
    assert!(h.bad("measure_map", json!({"path": "tiny.png"})).await.contains("256 pixels wide"));
    put("huge_dims.png", &gray_png(9000, 4500, 0)); // a small file, a big decode
    assert!(h.bad("measure_map", json!({"path": "huge_dims.png"})).await.contains("up to 8192x4096"));
    put("real.jpg", &[0xff, 0xd8, 0xff, 0xe0, 0, 16, b'J', b'F', b'I', b'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]);
    assert!(h.bad("measure_map", json!({"path": "real.jpg"})).await.contains("only PNG"));
    std::fs::create_dir_all(d.out.join("sub")).unwrap();
    assert!(h.bad("measure_map", json!({"path": "sub"})).await.contains("not an existing file"));
    assert!(h.bad("measure_map", json!({"path": "missing.png"})).await.contains("not an existing file"));
    assert!(h.bad("view_map", json!({"path": "text.png"})).await.contains("not a readable PNG"));
}

// ------------------------------------------------------------------------------------------------ bake_bodies

fn write_bodies(path: &Path, bodies: Value) -> String {
    std::fs::write(path, json!({"bodies": bodies}).to_string()).unwrap();
    path.to_str().unwrap().to_string()
}

fn slow() -> Value {
    json!({"key": "Dev:Slow", "rotationPeriodDays": 4.0, "radiusKm": 6371, "gravityMps2": 9.81, "scaleHeightKm": 8.0, "cloudRegime": "water"})
}

fn with(mut v: Value, k: &str, x: Value) -> Value {
    v[k] = x;
    v
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bake_bodies_writes_a_manifest_and_files_under_the_out_dir() {
    let (h, d) = harness().await;
    let mars = json!({"key": "mars", "rotationPeriodDays": 1.02596, "radiusKm": 3389.5, "gravityMps2": 3.721, "scaleHeightKm": 11.1, "cloudRegime": "thin water ice"});
    let real = json!({"key": "earth", "rotationPeriodDays": 0.99727, "radiusKm": 6371, "gravityMps2": 9.81, "scaleHeightKm": 8.0, "cloudRegime": "water", "measured": true});
    let bodies = write_bodies(&d.read.join("bodies.json"), json!([slow(), mars, real]));
    let r = h
        .call_with_progress("bake_bodies", json!({"bodies_file": bodies, "version": "t1", "url_prefix": "assets/clouds/t1", "list_width": 256, "only": "list"}))
        .await;
    assert!(r.is_error != Some(true), "{}", text_of(&r));
    let st = r.structured_content.clone().unwrap();
    assert_eq!(st["version"], "t1");
    assert_eq!(std::path::PathBuf::from(s(&st, "folder")), cloud_map_generator_mcp::sandbox::resolve_path(&d.out.join("bake").join("t1")).unwrap());
    let list = st["manifest"]["list"].as_object().unwrap();
    assert_eq!(list.keys().collect::<Vec<_>>(), vec!["Dev:Slow"]);
    assert_eq!(list["Dev:Slow"]["hires"]["width"], 256);
    assert!(s(&list["Dev:Slow"], "slice").starts_with("assets/clouds/t1/"));
    let skipped: Vec<&str> = st["skipped"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    let mut keys: Vec<&str> = skipped.iter().map(|x| x.split(':').next().unwrap()).collect();
    keys.sort();
    assert_eq!(keys, vec!["earth", "mars"]);
    assert!(skipped.join(" ").contains("measured"));
    let files: Vec<&str> = st["files"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
    assert!(!files.is_empty() && files.iter().all(|f| f.starts_with("bake/t1/") && d.out.join(f).is_file()));
    assert!(st["files_total"].as_u64().unwrap() == files.len() as u64 && st["truncated"] == false);
    assert!(!d.out.join("bake").join("t1").join("manifest.json").exists(), "a partial bake writes none");
    assert!(text_of(&r).contains("partial bake"));
    assert!(eventually(5, || !h.progress.lock().unwrap().is_empty()).await);
    let seen: Vec<f64> = h.progress.lock().unwrap().iter().map(|p| p.progress).collect();
    assert!(seen.windows(2).all(|w| w[0] <= w[1]) && *seen.last().unwrap() == 1.0, "{seen:?}");
    let cov = files.iter().find(|f| f.ends_with("_coverage_2048.png")).expect("a baked coverage map");
    let v = h.good("view_map", json!({"path": cov, "size": 256})).await; // a baked map can be looked at
    assert_eq!(decode_image(&v.content[1]).1, (256, 128));
    // a full bake writes the manifest
    let full = h.good_json("bake_bodies", json!({"bodies_file": write_bodies(&d.read.join("one.json"), json!([slow()])), "version": "full", "list_width": 256, "library_width": 256})).await;
    assert!(d.out.join("bake").join("full").join("manifest.json").is_file());
    assert_eq!(full["manifest"]["library"].as_array().unwrap().len(), 4);
    assert!(full["files"].as_array().unwrap().iter().any(|f| f == "bake/full/manifest.json"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bake_bodies_refuses_hostile_or_oversized_requests() {
    let (h, d) = harness().await;
    let ok = write_bodies(&d.read.join("ok.json"), json!([slow()]));
    let secret = write_bodies(&d.outside.join("bodies.json"), json!([slow()]));
    assert!(h.bad("bake_bodies", json!({"bodies_file": secret})).await.contains("outside the readable"));
    assert!(h.bad("bake_bodies", json!({"bodies_file": "../outside/bodies.json"})).await.contains("outside the readable"));
    for version in ["../x", "a/b", "..", "", "v 1", &"a".repeat(40), ".hidden"] {
        assert!(h.bad("bake_bodies", json!({"bodies_file": ok, "version": version})).await.contains("letters, digits"), "{version}");
    }
    assert!(h.bad("bake_bodies", json!({"bodies_file": ok, "url_prefix": "a b\n<script>"})).await.contains("url_prefix"));
    assert!(h.bad("bake_bodies", json!({"bodies_file": ok, "list_width": 2048})).await.contains("--max-width 2048"));
    assert!(h.bad("bake_bodies", json!({"bodies_file": ok, "library_width": 300})).await.contains("library_width"));
    assert!(h.bad("bake_bodies", json!({"bodies_file": ok, "k": 50})).await.contains("less than or equal to 8"));
    for key in ["../../evil", "a/b", "a\\b", "", "..", &"x".repeat(80), "a..b"] {
        let bad = write_bodies(&d.read.join("badkey.json"), json!([with(slow(), "key", json!(key))]));
        assert!(h.bad("bake_bodies", json!({"bodies_file": bad, "only": "list"})).await.contains("key"), "{key}");
    }
    let many: Vec<Value> = (0..33).map(|i| with(slow(), "key", json!(format!("b{i}")))).collect();
    write_bodies(&d.read.join("many.json"), json!(many));
    assert!(h.bad("bake_bodies", json!({"bodies_file": "many.json"})).await.contains("at most 32"));
    std::fs::write(d.read.join("broken.json"), "{not json").unwrap();
    assert!(h.bad("bake_bodies", json!({"bodies_file": "broken.json"})).await.contains("not valid"));
    std::fs::write(d.read.join("empty.json"), "{\"bodies\": []}").unwrap();
    assert!(h.bad("bake_bodies", json!({"bodies_file": "empty.json"})).await.contains("non-empty"));
    assert!(h.bad("bake_bodies", json!({"bodies_file": "nothing.json"})).await.contains("not an existing file"));
    std::fs::write(d.read.join("land.u8"), [0u8; 8]).unwrap();
    let cases = [
        (json!({"width": 4, "height": 2, "land": "../outside/secret.png", "elevation": "land.u8"}), "outside"),
        (json!({"width": 4, "height": 2, "land": d.outside.join("secret.png").to_str().unwrap(), "elevation": "land.u8"}), "outside"),
        (json!({"width": 4, "height": 2, "land": "land.u8", "elevation": "land.u8"}), "bytes"),
        (json!({"width": 4, "height": 2, "land": "nope.u8", "elevation": "land.u8"}), "not an existing file"),
        (json!({"width": 4, "height": 2, "land": 1, "elevation": "land.u8"}), "file name"),
    ];
    for (surface, why) in cases {
        write_bodies(&d.read.join("surf.json"), json!([with(slow(), "surface", surface.clone())]));
        let m = h.bad("bake_bodies", json!({"bodies_file": "surf.json", "only": "list"})).await;
        assert!(m.contains(why), "{surface}: {m}");
    }
    assert!(!d.out.join("bake").join("evil").exists() && !d.base.join("evil").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bake_bodies_takes_the_planets_inline_without_a_file() {
    let (h, d) = harness().await;
    let inline = json!([
        {"key": "alpha", "rotation_period_days": 4.0, "radius_km": 6000, "gravity_mps2": 9, "scale_height_km": 8},
        {"key": "gas", "rotation_period_days": 0.4, "radius_km": 20000, "gravity_mps2": 20, "scale_height_km": 30, "cloud_regime": "methane"},
    ]);
    let st = h.good_json("bake_bodies", json!({"bodies": inline, "version": "inl", "list_width": 256, "only": "list"})).await;
    assert_eq!(st["manifest"]["list"].as_object().unwrap().keys().collect::<Vec<_>>(), vec!["alpha"]);
    let skipped: Vec<&str> = st["skipped"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().split(':').next().unwrap()).collect();
    assert_eq!(skipped, vec!["gas"]);
    let written: Value = serde_json::from_str(&std::fs::read_to_string(d.out.join("bodies-inl.json")).unwrap()).unwrap(); // what a file would hold
    assert!(written["bodies"][0]["rotationPeriodDays"] == 4.0 && written["bodies"][0]["radiusKm"] == 6000.0);
    let schema = &h.tool("bake_bodies").input_schema;
    assert!(schema["properties"].get("bodies").is_some() && schema.get("required").is_none(), "the format is in the schema itself");
    assert!(schema["$defs"]["BakeBody"]["properties"].get("rotation_period_days").is_some());
    let m = h.bad("bake_bodies", json!({})).await;
    assert!(m.contains("`bodies`") && m.contains("bodies_file") && m.contains("exactly one"), "{m}");
    assert!(h.bad("bake_bodies", json!({"bodies": inline, "bodies_file": "x.json"})).await.contains("exactly one"));
    assert!(h.bad("bake_bodies", json!({"bodies": []})).await.contains("at least one"));
    let too_small = json!([{"key": "alpha", "rotation_period_days": 4.0, "radius_km": 5, "gravity_mps2": 9, "scale_height_km": 8}]);
    assert!(h.bad("bake_bodies", json!({"bodies": too_small, "list_width": 256, "only": "list"})).await.contains("radius_km"), "the ranges are enforced");
    assert!(h.bad("bake_bodies", json!({"bodies": [{"key": "x"}]})).await.contains("rotation_period_days"), "what is missing is named");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_bodies_format_resource_documents_both_spellings_and_the_grids() {
    let (h, _d) = harness().await;
    let (text, _) = h.read_text("cloud-map-generator://docs/bodies-format").await;
    for needle in ["rotationPeriodDays", "rotation_period_days", "camelCase", "little-endian int16", "output_folder", "cloudRegime", "measured", "bodies=["] {
        assert!(text.contains(needle), "{needle}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_methane_planet_error_names_the_experiment_argument() {
    let (h, _d) = harness().await;
    let m = h.bad("generate_map", json!({"preset": "titan", "width": 256, "save": false})).await;
    assert!(m.contains("experiment=true") && m.contains("methane"), "{m}");
    let m = h.bad("generate_map", json!({"preset": "earth", "overrides": {"cloud_regime": "methane"}, "width": 256, "save": false})).await;
    assert!(m.contains("experiment=true"), "{m}");
    let ok = h.good_json("generate_map", json!({"preset": "titan", "width": 256, "save": false, "experiment": true})).await;
    assert!(ok["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("Experiment")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bake_bodies_estimates_before_it_runs() {
    let d = Dirs::new();
    let cfg = cloud_map_generator_mcp::config::ServerConfig::new(&d.out).with_max_width(4096).with_read_dirs(vec![d.base.path().to_path_buf()]);
    let h = connect(cfg, MockBackend::new()).await;
    let many: Vec<Value> = (0..30).map(|i| with(slow(), "key", json!(format!("b{i}")))).collect();
    write_bodies(&d.base.join("b.json"), json!(many));
    let m = h.bad("bake_bodies", json!({"bodies_file": "b.json", "list_width": 4096, "library_width": 4096, "k": 8})).await;
    assert!(m.contains("minutes") && m.contains("Lower the widths"), "{m}");
    assert!(!d.out.join("bake").exists());
}

// ------------------------------------------------------------------------------------------------ explain, resources, prompts

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explain_model_matches_its_resource() {
    let (h, _d) = harness().await;
    let r = h.good("explain_model", json!({})).await;
    let text = s(r.structured_content.as_ref().unwrap(), "markdown").to_string();
    assert_eq!(text, text_of(&r));
    assert_eq!(text, h.backend.explain_model());
    assert!(text.contains("Water clouds only") && text.contains("Frozen flow") && text.starts_with("# cloud-map-generator"));
    let (res, mime) = h.read_text("cloud-map-generator://docs/model").await;
    assert_eq!((res, mime.as_deref()), (text, Some("text/markdown")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn static_resources() {
    let (h, _d) = harness().await;
    let (text, mime) = h.read_text("cloud-map-generator://presets").await;
    let data: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(mime.as_deref(), Some("application/json"));
    let keys: Vec<&str> = data["presets"].as_array().unwrap().iter().map(|p| p["key"].as_str().unwrap()).collect();
    assert!(keys.contains(&"earth") && keys.contains(&"mars"));
    assert_eq!(data["widths"], json!([256, 512, 1024, 2048, 4096]));
    let (png, mime) = h.read_blob("cloud-map-generator://reference/earth-coverage.png").await;
    assert_eq!((png_dims(&png), mime.as_deref()), ((2048, 1024), Some("image/png")));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn result_files_are_resources() {
    let (h, _d) = harness().await;
    let made = h.made().await;
    let rid = s(&made, "id");
    for name in ["coverage.png", "type.png", "high_share.png", "wind.png"] {
        let (png, mime) = h.read_blob(&format!("cloud-map-generator://result/{rid}/{name}")).await;
        let (w, hh) = png_dims(&png);
        assert!(mime.as_deref() == Some("image/png") && w == 2 * hh && w >= 64, "{name}");
    }
    let (meta, mime) = h.read_text(&format!("cloud-map-generator://result/{rid}/meta.json")).await;
    assert_eq!(mime.as_deref(), Some("application/json"));
    assert_eq!(serde_json::from_str::<Value>(&meta).unwrap()["seed"], 7);
    for uri in made["resources"].as_object().unwrap().values() {
        assert!(h.read(uri.as_str().unwrap()).await.is_ok(), "{uri}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn result_resources_refuse_what_is_not_a_saved_result() {
    let (h, _d) = harness().await;
    let fails = |uri: &str, texts: &[&str]| {
        let uri = uri.to_string();
        let texts: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
        let h = &h;
        async move {
            let e = h.read(&uri).await.expect_err(&uri).to_string();
            assert!(texts.iter().any(|t| e.contains(t.as_str())), "{uri}: {e}");
            assert!(!e.contains("Traceback"));
        }
    };
    fails("cloud-map-generator://result/0123456789ab/coverage.png", &["Unknown result id"]).await;
    fails("cloud-map-generator://result/not-hex-at-all/coverage.png", &["not a result id"]).await;
    fails("cloud-map-generator://result/../coverage.png", &["Unknown resource"]).await;
    fails("cloud-map-generator://result/%2e%2e/coverage.png", &["Unknown resource", "not a result id"]).await;
    fails("cloud-map-generator://result/..%2F..%2Fetc/meta.json", &["Unknown resource", "not a result id"]).await;
    fails("cloud-map-generator://result/0123456789ab/../../secret.png", &["Unknown resource"]).await;
    fails("cloud-map-generator://result/0123456789ab/master.png", &["Unknown resource"]).await; // only the listed names
    fails("cloud-map-generator://nothing", &["Unknown resource"]).await;
    let unsaved = h.good_json("generate_map", json!({"preset": "earth", "width": 256, "isotropic": true, "save": false})).await;
    fails(&format!("cloud-map-generator://result/{}/coverage.png", s(&unsaved, "id")), &["save=false"]).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prompts_render_the_loops() {
    let (h, _d) = harness().await;
    let prompt = |name: &str, key: &str, value: &str| {
        let params = GetPromptRequestParams::new(name.to_string()).with_arguments(args(json!({ key: value })));
        let peer = h.client.peer().clone();
        async move { peer.get_prompt(params).await }
    };
    let p = prompt("design_planet_clouds", "planet_description", "a tidally slow ocean world").await.unwrap();
    assert_eq!(p.messages[0].role, Role::User);
    let text = p.messages[0].content.as_text().unwrap().text.clone();
    assert!(text.contains("a tidally slow ocean world"));
    for tool in ["derive_physics", "generate_map", "view_map", "start_generation", "get_generation", "explain_model"] {
        assert!(text.contains(tool), "{tool}");
    }
    assert!(text.contains("experiment"), "non-water clouds are flagged");
    let long = prompt("design_planet_clouds", "planet_description", &"x ".repeat(5000)).await.unwrap();
    assert!(long.messages[0].content.as_text().unwrap().text.len() < 6000, "the argument is shortened");
    let b = prompt("bake_for_engine", "bodies_file", "bodies.json").await.unwrap();
    let text = b.messages[0].content.as_text().unwrap().text.clone();
    for needle in ["bodies.json", "bake_bodies", "url_prefix", "measure_map"] {
        assert!(text.contains(needle), "{needle}");
    }
    let missing = h.client.get_prompt(GetPromptRequestParams::new("design_planet_clouds".to_string())).await;
    assert!(missing.is_err(), "the argument is required");
    assert!(h.client.get_prompt(GetPromptRequestParams::new("nope".to_string())).await.is_err());
}

// ------------------------------------------------------------------------------------------------ errors do not leak

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unexpected_errors_and_paths_never_reach_the_caller() {
    let d = Dirs::new();
    let out = d.out.display().to_string();
    let leak = format!("cannot read {out}{}x.png nor /etc/passwd nor C:\\Users\\bob\\k.txt", std::path::MAIN_SEPARATOR);
    let h = connect(d.config(), MockBackend::new().with_failure(MockFailure::Refuse(leak))).await;
    let m = h.bad("generate_map", json!({"preset": "leak", "width": 256})).await;
    assert!(m.contains("preset"), "an unknown preset is still the rules' error: {m}"); // sanity: the failure is on generate
    let m = h.bad("generate_map", json!({"preset": "earth", "width": 256})).await;
    assert!(m.contains("<out-dir>") && m.contains("<path>"), "{m}");
    let base = d.base.path().to_string_lossy().to_string();
    assert!(!m.contains(&base) && !m.contains("passwd") && !m.contains("bob"), "{m}");
    assert!(!m.contains("Traceback") && !m.contains("File \""));
    let h = connect(d.config(), MockBackend::new().with_failure(MockFailure::Panic(format!("boom in {} at C:\\Users\\bob\\secret.py", file!())))).await;
    let m = h.bad("derive_physics", json!({"preset": "earth"})).await;
    assert_eq!(m, "Error executing tool derive_physics", "a crash keeps its text on the server");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failing_job_reports_a_clean_error() {
    let d = Dirs::new();
    let secret = format!("secret detail {}", d.base.path().display());
    let backend = MockBackend::new().with_failure(MockFailure::Internal(secret)).with_stage_delay(Duration::from_millis(400)); // long enough for start_generation to return first
    let h = connect(d.config(), backend).await;
    let m = h.bad("generate_map", json!({"preset": "earth", "width": 256, "save": false})).await;
    let base = d.base.path().to_string_lossy().to_string();
    assert!(m.contains("internal error (") && !m.contains(&base) && !m.contains("secret detail"), "{m}");
    let st = h.good_json("start_generation", json!({"preset": "earth", "width": 256, "save": false})).await;
    let done = h.wait_job(s(&st, "job_id"), Duration::from_secs(20)).await;
    assert!(done["status"] == "failed" && s(&done, "error").contains("internal error") && !s(&done, "error").contains("secret detail"), "{done}");
    assert!(!d.out.join(".pending").exists(), "the scratch folder of a failed run is gone");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_panicking_engine_is_an_internal_error_not_a_dead_server() {
    let d = Dirs::new();
    let h = connect(d.config(), MockBackend::new().with_failure(MockFailure::Panic("kaboom".into()))).await;
    let m = h.bad("generate_map", json!({"preset": "earth", "width": 256, "save": false})).await;
    assert!(m.contains("internal error (Panic)") && !m.contains("kaboom"), "{m}");
    assert!(h.good("list_presets", json!({})).await.structured_content.is_some(), "the server is alive");
    assert!(!d.out.join(".pending").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_rejected_request_fails_fast_from_start_generation() {
    let (h, _d) = harness().await;
    let m = h.bad("start_generation", json!({"preset": "mars", "width": 256})).await;
    assert!(m.contains("experiment"), "the engine refuses in milliseconds: the call reports it, not a later poll: {m}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unknown_tool_is_a_tool_error() {
    let (h, _d) = harness().await;
    assert_eq!(h.bad("nonexistent_tool", json!({})).await, "Unknown tool: nonexistent_tool");
}

// ------------------------------------------------------------------------------------------------ compute modes

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compute_is_a_four_valued_argument_of_both_generation_tools() {
    let (h, _d) = harness().await;
    for name in ["generate_map", "start_generation"] {
        let schema = &h.tool(name).input_schema;
        let prop = &schema["properties"]["compute"];
        assert_eq!(prop["enum"], json!(["auto", "cpu", "gpu", "hybrid"]));
        assert!(prop["default"] == "auto" && !schema.get("required").is_some_and(|r| r.as_array().unwrap().iter().any(|x| x == "compute")));
        let text = prop["description"].as_str().unwrap();
        for needle in ["GPU-capable engine", "cloud-map-generator://capabilities", "ERROR", "silent fallback", "never fails for lack of a device"] {
            assert!(text.contains(needle), "{name}: {needle}");
        }
        assert!(text.contains("offers: auto, cpu; not available: gpu, hybrid"), "what THIS server really offers: {text}");
        let description = h.tool(name).description.as_deref().unwrap();
        assert!(description.contains("compute") && description.contains("GPU-capable engine") && description.contains("capabilities"));
    }
    let lp = h.tool("list_presets").description.as_deref().unwrap();
    assert!(lp.contains("GPU-capable") && lp.contains("compute"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_output_schemas_carry_the_compute_block_and_the_capabilities() {
    let (h, _d) = harness().await;
    let out = Value::Object(h.tool("generate_map").output_schema.as_ref().unwrap().as_ref().clone());
    assert!(out["properties"].get("compute").is_some());
    let block = out["$defs"]["ComputeOut"]["properties"].as_object().unwrap();
    for k in ["requested", "used", "devices", "note"] {
        assert!(block.contains_key(k), "{k}");
    }
    for k in ["kind", "name", "seconds", "share"] {
        assert!(out["$defs"]["ComputeDeviceUsedOut"]["properties"].get(k).is_some(), "{k}");
    }
    let presets = Value::Object(h.tool("list_presets").output_schema.as_ref().unwrap().as_ref().clone());
    assert!(presets["required"].as_array().unwrap().iter().any(|x| x == "compute"));
    for k in ["default", "engine", "modes", "devices"] {
        assert!(presets["$defs"]["CapabilitiesOut"]["properties"].get(k).is_some(), "{k}");
    }
    let job = Value::Object(h.tool("get_generation").output_schema.as_ref().unwrap().as_ref().clone());
    assert!(job["$defs"]["GenerationOut"]["properties"].get("compute").is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_capabilities_resource_is_the_engines_json() {
    let (h, _d) = harness().await;
    let (text, mime) = h.read_text("cloud-map-generator://capabilities").await;
    assert_eq!(mime.as_deref(), Some("application/json"));
    let caps: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(caps, h.backend.capabilities());
    let keys: Vec<&str> = caps["modes"].as_array().unwrap().iter().map(|m| m["key"].as_str().unwrap()).collect();
    assert_eq!((keys, s(&caps, "engine")), (vec!["auto", "cpu", "gpu", "hybrid"], "mock"));
    let available: Vec<bool> = caps["modes"].as_array().unwrap().iter().map(|m| m["available"].as_bool().unwrap()).collect();
    assert_eq!(available, vec![true, true, false, false]);
    assert!(caps["modes"].as_array().unwrap().iter().filter(|m| m["available"] == false).all(|m| !m["reason"].is_null()));
    assert!(caps["devices"]["gpu"].is_null() && caps["devices"]["cpu"]["threads"].as_u64().unwrap() >= 1);
    let res = h.client.list_all_resources().await.unwrap();
    assert_eq!(res.iter().find(|r| r.uri == "cloud-map-generator://capabilities").unwrap().mime_type.as_deref(), Some("application/json"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn list_presets_carries_the_compute_modes_in_data_and_in_text() {
    let (h, _d) = harness().await;
    let r = h.good("list_presets", json!({})).await;
    assert_eq!(r.structured_content.as_ref().unwrap()["compute"], h.backend.capabilities());
    let text = text_of(&r);
    assert!(text.contains("Compute modes (engine mock") && text.contains("available auto, cpu"), "{text}");
    assert!(text.contains("not available: gpu (") && text.contains("hybrid (") && text.contains("no GPU path"), "{text}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_result_says_what_ran_it() {
    let (h, _d) = harness().await;
    let r = h.good("generate_map", json!({"preset": "slow-rotator", "width": 256, "seed": 7})).await;
    let st = r.structured_content.unwrap();
    let c = &st["compute"];
    assert_eq!((s(c, "requested"), s(c, "used"), c["note"].is_null()), ("auto", "cpu", true));
    let devices = c["devices"].as_array().unwrap();
    assert_eq!(devices.len(), 1);
    let (d0, secs) = (&devices[0], devices[0]["seconds"].as_f64().unwrap());
    assert!(d0["kind"] == "cpu" && d0["share"] == 1.0 && 0.0 < secs && secs <= st["seconds"].as_f64().unwrap() + 0.01, "{d0}");
    let text = text_of(&h.good("generate_map", json!({"preset": "slow-rotator", "width": 256, "save": false})).await);
    assert!(text.contains("Compute: cpu (requested auto): cpu ") && text.contains("100%"), "{text}");
    assert!(st["options"].get("compute").is_none(), "where it ran is not a property of the map");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unavailable_compute_mode_is_a_tool_error_with_the_reason_and_queues_nothing() {
    let (h, _d) = harness().await;
    for tool in ["generate_map", "start_generation"] {
        for mode in ["gpu", "hybrid"] {
            let reason = h.backend.capabilities()["modes"].as_array().unwrap().iter().find(|m| m["key"] == mode).unwrap()["reason"].as_str().unwrap().to_string();
            let queued = h.server.rt.jobs.job_count();
            let m = h.bad(tool, json!({"preset": "slow-rotator", "width": 256, "compute": mode})).await;
            assert!(m.contains(&format!("compute: '{mode}' is not available:")) && m.contains(&reason), "no silent fallback: {m}");
            assert!(m.contains("'auto'") && m.contains("'cpu'"), "{m}");
            assert_eq!(h.server.rt.jobs.job_count(), queued, "refused before a job was queued");
        }
    }
    assert_eq!(h.backend.generate_calls(), 0, "no CPU map was made instead");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_value_outside_the_four_modes_is_refused_by_the_schema() {
    let (h, _d) = harness().await;
    for tool in ["generate_map", "start_generation"] {
        let m = h.bad(tool, json!({"preset": "slow-rotator", "width": 256, "compute": "turbo"})).await;
        assert!(m.contains("compute") && m.contains("'auto'") && m.contains("'hybrid'"), "{m}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compute_reaches_the_engine_and_every_device_is_reported() {
    let d = Dirs::new();
    let h = connect(d.config(), MockBackend::with_gpu()).await;
    let listed = h.good("list_presets", json!({})).await;
    let st = listed.structured_content.clone().unwrap();
    assert_eq!((s(&st["compute"]["devices"]["gpu"], "name"), s(&st["compute"], "engine")), ("Mock GPU", "mock"));
    let text = text_of(&listed);
    assert!(text.contains("available auto, cpu, gpu, hybrid") && !text.contains("not available"), "{text}");
    let offered = h.tool("generate_map").input_schema["properties"]["compute"]["description"].as_str().unwrap().to_string();
    assert!(offered.contains("offers: auto, cpu, gpu, hybrid.") && !offered.contains("; not available:"), "{offered}");

    let hybrid = h.good("generate_map", json!({"preset": "earth", "width": 256, "save": false, "compute": "hybrid"})).await;
    assert_eq!(h.backend.options_seen().last().unwrap().compute, "hybrid");
    let c = hybrid.structured_content.clone().unwrap()["compute"].clone();
    assert_eq!((s(&c, "requested"), s(&c, "used")), ("hybrid", "hybrid"));
    let kinds: Vec<&str> = c["devices"].as_array().unwrap().iter().map(|x| x["kind"].as_str().unwrap()).collect();
    assert_eq!(kinds, vec!["gpu", "cpu"]);
    let share: f64 = c["devices"].as_array().unwrap().iter().map(|x| x["share"].as_f64().unwrap()).sum();
    assert!((share - 1.0).abs() < 1e-9);
    let t = text_of(&hybrid);
    assert!(t.contains("Compute: hybrid (requested hybrid): gpu ") && t.contains("70%") && t.contains(", cpu ") && t.contains("30%."), "{t}");

    let auto = h.good("generate_map", json!({"preset": "earth", "width": 256, "save": false})).await; // compute left out: auto
    assert_eq!(h.backend.options_seen().last().unwrap().compute, "auto");
    let c = &auto.structured_content.as_ref().unwrap()["compute"];
    assert_eq!((s(c, "used"), s(c, "requested")), ("gpu", "auto"));
    assert!(text_of(&auto).contains("Compute: gpu (requested auto): gpu ") && text_of(&auto).contains("100%. auto chose the GPU"));

    let started = h.good_json("start_generation", json!({"preset": "earth", "width": 256, "save": false, "compute": "gpu"})).await;
    let job = h.wait_job(s(&started, "job_id"), Duration::from_secs(20)).await;
    assert_eq!(job["result"]["compute"]["used"], "gpu");
    assert_eq!(h.backend.options_seen().last().unwrap().compute, "gpu");
    assert!(!h.bad("generate_map", json!({"preset": "earth", "width": 256, "compute": "turbo"})).await.is_empty(), "the schema still refuses it");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_mock_is_deterministic_and_honest_about_what_it_is() {
    let (h, _d) = harness().await;
    let a = h.good_json("generate_map", json!({"preset": "earth", "width": 256, "seed": 3, "save": false})).await;
    let b = h.good_json("generate_map", json!({"preset": "earth", "width": 256, "seed": 3, "save": false})).await;
    let c = h.good_json("generate_map", json!({"preset": "earth", "width": 256, "seed": 4, "save": false})).await;
    assert_eq!(a["metrics"], b["metrics"], "the same inputs and seed give the same map");
    assert_ne!(a["metrics"], c["metrics"]);
    let va = decode_image(&h.good("view_map", json!({"id": a["id"], "size": 256})).await.content[1]).0;
    let vb = decode_image(&h.good("view_map", json!({"id": b["id"], "size": 256})).await.content[1]).0;
    assert_eq!(va, vb);
    assert!(h.backend.explain_model().contains("MOCK engine"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_globe_of_a_saved_png_has_the_sub_point_at_its_centre_pixel() {
    let (h, d) = harness().await;
    std::fs::write(d.read.join("blob.png"), blob_png(512, 40.0, -30.0)).unwrap();
    let path = d.read.join("blob.png").to_str().unwrap().to_string();
    let n = 129usize; // an odd size: the centre pixel is exactly the point that faces the viewer
    let at = |lon: f64, lat: f64| {
        let (h, path) = (&h, path.clone());
        async move {
            let r = h.good("view_map", json!({"path": path, "view": "globe", "lon_deg": lon, "lat_deg": lat, "size": n})).await;
            let (png, dims) = decode_image(&r.content[1]);
            assert_eq!(dims, (n as u32, n as u32));
            png
        }
    };
    let on_the_cloud = at(40.0, -30.0).await;
    let c = rgb_pixel(&on_the_cloud, n / 2, n / 2);
    assert!(c[0] > 200 && c[1] > 200 && c[2] > 200, "the sub-point is on the cloud: {c:?}");
    assert_eq!(rgb_pixel(&on_the_cloud, 0, 0), [7, 9, 14], "the corner is space");
    let away = at(160.0, -30.0).await;
    let c = rgb_pixel(&away, n / 2, n / 2);
    assert!(c[0] < 60, "the sub-point is on the dark sphere: {c:?}");
    // north up, east to the right: a cloud 25 degrees east of the viewer is on the right half
    std::fs::write(d.read.join("east.png"), blob_png(512, 25.0, 0.0)).unwrap();
    let r = h.good("view_map", json!({"path": d.read.join("east.png").to_str().unwrap(), "view": "globe", "lon_deg": 0, "lat_deg": 0, "size": n})).await;
    let png = decode_image(&r.content[1]).0;
    assert!(rgb_pixel(&png, n / 2 + 30, n / 2)[0] > rgb_pixel(&png, n / 2 - 30, n / 2)[0] + 40);
    let a = decode_image(&h.good("view_map", json!({"path": path, "view": "globe", "lon_deg": 10, "lat_deg": 20, "size": 64})).await.content[1]).0;
    let b = decode_image(&h.good("view_map", json!({"path": path, "view": "globe", "lon_deg": 10, "lat_deg": 20, "size": 64})).await.content[1]).0;
    assert_eq!(a, b, "deterministic");
}
