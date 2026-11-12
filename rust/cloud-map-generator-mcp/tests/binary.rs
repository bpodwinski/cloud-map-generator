//! The built binary, over stdio, through the rmcp child-process transport: a clean protocol (nothing but
//! JSON-RPC on stdout), nothing created unasked, the flags, and the exit codes.

mod common;

use std::process::Command;

use common::*;
use rmcp::model::{CallToolRequestParams, ReadResourceRequestParams, ResourceContents};
use rmcp::transport::{ConfigureCommandExt, TokioChildProcess};
use rmcp::ServiceExt;
use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_cloud-map-generator-mcp");

async fn spawn(args: &[&str], envs: &[(&str, &str)]) -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    spawn_with(args, envs, true).await
}

/// Spawns the server; `mock` serves the stand-in engine (`--mock`), otherwise the real one.
async fn spawn_with(args: &[&str], envs: &[(&str, &str)], mock: bool) -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    let cmd = tokio::process::Command::new(BIN).configure(|c| {
        if mock {
            c.arg("--mock");
        }
        c.args(args);
        for (k, v) in envs {
            c.env(k, v);
        }
    });
    let transport = TokioChildProcess::new(cmd).expect("the binary starts");
    ().serve(transport).await.expect("the client connects to the binary")
}

async fn call(client: &rmcp::service::RunningService<rmcp::RoleClient, ()>, name: &str, a: Value) -> rmcp::model::CallToolResult {
    client.call_tool(CallToolRequestParams::new(name.to_string()).with_arguments(args(a))).await.expect("a tool result")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_binary_speaks_a_clean_protocol_and_creates_nothing_unasked() {
    let t = TempDir::new("binary");
    let out = t.join("o");
    let client = spawn(&["--out-dir", out.to_str().unwrap(), "--max-width", "256"], &[]).await;
    let tools = client.list_all_tools().await.unwrap();
    assert_eq!(tools.len(), 10);
    let r = call(&client, "derive_physics", json!({"preset": "earth", "overrides": {"rotation_period_days": 2}})).await;
    assert!(r.is_error != Some(true));
    assert_eq!(r.structured_content.unwrap()["physics"]["rotation_period_days"], 2.0);
    let r = call(&client, "measure_map", json!({"path": "../outside.png"})).await;
    assert_eq!(r.is_error, Some(true));
    assert!(text_of(&r).contains("outside the readable"), "{}", text_of(&r));
    // a 2 MB frame
    let res = client.read_resource(ReadResourceRequestParams::new("cloud-map-generator://reference/earth-coverage.png")).await.unwrap();
    match &res.contents[0] {
        ResourceContents::BlobResourceContents { blob, .. } => {
            use base64::Engine as _;
            let png = base64::engine::general_purpose::STANDARD.decode(blob).unwrap();
            assert_eq!(png_dims(&png), (2048, 1024));
        }
        other => panic!("{other:?}"),
    }
    let r = call(&client, "explain_model", json!({})).await;
    assert!(r.is_error != Some(true) && r.structured_content.unwrap()["markdown"].as_str().unwrap().starts_with("# cloud-map-generator"));
    let r = call(&client, "generate_map", json!({"preset": "earth", "width": 256, "seed": 1})).await;
    assert!(r.is_error != Some(true), "{}", text_of(&r));
    assert!(out.join(r.structured_content.unwrap()["id"].as_str().unwrap()).is_dir(), "a saved result is a folder named by its id");
    client.cancel().await.unwrap();
    // a second server that is only asked to derive never touches the disk
    let quiet = t.join("quiet");
    let client = spawn(&["--out-dir", quiet.to_str().unwrap()], &[]).await;
    assert!(call(&client, "derive_physics", json!({"preset": "mars"})).await.is_error != Some(true));
    client.cancel().await.unwrap();
    assert!(!quiet.exists(), "nothing was asked to write, so nothing was created");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_environment_names_the_out_dir_and_a_gpu_can_be_pretended() {
    let t = TempDir::new("binary-env");
    let out = t.join("env-out");
    let client = spawn(&["--max-width", "256"], &[("PLANET_CLOUDGEN_OUT", out.to_str().unwrap()), ("PLANET_CLOUDGEN_MOCK_GPU", "1")]).await;
    let r = call(&client, "list_presets", json!({})).await;
    let st = r.structured_content.unwrap();
    assert!(st["output_folder"].as_str().unwrap().ends_with("env-out"), "{st}");
    assert_eq!(st["compute"]["devices"]["gpu"]["name"], "Mock GPU");
    let r = call(&client, "generate_map", json!({"preset": "earth", "width": 256, "save": false, "compute": "gpu"})).await;
    assert!(r.is_error != Some(true), "{}", text_of(&r));
    assert_eq!(r.structured_content.unwrap()["compute"]["used"], "gpu");
    client.cancel().await.unwrap();
}

#[test]
fn version_and_bad_flags() {
    let out = Command::new(BIN).arg("--version").output().unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), format!("cloud-map-generator-mcp {}", env!("CARGO_PKG_VERSION")));
    let out = Command::new(BIN).args(["--max-width", "300"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty(), "stdout carries the protocol and nothing else");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--max-width must be one of 256, 512, 1024, 2048, 4096, got 300"), "{err}");
    let out = Command::new(BIN).args(["--port", "70000"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--port must be between 1 and 65535"));
    let help = Command::new(BIN).arg("--help").output().unwrap();
    let text = String::from_utf8_lossy(&help.stdout);
    for flag in ["--out-dir", "--max-width", "--read-dir", "--http", "--port", "--log-level", "--version", "PLANET_CLOUDGEN_OUT"] {
        assert!(text.contains(flag), "{flag}");
    }
    for tool in ["list_presets", "bake_bodies", "explain_model"] {
        assert!(text.contains(tool), "{tool}");
    }
}

/// Kills the server process when a test ends, however it ends.
struct KillOnDrop(std::process::Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_binary_serves_http_on_loopback_only_with_the_rebinding_guard() {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    let t = TempDir::new("binary-http");
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let mut child = KillOnDrop(
        Command::new(BIN)
            .args(["--mock", "--http", "--port", &port.to_string(), "--out-dir", t.join("o").to_str().unwrap(), "--max-width", "256"])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let stderr = child.0.stderr.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            let _ = tx.send(line);
        }
    });
    let mut announced = String::new();
    let end = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while std::time::Instant::now() < end {
        if let Ok(line) = rx.recv_timeout(std::time::Duration::from_millis(200)) {
            if line.contains("streamable HTTP on") {
                announced = line;
                break;
            }
        }
    }
    assert!(announced.contains(&format!("http://127.0.0.1:{port}/mcp")) && announced.contains("local only"), "{announced:?}");
    let host = format!("127.0.0.1:{port}");
    let ok = post(port, &host, &[], &initialize()).await;
    assert!(ok.status != 403 && ok.status != 421, "{}", ok.status);
    let info = ok.messages().into_iter().find(|m| m["id"] == 1).expect("the initialize response");
    assert_eq!(info["result"]["serverInfo"]["name"], "cloud-map-generator");
    assert_eq!(post(port, &format!("evil.example.com:{port}"), &[], &initialize()).await.status, 421);
    assert_eq!(post(port, "attacker.test", &[], &initialize()).await.status, 421);
    assert_eq!(post(port, &host, &[("Origin", "http://evil.example.com")], &initialize()).await.status, 403);
}

/// The wire itself, read line by line: nothing but JSON-RPC on stdout, the progress notifications of a call in
/// the order the server sent them (never going backwards, ending at 1) and before its response, and a clean
/// exit when the client closes stdin.
#[test]
fn the_wire_carries_only_json_rpc_and_progress_in_order() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::Stdio;

    let t = TempDir::new("wire");
    let out = t.join("o");
    let mut child = Command::new(BIN)
        .args(["--mock", "--out-dir", out.to_str().unwrap(), "--max-width", "256"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("the binary starts");
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let messages = [
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "wire", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "generate_map", "_meta": {"progressToken": "tok"},
            "arguments": {"preset": "earth", "width": 256, "isotropic": true, "save": false}}}),
    ];
    for m in &messages {
        writeln!(stdin, "{m}").unwrap();
    }
    stdin.flush().unwrap();
    let mut progress: Vec<f64> = Vec::new();
    let mut response: Option<Value> = None;
    let mut line = String::new();
    while response.is_none() {
        line.clear();
        assert!(stdout.read_line(&mut line).unwrap() > 0, "stdout closed before the response");
        let v: Value = serde_json::from_str(line.trim_end()).unwrap_or_else(|e| panic!("stdout is not JSON ({e}): {line}"));
        assert_eq!(v["jsonrpc"], "2.0", "{line}");
        if v["method"] == "notifications/progress" {
            assert_eq!(v["params"]["progressToken"], "tok");
            progress.push(v["params"]["progress"].as_f64().unwrap());
        } else if v["id"] == 2 {
            response = Some(v);
        }
    }
    assert!(progress.len() >= 3 && progress.windows(2).all(|w| w[0] <= w[1]), "{progress:?}");
    assert_eq!(*progress.last().unwrap(), 1.0);
    assert!(response.unwrap()["result"]["isError"] != true);
    drop(stdin); // the client goes away: the server ends on its own
    let mut rest = String::new();
    for l in stdout.lines() {
        rest.push_str(&l.unwrap());
    }
    assert!(rest.is_empty(), "nothing is printed after the last response: {rest}");
    assert!(child.wait().unwrap().success());
    assert!(!out.exists(), "save=false: nothing was written");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_real_engine_makes_a_map_on_the_cpu_and_says_so() {
    let t = TempDir::new("binary-real");
    let out = t.join("o");
    let client = spawn_with(&["--out-dir", out.to_str().unwrap(), "--max-width", "256"], &[], false).await;
    let r = call(&client, "list_presets", json!({})).await;
    let st = r.structured_content.unwrap();
    assert_eq!(st["compute"]["engine"], "rust", "{st}");
    assert_eq!(st["compute"]["modes"][1]["available"], true);
    let r = call(&client, "generate_map", json!({"preset": "earth", "width": 256, "seed": 3, "compute": "cpu"})).await;
    assert!(r.is_error != Some(true), "{}", text_of(&r));
    let st = r.structured_content.unwrap();
    assert_eq!((st["compute"]["requested"].as_str(), st["compute"]["used"].as_str()), (Some("cpu"), Some("cpu")));
    assert_eq!(st["metrics"].as_array().unwrap().len(), 10);
    let folder = out.join(st["id"].as_str().unwrap());
    assert!(folder.join("cloudgen_s3_256x128_16bit.png").is_file(), "the master is in the result's folder");
    client.cancel().await.unwrap();
}
