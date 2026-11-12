//! End-to-end tests of `cloud-map-generator serve`: the real binary, a real socket, a raw HTTP client.
//!
//! The routes and their contract are tested as functions in `src/serve/http_tests.rs`; this file tests what only a
//! socket shows: that the server listens on a port it prints, keeps connections alive, answers HEAD with the
//! length of the body it does not send, always sends a `Content-Length` (also for a download of megabytes), reads a
//! body or refuses it, checks `Host` and `Origin`, and that the command line refuses what it should.
//! `tests/test_api_contract.py` runs the same API contract from Python against this server and the Python one.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

fn root() -> PathBuf {
    std::env::var_os("CLOUD_MAP_GENERATOR_ROOT").map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(".."))
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cloud-map-generator-cli-serve-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A running `cloud-map-generator serve --port 0`, killed when dropped.
struct Server {
    child: Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    fn start(extra: &[&str]) -> Server {
        let mut child = Command::new(env!("CARGO_BIN_EXE_cloud-map-generator"))
            .args(["serve", "--port", "0", "--widths", "64,128", "--quiet"])
            .args(extra)
            .env("CLOUD_MAP_GENERATOR_ROOT", root())
            .stderr(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .expect("the binary runs");
        let stderr = child.stderr.take().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut lines = BufReader::new(stderr).lines();
            let first = lines.next().and_then(|l| l.ok()).unwrap_or_default();
            let _ = tx.send(first);
            // Keep draining, so that the child never blocks on a full pipe.
            for _ in lines {}
        });
        let line = rx.recv_timeout(Duration::from_secs(60)).expect("the server prints its address");
        let port = line
            .split("http://127.0.0.1:")
            .nth(1)
            .and_then(|rest| rest.split('/').next())
            .and_then(|p| p.parse().ok())
            .unwrap_or_else(|| panic!("an address line, got {line:?}"));
        Server { child, port }
    }
}

struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Resp {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
    fn json(&self) -> serde_json::Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|e| panic!("JSON: {e}: {}", String::from_utf8_lossy(&self.body)))
    }
}

fn write_request(stream: &mut TcpStream, port: u16, method: &str, path: &str, headers: &[(&str, &str)], body: Option<&[u8]>, close: bool) {
    let mut head = format!("{method} {path} HTTP/1.1\r\n");
    if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("host")) {
        head.push_str(&format!("Host: 127.0.0.1:{port}\r\n"));
    }
    if close {
        head.push_str("Connection: close\r\n");
    }
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(b) = body {
        if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-length")) {
            head.push_str(&format!("Content-Length: {}\r\n", b.len()));
        }
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes()).unwrap();
    if let Some(b) = body {
        stream.write_all(b).unwrap();
    }
    stream.flush().unwrap();
}

/// One response off the stream: the head, then exactly `Content-Length` bytes (none for HEAD).
fn read_response(reader: &mut BufReader<TcpStream>, head_only: bool) -> Resp {
    let mut line = String::new();
    reader.read_line(&mut line).expect("a status line");
    let status: u16 = line.split_whitespace().nth(1).unwrap_or_else(|| panic!("a status line, got {line:?}")).parse().unwrap();
    let mut headers = Vec::new();
    loop {
        line.clear();
        reader.read_line(&mut line).unwrap();
        let l = line.trim_end();
        if l.is_empty() {
            break;
        }
        let (k, v) = l.split_once(':').unwrap_or_else(|| panic!("a header line, got {l:?}"));
        headers.push((k.trim().to_string(), v.trim().to_string()));
    }
    let mut resp = Resp { status, headers, body: Vec::new() };
    assert!(resp.header("Transfer-Encoding").is_none(), "no chunked transfer: {:?}", resp.headers);
    if !head_only && status != 204 {
        let n: usize = resp.header("Content-Length").unwrap_or_else(|| panic!("a Content-Length on {status}: {:?}", resp.headers)).parse().unwrap();
        resp.body = vec![0; n];
        reader.read_exact(&mut resp.body).unwrap();
    }
    resp
}

fn request(port: u16, method: &str, path: &str, headers: &[(&str, &str)], body: Option<&[u8]>) -> Resp {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    write_request(&mut stream, port, method, path, headers, body, true);
    read_response(&mut BufReader::new(stream), method == "HEAD")
}

fn get(port: u16, path: &str) -> Resp {
    request(port, "GET", path, &[], None)
}

fn post(port: u16, path: &str, payload: &serde_json::Value) -> Resp {
    request(port, "POST", path, &[("Content-Type", "application/json")], Some(payload.to_string().as_bytes()))
}

const EARTH: &str = r#"{"rotation_period_days": 0.99727, "radius_km": 6371, "gravity_mps2": 9.81, "scale_height_km": 8.0, "cloud_regime": "water"}"#;

fn earth() -> serde_json::Value {
    serde_json::from_str(EARTH).unwrap()
}

#[test]
fn the_page_and_the_api_are_served_over_a_socket() {
    let s = Server::start(&[]);
    let index = get(s.port, "/");
    assert_eq!((index.status, index.header("Content-Type")), (200, Some("text/html; charset=utf-8")));
    assert!(String::from_utf8_lossy(&index.body).to_lowercase().contains("<html"));
    assert_eq!(index.header("Server"), Some("cloudgen-ui"));
    assert_eq!(index.header("Cache-Control"), Some("no-store"));
    let d = get(s.port, "/api/defaults").json();
    assert_eq!(d["backend"], "rust");
    assert_eq!(d["widths"], serde_json::json!([64, 128]));
    assert_eq!(d["compute"]["engine"], "rust");
    assert_eq!(d["compute"]["modes"].as_array().unwrap().len(), 4);
    // The probe found a GPU or did not: either way the answer is the contract's, and `gpu` mode says why it is not offered yet.
    let gpu = d["compute"]["modes"].as_array().unwrap().iter().find(|m| m["key"] == "gpu").unwrap();
    assert_eq!(gpu["available"].as_bool().unwrap(), gpu["reason"].is_null());
}

#[test]
fn a_download_of_megabytes_has_a_content_length_and_head_reports_it_without_a_body() {
    let s = Server::start(&[]);
    // The reference map is 1.5 MB: past the size where a server would switch to chunked transfer.
    let full = get(s.port, "/api/reference/coverage.png");
    assert_eq!((full.status, full.header("Content-Type")), (200, Some("image/png")));
    assert!(full.body.len() > 1_000_000);
    assert_eq!(full.header("Content-Length"), Some(full.body.len().to_string().as_str()));
    let head = request(s.port, "HEAD", "/api/reference/coverage.png", &[], None);
    assert_eq!((head.status, head.body.len()), (200, 0));
    assert_eq!(head.header("Content-Length"), full.header("Content-Length"));
    assert_eq!(head.header("Content-Type"), Some("image/png"));
    assert_eq!(request(s.port, "HEAD", "/api/nothing", &[], None).status, 404);
}

#[test]
fn a_connection_is_kept_alive_for_several_requests() {
    let s = Server::start(&[]);
    let mut stream = TcpStream::connect(("127.0.0.1", s.port)).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    write_request(&mut stream, s.port, "GET", "/api/defaults", &[], None, false);
    let first = read_response(&mut reader, false);
    write_request(&mut stream, s.port, "POST", "/api/derive", &[("Content-Type", "application/json")], Some(format!("{{\"body\": {EARTH}}}").as_bytes()), false);
    let second = read_response(&mut reader, false);
    write_request(&mut stream, s.port, "GET", "/api/nothing", &[], None, false);
    let third = read_response(&mut reader, false);
    assert_eq!((first.status, second.status, third.status), (200, 200, 404));
    assert_eq!(second.json()["cyclones_per_hemisphere"], 5);
}

#[test]
fn posts_are_read_by_their_content_length_and_refused_without_one() {
    let s = Server::start(&[]);
    let ok = post(s.port, "/api/derive", &serde_json::json!({"body": earth()}));
    assert_eq!(ok.status, 200);
    assert_eq!(ok.json()["hadley_edge_deg"], 30.0);
    // No Content-Length: a 400 that says so.
    let none = request(s.port, "POST", "/api/derive", &[("Content-Type", "application/json")], None);
    assert_eq!(none.status, 400);
    assert!(none.json()["error"].as_str().unwrap().contains("Content-Length is required"));
    // Not JSON, and too big: 400, and the connection still answers afterwards.
    let bad = request(s.port, "POST", "/api/derive", &[], Some(b"not json"));
    assert_eq!(bad.status, 400);
    assert!(bad.json()["error"].as_str().unwrap().starts_with("invalid JSON"));
    let big = request(s.port, "POST", "/api/derive", &[], Some(&vec![b'x'; 70_000]));
    assert_eq!(big.status, 400);
    assert!(big.json()["error"].as_str().unwrap().contains("at most 65536 bytes"));
    let nan = request(s.port, "POST", "/api/derive", &[], Some(br#"{"body": NaN}"#));
    assert_eq!(nan.status, 400);
    assert_eq!(request(s.port, "DELETE", "/api/defaults", &[], None).status, 501);
    assert_eq!(request(s.port, "GET", "/api/derive", &[], None).status, 405);
}

#[test]
fn a_foreign_host_or_origin_is_refused_with_a_403() {
    let s = Server::start(&[]);
    let body = format!("{{\"body\": {EARTH}}}");
    let call = |headers: &[(&str, &str)]| request(s.port, "POST", "/api/derive", headers, Some(body.as_bytes()));
    assert_eq!(call(&[("Origin", "http://evil.example")]).status, 403);
    assert_eq!(call(&[("Origin", "null")]).status, 403);
    assert_eq!(call(&[("Host", "evil.example")]).status, 403);
    assert_eq!(call(&[("Host", &format!("evil.example:{}", s.port))]).status, 403);
    assert!(call(&[("Origin", "http://evil.example")]).json()["error"].as_str().unwrap().contains("Origin"));
    assert_eq!(call(&[("Origin", &format!("http://127.0.0.1:{}", s.port))]).status, 200);
    assert_eq!(call(&[("Origin", &format!("http://localhost:{}", s.port)), ("Host", &format!("localhost:{}", s.port))]).status, 200);
    assert_eq!(call(&[]).status, 200, "a client with no Origin (curl) passes");
}

#[test]
fn a_map_is_generated_and_downloaded_over_a_socket() {
    let s = Server::start(&[]);
    let req = serde_json::json!({"body": earth(), "seed": 7, "width": 64, "options": {"compute": "cpu", "isotropic": true}});
    let out = post(s.port, "/api/generate", &req);
    assert_eq!(out.status, 200, "{}", String::from_utf8_lossy(&out.body));
    let j = out.json();
    assert_eq!((j["width"].as_u64(), j["height"].as_u64()), (Some(64), Some(32)));
    assert_eq!((j["compute"]["requested"].as_str(), j["compute"]["used"].as_str()), (Some("cpu"), Some("cpu")));
    let cov = get(s.port, j["images"]["coverage"].as_str().unwrap());
    assert_eq!((cov.status, cov.header("Content-Type")), (200, Some("image/png")));
    let mut reader = png::Decoder::new(std::io::Cursor::new(&cov.body)).read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!((info.width, info.height, info.color_type, info.bit_depth), (64, 32, png::ColorType::Grayscale, png::BitDepth::Eight));
    let href = j["downloads"][1]["href"].as_str().unwrap();
    let r8 = get(s.port, href);
    let disp = r8.header("Content-Disposition").unwrap();
    assert!(disp.starts_with("attachment; filename=\"cloudgen_s7_64x32_") && disp.ends_with(".r8.gz\""), "{disp}");
    let head = request(s.port, "HEAD", href, &[], None);
    assert_eq!((head.status, head.body.len()), (200, 0));
    assert_eq!(head.header("Content-Length"), Some(r8.body.len().to_string().as_str()));
    assert_eq!(head.header("Content-Disposition"), Some(disp));
    let mut raw = Vec::new();
    flate2::read::GzDecoder::new(&r8.body[..]).read_to_end(&mut raw).unwrap();
    assert_eq!(raw.len(), 64 * 32);
    // A GPU mode that is not offered is a 400 that says why, and the server still answers.
    let defaults = get(s.port, "/api/defaults").json();
    for mode in defaults["compute"]["modes"].as_array().unwrap().iter().filter(|m| !m["available"].as_bool().unwrap()) {
        let key = mode["key"].as_str().unwrap();
        let r = post(s.port, "/api/generate", &serde_json::json!({"body": earth(), "seed": 7, "width": 64, "options": {"compute": key}}));
        assert_eq!(r.status, 400, "{key}");
        let msg = r.json()["error"].as_str().unwrap().to_string();
        assert!(msg.starts_with("options.compute:") && msg.contains(mode["reason"].as_str().unwrap().trim_end_matches('.')), "{msg}");
    }
    assert_eq!(get(s.port, "/api/defaults").status, 200);
}

#[test]
fn a_static_dir_is_served_instead_of_the_embedded_page() {
    let dir = scratch("static");
    std::fs::write(dir.join("index.html"), "<!doctype html><title>mine</title>").unwrap();
    std::fs::write(dir.join("extra.css"), "body{}").unwrap();
    let s = Server::start(&["--static-dir", dir.to_str().unwrap()]);
    let index = get(s.port, "/");
    assert_eq!(index.status, 200);
    assert!(String::from_utf8_lossy(&index.body).contains("<title>mine</title>"));
    let css = get(s.port, "/static/extra.css");
    assert_eq!((css.status, css.header("Content-Type")), (200, Some("text/css; charset=utf-8")));
    assert_eq!(get(s.port, "/static/app.js").status, 404, "the embedded files are not served when a folder is");
    assert_eq!(get(s.port, "/static/../index.html").status, 404);
}

fn run_failing(args: &[&str]) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_cloud-map-generator")).args(args).env("CLOUD_MAP_GENERATOR_ROOT", root()).output().expect("the binary runs");
    assert!(!out.status.success(), "cloudgen {args:?} should fail");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn the_command_line_refuses_what_cannot_work() {
    assert!(run_failing(&["serve", "--port", "0", "--widths", "63"]).contains("--widths: 63"));
    assert!(run_failing(&["serve", "--port", "0", "--widths", "257"]).contains("--widths: 257"));
    assert!(run_failing(&["serve", "--port", "0", "--max-width", "65536"]).contains("--max-width: 65536"));
    assert!(run_failing(&["serve", "--port", "0", "--keep", "0"]).contains("--keep"));
    assert!(run_failing(&["serve", "--port", "0", "--static-dir", "no/such/folder"]).contains("--static-dir"));
    assert!(run_failing(&["serve", "--port", "0", "--reference", "no/such/file.png"]).contains("--reference"));
    // A port that is taken: the message names it.
    let taken = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = taken.local_addr().unwrap().port().to_string();
    let msg = run_failing(&["serve", "--port", &port]);
    assert!(msg.contains(&format!("cannot listen on 127.0.0.1:{port}")), "{msg}");
}

#[test]
fn the_server_listens_on_loopback_only() {
    let s = Server::start(&[]);
    assert!(TcpStream::connect(("127.0.0.1", s.port)).is_ok());
    // The address of the machine's main network interface (a UDP "connection" sends nothing), if it has one: the port is closed there.
    let Some(ip) = std::net::UdpSocket::bind("0.0.0.0:0").and_then(|u| u.connect("192.0.2.1:9").map(|()| u)).and_then(|u| u.local_addr()).ok().map(|a| a.ip()).filter(|ip| !ip.is_loopback() && !ip.is_unspecified()) else {
        eprintln!("no network interface besides loopback: nothing more to check");
        return;
    };
    assert!(TcpStream::connect_timeout(&(ip, s.port).into(), Duration::from_secs(2)).is_err(), "{ip}:{} answers: the server must listen on 127.0.0.1 only", s.port);
}

/// One request written as it is and the answer read until the server closes (or the timeout): `(status, body)`.
fn raw_exchange(port: u16, head: &str, close_write: bool) -> (u16, Vec<u8>) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    stream.write_all(head.as_bytes()).unwrap();
    if close_write {
        stream.shutdown(std::net::Shutdown::Write).unwrap();
    }
    let mut all = Vec::new();
    let _ = stream.read_to_end(&mut all);
    let text = String::from_utf8_lossy(&all).into_owned();
    let status = text.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let body = all.windows(4).position(|w| w == b"\r\n\r\n").map_or_else(Vec::new, |i| all[i + 4..].to_vec());
    (status, body)
}

/// A body declared far larger than the limit, never sent: an HTTP server crate once allocated the whole declared
/// length when it dropped the request, and the process aborted. The answer is the limit's 400 (as `ui/server.py`'s),
/// and the server answers the next request.
#[test]
fn a_declared_body_over_the_limit_is_refused_and_the_server_lives() {
    let s = Server::start(&[]);
    for length in ["99999999999", "4294967296", "18446744073709551615", "18446744073709551616", "99999999999999999999999", "-1", "65537"] {
        for close_write in [true, false] {
            let head = format!("POST /api/derive HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Type: application/json\r\nContent-Length: {length}\r\n\r\n", s.port);
            let (status, body) = raw_exchange(s.port, &head, close_write);
            assert_eq!(status, 400, "length {length}: {}", String::from_utf8_lossy(&body));
            let msg: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert!(msg["error"].as_str().unwrap().contains("at most 65536 bytes"), "{msg}");
            assert_eq!(get(s.port, "/api/defaults").status, 200, "the server is gone after Content-Length: {length}");
        }
    }
    // Not a length at all.
    for length in ["abc", "1e3", "0x10", "5 6", ""] {
        let head = format!("POST /api/derive HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nContent-Length: {length}\r\n\r\n", s.port);
        let (status, body) = raw_exchange(s.port, &head, true);
        assert_eq!(status, 400, "length {length:?}");
        assert!(String::from_utf8_lossy(&body).contains("Content-Length is not a number"), "{length:?}: {}", String::from_utf8_lossy(&body));
    }
    // A head that is not a request, and one too large: answered, not a crash.
    assert_eq!(raw_exchange(s.port, "GET\r\n\r\n", true).0, 400);
    let huge = format!("GET / HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nX-Pad: {}\r\n\r\n", s.port, "x".repeat(70_000));
    assert_eq!(raw_exchange(s.port, &huge, true).0, 431);
    assert_eq!(get(s.port, "/api/defaults").status, 200);
}
