//! Shared helpers of the integration tests: temporary folders, a JSON-schema checker for the output schemas, and
//! a harness that runs the server in this process and talks to it through the official rmcp client.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use base64::Engine as _;
use cloud_map_generator_mcp::backend::Backend;
use cloud_map_generator_mcp::config::ServerConfig;
use cloud_map_generator_mcp::mock::MockBackend;
use cloud_map_generator_mcp::server::PlanetCloudgenServer;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ClientRequest, ContentBlock, ImageContent, ProgressNotificationParam,
    ReadResourceRequestParams, ReadResourceResult, Request, ResourceContents, ServerResult, Tool,
};
use rmcp::service::{NotificationContext, PeerRequestOptions, RunningService};
use rmcp::{ClientHandler, RoleClient, ServiceExt};
use serde_json::{json, Value};

// ------------------------------------------------------------------------------------------------ temp folders

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A folder under the system temp dir, removed when dropped.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(name: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("cloud-map-generator-mcp-test-{}-{n}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("create a temp folder");
        TempDir(p)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn join(&self, p: impl AsRef<Path>) -> PathBuf {
        self.0.join(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// `out`, `read` and `outside` folders under one temp dir (`out` is not created: the server makes it on first use).
pub struct Dirs {
    pub base: TempDir,
    pub out: PathBuf,
    pub read: PathBuf,
    pub outside: PathBuf,
}

impl Dirs {
    pub fn new() -> Self {
        let base = TempDir::new("mcp");
        let (out, read, outside) = (base.join("out"), base.join("read"), base.join("outside"));
        std::fs::create_dir_all(&read).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        Dirs { base, out, read, outside }
    }

    /// A configuration over these folders: max width 1024 and `read` as a read directory.
    pub fn config(&self) -> ServerConfig {
        ServerConfig::new(&self.out).with_max_width(1024).with_read_dirs(vec![self.read.clone()])
    }
}

/// A directory link (a symlink, or a junction on Windows when symlinks are not allowed). False: cannot.
pub fn link_dir(link: &Path, target: &Path) -> bool {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).is_ok()
    }
    #[cfg(windows)]
    {
        if std::os::windows::fs::symlink_dir(target, link).is_ok() {
            return true;
        }
        std::process::Command::new("cmd")
            .args(["/c", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

/// A file symlink. False: cannot create one here.
pub fn link_file(link: &Path, target: &Path) -> bool {
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(target, link).is_ok()
    }
    #[cfg(windows)]
    {
        std::os::windows::fs::symlink_file(target, link).is_ok()
    }
}

/// Remove a link made by `link_dir` without touching its target.
pub fn unlink_dir(link: &Path) {
    if std::fs::remove_file(link).is_err() {
        let _ = std::fs::remove_dir(link);
    }
}

// ------------------------------------------------------------------------------------------------ pictures

pub fn png_dims(png: &[u8]) -> (u32, u32) {
    assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a], "a PNG signature");
    let reader = png::Decoder::new(std::io::Cursor::new(png)).read_info().expect("a decodable PNG header");
    (reader.info().width, reader.info().height)
}

/// The image block of a result decoded: its PNG bytes and size.
pub fn decode_image(block: &ContentBlock) -> (Vec<u8>, (u32, u32)) {
    let img: &ImageContent = block.as_image().expect("an image block");
    assert_eq!(img.mime_type, "image/png");
    let raw = base64::engine::general_purpose::STANDARD.decode(&img.data).expect("base64 image data");
    let dims = png_dims(&raw);
    (raw, dims)
}

pub fn gray_png(width: usize, height: usize, value: u8) -> Vec<u8> {
    cloud_map_generator_mcp::render::encode_gray8(width, height, &vec![value; width * height])
}

// ------------------------------------------------------------------------------------------------ JSON schema

fn resolve<'a>(root: &'a Value, schema: &'a Value) -> &'a Value {
    match schema.get("$ref").and_then(Value::as_str) {
        Some(r) => {
            let mut cur = root;
            for part in r.trim_start_matches("#/").split('/') {
                cur = &cur[part];
            }
            resolve(root, cur)
        }
        None => schema,
    }
}

fn type_ok(v: &Value, t: &str) -> bool {
    match t {
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64() || v.as_f64().is_some_and(|f| f.fract() == 0.0 && v.is_number()),
        "boolean" => v.is_boolean(),
        "object" => v.is_object(),
        "array" => v.is_array(),
        "null" => v.is_null(),
        _ => true,
    }
}

/// Validate `v` against the subset of JSON Schema the output schemas use: `$ref`, `anyOf`, `type`, `enum`,
/// `properties`, `required`, `additionalProperties: false`, `items`, `minimum`, `maximum`, string lengths.
pub fn validate(root: &Value, schema: &Value, v: &Value, at: &str) -> Result<(), String> {
    let schema = resolve(root, schema);
    if let Some(any) = schema.get("anyOf").and_then(Value::as_array) {
        return if any.iter().any(|s| validate(root, s, v, at).is_ok()) {
            Ok(())
        } else {
            Err(format!("{at}: {v} matches none of the anyOf branches"))
        };
    }
    if let Some(t) = schema.get("type") {
        let ok = match t {
            Value::String(s) => type_ok(v, s),
            Value::Array(a) => a.iter().filter_map(Value::as_str).any(|s| type_ok(v, s)),
            _ => true,
        };
        if !ok {
            return Err(format!("{at}: {v} is not of type {t}"));
        }
    }
    if let Some(e) = schema.get("enum").and_then(Value::as_array) {
        if !e.contains(v) {
            return Err(format!("{at}: {v} is not one of {e:?}"));
        }
    }
    if let (Some(n), Some(min)) = (v.as_f64(), schema.get("minimum").and_then(Value::as_f64)) {
        if n < min {
            return Err(format!("{at}: {n} is below {min}"));
        }
    }
    if let (Some(n), Some(max)) = (v.as_f64(), schema.get("maximum").and_then(Value::as_f64)) {
        if n > max {
            return Err(format!("{at}: {n} is above {max}"));
        }
    }
    if let Some(s) = v.as_str() {
        if let Some(max) = schema.get("maxLength").and_then(Value::as_u64) {
            if s.chars().count() as u64 > max {
                return Err(format!("{at}: too long"));
            }
        }
    }
    if let Some(obj) = v.as_object() {
        for req in schema.get("required").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
            if !obj.contains_key(req) {
                return Err(format!("{at}: missing required {req}"));
            }
        }
        let props = schema.get("properties").and_then(Value::as_object);
        for (k, val) in obj {
            match props.and_then(|p| p.get(k)) {
                Some(s) => validate(root, s, val, &format!("{at}.{k}"))?,
                None => {
                    if schema.get("additionalProperties") == Some(&json!(false)) {
                        return Err(format!("{at}: unexpected property {k}"));
                    }
                }
            }
        }
    }
    if let (Some(arr), Some(items)) = (v.as_array(), schema.get("items")) {
        for (i, x) in arr.iter().enumerate() {
            validate(root, items, x, &format!("{at}[{i}]"))?;
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------ the harness

/// The client side: records every progress notification the server sends.
#[derive(Clone, Default)]
pub struct Recorder {
    pub progress: Arc<Mutex<Vec<ProgressNotificationParam>>>,
}

impl ClientHandler for Recorder {
    async fn on_progress(&self, params: ProgressNotificationParam, _context: NotificationContext<RoleClient>) {
        self.progress.lock().unwrap().push(params);
    }
}

/// A server with a mock engine and a client connected to it through an in-memory pipe.
pub struct Harness {
    pub client: RunningService<RoleClient, Recorder>,
    pub server: PlanetCloudgenServer,
    pub backend: Arc<MockBackend>,
    pub progress: Arc<Mutex<Vec<ProgressNotificationParam>>>,
    pub tools: Vec<Tool>,
}

pub async fn connect(config: ServerConfig, backend: MockBackend) -> Harness {
    let backend = Arc::new(backend);
    let dyn_backend: Arc<dyn Backend> = backend.clone();
    let server = PlanetCloudgenServer::new(config, dyn_backend);
    let (server_io, client_io) = tokio::io::duplex(8 << 20);
    let s = server.clone();
    tokio::spawn(async move {
        if let Ok(svc) = s.serve(server_io).await {
            let _ = svc.waiting().await;
        }
    });
    let recorder = Recorder::default();
    let progress = recorder.progress.clone();
    let client = recorder.serve(client_io).await.expect("the client connects");
    let tools = client.list_all_tools().await.expect("tools/list");
    Harness { client, server, backend, progress, tools }
}

pub fn args(v: Value) -> serde_json::Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}

pub fn text_of(r: &CallToolResult) -> String {
    r.content.first().and_then(ContentBlock::as_text).map(|t| t.text.clone()).unwrap_or_default()
}

impl Harness {
    pub async fn call(&self, name: &str, arguments: Value) -> CallToolResult {
        self.client
            .call_tool(CallToolRequestParams::new(name.to_string()).with_arguments(args(arguments)))
            .await
            .unwrap_or_else(|e| panic!("{name}: a protocol error: {e}"))
    }

    pub fn tool(&self, name: &str) -> &Tool {
        self.tools.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("no tool {name}"))
    }

    /// Call a tool that must succeed; check the contract every tool shares and return the result.
    pub async fn good(&self, name: &str, arguments: Value) -> CallToolResult {
        let r = self.call(name, arguments).await;
        assert!(r.is_error != Some(true), "{name} failed: {}", text_of(&r));
        let text = text_of(&r);
        let cap = if name == "explain_model" { 12000 } else { 3000 };
        assert!(!text.is_empty() && text.len() < cap, "{name}: the text is {} bytes", text.len());
        let structured = r.structured_content.as_ref().expect("structured content");
        assert!(structured.is_object());
        let schema = Value::Object(self.tool(name).output_schema.as_ref().expect("an output schema").as_ref().clone());
        if let Err(e) = validate(&schema, &schema, structured, "$") {
            panic!("{name}: the structured result does not match its output schema: {e}\n{structured}");
        }
        r
    }

    /// Call a tool that must fail with a message an AI can act on; return the message.
    pub async fn bad(&self, name: &str, arguments: Value) -> String {
        let r = self.call(name, arguments).await;
        assert_eq!(r.is_error, Some(true), "{name} should have failed: {}", text_of(&r));
        text_of(&r)
    }

    pub async fn good_json(&self, name: &str, arguments: Value) -> Value {
        self.good(name, arguments).await.structured_content.expect("structured content")
    }

    pub async fn read(&self, uri: &str) -> Result<ReadResourceResult, rmcp::ServiceError> {
        self.client.read_resource(ReadResourceRequestParams::new(uri)).await
    }

    pub async fn read_text(&self, uri: &str) -> (String, Option<String>) {
        let r = self.read(uri).await.unwrap_or_else(|e| panic!("{uri}: {e}"));
        match &r.contents[0] {
            ResourceContents::TextResourceContents { text, mime_type, .. } => (text.clone(), mime_type.clone()),
            other => panic!("{uri}: not text: {other:?}"),
        }
    }

    pub async fn read_blob(&self, uri: &str) -> (Vec<u8>, Option<String>) {
        let r = self.read(uri).await.unwrap_or_else(|e| panic!("{uri}: {e}"));
        match &r.contents[0] {
            ResourceContents::BlobResourceContents { blob, mime_type, .. } => {
                (base64::engine::general_purpose::STANDARD.decode(blob).unwrap(), mime_type.clone())
            }
            other => panic!("{uri}: not a blob: {other:?}"),
        }
    }

    /// Call a tool with a progress token, so the server reports its progress; returns the result.
    pub async fn call_with_progress(&self, name: &str, arguments: Value) -> CallToolResult {
        let params = CallToolRequestParams::new(name.to_string()).with_arguments(args(arguments));
        let handle = self
            .client
            .send_cancellable_request(ClientRequest::CallToolRequest(Request::new(params)), PeerRequestOptions::no_options())
            .await
            .expect("the request is sent");
        match handle.await_response().await.expect("a response") {
            ServerResult::CallToolResult(r) => r,
            other => panic!("not a tool result: {other:?}"),
        }
    }

    /// Poll a job until it is final; returns its last structured status.
    pub async fn wait_job(&self, job_id: &str, timeout: Duration) -> Value {
        let end = Instant::now() + timeout;
        loop {
            let st = self.good_json("get_generation", json!({"job_id": job_id})).await;
            if matches!(st["status"].as_str(), Some("done" | "failed" | "cancelled")) {
                return st;
            }
            assert!(Instant::now() < end, "the job did not finish: {st}");
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    }

    /// One small map, saved: the structured summary of its `generate_map` result.
    pub async fn made(&self) -> Value {
        self.good_json("generate_map", json!({"preset": "slow-rotator", "width": 256, "seed": 7})).await
    }
}

/// Poll `cond` every 10 ms for up to `secs` seconds.
pub async fn eventually(secs: u64, mut cond: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < end {
        if cond() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    cond()
}

// ------------------------------------------------------------------------------------------------ a tiny HTTP client

pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: String,
}


impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    /// The JSON-RPC messages of the body, whether it is plain JSON or a stream of server-sent events.
    pub fn messages(&self) -> Vec<Value> {
        let t = self.body.trim();
        if t.starts_with('{') {
            return serde_json::from_str(t).into_iter().collect();
        }
        self.body
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .filter_map(|d| serde_json::from_str::<Value>(d.trim()).ok())
            .collect()
    }
}

pub fn dechunk(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let Some(eol) = raw[i..].windows(2).position(|w| w == b"\r\n") else { break };
        let size = usize::from_str_radix(std::str::from_utf8(&raw[i..i + eol]).unwrap_or("0").split(';').next().unwrap_or("0").trim(), 16).unwrap_or(0);
        i += eol + 2;
        if size == 0 || i + size > raw.len() {
            if size != 0 {
                out.extend_from_slice(&raw[i..]);
            }
            break;
        }
        out.extend_from_slice(&raw[i..i + size]);
        i += size + 2;
    }
    out
}

pub async fn post(port: u16, host: &str, extra: &[(&str, &str)], body: &Value) -> Response {
    let body = body.to_string();
    let mut req = format!(
        "POST /mcp HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in extra {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    req.push_str(&body);
    let mut stream = tokio::net::TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(15), stream.read_to_end(&mut raw)).await; // an event stream that stays open ends here
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").map_or(raw.len(), |p| p + 4);
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let mut lines = head.lines();
    let status: u16 = lines.next().unwrap_or("").split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    let headers: Vec<(String, String)> =
        lines.filter_map(|l| l.split_once(':')).map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).collect();
    let mut bytes = raw[split..].to_vec();
    if headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding") && v.to_lowercase().contains("chunked")) {
        bytes = dechunk(&bytes);
    }
    Response { status, headers, body: String::from_utf8_lossy(&bytes).to_string() }
}

pub fn initialize() -> Value {
    json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}})
}


/// The RGB of one pixel of an RGB PNG.
pub fn rgb_pixel(png_bytes: &[u8], x: usize, y: usize) -> [u8; 3] {
    let mut reader = png::Decoder::new(std::io::Cursor::new(png_bytes)).read_info().expect("a PNG");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("a buffer size")];
    let info = reader.next_frame(&mut buf).expect("a frame");
    assert_eq!(info.color_type, png::ColorType::Rgb);
    let i = (y * info.width as usize + x) * 3;
    [buf[i], buf[i + 1], buf[i + 2]]
}

/// An 8-bit gray coverage map, 2:1, dark except a bright blob around (`lon_deg`, `lat_deg`).
pub fn blob_png(width: usize, lon_deg: f64, lat_deg: f64) -> Vec<u8> {
    let height = width / 2;
    let mut data = vec![0u8; width * height];
    for r in 0..height {
        let lat = 90.0 - (r as f64 + 0.5) * 180.0 / height as f64;
        for c in 0..width {
            let lon = -180.0 + (c as f64 + 0.5) * 360.0 / width as f64;
            let mut dlon = (lon - lon_deg).abs();
            if dlon > 180.0 {
                dlon = 360.0 - dlon;
            }
            if dlon * lat.to_radians().cos().max(0.2) < 12.0 && (lat - lat_deg).abs() < 12.0 {
                data[r * width + c] = 255;
            }
        }
    }
    cloud_map_generator_mcp::render::encode_gray8(width, height, &data)
}
