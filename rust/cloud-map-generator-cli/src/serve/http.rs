//! The transport and the routes of `cloud-map-generator serve`.
//!
//! [`route`] is the whole server as a function from a plain request to a plain response (so that the tests call
//! it without a socket); [`serve`] puts it behind a small HTTP/1.1 transport on `std::net`, one thread per connection. The behaviors are
//! `ui/server.py`'s: JSON error pages, `Host` and `Origin` checks (a foreign one is a 403), 405 for a known path
//! with the wrong method and 404 otherwise, 501 for a method the server does not speak, HEAD as GET without the
//! body, `Cache-Control: no-store` and `X-Content-Type-Options: nosniff` on everything, downloads with
//! `Content-Disposition: attachment`, and a `/static/` that cannot leave its folder.

use super::generate::run_generation;
use super::json::{compact, num};
use super::physics::{build_presets, check_generatable, derive, limits_json, preset_json};
use super::pyfmt::unquote;
use super::request::{bad, parse_derive, parse_generate_text, ApiError, MAX_CYCLONES_PER_HEMISPHERE, MAX_REQUEST_BYTES};
use super::store::RESULT_FILES;
use super::{assets, estimate_seconds, AppState};
use crate::pyjson::J;
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, TryLockError};
use std::time::{Duration, Instant};

/// What a request carries as a body, decided before any route runs (`Content-Length` and its size).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Body {
    /// The body (empty for a request that has none).
    Bytes(Vec<u8>),
    /// A POST with no `Content-Length`.
    Missing,
    /// A `Content-Length` that is not a number.
    BadLength,
    /// A `Content-Length` over [`MAX_REQUEST_BYTES`].
    TooLarge,
}

/// A request, as the routes see it. Header names are lower case.
#[derive(Clone, Debug)]
pub struct HttpRequest {
    pub method: String,
    pub target: String,
    pub headers: Vec<(String, String)>,
    pub body: Body,
}

impl HttpRequest {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }
}

/// A response, before the transport adds `Content-Length` and the `Server` header.
#[derive(Clone, Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    #[cfg(test)]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

const JSON_TYPE: &str = "application/json; charset=utf-8";

fn send(status: u16, content_type: &str, body: Vec<u8>, extra: &[(&str, String)]) -> HttpResponse {
    let mut headers = vec![
        ("Content-Type".to_string(), content_type.to_string()),
        ("Cache-Control".to_string(), "no-store".to_string()),
        ("X-Content-Type-Options".to_string(), "nosniff".to_string()),
    ];
    headers.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
    HttpResponse { status, headers, body }
}

fn json_response(status: u16, payload: &J) -> HttpResponse {
    send(status, JSON_TYPE, compact(payload).into_bytes(), &[])
}

fn error_response(status: u16, message: &str) -> HttpResponse {
    json_response(status, &J::obj([("error", J::s(message))]))
}

/// Content types by extension (`CONTENT_TYPES` of `ui/server.py`).
fn content_type(path: &str) -> &'static str {
    let ext = path.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "html" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "json" | "map" => JSON_TYPE,
        "txt" => "text/plain; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        _ => "application/octet-stream",
    }
}

/// The path of a request target: no query, no fragment.
fn path_of(target: &str) -> &str {
    // The absolute form (`http://host:port/path`, what a proxy is sent): the path after the authority.
    let mut t = target;
    if let Some(i) = t.find("://") {
        if i > 0 && t[..i].bytes().all(|b| b.is_ascii_alphabetic()) {
            let rest = &t[i + 3..];
            t = rest.find(['/', '?', '#']).map_or("/", |j| &rest[j..]);
        }
    }
    let end = t.find(['?', '#']).unwrap_or(t.len());
    let t = if end == 0 { "/" } else { &t[..end] };
    // `//path` is `/path` (Python's http.server collapses leading slashes, so as not to read a host in the path).
    let bare = t.trim_start_matches('/');
    if t.starts_with("//") {
        &t[t.len() - bare.len() - 1..]
    } else {
        t
    }
}

/// The `Origin` a page sends is `http://<host>:<port>`: the netloc of it.
fn origin_netloc(origin: &str) -> Option<String> {
    // The scheme is case-insensitive (`Http://127.0.0.1:8765` is the page's own origin).
    let rest = origin.get(..7).filter(|scheme| scheme.eq_ignore_ascii_case("http://")).map(|_| &origin[7..])?;
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    Some(rest[..end].to_ascii_lowercase())
}

/// Only this machine's own page may call the API: a foreign `Host` (DNS rebinding) or `Origin` (a page on
/// another site posting here) is refused. A request with neither header (`curl`) passes.
fn check_origin(state: &AppState, req: &HttpRequest) -> Result<(), ApiError> {
    if let Some(host) = req.header("host") {
        if !state.allowed_hosts.iter().any(|h| *h == host.trim().to_ascii_lowercase()) {
            return Err(ApiError::new(403, "unexpected Host header: this server answers on 127.0.0.1 and localhost only"));
        }
    }
    if let Some(origin) = req.header("origin") {
        if !origin_netloc(origin).is_some_and(|n| state.allowed_hosts.contains(&n)) {
            return Err(ApiError::new(403, "foreign Origin: only this server's own page may call its API"));
        }
    }
    Ok(())
}

/// `result id` and `name` of `/api/result/<12 hex>/<name>`.
fn result_route(path: &str) -> Option<(&str, &str)> {
    let rest = path.strip_prefix("/api/result/")?;
    let (id, name) = rest.split_once('/')?;
    let id_ok = id.len() == 12 && id.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    let name_ok = !name.is_empty() && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.');
    (id_ok && name_ok).then_some((id, name))
}

/// A file of the static folder (or of the embedded page) for the part of `/static/<rel>` after the slash: `rel` is
/// percent-decoded, and refused when it is empty, holds a NUL, a backslash or a colon, starts with a slash, or has a
/// `..` segment; a file that resolves outside the folder (a symlink) is not served either.
fn static_bytes(state: &AppState, rel: &str) -> Option<Vec<u8>> {
    let rel = unquote(rel);
    if rel.is_empty() || rel.contains(['\0', '\\', ':']) || rel.starts_with('/') || rel.split('/').any(|s| s == "..") {
        return None;
    }
    match &state.static_dir {
        None => assets::static_file(&rel).map(<[u8]>::to_vec),
        Some(root) => {
            let candidate: PathBuf = std::fs::canonicalize(root.join(&rel)).ok()?;
            if !candidate.starts_with(root) || !candidate.is_file() {
                return None;
            }
            std::fs::read(&candidate).ok()
        }
    }
}

fn static_response(state: &AppState, rel: &str) -> Result<HttpResponse, ApiError> {
    match static_bytes(state, rel) {
        Some(bytes) => Ok(send(200, content_type(&unquote(rel)), bytes, &[])),
        None if rel == "index.html" => Err(ApiError::new(404, "ui/static/index.html not found")),
        None => Err(ApiError::new(404, format!("not found: /static/{rel}"))),
    }
}

/// `GET /api/defaults`.
pub fn defaults_json(state: &AppState) -> J {
    J::obj([
        ("version", J::s(env!("CARGO_PKG_VERSION"))),
        ("backend", J::s("rust")),
        ("presets", J::Arr(build_presets(&state.project).iter().map(preset_json).collect())),
        ("widths", J::Arr(state.offered_widths().iter().map(|&w| J::Int(w as i64)).collect())),
        ("limits", limits_json()),
        ("max_cyclones_per_hemisphere", J::Int(MAX_CYCLONES_PER_HEMISPHERE as i64)),
        ("estimate_seconds", J::obj(estimate_seconds(&state.offered_widths()).into_iter().map(|(k, s)| (k, num(s))))),
        ("compute", state.runtime.capabilities_json()),
    ])
}

fn get(state: &AppState, path: &str) -> Result<HttpResponse, ApiError> {
    if path == "/" {
        return static_response(state, "index.html");
    }
    if let Some(rel) = path.strip_prefix("/static/") {
        return static_response(state, rel);
    }
    match path {
        "/favicon.ico" => match static_bytes(state, "favicon.ico") {
            Some(bytes) => Ok(send(200, "image/x-icon", bytes, &[])),
            None => Ok(send(204, "image/x-icon", Vec::new(), &[])),
        },
        // The API described (OpenAPI 3.1) and browsable (Swagger UI): files of ui/static/ at short paths.
        "/openapi.json" => static_response(state, "openapi.json"),
        "/docs" => static_response(state, "docs.html"),
        "/api/defaults" => Ok(json_response(200, &defaults_json(state))),
        "/api/reference/coverage.png" => Ok(send(200, "image/png", state.reference_png().to_vec(), &[])),
        _ => {
            let (id, name) = result_route(path).expect("a known GET route");
            let gone = || ApiError::new(404, format!("no such result (results are kept in memory, the last {}): generate again", state.results.keep));
            let item = state.results.get(id).ok_or_else(gone)?;
            let (key, ctype, download) = RESULT_FILES.iter().find(|(n, _, _)| *n == name).ok_or_else(gone)?;
            let bytes = item.files.get(key).cloned().ok_or_else(gone)?;
            let mut extra = Vec::new();
            if *download {
                extra.push(("Content-Disposition", format!("attachment; filename=\"{}\"", item.names.get(key).map_or("", String::as_str))));
            }
            Ok(send(200, ctype, bytes, &extra))
        }
    }
}

fn read_json(body: &Body) -> Result<(Value, &[u8]), ApiError> {
    match body {
        Body::Missing => Err(bad("Content-Length is required (a JSON body)")),
        Body::BadLength => Err(bad("Content-Length is not a number")),
        Body::TooLarge => Err(bad(format!("request body must be at most {MAX_REQUEST_BYTES} bytes"))),
        Body::Bytes(raw) => serde_json::from_slice(raw).map(|v| (v, raw.as_slice())).map_err(|e| bad(format!("invalid JSON: {e}"))),
    }
}

fn post(state: &AppState, path: &str, body: &Body) -> Result<HttpResponse, ApiError> {
    if path == "/api/generate/cancel" {
        // No body: the page asks the running generation to stop. Only a run in progress is asked (the lock is held),
        // so a late click cannot cancel the next one.
        let running = matches!(state.gen_lock.try_lock(), Err(TryLockError::WouldBlock));
        if running {
            state.cancel.store(true, Ordering::Relaxed);
        }
        return Ok(json_response(200, &J::obj([("cancelled", J::Bool(running))])));
    }
    let (payload, raw) = read_json(body)?;
    if path == "/api/derive" {
        let body = parse_derive(&payload)?;
        return Ok(json_response(200, &derive(&body, &state.offered_widths())));
    }
    let req = parse_generate_text(&payload, raw, &state.offered_widths(), &state.runtime)?;
    check_generatable(&req.body, req.options.experiment).map_err(bad)?;
    let guard = match state.gen_lock.try_lock() {
        Ok(g) => g,
        // A generation that panicked poisoned the lock; the lock itself is fine.
        Err(TryLockError::Poisoned(p)) => p.into_inner(),
        Err(TryLockError::WouldBlock) => return Err(ApiError::new(429, "a generation is already running")),
    };
    state.cancel.store(false, Ordering::Relaxed);
    let result = run_generation(state, &req);
    drop(guard);
    Ok(json_response(200, &result?))
}

fn dispatch(state: &AppState, req: &HttpRequest) -> Result<HttpResponse, ApiError> {
    let method = req.method.to_ascii_uppercase();
    if !matches!(method.as_str(), "GET" | "HEAD" | "POST") {
        return Err(ApiError::new(501, format!("Unsupported method ('{}')", req.method)));
    }
    check_origin(state, req)?;
    let method = if method == "HEAD" { "GET" } else { method.as_str() };
    let path = path_of(&req.target);
    let known_get = matches!(path, "/" | "/favicon.ico" | "/docs" | "/openapi.json" | "/api/defaults" | "/api/reference/coverage.png") || path.starts_with("/static/") || result_route(path).is_some();
    let known_post = matches!(path, "/api/derive" | "/api/generate" | "/api/generate/cancel");
    if method == "GET" && known_get {
        return get(state, path);
    }
    if method == "POST" && known_post {
        return post(state, path, &req.body);
    }
    if known_get || known_post {
        return Err(ApiError::new(405, format!("{method} not allowed on {path}")));
    }
    Err(ApiError::new(404, format!("not found: {path}")))
}

/// The server as a function: a request in, a response out. A panic inside a route is a 500 (the lock it held is
/// released), never a dead connection.
pub fn route(state: &AppState, req: &HttpRequest) -> HttpResponse {
    match catch_unwind(AssertUnwindSafe(|| dispatch(state, req))) {
        Ok(Ok(response)) => response,
        Ok(Err(e)) => error_response(e.status, &e.message),
        Err(panic) => {
            let what = panic.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| panic.downcast_ref::<String>().cloned()).unwrap_or_else(|| "unknown".to_string());
            state.log(&format!("panic in {} {}: {what}", req.method, req.target));
            error_response(500, &format!("internal error: {what}"))
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// The socket
//
// The transport is written here, on `std::net`, and not taken from a crate: the HTTP server crate this module used
// (tiny_http 0.12) drops a request whose declared body was not read by allocating a buffer of the REMAINING declared
// length, so `Content-Length: 99999999999` and no body aborted the whole process ("memory allocation of ... bytes
// failed"). Here a declared length over the limit is never read, never allocated, and answered with the JSON 400 of
// `ui/server.py`; the connection is then closed (its stream is out of step) after a bounded drain, so that the
// client reads the answer instead of a reset.

/// The longest request head (request line and headers) accepted, bytes.
const MAX_HEAD_BYTES: usize = 64 * 1024;
/// How long a connection may sit without sending, and how long a send may take.
const IO_TIMEOUT: Duration = Duration::from_secs(60);
/// How much a refused request's unread body is drained, bytes, and for how long, before the connection closes.
const DRAIN_BYTES: u64 = 1 << 20;
const DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// The reason phrase of a status the server sends.
fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        _ => "Status",
    }
}

/// `Date:` in the format of RFC 9110 (`Sun, 06 Nov 1994 08:49:37 GMT`), from the system clock.
fn http_date() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    http_date_of(secs)
}

fn http_date_of(secs: u64) -> String {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // The civil date from the days since 1970-01-01 (the algorithm of Howard Hinnant's date library).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    const WEEKDAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    format!(
        "{}, {day:02} {} {year} {:02}:{:02}:{:02} GMT",
        WEEKDAYS[days.rem_euclid(7) as usize],
        MONTHS[(month - 1) as usize],
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// What a client sent before its body: the request line and the headers (names lower case).
struct Head {
    method: String,
    target: String,
    /// `true` for HTTP/1.1 and later (persistent connections by default).
    http11: bool,
    headers: Vec<(String, String)>,
}

/// Why a head could not be read.
enum HeadError {
    /// The client closed or went quiet before a request: nothing to answer.
    Gone,
    /// Not a request, or one too large: the status to answer with, then close.
    Refused(u16, &'static str),
}

/// One line of the head into `line`, counting what has been read against [`MAX_HEAD_BYTES`].
fn head_line(reader: &mut BufReader<TcpStream>, line: &mut Vec<u8>, used: &mut usize) -> Result<(), HeadError> {
    line.clear();
    let room = (MAX_HEAD_BYTES - *used) as u64;
    let n = reader.by_ref().take(room).read_until(b'\n', line).map_err(|_| HeadError::Gone)?;
    *used += n;
    if n == 0 {
        return Err(HeadError::Gone);
    }
    if !line.ends_with(b"\n") {
        return Err(if *used >= MAX_HEAD_BYTES { HeadError::Refused(431, "request headers are too large") } else { HeadError::Gone });
    }
    Ok(())
}

/// Reads the request line and the headers; at most [`MAX_HEAD_BYTES`] are read, a line at a time.
fn read_head(reader: &mut BufReader<TcpStream>) -> Result<Head, HeadError> {
    let (mut used, mut line) = (0usize, Vec::new());
    // Blank lines before a request line are tolerated (RFC 9112).
    let request_line = loop {
        head_line(reader, &mut line, &mut used)?;
        let text = String::from_utf8_lossy(&line).trim_end().to_string();
        if !text.is_empty() {
            break text;
        }
    };
    let mut parts = request_line.split_whitespace();
    let (Some(method), Some(target), Some(version), None) = (parts.next(), parts.next(), parts.next(), parts.next()) else {
        return Err(HeadError::Refused(400, "bad request line"));
    };
    if !version.starts_with("HTTP/") {
        return Err(HeadError::Refused(400, "bad request line"));
    }
    let http11 = !matches!(version, "HTTP/0.9" | "HTTP/1.0");
    let mut headers = Vec::new();
    loop {
        head_line(reader, &mut line, &mut used)?;
        let text = String::from_utf8_lossy(&line).trim_end().to_string();
        if text.is_empty() {
            break;
        }
        let Some((name, value)) = text.split_once(':') else {
            return Err(HeadError::Refused(400, "bad header line"));
        };
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
    }
    Ok(Head { method: method.to_string(), target: target.to_string(), http11, headers })
}

/// The body of a request, as far as the declared length and the limit allow, and whether the stream is still in
/// step with the client (an unread body is not: the connection must close after the answer).
fn read_body(reader: &mut BufReader<TcpStream>, writer: &mut TcpStream, head: &Head) -> (Body, bool) {
    let header = |name: &str| head.headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str());
    let is_post = head.method.eq_ignore_ascii_case("POST");
    let Some(length) = header("content-length") else {
        // No length: a POST has none to read (a chunked one is refused like a missing length, as Python has no chunked
        // reader either), and its stream is out of step if it sent a body.
        let chunked = header("transfer-encoding").is_some();
        return if is_post { (Body::Missing, !chunked) } else { (Body::Bytes(Vec::new()), true) };
    };
    // Digits only: `+5`, `1e3`, `0x10`, `5 6` and an empty value are not lengths. A number past usize is still a
    // number, and far over the limit.
    if length.is_empty() || !length.bytes().all(|b| b.is_ascii_digit()) {
        // Python's int() takes a negative number and answers it as a body over the limit: so does the contract.
        let negative = length.strip_prefix('-').is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()));
        return if negative { (Body::TooLarge, false) } else { (Body::BadLength, false) };
    }
    let declared = length.parse::<usize>().unwrap_or(usize::MAX);
    if declared > MAX_REQUEST_BYTES {
        return (Body::TooLarge, false);
    }
    if declared > 0 && head.headers.iter().any(|(k, v)| k == "expect" && v.eq_ignore_ascii_case("100-continue")) {
        let _ = writer.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
    }
    let mut raw = vec![0u8; declared];
    match reader.read_exact(&mut raw) {
        Ok(()) => (if is_post { Body::Bytes(raw) } else { Body::Bytes(Vec::new()) }, true),
        // The client left before it sent what it declared.
        Err(_) => (Body::Missing, false),
    }
}

/// Writes a response (status line, headers, body unless `head_only`); `close` says whether the connection ends.
fn write_response(stream: &mut TcpStream, response: &HttpResponse, http11: bool, head_only: bool, close: bool) -> std::io::Result<()> {
    let mut head = format!("{} {} {}\r\n", if http11 { "HTTP/1.1" } else { "HTTP/1.0" }, response.status, reason(response.status));
    for (name, value) in &response.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    // Always a Content-Length, never chunked: a download's size is visible, and a HEAD reports the body it omits.
    head.push_str(&format!("Server: cloudgen-ui\r\nDate: {}\r\nContent-Length: {}\r\n", http_date(), response.body.len()));
    head.push_str(if close {
        "Connection: close\r\n\r\n"
    } else if http11 {
        "\r\n"
    } else {
        "Connection: keep-alive\r\n\r\n"
    });
    stream.write_all(head.as_bytes())?;
    if !head_only && response.status != 204 {
        stream.write_all(&response.body)?;
    }
    stream.flush()
}

/// Ends a connection after an answer to a request whose body was not (fully) read: the write side closes, what the
/// client is still sending is drained for a moment (closing on unread data would reset the connection and the client
/// would lose the answer), and the socket closes.
fn close_after_refusal(stream: &TcpStream, mut reader: BufReader<TcpStream>) {
    let _ = stream.shutdown(Shutdown::Write);
    let _ = stream.set_read_timeout(Some(DRAIN_TIMEOUT));
    let _ = std::io::copy(&mut reader.by_ref().take(DRAIN_BYTES), &mut std::io::sink());
}

fn serve_connection(state: &AppState, stream: TcpStream) {
    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
    let _ = stream.set_nodelay(true);
    let Ok(read_half) = stream.try_clone() else { return };
    let Ok(mut writer) = stream.try_clone() else { return };
    let mut reader = BufReader::new(read_half);
    loop {
        let t0 = Instant::now();
        let head = match read_head(&mut reader) {
            Ok(head) => head,
            Err(HeadError::Gone) => return,
            Err(HeadError::Refused(status, message)) => {
                let _ = write_response(&mut writer, &error_response(status, message), true, false, true);
                close_after_refusal(&stream, reader);
                return;
            }
        };
        let (body, in_step) = read_body(&mut reader, &mut writer, &head);
        let wants_close = head.headers.iter().any(|(k, v)| k == "connection" && v.to_ascii_lowercase().contains("close"));
        let keep_alive = head.http11 && !wants_close && in_step;
        let head_only = head.method.eq_ignore_ascii_case("HEAD");
        let (method, target, http11) = (head.method.clone(), head.target.clone(), head.http11);
        let response = route(state, &HttpRequest { method: method.clone(), target: target.clone(), headers: head.headers, body });
        state.log(&format!("[ui] {method} {target} -> {} ({:.0} ms)", response.status, t0.elapsed().as_secs_f64() * 1000.0));
        // A client that went away is not an error worth a line.
        if write_response(&mut writer, &response, http11, head_only, !keep_alive).is_err() {
            return;
        }
        if !keep_alive {
            if in_step {
                let _ = stream.shutdown(Shutdown::Both);
            } else {
                close_after_refusal(&stream, reader);
            }
            return;
        }
    }
}

/// Serves requests until the process ends: one thread per connection, so that a generation does not block the rest
/// of the API (the 429 for a second one is the generation lock's, not the transport's).
pub fn serve(state: Arc<AppState>, listener: TcpListener) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let state = Arc::clone(&state);
        std::thread::spawn(move || serve_connection(&state, stream));
    }
}

#[cfg(test)]
#[path = "http_tests.rs"]
mod tests;
