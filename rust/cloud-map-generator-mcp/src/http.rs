//! The streamable HTTP transport, bound to 127.0.0.1 only, with a guard against DNS rebinding: a request whose
//! `Host` is not a loopback name is refused with 421, one whose `Origin` is not a local page with 403, before
//! the MCP service sees it. Without this a web page on another origin could drive a server on the developer's
//! machine through the browser.

use std::net::SocketAddr;

use axum::extract::Request;
use axum::http::{header, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::Router;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};

use crate::config::DEFAULT_HOST;
use crate::server::PlanetCloudgenServer;

/// `127.0.0.1`, `localhost` or `[::1]`, with or without a numeric port.
pub fn allowed_host(authority: &str) -> bool {
    let a = authority.trim();
    let (host, port) = if let Some(rest) = a.strip_prefix('[') {
        match rest.split_once(']') {
            Some((h, tail)) => (format!("[{h}]"), tail.strip_prefix(':').map(str::to_string)),
            None => return false,
        }
    } else {
        match a.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), Some(p.to_string())),
            None => (a.to_string(), None),
        }
    };
    let host_ok = matches!(host.to_lowercase().as_str(), "127.0.0.1" | "localhost" | "[::1]");
    let port_ok = port.is_none_or(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()));
    host_ok && port_ok
}

/// `http://` followed by an allowed host.
pub fn allowed_origin(origin: &str) -> bool {
    origin.trim().strip_prefix("http://").is_some_and(|rest| !rest.contains('/') && allowed_host(rest))
}

async fn guard(req: Request, next: Next) -> Response {
    let host = req.headers().get(header::HOST).and_then(|h| h.to_str().ok()).map(str::to_string);
    if !host.as_deref().is_some_and(allowed_host) {
        return (StatusCode::MISDIRECTED_REQUEST, "Invalid Host header").into_response();
    }
    if let Some(origin) = req.headers().get(header::ORIGIN) {
        if !origin.to_str().ok().is_some_and(allowed_origin) {
            return (StatusCode::FORBIDDEN, "Invalid Origin header").into_response();
        }
    }
    next.run(req).await
}

/// The router: the MCP endpoint at `/mcp` behind the guard.
pub fn router(server: PlanetCloudgenServer) -> Router {
    let config = StreamableHttpServerConfig::default()
        .with_allowed_hosts(["localhost", "127.0.0.1", "[::1]"])
        .with_allowed_origins(["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]);
    let service = StreamableHttpService::new(move || Ok(server.clone()), LocalSessionManager::default().into(), config);
    Router::new().nest_service("/mcp", service).layer(middleware::from_fn(guard))
}

/// Serve on `127.0.0.1:port` until Ctrl-C. `port` 0 picks a free one; `on_bound` learns the address.
pub async fn serve(
    server: PlanetCloudgenServer,
    port: u16,
    on_bound: impl FnOnce(SocketAddr),
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind((DEFAULT_HOST, port)).await?;
    on_bound(listener.local_addr()?);
    axum::serve(listener, router(server))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loopback_hosts_and_origins_pass() {
        for ok in ["127.0.0.1:8766", "localhost:19000", "[::1]:8766", "LOCALHOST:1", "127.0.0.1", "localhost"] {
            assert!(allowed_host(ok), "{ok}");
        }
        for bad in [
            "evil.example.com:8766", "attacker.test", "127.0.0.1.evil.test:1", "localhost.evil.test", "0.0.0.0:8766", "[::2]:1",
            "127.0.0.1:abc", "", "[::1", "evil.test:127.0.0.1",
        ] {
            assert!(!allowed_host(bad), "{bad}");
        }
        assert!(allowed_origin("http://127.0.0.1:3000") && allowed_origin("http://localhost:5173") && allowed_origin("http://[::1]:9"));
        for bad in ["http://evil.example.com", "https://localhost:3000", "null", "http://localhost.evil.test:1", "http://localhost:1/x"] {
            assert!(!allowed_origin(bad), "{bad}");
        }
    }
}
