//! The streamable HTTP transport: it works on loopback, and a request that carries a foreign Host or Origin never
//! reaches the MCP service (DNS-rebinding guard). The client is a hand-written HTTP/1.1 speaker, so the tests
//! need no HTTP client library.

mod common;

use std::sync::Arc;

use cloud_map_generator_mcp::mock::MockBackend;
use cloud_map_generator_mcp::server::PlanetCloudgenServer;
use common::*;
use serde_json::json;

async fn start() -> (u16, Dirs) {
    let d = Dirs::new();
    let server = PlanetCloudgenServer::new(d.config(), Arc::new(MockBackend::new()));
    let (tx, rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let _ = cloud_map_generator_mcp::http::serve(server, 0, move |addr| {
            let _ = tx.send(addr);
        })
        .await;
    });
    let addr = rx.await.expect("the server binds");
    assert!(addr.ip().is_loopback(), "bound to loopback only: {addr}");
    (addr.port(), d)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_works_locally_and_rejects_foreign_hosts() {
    let (port, _d) = start().await;
    let host = format!("127.0.0.1:{port}");
    // initialize, then ask for the tools and call one: a whole conversation over HTTP
    let init = post(port, &host, &[], &initialize()).await;
    assert!(init.status == 200, "{} {:?}", init.status, init.body);
    let session = init.header("mcp-session-id").map(str::to_string);
    let version = [("MCP-Protocol-Version", "2025-06-18")];
    let mut extra: Vec<(&str, &str)> = version.to_vec();
    if let Some(s) = &session {
        extra.push(("Mcp-Session-Id", s));
    }
    let info = init.messages().into_iter().find(|m| m["id"] == 1).expect("the initialize response");
    assert_eq!(info["result"]["serverInfo"]["name"], "cloud-map-generator");
    let ack = post(port, &host, &extra, &json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).await;
    assert!(ack.status == 202 || ack.status == 200, "{}", ack.status);
    let list = post(port, &host, &extra, &json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})).await;
    let tools = list.messages().into_iter().find(|m| m["id"] == 2).expect("the tool list");
    assert_eq!(tools["result"]["tools"].as_array().unwrap().len(), 10);
    let call = post(
        port,
        &host,
        &extra,
        &json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "derive_physics", "arguments": {"preset": "earth"}}}),
    )
    .await;
    let result = call.messages().into_iter().find(|m| m["id"] == 3).expect("the tool result");
    assert_eq!(result["result"]["structuredContent"]["derived"]["has_eddies"], true);

    // the rebinding guard: only loopback names, only local pages
    for ok in [format!("127.0.0.1:{port}"), format!("localhost:{port}"), format!("[::1]:{port}")] {
        let r = post(port, &ok, &[], &initialize()).await;
        assert!(r.status != 403 && r.status != 421, "{ok}: {}", r.status);
    }
    assert_eq!(post(port, &format!("evil.example.com:{port}"), &[], &initialize()).await.status, 421);
    assert_eq!(post(port, "attacker.test", &[], &initialize()).await.status, 421);
    assert_eq!(post(port, &host, &[("Origin", "http://evil.example.com")], &initialize()).await.status, 403);
    assert_eq!(post(port, &host, &[("Origin", "null")], &initialize()).await.status, 403);
    let local = post(port, &host, &[("Origin", &format!("http://localhost:{port}"))], &initialize()).await;
    assert!(local.status != 403 && local.status != 421, "{}", local.status);
}
