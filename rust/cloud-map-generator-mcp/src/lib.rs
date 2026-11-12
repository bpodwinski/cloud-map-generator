//! cloud-map-generator as a Model Context Protocol server: any MCP client can design, look at, measure and export
//! planet clouds.
//!
//! The server is engine-agnostic. It validates the arguments of ten tools, keeps every path inside a sandbox,
//! runs the heavy work one job at a time and shapes the answers; the maps themselves come from a [`Backend`].
//! The engine is [`service_backend::ServiceBackend`], an adapter over `cloud-map-generator-service` (the layer `cloud-map-generator serve`
//! runs on: CPU always, GPU when an adapter exists). Another engine implements [`Backend`] and calls [`run`]:
//!
//! ```no_run
//! use cloud_map_generator_mcp::service_backend::ServiceBackend;
//!
//! fn main() {
//!     std::process::exit(cloud_map_generator_mcp::run(ServiceBackend::new()));
//! }
//! ```
//!
//! [`mock::MockBackend`] is a deterministic stand-in that fabricates plausible results: the tests use it, and
//! `cloud-map-generator-mcp --mock` serves it.
//!
//! stdout carries the protocol and is never written to by this crate; logs go to stderr.

pub mod args;
pub mod backend;
pub mod config;
pub mod errors;
pub mod http;
pub mod jobs;
pub mod logging;
pub mod mock;
pub mod prompts;
pub mod pyfmt;
pub mod render;
pub mod resources;
pub mod results;
pub mod rules;
pub mod sandbox;
pub mod server;
pub mod service_backend;
pub mod texts;
pub mod tools;

use std::ffi::OsString;
use std::sync::Arc;

use clap::{CommandFactory, Parser};
use rmcp::transport::stdio;
use rmcp::ServiceExt;

pub use backend::{Backend, BackendError};
pub use config::ServerConfig;
pub use server::PlanetCloudgenServer;

/// The version of this server (the workspace version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Serve over stdio in this process, for tests and embedders that build their own transport: returns when the
/// client disconnects.
pub async fn serve_stdio(server: PlanetCloudgenServer) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// The command line of `cloud-map-generator-mcp` over `backend`: parse the flags, serve over stdio (or local HTTP with
/// `--http`) and return the process exit code. A binary is `std::process::exit(cloud_map_generator_mcp::run(backend))`.
pub fn run(backend: Arc<dyn Backend>) -> i32 {
    run_with_args(backend, std::env::args_os())
}

/// [`run`] with explicit arguments (the first is the program name).
pub fn run_with_args(backend: Arc<dyn Backend>, args: impl IntoIterator<Item = OsString>) -> i32 {
    run_with_factory(move |_| backend, args)
}

/// [`run_with_args`] where the backend is chosen after the command line is read (`--mock` picks the stand-in engine):
/// `make` is called once, with the parsed flags, and only when the server really starts (not for `--help` or `--version`).
pub fn run_with_factory(make: impl FnOnce(&config::Cli) -> Arc<dyn Backend>, args: impl IntoIterator<Item = OsString>) -> i32 {
    let cli = match config::Cli::try_parse_from(args) {
        Ok(c) => c,
        Err(e) => e.exit(),
    };
    let widths = backend::WIDTHS;
    let config = match config::config_from_cli(&cli, &widths, &config::process_environ()) {
        Ok(c) => c,
        Err(message) => config::Cli::command().error(clap::error::ErrorKind::ValueValidation, message).exit(),
    };
    if let Some(level) = logging::Level::parse(&config.log_level) {
        logging::set_level(level);
    }
    // The CPU engine's thread pool, fixed once before any map: `--threads`, else CLOUD_MAP_GENERATOR_THREADS, else every logical core.
    if let Err(message) = cloud_map_generator_service::core::par::init_threads(cli.threads) {
        config::Cli::command().error(clap::error::ErrorKind::ValueValidation, message).exit();
    }
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("cloud-map-generator-mcp: cannot start the async runtime: {e}");
            return 1;
        }
    };
    let server = PlanetCloudgenServer::new(config.clone(), make(&cli));
    let result = runtime.block_on(async {
        if config.http {
            http::serve(server.clone(), config.port, |addr| {
                eprintln!(
                    "cloud-map-generator-mcp {VERSION}: streamable HTTP on http://{addr}/mcp (local only); output folder {}",
                    config.out_dir.display()
                );
            })
            .await
            .map_err(|e| e.to_string())
        } else {
            serve_stdio(server.clone()).await.map_err(|e| e.to_string())
        }
    });
    server.close();
    match result {
        Ok(()) => 0,
        Err(message) => {
            eprintln!("cloud-map-generator-mcp: {message}");
            1
        }
    }
}
