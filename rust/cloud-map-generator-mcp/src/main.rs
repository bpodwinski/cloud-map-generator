//! `cloud-map-generator-mcp`: the MCP server over the real engine (`cloud-map-generator-service`: CPU, and GPU when the machine has
//! one), or over the deterministic mock engine with `--mock`.

use std::sync::Arc;

use cloud_map_generator_mcp::logging::{log, Level};
use cloud_map_generator_mcp::mock::MockBackend;
use cloud_map_generator_mcp::service_backend::ServiceBackend;
use cloud_map_generator_mcp::Backend;

fn main() {
    // `--version` and `--help` are answered by the command line, whatever engine is behind it
    let code = cloud_map_generator_mcp::run_with_factory(
        |cli| -> Arc<dyn Backend> {
            if cli.mock {
                let gpu = std::env::var("PLANET_CLOUDGEN_MOCK_GPU").is_ok_and(|v| !v.is_empty() && v != "0");
                log(
                    Level::Warning,
                    "cloud-map-generator-mcp",
                    format_args!("--mock: the engine behind this server is the MOCK backend: the maps and numbers are fabricated stand-ins, not predictions"),
                );
                Arc::new(if gpu { MockBackend::with_gpu() } else { MockBackend::new() })
            } else {
                ServiceBackend::new()
            }
        },
        std::env::args_os(),
    );
    std::process::exit(code);
}
