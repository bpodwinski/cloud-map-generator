//! `cloud-map-generator serve`: the local web server of the cloud map generator UI, in Rust.
//!
//! The same one JSON API as `ui/server.py` (documented in `ui/README.md`), at the same paths, with the same
//! JSON and status codes, so that the page in `ui/static/` works against it unchanged: `GET /api/defaults`,
//! `POST /api/derive`, `POST /api/generate`, `GET /api/result/<id>/<name>`, `GET /api/reference/coverage.png`,
//! `GET /` and `GET /static/*`. One generation at a time (429 when busy), the last 8 results kept in memory,
//! `127.0.0.1` only, `Host` and `Origin` checked, no CORS.
//!
//! The page and Earth's reference map are embedded in the binary (`build.rs`); `--static-dir` and `--reference`
//! read them from disk instead.
//!
//! Layout: `http` is the transport and the routes, `request` the validation of what clients send, `physics`
//! the presets and the numbers derived from a body, `generate` one generation, `engine` where the work runs
//! (the compute modes and their capabilities; `gpu_engine` is the GPU pipeline behind the `Engine` trait, `gpu_slot`
//! builds it from the device `gpu_probe` found), `store` the kept
//! results, `images`, `json` and `pyfmt` the encodings and the Python behaviors the messages mimic.

pub mod assets;
pub mod generate;
pub mod http;
pub mod request;
pub mod store;

// The engine, the physics, the encodings and the reference map are the service crate's.
pub use cloud_map_generator_service::estimates::estimate_seconds;
pub use cloud_map_generator_service::{engine, json, physics, pyfmt, reference};

use crate::project::Project;
use engine::Runtime;
use reference::Reference;
use request::ApiError;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use store::ResultStore;

/// The results kept in memory by default.
pub const DEFAULT_KEEP: usize = 8;

/// What `serve` is started with.
pub struct Config {
    /// The map widths the API accepts, in the order the page lists them.
    pub widths: Vec<usize>,
    /// How many results are kept.
    pub keep: usize,
    /// A folder to serve the page from instead of the embedded one.
    pub static_dir: Option<PathBuf>,
    /// Earth's reference map as a PNG file, instead of the embedded one.
    pub reference: Option<PathBuf>,
    /// No log lines on stderr.
    pub quiet: bool,
}

/// The state every request thread shares.
pub struct AppState {
    pub widths: Vec<usize>,
    /// The folder `/static/` is served from, resolved (`None`: the embedded files).
    pub static_dir: Option<PathBuf>,
    pub runtime: Runtime,
    pub results: ResultStore,
    /// Held while a generation runs: a second one gets a 429.
    pub gen_lock: Mutex<()>,
    pub project: Project,
    /// `Host` values this server answers to, lower case.
    pub allowed_hosts: Vec<String>,
    reference_png: Vec<u8>,
    reference: OnceLock<Result<Reference, String>>,
    started: Instant,
    quiet: bool,
}

impl AppState {
    /// The widths offered now: the configured ones, the large ones (8192 and up) only when the machine
    /// can write them (free disk and memory; see `cloud_map_generator_service::machine`).
    pub fn offered_widths(&self) -> Vec<usize> {
        let large_from = cloud_map_generator_service::large::LARGE_FROM_WIDTH;
        let wants_large = self.widths.iter().any(|&w| w >= large_from);
        let large = if wants_large { cloud_map_generator_service::machine::large_widths(&cloud_map_generator_service::machine::snapshot()).0 } else { Vec::new() };
        self.widths.iter().copied().filter(|&w| w < large_from || large.contains(&w)).collect()
    }

    /// The state of a server listening on `port`. Fails when the reference map or the static folder cannot be read.
    pub fn new(config: Config, runtime: Runtime, project: Project, port: u16) -> Result<AppState, String> {
        let reference_png = match &config.reference {
            Some(path) => std::fs::read(path).map_err(|e| format!("--reference {}: {e}", path.display()))?,
            None => cloud_map_generator_service::REFERENCE_PNG.to_vec(),
        };
        let static_dir = match &config.static_dir {
            Some(dir) => Some(std::fs::canonicalize(dir).map_err(|e| format!("--static-dir {}: {e}", dir.display()))?),
            None => None,
        };
        Ok(AppState {
            widths: config.widths,
            static_dir,
            runtime,
            results: ResultStore::new(config.keep),
            gen_lock: Mutex::new(()),
            project,
            allowed_hosts: ["127.0.0.1", "localhost", "[::1]"].iter().map(|h| format!("{h}:{port}")).collect(),
            reference_png,
            reference: OnceLock::new(),
            started: Instant::now(),
            quiet: config.quiet,
        })
    }

    /// The Earth reference and its measures, computed on first use (the server asks for them at start-up, on a
    /// thread, so that the first generation does not pay).
    pub fn reference(&self) -> Result<&Reference, ApiError> {
        self.reference
            .get_or_init(|| Reference::from_png(&self.reference_png))
            .as_ref()
            .map_err(|e| ApiError::new(500, e.clone()))
    }

    /// The bytes of the reference PNG the page shows.
    pub fn reference_png(&self) -> &[u8] {
        &self.reference_png
    }

    /// One line on stderr, stamped with the seconds since start-up.
    pub fn log(&self, line: &str) {
        if !self.quiet {
            eprintln!("[{:8.2}] {line}", self.started.elapsed().as_secs_f64());
        }
    }
}

