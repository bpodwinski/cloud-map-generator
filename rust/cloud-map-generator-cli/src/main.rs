//! `cloud-map-generator`: the command line of the Rust port of cloud-map-generator. One binary, five subcommands that replace
//! the Python scripts: `gen` (gen.py), `measure` (measure.py), `bake` (bake.py), `tiles` (bake_tiles.py) and
//! `serve` (ui/server.py: the local web UI, with the compute modes), with the same flags and the same output files.
//!
//! The model is `cloud-map-generator-core`; this crate adds what the core leaves out on purpose: the command line, the
//! container I/O (PNG with the `png` crate, gzip with `flate2`), the bodies file and the JSON sidecars.
//!
//! The random draws are the Rust port's own (`cloud_map_generator_core::rng`), not numpy's: a map of a seed is
//! statistically equivalent to the Python map of that seed, not the same map.

mod cmd_bake;
mod cmd_gen;
mod cmd_measure;
mod cmd_serve;
mod cmd_tiles;
mod refmap;
mod serve;

// The engine-facing layer is the `cloud-map-generator-service` crate (shared with `cloud-map-generator-mcp`); these keep the
// module paths the commands use.
pub(crate) use cloud_map_generator_service::{bodies, fail, imageio, meta, project, pyjson};

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser, Debug)]
#[command(name = "cloudgen", version, about = "Global cloud-coverage maps from a planet's physics (Rust port of cloud-map-generator)")]
struct Cli {
    /// The project folder holding data/ and examples/ (default: $CLOUD_MAP_GENERATOR_ROOT, else found from the working directory).
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    /// Threads of the CPU engine (default: $CLOUD_MAP_GENERATOR_THREADS, else every logical core). The maps are bit-identical for any count.
    #[arg(long, global = true, value_name = "N", value_parser = parse_threads)]
    threads: Option<usize>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Generate one coverage map (gen.py).
    Gen(cmd_gen::GenArgs),
    /// Measure coverage maps against Earth's shipped cloud fraction (measure.py).
    Measure(cmd_measure::MeasureArgs),
    /// Bake the cloud maps of a set of bodies: the list and the library (bake.py).
    Bake(cmd_bake::BakeArgs),
    /// Bake the cell-field detail tile (bake_tiles.py).
    Tiles(cmd_tiles::TilesArgs),
    /// Serve the local web UI with the Rust engine, CPU and GPU (ui/server.py).
    Serve(cmd_serve::ServeArgs),
}

/// `--threads N`: a whole number of at least 1.
fn parse_threads(text: &str) -> Result<usize, String> {
    cloud_map_generator_core::par::parse_threads(text, "--threads")
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    // The thread pool of the CPU engine, fixed once before any map: `--threads`, else CLOUD_MAP_GENERATOR_THREADS, else all cores.
    if let Err(e) = cloud_map_generator_core::par::init_threads(cli.threads) {
        eprintln!("cloudgen: {e}");
        return ExitCode::FAILURE;
    }
    // `--root`, else the CLOUD_MAP_GENERATOR_ROOT environment variable (read here: clap's `env` feature is one more dependency).
    let root = cli.root.or_else(|| std::env::var_os("CLOUD_MAP_GENERATOR_ROOT").map(PathBuf::from));
    let project = project::Project::discover(root);
    let result = match cli.command {
        Command::Gen(a) => cmd_gen::run(a, &project),
        Command::Measure(a) => cmd_measure::run(a, &project),
        Command::Bake(a) => cmd_bake::run(a, &project),
        Command::Tiles(a) => cmd_tiles::run(a, &project),
        Command::Serve(a) => cmd_serve::run(a, &project),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cloudgen: {e}");
            ExitCode::FAILURE
        }
    }
}
