//! `cloud-map-generator bake`: bake the cloud maps of a set of bodies (`bake.py`): generation runs here, offline, and a
//! renderer only loads the files. Two products:
//!
//! - LIST: every body of the bodies file whose clouds are water and that has no measured map. Its own physics,
//!   a seed from its key; its land and relief (`surface`) enter the bake when the record has one.
//! - LIBRARY: K seeds per climate class, classes in bins of the Hadley edge. Each class is baked on an
//!   Earth-sized body whose rotation puts its Hadley edge at the class center (`bake::library_body`).
//!
//! Each entry: the high-resolution coverage (R8, rows SOUTH-up, gzip), its 2048 x 1024 box downsample
//! (north-up PNG), the high-cloud share and the cloud type at 2048 (north-up PNGs), the wind (RGB PNG) and a
//! sidecar JSON. Files go to `output/bake/<version>/`, the manifest to `<out>/manifest.json`.

use crate::bodies::load_bodies;
use crate::fail::{Fail, Result};
use crate::imageio::write_file;
use crate::project::Project;
use crate::pyjson::dumps;
use clap::{Args, ValueEnum};
use cloud_map_generator_core::bake::{DEFAULT_LIBRARY_SEEDS, DEFAULT_LIBRARY_WIDTH};
use cloud_map_generator_service::bake::{bake, BakeError, BakeOptions, EntryLog, Observer, Only as BakeOnly};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Only {
    List,
    Library,
}

#[derive(Args, Debug)]
pub struct BakeArgs {
    #[arg(long, default_value = "v1")]
    pub version: String,
    /// Width of the list maps (default: the texel rule capped at 8192).
    #[arg(long)]
    pub list_width: Option<usize>,
    #[arg(long, default_value_t = DEFAULT_LIBRARY_WIDTH)]
    pub library_width: usize,
    /// Seeds per library class.
    #[arg(long, default_value_t = DEFAULT_LIBRARY_SEEDS)]
    pub k: usize,
    #[arg(long, value_enum)]
    pub only: Option<Only>,
    /// Bodies file (default examples/bodies.json).
    #[arg(long)]
    pub bodies: Option<PathBuf>,
    /// Manifest path prefix (default baked/<version>).
    #[arg(long)]
    pub url_prefix: Option<String>,
    /// Manifest file (default <out>/manifest.json).
    #[arg(long)]
    pub manifest: Option<PathBuf>,
    /// Output folder (default output/bake/<version>).
    #[arg(long)]
    pub out: Option<PathBuf>,
}

/// Prints the progress of a bake the way `bake.py` does.
struct Printer;

impl Observer for Printer {
    fn skipped(&mut self, key: &str, reason: &str) {
        println!("  {key:<28} skipped: {reason}");
    }

    fn starting(&mut self, _label: &str, _index: usize, _total: usize) -> bool {
        true
    }

    fn entry(&mut self, log: &EntryLog) {
        let land = match log.land_fraction {
            Some(f) => format!("  land {f:.2}"),
            None => "  no surface".to_string(),
        };
        println!("  {:<28} {}x{}  edge {:5.1} deg{land}  {:6.1} s", log.label, log.width, log.height, log.hadley_edge_deg, log.seconds);
    }

    fn section(&mut self, name: &str) {
        println!("{name}:");
    }
}

pub fn run(a: BakeArgs, project: &Project) -> Result<()> {
    let out = a.out.clone().unwrap_or_else(|| project.output().join("bake").join(&a.version));
    let manifest_path = a.manifest.clone().unwrap_or_else(|| out.join("manifest.json"));
    let bodies = load_bodies(project, a.bodies.as_deref())?;
    let opts = BakeOptions {
        version: a.version.clone(),
        url_prefix: a.url_prefix.clone(),
        list_width: a.list_width,
        library_width: a.library_width,
        k: a.k,
        only: a.only.map(|o| match o {
            Only::List => BakeOnly::List,
            Only::Library => BakeOnly::Library,
        }),
    };
    let baked = match bake(&bodies, &opts, &out, &mut Printer) {
        Ok(b) => b,
        Err(BakeError::Fail(f)) => return Err(f),
        Err(BakeError::Cancelled) => return Err(Fail::new("the bake was cancelled")),
    };
    if a.only.is_none() {
        write_file(&manifest_path, format!("{}
", dumps(&baked.manifest)).as_bytes())?;
        println!("manifest: {}", manifest_path.display());
    } else {
        println!("partial run: manifest not written");
    }
    println!("files: {}", out.display());
    Ok(())
}
