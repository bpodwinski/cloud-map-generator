//! The engine-facing layer of cloud-map-generator, shared by its two front ends: `cloud-map-generator serve` (the local web UI's
//! HTTP API) and `cloud-map-generator-mcp` (the Model Context Protocol server). Neither speaks HTTP or MCP here; this
//! crate knows bodies, maps and files.
//!
//! - [`limits`], [`physics`]: the ranges inputs must respect, the presets, the numbers a body implies and the
//!   refusals about a body.
//! - [`engine`]: the compute modes (`auto`, `cpu`, `gpu`, `hybrid`), the [`engine::Engine`] trait, the CPU engine,
//!   the machine's capabilities. With the `gpu` cargo feature (default on) [`gpu_probe`], [`gpu_slot`] and
//!   [`gpu_engine`] add the wgpu pipeline.
//! - [`generate`]: one generation over an engine, with its encodings, its metrics against Earth and its zonal
//!   profile; [`reference`] is Earth's reference map; [`synthetic`] the made-up world of `surface = synthetic`.
//! - [`bake`]: the bake of a catalog of bodies to files.
//! - [`pyjson`], [`json`], [`pyfmt`], [`images`], [`imageio`], [`meta`], [`bodies`], [`project`], [`fail`]: the
//!   encodings and file formats of the Python tools, kept byte-compatible.
//!
//! The core stays dependency-light: PNG and gzip containers, JSON and the wgpu pipeline are here.

pub mod bake;
pub mod bands;
pub mod bodies;
pub mod engine;
pub mod estimates;
pub mod fail;
pub mod generate;
#[cfg(feature = "gpu")]
pub mod gpu_engine;
#[cfg(feature = "gpu")]
pub mod gpu_probe;
#[cfg(feature = "gpu")]
pub mod gpu_slot;
pub mod imageio;
pub mod images;
pub mod large;
pub mod json;
pub mod limits;
pub mod machine;
pub mod meta;
pub mod physics;
pub mod project;
pub mod pyfmt;
pub mod pyjson;
pub mod reference;
pub mod synthetic;

/// Earth's reference map (`data/earth_cloud_fraction_bluemarble_2048.png`, NASA Blue Marble's cloud fraction),
/// embedded so that every binary carries it.
pub const REFERENCE_PNG: &[u8] = include_bytes!("../../../data/earth_cloud_fraction_bluemarble_2048.png");

pub use cloud_map_generator_core as core;
