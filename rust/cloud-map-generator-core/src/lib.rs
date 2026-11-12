//! cloud-map-generator-core: the Rust port of the Python generator (`../../cloudgen/`), checked against the golden
//! cases (`../../golden/`). Each module mirrors one Python module and keeps its function names.
//!
//! Conventions: see `../README.md`.

pub mod par;
pub mod rng;
pub mod sphere;
pub mod noise;
pub mod climate;
pub mod flow;
pub mod surface;
pub mod texture;
pub mod products;
pub mod generate;
pub mod counter;
pub mod banded;
pub mod metrics;
pub mod matching;
pub mod tiles;
pub mod export;
pub mod bake;
