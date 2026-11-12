//! `cloud-map-generator tiles`: bake the cell-field detail tile (`bake_tiles.py`, `tiles.rs`): four regimes in one
//! tileable RGBA tile.
//!
//! Writes `output/tiles/<version>/` (or `--out`):
//! - `cloud_cell_tiles_<size>_<version>.rgba.gz`: the tile, RGBA8 rows in order (the engine uploads them as
//!   they are; the tile has no north), gzipped;
//! - `cloud_cell_tiles_<size>_<version>_preview.png`: one row a channel, the rank then what the deck draws of
//!   it at 30, 50 and 70 % coverage;
//! - `cloud_cell_tiles_<size>_<version>.json`: the sidecar (channels, sizes, seed, encoding).

use crate::fail::{Fail, Result};
use crate::imageio::{gzip9, write_file, write_gray8};
use crate::project::Project;
use clap::Args;
use cloud_map_generator_core::tiles::{
    bake_preview, bake_rgba, bake_sidecar_json, bake_stem, build_tile_seeded, BakeArgs, DEFAULT_BAKE_SEED, DEFAULT_BAKE_VERSION, TILE_SIZE,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Args, Debug)]
pub struct TilesArgs {
    #[arg(long, default_value = DEFAULT_BAKE_VERSION)]
    pub version: String,
    #[arg(long, default_value_t = DEFAULT_BAKE_SEED as u64)]
    pub seed: u64,
    #[arg(long, default_value_t = TILE_SIZE)]
    pub size: usize,
    /// Output folder (default output/tiles/<version>). Not in bake_tiles.py, which has none.
    #[arg(long)]
    pub out: Option<PathBuf>,
}

pub fn run(a: TilesArgs, project: &Project) -> Result<()> {
    if a.size < 2 {
        return Err(Fail::new(format!("--size {}: a tile is at least 2 texels across", a.size)));
    }
    let seed = i64::try_from(a.seed).map_err(|_| Fail::new(format!("--seed {}: too large", a.seed)))?;
    let t0 = Instant::now();
    let tile = build_tile_seeded(a.size, a.seed);
    let out = a.out.clone().unwrap_or_else(|| project.output().join("tiles").join(&a.version));
    std::fs::create_dir_all(&out).map_err(|e| Fail::new(format!("{}: {e}", out.display())))?;
    let stem = bake_stem(a.size, &a.version);
    let gz_path = out.join(format!("{stem}.rgba.gz"));
    write_file(&gz_path, &gzip9(&bake_rgba(&tile)))?;
    let side = 4 * a.size;
    write_gray8(&out.join(format!("{stem}_preview.png")), side, side, &bake_preview(&tile, a.size))?;
    let sidecar = bake_sidecar_json(&BakeArgs { version: a.version.clone(), seed, size: a.size });
    write_file(&out.join(format!("{stem}.json")), sidecar.as_bytes())?;
    let mb = std::fs::metadata(&gz_path).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);
    println!("{stem}: {:.1} s, {mb:.2} MB gz -> {}", t0.elapsed().as_secs_f64(), out.display());
    Ok(())
}
