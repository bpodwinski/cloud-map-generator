//! `cloud-map-generator gen`: generate one coverage map (`gen.py`).
//!
//!     cloud-map-generator gen --seed 42 --width 4096 [--body earth|fast|mars|titan|venus] [--out <dir>]
//!                  [--surface <key> [--bodies <file>]]
//!
//! Output defaults to `output/<body>_s<seed>/`. Files, as `gen.py` writes them: the products (type, high
//! share, wind: north-up PNGs), the 16-bit master, the 8-bit preview, the raw R8 (rows south-up, gzip) and the
//! sidecar JSON.

use crate::bodies::{body_surface, TestBody};
use crate::fail::{Fail, Result};
use crate::imageio::{gzip9, write_file, write_gray16, write_gray8, write_rgb8};
use crate::meta::gen_meta;
use crate::project::Project;
use crate::pyjson::{dumps, J};
use crate::refmap::load_map;
use clap::Args;
use cloud_map_generator_core::climate::{hadley_edge_deg, is_bakeable, map_width_for_texel, MAP_MAX_WIDTH};
use cloud_map_generator_core::export::{encode_map, encode_products, Image8, Plane, ProductExtras};
use cloud_map_generator_core::generate::{generate, GenParams};
use cloud_map_generator_core::matching::{match_to_earth, DEFAULT_LEVELS, DEFAULT_STRENGTH};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Args, Debug)]
pub struct GenArgs {
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Map width; default from --texel-km (climate::map_width_for_texel).
    #[arg(long)]
    pub width: Option<usize>,
    #[arg(long, default_value_t = 3.0)]
    pub texel_km: f64,
    #[arg(long, value_enum, default_value = "earth")]
    pub body: TestBody,
    /// Output folder (default output/<body>_s<seed>).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Baseline: no flow, isotropic fBm.
    #[arg(long)]
    pub isotropic: bool,
    /// Post-process: Earth's per-scale, per-zone statistics (matching.rs).
    #[arg(long)]
    pub match_earth: bool,
    /// Generate for a non-water cloud regime anyway (never a bake).
    #[arg(long)]
    pub experiment: bool,
    /// A body key whose land and relief to use (from --bodies); default none.
    #[arg(long)]
    pub surface: Option<String>,
    /// Bodies file (default examples/bodies.json).
    #[arg(long)]
    pub bodies: Option<PathBuf>,
    /// Rows per band of the out-of-core (banded) engine; by default about 4 million texels per band, so the memory
    /// does not depend on the width. Giving it, or a width above 8192, runs the banded engine.
    #[arg(long)]
    pub band_rows: Option<usize>,
    /// Run the banded engine even for a small width.
    #[arg(long)]
    pub banded: bool,
    /// Where the banded engine runs: auto, cpu, gpu or hybrid (CPU and GPU share the bands).
    #[arg(long, default_value = "auto")]
    pub compute: String,
    /// A file that, once it exists, cancels a running banded map (so does `q` + Enter on the standard input).
    #[arg(long)]
    pub cancel_file: Option<PathBuf>,
}

/// The name of the run's files: `coverage_<body>_s<seed>` and the tags of the options in effect.
pub fn run_stem(body: &str, seed: u64, isotropic: bool, match_earth: bool, surface: bool) -> String {
    let tag = format!(
        "{}{}{}",
        if isotropic { "_iso" } else { "" },
        if match_earth { "_m" } else { "" },
        if surface { "_surf" } else { "" }
    );
    format!("coverage_{body}_s{seed}{tag}")
}

fn write_image8(path: &Path, img: &Image8) -> Result<()> {
    match img.channels {
        1 => write_gray8(path, img.width, img.height, &img.data),
        3 => write_rgb8(path, img.width, img.height, &img.data),
        c => Err(Fail::new(format!("{}: {c}-channel image", path.display()))),
    }
}


/// Above this width the banded (out-of-core) engine runs by default: the whole-array engine needs about 30 GiB at 32768.
pub const BANDED_FROM_WIDTH: usize = 8192;

/// The banded engine: three passes over bands, the files streamed (see `cloud_map_generator_service::large`).
fn run_banded(a: &GenArgs, project: &Project, body: cloud_map_generator_core::climate::BodyInputs, width: usize, out: &Path, stem: &str, surface: Option<cloud_map_generator_core::surface::Surface>) -> Result<()> {
    use cloud_map_generator_service::engine::{ComputeMode, MapOptions, Runtime, SurfaceChoice};
    use cloud_map_generator_service::large::{check_large, run_large, LargeJob};
    use cloud_map_generator_service::reference::Reference;
    use std::sync::atomic::{AtomicBool, Ordering};
    let _ = project;
    if a.match_earth {
        return Err(Fail::new("--match-earth needs the whole image in memory: it is not offered for the banded engine (use a width of 8192 or less)"));
    }
    let compute = ComputeMode::from_key(&a.compute).ok_or_else(|| Fail::new(format!("--compute {}: expected auto, cpu, gpu or hybrid", a.compute)))?;
    let job = LargeJob {
        body,
        seed: a.seed,
        width,
        options: MapOptions { match_earth: false, isotropic: a.isotropic, experiment: a.experiment, surface: SurfaceChoice::None },
        compute,
        stem: stem.to_string(),
        out_dir: out.to_path_buf(),
        surface,
        band_rows: a.band_rows,
        sidecar_seconds: true,
    };
    check_large(&job).map_err(|e| Fail::new(run_error_text(e)))?;
    let rt = if compute == ComputeMode::Cpu { Runtime::cpu_only("the CPU was asked for") } else { Runtime::probe() };
    let reference = Reference::from_png(cloud_map_generator_service::REFERENCE_PNG).map_err(Fail::new)?;
    // Cancel: a file that appears, or `q` on the standard input (end of input does not cancel).
    let cancel = std::sync::Arc::new(AtomicBool::new(false));
    {
        let flag = cancel.clone();
        std::thread::spawn(move || {
            let mut line = String::new();
            while let Ok(n) = std::io::stdin().read_line(&mut line) {
                if n == 0 {
                    return;
                }
                if line.trim().eq_ignore_ascii_case("q") {
                    flag.store(true, Ordering::SeqCst);
                    return;
                }
                line.clear();
            }
        });
    }
    let cancel_file = a.cancel_file.clone();
    let started = Instant::now();
    let progress = |fraction: f64, text: &str, pass: Option<cloud_map_generator_core::banded::Progress>| {
        let elapsed = started.elapsed().as_secs_f64();
        let eta = if fraction > 0.03 { format!(", about {:.0} s left", elapsed * (1.0 - fraction) / fraction) } else { String::new() };
        // A band is a line; the starts of the passes and the file steps too.
        let _ = pass;
        println!("[{elapsed:7.1} s] {text} ({:.0} %{eta})", fraction * 100.0);
    };
    let cancelled = || cancel.load(Ordering::SeqCst) || cancel_file.as_ref().is_some_and(|p| p.exists());
    println!("banded engine: {} seed {} width {} (cancel: q + Enter{})", a.body.name(), a.seed, width, a.cancel_file.as_ref().map_or(String::new(), |p| format!(" or create {}", p.display())));
    let g = run_large(&rt, &reference, &job, &progress, &cancelled).map_err(|e| Fail::new(run_error_text(e)))?;
    println!("done in {:.1} s (engine {:.1} s), {} bands of {} rows", g.seconds, g.engine_seconds, g.bands, g.band_rows);
    for d in &g.compute.devices {
        println!("  device {} {}: {:.1} s, share {:.4}", d.kind.key(), d.name.as_deref().unwrap_or(""), d.seconds, d.share);
    }
    if let Some(n) = &g.compute.note {
        println!("  note: {n}");
    }
    for f in &g.files {
        println!("  {}: {} ({:.1} MiB)", f.key, f.path.display(), f.bytes as f64 / 1048576.0);
    }
    println!("  mean cloud fraction {:.4} (Earth {:.4})", g.measured.measures.global_mean, reference.measures.global_mean);
    if let Some(n) = &g.metrics_note {
        println!("  {n}");
    }
    Ok(())
}

fn run_error_text(e: cloud_map_generator_service::generate::RunError) -> String {
    use cloud_map_generator_service::generate::RunError;
    match e {
        RunError::Refused(m) | RunError::Failed(m) => m,
        RunError::Cancelled => "cancelled: the partial files were removed".to_string(),
    }
}

pub fn run(a: GenArgs, project: &Project) -> Result<()> {
    let surface = match &a.surface {
        Some(key) => Some(body_surface(project, key, a.bodies.as_deref())?),
        None => None,
    };
    let body = a.body.inputs();
    if !is_bakeable(&body) && !a.experiment {
        return Err(Fail::new(format!(
            "{}: clouds of {}, not water; this generator's cloud amount is Earth's (see climate.is_bakeable). Use --experiment to generate anyway.",
            a.body.name(),
            body.cloud_regime
        )));
    }
    let width = match a.width {
        Some(w) => w,
        None => map_width_for_texel(body.radius_km, a.texel_km, MAP_MAX_WIDTH) as usize,
    };
    if width < 2 {
        return Err(Fail::new(format!("--width {width}: a map is at least 2 texels wide")));
    }
    let out = a.out.clone().unwrap_or_else(|| project.output().join(format!("{}_s{}", a.body.name(), a.seed)));
    println!("{} seed {} width {}", a.body.name(), a.seed, width);
    let stem = run_stem(a.body.name(), a.seed, a.isotropic, a.match_earth, surface.is_some());

    if a.banded || a.band_rows.is_some() || width > BANDED_FROM_WIDTH {
        return run_banded(&a, project, body, width, &out, &stem, surface);
    }

    let params = GenParams { isotropic: a.isotropic, allow_any_regime: a.experiment, ..GenParams::default() };
    let t0 = Instant::now();
    let gen = generate(&body, a.seed, width, &params, surface.as_ref(), !a.isotropic).map_err(|e| Fail::new(e.to_string()))?;
    let seconds = t0.elapsed().as_secs_f64();
    let (h, w) = (gen.height, gen.width);
    let mut meta = gen_meta(&body, &params, &gen.meta, seconds);
    meta.push("body_name", J::s(a.body.name()));

    let mut cov = gen.coverage;
    if a.match_earth {
        let reference = project.reference();
        let earth = load_map(&reference)?;
        cov = match_to_earth(&cov, h, w, &earth.data, earth.h, earth.w, gen.meta.hadley_edge_deg, DEFAULT_LEVELS, DEFAULT_STRENGTH);
        meta.push("matched_to", J::s(crate::project::REFERENCE));
    }
    debug_assert_eq!(hadley_edge_deg(&body), gen.meta.hadley_edge_deg);

    std::fs::create_dir_all(&out).map_err(|e| Fail::new(format!("{}: {e}", out.display())))?;
    let mut written: Vec<(&str, PathBuf)> = Vec::new();

    // Products first: they record their encoding in the sidecar that the map's files then write.
    if let Some(ex) = &gen.extras {
        let products = encode_products(
            &ProductExtras {
                type_: Some(Plane::new(&ex.kind, h, w)),
                high_share: Some(Plane::new(&ex.high_share, h, w)),
                wind: Some(Plane::new(&ex.wind, ex.wind_height, ex.wind_width)),
            },
            &stem,
        );
        let images = [("type", &products.type_png), ("high_share", &products.high_share_png), ("wind", &products.wind_png)];
        for (key, name) in &products.sidecar.files {
            let img = images.iter().find(|(k, _)| k == key).and_then(|(_, i)| i.as_ref()).expect("a product that has a file name has an image");
            let path = out.join(name);
            write_image8(&path, img)?;
            written.push((key, path));
        }
        meta.push(
            "products",
            J::obj([
                ("type", J::s(products.sidecar.type_)),
                ("high_share", J::s(products.sidecar.high_share)),
                ("wind", J::s(products.sidecar.wind.as_str())),
                ("files", J::obj(products.sidecar.files.iter().map(|(k, n)| (*k, J::s(n.as_str()))))),
            ]),
        );
    }

    let map = encode_map(&cov, h, w, &stem);
    let paths = [
        ("master", out.join(&map.files.master)),
        ("preview", out.join(&map.files.preview)),
        ("r8gz", out.join(&map.files.r8gz)),
        ("meta", out.join(&map.files.meta)),
    ];
    write_gray16(&paths[0].1, w, h, &map.master16)?;
    write_gray8(&paths[1].1, w, h, &map.preview8)?;
    write_file(&paths[2].1, &gzip9(&map.r8_south_up))?;
    write_file(&paths[3].1, dumps(&meta).as_bytes())?;
    written.extend(paths);
    for (k, p) in &written {
        println!("  {k}: {}", p.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stem_carries_the_options_in_gen_pys_order() {
        assert_eq!(run_stem("earth", 42, false, false, false), "coverage_earth_s42");
        assert_eq!(run_stem("earth", 7, true, false, false), "coverage_earth_s7_iso");
        assert_eq!(run_stem("fast", 1, false, true, true), "coverage_fast_s1_m_surf");
        assert_eq!(run_stem("fast", 1, true, true, true), "coverage_fast_s1_iso_m_surf");
    }
}
