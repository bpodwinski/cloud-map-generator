//! `cloud-map-generator measure`: measure coverage maps against Earth's shipped cloud fraction (`measure.py`).
//!
//!     cloud-map-generator measure <map.png> [<map.png> ...] [--sheet <out.png>] [--json <out.json>]
//!     cloud-map-generator measure <maps generated with --surface <key>> --surface <key> [--bodies <file>] [--surface-json <json>]
//!     cloud-map-generator measure --earth-tables --surface <earth key> [--bodies <file>]
//!
//! Maps are north-up grayscale PNGs (8 or 16 bit). The reference is `data/earth_cloud_fraction_bluemarble_2048.png`
//! (NASA Blue Marble, see `data/README.md`). Every map is box-resampled to the reference's 2048 x 1024 before
//! measuring, so spectra and object sizes compare like for like (`refmap.rs`).
//!
//! Not implemented: `--surface-sheet` (the coast-drawn sheet of the surface terms' places; it needs scipy's
//! binary erosion). The plain `--sheet` is.

use crate::bodies::body_surface;
use crate::fail::{Fail, Result};
use crate::imageio::write_gray8;
use crate::project::Project;
use crate::pyjson::{dumps, py_float, J};
use crate::refmap::{load_map, REF_H, REF_W};
use clap::Args;
use cloud_map_generator_core::generate::{py_round, SURFACE_TERM_WIDTH};
use cloud_map_generator_core::metrics::{compare, judge_surface, measure, surface_contrasts, Comparison, JudgeRow, Measure, MetricMap, PATCH_LATS};
use cloud_map_generator_core::sphere::{gnomonic_patch, grid_lonlat, sample_equirect};
use cloud_map_generator_core::surface::{deck_window, earth_tables, surface_terms};
use std::path::{Path, PathBuf};

#[derive(Args, Debug)]
pub struct MeasureArgs {
    /// North-up grayscale coverage PNGs (8 or 16 bit).
    pub maps: Vec<PathBuf>,
    /// A comparison sheet: whole map and gnomonic views of every map (the reference first).
    #[arg(long)]
    pub sheet: Option<PathBuf>,
    /// Every number measured, as JSON.
    #[arg(long)]
    pub json: Option<PathBuf>,
    /// Judge land/sea contrasts on this body key's land (from --bodies).
    #[arg(long)]
    pub surface: Option<String>,
    /// Bodies file (default examples/bodies.json).
    #[arg(long)]
    pub bodies: Option<PathBuf>,
    #[arg(long)]
    pub surface_json: Option<PathBuf>,
    /// Not implemented (see the module comment).
    #[arg(long)]
    pub surface_sheet: Option<PathBuf>,
    /// Print surface.py's centering tables for the Earth record named by --surface.
    #[arg(long)]
    pub earth_tables: bool,
}

// ---------------------------------------------------------------------------------------------------------
// Python's number formatting (`f'{x:7.3f}'`): a NaN is `nan`, an infinity `inf`.

/// `f'{x:{width}.{prec}f}'`.
pub fn pf(x: f64, width: usize, prec: usize) -> String {
    let s = if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        format!("{x:.prec$}")
    };
    format!("{s:>width$}")
}

/// `f'{x:+{width}.{prec}f}'`.
pub fn pfs(x: f64, width: usize, prec: usize) -> String {
    let s = if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "+inf" } else { "-inf" }.to_string()
    } else {
        format!("{x:+.prec$}")
    };
    format!("{s:>width$}")
}

/// `name[:n]`, by characters.
fn clip(name: &str, n: usize) -> String {
    name.chars().take(n).collect()
}

// ---------------------------------------------------------------------------------------------------------
// The table

/// The two header and per-map lines of the numeric table (`measure.py`'s prints), as text.
pub fn table(rows: &[(String, Measure, Option<Comparison>)]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:<40} {:>5} {:>7} {:>6} {:>5} {:>6} {:>5} {:>7} {:>5} {:>5} {:>5}\n",
        "map", "mean", "zonRMSE", "histL1", "coher", "slope", "big%", "medKm2", "seam", "poleN", "poleS"
    ));
    for (name, m, c) in rows {
        out.push_str(&format!(
            "{:<40} {} {} {} {} {} {} {} {} {} {}\n",
            clip(name, 40),
            pf(m.global_mean, 5, 3),
            pf(c.map_or(0.0, |c| c.zonal_rmse), 7, 3),
            pf(c.map_or(0.0, |c| c.hist_l1), 6, 3),
            pf(m.coherence, 5, 3),
            pf(m.spectral_slope, 6, 2),
            pf(100.0 * m.big_object_share, 5, 1),
            pf(m.median_object_km2, 7, 0),
            pf(m.seam_ratio, 5, 2),
            pf(m.pole_std.0, 5, 3),
            pf(m.pole_std.1, 5, 3),
        ));
    }
    for (name, m, _) in rows {
        let hist: Vec<String> = m.hist.iter().map(|h| pf(*h, 0, 2)).collect();
        out.push_str(&format!("  {:<30} hist {}\n", clip(name, 30), hist.join(" ")));
        let by_lat: Vec<String> = PATCH_LATS.iter().zip(&m.coherence_by_lat).map(|(k, v)| format!("{k}:{}", pf(*v, 0, 2))).collect();
        out.push_str(&format!("  {:<30} coherence by lat {}\n", "", by_lat.join(" ")));
    }
    out
}

fn measure_json(m: &Measure) -> J {
    J::obj([
        ("global_mean", J::Float(m.global_mean)),
        ("zonal_lat", J::floats(&m.zonal_lat)),
        ("zonal", J::floats(&m.zonal)),
        ("hist", J::floats(&m.hist)),
        ("coherence", J::Float(m.coherence)),
        ("coherence_by_lat", J::obj(PATCH_LATS.iter().zip(&m.coherence_by_lat).map(|(k, v)| (k.to_string(), J::Float(*v))))),
        ("spectral_slope", J::Float(m.spectral_slope)),
        ("big_object_share", J::Float(m.big_object_share)),
        ("median_object_km2", J::Float(m.median_object_km2)),
        ("seam_ratio", J::Float(m.seam_ratio)),
        ("pole_std", J::floats(&[m.pole_std.0, m.pole_std.1])),
    ])
}

// ---------------------------------------------------------------------------------------------------------
// The sheet

/// PIL's `Image.resize(size, Image.BILINEAR)` of an 8-bit gray image: a separable triangle filter whose support
/// grows with the reduction (antialiased), horizontal pass first, each pass rounded to 8 bits.
pub fn resize_bilinear(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    // The weights of one axis: for each output index the first source index and its normalized weights.
    fn coeffs(inn: usize, out: usize) -> Vec<(usize, Vec<f64>)> {
        let scale = inn as f64 / out as f64;
        let filterscale = scale.max(1.0);
        let support = filterscale; // the triangle filter's support is 1
        (0..out)
            .map(|x| {
                let center = (x as f64 + 0.5) * scale;
                let xmin = ((center - support + 0.5).floor().max(0.0)) as usize;
                let xmax = ((center + support + 0.5).floor().min(inn as f64)) as usize;
                let mut w: Vec<f64> = (xmin..xmax)
                    .map(|k| {
                        let t = ((k as f64 - center + 0.5) / filterscale).abs();
                        if t < 1.0 { 1.0 - t } else { 0.0 }
                    })
                    .collect();
                let total: f64 = w.iter().sum();
                if total != 0.0 {
                    w.iter_mut().for_each(|v| *v /= total);
                }
                (xmin, w)
            })
            .collect()
    }
    let round8 = |v: f64| v.round().clamp(0.0, 255.0) as u8;
    let (cx, cy) = (coeffs(sw, dw), coeffs(sh, dh));
    let mut mid = vec![0u8; dw * sh];
    for r in 0..sh {
        for (x, (x0, w)) in cx.iter().enumerate() {
            let acc: f64 = w.iter().enumerate().map(|(i, wk)| wk * f64::from(src[r * sw + x0 + i])).sum();
            mid[r * dw + x] = round8(acc);
        }
    }
    let mut out = vec![0u8; dw * dh];
    for (y, (y0, w)) in cy.iter().enumerate() {
        for x in 0..dw {
            let acc: f64 = w.iter().enumerate().map(|(i, wk)| wk * f64::from(mid[(y0 + i) * dw + x])).sum();
            out[y * dw + x] = round8(acc);
        }
    }
    out
}

/// `measure.sheet`: rows = maps, columns = a strip of the whole map and gnomonic views (equator, 20 N, 40 N,
/// 55 N, 40 S, 55 S, both poles): what a camera above each point would see, 2200 km across. Returns the image
/// (8-bit gray) and its width and height.
pub fn sheet(maps: &[(String, Vec<f32>, usize, usize)]) -> (Vec<u8>, usize, usize) {
    const VIEWS: [(f64, f64); 8] = [(0.0, -30.0), (20.0, 150.0), (40.0, -150.0), (55.0, 10.0), (-40.0, 70.0), (-55.0, -100.0), (89.0, 0.0), (-89.0, 0.0)];
    const N: usize = 256;
    let (full_w, full_h) = (512, 256);
    let width = full_w + VIEWS.len() * N;
    let mut rows: Vec<Vec<u8>> = Vec::new(); // each entry: a block of full-width rows
    for (_, img, h, w) in maps {
        let img64: Vec<f64> = img.iter().map(|&v| f64::from(v)).collect();
        // `(img * 255).astype(np.uint8)` truncates; PIL resizes that 8-bit image.
        let as8: Vec<u8> = img.iter().map(|&v| (f64::from(v) * 255.0) as u8).collect();
        let full = resize_bilinear(&as8, *w, *h, full_w, full_h);
        let tiles: Vec<Vec<f64>> = VIEWS
            .iter()
            .map(|&(la, lo)| sample_equirect(&img64, *h, *w, 1, &gnomonic_patch(lo.to_radians(), la.to_radians(), 10.0_f64.to_radians(), N)))
            .collect();
        let mut block = vec![0u8; N * width];
        for r in 0..N {
            let row = &mut block[r * width..(r + 1) * width];
            for c in 0..full_w {
                // The strip is float32(v / 255) in numpy, then x 255 and truncated with the float64 tiles.
                row[c] = ((f64::from(f32::from(full[r * full_w + c]) / 255.0)) * 255.0) as u8;
            }
            for (t, tile) in tiles.iter().enumerate() {
                for c in 0..N {
                    row[full_w + t * N + c] = (tile[r * N + c] * 255.0) as u8;
                }
            }
        }
        rows.push(block);
        rows.push(vec![127u8; 6 * width]); // np.ones * 0.5 * 255 = 127.5, truncated
    }
    rows.pop(); // the last separator
    let data: Vec<u8> = rows.concat();
    let height = data.len() / width;
    (data, width, height)
}

// ---------------------------------------------------------------------------------------------------------
// The surface report

const SURFACE_KEYS: [&str; 9] = [
    "land",
    "ocean",
    "land_minus_ocean",
    "subtropics_land_minus_ocean",
    "north_minus_south",
    "ocean_north_minus_south",
    "midlat_north_minus_south",
    "itcz_lat",
    "deck_minus_open",
];

fn metric_map_json(m: &MetricMap, order: &[(&'static str, f64)]) -> J {
    debug_assert_eq!(m.len(), order.len());
    J::obj(order.iter().map(|(k, v)| (*k, J::Float(*v))))
}

fn judge_json(r: &JudgeRow) -> J {
    J::obj([
        ("metric", J::s(r.metric)),
        ("reference", J::Float(r.reference)),
        ("mean", J::Float(r.mean)),
        ("spread", J::Float(r.spread)),
        ("season", J::Float(r.season)),
        ("tolerance", J::Float(r.tolerance)),
        ("diff", J::Float(r.diff)),
        ("pass", J::Bool(r.pass)),
    ])
}

/// `surface_report`: land/sea and north/south contrasts of the maps against the reference, on the body's land.
fn surface_report(project: &Project, key: &str, reference: &[f32], maps: &[(String, Vec<f32>)], out: Option<&Path>, bodies: Option<&Path>) -> Result<()> {
    let s = body_surface(project, key, bodies)?;
    let at = s.at(REF_W);
    let land = at.land.clone();
    let terms = surface_terms(&s, REF_W, 30.0, 6371.0, 1.0, None, SURFACE_TERM_WIDTH);
    let (_, lat) = grid_lonlat(REF_W, REF_H);
    let mut deck = vec![0.0_f64; REF_H * REF_W];
    for r in 0..REF_H {
        let x = lat[r].to_degrees().abs() / 30.0;
        let dw = deck_window(x);
        for c in 0..REF_W {
            deck[r * REF_W + c] = dw * f64::from(terms.deck_strength[r * REF_W + c]);
        }
    }
    let contrasts = |img: &[f32]| surface_contrasts(img, &land, Some(&deck), REF_H, REF_W);
    let rc = contrasts(reference);
    let cs: Vec<_> = maps.iter().map(|(_, img)| contrasts(img)).collect();

    // np.average(land.mean(axis=1), weights=cos(lat)), in percent.
    let (mut num, mut den) = (0.0, 0.0);
    for r in 0..REF_H {
        let row_mean = land[r * REF_W..(r + 1) * REF_W].iter().map(|&v| f64::from(v)).sum::<f64>() / REF_W as f64;
        num += row_mean * lat[r].cos();
        den += lat[r].cos();
    }
    println!();
    println!("surface contrasts on {key}'s land ({:.1} % land), |lat| < 60", 100.0 * num / den);
    let header: Vec<String> = SURFACE_KEYS.iter().map(|k| format!("{:>13}", clip(k, 13))).collect();
    println!("{:<34} {}", "map", header.join(" "));
    let value = |c: &cloud_map_generator_core::metrics::SurfaceContrasts, k: &str| c.entries().into_iter().find(|(n, _)| *n == k).map_or(f64::NAN, |(_, v)| v);
    let line = |name: &str, c: &cloud_map_generator_core::metrics::SurfaceContrasts| {
        let cells: Vec<String> = SURFACE_KEYS.iter().map(|k| pfs(value(c, k), 13, 3)).collect();
        println!("{:<34} {}", clip(name, 34), cells.join(" "));
    };
    line("earth (Blue Marble)", &rc);
    for ((name, _), c) in maps.iter().zip(&cs) {
        line(name, c);
    }
    let rows = if cs.is_empty() { Vec::new() } else { judge_surface(&rc.to_map(), &cs.iter().map(|c| c.to_map()).collect::<Vec<_>>()) };
    if !rows.is_empty() {
        println!();
        println!("{:<30} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}", "judged metric", "earth", "mean", "spread", "season", "tol", "diff");
        for r in &rows {
            println!(
                "{:<30} {} {} {} {} {} {}  {}",
                r.metric,
                pfs(r.reference, 7, 3),
                pfs(r.mean, 7, 3),
                pf(r.spread, 7, 3),
                pf(r.season, 7, 3),
                pf(r.tolerance, 7, 3),
                pfs(r.diff, 7, 3),
                if r.pass { "PASS" } else { "MISS" }
            );
        }
    }
    if let Some(path) = out {
        let j = J::obj([
            ("reference", metric_map_json(&rc.to_map(), &rc.entries())),
            ("maps", J::obj(maps.iter().zip(&cs).map(|((n, _), c)| (n.as_str(), metric_map_json(&c.to_map(), &c.entries()))))),
            ("judged", J::Arr(rows.iter().map(judge_json).collect())),
        ]);
        crate::imageio::write_file(path, dumps(&j).as_bytes())?;
    }
    Ok(())
}

/// `earth_tables`: the centering tables of surface.py, printed as Python lists of floats rounded to 3 and 4
/// decimals.
fn print_earth_tables(project: &Project, key: &str, bodies: Option<&Path>) -> Result<()> {
    let s = body_surface(project, key, bodies)?;
    let (rc, rd) = earth_tables(&s, 6371.0, 1.0, 30.0, 1024);
    let list = |v: &[f64], digits: usize| -> String {
        let items: Vec<String> = v
            .iter()
            .map(|&x| if x.is_nan() { "nan".to_string() } else { py_float(py_round(x, digits)) })
            .collect();
        format!("[{}]", items.join(", "))
    };
    println!("EARTH_CONTINENTALITY = {}", list(&rc, 3));
    println!("EARTH_DECK = {}", list(&rd, 4));
    Ok(())
}

pub fn run(a: MeasureArgs, project: &Project) -> Result<()> {
    if a.earth_tables {
        let Some(key) = &a.surface else {
            return Err(Fail::new("--earth-tables needs --surface <the Earth record key>"));
        };
        return print_earth_tables(project, key, a.bodies.as_deref());
    }
    if a.surface_sheet.is_some() {
        return Err(Fail::new("--surface-sheet is not implemented in cloudgen (it needs a binary erosion for the coastlines); use measure.py for it"));
    }
    let reference = load_map(&project.reference())?;
    let rm = measure(&reference.data, reference.h, reference.w);
    let mut rows: Vec<(String, Measure, Option<Comparison>)> = vec![("earth (Blue Marble)".to_string(), rm.clone(), None)];
    let mut imgs: Vec<(String, Vec<f32>, usize, usize)> = vec![("earth".to_string(), reference.data.clone(), reference.h, reference.w)];
    for p in &a.maps {
        let img = load_map(p)?;
        let name = p.file_stem().map_or_else(|| p.display().to_string(), |s| s.to_string_lossy().into_owned());
        if img.kept_small {
            eprintln!(
                "note: {name} is {} x {}, narrower than the {REF_W} x {REF_H} reference: measured at its own resolution, not resampled (measure.py gives NaN for such a map)",
                img.w, img.h
            );
        }
        let m = measure(&img.data, img.h, img.w);
        let c = compare(&m, &rm);
        rows.push((name.clone(), m, Some(c)));
        imgs.push((name, img.data, img.h, img.w));
    }
    print!("{}", table(&rows));
    if let Some(path) = &a.json {
        let entries = rows.iter().map(|(n, m, c)| {
            let vs = c.map_or(J::Null, |c| J::obj([("zonal_rmse", J::Float(c.zonal_rmse)), ("hist_l1", J::Float(c.hist_l1))]));
            (n.as_str(), J::obj([("metrics", measure_json(m)), ("vs_earth", vs)]))
        });
        crate::imageio::write_file(path, dumps(&J::obj(entries)).as_bytes())?;
    }
    if let Some(key) = &a.surface {
        let maps: Vec<(String, Vec<f32>)> = imgs[1..].iter().map(|(n, d, h, w)| {
            debug_assert_eq!((*h, *w), (REF_H, REF_W));
            (n.clone(), d.clone())
        }).collect();
        if imgs[1..].iter().any(|(_, _, h, w)| (*h, *w) != (REF_H, REF_W)) {
            return Err(Fail::new("--surface judges maps on the 2048 x 1024 grid: every map must be at least that wide"));
        }
        surface_report(project, key, &reference.data, &maps, a.surface_json.as_deref(), a.bodies.as_deref())?;
    }
    if let Some(path) = &a.sheet {
        let (data, w, h) = sheet(&imgs);
        write_gray8(path, w, h, &data)?;
        println!("sheet: {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_number_formatting() {
        assert_eq!(pf(0.5, 5, 3), "0.500");
        assert_eq!(pf(f64::NAN, 5, 3), "  nan");
        assert_eq!(pf(f64::INFINITY, 6, 1), "   inf");
        assert_eq!(pfs(0.25, 8, 3), "  +0.250");
        assert_eq!(pfs(-0.25, 8, 3), "  -0.250");
        assert_eq!(pfs(f64::NAN, 5, 3), "  nan");
        // Python rounds the exact binary value half to even: 12.5 is exact, 0.125 too.
        assert_eq!(pf(12.5, 7, 0), "     12");
        assert_eq!(pf(13.5, 7, 0), "     14");
        assert_eq!(pf(0.125, 0, 2), "0.12");
        assert_eq!(clip("abcdef", 3), "abc");
        assert_eq!(clip("\u{e9}\u{e8}x", 2), "\u{e9}\u{e8}");
    }

    #[test]
    fn a_flat_resize_stays_flat_and_halving_averages() {
        let flat = vec![100u8; 8 * 4];
        assert!(resize_bilinear(&flat, 8, 4, 4, 2).iter().all(|&v| v == 100));
        // 4 -> 2 columns: the triangle filter of a halving has a support of 2 source pixels each side.
        let ramp: Vec<u8> = (0..16u8).map(|c| c * 10).collect();
        let r = resize_bilinear(&ramp, 16, 1, 8, 1);
        assert_eq!(r.len(), 8);
        assert!(r.windows(2).all(|p| p[0] < p[1]), "{r:?}");
        // Identity size: the filter has a support of 1 pixel and keeps every value.
        assert_eq!(resize_bilinear(&ramp, 16, 1, 16, 1), ramp);
    }

    #[test]
    fn the_sheet_has_the_strip_and_eight_views_per_map() {
        let img = vec![0.5f32; 64 * 32];
        let maps = vec![("a".to_string(), img.clone(), 32, 64), ("b".to_string(), img, 32, 64)];
        let (data, w, h) = sheet(&maps);
        assert_eq!((w, h), (512 + 8 * 256, 256 + 6 + 256));
        assert_eq!(data.len(), w * h);
        // The separator rows are mid-gray (127: 127.5 truncated).
        assert!(data[256 * w..257 * w].iter().all(|&v| v == 127));
    }

    #[test]
    fn the_table_reads_like_measure_pys() {
        let m = Measure {
            global_mean: 0.67,
            zonal_lat: vec![],
            zonal: vec![],
            hist: vec![0.1, 0.2],
            coherence: 0.5,
            coherence_by_lat: vec![0.1; 7],
            spectral_slope: -2.5,
            big_object_share: 0.123,
            median_object_km2: 1234.4,
            seam_ratio: 1.0,
            pole_std: (0.01, 0.02),
        };
        let t = table(&[("earth (Blue Marble)".to_string(), m, None)]);
        let lines: Vec<&str> = t.lines().collect();
        assert_eq!(lines[0], format!("{:<40} {:>5} {:>7} {:>6} {:>5} {:>6} {:>5} {:>7} {:>5} {:>5} {:>5}", "map", "mean", "zonRMSE", "histL1", "coher", "slope", "big%", "medKm2", "seam", "poleN", "poleS"));
        assert_eq!(lines[1], format!("{:<40} 0.670   0.000  0.000 0.500  -2.50  12.3    1234  1.00 0.010 0.020", "earth (Blue Marble)"));
        assert_eq!(lines[2], format!("  {:<30} hist 0.10 0.20", "earth (Blue Marble)"));
        assert!(lines[3].ends_with("coherence by lat -55:0.10 -40:0.10 -20:0.10 0:0.10 20:0.10 40:0.10 55:0.10"));
    }
}
