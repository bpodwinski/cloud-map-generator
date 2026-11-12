//! The bake of a catalog of bodies to files (`bake.py`, `service.bake`): generation runs here, offline, and a
//! renderer only loads the files. Two products:
//!
//! - LIST: every body of the bodies file whose clouds are water and that has no measured map. Its own physics,
//!   a seed from its key; its land and relief (`surface`) enter the bake when the record has one.
//! - LIBRARY: K seeds per climate class, classes in bins of the Hadley edge. Each class is baked on an
//!   Earth-sized body whose rotation puts its Hadley edge at the class center (`cloud_map_generator_core::bake::library_body`).
//!
//! Each entry: the high-resolution coverage (R8, rows SOUTH-up, gzip), its 2048 x 1024 box downsample
//! (north-up PNG), the high-cloud share and the cloud type at 2048 (north-up PNGs), the wind (RGB PNG) and a
//! sidecar JSON. The caller writes the manifest (`manifest.json`) from the returned value: a partial bake
//! (`only`) writes none.

use crate::bodies::{load_surface, Bodies};
use crate::fail::{Context, Fail};
use crate::imageio::{gzip9, write_file, write_gray8, write_rgb8};
use crate::meta::gen_meta;
use crate::pyjson::{dumps, J};
use cloud_map_generator_core::bake::{bake_one, key_seed, library_entries, list_stem, list_width, LibraryEntry, LIBRARY_CLASSES};
use cloud_map_generator_core::climate::{is_bakeable, BodyInputs};
use cloud_map_generator_core::export::{ManifestRecord, ENTRY_ENCODING};
use cloud_map_generator_core::generate::GenParams;
use cloud_map_generator_core::surface::Surface;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Which part of a bake runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Only {
    List,
    Library,
}

/// What to bake.
#[derive(Clone, Debug)]
pub struct BakeOptions {
    pub version: String,
    /// The manifest's path prefix (default `baked/<version>`).
    pub url_prefix: Option<String>,
    /// Width of the list maps (default: the texel rule capped at 8192).
    pub list_width: Option<usize>,
    pub library_width: usize,
    /// Seeds per library class.
    pub k: usize,
    pub only: Option<Only>,
}

/// One baked entry, for a log line.
#[derive(Clone, Debug)]
pub struct EntryLog {
    pub label: String,
    pub width: usize,
    pub height: usize,
    pub hadley_edge_deg: f64,
    /// `None`: the entry has no surface.
    pub land_fraction: Option<f64>,
    pub seconds: f64,
}

/// What a bake tells the caller as it goes (and asks: `starting` returns false to stop).
pub trait Observer {
    /// A body of the list left out, and why.
    fn skipped(&mut self, key: &str, reason: &str);
    /// The entry `label` (the `index`-th of `total`, from 0) is about to be baked; false cancels the bake.
    fn starting(&mut self, label: &str, index: usize, total: usize) -> bool;
    /// An entry's files are written.
    fn entry(&mut self, log: &EntryLog);
    /// The library part starts (after the list part), for a printed heading.
    fn section(&mut self, _name: &str) {}
}

/// Why a bake stopped.
#[derive(Debug)]
pub enum BakeError {
    Fail(Fail),
    Cancelled,
}

impl From<Fail> for BakeError {
    fn from(f: Fail) -> BakeError {
        BakeError::Fail(f)
    }
}

/// What a bake wrote.
#[derive(Clone, Debug)]
pub struct Baked {
    /// The manifest (to be written to `manifest.json` unless `only` was given).
    pub manifest: J,
    /// Every entry file written, under the output folder.
    pub files: Vec<PathBuf>,
    /// Bodies the list left out: key and reason.
    pub skipped: Vec<(String, String)>,
}

fn record_json(r: &ManifestRecord) -> J {
    J::obj([
        ("seed", J::uint(r.seed)),
        ("hires", J::obj([("url", J::s(r.hires.url.as_str())), ("width", J::uint(r.hires.width as u64)), ("height", J::uint(r.hires.height as u64))])),
        ("slice", J::s(r.slice.as_str())),
        ("highShare", J::s(r.high_share.as_str())),
        ("type", J::s(r.type_.as_str())),
        ("wind", J::s(r.wind.as_str())),
    ])
}

/// `bake_one` plus `write_entry`: generates the entry, writes its files and returns its manifest record.
#[allow(clippy::too_many_arguments)]
fn bake_entry(
    body: &BodyInputs,
    seed: u64,
    width: usize,
    out: &Path,
    stem: &str,
    prefix: &str,
    label: &str,
    surface: Option<&Surface>,
    files_written: &mut Vec<PathBuf>,
) -> Result<(ManifestRecord, EntryLog), Fail> {
    let t0 = Instant::now();
    let baked = bake_one(body, seed, width, stem, prefix, label, surface).map_err(|e| Fail::new(format!("{label}: {e}")))?;
    let enc = &baked.encoded;
    let files = &enc.files;
    let (h, w) = (baked.generated.height, baked.generated.width);
    debug_assert_eq!((h, w), (width / 2, width));

    let hires_path = out.join(&files.hires);
    write_file(&hires_path, &gzip9(&enc.hires_south_up))?;
    files_written.push(hires_path);
    for (name, img) in [(&files.slice, &enc.slice), (&files.high_share, &enc.high_share), (&files.type_, &enc.type_)] {
        let path = out.join(name);
        write_gray8(&path, img.width, img.height, &img.data)?;
        files_written.push(path);
    }
    let wind_path = out.join(&files.wind);
    write_rgb8(&wind_path, enc.wind.width, enc.wind.height, &enc.wind.data)?;
    files_written.push(wind_path);

    let seconds = t0.elapsed().as_secs_f64();
    let mut meta = gen_meta(body, &GenParams::default(), &baked.generated.meta, seconds);
    meta.push("label", J::s(label));
    meta.push("files", J::obj(files.pairs().iter().map(|(k, v)| (*k, J::s(*v)))));
    meta.push("encoding", J::obj(ENTRY_ENCODING.iter().map(|(k, v)| (*k, J::s(*v)))));
    let meta_path = out.join(&files.meta);
    write_file(&meta_path, dumps(&meta).as_bytes())?;
    files_written.push(meta_path);

    let log = EntryLog {
        label: label.to_string(),
        width,
        height: h,
        hadley_edge_deg: baked.generated.meta.hadley_edge_deg,
        land_fraction: baked.generated.meta.surface.as_ref().map(|s| s.land_fraction),
        seconds: t0.elapsed().as_secs_f64(),
    };
    Ok((enc.record.clone(), log))
}

/// Bakes `bodies` into `out` (created). The manifest of a full bake is in the result; writing it is the caller's.
pub fn bake(bodies: &Bodies, opts: &BakeOptions, out: &Path, obs: &mut dyn Observer) -> Result<Baked, BakeError> {
    std::fs::create_dir_all(out).context(|| format!("{}", out.display()))?;
    let prefix = opts.url_prefix.clone().unwrap_or_else(|| format!("baked/{}", opts.version));
    let do_list = matches!(opts.only, None | Some(Only::List));
    let do_library = matches!(opts.only, None | Some(Only::Library));
    let n_list = if do_list { bodies.records.iter().filter(|r| !r.measured && is_bakeable(&r.body)).count() } else { 0 };
    let total = n_list + if do_library { LIBRARY_CLASSES.len() * opts.k } else { 0 };
    let mut done = 0usize;
    let mut list: Vec<(String, J)> = Vec::new();
    let mut library: Vec<J> = Vec::new();
    let mut files = Vec::new();
    let mut skipped = Vec::new();

    if do_list {
        obs.section("list");
        for rec in &bodies.records {
            if rec.measured {
                obs.skipped(&rec.key, "measured map");
                skipped.push((rec.key.clone(), "measured map".to_string()));
                continue;
            }
            if !is_bakeable(&rec.body) {
                let reason = format!("clouds of {}", rec.body.cloud_regime);
                obs.skipped(&rec.key, &reason);
                skipped.push((rec.key.clone(), reason));
                continue;
            }
            if !obs.starting(&rec.key, done, total) {
                return Err(BakeError::Cancelled);
            }
            done += 1;
            let width = list_width(rec.body.radius_km, opts.list_width);
            let surface = match &rec.surface {
                Some(s) => Some(load_surface(s, &bodies.base)?),
                None => None,
            };
            let (record, log) = bake_entry(
                &rec.body,
                u64::from(key_seed(&rec.key)),
                width,
                out,
                &list_stem(&rec.key, &opts.version),
                &prefix,
                &rec.key,
                surface.as_ref(),
                &mut files,
            )?;
            obs.entry(&log);
            list.push((rec.key.clone(), record_json(&record)));
        }
    }
    if do_library {
        obs.section("library");
        let all = library_entries(opts.k);
        for (class_index, &(lo, hi, center)) in LIBRARY_CLASSES.iter().enumerate() {
            let mut entries = Vec::new();
            for e in all.iter().filter(|e: &&LibraryEntry| e.class_index == class_index) {
                if !obs.starting(&e.label(), done, total) {
                    return Err(BakeError::Cancelled);
                }
                done += 1;
                let (record, log) = bake_entry(&e.body(), e.seed(), opts.library_width, out, &e.stem(&opts.version), &prefix, &e.label(), None, &mut files)?;
                obs.entry(&log);
                entries.push(record_json(&record));
            }
            library.push(J::obj([
                ("hadleyMinDeg", J::Float(lo)),
                ("hadleyMaxDeg", J::Float(hi)),
                ("bakedAtDeg", J::Float(center)),
                ("entries", J::Arr(entries)),
            ]));
        }
    }
    let manifest = J::obj([
        ("version", J::s(opts.version.as_str())),
        ("generatedBy", J::s("planet-cloudgen bake.py")),
        ("note", J::s("Baked offline. Paths are relative to the host the files are served from.")),
        ("list", J::Obj(list)),
        ("library", J::Arr(library)),
    ]);
    Ok(Baked { manifest, files, skipped })
}
