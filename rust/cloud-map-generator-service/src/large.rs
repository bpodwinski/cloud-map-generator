//! A large map, out of core: the banded engine ([`cloud_map_generator_core::banded`]) run over the compute workers
//! ([`crate::bands`]) with its bands streamed to FILES, never held as an image.
//!
//! `docs/large-maps.html` is the design. What stays in memory is a few bands, the 2048-wide slices (coverage, type,
//! high share: 8 MiB each) and the per-row vectors; what goes to disk, streamed band by band, is the raw R8 for the
//! renderer (rows south-up, gzip) and the 16-bit master PNG. The slices become the small PNGs the page can load (the
//! coverage one is the preview, at most 2048 px wide), and the metrics are measured on the same 2048-wide downsample
//! as `measure.py` does for any map wider than the reference.
//!
//! Files of a run, in `out_dir`:
//!
//! | key          | name                                  | what                                           |
//! |--------------|---------------------------------------|------------------------------------------------|
//! | `master`     | `<stem>_<w>x<h>_16bit.png`            | the 16-bit master, streamed                    |
//! | `r8gz`       | `<stem>_<w>x<h>.r8.gz`                | the raw R8, rows south-up, gzip                |
//! | `preview`    | `<stem>_preview_2048x1024.png`        | the 8-bit coverage slice (the preview)         |
//! | `type`       | `<stem>_type_2048x1024.png`           | the cloud type slice                           |
//! | `high_share` | `<stem>_high_share_2048x1024.png`     | the high-cloud share slice                     |
//! | `wind`       | `<stem>_wind_<w>x<h>.png`             | the wind product (a small grid)                |
//! | `meta`       | `<stem>.json`                         | the sidecar                                    |

use crate::bands::{run_bands, BandCompute, CpuBands, WorkerReport};
use crate::engine::{ComputeMode, ComputeUse, DeviceUse, MapOptions, Runtime, SurfaceChoice, Used};
use crate::generate::{compute_json, measure_map, Measured, RunError, REF_W};
use crate::imageio::{gzip_level, PNG_ROW_STREAM};
use crate::meta::gen_meta;
use crate::physics::{body_warnings, cyclones_per_hemisphere};
use crate::pyjson::{dumps, J};
use crate::reference::Reference;
use crate::synthetic::synthetic_surface;
use cloud_map_generator_core::banded::{auto_band_rows, BandOut, BandSink, Banded, BandedDone, BandedError, BandedInfo, Progress, SubNoise};
use cloud_map_generator_core::climate::BodyInputs;
use cloud_map_generator_core::export::{box_downsample, coverage_to_u16, encode_products, map_files, preview_u8, Plane, ProductExtras};
use cloud_map_generator_core::generate::GenParams;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Texels of one band when the caller does not choose its height: the memory of a run is about this many texels
/// times the fields a band holds (measured: see `docs/large-maps.html`), whatever the width.
pub const BAND_TEXELS: usize = 1 << 22;

/// The widest map the engine writes (the texel index is a `u32`).
pub const MAX_LARGE_WIDTH: usize = 32768;

/// From this width the service and the front ends run the banded engine (the whole-array one needs 4 GiB at 8192).
pub const LARGE_FROM_WIDTH: usize = 8192;

/// What to make and where to write it.
#[derive(Clone, Debug)]
pub struct LargeJob {
    pub body: BodyInputs,
    pub seed: u64,
    pub width: usize,
    pub options: MapOptions,
    pub compute: ComputeMode,
    /// The stem of the files' names.
    pub stem: String,
    pub out_dir: PathBuf,
    /// A real surface (land and relief); when `None`, `options.surface` says whether to build the synthetic one.
    pub surface: Option<cloud_map_generator_core::surface::Surface>,
    /// Rows per band (a multiple of `width / 2048` for the slices); `None`: [`BAND_TEXELS`] texels.
    pub band_rows: Option<usize>,
    /// Keep the wall-clock `seconds` in the sidecar.
    pub sidecar_seconds: bool,
}

/// One file of a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileInfo {
    pub key: &'static str,
    pub path: PathBuf,
    pub bytes: u64,
}

/// A finished large map: the files and the numbers; no image.
#[derive(Clone, Debug)]
pub struct LargeGeneration {
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub requested: ComputeMode,
    pub compute: ComputeUse,
    pub meta: cloud_map_generator_core::generate::GenMeta,
    pub sidecar: J,
    pub files: Vec<FileInfo>,
    pub measured: Measured,
    pub warnings: Vec<String>,
    /// The note about the resolution the metrics were measured at.
    pub metrics_note: Option<String>,
    pub band_rows: usize,
    pub bands: usize,
    /// Wall seconds of the banded engine (three passes and the streaming of the files).
    pub engine_seconds: f64,
    /// Wall seconds of the whole run.
    pub seconds: f64,
    /// The slice grid (`REF_W` wide) of the coverage, type and high-share, north-up, and the wind product.
    pub preview: Slices,
}

/// The 2048-wide downsamples a run keeps: the page's preview and the input of the metrics.
#[derive(Clone, Debug, Default)]
pub struct Slices {
    pub width: usize,
    pub height: usize,
    pub coverage: Vec<f32>,
    pub kind: Vec<f32>,
    pub high_share: Vec<f32>,
}

// ---------------------------------------------------------------------------------------------------------
// Sinks

/// The raw R8 of the renderer: rows south-up, gzip. Bands arrive north to south, the file wants the south first, so the
/// bytes go to a temporary file back to front (each band at its place) and are gzipped from it at the end.
struct R8Sink {
    tmp: File,
    tmp_path: PathBuf,
    out_path: PathBuf,
    width: usize,
    height: usize,
}

impl R8Sink {
    fn new(dir: &Path, name: &str, width: usize, height: usize) -> std::io::Result<R8Sink> {
        let tmp_path = dir.join(format!("{name}.partial"));
        let tmp = File::options().read(true).write(true).create(true).truncate(true).open(&tmp_path)?;
        tmp.set_len((width * height) as u64)?;
        Ok(R8Sink { tmp, tmp_path, out_path: dir.join(name), width, height })
    }

    fn band(&mut self, out: &BandOut<'_>) -> std::io::Result<()> {
        let bytes = preview_u8(out.coverage);
        // The band's rows, last row first, land at the band's place counted from the south.
        let mut flipped = Vec::with_capacity(bytes.len());
        for r in (0..out.rows).rev() {
            flipped.extend_from_slice(&bytes[r * self.width..(r + 1) * self.width]);
        }
        let first_row_from_south = self.height - (out.r0 + out.rows);
        self.tmp.seek(SeekFrom::Start((first_row_from_south * self.width) as u64))?;
        self.tmp.write_all(&flipped)
    }

    fn finish(mut self) -> std::io::Result<()> {
        self.tmp.seek(SeekFrom::Start(0))?;
        let mut enc = gzip_level(BufWriter::with_capacity(1 << 20, File::create(&self.out_path)?));
        let mut buf = vec![0u8; 4 << 20];
        let mut left = (self.width * self.height) as u64;
        while left > 0 {
            let want = left.min(buf.len() as u64) as usize;
            let n = self.tmp.read(&mut buf[..want])?;
            if n == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "the temporary R8 file is short"));
            }
            enc.write_all(&buf[..n])?;
            left -= n as u64;
        }
        enc.finish()?.flush()?;
        drop(self.tmp);
        std::fs::remove_file(&self.tmp_path)
    }

    fn abandon(self) {
        let (tmp, out) = (self.tmp_path.clone(), self.out_path.clone());
        drop(self);
        let _ = std::fs::remove_file(tmp);
        let _ = std::fs::remove_file(out);
    }
}

/// The 16-bit master: a streaming PNG encoder, one band at a time, north to south as the format wants.
struct Png16Sink {
    stream: Option<png::StreamWriter<'static, BufWriter<File>>>,
    path: PathBuf,
}

impl Png16Sink {
    fn new(path: &Path, width: usize, height: usize) -> Result<Png16Sink, String> {
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut enc = png::Encoder::new(BufWriter::with_capacity(1 << 20, file), width as u32, height as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Sixteen);
        // Fast: the master is streamed at the speed of the bands; the default level compresses 1 GiB at 30 MB/s, slower than the GPU.
        enc.set_compression(png::Compression::Fast);
        let stream = enc.write_header().and_then(|w| w.into_stream_writer_with_size(PNG_ROW_STREAM)).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Png16Sink { stream: Some(stream), path: path.to_path_buf() })
    }

    fn band(&mut self, out: &BandOut<'_>) -> Result<(), String> {
        let samples = coverage_to_u16(out.coverage);
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for v in samples {
            bytes.extend_from_slice(&v.to_be_bytes());
        }
        self.stream.as_mut().expect("open").write_all(&bytes).map_err(|e| format!("{}: {e}", self.path.display()))
    }

    fn finish(&mut self) -> Result<(), String> {
        let s = self.stream.take().expect("open");
        s.finish().map_err(|e| format!("{}: {e}", self.path.display()))
    }
}

/// The 2048-wide box downsamples, accumulated band by band (a group of `width / 2048` rows makes one slice row).
struct SliceSink {
    width: usize,
    f: usize,
    slice_w: usize,
    out: Slices,
    carry: [Vec<f32>; 3],
    carry_rows: usize,
}

impl SliceSink {
    fn new(width: usize, height: usize) -> SliceSink {
        let (slice_w, f) = if width > REF_W { (REF_W, width / REF_W) } else { (width, 1) };
        let slice_h = height / f;
        let _ = slice_h;
        SliceSink { width, f, slice_w, out: Slices { width: slice_w, height: 0, ..Slices::default() }, carry: [vec![], vec![], vec![]], carry_rows: 0 }
    }

    /// Downsamples `groups` complete groups of rows of `field` (`groups * f` rows) and appends them to `into`.
    fn downsample(&self, field: &[f32], groups: usize, into: &mut Vec<f32>) {
        if self.f == 1 {
            into.extend_from_slice(field);
            return;
        }
        let (f, w, sw) = (self.f, self.width, self.slice_w);
        let rows = cloud_map_generator_core::par::map_rows(groups, |g| box_downsample(&field[g * f * w..(g + 1) * f * w], f, w, sw).0);
        for r in rows {
            into.extend_from_slice(&r);
        }
    }

    fn push(&mut self, rows: usize, fields: [Option<&[f32]>; 3]) {
        let (f, w) = (self.f, self.width);
        let mut start = 0;
        // Finish a group left over from the previous band.
        if self.carry_rows > 0 {
            let need = (f - self.carry_rows).min(rows);
            for (c, field) in self.carry.iter_mut().zip(fields) {
                if let Some(field) = field {
                    c.extend_from_slice(&field[..need * w]);
                }
            }
            self.carry_rows += need;
            start = need;
            if self.carry_rows == f {
                self.flush_carry();
            }
        }
        let groups = (rows - start) / f;
        if groups > 0 {
            for (k, field) in fields.iter().enumerate() {
                if let Some(field) = field {
                    let mut into = std::mem::take(self.target(k));
                    self.downsample(&field[start * w..(start + groups * f) * w], groups, &mut into);
                    *self.target(k) = into;
                }
            }
            self.out.height += groups;
            start += groups * f;
        }
        if start < rows {
            for (c, field) in self.carry.iter_mut().zip(fields) {
                if let Some(field) = field {
                    c.extend_from_slice(&field[start * w..rows * w]);
                }
            }
            self.carry_rows = rows - start;
        }
    }

    fn flush_carry(&mut self) {
        for k in 0..3 {
            if !self.carry[k].is_empty() {
                let c = std::mem::take(&mut self.carry[k]);
                let mut into = std::mem::take(self.target(k));
                self.downsample(&c, 1, &mut into);
                *self.target(k) = into;
            }
        }
        self.out.height += 1;
        self.carry_rows = 0;
    }

    fn target(&mut self, k: usize) -> &mut Vec<f32> {
        match k {
            0 => &mut self.out.coverage,
            1 => &mut self.out.kind,
            _ => &mut self.out.high_share,
        }
    }
}

/// Every sink of a run behind one [`BandSink`].
struct LargeSink {
    r8: Option<R8Sink>,
    master: Option<Png16Sink>,
    slices: SliceSink,
    info: Option<BandedInfo>,
}

impl BandSink for LargeSink {
    fn begin(&mut self, info: &BandedInfo) -> Result<(), String> {
        self.info = Some(info.clone());
        Ok(())
    }

    fn band(&mut self, out: &BandOut<'_>) -> Result<(), String> {
        if let Some(r8) = self.r8.as_mut() {
            r8.band(out).map_err(|e| format!("the R8 file: {e}"))?;
        }
        if let Some(m) = self.master.as_mut() {
            m.band(out)?;
        }
        self.slices.push(out.rows, [Some(out.coverage), out.kind, out.high_share]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------
// The run

/// A progress report of a large run: the fraction of the whole (never decreasing) and the stage, and the pass and
/// band for a progress bar.
pub type LargeProgress<'a> = &'a (dyn Fn(f64, &str, Option<Progress>) + Sync);

/// The widest band height that is a multiple of what the slices need and at most `rows`.
pub fn aligned_band_rows(width: usize, height: usize, rows: usize) -> usize {
    let f = if width > REF_W { width / REF_W } else { 1 };
    let rows = rows.clamp(1, height);
    if rows >= f {
        rows / f * f
    } else {
        f.min(height)
    }
}

/// Checks a large job before anything is allocated or written: the width the slices need, and the options that need the
/// whole image.
pub fn check_large(job: &LargeJob) -> Result<(), RunError> {
    let w = job.width;
    if w > MAX_LARGE_WIDTH {
        return Err(RunError::Refused(format!("width {w}: the banded engine writes at most {MAX_LARGE_WIDTH} texels across")));
    }
    if w > REF_W && w % REF_W != 0 {
        return Err(RunError::Refused(format!("width {w}: a map wider than {REF_W} must be a multiple of {REF_W} (its slices are box downsamples)")));
    }
    if job.options.match_earth {
        return Err(RunError::Refused("match_earth needs the whole image in memory: it is not offered for banded maps (use a width of 8192 or less)".to_string()));
    }
    Ok(())
}

/// Runs a large job on `rt`'s CPU (and, when asked and available, GPU) workers and writes its files.
pub fn run_large(rt: &Runtime, reference: &Reference, job: &LargeJob, progress: LargeProgress<'_>, cancel: &(dyn Fn() -> bool + Sync)) -> Result<LargeGeneration, RunError> {
    let t0 = Instant::now();
    check_large(job)?;
    let (used, note) = rt.resolve(job.compute, job.width);
    let opt = job.options;
    let (width, height) = (job.width, job.width / 2);
    let synthetic = (job.surface.is_none() && opt.surface == SurfaceChoice::Synthetic).then(|| synthetic_surface(width.min(1024)));
    let surface = job.surface.as_ref().or(synthetic.as_ref());
    let params = GenParams { isotropic: opt.isotropic, allow_any_regime: opt.experiment, ..GenParams::default() };
    // Hybrid shares bands, so it wants more of them: a quarter of the texels per band.
    let budget = if used == Used::Hybrid { BAND_TEXELS / 4 } else { BAND_TEXELS };
    let band_rows = aligned_band_rows(width, height, job.band_rows.unwrap_or_else(|| auto_band_rows(width, height, budget)));
    progress(0.0, &format!("seed {} width {} on the {}, bands of {} rows", job.seed, width, used.key(), band_rows), None);
    let banded = Banded::new(&job.body, job.seed, width, &params, surface, SubNoise::Counter, band_rows, !opt.isotropic)
        .map_err(|e| RunError::Refused(e.to_string()))?;
    let bands = banded.bands();

    std::fs::create_dir_all(&job.out_dir).map_err(|e| RunError::Failed(format!("{}: {e}", job.out_dir.display())))?;
    let names = map_files(&job.stem, width, height);
    let master_path = job.out_dir.join(&names.master);
    let r8_path = job.out_dir.join(&names.r8gz);
    let cleanup = |sink: &mut LargeSink| {
        if let Some(r8) = sink.r8.take() {
            r8.abandon();
        }
        sink.master = None;
        let _ = std::fs::remove_file(&master_path);
    };
    let mut sink = LargeSink {
        r8: Some(R8Sink::new(&job.out_dir, &names.r8gz, width, height).map_err(|e| RunError::Failed(format!("{}: {e}", r8_path.display())))?),
        master: Some(Png16Sink::new(&master_path, width, height).map_err(RunError::Failed)?),
        slices: SliceSink::new(width, height),
        info: None,
    };

    // The workers: the CPU always; the GPU where the mode says so.
    let cpu_worker = CpuBands(&banded, (used == Used::Hybrid).then(|| cloud_map_generator_core::par::threads().saturating_sub(2).max(1)));
    let gpu_workers = rt.band_workers(used, &banded).map_err(|e| {
        cleanup(&mut sink);
        RunError::Refused(e)
    })?;
    let mut workers: Vec<&dyn BandCompute> = Vec::new();
    match used {
        Used::Cpu => workers.push(&cpu_worker),
        Used::Gpu => workers.extend(gpu_workers.iter().map(|w| w.as_ref())),
        Used::Hybrid => {
            workers.extend(gpu_workers.iter().map(|w| w.as_ref()));
            workers.push(&cpu_worker);
        }
    }

    let total = (bands * 3) as f64;
    let done = std::sync::atomic::AtomicUsize::new(0);
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    let on_progress = |p: Progress| -> bool {
        if cancel() {
            cancelled.store(true, std::sync::atomic::Ordering::SeqCst);
            return false;
        }
        let n = if p.band > 0 { done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1 } else { done.load(std::sync::atomic::Ordering::SeqCst) };
        let fraction = 0.02 + 0.88 * (n as f64 / total).min(1.0);
        let text = if p.band == 0 { format!("pass {} starts ({} bands)", p.pass.letter(), p.bands) } else { format!("pass {}, band {} of {}", p.pass.letter(), p.band, p.bands) };
        progress(fraction, &text, Some(p));
        true
    };
    let t_engine = Instant::now();
    let run = run_bands(&banded, &workers, &mut sink, &on_progress);
    let engine_seconds = t_engine.elapsed().as_secs_f64();
    let (done_run, reports) = match run {
        Ok(r) => r,
        Err(e) => {
            cleanup(&mut sink);
            return Err(match e {
                BandedError::Cancelled => RunError::Cancelled,
                BandedError::Sink(m) => RunError::Failed(m),
                BandedError::Generate(g) => RunError::Refused(g.to_string()),
            });
        }
    };
    progress(0.90, "writing the files", None);
    sink.master.as_mut().expect("master").finish().map_err(RunError::Failed)?;
    sink.master = None;
    progress(0.92, "compressing the raw R8", None);
    sink.r8.take().expect("r8").finish().map_err(|e| RunError::Failed(format!("{}: {e}", r8_path.display())))?;

    let compute = compute_use(used, note, &workers, &reports);
    let slices = std::mem::take(&mut sink.slices.out);
    let (files, sidecar, measured, warnings, metrics_note) = finish_files(reference, job, &params, &done_run, &banded, &slices, &compute, engine_seconds, &names)?;
    Ok(LargeGeneration {
        width,
        height,
        seed: job.seed,
        requested: job.compute,
        compute,
        meta: done_run.meta.clone(),
        sidecar,
        files,
        measured,
        warnings,
        metrics_note,
        band_rows,
        bands,
        engine_seconds,
        seconds: t0.elapsed().as_secs_f64(),
        preview: slices,
    })
}

/// The `compute` block of a banded run, from what each worker really did: rows completed over the total rows of the
/// three passes, and the wall seconds each worked.
fn compute_use(used: Used, note: Option<String>, workers: &[&dyn BandCompute], reports: &[WorkerReport]) -> ComputeUse {
    let total: usize = reports.iter().map(|r| r.rows).sum::<usize>().max(1);
    let mut devices: Vec<DeviceUse> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    for (w, r) in workers.iter().zip(reports) {
        if let Some(why) = &r.lost {
            notes.push(format!("The {} was lost during the run ({why}); the other device finished the map.", w.kind().key().to_uppercase()));
        }
        match devices.iter_mut().find(|d| d.kind == w.kind()) {
            Some(d) => {
                d.seconds += r.seconds;
                d.share += r.rows as f64 / total as f64;
            }
            None => devices.push(DeviceUse { kind: w.kind(), name: w.name(), seconds: r.seconds, share: r.rows as f64 / total as f64 }),
        }
    }
    let note = [note].into_iter().flatten().chain(notes).collect::<Vec<_>>();
    ComputeUse { used, devices, note: (!note.is_empty()).then(|| note.join(" ")) }
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn finish_files(
    reference: &Reference,
    job: &LargeJob,
    params: &GenParams,
    done: &BandedDone,
    banded: &Banded<'_>,
    slices: &Slices,
    compute: &ComputeUse,
    engine_seconds: f64,
    names: &cloud_map_generator_core::export::MapFiles,
) -> Result<(Vec<FileInfo>, J, Measured, Vec<String>, Option<String>), RunError> {
    let opt = job.options;
    let (width, height) = (job.width, job.width / 2);
    let mut meta = gen_meta(&job.body, params, &done.meta, engine_seconds);
    if !job.sidecar_seconds {
        meta.remove("seconds");
    }
    meta.push(
        "options",
        J::obj([
            ("match_earth", J::Bool(opt.match_earth)),
            ("isotropic", J::Bool(opt.isotropic)),
            ("experiment", J::Bool(opt.experiment)),
            ("surface", J::s(opt.surface.key())),
        ]),
    );
    meta.push("compute", compute_json(job.compute.key(), compute, None));
    meta.push(
        "banded",
        J::obj([
            ("band_rows", J::uint(banded.band_rows as u64)),
            ("bands", J::uint(banded.bands() as u64)),
            ("passes", J::s("A (statistics), B (row thresholds), C (compose and write)")),
            ("noise", J::s("counter-based normal(seed, texel_index)")),
            ("slice_width", J::uint(slices.width as u64)),
        ]),
    );

    let (sw, sh) = (slices.width, slices.height);
    let wind = banded.info().wind;
    let products = encode_products(
        &ProductExtras {
            type_: (!slices.kind.is_empty()).then(|| Plane::new(&slices.kind, sh, sw)),
            high_share: (!slices.high_share.is_empty()).then(|| Plane::new(&slices.high_share, sh, sw)),
            wind: wind.as_ref().map(|(w, h, ww)| Plane::new(w.as_slice(), *h, *ww)),
        },
        &job.stem,
    );
    let io = |p: &Path, e: crate::fail::Fail| RunError::Failed(format!("{}: {e}", p.display()));
    let mut files: Vec<FileInfo> = Vec::new();
    let size = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let push = |files: &mut Vec<FileInfo>, key: &'static str, path: PathBuf| {
        let bytes = size(&path);
        files.push(FileInfo { key, path, bytes });
    };
    push(&mut files, "master", job.out_dir.join(&names.master));
    push(&mut files, "r8gz", job.out_dir.join(&names.r8gz));
    let preview_path = job.out_dir.join(format!("{}_preview_{sw}x{sh}.png", job.stem));
    crate::imageio::write_gray8(&preview_path, sw, sh, &preview_u8(&slices.coverage)).map_err(|e| io(&preview_path, e))?;
    push(&mut files, "preview", preview_path);
    meta.push(
        "products",
        J::obj([
            ("type", J::s(products.sidecar.type_)),
            ("high_share", J::s(products.sidecar.high_share)),
            ("wind", J::s(products.sidecar.wind.as_str())),
            ("files", J::obj(products.sidecar.files.iter().map(|(k, n)| (*k, J::s(n.as_str()))))),
        ]),
    );
    for (key, name) in &products.sidecar.files {
        let img = match *key {
            "type" => products.type_png.as_ref(),
            "high_share" => products.high_share_png.as_ref(),
            _ => products.wind_png.as_ref(),
        }
        .expect("a product with a file name has an image");
        let path = job.out_dir.join(name);
        match img.channels {
            1 => crate::imageio::write_gray8(&path, img.width, img.height, &img.data),
            _ => crate::imageio::write_rgb8(&path, img.width, img.height, &img.data),
        }
        .map_err(|e| io(&path, e))?;
        push(&mut files, key, path);
    }
    let meta_path = job.out_dir.join(&names.meta);
    meta.push(
        "files",
        J::obj(files.iter().map(|f| (f.key, J::s(f.path.file_name().and_then(|n| n.to_str()).unwrap_or(""))))),
    );
    crate::imageio::write_file(&meta_path, dumps(&meta).as_bytes()).map_err(|e| io(&meta_path, e))?;
    push(&mut files, "meta", meta_path);

    let itcz = done.meta.surface.as_ref().map_or(0.0, |s| s.itcz_shift_deg);
    let measured = measure_map(reference, &slices.coverage, sh, sw, Some((done.meta.hadley_edge_deg, itcz)));
    let mut warnings = body_warnings(&job.body, cyclones_per_hemisphere(&job.body));
    if opt.isotropic {
        warnings.push("Isotropic baseline: no flow, no vortices; the surface is ignored. The type, high-share and wind maps still come from the same fields.".to_string());
    }
    let metrics_note = Some(format!(
        "Measured on the {sw}-px downsample of the {width}-px map (box means, as measure does for any map wider than the reference), not on its {} megatexels.",
        (width * height) as f64 / 1e6
    ));
    Ok((files, meta, measured, warnings, metrics_note))
}


/// What a banded run asks of the disk, bytes: the temporary raw R8 and the 16-bit PNG at their worst, the gzip, and a
/// margin for the small files.
pub fn disk_bytes(width: usize) -> u64 {
    let texels = (width as u64) * (width as u64 / 2);
    texels * 2 + texels + texels / 2 + (64 << 20)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::Reference;
    use std::io::Read;

    fn job(dir: &Path, width: usize) -> LargeJob {
        LargeJob {
            body: cloud_map_generator_core::climate::EARTH,
            seed: 5,
            width,
            options: MapOptions::default(),
            compute: ComputeMode::Cpu,
            stem: "t".to_string(),
            out_dir: dir.to_path_buf(),
            surface: None,
            band_rows: Some(128),
            sidecar_seconds: false,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cloudgen-large-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn a_large_run_writes_its_files_and_the_r8_is_the_master_upside_down() {
        let dir = scratch("files");
        let rt = Runtime::cpu_only("no adapter");
        let reference = Reference::from_png(crate::REFERENCE_PNG).unwrap();
        let (w, h) = (4096, 2048);
        let g = run_large(&rt, &reference, &job(&dir, w), &|_, _, _| {}, &|| false).unwrap();
        assert_eq!((g.width, g.height, g.bands, g.band_rows), (w, h, 16, 128));
        let keys: Vec<&str> = g.files.iter().map(|f| f.key).collect();
        assert_eq!(keys, ["master", "r8gz", "preview", "type", "high_share", "wind", "meta"]);
        assert!(g.files.iter().all(|f| f.bytes > 0), "{:?}", g.files);
        assert!(!dir.join("t_4096x2048.r8.gz.partial").exists(), "the temporary file is gone");
        // The raw R8: w x h bytes, rows south-up; its first row is the master's last row, to the 8-bit rounding.
        let mut r8 = Vec::new();
        flate2::read::GzDecoder::new(File::open(&g.files[1].path).unwrap()).read_to_end(&mut r8).unwrap();
        assert_eq!(r8.len(), w * h);
        let master = crate::imageio::read_gray(&g.files[0].path).unwrap();
        assert_eq!((master.width, master.height), (w, h));
        for (rr, mr) in [(0, h - 1), (h - 1, 0), (777, h - 1 - 777)] {
            for x in (0..w).step_by(37) {
                let from_master = (f64::from(master.data[mr * w + x]) / 257.0).round();
                let diff = (f64::from(r8[rr * w + x]) - from_master).abs();
                assert!(diff <= 1.0, "row {rr} col {x}: {} vs {from_master}", r8[rr * w + x]);
            }
        }
        // The slice is a 2048-wide preview; the metrics were measured on it.
        let preview = crate::imageio::read_gray(&g.files[2].path).unwrap();
        assert_eq!((preview.width, preview.height), (2048, 1024));
        assert_eq!(g.measured.rows.len(), 10);
        assert!(g.compute.devices.iter().any(|d| d.share == 1.0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_cancelled_large_run_leaves_no_files_and_match_earth_is_refused() {
        let dir = scratch("cancel");
        let rt = Runtime::cpu_only("no adapter");
        let reference = Reference::from_png(crate::REFERENCE_PNG).unwrap();
        let n = std::sync::atomic::AtomicUsize::new(0);
        let err = run_large(&rt, &reference, &job(&dir, 4096), &|_, _, _| {}, &|| n.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 20).unwrap_err();
        assert_eq!(err, RunError::Cancelled);
        let left: Vec<_> = std::fs::read_dir(&dir).map(|d| d.flatten().map(|e| e.file_name()).collect()).unwrap_or_default();
        assert!(left.is_empty(), "{left:?}");
        let mut j = job(&dir, 4096);
        j.options.match_earth = true;
        assert!(matches!(check_large(&j), Err(RunError::Refused(m)) if m.contains("match_earth")));
        let mut j = job(&dir, 3000);
        j.options.match_earth = false;
        assert!(matches!(check_large(&j), Err(RunError::Refused(m)) if m.contains("multiple of 2048")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_slices_are_the_box_downsample_whatever_the_band_height() {
        let (w, h) = (4096usize, 2048usize);
        let field: Vec<f32> = (0..w * h).map(|i| ((i * 2654435761usize) % 1000) as f32 / 1000.0).collect();
        let (want, wh, ww) = box_downsample(&field, h, w, 2048);
        for rows in [2, 3, 7, 64, 2048] {
            let mut s = SliceSink::new(w, h);
            for r0 in (0..h).step_by(rows) {
                let r1 = (r0 + rows).min(h);
                s.push(r1 - r0, [Some(&field[r0 * w..r1 * w]), None, None]);
            }
            assert_eq!((s.out.height, s.out.width), (wh, ww));
            assert_eq!(s.out.coverage, want, "bands of {rows} rows");
        }
    }
}
