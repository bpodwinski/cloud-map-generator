//! The banded engine in the browser: the native large-map path (`cloud-map-generator-service/src/large.rs`) over the
//! page's Web Workers and its WebGPU device, with the files kept in memory instead of written to disk.
//!
//! The local server runs a map through the banded engine ([`cloud_map_generator_core::banded`]) when it is 8192 wide or
//! more, and for `compute: hybrid` at any width (the bands are what the GPU and the CPU share). The page does the same,
//! so its maps equal the server's to float rounding:
//!
//! * every worker of the run opens the same [`Banded`] (plans, coarse back-trace, surface terms: deterministic in the
//!   request) with the counter-based sub-texel noise;
//! * the page hands each band of each pass to whichever unit is free: a CPU worker (the core's passes) or the GPU worker
//!   (`cloud_map_generator_gpu::pipeline::bands`, the native GPU band worker), as `cloud-map-generator-service/src/bands.rs`
//!   does with threads;
//! * between the passes the page sends every worker the moments of all bands (each computes the same statistics) and the
//!   raw thresholds of all rows (each smooths them the same way): only per-row vectors cross;
//! * the finished bands go, north to south, to worker 0, which streams them into the 16-bit master (PNG), the raw R8 (rows
//!   south-up) and the 2048-wide slices, then ends the run with the server's own [`finish_parts`].
//!
//! The memory of a worker is a few bands; the sink holds the two big files and the slices: at 8192 x 4096 about 64 MB of
//! master and 32 MiB of R8, at 32768 x 16384 about 1 GB and 512 MiB, which a tab may not have (the local server streams
//! them to disk instead).

use crate::finalize::{meta_json, now_ms, params_of, reference, Finished};
use crate::request::GenRequest;
use cloud_map_generator_core::banded::{auto_band_rows, BandMoments, BandOut, BandResult, BandThresholds, Banded, BandedDone, SubNoise};
use cloud_map_generator_core::climate::BodyInputs;
use cloud_map_generator_core::export::{coverage_to_u16, map_files, preview_u8};
use cloud_map_generator_core::generate::{py_round, NormStats};
use cloud_map_generator_core::surface::Surface;
use cloud_map_generator_service::engine::{ComputeMode, ComputeUse, SurfaceChoice, Used};
use cloud_map_generator_service::generate::{compute_json, metric_table, round5};
use cloud_map_generator_service::imageio::{gzip_level, PNG_ROW_STREAM};
use cloud_map_generator_service::images;
use cloud_map_generator_service::json::opt_str;
use cloud_map_generator_service::large::{aligned_band_rows, finish_parts, LargeJob, LargeParts, SliceSink, BAND_TEXELS, LARGE_FROM_WIDTH};
use cloud_map_generator_service::pyjson::{dumps, J};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::rc::Rc;

/// Whether a request runs on the banded engine: as the local server decides (`serve/generate.rs`).
pub fn is_large(width: usize, requested: ComputeMode) -> bool {
    width >= LARGE_FROM_WIDTH || requested == ComputeMode::Hybrid
}

/// Rows per band: the server's height for the mode, lowered so that every unit of the run (`units`: the workers and the
/// GPU that take bands) gets at least two bands to share, and aligned to what the slices need.
pub fn band_rows_for(width: usize, used: Used, units: usize) -> usize {
    let height = width / 2;
    let budget = if used == Used::Hybrid { BAND_TEXELS / 4 } else { BAND_TEXELS };
    let server = auto_band_rows(width, height, budget);
    let shared = height.div_ceil(2 * units.max(1));
    aligned_band_rows(width, height, server.min(shared).max(1))
}

/// The bands of a map of `width` cut in bands of `band_rows` rows.
pub fn band_count(width: usize, band_rows: usize) -> usize {
    (width / 2).div_ceil(band_rows)
}

// ---------------------------------------------------------------------------------------------------------
// The run held by a worker

/// The file sink of worker 0: the master streamed, the R8 placed row by row, the slices.
struct Sink {
    master: Option<png::StreamWriter<'static, SharedBuf>>,
    master_bytes: SharedBuf,
    r8: Vec<u8>,
    slices: Option<SliceSink>,
    next: usize,
}

/// A byte buffer the PNG stream writes into and the run reads back when it is done.
#[derive(Clone, Default)]
struct SharedBuf(Rc<RefCell<Vec<u8>>>);

impl Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) struct Run {
    pub(crate) req: GenRequest,
    pub(crate) banded: Rc<Banded<'static>>,
    pub(crate) norm: Option<Rc<NormStats>>,
    pub(crate) thr: Option<Rc<(Vec<f64>, Vec<f64>)>>,
    sink: Option<Sink>,
}

thread_local! {
    pub(crate) static RUN: RefCell<Option<Run>> = const { RefCell::new(None) };
    /// The synthetic world of each size, made once per worker (a run borrows it for its life).
    static SURFACES: RefCell<HashMap<usize, &'static Surface>> = RefCell::new(HashMap::new());
}

fn surface_for(width: usize) -> &'static Surface {
    let size = width.min(1024);
    SURFACES.with(|s| *s.borrow_mut().entry(size).or_insert_with(|| Box::leak(Box::new(cloud_map_generator_service::synthetic::synthetic_surface(size)))))
}

fn with_run<T>(f: impl FnOnce(&mut Run) -> Result<T, String>) -> Result<T, String> {
    RUN.with(|r| f(r.borrow_mut().as_mut().ok_or("no banded run is open in this worker")?))
}

/// Opens the run of `req` in this worker, cut in bands of `band_rows` rows.
pub fn open(req: GenRequest, band_rows: usize) -> Result<(), String> {
    let params = params_of(&req.options);
    // The run borrows the body for its life; a body is a few numbers, kept once per run.
    let body: &'static BodyInputs = Box::leak(Box::new(req.body.clone()));
    let surface = (req.options.surface == SurfaceChoice::Synthetic).then(|| surface_for(req.width));
    let banded = Banded::new(body, req.seed, req.width, &params, surface, SubNoise::Counter, band_rows, !req.options.isotropic).map_err(|e| e.to_string())?;
    RUN.with(|r| *r.borrow_mut() = Some(Run { req, banded: Rc::new(banded), norm: None, thr: None, sink: None }));
    Ok(())
}

pub fn close() {
    RUN.with(|r| *r.borrow_mut() = None);
}

pub(crate) fn banded() -> Result<Rc<Banded<'static>>, String> {
    with_run(|r| Ok(r.banded.clone()))
}

pub(crate) fn norm() -> Result<Rc<NormStats>, String> {
    with_run(|r| r.norm.clone().ok_or_else(|| "pass B before the statistics of pass A".to_string()))
}

pub(crate) fn thresholds() -> Result<Rc<(Vec<f64>, Vec<f64>)>, String> {
    with_run(|r| r.thr.clone().ok_or_else(|| "pass C before the thresholds of pass B".to_string()))
}

pub(crate) fn check_band(banded: &Banded<'_>, band: usize) -> Result<(usize, usize), String> {
    if band >= banded.bands() {
        return Err(format!("band {band}: the map has {} bands", banded.bands()));
    }
    Ok(banded.band_range(band))
}

// ---------------------------------------------------------------------------------------------------------
// What crosses between the workers, as flat arrays

/// Pass A of a band, flat: the five lengths, then the `(sum, sum of squares)` pairs of `s`, generic, street, closed, open.
pub fn moments_to_flat(m: &BandMoments) -> Vec<f64> {
    let parts = [&m.s, &m.generic, &m.street, &m.closed, &m.open];
    let mut out: Vec<f64> = parts.iter().map(|p| p.len() as f64).collect();
    for p in parts {
        for &(a, b) in p {
            out.push(a);
            out.push(b);
        }
    }
    out
}

/// The moments of every band, in band order, from their flat forms concatenated.
pub fn moments_from_flat(flat: &[f64], bands: usize) -> Result<Vec<BandMoments>, String> {
    let mut at = 0;
    let mut out = Vec::with_capacity(bands);
    for _ in 0..bands {
        let lens: Vec<usize> = flat.get(at..at + 5).ok_or("moments: short")?.iter().map(|&n| n as usize).collect();
        at += 5;
        let mut take = |n: usize| -> Result<Vec<(f64, f64)>, String> {
            let v = flat.get(at..at + 2 * n).ok_or("moments: short")?;
            at += 2 * n;
            Ok(v.chunks_exact(2).map(|p| (p[0], p[1])).collect())
        };
        out.push(BandMoments { s: take(lens[0])?, generic: take(lens[1])?, street: take(lens[2])?, closed: take(lens[3])?, open: take(lens[4])? });
    }
    if at != flat.len() {
        return Err("moments: trailing numbers".to_string());
    }
    Ok(out)
}

/// Pass B of a band, flat: `[anomaly_min, anomaly_max, rows, sensitivity rows, thr_raw..., sensitivity...]`.
pub fn thresholds_to_flat(t: &BandThresholds) -> Vec<f64> {
    let mut out = vec![f64::from(t.anomaly_min), f64::from(t.anomaly_max), t.thr_raw.len() as f64, t.sensitivity.len() as f64];
    out.extend_from_slice(&t.thr_raw);
    out.extend_from_slice(&t.sensitivity);
    out
}

/// Pass C of a band, flat: the coverage rows, then the type and high-share rows when the run makes them.
pub fn result_to_flat(r: BandResult) -> Vec<f32> {
    let mut out = r.coverage;
    if let (Some(k), Some(h)) = (r.kind, r.high_share) {
        out.extend_from_slice(&k);
        out.extend_from_slice(&h);
    }
    out
}

// ---------------------------------------------------------------------------------------------------------
// The CPU passes and the barriers

pub fn cpu_pass_a(band: usize) -> Result<Vec<f64>, String> {
    let b = banded()?;
    check_band(&b, band)?;
    Ok(moments_to_flat(&b.pass_a(band)))
}

pub fn cpu_pass_b(band: usize) -> Result<Vec<f64>, String> {
    let (b, norm) = (banded()?, norm()?);
    check_band(&b, band)?;
    Ok(thresholds_to_flat(&b.pass_b(band, &norm)))
}

pub fn cpu_pass_c(band: usize) -> Result<Vec<f32>, String> {
    let (b, norm, thr) = (banded()?, norm()?, thresholds()?);
    check_band(&b, band)?;
    Ok(result_to_flat(b.pass_c(band, &norm, &thr.0, &thr.1)))
}

/// Barrier 1: the statistics of the whole map from the moments of every band (each worker computes the same).
pub fn set_moments(flat: &[f64]) -> Result<(), String> {
    let b = banded()?;
    let moments = moments_from_flat(flat, b.bands())?;
    let norm = b.norm_from(&moments);
    with_run(|r| {
        r.norm = Some(Rc::new(norm));
        Ok(())
    })
}

/// Barrier 2: the raw thresholds of every row smoothed in latitude, and the row sensitivity, for pass C.
pub fn set_thresholds(thr_raw: &[f64], sens: Vec<f64>) -> Result<(), String> {
    let b = banded()?;
    if thr_raw.len() != b.height {
        return Err(format!("thresholds: {} rows, the map has {}", thr_raw.len(), b.height));
    }
    let thr = b.smooth(thr_raw);
    with_run(|r| {
        r.thr = Some(Rc::new((thr, sens)));
        Ok(())
    })
}

// ---------------------------------------------------------------------------------------------------------
// The sink (worker 0)

pub fn sink_begin() -> Result<(), String> {
    with_run(|r| {
        let (w, h) = (r.req.width, r.req.width / 2);
        let master_bytes = SharedBuf::default();
        let mut enc = png::Encoder::new(master_bytes.clone(), w as u32, h as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Sixteen);
        // As the server: fast, the master is streamed at the speed of the bands.
        enc.set_compression(png::Compression::Fast);
        let stream = enc.write_header().and_then(|wr| wr.into_stream_writer_with_size(PNG_ROW_STREAM)).map_err(|e| format!("the 16-bit master: {e}"))?;
        // The master grows band by band: reserve about what it ends at (1.9 bytes a texel measured at 8192, fast PNG) so a
        // 32768-wide one is not copied as it doubles.
        master_bytes.0.borrow_mut().reserve(w * h * 2);
        r.sink = Some(Sink { master: Some(stream), master_bytes, r8: vec![0; w * h], slices: Some(SliceSink::new(w, h)), next: 0 });
        Ok(())
    })
}

/// Band `band` of pass C (bands arrive north to south: the page orders them).
pub fn sink_band(band: usize, data: &[f32]) -> Result<(), String> {
    with_run(|r| {
        let (r0, r1) = check_band(&r.banded, band)?;
        let w = r.req.width;
        let h = w / 2;
        let rows = r1 - r0;
        let n = rows * w;
        let extras = r.banded.want_extras;
        let want = if extras { 3 * n } else { n };
        if data.len() != want {
            return Err(format!("band {band}: {} values, expected {want}", data.len()));
        }
        let sink = r.sink.as_mut().ok_or("the sink was not begun")?;
        if band != sink.next {
            return Err(format!("band {band} reached the sink before band {}", sink.next));
        }
        sink.next += 1;
        let out = BandOut {
            index: band,
            r0,
            rows,
            width: w,
            coverage: &data[..n],
            kind: extras.then(|| &data[n..2 * n]),
            high_share: extras.then(|| &data[2 * n..3 * n]),
        };
        // The 16-bit master, north to south as the format wants.
        let mut be = Vec::with_capacity(n * 2);
        for v in coverage_to_u16(out.coverage) {
            be.extend_from_slice(&v.to_be_bytes());
        }
        sink.master.as_mut().ok_or("the master is closed")?.write_all(&be).map_err(|e| format!("the 16-bit master: {e}"))?;
        // The raw R8, rows south-up: this band's rows, last first, at their place counted from the south.
        let bytes = preview_u8(out.coverage);
        let first_from_south = h - r1;
        for (k, row) in (0..rows).rev().enumerate() {
            let at = (first_from_south + k) * w;
            sink.r8[at..at + w].copy_from_slice(&bytes[row * w..(row + 1) * w]);
        }
        sink.slices.as_mut().ok_or("the slices are closed")?.band(&out);
        Ok(())
    })
}

/// The end of the run on the sink: the server's sidecar, products and measures ([`finish_parts`]), the files as bytes
/// and the answer, which is the server's for a large map minus what only a disk has (`files`, `banded.dir`).
#[allow(clippy::too_many_arguments)]
pub fn finish(id: &str, compute: ComputeUse, anomaly: Option<[f32; 2]>, engine_seconds: f64, elapsed_before: f64) -> Result<Finished, String> {
    let t_start = now_ms();
    let (req, banded, norm, thr, mut sink) = RUN.with(|r| -> Result<_, String> {
        let mut r = r.borrow_mut();
        let run = r.as_mut().ok_or("no banded run is open in this worker")?;
        let sink = run.sink.take().ok_or("the sink was not begun")?;
        Ok((run.req.clone(), run.banded.clone(), run.norm.clone(), run.thr.clone(), sink))
    })?;
    if sink.next != banded.bands() {
        return Err(format!("the sink has {} of the {} bands", sink.next, banded.bands()));
    }
    let (width, height) = (req.width, req.width / 2);
    sink.master.take().ok_or("the master is closed")?.finish().map_err(|e| format!("the 16-bit master: {e}"))?;
    let master = std::mem::take(&mut *sink.master_bytes.0.borrow_mut());
    let mut gz = gzip_level(Vec::with_capacity(width * height / 4));
    gz.write_all(&sink.r8).map_err(|e| e.to_string())?;
    let r8gz = gz.finish().map_err(|e| e.to_string())?;
    drop(sink.r8);
    let slices = sink.slices.take().ok_or("the slices are closed")?.into_slices();

    let params = params_of(&req.options);
    let done = BandedDone {
        meta: banded.meta(banded.uses_surface().then_some(anomaly.unwrap_or([f32::NAN, f32::NAN]))),
        thresholds: thr.map(|t| t.0.clone()).unwrap_or_default(),
        norm: norm.map(|n| *n).ok_or("the run has no statistics")?,
    };
    let job = LargeJob {
        body: req.body.clone(),
        seed: req.seed,
        width,
        options: req.options,
        compute: req.compute,
        stem: "coverage".to_string(),
        out_dir: PathBuf::new(),
        surface: None,
        band_rows: Some(banded.band_rows),
        sidecar_seconds: true,
    };
    let reference = reference()?;
    let LargeParts { mut meta, preview8, slice_width: sw, slice_height: sh, products, measured, warnings, metrics_note } =
        finish_parts(&reference, &job, &params, &done, &banded, &slices, &compute, engine_seconds);
    let names = map_files(&job.stem, width, height);
    let mut file_names: Vec<(&'static str, String)> =
        vec![("master", names.master.clone()), ("r8gz", names.r8gz.clone()), ("preview", format!("{}_preview_{sw}x{sh}.png", job.stem))];
    file_names.extend(products.sidecar.files.iter().map(|(k, n)| (*k, n.clone())));
    file_names.push(("meta", names.meta.clone()));
    meta.push("files", J::obj(file_names.iter().map(|(k, n)| (*k, J::s(n.as_str())))));

    let mut files: Vec<(&'static str, &'static str, Vec<u8>)> = vec![("coverage.png", "image/png", images::gray8(sw, sh, &preview8))];
    if let Some(t) = &products.type_png {
        files.push(("type.png", "image/png", images::gray8(t.width, t.height, &t.data)));
    }
    if let Some(s) = &products.high_share_png {
        files.push(("high_share.png", "image/png", images::gray8(s.width, s.height, &s.data)));
    }
    if let Some(wp) = &products.wind_png {
        files.push(("wind.png", "image/png", images::rgb8(wp.width, wp.height, &wp.data)));
    }
    files.push(("master16.png", "image/png", master));
    files.push(("coverage.r8.gz", "application/gzip", r8gz));
    files.push(("meta.json", "application/json; charset=utf-8", dumps(&meta).into_bytes()));

    let total_seconds = elapsed_before + (now_ms() - t_start) / 1000.0;
    let stem = format!("cloudgen_s{}_{width}x{height}_{id}", req.seed);
    let href = |name: &str| format!("/api/result/{id}/{name}");
    let floats = |v: &[f64]| J::Arr(v.iter().map(|&x| J::Float(round5(x))).collect());
    let m = &measured;
    let mut answer = vec![
        ("id", J::s(id)),
        ("seconds", J::Float(py_round(total_seconds, 2))),
        ("width", J::Int(width as i64)),
        ("height", J::Int(height as i64)),
        ("compute", compute_json(req.compute.key(), &compute, Some(total_seconds))),
        ("meta", meta_json(&done.meta)),
        ("images", J::obj(["coverage", "type", "high_share", "wind"].map(|k| (k, J::s(href(&format!("{k}.png"))))))),
    ];
    if sw < width {
        answer.push(("image_note", J::s(format!("The images are the {sw}-px downsample of the {width}-px map: the full map is in the downloads."))));
    }
    answer.extend([
        (
            "downloads",
            J::Arr(vec![
                J::obj([("label", J::s("Cloud fraction, 16-bit PNG")), ("href", J::s(href("master16.png"))), ("filename", J::s(format!("{stem}_16bit.png")))]),
                J::obj([("label", J::s("Raw R8 for the shader (gzip, rows south-up)")), ("href", J::s(href("coverage.r8.gz"))), ("filename", J::s(format!("{stem}.r8.gz")))]),
                J::obj([("label", J::s("Sidecar JSON")), ("href", J::s(href("meta.json"))), ("filename", J::s(format!("{stem}.json")))]),
            ]),
        ),
        ("banded", J::obj([("band_rows", J::Int(banded.band_rows as i64)), ("bands", J::Int(banded.bands() as i64))])),
        (
            "metrics",
            J::obj([
                ("table", metric_table(&m.measures, &m.comparison, &reference.measures)),
                ("earth_label", J::s("Earth (Blue Marble)")),
                ("note", opt_str(metrics_note.as_deref())),
            ]),
        ),
        (
            "zonal",
            J::obj([("lat_deg", J::floats(&m.zonal.lat_deg)), ("generated", floats(&m.zonal.generated)), ("earth", floats(&m.zonal.earth)), ("target", floats(&m.zonal.target))]),
        ),
        ("warnings", J::Arr(warnings.into_iter().map(J::Str).collect())),
    ]);
    close();
    Ok(Finished { json: cloud_map_generator_service::json::compact(&J::obj(answer)), engine_seconds, files })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::parse_generate;

    fn request(width: usize, options: &str) -> GenRequest {
        let v: serde_json::Value = serde_json::from_str(&format!(
            r#"{{"body": {{"rotation_period_days": 0.99727, "radius_km": 6371, "gravity_mps2": 9.81, "scale_height_km": 8}}, "seed": 42, "width": {width}, "options": {options}}}"#
        ))
        .unwrap();
        let rt = cloud_map_generator_service::engine::Runtime::cpu_only("no GPU in this test");
        parse_generate(&v, &[256, 512, 1024, 2048, 4096, 8192], &rt).unwrap()
    }

    /// The page's protocol run in one process: three workers' worth of passes in a shuffled order, the barriers from the
    /// flat forms, the sink fed north to south. The map must equal the core's one-call banded run (same bands) exactly:
    /// the order the bands are made in changes nothing.
    fn run_like_the_page(req: GenRequest, band_rows: usize) -> (Vec<f32>, Finished) {
        open(req.clone(), band_rows).unwrap();
        let bands = banded().unwrap().bands();
        let order: Vec<usize> = (0..bands).rev().collect();
        let mut moments = vec![Vec::new(); bands];
        for &b in &order {
            moments[b] = cpu_pass_a(b).unwrap();
        }
        set_moments(&moments.concat()).unwrap();
        let (mut thr_raw, mut sens, mut lo, mut hi) = (Vec::new(), Vec::new(), f32::INFINITY, f32::NEG_INFINITY);
        let mut parts = vec![Vec::new(); bands];
        for &b in &order {
            parts[b] = cpu_pass_b(b).unwrap();
        }
        for p in &parts {
            let (n, ns) = (p[2] as usize, p[3] as usize);
            lo = lo.min(p[0] as f32);
            hi = hi.max(p[1] as f32);
            thr_raw.extend_from_slice(&p[4..4 + n]);
            sens.extend_from_slice(&p[4 + n..4 + n + ns]);
        }
        set_thresholds(&thr_raw, sens).unwrap();
        let mut results = vec![Vec::new(); bands];
        for &b in &order {
            results[b] = cpu_pass_c(b).unwrap();
        }
        sink_begin().unwrap();
        let w = req.width;
        let mut coverage = Vec::new();
        for (b, r) in results.iter().enumerate() {
            let (r0, r1) = banded().unwrap().band_range(b);
            coverage.extend_from_slice(&r[..(r1 - r0) * w]);
            sink_band(b, r).unwrap();
        }
        let compute = ComputeUse { used: Used::Cpu, devices: vec![], note: None };
        let anomaly = banded().unwrap().uses_surface().then_some([lo, hi]);
        let done = finish("0123456789ab", compute, anomaly, 1.0, 1.0).unwrap();
        (coverage, done)
    }

    fn core_banded(req: &GenRequest, band_rows: usize) -> Vec<f32> {
        struct Collect(Vec<f32>);
        impl cloud_map_generator_core::banded::BandSink for Collect {
            fn band(&mut self, out: &BandOut<'_>) -> Result<(), String> {
                self.0.extend_from_slice(out.coverage);
                Ok(())
            }
        }
        let params = params_of(&req.options);
        let surface = (req.options.surface == SurfaceChoice::Synthetic).then(|| cloud_map_generator_service::synthetic::synthetic_surface(req.width.min(1024)));
        let b = Banded::new(&req.body, req.seed, req.width, &params, surface.as_ref(), SubNoise::Counter, band_rows, !req.options.isotropic).unwrap();
        let mut sink = Collect(Vec::new());
        b.run(&mut sink, &mut |_| true).unwrap();
        sink.0
    }

    #[test]
    fn the_pages_protocol_equals_the_cores_banded_run() {
        for options in [r#"{}"#, r#"{"surface": "synthetic"}"#, r#"{"isotropic": true}"#] {
            let req = request(256, options);
            let (page, done) = run_like_the_page(req.clone(), 24);
            assert_eq!(page, core_banded(&req, 24), "{options}");
            let names: Vec<&str> = done.files.iter().map(|f| f.0).collect();
            assert!(names.contains(&"master16.png") && names.contains(&"coverage.r8.gz") && names.contains(&"meta.json"), "{names:?}");
            let answer: serde_json::Value = serde_json::from_str(&done.json).unwrap();
            assert_eq!(answer["banded"]["bands"], 6, "{options}");
            assert_eq!(answer["width"], 256);
            assert!(answer["metrics"]["table"].as_array().is_some_and(|t| !t.is_empty()));
            // The R8 is the coverage, south-up.
            let r8gz = &done.files.iter().find(|f| f.0 == "coverage.r8.gz").unwrap().2;
            let mut r8 = Vec::new();
            std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(&r8gz[..]), &mut r8).unwrap();
            let north_up = preview_u8(&page);
            let w = 256;
            assert_eq!(&r8[..w], &north_up[north_up.len() - w..], "the first R8 row is the southernmost");
        }
    }

    #[test]
    fn bands_are_shared_between_every_unit() {
        assert_eq!(band_count(8192, band_rows_for(8192, Used::Hybrid, 16)), 32);
        assert!(band_count(1024, band_rows_for(1024, Used::Hybrid, 16)) >= 16);
        assert!(band_rows_for(8192, Used::Gpu, 1) % 4 == 0, "a multiple of width / 2048 for the slices");
        assert!(is_large(8192, ComputeMode::Auto) && is_large(512, ComputeMode::Hybrid) && !is_large(4096, ComputeMode::Gpu));
    }
}
