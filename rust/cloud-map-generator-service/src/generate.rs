//! One generation, start to finish (`run_generation` of `ui/server.py`, `generate` of `cloudgen/service.py`): the map
//! from an engine, the match to Earth, the encodings, the measures against Earth.
//!
//! What runs where: the engine makes the map ([`crate::engine`]); everything here is CPU work on every engine (the
//! match to Earth, the PNG and gzip encodings, the metrics, which are FFTs of gnomonic patches). The result is a
//! typed [`Generation`]: `cloud-map-generator serve` turns it into its JSON answer and keeps the files in memory,
//! `cloud-map-generator-mcp` writes them under its output folder.
//!
//! Progress and cancellation: the run reports its stages as fractions of the whole (the engine's own progress is
//! scaled into its share) and polls `cancel` at every stage boundary, so a cancelled run stops before the next
//! stage starts and nothing has been written (this module writes nothing).

use crate::engine::{ComputeMode, ComputeUse, DeviceKind, DeviceUse, EngineError, GenerateRequest, Generated, MapOptions, Runtime, SurfaceChoice};
use crate::imageio::gzip9;
use crate::images;
use crate::json::opt_str;
use crate::meta::gen_meta;
use crate::physics::{body_warnings, cyclones_per_hemisphere};
use crate::project::REFERENCE;
use crate::pyjson::{dumps, J};
use crate::reference::Reference;
use crate::synthetic::synthetic_surface;
use cloud_map_generator_core::climate::{target_zonal_coverage, BodyInputs};
use cloud_map_generator_core::export::{box_downsample, encode_map, encode_products, Plane, ProductExtras};
use cloud_map_generator_core::generate::{py_round, GenOutput, GenParams};
use cloud_map_generator_core::matching::{refine_to_earth, DEFAULT_LEVELS, DEFAULT_STRENGTH};
use cloud_map_generator_core::metrics::{compare, measure, zonal_profile, Comparison, Measure, DEFAULT_BAND_DEG};
use cloud_map_generator_core::sphere::grid_lonlat;
use std::borrow::Cow;
use std::f64::consts::PI;
use std::time::Instant;

/// The reference's grid: a map wider than this (an exact multiple) is box-resampled to it before it is measured.
pub const REF_W: usize = 2048;

/// The widest map the whole-array engine makes (4 GiB of memory at 8192); the banded engine takes over from [`crate::large::LARGE_FROM_WIDTH`].
pub const WHOLE_ARRAY_MAX_WIDTH: usize = 4096;

/// A synthetic world is made at most this wide (the surface terms work at 1024).
const SYNTHETIC_SURFACE_MAX_WIDTH: usize = 1024;

/// `measured_image`: what `measure.py` measures: the map box-resampled to Earth's 2048 x 1024 when it is wider
/// (an exact multiple), as is otherwise (the metrics sample gnomonic patches at any resolution).
pub fn measured_image(cov: &[f32], h: usize, w: usize) -> (Cow<'_, [f32]>, usize, usize) {
    if w > REF_W && w % REF_W == 0 {
        let (data, oh, ow) = box_downsample(cov, h, w, REF_W);
        (Cow::Owned(data), oh, ow)
    } else {
        (Cow::Borrowed(cov), h, w)
    }
}

/// `np.round(x, 5)`: scaled, rounded half to even, scaled back.
pub fn round5(x: f64) -> f64 {
    (x * 1e5).round_ties_even() / 1e5
}

// ---------------------------------------------------------------------------------------------------------
// The comparison table

type RowGetter = fn(&cloud_map_generator_core::metrics::Measure, Option<&Comparison>) -> f64;

/// One row of the comparison table: a statistic of the map next to Earth's.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricRow {
    pub key: &'static str,
    pub label: &'static str,
    /// The map's value, rounded to 6 decimals (`NaN` when the statistic is undefined).
    pub generated: f64,
    /// Earth's value, rounded to 6 decimals.
    pub earth: f64,
    /// The number format the page prints it with (digits after the dot).
    pub format: &'static str,
    /// The page's one-line hint.
    pub hint: &'static str,
    /// The longer explanation the MCP tools give: how to read the row and what moves it.
    pub explanation: &'static str,
}

/// The rows of the comparison table: key, label, getter for the generated map and for Earth, number format, hint,
/// explanation. The distances to Earth are 0 for Earth by definition.
#[allow(clippy::type_complexity)]
fn row_defs() -> [(&'static str, &'static str, RowGetter, &'static str, &'static str, &'static str); 10] {
    [
        (
            "mean",
            "Mean cloud fraction",
            |m, _| m.global_mean,
            "0.000",
            "area-weighted; closer to Earth is better",
            "Area-weighted global mean. It follows the Hadley edge: a smaller edge (faster spin, larger radius, smaller gravity x scale height) is cloudier, a larger one clearer; a seed changes the weather, not the mean. A gap is expected when the body is not Earth-like.",
        ),
        (
            "zonal_rmse",
            "Zonal-mean RMSE vs Earth",
            |_, c| c.map_or(0.0, |c| c.zonal_rmse),
            "0.000",
            "5-degree bands, |lat| < 70; lower is better, but a body with another Hadley edge differs by design",
            "Distance of the zonal-mean profile from Earth's (5-degree bands, |lat| < 70), in the same latitudes. Large by design when the Hadley edge is not Earth's 30 degrees: judge the calibration by the zonal profile's `target` column instead.",
        ),
        (
            "hist_l1",
            "Histogram distance to Earth (L1)",
            |_, c| c.map_or(0.0, |c| c.hist_l1),
            "0.000",
            "10-bin area-weighted histogram; lower is better",
            "Distance between the 10-bin, area-weighted cloud-fraction histograms (0 identical, 2 disjoint). High means the map is too uniform or too binary next to Earth's. match_earth closes it by construction (best at width 2048 or more).",
        ),
        (
            "coherence",
            "Structure coherence",
            |m, _| m.coherence,
            "0.000",
            "0 isotropic, toward 1 parallel bands; closer to Earth is better",
            "0 for an isotropic field, toward 1 for parallel bands. Too high: bands and jets dominate (a fast spin); too low: blobby. isotropic=true gives the no-flow baseline to compare with.",
        ),
        (
            "spectral_slope",
            "Spectral slope (40-400 km)",
            |m, _| m.spectral_slope,
            "0.00",
            "log-log radial power spectrum; closer to Earth is better",
            "Log-log slope of the radial power spectrum; more negative is blurrier. It only compares with Earth at width 2048 or more, where the 40-400 km scales are resolved.",
        ),
        (
            "big_object_share",
            "Cloud in big objects (%)",
            |m, _| 100.0 * m.big_object_share,
            "0.0",
            "share of cloudy area in objects over 100,000 km2; closer to Earth is better",
            "Share of the cloudy area held by objects over 100,000 km2: organized systems against speckle. Bigger cyclones (a slower spin gives a larger deformation radius) raise it.",
        ),
        (
            "median_object_km2",
            "Median object area (km2)",
            |m, _| m.median_object_km2,
            "0",
            "objects above 0.5 coverage, measured on Earth-sized patches; closer to Earth is better",
            "Median area of cloud objects (fraction above 0.5), measured on Earth-sized patches so bodies compare in km2. It follows the size of the cyclones and cells.",
        ),
        (
            "seam_ratio",
            "Seam ratio",
            |m, _| m.seam_ratio,
            "0.00",
            "antimeridian jump over the neighbour-column jump; 1 is invisible",
            "Jump across the antimeridian over the typical neighbor-column jump; about 1 means no visible seam. The generator has none by construction.",
        ),
        (
            "pole_std_north",
            "Pole spread, north",
            |m, _| m.pole_std.0,
            "0.000",
            "std of the first row; a pole is one point, so lower is better",
            "Std across longitude of the first row. A pole is one point, so a high value is a visible stretch or pinch there; lower is better.",
        ),
        (
            "pole_std_south",
            "Pole spread, south",
            |m, _| m.pole_std.1,
            "0.000",
            "std of the last row; a pole is one point, so lower is better",
            "Std across longitude of the last row. A pole is one point, so a high value is a visible stretch or pinch there; lower is better.",
        ),
    ]
}

/// The comparison table of a map measured against Earth: ten rows.
pub fn metric_rows(generated: &Measure, comparison: &Comparison, earth: &Measure) -> Vec<MetricRow> {
    row_defs()
        .into_iter()
        .map(|(key, label, get, format, hint, explanation)| MetricRow {
            key,
            label,
            generated: py_round(get(generated, Some(comparison)), 6),
            earth: py_round(get(earth, None), 6),
            format,
            hint,
            explanation,
        })
        .collect()
}

/// The `metrics.table` of an answer.
pub fn metric_table(generated: &Measure, comparison: &Comparison, earth: &Measure) -> J {
    J::Arr(
        metric_rows(generated, comparison, earth)
            .into_iter()
            .map(|r| {
                J::obj([
                    ("key", J::s(r.key)),
                    ("label", J::s(r.label)),
                    ("generated", J::Float(r.generated)),
                    ("earth", J::Float(r.earth)),
                    ("format", J::s(r.format)),
                    ("hint", J::s(r.hint)),
                ])
            })
            .collect(),
    )
}

/// The zonal-mean cloud fraction in 5-degree bands, south first: the band centers, the map's, Earth's and the
/// profile the climate model aims for (empty when the map does not belong to a known body).
#[derive(Clone, Debug, PartialEq)]
pub struct Zonal {
    pub lat_deg: Vec<f64>,
    pub generated: Vec<f64>,
    pub earth: Vec<f64>,
    pub target: Vec<f64>,
}

/// The target zonal profile of a body: the Hadley edge and the ITCZ shift, in the bands of `zonal_profile`, on a
/// grid of `height` rows.
pub fn target_zonal(height: usize, hadley_edge_deg: f64, itcz_shift_deg: f64) -> Vec<f64> {
    // The grid's own latitudes: only the rows matter, one column.
    let (_, lat) = grid_lonlat(2 * height, height);
    let target: Vec<f64> = lat.iter().map(|&la| target_zonal_coverage(la * (180.0 / PI), hadley_edge_deg, itcz_shift_deg)).collect();
    zonal_profile(&target, height, 1, DEFAULT_BAND_DEG).1
}

/// A map measured against Earth: the measures, the comparison and the zonal profile (the target profile only when
/// `target` gives the body's Hadley edge and ITCZ shift).
#[derive(Clone, Debug)]
pub struct Measured {
    pub measures: Measure,
    pub comparison: Comparison,
    pub rows: Vec<MetricRow>,
    pub zonal: Zonal,
}

/// Measures a coverage map (`h x w`, north-up, 0..1) against Earth: what `measure.py` does.
pub fn measure_map(reference: &Reference, cov: &[f32], h: usize, w: usize, target: Option<(f64, f64)>) -> Measured {
    let (measured, mh, mw) = measured_image(cov, h, w);
    let measures = measure(&measured, mh, mw);
    let comparison = compare(&measures, &reference.measures);
    let (lat_c, zonal_gen) = zonal_profile(cov, h, w, DEFAULT_BAND_DEG);
    let zonal = Zonal {
        lat_deg: lat_c,
        generated: zonal_gen.iter().map(|&x| f64::from(x)).collect(),
        earth: reference.measures.zonal.clone(),
        target: target.map(|(hadley, itcz)| target_zonal(h, hadley, itcz)).unwrap_or_default(),
    };
    let rows = metric_rows(&measures, &comparison, &reference.measures);
    Measured { measures, comparison, rows, zonal }
}

// ---------------------------------------------------------------------------------------------------------
// The compute block

/// The devices an answer lists: the engine's, and the CPU too when the engine did not list it (a GPU run still spends CPU time
/// on the plans, the metrics and the encodings: the reply shows where the time went, with share 0.0).
fn listed_devices(devices: &[DeviceUse], cpu_name: Option<&str>) -> Vec<DeviceUse> {
    let mut listed = devices.to_vec();
    if !listed.iter().any(|d| d.kind == DeviceKind::Cpu) {
        listed.push(DeviceUse { kind: DeviceKind::Cpu, name: cpu_name.map(str::to_string), seconds: 0.0, share: 0.0 });
    }
    listed
}

/// The `compute` block: what was requested and what ran, with the seconds each device worked (`seconds`: None for
/// the sidecar, which stays byte-for-byte the same for the same inputs whichever device ran them).
pub fn compute_json(requested: &str, used: &ComputeUse, seconds: Option<f64>) -> J {
    // One entry per device, the larger share first (the CPU first on a tie): a reader finds the device that did the work first.
    let mut ordered: Vec<&DeviceUse> = used.devices.iter().collect();
    ordered.sort_by(|a, b| b.share.total_cmp(&a.share).then((a.kind != DeviceKind::Cpu).cmp(&(b.kind != DeviceKind::Cpu))));
    let devices = ordered
        .into_iter()
        .map(|d| {
            let mut items = vec![("kind", J::s(d.kind.key())), ("name", opt_str(d.name.as_deref()))];
            if let Some(total) = seconds {
                // The server's own work (match, encodings, metrics) runs on the CPU: the CPU device carries it.
                let engine_total: f64 = used.devices.iter().map(|d| d.seconds).sum();
                let extra = if d.kind == DeviceKind::Cpu { (total - engine_total).max(0.0) } else { 0.0 };
                items.push(("seconds", J::Float(py_round(d.seconds + extra, 2))));
            }
            items.push(("share", J::Float(py_round(d.share, 4))));
            J::obj(items)
        })
        .collect();
    J::obj([("requested", J::s(requested)), ("used", J::s(used.used.key())), ("devices", J::Arr(devices)), ("note", opt_str(used.note.as_deref()))])
}

// ---------------------------------------------------------------------------------------------------------
// One generation

/// A progress report: the fraction of the whole run (never decreasing) and the stage.
pub type Progress<'a> = &'a dyn Fn(f64, &str);
/// Polled at stage boundaries: true means stop.
pub type Cancel<'a> = &'a dyn Fn() -> bool;

/// What to make and how to name it.
#[derive(Clone, Debug)]
pub struct Job {
    pub body: BodyInputs,
    pub seed: u64,
    pub width: usize,
    pub options: MapOptions,
    pub compute: ComputeMode,
    /// The stem of the encoded files' names (`<stem>_<width>x<height>_16bit.png`...), which the sidecar records.
    pub stem: String,
    /// Keep the wall-clock `seconds` in the sidecar (`ui/server.py` does, the MCP service does not: its files stay
    /// byte-identical for the same inputs).
    pub sidecar_seconds: bool,
}

/// Why a generation did not finish.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    /// The request cannot be run as asked: the message says why (HTTP 400).
    Refused(String),
    /// The run failed (HTTP 500).
    Failed(String),
    /// `cancel` returned true at a stage boundary.
    Cancelled,
}

impl From<EngineError> for RunError {
    fn from(e: EngineError) -> RunError {
        match e {
            EngineError::Refused(m) => RunError::Refused(m),
            EngineError::Failed(m) => RunError::Failed(m),
        }
    }
}

/// The encoded files of a map, as bytes.
#[derive(Clone, Debug)]
pub struct Files {
    /// The 8-bit preview (`<stem>_<w>x<h>.png`).
    pub coverage_png: Vec<u8>,
    pub type_png: Vec<u8>,
    pub high_share_png: Vec<u8>,
    pub wind_png: Vec<u8>,
    /// The 16-bit master.
    pub master16_png: Vec<u8>,
    /// The raw R8 for the shader, rows south-up, gzip.
    pub r8_gz: Vec<u8>,
    /// The sidecar JSON text.
    pub meta_json: Vec<u8>,
}

/// The file names the encodings carry, from the stem.
#[derive(Clone, Debug)]
pub struct FileNames {
    pub master: String,
    pub preview: String,
    pub r8gz: String,
    pub meta: String,
    pub type_: String,
    pub high_share: String,
    pub wind: String,
}

/// A finished map.
#[derive(Clone, Debug)]
pub struct Generation {
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    /// What the caller asked for.
    pub requested: ComputeMode,
    /// What ran it: the note of the choice and the engine's own joined, the CPU listed on every run.
    pub compute: ComputeUse,
    /// The engine's output: the raw coverage (before the match to Earth), the sidecar numbers and the derived maps.
    pub output: GenOutput,
    /// The final coverage (after the match to Earth, when asked).
    pub coverage: Vec<f32>,
    /// The sidecar, as written to `meta_json`.
    pub sidecar: J,
    pub files: Files,
    pub names: FileNames,
    pub measured: Measured,
    /// The notes about the body and the options (the API's wording).
    pub warnings: Vec<String>,
    /// The note about the width the metrics were measured at, when it is narrow.
    pub metrics_note: Option<String>,
    /// Wall seconds of the engine alone.
    pub engine_seconds: f64,
    /// Wall seconds of the whole run (engine, match, encodings, metrics).
    pub seconds: f64,
}

struct Stages<'a> {
    progress: Progress<'a>,
    cancel: Cancel<'a>,
    last: f64,
}

impl Stages<'_> {
    /// Polls `cancel`, then reports.
    fn stage(&mut self, fraction: f64, message: &str) -> Result<(), RunError> {
        if (self.cancel)() {
            return Err(RunError::Cancelled);
        }
        self.report(fraction, message);
        Ok(())
    }

    fn report(&mut self, fraction: f64, message: &str) {
        self.last = self.last.max(fraction.min(1.0));
        (self.progress)(self.last, message);
    }
}

/// Where the engine's own 0..1 progress lands on the scale of the whole run.
const ENGINE_START: f64 = 0.06;
const ENGINE_END: f64 = 0.75;

/// The whole job, as `gen.py` then `measure.py` do it. The caller holds the generation lock (one run at a time).
pub fn run_generation(rt: &Runtime, reference: &Reference, job: &Job, progress: Progress<'_>, cancel: Cancel<'_>) -> Result<Generation, RunError> {
    let t0 = Instant::now();
    let opt = job.options;
    let mut run = Stages { progress, cancel, last: 0.0 };
    let (used, note) = rt.resolve(job.compute, job.width);
    let engine = rt.engine(used)?;
    run.stage(0.0, &format!("seed {} width {} on the {}", job.seed, job.width, used.key()))?;
    let surface = if opt.surface == SurfaceChoice::Synthetic {
        run.stage(0.03, "building the synthetic surface")?;
        Some(synthetic_surface(job.width.min(SYNTHETIC_SURFACE_MAX_WIDTH)))
    } else {
        None
    };
    run.stage(ENGINE_START, "generating the cloud field")?;
    let greq = GenerateRequest { body: job.body.clone(), seed: job.seed, width: job.width, options: opt, compute: job.compute, surface };
    let params: GenParams = greq.params();
    let t_engine = Instant::now();
    let Generated { mut output, compute } = {
        let base = run.last;
        let progress = run.progress;
        let scaled = |f: f64, msg: &str| progress(base + (ENGINE_END - base) * f.clamp(0.0, 1.0), msg);
        engine.generate(&greq, &scaled)?
    };
    run.last = run.last.max(ENGINE_END);
    let engine_seconds = t_engine.elapsed().as_secs_f64();
    // The note of the choice (auto) and the engine's own, if it has one (a device lost mid-run).
    let note = match (note, compute.note) {
        (Some(chose), Some(engine)) => Some(format!("{chose} {engine}")),
        (chose, engine) => chose.or(engine),
    };
    let compute = ComputeUse { note, devices: listed_devices(&compute.devices, rt.cpu.name.as_deref()), ..compute };
    let (height, width) = (output.height, output.width);
    if params.polar_smooth {
        cloud_map_generator_core::polar::polar_filter(&mut output.coverage, 0, height, width, height);
        if let Some(e) = output.extras.as_mut() {
            cloud_map_generator_core::polar::polar_filter(&mut e.kind, 0, height, width, height);
            cloud_map_generator_core::polar::polar_filter(&mut e.high_share, 0, height, width, height);
        }
    }

    let extras = output.extras.as_ref().ok_or_else(|| RunError::Failed("the engine returned no derived maps".to_string()))?;
    let mut cov = output.coverage.clone();
    let mut meta = gen_meta(&job.body, &params, &output.meta, engine_seconds);
    if !job.sidecar_seconds {
        meta.remove("seconds");
    }
    if opt.match_earth {
        run.stage(0.80, "matching Earth's statistics")?;
        cov = refine_to_earth(&cov, height, width, &reference.data, reference.h, reference.w, output.meta.hadley_edge_deg, DEFAULT_LEVELS, DEFAULT_STRENGTH);
        if params.polar_smooth {
            // The match stretched the rows near the poles to Earth's tables: average them again.
            cloud_map_generator_core::polar::polar_filter(&mut cov, 0, height, width, height);
        }
        meta.push("matched_to", J::s(REFERENCE));
    }
    if opt.refine {
        run.stage(0.82, "refining the clouds")?;
        let refiner = cloud_map_generator_core::refiner::Refiner::embedded().map_err(RunError::Failed)?;
        cov = refiner.refine(&cov, height, width);
        meta.push(
            "refined_by",
            J::obj([
                ("model", J::s("cloud-refiner")),
                ("version", J::Int(i64::from(cloud_map_generator_core::refiner::VERSION))),
                ("weights", J::s(cloud_map_generator_core::refiner::weights_id())),
            ]),
        );
    }
    meta.push(
        "options",
        J::obj([
            ("match_earth", J::Bool(opt.match_earth)),
            ("isotropic", J::Bool(opt.isotropic)),
            ("experiment", J::Bool(opt.experiment)),
            ("surface", J::s(opt.surface.key())),
        ].into_iter().chain(opt.refine.then(|| ("refine", J::Bool(true))))),
    );
    meta.push("compute", compute_json(job.compute.key(), &compute, None));

    // Products first: they record their encoding in the sidecar the map's files then complete.
    run.stage(0.83, "encoding the files")?;
    let products = encode_products(
        &ProductExtras {
            type_: Some(Plane::new(&extras.kind, height, width)),
            high_share: Some(Plane::new(&extras.high_share, height, width)),
            wind: Some(Plane::new(&extras.wind, extras.wind_height, extras.wind_width)),
        },
        &job.stem,
    );
    meta.push(
        "products",
        J::obj([
            ("type", J::s(products.sidecar.type_)),
            ("high_share", J::s(products.sidecar.high_share)),
            ("wind", J::s(products.sidecar.wind.as_str())),
            ("files", J::obj(products.sidecar.files.iter().map(|(k, n)| (*k, J::s(n.as_str()))))),
        ]),
    );
    let map = encode_map(&cov, height, width, &job.stem);
    let (type_img, share_img, wind_img) = (
        products.type_png.as_ref().expect("the type map was given"),
        products.high_share_png.as_ref().expect("the high-share map was given"),
        products.wind_png.as_ref().expect("the wind map was given"),
    );
    let product_name = |key: &str| products.sidecar.files.iter().find(|(k, _)| *k == key).map(|(_, n)| n.clone()).expect("a product that was given has a file name");
    // The six encodings (five PNG or gzip streams and the sidecar text) do not depend on each other: one task each.
    let mut encoded = cloud_map_generator_core::par::map_rows(6, |i| match i {
        0 => images::gray8(width, height, &map.preview8),
        1 => images::gray8(type_img.width, type_img.height, &type_img.data),
        2 => images::gray8(share_img.width, share_img.height, &share_img.data),
        3 => images::rgb8(wind_img.width, wind_img.height, &wind_img.data),
        4 => images::gray16(width, height, &map.master16),
        _ => gzip9(&map.r8_south_up),
    })
    .into_iter();
    let mut next = || encoded.next().expect("six encodings");
    let files = Files {
        coverage_png: next(),
        type_png: next(),
        high_share_png: next(),
        wind_png: next(),
        master16_png: next(),
        r8_gz: next(),
        meta_json: dumps(&meta).into_bytes(),
    };
    let names = FileNames {
        master: map.files.master.clone(),
        preview: map.files.preview.clone(),
        r8gz: map.files.r8gz.clone(),
        meta: map.files.meta.clone(),
        type_: product_name("type"),
        high_share: product_name("high_share"),
        wind: product_name("wind"),
    };

    run.stage(0.85, "measuring against Earth")?;
    let itcz = output.meta.surface.as_ref().map_or(0.0, |s| s.itcz_shift_deg);
    let measured = measure_map(reference, &cov, height, width, Some((output.meta.hadley_edge_deg, itcz)));
    let mut warnings = body_warnings(&job.body, cyclones_per_hemisphere(&job.body));
    if opt.isotropic {
        warnings.push("Isotropic baseline: no flow, no vortices; the surface is ignored. The type, high-share and wind maps still come from the same fields.".to_string());
    }
    let metrics_note = (width < REF_W).then(|| {
        format!("Measured on a {width}-px map: its finest scales are interpolated by the gnomonic patches, so spectrum and object sizes only compare at 2048 or more.")
    });
    Ok(Generation {
        width,
        height,
        seed: job.seed,
        requested: job.compute,
        compute,
        output,
        coverage: cov,
        sidecar: meta,
        files,
        names,
        measured,
        warnings,
        metrics_note,
        engine_seconds,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_map_is_measured_box_resampled_and_a_narrow_one_as_is() {
        let small = vec![0.5f32; 64 * 32];
        let (m, h, w) = measured_image(&small, 32, 64);
        assert_eq!((m.len(), h, w), (64 * 32, 32, 64));
        assert!(matches!(m, Cow::Borrowed(_)));
        let big = vec![1.0f32; 4096 * 2048];
        let (m, h, w) = measured_image(&big, 2048, 4096);
        assert_eq!((h, w, m.len()), (1024, 2048, 1024 * 2048));
        let odd = vec![1.0f32; 3000 * 1500];
        assert_eq!(measured_image(&odd, 1500, 3000).1, 1500, "a width that is not a multiple stays as it is");
    }

    #[test]
    fn numpy_round_is_half_even_at_five_places() {
        assert_eq!(round5(0.123456), 0.12346);
        assert_eq!(round5(0.000005), 0.0);
        assert_eq!(round5(0.000015), 0.00002);
        assert!(round5(f64::NAN).is_nan());
    }

    #[test]
    fn the_table_has_the_ten_rows_in_order() {
        let keys: Vec<&str> = row_defs().iter().map(|r| r.0).collect();
        assert_eq!(
            keys,
            ["mean", "zonal_rmse", "hist_l1", "coherence", "spectral_slope", "big_object_share", "median_object_km2", "seam_ratio", "pole_std_north", "pole_std_south"]
        );
    }

    #[test]
    fn a_cancelled_run_stops_at_the_first_stage_boundary() {
        let rt = Runtime::cpu_only("no adapter");
        let reference = Reference::from_png(crate::REFERENCE_PNG).unwrap();
        let job = Job {
            body: BodyInputs::new(Some(0.99727), 6371.0, 9.81, 8.0),
            seed: 1,
            width: 64,
            options: MapOptions::default(),
            compute: ComputeMode::Cpu,
            stem: "t".to_string(),
            sidecar_seconds: false,
        };
        let err = run_generation(&rt, &reference, &job, &|_, _| {}, &|| true).unwrap_err();
        assert_eq!(err, RunError::Cancelled);
    }

    #[test]
    fn a_small_run_reports_rising_progress_and_a_deterministic_sidecar() {
        let rt = Runtime::cpu_only("no adapter");
        let reference = Reference::from_png(crate::REFERENCE_PNG).unwrap();
        let job = Job {
            body: BodyInputs::new(Some(0.99727), 6371.0, 9.81, 8.0),
            seed: 7,
            width: 64,
            options: MapOptions { isotropic: true, ..MapOptions::default() },
            compute: ComputeMode::Cpu,
            stem: "cloudgen_s7".to_string(),
            sidecar_seconds: false,
        };
        let seen = std::cell::RefCell::new(Vec::new());
        let g = run_generation(&rt, &reference, &job, &|f, m| seen.borrow_mut().push((f, m.to_string())), &|| false).unwrap();
        let fractions: Vec<f64> = seen.borrow().iter().map(|(f, _)| *f).collect();
        assert!(fractions.windows(2).all(|w| w[0] <= w[1]), "{fractions:?}");
        assert!(fractions.first().is_some_and(|f| *f == 0.0) && fractions.last().is_some_and(|f| *f >= 0.85));
        assert_eq!((g.width, g.height, g.measured.rows.len()), (64, 32, 10));
        assert_eq!(g.names.master, "cloudgen_s7_64x32_16bit.png");
        assert!(!dumps(&g.sidecar).contains("\"seconds\""), "no wall time in a deterministic sidecar");
        assert!(g.compute.devices.iter().any(|d| d.kind == DeviceKind::Cpu));
        assert_eq!(g.measured.zonal.lat_deg.len(), 36);
        assert_eq!(g.measured.zonal.target.len(), 36);
    }
}
