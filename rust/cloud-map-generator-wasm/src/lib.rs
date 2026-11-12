//! cloud-map-generator-wasm: cloud-map-generator in the browser.
//!
//! The Rust core (`cloud-map-generator-core`) and the service layer's encodings and metrics (`cloud-map-generator-service`, no wgpu)
//! compiled to `wasm32-unknown-unknown` and exposed to JavaScript with `wasm-bindgen`. A GitHub Pages site cannot set
//! the headers that give a page `SharedArrayBuffer`, so there are no WebAssembly threads and no rayon here: the CPU
//! engine is a pool of Web Workers that each load their own instance of this module and make a band of rows of the
//! map ([`band`]). The page's engine adapter (`web/engine.js`) speaks the JSON of `cloud-map-generator serve`'s `/api/*` and
//! calls the functions below to answer it.
//!
//! Every function takes and returns JSON text (or typed arrays for the maps): the shapes are the ones of
//! `ui/README.md`. A function that can refuse input returns `Err(message)`; the message is the sentence the page shows.

pub mod band;
pub mod finalize;
#[cfg(target_arch = "wasm32")]
pub mod gpu;
pub mod large;
pub mod request;

use band::{Band, Setup, FIELDS};
use cloud_map_generator_service::engine::{ComputeMode, ComputeUse, CpuEngine, DeviceKind, DeviceUse, Engine, EngineError, GenerateRequest, Generated, GpuInfo, GpuState, ModeInfo, Progress, Runtime, Used};
use cloud_map_generator_service::json::{num, opt_str};
use cloud_map_generator_service::limits::MAX_CYCLONES_PER_HEMISPHERE;
use cloud_map_generator_service::physics::{build_presets, check_generatable, derive as derive_physics, limits_json, preset_json};
use cloud_map_generator_service::project::Project;
use cloud_map_generator_service::pyjson::J;
use finalize::{finish, generate_whole, now_ms, output_from_maps, Run};
use request::{bad, parse_derive, parse_generate, ApiError, GenRequest};
use serde_json::Value;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// The map widths the page offers, as the local server: up to 4096 the one-piece engine or the bands of [`band`]; from
/// 8192 the banded engine of the server ([`large`]), whose two big files (at 32768 x 16384 a 16-bit master of about
/// 1 GB and an R8 of 512 MiB before compression) are streamed to the origin private file system, as the server streams
/// them to its disk; in memory only when the browser has no such file system.
pub const WIDTHS: [usize; 7] = [512, 1024, 2048, 4096, 8192, 16384, 32768];

/// Seconds a whole generation takes on the CPU engine of the page (the workers' bands, then the encodings and the metrics
/// against Earth on one worker), per width: what the page shows as "about N s on an idle machine". MEASURED in headed Chromium
/// 157 on Windows 11, a 16-thread desktop CPU (15 workers) and an RTX 5080, 2026-11-12, `compute: cpu`, Earth, seed 42, the best
/// of three runs after the page had settled (`web/README.md` has the whole table). A visitor's machine is slower or faster.
/// 8192: the banded engine on the same machine, headless Chrome, 2026-11-12, `compute: cpu` (15 workers), one run: 33.9 s
/// (`gpu` 13.0 s, `hybrid` 22.5 s).
/// 16384 and 32768: NOT measured, the 8192 time scaled by the texels (x4, x16).
const CPU_ESTIMATE_SECONDS: [(usize, f64); 7] =
    [(512, 2.0), (1024, 2.5), (2048, 3.0), (4096, 6.1), (8192, CPU_ESTIMATE_8192), (16384, 4.0 * CPU_ESTIMATE_8192), (32768, 16.0 * CPU_ESTIMATE_8192)];
const CPU_ESTIMATE_8192: f64 = 34.0;

/// Where `auto` starts to prefer the GPU, texels wide. MEASURED on the same machine and runs as the table above: the GPU's
/// engine time (plans, kernels, readbacks; 0.13 / 0.34 / 0.30 / 0.47 s at 512 / 1024 / 2048 / 4096) beats the workers' (0.65 / 0.82 /
/// 1.2 / 2.9 s) at every width the page offers, and a whole request is shorter on the GPU at each (1.4 / 1.8 / 1.9 / 3.4 s against
/// 2.0 / 2.5 / 3.0 / 6.1 s: the encodings and the metrics, 1.3 s and more, are CPU work in both). So `auto` takes the GPU from the
/// narrowest width. Re-measure on an integrated GPU, where the crossover may move up.
const GPU_AUTO_MIN_WIDTH: usize = 512;

// ---------------------------------------------------------------------------------------------------------
// What the page found out about its machine

/// The facts the page's probe gives: the workers it will start, the hardware threads, and the WebGPU adapter.
struct Caps {
    threads: usize,
    workers: usize,
    gpu: Option<(String, String, Option<String>)>,
    gpu_reason: String,
}

pub(crate) fn parse_caps(text: &str) -> Result<Caps, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("capabilities: not JSON: {e}"))?;
    let count = |key: &str| v.get(key).and_then(Value::as_u64).map_or(1, |n| n.max(1) as usize);
    let gpu = v.get("gpu").filter(|g| g.is_object()).map(|g| {
        let text = |k: &str| g.get(k).and_then(Value::as_str).map(str::to_string);
        (text("name").unwrap_or_else(|| "WebGPU adapter".to_string()), text("backend").unwrap_or_else(|| "WebGPU".to_string()), text("df64"))
    });
    let gpu_reason = v.get("gpu_reason").and_then(Value::as_str).unwrap_or("This browser does not offer a GPU to the page.").to_string();
    Ok(Caps { threads: count("threads"), workers: count("workers"), gpu, gpu_reason })
}

/// Stands for the page's WebGPU pipeline in the service's `Runtime`, which only needs to know that it exists (the page runs it).
struct BrowserGpu;

impl Engine for BrowserGpu {
    fn generate(&self, _req: &GenerateRequest, _progress: Progress<'_>) -> Result<Generated, EngineError> {
        Err(EngineError::Failed("the GPU pipeline runs in the page, not in this module".to_string()))
    }
    fn min_auto_width(&self) -> usize {
        GPU_AUTO_MIN_WIDTH
    }
}

fn runtime(caps: &Caps) -> Runtime {
    let gpu = match &caps.gpu {
        Some((name, backend, df64)) => GpuState::Found {
            info: GpuInfo { name: name.clone(), backend: backend.clone(), float64: false, df64: df64.clone() },
            engine: Ok(Arc::new(BrowserGpu)),
        },
        None => GpuState::Absent { reason: caps.gpu_reason.clone() },
    };
    Runtime { cpu: CpuEngine { name: None, workers: Some(caps.workers) }, threads: caps.threads, gpu }
}

/// Why `hybrid` is unavailable in a page with a single Web Worker (the service says "a single thread"; here the unit is
/// the worker). With a GPU and two workers or more, the page shares the bands between them ([`large`]).
const BROWSER_HYBRID_REASON: &str = "This browser gives the page a single Web Worker, so there is nothing to share with the GPU.";

/// The service's refusal of `hybrid`, in the browser's words (the capabilities and the 400 of a request say the same).
fn browser_reason(reason: &str) -> String {
    reason.replace(cloud_map_generator_service::engine::HYBRID_ONE_THREAD_REASON, BROWSER_HYBRID_REASON)
}

/// `compute` of `GET /api/defaults`, for this browser: the four modes (CPU + GPU when there is a GPU and more than one
/// worker), `engine: "wasm"`, and the devices.
fn capabilities_json(rt: &Runtime, caps: &Caps) -> J {
    let workers = caps.workers;
    let modes = rt
        .modes()
        .into_iter()
        .map(|m: ModeInfo| {
            let detail = match m.mode {
                ComputeMode::Cpu => Some(format!("{workers} {}", if workers == 1 { "worker" } else { "workers" })),
                // The GPU is driven from one of the workers; the others take bands.
                ComputeMode::Hybrid if m.available => Some(format!("GPU + {} {}", workers - 1, if workers == 2 { "worker" } else { "workers" })),
                _ => m.detail.clone(),
            };
            J::obj([
                ("key", J::s(m.mode.key())),
                ("label", J::s(m.mode.label())),
                ("available", J::Bool(m.available)),
                ("reason", opt_str(m.reason.as_deref().map(browser_reason).as_deref())),
                ("description", J::s(m.mode.description())),
                ("detail", opt_str(detail.as_deref())),
            ])
        })
        .collect();
    let gpu = match &rt.gpu {
        GpuState::Found { info, .. } => J::obj([
            ("name", J::s(info.name.as_str())),
            ("backend", J::s(info.backend.as_str())),
            ("float64", J::Bool(info.float64)),
            ("df64", opt_str(info.df64.as_deref())),
        ]),
        GpuState::Absent { .. } => J::Null,
    };
    J::obj([
        ("default", J::s(ComputeMode::Auto.key())),
        ("engine", J::s("wasm")),
        ("modes", J::Arr(modes)),
        // A browser never tells the page its processor's name.
        ("devices", J::obj([("cpu", J::obj([("name", J::Null), ("threads", J::Int(caps.threads as i64))])), ("gpu", gpu)])),
    ])
}

fn estimates() -> J {
    J::obj(WIDTHS.iter().map(|w| {
        let seconds = CPU_ESTIMATE_SECONDS.iter().find(|(k, _)| k == w).map_or(0.0, |&(_, s)| s);
        (w.to_string(), num(seconds))
    }))
}

// ---------------------------------------------------------------------------------------------------------
// The API: defaults, derive, plan

/// Makes a panic of the module readable in the page's console (the default is an anonymous "unreachable").
#[wasm_bindgen]
pub fn init() {
    std::panic::set_hook(Box::new(|info| {
        let message = format!("cloud-map-generator-wasm panicked: {info}");
        #[cfg(target_arch = "wasm32")]
        console::error(&message);
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!("{message}");
    }));
}

#[cfg(target_arch = "wasm32")]
mod console {
    use wasm_bindgen::prelude::wasm_bindgen;
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        pub fn error(message: &str);
    }
}

/// The version of the crate (the page's version badge).
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// `GET /api/defaults`, from the machine's `caps` (JSON: `threads`, `workers`, `gpu`, `gpu_reason`).
#[wasm_bindgen]
pub fn defaults(caps_json: &str) -> Result<String, String> {
    let caps = parse_caps(caps_json)?;
    let rt = runtime(&caps);
    // The presets of the local server's page, without its bodies file (nothing to read in a browser): the file adds
    // no preset that the built-in list does not already have.
    let presets = build_presets(&Project { root: PathBuf::from("/") });
    let answer = J::obj([
        ("version", J::s(env!("CARGO_PKG_VERSION"))),
        ("backend", J::s("wasm")),
        ("presets", J::Arr(presets.iter().map(preset_json).collect())),
        ("widths", J::Arr(WIDTHS.iter().map(|&w| J::Int(w as i64)).collect())),
        ("limits", limits_json()),
        ("max_cyclones_per_hemisphere", J::Int(MAX_CYCLONES_PER_HEMISPHERE as i64)),
        ("estimate_seconds", estimates()),
        ("compute", capabilities_json(&rt, &caps)),
    ]);
    Ok(cloud_map_generator_service::json::compact(&answer))
}

/// `POST /api/derive`: the physics numbers of a body. `Err` is the 400's message.
#[wasm_bindgen]
pub fn derive(request_json: &str) -> Result<String, String> {
    let v: Value = serde_json::from_str(request_json).map_err(|e| format!("invalid JSON: {e}"))?;
    let body = parse_derive(&v).map_err(|e| e.message)?;
    Ok(cloud_map_generator_service::json::compact(&derive_physics(&body, &WIDTHS)))
}

pub(crate) fn parse(request_json: &str, caps: &Caps) -> Result<GenRequest, ApiError> {
    let v: Value = serde_json::from_str(request_json).map_err(|e| bad(format!("invalid JSON: {e}")))?;
    let rt = runtime(caps);
    let req = parse_generate(&v, &WIDTHS, &rt).map_err(|e| ApiError { message: browser_reason(&e.message), ..e })?;
    check_generatable(&req.body, req.options.experiment).map_err(bad)?;
    // What the server's banded engine refuses (`large::check_large`): the match to Earth needs the whole image.
    if large::is_large(req.width, req.compute) && req.options.match_earth {
        return Err(bad("match_earth needs the whole image in memory: it is not offered for banded maps (a width of 8192, or CPU + GPU); use a width of 4096 or less with CPU or GPU"));
    }
    Ok(req)
}

/// Validates a generate request BEFORE any work (the contract: a refused request costs nothing) and says how the engine will
/// run it: `{"width", "seed", "requested", "used", "note", "bands"}`. `Err` is the 400's message.
#[wasm_bindgen]
pub fn plan(request_json: &str, caps_json: &str) -> Result<String, String> {
    let caps = parse_caps(caps_json)?;
    let req = parse(request_json, &caps).map_err(|e| e.message)?;
    let rt = runtime(&caps);
    let (used, note) = rt.resolve(req.compute, req.width);
    // The surface terms are not banded by the page's own bands: one worker makes a synthetic world.
    let banded = req.options.surface != cloud_map_generator_service::engine::SurfaceChoice::Synthetic;
    // The server's banded engine: 8192 wide, or CPU + GPU at any width (as `cloud-map-generator serve`).
    let large = large::is_large(req.width, req.compute);
    // As the server's GPU band worker refuses (`GpuBandState::new`), before any work: a 400, not a failure midway.
    let params = finalize::params_of(&req.options);
    if large && used != Used::Cpu && !params.isotropic && params.structure_width.min(req.width) == req.width {
        return Err(format!("a {}-px map is as narrow as the back-trace grid: the GPU bands need a wider one", req.width));
    }
    let units = match used {
        Used::Cpu => caps.workers,
        Used::Gpu => 1,
        Used::Hybrid => caps.workers,
    };
    let band_rows = large::band_rows_for(req.width, used, units);
    Ok(cloud_map_generator_service::json::compact(&J::obj([
        ("width", J::Int(req.width as i64)),
        ("seed", J::Int(req.seed as i64)),
        ("requested", J::s(req.compute.key())),
        ("used", J::s(used.key())),
        ("note", opt_str(note.as_deref())),
        ("banded", J::Bool(banded)),
        ("large", J::Bool(large)),
        ("band_rows", J::Int(band_rows as i64)),
        ("bands", J::Int(large::band_count(req.width, band_rows) as i64)),
    ])))
}

/// Computes Earth's reference measures now (about two seconds in WebAssembly) so the first map does not wait for them.
#[wasm_bindgen]
pub fn warm() -> Result<(), String> {
    finalize::reference().map(|_| ())
}

/// Earth's reference map (`GET /api/reference/coverage.png`).
#[wasm_bindgen]
pub fn reference_png() -> Vec<u8> {
    cloud_map_generator_service::REFERENCE_PNG.to_vec()
}

// ---------------------------------------------------------------------------------------------------------
// The CPU engine: one band per worker

/// The cuts of a map `height` rows tall into `bands` bands, flat: `[r0, r1, r0, r1, ...]`.
#[wasm_bindgen]
pub fn row_cuts(height: usize, bands: usize) -> Vec<u32> {
    band::row_cuts(height, bands).into_iter().flat_map(|(a, b)| [a as u32, b as u32]).collect()
}

/// Barrier 1: merges the moments of the bands (each [`FIELDS`] x 3 numbers, concatenated in band order) into the
/// standardization constants (`2 x FIELDS` numbers).
#[wasm_bindgen]
pub fn merge_moments(flat: &[f64]) -> Vec<f64> {
    let stride = FIELDS * band::MOMENT_STRIDE;
    let bands: Vec<Vec<f64>> = flat.chunks_exact(stride).map(<[f64]>::to_vec).collect();
    band::merge_flat(&bands)
}

/// Barrier 2: the row thresholds smoothed in latitude.
#[wasm_bindgen]
pub fn smooth_thresholds(thr_raw: &[f64]) -> Vec<f64> {
    band::smooth_thresholds(thr_raw, &cloud_map_generator_core::generate::GenParams::default())
}

/// One band of one map, held by a worker between the passes (see [`band`]).
#[wasm_bindgen]
pub struct BandJob {
    setup: Rc<Setup>,
    r0: usize,
    r1: usize,
    band: Option<Band>,
    output: Option<band::BandOutput>,
}

#[wasm_bindgen]
impl BandJob {
    /// Draws the plans of the request and takes rows `r0..r1`. `Err` if the request is not one `plan` accepts.
    #[wasm_bindgen(constructor)]
    pub fn new(request_json: &str, caps_json: &str, r0: usize, r1: usize) -> Result<BandJob, String> {
        let caps = parse_caps(caps_json)?;
        let req = parse(request_json, &caps).map_err(|e| e.message)?;
        let params = finalize::params_of(&req.options);
        let height = req.width / 2;
        if r0 >= r1 || r1 > height {
            return Err(format!("band {r0}..{r1} is not inside the {height} rows of the map"));
        }
        let setup = Rc::new(Setup::new(&req.body, req.seed, req.width, &params, r0, r1));
        Ok(BandJob { setup, r0, r1, band: None, output: None })
    }

    /// Pass A. Returns the moments of the band (`FIELDS` x 3 numbers).
    pub fn prepare(&mut self) -> Vec<f64> {
        let (band, moments) = Band::prepare(&self.setup, self.r0, self.r1);
        self.band = Some(band);
        band::moments_to_flat(&moments)
    }

    /// Pass B with the constants of barrier 1. Returns the raw thresholds of the band's rows.
    pub fn solve(&mut self, constants: &[f64]) -> Result<Vec<f64>, String> {
        let band = self.band.as_mut().ok_or("solve before prepare")?;
        Ok(band.solve(constants))
    }

    /// Pass C with the smoothed thresholds of the whole map (`height` numbers). The rows are read with the getters.
    pub fn compose(&mut self, thr: &[f64]) -> Result<(), String> {
        if thr.len() != self.setup.height {
            return Err(format!("the thresholds must hold one value per row ({}), got {}", self.setup.height, thr.len()));
        }
        let band = self.band.as_mut().ok_or("compose before solve")?;
        self.output = Some(band.compose(thr));
        Ok(())
    }

    /// The band's coverage rows (north-up, 0..1); empty before `compose`.
    pub fn coverage(&mut self) -> Vec<f32> {
        self.output.as_mut().map(|o| std::mem::take(&mut o.coverage)).unwrap_or_default()
    }

    pub fn kind(&mut self) -> Vec<f32> {
        self.output.as_mut().map(|o| std::mem::take(&mut o.kind)).unwrap_or_default()
    }

    pub fn high_share(&mut self) -> Vec<f32> {
        self.output.as_mut().map(|o| std::mem::take(&mut o.high_share)).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------------------------------------
// After the maps: encodings, measures, the answer

/// A finished generation: the answer's JSON and its seven files.
#[wasm_bindgen]
pub struct Finished {
    pub(crate) inner: finalize::Finished,
}

#[wasm_bindgen]
impl Finished {
    /// The JSON of `POST /api/generate`'s answer (the hrefs are `/api/result/<id>/<name>`).
    pub fn json(&self) -> String {
        self.inner.json.clone()
    }

    /// Wall seconds of the engine alone (without the encodings and the metrics).
    pub fn engine_seconds(&self) -> f64 {
        self.inner.engine_seconds
    }

    pub fn file_count(&self) -> usize {
        self.inner.files.len()
    }

    /// The name of file `i`: `coverage.png`, `type.png`, `high_share.png`, `wind.png`, `master16.png`, `coverage.r8.gz`, `meta.json`.
    pub fn file_name(&self, i: usize) -> String {
        self.inner.files[i].0.to_string()
    }

    pub fn file_type(&self, i: usize) -> String {
        self.inner.files[i].1.to_string()
    }

    /// The bytes of file `i`, handed over (a second call gets nothing): a banded map's files are hundreds of MB, never
    /// held twice.
    pub fn file_bytes(&mut self, i: usize) -> Vec<u8> {
        std::mem::take(&mut self.inner.files[i].2)
    }
}

fn devices_from(text: &str) -> Result<Vec<DeviceUse>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("devices: not JSON: {e}"))?;
    v.as_array()
        .ok_or("devices: expected a list")?
        .iter()
        .map(|d| {
            let kind = match d.get("kind").and_then(Value::as_str) {
                Some("cpu") => DeviceKind::Cpu,
                Some("gpu") => DeviceKind::Gpu,
                _ => return Err("devices: kind must be cpu or gpu".to_string()),
            };
            Ok(DeviceUse {
                kind,
                name: d.get("name").and_then(Value::as_str).map(str::to_string),
                seconds: d.get("seconds").and_then(Value::as_f64).unwrap_or(0.0),
                share: d.get("share").and_then(Value::as_f64).unwrap_or(0.0),
            })
        })
        .collect()
}

fn used_from(key: &str) -> Result<Used, String> {
    match key {
        "cpu" => Ok(Used::Cpu),
        "gpu" => Ok(Used::Gpu),
        "hybrid" => Ok(Used::Hybrid),
        other => Err(format!("used: expected cpu, gpu or hybrid, got {other}")),
    }
}

/// The answer for maps made elsewhere (the bands of the workers, or the GPU): `coverage`, `kind` and `high_share` are the
/// `height x width` maps in row order. `devices_json` is `[{"kind", "name", "seconds", "share"}]` as measured by the page;
/// `engine_seconds` the wall time of the engine alone; `elapsed_before` the wall seconds since the request began.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn finalize_maps(
    request_json: &str,
    caps_json: &str,
    id: &str,
    coverage: Vec<f32>,
    kind: Vec<f32>,
    high_share: Vec<f32>,
    used: &str,
    devices_json: &str,
    note: Option<String>,
    engine_seconds: f64,
    elapsed_before: f64,
) -> Result<Finished, String> {
    let caps = parse_caps(caps_json)?;
    let req = parse(request_json, &caps).map_err(|e| e.message)?;
    let n = req.width * (req.width / 2);
    if coverage.len() != n || kind.len() != n || high_share.len() != n {
        return Err(format!("the maps must hold {n} values, got {}, {} and {}", coverage.len(), kind.len(), high_share.len()));
    }
    let compute = ComputeUse { used: used_from(used)?, devices: devices_from(devices_json)?, note };
    let output = output_from_maps(&req, coverage, kind, high_share);
    let inner = finish(Run { req: &req, id, compute, engine_seconds, elapsed_before }, output)?;
    Ok(Finished { inner })
}

/// A whole generation in this instance, for what the bands do not cover (a synthetic surface). The CPU is the one device.
#[wasm_bindgen]
pub fn run_whole(request_json: &str, caps_json: &str, id: &str, note: Option<String>, elapsed_before: f64) -> Result<Finished, String> {
    let caps = parse_caps(caps_json)?;
    let req = parse(request_json, &caps).map_err(|e| e.message)?;
    let t0 = now_ms();
    let output = generate_whole(&req)?;
    let engine_seconds = (now_ms() - t0) / 1000.0;
    let compute = ComputeUse {
        used: Used::Cpu,
        devices: vec![DeviceUse { kind: DeviceKind::Cpu, name: None, seconds: engine_seconds, share: 1.0 }],
        note,
    };
    let inner = finish(Run { req: &req, id, compute, engine_seconds, elapsed_before: elapsed_before + engine_seconds }, output)?;
    Ok(Finished { inner })
}

// ---------------------------------------------------------------------------------------------------------
// The banded engine of the server, over the page's workers (see [`large`])

/// Opens the banded run of a request in this worker, cut in bands of `band_rows` rows (the `band_rows` of `plan`).
#[wasm_bindgen]
pub fn large_open(request_json: &str, caps_json: &str, band_rows: usize) -> Result<(), String> {
    let caps = parse_caps(caps_json)?;
    let req = parse(request_json, &caps).map_err(|e| e.message)?;
    large::open(req, band_rows)
}

#[wasm_bindgen]
pub fn large_close() {
    large::close();
}

/// Rows `[r0, r1)` of band `band` of the open run.
#[wasm_bindgen]
pub fn large_band_range(band: usize) -> Result<Vec<u32>, String> {
    let b = large::banded()?;
    let (r0, r1) = large::check_band(&b, band)?;
    Ok(vec![r0 as u32, r1 as u32])
}

/// Pass A of a band on this worker's CPU: its moments, flat.
#[wasm_bindgen]
pub fn large_pass_a(band: usize) -> Result<Vec<f64>, String> {
    large::cpu_pass_a(band)
}

/// Barrier 1: the moments of every band, flat and in band order.
#[wasm_bindgen]
pub fn large_set_moments(flat: &[f64]) -> Result<(), String> {
    large::set_moments(flat)
}

/// Pass B of a band on this worker's CPU: `[anomaly_min, anomaly_max, rows, sensitivity rows, thr_raw..., sensitivity...]`.
#[wasm_bindgen]
pub fn large_pass_b(band: usize) -> Result<Vec<f64>, String> {
    large::cpu_pass_b(band)
}

/// Barrier 2: the raw thresholds and the sensitivity of every row, north to south.
#[wasm_bindgen]
pub fn large_set_thresholds(thr_raw: &[f64], sensitivity: Vec<f64>) -> Result<(), String> {
    large::set_thresholds(thr_raw, sensitivity)
}

/// Pass C of a band on this worker's CPU: the coverage rows, then the type and high-share rows when the run makes them.
#[wasm_bindgen]
pub fn large_pass_c(band: usize) -> Result<Vec<f32>, String> {
    large::cpu_pass_c(band)
}

/// `disk`: the page writes the master and the R8 to files (see [`large`]); otherwise they are kept in memory.
#[wasm_bindgen]
pub fn large_sink_begin(disk: bool) -> Result<(), String> {
    large::sink_begin(disk)
}

/// On disk: the master's bytes made since the last call.
#[wasm_bindgen]
pub fn large_sink_take_master() -> Result<Vec<u8>, String> {
    large::sink_take_master()
}

/// On disk: the last band's R8 rows.
#[wasm_bindgen]
pub fn large_sink_take_r8() -> Result<Vec<u8>, String> {
    large::sink_take_r8().map(|(b, _)| b)
}

/// On disk: where the last band's R8 rows go (bytes from the start of the file).
#[wasm_bindgen]
pub fn large_sink_r8_offset() -> Result<f64, String> {
    large::sink_r8_offset().map(|o| o as f64)
}

/// On disk, after the last band: the master's last bytes.
#[wasm_bindgen]
pub fn large_sink_end_master() -> Result<Vec<u8>, String> {
    large::sink_end_master()
}

#[wasm_bindgen]
pub fn large_gz_begin() {
    large::gz_begin();
}

#[wasm_bindgen]
pub fn large_gz_feed(chunk: &[u8]) -> Result<Vec<u8>, String> {
    large::gz_feed(chunk)
}

#[wasm_bindgen]
pub fn large_gz_finish() -> Result<Vec<u8>, String> {
    large::gz_finish()
}

/// A finished band, in band order (the page orders them).
#[wasm_bindgen]
pub fn large_sink_band(band: usize, data: &[f32]) -> Result<(), String> {
    large::sink_band(band, data)
}

/// The end of the run on the sink worker: the files and the answer. `devices_json` as for [`finalize_maps`];
/// `anomaly_min` / `anomaly_max` the range of the surface anomaly over the bands of pass B (NaN without a surface).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn large_finish(
    id: &str,
    used: &str,
    devices_json: &str,
    note: Option<String>,
    anomaly_min: f32,
    anomaly_max: f32,
    engine_seconds: f64,
    elapsed_before: f64,
) -> Result<Finished, String> {
    let compute = ComputeUse { used: used_from(used)?, devices: devices_from(devices_json)?, note };
    let anomaly = (anomaly_min.is_finite() && anomaly_max.is_finite()).then_some([anomaly_min, anomaly_max]);
    let inner = large::finish(id, compute, anomaly, engine_seconds, elapsed_before)?;
    Ok(Finished { inner })
}

#[cfg(test)]
mod tests {
    use super::*;

    const WITH_GPU: &str = r#"{"threads": 16, "workers": 15, "gpu": {"name": "NVIDIA Blackwell", "backend": "WebGPU", "df64": "fma"}}"#;

    const ONE_WORKER: &str = r#"{"threads": 2, "workers": 1, "gpu": {"name": "NVIDIA Blackwell", "backend": "WebGPU", "df64": "fma"}}"#;
    const HYBRID: &str = r#"{"body": {"rotation_period_days": 0.99727, "radius_km": 6371, "gravity_mps2": 9.81, "scale_height_km": 8}, "seed": 1, "width": 1024, "options": {"compute": "hybrid"}}"#;

    fn mode(caps: &str, key: &str) -> Value {
        let d: Value = serde_json::from_str(&defaults(caps).unwrap()).unwrap();
        d["compute"]["modes"].as_array().unwrap().iter().find(|m| m["key"] == key).unwrap().clone()
    }

    #[test]
    fn hybrid_is_offered_with_a_gpu_and_workers_and_runs_on_the_banded_engine() {
        let hybrid = mode(WITH_GPU, "hybrid");
        assert_eq!((hybrid["available"].clone(), hybrid["detail"].clone()), (Value::Bool(true), Value::from("GPU + 14 workers")));
        let p: Value = serde_json::from_str(&plan(HYBRID, WITH_GPU).unwrap()).unwrap();
        assert_eq!((p["used"].clone(), p["large"].clone()), (Value::from("hybrid"), Value::Bool(true)));
        assert!(p["bands"].as_i64().unwrap() >= 15, "{p}");
        // 8192 is offered, on the banded engine, and the match to Earth is refused there as on the server.
        let big = HYBRID.replace("\"width\": 1024", "\"width\": 8192").replace("\"hybrid\"", "\"cpu\"");
        let p: Value = serde_json::from_str(&plan(&big, WITH_GPU).unwrap()).unwrap();
        assert_eq!((p["large"].clone(), p["used"].clone()), (Value::Bool(true), Value::from("cpu")));
        let matched = big.replace("\"compute\": \"cpu\"", "\"compute\": \"cpu\", \"match_earth\": true");
        assert!(plan(&matched, WITH_GPU).unwrap_err().contains("match_earth"));
        // At 512 the GPU bands are refused before any work, with the server's message.
        let narrow = HYBRID.replace("\"width\": 1024", "\"width\": 512");
        assert!(plan(&narrow, WITH_GPU).unwrap_err().contains("as narrow as the back-trace grid"));
        // 16384 and 32768 are offered, on the banded engine, as by the server.
        let d: Value = serde_json::from_str(&defaults(WITH_GPU).unwrap()).unwrap();
        assert_eq!(d["widths"], serde_json::json!([512, 1024, 2048, 4096, 8192, 16384, 32768]));
        let huge = big.replace("\"width\": 8192", "\"width\": 32768");
        let p: Value = serde_json::from_str(&plan(&huge, WITH_GPU).unwrap()).unwrap();
        assert_eq!(p["large"], true);
        assert_eq!(p["band_rows"].as_i64().unwrap() % 16, 0, "a multiple of 32768 / 2048 for the slices");
        let small = HYBRID.replace("\"hybrid\"", "\"cpu\"");
        let p: Value = serde_json::from_str(&plan(&small, WITH_GPU).unwrap()).unwrap();
        assert_eq!(p["large"], false);
    }

    #[test]
    fn hybrid_with_a_single_worker_is_refused_with_the_browsers_reason() {
        let hybrid = mode(ONE_WORKER, "hybrid");
        assert_eq!(hybrid["available"], false);
        assert_eq!(hybrid["reason"], BROWSER_HYBRID_REASON);
        let err = plan(HYBRID, ONE_WORKER).unwrap_err();
        assert!(err.contains(BROWSER_HYBRID_REASON) && !err.contains("single thread"), "{err}");
    }
}
