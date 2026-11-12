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

/// The map widths the page offers: the largest the in-browser engine takes (4096 x 2048 floats are 32 MiB a field).
pub const WIDTHS: [usize; 4] = [512, 1024, 2048, 4096];

/// Seconds a whole generation takes on the CPU engine of the page (the workers' bands, then the encodings and the metrics
/// against Earth on one worker), per width: what the page shows as "about N s on an idle machine". MEASURED in headed Chromium
/// 157 on Windows 11, a 16-thread desktop CPU (15 workers) and an RTX 5080, 2026-11-12, `compute: cpu`, Earth, seed 42, the best
/// of three runs after the page had settled (`web/README.md` has the whole table). A visitor's machine is slower or faster.
const CPU_ESTIMATE_SECONDS: [(usize, f64); 4] = [(512, 2.0), (1024, 2.5), (2048, 3.0), (4096, 6.1)];

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
    Runtime { cpu: CpuEngine { name: None }, threads: caps.threads, gpu }
}

/// `compute` of `GET /api/defaults`, for this browser: the four modes (hybrid is listed and unavailable: the scheduler that
/// shares bands between the GPU and the workers is not built), `engine: "wasm"`, and the devices.
fn capabilities_json(rt: &Runtime, caps: &Caps) -> J {
    let workers = caps.workers;
    let modes = rt
        .modes()
        .into_iter()
        .map(|m: ModeInfo| {
            let detail = match m.mode {
                ComputeMode::Cpu => Some(format!("{workers} {}", if workers == 1 { "worker" } else { "workers" })),
                _ => m.detail.clone(),
            };
            J::obj([
                ("key", J::s(m.mode.key())),
                ("label", J::s(m.mode.label())),
                ("available", J::Bool(m.available)),
                ("reason", opt_str(m.reason.as_deref())),
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
    let req = parse_generate(&v, &WIDTHS, &rt)?;
    check_generatable(&req.body, req.options.experiment).map_err(bad)?;
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
    // The surface terms are not banded: one worker makes a synthetic world.
    let banded = req.options.surface != cloud_map_generator_service::engine::SurfaceChoice::Synthetic;
    Ok(cloud_map_generator_service::json::compact(&J::obj([
        ("width", J::Int(req.width as i64)),
        ("seed", J::Int(req.seed as i64)),
        ("requested", J::s(req.compute.key())),
        ("used", J::s(used.key())),
        ("note", opt_str(note.as_deref())),
        ("banded", J::Bool(banded)),
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

    pub fn file_bytes(&self, i: usize) -> Vec<u8> {
        self.inner.files[i].2.clone()
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
