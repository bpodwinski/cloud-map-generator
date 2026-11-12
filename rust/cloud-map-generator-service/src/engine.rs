//! Where the work runs: the compute modes, the engines that run a map, and the capabilities the API reports.
//!
//! The contract (wire values and shapes) is the one of `cloudgen/service.py`, which the Python server also
//! implements: four modes in a fixed order (`auto`, `cpu`, `gpu`, `hybrid`), each available or refused with a
//! reason; a `devices` block; and, on every answer, a `compute` block that says what was requested, what ran
//! and where the time went.
//!
//! - `cpu`: every stage on the CPU (`cloud_map_generator_core::generate::generate`).
//! - `gpu`: the per-texel stages on the GPU. The [`Engine`] for it comes from [`crate::gpu_slot`]; until that
//!   function returns one, `gpu` is reported unavailable with the reason it gives, and asking for it is an error.
//! - `hybrid`: row bands shared between the two. Not built: always unavailable, with a reason.
//! - `auto`: the engine picks (the GPU when it exists and the map is wide enough for the upload to pay, else
//!   the CPU). It never fails for lack of a device.
//!
//! An explicit mode that cannot run is an error with the reason (HTTP 400), never a silent fallback.

use crate::pyfmt::{close_match, py_repr};
use crate::pyjson::J;
use cloud_map_generator_core::climate::BodyInputs;
use cloud_map_generator_core::generate::{generate, GenOutput, GenParams, GenerateError};
use cloud_map_generator_core::surface::Surface;
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;

// ---------------------------------------------------------------------------------------------------------
// Modes

/// The compute mode of a request (`options.compute`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeMode {
    Auto,
    Cpu,
    Gpu,
    Hybrid,
}

impl ComputeMode {
    /// The four modes, in the order of the contract.
    pub const ALL: [ComputeMode; 4] = [ComputeMode::Auto, ComputeMode::Cpu, ComputeMode::Gpu, ComputeMode::Hybrid];

    /// The wire value.
    pub fn key(self) -> &'static str {
        match self {
            ComputeMode::Auto => "auto",
            ComputeMode::Cpu => "cpu",
            ComputeMode::Gpu => "gpu",
            ComputeMode::Hybrid => "hybrid",
        }
    }

    pub fn from_key(key: &str) -> Option<ComputeMode> {
        ComputeMode::ALL.into_iter().find(|m| m.key() == key)
    }

    /// The label of the mode's button.
    pub fn label(self) -> &'static str {
        match self {
            ComputeMode::Auto => "Auto",
            ComputeMode::Cpu => "CPU",
            ComputeMode::Gpu => "GPU",
            ComputeMode::Hybrid => "CPU + GPU",
        }
    }

    /// What the mode does, in at most 160 characters: the page shows it in a narrow column under
    /// the selector (the long meaning is in `docs/compute-modes.html`). The same sentences on every engine: they
    /// describe the mode, not the engine.
    pub fn description(self) -> &'static str {
        match self {
            ComputeMode::Auto => "Picks the GPU when one is available and the map is big enough for the upload to pay, else the CPU. Never fails for lack of a device.",
            ComputeMode::Cpu => "Every stage on the CPU.",
            ComputeMode::Gpu => "The per-texel stages on the GPU in float32: the CPU map to float32 rounding, not bit for bit. Random plans, metrics and file encoding stay on the CPU.",
            ComputeMode::Hybrid => "Row bands go to whichever of the GPU or a CPU thread is free, so both work at once. Needs a GPU and more than one CPU thread.",
        }
    }
}

/// The longest `description` the contract allows, characters (a test holds the four descriptions to it).
#[cfg(test)]
pub const DESCRIPTION_MAX_CHARS: usize = 160;

/// What really ran a map (`compute.used`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Used {
    Cpu,
    Gpu,
    Hybrid,
}

impl Used {
    pub fn key(self) -> &'static str {
        match self {
            Used::Cpu => "cpu",
            Used::Gpu => "gpu",
            Used::Hybrid => "hybrid",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    // Constructed by the GPU engine that plugs into `gpu_slot` (and by the tests).
    #[allow(dead_code)]
    Gpu,
}

impl DeviceKind {
    pub fn key(self) -> &'static str {
        match self {
            DeviceKind::Cpu => "cpu",
            DeviceKind::Gpu => "gpu",
        }
    }
}

/// One device's part in a map: the wall time it worked and the fraction of the work it did (the shares of a
/// map add to 1).
#[derive(Clone, Debug, PartialEq)]
pub struct DeviceUse {
    pub kind: DeviceKind,
    pub name: Option<String>,
    pub seconds: f64,
    pub share: f64,
}

/// The `compute` block of an answer, before the server adds the time of the work it does itself.
#[derive(Clone, Debug, PartialEq)]
pub struct ComputeUse {
    pub used: Used,
    pub devices: Vec<DeviceUse>,
    pub note: Option<String>,
}

// ---------------------------------------------------------------------------------------------------------
// The request an engine runs

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceChoice {
    None,
    Synthetic,
}

impl SurfaceChoice {
    pub fn key(self) -> &'static str {
        match self {
            SurfaceChoice::None => "none",
            SurfaceChoice::Synthetic => "synthetic",
        }
    }
}

/// The options of the map (`options` without `compute`: where the work runs is not a property of the map).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MapOptions {
    pub match_earth: bool,
    /// The cloud refiner (`cloud_map_generator_core::refiner`): a trained network adds the fine structure of real clouds. Off by default.
    pub refine: bool,
    pub isotropic: bool,
    pub experiment: bool,
    pub surface: SurfaceChoice,
}

impl Default for MapOptions {
    fn default() -> MapOptions {
        MapOptions { match_earth: false, refine: false, isotropic: false, experiment: false, surface: SurfaceChoice::None }
    }
}

/// One map to make. `surface` is the land and relief the server built from `options.surface` (the surface
/// terms are CPU work on every engine: they draw nothing random and cost milliseconds).
#[derive(Clone, Debug)]
pub struct GenerateRequest {
    pub body: BodyInputs,
    pub seed: u64,
    pub width: usize,
    pub options: MapOptions,
    pub compute: ComputeMode,
    pub surface: Option<Surface>,
}

impl GenerateRequest {
    /// The generator's knobs for this request (`gen.py`'s `GenParams(isotropic=..., allow_any_regime=...)`).
    pub fn params(&self) -> GenParams {
        GenParams { isotropic: self.options.isotropic, allow_any_regime: self.options.experiment, ..GenParams::realistic() }
    }
}

/// What an engine returns: the coverage map with its derived maps and sidecar numbers, and where it ran.
#[derive(Clone, Debug)]
pub struct Generated {
    pub output: GenOutput,
    pub compute: ComputeUse,
}

/// Why an engine did not make the map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineError {
    /// The request cannot be run as asked (HTTP 400): the message says why.
    Refused(String),
    /// The run failed (HTTP 500).
    // Returned by the GPU engine that plugs into `gpu_slot` (a device lost, a kernel that failed).
    #[allow(dead_code)]
    Failed(String),
}

impl From<GenerateError> for EngineError {
    fn from(e: GenerateError) -> EngineError {
        EngineError::Refused(e.to_string())
    }
}

/// Progress of a run: a fraction in `[0, 1]` and what is happening (the server logs them).
pub type Progress<'a> = &'a dyn Fn(f64, &str);

/// Something that can make a map. The server holds one per device kind and picks by the compute mode.
pub trait Engine: Send + Sync {
    /// Makes the map of `req` (its coverage, the type, high-share and wind maps, the flow and surface numbers of
    /// the sidecar) and says what ran it. The server does everything after that: the match to Earth, the
    /// encodings and the measures.
    fn generate(&self, req: &GenerateRequest, progress: Progress<'_>) -> Result<Generated, EngineError>;

    /// The narrowest map, in texels, for which `auto` prefers this engine (the upload of a small map costs
    /// more than it saves). A GPU engine puts its measured crossover here.
    fn min_auto_width(&self) -> usize {
        0
    }

    /// The engine's worker for the banded passes of a large map, borrowing the run's [`Banded`] setup; `Err` when
    /// this engine cannot run bands (the CPU has none here: the scheduler's CPU worker is `bands::CpuBands`).
    fn band_worker<'a>(&'a self, _banded: &'a cloud_map_generator_core::banded::Banded<'a>) -> Result<Box<dyn crate::bands::BandCompute + 'a>, String> {
        Err("this engine cannot run bands".to_string())
    }

    /// The threads this engine uses (the `detail` of its mode).
    fn threads(&self) -> usize {
        1
    }
}

/// The CPU engine: `cloud_map_generator_core::generate::generate`, on every core the process was given (`--threads`, then
/// `CLOUD_MAP_GENERATOR_THREADS`, then all logical cores; one thread when the build lacks the `rayon` feature). The result is
/// bit-identical for any thread count.
pub struct CpuEngine {
    pub name: Option<String>,
    /// The CPU units that work at once when they are not this process's thread pool: the Web Workers of the browser's
    /// engine, each one WebAssembly thread. `None`: the pool's threads.
    pub workers: Option<usize>,
}

impl Engine for CpuEngine {
    fn generate(&self, req: &GenerateRequest, progress: Progress<'_>) -> Result<Generated, EngineError> {
        progress(0.0, &format!("generating a {} px map on the CPU (compute mode {})", req.width, req.compute.key()));
        let t0 = Instant::now();
        let output = generate(&req.body, req.seed, req.width, &req.params(), req.surface.as_ref(), true)?;
        let seconds = t0.elapsed().as_secs_f64();
        progress(1.0, &format!("generated in {seconds:.2} s"));
        Ok(Generated {
            output,
            compute: ComputeUse {
                used: Used::Cpu,
                devices: vec![DeviceUse { kind: DeviceKind::Cpu, name: self.name.clone(), seconds, share: 1.0 }],
                note: None,
            },
        })
    }

    fn threads(&self) -> usize {
        self.workers.unwrap_or_else(cloud_map_generator_core::par::threads)
    }
}

// ---------------------------------------------------------------------------------------------------------
// The machine and its capabilities

/// The GPU adapter found at start-up, as `devices.gpu` reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
    /// Whether the device runs f64 shaders. The engine opens its device with no optional feature and the kernels
    /// are f32 (with double-single for the few places that need more), so this is false.
    pub float64: bool,
    /// The cheapest exact configuration of the double-single arithmetic on this device (`"fma+barrier"`, `"fma"`,
    /// `"split+barrier"`, `"split"`), `None` when none is exact (the kernels then stay in f32).
    pub df64: Option<String>,
}

/// What the machine offers the GPU modes.
pub enum GpuState {
    /// No usable GPU, or no GPU support in this build: the reason, written for the person who asked.
    Absent { reason: String },
    /// An adapter. `engine` is the GPU pipeline, or why it is not available.
    // Constructed by `gpu_probe` (the `gpu` feature) and by the tests.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    Found { info: GpuInfo, engine: Result<Arc<dyn Engine>, String> },
}

/// One row of `compute.modes`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModeInfo {
    pub mode: ComputeMode,
    pub available: bool,
    pub reason: Option<String>,
    pub detail: Option<String>,
}

#[cfg(any(test, feature = "gpu"))]
pub const NO_ADAPTER_REASON: &str = "No GPU adapter was found on this machine (wgpu looked for Vulkan, DirectX 12 and Metal devices).";
#[cfg(not(feature = "gpu"))]
pub const NO_GPU_FEATURE_REASON: &str = "This binary was built without the `gpu` cargo feature (cargo build --no-default-features); rebuild with the default features to use the GPU.";
/// Why `hybrid` is refused on a machine that has a GPU (the 400 of a request for it, and the capabilities).
pub const HYBRID_NEEDS_BANDS_REASON: &str = "Hybrid CPU + GPU shares row bands between the two devices: it runs the banded engine, so it is offered from a width of 8192 (and by `cloud-map-generator gen`, `cloud-map-generator serve` at any width)";
/// Why `hybrid` is refused when the CPU side has a single thread.
pub const HYBRID_ONE_THREAD_REASON: &str = "This machine runs the CPU engine on a single thread, so there is nothing to share with the GPU.";

/// The widths whose estimates the capabilities list (seconds per mode, disk bytes).
pub const ESTIMATE_WIDTHS: [usize; 7] = [512, 1024, 2048, 4096, 8192, 16384, 32768];

/// The engines of this process and the machine they run on.
pub struct Runtime {
    pub cpu: CpuEngine,
    pub threads: usize,
    pub gpu: GpuState,
}

impl Runtime {
    /// A runtime that has only the CPU (tests, a build without the `gpu` feature, `--mock`-free embedders), with `reason` as the
    /// explanation the GPU modes give.
    pub fn cpu_only(reason: &str) -> Runtime {
        Runtime { cpu: CpuEngine { name: cpu_name(), workers: None }, threads: cpu_threads(), gpu: GpuState::Absent { reason: reason.to_string() } }
    }

    /// The CPU, and the GPU if this build has the feature and the machine an adapter (a probe of about a second:
    /// do it once, at start-up).
    pub fn probe() -> Runtime {
        #[cfg(feature = "gpu")]
        {
            Runtime { cpu: CpuEngine { name: cpu_name(), workers: None }, threads: cpu_threads(), gpu: crate::gpu_probe::probe() }
        }
        #[cfg(not(feature = "gpu"))]
        {
            Runtime::cpu_only(NO_GPU_FEATURE_REASON)
        }
    }

    fn gpu_engine(&self) -> Option<&Arc<dyn Engine>> {
        match &self.gpu {
            GpuState::Found { engine: Ok(e), .. } => Some(e),
            _ => None,
        }
    }

    /// The four modes with their availability, in the contract's order.
    pub fn modes(&self) -> Vec<ModeInfo> {
        let gpu_reason = match &self.gpu {
            GpuState::Absent { reason } => Some(reason.clone()),
            GpuState::Found { engine: Err(why), .. } => Some(why.clone()),
            GpuState::Found { engine: Ok(_), .. } => None,
        };
        // CPU + GPU needs both devices: it is refused with the GPU's reason when there is none, and in any case it is
        // not built yet.
        let hybrid_reason = match &self.gpu {
            GpuState::Absent { reason } => format!("{} CPU + GPU needs both devices, and there is no GPU to share the work with.", sentence(reason)),
            GpuState::Found { engine: Err(why), .. } => format!("{} CPU + GPU needs the GPU pipeline.", sentence(why)),
            GpuState::Found { engine: Ok(_), .. } => HYBRID_ONE_THREAD_REASON.to_string(),
        };
        let gpu_detail = match &self.gpu {
            GpuState::Found { info, engine: Ok(_) } => Some(info.name.clone()),
            _ => None,
        };
        let plural = |n: usize| if n == 1 { "thread" } else { "threads" };
        let cpu_threads = self.cpu.threads();
        ComputeMode::ALL
            .into_iter()
            .map(|mode| match mode {
                ComputeMode::Auto => ModeInfo { mode, available: true, reason: None, detail: None },
                ComputeMode::Cpu => ModeInfo { mode, available: true, reason: None, detail: Some(format!("{cpu_threads} {}", plural(cpu_threads))) },
                ComputeMode::Gpu => ModeInfo { mode, available: gpu_reason.is_none(), reason: gpu_reason.clone(), detail: gpu_detail.clone() },
                ComputeMode::Hybrid => {
                    let ok = gpu_reason.is_none() && cpu_threads > 1;
                    ModeInfo { mode, available: ok, reason: (!ok).then(|| hybrid_reason.clone()), detail: ok.then(|| format!("GPU + {} {}", cpu_threads - 1, plural(cpu_threads - 1))) }
                }
            })
            .collect()
    }

    /// `GET /api/defaults` `compute`: the default mode, the engine, the four modes and the devices.
    pub fn capabilities_json(&self) -> J {
        let machine = crate::machine::snapshot();
        let (large, refused) = crate::machine::large_widths(&machine);
        let max = crate::machine::max_width(&machine, crate::generate::WHOLE_ARRAY_MAX_WIDTH);
        let modes = self
            .modes()
            .into_iter()
            .map(|m| {
                J::obj([
                    ("key", J::s(m.mode.key())),
                    ("label", J::s(m.mode.label())),
                    ("available", J::Bool(m.available)),
                    ("reason", crate::json::opt_str(m.reason.as_deref())),
                    ("description", J::s(m.mode.description())),
                    ("detail", crate::json::opt_str(m.detail.as_deref())),
                    ("max_width", if m.available { J::Int(max as i64) } else { J::Null }),
                    (
                        "estimate_seconds",
                        J::obj(crate::estimates::mode_estimates(m.mode, &ESTIMATE_WIDTHS).into_iter().map(|(k, s)| (k, J::Float(s)))),
                    ),
                ])
            })
            .collect();
        let gpu = match &self.gpu {
            GpuState::Found { info, .. } => J::obj([
                ("name", J::s(info.name.as_str())),
                ("backend", J::s(info.backend.as_str())),
                ("float64", J::Bool(info.float64)),
                ("df64", crate::json::opt_str(info.df64.as_deref())),
            ]),
            GpuState::Absent { .. } => J::Null,
        };
        J::obj([
            ("default", J::s(ComputeMode::Auto.key())),
            ("engine", J::s("rust")),
            (
                "machine",
                J::obj([
                    ("free_ram_bytes", machine.free_ram.map_or(J::Null, |b| J::Int(b as i64))),
                    ("free_disk_bytes", machine.free_disk.map_or(J::Null, |b| J::Int(b as i64))),
                    ("large_widths", J::Arr(large.iter().map(|&w| J::Int(w as i64)).collect())),
                    ("estimate_disk_bytes", J::obj(ESTIMATE_WIDTHS.iter().map(|&w| (w.to_string(), J::Int(crate::estimates::disk_bytes(w) as i64))))),
                    ("calibrated", J::Bool(crate::estimates::factors().calibrated)),
                    ("large_widths_refused", J::obj(refused.iter().map(|(w, why)| (w.to_string(), J::s(why.as_str()))))),
                ]),
            ),
            ("modes", J::Arr(modes)),
            (
                "devices",
                J::obj([
                    ("cpu", J::obj([("name", crate::json::opt_str(self.cpu.name.as_deref())), ("threads", J::Int(self.threads as i64))])),
                    ("gpu", gpu),
                ]),
            ),
        ])
    }

    /// `options.compute`: the mode a request's value names, or the message of the 400. Anything that is not one of
    /// the four keys, and a mode this engine does not offer (the message gives the reason and says to ask for
    /// `auto` or `cpu`), is refused: there is no silent fallback. `name` is the field the message blames.
    pub fn check_compute(&self, value: &Value, name: &str) -> Result<ComputeMode, String> {
        let allowed = ComputeMode::ALL.iter().map(|m| format!("'{}'", m.key())).collect::<Vec<_>>().join(", ");
        let mode = match value.as_str().and_then(ComputeMode::from_key) {
            Some(m) => m,
            None => {
                let keys: Vec<&str> = ComputeMode::ALL.iter().map(|m| m.key()).collect();
                let hint = match value.as_str().and_then(|s| close_match(&s.trim().to_lowercase(), &keys)) {
                    Some(close) => format!(" Did you mean '{close}'?"),
                    None => String::new(),
                };
                return Err(format!(
                    "{name}: expected one of {allowed}, got {}.{hint} 'auto' lets the engine choose; the engine's capabilities list what it offers.",
                    kind(value)
                ));
            }
        };
        let row = self.modes().into_iter().find(|m| m.mode == mode).expect("every mode has a row");
        if !row.available {
            return Err(format!(
                "{name}: '{}' is not available: {} On this engine use 'auto' (it never fails for lack of a device) or 'cpu'; the engine's capabilities say what it offers.",
                mode.key(),
                sentence(&row.reason.unwrap_or_default())
            ));
        }
        Ok(mode)
    }

    /// (used, note): what will really run a `width`-px map asked in `mode`, and a sentence on why when it is not
    /// what the person might expect. `auto` picks by capability and by width; an explicit mode runs as asked
    /// (the caller checked it with [`Runtime::check_compute`]).
    pub fn resolve(&self, mode: ComputeMode, width: usize) -> (Used, Option<String>) {
        match mode {
            ComputeMode::Cpu => (Used::Cpu, None),
            ComputeMode::Gpu => (Used::Gpu, None),
            ComputeMode::Hybrid => (Used::Hybrid, None),
            ComputeMode::Auto => match self.gpu_engine() {
                Some(gpu) if width >= gpu.min_auto_width() => (
                    Used::Gpu,
                    Some(format!("auto chose the GPU: a {width}-wide map is wide enough (from {}) for the GPU to pay for its upload.", gpu.min_auto_width())),
                ),
                Some(_) => (Used::Cpu, Some(format!("auto chose the CPU: a {width}-wide map is below the width where the GPU pays on this device."))),
                None => (Used::Cpu, None),
            },
        }
    }

    /// The GPU's workers for the bands of a large map (none for `cpu`: the scheduler adds its CPU worker itself), or
    /// why this run cannot use the device it asked for.
    pub fn band_workers<'a>(&'a self, used: Used, banded: &'a cloud_map_generator_core::banded::Banded<'a>) -> Result<Vec<Box<dyn crate::bands::BandCompute + 'a>>, String> {
        match used {
            Used::Cpu => Ok(Vec::new()),
            Used::Gpu | Used::Hybrid => {
                let engine = self.gpu_engine().ok_or_else(|| "the GPU pipeline is not available".to_string())?;
                engine.band_worker(banded).map(|w| vec![w])
            }
        }
    }

    /// The engine that runs `used`.
    pub fn engine(&self, used: Used) -> Result<&dyn Engine, EngineError> {
        match used {
            Used::Cpu => Ok(&self.cpu),
            Used::Gpu => self.gpu_engine().map(|e| e.as_ref()).ok_or_else(|| EngineError::Refused("the GPU pipeline is not available".to_string())),
            Used::Hybrid => Err(EngineError::Refused(HYBRID_NEEDS_BANDS_REASON.to_string())),
        }
    }
}

/// `text` as a sentence: a full stop is added when it does not end in one.
fn sentence(text: &str) -> String {
    if text.ends_with(['.', '!', '?']) || text.is_empty() {
        text.to_string()
    } else {
        format!("{text}.")
    }
}

/// Python's `_kind(value)`: how a value is named in a message.
pub fn kind(value: &Value) -> String {
    match value {
        Value::Null => "nothing (null)".to_string(),
        Value::Bool(b) => format!("a boolean ({})", if *b { "True" } else { "False" }),
        Value::String(s) => format!("the text {}", py_repr(s)),
        Value::Number(n) if n.is_f64() => "a float".to_string(),
        Value::Number(_) => "an int".to_string(),
        Value::Array(_) => "a list".to_string(),
        Value::Object(_) => "a dict".to_string(),
    }
}

/// The number of hardware threads (`os.cpu_count()`).
pub fn hardware_threads() -> usize {
    cloud_map_generator_core::par::hardware_threads()
}

/// The threads the CPU engine really uses: what `--threads` or `CLOUD_MAP_GENERATOR_THREADS` asked for, else every logical
/// core (1 in a build without the `rayon` feature). This is what `devices.cpu.threads` reports.
pub fn cpu_threads() -> usize {
    cloud_map_generator_core::par::threads()
}

/// The processor's name, when the system says it: the registry on Windows, `/proc/cpuinfo` on Linux, `sysctl` on macOS.
pub fn cpu_name() -> Option<String> {
    let clean = |s: &str| {
        let t = s.split_whitespace().collect::<Vec<_>>().join(" ");
        (!t.is_empty()).then_some(t)
    };
    #[cfg(windows)]
    {
        let out = std::process::Command::new("reg")
            .args(["query", r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "/v", "ProcessorNameString"])
            .output()
            .ok()
            .filter(|o| o.status.success());
        if let Some(o) = out {
            let text = String::from_utf8_lossy(&o.stdout);
            if let Some(name) = text.lines().find_map(|l| l.split_once("REG_SZ").and_then(|(_, v)| clean(v))) {
                return Some(name);
            }
        }
        std::env::var("PROCESSOR_IDENTIFIER").ok().and_then(|v| clean(&v))
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        text.lines().find_map(|l| l.strip_prefix("model name").and_then(|rest| rest.split_once(':')).and_then(|(_, v)| clean(v)))
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]).output().ok()?;
        clean(&String::from_utf8_lossy(&out.stdout))
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        let _ = clean;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn runtime() -> Runtime {
        Runtime::cpu_only(NO_ADAPTER_REASON)
    }

    #[test]
    fn the_four_modes_come_in_the_contracts_order_with_their_reasons() {
        let r = runtime();
        let modes = r.modes();
        assert_eq!(modes.iter().map(|m| m.mode.key()).collect::<Vec<_>>(), ["auto", "cpu", "gpu", "hybrid"]);
        assert_eq!(modes.iter().map(|m| m.available).collect::<Vec<_>>(), [true, true, false, false]);
        for m in &modes {
            assert_eq!(m.reason.is_none(), m.available, "{m:?}");
        }
        let n = cpu_threads();
        assert_eq!(modes[1].detail.as_deref(), Some(format!("{n} {}", if n == 1 { "thread" } else { "threads" }).as_str()), "the CPU detail names the real thread count");
        assert_eq!(r.threads, n, "devices.cpu.threads is the real count too");
        assert_eq!(modes[2].reason.as_deref(), Some(NO_ADAPTER_REASON));
    }

    #[test]
    fn a_gpu_that_is_found_but_not_wired_gives_the_slots_reason_for_both_modes() {
        let info = GpuInfo { name: "Test GPU".into(), backend: "Vulkan".into(), float64: false, df64: Some("fma+barrier".into()) };
        let r = Runtime {
            cpu: CpuEngine { name: None, workers: None },
            threads: 4,
            gpu: GpuState::Found { info, engine: Err("The GPU pipeline is not wired into this build yet".to_string()) },
        };
        let modes = r.modes();
        assert!(!modes[2].available && modes[2].reason.as_deref() == Some("The GPU pipeline is not wired into this build yet"));
        assert!(!modes[3].available && modes[3].reason.as_deref().unwrap().starts_with("The GPU pipeline is not wired into this build yet"));
        let caps = crate::json::compact(&r.capabilities_json());
        assert!(caps.contains("\"gpu\":{\"name\":\"Test GPU\",\"backend\":\"Vulkan\",\"float64\":false,\"df64\":\"fma+barrier\"}"), "{caps}");
        assert!(caps.contains("\"threads\":4"));
    }

    #[test]
    fn capabilities_have_the_contracts_shape() {
        let caps = serde_json::from_str::<Value>(&crate::json::compact(&runtime().capabilities_json())).unwrap();
        assert_eq!(caps["default"], "auto");
        assert_eq!(caps["engine"], "rust");
        assert_eq!(caps["modes"].as_array().unwrap().len(), 4);
        for m in caps["modes"].as_array().unwrap() {
            let mut keys: Vec<&str> = m.as_object().unwrap().keys().map(String::as_str).collect();
            keys.sort_unstable();
            assert_eq!(keys, ["available", "description", "detail", "estimate_seconds", "key", "label", "max_width", "reason"]);
        }
        assert!(caps["devices"]["gpu"].is_null());
        assert!(caps["devices"]["cpu"]["threads"].as_u64().unwrap() >= 1);
    }

    #[test]
    fn descriptions_are_one_short_sentence_and_labels_are_the_contracts() {
        for mode in ComputeMode::ALL {
            let d = mode.description();
            assert!(d.chars().count() <= DESCRIPTION_MAX_CHARS, "{} is {} characters", mode.key(), d.chars().count());
            assert!(d.ends_with('.') && !d.is_empty());
        }
        let labels: Vec<&str> = ComputeMode::ALL.iter().map(|m| m.label()).collect();
        assert_eq!(labels, ["Auto", "CPU", "GPU", "CPU + GPU"]);
    }

    #[test]
    fn check_compute_names_the_allowed_values_and_the_reason() {
        let r = runtime();
        assert_eq!(r.check_compute(&json!("auto"), "options.compute"), Ok(ComputeMode::Auto));
        assert_eq!(r.check_compute(&json!("cpu"), "options.compute"), Ok(ComputeMode::Cpu));
        for mode in ["gpu", "hybrid"] {
            let e = r.check_compute(&json!(mode), "options.compute").unwrap_err();
            assert!(e.starts_with("options.compute:") && e.contains("is not available") && e.contains("No GPU adapter"), "{e}");
            assert!(e.contains("'auto'") && e.contains("'cpu'"), "{e}");
        }
        for bad in [json!("turbo"), json!("GPU"), json!(""), Value::Null, json!(5), json!(true), json!(["cpu"])] {
            let e = r.check_compute(&bad, "options.compute").unwrap_err();
            assert!(e.starts_with("options.compute: expected one of 'auto', 'cpu', 'gpu', 'hybrid', got "), "{e}");
        }
        let e = r.check_compute(&json!("GPU"), "options.compute").unwrap_err();
        assert!(e.contains("the text 'GPU'") && e.contains("Did you mean 'gpu'?"), "{e}");
        assert!(r.check_compute(&json!(true), "options.compute").unwrap_err().contains("a boolean (True)"));
    }

    #[test]
    fn auto_runs_on_the_cpu_without_a_gpu_engine_and_says_nothing() {
        let r = runtime();
        assert_eq!(r.resolve(ComputeMode::Auto, 4096), (Used::Cpu, None));
        assert_eq!(r.resolve(ComputeMode::Cpu, 64), (Used::Cpu, None));
    }

    struct Fake;
    impl Engine for Fake {
        fn generate(&self, _: &GenerateRequest, _: Progress<'_>) -> Result<Generated, EngineError> {
            Err(EngineError::Failed("not a real engine".into()))
        }
        fn min_auto_width(&self) -> usize {
            1024
        }
    }

    #[test]
    fn auto_goes_to_the_gpu_from_its_crossover_width_and_explains_the_cpu_below_it() {
        let info = GpuInfo { name: "G".into(), backend: "Vulkan".into(), float64: false, df64: None };
        let r = Runtime { cpu: CpuEngine { name: None, workers: None }, threads: 2, gpu: GpuState::Found { info, engine: Ok(Arc::new(Fake)) } };
        assert!(r.modes()[2].available && r.modes()[2].detail.as_deref() == Some("G"));
        for width in [1024, 4096] {
            let (used, note) = r.resolve(ComputeMode::Auto, width);
            assert_eq!(used, Used::Gpu);
            assert!(note.unwrap().starts_with(&format!("auto chose the GPU: a {width}-wide map")));
        }
        let (used, note) = r.resolve(ComputeMode::Auto, 512);
        assert_eq!(used, Used::Cpu);
        assert_eq!(note.unwrap(), "auto chose the CPU: a 512-wide map is below the width where the GPU pays on this device.");
        assert_eq!(r.resolve(ComputeMode::Gpu, 512), (Used::Gpu, None), "an explicit mode runs as asked, with no note");
        assert_eq!(r.check_compute(&json!("gpu"), "options.compute"), Ok(ComputeMode::Gpu));
        assert!(r.engine(Used::Gpu).is_ok() && r.engine(Used::Hybrid).is_err());
    }
}
