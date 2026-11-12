//! The seam between the MCP server and the engine that makes maps: everything a tool needs, nothing more.
//!
//! The server never computes a map. It validates what the AI asked for, keeps it inside the sandbox, runs the
//! heavy work one job at a time and shapes the answers; the work itself is a [`Backend`]. The types here mirror
//! `cloudgen/service.py` (`Physics`, `LIMITS`, `WIDTHS`, `GenerationResult`, the metric rows, the zonal profile),
//! and an adapter over the real engine implements the trait, then calls [`crate::run`] with it. [`crate::mock`]
//! is a deterministic stand-in that fabricates plausible results, used by the tests and until the adapter lands.
//!
//! The service-level rules every engine shares (the range of each input, the messages of a bad preset or compute
//! mode) are in [`crate::rules`], so two adapters cannot word them differently.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

/// The ranges every public function enforces: (name, min, max), both included. `seed` is an integer range.
pub const LIMITS: [(&str, f64, f64); 5] = [
    ("seed", 0.0, 2147483647.0),
    ("rotation_period_days", -400.0, 400.0), // negative: retrograde spin; 0: no spin at all
    ("radius_km", 500.0, 30000.0),
    ("gravity_mps2", 0.1, 100.0),
    ("scale_height_km", 0.5, 100.0),
];

/// The unit of each limit, for the messages.
pub const LIMIT_UNITS: [(&str, &str); 5] = [
    ("seed", "an integer"),
    ("rotation_period_days", "days, negative for a retrograde spin"),
    ("radius_km", "km"),
    ("gravity_mps2", "m/s^2"),
    ("scale_height_km", "km"),
];

/// The four physics inputs a planet is made of (`cloud_regime` is the fifth, a word).
pub const PHYSICS_FIELDS: [&str; 4] = ["rotation_period_days", "radius_km", "gravity_mps2", "scale_height_km"];

/// The map widths a generation accepts; the map is `width` x `width / 2` texels.
pub const WIDTHS: [u32; 5] = [256, 512, 1024, 2048, 4096];

/// Typical wall time of one generation in seconds, metrics included, on a quiet CPU (the default of
/// [`Backend::estimate_seconds`]; an engine that is faster or slower overrides it).
pub const ESTIMATE_SECONDS: [(u32, f64); 5] = [(256, 6.0), (512, 30.0), (1024, 35.0), (2048, 50.0), (4096, 100.0)];

/// Where the work runs. `docs/compute-modes.html` is the contract every engine implements.
pub const COMPUTE_MODES: [&str; 4] = ["auto", "cpu", "gpu", "hybrid"];
pub const COMPUTE_DEFAULT: &str = "auto";

/// `(min, max)` of a limit by name.
pub fn limit(name: &str) -> (f64, f64) {
    LIMITS.iter().find(|(n, _, _)| *n == name).map(|(_, lo, hi)| (*lo, *hi)).unwrap_or((f64::MIN, f64::MAX))
}

/// What goes wrong inside an engine. The text is written for an AI reader: which field was wrong, the allowed
/// range or values, and what to do next.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    /// A request the engine refuses (Python: `ServiceError`). Returned to the caller with absolute paths scrubbed.
    Message(String),
    /// The `cancel` callback returned true; nothing was written after that point.
    Cancelled,
    /// A failure the caller cannot fix. The server tells the caller only that it was internal; the text goes to
    /// the server log.
    Internal(String),
}

impl BackendError {
    pub fn message(text: impl Into<String>) -> Self {
        BackendError::Message(text.into())
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::Message(m) | BackendError::Internal(m) => f.write_str(m),
            BackendError::Cancelled => f.write_str("cancelled"),
        }
    }
}

impl std::error::Error for BackendError {}

impl From<BackendError> for crate::errors::AppError {
    fn from(e: BackendError) -> Self {
        match e {
            BackendError::Message(m) => crate::errors::AppError::Service(m),
            BackendError::Cancelled => crate::errors::AppError::Cancelled,
            BackendError::Internal(d) => crate::errors::AppError::internal("InternalError", d),
        }
    }
}

/// The inputs of one planet. A negative `rotation_period_days` is a retrograde spin, 0 is no spin.
/// `cloud_regime` names the condensate; only `water` is generated without the experiment option.
#[derive(Debug, Clone, PartialEq)]
pub struct Physics {
    pub rotation_period_days: f64,
    pub radius_km: f64,
    pub gravity_mps2: f64,
    pub scale_height_km: f64,
    pub cloud_regime: String,
}

impl Physics {
    pub fn new(rotation_period_days: f64, radius_km: f64, gravity_mps2: f64, scale_height_km: f64) -> Self {
        Physics { rotation_period_days, radius_km, gravity_mps2, scale_height_km, cloud_regime: "water".into() }
    }

    pub fn with_regime(mut self, regime: &str) -> Self {
        self.cloud_regime = regime.to_string();
        self
    }

    /// The five fields as a JSON object (`asdict(Physics)`).
    pub fn to_json(&self) -> Value {
        json!({
            "rotation_period_days": self.rotation_period_days,
            "radius_km": self.radius_km,
            "gravity_mps2": self.gravity_mps2,
            "scale_height_km": self.scale_height_km,
            "cloud_regime": self.cloud_regime,
        })
    }

    pub fn is_water(&self) -> bool {
        self.cloud_regime == "water"
    }
}

/// A built-in planet.
#[derive(Debug, Clone, PartialEq)]
pub struct Preset {
    pub key: String,
    pub label: String,
    pub physics: Physics,
    /// The clouds are not water: generation needs `experiment=true`.
    pub experiment_required: bool,
    pub note: String,
}

impl Preset {
    pub fn to_json(&self) -> Value {
        let p = &self.physics;
        json!({
            "key": self.key,
            "label": self.label,
            "rotation_period_days": p.rotation_period_days,
            "radius_km": p.radius_km,
            "gravity_mps2": p.gravity_mps2,
            "scale_height_km": p.scale_height_km,
            "cloud_regime": p.cloud_regime,
            "experiment_required": self.experiment_required,
            "note": self.note,
        })
    }
}

/// What a physics implies, without generating anything (see `service.derive`).
#[derive(Debug, Clone, PartialEq)]
pub struct DeriveOut {
    pub hadley_edge_deg: f64,
    pub storm_track_deg: f64,
    pub has_eddies: bool,
    pub deformation_radius_km: Option<f64>,
    pub cyclones_per_hemisphere: Option<u32>,
    /// `counterclockwise` or `clockwise`, seen from above the north pole; `None` without eddies.
    pub cyclone_spin_north: Option<String>,
    pub cyclone_spin_south: Option<String>,
    /// Kilometres per texel at the equator, per map width.
    pub texel_km: Vec<(u32, f64)>,
    pub bakeable: bool,
    pub warnings: Vec<String>,
}

/// Whether the land is blind (`none`) or a made-up world (`synthetic`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    None,
    Synthetic,
}

impl Surface {
    pub fn as_str(self) -> &'static str {
        match self {
            Surface::None => "none",
            Surface::Synthetic => "synthetic",
        }
    }
}

/// The options of a generation. `compute` says where the work runs; it is not part of the map.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOptions {
    pub match_earth: bool,
    pub isotropic: bool,
    pub experiment: bool,
    pub surface: Surface,
    /// One of [`COMPUTE_MODES`], already checked against [`Backend::capabilities`].
    pub compute: String,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        GenerateOptions {
            match_earth: false,
            isotropic: false,
            experiment: false,
            surface: Surface::None,
            compute: COMPUTE_DEFAULT.to_string(),
        }
    }
}

impl GenerateOptions {
    /// The options record of a map (`compute` is not part of it: where it ran is not a property of the map).
    pub fn record(&self) -> Value {
        json!({
            "match_earth": self.match_earth,
            "isotropic": self.isotropic,
            "experiment": self.experiment,
            "surface": self.surface.as_str(),
        })
    }
}

/// One generation. `out_dir` is a folder the server created and owns: write the files there (a subfolder of
/// its own is fine) and name them in [`GenerationResult::files`]. `None` means write nothing.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub physics: Physics,
    pub seed: u32,
    pub width: u32,
    pub options: GenerateOptions,
    pub out_dir: Option<PathBuf>,
}

/// An 8-bit north-up grayscale image, row 0 at the north pole, column 0 at longitude -180; 0..255 = 0..1.
#[derive(Debug, Clone, PartialEq)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub bytes: Vec<u8>,
}

impl Plane {
    pub fn new(width: usize, height: usize, bytes: Vec<u8>) -> Self {
        assert_eq!(width * height, bytes.len(), "a plane holds width x height bytes");
        Plane { width, height, bytes }
    }

    pub fn nbytes(&self) -> usize {
        self.bytes.len()
    }
}

/// One row of the comparison table: a statistic of the map next to Earth's.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricRow {
    pub key: String,
    pub label: String,
    pub generated: Option<f64>,
    pub earth: Option<f64>,
    /// `(generated - earth) / |earth|`; `None` where Earth's value is 0.
    pub relative_gap: Option<f64>,
    /// How to read the row and what moves it.
    pub hint: String,
}

/// Zonal-mean cloud fraction in 5-degree bands: `lat_deg` (band centres, south first), the map's, Earth's and
/// what the climate model aimed for (`None` when the map does not belong to a known body).
#[derive(Debug, Clone, PartialEq)]
pub struct ZonalProfile {
    pub lat_deg: Vec<f64>,
    pub generated: Vec<Option<f64>>,
    pub earth: Vec<Option<f64>>,
    pub target: Option<Vec<Option<f64>>>,
}

/// A finished map: the data the tools report and the pixels the server draws previews and globes from.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// A lowercase hex token (8 to 32 characters) naming the result; the server makes one up when it is not.
    pub id: String,
    pub seconds: f64,
    pub width: u32,
    pub height: u32,
    pub physics: Physics,
    pub seed: u32,
    /// The options record of the map (no `compute`).
    pub options: Value,
    pub metrics: Vec<MetricRow>,
    pub zonal: ZonalProfile,
    /// Written files: `master`, `preview`, `r8gz`, `type`, `high_share`, `wind`, `meta`, `metrics`; empty when the
    /// request had no `out_dir`. Every path is under that folder.
    pub files: BTreeMap<String, PathBuf>,
    /// The coverage, 8-bit north-up: what the server's previews and globes are drawn from.
    pub coverage: Plane,
    /// The other products a map can be looked at as (`type`, `high_share`, `wind`; wind is the speed), same grid,
    /// 8-bit. Optional: `view_map` refuses a product the result does not carry.
    pub extra_planes: BTreeMap<String, Plane>,
    pub warnings: Vec<String>,
    /// What ran the map: `{requested, used, devices: [{kind, name, seconds, share}], note}`.
    pub compute: Option<Value>,
}

/// A coverage PNG measured against Earth.
#[derive(Debug, Clone)]
pub struct MetricsOut {
    pub width: u32,
    pub height: u32,
    pub metrics: Vec<MetricRow>,
    pub zonal: ZonalProfile,
    pub warnings: Vec<String>,
}

/// Which part of a bake runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BakeOnly {
    List,
    Library,
}

impl BakeOnly {
    pub fn as_str(self) -> &'static str {
        match self {
            BakeOnly::List => "list",
            BakeOnly::Library => "library",
        }
    }
}

/// A bake of a catalog of bodies. `bodies_file` is a bodies JSON file the sandbox already resolved and the
/// server already checked (count, key spelling, data files inside a readable root); `out_dir` is
/// `<out>/bake/<version>`, created by the server.
#[derive(Debug, Clone)]
pub struct BakeRequest {
    pub bodies_file: PathBuf,
    pub out_dir: PathBuf,
    pub version: String,
    pub url_prefix: Option<String>,
    pub list_width: u32,
    pub library_width: u32,
    pub k: u32,
    pub only: Option<BakeOnly>,
}

/// A body the bake left out and why.
#[derive(Debug, Clone, PartialEq)]
pub struct Skipped {
    pub key: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct BakeResult {
    /// The manifest (written to `manifest.json` unless `only` was given: a partial bake writes none).
    pub manifest: Value,
    pub manifest_path: Option<PathBuf>,
    /// Every file written, under `out_dir`.
    pub files: Vec<PathBuf>,
    pub skipped: Vec<Skipped>,
    pub seconds: f64,
}

/// Progress of a long call: a fraction 0..1 (never decreasing) and the current stage.
pub type Progress<'a> = &'a mut dyn FnMut(f32, &str);
/// Polled at stage boundaries: true means stop and return [`BackendError::Cancelled`].
pub type CancelCheck<'a> = &'a dyn Fn() -> bool;

/// What the server needs from an engine. Every method is safe to call from several threads; the server runs
/// `generate` and `bake` on one worker thread, one at a time, and `derive`, `measure_png`, `presets` and
/// `capabilities` from the request threads.
pub trait Backend: Send + Sync + 'static {
    /// The compute-mode contract of `docs/compute-modes.html`:
    /// `{default, engine, modes: [{key, label, available, reason, description, detail}], devices: {cpu: {name,
    /// threads}, gpu: null | {name, backend, float64, df64}}}`. `modes` always lists auto, cpu, gpu, hybrid.
    fn capabilities(&self) -> Value;

    /// The built-in planets, water ones first.
    fn presets(&self) -> Vec<Preset>;

    /// What a physics implies, in milliseconds, without generating anything.
    fn derive(&self, physics: &Physics) -> Result<DeriveOut, BackendError>;

    /// Generate one map, measure it against Earth and write the files into `request.out_dir`.
    /// Call `progress(fraction, stage)` as the work moves forward and poll `cancel` at stage boundaries.
    fn generate(
        &self,
        request: &GenerateRequest,
        progress: Progress<'_>,
        cancel: CancelCheck<'_>,
    ) -> Result<GenerationResult, BackendError>;

    /// Measure a coverage PNG (8 or 16 bit gray, 2:1, 256 to 8192 wide) against Earth's reference.
    fn measure_png(&self, path: &Path) -> Result<MetricsOut, BackendError>;

    /// Bake the maps of a set of bodies to files.
    fn bake(
        &self,
        request: &BakeRequest,
        progress: Progress<'_>,
        cancel: CancelCheck<'_>,
    ) -> Result<BakeResult, BackendError>;

    /// A Markdown summary of what the model does and does not do.
    fn explain_model(&self) -> String;

    /// Typical seconds of one generation at `width` on this engine (`None` when unknown).
    fn estimate_seconds(&self, width: u32) -> Option<f64> {
        ESTIMATE_SECONDS.iter().find(|(w, _)| *w == width).map(|(_, s)| *s)
    }
}
