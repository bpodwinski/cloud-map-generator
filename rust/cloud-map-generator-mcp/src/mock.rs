//! A deterministic stand-in for the real engine: it fabricates plausible results so the server, its tests and its
//! documentation work end to end before the adapter over the real engine lands.
//!
//! What is real: the presets, the derived physics (Held-Hou scaling of the Hadley edge, the storm track, the
//! deformation radius and the cyclone count, from `cloudgen/climate.py`), the labels, hints and formats of the
//! metric rows, the zonal profile and the Earth statistics (measured on the embedded Blue Marble reference), the
//! file set and its names, the manifest of a bake, every error message of the service layer. What is made up: the
//! map itself (analytic latitude bands and a few Gaussians, seeded), its structure statistics (coherence,
//! spectrum, object sizes), and the timings. A map from this engine is NOT a prediction of anything.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::backend::{
    Backend, BackendError, BakeOnly, BakeRequest, BakeResult, CancelCheck, DeriveOut, GenerateOptions, GenerateRequest,
    GenerationResult, MetricRow, MetricsOut, Physics, Plane, Preset, Progress, Skipped, Surface, ZonalProfile,
    WIDTHS,
};
use crate::pyfmt::{fmt_g, round_to, token_hex};
use crate::render::{decode_coverage, encode_gray16, encode_gray8, encode_rgb8, shrink, sniff, Gray, Sniff};
use crate::resources::EARTH_REFERENCE_PNG;
use crate::rules::{check_bodies, physics_number};

const EARTH_ROTATION_DAYS: f64 = 0.99727;
const EARTH_RADIUS_KM: f64 = 6371.0;
const EARTH_GRAVITY: f64 = 9.81;
const EARTH_SCALE_HEIGHT_KM: f64 = 8.0;
const EARTH_HADLEY_EDGE_DEG: f64 = 30.0;
const HADLEY_MIN_DEG: f64 = 10.0;
const HADLEY_MAX_DEG: f64 = 90.0;
const DEFORMATION_FRACTION_OF_HADLEY: f64 = 0.30;
const EARTH_STORM_TRACK_DEG: f64 = 51.0;
const STORM_TRACK_IN_HADLEY: f64 = 1.7;
const STORM_TRACK_MAX_DEG: f64 = 80.0;
const MAX_CYCLONES_PER_HEMISPHERE: i64 = 32;
const SLOW_CYCLONES_PER_HEMISPHERE: i64 = 14;
const REF_W: usize = 2048;
const SLICE_WIDTH: usize = 2048;

// ------------------------------------------------------------------------------------------------ the gate

/// A switch tests use to hold a generation at its second stage: the job is then observably running (or queued
/// behind it) for as long as the gate is shut.
#[derive(Default)]
pub struct Gate {
    open: Mutex<bool>,
    entered: Mutex<usize>,
    cv: Condvar,
}

impl Gate {
    pub fn new() -> Arc<Gate> {
        Arc::new(Gate::default())
    }

    pub fn open(&self) {
        *self.open.lock().unwrap_or_else(|p| p.into_inner()) = true;
        self.cv.notify_all();
    }

    /// How many generations have reached the gate.
    pub fn entered(&self) -> usize {
        *self.entered.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Block until a generation has reached the gate (true) or `timeout` passes (false).
    pub fn wait_entered(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while self.entered() == 0 {
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        true
    }

    fn pass(&self, cancel: CancelCheck<'_>) -> Result<(), BackendError> {
        *self.entered.lock().unwrap_or_else(|p| p.into_inner()) += 1;
        loop {
            if cancel() {
                return Err(BackendError::Cancelled);
            }
            let open = self.open.lock().unwrap_or_else(|p| p.into_inner());
            if *open {
                return Ok(());
            }
            let _ = self.cv.wait_timeout(open, Duration::from_millis(10));
        }
    }
}

/// How a generation fails on purpose (tests).
#[derive(Debug, Clone)]
pub enum MockFailure {
    /// The engine refuses with this message (a `ServiceError`).
    Refuse(String),
    /// The engine fails in a way the caller cannot fix (the text goes to the log only).
    Internal(String),
    /// The engine panics.
    Panic(String),
}

/// The mock engine. [`MockBackend::new`] is a CPU-only engine; [`MockBackend::with_gpu`] pretends to have a GPU.
pub struct MockBackend {
    gpu: bool,
    stage_delay: Duration,
    gate: Option<Arc<Gate>>,
    failure: Option<MockFailure>,
    generate_calls: AtomicUsize,
    options_seen: Mutex<Vec<GenerateOptions>>,
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockBackend {
    pub fn new() -> Self {
        MockBackend {
            gpu: false,
            stage_delay: Duration::ZERO,
            gate: None,
            failure: None,
            generate_calls: AtomicUsize::new(0),
            options_seen: Mutex::new(Vec::new()),
        }
    }

    /// An engine that offers every compute mode and reports a GPU.
    pub fn with_gpu() -> Self {
        MockBackend { gpu: true, ..Self::new() }
    }

    /// Hold every generation at its second stage until the gate opens.
    pub fn with_gate(mut self, gate: Arc<Gate>) -> Self {
        self.gate = Some(gate);
        self
    }

    /// Pause this long at every stage of a generation.
    pub fn with_stage_delay(mut self, d: Duration) -> Self {
        self.stage_delay = d;
        self
    }

    /// Make every generation fail.
    pub fn with_failure(mut self, f: MockFailure) -> Self {
        self.failure = Some(f);
        self
    }

    /// How many times `generate` was called.
    pub fn generate_calls(&self) -> usize {
        self.generate_calls.load(Ordering::SeqCst)
    }

    /// The options of every generation, in order.
    pub fn options_seen(&self) -> Vec<GenerateOptions> {
        self.options_seen.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn cpu_name(&self) -> String {
        "Mock CPU".to_string()
    }

    fn compute_block(&self, requested: &str, width: u32, seconds: f64) -> Value {
        let (used, note) = if self.gpu {
            match requested {
                "auto" => ("gpu", Some("auto chose the GPU: the map is large enough for the upload to pay.")),
                other => (if other == "hybrid" { "hybrid" } else if other == "gpu" { "gpu" } else { "cpu" }, None),
            }
        } else {
            ("cpu", None)
        };
        let _ = width;
        let devices = match used {
            "hybrid" => json!([
                {"kind": "gpu", "name": "Mock GPU", "seconds": round_to(seconds * 0.8, 2), "share": 0.7},
                {"kind": "cpu", "name": self.cpu_name(), "seconds": round_to(seconds * 0.9, 2), "share": 0.3},
            ]),
            "gpu" => json!([{"kind": "gpu", "name": "Mock GPU", "seconds": seconds, "share": 1.0}]),
            _ => json!([{"kind": "cpu", "name": self.cpu_name(), "seconds": seconds, "share": 1.0}]),
        };
        json!({"requested": requested, "used": used, "devices": devices, "note": note})
    }
}

// ------------------------------------------------------------------------------------------------ presets

fn presets_table() -> Vec<Preset> {
    let mk = |key: &str, label: &str, physics: Physics, note: &str| Preset {
        key: key.into(),
        label: label.into(),
        experiment_required: !physics.is_water(),
        physics,
        note: note.into(),
    };
    let earth = || Physics::new(EARTH_ROTATION_DAYS, EARTH_RADIUS_KM, EARTH_GRAVITY, EARTH_SCALE_HEIGHT_KM);
    vec![
        mk(
            "earth",
            "Earth",
            earth(),
            "The reference body: the zonal profile is anchored on Earth's measured cloud fraction. Hadley edge 30 degrees, storm track 51, five lows per hemisphere.",
        ),
        mk(
            "earth-retrograde",
            "Retrograde Earth",
            Physics { rotation_period_days: -EARTH_ROTATION_DAYS, ..earth() },
            "Earth spun the other way: same bands, and the lows turn the opposite way in each hemisphere.",
        ),
        mk(
            "fast-rotator",
            "Fast rotator (0.4 day)",
            Physics { rotation_period_days: 0.4, ..earth() },
            "A made-up Earth-sized world that spins in 9.6 hours: narrower bands, a Hadley edge near 12 degrees, more but smaller cyclones.",
        ),
        mk(
            "slow-rotator",
            "Slow rotator (4 days)",
            Physics { rotation_period_days: 4.0, ..earth() },
            "Earth with a 4-day day: the Hadley cell reaches the pole, so there is no storm track, no cyclones and no wind field.",
        ),
        mk(
            "super-earth",
            "Super-Earth (2 radii)",
            Physics::new(EARTH_ROTATION_DAYS, 2.0 * EARTH_RADIUS_KM, 2.0 * EARTH_GRAVITY, EARTH_SCALE_HEIGHT_KM / 2.0),
            "Twice Earth's radius at Earth's density: gravity doubles and, for the same air, the scale height halves. Hadley edge 15 degrees.",
        ),
        mk(
            "mars",
            "Mars",
            Physics::new(1.02596, 3389.5, 3.721, 11.1).with_regime("thin water ice"),
            "Thin water-ice clouds: not water, so the cloud amount (Earth's) is wrong for it and it needs the experiment option.",
        ),
        mk(
            "titan",
            "Titan",
            Physics::new(15.945, 2574.7, 1.352, 20.6).with_regime("methane"),
            "Methane clouds and a 16-day spin: not water, so it needs the experiment option. A global Hadley cell, no storm track.",
        ),
        mk(
            "venus",
            "Venus",
            Physics::new(-243.02, 6051.8, 8.87, 15.7).with_regime("sulfuric acid"),
            "A sulfuric-acid deck and a retrograde 243-day spin: not water, so it needs the experiment option. A global Hadley cell; its real super-rotation is outside the model.",
        ),
    ]
}

// ------------------------------------------------------------------------------------------------ climate

struct Climate {
    raw: f64,
    hadley: f64,
    storm: f64,
    eddies: bool,
    lr_rad: f64,
}

fn climate(p: &Physics) -> Climate {
    let raw = if p.rotation_period_days == 0.0 || !p.rotation_period_days.is_finite() {
        f64::INFINITY
    } else {
        let omega_ratio = (EARTH_ROTATION_DAYS / p.rotation_period_days).abs();
        let radius_ratio = p.radius_km / EARTH_RADIUS_KM;
        let gh = (p.gravity_mps2 * p.scale_height_km).max(0.0);
        let gh_ratio = gh / (EARTH_GRAVITY * EARTH_SCALE_HEIGHT_KM);
        EARTH_HADLEY_EDGE_DEG * gh_ratio.sqrt() / (omega_ratio * radius_ratio)
    };
    let hadley = if raw.is_finite() { raw.clamp(HADLEY_MIN_DEG, HADLEY_MAX_DEG) } else { HADLEY_MAX_DEG };
    let storm = STORM_TRACK_MAX_DEG.min(STORM_TRACK_IN_HADLEY * hadley);
    let lat_factor = EARTH_STORM_TRACK_DEG.to_radians().sin() / storm.to_radians().sin();
    let lr_rad = DEFORMATION_FRACTION_OF_HADLEY * raw.to_radians() * lat_factor;
    Climate { raw, hadley, storm, eddies: raw < 50.0, lr_rad }
}

/// Lows per hemisphere from the closed form (the storm-track circle over five deformation radii).
fn cyclone_estimate(c: &Climate) -> Option<i64> {
    if !c.eddies {
        return None;
    }
    let n = if c.lr_rad > 0.0 { 2.0 * PI * c.storm.to_radians().cos() / (5.0 * c.lr_rad) } else { f64::INFINITY };
    Some(if !n.is_finite() || n > 1e9 { 1_000_000_000 } else { (n.round_ties_even() as i64).max(2) })
}

fn round_up(x: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(x.log10().floor() as i32 - digits + 1);
    (x / scale).ceil() * scale
}

/// The shortest |rotation period| (days), to two digits, that keeps the cyclone count within the limit.
fn min_period_for_limit(p: &Physics) -> Option<f64> {
    let fine = |period: f64| {
        let c = climate(&Physics { rotation_period_days: period, ..p.clone() });
        cyclone_estimate(&c).is_none_or(|n| n <= MAX_CYCLONES_PER_HEMISPHERE)
    };
    let (mut lo, mut hi) = (if p.rotation_period_days == 0.0 { 1.0 } else { p.rotation_period_days.abs() }, 400.0);
    if !fine(hi) {
        return None;
    }
    for _ in 0..50 {
        let mid = (lo * hi).sqrt();
        if fine(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let out = round_up(hi, 2);
    Some(if fine(out) { out } else { hi })
}

fn body_warnings(p: &Physics, c: &Climate, cyclones: Option<i64>) -> Vec<String> {
    let mut out = Vec::new();
    if !p.is_water() {
        out.push(format!(
            "Clouds of {}, not water: this generator's cloud amount is Earth's, so it refuses unless the experiment option is on (experiment=true), and the map is never a bake.",
            p.cloud_regime
        ));
    }
    if p.rotation_period_days == 0.0 {
        out.push("A zero rotation period means no spin: one global Hadley cell, no storm track.".into());
    } else if p.rotation_period_days < 0.0 {
        out.push("Retrograde spin: lows turn the opposite way in each hemisphere and the zonal winds flip sign.".into());
    }
    if c.raw.is_finite() && c.raw < 10.0 {
        out.push(format!("Hadley edge clamped to 10 degrees (raw value {:.1}): the bands saturate.", c.raw));
    }
    if !c.eddies {
        out.push(
            "No storm track: the cloud stays zonal, with no cyclones and no wind field (a global Hadley cell, like Venus or Titan)."
                .into(),
        );
    }
    match cyclones {
        Some(n) if n > MAX_CYCLONES_PER_HEMISPHERE => out.push(format!(
            "{n} cyclones per hemisphere: beyond the {MAX_CYCLONES_PER_HEMISPHERE} this service generates, a spin this fast would take minutes per map."
        )),
        Some(n) if n > SLOW_CYCLONES_PER_HEMISPHERE => {
            out.push(format!("{n} cyclones per hemisphere: generation takes noticeably longer than the estimate."))
        }
        _ => {}
    }
    out
}

fn check_physics(p: &Physics) -> Result<(), BackendError> {
    for (name, v) in [
        ("rotation_period_days", p.rotation_period_days),
        ("radius_km", p.radius_km),
        ("gravity_mps2", p.gravity_mps2),
        ("scale_height_km", p.scale_height_km),
    ] {
        physics_number(v, name)?;
    }
    crate::rules::regime(&p.cloud_regime)?;
    Ok(())
}

fn derive_out(p: &Physics) -> Result<DeriveOut, BackendError> {
    check_physics(p)?;
    let c = climate(p);
    let cyclones = cyclone_estimate(&c);
    let prograde = p.rotation_period_days > 0.0;
    let spin = |north: bool| {
        cyclones.map(|_| if prograde == north { "counterclockwise".to_string() } else { "clockwise".to_string() })
    };
    Ok(DeriveOut {
        hadley_edge_deg: round_to(c.hadley, 2),
        storm_track_deg: round_to(c.storm, 2),
        has_eddies: c.eddies,
        deformation_radius_km: (c.lr_rad.is_finite()).then(|| round_to(c.lr_rad * p.radius_km, 1)),
        cyclones_per_hemisphere: cyclones.map(|n| n.clamp(0, i64::from(u32::MAX)) as u32),
        cyclone_spin_north: spin(true),
        cyclone_spin_south: spin(false),
        texel_km: WIDTHS.iter().map(|w| (*w, round_to(2.0 * PI * p.radius_km / f64::from(*w), 1))).collect(),
        bakeable: p.is_water(),
        warnings: body_warnings(p, &c, cyclones),
    })
}

// ------------------------------------------------------------------------------------------------ the fake map

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn hash01(a: i64, b: i64, seed: u64) -> f64 {
    let mut r = Rng((a as u64).wrapping_mul(0x1234_5678_9ABC_DEF1) ^ (b as u64).wrapping_mul(0x0FED_CBA9_8765_4321) ^ seed);
    r.unit()
}

/// Value noise, periodic in longitude: `x` in [0, 1) around the globe, `y` in [0, 1] pole to pole.
fn value_noise(x: f64, y: f64, nx: i64, ny: i64, seed: u64) -> f64 {
    let (fx, fy) = (x * nx as f64, y * ny as f64);
    let (x0, y0) = (fx.floor() as i64, fy.floor() as i64);
    let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
    let s = |t: f64| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(tx), s(ty));
    let at = |ix: i64, iy: i64| hash01(ix.rem_euclid(nx), iy.clamp(0, ny), seed);
    let a = at(x0, y0) * (1.0 - sx) + at(x0 + 1, y0) * sx;
    let b = at(x0, y0 + 1) * (1.0 - sx) + at(x0 + 1, y0 + 1) * sx;
    a * (1.0 - sy) + b * sy
}

fn smoothstep(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn gauss(x: f64, mu: f64, sigma: f64) -> f64 {
    let d = (x - mu) / sigma;
    (-d * d).exp()
}

/// The zonal-mean cloud fraction in units of the Hadley edge (`climate.target_zonal_coverage`, no ITCZ shift).
fn target_zonal(lat_deg: f64, hadley: f64) -> f64 {
    let x = lat_deg.abs() / hadley.max(1e-3);
    0.20 + 0.17 * gauss(x, 0.1, 0.25) + 0.45 * smoothstep(1.0, 2.0, x) + 0.05 * smoothstep(2.0, 2.6, x)
}

fn convective_tendency(lat_deg: f64, hadley: f64) -> f64 {
    let x = lat_deg.abs() / hadley.max(1e-3);
    (0.15 + 0.85 * gauss(x, 0.0, 0.3) + 0.4 * gauss(x, 1.7, 0.35) - 0.05 * smoothstep(2.0, 2.6, x)).clamp(0.0, 1.0)
}

struct Fields {
    width: usize,
    height: usize,
    cov: Vec<f32>,
    typ: Vec<f32>,
    high: Vec<f32>,
    east: Vec<f32>,
    north: Vec<f32>,
}

fn row_lat_deg(r: usize, h: usize) -> f64 {
    90.0 - (r as f64 + 0.5) * 180.0 / h as f64
}

/// Seeded analytic bands plus a few Gaussians where the storm track would put its lows.
fn synth_fields(p: &Physics, seed: u64, width: usize, opt: &GenerateOptions) -> Fields {
    let c = climate(p);
    let (w, h) = (width, width / 2);
    let mut rng = Rng(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xC10D);
    // lows on the storm track: (lon, lat, sigma in radians, amplitude)
    let mut blobs: Vec<(f64, f64, f64, f64)> = Vec::new();
    if !opt.isotropic {
        let n = if c.eddies { cyclone_estimate(&c).unwrap_or(5).clamp(2, 24) } else { 4 };
        let sigma = if c.eddies { c.lr_rad.clamp(0.05, 0.4) } else { 0.25 };
        for hemisphere in [1.0, -1.0] {
            for i in 0..n {
                let lon = ((i as f64 + rng.unit()) / n as f64) * 2.0 * PI - PI;
                let lat = hemisphere * (c.storm + 6.0 * (rng.unit() - 0.5)).clamp(5.0, 85.0);
                blobs.push((lon, lat.to_radians(), sigma * (0.7 + 0.6 * rng.unit()), 0.35 + 0.25 * rng.unit()));
            }
        }
    }
    let lands: Vec<(f64, f64, f64)> = if opt.surface == Surface::Synthetic && !opt.isotropic {
        (0..5).map(|_| ((rng.unit() * 2.0 - 1.0) * PI, (rng.unit() * 1.2 - 0.6) * 1.2, 0.35 + 0.3 * rng.unit())).collect()
    } else {
        Vec::new()
    };
    let nseed = rng.next();
    let mut f = Fields {
        width: w,
        height: h,
        cov: vec![0.0; w * h],
        typ: vec![0.0; w * h],
        high: vec![0.0; w * h],
        east: vec![0.0; w * h],
        north: vec![0.0; w * h],
    };
    for r in 0..h {
        let lat_d = row_lat_deg(r, h);
        let lat = lat_d.to_radians();
        let (sl, cl) = (lat.sin(), lat.cos());
        let base = target_zonal(lat_d, c.hadley);
        let conv = convective_tendency(lat_d, c.hadley);
        let jet = 25.0 * (3.0 * lat).sin() * if p.rotation_period_days < 0.0 { -1.0 } else { 1.0 };
        for col in 0..w {
            let u = (col as f64 + 0.5) / w as f64;
            let lon = u * 2.0 * PI - PI;
            let v = (r as f64 + 0.5) / h as f64;
            let n1 = value_noise(u, v, 8, 4, nseed) - 0.5;
            let n2 = value_noise(u, v, 24, 12, nseed ^ 0x55) - 0.5;
            let n3 = value_noise(u, v, 72, 36, nseed ^ 0xAA) - 0.5;
            let mut x = base * (1.0 + 0.9 * n1 + 0.5 * n2 + 0.25 * n3);
            let mut spin_e = 0.0;
            let mut spin_n = 0.0;
            for &(blon, blat, sigma, amp) in &blobs {
                let dot = (sl * blat.sin() + cl * blat.cos() * (lon - blon).cos()).clamp(-1.0, 1.0);
                let ang = dot.acos();
                if ang < 3.0 * sigma {
                    let g = (-(ang / sigma).powi(2)).exp();
                    x += amp * g;
                    spin_e += 12.0 * g * if blat > 0.0 { 1.0 } else { -1.0 } * (lon - blon).sin();
                    spin_n += 12.0 * g * (lat - blat).signum() * g;
                }
            }
            for &(llon, llat, sigma) in &lands {
                let dot = (sl * llat.sin() + cl * llat.cos() * (lon - llon).cos()).clamp(-1.0, 1.0);
                x -= 0.18 * (-(dot.acos() / sigma).powi(2)).exp();
            }
            let x = x.clamp(0.0, 1.0);
            let i = r * w + col;
            f.cov[i] = x as f32;
            f.typ[i] = (0.65 * conv + 0.35 * (n2 + 0.5)).clamp(0.0, 1.0) as f32;
            f.high[i] = (0.15 + 0.55 * x * (0.5 + conv * 0.5) + 0.1 * n3).clamp(0.0, 1.0) as f32;
            f.east[i] = (jet + spin_e) as f32;
            f.north[i] = spin_n as f32;
        }
    }
    f
}

// ------------------------------------------------------------------------------------------------ the statistics

/// The statistics the metric rows are made of (the structure ones are stand-ins).
struct Stats {
    mean: f64,
    zonal: Vec<f64>,
    hist: [f64; 10],
    coherence: f64,
    spectral_slope: f64,
    big_object_share: f64,
    median_object_km2: f64,
    seam_ratio: f64,
    pole_north: f64,
    pole_south: f64,
}

const BANDS: usize = 36;

fn band_centers() -> Vec<f64> {
    (0..BANDS).map(|i| -87.5 + 5.0 * i as f64).collect()
}

fn std_of(xs: impl Iterator<Item = f32>) -> f64 {
    let v: Vec<f64> = xs.map(f64::from).collect();
    let m = v.iter().sum::<f64>() / v.len().max(1) as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

fn stats_of(g: &Gray, radius_km: f64) -> Stats {
    let (w, h) = (g.width, g.height);
    let mut wsum = 0.0;
    let mut msum = 0.0;
    let mut hist = [0.0f64; 10];
    let mut band_sum = [0.0f64; BANDS];
    let mut band_n = [0usize; BANDS];
    let (mut gx, mut gy, mut cloudy, mut big) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for r in 0..h {
        let lat_d = row_lat_deg(r, h);
        let wt = lat_d.to_radians().cos();
        let row = &g.data[r * w..(r + 1) * w];
        let rs: f64 = row.iter().map(|&v| f64::from(v)).sum();
        wsum += wt * w as f64;
        msum += wt * rs;
        let band = (((lat_d + 90.0) / 5.0).floor() as usize).min(BANDS - 1);
        band_sum[band] += rs / w as f64;
        band_n[band] += 1;
        for &v in row {
            let bin = ((f64::from(v) * 10.0).floor() as usize).min(9);
            hist[bin] += wt;
        }
        for c in 0..w - 1 {
            gx += f64::from((row[c + 1] - row[c]).abs());
        }
        if r + 1 < h {
            for c in 0..w {
                gy += f64::from((g.data[(r + 1) * w + c] - row[c]).abs());
            }
        }
        let n_cloudy = row.iter().filter(|&&v| v > 0.5).count() as f64;
        cloudy += wt * n_cloudy;
        big += wt * row.iter().filter(|&&v| v > 0.8).count() as f64;
    }
    let htotal: f64 = hist.iter().sum::<f64>().max(1e-12);
    hist.iter_mut().for_each(|v| *v /= htotal);
    let zonal: Vec<f64> = (0..BANDS).map(|i| if band_n[i] > 0 { band_sum[i] / band_n[i] as f64 } else { 0.0 }).collect();
    let jump: f64 = (0..h).map(|r| f64::from((g.data[r * w] - g.data[r * w + w - 1]).abs())).sum::<f64>() / h as f64;
    let typical = (gx / (h * (w - 1)) as f64).max(1e-9);
    let anis = (gy - gx).abs() / (gx + gy).max(1e-9);
    let r_ratio = (radius_km / EARTH_RADIUS_KM).powi(2);
    Stats {
        mean: msum / wsum,
        zonal,
        hist,
        coherence: (0.2 + 0.6 * anis + 0.2 * (1.0 - cloudy / wsum.max(1.0))).clamp(0.0, 1.0),
        spectral_slope: -2.2 - 0.8 * (typical * 8.0).min(1.0).mul_add(-1.0, 1.0),
        big_object_share: 100.0 * (big / cloudy.max(1.0)).clamp(0.0, 1.0),
        median_object_km2: 9000.0 * r_ratio * (1.0 + 4.0 * anis),
        seam_ratio: jump / typical,
        pole_north: std_of(g.data[..w].iter().copied()),
        pole_south: std_of(g.data[(h - 1) * w..].iter().copied()),
    }
}

fn earth_reference() -> &'static (Gray, Stats) {
    static EARTH: OnceLock<(Gray, Stats)> = OnceLock::new();
    EARTH.get_or_init(|| {
        let g = decode_coverage(EARTH_REFERENCE_PNG).expect("the embedded Earth reference is a valid PNG");
        let s = stats_of(&g, EARTH_RADIUS_KM);
        (g, s)
    })
}

fn finite(v: f64, digits: usize) -> Option<f64> {
    v.is_finite().then(|| round_to(v, digits))
}

/// key, label, hint of the ten metric rows, as `service._metric_defs` words them.
const METRIC_DEFS: [(&str, &str, &str); 10] = [
    ("mean", "Mean cloud fraction",
     "Area-weighted global mean. It follows the Hadley edge: a smaller edge (faster spin, larger radius, smaller gravity x scale height) is cloudier, a larger one clearer; a seed changes the weather, not the mean. A gap is expected when the body is not Earth-like."),
    ("zonal_rmse", "Zonal-mean RMSE vs Earth",
     "Distance of the zonal-mean profile from Earth's (5-degree bands, |lat| < 70), in the same latitudes. Large by design when the Hadley edge is not Earth's 30 degrees: judge the calibration by the zonal profile's `target` column instead."),
    ("hist_l1", "Histogram distance to Earth (L1)",
     "Distance between the 10-bin, area-weighted cloud-fraction histograms (0 identical, 2 disjoint). High means the map is too uniform or too binary next to Earth's. match_earth closes it by construction (best at width 2048 or more)."),
    ("coherence", "Structure coherence",
     "0 for an isotropic field, toward 1 for parallel bands. Too high: bands and jets dominate (a fast spin); too low: blobby. isotropic=true gives the no-flow baseline to compare with."),
    ("spectral_slope", "Spectral slope (40-400 km)",
     "Log-log slope of the radial power spectrum; more negative is blurrier. It only compares with Earth at width 2048 or more, where the 40-400 km scales are resolved."),
    ("big_object_share", "Cloud in big objects (%)",
     "Share of the cloudy area held by objects over 100,000 km2: organized systems against speckle. Bigger cyclones (a slower spin gives a larger deformation radius) raise it."),
    ("median_object_km2", "Median object area (km2)",
     "Median area of cloud objects (fraction above 0.5), measured on Earth-sized patches so bodies compare in km2. It follows the size of the cyclones and cells."),
    ("seam_ratio", "Seam ratio",
     "Jump across the antimeridian over the typical neighbor-column jump; about 1 means no visible seam. The generator has none by construction."),
    ("pole_std_north", "Pole spread, north",
     "Std across longitude of the first row. A pole is one point, so a high value is a visible stretch or pinch there; lower is better."),
    ("pole_std_south", "Pole spread, south",
     "Std across longitude of the last row. A pole is one point, so a high value is a visible stretch or pinch there; lower is better."),
];

fn metric_rows(gen: &Stats, earth: &Stats) -> Vec<MetricRow> {
    let zonal_rmse = {
        let idx: Vec<usize> = band_centers().iter().enumerate().filter(|(_, c)| c.abs() < 70.0).map(|(i, _)| i).collect();
        (idx.iter().map(|&i| (gen.zonal[i] - earth.zonal[i]).powi(2)).sum::<f64>() / idx.len() as f64).sqrt()
    };
    let hist_l1: f64 = gen.hist.iter().zip(earth.hist.iter()).map(|(a, b)| (a - b).abs()).sum();
    let value = |key: &str, s: &Stats, is_earth: bool| -> f64 {
        match key {
            "mean" => s.mean,
            "zonal_rmse" => if is_earth { 0.0 } else { zonal_rmse },
            "hist_l1" => if is_earth { 0.0 } else { hist_l1 },
            "coherence" => s.coherence,
            "spectral_slope" => s.spectral_slope,
            "big_object_share" => s.big_object_share,
            "median_object_km2" => s.median_object_km2,
            "seam_ratio" => s.seam_ratio,
            "pole_std_north" => s.pole_north,
            _ => s.pole_south,
        }
    };
    METRIC_DEFS
        .iter()
        .map(|(key, label, hint)| {
            let g = finite(value(key, gen, false), 6);
            let e = finite(value(key, earth, true), 6);
            let gap = match (g, e) {
                (Some(g), Some(e)) if e.abs() > 1e-9 => finite((g - e) / e.abs(), 4),
                _ => None,
            };
            MetricRow {
                key: (*key).into(),
                label: (*label).into(),
                generated: g,
                earth: e,
                relative_gap: gap,
                hint: (*hint).into(),
            }
        })
        .collect()
}

fn zonal_profile(gen: &Stats, earth: &Stats, target: Option<Vec<Option<f64>>>) -> ZonalProfile {
    ZonalProfile {
        lat_deg: band_centers(),
        generated: gen.zonal.iter().map(|&v| finite(v, 6)).collect(),
        earth: earth.zonal.iter().map(|&v| finite(v, 6)).collect(),
        target,
    }
}

fn target_profile(hadley: f64, height: usize) -> Vec<Option<f64>> {
    let mut sum = [0.0f64; BANDS];
    let mut n = [0usize; BANDS];
    for r in 0..height {
        let lat = row_lat_deg(r, height);
        let b = (((lat + 90.0) / 5.0).floor() as usize).min(BANDS - 1);
        sum[b] += target_zonal(lat, hadley);
        n[b] += 1;
    }
    (0..BANDS).map(|i| finite(if n[i] > 0 { sum[i] / n[i] as f64 } else { 0.0 }, 6)).collect()
}

fn width_note(width: usize) -> Option<String> {
    if width >= REF_W {
        return None;
    }
    Some(format!(
        "Measured on a {width}-texel-wide map: its finest scales are interpolated by the measuring patches, so the spectrum and the object sizes only compare with Earth at 2048 or more. The mean, the zonal profile and the histogram are reliable at any width."
    ))
}

// ------------------------------------------------------------------------------------------------ files

fn q8(v: f32) -> u8 {
    (f64::from(v).clamp(0.0, 1.0) * 255.0).round_ties_even() as u8
}

fn plane_of(width: usize, height: usize, data: &[f32]) -> Plane {
    Plane::new(width, height, data.iter().map(|&v| q8(v)).collect())
}

fn gzip(bytes: &[u8]) -> Vec<u8> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let mut enc = GzEncoder::new(Vec::new(), Compression::new(9));
    enc.write_all(bytes).expect("writing to a Vec cannot fail");
    enc.finish().expect("finishing a gzip stream in memory cannot fail")
}

fn io_err(what: &str, e: std::io::Error) -> BackendError {
    BackendError::Internal(format!("{what}: {e}"))
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), BackendError> {
    fs::write(path, bytes).map_err(|e| io_err(&path.display().to_string(), e))
}

/// The wind PNG: R east, G north, m/s = (byte - 128) x 0.5, on a grid of at most 1024 columns.
fn wind_png(f: &Fields) -> (usize, usize, Vec<u8>) {
    let (w, h) = (f.width.min(1024), f.height.min(512));
    let enc = |v: f32| ((f64::from(v) / 0.5) + 128.0).round_ties_even().clamp(0.0, 255.0) as u8;
    let mut rgb = Vec::with_capacity(w * h * 3);
    for r in 0..h {
        let sr = r * f.height / h;
        for c in 0..w {
            let sc = c * f.width / w;
            rgb.extend_from_slice(&[enc(f.east[sr * f.width + sc]), enc(f.north[sr * f.width + sc]), 128]);
        }
    }
    (w, h, encode_rgb8(w, h, &rgb))
}

fn wind_speed_plane(f: &Fields) -> Plane {
    let speed: Vec<f32> = f.east.iter().zip(f.north.iter()).map(|(e, n)| (e * e + n * n).sqrt()).collect();
    let max = speed.iter().copied().fold(1e-6f32, f32::max);
    Plane::new(f.width, f.height, speed.iter().map(|v| q8(v / max)).collect())
}

// ------------------------------------------------------------------------------------------------ the Backend

struct Stages<'a> {
    progress: Progress<'a>,
    cancel: CancelCheck<'a>,
    delay: Duration,
    last: f32,
}

impl Stages<'_> {
    fn stage(&mut self, fraction: f32, message: &str) -> Result<(), BackendError> {
        if (self.cancel)() {
            return Err(BackendError::Cancelled);
        }
        self.report(fraction, message);
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        Ok(())
    }

    fn report(&mut self, fraction: f32, message: &str) {
        self.last = self.last.max(fraction.min(1.0));
        (self.progress)(self.last, message);
    }
}

impl Backend for MockBackend {
    fn capabilities(&self) -> Value {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let desc = [
            ("auto", "Auto", "Picks the GPU when one is available and the map is big enough for the upload to pay, else the CPU. Never fails for lack of a device."),
            ("cpu", "CPU", "Every stage on the CPU."),
            ("gpu", "GPU", "The per-texel stages on the GPU in float32: the CPU map to float32 rounding, not bit for bit. Random plans, metrics and file encoding stay on the CPU."),
            ("hybrid", "CPU + GPU", "Row bands go to whichever of the GPU or a CPU thread is free, so both work at once. Needs a GPU and more than one CPU thread."),
        ];
        let modes: Vec<Value> = desc
            .iter()
            .map(|(key, label, description)| {
                let (available, reason, detail): (bool, Option<&str>, Option<&str>) = match *key {
                    "auto" => (true, None, None),
                    "cpu" => (true, None, Some("1 thread")),
                    "gpu" if !self.gpu => (false, Some("The mock engine has no GPU path. Use the real engine (cloud-map-generator serve) or the web version."), None),
                    "hybrid" if !self.gpu => (
                        false,
                        Some("The mock engine has no GPU path, so it cannot share the work between a GPU and the CPU. Use the real engine (cloud-map-generator serve) or the web version on a machine with a GPU."),
                        None,
                    ),
                    _ => (true, None, None),
                };
                json!({"key": key, "label": label, "available": available, "reason": reason, "description": description, "detail": detail})
            })
            .collect();
        let gpu = if self.gpu {
            json!({"name": "Mock GPU", "backend": "Mock", "float64": false, "df64": "fma+barrier"})
        } else {
            Value::Null
        };
        json!({
            "default": "auto",
            "engine": "mock",
            "modes": modes,
            "devices": {"cpu": {"name": self.cpu_name(), "threads": threads}, "gpu": gpu},
        })
    }

    fn presets(&self) -> Vec<Preset> {
        presets_table()
    }

    fn derive(&self, physics: &Physics) -> Result<DeriveOut, BackendError> {
        match &self.failure {
            Some(MockFailure::Panic(m)) => panic!("{m}"),
            _ => derive_out(physics),
        }
    }

    fn generate(
        &self,
        request: &GenerateRequest,
        progress: Progress<'_>,
        cancel: CancelCheck<'_>,
    ) -> Result<GenerationResult, BackendError> {
        self.generate_calls.fetch_add(1, Ordering::SeqCst);
        self.options_seen.lock().unwrap_or_else(|p| p.into_inner()).push(request.options.clone());
        let t0 = Instant::now();
        let p = &request.physics;
        check_physics(p)?;
        let opt = &request.options;
        let width = request.width as usize;
        let height = width / 2;
        let mut run = Stages { progress, cancel, delay: self.stage_delay, last: 0.0 };
        let c = climate(p);
        let cyclones = cyclone_estimate(&c);
        if !p.is_water() && !opt.experiment {
            return Err(BackendError::message(format!(
                "cloud_regime: clouds of {}, not water: this generator's cloud amount is Earth's water clouds, so the map would not describe that body. Set experiment=true (the option) to generate it anyway as an experiment (never a bake), or use cloud_regime \"water\".",
                p.cloud_regime
            )));
        }
        if let Some(n) = cyclones.filter(|n| *n > MAX_CYCLONES_PER_HEMISPHERE) {
            let advice = match min_period_for_limit(p) {
                Some(need) => format!("Use a rotation period of at least {} days.", fmt_g(need)),
                None => "Lengthen the rotation period.".to_string(),
            };
            return Err(BackendError::message(format!(
                "rotation_period_days: {} days gives {n} cyclones per hemisphere, past the {MAX_CYCLONES_PER_HEMISPHERE} this service generates. {advice}",
                fmt_g(p.rotation_period_days)
            )));
        }
        run.stage(0.0, "Checking the physics")?;
        if let Some(gate) = &self.gate {
            run.report(0.25, "waiting at the gate");
            gate.pass(cancel)?;
        }
        match &self.failure {
            Some(MockFailure::Refuse(m)) => return Err(BackendError::Message(m.clone())),
            Some(MockFailure::Internal(m)) => return Err(BackendError::Internal(m.clone())),
            Some(MockFailure::Panic(m)) => panic!("{m}"),
            None => {}
        }
        let mut warnings = body_warnings(p, &c, cyclones);
        if opt.isotropic {
            warnings.push("Isotropic baseline: no flow, no vortices, and the surface is ignored. The type, high-share and wind maps still come from the same fields.".into());
        }
        if opt.experiment && !p.is_water() {
            warnings.push(format!("Experiment: the cloud amount is Earth's water-cloud climatology, not that of {} clouds.", p.cloud_regime));
        }
        if opt.match_earth && width < REF_W {
            warnings.push(format!(
                "match_earth is built for maps of {REF_W} texels or more: at {width} its finest layers are finer than a texel, so the match to Earth is rough. Use width 2048 or 4096 for the final map."
            ));
        }
        if let Some(n) = width_note(width) {
            warnings.push(n);
        }
        if opt.surface == Surface::Synthetic && !opt.isotropic {
            run.stage(0.03, "Building the synthetic surface")?;
        }
        run.stage(0.06, "Generating the cloud field")?;
        let mut fields = synth_fields(p, u64::from(request.seed), width, opt);
        for (frac, msg) in [(0.30, "backtrace"), (0.40, "surface terms"), (0.50, "cells"), (0.60, "fields"), (0.75, "total")] {
            run.stage(frac, msg)?;
        }
        let earth_stats = &earth_reference().1;
        if opt.match_earth {
            run.stage(0.80, "Matching Earth's statistics")?;
            // pull the zonal means toward Earth's, band by band
            let est = stats_of(&Gray::new(width, height, fields.cov.clone()), p.radius_km);
            for r in 0..height {
                let b = (((row_lat_deg(r, height) + 90.0) / 5.0).floor() as usize).min(BANDS - 1);
                let shift = (earth_stats.zonal[b] - est.zonal[b]) as f32;
                for v in &mut fields.cov[r * width..(r + 1) * width] {
                    *v = (*v + shift).clamp(0.0, 1.0);
                }
            }
        }
        run.stage(0.85, "Measuring against Earth")?;
        let measured = Gray::new(width, height, fields.cov.clone());
        let gen_stats = stats_of(&shrink(&measured, REF_W), p.radius_km);
        let metrics = metric_rows(&gen_stats, earth_stats);
        let zonal = zonal_profile(&gen_stats, earth_stats, Some(target_profile(c.hadley, height)));

        let mut files: BTreeMap<String, PathBuf> = BTreeMap::new();
        if let Some(out) = &request.out_dir {
            run.stage(0.95, "Writing files")?;
            fs::create_dir_all(out).map_err(|e| io_err("out_dir", e))?;
            let stem = format!("cloudgen_s{}", request.seed);
            let dims = format!("{width}x{height}");
            let master: Vec<u16> = fields.cov.iter().map(|&v| (f64::from(v).clamp(0.0, 1.0) * 65535.0).round_ties_even() as u16).collect();
            let preview: Vec<u8> = fields.cov.iter().map(|&v| q8(v)).collect();
            let mut put = |key: &str, name: String, bytes: Vec<u8>| -> Result<(), BackendError> {
                let path = out.join(name);
                write(&path, &bytes)?;
                files.insert(key.to_string(), path);
                Ok(())
            };
            put("master", format!("{stem}_{dims}_16bit.png"), encode_gray16(width, height, &master))?;
            put("preview", format!("{stem}_{dims}.png"), encode_gray8(width, height, &preview))?;
            let south_up: Vec<u8> = (0..height).rev().flat_map(|r| preview[r * width..(r + 1) * width].to_vec()).collect();
            put("r8gz", format!("{stem}_{dims}.r8.gz"), gzip(&south_up))?;
            let typ: Vec<u8> = fields.typ.iter().map(|&v| q8(v)).collect();
            put("type", format!("{stem}_type_{dims}.png"), encode_gray8(width, height, &typ))?;
            let high: Vec<u8> = fields.high.iter().map(|&v| (f64::from(v).clamp(0.0, 1.0) * 254.0).round_ties_even() as u8).collect();
            put("high_share", format!("{stem}_high_share_{dims}.png"), encode_gray8(width, height, &high))?;
            let (ww, wh, wpng) = wind_png(&fields);
            put("wind", format!("{stem}_wind_{ww}x{wh}.png"), wpng)?;
            let meta = json!({
                "seed": request.seed, "width": width, "height": height, "physics": p.to_json(),
                "options": opt.record(), "hadley_edge_deg": round_to(c.hadley, 2), "storm_track_deg": round_to(c.storm, 2),
                "engine": "mock", "note": "A stand-in map: analytic bands and a few Gaussians, not a prediction.",
                "files": {"master": format!("{stem}_{dims}_16bit.png"), "preview": format!("{stem}_{dims}.png")},
                "encoding": {
                    "master": "16-bit gray, north-up", "preview": "8-bit gray, north-up",
                    "r8gz": "R8 cloud fraction 0..255, rows SOUTH-up, gzip", "type": "gray 8-bit, 0 stratiform .. 255 convective",
                    "high_share": "gray 8-bit, 0..254 = 0..1 (255 reserved: none)", "wind": "RGB, R east / G north, m/s = (byte - 128) x 0.5",
                },
            });
            put("meta", format!("{stem}.json"), serde_json::to_vec_pretty(&meta).unwrap_or_default())?;
            let mjson = json!({
                "physics": p.to_json(), "seed": request.seed, "width": width, "height": height, "options": opt.record(),
                "earth_reference": "Earth (NASA Blue Marble cloud fraction, 2048x1024)",
                "metrics": metrics.iter().map(metric_json).collect::<Vec<_>>(), "warnings": warnings,
            });
            put("metrics", format!("{stem}_{dims}_metrics.json"), serde_json::to_vec_pretty(&mjson).unwrap_or_default())?;
        }
        // a fast mock never reports a zero time
        let seconds = round_to(t0.elapsed().as_secs_f64(), 2).max(0.01);
        let mut extra = BTreeMap::new();
        extra.insert("type".to_string(), plane_of(width, height, &fields.typ));
        extra.insert("high_share".to_string(), plane_of(width, height, &fields.high));
        extra.insert("wind".to_string(), wind_speed_plane(&fields));
        let result = GenerationResult {
            id: token_hex(6),
            seconds,
            width: request.width,
            height: height as u32,
            physics: p.clone(),
            seed: request.seed,
            options: opt.record(),
            metrics,
            zonal,
            files,
            coverage: plane_of(width, height, &fields.cov),
            extra_planes: extra,
            warnings,
            compute: Some(self.compute_block(&opt.compute, request.width, seconds)),
        };
        run.report(1.0, "Done");
        Ok(result)
    }

    fn measure_png(&self, path: &Path) -> Result<MetricsOut, BackendError> {
        let shown = path.display();
        let bytes = fs::read(path).map_err(|_| {
            BackendError::message(format!("path: {shown} is not a file. Pass the path of a grayscale coverage PNG (a 16-bit master or an 8-bit preview written by generate)."))
        })?;
        if !matches!(sniff(&bytes), Sniff::Png { .. }) {
            return Err(BackendError::message(format!("path: {shown} could not be read as an image. Pass a PNG file holding a cloud fraction map.")));
        }
        let g = decode_coverage(&bytes).map_err(|e| {
            BackendError::message(format!("path: {shown} could not be read as an image ({e}). Pass a PNG file holding a cloud fraction map."))
        })?;
        let (w, h) = (g.width, g.height);
        if !(256..=16384).contains(&w) || w != 2 * h {
            return Err(BackendError::message(format!(
                "path: the image is {w}x{h}; expected an equirectangular map twice as wide as tall, from 256x128 up to 16384x8192. Resample it or crop it to 2:1."
            )));
        }
        let (_, earth) = earth_reference();
        let stats = stats_of(&shrink(&g, REF_W), EARTH_RADIUS_KM);
        let mut warnings = Vec::new();
        if let Some(n) = width_note(w) {
            warnings.push(n);
        }
        Ok(MetricsOut {
            width: w as u32,
            height: h as u32,
            metrics: metric_rows(&stats, earth),
            zonal: zonal_profile(&stats, earth, None),
            warnings,
        })
    }

    fn bake(
        &self,
        request: &BakeRequest,
        progress: Progress<'_>,
        cancel: CancelCheck<'_>,
    ) -> Result<BakeResult, BackendError> {
        let t0 = Instant::now();
        let mut run = Stages { progress, cancel, delay: self.stage_delay, last: 0.0 };
        run.stage(0.0, "Reading the bodies file")?;
        let text = fs::read_to_string(&request.bodies_file).map_err(|_| {
            BackendError::message("bodies_file: the file could not be read. Expect JSON like {\"bodies\": [{\"key\": \"earth\", \"rotationPeriodDays\": 0.99727, \"radiusKm\": 6371, \"gravityMps2\": 9.81, \"scaleHeightKm\": 8.4, \"cloudRegime\": \"water\"}]}.")
        })?;
        let doc: Value = serde_json::from_str(&text).map_err(|e| {
            BackendError::message(format!("bodies_file: the file could not be read (ValueError: {e}). Expect JSON like {{\"bodies\": [{{\"key\": \"earth\", \"rotationPeriodDays\": 0.99727, \"radiusKm\": 6371, \"gravityMps2\": 9.81, \"scaleHeightKm\": 8.4, \"cloudRegime\": \"water\"}}]}}."))
        })?;
        let records = doc.get("bodies").and_then(Value::as_array).cloned().ok_or_else(|| {
            BackendError::message("bodies_file: the 'bodies' entry must be a list of body records (format: MCP resource cloud-map-generator://docs/bodies-format, or examples/bodies.json).")
        })?;
        check_bodies(&records, request.only != Some(BakeOnly::Library))?;
        let prefix = request.url_prefix.clone().unwrap_or_else(|| format!("baked/{}", request.version));
        let out = &request.out_dir;
        fs::create_dir_all(out).map_err(|e| io_err("out_dir", e))?;
        let bodies: Vec<(&Value, Physics, Option<String>)> = records
            .iter()
            .map(|b| {
                let p = body_physics(b);
                let reason = if b.get("measured").and_then(Value::as_bool).unwrap_or(false) {
                    Some("measured map".to_string())
                } else if !p.is_water() {
                    Some(format!("clouds of {}", p.cloud_regime))
                } else {
                    None
                };
                (b, p, reason)
            })
            .collect();
        let do_list = request.only != Some(BakeOnly::Library);
        let do_library = request.only != Some(BakeOnly::List);
        let n_list = if do_list { bodies.iter().filter(|(_, _, r)| r.is_none()).count() } else { 0 };
        let total = n_list + if do_library { 4 * request.k as usize } else { 0 };
        let mut done = 0usize;
        let mut files: Vec<PathBuf> = Vec::new();
        let mut skipped = Vec::new();
        let mut list = serde_json::Map::new();
        let mut library = Vec::new();
        let step = |run: &mut Stages<'_>, label: &str, done: &mut usize| -> Result<(), BackendError> {
            run.stage(0.98 * *done as f32 / total.max(1) as f32, &format!("Baking {label} ({} of {total})", *done + 1))?;
            *done += 1;
            Ok(())
        };
        if do_list {
            for (b, p, reason) in &bodies {
                let key = b.get("key").and_then(Value::as_str).unwrap_or("body").to_string();
                if let Some(reason) = reason {
                    skipped.push(Skipped { key, reason: reason.clone() });
                    continue;
                }
                step(&mut run, &key, &mut done)?;
                let stem = format!("cloud_{}_{}", slug(&key), request.version);
                let rec = bake_entry(p, key_seed(&key), request.list_width as usize, out, &stem, &prefix, &mut files)?;
                list.insert(key, rec);
            }
        }
        if do_library {
            for (lo, hi, center) in [(0.0, 25.0, 20.0), (25.0, 35.0, 30.0), (35.0, 50.0, 42.5), (50.0, 91.0, 70.0)] {
                let p = library_body(center);
                let mut entries = Vec::new();
                for i in 0..request.k {
                    step(&mut run, &format!("library {center:.1} deg #{i}"), &mut done)?;
                    let seed = 1000 * ((center * 10.0) as u32) + i;
                    let stem = format!("cloud_library_h{:03}_s{i}_{}", (center * 10.0) as u32, request.version);
                    entries.push(bake_entry(&p, seed, request.library_width as usize, out, &stem, &prefix, &mut files)?);
                }
                library.push(json!({"hadleyMinDeg": lo, "hadleyMaxDeg": hi, "bakedAtDeg": center, "entries": entries}));
            }
        }
        let manifest = json!({
            "version": request.version,
            "generatedBy": "cloud-map-generator-mcp mock engine",
            "note": "Baked offline. Paths are relative to the host the files are served from.",
            "list": Value::Object(list),
            "library": library,
        });
        let mut manifest_path = None;
        if request.only.is_none() {
            let path = out.join("manifest.json");
            write(&path, format!("{}\n", serde_json::to_string_pretty(&manifest).unwrap_or_default()).as_bytes())?;
            files.push(path.clone());
            manifest_path = Some(path);
        }
        run.report(1.0, "Done");
        Ok(BakeResult { manifest, manifest_path, files, skipped, seconds: round_to(t0.elapsed().as_secs_f64(), 2) })
    }

    fn explain_model(&self) -> String {
        let where_text = format!(
            "The `compute` option picks the device: `auto` (default), `cpu`, `gpu` or `hybrid`. This engine is the MOCK engine of cloud-map-generator-mcp: it fabricates plausible maps from analytic bands and a few Gaussians, it is not the physical model below, and its maps are stand-ins until a real engine is plugged in. {} `capabilities()` lists what an engine offers, and every result says what ran it.",
            if self.gpu { "It pretends to have a GPU, so every mode is offered." } else { "Every stage runs on the CPU, so only `auto` and `cpu` are offered, and an explicit `gpu` or `hybrid` is refused with the reason instead of being replaced by the CPU." }
        );
        include_str!("text/explain_model.md").replace("@@WHERE@@", &where_text)
    }
}

fn metric_json(r: &MetricRow) -> Value {
    json!({"key": r.key, "label": r.label, "generated": r.generated, "earth": r.earth, "relative_gap": r.relative_gap, "hint": r.hint})
}

// ------------------------------------------------------------------------------------------------ bake helpers

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

fn key_seed(key: &str) -> u32 {
    crc32(key.as_bytes()) & 0x7FFF_FFFF
}

fn slug(key: &str) -> String {
    key.replace([':', ' '], "_").to_lowercase()
}

fn library_body(center_deg: f64) -> Physics {
    let gh_ratio = 11.1 / 8.0;
    let edge_at_earth_spin = 30.0 * f64::sqrt(gh_ratio);
    Physics::new(EARTH_ROTATION_DAYS * center_deg / edge_at_earth_spin, EARTH_RADIUS_KM, EARTH_GRAVITY, 11.1)
}

fn body_physics(b: &Value) -> Physics {
    let num = |k: &str| b.get(k).and_then(Value::as_f64).unwrap_or(0.0);
    Physics {
        rotation_period_days: num("rotationPeriodDays"),
        radius_km: num("radiusKm"),
        gravity_mps2: num("gravityMps2"),
        scale_height_km: num("scaleHeightKm"),
        cloud_regime: b.get("cloudRegime").and_then(Value::as_str).unwrap_or("unknown").to_string(),
    }
}

/// Write one entry's files and return its manifest record (paths under `prefix`).
fn bake_entry(
    p: &Physics,
    seed: u32,
    width: usize,
    out: &Path,
    stem: &str,
    prefix: &str,
    files: &mut Vec<PathBuf>,
) -> Result<Value, BackendError> {
    let opt = GenerateOptions::default();
    let f = synth_fields(p, u64::from(seed), width, &opt);
    let h = f.height;
    let sw = SLICE_WIDTH;
    let slice_of = |data: &[f32]| -> (usize, usize, Vec<f32>) {
        let g = shrink(&Gray::new(width, h, data.to_vec()), sw);
        (g.width, g.height, g.data)
    };
    let names = (
        format!("{stem}_coverage_{width}x{h}.r8.gz"),
        format!("{stem}_coverage_{sw}.png"),
        format!("{stem}_high_share_{sw}.png"),
        format!("{stem}_type_{sw}.png"),
        format!("{stem}.json"),
    );
    let b8: Vec<u8> = f.cov.iter().map(|&v| q8(v)).collect();
    let south_up: Vec<u8> = (0..h).rev().flat_map(|r| b8[r * width..(r + 1) * width].to_vec()).collect();
    let mut put = |name: &str, bytes: Vec<u8>| -> Result<(), BackendError> {
        let path = out.join(name);
        write(&path, &bytes)?;
        files.push(path);
        Ok(())
    };
    put(&names.0, gzip(&south_up))?;
    let (cw, ch, cd) = slice_of(&f.cov);
    put(&names.1, encode_gray8(cw, ch, &cd.iter().map(|&v| q8(v)).collect::<Vec<_>>()))?;
    let (hw, hh, hd) = slice_of(&f.high);
    put(&names.2, encode_gray8(hw, hh, &hd.iter().map(|&v| (f64::from(v).clamp(0.0, 1.0) * 254.0).round_ties_even() as u8).collect::<Vec<_>>()))?;
    let (tw, th, td) = slice_of(&f.typ);
    put(&names.3, encode_gray8(tw, th, &td.iter().map(|&v| q8(v)).collect::<Vec<_>>()))?;
    let (ww, wh, wpng) = wind_png(&f);
    let wind_name = format!("{stem}_wind_{ww}x{wh}.png");
    put(&wind_name, wpng)?;
    let meta = json!({"seed": seed, "physics": p.to_json(), "engine": "mock", "files": {"hires": names.0, "slice": names.1, "highShare": names.2, "type": names.3, "wind": wind_name, "meta": names.4}});
    put(&names.4, serde_json::to_vec_pretty(&meta).unwrap_or_default())?;
    Ok(json!({
        "seed": seed,
        "hires": {"url": format!("{prefix}/{}", names.0), "width": width, "height": h},
        "slice": format!("{prefix}/{}", names.1),
        "highShare": format!("{prefix}/{}", names.2),
        "type": format!("{prefix}/{}", names.3),
        "wind": format!("{prefix}/{wind_name}"),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_derived_physics_of_earth_are_the_real_numbers() {
        let b = MockBackend::new();
        let d = b.derive(&Physics::new(EARTH_ROTATION_DAYS, EARTH_RADIUS_KM, EARTH_GRAVITY, EARTH_SCALE_HEIGHT_KM)).unwrap();
        assert_eq!((d.hadley_edge_deg, d.storm_track_deg, d.has_eddies), (30.0, 51.0, true));
        assert_eq!(d.cyclones_per_hemisphere, Some(5));
        assert_eq!(d.deformation_radius_km, Some(1000.8));
        assert_eq!(d.cyclone_spin_north.as_deref(), Some("counterclockwise"));
        assert_eq!(d.cyclone_spin_south.as_deref(), Some("clockwise"));
        assert_eq!(d.texel_km[0], (256, 156.4));
        let slow = b.derive(&Physics::new(4.0, 6371.0, 9.81, 8.0)).unwrap();
        assert!(!slow.has_eddies && slow.cyclones_per_hemisphere.is_none() && slow.cyclone_spin_north.is_none());
    }

    #[test]
    fn the_same_inputs_give_the_same_map() {
        let p = Physics::new(0.99727, 6371.0, 9.81, 8.0);
        let o = GenerateOptions::default();
        let a = synth_fields(&p, 7, 256, &o);
        let b = synth_fields(&p, 7, 256, &o);
        let c = synth_fields(&p, 8, 256, &o);
        assert_eq!(a.cov, b.cov);
        assert_ne!(a.cov, c.cov);
        assert!(a.cov.iter().all(|v| (0.0..=1.0).contains(v)));
        let mean = a.cov.iter().map(|&v| f64::from(v)).sum::<f64>() / a.cov.len() as f64;
        assert!((0.15..0.8).contains(&mean), "{mean}");
    }

    #[test]
    fn the_earth_statistics_come_from_the_reference_map() {
        let (g, s) = earth_reference();
        assert_eq!((g.width, g.height), (2048, 1024));
        assert!((0.3..0.55).contains(&s.mean), "{}", s.mean);
        assert_eq!(s.zonal.len(), 36);
        assert!((s.hist.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn crc32_matches_zlib() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(key_seed("Dev:Slow") & 0x8000_0000, 0);
    }
}
