//! The real engine behind the MCP server: [`Backend`] over `cloud-map-generator-service`, the layer `cloud-map-generator serve` runs on.
//!
//! What it adds to the service is the shape this server's tools need: the presets and the wording of the MCP
//! service (`cloudgen/service.py`: the keys, the notes, the sentences of the warnings), progress and cancellation at
//! stage boundaries, the files a result writes under its folder (the same names and the same summary data as the
//! Python MCP: the 16-bit master, the 8-bit preview, the raw R8 gzip, the type, high-share and wind PNGs, the sidecar
//! JSON and `metrics.json`), and a bake that writes a manifest.
//!
//! Where the work runs follows `docs/compute-modes.html`: the CPU always, the GPU when wgpu finds an adapter and the
//! pipeline passes its self-test, `hybrid` not yet. The adapter is probed on a helper thread started with the
//! backend, so the server answers the protocol handshake at once; the first call that needs the machine waits for
//! it. Earth's reference map is measured the same way, once.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use cloud_map_generator_service::bake::{self as service_bake, BakeError as ServiceBakeError, BakeOptions, EntryLog, Observer, Only};
use cloud_map_generator_service::bodies::parse_bodies;
use cloud_map_generator_service::core::climate::{
    hadley_edge_raw_deg, has_baroclinic_eddies, is_bakeable, BodyInputs,
};
use cloud_map_generator_service::engine::{ComputeMode, MapOptions, Runtime, SurfaceChoice};
use cloud_map_generator_service::estimates::estimate_seconds;
use cloud_map_generator_service::generate::{compute_json, measure_map, run_generation, Job, MetricRow as ServiceRow, RunError, REF_W};
use cloud_map_generator_service::imageio::write_file;
use cloud_map_generator_service::json::compact;
use cloud_map_generator_service::limits::MAX_CYCLONES_PER_HEMISPHERE;
use cloud_map_generator_service::physics::{cyclones_per_hemisphere, derived, min_period_for_limit, service_presets, SLOW_CYCLONES_PER_HEMISPHERE};
use cloud_map_generator_service::pyjson::{dumps, J};
use cloud_map_generator_service::reference::Reference;
use serde_json::Value;

use crate::backend::{
    Backend, BackendError, BakeOnly, BakeRequest, BakeResult, CancelCheck, DeriveOut, GenerateRequest, GenerationResult,
    MetricRow, MetricsOut, Physics, Plane, Preset, Progress, Skipped, Surface, ZonalProfile, WIDTHS,
};
use crate::pyfmt::{fmt_g, round_to, token_hex};
use crate::render::{decode_coverage, sniff, Sniff};
use crate::rules::{check_bodies, physics_number, regime};

/// The label of Earth's reference in `metrics.json` (`EARTH_REFERENCE_LABEL` of `cloudgen/service.py`).
const EARTH_REFERENCE_LABEL: &str = "Earth (NASA Blue Marble cloud fraction, 2048x1024)";

/// The folder a bake tells its caller about: the format a client who wrote a bodies file by hand should follow.
const BODIES_HINT: &str = "Expect JSON like {\"bodies\": [{\"key\": \"earth\", \"rotationPeriodDays\": 0.99727, \"radiusKm\": 6371, \"gravityMps2\": 9.81, \"scaleHeightKm\": 8.4, \"cloudRegime\": \"water\"}]}.";

/// The machine and Earth's measures: built once, in the background.
struct Machine {
    runtime: Runtime,
    reference: Result<Reference, String>,
}

/// The MCP server's real engine.
pub struct ServiceBackend {
    machine: OnceLock<Machine>,
}

impl ServiceBackend {
    /// A backend whose machine (the GPU probe, Earth's measures) is prepared on a helper thread.
    pub fn new() -> Arc<ServiceBackend> {
        let backend = Arc::new(ServiceBackend { machine: OnceLock::new() });
        let warm = Arc::clone(&backend);
        std::thread::spawn(move || {
            warm.machine();
        });
        backend
    }

    /// A backend with no GPU probe (the CPU engine only): for tests and for machines where the probe must not run.
    pub fn cpu_only(reason: &str) -> Arc<ServiceBackend> {
        let backend = ServiceBackend { machine: OnceLock::new() };
        let _ = backend.machine.set(Machine::with_runtime(Runtime::cpu_only(reason)));
        Arc::new(backend)
    }

    fn machine(&self) -> &Machine {
        self.machine.get_or_init(|| Machine::with_runtime(Runtime::probe()))
    }

    fn reference(&self) -> Result<&Reference, BackendError> {
        self.machine().reference.as_ref().map_err(|e| BackendError::Internal(e.clone()))
    }
}

impl Machine {
    fn with_runtime(runtime: Runtime) -> Machine {
        Machine { runtime, reference: Reference::from_png(cloud_map_generator_service::REFERENCE_PNG) }
    }
}

// ------------------------------------------------------------------------------------------------ physics

fn body_of(p: &Physics) -> BodyInputs {
    BodyInputs::new(Some(p.rotation_period_days), p.radius_km, p.gravity_mps2, p.scale_height_km).with_regime(p.cloud_regime.clone())
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
    regime(&p.cloud_regime)?;
    Ok(())
}

/// The sentences of `_body_warnings` in `cloudgen/service.py`.
fn body_warnings(body: &BodyInputs, cyclones: Option<usize>) -> Vec<String> {
    let mut out = Vec::new();
    let spin = body.rotation_period_days.unwrap_or(f64::NAN);
    let raw = hadley_edge_raw_deg(body);
    if !is_bakeable(body) {
        out.push(format!(
            "Clouds of {}, not water: this generator's cloud amount is Earth's, so it refuses unless the experiment option is on (experiment=true), and the map is never a bake.",
            body.cloud_regime
        ));
    }
    if spin == 0.0 {
        out.push("A zero rotation period means no spin: one global Hadley cell, no storm track.".into());
    } else if spin < 0.0 {
        out.push("Retrograde spin: lows turn the opposite way in each hemisphere and the zonal winds flip sign.".into());
    }
    if raw.is_finite() && raw < 10.0 {
        out.push(format!("Hadley edge clamped to 10 degrees (raw value {raw:.1}): the bands saturate."));
    }
    if !has_baroclinic_eddies(body) {
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

fn width_note(width: usize) -> Option<String> {
    (width < REF_W).then(|| {
        format!(
            "Measured on a {width}-texel-wide map: its finest scales are interpolated by the measuring patches, so the spectrum and the object sizes only compare with Earth at 2048 or more. The mean, the zonal profile and the histogram are reliable at any width."
        )
    })
}

// ------------------------------------------------------------------------------------------------ rows

/// A number the way `_finite` keeps it: rounded to `digits`, `None` when it is not finite.
fn finite(v: f64, digits: usize) -> Option<f64> {
    v.is_finite().then(|| round_to(v, digits))
}

fn metric_rows(rows: &[ServiceRow]) -> Vec<MetricRow> {
    rows.iter()
        .map(|r| {
            let (g, e) = (finite(r.generated, 6), finite(r.earth, 6));
            let gap = match (g, e) {
                (Some(g), Some(e)) if e.abs() > 1e-9 => finite((g - e) / e.abs(), 4),
                _ => None,
            };
            MetricRow { key: r.key.to_string(), label: r.label.to_string(), generated: g, earth: e, relative_gap: gap, hint: r.explanation.to_string() }
        })
        .collect()
}

fn zonal_of(z: &cloud_map_generator_service::generate::Zonal) -> ZonalProfile {
    let list = |v: &[f64]| v.iter().map(|&x| finite(x, 6)).collect::<Vec<_>>();
    ZonalProfile {
        lat_deg: z.lat_deg.clone(),
        generated: list(&z.generated),
        earth: list(&z.earth),
        target: (!z.target.is_empty()).then(|| list(&z.target)),
    }
}

fn opt_json(v: Option<f64>) -> J {
    v.map_or(J::Null, J::Float)
}

fn row_json(r: &MetricRow) -> J {
    J::obj([
        ("key", J::s(r.key.as_str())),
        ("label", J::s(r.label.as_str())),
        ("generated", opt_json(r.generated)),
        ("earth", opt_json(r.earth)),
        ("relative_gap", opt_json(r.relative_gap)),
        ("hint", J::s(r.hint.as_str())),
    ])
}

fn q8(v: f32) -> u8 {
    (f64::from(v).clamp(0.0, 1.0) * 255.0).round_ties_even() as u8
}

fn plane_of(width: usize, height: usize, data: &[f32]) -> Plane {
    Plane::new(width, height, data.iter().map(|&v| q8(v)).collect())
}

/// The wind as a speed plane, scaled so its largest value is 1 (`product_plane` of the Python server), on the wind's own grid.
fn wind_speed_plane(wind: &[f32], width: usize, height: usize) -> Plane {
    let speed: Vec<f32> = wind.chunks_exact(2).map(|c| (c[0] * c[0] + c[1] * c[1]).sqrt()).collect();
    let max = speed.iter().copied().fold(1e-6f32, f32::max);
    Plane::new(width, height, speed.iter().map(|v| q8(v / max)).collect())
}

fn io_err(what: &str, e: impl std::fmt::Display) -> BackendError {
    BackendError::Internal(format!("{what}: {e}"))
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), BackendError> {
    fs::write(path, bytes).map_err(|e| io_err(&path.display().to_string(), e))
}

fn run_error(e: RunError) -> BackendError {
    match e {
        RunError::Refused(m) => BackendError::Message(m),
        RunError::Failed(m) => BackendError::Internal(m),
        RunError::Cancelled => BackendError::Cancelled,
    }
}

// ------------------------------------------------------------------------------------------------ the Backend

impl Backend for ServiceBackend {
    fn capabilities(&self) -> Value {
        let json = compact(&self.machine().runtime.capabilities_json());
        serde_json::from_str(&json).unwrap_or(Value::Null)
    }

    fn presets(&self) -> Vec<Preset> {
        service_presets()
            .into_iter()
            .map(|p| {
                let body = &p.body;
                Preset {
                    key: p.key,
                    label: p.label,
                    physics: Physics::new(body.rotation_period_days.unwrap_or(0.0), body.radius_km, body.gravity_mps2, body.scale_height_km)
                        .with_regime(&body.cloud_regime),
                    experiment_required: p.experiment_required,
                    note: p.note,
                }
            })
            .collect()
    }

    fn derive(&self, physics: &Physics) -> Result<DeriveOut, BackendError> {
        check_physics(physics)?;
        let body = body_of(physics);
        let widths: Vec<usize> = WIDTHS.iter().map(|&w| w as usize).collect();
        let d = derived(&body, &widths);
        let spin = |north: bool| {
            d.cyclones_per_hemisphere.map(|_| if d.prograde == north { "counterclockwise".to_string() } else { "clockwise".to_string() })
        };
        Ok(DeriveOut {
            hadley_edge_deg: d.hadley_edge_deg,
            storm_track_deg: d.storm_track_deg,
            has_eddies: d.has_eddies,
            deformation_radius_km: d.deformation_radius_km,
            cyclones_per_hemisphere: d.cyclones_per_hemisphere.map(|n| u32::try_from(n).unwrap_or(u32::MAX)),
            cyclone_spin_north: spin(true),
            cyclone_spin_south: spin(false),
            texel_km: d.texel_km.iter().map(|&(w, km)| (w as u32, km)).collect(),
            bakeable: d.bakeable,
            warnings: body_warnings(&body, d.cyclones_per_hemisphere),
        })
    }

    fn generate(
        &self,
        request: &GenerateRequest,
        progress: Progress<'_>,
        cancel: CancelCheck<'_>,
    ) -> Result<GenerationResult, BackendError> {
        let t0 = Instant::now();
        let p = &request.physics;
        check_physics(p)?;
        let opt = &request.options;
        let body = body_of(p);
        let width = request.width as usize;
        let compute = ComputeMode::from_key(&opt.compute)
            .ok_or_else(|| BackendError::message(format!("compute: '{}' is not one of 'auto', 'cpu', 'gpu', 'hybrid'.", opt.compute)))?;
        let machine = self.machine();
        // The mode must be one the machine offers; the server checked the capabilities it saw, this is the same table.
        machine
            .runtime
            .check_compute(&Value::String(opt.compute.clone()), "options.compute")
            .map_err(BackendError::Message)?;
        if !is_bakeable(&body) && !opt.experiment {
            return Err(BackendError::message(format!(
                "cloud_regime: clouds of {}, not water: this generator's cloud amount is Earth's water clouds, so the map would not describe that body. Set experiment=true (the option) to generate it anyway as an experiment (never a bake), or use cloud_regime \"water\".",
                p.cloud_regime
            )));
        }
        let cyclones = cyclones_per_hemisphere(&body);
        if let Some(n) = cyclones.filter(|n| *n > MAX_CYCLONES_PER_HEMISPHERE) {
            let advice = match min_period_for_limit(&body) {
                Some(need) => format!("Use a rotation period of at least {} days.", fmt_g(need)),
                None => "Lengthen the rotation period.".to_string(),
            };
            return Err(BackendError::message(format!(
                "rotation_period_days: {} days gives {n} cyclones per hemisphere, past the {MAX_CYCLONES_PER_HEMISPHERE} this service generates. {advice}",
                fmt_g(p.rotation_period_days)
            )));
        }
        let mut warnings = body_warnings(&body, cyclones);
        if opt.isotropic {
            warnings.push("Isotropic baseline: no flow, no vortices, and the surface is ignored. The type, high-share and wind maps still come from the same fields.".into());
        }
        if opt.experiment && !is_bakeable(&body) {
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

        let reference = self.reference()?;
        let stem = format!("cloudgen_s{}", request.seed);
        let job = Job {
            body: body.clone(),
            seed: u64::from(request.seed),
            width,
            options: MapOptions {
                match_earth: opt.match_earth,
                isotropic: opt.isotropic,
                experiment: opt.experiment,
                surface: match opt.surface {
                    Surface::None => SurfaceChoice::None,
                    Surface::Synthetic => SurfaceChoice::Synthetic,
                },
            },
            compute,
            stem: stem.clone(),
            // No wall time in the sidecar: the same inputs write the same bytes.
            sidecar_seconds: false,
        };
        let progress = RefCell::new(progress);
        let report = |fraction: f64, message: &str| (progress.borrow_mut())(fraction as f32, message);
        let g = run_generation(&machine.runtime, reference, &job, &report, cancel).map_err(run_error)?;

        let rows = metric_rows(&g.measured.rows);
        let zonal = zonal_of(&g.measured.zonal);
        let height = g.height;
        let mut files: BTreeMap<String, PathBuf> = BTreeMap::new();
        if let Some(out) = &request.out_dir {
            if cancel() {
                return Err(BackendError::Cancelled);
            }
            report(0.95, "writing files");
            fs::create_dir_all(out).map_err(|e| io_err("out_dir", e))?;
            let mut put = |key: &str, name: &str, bytes: &[u8]| -> Result<(), BackendError> {
                let path = out.join(name);
                write(&path, bytes)?;
                files.insert(key.to_string(), path);
                Ok(())
            };
            put("type", &g.names.type_, &g.files.type_png)?;
            put("high_share", &g.names.high_share, &g.files.high_share_png)?;
            put("wind", &g.names.wind, &g.files.wind_png)?;
            put("master", &g.names.master, &g.files.master16_png)?;
            put("preview", &g.names.preview, &g.files.coverage_png)?;
            put("r8gz", &g.names.r8gz, &g.files.r8_gz)?;
            put("meta", &g.names.meta, &g.files.meta_json)?;
            let metrics = J::obj([
                ("physics", physics_json(p)),
                ("seed", J::uint(u64::from(request.seed))),
                ("width", J::uint(width as u64)),
                ("height", J::uint(height as u64)),
                ("options", options_json(request)),
                ("earth_reference", J::s(EARTH_REFERENCE_LABEL)),
                ("metrics", J::Arr(rows.iter().map(row_json).collect())),
                (
                    "zonal",
                    J::obj([
                        ("lat_deg", J::floats(&zonal.lat_deg)),
                        ("generated", J::Arr(zonal.generated.iter().map(|v| opt_json(*v)).collect())),
                        ("earth", J::Arr(zonal.earth.iter().map(|v| opt_json(*v)).collect())),
                        ("target", zonal.target.as_ref().map_or(J::Null, |t| J::Arr(t.iter().map(|v| opt_json(*v)).collect()))),
                    ]),
                ),
                ("warnings", J::Arr(warnings.iter().map(|w| J::s(w.as_str())).collect())),
            ]);
            put("metrics", &format!("{stem}_{width}x{height}_metrics.json"), format!("{}\n", dumps(&metrics)).as_bytes())?;
        }

        let extras = g.output.extras.as_ref().ok_or_else(|| BackendError::Internal("the engine returned no derived maps".to_string()))?;
        let mut extra = BTreeMap::new();
        extra.insert("type".to_string(), plane_of(width, height, &extras.kind));
        extra.insert("high_share".to_string(), plane_of(width, height, &extras.high_share));
        extra.insert("wind".to_string(), wind_speed_plane(&extras.wind, extras.wind_width, extras.wind_height));
        let seconds = round_to(t0.elapsed().as_secs_f64(), 2).max(0.01);
        let compute_block: Value = serde_json::from_str(&compact(&compute_json(compute.key(), &g.compute, Some(seconds)))).unwrap_or(Value::Null);
        let result = GenerationResult {
            id: token_hex(6),
            seconds,
            width: request.width,
            height: height as u32,
            physics: p.clone(),
            seed: request.seed,
            options: opt.record(),
            metrics: rows,
            zonal,
            files,
            coverage: plane_of(width, height, &g.coverage),
            extra_planes: extra,
            warnings,
            compute: Some(compute_block),
        };
        report(1.0, "done");
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
        let measured = measure_map(self.reference()?, &g.data, h, w, None);
        Ok(MetricsOut {
            width: w as u32,
            height: h as u32,
            metrics: metric_rows(&measured.rows),
            zonal: zonal_of(&measured.zonal),
            warnings: width_note(w).into_iter().collect(),
        })
    }

    fn bake(
        &self,
        request: &BakeRequest,
        progress: Progress<'_>,
        cancel: CancelCheck<'_>,
    ) -> Result<BakeResult, BackendError> {
        let t0 = Instant::now();
        let progress = RefCell::new(progress);
        let report = |fraction: f32, message: &str| (progress.borrow_mut())(fraction, message);
        if cancel() {
            return Err(BackendError::Cancelled);
        }
        report(0.0, "reading the bodies file");
        let text = fs::read_to_string(&request.bodies_file)
            .map_err(|_| BackendError::message(format!("bodies_file: the file could not be read. {BODIES_HINT}")))?;
        let doc: Value = serde_json::from_str(&text)
            .map_err(|e| BackendError::message(format!("bodies_file: the file could not be read (ValueError: {e}). {BODIES_HINT}")))?;
        let records = doc.get("bodies").and_then(Value::as_array).cloned().ok_or_else(|| {
            BackendError::message("bodies_file: the 'bodies' entry must be a list of body records (format: MCP resource cloud-map-generator://docs/bodies-format, or examples/bodies.json).")
        })?;
        check_bodies(&records, request.only != Some(BakeOnly::Library))?;
        let base = request.bodies_file.parent().map(Path::to_path_buf).unwrap_or_default();
        let bodies = parse_bodies(&text, base).map_err(|e| {
            BackendError::message(format!("bodies_file: baking failed on a record ({e}). Check the surface files and the numbers of each body."))
        })?;
        let opts = BakeOptions {
            version: request.version.clone(),
            url_prefix: request.url_prefix.clone(),
            list_width: Some(request.list_width as usize),
            library_width: request.library_width as usize,
            k: request.k as usize,
            only: request.only.map(|o| match o {
                BakeOnly::List => Only::List,
                BakeOnly::Library => Only::Library,
            }),
        };
        struct Steps<'a> {
            report: &'a dyn Fn(f32, &str),
            cancel: CancelCheck<'a>,
            skipped: Vec<Skipped>,
        }
        impl Observer for Steps<'_> {
            fn skipped(&mut self, key: &str, reason: &str) {
                self.skipped.push(Skipped { key: key.to_string(), reason: reason.to_string() });
            }
            fn starting(&mut self, label: &str, index: usize, total: usize) -> bool {
                if (self.cancel)() {
                    return false;
                }
                (self.report)(0.98 * index as f32 / total.max(1) as f32, &format!("baking {label} ({} of {total})", index + 1));
                true
            }
            fn entry(&mut self, _log: &EntryLog) {}
        }
        let mut steps = Steps { report: &report, cancel, skipped: Vec::new() };
        // The list width is the caller's (the server resolved the default).
        let baked = match service_bake::bake(&bodies, &opts, &request.out_dir, &mut steps) {
            Ok(b) => b,
            Err(ServiceBakeError::Cancelled) => return Err(BackendError::Cancelled),
            Err(ServiceBakeError::Fail(f)) => {
                return Err(BackendError::message(format!(
                    "bodies_file: baking failed on a record ({f}). Check the surface files and the numbers of each body."
                )))
            }
        };
        let mut files = baked.files.clone();
        let mut manifest_path = None;
        if request.only.is_none() {
            let path = request.out_dir.join("manifest.json");
            write_file(&path, format!("{}\n", dumps(&baked.manifest)).as_bytes()).map_err(|e| io_err("manifest", e))?;
            files.push(path.clone());
            manifest_path = Some(path);
        }
        let manifest: Value = serde_json::from_str(&compact(&baked.manifest)).unwrap_or(Value::Null);
        report(1.0, "done");
        Ok(BakeResult { manifest, manifest_path, files, skipped: steps.skipped, seconds: round_to(t0.elapsed().as_secs_f64(), 2) })
    }

    fn explain_model(&self) -> String {
        let machine = self.machine();
        let modes = machine.runtime.modes();
        let available = |m: ComputeMode| modes.iter().find(|r| r.mode == m).is_some_and(|r| r.available);
        let where_text = format!(
            "The `compute` option picks the device: `auto` (default), `cpu`, `gpu` or `hybrid`. This engine is the Rust one: the CPU path runs every stage on one thread; {} `hybrid` (row bands shared between the two) is not built yet and is refused with the reason, and an explicit mode the machine cannot run is refused too instead of being replaced by another. `capabilities()` lists what this engine offers, and every result says what ran it.",
            if available(ComputeMode::Gpu) {
                "the GPU path runs the per-texel stages in float32 on the machine's graphics card (the CPU map to float32 rounding, not bit for bit; `auto` picks it) while the random plans, the metrics and the file encoding stay on the CPU."
            } else {
                "no GPU path is available on this machine (the capabilities say why), so only `auto` and `cpu` are offered."
            }
        );
        include_str!("text/explain_model.md").replace("@@WHERE@@", &where_text)
    }

    /// The seconds of the CPU engine on every logical core, on a quiet machine (measured; see `cloud_map_generator_service::estimates`).
    /// An upper bound when a GPU runs the map.
    fn estimate_seconds(&self, width: u32) -> Option<f64> {
        let w = width as usize;
        estimate_seconds(&[w]).first().map(|(_, s)| *s)
    }
}

fn physics_json(p: &Physics) -> J {
    J::obj([
        ("rotation_period_days", J::Float(p.rotation_period_days)),
        ("radius_km", J::Float(p.radius_km)),
        ("gravity_mps2", J::Float(p.gravity_mps2)),
        ("scale_height_km", J::Float(p.scale_height_km)),
        ("cloud_regime", J::s(p.cloud_regime.as_str())),
    ])
}

fn options_json(request: &GenerateRequest) -> J {
    let o = &request.options;
    J::obj([
        ("match_earth", J::Bool(o.match_earth)),
        ("isotropic", J::Bool(o.isotropic)),
        ("experiment", J::Bool(o.experiment)),
        ("surface", J::s(o.surface.as_str())),
    ])
}

