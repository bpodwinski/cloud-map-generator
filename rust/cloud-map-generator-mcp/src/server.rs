//! cloud-map-generator as an MCP server: the tools, and the handler that serves them over any transport.
//!
//! Everything that makes or measures a map is the [`Backend`]'s. This module validates what the AI asked for,
//! keeps it inside the sandbox ([`crate::sandbox`]), runs the heavy work one job at a time on a worker thread
//! ([`crate::jobs`]), and shapes the answers: a short text for the model, a structured result with an output
//! schema, and an image block when there is something to look at.
//!
//! Nothing here touches the network, starts a subprocess or evaluates text. stdout carries the protocol and is
//! never written to by this crate.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, GetPromptRequestParams, GetPromptResponse,
    GetPromptResult, Implementation, ListPromptsResult, ListResourceTemplatesResult, ListResourcesResult, ListToolsResult,
    PaginatedRequestParams, Prompt, PromptArgument, PromptMessage, ProgressNotificationParam, ReadResourceRequestParams,
    ReadResourceResponse, Role, ServerCapabilities, ServerConfig, Tool,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, ServerHandler};
use serde_json::{json, Map, Value};

use crate::args::{BakeBody, Parser};
use crate::backend::{
    limit, Backend, BakeOnly, BakeRequest, BakeResult, DeriveOut, GenerateOptions, GenerateRequest, GenerationResult, Physics, Plane,
    Surface, COMPUTE_MODES, LIMITS, WIDTHS,
};
use crate::config::ServerConfig as Config;
use crate::errors::{AppError, AppResult};
use crate::jobs::{Job, JobControl, JobManager, Status};
use crate::prompts::{bake_for_engine, design_planet_clouds, INSTRUCTIONS};
use crate::pyfmt::{fixed, fmt_g, py_repr, round_to, snippet, token_hex};
use crate::render::{decode_coverage, globe_png, preview_png, sniff, Gray, Sniff};
use crate::results::{Planes, ResultStore, StoredResult};
use crate::sandbox::{is_within, resolve_path, scrub, valid_result_id, Sandbox};
use crate::texts::{coarse_zonal, compute_summary, derive_text, f, generation_text, headline, metric_rows, next_hint};
use crate::tools::{
    build_tools, BAKE_DEFAULT_WIDTH, MAX_BAKE_BODIES, MAX_BAKE_K, MAX_PNG_HEIGHT, MAX_PNG_WIDTH, PREVIEW_MAX_WIDTH, SYNC_MAX_WIDTH,
    VIEW_MAX_SIZE, VIEW_MIN_SIZE,
};

pub const URI_PREFIX: &str = "cloud-map-generator://";
const MAX_BAKE_ESTIMATE_SECONDS: f64 = 3600.0;
const MAX_BAKE_FILES_LISTED: usize = 400;
/// The files of a saved result a client can read as resources: resource name, key of the result's files, MIME type.
pub const RESULT_FILES: [(&str, &str, &str); 5] = [
    ("coverage.png", "preview", "image/png"), // the 8-bit north-up preview: what to look at
    ("type.png", "type", "image/png"),
    ("high_share.png", "high_share", "image/png"),
    ("wind.png", "wind", "image/png"),
    ("meta.json", "meta", "application/json"), // the sidecar
];

/// What a job hands back.
#[derive(Clone)]
pub enum JobOutput {
    /// The id of a stored generation result.
    Generation(String),
    /// A finished bake.
    Bake(Arc<BakeResult>),
}

type Jobs = JobManager<JobOutput>;
type JobRef = Arc<Job<JobOutput>>;

/// What one tool call hands back: the text, the structured result and perhaps a picture.
pub struct ToolOut {
    pub text: String,
    pub structured: Value,
    pub image: Option<Vec<u8>>,
}

impl ToolOut {
    fn new(text: String, structured: Value) -> Self {
        ToolOut { text, structured, image: None }
    }

    fn with_image(mut self, png: Vec<u8>) -> Self {
        self.image = Some(png);
        self
    }
}

fn user<T>(m: impl Into<String>) -> AppResult<T> {
    Err(AppError::User(m.into()))
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> AppResult<T> {
    tokio::task::spawn_blocking(f).await.map_err(|e| AppError::internal(if e.is_panic() { "Panic" } else { "JoinError" }, e.to_string()))
}

/// The message the AI sees for a failed job: the text of an anticipated error with paths scrubbed, nothing for a crash.
fn format_error(roots: &[PathBuf], e: &AppError) -> String {
    match e {
        AppError::User(m) => m.clone(),
        AppError::Service(m) => scrub(m, roots),
        AppError::Cancelled => "The job was cancelled.".to_string(),
        AppError::Internal { kind, .. } => format!(
            "The server hit an internal error ({kind}). Nothing was changed; the details are in the server log (stderr). Try again with other arguments, or report it."
        ),
    }
}

/// The sandbox, the job queue, the result cache and the backend of one server instance.
pub struct Runtime {
    pub config: Config,
    pub backend: Arc<dyn Backend>,
    pub sandbox: Arc<Sandbox>,
    pub results: Arc<ResultStore>,
    pub jobs: Jobs,
    pub caps: Value,
}

impl Runtime {
    pub fn new(config: Config, backend: Arc<dyn Backend>) -> Arc<Runtime> {
        let sandbox = Arc::new(Sandbox::new(&config.out_dir, &config.read_dirs));
        let roots: Vec<PathBuf> = sandbox.roots().to_vec();
        let caps = backend.capabilities();
        Arc::new(Runtime {
            config,
            backend,
            sandbox,
            results: Arc::new(ResultStore::default()),
            jobs: JobManager::new(move |e| format_error(&roots, e)),
            caps,
        })
    }

    /// Turn an error into the text of a tool error.
    pub fn error_text(&self, tool: &str, e: &AppError) -> String {
        match e {
            AppError::Internal { kind, detail } => {
                crate::logging::log(
                    crate::logging::Level::Error,
                    "cloud_map_generator_mcp",
                    format_args!("tool {tool} crashed: {kind}: {detail}"),
                );
                format!("Error executing tool {tool}")
            }
            AppError::Cancelled => "The generation was cancelled.".to_string(),
            other => format_error(self.sandbox.roots(), other),
        }
    }

    fn est(&self, width: u32) -> Option<f64> {
        self.backend.estimate_seconds(width).filter(|s| *s != 0.0)
    }

    fn est_text(&self, w: u32) -> String {
        match self.est(w) {
            Some(s) => format!("about {} s", fmt_g(s)),
            None => "seconds".to_string(),
        }
    }

    // ---- validation shared by the generation tools

    fn check_width(&self, width: i64, sync: bool) -> AppResult<u32> {
        if !WIDTHS.iter().any(|w| i64::from(*w) == width) {
            let list: Vec<String> = WIDTHS.iter().map(u32::to_string).collect();
            return user(format!("width must be one of {}; got {width}.", list.join(", ")));
        }
        let w = width as u32;
        let cap = self.config.max_width;
        let est = self.est(w);
        if w > cap {
            let took = est.map(|e| format!(" (a {w} map takes about {} s)", fixed(e, 0))).unwrap_or_default();
            return user(format!(
                "width {w} is above this server's --max-width {cap}{took}. This limit is set when the server starts and cannot be changed from a tool call: either use width={cap} (the largest this server accepts; start_generation(width={cap}) runs it in the background), or ask the user to restart cloud-map-generator-mcp with --max-width {w} (an argument of the server command in the MCP client configuration)."
            ));
        }
        if sync && w > SYNC_MAX_WIDTH {
            return match est {
                Some(e) => user(format!(
                    "generate_map runs inside one tool call, so it only takes widths up to {SYNC_MAX_WIDTH}; a {w} map takes about {} s. Call start_generation with the same arguments, then poll get_generation(job_id) until it is done.",
                    fixed(e, 0)
                )),
                None => user(format!("generate_map only takes widths up to {SYNC_MAX_WIDTH}. Use start_generation for {w}.")),
            };
        }
        Ok(w)
    }

    fn physics(&self, preset: Option<&str>, overrides: &crate::rules::Overrides) -> AppResult<Physics> {
        Ok(crate::rules::resolve_physics(&self.backend.presets(), preset, overrides)?)
    }

    fn derived(&self, physics: &Physics) -> AppResult<Value> {
        Ok(derived_json(&self.backend.derive(physics)?))
    }

    // ---- generation results

    /// The function a worker thread runs for one generation: generate, move the files to their final folder, cache
    /// the result. Public for tests and embedders that queue their own jobs.
    pub fn generation_job(
        self: &Arc<Self>,
        physics: Physics,
        seed: u32,
        width: u32,
        options: GenerateOptions,
        save: bool,
    ) -> impl FnOnce(&JobControl<JobOutput>) -> AppResult<JobOutput> + Send + 'static {
        let backend = self.backend.clone();
        let sandbox = self.sandbox.clone();
        let results = self.results.clone();
        move |control| {
            let mut pending = PendingGuard::new(sandbox.clone());
            if save {
                sandbox.ensure_out_dir()?;
                pending.set(sandbox.new_pending_dir()?);
            }
            let request = GenerateRequest { physics, seed, width, options, out_dir: pending.path().map(Path::to_path_buf) };
            let mut progress = |frac: f32, message: &str| control.progress(0.95 * f64::from(frac), message);
            let cancel = || control.cancelled();
            let result = backend.generate(&request, &mut progress, &cancel)?;
            let rid = if valid_result_id(&result.id) { result.id.clone() } else { token_hex(6) };
            let (folder, files) = match pending.path().map(Path::to_path_buf) {
                Some(p) => {
                    let (folder, files) = promote_pending(&sandbox, &p, &result, &rid)?;
                    pending.disarm();
                    (Some(folder), files)
                }
                None => (None, BTreeMap::new()),
            };
            let stored = store_result(&*backend, &results, &result, &rid, folder, files)?;
            control.progress(1.0, "done");
            Ok(JobOutput::Generation(stored.id.clone()))
        }
    }

    fn stored_for_job(&self, job: &JobRef) -> AppResult<Arc<StoredResult>> {
        let stored = match (job.status(), job.result()) {
            (Status::Done, Some(JobOutput::Generation(id))) => self.results.get(&id),
            _ => None,
        };
        match stored {
            Some(s) => Ok(s),
            None => user(format!(
                "The result of this job is no longer in the cache (only the last {} results are kept). Run the generation again.",
                self.results.max_results()
            )),
        }
    }

    /// Block the calling tool until the job is final, forwarding every progress value; cancel the job if the
    /// call is dropped or cancelled by the client.
    pub async fn wait_for(&self, job: &JobRef, ctx: Option<&RequestContext<RoleServer>>) {
        struct CancelOnDrop<'a> {
            rt: &'a Runtime,
            job: &'a JobRef,
        }
        impl Drop for CancelOnDrop<'_> {
            fn drop(&mut self) {
                if !self.job.is_final() {
                    let _ = self.rt.jobs.cancel(&self.job.id);
                }
            }
        }
        let _guard = CancelOnDrop { rt: self, job };
        let token = ctx.and_then(|c| c.meta.get_progress_token());
        let notify = |p: f64, m: String| {
            let token = token.clone();
            async move {
                if let (Some(ctx), Some(token)) = (ctx, token) {
                    let _ = ctx.peer.notify_progress(ProgressNotificationParam::new(token, p).with_total(1.0).with_message(m)).await;
                }
            }
        };
        notify(0.0, job.message()).await;
        let mut sent = 0usize;
        loop {
            for (p, m) in job.history_since(sent) {
                sent += 1;
                notify(p, m).await;
            }
            if job.is_final() {
                for (p, m) in job.history_since(sent) {
                    notify(p, m).await;
                }
                break;
            }
            let cancelled = async {
                match ctx {
                    Some(c) => c.ct.cancelled().await,
                    None => std::future::pending().await,
                }
            };
            tokio::select! {
                _ = job.changed() => {}
                _ = tokio::time::sleep(Duration::from_millis(200)) => {}
                _ = cancelled => break,
            }
        }
    }

    // ============================================================================================ the tools

    async fn list_presets(self: &Arc<Self>, args: &Map<String, Value>) -> AppResult<ToolOut> {
        Parser::new(args).finish("list_presets").map_err(AppError::User)?;
        let presets = self.backend.presets();
        let caps = self.backend.capabilities();
        let limits: Map<String, Value> = LIMITS.iter().map(|(n, lo, hi)| ((*n).to_string(), json!([lo, hi]))).collect();
        let estimates: Map<String, Value> =
            WIDTHS.iter().filter_map(|w| self.backend.estimate_seconds(*w).map(|s| (w.to_string(), json!(s)))).collect();
        let roots = self.sandbox.roots();
        let out = json!({
            "presets": presets.iter().map(|p| p.to_json()).collect::<Vec<_>>(),
            "limits": limits,
            "widths": WIDTHS,
            "max_width": self.config.max_width,
            "estimate_seconds": estimates,
            "output_folder": self.sandbox.out_dir().display().to_string(),
            "read_dirs": roots[1..].iter().map(|r| r.display().to_string()).collect::<Vec<_>>(),
            "compute": caps,
        });
        let mut lines: Vec<String> = presets
            .iter()
            .map(|p| {
                let ph = &p.physics;
                format!(
                    "{}: {} - rotation {} d, radius {} km, gravity {} m/s2, scale height {} km{}",
                    p.key,
                    p.label,
                    fmt_g(ph.rotation_period_days),
                    fmt_g(ph.radius_km),
                    fmt_g(ph.gravity_mps2),
                    fmt_g(ph.scale_height_km),
                    if p.experiment_required { " [experiment_required: not water clouds]" } else { "" }
                )
            })
            .collect();
        let widths: Vec<String> = WIDTHS.iter().map(u32::to_string).collect();
        lines.push(format!("Widths: {} px (this server accepts up to {}).", widths.join(", "), self.config.max_width));
        lines.push(compute_summary(&out["compute"]));
        lines.push(format!(
            "Output folder (results are saved here; files are read only from here{}): {}",
            if roots.len() > 1 { " and --read-dir directories" } else { "" },
            self.sandbox.out_dir().display()
        ));
        Ok(ToolOut::new(lines.join("\n"), out))
    }

    async fn derive_physics(self: &Arc<Self>, args: &Map<String, Value>) -> AppResult<ToolOut> {
        let mut p = Parser::new(args);
        let preset = p.str_opt("preset");
        let overrides = p.overrides("overrides");
        p.finish("derive_physics").map_err(AppError::User)?;
        let rt = self.clone();
        let (physics, derived) = blocking(move || -> AppResult<(Physics, Value)> {
            let physics = rt.physics(preset.as_deref(), &overrides.unwrap_or_default())?;
            let derived = rt.derived(&physics)?;
            Ok((physics, derived))
        })
        .await??;
        let preset_echo = args.get("preset").and_then(Value::as_str).map(str::to_string);
        let out = json!({"preset": preset_echo, "physics": physics.to_json(), "derived": derived});
        Ok(ToolOut::new(derive_text(&out), out))
    }

    fn generation_args(&self, tool: &str, args: &Map<String, Value>) -> AppResult<GenArgs> {
        let mut p = Parser::new(args);
        let preset = p.str_opt("preset");
        let overrides = p.overrides("overrides");
        let (seed_lo, seed_hi) = limit("seed");
        let seed = p.int("seed", 42, Some(seed_lo as i64), Some(seed_hi as i64));
        let width = p.int("width", 512, None, None);
        let match_earth = p.boolean("match_earth", false);
        let isotropic = p.boolean("isotropic", false);
        let experiment = p.boolean("experiment", false);
        let surface = p.literal("surface", "none", &["none", "synthetic"]);
        let save = p.boolean("save", true);
        let compute = p.literal("compute", "auto", &COMPUTE_MODES);
        p.finish(tool).map_err(AppError::User)?;
        Ok(GenArgs { preset, overrides: overrides.unwrap_or_default(), seed: seed as u32, width, match_earth, isotropic, experiment, surface, save, compute })
    }

    /// Checks every generation tool makes before it queues anything; returns the physics, the options and the width.
    async fn prepare_generation(self: &Arc<Self>, a: &GenArgs, sync: bool) -> AppResult<(Physics, GenerateOptions, u32)> {
        let width = self.check_width(a.width, sync)?;
        crate::rules::check_compute(&self.backend.capabilities(), &a.compute, "compute")?;
        let rt = self.clone();
        let (preset, overrides) = (a.preset.clone(), a.overrides.clone());
        let physics = blocking(move || rt.physics(preset.as_deref(), &overrides)).await??;
        let options = GenerateOptions {
            match_earth: a.match_earth,
            isotropic: a.isotropic,
            experiment: a.experiment,
            surface: if a.surface == "synthetic" { Surface::Synthetic } else { Surface::None },
            compute: a.compute.clone(),
        };
        Ok((physics, options, width))
    }

    async fn generate_map(self: &Arc<Self>, args: &Map<String, Value>, ctx: Option<&RequestContext<RoleServer>>) -> AppResult<ToolOut> {
        let a = self.generation_args("generate_map", args)?;
        let (physics, options, width) = self.prepare_generation(&a, true).await?;
        let job = self.jobs.submit(
            "generate",
            &format!("generate {width} seed {}", a.seed),
            self.generation_job(physics, a.seed, width, options, a.save),
            self.est(width),
        )?;
        self.wait_for(&job, ctx).await;
        match job.status() {
            Status::Failed => return user(job.error().unwrap_or_else(|| "generation failed".to_string())),
            Status::Cancelled => return Err(AppError::Cancelled),
            _ => {}
        }
        let stored = self.stored_for_job(&job)?;
        Ok(ToolOut::new(generation_text(&stored.summary), stored.summary.clone()).with_image(stored.preview.clone()))
    }

    async fn start_generation(self: &Arc<Self>, args: &Map<String, Value>) -> AppResult<ToolOut> {
        let a = self.generation_args("start_generation", args)?;
        let (physics, options, width) = self.prepare_generation(&a, false).await?;
        let job = self.jobs.submit(
            "generate",
            &format!("generate {width} seed {}", a.seed),
            self.generation_job(physics, a.seed, width, options, a.save),
            self.est(width),
        )?;
        // a request the engine rejects fails in milliseconds: report that now, not on a poll
        tokio::time::sleep(Duration::from_millis(250)).await;
        if job.status() == Status::Failed {
            return user(job.error().unwrap_or_else(|| "generation failed".to_string()));
        }
        let ahead = self.jobs.queue_ahead(&job);
        let next = format!("get_generation(job_id='{}') every few seconds until done.", job.id);
        let out = json!({
            "job_id": job.id, "status": job.status().as_str(), "width": width, "estimate_seconds": self.est(width),
            "queued_ahead": ahead, "next": next,
        });
        let mut text = format!("Job {} {}", job.id, job.status().as_str());
        if ahead > 0 {
            text.push_str(&format!(" ({ahead} ahead in the queue)"));
        }
        if let Some(e) = self.est(width) {
            text.push_str(&format!(", estimated {} s", fmt_g(e)));
        }
        text.push_str(&format!(". {next}"));
        Ok(ToolOut::new(text, out))
    }

    fn job_out(&self, job: &JobRef) -> Value {
        json!({
            "job_id": job.id, "kind": job.kind, "status": job.status().as_str(), "progress": round_to(job.progress(), 3),
            "message": job.message(), "elapsed_seconds": round_to(job.elapsed(), 2), "estimate_seconds": job.estimate_seconds,
            "error": job.error(), "result": Value::Null,
        })
    }

    async fn get_generation(self: &Arc<Self>, args: &Map<String, Value>) -> AppResult<ToolOut> {
        let mut p = Parser::new(args);
        let job_id = p.str_req("job_id");
        p.finish("get_generation").map_err(AppError::User)?;
        let job = self.jobs.get(&job_id)?;
        let mut out = self.job_out(&job);
        let id = &job.id;
        match job.status() {
            Status::Done => {
                let stored = match job.result() {
                    Some(JobOutput::Generation(rid)) => self.results.get(&rid),
                    _ => None,
                };
                let Some(stored) = stored else {
                    // the job is kept longer than some results are: say so instead of failing the poll
                    let text = format!(
                        "Job {id} is done, but its result has left the cache (only the last {} results are kept). Run the generation again, or measure_map / view_map its saved files.",
                        self.results.max_results()
                    );
                    return Ok(ToolOut::new(text, out));
                };
                out["result"] = stored.summary.clone();
                let text = format!("Job {id} done in {} s.\n{}", fixed(job.elapsed(), 1), generation_text(&stored.summary));
                Ok(ToolOut::new(text, out).with_image(stored.preview.clone()))
            }
            Status::Failed => Ok(ToolOut::new(format!("Job {id} failed: {}", job.error().unwrap_or_default()), out)),
            Status::Cancelled => Ok(ToolOut::new(format!("Job {id} was cancelled."), out)),
            Status::Queued => Ok(ToolOut::new(
                format!("Job {id} is queued ({} ahead). Call get_generation again in a few seconds.", self.jobs.queue_ahead(&job)),
                out,
            )),
            Status::Running => {
                let mut text = format!(
                    "Job {id} running, {}% ({}), {} s so far",
                    fixed(job.progress() * 100.0, 0),
                    job.message(),
                    fixed(job.elapsed(), 0)
                );
                if let Some(e) = job.estimate_seconds.filter(|e| *e != 0.0) {
                    text.push_str(&format!(" of about {} s", fmt_g(e)));
                }
                text.push_str(". Call get_generation again in a few seconds.");
                Ok(ToolOut::new(text, out))
            }
        }
    }

    async fn cancel_generation(self: &Arc<Self>, args: &Map<String, Value>) -> AppResult<ToolOut> {
        let mut p = Parser::new(args);
        let job_id = p.str_req("job_id");
        p.finish("cancel_generation").map_err(AppError::User)?;
        let job = self.jobs.cancel(&job_id)?;
        let status = job.status();
        let msg = if job.is_final() {
            format!("Job {} is already {}; nothing to cancel.", job.id, status.as_str())
        } else if job.cancel_requested() && status == Status::Running {
            format!("Cancellation requested for job {}; it stops at the next stage boundary.", job.id)
        } else {
            format!("Job {} {}.", job.id, status.as_str())
        };
        let out = json!({"job_id": job.id, "status": status.as_str(), "message": msg});
        Ok(ToolOut::new(msg, out))
    }

    async fn view_map(self: &Arc<Self>, args: &Map<String, Value>) -> AppResult<ToolOut> {
        let mut p = Parser::new(args);
        let id = p.str_opt("id");
        let path = p.str_opt("path");
        let view = p.literal("view", "map", &["map", "globe"]);
        let lon_deg = p.float("lon_deg", 0.0, None, None);
        let lat_deg = p.float("lat_deg", 20.0, Some(-90.0), Some(90.0));
        let product = p.literal("product", "coverage", &["coverage", "type", "high_share", "wind"]);
        let size = p.int("size", 512, Some(VIEW_MIN_SIZE), Some(VIEW_MAX_SIZE)) as usize;
        p.finish("view_map").map_err(AppError::User)?;
        if id.is_some() == path.is_some() {
            return user("Pass exactly one of `id` (a result from generate_map) or `path` (a coverage PNG inside the output folder or a --read-dir directory).");
        }
        let lon = (lon_deg + 180.0).rem_euclid(360.0) - 180.0;
        let (gray, source): (Gray, String) = if let Some(id) = &id {
            if !valid_result_id(id) {
                return user(format!("{} is not a result id: ids are lowercase hex tokens returned by generate_map.", py_repr(&snippet(id, 40))));
            }
            let Some(stored) = self.results.get(id) else {
                return user(format!(
                    "Unknown result id {}. Only the last {} results of this server session are kept: run generate_map again, or pass path=<a saved coverage PNG>.",
                    py_repr(id),
                    self.results.max_results()
                ));
            };
            let gray = match stored.planes() {
                None => {
                    let saved = stored.files.get("preview").filter(|p| p.is_file());
                    match (product.as_str(), saved) {
                        ("coverage", Some(saved)) => {
                            let (rt, saved) = (self.clone(), saved.display().to_string());
                            blocking(move || rt.load_coverage(&rt.sandbox.resolve_readable(&saved, "path")?)).await??
                        }
                        _ => {
                            return user("The arrays of this result were dropped to save memory. Run generate_map again, or pass path=<the saved coverage preview PNG> to look at its coverage.")
                        }
                    }
                }
                Some(planes) => product_plane(&planes, &product)?,
            };
            (gray, format!("result {id}"))
        } else {
            if product != "coverage" {
                return user("Only product='coverage' can be drawn from a file path; type, high_share and wind are available for results of this session (pass id).");
            }
            let resolved = self.sandbox.resolve_readable(path.as_deref().unwrap_or(""), "path")?;
            let name = resolved.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let rt = self.clone();
            let gray = blocking(move || rt.load_coverage(&resolved)).await??;
            (gray, name)
        };
        let (png, desc, dims) = if view == "globe" {
            let png = blocking(move || globe_png(&gray, lon, lat_deg, size)).await?;
            let desc = if lat_deg.abs() >= 90.0 {
                let pole = if lat_deg > 0.0 { "north" } else { "south" };
                let place = if lat_deg > 0.0 {
                    "top of the image is longitude 180, 90E on the right: east runs counterclockwise"
                } else {
                    "top of the image is longitude 0, 90E on the right: east runs clockwise"
                };
                format!("Orthographic globe of the {product} product seen from directly above the {pole} pole (lon_deg ignored); {place}; clouds white over a dark sphere.")
            } else {
                format!(
                    "Orthographic globe of the {product} product centred on lon {}, lat {} (north up, east to the right); clouds white over a dark sphere.",
                    fixed(lon, 0),
                    fixed(lat_deg, 0)
                )
            };
            (png, desc, (size, size))
        } else {
            let wide = size.min(gray.width);
            let png = blocking(move || preview_png(&gray, size)).await?;
            (png, format!("Equirectangular {product} map (north up, longitude -180 to 180), {wide} px wide."), (wide, wide / 2))
        };
        let globe = view == "globe";
        let out = json!({
            "source": source, "view": view, "product": product, "width_px": dims.0, "height_px": dims.1,
            "lon_deg": if globe { json!(lon) } else { Value::Null },
            "lat_deg": if globe { json!(lat_deg) } else { Value::Null },
            "description": desc,
        });
        Ok(ToolOut::new(format!("{source}: {desc}"), out).with_image(png))
    }

    /// A coverage PNG inside the sandbox as fractions; the file is read under the size cap and checked before it is decoded.
    fn load_coverage(&self, resolved: &Path) -> AppResult<Gray> {
        let bytes = self.sandbox.read_bytes(resolved)?;
        let name = resolved.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        png_size(&name, &bytes)?;
        decode_coverage(&bytes).map_err(|_| AppError::User(format!("{name} is not a readable PNG image.")))
    }

    async fn measure_map(self: &Arc<Self>, args: &Map<String, Value>) -> AppResult<ToolOut> {
        let mut p = Parser::new(args);
        let path = p.str_req("path");
        p.finish("measure_map").map_err(AppError::User)?;
        let resolved = self.sandbox.resolve_readable(&path, "path")?;
        let name = resolved.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let head_bytes = read_head(&resolved, 1 << 20);
        let (w, h) = png_size(&name, &head_bytes)?;
        let rt = self.clone();
        let target = resolved.clone();
        let res = blocking(move || rt.backend.measure_png(&target)).await??;
        let rows = metric_rows(&res.metrics, Some(w));
        let zonal = coarse_zonal(&res.zonal, 15.0);
        let head = headline(&rows, w);
        let mut summary = format!("{} ({w}x{h}) measured against Earth.", snippet(&name, 80));
        if !head.is_empty() {
            summary.push_str(&format!(" {head}"));
        }
        let out = json!({
            "path": resolved.display().to_string(), "width": w, "height": h, "metrics": rows, "zonal": zonal,
            "warnings": res.warnings, "summary": summary,
        });
        let mut lines = vec![summary.clone()];
        lines.extend(res.warnings.iter().take(2).map(|x| format!("Note: {x}")));
        for r in out["metrics"].as_array().into_iter().flatten() {
            let g = |k: &str| r.get(k).and_then(Value::as_f64);
            let mut line = format!("{}: {} (Earth {})", r["label"].as_str().unwrap_or(""), f(g("generated"), 3, ""), f(g("earth"), 3, ""));
            if let Some(gap) = g("relative_gap") {
                line.push_str(&format!(", gap {}", crate::pyfmt::signed(gap, 2)));
            }
            if let Some(hint) = r["hint"].as_str().filter(|h| !h.is_empty()) {
                line.push_str(&format!(" - {hint}"));
            }
            lines.push(line);
        }
        Ok(ToolOut::new(lines.join("\n"), out))
    }

    /// Parse a bodies file and check what the bake would read BEFORE it runs: count, shapes, data files.
    ///
    /// Every land or elevation grid a record names must sit inside a readable root (the engine would read it
    /// wherever it points); the grids of the bodies that will really be baked must also have the right size.
    fn validate_bodies_file(&self, bodies_path: &Path, only: Option<BakeOnly>) -> AppResult<Vec<Value>> {
        let bytes = self.sandbox.read_bytes(bodies_path)?;
        const BAD_JSON: &str = "The bodies file is not valid UTF-8 JSON. Expected {\"bodies\": [{\"key\": ..., \"rotationPeriodDays\": ..., \"radiusKm\": ..., \"gravityMps2\": ..., \"scaleHeightKm\": ..., \"cloudRegime\": \"water\"}]}.";
        let doc: Value = match std::str::from_utf8(&bytes).ok().and_then(|t| serde_json::from_str(t).ok()) {
            Some(d) => d,
            None => return user(BAD_JSON),
        };
        let bodies = doc.get("bodies").and_then(Value::as_array).filter(|b| !b.is_empty());
        let Some(bodies) = bodies else {
            return user("The bodies file must be a JSON object with a non-empty \"bodies\" list, each body a record with key, rotationPeriodDays, radiusKm, gravityMps2, scaleHeightKm and cloudRegime.");
        };
        if bodies.len() > MAX_BAKE_BODIES {
            return user(format!(
                "The bodies file lists {} bodies; this server bakes at most {MAX_BAKE_BODIES} per call. Split the file.",
                bodies.len()
            ));
        }
        for (i, b) in bodies.iter().enumerate() {
            let key = b.get("key").and_then(Value::as_str).filter(|k| !k.is_empty());
            let Some(key) = key else {
                return user(format!("bodies[{i}] must be an object with a non-empty string \"key\"."));
            };
            if !body_key_ok(key) || key.contains("..") {
                return user(format!(
                    "bodies[{i}].key {} is not allowed: a key names the output files, so use 1 to 64 letters, digits, spaces and . _ : + - (no slashes, no \"..\").",
                    py_repr(&snippet(key, 40))
                ));
            }
            let Some(sf) = b.get("surface").filter(|s| !is_falsy(s)) else { continue };
            let Some(sf) = sf.as_object() else {
                return user(format!("bodies[{i}].surface must be an object with width, height, land and elevation."));
            };
            let mut sizes: BTreeMap<&str, u64> = BTreeMap::new();
            for field in ["land", "elevation"] {
                let Some(name) = sf.get(field).and_then(Value::as_str) else {
                    return user(format!("bodies[{i}].surface needs a \"{field}\" file name (text)."));
                };
                let (_, size) = self.sandbox.resolve_sibling(bodies_path, name, &format!("bodies[{i}].surface.{field}"))?;
                sizes.insert(field, size);
            }
            let measured = b.get("measured").map(|m| !is_falsy(m)).unwrap_or(false);
            let regime = b.get("cloudRegime").and_then(Value::as_str).unwrap_or("unknown");
            if only != Some(BakeOnly::Library) && !measured && regime == "water" {
                let int_of = |k: &str| sf.get(k).and_then(py_int);
                let (Some(sw), Some(sh)) = (int_of("width"), int_of("height")) else {
                    return user(format!("bodies[{i}].surface needs integer \"width\" and \"height\"."));
                };
                if !(0 < sw && sw <= 16384 && 0 < sh && sh <= 8192) {
                    return user(format!("bodies[{i}].surface size {sw}x{sh} is out of range."));
                }
                for (field, per, kind) in [("land", 1i64, "u8"), ("elevation", 2, "little-endian int16")] {
                    let need = sw * sh * per;
                    if sizes[field] as i64 != need {
                        return user(format!(
                            "bodies[{i}].surface.{field} has {} bytes; a {sw}x{sh} grid of {kind} needs {need}.",
                            sizes[field]
                        ));
                    }
                }
            }
        }
        Ok(bodies.clone())
    }

    async fn bake_bodies(self: &Arc<Self>, args: &Map<String, Value>, ctx: Option<&RequestContext<RoleServer>>) -> AppResult<ToolOut> {
        let mut p = Parser::new(args);
        let bodies = p.bodies("bodies");
        let bodies_file = p.str_opt("bodies_file");
        let version = p.str_default("version", "v1");
        let url_prefix = p.str_opt("url_prefix");
        let list_width = p.int_opt("list_width");
        let library_width = p.int("library_width", i64::from(BAKE_DEFAULT_WIDTH), None, None);
        let k = p.int("k", 1, Some(1), Some(MAX_BAKE_K));
        let only = p.literal_opt("only", &["list", "library"]).map(|o| if o == "list" { BakeOnly::List } else { BakeOnly::Library });
        p.finish("bake_bodies").map_err(AppError::User)?;

        let folder = self.sandbox.bake_dir(&version)?;
        if let Some(u) = &url_prefix {
            if !url_prefix_ok(u) {
                return user("url_prefix may only contain letters, digits and . _ ~ : / @ % + = , - (at most 200 characters).");
            }
        }
        let max_w = self.config.max_width;
        let lw = list_width.unwrap_or_else(|| i64::from(BAKE_DEFAULT_WIDTH.min(max_w)));
        let mut widths = [0u32; 2];
        for (n, (name, w)) in [("list_width", lw), ("library_width", library_width)].into_iter().enumerate() {
            if !WIDTHS.iter().any(|x| i64::from(*x) == w) {
                let list: Vec<String> = WIDTHS.iter().map(u32::to_string).collect();
                return user(format!("{name} must be one of {}; got {w}.", list.join(", ")));
            }
            if w > i64::from(max_w) {
                return user(format!(
                    "{name} {w} is above this server's --max-width {max_w}, which cannot be changed from a tool call: use {name}={max_w}, or ask the user to restart cloud-map-generator-mcp with --max-width {w} (a {w} map takes {}).",
                    self.est_text(w as u32)
                ));
            }
            widths[n] = w as u32;
        }
        let [list_w, lib_w] = widths;
        if bodies.is_some() == bodies_file.is_some() {
            return user("Pass exactly one of `bodies` (the planets inline: a list of {key, rotation_period_days, radius_km, gravity_mps2, scale_height_km, cloud_regime}) or `bodies_file` (a bodies JSON file inside the output folder).");
        }
        let mut bodies_file = bodies_file;
        if let Some(bodies) = &bodies {
            if bodies.is_empty() {
                return user("`bodies` is empty: pass at least one planet.");
            }
            if bodies.len() > MAX_BAKE_BODIES {
                return user(format!("`bodies` lists {} bodies; this server bakes at most {MAX_BAKE_BODIES} per call. Split the list.", bodies.len()));
            }
            let records: Vec<Value> = bodies.iter().map(body_record).collect();
            self.sandbox.ensure_out_dir()?;
            let name = format!("bodies-{version}.json");
            let text = serde_json::to_string_pretty(&json!({"bodies": records})).unwrap_or_default();
            std::fs::write(self.sandbox.out_dir().join(&name), text)
                .map_err(|e| AppError::internal("IoError", e.to_string()))?;
            bodies_file = Some(name);
        }
        let resolved = self.sandbox.resolve_readable(bodies_file.as_deref().unwrap_or(""), "bodies_file")?;
        let rt = self.clone();
        let target = resolved.clone();
        let records = blocking(move || rt.validate_bodies_file(&target, only)).await??;
        let n_list = if only == Some(BakeOnly::Library) {
            0
        } else {
            records
                .iter()
                .filter(|b| {
                    !b.get("measured").map(|m| !is_falsy(m)).unwrap_or(false)
                        && b.get("cloudRegime").and_then(Value::as_str).unwrap_or("unknown") == "water"
                })
                .count()
        };
        let n_lib = if only == Some(BakeOnly::List) { 0 } else { 4 * k as usize };
        let total = n_list as f64 * self.est(list_w).unwrap_or(10.0) + n_lib as f64 * self.est(lib_w).unwrap_or(10.0);
        if total > MAX_BAKE_ESTIMATE_SECONDS {
            return user(format!(
                "This bake would take about {} minutes ({n_list} bodies at {list_w} px, {n_lib} library maps at {lib_w} px). Lower the widths or k, bake with only='list' or only='library', or split the bodies file; the limit is {} minutes.",
                fixed(total / 60.0, 0),
                (MAX_BAKE_ESTIMATE_SECONDS / 60.0) as i64
            ));
        }
        let request = BakeRequest {
            bodies_file: resolved,
            out_dir: folder.clone(),
            version: version.clone(),
            url_prefix: url_prefix.clone(),
            list_width: list_w,
            library_width: lib_w,
            k: k as u32,
            only,
        };
        let (backend, sandbox) = (self.backend.clone(), self.sandbox.clone());
        let job = self.jobs.submit(
            "bake",
            &format!("bake {version}"),
            move |control| {
                sandbox.ensure_out_dir()?;
                std::fs::create_dir_all(&request.out_dir).map_err(|e| AppError::internal("IoError", e.to_string()))?;
                let mut progress = |frac: f32, message: &str| control.progress(f64::from(frac), message);
                let cancel = || control.cancelled();
                let res = backend.bake(&request, &mut progress, &cancel)?;
                Ok(JobOutput::Bake(Arc::new(res)))
            },
            Some(total),
        )?;
        self.wait_for(&job, ctx).await;
        match job.status() {
            Status::Failed => return user(job.error().unwrap_or_else(|| "the bake failed".to_string())),
            Status::Cancelled => return user("The bake was cancelled."),
            _ => {}
        }
        let res = match job.result() {
            Some(JobOutput::Bake(r)) => Some(r),
            _ => None,
        };
        let manifest = res.as_ref().map_or(Value::Null, |r| r.manifest.clone());
        let mut files: Vec<String> = Vec::new();
        for path in res.iter().flat_map(|r| r.files.iter()) {
            let resolved = resolve_path(path).unwrap_or_else(|_| path.clone());
            if self.sandbox.contains(&resolved) && resolved.is_file() {
                if let Ok(rel) = resolved.strip_prefix(self.sandbox.out_dir()) {
                    files.push(rel.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect::<Vec<_>>().join("/"));
                }
            }
        }
        let skipped: Vec<String> = res.iter().flat_map(|r| r.skipped.iter()).map(|s| format!("{}: {}", s.key, s.reason)).collect();
        let next = "measure_map or view_map on a baked coverage PNG; repeat with larger widths when the draft is right.";
        let listed: Vec<&String> = files.iter().take(MAX_BAKE_FILES_LISTED).collect();
        let out = json!({
            "version": version, "folder": folder.display().to_string(), "seconds": round_to(job.elapsed(), 1), "manifest": manifest,
            "files": listed, "files_total": files.len(), "truncated": files.len() > MAX_BAKE_FILES_LISTED, "skipped": skipped, "next": next,
        });
        let mut text = format!("Baked {} files in {} s into {}", files.len(), fixed(job.elapsed(), 0), folder.display());
        if only.is_some() {
            text.push_str(" (partial bake: no manifest written).");
        } else {
            text.push_str(&format!(", manifest bake/{version}/manifest.json."));
        }
        if !skipped.is_empty() {
            text.push_str(&format!(" Skipped: {}.", skipped.join("; ")));
        }
        text.push_str(&format!(" Next: {next}"));
        Ok(ToolOut::new(text, out))
    }

    async fn explain_model(self: &Arc<Self>, args: &Map<String, Value>) -> AppResult<ToolOut> {
        Parser::new(args).finish("explain_model").map_err(AppError::User)?;
        let md = self.backend.explain_model();
        Ok(ToolOut::new(md.clone(), json!({"markdown": md})))
    }

    /// Run one tool by name. Every error becomes a tool error whose text the AI can act on.
    pub async fn call(self: &Arc<Self>, name: &str, args: Map<String, Value>, ctx: Option<&RequestContext<RoleServer>>) -> CallToolResult {
        let r = match name {
            "list_presets" => self.list_presets(&args).await,
            "derive_physics" => self.derive_physics(&args).await,
            "generate_map" => self.generate_map(&args, ctx).await,
            "start_generation" => self.start_generation(&args).await,
            "get_generation" => self.get_generation(&args).await,
            "cancel_generation" => self.cancel_generation(&args).await,
            "view_map" => self.view_map(&args).await,
            "measure_map" => self.measure_map(&args).await,
            "bake_bodies" => self.bake_bodies(&args, ctx).await,
            "explain_model" => self.explain_model(&args).await,
            _ => return CallToolResult::error(vec![ContentBlock::text(format!("Unknown tool: {name}"))]),
        };
        match r {
            Ok(out) => {
                let mut content = vec![ContentBlock::text(out.text)];
                if let Some(png) = out.image {
                    content.push(ContentBlock::image(base64::engine::general_purpose::STANDARD.encode(png), "image/png"));
                }
                let mut result = CallToolResult::success(content);
                result.structured_content = Some(out.structured);
                result
            }
            Err(e) => CallToolResult::error(vec![ContentBlock::text(self.error_text(name, &e))]),
        }
    }
}

// ================================================================================================ helpers

/// The arguments of generate_map and start_generation.
struct GenArgs {
    preset: Option<String>,
    overrides: crate::rules::Overrides,
    seed: u32,
    width: i64,
    match_earth: bool,
    isotropic: bool,
    experiment: bool,
    surface: String,
    save: bool,
    compute: String,
}

/// Removes a scratch folder when a run fails, is cancelled or panics.
struct PendingGuard {
    sandbox: Arc<Sandbox>,
    path: Option<PathBuf>,
}

impl PendingGuard {
    fn new(sandbox: Arc<Sandbox>) -> Self {
        PendingGuard { sandbox, path: None }
    }

    fn set(&mut self, p: PathBuf) {
        self.path = Some(p);
    }

    fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    fn disarm(&mut self) {
        self.path = None;
    }
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        if let Some(p) = self.path.take() {
            self.sandbox.discard_pending(&p);
        }
    }
}

fn promote_pending(
    sandbox: &Sandbox,
    pending: &Path,
    result: &GenerationResult,
    rid: &str,
) -> AppResult<(PathBuf, BTreeMap<String, PathBuf>)> {
    let base = resolve_path(pending).unwrap_or_else(|_| pending.to_path_buf());
    let mut rel: BTreeMap<String, PathBuf> = BTreeMap::new();
    for (name, p) in &result.files {
        let resolved = resolve_path(p).unwrap_or_else(|_| p.clone());
        if is_within(&resolved, &base) {
            if let Ok(r) = resolved.strip_prefix(&base) {
                rel.insert(name.clone(), r.to_path_buf());
            }
        }
    }
    let mut src = pending.to_path_buf();
    let tops: std::collections::BTreeSet<std::ffi::OsString> =
        rel.values().filter(|r| r.components().count() > 1).filter_map(|r| r.components().next().map(|c| c.as_os_str().to_os_string())).collect();
    if !rel.is_empty() && tops.len() == 1 && rel.values().all(|r| r.components().count() > 1) {
        // the engine wrote a subfolder of its own: that is the result folder
        let top = tops.into_iter().next().unwrap_or_default();
        src = pending.join(&top);
        rel = rel.into_iter().map(|(n, r)| (n, r.strip_prefix(&top).map(Path::to_path_buf).unwrap_or(r))).collect();
    }
    let folder = sandbox.promote(&src, rid)?;
    if src != pending {
        sandbox.discard_pending(pending);
    }
    let files = rel.into_iter().map(|(n, r)| (n, folder.join(r))).collect();
    Ok((folder, files))
}

/// The derived physics in its output form.
pub fn derived_json(d: &DeriveOut) -> Value {
    let texel: Map<String, Value> = d.texel_km.iter().map(|(w, km)| (w.to_string(), json!(km))).collect();
    json!({
        "hadley_edge_deg": d.hadley_edge_deg,
        "storm_track_deg": d.storm_track_deg,
        "has_eddies": d.has_eddies,
        "deformation_radius_km": d.deformation_radius_km,
        "cyclones_per_hemisphere": d.cyclones_per_hemisphere,
        "cyclone_spin_north": d.cyclone_spin_north,
        "cyclone_spin_south": d.cyclone_spin_south,
        "texel_km": texel,
        "bakeable": d.bakeable,
        "warnings": d.warnings,
    })
}

/// Cache a finished generation: its picture, its pixels and the summary every tool returns for it.
fn store_result(
    backend: &dyn Backend,
    results: &ResultStore,
    result: &GenerationResult,
    rid: &str,
    folder: Option<PathBuf>,
    files: BTreeMap<String, PathBuf>,
) -> AppResult<Arc<StoredResult>> {
    let planes = Planes { coverage: result.coverage.clone(), extra: result.extra_planes.clone() };
    let preview = preview_png(&Gray::from_plane(&result.coverage), PREVIEW_MAX_WIDTH);
    let mut resources = Map::new();
    if folder.is_some() {
        for (name, key, _) in RESULT_FILES {
            if files.get(key).is_some_and(|p| p.is_file()) {
                resources.insert(name.to_string(), Value::String(format!("{URI_PREFIX}result/{rid}/{name}")));
            }
        }
    }
    let derived = derived_json(&backend.derive(&result.physics)?);
    let summary = json!({
        "id": rid,
        "seconds": if result.seconds.is_finite() { round_to(result.seconds, 2) } else { 0.0 },
        "width": result.width,
        "height": result.height,
        "seed": result.seed,
        "physics": result.physics.to_json(),
        "options": result.options,
        "derived": derived,
        "metrics": metric_rows(&result.metrics, Some(result.width)),
        "zonal": coarse_zonal(&result.zonal, 15.0),
        "warnings": result.warnings,
        "compute": result.compute.clone().unwrap_or(Value::Null),
        "saved": folder.is_some(),
        "folder": folder.as_ref().map(|p| p.display().to_string()),
        "files": files.iter().map(|(n, p)| (n.clone(), Value::String(p.display().to_string()))).collect::<Map<String, Value>>(),
        "resources": resources,
        "next": next_hint(rid),
    });
    let stored = Arc::new(StoredResult::new(rid.to_string(), summary, preview, Some(planes), folder, files));
    results.add(stored.clone());
    Ok(stored)
}

/// One product of a result as a grid in [0, 1], ready to be drawn like a coverage map.
fn product_plane(planes: &Planes, product: &str) -> AppResult<Gray> {
    let Some(plane) = planes.get(product) else {
        return user(format!("This result has no {} product. Available: {}.", py_repr(product), planes.names().join(", ")));
    };
    Ok(plane_gray(plane, product))
}

fn plane_gray(plane: &Plane, product: &str) -> Gray {
    if product == "wind" {
        Gray::from_plane_normalized(plane)
    } else {
        Gray::from_plane(plane)
    }
}

fn read_head(path: &Path, n: u64) -> Vec<u8> {
    use std::io::Read;
    let mut buf = Vec::new();
    if let Ok(file) = std::fs::File::open(path) {
        let _ = file.take(n).read_to_end(&mut buf);
    }
    buf
}

/// Width and height of a PNG from its header, refusing anything the server will not decode.
fn png_size(name: &str, bytes: &[u8]) -> AppResult<(u32, u32)> {
    let (w, h) = match sniff(bytes) {
        Sniff::Other(format) => {
            return user(format!("{name} is a {format} file; only PNG maps are read. Save the coverage map as an 8-bit or 16-bit gray PNG (2:1, equirectangular)."))
        }
        Sniff::Unreadable => return user(format!("{name} is not a readable PNG image.")),
        Sniff::Png { width, height } => (width, height),
    };
    if w > MAX_PNG_WIDTH || h > MAX_PNG_HEIGHT {
        return user(format!("{name} is {w}x{h} pixels; this server reads maps up to {MAX_PNG_WIDTH}x{MAX_PNG_HEIGHT}. Downsample it first."));
    }
    if w < 256 || w != 2 * h {
        return user(format!(
            "{name} is {w}x{h} pixels. A coverage map is equirectangular: 2:1 (width = 2 x height) and at least 256 pixels wide."
        ));
    }
    Ok((w, h))
}

/// `[\w][\w .:+-]{0,63}`: the spelling of a body key (it names the output files).
fn body_key_ok(key: &str) -> bool {
    let mut chars = key.chars();
    let Some(first) = chars.next() else { return false };
    let word = |c: char| c.is_alphanumeric() || c == '_';
    word(first) && key.chars().count() <= 64 && chars.all(|c| word(c) || matches!(c, ' ' | '.' | ':' | '+' | '-'))
}

/// `^[A-Za-z0-9._~:/@%+=,-]{0,200}$`.
fn url_prefix_ok(s: &str) -> bool {
    s.chars().count() <= 200 && s.chars().all(|c| c.is_ascii_alphanumeric() || "._~:/@%+=,-".contains(c))
}

/// Python truthiness of a JSON value.
fn is_falsy(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Number(n) => n.as_f64() == Some(0.0),
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
    }
}

/// `int(x)` of a JSON number or numeric string.
fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().filter(|f| f.is_finite()).map(|f| f.trunc() as i64)),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

/// An inline body as the camelCase record a bodies file holds.
fn body_record(b: &BakeBody) -> Value {
    json!({
        "key": b.key, "rotationPeriodDays": b.rotation_period_days, "radiusKm": b.radius_km, "gravityMps2": b.gravity_mps2,
        "scaleHeightKm": b.scale_height_km, "cloudRegime": b.cloud_regime, "measured": b.measured,
    })
}

// ================================================================================================ the handler

/// The MCP server: one value per connection to a client, all sharing one [`Runtime`].
#[derive(Clone)]
pub struct PlanetCloudgenServer {
    pub rt: Arc<Runtime>,
    tools: Arc<Vec<Tool>>,
}

impl PlanetCloudgenServer {
    /// Build the server. `backend` is the engine; tests pass a [`crate::mock::MockBackend`].
    pub fn new(config: Config, backend: Arc<dyn Backend>) -> Self {
        let rt = Runtime::new(config, backend);
        let est = |w: u32| rt.backend.estimate_seconds(w);
        let tools = Arc::new(build_tools(&rt.caps, rt.config.max_width, &est));
        PlanetCloudgenServer { rt, tools }
    }

    /// The tools as listed to a client.
    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// Stop accepting jobs and cancel the ones that wait or run (the server is shutting down).
    pub fn close(&self) {
        self.rt.jobs.close();
    }
}

fn prompt_list() -> Vec<Prompt> {
    vec![
        Prompt::new(
            "design_planet_clouds",
            Some("The loop for designing a planet's cloud cover: derive the physics, generate, look from several angles, measure against Earth, adjust, generate large."),
            Some(vec![PromptArgument::new("planet_description")
                .with_description("The planet in words: rotation, size, atmosphere, anything known.")
                .with_required(true)]),
        )
        .with_title("Design the clouds of a planet"),
        Prompt::new(
            "bake_for_engine",
            Some("The flow for baking the cloud maps of a bodies file: check the physics, try one body, bake small, then large, and report the manifest."),
            Some(vec![PromptArgument::new("bodies_file").with_description("Path of the bodies JSON file.").with_required(true)]),
        )
        .with_title("Bake a set of bodies for a renderer"),
    ]
}

impl ServerHandler for PlanetCloudgenServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().enable_resources().enable_prompts().build())
            .with_server_info(
                Implementation::new("cloud-map-generator", crate::VERSION)
                    .with_title("cloud-map-generator")
                    .with_description("Physics-driven global cloud-coverage maps for any planet."),
            )
            .with_instructions(INSTRUCTIONS)
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(self.tools.to_vec()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let args = request.arguments.unwrap_or_default();
        let result = self.rt.call(&request.name, args, Some(&context)).await;
        Ok(result.into())
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult::with_all_items(crate::resources::list_resources()))
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult::with_all_items(crate::resources::list_templates()))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, McpError> {
        let rt = self.rt.clone();
        let uri = request.uri.clone();
        let result = tokio::task::spawn_blocking(move || crate::resources::read(&rt, &uri))
            .await
            .map_err(|e| McpError::internal_error(format!("resource read failed: {e}"), None))??;
        Ok(result.into())
    }

    async fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        Ok(ListPromptsResult::with_all_items(prompt_list()))
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResponse, McpError> {
        let args = request.arguments.unwrap_or_default();
        let (arg_name, description, render): (&str, &str, fn(&str) -> String) = match request.name.as_str() {
            "design_planet_clouds" => (
                "planet_description",
                "The loop for designing a planet's cloud cover: derive the physics, generate, look from several angles, measure against Earth, adjust, generate large.",
                design_planet_clouds,
            ),
            "bake_for_engine" => (
                "bodies_file",
                "The flow for baking the cloud maps of a bodies file: check the physics, try one body, bake small, then large, and report the manifest.",
                bake_for_engine,
            ),
            other => return Err(McpError::invalid_params(format!("Unknown prompt: {other}"), None)),
        };
        let value = match args.get(arg_name) {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Null) | None => {
                return Err(McpError::invalid_params(format!("Missing required arguments: {{'{arg_name}'}}"), None));
            }
            Some(other) => other.to_string(),
        };
        let result = GetPromptResult::new(vec![PromptMessage::new_text(Role::User, render(&value))]).with_description(description);
        Ok(result.into())
    }
}
