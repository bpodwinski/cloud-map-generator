//! The ten tools as a client sees them: name, title, description, input schema, output schema, annotations.
//!
//! The descriptions are the Python server's, copied word for word (a blind usability test refined them); the few
//! that mention the server's own limits are built at start from the configuration, as in Python. The schemas are
//! the ones pydantic generated for the Python server, kept in `schemas.json`, with the two descriptions that
//! depend on the engine (the widths and the compute modes) written at start.

use std::sync::Arc;

use rmcp::model::{JsonObject, Tool, ToolAnnotations};
use serde_json::Value;

use crate::backend::WIDTHS;

/// A generate_map runs inside one tool call; wider maps use start_generation.
pub const SYNC_MAX_WIDTH: u32 = 1024;
/// The image block of a generation result is at most this wide.
pub const PREVIEW_MAX_WIDTH: usize = 768;
pub const VIEW_MIN_SIZE: i64 = 64;
pub const VIEW_MAX_SIZE: i64 = 1024;
pub const MAX_PNG_WIDTH: u32 = 8192;
pub const MAX_PNG_HEIGHT: u32 = 4096;
pub const MAX_BAKE_BODIES: usize = 32;
pub const MAX_BAKE_K: i64 = 8;
/// Small on purpose: about a minute for a few bodies.
pub const BAKE_DEFAULT_WIDTH: u32 = 1024;

pub const TOOL_NAMES: [&str; 10] = [
    "list_presets",
    "derive_physics",
    "generate_map",
    "start_generation",
    "get_generation",
    "cancel_generation",
    "view_map",
    "measure_map",
    "bake_bodies",
    "explain_model",
];

/// The tools that only read.
pub const READ_ONLY: [&str; 6] = ["list_presets", "derive_physics", "get_generation", "view_map", "measure_map", "explain_model"];

fn hints(title: &str, read_only: bool, idempotent: bool, destructive: bool) -> ToolAnnotations {
    let a = ToolAnnotations::with_title(title).read_only(read_only);
    let a = if read_only { a } else { a.destructive(destructive) };
    a.idempotent(idempotent).open_world(false)
}

fn widths_text() -> String {
    WIDTHS.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
}

/// The description of the `width` argument.
pub fn width_description(max_w: u32) -> String {
    format!("Map width in pixels, one of {} (the height is half). This server accepts up to {max_w}.", widths_text())
}

/// The description of the `compute` argument, built from what the engine offers.
pub fn compute_description(caps: &Value) -> String {
    let modes = caps.get("modes").and_then(Value::as_array).cloned().unwrap_or_default();
    let key = |m: &Value| m.get("key").and_then(Value::as_str).unwrap_or("").to_string();
    let avail = |m: &Value| m.get("available").and_then(Value::as_bool).unwrap_or(false);
    let offered = modes.iter().filter(|m| avail(m)).map(key).collect::<Vec<_>>().join(", ");
    let refused = modes.iter().filter(|m| !avail(m)).map(key).collect::<Vec<_>>().join(", ");
    let engine = caps.get("engine").and_then(Value::as_str).unwrap_or("");
    let mut s = String::from(
        "Where the work runs. 'auto' (default): the engine picks the GPU when one is available and the map is large \
         enough for the upload to pay, otherwise the CPU; it never fails for lack of a device. 'cpu': every stage on \
         the CPU. 'gpu': the per-texel stages on the GPU, in float32 (the same map to float32 rounding and \
         statistically, not bit for bit). 'hybrid': row bands shared between the GPU and the CPU threads. 'gpu' and \
         'hybrid' need a GPU-capable engine: an explicit mode that is not available is an ERROR that gives the reason \
         (never a silent fallback). Check what is offered in the resource cloud-map-generator://capabilities or the \
         `compute` key of list_presets. ",
    );
    s.push_str(&format!("This server (engine {engine}) offers: {offered}"));
    if !refused.is_empty() {
        s.push_str(&format!("; not available: {refused}"));
    }
    s.push_str(". The result's `compute` says what was used.");
    s
}

fn schemas() -> Value {
    serde_json::from_str(include_str!("schemas.json")).expect("schemas.json is valid JSON")
}

fn object(v: &Value) -> Arc<JsonObject> {
    Arc::new(v.as_object().cloned().unwrap_or_default())
}

/// `about 50 s` or `seconds`.
fn est_text(est: &dyn Fn(u32) -> Option<f64>, w: u32) -> String {
    match est(w).filter(|s| *s != 0.0) {
        Some(s) => format!("about {} s", crate::pyfmt::fmt_g(s)),
        None => "seconds".to_string(),
    }
}

/// All ten tools. `caps` is the engine's capabilities, `max_w` the server's `--max-width`, `est` the typical
/// seconds of a generation per width.
pub fn build_tools(caps: &Value, max_w: u32, est: &dyn Fn(u32) -> Option<f64>) -> Vec<Tool> {
    let mut sch = schemas();
    let width_desc = width_description(max_w);
    let compute_desc = compute_description(caps);
    for name in ["generate_map", "start_generation"] {
        let props = &mut sch[name]["input"]["properties"];
        props["width"]["description"] = Value::String(width_desc.clone());
        props["compute"]["description"] = Value::String(compute_desc.clone());
    }
    let widths = widths_text();
    let t2048 = est_text(est, 2048);
    let t512 = est_text(est, 512);

    let list_presets = "List the built-in planets (Earth, a retrograde Earth, a fast and a slow rotator, a super-Earth, and Mars, Titan and Venus) with the physics behind each, the valid range of every physics input, the supported map widths, the largest width this server accepts (max_width) and the path of its output folder (output_folder: where results are saved and the only place, with read_dirs, that view_map(path), measure_map and bake_bodies read files from) and the compute modes this server offers (compute: which of auto, cpu, gpu and hybrid are available and, for the others, the reason; the GPU modes need a GPU-capable engine). Call this first to learn the preset keys and the units (days, km, m/s^2). A preset with experiment_required=true has clouds that are not water: the model only knows Earth's cloud amount, so it generates only with experiment=true and the result is an experiment, not a prediction. Read-only and instant. Next: derive_physics or generate_map.".to_string();

    let derive = "Compute in milliseconds, without generating anything, what the physics implies for the clouds: the Hadley-cell edge and storm-track latitudes (degrees), whether baroclinic eddies exist, the Rossby deformation radius (km), the number of cyclones per hemisphere (null without eddies) and the sense in which they turn in each hemisphere (cyclone_spin_north / cyclone_spin_south, seen from above that pole; a retrograde spin reverses it), the size of one map pixel in km at every width, whether the planet is bakeable (water clouds only) and warnings. Give a `preset` and/or `overrides`; overrides win. Call this BEFORE generate_map to check the planet is sane (a Hadley edge of 90 degrees means one global cell and no storm tracks, a huge cyclone count means a very fast spin). Example: preset='earth', overrides={'rotation_period_days': 4}. Read-only. Next: generate_map.".to_string();

    let generate = format!(
        "Generate a global cloud-coverage map for a planet and judge it against Earth. Define the planet with `preset` (a starting point such as 'earth') and `overrides` (only what differs, for example a 36 hour day is overrides={{'rotation_period_days': 1.5}}); the physics you do not override comes from the preset, and without a preset all four numbers are required. The map is equirectangular and north-up, one value in [0, 1] per pixel: the fraction of the sky covered by cloud, with an equatorial band, clear subtropics, storm tracks, cyclones and fronts, and no seam or polar pinch. Returns a summary (id, seconds, derived physics with the cyclone spin sense, metric rows generated-vs-Earth, a coarse zonal profile every 15 degrees), a PNG preview of the map (at most {PREVIEW_MAX_WIDTH} px wide) and, when save=true, the files written under the output folder (a subfolder named by the id): 16-bit PNG master, 8-bit preview, cloud type, high-cloud share, wind, raw R8 gzip with rows south-up, a JSON sidecar. Reading the metrics: relative_gap = (generated - Earth) / Earth, and the hint says what to change. The mean, the zonal profile and the histogram are reliable at any width; coherence, spectral slope and object sizes only at width 2048 or more (comparable=false below it), so to answer how close a planet's statistics are to Earth's, generate it at width 2048 with start_generation ({t2048}). width: {widths} (height is half); this server accepts up to {max_w}. Widths up to {SYNC_MAX_WIDTH} run inside this call ({t512} at 512); wider maps use start_generation. Options: match_earth forces the statistics toward Earth's (do not use it to measure closeness); isotropic is a baseline without the flow; experiment allows non-water clouds (a methane planet; not realistic); surface='synthetic' adds land and relief. The same inputs and seed give the same map. compute chooses where the work runs ('auto' by default, 'cpu', 'gpu', 'hybrid'): 'gpu' and 'hybrid' need a GPU-capable engine, and an explicit mode this server does not offer is an error that gives the reason, never a silent fallback; the resource cloud-map-generator://capabilities (or list_presets) says which modes are offered, and the result carries compute.used, the devices and the seconds each worked. Example: preset='earth', seed=42, width=512. Next: view_map(id, view='globe') to look from other angles, read the metrics, adjust overrides, regenerate; then start_generation for a large map."
    );

    let start = format!(
        "Queue a map generation and return at once with a job_id and an estimate in seconds, so you can do something else meanwhile (this is the background mode). Takes exactly the same arguments as generate_map, but any width up to this server's --max-width ({max_w}; a larger width is refused and cannot be raised from a tool call); use it for widths above {SYNC_MAX_WIDTH} ({t2048} at 2048) and for the width-2048 maps that compare a planet's full statistics with Earth. compute (auto, cpu, gpu, hybrid) is the same as in generate_map: the GPU modes need a GPU-capable engine and an explicit mode this server does not offer is refused at once with the reason (see the resource cloud-map-generator://capabilities). One job runs at a time and at most 4 wait; a fifth is refused. You are not notified: call get_generation(job_id) every few seconds until status is done (it then returns the same summary, metrics and preview as generate_map); cancel_generation(job_id) stops it. The job_id is only valid in this server session. Example: preset='earth', width=2048. Next: get_generation."
    );

    let get_generation = "Status of a job from start_generation: queued, running, done, failed or cancelled, with progress 0..1 and the current stage. When done it returns the same summary and preview image as generate_map; when failed, the error to act on. Cheap: call it every few seconds. The last 16 jobs of this server session are kept. Read-only.".to_string();

    let cancel = "Cancel a job from start_generation. A waiting job is cancelled at once; a running one stops at its next stage boundary (a few seconds; poll get_generation to see status cancelled). A finished job is left as it is. Nothing is written for a cancelled job.".to_string();

    let view = "Look at a cloud map as an image. Pass `id` (a result of this server session, from generate_map or get_generation) or `path` (a coverage PNG inside the output folder or a --read-dir directory), not both. view='map' returns the equirectangular map (size = width in pixels); view='globe' renders an orthographic globe seen from lon_deg (-180 to 180, east positive) and lat_deg (-90 to 90): clouds white over a dark sphere, north up and east to the right. This is how you check the poles, the fronts and the cyclone spirals from other angles: look from at least three viewpoints. A pole seen from above: lat_deg=90 is the north pole, lat_deg=-90 the south pole (lon_deg has no effect at exactly +-90; use 89.9 or -89.9 to choose the meridian). Seen from above a pole the surface turns the other way than from the equator: east runs counterclockwise around the north pole and clockwise around the south pole, so a cyclone turning the Earth way looks counterclockwise from lat_deg=90 and clockwise from lat_deg=-90; the expected sense of a given planet is derived.cyclone_spin_north / cyclone_spin_south in the generate_map result (to compare two planets, view both from the same pole with the same size). product (ids only): coverage (default), type (0 stratiform to 1 convective), high_share (share of cloud above 440 hPa) or wind (speed). size 64 to 1024 pixels, default 512. Example: id='3f9a1c2b4d5e', view='globe', lon_deg=120, lat_deg=-35. Read-only.".to_string();

    let measure = format!(
        "Measure any coverage PNG against Earth's reference map (NASA Blue Marble): global mean, zonal-profile error, coherence, spectral slope, cloud-object sizes, antimeridian seam and pole checks. The file (8-bit or 16-bit gray PNG, equirectangular 2:1, at least 256 px wide, up to {MAX_PNG_WIDTH}x{MAX_PNG_HEIGHT}) must be inside the output folder or a --read-dir directory; a relative path is looked up from there; any other location (system files, other absolute paths, ../ escapes, links) is refused. Returns the metric rows (generated, earth, relative_gap, comparable, hint) and the coarse zonal profile; structure rows only compare with Earth from width 2048. A map you just generated is already measured in its result, so use this for maps of earlier runs or made by other tools. Example: the 'preview' or 'master' path of a generate_map result's `files`. Read-only."
    );

    let bake = format!(
        "Bake the cloud maps of a catalog of planets to files, for a renderer that loads files instead of running the generator. Give the planets INLINE in `bodies` (a list of {{key, rotation_period_days, radius_km, gravity_mps2, scale_height_km, cloud_regime, measured}}; no file needed, the easy way: for example bodies=[{{key: 'alpha', rotation_period_days: 1.2, radius_km: 6000, gravity_mps2: 9, scale_height_km: 8}}]) or, for bodies with land and relief grids, as a bodies JSON file in `bodies_file` (spelling and format in the resource cloud-map-generator://docs/bodies-format: JSON {{\"bodies\": [{{\"key\", \"rotationPeriodDays\", \"radiusKm\", \"gravityMps2\", \"scaleHeightKm\", \"cloudRegime\", \"measured\"?, \"surface\"?}}]}}, camelCase, inside the output folder (list_presets gives its path) or a --read-dir directory). At most {MAX_BAKE_BODIES} bodies. Writes, under <out>/bake/<version>/, one map set per water-cloud body (the list) and a library of k seeds per climate class (for procedural worlds; it is baked too unless only='list'), plus manifest.json; rerunning the same version overwrites its files. Bodies with measured=true or non-water clouds are skipped (named in `skipped`). The widths are small ON PURPOSE (list_width and library_width default to {BAKE_DEFAULT_WIDTH}, k=1: about a minute for a few bodies): pass list_width=2048 and k=4 for production quality and expect minutes. url_prefix is written in front of the file paths of the manifest (default 'baked/<version>'); only='list' or 'library' bakes one part and writes NO manifest, so omit `only` for a catalog a renderer will load. Runs as one queued call and reports progress. Returns the manifest and the list of files. Example: bodies=[...three bodies...], version='v1'. Next: view_map or measure_map on a baked coverage PNG."
    );

    let explain = "A Markdown summary of the model: the physics inputs, the products, the known limits (frozen flow, no season, one storm track per hemisphere, Earth-anchored cloud amount, water clouds only) and what is deliberately simplified. Read it before trusting a map for a planet very unlike Earth. The same text is the resource cloud-map-generator://docs/model. Read-only.".to_string();

    // name, title, description, read_only, idempotent, destructive
    let defs: [(&str, &str, String, bool, bool, bool); 10] = [
        ("list_presets", "List planet presets", list_presets, true, true, false),
        ("derive_physics", "Derive the climate of a planet", derive, true, true, false),
        ("generate_map", "Generate a cloud map", generate, false, false, false),
        ("start_generation", "Start a generation job", start, false, false, false),
        ("get_generation", "Check a generation job", get_generation, true, true, false),
        ("cancel_generation", "Cancel a generation job", cancel, false, true, false),
        ("view_map", "Look at a map or a globe", view, true, true, false),
        ("measure_map", "Measure a map against Earth", measure, true, true, false),
        ("bake_bodies", "Bake the maps of a set of bodies", bake, false, true, true),
        ("explain_model", "What the model does and does not do", explain, true, true, false),
    ];
    defs.into_iter()
        .map(|(name, title, description, read_only, idempotent, destructive)| {
            Tool::new(name, description, object(&sch[name]["input"]))
                .with_title(title)
                .with_raw_output_schema(object(&sch[name]["output"]))
                .with_annotations(hints(title, read_only, idempotent, destructive))
        })
        .collect()
}
