//! The resources: the model's documentation, the presets, the capabilities, Earth's reference map, and the files
//! of the results saved in this server session (`cloud-map-generator://result/{id}/{name}`).

use base64::Engine as _;
use rmcp::model::{ReadResourceResult, Resource, ResourceContents, ResourceTemplate};
use rmcp::ErrorData as McpError;
use serde_json::{json, Map, Value};

use crate::backend::{LIMITS, WIDTHS};
use crate::prompts::BODIES_FORMAT;
use crate::pyfmt::{py_repr, snippet};
use crate::sandbox::{resolve_path, valid_result_id};
use crate::server::{Runtime, RESULT_FILES, URI_PREFIX};

/// Earth's cloud fraction (NASA Blue Marble, 2048x1024 gray PNG, north-up): what every map is measured against.
pub static EARTH_REFERENCE_PNG: &[u8] = cloud_map_generator_service::REFERENCE_PNG;

fn uri(tail: &str) -> String {
    format!("{URI_PREFIX}{tail}")
}

/// The five fixed resources.
pub fn list_resources() -> Vec<Resource> {
    vec![
        Resource::new(uri("docs/model"), "model")
            .with_title("What the model does and does not do")
            .with_description("Markdown summary of the model, its products and its known limits.")
            .with_mime_type("text/markdown"),
        Resource::new(uri("docs/bodies-format"), "bodies-format")
            .with_title("Bodies file format")
            .with_description("The JSON bodies file bake_bodies reads (camelCase keys, optional surface grids), with an example; bake_bodies can also take the bodies inline.")
            .with_mime_type("text/markdown"),
        Resource::new(uri("presets"), "presets")
            .with_title("Planet presets")
            .with_description("The built-in planets with their physics, plus the valid ranges, as JSON.")
            .with_mime_type("application/json"),
        Resource::new(uri("capabilities"), "capabilities")
            .with_title("Compute modes and devices")
            .with_description("Which compute modes (auto, cpu, gpu, hybrid) this server offers and why it does not offer the others, the engine that answers and the CPU and GPU it found, as JSON. Read it before asking for compute=gpu or hybrid.")
            .with_mime_type("application/json"),
        Resource::new(uri("reference/earth-coverage.png"), "earth-coverage")
            .with_title("Earth's reference cloud map")
            .with_description("Earth's cloud fraction (NASA Blue Marble, 2048x1024 gray PNG, north-up): what every map is measured against.")
            .with_mime_type("image/png"),
    ]
}

/// The five templates, one per file of a saved result.
pub fn list_templates() -> Vec<ResourceTemplate> {
    RESULT_FILES
        .iter()
        .map(|(name, _, mime)| {
            ResourceTemplate::new(uri(&format!("result/{{id}}/{name}")), format!("result-{name}"))
                .with_title(format!("{name} of a saved result"))
                .with_description(format!(
                    "{name} of a result made by generate_map or start_generation in this server session (saved results only; the id is in the summary)."
                ))
                .with_mime_type(*mime)
        })
        .collect()
}

fn text(uri: &str, body: String, mime: &str) -> ReadResourceResult {
    ReadResourceResult::new(vec![ResourceContents::text(body, uri).with_mime_type(mime)])
}

fn blob(uri: &str, bytes: &[u8], mime: &str) -> ReadResourceResult {
    ReadResourceResult::new(vec![
        ResourceContents::blob(base64::engine::general_purpose::STANDARD.encode(bytes), uri).with_mime_type(mime),
    ])
}

fn not_found(message: String) -> McpError {
    McpError::resource_not_found(message, None)
}

/// Read one resource. Errors say what was wrong, like the tools'.
pub fn read(rt: &Runtime, requested: &str) -> Result<ReadResourceResult, McpError> {
    match requested.strip_prefix(URI_PREFIX) {
        Some("docs/model") => return Ok(text(requested, rt.backend.explain_model(), "text/markdown")),
        Some("docs/bodies-format") => return Ok(text(requested, BODIES_FORMAT.to_string(), "text/markdown")),
        Some("presets") => {
            let limits: Map<String, Value> = LIMITS.iter().map(|(n, lo, hi)| ((*n).to_string(), json!([lo, hi]))).collect();
            let doc = json!({
                "presets": rt.backend.presets().iter().map(|p| p.to_json()).collect::<Vec<_>>(),
                "limits": limits,
                "widths": WIDTHS,
            });
            return Ok(text(requested, serde_json::to_string_pretty(&doc).unwrap_or_default(), "application/json"));
        }
        Some("capabilities") => {
            return Ok(text(requested, serde_json::to_string_pretty(&rt.backend.capabilities()).unwrap_or_default(), "application/json"));
        }
        Some("reference/earth-coverage.png") => return Ok(blob(requested, EARTH_REFERENCE_PNG, "image/png")),
        _ => {}
    }
    let unknown = || not_found(format!("Unknown resource: {requested}"));
    let rest = requested.strip_prefix(&uri("result/")).ok_or_else(unknown)?;
    let (id, name) = rest.split_once('/').ok_or_else(unknown)?;
    let (_, key, mime) = *RESULT_FILES.iter().find(|(n, _, _)| *n == name).ok_or_else(unknown)?;
    if id.is_empty() || id.chars().all(|c| c == '.') {
        return Err(unknown());
    }
    if !valid_result_id(id) {
        return Err(not_found(format!("{} is not a result id (a lowercase hex token).", py_repr(&snippet(id, 40)))));
    }
    let stored = rt.results.get(id).ok_or_else(|| {
        not_found(format!(
            "Unknown result id {}: only the last {} results of this server session are kept.",
            py_repr(id),
            rt.results.max_results()
        ))
    })?;
    if stored.folder.is_none() {
        return Err(McpError::invalid_params(
            format!("Result {id} was generated with save=false, so it has no files. Run generate_map with save=true."),
            None,
        ));
    }
    let target = stored.files.get(key).map(|p| resolve_path(p).unwrap_or_else(|_| p.clone()));
    let target = match target {
        Some(t) if rt.sandbox.contains(&t) && t.is_file() => t,
        _ => return Err(not_found(format!("{name} is not among the files of result {id}."))),
    };
    let data = rt.sandbox.read_bytes(&target).map_err(|e| McpError::invalid_params(e.to_string(), None))?;
    if mime.starts_with("application/json") {
        Ok(text(requested, String::from_utf8_lossy(&data).to_string(), mime))
    } else {
        Ok(blob(requested, &data, mime))
    }
}
