//! The request side of the in-browser API: the validation of `POST /api/derive` and `POST /api/generate`.
//!
//! The rules and the messages are the local server's (`cloud-map-generator-cli/src/serve/request.rs`): the message names
//! the field, so the page shows the same sentence whichever engine answers. Differences: the seed is read from the
//! parsed JSON number (the page is the only client, so there is no raw-digits scan), and the widths are this
//! engine's own list.

use cloud_map_generator_core::climate::BodyInputs;
use cloud_map_generator_service::engine::{ComputeMode, MapOptions, Runtime, SurfaceChoice};
use cloud_map_generator_service::limits::{limit, PHYSICS_KEYS};
use cloud_map_generator_service::pyfmt::{fmt_g, is_printable_str};
use serde_json::{Map, Value};

/// An error answer: the HTTP status the page would see and the `{"error": message}` text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
}

/// A 400.
pub fn bad(message: impl Into<String>) -> ApiError {
    ApiError { status: 400, message: message.into() }
}

fn number(value: &Value, name: &str, lo: f64, hi: f64) -> Result<f64, ApiError> {
    let Value::Number(n) = value else {
        return Err(bad(format!("{name}: expected a number")));
    };
    let v = n.as_f64().filter(|v| v.is_finite()).ok_or_else(|| bad(format!("{name}: not a finite number")))?;
    if v < lo || v > hi {
        return Err(bad(format!("{name}: {} is outside [{}, {}]", fmt_g(v), fmt_g(lo), fmt_g(hi))));
    }
    Ok(v)
}

fn integer(value: &Value, name: &str, lo: i64, hi: i64) -> Result<i64, ApiError> {
    let Value::Number(n) = value else {
        return Err(bad(format!("{name}: expected an integer")));
    };
    let wide: i128 = if let Some(i) = n.as_i64() {
        i128::from(i)
    } else if let Some(u) = n.as_u64() {
        i128::from(u)
    } else {
        return Err(bad(format!("{name}: expected an integer")));
    };
    if wide < i128::from(lo) || wide > i128::from(hi) {
        return Err(bad(format!("{name}: {wide} is outside [{lo}, {hi}]")));
    }
    Ok(wide as i64)
}

fn object<'a>(value: &'a Value, name: &str, allowed: &[&str]) -> Result<&'a Map<String, Value>, ApiError> {
    let Value::Object(map) = value else {
        return Err(bad(format!("{name}: expected an object")));
    };
    if let Some(unknown) = map.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(bad(format!("{name}: unknown field '{unknown}'")));
    }
    Ok(map)
}

/// The four numbers (required, inside the limits) and `cloud_regime` (optional, default `"water"`) of a body.
pub fn parse_body(value: &Value, name: &str) -> Result<BodyInputs, ApiError> {
    let allowed = [PHYSICS_KEYS[0], PHYSICS_KEYS[1], PHYSICS_KEYS[2], PHYSICS_KEYS[3], "cloud_regime"];
    let map = object(value, name, &allowed)?;
    let mut nums = [0.0; 4];
    for (slot, key) in nums.iter_mut().zip(PHYSICS_KEYS) {
        let v = map.get(key).ok_or_else(|| bad(format!("{name}.{key}: missing")))?;
        let (lo, hi) = limit(key);
        *slot = number(v, &format!("{name}.{key}"), lo, hi)?;
    }
    let regime = match map.get("cloud_regime") {
        None => "water".to_string(),
        Some(Value::String(s)) if !s.trim().is_empty() && s.chars().count() <= 64 && is_printable_str(s) => s.trim().to_lowercase(),
        Some(_) => return Err(bad(format!("{name}.cloud_regime: expected a short text such as \"water\""))),
    };
    Ok(BodyInputs::new(Some(nums[0]), nums[1], nums[2], nums[3]).with_regime(regime))
}

/// `{"body": {...}}` of `POST /api/derive`.
pub fn parse_derive(value: &Value) -> Result<BodyInputs, ApiError> {
    let map = object(value, "request", &["body"])?;
    parse_body(map.get("body").ok_or_else(|| bad("body: missing"))?, "body")
}

/// The options of the map (`compute` is allowed in the same object and is read apart).
pub fn parse_options(value: Option<&Value>) -> Result<MapOptions, ApiError> {
    let empty = Value::Object(Map::new());
    let value = match value {
        None | Some(Value::Null) => &empty,
        Some(v) => v,
    };
    let map = object(value, "options", &["match_earth", "isotropic", "experiment", "surface", "compute"])?;
    let mut options = MapOptions::default();
    for (key, v) in map {
        match key.as_str() {
            "compute" => {}
            "surface" => {
                options.surface = match v.as_str() {
                    Some("none") => SurfaceChoice::None,
                    Some("synthetic") => SurfaceChoice::Synthetic,
                    _ => return Err(bad("options.surface: expected one of ['none', 'synthetic']")),
                }
            }
            flag => {
                let Value::Bool(on) = v else {
                    return Err(bad(format!("options.{flag}: expected true or false")));
                };
                match flag {
                    "match_earth" => options.match_earth = *on,
                    "isotropic" => options.isotropic = *on,
                    _ => options.experiment = *on,
                }
            }
        }
    }
    Ok(options)
}

/// `options.compute` (default `auto`): an unknown value, and a mode this engine does not offer, are a 400 with the reason.
pub fn parse_compute(value: Option<&Value>, runtime: &Runtime) -> Result<ComputeMode, ApiError> {
    match value.and_then(|v| v.get("compute")) {
        None => Ok(ComputeMode::Auto),
        Some(v) => runtime.check_compute(v, "options.compute").map_err(bad),
    }
}

/// A validated `POST /api/generate` request.
#[derive(Clone, Debug)]
pub struct GenRequest {
    pub body: BodyInputs,
    pub seed: u64,
    pub width: usize,
    pub options: MapOptions,
    pub compute: ComputeMode,
}

/// `{"body", "seed", "width", "options"?}` against the widths this engine offers.
pub fn parse_generate(value: &Value, widths: &[usize], runtime: &Runtime) -> Result<GenRequest, ApiError> {
    let map = object(value, "request", &["body", "seed", "width", "options"])?;
    for key in ["body", "seed", "width"] {
        if !map.contains_key(key) {
            return Err(bad(format!("{key}: missing")));
        }
    }
    let body = parse_body(&map["body"], "body")?;
    let (lo, hi) = limit("seed");
    let seed = integer(&map["seed"], "seed", lo as i64, hi as i64)? as u64;
    let width = match &map["width"] {
        Value::Number(n) if n.is_i64() || n.is_u64() => n.as_u64().and_then(|w| usize::try_from(w).ok()).filter(|w| widths.contains(w)),
        _ => None,
    }
    .ok_or_else(|| bad(format!("width: must be one of [{}]", widths.iter().map(usize::to_string).collect::<Vec<_>>().join(", "))))?;
    let options = parse_options(map.get("options"))?;
    let compute = parse_compute(map.get("options"), runtime)?;
    Ok(GenRequest { body, seed, width, options, compute })
}
