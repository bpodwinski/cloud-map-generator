//! The request side of the API: limits, and the validation of `POST /api/derive` and `POST /api/generate`.
//!
//! The rules and the 400 messages are `ui/server.py`'s (`parse_body`, `parse_options`, `parse_generate`): the
//! message names the field. One difference: `serde_json` keeps an object's keys sorted, not in the order the
//! client wrote them, so when SEVERAL fields of one object are wrong, the first one named is the first in
//! alphabetical order, not in the order of the request.

use super::engine::{ComputeMode, MapOptions, Runtime, SurfaceChoice};
use super::pyfmt::{fmt_g, is_printable_str};
use cloud_map_generator_core::climate::BodyInputs;
use serde_json::{Map, Value};

/// The largest request body, bytes.
pub const MAX_REQUEST_BYTES: usize = 64 * 1024;

/// An error answer: the status and the `{"error": message}` text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
}

impl ApiError {
    pub fn new(status: u16, message: impl Into<String>) -> ApiError {
        ApiError { status, message: message.into() }
    }
}

/// A 400.
pub fn bad(message: impl Into<String>) -> ApiError {
    ApiError::new(400, message)
}

// The ranges the API enforces and the cap on cyclones are the service crate's (the MCP tools enforce the same).
pub use cloud_map_generator_service::limits::{limit, MAX_CYCLONES_PER_HEMISPHERE, PHYSICS_KEYS};

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

/// `_object`: the value must be an object whose keys are all in `allowed` (the first unknown one, in sorted
/// order, is named).
fn object<'a>(value: &'a Value, name: &str, allowed: &[&str]) -> Result<&'a Map<String, Value>, ApiError> {
    let Value::Object(map) = value else {
        return Err(bad(format!("{name}: expected an object")));
    };
    if let Some(unknown) = map.keys().find(|k| !allowed.contains(&k.as_str())) {
        return Err(bad(format!("{name}: unknown field '{unknown}'")));
    }
    Ok(map)
}

/// `parse_body`: the four numbers (required, inside the limits) and `cloud_regime` (optional, default `"water"`,
/// trimmed and lower-cased) of a body.
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

/// `parse_derive`: `{"body": {...}}`.
pub fn parse_derive(value: &Value) -> Result<BodyInputs, ApiError> {
    let map = object(value, "request", &["body"])?;
    parse_body(map.get("body").ok_or_else(|| bad("body: missing"))?, "body")
}

/// `parse_options`: the options of the map (`compute` is allowed in the same object and is read apart).
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

/// `parse_compute`: `options.compute` (default `auto`); an unknown value, and a mode this engine does not offer,
/// are a 400 with the reason.
pub fn parse_compute(value: Option<&Value>, runtime: &Runtime) -> Result<ComputeMode, ApiError> {
    match value.and_then(|v| v.get("compute")) {
        None => Ok(ComputeMode::Auto),
        Some(v) => runtime.check_compute(v, "options.compute").map_err(bad),
    }
}

// ---------------------------------------------------------------------------------------------------------
// The raw text of the seed
//
// `serde_json` reads a number that does not fit 64 bits as a float, and `-0` as a float too, so the digits a
// client wrote (the message of the 400 names them, as Python's integer does) and the integer `-0` are only in the
// raw text. A small scan finds the top-level `seed` member's token there.

fn skip_ws(raw: &[u8], mut i: usize) -> usize {
    while i < raw.len() && matches!(raw[i], b' ' | b'\t' | b'\r' | b'\n') {
        i += 1;
    }
    i
}

/// The index after the string that starts at `i` (a quote).
fn skip_string(raw: &[u8], mut i: usize) -> usize {
    i += 1;
    while i < raw.len() {
        match raw[i] {
            b'\\' => i += 2,
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    raw.len()
}

/// The index after the JSON value that starts at `i` (the text is valid JSON: `serde_json` read it first).
fn skip_value(raw: &[u8], i: usize) -> usize {
    match raw.get(i) {
        Some(b'"') => skip_string(raw, i),
        Some(b'{') | Some(b'[') => {
            let (mut depth, mut j) = (0usize, i);
            while j < raw.len() {
                match raw[j] {
                    b'"' => {
                        j = skip_string(raw, j);
                        continue;
                    }
                    b'{' | b'[' => depth += 1,
                    b'}' | b']' => {
                        depth -= 1;
                        if depth == 0 {
                            return j + 1;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            raw.len()
        }
        _ => {
            let mut j = i;
            while j < raw.len() && !matches!(raw[j], b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n') {
                j += 1;
            }
            j
        }
    }
}

/// The text of the number of the top-level `seed` member (the last one, as a JSON parser keeps the last duplicate),
/// when it is an integer literal: an optional minus sign and digits, no point and no exponent.
pub fn raw_seed_token(raw: &[u8]) -> Option<&str> {
    let mut i = skip_ws(raw, 0);
    if raw.get(i) != Some(&b'{') {
        return None;
    }
    i += 1;
    let mut seed = None;
    loop {
        i = skip_ws(raw, i);
        if raw.get(i) != Some(&b'"') {
            break;
        }
        let end = skip_string(raw, i);
        let key: String = serde_json::from_slice(&raw[i..end]).ok()?;
        i = skip_ws(raw, end);
        if raw.get(i) != Some(&b':') {
            break;
        }
        i = skip_ws(raw, i + 1);
        let value_end = skip_value(raw, i);
        if key == "seed" {
            seed = Some(&raw[i..value_end]);
        }
        i = skip_ws(raw, value_end);
        if raw.get(i) == Some(&b',') {
            i += 1;
        } else {
            break;
        }
    }
    let token = std::str::from_utf8(seed?).ok()?;
    let digits = token.strip_prefix('-').unwrap_or(token);
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then_some(token)
}

/// The seed an integer-literal token names, or the 400 that names its digits: `-0` is the integer 0, a token too
/// wide for 64 bits is outside the range like any other.
fn seed_from_token(token: &str, lo: i64, hi: i64) -> Result<i64, ApiError> {
    let digits = token.strip_prefix('-').unwrap_or(token);
    if digits.bytes().all(|b| b == b'0') {
        return Ok(0);
    }
    match token.parse::<i64>() {
        Ok(v) if v >= lo && v <= hi => Ok(v),
        _ => Err(bad(format!("seed: {token} is outside [{lo}, {hi}]"))),
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

/// `parse_generate`: `{"body", "seed", "width", "options"?}` against the widths this server offers.
#[cfg(test)]
pub fn parse_generate(value: &Value, widths: &[usize], runtime: &Runtime) -> Result<GenRequest, ApiError> {
    parse_generate_with(value, None, widths, runtime)
}

/// [`parse_generate`] for a request that arrived as text: the seed is read from its raw digits (see [`raw_seed_token`]).
pub fn parse_generate_text(value: &Value, raw: &[u8], widths: &[usize], runtime: &Runtime) -> Result<GenRequest, ApiError> {
    parse_generate_with(value, raw_seed_token(raw), widths, runtime)
}

fn parse_generate_with(value: &Value, seed_token: Option<&str>, widths: &[usize], runtime: &Runtime) -> Result<GenRequest, ApiError> {
    let map = object(value, "request", &["body", "seed", "width", "options"])?;
    for key in ["body", "seed", "width"] {
        if !map.contains_key(key) {
            return Err(bad(format!("{key}: missing")));
        }
    }
    let body = parse_body(&map["body"], "body")?;
    let (lo, hi) = limit("seed");
    let seed = match seed_token {
        Some(token) => seed_from_token(token, lo as i64, hi as i64)?,
        None => integer(&map["seed"], "seed", lo as i64, hi as i64)?,
    } as u64;
    let width = match &map["width"] {
        Value::Number(n) if n.is_i64() || n.is_u64() => n.as_u64().and_then(|w| usize::try_from(w).ok()).filter(|w| widths.contains(w)),
        _ => None,
    }
    .ok_or_else(|| bad(format!("width: must be one of [{}]", widths.iter().map(usize::to_string).collect::<Vec<_>>().join(", "))))?;
    let options = parse_options(map.get("options"))?;
    let compute = parse_compute(map.get("options"), runtime)?;
    Ok(GenRequest { body, seed, width, options, compute })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn earth() -> Value {
        json!({"rotation_period_days": 0.99727, "radius_km": 6371, "gravity_mps2": 9.81, "scale_height_km": 8.0, "cloud_regime": "water"})
    }

    fn runtime() -> Runtime {
        Runtime::cpu_only("no adapter")
    }

    fn err(r: Result<impl std::fmt::Debug, ApiError>) -> String {
        let e = r.unwrap_err();
        assert_eq!(e.status, 400);
        e.message
    }

    #[test]
    fn a_good_body_parses_and_normalizes_its_regime() {
        let b = parse_body(&earth(), "body").unwrap();
        assert_eq!((b.rotation_period_days, b.radius_km, b.gravity_mps2, b.scale_height_km), (Some(0.99727), 6371.0, 9.81, 8.0));
        assert_eq!(b.cloud_regime, "water");
        let mut v = earth();
        v["cloud_regime"] = json!(" Thin Water ICE ");
        assert_eq!(parse_body(&v, "body").unwrap().cloud_regime, "thin water ice");
        v.as_object_mut().unwrap().remove("cloud_regime");
        assert_eq!(parse_body(&v, "body").unwrap().cloud_regime, "water");
    }

    #[test]
    fn every_bad_body_is_a_400_that_names_the_field() {
        let with = |k: &str, v: Value| {
            let mut e = earth();
            e[k] = v;
            e
        };
        let cases: Vec<(Value, &str)> = vec![
            (json!([1, 2]), "body: expected an object"),
            (json!({}), "body.rotation_period_days: missing"),
            (with("mass_kg", json!(1)), "body: unknown field 'mass_kg'"),
            (with("radius_km", json!("6371")), "body.radius_km: expected a number"),
            (with("radius_km", json!(true)), "body.radius_km: expected a number"),
            (with("radius_km", Value::Null), "body.radius_km: expected a number"),
            (with("rotation_period_days", json!(401)), "body.rotation_period_days: 401 is outside [-400, 400]"),
            (with("rotation_period_days", json!(-401)), "body.rotation_period_days: -401 is outside [-400, 400]"),
            (with("radius_km", json!(499)), "body.radius_km: 499 is outside [500, 30000]"),
            (with("radius_km", json!(30001)), "body.radius_km: 30001 is outside [500, 30000]"),
            (with("gravity_mps2", json!(0.05)), "body.gravity_mps2: 0.05 is outside [0.1, 100]"),
            (with("gravity_mps2", json!(101)), "body.gravity_mps2: 101 is outside [0.1, 100]"),
            (with("scale_height_km", json!(0.1)), "body.scale_height_km: 0.1 is outside [0.5, 100]"),
            (with("scale_height_km", json!(100.5)), "body.scale_height_km: 100.5 is outside [0.5, 100]"),
            (with("cloud_regime", json!(5)), "body.cloud_regime: expected a short text such as \"water\""),
            (with("cloud_regime", json!("  ")), "body.cloud_regime: expected a short text such as \"water\""),
            (with("cloud_regime", json!("a\tb")), "body.cloud_regime: expected a short text such as \"water\""),
            (with("cloud_regime", json!("x".repeat(65))), "body.cloud_regime: expected a short text such as \"water\""),
        ];
        for (body, want) in cases {
            assert_eq!(err(parse_body(&body, "body")), want);
        }
        let mut missing = earth();
        missing.as_object_mut().unwrap().remove("gravity_mps2");
        assert_eq!(err(parse_body(&missing, "body")), "body.gravity_mps2: missing");
        assert_eq!(parse_body(&with("cloud_regime", json!("x".repeat(64))), "body").unwrap().cloud_regime.len(), 64);
    }

    #[test]
    fn the_limits_are_inclusive() {
        let mut e = earth();
        e["rotation_period_days"] = json!(-400);
        e["radius_km"] = json!(30000);
        e["gravity_mps2"] = json!(0.1);
        e["scale_height_km"] = json!(100);
        assert!(parse_body(&e, "body").is_ok());
        e["rotation_period_days"] = json!(0);
        assert!(parse_body(&e, "body").is_ok(), "no spin is allowed");
    }

    #[test]
    fn derive_wants_exactly_a_body() {
        assert!(parse_derive(&json!({"body": earth()})).is_ok());
        assert_eq!(err(parse_derive(&json!([1]))), "request: expected an object");
        assert_eq!(err(parse_derive(&json!({}))), "body: missing");
        assert_eq!(err(parse_derive(&json!({"body": earth(), "extra": 1}))), "request: unknown field 'extra'");
    }

    fn generate(extra: Value) -> Result<GenRequest, ApiError> {
        let mut req = json!({"body": earth(), "seed": 42, "width": 256});
        for (k, v) in extra.as_object().unwrap() {
            req[k] = v.clone();
        }
        parse_generate(&req, &[256, 512], &runtime())
    }

    #[test]
    fn a_good_generate_request_parses_with_defaults() {
        let r = generate(json!({})).unwrap();
        assert_eq!((r.seed, r.width, r.compute), (42, 256, ComputeMode::Auto));
        assert_eq!(r.options, MapOptions::default());
        let r = generate(json!({"options": {"surface": "synthetic", "isotropic": true, "compute": "cpu"}})).unwrap();
        assert_eq!((r.options.surface, r.options.isotropic, r.options.match_earth, r.compute), (SurfaceChoice::Synthetic, true, false, ComputeMode::Cpu));
        assert_eq!(generate(json!({"options": null})).unwrap().options, MapOptions::default());
        assert_eq!(generate(json!({"seed": 0})).unwrap().seed, 0);
        assert_eq!(generate(json!({"seed": 2147483647})).unwrap().seed, 2147483647);
    }

    #[test]
    fn every_bad_generate_request_is_a_400_that_names_the_field() {
        let cases: Vec<(Value, &str)> = vec![
            (json!({"width": 300}), "width: must be one of [256, 512]"),
            (json!({"width": 1024}), "width: must be one of [256, 512]"),
            (json!({"width": "256"}), "width: must be one of [256, 512]"),
            (json!({"width": 256.0}), "width: must be one of [256, 512]"),
            (json!({"width": true}), "width: must be one of [256, 512]"),
            (json!({"width": -256}), "width: must be one of [256, 512]"),
            (json!({"seed": -1}), "seed: -1 is outside [0, 2147483647]"),
            (json!({"seed": 2147483648u64}), "seed: 2147483648 is outside [0, 2147483647]"),
            (json!({"seed": 18446744073709551615u64}), "seed: 18446744073709551615 is outside [0, 2147483647]"),
            (json!({"seed": 4.5}), "seed: expected an integer"),
            (json!({"seed": "42"}), "seed: expected an integer"),
            (json!({"seed": true}), "seed: expected an integer"),
            (json!({"quality": "high"}), "request: unknown field 'quality'"),
            (json!({"options": {"turbo": true}}), "options: unknown field 'turbo'"),
            (json!({"options": {"match_earth": 1}}), "options.match_earth: expected true or false"),
            (json!({"options": {"isotropic": "yes"}}), "options.isotropic: expected true or false"),
            (json!({"options": {"experiment": null}}), "options.experiment: expected true or false"),
            (json!({"options": {"surface": "earth"}}), "options.surface: expected one of ['none', 'synthetic']"),
            (json!({"options": {"surface": true}}), "options.surface: expected one of ['none', 'synthetic']"),
            (json!({"options": ["experiment"]}), "options: expected an object"),
            (json!({"body": {"rotation_period_days": 1}}), "body.radius_km: missing"),
            (json!({"body": {"rotation_period_days": 1, "radius_km": 6371, "gravity_mps2": 500, "scale_height_km": 8}}), "body.gravity_mps2: 500 is outside [0.1, 100]"),
        ];
        for (extra, want) in cases {
            assert_eq!(err(generate(extra.clone())), want, "{extra}");
        }
        for key in ["body", "seed", "width"] {
            let mut req = json!({"body": earth(), "seed": 42, "width": 256});
            req.as_object_mut().unwrap().remove(key);
            assert_eq!(err(parse_generate(&req, &[256], &runtime())), format!("{key}: missing"));
        }
        assert_eq!(err(parse_generate(&json!(7), &[256], &runtime())), "request: expected an object");
    }

    #[test]
    fn compute_is_read_apart_from_the_options_of_the_map() {
        assert_eq!(parse_options(Some(&json!({"compute": "cpu"}))).unwrap(), MapOptions::default());
        assert_eq!(parse_compute(None, &runtime()), Ok(ComputeMode::Auto));
        assert_eq!(parse_compute(Some(&json!({})), &runtime()), Ok(ComputeMode::Auto));
        let e = parse_compute(Some(&json!({"compute": "gpu"})), &runtime()).unwrap_err();
        assert!(e.status == 400 && e.message.starts_with("options.compute: 'gpu' is not available: no adapter"), "{e:?}");
        assert_eq!(err(generate(json!({"options": {"compute": "cpu", "turbo": true}}))), "options: unknown field 'turbo'");
        assert!(err(generate(json!({"options": {"compute": "turbo"}}))).starts_with("options.compute: expected one of 'auto', 'cpu', 'gpu', 'hybrid'"));
    }

    fn from_text(text: &str) -> Result<GenRequest, ApiError> {
        let value: Value = serde_json::from_str(text).unwrap();
        parse_generate_text(&value, text.as_bytes(), &[256, 512], &runtime())
    }

    fn request_text(seed: &str) -> String {
        format!(r#"{{"body": {}, "seed": {seed}, "width": 256}}"#, earth())
    }

    #[test]
    fn the_seed_token_is_found_in_the_raw_text() {
        assert_eq!(raw_seed_token(br#"{"a": {"seed": 5}, "seed": 12, "b": [1, {"seed": 9}]}"#), Some("12"));
        assert_eq!(raw_seed_token(br#"{"seed": 1, "seed": -0}"#), Some("-0"), "the last duplicate wins");
        assert_eq!(raw_seed_token(br#"{ "se\u0065d" : 7 }"#), Some("7"), "an escaped key is the same key");
        assert_eq!(raw_seed_token(br#"{"s": "{\"seed\": 3}", "seed": 4}"#), Some("4"), "text inside strings is skipped");
        for not_integer in ["4.5", "1e3", "\"7\"", "true", "null", "[1]", "1180591620717411303424.0"] {
            assert_eq!(raw_seed_token(format!(r#"{{"seed": {not_integer}}}"#).as_bytes()), None, "{not_integer}");
        }
        assert_eq!(raw_seed_token(b"[1]"), None);
        assert_eq!(raw_seed_token(br#"{"width": 5}"#), None);
    }

    #[test]
    fn a_seed_beyond_64_bits_is_named_by_its_digits() {
        for digits in ["1180591620717411303424", "10000000000000000000000000", "-1180591620717411303424", "18446744073709551616"] {
            assert_eq!(err(from_text(&request_text(digits))), format!("seed: {digits} is outside [0, 2147483647]"));
        }
        assert_eq!(err(from_text(&request_text("2147483648"))), "seed: 2147483648 is outside [0, 2147483647]");
        assert_eq!(err(from_text(&request_text("-1"))), "seed: -1 is outside [0, 2147483647]");
        assert_eq!(from_text(&request_text("2147483647")).unwrap().seed, 2147483647);
        // The body is still checked first, as Python does.
        let bad_body = r#"{"body": {}, "seed": 1180591620717411303424, "width": 256}"#;
        assert_eq!(err(parse_generate_text(&serde_json::from_str(bad_body).unwrap(), bad_body.as_bytes(), &[256], &runtime())), "body.rotation_period_days: missing");
    }

    #[test]
    fn minus_zero_is_the_seed_zero_and_other_non_integers_stay_refused() {
        assert_eq!(from_text(&request_text("-0")).unwrap().seed, 0);
        assert_eq!(from_text(&request_text("0")).unwrap().seed, 0);
        for text in ["4.5", "1e3", "\"7\"", "true", "null", "1180591620717411303424.0"] {
            assert_eq!(err(from_text(&request_text(text))), "seed: expected an integer", "{text}");
        }
    }
}
