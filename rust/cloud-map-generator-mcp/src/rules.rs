//! The service-level rules every engine shares, worded once: the range of each physics input, what a preset key
//! may be, what a compute mode may be. A port of the validation half of `cloudgen/service.py`; the messages are
//! written for a reader who cannot see the code (a person or an AI agent).

use serde_json::Value;

use crate::backend::{limit, BackendError, Physics, Preset, COMPUTE_MODES, LIMIT_UNITS, PHYSICS_FIELDS};
use crate::pyfmt::{close_match, fmt_g, py_repr};

fn unit_of(name: &str) -> &'static str {
    LIMIT_UNITS.iter().find(|(n, _)| *n == name).map(|(_, u)| *u).unwrap_or("")
}

/// A finite number inside `[lo, hi]`.
pub fn number(value: f64, name: &str, lo: f64, hi: f64, unit: &str) -> Result<f64, BackendError> {
    let suffix = if unit.is_empty() { String::new() } else { format!(" ({unit})") };
    if !value.is_finite() {
        return Err(BackendError::message(format!(
            "{name}: {} is not a finite number. Pass a number between {} and {}{suffix}.",
            fmt_g(value),
            fmt_g(lo),
            fmt_g(hi)
        )));
    }
    if value < lo || value > hi {
        return Err(BackendError::message(format!(
            "{name}: {} is outside the allowed range [{}, {}]{suffix}. Choose a value inside that range.",
            fmt_g(value),
            fmt_g(lo),
            fmt_g(hi)
        )));
    }
    Ok(value)
}

/// A physics input checked against its range in [`crate::backend::LIMITS`].
pub fn physics_number(value: f64, name: &str) -> Result<f64, BackendError> {
    let (lo, hi) = limit(name);
    number(value, name, lo, hi, unit_of(name))
}

/// A cloud regime: a short printable word, stripped and lower-cased.
pub fn regime(value: &str) -> Result<String, BackendError> {
    if value.trim().is_empty() || value.chars().count() > 64 || !value.chars().all(crate::pyfmt::is_printable) {
        return Err(BackendError::message(format!(
            "cloud_regime: expected a short text such as \"water\", got the text {}. Only \"water\" is generated \
             without the experiment option.",
            py_repr(value)
        )));
    }
    Ok(value.trim().to_lowercase())
}

/// The fields a caller may replace on a preset (or give all of, without one). `None` counts as absent.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Overrides {
    pub rotation_period_days: Option<f64>,
    pub radius_km: Option<f64>,
    pub gravity_mps2: Option<f64>,
    pub scale_height_km: Option<f64>,
    pub cloud_regime: Option<String>,
}

fn find_preset<'a>(presets: &'a [Preset], preset: &str) -> Result<&'a Preset, BackendError> {
    let keys: Vec<&str> = presets.iter().map(|p| p.key.as_str()).collect();
    if preset.trim().is_empty() {
        return Err(BackendError::message(format!(
            "preset: expected a preset key such as \"earth\", got the text {}. Available presets: {}.",
            py_repr(preset),
            keys.join(", ")
        )));
    }
    let mut norm = preset.trim().to_lowercase().replace('_', "-").replace(' ', "-");
    norm = match norm.as_str() {
        "fast" => "fast-rotator".to_string(),
        "slow" => "slow-rotator".to_string(),
        "retrograde-earth" => "earth-retrograde".to_string(),
        "superearth" => "super-earth".to_string(),
        _ => norm,
    };
    if let Some(p) = presets.iter().find(|p| p.key == norm) {
        return Ok(p);
    }
    let hint = close_match(&norm, &keys).map(|c| format!(" Did you mean '{c}'?")).unwrap_or_default();
    Err(BackendError::message(format!(
        "preset: unknown preset '{preset}'.{hint} Available presets: {}. list_presets() gives their physics; or \
         leave the preset out and give {}.",
        keys.join(", "),
        PHYSICS_FIELDS.join(", ")
    )))
}

/// A validated [`Physics`] from a preset key, explicit fields, or a preset with some fields replaced. Without a
/// preset all four numbers are required and the regime defaults to `water`; every number is checked against
/// [`crate::backend::LIMITS`].
pub fn resolve_physics(presets: &[Preset], preset: Option<&str>, overrides: &Overrides) -> Result<Physics, BackendError> {
    let mut rotation = None;
    let mut radius = None;
    let mut gravity = None;
    let mut scale_height = None;
    let mut cloud = None;
    if let Some(key) = preset {
        let p = &find_preset(presets, key)?.physics;
        rotation = Some(p.rotation_period_days);
        radius = Some(p.radius_km);
        gravity = Some(p.gravity_mps2);
        scale_height = Some(p.scale_height_km);
        cloud = Some(p.cloud_regime.clone());
    }
    rotation = overrides.rotation_period_days.or(rotation);
    radius = overrides.radius_km.or(radius);
    gravity = overrides.gravity_mps2.or(gravity);
    scale_height = overrides.scale_height_km.or(scale_height);
    cloud = overrides.cloud_regime.clone().or(cloud);
    let given = [rotation, radius, gravity, scale_height];
    let missing: Vec<&str> = PHYSICS_FIELDS.iter().zip(given.iter()).filter(|(_, v)| v.is_none()).map(|(n, _)| *n).collect();
    if !missing.is_empty() {
        let keys: Vec<&str> = presets.iter().map(|p| p.key.as_str()).collect();
        return Err(BackendError::message(format!(
            "physics: missing {}. Give a preset key ({}) or all of: {}.",
            missing.join(", "),
            keys.join(", "),
            PHYSICS_FIELDS.join(", ")
        )));
    }
    Ok(Physics {
        rotation_period_days: physics_number(rotation.unwrap_or_default(), "rotation_period_days")?,
        radius_km: physics_number(radius.unwrap_or_default(), "radius_km")?,
        gravity_mps2: physics_number(gravity.unwrap_or_default(), "gravity_mps2")?,
        scale_height_km: physics_number(scale_height.unwrap_or_default(), "scale_height_km")?,
        cloud_regime: regime(cloud.as_deref().unwrap_or("water"))?,
    })
}

/// The compute mode `value` names, or an error. Refused: anything that is not one of the four modes (the message
/// names them) and a mode the engine does not offer (the message gives the reason from `capabilities` and says to
/// use `auto` or `cpu`). There is no silent fallback: an explicit mode that cannot run is an error, only `auto`
/// picks for itself. `name` is the field the message blames.
pub fn check_compute(capabilities: &Value, value: &str, name: &str) -> Result<String, BackendError> {
    let allowed = COMPUTE_MODES.iter().map(|m| format!("'{m}'")).collect::<Vec<_>>().join(", ");
    if !COMPUTE_MODES.contains(&value) {
        let hint = close_match(&value.trim().to_lowercase(), &COMPUTE_MODES)
            .map(|c| format!(" Did you mean '{c}'?"))
            .unwrap_or_default();
        return Err(BackendError::message(format!(
            "{name}: expected one of {allowed}, got the text {}.{hint} 'auto' lets the engine choose; the engine's \
             capabilities list what it offers.",
            py_repr(value)
        )));
    }
    let modes = capabilities.get("modes").and_then(Value::as_array).cloned().unwrap_or_default();
    let row = modes.iter().find(|m| m.get("key").and_then(Value::as_str) == Some(value));
    if let Some(row) = row {
        if row.get("available").and_then(Value::as_bool) == Some(false) {
            let reason = row.get("reason").and_then(Value::as_str).unwrap_or("not available");
            return Err(BackendError::message(format!(
                "{name}: '{value}' is not available: {reason} On this engine use 'auto' (it never fails for lack \
                 of a device) or 'cpu'; the engine's capabilities say what it offers."
            )));
        }
    }
    Ok(value.to_string())
}

/// `service._check_bodies`: unique keys, the four numbers present and finite, in range for a body that bakes.
pub fn check_bodies(records: &[Value], bake_list: bool) -> Result<(), BackendError> {
    let mut seen: Vec<String> = Vec::new();
    for (i, rec) in records.iter().enumerate() {
        let key = rec.get("key").and_then(Value::as_str).filter(|k| !k.trim().is_empty());
        let Some(key) = key else {
            return Err(BackendError::message(format!("bodies_file: record {i} needs an object with a non-empty text \"key\".")));
        };
        let where_ = format!("bodies_file: body '{key}'");
        if seen.iter().any(|k| k == key) {
            return Err(BackendError::message(format!("{where_} appears twice; keys must be unique (they name the output files).")));
        }
        seen.push(key.to_string());
        if !bake_list {
            continue;
        }
        let mut nums = Vec::new();
        for (jf, pf) in [
            ("rotationPeriodDays", "rotation_period_days"),
            ("radiusKm", "radius_km"),
            ("gravityMps2", "gravity_mps2"),
            ("scaleHeightKm", "scale_height_km"),
        ] {
            let unit = LIMIT_UNITS.iter().find(|(n, _)| *n == pf).map_or("", |(_, u)| *u);
            let Some(v) = rec.get(jf) else {
                return Err(BackendError::message(format!("{where_} is missing \"{jf}\" (a number, {unit}).")));
            };
            match v.as_f64().filter(|x| x.is_finite() && !v.is_boolean()) {
                Some(x) => nums.push((pf, x)),
                None => {
                    return Err(BackendError::message(format!("{where_}: \"{jf}\" must be a finite number, got {v}.")));
                }
            }
        }
        let measured = rec.get("measured").and_then(Value::as_bool).unwrap_or(false);
        let regime = rec.get("cloudRegime").and_then(Value::as_str).unwrap_or("unknown");
        if !measured && regime == "water" {
            for (pf, v) in nums {
                let (lo, hi) = limit(pf);
                if v < lo || v > hi {
                    let unit = LIMIT_UNITS.iter().find(|(n, _)| *n == pf).map_or("", |(_, u)| *u);
                    return Err(BackendError::message(format!(
                        "{where_}: \"{pf}\" {} is outside the allowed range [{}, {}] ({unit}). Fix the record, or mark the body \"measured\": true to leave it out of the bake.",
                        fmt_g(v),
                        fmt_g(lo),
                        fmt_g(hi)
                    )));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn presets() -> Vec<Preset> {
        vec![
            Preset {
                key: "earth".into(),
                label: "Earth".into(),
                physics: Physics::new(0.99727, 6371.0, 9.81, 8.0),
                experiment_required: false,
                note: String::new(),
            },
            Preset {
                key: "mars".into(),
                label: "Mars".into(),
                physics: Physics::new(1.02596, 3389.5, 3.721, 11.1).with_regime("thin water ice"),
                experiment_required: true,
                note: String::new(),
            },
        ]
    }

    #[test]
    fn a_preset_and_its_overrides_make_a_physics() {
        let p = resolve_physics(&presets(), Some("earth"), &Overrides { rotation_period_days: Some(4.0), ..Default::default() }).unwrap();
        assert_eq!(p.rotation_period_days, 4.0);
        assert_eq!(p.radius_km, 6371.0);
        assert_eq!(p.cloud_regime, "water");
        let mars = resolve_physics(&presets(), Some(" Mars "), &Overrides::default()).unwrap();
        assert_eq!(mars.cloud_regime, "thin water ice");
    }

    #[test]
    fn errors_say_what_to_change() {
        let e = resolve_physics(&presets(), Some("nowhere"), &Overrides::default()).unwrap_err().to_string();
        assert!(e.contains("unknown preset 'nowhere'") && e.contains("earth, mars"), "{e}");
        let e = resolve_physics(&presets(), Some("eart"), &Overrides::default()).unwrap_err().to_string();
        assert!(e.contains("Did you mean 'earth'?"), "{e}");
        let e = resolve_physics(&presets(), Some("earth"), &Overrides { radius_km: Some(10.0), ..Default::default() })
            .unwrap_err()
            .to_string();
        assert_eq!(e, "radius_km: 10 is outside the allowed range [500, 30000] (km). Choose a value inside that range.");
        let e = resolve_physics(&presets(), None, &Overrides { radius_km: Some(6000.0), ..Default::default() }).unwrap_err().to_string();
        assert!(e.starts_with("physics: missing rotation_period_days, gravity_mps2, scale_height_km."), "{e}");
    }

    #[test]
    fn compute_modes_are_checked_against_the_capabilities() {
        let caps = serde_json::json!({"modes": [
            {"key": "auto", "available": true, "reason": null},
            {"key": "cpu", "available": true, "reason": null},
            {"key": "gpu", "available": false, "reason": "No GPU here."},
            {"key": "hybrid", "available": false, "reason": "No GPU here either."}]});
        assert_eq!(check_compute(&caps, "cpu", "compute").unwrap(), "cpu");
        let e = check_compute(&caps, "gpu", "compute").unwrap_err().to_string();
        assert!(e.starts_with("compute: 'gpu' is not available: No GPU here. On this engine use 'auto'"), "{e}");
        let e = check_compute(&caps, "hybird", "compute").unwrap_err().to_string();
        assert!(e.contains("expected one of 'auto', 'cpu', 'gpu', 'hybrid'") && e.contains("Did you mean 'hybrid'?"), "{e}");
    }
}
