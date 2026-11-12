//! The physics numbers of the API: the presets, `POST /api/derive`, the warnings of a body, and the refusals that
//! are about the body and not the request's format (`ui/server.py`: `build_presets`, `derive`, `body_warnings`,
//! `check_generatable`, `cyclones_per_hemisphere`).
//!
//! Every number comes from the core: the Hadley edge, the storm track and the deformation radius from
//! `climate`, the cyclone count from `flow::wave_count`.

use crate::json::{num, opt_float, opt_int};
use crate::pyfmt::capitalize;
use crate::limits::{limit, MAX_CYCLONES_PER_HEMISPHERE, PHYSICS_KEYS};
use crate::bodies::{load_bodies, TestBody};
use crate::project::Project;
use crate::pyjson::J;
use cloud_map_generator_core::climate::{deformation_radius_rad, hadley_edge_deg, hadley_edge_raw_deg, has_baroclinic_eddies, is_bakeable, storm_track_deg, BodyInputs, EARTH};
use cloud_map_generator_core::flow::wave_count;
use cloud_map_generator_core::generate::py_round;
use std::f64::consts::PI;

/// More cyclones per hemisphere than this and the answer warns that generation takes noticeably longer.
pub const SLOW_CYCLONES_PER_HEMISPHERE: usize = 14;

// ---------------------------------------------------------------------------------------------------------
// Presets

/// One row of `presets`.
#[derive(Clone, Debug, PartialEq)]
pub struct Preset {
    pub key: String,
    pub label: String,
    pub body: BodyInputs,
    pub experiment_required: bool,
    pub note: String,
}

fn preset(key: &str, label: &str, body: BodyInputs, note: &str) -> Preset {
    Preset { key: key.to_string(), label: label.to_string(), experiment_required: !is_bakeable(&body), body, note: note.to_string() }
}

fn period(b: &BodyInputs) -> f64 {
    b.rotation_period_days.unwrap_or(f64::NAN)
}

fn in_limits(b: &BodyInputs) -> bool {
    let values = [period(b), b.radius_km, b.gravity_mps2, b.scale_height_km];
    PHYSICS_KEYS.iter().zip(values).all(|(key, v)| {
        let (lo, hi) = limit(key);
        lo <= v && v <= hi
    })
}

/// The presets: the bodies of `gen.py` (Earth, Fast rotator, Mars, Titan, Venus), a retrograde Earth, a 4-day
/// slow rotator, a 2-radius super-Earth, then any record of the bodies file (`examples/bodies.json` of the
/// project, when it is found) that is not already there (same key, or the same physics under another key).
/// Water bodies first, the others (`experiment_required`) after, each group in that order; only bodies inside
/// the limits are offered.
pub fn build_presets(project: &Project) -> Vec<Preset> {
    let notes = |key: &str| match key {
        "earth" => "The reference body: the zonal profile is anchored on its Blue Marble cloud fraction. Hadley edge 30 degrees.",
        "fast" => "A made-up Earth-sized world spinning in 9.6 hours: narrower bands, more cyclones.",
        "mars" => "Thin water-ice clouds: not water, so the amount is Earth-anchored and needs the experiment option.",
        "titan" => "Methane clouds and a slow spin: not water, so it needs the experiment option.",
        "venus" => "Sulfuric acid deck and a retrograde 243-day spin: not water, so it needs the experiment option.",
        _ => "",
    };
    let label = |key: &str| match key {
        "earth" => "Earth".to_string(),
        "fast" => "Fast rotator".to_string(),
        "mars" => "Mars".to_string(),
        "titan" => "Titan".to_string(),
        "venus" => "Venus".to_string(),
        other => capitalize(other),
    };
    // gen.py's BODIES, in its dict order.
    let mut presets: Vec<Preset> = [TestBody::Earth, TestBody::Mars, TestBody::Titan, TestBody::Venus, TestBody::Fast]
        .into_iter()
        .map(|b| preset(b.name(), &label(b.name()), b.inputs(), notes(b.name())))
        .collect();
    let earth_period = EARTH.rotation_period_days;
    presets.push(preset(
        "earth-retrograde",
        "Retrograde Earth",
        BodyInputs::new(earth_period.map(|p| -p), EARTH.radius_km, EARTH.gravity_mps2, EARTH.scale_height_km),
        "Earth spun the other way: same Hadley edge, lows turn the opposite way in each hemisphere.",
    ));
    presets.push(preset(
        "slow-rotator",
        "Slow rotator (4 days)",
        BodyInputs::new(Some(4.0), EARTH.radius_km, EARTH.gravity_mps2, EARTH.scale_height_km),
        "Earth with a 4-day day: the Hadley cell reaches the pole, so no storm track and no cyclones.",
    ));
    presets.push(preset(
        "super-earth",
        "Super-Earth (2 radii)",
        BodyInputs::new(earth_period, 2.0 * EARTH.radius_km, 2.0 * EARTH.gravity_mps2, EARTH.scale_height_km / 2.0),
        "Twice Earth's radius at Earth's density: gravity doubles and, for the same air, the scale height halves. Hadley edge 15 degrees.",
    ));
    // The bodies file adds what is not already there.
    if let Ok(file) = load_bodies(project, None) {
        let signature = |b: &BodyInputs| (period(b).to_bits(), b.radius_km.to_bits(), b.gravity_mps2.to_bits(), b.scale_height_km.to_bits(), b.cloud_regime.to_string());
        let mut seen: Vec<_> = presets.iter().map(|p| signature(&p.body)).collect();
        for rec in file.records {
            // A record with no spin cannot be a preset (the API's period is a number).
            if rec.body.rotation_period_days.is_none() {
                continue;
            }
            let sig = signature(&rec.body);
            if presets.iter().any(|p| p.key == rec.key) || seen.contains(&sig) {
                continue;
            }
            seen.push(sig);
            let label = capitalize(&rec.key.replace(['-', '_'], " "));
            presets.push(Preset {
                key: rec.key,
                label,
                experiment_required: !is_bakeable(&rec.body),
                body: rec.body,
                note: rec.surface_note.unwrap_or_default(),
            });
        }
    }
    let (water, other): (Vec<Preset>, Vec<Preset>) = presets.into_iter().partition(|p| !p.experiment_required);
    water.into_iter().chain(other).filter(|p| in_limits(&p.body)).collect()
}


/// The presets of the MCP service (`_PRESETS` of `cloudgen/service.py`): Earth, a retrograde Earth, a fast and a
/// slow rotator, a super-Earth, Mars, Titan and Venus, with the keys and notes the tools document. Not the
/// bodies file: an AI client needs the same list on every machine.
pub fn service_presets() -> Vec<Preset> {
    let earth_period = EARTH.rotation_period_days;
    let earth = |period: Option<f64>| BodyInputs::new(period, EARTH.radius_km, EARTH.gravity_mps2, EARTH.scale_height_km);
    let mk = |key: &str, label: &str, body: BodyInputs, note: &str| preset(key, label, body, note);
    vec![
        mk(
            "earth",
            "Earth",
            earth(earth_period),
            "The reference body: the zonal profile is anchored on Earth's measured cloud fraction. Hadley edge 30 degrees, storm track 51, five lows per hemisphere.",
        ),
        mk(
            "earth-retrograde",
            "Retrograde Earth",
            earth(earth_period.map(|p| -p)),
            "Earth spun the other way: same bands, and the lows turn the opposite way in each hemisphere.",
        ),
        mk(
            "fast-rotator",
            "Fast rotator (0.4 day)",
            earth(Some(0.4)),
            "A made-up Earth-sized world that spins in 9.6 hours: narrower bands, a Hadley edge near 12 degrees, more but smaller cyclones.",
        ),
        mk(
            "slow-rotator",
            "Slow rotator (4 days)",
            earth(Some(4.0)),
            "Earth with a 4-day day: the Hadley cell reaches the pole, so there is no storm track, no cyclones and no wind field.",
        ),
        mk(
            "super-earth",
            "Super-Earth (2 radii)",
            BodyInputs::new(earth_period, 2.0 * EARTH.radius_km, 2.0 * EARTH.gravity_mps2, EARTH.scale_height_km / 2.0),
            "Twice Earth's radius at Earth's density: gravity doubles and, for the same air, the scale height halves. Hadley edge 15 degrees.",
        ),
        mk(
            "mars",
            "Mars",
            BodyInputs::new(Some(1.02596), 3389.5, 3.721, 11.1).with_regime("thin water ice"),
            "Thin water-ice clouds: not water, so the cloud amount (Earth's) is wrong for it and it needs the experiment option.",
        ),
        mk(
            "titan",
            "Titan",
            BodyInputs::new(Some(15.945), 2574.7, 1.352, 20.6).with_regime("methane"),
            "Methane clouds and a 16-day spin: not water, so it needs the experiment option. A global Hadley cell, no storm track.",
        ),
        mk(
            "venus",
            "Venus",
            BodyInputs::new(Some(-243.02), 6051.8, 8.87, 15.7).with_regime("sulfuric acid"),
            "A sulfuric-acid deck and a retrograde 243-day spin: not water, so it needs the experiment option. A global Hadley cell; its real super-rotation is outside the model.",
        ),
    ]
}

pub fn preset_json(p: &Preset) -> J {
    J::obj([
        ("key", J::s(p.key.as_str())),
        ("label", J::s(p.label.as_str())),
        ("rotation_period_days", J::Float(period(&p.body))),
        ("radius_km", J::Float(p.body.radius_km)),
        ("gravity_mps2", J::Float(p.body.gravity_mps2)),
        ("scale_height_km", J::Float(p.body.scale_height_km)),
        ("cloud_regime", J::s(p.body.cloud_regime.as_ref())),
        ("experiment_required", J::Bool(p.experiment_required)),
        ("note", J::s(p.note.as_str())),
    ])
}

/// The ranges of `limits`, as the API prints them (integers where the bound is one).
pub fn limits_json() -> J {
    J::obj(crate::limits::LIMITS.iter().map(|&(k, lo, hi)| (k, J::Arr(vec![num(lo), num(hi)]))))
}

// ---------------------------------------------------------------------------------------------------------
// Derived numbers

/// The lows per hemisphere from the closed form `flow::build_flow` uses (the storm-track circle over five
/// deformation radii), or `None` without a storm track. Cheap at any spin, and capped (an absurd spin gives
/// `10^9`) so that nothing downstream builds thousands of vortices.
fn cyclone_estimate(body: &BodyInputs) -> Option<usize> {
    if !has_baroclinic_eddies(body) {
        return None;
    }
    let lr = deformation_radius_rad(body);
    let track = storm_track_deg(body).to_radians();
    let n = if lr > 0.0 { 2.0 * PI * track.cos() / (5.0 * lr) } else { f64::INFINITY };
    Some(if !n.is_finite() || n > 1e9 { 1_000_000_000 } else { (n.round_ties_even().max(2.0)) as usize })
}

/// Mid-latitude lows per hemisphere, or `None` without a storm track: `flow::wave_count` (what `build_flow`
/// places), except that a spin already past the limit answers with the closed-form estimate, so that an absurd
/// one costs nothing.
pub fn cyclones_per_hemisphere(body: &BodyInputs) -> Option<usize> {
    let estimate = cyclone_estimate(body)?;
    if estimate > MAX_CYCLONES_PER_HEMISPHERE {
        return Some(estimate);
    }
    // At most the limit: the wavenumber is finite and the count is the flow's own.
    Some(wave_count(body).expect("a body with a storm track has a wave count"))
}

/// The size of one texel of a `width`-px map, km.
pub fn texel_km(radius_km: f64, width: usize) -> f64 {
    2.0 * PI * radius_km / width as f64
}

/// The plain sentences a page shows next to a body (`body_warnings`).
pub fn body_warnings(body: &BodyInputs, cyclones: Option<usize>) -> Vec<String> {
    let mut out = Vec::new();
    let raw = hadley_edge_raw_deg(body);
    if !is_bakeable(body) {
        out.push(format!(
            "Clouds of {}, not water: this generator's cloud amount is Earth's, so it refuses unless the experiment option is on, and the map is never a bake.",
            body.cloud_regime
        ));
    }
    let spin = period(body);
    if spin == 0.0 {
        out.push("A zero rotation period means no spin: one global Hadley cell, no storm track.".to_string());
    } else if spin < 0.0 {
        out.push("Retrograde spin: lows turn the opposite way in each hemisphere and the zonal winds flip sign.".to_string());
    }
    if raw.is_finite() && raw < 10.0 {
        out.push(format!("Hadley edge clamped to 10 degrees (raw {raw:.1}): the bands saturate."));
    }
    if !has_baroclinic_eddies(body) {
        out.push("No storm track: the cloud stays zonal, with no cyclones and no wind field (a global Hadley cell, like Venus or Titan).".to_string());
    }
    match cyclones {
        Some(n) if n > MAX_CYCLONES_PER_HEMISPHERE => out.push(format!(
            "{n} cyclones per hemisphere: beyond the {MAX_CYCLONES_PER_HEMISPHERE} this tool generates, a spin this fast would take minutes per map."
        )),
        Some(n) if n > SLOW_CYCLONES_PER_HEMISPHERE => out.push(format!("{n} cyclones per hemisphere: generation takes noticeably longer than the estimate.")),
        _ => {}
    }
    out
}

/// What a body implies, rounded as the API and the MCP tools report it (`derive` of `ui/server.py` and of
/// `cloudgen/service.py`): the numbers only, the sentences are each front end's own.
#[derive(Clone, Debug, PartialEq)]
pub struct Derived {
    pub hadley_edge_deg: f64,
    pub storm_track_deg: f64,
    pub has_eddies: bool,
    pub deformation_radius_km: Option<f64>,
    pub cyclones_per_hemisphere: Option<usize>,
    /// The kilometres per texel at the equator of each width asked for.
    pub texel_km: Vec<(usize, f64)>,
    pub bakeable: bool,
    /// The spin is prograde (a positive rotation period): lows turn counterclockwise in the north.
    pub prograde: bool,
}

/// The physics numbers of a body, with no generation.
pub fn derived(body: &BodyInputs, widths: &[usize]) -> Derived {
    let lr = deformation_radius_rad(body);
    Derived {
        hadley_edge_deg: py_round(hadley_edge_deg(body), 2),
        storm_track_deg: py_round(storm_track_deg(body), 2),
        has_eddies: has_baroclinic_eddies(body),
        deformation_radius_km: lr.is_finite().then(|| py_round(lr * body.radius_km, 1)),
        cyclones_per_hemisphere: cyclones_per_hemisphere(body),
        texel_km: widths.iter().map(|&w| (w, py_round(texel_km(body.radius_km, w), 1))).collect(),
        bakeable: is_bakeable(body),
        prograde: period(body) > 0.0,
    }
}

/// `POST /api/derive`: the physics numbers of a body, with no generation.
pub fn derive(body: &BodyInputs, widths: &[usize]) -> J {
    let d = derived(body, widths);
    J::obj([
        ("hadley_edge_deg", J::Float(d.hadley_edge_deg)),
        ("storm_track_deg", J::Float(d.storm_track_deg)),
        ("has_eddies", J::Bool(d.has_eddies)),
        ("deformation_radius_km", opt_float(d.deformation_radius_km)),
        ("cyclones_per_hemisphere", opt_int(d.cyclones_per_hemisphere.map(|n| i64::try_from(n).unwrap_or(i64::MAX)))),
        ("texel_km", J::obj(d.texel_km.iter().map(|&(w, km)| (w.to_string(), J::Float(km))))),
        ("bakeable", J::Bool(d.bakeable)),
        ("warnings", J::Arr(body_warnings(body, d.cyclones_per_hemisphere).into_iter().map(J::Str).collect())),
    ])
}

/// `check_generatable`: the refusals that are about the body, not the format of the request. The message is the
/// API's (HTTP 400).
pub fn check_generatable(body: &BodyInputs, experiment: bool) -> Result<(), String> {
    if !is_bakeable(body) && !experiment {
        return Err(format!(
            "clouds of {}, not water: this generator's cloud amount is Earth's (climate.is_bakeable); turn the experiment option on to generate anyway",
            body.cloud_regime
        ));
    }
    if let Some(n) = cyclones_per_hemisphere(body) {
        if n > MAX_CYCLONES_PER_HEMISPHERE {
            return Err(format!(
                "rotation too fast for this tool: {n} cyclones per hemisphere (limit {MAX_CYCLONES_PER_HEMISPHERE}); lengthen the rotation period"
            ));
        }
    }
    Ok(())
}

/// `x` rounded up to `digits` significant digits (`_round_up` of `cloudgen/service.py`).
fn round_up(x: f64, digits: i32) -> f64 {
    let scale = 10f64.powi(x.log10().floor() as i32 - digits + 1);
    (x / scale).ceil() * scale
}

/// The shortest |rotation period| (days), to two digits, that keeps the cyclone count within the limit, or `None`
/// when even the longest spin the limits allow does not (`_min_period_for_limit`). The count it bisects on is the
/// closed form, which costs nothing at any spin.
pub fn min_period_for_limit(body: &BodyInputs) -> Option<f64> {
    let fine = |period: f64| {
        let probe = BodyInputs { rotation_period_days: Some(period), ..body.clone() };
        cyclone_estimate(&probe).is_none_or(|n| n <= MAX_CYCLONES_PER_HEMISPHERE)
    };
    let spin = period(body);
    let (mut lo, mut hi) = (if spin == 0.0 || spin.is_nan() { 1.0 } else { spin.abs() }, 400.0);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::compact;
    use cloud_map_generator_core::flow::{build_flow, VortexKind};
    use serde_json::Value;

    fn project() -> Project {
        Project::discover(Some(std::path::PathBuf::from("no/such/root")))
    }

    fn body(period: f64, radius: f64) -> BodyInputs {
        BodyInputs::new(Some(period), radius, 9.81, 8.0)
    }

    fn json(j: &J) -> Value {
        serde_json::from_str(&compact(j)).unwrap()
    }

    #[test]
    fn the_presets_are_the_python_servers_in_its_order() {
        let p = build_presets(&project());
        let keys: Vec<&str> = p.iter().map(|p| p.key.as_str()).collect();
        assert_eq!(keys, ["earth", "fast", "earth-retrograde", "slow-rotator", "super-earth", "mars", "titan", "venus"]);
        let labels: Vec<&str> = p.iter().map(|p| p.label.as_str()).collect();
        assert_eq!(labels, ["Earth", "Fast rotator", "Retrograde Earth", "Slow rotator (4 days)", "Super-Earth (2 radii)", "Mars", "Titan", "Venus"]);
        let flags: Vec<bool> = p.iter().map(|p| p.experiment_required).collect();
        assert_eq!(flags, [false, false, false, false, false, true, true, true]);
        let earth = &p[0].body;
        assert_eq!((earth.rotation_period_days, earth.radius_km, earth.gravity_mps2, earth.scale_height_km), (Some(0.99727), 6371.0, 9.81, 8.0));
        let sup = &p[4].body;
        assert_eq!((sup.radius_km, sup.gravity_mps2, sup.scale_height_km), (12742.0, 19.62, 4.0));
        assert_eq!(p[2].body.rotation_period_days, Some(-0.99727));
        assert_eq!(p[7].body.cloud_regime, "sulfuric acid");
        assert!(p.iter().all(|p| in_limits(&p.body)));
        let j = json(&preset_json(&p[0]));
        let mut keys: Vec<&str> = j.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["cloud_regime", "experiment_required", "gravity_mps2", "key", "label", "note", "radius_km", "rotation_period_days", "scale_height_km"]);
    }

    #[test]
    fn a_bodies_file_adds_only_what_is_new() {
        let dir = std::env::temp_dir().join(format!("cloudgen-serve-presets-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("examples")).unwrap();
        std::fs::write(
            dir.join("examples").join("bodies.json"),
            r#"{"bodies": [
                {"key": "earth", "rotationPeriodDays": 0.99727, "radiusKm": 6371, "gravityMps2": 9.81, "scaleHeightKm": 8.4, "cloudRegime": "water"},
                {"key": "fast-rotator", "rotationPeriodDays": 0.4, "radiusKm": 6371, "gravityMps2": 9.81, "scaleHeightKm": 8.0, "cloudRegime": "water"},
                {"key": "ocean_world", "rotationPeriodDays": 2.5, "radiusKm": 7000, "gravityMps2": 12, "scaleHeightKm": 9, "cloudRegime": "water", "surfaceNote": "all sea"},
                {"key": "spinless", "rotationPeriodDays": null, "radiusKm": 1000, "gravityMps2": 1, "scaleHeightKm": 5},
                {"key": "huge", "rotationPeriodDays": 1, "radiusKm": 90000, "gravityMps2": 1, "scaleHeightKm": 5, "cloudRegime": "water"}
            ]}"#,
        )
        .unwrap();
        let p = build_presets(&Project::discover(Some(dir)));
        let keys: Vec<&str> = p.iter().map(|p| p.key.as_str()).collect();
        assert_eq!(keys.iter().filter(|k| **k == "earth").count(), 1);
        assert!(!keys.contains(&"fast-rotator") && !keys.contains(&"spinless") && !keys.contains(&"huge"), "{keys:?}");
        let extra = p.iter().find(|p| p.key == "ocean_world").unwrap();
        assert_eq!((extra.label.as_str(), extra.note.as_str(), extra.experiment_required), ("Ocean world", "all sea", false));
        assert!(keys.iter().position(|k| *k == "ocean_world").unwrap() < keys.iter().position(|k| *k == "mars").unwrap(), "water bodies come first");
    }

    #[test]
    fn derive_earth() {
        let d = json(&derive(&body(0.99727, 6371.0), &[256, 512]));
        assert_eq!(d["hadley_edge_deg"], 30.0);
        assert_eq!(d["storm_track_deg"], 51.0);
        assert_eq!(d["has_eddies"], true);
        assert!((d["deformation_radius_km"].as_f64().unwrap() - 1000.0).abs() < 5.0);
        assert_eq!(d["cyclones_per_hemisphere"], 5);
        assert!((d["texel_km"]["512"].as_f64().unwrap() - 2.0 * PI * 6371.0 / 512.0).abs() < 0.1);
        assert_eq!(d["texel_km"].as_object().unwrap().len(), 2);
        assert_eq!(d["bakeable"], true);
        assert_eq!(d["warnings"], serde_json::json!([]));
        let mut keys: Vec<&str> = d.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["bakeable", "cyclones_per_hemisphere", "deformation_radius_km", "hadley_edge_deg", "has_eddies", "storm_track_deg", "texel_km", "warnings"]);
    }

    #[test]
    fn derive_the_special_bodies() {
        let retro = json(&derive(&body(-0.99727, 6371.0), &[512]));
        assert_eq!((retro["hadley_edge_deg"].as_f64().unwrap(), retro["has_eddies"].as_bool().unwrap()), (30.0, true));
        assert!(retro["warnings"][0].as_str().unwrap().contains("Retrograde"));
        let methane = json(&derive(&body(0.99727, 6371.0).with_regime("methane"), &[512]));
        assert_eq!(methane["bakeable"], false);
        let w = methane["warnings"][0].as_str().unwrap();
        assert!(w.contains("methane") && w.contains("experiment"), "{w}");
        let slow = json(&derive(&body(4.0, 6371.0), &[512]));
        assert_eq!((slow["hadley_edge_deg"].as_f64().unwrap(), slow["has_eddies"].as_bool().unwrap()), (90.0, false));
        assert!(slow["cyclones_per_hemisphere"].is_null());
        assert!(slow["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("No storm track")));
        let none = json(&derive(&body(0.0, 6371.0), &[512]));
        assert_eq!((none["hadley_edge_deg"].as_f64().unwrap(), none["has_eddies"].as_bool().unwrap()), (90.0, false));
        assert!(none["deformation_radius_km"].is_null(), "an infinite radius is null");
        assert!(none["warnings"][0].as_str().unwrap().contains("zero rotation period"));
    }

    #[test]
    fn a_very_fast_spin_answers_at_once_and_warns() {
        let t0 = std::time::Instant::now();
        let d = json(&derive(&body(1e-6, 6371.0), &[512]));
        assert!(t0.elapsed().as_secs() < 5);
        assert!(d["cyclones_per_hemisphere"].as_u64().unwrap() > MAX_CYCLONES_PER_HEMISPHERE as u64);
        assert!(d["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("beyond")));
        // The smallest period the limits allow: the deformation radius underflows to 0, the count saturates, nothing panics.
        let d = derive(&body(f64::from_bits(1), 6371.0), &[512]);
        assert!(compact(&d).contains("\"cyclones_per_hemisphere\":1000000000"));
        let e = check_generatable(&body(f64::from_bits(1), 6371.0), false).unwrap_err();
        assert!(e.starts_with("rotation too fast for this tool: 1000000000 cyclones per hemisphere (limit 32)"), "{}", e);
    }

    #[test]
    fn the_cyclone_count_is_the_flows_own() {
        for (p, r) in [(0.99727, 6371.0), (-0.99727, 6371.0), (0.4, 6371.0), (0.7, 5000.0), (1.0, 12742.0)] {
            let b = body(p, r);
            let flow = build_flow(&b, 123);
            let north = flow.vortices.iter().filter(|v| v.kind == VortexKind::Cyclone && v.lat > 0.0).count();
            assert_eq!(cyclones_per_hemisphere(&b), Some(north), "period {p} radius {r}");
            assert_eq!(cyclone_estimate(&b), Some(north), "the closed form agrees with the flow: {p} {r}");
        }
    }

    #[test]
    fn warnings_say_what_the_python_server_says() {
        let w = body_warnings(&body(0.5, 6371.0), Some(15));
        assert_eq!(w, vec!["15 cyclones per hemisphere: generation takes noticeably longer than the estimate."]);
        let w = body_warnings(&body(0.99727, 6371.0), Some(5));
        assert!(w.is_empty());
        let w = body_warnings(&body(0.02, 6371.0), Some(40));
        assert!(w.iter().any(|w| w.starts_with("40 cyclones per hemisphere: beyond the 32 this tool generates")), "{w:?}");
        // A Hadley edge below 10 degrees is clamped and says so.
        let w = body_warnings(&body(0.1, 6371.0), cyclones_per_hemisphere(&body(0.1, 6371.0)));
        assert!(w.iter().any(|w| w.starts_with("Hadley edge clamped to 10 degrees (raw 3.0)")), "{w:?}");
    }

    #[test]
    fn check_generatable_refuses_non_water_and_too_fast() {
        let mars = body(1.02596, 3389.5).with_regime("thin water ice");
        let e = check_generatable(&mars, false).unwrap_err();
        assert!(e.contains("experiment") && e.contains("thin water ice"));
        assert!(check_generatable(&mars, true).is_ok());
        let e = check_generatable(&body(0.02, 6371.0), false).unwrap_err();
        assert!(e.starts_with("rotation too fast for this tool: "), "{e}");
        assert!(check_generatable(&body(0.99727, 6371.0), false).is_ok());
    }

    #[test]
    fn limits_print_like_python() {
        assert_eq!(
            compact(&limits_json()),
            "{\"seed\":[0,2147483647],\"rotation_period_days\":[-400,400],\"radius_km\":[500,30000],\"gravity_mps2\":[0.1,100],\"scale_height_km\":[0.5,100]}"
        );
    }
}
