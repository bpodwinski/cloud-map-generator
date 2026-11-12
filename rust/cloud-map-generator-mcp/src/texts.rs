//! The sentences a model reads: the short text that goes with every structured result, and the shaping of the
//! metric rows and the zonal profile into their output form. A port of the helpers of `cloud_map_generator_mcp/server.py`,
//! word for word.

use serde_json::{json, Value};

use crate::backend::{MetricRow, ZonalProfile};
use crate::pyfmt::{fixed, fmt_g, py_bool, round_to, signed};

/// Below this width only the mean, the zonal profile and the histogram compare with Earth.
pub const STRUCTURE_WIDTH: u32 = 2048;
/// The rows that need [`STRUCTURE_WIDTH`].
pub const STRUCTURE_KEYS: [&str; 4] = ["coherence", "spectral_slope", "big_object_share", "median_object_km2"];
/// Defect checks of the map, not Earth statistics.
pub const CHECK_KEYS: [&str; 3] = ["seam_ratio", "pole_std_north", "pole_std_south"];

fn opt_num(x: Option<f64>) -> Value {
    match x.filter(|v| v.is_finite()) {
        Some(v) => json!(round_to(v, 4)),
        None => Value::Null,
    }
}

/// The engine's metric rows in their output shape (`key, label, generated, earth, relative_gap, comparable, hint`).
/// `comparable` is false for a structure row measured below [`STRUCTURE_WIDTH`] (null when the width is unknown).
pub fn metric_rows(rows: &[MetricRow], width: Option<u32>) -> Vec<Value> {
    rows.iter()
        .map(|r| {
            let comparable = match width {
                Some(w) => json!(!(STRUCTURE_KEYS.contains(&r.key.as_str()) && w < STRUCTURE_WIDTH)),
                None => Value::Null,
            };
            json!({
                "key": r.key,
                "label": if r.label.is_empty() { r.key.clone() } else { r.label.clone() },
                "generated": opt_num(r.generated),
                "earth": opt_num(r.earth),
                "relative_gap": opt_num(r.relative_gap),
                "comparable": comparable,
                "hint": if r.hint.is_empty() { Value::Null } else { json!(r.hint) },
            })
        })
        .collect()
}

/// `np.interp` at one point: linear between the neighbours, clamped to the end values; `None` when a value
/// involved is missing.
fn interp(x: f64, xs: &[f64], ys: &[Option<f64>]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[xs.len() - 1] {
        return ys[xs.len() - 1];
    }
    let i = xs.partition_point(|&v| v <= x) - 1;
    let (x0, x1) = (xs[i], xs[i + 1]);
    let (y0, y1) = (ys[i]?, ys[i + 1]?);
    Some(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
}

/// A zonal profile (latitude, generated, Earth, target) sampled every `step` degrees, north first.
pub fn coarse_zonal(zonal: &ZonalProfile, step: f64) -> Vec<Value> {
    if zonal.lat_deg.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..zonal.lat_deg.len()).collect();
    order.sort_by(|&a, &b| zonal.lat_deg[a].total_cmp(&zonal.lat_deg[b]));
    let lat: Vec<f64> = order.iter().map(|&i| zonal.lat_deg[i]).collect();
    let column = |v: &[Option<f64>]| -> Option<Vec<Option<f64>>> {
        (v.len() == lat.len()).then(|| order.iter().map(|&i| v[i].filter(|x| x.is_finite())).collect())
    };
    let generated = column(&zonal.generated);
    let earth = column(&zonal.earth);
    let target = zonal.target.as_deref().and_then(column);
    let (lo, hi) = (lat[0], lat[lat.len() - 1]);
    let mut rows = Vec::new();
    let mut w = 90.0;
    while w >= -90.0 {
        if !(w < lo - step / 2.0 || w > hi + step / 2.0) {
            let at = |c: &Option<Vec<Option<f64>>>| match c {
                Some(v) => interp(w, &lat, v).map_or(Value::Null, |x| json!(round_to(x, 4))),
                None => Value::Null,
            };
            rows.push(json!({"lat_deg": w, "generated": at(&generated), "earth": at(&earth), "target": at(&target)}));
        }
        w -= step;
    }
    rows
}

/// `_f`: a number with a fixed number of decimals and a unit, or `n/a`.
pub fn f(x: Option<f64>, nd: usize, unit: &str) -> String {
    match x {
        None => "n/a".to_string(),
        Some(v) => format!("{}{unit}", fixed(v, nd)),
    }
}

fn num(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn text<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

/// The three Earth statistics furthest from Earth (the seam and pole checks are defect checks, not statistics).
pub fn largest_gaps(rows: &[Value], n: usize) -> String {
    let mut scored: Vec<(f64, &Value)> = rows
        .iter()
        .filter_map(|r| {
            let gap = num(r, "relative_gap")?;
            let key = text(r, "key");
            (!CHECK_KEYS.contains(&key) && r.get("comparable").and_then(Value::as_bool) != Some(false)).then_some((gap.abs(), r))
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored
        .iter()
        .take(n)
        .map(|(_, r)| format!("{} {}", text(r, "label"), signed(num(r, "relative_gap").unwrap_or(0.0), 2)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// One sentence on how the map differs from Earth, naming only the metrics that are meaningful at this width.
pub fn headline(rows: &[Value], width: u32) -> String {
    let by = |k: &str| rows.iter().find(|r| text(r, "key") == k);
    if width >= STRUCTURE_WIDTH {
        let gaps = largest_gaps(rows, 3);
        if gaps.is_empty() {
            return String::new();
        }
        let every = rows
            .iter()
            .filter(|r| num(r, "relative_gap").is_some() && !CHECK_KEYS.contains(&text(r, "key")))
            .map(|r| {
                format!(
                    "{} {} vs {} ({})",
                    text(r, "label"),
                    f(num(r, "generated"), 3, ""),
                    f(num(r, "earth"), 3, ""),
                    signed(num(r, "relative_gap").unwrap_or(0.0), 2)
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let dist = ["zonal_rmse", "hist_l1"]
            .iter()
            .filter_map(|k| by(k).map(|r| format!("{} {}", text(r, "label"), f(num(r, "generated"), 3, ""))))
            .collect::<Vec<_>>()
            .join(", ");
        let mut out = format!("Largest gaps to Earth (relative): {gaps}. All statistics, generated vs Earth (relative gap): {every}.");
        if !dist.is_empty() {
            out.push_str(&format!(" Distances to Earth (0 = identical): {dist}."));
        }
        return out;
    }
    let mut parts = Vec::new();
    if let Some(r) = by("mean") {
        parts.push(format!("mean cloud fraction {} (Earth {})", f(num(r, "generated"), 3, ""), f(num(r, "earth"), 3, "")));
    }
    if let Some(r) = by("zonal_rmse") {
        parts.push(format!("zonal-mean RMSE {}", f(num(r, "generated"), 3, "")));
    }
    if let Some(r) = by("hist_l1") {
        parts.push(format!("histogram distance {}", f(num(r, "generated"), 2, "")));
    }
    if parts.is_empty() {
        return String::new();
    }
    format!(
        "Reliable at this width: {}. Spectrum, object sizes and coherence only compare with Earth from width {STRUCTURE_WIDTH}: for a full comparison call start_generation with the same arguments and width={STRUCTURE_WIDTH}.",
        parts.join(", ")
    )
}

/// One sentence on what ran a map: the mode used, the mode asked for, each device's seconds and share of the
/// work, and the engine's note about an automatic choice.
pub fn compute_line(compute: &Value) -> Option<String> {
    if !compute.is_object() {
        return None;
    }
    let parts: Vec<String> = compute
        .get("devices")
        .and_then(Value::as_array)
        .map(|devs| {
            devs.iter()
                .map(|d| {
                    let seconds = num(d, "seconds").map(|s| format!(" {} s", fixed(s, 1))).unwrap_or_default();
                    format!("{}{seconds} {}%", text(d, "kind"), fixed(100.0 * num(d, "share").unwrap_or(0.0), 0))
                })
                .collect()
        })
        .unwrap_or_default();
    let mut line = format!("Compute: {} (requested {})", text(compute, "used"), text(compute, "requested"));
    if !parts.is_empty() {
        line.push_str(": ");
        line.push_str(&parts.join(", "));
    }
    line.push('.');
    let note = text(compute, "note");
    if !note.is_empty() {
        line.push(' ');
        line.push_str(note);
    }
    Some(line)
}

/// The compute modes an engine offers, in one sentence: which are available and why the others are not.
pub fn compute_summary(caps: &Value) -> String {
    let modes = caps.get("modes").and_then(Value::as_array).cloned().unwrap_or_default();
    let ok: Vec<&str> = modes.iter().filter(|m| m.get("available").and_then(Value::as_bool) == Some(true)).map(|m| text(m, "key")).collect();
    let refused: Vec<String> = modes
        .iter()
        .filter(|m| m.get("available").and_then(Value::as_bool) != Some(true))
        .map(|m| {
            let reason = text(m, "reason");
            format!("{} ({})", text(m, "key"), if reason.is_empty() { "not available" } else { reason })
        })
        .collect();
    let mut out = format!(
        "Compute modes (engine {}, argument compute=): available {}",
        text(caps, "engine"),
        if ok.is_empty() { "none".to_string() } else { ok.join(", ") }
    );
    if !refused.is_empty() {
        out.push_str(&format!("; not available: {}", refused.join("; ")));
    }
    out.push('.');
    out
}

fn derived_str(d: &Value, key: &str) -> Option<String> {
    d.get(key).and_then(Value::as_str).map(str::to_string)
}

/// The text of a generation result: the first line, the compute line, the headline, the notes, the files and
/// the next step.
pub fn generation_text(s: &Value) -> String {
    let d = &s["derived"];
    let width = s["width"].as_u64().unwrap_or(0) as u32;
    let mut facts = vec![format!("Hadley edge {}", f(num(d, "hadley_edge_deg"), 1, " deg"))];
    if let Some(v) = num(d, "storm_track_deg") {
        facts.push(format!("storm track {}", f(Some(v), 1, " deg")));
    }
    facts.push(match d.get("cyclones_per_hemisphere").and_then(Value::as_i64) {
        Some(n) => format!("{n} cyclones per hemisphere"),
        None => "no storm-track cyclones".to_string(),
    });
    if let Some(north) = derived_str(d, "cyclone_spin_north") {
        facts.push(format!(
            "lows turn {north} in the north, {} in the south",
            derived_str(d, "cyclone_spin_south").unwrap_or_default()
        ));
    }
    let mut lines = vec![format!(
        "Generated a {}x{} cloud-coverage map in {} s (id {}, seed {}). {}.",
        s["width"].as_u64().unwrap_or(0),
        s["height"].as_u64().unwrap_or(0),
        fixed(num(s, "seconds").unwrap_or(0.0), 1),
        text(s, "id"),
        s["seed"].as_u64().unwrap_or(0),
        facts.join(", ")
    )];
    if let Some(used) = s.get("compute").and_then(compute_line) {
        lines.push(used);
    }
    let rows = s["metrics"].as_array().cloned().unwrap_or_default();
    let head = headline(&rows, width);
    if !head.is_empty() {
        lines.push(head);
    }
    let mut notes: Vec<String> = Vec::new();
    for key in [&s["warnings"], &d["warnings"]] {
        for w in key.as_array().into_iter().flatten().filter_map(Value::as_str) {
            if !notes.iter().any(|n| n == w) {
                notes.push(w.to_string());
            }
        }
    }
    if !notes.is_empty() {
        let shown: Vec<String> = notes
            .iter()
            .take(3)
            .map(|n| if n.chars().count() <= 220 { n.clone() } else { format!("{}...", n.chars().take(217).collect::<String>()) })
            .collect();
        let mut line = format!("Notes: {}", shown.join(" | "));
        if notes.len() > 3 {
            line.push_str(&format!(" (+{} more in the structured result)", notes.len() - 3));
        }
        lines.push(line);
    }
    if s["saved"].as_bool() == Some(true) {
        lines.push(format!("Files: {}", text(s, "folder")));
    } else {
        lines.push("Not saved (save=false).".to_string());
    }
    lines.push(format!("Next: {}", text(s, "next")));
    lines.join("\n")
}

/// The `next` hint of a generation result.
pub fn next_hint(id: &str) -> String {
    format!(
        "view_map(id='{id}', view='globe') to look at it from other angles; adjust the physics overrides and call generate_map again; start_generation for a larger width."
    )
}

/// `Hadley edge ... bakeable True`: the text of `derive_physics`.
pub fn derive_text(out: &Value) -> String {
    let ph = &out["physics"];
    let d = &out["derived"];
    let g = |k: &str| fmt_g(ph[k].as_f64().unwrap_or(0.0));
    let mut lines = vec![format!(
        "Physics: rotation {} d, radius {} km, gravity {} m/s2, scale height {} km, clouds {}.",
        g("rotation_period_days"),
        g("radius_km"),
        g("gravity_mps2"),
        g("scale_height_km"),
        text(ph, "cloud_regime")
    )];
    let cyc = d.get("cyclones_per_hemisphere").and_then(Value::as_i64).map_or("None".to_string(), |n| n.to_string());
    let mut second = format!(
        "Hadley edge {}, storm track {}, eddies {}, deformation radius {}, cyclones per hemisphere {}",
        f(num(d, "hadley_edge_deg"), 1, " deg"),
        f(num(d, "storm_track_deg"), 1, " deg"),
        py_bool(d.get("has_eddies").and_then(Value::as_bool)),
        f(num(d, "deformation_radius_km"), 0, " km"),
        cyc
    );
    if let Some(north) = derived_str(d, "cyclone_spin_north") {
        second.push_str(&format!(
            ", turning {north} in the north and {} in the south (seen from above each pole)",
            derived_str(d, "cyclone_spin_south").unwrap_or_default()
        ));
    }
    second.push_str(&format!(", bakeable {}.", py_bool(d.get("bakeable").and_then(Value::as_bool))));
    lines.push(second);
    let warnings: Vec<&str> = d["warnings"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
    if !warnings.is_empty() {
        lines.push(format!("Warnings: {}", warnings.join("; ")));
    }
    lines.push("Next: generate_map at width 512.".to_string());
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(key: &str, label: &str, g: f64, e: f64, gap: Option<f64>) -> Value {
        json!({"key": key, "label": label, "generated": g, "earth": e, "relative_gap": gap, "comparable": null, "hint": "h"})
    }

    #[test]
    fn the_headline_names_only_the_metrics_that_mean_something_at_the_width() {
        let rows = vec![
            row("mean", "Mean", 0.3, 0.4, Some(-0.25)),
            row("zonal_rmse", "Zonal", 0.1, 0.0, None),
            row("hist_l1", "Hist", 0.2, 0.0, None),
            row("median_object_km2", "Median object", 9000.0, 300.0, Some(29.0)),
        ];
        let narrow = headline(&rows, 512);
        assert!(narrow.contains("mean cloud fraction 0.300 (Earth 0.400)") && !narrow.contains("Median object"), "{narrow}");
        let wide = headline(&rows, 2048);
        assert!(wide.starts_with("Largest gaps"), "{wide}");
        assert!(wide.find("Median object") < wide.find("Mean"));
        assert_eq!(headline(&[], 512), "");
        assert_eq!(headline(&[], 4096), "");
        assert!(narrow.contains("start_generation") && narrow.contains("width=2048"));
        assert!(wide.contains("Mean 0.300 vs 0.400 (-0.25)"), "{wide}");
    }

    #[test]
    fn the_seam_and_pole_checks_are_not_headline_gaps_and_structure_rows_say_when_they_compare() {
        let rows = vec![
            row("mean", "Mean", 0.3, 0.4, Some(-0.25)),
            row("pole_std_south", "Pole spread, south", 0.12, 0.03, Some(3.4)),
            row("seam_ratio", "Seam ratio", 0.8, 1.4, Some(-0.42)),
        ];
        let wide = headline(&rows, 2048);
        assert!(wide.starts_with("Largest gaps to Earth (relative): Mean -0.25.") && !wide.contains("Pole spread"), "{wide}");
        let raw = vec![
            MetricRow { key: "coherence".into(), label: "C".into(), generated: Some(0.3), earth: Some(0.5), relative_gap: Some(-0.4), hint: "h".into() },
            MetricRow { key: "mean".into(), label: "M".into(), generated: Some(0.3), earth: Some(0.4), relative_gap: Some(-0.25), hint: "h".into() },
        ];
        let narrow: Vec<(String, Value)> = metric_rows(&raw, Some(512)).iter().map(|r| (text(r, "key").to_string(), r["comparable"].clone())).collect();
        assert_eq!(narrow, vec![("coherence".to_string(), json!(false)), ("mean".to_string(), json!(true))]);
        assert!(metric_rows(&raw, Some(2048)).iter().all(|r| r["comparable"] == json!(true)));
        assert!(metric_rows(&raw, None).iter().all(|r| r["comparable"].is_null()));
    }

    #[test]
    fn the_coarse_zonal_profile_samples_every_fifteen_degrees_north_first() {
        let lat: Vec<f64> = (0..36).map(|i| -87.5 + 5.0 * i as f64).collect();
        let z = ZonalProfile {
            lat_deg: lat.clone(),
            generated: lat.iter().map(|l| Some(l / 100.0)).collect(),
            earth: lat.iter().map(|_| Some(0.4)).collect(),
            target: None,
        };
        let rows = coarse_zonal(&z, 15.0);
        let lats: Vec<f64> = rows.iter().map(|r| r["lat_deg"].as_f64().unwrap()).collect();
        assert_eq!(lats, (-6..=6).rev().map(|i| f64::from(i) * 15.0).collect::<Vec<_>>());
        assert_eq!(rows[0]["generated"], json!(0.875), "clamped to the last band");
        assert_eq!(rows[6]["generated"], json!(0.0));
        assert!(rows.iter().all(|r| r["target"].is_null() && r["earth"] == json!(0.4)));
    }

    #[test]
    fn python_style_numbers_in_the_texts() {
        let caps = json!({"engine": "mock", "modes": [
            {"key": "auto", "available": true}, {"key": "gpu", "available": false, "reason": "No GPU."}]});
        assert_eq!(compute_summary(&caps), "Compute modes (engine mock, argument compute=): available auto; not available: gpu (No GPU.).");
        let c = json!({"requested": "auto", "used": "cpu", "devices": [{"kind": "cpu", "name": "x", "seconds": 0.84, "share": 1.0}], "note": null});
        assert_eq!(compute_line(&c).unwrap(), "Compute: cpu (requested auto): cpu 0.8 s 100%.");
        assert_eq!(f(None, 1, " deg"), "n/a");
    }
}
