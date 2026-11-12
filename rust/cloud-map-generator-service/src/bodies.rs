//! Body records: the physics a bake or a run reads for each body (`cloudgen/bodies.py`, `gen.py`'s BODIES).
//!
//! A bodies file (`examples/bodies.json`) is `{"bodies": [record, ...]}`; a record holds `key`,
//! `rotationPeriodDays`, `radiusKm`, `gravityMps2`, `scaleHeightKm`, `cloudRegime` (only `water` bakes),
//! optional `measured` (true: the body has a real map and the bake skips it) and optional `surface` (land and
//! relief: `width`, `height`, `land`, `elevation`, `source`; the file paths are relative to the bodies file).
//! This is the loading `cloud-map-generator-core` leaves to the caller (`Surface::load` takes a file reader).

use crate::fail::{Context, Fail, Result};
use crate::project::Project;
use cloud_map_generator_core::climate::{BodyInputs, EARTH};
use cloud_map_generator_core::surface::{Surface, SurfaceError, SurfaceRecord};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// One record of a bodies file.
#[derive(Clone, Debug)]
pub struct BodyRecord {
    pub key: String,
    pub body: BodyInputs,
    /// `measured`: the body has a real map and the bake skips it (any truthy value).
    pub measured: bool,
    pub surface: Option<SurfaceRecord>,
    /// `surfaceNote`: why a record has no surface, for the error that asks for one.
    pub surface_note: Option<String>,
}

/// A bodies file read: its records and the folder the surface paths are relative to.
#[derive(Clone, Debug)]
pub struct Bodies {
    pub records: Vec<BodyRecord>,
    pub base: PathBuf,
}

/// Python truthiness of a JSON value (`b.get('measured')`).
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|x| x != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

fn number(rec: &Value, key: &str, field: &str) -> Result<f64> {
    rec.get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| Fail::new(format!("bodies file: record '{key}' needs a number for '{field}'")))
}

fn size(rec: &Value, key: &str, field: &str) -> Result<usize> {
    // `int(record['width'])`: an integer, or a float that holds one.
    let v = rec.get(field).ok_or_else(|| Fail::new(format!("bodies file: the surface of '{key}' lacks '{field}'")))?;
    v.as_u64()
        .map(|n| n as usize)
        .or_else(|| v.as_f64().filter(|x| *x >= 0.0).map(|x| x as usize))
        .ok_or_else(|| Fail::new(format!("bodies file: the surface of '{key}': '{field}' is not a size")))
}

fn text(rec: &Value, key: &str, field: &str) -> Result<String> {
    rec.get(field)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| Fail::new(format!("bodies file: the surface of '{key}' needs a string for '{field}'")))
}

fn parse_surface(key: &str, rec: &Value) -> Result<Option<SurfaceRecord>> {
    // Python: `if not record: return None` (null, {}).
    if !truthy(Some(rec)) {
        return Ok(None);
    }
    Ok(Some(SurfaceRecord {
        width: size(rec, key, "width")?,
        height: size(rec, key, "height")?,
        land: text(rec, key, "land")?,
        elevation: text(rec, key, "elevation")?,
        source: rec.get("source").and_then(Value::as_str).map(str::to_string),
    }))
}

fn parse_record(rec: &Value) -> Result<BodyRecord> {
    let key = rec
        .get("key")
        .and_then(Value::as_str)
        .ok_or_else(|| Fail::new("bodies file: a record has no string 'key'"))?
        .to_string();
    // `rotationPeriodDays` may be null: a body with no spin.
    let period = match rec.get("rotationPeriodDays") {
        Some(Value::Null) => None,
        _ => Some(number(rec, &key, "rotationPeriodDays")?),
    };
    let regime = rec.get("cloudRegime").and_then(Value::as_str).unwrap_or("unknown").to_string();
    let body = BodyInputs::new(
        period,
        number(rec, &key, "radiusKm")?,
        number(rec, &key, "gravityMps2")?,
        number(rec, &key, "scaleHeightKm")?,
    )
    .with_regime(regime);
    let surface = match rec.get("surface") {
        Some(s) => parse_surface(&key, s)?,
        None => None,
    };
    Ok(BodyRecord {
        measured: truthy(rec.get("measured")),
        surface,
        surface_note: rec.get("surfaceNote").and_then(Value::as_str).map(str::to_string),
        key,
        body,
    })
}

/// Parses the text of a bodies file (`json.loads(text)['bodies']`).
pub fn parse_bodies(text: &str, base: PathBuf) -> Result<Bodies> {
    let doc: Value = serde_json::from_str(text).map_err(|e| Fail::new(format!("bodies file: not JSON: {e}")))?;
    let list = doc
        .get("bodies")
        .and_then(Value::as_array)
        .ok_or_else(|| Fail::new("bodies file: no 'bodies' list"))?;
    let records = list.iter().map(parse_record).collect::<Result<Vec<_>>>()?;
    Ok(Bodies { records, base })
}

/// `load_bodies(path)`: the records and the folder their surface paths are relative to (the file's own).
pub fn load_bodies(project: &Project, path: Option<&Path>) -> Result<Bodies> {
    let p = path.map(Path::to_path_buf).unwrap_or_else(|| project.default_bodies());
    let text = std::fs::read_to_string(&p).context(|| format!("{}", p.display()))?;
    // Python: `p.resolve().parent`. `absolute` gives an absolute path without the `\\?\` form that `canonicalize`
    // returns on Windows (and does not follow a symlinked bodies file).
    let base = std::path::absolute(&p).ok().and_then(|a| a.parent().map(Path::to_path_buf)).unwrap_or_default();
    parse_bodies(&text, base).map_err(|e| Fail::new(format!("{}: {e}", p.display())))
}

/// `Surface.load(record, base)`: reads the two raw files named by the record.
pub fn load_surface(record: &SurfaceRecord, base: &Path) -> Result<Surface> {
    let read = |name: &str| std::fs::read(base.join(name)).ok();
    match Surface::load(Some(record), read) {
        Ok(Some(s)) => Ok(s),
        Ok(None) => unreachable!("a record was given"),
        // Python's message names the land file's full path: `{base / record['land']}: surface file not found`.
        Err(SurfaceError::FileNotFound(name)) => Err(Fail::new(format!("{}: surface file not found", base.join(name).display()))),
        Err(e) => Err(Fail::new(format!("{}: {e}", base.display()))),
    }
}

/// `body_surface(key, path)`: the land and relief of one body, or an error naming why it has none.
pub fn body_surface(project: &Project, key: &str, path: Option<&Path>) -> Result<Surface> {
    let bodies = load_bodies(project, path)?;
    let rec = bodies.records.iter().find(|r| r.key == key);
    match rec.and_then(|r| r.surface.as_ref()) {
        Some(s) => load_surface(s, &bodies.base),
        None => {
            let note = rec.and_then(|r| r.surface_note.as_deref()).unwrap_or("unknown key");
            Err(Fail::new(format!("{key}: no surface in the bodies file ({note})")))
        }
    }
}

/// The test bodies of `gen.py`'s `--body` (published bulk values [measured], scale heights R T / (M g) at the
/// surface [derived]). Only water-cloud bodies bake: Mars, Titan and Venus are here to show the refusal, or for
/// an `--experiment`; `fast` is a made-up fast rotator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
pub enum TestBody {
    Earth,
    Fast,
    Mars,
    Titan,
    Venus,
}

impl TestBody {
    pub fn name(self) -> &'static str {
        match self {
            TestBody::Earth => "earth",
            TestBody::Fast => "fast",
            TestBody::Mars => "mars",
            TestBody::Titan => "titan",
            TestBody::Venus => "venus",
        }
    }

    pub fn inputs(self) -> BodyInputs {
        match self {
            TestBody::Earth => EARTH,
            TestBody::Mars => BodyInputs::new(Some(1.02596), 3389.5, 3.721, 11.1).with_regime("thin water ice"),
            TestBody::Titan => BodyInputs::new(Some(15.945), 2574.7, 1.352, 20.6).with_regime("methane"),
            TestBody::Venus => BodyInputs::new(Some(-243.02), 6051.8, 8.87, 15.7).with_regime("sulfuric acid"),
            TestBody::Fast => BodyInputs::new(Some(0.4), 6371.0, 9.81, 8.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cloud_map_generator_core::climate::is_bakeable;

    const FILE: &str = r#"{"note": "x", "bodies": [
        {"key": "earth", "rotationPeriodDays": 0.99727, "radiusKm": 6371, "gravityMps2": 9.81,
         "scaleHeightKm": 8.4, "cloudRegime": "water", "measured": false, "surface": null},
        {"key": "mars", "rotationPeriodDays": 1.02596, "radiusKm": 3389.5, "gravityMps2": 3.721,
         "scaleHeightKm": 11.1, "cloudRegime": "thin water ice", "surfaceNote": "not water"},
        {"key": "rock", "rotationPeriodDays": null, "radiusKm": 1000, "gravityMps2": 1, "scaleHeightKm": 5,
         "measured": 1, "surface": {"width": 4, "height": 2, "land": "l.u8", "elevation": "e.i16", "source": "s"}},
        {"key": "bare", "rotationPeriodDays": 2, "radiusKm": 1, "gravityMps2": 1, "scaleHeightKm": 1, "surface": {}}
    ]}"#;

    #[test]
    fn a_bodies_file_parses_like_bodies_py() {
        let b = parse_bodies(FILE, PathBuf::from("base")).unwrap();
        assert_eq!(b.records.len(), 4);
        let (earth, mars, rock, bare) = (&b.records[0], &b.records[1], &b.records[2], &b.records[3]);
        assert_eq!(earth.body, BodyInputs::new(Some(0.99727), 6371.0, 9.81, 8.4));
        assert!(is_bakeable(&earth.body) && !earth.measured && earth.surface.is_none());
        // `cloudRegime` absent means 'unknown' (not bakeable); a regime string is kept as is.
        assert_eq!(mars.body.cloud_regime, "thin water ice");
        assert_eq!(mars.surface_note.as_deref(), Some("not water"));
        assert!(!is_bakeable(&mars.body));
        let unknown = parse_bodies(r#"{"bodies":[{"key":"k","rotationPeriodDays":1,"radiusKm":1,"gravityMps2":1,"scaleHeightKm":1}]}"#, PathBuf::new()).unwrap();
        assert_eq!(unknown.records[0].body.cloud_regime, "unknown");
        // null period = no spin; any truthy `measured`; a surface record with its fields.
        assert_eq!(rock.body.rotation_period_days, None);
        assert!(rock.measured);
        let s = rock.surface.as_ref().unwrap();
        assert_eq!((s.width, s.height, s.land.as_str(), s.elevation.as_str(), s.source.as_deref()), (4, 2, "l.u8", "e.i16", Some("s")));
        // An empty surface record is no surface (`if not record`).
        assert!(bare.surface.is_none());
    }

    #[test]
    fn a_record_missing_a_field_names_it() {
        let e = parse_bodies(r#"{"bodies":[{"key":"k","radiusKm":1}]}"#, PathBuf::new()).unwrap_err();
        assert!(e.0.contains("'k'") && e.0.contains("rotationPeriodDays"), "{e}");
        let e = parse_bodies(r#"{"nothing": []}"#, PathBuf::new()).unwrap_err();
        assert!(e.0.contains("bodies"), "{e}");
    }

    #[test]
    fn the_test_bodies_are_gen_pys() {
        assert!(is_bakeable(&TestBody::Earth.inputs()) && is_bakeable(&TestBody::Fast.inputs()));
        assert!(!is_bakeable(&TestBody::Mars.inputs()) && !is_bakeable(&TestBody::Titan.inputs()));
        assert_eq!(TestBody::Venus.inputs().rotation_period_days, Some(-243.02));
        assert_eq!(TestBody::Fast.inputs().rotation_period_days, Some(0.4));
    }

    #[test]
    fn a_surface_loads_from_its_two_raw_files() {
        let dir = std::env::temp_dir().join(format!("cloud-map-generator-cli-bodies-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("l.u8"), [0u8, 255, 255, 0, 0, 255, 0, 255]).unwrap();
        let elev: Vec<u8> = [0i16, 1500, 3000, -20, 0, 500, 0, 250].iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(dir.join("e.i16"), elev).unwrap();
        let rec = SurfaceRecord { width: 4, height: 2, land: "l.u8".into(), elevation: "e.i16".into(), source: None };
        let s = load_surface(&rec, &dir).unwrap();
        assert_eq!((s.h, s.w, s.source.as_str()), (2, 4, "l.u8"));
        assert_eq!(s.land[1], 1.0);
        assert_eq!(s.elevation_km[2], 3.0);
        assert_eq!(s.elevation_km[3], 0.0, "below the sea is clamped to 0");
        let gone = SurfaceRecord { land: "nope.u8".into(), ..rec };
        assert!(load_surface(&gone, &dir).unwrap_err().0.contains("surface file not found"));
    }
}
