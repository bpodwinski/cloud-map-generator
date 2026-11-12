//! The arguments of the ten tools, parsed and checked the way the Python server's pydantic models did: the same
//! defaults, the same bounds, the same lax coercions (`"256"` for an int, `"true"` for a bool), unknown
//! top-level arguments ignored, unknown fields of `overrides` refused, and the same words in the error.
//!
//! An invalid call is one tool error that lists every field that was wrong, for example
//! `Error executing tool view_map: 1 validation error for view_mapArguments`, then `size`, then
//! `  Input should be less than or equal to 1024 [type=less_than_equal, input_value=5000, input_type=int]`.

use serde_json::{Map, Value};

use crate::pyfmt::{fmt_g, py_type_name, py_value_repr};
use crate::rules::Overrides;

type Obj = Map<String, Value>;

struct ArgErr {
    loc: Vec<String>,
    msg: String,
    kind: &'static str,
    input: Value,
}

/// Collects the errors of one call while its fields are read.
pub struct Parser<'a> {
    root: &'a Obj,
    errs: Vec<ArgErr>,
}

fn bound_text(x: f64) -> String {
    if x.fract() == 0.0 && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        fmt_g(x)
    }
}

type Fail = (String, &'static str);

fn as_int(v: &Value) -> Result<i64, Fail> {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i)
            } else if let Some(u) = n.as_u64() {
                Ok(u.min(i64::MAX as u64) as i64)
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.0e18 {
                    Ok(f as i64)
                } else if f.is_finite() {
                    Err(("Input should be a valid integer, got a number with a fractional part".into(), "int_from_float"))
                } else {
                    Err(("Input should be a finite number".into(), "finite_number"))
                }
            }
        }
        Value::String(s) => s
            .trim()
            .parse::<i64>()
            .map_err(|_| ("Input should be a valid integer, unable to parse string as an integer".to_string(), "int_parsing")),
        _ => Err(("Input should be a valid integer".into(), "int_type")),
    }
}

fn as_float(v: &Value) -> Result<f64, Fail> {
    let f = match v {
        Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
        Value::String(s) => s
            .trim()
            .to_lowercase()
            .parse::<f64>()
            .map_err(|_| ("Input should be a valid number, unable to parse string as a number".to_string(), "float_parsing"))?,
        _ => return Err(("Input should be a valid number".into(), "float_type")),
    };
    if f.is_finite() {
        Ok(f)
    } else {
        Err(("Input should be a finite number".into(), "finite_number"))
    }
}

fn as_bool(v: &Value) -> Result<bool, Fail> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Number(n) if n.as_i64() == Some(0) => Ok(false),
        Value::Number(n) if n.as_i64() == Some(1) => Ok(true),
        Value::String(s) => match s.trim().to_lowercase().as_str() {
            "true" | "t" | "yes" | "y" | "on" | "1" => Ok(true),
            "false" | "f" | "no" | "n" | "off" | "0" => Ok(false),
            _ => Err(("Input should be a valid boolean, unable to interpret input".into(), "bool_parsing")),
        },
        _ => Err(("Input should be a valid boolean".into(), "bool_type")),
    }
}

fn as_str(v: &Value) -> Result<String, Fail> {
    match v {
        Value::String(s) => Ok(s.clone()),
        _ => Err(("Input should be a valid string".into(), "string_type")),
    }
}

fn literal_text(options: &[&str]) -> String {
    let quoted: Vec<String> = options.iter().map(|o| format!("'{o}'")).collect();
    match quoted.len() {
        0 => String::new(),
        1 => quoted[0].clone(),
        n => format!("{} or {}", quoted[..n - 1].join(", "), quoted[n - 1]),
    }
}

impl<'a> Parser<'a> {
    pub fn new(root: &'a Obj) -> Self {
        Parser { root, errs: Vec::new() }
    }

    fn push(&mut self, loc: &[String], fail: Fail, input: &Value) {
        self.errs.push(ArgErr { loc: loc.to_vec(), msg: fail.0, kind: fail.1, input: input.clone() });
    }

    fn missing(&mut self, loc: &[String], parent: &Obj) {
        self.errs.push(ArgErr {
            loc: loc.to_vec(),
            msg: "Field required".into(),
            kind: "missing",
            input: Value::Object(parent.clone()),
        });
    }

    fn loc(parent: &[String], name: &str) -> Vec<String> {
        let mut l = parent.to_vec();
        l.push(name.to_string());
        l
    }

    // ---- fields of the top-level object

    pub fn str_opt(&mut self, name: &str) -> Option<String> {
        let root = self.root;
        self.str_opt_in(root, &[], name, None, None)
    }

    pub fn str_req(&mut self, name: &str) -> String {
        let root = self.root;
        self.str_in(root, &[], name, None, 0, usize::MAX).unwrap_or_default()
    }

    pub fn str_default(&mut self, name: &str, default: &str) -> String {
        let root = self.root;
        self.str_in(root, &[], name, Some(default), 0, usize::MAX).unwrap_or_else(|| default.to_string())
    }

    pub fn int(&mut self, name: &str, default: i64, lo: Option<i64>, hi: Option<i64>) -> i64 {
        let root = self.root;
        match root.get(name) {
            None => default,
            Some(v) => self.int_value(&[name.to_string()], v, lo, hi).unwrap_or(default),
        }
    }

    pub fn int_opt(&mut self, name: &str) -> Option<i64> {
        let root = self.root;
        match root.get(name) {
            None | Some(Value::Null) => None,
            Some(v) => self.int_value(&[name.to_string()], v, None, None),
        }
    }

    pub fn float(&mut self, name: &str, default: f64, lo: Option<f64>, hi: Option<f64>) -> f64 {
        let root = self.root;
        match root.get(name) {
            None => default,
            Some(v) => self.float_value(&[name.to_string()], v, lo, hi).unwrap_or(default),
        }
    }

    pub fn boolean(&mut self, name: &str, default: bool) -> bool {
        let root = self.root;
        match root.get(name) {
            None => default,
            Some(v) => self.bool_value(&[name.to_string()], v).unwrap_or(default),
        }
    }

    pub fn literal(&mut self, name: &str, default: &str, options: &[&str]) -> String {
        let root = self.root;
        match root.get(name) {
            None => default.to_string(),
            Some(v) => self.literal_value(&[name.to_string()], v, options).unwrap_or_else(|| default.to_string()),
        }
    }

    pub fn literal_opt(&mut self, name: &str, options: &[&str]) -> Option<String> {
        let root = self.root;
        match root.get(name) {
            None | Some(Value::Null) => None,
            Some(v) => self.literal_value(&[name.to_string()], v, options),
        }
    }

    /// The `overrides` model: any of the five physics fields, nothing else.
    pub fn overrides(&mut self, name: &str) -> Option<Overrides> {
        let root = self.root;
        let v = match root.get(name) {
            None | Some(Value::Null) => return None,
            Some(v) => v,
        };
        let loc = vec![name.to_string()];
        let Some(obj) = v.as_object() else {
            self.push(&loc, ("Input should be a valid dictionary or instance of PhysicsOverrides".into(), "model_type"), v);
            return None;
        };
        const FIELDS: [&str; 5] = ["rotation_period_days", "radius_km", "gravity_mps2", "scale_height_km", "cloud_regime"];
        let mut out = Overrides::default();
        for (k, val) in obj {
            if !FIELDS.contains(&k.as_str()) {
                let l = Self::loc(&loc, k);
                self.push(&l, ("Extra inputs are not permitted".into(), "extra_forbidden"), val);
            }
        }
        for f in &FIELDS[..4] {
            if let Some(val) = obj.get(*f) {
                if !val.is_null() {
                    let l = Self::loc(&loc, f);
                    let parsed = self.float_value(&l, val, None, None);
                    match *f {
                        "rotation_period_days" => out.rotation_period_days = parsed,
                        "radius_km" => out.radius_km = parsed,
                        "gravity_mps2" => out.gravity_mps2 = parsed,
                        _ => out.scale_height_km = parsed,
                    }
                }
            }
        }
        out.cloud_regime = self.str_opt_in(obj, &loc, "cloud_regime", None, Some(64));
        Some(out)
    }

    /// The `bodies` list of `bake_bodies`: records of one name and the four physics numbers.
    pub fn bodies(&mut self, name: &str) -> Option<Vec<BakeBody>> {
        let root = self.root;
        let v = match root.get(name) {
            None | Some(Value::Null) => return None,
            Some(v) => v,
        };
        let loc = vec![name.to_string()];
        let Some(items) = v.as_array() else {
            self.push(&loc, ("Input should be a valid list".into(), "list_type"), v);
            return None;
        };
        let mut out = Vec::new();
        for (i, item) in items.iter().enumerate() {
            let iloc = Self::loc(&loc, &i.to_string());
            let Some(obj) = item.as_object() else {
                self.push(&iloc, ("Input should be a valid dictionary or instance of BakeBody".into(), "model_type"), item);
                continue;
            };
            const FIELDS: [&str; 7] =
                ["key", "rotation_period_days", "radius_km", "gravity_mps2", "scale_height_km", "cloud_regime", "measured"];
            for (k, val) in obj {
                if !FIELDS.contains(&k.as_str()) {
                    let l = Self::loc(&iloc, k);
                    self.push(&l, ("Extra inputs are not permitted".into(), "extra_forbidden"), val);
                }
            }
            let key = self.str_in(obj, &iloc, "key", None, 1, 64).unwrap_or_default();
            let mut nums = [0.0f64; 4];
            for (n, f) in FIELDS[1..5].iter().enumerate() {
                match obj.get(*f) {
                    None => self.missing(&Self::loc(&iloc, f), obj),
                    Some(val) => nums[n] = self.float_value(&Self::loc(&iloc, f), val, None, None).unwrap_or(0.0),
                }
            }
            let cloud_regime = self.str_in(obj, &iloc, "cloud_regime", Some("water"), 0, 64).unwrap_or_else(|| "water".into());
            let measured = match obj.get("measured") {
                None => false,
                Some(val) => self.bool_value(&Self::loc(&iloc, "measured"), val).unwrap_or(false),
            };
            out.push(BakeBody {
                key,
                rotation_period_days: nums[0],
                radius_km: nums[1],
                gravity_mps2: nums[2],
                scale_height_km: nums[3],
                cloud_regime,
                measured,
            });
        }
        Some(out)
    }

    // ---- values at any depth

    fn str_opt_in(&mut self, obj: &Obj, parent: &[String], name: &str, _default: Option<&str>, max: Option<usize>) -> Option<String> {
        match obj.get(name) {
            None | Some(Value::Null) => None,
            Some(v) => {
                let loc = Self::loc(parent, name);
                match as_str(v) {
                    Err(f) => {
                        self.push(&loc, f, v);
                        None
                    }
                    Ok(s) => {
                        if max.is_some_and(|m| s.chars().count() > m) {
                            self.push(
                                &loc,
                                (format!("String should have at most {} characters", max.unwrap_or(0)), "string_too_long"),
                                v,
                            );
                            return None;
                        }
                        Some(s)
                    }
                }
            }
        }
    }

    /// A string field: required without a default (`Field required` when absent), checked against a length range.
    fn str_in(&mut self, obj: &Obj, parent: &[String], name: &str, default: Option<&str>, min: usize, max: usize) -> Option<String> {
        let loc = Self::loc(parent, name);
        match obj.get(name) {
            None => {
                if default.is_none() {
                    self.missing(&loc, obj);
                }
                None
            }
            Some(v) => match as_str(v) {
                Err(f) => {
                    self.push(&loc, f, v);
                    None
                }
                Ok(s) => {
                    let n = s.chars().count();
                    if n < min {
                        let plural = if min == 1 { "character" } else { "characters" };
                        self.push(&loc, (format!("String should have at least {min} {plural}"), "string_too_short"), v);
                        None
                    } else if n > max {
                        self.push(&loc, (format!("String should have at most {max} characters"), "string_too_long"), v);
                        None
                    } else {
                        Some(s)
                    }
                }
            },
        }
    }

    fn int_value(&mut self, loc: &[String], v: &Value, lo: Option<i64>, hi: Option<i64>) -> Option<i64> {
        match as_int(v) {
            Err(f) => {
                self.push(loc, f, v);
                None
            }
            Ok(i) => {
                if hi.is_some_and(|h| i > h) {
                    self.push(loc, (format!("Input should be less than or equal to {}", hi.unwrap_or(0)), "less_than_equal"), v);
                    None
                } else if lo.is_some_and(|l| i < l) {
                    self.push(loc, (format!("Input should be greater than or equal to {}", lo.unwrap_or(0)), "greater_than_equal"), v);
                    None
                } else {
                    Some(i)
                }
            }
        }
    }

    fn float_value(&mut self, loc: &[String], v: &Value, lo: Option<f64>, hi: Option<f64>) -> Option<f64> {
        match as_float(v) {
            Err(f) => {
                self.push(loc, f, v);
                None
            }
            Ok(x) => {
                if hi.is_some_and(|h| x > h) {
                    self.push(
                        loc,
                        (format!("Input should be less than or equal to {}", bound_text(hi.unwrap_or(0.0))), "less_than_equal"),
                        v,
                    );
                    None
                } else if lo.is_some_and(|l| x < l) {
                    self.push(
                        loc,
                        (format!("Input should be greater than or equal to {}", bound_text(lo.unwrap_or(0.0))), "greater_than_equal"),
                        v,
                    );
                    None
                } else {
                    Some(x)
                }
            }
        }
    }

    fn bool_value(&mut self, loc: &[String], v: &Value) -> Option<bool> {
        match as_bool(v) {
            Ok(b) => Some(b),
            Err(f) => {
                self.push(loc, f, v);
                None
            }
        }
    }

    fn literal_value(&mut self, loc: &[String], v: &Value, options: &[&str]) -> Option<String> {
        if let Value::String(s) = v {
            if options.contains(&s.as_str()) {
                return Some(s.clone());
            }
        }
        self.push(loc, (format!("Input should be {}", literal_text(options)), "literal_error"), v);
        None
    }

    /// `Ok` when every field was right, else the text of the tool error.
    pub fn finish(self, tool: &str) -> Result<(), String> {
        if self.errs.is_empty() {
            return Ok(());
        }
        let n = self.errs.len();
        let mut out = format!(
            "Error executing tool {tool}: {n} validation error{} for {tool}Arguments",
            if n == 1 { "" } else { "s" }
        );
        for e in &self.errs {
            out.push_str(&format!(
                "\n{}\n  {} [type={}, input_value={}, input_type={}]",
                e.loc.join("."),
                e.msg,
                e.kind,
                py_value_repr(&e.input),
                py_type_name(&e.input)
            ));
        }
        Err(out)
    }
}

/// One body of an inline `bake_bodies(bodies=[...])` call.
#[derive(Debug, Clone, PartialEq)]
pub struct BakeBody {
    pub key: String,
    pub rotation_period_days: f64,
    pub radius_km: f64,
    pub gravity_mps2: f64,
    pub scale_height_km: f64,
    pub cloud_regime: String,
    pub measured: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: Value) -> Obj {
        v.as_object().cloned().unwrap()
    }

    #[test]
    fn bounds_literals_and_defaults_read_like_pydantic() {
        let a = obj(json!({"id": "abcdef123456", "size": 5000, "view": "cube", "lat_deg": 120}));
        let mut p = Parser::new(&a);
        assert_eq!(p.str_opt("id").as_deref(), Some("abcdef123456"));
        p.int("size", 512, Some(64), Some(1024));
        p.literal("view", "map", &["map", "globe"]);
        p.float("lat_deg", 20.0, Some(-90.0), Some(90.0));
        let e = p.finish("view_map").unwrap_err();
        assert!(e.starts_with("Error executing tool view_map: 3 validation errors for view_mapArguments\nsize\n  Input should be less than or equal to 1024 [type=less_than_equal, input_value=5000, input_type=int]"), "{e}");
        assert!(e.contains("view\n  Input should be 'map' or 'globe' [type=literal_error, input_value='cube', input_type=str]"), "{e}");
        assert!(e.contains("lat_deg\n  Input should be less than or equal to 90 [type=less_than_equal, input_value=120, input_type=int]"), "{e}");
    }

    #[test]
    fn lax_coercions_and_the_four_way_literal() {
        let a = obj(json!({"width": "256", "save": "false", "compute": "turbo", "seed": 2.0}));
        let mut p = Parser::new(&a);
        assert_eq!(p.int("width", 512, None, None), 256);
        assert!(!p.boolean("save", true));
        assert_eq!(p.int("seed", 42, Some(0), Some(2147483647)), 2);
        p.literal("compute", "auto", &["auto", "cpu", "gpu", "hybrid"]);
        let e = p.finish("generate_map").unwrap_err();
        assert!(e.contains("Input should be 'auto', 'cpu', 'gpu' or 'hybrid' [type=literal_error, input_value='turbo'"), "{e}");
    }

    #[test]
    fn overrides_refuse_unknown_fields_and_non_finite_numbers() {
        let a = obj(json!({"overrides": {"rotation": 1, "radius_km": "nan", "gravity_mps2": "abc", "cloud_regime": "x".repeat(70)}}));
        let mut p = Parser::new(&a);
        assert!(p.overrides("overrides").is_some());
        let e = p.finish("derive_physics").unwrap_err();
        assert!(e.contains("overrides.rotation\n  Extra inputs are not permitted [type=extra_forbidden, input_value=1, input_type=int]"), "{e}");
        assert!(e.contains("overrides.radius_km\n  Input should be a finite number [type=finite_number, input_value='nan', input_type=str]"), "{e}");
        assert!(e.contains("overrides.gravity_mps2\n  Input should be a valid number, unable to parse string as a number"), "{e}");
        assert!(e.contains("overrides.cloud_regime\n  String should have at most 64 characters"), "{e}");
        let a = obj(json!({"overrides": 5}));
        let mut p = Parser::new(&a);
        p.overrides("overrides");
        assert!(p.finish("t").unwrap_err().contains("Input should be a valid dictionary or instance of PhysicsOverrides [type=model_type, input_value=5, input_type=int]"));
    }

    #[test]
    fn bodies_name_what_is_missing() {
        let a = obj(json!({"bodies": [{"key": "x"}, {"key": "", "rotation_period_days": 1, "radius_km": 1000, "gravity_mps2": 9, "scale_height_km": 8}]}));
        let mut p = Parser::new(&a);
        p.bodies("bodies");
        let e = p.finish("bake_bodies").unwrap_err();
        assert!(e.starts_with("Error executing tool bake_bodies: 5 validation errors"), "{e}");
        assert!(e.contains("bodies.0.rotation_period_days\n  Field required [type=missing, input_value={'key': 'x'}, input_type=dict]"), "{e}");
        assert!(e.contains("bodies.1.key\n  String should have at least 1 character"), "{e}");
    }

    #[test]
    fn required_strings_and_wrong_types() {
        let a = obj(json!({"job_id": 5}));
        let mut p = Parser::new(&a);
        p.str_req("job_id");
        assert!(p.finish("get_generation").unwrap_err().contains("job_id\n  Input should be a valid string [type=string_type, input_value=5, input_type=int]"));
        let a = obj(json!({}));
        let mut p = Parser::new(&a);
        p.str_req("path");
        assert!(p.finish("measure_map").unwrap_err().contains("path\n  Field required [type=missing, input_value={}, input_type=dict]"));
        let a = obj(json!({"unknown": 1}));
        assert!(Parser::new(&a).finish("explain_model").is_ok(), "unknown top-level arguments are ignored");
    }
}
