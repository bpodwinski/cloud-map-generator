//! Reader, comparer and runner of the golden cases (`golden/README.md`): the oracle of the Rust port.
//!
//! A test of the port loads a group, hands every case to a dispatcher that calls the port's function and
//! returns an [`Out`], and the runner compares it with the frozen outputs under each case's tolerance:
//!
//! ```ignore
//! let cfg = Config { group: "climate_flow", prefixes: &["cloudgen.climate."], expected_failures: &[] };
//! run(&cfg, |case| match case.function.as_str() {
//!     "cloudgen.climate.hadley_edge_deg" => Some(Out::new().f64("value", &[], &[...])),
//!     _ => None, // not ported yet: the runner reports it
//! }).assert_clean();
//! ```
//!
//! Honesty rules, enforced by [`Report::assert_clean`]: a case that fails, a case the dispatcher does not
//! handle, and a case listed in `expected_failures` that now passes are ALL errors. A known failure is
//! listed with its reason, never silently skipped. The tolerances are the cases' own: do not loosen them.

use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------------------------------------
// Arrays

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    Bool,
}

impl DType {
    fn parse(s: &str) -> DType {
        match s {
            "f32" => DType::F32,
            "f64" => DType::F64,
            "u8" => DType::U8,
            "i16" => DType::I16,
            "u16" => DType::U16,
            "i32" => DType::I32,
            "u32" => DType::U32,
            "i64" => DType::I64,
            "bool" => DType::Bool,
            other => panic!("unknown dtype {other:?} in a golden case"),
        }
    }
    pub fn size(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::I16 | DType::U16 => 2,
            DType::U8 | DType::Bool => 1,
        }
    }
    pub fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }
    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::U8 => "u8",
            DType::I16 => "i16",
            DType::U16 => "u16",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::Bool => "bool",
        }
    }
}

/// An n-dimensional array, C order, little-endian bytes (the `data.bin` layout).
#[derive(Clone, Debug)]
pub struct Array {
    pub dtype: DType,
    pub shape: Vec<usize>,
    bytes: Vec<u8>,
}

macro_rules! accessors {
    ($(($strict:ident, $conv:ident, $t:ty, $dt:path)),* $(,)?) => {$(
        /// The elements, which must be stored with exactly this dtype (panics otherwise).
        pub fn $strict(&self) -> Vec<$t> {
            assert_eq!(self.dtype, $dt, "array is {}, not {}", self.dtype.name(), stringify!($t));
            self.bytes.chunks_exact(std::mem::size_of::<$t>())
                .map(|c| <$t>::from_le_bytes(c.try_into().unwrap())).collect()
        }
    )*};
}

impl Array {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    accessors!(
        (as_f32, to_f32, f32, DType::F32),
        (as_f64, to_f64_strict, f64, DType::F64),
        (as_i16, to_i16, i16, DType::I16),
        (as_u16, to_u16, u16, DType::U16),
        (as_i32, to_i32, i32, DType::I32),
        (as_u32, to_u32, u32, DType::U32),
        (as_i64, to_i64_strict, i64, DType::I64),
    );

    pub fn as_u8(&self) -> Vec<u8> {
        assert_eq!(self.dtype, DType::U8, "array is {}, not u8", self.dtype.name());
        self.bytes.clone()
    }
    pub fn as_bool(&self) -> Vec<bool> {
        assert_eq!(self.dtype, DType::Bool, "array is {}, not bool", self.dtype.name());
        self.bytes.iter().map(|&b| b != 0).collect()
    }

    /// Any dtype widened to f64 (an i64 above 2^53 loses its low bits).
    pub fn to_f64(&self) -> Vec<f64> {
        match self.dtype {
            DType::F32 => self.as_f32().into_iter().map(f64::from).collect(),
            DType::F64 => self.as_f64(),
            DType::U8 => self.as_u8().into_iter().map(f64::from).collect(),
            DType::Bool => self.as_bool().into_iter().map(|b| b as u8 as f64).collect(),
            DType::I16 => self.as_i16().into_iter().map(f64::from).collect(),
            DType::U16 => self.as_u16().into_iter().map(f64::from).collect(),
            DType::I32 => self.as_i32().into_iter().map(f64::from).collect(),
            DType::U32 => self.as_u32().into_iter().map(f64::from).collect(),
            DType::I64 => self.as_i64().into_iter().map(|v| v as f64).collect(),
        }
    }

    /// Any integer or bool dtype as i64.
    pub fn to_i64(&self) -> Vec<i64> {
        match self.dtype {
            DType::U8 => self.as_u8().into_iter().map(i64::from).collect(),
            DType::Bool => self.as_bool().into_iter().map(i64::from).collect(),
            DType::I16 => self.as_i16().into_iter().map(i64::from).collect(),
            DType::U16 => self.as_u16().into_iter().map(i64::from).collect(),
            DType::I32 => self.as_i32().into_iter().map(i64::from).collect(),
            DType::U32 => self.as_u32().into_iter().map(i64::from).collect(),
            DType::I64 => self.as_i64(),
            _ => panic!("to_i64 on a {} array", self.dtype.name()),
        }
    }

    fn from_bytes(dtype: DType, shape: &[usize], bytes: Vec<u8>) -> Array {
        let n: usize = shape.iter().product();
        assert_eq!(bytes.len(), n * dtype.size(), "shape {shape:?} of {} does not match the data", dtype.name());
        Array { dtype, shape: shape.to_vec(), bytes }
    }
    pub fn from_f32(shape: &[usize], data: &[f32]) -> Array {
        Array::from_bytes(DType::F32, shape, data.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
    pub fn from_f64(shape: &[usize], data: &[f64]) -> Array {
        Array::from_bytes(DType::F64, shape, data.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
    pub fn from_u8(shape: &[usize], data: &[u8]) -> Array {
        Array::from_bytes(DType::U8, shape, data.to_vec())
    }
    pub fn from_bool(shape: &[usize], data: &[bool]) -> Array {
        Array::from_bytes(DType::Bool, shape, data.iter().map(|&b| b as u8).collect())
    }
    pub fn from_i16(shape: &[usize], data: &[i16]) -> Array {
        Array::from_bytes(DType::I16, shape, data.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
    pub fn from_u16(shape: &[usize], data: &[u16]) -> Array {
        Array::from_bytes(DType::U16, shape, data.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
    pub fn from_i32(shape: &[usize], data: &[i32]) -> Array {
        Array::from_bytes(DType::I32, shape, data.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
    pub fn from_u32(shape: &[usize], data: &[u32]) -> Array {
        Array::from_bytes(DType::U32, shape, data.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
    pub fn from_i64(shape: &[usize], data: &[i64]) -> Array {
        Array::from_bytes(DType::I64, shape, data.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
}

// ---------------------------------------------------------------------------------------------------------
// Cases

#[derive(Clone, Debug)]
pub enum Tol {
    Exact,
    /// |got - want| <= atol + rtol |want|, NaN equal to NaN, an infinity equal to itself; `frac` is the
    /// fraction of elements allowed outside that band.
    Close { rtol: f64, atol: f64, frac: f64 },
}

#[derive(Clone, Debug)]
pub struct Case {
    pub id: String,
    pub group: String,
    pub function: String,
    pub doc: String,
    pub params: Value,
    pub tags: Vec<String>,
    pub inputs: BTreeMap<String, Array>,
    pub expected: BTreeMap<String, (Array, Tol)>,
    pub dir: PathBuf,
}

impl Case {
    pub fn input(&self, name: &str) -> &Array {
        self.inputs.get(name).unwrap_or_else(|| {
            panic!("case {}: no input {name:?} (has {:?})", self.id, self.inputs.keys().collect::<Vec<_>>())
        })
    }
    pub fn has_input(&self, name: &str) -> bool {
        self.inputs.contains_key(name)
    }
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
    pub fn param(&self, name: &str) -> &Value {
        self.params
            .get(name)
            .unwrap_or_else(|| panic!("case {}: no param {name:?} (has {:?})", self.id, self.param_names()))
    }
    pub fn has_param(&self, name: &str) -> bool {
        self.params.get(name).is_some_and(|v| !v.is_null())
    }
    pub fn param_names(&self) -> Vec<String> {
        self.params.as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default()
    }
    pub fn p_f64(&self, name: &str) -> f64 {
        self.param(name).as_f64().unwrap_or_else(|| panic!("case {}: param {name:?} is not a number", self.id))
    }
    pub fn p_i64(&self, name: &str) -> i64 {
        self.param(name).as_i64().unwrap_or_else(|| panic!("case {}: param {name:?} is not an integer", self.id))
    }
    pub fn p_usize(&self, name: &str) -> usize {
        self.p_i64(name) as usize
    }
    pub fn p_bool(&self, name: &str) -> bool {
        self.param(name).as_bool().unwrap_or_else(|| panic!("case {}: param {name:?} is not a bool", self.id))
    }
    pub fn p_str(&self, name: &str) -> &str {
        self.param(name).as_str().unwrap_or_else(|| panic!("case {}: param {name:?} is not a string", self.id))
    }
    pub fn p_f64s(&self, name: &str) -> Vec<f64> {
        self.param(name)
            .as_array()
            .unwrap_or_else(|| panic!("case {}: param {name:?} is not a list", self.id))
            .iter()
            .map(|v| v.as_f64().unwrap_or_else(|| panic!("case {}: param {name:?} holds a non-number", self.id)))
            .collect()
    }
    pub fn is_slow(&self) -> bool {
        self.has_tag("slow")
    }
}

/// `<repo>/golden/data/v1`, or `$GOLDEN_ROOT`.
pub fn golden_root() -> PathBuf {
    match std::env::var_os("GOLDEN_ROOT") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../golden/data/v1"),
    }
}

fn read_arrays(rows: &Value, data: &[u8], with_tol: bool, case_id: &str) -> Vec<(String, Array, Option<Tol>)> {
    rows.as_array()
        .unwrap_or_else(|| panic!("case {case_id}: arrays missing"))
        .iter()
        .map(|r| {
            let name = r["name"].as_str().unwrap().to_string();
            let dtype = DType::parse(r["dtype"].as_str().unwrap());
            let shape: Vec<usize> = r["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            let off = r["offset"].as_u64().unwrap() as usize;
            let nbytes = r["nbytes"].as_u64().unwrap() as usize;
            assert!(off + nbytes <= data.len(), "case {case_id}: array {name} runs past data.bin");
            let arr = Array::from_bytes(dtype, &shape, data[off..off + nbytes].to_vec());
            let tol = if with_tol { Some(parse_tol(&r["tol"])) } else { None };
            (name, arr, tol)
        })
        .collect()
}

fn parse_tol(t: &Value) -> Tol {
    match t["mode"].as_str() {
        Some("exact") => Tol::Exact,
        Some("close") => Tol::Close {
            rtol: t["rtol"].as_f64().unwrap(),
            atol: t["atol"].as_f64().unwrap(),
            frac: t.get("frac").and_then(Value::as_f64).unwrap_or(0.0),
        },
        other => panic!("unknown tolerance mode {other:?}"),
    }
}

pub fn load_case(dir: &Path) -> Case {
    let meta: Value = serde_json::from_slice(&fs::read(dir.join("case.json")).expect("case.json")).expect("case.json parses");
    let id = meta["id"].as_str().unwrap().to_string();
    let data = fs::read(dir.join(meta["data"].as_str().unwrap())).expect("data.bin");
    let inputs = read_arrays(&meta["inputs"], &data, false, &id).into_iter().map(|(n, a, _)| (n, a)).collect();
    let expected = read_arrays(&meta["outputs"], &data, true, &id).into_iter().map(|(n, a, t)| (n, (a, t.unwrap()))).collect();
    Case {
        id,
        group: meta["group"].as_str().unwrap().to_string(),
        function: meta["function"].as_str().unwrap().to_string(),
        doc: meta["doc"].as_str().unwrap_or("").to_string(),
        params: meta["params"].clone(),
        tags: meta["tags"].as_array().map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect()).unwrap_or_default(),
        inputs,
        expected,
        dir: dir.to_path_buf(),
    }
}

/// Every case of a group, sorted by id.
pub fn load_group(group: &str) -> Vec<Case> {
    let root = golden_root().join(group);
    let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("golden group {group} not found at {}: {e}", root.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("case.json").exists())
        .collect();
    dirs.sort();
    dirs.iter().map(|d| load_case(d)).collect()
}

// ---------------------------------------------------------------------------------------------------------
// Outputs and comparison

/// What the port returned for one case: named arrays, to compare with the frozen outputs.
#[derive(Default, Debug)]
pub struct Out {
    pub arrays: BTreeMap<String, Array>,
}

impl Out {
    pub fn new() -> Out {
        Out::default()
    }
    pub fn put(mut self, name: &str, a: Array) -> Out {
        self.arrays.insert(name.to_string(), a);
        self
    }
    pub fn f32(self, name: &str, shape: &[usize], data: &[f32]) -> Out {
        self.put(name, Array::from_f32(shape, data))
    }
    pub fn f64(self, name: &str, shape: &[usize], data: &[f64]) -> Out {
        self.put(name, Array::from_f64(shape, data))
    }
    pub fn u8(self, name: &str, shape: &[usize], data: &[u8]) -> Out {
        self.put(name, Array::from_u8(shape, data))
    }
    pub fn bool(self, name: &str, shape: &[usize], data: &[bool]) -> Out {
        self.put(name, Array::from_bool(shape, data))
    }
    pub fn i16(self, name: &str, shape: &[usize], data: &[i16]) -> Out {
        self.put(name, Array::from_i16(shape, data))
    }
    pub fn u16(self, name: &str, shape: &[usize], data: &[u16]) -> Out {
        self.put(name, Array::from_u16(shape, data))
    }
    pub fn i32(self, name: &str, shape: &[usize], data: &[i32]) -> Out {
        self.put(name, Array::from_i32(shape, data))
    }
    pub fn u32(self, name: &str, shape: &[usize], data: &[u32]) -> Out {
        self.put(name, Array::from_u32(shape, data))
    }
    pub fn i64(self, name: &str, shape: &[usize], data: &[i64]) -> Out {
        self.put(name, Array::from_i64(shape, data))
    }
}

fn close_one(g: f64, w: f64, rtol: f64, atol: f64) -> bool {
    if g.is_nan() || w.is_nan() {
        return g.is_nan() && w.is_nan();
    }
    if g.is_infinite() || w.is_infinite() {
        return g == w;
    }
    (g - w).abs() <= atol + rtol * w.abs()
}

/// What differs between the port's output and a frozen array (empty: it matches).
pub fn compare_array(name: &str, got: &Array, want: &Array, tol: &Tol) -> Vec<String> {
    if got.shape != want.shape {
        return vec![format!("{name}: shape {:?}, expected {:?}", got.shape, want.shape)];
    }
    if want.dtype.is_float() != got.dtype.is_float() {
        return vec![format!("{name}: dtype {}, expected {}", got.dtype.name(), want.dtype.name())];
    }
    if !want.dtype.is_float() {
        // Integers, bytes, bools: always exact (any integer width: the values are what counts).
        let (g, w) = (got.to_i64(), want.to_i64());
        let bad: Vec<usize> = (0..g.len()).filter(|&i| g[i] != w[i]).collect();
        if bad.is_empty() {
            return vec![];
        }
        let i = bad[0];
        return vec![format!("{name}: {} of {} differ (exact), first at {i}: got {}, want {}", bad.len(), g.len(), g[i], w[i])];
    }
    let (g, w) = (got.to_f64(), want.to_f64());
    let (rtol, atol, frac, exact) = match tol {
        Tol::Exact => (0.0, 0.0, 0.0, true),
        Tol::Close { rtol, atol, frac } => (*rtol, *atol, *frac, false),
    };
    let ok = |a: f64, b: f64| if exact { a == b || (a.is_nan() && b.is_nan()) } else { close_one(a, b, rtol, atol) };
    let bad: Vec<usize> = (0..g.len()).filter(|&i| !ok(g[i], w[i])).collect();
    if g.is_empty() || (bad.len() as f64) <= frac * g.len() as f64 {
        return vec![];
    }
    let max_diff = bad.iter().filter(|&&i| g[i].is_finite() && w[i].is_finite()).map(|&i| (g[i] - w[i]).abs()).fold(0.0, f64::max);
    let i = bad[0];
    vec![format!(
        "{name}: {} of {} outside {tol:?}, max |diff| {max_diff:.3e}, first at {i}: got {:e}, want {:e}",
        bad.len(),
        g.len(),
        g[i],
        w[i]
    )]
}

fn widened_case(c: &Case, widen: &[Widen]) -> Case {
    let Some(w) = widen.iter().find(|w| c.function.starts_with(w.prefix)) else { return c.clone() };
    let mut out = c.clone();
    for (_, tol) in out.expected.values_mut() {
        if let Tol::Close { rtol, atol, .. } = tol {
            *rtol *= w.factor;
            *atol *= w.factor;
        }
    }
    out
}

/// What differs between the port's outputs and the case's frozen outputs (empty: the case passes).
pub fn compare(case: &Case, got: &Out) -> Vec<String> {
    let mut problems = Vec::new();
    for (name, (want, tol)) in &case.expected {
        match got.arrays.get(name) {
            None => problems.push(format!("{name}: output missing")),
            Some(g) => problems.extend(compare_array(name, g, want, tol)),
        }
    }
    problems
}

// ---------------------------------------------------------------------------------------------------------
// Runner

pub struct Config<'a> {
    /// Golden group directory name, e.g. "climate_flow".
    pub group: &'a str,
    /// Run only the cases whose `function` starts with one of these (empty: all of the group).
    pub prefixes: &'a [&'a str],
    /// (case id, reason): known failures. Listing is the ONLY way to leave a case failing; a listed case
    /// that passes is reported as stale.
    pub expected_failures: &'a [(&'a str, &'a str)],
}

#[derive(Debug, Default)]
pub struct Report {
    pub group: String,
    pub passed: Vec<String>,
    pub failed: Vec<(String, Vec<String>)>,
    pub unported: Vec<String>,
    pub expected_failed: Vec<(String, String)>,
    pub stale_expected: Vec<String>,
    pub skipped_slow: Vec<String>,
    pub widened: Vec<String>,
}

impl Report {
    pub fn summary(&self) -> String {
        let mut s = format!(
            "golden {}: {} passed, {} failed, {} unported, {} expected failures, {} stale expectations, {} slow skipped",
            self.group,
            self.passed.len(),
            self.failed.len(),
            self.unported.len(),
            self.expected_failed.len(),
            self.stale_expected.len(),
            self.skipped_slow.len()
        );
        for (id, why) in self.failed.iter().take(40) {
            s += &format!("\n  FAIL {id}: {}", why.join("; "));
        }
        if self.failed.len() > 40 {
            s += &format!("\n  ... and {} more failures", self.failed.len() - 40);
        }
        for id in self.unported.iter().take(40) {
            s += &format!("\n  UNPORTED {id}");
        }
        if self.unported.len() > 40 {
            s += &format!("\n  ... and {} more unported", self.unported.len() - 40);
        }
        for id in &self.stale_expected {
            s += &format!("\n  STALE expected failure {id}: it passes now, remove it from expected_failures");
        }
        s
    }

    pub fn is_clean(&self) -> bool {
        self.failed.is_empty() && self.unported.is_empty() && self.stale_expected.is_empty()
    }

    /// Prints the summary; panics unless every case passed or is a listed, still-failing known failure.
    pub fn assert_clean(&self) {
        println!("{}", self.summary());
        for (id, why) in &self.expected_failed {
            println!("  known failure {id}: {why}");
        }
        for w in &self.widened {
            println!("  tolerance widened: {w}");
        }
        assert!(self.is_clean(), "{}", self.summary());
    }
}

/// A documented widening of the tolerances of the cases whose `function` starts with `prefix`: both `rtol` and
/// `atol` of every `Close` output are multiplied by `factor`. For a port that cannot reach the frozen tolerance
/// for a stated, measured reason (the GPU path is float32 only). Exact outputs are never widened. Each use is
/// printed by [`Report::assert_clean`], so it cannot hide.
pub struct Widen<'a> {
    pub prefix: &'a str,
    pub factor: f64,
    pub reason: &'a str,
}

/// [`run`] with documented tolerance widenings.
pub fn run_widened(cfg: &Config, widen: &[Widen], port: impl Fn(&Case) -> Option<Out>) -> Report {
    let mut report = run_inner(cfg, widen, port);
    report.widened = widen.iter().map(|w| format!("{} x{}: {}", w.prefix, w.factor, w.reason)).collect();
    report
}

/// Runs the group's cases through `port`: `None` = the case is not handled (not ported yet).
pub fn run(cfg: &Config, port: impl Fn(&Case) -> Option<Out>) -> Report {
    run_inner(cfg, &[], port)
}

fn run_inner(cfg: &Config, widen: &[Widen], port: impl Fn(&Case) -> Option<Out>) -> Report {
    let mut report = Report { group: cfg.group.to_string(), ..Report::default() };
    let slow = std::env::var_os("GOLDEN_SLOW").is_some();
    let cases: Vec<Case> = load_group(cfg.group)
        .into_iter()
        .filter(|c| cfg.prefixes.is_empty() || cfg.prefixes.iter().any(|p| c.function.starts_with(p)))
        .collect();
    assert!(!cases.is_empty(), "no golden case of group {} matches the prefixes {:?}", cfg.group, cfg.prefixes);
    for c in &cases {
        let known = cfg.expected_failures.iter().find(|(id, _)| *id == c.id).map(|(_, why)| why.to_string());
        if c.is_slow() && !slow {
            report.skipped_slow.push(c.id.clone());
            continue;
        }
        let outcome: Result<Option<Out>, String> = match catch_unwind(AssertUnwindSafe(|| port(c))) {
            Ok(o) => Ok(o),
            Err(p) => Err(p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic".into())),
        };
        let problems: Option<Vec<String>> = match outcome {
            Err(msg) => Some(vec![format!("panicked: {msg}")]),
            Ok(None) => {
                if let Some(why) = known {
                    report.expected_failed.push((c.id.clone(), why));
                } else {
                    report.unported.push(c.id.clone());
                }
                continue;
            }
            Ok(Some(out)) => {
                let p = compare(&widened_case(c, widen), &out);
                if p.is_empty() {
                    None
                } else {
                    Some(p)
                }
            }
        };
        match (problems, known) {
            (None, None) => report.passed.push(c.id.clone()),
            (None, Some(_)) => report.stale_expected.push(c.id.clone()),
            (Some(_), Some(why)) => report.expected_failed.push((c.id.clone(), why)),
            (Some(p), None) => report.failed.push((c.id.clone(), p)),
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: &[f64]) -> Array {
        Array::from_f64(&[v.len()], v)
    }

    #[test]
    fn close_and_exact_comparisons() {
        let want = arr(&[1.0, 2.0, f64::NAN, f64::INFINITY]);
        let tol = Tol::Close { rtol: 1e-6, atol: 0.0, frac: 0.0 };
        assert!(compare_array("x", &arr(&[1.0 + 1e-8, 2.0, f64::NAN, f64::INFINITY]), &want, &tol).is_empty());
        assert!(!compare_array("x", &arr(&[1.1, 2.0, f64::NAN, f64::INFINITY]), &want, &tol).is_empty());
        assert!(!compare_array("x", &arr(&[1.0, 2.0, 0.0, f64::INFINITY]), &want, &tol).is_empty());
        assert!(!compare_array("x", &arr(&[1.0 + 1e-8, 2.0, f64::NAN, f64::INFINITY]), &want, &Tol::Exact).is_empty());
    }

    #[test]
    fn frac_allows_a_few_outliers_and_shapes_must_match() {
        let want = arr(&vec![1.0; 100]);
        let mut got = vec![1.0; 100];
        got[3] = 2.0;
        let t = |frac| Tol::Close { rtol: 1e-6, atol: 1e-6, frac };
        assert!(!compare_array("x", &arr(&got), &want, &t(0.0)).is_empty());
        assert!(compare_array("x", &arr(&got), &want, &t(0.02)).is_empty());
        assert!(!compare_array("x", &arr(&got[..99]), &want, &t(0.5)).is_empty());
    }

    #[test]
    fn integers_compare_by_value_across_widths_and_float_kind_must_match() {
        let want = Array::from_i64(&[3], &[1, 2, 3]);
        assert!(compare_array("x", &Array::from_i32(&[3], &[1, 2, 3]), &want, &Tol::Exact).is_empty());
        assert!(!compare_array("x", &Array::from_i32(&[3], &[1, 2, 4]), &want, &Tol::Exact).is_empty());
        assert!(!compare_array("x", &arr(&[1.0, 2.0, 3.0]), &want, &Tol::Exact).is_empty());
        let f32w = Array::from_f32(&[1], &[0.5]);
        assert!(compare_array("x", &arr(&[0.5]), &f32w, &Tol::Exact).is_empty()); // f32 vs f64: by value
    }

    #[test]
    fn array_round_trip() {
        let a = Array::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.as_f32(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.to_f64(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Array::from_bool(&[2], &[true, false]).to_i64(), vec![1, 0]);
    }

    #[test]
    fn the_frozen_data_loads() {
        let cases = load_group("sphere_noise");
        assert!(cases.len() > 100);
        assert!(cases.iter().all(|c| !c.expected.is_empty() && !c.doc.is_empty()));
    }
}
