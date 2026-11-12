//! Text helpers that reproduce Python's formatting, so the sentences this server hands to an AI are the ones the
//! Python server handed (`{x:g}`, `{x:.1f}`, `repr(str)`, `round(x, 4)`, `str.isprintable`).

use serde_json::Value;

/// `format(x, 'g')`: six significant digits, trailing zeros removed, an exponent below 1e-4 and from 1e6.
pub fn fmt_g(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    const P: i32 = 6;
    let e = format!("{:.*e}", (P - 1) as usize, x);
    let (mantissa, exponent) = e.split_once('e').expect("exponent form");
    let exponent: i32 = exponent.parse().expect("exponent digits");
    if exponent < -4 || exponent >= P {
        format!("{}e{}{:02}", trim_zeros(mantissa), if exponent < 0 { '-' } else { '+' }, exponent.abs())
    } else {
        let decimals = (P - 1 - exponent) as usize;
        trim_zeros(&format!("{:.*}", decimals, x))
    }
}

fn trim_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// `{x:.nd$f}`, with `n/a` never involved: the caller passes a finite number.
pub fn fixed(x: f64, nd: usize) -> String {
    format!("{x:.nd$}")
}

/// `{x:+.nd$f}`.
pub fn signed(x: f64, nd: usize) -> String {
    format!("{x:+.nd$}")
}

/// `round(x, digits)` for a finite float: the correctly rounded decimal, ties to even like Python.
pub fn round_to(x: f64, digits: usize) -> f64 {
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// `str.isprintable` for one character (a space is printable, other separators and controls are not).
pub fn is_printable(c: char) -> bool {
    c == ' ' || (!c.is_control() && !c.is_whitespace())
}

/// The caller's own text, made safe to echo back: one line, no control characters, shortened to `limit`.
pub fn snippet(text: &str, limit: usize) -> String {
    let s: String = text.chars().map(|c| if is_printable(c) { c } else { '?' }).collect();
    if s.chars().count() <= limit {
        s
    } else {
        let head: String = s.chars().take(limit.saturating_sub(3)).collect();
        format!("{head}...")
    }
}

/// `repr(str)`: quotes chosen like Python, backslashes doubled.
pub fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if is_printable(c) => out.push(c),
            c if (c as u32) < 0x100 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c if (c as u32) < 0x10000 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push_str(&format!("\\U{:08x}", c as u32)),
        }
    }
    out.push(quote);
    out
}

/// `repr(float)`-like: `120.0`, `0.1`, `1e-05`.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    let a = x.abs();
    if a != 0.0 && !(1e-4..1e16).contains(&a) {
        let e = format!("{x:e}");
        let (m, ex) = e.split_once('e').expect("exponent form");
        let ex: i32 = ex.parse().expect("exponent digits");
        return format!("{m}e{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs());
    }
    format!("{x:?}")
}

/// `repr` of a JSON value as a Python value (`None`, `True`, single quotes), cut at 80 characters, for the
/// `input_value=` part of an argument error.
pub fn py_value_repr(v: &Value) -> String {
    let s = match v {
        Value::Null => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() => py_float_repr(f),
            _ => n.to_string(),
        },
        Value::String(s) => py_repr(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_value_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter().map(|(k, v)| format!("{}: {}", py_repr(k), py_value_repr(v))).collect::<Vec<_>>().join(", ")
        ),
    };
    if s.chars().count() > 80 {
        let head: String = s.chars().take(77).collect();
        format!("{head}...")
    } else {
        s
    }
}

/// The Python type name of a JSON value (`input_type=`).
pub fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if n.is_f64() {
                "float"
            } else {
                "int"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// `True` / `False` / `None` as Python prints a bool or nothing.
pub fn py_bool(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "True",
        Some(false) => "False",
        None => "None",
    }
}

/// `difflib.get_close_matches(word, possibilities, n=1)` with the default cutoff of 0.6.
pub fn close_match(word: &str, possibilities: &[&str]) -> Option<String> {
    let w: Vec<char> = word.chars().collect();
    let mut best: Option<(f64, &str)> = None;
    for p in possibilities {
        let c: Vec<char> = p.chars().collect();
        let r = ratio(&w, &c);
        if r >= 0.6 {
            match best {
                Some((br, bs)) if br > r || (br == r && bs >= *p) => {}
                _ => best = Some((r, p)),
            }
        }
    }
    best.map(|(_, s)| s.to_string())
}

/// `difflib.SequenceMatcher(None, a, b).ratio()` (no junk heuristics: the strings are short).
pub fn ratio(a: &[char], b: &[char]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let m = matching_total(a, b);
    2.0 * m as f64 / (a.len() + b.len()) as f64
}

fn matching_total(a: &[char], b: &[char]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let (i, j, k) = longest_match(a, b);
    if k == 0 {
        return 0;
    }
    k + matching_total(&a[..i], &b[..j]) + matching_total(&a[i + k..], &b[j + k..])
}

/// The first longest common block, as difflib picks it: earliest in `a`, then earliest in `b`.
fn longest_match(a: &[char], b: &[char]) -> (usize, usize, usize) {
    let (mut best_i, mut best_j, mut best_k) = (0, 0, 0);
    for i in 0..a.len() {
        for j in 0..b.len() {
            let mut k = 0;
            while i + k < a.len() && j + k < b.len() && a[i + k] == b[j + k] {
                k += 1;
            }
            if k > best_k {
                best_i = i;
                best_j = j;
                best_k = k;
            }
        }
    }
    (best_i, best_j, best_k)
}

/// An unpredictable lowercase hex token of `bytes` bytes (`secrets.token_hex`): the std hasher is seeded by the
/// operating system for every `RandomState`. Result ids and job ids are names, not secrets.
pub fn token_hex(bytes: usize) -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut out = String::with_capacity(bytes * 2);
    while out.len() < bytes * 2 {
        let mut h = RandomState::new().build_hasher();
        h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
        if let Ok(d) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            h.write_u128(d.as_nanos());
        }
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out.truncate(bytes * 2);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_format_matches_python() {
        assert_eq!(fmt_g(0.99727), "0.99727");
        assert_eq!(fmt_g(6371.0), "6371");
        assert_eq!(fmt_g(-400.0), "-400");
        assert_eq!(fmt_g(1.026), "1.026");
        assert_eq!(fmt_g(3389.5), "3389.5");
        assert_eq!(fmt_g(12742.0), "12742");
        assert_eq!(fmt_g(0.1), "0.1");
        assert_eq!(fmt_g(2147483647.0), "2.14748e+09");
        assert_eq!(fmt_g(1e-5), "1e-05");
        assert_eq!(fmt_g(123456.0), "123456");
        assert_eq!(fmt_g(1234567.0), "1.23457e+06");
        assert_eq!(fmt_g(0.0), "0");
        assert_eq!(fmt_g(30.0), "30");
        assert_eq!(fmt_g(100.0), "100");
        assert_eq!(fmt_g(243.02), "243.02");
    }

    #[test]
    fn repr_matches_python() {
        assert_eq!(py_repr("a.png"), "'a.png'");
        assert_eq!(py_repr("C:\\a"), "'C:\\\\a'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a\nb"), "'a\\nb'");
    }

    #[test]
    fn close_matches_like_difflib() {
        let keys = ["earth", "earth-retrograde", "fast-rotator", "slow-rotator", "super-earth", "mars", "titan", "venus"];
        assert_eq!(close_match("eart", &keys).as_deref(), Some("earth"));
        assert_eq!(close_match("marz", &keys).as_deref(), Some("mars"));
        assert_eq!(close_match("nowhere", &keys), None);
        assert_eq!(close_match("turbo", &["auto", "cpu", "gpu", "hybrid"]), None);
        assert_eq!(close_match("hybird", &["auto", "cpu", "gpu", "hybrid"]).as_deref(), Some("hybrid"));
    }

    #[test]
    fn rounding_and_snippets() {
        assert_eq!(round_to(0.123456, 4), 0.1235);
        assert_eq!(round_to(30.0, 2), 30.0);
        assert_eq!(snippet("a\x1b[31mb", 40), "a?[31mb");
        assert_eq!(snippet(&"x".repeat(200), 120).chars().count(), 120);
        assert!(snippet(&"x".repeat(200), 120).ends_with("..."));
        assert_eq!(token_hex(6).len(), 12);
        assert_ne!(token_hex(6), token_hex(6));
    }
}
