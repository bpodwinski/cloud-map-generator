//! An ordered JSON value that prints like Python's `json.dumps(value, indent=2)`.
//!
//! The sidecars and the manifest of the Python tools are `json.dumps(meta, indent=2)` of dicts whose key ORDER
//! is the order of insertion, and a person diffs them against the Python ones: `serde_json`'s default map
//! sorts its keys and its `preserve_order` feature would change the golden runner's maps when the workspace is
//! built as one, so this module keeps its own small writer. Floats print as Python's `repr` does (`1e-05`,
//! `1e+16`, `0.0`, `NaN`, `Infinity`), strings are ASCII-escaped (`ensure_ascii`), an empty list is `[]`.

/// A JSON value with ordered objects.
#[derive(Clone, Debug, PartialEq)]
pub enum J {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

impl J {
    /// A string value.
    pub fn s(text: impl Into<String>) -> J {
        J::Str(text.into())
    }
    /// An object from `(key, value)` pairs, in the given order.
    pub fn obj<K: Into<String>>(pairs: impl IntoIterator<Item = (K, J)>) -> J {
        J::Obj(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }
    /// A list of floats.
    pub fn floats(values: &[f64]) -> J {
        J::Arr(values.iter().map(|&v| J::Float(v)).collect())
    }
    /// An unsigned integer (every count and seed here fits an `i64`).
    pub fn uint(v: u64) -> J {
        J::Int(i64::try_from(v).expect("an integer that fits i64"))
    }
    /// Appends a key to an object (panics on any other value: a programming error).
    pub fn push(&mut self, key: &str, value: J) {
        match self {
            J::Obj(items) => items.push((key.to_string(), value)),
            other => panic!("push on a non-object: {other:?}"),
        }
    }

    /// Removes a key of an object and returns its value (`dict.pop(key, None)`); nothing on any other value.
    pub fn remove(&mut self, key: &str) -> Option<J> {
        match self {
            J::Obj(items) => items.iter().position(|(k, _)| k == key).map(|i| items.remove(i).1),
            _ => None,
        }
    }
}

/// Python's `repr(float)` as `json.dumps` writes it: the shortest digits that read back to the same value,
/// fixed notation for decimal exponents in `[-4, 16)`, scientific with a signed two-digit exponent outside.
pub fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    // `{:e}` prints the shortest round-trip digits: "1.2345e-5", "1e16".
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("a decimal exponent");
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                format!("{sign}{digits}{}.0", "0".repeat(int_len - digits.len()))
            } else {
                format!("{sign}{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize))
        }
    } else {
        let m = if digits.len() == 1 { digits } else { format!("{}.{}", &digits[..1], &digits[1..]) };
        format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

fn quote(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            // Python's ESCAPE_ASCII is `[^ -~]`: everything outside space..tilde, DEL included.
            c if !(0x20..=0x7e).contains(&(c as u32)) => {
                let mut units = [0u16; 2];
                for u in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write(j: &J, depth: usize, out: &mut String) {
    let pad = |n: usize| "  ".repeat(n);
    match j {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Int(v) => out.push_str(&v.to_string()),
        J::Float(v) => out.push_str(&py_float(*v)),
        J::Str(s) => quote(s, out),
        J::Arr(items) if items.is_empty() => out.push_str("[]"),
        J::Arr(items) => {
            out.push_str("[\n");
            for (i, item) in items.iter().enumerate() {
                out.push_str(&pad(depth + 1));
                write(item, depth + 1, out);
                out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad(depth));
            out.push(']');
        }
        J::Obj(items) if items.is_empty() => out.push_str("{}"),
        J::Obj(items) => {
            out.push_str("{\n");
            for (i, (k, v)) in items.iter().enumerate() {
                out.push_str(&pad(depth + 1));
                quote(k, out);
                out.push_str(": ");
                write(v, depth + 1, out);
                out.push_str(if i + 1 < items.len() { ",\n" } else { "\n" });
            }
            out.push_str(&pad(depth));
            out.push('}');
        }
    }
}

/// `json.dumps(value, indent=2)` (no trailing newline, as Python).
pub fn dumps(j: &J) -> String {
    let mut out = String::new();
    write(j, 0, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_like_python_repr() {
        // Each expected string is Python's repr(float).
        let cases: [(f64, &str); 18] = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.5, "0.5"),
            (0.99727, "0.99727"),
            (6371.0, "6371.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (0.00012345, "0.00012345"),
            (1.5e-7, "1.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345678901234568e17, "1.2345678901234568e+17"),
            (123456.789, "123456.789"),
            (-2.5, "-2.5"),
            (1e100, "1e+100"),
            (1e-300, "1e-300"),
            (0.1 + 0.2, "0.30000000000000004"),
        ];
        for (x, want) in cases {
            assert_eq!(py_float(x), want, "{x:e}");
        }
        assert_eq!(py_float(f64::NAN), "NaN");
        assert_eq!(py_float(f64::INFINITY), "Infinity");
        assert_eq!(py_float(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn dumps_matches_json_dumps_indent_2() {
        let mut o = J::obj([("seed", J::Int(42)), ("name", J::s("a\"b\u{e9}\n")), ("empty", J::Arr(vec![]))]);
        o.push("list", J::Arr(vec![J::Float(1.0), J::Null, J::Bool(true)]));
        o.push("nested", J::obj([("k", J::floats(&[0.5]))]));
        o.push("none", J::obj(Vec::<(String, J)>::new()));
        // Python: json.dumps({...}, indent=2)
        let want = "{\n  \"seed\": 42,\n  \"name\": \"a\\\"b\\u00e9\\n\",\n  \"empty\": [],\n  \"list\": [\n    1.0,\n    null,\n    true\n  ],\n  \"nested\": {\n    \"k\": [\n      0.5\n    ]\n  },\n  \"none\": {}\n}";
        assert_eq!(dumps(&o), want);
    }

    #[test]
    fn non_bmp_characters_become_surrogate_pairs() {
        assert_eq!(dumps(&J::s("\u{1F600}")), "\"\\ud83d\\ude00\"");
    }
}
