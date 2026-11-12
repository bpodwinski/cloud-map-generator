//! The API's JSON: ordered values written compactly, as `json.dumps(clean(payload), separators=(',', ':'))` does.
//!
//! Answers are built with the command line's ordered [`J`] (`serde_json`'s map sorts its keys, and the order of
//! the keys is part of what a person reads in the sidecar). `clean` of the Python server turns every non-finite
//! number into `null`, because JSON has no NaN: [`compact`] does the same, and writes ASCII only (non-ASCII
//! escaped as `\uXXXX`, like `json.dumps`).

use crate::pyjson::{py_float, J};

/// A number the way Python's own value would print: an integer when the float holds one exactly (a limit of
/// `100.0` prints `100`), else the float.
pub fn num(x: f64) -> J {
    if x.is_finite() && x == x.trunc() && x.abs() < 1e15 {
        J::Int(x as i64)
    } else {
        J::Float(x)
    }
}

/// A count that may be absent (`null`).
pub fn opt_int(n: Option<i64>) -> J {
    n.map_or(J::Null, J::Int)
}

/// A float that may be absent (`null`).
pub fn opt_float(x: Option<f64>) -> J {
    x.map_or(J::Null, J::Float)
}

/// A string that may be absent (`null`).
pub fn opt_str(s: Option<&str>) -> J {
    s.map_or(J::Null, J::s)
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

fn write(j: &J, out: &mut String) {
    match j {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Int(v) => out.push_str(&v.to_string()),
        J::Float(v) if v.is_finite() => out.push_str(&py_float(*v)),
        J::Float(_) => out.push_str("null"),
        J::Str(s) => quote(s, out),
        J::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write(item, out);
            }
            out.push(']');
        }
        J::Obj(items) => {
            out.push('{');
            for (i, (k, v)) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                quote(k, out);
                out.push(':');
                write(v, out);
            }
            out.push('}');
        }
    }
}

/// The compact text of a value: no spaces, ASCII only, non-finite floats as `null`.
pub fn compact(j: &J) -> String {
    let mut out = String::new();
    write(j, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_is_json_dumps_with_tight_separators_and_clean() {
        let j = J::obj([
            ("a", J::Float(f64::NAN)),
            ("b", J::Arr(vec![J::Int(1), J::Float(2.0), J::Float(f64::INFINITY), J::Null, J::Bool(true)])),
            ("c", J::s("\u{e9}\"x\n")),
            ("d", J::obj(Vec::<(String, J)>::new())),
        ]);
        assert_eq!(compact(&j), "{\"a\":null,\"b\":[1,2.0,null,null,true],\"c\":\"\\u00e9\\\"x\\n\",\"d\":{}}");
    }

    #[test]
    fn num_keeps_integers_integral() {
        assert_eq!(compact(&num(100.0)), "100");
        assert_eq!(compact(&num(0.1)), "0.1");
        assert_eq!(compact(&num(-400.0)), "-400");
        assert_eq!(compact(&num(2147483647.0)), "2147483647");
    }
}
