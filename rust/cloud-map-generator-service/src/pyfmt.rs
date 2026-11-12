//! Small Python behaviors the API's messages and routing depend on, so that the Rust server's sentences read like
//! `ui/server.py`'s: `f'{x:g}'`, `repr(str)`, `str.capitalize()`, `difflib.get_close_matches` and
//! `urllib.parse.unquote`.

/// `f'{v:g}'` of a finite float: six significant digits, trailing zeros dropped, scientific notation (`1e+06`,
/// `1e-05`) below 1e-4 and from 1e6 on.
pub fn fmt_g(v: f64) -> String {
    if v == 0.0 {
        return if v.is_sign_negative() { "-0" } else { "0" }.to_string();
    }
    if !v.is_finite() {
        return if v.is_nan() { "nan" } else if v > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    // Six significant digits in scientific form decide the exponent (rounding can carry into the next power of ten).
    let sci = format!("{v:.5e}");
    let (mantissa, exp) = sci.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("a decimal exponent");
    if !(-4..6).contains(&exp) {
        let mantissa = trim_zeros(mantissa);
        return format!("{mantissa}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    let decimals = (5 - exp).max(0) as usize;
    trim_zeros(&format!("{v:.decimals$}"))
}

fn trim_zeros(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

/// `repr(s)` of a Python `str`: single quotes (double quotes when the text holds a single quote and no double
/// quote), backslash escapes for the quote, `\\`, `\n`, `\r`, `\t` and non-printable characters.
pub fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::new();
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
            c if !is_printable(c) => {
                let n = c as u32;
                if n < 0x100 {
                    out.push_str(&format!("\\x{n:02x}"));
                } else if n < 0x10000 {
                    out.push_str(&format!("\\u{n:04x}"));
                } else {
                    out.push_str(&format!("\\U{n:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `str.isprintable` for one character, approximated with what the standard library knows: control
/// characters, whitespace other than the plain space, and the common invisible format characters (zero-width
/// marks, bidi controls, the BOM, the soft hyphen) are not printable.
pub fn is_printable(c: char) -> bool {
    if c == ' ' {
        return true;
    }
    if c.is_control() || c.is_whitespace() {
        return false;
    }
    !matches!(c as u32, 0xAD | 0x200B..=0x200F | 0x2028..=0x202E | 0x2060..=0x2064 | 0xFEFF)
}

/// `str.isprintable()`: true for the empty string too.
pub fn is_printable_str(s: &str) -> bool {
    s.chars().all(is_printable)
}

/// `s.capitalize()`: the first character upper-cased, the rest lower-cased.
pub fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().chain(chars.flat_map(char::to_lowercase)).collect(),
    }
}

/// The number of matching characters `difflib.SequenceMatcher(None, a, b).ratio()` counts: the longest common
/// block, then the same on each side of it, recursively (no junk heuristic: the strings here are a few characters long).
fn matching_chars(a: &[char], b: &[char]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    // Longest common substring; ties go to the earliest start in `a`, then in `b` (as difflib does).
    let (mut best_len, mut best_i, mut best_j) = (0, 0, 0);
    let mut prev = vec![0usize; b.len() + 1];
    for i in 0..a.len() {
        let mut cur = vec![0usize; b.len() + 1];
        for j in 0..b.len() {
            if a[i] == b[j] {
                cur[j + 1] = prev[j] + 1;
                if cur[j + 1] > best_len {
                    best_len = cur[j + 1];
                    best_i = i + 1 - best_len;
                    best_j = j + 1 - best_len;
                }
            }
        }
        prev = cur;
    }
    if best_len == 0 {
        return 0;
    }
    best_len + matching_chars(&a[..best_i], &b[..best_j]) + matching_chars(&a[best_i + best_len..], &b[best_j + best_len..])
}

/// `difflib.get_close_matches(word, possibilities, n=1)` (cutoff 0.6): the closest possibility, or None. Ties in
/// the score go to the larger string, as the heap of `get_close_matches` orders them.
pub fn close_match<'a>(word: &str, possibilities: &[&'a str]) -> Option<&'a str> {
    let w: Vec<char> = word.chars().collect();
    let mut best: Option<(f64, &'a str)> = None;
    for &p in possibilities {
        let pc: Vec<char> = p.chars().collect();
        let total = w.len() + pc.len();
        if total == 0 {
            continue;
        }
        let score = 2.0 * matching_chars(&w, &pc) as f64 / total as f64;
        if score < 0.6 {
            continue;
        }
        if best.is_none_or(|(s, name)| score > s || (score == s && p > name)) {
            best = Some((score, p));
        }
    }
    best.map(|(_, p)| p)
}

/// `urllib.parse.unquote`: percent escapes decoded as UTF-8 (a bad sequence becomes U+FFFD, a stray `%` stays).
pub fn unquote(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    let hex = |b: u8| (b as char).to_digit(16);
    while i < bytes.len() {
        let pair = if bytes[i] == b'%' && i + 2 < bytes.len() { hex(bytes[i + 1]).zip(hex(bytes[i + 2])) } else { None };
        if let Some((hi, lo)) = pair {
            out.push((hi * 16 + lo) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g_format_is_pythons() {
        let cases: [(f64, &str); 14] = [
            (0.0, "0"),
            (1.0, "1"),
            (0.1, "0.1"),
            (0.05, "0.05"),
            (401.0, "401"),
            (500.0, "500"),
            (6371.0, "6371"),
            (100.5, "100.5"),
            (1.0e6, "1e+06"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (-0.35, "-0.35"),
        ];
        for (v, want) in cases {
            assert_eq!(fmt_g(v), want, "{v:e}");
        }
        assert_eq!(fmt_g(999999.5), "1e+06", "rounding carries into the exponent");
        assert_eq!(fmt_g(2147483648.0), "2.14748e+09");
    }

    #[test]
    fn repr_quotes_like_python() {
        assert_eq!(py_repr("turbo"), "'turbo'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(py_repr("a\nb\\"), "'a\\nb\\\\'");
        assert_eq!(py_repr("\u{1}"), "'\\x01'");
        assert_eq!(py_repr("\u{e9}"), "'\u{e9}'");
    }

    #[test]
    fn printable_follows_str_isprintable() {
        assert!(is_printable_str("thin water ice") && is_printable_str(""));
        assert!(!is_printable_str("a\tb") && !is_printable_str("a\nb") && !is_printable_str("a\u{200b}b") && !is_printable_str("a\u{a0}b"));
    }

    #[test]
    fn capitalize_lowers_the_rest() {
        assert_eq!(capitalize("fast-rotator"), "Fast-rotator");
        assert_eq!(capitalize("MARS"), "Mars");
        assert_eq!(capitalize(""), "");
    }

    #[test]
    fn close_matches_follow_difflib() {
        let modes = ["auto", "cpu", "gpu", "hybrid"];
        assert_eq!(close_match("gpu", &modes), Some("gpu"));
        assert_eq!(close_match("cpus", &modes), Some("cpu"));
        assert_eq!(close_match("hybird", &modes), Some("hybrid"));
        assert_eq!(close_match("turbo", &modes), None);
        assert_eq!(close_match("", &modes), None);
        let opts = ["match_earth", "isotropic", "experiment", "surface", "compute"];
        assert_eq!(close_match("isotropy", &opts), Some("isotropic"));
        assert_eq!(close_match("turbo", &opts), None);
    }

    #[test]
    fn unquote_decodes_percent_escapes() {
        assert_eq!(unquote("%2e%2e/secret"), "../secret");
        assert_eq!(unquote("%2E%2E%2Fsecret"), "../secret");
        assert_eq!(unquote("..%5csecret"), "..\\secret");
        assert_eq!(unquote("a%00b"), "a\u{0}b");
        assert_eq!(unquote("100%"), "100%");
        assert_eq!(unquote("%zz%4"), "%zz%4");
        assert_eq!(unquote("%C3%A9"), "\u{e9}");
        assert_eq!(unquote("%ff"), "\u{fffd}");
    }
}
