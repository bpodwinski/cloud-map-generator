//! The filesystem sandbox: what the server may read and where it may write.
//!
//! The server runs on a developer's machine on behalf of an AI, so every path that comes from a tool argument is
//! hostile until proven otherwise:
//!
//! - writes happen ONLY under the out dir, in folders the server names itself (a result id it checked, a
//!   validated bake version), never in a folder whose name came from the caller;
//! - reads happen ONLY under the out dir and the `--read-dir` roots. A path is resolved first (symbolic links and
//!   `..` followed) and the RESOLVED path must sit inside a root;
//! - network paths (UNC, `\\?\` device paths), drive-relative paths, alternate data streams and Windows device
//!   names are refused on every platform, not only where they are dangerous;
//! - a file larger than [`MAX_READ_BYTES`] is never read.
//!
//! Standard library only. Messages are written for an AI reader: what was wrong and what to do. They name the
//! readable locations by role, never by real path.

use std::fs;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

use crate::errors::SandboxError;
use crate::pyfmt::{py_repr, snippet, token_hex};

pub const MAX_READ_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_PATH_CHARS: usize = 1024;

const WINDOWS_DEVICES: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

fn is_device_name(name: &str) -> bool {
    WINDOWS_DEVICES.contains(&name.to_uppercase().as_str())
}

/// A result id is a lowercase hex token of 8 to 32 characters. It is validated here and then against the cache.
pub fn valid_result_id(value: &str) -> bool {
    (8..=32).contains(&value.len()) && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A bake version names a folder: letters, digits, `.`, `_` and `-`, starting with a letter or digit.
pub fn valid_version(value: &str) -> bool {
    let b = value.as_bytes();
    if b.is_empty() || b.len() > 32 || !b[0].is_ascii_alphanumeric() {
        return false;
    }
    if !b.iter().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-')) {
        return false;
    }
    // 'con' or 'nul.x' names no folder on Windows
    !is_device_name(value.split('.').next().unwrap_or(""))
}

// ------------------------------------------------------------------------------------------------ resolving

fn has_prefix(p: &Path) -> bool {
    matches!(p.components().next(), Some(Component::Prefix(_)))
}

fn is_rooted(p: &Path) -> bool {
    p.has_root() || has_prefix(p)
}

/// `\\?\C:\x` as `C:\x`: a junction target read back from the disk is in verbatim form.
fn strip_verbatim(p: PathBuf) -> PathBuf {
    if !cfg!(windows) {
        return p;
    }
    let s = p.to_string_lossy().to_string();
    for lead in ["\\\\?\\", "\\??\\"] {
        if let Some(rest) = s.strip_prefix(lead) {
            let b = rest.as_bytes();
            if b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'\\' || b[2] == b'/') {
                return PathBuf::from(rest);
            }
        }
    }
    p
}

fn make_absolute(p: &Path) -> io::Result<PathBuf> {
    if p.is_absolute() {
        return Ok(p.to_path_buf());
    }
    let cwd = std::env::current_dir()?;
    if p.has_root() {
        // Windows `\dir`: the root of the current drive
        let mut out = PathBuf::new();
        if let Some(Component::Prefix(pre)) = cwd.components().next() {
            out.push(pre.as_os_str());
        }
        out.push(p);
        return Ok(out);
    }
    Ok(cwd.join(p))
}

fn resolve_inner(path: &Path, depth: u32) -> io::Result<PathBuf> {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(comp.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(name) => {
                out.push(name);
                if let Ok(meta) = fs::symlink_metadata(&out) {
                    if meta.file_type().is_symlink() {
                        if depth >= 40 {
                            return Err(io::Error::other("too many levels of symbolic links"));
                        }
                        let target = strip_verbatim(fs::read_link(&out)?);
                        out.pop();
                        let joined = if is_rooted(&target) { target } else { out.join(target) };
                        out = resolve_inner(&make_absolute(&joined)?, depth + 1)?;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// `Path.resolve(strict=False)`: an absolute path with `.` and `..` removed and every symbolic link (and
/// junction) of the part that exists followed, component by component; what does not exist is kept as written.
pub fn resolve_path(path: &Path) -> io::Result<PathBuf> {
    resolve_inner(&make_absolute(path)?, 0)
}

fn comp_eq(a: &Component<'_>, b: &Component<'_>) -> bool {
    if cfg!(windows) {
        a.as_os_str().to_string_lossy().to_lowercase() == b.as_os_str().to_string_lossy().to_lowercase()
    } else {
        a == b
    }
}

/// Whether an already RESOLVED `path` is `root` or sits under it (case-insensitive on Windows).
pub fn is_within(path: &Path, root: &Path) -> bool {
    let a: Vec<_> = path.components().collect();
    let r: Vec<_> = root.components().collect();
    a.len() >= r.len() && r.iter().zip(a.iter()).all(|(x, y)| comp_eq(x, y))
}

fn same_path(a: &Path, b: &Path) -> bool {
    is_within(a, b) && is_within(b, a)
}

fn err<T>(message: String) -> Result<T, SandboxError> {
    Err(SandboxError(message))
}

// ------------------------------------------------------------------------------------------------ the sandbox

/// Resolves the paths a tool may read and names the folders it may write.
#[derive(Debug, Clone)]
pub struct Sandbox {
    out_dir: PathBuf,
    roots: Vec<PathBuf>,
    max_read_bytes: u64,
    windows_rules: bool,
}

impl Sandbox {
    pub fn new(out_dir: &Path, read_dirs: &[PathBuf]) -> Self {
        let out = resolve_path(out_dir).unwrap_or_else(|_| out_dir.to_path_buf());
        let mut roots = vec![out.clone()];
        for d in read_dirs {
            let r = resolve_path(d).unwrap_or_else(|_| d.to_path_buf());
            if !roots.iter().any(|x| same_path(x, &r)) {
                roots.push(r);
            }
        }
        Sandbox { out_dir: out, roots, max_read_bytes: MAX_READ_BYTES, windows_rules: cfg!(windows) }
    }

    /// A lower read cap (tests).
    pub fn with_max_read_bytes(mut self, n: u64) -> Self {
        self.max_read_bytes = n;
        self
    }

    /// Apply the Windows-only refusals (device names, alternate data streams) whatever the platform (tests).
    pub fn with_windows_rules(mut self, on: bool) -> Self {
        self.windows_rules = on;
        self
    }

    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }

    /// The out dir first, then the `--read-dir` directories.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    pub fn max_read_bytes(&self) -> u64 {
        self.max_read_bytes
    }

    // ---- reading

    /// The readable locations for an error message, by role only: the real paths stay out of the text.
    pub fn describe_roots(&self) -> String {
        let n = self.roots.len() - 1;
        let mut text = String::from(
            "the server's output folder (its path is output_folder in list_presets and the `folder` of a generate_map result)",
        );
        if n > 0 {
            text.push_str(&format!(" and the {n} director{} given with --read-dir", if n == 1 { "y" } else { "ies" }));
        }
        text
    }

    /// Whether an already RESOLVED path sits inside a readable root.
    pub fn contains(&self, path: &Path) -> bool {
        self.roots.iter().any(|r| is_within(path, r))
    }

    /// The textual refusals, independent of the disk.
    fn check_text(&self, raw: &str, what: &str) -> Result<(), SandboxError> {
        if raw.trim().is_empty() {
            return err(format!(
                "{what} is empty. Pass a file path such as \"coverage.png\" (relative to the output folder) or an \
                 absolute path inside a readable directory."
            ));
        }
        if raw.chars().count() > MAX_PATH_CHARS {
            return err(format!("{what} is longer than {MAX_PATH_CHARS} characters."));
        }
        if raw.chars().any(|c| (c as u32) < 32) {
            return err(format!("{what} contains control characters (NUL, newline, ...). Pass a plain file path."));
        }
        let unified = raw.replace('\\', "/");
        if unified.starts_with("//") {
            return err(format!(
                "{what} {} is a network (UNC) or device path. Those are never read; copy the file into the output \
                 folder or a --read-dir directory and pass that path.",
                py_repr(&snippet(raw, 120))
            ));
        }
        let chars: Vec<char> = raw.chars().collect();
        let has_drive = chars.len() >= 2 && chars[1] == ':' && chars[0].is_alphabetic();
        if has_drive && !(chars.len() >= 3 && (chars[2] == '/' || chars[2] == '\\')) {
            return err(format!(
                "{what} {} is a drive-relative path (a drive letter without a slash). Pass a full path such as \
                 C:\\dir\\file.png or a path relative to the output folder.",
                py_repr(&snippet(raw, 120))
            ));
        }
        if self.windows_rules {
            let rest: String = if has_drive { chars[2..].iter().collect() } else { raw.to_string() };
            if rest.contains(':') {
                return err(format!(
                    "{what} {} contains \":\" after the drive letter (alternate data streams and device syntax are \
                     refused).",
                    py_repr(&snippet(raw, 120))
                ));
            }
            for part in unified.split('/') {
                let stem = part.trim_end_matches([' ', '.']).split('.').next().unwrap_or("");
                if is_device_name(stem) {
                    return err(format!(
                        "{what} {} names a Windows device ({part}). Pass a real file.",
                        py_repr(&snippet(raw, 120))
                    ));
                }
            }
        }
        Ok(())
    }

    /// The resolved path of an existing regular file inside a readable root, or a [`SandboxError`].
    ///
    /// A relative path is tried against each root in turn (the out dir first). An absolute path is used as
    /// given. Either way the path is resolved (symbolic links and `..` followed) BEFORE the containment check, so
    /// a link or a `..` cannot lead out.
    pub fn resolve_readable(&self, text: &str, what: &str) -> Result<PathBuf, SandboxError> {
        self.check_text(text, what)?;
        let p = Path::new(text);
        let candidates: Vec<PathBuf> =
            if is_rooted(p) { vec![p.to_path_buf()] } else { self.roots.iter().map(|r| r.join(p)).collect() };
        let mut escaped = false;
        for cand in candidates {
            let Ok(resolved) = resolve_path(&cand) else { continue };
            if !self.contains(&resolved) {
                escaped = true;
                continue;
            }
            let Ok(meta) = fs::metadata(&resolved) else { continue };
            if !meta.is_file() {
                continue;
            }
            if meta.len() > self.max_read_bytes {
                return err(format!(
                    "{what} {} is {:.0} MiB, above the {} MiB limit this server reads.",
                    py_repr(&snippet(text, 120)),
                    meta.len() as f64 / (1u64 << 20) as f64,
                    self.max_read_bytes / (1u64 << 20)
                ));
            }
            return Ok(resolved);
        }
        if escaped {
            return err(format!(
                "{what} {} resolves outside the readable locations (symbolic links and \"..\" are followed before \
                 the check). Readable: {}. This is refused on purpose: do not retry with another path outside them. \
                 If the file is a map the user wants measured, ask the user to copy it into the output folder (or to \
                 start the server with --read-dir <directory>).",
                py_repr(&snippet(text, 120)),
                self.describe_roots()
            ));
        }
        err(format!(
            "{what} {} is not an existing file inside the readable locations: {}. Pass a path relative to the \
             output folder or an absolute path.",
            py_repr(&snippet(text, 120)),
            self.describe_roots()
        ))
    }

    /// Read a resolved file, never more than the size limit (the file may have grown since the check).
    pub fn read_bytes(&self, path: &Path) -> Result<Vec<u8>, SandboxError> {
        let file = fs::File::open(path).map_err(|_| SandboxError("the file could not be read.".to_string()))?;
        let mut data = Vec::new();
        file.take(self.max_read_bytes + 1)
            .read_to_end(&mut data)
            .map_err(|_| SandboxError("the file could not be read.".to_string()))?;
        if data.len() as u64 > self.max_read_bytes {
            return err(format!("file is larger than the {} MiB limit this server reads.", self.max_read_bytes / (1u64 << 20)));
        }
        Ok(data)
    }

    /// A path named INSIDE a file the caller supplied (a land grid in a bodies file), relative to that file. It
    /// is held to the same rules as a path the caller typed: it must resolve inside a readable root. Returns the
    /// resolved path and the size of the file.
    pub fn resolve_sibling(&self, base_file: &Path, text: &str, what: &str) -> Result<(PathBuf, u64), SandboxError> {
        self.check_text(text, what)?;
        let p = Path::new(text);
        let cand = if is_rooted(p) { p.to_path_buf() } else { base_file.parent().unwrap_or(Path::new("")).join(p) };
        let Ok(resolved) = resolve_path(&cand) else {
            return err(format!("{what} {} cannot be resolved.", py_repr(&snippet(text, 120))));
        };
        if !self.contains(&resolved) {
            return err(format!(
                "{what} {} resolves outside the readable locations: {}. Keep the data files next to the bodies file \
                 or inside a --read-dir directory.",
                py_repr(&snippet(text, 120)),
                self.describe_roots()
            ));
        }
        let meta = match fs::metadata(&resolved) {
            Ok(m) if m.is_file() => m,
            Ok(_) => return err(format!("{what} {} is not an existing file.", py_repr(&snippet(text, 120)))),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                return err(format!("{what} {} is not an existing file.", py_repr(&snippet(text, 120))))
            }
            Err(_) => return err(format!("{what} {} cannot be read.", py_repr(&snippet(text, 120)))),
        };
        if meta.len() > self.max_read_bytes {
            return err(format!(
                "{what} {} is {:.0} MiB, above the {} MiB limit.",
                py_repr(&snippet(text, 120)),
                meta.len() as f64 / (1u64 << 20) as f64,
                self.max_read_bytes / (1u64 << 20)
            ));
        }
        Ok((resolved, meta.len()))
    }

    // ---- writing

    pub fn ensure_out_dir(&self) -> Result<&Path, SandboxError> {
        fs::create_dir_all(&self.out_dir)
            .map_err(|_| SandboxError("the output folder could not be created. Ask the user to check --out-dir.".into()))?;
        Ok(&self.out_dir)
    }

    /// `<out>/<id>`: the id was produced by the engine and is validated as a hex token.
    pub fn result_dir(&self, result_id: &str) -> Result<PathBuf, SandboxError> {
        if !valid_result_id(result_id) {
            return err("internal: the result id is not a hex token".to_string());
        }
        Ok(self.out_dir.join(result_id))
    }

    fn pending_root(&self) -> PathBuf {
        self.out_dir.join(".pending")
    }

    /// A scratch folder `<out>/.pending/<token>` for a run whose id is not known yet.
    pub fn new_pending_dir(&self) -> Result<PathBuf, SandboxError> {
        let d = self.pending_root().join(token_hex(8));
        fs::create_dir_all(self.pending_root())
            .and_then(|_| fs::create_dir(&d))
            .map_err(|_| SandboxError("the output folder is not writable. Ask the user to check --out-dir.".into()))?;
        Ok(d)
    }

    /// Move a finished pending folder to `<out>/<id>`, replacing an older folder of the same id.
    pub fn promote(&self, pending: &Path, result_id: &str) -> Result<PathBuf, SandboxError> {
        let target = self.result_dir(result_id)?;
        let resolved = resolve_path(pending).unwrap_or_else(|_| pending.to_path_buf());
        if !is_within(&resolved, &self.pending_root()) {
            return err("internal: refusing to move a folder that is not a pending run".to_string());
        }
        if target.exists() {
            let _ = fs::remove_dir_all(&target);
        }
        fs::rename(pending, &target).map_err(|_| SandboxError("the result folder could not be created.".into()))?;
        self.tidy_pending();
        Ok(target)
    }

    /// Remove a pending folder after a failed or cancelled run (only ever under `.pending`).
    pub fn discard_pending(&self, pending: &Path) {
        let resolved = resolve_path(pending).unwrap_or_else(|_| pending.to_path_buf());
        if is_within(&resolved, &self.pending_root()) {
            let _ = fs::remove_dir_all(pending);
        }
        self.tidy_pending();
    }

    /// Remove the `.pending` folder once nothing runs in it (`remove_dir` only succeeds on an empty folder).
    fn tidy_pending(&self) {
        let _ = fs::remove_dir(self.pending_root());
    }

    /// `<out>/bake/<version>` for a validated version string.
    pub fn bake_dir(&self, version: &str) -> Result<PathBuf, SandboxError> {
        if !valid_version(version) {
            return err(format!(
                "version {} is not allowed: use 1 to 32 letters, digits, \".\" \"_\" or \"-\", starting with a letter \
                 or digit (for example \"v1\").",
                py_repr(&snippet(version, 120))
            ));
        }
        Ok(self.out_dir.join("bake").join(version))
    }
}

// ------------------------------------------------------------------------------------------------ scrubbing

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_path_seg_char(c: char) -> bool {
    is_word(c) || matches!(c, '.' | '-' | '~')
}

/// Remove absolute filesystem paths from a message that came from somewhere else (the engine, a library). The
/// configured roots become `<out-dir>` / `<read-dir N>`; any other absolute path becomes `<path>`.
pub fn scrub(message: &str, roots: &[PathBuf]) -> String {
    let mut text = message.to_string();
    for (i, r) in roots.iter().enumerate() {
        let label = if i == 0 { "<out-dir>".to_string() } else { format!("<read-dir {i}>") };
        let native = r.to_string_lossy().to_string();
        let forms = [native.clone(), native.replace('\\', "/")];
        for f in forms {
            if !f.is_empty() {
                text = text.replace(&f, &label);
            }
        }
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let behind_blocks = i > 0 && (is_word(chars[i - 1]) || matches!(chars[i - 1], '.' | ':' | '/' | '\\'));
        if !behind_blocks {
            // a Windows path: C:\... or C:/...
            if chars[i].is_ascii_alphabetic()
                && i + 2 < chars.len()
                && chars[i + 1] == ':'
                && (chars[i + 2] == '\\' || chars[i + 2] == '/')
            {
                let mut j = i + 3;
                while j < chars.len() && !(chars[j].is_whitespace() || matches!(chars[j], '\'' | '"' | '<' | '>' | '|' | '*' | '?')) {
                    j += 1;
                }
                out.push_str("<path>");
                i = j;
                continue;
            }
            // a POSIX path: /seg/seg
            if chars[i] == '/' {
                let mut j = i + 1;
                let mut segments = 0;
                loop {
                    let mut k = j;
                    while k < chars.len() && is_path_seg_char(chars[k]) {
                        k += 1;
                    }
                    if k > j && k < chars.len() && chars[k] == '/' {
                        segments += 1;
                        j = k + 1;
                    } else {
                        break;
                    }
                }
                if segments >= 1 {
                    while j < chars.len() && is_path_seg_char(chars[j]) {
                        j += 1;
                    }
                    out.push_str("<path>");
                    i = j;
                    continue;
                }
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_keeps_urls_and_sentences() {
        assert_eq!(scrub("see https://example.org/a/b for details", &[]), "see https://example.org/a/b for details");
        assert_eq!(scrub("use list_presets() and/or the web.", &[]), "use list_presets() and/or the web.");
        assert_eq!(scrub("also /etc/passwd and C:\\Users\\bob\\secret.txt end", &[]), "also <path> and <path> end");
    }

    #[test]
    fn tokens_are_validated() {
        assert!(valid_result_id("3f9a1c2b4d5e"));
        assert!(!valid_result_id("../x") && !valid_result_id("ABCDEF12") && !valid_result_id("abc"));
        assert!(!valid_result_id(&"a".repeat(40)) && !valid_result_id("abcdef12\n"));
        assert!(valid_version("v1") && valid_version("2026-11.rc_1"));
        for bad in ["", ".", "..", "../v1", "v1/x", "v 1", &"a".repeat(33), "-v", "con", "NUL", "com1.x", "Lpt9"] {
            assert!(!valid_version(bad), "{bad}");
        }
    }
}
