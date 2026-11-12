//! Build script of `cloud-map-generator`: embeds the web page of `cloud-map-generator serve` into the binary.
//!
//! Every file of `ui/static/` (the page, its script, anything the page grows to need) becomes an `include_bytes!`
//! entry of a generated table (`$OUT_DIR/embedded.rs`, included by `src/serve/assets.rs`), so the server is one
//! binary: `--static-dir` reads the files from disk instead. Scanning the folder, rather than naming `index.html`
//! and `app.js`, means a file the UI adds later is served without touching this crate. Earth's reference map is
//! embedded by `cloud-map-generator-service`.
//!
//! The path is relative to the crate (`../../ui/static`), the repository layout. A build from a source tree that
//! lacks it fails here, with the path in the message, instead of serving a page that is not there.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::{env, fs};

/// Every file under `dir`, as (path relative to `base` with forward slashes, absolute path), sorted. Dotfiles are left out.
fn collect(dir: &Path, base: &Path, out: &mut Vec<(String, PathBuf)>) {
    let read = fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    let mut paths: Vec<PathBuf> = read.map(|e| e.unwrap_or_else(|e| panic!("cannot list {}: {e}", dir.display())).path()).collect();
    paths.sort();
    for p in paths {
        if p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.')) {
            continue;
        }
        if p.is_dir() {
            collect(&p, base, out);
        } else if p.is_file() {
            let rel = p.strip_prefix(base).expect("a file under its folder").to_string_lossy().replace('\\', "/");
            out.push((rel, p));
        }
    }
}

/// A path as a Rust string literal with forward slashes (`include_bytes!` takes them on every platform).
fn literal(p: &Path) -> String {
    format!("{:?}", p.to_string_lossy().replace('\\', "/"))
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let static_dir = manifest.join("..").join("..").join("ui").join("static");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", static_dir.display());

    let mut files = Vec::new();
    if static_dir.is_dir() {
        collect(&static_dir, &static_dir, &mut files);
    }
    assert!(
        files.iter().any(|(rel, _)| rel == "index.html"),
        "cloud-map-generator serve embeds the page of ui/static/: {} has no index.html (build from a checkout of the repository)",
        static_dir.display()
    );

    let mut code = String::new();
    writeln!(code, "/// The files of `ui/static/`, embedded at build time: (path under the folder, bytes).").unwrap();
    writeln!(code, "pub const STATIC_FILES: &[(&str, &[u8])] = &[").unwrap();
    for (rel, path) in &files {
        writeln!(code, "    ({rel:?}, include_bytes!({})),", literal(path)).unwrap();
    }
    writeln!(code, "];").unwrap();

    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("embedded.rs");
    fs::write(&out, code).unwrap_or_else(|e| panic!("cannot write {}: {e}", out.display()));
}
