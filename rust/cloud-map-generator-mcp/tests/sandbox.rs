//! The filesystem sandbox, case by case: every way a path can try to leave the folders the server may read, and
//! the folders the server writes. Ports of the sandbox tests of the Python server.

mod common;

use std::path::{Path, PathBuf};

use cloud_map_generator_mcp::sandbox::{resolve_path, scrub, valid_result_id, valid_version, Sandbox, MAX_READ_BYTES};

use common::{link_dir, link_file, unlink_dir, TempDir};

struct Tree {
    _t: TempDir,
    out: PathBuf,
    extra: PathBuf,
    outside: PathBuf,
    sandbox: Sandbox,
}

fn tree() -> Tree {
    let t = TempDir::new("sandbox");
    let (out, extra, outside) = (t.join("out"), t.join("extra"), t.join("outside"));
    for d in [&out, &extra, &outside] {
        std::fs::create_dir_all(d).unwrap();
    }
    std::fs::write(out.join("a.png"), b"png-a").unwrap();
    std::fs::create_dir_all(out.join("sub")).unwrap();
    std::fs::write(out.join("sub").join("b.png"), b"png-b").unwrap();
    std::fs::write(extra.join("c.png"), b"png-c").unwrap();
    std::fs::write(outside.join("secret.png"), b"secret").unwrap();
    let sandbox = Sandbox::new(&out, &[extra.clone()]);
    Tree { _t: t, out, extra, outside, sandbox }
}

fn resolved(p: &Path) -> PathBuf {
    resolve_path(p).unwrap()
}

fn refusal(t: &Tree, text: &str) -> String {
    t.sandbox.resolve_readable(text, "path").expect_err(text).to_string()
}

#[test]
fn reads_relative_to_the_out_dir_then_the_read_dirs() {
    let t = tree();
    let read = |s: &str| t.sandbox.resolve_readable(s, "path").unwrap();
    assert_eq!(read("a.png"), resolved(&t.out.join("a.png")));
    assert_eq!(read("sub/b.png"), resolved(&t.out.join("sub").join("b.png")));
    assert_eq!(read("c.png"), resolved(&t.extra.join("c.png")), "found in the second root");
    assert_eq!(read(t.extra.join("c.png").to_str().unwrap()), resolved(&t.extra.join("c.png")));
    assert_eq!(read("./sub/../a.png"), resolved(&t.out.join("a.png")), "stays inside");
}

#[test]
fn dotdot_escapes_are_refused() {
    let t = tree();
    for bad in ["../outside/secret.png", "sub/../../outside/secret.png", "a.png/../../outside/secret.png"] {
        let e = refusal(&t, bad);
        assert!(e.contains("outside the readable"), "{bad}: {e}");
    }
}

#[test]
fn an_absolute_path_outside_is_refused_and_inside_a_sibling_root_is_not() {
    let t = tree();
    let e = refusal(&t, t.outside.join("secret.png").to_str().unwrap());
    assert!(e.contains("outside the readable"), "{e}");
    let ok = t.sandbox.resolve_readable(t.extra.join("c.png").to_str().unwrap(), "path").unwrap();
    assert_eq!(ok.file_name().unwrap(), "c.png");
}

#[test]
fn a_directory_link_out_of_the_sandbox_is_followed_before_the_check() {
    let t = tree();
    if !link_dir(&t.out.join("escape"), &t.outside) {
        eprintln!("cannot create a symlink or junction here: skipped");
        return;
    }
    assert!(refusal(&t, "escape/secret.png").contains("outside the readable"));
    assert!(refusal(&t, t.out.join("escape").join("secret.png").to_str().unwrap()).contains("outside the readable"));
    unlink_dir(&t.out.join("escape"));
    assert!(t.outside.join("secret.png").exists(), "the target is untouched");
}

#[test]
fn a_file_symlink_out_of_the_sandbox_is_refused() {
    let t = tree();
    if !link_file(&t.out.join("link.png"), &t.outside.join("secret.png")) {
        eprintln!("cannot create a file symlink here: skipped");
        return;
    }
    assert!(refusal(&t, "link.png").contains("outside the readable"));
}

#[test]
fn a_symlink_that_stays_inside_is_followed() {
    let t = tree();
    if !link_file(&t.out.join("alias.png"), &t.out.join("a.png")) {
        eprintln!("cannot create a file symlink here: skipped");
        return;
    }
    assert_eq!(t.sandbox.resolve_readable("alias.png", "path").unwrap(), resolved(&t.out.join("a.png")));
}

#[test]
fn network_and_device_paths_are_refused_on_every_platform() {
    let t = tree();
    for bad in ["\\\\server\\share\\x.png", "//server/share/x.png", "\\\\?\\C:\\Windows\\win.ini", "\\\\.\\pipe\\x", "//./c/x.png"] {
        assert!(refusal(&t, bad).contains("network (UNC) or device path"), "{bad}");
    }
}

#[test]
fn drive_relative_and_odd_text_is_refused() {
    let t = tree();
    assert!(refusal(&t, "C:secret.png").contains("drive-relative"));
    for bad in ["", "   ", "a\0.png", "a\n.png", &"x".repeat(2000)] {
        let e = t.sandbox.resolve_readable(bad, "path").expect_err("refused").to_string();
        assert!(!e.contains('\u{0}') && e.len() < 400, "{e:?}");
    }
    assert!(refusal(&t, "").contains("empty"));
    assert!(refusal(&t, "a\0.png").contains("control characters"));
    assert!(refusal(&t, &"x".repeat(2000)).contains("longer than 1024 characters"));
}

#[test]
fn windows_device_names_and_streams_are_refused_when_the_windows_rules_are_on() {
    let t = tree();
    let sandbox = Sandbox::new(&t.out, &[]).with_windows_rules(true);
    for bad in ["NUL", "con.png", "sub/COM1", "a.png:stream", "aux.txt.", "LPT9"] {
        assert!(sandbox.resolve_readable(bad, "path").is_err(), "{bad}");
    }
}

#[test]
fn a_missing_file_and_a_directory_say_what_is_readable() {
    let t = tree();
    let e = refusal(&t, "nope.png");
    assert!(e.contains("not an existing file inside the readable locations"), "{e}");
    assert!(refusal(&t, "sub").contains("not an existing file"), "a directory is not a file");
    assert!(e.contains("output folder"));
    let (out, parent) = (t.out.to_string_lossy().to_string(), t.out.parent().unwrap().to_string_lossy().to_string());
    assert!(!e.contains(&out) && !e.contains(&parent), "roles, never real paths: {e}");
    assert!(e.contains("1 directory given with --read-dir") || e.contains("the 1 directory"), "{e}");
}

#[test]
fn a_file_over_the_size_cap_is_never_read() {
    let t = TempDir::new("cap");
    let out = t.join("out");
    std::fs::create_dir_all(&out).unwrap();
    let big = std::fs::File::create(out.join("big.png")).unwrap();
    big.set_len(MAX_READ_BYTES + 1).unwrap();
    drop(big);
    let sandbox = Sandbox::new(&out, &[]);
    let e = sandbox.resolve_readable("big.png", "path").unwrap_err().to_string();
    assert!(e.contains("above the 64 MiB limit"), "{e}");
    let small = Sandbox::new(&out, &[]).with_max_read_bytes(10);
    std::fs::write(out.join("ok.bin"), b"0123456789").unwrap();
    let ok = small.resolve_readable("ok.bin", "path").unwrap();
    assert_eq!(small.read_bytes(&ok).unwrap(), b"0123456789");
    std::fs::write(out.join("grown.bin"), b"0123456789A").unwrap(); // grew after the stat: the read itself is bounded
    assert!(small.read_bytes(&out.join("grown.bin")).is_err());
}

#[test]
fn sibling_paths_named_inside_a_file_follow_the_same_rules() {
    let t = tree();
    let bodies = t.out.join("bodies.json");
    std::fs::write(&bodies, "{}").unwrap();
    std::fs::write(t.out.join("land.u8"), b"1234").unwrap();
    let (p, size) = t.sandbox.resolve_sibling(&bodies, "land.u8", "land").unwrap();
    assert_eq!((p.file_name().unwrap().to_str().unwrap(), size), ("land.u8", 4));
    for bad in ["../outside/secret.png", t.outside.join("secret.png").to_str().unwrap(), "\\\\srv\\x\\y", "missing.u8"] {
        assert!(t.sandbox.resolve_sibling(&bodies, bad, "land").is_err(), "{bad}");
    }
}

#[test]
fn ids_and_versions_are_validated_tokens() {
    assert!(valid_result_id("3f9a1c2b4d5e") && !valid_result_id("../x") && !valid_result_id("ABCDEF12"));
    assert!(!valid_result_id("abc") && !valid_result_id(&"a".repeat(40)) && !valid_result_id("abcdef12/"));
    assert!(valid_version("v1") && valid_version("2026-11.rc_1"));
    for bad in ["", ".", "..", "../v1", "v1/x", "v 1", &"a".repeat(33), "-v", "con", "NUL", "com1.x", "Lpt9"] {
        assert!(!valid_version(bad), "{bad}");
    }
    let e = Sandbox::new(Path::new("."), &[]).bake_dir("../etc").unwrap_err().to_string();
    assert!(e.contains("letters, digits"), "{e}");
}

#[test]
fn writes_happen_in_pending_folders_promoted_by_id() {
    let t = TempDir::new("writes");
    let out = t.join("out");
    let sandbox = Sandbox::new(&out, &[]);
    sandbox.ensure_out_dir().unwrap();
    let pending = sandbox.new_pending_dir().unwrap();
    assert_eq!(pending.parent().unwrap(), sandbox.out_dir().join(".pending"));
    std::fs::write(pending.join("f.txt"), "hello").unwrap();
    let final_dir = sandbox.promote(&pending, "abcdef123456").unwrap();
    assert_eq!(final_dir, sandbox.out_dir().join("abcdef123456"));
    assert_eq!(std::fs::read_to_string(final_dir.join("f.txt")).unwrap(), "hello");
    assert!(!sandbox.out_dir().join(".pending").exists(), "tidied");
    let again = sandbox.new_pending_dir().unwrap();
    std::fs::write(again.join("g.txt"), "new").unwrap();
    sandbox.promote(&again, "abcdef123456").unwrap(); // same id: replaced, not merged
    let names: Vec<String> = std::fs::read_dir(&final_dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
    assert_eq!(names, vec!["g.txt"]);
    let other = sandbox.new_pending_dir().unwrap();
    sandbox.discard_pending(&other);
    assert!(!other.exists());
    assert!(sandbox.promote(t.path(), "abcdef123456").is_err(), "not a pending folder");
    assert!(sandbox.result_dir("../evil").is_err());
    assert!(t.path().exists(), "the refused promote moved nothing");
}

#[test]
fn scrub_removes_absolute_paths_but_keeps_the_sentence() {
    let t = TempDir::new("scrub");
    let out = t.join("out");
    let other = t.join("other").to_string_lossy().replace('\\', "/");
    let msg = format!(
        "path: {}{}x.png is not a file; also /etc/passwd and C:\\Users\\bob\\secret.txt and {other}/y.png; use list_presets() and/or the web.",
        out.display(),
        std::path::MAIN_SEPARATOR
    );
    let text = scrub(&msg, &[out.clone()]);
    for gone in [out.to_string_lossy().as_ref(), "/etc/passwd", "bob", "other/y.png"] {
        assert!(!text.contains(gone), "{gone}: {text}");
    }
    assert!(text.starts_with("path: <out-dir>") && text.contains("list_presets() and/or the web"), "{text}");
    assert_eq!(scrub("see https://example.org/a/b for details", &[]), "see https://example.org/a/b for details");
    let read = t.join("read");
    assert_eq!(scrub(&format!("cannot read {}/m.png", read.to_string_lossy().replace('\\', "/")), &[out, read]), "cannot read <read-dir 1>/m.png");
}
