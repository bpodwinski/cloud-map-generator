//! Server configuration: command-line flags, environment, and the default output directory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use clap::Parser;

use crate::sandbox::resolve_path;

pub const OUT_ENV: &str = "PLANET_CLOUDGEN_OUT";
pub const DEFAULT_MAX_WIDTH: u32 = 2048;
pub const DEFAULT_PORT: u16 = 8766;
/// The HTTP transport never binds anything else.
pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const LOG_LEVELS: [&str; 4] = ["debug", "info", "warning", "error"];

/// The command line of `cloud-map-generator-mcp`.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "cloud-map-generator-mcp",
    version,
    about = "MCP server for cloud-map-generator: physics-driven global cloud-coverage maps for any planet. Speaks MCP over stdio by default, or over local HTTP with --http.",
    after_help = "Tools: list_presets, derive_physics, generate_map, start_generation, get_generation, cancel_generation, view_map, measure_map, bake_bodies, explain_model."
)]
pub struct Cli {
    /// the only directory the server writes to (env PLANET_CLOUDGEN_OUT; default: <user data dir>/cloud-map-generator/output, created on first use)
    #[arg(long, value_name = "DIR")]
    pub out_dir: Option<PathBuf>,

    /// largest map width a tool may ask for, one of 256, 512, 1024, 2048, 4096 (default 2048)
    #[arg(long, value_name = "PX", default_value_t = i64::from(DEFAULT_MAX_WIDTH), allow_hyphen_values = true)]
    pub max_width: i64,

    /// an extra directory measure_map, view_map and bake_bodies may read; repeatable (the out dir is always readable)
    #[arg(long = "read-dir", value_name = "DIR")]
    pub read_dir: Vec<PathBuf>,

    /// serve streamable HTTP on 127.0.0.1 instead of stdio
    #[arg(long)]
    pub http: bool,

    /// port of the HTTP transport (default 8766)
    #[arg(long, default_value_t = i64::from(DEFAULT_PORT), allow_hyphen_values = true)]
    pub port: i64,

    /// serve the deterministic MOCK engine (fabricated maps, for trying a client without the real engine) instead of the real one; the environment variable PLANET_CLOUDGEN_MOCK_GPU=1 makes it pretend to have a GPU
    #[arg(long)]
    pub mock: bool,

    /// threads of the CPU engine (env CLOUD_MAP_GENERATOR_THREADS; default: every logical core); maps are bit-identical for any count
    #[arg(long, value_name = "N", value_parser = parse_threads)]
    pub threads: Option<usize>,

    /// log verbosity on stderr (stdout carries the protocol and is never logged to)
    #[arg(long, default_value = "warning", value_parser = clap::builder::PossibleValuesParser::new(LOG_LEVELS))]
    pub log_level: String,
}

/// `--threads N`: a whole number of at least 1.
fn parse_threads(text: &str) -> Result<usize, String> {
    cloud_map_generator_service::core::par::parse_threads(text, "--threads")
}

/// What a server instance was started with.
///
/// `out_dir` is the only place the server writes. It is created on first use, not at startup, so a server that
/// is only asked to explain or derive never touches the disk.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerConfig {
    pub out_dir: PathBuf,
    pub max_width: u32,
    pub read_dirs: Vec<PathBuf>,
    pub http: bool,
    pub port: u16,
    pub log_level: String,
}

impl ServerConfig {
    /// A configuration with the defaults: stdio, width 2048, no extra read directories.
    pub fn new(out_dir: impl Into<PathBuf>) -> Self {
        ServerConfig {
            out_dir: out_dir.into(),
            max_width: DEFAULT_MAX_WIDTH,
            read_dirs: Vec::new(),
            http: false,
            port: DEFAULT_PORT,
            log_level: "warning".to_string(),
        }
    }

    pub fn with_max_width(mut self, w: u32) -> Self {
        self.max_width = w;
        self
    }

    pub fn with_read_dirs(mut self, dirs: Vec<PathBuf>) -> Self {
        self.read_dirs = dirs;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Windows,
    MacOs,
    Other,
}

impl Platform {
    pub fn current() -> Platform {
        if cfg!(windows) {
            Platform::Windows
        } else if cfg!(target_os = "macos") {
            Platform::MacOs
        } else {
            Platform::Other
        }
    }
}

/// The per-user data directory of a platform, without any third-party helper.
pub fn user_data_dir_for(platform: Platform, env: &dyn Fn(&str) -> Option<String>, home: &Path) -> PathBuf {
    match platform {
        Platform::Windows => match env("LOCALAPPDATA").filter(|s| !s.is_empty()) {
            Some(base) => PathBuf::from(base),
            None => home.join("AppData").join("Local"),
        },
        Platform::MacOs => home.join("Library").join("Application Support"),
        Platform::Other => match env("XDG_DATA_HOME").filter(|s| Path::new(s).is_absolute()) {
            Some(base) => PathBuf::from(base),
            None => home.join(".local").join("share"),
        },
    }
}

fn home_dir() -> PathBuf {
    std::env::home_dir().unwrap_or_else(|| PathBuf::from("."))
}

fn real_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// `<user data dir>/cloud-map-generator/output`.
pub fn default_out_dir_for(platform: Platform, env: &dyn Fn(&str) -> Option<String>, home: &Path) -> PathBuf {
    user_data_dir_for(platform, env, home).join("cloud-map-generator").join("output")
}

pub fn default_out_dir() -> PathBuf {
    default_out_dir_for(Platform::current(), &real_env, &home_dir())
}

fn expand_user(p: &Path) -> PathBuf {
    let s = p.to_string_lossy();
    if s == "~" {
        home_dir()
    } else if let Some(rest) = s.strip_prefix("~/").or_else(|| s.strip_prefix("~\\")) {
        home_dir().join(rest)
    } else {
        p.to_path_buf()
    }
}

/// Validate parsed flags into a config. The error is a message fit for a terminal.
pub fn config_from_cli(cli: &Cli, widths: &[u32], environ: &HashMap<String, String>) -> Result<ServerConfig, String> {
    if !widths.iter().any(|w| i64::from(*w) == cli.max_width) {
        let list: Vec<String> = widths.iter().map(u32::to_string).collect();
        return Err(format!("--max-width must be one of {}, got {}", list.join(", "), cli.max_width));
    }
    if !(1..=65535).contains(&cli.port) {
        return Err(format!("--port must be between 1 and 65535, got {}", cli.port));
    }
    let chosen = cli
        .out_dir
        .clone()
        .or_else(|| environ.get(OUT_ENV).filter(|s| !s.is_empty()).map(PathBuf::from))
        .unwrap_or_else(default_out_dir);
    let expanded = expand_user(&chosen);
    let out = resolve_path(&expanded).unwrap_or(expanded);
    if out.exists() && !out.is_dir() {
        return Err(format!("the out dir {} exists and is not a directory", out.display()));
    }
    let mut read_dirs: Vec<PathBuf> = Vec::new();
    for d in &cli.read_dir {
        let expanded = expand_user(d);
        let p = resolve_path(&expanded).unwrap_or(expanded);
        if !p.is_dir() {
            return Err(format!("--read-dir {} is not an existing directory", d.display()));
        }
        if !read_dirs.contains(&p) {
            read_dirs.push(p);
        }
    }
    Ok(ServerConfig {
        out_dir: out,
        max_width: cli.max_width as u32,
        read_dirs,
        http: cli.http,
        port: cli.port as u16,
        log_level: cli.log_level.clone(),
    })
}

/// The process environment as a map (only what the configuration reads).
pub fn process_environ() -> HashMap<String, String> {
    std::env::vars().filter(|(k, _)| k == OUT_ENV).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::WIDTHS;

    fn parse(argv: &[&str], environ: &[(&str, &str)]) -> Result<ServerConfig, String> {
        let cli = Cli::try_parse_from(std::iter::once("cloud-map-generator-mcp").chain(argv.iter().copied())).map_err(|e| e.to_string())?;
        let env: HashMap<String, String> = environ.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        config_from_cli(&cli, &WIDTHS, &env)
    }

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("cloud-map-generator-mcp-config-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn the_default_out_dir_is_under_the_user_data_dir() {
        let home = Path::new("/home/u");
        let none = |_: &str| None;
        let d = default_out_dir_for(Platform::Other, &none, home);
        assert_eq!(d, home.join(".local").join("share").join("cloud-map-generator").join("output"));
        let xdg = |k: &str| (k == "XDG_DATA_HOME").then(|| if cfg!(windows) { "C:\\data".to_string() } else { "/data".to_string() });
        assert!(default_out_dir_for(Platform::Other, &xdg, home).starts_with(if cfg!(windows) { "C:\\data" } else { "/data" }));
        let local = |k: &str| (k == "LOCALAPPDATA").then(|| "C:\\Users\\u\\AppData\\Local".to_string());
        assert!(default_out_dir_for(Platform::Windows, &local, home).ends_with("cloud-map-generator/output"));
        assert_eq!(
            default_out_dir_for(Platform::MacOs, &none, home),
            home.join("Library").join("Application Support").join("cloud-map-generator").join("output")
        );
        assert!(!default_out_dir().exists() || default_out_dir().is_dir(), "never created at startup");
    }

    #[test]
    fn the_out_dir_flag_beats_the_environment_beats_the_default() {
        let t = temp("out");
        let (flag, env) = (t.join("flag"), t.join("env"));
        let c = parse(&["--out-dir", flag.to_str().unwrap()], &[(OUT_ENV, env.to_str().unwrap())]).unwrap();
        assert_eq!(c.out_dir, resolve_path(&flag).unwrap());
        let c = parse(&[], &[(OUT_ENV, env.to_str().unwrap())]).unwrap();
        assert_eq!(c.out_dir, resolve_path(&env).unwrap());
        let c = parse(&[], &[]).unwrap();
        assert_eq!(c.out_dir, resolve_path(&default_out_dir()).unwrap());
        assert!(!flag.exists(), "created on first use, not at startup");
    }

    #[test]
    fn the_flag_validation_messages() {
        let t = temp("flags");
        assert_eq!(parse(&["--max-width", "300"], &[]).unwrap_err(), "--max-width must be one of 256, 512, 1024, 2048, 4096, got 300");
        assert!(parse(&["--port", "70000"], &[]).unwrap_err().starts_with("--port must be between 1 and 65535"));
        let e = parse(&["--read-dir", t.join("nope").to_str().unwrap()], &[]).unwrap_err();
        assert!(e.contains("--read-dir") && e.contains("is not an existing directory"), "{e}");
        let f = t.join("file");
        std::fs::write(&f, "x").unwrap();
        assert!(parse(&["--out-dir", f.to_str().unwrap()], &[]).unwrap_err().contains("is not a directory"));
        assert!(parse(&["--log-level", "loud"], &[]).is_err());
    }

    #[test]
    fn read_dirs_repeat_and_deduplicate() {
        let t = temp("read");
        let (a, b) = (t.join("a"), t.join("b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        let c = parse(
            &["--read-dir", a.to_str().unwrap(), "--read-dir", b.to_str().unwrap(), "--read-dir", a.to_str().unwrap()],
            &[],
        )
        .unwrap();
        assert_eq!(c.read_dirs, vec![resolve_path(&a).unwrap(), resolve_path(&b).unwrap()]);
        assert_eq!((c.max_width, c.port, c.http), (2048, 8766, false));
    }
}
