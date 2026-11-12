//! `cloud-map-generator serve`: the local web UI of the cloud map generator, served by the Rust engine (`ui/server.py`, rewritten).
//!
//!     cloud-map-generator serve [--port 8765] [--open] [--max-width 4096] [--static-dir <dir>]
//!
//! One binary: the page (`ui/static/`) and Earth's reference map are embedded at build time. The server listens on
//! `127.0.0.1` only and answers the API documented in `ui/README.md`, plus a `compute` block that says where the
//! work ran: `auto`, `cpu`, `gpu` or `hybrid` (the engine and the capabilities are in `serve/engine.rs`).

use crate::fail::{Fail, Result};
use crate::project::Project;
use crate::serve::engine::Runtime;
use crate::serve::{http, AppState, Config, DEFAULT_KEEP};
use clap::Args;
use std::path::PathBuf;
use std::sync::Arc;

/// The widest map the server offers (the CPU engine's memory grows about 4x per doubling: see the CLI README).
const WIDEST_ALLOWED: usize = 32768;

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// The port on 127.0.0.1 (0: any free port, the address is printed).
    #[arg(long, default_value_t = 8765)]
    pub port: u16,
    /// Open the page in the default browser.
    #[arg(long)]
    pub open: bool,
    /// The widest map the API offers; the widths are the powers of two from 512 up to it (and it, when it is
    /// not one). Ignored when --widths is given.
    #[arg(long, default_value_t = 32768)]
    pub max_width: usize,
    /// The map widths the API accepts, comma separated (each even, 64 to 16384). A small one such as 256 is only
    /// useful to try the API quickly.
    #[arg(long, value_delimiter = ',')]
    pub widths: Option<Vec<usize>>,
    /// Serve the page from this folder instead of the one embedded in the binary.
    #[arg(long)]
    pub static_dir: Option<PathBuf>,
    /// Earth's reference map (a gray PNG, 2048 x 1024) instead of the embedded one.
    #[arg(long)]
    pub reference: Option<PathBuf>,
    /// How many results are kept in memory.
    #[arg(long, default_value_t = DEFAULT_KEEP)]
    pub keep: usize,
    /// No log lines on stderr.
    #[arg(long)]
    pub quiet: bool,
}

/// The widths the page lists: `--widths` as given, else 512, 1024, ... up to `max_width` (and `max_width` itself
/// when it is not a power of two times 512).
pub fn widths_for(max_width: usize, explicit: Option<&[usize]>) -> std::result::Result<Vec<usize>, String> {
    if explicit.is_none() && max_width > WIDEST_ALLOWED {
        return Err(format!("--max-width: {max_width} is wider than the {WIDEST_ALLOWED} px this server offers"));
    }
    let widths: Vec<usize> = match explicit {
        Some(list) => list.to_vec(),
        None => {
            let mut v: Vec<usize> = std::iter::successors(Some(512usize), |w| w.checked_mul(2)).take_while(|&w| w <= max_width).collect();
            if v.last() != Some(&max_width) {
                v.push(max_width);
            }
            v
        }
    };
    let what = if explicit.is_some() { "--widths" } else { "--max-width" };
    if widths.is_empty() {
        return Err(format!("{what}: no width"));
    }
    if let Some(w) = widths.iter().find(|&&w| w < 64 || w % 2 != 0 || w > WIDEST_ALLOWED) {
        return Err(format!("{what}: {w} is not an even width from 64 to {WIDEST_ALLOWED}"));
    }
    Ok(widths)
}

#[cfg(windows)]
fn open_in_browser(url: &str) {
    let _ = std::process::Command::new("cmd").args(["/c", "start", "", url]).spawn();
}

#[cfg(target_os = "macos")]
fn open_in_browser(url: &str) {
    let _ = std::process::Command::new("open").arg(url).spawn();
}

#[cfg(not(any(windows, target_os = "macos")))]
fn open_in_browser(url: &str) {
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

pub fn run(a: ServeArgs, project: &Project) -> Result<()> {
    let widths = widths_for(a.max_width, a.widths.as_deref()).map_err(Fail::new)?;
    cloud_map_generator_service::machine::set_output_dir(&project.output());
    if a.keep == 0 {
        return Err(Fail::new("--keep: at least one result must be kept"));
    }
    let listener = std::net::TcpListener::bind(("127.0.0.1", a.port)).map_err(|e| Fail::new(format!("cannot listen on 127.0.0.1:{}: {e}", a.port)))?;
    let port = listener.local_addr().map_or(a.port, |addr| addr.port());
    // The probe of the GPU adapter takes a second or two: the address is printed after it, when the server is ready.
    let runtime = Runtime::probe();
    let config = Config { widths, keep: a.keep, static_dir: a.static_dir, reference: a.reference, quiet: a.quiet };
    let state = Arc::new(AppState::new(config, runtime, project.clone(), port).map_err(Fail::new)?);
    // The speed of this machine against the stored estimates: one small banded map per device, a second or two.
    {
        let calib = Arc::clone(&state);
        std::thread::spawn(move || {
            let f = cloud_map_generator_service::estimates::calibrate(&calib.runtime);
            calib.log(&format!("estimates calibrated: cpu x{:.2}{}", f.cpu, f.gpu.map_or(String::new(), |g| format!(", gpu x{g:.2}"))));
        });
    }
    let url = format!("http://127.0.0.1:{port}/");
    eprintln!("cloud-map-generator UI on {url} (Ctrl+C to stop)");
    state.log(&format!("engine rust; compute modes: {}", state.runtime.modes().iter().map(|m| format!("{}{}", m.mode.key(), if m.available { "" } else { " (unavailable)" })).collect::<Vec<_>>().join(", ")));
    if a.open {
        open_in_browser(&url);
    }
    // Measure Earth once now, so the first generation does not pay for it.
    let warm = Arc::clone(&state);
    std::thread::spawn(move || {
        if let Err(e) = warm.reference() {
            warm.log(&format!("the Earth reference map could not be measured: {}", e.message));
        }
    });
    http::serve(state, listener);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_widths_are_the_powers_of_two_from_512() {
        assert_eq!(widths_for(4096, None).unwrap(), [512, 1024, 2048, 4096]);
        assert_eq!(widths_for(8192, None).unwrap(), [512, 1024, 2048, 4096, 8192]);
        assert_eq!(widths_for(1500, None).unwrap(), [512, 1024, 1500]);
        assert_eq!(widths_for(256, None).unwrap(), [256]);
        assert_eq!(widths_for(4096, Some(&[256, 512])).unwrap(), [256, 512]);
    }

    #[test]
    fn bad_widths_are_refused_with_the_flag_named() {
        assert!(widths_for(4096, Some(&[63])).unwrap_err().starts_with("--widths: 63"));
        assert!(widths_for(4096, Some(&[257])).unwrap_err().contains("257"));
        assert!(widths_for(65536, None).unwrap_err().starts_with("--max-width: 65536"));
        assert!(widths_for(0, None).unwrap_err().starts_with("--max-width"));
        assert!(widths_for(4096, Some(&[])).is_err());
    }
}
