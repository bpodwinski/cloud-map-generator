//! What the machine has free, for the widths it can offer: free RAM and free disk, read without a dependency
//! (PowerShell on Windows, `/proc/meminfo` and `df` on Linux, `vm_stat` and `df` on macOS). `None` when the system does
//! not say: the caller then assumes enough and the run itself stops cleanly if it is not.

use std::path::Path;

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(cmd).args(args).output().ok().filter(|o| o.status.success())?;
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Free physical memory, bytes.
pub fn free_ram_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        let t = run("powershell", &["-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"])?;
        t.parse::<u64>().ok().map(|kb| kb * 1024)
    }
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
        line.split_whitespace().nth(1)?.parse::<u64>().ok().map(|kb| kb * 1024)
    }
    #[cfg(target_os = "macos")]
    {
        let t = run("vm_stat", &[])?;
        let page: u64 = 16384;
        let free = t.lines().find(|l| l.starts_with("Pages free"))?.split(':').nth(1)?.trim().trim_end_matches('.').parse::<u64>().ok()?;
        Some(free * page)
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        let _ = run;
        None
    }
}

/// Free disk space on the volume that holds `dir` (which may not exist yet: its nearest existing parent is used), bytes.
pub fn free_disk_bytes(dir: &Path) -> Option<u64> {
    let mut at = dir.to_path_buf();
    while !at.exists() {
        if !at.pop() {
            break;
        }
    }
    let at = std::fs::canonicalize(&at).unwrap_or(at);
    #[cfg(windows)]
    {
        let s = at.to_string_lossy().replace(r"\\?\", "");
        let letter = s.chars().next().filter(|c| c.is_ascii_alphabetic())?;
        let cmd = format!("(Get-PSDrive -Name {letter}).Free");
        run("powershell", &["-NoProfile", "-Command", &cmd])?.parse::<u64>().ok()
    }
    #[cfg(not(windows))]
    {
        let t = run("df", &["-Pk", &at.to_string_lossy()])?;
        let kb: u64 = t.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()?;
        Some(kb * 1024)
    }
}

// ---------------------------------------------------------------------------------------------------------
// The widths the machine offers

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// What the machine has free now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Machine {
    pub free_ram: Option<u64>,
    pub free_disk: Option<u64>,
}

/// The widths above the whole-array engine's, which run on the banded engine.
pub const LARGE_WIDTHS: [usize; 3] = [8192, 16384, 32768];

/// A banded run needs this much free memory whatever the width (a few bands: measured 0.5 GiB with the default bands,
/// 1.0 GiB with 256-row bands at 32768; the margin covers a GPU worker's host copies).
pub const BANDED_RAM_BYTES: u64 = 1536 << 20;

static OUTPUT_DIR: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);
static CACHE: Mutex<Option<(Instant, Machine)>> = Mutex::new(None);

/// The folder the large maps are written to (its volume's free space limits the widths).
pub fn set_output_dir(dir: &Path) {
    *OUTPUT_DIR.lock().unwrap_or_else(|p| p.into_inner()) = Some(dir.to_path_buf());
    *CACHE.lock().unwrap_or_else(|p| p.into_inner()) = None;
}

/// Free RAM and disk, measured at most every 20 seconds.
pub fn snapshot() -> Machine {
    let mut cache = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((t, m)) = *cache {
        if t.elapsed() < Duration::from_secs(20) {
            return m;
        }
    }
    let dir = OUTPUT_DIR.lock().unwrap_or_else(|p| p.into_inner()).clone().unwrap_or_else(|| std::path::PathBuf::from("output"));
    let m = Machine { free_ram: free_ram_bytes(), free_disk: free_disk_bytes(&dir) };
    *cache = Some((Instant::now(), m));
    m
}

fn gib(b: u64) -> String {
    format!("{:.1} GiB", b as f64 / (1u64 << 30) as f64)
}

/// The large widths the machine can write, and why it cannot offer the others (width and the reason).
pub fn large_widths(m: &Machine) -> (Vec<usize>, Vec<(usize, String)>) {
    let (mut ok, mut refused) = (Vec::new(), Vec::new());
    for w in LARGE_WIDTHS {
        let need = crate::large::disk_bytes(w);
        if let Some(ram) = m.free_ram.filter(|&r| r < BANDED_RAM_BYTES) {
            refused.push((w, format!("a banded run needs {} of free memory and {} is free", gib(BANDED_RAM_BYTES), gib(ram))));
        } else if let Some(disk) = m.free_disk.filter(|&d| d < need) {
            refused.push((w, format!("its files need {} of free disk and {} is free", gib(need), gib(disk))));
        } else {
            ok.push(w);
        }
    }
    (ok, refused)
}

/// The widest map a mode offers on this machine, texels: the largest advertised large width, else the whole-array
/// engine's own limit.
pub fn max_width(m: &Machine, whole_array_max: usize) -> usize {
    large_widths(m).0.last().copied().unwrap_or(whole_array_max)
}
