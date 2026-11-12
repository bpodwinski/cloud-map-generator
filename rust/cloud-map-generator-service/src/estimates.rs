//! How long a map takes and how much disk it needs, shown before a run.
//!
//! Two sources, both measured, none a guess:
//!
//! * [`STORED`]: the wall seconds of every compute mode at each width, measured on the development machine
//!   ([`STORED_MACHINE`]) with `cloud-map-generator gen --banded` (and, below the banded engine's threshold, the whole-array
//!   engine through `cargo run --release -p cloud-map-generator-service --example cpu_bench`); the table of `docs/large-maps.html`
//!   holds the same numbers and their dates. A width that was not measured for a mode is interpolated in log-log
//!   between the neighbors that were (the cost is about the area), never taken from another mode.
//! * [`Factors`]: the speed of THIS machine against the stored one, measured at start-up by running one small banded map
//!   per device ([`calibrate`]) and dividing by the stored seconds of the same map. A machine as fast as the stored
//!   one has factors near 1; a laptop without a discrete GPU has a GPU factor of several. Until the measurement is done
//!   (a few seconds after start-up) the factors are 1 and `calibrated` is false.

use crate::engine::{ComputeMode, Runtime};
use std::sync::OnceLock;

/// The machine the stored table was measured on.
pub const STORED_MACHINE: &str = "AMD Ryzen 7 5800X3D (8 cores, 16 threads) with an NVIDIA GeForce RTX 5080, Windows 11, release build";

/// The width of the calibration map (it runs on the banded engine on every device).
pub const CALIBRATION_WIDTH: usize = 1024;

/// `(mode, width, wall seconds)`: a whole generation (the CPU rows up to 4096 are the whole-array engine's, measured 2026-11-12 with `cpu_bench`; from 8192 every row is the banded engine's, `cloud-map-generator gen --banded --compute <mode>`; the GPU has no row below 8192 because its whole-array engine was not timed end to end) as the server reports it (map, files, metrics against Earth), the
/// best or only run of the measurement, on [`STORED_MACHINE`]. See `docs/large-maps.html` for the dates and the runs.
pub const STORED: &[(&str, usize, f64)] = &[
    ("cpu", 256, 0.2),
    ("cpu", 512, 0.5),
    ("cpu", 1024, 0.7),
    ("cpu", 2048, 1.3),
    ("cpu", 4096, 3.9),
    ("cpu", 8192, 24.5),
    ("cpu", 16384, 99.8),
    ("gpu", 8192, 5.6),
    ("gpu", 16384, 19.0),
    ("gpu", 32768, 78.4),
    ("hybrid", 4096, 3.0),
    ("hybrid", 8192, 7.0),
    ("hybrid", 16384, 25.4),
];

/// `(mode, seconds)` of the calibration map (banded, no files), on [`STORED_MACHINE`].
pub const CALIBRATION: &[(&str, f64)] = &[("cpu", 0.26), ("gpu", 0.07)];

/// The speed of this machine against the stored one, per device (1.0: as fast; 2.0: twice as slow).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Factors {
    pub cpu: f64,
    pub gpu: Option<f64>,
    pub calibrated: bool,
}

impl Default for Factors {
    fn default() -> Factors {
        Factors { cpu: 1.0, gpu: None, calibrated: false }
    }
}

static FACTORS: OnceLock<Factors> = OnceLock::new();

/// The factors measured at start-up (the defaults until [`calibrate`] has run).
pub fn factors() -> Factors {
    FACTORS.get().copied().unwrap_or_default()
}

/// Seconds of the stored table for `mode` at `width`, interpolated in log-log between the measured widths of that mode
/// (extrapolated by area past the ends); `None` when the mode has no measurement at all.
pub fn stored_seconds(mode: &str, width: usize) -> Option<f64> {
    let mut pts: Vec<(f64, f64)> = STORED.iter().filter(|(m, _, _)| *m == mode).map(|&(_, w, s)| (w as f64, s)).collect();
    if pts.is_empty() {
        return None;
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let w = width as f64;
    if let Some(&(_, s)) = pts.iter().find(|p| p.0 == w) {
        return Some(s);
    }
    let (lo, hi) = match pts.iter().position(|p| p.0 > w) {
        Some(0) => return Some(pts[0].1 * (w / pts[0].0).powi(2)),
        Some(i) => (pts[i - 1], pts[i]),
        None => {
            let last = pts[pts.len() - 1];
            return Some(last.1 * (w / last.0).powi(2));
        }
    };
    let t = (w.ln() - lo.0.ln()) / (hi.0.ln() - lo.0.ln());
    Some((lo.1.ln() + t * (hi.1.ln() - lo.1.ln())).exp())
}

/// The estimate of a run in `mode` (`"cpu"`, `"gpu"`, `"hybrid"`; `"auto"` is the GPU's when there is one) at `width`,
/// seconds, scaled by the machine's factors. `None` when nothing was measured for it.
pub fn estimate_for(mode: &str, width: usize, f: &Factors) -> Option<f64> {
    let factor = match mode {
        "cpu" => f.cpu,
        "gpu" => f.gpu?,
        // Both devices work: the slower of the two factors is the safe side.
        "hybrid" => f.gpu.map_or(f.cpu, |g| g.max(f.cpu)),
        _ => return None,
    };
    stored_seconds(mode, width).map(|s| (s * factor * 100.0).round() / 100.0)
}

/// The disk a run of `width` asks, bytes (the same figure the widths are offered by).
pub fn disk_bytes(width: usize) -> u64 {
    if width >= crate::large::LARGE_FROM_WIDTH {
        crate::large::disk_bytes(width)
    } else {
        // The whole-array engine's files: the 16-bit master, the 8-bit preview, the gzip, the products.
        (width as u64) * (width as u64 / 2) * 4
    }
}

/// `estimate_seconds` of the legacy `GET /api/defaults`: the CPU's seconds per width.
pub fn estimate_seconds(widths: &[usize]) -> Vec<(String, f64)> {
    let f = factors();
    widths.iter().filter_map(|&w| estimate_for("cpu", w, &f).map(|s| (w.to_string(), s))).collect()
}

/// The estimate of each mode at each width, for `modes[].estimate_seconds`: the mode's own measurement (never another
/// mode's), `None` per width when it was not measured.
pub fn mode_estimates(mode: ComputeMode, widths: &[usize]) -> Vec<(String, f64)> {
    let f = factors();
    let key = match mode {
        ComputeMode::Auto => {
            if f.gpu.is_some() {
                "gpu"
            } else {
                "cpu"
            }
        }
        other => other.key(),
    };
    widths.iter().filter_map(|&w| estimate_for(key, w, &f).map(|s| (w.to_string(), s))).collect()
}

/// Measures the machine's factors: one `CALIBRATION_WIDTH` banded map per device, timed, against the stored seconds of
/// the same map. Run once at start-up on a helper thread (a few seconds); the result is read with [`factors`].
pub fn calibrate(rt: &Runtime) -> Factors {
    use crate::bands::{run_bands, BandCompute, CpuBands};
    use cloud_map_generator_core::banded::{BandOut, BandSink, Banded, SubNoise};
    use cloud_map_generator_core::climate::EARTH;
    use cloud_map_generator_core::generate::GenParams;
    struct Null;
    impl BandSink for Null {
        fn band(&mut self, _: &BandOut<'_>) -> Result<(), String> {
            Ok(())
        }
    }
    let body = EARTH;
    let params = GenParams::default();
    let time = |gpu: bool| -> Option<f64> {
        let banded = Banded::new(&body, 1, CALIBRATION_WIDTH, &params, None, SubNoise::Counter, 128, true).ok()?;
        let cpu = CpuBands(&banded, None);
        let gpu_worker = if gpu { rt.band_workers(crate::engine::Used::Gpu, &banded).ok()?.pop() } else { None };
        let worker: &dyn BandCompute = match &gpu_worker {
            Some(w) => w.as_ref(),
            None if gpu => return None,
            None => &cpu,
        };
        // One run to warm the caches and the kernels, one timed.
        for timed in [false, true] {
            let t = std::time::Instant::now();
            run_bands(&banded, &[worker], &mut Null, &|_| true).ok()?;
            if timed {
                return Some(t.elapsed().as_secs_f64());
            }
        }
        None
    };
    let reference = |mode: &str| stored_seconds_calibration(mode);
    let cpu = time(false).zip(reference("cpu")).map_or(1.0, |(now, stored)| (now / stored).clamp(0.05, 50.0));
    let gpu = time(true).zip(reference("gpu")).map(|(now, stored)| (now / stored).clamp(0.05, 50.0));
    let f = Factors { cpu, gpu, calibrated: true };
    let _ = FACTORS.set(f);
    f
}

/// The stored seconds of the calibration map (banded, no encodings: the `CALIBRATION` rows of the table).
fn stored_seconds_calibration(mode: &str) -> Option<f64> {
    CALIBRATION.iter().find(|(m, _)| *m == mode).map(|&(_, s)| s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_measured_width_is_the_tables_and_others_are_interpolated_in_log_log() {
        let (w0, s0) = (STORED[0].1, STORED[0].2);
        assert_eq!(stored_seconds(STORED[0].0, w0), Some(s0));
        assert!(stored_seconds("nonexistent", 1024).is_none());
        // Between two measured widths the estimate stays between their seconds.
        let cpu: Vec<(usize, f64)> = STORED.iter().filter(|r| r.0 == "cpu").map(|r| (r.1, r.2)).collect();
        if cpu.len() >= 2 {
            let (a, b) = (cpu[0], cpu[1]);
            let mid = ((a.0 as f64 * b.0 as f64).sqrt()) as usize;
            let s = stored_seconds("cpu", mid).unwrap();
            assert!(s >= a.1.min(b.1) && s <= a.1.max(b.1), "{s}");
        }
    }

    #[test]
    fn factors_scale_the_estimate_and_gpu_modes_need_a_gpu() {
        let slow = Factors { cpu: 2.0, gpu: None, calibrated: true };
        if let (Some(base), Some(twice)) = (estimate_for("cpu", 4096, &Factors::default()), estimate_for("cpu", 4096, &slow)) {
            assert!((twice - 2.0 * base).abs() < 0.02);
        }
        assert_eq!(estimate_for("gpu", 4096, &slow), None);
        assert_eq!(estimate_for("auto", 4096, &slow), None, "auto is resolved by the caller");
    }

    #[test]
    fn the_disk_estimate_grows_with_the_area() {
        assert!(disk_bytes(32768) > 16 * disk_bytes(8192) / 2);
        assert!(disk_bytes(32768) < 4 << 30);
    }
}
