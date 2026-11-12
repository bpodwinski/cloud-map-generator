//! Times the banded engine on each compute mode and checks that the devices agree:
//!
//!     cargo run --release -p cloud-map-generator-service --example band_bench -- <width> [modes: cpu,gpu,hybrid] [band_rows] [parity [surface]]
//!
//! Each mode runs the three passes into a sink that keeps only the 2048-wide slice statistics (so a 32768 map needs no
//! disk), and prints the wall seconds, the devices' seconds and shares. With `parity` the coverage of every mode is
//! kept (use a width of 4096 or less) and compared with the CPU's: mean and largest absolute difference.

use cloud_map_generator_core::banded::{BandOut, BandSink, Banded, SubNoise};
use cloud_map_generator_core::climate::EARTH;
use cloud_map_generator_core::generate::GenParams;
use cloud_map_generator_service::bands::{run_bands, BandCompute, CpuBands};
use cloud_map_generator_service::engine::{Runtime, Used};
use std::time::Instant;

#[derive(Default)]
struct Stats {
    keep: bool,
    coverage: Vec<f32>,
    sum: f64,
    n: usize,
}

impl BandSink for Stats {
    fn band(&mut self, out: &BandOut<'_>) -> Result<(), String> {
        self.sum += out.coverage.iter().map(|&v| f64::from(v)).sum::<f64>();
        self.n += out.coverage.len();
        if self.keep {
            self.coverage.extend_from_slice(out.coverage);
        }
        Ok(())
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let width: usize = args.get(1).and_then(|s| s.parse().ok()).expect("a width");
    let modes: Vec<&str> = args.get(2).map_or("cpu,gpu,hybrid", |s| s.as_str()).split(',').collect();
    let band_rows: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(cloud_map_generator_core::banded::auto_band_rows(width, width / 2, 1 << 22));
    let parity = args.get(4).is_some_and(|s| s == "parity");
    cloud_map_generator_core::par::init_threads(None).expect("threads");
    let rt = Runtime::probe();
    let body = EARTH;
    let params = GenParams::default();
    let surface = args.get(5).is_some_and(|s| s == "surface").then(|| cloud_map_generator_service::synthetic::synthetic_surface(width.min(1024)));
    let mut baseline: Option<Vec<f32>> = None;
    for mode in modes {
        let used = match mode {
            "cpu" => Used::Cpu,
            "gpu" => Used::Gpu,
            _ => Used::Hybrid,
        };
        let banded = Banded::new(&body, 42, width, &params, surface.as_ref(), SubNoise::Counter, band_rows, true).expect("a banded run");
        let cpu = CpuBands(&banded, (mode == "hybrid").then(|| std::env::var("HYB_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| cloud_map_generator_core::par::threads().saturating_sub(2).max(1))));
        let gpus = match rt.band_workers(used, &banded) {
            Ok(g) => g,
            Err(e) => {
                println!("{mode:7} width {width}: not available: {e}");
                continue;
            }
        };
        let mut workers: Vec<&dyn BandCompute> = Vec::new();
        match used {
            Used::Cpu => workers.push(&cpu),
            Used::Gpu => workers.extend(gpus.iter().map(|w| w.as_ref())),
            Used::Hybrid => {
                workers.extend(gpus.iter().map(|w| w.as_ref()));
                workers.push(&cpu);
            }
        }
        let mut sink = Stats { keep: parity, ..Stats::default() };
        let t = Instant::now();
        let (_, reports) = run_bands(&banded, &workers, &mut sink, &|_| true).expect("the run");
        let wall = t.elapsed().as_secs_f64();
        let total: usize = reports.iter().map(|r| r.rows).sum();
        let parts: Vec<String> = workers
            .iter()
            .zip(&reports)
            .map(|(w, r)| format!("{} {:.1} s share {:.3}{}", w.kind().key(), r.seconds, r.rows as f64 / total as f64, r.lost.as_ref().map_or(String::new(), |l| format!(" LOST: {l}"))))
            .collect();
        println!("{mode:7} width {width} bands of {band_rows}: wall {wall:.2} s, mean coverage {:.4}; {}", sink.sum / sink.n as f64, parts.join("; "));
        if parity {
            match &baseline {
                None => baseline = Some(sink.coverage),
                Some(b) => {
                    let (mut worst, mut sum) = (0.0_f32, 0.0_f64);
                    for (x, y) in sink.coverage.iter().zip(b) {
                        let d = (x - y).abs();
                        worst = worst.max(d);
                        sum += f64::from(d);
                    }
                    println!("        vs the first mode: mean |diff| {:.3e}, max |diff| {:.3e}", sum / b.len() as f64, worst);
                }
            }
        }
    }
}
