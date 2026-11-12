//! Times the whole CPU generation (map, encodings, metrics against Earth: what the server reports as `seconds`) at
//! one thread and at all of them, and checks that both give the same coverage bytes.
//!
//!     cargo run --release --example cpu_bench -- [widths...] [--threads N,M,...]
//!
//! Earth, seed 42, `compute: cpu`, default options; the best of two runs per width and thread count. The seconds
//! of `estimates::CPU_ESTIMATE_SECONDS` come from here.

use cloud_map_generator_core::climate::EARTH;
use cloud_map_generator_core::par::{hardware_threads, with_threads};
use cloud_map_generator_service::engine::{ComputeMode, MapOptions, Runtime};
use cloud_map_generator_service::generate::{run_generation, Job};
use cloud_map_generator_service::reference::Reference;

fn main() {
    let mut widths = Vec::new();
    let mut counts = vec![1, hardware_threads()];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--threads" {
            counts = args.next().expect("--threads needs a list").split(',').map(|n| n.parse().expect("a thread count")).collect();
        } else {
            widths.push(a.parse::<usize>().expect("a width"));
        }
    }
    if widths.is_empty() {
        widths = vec![512, 1024, 2048, 4096];
    }
    let rt = Runtime::cpu_only("benchmark");
    let reference = Reference::from_png(cloud_map_generator_service::REFERENCE_PNG).expect("the embedded reference");
    for &width in &widths {
        let job = Job {
            body: EARTH,
            seed: 42,
            width,
            options: MapOptions::default(),
            compute: ComputeMode::Cpu,
            stem: format!("bench_{width}"),
            sidecar_seconds: false,
        };
        let mut first: Option<Vec<u8>> = None;
        for &n in &counts {
            let mut best = (f64::MAX, f64::MAX);
            for _ in 0..2 {
                let g = with_threads(n, || run_generation(&rt, &reference, &job, &|_, _| {}, &|| false)).expect("a generation");
                best = (best.0.min(g.seconds), best.1.min(g.engine_seconds));
                let bytes: Vec<u8> = g.coverage.iter().flat_map(|x| x.to_le_bytes()).collect();
                match &first {
                    None => first = Some(bytes),
                    Some(f) => assert!(*f == bytes, "the coverage differs between thread counts"),
                }
            }
            println!("width {width:>5}  threads {n:>3}  whole run {:>7.2} s  engine {:>7.2} s", best.0, best.1);
        }
    }
}
