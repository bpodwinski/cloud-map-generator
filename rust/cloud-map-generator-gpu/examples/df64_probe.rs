//! Prints the df64 self-test of this machine's GPU: which of the four configurations are exact.
//! `cargo run -p cloud-map-generator-gpu --example df64_probe` (set WGPU_BACKEND=vulkan|dx12|metal to pick a backend).
use cloud_map_generator_gpu::{df64, GpuContext};

fn main() {
    let Some(gpu) = GpuContext::new() else {
        println!("no GPU adapter");
        return;
    };
    println!("adapter {:?} ({:?})", gpu.adapter_info.name, gpu.adapter_info.backend);
    let p = df64::probe(&gpu);
    for r in &p.results {
        println!("  fma={:<5} barrier={:<5}  two_sum exact: {:<5}  two_prod exact: {}", r.config.fma, r.config.barrier, r.two_sum_exact, r.two_prod_exact);
    }
    match p.best() {
        Some(c) => println!("use: DF64_FMA={} DF64_BARRIER={}", c.fma as u8, c.barrier as u8),
        None => println!("df64 is not usable on this device: stay in f32"),
    }
}
