//! The GPU probe of `cloud-map-generator serve`: which adapter wgpu finds, whether the double-single arithmetic is exact on
//! it, and the engine the slot builds for it. Compiled with the `gpu` cargo feature (on by default).

use crate::engine::{GpuInfo, GpuState, NO_ADAPTER_REASON};
use crate::gpu_slot;
use cloud_map_generator_gpu::df64::{self, Df64Config};
use cloud_map_generator_gpu::GpuContext;
use std::sync::mpsc;
use std::time::Duration;

/// How long start-up waits for a driver that does not answer.
const PROBE_TIMEOUT: Duration = Duration::from_secs(60);

/// `devices.gpu.df64`: the configuration of the library the probe found cheapest and exact.
pub fn df64_label(config: Option<Df64Config>) -> Option<String> {
    config.map(|c| format!("{}{}", if c.fma { "fma" } else { "split" }, if c.barrier { "+barrier" } else { "" }))
}

/// wgpu's backend names, as a person reads them.
fn backend_label(debug_name: &str) -> String {
    match debug_name {
        "Dx12" => "DirectX 12",
        "Gl" => "OpenGL",
        "BrowserWebGpu" => "WebGPU",
        other => other,
    }
    .to_string()
}

fn probe_blocking() -> GpuState {
    let Some(ctx) = GpuContext::new() else {
        return GpuState::Absent { reason: NO_ADAPTER_REASON.to_string() };
    };
    let config = df64::probe(&ctx).best();
    let info = GpuInfo {
        name: ctx.adapter_info.name.clone(),
        backend: backend_label(&format!("{:?}", ctx.adapter_info.backend)),
        // The device is opened with no optional feature: no f64 shaders.
        float64: false,
        df64: df64_label(config),
    };
    let engine = gpu_slot::build(ctx, config);
    GpuState::Found { info, engine }
}

/// Probes the GPU on a helper thread, so that a driver that hangs costs a timeout and not the server.
pub fn probe() -> GpuState {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(probe_blocking);
        let _ = tx.send(result);
    });
    match rx.recv_timeout(PROBE_TIMEOUT) {
        Ok(Ok(state)) => state,
        Ok(Err(_)) => GpuState::Absent { reason: "The GPU probe failed (the driver panicked while opening the adapter).".to_string() },
        Err(_) => GpuState::Absent { reason: format!("The GPU probe did not answer within {} s.", PROBE_TIMEOUT.as_secs()) },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_df64_label_names_the_configuration() {
        assert_eq!(df64_label(Some(Df64Config { fma: true, barrier: true })).as_deref(), Some("fma+barrier"));
        assert_eq!(df64_label(Some(Df64Config { fma: true, barrier: false })).as_deref(), Some("fma"));
        assert_eq!(df64_label(Some(Df64Config { fma: false, barrier: true })).as_deref(), Some("split+barrier"));
        assert_eq!(df64_label(Some(Df64Config { fma: false, barrier: false })).as_deref(), Some("split"));
        assert_eq!(df64_label(None), None);
        assert_eq!(backend_label("Dx12"), "DirectX 12");
        assert_eq!(backend_label("Vulkan"), "Vulkan");
    }
}
