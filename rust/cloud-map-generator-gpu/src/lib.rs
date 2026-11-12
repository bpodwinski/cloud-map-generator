//! cloud-map-generator-gpu: WGSL compute kernels of the cloud map generator, run through wgpu.
//!
//! The same code runs natively (Vulkan, DX12, Metal: this is how the kernels are tested against the golden
//! cases) and in a browser (wgpu's WebGPU backend, compiled to WASM). Everything is f32: WGSL has no f64, so a
//! GPU result matches the CPU port within f32 tolerances and statistically, not bit for bit.
//!
//! [`GpuContext::new`] returns `None` when no adapter exists: callers (and tests) must then fall back to the
//! CPU path, never fail.

pub mod compose;
pub mod df64;
pub mod flow;
pub mod noise;
pub mod pipeline;
pub mod products;
pub mod stages;
pub mod texture;

use std::borrow::Cow;

/// `std::time::Instant` natively, `web_time::Instant` (`performance.now()`) in the browser, where the std one panics.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub(crate) use web_time::Instant;

/// A wgpu device and queue plus what the kernels need to know about them.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pub limits: wgpu::Limits,
}

impl GpuContext {
    /// The best adapter (high performance), with the adapter's own buffer limits requested so that large
    /// maps fit; `None` when there is no adapter.
    pub fn new() -> Option<GpuContext> {
        pollster::block_on(GpuContext::new_async())
    }

    pub async fn new_async() -> Option<GpuContext> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok()?;
        let adapter_info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("cloudgen"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .ok()?;
        Some(GpuContext { device, queue, adapter_info, limits })
    }

    /// A compute pipeline from WGSL source with one entry point (auto bind group layout).
    pub fn pipeline(&self, label: &str, wgsl: &str, entry: &str) -> wgpu::ComputePipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// [`GpuContext::pipeline`] with pipeline-overridable constants (`override NAME: ...;` in the WGSL).
    pub fn pipeline_with(&self, label: &str, wgsl: &str, entry: &str, constants: &[(&str, f64)]) -> wgpu::ComputePipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
            cache: None,
        })
    }

    /// A storage buffer initialised with `data`.
    pub fn storage_with<T: bytemuck::Pod>(&self, label: &str, data: &[T]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// An uninitialised storage buffer of `len` elements of `T`.
    pub fn storage_empty<T: bytemuck::Pod>(&self, label: &str, len: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (len * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// A uniform buffer initialised with `data` (a single `Pod` struct, 16-byte aligned by the caller).
    pub fn uniform_with<T: bytemuck::Pod>(&self, label: &str, data: &T) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Reads a storage buffer back to the CPU (blocking).
    ///
    /// Not available in the browser, which cannot block on the GPU: there it panics, and the code that has to run there
    /// (the pipeline) reads back with [`GpuContext::read_back_maybe`] instead.
    #[cfg(target_arch = "wasm32")]
    pub fn read_back<T: bytemuck::Pod>(&self, _buffer: &wgpu::Buffer, _len: usize) -> Vec<T> {
        panic!("GpuContext::read_back blocks on the GPU, which a browser cannot do: use read_back_maybe(..).await")
    }

    /// [`GpuContext::read_back`] where the caller is `maybe_async`: the same blocking call natively (the `.await` is
    /// stripped), an awaited readback in the browser.
    #[maybe_async::maybe_async]
    pub async fn read_back_maybe<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<T> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.read_back(buffer, len)
        }
        #[cfg(target_arch = "wasm32")]
        {
            let size = (len * std::mem::size_of::<T>()) as u64;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
            self.queue.submit([enc.finish()]);
            let slice = staging.slice(..);
            let (tx, rx) = oneshot::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r));
            rx.await.expect("map");
            let out = bytemuck::cast_slice::<u8, T>(&slice.get_mapped_range()).to_vec();
            staging.unmap();
            out
        }
    }

    /// Reads a storage buffer back to the CPU (blocking).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_back<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<T> {
        let size = (len * std::mem::size_of::<T>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        rx.recv().unwrap().expect("map");
        let out = bytemuck::cast_slice::<u8, T>(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        out
    }

    /// Dispatches `groups` workgroups of `pipeline` with several bind groups: `sets[g]` is the list of buffers of
    /// bind group `g` (binding i = buffers[i]); an empty list binds an empty group (for a kernel that skips that index).
    pub fn dispatch_groups(&self, pipeline: &wgpu::ComputePipeline, sets: &[&[&wgpu::Buffer]], groups: (u32, u32, u32)) {
        let bind_groups: Vec<Option<wgpu::BindGroup>> = sets
            .iter()
            .enumerate()
            .map(|(g, buffers)| {
                // An empty list still needs a (layout-empty) bind group when a LATER group is bound: wgpu wants every
                // index below the highest used one to be set.
                let layout = pipeline.get_bind_group_layout(g as u32);
                let entries: Vec<wgpu::BindGroupEntry> = buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                    .collect();
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries }))
            })
            .collect();
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            for (g, bg) in bind_groups.iter().enumerate() {
                if let Some(bg) = bg {
                    pass.set_bind_group(g as u32, bg, &[]);
                }
            }
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Dispatches `groups` workgroups of `pipeline` with one bind group of `buffers` (binding i = buffers[i]).
    pub fn dispatch(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer], groups: (u32, u32, u32)) {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &layout, entries: &entries });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        self.queue.submit([enc.finish()]);
    }
}

/// A one-value channel whose receiver is a future: what a `map_async` callback hands the awaiting code in the browser,
/// where the callback runs on the page's event loop after the function that asked has returned.
#[cfg(target_arch = "wasm32")]
mod oneshot {
    use std::cell::RefCell;
    use std::future::Future;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll, Waker};

    struct State<T> {
        value: Option<T>,
        waker: Option<Waker>,
    }

    pub struct Sender<T>(Rc<RefCell<State<T>>>);
    pub struct Receiver<T>(Rc<RefCell<State<T>>>);

    pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
        let state = Rc::new(RefCell::new(State { value: None, waker: None }));
        (Sender(state.clone()), Receiver(state))
    }

    impl<T> Sender<T> {
        pub fn send(self, value: T) {
            let mut s = self.0.borrow_mut();
            s.value = Some(value);
            if let Some(w) = s.waker.take() {
                w.wake();
            }
        }
    }

    impl<T> Future for Receiver<T> {
        type Output = T;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
            let mut s = self.0.borrow_mut();
            match s.value.take() {
                Some(v) => Poll::Ready(v),
                None => {
                    s.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADD_ONE: &str = include_str!("../shaders/smoke.wgsl");

    #[test]
    fn a_trivial_kernel_runs_and_reads_back() {
        let Some(gpu) = GpuContext::new() else {
            eprintln!("no GPU adapter: smoke test skipped");
            return;
        };
        eprintln!("adapter: {:?} {:?} max buffer {}", gpu.adapter_info.name, gpu.adapter_info.backend, gpu.limits.max_buffer_size);
        let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let a = gpu.storage_with("in", &input);
        let b = gpu.storage_empty::<f32>("out", input.len());
        let p = gpu.pipeline("smoke", ADD_ONE, "main");
        gpu.dispatch(&p, &[&a, &b], (input.len().div_ceil(64) as u32, 1, 1));
        let out: Vec<f32> = gpu.read_back(&b, input.len());
        assert!(out.iter().zip(&input).all(|(o, i)| *o == i + 1.0));
    }
}
