# The web version

The same page as `cloud-map-generator serve` (`ui/static`, daisyUI, inline Lucide icons), with the engine running in the visitor's
browser: the Rust core compiled to WebAssembly on the CPU, the per-texel stages as WGSL compute shaders on the GPU
(WebGPU through wgpu). Static files only: no server, no upload, nothing leaves the machine. It is what GitHub Pages
publishes (`.github/workflows/pages.yml`). Plan and constraints: `docs/web-plan.html`; the compute-mode contract:
`docs/compute-modes.html`.

## Build and run

```bash
node scripts/build-web.mjs        # builds rust/cloud-map-generator-wasm and the page into web/ (about a minute the first time)
node scripts/serve-web.mjs 8790   # any static server works; this one has no dependency and sends no special header
# http://127.0.0.1:8790/
```

Needs rustc with the `wasm32-unknown-unknown` target and the `wasm-bindgen` CLI at the version of the `wasm-bindgen`
crate in `rust/Cargo.lock` (the script refuses a mismatch and prints the `cargo install` line). `wasm-opt` (binaryen) is
used when it is on the PATH. The Rust build goes to `rust/target/web`, apart from `rust/target`.

What is committed in `web/`: `engine.js`, `worker.js`, this file. What the build generates (`web/.gitignore`): `index.html`
and `app.js` (copies of `ui/static`'s, with relative URLs and the engine script added, so the two pages cannot drift apart)
and `pkg/` (the module). `index.html` still loads daisyUI and Tailwind from jsdelivr, as `ui/static` does.

## How it works

```
page (ui/static/app.js, unchanged)
  fetch('/api/...') ──► engine.js  (replaces window.fetch for /api/* only: same JSON, same status codes, same messages)
                           │
                           ├─ Web Worker pool (hardwareConcurrency - 1), one WebAssembly instance each (worker.js)
                           │     CPU: a map is cut in bands of rows, one band per worker
                           │     the last worker also does the encodings and the metrics against Earth, and the GPU
                           └─ WebGPU, in that last worker: cloud-map-generator-gpu compiled to wasm32 (wgpu's WebGPU backend)
```

* **No threads in WebAssembly.** GitHub Pages cannot set the headers that give a page `SharedArrayBuffer`, so there is no
  rayon and no shared memory (`cloud-map-generator-core` is built with its default features). Workers exchange bands by message
  passing with transferable buffers.
* **Banding** (`rust/cloud-map-generator-wasm/src/band.rs`, the three-pass design of `docs/large-maps.html` specialised to maps up to
  4096 wide). Every stage is a pure function of the texel's direction and of the request's random plans, except five
  global couplings, which are the barriers: the standardization of the synoptic field and of the four regime-texture
  fields (their moments are merged in `f64`, in band order), and the smoothing of the row thresholds. A band keeps its fields
  between the passes, so only moments (15 numbers), thresholds (one per row) and finished rows cross between workers. Every
  worker draws the same plans from the seed, and back-traces only the coarse rows its band interpolates between (the
  back-trace is the largest cost in WebAssembly, whose software libm makes it about 6 s on one worker at 512 wide).
  **Result: the same map for any number of bands.** Natively the banded map differs from the one-piece engine by at most
  1.2e-7 (one float32 ulp) in coverage and 6e-8 in the type and high-share maps (`cargo test -p cloud-map-generator-wasm`); in Chromium a
  1-band and a 15-band map of the same request are bit-identical (max difference 0 at 512, 1024, 2048).
* **GPU.** The same `cloud_map_generator_gpu::pipeline` as the native engine, in float32. A browser cannot block on the GPU, so the
  few readbacks of the pipeline (the moments, the row thresholds, the finished rows) are awaited: `cloud-map-generator-gpu` is written
  once with `maybe-async`, plain blocking calls natively (the `is_sync` feature strips `async` and `.await`: the native code is
  unchanged) and futures in the browser. The probe at page load opens the device, runs the **df64 probe** (all four
  configurations are tested for exact TwoSum and TwoProduct), compiles the pipeline and makes a small map; the GPU mode is
  offered only when all that has just worked (2 s at page load on the machine below). A browser shows a page only the
  vendor and the architecture of the adapter, so that is the name ("NVIDIA Blackwell").
* **Same maps, within tolerance.** CPU and GPU maps of one request, decoded from the 8-bit PNGs in Chromium: coverage differs by
  1.0 / 2.1 (mean absolute, out of 255) at 512 / 1024, almost all of it the GPU's own sub-texel grain (the `compute.note` says
  so); type and high-share maps by 0.01. The metrics rows agree to the third decimal.
* **Compute modes.** `auto`, `cpu`, `gpu` selectable; `hybrid` is listed and unavailable ("Hybrid CPU + GPU scheduling is not
  built yet", as in the native engine). `auto` takes the GPU when there is one (see the table), else the CPU, and never fails
  for lack of a device. Without WebGPU the GPU rows carry the reason ("This browser does not expose WebGPU ...").
* **Files.** The two PNG sets, the 16-bit master, the R8 and the sidecar are made in the browser and kept (the last 8 results)
  as blobs; `Download` hands out `blob:` links with the same names as the server's.

## Measured (headed Chromium 157, Windows 11, 16-thread desktop CPU = 15 workers, RTX 5080; 2026-11-12)

Earth, seed 42, `surface: none`; each cell is the best of three runs after the page had settled. `total` is what the page
shows ("in N s": the engine, the encodings, the metrics against Earth); `engine` is the engine alone.

| width | CPU pool total | CPU engine | GPU total | GPU engine |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1.95 s | 0.63 s | 1.36 s | 0.10 s |
| 1024 | 2.52 s | 0.82 s | 1.77 s | 0.34 s |
| 2048 | 3.00 s | 1.18 s | 1.87 s | 0.30 s |
| 4096 | 6.10 s | 2.84 s | 3.42 s | 0.44 s |

The page is interactive about 3 s after load (workers started, the module compiled once and shared, the GPU probe of 2 s).

One worker for comparison (all rows in one band, engine only): 5.5 s at 512, 6.8 s at 1024, 12.1 s at 2048; the pool takes 0.8,
1.4 and 1.5 s. The rest of a request, 1.3 s at 512 up to 3 s at 4096, is the encodings and the metrics (FFTs of gnomonic patches)
on one worker, the same in both modes; it is what the GPU cannot shorten. A synthetic surface (`surface: synthetic`, at most
1024 wide) runs in one worker on the CPU (12 s at 1024) and on the GPU in 6 s.

## What does not work, or not yet

* `hybrid` (the scheduler that shares bands between the GPU and the workers) is not built.
* The metrics and the encodings run on one worker after the maps: they are now most of a request. Splitting them is the
  next win (the gnomonic patches of `metrics::measure` are independent).
* `surface: synthetic` is not banded (the surface terms couple the whole map twice more): one worker.
* Chromium only was driven. Safari and Firefox were not tried (`navigator.gpu` in workers is the thing to look at). Without
  WebGPU in workers the page runs the CPU path and says why.
* The page loads daisyUI and Tailwind from a CDN, as the local page does (vendoring them is the one change needed for a page
  with no third-party request). The module and the scripts are not cache-busted by hash: Pages caches them for ten minutes.
* 8192 and wider need the streamed design of `docs/large-maps.html` (output to a file); the page offers 512 to 4096.
