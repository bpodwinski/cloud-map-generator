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
* **Compute modes.** `auto`, `cpu`, `gpu` and `hybrid` (CPU + GPU, offered with a GPU and two workers or more). `auto` takes the GPU when there is one (see the table), else the CPU, and never fails
  for lack of a device. Without WebGPU the GPU rows carry the reason ("This browser does not expose WebGPU ...").
* **Files.** The two PNG sets, the 16-bit master, the R8 and the sidecar are made in the browser and kept (the last 8 results)
  as blobs; `Download` hands out `blob:` links with the same names as the server's.

* **The banded engine of the server** (`rust/cloud-map-generator-wasm/src/large.rs`): for 8192 wide, and for `hybrid` at any
  width, the page runs what `cloud-map-generator serve` runs there, `cloud_map_generator_core::banded` with the counter-based
  noise. Each band of each pass goes to whichever unit is free, a CPU worker or the GPU worker (the native GPU band worker);
  between the passes every worker gets the moments, then the thresholds, of all bands; the finished bands stream, north to
  south, into the 16-bit master, the R8 and the 2048-wide slices on the last worker, which ends with the server's own
  `finish_parts`. Same refusals as the server: `match_earth` on a banded map, GPU bands at 512.
* **Same maps as the server.** For the same request the browser's banded maps and the server's have the same metrics (mean
  cloud fraction 0.365548 at 8192 on cpu, gpu and hybrid, 0.365556 at 1024 hybrid, in both).

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

8192 x 4096 on the banded engine (headless Chrome, same machine, one run): 33.9 s on the CPU (31 bands), 13.0 s on the GPU
(8 bands), 22.5 s on CPU + GPU (32 bands, the GPU did 56 % of the rows). Files: a 63 MB master, a 25 MB R8.

One worker for comparison (all rows in one band, engine only): 5.5 s at 512, 6.8 s at 1024, 12.1 s at 2048; the pool takes 0.8,
1.4 and 1.5 s. The rest of a request, 1.3 s at 512 up to 3 s at 4096, is the encodings and the metrics (FFTs of gnomonic patches)
on one worker, the same in both modes; it is what the GPU cannot shorten. A synthetic surface (`surface: synthetic`, at most
1024 wide) runs in one worker on the CPU (12 s at 1024) and on the GPU in 6 s.

## What does not work, or not yet

* The metrics and the encodings run on one worker after the maps: they are now most of a request. Splitting them is the
  next win (the gnomonic patches of `metrics::measure` are independent).
* `surface: synthetic` is not banded (the surface terms couple the whole map twice more): one worker.
* Chromium only was driven. Safari and Firefox were not tried (`navigator.gpu` in workers is the thing to look at). Without
  WebGPU in workers the page runs the CPU path and says why.
* The page loads daisyUI and Tailwind from a CDN, as the local page does (vendoring them is the one change needed for a page
  with no third-party request). The engine's files carry a build stamp (`?v=`), so a release never mixes with a cached one.
* 16384 and 32768 are offered but were not run in the browser. Their estimates are the 8192 time scaled by the texels, not
  measured.
* **The page's thread only runs the interface.** The finished bands of a banded map go from their unit straight to the
  sink worker over a `MessageChannel` (each worker has a port to it), never through the page; a unit's call returns once
  the sink has written its band, so at most one band per unit is in flight, and a band is at most about 16 MB of rows. The
  page only hands out band numbers and the small vectors of the barriers: during an 8192 run its longest stall is 2 ms.
* **Files on disk.** The 16-bit master and the R8 of a banded map are streamed, band by band, to the origin private file
  system (`navigator.storage.getDirectory()`, no permission asked, a share of the disk), through synchronous access handles
  in the sink worker: the master appended, the R8 rows written at their place counted from the south, then read back in
  8 MiB chunks through the module's gzip. The downloads are disk-backed `File`s, so a 1 GB master is never in memory. Each
  open page has its own folder, held by a Web Lock; a page removes the folders of pages that are gone, and the files of a
  result when it leaves the last 8. Without that file system the files are kept in memory. At 8192 on the GPU the disk path
  writes the same 63 MB master and 25 MB R8 as the memory path (11.9 s).
* On CPU + GPU the CPU workers are slower than the GPU per band, so at 8192 hybrid is slower than the GPU alone (as natively
  when the GPU is fast): `auto` picks the GPU.
