# Cloud map generator UI

A local web page to generate and inspect cloud-coverage maps with sliders instead of the command
line. `cloud-map-generator serve` is the backend: one JSON API over the Rust engine, plus the static page in
`ui/static/`, which is embedded in the binary at build time. There is nothing to install beyond the binary.

## Run

```bash
cloud-map-generator serve              # http://127.0.0.1:8765/
cloud-map-generator serve --port 9000 --open
```

- `--port N`: the port, default 8765. The server listens on `127.0.0.1` only.
- `--open`: opens the page in the default browser.
- `--widths 256,512,...`: the map widths the API accepts, default `512,1024,2048,4096`. A small one
  such as 256 (about one second of generation) is only useful to try the API quickly.

One generation runs at a time. A second `POST /api/generate` while one runs answers `429`. A generation
takes seconds to minutes: see `estimate_seconds` in `GET /api/defaults` (quiet-CPU figures; a busy
machine is slower). The server logs one line per request on stderr, and the stages of a generation.

The last 8 results stay in memory and are served until a newer one pushes them out (then `404`).
Restarting the server forgets them.

## API

JSON in and out, UTF-8. Errors are always `{"error": "<message>"}`.

| Method and path | Answer |
| --- | --- |
| `GET /` | `ui/static/index.html` |
| `GET /static/<file>` | A file of `ui/static/` (no way out of that folder) |
| `GET /api/defaults` | Version, backend, presets, widths, limits, time estimates, compute capabilities |
| `POST /api/derive` | The physics numbers of a body (fast, no generation) |
| `POST /api/generate` | Generate one map, measure it, keep the files |
| `GET /api/result/<id>/<name>` | A file of a kept result. `name`: `coverage.png`, `type.png`, `high_share.png`, `wind.png`, `master16.png`, `coverage.r8.gz`, `meta.json` |
| `GET /api/reference/coverage.png` | Earth's 2048x1024 reference (Blue Marble cloud fraction) |

Status codes: `200`; `400` bad input (the message names the field); `403` foreign `Host` or `Origin`
header; `404` unknown route, file or result; `405` wrong method; `429` a generation is running; `500`
anything else (the message is the exception).

### `GET /api/defaults`

```json
{"version": "0.1.0", "backend": "rust",
 "presets": [{"key": "earth", "label": "Earth", "rotation_period_days": 0.99727, "radius_km": 6371,
              "gravity_mps2": 9.81, "scale_height_km": 8.0, "cloud_regime": "water",
              "experiment_required": false, "note": "..."}],
 "widths": [512, 1024, 2048, 4096],
 "limits": {"seed": [0, 2147483647], "rotation_period_days": [-400, 400], "radius_km": [500, 30000],
            "gravity_mps2": [0.1, 100], "scale_height_km": [0.5, 100]},
 "max_cyclones_per_hemisphere": 32,
 "estimate_seconds": {"512": 1.5, "1024": 6, "2048": 25, "4096": 90},
 "compute": {"default": "auto", "engine": "rust", "modes": ["..."], "devices": {"...": "..."}}}
```

Presets: the bodies of `cloud-map-generator gen` (Earth, Fast rotator, Mars, Titan, Venus), a retrograde Earth, a 4-day
slow rotator, a 2-radius super-Earth, then any record of `examples/bodies.json` that is not already
there. `experiment_required` is true when the cloud regime is not water: the generator refuses those
unless `options.experiment` is on. `compute` says which compute modes this backend offers: see
[Compute mode](#compute-mode).

### `POST /api/derive`

Request: `{"body": {"rotation_period_days", "radius_km", "gravity_mps2", "scale_height_km",
"cloud_regime"}}`. `cloud_regime` is optional (default `"water"`); the four numbers are required and
must lie inside `limits`. Unknown fields are refused.

```json
{"hadley_edge_deg": 30.0, "storm_track_deg": 51.0, "has_eddies": true, "deformation_radius_km": 1000.8,
 "cyclones_per_hemisphere": 5,
 "texel_km": {"512": 78.2, "1024": 39.1, "2048": 19.5, "4096": 9.8},
 "bakeable": true, "warnings": ["..."]}
```

`deformation_radius_km` is `null` for a body with no spin, `cyclones_per_hemisphere` is `null` without
a storm track (`has_eddies` false). A zero rotation period is accepted: no spin, one global Hadley cell.
`warnings` are plain sentences for the page to show: non-water regime, retrograde spin, no storm
track, a clamped Hadley edge, too many cyclones.

### `POST /api/generate`

```json
{"body": {"rotation_period_days": 0.99727, "radius_km": 6371, "gravity_mps2": 9.81,
          "scale_height_km": 8.0, "cloud_regime": "water"},
 "seed": 42, "width": 1024,
 "options": {"match_earth": false, "isotropic": false, "experiment": false, "surface": "none",
             "compute": "auto"}}
```

`body`, `seed` (integer in `limits.seed`) and `width` (one of `widths`) are required. `options` and each
of its keys are optional, off by default; unknown keys are refused. `compute` is accepted in `options` but
says where the work runs, not what map it makes: see [Compute mode](#compute-mode).

| Option | Meaning | Same as |
| --- | --- | --- |
| `match_earth` | Match Earth's per-scale, per-zone statistics (`cloud-map-generator-core/src/matching.rs`) | `cloud-map-generator gen --match-earth` |
| `isotropic` | Baseline with no flow: no stirring, no vortices, the surface ignored | `cloud-map-generator gen --isotropic` |
| `experiment` | Allow a cloud regime that is not water | `cloud-map-generator gen --experiment` |
| `surface` | `"none"` or `"synthetic"`: the made-up five-continent world of `cloud-map-generator-service/src/synthetic.rs` (land and relief, at most 1024 wide) | `cloud-map-generator gen --surface` |

Answer:

```json
{"id": "9f2c41aa07be", "seconds": 4.2, "width": 1024, "height": 512,
 "compute": {"requested": "auto", "used": "cpu",
             "devices": [{"kind": "cpu", "name": "<processor name>", "seconds": 4.2, "share": 1.0}], "note": null},
 "meta": {"hadley_edge_deg": 30.0, "fine_octaves": 7,
          "flow": {"jet": 0.35, "trades": 0.05, "polar_easterlies": 0.25,
                   "vortices": [{"kind": "cyclone", "lon_deg": 12.3, "lat_deg": 51.0, "radius": 0.15,
                                 "omega": 2.1, "ascent": 1.0, "age": 0.9}]},
          "surface": null},
 "images": {"coverage": "/api/result/<id>/coverage.png", "type": "...", "high_share": "...", "wind": "..."},
 "downloads": [{"label": "Cloud fraction, 16-bit PNG", "href": "/api/result/<id>/master16.png"},
               {"label": "Raw R8 for the shader (gzip, rows south-up)", "href": "/api/result/<id>/coverage.r8.gz"},
               {"label": "Sidecar JSON", "href": "/api/result/<id>/meta.json"}],
 "metrics": {"table": [{"key": "mean", "label": "Mean cloud fraction", "generated": 0.366, "earth": 0.404,
                        "format": "0.000", "hint": "area-weighted; closer to Earth is better"}],
             "earth_label": "Earth (Blue Marble)", "note": null},
 "zonal": {"lat_deg": [-87.5, "..."], "generated": ["..."], "earth": ["..."], "target": ["..."]},
 "warnings": ["..."]}
```

- `meta.surface` is `null`, or `{"land_fraction", "itcz_shift_deg"}` with `surface: "synthetic"`.
  `vortices[].lon_deg` is wrapped to [-180, 180).
- `metrics.table` rows come from `cloud-map-generator-core/src/metrics.rs`: the server owns the rows,
  labels, hints and formats, the page only renders them. `format` is a number pattern: the zeros after
  the dot are the decimals (`"0.000"` three, `"0"` none). The zonal RMSE and histogram distance are
  distances to Earth, so Earth's own value is 0. A map is measured like `cloud-map-generator measure` does: box-resampled
  to 2048x1024 when wider, as is otherwise. Below 2048 px the finest scales are interpolated, so
  `metrics.note` says so (it is `null` from 2048 up) and the spectrum and object sizes are only comparable
  from 2048.
- `zonal` is the 5-degree band means (36 bands) of the generated map, of Earth, and of the target
  profile `climate.target_zonal_coverage` (with the ITCZ shift of the surface when there is one).
- `warnings`: the derive warnings of the body, plus one about `isotropic` when it is on.
- Images are 8-bit PNGs, **north-up**, `width` x `height`: `coverage` gray (0..255 = fraction 0..1);
  `type` gray (0 stratiform .. 255 convective); `high_share` gray (0..254 = 0..1, 255 = none); `wind`
  RGB (R east, G north, m/s = (byte - 128) x 0.5, B unused) at the wind grid size, `min(1024, width)`
  wide. Their encodings are the CLI's (`cloud-map-generator-core/src/export.rs`, `products.rs`).
- Downloads are sent with `Content-Disposition: attachment`: the 16-bit master (a 16-bit gray PNG,
  north-up), the raw R8 for the shader (gzip of `width` x `height` bytes, rows **south-up**), and the
  sidecar JSON (the CLI's, plus the `options` of the request). Their file names carry the seed, size
  and result id.

Refusals beyond the field checks, as `400`: a non-water `cloud_regime` without `options.experiment`,
a spin so fast that the body has more than `max_cyclones_per_hemisphere` (32) cyclones per
hemisphere (the generation time grows with the number of vortices: about 80 s at width 256 for 229 of
them), and a `compute` mode the backend does not offer (the message gives the reason). Derive still
answers for those, with a warning.

### Compute mode

The page chooses where the work runs, CPU or GPU, with `options.compute`. The contract below is shared by
every backend (this Python one, the Rust one, the web version) and by the MCP server; the full
specification, with what a new engine must implement, is `docs/compute-modes.html`.

| `options.compute` | Meaning |
| --- | --- |
| `"auto"` (default) | The engine chooses: the GPU when one is available and the map is large enough for the upload to pay, otherwise the CPU. It never fails for lack of a device. |
| `"cpu"` | Every stage on the CPU (the Python engine: one thread; the Rust engine: all cores; the web version: a pool of Web Workers). |
| `"gpu"` | The per-texel stages (noise, flow back-trace and ascent, regime blend, compose, the per-row threshold solve) on the GPU, in float32: the map equals the CPU map to float32 rounding and statistically, not bit for bit. The stages that suit the CPU badly (random plans, Poisson tiles, FFT metrics, file encoding) stay on the CPU. |
| `"hybrid"` | The map is cut in row bands and each band goes to whichever of the GPU queue or a CPU worker is free; measured throughput rebalances. Offered only when a GPU and more than one CPU thread exist. |

Rules: an explicit mode that is not available is an error (`400`, `{"error": "..."}`) that names the reason and
what to do, never a silent fallback; `"auto"` never fails for lack of a device; the answer always says what was
used. An unknown value (including `null`, a wrong case or a non-string) is a `400` that names the allowed ones.
The check runs before anything is generated: a refused request costs nothing and keeps no result.

`GET /api/defaults` answers `compute`, the capabilities the page builds its selector from: show every mode in
`modes`, disable the ones with `"available": false` and show their `reason` (for example as a tooltip).

```json
{"default": "auto", "engine": "rust",
 "modes": [
   {"key": "auto", "label": "Auto", "available": true, "reason": null, "description": "...", "detail": null},
   {"key": "cpu", "label": "CPU", "available": true, "reason": null, "description": "...", "detail": "1 thread"},
   {"key": "gpu", "label": "GPU", "available": false,
    "reason": "No GPU adapter was found.",
    "description": "...", "detail": null},
   {"key": "hybrid", "label": "CPU + GPU", "available": false, "reason": "...", "description": "...", "detail": null}],
 "devices": {"cpu": {"name": "<processor name or null>", "threads": 16},
             "gpu": null}}
```

- `modes` always holds the four keys, in this order. `available` and `reason` (null when available) are what the
  page shows; `detail` is a short extra such as `"1 thread"` (null on the other rows). `engine` is the backend
  that answers: `"rust"` here, `"wasm"` for the web version.
- `devices.cpu.name` is the processor name the system reports, or `null`; `threads` is the machine's hardware
  threads. `devices.gpu` is `null` without a GPU, else `{"name", "backend", "float64", "df64"}` (for example
  `{"name": "NVIDIA GeForce RTX 5080", "backend": "Vulkan", "float64": false, "df64": "fma+barrier"}`).

The answer of `POST /api/generate` carries `compute`, at the top level:

```json
"compute": {"requested": "auto", "used": "cpu",
            "devices": [{"kind": "cpu", "name": "<processor name>", "seconds": 3.1, "share": 1.0}],
            "note": null}
```

- `requested` echoes `options.compute` (`"auto"` when it was left out); `used` is `"cpu"`, `"gpu"` or `"hybrid"`.
- `devices` lists the devices that worked: `seconds` is the wall time each worked, `share` the fraction of the work
  (bands or texels) it did; the shares add to 1. The Python engine reports one `cpu` device with share 1.0.
- `note` is `null`, or one sentence on why an automatic choice went the way it did (for example
  `"auto chose the CPU: ..."`).
- The sidecar JSON (`meta.json`) records the same block without the per-device `seconds`, so the same inputs write
  the same bytes. The `options` of the sidecar stay the four options of the map: `compute` is not one of them.

## Safety

The server binds `127.0.0.1` and sends no CORS header. Requests whose `Host` is not
`127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>` (DNS rebinding), or that carry a foreign
`Origin` (a page of another site posting here), get `403`. Requests without an `Origin`, such as `curl`,
pass. Bodies are limited to 64 KiB, and `/static/` cannot leave `ui/static/`.

## The Rust backend

The Rust engine (`rust/`, `cloud-map-generator serve`) serves this API. The page asks `GET /api/defaults`
for `backend` (`"rust"` here) and for the widths, limits, estimates and compute modes it offers, so a
backend with other numbers, or one with a GPU, needs no change in the page.

## Tests

```bash
cd rust && cargo test -p cloud-map-generator-cli --test serve
```

The tests start a real server on a free port and call every endpoint with `urllib`. They generate at
width 256 (the `widths` argument of `make_server`), which takes a few seconds per map.
