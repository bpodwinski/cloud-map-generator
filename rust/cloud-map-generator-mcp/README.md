# cloud-map-generator-mcp

An [MCP](https://modelcontextprotocol.io) server that lets an AI agent design the cloud cover of a planet,
look at it, measure it against Earth, and export the files. It works with any MCP client: Claude Code,
Claude Desktop, Codex, Cursor, VS Code, Zed, and anything else that can launch a command and speak MCP
over stdio.

cloud-map-generator turns a planet's physics (rotation period, radius, gravity, atmospheric scale height) into
a global, equirectangular cloud-coverage map with weather in it: an equatorial convergence band, clear
subtropics, storm tracks with cyclones that turn the right way in each hemisphere, fronts. This server puts
that generator in an agent's hands, with a loop it can follow on its own: **derive the physics, generate a
small map, look at it from several angles, read how it differs from Earth, adjust, generate large, export.**

It is meant for whoever consumes the maps: a game, a planetarium, a worldbuilding tool, a research
notebook. Nothing here assumes any of them. The server runs on your machine, reads and writes only the
folders you allow, and never touches the network.

This crate is the Rust server: one static binary for Windows, macOS and Linux, with no Python and no
runtime to install. It replaces the Python MCP server of the first implementation, tool for tool: same names,
same schemas, same descriptions, same resources, prompts, flags and error messages.

> **The engine is real.** The binary runs the Rust generator of this repository (`cloud-map-generator-service`, the layer
> `cloud-map-generator serve` runs on): the CPU always, and the GPU when the machine has one (wgpu: Vulkan, DirectX 12 or Metal).
> It is the same model as the Python generator the first server wrapped, checked against it case by case; a map of a
> seed is statistically equivalent to the Python map of that seed, not the same bytes. `--mock` serves a deterministic
> stand-in instead (see [Flags and environment](#flags-and-environment)); it is meant for trying a client and for the
> tests, and its maps are not a prediction of anything. Another engine plugs in through the `Backend` trait, see
> [Plugging in the engine](#plugging-in-the-engine).

- [Install](#install)
- [What the agent gets](#what-the-agent-gets)
- [Flags and environment](#flags-and-environment)
- [Where files go](#where-files-go)
- [Security model](#security-model)
- [Connect a client](#connect-a-client)
- [Example conversations](#example-conversations)
- [Limits and troubleshooting](#limits-and-troubleshooting)
- [Plugging in the engine](#plugging-in-the-engine)
- [Development](#development)

## Install

From a checkout of the repository (Rust 1.88 or newer):

```bash
cargo install --path rust/cloud-map-generator-mcp
cloud-map-generator-mcp --version
```

That puts `cloud-map-generator-mcp` in Cargo's bin folder (`~/.cargo/bin`, `%USERPROFILE%\.cargo\bin` on Windows),
which is on `PATH` for a normal Rust installation. To build without installing:
`cargo build --release -p cloud-map-generator-mcp` (from the `rust/` folder), then run `rust/target/release/cloud-map-generator-mcp`.

Prebuilt binaries for Windows, macOS and Linux will be attached to the releases; downloading one and putting it
on `PATH` is the whole installation.

The server starts in milliseconds and needs no network. The GPU is probed on a helper thread at start, so the first
call that needs the machine (`capabilities`, a generation) may wait a second or two for it, never the protocol
handshake. A build without the GPU kernels (`cargo build --release -p cloud-map-generator-mcp --no-default-features`) has no
wgpu in it and offers the CPU only.

The maps and the numbers are made by the real engine. To try a client without it, `cloud-map-generator-mcp --mock`
serves a deterministic stand-in (set `PLANET_CLOUDGEN_MOCK_GPU=1` to make it pretend to have a GPU, so that the
`gpu` and `hybrid` modes of a client can be tried).

## What the agent gets

### Tools

Every tool returns a short text for the model and a structured result with an output schema. Tools that
make something to look at also return a PNG image. Errors come back as tool errors whose message says what
was wrong and what to do.

| Tool | What it does |
| --- | --- |
| `list_presets` | The built-in planets (Earth, a retrograde Earth, fast and slow rotators, a super-Earth, Mars, Titan, Venus) with their physics, the valid range of every input, the map widths this server allows, the path of the output folder, and the compute modes this server offers (`compute`). |
| `derive_physics` | In milliseconds: Hadley-cell edge, storm-track latitude, cyclones per hemisphere, the sense in which they turn, texel size per width, warnings. "Call this before `generate_map` to check the planet is sane." |
| `generate_map` | Generates a map (widths up to 1024), measures it against Earth, saves the files, returns a summary, the metric rows, a coarse zonal profile and a preview image. Reports progress. `compute` chooses where the work runs ([below](#choosing-where-the-work-runs-compute)). |
| `start_generation` | The same (including `compute`) for any width up to `--max-width`: queues a job and returns a `job_id` and an estimate. |
| `get_generation` | Status of a job (`queued`, `running`, `done`, `failed`, `cancelled`), progress 0..1, and the full result when done. |
| `cancel_generation` | Cancels a waiting job at once, or a running one at its next stage boundary. |
| `view_map` | An image of a result or of a saved PNG: the flat map, or an orthographic **globe** from any longitude and latitude. Also the cloud type, high-cloud share and wind products. This is how an agent looks at the poles, the fronts and the cyclone spirals. |
| `measure_map` | Measures any coverage PNG (inside the allowed folders) against Earth: mean, zonal profile, coherence, spectral slope, object sizes, seam, poles. |
| `bake_bodies` | The offline bake pipeline for a set of bodies, passed inline (`bodies`) or in a JSON file (`bodies_file`, needed only for land and relief grids): a map set per body plus a library of seeds per climate class, with a manifest. Small widths by default. |
| `explain_model` | What the model does and does not do: inputs, products, known limits. |

Tool annotations are honest hints: `list_presets`, `derive_physics`, `get_generation`, `view_map`, `measure_map`
and `explain_model` only read; `bake_bodies` may overwrite the files of a bake it reruns.

One heavy job runs at a time (generation and baking share a queue); at most four more wait. Jobs run on a
worker thread, so the server keeps answering while a map is made. The last 16 results and jobs of a server
session are kept. A client that sends a progress token receives progress notifications during `generate_map`
and `bake_bodies`; a call the client drops cancels its job.

The server draws the previews and the globes itself from the coverage pixels (a box-averaged preview of at most
768 px, an orthographic globe with clouds white over a dark blue-gray sphere, north up and east to the right,
deterministic to the byte), so they look the same whatever engine made the map.

### Choosing where the work runs: `compute`

`generate_map` and `start_generation` take a `compute` argument with four values (the schema lists them):

| `compute` | What it means |
| --- | --- |
| `auto` (default) | The engine chooses: the GPU when one is available and the map is large enough for the upload to pay, otherwise the CPU. It never fails for lack of a device. |
| `cpu` | Every stage on the CPU, on every logical core (`--threads` or `CLOUD_MAP_GENERATOR_THREADS` limits it); the map is bit-identical for any thread count. |
| `gpu` | The per-texel stages on the GPU, in float32: the map equals the CPU map to float32 rounding and statistically, not bit for bit. The stages that suit the CPU badly stay on the CPU. |
| `hybrid` | The map is cut in row bands and each band goes to whichever of the GPU or a CPU worker is free. Offered only when a GPU and more than one CPU thread exist. |

The rules are the same for every engine behind this server:

- **No silent fallback.** An explicit mode the engine does not offer is a tool error that gives the reason and
  says what to use instead. Only `auto` picks for itself, and it never errors for lack of a device.
- **The result says what was used.** Every `generate_map` and `get_generation` result carries a `compute` block
  (`requested`, `used`, the devices with the seconds each worked and their share of the work, and a note when
  `auto` chose) and the text has a `Compute:` line. `compute.used` is `cpu`, `gpu` or `hybrid`.
- **Look before asking.** The resource `cloud-map-generator://capabilities`, and the `compute` key of `list_presets`,
  list the four modes with `available` and, for a refused one, the `reason`, plus the devices found.

`gpu` and `hybrid` need a GPU-capable engine. The real engine offers `gpu` when wgpu finds an adapter and the
pipeline passes its self-test at start (a small map through every kernel), and `hybrid` is not built yet: it is
always refused, with the reason. Without an adapter `auto` and `cpu` work and the other two are **not available**;
the tools, the schema and the resource stay the same, only what `capabilities` says changes with the machine. On a
machine with a GPU:

```json
{ "default": "auto", "engine": "rust",
  "modes": [ { "key": "auto",   "available": true,  "reason": null, "detail": null },
             { "key": "cpu",    "available": true,  "reason": null, "detail": "16 threads" },
             { "key": "gpu",    "available": true,  "reason": null, "detail": "NVIDIA GeForce RTX 5080" },
             { "key": "hybrid", "available": false, "reason": "Hybrid CPU + GPU scheduling is not built yet", "detail": null } ],
  "devices": { "cpu": { "name": "AMD Ryzen 7 5800X3D 8-Core Processor", "threads": 16 },
               "gpu": { "name": "NVIDIA GeForce RTX 5080", "backend": "Vulkan", "float64": false, "df64": "fma+barrier" } } }
```

Without a GPU, `gpu` says "No GPU adapter was found on this machine (wgpu looked for Vulkan, DirectX 12 and Metal
devices)." and `devices.gpu` is `null`.

(Each mode also has a `label` and a `description`, left out here.)

### Resources

| URI | What it is |
| --- | --- |
| `cloud-map-generator://docs/model` | The same Markdown as `explain_model`. |
| `cloud-map-generator://presets` | The presets and the valid ranges, as JSON. |
| `cloud-map-generator://capabilities` | The compute modes this server offers (`auto`, `cpu`, `gpu`, `hybrid`), which are available and why not, the engine and the devices found, as JSON. |
| `cloud-map-generator://docs/bodies-format` | The JSON bodies file `bake_bodies` reads (camelCase keys, optional land and relief grids), with an example. |
| `cloud-map-generator://reference/earth-coverage.png` | Earth's reference cloud map (2048x1024) that every map is measured against, embedded in the binary. |
| `cloud-map-generator://result/{id}/coverage.png` | The 8-bit coverage preview of a saved result. |
| `cloud-map-generator://result/{id}/type.png`, `.../high_share.png`, `.../wind.png` | The cloud type, high-cloud share and wind products of a saved result. |
| `cloud-map-generator://result/{id}/meta.json` | The sidecar of a saved result: physics, flow, encodings. |

The `{id}` is the hex token returned by `generate_map`. A result is found only if this server session made it
and saved it (`save=true`, the default): a resource URI is never a path.

### Prompts

| Prompt | What it sets up |
| --- | --- |
| `design_planet_clouds(planet_description)` | The design loop: derive the physics, generate at 512, look from several angles, measure against Earth, change one thing at a time, generate large, report. |
| `bake_for_engine(bodies_file)` | The bake flow: check each body, try one, bake small, then large, report the manifest. |

## Flags and environment

```text
cloud-map-generator-mcp [--out-dir DIR] [--max-width PX] [--read-dir DIR]... [--http] [--port PORT]
                    [--threads N] [--mock] [--log-level LEVEL] [--version]
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--out-dir DIR` | env `PLANET_CLOUDGEN_OUT`, else `<user data dir>/cloud-map-generator/output` | The only directory the server writes to. Created on first use. The user data dir is `%LOCALAPPDATA%` on Windows, `~/Library/Application Support` on macOS, `$XDG_DATA_HOME` or `~/.local/share` elsewhere. |
| `--max-width PX` | `2048` | The largest map width a tool may ask for: one of 256, 512, 1024, 2048, 4096. A larger width takes longer, so it must be asked for: `--max-width 4096`. |
| `--read-dir DIR` | none | An extra directory `measure_map`, `view_map` and `bake_bodies` may read. Repeatable. The out dir is always readable. |
| `--http` | off | Serve streamable HTTP on `127.0.0.1` instead of stdio. |
| `--port PORT` | `8766` | Port of the HTTP transport; the endpoint is `http://127.0.0.1:<port>/mcp`. |
| `--threads N` | env `CLOUD_MAP_GENERATOR_THREADS`, else every logical core | Threads of the CPU engine; `capabilities` reports the count (`devices.cpu.threads` and the `cpu` mode's `detail`). The maps are bit-identical for any count. |
| `--mock` | off | Serve the deterministic mock engine instead of the real one: fabricated maps, for trying a client without the generator (`PLANET_CLOUDGEN_MOCK_GPU=1` makes it pretend to have a GPU). |
| `--log-level` | `warning` | Log verbosity on stderr (`debug`, `info`, `warning`, `error`). stdout carries the protocol and is never logged to. |
| `--version` | | Print the version and exit. |

Environment: `PLANET_CLOUDGEN_OUT` sets the out dir (the flag wins); `CLOUD_MAP_GENERATOR_THREADS` sets the CPU engine's thread count (`--threads` wins). A relative `--out-dir` or `--read-dir` is
taken from the directory the client starts the server in, which varies from client to client: prefer absolute
paths in a client configuration. A bad flag stops the server with exit code 2 and a message on stderr.

## Where files go

```text
<out dir>/
  <id>/                         one folder per saved result (the id generate_map returned)
    cloudgen_s42_512x256_16bit.png     16-bit master, north-up
    cloudgen_s42_512x256.png           8-bit preview, north-up
    cloudgen_s42_512x256.r8.gz         raw R8, rows SOUTH-up, gzip (a GPU upload layout)
    cloudgen_s42_type_...png, ..._high_share_...png, ..._wind_...png
    cloudgen_s42.json                  sidecar: physics, flow, encodings
    cloudgen_s42_512x256_metrics.json  the map measured against Earth
  bake/<version>/               one folder per bake: map sets and manifest.json
  bodies-<version>.json         the inline bodies of a bake_bodies call, as the file they stand for
```

The encodings (which row is first, how wind and high-cloud share are packed) are recorded in the sidecar.
The folder grows with every `generate_map`; pass `save=false` while exploring, and delete old folders whenever
you like.

## Security model

The server runs code on your machine on behalf of an AI, so it treats every argument as hostile.

- **Writes** happen only under the out dir, in folders the server names itself (a result id it checked, a
  validated bake version). A caller never supplies the name of a folder to write to.
- **Reads** happen only under the out dir and the `--read-dir` roots. A path is resolved first (symbolic links,
  junctions and `..` followed) and the *resolved* path must be inside a root. A relative path is looked up from
  the roots. Network paths (`\\server\share`, `//host/x`, `\\?\...`), drive-relative paths (`C:file`), alternate
  data streams, Windows device names and control characters are refused on every platform.
- A **result id** is a lowercase hex token, checked against the session's cache. It is never turned into a path.
- **Bodies files** are parsed and checked before the bake runs: at most 32 bodies, safe keys (keys name the
  output files), and every land or elevation grid a record names must resolve inside a readable root and have
  the right size.
- **Sizes are capped**: map width (`--max-width`), preview and view sizes, PNGs the server decodes (up to
  8192x4096, and the header is read before anything is decoded), files it reads (64 MiB), bake size (an estimate
  over one hour is refused), queue length (4).
- **No network access, no subprocess, no `eval`.** The server never opens a connection of its own.
- **Errors never leak** stack traces or real filesystem paths: a refused path is answered with what is wrong
  and which locations are readable, by role ("the output folder", "a --read-dir directory"), not by address.
  A failure the caller can fix returns the fix; anything else returns a generic message and the details go to
  the server log on stderr. A panic inside the engine is caught: it fails one job, not the server.
- **HTTP** binds `127.0.0.1` only and rejects requests whose `Host` is not a loopback name (421) or whose
  `Origin` is not a local page (403), before the MCP service sees them (the DNS-rebinding guard). There is no
  authentication: any process on your machine can reach the port. Prefer stdio, where only the client that
  launched the server can talk to it.

## Connect a client

Every client below launches the same command: `cloud-map-generator-mcp`, with optional flags after it
(`--out-dir ./clouds --max-width 4096 --read-dir ./bodies`). If the client cannot find it, give the full path of
the executable (`where cloud-map-generator-mcp` on Windows, `which cloud-map-generator-mcp` elsewhere).

The generic stdio command, for any client:

```text
command:  cloud-map-generator-mcp
args:     (optional flags: --out-dir ./clouds --max-width 4096 --read-dir ./bodies)
```

### Claude Code

```bash
claude mcp add cloud-map-generator -- cloud-map-generator-mcp

# share it with the team through the repository (it writes .mcp.json), with 4096 maps allowed.
# Claude Code's own options go BEFORE the name; everything after `--` is the server's command line.
claude mcp add --scope project --transport stdio cloud-map-generator -- cloud-map-generator-mcp --max-width 4096
```

Check it with `claude mcp list` (or `/mcp` inside Claude Code).

`.mcp.json` at the root of a project:

```json
{
  "mcpServers": {
    "cloud-map-generator": {
      "command": "cloud-map-generator-mcp",
      "args": ["--read-dir", "${CLAUDE_PROJECT_DIR:-.}/bodies"],
      "env": { "PLANET_CLOUDGEN_OUT": "${CLAUDE_PROJECT_DIR:-.}/clouds-out" }
    }
  }
}
```

The `bodies` folder must exist (the server refuses to start with a `--read-dir` that does not). The working
directory of a server started by a client is not guaranteed, so give absolute paths or the client's project
variable, as above.

Or over HTTP, with the server started yourself (`cloud-map-generator-mcp --http`):
`claude mcp add --transport http cloud-map-generator http://127.0.0.1:8766/mcp`.

### Claude Desktop

Edit `claude_desktop_config.json` (Settings, Developer, Edit Config; on Windows `%APPDATA%\Claude\`, on macOS
`~/Library/Application Support/Claude/`) and restart the app:

```json
{
  "mcpServers": {
    "cloud-map-generator": {
      "command": "cloud-map-generator-mcp",
      "args": ["--out-dir", "C:\\Users\\you\\clouds"]
    }
  }
}
```

If the app reports that it cannot start `cloud-map-generator-mcp`, give the full path of the executable, for example
`"C:\\Users\\you\\.cargo\\bin\\cloud-map-generator-mcp.exe"`: a desktop app does not always inherit the `PATH` of your terminal.

### Codex

From the command line: `codex mcp add cloud-map-generator -- cloud-map-generator-mcp`.
To set the timeouts, edit `~/.codex/config.toml` (or the project's `.codex/config.toml`, which Codex reads only in
a trusted project):

```toml
[mcp_servers.cloud-map-generator]
command = "cloud-map-generator-mcp"
args = []
startup_timeout_sec = 30
tool_timeout_sec = 300      # the default is 60 s; a real engine takes tens of seconds for a 1024-wide map

[mcp_servers.cloud-map-generator.env]
PLANET_CLOUDGEN_OUT = "/home/you/clouds"
```

In a TOML basic string a Windows path needs doubled backslashes; a literal string avoids it:
`'C:\Users\you\clouds'`.

### Cursor

`.cursor/mcp.json` in a project (or `~/.cursor/mcp.json` for every project). Cursor expands
`${workspaceFolder}` in `command`, `args` and `env`:

```json
{
  "mcpServers": {
    "cloud-map-generator": {
      "type": "stdio",
      "command": "cloud-map-generator-mcp",
      "args": ["--out-dir", "${workspaceFolder}/clouds"]
    }
  }
}
```

### VS Code

`.vscode/mcp.json` in the workspace (or "MCP: Open User Configuration" from the command palette). The top-level
key is `servers`, not `mcpServers`, `type` is required, and `${workspaceFolder}` is expanded:

```json
{
  "servers": {
    "cloud-map-generator": {
      "type": "stdio",
      "command": "cloud-map-generator-mcp",
      "args": ["--out-dir", "${workspaceFolder}/clouds"]
    }
  }
}
```

### Zed

`settings.json` (Settings, AI, MCP Servers, Add Server, or edit the file), key `context_servers`:

```json
{
  "context_servers": {
    "cloud-map-generator": {
      "command": "cloud-map-generator-mcp",
      "args": [],
      "env": {}
    }
  }
}
```

### Try it without an AI client

The MCP Inspector (needs Node.js) opens a web page where you can call every tool by hand:

```bash
npx @modelcontextprotocol/inspector cloud-map-generator-mcp
```

### Windows notes

- In JSON, a backslash must be doubled: `"C:\\Users\\you\\clouds"`. Forward slashes also work in every path the
  server takes: `"C:/Users/you/clouds"`.
- The command is `cloud-map-generator-mcp` and Windows finds `cloud-map-generator-mcp.exe` by itself when it is on `PATH`.
  A desktop app (Claude Desktop, Cursor started from the Start menu) may not see the `PATH` your terminal has:
  use the full path of the `.exe`.
- `\\server\share` paths and `\\?\` paths are never read; copy the file into the out dir or a `--read-dir` folder.
- Junctions (`mklink /J`) and symbolic links are followed before the sandbox check, so a link that points out of
  the allowed folders is refused like the path it points to.

## Example conversations

The tool calls below are what an agent sends. The answers are shortened, and their numbers come from a real
run of the Python implementation (seed 42, the reference the Rust engine is checked against); the Rust engine's
maps of the same seed are statistically equivalent, so its numbers differ a little, and the agent's own comments
are illustrative.

### 1. Designing the clouds of a planet that turns slowly

> **You:** My planet is a bit smaller than Earth, 5200 km radius, gravity 7 m/s², and one day lasts 30 Earth
> days. Make me a cloud map and tell me what is different from Earth.

The agent derives the physics first.

```text
derive_physics  {"overrides": {"rotation_period_days": 30, "radius_km": 5200, "gravity_mps2": 7.0, "scale_height_km": 9}}
```

```text
Hadley edge 90.0 deg, storm track 80.0 deg, eddies False, ..., cyclones per hemisphere None, bakeable True.
Warnings: No storm track: the cloud stays zonal, with no cyclones and no wind field (a global Hadley cell,
like Venus or Titan).
```

It reads the warning (one global Hadley cell, so zonal bands and no storm systems), then generates at 512 and
looks at the result.

```text
generate_map  {"overrides": {"rotation_period_days": 30, "radius_km": 5200, "gravity_mps2": 7.0, "scale_height_km": 9}, "seed": 42, "width": 512}
view_map      {"id": "<id>", "view": "globe", "lon_deg": 0,   "lat_deg": 80}
view_map      {"id": "<id>", "view": "globe", "lon_deg": 120, "lat_deg": -30}
view_map      {"id": "<id>", "view": "map", "product": "type"}
```

```text
generate_map  ->  Generated a 512x256 cloud-coverage map in 7.8 s (id ba74ff1637ec, seed 42).
                  Hadley edge 90.0 deg, storm track 80.0 deg, no storm-track cyclones.
                  Reliable at this width: mean cloud fraction 0.280 (Earth 0.404), zonal-mean RMSE 0.284,
                  histogram distance 0.37. Spectrum, object sizes and coherence only compare with Earth from
                  width 2048.
                  Files: <out dir>/ba74ff1637ec        (+ the preview image)
view_map      ->  an orthographic globe centred at 80 N, then one centred at 30 S, then the cloud-type map
```

The agent reads the structured result too. Its `zonal` rows give the cloud fraction every 15 degrees of latitude,
with Earth's beside it:

| latitude | 90 | 60 | 45 | 30 | 15 | 0 |
| --- | --- | --- | --- | --- | --- | --- |
| this planet | 0.20 | 0.20 | 0.22 | 0.27 | 0.36 | 0.36 |
| Earth | 1.00 | 0.59 | 0.37 | 0.23 | 0.26 | 0.35 |

> **Agent:** With one global Hadley cell there is no storm track, so the mid-latitude cloud belt Earth has
> (59 % at 60 degrees) is missing: 20 % there, rising smoothly to 36 % at the equator. That is the model working as
> intended, not a defect; the mean (28 % against Earth's 40 %) is lower for the same reason. The globes show
> broad cells and no fronts or cyclone spirals, as `derive_physics` predicted. Spectrum and object-size metrics
> are not meaningful at 512, so I will make the final map at 2048.

```text
start_generation  {"overrides": {...}, "seed": 42, "width": 2048}
get_generation    {"job_id": "89ff52885681"}      (every few seconds)
```

```text
start_generation ->  Job 89ff52885681 running, estimated 50 s. get_generation(job_id='89ff52885681') every few seconds ...
get_generation   ->  Job 89ff52885681 running, 28% (backtrace 512x256, 0 vortices: 2.8 s), 6 s so far of about 50 s. ...
get_generation   ->  Job 89ff52885681 running, 57% (fields (2 fine octaves): 19.6 s), 24 s so far of about 50 s. ...
get_generation   ->  Job 89ff52885681 done in 38.8 s.
                     Generated a 2048x1024 cloud-coverage map in 38.7 s (id 68de2dde136c, seed 42). ...
                     Largest gaps to Earth (relative): Median object area (km2) +1.25, Pole spread, south +1.01,
                     Seam ratio -0.54.
                     Files: <out dir>/68de2dde136c
```

> **Agent:** Done. The 16-bit master, the 8-bit preview, the raw south-up file, the cloud-type, high-cloud and
> wind maps, the sidecar and the metrics are in `<out dir>/68de2dde136c`. At 2048 the structure rows are
> meaningful: the median cloud-object size is the largest gap to Earth, which is unsurprising for a planet that
> differs from Earth by design. `explain_model` lists the known limits of the fine texture if that matters
> for your use.

### 2. Comparing a retrograde Earth

> **You:** Does spinning the Earth backwards change its clouds? Compare them.

```text
generate_map  {"preset": "earth",            "seed": 42, "width": 512, "save": false}
generate_map  {"preset": "earth-retrograde", "seed": 42, "width": 512, "save": false}
view_map      {"id": "<earth id>",      "view": "globe", "lon_deg": -40, "lat_deg": 45}
view_map      {"id": "<retrograde id>", "view": "globe", "lon_deg": -40, "lat_deg": 45}
```

```text
generate_map (earth)           ->  Generated a 512x256 cloud-coverage map in 30.1 s (id 4330ebbd1700, seed 42).
                                   Hadley edge 30.0 deg, storm track 51.0 deg, 5 cyclones per hemisphere.
generate_map (earth-retrograde) -> Generated a 512x256 cloud-coverage map in 32.5 s (id 9331a3207acc, seed 42).
                                   Hadley edge 30.0 deg, storm track 51.0 deg, 5 cyclones per hemisphere.
                                   Notes: Retrograde spin: lows turn the opposite way in each hemisphere and the
                                   zonal winds flip sign. | ...
```

The two results carry the same structure numbers (`metrics` rows of the 512 maps, Earth then retrograde):

| metric | Earth | Retrograde Earth |
| --- | --- | --- |
| mean cloud fraction | 0.366 | 0.366 |
| zonal-mean RMSE vs Earth | 0.09 | 0.09 |
| histogram distance | 0.27 | 0.27 |

> **Agent:** The statistics do not move: the bands are placed by the Hadley edge, which depends on how fast the
> planet turns, not on the direction. What changes is the sense in which the lows spiral, as the retrograde
> result's note says. I looked at both globes from the same viewpoint (lon -40, lat 45) to compare the
> mid-latitude systems side by side. Both maps use the same seed, so the large-scale layout is shared and the
> handedness of the lows is the difference to look for.

### 3. Baking a catalog of bodies

> **You:** `bodies.json` lists the planets of my game. Bake their clouds so the renderer can load them.

The file sits in a folder the server may read (`--read-dir ./bodies`):

```json
{ "bodies": [
  { "key": "Dev:Gaia",  "rotationPeriodDays": 1.0, "radiusKm": 6371, "gravityMps2": 9.81, "scaleHeightKm": 8.0, "cloudRegime": "water" },
  { "key": "Dev:Drift", "rotationPeriodDays": 6.0, "radiusKm": 4200, "gravityMps2": 6.2,  "scaleHeightKm": 9.0, "cloudRegime": "water" },
  { "key": "Dev:Mars",  "rotationPeriodDays": 1.026, "radiusKm": 3389.5, "gravityMps2": 3.721, "scaleHeightKm": 11.1, "cloudRegime": "thin water ice" }
] }
```

```text
bake_bodies  {"bodies_file": "bodies.json", "version": "v1", "url_prefix": "assets/clouds/v1",
              "list_width": 512, "only": "list"}
```

```text
Baked 12 files in 36 s into <out dir>/bake/v1 (partial bake: no manifest written).
Skipped: Dev:Mars: clouds of thin water ice.
Next: measure_map or view_map on a baked coverage PNG; repeat with larger widths when the draft is right.
```

`Dev:Mars` was skipped because its clouds are not water (a bake never contains a body the model cannot describe);
the other two got six files each, and the manifest records the paths under the prefix the renderer will use:

```json
"Dev:Gaia": {
  "seed": 1724743284,
  "hires":     { "url": "assets/clouds/v1/cloud_dev_gaia_v1_coverage_512x256.r8.gz", "width": 512, "height": 256 },
  "slice":     "assets/clouds/v1/cloud_dev_gaia_v1_coverage_2048.png",
  "highShare": "assets/clouds/v1/cloud_dev_gaia_v1_high_share_2048.png",
  "type":      "assets/clouds/v1/cloud_dev_gaia_v1_type_2048.png",
  "wind":      "assets/clouds/v1/cloud_dev_gaia_v1_wind_512x256.png"
}
```

> **Agent:** I looked at the baked slices (`view_map` with the path of `cloud_dev_gaia_v1_coverage_2048.png`) and
> measured one (`measure_map`): the draft is sound. Now the real bake: the whole list plus the library of
> seeds per climate class, with the manifest.

```text
bake_bodies  {"bodies_file": "bodies.json", "version": "v1", "url_prefix": "assets/clouds/v1",
              "list_width": 2048, "library_width": 1024, "k": 2}
```

This runs as one queued job and takes minutes with a real engine (two 2048 maps and eight 1024 maps), reporting
progress as each map starts. It returns the manifest, the file list and any skipped body, and writes
`<out dir>/bake/v1/manifest.json`. Rerunning the same version overwrites its files; a new version label makes a
new folder.

## Limits and troubleshooting

- **What the model is not.** Water clouds only, with Earth's cloud amount; frozen flow (one still map, no
  seasons, no day and night); one storm track per hemisphere. `explain_model` has the full list, and the
  `warnings` of every result say what applies to that map. Metrics on maps narrower than 2048 only compare
  the mean, the zonal profile and the histogram reliably with Earth.
- **The engine.** The maps come from the Rust generator (`cloud-map-generator-service`); `--mock` serves the stand-in, whose
  maps are not predictions (the `engine` of `capabilities` says which one answers: `rust` or `mock`).
- **Timeouts.** The CPU engine takes about half a second for a 512 map and about 4 s for 4096 on 16 threads (one thread: about 5 s and 26 s), and the GPU well under a second, but a busy machine or one with few cores is slower, so a client with a short tool
  timeout should use `start_generation` and poll `get_generation`; a client that sends progress tokens also sees
  progress notifications during `generate_map` and `bake_bodies`.
- **`surface='synthetic'`** is a made-up world of five continents and a range, to see how land and relief act on
  the clouds. Your own land and relief come in through a bodies file and `bake_bodies`.
- **Baked file names.** The manifest names the downsampled slices `..._2048.png` whatever the map width; when
  `list_width` is below 2048 nothing is downsampled and those files are simply as wide as the map (`measure_map`
  tells you the real size).
- **`compute='gpu'` or `'hybrid'` is refused.** The machine or the engine does not offer them (no GPU adapter, a
  build without the GPU kernels, and always `hybrid` for now); the error
  gives the reason and the way out (`Error executing tool generate_map: compute: 'gpu' is not available: ... On
  this engine use 'auto' (it never fails for lack of a device) or 'cpu'; ...`). Use `compute='auto'` (the default)
  or `'cpu'`. The resource `cloud-map-generator://capabilities` always says which modes are
  available.
- **"Unknown result id".** Results live in the memory of the server process: the last 16, and none after a
  restart. The files stay on disk; give `view_map` or `measure_map` the path of a saved PNG instead.
- **Permission errors on a path.** The path is outside the out dir and the `--read-dir` roots. Copy the file
  into one of them, or add `--read-dir <folder>` to the server's arguments.
- **Logs.** Everything the server says goes to stderr (`--log-level debug` for more); most clients show it in
  their MCP log view.

## Plugging in the engine

The server never computes a map. It talks to a `cloud_map_generator_mcp::Backend`, a trait with what the tools need and
nothing more (`src/backend.rs`):

| Method | What the server asks for |
| --- | --- |
| `capabilities()` | The compute-mode contract (`docs/compute-modes.html`): the four modes, which are available and why not, the devices. |
| `presets()` | The built-in planets. |
| `derive(&Physics)` | Hadley edge, storm track, deformation radius, cyclones and their sense, texel sizes, warnings. |
| `generate(&GenerateRequest, progress, cancel)` | One map: write the files into the folder the request carries and return the id, the seconds, the metric rows, the zonal profile, the files and the coverage as an 8-bit north-up image buffer (the server draws previews and globes from it). |
| `measure_png(&Path)` | A coverage PNG measured against Earth. |
| `bake(&BakeRequest, progress, cancel)` | The bake of a bodies file into a folder, with the manifest. |
| `explain_model()` | The Markdown of `explain_model` and of `cloud-map-generator://docs/model`. |

Errors are a `BackendError` (a message written for an AI reader, a `Cancelled` and an `Internal` for what the
caller cannot fix). The rules that belong to the service layer, not to an engine (the range of each input, the
text of a bad preset or compute mode), are in `src/rules.rs`, so two engines cannot word them differently.

An adapter is a binary of a few lines:

```rust
use std::sync::Arc;

fn main() {
    let engine = MyEngine::new();                         // implements cloud_map_generator_mcp::Backend
    std::process::exit(cloud_map_generator_mcp::run(Arc::new(engine)));   // flags, stdio or HTTP, exit code
}
```

`run` parses the command line of `cloud-map-generator-mcp`, builds the server and serves until the client goes away.
To embed the server in another process, build a `cloud_map_generator_mcp::PlanetCloudgenServer::new(config, backend)` and serve
it over any rmcp transport. The shipped engine is `cloud_map_generator_mcp::service_backend::ServiceBackend` (an adapter over `cloud-map-generator-service`); the mock
(`cloud_map_generator_mcp::mock::MockBackend`) is the deterministic stand-in the tests use.

## Development

```bash
cd rust
cargo test -p cloud-map-generator-mcp      # unit tests, the sandbox, the MCP conversation, the HTTP guard, the binary over stdio
cargo run -p cloud-map-generator-mcp -- --version
```

The tests talk to the server in-process through the official `rmcp` client over an in-memory pipe, with the mock
engine at width 256; `tests/service_backend.rs` calls the real engine directly (files, rows, cancel, bake); the binary
tests spawn `cloud-map-generator-mcp --mock` and speak MCP to it over its stdio, and run the real engine once.

```text
src/backend.rs    the Backend trait and its types        src/sandbox.rs   every path and id check
src/service_backend.rs the real engine (cloud-map-generator-service)   src/jobs.rs      one heavy job at a time, a queue of 4
src/mock.rs       the deterministic stand-in engine
src/server.rs     the ten tools and the rmcp handler     src/results.rs   the last 16 results
src/tools.rs      names, descriptions, schemas           src/resources.rs resources and result templates
src/schemas.json  the input and output schemas           src/prompts.rs   instructions and the two prompts
src/render.rs     preview, globe, PNG reading           src/http.rs      streamable HTTP, Host and Origin guard
src/config.rs     flags, environment, default folders    src/rules.rs     service-level validation
```
