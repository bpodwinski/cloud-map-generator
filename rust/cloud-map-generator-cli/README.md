# cloudgen (command line)

One binary, `cloud-map-generator`, that replaces the four Python scripts and the web UI server. The model is `cloud-map-generator-core`;
the container I/O the core leaves out (PNG with the `png` crate, gzip with `flate2`), the bodies file, the JSON
sidecars, the presets, the generation over the CPU and GPU engines, the metrics table and the bake are
`cloud-map-generator-service`, shared with `cloud-map-generator-mcp`; this crate adds the command line and, for `serve`, the HTTP
transport, the validation of what a client sends and the page (`src/serve/`).

```bash
cd rust
cargo run --release -p cloud-map-generator-cli -- gen --seed 42 --width 4096          # gen.py
cargo run --release -p cloud-map-generator-cli -- measure output/earth_s42/coverage_earth_s42_4096x2048.png --sheet sheet.png
cargo run --release -p cloud-map-generator-cli -- bake --version v1 --bodies ../examples/bodies.json   # bake.py
cargo run --release -p cloud-map-generator-cli -- tiles --version v1                                   # bake_tiles.py
```

Run it from anywhere inside the repository (or set `CLOUD_MAP_GENERATOR_ROOT` / `--root`): the root is the folder that holds
`data/earth_cloud_fraction_bluemarble_2048.png` and `examples/bodies.json`, and `output/` hangs off it, as in
Python.

| Python | `cloud-map-generator` | Flags (same names and defaults) |
| --- | --- | --- |
| `gen.py` | `gen` | `--seed --width --texel-km --body --out --isotropic --match-earth --experiment --surface --bodies` |
| `measure.py` | `measure` | `<maps> --sheet --json --surface --bodies --surface-json --earth-tables` |
| `bake.py` | `bake` | `--version --bodies --url-prefix --list-width --library-width --k --only --out --manifest` |
| `bake_tiles.py` | `tiles` | `--version --seed --size` (and `--out`, which the script does not have) |

Same output files, same names, same encodings and the same sidecar key order as the Python tools. Differences:

- **The random draws are the port's own** (`cloud_map_generator_core::rng`), not numpy's. A map of a seed is statistically
  equivalent to the Python map of that seed, not the same map. Everything deterministic (seeds of a key, the names,
  the manifest, the land fraction and ITCZ shift of a surface, every number `measure` prints) is identical.
- `measure` on a map narrower than 2048 measures it at its own resolution (and says so on stderr); `measure.py`
  returns NaN for such a map. `--surface-sheet` is not implemented (it needs a binary erosion for the coastlines).
- Sidecars and the manifest end their lines with `\n` (Python's `write_text` writes `\r\n` on Windows).
- `generatedBy` stays `planet-cloudgen bake.py` / `bake_tiles.py`, so a manifest compares equal to the Python one.
- **Threads.** The CPU engine uses every logical core (rayon, the core's `rayon` feature, on in this binary). `--threads N`
  (any subcommand) or the environment variable `CLOUD_MAP_GENERATOR_THREADS` sets the count; `--threads` wins. The maps and every
  file are byte-identical for any count (only the sidecar's wall-clock `seconds` differ). Earth at 2048: 6.5 s on one
  thread, 0.8 s on 16 (engine only); at 4096: 18.9 s and 2.6 s. `serve` reports the count in `devices.cpu.threads`
  and in the `detail` of the `cpu` compute mode.
- `--seed` is an unsigned integer. Peak memory of `gen` (measured): about 1 GiB at width 4096 and 3.9 GiB at 8192,
  so about 4x per doubling: the default Earth width (no `--width`: 16384) would need on the order of 16 GiB.

`cargo test -p cloud-map-generator-cli` runs the unit tests and an end-to-end test of every subcommand on small sizes
(`tests/cli.rs`); they need the repository's `data/` folder (`CLOUD_MAP_GENERATOR_ROOT` points them elsewhere).
