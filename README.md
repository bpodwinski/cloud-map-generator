# Cloud Map Generator

Global cloud-coverage maps for any planet, generated offline from its physics. You give a body its
rotation, radius, gravity and air, and optionally its land and relief. You get an equirectangular
cloud-fraction map with weather in it: an equatorial convergence band, clear subtropics, storm tracks
with cyclones that turn the right way in each hemisphere, fronts, and mesoscale cells. The map has no
seam at the antimeridian and no pinch at the poles. No training, no AI: a Rust engine on the CPU and, when there is one, the GPU (wgpu).

It is meant for a renderer that loads baked maps, such as a game or a planetarium. The generator never
runs at render time.

## What drives what

| Input | Effect | Module |
| --- | --- | --- |
| Rotation, radius, gravity, scale height | Hadley edge (Held-Hou), placing every latitude band | `rust/cloud-map-generator-core/src/climate.rs` |
| Same | Rossby deformation radius: number and size of the cyclones, their handedness | `rust/cloud-map-generator-core/src/flow.rs` |
| Weather regime | Fine texture: closed cells, open cells, streets | `rust/cloud-map-generator-core/src/texture.rs` |
| Land and relief (optional) | Marine decks, dry subtropical land, uplift on the windward side and rain shadow, shift of the convergence band | `rust/cloud-map-generator-core/src/surface.rs` |
| — | Cloud type, high-cloud share, wind | `rust/cloud-map-generator-core/src/products.rs` |
| — | Tileable cell-field detail tile (heaps, closed cells, open cells, sheet) | `rust/cloud-map-generator-core/src/tiles.rs` |
| — | Measurements against Earth (spectrum, object sizes, zonal profile, land/sea contrasts) | `rust/cloud-map-generator-core/src/metrics.rs` |

The zonal profile is anchored on Earth's measured cloud fraction (`data/`, NASA Blue Marble). Only
water-cloud bodies are generated: Venus, Mars and Titan are refused, because their cloud amount is not
Earth's (`climate.is_bakeable`).

## Install

A Rust toolchain (stable). Nothing else: the page of the web UI is embedded in the binary.

```bash
cd rust
cargo build --release          # target/release/cloud-map-generator and target/release/cloud-map-generator-mcp
```

Without a GPU stack: `cargo build --release --no-default-features -p cloud-map-generator-cli` (CPU only, no wgpu).

## Use

The `cloud-map-generator` binary has four commands and a server. Run it from anywhere inside the repository (the root is
the folder that holds `data/` and `examples/`; `CLOUD_MAP_GENERATOR_ROOT` or `--root` overrides it).

```bash
cloud-map-generator gen --seed 42 --width 4096                       # one map
cloud-map-generator gen --width 32768 --band-rows 256                # a 537-megatexel map, out of core (about 1 GiB of memory; see docs/large-maps.html)
cloud-map-generator measure output/earth_s42/coverage_earth_s42_4096x2048.png --sheet output/sheet.png
cloud-map-generator bake --version v1 --bodies examples/bodies.json  # a set of bodies, with a manifest
cloud-map-generator tiles --version v1                               # the cell-field detail tile
cloud-map-generator serve --port 8765                                # the local web UI (page and JSON API)
```

- `gen` writes to `output/earth_s42/`: a 16-bit master, an 8-bit preview, a raw R8 `.gz` file with rows
  south-up, the products, and a sidecar JSON. Flags: `--body earth|mars|titan|venus|fast` (only water-cloud
  bodies generate unless `--experiment`), `--texel-km`, `--match-earth`, `--isotropic`,
  `--surface <key> [--bodies <file>]`.
- `bake` writes to `output/bake/v1/` with `manifest.json`; paths start with `--url-prefix` (default
  `baked/v1`). It also bakes a library of K seeds per climate class for procedural worlds
  (`--only list|library`, `--k`, `--list-width`, `--library-width`).
- `serve` offers the compute modes auto, cpu, gpu (hybrid is not built yet); see [ui/README.md](ui/README.md).
- `cloud-map-generator-mcp` is the Model Context Protocol server for AI clients: see
  [rust/cloud-map-generator-mcp/README.md](rust/cloud-map-generator-mcp/README.md).

Per-command details: [rust/cloud-map-generator-cli/README.md](rust/cloud-map-generator-cli/README.md). Measured on one dev machine,
a 4096x2048 map takes about 25 s on one CPU thread and about 0.3 s on the GPU.

The tile preview shows each channel as a renderer draws it at 30, 50 and 70 % coverage. Judge the tile
through that threshold, not as gray.

Run the tests:

```bash
cd rust
GOLDEN_ROOT=../golden/data/v1 cargo test --workspace
```

The first Python version of the generator (the oracle the golden cases were frozen from) is kept in the git
tag `python-reference`.

## Bodies file

`examples/bodies.json`, schema in `rust/cloud-map-generator-service/src/bodies.rs`. One record per body:

```json
{
  "key": "earth",
  "rotationPeriodDays": 0.99727,
  "radiusKm": 6371,
  "gravityMps2": 9.81,
  "scaleHeightKm": 8.4,
  "cloudRegime": "water",
  "measured": false,
  "surface": null
}
```

- A negative rotation period is a retrograde spin.
- `measured: true` makes the bake skip the body (it already has a real map).
- `surface` gives a body land and relief:

  ```json
  {"width": 1024, "height": 512, "land": "earth_land_1024x512.u8", "elevation": "earth_elevation_1024x512.i16", "source": "..."}
  ```

  - `land` is a raw u8 land fraction, 0 to 255.
  - `elevation` is a raw little-endian int16, in meters above the sea.
  - Both grids are equirectangular, north-up, and stored relative to the bodies file.
  - No land data ships with this repository: build the grids from a public DEM of your choice.

## Output encodings

The sidecar JSON of each map records them:

| File | Encoding |
| --- | --- |
| `*_16bit.png` | Cloud fraction, 16-bit gray, north-up (the master) |
| `*.r8.gz` | Cloud fraction 0..255, rows SOUTH-up, gzip: ready for a GPU upload with the first row at the south pole |
| `*_high_share_*.png` | Share of the cloud above 440 hPa, 0..254 = 0..1 (255 = none) |
| `*_type_*.png` | 0 stratiform .. 255 convective |
| `*_wind_*.png` | RGB, R east / G north, m/s = (byte - 128) x 0.5 |
| `cloud_cell_tiles_*.rgba.gz` | Detail tile, RGBA8; each channel a rank, `floor(256 rank / texels)` |

## Limits

Each limit is a deliberate simplification, a model choice; the design log explains why.

- **Frozen flow:** vortices neither move nor evolve.
- **No season.**
- **One storm track per hemisphere.** A fast rotator would have several jets.
- **Earth's cell sizes on every body.**
- **Weak marine decks:** they hold on only 2 of 4 seeds.
- **The fine texture is still recognizably procedural up close.** That covers the fibres inside
  frontal shields and the irregularity of real cells.

The design log, with the measurements, the tuning rounds and a physics audit, is in
[docs/HISTORY.md](docs/HISTORY.md).

## Credits

- Earth's reference cloud fraction comes from NASA Goddard Space Flight Center / NASA Visible Earth,
  *Blue Marble* (2001, Terra MODIS). Details in [data/README.md](data/README.md).
- The tool was first written for the Exokosm game.

## License

MIT, see [LICENSE](LICENSE). The license requires keeping the copyright notice in every copy or substantial
portion of the software: anyone who uses or redistributes Cloud Map Generator must credit Benoit Podwinski.
