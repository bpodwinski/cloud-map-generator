# Rust port

The Rust core of cloud-map-generator, ported module by module from the Python generator and checked against the
golden cases (`../golden/`, format in `../golden/README.md`). Plan: `../docs/port-plan.html`.

```
rust/
  cloud-map-generator-golden/   reader, comparer and runner of the golden cases (dev-dependency of the core)
  cloud-map-generator-core/     the port: src/<module>.rs mirrors cloudgen/<module>.py; tests/golden_<name>.rs run the cases
  cloud-map-generator-gpu/      the WGSL kernels and the generate_gpu pipeline (wgpu)
  cloud-map-generator-service/  the engine-facing layer both front ends share: presets, limits, derive, the compute modes and
                     their engines (CPU, GPU), one generation with its files and metrics, the bake
  cloud-map-generator-core/src/banded.rs   the banded (out-of-core) engine: three passes over row bands, a Sink trait (docs/large-maps.html)
  cloud-map-generator-core/src/counter.rs  the counter-based normal(seed, texel_index) the banded sub-texel noise uses (twin of the WGSL one)
  cloud-map-generator-service/src/{bands,large,machine,estimates}.rs   the band scheduler (CPU + GPU workers), the file sinks and run of a large map,
                     free RAM/disk and the widths they allow, the measured time estimates
  cloud-map-generator-cli/      the `cloud-map-generator` binary: gen, measure, bake, tiles, serve (the local web UI API)
  cloud-map-generator-mcp/      the `cloud-map-generator-mcp` binary: the Model Context Protocol server, over the service
```

The whole workspace: `cargo build --workspace`, `cargo test --workspace` (`GOLDEN_ROOT=../golden/data/v1`;
`REQUIRE_GPU=1` makes the GPU tests fail instead of skipping when there is no adapter). `--no-default-features` on
`cloud-map-generator-cli` or `cloud-map-generator-mcp` builds without wgpu.

```bash
cd rust
cargo test -p cloud-map-generator-core                          # everything
cargo test -p cloud-map-generator-core --test golden_sphere     # one module's golden cases
GOLDEN_SLOW=1 cargo test -p cloud-map-generator-core --release  # also the 'slow' cases
GOLDEN_ROOT=/path/to/golden/data/v1 cargo test ...   # another copy of the frozen data
```

## Rules of the port

1. **A direct port.** One Rust module per Python module, the same function names (snake_case), the same
   order of floating-point operations, the same precision at every step. Python's `float32` arrays are `f32`
   here, its `float64` are `f64`; a function numpy runs in the dtype of its input is generic over `f32`/`f64`
   (`num_traits::Float`). Do not "improve" the model: a port that differs from Python is a bug, even when
   Python is wrong. Say so in a doc comment if you find something that looks wrong.
2. **numpy and scipy semantics, exactly.** `../golden/PORT_NOTES.md` lists the traps: `np.round` is
   half-to-even (`f64::round_ties_even`), `astype(int)` truncates toward zero, `%` on integers is a modulo
   with the sign of the divisor (`rem_euclid`), `np.interp` clamps, `gaussian_filter1d` has its own truncation
   and boundary modes, `cKDTree(boxsize=...)` makes distances periodic, `np.unique` ties take their mean rank,
   NaN propagates unguarded (`normalize(0)` is NaN). When in doubt, read the case's `doc`: it says which.
3. **Random draws are inputs of the kernels.** Python's numpy draws are recorded in the cases as `plan<i>.<field>`
   arrays. Port each kernel as a pure function that takes the draws as arguments (for example
   `fbm(points, &FbmPlan)`), so the golden cases can feed it Python's draws, and give it a constructor that draws
   the plan with `crate::rng::Rng` for real use. Never try to reproduce numpy's bit stream.
4. **Honest tests.** The golden runner (`cloud-map-generator-golden`) fails on a case that fails, on a case the glue does
   not handle, and on a known failure that now passes. A case that cannot pass goes in the test file's
   `EXPECTED_FAILURES` with a reason a reviewer can check (for example "rng: the output is numpy's bit stream");
   never loosen a tolerance, never skip, never special-case a case id in the port.
5. **Dependencies** are `num-traits` and `rustfft`, nothing else without asking (`rayon`, optional, behind the
   core's `rayon` cargo feature). No `unsafe`. Threads are data parallelism only, in `cloud_map_generator_core::par`: an
   order-preserving map over independent texels or rows, and reductions in fixed blocks combined in order, so a map is
   **bit-identical for any thread count** (`cloud-map-generator-core/tests/threads.rs` pins 1, 3 and 16 threads). The feature is
   off by default in the core (it builds for wasm32; every loop then runs on the calling thread) and on in
   `cloud-map-generator-service`, `cloud-map-generator-cli` and `cloud-map-generator-mcp`. The count is `--threads N`, else `CLOUD_MAP_GENERATOR_THREADS`, else
   every logical core; `cargo run --release -p cloud-map-generator-service --example cpu_bench` times one thread against all.
   A new loop over texels goes through `par::map`, `par::map_index`, `par::map_rows` or `par::for_each_chunk_mut`;
   never accumulate across tasks.
6. **The banded engine** (`cloud_map_generator_core::banded`) computes the same map in horizontal bands so a 32768 x 16384 map needs about
   1 GiB instead of 30. Its contract: the output equals `generate_with_plans_norm` bit for bit on the CPU for any band height
   when both use the same noise and the same standardization statistics (`cloud-map-generator-core/tests/banded.rs`: 1, 7, 64 rows and the
   whole height, at 512, 1024 and 2048, with and without a surface). The statistics (the mean and standard deviation of the
   synoptic field and of the four fields of the regime texture) are float64 sums over rows combined in row order, so they do not
   depend on the band height; the sub-texel noise is `counter::counter_normal(seed, texel_index)`, a hash that WGSL runs too. A band
   is a pure function of its position: keep it so (no state across bands, no `std::fs`, no threads outside `par`; the module builds
   for wasm32).
```bash
cloud-map-generator gen --width 32768 --band-rows 256 --compute auto   # banded from 8192 up, or with --banded / --band-rows
```
   Per-band progress prints one line per band (`pass B, band 41 of 64`); `q` + Enter, or a file named by `--cancel-file`, cancels and
   removes the partial files. `cargo run --release -p cloud-map-generator-service --example band_bench -- 4096 cpu,gpu,hybrid 128 parity` times
   the three compute modes and prints how far the GPU's bands are from the CPU's; `--example crop` cuts a huge 16-bit PNG into
   crops at three scales to look for seams.
7. **Warnings are errors in spirit.** `cargo build` and `cargo test` must be warning-free.
