# Golden cases

> **Tooling moved.** The Python code that froze and checked these cases (`golden/freeze.py`, `check.py`, `store.py`,
> `cases/`, the generator itself) is no longer in the working tree: it lives in the git tag `python-reference`.
> The frozen data in `data/v1/` is the oracle; the Rust runner is `rust/cloud-map-generator-golden`
> (`cd rust && GOLDEN_ROOT=../golden/data/v1 cargo test -p cloud-map-generator-core`). To recompute or extend a case, check the
> tag out in a scratch folder and use the commands below there.


Frozen inputs and outputs of the Python generator: the oracle every later phase of the port (Rust
core, then GPU kernels) is checked against. [docs/port-plan.html](../docs/port-plan.html) is the plan.

```bash
python -m golden.freeze [--group <g>]     # compute the cases from the code, write golden/data/v1/
python -m golden.check  [--group <g>]     # recompute and compare to the files (tolerances)
python -m golden.check --strict           # bit-identical outputs: the determinism check, one machine
python -m golden.check --slow             # also the cases tagged 'slow' (statistics over seeds, bigger maps)
python -m pytest -q tests/test_golden.py  # the same, per group
```

Run from the repository root with the project's Python. Freeze replaces a group's directory. The
frozen data changes only for a DELIBERATE change of the model: say so in the commit message.

## The one fact a port must know: random draws

Every random table of the generator comes from numpy's `Generator` (PCG64): permutation tables,
rotation matrices (QR of a normal matrix), offsets, jitter grids, white-noise fields, the vortex
list. Reproducing numpy's bit stream (SeedSequence, PCG64, the Lemire bounded integers of
`permutation`, the ziggurat normal, LAPACK's QR) is possible but fragile, and a port does not have to.

So the draws are **inputs** of the cases. `cloudgen/plans.py` records them as the code runs
(`plans.capture()`), and each case that needs them lists them as input arrays named `plan<i>.<field>`
(`i` = call order) with their scalar descriptors in `params.plans`. A kernel is verified by feeding it
the recorded draws and comparing its outputs. A port is free to draw its own tables with its own
generator: its maps are then statistically equivalent to Python's for the same seed, not identical,
and the `stats` cases (`pipeline` group) judge that.

| `kind` | scalar fields | array fields | drawn in |
| --- | --- | --- | --- |
| `perlin3` | `seed` | `perm` (256, i32; the kernel indexes the DOUBLED table `perm ++ perm`, 512 entries) | `noise.Perlin3` |
| `fbm` | `seed`, `perlin_seed`, `frequency`, `octaves`, `gain`, `lacunarity`, `billow`, `norm` | `amp`, `freq` (octaves, f64), `rot` (octaves, 3, 3, f32), `off` (octaves, 3, f32) | `noise.fbm` |
| `worley3` | `seed` | `perm` (1024, i64; indexed doubled, 2048 entries), `jitter` (1024, 3, f32) | `noise.Worley3` |
| `cell_edges` | `seed`, `frequency`, `worley_seed` | `rot` (3, 3, f32) | `noise.cell_edges` |
| `tile.worley` | `size`, `freq`, `seed` | `jitter` (freq, freq, 2, f64) | `tiles.tileable_worley` |
| `tile.fbm` | `size`, `low`, `slope`, `seed` | `white` (size, size, f64) | `tiles.periodic_fbm` |
| `tile.poisson` | `seed`, `spacing` | `cand` (n, 2, f64), `keep` (n, bool) | `tiles.poisson_points` |
| `tile.lumps` | `size`, `seed` | `g` (n, 2, f64), `r` (n, f64) | `tiles.lumps` |

A `fbm` note is always preceded by its own `perlin3` note (same call), and a `cell_edges` note by its
`worley3` note. The vortex list (`flow.build_flow`) and the sub-texel white noise
(`generate`'s `subtexel`) are explicit data in the cases that use them.

## Layout

```
golden/data/v1/
  manifest.json      every case: group, id, function, bytes, sha256, tags, doc; the toolchain that froze them
  CATALOG.md         the same, readable
  <group>/<case id>/
    case.json        what the case is, its params, and where each array sits in data.bin
    data.bin         the arrays back to back (inputs, then outputs), little-endian, C order, 8-byte aligned
```

`case.json`:

```json
{
  "format": 1, "id": "fbm.4oct.seed11", "group": "sphere_noise", "function": "cloudgen.noise.fbm",
  "doc": "what it checks and why the tolerance is what it is",
  "params": {"frequency": 3.0, "octaves": 4, "seed": 11, "plans": [{"index": 0, "kind": "perlin3", "seed": 287335975}]},
  "tags": [],
  "inputs":  [{"name": "points", "dtype": "f32", "shape": [512, 3], "offset": 0, "nbytes": 6144}],
  "outputs": [{"name": "value", "dtype": "f32", "shape": [512], "offset": 7424, "nbytes": 2048,
               "tol": {"mode": "close", "rtol": 0.0001, "atol": 1e-05}}],
  "data": "data.bin", "sha256": "..."
}
```

- `dtype`: `f32 f64 u8 i16 u16 i32 u32 i64 bool` (bool is one byte, 0 or 1). Shapes are C order.
- `offset` and `nbytes` index `data.bin`. No other file format is needed: a port reads `case.json` and
  slices the blob.
- `tol`: `{"mode": "exact"}` (integers, bytes, ranks) or `{"mode": "close", "rtol", "atol", "frac"}`:
  `|got - want| <= atol + rtol |want|`, NaN equal to NaN, an infinity equal to itself. `frac`
  (optional, default 0) is the fraction of elements allowed outside that band: a field with a hard
  threshold or a tie, where an ulp of difference flips a few texels. The `doc` says why the tolerance
  is what it is.
- `function` names the Python function (or `pipeline.<stage>`) the case exercises: the port's test
  maps it to its own function. `params` are its scalar arguments; `inputs` its array arguments.
- A map is north-up, column 0 at longitude -180, unless the case says otherwise. The R8 export is
  south-up (`cloudgen/export.py`).

## Writing a case

A group is a module `golden/cases/<group>.py` with `cases() -> list[Case]` (see `sphere_noise.py`).

- **Deterministic.** No wall-clock time, no unseeded randomness, no dependence on the working
  directory. `python -m golden.check --strict` must pass.
- **Small.** A case is at most about 1 MB, the whole set about 15 MB. Kernels per-point: a few hundred to a
  few thousand points. Maps: 256x128 or 512x256.
- **Inputs include the random draws** the function makes (use `golden.util.captured`).
- **Cover the edges:** poles, the antimeridian, empty and saturated rows, a retrograde spin (negative
  rotation period), a body with no storm track, zero and large frequencies, the branch of every `if`.
- **Tolerances are measured, not guessed.** Integer, rank and byte outputs are exact. A float64
  closed-form formula: 1e-9. A float32 kernel: 1e-5 relative. A chained float32 field (noise, a
  back-trace of 48 steps, a bisection): estimate by recomputing the case with the inputs perturbed
  by one float32 ulp and take 10x the observed spread. Say so in `doc`.
- **Name by what is tested,** e.g. `hadley_edge.earth`, `compose.saturated`, `fbm.billow.seed11`.
- **Slow cases** (more than a few seconds to recompute) go in a `slow_cases()` function of the module:
  they are frozen with the rest, and `check` recomputes them only with `--slow` (`GOLDEN_SLOW=1` in pytest).
- **Statistical cases** (a metric of a map the port draws with its own random generator) are tagged
  `stat`: their output holds the Python distribution (mean and standard deviation over seeds), and a
  port checks that its own value lies within the stated number of standard deviations.

## Tags

`slow` (recomputed only with `--slow`), `stat` (a statistic over seeds or points: the port lands inside the
stated band, not on the value), `rng` (the output comes from numpy's generator: a port with another
generator checks structure and ranges, see the case `doc`), `degenerate` (an edge: zero vector, empty row,
smallest size), `fragile` (a one-ulp change of an input flips the output, so the case freezes float64 inputs or
keeps clear of the tie: a float32 port may fail it by design, the doc says how), `pin` (pins a constant or a
default), `artifact` (pins a known quirk of the Python code that a port should reproduce, not fix).

## Other files

- [PORT_NOTES.md](PORT_NOTES.md): numpy and scipy semantics a port must reproduce, by group.
- [STATUS.md](STATUS.md): how the cases were mutation-tested, the kill rates, what is not covered and what is
  not yet re-verified. READ IT before trusting a group as a regression net.
- `data/v1/CATALOG.md`: every case with its function, size and doc.
