# Verification status of the golden cases

How the cases were checked when phase 1 closed (2026-11-11). An independent agent per group applied single
mutations to the Python source in a scratch copy and ran the check: a mutant the check does not fail on
(a "survivor") is either equivalent (it cannot change an output) or a gap in the cases.

| Group | Mutants | Killed | Equivalent survivors | Real gaps (before repair) | Repair run |
| --- | --- | --- | --- | --- | --- |
| sphere_noise | 193 | 166 | 14 | 13 | yes |
| climate_flow | 241 | 204 | 23 | 14 | yes |
| texture_surface | 69 | 46 | 14 | 9 | yes |
| products_generate | 105 | 84 | 19 | 2 | yes |
| pipeline | 142 | 102 | 20 | 20 | no |
| metrics_match | 201 | 162 | 28 | 11 | no |
| tiles | 182 | 164 | 14 | 4 | yes |

The repair round added cases for the gaps and was NOT re-verified by a second mutation run: the kill rates
above are the BEFORE-repair figures. The repairs of `pipeline` and `metrics_match` were cut short by the owner
("bypass the tests if needed"): their case modules are the state at the cut. Re-run a mutation pass before
trusting those two groups as a regression net for the port (a driver per group is easy: apply a one-line
change in a scratch copy of cloudgen/, run `python -m golden.check --group <g> --slow`).

## Functions the cases do not exercise (last report of each group)

### sphere_noise

- `fbm octaves = 0`: Illegal input: np.stack([]) raises ValueError before any result exists; no caller passes it.
- `octaves_below / octave_ladder with frequency <= 0 or lacunarity <= 1`: octaves_below loops forever (frequency <= 0 never grows, lacunarity <= 1 never reaches the maximum) and octave_ladder divides by ln(lacunarity) = 0 or takes ln of a non-positive ratio; not legal inputs.
- `empty arrays and NaN inputs to the sphere functions`: No caller passes them and NaN propagation is plain IEEE arithmetic; normalize.degenerate.* and dir_to_lonlat.signed_zero cover the unguarded behavior the code really has.
- `sample_equirect with an integer-dtype image`: No caller passes one (all call sites hand float32 or float64 maps); the 3-D branch is covered with 1, 3 and 4 channels.
- `multi-chunk loop of fbm and cell_edges (range(0, n, 1 << 20) with a second iteration)`: Needs more than 2^20 points, far over the 1 MB case limit. Covered at build time instead: every fbm and cell_edges spec is recomputed with noise._CHUNK patched to 7 and 97 and must match within the case tolerance (kills the chunk-slice and chunk-stride mutants, which no stored data can). Port-irrelevant: a port may chunk differently.
- `upsample_dirs band loop with the band-split in a fast case`: The step is a literal (1 << 22) // width with no module constant to patch, and splitting needs more than 4M target texels; covered only by the slow case upsample_dirs.bands.4096x2048 (stored: 8 full rows plus the per-row sum of all 2048 rows).
- `upsample_dirs widths above 4096 (16K maps)`: A 16384-wide target is 134M texels and, measured, the float32-vs-float64 target error depends on the coarse grid height and not the target width, so a wide case would not discriminate anything new; the 512-row tall case pins the ill-conditioning that matters.

### climate_flow

- `climate.convective_tendency: np.clip lower bound 0.0`: Dead branch. The sum is at least 0.15 - 0.05 = 0.10 for every finite input, so the lower clip never engages; mutating 0.0 to 1e-3 survives and is equivalent. The upper clip (engages only at x = 0, value 1 + 2.3e-11) is pinned at rtol 1e-12.
- `climate.map_width_for_texel with a NaN or infinite radius, or texel_km = 0`: Python raises (ValueError / OverflowError from math.ceil, ZeroDivisionError), so there is no value to freeze. Documented in the case doc.
- `flow.Flow.backtrace with steps = 0`: ZeroDivisionError in Python (dt = time / steps); steps = 1 is the smallest frozen count.
- `flow.build_flow for a body with a storm track and a deformation radius of 0, negative or NaN (zero gravity, negative radius)`: Not a legal input (ZeroDivisionError or negative vortex radii); those bodies are covered by hadley_edge_raw, deformation_radius and has_baroclinic_eddies, not by build_flow.
- `flow.build_flow: round-half-to-even tie of m = max(2, round(x)) at x exactly k + 0.5`: Whether x lands on a half-integer to the last bit depends on the libm sin and cos, so it is not portable and is not frozen. The 16 wave_<k>p5 bodies sit 5e-5 relative either side of the tie instead, which pins the 5.0 wavelength factor to 0.005 per cent.
- `flow._comma_ascent: n < 1e-9 versus n <= 1e-9 at n exactly 1e-9`: Unreachable (a double colatitude whose computed east norm is exactly 1e-9). The threshold is pinned to 1e-5 relative by eight pole_colat cases; the strict-versus-non-strict mutant is equivalent in practice.
- `flow._comma_ascent age > 1.0 versus >= 1.0, and lat_sign * spin versus lat_sign / spin, and * copysign(1, lat) versus / copysign`: Equivalent mutants: the dry-slot weight (age - 1) * 2 is 0 at age 1, and dividing by +-1.0 equals multiplying by it.

### texture_surface

- `cloudgen.surface.Surface.load ValueError of a size mismatch (reshape of a file whose size does not match width x height)`: numpy reshape error, not a branch of the code and not a legal record; a port should simply reject such a record
- `cloudgen.surface.hemispheric_land_asymmetry on a 1-row map`: no hemisphere exists (0/0 gives NaN with a numpy warning); not a legal map, no branch involved
- `cloudgen.texture.regime_weights `total >= 1.0` versus `> 1.0``: equivalent mutant: at total == 1 the scale is 1 / max(1, 1e-9) = 1 either way, so no case can tell them apart

### products_generate

- `cloudgen.generate.generate`: Out of scope for this group (the pipeline group freezes it).
- `cloudgen.products.wind_map: np.where(n > 1e-9, east / max(n, 1e-12), (0, 1, 0)) fallback arm`: Unreachable: needs cos(lat) < 1e-9 and wind_map builds a cell-centred grid whose nearest row is 90/h degrees from the pole. Mutants on the fallback vector, the 1e-9 test and the 1e-12 floor are equivalent. Not exercised.
- `cloudgen.products.high_share: final np.clip(base + (1 - base) * extra, 0, 1)`: Dead for any input: base is in [0.2, 0.39] and extra in [0, 0.4], so the result stays in [0.2, 0.63] and neither bound can trigger. The lower bound of the inner clip(shield / max(coverage, 0.05), 0, 1) is now taken by high_share.out_of_contract (negative shield).

### pipeline

- `bake.bake_one, bake.main`: CLI orchestration and file I/O; its pieces (generate, write_entry, key_seed, slug, library_body) are covered. The library seed formula 1000 * int(center * 10) + i and the stem format cloud_library_h{int(center*10):03d}_s{i} live only in main(): a port reads them from bake.py.
- `cloudgen.bodies.load_bodies, body_surface; surface.Surface.load`: JSON and raw-file decoding (u8 land / 255, int16 m / 1000 clipped at 0), no numerics in this group; the files are not frozen.
- `surface.surface_terms upsample branch (tw != width) and surface._resample`: generate() calls surface_terms with term_width 1024, so the branch needs a map wider than 1024: a frozen case would be several MB. Covered only by direct calls with term_width in texture_surface, if that group does.
- `surface.Surface.at with a non-2:1 surface (w % width == 0 but h % (width/2) != 0), surface.itcz_shift_deg(None), orographic with wind None`: not reachable through generate() from a 2:1 grid; direct-call cases of texture_surface.
- `climate.hadley_edge_raw_deg with a non-finite (inf or NaN) period`: params are JSON: inf and NaN cannot be stored. Period 0.0 and None are covered (no_spin.zero_period.w16, no_spin.surface).
- `upper clip 0.99 of np.clip(target + anomaly, 0.01, 0.99) in generate`: the lower bound is taken (generate.surface.on); the upper needs target + anomaly > 0.99, beyond the largest target (0.75) plus the largest term the model produces.
- `noise.octaves_below, sphere.tangent_frame, texture._standardize with zero variance`: not called by generate() on any input of this group.

### metrics_match

- `metrics.spectral_slope on a non-square patch`: raises (the Hann window is n x n from shape[0]): an error path, not a behavior to port
- `metrics.detail_lost_2x on an odd height or width`: reshape raises ValueError: an error path
- `match._match_by_zone with no zone shared by gen and ze`: empty tables list raises IndexError: an error path
- `match._match_by_zone upper clip of i = clip(searchsorted - 1, 0, nz - 1)`: unreachable: searchsorted never exceeds nz, so only the lower bound of the clip is active (exercised)
- `match.match_to_earth with strength outside [0, 1] or levels < 0`: outside the documented domain
- `metrics.judge_surface with an empty maps list`: np.array([]).mean() warns and the verdicts are meaningless: not a behavior to port
- `metrics.zonal_profile with band_deg <= 0`: np.arange produces no or infinite edges: error path
- `metrics.measure on the real 2048x1024 reference (what measure.py feeds it, after load's box resample)`: a u8 2048x1024 map is 2 MB, above the 1 MB case limit; measure is size-agnostic and is frozen at 256x128 (default) and 512x256 (slow); load() lives in the measure.py script, not in the two modules

### tiles

- `bake_tiles.main: the final print (elapsed seconds, gz size)`: wall-clock time, nondeterministic, stdout only; sweep survivors on it (/ 1e6, - t0) are equivalent
- `bake_tiles.main: compressed bytes of the .rgba.gz and of the PNG`: encoder-specific; the decoded content (gunzipped rgba, decoded grey PNG) is frozen, so a gzip level change survives by design
- `bake_tiles.py: indent=2 of the sidecar text, sys.path.insert index`: layout and import path with no numerical content; the sidecar is frozen as parsed JSON
- `bake_tiles.py: the false direction of `if __name__ == '__main__'``: importing the script without running it; no behavior to freeze (the true direction is executed: every line of bake_tiles.py is hit)
- `raw_channels / build_tile: the default size=TILE_SIZE (1024)`: a 1024 tile is far over the case budget and no branch depends on the size; the default values are pinned through params.defaults (read from the signature), and raw_channels is called without seed so its default seed 1 is exercised
- `tileable_worley: `d < f1` tie and the sign of ix + dx / iy + dy`: equivalent mutants: a tie sets F2 to the same value either way, and the (-1, 0, 1) window is symmetric
- `warped_voronoi: F1 == F2 tie`: measure zero; border is 0 either way
- `warped_voronoi: np.mod(q + warp, 1.0) and the k=2 versus k=3 query`: equivalent: cKDTree with boxsize wraps its query points itself, and the third neighbor is never read
- `periodic_fbm: the DC guard max(kk, 1e-9)`: only changes a constant that the mean subtraction removes (equivalent); the guard is exercised by fbm.slope0.low0.s16 and fbm.slope_negative.low0.s16
- `lumps: the upper clip bound 1.0; rank_bytes: the clamp min(255, .)`: dead bounds (dist >= 0, and floor(256 rank / n) <= 255 for rank <= n - 1); lower caps such as 0.99 are caught by lumps.fourth_nearest_and_peak.s64.seed732

## Core issues the agents reported

- sphere_noise: golden/data/v1/manifest.json and CATALOG.md are stale for this group (manifest lists only 2 sphere_noise cases from an older freeze; 121 case directories exist, frozen with --no-manifest as instructed): regenerate with a freeze without --no-manifest at merge time.
- sphere_noise: golden/README.md tag vocabulary defines only slow and stat; this group also uses degenerate and fragile (and the function prefix property. for a property case). Their meaning is now in the module docstring and each case doc; the README should list them.
- sphere_noise: golden/README.md plan table gives perm as (256, i32) and (1024, i64), but the Perlin3 and Worley3 kernels index the doubled table concat(perm, perm) (512 and 2048 entries). The cases now say so in doc and params.perm_doubled; the README row should too.
- sphere_noise: No seam is missing in cloudgen/: fbm and cell_edges chunking is exercised by patching the module global noise._CHUNK at build time; the band step of upsample_dirs is a literal in the function body and cannot be patched, so its band split is covered only by the slow case.
- climate_flow: golden/data/v1/manifest.json and CATALOG.md have no climate_flow rows: the group was frozen with --no-manifest as the brief requires. The manifest must be rebuilt (python -m golden.freeze on all groups, or store.write_manifest) once every group is frozen.
- climate_flow: golden/README.md defines the tags 'stat' and 'slow' but not 'rng', which this group uses on 18 cases (12 build_flow.<body>.seed<k> and 6 build_flow.stats.*). A harness driven by the README would not know that a port with another random generator must skip them (except the flow_scalars and per-kind counts, which equal build_flow.structure.bodies). The README and the port harness need a rule for it.
- climate_flow: Seam: flow.build_flow's draws (rng.uniform, rng.poisson) are not recorded by plans.note, so a build_flow case cannot be replayed from its stored inputs. Worked around with explicit vortex lists for every kernel case, rng-tagged vortex-array cases, the seed-independent build_flow.structure.bodies case and the 300-seed stat cases; the core was not patched.
- texture_surface: golden/data/v1/manifest.json and CATALOG.md still list no case of group texture_surface because the group is frozen with --no-manifest: the orchestrator must run `python -m golden.freeze --group texture_surface` (or the whole set) at the end to regenerate the manifest sha256 and doc lines.
- texture_surface: golden/README.md does not say that a 0-d output (itcz_shift_deg, hemispheric_land_asymmetry, land_contrast.scalar, regime_weights.scalar.*) has shape [] with nbytes equal to one item; every such case says so in its own doc (golden/cases/texture_surface.py:_case appends the sentence), but the README should state it.
- texture_surface: golden/README.md does not say that NaN is a legal value in an OUTPUT array (earth_tables empty bands); store.compare and the case.json writer accept it (NaN equals NaN, no bare NaN token in the JSON) and the three affected cases document it, but a reader that rejects non-finite floats would fail.
- texture_surface: golden.store.Case outputs must be numpy arrays, so a string result (Surface.load's source) has no native encoding; this group stores it as the UTF-8 bytes of a u8 output.
- texture_surface: cloudgen/surface.py: the three floors of 1e-3 in orographic_anomaly and smooth_on_sphere only bind above 1570 rows or below 1 mm/s; they are covered here with purpose-built tall (1600 x 8, 1700 x 8, 2000 x 8) and near-calm inputs, which are not shapes the generator produces.
- products_generate: Seam: products.wind_map exposes no intermediates (vortex-only velocity in m/s, east and north bases, zonal row profile). The wind_map cases rebuild them in the case runner by repeating the source expressions through public functions (still-Flow.velocity, np.cross, climatological_zonal_mps) and verified bit for bit when frozen that float32(v_east + zonal[:, None]) and float32(v_north) equal wind_map's result. If wind_map changes, that copy can drift silently; a trace= hook like generate()'s would remove the duplication.
- products_generate: Quirk in surface_regime_weights (generate.py 160-163): (1.0 - closed) is rounded to float32 before it meets the float64 deck, so closed can land up to about 3e-8 above 1 and the scale (1 - closed) / rest goes slightly negative; the open and street shares come out down to -2.3e-8 (seen in surface_regime_weights.earth.hadley30). The source does not clamp it. Harmless at the tolerance, documented in the case docs, not edited.
- products_generate: Quirk: _smoothstep(a, b, x) has no guard for a == b (NaN at x == a, 0 or 1 elsewhere); no call site uses it. Pinned as smoothstep.degenerate_window.
- products_generate: golden/README.md has no section for 'pin' cases (a function name that is not a Python function: constants.products, whose OUTPUT arrays are the reference table) nor a place for port notes. The two docs that said 'see the port notes' pointed at nothing; the notes are now inline in the case docs. A 'pin' tag is used on constants.products and genparams.defaults and is not documented in the README.
- products_generate: golden.check builds every case of the group up front (build(group)), so a runner that raises aborts the whole group check with a traceback instead of one per-case failure. I removed an assertion I had put in the wind_map runner for that reason; store/check were not edited.
- products_generate: The 36-step bisection makes solve_thresholds tolerance-blind to 35 or 37 steps and to returning the bracket's lower end (effect 1.5e-10 to 2.9e-10 against atol 1e-9 of solve_thresholds.all_f64): five mutants survive for that reason. A tighter atol (about 4e-10) would kill them, but the summation-order flip risk is of the same size, so it was left.
- pipeline: generate() draws the sub-texel white noise inline (np.random.default_rng(seed + 7).standard_normal) with no plans.note, and the trace has neither it nor the Flow: the cases recompute the white noise with the same call (asserting white / float32(sqrt(n_clouds)) is bit-identical to trace['subtexel']) and rebuild the flow with build_flow (asserting its vortex list equals meta['flow']). A plans.note('subtexel', ...) and a trace['flow'] would remove the workaround.
- pipeline: Trace quirks: trace['target'] is the same object as trace['target_zonal'] without a surface; trace['offset'] is a 0-d float64 0.0 without one; trace['terms.itcz_shift_deg'] is an np.float64 scalar, not an ndarray; weights.* and shield are float64 (twice the bytes). meta['seconds'] is wall-clock, and meta['surface'] values are rounded (4, 2 and 3 decimals): the cases freeze unrounded recomputations instead.
- pipeline: measure.load cannot read a map narrower than the 2048x1024 reference (fy = h // 1024 = 0 gives an empty reshape and NaN means): the 1024x512 stat maps are passed to metrics.measure as generated, with no resampling.
- pipeline: Model behaviour pinned by stat.earth.zonal_residual: on a 1024x512 map the per-row zonal mean matches climate.target_zonal_coverage to rms 0.017 (max 0.067) away from the poles, but the pole rows are far off (north rows 0 to 6: +0.15, -0.06, -0.34, -0.27, -0.17, -0.15, -0.19; -0.078 in the last 5-degree band): a pole row is one point, its coverage cannot follow its target, and the 2-degree threshold smoothing spreads it. A port should reproduce it, not fix it.
- pipeline: A cyclone exactly on a pole has an arbitrary east vector (|east| < 1e-9 falls back to +y) and a one-ulp change of its latitude flips it (lat past the pole mirrors the point): custom_flow.exact_poles.w32 freezes it with float64 centres, a float32 port will fail it by design. The seam/pole scenario keeps its cyclones 1e-6 rad away so the nudge measurements stay valid.
- pipeline: The only hard float threshold in generate() is the coast test cont >= 0.5 in surface.deck_strength: a one-ulp tie flips a coast and moves whole rows (retrograde.box on a 128-wide synthetic surface had min |cont - 0.5| = 0 and moved weights.closed by 0.24 under a one-ulp nudge). The surface cases are built with |cont - 0.5| >= 1e-4 (asserted when the case is built), so no tolerance needs frac.
- pipeline: Under a loaded machine eight worker processes with full OpenBLAS pools failed (OpenBLAS 'Memory allocation still failed', numpy _ArrayMemoryError) in slow_cases: the pool now uses at most 4 workers on one BLAS thread each and falls back to in-process generation on BrokenProcessPool or MemoryError; results are bit-identical either way (check --slow --strict passed).
- metrics_match: No missing seam: metrics.py and match.py draw nothing from numpy's generator, so no random plan is needed in this group.
- metrics_match: metrics.surface_contrasts: itcz_lat is NaN for any map with fewer than 72 rows (verified at 128x64; 256x128 and 1024x512 are fine): a 2.5-degree band inside +-20 degrees then holds no row, zonal_profile gives NaN for it and ps.max() propagates it. measure.py --surface runs at 2048x1024 and never hits it. Frozen as surface_contrasts.coarse.nan_itcz.
- metrics_match: metrics.judge_surface: max(SURFACE_SPREAD_FACTOR * sd, floor) stays NaN with a single map only because of Python's max argument order (the first argument wins when a comparison fails); max(floor, ...) would turn a one-map run into a pass. Not a bug, a fragile idiom a port will reorder. Frozen as judge_surface.one_map.nan.
- metrics_match: metrics.spectral_slope: a constant patch whose value is not exactly representable (0.4) does not give NaN: patch.mean() leaves a last-bit residual, the mean-subtracted power is ~1e-32, `radial > 0` holds and polyfit fits rounding noise (slope about -7.3, different per summation order). Only an exactly representable constant (0.5) is frozen as NaN.
- metrics_match: match._match_by_zone: np.interp over src + k * 1e-9 is extremely steep at tied values (slopes up to ~1e6). A map made of bytes has many exact ties; a one-ulp change of an input texel equal to a plateau value moves its output by up to 2.5e-4 per quantile stage and up to 8e-2 for match_to_earth with levels=0 (the whole map goes through one table). A float32/GPU port will not reproduce those texels bit for bit. Frozen with a `frac` of outliers plus a block-mean summary, and fed with the exact stored inputs.
- metrics_match: metrics.area_histogram: the bin edges take the dtype of the map (float32 edges for a float32 map, float64 k * 0.1 for float64: the literal 0.7 falls in bin 6), and byte-derived maps (u8 / 255) put thousands of texels exactly on 0.2, 0.4, 0.6, 0.8, so how a map is converted changes hist and hist_l1 by 0.03 to 0.09 (a one-ulp nudge of every texel).
- metrics_match: metrics.patches silently casts a float64 map to float32 (img.astype(np.float32)) before sampling; a port that keeps float64 texels differs by up to 2.7e-8 (frozen as patches.f64_input.tiny8x4).
- metrics_match: metrics.compare reads the generated dict's zonal_lat for the selection (the reference dict's own zonal_lat is ignored) and a NaN band (empty latitude band) inside |lat| < 70 poisons zonal_rmse silently.
- tiles: cloudgen/plans.py records the thinning MASK of poisson_points (plan.keep) but not the uniforms it came from, so a port cannot test its own thinning from the plan alone; worked around inside the case module with a recording proxy of numpy's Generator: every poisson.* case stores the uniforms as input thin_u and the mask a port must derive as output keep_from_u (asserted equal to plan0.keep when the case is built)
- tiles: cloudgen/tiles.py: _sample uses field.shape[0] for BOTH axes, so it is silently wrong on a non-square field (no tile or size field of the product is non-square); pinned as out of contract by sample.non_square.rows_index_both_axes.5x9, not changed
- tiles: cloudgen/tiles.py: periodic_fbm computes k as np.fft.fftfreq(size) * size, which is inexact for non-power-of-two sizes (size 14: 4.999999999999999 for k = 5) and changes which modes pass `kk >= low`; harmless for the shipped power-of-two sizes, pinned as an artifact case, not changed
- tiles: golden/store.py compare() reports a params mismatch only as 'params differ' without saying which key, which makes a stale case hard to debug; not touched (store code)
- tiles: golden/data/v1/manifest.json and CATALOG.md still contain no tiles rows because the group was frozen with --no-manifest (hard rule 1): run `python -m golden.freeze` without --no-manifest once, after every group is final, before committing
