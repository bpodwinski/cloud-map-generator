# Design log

This tool started inside the Exokosm game (2026-11-10), as a timeboxed spike promoted to a dev-time bake. This is its design log as written there, kept for the measurements, the tuning rounds and the physics audit. Paths that point into the game's repository (`src/`, `public/`, `.agents/`, `scripts/`) do not exist here; `data/earth_cloud_fraction_bluemarble_2048.png` is the reference it names `earth_cloud_coverage_bluemarble_2048.png`.


Phase 0 of [.agents/plans/cloud-map-generator.md](../../.agents/plans/cloud-map-generator.md): can a
physics-driven weather structure, with no training, produce global cloud-coverage maps that look
like weather (fronts, cyclone spirals with the right handedness, an equatorial band, clear
subtropics) with no seam on the sphere? Started as a timeboxed spike; promoted to `scripts/cloudgen/` at P1. It runs at DEV TIME ONLY (owner decision): the game loads what it bakes, never runs it.

## Verdict

**The structure works; the fine texture is where procedural detail stops and the refiner (P3) has to take over.**

- For: cyclones and anticyclones turn the right way in each hemisphere (and reverse on a retrograde
  body). Their count and size come from the Rossby deformation radius (Earth: 5 lows per
  hemisphere at 51 deg). Fronts are drawn by stirring a humidity tracer, not painted on. The zonal
  mean is calibrated in units of the Hadley edge, a twin of `cloud_climate_type.ts` checked against
  the TypeScript itself. There is no antimeridian seam (ratio 0.94) and no polar pinch, because
  everything is evaluated on 3D directions. A 4096x2048 map takes 80 s on the CPU, numpy only.
- P0b (same day) answered part of that without AI: a fine texture chosen by the weather regime (closed cells, open cells, streets) brings the spectrum to Earth's slope (-2.95 against -2.90) and the median object from 751 to 462 km^2 (Earth 308). What stays procedural: the fibres inside frontal shields, and how irregular real cells are. Next without AI: synthesis from real satellite patches.
- Against (P0): the small-scale texture (cumulus fields, the fibres inside frontal shields) is still
  further from Earth than the structure is. Earth's spectrum is flatter (slope -2.90 against -3.29)
  and its cloud objects smaller (median 308 against 751 km^2). Five rounds of tuning (below) moved
  the numbers, but the look stays recognisably procedural up close. That is the argument for P3.
- Not shown yet: the generated map drawn in the engine. The Cloud Lab override loads it
  (`?coverage=`, status `ready`, the deck reads it), but at the lab's poses the deck looks the
  same with Earth's real map, the generated one and the baseline: a uniform veil from 4 radii, and
  near-identical frames above the deck. Showing a generated map in the engine needs a body that
  draws its coverage from orbit (the game's Sol:Earth) or a lab pose that resolves it. That is the
  first item of P1.

## Measurements (4096x2048 seed 42, box-averaged to 2048 like Earth's map)

`python -I measure.py <maps> --sheet <png>`; reference = `public/assets/clouds/earth_cloud_coverage_bluemarble_2048.png`.

| Map | Mean | Zonal RMSE | Hist L1 | Coherence | Spectral slope | Big objects | Median object | Seam |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Earth (Blue Marble) | 0.404 | 0 | 0 | 0.487 | -2.90 | 80 % | 308 km^2 | 1.39 |
| Generated, P0 (one fBm texture) | 0.366 | 0.092 | 0.27 | 0.419 | -3.29 | 75 % | 751 km^2 | 0.94 |
| **Generated, P0b (regime texture)** | 0.366 | 0.092 | 0.26 | 0.413 | **-2.95** | 76 % | **462 km^2** | 0.95 |
| Isotropic baseline (fBm) | 0.366 | 0.092 | 0.26 | 0.360 | -3.26 | 70 % | 847 km^2 | 1.03 |
| P0c (life cycles, the default) | 0.366 | 0.091 | 0.27 | 0.395 | -2.68 | 76 % | **385 km^2** | 1.03 |
| P0c at 8192x4096 (350 s) | 0.366 | 0.091 | 0.27 | 0.411 | -2.81 | 77 % | 385 km^2 | 1.16 |
| P0c + `--match-earth` (optional) | 0.407 | 0.067 | 0.04 | 0.431 | -3.11 | 83 % | 424 km^2 | 1.14 |

## P0c (2026-11-10): life cycles, texel detail, products for the deck

Requested by the owner, and by the cloud-deck session after PR 376 (main merged into this branch):

- **Cyclone life cycles** (`flow.py`). Each low has a stage from 0.25 to 1.45. A young open wave
  lifts air on its warm front, east of the center. A mature comma has its head poleward of the
  center. An occluded low has a dry slot at its center. Winding and size follow the stage. Median
  object 462 -> 385 km^2 (Earth 308). This is now the default.
- **Multi-scale match to Earth** (`match.py`, `--match-earth`, off by default). This is
  Heeger-Bergen per band and per latitude zone, with the maps interpolated between zone centers (a
  hard zone switch printed horizontal lines, measured). The histogram and the zonal profile match by
  construction. On what the match does NOT target, it is mixed: coherence improves (0.43), but the
  spectrum gets steeper and the objects bigger than with life cycles alone. Kept as an option, not
  the default.
- **Texel octave** (`noise.octave_ladder` + sub-texel cumulus). The fine ladder is now anchored at
  the top, 0.95 x the map's limit. A texel holding N independent cumulus of ~1.4 km gets the
  binomial variance f(1-f)/N, which is resolution-aware by construction. `detailLostTo2x` (the
  deck session's measure, Python twin in `metrics.detail_lost_2x`) goes 0.014 -> 0.027 at 4096
  (Storm Map 0.039, Blue Marble 0.040 at 2048), 0.046 at 8192 where the 15 km cells resolve. Pushing it to 0.038 (bigger sub-texel term) flattens
  the spectrum past Earth's (-2.48) and costs coherence: the white sub-texel term is what flattens,
  so it is kept small.
- **Products** (`products.py`, written next to every map, encoding in the sidecar):
  - **type**: the weather map B ladder, 0 = stratiform, 1 = convective. Shields take the convective
    tendency of their latitude, the cumulus population takes its regime's type.
  - **high_share**: of the cloud present, the share with tops above 440 hPa. It is conditional, as
    MODIS defines it and as the deck draws it. Its zonal base is MODIS's own profile in units of the
    Hadley edge, plus cirrus over comma heads and anvils over deep convection. Zonal means against
    MODIS, both hemispheres:

    | Latitude | 0 | 10 | 20 | 30 | 45 | 55 | 65 |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | Generated | 0.35 | 0.39 | 0.25 | 0.29 | 0.28 | 0.26 | 0.26 |
    | MODIS | 0.39 | 0.42 | 0.27 | 0.32 | 0.31 | 0.26 | 0.25 |

  - **wind**: east and north components in m/s, RGB at (byte - 128) x 0.5 m/s, on the back-trace
    grid (1024x512, smooth). The vortices' flow plus a climatological zonal profile scaled by
    sqrt(g H). Earth: westerlies 14-15 m/s at 45 deg, trades -8 m/s at 15 deg. The model's advection
    time is taken as 3 days [model choice].
- **Width from a target texel** (`climate.map_width_for_texel`, `--texel-km 3`, owner-agreed rule):
  circumference / texel, rounded up to a power of two, capped at 16384. Earth 16384, Mars and Titan
  8192.
- **Pole rows** (raised by the deck session): the first rows of the PNG look stretched because a row
  near the pole is a circle tens of km around spread over the whole width. Their spread across
  longitude falls toward the pole (std 22 at row 0 against 76 at row 100, 0..255) as it should.
  Polar gnomonic patches show no pinch. It is the equirectangular projection, the same on Earth's map.

## Surface terms (2026-11-11): the generator sees land, sea and relief

`cloudgen/surface.py`, decision records in its docstring (`game-design`, `astrophysics` tags). Four terms move
the CLIMATOLOGICAL cloud amount the field is cut at (a per-texel threshold offset = anomaly / row sensitivity,
rows still solved for their mean), each centered on Earth's own land/sea mixture (tables from Earth's DEM):

1. land is drier (continentality, 300 km): ocean - land -0.08 in the convergence zone, +0.12 subtropics to storm
   tracks, +0.04 more under the highs;
2. marine stratocumulus off the west coasts (east of the highs, on a prograde spin), +0.35 at the coast,
   e-folding 1200 km, closed cells over the decks and thinned over land;
3. orographic uplift tanh(V . grad h / 0.05 m/s) x 0.25 with the generator's own wind, and a rain shadow 0.2
   over 800 km downwind where the air came down 0.2-1 km;
4. the ITCZ crest moved 5 deg per 0.20 of hemispheric land contrast (Earth: 0.39 N against 0.19 S -> +5.0 deg),
   the flanks fading over half the Hadley edge.

**The surface comes from the game's own data** (`export_bodies.mjs`): the heightfield DEM and the sea level
(hydrosphere level, else the DEM datum of a water-cloud body). Only **Sol:Earth** has one (29.1 % land).
Dev:Gaia and Dev:Aurora have procedural GPU relief and no hydrosphere: `surface: null`, they stay blind. Sol:Mars is
all land and not bakeable. A bake of today's list and library would therefore come out as v1.

**Measured against Earth** (`measure.py --surface Sol:Earth`, 4 seeds 42/7/1/2 at 4096 on Earth's surface,
box-averaged to 2048; |lat| < 60). The reference is a JJAS 2001 3-day composite. Its north-south contrasts are mostly
season: its ocean-only N-S is -0.090, and its monsoon makes northern subtropical land cloudy. One map's contrasts
swing by up to 0.04 between seeds. A metric passes when |seed mean - Earth| <= 2.2 x the spread across seeds
(at least 0.01, 2.5 deg for the ITCZ) + the reference's ocean-only season term (`metrics.judge_surface`):

| Metric | Earth | Surface (mean, spread) | Tolerance | Verdict | Blind (mean) | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| land - ocean | -0.053 | -0.072 (0.034) | 0.075 | PASS | -0.017 | PASS |
| subtropical land - ocean (15-35) | -0.063 | -0.126 (0.043) | 0.095 | PASS | -0.045 | PASS |
| north - south (0-60) | -0.051 | -0.008 (0.001) | 0.100 | PASS | -0.000 | PASS |
| mid-latitude north - south (40-60) | -0.137 | -0.066 (0.003) | 0.092 | PASS | +0.001 | MISS |
| ITCZ crest | +5.0 N | +4.1 N (0.6) | 2.5 | PASS | -0.3 | MISS |
| deck - open ocean | +0.046 | +0.042 (0.144) | 0.317 | PASS | -0.102 | PASS |

Also (blind -> surface, seed means): zonal RMSE 0.092 -> 0.076, spectral slope -2.71 -> -2.87 (Earth -2.90), median object ~414 -> 462 km^2 (seed medians)
(Earth 308, worse), mean 0.366 -> 0.361. Read honestly:
- the tolerances are wide (one map's weather, plus the season), so the blind generator passes 4 of 6 too. The
  surface run is the one that gets the sign of every contrast and moves each one toward Earth;
- the north-south contrast that a seasonless model can hold is the land part: -0.008 overall, -0.066 at 40-60;
- **the decks are the weak term**: visible off Peru and Namibia in 2 seeds of 4, swamped by the weather in the
  others (deck - open from -0.130 to +0.215). Earth's Peru deck is there in every month. A deck that holds would
  need its own boundary-layer population (not a threshold offset), to decide;
- contact sheet: `output/cloudgen/surface_sheet.png` (row 1 Earth, rows 2-5 the four seeds; the whole map and
  views of Peru, California, Namibia, the Sahara, the Andes and the Congo, coasts in orange).

## Astrophysics audit (2026-11-10, `astrophysics` skill, three fixes applied by the owner's decision)

Reference bodies swept through the model: Earth, Mars, Titan, Venus, a 0.4-day and a 4-day Earth,
and a 2-radius super-Earth.

1. **Cloud amount was Earth's on every body** (wrong look). The target zonal coverage and the high
   share are anchored on Blue Marble and MODIS. Venus is 100 % sulfuric-acid overcast; Mars has thin
   water-ice belts and polar hoods; Titan has sparse methane clouds of order 1 % [measured: Pioneer
   Venus and Venus Express, MGS and MRO, Cassini]. **Fixed:** `BodyInputs.cloud_regime`, and the
   generator refuses anything but water (`climate.is_bakeable`; `--experiment` only for a test,
   never a bake).
2. **The deformation radius ignored latitude** (wrong number). L_R = N H / f with f = 2 Omega
   sin(lat) [derived]. **Fixed:** f is taken at the body's own storm track, with Earth's 51 deg as
   the anchor. The 0.4-day Earth goes from 401 to 893 km, and from 19 to 8 lows per hemisphere;
   Earth is unchanged.
3. **The wind scale was sqrt(g H)** (wrong number), inconsistent with Held-Hou, whose
   angular-momentum wind is U ~ g H / (Omega a) [derived]. **Fixed:** `products.zonal_wind_scale`.
   Mars goes from 8.7 to 12.2 m/s, the 0.4-day Earth from 12 to 4.8. A global Hadley cell bakes NO
   wind: Venus super-rotates at ~100 m/s retrograde at its cloud tops [measured], a regime outside
   the model.

Kept as concessions:
- The Hadley edge uses the scale height, not the tropopause height, and Earth's equator-to-pole
  contrast (the TS twin says the same). Mars comes out at 42 deg.
- One storm track, where a fast rotator has several jets (Rhines scale) [unverified recall: GCM
  studies].
- The eddy threshold (raw edge < 50 deg) [model choice].
- Cell sizes stay Earth's. The scale height is not a proxy for the convective boundary layer
  (Earth: H 8 km, marine BL 1-2 km; Mars: H 11 km, daytime CBL up to ~10 km [unverified recall:
  radio occultations]). The fix in (1) keeps the bake on Earth-like bodies.

Correct, per the audit: the reversed handedness of a retrograde spin [derived]; Titan and Venus as
global cells without storm tracks [measured]; the 3 km texel sizing.

Not done (proposed by the deck session, to decide):
- **Texel from the atmosphere.** Cell size follows the depth of the convective boundary layer, so a
  body's target texel could scale with it. Not done: it needs the `astrophysics` skill's check of
  the boundary-layer depths first (that is not the scale height).
- **Cube faces as the delivered product.** Not free here: the calibration solves a threshold per
  LATITUDE row and the back-trace is on an equirectangular grid. It would need a resampled
  calibration, and it is a shader change on the deck's side.

- The zonal RMSE is the same with and without the flow. It is the symmetric target against Earth's
  real asymmetry (no season, no land and sea), not the generator.
- Coherence (bands and fronts) moves from the baseline toward Earth; tuning toward it overshot to
  0.70 at first.

## Realism tuning (2026-11-12): `GenParams::realistic()`

The products (`gen`, `serve`, the MCP server, the browser) run `GenParams::realistic()`; `GenParams::default()` stays the port of the Python generator,
which the golden cases pin. Measured against Earth's Blue Marble map at 2048 wide, `compute: cpu`, Earth, mean of seeds 1 to 4 (`measure` rows):

| metric | port (`default`) | `realistic` | Earth |
| --- | ---: | ---: | ---: |
| histogram distance to Earth (L1) | 0.287 | 0.128 | 0 |
| structure coherence | 0.386 | 0.496 | 0.487 |
| cloud in big objects (%) | 72.6 | 83.9 | 80.3 |
| median object area (km2) | 534 | 298 | 308 |
| spectral slope (40-400 km) | -2.63 | -2.65 | -2.90 |
| pole spread, north / south | 0.131 / 0.104 | 0.051 / 0.056 | 0 / 0.029 |

What changed, one knob at a time first (each swept alone, then together; the knobs interact: the histogram and the coherence pull apart):
- `shield_ramp` 0.35 -> 1.0 and `shield_texture` 0.3 -> 0.5: softer shield edges and thinner interiors, so the map has the grays of thin cloud
  (histogram L1 0.287 -> 0.12 on its own).
- `a_fine` 0.6 -> 0.3, `fine_top` 0.95 -> 0.85, `cumulus_depth` 1.6 -> 1.5: less texel-scale grain (median object 534 -> 300 km2).
- `backtrace_time` 1.0 -> 1.3 (new knob; 48 steps per unit): the air is traced back further, so fronts wrap into commas and filaments
  (coherence +0.03 to +0.1 by value; 1.6 and more marbles the map like Jupiter).
- `polar_smooth` (new knob): each row is averaged along longitude over the `1 / cos(latitude)` texels it oversamples, from about 70 degrees up
  (`core/src/polar.rs`, applied by every engine's output and by the banded sinks). A pole is one point; the unfiltered rows held a texel of noise each.

Not changed: the mean and the zonal RMSE vs Earth (0.366 and 0.092 against 0.404 and 0) come from the target zonal profile, which the thresholds
are solved to, not from these knobs; Earth's own pole rows are the satellite's saturated ice (a pole spread of 0), so those two metrics are not
targets. Not done: a separate cirrus layer and fronts that join one low to the next need a new term in `compose` and in its WGSL kernel and a new
flow plan (and new golden cases).

## Tuning record (sheets v1-v8, `output/cloudgen/`)

| Version | Change | Result |
| --- | --- | --- |
| v1 | sigmoid of the stirred field | spirals and fronts, but binary (52 % clear, 25 % full) and over-banded (coherence 0.70), chevrons in the tropics |
| v2 | linear ramp, softer shear | Earth's histogram, but fine billows drowned the cyclones in worms |
| v3 | fine detail weighted by 4c(1-c) | leopard spots |
| v4 | jet on the storm track, trades nearly off | uniform grain everywhere |
| v5-v6 | **two populations**: crisp synoptic shields + partial cumulus around them; streaks inside shields | cyclones read again, fibrous shields |
| v7-v8 | softer cumulus, octaves below Nyquist, 4096 | the P0 row above |
| P0b | fine texture chosen by the regime (`texture.py`): closed cells under the subtropical subsidence, open cells in cold air behind fronts, streets along the trades and outbreaks, three physical cell sizes (30/60/120 km) | spectrum at Earth's slope, smaller objects, visible open-cell mesh at 40 N |

## Files

- `cloudgen/climate.py`: Hadley edge and convective tendency (TS twins), target zonal coverage,
  deformation radius.
- `cloudgen/flow.py`: zonal shear plus vortices, back-trace on the sphere. The decision record is in
  its docstring.
- `cloudgen/generate.py`: the synoptic field and the two-population composition, calibrated per
  latitude.
- `cloudgen/noise.py`: seeded 3D Perlin, fBm and Worley cells, chunked; fBm has unit variance by construction.
- `cloudgen/texture.py` (P0b): the regime weights and the regime texture.
- `cloudgen/products.py` (P0c): type, high share and wind maps. `cloudgen/match.py`: the optional
  multi-scale match to Earth.
- `cloudgen/surface.py` (2026-11-11): the body's surface (land fraction, elevation) and the four land/sea
  terms; their decision records are its docstring. `export_bodies.mjs` writes the surfaces from the game's DEM.
- `cloudgen/metrics.py`, `measure.py`: the comparison against Earth, measured on gnomonic patches;
  `metrics.surface_contrasts` / `judge_surface` and `measure.py --surface` for the land/sea contrasts.
- `cloudgen/export.py`: 16-bit north-up master, 8-bit preview, the shader's raw R8 south-up `.gz`,
  and a sidecar JSON.
- `lab_shots.mjs`: the maps on Earth's deck in the Cloud Lab. Headed, with client diagnostics.
- `parity_fixture.mjs`: writes `tests/fixtures/climate_parity.json` from the engine's TS.

## Run

```bash
# environment outside the repo (uv, Python 3.12, numpy/scipy/pillow/pytest)
UV_PROJECT_ENVIRONMENT=D:/tmp/cloudgen-spike/.venv uv sync --project scripts/cloudgen
cd scripts/cloudgen
python -I gen.py --seed 42 --width 4096            # width from --texel-km 3 when omitted; --isotropic, --match-earth, --body mars|titan|fast
python -I measure.py ../../output/cloudgen/earth_s42/coverage_earth_s42_4096x2048.png --sheet ../../output/cloudgen/sheet.png
node export_bodies.mjs                             # bodies.json + output/cloudgen/surface/ (the DEMs' land and relief)
python -I gen.py --seed 42 --width 4096 --surface Sol:Earth   # an Earth-like map on Earth's own surface
python -I measure.py <maps>_surf_4096x2048.png --surface Sol:Earth --surface-sheet ../../output/cloudgen/surface_sheet.png
python -I measure.py --earth-tables                # surface.py's centering tables, from Earth's DEM
python -m pytest -q                                # 69 tests
node parity_fixture.mjs                            # after changing cloud_climate_type.ts
```

Cloud Lab: `/play/admin/dev/cloud-lab.html?profile=earth&coverage=/play/output/cloudgen/earth_s42/coverage_earth_s42_4096x2048.png&coverageLonDeg=0`.

## Concessions [model choice]

Frozen flow (vortices neither move nor evolve). No season. Land and sea, relief and the
hemispheric asymmetry only through `surface.py`, and only for a body the game gives a surface (today
Sol:Earth); the flow itself ignores the surface (no stationary waves, no monsoon). The target profile is anchored on one Blue Marble composite, whose polar caps (past 70
deg) are contaminated by ice. The cloud type (weather B) still comes from the procedural slices in
the lab, not from this generator.
