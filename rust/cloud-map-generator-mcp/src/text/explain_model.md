# cloud-map-generator: what the model does

## What it is
A physics-driven generator of global cloud-coverage maps: cloud fraction 0..1 on an equirectangular grid (north-up, no seam at the antimeridian, no pinch at the poles). No training and no AI: seeded noise, a model flow and a per-latitude calibration. It runs offline; a renderer loads the files.

## Inputs
- `rotation_period_days` (-400 to 400; negative = retrograde spin, 0 = no spin), `radius_km`, `gravity_mps2`, `scale_height_km`, `cloud_regime` (only "water" is generated unless `experiment` is on).
- A `seed` (same seed, same map, byte for byte) and a `width` of 256, 512, 1024, 2048, 4096 texels (the map is width x width/2).
- Optionally a surface (land fraction and relief): `surface='synthetic'` here, or real grids through a bodies file and `bake`.

## How a map is built
1. Held-Hou scaling turns rotation, radius and g x H into the Hadley edge, the latitude where the tropical cell ends (Earth 30 degrees). Every band is placed in units of it, so a faster spin gives narrower bands.
2. The Rossby deformation radius sets how many cyclones each hemisphere holds and how big they are (Earth: five lows at the 51-degree storm track). Handedness follows the Coriolis sign, so a retrograde spin reverses it.
3. A humidity tracer is back-traced through a model flow (zonal jets plus those vortices), so fronts and comma shapes are stirred out of the moisture field, not painted on.
4. Two populations are composed: crisp synoptic shields, and partial boundary-layer cumulus in the air around them. The fine texture follows the weather regime: closed cells, open cells, streets.
5. A threshold per latitude row is solved so the zonal mean equals the target profile, anchored on NASA's Blue Marble cloud fraction of Earth.
6. With a surface: drier land, marine stratocumulus decks off west coasts, uplift on windward slopes with a rain shadow behind, and the convergence band shifted toward the land-heavy hemisphere.

## Products
Coverage (16-bit master, 8-bit preview, raw R8 with rows south-up), cloud `type` (0 stratiform to 1 convective), `high_share` (share of cloud above 440 hPa), `wind` (m/s, east and north, coarser grid), a sidecar JSON and metrics.json.

## Where it runs
@@WHERE@@

## What it does not do
- Water clouds only. The cloud AMOUNT is Earth's: Venus (a total sulfuric deck), Mars (thin ice belts) and Titan (sparse methane) need `experiment` and are not those bodies' clouds.
- Frozen flow: vortices neither move nor evolve. No season, no day-night cycle, no time at all: one still map.
- One storm track per hemisphere; a fast rotator would really have several jets.
- Earth's cloud-cell sizes on every body; the depth of the convective layer is not derived.
- A slow spin gives one global Hadley cell: zonal bands, no cyclones, no wind field. Venus's real super-rotation is outside the model.
- The flow ignores the surface (no stationary waves, no monsoon); marine decks are weak, present in about half the seeds.
- The fine texture still reads as procedural up close, and the cloud type and high share are model fields calibrated on Earth, not measurements of the body.
- A spin faster than about 0.1 day (over 32 cyclones per hemisphere) is refused: the work grows with the vortices.

## Reading the metrics
Every row compares the map with Earth, measured the same way on both (gnomonic patches, so the grid's east-west stretch does not read as anisotropy). A gap against Earth is not a defect for a body that differs from Earth: a fast rotator is meant to have a different zonal profile. Judge the structure rows (coherence, spectral slope, object sizes) at width 2048 or more; the mean, zonal profile and histogram are reliable at any width. `match_earth` (made for widths of 2048 or more) forces Earth's statistics scale by scale: the histogram and zonal profile then agree by construction, so a closer score proves nothing.

## A good loop
`derive` first (milliseconds: bands, cyclone count, warnings), then `generate` at 512 to look at the picture and the metrics, change one input, repeat; render a final map at 2048 or 4096. Always look at the preview and the globe, not only the numbers.
