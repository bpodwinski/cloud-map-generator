# Bodies file format

`bake_bodies` bakes a catalog of planets. The easy way is to pass them inline: `bake_bodies(bodies=[{"key": "alpha", "rotation_period_days": 1.2, "radius_km": 6000, "gravity_mps2": 9, "scale_height_km": 8}, ...])`, with the same physics and units as `generate_map` overrides (days, km, m/s^2, km; negative period = retrograde) and an optional `"cloud_regime"` (default "water") and `"measured"` (default false). No file is needed.

A bodies FILE is only needed for bodies with land and relief grids. It is a JSON file inside the server's output folder (its path is `output_folder` of `list_presets`) or a `--read-dir` directory, and `bake_bodies` takes it as `bodies_file` (a path relative to the output folder is fine). Its keys are camelCase, NOT the snake_case of the tool arguments:

```json
{
  "bodies": [
    {"key": "alpha", "rotationPeriodDays": 1.2, "radiusKm": 6000, "gravityMps2": 9,
     "scaleHeightKm": 8, "cloudRegime": "water"},
    {"key": "beta", "rotationPeriodDays": -2, "radiusKm": 7000, "gravityMps2": 11,
     "scaleHeightKm": 7, "cloudRegime": "water",
     "surface": {"width": 1024, "height": 512, "land": "beta_land.u8", "elevation": "beta_elev.i16"}},
    {"key": "mars", "rotationPeriodDays": 1.026, "radiusKm": 3389.5, "gravityMps2": 3.721,
     "scaleHeightKm": 11.1, "cloudRegime": "thin water ice"}
  ]
}
```

- `key` (required, unique, 1 to 64 letters, digits, spaces and `. _ : + -`): names the output files.
- `rotationPeriodDays`, `radiusKm`, `gravityMps2`, `scaleHeightKm` (numbers, required, within the ranges of `list_presets` `limits`).
- `cloudRegime`: only "water" bodies are baked; any other regime (or `"measured": true`, a real map exists) is skipped and named in the result's `skipped`.
- `surface` (optional): `width` and `height` (integers) of two raw grids stored next to the bodies file: `land` (file name; one unsigned byte per cell = land fraction, 0 sea to 255 all land; width x height bytes) and `elevation` (file name; little-endian int16, metres above the sea; width x height x 2 bytes). Both grids are equirectangular, north row first, row by row, and sit in the same folder as the bodies file.
- At most 32 bodies per call.
