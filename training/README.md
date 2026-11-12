# Cloud refiner: training

A small neural network (a residual U-Net, about 106 thousand weights, input: the cloud fraction and sin(latitude)) that gives a generated map the look of a real one. The generator
can run it as an optional last pass (`refine`): `rust/cloud-map-generator-core/src/refiner.rs` runs the same network in plain Rust, so no
ML runtime is needed, natively or in the browser.

Nothing here is needed to use the tool: the trained weights are committed as
`rust/cloud-map-generator-core/models/cloud-refiner.bin` (and its card, `cloud-refiner.json`: data, metrics, sha256).

## Retrain

Needs [uv](https://docs.astral.sh/uv/) and, for a reasonable time, a CUDA GPU (about 7 minutes on an RTX 5080).

```
uv run training/fetch_truecolor.py --count 450      # graded maps, the recipe of the reference map (what is shipped)
bash training/make_generated.sh rust/target/debug/cloud-map-generator.exe 100 200          # generated maps, matched to Earth
bash training/make_generated.sh rust/target/debug/cloud-map-generator.exe 1000 150 raw      # generated maps, as the generator makes them
uv run --index https://download.pytorch.org/whl/cu128 --with torch training/train.py --steps 6000
cargo test -p cloud-map-generator-core refiner     # the Rust network gives what PyTorch gave
```

- `fetch_truecolor.py` downloads days of MODIS Terra true colour and the cloud-free Blue Marble (GIBS, public tiles, no account) and inverts one against
  the other as the reference map was made, to graded cloud fractions at 2048 x 1024 (`training/data/gibs_tc/`, not committed). `fetch_gibs.py` is the older
  source: MODIS cloud-fraction masks, nearly binary, worse for this (`--sets gibs` to use them); it inverts
  the colour palette with GIBS's own colormap, and averages them to 2048 x 1024 (`training/data/`, not committed). NASA imagery is public
  domain; credit NASA GSFC / EOSDIS GIBS.
- `make_generated.sh` makes the generated maps the network is judged on.
- `train.py` does two things per step. On real maps, it degrades random 256 x 256 crops (blur, correlated noise, both) and trains the network to give the
  original back. On generated maps, it pulls the output's spectrum slope, structure coherence and histogram to Earth's reference (differentiable proxies of the
  tool's metrics), holding the layout. It writes the
  weights as a flat binary, not ONNX: the architecture lives twice, in `ARCH` of `train.py` and in `ARCH` of `refiner.rs`, and the Rust side
  refuses a file that does not match. Change both together and bump `VERSION`.
- The held-out maps give the card's metrics (`val_l1_degraded`, `val_l1_refined`).

An adversarial variant (a patch discriminator on unpaired maps) was tried and dropped: see `docs/HISTORY.md`.

## What the model is not

It learns what real clouds look like from graded maps made like the reference, and is steered on generated maps by proxies of the tool's metrics. Whether it
improves a map is measured with the tool's own metrics against Earth (`measure`), see `docs/HISTORY.md`. It is trained on Earth's water clouds: it is meant for Earth-like bodies.
