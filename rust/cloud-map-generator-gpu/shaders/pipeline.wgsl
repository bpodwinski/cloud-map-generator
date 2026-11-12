// Kernel of the GPU pipeline (src/pipeline.rs) that no other module owns: the reshaping of the regime
// weights by the surface (`surface_regime_weights` of cloudgen/generate.py, rust: cloud_map_generator_core::generate).
//
// Closed cells are a marine regime: thinned over land (by the continentality of the texel), filled in over
// the stratocumulus decks (the deck window of the row times the deck strength of the texel); the other two
// regimes give up the share the decks take. Per texel; the window of a row is a function of its latitude only
// and arrives as one value per row (the host evaluates it in f64, as the CPU port does).
//
// The CPU port runs the arithmetic in f64 on f64 weights and rounds the three results to f32; this kernel runs
// it in f32 (the weights are f32 on the GPU): the difference is a few 1e-7 of weight (a test measures it).
// The weights are updated in place.

struct Params {
    width: u32,
    rows: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> prm: Params;
@group(0) @binding(1) var<storage, read> continentality: array<f32>;
@group(0) @binding(2) var<storage, read> deck_strength: array<f32>;
// The deck window of each row of the band.
@group(0) @binding(3) var<storage, read> window: array<f32>;
@group(0) @binding(4) var<storage, read_write> closed_w: array<f32>;
@group(0) @binding(5) var<storage, read_write> open_w: array<f32>;
@group(0) @binding(6) var<storage, read_write> street_w: array<f32>;

// The guard and floor of the remaining share: 1e-9 of the CPU port.
const GUARD: f32 = 1e-9;

@compute @workgroup_size(64)
fn surface_weights_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= prm.width || gid.y >= prm.rows) {
        return;
    }
    let i = gid.y * prm.width + gid.x;
    let sea = 1.0 - continentality[i];
    let thinned = closed_w[i] * sea;
    let room = 1.0 - thinned;
    let deck = window[gid.y] * deck_strength[i];
    let filled = thinned + room * deck;
    let rest = open_w[i] + street_w[i];
    var scale = 1.0;
    if (rest > GUARD) {
        scale = min(1.0, (1.0 - filled) / max(rest, GUARD));
    }
    closed_w[i] = filled;
    open_w[i] = open_w[i] * scale;
    street_w[i] = street_w[i] * scale;
}
