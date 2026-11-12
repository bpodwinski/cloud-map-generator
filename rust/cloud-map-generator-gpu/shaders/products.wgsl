// GPU kernels of cloudgen/products.py and of the shield of generate(): the shield fraction of every texel,
// the cloud TYPE map and the HIGH-CLOUD SHARE map. The CPU port they are checked against is
// cloud-map-generator-core/src/products.rs (the wind map is coarse and stays on the CPU).
//
// Everything is f32 (the CPU port runs the shield and the latitude terms in f64: the difference is a few 1e-8).
// What depends on the latitude only (the convective tendency c, the zonal base of the high share and the deep
// core term) is a function of the row: the host evaluates it per row in f64, exactly as the CPU port does, and
// uploads one value per row (`row_c`, `row_hi`); the kernels do the per-texel arithmetic.
//
// Layout: `rows` rows of `width` texels, row-major (a band of the map, see compose.wgsl). One thread per texel,
// a 2-D dispatch (x = column blocks of 64, y = rows).
//
// Bindings (group 0), each entry point statically uses its own subset:
//   0 params     uniform
//   1 buf_a      storage, read   s        (shield_main)  | shield (type_map_main, high_share_main)
//   2 buf_b      storage, read   thr, one per row (shield_main) | w_closed (type_map_main) | ascent (high_share_main)
//   3 buf_c      storage, read   threshold offset (shield_main) | w_open (type_map_main) | coverage (high_share_main)
//   4 buf_d      storage, read   w_street (type_map_main)
//   5 rows       storage, read   one f32 per row (c: type_map_main) or two (base, deep core: high_share_main)
//   7 out        storage, read_write  one f32 per texel

struct Params {
    width: u32,
    rows: u32,
    flags: u32,           // bits 0..1: offset mode of shield_main (0 constant offset_scalar, 1 per row, 2 per texel)
    pad0: u32,
    ramp: f32,            // shield_ramp
    inv_ramp: f32,        // float32(1 / shield_ramp)
    offset_scalar: f32,
    pad1: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> buf_a: array<f32>;
@group(0) @binding(2) var<storage, read> buf_b: array<f32>;
@group(0) @binding(3) var<storage, read> buf_c: array<f32>;
@group(0) @binding(4) var<storage, read> buf_d: array<f32>;
@group(0) @binding(5) var<storage, read> rows_in: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;

// The levels of products.py: closed cells are stratocumulus, open cells and streets are cumulus, the rest of
// the weight is the broken fields.
const TYPE_CLOSED: f32 = 0.45;
const TYPE_OPEN: f32 = 0.85;
const TYPE_STREET: f32 = 0.8;
const TYPE_BROKEN: f32 = 0.7;
const HIGH_COMMA_ASCENT: f32 = 0.25;
const HIGH_COVERAGE_FLOOR: f32 = 0.05;

// shield = clip(0.5 + (s - t) / shield_ramp, 0, 1) with t = the row threshold + the texel's offset. u is
// clamped to the width of the ramp first: anything beyond saturates anyway, and u / ramp would overflow f32
// for |u| near 3e38.
@compute @workgroup_size(64)
fn shield_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= P.width || row >= P.rows) {
        return;
    }
    let i = row * P.width + col;
    var off = P.offset_scalar;
    let mode = P.flags & 3u;
    if (mode == 2u) {
        off = buf_c[i];
    } else if (mode == 1u) {
        off = buf_c[row];
    }
    let u = buf_a[i] - (buf_b[row] + off);
    out[i] = clamp(0.5 + clamp(u, -P.ramp, P.ramp) * P.inv_ramp, 0.0, 1.0);
}

// type_map: clip(shield c + (1 - shield) cumulus, 0, 1) with c the convective tendency of the row and
// cumulus = the weights times the four type levels (the rest of the weight is the broken fields).
@compute @workgroup_size(64)
fn type_map_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= P.width || row >= P.rows) {
        return;
    }
    let i = row * P.width + col;
    let wc = buf_b[i];
    let wo = buf_c[i];
    let ws = buf_d[i];
    let cumulus = wc * TYPE_CLOSED + wo * TYPE_OPEN + ws * TYPE_STREET + (1.0 - wc - wo - ws) * TYPE_BROKEN;
    let shield = buf_a[i];
    out[i] = clamp(shield * rows_in[row] + (1.0 - shield) * cumulus, 0.0, 1.0);
}

// high_share: the CONDITIONAL share of the cloud with tops above 440 hPa:
//   share = clip(shield / max(coverage, 0.05), 0, 1)
//   extra = share (0.25 clip(ascent, 0, 1) + deep)         deep = 0.15 c^4 of the row
//   high  = clip(base + (1 - base) extra, 0, 1)            base = the MODIS zonal profile of the row
@compute @workgroup_size(64)
fn high_share_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= P.width || row >= P.rows) {
        return;
    }
    let i = row * P.width + col;
    let base = rows_in[2u * row];
    let deep = rows_in[2u * row + 1u];
    let share = clamp(buf_a[i] / max(buf_c[i], HIGH_COVERAGE_FLOOR), 0.0, 1.0);
    let extra = share * (HIGH_COMMA_ASCENT * clamp(buf_b[i], 0.0, 1.0) + deep);
    out[i] = clamp(base + (1.0 - base) * extra, 0.0, 1.0);
}
