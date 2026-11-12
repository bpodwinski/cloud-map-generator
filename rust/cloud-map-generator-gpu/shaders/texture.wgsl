// GPU kernels of cloudgen/texture.py (the CPU twin is cloud-map-generator-core/src/texture.rs): the regime weights of every
// texel, the cell-ramp accumulation of the regime texture and its final blend. The noise fields themselves
// (generic and street fBm, cell edges) come from noise.wgsl; the mixes of the directions and the mean/std
// reduction and standardization from stages.wgsl.
//
// Everything is f32 and follows the CPU port's order of operations (the precision of the golden `regime_weights`
// cases of float32 latitudes); a multiply and an add may be fused by the driver. The knees of the ramps are the
// f32 roundings of the Python floats (the weak scalars of numpy 2); a ramp of width `b - a` is written with the
// f32 rounding of that float64 difference (0.7 - 0.4 is 0.3 to f32, and so on): the Rust unit test
// `the_ramp_constants_are_the_f32_roundings_of_the_python_floats` reads this file and checks every one.
//
// One uniform struct (the layout of stages.wgsl); one thread per texel. Bindings (group 0): 0 params,
// 1..8 inputs (read), 9..14 outputs (read_write); an entry point statically uses a subset.

struct Params {
    n: u32,          // items of this dispatch (a chunk of the band)
    row_stride: u32, // threads per row of the 2-D dispatch
    base: u32,       // index, within the band, of the first item of this chunk
    width: u32,      // map columns (the row of an item is (base + i) / width)
    u0: u32,
    u1: u32,
    u2: u32,
    u3: u32,
    kv: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> in1: array<f32>;
@group(0) @binding(2) var<storage, read> in2: array<f32>;
@group(0) @binding(3) var<storage, read> in3: array<f32>;
@group(0) @binding(4) var<storage, read> in4: array<f32>;
@group(0) @binding(5) var<storage, read> in5: array<f32>;
@group(0) @binding(6) var<storage, read> in6: array<f32>;
@group(0) @binding(7) var<storage, read> in7: array<f32>;
@group(0) @binding(9) var<storage, read_write> out1: array<f32>;
@group(0) @binding(10) var<storage, read_write> out2: array<f32>;
@group(0) @binding(11) var<storage, read_write> out3: array<f32>;
@group(0) @binding(12) var<storage, read_write> out4: array<f32>;
@group(0) @binding(13) var<storage, read_write> out5: array<f32>;
@group(0) @binding(14) var<storage, read_write> out6: array<f32>;

fn k(i: u32) -> f32 {
    return P.kv[i / 4u][i % 4u];
}

fn thread_index(gid: vec3<u32>) -> u32 {
    return gid.y * P.row_stride + gid.x;
}

// _smoothstep(a, b, x): k^2 (3 - 2 k), k = clip((x - a) / (b - a), 0, 1); `ba` is f32(b - a).
fn ramp(a: f32, ba: f32, x: f32) -> f32 {
    let t = clamp((x - a) / ba, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// _gauss(x, mu, s): exp(-((x - mu) / s)^2), no 1/2 in the exponent.
fn gauss(x: f32, mu: f32, s: f32) -> f32 {
    let d = (x - mu) / s;
    return exp(-(d * d));
}

// ---------------------------------------------------------------------------------------------------------
// regime_weights(lat, lat_back, hadley_deg): the weights (sum <= 1, the rest is fBm) of the closed-cell,
// open-cell and street textures of every texel, with `convective`, `cold` and `total_raw` for diagnosis.
//   in1: lat (radians). u0 bit 0 clear: one f32 per texel; set: one f32 per ROW of the band (the latitude of a
//        grid row is the same for every column), indexed by (base + i) / width.
//   in2: lat_back (radians, one f32 per texel).
//   k0 = f32(180) / f32(pi) (np.degrees of a float32), k1 = f32(hadley_deg),
//   k2 = f32(max(1e-3, hadley_deg)) (the floor of convective_tendency).
//   out1 closed, out2 open, out3 street; the diagnostic entry point also out4 convective, out5 cold,
//   out6 total_raw (closed + open + street before the normalization).

struct Terms {
    closed: f32,
    open: f32,
    street: f32,
    convective: f32,
    cold: f32,
    total: f32,
}

fn lat_at(i: u32) -> f32 {
    if ((P.u0 & 1u) != 0u) {
        return in1[(P.base + i) / P.width];
    }
    return in1[i];
}

fn terms_at(lat: f32, lat_back: f32) -> Terms {
    let latd = lat * k(0u);
    let x = abs(latd) / k(1u);
    // convective_tendency(latd, hadley_deg): its own x, with the floor on the edge.
    let xc = abs(latd) / k(2u);
    let conv = clamp(0.15 + 0.85 * gauss(xc, 0.0, 0.3) + 0.4 * gauss(xc, 1.7, 0.35) - 0.05 * ramp(2.0, 0.6, xc), 0.0, 1.0);
    // Equatorward displacement of the air over the advection time: a cold-air outbreak.
    let cold = ramp(0.0, 8.0, abs(lat_back * k(0u)) - abs(latd));
    let closed = (1.0 - conv) * ramp(0.4, 0.3, x) * (1.0 - ramp(1.1, 0.4, x)) * (1.0 - cold);
    let open = cold * ramp(0.8, 0.4, x);
    let street = max(gauss(x, 0.55, 0.3) * (1.0 - closed), 0.6 * cold) * (1.0 - open);
    let total = closed + open + street;
    var scale = 1.0;
    if (total > 1.0) {
        scale = 1.0 / max(total, 1e-9);
    }
    return Terms(closed * scale, open * scale, street * scale, conv, cold, total);
}

@compute @workgroup_size(64)
fn regime_weights_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    let t = terms_at(lat_at(i), in2[i]);
    out1[i] = t.closed;
    out2[i] = t.open;
    out3[i] = t.street;
}

@compute @workgroup_size(64)
fn regime_weight_terms_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    let t = terms_at(lat_at(i), in2[i]);
    out1[i] = t.closed;
    out2[i] = t.open;
    out3[i] = t.street;
    out4[i] = t.convective;
    out5[i] = t.cold;
    out6[i] = t.total;
}

// ---------------------------------------------------------------------------------------------------------
// The cell fields of regime_texture: for one used cell diameter, with edges = cell_edges(...) (F2 - F1 of the
// Worley lattice), the closed-cell ramp smoothstep(0.03, 0.30, edge) and the open-cell ramp
// 1 - smoothstep(0.05, 0.25, edge), each weighted by the cell's weight w = k0 and accumulated in f32 in the
// order of the diameters (u0 = 1: the first cell, the accumulators are written, not added to).
//   in1: edges, out1: closed accumulator, out2: open accumulator.
@compute @workgroup_size(64)
fn cell_ramps_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    let e = in1[i];
    let w = k(0u);
    let c = w * ramp(0.03, 0.27, e);
    let o = w * (1.0 - ramp(0.05, 0.2, e));
    if (P.u0 != 0u) {
        out1[i] = c;
        out2[i] = o;
    } else {
        out1[i] = out1[i] + c;
        out2[i] = out2[i] + o;
    }
}

// ---------------------------------------------------------------------------------------------------------
// The blend of regime_texture. Each of the four fields is standardized with its whole-array statistics
// ((x - mean_hi) - mean_lo) / divisor, then
//   tex = generic (1 - wc - wo) + wc closed + wo open     (only when a cell diameter is used: u0 = 1)
//   tex = tex (1 - ws) + ws street
//   in1 generic fBm, in2 street fBm, in3 closed-cell sum, in4 open-cell sum (any buffer when u0 = 0),
//   in5 wc, in6 wo, in7 ws (the regime weights), out1 the texture.
//   k0..2 generic (hi, lo, divisor), k3..5 street, k6..8 closed, k9..11 open.
@compute @workgroup_size(64)
fn regime_blend_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    var t = ((in1[i] - k(0u)) - k(1u)) / k(2u);
    if (P.u0 != 0u) {
        let wc = in5[i];
        let wo = in6[i];
        let closed = ((in3[i] - k(6u)) - k(7u)) / k(8u);
        let open = ((in4[i] - k(9u)) - k(10u)) / k(11u);
        t = t * (1.0 - wc - wo) + wc * closed + wo * open;
    }
    let ws = in7[i];
    let street = ((in2[i] - k(3u)) - k(4u)) / k(5u);
    out1[i] = t * (1.0 - ws) + ws * street;
}
