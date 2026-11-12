// GPU kernels of cloudgen/generate.py: compose (the two cloud populations), the 36-step per-row threshold solve
// (solve_thresholds), the row sensitivity, the surface-term targets and offsets, and the counter-based sub-texel
// noise. The CPU port they are checked against is cloud-map-generator-core/src/generate.rs.
//
// Everything is f32. Where the CPU port divides by a constant of GenParams the kernels multiply by the
// reciprocal the host rounds once (one ulp, far inside the f32 tolerances), and where the CPU clips a quotient
// that can overflow f32 (compose.sweep_u freezes s = 3e38) the numerator is clamped first: a clamp to the
// width of the ramp changes nothing for any finite u, because everything beyond it saturates anyway.
//
// Layout of a band: `rows` rows of `width` texels, row-major, texel i = row * width + col. A band is a slice of
// the map's rows (a map larger than a storage binding is processed band by band). Every row is independent, so
// a band needs nothing of its neighbors; `row0` (the band's first row in the whole map) only matters for the
// counter-based noise, whose stream is indexed by the texel's index in the WHOLE map.
//
// Bindings (group 0); each entry point statically uses only its own subset, and no entry uses more than 7
// storage buffers (the WebGPU default limit is 8):
//   0 params   uniform
//   1 buf_a    storage, read        s          | anomaly (row_target_main, surface_offset_main)
//   2 buf_b    storage, read        fine       | target_zonal, one per row (surface kernels)
//   3 buf_c    storage, read        streak (when FLAG_STREAK_ARRAY)       | target, one per row (surface_offset_main)
//   4 buf_d    storage, read        subtexel (when FLAG_SUB)              | sens, one per row (surface_offset_main)
//   5 buf_off  storage, read        threshold offset (when offset mode is column or full)
//   6 row_in   storage, read        one value per row: the target (solve_main), the threshold (the others)
//   7 out      storage, read_write  thresholds / sensitivities / targets (one per row) or one value per texel
//   group 3, binding 0: zero_ctx, uniform of four u32 zeros, only for fill_subtexel_main (see the barrier below)
//
// The row kernels run ONE WORKGROUP PER ROW (256 threads, a strided pass over the row, then a tree reduction in
// workgroup memory): the bisection evaluates the whole row at every step and only the row mean decides.

const WG: u32 = 256u;

const FLAG_SUB: u32 = 1u;           // subtexel is bound and used
const FLAG_STREAK_ARRAY: u32 = 2u;  // streak is an array (else the constant factor streak_factor)
// bits 2..3: offset mode. 0: the constant offset_scalar, 1: one value per row (buf_off[row]), 2: one per texel.
const OFFSET_SHIFT: u32 = 2u;

struct Params {
    width: u32,
    rows: u32,
    row0: u32,
    flags: u32,
    ramp: f32,           // shield_ramp
    inv_ramp: f32,       // float32(1 / shield_ramp)
    texture: f32,        // shield_texture
    cumulus_max: f32,
    depth: f32,          // cumulus_depth
    inv_depth: f32,      // float32(1 / cumulus_depth)
    a_fine: f32,
    streak_factor: f32,  // 1 - texture * clip(0.5 + 0.5 streak, 0, 1) of a scalar streak
    offset_scalar: f32,
    step: f32,           // sensitivity_step
    sens_floor: f32,     // 0.05
    inv_width: f32,      // float32(1 / width)
    steps: u32,          // bisection steps (36)
    lo0: f32,            // initial bracket (-10)
    hi0: f32,            // initial bracket (10)
    noise_scale: f32,    // fill_subtexel_main: the factor applied to the unit normal, float32(1 / sqrt(n_clouds))
    seed_lo: u32,        // fill_subtexel_main: the 64-bit seed of the counter-based generator
    seed_hi: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> buf_a: array<f32>;
@group(0) @binding(2) var<storage, read> buf_b: array<f32>;
@group(0) @binding(3) var<storage, read> buf_c: array<f32>;
@group(0) @binding(4) var<storage, read> buf_d: array<f32>;
@group(0) @binding(5) var<storage, read> buf_off: array<f32>;
@group(0) @binding(6) var<storage, read> row_in: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(3) @binding(0) var<uniform> zero_ctx: vec4<u32>;

var<workgroup> partial: array<f32, 256>;

// ---------------------------------------------------------------------------------------------------------
// compose

fn offset_at(i: u32, row: u32) -> f32 {
    let mode = (P.flags >> OFFSET_SHIFT) & 3u;
    if (mode == 2u) {
        return buf_off[i];
    }
    if (mode == 1u) {
        return buf_off[row];
    }
    return P.offset_scalar;
}

fn streak_factor_at(i: u32) -> f32 {
    if ((P.flags & FLAG_STREAK_ARRAY) != 0u) {
        // A shield is not uniformly overcast: thinner streaks along the flow let its fraction dip.
        return 1.0 - P.texture * clamp(0.5 + 0.5 * buf_c[i], 0.0, 1.0);
    }
    return P.streak_factor;
}

// compose() of generate.py for texel i at the threshold t (the row's threshold plus the texel's offset):
//   u = s - t
//   shield  = clip(0.5 + u / ramp, 0, 1) * (1 - texture * clip(0.5 + 0.5 streak, 0, 1))
//   density = cumulus_max * smoothstep(-depth, 0, u)
//   cumulus = clip(density * (1 + a_fine * fine), 0, 1)
//   cumulus = clip(cumulus + sqrt(cumulus (1 - cumulus)) * subtexel, 0, 1)      (with the sub-texel term)
//   coverage = shield + (1 - shield) * cumulus
fn compose_texel(i: u32, t: f32) -> f32 {
    let u = buf_a[i] - t;
    let shield = clamp(0.5 + clamp(u, -P.ramp, P.ramp) * P.inv_ramp, 0.0, 1.0) * streak_factor_at(i);
    // smoothstep(-depth, 0, u): k = clip((u + depth) / depth, 0, 1), clamped first like the ramp
    let k = (clamp(u, -P.depth, 0.0) + P.depth) * P.inv_depth;
    let density = P.cumulus_max * (k * k * (3.0 - 2.0 * k));
    var cumulus = clamp(density * (1.0 + P.a_fine * buf_b[i]), 0.0, 1.0);
    if ((P.flags & FLAG_SUB) != 0u) {
        // A texel holding N independent clouds has a fraction of binomial variance f (1 - f) / N around the
        // field's value (subtexel = unit white noise / sqrt(N)).
        cumulus = clamp(cumulus + sqrt(cumulus * (1.0 - cumulus)) * buf_d[i], 0.0, 1.0);
    }
    return shield + (1.0 - shield) * cumulus;
}

// The sum over the 256 threads of a workgroup: a tree reduction in workgroup memory. Every thread of the
// workgroup must call it (it holds barriers) and every thread gets the total.
fn reduce_row(acc: f32, lid: u32) -> f32 {
    partial[lid] = acc;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            partial[lid] = partial[lid] + partial[lid + stride];
        }
        workgroupBarrier();
    }
    let total = partial[0];
    workgroupBarrier();
    return total;
}

// The sum of compose over one row at the threshold t (+ the offsets of the row).
fn row_sum(base: u32, row: u32, lid: u32, t: f32) -> f32 {
    var acc = 0.0;
    for (var c = lid; c < P.width; c = c + WG) {
        let i = base + c;
        acc = acc + compose_texel(i, t + offset_at(i, row));
    }
    return reduce_row(acc, lid);
}

// solve_thresholds: per row, the threshold t with mean(compose(t + offset)) == target, by bisection on
// [lo0, hi0] (36 steps). The decision is `sum > target * width`, the CPU's strict `mean > target` without a
// division (an exactly overcast row sums to exactly `width` and never triggers, as on the CPU). The result is
// the midpoint of the final bracket.
@compute @workgroup_size(256)
fn solve_main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    if (row >= P.rows) {
        return;
    }
    let base = row * P.width;
    let limit = row_in[row] * f32(P.width);
    var lo = P.lo0;
    var hi = P.hi0;
    for (var it = 0u; it < P.steps; it = it + 1u) {
        let mid = 0.5 * (lo + hi);
        let sum = row_sum(base, row, lid, mid);
        if (sum > limit) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if (lid == 0u) {
        out[row] = 0.5 * (lo + hi);
    }
}

// row_sensitivity: -d mean(compose) / d t at the row's threshold by a central difference of half-width `step`,
// floored at 0.05. The two means are formed as ONE mean of the per-texel difference (the same number, without
// the cancellation of two large f32 sums: the difference of two means near 0.5 loses about 20 times more
// precision than the mean of the differences).
@compute @workgroup_size(256)
fn sensitivity_main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    if (row >= P.rows) {
        return;
    }
    let base = row * P.width;
    let thr = row_in[row];
    var acc = 0.0;
    for (var c = lid; c < P.width; c = c + WG) {
        let i = base + c;
        acc = acc + (compose_texel(i, thr - P.step) - compose_texel(i, thr + P.step));
    }
    let total = reduce_row(acc, lid);
    if (lid == 0u) {
        out[row] = max(total * P.inv_width / (2.0 * P.step), P.sens_floor);
    }
}

// The final coverage: one thread per texel, a 2-D dispatch (x = column blocks of 64, y = rows).
@compute @workgroup_size(64)
fn compose_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= P.width || row >= P.rows) {
        return;
    }
    let i = row * P.width + col;
    out[i] = compose_texel(i, row_in[row] + offset_at(i, row));
}

// ---------------------------------------------------------------------------------------------------------
// The surface terms of generate(): local = clip(target_zonal + anomaly, 0.01, 0.99); the row target is the mean
// of local; the threshold offset of a texel is -(local - target) / sensitivity.

// buf_a = anomaly (texels), buf_b = target_zonal (one per row); out = the row targets.
@compute @workgroup_size(256)
fn row_target_main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    if (row >= P.rows) {
        return;
    }
    let base = row * P.width;
    let tz = buf_b[row];
    var acc = 0.0;
    for (var c = lid; c < P.width; c = c + WG) {
        acc = acc + clamp(tz + buf_a[base + c], 0.01, 0.99);
    }
    let total = reduce_row(acc, lid);
    if (lid == 0u) {
        out[row] = total / f32(P.width);
    }
}

// buf_a = anomaly, buf_b = target_zonal, buf_c = target, buf_d = sensitivity (one per row each); out = the
// per-texel offset.
@compute @workgroup_size(64)
fn surface_offset_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= P.width || row >= P.rows) {
        return;
    }
    let i = row * P.width + col;
    let local = clamp(buf_b[row] + buf_a[i], 0.01, 0.99);
    out[i] = -(local - buf_c[row]) / buf_d[row];
}

// ---------------------------------------------------------------------------------------------------------
// The counter-based normal: normal(seed, texel index), a pure function of its two arguments, implemented
// identically in Rust (cloud_map_generator_gpu::compose::counter_normal) and tested bit for bit against it.
//
// A PCG hash of the index and the seed gives two 32-bit words; Box-Muller turns them into one normal:
//   z = sqrt(-2 ln u1) cos(2 pi u2),  u1 = (24 bits + 1) / 2^24 in (0, 1],  u2 = 22 bits / 2^22 in [0, 1).
// The transcendental functions are the single-precision polynomials of the Cephes library (logf, sinf, cosf on
// [0, pi/4]) and a Newton sqrt, written with additions and multiplications only, in a fixed order: IEEE
// operations a GPU and a CPU round identically. What breaks that is a compiler fusing a multiply and an add
// into one fma, or reassociating (the same trap as in df64.wgsl): every arithmetic result therefore goes
// through zb(), an OR with a zero the compiler cannot see (the uniform zero_ctx, bound as group 3), which
// makes it opaque. Measured on the RTX 5080 (Vulkan): with zb() replaced by a no-op, 14 percent of 1M normals
// differ from the Rust twin in the last bits; with it, none, on Vulkan and on DX12. The cost (one integer op
// per arithmetic op) is paid once per map.

fn zb(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) | zero_ctx.x);
}
fn bm(a: f32, b: f32) -> f32 {
    return zb(a * b);
}
fn ba(a: f32, b: f32) -> f32 {
    return zb(a + b);
}
fn bs(a: f32, b: f32) -> f32 {
    return zb(a - b);
}

// PCG hash (M. E. O'Neill's pcg output permutation over a 32-bit LCG step, as popularized for GPUs by
// Jarzynski and Olano, "Hash Functions for GPU Rendering", 2020).
fn pcg32(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// ln(u) for u in (0, 1]: Cephes logf (single precision, about 1e-7 relative).
fn cn_ln(u: f32) -> f32 {
    let bits = bitcast<u32>(u);
    var e = i32((bits >> 23u) & 255u) - 126;
    var x = bitcast<f32>((bits & 0x007FFFFFu) | 0x3F000000u); // the fraction, in [0.5, 1)
    if (x < 0.70710678) {
        e = e - 1;
        x = bs(ba(x, x), 1.0);
    } else {
        x = bs(x, 1.0);
    }
    let z = bm(x, x);
    var y = 7.0376836292e-2;
    y = ba(bm(y, x), -1.1514610310e-1);
    y = ba(bm(y, x), 1.1676998740e-1);
    y = ba(bm(y, x), -1.2420140846e-1);
    y = ba(bm(y, x), 1.4249322787e-1);
    y = ba(bm(y, x), -1.6668057665e-1);
    y = ba(bm(y, x), 2.0000714765e-1);
    y = ba(bm(y, x), -2.4999993993e-1);
    y = ba(bm(y, x), 3.3333331174e-1);
    y = bm(bm(y, x), z);
    let ef = f32(e);
    if (e != 0) {
        y = ba(y, bm(ef, -2.12194440e-4));
    }
    y = bs(y, bm(0.5, z));
    var r = ba(x, y);
    if (e != 0) {
        r = ba(r, bm(ef, 0.693359375));
    }
    return r;
}

// sqrt(w) for w >= 0: the inverse square root from the integer estimate, three Newton steps, times w.
fn cn_sqrt(w: f32) -> f32 {
    if (w <= 0.0) {
        return 0.0;
    }
    let xh = bm(0.5, w);
    var y = bitcast<f32>(0x5F3759DFu - (bitcast<u32>(w) >> 1u));
    for (var k = 0; k < 3; k = k + 1) {
        let y2 = bm(y, y);
        let t = bm(xh, y2);
        y = bm(y, bs(1.5, t));
    }
    return bm(w, y);
}

// (sin psi, cos psi) for psi in [0, pi/4]: Cephes sinf and cosf.
fn cn_sincos(psi: f32) -> vec2<f32> {
    let z = bm(psi, psi);
    var ys = -1.9515295891e-4;
    ys = ba(bm(ys, z), 8.3321608736e-3);
    ys = ba(bm(ys, z), -1.6666654611e-1);
    ys = ba(bm(bm(ys, z), psi), psi);
    var yc = 2.443315711809948e-5;
    yc = ba(bm(yc, z), -1.388731625493765e-3);
    yc = ba(bm(yc, z), 4.166664568298827e-2);
    yc = bm(bm(yc, z), z);
    yc = bs(yc, bm(0.5, z));
    yc = ba(yc, 1.0);
    return vec2<f32>(ys, yc);
}

// pi / 2 / 2^22 as an f32
const CN_PSI_STEP: f32 = 3.7450704e-7;
// 2^-24 (exact)
const CN_TWO_M24: f32 = 5.9604644775390625e-8;

fn counter_normal(seed_lo: u32, seed_hi: u32, idx: u32) -> f32 {
    let k0 = pcg32(seed_lo ^ 0x9E3779B9u);
    let k1 = pcg32(seed_hi ^ k0 ^ 0x85EBCA6Bu);
    let x = pcg32(idx);
    let a = pcg32(x ^ k0);
    let b = pcg32((x + 0x9E3779B9u) ^ k1);
    // radius: sqrt(-2 ln u1) with u1 in (0, 1]
    let u1 = (f32(a >> 8u) + 1.0) * CN_TWO_M24;
    let radius = cn_sqrt(bm(-2.0, cn_ln(u1)));
    // angle: the quadrant from the top two bits of b, the position inside it from 22 more bits, folded to
    // [0, 2^21] with integer arithmetic so that only psi in [0, pi/4] reaches the polynomials
    let quadrant = b >> 30u;
    let r22 = (b >> 8u) & 0x3FFFFFu;
    let upper = r22 >= 0x200000u;
    let folded = select(r22, 0x400000u - r22, upper);
    let sc = cn_sincos(bm(f32(folded), CN_PSI_STEP));
    // phi = psi in the lower half of the quadrant and pi/2 - psi in the upper half
    let cos_phi = select(sc.y, sc.x, upper);
    let sin_phi = select(sc.x, sc.y, upper);
    var c = cos_phi;
    if (quadrant == 1u) {
        c = -sin_phi;
    } else if (quadrant == 2u) {
        c = -cos_phi;
    } else if (quadrant == 3u) {
        c = sin_phi;
    }
    return bm(radius, c);
}

// The sub-texel field: the unit counter-based normal of every texel of the band times noise_scale (the CPU
// divides the unit noise by sqrt(n_clouds); the host passes float32(1 / sqrt(n_clouds))). The counter is the
// texel's index in the whole map, (row0 + row) * width + col.
@compute @workgroup_size(64)
fn fill_subtexel_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if (col >= P.width || row >= P.rows) {
        return;
    }
    let idx = (P.row0 + row) * P.width + col;
    out[row * P.width + col] = bm(counter_normal(P.seed_lo, P.seed_hi, idx), P.noise_scale);
}
