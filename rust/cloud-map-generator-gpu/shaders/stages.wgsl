// GPU kernels of the per-texel stages of generate() that every map needs (cloudgen/generate.py and
// cloudgen/sphere.py; the CPU twins are cloud-map-generator-core/src/generate.rs and sphere.rs): the grid directions, the
// bilinear upsampling of the coarse back-trace, along(p, back, f), the latitude of a direction, the moisture
// profile, the assembly of the synoptic field, a hierarchical mean/variance reduction and the standardization.
//
// Everything is f32 and follows the CPU port's order of operations (a multiply and an add may be fused by the
// driver; the golden tolerances were measured with that freedom). Directions are array<f32> with a stride of 3
// (a vec3 in a storage array has a 16-byte stride and would pad every texel), one thread per texel. A band is a
// run of whole map rows: buffers hold the band only (item = row_in_band * width + column) and `row0` is the
// band's first row in the map.
//
// One uniform struct for every kernel: the generic u32/f32 slots are named by each entry point below. Bindings
// (group 0): 0 params, 1..4 inputs (read), 5..6 outputs (read_write); an entry point statically uses a subset.

struct Params {
    n: u32,          // items of this dispatch (a chunk of the band)
    row_stride: u32, // threads per row of the 2-D dispatch
    base: u32,       // index, within the band, of the first item of this chunk
    width: u32,      // map columns
    row0: u32,       // first map row of the band
    height: u32,     // map rows
    src_w: u32,      // coarse grid columns (upsample)
    src_h: u32,      // coarse grid rows (upsample)
    kv: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> in1: array<f32>;
@group(0) @binding(2) var<storage, read> in2: array<f32>;
@group(0) @binding(3) var<storage, read> in3: array<f32>;
@group(0) @binding(4) var<storage, read> in4: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

const HALF_PI: f32 = 1.5707963267948966;

fn k(i: u32) -> f32 {
    return P.kv[i / 4u][i % 4u];
}

fn thread_index(gid: vec3<u32>) -> u32 {
    return gid.y * P.row_stride + gid.x;
}

// asin on [-1, 1] by the Cephes asinf polynomial (about 1e-7 absolute): the WGSL builtin is only specified to a
// loose bound that differs from one adapter to another. Same polynomial as flow.wgsl.
fn asin_f32(x: f32) -> f32 {
    let a = abs(x);
    var r: f32;
    if (a > 0.5) {
        let z = 0.5 * (1.0 - a);
        let s = sqrt(z);
        let p = (((4.2163199048e-2 * z + 2.4181311049e-2) * z + 4.5470025998e-2) * z + 7.4953002686e-2) * z
            + 1.6666752422e-1;
        r = HALF_PI - 2.0 * (s + s * z * p);
    } else {
        let z = a * a;
        let p = (((4.2163199048e-2 * z + 2.4181311049e-2) * z + 4.5470025998e-2) * z + 7.4953002686e-2) * z
            + 1.6666752422e-1;
        r = a + a * z * p;
    }
    return select(r, -r, x < 0.0);
}

// ---------------------------------------------------------------------------------------------------------
// grid_dirs: [cos lat cos lon, cos lat sin lon, sin lat] of every texel centre of a north-up equirectangular
// grid. The trigonometry of the columns (cos, sin of the longitude: in1, 2 f32 per column) and of the rows of
// the band (cos, sin of the latitude: in2, 2 f32 per row) is tabulated by the host in f64 and rounded to f32;
// a texel is two products, so it is within 1.5 ulp of the correctly rounded f64 direction the CPU port casts.
//   out1: 3 f32 per texel of the band.

@compute @workgroup_size(64)
fn grid_dirs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    let item = P.base + i;
    let r = item / P.width;
    let c = item - r * P.width;
    let cl = in2[2u * r];
    let sl = in2[2u * r + 1u];
    out1[3u * i] = cl * in1[2u * c];
    out1[3u * i + 1u] = cl * in1[2u * c + 1u];
    out1[3u * i + 2u] = sl;
}

// ---------------------------------------------------------------------------------------------------------
// upsample_dirs: the coarse grid of unit vectors (in1: src_w x src_h, 3 f32 each) sampled at the texel centres
// of the target (bilinear on the 3D vectors, longitude wraps, latitude clamps) and renormalized.
//
// The CPU port builds the target directions, takes their longitude and latitude back with atan2 and asin and
// computes the coarse pixel position in f64. The position of a target texel centre is known exactly from its
// indices, so it is computed here as an exact rational: fx = ((2 c + 1) src_w - width) / (2 width) (and fy
// alike), split by integer arithmetic into the cell x0 and the fraction tx = remainder / (2 width). No arcsine
// of a z quantized at 6e-8 (ill-conditioned just outside the poles: see the golden case tall_poles) and no
// f32 position error that grows with the width.
//
// Every integer division and remainder below has POSITIVE operands: the numerator is shifted by 2 width (it is
// at least src_w - width > -width before the shift). Measured on the RTX 5080 (Vulkan, wgpu 27), WGSL `%` of a
// negative i32 returns the FLOOR modulo (-8 % 32 = 24), not the truncated remainder the specification gives
// (-8): do not rely on the sign of `%` or of `/` with a negative operand.
//   out1: 3 f32 per texel of the band.

fn coarse_at(y: i32, x: i32) -> vec3<f32> {
    let j = 3u * u32(y * i32(P.src_w) + x);
    return vec3<f32>(in1[j], in1[j + 1u], in1[j + 2u]);
}

@compute @workgroup_size(64)
fn upsample_dirs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    let item = P.base + i;
    let r = item / P.width;
    let c = item - r * P.width;
    let row = P.row0 + r;
    let w = i32(P.width);
    let h = i32(P.height);
    let sw = i32(P.src_w);
    let sh = i32(P.src_h);
    // numerator + 2 width and + 2 height: positive, so that / and % are the unsigned ones.
    let px = (2 * i32(c) + 1) * sw - w + 2 * w;
    let py = (2 * i32(row) + 1) * sh - h + 2 * h;
    let x0 = px / (2 * w) - 1;       // floor(fx), in [-1, src_w - 1]
    let y0 = py / (2 * h) - 1;       // floor(fy), in [-1, src_h - 1]
    let tx = f32(px % (2 * w)) / f32(2 * w);
    let ty = f32(py % (2 * h)) / f32(2 * h);
    // longitude wraps, latitude clamps
    let x0w = select(x0, x0 + sw, x0 < 0);
    let x1w = select(x0 + 1, 0, x0 + 1 == sw);
    let y0c = clamp(y0, 0, sh - 1);
    let y1c = clamp(y0 + 1, 0, sh - 1);
    let a = coarse_at(y0c, x0w) * (1.0 - tx) + coarse_at(y0c, x1w) * tx;
    let b = coarse_at(y1c, x0w) * (1.0 - tx) + coarse_at(y1c, x1w) * tx;
    let v = a * (1.0 - ty) + b * ty;
    let nrm = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    let o = v / nrm;
    out1[3u * i] = o.x;
    out1[3u * i + 1u] = o.y;
    out1[3u * i + 2u] = o.z;
}

// ---------------------------------------------------------------------------------------------------------
// mix_dirs: normalize(a ka + b kb) per direction, ka = k0, kb = k1 (f32). along(p, back, f) is ka = f32(1 - f),
// kb = f32(f); the regime texture mixes with (0.75, 0.25), (0.1, 0.9) and (0.85, 0.15). No guard: an exactly
// antipodal pair at ka = kb gives a zero vector and NaN, as the CPU port does.
//   in1: a, in2: b (3 f32 per item), out1: 3 f32 per item.

@compute @workgroup_size(64)
fn mix_dirs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    let ka = k(0u);
    let kb = k(1u);
    let a = vec3<f32>(in1[3u * i], in1[3u * i + 1u], in1[3u * i + 2u]);
    let b = vec3<f32>(in2[3u * i], in2[3u * i + 1u], in2[3u * i + 2u]);
    let q = vec3<f32>(a.x * ka + b.x * kb, a.y * ka + b.y * kb, a.z * ka + b.z * kb);
    let nrm = sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    let o = q / nrm;
    out1[3u * i] = o.x;
    out1[3u * i + 1u] = o.y;
    out1[3u * i + 2u] = o.z;
}

// ---------------------------------------------------------------------------------------------------------
// lat_of_dirs: asin(clip(z, -1, 1)), the latitude (radians) of a direction.
//   in1: 3 f32 per item, out1: 1 f32 per item.

@compute @workgroup_size(64)
fn lat_of_dirs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    out1[i] = asin_f32(clamp(in1[3u * i + 2u], -1.0, 1.0));
}

// ---------------------------------------------------------------------------------------------------------
// moisture_profile(lat, hadley): 1 - 0.8 smoothstep(0.6, 2.2, |lat in degrees| / hadley), in f32 with the CPU
// port's constants: np.degrees of a float32 is x * (f32(180) / f32(pi)) (k0, formed on the host in f32), k1 is
// f32(hadley_deg).

fn moisture_profile_at(lat: f32) -> f32 {
    let x = abs(lat * k(0u)) / k(1u);
    let t = clamp((x - 0.6) / 1.6, 0.0, 1.0);
    return 1.0 - 0.8 * t * t * (3.0 - 2.0 * t);
}

//   in1: lat (1 f32 per item), out1: profile.
@compute @workgroup_size(64)
fn moisture_profile_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    out1[i] = moisture_profile_at(in1[i]);
}

// The synoptic field before standardization, in the CPU port's order:
//   s = a_moisture (profile(lat_back) + moisture_noise fbm_moisture)
//   s += a_ascent ascent
//   s += a_mid fbm_mid
// k0 = np.degrees constant, k1 = hadley, k2 = a_moisture, k3 = moisture_noise, k4 = a_ascent, k5 = a_mid (f32).
//   in1: lat_back, in2: fbm of the moisture texture, in3: ascent, in4: fbm of the mid-scale texture,
//   out1: s (1 f32 per item).
@compute @workgroup_size(64)
fn synoptic_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    let profile = moisture_profile_at(in1[i]);
    var s = k(2u) * (profile + k(3u) * in2[i]);
    s = s + k(4u) * in3[i];
    s = s + k(5u) * in4[i];
    out1[i] = s;
}

// ---------------------------------------------------------------------------------------------------------
// Mean and variance of a whole field: hierarchical, one pass over the data. A workgroup (256 threads) takes a
// tile of 4096 values (16 per thread, coalesced: thread t reads t, t + 256, ...), sums it pairwise in f32
// (a tree of 16 per thread, then a tree of 256 in shared memory), takes c = sum / count (a reference only) and
// reduces the deviations from it, S1 = sum (x - c) and S2 = sum (x - c)^2, the same way. The host recovers
// the exact tile statistics in f64, mean = c + S1 / count and M2 = S2 - S1^2 / count (the corrected two-pass
// algorithm: the f32 sums only ever add numbers of the size of the deviations), and merges the tiles with
// Chan's formula in f64. The error of the field's std is the f32 rounding of S2 over a tree of depth 12:
// about 1e-7 relative, 1e-6 at the very worst.
//   in1: the values, n = count, out1: 3 f32 per workgroup (c, S1, S2).

const TILE: u32 = 4096u;
var<workgroup> sh_sum: array<f32, 256>;
var<workgroup> sh_s1: array<f32, 256>;
var<workgroup> sh_s2: array<f32, 256>;

fn tree16(v: array<f32, 16>) -> f32 {
    let a = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
    let b = ((v[8] + v[9]) + (v[10] + v[11])) + ((v[12] + v[13]) + (v[14] + v[15]));
    return a + b;
}

@compute @workgroup_size(256)
fn moments_main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let tile = wid.x * TILE;
    var v: array<f32, 16>;
    for (var j = 0u; j < 16u; j++) {
        let idx = tile + j * 256u + t;
        var x = 0.0;
        if (idx < P.n) {
            x = in1[idx];
        }
        v[j] = x;
    }
    sh_sum[t] = tree16(v);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            sh_sum[t] = sh_sum[t] + sh_sum[t + s];
        }
        workgroupBarrier();
    }
    let count = min(TILE, P.n - tile);
    let c = sh_sum[0] / f32(count);
    var d: array<f32, 16>;
    var q: array<f32, 16>;
    for (var j = 0u; j < 16u; j++) {
        let idx = tile + j * 256u + t;
        var e = 0.0;
        if (idx < P.n) {
            e = v[j] - c;
        }
        d[j] = e;
        q[j] = e * e;
    }
    sh_s1[t] = tree16(d);
    sh_s2[t] = tree16(q);
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            sh_s1[t] = sh_s1[t] + sh_s1[t + s];
            sh_s2[t] = sh_s2[t] + sh_s2[t + s];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        out1[3u * wid.x] = c;
        out1[3u * wid.x + 1u] = sh_s1[0];
        out1[3u * wid.x + 2u] = sh_s2[0];
    }
}

// ---------------------------------------------------------------------------------------------------------
// standardize: ((x - mean_hi) - mean_lo) / divisor, the mean as a (hi, lo) pair of f32 (so a field whose mean is
// far above its std keeps its precision), k0 = hi, k1 = lo, k2 = the divisor (std, or 1 where it is not
// positive: numpy's rule of the CPU port).
//   in1: x, out1: the result (1 f32 per item).
@compute @workgroup_size(64)
fn standardize_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    out1[i] = ((in1[i] - k(0u)) - k(1u)) / k(2u);
}

// The same in place (out1 is read and written).
@compute @workgroup_size(64)
fn standardize_in_place_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = thread_index(gid);
    if (i >= P.n) {
        return;
    }
    out1[i] = ((out1[i] - k(0u)) - k(1u)) / k(2u);
}
