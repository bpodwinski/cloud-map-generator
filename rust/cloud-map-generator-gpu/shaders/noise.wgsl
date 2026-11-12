// GPU kernels of cloudgen/noise.py: improved Perlin noise (fBm of octaves with their own rotation and offset,
// billow, normalization), Worley F1/F2 and cell_edges (F2 - F1 of a rotated Worley lattice).
//
// Everything is f32 and follows the order of operations of the CPU port (cloud-map-generator-core/src/noise.rs), which
// follows numpy's: q = (p @ rot^T) * f + off, then the lattice cell, the fraction and the hash. The driver may
// contract a multiply and an add into one fused multiply-add, which numpy's BLAS matmul does too; the golden
// tolerances were measured with that freedom.
//
// Points are packed as three consecutive f32 (array<f32>, stride 3): a vec3 in a storage array has a 16-byte
// stride and would pad every point.
//
// One thread per point. A dispatch is 2-D (groups_x x groups_y workgroups of WG threads): the point index is
// gid.y * row_stride + gid.x with row_stride = groups_x * WG, so inputs larger than 65535 * WG points fit.
//
// Bindings (group 0); each entry point statically uses only its own subset:
//   0 params            uniform
//   1 points            storage, read        3 f32 per point
//   2 out               storage, read_write  one f32 per point (Worley: F1)
//   3 perlin_perm       storage, read        the DOUBLED Perlin table, 512 i32
//   4 octaves           storage, read        one 16-f32 record per octave (see OCT_*)
//   5 worley_perm       storage, read        the DOUBLED Worley table, 2048 i32
//   6 worley_jitter     storage, read        1024 x 3 f32
//   7 out2              storage, read_write  Worley: F2

const WG: u32 = 64u;

// Layout of one octave record (16 f32). cell_edges reads record 0: its rotation and frequency.
const OCT_STRIDE: u32 = 16u;
const OCT_ROT: u32 = 0u;   // rot[row][col] at OCT_ROT + 3 * row + col
const OCT_OFF: u32 = 9u;   // offset x, y, z
const OCT_FREQ: u32 = 12u; // float32(freq)
const OCT_AMP: u32 = 13u;  // float32(amp)

struct Params {
    n: u32,          // points of this dispatch
    row_stride: u32, // threads per row of the 2-D dispatch
    octaves: u32,
    billow: u32,     // 0 or 1
    norm: f32,       // float32(norm)
    mean1: f32,      // float32(PERLIN3_ABS_MEAN) for billow, else 0
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> perlin_perm: array<i32>;
@group(0) @binding(4) var<storage, read> octaves: array<f32>;
@group(0) @binding(5) var<storage, read> worley_perm: array<i32>;
@group(0) @binding(6) var<storage, read> worley_jitter: array<f32>;
@group(0) @binding(7) var<storage, read_write> out2: array<f32>;

fn point_at(i: u32) -> vec3<f32> {
    return vec3<f32>(points[3u * i], points[3u * i + 1u], points[3u * i + 2u]);
}

// c (an integer-valued f32, the result of floor) modulo m (a power of two) in [0, m): numpy's
// `floor(x).astype(int64) & (m - 1)` (two's complement for negative cells). Division and multiplication by a
// power of two are exact and so is the final subtraction, for any magnitude: no i32 conversion that could
// saturate for |x| >= 2^31 (where an f32 is a multiple of 256 and the CPU result is 0 or a multiple of 256).
fn cell_mod(c: f32, m: f32) -> i32 {
    return i32(c - m * floor(c * (1.0 / m)));
}

// ---------------------------------------------------------------------------------------------------------
// Perlin

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp(t: f32, a: f32, b: f32) -> f32 {
    return a + t * (b - a);
}

// The gradient selected by h & 15 dotted with (x, y, z): u = x if h < 8 else y; v = y if h < 4, x if h is 12
// or 14, else z; bit 0 of h negates u and bit 1 negates v.
fn grad(h: i32, x: f32, y: f32, z: f32) -> f32 {
    let g = h & 15;
    let u = select(y, x, g < 8);
    let v = select(select(z, x, g == 12 || g == 14), y, g < 4);
    return select(u, -u, (g & 1) != 0) + select(v, -v, (g & 2) != 0);
}

fn perlin(x: f32, y: f32, z: f32) -> f32 {
    let xi = floor(x);
    let yi = floor(y);
    let zi = floor(z);
    let xf = x - xi;
    let yf = y - yi;
    let zf = z - zi;
    let cx = cell_mod(xi, 256.0);
    let cy = cell_mod(yi, 256.0);
    let cz = cell_mod(zi, 256.0);
    let u = fade(xf);
    let v = fade(yf);
    let w = fade(zf);
    let a = perlin_perm[cx] + cy;
    let aa = perlin_perm[a] + cz;
    let ab = perlin_perm[a + 1] + cz;
    let b = perlin_perm[cx + 1] + cy;
    let ba = perlin_perm[b] + cz;
    let bb = perlin_perm[b + 1] + cz;
    let x1 = xf - 1.0;
    let y1 = yf - 1.0;
    let z1 = zf - 1.0;
    return lerp(
        w,
        lerp(
            v,
            lerp(u, grad(perlin_perm[aa], xf, yf, zf), grad(perlin_perm[ba], x1, yf, zf)),
            lerp(u, grad(perlin_perm[ab], xf, y1, zf), grad(perlin_perm[bb], x1, y1, zf)),
        ),
        lerp(
            v,
            lerp(u, grad(perlin_perm[aa + 1], xf, yf, z1), grad(perlin_perm[ba + 1], x1, yf, z1)),
            lerp(u, grad(perlin_perm[ab + 1], xf, y1, z1), grad(perlin_perm[bb + 1], x1, y1, z1)),
        ),
    );
}

// One row of `p @ rot^T` for the rotation at octaves[base ..]: sum_j p[j] rot[row][j], summed in order.
fn rotated(p: vec3<f32>, base: u32) -> vec3<f32> {
    let r = base + OCT_ROT;
    return vec3<f32>(
        p.x * octaves[r] + p.y * octaves[r + 1u] + p.z * octaves[r + 2u],
        p.x * octaves[r + 3u] + p.y * octaves[r + 4u] + p.z * octaves[r + 5u],
        p.x * octaves[r + 6u] + p.y * octaves[r + 7u] + p.z * octaves[r + 8u],
    );
}

@compute @workgroup_size(64)
fn fbm_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * params.row_stride + gid.x;
    if (i >= params.n) {
        return;
    }
    let p = point_at(i);
    var total = 0.0;
    for (var o = 0u; o < params.octaves; o++) {
        let base = o * OCT_STRIDE;
        let f = octaves[base + OCT_FREQ];
        let r = rotated(p, base);
        let q = vec3<f32>(
            r.x * f + octaves[base + OCT_OFF],
            r.y * f + octaves[base + OCT_OFF + 1u],
            r.z * f + octaves[base + OCT_OFF + 2u],
        );
        var v = perlin(q.x, q.y, q.z);
        if (params.billow != 0u) {
            v = abs(v) - params.mean1;
        }
        total += octaves[base + OCT_AMP] * v;
    }
    out[i] = total * params.norm;
}

// Raw Perlin noise at the points (no rotation, no octaves).
@compute @workgroup_size(64)
fn perlin_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * params.row_stride + gid.x;
    if (i >= params.n) {
        return;
    }
    let p = point_at(i);
    out[i] = perlin(p.x, p.y, p.z);
}

// ---------------------------------------------------------------------------------------------------------
// Worley

// (F1, F2): distances to the nearest and second nearest feature point (one jittered point per unit cell, in
// cell units). The cell index is floor of the point and the fraction is the point minus it; the 27 neighbor
// cells are searched with f32 squared distances and the square root is taken at the end. An exact tie gives
// F2 == F1. Cells wrap by & 1023 (negative cells wrap as two's complement).
fn worley(p: vec3<f32>) -> vec2<f32> {
    let cell = floor(p);
    let f = p - cell;
    let cx = cell_mod(cell.x, 1024.0);
    let cy = cell_mod(cell.y, 1024.0);
    let cz = cell_mod(cell.z, 1024.0);
    // Any real squared distance over the 27 cells is below 12: the sentinel never survives the first two cells.
    var f1 = 3.0e38;
    var f2 = 3.0e38;
    for (var dx = -1; dx <= 1; dx++) {
        let x = (cx + dx) & 1023;
        for (var dy = -1; dy <= 1; dy++) {
            let xy = worley_perm[x] + ((cy + dy) & 1023);
            for (var dz = -1; dz <= 1; dz++) {
                let h = worley_perm[worley_perm[xy] + ((cz + dz) & 1023)];
                let j = vec3<f32>(worley_jitter[3 * h], worley_jitter[3 * h + 1], worley_jitter[3 * h + 2]);
                let ex = j.x + f32(dx) - f.x;
                let ey = j.y + f32(dy) - f.y;
                let ez = j.z + f32(dz) - f.z;
                let d2 = ex * ex + ey * ey + ez * ez;
                f2 = select(min(f2, d2), f1, d2 < f1);
                f1 = min(f1, d2);
            }
        }
    }
    return vec2<f32>(sqrt(f1), sqrt(f2));
}

// Raw Worley at the points: F1 in out, F2 in out2.
@compute @workgroup_size(64)
fn worley_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * params.row_stride + gid.x;
    if (i >= params.n) {
        return;
    }
    let w = worley(point_at(i));
    out[i] = w.x;
    out2[i] = w.y;
}

// F2 - F1 of Worley at (p @ rot^T) * float32(frequency): the rotation and the frequency are octave record 0.
@compute @workgroup_size(64)
fn cell_edges_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * params.row_stride + gid.x;
    if (i >= params.n) {
        return;
    }
    let f = octaves[OCT_FREQ];
    let r = rotated(point_at(i), 0u);
    let w = worley(vec3<f32>(r.x * f, r.y * f, r.z * f));
    out[i] = w.y - w.x;
}
