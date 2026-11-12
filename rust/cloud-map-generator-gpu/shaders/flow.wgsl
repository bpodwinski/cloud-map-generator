// GPU kernels of cloudgen/flow.py: the zonal rate profile, the tangent velocity of the flow (zonal jets plus the
// vortex population), the midpoint back-trace on the sphere and the local ascent of every vortex kind (the comma
// ascent of a cyclone and the clear eye of a tropical storm included).
//
// Everything is f32. Points are packed as array<f32> with a stride of 3 (a vec3 has a 16-byte alignment in a
// storage buffer), one thread per point. Every per-vortex constant that does not depend on the point (the
// squared radii, the unit center, the comma head) is computed in f64 by the host and uploaded rounded to f32.

struct Params {
    hadley_deg: f32,
    jet: f32,
    trades: f32,
    polar_easterlies: f32,
    // prof(1.7): the zonal profile at the storm track, subtracted so that the rate is zero there (f64 on the host).
    prof_ref: f32,
    dt: f32,
    half_dt: f32,
    n_vortices: u32,
    steps: u32,
    // Always 0, filled by the host: a zero the shader compiler cannot see (see opaque3).
    zero: u32,
    pad1: u32,
    pad2: u32,
}

struct Vortex {
    center: vec3<f32>,
    omega: f32,
    // radius^2: the Gaussian of the velocity.
    rr_vel: f32,
    // Squared radius of the ascent Gaussian: (1.3 r)^2 anticyclone, (0.6 r)^2 tropical, (r (1 + 0.3 age))^2
    // for the comma of a cyclone.
    rr_asc: f32,
    // (0.12 r)^2 the clear eye of a tropical storm, (0.45 r)^2 the dry slot of an occluded cyclone.
    eye2: f32,
    // np.float32(ascent).
    weight: f32,
    // Comma head of a cyclone (unit vector).
    head: vec3<f32>,
    // (age - 1) * 2, used when dry_slot is set (age > 1).
    dry_weight: f32,
    // 0 cyclone, 1 anticyclone, 2 tropical.
    kind: u32,
    dry_slot: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> V: array<Vortex>;
@group(0) @binding(2) var<storage, read> pts: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;

const RAD_TO_DEG: f32 = 57.29577951308232;
const HALF_PI: f32 = 1.5707963267948966;
const WG: u32 = 64u;

// asin on [-1, 1] by the Cephes asinf polynomial (about 1e-7 absolute). The WGSL builtin is only specified to a
// loose bound, which differs from one adapter to another; this one is the same everywhere.
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

fn gauss(x: f32, mu: f32, s: f32) -> f32 {
    let d = (x - mu) / s;
    return exp(-(d * d));
}

// The profile prof(x) of Flow.zonal_rate, x = |lat| / phi_H.
fn prof(x: f32) -> f32 {
    let t = clamp((x - 2.0) / 0.8, 0.0, 1.0);
    let sm = t * t * (3.0 - 2.0 * t);
    return P.jet * gauss(x, 1.5, 0.4) - P.trades * gauss(x, 0.4, 0.4) - P.polar_easterlies * sm;
}

// Flow.zonal_rate: longitude displacement over the advection time at the latitude `lat` (radians).
fn zonal_rate(lat: f32) -> f32 {
    let x = abs(lat * RAD_TO_DEG) / P.hadley_deg;
    return prof(x) - P.prof_ref;
}

fn sq_dist(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let d = a - b;
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

fn normalize_v(v: vec3<f32>) -> vec3<f32> {
    let n = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return v / n;
}

// Flow.velocity at one point: the latitude is asin(clip(p_z, -1, 1)); the vortex distance and the cross product
// use the raw point. Vortex terms accumulate in vortex order, as the CPU does.
fn velocity(p: vec3<f32>) -> vec3<f32> {
    let lat = asin_f32(clamp(p.z, -1.0, 1.0));
    let zr = zonal_rate(lat);
    // cross(z, p) = (-p_y, p_x, 0)
    var v = vec3<f32>(zr * (-p.y), zr * p.x, 0.0);
    for (var i = 0u; i < P.n_vortices; i = i + 1u) {
        let c = V[i].center;
        let w = V[i].omega * exp(-sq_dist(p, c) / V[i].rr_vel);
        v = v + w * cross(c, p);
    }
    return v;
}

fn load3(i: u32) -> vec3<f32> {
    return vec3<f32>(pts[3u * i], pts[3u * i + 1u], pts[3u * i + 2u]);
}

fn store3(i: u32, v: vec3<f32>) {
    outp[3u * i] = v.x;
    outp[3u * i + 1u] = v.y;
    outp[3u * i + 2u] = v.z;
}

// Linear thread index of a (possibly 2-D tiled) dispatch: 65535 workgroups per dimension at most.
fn thread_index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.y * (nwg.x * WG) + gid.x;
}

@compute @workgroup_size(64)
fn zonal_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread_index(gid, nwg);
    if (i >= arrayLength(&pts)) {
        return;
    }
    outp[i] = zonal_rate(pts[i]);
}

@compute @workgroup_size(64)
fn velocity_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread_index(gid, nwg);
    if (i >= arrayLength(&pts) / 3u) {
        return;
    }
    store3(i, velocity(load3(i)));
}

// The velocity at the midpoint of a step from q: v(normalize(q - dt/2 v(q))). The midpoint is only the place
// where the velocity is sampled, so its f32 rounding does not accumulate.
fn midpoint_velocity(q: vec3<f32>) -> vec3<f32> {
    let v = velocity(q);
    let mid = normalize_v(q - P.half_dt * v);
    return velocity(mid);
}

// Flow.backtrace in plain f32: q <- normalize(q - dt v(mid)). Every step rounds the position to f32 (6e-8 per
// component): over 48 steps that is the dominant error, amplified by the shear around a strong vortex (see
// tests/gpu_flow_accuracy.rs for the measured size).
@compute @workgroup_size(64)
fn backtrace_f32_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread_index(gid, nwg);
    if (i >= arrayLength(&pts) / 3u) {
        return;
    }
    var q = load3(i);
    for (var s = 0u; s < P.steps; s = s + 1u) {
        q = normalize_v(q - P.dt * midpoint_velocity(q));
    }
    store3(i, q);
}

struct DS3 {
    hi: vec3<f32>,
    lo: vec3<f32>,
}

// x made opaque to the shader compiler: the same value, but it cannot be folded through or reassociated around.
// The error-free transforms below are rewritten to nothing otherwise (measured on an RTX 5080, Vulkan and DX12:
// a naive TwoSum returns an error term of 0 for most inputs). The zero comes from a uniform the compiler cannot see.
fn opaque3(x: vec3<f32>) -> vec3<f32> {
    return bitcast<vec3<f32>>(bitcast<vec3<u32>>(x) | vec3<u32>(P.zero));
}

// TwoSum (Knuth): hi = fl(a + b), a + b = hi + lo exactly.
fn two_sum3(a: vec3<f32>, b: vec3<f32>) -> DS3 {
    let s = opaque3(a + b);
    let bb = opaque3(s - a);
    return DS3(s, (a - (s - bb)) + (b - bb));
}

// QuickTwoSum (Dekker): the same when |a| >= |b|.
fn quick_two_sum3(a: vec3<f32>, b: vec3<f32>) -> DS3 {
    let s = opaque3(a + b);
    return DS3(s, b - opaque3(s - a));
}

// Flow.backtrace with the position kept as an unevaluated sum q = hi + lo of two f32 (about 48 bits): the
// update q <- normalize(q - dt v) is done with error-free sums, so no step rounds the position.
//
// The velocity is tangent (every term is a cross product with p), so |q - dt v|^2 = |q|^2 + dt^2 |v|^2 and the
// normalization is a scaling by 1 - eps with eps = 1 - 1/|q - dt v|, a number of size 1e-5: rounding eps or the
// product eps (q - dt v) is radial (along q) or a thousand times smaller than the position ulp, never a
// tangential error. The product dt v is also small (0.01) next to q, so its own rounding is far below the ulp
// of q. Only the sums that move q are kept exact.
@compute @workgroup_size(64)
fn backtrace_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread_index(gid, nwg);
    if (i >= arrayLength(&pts) / 3u) {
        return;
    }
    var q = load3(i);
    var ql = vec3<f32>(0.0);
    for (var s = 0u; s < P.steps; s = s + 1u) {
        let step = P.dt * midpoint_velocity(q);
        // u = q - step, exactly.
        let a = two_sum3(q, -step);
        let u = quick_two_sum3(a.hi, ql + a.lo);
        // q <- u (1 - eps): the norm of u from its high and low parts.
        let eps = 1.0 - inverseSqrt(dot(u.hi, u.hi) + 2.0 * dot(u.hi, u.lo));
        let b = two_sum3(u.hi, -(eps * u.hi));
        let r = quick_two_sum3(b.hi, u.lo - eps * u.lo + b.lo);
        q = r.hi;
        ql = r.lo;
    }
    store3(i, q);
}

// Flow.ascent: the local ascent (> 0) or subsidence (< 0) of every vortex, accumulated in vortex order.
@compute @workgroup_size(64)
fn ascent_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread_index(gid, nwg);
    if (i >= arrayLength(&pts) / 3u) {
        return;
    }
    let p = load3(i);
    var a = 0.0;
    for (var k = 0u; k < P.n_vortices; k = k + 1u) {
        let kind = V[k].kind;
        var g: f32;
        if (kind == 0u) {
            // The comma of a cyclone (Flow._comma_ascent): a Gaussian about the head, minus the dry slot of an
            // occluded low.
            g = exp(-sq_dist(p, V[k].head) / V[k].rr_asc);
            if (V[k].dry_slot != 0u) {
                g = g - V[k].dry_weight * exp(-sq_dist(p, V[k].center) / V[k].eye2);
            }
        } else if (kind == 1u) {
            g = exp(-sq_dist(p, V[k].center) / V[k].rr_asc);
        } else {
            // A tropical storm: a dense ring and a clear eye of radius 0.12 times the VORTEX radius.
            let d2 = sq_dist(p, V[k].center);
            g = exp(-d2 / V[k].rr_asc) - 1.6 * exp(-d2 / V[k].eye2);
        }
        a = a + V[k].weight * g;
    }
    outp[i] = a;
}

// Self-test of the error-free sum on this device: (a, b) pairs in, (hi, lo) of TwoSum(a, b) out. `lo` is the exact
// rounding error of a + b, so it is nonzero for most inputs unless the shader compiler has folded the transform.
@compute @workgroup_size(64)
fn two_sum_probe_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = thread_index(gid, nwg);
    if (i >= arrayLength(&pts) / 2u) {
        return;
    }
    let r = two_sum3(vec3<f32>(pts[2u * i]), vec3<f32>(pts[2u * i + 1u]));
    outp[2u * i] = r.hi.x;
    outp[2u * i + 1u] = r.lo.x;
}
