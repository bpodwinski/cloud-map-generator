// df64: double-single arithmetic for WGSL, which has no f64.
//
// A df64 value is a vec2<f32> (hi, lo) with value = hi + lo and |lo| <= ulp(hi) / 2: about 48 bits of mantissa
// (relative error near 2^-45 for add and mul, 2^-43 for div) instead of f32's 24. Every function is prefixed
// `df64_` because WGSL has one global namespace: this file is concatenated in front of the kernels that use it.
//
// Algorithms (implemented here from the literature):
//   - TwoSum: D. E. Knuth, The Art of Computer Programming vol. 2, 1969; QuickTwoSum: T. J. Dekker, 1971.
//   - Veltkamp splitting and the split-based TwoProduct: G. W. Veltkamp 1968, Dekker 1971.
//   - The fma-based TwoProduct: Boldo and Muller, 2011 (exact when fma is a true fused multiply-add).
//   - add, mul, div, sqrt, floor on double-single pairs: Hida, Li and Bailey, "Quad-double arithmetic" (QD
//     library), 2000/2001; the GPU form: A. Thall, "Extended-precision floating-point numbers for GPU
//     computation", 2006.
//
// THE TRAP THAT WAS MEASURED (RTX 5080, wgpu 27, Vulkan and DX12): written naively, TwoSum returns an error
// term of exactly 0 for 84 percent of random inputs, because the shader compiler treats floating-point
// arithmetic as exact real arithmetic and folds (a + b) - a to b. WGSL does not guarantee otherwise. The cure
// used here is df64_barrier: x is passed through a bit-pattern OR with a ZERO THE COMPILER CANNOT SEE (a
// uniform the host fills with zeros), so it can neither fold through x nor reassociate around it. Cost: one
// integer op. Two more traps:
//   - fma() in WGSL is not required to be fused. `override DF64_FMA` picks the product: true = fma (cheaper),
//     false = the Veltkamp split (no fma needed, about 17 flops).
//   - the split multiplies by 4097: |a| above about 8e34 overflows. Subnormal results lose the error term:
//     stay clear of 1e-30. Out of range for this generator.
// Run df64.rs::probe on each device: it measures which of (fma | split) x (barrier | none) is exact there and
// returns the pipeline constants to use. `override DF64_BARRIER` turns the barrier off for devices that do not need it.
//
// REQUIRED BINDING: this file declares `@group(3) @binding(0) var<uniform> df64_ctx: vec4<u32>;` holding zeros.
// Bind df64.rs::Df64Zero there (GpuContext::dispatch_groups does it for you: pass it as group 3).

override DF64_FMA: bool = true;
override DF64_BARRIER: bool = true;

@group(3) @binding(0) var<uniform> df64_ctx: vec4<u32>;

const DF64_SPLITTER: f32 = 4097.0; // 2^12 + 1: splits a 24-bit mantissa into two 12-bit halves

// x, made opaque to the compiler: the same value, but it cannot be folded through or reassociated around.
fn df64_barrier(x: f32) -> f32 {
    if (DF64_BARRIER) {
        return bitcast<f32>(bitcast<u32>(x) | df64_ctx.x);
    }
    return x;
}

// TwoSum: (s, e) with s = fl(a + b) and a + b = s + e exactly, for any a and b.
fn df64_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = df64_barrier(a + b);
    let bb = df64_barrier(s - a);
    let e = (a - (s - bb)) + (b - bb);
    return vec2<f32>(s, e);
}

// QuickTwoSum: the same when |a| >= |b| (or a == 0).
fn df64_quick_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = df64_barrier(a + b);
    let d = df64_barrier(s - a);
    return vec2<f32>(s, b - d);
}

// Veltkamp split: a = hi + lo exactly, hi and lo each with at most 12 significant bits.
fn df64_split(a: f32) -> vec2<f32> {
    let t = df64_barrier(DF64_SPLITTER * a);
    let d = df64_barrier(t - a);
    let hi = t - d;
    return vec2<f32>(hi, a - hi);
}

// TwoProduct from a fused multiply-add: p = fl(a b), a b = p + e exactly (only if fma is fused).
fn df64_two_prod_fma(a: f32, b: f32) -> vec2<f32> {
    let p = df64_barrier(a * b);
    return vec2<f32>(p, fma(a, b, -p));
}

// TwoProduct from splitting: exact without fma.
fn df64_two_prod_split(a: f32, b: f32) -> vec2<f32> {
    let p = df64_barrier(a * b);
    let sa = df64_split(a);
    let sb = df64_split(b);
    let h = df64_barrier(sa.x * sb.x - p);
    let m = df64_barrier(sa.x * sb.y + sa.y * sb.x);
    let e = ((h + m) + sa.y * sb.y);
    return vec2<f32>(p, e);
}

fn df64_two_prod(a: f32, b: f32) -> vec2<f32> {
    if (DF64_FMA) {
        return df64_two_prod_fma(a, b);
    }
    return df64_two_prod_split(a, b);
}

fn df64_from_f32(x: f32) -> vec2<f32> {
    return vec2<f32>(x, 0.0);
}

fn df64_to_f32(a: vec2<f32>) -> f32 {
    return a.x + a.y;
}

// -a. The negation goes through the barrier: left plain, the compiler merged it into the two_sum inside
// df64_add (a + (-b) rewritten as a - b around the folded transforms) and df64_sub lost its low word.
fn df64_neg(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(df64_barrier(-a.x), df64_barrier(-a.y));
}

// a + b, the accurate ("IEEE") double-single addition of the QD library.
fn df64_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let s = df64_two_sum(a.x, b.x);
    let t = df64_two_sum(a.y, b.y);
    let u = df64_quick_two_sum(s.x, s.y + t.x);
    return df64_quick_two_sum(u.x, u.y + t.y);
}

fn df64_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return df64_add(a, df64_neg(b));
}

fn df64_add_f32(a: vec2<f32>, b: f32) -> vec2<f32> {
    let s = df64_two_sum(a.x, b);
    return df64_quick_two_sum(s.x, s.y + a.y);
}

fn df64_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let p = df64_two_prod(a.x, b.x);
    return df64_quick_two_sum(p.x, p.y + (a.x * b.y + a.y * b.x));
}

fn df64_mul_f32(a: vec2<f32>, b: f32) -> vec2<f32> {
    let p = df64_two_prod(a.x, b);
    return df64_quick_two_sum(p.x, p.y + a.y * b);
}

// a / b by long division: three quotient digits of f32 each, the remainder formed in df64.
fn df64_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let q1 = a.x / b.x;
    var r = df64_sub(a, df64_mul_f32(b, q1));
    let q2 = r.x / b.x;
    r = df64_sub(r, df64_mul_f32(b, q2));
    let q3 = r.x / b.x;
    return df64_add_f32(df64_quick_two_sum(q1, q2), q3);
}

// sqrt(a): one Newton step on the f32 inverse square root, the residual formed exactly (QD library).
fn df64_sqrt(a: vec2<f32>) -> vec2<f32> {
    if (a.x <= 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let x = inverseSqrt(a.x);
    let ax = a.x * x;
    let residual = df64_sub(a, df64_two_prod(ax, ax));
    return df64_add_f32(vec2<f32>(ax, 0.0), residual.x * (x * 0.5));
}

// floor(a) as a df64 (an integer): the low word decides when the high word is already an integer.
fn df64_floor(a: vec2<f32>) -> vec2<f32> {
    let fh = floor(a.x);
    if (fh == a.x) {
        return df64_quick_two_sum(fh, floor(a.y));
    }
    return vec2<f32>(fh, 0.0);
}

// a - floor(a), in [0, 1).
fn df64_fract(a: vec2<f32>) -> vec2<f32> {
    return df64_sub(a, df64_floor(a));
}
