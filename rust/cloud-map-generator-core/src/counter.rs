//! The counter-based standard normal of the banded engine: `normal(seed, texel_index)`, a pure function of its two
//! arguments, so that a band draws the sub-texel noise of its own texels without any other band (a stored array of
//! 537 million normals is what the banded engine exists to avoid).
//!
//! It is the twin of `counter_normal` in `cloud-map-generator-gpu/src/compose.rs` and of the shader's function in `compose.wgsl`,
//! bit for bit: a PCG hash of the index and the seed, then Box-Muller with a polynomial `ln`, an inverse-square-root
//! `sqrt` and a folded `cos` (additions, multiplications and integer operations only: no libm, so the value does not
//! depend on the platform, wasm32 included). About 1e-6 from the exact transform, mean 0 and standard deviation 1.
//!
//! The index is the texel's index in the whole map, `row * width + col`, as a `u32` (a 32768 x 16384 map has 2^29
//! texels). The map's sub-texel field is `counter_normal(seed, i) * subtexel_noise_scale(divisor)`.

/// `(pi / 2) / 2^22` as an f32: the angle step of the folded 22-bit position inside a quadrant.
pub const CN_PSI_STEP: f32 = 3.7450704e-7;
/// `2^-24`, exact.
#[allow(clippy::excessive_precision)]
const CN_TWO_M24: f32 = 5.9604644775390625e-8;

/// The PCG hash the generator is built on (the shader's `pcg32`).
pub fn pcg32(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// `ln(u)` for `u` in `(0, 1]`: the Cephes `logf` polynomial, additions and multiplications in the shader's order.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
fn cn_ln(u: f32) -> f32 {
    let bits = u.to_bits();
    let mut e = ((bits >> 23) & 255) as i32 - 126;
    let mut x = f32::from_bits((bits & 0x007F_FFFF) | 0x3F00_0000);
    if x < 0.70710678 {
        e -= 1;
        x = (x + x) - 1.0;
    } else {
        x -= 1.0;
    }
    let z = x * x;
    let mut y: f32 = 7.0376836292e-2;
    y = y * x + -1.1514610310e-1;
    y = y * x + 1.1676998740e-1;
    y = y * x + -1.2420140846e-1;
    y = y * x + 1.4249322787e-1;
    y = y * x + -1.6668057665e-1;
    y = y * x + 2.0000714765e-1;
    y = y * x + -2.4999993993e-1;
    y = y * x + 3.3333331174e-1;
    y = (y * x) * z;
    let ef = e as f32;
    if e != 0 {
        y += ef * -2.12194440e-4;
    }
    y -= 0.5 * z;
    let mut r = x + y;
    if e != 0 {
        r += ef * 0.693359375;
    }
    r
}

/// `sqrt(w)` for `w >= 0`: the inverse square root from the integer estimate, three Newton steps, times `w`.
fn cn_sqrt(w: f32) -> f32 {
    if w <= 0.0 {
        return 0.0;
    }
    let xh = 0.5 * w;
    let mut y = f32::from_bits(0x5F37_59DF - (w.to_bits() >> 1));
    for _ in 0..3 {
        let y2 = y * y;
        let t = xh * y2;
        y *= 1.5 - t;
    }
    w * y
}

/// `(sin psi, cos psi)` for `psi` in `[0, pi/4]`: the Cephes `sinf` and `cosf` polynomials.
#[allow(clippy::excessive_precision)]
fn cn_sincos(psi: f32) -> (f32, f32) {
    let z = psi * psi;
    let mut ys: f32 = -1.9515295891e-4;
    ys = ys * z + 8.3321608736e-3;
    ys = ys * z + -1.6666654611e-1;
    ys = ((ys * z) * psi) + psi;
    let mut yc: f32 = 2.443315711809948e-5;
    yc = yc * z + -1.388731625493765e-3;
    yc = yc * z + 4.166664568298827e-2;
    yc = (yc * z) * z;
    yc -= 0.5 * z;
    yc += 1.0;
    (ys, yc)
}

/// The counter-based standard normal of texel `index` for `seed`.
pub fn counter_normal(seed: u64, index: u32) -> f32 {
    let (seed_lo, seed_hi) = (seed as u32, (seed >> 32) as u32);
    let k0 = pcg32(seed_lo ^ 0x9E37_79B9);
    let k1 = pcg32(seed_hi ^ k0 ^ 0x85EB_CA6B);
    let x = pcg32(index);
    let a = pcg32(x ^ k0);
    let b = pcg32(x.wrapping_add(0x9E37_79B9) ^ k1);
    let u1 = ((a >> 8) as f32 + 1.0) * CN_TWO_M24;
    let radius = cn_sqrt(-2.0 * cn_ln(u1));
    let quadrant = b >> 30;
    let r22 = (b >> 8) & 0x003F_FFFF;
    let upper = r22 >= 0x0020_0000;
    let folded = if upper { 0x0040_0000 - r22 } else { r22 };
    let (sin_psi, cos_psi) = cn_sincos(folded as f32 * CN_PSI_STEP);
    let (cos_phi, sin_phi) = if upper { (sin_psi, cos_psi) } else { (cos_psi, sin_psi) };
    let c = match quadrant {
        0 => cos_phi,
        1 => -sin_phi,
        2 => -cos_phi,
        _ => sin_phi,
    };
    radius * c
}

/// `float32(1 / sqrt(n_clouds))`: the factor applied to the unit normal, from the divisor `float32(sqrt(n_clouds))`
/// of [`crate::generate::subtexel_scale`].
pub fn subtexel_noise_scale(divisor: f32) -> f32 {
    1.0 / divisor
}

/// The sub-texel field of the texels `first .. first + out.len()` of a map: `counter_normal(seed, i) * scale`.
pub fn fill_subtexel(seed: u64, first: u32, scale: f32, out: &mut [f32]) {
    crate::par::for_each_chunk_mut(out, crate::par::BLOCK, |b, chunk| {
        let base = first.wrapping_add((b * crate::par::BLOCK) as u32);
        for (j, v) in chunk.iter_mut().enumerate() {
            *v = counter_normal(seed, base.wrapping_add(j as u32)) * scale;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_normal_has_mean_zero_and_unit_variance() {
        let n = 1 << 20;
        let (mut s, mut s2) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            let v = f64::from(counter_normal(42, i));
            s += v;
            s2 += v * v;
        }
        let mean = s / f64::from(n);
        let var = s2 / f64::from(n) - mean * mean;
        assert!(mean.abs() < 0.005, "mean {mean}");
        assert!((var - 1.0).abs() < 0.01, "variance {var}");
    }

    #[test]
    fn a_texel_does_not_depend_on_its_neighbors_or_on_how_it_is_filled() {
        let mut a = vec![0.0_f32; 5000];
        fill_subtexel(7, 1000, 0.5, &mut a);
        for (j, &v) in a.iter().enumerate() {
            assert_eq!(v, counter_normal(7, 1000 + j as u32) * 0.5);
        }
        assert_ne!(counter_normal(7, 5), counter_normal(8, 5));
    }
}
