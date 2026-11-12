//! The port's own random generator (xoshiro256** seeded by SplitMix64).
//!
//! The Python generator draws from numpy's PCG64 and the golden cases record those draws as INPUTS
//! (`golden/README.md`), so no kernel of the port needs numpy's bit stream. Where the port draws numbers
//! itself (the vortex population, noise permutations, tile jitter, the sub-texel white noise), it draws them
//! here: a map is then statistically equivalent to Python's for the same seed, not identical.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        Rng { s: [splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x), splitmix64(&mut x)] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1), 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Uniform integer in [0, n), unbiased (Lemire's multiply-shift with rejection).
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        let mut m = (self.next_u64() as u128) * (n as u128);
        let mut low = m as u64;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = (self.next_u64() as u128) * (n as u128);
                low = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Standard normal (Marsaglia polar method).
    pub fn normal(&mut self) -> f64 {
        loop {
            let (u, v) = (2.0 * self.uniform() - 1.0, 2.0 * self.uniform() - 1.0);
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                return u * (-2.0 * s.ln() / s).sqrt();
            }
        }
    }

    /// A uniformly random permutation of 0..n (Fisher-Yates).
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.below(i as u64 + 1) as usize;
            p.swap(i, j);
        }
        p
    }

    /// Poisson-distributed count (Knuth's product method: for a small mean only).
    pub fn poisson(&mut self, lambda: f64) -> u32 {
        assert!((0.0..=30.0).contains(&lambda), "poisson() is for small means");
        let limit = (-lambda).exp();
        let (mut k, mut p) = (0u32, 1.0);
        loop {
            p *= self.uniform();
            if p <= limit {
                return k;
            }
            k += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_seed_dependent() {
        let a: Vec<u64> = (0..4).map(|_| Rng::new(7).next_u64()).collect();
        assert!(a.windows(2).all(|w| w[0] == w[1]));
        assert_ne!(Rng::new(7).next_u64(), Rng::new(8).next_u64());
    }

    #[test]
    fn uniform_normal_permutation_poisson_statistics() {
        let mut r = Rng::new(1);
        let n = 200_000;
        let u: Vec<f64> = (0..n).map(|_| r.uniform()).collect();
        assert!(u.iter().all(|&x| (0.0..1.0).contains(&x)));
        let mean = u.iter().sum::<f64>() / n as f64;
        assert!((mean - 0.5).abs() < 0.005, "{mean}");
        let z: Vec<f64> = (0..n).map(|_| r.normal()).collect();
        let m = z.iter().sum::<f64>() / n as f64;
        let v = z.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
        assert!(m.abs() < 0.01 && (v - 1.0).abs() < 0.02, "{m} {v}");
        let mut p = r.permutation(256);
        p.sort_unstable();
        assert_eq!(p, (0..256).collect::<Vec<_>>());
        let k: f64 = (0..n).map(|_| r.poisson(1.5) as f64).sum::<f64>() / n as f64;
        assert!((k - 1.5).abs() < 0.02, "{k}");
        let counts = (0..60_000).fold([0u32; 6], |mut c, _| {
            c[r.below(6) as usize] += 1;
            c
        });
        assert!(counts.iter().all(|&c| (c as f64 - 10_000.0).abs() < 400.0), "{counts:?}");
    }
}
