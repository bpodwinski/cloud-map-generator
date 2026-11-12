//! Double-single ("df64") arithmetic for the GPU kernels, and the self-test that says how a device must run it.
//!
//! WGSL has no f64. [`WGSL`] is a library of `df64_*` functions on `vec2<f32>` (hi, lo) pairs: about 48 bits
//! of mantissa for the few places where float32 is not enough (the back-trace position update, the lattice
//! coordinate of very high noise octaves). Each operation costs 10 to 20 f32 operations: use it sparingly.
//!
//! The error-free transforms it is built on are NOT safe by default: on the development machine's RTX 5080 the
//! shader compilers (through wgpu 27 on Vulkan and DX12) fold `(a + b) - a` to `b`, so a naive TwoSum returns a
//! zero error term for most inputs. The library therefore passes intermediate values through an opaque barrier
//! (see the header of `shaders/df64.wgsl`), needs the zero buffer [`Df64Zero`] bound as group 3, and exposes two
//! pipeline constants, `DF64_FMA` and `DF64_BARRIER`. [`probe`] measures which configuration is exact on a
//! device and returns the cheapest one: call it once per device and build the pipelines with
//! [`Df64Config::constants`].

use crate::GpuContext;

/// The df64 library, to be concatenated IN FRONT of a kernel that uses it: `format!("{}{}", df64::WGSL, kernel)`.
pub const WGSL: &str = include_str!("../shaders/df64.wgsl");

const PROBE_WGSL: &str = include_str!("../shaders/df64_probe.wgsl");
const OPS_WGSL: &str = include_str!("../shaders/df64_ops.wgsl");

/// An f64 as a (hi, lo) pair of f32: `hi = x as f32`, `lo = (x - hi) as f32`.
pub fn split_f64(x: f64) -> [f32; 2] {
    let hi = x as f32;
    [hi, (x - hi as f64) as f32]
}

/// The f64 value of a (hi, lo) pair.
pub fn join_f64(d: [f32; 2]) -> f64 {
    d[0] as f64 + d[1] as f64
}

/// The uniform buffer of zeros the library's barrier reads (bind it as group 3, binding 0).
pub struct Df64Zero {
    pub buffer: wgpu::Buffer,
}

impl Df64Zero {
    pub fn new(gpu: &GpuContext) -> Df64Zero {
        Df64Zero { buffer: gpu.uniform_with("df64_zero", &[0u32; 4]) }
    }
}

/// The two pipeline constants of the library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Df64Config {
    /// true: the product uses `fma` (it must be fused on this device); false: the Veltkamp split.
    pub fma: bool,
    /// true: intermediate values go through the opaque barrier (needed when the compiler folds the transforms).
    pub barrier: bool,
}

impl Df64Config {
    /// The `override` constants to pass to `GpuContext::pipeline_with`.
    pub fn constants(&self) -> [(&'static str, f64); 2] {
        [("DF64_FMA", self.fma as u8 as f64), ("DF64_BARRIER", self.barrier as u8 as f64)]
    }
}

/// The four configurations, cheapest first.
pub const CONFIGS: [Df64Config; 4] = [
    Df64Config { fma: true, barrier: false },
    Df64Config { fma: true, barrier: true },
    Df64Config { fma: false, barrier: false },
    Df64Config { fma: false, barrier: true },
];

#[derive(Clone, Debug)]
pub struct ConfigResult {
    pub config: Df64Config,
    pub two_sum_exact: bool,
    pub two_prod_exact: bool,
}

#[derive(Clone, Debug)]
pub struct Df64Probe {
    pub results: Vec<ConfigResult>,
    pub samples: usize,
}

impl Df64Probe {
    /// The cheapest configuration in which both the sum and the product are exact; None: do not use df64 here.
    pub fn best(&self) -> Option<Df64Config> {
        self.results.iter().find(|r| r.two_sum_exact && r.two_prod_exact).map(|r| r.config)
    }
    /// Every configuration that is exact (the tests check all of them).
    pub fn exact_configs(&self) -> Vec<Df64Config> {
        self.results.iter().filter(|r| r.two_sum_exact && r.two_prod_exact).map(|r| r.config).collect()
    }
}

/// A deterministic stream of f32 test values: mixed exponents (2^-8 .. 2^8), plus adversarial mantissas (all
/// ones, the Veltkamp boundary, powers of two and their neighbours).
pub fn probe_values(n: usize) -> Vec<f32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let specials = [1.0f32, f32::from_bits(0x3F7F_FFFF), f32::from_bits(0x3F80_0001), 4097.0, 1.0 / 4097.0, 0.1, 3.0, 1.0e-3, 123456.789];
    (0..n)
        .map(|i| {
            if i < specials.len() {
                return specials[i];
            }
            let r = next();
            let mantissa = (r & 0x007F_FFFF) as u32;
            let exp = 127 - 8 + ((r >> 24) % 17) as u32;
            let sign = ((r >> 40) & 1) as u32;
            f32::from_bits((sign << 31) | (exp << 23) | mantissa)
        })
        .collect()
}

/// Runs the self-test on `gpu` in each of the four configurations: TwoSum and TwoProduct must be exact against f64.
#[maybe_async::maybe_async]
pub async fn probe(gpu: &GpuContext) -> Df64Probe {
    let n = 8192;
    let values = probe_values(2 * n);
    let pairs: Vec<[f32; 2]> = values.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    let input = gpu.storage_with("df64_probe_in", &pairs);
    let output = gpu.storage_empty::<[f32; 2]>("df64_probe_out", n);
    let zero = Df64Zero::new(gpu);
    let source = format!("{}\n{}", WGSL, PROBE_WGSL);
    // a + b and a * b are exact in f64 for these inputs (exponents within 24 of each other, 24-bit mantissas).
    let mut results = Vec::with_capacity(CONFIGS.len());
    for &config in CONFIGS.iter() {
        let sum = probe_run(gpu, &source, &input, &output, &zero, n, "probe_two_sum", config).await;
        let prod = probe_run(gpu, &source, &input, &output, &zero, n, "probe_two_prod", config).await;
        results.push(ConfigResult {
            config,
            two_sum_exact: pairs.iter().zip(&sum).all(|(p, r)| p[0] as f64 + p[1] as f64 == r[0] as f64 + r[1] as f64 && r[0] == p[0] + p[1]),
            two_prod_exact: pairs.iter().zip(&prod).all(|(p, r)| p[0] as f64 * p[1] as f64 == r[0] as f64 + r[1] as f64 && r[0] == p[0] * p[1]),
        });
    }
    Df64Probe { results, samples: n }
}

/// One entry point of the probe under one configuration: dispatch, then read the `n` results back.
#[maybe_async::maybe_async]
#[allow(clippy::too_many_arguments)]
async fn probe_run(gpu: &GpuContext, source: &str, input: &wgpu::Buffer, output: &wgpu::Buffer, zero: &Df64Zero, n: usize, entry: &str, cfg: Df64Config) -> Vec<[f32; 2]> {
    let pipeline = gpu.pipeline_with("df64_probe", source, entry, &cfg.constants());
    gpu.dispatch_groups(&pipeline, &[&[input, output], &[], &[], &[&zero.buffer]], (n.div_ceil(64) as u32, 1, 1));
    gpu.read_back_maybe(output, n).await
}

/// Runs one df64 operation (`op_add`, `op_mul`, ... of shaders/df64_ops.wgsl) on `a` and `b` (f64 inputs split
/// into df64) under `config`; returns the results joined back to f64.
pub fn run_op(gpu: &GpuContext, op: &str, a: &[f64], b: &[f64], config: Df64Config) -> Vec<f64> {
    let da: Vec<[f32; 2]> = a.iter().map(|&x| split_f64(x)).collect();
    let db: Vec<[f32; 2]> = b.iter().map(|&x| split_f64(x)).collect();
    let ba = gpu.storage_with("df64_a", &da);
    let bb = gpu.storage_with("df64_b", &db);
    let bc = gpu.storage_empty::<[f32; 2]>("df64_c", a.len());
    let zero = Df64Zero::new(gpu);
    let source = format!("{}\n{}", WGSL, OPS_WGSL);
    let pipeline = gpu.pipeline_with("df64_ops", &source, op, &config.constants());
    gpu.dispatch_groups(&pipeline, &[&[&ba, &bb, &bc], &[], &[], &[&zero.buffer]], (a.len().div_ceil(64) as u32, 1, 1));
    gpu.read_back::<[f32; 2]>(&bc, a.len()).into_iter().map(join_f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_and_join_round_trip_to_48_bits() {
        for &x in &[1.0, std::f64::consts::PI, 1.0e-3 * 7.0, -123456.789012345, 6371.0e3 + 1.0e-4] {
            let back = join_f64(split_f64(x));
            assert!(((back - x) / x).abs() < 2f64.powi(-47), "{x} -> {back}");
        }
        assert_eq!(split_f64(0.5), [0.5, 0.0]);
    }

    #[test]
    fn probe_values_are_deterministic_and_finite() {
        let a = probe_values(1000);
        assert_eq!(a, probe_values(1000));
        assert!(a.iter().all(|v| v.is_finite() && v.abs() > 1e-4 && v.abs() < 1e6));
    }

    #[test]
    fn config_constants_are_the_two_overrides() {
        assert_eq!(CONFIGS[1].constants(), [("DF64_FMA", 1.0), ("DF64_BARRIER", 1.0)]);
        assert_eq!(CONFIGS[2].constants(), [("DF64_FMA", 0.0), ("DF64_BARRIER", 0.0)]);
        assert_eq!(CONFIGS.len(), 4);
    }

    fn lcg_values(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                lo + (hi - lo) * ((s >> 11) as f64 / (1u64 << 53) as f64)
            })
            .collect()
    }

    /// The exact f64 value a df64 input represents (the inputs are split f64s: compare with what they ARE).
    fn as_f64(v: &[f64]) -> Vec<f64> {
        v.iter().map(|&x| join_f64(split_f64(x))).collect()
    }

    fn require_gpu() -> Option<GpuContext> {
        let gpu = GpuContext::new();
        if gpu.is_none() {
            assert!(std::env::var_os("REQUIRE_GPU").is_none(), "REQUIRE_GPU is set and no GPU adapter exists");
            eprintln!("no GPU adapter: df64 test skipped");
        }
        gpu
    }

    #[test]
    fn the_probe_finds_an_exact_configuration_or_says_why() {
        let Some(gpu) = require_gpu() else { return };
        let p = probe(&gpu);
        for r in &p.results {
            eprintln!("df64 probe on {:?} {:?}: {:?}", gpu.adapter_info.name, gpu.adapter_info.backend, r);
        }
        eprintln!("df64 best configuration: {:?}", p.best());
        assert!(p.best().is_some(), "no exact df64 configuration on this device: {p:?}");
        // The barrier is the cure for the folding compilers: with it, the sum must be exact whatever the product.
        assert!(p.results.iter().filter(|r| r.config.barrier).all(|r| r.two_sum_exact), "{p:?}");
    }

    #[test]
    fn operations_reach_about_44_bits_in_every_exact_configuration() {
        let Some(gpu) = require_gpu() else { return };
        let p = probe(&gpu);
        let configs = p.exact_configs();
        assert!(!configs.is_empty(), "no exact df64 configuration on this device: {p:?}");
        let n = 20_000;
        let a = lcg_values(n, 1, 0.5, 8.0);
        let b = lcg_values(n, 2, 0.5, 8.0);
        let (fa, fb) = (as_f64(&a), as_f64(&b));
        for cfg in configs {
            let mut report = vec![];
            let mut check = |name: &str, got: Vec<f64>, want: Vec<f64>, scale: Vec<f64>, bound: f64| {
                let worst = got.iter().zip(&want).zip(&scale).map(|((g, w), s)| (g - w).abs() / s.abs()).fold(0.0, f64::max);
                report.push(format!("{name}: {worst:.2e}"));
                assert!(worst < bound, "{cfg:?} {name}: worst relative error {worst:.3e} >= {bound:e}");
            };
            let scale_ab: Vec<f64> = fa.iter().zip(&fb).map(|(x, y)| x.abs().max(y.abs())).collect();
            check("add", run_op(&gpu, "op_add", &a, &b, cfg), fa.iter().zip(&fb).map(|(x, y)| x + y).collect(), scale_ab.clone(), 2f64.powi(-43));
            check("sub", run_op(&gpu, "op_sub", &a, &b, cfg), fa.iter().zip(&fb).map(|(x, y)| x - y).collect(), scale_ab, 2f64.powi(-43));
            let prod: Vec<f64> = fa.iter().zip(&fb).map(|(x, y)| x * y).collect();
            check("mul", run_op(&gpu, "op_mul", &a, &b, cfg), prod.clone(), prod, 2f64.powi(-42));
            let quot: Vec<f64> = fa.iter().zip(&fb).map(|(x, y)| x / y).collect();
            check("div", run_op(&gpu, "op_div", &a, &b, cfg), quot.clone(), quot, 2f64.powi(-41));
            let root: Vec<f64> = fa.iter().map(|x| x.sqrt()).collect();
            check("sqrt", run_op(&gpu, "op_sqrt", &a, &b, cfg), root.clone(), root, 2f64.powi(-41));
            // mul_f32 multiplies by the high word of b only
            let bx: Vec<f64> = b.iter().map(|&x| split_f64(x)[0] as f64).collect();
            let pf: Vec<f64> = fa.iter().zip(&bx).map(|(x, y)| x * y).collect();
            check("mul_f32", run_op(&gpu, "op_mul_f32", &a, &b, cfg), pf.clone(), pf, 2f64.powi(-42));
            eprintln!("df64 {cfg:?} on {:?}: worst relative errors {}", gpu.adapter_info.name, report.join(", "));
        }
    }

    #[test]
    fn the_barrier_is_what_makes_the_naive_transforms_exact_on_this_device() {
        // Documents the measured trap: where the compiler folds, no barrier means a wrong sum. Not an assertion
        // on every device (a compiler that keeps the order needs no barrier): it asserts the invariant that
        // matters, that the BARRIER configurations are exact whenever the plain ones are not.
        let Some(gpu) = require_gpu() else { return };
        let p = probe(&gpu);
        let plain = p.results.iter().find(|r| r.config == CONFIGS[0]).unwrap();
        let barrier = p.results.iter().find(|r| r.config == CONFIGS[1]).unwrap();
        eprintln!("plain fma sum exact: {}, barrier fma sum exact: {}", plain.two_sum_exact, barrier.two_sum_exact);
        if !plain.two_sum_exact {
            assert!(barrier.two_sum_exact, "the barrier must restore the exact sum: {p:?}");
        }
    }

    #[test]
    fn floor_and_fract_are_exact_including_integer_high_words() {
        let Some(gpu) = require_gpu() else { return };
        let Some(cfg) = probe(&gpu).best() else { return };
        // 3 + 1e-9 (hi = 3 exactly, lo > 0), 3 - 1e-9 (hi = 3, lo < 0: floor must be 2), -0.5, 12345.678, 1e5 + 0.25
        let a = vec![3.0 + 1e-9, 3.0 - 1e-9, -0.5, 12345.678, 1e5 + 0.25, 7.0, -7.0 - 1e-9, 0.999999999999, 1.0e6 - 1e-7];
        let b = vec![0.0; a.len()];
        let fa = as_f64(&a);
        let fl = run_op(&gpu, "op_floor", &a, &b, cfg);
        let fr = run_op(&gpu, "op_fract", &a, &b, cfg);
        for (i, x) in fa.iter().enumerate() {
            assert_eq!(fl[i], x.floor(), "floor({x:e})");
            let want = x - x.floor();
            assert!((fr[i] - want).abs() < 1e-14 * x.abs().max(1.0), "fract({x:e}) = {:e}, want {want:e}", fr[i]);
            assert!((0.0..1.0 + 1e-9).contains(&fr[i]));
        }
    }
}
