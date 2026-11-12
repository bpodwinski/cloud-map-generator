//! Golden cases run through the GPU kernels of `stages` and `texture`:
//!
//! * group `texture_surface`: `cloudgen.texture.regime_weights` (with its intermediates), `_standardize` and
//!   `regime_texture` (with its intermediates);
//! * group `products_generate`: `cloudgen.generate.moisture_profile` and `cloudgen.generate.along`;
//! * group `sphere_noise`: `cloudgen.sphere.upsample_dirs` (the slow banded 4096 x 2048 case runs with
//!   GOLDEN_SLOW=1) and `cloudgen.sphere.grid_dirs`.
//!
//! The kernels are f32 (WGSL has no f64) and take float64 inputs rounded to f32. The cases whose frozen oracle
//! is float64 (rtol 1e-9) cannot pass at that tolerance: each is listed in `EXPECTED_FAILURES` with the
//! measured error, and `float64_oracle_cases_hold_at_the_float32_tier` holds them to the tolerance of the same
//! function's float32 case. The measured tolerance use of every case is printed (`tolerance use`), and a
//! widening is declared only where the GPU needs it, with its measurement as the reason. Without a GPU adapter
//! the tests print why and skip, unless the environment variable REQUIRE_GPU is set, in which case they fail.

use cloud_map_generator_core::noise::{octave_amp_freq, octave_ladder, CellEdgesPlan, FbmOctave, FbmPlan, Perlin3, Worley3, DEFAULT_LACUNARITY, DEFAULT_TOP};
use cloud_map_generator_core::texture::{cell_seed, used_cells, RegimeTexturePlans, CELL_DIAMETERS_KM, CELL_WEIGHTS, GENERIC_SEED_OFFSET, STREET_SEED_OFFSET};
use cloud_map_generator_golden::{compare_array, load_group, run_widened, Array, Case, Config, Out, Tol, Widen};
use cloud_map_generator_gpu::stages::StageKernels;
use cloud_map_generator_gpu::texture::TextureKernels;
use cloud_map_generator_gpu::GpuContext;
use std::cell::RefCell;
use std::collections::BTreeMap;

/// Known failures. All but one are of the same kind: the frozen oracle is float64 (rtol 1e-9, atol 1e-12) and
/// the GPU is float32 (its float64 inputs rounded to float32 too); each reason quotes the largest |GPU - frozen|
/// of the run on the RTX 5080 (Vulkan, and the same on DX12), and `float64_oracle_cases_hold_at_the_float32_tier`
/// binds the GPU result on each at the tolerance the same function's float32 case freezes. The exception is
/// `standardize.constant_non_dyadic.f32_n7`, a numpy float32 artifact: `numpy_float32_artifact_is_an_exact_zero_here`.
const EXPECTED_FAILURES: &[(&str, &str)] = &[
    ("grid_dirs.f64.16x8", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 7.4e-8 (the float64 direction rounded to f32 once is 6e-8 off)"),
    ("grid_dirs.f64.64x32", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 5.5e-8"),
    ("along.points.f64", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.3e-7 (float64 inputs rounded to f32 too)"),
    ("moisture_profile.f64.hadley10", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 7.7e-8 (float64 latitudes rounded to f32 too)"),
    ("moisture_profile.f64.hadley30", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.2e-7 (float64 latitudes rounded to f32 too)"),
    ("moisture_profile.f64.hadley90", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 7.9e-8 (float64 latitudes rounded to f32 too)"),
    ("regime_weights.hand_picked.earth", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 6.6e-7 over the weights and the intermediates"),
    ("regime_weights.scalar.closed_deck", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.7e-8"),
    ("regime_weights.scalar.total_over_one", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 6.6e-7"),
    ("regime_weights.sweep.earth", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.5e-6 (the cold-air ramp turns the 3e-6 degree rounding of the f32 latitudes into weight)"),
    ("regime_weights.sweep.fast_rotator", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.5e-6"),
    ("regime_weights.sweep.mid", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.6e-6"),
    ("regime_weights.sweep.slow_rotator", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 1.4e-6"),
    ("regime_weights.total_near_one.threshold", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 6.6e-7, on a case whose raw totals sit within 1e-6 to 1e-2 of the normalization threshold 1"),
    ("standardize.f64.shifted", "float64 oracle (rtol 1e-9): the f32 GPU differs by at most 2.7e-7 (a float64 field rounded to f32)"),
    (
        "standardize.constant_non_dyadic.f32_n7",
        "numpy float32 artifact: 7 values of 0.1f have a numpy float32 mean one ulp above 0.1f, so numpy's std is 7.5e-9 > 0 and its result -1; the GPU takes the statistics in f64 from the f32 partial sums, the mean of a constant field is exact, the std 0, the divisor 1 and the result 0",
    ),
];

/// The expected failures that are not float64 oracles (held to the float32 tier of their function): the numpy
/// float32 artifact, checked by `numpy_float32_artifact_is_an_exact_zero_here`.
const NOT_A_FLOAT64_ORACLE: &[&str] = &["standardize.constant_non_dyadic.f32_n7"];

/// Widenings of the frozen tolerances, per function prefix: none needed so far.
const WIDEN: &[Widen] = &[];

fn gpu_or_skip(test: &str) -> Option<GpuContext> {
    match GpuContext::new() {
        Some(g) => {
            println!("{test}: adapter {} ({:?}), max storage binding {} bytes", g.adapter_info.name, g.adapter_info.backend, g.limits.max_storage_buffer_binding_size);
            Some(g)
        }
        None if std::env::var_os("REQUIRE_GPU").is_some() => panic!("{test}: no GPU adapter and REQUIRE_GPU is set"),
        None => {
            println!("{test}: no GPU adapter (wgpu found none): skipped; set REQUIRE_GPU to fail instead");
            None
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// Reading cases

/// An array of any float dtype as f32: what the GPU can hold.
fn f32s(a: &Array) -> Vec<f32> {
    assert!(a.dtype.is_float(), "expected a float array, got {}", a.dtype.name());
    a.to_f64().into_iter().map(|v| v as f32).collect()
}

fn dirs_f32(a: &Array) -> Vec<[f32; 3]> {
    f32s(a).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn flat(v: &[[f32; 3]]) -> Vec<f32> {
    v.iter().flatten().copied().collect()
}

/// Shape of an array without its trailing axis of 3.
fn lead(a: &Array) -> Vec<usize> {
    a.shape[..a.shape.len() - 1].to_vec()
}

fn mat3_f32(v: &[f32]) -> [[f32; 3]; 3] {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

// ---------------------------------------------------------------------------------------------------------
// The plans of a regime_texture case (the recorded draws), as in cloud-map-generator-core/tests/golden_texture.rs

/// The recorded scalars of plan `i` (a JSON object of `params.plans`, found by its `index`); a macro so that this
/// file does not need to name serde_json's `Value`.
macro_rules! plan_record {
    ($c:expr, $i:expr) => {{
        let (c, i) = ($c, $i);
        c.param("plans")
            .as_array()
            .expect("plans is a list")
            .iter()
            .find(|r| r["index"].as_u64() == Some(i as u64))
            .unwrap_or_else(|| panic!("case {}: no recorded plan {i}", c.id))
    }};
}

macro_rules! rec_f64 {
    ($rec:expr, $key:expr) => {{
        let (rec, key) = ($rec, $key);
        rec[key].as_f64().unwrap_or_else(|| panic!("recorded plan has no number {key}: {rec}"))
    }};
}

fn fbm_plan_of(c: &Case, perm_idx: usize, fbm_idx: usize, seed: u64, frequency: f64, octaves: usize, gain: f64) -> FbmPlan {
    assert_eq!(plan_record!(c, perm_idx)["kind"], "perlin3");
    let rec = plan_record!(c, fbm_idx);
    assert_eq!(rec["kind"], "fbm");
    assert_eq!(rec["seed"].as_u64(), Some(seed), "case {}: the fbm seed", c.id);
    assert_eq!(rec_f64!(rec, "frequency"), frequency, "case {}: the ladder's start frequency", c.id);
    assert_eq!(rec["octaves"].as_u64(), Some(octaves as u64), "case {}: the ladder's octave count", c.id);
    assert_eq!(rec_f64!(rec, "gain"), gain);
    assert_eq!(rec["billow"].as_bool(), Some(false));
    let key = |i: usize, name: &str| format!("plan{i}.{name}");
    let amp = c.input(&key(fbm_idx, "amp")).as_f64();
    let freq = c.input(&key(fbm_idx, "freq")).as_f64();
    let rot = c.input(&key(fbm_idx, "rot")).as_f32();
    let off = c.input(&key(fbm_idx, "off")).as_f32();
    assert_eq!(
        octave_amp_freq(frequency, octaves, gain, DEFAULT_LACUNARITY),
        amp.iter().copied().zip(freq.iter().copied()).collect::<Vec<_>>(),
        "case {}: the amplitude/frequency schedule is not the recorded one",
        c.id
    );
    let octs = (0..octaves)
        .map(|k| FbmOctave { amp: amp[k], freq: freq[k], rot: mat3_f32(&rot[9 * k..9 * k + 9]), off: [off[3 * k], off[3 * k + 1], off[3 * k + 2]] })
        .collect();
    let plan = FbmPlan::new(Perlin3::new(&c.input(&key(perm_idx, "perm")).as_i32()), octs, false);
    let norm = rec_f64!(rec, "norm");
    assert!((plan.norm - norm).abs() <= 1e-13 * norm.abs(), "case {}: norm {} vs recorded {norm}", c.id, plan.norm);
    plan
}

fn cell_plan_of(c: &Case, worley_idx: usize, edges_idx: usize, seed: u64, frequency: f64) -> CellEdgesPlan {
    assert_eq!(plan_record!(c, worley_idx)["kind"], "worley3");
    let rec = plan_record!(c, edges_idx);
    assert_eq!(rec["kind"], "cell_edges");
    assert_eq!(rec["seed"].as_u64(), Some(seed), "case {}: the cell seed", c.id);
    assert_eq!(rec_f64!(rec, "frequency"), frequency, "case {}: cells per radian", c.id);
    let perm = c.input(&format!("plan{worley_idx}.perm")).as_i64();
    let jitter: Vec<[f32; 3]> = c.input(&format!("plan{worley_idx}.jitter")).as_f32().chunks_exact(3).map(|v| [v[0], v[1], v[2]]).collect();
    CellEdgesPlan { worley: Worley3::new(&perm, &jitter), rot: mat3_f32(&c.input(&format!("plan{edges_idx}.rot")).as_f32()), frequency }
}

fn plans_of(c: &Case) -> RegimeTexturePlans {
    let (radius, fmax, seed) = (c.p_f64("radius_km"), c.p_f64("fmax"), c.p_i64("seed") as u64);
    assert_eq!(c.p_f64s("cell_diameters_km"), CELL_DIAMETERS_KM);
    assert_eq!(c.p_f64s("cell_weights"), CELL_WEIGHTS);
    let (f_fine, n_fine) = octave_ladder(c.p_f64("fine_frequency"), fmax, DEFAULT_LACUNARITY, DEFAULT_TOP);
    let gain = c.p_f64("fine_gain");
    let generic = fbm_plan_of(c, 0, 1, seed + GENERIC_SEED_OFFSET, f_fine, n_fine, gain);
    let street = fbm_plan_of(c, 2, 3, seed + STREET_SEED_OFFSET, f_fine, n_fine, gain);
    let used = used_cells(radius, fmax);
    assert_eq!(used.iter().map(|u| u.diameter_km).collect::<Vec<_>>(), c.p_f64s("cells_used_km"), "case {}: the cutoff keeps other diameters", c.id);
    let cells = used.iter().enumerate().map(|(k, u)| cell_plan_of(c, 4 + 2 * k, 5 + 2 * k, cell_seed(seed, u.diameter_km), u.frequency)).collect();
    RegimeTexturePlans { generic, street, cells }
}

// ---------------------------------------------------------------------------------------------------------
// The cases on the GPU

struct Gpu<'a> {
    kernels: TextureKernels<'a>,
}

impl<'a> Gpu<'a> {
    fn new(ctx: &'a GpuContext) -> Gpu<'a> {
        Gpu { kernels: TextureKernels::new(ctx) }
    }

    fn stages(&self) -> &StageKernels<'a> {
        self.kernels.stages()
    }

    fn regime_weights(&self, c: &Case) -> Out {
        let (lat, back) = (c.input("lat"), c.input("lat_back"));
        let ([closed, open, street], [convective, cold, total]) = self.kernels.regime_weight_terms_to_vec(&f32s(lat), &f32s(back), c.p_f64("hadley_deg"));
        let s = &lat.shape;
        Out::new()
            .f32("closed", s, &closed)
            .f32("open", s, &open)
            .f32("street", s, &street)
            .f32("convective", s, &convective)
            .f32("cold", s, &cold)
            .f32("total_raw", s, &total)
    }

    fn standardize(&self, c: &Case) -> Out {
        let v = c.input("v");
        Out::new().f32("y", &v.shape, &self.stages().standardize_to_vec(&f32s(v)))
    }

    fn regime_texture(&self, c: &Case) -> Out {
        let pts = c.input("p");
        let (p, back) = (dirs_f32(pts), dirs_f32(c.input("back")));
        let weights = [f32s(c.input("weights.closed")), f32s(c.input("weights.open")), f32s(c.input("weights.street"))];
        let t = self.kernels.regime_texture_terms_to_vec(&p, &back, [&weights[0], &weights[1], &weights[2]], c.p_f64("radius_km"), c.p_f64("fmax"), &plans_of(c));
        let shape = lead(pts);
        let mut out = Out::new().f32("texture", &shape, &t.texture).f32("generic_std", &shape, &t.generic_std).f32("street_std", &shape, &t.street_std);
        if let Some(cf) = &t.cells {
            out = out
                .f32("closed_sum", &shape, &cf.closed_sum)
                .f32("open_sum", &shape, &cf.open_sum)
                .f32("closed_std", &shape, &cf.closed_std)
                .f32("open_std", &shape, &cf.open_std);
        }
        out
    }

    fn moisture_profile(&self, c: &Case) -> Out {
        let lat = c.input("lat");
        Out::new().f32("profile", &lat.shape, &self.stages().moisture_profile_to_vec(&f32s(lat), c.p_f64("hadley_deg")))
    }

    fn along(&self, c: &Case) -> Out {
        let (p, back) = (c.input("p"), c.input("back"));
        let (pd, bd) = (dirs_f32(p), dirs_f32(back));
        let mut out = Out::new().f32("q", &p.shape, &flat(&self.stages().along_to_vec(&pd, &bd, c.p_f64("f"))));
        if c.has_param("f_neg") {
            out = out.f32("q_neg", &p.shape, &flat(&self.stages().along_to_vec(&pd, &bd, c.p_f64("f_neg"))));
        }
        out
    }

    fn grid_dirs(&self, c: &Case) -> Out {
        let (w, h) = (c.p_usize("width"), c.p_usize("height"));
        Out::new().f32("dirs", &[h, w, 3], &flat(&self.stages().grid_dirs_to_vec(w, h, 0, h)))
    }

    /// `upsample_dirs`; the banded case (`params.band_rows`) stores 8 rows and the sums of every row instead of
    /// the 100 MB map, so the target is made band by band, as the function is.
    fn upsample_dirs(&self, c: &Case) -> Out {
        let dirs = c.input("dirs");
        let (sh, sw) = (dirs.shape[0], dirs.shape[1]);
        let (w, h) = (c.p_usize("width"), c.p_usize("height"));
        let k = self.stages();
        let coarse = k.upload_dirs(&dirs_f32(dirs));
        if !c.has_param("band_rows") {
            let buf = k.upsample_dirs_to_buffer(&coarse, sw, sh, w, h, 0, h);
            return Out::new().f32("out", &[h, w, 3], &flat(&k.dirs_to_vec(&buf, w * h)));
        }
        let band = c.p_usize("band_rows");
        let want_rows: Vec<usize> = c.param("rows").as_array().expect("rows").iter().map(|v| v.as_u64().expect("row") as usize).collect();
        let mut rows = vec![0.0f32; want_rows.len() * w * 3];
        let mut row_sum = vec![0.0f64; h * 3];
        for row0 in (0..h).step_by(band) {
            let n_rows = band.min(h - row0);
            let buf = k.upsample_dirs_to_buffer(&coarse, sw, sh, w, h, row0, n_rows);
            let dirs = k.dirs_to_vec(&buf, w * n_rows);
            for r in 0..n_rows {
                for d in &dirs[r * w..(r + 1) * w] {
                    for (a, v) in row_sum[(row0 + r) * 3..(row0 + r) * 3 + 3].iter_mut().zip(d) {
                        *a += f64::from(*v);
                    }
                }
                if let Some(i) = want_rows.iter().position(|&x| x == row0 + r) {
                    rows[i * w * 3..(i + 1) * w * 3].copy_from_slice(&flat(&dirs[r * w..(r + 1) * w]));
                }
            }
        }
        Out::new().f32("rows", &[want_rows.len(), w, 3], &rows).f64("row_sum", &[h, 3], &row_sum)
    }

    fn port(&self, c: &Case) -> Option<Out> {
        Some(match c.function.as_str() {
            "cloudgen.texture.regime_weights" => self.regime_weights(c),
            "cloudgen.texture._standardize" => self.standardize(c),
            "cloudgen.texture.regime_texture" => self.regime_texture(c),
            "cloudgen.generate.moisture_profile" => self.moisture_profile(c),
            "cloudgen.generate.along" => self.along(c),
            "cloudgen.sphere.grid_dirs" => self.grid_dirs(c),
            "cloudgen.sphere.upsample_dirs" => self.upsample_dirs(c),
            _ => return None,
        })
    }
}

/// `(use, max |diff|)`: the share of its frozen tolerance the GPU result uses, the max over the elements of
/// `|got - want| / (atol + rtol |want|)`, and the largest absolute difference. A use above 1 means the case fails
/// at its own tolerance; it is the factor a widening would need.
fn tolerance_use(c: &Case, out: &Out) -> (f64, f64) {
    let (mut worst, mut max_abs) = (0.0f64, 0.0f64);
    for (name, (want, tol)) in &c.expected {
        let (Some(got), Tol::Close { rtol, atol, .. }) = (out.arrays.get(name), tol) else { continue };
        if want.dtype.is_float() && got.shape == want.shape {
            let (g, w) = (got.to_f64(), want.to_f64());
            for i in 0..g.len() {
                let e = (g[i] - w[i]).abs();
                if e.is_finite() {
                    worst = worst.max(e / (atol + rtol * w[i].abs()));
                    max_abs = max_abs.max(e);
                }
            }
        }
    }
    (worst, max_abs)
}

/// The groups and function prefixes of this file.
const GROUPS: &[(&str, &[&str])] = &[
    ("texture_surface", &["cloudgen.texture.regime_weights", "cloudgen.texture._standardize", "cloudgen.texture.regime_texture"]),
    ("products_generate", &["cloudgen.generate.moisture_profile", "cloudgen.generate.along"]),
    ("sphere_noise", &["cloudgen.sphere.upsample_dirs", "cloudgen.sphere.grid_dirs"]),
];

/// Runs the cases of one group through `gpu` and prints the tolerance use of every case.
fn run_group(gpu: &GpuContext, group: &str) -> cloud_map_generator_golden::Report {
    let (_, prefixes) = GROUPS.iter().find(|(g, _)| *g == group).expect("a group of this file");
    let g = Gpu::new(gpu);
    let cfg = Config { group: GROUPS.iter().find(|(n, _)| *n == group).unwrap().0, prefixes, expected_failures: EXPECTED_FAILURES };
    let uses: RefCell<BTreeMap<String, (f64, f64)>> = RefCell::new(BTreeMap::new());
    let report = run_widened(&cfg, WIDEN, |c| {
        let out = g.port(c)?;
        uses.borrow_mut().insert(c.id.clone(), tolerance_use(c, &out));
        Some(out)
    });
    // One line per case: the share of its tolerance the GPU uses (1.00 = at the limit).
    for (id, (u, abs)) in &uses.into_inner() {
        println!("  tolerance use {u:10.3}  max |diff| {abs:10.3e}  {id}");
    }
    report
}

#[test]
fn texture_surface_cases_on_the_gpu() {
    let Some(gpu) = gpu_or_skip("texture_surface_cases_on_the_gpu") else { return };
    run_group(&gpu, "texture_surface").assert_clean();
}

#[test]
fn products_generate_cases_on_the_gpu() {
    let Some(gpu) = gpu_or_skip("products_generate_cases_on_the_gpu") else { return };
    run_group(&gpu, "products_generate").assert_clean();
}

#[test]
fn sphere_cases_on_the_gpu() {
    let Some(gpu) = gpu_or_skip("sphere_cases_on_the_gpu") else { return };
    run_group(&gpu, "sphere_noise").assert_clean();
}

/// An adapter of one native backend (the default context picks one for the machine).
fn context_for(backends: wgpu::Backends) -> Option<GpuContext> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends, ..Default::default() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok()?;
    let (adapter_info, limits) = (adapter.get_info(), adapter.limits());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cloudgen"),
        required_features: wgpu::Features::empty(),
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .ok()?;
    Some(GpuContext { device, queue, adapter_info, limits })
}

/// The same cases on every native backend the machine has (Vulkan, DX12, Metal): the WGSL goes through a
/// different translator on each (SPIR-V, HLSL, MSL), so a backend-specific miscompile (a `%` of a negative
/// integer, a fused multiply-add) would show here. Backends without an adapter are reported and skipped.
#[test]
fn cases_on_every_native_backend() {
    let mut ran = 0;
    for (name, backends) in [("Vulkan", wgpu::Backends::VULKAN), ("DX12", wgpu::Backends::DX12), ("Metal", wgpu::Backends::METAL)] {
        match context_for(backends) {
            Some(gpu) => {
                println!("backend {name}: adapter {} ({:?}, {:?})", gpu.adapter_info.name, gpu.adapter_info.backend, gpu.adapter_info.device_type);
                for (group, _) in GROUPS {
                    run_group(&gpu, group).assert_clean();
                }
                ran += 1;
            }
            None => println!("backend {name}: no adapter, skipped"),
        }
    }
    if ran == 0 && std::env::var_os("REQUIRE_GPU").is_some() {
        panic!("no native backend has an adapter and REQUIRE_GPU is set");
    }
}

/// The tolerance a function's float32 cases carry in the frozen data: (function, sibling float32 case, group,
/// output).
const F32_TIER: &[(&str, &str, &str, &str)] = &[
    ("cloudgen.texture.regime_weights", "regime_weights.mixed_dtype.float32_back", "texture_surface", "*"),
    ("cloudgen.texture._standardize", "standardize.f32.unit_interval", "texture_surface", "y"),
    ("cloudgen.generate.moisture_profile", "moisture_profile.f32.hadley30", "products_generate", "profile"),
    ("cloudgen.generate.along", "along.grid_f32.f0p6", "products_generate", "q"),
    ("cloudgen.sphere.grid_dirs", "grid_dirs.16x8", "sphere_noise", "dirs"),
];

/// The float64-oracle cases, on the GPU, against the same frozen outputs at the tolerance the oracle itself
/// froze for float32 arithmetic of that function.
#[test]
fn float64_oracle_cases_hold_at_the_float32_tier() {
    let Some(ctx) = gpu_or_skip("float64_oracle_cases_hold_at_the_float32_tier") else { return };
    let gpu = Gpu::new(&ctx);
    for (id, _) in EXPECTED_FAILURES.iter().filter(|(id, _)| !NOT_A_FLOAT64_ORACLE.contains(id)) {
        let c = GROUPS
            .iter()
            .find_map(|(g, _)| load_group(g).into_iter().find(|c| c.id == *id))
            .unwrap_or_else(|| panic!("no golden case {id}"));
        let (_, sibling, tier_group, output) = F32_TIER.iter().find(|(f, ..)| *f == c.function).unwrap_or_else(|| panic!("case {id}: no float32 tier known for {}", c.function));
        let cases = load_group(tier_group);
        let sib = cases.iter().find(|s| s.id == *sibling).unwrap_or_else(|| panic!("no golden case {sibling}"));
        let out = gpu.port(&c).expect("the case is ported");
        let mut checked = 0;
        for (name, (want, frozen)) in &c.expected {
            let tier = if *output == "*" { sib.expected.get(name).map(|(_, t)| t.clone()) } else { sib.expected.get(*output).map(|(_, t)| t.clone()) };
            let Some(tier) = tier else { continue };
            assert!(matches!(tier, Tol::Close { .. }), "the float32 tier of {} is a close tolerance", c.function);
            if let (Tol::Close { rtol: own, .. }, Tol::Close { rtol: loose, .. }) = (frozen, &tier) {
                assert!(own < loose, "case {id}: the oracle's own tolerance is tighter than its float32 tier");
            }
            let problems = compare_array(name, &out.arrays[name], want, &tier);
            assert!(problems.is_empty(), "case {id} at the float32 tier {tier:?}: {problems:?}");
            checked += 1;
        }
        assert!(checked > 0, "case {id}: nothing was checked");
    }
}

/// `standardize.constant_non_dyadic.f32_n7` freezes a numpy float32 artifact (7 values of 0.1f standardize to -1
/// because numpy's float32 mean is one ulp off): the GPU's mean of a constant field is exact, so the std is 0,
/// the divisor 1 and every value `x - mean = 0`, whatever the length (7, 8, 64 and a tile boundary).
#[test]
fn numpy_float32_artifact_is_an_exact_zero_here() {
    let Some(ctx) = gpu_or_skip("numpy_float32_artifact_is_an_exact_zero_here") else { return };
    let gpu = Gpu::new(&ctx);
    let case = load_group("texture_surface").into_iter().find(|c| c.id == "standardize.constant_non_dyadic.f32_n7").expect("the case");
    let got = gpu.port(&case).expect("ported").arrays["y"].as_f32();
    assert_eq!(got, vec![0.0f32; 7]);
    for n in [1usize, 7, 8, 16, 64, 100, 4096, 4097, 10_000] {
        let y = gpu.stages().standardize_to_vec(&vec![0.1f32; n]);
        assert!(y.iter().all(|&v| v == 0.0), "n = {n}: a constant field standardizes to exactly 0");
    }
}
