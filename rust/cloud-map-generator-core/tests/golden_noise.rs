//! Golden cases of cloudgen/noise.py (group sphere_noise, functions cloudgen.noise.* and the property case of
//! `_random_rotation`).

use cloud_map_generator_core::noise::*;
use cloud_map_generator_core::rng::Rng;
use cloud_map_generator_core::sphere::Vec3;
use cloud_map_generator_golden::{load_group, run, Array, Case, Config, DType, Out};

/// The six `fbm.stat.*` cases freeze Python's per-seed mean and std over numpy's draws: the output cannot match
/// a port that draws with its own generator. They are judged by `fbm_stat_cases_stay_in_their_bands`
/// (GOLDEN_SLOW=1), which draws the plans with the port's own Rng, as the case docs prescribe.
const EXPECTED_FAILURES: &[(&str, &str)] = &[
    ("fbm.stat.1oct.billow", "rng+stat: Python's 16 per-seed means and stds over numpy's bit stream; band test fbm_stat_cases_stay_in_their_bands"),
    ("fbm.stat.1oct.plain", "rng+stat: Python's 16 per-seed means and stds over numpy's bit stream; band test fbm_stat_cases_stay_in_their_bands"),
    ("fbm.stat.3oct.billow", "rng+stat: Python's 16 per-seed means and stds over numpy's bit stream; band test fbm_stat_cases_stay_in_their_bands"),
    ("fbm.stat.3oct.plain", "rng+stat: Python's 16 per-seed means and stds over numpy's bit stream; band test fbm_stat_cases_stay_in_their_bands"),
    ("fbm.stat.6oct.billow", "rng+stat: Python's 16 per-seed means and stds over numpy's bit stream; band test fbm_stat_cases_stay_in_their_bands"),
    ("fbm.stat.6oct.plain", "rng+stat: Python's 16 per-seed means and stds over numpy's bit stream; band test fbm_stat_cases_stay_in_their_bands"),
];

fn vec3_f32(a: &Array) -> Vec<Vec3<f32>> {
    a.as_f32().chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}
fn vec3_f64(a: &Array) -> Vec<Vec3<f64>> {
    a.to_f64().chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}
/// Points of any float dtype as float32, which is what `fbm` and `cell_edges` do first (`astype(np.float32)`).
fn points_as_f32(a: &Array) -> Vec<Vec3<f32>> {
    a.to_f64().chunks_exact(3).map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect()
}
/// Shape of an array without its trailing axis of 3.
fn lead(a: &Array) -> Vec<usize> {
    a.shape[..a.shape.len() - 1].to_vec()
}
fn mat3_f32(v: &[f32]) -> [[f32; 3]; 3] {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}
fn mat3_f64(v: &[f64]) -> [[f64; 3]; 3] {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}
fn flat9(m: &[[f64; 3]; 3]) -> [f64; 9] {
    [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]]
}

fn perlin_of(c: &Case) -> Perlin3 {
    let perm = c.input("plan0.perm").as_i32();
    Perlin3::new(&perm)
}

fn worley_of(c: &Case) -> Worley3 {
    let perm = c.input("plan0.perm").as_i64();
    let jitter = vec3_f32(c.input("plan0.jitter"));
    Worley3::new(&perm, &jitter)
}

/// The fBm plan of an fbm case: the recorded tables (plan0 = the Perlin permutation, plan1 = the octaves). The
/// derived quantities are recomputed by the port and checked against the recorded ones, so a wrong schedule
/// or normalization fails the case instead of being papered over by the recording.
fn fbm_plan_of(c: &Case) -> FbmPlan {
    let amp = c.input("plan1.amp").as_f64();
    let freq = c.input("plan1.freq").as_f64();
    let rot = c.input("plan1.rot").as_f32();
    let off = c.input("plan1.off").as_f32();
    let n = amp.len();
    let (frequency, gain, lacunarity) = (c.p_f64("frequency"), c.p_f64("gain"), c.p_f64("lacunarity"));
    assert_eq!(n, c.p_usize("octaves"));
    assert_eq!(
        octave_amp_freq(frequency, n, gain, lacunarity),
        amp.iter().copied().zip(freq.iter().copied()).collect::<Vec<_>>(),
        "the amplitude/frequency schedule is not the recorded one"
    );
    let octaves = (0..n)
        .map(|k| FbmOctave { amp: amp[k], freq: freq[k], rot: mat3_f32(&rot[9 * k..9 * k + 9]), off: [off[3 * k], off[3 * k + 1], off[3 * k + 2]] })
        .collect();
    let plan = FbmPlan::new(perlin_of(c), octaves, c.p_bool("billow"));
    let recorded = c.param("plans")[1]["norm"].as_f64().expect("recorded norm");
    assert!((plan.norm - recorded).abs() <= 1e-13 * recorded.abs(), "norm {} vs recorded {recorded}", plan.norm);
    plan
}

fn cell_edges_plan_of(c: &Case) -> CellEdgesPlan {
    CellEdgesPlan {
        worley: worley_of(c),
        rot: mat3_f32(&c.input("plan1.rot").as_f32()),
        frequency: c.p_f64("frequency"),
    }
}

/// Dispatches Perlin3 on the dtype of the coordinates (numpy runs it in that dtype).
fn perlin_call(c: &Case) -> Out {
    let (x, y, z) = (c.input("x"), c.input("y"), c.input("z"));
    let p = perlin_of(c);
    if x.dtype == DType::F32 {
        let v = p.call(&x.as_f32(), &y.as_f32(), &z.as_f32());
        Out::new().f32("value", &x.shape, &v)
    } else {
        let v = p.call(&x.as_f64(), &y.as_f64(), &z.as_f64());
        Out::new().f64("value", &x.shape, &v)
    }
}

fn port(c: &Case) -> Option<Out> {
    Some(match c.function.as_str() {
        "cloudgen.noise._fade" => {
            let t = c.input("t");
            if t.dtype == DType::F32 {
                let o: Vec<f32> = t.as_f32().into_iter().map(fade).collect();
                Out::new().f32("out", &t.shape, &o)
            } else {
                let o: Vec<f64> = t.as_f64().into_iter().map(fade).collect();
                Out::new().f64("out", &t.shape, &o)
            }
        }
        "cloudgen.noise._grad" => {
            let (h, x, y, z) = (c.input("h").as_i32(), c.input("x").as_f32(), c.input("y").as_f32(), c.input("z").as_f32());
            let o: Vec<f32> = (0..h.len()).map(|i| grad(h[i], x[i], y[i], z[i])).collect();
            Out::new().f32("out", &c.input("h").shape, &o)
        }
        "cloudgen.noise.Perlin3.__init__" => {
            let p = perlin_of(c);
            Out::new().i32("table", &[p.perm.len()], &p.perm)
        }
        "cloudgen.noise.Perlin3.__call__" => perlin_call(c),
        "cloudgen.noise.Worley3.__init__" => {
            let w = worley_of(c);
            assert_eq!(w.perm.len(), 2 * Worley3::SIZE);
            Out::new().i64("perm", &[w.perm.len()], &w.perm)
        }
        "cloudgen.noise.Worley3.__call__" => {
            let pts = c.input("pts");
            let w = worley_of(c);
            let (f1, f2) = if pts.dtype == DType::F32 { w.call(&vec3_f32(pts)) } else { w.call(&vec3_f64(pts)) };
            Out::new().f32("f1", &lead(pts), &f1).f32("f2", &lead(pts), &f2)
        }
        "cloudgen.noise._random_rotation" => {
            let g = c.input("g");
            let n = g.shape[0];
            let gv = g.as_f64();
            let q: Vec<f64> = (0..n).flat_map(|i| flat9(&rotation_from_normals(&mat3_f64(&gv[9 * i..9 * i + 9])))).collect();
            Out::new().f64("q", &g.shape, &q)
        }
        // A property of drawn rotations: the port draws its own (the recorded ones are numpy's) and checks them
        // against the same two outputs, computed in float64 from the float32 matrices as the case does.
        "property.cloudgen.noise._random_rotation" => {
            let n = c.input("rot").shape[0];
            let mut rng = Rng::new(101);
            let mut gram = Vec::with_capacity(9 * n);
            let mut det = Vec::with_capacity(n);
            for _ in 0..n {
                let q = rotation_to_f32(&random_rotation(&mut rng));
                let m: [[f64; 3]; 3] = q.map(|r| r.map(f64::from));
                for i in 0..3 {
                    for j in 0..3 {
                        gram.push((0..3).map(|k| m[i][k] * m[j][k]).sum::<f64>());
                    }
                }
                det.push(
                    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]),
                );
            }
            Out::new().f64("gram", &[n, 3, 3], &gram).f64("det", &[n], &det)
        }
        "cloudgen.noise.cell_edges" => {
            let pts = c.input("points");
            let v = cell_edges(&points_as_f32(pts), &cell_edges_plan_of(c));
            Out::new().f32("value", &lead(pts), &v)
        }
        "cloudgen.noise.fbm" => {
            if c.has_tag("stat") {
                return None; // judged by fbm_stat_cases_stay_in_their_bands, see EXPECTED_FAILURES
            }
            let pts = c.input("points");
            let v = fbm(&points_as_f32(pts), &fbm_plan_of(c));
            Out::new().f32("value", &lead(pts), &v)
        }
        "cloudgen.noise.octaves_below" => {
            let (f, m) = (c.input("frequency").as_f64(), c.input("max_frequency").as_f64());
            let lac: Vec<f64> = if c.has_input("lacunarity") { c.input("lacunarity").as_f64() } else { vec![DEFAULT_LACUNARITY; f.len()] };
            let n: Vec<i64> = (0..f.len()).map(|i| octaves_below(f[i], m[i], lac[i]) as i64).collect();
            Out::new().i64("n", &[n.len()], &n)
        }
        "cloudgen.noise.octave_ladder" => {
            let (low, m) = (c.input("low_frequency").as_f64(), c.input("max_frequency").as_f64());
            let lac: Vec<f64> = if c.has_input("lacunarity") { c.input("lacunarity").as_f64() } else { vec![DEFAULT_LACUNARITY; low.len()] };
            let top: Vec<f64> = if c.has_input("top") { c.input("top").as_f64() } else { vec![DEFAULT_TOP; low.len()] };
            let r: Vec<(f64, usize)> = (0..low.len()).map(|i| octave_ladder(low[i], m[i], lac[i], top[i])).collect();
            let start: Vec<f64> = r.iter().map(|&(s, _)| s).collect();
            let n: Vec<i64> = r.iter().map(|&(_, n)| n as i64).collect();
            Out::new().f64("start", &[start.len()], &start).i64("n", &[n.len()], &n)
        }
        "cloudgen.noise.map_max_frequency" => {
            let w = c.input("width").as_f64();
            let mt: Vec<f64> = if c.has_input("min_texels") { c.input("min_texels").as_f64() } else { vec![DEFAULT_MIN_TEXELS; w.len()] };
            let f: Vec<f64> = (0..w.len()).map(|i| map_max_frequency(w[i], mt[i])).collect();
            Out::new().f64("fmax", &[f.len()], &f)
        }
        _ => return None,
    })
}

#[test]
fn noise_cases() {
    let cfg = Config {
        group: "sphere_noise",
        prefixes: &["cloudgen.noise.", "property.cloudgen.noise."],
        expected_failures: EXPECTED_FAILURES,
    };
    run(&cfg, port).assert_clean();
}

/// Points uniform on the unit sphere (normalized normal draws), as float32.
fn sphere_points(rng: &mut Rng, n: usize) -> Vec<Vec3<f32>> {
    (0..n)
        .map(|_| {
            let (x, y, z) = (rng.normal(), rng.normal(), rng.normal());
            let r = (x * x + y * y + z * z).sqrt();
            [(x / r) as f32, (y / r) as f32, (z / r) as f32]
        })
        .collect()
}

/// The slow `fbm.stat.*` cases for a port with its own generator: per seed, 200000 unit-sphere points and a plan
/// drawn by the port's Rng; the mean and the (population) std of the field must each lie within `k_sigma`
/// standard deviations of the Python distribution (`params.expect`). Runs with GOLDEN_SLOW=1, like the runner's
/// slow cases (use `--release`).
#[test]
fn fbm_stat_cases_stay_in_their_bands() {
    if std::env::var_os("GOLDEN_SLOW").is_none() {
        println!("fbm_stat_cases_stay_in_their_bands: skipped (set GOLDEN_SLOW=1)");
        return;
    }
    let cases: Vec<Case> = load_group("sphere_noise").into_iter().filter(|c| c.function == "cloudgen.noise.fbm" && c.has_tag("stat")).collect();
    assert_eq!(cases.len(), 6, "the six stat cases of fbm");
    for c in &cases {
        let expect = c.param("expect");
        let (em, esm) = (expect["mean"].as_f64().unwrap(), expect["mean_sd"].as_f64().unwrap());
        let (es, ess) = (expect["std"].as_f64().unwrap(), expect["std_sd"].as_f64().unwrap());
        let k = expect["k_sigma"].as_f64().unwrap();
        let (n, seeds) = (c.p_usize("n_points"), c.p_usize("seeds"));
        let (pseed, fseed) = (c.p_i64("point_seed_base") as u64, c.p_i64("fbm_seed_base") as u64);
        for s in 0..seeds as u64 {
            let pts = sphere_points(&mut Rng::new(pseed + s), n);
            let plan = FbmPlan::from_seed(fseed + s, c.p_f64("frequency"), c.p_usize("octaves"), c.p_f64("gain"), c.p_f64("lacunarity"), c.p_bool("billow"));
            let v = fbm(&pts, &plan);
            let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
            let std = (v.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
            assert!((mean - em).abs() <= k * esm, "{} seed {s}: mean {mean} outside {em} +- {k} x {esm}", c.id);
            assert!((std - es).abs() <= k * ess, "{} seed {s}: std {std} outside {es} +- {k} x {ess}", c.id);
        }
        println!("{}: {seeds} seeds inside the band (mean {em} +- {}, std {es} +- {})", c.id, k * esm, k * ess);
    }
}
