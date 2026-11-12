//! Golden cases of cloudgen/tiles.py and bake_tiles.py (group tiles: `cloudgen.tiles.*`,
//! `scipy.special.ndtr`, `bake_tiles.main`).
//!
//! Two kinds of case need a word:
//!
//! * `stat` cases (tagged slow, run with `GOLDEN_SLOW=1`) freeze Python's own distribution of a metric over
//!   its seeds (`values`, `mean`, `std`), which no port that draws its own numbers can reproduce. Their
//!   contract, stated in the case doc, is a statistical check: the glue below computes every metric as
//!   defined for the port's own `Rng` seeds (the seeds the case lists, fed to the port's generator) and
//!   PANICS (so the case FAILS) when the first seed is farther than `n_sigma` standard deviations from the
//!   frozen mean, or when the mean of the port's `m` seeds is farther than `n_sigma * std / sqrt(m)`: the two
//!   checks the case doc states. (Not every seed is held to `n_sigma * std`: the frozen std is itself an
//!   estimate from the 30 to 40 seeds of Python, off by up to ~25 percent in the worst metrics, which would
//!   fail a correct port on a seed in the tail.) Only after both checks hold does the glue hand the frozen
//!   arrays back to the runner, whose comparison is then trivially true: the verdict is the panic.
//! * `bake_tiles.main` freezes the byte layout of the bake at small size: the flags, the file names, the RGBA
//!   bytes, the preview image and the sidecar (compared as canonical JSON, the way the case stores it). The
//!   gzip and PNG encoders are a later CLI job and are not frozen.
//!
//! The cases feed the kernels Python's draws, so nothing in the outputs exercises the constructors that draw
//! a plan with the port's own `Rng`, nor the literals only those constructors read. The glue closes that gap
//! with data the cases record: it checks every recorded `params.constants`, `params.literals` and
//! `params.defaults` against the port's constants, and draws a plan with the constructor and requires the
//! shape of the recorded one (the lattice sizes, the number of candidates for the case's size field, the
//! number of lumps, the length of every white noise).

use cloud_map_generator_core::rng::Rng;
use cloud_map_generator_core::tiles::*;
use cloud_map_generator_golden::{run, Array, Case, Config, DType, Out};
use serde_json::Value;

/// Known failures: none. Every case, slow ones included, passes with the port's own plans.
const EXPECTED_FAILURES: &[(&str, &str)] = &[];

// ---------------------------------------------------------------------------------------------------------
// Reading the plans a case records

fn pts(a: &Array) -> Vec<[f64; 2]> {
    a.to_f64().chunks_exact(2).map(|c| [c[0], c[1]]).collect()
}

fn flat(p: &[[f64; 2]]) -> Vec<f64> {
    p.iter().flatten().copied().collect()
}

fn plan_fbm(c: &Case, i: usize) -> FbmPlan {
    FbmPlan { white: c.input(&format!("plan{i}.white")).as_f64() }
}

fn plan_worley(c: &Case, i: usize) -> WorleyPlan {
    let a = c.input(&format!("plan{i}.jitter"));
    WorleyPlan { freq: a.shape[0], jitter: a.as_f64() }
}

fn plan_poisson(c: &Case, i: usize) -> PoissonPlan {
    PoissonPlan { cand: pts(c.input(&format!("plan{i}.cand"))), keep: c.input(&format!("plan{i}.keep")).as_bool() }
}

fn plan_lumps(c: &Case, i: usize) -> LumpsPlan {
    LumpsPlan { g: pts(c.input(&format!("plan{i}.g"))), r: c.input(&format!("plan{i}.r")).as_f64() }
}

/// The three draws of a `warped_voronoi` starting at plan `i`: Poisson, warp x, warp y.
fn plan_voronoi(c: &Case, i: usize) -> WarpedVoronoiPlan {
    WarpedVoronoiPlan { poisson: plan_poisson(c, i), warp_x: plan_fbm(c, i + 1), warp_y: plan_fbm(c, i + 2) }
}

/// The 20 draws of `raw_channels` / `build_tile`, in call order (params.plan_roles lists the roles). Plans 6
/// and 7 (and 15 and 16) are the same recorded draw, and the case stores both: the port has one field for
/// each pair (the crack width and the wall width are the y-warp noise of their channel), so the glue requires
/// the two recorded copies to be equal and feeds one.
fn raw_plan(c: &Case) -> RawChannelsPlan {
    assert_eq!(plan_fbm(c, 7), plan_fbm(c, 6), "{}: the crack width must be the y-warp draw (plans 7 and 6)", c.id);
    assert_eq!(plan_fbm(c, 16), plan_fbm(c, 15), "{}: the wall width must be the y-warp draw (plans 16 and 15)", c.id);
    RawChannelsPlan {
        clusters: plan_worley(c, 0),
        cells: plan_worley(c, 1),
        puffs: plan_worley(c, 2),
        closed: ClosedPlan {
            cell_size: plan_fbm(c, 3),
            voronoi: plan_voronoi(c, 4),
            open_crack: plan_fbm(c, 8),
            depth: plan_fbm(c, 9),
            tops: plan_worley(c, 10),
            texture: plan_fbm(c, 11),
        },
        open: OpenPlan {
            cell_size: plan_fbm(c, 12),
            voronoi: plan_voronoi(c, 13),
            wall_gaps: plan_fbm(c, 17),
            lumps: plan_lumps(c, 18),
        },
        sheet: plan_fbm(c, 19),
    }
}

// ---------------------------------------------------------------------------------------------------------
// What the cases record about the port's constants and its plan constructors

fn json_f64(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("a number was expected, got {v}"))
}

fn json_pair(v: &Value) -> (f64, f64) {
    let a = v.as_array().unwrap_or_else(|| panic!("a pair was expected, got {v}"));
    assert_eq!(a.len(), 2, "a pair was expected, got {v}");
    (json_f64(&a[0]), json_f64(&a[1]))
}

/// The entries of a recorded table (`params.constants`, `params.literals`, `params.defaults`), if the case has it.
fn recorded_table(c: &Case, key: &str) -> Vec<(String, Value)> {
    if !c.has_param(key) {
        return Vec::new();
    }
    let table = c.param(key).as_object().unwrap_or_else(|| panic!("{}: params.{key} is not an object", c.id));
    table.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
}

/// Every constant a case records (`params.constants`, `params.literals`, `params.defaults`) is the port's.
/// Most of them reach the outputs too, but `TILE_SIZE`, `freq` and `oversample` are read only by code the
/// recorded draws bypass; an unknown name panics, so a new recorded constant cannot go unchecked.
fn check_recorded_constants(c: &Case) {
    for (name, v) in recorded_table(c, "constants") {
        let same = match name.as_str() {
            "TILE_SIZE" => v.as_u64() == Some(TILE_SIZE as u64),
            "REFERENCE_TILE_KM" => json_f64(&v) == REFERENCE_TILE_KM,
            "HEAP_CLUSTERS" => v.as_u64() == Some(HEAP_CLUSTERS as u64),
            "HEAP_CELLS" => v.as_u64() == Some(HEAP_CELLS as u64),
            "PUFFS" => v.as_u64() == Some(PUFFS as u64),
            "HEAP_BASE" => json_f64(&v) == HEAP_BASE,
            "ENVELOPE_FLOOR" => json_f64(&v) == ENVELOPE_FLOOR,
            "CLOSED_CELL_KM" => json_pair(&v) == CLOSED_CELL_KM,
            "OPEN_CELL_KM" => json_pair(&v) == OPEN_CELL_KM,
            "LUMP_KM" => json_f64(&v) == LUMP_KM,
            "LUMP_RADIUS" => json_pair(&v) == LUMP_RADIUS,
            "BORDER_WARP" => json_f64(&v) == BORDER_WARP,
            "SHEET_LOW" => json_f64(&v) == SHEET_LOW,
            "SHEET_SLOPE" => json_f64(&v) == SHEET_SLOPE,
            "CHANNELS" => v == serde_json::json!(CHANNELS),
            "freq" => v.as_u64() == Some(lump_freq() as u64),
            "u_low" => json_f64(&v) == CELL_SIZE_LOW,
            "u_slope" => json_f64(&v) == CELL_SIZE_SLOPE,
            other => panic!("{}: unknown recorded constant {other}", c.id),
        };
        assert!(same, "{}: the recorded constant {name} = {v} is not the port's", c.id);
    }
    for (name, v) in recorded_table(c, "literals") {
        let same = match name.as_str() {
            "oversample" => json_f64(&v) == POISSON_OVERSAMPLE,
            "poisson_spacing" => json_f64(&v) == POISSON_SPACING,
            "warp_low" => json_f64(&v) == WARP_LOW,
            "warp_slope" => json_f64(&v) == WARP_SLOPE,
            // warped_voronoi queries the two nearest points, lumps the LUMP_NEAREST nearest.
            "kd_k" => v.as_u64() == Some(if c.function == "cloudgen.tiles.lumps" { LUMP_NEAREST as u64 } else { 2 }),
            "exponent" => json_f64(&v) == LUMP_EXPONENT,
            // The seeds of the two warp fBms are the plans' business (the plans are recorded).
            "warp_seed_offsets" => v == serde_json::json!([1, 2]),
            other => panic!("{}: unknown recorded literal {other}", c.id),
        };
        assert!(same, "{}: the recorded literal {name} = {v} is not the port's", c.id);
    }
    for (name, v) in recorded_table(c, "defaults") {
        let same = match name.as_str() {
            "size" => v.as_u64() == Some(TILE_SIZE as u64),
            // The default seed of build_tile and of the bake flag are both 1.
            "seed" => v.as_i64() == Some(DEFAULT_BAKE_SEED),
            other => panic!("{}: unknown recorded default {other}", c.id),
        };
        assert!(same, "{}: the recorded default {name} = {v} is not the port's", c.id);
    }
}

/// A Worley plan drawn by the constructor has the lattice of the recorded one.
fn check_worley_draw(c: &Case, rec: &WorleyPlan) {
    let drawn = WorleyPlan::draw(rec.freq, &mut Rng::new(1));
    assert_eq!(drawn.freq, rec.freq, "{}: Worley plan lattice", c.id);
    assert_eq!(drawn.jitter.len(), rec.jitter.len(), "{}: Worley plan shape", c.id);
}

/// An fBm plan drawn by the constructor is a `size x size` field like the recorded one.
fn check_fbm_draw(c: &Case, size: usize, rec: &FbmPlan) {
    assert_eq!(FbmPlan::draw(size, &mut Rng::new(1)).white.len(), rec.white.len(), "{}: fBm plan shape", c.id);
}

/// A Poisson plan drawn by the constructor for the case's size field has as many candidates (and as many
/// thinning flags) as Python drew: `int(12 / min^2)`.
fn check_poisson_draw(c: &Case, size_field: &[f64], m: usize, rec: &PoissonPlan) {
    let drawn = PoissonPlan::draw(size_field, m, &mut Rng::new(1));
    assert_eq!(drawn.cand.len(), rec.cand.len(), "{}: Poisson candidates drawn for this size field", c.id);
    assert_eq!(drawn.keep.len(), rec.keep.len(), "{}: Poisson thinning flags drawn for this size field", c.id);
}

/// A lump plan drawn by the constructor has as many centers and radii as the recorded one (60 x 60).
fn check_lumps_draw(c: &Case, rec: &LumpsPlan) {
    let drawn = LumpsPlan::draw(&mut Rng::new(1));
    assert_eq!((drawn.g.len(), drawn.r.len()), (rec.g.len(), rec.r.len()), "{}: lump plan shape", c.id);
}

/// A `warped_voronoi` plan drawn by the constructor has the three parts of the recorded one.
fn check_voronoi_draw(c: &Case, size: usize, size_field: &[f64], rec: &WarpedVoronoiPlan) {
    check_poisson_draw(c, size_field, size, &rec.poisson);
    check_fbm_draw(c, size, &rec.warp_x);
    check_fbm_draw(c, size, &rec.warp_y);
}

/// A `raw_channels` plan drawn by the constructor has the shape of the recorded one, plan by plan (the
/// Poisson sets aside: their size follows the size field a plan draws for itself, and is checked by the
/// `poisson_points` and `warped_voronoi` cases, which record the size field).
fn check_raw_draw(c: &Case, size: usize, rec: &RawChannelsPlan) {
    let drawn = RawChannelsPlan::draw(size, &mut Rng::new(1));
    for (d, r) in [
        (&drawn.clusters, &rec.clusters),
        (&drawn.cells, &rec.cells),
        (&drawn.puffs, &rec.puffs),
        (&drawn.closed.tops, &rec.closed.tops),
    ] {
        check_worley_draw(c, r);
        assert_eq!((d.freq, d.jitter.len()), (r.freq, r.jitter.len()), "{}: Worley plan of the draw", c.id);
    }
    for (d, r) in [
        (&drawn.closed.cell_size, &rec.closed.cell_size),
        (&drawn.closed.voronoi.warp_x, &rec.closed.voronoi.warp_x),
        (&drawn.closed.voronoi.warp_y, &rec.closed.voronoi.warp_y),
        (&drawn.closed.open_crack, &rec.closed.open_crack),
        (&drawn.closed.depth, &rec.closed.depth),
        (&drawn.closed.texture, &rec.closed.texture),
        (&drawn.open.cell_size, &rec.open.cell_size),
        (&drawn.open.voronoi.warp_x, &rec.open.voronoi.warp_x),
        (&drawn.open.voronoi.warp_y, &rec.open.voronoi.warp_y),
        (&drawn.open.wall_gaps, &rec.open.wall_gaps),
        (&drawn.sheet, &rec.sheet),
    ] {
        assert_eq!(d.white.len(), r.white.len(), "{}: fBm plan of the draw", c.id);
    }
    check_lumps_draw(c, &rec.open.lumps);
    assert_eq!(
        (drawn.open.lumps.g.len(), drawn.open.lumps.r.len()),
        (rec.open.lumps.g.len(), rec.open.lumps.r.len()),
        "{}: lump plan of the draw",
        c.id
    );
}

fn square(c: &Case, name: &str) -> (usize, Vec<usize>) {
    let n = c.p_usize(name);
    (n, vec![n, n])
}

// ---------------------------------------------------------------------------------------------------------
// The statistical cases

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn pop_std(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
}

/// Pearson correlation of two equally long series.
fn corr(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa * sbb).sqrt()
}

/// `np.roll(f, lag, axis)` of a square `n x n` field (axis 1 = along x).
fn roll(f: &[f64], n: usize, lag: usize, axis: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for r in 0..n {
        for col in 0..n {
            let (rr, cc) = if axis == 1 { (r, (col + lag) % n) } else { ((r + lag) % n, col) };
            out[rr * n + cc] = f[r * n + col];
        }
    }
    out
}

/// Pearson correlation of a field with itself shifted by `lag` texels along `axis`, circular.
fn acorr(f: &[f64], n: usize, lag: usize, axis: usize) -> f64 {
    corr(f, &roll(f, n, lag, axis))
}

fn seeds(c: &Case) -> Vec<u64> {
    c.param("seeds").as_array().expect("seeds").iter().map(|v| v.as_u64().expect("seed")).collect()
}

/// Mean over the accepted points of (periodic distance to the nearest OTHER point) / (field size at the
/// point), by brute force.
fn nn_ratio(points: &[[f64; 2]], field: &[f64], m: usize) -> f64 {
    let sizes = sample(field, m, m, points);
    let mut total = 0.0;
    for (i, p) in points.iter().enumerate() {
        let mut best = f64::INFINITY;
        for (j, q) in points.iter().enumerate() {
            if i != j {
                let dx = (p[0] - q[0]).abs();
                let dy = (p[1] - q[1]).abs();
                best = best.min(dx.min(1.0 - dx).hypot(dy.min(1.0 - dy)));
            }
        }
        total += best / sizes[i];
    }
    total / points.len() as f64
}

/// Computes the case's metrics for the port's own seeds, checks them against the frozen statistics (panics on
/// a violation) and returns the frozen arrays for the runner (see the module doc).
fn stat_case(c: &Case) -> Out {
    let names: Vec<String> = c
        .param("metrics")
        .as_array()
        .expect("metrics")
        .iter()
        .map(|v| v.as_str().expect("metric name").to_string())
        .collect();
    let rows: Vec<Vec<f64>> = seeds(c).into_iter().map(|seed| stat_metrics(c, seed)).collect();
    let (want_mean, want_std) = (c.expected["mean"].0.as_f64(), c.expected["std"].0.as_f64());
    let n_sigma = c.p_f64("n_sigma");
    let m = rows.len() as f64;
    assert_eq!(rows[0].len(), names.len(), "{}: {} metrics computed, {} listed", c.id, rows[0].len(), names.len());
    for (k, name) in names.iter().enumerate() {
        // One seed: within n_sigma standard deviations of the frozen mean.
        assert!(
            (rows[0][k] - want_mean[k]).abs() <= n_sigma * want_std[k] + 1e-12,
            "{}: metric {name}, first seed: {} is more than {n_sigma} std ({:e}) from the frozen mean {}",
            c.id,
            rows[0][k],
            want_std[k],
            want_mean[k]
        );
        // The mean of the m seeds: within n_sigma * std / sqrt(m).
        let col_mean = rows.iter().map(|r| r[k]).sum::<f64>() / m;
        assert!(
            (col_mean - want_mean[k]).abs() <= n_sigma * want_std[k] / m.sqrt() + 1e-12,
            "{}: metric {name}: the mean over {m} seeds, {col_mean}, is more than {n_sigma} std / sqrt(m) ({:e}) from the frozen mean {}",
            c.id,
            n_sigma * want_std[k] / m.sqrt(),
            want_mean[k]
        );
    }
    let echo = |n: &str| c.expected[n].0.clone();
    Out::new().put("values", echo("values")).put("mean", echo("mean")).put("std", echo("std"))
}

/// The metrics of one seed, in the order of params.metrics.
fn stat_metrics(c: &Case, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    match c.function.as_str() {
        "cloudgen.tiles.tileable_worley" => {
            let (size, freq) = (c.p_usize("size"), c.p_usize("freq"));
            let (f1, f2) = tileable_worley(size, &WorleyPlan::draw(freq, &mut rng));
            let diff: Vec<f64> = f1.iter().zip(&f2).map(|(a, b)| b - a).collect();
            vec![mean(&f1), mean(&f2), mean(&diff), pop_std(&f1)]
        }
        "cloudgen.tiles.periodic_fbm" => {
            let (size, low, slope) = (c.p_usize("size"), c.p_f64("low"), c.p_f64("slope"));
            let f = periodic_fbm(size, low, slope, &FbmPlan::draw(size, &mut rng));
            let big = f.iter().filter(|v| v.abs() > 2.0).count() as f64 / f.len() as f64;
            let kurt = f.iter().map(|v| v.powi(4)).sum::<f64>() / f.len() as f64 - 3.0;
            vec![acorr(&f, size, 1, 1), acorr(&f, size, 1, 0), acorr(&f, size, 4, 1), big, kurt]
        }
        "cloudgen.tiles.poisson_points" => {
            let (open, closed) = (c.input("field_open").as_f64(), c.input("field_closed").as_f64());
            let m = c.input("field_open").shape[0];
            let spacing = c.p_f64("spacing");
            let po = poisson_points(&open, m, &PoissonPlan::draw(&open, m, &mut rng), spacing);
            let pc = poisson_points(&closed, m, &PoissonPlan::draw(&closed, m, &mut rng), spacing);
            vec![po.len() as f64, pc.len() as f64, nn_ratio(&po, &open, m), nn_ratio(&pc, &closed, m)]
        }
        "cloudgen.tiles.lumps" => {
            let d = lumps(c.p_usize("size"), &LumpsPlan::draw(&mut rng));
            let n = d.len() as f64;
            vec![
                d.iter().filter(|&&v| v == 0.0).count() as f64 / n,
                mean(&d),
                d.iter().filter(|&&v| v > 0.5).count() as f64 / n,
            ]
        }
        "cloudgen.tiles.warped_voronoi" => {
            let size = c.p_usize("size");
            let field = c.input("size_field").as_f64();
            let (border, s_cell) = warped_voronoi(size, &field, &WarpedVoronoiPlan::draw(size, &field, &mut rng));
            let n = border.len() as f64;
            vec![border.iter().filter(|&&v| v < 0.05).count() as f64 / n, mean(&border), mean(&s_cell)]
        }
        "cloudgen.tiles.raw_channels" => {
            let size = c.p_usize("size");
            let raw = raw_channels(size, &RawChannelsPlan::draw(size, &mut rng));
            let ranks: Vec<Vec<f64>> = raw.in_order().iter().map(|f| rank_bytes(f).iter().map(|&b| f64::from(b)).collect()).collect();
            let mut v = vec![
                mean(&raw.heaps),
                mean(&raw.closed),
                mean(&raw.open),
                raw.open.iter().filter(|&&x| x == 0.0).count() as f64 / raw.open.len() as f64,
            ];
            v.extend(ranks.iter().map(|r| acorr(r, size, 1, 1)));
            v.extend(ranks.iter().map(|r| acorr(r, size, 8, 1)));
            for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
                v.push(corr(&ranks[a], &ranks[b]));
            }
            v
        }
        other => panic!("stat case of an unknown function {other}"),
    }
}

// ---------------------------------------------------------------------------------------------------------
// bake_tiles.main

/// The case stores the sidecar as canonical JSON bytes (parsed, then dumped with sorted keys and compact
/// separators): do the same with the text the port writes (serde_json keeps its maps sorted).
fn canonical_json(text: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(text).expect("the sidecar is valid JSON");
    serde_json::to_string(&v).expect("serializes")
}

fn bake_case(c: &Case) -> Out {
    let argv: Vec<String> =
        c.param("argv").as_array().expect("argv").iter().map(|v| v.as_str().expect("flag").to_string()).collect();
    let args = BakeArgs::parse(&argv).expect("the flags parse");
    let tile = c.input("tile");
    let (bytes, side) = (tile.as_u8(), tile.shape[0]);
    let rgba = bake_rgba(&bytes);
    let preview = bake_preview(&bytes, side);
    let sidecar = canonical_json(&bake_sidecar_json(&args));
    let files = bake_files(&args.version, args.size).join("\n");
    Out::new()
        .u8("rgba", &[rgba.len()], &rgba)
        .u8("preview", &[4 * side, 4 * side], &preview)
        .u8("sidecar_json", &[sidecar.len()], sidecar.as_bytes())
        .u8("files", &[files.len()], files.as_bytes())
        .i64("build_args", &[2], &[args.size as i64, args.seed])
}

// ---------------------------------------------------------------------------------------------------------
// The dispatcher

fn port(c: &Case) -> Option<Out> {
    check_recorded_constants(c);
    if c.has_tag("stat") {
        return Some(stat_case(c));
    }
    Some(match c.function.as_str() {
        "cloudgen.tiles._smoothstep" => {
            let (a, b) = (c.p_f64("a"), c.p_f64("b"));
            let x = c.input("x");
            if x.dtype == DType::F32 {
                Out::new().f32("y", &x.shape, &smoothstep::<f32>(a, b, &x.as_f32()))
            } else {
                Out::new().f64("y", &x.shape, &smoothstep::<f64>(a, b, &x.as_f64()))
            }
        }
        "cloudgen.tiles._inverted" => {
            let f1 = c.input("f1");
            Out::new().f64("value", &f1.shape, &inverted(&f1.as_f64()))
        }
        "cloudgen.tiles.tileable_worley" => {
            let (size, shape) = square(c, "size");
            let plan = plan_worley(c, 0);
            assert_eq!(plan.freq, c.p_usize("freq"));
            check_worley_draw(c, &plan);
            let (f1, f2) = tileable_worley(size, &plan);
            Out::new().f64("f1", &shape, &f1).f64("f2", &shape, &f2)
        }
        "cloudgen.tiles.periodic_fbm" => {
            let (size, shape) = square(c, "size");
            let plan = plan_fbm(c, 0);
            check_fbm_draw(c, size, &plan);
            Out::new().f64("field", &shape, &periodic_fbm(size, c.p_f64("low"), c.p_f64("slope"), &plan))
        }
        "cloudgen.tiles._uniform" => {
            let (size, shape) = square(c, "size");
            let plan = plan_fbm(c, 0);
            check_fbm_draw(c, size, &plan);
            Out::new().f64("value", &shape, &uniform(size, c.p_f64("low"), c.p_f64("slope"), &plan))
        }
        "cloudgen.tiles.cell_size_field" => {
            let (size, shape) = square(c, "size");
            let km = c.p_f64s("km");
            let plan = plan_fbm(c, 0);
            check_fbm_draw(c, size, &plan);
            Out::new().f64("field", &shape, &cell_size_field(size, (km[0], km[1]), &plan))
        }
        "cloudgen.tiles._sample" => {
            let (field, points) = (c.input("field"), pts(c.input("points")));
            let (rows, cols) = (field.shape[0], field.shape[1]);
            let v = sample(&field.as_f64(), rows, cols, &points);
            Out::new().f64("value", &[v.len()], &v)
        }
        "cloudgen.tiles._texel_centers" => {
            let size = c.p_usize("size");
            Out::new().f64("centers", &[size * size, 2], &flat(&texel_centers(size)))
        }
        "cloudgen.tiles.poisson_points" => {
            let field = c.input("size_field");
            let (m, size_field) = (field.shape[0], field.as_f64());
            let plan = plan_poisson(c, 0);
            check_poisson_draw(c, &size_field, m, &plan);
            let (points, index) = poisson_points_indexed(&size_field, m, &plan, c.p_f64("spacing"));
            let index: Vec<i64> = index.into_iter().map(|i| i as i64).collect();
            let thin = PoissonPlan::from_uniforms(&size_field, m, plan.cand.clone(), &c.input("thin_u").as_f64());
            Out::new()
                .f64("points", &[points.len(), 2], &flat(&points))
                .i64("index", &[index.len()], &index)
                .bool("keep_from_u", &[thin.keep.len()], &thin.keep)
        }
        "cloudgen.tiles.warped_voronoi" => {
            let (size, shape) = square(c, "size");
            let (size_field, plan) = (c.input("size_field").as_f64(), plan_voronoi(c, 0));
            check_voronoi_draw(c, size, &size_field, &plan);
            let v = warped_voronoi_cells(size, &size_field, &plan);
            Out::new()
                .f64("border", &shape, &v.border)
                .f64("s_cell", &shape, &v.s_cell)
                .f64("points", &[v.points.len(), 2], &flat(&v.points))
        }
        "cloudgen.tiles.lumps" => {
            let (size, shape) = square(c, "size");
            let plan = plan_lumps(c, 0);
            check_lumps_draw(c, &plan);
            Out::new().f64("dome", &shape, &lumps(size, &plan))
        }
        "cloudgen.tiles.raw_channels" => {
            let (size, shape) = square(c, "size");
            let plan = raw_plan(c);
            check_raw_draw(c, size, &plan);
            let raw = raw_channels(size, &plan);
            Out::new()
                .f64("heaps", &shape, &raw.heaps)
                .f64("closed", &shape, &raw.closed)
                .f64("open", &shape, &raw.open)
                .f64("sheet", &shape, &raw.sheet)
        }
        "cloudgen.tiles.rank_bytes" => {
            let f = c.input("field");
            let rank = if f.dtype == DType::F32 { rank_bytes(&f.as_f32()) } else { rank_bytes(&f.as_f64()) };
            Out::new().u8("rank", &f.shape, &rank)
        }
        "cloudgen.tiles.coverage_masks" => {
            let r = c.input("rank");
            let masks = coverage_masks(&r.as_u8(), &c.p_f64s("coverages"));
            let mut shape = vec![masks.len()];
            shape.extend(&r.shape);
            Out::new().bool("masks", &shape, &masks.concat())
        }
        "cloudgen.tiles.build_tile" => {
            let size = c.p_usize("size");
            let plan = raw_plan(c);
            check_raw_draw(c, size, &plan);
            Out::new().u8("tile", &[size, size, 4], &build_tile(size, &plan))
        }
        "scipy.special.ndtr" => {
            let x = c.input("x");
            let v: Vec<f64> = x.as_f64().into_iter().map(ndtr).collect();
            Out::new().f64("value", &x.shape, &v)
        }
        "bake_tiles.main" => bake_case(c),
        _ => return None,
    })
}

#[test]
fn tiles_cases() {
    run(&Config { group: "tiles", prefixes: &[], expected_failures: EXPECTED_FAILURES }, port).assert_clean();
}
