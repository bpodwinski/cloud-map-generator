//! Golden cases of cloudgen/metrics.py and cloudgen/match.py (group metrics_match).
//!
//! Conventions of the cases (see each case's doc): a u8 input is a map stored as bytes that the function receives as
//! `float32(u8) / float32(255)` (the division in float32), unless `params.u8_inputs` says "raw bytes"; a case with
//! `params.called_with` calls the function on the float64 promotion of its stored float32 patch; a scalar result is
//! an array of one element, a dict result is one array per key.

use cloud_map_generator_core::matching;
use cloud_map_generator_core::metrics::{self, Measure, MetricMap, PATCH_LATS, PATCH_LONS, PATCH_N};
use cloud_map_generator_golden::{run, Array, Case, Config, DType, Out};
use num_traits::Float;

const EXPECTED_FAILURES: &[(&str, &str)] = &[];

/// A map in the dtype the function receives.
enum Img {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

fn is_raw_u8(c: &Case, name: &str) -> bool {
    c.params
        .get("u8_inputs")
        .and_then(|m| m.get(name))
        .and_then(|v| v.as_str())
        .is_some_and(|s| s.starts_with("raw bytes"))
}

/// The input `name` as the function receives it: float32 and float64 as stored, a u8 map divided by 255 in float32.
fn read_img(c: &Case, name: &str) -> Img {
    let a = c.input(name);
    match a.dtype {
        DType::F32 => Img::F32(a.as_f32()),
        DType::F64 => Img::F64(a.as_f64()),
        DType::U8 => {
            assert!(!is_raw_u8(c, name), "case {}: {name} is raw bytes, which this function does not take", c.id);
            Img::F32(a.as_u8().iter().map(|&b| f32::from(b) / 255.0_f32).collect())
        }
        other => panic!("case {}: input {name} has dtype {}", c.id, other.name()),
    }
}

fn promote(img: Img) -> Img {
    match img {
        Img::F32(v) => Img::F64(v.into_iter().map(f64::from).collect()),
        f64s => f64s,
    }
}

/// Runs `$body` with `$v` bound to the slice of whichever dtype the map has.
macro_rules! on_img {
    ($img:expr, $v:ident => $body:expr) => {
        match &$img {
            Img::F32(v) => {
                let $v = v.as_slice();
                $body
            }
            Img::F64(v) => {
                let $v = v.as_slice();
                $body
            }
        }
    };
}

/// Adds a float array in its own dtype (float32 or float64).
fn put<F: Float>(out: Out, name: &str, shape: &[usize], v: &[F]) -> Out {
    if std::mem::size_of::<F>() == 4 {
        out.f32(name, shape, &v.iter().map(|x| x.to_f32().unwrap()).collect::<Vec<_>>())
    } else {
        out.f64(name, shape, &v.iter().map(|x| x.to_f64().unwrap()).collect::<Vec<_>>())
    }
}

fn hw(a: &Array) -> (usize, usize) {
    assert_eq!(a.shape.len(), 2);
    (a.shape[0], a.shape[1])
}

fn one(out: Out, name: &str, v: f64) -> Out {
    out.f64(name, &[1], &[v])
}

fn strings(c: &Case, name: &str) -> Vec<String> {
    c.param(name).as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()
}

fn contrasts<F: Float, L: Float>(img: &[F], land: &[L], deck: &Option<Img>, h: usize, w: usize) -> metrics::SurfaceContrasts {
    match deck {
        None => metrics::surface_contrasts(img, land, None::<&[f32]>, h, w),
        Some(Img::F32(d)) => metrics::surface_contrasts(img, land, Some(d.as_slice()), h, w),
        Some(Img::F64(d)) => metrics::surface_contrasts(img, land, Some(d.as_slice()), h, w),
    }
}

fn measure_out(m: &Measure, per: &[metrics::PatchMeasure]) -> Out {
    let n = per.len();
    let col = |f: fn(&metrics::PatchMeasure) -> f64| per.iter().map(f).collect::<Vec<f64>>();
    let out = Out::new();
    let out = one(out, "global_mean", m.global_mean)
        .f64("zonal_lat", &[m.zonal_lat.len()], &m.zonal_lat)
        .f64("zonal", &[m.zonal.len()], &m.zonal)
        .f64("hist", &[m.hist.len()], &m.hist);
    let out = one(out, "coherence", m.coherence).f64("coherence_by_lat", &[m.coherence_by_lat.len()], &m.coherence_by_lat);
    let out = one(out, "spectral_slope", m.spectral_slope);
    let out = one(out, "big_object_share", m.big_object_share);
    let out = one(out, "median_object_km2", m.median_object_km2);
    let out = one(out, "seam_ratio", m.seam_ratio).f64("pole_std", &[2], &[m.pole_std.0, m.pole_std.1]);
    out.f64("patch_coherence", &[n], &col(|p| p.coherence))
        .f64("patch_slope", &[n], &col(|p| p.spectral_slope))
        .f64("patch_big", &[n], &col(|p| p.big_object_share))
        .f64("patch_median", &[n], &col(|p| p.median_object_km2))
}

/// `a.astype(float64).reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))`.
fn block_mean8(a: &[f32], h: usize, w: usize) -> Vec<f64> {
    let (bh, bw) = (h / 8, w / 8);
    let mut out = Vec::with_capacity(bh * bw);
    for bi in 0..bh {
        for bj in 0..bw {
            let mut s = 0.0_f64;
            for i in 0..8 {
                for j in 0..8 {
                    s += f64::from(a[(bi * 8 + i) * w + bj * 8 + j]);
                }
            }
            out.push(s / 64.0);
        }
    }
    out
}

fn port(c: &Case) -> Option<Out> {
    Some(match c.function.as_str() {
        // ------------------------------------------------------------------------------------- metrics
        "cloudgen.metrics.zonal_profile" => {
            let img = read_img(c, "img");
            let (h, w) = hw(c.input("img"));
            let band = c.p_f64("band_deg");
            on_img!(img, v => {
                let (centers, prof) = metrics::zonal_profile(v, h, w, band);
                put(Out::new().f64("centers", &[centers.len()], &centers), "profile", &[prof.len()], &prof)
            })
        }
        "cloudgen.metrics.area_histogram" => {
            let img = read_img(c, "img");
            let (h, w) = hw(c.input("img"));
            let bins = c.p_usize("bins");
            let hist = on_img!(img, v => metrics::area_histogram(v, h, w, bins));
            Out::new().f64("hist", &[bins], &hist)
        }
        "cloudgen.metrics.global_mean" => {
            let img = read_img(c, "img");
            let (h, w) = hw(c.input("img"));
            one(Out::new(), "value", on_img!(img, v => metrics::global_mean(v, h, w)))
        }
        "cloudgen.metrics.patches" => {
            let img = read_img(c, "img");
            let (h, w) = hw(c.input("img"));
            let ps = on_img!(img, v => metrics::patches(v, h, w));
            assert_eq!(ps.len(), PATCH_LATS.len() * PATCH_LONS.len());
            let mut out = Out::new();
            if c.has_param("stride_idx") {
                let idx: Vec<usize> = c.p_f64s("stride_idx").iter().map(|&i| i as usize).collect();
                let mut samples = Vec::with_capacity(ps.len() * idx.len() * idx.len());
                for p in &ps {
                    for &r in &idx {
                        for &col in &idx {
                            samples.push(p.data[r * PATCH_N + col]);
                        }
                    }
                }
                let lat: Vec<i32> = ps.iter().map(|p| p.lat).collect();
                let lon: Vec<i32> = ps.iter().map(|p| p.lon).collect();
                out = out
                    .i32("lat", &[ps.len()], &lat)
                    .i32("lon", &[ps.len()], &lon)
                    .f64("samples", &[ps.len(), idx.len(), idx.len()], &samples);
            }
            if c.has_param("crop") {
                let crop = c.p_f64s("crop");
                let (r0, r1, c0, c1) = (crop[0] as usize, crop[1] as usize, crop[2] as usize, crop[3] as usize);
                let p = &ps[c.p_usize("patch_index")];
                assert_eq!((p.lat as i64, p.lon as i64), (c.p_i64("lat"), c.p_i64("lon")));
                let mut data = Vec::new();
                for r in r0..r1 {
                    data.extend_from_slice(&p.data[r * PATCH_N + c0..r * PATCH_N + c1]);
                }
                out = out.f64("patch", &[r1 - r0, c1 - c0], &data);
            }
            out
        }
        "cloudgen.metrics.patch_px_km" => one(Out::new(), "px_km", metrics::patch_px_km()),
        "cloudgen.metrics.coherence" => {
            let img = read_img(c, "patch");
            let img = if c.has_param("called_with") { promote(img) } else { img };
            let (h, w) = hw(c.input("patch"));
            let sigma = c.p_f64("sigma");
            one(Out::new(), "coherence", on_img!(img, v => metrics::coherence(v, h, w, sigma)))
        }
        "cloudgen.metrics.spectral_slope" => {
            let img = read_img(c, "patch");
            let img = if c.has_param("called_with") { promote(img) } else { img };
            let (n, n2) = hw(c.input("patch"));
            assert_eq!(n, n2, "case {}: spectral_slope needs a square patch", c.id);
            // The calls that omit the band use the Python defaults: the port's defaults must be those.
            let (lo, hi) = if c.has_param("bounds_are_python_defaults") {
                (metrics::DEFAULT_LO_KM, metrics::DEFAULT_HI_KM)
            } else {
                (c.p_f64("lo_km"), c.p_f64("hi_km"))
            };
            one(Out::new(), "slope", on_img!(img, v => metrics::spectral_slope(v, n, lo, hi)))
        }
        "cloudgen.metrics.object_stats" => {
            let img = read_img(c, "patch");
            let img = if c.has_param("called_with") { promote(img) } else { img };
            let (h, w) = hw(c.input("patch"));
            let ((big, median), areas) =
                on_img!(img, v => (metrics::object_stats(v, h, w), metrics::object_areas_km2(v, h, w)));
            Out::new()
                .f64("stats", &[2], &[big, median])
                .i32("n_objects", &[1], &[areas.len() as i32])
                .f64("areas_km2", &[areas.len()], &areas)
        }
        "cloudgen.metrics.detail_lost_2x" => {
            let (h, w) = hw(c.input("img"));
            let lost = if c.input("img").dtype == DType::U8 && is_raw_u8(c, "img") {
                metrics::detail_lost_2x(&c.input("img").as_u8(), h, w)
            } else {
                on_img!(read_img(c, "img"), v => metrics::detail_lost_2x(v, h, w))
            };
            one(Out::new(), "lost", lost)
        }
        "cloudgen.metrics.seam_ratio" => {
            let img = read_img(c, "img");
            let (h, w) = hw(c.input("img"));
            one(Out::new(), "ratio", on_img!(img, v => metrics::seam_ratio(v, h, w)))
        }
        "cloudgen.metrics.pole_std" => {
            let img = read_img(c, "img");
            let (h, w) = hw(c.input("img"));
            let (n, s) = on_img!(img, v => metrics::pole_std(v, h, w));
            Out::new().f64("std", &[2], &[n, s])
        }
        "cloudgen.metrics.measure" => {
            let img = read_img(c, "img");
            let (h, w) = hw(c.input("img"));
            let (m, per) = on_img!(img, v => metrics::measure_patches(v, h, w));
            // The aggregated Measure is what `measure` returns.
            assert_eq!(m.coherence_by_lat.len(), PATCH_LATS.len());
            measure_out(&m, &per)
        }
        "cloudgen.metrics.compare" => {
            let side = |p: &str| Measure {
                zonal_lat: c.input(&format!("{p}_zonal_lat")).as_f64(),
                zonal: c.input(&format!("{p}_zonal")).as_f64(),
                hist: c.input(&format!("{p}_hist")).as_f64(),
                ..Measure::default()
            };
            let r = metrics::compare(&side("gen"), &side("ref"));
            Out::new().f64("result", &[2], &[r.zonal_rmse, r.hist_l1])
        }
        "cloudgen.metrics.surface_contrasts" => {
            let (h, w) = hw(c.input("img"));
            let (img, land) = (read_img(c, "img"), read_img(c, "land"));
            let deck = c.has_input("deck").then(|| read_img(c, "deck"));
            let sc = on_img!(img, a => on_img!(land, b => contrasts(a, b, &deck, h, w)));
            let entries = sc.entries();
            let keys = strings(c, "keys");
            assert_eq!(entries.iter().map(|e| e.0.to_string()).collect::<Vec<_>>(), keys, "case {}: key order", c.id);
            let values: Vec<f64> = entries.iter().map(|e| e.1).collect();
            Out::new().f64("values", &[values.len()], &values)
        }
        "cloudgen.metrics.judge_surface" => {
            let keys = strings(c, "keys");
            let judged = strings(c, "judged");
            assert_eq!(judged, metrics::SURFACE_JUDGED.iter().map(|j| j.0.to_string()).collect::<Vec<_>>());
            let reference: MetricMap = keys.iter().cloned().zip(c.input("ref").as_f64()).collect();
            let maps = c.input("maps");
            let (n, k) = (maps.shape[0], maps.shape[1]);
            let flat = maps.as_f64();
            let ms: Vec<MetricMap> =
                (0..n).map(|i| keys.iter().cloned().zip(flat[i * k..(i + 1) * k].iter().copied()).collect()).collect();
            let rows = metrics::judge_surface(&reference, &ms);
            assert_eq!(rows.iter().map(|r| r.metric.to_string()).collect::<Vec<_>>(), judged);
            let col = |f: fn(&metrics::JudgeRow) -> f64| rows.iter().map(f).collect::<Vec<f64>>();
            let pass: Vec<bool> = rows.iter().map(|r| r.pass).collect();
            let m = rows.len();
            Out::new()
                .f64("reference", &[m], &col(|r| r.reference))
                .f64("mean", &[m], &col(|r| r.mean))
                .f64("spread", &[m], &col(|r| r.spread))
                .f64("season", &[m], &col(|r| r.season))
                .f64("tolerance", &[m], &col(|r| r.tolerance))
                .f64("diff", &[m], &col(|r| r.diff))
                .bool("pass", &[m], &pass)
        }
        "cloudgen.metrics (module constants)" => {
            // The season term of every judged metric is part of the constants.
            let judged = c.param("surface_judged");
            for (k, season) in metrics::SURFACE_JUDGED {
                let want = judged.get(k).unwrap_or_else(|| panic!("surface_judged lacks {k}"));
                assert_eq!(want.as_str(), season, "season term of {k}");
            }
            assert_eq!(judged.as_object().unwrap().len(), metrics::SURFACE_JUDGED.len());
            let scalars = [
                metrics::PATCH_HALF_DEG,
                metrics::EARTH_RADIUS_KM,
                metrics::SURFACE_LAT_MAX,
                metrics::SURFACE_SPREAD_FACTOR,
                metrics::surface_floor("itcz_lat"),
                metrics::surface_floor("land_minus_ocean"),
            ];
            Out::new()
                .i32("patch_lats", &[PATCH_LATS.len()], &PATCH_LATS)
                .i32("patch_lons", &[PATCH_LONS.len()], &PATCH_LONS)
                .i32("patch_n", &[1], &[PATCH_N as i32])
                .f64("scalars", &[scalars.len()], &scalars)
        }
        // --------------------------------------------------------------------------------------- match
        "cloudgen.match (module constants)" => Out::new()
            .f64("quantiles", &[matching::quantiles().len()], matching::quantiles())
            .f64("scalars", &[2], &[matching::ZONE_DEG, matching::EARTH_CAP_DEG]),
        "cloudgen.match._blur" => {
            let img = read_img(c, "img");
            let (h, w) = hw(c.input("img"));
            let sigma = c.p_f64("sigma");
            on_img!(img, v => put(Out::new(), "out", &[h, w], &matching::blur(v, h, w, sigma)))
        }
        "cloudgen.match.band_layers" => {
            let img = read_img(c, "img");
            let (h, w) = hw(c.input("img"));
            let sigmas = c.p_f64s("sigmas");
            on_img!(img, v => {
                let (layers, low) = matching::band_layers(v, h, w, &sigmas);
                let stacked: Vec<_> = layers.iter().flatten().copied().collect();
                put(put(Out::new(), "layers", &[sigmas.len(), h, w], &stacked), "residual", &[h, w], &low)
            })
        }
        "cloudgen.match._earth_lat" => {
            let lat = read_img(c, "lat");
            let hadley = c.p_f64("hadley_deg");
            on_img!(lat, v => put(Out::new(), "earth_lat", &[v.len()], &matching::earth_lat(v, hadley)))
        }
        "cloudgen.match._zone_index" => {
            let lat = read_img(c, "lat");
            let hadley = c.p_f64("hadley_deg");
            let zone = on_img!(lat, v => matching::zone_index(v, hadley));
            Out::new().i64("zone", &[zone.len()], &zone)
        }
        "cloudgen.match._match_by_zone" => {
            let (gen, earth) = (read_img(c, "gen"), read_img(c, "earth"));
            let ((h, w), (he, we)) = (hw(c.input("gen")), hw(c.input("earth")));
            let lat_g = c.input("lat_g").as_f64();
            let ze = c.input("ze").to_i64();
            let hadley = c.p_f64("hadley_deg");
            on_img!(gen, g => on_img!(earth, e => {
                let o = matching::match_by_zone(g, h, w, e, he, we, &lat_g, hadley, &ze);
                put(Out::new(), "out", &[h, w], &o)
            }))
        }
        "cloudgen.match.match_to_earth" => {
            let (cov, earth) = (read_img(c, "cov"), read_img(c, "earth"));
            // match_to_earth.cov_f64: the generated map reaches the function as float64.
            let cov = if c.doc.contains("The cov handed to the function is float64") { promote(cov) } else { cov };
            let ((h, w), (he, we)) = (hw(c.input("cov")), hw(c.input("earth")));
            let hadley = c.p_f64("hadley_deg");
            // The calls that omit levels and strength use the Python defaults: the port's defaults must be those.
            let (levels, strength) = if c.has_param("levels_strength_are_python_defaults") {
                (matching::DEFAULT_LEVELS, matching::DEFAULT_STRENGTH)
            } else {
                (c.p_usize("levels"), c.p_f64("strength"))
            };
            let out = on_img!(cov, a => on_img!(earth, e => matching::match_to_earth(a, h, w, e, he, we, hadley, levels, strength)));
            Out::new().f32("out", &[h, w], &out).f64("block_mean8", &[h / 8, w / 8], &block_mean8(&out, h, w))
        }
        _ => return None,
    })
}

#[test]
fn metrics_match_cases() {
    run(&Config { group: "metrics_match", prefixes: &["cloudgen.metrics", "cloudgen.match"], expected_failures: EXPECTED_FAILURES }, port)
        .assert_clean();
}
