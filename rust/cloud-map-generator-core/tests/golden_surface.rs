//! Golden cases of cloudgen/surface.py (group texture_surface, functions cloudgen.surface.*). The
//! cloudgen.texture.* cases of the same group belong to the texture module.

use cloud_map_generator_core::surface::*;
use cloud_map_generator_golden::{run, Array, Case, Config, DType, Out};
use serde_json::Value;

const EXPECTED_FAILURES: &[(&str, &str)] = &[];

/// The constants a case lists in `params.constants` ("a port copies them from its own source and may assert
/// that they equal these"): every one must be known here and equal the port's, exactly.
fn check_constants(c: &Case) {
    let Some(map) = c.params.get("constants").and_then(Value::as_object) else {
        return;
    };
    for (name, want) in map {
        let want: Vec<f64> = match want {
            Value::Array(a) => a.iter().map(|v| v.as_f64().expect("number")).collect(),
            v => vec![v.as_f64().expect("number")],
        };
        let got: Vec<f64> = match name.as_str() {
            "LAND_CONTRAST_CONVERGENCE" => vec![LAND_CONTRAST_CONVERGENCE],
            "LAND_CONTRAST_EXTRATROPICS" => vec![LAND_CONTRAST_EXTRATROPICS],
            "LAND_CONTRAST_DESERT" => vec![LAND_CONTRAST_DESERT],
            "CONTINENTALITY_KM" => vec![CONTINENTALITY_KM],
            "DECK_AMPLITUDE" => vec![DECK_AMPLITUDE],
            "DECK_EFOLD_KM" => vec![DECK_EFOLD_KM],
            "DECK_COAST_LAND" => vec![DECK_COAST_LAND],
            "DECK_MAX_SEARCH_KM" => vec![DECK_MAX_SEARCH_KM],
            "OROGRAPHIC_AMPLITUDE" => vec![OROGRAPHIC_AMPLITUDE],
            "RAIN_SHADOW_AMPLITUDE" => vec![RAIN_SHADOW_AMPLITUDE],
            "OROGRAPHIC_W0_MPS" => vec![OROGRAPHIC_W0_MPS],
            "RAIN_SHADOW_KM" => vec![RAIN_SHADOW_KM],
            "SHADOW_DESCENT_KM" => vec![SHADOW_DESCENT_KM.0, SHADOW_DESCENT_KM.1],
            "EARTH_ITCZ_SHIFT_X" => vec![EARTH_ITCZ_SHIFT_X],
            "EARTH_LAND_ASYMMETRY" => vec![EARTH_LAND_ASYMMETRY],
            "ITCZ_SHIFT_MAX_X" => vec![ITCZ_SHIFT_MAX_X],
            "band_edges_x" => EARTH_BAND_EDGES_X.to_vec(),
            "cos_floor" | "dx_cos_floor" | "backtrace_cos_floor" => vec![COS_FLOOR],
            "row_min_sigma_px" => vec![ROW_MIN_SIGMA_PX],
            "row_max_sigma_px_fraction_of_width" => vec![ROW_MAX_SIGMA_WIDTH_FRACTION],
            "speed_floor_mps" => vec![SPEED_FLOOR_MPS],
            "shadow_fractions" => SHADOW_FRACTIONS.to_vec(),
            "shadow_weight_per_fraction" => vec![SHADOW_WEIGHT_PER_FRACTION],
            "convergence_gauss_mu_s" => vec![CONVERGENCE_GAUSS.0, CONVERGENCE_GAUSS.1],
            "extratropics_rise" => vec![EXTRATROPICS_RISE.0, EXTRATROPICS_RISE.1],
            "extratropics_fade" => vec![EXTRATROPICS_FADE.0, EXTRATROPICS_FADE.1],
            "desert_gauss_mu_s" => vec![DESERT_GAUSS.0, DESERT_GAUSS.1],
            "rise" | "window_rise" => vec![WINDOW_RISE.0, WINDOW_RISE.1],
            "fall" | "window_fall" => vec![WINDOW_FALL.0, WINDOW_FALL.1],
            other => panic!("case {}: constant {other:?} is not mapped to the port", c.id),
        };
        assert_eq!(got, want, "case {}: constant {name} differs from the Python source's", c.id);
    }
}

/// The Earth centering tables are inputs of some cases: they must be the port's own module constants.
fn check_tables(c: &Case, x: &str, cont: &str, deck: &str) {
    assert_eq!(c.input(x).as_f64(), EARTH_TABLE_X.to_vec(), "case {}: EARTH_TABLE_X", c.id);
    assert_eq!(c.input(cont).as_f64(), EARTH_CONTINENTALITY.to_vec(), "case {}: EARTH_CONTINENTALITY", c.id);
    assert_eq!(c.input(deck).as_f64(), EARTH_DECK.to_vec(), "case {}: EARTH_DECK", c.id);
}

/// A float input of any width as its shape and f64 values.
fn f64s(a: &Array) -> (Vec<usize>, Vec<f64>) {
    (a.shape.clone(), a.to_f64())
}

/// A surface from a land array (elevation optional): the shapes are (h, w).
fn surface_of(land: &Array, elevation: Option<&Array>) -> Surface {
    let (h, w) = (land.shape[0], land.shape[1]);
    let elev = elevation.map_or_else(|| vec![0.0; h * w], Array::as_f32);
    Surface::new(land.as_f32(), elev, h, w, "")
}

fn ratio_params(c: &Case) -> (f64, f64) {
    (c.p_f64("radius_km"), c.p_f64("spin"))
}

fn port(c: &Case) -> Option<Out> {
    if c.function.starts_with("cloudgen.surface.") {
        check_constants(c);
    }
    Some(match c.function.as_str() {
        "cloudgen.surface._gauss" => {
            let (sh, x) = f64s(c.input("x"));
            let y: Vec<f64> = x.iter().map(|&v| gauss(v, c.p_f64("mu"), c.p_f64("s"))).collect();
            Out::new().f64("y", &sh, &y)
        }
        "cloudgen.surface._smoothstep" => {
            let (sh, x) = f64s(c.input("x"));
            let y: Vec<f64> = x.iter().map(|&v| smoothstep(c.p_f64("a"), c.p_f64("b"), v)).collect();
            Out::new().f64("y", &sh, &y)
        }
        "cloudgen.surface.land_contrast" => {
            let (sh, x) = f64s(c.input("x"));
            let y: Vec<f64> = x.iter().map(|&v| land_contrast(v)).collect();
            Out::new().f64("contrast", &sh, &y)
        }
        "cloudgen.surface.deck_window" => {
            let (sh, x) = f64s(c.input("x"));
            let y: Vec<f64> = x.iter().map(|&v| deck_window(v)).collect();
            Out::new().f64("window", &sh, &y)
        }
        "cloudgen.surface.earth_tables_at" => {
            check_tables(c, "table_x", "table_continentality", "table_deck");
            let (sh, x) = f64s(c.input("x"));
            let (cont, deck): (Vec<f64>, Vec<f64>) = x.iter().map(|&v| earth_tables_at(v)).unzip();
            Out::new().f64("continentality", &sh, &cont).f64("deck", &sh, &deck)
        }
        "cloudgen.surface.Surface.from_elevation" => {
            let e = c.input("elevation_in_km");
            let (h, w) = (e.shape[0], e.shape[1]);
            let (sea, src) = (c.p_f64("sea_level_km"), c.p_str("source"));
            let s = match e.dtype {
                DType::F32 => Surface::from_elevation(&e.as_f32(), h, w, sea, src),
                _ => Surface::from_elevation(&e.as_f64(), h, w, sea, src),
            };
            assert_eq!(s.source, src);
            Out::new().f32("land", &[h, w], &s.land).f32("elevation_km", &[h, w], &s.elevation_km)
        }
        "cloudgen.surface.Surface.at" => {
            let s = surface_of(c.input("src_land"), Some(c.input("src_elevation_km")));
            let r = s.at(c.p_usize("width"));
            Out::new().f32("land", &[r.h, r.w], &r.land).f32("elevation_km", &[r.h, r.w], &r.elevation_km)
        }
        "cloudgen.surface.Surface.load" => {
            if c.id.starts_with("load.none_and_errors") {
                return Some(load_none_and_errors());
            }
            let rec = &c.params["record"];
            let (name_land, name_elev) = (rec["land"].as_str().unwrap().to_string(), rec["elevation"].as_str().unwrap().to_string());
            let record = SurfaceRecord {
                width: rec["width"].as_u64().unwrap() as usize,
                height: rec["height"].as_u64().unwrap() as usize,
                land: name_land.clone(),
                elevation: name_elev.clone(),
                source: rec.get("source").and_then(Value::as_str).map(String::from),
            };
            let land_bytes = c.input("land_u8").as_u8();
            let elev_bytes: Vec<u8> = c.input("elevation_i16").as_i16().iter().flat_map(|v| v.to_le_bytes()).collect();
            let reader = |name: &str| -> Option<Vec<u8>> {
                if name == name_land {
                    Some(land_bytes.clone())
                } else if name == name_elev {
                    Some(elev_bytes.clone())
                } else {
                    None
                }
            };
            let s = Surface::load(Some(&record), reader).expect("loads").expect("a surface");
            assert_eq!(s.source, c.p_str("expected_source"));
            let (h, w) = (s.h, s.w);
            let src = s.source.as_bytes();
            Out::new().f32("land", &[h, w], &s.land).f32("elevation_km", &[h, w], &s.elevation_km).u8("source_utf8", &[src.len()], src)
        }
        "cloudgen.surface.smooth_on_sphere" => {
            let a = c.input("a");
            let (h, w) = (a.shape[0], a.shape[1]);
            let (sigma, radius) = (c.p_f64("sigma_km"), c.p_f64("radius_km"));
            let o = match a.dtype {
                DType::F32 => smooth_on_sphere(&a.as_f32(), h, w, sigma, radius),
                _ => smooth_on_sphere(&a.as_f64(), h, w, sigma, radius),
            };
            Out::new().f32("smoothed", &[h, w], &o)
        }
        "cloudgen.surface.continentality" => {
            let s = surface_of(c.input("land"), None);
            Out::new().f32("continentality", &[s.h, s.w], &continentality(&s, c.p_f64("radius_km")))
        }
        "cloudgen.surface.deck_strength" => {
            let cont = c.input("cont");
            let (h, w) = (cont.shape[0], cont.shape[1]);
            let (radius, spin) = ratio_params(c);
            Out::new().f32("strength", &[h, w], &deck_strength(&cont.as_f32(), h, w, radius, spin))
        }
        "cloudgen.surface.earth_tables" => {
            let s = surface_of(c.input("land"), Some(c.input("elevation_km")));
            let (radius, spin) = ratio_params(c);
            let (hadley, width) = (c.p_f64("hadley_deg"), c.p_usize("width"));
            let (cont, deck) = earth_tables(&s, radius, spin, hadley, width);
            let row_x = hadley_x(width, width / 2, hadley);
            Out::new().f64("continentality", &[15], &cont).f64("deck", &[15], &deck).f64("row_x", &[row_x.len()], &row_x)
        }
        "cloudgen.surface.hemispheric_land_asymmetry" => {
            let s = surface_of(c.input("land"), None);
            Out::new().f64("asymmetry", &[], &[hemispheric_land_asymmetry(&s)])
        }
        "cloudgen.surface.itcz_shift_deg" => {
            let hadley = c.p_f64("hadley_deg");
            let shift = if c.p_bool("surface_is_none") {
                itcz_shift_deg(None, hadley)
            } else {
                itcz_shift_deg(Some(&surface_of(c.input("land"), None)), hadley)
            };
            Out::new().f64("shift_deg", &[], &[shift])
        }
        "cloudgen.surface.orographic_anomaly" => {
            let elev = c.input("elevation_km");
            let surface = Surface::new(vec![0.0; elev.len()], elev.as_f32(), elev.shape[0], elev.shape[1], "");
            let wind = c.input("wind");
            let (h, w) = (wind.shape[0], wind.shape[1]);
            let o = orographic_anomaly(&surface, &wind.as_f32(), h, w, c.p_f64("radius_km"));
            Out::new().f32("anomaly", &[h, w], &o)
        }
        "cloudgen.surface._resample" => {
            let a = c.input("a");
            let (h, w, width) = (a.shape[0], a.shape[1], c.p_usize("width"));
            let oshape = [if w == width { h } else { width / 2 }, width];
            match a.dtype {
                DType::F32 => Out::new().f32("out", &oshape, &resample(&a.as_f32(), h, w, width)),
                _ => Out::new().f64("out", &oshape, &resample(&a.as_f64(), h, w, width)),
            }
        }
        "cloudgen.surface.surface_terms" => {
            check_tables(c, "earth_table_x", "earth_table_continentality", "earth_table_deck");
            let land = c.input("surface_land");
            let surface = surface_of(land, c.has_input("surface_elevation_km").then(|| c.input("surface_elevation_km")));
            let wind_arr = c.has_input("wind").then(|| c.input("wind"));
            let wind_data = wind_arr.map(Array::as_f32);
            let wind = wind_arr.zip(wind_data.as_deref()).map(|(a, uv)| WindGrid { uv, h: a.shape[0], w: a.shape[1] });
            let (radius, spin) = ratio_params(c);
            let t = surface_terms(&surface, c.p_usize("width"), c.p_f64("hadley_deg"), radius, spin, wind.as_ref(), c.p_usize("term_width"));
            let sh = [t.h, t.w];
            Out::new()
                .f32("anomaly", &sh, &t.anomaly)
                .f32("land", &sh, &t.land)
                .f32("deck", &sh, &t.deck)
                .f32("orographic", &sh, &t.orographic)
                .f32("continentality", &sh, &t.continentality)
                .f32("deck_strength", &sh, &t.deck_strength)
                .f64("itcz_shift_deg", &[], &[t.itcz_shift_deg])
        }
        _ => return None,
    })
}

/// `load(None)` and `load({})` give no surface (here both are `None`: the JSON-level difference between a null
/// and an empty record lives in the bodies-file reader), a record naming a missing land or elevation file is an
/// error, a record whose files exist loads. Each output is a boolean that must be true.
fn load_none_and_errors() -> Out {
    let record = SurfaceRecord { width: 2, height: 1, land: "land.u8".into(), elevation: "elev.i16".into(), source: None };
    let both = |name: &str| match name {
        "land.u8" => Some(vec![0u8, 255]),
        "elev.i16" => Some(vec![0u8; 4]),
        _ => None,
    };
    let only_elev = |name: &str| (name == "elev.i16").then(|| vec![0u8; 4]);
    let only_land = |name: &str| (name == "land.u8").then(|| vec![0u8, 255]);
    let none_for_none = matches!(Surface::load(None, both), Ok(None));
    let none_for_empty = matches!(Surface::load(None, only_elev), Ok(None));
    let missing_land = matches!(Surface::load(Some(&record), only_elev), Err(SurfaceError::FileNotFound(_)));
    let missing_elev = matches!(Surface::load(Some(&record), only_land), Err(SurfaceError::FileNotFound(_)));
    let ok = matches!(Surface::load(Some(&record), both), Ok(Some(_)));
    Out::new()
        .bool("none_for_none", &[1], &[none_for_none])
        .bool("none_for_empty_dict", &[1], &[none_for_empty])
        .bool("raises_missing_land", &[1], &[missing_land])
        .bool("raises_missing_elevation", &[1], &[missing_elev])
        .bool("ok_when_both_exist", &[1], &[ok])
}

#[test]
fn surface_cases() {
    run(&Config { group: "texture_surface", prefixes: &["cloudgen.surface."], expected_failures: EXPECTED_FAILURES }, port).assert_clean();
}
