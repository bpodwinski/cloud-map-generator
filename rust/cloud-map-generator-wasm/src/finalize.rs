//! Everything after the engine has made the maps: the match to Earth, the encodings, the measures against Earth and
//! the JSON answer. It is `run_generation` of `cloud-map-generator-service/src/generate.rs` from the point where the engine
//! returns, minus the `std::time::Instant` that panics on `wasm32-unknown-unknown` (the caller passes the seconds),
//! and `serve/generate.rs`'s answer, with the files returned as bytes instead of kept in a store.

use crate::request::GenRequest;
use cloud_map_generator_core::export::{encode_map, encode_products, Plane, ProductExtras};
use cloud_map_generator_core::climate::hadley_edge_deg;
use cloud_map_generator_core::flow::{build_flow, Flow, VortexKind};
use cloud_map_generator_core::generate::{fine_ladder, isotropic_flow, py_round, Extras, FlowMeta, GenMeta, GenOutput, GenParams, VortexMeta, ORIENTATION};
use cloud_map_generator_core::matching::{match_to_earth, DEFAULT_LEVELS, DEFAULT_STRENGTH};
use cloud_map_generator_core::products::wind_map;
use cloud_map_generator_service::engine::{ComputeUse, MapOptions, SurfaceChoice};
use cloud_map_generator_service::generate::{compute_json, metric_table, measure_map, round5, REF_W};
use cloud_map_generator_service::imageio::gzip9;
use cloud_map_generator_service::images;
use cloud_map_generator_service::json::opt_str;
use cloud_map_generator_service::meta::gen_meta;
use cloud_map_generator_service::physics::{body_warnings, cyclones_per_hemisphere};
use cloud_map_generator_service::project::REFERENCE;
use cloud_map_generator_service::pyjson::J;
use cloud_map_generator_service::reference::Reference;
use std::cell::RefCell;

thread_local! {
    /// Earth's reference map and its measures: about a second to compute, kept for the life of the instance.
    static REFERENCE_CACHE: RefCell<Option<std::rc::Rc<Reference>>> = const { RefCell::new(None) };
}

/// Earth's reference, computed on first use.
pub fn reference() -> Result<std::rc::Rc<Reference>, String> {
    REFERENCE_CACHE.with(|cache| {
        if let Some(r) = cache.borrow().as_ref() {
            return Ok(r.clone());
        }
        let r = std::rc::Rc::new(Reference::from_png(cloud_map_generator_service::REFERENCE_PNG)?);
        *cache.borrow_mut() = Some(r.clone());
        Ok(r)
    })
}

#[cfg(target_arch = "wasm32")]
mod clock {
    use wasm_bindgen::prelude::wasm_bindgen;
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = Date, js_name = now)]
        pub fn date_now() -> f64;
    }
}

/// The wall clock in milliseconds: the page's `Date.now()` in the browser, 0 natively (no test measures time).
#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> f64 {
    clock::date_now()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> f64 {
    0.0
}

/// A finished generation: the answer's JSON and the files it names.
pub struct Finished {
    pub json: String,
    /// Wall seconds of the engine alone, as the caller measured them (for whoever measures the engine).
    pub engine_seconds: f64,
    /// `(name, content type, bytes)`, the seven files of `GET /api/result/<id>/<name>`.
    pub files: Vec<(&'static str, &'static str, Vec<u8>)>,
}

/// `wrap_lon_deg`: a longitude in [-180, 180).
fn wrap_lon_deg(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

fn kind_name(k: VortexKind) -> &'static str {
    match k {
        VortexKind::Cyclone => "cyclone",
        VortexKind::Anticyclone => "anticyclone",
        VortexKind::Tropical => "tropical",
    }
}

/// The sidecar numbers of a map made from `flow` (`GenMeta` of `generate_with_plans`, without a surface).
pub fn gen_meta_of(width: usize, seed: u64, hadley_edge_deg: f64, params: &GenParams, flow: &Flow) -> GenMeta {
    GenMeta {
        seed,
        width,
        height: width / 2,
        orientation: ORIENTATION,
        hadley_edge_deg,
        fine_octaves: fine_ladder(width, params).octaves,
        flow: FlowMeta {
            jet: flow.jet,
            trades: flow.trades,
            polar_easterlies: flow.polar_easterlies,
            vortices: flow
                .vortices
                .iter()
                .map(|v| VortexMeta {
                    kind: v.kind,
                    radius: v.radius,
                    omega: v.omega,
                    ascent: v.ascent,
                    age: v.age,
                    lon_deg: v.lon.to_degrees(),
                    lat_deg: v.lat.to_degrees(),
                })
                .collect(),
        },
        surface: None,
    }
}

/// The `GenOutput` of maps made elsewhere (the bands, the GPU): the three maps, the wind grid (small, so made here in one
/// piece) and the sidecar numbers, from the flow the request implies (`build_flow` is deterministic in the seed).
pub fn output_from_maps(req: &GenRequest, coverage: Vec<f32>, kind: Vec<f32>, high_share: Vec<f32>) -> GenOutput {
    let params = params_of(&req.options);
    let (width, height) = (req.width, req.width / 2);
    let flow = if params.isotropic { isotropic_flow(&req.body) } else { build_flow(&req.body, req.seed) };
    let wind_width = (params.structure_width * 2).min(width);
    let wind = wind_map(&flow, &req.body, wind_width);
    GenOutput {
        coverage,
        height,
        width,
        meta: gen_meta_of(width, req.seed, hadley_edge_deg(&req.body), &params, &flow),
        extras: Some(Extras { kind, high_share, wind, wind_height: wind_width / 2, wind_width }),
        trace: None,
    }
}

/// The `meta` block of the answer.
pub(crate) fn meta_json(m: &GenMeta) -> J {
    J::obj([
        ("hadley_edge_deg", J::Float(m.hadley_edge_deg)),
        ("fine_octaves", J::Int(m.fine_octaves as i64)),
        (
            "flow",
            J::obj([
                ("jet", J::Float(m.flow.jet)),
                ("trades", J::Float(m.flow.trades)),
                ("polar_easterlies", J::Float(m.flow.polar_easterlies)),
                (
                    "vortices",
                    J::Arr(
                        m.flow
                            .vortices
                            .iter()
                            .map(|v| {
                                J::obj([
                                    ("kind", J::s(kind_name(v.kind))),
                                    ("lon_deg", J::Float(py_round(wrap_lon_deg(v.lon_deg), 2))),
                                    ("lat_deg", J::Float(py_round(v.lat_deg, 2))),
                                    ("radius", J::Float(py_round(v.radius, 4))),
                                    ("omega", J::Float(py_round(v.omega, 4))),
                                    ("ascent", J::Float(py_round(v.ascent, 4))),
                                    ("age", J::Float(py_round(v.age, 4))),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
        ),
        (
            "surface",
            m.surface.as_ref().map_or(J::Null, |s| J::obj([("land_fraction", J::Float(s.land_fraction)), ("itcz_shift_deg", J::Float(s.itcz_shift_deg))])),
        ),
    ])
}

/// The generator's knobs of a request (`GenerateRequest::params`).
pub fn params_of(options: &MapOptions) -> GenParams {
    GenParams { isotropic: options.isotropic, allow_any_regime: options.experiment, ..GenParams::default() }
}

/// What the answer needs besides the maps.
pub struct Run<'a> {
    pub req: &'a GenRequest,
    /// The 12 hex digits that name the result in the URLs of the answer.
    pub id: &'a str,
    pub compute: ComputeUse,
    /// Wall seconds of the engine alone (the sidecar's `seconds`).
    pub engine_seconds: f64,
    /// Wall seconds from the start of the request to the start of this call; the call's own time is added.
    pub elapsed_before: f64,
}

/// The match to Earth, the encodings, the measures and the answer.
pub fn finish(run: Run<'_>, output: GenOutput) -> Result<Finished, String> {
    let t_start = now_ms();
    let GenRequest { body, seed, width: _, options, compute: requested } = run.req;
    let (height, width) = (output.height, output.width);
    let params = params_of(options);
    let reference = reference()?;
    let extras = output.extras.as_ref().ok_or("the engine returned no derived maps")?;
    let mut cov = output.coverage.clone();
    let mut meta = gen_meta(body, &params, &output.meta, run.engine_seconds);
    if options.match_earth {
        cov = match_to_earth(&cov, height, width, &reference.data, reference.h, reference.w, output.meta.hadley_edge_deg, DEFAULT_LEVELS, DEFAULT_STRENGTH);
        meta.push("matched_to", J::s(REFERENCE));
    }
    meta.push(
        "options",
        J::obj([
            ("match_earth", J::Bool(options.match_earth)),
            ("isotropic", J::Bool(options.isotropic)),
            ("experiment", J::Bool(options.experiment)),
            ("surface", J::s(options.surface.key())),
        ]),
    );
    meta.push("compute", compute_json(requested.key(), &run.compute, None));

    // Products first: they record their encoding in the sidecar the map's files then complete.
    let products = encode_products(
        &ProductExtras {
            type_: Some(Plane::new(&extras.kind, height, width)),
            high_share: Some(Plane::new(&extras.high_share, height, width)),
            wind: Some(Plane::new(&extras.wind, extras.wind_height, extras.wind_width)),
        },
        "coverage",
    );
    meta.push(
        "products",
        J::obj([
            ("type", J::s(products.sidecar.type_)),
            ("high_share", J::s(products.sidecar.high_share)),
            ("wind", J::s(products.sidecar.wind.as_str())),
            ("files", J::obj(products.sidecar.files.iter().map(|(k, n)| (*k, J::s(n.as_str()))))),
        ]),
    );
    let map = encode_map(&cov, height, width, "coverage");
    let (type_img, share_img, wind_img) = (
        products.type_png.as_ref().ok_or("the type map was not encoded")?,
        products.high_share_png.as_ref().ok_or("the high-share map was not encoded")?,
        products.wind_png.as_ref().ok_or("the wind map was not encoded")?,
    );
    let files = vec![
        ("coverage.png", "image/png", images::gray8(width, height, &map.preview8)),
        ("type.png", "image/png", images::gray8(type_img.width, type_img.height, &type_img.data)),
        ("high_share.png", "image/png", images::gray8(share_img.width, share_img.height, &share_img.data)),
        ("wind.png", "image/png", images::rgb8(wind_img.width, wind_img.height, &wind_img.data)),
        ("master16.png", "image/png", images::gray16(width, height, &map.master16)),
        ("coverage.r8.gz", "application/gzip", gzip9(&map.r8_south_up)),
        ("meta.json", "application/json; charset=utf-8", cloud_map_generator_service::pyjson::dumps(&meta).into_bytes()),
    ];

    let itcz = output.meta.surface.as_ref().map_or(0.0, |s| s.itcz_shift_deg);
    let measured = measure_map(&reference, &cov, height, width, Some((output.meta.hadley_edge_deg, itcz)));
    let mut warnings = body_warnings(body, cyclones_per_hemisphere(body));
    if options.isotropic {
        warnings.push("Isotropic baseline: no flow, no vortices; the surface is ignored. The type, high-share and wind maps still come from the same fields.".to_string());
    }
    let metrics_note = (width < REF_W).then(|| {
        format!("Measured on a {width}-px map: its finest scales are interpolated by the gnomonic patches, so spectrum and object sizes only compare at 2048 or more.")
    });

    let total_seconds = run.elapsed_before + (now_ms() - t_start) / 1000.0;
    let stem = format!("cloudgen_s{seed}_{width}x{height}_{}", run.id);
    let href = |name: &str| format!("/api/result/{}/{name}", run.id);
    let floats = |v: &[f64]| J::Arr(v.iter().map(|&x| J::Float(round5(x))).collect());
    let m = &measured;
    let answer = J::obj([
        ("id", J::s(run.id)),
        ("seconds", J::Float(py_round(total_seconds, 2))),
        ("width", J::Int(width as i64)),
        ("height", J::Int(height as i64)),
        ("compute", compute_json(requested.key(), &run.compute, Some(total_seconds))),
        ("meta", meta_json(&output.meta)),
        ("images", J::obj(["coverage", "type", "high_share", "wind"].map(|k| (k, J::s(href(&format!("{k}.png"))))))),
        (
            "downloads",
            J::Arr(vec![
                J::obj([("label", J::s("Cloud fraction, 16-bit PNG")), ("href", J::s(href("master16.png"))), ("filename", J::s(format!("{stem}_16bit.png")))]),
                J::obj([
                    ("label", J::s("Raw R8 for the shader (gzip, rows south-up)")),
                    ("href", J::s(href("coverage.r8.gz"))),
                    ("filename", J::s(format!("{stem}.r8.gz"))),
                ]),
                J::obj([("label", J::s("Sidecar JSON")), ("href", J::s(href("meta.json"))), ("filename", J::s(format!("{stem}.json")))]),
            ]),
        ),
        (
            "metrics",
            J::obj([
                ("table", metric_table(&m.measures, &m.comparison, &reference.measures)),
                ("earth_label", J::s("Earth (Blue Marble)")),
                ("note", opt_str(metrics_note.as_deref())),
            ]),
        ),
        (
            "zonal",
            J::obj([("lat_deg", J::floats(&m.zonal.lat_deg)), ("generated", floats(&m.zonal.generated)), ("earth", floats(&m.zonal.earth)), ("target", floats(&m.zonal.target))]),
        ),
        ("warnings", J::Arr(warnings.into_iter().map(J::Str).collect())),
    ]);
    Ok(Finished { json: cloud_map_generator_service::json::compact(&answer), engine_seconds: run.engine_seconds, files })
}

/// The one-piece engine for what the bands do not cover: the surface terms (a synthetic world, at most 1024 wide).
pub fn generate_whole(req: &GenRequest) -> Result<GenOutput, String> {
    let params = params_of(&req.options);
    let surface = (req.options.surface == SurfaceChoice::Synthetic).then(|| cloud_map_generator_service::synthetic::synthetic_surface(req.width.min(1024)));
    cloud_map_generator_core::generate::generate(&req.body, req.seed, req.width, &params, surface.as_ref(), true).map_err(|e| e.to_string())
}

