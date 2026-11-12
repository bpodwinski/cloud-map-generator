//! The sidecar dict of a generated map, as `generate()` builds it in Python (`meta`), in its key order.
//!
//! `cloud-map-generator-core` returns a [`GenMeta`] without the three entries that are not the generator's to know:
//! `body` and `params` (`asdict` of its inputs) and the wall-clock `seconds`. The command line has all three.

use crate::pyjson::J;
use cloud_map_generator_core::climate::BodyInputs;
use cloud_map_generator_core::flow::VortexKind;
use cloud_map_generator_core::generate::{py_round, GenMeta, GenParams, VortexMeta};

fn int(v: usize) -> J {
    J::uint(v as u64)
}

/// `asdict(body)`.
pub fn body_json(b: &BodyInputs) -> J {
    J::obj([
        ("rotation_period_days", b.rotation_period_days.map_or(J::Null, J::Float)),
        ("radius_km", J::Float(b.radius_km)),
        ("gravity_mps2", J::Float(b.gravity_mps2)),
        ("scale_height_km", J::Float(b.scale_height_km)),
        ("cloud_regime", J::s(b.cloud_regime.as_ref())),
    ])
}

/// `asdict(params)`, in the field order of `GenParams`.
pub fn params_json(p: &GenParams) -> J {
    J::obj([
        ("structure_width", int(p.structure_width)),
        ("a_moisture", J::Float(p.a_moisture)),
        ("a_ascent", J::Float(p.a_ascent)),
        ("a_mid", J::Float(p.a_mid)),
        ("a_fine", J::Float(p.a_fine)),
        ("moisture_noise", J::Float(p.moisture_noise)),
        ("advect_mid", J::Float(p.advect_mid)),
        ("advect_fine", J::Float(p.advect_fine)),
        ("mid_frequency", J::Float(p.mid_frequency)),
        ("fine_frequency", J::Float(p.fine_frequency)),
        ("cumulus_cell_km", J::Float(p.cumulus_cell_km)),
        ("fine_top", J::Float(p.fine_top)),
        ("fine_gain", J::Float(p.fine_gain)),
        ("fine_billow", J::Bool(p.fine_billow)),
        ("fine_min_texels", J::Float(p.fine_min_texels)),
        ("shield_ramp", J::Float(p.shield_ramp)),
        ("cumulus_depth", J::Float(p.cumulus_depth)),
        ("cumulus_max", J::Float(p.cumulus_max)),
        ("shield_texture", J::Float(p.shield_texture)),
        ("threshold_smooth_deg", J::Float(p.threshold_smooth_deg)),
        ("regime_texture", J::Bool(p.regime_texture)),
        ("isotropic", J::Bool(p.isotropic)),
        ("allow_any_regime", J::Bool(p.allow_any_regime)),
        ("surface_terms", J::Bool(p.surface_terms)),
        ("sensitivity_step", J::Float(p.sensitivity_step)),
    ])
}

fn kind_name(k: VortexKind) -> &'static str {
    match k {
        VortexKind::Cyclone => "cyclone",
        VortexKind::Anticyclone => "anticyclone",
        VortexKind::Tropical => "tropical",
    }
}

/// One entry of `meta['flow']['vortices']`: `asdict(vortex)` without `lon` and `lat`, then `lon_deg`, `lat_deg`.
fn vortex_json(v: &VortexMeta) -> J {
    J::obj([
        ("kind", J::s(kind_name(v.kind))),
        ("radius", J::Float(v.radius)),
        ("omega", J::Float(v.omega)),
        ("ascent", J::Float(v.ascent)),
        ("age", J::Float(v.age)),
        ("lon_deg", J::Float(v.lon_deg)),
        ("lat_deg", J::Float(v.lat_deg)),
    ])
}

/// The `meta` dict `generate()` returns: `seed`, `width`, `height`, `orientation`, `body`, `hadley_edge_deg`,
/// `params`, `fine_octaves`, `flow`, `seconds` (rounded to a tenth like Python), and `surface` when the body
/// had one.
pub fn gen_meta(body: &BodyInputs, params: &GenParams, meta: &GenMeta, seconds: f64) -> J {
    let mut m = J::obj([
        ("seed", J::uint(meta.seed)),
        ("width", int(meta.width)),
        ("height", int(meta.height)),
        ("orientation", J::s(meta.orientation)),
        ("body", body_json(body)),
        ("hadley_edge_deg", J::Float(meta.hadley_edge_deg)),
        ("params", params_json(params)),
        ("fine_octaves", int(meta.fine_octaves)),
        (
            "flow",
            J::obj([
                ("jet", J::Float(meta.flow.jet)),
                ("trades", J::Float(meta.flow.trades)),
                ("polar_easterlies", J::Float(meta.flow.polar_easterlies)),
                ("vortices", J::Arr(meta.flow.vortices.iter().map(vortex_json).collect())),
            ]),
        ),
        ("seconds", J::Float(py_round(seconds, 1))),
    ]);
    if let Some(s) = &meta.surface {
        m.push(
            "surface",
            J::obj([
                ("source", J::s(s.source.as_str())),
                ("land_fraction", J::Float(s.land_fraction)),
                ("itcz_shift_deg", J::Float(s.itcz_shift_deg)),
                ("anomaly_range", J::floats(&s.anomaly_range)),
            ]),
        );
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyjson::dumps;
    use cloud_map_generator_core::climate::EARTH;
    use cloud_map_generator_core::generate::{FlowMeta, SurfaceMeta};

    fn meta(surface: Option<SurfaceMeta>) -> GenMeta {
        GenMeta {
            seed: 42,
            width: 8,
            height: 4,
            orientation: "north-up",
            hadley_edge_deg: 35.0,
            fine_octaves: 3,
            flow: FlowMeta {
                jet: 1.0,
                trades: 0.05,
                polar_easterlies: 0.25,
                vortices: vec![VortexMeta { kind: VortexKind::Cyclone, radius: 0.1, omega: -1.5, ascent: 0.5, age: 1.0, lon_deg: 12.5, lat_deg: -40.0 }],
            },
            surface,
        }
    }

    #[test]
    fn the_key_order_is_generate_pys() {
        let m = gen_meta(&EARTH, &GenParams::default(), &meta(None), 12.34);
        let J::Obj(items) = &m else { panic!("an object") };
        let keys: Vec<&str> = items.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["seed", "width", "height", "orientation", "body", "hadley_edge_deg", "params", "fine_octaves", "flow", "seconds"]);
        let text = dumps(&m);
        assert!(text.contains("\"seconds\": 12.3\n"), "{text}");
        assert!(text.contains("\"rotation_period_days\": 0.99727"), "{text}");
        assert!(text.contains("\"kind\": \"cyclone\""), "{text}");
        assert!(text.contains("\"structure_width\": 512,"), "{text}");
    }

    #[test]
    fn a_surface_adds_its_block_after_the_seconds() {
        let s = SurfaceMeta { source: "land.u8".into(), land_fraction: 0.2915, itcz_shift_deg: 1.25, anomaly_range: [-0.1, 0.25] };
        let m = gen_meta(&EARTH, &GenParams::default(), &meta(Some(s)), 1.0);
        let J::Obj(items) = &m else { panic!("an object") };
        assert_eq!(items.last().unwrap().0, "surface");
        assert!(dumps(&m).contains("\"anomaly_range\": [\n      -0.1,\n      0.25\n    ]"));
    }

    #[test]
    fn params_list_every_field_of_gen_params() {
        let J::Obj(items) = params_json(&GenParams::default()) else { panic!("an object") };
        assert_eq!(items.len(), 25, "GenParams has 25 fields; add the new one here too");
        let no_spin = BodyInputs::new(None, 1.0, 1.0, 1.0);
        assert!(dumps(&body_json(&no_spin)).contains("\"rotation_period_days\": null"));
    }
}
