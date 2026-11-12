//! The ranges every engine-facing entry point enforces on the inputs of a body, and the cap on its cyclones.
//!
//! Shared by `cloud-map-generator serve` (the API) and `cloud-map-generator-mcp`: both refuse the same values.

/// The ranges, both ends included (`LIMITS` of `ui/server.py` and `cloudgen/service.py`).
pub const LIMITS: [(&str, f64, f64); 5] = [
    ("seed", 0.0, 2147483647.0),
    ("rotation_period_days", -400.0, 400.0),
    ("radius_km", 500.0, 30000.0),
    ("gravity_mps2", 0.1, 100.0),
    ("scale_height_km", 0.5, 100.0),
];

/// The four physical numbers of a body, in the order of the API.
pub const PHYSICS_KEYS: [&str; 4] = ["rotation_period_days", "radius_km", "gravity_mps2", "scale_height_km"];

/// `(min, max)` of the limit named `key`.
pub fn limit(key: &str) -> (f64, f64) {
    LIMITS.iter().find(|(k, _, _)| *k == key).map(|&(_, lo, hi)| (lo, hi)).expect("a known limit")
}

/// A spin so fast that the body has more cyclones per hemisphere than this is refused: the generation time
/// grows with the number of vortices.
pub const MAX_CYCLONES_PER_HEMISPHERE: usize = 32;
