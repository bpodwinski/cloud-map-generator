//! Port of `cloudgen/flow.py`: the large-scale flow that stirs the cloud field, zonal jets plus a population
//! of vortices (decision record in the Python docstring: cyclones turn counterclockwise in the north,
//! anticyclones the reverse, sizes from the Rossby deformation radius, placement in units of the Hadley edge).
//!
//! Precision follows numpy: [`Flow::velocity`], [`Flow::ascent`] and [`Flow::comma_ascent`] run in the dtype
//! of the points they are given (generic over `f32`/`f64`, vortex centers cast to it, the Python floats
//! rounded to it), [`Flow::backtrace`] integrates in `f64` whatever the dtype of its input and returns `f32`,
//! and [`Flow::zonal_rate`] returns `f64` even for an `f32` latitude (numpy 2 subtracts an `np.float64` scalar,
//! which is not weak). The order of the floating-point operations of the Python is kept: sums over the three
//! coordinates are `(x + y) + z`, vortex terms accumulate in vortex order.
//!
//! The vortex population is random. Following `../README.md` rule 3, [`build_flow`] draws a [`FlowPlan`]
//! (every random number the population needs, in the order Python's `build_flow` draws them) with
//! [`crate::rng::Rng`] and hands it to [`build_flow_from_plan`], the pure function that turns draws into
//! vortices; the two are not bit-compatible with numpy's `default_rng(seed ^ 0x5F10)`, so a map built from the
//! same seed is statistically equivalent to Python's, not identical.

use crate::climate::{
    deformation_radius_rad, hadley_edge_deg, has_baroclinic_eddies, lit, np_clip, np_degrees, py_min,
    storm_track_deg, BodyInputs,
};
use crate::rng::Rng;
use crate::sphere::{lonlat_to_dir, normalize, Vec3};
use num_traits::Float;
use std::f64::consts::PI;

/// `Flow.jet` default: longitude displacement of the jet over the advection time, rad.
pub const DEFAULT_JET: f64 = 0.35;
/// `Flow.trades` default.
pub const DEFAULT_TRADES: f64 = 0.05;
/// `Flow.polar_easterlies` default.
pub const DEFAULT_POLAR_EASTERLIES: f64 = 0.25;
/// `Vortex.age` default (a mature comma).
pub const DEFAULT_AGE: f64 = 1.0;
/// `Flow.backtrace` default `time`.
pub const DEFAULT_BACKTRACE_TIME: f64 = 1.0;
/// `Flow.backtrace` default `steps`.
pub const DEFAULT_BACKTRACE_STEPS: usize = 48;

/// Shear of a body with no storm track (`build_flow`): `(jet, trades, polar_easterlies)`.
pub const NO_STORM_TRACK_SHEAR: (f64, f64, f64) = (0.3, 0.1, 0.0);

// ---------------------------------------------------------------------------------------------------------
// Vortex and flow

/// `Vortex.kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VortexKind {
    Cyclone,
    Anticyclone,
    Tropical,
}

/// Port of `Vortex` (frozen dataclass).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vortex {
    pub kind: VortexKind,
    /// Longitude of the center, rad (not wrapped: `build_flow` leaves `lon0 + (i + jitter) 2 pi / m` as is).
    pub lon: f64,
    pub lat: f64,
    /// Angular e-folding radius, rad (chord distance, small angle).
    pub radius: f64,
    /// Signed rotation over the advection time, rad; `> 0` is counterclockwise about `+center`.
    pub omega: f64,
    /// Signed local ascent weight (`> 0` cloudier, `< 0` clearer).
    pub ascent: f64,
    /// Life stage of a mid-latitude low: about 0.3 an open frontal wave, 1 a mature comma, 1.4 occluded.
    pub age: f64,
}

impl Vortex {
    /// `Vortex(kind, lon, lat, radius, omega, ascent)` with the default age of 1.
    pub fn new(kind: VortexKind, lon: f64, lat: f64, radius: f64, omega: f64, ascent: f64) -> Vortex {
        Vortex { kind, lon, lat, radius, omega, ascent, age: DEFAULT_AGE }
    }

    /// `Vortex.center`: the unit direction of the center, `lonlat_to_dir(lon, lat)` in float64.
    pub fn center(&self) -> Vec3<f64> {
        lonlat_to_dir::<f64>(self.lon, self.lat)
    }
}

/// Port of `Flow`: the general circulation that advects the moisture tracer.
#[derive(Clone, Debug, PartialEq)]
pub struct Flow {
    pub hadley_deg: f64,
    pub vortices: Vec<Vortex>,
    /// Longitude displacement over the advection time, rad, as a function of `x = |lat| / phi_H`, relative to
    /// the storm track (zero there): the shear, not the drift.
    pub jet: f64,
    pub trades: f64,
    pub polar_easterlies: f64,
}

fn cross<F: Float>(a: Vec3<F>, b: Vec3<F>) -> Vec3<F> {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// `np.sum((p - c) ** 2, axis=1)` of one point: `(dx^2 + dy^2) + dz^2` in the precision of `F`.
fn sq_dist<F: Float>(p: Vec3<F>, c: Vec3<F>) -> F {
    let (dx, dy, dz) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
    dx * dx + dy * dy + dz * dz
}

fn cast3<F: Float>(v: Vec3<f64>) -> Vec3<F> {
    [lit(v[0]), lit(v[1]), lit(v[2])]
}

impl Flow {
    /// `Flow(hadley_deg)`: no vortices, the default shear.
    pub fn new(hadley_deg: f64) -> Flow {
        Flow {
            hadley_deg,
            vortices: Vec::new(),
            jet: DEFAULT_JET,
            trades: DEFAULT_TRADES,
            polar_easterlies: DEFAULT_POLAR_EASTERLIES,
        }
    }

    /// The zonal profile `prof(x)` of the Python, in the precision of `x`: jet minus trades minus the polar
    /// easterlies as functions of `x = |lat| / phi_H`.
    fn prof<F: Float>(&self, x: F) -> F {
        let g = |mu: f64, s: f64| {
            let d = (x - lit::<F>(mu)) / lit::<F>(s);
            (-(d * d)).exp()
        };
        let sm = np_clip((x - lit::<F>(2.0)) / lit::<F>(0.8), F::zero(), F::one());
        let sm = sm * sm * (lit::<F>(3.0) - lit::<F>(2.0) * sm);
        lit::<F>(self.jet) * g(1.5, 0.4) - lit::<F>(self.trades) * g(0.4, 0.4) - lit::<F>(self.polar_easterlies) * sm
    }

    /// `Flow.zonal_rate`: longitude displacement over the advection time at latitude `lat` (radians),
    /// relative to the storm track (zero at `x = 1.7`). `|lat|` makes it symmetric about the equator and it
    /// accepts latitudes beyond pi/2. For an `f32` latitude the profile runs in `f32` and the constant
    /// `prof(1.7)` is subtracted in `f64` (numpy 2: a `np.float64` scalar is not weak), so the result is `f64`
    /// holding `f32`-rounded values. The degrees are numpy's (`np.degrees`: [`np_degrees`], which for `f32`
    /// multiplies by `180f / pi_f32`, not by Rust's `f32::to_degrees` constant).
    pub fn zonal_rate<F: Float>(&self, lat: F) -> f64 {
        let x = np_degrees(lat).abs() / lit::<F>(self.hadley_deg);
        self.prof(x).to_f64().expect("float conversion") - self.prof(1.7_f64)
    }

    /// `Flow.velocity`: tangent velocity (rad per unit advection time) at the points `p`, in the precision of
    /// `p`. The latitude is `asin(clip(p_z, -1, 1))`, so a non-unit point with `|z| > 1` acts as a pole; the
    /// vortex distance and the cross product use the raw, unnormalized point.
    pub fn velocity<F: Float + Send + Sync>(&self, p: &[Vec3<F>]) -> Vec<Vec3<F>> {
        crate::par::map_chunks(p, crate::par::BLOCK, |block| self.velocity_serial(block))
    }

    /// [`Flow::velocity`] of one block of points on the calling thread (a point does not depend on the others).
    fn velocity_serial<F: Float>(&self, p: &[Vec3<F>]) -> Vec<Vec3<F>> {
        let z = [F::zero(), F::zero(), F::one()];
        let vortices: Vec<(Vec3<F>, F, F)> =
            self.vortices.iter().map(|vx| (cast3(vx.center()), lit(vx.omega), lit(vx.radius * vx.radius))).collect();
        p.iter()
            .map(|&pt| {
                let lat = np_clip(pt[2], -F::one(), F::one()).asin();
                let zr: F = lit(self.zonal_rate(lat));
                let cz = cross(z, pt);
                let mut v = [zr * cz[0], zr * cz[1], zr * cz[2]];
                for &(c, omega, rr) in &vortices {
                    let d2 = sq_dist(pt, c);
                    let w = omega * (-d2 / rr).exp();
                    let cc = cross(c, pt);
                    for k in 0..3 {
                        v[k] = v[k] + w * cc[k];
                    }
                }
                v
            })
            .collect()
    }

    /// `Flow.backtrace`: where the air at `p` was `time` ago (midpoint rule on the sphere, `steps` steps).
    /// The integration is `f64` whatever the dtype of `p`; only the returned directions are rounded to `f32`.
    /// `time = 0` only renormalizes; a negative `time` advects forward. Python raises ZeroDivisionError for
    /// `steps = 0`; here it panics.
    pub fn backtrace<F: Float + Send + Sync>(&self, p: &[Vec3<F>], time: f64, steps: usize) -> Vec<Vec3<f32>> {
        assert!(steps > 0, "backtrace needs at least one step (Python: ZeroDivisionError)");
        // Every point is integrated on its own, so blocks of points run in parallel and give the serial result.
        crate::par::map_chunks(p, crate::par::BLOCK, |block| self.backtrace_serial(block, time, steps))
    }

    /// [`Flow::backtrace`] of one block of points on the calling thread.
    fn backtrace_serial<F: Float>(&self, p: &[Vec3<F>], time: f64, steps: usize) -> Vec<Vec3<f32>> {
        let dt = time / steps as f64;
        let half = 0.5 * dt;
        let mut q: Vec<Vec3<f64>> = p
            .iter()
            .map(|v| {
                [
                    v[0].to_f64().expect("float conversion"),
                    v[1].to_f64().expect("float conversion"),
                    v[2].to_f64().expect("float conversion"),
                ]
            })
            .collect();
        for _ in 0..steps {
            let v = self.velocity_serial(&q);
            let mid: Vec<Vec3<f64>> = q
                .iter()
                .zip(&v)
                .map(|(q, v)| normalize([q[0] - half * v[0], q[1] - half * v[1], q[2] - half * v[2]]))
                .collect();
            let vm = self.velocity_serial(&mid);
            q = q
                .iter()
                .zip(&vm)
                .map(|(q, v)| normalize([q[0] - dt * v[0], q[1] - dt * v[1], q[2] - dt * v[2]]))
                .collect();
        }
        q.iter().map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]).collect()
    }

    /// `Flow.backtrace(p)` with the defaults (`time = 1`, 48 steps).
    pub fn backtrace_default<F: Float + Send + Sync>(&self, p: &[Vec3<F>]) -> Vec<Vec3<f32>> {
        self.backtrace(p, DEFAULT_BACKTRACE_TIME, DEFAULT_BACKTRACE_STEPS)
    }

    /// `Flow.ascent`: local ascent (`> 0`) or subsidence (`< 0`) at the points `p`, from the vortices, as
    /// `f32`. Each vortex term is computed in the precision of `p`, cast to `f32` and accumulated in `f32` in
    /// vortex order (`a += np.float32(ascent) * g.astype(float32)`). A cyclone takes the comma
    /// ([`Flow::comma_ascent`]); a tropical storm subtracts a clear eye of radius `0.12 radius` (the VORTEX
    /// radius, not the `0.6 radius` of its ring) from its dense ring.
    pub fn ascent<F: Float + Send + Sync>(&self, p: &[Vec3<F>]) -> Vec<f32> {
        crate::par::map_chunks(p, crate::par::BLOCK, |block| self.ascent_serial(block))
    }

    /// [`Flow::ascent`] of one block of points on the calling thread (a point does not depend on the others).
    fn ascent_serial<F: Float>(&self, p: &[Vec3<F>]) -> Vec<f32> {
        let mut a = vec![0.0_f32; p.len()];
        for vx in &self.vortices {
            let c: Vec3<F> = cast3(vx.center());
            let r = vx.radius * if vx.kind == VortexKind::Tropical { 0.6 } else { 1.3 };
            let rr: F = lit(r * r);
            let eye = 0.12 * vx.radius;
            let eye2: F = lit(eye * eye);
            let weight = vx.ascent as f32;
            let comma = if vx.kind == VortexKind::Cyclone { Some(Flow::comma_ascent(vx, p)) } else { None };
            for (i, &pt) in p.iter().enumerate() {
                let g = match vx.kind {
                    // The plain Gaussian Python computes first is overwritten by the comma for a cyclone.
                    VortexKind::Cyclone => comma.as_ref().expect("comma ascent")[i],
                    VortexKind::Anticyclone => (-sq_dist(pt, c) / rr).exp(),
                    VortexKind::Tropical => {
                        let d2 = sq_dist(pt, c);
                        (-d2 / rr).exp() - lit::<F>(1.6) * (-d2 / eye2).exp()
                    }
                };
                a[i] += weight * g.to_f32().expect("float conversion");
            }
        }
        a
    }

    /// `Flow._comma_ascent`: the cloud-producing ascent of a mid-latitude low by life stage, in the precision
    /// of `p`. A young wave lifts air along its warm front, east of the center; a mature low's comma head sits
    /// poleward of it; an occluded low (`age > 1`) dries out at its center (the dry slot) and keeps a ring.
    /// The east vector falls back to `(0, 1, 0)` when `|cross(z, c)| < 1e-9` (a vortex on a pole), and the
    /// poleward shift takes the SIGN BIT of the latitude (`copysign(1, lat)`: `+0.0` is north, `-0.0` south).
    pub fn comma_ascent<F: Float>(vx: &Vortex, p: &[Vec3<F>]) -> Vec<F> {
        let c = vx.center();
        let east = cross([0.0, 0.0, 1.0], c);
        let n = (east[0] * east[0] + east[1] * east[1] + east[2] * east[2]).sqrt();
        let east = if n < 1e-9 { [0.0, 1.0, 0.0] } else { [east[0] / n, east[1] / n, east[2] / n] };
        let sign = 1.0_f64.copysign(vx.lat);
        let pole = cross(c, east).map(|v| v * sign);
        let grown = py_min(vx.age, 1.0);
        let (e_w, p_w) = (0.8 - 0.5 * grown, 0.5);
        let shift: Vec3<f64> = [0, 1, 2].map(|k| east[k] * e_w + pole[k] * p_w * grown);
        let head: Vec3<F> = cast3(normalize([0, 1, 2].map(|k| c[k] + vx.radius * shift[k])));
        let r = vx.radius * (1.0 + 0.3 * vx.age);
        let rr: F = lit(r * r);
        let dry_slot = vx.age > 1.0;
        let core = 0.45 * vx.radius;
        let core2: F = lit(core * core);
        let dry_weight: F = lit((vx.age - 1.0) * 2.0);
        let c_f: Vec3<F> = cast3(c);
        p.iter()
            .map(|&pt| {
                let g = (-sq_dist(pt, head) / rr).exp();
                if dry_slot {
                    g - dry_weight * (-sq_dist(pt, c_f) / core2).exp()
                } else {
                    g
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------------------------------------
// The vortex population

/// The random numbers one low and the high that follows it consume, in the order `build_flow` draws them.
/// Every field is the value of the `rng.uniform(lo, hi)` call of the same name in the Python (so it lies in
/// the range noted).
#[derive(Clone, Debug, PartialEq)]
pub struct LowDraws {
    /// Longitude jitter of the low, in units of the spacing, `[-0.3, 0.3]`.
    pub lon_jitter: f64,
    /// Latitude offset from the storm track, in units of the Hadley edge, `[-0.25, 0.25]`.
    pub lat_offset: f64,
    /// Life stage, `[0.25, 1.45]`.
    pub age: f64,
    /// Radius factor of the low (times `L_R (0.7 + 0.3 age)`), `[0.8, 1.4]`.
    pub radius_factor: f64,
    /// Winding factor of the low (times `hemi 3 age`), `[0.8, 1.2]`.
    pub omega_factor: f64,
    /// Longitude of the high past its low, in units of half the spacing, `[0.7, 1.3]`.
    pub high_lon_factor: f64,
    /// Latitude of the high, in units of the Hadley edge, `[0.9, 1.3]`.
    pub high_lat_factor: f64,
    /// Radius of the high, in units of `L_R`, `[1.3, 2.0]`.
    pub high_radius_factor: f64,
    /// Winding factor of the high (times `-hemi 1.2`), `[0.7, 1.3]`.
    pub high_omega_factor: f64,
}

/// The random numbers of one tropical cyclone, in draw order (latitude first, then the arguments of the
/// `Vortex` call).
#[derive(Clone, Debug, PartialEq)]
pub struct TropicalDraws {
    /// Latitude in units of the Hadley edge, `[0.35, 0.65]`.
    pub lat_factor: f64,
    /// Longitude, rad, `[-pi, pi]`.
    pub lon: f64,
    /// Radius factor (times `0.35 L_R`), `[0.8, 1.2]`.
    pub radius_factor: f64,
    /// Winding factor (times `hemi 12`), `[0.8, 1.2]`.
    pub omega_factor: f64,
}

/// The random numbers of one hemisphere: the first longitude, `m` lows with their highs, then the tropical
/// cyclones (a Poisson(1.5) count of them).
#[derive(Clone, Debug, PartialEq)]
pub struct HemispherePlan {
    /// Longitude of the first low before its jitter, rad, `[-pi, pi]`.
    pub lon0: f64,
    pub lows: Vec<LowDraws>,
    pub tropicals: Vec<TropicalDraws>,
}

/// Every random number `build_flow` consumes for a body with a storm track: the northern hemisphere first,
/// then the southern.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowPlan {
    pub north: HemispherePlan,
    pub south: HemispherePlan,
}

impl FlowPlan {
    /// Draws a plan of `wave_count` lows per hemisphere, in the order of Python's `build_flow`: per hemisphere
    /// `lon0`; per low the jitter, latitude offset, age, radius factor, winding factor, then the high's
    /// longitude factor, latitude, radius and winding; then the Poisson(1.5) tropical count and per tropical
    /// storm its latitude, longitude, radius and winding.
    pub fn draw(wave_count: usize, rng: &mut Rng) -> FlowPlan {
        let hemisphere = |rng: &mut Rng| {
            let lon0 = rng.uniform_range(-PI, PI);
            let lows = (0..wave_count)
                .map(|_| {
                    let lon_jitter = rng.uniform_range(-0.3, 0.3);
                    let lat_offset = rng.uniform_range(-0.25, 0.25);
                    let age = rng.uniform_range(0.25, 1.45);
                    let radius_factor = rng.uniform_range(0.8, 1.4);
                    let omega_factor = rng.uniform_range(0.8, 1.2);
                    let high_lon_factor = rng.uniform_range(0.7, 1.3);
                    let high_lat_factor = rng.uniform_range(0.9, 1.3);
                    let high_radius_factor = rng.uniform_range(1.3, 2.0);
                    let high_omega_factor = rng.uniform_range(0.7, 1.3);
                    LowDraws {
                        lon_jitter,
                        lat_offset,
                        age,
                        radius_factor,
                        omega_factor,
                        high_lon_factor,
                        high_lat_factor,
                        high_radius_factor,
                        high_omega_factor,
                    }
                })
                .collect();
            let n_tropical = rng.poisson(1.5);
            let tropicals = (0..n_tropical)
                .map(|_| {
                    let lat_factor = rng.uniform_range(0.35, 0.65);
                    let lon = rng.uniform_range(-PI, PI);
                    let radius_factor = rng.uniform_range(0.8, 1.2);
                    let omega_factor = rng.uniform_range(0.8, 1.2);
                    TropicalDraws { lat_factor, lon, radius_factor, omega_factor }
                })
                .collect();
            HemispherePlan { lon0, lows, tropicals }
        };
        let north = hemisphere(rng);
        let south = hemisphere(rng);
        FlowPlan { north, south }
    }
}

/// `x = 2 pi cos(track) / (5 L_R)` of `build_flow`: the zonal wavenumber of the storm track (circumference of
/// the track over a baroclinic wavelength of 5 deformation radii); `None` for a body without a storm track.
/// Python raises (ZeroDivisionError, OverflowError) where the deformation radius is 0 or NaN, which is not a
/// legal input of `build_flow`; here it panics.
pub fn wavenumber(body: &BodyInputs) -> Option<f64> {
    if !has_baroclinic_eddies(body) {
        return None;
    }
    let lr = deformation_radius_rad(body);
    let track = storm_track_deg(body).to_radians();
    let wavelength = 5.0 * lr;
    let x = 2.0 * PI * track.cos() / wavelength;
    assert!(x.is_finite(), "build_flow needs a finite wavenumber, got {x} (deformation radius {lr})");
    Some(x)
}

/// `m = max(2, round(x))` of `build_flow` (Python's `round` is half-to-even): the number of lows (and of
/// highs) per hemisphere, `None` for a body without a storm track.
pub fn wave_count(body: &BodyInputs) -> Option<usize> {
    wavenumber(body).map(waves_from_wavenumber)
}

/// `max(2, round(x))` with Python's round-half-to-even: the number of lows per hemisphere of the zonal
/// wavenumber `x` (a negative `x` gives 2; [`wavenumber`] never returns a NaN or an infinity).
pub fn waves_from_wavenumber(x: f64) -> usize {
    x.round_ties_even().max(2.0) as usize
}

/// `build_flow(body, seed)`: the vortex population of one body and seed. Draws its [`FlowPlan`] from
/// `Rng::new(seed ^ 0x5F10)` (Python: `default_rng(seed ^ 0x5F10)`) and builds the flow from it.
pub fn build_flow(body: &BodyInputs, seed: u64) -> Flow {
    let mut rng = Rng::new(seed ^ 0x5F10);
    match wave_count(body) {
        Some(m) => build_flow_from_plan(body, &FlowPlan::draw(m, &mut rng)),
        None => build_flow_from_plan(body, &FlowPlan { north: empty_hemisphere(), south: empty_hemisphere() }),
    }
}

fn empty_hemisphere() -> HemispherePlan {
    HemispherePlan { lon0: 0.0, lows: Vec::new(), tropicals: Vec::new() }
}

/// The deterministic half of `build_flow`: the vortices of a body from the random numbers of `plan`.
///
/// A body without a storm track (no baroclinic eddies: the global Hadley cell of Venus or Titan) gets no
/// vortex and the weak shear [`NO_STORM_TRACK_SHEAR`] whatever the plan. Otherwise the plan must hold
/// [`wave_count`] lows per hemisphere. Per hemisphere, in order: the lows with their highs (`hemi = lat_sign
/// spin` flips every winding for a retrograde spin but not the latitudes), then the tropical cyclones.
pub fn build_flow_from_plan(body: &BodyInputs, plan: &FlowPlan) -> Flow {
    let phi_h = hadley_edge_deg(body);
    let mut flow = Flow::new(phi_h);
    let Some(m) = wave_count(body) else {
        // Global Hadley cell: no storm track, the cloud field stays zonal; keep only weak shear.
        (flow.jet, flow.trades, flow.polar_easterlies) = NO_STORM_TRACK_SHEAR;
        return flow;
    };
    let lr = deformation_radius_rad(body);
    let track = storm_track_deg(body).to_radians();
    // f = 2 Omega sin(lat): a retrograde spin (negative period) flips the sign of f in each hemisphere.
    // A body with a storm track has a finite, non-zero period (no spin gives an infinite raw edge).
    let spin = 1.0_f64.copysign(body.rotation_period_days.expect("a body with a storm track has a rotation period"));
    let mf = m as f64;
    for (lat_sign, hemi_plan) in [(1.0_f64, &plan.north), (-1.0_f64, &plan.south)] {
        let hemi = lat_sign * spin;
        assert_eq!(hemi_plan.lows.len(), m, "the plan must hold {m} lows per hemisphere");
        for (i, d) in hemi_plan.lows.iter().enumerate() {
            let lon = hemi_plan.lon0 + (i as f64 + d.lon_jitter) * 2.0 * PI / mf;
            let lat = lat_sign * (track + (d.lat_offset * phi_h).to_radians());
            let age = d.age;
            flow.vortices.push(Vortex {
                kind: VortexKind::Cyclone,
                lon,
                lat,
                radius: lr * d.radius_factor * (0.7 + 0.3 * age),
                omega: hemi * 3.0 * age * d.omega_factor,
                ascent: 1.0,
                age,
            });
            // An anticyclone between consecutive lows, equatorward of the track.
            let lon_a = lon + PI / mf * d.high_lon_factor;
            let lat_a = lat_sign * (d.high_lat_factor * phi_h).to_radians();
            flow.vortices.push(Vortex {
                kind: VortexKind::Anticyclone,
                lon: lon_a,
                lat: lat_a,
                radius: lr * d.high_radius_factor,
                omega: -hemi * 1.2 * d.high_omega_factor,
                ascent: -0.45,
                age: DEFAULT_AGE,
            });
        }
        // Tropical cyclones: a handful, compact and tightly wound.
        for t in &hemi_plan.tropicals {
            let lat = lat_sign * (t.lat_factor * phi_h).to_radians();
            flow.vortices.push(Vortex {
                kind: VortexKind::Tropical,
                lon: t.lon,
                lat,
                radius: 0.35 * lr * t.radius_factor,
                omega: hemi * 12.0 * t.omega_factor,
                ascent: 1.6,
                age: DEFAULT_AGE,
            });
        }
    }
    flow
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::climate::EARTH;

    fn earth_like() -> Flow {
        let mut f = Flow::new(30.0);
        f.vortices.push(Vortex {
            kind: VortexKind::Cyclone,
            lon: 0.35,
            lat: 0.87,
            radius: 0.2,
            omega: 3.0,
            ascent: 1.0,
            age: 0.8,
        });
        f.vortices.push(Vortex::new(VortexKind::Anticyclone, -1.0, 0.5, 0.3, -1.2, -0.45));
        f.vortices.push(Vortex::new(VortexKind::Tropical, 2.0, -0.25, 0.06, -12.0, 1.6));
        f
    }

    fn some_points() -> Vec<Vec3<f64>> {
        let mut v = Vec::new();
        for lat in [-1.5, -0.9, -0.3, 0.0, 0.4, 0.87, 1.2, 1.55] {
            for lon in [-3.1, -1.0, 0.35, 2.0, 3.0] {
                v.push(lonlat_to_dir::<f64>(lon, lat));
            }
        }
        v
    }

    #[test]
    fn defaults_and_center() {
        let f = Flow::new(30.0);
        assert_eq!((f.jet, f.trades, f.polar_easterlies), (0.35, 0.05, 0.25));
        assert!(f.vortices.is_empty());
        let v = Vortex::new(VortexKind::Cyclone, 0.0, 0.0, 0.1, 1.0, 1.0);
        assert_eq!(v.age, 1.0);
        assert_eq!(v.center(), [1.0, 0.0, 0.0]);
        let north = Vortex::new(VortexKind::Cyclone, 1.0, std::f64::consts::FRAC_PI_2, 0.1, 1.0, 1.0).center();
        assert!((north[2] - 1.0).abs() < 1e-15 && north[0].abs() < 1e-15);
    }

    #[test]
    fn zonal_rate_is_zero_at_the_storm_track_symmetric_and_widens_for_f32() {
        let f = Flow::new(30.0);
        assert!(f.zonal_rate((51.0_f64).to_radians()).abs() < 1e-12);
        assert_eq!(f.zonal_rate(0.4_f64), f.zonal_rate(-0.4_f64));
        let a = f.zonal_rate(0.4_f32);
        assert!((a - f.zonal_rate(0.4_f64)).abs() < 1e-6);
        // Strong jet at x = 1.5 (lat 45), weak easterlies at the pole.
        assert!(f.zonal_rate((45.0_f64).to_radians()) > 0.0);
        assert!(f.zonal_rate(std::f64::consts::FRAC_PI_2) < 0.0);
        let still = Flow { jet: 0.0, trades: 0.0, polar_easterlies: 0.0, ..Flow::new(30.0) };
        assert_eq!(still.zonal_rate(0.7_f64), 0.0);
    }

    #[test]
    fn velocity_of_the_zonal_flow_is_east_west_and_zero_on_the_axis() {
        let f = Flow::new(30.0);
        let p = [lonlat_to_dir::<f64>(0.0, 0.0), lonlat_to_dir::<f64>(1.0, 0.0), [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
        let v = f.velocity(&p);
        // cross(z, p) = (-p_y, p_x, 0) times the rate: purely zonal.
        let rate0 = f.zonal_rate(0.0_f64);
        assert!((v[0][1] - rate0).abs() < 1e-15 && v[0][0].abs() < 1e-15 && v[0][2] == 0.0);
        assert_eq!(v[2], [0.0, 0.0, 0.0]);
        assert_eq!(v[3], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_vortex_turns_counterclockwise_for_a_positive_omega_and_vanishes_at_its_center() {
        let mut f = Flow { jet: 0.0, trades: 0.0, polar_easterlies: 0.0, ..Flow::new(30.0) };
        f.vortices.push(Vortex::new(VortexKind::Cyclone, 0.0, 0.0, 0.3, 2.0, 1.0));
        let at = |lon: f64, lat: f64| f.velocity(&[lonlat_to_dir::<f64>(lon, lat)])[0];
        assert!(at(0.0, 0.0).iter().all(|c| c.abs() < 1e-15));
        // North of the center the air moves west (counterclockwise seen from above), south of it east.
        assert!(at(0.0, 0.1)[1] < 0.0);
        assert!(at(0.0, -0.1)[1] > 0.0);
        // East of the center it moves north.
        assert!(at(0.1, 0.0)[2] > 0.0);
    }

    #[test]
    fn f32_velocity_tracks_f64() {
        let f = earth_like();
        let p64 = some_points();
        let p32: Vec<Vec3<f32>> = p64.iter().map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]).collect();
        let (a, b) = (f.velocity(&p64), f.velocity(&p32));
        for (x, y) in a.iter().zip(&b) {
            for k in 0..3 {
                assert!((x[k] - f64::from(y[k])).abs() < 2e-4, "{x:?} {y:?}");
            }
        }
    }

    #[test]
    fn backtrace_renormalizes_and_a_zero_time_is_the_identity_on_unit_vectors() {
        let f = earth_like();
        let p: Vec<Vec3<f32>> = some_points().iter().map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]).collect();
        let q = f.backtrace(&p, 0.0, 3);
        for (a, b) in p.iter().zip(&q) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 2e-7);
            }
        }
        for v in f.backtrace_default(&p) {
            let n = f64::from(v[0]).powi(2) + f64::from(v[1]).powi(2) + f64::from(v[2]).powi(2);
            assert!((n - 1.0).abs() < 1e-6);
        }
        // A non-unit input comes back normalized.
        let q = f.backtrace(&[[0.0_f64, 0.0, 3.0]], 0.0, 1);
        assert_eq!(q[0], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn backtrace_forward_then_back_is_nearly_the_identity() {
        let f = earth_like();
        let p = some_points();
        let there = f.backtrace(&p, -1.0, 96);
        let back = f.backtrace(&there, 1.0, 96);
        for (a, b) in p.iter().zip(&back) {
            for k in 0..3 {
                assert!((a[k] - f64::from(b[k])).abs() < 5e-3, "{a:?} {b:?}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "at least one step")]
    fn backtrace_needs_a_step() {
        Flow::new(30.0).backtrace(&[[1.0_f64, 0.0, 0.0]], 1.0, 0);
    }

    #[test]
    fn ascent_signs_follow_the_vortex_kinds() {
        let mut f = Flow::new(30.0);
        f.vortices.push(Vortex::new(VortexKind::Anticyclone, 0.0, 0.0, 0.3, -1.2, -0.45));
        let c = [lonlat_to_dir::<f64>(0.0, 0.0)];
        let a = f.ascent(&c);
        assert!((f64::from(a[0]) + 0.45).abs() < 1e-6); // exp(0) = 1 at the center
        assert_eq!(Flow::new(30.0).ascent(&c), vec![0.0]);
        // A tropical storm is dense on its ring and clear in its eye: the center is 1 - 1.6 = -0.6 times the weight.
        let mut t = Flow::new(30.0);
        t.vortices.push(Vortex::new(VortexKind::Tropical, 0.0, 0.0, 0.06, 12.0, 1.6));
        let at = |lon: f64| t.ascent(&[lonlat_to_dir::<f64>(lon, 0.0)])[0];
        assert!((f64::from(at(0.0)) - 1.6 * (1.0 - 1.6)).abs() < 1e-6);
        assert!(at(0.036) > 0.5); // the ring
    }

    #[test]
    fn ascent_accumulates_per_vortex_in_f32() {
        let f = earth_like();
        let p = some_points();
        let a = f.ascent(&p);
        let mut want = vec![0.0_f32; p.len()];
        for vx in &f.vortices {
            let one = Flow { vortices: vec![*vx], ..Flow::new(30.0) }.ascent(&p);
            for (w, o) in want.iter_mut().zip(one) {
                *w += o;
            }
        }
        assert_eq!(a, want);
    }

    #[test]
    fn comma_head_moves_from_east_to_poleward_and_the_dry_slot_only_past_age_one() {
        let cyc = |lat: f64, age: f64| Vortex {
            kind: VortexKind::Cyclone,
            lon: 0.0,
            lat,
            radius: 0.2,
            omega: 3.0,
            ascent: 1.0,
            age,
        };
        let center = [lonlat_to_dir::<f64>(0.0, 0.5)];
        let at = |v: &Vortex, lon: f64, lat: f64| Flow::comma_ascent(v, &[lonlat_to_dir::<f64>(lon, lat)])[0];
        // Young wave (age 0): the head sits 0.8 radius east of the center.
        let young = cyc(0.5, 0.0);
        assert!(at(&young, 0.16, 0.5) > at(&young, -0.16, 0.5));
        // Mature (age 1): 0.3 east and 0.5 poleward: north of the center beats south of it.
        let mature = cyc(0.5, 1.0);
        assert!(at(&mature, 0.06, 0.6) > at(&mature, 0.06, 0.4));
        // Southern low: the poleward shift flips.
        let south = cyc(-0.5, 1.0);
        assert!(at(&south, 0.06, -0.6) > at(&south, 0.06, -0.4));
        // The sign bit decides the hemisphere of an equatorial low.
        let (pz, nz) = (cyc(0.0, 1.0), cyc(-0.0, 1.0));
        assert!(at(&pz, 0.06, 0.1) > at(&pz, 0.06, -0.1));
        assert!(at(&nz, 0.06, -0.1) > at(&nz, 0.06, 0.1));
        // The dry slot: nothing at age 1, a clear center past it.
        let occluded = cyc(0.5, 1.4);
        let plain = Flow::comma_ascent(&mature, &center)[0];
        assert!(plain > 0.0);
        assert!(Flow::comma_ascent(&occluded, &center)[0] < plain);
    }

    #[test]
    fn comma_east_vector_falls_back_on_a_pole() {
        // On the pole cross(z, c) = 0: the east vector is (0, 1, 0) and the result is finite.
        let v = Vortex {
            kind: VortexKind::Cyclone,
            lon: 0.3,
            lat: std::f64::consts::FRAC_PI_2,
            radius: 0.2,
            omega: 3.0,
            ascent: 1.0,
            age: 0.6,
        };
        let g = Flow::comma_ascent(&v, &[[0.0_f64, 0.0, 1.0], [0.0, 0.1, 0.99]]);
        assert!(g.iter().all(|x| x.is_finite() && *x > 0.0));
        // 1e-9 is the threshold: 0.5e-9 falls back, 2e-9 does not (the heads differ).
        let near = |colat: f64| Vortex { lat: std::f64::consts::FRAC_PI_2 - colat, lon: 1.0, ..v };
        let p = [lonlat_to_dir::<f64>(1.0, 1.5)];
        let (inside, outside) = (Flow::comma_ascent(&near(5e-10), &p)[0], Flow::comma_ascent(&near(2e-9), &p)[0]);
        assert!((inside - outside).abs() > 1e-6, "{inside} {outside}");
    }

    fn body(p: f64, r: f64) -> BodyInputs {
        BodyInputs::new(Some(p), r, 9.81, 8.0)
    }

    #[test]
    fn wave_count_and_wavenumber_for_earth_and_for_bodies_without_a_storm_track() {
        let x = wavenumber(&EARTH).expect("Earth has a storm track");
        // 2 pi cos(51 deg) / (5 * 0.3 radians(30) sin(51)/sin(51)) = 2 pi cos(51 deg) / (1.5 radians(30)).
        let want = 2.0 * PI * 51.0_f64.to_radians().cos() / (1.5 * 30.0_f64.to_radians());
        assert!((x - want).abs() < 1e-12, "{x} {want}");
        assert_eq!(wave_count(&EARTH), Some(5));
        let venus = body(-243.0185, 6051.8);
        assert_eq!((wavenumber(&venus), wave_count(&venus)), (None, None));
        assert_eq!(wave_count(&BodyInputs::new(None, 6371.0, 9.81, 8.0)), None);
        // The floor of 2 (a body just under the threshold has a very long wavelength).
        assert_eq!(wave_count(&body(1.6621166666666665, 6371.0)), Some(2));
    }

    #[test]
    fn the_wave_count_rounds_half_to_even_with_a_floor_of_two() {
        let m = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.49, 6.51, 20.5].map(waves_from_wavenumber);
        assert_eq!(m, [2, 2, 2, 4, 4, 6, 6, 7, 20]);
        assert_eq!(waves_from_wavenumber(-3.0), 2);
    }

    #[test]
    fn an_f32_zonal_rate_subtracts_an_f64_constant_from_a_profile_at_numpys_degrees() {
        // (latitude bits, `np.degrees` of it as float32 bits), numpy 2.5.4: Rust's `f32::to_degrees` is one ulp
        // above numpy's on each of these (see `np_degrees_is_numpys_not_rusts_for_f32` in climate.rs).
        let pins: [(u32, u32); 8] = [
            (1067571966, 1116792456),
            (1067610678, 1116827113),
            (1055251176, 1104004840),
            (1034780317, 1083920903),
            (1058679146, 1107952454),
            (1059340735, 1108544739),
            (1032955487, 1082287230),
            (1070130546, 1119083016),
        ];
        let f = Flow::new(30.0);
        let mut rust_degrees_would_differ = 0;
        for (lat_bits, deg_bits) in pins {
            let lat = f32::from_bits(lat_bits);
            let x = f32::from_bits(deg_bits).abs() / 30.0_f32;
            let want = f64::from(f.prof(x)) - f.prof(1.7_f64);
            assert_eq!(f.zonal_rate(lat), want, "{lat}");
            let with_rust_degrees = f64::from(f.prof(lat.to_degrees().abs() / 30.0_f32)) - f.prof(1.7_f64);
            rust_degrees_would_differ += usize::from(with_rust_degrees != want);
        }
        // The pins discriminate: the profile is not flat to the ulp of its argument.
        assert!(rust_degrees_would_differ >= 2, "{rust_degrees_would_differ}");
        for lat in [0.1_f32, 0.4, 0.8, 1.2, 1.5] {
            let x = np_degrees(lat).abs() / 30.0_f32;
            assert_eq!(f.zonal_rate(lat), f64::from(f.prof(x)) - f.prof(1.7_f64), "{lat}");
        }
    }

    #[test]
    fn a_body_without_a_storm_track_gets_the_weak_shear_and_no_vortex() {
        let f = build_flow(&body(-243.0185, 6051.8), 7);
        assert!(f.vortices.is_empty());
        assert_eq!((f.hadley_deg, f.jet, f.trades, f.polar_easterlies), (90.0, 0.3, 0.1, 0.0));
        let no_spin = build_flow(&BodyInputs::new(None, 6371.0, 9.81, 8.0), 1);
        assert!(no_spin.vortices.is_empty());
    }

    #[test]
    fn the_plan_is_drawn_in_pythons_order() {
        // Replay the draws by hand from a second generator of the same seed.
        let mut a = Rng::new(99);
        let plan = FlowPlan::draw(2, &mut a);
        let mut b = Rng::new(99);
        let mut u = |lo: f64, hi: f64| b.uniform_range(lo, hi);
        assert_eq!(plan.north.lon0, u(-PI, PI));
        for low in &plan.north.lows {
            let want = [
                u(-0.3, 0.3),
                u(-0.25, 0.25),
                u(0.25, 1.45),
                u(0.8, 1.4),
                u(0.8, 1.2),
                u(0.7, 1.3),
                u(0.9, 1.3),
                u(1.3, 2.0),
                u(0.7, 1.3),
            ];
            let got = [
                low.lon_jitter,
                low.lat_offset,
                low.age,
                low.radius_factor,
                low.omega_factor,
                low.high_lon_factor,
                low.high_lat_factor,
                low.high_radius_factor,
                low.high_omega_factor,
            ];
            assert_eq!(got, want);
        }
        // The Poisson count comes before the tropical draws, and the southern hemisphere follows the northern.
        let mut replay = Rng::new(99);
        for _ in 0..1 + 2 * 9 {
            replay.uniform();
        }
        let n = replay.poisson(1.5) as usize;
        assert_eq!(plan.north.tropicals.len(), n);
        for t in &plan.north.tropicals {
            let want = [
                replay.uniform_range(0.35, 0.65),
                replay.uniform_range(-PI, PI),
                replay.uniform_range(0.8, 1.2),
                replay.uniform_range(0.8, 1.2),
            ];
            assert_eq!([t.lat_factor, t.lon, t.radius_factor, t.omega_factor], want);
        }
        assert_eq!(plan.south.lon0, replay.uniform_range(-PI, PI));
    }

    /// The statistics of [`FlowPlan::draw`] against the `rng.uniform(lo, hi)` and `rng.poisson(1.5)` calls of
    /// the Python `build_flow` (the ranges below are copied from that source, NOT from the port): every field
    /// lies in its range, reaches both ends of it (a missed end of 0.5 per cent of the width has a probability
    /// below 1e-80 over this many draws), has the mean and the variance of a uniform draw, and the tropical
    /// count is Poisson(1.5). The replay test above pins the ORDER of the draws; this one pins what each draw is.
    #[test]
    fn the_plan_fields_are_uniform_over_the_ranges_of_the_python_draws() {
        // (field, lo, hi), in the draw order of `build_flow`.
        let ranges: [(&str, f64, f64); 14] = [
            ("lon0", -PI, PI),
            ("low.lon_jitter", -0.3, 0.3),
            ("low.lat_offset", -0.25, 0.25),
            ("low.age", 0.25, 1.45),
            ("low.radius_factor", 0.8, 1.4),
            ("low.omega_factor", 0.8, 1.2),
            ("high.lon_factor", 0.7, 1.3),
            ("high.lat_factor", 0.9, 1.3),
            ("high.radius_factor", 1.3, 2.0),
            ("high.omega_factor", 0.7, 1.3),
            ("tropical.lat_factor", 0.35, 0.65),
            ("tropical.lon", -PI, PI),
            ("tropical.radius_factor", 0.8, 1.2),
            ("tropical.omega_factor", 0.8, 1.2),
        ];
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); ranges.len()];
        let mut tropical_counts: Vec<usize> = Vec::new();
        let mut rng = Rng::new(2026);
        for _ in 0..20_000 {
            let plan = FlowPlan::draw(3, &mut rng);
            for h in [&plan.north, &plan.south] {
                samples[0].push(h.lon0);
                for l in &h.lows {
                    let row = [
                        l.lon_jitter,
                        l.lat_offset,
                        l.age,
                        l.radius_factor,
                        l.omega_factor,
                        l.high_lon_factor,
                        l.high_lat_factor,
                        l.high_radius_factor,
                        l.high_omega_factor,
                    ];
                    for (k, v) in row.into_iter().enumerate() {
                        samples[1 + k].push(v);
                    }
                }
                tropical_counts.push(h.tropicals.len());
                for t in &h.tropicals {
                    for (k, v) in [t.lat_factor, t.lon, t.radius_factor, t.omega_factor].into_iter().enumerate() {
                        samples[10 + k].push(v);
                    }
                }
            }
        }
        for ((name, lo, hi), s) in ranges.iter().zip(&samples) {
            let (n, width) = (s.len() as f64, hi - lo);
            assert!(n >= 10_000.0, "{name}: only {n} samples");
            let min = s.iter().copied().fold(f64::INFINITY, f64::min);
            let max = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!(min >= *lo && max <= *hi, "{name}: [{min}, {max}] leaves [{lo}, {hi}]");
            assert!(
                min - lo < 0.005 * width && hi - max < 0.005 * width,
                "{name}: the sample [{min}, {max}] does not reach both ends of [{lo}, {hi}]"
            );
            let mean = s.iter().sum::<f64>() / n;
            let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
            let sd = width / 12.0_f64.sqrt();
            assert!((mean - (lo + hi) / 2.0).abs() < 5.0 * sd / n.sqrt(), "{name}: mean {mean}");
            assert!((var / (sd * sd) - 1.0).abs() < 0.05, "{name}: variance {var} against {}", sd * sd);
        }
        // Poisson(1.5): the mean and the first bins of the histogram, within 5 standard errors.
        let n = tropical_counts.len() as f64;
        let mean = tropical_counts.iter().sum::<usize>() as f64 / n;
        assert!((mean - 1.5).abs() < 5.0 * (1.5 / n).sqrt(), "tropical count mean {mean}");
        let mut pmf = (-1.5_f64).exp();
        for k in 0..5usize {
            let p_hat = tropical_counts.iter().filter(|&&c| c == k).count() as f64 / n;
            assert!((p_hat - pmf).abs() < 5.0 * (pmf * (1.0 - pmf) / n).sqrt(), "P(count = {k}) = {p_hat}, want {pmf}");
            pmf *= 1.5 / (k + 1) as f64;
        }
    }

    #[test]
    fn build_flow_is_deterministic_and_the_plan_maps_to_the_documented_vortices() {
        let (a, b) = (build_flow(&EARTH, 7), build_flow(&EARTH, 7));
        assert_eq!(a, b);
        assert_ne!(a, build_flow(&EARTH, 8));
        let m = wave_count(&EARTH).unwrap();
        // The generator is seeded with `seed ^ 0x5F10`, as Python's `default_rng(seed ^ 0x5F10)`.
        assert_eq!(a, build_flow_from_plan(&EARTH, &FlowPlan::draw(m, &mut Rng::new(7 ^ 0x5F10))));
        let cyc = a.vortices.iter().filter(|v| v.kind == VortexKind::Cyclone).count();
        let anti = a.vortices.iter().filter(|v| v.kind == VortexKind::Anticyclone).count();
        assert_eq!((cyc, anti), (2 * m, 2 * m));
        // Hand-built plan: one low and its high per hemisphere, no tropical storm, unit factors.
        let low = LowDraws {
            lon_jitter: 0.0,
            lat_offset: 0.0,
            age: 1.0,
            radius_factor: 1.0,
            omega_factor: 1.0,
            high_lon_factor: 1.0,
            high_lat_factor: 1.0,
            high_radius_factor: 1.5,
            high_omega_factor: 1.0,
        };
        let hemi = |lon0: f64| HemispherePlan {
            lon0,
            lows: vec![low.clone(); m],
            tropicals: vec![TropicalDraws { lat_factor: 0.5, lon: 1.0, radius_factor: 1.0, omega_factor: 1.0 }],
        };
        let flow = build_flow_from_plan(&EARTH, &FlowPlan { north: hemi(0.5), south: hemi(-0.5) });
        let lr = deformation_radius_rad(&EARTH);
        let v0 = flow.vortices[0];
        assert_eq!(v0.kind, VortexKind::Cyclone);
        assert!((v0.lat - 51.0_f64.to_radians()).abs() < 1e-12 && (v0.lon - 0.5).abs() < 1e-15);
        assert!((v0.radius - lr).abs() < 1e-15 && (v0.omega - 3.0).abs() < 1e-15 && v0.ascent == 1.0 && v0.age == 1.0);
        let v1 = flow.vortices[1];
        assert_eq!(v1.kind, VortexKind::Anticyclone);
        assert!((v1.lat - 30.0_f64.to_radians()).abs() < 1e-12);
        assert!((v1.lon - (0.5 + PI / m as f64)).abs() < 1e-12);
        assert!((v1.radius - 1.5 * lr).abs() < 1e-15 && (v1.omega + 1.2).abs() < 1e-15 && v1.ascent == -0.45);
        let tropical = flow.vortices[2 * m];
        assert_eq!(tropical.kind, VortexKind::Tropical);
        assert!((tropical.lat - 15.0_f64.to_radians()).abs() < 1e-12);
        assert!(
            (tropical.radius - 0.35 * lr).abs() < 1e-15
                && (tropical.omega - 12.0).abs() < 1e-15
                && tropical.ascent == 1.6
        );
        // The southern hemisphere mirrors the latitudes and flips the windings.
        let south0 = flow.vortices[2 * m + 1];
        assert!((south0.lat + 51.0_f64.to_radians()).abs() < 1e-12 && (south0.omega + 3.0).abs() < 1e-15);
        // A retrograde spin flips every winding but not a latitude.
        let retro = build_flow_from_plan(
            &BodyInputs::new(Some(-0.99727), 6371.0, 9.81, 8.0),
            &FlowPlan { north: hemi(0.5), south: hemi(-0.5) },
        );
        assert!((retro.vortices[0].omega + 3.0).abs() < 1e-15 && (retro.vortices[0].lat - v0.lat).abs() < 1e-15);
    }

    #[test]
    #[should_panic(expected = "lows per hemisphere")]
    fn a_plan_of_the_wrong_size_is_refused() {
        let empty = HemispherePlan { lon0: 0.0, lows: Vec::new(), tropicals: Vec::new() };
        build_flow_from_plan(&EARTH, &FlowPlan { north: empty.clone(), south: empty });
    }
}
