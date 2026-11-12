//! The banded engine equals the non-banded `generate` for any band height: bit for bit on the CPU when the noise and
//! the standardization statistics are the same (the statistics are the engine's own float64 block statistics, handed
//! to the reference), and within float rounding of the numpy-style statistics of the plain `generate`.
//!
//! Band heights 1, 7, 64 and the full height at 1024 and 2048, with and without a surface.

use cloud_map_generator_core::banded::{plans_without_subtexel, BandOut, BandSink, Banded, SubNoise};
use cloud_map_generator_core::climate::EARTH as EARTH_BODY;

static EARTH: std::sync::LazyLock<cloud_map_generator_core::climate::BodyInputs> = std::sync::LazyLock::new(|| EARTH_BODY);
use cloud_map_generator_core::counter::{counter_normal, subtexel_noise_scale};
use cloud_map_generator_core::generate::{generate_with_plans, generate_with_plans_norm, subtexel_scale, GenParams, SEED_SUBTEXEL};
use cloud_map_generator_core::par::with_threads;
use cloud_map_generator_core::surface::Surface;

fn surface(h: usize, w: usize) -> Surface {
    let mut land = vec![0.0_f32; h * w];
    let mut elevation = vec![0.0_f32; h * w];
    for r in 0..h {
        let lat = std::f64::consts::FRAC_PI_2 - (r as f64 + 0.5) * std::f64::consts::PI / h as f64;
        for c in 0..w {
            let lon = -std::f64::consts::PI + (c as f64 + 0.5) * 2.0 * std::f64::consts::PI / w as f64;
            let v = (3.0 * lon).sin() * (2.0 * lat).cos() + 0.4 * (lon + 1.0).cos();
            if v > 0.25 {
                land[r * w + c] = 1.0;
                elevation[r * w + c] = ((v - 0.25) * 3.0) as f32;
            }
        }
    }
    Surface::new(land, elevation, h, w, "banded test")
}

#[derive(Default)]
struct Collect {
    coverage: Vec<f32>,
    kind: Vec<f32>,
    high: Vec<f32>,
    rows_seen: usize,
}

impl BandSink for Collect {
    fn band(&mut self, out: &BandOut<'_>) -> Result<(), String> {
        assert_eq!(out.r0, self.rows_seen, "bands arrive north to south, without a gap");
        self.rows_seen += out.rows;
        self.coverage.extend_from_slice(out.coverage);
        self.kind.extend_from_slice(out.kind.expect("extras"));
        self.high.extend_from_slice(out.high_share.expect("extras"));
        Ok(())
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn check(width: usize, band_rows: &[usize], surf: Option<&Surface>) {
    let params = GenParams::default();
    let height = width / 2;
    let mut reference: Option<(cloud_map_generator_core::generate::GenOutput, cloud_map_generator_core::generate::NormStats)> = None;
    for &rows in band_rows {
        let banded = Banded::new(&*EARTH, 42, width, &params, surf, SubNoise::Counter, rows, true).expect("bakeable");
        let mut sink = Collect::default();
        let done = banded.run(&mut sink, &mut |_| true).expect("a banded run");
        assert_eq!(sink.rows_seen, height);
        if reference.is_none() {
            // The non-banded engine with the same noise and the same statistics.
            let mut plans = plans_without_subtexel(&*EARTH, 42, width, &params);
            let divisor = subtexel_scale(&*EARTH, width, &params).expect("the sub-texel term is on");
            let scale = subtexel_noise_scale(divisor);
            let seed = 42u64.wrapping_add(SEED_SUBTEXEL);
            plans.subtexel = Some((0..height * width).map(|i| counter_normal(seed, i as u32) * scale).collect());
            let out = generate_with_plans_norm(&*EARTH, &plans, width, &params, surf, true, false, Some(&done.norm)).expect("bakeable");
            reference = Some((out, done.norm));
        }
        let (r, norm) = reference.as_ref().unwrap();
        assert_eq!(&done.norm, norm, "the statistics do not depend on the band height ({rows} rows, width {width})");
        assert_eq!(bits(&sink.coverage), bits(&r.coverage), "coverage, width {width}, {rows} rows");
        let e = r.extras.as_ref().expect("extras");
        assert_eq!(bits(&sink.kind), bits(&e.kind), "type, width {width}, {rows} rows");
        assert_eq!(bits(&sink.high), bits(&e.high_share), "high share, width {width}, {rows} rows");
        assert_eq!(done.meta.surface.is_some(), surf.is_some());
        if let (Some(a), Some(b)) = (&done.meta.surface, &r.meta.surface) {
            assert_eq!(a.anomaly_range, b.anomaly_range, "the anomaly range of the surface");
            assert_eq!(a.land_fraction, b.land_fraction);
        }
    }
}

#[test]
fn banded_equals_the_reference_at_1024() {
    with_threads(4, || {
        check(1024, &[512, 1, 7, 64], None);
        check(1024, &[512, 1, 7, 64], Some(&surface(256, 512)));
    });
}

#[test]
fn banded_equals_the_reference_at_2048() {
    with_threads(4, || {
        check(2048, &[1024, 1, 7, 64], None);
        check(2048, &[1024, 7, 64], Some(&surface(256, 512)));
    });
}

#[test]
fn banded_equals_the_reference_on_a_narrow_map_where_the_back_trace_is_the_maps_own_grid() {
    with_threads(2, || {
        check(512, &[256, 5, 64], Some(&surface(256, 512)));
        check(256, &[128, 3], None);
    });
}

/// The block statistics are those of numpy within float rounding: the plain `generate` (its own pairwise float32
/// statistics, its own array noise) gives the same map to a rounding error, summarized by the mean and the largest
/// difference when the noise is the same.
#[test]
fn banded_matches_the_plain_generate_within_rounding() {
    with_threads(4, || {
        let (width, params) = (1024, GenParams::default());
        let surf = surface(256, 512);
        let banded = Banded::new(&*EARTH, 42, width, &params, Some(&surf), SubNoise::Counter, 64, true).unwrap();
        let mut sink = Collect::default();
        banded.run(&mut sink, &mut |_| true).unwrap();
        let mut plans = plans_without_subtexel(&*EARTH, 42, width, &params);
        let scale = subtexel_noise_scale(subtexel_scale(&*EARTH, width, &params).unwrap());
        let seed = 42u64.wrapping_add(SEED_SUBTEXEL);
        plans.subtexel = Some((0..width / 2 * width).map(|i| counter_normal(seed, i as u32) * scale).collect());
        let plain = generate_with_plans(&*EARTH, &plans, width, &params, Some(&surf), true, false).unwrap();
        let (mut worst, mut sum) = (0.0_f32, 0.0_f64);
        for (a, b) in sink.coverage.iter().zip(&plain.coverage) {
            let d = (a - b).abs();
            worst = worst.max(d);
            sum += f64::from(d);
        }
        let mean = sum / sink.coverage.len() as f64;
        eprintln!("banded vs plain generate at 1024: mean |diff| {mean:.3e}, max |diff| {worst:.3e}");
        assert!(mean < 1e-5, "mean difference {mean}");
        assert!(worst < 5e-3, "max difference {worst}");
    });
}

#[test]
fn a_cancelled_run_stops() {
    let params = GenParams::default();
    let banded = Banded::new(&*EARTH, 1, 256, &params, None, SubNoise::Counter, 16, false).unwrap();
    let mut n = 0;
    let mut sink = Collect::default();
    let r = banded.run(&mut sink, &mut |_| {
        n += 1;
        n < 5
    });
    assert!(matches!(r, Err(cloud_map_generator_core::banded::BandedError::Cancelled)));
}
