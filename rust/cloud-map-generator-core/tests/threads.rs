//! The CPU engine gives the same bytes on any number of threads: `generate` at 512 with 1, 3 and 16 threads, with
//! and without a surface (the surface runs the per-texel threshold offsets and the orography, the paths that cost
//! the most), compared byte for byte on the coverage and on the extras.
//!
//! Without the `rayon` feature every count runs on the calling thread, and the test then only pins that the helpers
//! keep the serial result.

use cloud_map_generator_core::climate::EARTH;
use cloud_map_generator_core::generate::{generate, GenOutput, GenParams};
use cloud_map_generator_core::par::with_threads;
use cloud_map_generator_core::surface::Surface;

/// A made-up planet: continents as lobes of longitude and latitude, with a height that rises inland.
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
    Surface::new(land, elevation, h, w, "thread test")
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn run(threads: usize, surf: Option<&Surface>) -> GenOutput {
    with_threads(threads, || generate(&EARTH, 42, 512, &GenParams::default(), surf, true).expect("a bakeable body"))
}

fn assert_same(a: &GenOutput, b: &GenOutput, what: &str) {
    assert_eq!(bytes(&a.coverage), bytes(&b.coverage), "{what}: coverage");
    let (ea, eb) = (a.extras.as_ref().expect("extras"), b.extras.as_ref().expect("extras"));
    assert_eq!(bytes(&ea.kind), bytes(&eb.kind), "{what}: type map");
    assert_eq!(bytes(&ea.high_share), bytes(&eb.high_share), "{what}: high share");
    assert_eq!(bytes(&ea.wind), bytes(&eb.wind), "{what}: wind");
}

#[test]
fn generate_is_byte_identical_for_1_3_and_16_threads() {
    let one = run(1, None);
    assert!(one.coverage.iter().any(|&c| c > 0.0), "a real map");
    for n in [3, 16] {
        assert_same(&one, &run(n, None), &format!("no surface, {n} threads"));
    }
}

#[test]
fn generate_with_a_surface_is_byte_identical_for_1_3_and_16_threads() {
    let surf = surface(256, 512);
    let one = run(1, Some(&surf));
    assert!(one.meta.surface.is_some(), "the surface terms ran");
    for n in [3, 16] {
        assert_same(&one, &run(n, Some(&surf)), &format!("surface, {n} threads"));
    }
}
