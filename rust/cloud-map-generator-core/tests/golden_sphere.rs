//! Golden cases of cloudgen/sphere.py (group sphere_noise, functions cloudgen.sphere.*).

use cloud_map_generator_core::sphere::*;
use cloud_map_generator_golden::{run, Array, Case, Config, DType, Out};

const EXPECTED_FAILURES: &[(&str, &str)] = &[];

fn vec3_f32(a: &Array) -> Vec<Vec3<f32>> {
    a.as_f32().chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}
fn vec3_f64(a: &Array) -> Vec<Vec3<f64>> {
    a.to_f64().chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}
fn flat<F: Copy>(v: &[Vec3<F>]) -> Vec<F> {
    v.iter().flatten().copied().collect()
}
/// Shape of an array without its trailing axis of 3.
fn lead(a: &Array) -> Vec<usize> {
    a.shape[..a.shape.len() - 1].to_vec()
}

fn port(c: &Case) -> Option<Out> {
    Some(match c.function.as_str() {
        "cloudgen.sphere.grid_lonlat" => {
            let (lon, lat) = grid_lonlat(c.p_usize("width"), c.p_usize("height"));
            Out::new().f64("lon", &[lon.len()], &lon).f64("lat", &[lat.len()], &lat)
        }
        "cloudgen.sphere.lonlat_to_dir" => {
            let (lon, lat) = (c.input("lon"), c.input("lat"));
            let mut sh = lon.shape.clone();
            sh.push(3);
            match lon.dtype {
                DType::F32 => {
                    let d: Vec<_> = lon.as_f32().iter().zip(lat.as_f32()).map(|(&a, b)| lonlat_to_dir(a, b)).collect();
                    Out::new().f32("dir", &sh, &flat(&d))
                }
                _ => {
                    let d: Vec<_> = lon.as_f64().iter().zip(lat.as_f64()).map(|(&a, b)| lonlat_to_dir(a, b)).collect();
                    Out::new().f64("dir", &sh, &flat(&d))
                }
            }
        }
        "cloudgen.sphere.dir_to_lonlat" => {
            let a = c.input("dir");
            let n = lead(a);
            if a.dtype == DType::F32 {
                let (lo, la): (Vec<f32>, Vec<f32>) = vec3_f32(a).into_iter().map(dir_to_lonlat).unzip();
                Out::new().f32("lon", &n, &lo).f32("lat", &n, &la)
            } else {
                let (lo, la): (Vec<f64>, Vec<f64>) = vec3_f64(a).into_iter().map(dir_to_lonlat).unzip();
                Out::new().f64("lon", &n, &lo).f64("lat", &n, &la)
            }
        }
        "cloudgen.sphere.grid_dirs" => {
            let (w, h) = (c.p_usize("width"), c.p_usize("height"));
            if c.p_str("dtype") == "float32" {
                Out::new().f32("dirs", &[h, w, 3], &flat(&grid_dirs::<f32>(w, h)))
            } else {
                Out::new().f64("dirs", &[h, w, 3], &flat(&grid_dirs::<f64>(w, h)))
            }
        }
        "cloudgen.sphere.normalize" => {
            let v = c.input("v");
            if v.dtype == DType::F32 {
                let o: Vec<_> = vec3_f32(v).into_iter().map(normalize).collect();
                Out::new().f32("out", &v.shape, &flat(&o))
            } else {
                let o: Vec<_> = vec3_f64(v).into_iter().map(normalize).collect();
                Out::new().f64("out", &v.shape, &flat(&o))
            }
        }
        "cloudgen.sphere.tangent_frame" => {
            let ctr = c.input("center");
            let (e, n): (Vec<_>, Vec<_>) = vec3_f64(ctr).into_iter().map(tangent_frame).unzip();
            Out::new().f64("east", &ctr.shape, &flat(&e)).f64("north", &ctr.shape, &flat(&n))
        }
        "cloudgen.sphere.gnomonic_patch" => {
            let n = c.p_usize("n");
            let p = gnomonic_patch(c.p_f64("lon0"), c.p_f64("lat0"), c.p_f64("half_angle"), n);
            Out::new().f64("patch", &[n, n, 3], &flat(&p))
        }
        "cloudgen.sphere.sample_equirect" => {
            let (img, dirs) = (c.input("img"), c.input("dirs"));
            let (h, w) = (img.shape[0], img.shape[1]);
            let ch = if img.shape.len() == 3 { img.shape[2] } else { 1 };
            let pix = img.to_f64();
            let o = if dirs.dtype == DType::F32 {
                sample_equirect::<f32>(&pix, h, w, ch, &vec3_f32(dirs))
            } else {
                sample_equirect::<f64>(&pix, h, w, ch, &vec3_f64(dirs))
            };
            let mut shape = lead(dirs);
            if img.shape.len() == 3 {
                shape.push(ch);
            }
            Out::new().f64("out", &shape, &o)
        }
        "cloudgen.sphere.upsample_dirs" => {
            let d = c.input("dirs");
            let (hc, wc) = (d.shape[0], d.shape[1]);
            let (w, h) = (c.p_usize("width"), c.p_usize("height"));
            let up = upsample_dirs(&vec3_f64(d), hc, wc, w, h);
            if c.has_param("rows") {
                let rows: Vec<usize> = c.p_f64s("rows").iter().map(|&r| r as usize).collect();
                let picked: Vec<f32> = rows.iter().flat_map(|&r| flat(&up[r * w..(r + 1) * w])).collect();
                let mut sums = Vec::with_capacity(h * 3);
                for r in 0..h {
                    let mut s = [0.0_f64; 3];
                    for v in &up[r * w..(r + 1) * w] {
                        for k in 0..3 {
                            s[k] += f64::from(v[k]);
                        }
                    }
                    sums.extend_from_slice(&s);
                }
                Out::new().f32("rows", &[rows.len(), w, 3], &picked).f64("row_sum", &[h, 3], &sums)
            } else {
                Out::new().f32("out", &[h, w, 3], &flat(&up))
            }
        }
        _ => return None,
    })
}

#[test]
fn sphere_cases() {
    run(&Config { group: "sphere_noise", prefixes: &["cloudgen.sphere."], expected_failures: EXPECTED_FAILURES }, port).assert_clean();
}
