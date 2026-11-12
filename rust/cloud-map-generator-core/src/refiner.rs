//! The cloud refiner: a small residual U-Net, trained on real satellite cloud maps (`training/`: MODIS true colour inverted against the cloud-free Blue Marble, as Earth's reference map was made), that adds the fine structure
//! of real clouds to a generated coverage map. It is an optional last pass (`refine` of the products), off by default: the
//! generator itself stays a pure function of its parameters; this pass depends on the weights (their version and sha256 are in
//! the sidecar).
//!
//! The network is run here, in plain Rust (no ONNX runtime): 9 convolutions at 3 scales, about 105 thousand weights, so the same
//! code runs natively, in the browser's WebAssembly and in the tests, and gives the same numbers everywhere. The architecture
//! is [`ARCH`]; `training/train.py` has the same table and writes the weights in the file format of [`Refiner::from_bytes`].
//!
//! The network was trained on 2048 x 1024 maps (about 20 km per texel). A map of another width is resampled to that grid, the
//! network's correction (`refined - input`) is resampled back and added to the map as it was, so the detail the generator made at
//! its own resolution is kept.

use crate::par;

/// `(in, out, kernel, stride)` of the convolutions, in the order the weights are stored. Keep in step with `ARCH` of `training/train.py`. The network
/// takes two channels: the cloud fraction and `sin(latitude)` of the row.
pub const ARCH: [(usize, usize, usize, usize); 9] = [
    (2, 16, 3, 1),
    (16, 16, 3, 1),
    (16, 32, 3, 2),
    (32, 32, 3, 1),
    (32, 64, 3, 2),
    (64, 64, 3, 1),
    (64 + 32, 32, 3, 1),
    (32 + 16, 16, 3, 1),
    (16, 1, 3, 1),
];
/// The file format version of `training/train.py` (`VERSION`).
pub const VERSION: u32 = 2;
/// The width the network was trained at.
pub const TRAIN_WIDTH: usize = 2048;
/// Rows of context kept around a strip: more than the network's reach (about 30 texels at full resolution).
const MARGIN: usize = 32;
/// Rows of a strip, without its margins (a multiple of 4: two stride-2 convolutions).
const STRIP: usize = 256;

/// The weights shipped with the crate (`models/cloud-refiner.bin`, written by `training/train.py`).
pub const EMBEDDED: &[u8] = include_bytes!("../models/cloud-refiner.bin");

/// A short identifier of the shipped weights (FNV-1a 64 of the file, hex), for the sidecar: two maps refined with the same id were refined
/// by the same network.
pub fn weights_id() -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in EMBEDDED {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    format!("{h:016x}")
}

#[derive(Debug)]
struct Conv {
    cin: usize,
    cout: usize,
    k: usize,
    stride: usize,
    /// `[cout][cin][k][k]`.
    w: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Debug)]
pub struct Refiner {
    convs: Vec<Conv>,
}

/// A feature map: `c` planes of `h x w`.
struct Tensor {
    c: usize,
    h: usize,
    w: usize,
    data: Vec<f32>,
}

impl Conv {
    /// `k x k` convolution with zero padding `k / 2`, then `relu` when asked.
    fn run(&self, x: &Tensor, relu: bool) -> Tensor {
        assert_eq!(x.c, self.cin);
        let (oh, ow) = ((x.h + self.stride - 1) / self.stride, (x.w + self.stride - 1) / self.stride);
        let mut out = vec![0.0_f32; self.cout * oh * ow];
        let pad = self.k / 2;
        par::for_each_chunk_mut(&mut out, oh * ow, |co, plane| {
            plane.fill(self.b[co]);
            for ci in 0..self.cin {
                let src = &x.data[ci * x.h * x.w..(ci + 1) * x.h * x.w];
                for ky in 0..self.k {
                    for kx in 0..self.k {
                        let wv = self.w[((co * self.cin + ci) * self.k + ky) * self.k + kx];
                        if wv == 0.0 {
                            continue;
                        }
                        for y in 0..oh {
                            let iy = (y * self.stride + ky) as isize - pad as isize;
                            if iy < 0 || iy >= x.h as isize {
                                continue;
                            }
                            let row = &src[iy as usize * x.w..(iy as usize + 1) * x.w];
                            let dst = &mut plane[y * ow..(y + 1) * ow];
                            if self.stride == 1 {
                                // dst[x] += wv * row[x + kx - pad], where that column exists.
                                let lo = pad.saturating_sub(kx);
                                let hi = ow.min((x.w + pad).saturating_sub(kx));
                                if lo < hi {
                                    let s = lo + kx - pad;
                                    for (d, v) in dst[lo..hi].iter_mut().zip(&row[s..s + (hi - lo)]) {
                                        *d += wv * v;
                                    }
                                }
                            } else {
                                for (xo, d) in dst.iter_mut().enumerate() {
                                    let ix = (xo * self.stride + kx) as isize - pad as isize;
                                    if ix >= 0 && (ix as usize) < x.w {
                                        *d += wv * row[ix as usize];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if relu {
                for v in plane.iter_mut() {
                    *v = v.max(0.0);
                }
            }
        });
        Tensor { c: self.cout, h: oh, w: ow, data: out }
    }
}

fn upsample2(x: &Tensor, h: usize, w: usize) -> Tensor {
    let mut data = vec![0.0_f32; x.c * h * w];
    for c in 0..x.c {
        for y in 0..h {
            let sy = (y / 2).min(x.h - 1);
            for xx in 0..w {
                data[(c * h + y) * w + xx] = x.data[(c * x.h + sy) * x.w + (xx / 2).min(x.w - 1)];
            }
        }
    }
    Tensor { c: x.c, h, w, data }
}

fn concat(a: Tensor, b: &Tensor) -> Tensor {
    assert_eq!((a.h, a.w), (b.h, b.w));
    let mut data = a.data;
    data.extend_from_slice(&b.data);
    Tensor { c: a.c + b.c, h: a.h, w: a.w, data }
}

impl Refiner {
    /// The weights shipped with the crate.
    pub fn embedded() -> Result<Refiner, String> {
        Refiner::from_bytes(EMBEDDED)
    }

    /// `CMGR`, the version and the number of convolutions (u32 little endian), then per convolution `in, out, kernel, stride` (u32) and
    /// the weights `[out][in][k][k]` and the biases (f32 little endian).
    pub fn from_bytes(bytes: &[u8]) -> Result<Refiner, String> {
        let u32_at = |at: usize| -> Result<u32, String> {
            bytes.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| "the refiner file is cut short".to_string())
        };
        if bytes.get(..4) != Some(b"CMGR") {
            return Err("not a refiner file (no CMGR header)".to_string());
        }
        let (version, count) = (u32_at(4)?, u32_at(8)? as usize);
        if version != VERSION || count != ARCH.len() {
            return Err(format!("refiner file version {version} with {count} convolutions: this build reads version {VERSION} with {}", ARCH.len()));
        }
        let mut at = 12;
        let mut convs = Vec::with_capacity(count);
        for (n, &(cin, cout, k, stride)) in ARCH.iter().enumerate() {
            let got = (u32_at(at)? as usize, u32_at(at + 4)? as usize, u32_at(at + 8)? as usize, u32_at(at + 12)? as usize);
            if got != (cin, cout, k, stride) {
                return Err(format!("convolution {n} is {got:?}, expected {:?}", (cin, cout, k, stride)));
            }
            at += 16;
            let nw = cout * cin * k * k;
            let floats = |from: usize, n: usize| -> Result<Vec<f32>, String> {
                let raw = bytes.get(from..from + 4 * n).ok_or_else(|| "the refiner file is cut short".to_string())?;
                Ok(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
            };
            let w = floats(at, nw)?;
            let b = floats(at + 4 * nw, cout)?;
            at += 4 * (nw + cout);
            convs.push(Conv { cin, cout, k, stride, w, b });
        }
        if at != bytes.len() {
            return Err("the refiner file has bytes after its last convolution".to_string());
        }
        Ok(Refiner { convs })
    }

    /// The network on one window (`h` and `w` multiples of 4; channel 0 the cloud fraction, channel 1 `sin(latitude)`): `x + out(u-net(x))`, not clamped.
    fn forward(&self, x: Tensor) -> Vec<f32> {
        let c = &self.convs;
        let a = c[1].run(&c[0].run(&x, true), true);
        let b = c[3].run(&c[2].run(&a, true), true);
        let d = c[5].run(&c[4].run(&b, true), true);
        let u = c[6].run(&concat(upsample2(&d, b.h, b.w), &b), true);
        let u = c[7].run(&concat(upsample2(&u, a.h, a.w), &a), true);
        let r = c[8].run(&u, false);
        x.data[..x.h * x.w].iter().zip(&r.data).map(|(a, b)| a + b).collect()
    }

    /// The refined map at the network's own grid: `cov` is `h x w`, rows north to south, columns wrapping around the globe.
    fn refine_native(&self, cov: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; h * w];
        let ww = (w + 2 * MARGIN + 3) / 4 * 4;
        let mut r0 = 0;
        while r0 < h {
            let core = STRIP.min(h - r0);
            let wh = (core + 2 * MARGIN + 3) / 4 * 4;
            let mut data = vec![0.0_f32; 2 * wh * ww];
            for y in 0..wh {
                // Rows past a pole repeat the pole row; columns wrap.
                let sy = (r0 as isize + y as isize - MARGIN as isize).clamp(0, h as isize - 1) as usize;
                let sin_lat = (std::f64::consts::FRAC_PI_2 - (sy as f64 + 0.5) / h as f64 * std::f64::consts::PI).sin() as f32;
                for x in 0..ww {
                    let sx = (x as isize - MARGIN as isize).rem_euclid(w as isize) as usize;
                    data[y * ww + x] = cov[sy * w + sx];
                    data[wh * ww + y * ww + x] = sin_lat;
                }
            }
            let res = self.forward(Tensor { c: 2, h: wh, w: ww, data });
            for y in 0..core {
                for x in 0..w {
                    out[(r0 + y) * w + x] = res[(MARGIN + y) * ww + MARGIN + x].clamp(0.0, 1.0);
                }
            }
            r0 += core;
        }
        out
    }

    /// `cov` (`h x w`, rows north to south, values in `[0, 1]`) refined. At the network's width (2048) this is the network's output; at
    /// another width the map is resampled to that grid, the network's correction is resampled back and added to the map as it was.
    pub fn refine(&self, cov: &[f32], h: usize, w: usize) -> Vec<f32> {
        assert_eq!(cov.len(), h * w);
        if w == TRAIN_WIDTH && h == TRAIN_WIDTH / 2 {
            return self.refine_native(cov, h, w);
        }
        let (th, tw) = (TRAIN_WIDTH / 2, TRAIN_WIDTH);
        let small = resample(cov, h, w, th, tw);
        let refined = self.refine_native(&small, th, tw);
        let delta: Vec<f32> = refined.iter().zip(&small).map(|(a, b)| a - b).collect();
        let back = resample(&delta, th, tw, h, w);
        cov.iter().zip(&back).map(|(a, d)| (a + d).clamp(0.0, 1.0)).collect()
    }
}

/// `src` (`sh x sw`) on a `dh x dw` grid: the mean of the covered texels when it gets smaller (a box), bilinear when it gets larger;
/// columns wrap, rows clamp.
pub fn resample(src: &[f32], sh: usize, sw: usize, dh: usize, dw: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; dh * dw];
    let (fy, fx) = (sh as f64 / dh as f64, sw as f64 / dw as f64);
    for y in 0..dh {
        for x in 0..dw {
            out[y * dw + x] = if fy >= 1.0 && fx >= 1.0 {
                let (y0, y1) = ((y as f64 * fy) as usize, (((y + 1) as f64 * fy).ceil() as usize).min(sh).max(1));
                let (x0, x1) = ((x as f64 * fx) as usize, (((x + 1) as f64 * fx).ceil() as usize).min(sw).max(1));
                let (mut s, mut n) = (0.0_f64, 0.0_f64);
                for yy in y0..y1.max(y0 + 1).min(sh) {
                    for xx in x0..x1.max(x0 + 1).min(sw) {
                        s += src[yy * sw + xx] as f64;
                        n += 1.0;
                    }
                }
                (s / n.max(1.0)) as f32
            } else {
                let gy = ((y as f64 + 0.5) * fy - 0.5).clamp(0.0, sh as f64 - 1.0);
                let gx = (x as f64 + 0.5) * fx - 0.5;
                let (y0, x0) = (gy.floor() as usize, gx.floor());
                let (ty, tx) = ((gy - y0 as f64) as f32, (gx - x0) as f32);
                let y1 = (y0 + 1).min(sh - 1);
                let xa = (x0 as isize).rem_euclid(sw as isize) as usize;
                let xb = (x0 as isize + 1).rem_euclid(sw as isize) as usize;
                let top = src[y0 * sw + xa] * (1.0 - tx) + src[y0 * sw + xb] * tx;
                let bot = src[y1 * sw + xa] * (1.0 - tx) + src[y1 * sw + xb] * tx;
                top * (1.0 - ty) + bot * ty
            };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file with the shape of [`ARCH`] and every weight zero except `last_bias`: the network adds that constant to its input.
    fn file_with_last_bias(last_bias: f32) -> Vec<u8> {
        let mut out = b"CMGR".to_vec();
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(ARCH.len() as u32).to_le_bytes());
        for (n, &(cin, cout, k, stride)) in ARCH.iter().enumerate() {
            for v in [cin, cout, k, stride] {
                out.extend_from_slice(&(v as u32).to_le_bytes());
            }
            out.extend(std::iter::repeat(0u8).take(4 * cout * cin * k * k));
            for _ in 0..cout {
                out.extend_from_slice(&(if n == ARCH.len() - 1 { last_bias } else { 0.0 }).to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn the_embedded_weights_load() {
        Refiner::embedded().unwrap();
    }

    #[test]
    fn a_zero_network_is_the_identity_and_the_last_bias_is_added() {
        let (h, w) = (1024, 2048);
        let cov: Vec<f32> = (0..h * w).map(|k| ((k * 2654435761) >> 11 & 255) as f32 / 255.0 * 0.8).collect();
        let id = Refiner::from_bytes(&file_with_last_bias(0.0)).unwrap();
        assert_eq!(id.refine(&cov, h, w), cov);
        // Another width goes through the resampling: the identity network gives the map back (to rounding).
        let (h2, w2) = (256, 512);
        let small: Vec<f32> = (0..h2 * w2).map(|k| ((k * 40503) >> 3 & 127) as f32 / 255.0).collect();
        let back = id.refine(&small, h2, w2);
        assert!(back.iter().zip(&small).all(|(a, b)| (a - b).abs() < 1e-5));
        let up = Refiner::from_bytes(&file_with_last_bias(0.1)).unwrap();
        let out = up.refine(&small, h2, w2);
        assert!(out.iter().zip(&small).all(|(o, s)| (o - (s + 0.1).min(1.0)).abs() < 2e-2), "the bias reaches every texel, even across the strip seams");
    }

    /// The shipped weights give, on the window `training/train.py` saved, what PyTorch gave. Skipped when the files are not there (they are
    /// not committed: `training/data/` is the trainer's scratch folder) or are those of another training run than the embedded weights.
    #[test]
    fn the_shipped_weights_match_pytorch_on_the_window_the_trainer_saved() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../training/data");
        let (Ok(input), Ok(want)) = (std::fs::read(dir.join("parity_input.f32")), std::fs::read(dir.join("parity_output.f32"))) else {
            eprintln!("no parity window: skipped");
            return;
        };
        let floats = |b: &[u8]| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
        let (x, want) = (floats(&input), floats(&want));
        assert_eq!(x.len(), 2 * 128 * 128, "the window has two channels");
        let got = Refiner::embedded().unwrap().forward(Tensor { c: 2, h: 128, w: 128, data: x });
        let worst = got.iter().zip(&want).map(|(g, w)| (g.clamp(0.0, 1.0) - w).abs()).fold(0.0_f32, f32::max);
        assert!(worst < 2e-3, "worst difference {worst}");
    }

    #[test]
    fn bad_files_are_refused_not_run() {
        assert!(Refiner::from_bytes(b"nope").is_err());
        let good = file_with_last_bias(0.0);
        assert!(Refiner::from_bytes(&good[..good.len() - 1]).unwrap_err().contains("short"));
        let mut longer = good.clone();
        longer.push(0);
        assert!(Refiner::from_bytes(&longer).unwrap_err().contains("after"));
        let mut other_version = good;
        other_version[4] = 9;
        assert!(Refiner::from_bytes(&other_version).unwrap_err().contains("version"));
    }

    #[test]
    fn resample_keeps_the_mean_and_a_constant() {
        let src: Vec<f32> = (0..16 * 32).map(|k| (k % 7) as f32 / 7.0).collect();
        let down = resample(&src, 16, 32, 8, 16);
        let (a, b) = (src.iter().sum::<f32>() / src.len() as f32, down.iter().sum::<f32>() / down.len() as f32);
        assert!((a - b).abs() < 1e-5);
        assert!(resample(&vec![0.3; 8 * 16], 8, 16, 32, 64).iter().all(|v| (v - 0.3).abs() < 1e-6));
    }
}
