//! Crops of a huge 16-bit gray PNG at three scales, read row by row (the image never fits memory as a whole):
//!
//!     cargo run --release -p cloud-map-generator-service --example crop -- <master.png> <out-dir> [x0 y0 size]
//!
//! Writes `crop_1to1.png` (size x size texels at x0, y0), `crop_1to8.png` (8 x size texels box-averaged down to
//! size x size, from the same corner) and `crop_whole.png` (the whole map averaged to at most 1024 px wide), all 8-bit.
//! Used to look for seams between bands and stripes in a banded map.

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

fn write8(path: &PathBuf, w: usize, h: usize, data: &[u8]) {
    let mut enc = png::Encoder::new(std::io::BufWriter::new(File::create(path).expect("create")), w as u32, h as u32);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().expect("header").write_image_data(data).expect("data");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (src, out) = (&args[1], PathBuf::from(&args[2]));
    std::fs::create_dir_all(&out).expect("out dir");
    let reader = png::Decoder::new(BufReader::with_capacity(1 << 20, File::open(src).expect("open")));
    let mut reader = reader.read_info().expect("a PNG");
    let (w, h) = (reader.info().width as usize, reader.info().height as usize);
    let num = |i: usize, d: usize| args.get(i).and_then(|s| s.parse::<usize>().ok()).unwrap_or(d);
    let size = num(5, 1024);
    let (x0, y0) = (num(3, w / 2), num(4, (h / 4 / 256) * 256 - size / 2));
    let big = size * 8;
    let whole_f = w.div_ceil(1024);
    let (ww, wh) = (w / whole_f, h / whole_f);
    let mut c1 = vec![0u8; size * size];
    let mut c8 = vec![0f64; size * size];
    let mut cw = vec![0f64; ww * wh];
    let mut y = 0;
    while let Some(row) = reader.next_row().expect("row") {
        let r = row.data();
        let v = |x: usize| f64::from(u16::from_be_bytes([r[2 * x], r[2 * x + 1]])) / 257.0;
        if y >= y0 && y < y0 + size {
            for x in 0..size {
                c1[(y - y0) * size + x] = v(x0 + x).round() as u8;
            }
        }
        if y >= y0 && y < y0 + big {
            let cy = (y - y0) / 8;
            for x in 0..big.min(w - x0.min(w)) {
                c8[cy * size + x / 8] += v(x0 + x) / 64.0;
            }
        }
        if y / whole_f < wh {
            let cy = y / whole_f;
            for x in 0..ww * whole_f {
                cw[cy * ww + x / whole_f] += v(x) / (whole_f * whole_f) as f64;
            }
        }
        y += 1;
        if y >= h {
            break;
        }
    }
    write8(&out.join("crop_1to1.png"), size, size, &c1);
    write8(&out.join("crop_1to8.png"), size, size, &c8.iter().map(|&v| v.round().min(255.0) as u8).collect::<Vec<_>>());
    write8(&out.join("crop_whole.png"), ww, wh, &cw.iter().map(|&v| v.round().min(255.0) as u8).collect::<Vec<_>>());
    println!("{w} x {h}: crops at ({x0}, {y0}), size {size}, whole at 1/{whole_f}");
}
