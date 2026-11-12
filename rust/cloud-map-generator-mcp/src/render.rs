//! The pictures this server draws itself: a box-averaged preview of a coverage map and an orthographic globe,
//! ported from `cloudgen/service.py` (`preview_png`, `globe_png`), plus the PNG reading and writing they need.
//!
//! Everything is deterministic: the same input always gives the same bytes. A map is a [`Gray`]: cloud fraction
//! as `f32` in [0, 1], north-up, row 0 at the north pole, column 0 at longitude -180, east to the right.

use std::io::Cursor;

use crate::backend::Plane;

/// A north-up equirectangular grid of cloud fraction in [0, 1].
#[derive(Debug, Clone)]
pub struct Gray {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl Gray {
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(width * height, data.len(), "a grid holds width x height samples");
        Gray { width, height, data }
    }

    /// An 8-bit plane as fractions (0..255 = 0..1).
    pub fn from_plane(p: &Plane) -> Self {
        Gray::new(p.width, p.height, p.bytes.iter().map(|&b| f32::from(b) / 255.0).collect())
    }

    /// An 8-bit plane scaled so its largest value is 1 (a speed has no natural full scale).
    pub fn from_plane_normalized(p: &Plane) -> Self {
        let max = f32::from(p.bytes.iter().copied().max().unwrap_or(0)).max(1e-6);
        Gray::new(p.width, p.height, p.bytes.iter().map(|&b| f32::from(b) / max).collect())
    }
}

// ------------------------------------------------------------------------------------------------ resizing

/// Resampling coefficients of PIL's BOX filter: for every output sample the first input sample and the weights
/// (equal, summing to 1) of the input samples whose centres fall inside the window.
fn box_coefficients(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 0.5 * filter_scale;
    let ss = 1.0 / filter_scale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0) as usize;
            let xmax = (((center + support + 0.5) as i64).max(0) as usize).min(in_size).saturating_sub(xmin);
            let mut w: Vec<f64> = (0..xmax)
                .map(|x| {
                    let v = (x as f64 + xmin as f64 - center + 0.5) * ss;
                    if (-0.5..0.5).contains(&v) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let sum: f64 = w.iter().sum();
            if sum != 0.0 {
                w.iter_mut().for_each(|v| *v /= sum);
            }
            (xmin, w)
        })
        .collect()
}

/// `Image.resize((dw, dh), Image.BOX)` on a float image: horizontal pass, then vertical.
pub fn box_resize(src: &Gray, dw: usize, dh: usize) -> Gray {
    let (sw, sh) = (src.width, src.height);
    let mut tmp = src.clone();
    if dw != sw {
        let coeffs = box_coefficients(sw, dw);
        let mut out = vec![0f32; dw * sh];
        for y in 0..sh {
            let row = &src.data[y * sw..(y + 1) * sw];
            for (x, (xmin, w)) in coeffs.iter().enumerate() {
                let acc: f64 = w.iter().enumerate().map(|(k, wk)| f64::from(row[xmin + k]) * wk).sum();
                out[y * dw + x] = acc as f32;
            }
        }
        tmp = Gray::new(dw, sh, out);
    }
    if dh != sh {
        let coeffs = box_coefficients(sh, dh);
        let w = tmp.width;
        let mut out = vec![0f32; w * dh];
        for (y, (ymin, wy)) in coeffs.iter().enumerate() {
            for x in 0..w {
                let acc: f64 = wy.iter().enumerate().map(|(k, wk)| f64::from(tmp.data[(ymin + k) * w + x]) * wk).sum();
                out[y * w + x] = acc as f32;
            }
        }
        tmp = Gray::new(w, dh, out);
    }
    tmp
}

/// Box average down to `max_width` columns (a no-op when already narrower), keeping the 2:1 shape.
pub fn shrink(a: &Gray, max_width: usize) -> Gray {
    if a.width <= max_width {
        return a.clone();
    }
    box_resize(a, max_width, (max_width / 2).max(1))
}

fn clipped(a: &Gray) -> Gray {
    Gray::new(a.width, a.height, a.data.iter().map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }).collect())
}

// ------------------------------------------------------------------------------------------------ PNG writing

fn encode(width: usize, height: usize, color: png::ColorType, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, width as u32, height as u32);
        enc.set_color(color);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().expect("writing a PNG header to memory cannot fail");
        w.write_image_data(data).expect("writing PNG pixels to memory cannot fail");
        w.finish().expect("finishing a PNG in memory cannot fail");
    }
    out
}

/// An 8-bit gray PNG.
pub fn encode_gray8(width: usize, height: usize, data: &[u8]) -> Vec<u8> {
    assert_eq!(data.len(), width * height);
    encode(width, height, png::ColorType::Grayscale, data)
}

/// An 8-bit RGB PNG, pixels interleaved.
pub fn encode_rgb8(width: usize, height: usize, data: &[u8]) -> Vec<u8> {
    assert_eq!(data.len(), width * height * 3);
    encode(width, height, png::ColorType::Rgb, data)
}

/// A 16-bit gray PNG (samples big-endian as the format wants).
pub fn encode_gray16(width: usize, height: usize, data: &[u16]) -> Vec<u8> {
    assert_eq!(data.len(), width * height);
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_be_bytes()).collect();
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, width as u32, height as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Sixteen);
        let mut w = enc.write_header().expect("writing a PNG header to memory cannot fail");
        w.write_image_data(&bytes).expect("writing PNG pixels to memory cannot fail");
        w.finish().expect("finishing a PNG in memory cannot fail");
    }
    out
}

/// `round(x)` to the nearest integer, ties to even (numpy's `np.round`).
fn to_u8(x: f64) -> u8 {
    x.clamp(0.0, 255.0).round_ties_even() as u8
}

/// An 8-bit gray PNG of a coverage map (white = cloud), box-averaged down to at most `max_width` columns (a
/// smaller map is not enlarged). Deterministic.
pub fn preview_png(coverage: &Gray, max_width: usize) -> Vec<u8> {
    let small = shrink(&clipped(coverage), max_width);
    let bytes: Vec<u8> = small.data.iter().map(|&v| to_u8(f64::from(v) * 255.0)).collect();
    encode_gray8(small.width, small.height, &bytes)
}

// ------------------------------------------------------------------------------------------------ the globe

const GLOBE_OCEAN: [f64; 3] = [28.0, 38.0, 54.0]; // the dark blue-gray sphere
const GLOBE_CLOUD: [f64; 3] = [244.0, 246.0, 250.0];
const GLOBE_SPACE: [f64; 3] = [7.0, 9.0, 14.0];
const GLOBE_FILL: f64 = 0.94; // the sphere's diameter as a share of the image

type V3 = [f64; 3];

fn lonlat_to_dir(lon: f64, lat: f64) -> V3 {
    let cl = lat.cos();
    [cl * lon.cos(), cl * lon.sin(), lat.sin()]
}

fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// East and north unit vectors at a direction (poles: an arbitrary but fixed frame).
fn tangent_frame(center: V3) -> (V3, V3) {
    let east = cross([0.0, 0.0, 1.0], center);
    let n = norm(east);
    let east = if n < 1e-9 { [0.0, 1.0, 0.0] } else { [east[0] / n, east[1] / n, east[2] / n] };
    (east, cross(center, east))
}

/// Bilinear lookup of a north-up equirectangular map at a unit direction (longitude wraps, latitude clamps).
fn sample_equirect(img: &Gray, d: V3) -> f64 {
    let (w, h) = (img.width as i64, img.height as i64);
    let lon = d[1].atan2(d[0]);
    let lat = d[2].clamp(-1.0, 1.0).asin();
    let fx = (lon + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * w as f64 - 0.5;
    let fy = (std::f64::consts::FRAC_PI_2 - lat) / std::f64::consts::PI * h as f64 - 0.5;
    let (x0, y0) = (fx.floor() as i64, fy.floor() as i64);
    let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
    let (x0w, x1w) = (x0.rem_euclid(w) as usize, (x0 + 1).rem_euclid(w) as usize);
    let (y0c, y1c) = (y0.clamp(0, h - 1) as usize, (y0 + 1).clamp(0, h - 1) as usize);
    let at = |y: usize, x: usize| f64::from(img.data[y * img.width + x]);
    let a = at(y0c, x0w) * (1.0 - tx) + at(y0c, x1w) * tx;
    let b = at(y1c, x0w) * (1.0 - tx) + at(y1c, x1w) * tx;
    a * (1.0 - ty) + b * ty
}

/// An orthographic view of the planet: clouds white over a dark blue-gray sphere, a thin limb darkening, space
/// around it. The point at (`lon_deg`, `lat_deg`) faces the viewer, north up and east to the right (the centre
/// pixel of an odd `size` is exactly that point). `size` is the square image's side. At lat_deg = +-90 the frame
/// is fixed and `lon_deg` has no effect: from the north pole the top of the image is longitude 180 and east
/// (90E) is on the right, so east runs counterclockwise; from the south pole the top is longitude 0 and east is
/// on the right, so east runs clockwise. Use lat_deg 89.9 or -89.9 to choose the meridian.
pub fn globe_png(coverage: &Gray, lon_deg: f64, lat_deg: f64, size: usize) -> Vec<u8> {
    let n = size;
    let cov = shrink(&clipped(coverage), 2 * n); // about one texel per pixel at the sphere's centre
    let radius = GLOBE_FILL * n as f64 / 2.0;
    let center = lonlat_to_dir(lon_deg.to_radians(), lat_deg.to_radians());
    let (east, north) = tangent_frame(center);
    let mut rgb = vec![0u8; n * n * 3];
    for r in 0..n {
        let y = -(r as f64 + 0.5 - n as f64 / 2.0) / radius;
        for c in 0..n {
            let x = (c as f64 + 0.5 - n as f64 / 2.0) / radius;
            let rho2 = x * x + y * y;
            let z = (1.0 - rho2).clamp(0.0, 1.0).sqrt();
            let p = [
                z * center[0] + x * east[0] + y * north[0],
                z * center[1] + x * east[1] + y * north[1],
                z * center[2] + x * east[2] + y * north[2],
            ];
            let pn = norm(p);
            let cloud = sample_equirect(&cov, [p[0] / pn, p[1] / pn, p[2] / pn]);
            let limb = 1.0 - 0.6 * (1.0 - z).powi(3);
            let inside = ((1.0 - rho2.sqrt()) * radius + 0.5).clamp(0.0, 1.0);
            for k in 0..3 {
                let surface = GLOBE_OCEAN[k] * (1.0 - cloud) + GLOBE_CLOUD[k] * cloud;
                let v = GLOBE_SPACE[k] * (1.0 - inside) + surface * limb * inside;
                rgb[(r * n + c) * 3 + k] = to_u8(v);
            }
        }
    }
    encode_rgb8(n, n, &rgb)
}

// ------------------------------------------------------------------------------------------------ PNG reading

/// What is known of an image file before it is decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sniff {
    /// A PNG of this size (read from its header only).
    Png { width: u32, height: u32 },
    /// A well-known image format that is not PNG (named as PIL names it).
    Other(&'static str),
    /// Not an image this server can read.
    Unreadable,
}

/// The kind of an image file from its first bytes, and the size of a PNG from its header: nothing is decoded,
/// so a small file that claims a huge picture is refused before it costs memory.
pub fn sniff(bytes: &[u8]) -> Sniff {
    const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    if bytes.len() >= 8 && bytes[..8] == SIGNATURE {
        let reader = png::Decoder::new(Cursor::new(bytes)).read_info();
        return match reader {
            Ok(r) => Sniff::Png { width: r.info().width, height: r.info().height },
            Err(_) => Sniff::Unreadable,
        };
    }
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return Sniff::Other("JPEG");
    }
    if bytes.starts_with(b"GIF8") {
        return Sniff::Other("GIF");
    }
    if bytes.starts_with(b"BM") {
        return Sniff::Other("BMP");
    }
    if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        return Sniff::Other("WEBP");
    }
    if bytes.starts_with(&[b'I', b'I', 42, 0]) || bytes.starts_with(&[b'M', b'M', 0, 42]) {
        return Sniff::Other("TIFF");
    }
    Sniff::Unreadable
}

/// A coverage PNG (8-bit or 16-bit gray, or the first channel of a colour image, north-up) as fractions in
/// [0, 1]. A 16-bit file is divided by 65535, anything else by 255.
pub fn decode_coverage(bytes: &[u8]) -> Result<Gray, String> {
    let mut dec = png::Decoder::new(Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND);
    let mut reader = dec.read_info().map_err(|e| e.to_string())?;
    let size = reader.output_buffer_size().ok_or_else(|| "image too large".to_string())?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => 1,
    };
    let (w, h) = (info.width as usize, info.height as usize);
    let wide = info.bit_depth == png::BitDepth::Sixteen;
    let stride = channels * if wide { 2 } else { 1 };
    let scale = if wide { 65535.0 } else { 255.0 };
    let data: Vec<f32> = buf[..w * h * stride]
        .chunks_exact(stride)
        .map(|px| {
            let v = if wide { f32::from(u16::from_be_bytes([px[0], px[1]])) } else { f32::from(px[0]) };
            (v / scale).clamp(0.0, 1.0)
        })
        .collect();
    Ok(Gray::new(w, h, data))
}

/// Decode an 8-bit gray PNG into its bytes (used to check what was written).
pub fn decode_gray8(bytes: &[u8]) -> Result<(usize, usize, Vec<u8>), String> {
    let mut reader = png::Decoder::new(Cursor::new(bytes)).read_info().map_err(|e| e.to_string())?;
    let size = reader.output_buffer_size().ok_or_else(|| "image too large".to_string())?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    Ok((info.width as usize, info.height as usize, buf[..(info.width * info.height) as usize].to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bands(w: usize) -> Gray {
        let h = w / 2;
        let data = (0..h).flat_map(|r| (0..w).map(move |c| ((r * 7 + c * 3) % 256) as f32 / 255.0)).collect();
        Gray::new(w, h, data)
    }

    #[test]
    fn the_preview_is_a_box_average_no_wider_than_asked() {
        let g = bands(512);
        let (w, h, px) = decode_gray8(&preview_png(&g, 256)).unwrap();
        assert_eq!((w, h), (256, 128));
        // the integer case is the mean of 2 x 2 samples
        let expect = (g.data[0] + g.data[1] + g.data[512] + g.data[513]) / 4.0;
        assert_eq!(px[0], (expect * 255.0).round_ties_even() as u8);
        let (w, h, _) = decode_gray8(&preview_png(&g, 768)).unwrap();
        assert_eq!((w, h), (512, 256), "a smaller map is not enlarged");
        let (w, h, _) = decode_gray8(&preview_png(&bands(1024), 768)).unwrap();
        assert_eq!((w, h), (768, 384));
        assert_eq!(preview_png(&g, 300), preview_png(&g, 300));
    }

    /// A map that is dark everywhere except a bright blob around (lon, lat).
    fn blob(lon_deg: f64, lat_deg: f64) -> Gray {
        let (w, h) = (512usize, 256usize);
        let mut data = vec![0f32; w * h];
        for r in 0..h {
            let lat = 90.0 - (r as f64 + 0.5) * 180.0 / h as f64;
            for c in 0..w {
                let lon = -180.0 + (c as f64 + 0.5) * 360.0 / w as f64;
                let mut dlon = (lon - lon_deg).abs();
                if dlon > 180.0 {
                    dlon = 360.0 - dlon;
                }
                if dlon * lat.to_radians().cos().max(0.2) < 12.0 && (lat - lat_deg).abs() < 12.0 {
                    data[r * w + c] = 1.0;
                }
            }
        }
        Gray::new(w, h, data)
    }

    fn pixel(png_bytes: &[u8], x: usize, y: usize) -> [u8; 3] {
        let mut reader = png::Decoder::new(Cursor::new(png_bytes)).read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!(info.color_type, png::ColorType::Rgb);
        let i = (y * info.width as usize + x) * 3;
        [buf[i], buf[i + 1], buf[i + 2]]
    }

    #[test]
    fn the_globe_centre_pixel_of_an_odd_size_is_the_sub_point() {
        let g = blob(40.0, -30.0);
        let n = 129;
        let at_blob = globe_png(&g, 40.0, -30.0, n);
        let c = pixel(&at_blob, n / 2, n / 2);
        assert!(c[0] > 200 && c[1] > 200, "the sub-point is on the cloud: {c:?}");
        let away = globe_png(&g, 40.0 + 120.0, -30.0, n);
        let c = pixel(&away, n / 2, n / 2);
        assert!(c[0] < 60, "the sub-point is on the dark sphere: {c:?}");
        // the corner is space
        let corner = pixel(&at_blob, 0, 0);
        assert_eq!(corner, [7, 9, 14]);
    }

    #[test]
    fn north_is_up_and_east_is_right() {
        let n = 129;
        // a cloud 25 degrees east of the sub-point lands to the right of the centre, and 25 north above it
        let east = globe_png(&blob(25.0, 0.0), 0.0, 0.0, n);
        let (left, right) = (pixel(&east, n / 2 - 30, n / 2), pixel(&east, n / 2 + 30, n / 2));
        assert!(right[0] > left[0] + 40, "east is to the right: {left:?} {right:?}");
        let north = globe_png(&blob(0.0, 30.0), 0.0, 0.0, n);
        let (up, down) = (pixel(&north, n / 2, n / 2 - 30), pixel(&north, n / 2, n / 2 + 30));
        assert!(up[0] > down[0] + 40, "north is up: {up:?} {down:?}");
    }

    #[test]
    fn the_globe_is_deterministic_and_the_viewpoint_matters() {
        let g = bands(512);
        let a = globe_png(&g, 10.0, 20.0, 64);
        assert_eq!(a, globe_png(&g, 10.0, 20.0, 64));
        assert_ne!(a, globe_png(&g, 100.0, 20.0, 64));
        // the poles are a fixed frame: lon has no effect there
        assert_eq!(globe_png(&g, 0.0, 90.0, 64), globe_png(&g, 77.0, 90.0, 64));
    }

    #[test]
    fn png_files_are_sniffed_without_decoding() {
        let png_bytes = encode_gray8(300, 100, &vec![0u8; 30000]);
        assert_eq!(sniff(&png_bytes), Sniff::Png { width: 300, height: 100 });
        assert_eq!(sniff(&[0xff, 0xd8, 0xff, 0xe0, 0, 0, 0, 0]), Sniff::Other("JPEG"));
        assert_eq!(sniff(b"not an image"), Sniff::Unreadable);
        let g = decode_coverage(&encode_gray16(2, 1, &[0, 65535])).unwrap();
        assert_eq!(g.data, vec![0.0, 1.0]);
        let g = decode_coverage(&encode_rgb8(2, 1, &[255, 0, 0, 51, 9, 9])).unwrap();
        assert_eq!(g.data, vec![1.0, 0.2]);
    }
}
