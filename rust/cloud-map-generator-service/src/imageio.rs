//! The container I/O the core leaves out: PNG (the `png` crate) and gzip (`flate2`).
//!
//! The core returns decoded arrays (`export.rs`, `tiles.rs`); the Python tools put them in PNG files with PIL
//! and in `.gz` files with `gzip.compress(.., 9)`. Here: 8-bit gray, 16-bit gray and 8-bit RGB PNGs, a reader
//! for gray (8 or 16 bit, first channel of anything else, as `measure.load` does), and gzip at level 9. The
//! gzip header carries no timestamp, so the same bytes always give the same file.

use crate::fail::{Context, Fail, Result};
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

fn writer(path: &Path, width: usize, height: usize, color: png::ColorType, depth: png::BitDepth) -> Result<png::Writer<BufWriter<File>>> {
    let file = File::create(path).context(|| format!("{}", path.display()))?;
    let w = u32::try_from(width).map_err(|_| Fail::new(format!("{}: width {width} does not fit a PNG", path.display())))?;
    let h = u32::try_from(height).map_err(|_| Fail::new(format!("{}: height {height} does not fit a PNG", path.display())))?;
    let mut enc = png::Encoder::new(BufWriter::new(file), w, h);
    enc.set_color(color);
    enc.set_depth(depth);
    enc.write_header().context(|| format!("{}", path.display()))
}

fn finish(w: png::Writer<BufWriter<File>>, path: &Path) -> Result<()> {
    w.finish().context(|| format!("{}", path.display()))
}

/// A `height x width` 8-bit gray PNG (PIL mode `L`).
pub fn write_gray8(path: &Path, width: usize, height: usize, data: &[u8]) -> Result<()> {
    assert_eq!(data.len(), width * height, "write_gray8: {} bytes for {height} x {width}", data.len());
    let mut w = writer(path, width, height, png::ColorType::Grayscale, png::BitDepth::Eight)?;
    w.write_image_data(data).context(|| format!("{}", path.display()))?;
    finish(w, path)
}

/// A `height x width` 16-bit gray PNG (PIL mode `I;16`), samples big-endian as the format wants.
pub fn write_gray16(path: &Path, width: usize, height: usize, data: &[u16]) -> Result<()> {
    assert_eq!(data.len(), width * height, "write_gray16: {} samples for {height} x {width}", data.len());
    let mut w = writer(path, width, height, png::ColorType::Grayscale, png::BitDepth::Sixteen)?;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_be_bytes()).collect();
    w.write_image_data(&bytes).context(|| format!("{}", path.display()))?;
    finish(w, path)
}

/// A `height x width` 8-bit RGB PNG (PIL mode `RGB`), pixels interleaved.
pub fn write_rgb8(path: &Path, width: usize, height: usize, data: &[u8]) -> Result<()> {
    assert_eq!(data.len(), width * height * 3, "write_rgb8: {} bytes for {height} x {width} x 3", data.len());
    let mut w = writer(path, width, height, png::ColorType::Rgb, png::BitDepth::Eight)?;
    w.write_image_data(data).context(|| format!("{}", path.display()))?;
    finish(w, path)
}

/// The gzip level of every file (`gzip.compress(.., 9)` of the Python tools).
pub const GZIP_LEVEL: u32 = 9;

/// The chunk size of a streamed PNG (the IDAT chunks of a banded map's master).
pub const PNG_ROW_STREAM: usize = 1 << 20;

/// A gzip stream into `w` at [`GZIP_LEVEL`], no timestamp (what [`gzip9`] does to a buffer, for a stream).
pub fn gzip_level<W: Write>(w: W) -> GzEncoder<W> {
    GzEncoder::new(w, Compression::new(GZIP_LEVEL))
}

/// `gzip.compress(bytes, 9)`: level 9, no timestamp.
pub fn gzip9(bytes: &[u8]) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::with_capacity(bytes.len() / 4 + 64), Compression::new(9));
    enc.write_all(bytes).expect("writing to a Vec cannot fail");
    enc.finish().expect("finishing a gzip stream in memory cannot fail")
}

/// Writes `bytes` to `path`, naming the path in the error.
pub fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).context(|| format!("{}", path.display()))
}

/// A decoded gray image: `data` holds one sample per pixel (the first channel of a color image), `max` is the
/// value of full scale (255 for an 8-bit file, 65535 for a 16-bit one).
#[derive(Clone, Debug)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub max: f32,
    pub data: Vec<u16>,
}

/// Reads a PNG as `measure.load` does through PIL and numpy: a gray or color image of 8 or 16 bits, the first
/// channel of the pixels. Palette images are refused (PIL would hand numpy the indices, not colors).
pub fn read_gray(path: &Path) -> Result<GrayImage> {
    let name = || format!("{}", path.display());
    let file = File::open(path).context(name)?;
    let mut dec = png::Decoder::new(BufReader::new(file));
    dec.set_transformations(png::Transformations::EXPAND);
    let mut reader = dec.read_info().context(name)?;
    if reader.info().color_type == png::ColorType::Indexed {
        return Err(Fail::new(format!("{}: a palette PNG is not a coverage map (expected gray)", path.display())));
    }
    let mut buf = vec![0u8; reader.output_buffer_size().ok_or_else(|| Fail::new(format!("{}: image too large", path.display())))?];
    let info = reader.next_frame(&mut buf).context(name)?;
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => unreachable!("EXPAND removes palettes"),
    };
    let (width, height) = (info.width as usize, info.height as usize);
    let (bytes_per_sample, max) = match info.bit_depth {
        png::BitDepth::Sixteen => (2, 65535.0),
        _ => (1, 255.0),
    };
    let stride = channels * bytes_per_sample;
    let data = buf[..width * height * stride]
        .chunks_exact(stride)
        .map(|px| if bytes_per_sample == 2 { u16::from_be_bytes([px[0], px[1]]) } else { u16::from(px[0]) })
        .collect();
    Ok(GrayImage { width, height, max, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("cloud-map-generator-cli-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn gray8_and_gray16_round_trip() {
        let p8 = scratch("g8.png");
        let d8: Vec<u8> = (0..6u8).map(|v| v * 50).collect();
        write_gray8(&p8, 3, 2, &d8).unwrap();
        let g = read_gray(&p8).unwrap();
        assert_eq!((g.width, g.height, g.max), (3, 2, 255.0));
        assert_eq!(g.data, d8.iter().map(|&v| u16::from(v)).collect::<Vec<_>>());

        let p16 = scratch("g16.png");
        let d16 = [0u16, 1, 256, 65535, 32768, 12345];
        write_gray16(&p16, 2, 3, &d16).unwrap();
        let g = read_gray(&p16).unwrap();
        assert_eq!((g.width, g.height, g.max), (2, 3, 65535.0));
        assert_eq!(g.data, d16);
    }

    #[test]
    fn rgb_reads_back_its_first_channel() {
        let p = scratch("rgb.png");
        write_rgb8(&p, 2, 1, &[10, 20, 30, 40, 50, 60]).unwrap();
        let g = read_gray(&p).unwrap();
        assert_eq!(g.data, vec![10, 40]);
    }

    #[test]
    fn gzip_is_deterministic_and_inflates_back() {
        use std::io::Read;
        let data: Vec<u8> = (0..10_000u32).map(|i| (i % 7) as u8).collect();
        let a = gzip9(&data);
        assert_eq!(a, gzip9(&data), "no timestamp in the header");
        assert!(a.len() < data.len() / 10);
        let mut back = Vec::new();
        flate2::read::GzDecoder::new(&a[..]).read_to_end(&mut back).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn a_missing_file_names_its_path() {
        let e = read_gray(Path::new("no/such/file.png")).unwrap_err();
        assert!(e.0.contains("no/such/file.png"), "{e}");
    }
}
