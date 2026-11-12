//! PNG in memory: the encodings the answers serve (the command line writes the same pixels to files) and the
//! decoder of the Earth reference. Container I/O only: every pixel value comes from `cloud_map_generator_core::export`.

use crate::imageio::GrayImage;
use std::io::Cursor;

fn encode(width: usize, height: usize, color: png::ColorType, depth: png::BitDepth, bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() / 2 + 1024);
    let mut enc = png::Encoder::new(&mut out, u32::try_from(width).expect("a PNG width"), u32::try_from(height).expect("a PNG height"));
    enc.set_color(color);
    enc.set_depth(depth);
    let mut writer = enc.write_header().expect("a PNG header into memory");
    writer.write_image_data(bytes).expect("PNG pixels into memory");
    writer.finish().expect("a PNG into memory");
    out
}

/// A `height x width` 8-bit gray PNG.
pub fn gray8(width: usize, height: usize, data: &[u8]) -> Vec<u8> {
    assert_eq!(data.len(), width * height, "gray8: {} bytes for {height} x {width}", data.len());
    encode(width, height, png::ColorType::Grayscale, png::BitDepth::Eight, data)
}

/// A `height x width` 16-bit gray PNG (samples big-endian, as the format wants).
pub fn gray16(width: usize, height: usize, data: &[u16]) -> Vec<u8> {
    assert_eq!(data.len(), width * height, "gray16: {} samples for {height} x {width}", data.len());
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_be_bytes()).collect();
    encode(width, height, png::ColorType::Grayscale, png::BitDepth::Sixteen, &bytes)
}

/// A `height x width` 8-bit RGB PNG.
pub fn rgb8(width: usize, height: usize, data: &[u8]) -> Vec<u8> {
    assert_eq!(data.len(), width * height * 3, "rgb8: {} bytes for {height} x {width} x 3", data.len());
    encode(width, height, png::ColorType::Rgb, png::BitDepth::Eight, data)
}

/// The first channel of a gray or color PNG of 8 or 16 bits (`measure.load` through PIL and numpy), from bytes.
pub fn decode_gray(bytes: &[u8]) -> Result<GrayImage, String> {
    let mut dec = png::Decoder::new(Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND);
    let mut reader = dec.read_info().map_err(|e| format!("not a PNG: {e}"))?;
    if reader.info().color_type == png::ColorType::Indexed {
        return Err("a palette PNG is not a coverage map (expected gray)".to_string());
    }
    let mut buf = vec![0u8; reader.output_buffer_size().ok_or("image too large")?];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("cannot decode the PNG: {e}"))?;
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

    #[test]
    fn the_encoders_round_trip_through_the_decoder() {
        let d8: Vec<u8> = (0..6u8).map(|v| v * 50).collect();
        let g = decode_gray(&gray8(3, 2, &d8)).unwrap();
        assert_eq!((g.width, g.height, g.max), (3, 2, 255.0));
        assert_eq!(g.data, d8.iter().map(|&v| u16::from(v)).collect::<Vec<_>>());
        let d16 = [0u16, 1, 256, 65535, 32768, 12345];
        let g = decode_gray(&gray16(2, 3, &d16)).unwrap();
        assert_eq!((g.width, g.height, g.max), (2, 3, 65535.0));
        assert_eq!(g.data, d16);
        let g = decode_gray(&rgb8(2, 1, &[10, 20, 30, 40, 50, 60])).unwrap();
        assert_eq!(g.data, vec![10, 40], "a color image reads its first channel");
        assert!(gray8(2, 1, &[1, 2]).starts_with(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(decode_gray(b"not a png").is_err());
    }
}
