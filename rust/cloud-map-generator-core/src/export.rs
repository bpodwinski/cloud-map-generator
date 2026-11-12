//! Port of `cloudgen/export.py` (the coverage map files) and of the pure helpers of `bake.py` (the bake
//! entry's files, the seed of a key, the slug of a key).
//!
//! What is ported is the ENCODING, as pure functions on arrays: the 16-bit master, the 8-bit preview, the
//! shader's raw R8 layout (rows SOUTH-up), the high-share / type / wind byte images, the file names, the
//! sidecar text fields and the manifest record. The container I/O (PNG, gzip, JSON, the filesystem) is NOT
//! part of the core: a cloud-map-generator-cli crate wraps these results in the `png` and `flate2` crates. The golden
//! cases freeze the DECODED arrays that the Python functions put in their files, which is what these
//! functions return.
//!
//! Left out on purpose: `bake.library_body` (it needs `climate::BodyInputs`) and the I/O and orchestration of
//! `bake_one` and `main`.
//!
//! numpy semantics kept throughout: images are float arrays in the dtype of the input (`f32` or `f64`, the
//! multiplication by the scale runs in that precision: a Python int is a weak scalar), `np.round` is
//! half-to-even ([`round_ties_even`]), `np.clip(x, 0, 1)` saturates at both ends and lets NaN through, and
//! `astype(uint8 / uint16)` of an integral value in range is exact.

use crate::products::{encode_wind, WIND_STEP_MPS};
use num_traits::Float;

/// `export.write_map`: the 16-bit master is `round(clip(cov) * 65535)`.
pub const MASTER_SCALE: f64 = 65535.0;
/// `export.write_map`: the 8-bit preview (and the raw R8) is `round(clip(cov) * 255)`.
pub const PREVIEW_SCALE: f64 = 255.0;
/// `export.write_products`: the cloud type image is `round(clip(a) * 255)`.
pub const TYPE_SCALE: f64 = 255.0;
/// `export.write_products`: the high-share image is `round(clip(a) * 254)`; 255 means "none authored" in the
/// engine (`HIGH_SHARE_NONE`), so a share of 1.0 maps to 254.
pub const HIGH_SHARE_SCALE: f64 = 254.0;
/// `bake.SLICE_WIDTH`: the width of an entry's slice, high-share and type images.
pub const SLICE_WIDTH: usize = 2048;

fn cast<F: Float>(x: f64) -> F {
    F::from(x).expect("float conversion")
}

/// `np.round` (= `np.rint`) for a float of either precision: half-to-even, so `0.5 -> 0`, `1.5 -> 2`,
/// `2.5 -> 2`, `-0.5 -> -0.0`, `-1.5 -> -2`; the sign of a zero result is kept; NaN and infinities pass
/// through. (`f32::round_ties_even` and `f64::round_ties_even` are the same functions, but the standard
/// library has no generic one, and the number crate's `round` is half away from zero.)
pub fn round_ties_even<F: Float>(x: F) -> F {
    if !x.is_finite() {
        return x;
    }
    let half = cast::<F>(0.5);
    let t = x.trunc();
    // x - trunc(x) is exact: the fractional part of a float is representable.
    let d = (x - t).abs();
    if d < half {
        t
    } else if d > half {
        t + x.signum()
    } else if (t % (F::one() + F::one())) == F::zero() {
        t // a tie: the even neighbor is the truncated value when that is even...
    } else {
        t + x.signum() // ... and the other one when it is odd
    }
}

/// `np.clip(x, 0.0, 1.0)`: saturates at both ends, NaN passes through, `-0.0` stays `-0.0`.
fn clip01<F: Float>(x: F) -> F {
    if x < F::zero() {
        F::zero()
    } else if x > F::one() {
        F::one()
    } else {
        x
    }
}

/// `np.round(np.clip(x, 0, 1) * scale).astype(np.uint8)` for a `scale` of at most 255. The product runs in
/// the precision of `F`. NaN, which numpy's `astype` leaves to the platform (0 on x86-64 and arm64), gives 0.
fn clipped_u8<F: Float>(x: F, scale: f64) -> u8 {
    round_ties_even(clip01(x) * cast::<F>(scale)).to_u8().unwrap_or(0)
}

/// `np.round(x * 255).astype(np.uint8)` WITHOUT a clip, for the values a caller guarantees to be in
/// `[0, 1]` (`bake.write_entry`'s slice). numpy's cast of an out-of-range float is undefined behavior in C
/// (it wraps on x86-64); here it saturates, which only differs for an input that breaks that contract. NaN
/// gives 0, as above.
fn saturating_u8<F: Float>(x: F) -> u8 {
    if x.is_nan() || x <= F::zero() {
        0
    } else if x >= cast::<F>(255.0) {
        255
    } else {
        x.to_u8().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------------------------------------
// Images and planes

/// A borrowed float plane, row-major, north-up unless said otherwise: `height` rows of `width` texels
/// (a wind plane carries two interleaved components per texel, so its data holds `2 * height * width`).
#[derive(Clone, Copy, Debug)]
pub struct Plane<'a, F> {
    pub data: &'a [F],
    pub height: usize,
    pub width: usize,
}

impl<'a, F> Plane<'a, F> {
    pub fn new(data: &'a [F], height: usize, width: usize) -> Plane<'a, F> {
        Plane { data, height, width }
    }
    fn check(&self, channels: usize, what: &str) {
        assert_eq!(
            self.data.len(),
            self.height * self.width * channels,
            "{what}: {} values for {} rows of {} texels with {channels} channel(s)",
            self.data.len(),
            self.height,
            self.width
        );
    }
}

/// A decoded 8-bit image as the PNG would hold it: row-major, `channels` interleaved bytes per pixel
/// (1 = gray `L`, 3 = `RGB`), north-up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image8 {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------------------------------------
// export.write_map

/// The 16-bit master of `export.write_map`: `round(clip(cov, 0, 1) * 65535)` as `uint16`, same layout as
/// `cov` (north-up). The product `c * 65535` runs in the precision of `F` before the half-to-even rounding.
pub fn coverage_to_u16<F: Float + Send + Sync>(cov: &[F]) -> Vec<u16> {
    let scale = cast::<F>(MASTER_SCALE);
    crate::par::map(cov, |&v| round_ties_even(clip01(v) * scale).to_u16().unwrap_or(0))
}

/// The 8-bit preview of `export.write_map`: `round(clip(cov, 0, 1) * 255)` as `uint8`, north-up. The raw R8
/// is this image with its rows reversed ([`r8_south_up`]).
pub fn preview_u8<F: Float + Send + Sync>(cov: &[F]) -> Vec<u8> {
    crate::par::map(cov, |&v| clipped_u8(v, PREVIEW_SCALE))
}

/// The raw R8 layout of `export.write_map` and `bake.write_entry` (`np.ascontiguousarray(b8[::-1])`): the
/// rows of a north-up `height x width` byte image in reverse order, so row 0 is the SOUTH edge
/// (cloud_hires_coverage.ts's layout). The core returns the bytes; gzip is the caller's.
pub fn r8_south_up(b8: &[u8], height: usize, width: usize) -> Vec<u8> {
    assert_eq!(b8.len(), height * width, "r8_south_up: {} bytes for {height} x {width}", b8.len());
    let mut out = Vec::with_capacity(b8.len());
    for row in (0..height).rev() {
        out.extend_from_slice(&b8[row * width..(row + 1) * width]);
    }
    out
}

/// The file names `export.write_map` writes for a `width x height` coverage under a stem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapFiles {
    /// `<stem>_<w>x<h>_16bit.png`: the 16-bit grayscale master (the archive).
    pub master: String,
    /// `<stem>_<w>x<h>.png`: the 8-bit north-up preview.
    pub preview: String,
    /// `<stem>_<w>x<h>.r8.gz`: the raw R8, rows south-up, gzipped.
    pub r8gz: String,
    /// `<stem>.json`: the sidecar.
    pub meta: String,
}

/// The names of `export.write_map`'s `paths` (the file names only; the caller joins them with a directory).
pub fn map_files(stem: &str, width: usize, height: usize) -> MapFiles {
    MapFiles {
        master: format!("{stem}_{width}x{height}_16bit.png"),
        preview: format!("{stem}_{width}x{height}.png"),
        r8gz: format!("{stem}_{width}x{height}.r8.gz"),
        meta: format!("{stem}.json"),
    }
}

/// What `export.write_map` puts in its files, decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedMap {
    /// The 16-bit master, `height x width`, north-up.
    pub master16: Vec<u16>,
    /// The 8-bit preview, `height x width`, north-up.
    pub preview8: Vec<u8>,
    /// The raw R8 before gzip, `height x width`, rows south-up.
    pub r8_south_up: Vec<u8>,
    pub files: MapFiles,
}

/// `export.write_map` without the I/O: the three images of a `height x width` coverage and the file names.
pub fn encode_map<F: Float + Send + Sync>(cov: &[F], height: usize, width: usize, stem: &str) -> EncodedMap {
    Plane::new(cov, height, width).check(1, "encode_map");
    let preview8 = preview_u8(cov);
    EncodedMap {
        master16: coverage_to_u16(cov),
        r8_south_up: r8_south_up(&preview8, height, width),
        preview8,
        files: map_files(stem, width, height),
    }
}

// ---------------------------------------------------------------------------------------------------------
// export.write_products

/// The cloud type image of `export.write_products`: `round(clip(a, 0, 1) * 255)` as `uint8` (gray `L`).
pub fn type_bytes<F: Float + Send + Sync>(a: &[F]) -> Vec<u8> {
    crate::par::map(a, |&v| clipped_u8(v, TYPE_SCALE))
}

/// The high-cloud share image of `export.write_products`: `round(clip(a, 0, 1) * 254)` as `uint8` (gray
/// `L`): 255 is reserved for "none authored", so a share of 1.0 is 254.
pub fn high_share_bytes<F: Float + Send + Sync>(a: &[F]) -> Vec<u8> {
    crate::par::map(a, |&v| clipped_u8(v, HIGH_SHARE_SCALE))
}

/// The wind image of `export.write_products` and `bake.write_entry`: RGB bytes, an alias of
/// [`encode_wind`] (`products.encode_wind`), the `height x width x 2` east/north wind to `height x width x 3`.
pub fn wind_bytes<F: Float>(wind: &[F]) -> Vec<u8> {
    encode_wind(wind)
}

/// The derived maps `export.write_products` is given (the keys of its `extras` dict): any of them may be
/// absent. Each is on its own grid; the wind holds two interleaved components per texel.
#[derive(Clone, Copy, Debug)]
pub struct ProductExtras<'a, F> {
    pub type_: Option<Plane<'a, F>>,
    pub high_share: Option<Plane<'a, F>>,
    pub wind: Option<Plane<'a, F>>,
}

/// What `export.write_products` records in the sidecar under `meta['products']`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductsSidecar {
    /// `products['type']`.
    pub type_: &'static str,
    /// `products['high_share']`.
    pub high_share: &'static str,
    /// `products['wind']`: embeds [`WIND_STEP_MPS`].
    pub wind: String,
    /// `products['files']`: `(key, file name)` of the products present, in Python's order (type, high_share,
    /// wind).
    pub files: Vec<(&'static str, String)>,
}

/// What `export.write_products` puts in its files, decoded (an absent product stays `None`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedProducts {
    pub type_png: Option<Image8>,
    pub high_share_png: Option<Image8>,
    pub wind_png: Option<Image8>,
    pub sidecar: ProductsSidecar,
}

const PRODUCT_TYPE_TEXT: &str = "gray 8-bit, 0 = stratiform bottom type .. 255 = convective top type (weather map B)";
const PRODUCT_HIGH_SHARE_TEXT: &str = "gray 8-bit, share of cloud with tops above 440 hPa, 0..1 x 254 (255 = none)";

/// `export.write_products` without the I/O: the byte images of the products present in `extras`, named
/// `<stem>_<key>_<w>x<h>.png` (`<stem>_wind_<w>x<h>.png` for the wind), and the sidecar text.
///
/// The wind text embeds the step as Python prints a float (`0.5`); `{:?}` is the same for the values this
/// constant can take (a decimal with a fraction).
pub fn encode_products<F: Float + Send + Sync>(extras: &ProductExtras<F>, stem: &str) -> EncodedProducts {
    let mut files: Vec<(&'static str, String)> = Vec::new();
    let mut gray = |key: &'static str, plane: &Option<Plane<F>>, scale: f64| -> Option<Image8> {
        let a = (*plane)?;
        a.check(1, key);
        files.push((key, format!("{stem}_{key}_{}x{}.png", a.width, a.height)));
        let data = a.data.iter().map(|&v| clipped_u8(v, scale)).collect();
        Some(Image8 { height: a.height, width: a.width, channels: 1, data })
    };
    let type_png = gray("type", &extras.type_, TYPE_SCALE);
    let high_share_png = gray("high_share", &extras.high_share, HIGH_SHARE_SCALE);
    let wind_png = extras.wind.map(|w| {
        w.check(2, "wind");
        files.push(("wind", format!("{stem}_wind_{}x{}.png", w.width, w.height)));
        Image8 { height: w.height, width: w.width, channels: 3, data: wind_bytes(w.data) }
    });
    EncodedProducts {
        type_png,
        high_share_png,
        wind_png,
        sidecar: ProductsSidecar {
            type_: PRODUCT_TYPE_TEXT,
            high_share: PRODUCT_HIGH_SHARE_TEXT,
            wind: format!("RGB, R east / G north, m/s = (byte - 128) x {WIND_STEP_MPS:?}, B unused"),
            files,
        },
    }
}

// ---------------------------------------------------------------------------------------------------------
// bake.py helpers

/// numpy's `pairwise_sum`, the inner loop of `np.add.reduce` over a contiguous run of floats, in the precision
/// of `F`: fewer than 8 values are added one after the other onto `-0.0`; from 8 to 128 values, eight
/// accumulators take the values `i, i + 1, .., i + 7` of each group of eight, are combined as
/// `((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))`, and the `n % 8` values left over are added on one by
/// one; beyond 128 values the run is cut in two at `n / 2` rounded down to a multiple of 8, and the halves are
/// summed on their own and added. [`box_downsample`] needs it because `mean` adds a block's rows with it.
pub fn pairwise_sum<F: Float>(a: &[F]) -> F {
    let n = a.len();
    if n < 8 {
        // numpy starts at -0.0, so that a sum of negative zeros stays -0.0 (0.0 + -0.0 would be 0.0)
        a.iter().fold(F::neg_zero(), |res, &v| res + v)
    } else if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - n % 8 {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc = *acc + a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &a[i..] {
            res = res + v;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// The mean's division, as numpy does it: `true_divide(sum, rcount, out=sum, casting='unsafe')` with `rcount`
/// an `intp` count, which divides in float64 and casts back to the dtype of the sum. For a float32 sum that
/// equals the float32 division as long as `count` is exact in float32 (under 2^24), and differs from it beyond.
fn divide_by_count<F: Float>(sum: F, count: usize) -> F {
    cast(sum.to_f64().expect("float conversion") / count as f64)
}

/// `bake.box_downsample`: the mean over `f x f` blocks of a `h x w` map, with `f = w // width`. `a` is
/// returned unchanged when `f <= 1` (the map is already at or below the slice size: the engine resamples
/// it); otherwise the map is first cropped to `(h // f) * f` rows and `(w // f) * f` columns (the
/// remainder is dropped, not resampled), so the output is `h // f` rows of `w // f` columns, NOT
/// `width` wide unless `w` is a multiple of `width`. Returns `(data, rows, columns)`.
///
/// The block sums run in the precision of `F` (numpy's `mean` of a float32 map stays float32) and in numpy's
/// own order, which `.reshape(h // f, f, w // f, f).mean(axis=(1, 3))` fixes: each block row is summed with
/// [`pairwise_sum`] and the row subtotals are added one after the other onto a running sum that starts at
/// `+0.0`, i.e. `(a00 + a01) + (a10 + a11)` for `f = 2`, NOT one running sum over the block. The one exception
/// is a map exactly one block wide (`w == f`, which only `width == 1` gives): its block is contiguous, numpy
/// merges the two reduced axes into one loop, and the whole `f * f` block goes through one [`pairwise_sum`].
/// The sum is then divided by `f * f` in float64 and rounded back ([`divide_by_count`]). Every one of these is
/// verified bit for bit against numpy 2.5.4 (the unit tests below).
///
/// Panics when `width == 0` (Python: `ZeroDivisionError`) or `a.len() != h * w`.
pub fn box_downsample<F: Float + Send + Sync>(a: &[F], h: usize, w: usize, width: usize) -> (Vec<F>, usize, usize) {
    assert_eq!(a.len(), h * w, "box_downsample: {} values for {h} x {w}", a.len());
    assert!(width > 0, "box_downsample: width must be positive");
    let f = w / width;
    if f <= 1 {
        return (a.to_vec(), h, w);
    }
    let (ho, wo) = (h / f, w / f);
    // Every output texel sums its own block, so the rows of output texels are independent (and run in parallel).
    let mut out = vec![F::zero(); ho * wo];
    crate::par::for_each_chunk_mut(&mut out, wo.max(1), |i, row_out| {
        for (j, slot) in row_out.iter_mut().enumerate() {
            let mut sum = F::zero();
            if w == f {
                // one block wide: rows i * f .. (i + 1) * f of a map f wide are one contiguous run of f * f
                let start = i * f * w;
                sum = sum + pairwise_sum(&a[start..start + f * f]);
            } else {
                for p in 0..f {
                    let row = (i * f + p) * w + j * f;
                    sum = sum + pairwise_sum(&a[row..row + f]);
                }
            }
            *slot = divide_by_count(sum, f * f);
        }
    });
    (out, ho, wo)
}

const CRC32_TABLE: [u32; 256] = crc32_table();

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

/// `zlib.crc32(data)`: the CRC-32 of the IEEE 802.3 standard (reflected polynomial 0xEDB88320, initial value
/// and final xor 0xFFFFFFFF), table-driven. `crc32(b"123456789") == 0xCBF43926`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFF_u32;
    for &b in data {
        c = CRC32_TABLE[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// `bake.key_seed`: the seed of a body key, `zlib.crc32(key.encode('utf-8')) & 0x7FFFFFFF` (31 bits, so it
/// fits every seed type). Rust strings are UTF-8, so the bytes are the key's own.
pub fn key_seed(key: &str) -> u32 {
    crc32(key.as_bytes()) & 0x7FFF_FFFF
}

/// `bake.slug`: `key.replace(':', '_').replace(' ', '_').lower()`. `to_lowercase` is Unicode's full lowercase
/// like `str.lower` (the context-sensitive final sigma, `"İ" -> "i" + U+0307`, `"ß"` unchanged); the two
/// differ only for characters whose lowercase mapping is newer than Python 3.12's Unicode 15.0 tables.
pub fn slug(key: &str) -> String {
    key.replace([':', ' '], "_").to_lowercase()
}

/// The `files` dict of `bake.write_entry`: the file names of one bake entry (all under the entry's
/// directory; the manifest record prefixes them with the URL prefix).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryFiles {
    /// `hires`: `<stem>_coverage_<w>x<h>.r8.gz`.
    pub hires: String,
    /// `slice`: `<stem>_coverage_<slice width>.png`.
    pub slice: String,
    /// `highShare`: `<stem>_high_share_<slice width>.png`.
    pub high_share: String,
    /// `type`: `<stem>_type_<slice width>.png`.
    pub type_: String,
    /// `wind`: `<stem>_wind_<wind w>x<wind h>.png`.
    pub wind: String,
    /// `meta`: `<stem>.json`.
    pub meta: String,
}

impl EntryFiles {
    /// The Python dict's `(key, value)` pairs, in its order (`hires`, `slice`, `highShare`, `type`, `wind`,
    /// `meta`).
    pub fn pairs(&self) -> [(&'static str, &str); 6] {
        [
            ("hires", &self.hires),
            ("slice", &self.slice),
            ("highShare", &self.high_share),
            ("type", &self.type_),
            ("wind", &self.wind),
            ("meta", &self.meta),
        ]
    }
}

/// `bake.write_entry`'s `meta['encoding']`: the text recorded in an entry's sidecar for each file, `(key,
/// text)` in Python's order.
pub const ENTRY_ENCODING: [(&str, &str); 5] = [
    ("hires", "R8 cloud fraction 0..255, rows SOUTH-up, gzip"),
    ("slice", "gray 8-bit fraction, north-up, box downsample of hires"),
    ("highShare", "gray 8-bit, conditional share above 440 hPa, 0..254 = 0..1 (255 reserved: none)"),
    ("type", "gray 8-bit, 0 stratiform .. 255 convective"),
    ("wind", "RGB, R east / G north, m/s = (byte - 128) x 0.5"),
];

/// The `hires` field of a manifest record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HiresRecord {
    pub url: String,
    pub width: usize,
    pub height: usize,
}

/// The manifest record `bake.write_entry` returns: the seed and the URLs (`<prefix>/<file name>`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManifestRecord {
    pub seed: u64,
    pub hires: HiresRecord,
    pub slice: String,
    pub high_share: String,
    pub type_: String,
    pub wind: String,
}

/// The derived maps `bake.write_entry` is given (its `extras['high_share' | 'type' | 'wind']`), each on its
/// own grid (the wind holds two interleaved components per texel).
#[derive(Clone, Copy, Debug)]
pub struct EntryExtras<'a, F> {
    pub high_share: Plane<'a, F>,
    pub type_: Plane<'a, F>,
    pub wind: Plane<'a, F>,
}

/// What `bake.write_entry` puts in its files, decoded, and its manifest record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedEntry {
    /// `hires` before gzip: `round(clip(cov) * 255)`, `height x width`, rows SOUTH-up.
    pub hires_south_up: Vec<u8>,
    /// `slice`: `round(box_downsample(cov, slice_width) * 255)`, gray, north-up.
    pub slice: Image8,
    /// `highShare`: `round(clip(box_downsample(high_share, slice_width)) * 254)`, gray, north-up.
    pub high_share: Image8,
    /// `type`: `round(clip(box_downsample(type, slice_width)) * 255)`, gray, north-up.
    pub type_: Image8,
    /// `wind`: [`encode_wind`] at the wind's own grid, RGB, north-up.
    pub wind: Image8,
    pub files: EntryFiles,
    pub record: ManifestRecord,
}

/// `bake.write_entry` without the I/O: the decoded files of one bake entry and its manifest record, whose
/// paths are `<prefix>/<file name>`. `slice_width` is `bake.SLICE_WIDTH` ([`SLICE_WIDTH`], 2048; the golden
/// cases also run it at 16, since a real 2048 slice needs a map wider than 4096): a parameter here because
/// Python reads a module global.
///
/// The slice is NOT clipped before the quantization (Python's `np.round(box * 255)`): `cov` must be in
/// `[0, 1]`, as the generator's output is; out-of-range values saturate here (`saturating_u8`) where numpy's
/// cast is undefined. The high-share and type images are clipped. A map narrower than the slice goes through
/// [`box_downsample`]'s passthrough, so the three images are the maps themselves.
///
/// Not returned: the sidecar JSON (`meta` plus `files` and `encoding`, built by the caller from
/// [`EncodedEntry::files`] and [`ENTRY_ENCODING`]).
pub fn encode_entry<F: Float + Send + Sync>(
    cov: Plane<F>,
    extras: &EntryExtras<F>,
    stem: &str,
    prefix: &str,
    seed: u64,
    slice_width: usize,
) -> EncodedEntry {
    cov.check(1, "cov");
    extras.high_share.check(1, "extras.high_share");
    extras.type_.check(1, "extras.type");
    extras.wind.check(2, "extras.wind");
    let (h, w) = (cov.height, cov.width);
    let hires_south_up = r8_south_up(&preview_u8(cov.data), h, w);

    let (sl, sl_h, sl_w) = box_downsample(cov.data, h, w, slice_width);
    let scale255 = cast::<F>(PREVIEW_SCALE);
    let slice = Image8 {
        height: sl_h,
        width: sl_w,
        channels: 1,
        data: sl.iter().map(|&v| saturating_u8(round_ties_even(v * scale255))).collect(),
    };
    let hs_plane = extras.high_share;
    let (hs, hs_h, hs_w) = box_downsample(hs_plane.data, hs_plane.height, hs_plane.width, slice_width);
    let high_share = Image8 { height: hs_h, width: hs_w, channels: 1, data: high_share_bytes(&hs) };
    let ty_plane = extras.type_;
    let (ty, ty_h, ty_w) = box_downsample(ty_plane.data, ty_plane.height, ty_plane.width, slice_width);
    let type_ = Image8 { height: ty_h, width: ty_w, channels: 1, data: type_bytes(&ty) };
    let wnd = extras.wind;
    let wind = Image8 { height: wnd.height, width: wnd.width, channels: 3, data: wind_bytes(wnd.data) };

    let files = EntryFiles {
        hires: format!("{stem}_coverage_{w}x{h}.r8.gz"),
        slice: format!("{stem}_coverage_{slice_width}.png"),
        high_share: format!("{stem}_high_share_{slice_width}.png"),
        type_: format!("{stem}_type_{slice_width}.png"),
        wind: format!("{stem}_wind_{}x{}.png", wnd.width, wnd.height),
        meta: format!("{stem}.json"),
    };
    let record = ManifestRecord {
        seed,
        hires: HiresRecord { url: format!("{prefix}/{}", files.hires), width: w, height: h },
        slice: format!("{prefix}/{}", files.slice),
        high_share: format!("{prefix}/{}", files.high_share),
        type_: format!("{prefix}/{}", files.type_),
        wind: format!("{prefix}/{}", files.wind),
    };
    EncodedEntry { hires_south_up, slice, high_share, type_, wind, files, record }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_ties_even_matches_the_standard_library() {
        let mut xs: Vec<f64> = vec![
            0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.49999999999999994, -0.49999999999999994, 4503599627370495.5,
            4503599627370497.0, 1e300, -1e300, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 127.5, 128.5, 254.5, 255.5,
            0.1, -0.1, 3.7, -3.7, 1e-300,
        ];
        for k in -300..300 {
            xs.push(f64::from(k) * 0.25);
            xs.push(f64::from(k) * 0.5 + 1e-12);
            xs.push(f64::from(k) * 0.5 - 1e-12);
        }
        for &x in &xs {
            let (g, w) = (round_ties_even(x), x.round_ties_even());
            assert!(g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()), "f64 {x}: {g} vs {w}");
            let y = x as f32;
            let (g, w) = (round_ties_even(y), y.round_ties_even());
            assert!(g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()), "f32 {y}: {g} vs {w}");
        }
    }

    #[test]
    fn quantizers_tie_to_even_clip_and_saturate() {
        // 0.5 x 255 = 127.5 is a tie of the 8-bit scale (even: 128); 0.5 x 65535 = 32767.5 of the 16-bit one.
        let cov = [0.5_f64, -3.0, 7.0, f64::INFINITY, f64::NEG_INFINITY, -0.0, 0.0, 1.0];
        assert_eq!(preview_u8(&cov), vec![128, 0, 255, 255, 0, 0, 0, 255]);
        assert_eq!(coverage_to_u16(&[0.5_f64, 0.25, 0.75, 2.0, -1.0, 1.0]), vec![32768, 16384, 49151, 65535, 0, 65535]);
        // 1.0 maps to 254, not 255, for the high-cloud share; 0.25 x 254 = 63.5 -> 64, 0.75 x 254 = 190.5 -> 190.
        assert_eq!(high_share_bytes(&[1.0_f32, 0.25, 0.75, 0.0, 2.0]), vec![254, 64, 190, 0, 254]);
        assert_eq!(type_bytes(&[1.0_f32, 0.5, 0.0]), vec![255, 128, 0]);
        // NaN: numpy's astype is platform-defined (0); so is ours.
        assert_eq!(preview_u8(&[f32::NAN]), vec![0]);
        assert_eq!(coverage_to_u16(&[f32::NAN]), vec![0]);
    }

    #[test]
    fn the_product_runs_in_the_precision_of_the_input() {
        // The float64 just under 0.5: x 255 is 127.5 minus an ulp, which rounds down to 127; the same number
        // as a float32 IS 0.5, whose product is the exact tie 127.5, which goes to the even 128.
        let below = 0.5_f64.next_down();
        assert_eq!(preview_u8(&[below]), vec![127]);
        assert_eq!(preview_u8(&[below as f32]), vec![128]);
    }

    #[test]
    fn r8_is_the_preview_upside_down() {
        let b8 = [1, 2, 3, 4, 5, 6];
        assert_eq!(r8_south_up(&b8, 3, 2), vec![5, 6, 3, 4, 1, 2]);
        assert_eq!(r8_south_up(&b8, 1, 6), b8.to_vec());
        assert_eq!(r8_south_up(&[], 0, 4), Vec::<u8>::new());
    }

    #[test]
    fn map_names_follow_write_map() {
        let m = encode_map(&[0.0_f32, 1.0, 0.5, 0.25], 2, 2, "earth");
        assert_eq!(m.files.master, "earth_2x2_16bit.png");
        assert_eq!(m.files.preview, "earth_2x2.png");
        assert_eq!(m.files.r8gz, "earth_2x2.r8.gz");
        assert_eq!(m.files.meta, "earth.json");
        assert_eq!(m.preview8, vec![0, 255, 128, 64]);
        assert_eq!(m.r8_south_up, vec![128, 64, 0, 255]);
        assert_eq!(m.master16, vec![0, 65535, 32768, 16384]);
        // a non-square map names width x height, in every file
        let f = map_files("s", 4096, 2048);
        assert_eq!(f.master, "s_4096x2048_16bit.png");
        assert_eq!(f.preview, "s_4096x2048.png");
        assert_eq!(f.r8gz, "s_4096x2048.r8.gz");
        // and the rows of a 3-row map come back reversed in the R8 only
        let m = encode_map(&[0.0_f32, 0.0, 1.0, 1.0, 0.5, 0.5], 3, 2, "t");
        assert_eq!(m.preview8, vec![0, 0, 255, 255, 128, 128]);
        assert_eq!(m.r8_south_up, vec![128, 128, 255, 255, 0, 0]);
        assert_eq!(m.files.preview, "t_2x3.png");
    }

    #[test]
    #[should_panic]
    fn a_plane_of_the_wrong_size_is_rejected() {
        let t = [0.0_f32; 3];
        let extras = ProductExtras { type_: Some(Plane::new(&t, 2, 2)), high_share: None, wind: None };
        encode_products(&extras, "x");
    }

    #[test]
    fn box_downsample_means_blocks_crops_and_passes_through() {
        // 4 x 6 map, width 3: f = 2; blocks of 2 x 2 -> 2 x 3.
        let a: Vec<f64> = (0..24).map(f64::from).collect();
        let (o, h, w) = box_downsample(&a, 4, 6, 3);
        assert_eq!((h, w), (2, 3));
        assert_eq!(o, vec![3.5, 5.5, 7.5, 15.5, 17.5, 19.5]);
        // width 4 on a 5 x 7 map: f = 1, passthrough (not even cropped); width 2: f = 3, 1 x 2 blocks of 3 x 3.
        let b: Vec<f32> = (0..35).map(|v| v as f32).collect();
        let (o, h, w) = box_downsample(&b, 5, 7, 4);
        assert_eq!((h, w, o), (5, 7, b.clone()));
        let (o, h, w) = box_downsample(&b, 5, 7, 2);
        assert_eq!((h, w), (1, 2)); // rows 0..3, columns 0..3 and 3..6; row 3.. and column 6 are dropped
        assert_eq!(o, vec![8.0, 11.0]);
        // wider than the map: f = 0, passthrough too
        let (o, h, w) = box_downsample(&b, 5, 7, 64);
        assert_eq!((h, w, o.len()), (5, 7, 35));
        // fewer rows than the factor: no rows at all
        let (o, h, w) = box_downsample(&a[..6], 1, 6, 2);
        assert_eq!((h, w, o.len()), (0, 2, 0));
    }

    #[test]
    #[should_panic]
    fn a_zero_width_is_rejected() {
        box_downsample(&[1.0_f32; 4], 2, 2, 0);
    }

    // ---- box_downsample: numpy's order of summation -------------------------------------------------------
    //
    // numpy's `a.reshape(H, f, W, f).mean(axis=(1, 3))` adds each block ROW first (a sequential sum for fewer
    // than 8 values, `pairwise_sum` with 8 accumulators from 8 on) and then adds the row subtotals one after
    // the other onto a running sum that starts at +0.0; one running sum over the whole block gives other bits.
    // The expected values below are numpy 2.5.4's output on the same inputs (not the port's own).

    /// A float whose bit pattern the tests can print and compare.
    trait Bits: Float {
        fn le_bytes(self) -> Vec<u8>;
        fn bits(self) -> u64;
    }
    impl Bits for f32 {
        fn le_bytes(self) -> Vec<u8> {
            self.to_bits().to_le_bytes().to_vec()
        }
        fn bits(self) -> u64 {
            u64::from(self.to_bits())
        }
    }
    impl Bits for f64 {
        fn le_bytes(self) -> Vec<u8> {
            self.to_bits().to_le_bytes().to_vec()
        }
        fn bits(self) -> u64 {
            self.to_bits()
        }
    }

    fn bits_of<F: Bits>(v: &[F]) -> Vec<u64> {
        v.iter().map(|&x| x.bits()).collect()
    }

    /// `n` pseudo-random values in `[-1, 2)`: two xorshift32 draws make 53 random bits (exact in float64), mapped
    /// as `x * 3 - 1` in float64 and rounded once to `F`, so a float64 value fills its whole mantissa. The same
    /// stream is generated in Python to pin the bits.
    fn stream<F: Float>(seed: u32, n: usize) -> Vec<F> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            u64::from(s)
        };
        (0..n)
            .map(|_| {
                let (hi, lo) = (next() >> 5, next() >> 6); // 27 and 26 bits
                let x = ((hi << 26) | lo) as f64 / 9007199254740992.0;
                F::from(x * 3.0 - 1.0).expect("float conversion")
            })
            .collect()
    }

    /// FNV-1a (64 bits) over the little-endian bytes of the values.
    fn fnv1a<F: Bits>(v: &[F]) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325_u64;
        for &x in v {
            for b in x.le_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        h
    }

    /// An `f x f` block tiled twice side by side: an `f x 2f` map, which `box_downsample(.., 2)` cuts back
    /// into the same two blocks (two blocks wide: the reduction does not coalesce into one loop).
    fn twice_wide<F: Float>(block: &[F], f: usize) -> Vec<F> {
        let mut a = Vec::new();
        for row in block.chunks_exact(f) {
            a.extend_from_slice(row);
            a.extend_from_slice(row);
        }
        a
    }

    // (height, width, slice width, FNV-1a of numpy's output bits) of `box_downsample` on `stream(1000 + k)`,
    // k the index in the table: f = 2, 3, 4, 8, 16 (8 accumulators, and with a remainder), cropped maps, maps
    // one block wide (slice width 1: numpy coalesces the block into one sum), f = 130 and 129 (the pairwise
    // recursion). Python: `a = stream(1000 + k).reshape(h, w); a[:h//f*f, :w//f*f].reshape(h//f, f, w//f,
    // f).mean(axis=(1, 3))`, f = w // width, hashed with the same FNV-1a.
    const PINS_F32: [(usize, usize, usize, u64); 13] = [
        (8, 12, 6, 0xB417_0891_9D44_AB1F),
        (9, 12, 4, 0x2298_565C_9221_5377),
        (16, 20, 5, 0x1EF0_285F_3590_08A6),
        (16, 24, 3, 0x09EA_D780_87C5_FE53),
        (32, 48, 3, 0x2472_88A0_DAD4_D4BA),
        (7, 11, 3, 0x49D4_2731_7B2C_C1C1),
        (17, 23, 5, 0x8D45_AD3C_F2D3_C17C),
        (16, 8, 1, 0x5114_CEE4_4710_79ED),
        (10, 5, 1, 0x48CF_D9C6_6AB6_2B31),
        (40, 40, 5, 0xA269_D39F_3D01_A7E9),
        (130, 260, 2, 0xFBDD_4CE0_A8F4_1FF2),
        (129, 129, 1, 0xB738_12F1_8C08_D556),
        (23, 40, 2, 0x4063_2203_FFE7_CBB4),
    ];
    const PINS_F64: [(usize, usize, usize, u64); 13] = [
        (8, 12, 6, 0xF7CD_B8F6_D262_4569),
        (9, 12, 4, 0x95D8_F0D1_CCF5_35AF),
        (16, 20, 5, 0xFB82_3A22_DB45_7019),
        (16, 24, 3, 0x20CB_9F08_B6FA_3B1D),
        (32, 48, 3, 0xE98A_624B_9437_F6C4),
        (7, 11, 3, 0x2319_08DA_0DE4_E627),
        (17, 23, 5, 0x4547_9237_6C20_8DB4),
        (16, 8, 1, 0x6A6E_7CFD_5B0E_2904),
        (10, 5, 1, 0x3877_332B_B616_7F71),
        (40, 40, 5, 0xDA52_EB1D_562B_E483),
        (130, 260, 2, 0x3E6C_14A2_3463_C2E3),
        (129, 129, 1, 0x44ED_9248_7DC6_A3E8),
        (23, 40, 2, 0x59E0_1603_DB37_917A),
    ];

    fn check_pins<F: Bits + Send + Sync>(pins: &[(usize, usize, usize, u64)]) {
        for (k, &(h, w, width, want)) in pins.iter().enumerate() {
            let a = stream::<F>(1000 + k as u32, h * w);
            let (o, oh, ow) = box_downsample(&a, h, w, width);
            let f = w / width;
            assert_eq!((oh, ow), (h / f, w / f), "case {k}: the shape");
            assert_eq!(fnv1a(&o), want, "case {k}: {h} x {w}, slice width {width}, f = {f}");
        }
    }

    #[test]
    fn box_downsample_matches_numpys_mean_bit_for_bit_in_float32() {
        check_pins::<f32>(&PINS_F32);
    }

    #[test]
    fn box_downsample_matches_numpys_mean_bit_for_bit_in_float64() {
        check_pins::<f64>(&PINS_F64);
    }

    #[test]
    fn box_downsample_adds_the_row_subtotals_of_each_block() {
        // e is half an ulp of 1.0: 1 + e is a tie (even: 1.0), so the order of the additions shows in the bits.
        let e = 2.0_f32.powi(-24);
        // f = 2: the rows (1 + e = 1.0) and (e + e = 2^-23) added: 1 + 2^-23, over 4. One block wide, numpy
        //   coalesces the 4 values into one sequential sum, 1.0.
        // f = 3: the rows 1.0, 3e and 0; the sum 1 + 3e is a tie that goes to 1 + 4e.
        // f = 4: the rows 1, 2e, 2e, 2e. f = 8: one row summed by the 8 accumulators, (1 + e) + 2e + (2e + 2e).
        let f2 = [1.0, e, e, e];
        let f3 = [1.0, e, e, e, e, e, 0.0, 0.0, 0.0];
        let mut f4 = [0.0_f32; 16];
        f4[0] = 1.0;
        for r in 1..4 {
            f4[r * 4] = e;
            f4[r * 4 + 1] = e;
        }
        let mut f8 = [0.0_f32; 64];
        f8[0] = 1.0;
        f8[1..8].fill(e);
        // (block, f, numpy's mean bits for two blocks side by side, numpy's for a map one block wide)
        let blocks: [(&[f32], usize, u32, u32); 4] = [
            (&f2, 2, 0x3E80_0001, 0x3E80_0000),
            (&f3, 3, 0x3DE3_8E3C, 0x3DE3_8E3C),
            (&f4, 4, 0x3D80_0003, 0x3D80_0002),
            (&f8, 8, 0x3C80_0003, 0x3C80_0003),
        ];
        for (block, f, wide, narrow) in blocks {
            let (o, h, w) = box_downsample(&twice_wide(block, f), f, 2 * f, 2);
            assert_eq!((h, w), (1, 2), "f = {f}");
            assert_eq!(bits_of(&o), vec![u64::from(wide); 2], "f = {f}, two blocks wide");
            let (o, h, w) = box_downsample(block, f, f, 1);
            assert_eq!((h, w), (1, 1), "f = {f}");
            assert_eq!(bits_of(&o), vec![u64::from(narrow)], "f = {f}, one block wide");
        }
        // float64 (e = half an ulp of 1.0 again): the same f = 2 block, numpy's 0.25 + 2^-54.
        let e = 2.0_f64.powi(-53);
        let (o, ..) = box_downsample(&twice_wide(&[1.0, e, e, e], 2), 2, 4, 2);
        assert_eq!(bits_of(&o), vec![0x3FD0_0000_0000_0001; 2]);
    }

    #[test]
    fn pairwise_sum_is_numpys_inner_loop() {
        let e = 2.0_f32.powi(-24);
        // fewer than 8 values: a sequential sum (1 + e is a tie that stays 1.0, three times)
        assert_eq!(pairwise_sum(&[1.0, e, e, e]), 1.0);
        assert_eq!(pairwise_sum(&[1.0, e, e, e, e, e, e]), 1.0);
        // 8 values: eight accumulators and a tree, ((1 + e) + 2e) + (2e + 2e) = 1 + 3 x 2^-23
        let eight = [1.0, e, e, e, e, e, e, e];
        assert_eq!(pairwise_sum(&eight), 1.0 + 3.0 * 2.0_f32.powi(-23));
        // 16 values: the second group of eight lands on the accumulators (r0 = 1 + e = 1.0, r1.. = 2e), and
        // the tree adds them as ((1.0 + 2e) + (2e + 2e)) + ((2e + 2e) + (2e + 2e)) = 1 + 7 x 2^-23
        let sixteen = [1.0, e, e, e, e, e, e, e, e, e, e, e, e, e, e, e];
        assert_eq!(pairwise_sum(&sixteen), 1.0 + 7.0 * 2.0_f32.powi(-23));
        // 9 values: the tree of the first 8, then the remainder added on (1 + 3 x 2^-23 + e is a tie: even)
        let nine = [1.0, e, e, e, e, e, e, e, e];
        assert_eq!(pairwise_sum(&nine), 1.0 + 4.0 * 2.0_f32.powi(-23));
        // more than 128: cut at 64 (130 / 2 = 65, down to a multiple of 8), 1.0 on its own in the first
        // half, 66 values of e in the second: 66e = 33 x 2^-23, exact, then added to 1.0
        let mut big = vec![0.0_f32; 130];
        big[0] = 1.0;
        big[64..].fill(e);
        assert_eq!(pairwise_sum(&big), 1.0 + 33.0 * 2.0_f32.powi(-23));
        // the empty sum and a sum of negative zeros are -0.0 (numpy starts at -0.0), any other zero sum +0.0
        assert_eq!(pairwise_sum::<f64>(&[]).to_bits(), (-0.0_f64).to_bits());
        assert_eq!(pairwise_sum(&[-0.0_f32, -0.0]).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(pairwise_sum(&[0.0_f32, -0.0]).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn pairwise_sum_matches_numpys_sum_bit_for_bit() {
        // `np.add.reduce` of `stream(2000 + k, n)` (numpy 2.5.4), (n, float32 bits, float64 bits): n = 7 and 8
        // (the switch to the accumulators), 128 and 129 (the switch to the recursion), the odd lengths and the
        // deeper recursion of 255 .. 4099.
        let table: [(usize, u32, u64); 21] = [
        (1, 0xBF20_CF2A, 0xBFE4_19E5_3A11_01E7),
        (2, 0xBF94_DA1A, 0xBFF2_9B43_45D6_7C8A),
        (7, 0x4060_7DAE, 0x400C_0FB5_DBFF_8C99),
        (8, 0x3FBE_6DC1, 0x3FF7_CDB7_FF66_FA9D),
        (9, 0x40D3_08A6, 0x401A_6114_BB6A_DF88),
        (15, 0x4117_2CC2, 0x4022_E598_42B8_CE7E),
        (16, 0x40DD_8C7B, 0x401B_B18F_716B_70DE),
        (17, 0x40E0_1241, 0x401C_0248_1EB5_6CFE),
        (63, 0x41E8_17B2, 0x403D_02F5_FA5D_3FB7),
        (64, 0x4195_AFFA, 0x4032_B5FF_57F5_03EA),
        (65, 0x419A_2FFE, 0x4033_45FF_C617_0FE7),
        (127, 0x4287_37E4, 0x4050_E6FC_8411_A842),
        (128, 0x4287_F91A, 0x4050_FF23_3BA1_C770),
        (129, 0x427D_B603, 0x404F_B6C0_3FC1_66B2),
        (130, 0x4260_767E, 0x404C_0ECF_C004_EF20),
        (131, 0x4282_4A0C, 0x4050_4941_9388_77D5),
        (255, 0x4300_1708, 0x4060_02E0_F37A_B866),
        (256, 0x42F0_20B9, 0x405E_0417_2623_506F),
        (257, 0x4302_C9C6, 0x4060_5938_CEA9_81A6),
        (1000, 0x43F7_8416, 0x407E_F082_BA44_7708),
        (4099, 0x44FC_056C, 0x409F_80AD_768C_1226),
        ];
        for (k, &(n, b32, b64)) in table.iter().enumerate() {
            let seed = 2000 + k as u32;
            assert_eq!(pairwise_sum(&stream::<f32>(seed, n)).to_bits(), b32, "f32, n = {n}");
            assert_eq!(pairwise_sum(&stream::<f64>(seed, n)).to_bits(), b64, "f64, n = {n}");
        }
    }

    #[test]
    fn a_map_at_or_below_the_slice_width_is_returned_as_it_is() {
        // f = 1: numpy returns `a` itself, so a negative zero survives (the box path would give +0.0)
        let (o, h, w) = box_downsample(&[-0.0_f32, 1.0, f32::NAN, -2.5], 2, 2, 2);
        assert_eq!((h, w), (2, 2));
        assert_eq!(o[0].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(o[1], 1.0);
        assert!(o[2].is_nan());
        assert_eq!(o[3], -2.5);
    }

    #[test]
    fn the_mean_is_divided_in_float64_like_numpys_true_divide() {
        // numpy divides a float32 sum by the intp count in float64 and casts back. For a count float32 cannot
        // hold (4097 x 4097 = 16785409 is odd and over 2^24) that is not a float32 division: numpy gives
        // 0x3F18836F for 1e7 / 16785409, the float32 division 0x3F188370.
        let sum = 1e7_f32;
        let count = 4097 * 4097;
        assert_eq!(divide_by_count(sum, count).to_bits(), 0x3F18_836F);
        assert_ne!((sum / count as f32).to_bits(), 0x3F18_836F, "this count must tell the two divisions apart");
        // and a count float32 holds gives the same either way
        assert_eq!(divide_by_count(7.0_f32, 4), 1.75);
        assert_eq!(divide_by_count(1.0_f32, 3).to_bits(), (1.0_f32 / 3.0).to_bits());
    }

    #[test]
    fn box_downsample_of_negative_zeros_is_positive_zero() {
        // numpy's reduction starts at +0.0, and +0.0 + -0.0 is +0.0 (one block wide or not).
        for width in [1, 2] {
            let (o, ..) = box_downsample(&[-0.0_f32; 16], 4, 4, width);
            assert!(o.iter().all(|v| v.to_bits() == 0.0_f32.to_bits()), "width {width}");
        }
    }

    #[test]
    fn the_slice_width_is_the_engines() {
        assert_eq!(SLICE_WIDTH, 2048); // bake.SLICE_WIDTH
    }

    #[test]
    fn crc32_is_the_standard_one() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b"The quick brown fox jumps over the lazy dog"), 0x414F_A339);
        // a key whose crc has the top bit set loses it in the seed
        assert_eq!(key_seed("123456789"), 0xCBF4_3926 & 0x7FFF_FFFF);
        assert_eq!(key_seed(""), 0);
    }

    #[test]
    fn slug_replaces_then_lowercases() {
        assert_eq!(slug("Sol:Earth Moon"), "sol_earth_moon");
        assert_eq!(slug("a b:C D"), "a_b_c_d");
        assert_eq!(slug("::"), "__");
        assert_eq!(slug("ΑΣ"), "ας"); // final sigma
        assert_eq!(slug("İstanbul"), "i\u{307}stanbul");
        assert_eq!(slug("Straße"), "straße");
        assert_eq!(slug("with\ttab"), "with\ttab"); // a tab is not a space
    }

    #[test]
    fn products_name_their_files_and_skip_the_absent() {
        let t = [0.0_f32, 1.0, 0.5, 2.0];
        let wnd = [0.0_f32; 12];
        let e = encode_products(
            &ProductExtras { type_: Some(Plane::new(&t, 2, 2)), high_share: None, wind: Some(Plane::new(&wnd, 2, 3)) },
            "x",
        );
        assert_eq!(e.type_png, Some(Image8 { height: 2, width: 2, channels: 1, data: vec![0, 255, 128, 255] }));
        assert!(e.high_share_png.is_none());
        let w = e.wind_png.unwrap();
        assert_eq!((w.height, w.width, w.channels, w.data.len()), (2, 3, 3, 18));
        assert_eq!(e.sidecar.files, vec![("type", "x_type_2x2.png".to_string()), ("wind", "x_wind_3x2.png".to_string())]);
        assert_eq!(e.sidecar.wind, "RGB, R east / G north, m/s = (byte - 128) x 0.5, B unused");
    }

    #[test]
    fn an_entry_names_its_files_after_the_slice_width_and_the_grids() {
        let cov = vec![0.5_f32; 8 * 16];
        let hs = vec![0.5_f32; 8 * 16];
        let ty = vec![1.0_f32; 8 * 16];
        let wnd = vec![0.0_f32; 4 * 6 * 2];
        let ex = EntryExtras {
            high_share: Plane::new(&hs, 8, 16),
            type_: Plane::new(&ty, 8, 16),
            wind: Plane::new(&wnd, 4, 6),
        };
        let e = encode_entry(Plane::new(&cov, 8, 16), &ex, "k", "baked/v9", 5, 8);
        assert_eq!(e.files.hires, "k_coverage_16x8.r8.gz");
        assert_eq!(e.files.slice, "k_coverage_8.png");
        assert_eq!(e.files.wind, "k_wind_6x4.png");
        assert_eq!(e.record.hires, HiresRecord { url: "baked/v9/k_coverage_16x8.r8.gz".into(), width: 16, height: 8 });
        assert_eq!(e.record.seed, 5);
        assert_eq!((e.slice.height, e.slice.width), (4, 8));
        assert_eq!(e.slice.data, vec![128; 32]); // 0.5 x 255 = 127.5: the tie goes to the even 128
        assert_eq!(e.high_share.data, vec![127; 32]); // 0.5 x 254 = 127
        assert_eq!(e.type_.data, vec![255; 32]);
        assert_eq!(e.files.pairs()[2], ("highShare", "k_high_share_8.png"));
    }
}
