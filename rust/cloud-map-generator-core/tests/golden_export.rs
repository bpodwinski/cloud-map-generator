//! Golden cases of cloudgen/export.py, the pure helpers of bake.py and products.encode_wind (group
//! pipeline). The cases freeze the DECODED arrays that write_map, write_products and write_entry put in
//! their files; the glue feeds the case arrays through the port's encoders. The text fields the cases keep
//! in their params (the sidecar, the manifest record, the file names) are checked here against what the
//! port builds: a mismatch panics, which the runner reports as a failure of the case.
//!
//! What the cases cannot see is pinned by the tests after the runner's: the golden inputs of
//! `bake.box_downsample` are dyadic or compared at rtol 1e-6 / 1e-12, so they accept any order of summation,
//! and the cases hold no negative non-integer, no file name of `write_map` and no `SLICE_WIDTH`.

use cloud_map_generator_core::export::*;
use cloud_map_generator_core::products::{encode_wind, WIND_STEP_MPS};
use cloud_map_generator_golden::{golden_root, load_case, run, Array, Case, Config, DType, Out};
use num_traits::Float;
use serde_json::{json, Value};

const EXPECTED_FAILURES: &[(&str, &str)] = &[];

/// The array's values in the precision of `F` (an f32 array widened to f64 and narrowed back is exact).
fn vals<F: Float>(a: &Array) -> Vec<F> {
    a.to_f64().into_iter().map(|v| F::from(v).expect("float conversion")).collect()
}

fn plane<F: Float>(a: &Array) -> (Vec<F>, usize, usize) {
    (vals(a), a.shape[0], a.shape[1])
}

fn as_plane<F>(p: &Option<(Vec<F>, usize, usize)>) -> Option<Plane<'_, F>> {
    p.as_ref().map(|(d, h, w)| Plane::new(d.as_slice(), *h, *w))
}

fn gray(out: Out, name: &str, img: &Image8) -> Out {
    assert_eq!(img.channels, 1);
    out.u8(name, &[img.height, img.width], &img.data)
}

fn rgb(out: Out, name: &str, img: &Image8) -> Out {
    assert_eq!(img.channels, 3);
    out.u8(name, &[img.height, img.width, 3], &img.data)
}

fn is_f32(a: &Array) -> bool {
    match a.dtype {
        DType::F32 => true,
        DType::F64 => false,
        other => panic!("a float array was expected, got {}", other.name()),
    }
}

fn keys(c: &Case) -> Vec<String> {
    c.param("keys")
        .as_array()
        .expect("params.keys is a list")
        .iter()
        .map(|k| k.as_str().expect("a key is a string").to_string())
        .collect()
}

fn check_scale(c: &Case, name: &str, ours: f64) {
    let theirs = c.param("scales")[name].as_f64().unwrap_or_else(|| panic!("case {}: no scale {name}", c.id));
    assert_eq!(theirs, ours, "case {}: the {name} scale", c.id);
}

fn write_map<F: Float + Send + Sync>(c: &Case) -> Out {
    let a = c.input("coverage");
    let (h, w) = (a.shape[0], a.shape[1]);
    let m = encode_map(&vals::<F>(a), h, w, c.p_str("stem"));
    check_scale(c, "master", MASTER_SCALE);
    check_scale(c, "preview", PREVIEW_SCALE);
    assert_eq!(c.p_str("master_mode"), "I;16"); // the master is a 16-bit grayscale image
    assert_eq!(c.p_str("preview_mode"), "L"); // the preview an 8-bit one
    Out::new()
        .put("master16", Array::from_u16(&[h, w], &m.master16))
        .u8("preview8", &[h, w], &m.preview8)
        .u8("r8_south_up", &[h, w], &m.r8_south_up)
}

fn write_products<F: Float + Send + Sync>(c: &Case) -> Out {
    let load = |name: &str| c.has_input(name).then(|| plane::<F>(c.input(name)));
    let (ty, hs, wnd) = (load("type"), load("high_share"), load("wind"));
    let extras = ProductExtras { type_: as_plane(&ty), high_share: as_plane(&hs), wind: as_plane(&wnd) };
    let e = encode_products(&extras, c.p_str("stem"));
    check_scale(c, "type", TYPE_SCALE);
    check_scale(c, "high_share", HIGH_SHARE_SCALE);

    // The sidecar text, key for key (the wind plane holds two components per texel).
    let files: serde_json::Map<String, Value> = e.sidecar.files.iter().map(|(k, f)| (k.to_string(), json!(f))).collect();
    let sidecar = json!({
        "type": e.sidecar.type_,
        "high_share": e.sidecar.high_share,
        "wind": e.sidecar.wind,
        "files": files,
    });
    assert_eq!(&sidecar, c.param("products"), "case {}: the sidecar's products field", c.id);

    let mut out = Out::new();
    if let Some(img) = &e.type_png {
        out = gray(out, "type_png", img);
    }
    if let Some(img) = &e.high_share_png {
        out = gray(out, "high_share_png", img);
    }
    if let Some(img) = &e.wind_png {
        out = rgb(out, "wind_png", img);
    }
    out
}

fn write_entry<F: Float + Send + Sync>(c: &Case) -> Out {
    let (cov, ch, cw) = plane::<F>(c.input("cov"));
    let (hs, hh, hw) = plane::<F>(c.input("extras.high_share"));
    let (ty, th, tw) = plane::<F>(c.input("extras.type"));
    let wnd_arr = c.input("extras.wind");
    let (wnd, wh, ww) = (vals::<F>(wnd_arr), wnd_arr.shape[0], wnd_arr.shape[1]);
    let extras = EntryExtras {
        high_share: Plane::new(&hs, hh, hw),
        type_: Plane::new(&ty, th, tw),
        wind: Plane::new(&wnd, wh, ww),
    };
    let e = encode_entry(
        Plane::new(&cov, ch, cw),
        &extras,
        c.p_str("stem"),
        c.p_str("prefix"),
        c.p_i64("seed") as u64,
        c.p_usize("slice_width"),
    );

    // The names, the encoding text and the manifest record, field for field.
    let files: serde_json::Map<String, Value> = e.files.pairs().iter().map(|(k, f)| (k.to_string(), json!(f))).collect();
    assert_eq!(&Value::Object(files), c.param("files"), "case {}: the files dict", c.id);
    let encoding: serde_json::Map<String, Value> = ENTRY_ENCODING.iter().map(|(k, t)| (k.to_string(), json!(t))).collect();
    assert_eq!(&Value::Object(encoding), c.param("encoding"), "case {}: the encoding dict", c.id);
    let r = &e.record;
    let record = json!({
        "seed": r.seed,
        "hires": {"url": r.hires.url, "width": r.hires.width, "height": r.hires.height},
        "slice": r.slice,
        "highShare": r.high_share,
        "type": r.type_,
        "wind": r.wind,
    });
    assert_eq!(&record, c.param("record"), "case {}: the manifest record", c.id);

    let out = Out::new().u8("hires_south_up", &[ch, cw], &e.hires_south_up);
    let out = gray(out, "slice", &e.slice);
    let out = gray(out, "high_share_png", &e.high_share);
    let out = gray(out, "type_png", &e.type_);
    rgb(out, "wind_png", &e.wind)
}

fn port(c: &Case) -> Option<Out> {
    Some(match c.function.as_str() {
        "bake.box_downsample" => {
            let a = c.input("a");
            let width = c.p_usize("width");
            if is_f32(a) {
                let (d, h, w) = plane::<f32>(a);
                let (o, oh, ow) = box_downsample(&d, h, w, width);
                Out::new().f32("out", &[oh, ow], &o)
            } else {
                let (d, h, w) = plane::<f64>(a);
                let (o, oh, ow) = box_downsample(&d, h, w, width);
                Out::new().f64("out", &[oh, ow], &o)
            }
        }
        "bake.key_seed" => {
            let ks = keys(c);
            let crc: Vec<u32> = ks.iter().map(|k| crc32(k.as_bytes())).collect();
            let seed: Vec<u32> = ks.iter().map(|k| key_seed(k)).collect();
            Out::new().u32("crc32", &[ks.len()], &crc).u32("seed", &[ks.len()], &seed)
        }
        "bake.slug" => {
            let ks = keys(c);
            let mut bytes: Vec<u8> = Vec::new();
            let mut offsets: Vec<i32> = vec![0];
            for k in &ks {
                bytes.extend_from_slice(slug(k).as_bytes());
                offsets.push(bytes.len() as i32);
            }
            Out::new().u8("slug.bytes", &[bytes.len()], &bytes).i32("slug.offsets", &[offsets.len()], &offsets)
        }
        "bake.write_entry" => {
            if is_f32(c.input("cov")) {
                write_entry::<f32>(c)
            } else {
                write_entry::<f64>(c)
            }
        }
        "cloudgen.export.write_map" => {
            if is_f32(c.input("coverage")) {
                write_map::<f32>(c)
            } else {
                write_map::<f64>(c)
            }
        }
        "cloudgen.export.write_products" => {
            // The products of a case share one dtype; take it from the first one present.
            let first = ["type", "high_share", "wind"].iter().find(|n| c.has_input(n)).expect("a product input");
            if is_f32(c.input(first)) {
                write_products::<f32>(c)
            } else {
                write_products::<f64>(c)
            }
        }
        "cloudgen.products.encode_wind" => {
            assert_eq!(c.p_f64("step_mps"), WIND_STEP_MPS, "case {}: the wind step", c.id);
            let wind = c.input("wind");
            let mut shape = wind.shape.clone();
            *shape.last_mut().unwrap() = 3;
            let rgb = if is_f32(wind) { encode_wind(&vals::<f32>(wind)) } else { encode_wind(&vals::<f64>(wind)) };
            Out::new().u8("rgb", &shape, &rgb)
        }
        _ => return None,
    })
}

#[test]
fn export_and_bake_helper_cases() {
    run(
        &Config {
            group: "pipeline",
            prefixes: &[
                "bake.box_downsample",
                "bake.key_seed",
                "bake.slug",
                "bake.write_entry",
                "cloudgen.export.",
                "cloudgen.products.encode_wind",
            ],
            expected_failures: EXPECTED_FAILURES,
        },
        port,
    )
    .assert_clean();
}

// ---------------------------------------------------------------------------------------------------------
// What the golden cases do not reach

/// `bake.SLICE_WIDTH` is a module constant: the frozen `bake.library_body` case records it in its params (the
/// case itself is another agent's, it needs `climate::BodyInputs`; only that parameter is read, as a check).
#[test]
fn slice_width_is_the_frozen_one() {
    let c = load_case(&golden_root().join("pipeline").join("bake.library_body"));
    assert_eq!(c.p_usize("slice_width"), SLICE_WIDTH);
    assert_eq!(SLICE_WIDTH, 2048);
}

/// `np.round` of a negative number that is not an integer (the cases only round values in `[0, 255]`, whose
/// fractional part is non-negative): half-to-even from both sides, the sign of a zero result kept.
#[test]
fn round_ties_even_rounds_negative_numbers_like_numpy() {
    for (x, want) in [(-3.7_f64, -4.0_f64), (-2.5, -2.0), (-1.5, -2.0), (-0.5, -0.0), (-0.2, -0.0), (-0.75, -1.0), (-6.25, -6.0)] {
        assert_eq!(round_ties_even(x).to_bits(), want.to_bits(), "f64 {x}");
        let y = x as f32;
        assert_eq!(round_ties_even(y).to_bits(), (want as f32).to_bits(), "f32 {y}");
    }
    assert_eq!(round_ties_even(3.5_f64), 4.0);
    assert_eq!(round_ties_even(2.5_f32), 2.0);
}

/// `write_map`'s file names: the width comes before the height in every image, the sidecar has no size. A map
/// that is not square tells the two apart (the cases only freeze the stem).
#[test]
fn write_map_names_put_the_width_before_the_height() {
    let f = map_files("moon", 4096, 2048);
    assert_eq!(f.master, "moon_4096x2048_16bit.png");
    assert_eq!(f.preview, "moon_4096x2048.png");
    assert_eq!(f.r8gz, "moon_4096x2048.r8.gz");
    assert_eq!(f.meta, "moon.json");
    let m = encode_map(&[0.0_f32; 6], 2, 3, "t"); // 2 rows of 3 texels
    assert_eq!(m.files, map_files("t", 3, 2));
    assert_eq!(m.files.preview, "t_3x2.png");
}

/// A xorshift32 stream: two draws make 53 random bits (exact in float64), mapped to `[-1, 2)` as `x * 3 - 1` in
/// float64 and rounded once to `F`. The same stream is generated in Python (numpy 2.5.4) to pin the bits below.
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

fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(0xCBF2_9CE4_8422_2325_u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3))
}

/// `bake.box_downsample` adds in numpy's order. `a.reshape(H, f, W, f).mean(axis=(1, 3))` sums each block ROW
/// first (a sequential sum under 8 values, eight accumulators from 8) and adds the row subtotals one by one
/// onto a running sum from +0.0: for `f = 2`, `(a00 + a01) + (a10 + a11)`, not `((a00 + a01) + a10) + a11`.
/// Dyadic data, which the cases use, cannot tell the two apart; these inputs can, and the bits are numpy's.
#[test]
fn box_downsample_adds_in_numpys_order() {
    // 1. Hand-made blocks (e = half an ulp of 1.0, so 1 + e is a tie that stays 1.0), two blocks side by side:
    //    f = 2 rows (1 + e) and (e + e) -> 1 + 2^-23; f = 4 rows 1, 2e, 2e, 2e; f = 8 one row of 8 accumulators.
    let e = 2.0_f32.powi(-24);
    let wide = |block: &[f32], f: usize| -> Vec<f32> {
        block.chunks_exact(f).flat_map(|row| row.iter().chain(row.iter()).copied().collect::<Vec<_>>()).collect()
    };
    let f2 = [1.0, e, e, e];
    let mut f4 = [0.0_f32; 16];
    f4[0] = 1.0;
    for r in 1..4 {
        f4[r * 4] = e;
        f4[r * 4 + 1] = e;
    }
    let mut f8 = [0.0_f32; 64];
    f8[0] = 1.0;
    f8[1..8].fill(e);
    for (block, f, want) in [(&f2[..], 2, 0x3E80_0001_u32), (&f4[..], 4, 0x3D80_0003), (&f8[..], 8, 0x3C80_0003)] {
        let (o, h, w) = box_downsample(&wide(block, f), f, 2 * f, 2);
        assert_eq!((h, w), (1, 2));
        assert_eq!(o.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), vec![want; 2], "f = {f}");
    }
    // one block wide, numpy merges the two reduced axes: a single run over the whole block (f = 2: 1.0)
    let (o, ..) = box_downsample(&f2, 2, 2, 1);
    assert_eq!(o[0].to_bits(), 0x3E80_0000);

    // 2. Random full-mantissa maps: (seed, height, width, slice width, FNV-1a of numpy's output bits), the data
    //    being `stream(seed)`: the f = 2 and f = 4 of the production maps, f = 8, and one block wide (f = 8).
    let pins32: [(u32, usize, usize, usize, u64); 4] = [
        (1000, 8, 12, 6, 0xB417_0891_9D44_AB1F),
        (1002, 16, 20, 5, 0x1EF0_285F_3590_08A6),
        (1003, 16, 24, 3, 0x09EA_D780_87C5_FE53),
        (1007, 16, 8, 1, 0x5114_CEE4_4710_79ED),
    ];
    let pins64: [(u32, usize, usize, usize, u64); 4] = [
        (1000, 8, 12, 6, 0xF7CD_B8F6_D262_4569),
        (1002, 16, 20, 5, 0xFB82_3A22_DB45_7019),
        (1003, 16, 24, 3, 0x20CB_9F08_B6FA_3B1D),
        (1007, 16, 8, 1, 0x6A6E_7CFD_5B0E_2904),
    ];
    for (seed, h, w, width, want) in pins32 {
        let (o, ..) = box_downsample(&stream::<f32>(seed, h * w), h, w, width);
        assert_eq!(fnv1a(o.iter().flat_map(|v| v.to_le_bytes())), want, "f32 {h} x {w}, slice width {width}");
    }
    for (seed, h, w, width, want) in pins64 {
        let (o, ..) = box_downsample(&stream::<f64>(seed, h * w), h, w, width);
        assert_eq!(fnv1a(o.iter().flat_map(|v| v.to_le_bytes())), want, "f64 {h} x {w}, slice width {width}");
    }
}
