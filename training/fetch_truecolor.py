# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "requests"]
# ///
"""Real cloud-fraction maps made the way the reference map (`data/earth_cloud_fraction_bluemarble_2048.png`) was made.

    uv run training/fetch_truecolor.py --count 300

The reference is a Blue Marble composite of clouded imagery inverted against the cloud-free image (`data/README.md`). GIBS publishes both:
`MODIS_Terra_CorrectedReflectance_TrueColor` (a day of MODIS, with clouds) and `BlueMarble_NextGeneration` (the same world without clouds).
For each day, per pixel and per colour channel, with C the day and B the cloud-free image (0..1) and a cloud of albedo 1:

    C = B (1 - c) + c      ->      c = (C - B) / max(1 - B, 0.4)      (the divisor is floored: a bright surface has no room for a cloud)

the channels are averaged, values under 0.03 are zero (JPEG noise), and the result is averaged (valid pixels only) to 2048 x 1024. The sets
`fetch_gibs.py` makes (MODIS cloud mask, nearly binary) and this one (graded, like the reference) are both read by the trainers.

    <out>/tc_<date>.png    2048 x 1024, 8-bit gray, 0..100 = cloud fraction in %, 255 = no data (the gaps between orbits, the polar night)

Snow, ice and deserts invert as cloud, as in the reference: the trainers keep to +-70 degrees. NASA imagery is public domain
(credit: NASA GSFC / EOSDIS GIBS).
"""
import argparse
import datetime as dt
import io
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best"
CLEAR = BASE + "/BlueMarble_NextGeneration/default/500m/3/{row}/{col}.jpeg"
DAY = BASE + "/MODIS_Terra_CorrectedReflectance_TrueColor/default/{day}/250m/3/{row}/{col}.jpeg"
COLS, ROWS, TILE = 10, 5, 512
OUT_W, OUT_H = 2048, 1024
NODATA = 255
FLOOR, DEADBAND = 0.4, 0.03


def get_tile(session, url):
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    return None


def mosaic(session, template, **kw):
    jobs = [(r, c) for r in range(ROWS) for c in range(COLS)]
    with ThreadPoolExecutor(8) as pool:
        tiles = list(pool.map(lambda rc: get_tile(session, template.format(row=rc[0], col=rc[1], **kw)), jobs))
    if all(t is None for t in tiles):
        return None
    full = np.zeros((ROWS * TILE, COLS * TILE, 3), dtype=np.uint8)
    for (r, c), t in zip(jobs, tiles):
        if t is not None:
            full[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = t
    return full


def box_to_output(values, valid):
    """Valid-weighted average of a 5120 x 2560 array into 2048 x 1024; NODATA where less than half of a texel was observed."""
    s = np.asarray(Image.fromarray(np.where(valid, values, 0).astype(np.float32), "F").resize((OUT_W, OUT_H), Image.BOX))
    n = np.asarray(Image.fromarray(valid.astype(np.float32), "F").resize((OUT_W, OUT_H), Image.BOX))
    out = np.where(n >= 0.5, np.rint(100 * s / np.maximum(n, 1e-6)), NODATA)
    return np.clip(out, 0, NODATA).astype(np.uint8)


def invert(day_rgb, clear_rgb):
    c, b = day_rgb.astype(np.float32) / 255.0, clear_rgb.astype(np.float32) / 255.0
    cloud = ((c - b) / np.maximum(1.0 - b, FLOOR)).mean(axis=2)
    cloud = np.where(cloud < DEADBAND, 0.0, cloud)
    valid = day_rgb.astype(np.int32).sum(axis=2) > 12  # black: no observation
    return np.clip(cloud, 0, 1), valid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="training/data/gibs_tc")
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--from", dest="start", default="2012-01-01")
    p.add_argument("--to", dest="end", default="2024-12-31")
    p.add_argument("--max-invalid", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "cloud-map-generator-training"

    cache = Path(a.out).parent / "bluemarble_clear_5120.npy"
    if cache.exists():
        clear = np.load(cache)
    else:
        clear = mosaic(session, CLEAR)
        if clear is None:
            raise SystemExit("could not download the cloud-free Blue Marble tiles")
        np.save(cache, clear)

    t0, t1 = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    days = [t0 + dt.timedelta(days=i) for i in range((t1 - t0).days + 1)]
    random.Random(a.seed).shuffle(days)
    have = len(list(out.glob("tc_*.png")))
    for day in days:
        if have >= a.count:
            break
        target = out / f"tc_{day.isoformat()}.png"
        if target.exists():
            continue
        img = mosaic(session, DAY, day=day.isoformat())
        if img is None:
            print(f"{day}: no answer")
            continue
        cloud, valid = invert(img, clear)
        invalid = 1.0 - float(valid.mean())
        if invalid > a.max_invalid:
            print(f"{day}: {invalid:.0%} missing, dropped")
            continue
        frac = box_to_output(cloud, valid)
        Image.fromarray(frac, "L").save(target, optimize=True)
        have += 1
        ok = frac != NODATA
        print(f"{day}: kept ({invalid:.1%} missing, mean {float(frac[ok].mean()):.0f} %, share of 0-100 values {float(((frac[ok] > 5) & (frac[ok] < 95)).mean()):.0%}) [{have}/{a.count}]")
    print(f"{have} maps in {out}")


if __name__ == "__main__":
    main()
