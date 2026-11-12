# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "requests"]
# ///
"""Downloads real cloud-fraction maps for training the cloud refiner (NASA GIBS, MODIS Aqua and Terra, daily, no account needed).

    uv run training/fetch_gibs.py --count 200

Each map is the 10 x 5 cached WMTS tiles of level 3 of the 2 km matrix (5120 x 2560, equirectangular, north up, longitude -180..180),
rendered by GIBS with a colour palette. The palette is inverted with the colormap GIBS publishes (`MODIS_Cloud_Fraction.xml`), and the
map is averaged (valid pixels only) to 2048 x 1024, so what is saved is the cloud fraction itself:

    <out>/<layer>_<date>.png    2048 x 1024, 8-bit gray, 0..100 = cloud fraction in %, 255 = no data (the gaps between orbits)

Maps with more than --max-invalid of their pixels missing are dropped. NASA imagery is public domain (credit: NASA GSFC / EOSDIS GIBS).
"""
import argparse
import datetime as dt
import io
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from PIL import Image

TILES = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/{layer}/default/{day}/2km/3/{row}/{col}.png"
COLS, ROWS, TILE = 10, 5, 512
OUT_W, OUT_H = 2048, 1024
COLORMAP = "https://gibs.earthdata.nasa.gov/colormaps/v1.3/MODIS_Cloud_Fraction.xml"
LAYERS = {"aqua": "MODIS_Aqua_Cloud_Fraction_Day", "terra": "MODIS_Terra_Cloud_Fraction_Day"}
NODATA = 255


def palette(session):
    """The sorted RGB keys of the palette and the cloud fraction (0..100) each stands for."""
    root = ET.fromstring(session.get(COLORMAP, timeout=60).content)
    for cmap in root.iter("ColorMap"):
        if cmap.get("title") == "Cloud Fraction":
            keys, values = [], []
            for e in cmap.iter("ColorMapEntry"):
                r, g, b = (int(x) for x in e.get("rgb").split(","))
                keys.append((r << 16) | (g << 8) | b)
                values.append(int(float(e.get("value"))))
            order = np.argsort(keys, kind="stable")
            return np.array(keys, dtype=np.uint32)[order], np.array(values, dtype=np.uint8)[order]
    sys.exit("the colormap has no 'Cloud Fraction' entries")


def to_fraction(rgba, keys, values):
    """RGBA palette image -> uint8 cloud fraction in %, NODATA where the pixel is transparent or not a palette colour."""
    a = np.asarray(rgba)
    code = (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2].astype(np.uint32)
    at = np.clip(np.searchsorted(keys, code), 0, len(keys) - 1)
    known = (keys[at] == code) & (a[..., 3] > 0)
    return np.where(known, values[at], NODATA).astype(np.uint8)


def get_tile(session, layer, day, row, col):
    url = TILES.format(layer=layer, day=day, row=row, col=col)
    for attempt in range(4):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert("RGBA")
            if r.status_code == 404:
                return None  # no tile there (nothing observed): no data
        except requests.RequestException:
            pass
        time.sleep(1 + attempt)
    return None


def fetch(session, layer, day, keys, values):
    """The day's map at 2048 x 1024 (uint8, NODATA where nothing was observed), or None when the service did not answer at all."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(r, c) for r in range(ROWS) for c in range(COLS)]
    with ThreadPoolExecutor(8) as pool:
        tiles = list(pool.map(lambda rc: get_tile(session, layer, day, *rc), jobs))
    if all(t is None for t in tiles):
        return None
    full = np.full((ROWS * TILE, COLS * TILE), NODATA, dtype=np.uint8)
    for (r, c), t in zip(jobs, tiles):
        if t is not None:
            full[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = to_fraction(t, keys, values)
    # Average of the valid pixels into the output grid (BOX filter on the values and on the validity).
    valid = (full != NODATA).astype(np.float32)
    summed = Image.fromarray(np.where(full != NODATA, full, 0).astype(np.float32), "F").resize((OUT_W, OUT_H), Image.BOX)
    count = Image.fromarray(valid, "F").resize((OUT_W, OUT_H), Image.BOX)
    s, n = np.asarray(summed), np.asarray(count)
    out = np.where(n >= 0.5, np.rint(s / np.maximum(n, 1e-6)), NODATA)
    return np.clip(out, 0, NODATA).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="training/data/gibs")
    p.add_argument("--count", type=int, default=200, help="how many (layer, day) maps to keep")
    p.add_argument("--from", dest="start", default="2019-01-01")
    p.add_argument("--to", dest="end", default="2024-12-31")
    p.add_argument("--max-invalid", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "cloud-map-generator-training"
    keys, values = palette(session)

    t0, t1 = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
    days = [t0 + dt.timedelta(days=i) for i in range((t1 - t0).days + 1)]
    rng = random.Random(a.seed)
    rng.shuffle(days)
    have = len(list(out.glob("*.png")))
    for day in days:
        for name, layer in LAYERS.items():
            if have >= a.count:
                print(f"{have} maps in {out}")
                return
            target = out / f"{name}_{day.isoformat()}.png"
            if target.exists():
                continue
            frac = fetch(session, layer, day.isoformat(), keys, values)
            if frac is None:
                print(f"{name} {day}: no answer")
                continue
            invalid = float((frac == NODATA).mean())
            if invalid > a.max_invalid:
                print(f"{name} {day}: {invalid:.0%} missing, dropped")
                continue
            Image.fromarray(frac, "L").save(target, optimize=True)
            have += 1
            print(f"{name} {day}: kept ({invalid:.1%} missing, mean {float(frac[frac != NODATA].mean()):.0f} %) [{have}/{a.count}]")
    print(f"{have} maps in {out}")


if __name__ == "__main__":
    main()
