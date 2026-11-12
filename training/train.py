# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Trains the cloud refiner (version 2): a small U-Net that gives a generated cloud map the look of a real one.

    uv run training/fetch_truecolor.py --count 450                                      # real maps
    bash training/make_generated.sh rust/target/debug/cloud-map-generator.exe 100 200    # generated maps, matched to Earth
    bash training/make_generated.sh rust/target/debug/cloud-map-generator.exe 1000 150 raw   # generated maps, as the generator makes them
    uv run --index https://download.pytorch.org/whl/cu128 --with torch training/train.py --steps 8000

(`--index` gives a CUDA build of torch on Windows; an RTX 50 series card needs cu128 or newer.)

Two terms per step:
  A. supervised, on REAL maps: a crop is degraded (blur, correlated noise, both) and the network must give the original back (L1). This is what
     teaches it what real cloud structure looks like.
  B. on GENERATED maps, judged by what the tool measures against Earth: the output's radial spectrum slope over 40 to 400 km, its structure
     coherence and its histogram are pulled to those of Earth's reference map (`data/earth_cloud_fraction_bluemarble_2048.png`, measured with the same
     differentiable proxies), while the low-pass of the output stays the low-pass of the input (the layout) and the output stays close to the input.

The network takes 2 channels: the cloud fraction and sin(latitude) (the detail of a front and that of a subtropical cell are not the same).
It is a residual U-Net of 9 convolutions at 3 scales (about 106 thousand weights); `rust/cloud-map-generator-core/src/refiner.rs` runs the same
architecture, so the weights leave as a flat binary (`cloud-refiner.bin`). Change `ARCH` here and there together, and bump `VERSION`.

Outputs, in --out (rust/cloud-map-generator-core/models/ by default, where the crate embeds them): cloud-refiner.bin, cloud-refiner.json (the card).
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = 2
# (in, out, kernel, stride) of the convolutions, in the order the weights are written; see refiner.rs.
C1, C2, C3 = 16, 32, 64
ARCH = [
    (2, C1, 3, 1), (C1, C1, 3, 1),          # scale 0 -> skip 0
    (C1, C2, 3, 2), (C2, C2, 3, 1),         # scale 1 -> skip 1
    (C2, C3, 3, 2), (C3, C3, 3, 1),         # scale 2
    (C3 + C2, C2, 3, 1),                    # up to scale 1, with skip 1
    (C2 + C1, C1, 3, 1),                    # up to scale 0, with skip 0
    (C1, 1, 3, 1),                          # residual out (no activation)
]
CROP = 256
NODATA = 255
ROWS = (114, 910)  # rows of a 2048 x 1024 map between +-70 degrees


# ---------------------------------------------------------------------------------------------------------------- data

def load_real(root: Path, sets):
    maps = []
    for path in sorted(p for s in sets for p in root.glob(f"{s}/*.png")):
        a = np.asarray(Image.open(path)).astype(np.float32)
        if a.shape == (1024, 2048):
            maps.append(a)
    return maps


def load_gray(folder: Path):
    return [np.asarray(Image.open(p).convert("L")).astype(np.float32) / 255.0 for p in sorted(folder.glob("*.png"))]


def lat_plane(rows0, torch, device):
    """sin(latitude) of the rows r .. r + CROP of a 1024-row map, for each r of rows0: (n,1,CROP,CROP)."""
    out = np.stack([np.sin(np.radians(90 - (np.arange(r, r + CROP) + 0.5) * 180 / 1024)) for r in rows0]).astype(np.float32)
    return torch.from_numpy(out)[:, None, :, None].expand(len(out), 1, CROP, CROP).contiguous().to(device)


def crops(maps, n, device, torch, real):
    """n random crops that wrap in longitude: (cloud fraction (n,1,H,W), sin lat (n,1,H,W), valid mask or None)."""
    cs, rs, ms = [], [], []
    while len(cs) < n:
        m = maps[np.random.randint(len(maps))]
        r = np.random.randint(ROWS[0], ROWS[1] - CROP + 1)
        cols = (np.arange(CROP) + np.random.randint(2048)) % 2048
        c = m[r:r + CROP][:, cols]
        if real:
            bad = c == NODATA
            if bad.mean() > 0.03:
                continue
            c = np.where(bad, c[~bad].mean(), c) / 100.0
            ms.append((~bad).astype(np.float32))
        cs.append(c)
        rs.append(r)
    x = torch.from_numpy(np.stack(cs)[:, None].astype(np.float32)).to(device)
    m = torch.from_numpy(np.stack(ms)[:, None]).to(device) if real else None
    return x, lat_plane(rs, torch, device), m


# ---------------------------------------------------------------------------------------------------------------- model

def build(torch):
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList([nn.Conv2d(i, o, k, s, k // 2) for (i, o, k, s) in ARCH])

        def forward(self, x, lat):
            c = self.convs
            z = torch.cat([x, lat], 1)
            a = F.relu(c[1](F.relu(c[0](z))))
            b = F.relu(c[3](F.relu(c[2](a))))
            d = F.relu(c[5](F.relu(c[4](b))))
            u = F.interpolate(d, scale_factor=2, mode="nearest")
            u = F.relu(c[6](torch.cat([u, b], 1)))
            u = F.interpolate(u, scale_factor=2, mode="nearest")
            u = F.relu(c[7](torch.cat([u, a], 1)))
            return (x + c[8](u)).clamp(0, 1)

    return Net()


def blur(x, sigma):
    import torch
    import torch.nn.functional as F
    r = max(1, int(3 * sigma))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.pad(x, (r, r, 0, 0), mode="circular")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    return F.conv2d(x, k.view(1, 1, -1, 1))


def degrade(y):
    """Originals (B,1,H,W in 0..1) -> degraded copies: blur, correlated noise, or both."""
    import torch
    x = y.clone()
    for i in range(y.shape[0]):
        xi = y[i:i + 1]
        mode = np.random.randint(3)
        if mode in (0, 2):
            xi = blur(xi, float(np.random.uniform(0.8, 3.5)))
        if mode in (1, 2):
            n = blur(torch.randn_like(xi), float(np.random.uniform(0.4, 2.0)))
            n = n / (n.std() + 1e-6)
            xi = xi + float(np.random.uniform(0.04, 0.25)) * n
        x[i] = xi.clamp(0, 1)[0]
    return x


# --------------------------------------------------------------------------------------------- what the tool measures

class Proxies:
    """Differentiable stand-ins for three of the tool's metrics on a batch of 256 x 256 crops (2048 wide: about 20 km a texel)."""

    def __init__(self, torch, device):
        self.torch = torch
        n = CROP
        fy = torch.fft.fftfreq(n, device=device) * n
        fx = torch.fft.rfftfreq(n, device=device) * n
        k = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        # 40 to 400 km at 20 km a texel is wavelengths of 2 to 20 texels: 12.8 to 128 cycles per crop; 12 log-spaced bins.
        edges = torch.logspace(math.log10(13.0), math.log10(120.0), 13, device=device)
        self.bin = torch.bucketize(k, edges) - 1
        self.valid = (self.bin >= 0) & (self.bin < 12)
        self.centers = torch.log10(torch.sqrt(edges[:-1] * edges[1:]))
        self.hann = (torch.hann_window(n, device=device)[:, None] * torch.hann_window(n, device=device)[None, :])[None, None]

    def slope(self, x):
        """Mean over the batch of the log-log slope of the radially averaged power spectrum."""
        torch = self.torch
        x = x - x.mean((2, 3), keepdim=True)
        p = (torch.fft.rfft2(x * self.hann).abs() ** 2)[:, 0]
        idx = self.bin[self.valid]
        pv = p[:, self.valid]
        sums = torch.zeros(p.shape[0], 12, device=p.device).index_add_(1, idx, pv)
        cnt = torch.bincount(idx, minlength=12).clamp(min=1).float()
        logp = torch.log10(sums / cnt + 1e-12)
        c = self.centers - self.centers.mean()
        return ((logp - logp.mean(1, keepdim=True)) * c).sum(1).div((c ** 2).sum()).mean()

    def coherence(self, x):
        """Mean orientation coherence of the structure tensor, smoothed over sigma = 4 texels."""
        import torch.nn.functional as F
        torch = self.torch
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3) / 8
        gx, gy = F.conv2d(x, kx), F.conv2d(x, kx.transpose(2, 3))
        jxx, jyy, jxy = blur(gx * gx, 4.0), blur(gy * gy, 4.0), blur(gx * gy, 4.0)
        coh = torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2 + 1e-12) / (jxx + jyy + 1e-6)
        return coh.mean()

    def quantiles(self, x, q):
        return self.torch.quantile(x.flatten(), q)


# ---------------------------------------------------------------------------------------------------------------- export

def write_bin(net, path):
    """CMGR, version, convolution count, then per convolution: in, out, kernel, stride (u32) and weights [out][in][k][k] then bias (f32)."""
    import struct
    out = bytearray(b"CMGR")
    out += struct.pack("<II", VERSION, len(ARCH))
    for conv, (i, o, k, s) in zip(net.convs, ARCH):
        out += struct.pack("<IIII", i, o, k, s)
        out += conv.weight.detach().cpu().numpy().astype("<f4").tobytes()
        out += conv.bias.detach().cpu().numpy().astype("<f4").tobytes()
    path.write_bytes(bytes(out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="training/data")
    p.add_argument("--reference", default="data/earth_cloud_fraction_bluemarble_2048.png")
    p.add_argument("--out", default="rust/cloud-map-generator-core/models")
    p.add_argument("--sets", default="gibs_tc", help="real maps: folders of training/data/, comma separated")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w-real", type=float, default=1.0, help="weight of the supervised term on real maps (A)")
    p.add_argument("--w-slope", type=float, default=4.0)
    p.add_argument("--w-coh", type=float, default=10.0)
    p.add_argument("--w-hist", type=float, default=5.0)
    p.add_argument("--w-content", type=float, default=10.0)
    p.add_argument("--w-identity", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    import torch

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(a.data)
    real = load_real(root, a.sets.split(","))
    gen = load_gray(root / "generated") + load_gray(root / "generated_raw")
    if len(real) < 8 or len(gen) < 8:
        raise SystemExit(f"{len(real)} real and {len(gen)} generated maps: run fetch_truecolor.py and make_generated.sh first")
    rng = np.random.RandomState(a.seed)
    order = rng.permutation(len(real))
    n_val = max(2, len(real) // 10)
    val, train = [real[i] for i in order[:n_val]], [real[i] for i in order[n_val:]]
    print(f"{len(train)} real maps (+{len(val)} held out), {len(gen)} generated maps, device {device}")

    # Earth's reference measured with the same proxies: the targets.
    ref = np.asarray(Image.open(a.reference).convert("L")).astype(np.float32) / 255.0
    px = Proxies(torch, device)
    ref_crops = [ref[r:r + CROP][:, (np.arange(CROP) + c) % 2048] for r in range(ROWS[0], ROWS[1] - CROP + 1, 40) for c in range(0, 2048, 256)]
    rc = torch.from_numpy(np.stack(ref_crops)[:, None]).to(device)
    q = torch.linspace(0.02, 0.98, 49, device=device)
    t_slope, t_coh, t_q = px.slope(rc).item(), px.coherence(rc).item(), px.quantiles(rc, q)
    xs, _, _ = crops(gen, 64, device, torch, False)
    print(f"proxies  Earth: slope {t_slope:.2f} coherence {t_coh:.3f}   generated input: slope {px.slope(xs).item():.2f} coherence {px.coherence(xs).item():.3f}")

    net = build(torch).to(device)
    with torch.no_grad():  # start as the identity
        net.convs[-1].weight.zero_()
        net.convs[-1].bias.zero_()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        # A: real maps, degraded -> original.
        y, lat, m = crops(train, a.batch, device, torch, True)
        la = ((net(degrade(y), lat) - y).abs() * m).mean()
        # B: generated maps judged by the proxies.
        x, lat, _ = crops(gen, a.batch, device, torch, False)
        out = net(x, lat)
        s_out, c_out = px.slope(out), px.coherence(out)
        l_hist = (px.quantiles(out, q) - t_q).abs().mean()
        l_content = (blur(out, 3.0) - blur(x, 3.0)).abs().mean()
        l_id = (out - x).abs().mean()
        loss = a.w_real * la + a.w_slope * (s_out - t_slope) ** 2 + a.w_coh * (c_out - t_coh) ** 2 + a.w_hist * l_hist + a.w_content * l_content + a.w_identity * l_id
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == a.steps:
            print(f"step {step}/{a.steps} A {la.item():.4f} slope {s_out.item():.2f} coh {c_out.item():.3f} hist {l_hist.item():.4f} "
                  f"content {l_content.item():.4f} id {l_id.item():.4f} ({time.time() - t0:.0f} s)")

    net.eval()
    with torch.no_grad():
        base, got, s_in, s_out, c_in, c_out = [], [], [], [], [], []
        for _ in range(30):
            y, lat, m = crops(val, 8, device, torch, True)
            xd = degrade(y)
            base.append(((xd - y).abs() * m).mean().item())
            got.append(((net(xd, lat) - y).abs() * m).mean().item())
            x, lat, _ = crops(gen, 8, device, torch, False)
            o = net(x, lat)
            s_in.append(px.slope(x).item()); s_out.append(px.slope(o).item())
            c_in.append(px.coherence(x).item()); c_out.append(px.coherence(o).item())
        # The window the parity test of refiner.rs reads: 2 channels, 128 x 128.
        x, lat, _ = crops(gen, 1, device, torch, False)
        x, lat = x[:, :, :128, :128].contiguous(), lat[:, :, :128, :128].contiguous()
        win = net(x, lat)
    metrics = {"val_l1_degraded": float(np.mean(base)), "val_l1_refined": float(np.mean(got)),
               "slope_target": t_slope, "slope_generated_in": float(np.mean(s_in)), "slope_generated_out": float(np.mean(s_out)),
               "coherence_target": t_coh, "coherence_generated_in": float(np.mean(c_in)), "coherence_generated_out": float(np.mean(c_out))}
    print(metrics)
    (root / "parity_input.f32").write_bytes(torch.cat([x, lat], 1).cpu().numpy().astype("<f4").tobytes())
    (root / "parity_output.f32").write_bytes(win.float().cpu().numpy().astype("<f4").tobytes())

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_bin(net, out_dir / "cloud-refiner.bin")
    sha = hashlib.sha256((out_dir / "cloud-refiner.bin").read_bytes()).hexdigest()
    card = {
        "name": "cloud-refiner", "version": VERSION, "sha256": sha, "parameters": sum(p.numel() for p in net.parameters()), "steps": a.steps, "batch": a.batch,
        "inputs": "cloud fraction and sin(latitude)",
        "data": {"real": f"NASA GIBS {a.sets}: MODIS Terra true colour inverted against the cloud-free Blue Marble, as the reference map was made",
                 "real_maps": len(real), "held_out": len(val), "generated_maps": len(gen), "license": "NASA imagery is public domain (credit: NASA GSFC / EOSDIS GIBS)"},
        "loss": {"real": a.w_real, "slope": a.w_slope, "coherence": a.w_coh, "histogram": a.w_hist, "content": a.w_content, "identity": a.w_identity},
        "metrics": metrics,
    }
    (out_dir / "cloud-refiner.json").write_text(json.dumps(card, indent=2) + "\n")
    print(f"wrote {out_dir / 'cloud-refiner.bin'} ({card['parameters']} weights, sha256 {sha[:12]})")


if __name__ == "__main__":
    main()
