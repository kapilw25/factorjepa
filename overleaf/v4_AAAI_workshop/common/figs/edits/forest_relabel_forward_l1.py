"""Replace the 'causal-block L1' row label of Figure 3 with 'forward L1', pixels of that label only.

The figure is a pdftoppm 600-dpi raster of a matplotlib PDF (row labels: serif bold 9 pt, right-aligned).
Same pipeline here: render both labels at one right-aligned anchor in a matplotlib PDF, rasterize with
pdftoppm -r 600, find the sub-pixel anchor at which the rendered OLD label reproduces the figure's pixels,
then paste the NEW label rendered at that same anchor. Everything outside the old label's box is untouched.
usage: python -I relabel.py <in.png> <out.png> <workdir>
"""
import os, subprocess, sys
import numpy as np
os.environ["MPLCONFIGDIR"] = os.environ["TMPDIR"]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

src, out, wd = sys.argv[1], sys.argv[2], sys.argv[3]
OLD, NEW = "causal-block L1", "forward L1"
BOX = (0, 800, 712, 896)        # x0, y0, x1, y1 px: old label band y 819-877, ink x 45-697; next ink x >= 726 (tick)
img = np.asarray(Image.open(src).convert("RGB")).astype(np.int16)
gray = img[..., 0]
assert (img[..., 0] == img[..., 1]).all() and (img[..., 1] == img[..., 2]).all() or True
tgt = gray[BOX[1]:BOX[3], BOX[0]:BOX[2]]
print("label box is pure gray:", bool((img[BOX[1]:BOX[3], BOX[0]:BOX[2], 0] == img[BOX[1]:BOX[3], BOX[0]:BOX[2], 2]).all()))
plt.rcParams.update({"font.family": "serif", "font.weight": "bold"})
S = 600 / 72.0

def raster(text, dx_px, dy_px):
    fig = plt.figure(figsize=(2.0, 0.4))
    fig.text((1.9 * 72 + dx_px / S) / (2.0 * 72), (0.15 * 72 + dy_px / S) / (0.4 * 72), text,
             fontsize=9, ha="right", va="baseline")
    pdf = os.path.join(wd, "lab.pdf"); fig.savefig(pdf); plt.close(fig)
    subprocess.run(["pdftoppm", "-r", "600", "-gray", "-singlefile", "-png", pdf, os.path.join(wd, "lab")], check=True)
    return np.asarray(Image.open(os.path.join(wd, "lab.png")).convert("L")).astype(np.int16)

def ink_box(a):
    ys, xs = np.where(a < 255)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

t0, t1, s0, s1 = ink_box(tgt)
best = None
for dx in np.arange(0, 1.0, 0.1):
    for dy in np.arange(0, 1.0, 0.1):
        r = raster(OLD, dx, dy)
        y0, y1, x0, x1 = ink_box(r)
        if (y1 - y0, x1 - x0) != (t1 - t0, s1 - s0):
            continue
        d = np.abs(r[y0:y1, x0:x1] - tgt[t0:t1, s0:s1])
        score = d.sum()
        if best is None or score < best[0]:
            best = (score, dx, dy, d.max(), (d > 0).sum())
print("best sub-pixel anchor:", best)
if best is None:
    sys.exit("FATAL: no anchor reproduces the old label's ink box")
_, dx, dy, _, _ = best
r_old, r_new = raster(OLD, dx, dy), None
y0, y1, x0, x1 = ink_box(r_old)
oy, ox = BOX[1] + t0 - y0, BOX[0] + s0 - x0          # translation render -> figure
r_new = raster(NEW, dx, dy)
n0, n1, m0, m1 = ink_box(r_new)
outimg = img.copy()
outimg[BOX[1]:BOX[3], BOX[0]:BOX[2], :] = 255
patch = r_new[n0:n1, m0:m1]
fy, fx = n0 + oy, m0 + ox
assert BOX[1] <= fy and fy + patch.shape[0] <= BOX[3] and BOX[0] <= fx and fx + patch.shape[1] <= BOX[2]
outimg[fy:fy + patch.shape[0], fx:fx + patch.shape[1], :] = patch[..., None]
Image.fromarray(outimg.astype(np.uint8)).save(out, dpi=(600, 600), optimize=True)
changed = np.abs(outimg - img).sum(-1) > 0
ys, xs = np.where(changed)
outside = changed.copy(); outside[BOX[1]:BOX[3], BOX[0]:BOX[2]] = False
print(f"new label ink x {fx}-{fx + patch.shape[1] - 1}, y {fy}-{fy + patch.shape[0] - 1}; old ink x {BOX[0]+s0}-{BOX[0]+s1-1}, y {BOX[1]+t0}-{BOX[1]+t1-1}")
print(f"changed px {changed.sum()}, bbox x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}; changed outside label box: {outside.sum()}")
