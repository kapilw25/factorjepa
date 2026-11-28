"""Figure 3: re-render the 13 value labels and 6 x-tick labels at 8.7 pt (9.0 pt as printed, x1.036), crop the
blank band above the title, fill the 2 px notch in the top frame. Owner pick Fri Nov 27 2026 ~9:40 PM PST.

Same method as the row-label swap: the figure is a pdftoppm 600-dpi raster of a matplotlib PDF (serif bold;
values 7.5 pt, ticks 7.5 pt). For each text item the OLD string is rendered through that pipeline at 10x10
sub-pixel phases and located by template matching; the item is changed only if the old render reproduces the
figure's pixels (max abs diff <= TOL). The NEW string at 8.7 pt is then rendered at the same anchor and phase and
composited over the exact background (solid bar colour, or white plus the flat grey gridlines).
usage: python -I numbers9pt.py <in.png> <out.png> <workdir>
"""
import os, subprocess, sys
import numpy as np
os.environ["MPLCONFIGDIR"] = os.environ["TMPDIR"]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from PIL import Image

src, out, wd = sys.argv[1], sys.argv[2], sys.argv[3]
OLD_PT, NEW_PT, TOL, TOP_CROP = 7.5, 8.7, 3, 45
img = np.asarray(Image.open(src).convert("RGB")).astype(np.float64)
H, W, _ = img.shape
res = img.copy()
plt.rcParams.update({"font.family": "serif", "font.weight": "bold"})
S = 600 / 72.0

def cov_render(text, pt, ha, va, dx, dy):
    """Coverage (0..1) of `text` rendered black on white, anchor at canvas px (1200+dx, 300+dy) from top-left."""
    fig = plt.figure(figsize=(4.0, 1.0))
    fig.text((2.0 * 72 + dx / S) / (4.0 * 72), 1 - (0.5 * 72 + dy / S) / (1.0 * 72), text, fontsize=pt, ha=ha, va=va)
    pdf = os.path.join(wd, "n.pdf"); fig.savefig(pdf); plt.close(fig)
    subprocess.run(["pdftoppm", "-r", "600", "-gray", "-singlefile", "-png", pdf, os.path.join(wd, "n")], check=True)
    a = np.asarray(Image.open(os.path.join(wd, "n.png")).convert("L")).astype(np.float64)
    return (255.0 - a) / 255.0

def ink(c):
    ys, xs = np.where(c > 0)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

# ---- background models -------------------------------------------------------------------------------------------
VCOLS = [(799, 805), (971, 976), (1143, 1148), (1697, 1702)]      # flat grey-245 vertical gridlines (x, inclusive)
HROWS = {996: (993, 999), 1145: (1142, 1148), 1294: (1291, 1297)} # flat grey-242 horizontal gridlines of the outside-label rows

def bg_outside(y0, y1, x0, x1, hrow):
    b = np.full((y1 - y0, x1 - x0, 3), 255.0)
    for a, c in VCOLS:
        lo, hi = max(a, x0), min(c + 1, x1)
        if lo < hi: b[:, lo - x0:hi - x0] = 245.0
    r0, r1 = hrow
    lo, hi = max(r0, y0), min(r1 + 1, y1)
    if lo < hi:
        b[lo - y0:hi - y0] = 242.0
        for a, c in VCOLS:                                         # crossings: value read from the figure itself
            xl, xh = max(a, x0), min(c + 1, x1)
            if xl < xh: b[lo - y0:hi - y0, xl - x0:xh - x0] = XVAL
    return b

XVAL = float(img[996, 973, 0])                                     # gridline crossing (row 993-999, column 971-976)
assert img[996, 973].std() == 0 and img[401, 1000, 0] == 242 and img[480, 973, 0] == 245, "gridline model off"

# ---- items -------------------------------------------------------------------------------------------------------
CENTRES = [401 + round(k * 148.8) for k in range(13)]
GREEN, AMBER, RED = (46, 125, 50), (249, 168, 37), (176, 44, 44)
VALUES = [("43.3×", GREEN), ("33.2×", GREEN), ("20.0×", GREEN), ("13.9×", GREEN), ("1.0×", AMBER), ("-0.1×", RED),
          ("-0.6×", RED), ("-8.3×", RED), ("-9.9×", RED), ("-12.1×", RED), ("-14.5×", RED), ("-18.1×", RED), ("-146.5×", RED)]
items = []
for (txt, bar), cy in zip(VALUES, CENTRES):
    v = float(txt.rstrip("×"))
    inside = abs(v) >= 2.0
    ha = ("left" if v < 0 else "right") if inside else ("left" if v >= 0 else "right")
    items.append(dict(text=txt, ha=ha, va="center", win=(cy - 70, cy + 70, 772, 1822),
                      kind="in" if inside else "out", bar=np.array(bar, float), color=255.0 if inside else 26.0, hrow=HROWS.get(cy)))
for t, x in (("-100", 802), ("-10", 973), ("-1", 1145), ("0", 1336), ("1", 1528), ("10", 1699)):
    items.append(dict(text=t, ha="center", va="top", win=(2298, H, max(0, x - 200), min(W, x + 200)), kind="tick",
                      color=0.0, bar=None, hrow=None))

def observed_cov(it, y0, y1, x0, x1):
    p = img[y0:y1, x0:x1]
    if it["kind"] == "in":
        ch = int(np.argmax(np.abs(255.0 - it["bar"])))
        return (p[..., ch] - it["bar"][ch]) / (255.0 - it["bar"][ch]), None
    b = bg_outside(y0, y1, x0, x1, it["hrow"]) if it["kind"] == "out" else np.full(p.shape, 255.0)
    return (b[..., 0] - p[..., 0]) / (b[..., 0] - it["color"]), b

report = []
for it in items:
    y0, y1, x0, x1 = it["win"]
    obs, _ = observed_cov(it, y0, y1, x0, x1)
    obs = np.clip(obs, 0, 1).astype(np.float32)
    best = None
    for dx in np.arange(0, 1.0, 0.1):
        for dy in np.arange(0, 1.0, 0.1):
            c = cov_render(it["text"], OLD_PT, it["ha"], it["va"], dx, dy)
            a0, a1, b0, b1 = ink(c)
            tpl = c[a0:a1, b0:b1].astype(np.float32)
            r = cv2.matchTemplate(obs, tpl, cv2.TM_SQDIFF)
            mn, _, loc, _ = cv2.minMaxLoc(r)
            if best is None or mn < best[0]:
                best = (mn, dx, dy, (y0 + loc[1] - a0, x0 + loc[0] - b0))   # translation canvas -> figure
    mn, dx, dy, (ty, tx) = best
    old = cov_render(it["text"], OLD_PT, it["ha"], it["va"], dx, dy)
    new = cov_render(it["text"], NEW_PT, it["ha"], it["va"], dx, dy)
    a0, a1, b0, b1 = ink(old); n0, n1, m0, m1 = ink(new)
    Y0, Y1, X0, X1 = min(a0, n0) + ty - 2, max(a1, n1) + ty + 2, min(b0, m0) + tx - 2, max(b1, m1) + tx + 2
    assert 0 <= Y0 and Y1 <= H and 0 <= X0 and X1 <= W, f"{it['text']}: new text leaves the image"
    if it["kind"] == "in":
        bg = np.broadcast_to(it["bar"], (Y1 - Y0, X1 - X0, 3)).copy()
    elif it["kind"] == "out":
        bg = bg_outside(Y0, Y1, X0, X1, it["hrow"])
    else:
        bg = np.full((Y1 - Y0, X1 - X0, 3), 255.0)
    oc, nc = old[Y0 - ty:Y1 - ty, X0 - tx:X1 - tx, None], new[Y0 - ty:Y1 - ty, X0 - tx:X1 - tx, None]
    recon = oc * it["color"] + (1 - oc) * bg
    err = np.abs(np.rint(recon) - img[Y0:Y1, X0:X1]).max()
    if err > TOL:
        sys.exit(f"FATAL: old '{it['text']}' does not reproduce the figure (max diff {err:.0f} > {TOL}); nothing written")
    if it["kind"] == "in":                                          # the new label must stay inside its bar
        rows_bar = (np.abs(img[Y0:Y1, X0:X1] - it["bar"]).sum(-1) < 6) | (oc[..., 0] > 0)
        assert rows_bar[nc[..., 0] > 0].all(), f"{it['text']}: 8.7 pt label would leave its bar"
    res[Y0:Y1, X0:X1] = nc * it["color"] + (1 - nc) * bg
    report.append(f"{it['text']:>8s} {it['kind']:4s} phase ({dx:.1f},{dy:.1f}) old-reproduction max diff {err:.0f}; "
                  f"box y {Y0}-{Y1} x {X0}-{X1}")

# ---- frame notch, top crop -----------------------------------------------------------------------------------------
notch = [x for x in range(760, 1823) if img[291, x, 0] == 255 and img[292, x, 0] == 255]
ref = img[291:293, 990].copy()
assert (ref == 0).all(), "reference column 990 rows 291-292 are not frame ink"
res[291:293, notch[0]:notch[-1] + 1] = ref[:, None, :]
report.append(f"notch rows 291-292 filled at x {notch[0]}-{notch[-1]} ({len(notch)} columns, contiguous: {notch[-1] - notch[0] + 1 == len(notch)})")
assert (res[:TOP_CROP] == 255).all(), "top crop would cut ink"
final = res[TOP_CROP:]
Image.fromarray(np.rint(final).astype(np.uint8)).save(out, dpi=(600, 600), optimize=True)
print("\n".join(report))
print(f"out {final.shape[1]}x{final.shape[0]} (cropped {TOP_CROP} px of blank top)")
