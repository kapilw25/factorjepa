"""Compose the DENSEWORLD teaser: a 10-thumbnail scene strip over the 3-panel
factor decomposition, as ONE 300-dpi PNG sized to \\textwidth (7.0 in).

The AAAI kit forbids minipage grouping and in-LaTeX cropping, so all
cropping and layout happen here. Scene labels are listed in the caption
(a 9 pt label does not fit under a 0.66 in thumbnail); factor labels sit
under their panels at 10 pt.

usage: python3 -c "$(cat make_composite.py)" <fig_dir> <out.png> <factor_frac>
  factor_frac: width fraction of the factor row (1.0 = full width, 0.72 = compact)
"""
import sys
from PIL import Image, ImageDraw, ImageFont

fig_dir, out_path, factor_frac = sys.argv[1], sys.argv[2], float(sys.argv[3])

DPI = 300
W = int(7.0 * DPI)                       # \textwidth = 504 pt = 7.0 in
GAP = int(3 / 72 * DPI)                  # 3 pt between thumbnails
ROW_GAP = int(5 / 72 * DPI)
LABEL_PT = 10
LABEL_H = int(13 / 72 * DPI)             # 10 pt type on 13 pt strip

SCENES = ["market", "residential", "commercial", "promenade", "transit",
          "highway", "heritage", "junction", "flyover", "beach"]
FACTORS = [("agents_scene.png", "(a) Agents"),
           ("layout_scene.png", "(b) Layout"),
           ("interaction_scene.png", "(c) Interactions")]


def load(name):
    for ext in (".png", ".jpg"):
        try:
            return Image.open(f"{fig_dir}/{name}{ext}").convert("RGB")
        except FileNotFoundError:
            continue
    raise FileNotFoundError(name)


def center_crop(im, aspect):
    w, h = im.size
    if w / h > aspect:            # too wide: trim width
        nw = int(h * aspect)
        x0 = (w - nw) // 2
        return im.crop((x0, 0, x0 + nw, h))
    nh = int(w / aspect)          # too tall: trim height
    y0 = (h - nh) // 2
    return im.crop((0, y0, w, y0 + nh))


font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",
                          int(LABEL_PT / 72 * DPI), index=1)   # index 1 = Bold

# Row 1: ten scene thumbnails, common 3:2 aspect (least destructive middle ground).
n = len(SCENES)
tw = (W - GAP * (n - 1)) // n
th = int(tw / 1.5)
row1 = Image.new("RGB", (W, th), "white")
x = 0
for s in SCENES:
    im = center_crop(load(s), 1.5).resize((tw, th), Image.LANCZOS)
    row1.paste(im, (x, 0))
    x += tw + GAP

# Row 2: three 4:3 factor panels, optionally narrower than the page, centered.
fw_total = int(W * factor_frac)
fgap = int(6 / 72 * DPI)
pw = (fw_total - fgap * 2) // 3
ph = int(pw / (4 / 3))
row2 = Image.new("RGB", (W, ph + LABEL_H), "white")
d = ImageDraw.Draw(row2)
x = (W - fw_total) // 2
for fname, label in FACTORS:
    im = load(fname.rsplit(".", 1)[0]).resize((pw, ph), Image.LANCZOS)
    row2.paste(im, (x, 0))
    tw_, _ = d.textbbox((0, 0), label, font=font)[2:]
    d.text((x + (pw - tw_) // 2, ph + (LABEL_H - int(LABEL_PT / 72 * DPI)) // 2),
           label, fill="black", font=font)
    x += pw + fgap

H = th + ROW_GAP + ph + LABEL_H
out = Image.new("RGB", (W, H), "white")
out.paste(row1, (0, 0))
out.paste(row2, (0, th + ROW_GAP))
out.save(out_path, dpi=(DPI, DPI), optimize=True)
print(f"  {out_path}: {W}x{H}px = {W/DPI*72:.0f}x{H/DPI*72:.0f} pt "
      f"({H/DPI:.2f} in tall); thumbs {tw}x{th}px, factor panels {pw}x{ph}px "
      f"({pw/DPI:.2f} in wide)")
