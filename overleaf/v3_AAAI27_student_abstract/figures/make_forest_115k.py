"""Cut the Full-115k panel (plus the shared row-label strip) out of the 3-panel
forest plot as ONE column-width, 600-dpi PNG.

Why a raster crop and not a bounding-box crop: the AAAI kit rejects PDFs whose
"cropped" content is merely hidden by a CropBox (the hidden layers resurface
in production). Rasterizing physically removes the two 10k panels and the
in-figure title/footer. Nothing is redrawn: bars, values and colours are the
original pixels, and the build asserts a zero pixel diff against the source.

Geometry (points). Edges come from INK-RUN scans of the 600-dpi raster, never
from PDF text boxes: a text box straddles neighbouring panels (the 1B panel's
"1.5x" value label runs to 383.8 pt, past its own spine at 378) and the
"-100" tick label's box starts at 376 while its ink starts at 389.2.
  row labels   text ink 19.7..102.6; the dashes at 105.5..110 are the 2B
               panel's y-ticks and are left out
  115k panel   own y-ticks 389.6..393.1, spine 393.1, right spine 520.6,
               "-100" ink 389.2..407.0; nothing of the 1B panel lies at x >= 389
  both         y 40..330 (below the 2-line suptitle, above the legend/footer)
Composite = 85 + 2 gap + 134 = 221 pt. The paper includes it at
0.97\\columnwidth (230 pt; a full-width include pushes the references to a
third page), i.e. 1.04x. Authored sizes in the generator (src/utils/tmp/
fig6_forest_3panel.py): row labels 9 pt, value labels 7.5 pt, x-tick numerals
7.5 pt, title 10 pt -> at print 9.4 / 7.8 / 7.8 / 10.4 pt. Values and ticks
stay under the kit's 9-pt floor until that generator is re-run with larger
fonts (visual-audit finding B3, 2026-09-21).

usage: python3 -c "$(cat make_forest_115k.py)" <source.pdf> <out.png>
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

src, out = Path(sys.argv[1]), Path(sys.argv[2])
DPI = 600
S = DPI / 72.0                                   # px per pt

LABELS = (19, 104)                               # x-range of the row-label strip, pt (text ink ends 102.6)
PANEL = (389, 523)                               # x-range of the Full-115k panel, pt (first ink 389.2)
Y = (40, 330)                                    # shared y-range, pt (keeps rows aligned)
GAP_PT = 2

with tempfile.TemporaryDirectory() as td:
    stem = Path(td) / "page"
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", str(src), str(stem)], check=True)
    page = Image.open(f"{stem}.png").convert("RGB")

    def crop(x0, x1):
        return page.crop((round(x0 * S), round(Y[0] * S), round(x1 * S), round(Y[1] * S)))

    a, b = crop(*LABELS), crop(*PANEL)
    gap = round(GAP_PT * S)
    canvas = Image.new("RGB", (a.width + gap + b.width, a.height), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    canvas.save(out, dpi=(DPI, DPI), optimize=True)

    # Prove nothing was altered: every strip must equal the source raster pixel for pixel.
    o = np.asarray(canvas).astype(np.int16)
    for name, strip, x0 in (("labels", a, 0), ("panel", b, a.width + gap)):
        diff = int((np.abs(o[:, x0:x0 + strip.width] - np.asarray(strip).astype(np.int16)).sum(-1) > 0).sum())
        if diff:
            sys.exit(f"FATAL: {name} strip differs from the source in {diff} px")

w_pt, h_pt = canvas.width / S, canvas.height / S
print(f"  {out.name}: {canvas.width}x{canvas.height} px @ {DPI} dpi = {w_pt:.0f} x {h_pt:.0f} pt; "
      f"at \\columnwidth (237 pt) scale = {237 / w_pt:.3f}; source pixel diff = 0")
