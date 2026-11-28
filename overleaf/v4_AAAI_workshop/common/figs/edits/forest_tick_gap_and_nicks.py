"""Figure 3, step 3 after forest_numbers_9pt.py (visual audit Fri Nov 27 2026 ~9:55 PM PST, owner pick 'pixel fixes'):
move the '-100' tick label 4 px left and '-10' 4 px right (their ink gap was 23 px after the 8.7 pt swap, now 29 px),
and fill the two 1 px nicks where a bar's white outline painted over the frame's innermost column
(right frame x=1810, y 304-408; left frame x=772, y 2090-2194). Coordinates are of the 1842x2372 image.
usage: python -I forest_tick_gap_and_nicks.py <png, edited in place>
"""
import sys
import numpy as np
from PIL import Image

P = sys.argv[1]
a = np.asarray(Image.open(P).convert("RGB")).astype(int)
b = a.copy()
y0, y1 = 2291, 2352
for x0, x1, d in ((712, 892, -4), (909, 1039, 4)):
    edge = a[y0:y1, x0 - 4:x0] if d < 0 else a[y0:y1, x1:x1 + 4]
    assert (edge == 255).all(), "pixels the label moves into are not white"
    box = a[y0:y1, x0:x1].copy()
    b[y0:y1, x0:x1] = 255
    b[y0:y1, x0 + d:x1 + d] = np.where(box < 255, box, b[y0:y1, x0 + d:x1 + d])
for x, (ya, yb) in ((1810, (304, 408)), (772, (2090, 2194))):
    assert (a[ya:yb + 1, x] == 255).all() and (a[ya - 1, x] == 0).all() and (a[yb + 1, x] == 0).all()
    b[ya:yb + 1, x] = 0
Image.fromarray(b.astype(np.uint8)).save(P, dpi=(600, 600), optimize=True)
print("changed px:", int((np.abs(a - b).sum(-1) > 0).sum()))
