"""Thicken the small label text inside Figure 1 (``fig:teaser``, DENSEWORLD-115k teaser).

The teaser is a finished 2100x718 raster (300 dpi at \\textwidth). Its tag labels in
panels (a) Agents and (b) Layout, and the legend of panel (c) Interactions, print at
about 2 pt. Text size cannot change on a flat bitmap without redrawing, so this script
only thickens the existing strokes (owner pick Fri Nov 27 2026: "Thicken in place"):

* Work happens only inside the 28 label boxes below (inside each tag's border).
  Every pixel outside them is left byte-identical, and the script checks that.
* Per box, the text is the bright end of either the min channel (white text on a
  coloured fill, panel a) or the max channel (white or coloured text on a dark fill,
  panel b, legend c): whichever separates text from fill more.
* Faux bold: the text alpha is OR-ed with a copy of itself shifted 1 px right, at
  weight SHIFT_WEIGHT < 1 so small counters (e, a, o) stay open. New stroke pixels take
  the box's own text colour.

USAGE:
    python -u src/utils/fig_teaser_embolden.py \\
        --source overleaf/v4_AAAI_workshop/common/figs/denseworld_teaser.png \\
        --output overleaf/v4_AAAI_workshop/common/figs/denseworld_teaser_bold.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXPECTED_SIZE = (2100, 718)   # the boxes below are pixel positions in this exact image
SHIFT_WEIGHT = 0.7            # weight of the 1 px shifted copy (1.0 closes counters)
FILL_PCT, TEXT_PCT = 20, 98   # percentiles of the box that give the fill and the text level
MIN_CONTRAST = 100            # text minus fill level below this means the box is misplaced

# (x0, y0, x1, y1), inclusive, inside each label's border; found by flood-filling each
# tag's fill from a hand-placed seed, then checked by eye on before/after crops.
LABEL_BOXES = [
    # panel (a) Agents: Bus, 6x Pedestrian, Bicycle, Pedestrian, Auto-rickshaw,
    # Scooter, Car, 3x Scooter, Auto-rickshaw
    (211, 342, 233, 357), (55, 402, 93, 414), (108, 407, 141, 420), (165, 407, 193, 418),
    (29, 438, 64, 449), (115, 455, 150, 465), (243, 470, 280, 481), (311, 410, 341, 423),
    (352, 411, 387, 422), (417, 398, 473, 413), (499, 390, 521, 399), (539, 394, 556, 406),
    (571, 393, 595, 403), (603, 392, 630, 402), (627, 430, 654, 443), (504, 457, 561, 469),
    # panel (b) Layout: Elevated Metro, Metro Station, Pole, Streetlight, Trees, Roadside
    # Building, Sidewalk, Road, Zebra Crossing, Curb, Bus Stop
    (768, 168, 843, 193), (956, 190, 1020, 214), (847, 215, 916, 227), (1105, 187, 1170, 211),
    (1229, 293, 1277, 317), (722, 384, 796, 407), (726, 492, 799, 515), (1066, 502, 1159, 515),
    (1066, 534, 1139, 558), (732, 578, 809, 591), (1331, 541, 1379, 565),
    # panel (c) Interactions: the legend box
    (1421, 577, 1544, 644),
]


def _thicken_box(box: np.ndarray, where: tuple) -> np.ndarray:
    """Faux-bold the text in one label box; returns the new box (float RGB)."""
    best = None
    for level in (box.min(2), box.max(2)):
        fill, text = np.percentile(level, FILL_PCT), np.percentile(level, TEXT_PCT)
        if best is None or text - fill > best[2] - best[1]:
            best = (level, fill, text)
    level, fill, text = best
    if text - fill < MIN_CONTRAST:
        raise RuntimeError(f"label box {where}: text/fill contrast {text - fill:.0f} < "
                           f"{MIN_CONTRAST}; the box is not on a label")
    alpha = np.clip((level - fill) / (text - fill), 0, 1)
    text_rgb = box[alpha > 0.9].mean(0)
    shifted = np.zeros_like(alpha)
    shifted[:, 1:] = alpha[:, :-1]
    new_alpha = np.maximum(alpha, SHIFT_WEIGHT * shifted)
    gain = np.where(alpha < 1, (new_alpha - alpha) / np.maximum(1 - alpha, 1e-6), 0)
    return box + (text_rgb - box) * gain[..., None]


def embolden_labels(source: Path, output: Path) -> None:
    im = Image.open(source)
    if im.size != EXPECTED_SIZE:
        raise RuntimeError(f"{source} is {im.size}, expected {EXPECTED_SIZE}; "
                           f"LABEL_BOXES only fit the original teaser")
    src = np.asarray(im.convert("RGB")).astype(float)
    out = src.copy()
    inside = np.zeros(src.shape[:2], bool)
    for (x0, y0, x1, y1) in LABEL_BOXES:
        out[y0:y1 + 1, x0:x1 + 1] = _thicken_box(src[y0:y1 + 1, x0:x1 + 1], (x0, y0, x1, y1))
        inside[y0:y1 + 1, x0:x1 + 1] = True
    out8 = np.clip(out + 0.5, 0, 255).astype("uint8")
    src8 = src.astype("uint8")
    if not np.array_equal(out8[~inside], src8[~inside]):
        raise RuntimeError("pixels outside the label boxes changed")
    changed = int((out8 != src8).any(2).sum())
    Image.fromarray(out8).save(output, dpi=im.info.get("dpi", (300, 300)), optimize=True)
    print(f"Wrote: {output}  ({len(LABEL_BOXES)} label boxes, {changed:,} pixels changed, "
          f"all of them inside the boxes)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Thicken Figure 1 label text in place.")
    p.add_argument("--source", type=Path, required=True, help="original teaser PNG")
    p.add_argument("--output", type=Path, required=True, help="output PNG with thicker labels")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.source.exists():
        sys.exit(f"FATAL: --source not found: {args.source}")
    if args.source.resolve() == args.output.resolve():
        sys.exit("FATAL: --output must differ from --source (keep the original)")
    embolden_labels(args.source, args.output)


if __name__ == "__main__":
    main()
