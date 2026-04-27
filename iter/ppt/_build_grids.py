"""Build two 3-row x 2-col PNG grids for PPT.

Each row holds the same plot type from two runs side-by-side.
Source PNGs are trimmed of white margins, then each row picks a
common width per column so plots fill cells with minimal whitespace.
"""
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent.parent

SET_3STAGE_DI = ROOT / "iter/iter11/outputs/epoch15_LR5e5/full/surgery_3stage_DI"
SET_2STAGE_NODI = ROOT / "iter/iter11/outputs/epoch15_LR5e5/full/surgery_2stage_noDI"
SET_V9A = ROOT / "iter/iter9/v9a_10k_LR_1e6/full/m09c_surgery"

PLOT_FILES = ["m09_train_loss.png", "m09_val_loss_jepa.png", "probe_trajectory.png"]

OUT_GRID_SURGERY = ROOT / "iter/ppt/images/m09_surgery_2stage_noDI_vs_3stage_DI.png"
OUT_GRID_1V15 = ROOT / "iter/ppt/images/m09_1ep_vs_15ep_3stage_DI.png"

CELL_W = 1500
HEADER_H = 110
COL_GAP = 24
ROW_GAP = 18
SIDE_PAD = 10


def trim_white(im: Image.Image, margin: int = 8) -> Image.Image:
    bg = Image.new(im.mode, im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return im
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(im.width, x1 + margin)
    y1 = min(im.height, y1 + margin)
    return im.crop((x0, y0, x1, y1))


def load(p: Path) -> Image.Image:
    return trim_white(Image.open(p).convert("RGB"))


def fit_to_width(im: Image.Image, w: int) -> Image.Image:
    s = w / im.width
    return im.resize((w, int(round(im.height * s))), Image.LANCZOS)


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build(col1_dir: Path, col2_dir: Path, col1_label: str, col2_label: str,
          suptitle: str, out: Path) -> None:
    rows = []
    for fname in PLOT_FILES:
        im1 = fit_to_width(load(col1_dir / fname), CELL_W)
        im2 = fit_to_width(load(col2_dir / fname), CELL_W)
        rows.append((im1, im2))

    row_heights = [max(a.height, b.height) for a, b in rows]
    total_w = SIDE_PAD * 2 + CELL_W * 2 + COL_GAP
    total_h = HEADER_H + sum(row_heights) + ROW_GAP * (len(rows) - 1) + SIDE_PAD

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    f_sup = font(40)
    f_col = font(34)
    draw.text((total_w // 2, 14), suptitle, font=f_sup, fill="black", anchor="mt")

    col1_x_center = SIDE_PAD + CELL_W // 2
    col2_x_center = SIDE_PAD + CELL_W + COL_GAP + CELL_W // 2
    draw.text((col1_x_center, 70), col1_label, font=f_col, fill="#1565C0", anchor="mt")
    draw.text((col2_x_center, 70), col2_label, font=f_col, fill="#C62828", anchor="mt")

    y = HEADER_H
    for (im1, im2), rh in zip(rows, row_heights):
        x1 = SIDE_PAD + (CELL_W - im1.width) // 2
        x2 = SIDE_PAD + CELL_W + COL_GAP + (CELL_W - im2.width) // 2
        y1 = y + (rh - im1.height) // 2
        y2 = y + (rh - im2.height) // 2
        canvas.paste(im1, (x1, y1))
        canvas.paste(im2, (x2, y2))
        y += rh + ROW_GAP

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    pdf_out = out.with_suffix(".pdf")
    canvas.save(pdf_out, "PDF", resolution=150.0)
    print(f"  wrote {out.relative_to(ROOT)}  ({canvas.size[0]}×{canvas.size[1]})")
    print(f"  wrote {pdf_out.relative_to(ROOT)}")


print("=== Grid 1: surgery_2stage_noDI vs surgery_3stage_DI ===")
build(
    col1_dir=SET_3STAGE_DI,
    col2_dir=SET_2STAGE_NODI,
    col1_label="3-stage DI (drift-control ON)",
    col2_label="2-stage noDI (drift-control OFF)",
    suptitle="m09c Surgery · 2,452 clips × 15 ep × BS=32 × LR=5e-5 — DI vs noDI",
    out=OUT_GRID_SURGERY,
)

print("\n=== Grid 2: v9a 1 ep vs 15 ep 3-stage DI ===")
build(
    col1_dir=SET_V9A,
    col2_dir=SET_3STAGE_DI,
    col1_label="1 ep · 9,566 clips · LR=1e-6 (v9a baseline)",
    col2_label="15 ep · 2,452 clips · LR=5e-5 (3-stage DI)",
    suptitle="m09c Surgery · 1-epoch baseline vs 15-epoch run (both 3-stage DI)",
    out=OUT_GRID_1V15,
)
