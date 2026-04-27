"""One-off: re-render 6 source PNGs with wrapped (\\n) titles for PPT grids."""
import sys
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils.training import render_training_plots
from utils.plots import plot_training_curves


DIRS = [
    {
        "dir": ROOT / "iter/iter11/outputs/epoch15_LR5e5/full/surgery_3stage_DI",
        "n_train_clips": 2452, "n_epochs": 15, "total_steps": 1140,
        "batch_size": 32, "lr": 5.0e-05,
    },
    {
        "dir": ROOT / "iter/iter11/outputs/epoch15_LR5e5/full/surgery_2stage_noDI",
        "n_train_clips": 2452, "n_epochs": 15, "total_steps": 1140,
        "batch_size": 32, "lr": 5.0e-05,
    },
    {
        "dir": ROOT / "iter/iter9/v9a_10k_LR_1e6/full/m09c_surgery",
        "n_train_clips": 9566, "n_epochs": 1, "total_steps": 297,
        "batch_size": 32, "lr": 1.0e-06,
    },
]


def read_probe_history(p: Path) -> list:
    raw = p.read_text()
    if "}{" in raw:
        raw = re.sub(r"\}\{", "}\n{", raw)
    out = []
    skipped = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    if skipped:
        print(f"  [probe] skipped {skipped} torn record(s)")
    return out


for cfg in DIRS:
    d = cfg["dir"]
    print(f"\n=== {d.relative_to(ROOT)}")

    probe_history = read_probe_history(d / "probe_history.jsonl")
    print(f"  read {len(probe_history)} probe records")

    render_training_plots(
        probe_history,
        d,
        forgetting_threshold_pct=2.0,
        forgetting_patience=3,
        bwt_trigger_enabled=False,
        bwt_ci_fraction=0.5,
        bwt_absolute_floor=1.0,
        bwt_patience=3,
        kill_state={"triggered": False},
        best_state={"global_step": -1, "prec_at_k": 0.0},
        probe_compute_val_loss=True,
        verbose=True,
        n_train_clips=cfg["n_train_clips"],
        n_epochs=cfg["n_epochs"],
        total_steps=cfg["total_steps"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
    )

    csv_path = d / "loss_log.csv"
    title_prefix = (
        f"Surgery · {cfg['n_train_clips']:,} clips × {cfg['n_epochs']} ep × "
        f"BS={cfg['batch_size']} × LR={cfg['lr']:.1e} ({cfg['total_steps']:,} steps)\n"
    )
    plot_training_curves(
        [{"csv_path": str(csv_path), "label": "Surgery", "batch_size": cfg["batch_size"]}],
        str(d),
        title_prefix=title_prefix,
        x_axis_mode="steps",
    )

print("\n[done] re-rendered all 3 dirs.")
