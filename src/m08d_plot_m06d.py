"""m08d — visualization for the m06d trio (action probe + motion cos + future MSE).

CPU-only. Pure-visualization function — like m08b, ALWAYS recomputes; no cache_policy.

Reads:
  <action-probe-root>/<enc>[/lr_*]/train_log.jsonl   probe train history (per-LR)
  <action-probe-root>/m06d_paired_delta.json         top-1 acc Δ + 95% CI
  <motion-cos-root>/m06d_motion_cos_paired.json      intra-inter cos Δ + 95% CI
  <future-mse-root>/m06d_future_mse_per_variant.json future L1 per variant + 95% CI

Writes (under --output-dir):
  m06d_action_probe_loss.{png,pdf}        train + val loss curves per encoder/LR
  m06d_action_probe_acc.{png,pdf}         train + val acc curves per encoder/LR
  m06d_encoder_comparison.{png,pdf}       3-panel (acc, cos, mse) — V-JEPA vs DINOv2 with 95% CI

USAGE:
  python -u src/m08d_plot_m06d.py --SANITY \\
    --action-probe-root outputs/sanity/m06d_action_probe \\
    --motion-cos-root   outputs/sanity/m06d_motion_cos \\
    --future-mse-root   outputs/sanity/m06d_future_mse \\
    --output-dir        outputs/sanity/m08d_plot_m06d \\
    2>&1 | tee logs/m08d_plot_m06d_sanity.log
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.plots import COLORS, ENCODER_COLORS, init_style, save_fig
from utils.progress import make_pbar
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics


# ── Encoders this plotter knows about ────────────────────────────────
# Display order = bar order in the comparison panel + curve order in legends.
# Color comes from ENCODER_COLORS in utils.plots (vjepa→blue, dinov2→green) so
# colors line up with every other m08* plot.
ENCODER_DISPLAY = (
    ("vjepa_2_1_frozen", "V-JEPA 2.1 frozen", "vjepa"),
    ("dinov2",           "DINOv2 frozen",     "dinov2"),
)

# ── Loaders ──────────────────────────────────────────────────────────

def _load_train_logs(action_probe_root: Path, encoder: str):
    """Return [{label, records[]}] — one entry per LR (or one if no sweep).

    Layout #1 (no sweep):  <action_probe_root>/<encoder>/train_log.jsonl
    Layout #2 (LR sweep):  <action_probe_root>/<encoder>/lr_*/train_log.jsonl
    """
    enc_dir = action_probe_root / encoder
    runs = []
    flat = enc_dir / "train_log.jsonl"
    if flat.exists():
        runs.append({"label": "single LR", "path": flat})
    for lr_dir in sorted(enc_dir.glob("lr_*")):
        p = lr_dir / "train_log.jsonl"
        if p.exists():
            runs.append({"label": lr_dir.name.replace("lr_", "lr="), "path": p})
    out = []
    for r in runs:
        records = []
        for line in r["path"].read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        if records:
            out.append({"label": r["label"], "records": records, "path": str(r["path"])})
    return out


def _load_paired_delta(action_probe_root: Path):
    p = action_probe_root / "m06d_paired_delta.json"
    if not p.exists():
        sys.exit(f"FATAL: {p} not found — run Stage 4 first")
    return json.loads(p.read_text())


def _load_motion_cos(motion_cos_root: Path):
    p = motion_cos_root / "m06d_motion_cos_paired.json"
    if not p.exists():
        sys.exit(f"FATAL: {p} not found — run Stage 7 first")
    return json.loads(p.read_text())


def _load_future_mse(future_mse_root: Path):
    p = future_mse_root / "m06d_future_mse_per_variant.json"
    if not p.exists():
        sys.exit(f"FATAL: {p} not found — run Stage 9 first")
    return json.loads(p.read_text())


# ── Plot 1: probe loss curves ────────────────────────────────────────

def plot_loss_curves(action_probe_root: Path, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    drew_anything = False
    for enc, enc_label, color_key in ENCODER_DISPLAY:
        color = ENCODER_COLORS.get(color_key, COLORS["gray"])
        runs = _load_train_logs(action_probe_root, enc)
        if not runs:
            print(f"  [loss] no train_log.jsonl for {enc} — skipping")
            continue
        for i, run in enumerate(runs):
            recs = run["records"]
            x = [r["step"] for r in recs]
            y = [r["train_loss"] for r in recs]
            linestyle = "-" if i == 0 else "--"
            label = f"{enc_label}" if len(runs) == 1 else f"{enc_label} ({run['label']})"
            ax.plot(x, y, linestyle=linestyle, color=color, linewidth=2.5, label=label)
            drew_anything = True
    if not drew_anything:
        print("  [loss] no curves to plot — skipping figure")
        plt.close(fig)
        return
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Train cross-entropy loss")
    ax.set_title("m06d Action Probe — Training Loss")
    ax.legend(loc="upper right")
    ax.set_yscale("log")  # CE drops several decades — log makes the long tail readable
    save_fig(fig, str(output_dir / "m06d_action_probe_loss"))


# ── Plot 2: probe accuracy curves ────────────────────────────────────

def plot_acc_curves(action_probe_root: Path, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    drew_anything = False
    for enc, enc_label, color_key in ENCODER_DISPLAY:
        color = ENCODER_COLORS.get(color_key, COLORS["gray"])
        runs = _load_train_logs(action_probe_root, enc)
        if not runs:
            continue
        for i, run in enumerate(runs):
            recs = run["records"]
            x = [r["step"] for r in recs]
            tr = [r["train_acc"] for r in recs]
            va = [r["val_acc"]   for r in recs]
            sweep_suffix = "" if len(runs) == 1 else f" ({run['label']})"
            ax.plot(x, tr, linestyle="--", color=color, linewidth=1.6, alpha=0.6,
                    label=f"{enc_label} train{sweep_suffix}")
            ax.plot(x, va, linestyle="-", color=color, linewidth=2.6,
                    label=f"{enc_label} val{sweep_suffix}")
            drew_anything = True
    if not drew_anything:
        print("  [acc] no curves to plot — skipping figure")
        plt.close(fig)
        return
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.02)
    ax.axhline(1.0 / 3, color="black", linestyle=":", linewidth=1.0,
               label="chance (3-class)")
    ax.set_title("m06d Action Probe — Train (dashed) vs Val (solid) Accuracy")
    ax.legend(loc="lower right", fontsize=10)
    save_fig(fig, str(output_dir / "m06d_action_probe_acc"))


# ── Plot 3: 3-panel encoder comparison ──────────────────────────────

def _bar_with_ci(ax, encoders, vals, errs, ylabel, title, na_idx=None):
    """One panel of the 3-panel comparison. na_idx marks bars to render hatched + 'N/A'."""
    x = np.arange(len(encoders))
    colors = [ENCODER_COLORS.get(c, COLORS["gray"]) for _, _, c in ENCODER_DISPLAY]
    plot_vals = [0.0 if (na_idx is not None and i in na_idx) else v
                 for i, v in enumerate(vals)]
    plot_errs = [0.0 if (na_idx is not None and i in na_idx) else e
                 for i, e in enumerate(errs)]
    bars = ax.bar(x, plot_vals, 0.6, color=colors, alpha=0.85,
                  yerr=plot_errs, capsize=4, error_kw={"lw": 1.2, "ecolor": "#222"})
    if na_idx:
        for i in na_idx:
            bars[i].set_hatch("//")
            bars[i].set_alpha(0.25)
    real_v = np.array([v for i, v in enumerate(plot_vals) if na_idx is None or i not in na_idx])
    real_e = np.array([e for i, e in enumerate(plot_errs) if na_idx is None or i not in na_idx])
    if real_v.size:
        lo = max(0.0, float((real_v - real_e).min()))
        hi = float((real_v + real_e).max())
        pad = max(0.15 * (hi - lo), 0.02 * hi) if hi > 0 else 1.0
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
    else:
        pad = 0.05
    y_lo, y_hi = ax.get_ylim()  # use actual visible range so N/A stays inside the panel
    for i, (xi, v, e) in enumerate(zip(x, plot_vals, plot_errs)):
        if na_idx is not None and i in na_idx:
            ax.text(xi, y_lo + (y_hi - y_lo) * 0.5, "N/A", ha="center", va="center",
                    fontsize=12, color="#555", fontweight="bold")
        else:
            ax.text(xi, v + e + pad * 0.1, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.replace(" ", "\n") for _, lbl, _ in ENCODER_DISPLAY], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")


def plot_encoder_comparison(paired_delta: dict, motion_cos: dict,
                            future_mse: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # Panel 1 — top-1 action accuracy. CI is the per-encoder pp interval, NOT the
    # paired-Δ interval (that's a different statistic — different audience).
    # paired_delta gives us point estimates only; we read each encoder's ci_half from
    # its own test_metrics.json. Fall back to ci_half_pp from the Δ if the per-encoder
    # file is missing (back-compat — earlier runs only emitted paired_delta).
    acc_vals = [paired_delta["vjepa_acc_pct"], paired_delta["dinov2_acc_pct"]]
    acc_errs = []
    for enc, _, _ in ENCODER_DISPLAY:
        tm_path = paired_delta["__per_encoder_paths"].get(enc)
        if tm_path and Path(tm_path).exists():
            tm = json.loads(Path(tm_path).read_text())
            acc_errs.append(tm["top1_ci"]["ci_half"] * 100.0)  # pp
        else:
            acc_errs.append(paired_delta["ci_half_pp"])
    _bar_with_ci(axes[0], ENCODER_DISPLAY, acc_vals, acc_errs,
                 ylabel="Top-1 accuracy (%)",
                 title=f"Action probe (top-1, n={paired_delta['n_clips_shared']})")

    # Panel 2 — motion cosine intra-minus-inter. Both encoders have CIs in this file.
    cos_vals = [motion_cos["vjepa_score_mean"], motion_cos["dinov2_score_mean"]]
    cos_errs = [motion_cos["vjepa_score_ci"]["ci_half"],
                motion_cos["dinov2_score_ci"]["ci_half"]]
    _bar_with_ci(axes[1], ENCODER_DISPLAY, cos_vals, cos_errs,
                 ylabel="Intra − Inter cosine",
                 title=f"Motion cosine (n={motion_cos['n_test']})")

    # Panel 3 — future-frame L1. DINOv2 has no predictor → mark N/A.
    bv = future_mse["by_variant"]
    vj = bv.get("vjepa_2_1_frozen")
    if vj is None or not isinstance(vj, dict):
        sys.exit("FATAL: future_mse JSON has no vjepa_2_1_frozen entry — run Stage 8")
    mse_vals = [vj["mse_mean"], 0.0]
    mse_errs = [vj["mse_ci"]["ci_half"], 0.0]
    _bar_with_ci(axes[2], ENCODER_DISPLAY, mse_vals, mse_errs,
                 ylabel="Future L1 (lower = better)",
                 title=f"Future-frame MSE (n={vj['n']})", na_idx={1})

    fig.suptitle("V-JEPA 2.1 frozen vs DINOv2 frozen — m06d trio · 95 % CI",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])  # reserve top 6% for suptitle
    save_fig(fig, str(output_dir / "m06d_encoder_comparison"))


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="m08d — visualize m06d action_probe + motion_cos + future_mse outputs.")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--action-probe-root", type=Path, required=True)
    p.add_argument("--motion-cos-root",   type=Path, required=True)
    p.add_argument("--future-mse-root",   type=Path, required=True)
    p.add_argument("--output-dir",        type=Path, required=True)
    add_wandb_args(p)
    # NO --cache-policy on purpose — m08d is pure visualization (mirrors m08b policy).
    args = p.parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        sys.exit("ERROR: specify --SANITY, --POC, or --FULL")
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")

    # Wipe + recreate output_dir — single-owner, always-recompute (m08b idiom).
    if args.output_dir.exists():
        n = sum(1 for _ in args.output_dir.rglob("*") if _.is_file())
        print(f"  [m08d] wiping output_dir ({args.output_dir.name}) — {n} stale file(s)")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wb = init_wandb("m08d_plot_m06d", mode,
                    config=vars(args), enabled=not args.no_wandb)
    try:
        init_style()
        paired_delta = _load_paired_delta(args.action_probe_root)
        # Attach per-encoder test_metrics paths so the comparison plot can pull
        # individual CI bars without re-walking the directory.
        paired_delta["__per_encoder_paths"] = {
            enc: str(args.action_probe_root / enc / "test_metrics.json")
            for enc, _, _ in ENCODER_DISPLAY
        }
        motion_cos  = _load_motion_cos(args.motion_cos_root)
        future_mse  = _load_future_mse(args.future_mse_root)

        pbar = make_pbar(total=3, desc="m08d_plot_m06d", unit="plot")
        plot_loss_curves(args.action_probe_root, args.output_dir);          pbar.update(1)
        plot_acc_curves(args.action_probe_root, args.output_dir);           pbar.update(1)
        plot_encoder_comparison(paired_delta, motion_cos, future_mse, args.output_dir); pbar.update(1)
        pbar.close()

        log_metrics(wb, {
            "vjepa_acc_pct":          float(paired_delta["vjepa_acc_pct"]),
            "dinov2_acc_pct":         float(paired_delta["dinov2_acc_pct"]),
            "delta_pp":               float(paired_delta["delta_pp"]),
            "vjepa_cos_score":        float(motion_cos["vjepa_score_mean"]),
            "dinov2_cos_score":       float(motion_cos["dinov2_score_mean"]),
            "vjepa_future_mse_mean":  float(future_mse["by_variant"]["vjepa_2_1_frozen"]["mse_mean"]),
        })
        print(f"  Plots written to: {args.output_dir}")
    finally:
        finish_wandb(wb)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        print(f"\n❌ FATAL: {Path(__file__).name} crashed — see traceback below", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
