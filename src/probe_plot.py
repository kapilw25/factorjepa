"""m08d — N-encoder visualization for the probe trio (action probe + motion cos + future MSE).

CPU-only. Pure-visualization function — like m08b, ALWAYS recomputes; no cache_policy.

Reads (auto-discovers ALL encoders present in the JSON `by_encoder` dict — no
hardcoded encoder list, so the same plotter works for {frozen, pretrain, surgical}
or any future fine-tuning technique that lands a row in by_encoder):
  <action-probe-root>/probe_paired_delta.json         {by_encoder, pairwise_deltas}
  <action-probe-root>/<enc>[/lr_*]/train_log.jsonl   probe train history (per-LR)
  <motion-cos-root>/probe_motion_cos_paired.json      {by_encoder, pairwise_deltas}
  <future-mse-root>/probe_future_mse_per_variant.json {by_variant: {<enc>: ... or "n/a"}}

Writes (under --output-dir):
  probe_action_loss.{png,pdf}       train loss per encoder/LR
  probe_action_acc.{png,pdf}        train (dashed) + val (solid) acc per encoder/LR
  probe_encoder_comparison.{png,pdf}      3-panel (acc, cos, mse) — N bars with 95% CI

USAGE:
  python -u src/probe_plot.py --SANITY \\
    --action-probe-root outputs/sanity/probe_action \\
    --motion-cos-root   outputs/sanity/probe_motion_cos \\
    --future-mse-root   outputs/sanity/probe_future_mse \\
    --output-dir        outputs/sanity/probe_plot \\
    2>&1 | tee logs/probe_plot_sanity.log
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


# ── Display helpers (no hardcoded encoder list — derived per-call) ───

def _display_label(enc: str) -> str:
    """Human-readable encoder name for plot legends. Falls back to enc verbatim."""
    return {
        "vjepa_2_1_frozen":   "V-JEPA 2.1 frozen",
        "vjepa_2_1_pretrain": "V-JEPA 2.1 pretrain",
        "vjepa_2_1_surgical": "V-JEPA 2.1 surgical",
        "dinov2":             "DINOv2 frozen",
    }.get(enc, enc.replace("_", " "))


# Color rotation for unrecognized encoder names (future fine-tuning techniques).
_FALLBACK_COLOR_CYCLE = ("blue", "green", "orange", "purple", "red", "cyan", "gold")


def _color_for(enc: str, idx: int) -> str:
    """Pick a color for an encoder bar. Tries the canonical map first
    (utils.plots.ENCODER_COLORS understands vjepa/vjepa_*/dinov2/clip/...),
    falls back to a deterministic rotation indexed by display-order position.
    """
    # Direct hit (e.g. "dinov2") or vjepa-family hit (anything starting with "vjepa")
    if enc in ENCODER_COLORS:
        return ENCODER_COLORS[enc]
    if enc.startswith("vjepa"):
        return ENCODER_COLORS["vjepa"]
    fallback_key = _FALLBACK_COLOR_CYCLE[idx % len(_FALLBACK_COLOR_CYCLE)]
    return COLORS.get(fallback_key, COLORS["gray"])


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


def _load_json(path: Path, stage_hint: str) -> dict:
    if not path.exists():
        sys.exit(f"FATAL: {path} not found — run {stage_hint} first")
    return json.loads(path.read_text())


# ── Plot 1: probe loss curves (auto-discovers encoders by directory presence) ─

def plot_loss_curves(action_probe_root: Path, encoders: list, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    drew = False
    for idx, enc in enumerate(encoders):
        color = _color_for(enc, idx)
        runs = _load_train_logs(action_probe_root, enc)
        if not runs:
            print(f"  [loss] no train_log.jsonl for {enc} — skipping")
            continue
        for i, run in enumerate(runs):
            recs = run["records"]
            x = [r["step"] for r in recs]
            y = [r["train_loss"] for r in recs]
            linestyle = "-" if i == 0 else "--"
            label = _display_label(enc) if len(runs) == 1 \
                else f"{_display_label(enc)} ({run['label']})"
            ax.plot(x, y, linestyle=linestyle, color=color, linewidth=2.5, label=label)
            drew = True
    if not drew:
        plt.close(fig)
        print("  [loss] no curves to plot — skipping figure")
        return
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Train cross-entropy loss")
    ax.set_title("probe Action Probe — Training Loss")
    ax.legend(loc="upper right")
    ax.set_yscale("log")
    save_fig(fig, str(output_dir / "probe_action_loss"))


# ── Plot 2: probe accuracy curves ────────────────────────────────────

def plot_acc_curves(action_probe_root: Path, encoders: list, n_classes: int,
                    output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    drew = False
    for idx, enc in enumerate(encoders):
        color = _color_for(enc, idx)
        runs = _load_train_logs(action_probe_root, enc)
        if not runs:
            continue
        for run in runs:
            recs = run["records"]
            x = [r["step"] for r in recs]
            tr = [r["train_acc"] for r in recs]
            va = [r["val_acc"]   for r in recs]
            sweep = "" if len(runs) == 1 else f" ({run['label']})"
            ax.plot(x, tr, linestyle="--", color=color, linewidth=1.6, alpha=0.6,
                    label=f"{_display_label(enc)} train{sweep}")
            ax.plot(x, va, linestyle="-", color=color, linewidth=2.6,
                    label=f"{_display_label(enc)} val{sweep}")
            drew = True
    if not drew:
        plt.close(fig)
        print("  [acc] no curves to plot — skipping figure")
        return
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.02)
    ax.axhline(1.0 / n_classes, color="black", linestyle=":", linewidth=1.0,
               label=f"chance ({n_classes}-class)")
    ax.set_title("probe Action Probe — Train (dashed) vs Val (solid) Accuracy")
    ax.legend(loc="lower right", fontsize=10)
    save_fig(fig, str(output_dir / "probe_action_acc"))


# ── Plot 3: 3-panel encoder comparison (N-bar generic) ───────────────

def _bar_with_ci(ax, encoders: list, vals: list, errs: list,
                 ylabel: str, title: str, na_set: set = None):
    """Render N bars with 95% CI error caps + value labels above each bar.
    `na_set` = encoder names with no measurement → render hatched 'N/A'.
    """
    na_set = na_set or set()
    x = np.arange(len(encoders))
    colors = [_color_for(e, i) for i, e in enumerate(encoders)]
    plot_vals = [0.0 if e in na_set else v for e, v in zip(encoders, vals)]
    plot_errs = [0.0 if e in na_set else er for e, er in zip(encoders, errs)]
    bars = ax.bar(x, plot_vals, 0.6, color=colors, alpha=0.85,
                  yerr=plot_errs, capsize=4, error_kw={"lw": 1.2, "ecolor": "#222"})
    for i, e in enumerate(encoders):
        if e in na_set:
            bars[i].set_hatch("//")
            bars[i].set_alpha(0.25)
    real_v = np.array([v for e, v in zip(encoders, plot_vals) if e not in na_set])
    real_e = np.array([er for e, er in zip(encoders, plot_errs) if e not in na_set])
    if real_v.size:
        # NaN-safe ylim: a degenerate BCa CI (perfect predictions on tiny test
        # sets — e.g. SANITY's N=22 with V-JEPA hitting 22/22) returns ci_half
        # = NaN, which propagates through .min()/.max() and makes
        # ax.set_ylim(NaN, NaN) raise ValueError. Substitute NaN errors with
        # 0 (renders as "no error bar" rather than crashing) BEFORE computing
        # ylim. matplotlib handles NaN in yerr itself fine — the bars draw
        # without error caps. Only the explicit set_ylim call needs sanitizing.
        real_e_safe = np.nan_to_num(real_e, nan=0.0)
        lo = max(0.0, float((real_v - real_e_safe).min()))
        hi = float((real_v + real_e_safe).max())
        pad = max(0.15 * (hi - lo), 0.02 * hi) if hi > 0 else 1.0
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
    else:
        pad = 0.05
    y_lo, y_hi = ax.get_ylim()
    for i, (xi, e, v, er) in enumerate(zip(x, encoders, plot_vals, plot_errs)):
        if e in na_set:
            ax.text(xi, y_lo + (y_hi - y_lo) * 0.5, "N/A", ha="center", va="center",
                    fontsize=12, color="#555", fontweight="bold")
        else:
            # NaN-safe value-label placement — degenerate-CI bars (NaN er) would
            # otherwise place text at v+NaN+... = NaN → matplotlib silently drops
            # the label. Substitute 0 so the label sits just above the bar top.
            er_safe = 0.0 if (isinstance(er, float) and np.isnan(er)) else er
            ax.text(xi, v + er_safe + pad * 0.1, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, color="#222")
    ax.set_xticks(x)
    ax.set_xticklabels([_display_label(e).replace(" ", "\n") for e in encoders], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")


def plot_encoder_comparison(paired_delta: dict, motion_cos: dict,
                            future_mse: dict, output_dir: Path):
    # Encoder set = union of all encoders that appear in any of the 3 JSONs'
    # by_encoder/by_variant blocks. Sorted for determinism. The same encoder
    # name may appear with `null` in future_mse.by_variant if its trainer
    # didn't run — we treat that as N/A in the MSE panel.
    ap_enc = set(paired_delta.get("by_encoder", {}).keys())
    mc_enc = set(motion_cos.get("by_encoder", {}).keys())
    bv = future_mse.get("by_variant", {})
    fm_enc = {k for k, v in bv.items() if isinstance(v, dict)}    # only real entries
    encoders = sorted(ap_enc | mc_enc | fm_enc)
    if not encoders:
        sys.exit("FATAL: no encoders found in any paired-Δ JSON")

    fig, axes = plt.subplots(1, 3, figsize=(max(12, 3.5 * len(encoders)), 5.5))

    # Panel 1 — top-1 action accuracy (per-encoder CI from by_encoder.top1_ci).
    ap_be = paired_delta.get("by_encoder", {})
    acc_vals = [ap_be.get(e, {}).get("acc_pct", 0.0) for e in encoders]
    acc_errs = [ap_be.get(e, {}).get("top1_ci", {}).get("ci_half", 0.0) * 100.0
                for e in encoders]
    ap_na = {e for e in encoders if e not in ap_be}
    _n_test = next(iter(ap_be.values()), {}).get("n", "?")
    _bar_with_ci(axes[0], encoders, acc_vals, acc_errs,
                 ylabel="Top-1 accuracy (%)",
                 title=f"Action probe (top-1, n={_n_test})",
                 na_set=ap_na)

    # Panel 2 — motion cosine intra-minus-inter (per-encoder CI from by_encoder.score_ci).
    mc_be = motion_cos.get("by_encoder", {})
    cos_vals = [mc_be.get(e, {}).get("score_mean", 0.0) for e in encoders]
    cos_errs = [mc_be.get(e, {}).get("score_ci", {}).get("ci_half", 0.0)
                for e in encoders]
    mc_na = {e for e in encoders if e not in mc_be}
    _n_cos = next(iter(mc_be.values()), {}).get("n", "?")
    _bar_with_ci(axes[1], encoders, cos_vals, cos_errs,
                 ylabel="Intra − Inter cosine",
                 title=f"Motion cosine (n={_n_cos})",
                 na_set=mc_na)

    # Panel 3 — future-frame L1 (V-JEPA-only by design — DINOv2/etc. have no predictor).
    mse_vals, mse_errs = [], []
    fm_na = set()
    n_test_mse = "?"
    for e in encoders:
        entry = bv.get(e)
        if isinstance(entry, dict):
            mse_vals.append(entry["mse_mean"])
            mse_errs.append(entry["mse_ci"]["ci_half"])
            n_test_mse = entry["n"]
        else:
            mse_vals.append(0.0)
            mse_errs.append(0.0)
            fm_na.add(e)
    _bar_with_ci(axes[2], encoders, mse_vals, mse_errs,
                 ylabel="Future L1 (lower = better)",
                 title=f"Future-frame MSE (n={n_test_mse})",
                 na_set=fm_na)

    fig.suptitle(f"probe trio · {len(encoders)} encoders · 95 % CI",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, str(output_dir / "probe_encoder_comparison"))


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="m08d — visualize probe action_probe + motion_cos + future_mse outputs (N-encoder generic).")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--action-probe-root", type=Path, required=True)
    p.add_argument("--motion-cos-root",   type=Path, required=True)
    p.add_argument("--future-mse-root",   type=Path, required=True)
    p.add_argument("--output-dir",        type=Path, required=True)
    add_wandb_args(p)
    args = p.parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        sys.exit("ERROR: specify --SANITY, --POC, or --FULL")
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")

    if args.output_dir.exists():
        n = sum(1 for _ in args.output_dir.rglob("*") if _.is_file())
        print(f"  [probe_plot] wiping output_dir ({args.output_dir.name}) — {n} stale file(s)")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wb = init_wandb("probe_plot", mode,
                    config=vars(args), enabled=not args.no_wandb)
    try:
        init_style()
        paired_delta = _load_json(args.action_probe_root / "probe_paired_delta.json", "Stage 4")
        motion_cos   = _load_json(args.motion_cos_root   / "probe_motion_cos_paired.json", "Stage 7")
        future_mse   = _load_json(args.future_mse_root   / "probe_future_mse_per_variant.json", "Stage 9")

        # Encoder list for loss/acc curves: discover from action_probe by_encoder.
        ap_be = paired_delta.get("by_encoder", {})
        if not ap_be:
            sys.exit("FATAL: paired_delta has empty by_encoder — re-run Stage 4 with N-way schema")
        encoders = sorted(ap_be.keys())
        n_classes = len(next(iter(ap_be.values())).get("top1_ci", {})) and 3
        # n_classes is best-effort; chance line on acc plot. Default to 3 if unknown.
        if not isinstance(n_classes, int) or n_classes < 2:
            n_classes = 3

        pbar = make_pbar(total=3, desc="probe_plot", unit="plot")
        plot_loss_curves(args.action_probe_root, encoders, args.output_dir);              pbar.update(1)
        plot_acc_curves(args.action_probe_root, encoders, n_classes, args.output_dir);    pbar.update(1)
        plot_encoder_comparison(paired_delta, motion_cos, future_mse, args.output_dir);   pbar.update(1)
        pbar.close()

        # wandb metric upload — generic prefix + encoder name (NO hardcoded vjepa/dinov2 keys).
        wb_metrics = {"n_encoders": len(encoders)}
        for e, v in ap_be.items():
            wb_metrics[f"acc_pct__{e}"] = float(v.get("acc_pct", 0.0))
        for e, v in motion_cos.get("by_encoder", {}).items():
            wb_metrics[f"motion_cos__{e}"] = float(v.get("score_mean", 0.0))
        for e, v in future_mse.get("by_variant", {}).items():
            if isinstance(v, dict):
                wb_metrics[f"future_mse_mean__{e}"] = float(v.get("mse_mean", 0.0))
        log_metrics(wb, wb_metrics)
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
