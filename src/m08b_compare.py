"""
CPU-only multi-encoder comparison: grouped bar chart, radar plot, LaTeX table.
Reads m06_metrics_*.json for all available encoders. No GPU needed.

USAGE:
    python -u src/m08b_compare.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m08b_compare.log
    python -u src/m08b_compare.py --FULL 2>&1 | tee logs/m08b_compare_full.log
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    ENCODER_REGISTRY, add_subset_arg, get_output_dir, get_encoder_files,
)
from utils.wandb_utils import add_wandb_args, init_wandb, log_image, finish_wandb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Encoder display config ───────────────────────────────────────────

ENCODER_LABELS = {
    "vjepa": "V-JEPA 2\n(ViT-G, 1408d)",
    "random": "Random\n(1408d)",
    "dinov2": "DINOv2\n(ViT-L, 1024d)",
    "clip": "CLIP\n(ViT-L, 768d)",
    "vjepa_shuffled": "Shuffled\nV-JEPA (1408d)",
}
ENCODER_COLORS = {
    "vjepa": "#2196F3",
    "random": "#9E9E9E",
    "dinov2": "#4CAF50",
    "clip": "#FF9800",
    "vjepa_shuffled": "#E91E63",
}
# Display order: V-JEPA first (main), then baselines
ENCODER_ORDER = ["vjepa", "random", "dinov2", "clip", "vjepa_shuffled"]

METRICS_DISPLAY = [
    ("cycle_at_k", "Cycle@K (%)"),
    ("overlap_at_k", "Overlap@K (%)"),
    ("prec_at_k", "Prec@K (%)"),
    ("map_at_k", "mAP@K"),
    ("ndcg_at_k", "nDCG@K"),
    ("silhouette", "Silhouette"),
]


# ── Load all available encoder metrics ───────────────────────────────

def load_all_metrics(output_dir: Path) -> dict:
    """Load m06_metrics_*.json for all available encoders."""
    results = {}
    for name in ENCODER_ORDER:
        files = get_encoder_files(name, output_dir)
        if files["metrics"].exists():
            with open(files["metrics"]) as f:
                results[name] = json.load(f)
            print(f"  Loaded: {name} ({files['metrics'].name})")
    return results


# ── Terminal Summary Table ───────────────────────────────────────────

def print_summary_table(all_metrics: dict):
    """Pretty-print comparison table to terminal."""
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if not encoders:
        print("No encoder metrics found.")
        return

    # Header
    name_w = 18
    col_w = 12
    print(f"\n{'='*80}")
    print(f"{'ENCODER COMPARISON':^80}")
    print(f"{'='*80}")

    header = f"{'Encoder':<{name_w}} {'Dim':>5}"
    for _, label in METRICS_DISPLAY:
        header += f" {label:>{col_w}}"
    print(header)
    print("-" * len(header))

    for mode in ["easy", "hard"]:
        if mode == "hard":
            print(f"\n--- Hard Mode ---")
        for enc in encoders:
            m = all_metrics[enc].get(mode, {})
            dim = all_metrics[enc].get("encoder_dim", ENCODER_REGISTRY[enc]["dim"])
            row = f"{enc:<{name_w}} {dim:>5}"
            for key, _ in METRICS_DISPLAY:
                val = m.get(key)
                if val is None:
                    row += f" {'N/A':>{col_w}}"
                elif key in ("cycle_at_k", "overlap_at_k", "prec_at_k"):
                    row += f" {val:>{col_w}.2f}"
                else:
                    row += f" {val:>{col_w}.4f}"
            print(row)

        if mode == "easy":
            print(f"\n--- Easy Mode (above) ---")


# ── Grouped Bar Chart ────────────────────────────────────────────────

def create_bar_chart(all_metrics: dict, output_dir: Path):
    """Grouped bar chart: metrics x encoders, Easy mode."""
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2:
        print("Need at least 2 encoders for comparison chart.")
        return

    metrics_to_plot = [(k, l) for k, l in METRICS_DISPLAY if k != "silhouette"]
    n_metrics = len(metrics_to_plot)
    n_encoders = len(encoders)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), squeeze=False)
    axes = axes[0]

    for ax_idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]
        x = np.arange(n_encoders)
        easy_vals = []
        hard_vals = []
        colors = []

        for enc in encoders:
            easy_v = all_metrics[enc].get("easy", {}).get(metric_key)
            hard_v = all_metrics[enc].get("hard", {}).get(metric_key)
            easy_vals.append(easy_v if easy_v is not None else 0)
            hard_vals.append(hard_v if hard_v is not None else 0)
            colors.append(ENCODER_COLORS.get(enc, "#888"))

        bar_w = 0.35
        bars_easy = ax.bar(x - bar_w / 2, easy_vals, bar_w, label="Easy",
                           color=colors, alpha=0.85)
        bars_hard = ax.bar(x + bar_w / 2, hard_vals, bar_w, label="Hard",
                           color=colors, alpha=0.45, hatch="//")

        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([enc.replace("_", "\n") for enc in encoders],
                           fontsize=8, rotation=0)
        ax.tick_params(axis="y", labelsize=9)

        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper right")

    plt.suptitle("Encoder Comparison (Easy vs Hard)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_encoder_comparison{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_encoder_comparison.png'}")


# ── Radar Plot ───────────────────────────────────────────────────────

def create_radar_plot(all_metrics: dict, output_dir: Path):
    """Radar/spider plot with one polygon per encoder (Easy mode)."""
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2:
        return

    radar_metrics = ["cycle_at_k", "overlap_at_k", "prec_at_k", "map_at_k", "ndcg_at_k"]
    radar_labels = ["Cycle@K", "Overlap@K", "Prec@K", "mAP@K", "nDCG@K"]

    # Collect raw values per encoder
    raw = {}
    for enc in encoders:
        easy = all_metrics[enc].get("easy", {})
        raw[enc] = [easy.get(m) for m in radar_metrics]

    # Normalize each metric to [0, 100]
    n_metrics = len(radar_metrics)
    maxes = []
    for i in range(n_metrics):
        vals = [raw[e][i] for e in encoders if raw[e][i] is not None]
        maxes.append(max(vals) if vals else 1)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))
    for enc in encoders:
        vals = []
        for i, v in enumerate(raw[enc]):
            if v is not None and maxes[i] > 0:
                vals.append(v / maxes[i] * 100)
            else:
                vals.append(0)
        vals += vals[:1]  # close
        ax.plot(angles, vals, 'o-', label=enc, color=ENCODER_COLORS.get(enc, "#888"),
                linewidth=2, markersize=5)
        ax.fill(angles, vals, color=ENCODER_COLORS.get(enc, "#888"), alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_title("Encoder Comparison (Easy, normalized)", fontsize=12,
                 fontweight="bold", pad=20)
    ax.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.2, 0))
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_radar{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_radar.png'}")


# ── LaTeX Table ──────────────────────────────────────────────────────

def create_latex_table(all_metrics: dict, output_dir: Path):
    """Generate LaTeX table for paper (Easy mode)."""
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if not encoders:
        return

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Encoder comparison on 10K Indian urban clips (Easy mode, $k=6$)}")
    lines.append(r"\label{tab:encoder_comparison}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Encoder & Dim & Cycle@K & Overlap@K & Prec@K & mAP@K & nDCG@K \\")
    lines.append(r"\midrule")

    for enc in encoders:
        easy = all_metrics[enc].get("easy", {})
        dim = all_metrics[enc].get("encoder_dim", ENCODER_REGISTRY[enc]["dim"])
        label = enc.replace("_", r"\_")

        vals = []
        for key in ["cycle_at_k", "overlap_at_k", "prec_at_k", "map_at_k", "ndcg_at_k"]:
            v = easy.get(key)
            if v is None:
                vals.append("--")
            elif key in ("cycle_at_k", "overlap_at_k", "prec_at_k"):
                vals.append(f"{v:.2f}")
            else:
                vals.append(f"{v:.4f}")

        lines.append(f"{label} & {dim} & {' & '.join(vals)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_file = output_dir / "m08b_comparison_table.tex"
    with open(tex_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {tex_file}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-encoder comparison plots + LaTeX table (CPU-only)")
    parser.add_argument("--SANITY", action="store_true", help="Placeholder for consistency")
    parser.add_argument("--FULL", action="store_true", help="Process all available encoders")
    add_subset_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    output_dir = get_output_dir(args.subset, sanity=args.SANITY)
    print(f"Output dir: {output_dir}")
    print(f"Scanning for encoder metrics...")

    all_metrics = load_all_metrics(output_dir)
    if not all_metrics:
        print("FATAL: No encoder metrics found. Run m06 first.")
        sys.exit(1)

    print(f"\nFound {len(all_metrics)} encoder(s): {', '.join(all_metrics.keys())}")

    mode = "SANITY" if args.SANITY else ("POC" if args.subset else "FULL")
    wb_run = init_wandb("m08b", mode, config=vars(args), enabled=not args.no_wandb)

    # Terminal table
    print_summary_table(all_metrics)

    # Plots + table (need >= 2 encoders for comparison)
    if len(all_metrics) >= 2:
        create_bar_chart(all_metrics, output_dir)
        create_radar_plot(all_metrics, output_dir)
        create_latex_table(all_metrics, output_dir)

        for name in ["m08b_encoder_comparison", "m08b_radar"]:
            log_image(wb_run, name, str(output_dir / f"{name}.png"))
    else:
        print("Only 1 encoder found — skipping comparison plots (need >= 2).")

    finish_wandb(wb_run)
    print("\n=== COMPARISON COMPLETE ===")


if __name__ == "__main__":
    main()
