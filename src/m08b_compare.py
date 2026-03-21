"""
CPU-only multi-encoder comparison: grouped bar chart, spatial-temporal tradeoff scatter,
combined radar, LaTeX table with 95% CI. Reads m06_metrics_*.json + m06b_temporal_corr_*.json.

USAGE:
    python -u src/m08b_compare.py --SANITY 2>&1 | tee logs/m08b_compare_sanity.log
    python -u src/m08b_compare.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m08b_compare.log
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


def load_all_temporal(output_dir: Path) -> dict:
    """Load m06b_temporal_corr_*.json for all available encoders."""
    results = {}
    for name in ENCODER_ORDER:
        sfx = ENCODER_REGISTRY[name]["suffix"]
        temporal_file = output_dir / f"m06b_temporal_corr{sfx}.json"
        if temporal_file.exists():
            with open(temporal_file) as f:
                results[name] = json.load(f)
            print(f"  Loaded temporal: {name} ({temporal_file.name})")
    return results


TEMPORAL_METRICS_DISPLAY = [
    ("spearman_rho", "Spearman rho"),
    ("temporal_prec_at_k", "Temp Prec@K (%)"),
    ("motion_retrieval_map", "Motion mAP"),
]


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

        easy_errs = []
        hard_errs = []
        for enc in encoders:
            easy_v = all_metrics[enc].get("easy", {}).get(metric_key)
            hard_v = all_metrics[enc].get("hard", {}).get(metric_key)
            easy_vals.append(easy_v if easy_v is not None else 0)
            hard_vals.append(hard_v if hard_v is not None else 0)
            colors.append(ENCODER_COLORS.get(enc, "#888"))
            # 95% CI half-width for error bars
            easy_ci = all_metrics[enc].get("easy", {}).get("ci", {}).get(metric_key, {})
            hard_ci = all_metrics[enc].get("hard", {}).get("ci", {}).get(metric_key, {})
            easy_errs.append(easy_ci.get("ci_half", 0))
            hard_errs.append(hard_ci.get("ci_half", 0))

        bar_w = 0.35
        bars_easy = ax.bar(x - bar_w / 2, easy_vals, bar_w, label="Easy",
                           color=colors, alpha=0.85,
                           yerr=easy_errs, capsize=3, error_kw={"lw": 1})
        bars_hard = ax.bar(x + bar_w / 2, hard_vals, bar_w, label="Hard",
                           color=colors, alpha=0.45, hatch="//",
                           yerr=hard_errs, capsize=3, error_kw={"lw": 1})

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


# ── Spatial-Temporal Grouped Bar Chart ────────────────────────────────

def create_grouped_bar_chart(all_metrics: dict, all_temporal: dict, output_dir: Path):
    """Grouped bar chart: spatial metrics (left) | temporal metrics (right)."""
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2:
        return

    spatial_keys = [("prec_at_k", "Prec@K (%)"), ("map_at_k", "mAP@K"),
                    ("cycle_at_k", "Cycle@K (%)"), ("overlap_at_k", "Overlap@K (%)"),
                    ("ndcg_at_k", "nDCG@K")]
    temporal_keys = [(k, l) for k, l in TEMPORAL_METRICS_DISPLAY if any(
        all_temporal.get(e, {}).get(k) is not None for e in encoders)]

    # Add locality ratio if available
    if any(all_temporal.get(e, {}).get("temporal_locality", {}).get("ratio") is not None
           for e in encoders):
        temporal_keys.append(("temporal_locality_ratio", "Locality Ratio"))

    all_keys = spatial_keys + temporal_keys
    n_total = len(all_keys)
    n_spatial = len(spatial_keys)
    if n_total == 0:
        return

    fig, axes = plt.subplots(1, n_total, figsize=(3.2 * n_total, 5), squeeze=False)
    axes = axes[0]

    for ax_idx, (metric_key, metric_label) in enumerate(all_keys):
        ax = axes[ax_idx]
        x = np.arange(len(encoders))
        vals = []
        errs = []

        for enc in encoders:
            if ax_idx < n_spatial:
                # Spatial metric from m06
                v = all_metrics[enc].get("easy", {}).get(metric_key)
                ci_h = all_metrics[enc].get("easy", {}).get("ci", {}).get(metric_key, {}).get("ci_half", 0)
            elif metric_key == "temporal_locality_ratio":
                v = all_temporal.get(enc, {}).get("temporal_locality", {}).get("ratio")
                ci_h = 0
            else:
                v = all_temporal.get(enc, {}).get(metric_key)
                ci_h = all_temporal.get(enc, {}).get(f"{metric_key}_ci", {}).get("ci_half", 0)
                if ci_h == 0 and metric_key == "spearman_rho":
                    ci_h = all_temporal.get(enc, {}).get("spearman_rho_ci", {}).get("ci_half", 0)
            vals.append(v if v is not None else 0)
            errs.append(ci_h)

        colors = [ENCODER_COLORS.get(e, "#888") for e in encoders]
        ax.bar(x, vals, color=colors, alpha=0.85, yerr=errs, capsize=3, error_kw={"lw": 1})
        ax.set_title(metric_label, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([e.replace("_", "\n") for e in encoders], fontsize=7, rotation=0)
        ax.tick_params(axis="y", labelsize=8)

        # Separator between spatial and temporal
        if ax_idx == n_spatial - 1:
            ax.axvline(x=len(encoders) - 0.5, color="gray", linestyle="--", alpha=0.3)

    # Add group labels
    fig.text(n_spatial / (2 * n_total), 0.01, "SPATIAL", ha="center",
             fontsize=11, fontweight="bold", color="#2E7D32")
    if temporal_keys:
        fig.text((n_spatial + n_total) / (2 * n_total), 0.01, "TEMPORAL", ha="center",
                 fontsize=11, fontweight="bold", color="#C62828")

    plt.suptitle("Spatial vs Temporal Encoder Comparison (Easy mode, 95% CI)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_spatial_temporal_bar{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_spatial_temporal_bar.png'}")


# ── Spatial-Temporal Tradeoff Scatter ────────────────────────────────

def create_tradeoff_scatter(all_metrics: dict, all_temporal: dict, output_dir: Path):
    """2D scatter: X=spatial aggregate, Y=temporal aggregate. Each encoder = labeled point."""
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2 or not all_temporal:
        return

    # Spatial aggregate: mean of normalized (Prec@K, mAP@K, nDCG@K)
    spatial_keys = ["prec_at_k", "map_at_k", "ndcg_at_k"]
    # Temporal aggregate: mean of available temporal metrics (normalized)
    temporal_keys = ["spearman_rho", "temporal_prec_at_k", "motion_retrieval_map"]

    # Collect raw values
    spatial_raw = {enc: [] for enc in encoders}
    temporal_raw = {enc: [] for enc in encoders}

    for enc in encoders:
        for sk in spatial_keys:
            v = all_metrics[enc].get("easy", {}).get(sk)
            spatial_raw[enc].append(v if v is not None else 0)
        for tk in temporal_keys:
            v = all_temporal.get(enc, {}).get(tk)
            temporal_raw[enc].append(v if v is not None else 0)

    # Normalize each metric to [0, 1] across encoders
    def normalize_list(raw_dict, keys):
        scores = {}
        for i, k in enumerate(keys):
            vals = [raw_dict[e][i] for e in encoders]
            vmax = max(vals) if max(vals) > 0 else 1
            for e in encoders:
                scores.setdefault(e, []).append(raw_dict[e][i] / vmax)
        return {e: np.mean(v) * 100 for e, v in scores.items()}

    spatial_scores = normalize_list(spatial_raw, spatial_keys)
    temporal_scores = normalize_list(temporal_raw, temporal_keys)

    # Check if temporal has any non-zero values
    if all(v == 0 for v in temporal_scores.values()):
        print("  WARN: All temporal scores are 0 — skipping tradeoff scatter (need m04d + m06b first)")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for enc in encoders:
        sx = spatial_scores[enc]
        ty = temporal_scores[enc]
        color = ENCODER_COLORS.get(enc, "#888")
        ax.scatter(sx, ty, s=200, c=color, edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(enc.replace("_", "\n"), (sx, ty), textcoords="offset points",
                    xytext=(10, 10), fontsize=9, fontweight="bold", color=color)

    # Diagonal reference
    lim = max(max(spatial_scores.values()), max(temporal_scores.values())) * 1.15
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.3, label="Balanced")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Spatial Score (normalized)", fontsize=11)
    ax.set_ylabel("Temporal Score (normalized)", fontsize=11)
    ax.set_title("Spatial-Temporal Tradeoff", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_tradeoff_scatter{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_tradeoff_scatter.png'}")


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

        ci_data = easy.get("ci", {})
        vals = []
        for key in ["cycle_at_k", "overlap_at_k", "prec_at_k", "map_at_k", "ndcg_at_k"]:
            v = easy.get(key)
            ci_half = ci_data.get(key, {}).get("ci_half")
            if v is None:
                vals.append("--")
            elif key in ("cycle_at_k", "overlap_at_k", "prec_at_k"):
                if ci_half is not None:
                    vals.append(f"{v:.1f}{{\\tiny$\\pm${ci_half:.1f}}}")
                else:
                    vals.append(f"{v:.2f}")
            else:
                if ci_half is not None:
                    vals.append(f"{v:.3f}{{\\tiny$\\pm${ci_half:.3f}}}")
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

    all_temporal = load_all_temporal(output_dir)

    print(f"\nFound {len(all_metrics)} encoder(s): {', '.join(all_metrics.keys())}")
    if all_temporal:
        print(f"Found {len(all_temporal)} temporal result(s): {', '.join(all_temporal.keys())}")

    mode = "SANITY" if args.SANITY else ("POC" if args.subset else "FULL")
    wb_run = init_wandb("m08b", mode, config=vars(args), enabled=not args.no_wandb)

    # Terminal table
    print_summary_table(all_metrics)

    # Plots + table (need >= 2 encoders for comparison)
    if len(all_metrics) >= 2:
        create_bar_chart(all_metrics, output_dir)
        create_radar_plot(all_metrics, output_dir)
        create_latex_table(all_metrics, output_dir)
        create_grouped_bar_chart(all_metrics, all_temporal, output_dir)
        create_tradeoff_scatter(all_metrics, all_temporal, output_dir)

        for name in ["m08b_encoder_comparison", "m08b_radar",
                     "m08b_spatial_temporal_bar", "m08b_tradeoff_scatter"]:
            png = output_dir / f"{name}.png"
            if png.exists():
                log_image(wb_run, name, str(png))
    else:
        print("Only 1 encoder found — skipping comparison plots (need >= 2).")

    finish_wandb(wb_run)
    print("\n=== COMPARISON COMPLETE ===")


if __name__ == "__main__":
    main()
