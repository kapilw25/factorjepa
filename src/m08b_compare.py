"""
CPU-only multi-encoder comparison: grouped bar chart, spatial-temporal tradeoff scatter,
combined radar, LaTeX table with 95% CI. Reads m06_metrics_*.json + m06b_temporal_corr_*.json.

USAGE:
    python -u src/m08b_compare.py --SANITY 2>&1 | tee logs/m08b_compare_sanity.log
    python -u src/m08b_compare.py --POC --subset data/subset_10k.json 2>&1 | tee logs/m08b_compare_poc.log
    python -u src/m08b_compare.py --FULL 2>&1 | tee logs/m08b_compare_full.log
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.progress import make_pbar
from utils.config import (
    ENCODER_REGISTRY, add_subset_arg, get_output_dir, get_encoder_files,
    get_encoder_info,
)
from utils.wandb_utils import add_wandb_args, init_wandb, log_image, finish_wandb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Encoder display config ───────────────────────────────────────────

ENCODER_LABELS = {
    "vjepa": "V-JEPA 2\n(frozen, 1408d)",
    "random": "Random\n(1408d)",
    "dinov2": "DINOv2\n(ViT-L, 1024d)",
    "clip": "CLIP\n(ViT-L, 768d)",
    "vjepa_shuffled": "Shuffled\nV-JEPA (frozen)",
    "vjepa_adapted": "V-JEPA 2\n(adapted, 1408d)",
    "vjepa_surgical": "V-JEPA 2\n(surgical, 1408d)",
}
ENCODER_COLORS = {
    "vjepa": "#2196F3",
    "random": "#9E9E9E",
    "dinov2": "#4CAF50",
    "clip": "#FF9800",
    "vjepa_shuffled": "#E91E63",
    "vjepa_adapted": "#7B1FA2",
    "vjepa_surgical": "#00695C",
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

def load_all_metrics(output_dir: Path, encoder_list: list = None) -> dict:
    """Load m06_metrics_*.json for specified or all registered encoders."""
    results = {}
    names = encoder_list if encoder_list else ENCODER_ORDER
    for name in names:
        files = get_encoder_files(name, output_dir)
        if files["metrics"].exists():
            with open(files["metrics"]) as f:
                results[name] = json.load(f)
            print(f"  Loaded: {name} ({files['metrics'].name})")
    return results


def load_all_temporal(output_dir: Path, encoder_list: list = None) -> dict:
    """Load m06b_temporal_corr_*.json for specified or all registered encoders."""
    results = {}
    names = encoder_list if encoder_list else ENCODER_ORDER
    for name in names:
        sfx = get_encoder_info(name)["suffix"]
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
            dim = all_metrics[enc].get("encoder_dim", get_encoder_info(enc)["dim"])
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
        ax.bar(x - bar_w / 2, easy_vals, bar_w, label="Easy",
               color=colors, alpha=0.85,
               yerr=easy_errs, capsize=3, error_kw={"lw": 1})
        ax.bar(x + bar_w / 2, hard_vals, bar_w, label="Hard",
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


# ── Radar Plot (Spatial + Temporal hemispheres) ─────────────────────

def create_radar_plot(all_metrics: dict, output_dir: Path,
                      all_temporal: dict = None):
    """Combined radar: 5 spatial axes (left) + 3 temporal axes (right).

    Spatial from m06, temporal from m06b. Hemisphere shading shows the
    spatial-temporal tradeoff at a glance: DINOv2 bulges left, V-JEPA
    bulges right, shuffled loses right side, random stays near center.
    """
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2:
        return

    # 5 spatial + 3 temporal axes (ordered so spatial=left, temporal=right)
    spatial_keys = ["prec_at_k", "map_at_k", "cycle_at_k", "overlap_at_k", "ndcg_at_k"]
    spatial_labels = ["Prec@K", "mAP@K", "Cycle@K", "Overlap@K", "nDCG@K"]

    temporal_keys = ["spearman_rho", "temporal_prec_at_k", "temporal_locality_inv"]
    temporal_labels = ["Spearman \u03c1", "Temp Prec@K", "Temp Locality"]

    has_temporal = all_temporal and any(
        all_temporal.get(e, {}).get("spearman_rho") is not None for e in encoders)

    if has_temporal:
        all_keys = spatial_keys + temporal_keys
        all_labels = spatial_labels + temporal_labels
    else:
        all_keys = spatial_keys
        all_labels = spatial_labels

    n_metrics = len(all_keys)
    n_spatial = len(spatial_keys)

    # Collect raw values
    raw = {}
    for enc in encoders:
        easy = all_metrics[enc].get("easy", {})
        vals = []
        for k in all_keys:
            if k in spatial_keys:
                vals.append(easy.get(k))
            elif k == "temporal_locality_inv":
                # Invert locality ratio: lower=better, so plot 1-ratio
                ratio = (all_temporal or {}).get(enc, {}).get(
                    "temporal_locality", {}).get("ratio")
                vals.append(1.0 - ratio if ratio is not None else None)
            else:
                vals.append((all_temporal or {}).get(enc, {}).get(k))
        raw[enc] = vals

    # Normalize each axis to [0, 100]
    maxes = []
    for i in range(n_metrics):
        axis_vals = [raw[e][i] for e in encoders if raw[e][i] is not None]
        maxes.append(max(axis_vals) if axis_vals else 1)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

    # Hemisphere shading (spatial left, temporal right)
    if has_temporal:
        spatial_start = angles[0]
        spatial_end = angles[n_spatial - 1]
        temporal_start = angles[n_spatial]
        temporal_end = angles[-2]
        ax.fill_between(
            np.linspace(spatial_start - 0.15, spatial_end + 0.15, 50),
            0, 115, alpha=0.04, color="#2E7D32", zorder=0)
        ax.fill_between(
            np.linspace(temporal_start - 0.15, temporal_end + 0.15, 50),
            0, 115, alpha=0.04, color="#C62828", zorder=0)

    # Plot each encoder polygon
    for enc in encoders:
        vals = []
        for i, v in enumerate(raw[enc]):
            if v is not None and maxes[i] > 0:
                vals.append(v / maxes[i] * 100)
            else:
                vals.append(0)
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', label=ENCODER_LABELS.get(enc, enc).replace('\n', ' '),
                color=ENCODER_COLORS.get(enc, "#888"), linewidth=2, markersize=5)
        ax.fill(angles, vals, color=ENCODER_COLORS.get(enc, "#888"), alpha=0.08)

    # Axis labels with color coding
    ax.set_xticks(angles[:-1])
    label_colors = (["#2E7D32"] * n_spatial +
                    ["#C62828"] * (n_metrics - n_spatial))
    labels = ax.set_xticklabels(all_labels, fontsize=10)
    for label, color in zip(labels, label_colors):
        label.set_color(color)
        label.set_fontweight("bold")

    ax.set_ylim(0, 115)
    title = "Spatial + Temporal Encoder Comparison" if has_temporal else "Encoder Comparison"
    ax.set_title(f"{title} (Easy, normalized)", fontsize=12,
                 fontweight="bold", pad=25)
    ax.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.3, -0.05))

    # Add hemisphere labels
    if has_temporal:
        fig.text(0.15, 0.92, "SPATIAL", fontsize=11, fontweight="bold",
                 color="#2E7D32", ha="center")
        fig.text(0.85, 0.92, "TEMPORAL", fontsize=11, fontweight="bold",
                 color="#C62828", ha="center")

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_radar{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_radar.png'}")


# ── Spatial-Temporal Bar Chart (2-row layout) ──────────────────────

def create_grouped_bar_chart(all_metrics: dict, all_temporal: dict, output_dir: Path):
    """2-row bar chart: spatial metrics (top row) | temporal metrics (bottom row).

    Each subplot = one metric, bars = encoders. Green header for spatial,
    red header for temporal. 95% CI error bars on all metrics.
    """
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2:
        return

    spatial_keys = [("prec_at_k", "Prec@K (%)"), ("map_at_k", "mAP@K"),
                    ("cycle_at_k", "Cycle@K (%)"), ("overlap_at_k", "Overlap@K (%)"),
                    ("ndcg_at_k", "nDCG@K")]
    temporal_keys = [(k, l) for k, l in TEMPORAL_METRICS_DISPLAY if any(
        all_temporal.get(e, {}).get(k) is not None for e in encoders)]

    if any(all_temporal.get(e, {}).get("temporal_locality", {}).get("ratio") is not None
           for e in encoders):
        temporal_keys.append(("temporal_locality_ratio", "Locality Ratio"))

    n_spatial = len(spatial_keys)
    n_temporal = len(temporal_keys)
    n_cols = max(n_spatial, n_temporal) if n_temporal > 0 else n_spatial
    n_rows = 2 if n_temporal > 0 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    def _plot_bar(ax, metric_key, metric_label, source, title_color):
        x = np.arange(len(encoders))
        vals, errs = [], []
        for enc in encoders:
            if source == "spatial":
                v = all_metrics[enc].get("easy", {}).get(metric_key)
                ci_h = all_metrics[enc].get("easy", {}).get("ci", {}).get(
                    metric_key, {}).get("ci_half", 0)
            elif metric_key == "temporal_locality_ratio":
                v = all_temporal.get(enc, {}).get("temporal_locality", {}).get("ratio")
                ci_h = all_temporal.get(enc, {}).get("temporal_locality", {}).get(
                    "ratio_ci", {}).get("ci_half", 0)
            else:
                v = all_temporal.get(enc, {}).get(metric_key)
                ci_h = all_temporal.get(enc, {}).get(f"{metric_key}_ci", {}).get("ci_half", 0)
                if ci_h == 0 and metric_key == "spearman_rho":
                    ci_h = all_temporal.get(enc, {}).get("spearman_rho_ci", {}).get("ci_half", 0)
            vals.append(v if v is not None else 0)
            errs.append(ci_h)

        colors = [ENCODER_COLORS.get(e, "#888") for e in encoders]
        ax.bar(x, vals, color=colors, alpha=0.85, yerr=errs, capsize=3, error_kw={"lw": 1})
        ax.set_title(metric_label, fontsize=10, fontweight="bold", color=title_color)
        ax.set_xticks(x)
        ax.set_xticklabels([e.replace("_", "\n") for e in encoders], fontsize=7)
        ax.tick_params(axis="y", labelsize=8)

    # Top row: spatial metrics
    for i, (key, label) in enumerate(spatial_keys):
        _plot_bar(axes[0][i], key, label, "spatial", "#2E7D32")
    # Hide unused top-row axes
    for i in range(n_spatial, n_cols):
        axes[0][i].set_visible(False)

    # Bottom row: temporal metrics
    if n_temporal > 0:
        for i, (key, label) in enumerate(temporal_keys):
            _plot_bar(axes[1][i], key, label, "temporal", "#C62828")
        for i in range(n_temporal, n_cols):
            axes[1][i].set_visible(False)

    # Row labels
    fig.text(0.02, 0.75, "SPATIAL", fontsize=13, fontweight="bold",
             color="#2E7D32", rotation=90, va="center")
    if n_temporal > 0:
        fig.text(0.02, 0.28, "TEMPORAL", fontsize=13, fontweight="bold",
                 color="#C62828", rotation=90, va="center")

    plt.suptitle("Spatial vs Temporal Encoder Comparison (Easy mode, 95% CI)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0.04, 0, 1, 0.98])
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
        print("  FATAL: All temporal scores are 0. Run m04d + m06b first.")
        sys.exit(1)

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


# ── Temporal Order Ablation (V-JEPA vs Shuffled) ───────────────────

def create_ablation_chart(all_metrics: dict, all_temporal: dict, output_dir: Path):
    """Paired bar chart: V-JEPA (normal) vs Shuffled on ALL metrics.

    Spatial metrics: shuffled >= normal proves temporal encoding hurts spatial.
    Temporal metrics: normal >> shuffled proves V-JEPA learns temporal structure.
    Delta annotation with propagated 95% CI on each pair.
    """
    if "vjepa" not in all_metrics or "vjepa_shuffled" not in all_metrics:
        print("  SKIP ablation chart: need both vjepa and vjepa_shuffled")
        return

    # Collect all metrics for both encoders
    spatial_defs = [
        ("prec_at_k", "Prec@K (%)", "spatial"),
        ("map_at_k", "mAP@K", "spatial"),
        ("cycle_at_k", "Cycle@K (%)", "spatial"),
        ("overlap_at_k", "Overlap@K (%)", "spatial"),
        ("ndcg_at_k", "nDCG@K", "spatial"),
    ]
    temporal_defs = [
        ("spearman_rho", "Spearman \u03c1", "temporal"),
        ("temporal_prec_at_k", "Temp Prec@K (%)", "temporal"),
        ("motion_retrieval_map", "Motion mAP", "temporal"),
    ]
    # Add temporal locality if available
    if all_temporal and all_temporal.get("vjepa", {}).get("temporal_locality", {}).get("ratio") is not None:
        temporal_defs.append(("temporal_locality_ratio", "Locality\n(lower=better)", "temporal"))

    has_temporal = all_temporal and any(
        all_temporal.get("vjepa", {}).get(k) is not None for k, _, _ in temporal_defs[:3])

    n_spatial_d = len(spatial_defs)
    n_temporal_d = len(temporal_defs) if has_temporal else 0
    n_cols = max(n_spatial_d, n_temporal_d) if n_temporal_d > 0 else n_spatial_d
    n_rows = 2 if n_temporal_d > 0 else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    def _ablation_bar(ax, key, label, domain):
        vals, errs = [], []
        for enc in ["vjepa", "vjepa_shuffled"]:
            if domain == "spatial":
                v = all_metrics[enc].get("easy", {}).get(key)
                ci_h = all_metrics[enc].get("easy", {}).get("ci", {}).get(
                    key, {}).get("ci_half", 0)
            elif key == "temporal_locality_ratio":
                v = (all_temporal or {}).get(enc, {}).get(
                    "temporal_locality", {}).get("ratio")
                ci_h = 0
            else:
                v = (all_temporal or {}).get(enc, {}).get(key)
                ci_h = (all_temporal or {}).get(enc, {}).get(
                    f"{key}_ci", {}).get("ci_half", 0)
                if ci_h == 0 and key == "spearman_rho":
                    ci_h = (all_temporal or {}).get(enc, {}).get(
                        "spearman_rho_ci", {}).get("ci_half", 0)
            vals.append(v if v is not None else 0)
            errs.append(ci_h)

        normal_v, shuffled_v = vals
        normal_e, shuffled_e = errs
        x = np.array([0, 1])
        ax.bar(x, vals, color=["#2196F3", "#E91E63"], alpha=0.85,
               yerr=errs, capsize=5, error_kw={"lw": 1.5})

        delta = normal_v - shuffled_v
        delta_ci = np.sqrt(normal_e**2 + shuffled_e**2)
        delta_sign = "+" if delta > 0 else ""
        if key == "temporal_locality_ratio":
            expected = delta < 0
        elif domain == "temporal":
            expected = delta > 0
        else:
            expected = delta < 0 or key == "cycle_at_k"
        delta_color = "#2E7D32" if expected else "#C62828"
        ci_str = f" \u00b1{delta_ci:.2f}" if delta_ci > 0 else ""
        ax.annotate(f"\u0394={delta_sign}{delta:.2f}{ci_str}",
                    xy=(0.5, max(vals) * 1.05), fontsize=7, fontweight="bold",
                    color=delta_color, ha="center")
        title_color = "#2E7D32" if domain == "spatial" else "#C62828"
        ax.set_title(label, fontsize=9, fontweight="bold", color=title_color)
        ax.set_xticks(x)
        ax.set_xticklabels(["V-JEPA\n(normal)", "V-JEPA\n(shuffled)"], fontsize=7)
        ax.tick_params(axis="y", labelsize=8)

    # Top row: spatial
    for i, (key, label, domain) in enumerate(spatial_defs):
        _ablation_bar(axes[0][i], key, label, domain)
    for i in range(n_spatial_d, n_cols):
        axes[0][i].set_visible(False)

    # Bottom row: temporal
    if n_temporal_d > 0:
        for i, (key, label, domain) in enumerate(temporal_defs):
            _ablation_bar(axes[1][i], key, label, domain)
        for i in range(n_temporal_d, n_cols):
            axes[1][i].set_visible(False)

    fig.text(0.02, 0.75, "SPATIAL", fontsize=13, fontweight="bold",
             color="#2E7D32", rotation=90, va="center")
    if n_temporal_d > 0:
        fig.text(0.02, 0.28, "TEMPORAL", fontsize=13, fontweight="bold",
                 color="#C62828", rotation=90, va="center")

    plt.suptitle("Temporal Order Ablation: Does Frame Order Matter?",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.text(0.5, -0.01,
             "Green \u0394 = expected  |  "
             "Red \u0394 = unexpected (temporal encoding not helping)",
             ha="center", fontsize=8, color="#555", style="italic")
    plt.tight_layout(rect=[0.04, 0.02, 1, 0.98])
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_temporal_ablation{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_temporal_ablation.png'}")


# ── Normalized Heatmap (Encoders × Metrics) ────────────────────────

def create_heatmap(all_metrics: dict, all_temporal: dict, output_dir: Path):
    """Encoder × metric heatmap. Spatial columns left, temporal right.

    Cell text shows value ± CI. Color intensity = normalized rank across
    encoders (per column). Visual: DINOv2 row hot left, V-JEPA hot right.
    """
    encoders = [e for e in ENCODER_ORDER if e in all_metrics]
    if len(encoders) < 2:
        return

    spatial_defs = [
        ("prec_at_k", "Prec@K"),
        ("map_at_k", "mAP@K"),
        ("cycle_at_k", "Cycle@K"),
        ("overlap_at_k", "Overlap@K"),
        ("ndcg_at_k", "nDCG@K"),
    ]
    temporal_defs = []
    if all_temporal and any(
            all_temporal.get(e, {}).get("spearman_rho") is not None for e in encoders):
        temporal_defs = [
            ("spearman_rho", "Spearman \u03c1"),
            ("temporal_prec_at_k", "Temp Prec@K"),
            ("motion_retrieval_map", "Motion mAP"),
        ]

    all_defs = spatial_defs + temporal_defs
    n_spatial = len(spatial_defs)
    n_metrics = len(all_defs)
    n_enc = len(encoders)

    # Build value + CI matrices
    val_matrix = np.zeros((n_enc, n_metrics))
    ci_matrix = np.zeros((n_enc, n_metrics))

    for ei, enc in enumerate(encoders):
        for mi, (key, _) in enumerate(all_defs):
            if mi < n_spatial:
                v = all_metrics[enc].get("easy", {}).get(key)
                ci_h = all_metrics[enc].get("easy", {}).get("ci", {}).get(
                    key, {}).get("ci_half", 0)
            else:
                v = (all_temporal or {}).get(enc, {}).get(key)
                ci_h = (all_temporal or {}).get(enc, {}).get(
                    f"{key}_ci", {}).get("ci_half", 0)
                if ci_h == 0 and key == "spearman_rho":
                    ci_h = (all_temporal or {}).get(enc, {}).get(
                        "spearman_rho_ci", {}).get("ci_half", 0)
            val_matrix[ei, mi] = v if v is not None else 0
            # Scale CI to match display units: cycle_at_k CI is in proportion
            # (0.008) but value is percentage (76.0) — multiply CI by 100
            if key == "cycle_at_k" and ci_h > 0 and ci_h < 1:
                ci_h = ci_h * 100
            ci_matrix[ei, mi] = ci_h

    # Normalize per column to [0, 1] for color mapping
    norm_matrix = np.zeros_like(val_matrix)
    for mi in range(n_metrics):
        col = val_matrix[:, mi]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            norm_matrix[:, mi] = (col - cmin) / (cmax - cmin)
        else:
            norm_matrix[:, mi] = 0.5

    # Custom colormap: spatial=green shades, temporal=red shades
    from matplotlib.colors import LinearSegmentedColormap
    spatial_cmap = LinearSegmentedColormap.from_list("spatial", ["#FFFFFF", "#2E7D32"])
    temporal_cmap = LinearSegmentedColormap.from_list("temporal", ["#FFFFFF", "#C62828"])

    fig, ax = plt.subplots(1, 1, figsize=(max(10, 1.8 * n_metrics), 0.9 * n_enc + 2))

    # Draw cells with appropriate colormap per column
    for mi in range(n_metrics):
        cmap = spatial_cmap if mi < n_spatial else temporal_cmap
        for ei in range(n_enc):
            color = cmap(norm_matrix[ei, mi] * 0.85 + 0.05)  # avoid pure white
            ax.add_patch(plt.Rectangle((mi, n_enc - 1 - ei), 1, 1,
                                       facecolor=color, edgecolor="white", lw=2))
            # Cell text: value ± CI
            v = val_matrix[ei, mi]
            ci = ci_matrix[ei, mi]
            key = all_defs[mi][0]
            if key in ("prec_at_k", "cycle_at_k", "overlap_at_k", "temporal_prec_at_k"):
                txt = f"{v:.1f}"
                if ci > 0:
                    txt += f"\n\u00b1{ci:.1f}"
            else:
                txt = f"{v:.3f}"
                if ci > 0:
                    txt += f"\n\u00b1{ci:.3f}"
            text_color = "white" if norm_matrix[ei, mi] > 0.6 else "black"
            ax.text(mi + 0.5, n_enc - 0.5 - ei, txt, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=text_color)

    # Separator line between spatial and temporal
    if temporal_defs:
        ax.axvline(x=n_spatial, color="#333", linewidth=2, linestyle="-")

    # Labels
    enc_labels = [ENCODER_LABELS.get(e, e).replace('\n', ' ') for e in encoders]
    ax.set_yticks(np.arange(n_enc) + 0.5)
    ax.set_yticklabels(enc_labels[::-1], fontsize=9)
    ax.set_xticks(np.arange(n_metrics) + 0.5)
    metric_labels = [d[1] for d in all_defs]
    xlabels = ax.set_xticklabels(metric_labels, fontsize=9, rotation=30, ha="right")
    for i, xl in enumerate(xlabels):
        xl.set_color("#2E7D32" if i < n_spatial else "#C62828")
        xl.set_fontweight("bold")

    ax.set_xlim(0, n_metrics)
    ax.set_ylim(0, n_enc)

    # Group headers
    if temporal_defs:
        ax.text(n_spatial / 2, n_enc + 0.3, "SPATIAL", ha="center",
                fontsize=12, fontweight="bold", color="#2E7D32")
        ax.text(n_spatial + len(temporal_defs) / 2, n_enc + 0.3, "TEMPORAL",
                ha="center", fontsize=12, fontweight="bold", color="#C62828")

    ax.set_title("Encoder × Metric Performance (normalized per column, 95% CI)",
                 fontsize=12, fontweight="bold", pad=25)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08b_heatmap{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'm08b_heatmap.png'}")


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
        dim = all_metrics[enc].get("encoder_dim", get_encoder_info(enc)["dim"])
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
    parser.add_argument("--POC", action="store_true", help="POC subset (~10K clips)")
    parser.add_argument("--FULL", action="store_true", help="Process all available encoders")
    parser.add_argument("--encoders", type=str, default=None,
                        help="Comma-separated encoder names to compare "
                             "(e.g., vjepa,vjepa_adapted,vjepa_lambda0_01). "
                             "If omitted, uses all registered encoders.")
    add_subset_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    # Parse custom encoder list
    encoder_list = None
    if args.encoders:
        encoder_list = [e.strip() for e in args.encoders.split(",")]
        # Auto-assign labels/colors for unregistered encoders
        _extra_colors = ["#7B1FA2", "#00695C", "#BF360C", "#1A237E",
                         "#33691E", "#880E4F", "#004D40", "#F57F17"]
        for i, enc in enumerate(encoder_list):
            if enc not in ENCODER_LABELS:
                # Dynamic labels for lambda encoders and their shuffled variants
                import re
                m_shuf = re.match(r'vjepa_lambda(\d+(?:_\d+)*)_shuffled', enc)
                m_lam = re.match(r'vjepa_lambda(\d+(?:_\d+)*)', enc)
                if m_shuf:
                    lam_str = m_shuf.group(1).replace("_", ".", 1)
                    ENCODER_LABELS[enc] = f"Shuffled Adapted\n(\u03bb={lam_str})"
                elif m_lam:
                    lam_str = m_lam.group(1).replace("_", ".", 1)
                    ENCODER_LABELS[enc] = f"V-JEPA Adapted\n(\u03bb={lam_str})"
                else:
                    ENCODER_LABELS[enc] = enc.replace("_", "\n")
                ENCODER_COLORS[enc] = _extra_colors[i % len(_extra_colors)]
        # Override ENCODER_ORDER with custom list so all plot functions include them
        global ENCODER_ORDER
        ENCODER_ORDER = encoder_list

    output_dir = get_output_dir(args.subset, sanity=args.SANITY)
    print(f"Output dir: {output_dir}")
    print(f"Scanning for encoder metrics...")

    all_metrics = load_all_metrics(output_dir, encoder_list=encoder_list)
    if not all_metrics:
        print("FATAL: No encoder metrics found. Run m06 first.")
        sys.exit(1)

    all_temporal = load_all_temporal(output_dir, encoder_list=encoder_list)

    print(f"\nFound {len(all_metrics)} encoder(s): {', '.join(all_metrics.keys())}")
    if all_temporal:
        print(f"Found {len(all_temporal)} temporal result(s): {', '.join(all_temporal.keys())}")

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb_run = init_wandb("m08b", mode, config=vars(args), enabled=not args.no_wandb)

    # Terminal table
    print_summary_table(all_metrics)

    # Plots + table (need >= 2 encoders for comparison)
    if len(all_metrics) >= 2:
        pbar = make_pbar(total=7, desc="m08b_compare", unit="plot")
        create_bar_chart(all_metrics, output_dir)
        pbar.update(1)
        create_radar_plot(all_metrics, output_dir, all_temporal=all_temporal)
        pbar.update(1)
        create_latex_table(all_metrics, output_dir)
        pbar.update(1)
        create_grouped_bar_chart(all_metrics, all_temporal, output_dir)
        pbar.update(1)
        create_tradeoff_scatter(all_metrics, all_temporal, output_dir)
        pbar.update(1)
        create_ablation_chart(all_metrics, all_temporal, output_dir)
        pbar.update(1)
        create_heatmap(all_metrics, all_temporal, output_dir)
        pbar.update(1)
        pbar.close()

        for name in ["m08b_encoder_comparison", "m08b_radar",
                     "m08b_spatial_temporal_bar", "m08b_tradeoff_scatter",
                     "m08b_temporal_ablation", "m08b_heatmap"]:
            png = output_dir / f"{name}.png"
            if png.exists():
                log_image(wb_run, name, str(png))
    else:
        print("Only 1 encoder found — skipping comparison plots (need >= 2).")

    finish_wandb(wb_run)
    print("\n=== COMPARISON COMPLETE ===")


if __name__ == "__main__":
    main()
