"""
3-way comparison plots: Baseline vs QwenOnly vs Proposed with statistical significance testing.

Commands:
    python -u src/m05_comparison.py --help
    python -u src/m05_comparison.py --compare 2>&1 | tee logs/m05_compare.log
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.plotting import apply_style, save_figure, METRICS, METRIC_LABELS, LEVEL_COLORS

# ─────────────────────────────────────────────────────────────────
# CONFIG - 3 Pipelines
# ─────────────────────────────────────────────────────────────────
BASELINE_DIR = "outputs_baseline"
QWENONLY_DIR = "outputs_qwenonly"
PROPOSED_DIR = "outputs"

BASELINE_LABEL = "Baseline\n(LLaVA-7B + Llama-8B)"
QWENONLY_LABEL = "QwenOnly\n(Qwen-8B)"
PROPOSED_LABEL = "Proposed\n(Qwen-32B + Llama-70B)"

BASELINE_COLOR = "#e74c3c"  # Red
QWENONLY_COLOR = "#f39c12"  # Orange
PROPOSED_COLOR = "#27ae60"  # Green


# ─────────────────────────────────────────────────────────────────
# 3-WAY COMPARISON CLASS
# ─────────────────────────────────────────────────────────────────
class ThreeWayComparison:
    """Compare 3 pipelines: Baseline vs QwenOnly vs Proposed."""

    def __init__(self, baseline_dir: str = BASELINE_DIR,
                 qwenonly_dir: str = QWENONLY_DIR,
                 proposed_dir: str = PROPOSED_DIR):
        self.baseline_dir = Path(baseline_dir)
        self.qwenonly_dir = Path(qwenonly_dir)
        self.proposed_dir = Path(proposed_dir)
        self.plots_dir = Path("outputs_comparison")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        apply_style()

    def load_metrics(self, output_dir: Path) -> pd.DataFrame:
        """Load metrics.csv from output directory."""
        csv_path = output_dir / "metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No metrics.csv found at {csv_path}")
        return pd.read_csv(csv_path)

    def get_scene_avg_scores(self, df: pd.DataFrame, scene_id: str) -> List[float]:
        """Get average scores across levels for each metric for a scene."""
        row = df[df['scene_id'] == scene_id]
        if len(row) == 0:
            return [0] * len(METRICS)

        scores = []
        for metric in METRICS:
            vals = []
            for level in [1, 2, 3]:
                col = f"L{level}_{metric}"
                if col in df.columns:
                    val = row[col].values[0]
                    if not pd.isna(val):
                        vals.append(val)
            scores.append(np.mean(vals) if vals else 0)
        return scores

    def get_overall_mean(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Get overall mean and std across all metrics and levels."""
        all_vals = []
        for level in [1, 2, 3]:
            for metric in METRICS:
                col = f"L{level}_{metric}"
                if col in df.columns:
                    all_vals.extend(df[col].dropna().values)
        return (np.mean(all_vals), np.std(all_vals)) if all_vals else (0, 0)

    def plot_comparison_heatmap(self, baseline: pd.DataFrame, qwenonly: pd.DataFrame,
                                 proposed: pd.DataFrame) -> str:
        """Heatmap with 3 rows per scene: Baseline / QwenOnly / Proposed."""
        # Find common scenes across all 3
        common_scenes = set(baseline['scene_id']) & set(qwenonly['scene_id']) & set(proposed['scene_id'])
        if not common_scenes:
            print("[m05] WARNING: No common scenes for 3-way heatmap")
            return ""

        sorted_scenes = sorted(common_scenes)
        n_scenes = len(sorted_scenes)

        # Build data: 3 rows per scene
        row_labels = []
        heatmap_values = []
        row_colors = []  # Track which pipeline each row belongs to

        for scene_id in sorted_scenes:
            # Baseline
            row_labels.append(f"{scene_id}\n(Baseline)")
            heatmap_values.append(self.get_scene_avg_scores(baseline, scene_id))
            row_colors.append(BASELINE_COLOR)

            # QwenOnly
            row_labels.append(f"{scene_id}\n(QwenOnly)")
            heatmap_values.append(self.get_scene_avg_scores(qwenonly, scene_id))
            row_colors.append(QWENONLY_COLOR)

            # Proposed
            row_labels.append(f"{scene_id}\n(Proposed)")
            heatmap_values.append(self.get_scene_avg_scores(proposed, scene_id))
            row_colors.append(PROPOSED_COLOR)

        heatmap_array = np.array(heatmap_values)
        n_rows = len(row_labels)

        # Figure size: scale with number of scenes
        fig, ax = plt.subplots(figsize=(12, max(8, n_rows * 0.45)))

        # Heatmap with score colormap (1-5 scale)
        cmap = plt.cm.RdYlGn
        im = ax.imshow(heatmap_array, cmap=cmap, aspect='auto', vmin=1, vmax=5)

        ax.set_xticks(np.arange(len(METRICS)))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in METRICS],
                          rotation=45, ha='right', fontweight='bold', color='black')
        ax.set_yticklabels(row_labels, fontweight='bold', color='black', fontsize=9)

        # Add horizontal lines to separate scenes (every 3 rows)
        for i in range(1, n_scenes):
            ax.axhline(y=i * 3 - 0.5, color='black', linewidth=2)

        # Add text annotations with raw scores
        for i in range(n_rows):
            for j in range(len(METRICS)):
                val = heatmap_array[i, j]
                color = 'white' if val < 2.5 or val > 4.0 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=9, fontweight='bold')

        ax.set_title('3-Way Comparison: Raw Scores by Scene and Metric\n(Baseline / QwenOnly / Proposed)',
                    fontsize=14, fontweight='bold', color='black')
        ax.set_xlabel('Metric', fontweight='bold', color='black')
        ax.set_ylabel('Scene', fontweight='bold', color='black')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Score (1-5)', fontweight='bold')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=BASELINE_COLOR, label='Baseline (LLaVA-7B + Llama-8B)'),
            Patch(facecolor=QWENONLY_COLOR, label='QwenOnly (Qwen-8B)'),
            Patch(facecolor=PROPOSED_COLOR, label='Proposed (Qwen-32B + Llama-70B)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))

        filepath = self.plots_dir / 'comparison_heatmap'
        return save_figure(fig, filepath)

    def plot_comparison_bar(self, baseline: pd.DataFrame, qwenonly: pd.DataFrame,
                            proposed: pd.DataFrame) -> str:
        """3-way bar chart comparing scores by metric."""
        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(METRICS))
        width = 0.25

        # Calculate means for each metric across all levels
        def get_metric_stats(df):
            means, stds = [], []
            for metric in METRICS:
                vals = []
                for level in [1, 2, 3]:
                    col = f"L{level}_{metric}"
                    if col in df.columns:
                        vals.extend(df[col].dropna().values)
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            return means, stds

        b_means, b_stds = get_metric_stats(baseline)
        q_means, q_stds = get_metric_stats(qwenonly)
        p_means, p_stds = get_metric_stats(proposed)

        # Plot bars
        bars1 = ax.bar(x - width, b_means, width, yerr=b_stds,
                       label=BASELINE_LABEL, color=BASELINE_COLOR, alpha=0.8,
                       capsize=3, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, q_means, width, yerr=q_stds,
                       label=QWENONLY_LABEL, color=QWENONLY_COLOR, alpha=0.8,
                       capsize=3, edgecolor='black', linewidth=0.5)
        bars3 = ax.bar(x + width, p_means, width, yerr=p_stds,
                       label=PROPOSED_LABEL, color=PROPOSED_COLOR, alpha=0.8,
                       capsize=3, edgecolor='black', linewidth=0.5)

        # Add value labels
        for bars, means in [(bars1, b_means), (bars2, q_means), (bars3, p_means)]:
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xlabel('Evaluation Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (1-5)', fontsize=12, fontweight='bold')
        ax.set_title('3-Way Pipeline Comparison: Evaluation Scores by Metric',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in METRICS], rotation=15, ha='right')
        ax.set_ylim(0, 5.5)
        ax.legend(loc='upper right', fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        filepath = self.plots_dir / 'comparison_by_metric'
        return save_figure(fig, filepath)

    def plot_comparison_by_level(self, baseline: pd.DataFrame, qwenonly: pd.DataFrame,
                                  proposed: pd.DataFrame) -> str:
        """3-way bar chart comparing scores by difficulty level."""
        fig, ax = plt.subplots(figsize=(10, 6))

        levels = ['L1', 'L2', 'L3']
        x = np.arange(len(levels))
        width = 0.25

        def get_level_stats(df):
            means, stds = [], []
            for level in [1, 2, 3]:
                vals = []
                for metric in METRICS:
                    col = f"L{level}_{metric}"
                    if col in df.columns:
                        vals.extend(df[col].dropna().values)
                means.append(np.mean(vals) if vals else 0)
                stds.append(np.std(vals) if vals else 0)
            return means, stds

        b_means, b_stds = get_level_stats(baseline)
        q_means, q_stds = get_level_stats(qwenonly)
        p_means, p_stds = get_level_stats(proposed)

        ax.bar(x - width, b_means, width, yerr=b_stds,
               label=BASELINE_LABEL, color=BASELINE_COLOR, alpha=0.8, capsize=4)
        ax.bar(x, q_means, width, yerr=q_stds,
               label=QWENONLY_LABEL, color=QWENONLY_COLOR, alpha=0.8, capsize=4)
        ax.bar(x + width, p_means, width, yerr=p_stds,
               label=PROPOSED_LABEL, color=PROPOSED_COLOR, alpha=0.8, capsize=4)

        ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Score (1-5)', fontsize=12, fontweight='bold')
        ax.set_title('3-Way Comparison: Performance by Difficulty Level',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Level 1\n(Easy)', 'Level 2\n(Medium)', 'Level 3\n(Hard)'])
        ax.set_ylim(0, 5.5)
        ax.legend(loc='upper right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        filepath = self.plots_dir / 'comparison_by_level'
        return save_figure(fig, filepath)

    def plot_summary_dashboard(self, baseline: pd.DataFrame, qwenonly: pd.DataFrame,
                               proposed: pd.DataFrame) -> str:
        """Summary dashboard with 3-way comparison statistics."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        b_mean, b_std = self.get_overall_mean(baseline)
        q_mean, q_std = self.get_overall_mean(qwenonly)
        p_mean, p_std = self.get_overall_mean(proposed)

        # 1. Overall comparison (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(['Baseline', 'QwenOnly', 'Proposed'],
                      [b_mean, q_mean, p_mean],
                      yerr=[b_std, q_std, p_std],
                      color=[BASELINE_COLOR, QWENONLY_COLOR, PROPOSED_COLOR],
                      capsize=5, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Overall Score (1-5)', fontweight='bold')
        ax1.set_title('Overall Performance', fontweight='bold')
        ax1.set_ylim(0, 5)

        for bar, val in zip(bars, [b_mean, q_mean, p_mean]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')

        # 2. Improvement over baseline (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        q_imp = ((q_mean - b_mean) / b_mean) * 100 if b_mean > 0 else 0
        p_imp = ((p_mean - b_mean) / b_mean) * 100 if b_mean > 0 else 0

        bars = ax2.bar(['QwenOnly', 'Proposed'], [q_imp, p_imp],
                      color=[QWENONLY_COLOR, PROPOSED_COLOR],
                      edgecolor='black', linewidth=0.5)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.set_ylabel('Improvement over Baseline (%)', fontweight='bold')
        ax2.set_title('Improvement vs Baseline', fontweight='bold')

        for bar, val in zip(bars, [q_imp, p_imp]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:+.1f}%', ha='center', fontsize=11, fontweight='bold')

        # 3. Summary statistics (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')

        summary_text = f"""
3-WAY COMPARISON SUMMARY
════════════════════════════════════

Overall Scores:
  Baseline:  {b_mean:.2f} ± {b_std:.2f}
  QwenOnly:  {q_mean:.2f} ± {q_std:.2f}
  Proposed:  {p_mean:.2f} ± {p_std:.2f}

Improvement over Baseline:
  QwenOnly: {q_imp:+.1f}%
  Proposed: {p_imp:+.1f}%

Pipelines:
  Baseline: LLaVA-7B + Llama-8B
  QwenOnly: Qwen-8B (VLM only)
  Proposed: Qwen-32B + Llama-70B
        """
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        # 4. By level comparison (bottom-left + center)
        ax4 = fig.add_subplot(gs[1, :2])

        def get_level_means(df):
            means = []
            for level in [1, 2, 3]:
                vals = []
                for metric in METRICS:
                    col = f"L{level}_{metric}"
                    if col in df.columns:
                        vals.extend(df[col].dropna().values)
                means.append(np.mean(vals) if vals else 0)
            return means

        levels = ['L1', 'L2', 'L3']
        x = np.arange(len(levels))
        width = 0.25

        b_means = get_level_means(baseline)
        q_means = get_level_means(qwenonly)
        p_means = get_level_means(proposed)

        ax4.bar(x - width, b_means, width, label=BASELINE_LABEL,
                color=BASELINE_COLOR, alpha=0.8)
        ax4.bar(x, q_means, width, label=QWENONLY_LABEL,
                color=QWENONLY_COLOR, alpha=0.8)
        ax4.bar(x + width, p_means, width, label=PROPOSED_LABEL,
                color=PROPOSED_COLOR, alpha=0.8)

        ax4.set_xlabel('Difficulty Level', fontweight='bold')
        ax4.set_ylabel('Score (1-5)', fontweight='bold')
        ax4.set_title('Performance by Difficulty Level', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Level 1 (Easy)', 'Level 2 (Medium)', 'Level 3 (Hard)'])
        ax4.set_ylim(0, 5)
        ax4.legend()
        ax4.yaxis.grid(True, linestyle='--', alpha=0.7)

        # 5. Per-metric comparison table (bottom-right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')

        def get_metric_mean(df, metric):
            vals = []
            for level in [1, 2, 3]:
                col = f"L{level}_{metric}"
                if col in df.columns:
                    vals.extend(df[col].dropna().values)
            return np.mean(vals) if vals else 0

        table_data = []
        for metric in METRICS:
            b_m = get_metric_mean(baseline, metric)
            q_m = get_metric_mean(qwenonly, metric)
            p_m = get_metric_mean(proposed, metric)
            table_data.append([
                metric.replace('_', ' ').title()[:15],
                f'{b_m:.2f}',
                f'{q_m:.2f}',
                f'{p_m:.2f}'
            ])

        table = ax5.table(cellText=table_data,
                         colLabels=['Metric', 'Base', 'Qwen', 'Prop'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax5.set_title('Scores by Metric', fontsize=11, fontweight='bold', pad=20)

        fig.suptitle('3-Way Pipeline Comparison Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        filepath = self.plots_dir / 'comparison_dashboard'
        return save_figure(fig, filepath)

    def generate_all_comparisons(self) -> List[str]:
        """Generate all 3-way comparison plots."""
        # Load data
        print(f"[m05] Loading baseline from: {self.baseline_dir}")
        baseline = self.load_metrics(self.baseline_dir)
        print(f"[m05] Baseline: {len(baseline)} scenes")

        print(f"[m05] Loading qwenonly from: {self.qwenonly_dir}")
        qwenonly = self.load_metrics(self.qwenonly_dir)
        print(f"[m05] QwenOnly: {len(qwenonly)} scenes")

        print(f"[m05] Loading proposed from: {self.proposed_dir}")
        proposed = self.load_metrics(self.proposed_dir)
        print(f"[m05] Proposed: {len(proposed)} scenes")

        # Get overall stats
        b_mean, _ = self.get_overall_mean(baseline)
        q_mean, _ = self.get_overall_mean(qwenonly)
        p_mean, _ = self.get_overall_mean(proposed)

        # Print summary
        print("\n" + "=" * 50)
        print("3-WAY COMPARISON RESULTS")
        print("=" * 50)
        print(f"Baseline:  {b_mean:.2f}")
        print(f"QwenOnly:  {q_mean:.2f} ({((q_mean-b_mean)/b_mean)*100:+.1f}% vs baseline)")
        print(f"Proposed:  {p_mean:.2f} ({((p_mean-b_mean)/b_mean)*100:+.1f}% vs baseline)")
        print("=" * 50 + "\n")

        # Generate plots
        saved_files = []
        print("[m05] Generating 3-way comparison plots...")

        saved_files.append(self.plot_comparison_bar(baseline, qwenonly, proposed))
        saved_files.append(self.plot_comparison_by_level(baseline, qwenonly, proposed))
        saved_files.append(self.plot_comparison_heatmap(baseline, qwenonly, proposed))
        saved_files.append(self.plot_summary_dashboard(baseline, qwenonly, proposed))

        print(f"[m05] Saved {len(saved_files)} plots to {self.plots_dir}/")
        return saved_files


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="3-way comparison: Baseline vs QwenOnly vs Proposed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 3-way comparison plots
  python -u src/m05_comparison.py --compare

  # Custom directories
  python -u src/m05_comparison.py --compare --baseline outputs_v1 --qwenonly outputs_v2 --proposed outputs_v3
        """
    )
    parser.add_argument("--compare", action="store_true", help="Generate comparison plots")
    parser.add_argument("--baseline", type=str, default=BASELINE_DIR,
                       help="Baseline output directory")
    parser.add_argument("--qwenonly", type=str, default=QWENONLY_DIR,
                       help="QwenOnly output directory")
    parser.add_argument("--proposed", type=str, default=PROPOSED_DIR,
                       help="Proposed output directory")
    args = parser.parse_args()

    if args.compare:
        comparator = ThreeWayComparison(args.baseline, args.qwenonly, args.proposed)
        try:
            saved_files = comparator.generate_all_comparisons()
            print("\n[m05] Generated plots:")
            for f in saved_files:
                print(f"  - {f}")
        except FileNotFoundError as e:
            print(f"[m05] ERROR: {e}")
            print("[m05] Make sure all 3 metrics.csv exist:")
            print(f"  - {args.baseline}/metrics.csv")
            print(f"  - {args.qwenonly}/metrics.csv")
            print(f"  - {args.proposed}/metrics.csv")
    else:
        print("[m05] No action specified. Use --compare to generate comparison plots.")
        print("[m05] Run with --help for more options.")


if __name__ == "__main__":
    main()
