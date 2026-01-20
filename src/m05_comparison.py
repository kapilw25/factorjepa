"""
Baseline vs Proposed comparison plots with statistical significance testing.

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
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.plotting import apply_style, save_figure, METRICS, METRIC_LABELS, LEVEL_COLORS

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
BASELINE_DIR = "outputs_baseline"
PROPOSED_DIR = "outputs"  # After running proposed pipeline

BASELINE_LABEL = "Baseline\n(LLaVA-7B + Llama-8B)"
PROPOSED_LABEL = "Proposed\n(Qwen-32B + Llama-70B)"

BASELINE_COLOR = "#e74c3c"  # Red
PROPOSED_COLOR = "#27ae60"  # Green


# ─────────────────────────────────────────────────────────────────
# COMPARISON CLASS
# ─────────────────────────────────────────────────────────────────
class PipelineComparison:
    """Compare baseline vs proposed pipeline evaluation results."""

    def __init__(self, baseline_dir: str = BASELINE_DIR, proposed_dir: str = PROPOSED_DIR):
        self.baseline_dir = Path(baseline_dir)
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

    def compute_statistics(self, baseline: pd.DataFrame, proposed: pd.DataFrame) -> Dict:
        """Compute comparison statistics and significance tests."""
        results = {
            "by_level": {},
            "by_metric": {},
            "overall": {}
        }

        # Overall comparison
        baseline_scores = []
        proposed_scores = []

        for level in [1, 2, 3]:
            level_baseline = []
            level_proposed = []

            for metric in METRICS:
                col = f"L{level}_{metric}"
                if col in baseline.columns and col in proposed.columns:
                    b_vals = baseline[col].dropna().values
                    p_vals = proposed[col].dropna().values

                    if len(b_vals) > 0 and len(p_vals) > 0:
                        level_baseline.extend(b_vals)
                        level_proposed.extend(p_vals)
                        baseline_scores.extend(b_vals)
                        proposed_scores.extend(p_vals)

            if level_baseline and level_proposed:
                b_mean = np.mean(level_baseline)
                p_mean = np.mean(level_proposed)
                improvement = ((p_mean - b_mean) / b_mean) * 100 if b_mean > 0 else 0

                # Statistical test (paired if same length, otherwise independent)
                if len(level_baseline) == len(level_proposed):
                    stat, pval = stats.wilcoxon(level_proposed, level_baseline, alternative='greater')
                    test_name = "Wilcoxon"
                else:
                    stat, pval = stats.mannwhitneyu(level_proposed, level_baseline, alternative='greater')
                    test_name = "Mann-Whitney"

                results["by_level"][f"L{level}"] = {
                    "baseline_mean": b_mean,
                    "baseline_std": np.std(level_baseline),
                    "proposed_mean": p_mean,
                    "proposed_std": np.std(level_proposed),
                    "improvement_pct": improvement,
                    "p_value": pval,
                    "significant": pval < 0.05,
                    "test": test_name
                }

        # By metric
        for metric in METRICS:
            metric_baseline = []
            metric_proposed = []

            for level in [1, 2, 3]:
                col = f"L{level}_{metric}"
                if col in baseline.columns and col in proposed.columns:
                    metric_baseline.extend(baseline[col].dropna().values)
                    metric_proposed.extend(proposed[col].dropna().values)

            if metric_baseline and metric_proposed:
                b_mean = np.mean(metric_baseline)
                p_mean = np.mean(metric_proposed)
                improvement = ((p_mean - b_mean) / b_mean) * 100 if b_mean > 0 else 0

                if len(metric_baseline) == len(metric_proposed):
                    stat, pval = stats.wilcoxon(metric_proposed, metric_baseline, alternative='greater')
                else:
                    stat, pval = stats.mannwhitneyu(metric_proposed, metric_baseline, alternative='greater')

                results["by_metric"][metric] = {
                    "baseline_mean": b_mean,
                    "proposed_mean": p_mean,
                    "improvement_pct": improvement,
                    "p_value": pval,
                    "significant": pval < 0.05
                }

        # Overall
        if baseline_scores and proposed_scores:
            b_mean = np.mean(baseline_scores)
            p_mean = np.mean(proposed_scores)
            improvement = ((p_mean - b_mean) / b_mean) * 100 if b_mean > 0 else 0

            if len(baseline_scores) == len(proposed_scores):
                stat, pval = stats.wilcoxon(proposed_scores, baseline_scores, alternative='greater')
            else:
                stat, pval = stats.mannwhitneyu(proposed_scores, baseline_scores, alternative='greater')

            results["overall"] = {
                "baseline_mean": b_mean,
                "baseline_std": np.std(baseline_scores),
                "proposed_mean": p_mean,
                "proposed_std": np.std(proposed_scores),
                "improvement_pct": improvement,
                "p_value": pval,
                "significant": pval < 0.05
            }

        return results

    def plot_comparison_bar(self, baseline: pd.DataFrame, proposed: pd.DataFrame, stats: Dict) -> str:
        """Side-by-side bar chart comparing baseline vs proposed."""
        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(METRICS))
        width = 0.35

        # Calculate means for each metric across all levels
        baseline_means = []
        baseline_stds = []
        proposed_means = []
        proposed_stds = []

        for metric in METRICS:
            b_vals = []
            p_vals = []
            for level in [1, 2, 3]:
                col = f"L{level}_{metric}"
                if col in baseline.columns:
                    b_vals.extend(baseline[col].dropna().values)
                if col in proposed.columns:
                    p_vals.extend(proposed[col].dropna().values)

            baseline_means.append(np.mean(b_vals) if b_vals else 0)
            baseline_stds.append(np.std(b_vals) if b_vals else 0)
            proposed_means.append(np.mean(p_vals) if p_vals else 0)
            proposed_stds.append(np.std(p_vals) if p_vals else 0)

        # Plot bars
        bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                       label=BASELINE_LABEL, color=BASELINE_COLOR, alpha=0.8,
                       capsize=4, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, proposed_means, width, yerr=proposed_stds,
                       label=PROPOSED_LABEL, color=PROPOSED_COLOR, alpha=0.8,
                       capsize=4, edgecolor='black', linewidth=0.5)

        # Add significance stars
        for i, metric in enumerate(METRICS):
            if metric in stats["by_metric"]:
                if stats["by_metric"][metric]["significant"]:
                    max_height = max(baseline_means[i] + baseline_stds[i],
                                    proposed_means[i] + proposed_stds[i])
                    ax.text(i, max_height + 0.15, '*', ha='center', fontsize=16, fontweight='bold')

        # Add value labels
        for bars, means in [(bars1, baseline_means), (bars2, proposed_means)]:
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Evaluation Metric', fontsize=12)
        ax.set_ylabel('Score (1-5)', fontsize=12)
        ax.set_title('Baseline vs Proposed Pipeline: Evaluation Scores by Metric\n(* = statistically significant, p < 0.05)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in METRICS], rotation=15, ha='right')
        ax.set_ylim(0, 5.5)
        ax.legend(loc='upper right', fontsize=10)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        filepath = self.plots_dir / 'comparison_by_metric'
        return save_figure(fig, filepath)

    def plot_comparison_by_level(self, baseline: pd.DataFrame, proposed: pd.DataFrame, stats: Dict) -> str:
        """Bar chart comparing scores by difficulty level."""
        fig, ax = plt.subplots(figsize=(10, 6))

        levels = ['L1', 'L2', 'L3']
        x = np.arange(len(levels))
        width = 0.35

        baseline_means = []
        baseline_stds = []
        proposed_means = []
        proposed_stds = []

        for level in levels:
            if level in stats["by_level"]:
                baseline_means.append(stats["by_level"][level]["baseline_mean"])
                baseline_stds.append(stats["by_level"][level]["baseline_std"])
                proposed_means.append(stats["by_level"][level]["proposed_mean"])
                proposed_stds.append(stats["by_level"][level]["proposed_std"])
            else:
                baseline_means.append(0)
                baseline_stds.append(0)
                proposed_means.append(0)
                proposed_stds.append(0)

        bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                       label=BASELINE_LABEL, color=BASELINE_COLOR, alpha=0.8,
                       capsize=4, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, proposed_means, width, yerr=proposed_stds,
                       label=PROPOSED_LABEL, color=PROPOSED_COLOR, alpha=0.8,
                       capsize=4, edgecolor='black', linewidth=0.5)

        # Add significance stars and improvement percentages
        for i, level in enumerate(levels):
            if level in stats["by_level"]:
                info = stats["by_level"][level]
                max_height = max(baseline_means[i] + baseline_stds[i],
                                proposed_means[i] + proposed_stds[i])

                # Star for significance
                if info["significant"]:
                    ax.text(i, max_height + 0.1, '*', ha='center', fontsize=16, fontweight='bold')

                # Improvement percentage
                ax.text(i, max_height + 0.3, f'+{info["improvement_pct"]:.1f}%',
                       ha='center', fontsize=9, color='green' if info["improvement_pct"] > 0 else 'red')

        ax.set_xlabel('Difficulty Level', fontsize=12)
        ax.set_ylabel('Average Score (1-5)', fontsize=12)
        ax.set_title('Baseline vs Proposed: Performance by Difficulty Level\n(* = p < 0.05)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Level 1\n(Easy)', 'Level 2\n(Medium)', 'Level 3\n(Hard)'])
        ax.set_ylim(0, 5.5)
        ax.legend(loc='upper right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        filepath = self.plots_dir / 'comparison_by_level'
        return save_figure(fig, filepath)

    def plot_improvement_heatmap(self, baseline: pd.DataFrame, proposed: pd.DataFrame) -> str:
        """Heatmap showing improvement percentage by scene and metric."""
        # Find common scenes
        common_scenes = set(baseline['scene_id']) & set(proposed['scene_id'])
        if not common_scenes:
            print("[m05] WARNING: No common scenes for heatmap")
            return ""

        improvement_data = {}
        for metric in METRICS:
            improvements = []
            for scene_id in sorted(common_scenes):
                b_row = baseline[baseline['scene_id'] == scene_id]
                p_row = proposed[proposed['scene_id'] == scene_id]

                b_vals = []
                p_vals = []
                for level in [1, 2, 3]:
                    col = f"L{level}_{metric}"
                    if col in baseline.columns and col in proposed.columns:
                        b_val = b_row[col].values[0] if len(b_row) > 0 else 0
                        p_val = p_row[col].values[0] if len(p_row) > 0 else 0
                        b_vals.append(b_val)
                        p_vals.append(p_val)

                b_mean = np.mean(b_vals) if b_vals else 0
                p_mean = np.mean(p_vals) if p_vals else 0
                improvement = ((p_mean - b_mean) / b_mean) * 100 if b_mean > 0 else 0
                improvements.append(improvement)

            improvement_data[metric.replace('_', ' ').title()] = improvements

        improvement_df = pd.DataFrame(improvement_data, index=sorted(common_scenes))

        fig, ax = plt.subplots(figsize=(12, max(4, len(common_scenes) * 0.8)))

        # Custom colormap: red for negative, white for zero, green for positive
        cmap = plt.cm.RdYlGn
        im = ax.imshow(improvement_df.values, cmap=cmap, aspect='auto',
                       vmin=-50, vmax=50)

        ax.set_xticks(np.arange(len(METRICS)))
        ax.set_yticks(np.arange(len(common_scenes)))
        ax.set_xticklabels([m.replace('_', ' ').title() for m in METRICS], rotation=45, ha='right')
        ax.set_yticklabels(sorted(common_scenes))

        # Add text annotations
        for i in range(len(common_scenes)):
            for j in range(len(METRICS)):
                val = improvement_df.values[i, j]
                color = 'white' if abs(val) > 25 else 'black'
                ax.text(j, i, f'{val:+.1f}%', ha='center', va='center', color=color, fontsize=9)

        ax.set_title('Improvement Over Baseline by Scene and Metric (%)\n(Green = Improvement, Red = Regression)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Scene')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Improvement (%)')

        filepath = self.plots_dir / 'improvement_heatmap'
        return save_figure(fig, filepath)

    def plot_summary_dashboard(self, baseline: pd.DataFrame, proposed: pd.DataFrame, stats: Dict) -> str:
        """Summary dashboard with key comparison statistics."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Overall comparison (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        overall = stats["overall"]
        bars = ax1.bar(['Baseline', 'Proposed'],
                      [overall["baseline_mean"], overall["proposed_mean"]],
                      yerr=[overall["baseline_std"], overall["proposed_std"]],
                      color=[BASELINE_COLOR, PROPOSED_COLOR], capsize=5,
                      edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Overall Score (1-5)')
        ax1.set_title('Overall Performance')
        ax1.set_ylim(0, 5)

        for bar, val in zip(bars, [overall["baseline_mean"], overall["proposed_mean"]]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', fontsize=11)

        # 2. Improvement by metric (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        improvements = [stats["by_metric"][m]["improvement_pct"] for m in METRICS if m in stats["by_metric"]]
        metric_names = [m.replace('_', '\n').title() for m in METRICS if m in stats["by_metric"]]
        colors = [PROPOSED_COLOR if imp > 0 else BASELINE_COLOR for imp in improvements]

        bars = ax2.barh(metric_names, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.set_xlabel('Improvement (%)')
        ax2.set_title('Improvement by Metric')

        for bar, val in zip(bars, improvements):
            ax2.text(val + (2 if val >= 0 else -2), bar.get_y() + bar.get_height()/2,
                    f'{val:+.1f}%', ha='left' if val >= 0 else 'right', va='center', fontsize=9)

        # 3. Summary statistics (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')

        sig_metrics = sum(1 for m in stats["by_metric"].values() if m["significant"])
        sig_levels = sum(1 for l in stats["by_level"].values() if l["significant"])

        summary_text = f"""
COMPARISON SUMMARY
══════════════════════════════════════

Overall Improvement: {overall["improvement_pct"]:+.1f}%
Statistical Significance: {'YES' if overall["significant"] else 'NO'} (p={overall["p_value"]:.4f})

Baseline:  {overall["baseline_mean"]:.2f} ± {overall["baseline_std"]:.2f}
Proposed:  {overall["proposed_mean"]:.2f} ± {overall["proposed_std"]:.2f}

Significant Improvements:
  • Metrics: {sig_metrics}/{len(METRICS)}
  • Levels: {sig_levels}/3

Pipeline Comparison:
  Baseline: LLaVA-1.5-7B + Llama-3-8B
  Proposed: Qwen-32B + Llama-3.1-70B
        """
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        # 4. By level comparison (bottom-left + center)
        ax4 = fig.add_subplot(gs[1, :2])
        levels = ['L1', 'L2', 'L3']
        x = np.arange(len(levels))
        width = 0.35

        baseline_means = [stats["by_level"][l]["baseline_mean"] for l in levels if l in stats["by_level"]]
        proposed_means = [stats["by_level"][l]["proposed_mean"] for l in levels if l in stats["by_level"]]
        baseline_stds = [stats["by_level"][l]["baseline_std"] for l in levels if l in stats["by_level"]]
        proposed_stds = [stats["by_level"][l]["proposed_std"] for l in levels if l in stats["by_level"]]

        ax4.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
               label=BASELINE_LABEL, color=BASELINE_COLOR, alpha=0.8, capsize=4)
        ax4.bar(x + width/2, proposed_means, width, yerr=proposed_stds,
               label=PROPOSED_LABEL, color=PROPOSED_COLOR, alpha=0.8, capsize=4)

        ax4.set_xlabel('Difficulty Level')
        ax4.set_ylabel('Score (1-5)')
        ax4.set_title('Performance by Difficulty Level')
        ax4.set_xticks(x)
        ax4.set_xticklabels(['Level 1 (Easy)', 'Level 2 (Medium)', 'Level 3 (Hard)'])
        ax4.set_ylim(0, 5)
        ax4.legend()
        ax4.yaxis.grid(True, linestyle='--', alpha=0.7)

        # 5. P-value table (bottom-right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')

        table_data = []
        for metric in METRICS:
            if metric in stats["by_metric"]:
                m = stats["by_metric"][metric]
                sig = "✓" if m["significant"] else "✗"
                table_data.append([metric.replace('_', ' ').title(),
                                  f'{m["improvement_pct"]:+.1f}%',
                                  f'{m["p_value"]:.3f}',
                                  sig])

        table = ax5.table(cellText=table_data,
                         colLabels=['Metric', 'Improvement', 'p-value', 'Sig.'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax5.set_title('Statistical Significance Test Results', fontsize=11, pad=20)

        fig.suptitle('Baseline vs Proposed Pipeline Comparison Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)

        filepath = self.plots_dir / 'comparison_dashboard'
        return save_figure(fig, filepath)

    def generate_all_comparisons(self) -> List[str]:
        """Generate all comparison plots."""
        # Load data
        print(f"[m05] Loading baseline from: {self.baseline_dir}")
        baseline = self.load_metrics(self.baseline_dir)
        print(f"[m05] Baseline: {len(baseline)} scenes")

        print(f"[m05] Loading proposed from: {self.proposed_dir}")
        proposed = self.load_metrics(self.proposed_dir)
        print(f"[m05] Proposed: {len(proposed)} scenes")

        # Compute statistics
        print("[m05] Computing statistics...")
        stats = self.compute_statistics(baseline, proposed)

        # Print summary
        print("\n" + "=" * 50)
        print("COMPARISON RESULTS")
        print("=" * 50)
        overall = stats["overall"]
        print(f"Overall: Baseline={overall['baseline_mean']:.2f} → Proposed={overall['proposed_mean']:.2f}")
        print(f"Improvement: {overall['improvement_pct']:+.1f}%")
        print(f"Significant: {'YES' if overall['significant'] else 'NO'} (p={overall['p_value']:.4f})")
        print("=" * 50 + "\n")

        # Generate plots
        saved_files = []
        print("[m05] Generating comparison plots...")

        saved_files.append(self.plot_comparison_bar(baseline, proposed, stats))
        saved_files.append(self.plot_comparison_by_level(baseline, proposed, stats))
        saved_files.append(self.plot_improvement_heatmap(baseline, proposed))
        saved_files.append(self.plot_summary_dashboard(baseline, proposed, stats))

        print(f"[m05] Saved {len(saved_files)} plots to {self.plots_dir}/")
        return saved_files


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs proposed pipeline evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate comparison plots (requires both outputs_baseline/ and outputs/)
  python -u src/m05_comparison.py --compare

  # Custom directories
  python -u src/m05_comparison.py --compare --baseline outputs_v1 --proposed outputs_v2
        """
    )
    parser.add_argument("--compare", action="store_true", help="Generate comparison plots")
    parser.add_argument("--baseline", type=str, default=BASELINE_DIR,
                       help="Baseline output directory")
    parser.add_argument("--proposed", type=str, default=PROPOSED_DIR,
                       help="Proposed output directory")
    args = parser.parse_args()

    if args.compare:
        comparator = PipelineComparison(args.baseline, args.proposed)
        try:
            saved_files = comparator.generate_all_comparisons()
            print("\n[m05] Generated plots:")
            for f in saved_files:
                print(f"  • {f}")
        except FileNotFoundError as e:
            print(f"[m05] ERROR: {e}")
            print("[m05] Make sure both baseline and proposed metrics.csv exist")
            print(f"[m05] Expected: {args.baseline}/metrics.csv and {args.proposed}/metrics.csv")
    else:
        print("[m05] No action specified. Use --compare to generate comparison plots.")
        print("[m05] Run with --help for more options.")


if __name__ == "__main__":
    main()
