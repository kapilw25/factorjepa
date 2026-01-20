"""
Research-grade plotting utilities for evaluation visualization.

Usage:
    from utils.plotting import EvaluationVisualizer
    viz = EvaluationVisualizer("outputs")
    viz.generate_all_plots()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────────
# STYLE CONFIG (Publication-Quality)
# ─────────────────────────────────────────────────────────────────
STYLE_CONFIG = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 11,
    'font.weight': 'bold',
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.labelcolor': 'black',
    'axes.titlecolor': 'black',
    'xtick.labelsize': 10,
    'xtick.color': 'black',
    'ytick.labelsize': 10,
    'ytick.color': 'black',
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.color': 'black',
}

METRICS = ['object_accuracy', 'spatial_coherence', 'task_clarity',
           'difficulty_alignment', 'executability']

METRIC_LABELS = ['Object\nAccuracy', 'Spatial\nCoherence', 'Task\nClarity',
                 'Difficulty\nAlignment', 'Executability']

LEVEL_COLORS = {'L1': '#2ecc71', 'L2': '#3498db', 'L3': '#e74c3c'}


def apply_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette("husl")


# ─────────────────────────────────────────────────────────────────
# EVALUATION VISUALIZER CLASS
# ─────────────────────────────────────────────────────────────────
class EvaluationVisualizer:
    """Generate Tier-1 research-grade plots for evaluation results."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "m03_plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        apply_style()

    def load_metrics(self) -> pd.DataFrame:
        """Load metrics.csv into DataFrame."""
        csv_path = self.output_dir / "metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No metrics.csv found at {csv_path}")
        return pd.read_csv(csv_path)

    def generate_all_plots(self) -> List[str]:
        """Generate all research-grade plots. Returns list of saved file paths."""
        df = self.load_metrics()
        saved_files = []

        print(f"[plot] Generating plots from {len(df)} scenes...")

        # 1. Radar Chart - Metric comparison across levels
        saved_files.append(self.plot_radar_chart(df))

        # 2. Grouped Bar Chart - Average scores per metric
        saved_files.append(self.plot_grouped_bar(df))

        # 3. Heatmap - Scene × Metric scores
        saved_files.append(self.plot_heatmap(df))

        # 4. Box Plot - Score distributions
        saved_files.append(self.plot_box_distributions(df))

        # 5. Summary Dashboard - Combined view
        saved_files.append(self.plot_summary_dashboard(df))

        print(f"[plot] Saved {len(saved_files)} plots to {self.plots_dir}/")
        return saved_files

    def plot_radar_chart(self, df: pd.DataFrame) -> str:
        """Radar chart comparing metrics across difficulty levels."""
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # Calculate mean scores per level
        level_data = {}
        for level in [1, 2, 3]:
            cols = [f'L{level}_{m}' for m in METRICS]
            existing_cols = [c for c in cols if c in df.columns]
            if existing_cols:
                level_data[f'L{level}'] = df[existing_cols].mean().values

        # Radar setup
        n_metrics = len(METRICS)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        for level, values in level_data.items():
            values = list(values) + [values[0]]  # Close polygon
            ax.plot(angles, values, 'o-', linewidth=2, label=level,
                   color=LEVEL_COLORS[level])
            ax.fill(angles, values, alpha=0.15, color=LEVEL_COLORS[level])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(METRIC_LABELS, fontweight='bold', color='black', fontsize=10)
        ax.set_ylim(0, 5)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=10, fontweight='bold', color='black')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), prop={'weight': 'bold'})
        ax.set_title('Evaluation Metrics by Difficulty Level', fontsize=14, pad=20,
                    fontweight='bold', color='black')
        ax.tick_params(colors='black')

        filepath = self.plots_dir / 'radar_metrics_by_level.pdf'
        fig.savefig(filepath, format='pdf')
        fig.savefig(filepath.with_suffix('.png'), format='png')
        plt.close(fig)
        return str(filepath)

    def plot_grouped_bar(self, df: pd.DataFrame) -> str:
        """Grouped bar chart of average scores per metric and level."""
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(METRICS))
        width = 0.25

        for i, level in enumerate([1, 2, 3]):
            cols = [f'L{level}_{m}' for m in METRICS]
            existing_cols = [c for c in cols if c in df.columns]
            if existing_cols:
                means = df[existing_cols].mean().values
                stds = df[existing_cols].std().values
                ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
                       label=f'Level {level}', color=LEVEL_COLORS[f'L{level}'],
                       alpha=0.85, edgecolor='black', linewidth=1)

        ax.set_xlabel('Evaluation Metric', fontweight='bold', color='black')
        ax.set_ylabel('Score (1-5)', fontweight='bold', color='black')
        ax.set_title('Average Evaluation Scores by Metric and Difficulty Level',
                    fontweight='bold', color='black')
        ax.set_xticks(x + width)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in METRICS],
                          rotation=15, ha='right', fontweight='bold', color='black')
        ax.set_ylim(0, 5.5)
        ax.legend(title='Difficulty', title_fontsize=10, prop={'weight': 'bold'})
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        # Make tick labels bold and black
        ax.tick_params(colors='black')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_color('black')

        # Add value labels on bars
        for container in ax.containers:
            if hasattr(container, 'datavalues'):
                ax.bar_label(container, fmt='%.2f', fontsize=9, padding=2,
                            fontweight='bold', color='black')

        filepath = self.plots_dir / 'bar_avg_scores.pdf'
        fig.tight_layout()
        fig.savefig(filepath, format='pdf')
        fig.savefig(filepath.with_suffix('.png'), format='png')
        plt.close(fig)
        return str(filepath)

    def plot_heatmap(self, df: pd.DataFrame) -> str:
        """Heatmap of scene × metric scores (averaged across levels)."""
        # Average across levels for each metric
        avg_data = {}
        for metric in METRICS:
            cols = [f'L{l}_{metric}' for l in [1, 2, 3]]
            existing_cols = [c for c in cols if c in df.columns]
            if existing_cols:
                avg_data[metric.replace('_', ' ').title()] = df[existing_cols].mean(axis=1).values

        if not avg_data:
            print("[plot] WARNING: No data for heatmap")
            return ""

        heatmap_df = pd.DataFrame(avg_data, index=df['scene_id'].values)

        # Height: min 2 inches for 1 scene, scale up with more scenes
        height = max(2, min(12, len(df) * 0.8 + 1.5))
        fig, ax = plt.subplots(figsize=(10, height))
        sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=1, vmax=5, center=3, linewidths=0.5,
                    cbar_kws={'label': 'Score (1-5)'},
                    annot_kws={'color': 'black', 'fontweight': 'bold', 'fontsize': 11},
                    ax=ax)
        ax.set_title('Evaluation Scores Heatmap (Averaged Across Levels)',
                    fontweight='bold', color='black')
        ax.set_xlabel('Metric', fontweight='bold', color='black')
        ax.set_ylabel('Scene', fontweight='bold', color='black')
        # Make tick labels bold and black
        ax.tick_params(colors='black')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_color('black')

        filepath = self.plots_dir / 'heatmap_scene_metrics.pdf'
        fig.tight_layout()
        fig.savefig(filepath, format='pdf')
        fig.savefig(filepath.with_suffix('.png'), format='png')
        plt.close(fig)
        return str(filepath)

    def plot_box_distributions(self, df: pd.DataFrame) -> str:
        """Box plots showing score distributions per metric."""
        fig, axes = plt.subplots(1, 5, figsize=(15, 5), sharey=True)

        for idx, metric in enumerate(METRICS):
            ax = axes[idx]
            data = []
            labels = []
            colors = []
            for level in [1, 2, 3]:
                col = f'L{level}_{metric}'
                if col in df.columns:
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        data.append(valid_data)
                        labels.append(f'L{level}')
                        colors.append(LEVEL_COLORS[f'L{level}'])

            if data:
                bp = ax.boxplot(data, patch_artist=True, labels=labels)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            ax.set_title(metric.replace('_', '\n').title(), fontsize=11,
                        fontweight='bold', color='black')
            ax.set_ylim(0.5, 5.5)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(colors='black')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_color('black')
            if idx == 0:
                ax.set_ylabel('Score (1-5)', fontweight='bold', color='black')

        fig.suptitle('Score Distributions by Metric and Difficulty Level',
                     fontsize=14, y=1.02, fontweight='bold', color='black')

        filepath = self.plots_dir / 'box_distributions.pdf'
        fig.tight_layout()
        fig.savefig(filepath, format='pdf')
        fig.savefig(filepath.with_suffix('.png'), format='png')
        plt.close(fig)
        return str(filepath)

    def plot_summary_dashboard(self, df: pd.DataFrame) -> str:
        """Summary dashboard with key statistics."""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Calculate overall stats
        overall_means = []
        overall_stds = []
        for level in [1, 2, 3]:
            cols = [f'L{level}_{m}' for m in METRICS]
            existing_cols = [c for c in cols if c in df.columns]
            if existing_cols:
                overall_means.append(df[existing_cols].mean().mean())
                overall_stds.append(df[existing_cols].std().mean())
            else:
                overall_means.append(0)
                overall_stds.append(0)

        # 1. Overall scores by level (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(['Level 1', 'Level 2', 'Level 3'], overall_means,
                       yerr=overall_stds, capsize=5,
                       color=[LEVEL_COLORS[f'L{i}'] for i in [1, 2, 3]],
                       edgecolor='black', linewidth=1)
        ax1.set_ylabel('Overall Score (1-5)', fontweight='bold', color='black')
        ax1.set_title('Overall Score by Difficulty', fontweight='bold', color='black')
        ax1.set_ylim(0, 5.5)
        ax1.tick_params(colors='black')
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')
            label.set_color('black')
        for bar, val in zip(bars, overall_means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11,
                    fontweight='bold', color='black')

        # 2. Best/Worst metrics (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        all_metric_means = {}
        for metric in METRICS:
            cols = [f'L{l}_{metric}' for l in [1, 2, 3]]
            existing_cols = [c for c in cols if c in df.columns]
            if existing_cols:
                all_metric_means[metric] = df[existing_cols].mean().mean()

        if all_metric_means:
            sorted_metrics = sorted(all_metric_means.items(), key=lambda x: x[1], reverse=True)
            names = [m[0].replace('_', ' ').title() for m in sorted_metrics]
            values = [m[1] for m in sorted_metrics]
            colors_ranked = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(names)))

            bars = ax2.barh(names, values, color=colors_ranked, edgecolor='black', linewidth=1)
            ax2.set_xlim(0, 5)
            ax2.set_xlabel('Average Score', fontweight='bold', color='black')
            ax2.set_title('Metrics Ranked by Performance', fontweight='bold', color='black')
            ax2.tick_params(colors='black')
            for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                label.set_fontweight('bold')
                label.set_color('black')
            for bar, val in zip(bars, values):
                ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}', ha='left', va='center', fontsize=10,
                        fontweight='bold', color='black')
        else:
            sorted_metrics = []

        # 3. Summary stats (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')

        best_metric = sorted_metrics[0] if sorted_metrics else ('N/A', 0)
        worst_metric = sorted_metrics[-1] if sorted_metrics else ('N/A', 0)

        stats_text = f"""
EVALUATION SUMMARY
═══════════════════════════════

Total Scenes: {len(df)}

Mean Scores:
  • Level 1: {overall_means[0]:.2f} ± {overall_stds[0]:.2f}
  • Level 2: {overall_means[1]:.2f} ± {overall_stds[1]:.2f}
  • Level 3: {overall_means[2]:.2f} ± {overall_stds[2]:.2f}

Best Metric: {best_metric[0].replace('_', ' ').title()}
             ({best_metric[1]:.2f})

Worst Metric: {worst_metric[0].replace('_', ' ').title()}
              ({worst_metric[1]:.2f})

Judge Model: GPT-4o
        """
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 fontweight='bold', color='black',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

        # 4. Radar (bottom-left + center)
        ax4 = fig.add_subplot(gs[1, :2], polar=True)
        level_data = {}
        for level in [1, 2, 3]:
            cols = [f'L{level}_{m}' for m in METRICS]
            existing_cols = [c for c in cols if c in df.columns]
            if existing_cols:
                level_data[f'L{level}'] = df[existing_cols].mean().values

        angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
        angles += angles[:1]

        for level, values in level_data.items():
            values = list(values) + [values[0]]
            ax4.plot(angles, values, 'o-', linewidth=2, label=level,
                    color=LEVEL_COLORS[level])
            ax4.fill(angles, values, alpha=0.15, color=LEVEL_COLORS[level])

        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(METRIC_LABELS, fontweight='bold', color='black', fontsize=10)
        ax4.set_ylim(0, 5)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), prop={'weight': 'bold'})
        ax4.set_title('Metric Comparison Across Levels', pad=20, fontweight='bold', color='black')
        ax4.tick_params(colors='black')
        for label in ax4.get_yticklabels():
            label.set_fontweight('bold')
            label.set_color('black')

        # 5. Score distribution histogram (bottom-right)
        ax5 = fig.add_subplot(gs[1, 2])
        all_scores = []
        for level in [1, 2, 3]:
            for metric in METRICS:
                col = f'L{level}_{metric}'
                if col in df.columns:
                    all_scores.extend(df[col].dropna().tolist())

        if all_scores:
            ax5.hist(all_scores, bins=np.arange(0.5, 6, 0.5), color='steelblue',
                     edgecolor='black', alpha=0.7, linewidth=1)
            ax5.axvline(np.mean(all_scores), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(all_scores):.2f}')
            ax5.set_xlabel('Score', fontweight='bold', color='black')
            ax5.set_ylabel('Frequency', fontweight='bold', color='black')
            ax5.set_title('Overall Score Distribution', fontweight='bold', color='black')
            ax5.legend(prop={'weight': 'bold'})
            ax5.tick_params(colors='black')
            for label in ax5.get_xticklabels() + ax5.get_yticklabels():
                label.set_fontweight('bold')
                label.set_color('black')

        fig.suptitle('VLM-as-Judge Evaluation Dashboard', fontsize=16,
                     fontweight='bold', color='black', y=0.98)

        filepath = self.plots_dir / 'summary_dashboard.pdf'
        fig.savefig(filepath, format='pdf')
        fig.savefig(filepath.with_suffix('.png'), format='png')
        plt.close(fig)
        return str(filepath)
