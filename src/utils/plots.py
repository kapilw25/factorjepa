"""Shared plotting utilities for publication-quality figures (Tier 1 research papers).

Style: Large bold text, high contrast, clean axes. Both .png (150 DPI) and .pdf.
All plots use this module's rcParams for consistency across m08, m08b, m09, etc.

USAGE:
    from utils.plots import init_style, save_fig, COLORS
    init_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color=COLORS["blue"])
    save_fig(fig, "outputs/full/my_plot")  # saves .png + .pdf
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ── Publication color palette (high contrast, colorblind-safe) ────────
COLORS = {
    "blue": "#1565C0",
    "red": "#C62828",
    "green": "#2E7D32",
    "orange": "#E65100",
    "purple": "#6A1B9A",
    "cyan": "#00838F",
    "gray": "#616161",
    "gold": "#F9A825",
    # Lambda-specific colors for ablation plots
    "lambda0": "#C62828",      # red
    "lambda0_001": "#1565C0",  # blue
    "lambda0_01": "#2E7D32",   # green
    "lambda0_1": "#E65100",    # orange
}

# Encoder colors (consistent across m08, m08b, radar)
ENCODER_COLORS = {
    "vjepa": "#1565C0",
    "random": "#9E9E9E",
    "dinov2": "#2E7D32",
    "clip": "#E65100",
    "vjepa_shuffled": "#6A1B9A",
    "vjepa_lambda0": "#C62828",
    "vjepa_lambda0_001": "#00838F",
    "vjepa_lambda0_01": "#F9A825",
    "vjepa_lambda0_1": "#795548",
}

# Scene colors
SCENE_COLORS = {
    "market": "#C62828", "junction": "#2E7D32", "residential_lane": "#6A1B9A",
    "promenade": "#1565C0", "transit": "#9E9E9E", "heritage_tourist": "#E65100",
    "highway": "#F9A825", "alley": "#795548", "commercial": "#00838F",
    "construction": "#AD1457", "beach_coastal": "#0277BD",
    "ghat": "#558B2F", "temple_tourist": "#FF6F00", "unknown": "#616161",
}


def init_style():
    """Set matplotlib rcParams for Tier 1 publication quality.

    Matches reference: trash/combined_accuracy_withSFT.png + trash/plotting.py.
    Font: serif, bold globally, black text. DPI: 300 for print.
    """
    plt.rcParams.update({
        # Font — serif + bold (matches reference plotting.py)
        "font.size": 14,
        "font.weight": "bold",
        "font.family": "serif",

        # Text color — all black for print
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",

        # Axes
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "axes.labelweight": "bold",
        "axes.linewidth": 1.5,
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.grid": True,
        "axes.axisbelow": True,

        # Grid
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
        "grid.color": "#CCCCCC",

        # Ticks
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Legend
        "legend.fontsize": 13,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",
        "legend.fancybox": False,

        # Lines
        "lines.linewidth": 2.5,
        "lines.markersize": 8,

        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 100,
        "figure.titlesize": 20,
        "figure.titleweight": "bold",

        # Savefig — 300 DPI for Tier 1 print quality
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.facecolor": "white",
    })


def save_fig(fig, path_stem: str, dpi: int = 300):
    """Save figure as both .png and .pdf.

    Args:
        fig: matplotlib Figure.
        path_stem: Path without extension (e.g., "outputs/full/m09_ablation").
                   Saves path_stem.png and path_stem.pdf.
        dpi: Resolution for PNG (default 150).
    """
    path = Path(path_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path) + ".png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path) + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}.png + .pdf")


def plot_val_loss_curves(ablation_dir, lambdas: list, winner: str,
                         output_path: str = None, batch_size: int = 32):
    """Plot val_loss curves from loss_log.csv for all lambda ablation runs.

    X-axis shows number of CLIPS (step × batch_size), not steps.
    This is critical for Tier 1 papers — reviewers need sample size context.

    Args:
        ablation_dir: Path to ablation directory containing m09_lambda*/ subdirs.
        lambdas: List of lambda string values (e.g., ["0", "0.001", "0.01", "0.1"]).
        winner: The winning lambda string (e.g., "0.001").
        output_path: Path stem for output (default: ablation_dir/m09_ablation_val_curves).
        batch_size: Batch size used during ablation (for step→clips conversion).
    """
    init_style()
    ablation_dir = Path(ablation_dir)

    lambda_colors = {
        "0": COLORS["lambda0"],
        "0.001": COLORS["lambda0_001"],
        "0.01": COLORS["lambda0_01"],
        "0.1": COLORS["lambda0_1"],
    }

    # Auto-detect batch_size from first lambda's training_summary
    for lam in lambdas:
        lam_dir = "lambda" + lam.replace(".", "_")
        summary = ablation_dir / f"m09_{lam_dir}" / "training_summary.json"
        if summary.exists():
            s = json.load(open(summary))
            if s.get("batch_size"):
                batch_size = s["batch_size"]
                break

    fig, ax = plt.subplots(figsize=(10, 6))

    for lam in lambdas:
        lam_dir = "lambda" + lam.replace(".", "_")
        csv_path = ablation_dir / f"m09_{lam_dir}" / "loss_log.csv"

        steps, val_losses = [], []

        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                val_col = None
                for idx, col in enumerate(header):
                    if col.strip() == "val_loss":
                        val_col = idx
                        break
                if val_col is not None:
                    for row in reader:
                        if len(row) > val_col and row[val_col].strip():
                            try:
                                steps.append(int(row[0]))
                                val_losses.append(float(row[val_col]))
                            except (ValueError, IndexError):
                                continue

        if not val_losses:
            summary = ablation_dir / f"m09_{lam_dir}" / "training_summary.json"
            if summary.exists():
                s = json.load(open(summary))
                if s.get("best_val_loss") and s["best_val_loss"] != float("inf"):
                    val_losses = [s["best_val_loss"]]
                    steps = [s.get("steps", 0)]

        if not val_losses:
            continue

        # Convert steps to clips (what reviewers care about)
        clips = [s * batch_size for s in steps]

        is_winner = (lam == winner)
        label = f"$\\lambda$={lam}"
        if is_winner:
            label += " (winner)"
        color = lambda_colors.get(lam, COLORS["gray"])
        linewidth = 3.5 if is_winner else 2.0
        marker = "s" if is_winner else "o"
        zorder = 10 if is_winner else 5

        ax.plot(clips, val_losses, marker=marker, label=label, color=color,
                linewidth=linewidth, markersize=7 if is_winner else 5,
                zorder=zorder)

        best_idx = np.argmin(val_losses)
        best_val = val_losses[best_idx]
        best_clip = clips[best_idx]
        ax.annotate(f"{best_val:.4f}",
                    xy=(best_clip, best_val),
                    xytext=(0, -18 if is_winner else 12),
                    textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold",
                    color=color)

    # Format x-axis as "K clips"
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"))
    ax.set_xlabel("Training Clips")
    ax.set_ylabel("Validation JEPA Loss")
    ax.set_title("Lambda Ablation: Validation Loss Curves")
    ax.legend(loc="upper right")

    if output_path is None:
        output_path = str(ablation_dir / "m09_ablation_val_curves")
    save_fig(fig, output_path)


def _read_loss_log(csv_path: str) -> dict:
    """Parse loss_log.jsonl (primary) or loss_log.csv (fallback) into arrays."""
    steps_train, jepa_loss, drift_loss, total_loss = [], [], [], []
    steps_val, val_loss = [], []

    # Try JSONL first (crash-safe, no data loss)
    jsonl_path = Path(csv_path).with_suffix(".jsonl")
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip corrupted last line (expected on SIGKILL)
                step = record["step"]
                if "val_loss" in record:
                    steps_val.append(step)
                    val_loss.append(record["val_loss"])
                elif "loss_jepa" in record:
                    steps_train.append(step)
                    jepa_loss.append(record["loss_jepa"])
                    drift_loss.append(record.get("loss_drift", 0.0))
                    total_loss.append(record.get("loss_total", 0.0))
        return {
            "steps_train": steps_train, "jepa_loss": jepa_loss,
            "drift_loss": drift_loss, "total_loss": total_loss,
            "steps_val": steps_val, "val_loss": val_loss,
        }

    # Fallback: CSV (legacy)
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 8:
                continue
            step = int(row[0])
            if row[2].strip() == "" and len(row) > 8 and row[8].strip():
                try:
                    steps_val.append(step)
                    val_loss.append(float(row[8]))
                except ValueError:
                    continue
            elif row[2].strip():
                try:
                    steps_train.append(step)
                    jepa_loss.append(float(row[2]))
                    drift_loss.append(float(row[3]))
                    total_loss.append(float(row[4]))
                except ValueError:
                    continue

    return {
        "steps_train": steps_train, "jepa_loss": jepa_loss,
        "drift_loss": drift_loss, "total_loss": total_loss,
        "steps_val": steps_val, "val_loss": val_loss,
    }


def plot_training_curves(runs: list, output_dir: str, title_prefix: str = ""):
    """Generate 3 separate publication-quality plots from training loss_log.csv files.

    Supports multiple runs for comparison (e.g., V-JEPA 2.0 vs 2.1).
    X-axis shows number of CLIPS (step × batch_size from training_summary.json).
    batch_size is read from training_summary.json — never hardcoded.

    Args:
        runs: List of dicts, each with:
            - "csv_path": path to loss_log.csv
            - "label": legend label (e.g., "V-JEPA 2.0")
            - "color": color key from COLORS (optional, auto-assigned)
        output_dir: Directory for output plots.
        title_prefix: Optional prefix for titles (e.g., "115K Clips, ").

    Generates:
        1. {output_dir}/m09_val_loss.png/.pdf
        2. {output_dir}/m09_train_loss.png/.pdf
        3. {output_dir}/m09_drift_loss.png/.pdf
    """
    init_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    auto_colors = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["orange"]]

    parsed_runs = []
    for i, run in enumerate(runs):
        data = _read_loss_log(run["csv_path"])
        label = run.get("label", f"Run {i+1}")
        color = COLORS.get(run.get("color", ""), auto_colors[i % len(auto_colors)])

        # Read batch_size: training_summary.json (completed) → caller-provided → FATAL
        batch_size = run.get("batch_size")
        summary_path = Path(run["csv_path"]).parent / "training_summary.json"
        if summary_path.exists():
            s = json.load(open(summary_path))
            batch_size = s["batch_size"]
        if batch_size is None:
            print(f"FATAL: batch_size unknown for {run['csv_path']}.")
            print("  Provide batch_size in run dict or wait for training_summary.json.")
            return

        data["clips_train"] = [step * batch_size for step in data["steps_train"]]
        data["clips_val"] = [step * batch_size for step in data["steps_val"]]
        data["label"] = label
        data["color"] = color
        parsed_runs.append(data)

    def _fmt_clips(x, _):
        return f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}"

    # ── Plot 1: Validation Loss ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for run in parsed_runs:
        if run["clips_val"]:
            ax.plot(run["clips_val"], run["val_loss"], "s-",
                    color=run["color"], linewidth=3.0, markersize=8,
                    label=run["label"], zorder=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_clips))
    ax.set_xlabel("Training Clips")
    ax.set_ylabel("Validation JEPA Loss")
    ax.set_title(f"{title_prefix}Validation Loss")
    ax.legend(loc="upper right")
    save_fig(fig, str(out / "m09_val_loss"))

    # ── Plot 2: Training Loss (raw + smoothed moving average) ──────
    fig, ax = plt.subplots(figsize=(10, 6))
    for run in parsed_runs:
        if run["clips_train"]:
            clips = np.array(run["clips_train"])
            loss = np.array(run["jepa_loss"])
            # Raw loss (light, thin)
            ax.plot(clips, loss,
                    color=run["color"], linewidth=0.5, alpha=0.3)
            # Smoothed moving average (bold red)
            window = max(1, len(loss) // 20)  # 5% window
            if window > 1:
                kernel = np.ones(window) / window
                smoothed = np.convolve(loss, kernel, mode="valid")
                smoothed_clips = clips[window - 1:]
                ax.plot(smoothed_clips, smoothed,
                        color=COLORS["red"], linewidth=3.0,
                        label=f"{run['label']} (smoothed)", zorder=10)
            else:
                ax.plot(clips, loss,
                        color=COLORS["red"], linewidth=3.0,
                        label=run["label"], zorder=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_clips))
    ax.set_xlabel("Training Clips")
    ax.set_ylabel("Training JEPA Loss")
    ax.set_title(f"{title_prefix}Training Loss")
    ax.legend(loc="upper right")
    save_fig(fig, str(out / "m09_train_loss"))

    # ── Plot 3: Drift Loss ───────────────────────────────────────────
    has_drift = any(run["drift_loss"] and max(run["drift_loss"]) > 0 for run in parsed_runs)
    if has_drift:
        fig, ax = plt.subplots(figsize=(10, 6))
        for run in parsed_runs:
            if run["clips_train"] and run["drift_loss"] and max(run["drift_loss"]) > 0:
                ax.plot(run["clips_train"], run["drift_loss"],
                        color=run["color"], linewidth=2.0,
                        label=run["label"])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_clips))
        ax.set_xlabel("Training Clips")
        ax.set_ylabel("Drift Loss ($\\lambda \\|\\theta - \\theta_0\\|^2$)")
        ax.set_title(f"{title_prefix}Drift Control Loss")
        ax.legend(loc="upper right")
        save_fig(fig, str(out / "m09_drift_loss"))
    else:
        print("  SKIP: drift loss plot (all zeros — drift control disabled)")
