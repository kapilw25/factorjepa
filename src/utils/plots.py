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
from matplotlib.ticker import MultipleLocator
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

# Encoder colors (consistent across m08, m08b, radar, probe_plot).
# iter13: explicit per-variant entries for the 4 V-JEPA encoders so they
# render with distinct hues across all plots (probe_action_loss/acc +
# probe_encoder_comparison). Without these, all 4 fall through to the
# generic "vjepa" key and render as identical #1565C0 blue → train/val
# curves overlap visually + bar chart shows 4 indistinguishable bars.
# Palette: colorblind-safe, distinct from dinov2 green / clip orange /
# random gray / vjepa_shuffled #6A1B9A purple. baseline-frozen keeps the
# canonical #1565C0 blue so legacy plots stay anchored.
ENCODER_COLORS = {
    "vjepa": "#1565C0",                       # generic fallback (legacy)
    "vjepa_2_1_frozen":              "#1565C0",   # blue — baseline anchor
    # iter14 / iter15 ENCODER-update variants (saturated tones)
    "vjepa_2_1_pretrain_encoder":            "#D81B60",   # magenta — continual SSL
    "vjepa_2_1_pretrain_2X_encoder":         "#FF8F00",   # amber — pretrain 2× arm (iter14 C)
    "vjepa_2_1_surgical_3stage_DI_encoder":  "#5E35B1",   # deep indigo — surgery WITH D_I
    "vjepa_2_1_surgical_noDI_encoder":       "#00ACC1",   # cyan — surgery WITHOUT D_I
    # iter15 HEAD-only variants (lighter tone of paired encoder counterpart)
    # so paired-Δ Δ5/Δ6/Δ7 are visually obvious — each pair shares a hue family.
    "vjepa_2_1_pretrain_head":               "#F06292",   # pink — paired w/ pretrain_encoder magenta
    "vjepa_2_1_surgical_3stage_DI_head":     "#9575CD",   # lavender — paired w/ 3stage_DI deep indigo
    "vjepa_2_1_surgical_noDI_head":          "#4DD0E1",   # light cyan — paired w/ noDI cyan
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
    """Parse loss_log.jsonl (primary) or loss_log.csv (fallback) into arrays.

    Also extracts per-step "stage" when present (surgery only) — returned as
    `stages_train` (parallel to `steps_train`). Empty list for ExPLoRA/pretrain.
    """
    steps_train, jepa_loss, drift_loss, total_loss = [], [], [], []
    steps_val, val_loss = [], []
    stages_train = []  # m09c only — empty for m09a/m09b
    lr_train: list = []  # iter13 (2026-05-05): per-step LR for live LR-curve plot

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
                    stages_train.append(record.get("stage", ""))
                    lr_train.append(record.get("lr", 0.0))
        return {
            "steps_train": steps_train, "jepa_loss": jepa_loss,
            "drift_loss": drift_loss, "total_loss": total_loss,
            "stages_train": stages_train, "lr_train": lr_train,
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
        "stages_train": stages_train, "lr_train": lr_train,
        "steps_val": steps_val, "val_loss": val_loss,
    }


def plot_training_curves(runs: list, output_dir: str, title_prefix: str = "",
                         x_axis_mode: str = "steps",
                         file_prefix: str = "m09"):
    """Generate 3 separate publication-quality plots from training loss_log.csv files.

    iter13 v12 (2026-05-06): added `file_prefix` so each m09{a,b,c} module's
    plots get its own filename namespace — m09a/* from m09a1_pretrain_encoder.py,
    m09c/* from m09c1_surgery_encoder.py, m09b/* from m09b_explora.py. Default "m09"
    preserves backwards-compat for any caller that hasn't migrated.

    Supports multiple runs for comparison (e.g., V-JEPA 2.0 vs 2.1).

    Args:
        runs: List of dicts, each with:
            - "csv_path": path to loss_log.csv
            - "label": legend label (e.g., "V-JEPA 2.0")
            - "color": color key from COLORS (optional, auto-assigned)
            - "batch_size": required only when x_axis_mode="clip_visits"
        output_dir: Directory for output plots.
        title_prefix: Optional prefix for titles (e.g., "115K Clips, ").
        x_axis_mode: "steps" (default — x = optimizer steps, always correct) or
            "clip_visits" (x = step × batch_size). "clip_visits" is MISLEADING
            for surgery/SANITY/POC where the producer samples with replacement
            from a small pool — it over-reports by the number of repeats. Only
            use "clip_visits" for FULL-mode continual pretraining where the
            producer draws fresh clips from a stream larger than step × BS.

    Generates:
        1. {output_dir}/m09_val_loss.png/.pdf
        2. {output_dir}/m09_train_loss.png/.pdf  (color-segmented by stage when
           JSONL contains "stage" field — i.e., m09c surgery runs)
        3. {output_dir}/m09_drift_loss.png/.pdf  (skipped when drift all zero)
    """
    assert x_axis_mode in ("steps", "clip_visits"), \
        f"x_axis_mode must be 'steps' or 'clip_visits', got {x_axis_mode!r}"
    init_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    auto_colors = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["orange"]]

    parsed_runs = []
    for i, run in enumerate(runs):
        data = _read_loss_log(run["csv_path"])
        label = run.get("label", f"Run {i+1}")
        color = COLORS.get(run.get("color", ""), auto_colors[i % len(auto_colors)])

        if x_axis_mode == "clip_visits":
            # Read batch_size: training_summary.json → caller-provided → FATAL
            batch_size = run.get("batch_size")
            summary_path = Path(run["csv_path"]).parent / "training_summary.json"
            if summary_path.exists():
                s = json.load(open(summary_path))
                batch_size = s["batch_size"]
            if batch_size is None:
                print(f"FATAL: batch_size unknown for {run['csv_path']}.")
                print("  Provide batch_size in run dict or wait for training_summary.json.")
                return
            data["x_train"] = [step * batch_size for step in data["steps_train"]]
            data["x_val"] = [step * batch_size for step in data["steps_val"]]
        else:  # "steps"
            data["x_train"] = list(data["steps_train"])
            data["x_val"] = list(data["steps_val"])

        data["label"] = label
        data["color"] = color
        parsed_runs.append(data)

    def _fmt_x(x, _):
        return f"{x/1000:.1f}K" if x >= 1000 else f"{x:.0f}"

    x_label = "Training Clip-Visits (step × BS)" if x_axis_mode == "clip_visits" else "Optimizer Steps"

    def _stage_markers(run):
        """Return list of (x_position, stage_name) where stage changes, for vlines."""
        stages = run.get("stages_train", [])
        if not stages or len(stages) != len(run["x_train"]):
            return []
        markers = []
        last = None
        for x, s in zip(run["x_train"], stages):
            if s and s != last:
                markers.append((x, s))
                last = s
        return markers

    # 8-color palette for stage segmentation (first 3 match common Ch11 stage colors)
    stage_colors = [COLORS["green"], COLORS["orange"], COLORS["purple"],
                    COLORS["red"], COLORS["blue"], "#8B4513", "#FF1493", "#00CED1"]

    # iter14 (2026-05-08): Plot 1 (basic val_loss.png) AND Plot 1b (val_loss_jepa
    # zoom) BOTH RETIRED. They were dead code:
    #   • Plot 1 was a simpler styling of the same data Plot 1b already showed.
    #   • Plot 1b's <prefix>_val_loss_jepa.png was IMMEDIATELY OVERWRITTEN for
    #     m09a by plot_val_loss_with_kill_switch_overlay (richer kill-switch +
    #     best-marker version). For m09c it was a no-op (m09c writes
    #     "val_jepa_loss" key, not "val_loss" → fell to else-branch's misleading
    #     skip message). Net: Plot 1b was wasted I/O for m09a + log-noise for m09c.
    #
    # Single source of truth = plot_val_loss_with_kill_switch_overlay (utils.plots
    # line ~865+), called directly from m09a:_render_m09a_probe_plots and
    # m09c:_render_live_plots. Reads probe_history (in-memory list), not loss_log.

    # ── Plot 2: Training Loss — color-segmented by stage when present ──
    fig, ax = plt.subplots(figsize=(12, 6))
    for run in parsed_runs:
        if not run["x_train"]:
            continue
        x = np.array(run["x_train"])
        loss = np.array(run["jepa_loss"])

        # Raw line always drawn (light, thin)
        ax.plot(x, loss, color=run["color"], linewidth=0.5, alpha=0.25, zorder=1)

        # If stages present → color-segment smoothed curve + shade stage regions
        markers = _stage_markers(run)
        if markers:
            # Build per-stage index ranges
            stage_ranges = []
            for i, (x_start, s_name) in enumerate(markers):
                x_end = markers[i + 1][0] if i + 1 < len(markers) else x[-1] + 1
                idx = np.where((x >= x_start) & (x < x_end))[0]
                if len(idx) > 0:
                    stage_ranges.append((s_name, idx, x_start, x_end))

            # Smoothed per-stage segment. iter14 (2026-05-08): markers added
            # so short segments (1-step stages at POC) render visibly — pre-fix
            # 1-point segments drew a 0-length invisible line.
            for i, (s_name, idx, x_start, x_end) in enumerate(stage_ranges):
                seg_x = x[idx]
                seg_loss = loss[idx]
                col = stage_colors[i % len(stage_colors)]
                window = max(1, len(seg_loss) // 10)
                if window > 1:
                    k = np.ones(window) / window
                    sm = np.convolve(seg_loss, k, mode="valid")
                    sm_x = seg_x[window - 1:]
                    ax.plot(sm_x, sm, color=col, linewidth=2.8, marker="o",
                            markersize=5, label=f"{s_name}", zorder=10)
                else:
                    ax.plot(seg_x, seg_loss, color=col, linewidth=2.8,
                            marker="o", markersize=8,
                            label=f"{s_name}", zorder=10)
                # Shade stage band + draw transition vline at stage start
                ax.axvspan(x_start, x_end, alpha=0.04, color=col, zorder=0)
                if i > 0:
                    ax.axvline(x_start, color="gray", linestyle="--",
                               linewidth=1.0, alpha=0.7, zorder=2)
        else:
            # No stages (m09a / m09b) — single smoothed line
            window = max(1, len(loss) // 20)
            if window > 1:
                k = np.ones(window) / window
                sm = np.convolve(loss, k, mode="valid")
                ax.plot(x[window - 1:], sm, color=COLORS["red"],
                        linewidth=3.0, label=f"{run['label']} (smoothed)", zorder=10)
            else:
                ax.plot(x, loss, color=COLORS["red"], linewidth=3.0,
                        label=run["label"], zorder=10)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_x))
    ax.set_xlabel(x_label)
    ax.set_ylabel("Training JEPA Loss")
    ax.set_title(f"{title_prefix}Training Loss")
    ax.legend(loc="upper right")

    # Dual x-axis: show "Training samples seen" (= n_unique × n_epochs) on top so
    # reader sees both granular step count AND "how many passes through the data"
    # at a glance. Identity: step × BS = n_unique_clips × n_epochs (by construction
    # since steps = n_clips × epochs / BS). Only added when x_axis_mode="steps" and
    # we have a training_summary.json nearby to read total_clips + derive n_epochs.
    if x_axis_mode == "steps":
        try:
            first_run = runs[0]
            summary_path = Path(first_run["csv_path"]).parent / "training_summary.json"
            if summary_path.exists():
                s = json.load(open(summary_path))
                n_unique = s["total_factor_clips"]
                n_steps = s["steps"]
                bs = s["batch_size"]
                n_epochs = round(n_steps * bs / n_unique) if n_unique else 0
                total_samples = n_unique * n_epochs
                ax2 = ax.twiny()
                ax2.set_xlim([v * bs for v in ax.get_xlim()])
                ax2.set_xlabel(
                    f"Training samples seen  (= {n_unique:,} unique clips × "
                    f"{n_epochs} epochs = {total_samples:,})", fontsize=9)
        except Exception as e:
            # iter13 (2026-05-05): per CLAUDE.md FAIL HARD.
            print(f"  [plots] FATAL: dual-axis on m09_train_loss failed: {e}", flush=True)
            raise

    # iter13 (2026-05-05): LR overlay on right y-axis. Reads `lr` per step from
    # loss_log.jsonl (per-step) so the warmup ramp + cosine decay are visible
    # alongside the training-loss trajectory. Helps diagnose "loss flat because
    # LR is still in warmup" (the iter13 v9 case where step 43 LR was 4.94e-5,
    # 49% of peak — drift was slow simply because LR hadn't peaked yet).
    if any(run.get("lr_train") for run in parsed_runs):
        ax_lr = ax.twinx()
        for run in parsed_runs:
            lr = run.get("lr_train") or []
            if not lr:
                continue
            lx = (np.array(run["x_train"]) if x_axis_mode == "steps"
                  else np.array(run["x_train"]))
            ax_lr.plot(lx[:len(lr)], np.array(lr),
                       color="#FF8C00", linewidth=1.5, linestyle="--",
                       alpha=0.85, zorder=20, label="LR")
        ax_lr.set_ylabel("Learning rate", color="#FF8C00", fontsize=10)
        ax_lr.tick_params(axis="y", labelcolor="#FF8C00")
        ax_lr.set_yscale("log")
        # Merge LR legend into main ax
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_lr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")

    save_fig(fig, str(out / f"{file_prefix}_train_loss"))

    # iter14 (2026-05-08): Plot 3 (standalone drift_loss.png) RETIRED — was a
    # strict subset of plot_combined_losses (loss_decomposition.png) which
    # already plots loss_drift on the right y-axis (1000× scale to match
    # drift's magnitude). Single source of drift visualization = loss_decomposition.png.


def plot_combined_losses(jsonl_path, output_dir, title_prefix: str = "",
                          file_prefix: str = "m09") -> None:
    """4-loss decomposition plot — single image, dual y-axis.

    iter13 v12: file_prefix for per-module namespacing (m09a/m09b/m09c).

    Plots `loss_jepa` / `loss_multi_task` / `loss_total` on the left axis (similar
    magnitude 0.4–1.5) and `loss_drift` on the right axis (1000× smaller scale).
    `loss_total` is BOLD/thick (linewidth=3.5) since it is the optimizer's actual
    minimization target — visually emphasizes which component dominates the gradient.

    Reads `loss_log.jsonl` (per-step training records); skips val rows (those
    have `val_loss` instead of `loss_jepa`). Saves
    `output_dir/m09_loss_decomposition.{png,pdf}`. No-op when fewer than 2 train
    rows are present (avoids degenerate single-point plot).
    """
    init_style()
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    if not jsonl_path.exists():
        print(f"  SKIP plot_combined_losses: {jsonl_path} not found")
        return

    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "loss_jepa" not in r:
                continue   # val rows have `val_loss` only
            rows.append(r)

    if len(rows) < 2:
        print(f"  SKIP plot_combined_losses: only {len(rows)} train rows in {jsonl_path}")
        return

    steps = [r["step"] for r in rows]
    jepa  = [r["loss_jepa"] for r in rows]
    drift = [r.get("loss_drift", 0.0) for r in rows]
    mt    = [r.get("loss_multi_task", 0.0) for r in rows]
    total = [r["loss_total"] for r in rows]

    # Detect whether multi-task is active in this run
    has_mt = any(v > 0 for v in mt)

    # Portrait orientation (7:11 ≈ 1:1.6) — fits NeurIPS single column and
    # stretches the y-axis so per-step Δ in jepa/total is eyeball-readable.
    fig, ax_left = plt.subplots(figsize=(7, 11))
    ax_right = ax_left.twinx()

    # Rolling-mean smoother. Mirrors plot_training_curves pattern (utils/plots.py:526-534)
    # so all m09* training plots share the same raw + smoothed convention.
    # window = max(1, len(y) // 20) → 5% of run, balances responsiveness vs noise.
    steps_arr = np.asarray(steps)
    def _smoothed(y):
        y = np.asarray(y)
        w = max(1, len(y) // 20)
        if w <= 1:
            return None, None
        k = np.ones(w) / w
        return steps_arr[w - 1:], np.convolve(y, k, mode="valid")

    # Left axis — raw (light, thin, alpha=0.20) + smoothed (full opacity, thick) for
    # jepa, mt, total. Raw shows variance band; smoothed carries the trend.
    ax_left.plot(steps, jepa, color=COLORS["orange"], linewidth=0.6, alpha=0.20, zorder=2)
    sx, sm = _smoothed(jepa)
    if sm is not None:
        ax_left.plot(sx, sm, color=COLORS["orange"], linewidth=2.2, alpha=0.95,
                     label="loss_jepa  (V-JEPA L1, native objective)", zorder=8)
    else:
        ax_left.plot(steps, jepa, color=COLORS["orange"], linewidth=1.8, alpha=0.85,
                     label="loss_jepa  (V-JEPA L1, native objective)")

    if has_mt:
        ax_left.plot(steps, mt, color=COLORS["purple"], linewidth=0.6, alpha=0.20, zorder=2)
        sx, sm = _smoothed(mt)
        if sm is not None:
            ax_left.plot(sx, sm, color=COLORS["purple"], linewidth=2.2, alpha=0.95,
                         label="loss_multi_task  (16-dim taxonomy CE+BCE)", zorder=8)
        else:
            ax_left.plot(steps, mt, color=COLORS["purple"], linewidth=1.8, alpha=0.85,
                         label="loss_multi_task  (16-dim taxonomy CE+BCE)")

    # total_loss = optimizer target — raw underlay + BOLD smoothed on top (zorder=10)
    ax_left.plot(steps, total, color=COLORS["red"], linewidth=0.6, alpha=0.20, zorder=2)
    sx, sm = _smoothed(total)
    if sm is not None:
        ax_left.plot(sx, sm, color=COLORS["red"], linewidth=3.5, alpha=1.0,
                     label="loss_total  (= jepa + 0.1*mt + drift)  [OPTIMIZER TARGET]",
                     zorder=10)
    else:
        ax_left.plot(steps, total, color=COLORS["red"], linewidth=3.5, alpha=1.0,
                     label="loss_total  (= jepa + 0.1*mt + drift)  [OPTIMIZER TARGET]",
                     zorder=10)

    # Right axis — drift raw + smoothed (1000× smaller scale)
    if max(drift) > 0:
        ax_right.plot(steps, drift, color=COLORS["green"], linewidth=0.5, alpha=0.20,
                      linestyle="--", zorder=2)
        sx, sm = _smoothed(drift)
        if sm is not None:
            ax_right.plot(sx, sm, color=COLORS["green"], linewidth=2.2, alpha=0.95,
                          linestyle="--",
                          label="loss_drift  (||θ−θ₀||²)  [right axis]", zorder=8)
        else:
            ax_right.plot(steps, drift, color=COLORS["green"], linewidth=1.8, alpha=0.85,
                          linestyle="--",
                          label="loss_drift  (||θ−θ₀||²)  [right axis]")
        ax_right.set_ylabel("Drift L2 (right axis, ~1000× smaller)",
                            color=COLORS["green"], fontsize=11)
        ax_right.tick_params(axis='y', labelcolor=COLORS["green"])
        ax_right.spines['right'].set_color(COLORS["green"])
    else:
        ax_right.set_ylabel("")
        ax_right.set_yticks([])

    ax_left.set_xlabel("Optimizer step", fontsize=11)
    ax_left.set_ylabel("Loss (left: jepa / multi_task / total)", fontsize=11)
    # Major ticks every 0.1 (was implicit 0.2) + minor ticks every 0.05 so per-step
    # Δ in jepa/total is eyeball-readable. Grid both: major solid, minor dotted.
    ax_left.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_left.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax_left.grid(axis='y', which='major', alpha=0.4)
    ax_left.grid(axis='y', which='minor', alpha=0.15, linestyle=':')
    ax_left.set_title(
        f"{title_prefix}Loss decomposition · {len(steps)} train steps\n"
        f"jepa={jepa[-1]:.4f}  mt={mt[-1]:.4f}  total={total[-1]:.4f}",
        fontsize=11)

    # Combined legend (left + right axes)
    h1, l1 = ax_left.get_legend_handles_labels()
    h2, l2 = ax_right.get_legend_handles_labels()
    ax_left.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9, framealpha=0.92)

    save_fig(fig, str(output_dir / f"{file_prefix}_loss_decomposition"))


def plot_block_drift_heatmap(drift_history: list, output_dir, title_prefix: str = "",
                              file_prefix: str = "m09") -> None:
    """Per-block weight-drift diagnostic — catches "frozen-in-practice" symptoms.

    Each row of `drift_history` is one validation checkpoint:
        {"step": int, "rel_l2_per_block": [float, ...]}
    where `rel_l2_per_block[i]` is the relative L2 norm
        ||θ_block_i_now − θ_block_i_init|| / ||θ_block_i_init||

    Renders TWO panels:
      - Top: heatmap (steps × blocks) on log color scale → spot uniform drift
        across blocks (the iter13 v5+v6+v7 pathology) vs healthy gradient where
        trainable blocks have visibly larger drift than frozen ones.
      - Bottom: per-block trajectory lines colored frozen vs trainable per
        `freeze_below` config. Healthy run: trainable lines climb to ~1e-3;
        frozen lines stay at 0. Stuck run: all lines bunch at ~1e-5.

    The whole point of this plot is to FAIL LOUD if pretrain isn't moving the
    encoder — visible in 1 min instead of waiting 5 hours for downstream metrics
    to confirm. Iter13 audit (2026-05-04) documented uniform ~1e-5 drift across
    all 48 blocks as proof the encoder didn't move.

    Args:
        drift_history: list of {"step", "rel_l2_per_block", optional "freeze_below"}
        output_dir:    where to save the PNG/PDF
        title_prefix:  optional prefix for the figure title
    """
    init_style()
    output_dir = Path(output_dir)
    if len(drift_history) < 1:
        print(f"  [plot] skip block_drift_heatmap: only {len(drift_history)} records")
        return

    steps = np.array([r["step"] for r in drift_history])
    drift = np.array([r["rel_l2_per_block"] for r in drift_history])  # (T, n_blocks)
    n_blocks = drift.shape[1]
    # Read freeze_below from the LAST record (it's a constant per-run, but
    # tolerate it being missing — diagnostic should still render).
    freeze_below = drift_history[-1].get("freeze_below", 0)

    fig, (ax_h, ax_t) = plt.subplots(2, 1, figsize=(12, 10),
                                      gridspec_kw={"height_ratios": [1, 1]})

    # ── Panel 1 — heatmap (log color scale) ─────────────────────────────
    # Log scale because healthy drift varies 10⁻⁵ (frozen) to 10⁻² (trainable).
    # Linear color would compress everything into one bin and hide the signal.
    drift_safe = np.maximum(drift, 1e-12)   # log10 needs strictly positive
    im = ax_h.imshow(np.log10(drift_safe.T), aspect="auto", origin="lower",
                     cmap="viridis", interpolation="nearest",
                     extent=[steps[0], steps[-1], -0.5, n_blocks - 0.5])
    cbar = fig.colorbar(im, ax=ax_h, pad=0.02)
    cbar.set_label("log₁₀(rel L2 drift)", fontsize=10)
    ax_h.set_ylabel("Block index (0=earliest, 47=top)", fontsize=11)
    ax_h.set_title(f"{title_prefix}Per-block weight drift vs Meta init  "
                   f"(rows=blocks, cols=val checkpoints)", fontsize=11)
    if freeze_below > 0:
        # Horizontal line marking the frozen/trainable boundary
        ax_h.axhline(freeze_below - 0.5, color="red", linestyle="--", linewidth=1.5,
                     alpha=0.7, label=f"freeze_below={freeze_below}")
        ax_h.legend(loc="upper right", fontsize=9, framealpha=0.92)

    # ── Panel 2 — per-block trajectory lines ────────────────────────────
    # Color frozen blocks blue, trainable blocks red, with low alpha so the
    # band-vs-band separation is the visual story.
    for i in range(n_blocks):
        is_frozen = i < freeze_below
        color = COLORS["blue"] if is_frozen else COLORS["red"]
        alpha = 0.25 if is_frozen else 0.5
        ax_t.plot(steps, drift[:, i], color=color, linewidth=1.0, alpha=alpha)
    # Mean of frozen vs trainable on top, thick
    if freeze_below > 0 and freeze_below < n_blocks:
        ax_t.plot(steps, drift[:, :freeze_below].mean(axis=1),
                  color=COLORS["blue"], linewidth=3.0,
                  label=f"frozen mean (blocks 0–{freeze_below-1})")
        ax_t.plot(steps, drift[:, freeze_below:].mean(axis=1),
                  color=COLORS["red"], linewidth=3.0,
                  label=f"trainable mean (blocks {freeze_below}–{n_blocks-1})")
    else:
        ax_t.plot(steps, drift.mean(axis=1),
                  color=COLORS["red"], linewidth=3.0,
                  label="all-block mean")
    # iter13 v12 (2026-05-06): five-zone reference bands matching /tmp/drift_table.py
    # verdict() — gives a 1-glance answer to "is the encoder moving healthily?"
    #   < 1e-5         STUCK         (v7 ceiling)
    #   1e-5 .. 1e-4   BORDERLINE    (early warmup)
    #   1e-4 .. 1e-2   HEALTHY       (active fine-tuning — the desirable zone)
    #   1e-2 .. 1e-1   AGGRESSIVE    (1-10 % rel L2; proceed with caution)
    #   >= 1e-1        CATASTROPHIC  (>=10 % rel L2; likely forgetting)
    ax_t.axhspan(1e-12, 1e-5, color="red",    alpha=0.07, zorder=0)
    ax_t.axhspan(1e-5,  1e-4, color="orange", alpha=0.07, zorder=0)
    ax_t.axhspan(1e-4,  1e-2, color="green",  alpha=0.07, zorder=0)
    ax_t.axhspan(1e-2,  1e-1, color="gold",   alpha=0.10, zorder=0)
    ax_t.axhspan(1e-1,  1.0,  color="red",    alpha=0.15, zorder=0)
    for y, name, c in [(1e-5, "STUCK",        "red"),
                       (1e-4, "BORDERLINE",   "orange"),
                       (1e-2, "HEALTHY",      "green"),
                       (1e-1, "AGGRESSIVE",   "gold"),
                       (1.0,  "CATASTROPHIC", "red")]:
        ax_t.axhline(y, color=c, linestyle=":", linewidth=1.0, alpha=0.6)
        ax_t.text(steps[0], y * 1.1, f"  {name} (>= {y:.0e})",
                  color=c, fontsize=8, va="bottom")
    ax_t.set_yscale("log")
    ax_t.set_ylim(1e-7, 1.0)                   # cap to keep zone bands visible
    ax_t.set_xlabel("Optimizer step", fontsize=11)
    ax_t.set_ylabel("rel L2 drift  (||Δθ_block||/||θ_block_init||)", fontsize=11)
    ax_t.grid(True, which="both", alpha=0.25)
    ax_t.legend(loc="best", fontsize=9, framealpha=0.92)

    save_fig(fig, str(output_dir / f"{file_prefix}_block_drift"))


def plot_val_loss_with_kill_switch_overlay(probe_history: list, output_dir,
                                            best_state: dict, kill_state: dict,
                                            file_prefix: str = "m09",
                                            title_prefix: str = "") -> None:
    """val_jepa_loss curve with best-marker + kill-switch annotation overlay.

    iter13 v13 C3-fix (2026-05-07): moved here from m09a1_pretrain_encoder.py:_render_m09a_probe_plots
    so m09c can reuse the same overlay (was m09a-only). Skips silently when
    `probe_history` empty OR no record has `val_jepa_loss`.

    Args:
        best_state: {"step": int, "probe_top1": float, "val_loss_at_best": float}
            (compatible with m09a + m09c best_state schemas)
        kill_state: {"triggered": bool, "reason": str}
    """
    init_style()
    if not probe_history:
        return
    recs = [r for r in probe_history if "val_jepa_loss" in r]
    if not recs:
        return

    steps = [r.get("step", r.get("global_step", 0)) for r in recs]
    val_losses = [r["val_jepa_loss"] for r in recs]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, val_losses, "o-", color=COLORS["blue"], linewidth=2.5,
            markersize=6, label="val_jepa")
    best_step = best_state.get("step", best_state.get("global_step", -1))
    if best_step >= 0:
        # Best is tracked by probe_top1; horizontal line marks val_loss AT best step.
        best_top1 = best_state.get("probe_top1", best_state.get("top1", -1.0))
        best_at_step_val = best_state.get("val_loss_at_best", best_state.get("val_loss", 0.0))
        ax.axhline(best_at_step_val, color=COLORS["green"], linestyle=":",
                   linewidth=1.5, alpha=0.7,
                   label=f"best top1={best_top1:.4f} @ step {best_step}")
    if kill_state.get("triggered"):
        ax.axvline(steps[-1], color=COLORS["red"], linestyle="--", linewidth=1.5,
                   label=f"early-stop: {kill_state.get('reason', '?')}")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("val_jepa loss")
    ax.set_title(f"{title_prefix}Validation JEPA loss + kill-switch state")
    ax.legend(loc="best")
    save_fig(fig, str(output_dir / f"{file_prefix}_val_loss_jepa"))


def plot_probe_trajectory_trio(probe_history: list, output_dir, title_prefix: str = "",
                                file_prefix: str = "m09") -> None:
    """3-panel trajectory: top-1 / motion-cos / future-L1 vs optimizer step.

    Mirrors iter12 probe_trajectory.png layout. Backward-compat: filters
    records that have all three trio keys, so a re-run over a mixed-schema
    jsonl renders only post-cutover records.

    Iter13 design (plan_code_dev.md §4): trajectory uses VAL split,
    paper-final m06d uses TEST split — numbers may differ; this is by design.
    """
    init_style()
    output_dir = Path(output_dir)
    recs = [r for r in probe_history
            if "probe_top1" in r and "motion_cos" in r and "future_l1" in r]
    # iter13 (2026-05-05): render from the 1st checkpoint. Was `< 2` (skipped
    # the first val cycle's plot, leaving the user wondering where it was). With
    # 1 record we plot a single marker per panel — degenerate trajectory but
    # the val-vs-step axis is still established and subsequent val cycles
    # extend the line in place. Empty (0) still skips with a clear log line.
    if len(recs) < 1:
        print("  [plot] skip probe_trajectory_trio: no trio records yet")
        return
    single_point = (len(recs) == 1)

    # iter13 v13 (2026-05-07): figsize height bumped 11 → 14; gridspec hspace
    # increased so each panel has room for its own caption between it and the
    # next panel (per-panel captions, not one combined block at the bottom).
    fig, axes = plt.subplots(3, 1, figsize=(7, 14), sharex=True,
                             gridspec_kw={"hspace": 0.55})
    steps = [r["step"] for r in recs]

    # iter13 v12+ (2026-05-06): each panel carries its optimization direction so
    # the reader knows whether ↑ trajectory is good (top-1, motion_cos) or bad (future-L1).
    # iter13 v13 (2026-05-07): per-panel caption text moved INTO the panel tuple
    # so each is anchored under its own axes (not in a combined block).
    panels = [
        ("Top-1 accuracy (action probe)",     "probe_top1", COLORS["green"],  "", "higher",
         "AttentiveClassifier accuracy on K-class motion-flow probe at each val cycle.\n"
         "Should TREND UP as the encoder learns to separate motion classes."),
        ("Intra−Inter cosine (motion sep.)",  "motion_cos", COLORS["blue"],   "", "higher",
         "Same-class minus different-class cosine separation.\n"
         "> 0 ⇒ semantic clustering;  near 0 ⇒ encoder doesn't separate motion classes."),
        ("Future-frame L1 (lower=better)",    "future_l1",  COLORS["orange"], "", "lower",
         "V-JEPA predictor's L1 on masked next-frame tokens — the JEPA objective.\n"
         "Should TREND DOWN.  If it climbs ⇒ encoder forgetting predictive structure."),
    ]
    for ax, (title, key, color, unit, direction, caption) in zip(axes, panels):
        y = [r[key] for r in recs]
        # Larger marker when single-point so the dot is visible without a line
        markersize = 12 if single_point else 4
        ax.plot(steps, y, marker="o", color=color, linewidth=2.0,
                markersize=markersize)
        ax.set_ylabel(f"{title}{(' (' + unit + ')') if unit else ''}", fontsize=10)
        ax.grid(True, alpha=0.3)
        # Direction badge ↑ higher = better / ↓ lower = better, top-left of panel.
        if direction == "higher":
            ax.text(0.02, 0.97, "↑ higher = better",
                    transform=ax.transAxes, fontsize=9, fontweight="bold",
                    color="#2E7D32", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#E8F5E9",
                              edgecolor="#2E7D32", linewidth=0.8, alpha=0.85))
        elif direction == "lower":
            ax.text(0.02, 0.97, "↓ lower = better",
                    transform=ax.transAxes, fontsize=9, fontweight="bold",
                    color="#E65100", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF3E0",
                              edgecolor="#E65100", linewidth=0.8, alpha=0.85))
        ax.annotate(f"{'val' if single_point else 'end'}={y[-1]:.4f}",
                    xy=(steps[-1], y[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=9, color=color, va="center", weight="bold")
        if single_point:
            # Pad x-axis so the single dot doesn't sit at the right edge
            ax.set_xlim(left=max(0, steps[0] - 5), right=steps[0] + 50)

        # Per-panel caption — anchored under THIS axes via transAxes so it
        # stays with its own panel. Lives in the gridspec_kw hspace gap.
        # iter13 v13 (2026-05-07): pushed to fontweight=bold + fontsize 10.0
        # + edgecolor #555 (darker border) — earlier "medium" looked washed out
        # against the line-plot panels (more whitespace + thinner data lines
        # than the bar-chart sibling, so the eye reads caption text as fainter).
        ax.text(0.5, -0.30, caption,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10.0, color="#000", fontweight="bold",
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF",
                          edgecolor="#555", linewidth=1.0))

    n_clips = recs[0].get("n_probe_clips", "?")
    axes[0].set_title(
        f"{title_prefix}Probe trajectory — encoder evolution across SSL training\n"
        f"(VAL split · N={n_clips} probe-clips · per-step diagnostic, NOT paper-final eval)",
        fontsize=11)
    axes[-1].set_xlabel("Optimizer step", fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_fig(fig, str(output_dir / f"{file_prefix}_probe_trajectory_trio"))


def compute_block_drift(student, init_params: dict) -> list:
    """Per-block relative L2 norm of (current θ − init θ).

    Iterates `student.named_parameters()`, groups by block index parsed from
    the parameter name (`blocks.{i}.*`). Returns a list of length `n_blocks`
    with mean rel-L2 per block. Catch-all bucket (patch_embed, norms outside
    blocks, etc.) is collapsed into block index 0 to keep the array shape
    simple — it's a diagnostic, not a paper number.

    `init_params` is the dict snapshotted at build_model time
    (m09a1_pretrain_encoder.py:412). On CPU. Same keys as student.state_dict().

    Cost: ~0.5 s on ViT-G with 588 named params. Negligible vs val cycle.
    """
    import re
    bucket: dict = {}   # block_idx → list of rel_l2 scalars
    for name, p_cur in student.named_parameters():
        if name not in init_params:
            continue
        p_init = init_params[name]
        if p_cur.shape != p_init.shape:
            continue
        # Cast to fp32 on CPU for stable norm; tiny tensors so cost is trivial.
        p_cur_f = p_cur.detach().float().cpu()
        p_init_f = p_init.detach().float().cpu()
        diff_l2 = (p_cur_f - p_init_f).norm().item()
        init_l2 = p_init_f.norm().item() + 1e-12
        rel = diff_l2 / init_l2
        m = re.search(r"blocks\.(\d+)\.", name)
        idx = int(m.group(1)) if m else 0   # catch-all → block 0
        bucket.setdefault(idx, []).append(rel)
    if not bucket:
        return []
    n_blocks = max(bucket.keys()) + 1
    return [float(np.mean(bucket.get(i, [0.0]))) for i in range(n_blocks)]

