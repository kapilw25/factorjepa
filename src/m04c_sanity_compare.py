"""CPU-only VLM sanity comparison: 4-metric table + 2x2 dashboard plot.
    python -u src/m04c_sanity_compare.py 2>&1 | tee logs/m04c_sanity_compare.log
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── paths (same pattern as m08_plot.py) ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils.progress import make_pbar
from utils.config import OUTPUTS_DIR, OUTPUTS_SANITY_DIR, TAG_TAXONOMY_JSON

MODELS = ["qwen", "videollama", "llava"]
MODEL_LABELS = {"qwen": "Qwen3-VL", "videollama": "VideoLLaMA3", "llava": "LLaVA-NeXT"}
VLM_COLORS = {"qwen": "#2196F3", "videollama": "#4CAF50", "llava": "#FF9800"}

# Scene palette from m08_plot.py
SCENE_COLORS = {
    "market": "#e41a1c", "junction": "#4daf4a", "residential_lane": "#984ea3",
    "promenade": "#377eb8", "transit": "#999999", "temple_tourist": "#ff7f00",
    "highway": "#ffff33", "alley": "#a65628", "commercial": "#17becf",
    "construction": "#f781bf", "unknown": "#666666",
}

CONFIDENCE_FIELDS = [
    "confidence_scene_type", "confidence_time_of_day", "confidence_weather",
    "confidence_crowd_density", "confidence_traffic_density", "confidence_road_layout",
    "confidence_road_surface", "confidence_infrastructure_quality",
    "confidence_notable_objects", "confidence_vegetation", "confidence_lighting",
]


# ── load taxonomy ────────────────────────────────────────────────────────────
with open(TAG_TAXONOMY_JSON) as f:
    taxonomy = json.load(f)
VALID_SCENE_TYPES = set(taxonomy["scene_type"]["values"])
VALID_OBJECTS = set(taxonomy["notable_objects"]["values"])


def is_valid_parse(clip: dict) -> bool:
    """A clip parsed correctly if scene_type is a single valid enum value (no pipe-delimited dumps)."""
    st = clip.get("scene_type", "")
    return "|" not in str(st) and str(st) in VALID_SCENE_TYPES


def compute_metrics(tags: list) -> dict:
    """Compute 4 quality metrics for a single VLM's sanity output."""
    n = len(tags)

    # 1. Parse rate
    valid_clips = [c for c in tags if is_valid_parse(c)]
    parse_rate = len(valid_clips) / n * 100 if n > 0 else 0.0

    # 2. Scene diversity (only from valid-parsed clips)
    scene_types = [c["scene_type"] for c in valid_clips]
    unique_scenes = set(scene_types)
    scene_counts = {}
    for st in scene_types:
        scene_counts[st] = scene_counts.get(st, 0) + 1

    # 3. Confidence calibration (exclude clips where all confidences are 0.0 — garbage)
    all_confs = []
    for c in tags:
        clip_confs = [c.get(f, 0.0) for f in CONFIDENCE_FIELDS if f in c]
        if clip_confs and any(v > 0.0 for v in clip_confs):
            all_confs.extend(clip_confs)
    conf_mean = float(np.mean(all_confs)) if all_confs else 0.0
    conf_std = float(np.std(all_confs)) if all_confs else 0.0

    # 4. Objects variety — split into on-taxonomy vs off-taxonomy (hallucinated)
    all_objects = set()
    for c in tags:
        objs = c.get("notable_objects", [])
        if isinstance(objs, list):
            all_objects.update(objs)
    # Filter out "subset of:" garbage strings
    all_objects = {o for o in all_objects if "subset of:" not in str(o)}
    on_taxonomy = all_objects & VALID_OBJECTS
    off_taxonomy = all_objects - VALID_OBJECTS

    return {
        "parse_rate": parse_rate,
        "valid_clips": len(valid_clips),
        "total_clips": n,
        "scene_diversity": len(unique_scenes),
        "scene_counts": scene_counts,
        "conf_mean": conf_mean,
        "conf_std": conf_std,
        "conf_values": all_confs,
        "objects_variety": len(all_objects),
        "objects_on_taxonomy": len(on_taxonomy),
        "objects_off_taxonomy": len(off_taxonomy),
        "objects_on_set": on_taxonomy,
        "objects_off_set": off_taxonomy,
    }


def print_table(results: dict):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("VLM SANITY COMPARISON (20 clips each)")
    print("=" * 70)

    header = f"{'Metric':<30} {'Qwen3-VL':>12} {'VideoLLaMA3':>12} {'LLaVA-NeXT':>12}"
    print(header)
    print("-" * 70)

    rows = [
        ("Parse rate", lambda r: f"{r['parse_rate']:.0f}% ({r['valid_clips']}/{r['total_clips']})"),
        ("Scene diversity", lambda r: f"{r['scene_diversity']} types"),
        ("Confidence mean", lambda r: f"{r['conf_mean']:.3f}"),
        ("Confidence std", lambda r: f"{r['conf_std']:.3f}"),
        ("Objects on-taxonomy", lambda r: f"{r['objects_on_taxonomy']}"),
        ("Objects off-taxonomy", lambda r: f"{r['objects_off_taxonomy']} halluc."),
    ]

    for label, fmt_fn in rows:
        vals = []
        for m in MODELS:
            vals.append(fmt_fn(results[m]))
        print(f"{label:<30} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    print("-" * 70)

    # Scene breakdown
    print("\nScene type breakdown:")
    all_scenes = set()
    for m in MODELS:
        all_scenes.update(results[m]["scene_counts"].keys())
    for scene in sorted(all_scenes):
        counts = []
        for m in MODELS:
            c = results[m]["scene_counts"].get(scene, 0)
            counts.append(f"{c:>3}")
        print(f"  {scene:<25} {counts[0]:>12} {counts[1]:>12} {counts[2]:>12}")
    print()


def plot_dashboard(results: dict, output_dir: Path):
    """Generate 2x2 dashboard plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("VLM Sanity Comparison (20 clips each)", fontsize=16, fontweight="bold", y=0.98)

    # ── Top-left: Parse rate bar chart ───────────────────────────────────────
    ax = axes[0, 0]
    x_pos = np.arange(len(MODELS))
    rates = [results[m]["parse_rate"] for m in MODELS]
    colors = [VLM_COLORS[m] for m in MODELS]
    bars = ax.bar(x_pos, rates, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_ylabel("Parse Rate (%)", fontsize=11)
    ax.set_title("JSON Parse Rate", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── Top-right: Scene type distribution (stacked horizontal bar) ──────────
    ax = axes[0, 1]
    all_scenes = sorted({s for m in MODELS for s in results[m]["scene_counts"]})
    y_pos = np.arange(len(MODELS))
    left_offsets = np.zeros(len(MODELS))
    for scene in all_scenes:
        widths = [results[m]["scene_counts"].get(scene, 0) for m in MODELS]
        color = SCENE_COLORS.get(scene, SCENE_COLORS["unknown"])
        ax.barh(y_pos, widths, left=left_offsets, height=0.5, color=color,
                edgecolor="white", linewidth=0.5, label=scene)
        left_offsets += np.array(widths)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_xlabel("Number of clips", fontsize=11)
    ax.set_title("Scene Type Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    # ── Bottom-left: Confidence distributions (box plot) ─────────────────────
    ax = axes[1, 0]
    conf_data = []
    for m in MODELS:
        vals = results[m]["conf_values"]
        conf_data.append(vals if vals else [0.0])
    bp = ax.boxplot(conf_data, patch_artist=True, widths=0.4,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(VLM_COLORS[m])
        patch.set_alpha(0.7)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_ylabel("Confidence Score", fontsize=11)
    ax.set_title("Confidence Distributions", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)

    # ── Bottom-right: Objects butterfly chart (on-taxonomy + / off-taxonomy -) ─
    ax = axes[1, 1]
    on_counts = [results[m]["objects_on_taxonomy"] for m in MODELS]
    off_counts = [results[m]["objects_off_taxonomy"] for m in MODELS]
    bar_w = 0.5
    # Positive bars: on-taxonomy
    bars_on = ax.bar(x_pos, on_counts, color=colors, width=bar_w,
                     edgecolor="white", linewidth=0.8, label="On-taxonomy")
    # Negative bars: off-taxonomy (hallucinated)
    bars_off = ax.bar(x_pos, [-c for c in off_counts], color=colors, width=bar_w,
                      edgecolor="white", linewidth=0.8, alpha=0.45, label="Off-taxonomy",
                      hatch="//")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_ylabel("Unique Objects", fontsize=11)
    ax.set_title("Notable Objects: On vs Off Taxonomy", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    # Labels on positive bars
    for bar, count in zip(bars_on, on_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")
    # Labels on negative bars
    for bar, count in zip(bars_off, off_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() - 0.3,
                    str(count), ha="center", va="top", fontsize=10, fontweight="bold",
                    color="#d32f2f")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = output_dir / "m04c_sanity_compare.png"
    pdf_path = output_dir / "m04c_sanity_compare.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    # Load all 3 sanity tag files
    all_tags = {}
    for model in MODELS:
        path = OUTPUTS_SANITY_DIR / f"tags_sanity_{model}.json"
        if not path.exists():
            print(f"ERROR: {path} not found. Run: python -u src/m04_vlm_tag.py --model {model} --SANITY")
            sys.exit(1)
        with open(path) as f:
            all_tags[model] = json.load(f)
        print(f"Loaded {path.name}: {len(all_tags[model])} clips")

    # Compute metrics
    results = {}
    pbar = make_pbar(total=len(MODELS), desc="m04c_compare", unit="vlm")
    for model in MODELS:
        results[model] = compute_metrics(all_tags[model])
        pbar.update(1)
    pbar.close()

    # Print comparison table
    print_table(results)

    # Generate dashboard plot
    plot_dashboard(results, OUTPUTS_SANITY_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
