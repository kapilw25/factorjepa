"""Bake-off: 5-criterion weighted VLM selection + 2x2 diagnostic dashboard.
    python -u src/m04b_vlm_select.py 2>&1 | tee logs/m04b_vlm_select.log
    python -u src/m04b_vlm_select.py --SANITY 2>&1 | tee logs/m04b_sanity_select.log
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    BAKEOFF_DIR, OUTPUTS_DIR, OUTPUTS_SANITY_DIR, TAG_TAXONOMY_JSON, VLM_MODELS,
)

# ── Config ────────────────────────────────────────────────────────────────

WEIGHTS = {
    "json_parse":       0.30,  # % clips with valid structured JSON
    "agreement":        0.25,  # cross-VLM majority vote agreement
    "speed":            0.20,  # clips/sec throughput
    "taxonomy":         0.15,  # % values within allowed categories
    "confidence_cal":   0.10,  # correlation(confidence, agreement)
}


# ── Load taxonomy ─────────────────────────────────────────────────────────

def load_taxonomy() -> dict:
    with open(TAG_TAXONOMY_JSON, 'r') as f:
        taxonomy = json.load(f)
    return {k: v for k, v in taxonomy.items() if not k.startswith("_")}

TAXONOMY = load_taxonomy()
TAG_FIELDS = list(TAXONOMY.keys())
VALID_SCENE_TYPES = set(TAXONOMY["scene_type"]["values"])
VALID_OBJECTS = set(TAXONOMY["notable_objects"]["values"])

# Dashboard constants (shared palette with m04c/m08)
MODEL_LABELS = {"qwen": "Qwen3-VL", "videollama": "VideoLLaMA3", "llava": "LLaVA-NeXT"}
VLM_COLORS = {"qwen": "#2196F3", "videollama": "#4CAF50", "llava": "#FF9800"}
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


# ── Load bake-off tags ────────────────────────────────────────────────────

def load_bakeoff_tags(sanity: bool = False) -> dict:
    """Load tag JSON files for each VLM. Returns: {model_name: [tag_dict, ...]}"""
    loaded = {}
    if sanity:
        tag_dir = OUTPUTS_SANITY_DIR
        pattern = "tags_sanity_{}.json"
    else:
        tag_dir = BAKEOFF_DIR
        pattern = "tags_{}.json"

    for model_name in VLM_MODELS:
        path = tag_dir / pattern.format(model_name)
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {model_name}")
            continue
        with open(path) as f:
            tags = json.load(f)
        if not isinstance(tags, list) or len(tags) == 0:
            print(f"WARNING: {path} is empty or invalid, skipping {model_name}")
            continue
        loaded[model_name] = tags
        print(f"Loaded {model_name}: {len(tags):,} clips from {path.name}")

    if len(loaded) < 2:
        print(f"ERROR: Need at least 2 VLM tag files in {tag_dir}/")
        print(f"Found: {list(loaded.keys())}")
        sys.exit(1)

    return loaded


# ── Criterion 1: JSON Parse Success % ────────────────────────────────────

def compute_json_parse_rate(tags: list) -> float:
    """% of clips that have at least 1 valid tag field (not all defaults)."""
    defaults = {field: spec["default"] for field, spec in TAXONOMY.items()}
    valid = 0
    for t in tags:
        # A clip is "parsed" if at least one tag field differs from default
        for field in TAG_FIELDS:
            if field in t and t[field] != defaults[field]:
                valid += 1
                break
    return valid / len(tags) if tags else 0.0


# ── Criterion 2: Cross-VLM Agreement % ───────────────────────────────────

def build_majority_vote(all_tags: dict) -> list:
    """
    For each clip index, compute majority vote across VLMs for each tag field.
    Returns: [{field: majority_value, ...}, ...] for N clips.
    """
    model_names = list(all_tags.keys())
    n_clips = min(len(all_tags[m]) for m in model_names)

    majority = []
    for i in range(n_clips):
        clip_vote = {}
        for field in TAG_FIELDS:
            values = []
            for m in model_names:
                val = all_tags[m][i].get(field)
                if val is not None:
                    # For multi-value fields, use frozenset for comparison
                    if isinstance(val, list):
                        values.append(tuple(sorted(val)))
                    else:
                        values.append(val)
            if values:
                counter = Counter(values)
                winner = counter.most_common(1)[0][0]
                # Convert back to list for multi-value
                if isinstance(winner, tuple):
                    clip_vote[field] = list(winner)
                else:
                    clip_vote[field] = winner
            else:
                clip_vote[field] = TAXONOMY[field]["default"]
        majority.append(clip_vote)

    return majority


def compute_agreement(tags: list, majority: list) -> float:
    """% of (clip, field) pairs where this VLM matches majority vote."""
    n = min(len(tags), len(majority))
    total = n * len(TAG_FIELDS)
    if total == 0:
        return 0.0

    matches = 0
    for i in range(n):
        for field in TAG_FIELDS:
            vlm_val = tags[i].get(field)
            maj_val = majority[i].get(field)
            if vlm_val is None or maj_val is None:
                continue
            if isinstance(vlm_val, list) and isinstance(maj_val, list):
                if sorted(vlm_val) == sorted(maj_val):
                    matches += 1
            elif vlm_val == maj_val:
                matches += 1

    return matches / total


# ── Criterion 3: Speed (clips/sec) ───────────────────────────────────────

def compute_speed(tags: list) -> float:
    """Extract speed from _tagged_at timestamps (first vs last clip)."""
    if len(tags) < 2:
        return 0.0

    from datetime import datetime

    timestamps = []
    for t in tags:
        ts = t.get("_tagged_at")
        if ts:
            try:
                timestamps.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except (ValueError, TypeError):
                continue

    if len(timestamps) < 2:
        return 0.0

    timestamps.sort()
    duration_sec = (timestamps[-1] - timestamps[0]).total_seconds()
    if duration_sec <= 0:
        return 0.0

    return len(timestamps) / duration_sec


# ── Criterion 4: Taxonomy Compliance % ────────────────────────────────────

def compute_taxonomy_compliance(tags: list) -> float:
    """% of (clip, field) pairs where value is within allowed categories."""
    total = 0
    compliant = 0

    for t in tags:
        for field, spec in TAXONOMY.items():
            val = t.get(field)
            if val is None:
                total += 1
                continue

            allowed = set(spec["values"])
            total += 1

            if spec["type"] == "multi":
                if isinstance(val, list) and all(v in allowed for v in val):
                    compliant += 1
            else:
                if val in allowed:
                    compliant += 1

    return compliant / total if total > 0 else 0.0


# ── Criterion 5: Confidence Calibration ───────────────────────────────────

def compute_confidence_calibration(tags: list, majority: list) -> float:
    """
    Correlation between per-field confidence and agreement with majority.
    Higher = better calibrated (confident when correct, uncertain when wrong).
    Returns Pearson r in [0, 1] (clamped).
    """
    confidences = []
    agreements = []

    n = min(len(tags), len(majority))
    for i in range(n):
        for field in TAG_FIELDS:
            conf_key = f"confidence_{field}"
            conf = tags[i].get(conf_key)
            if conf is None or not isinstance(conf, (int, float)):
                continue

            vlm_val = tags[i].get(field)
            maj_val = majority[i].get(field)

            if vlm_val is None or maj_val is None:
                continue

            # Agreement: 1 if matches majority, 0 otherwise
            if isinstance(vlm_val, list) and isinstance(maj_val, list):
                agree = 1.0 if sorted(vlm_val) == sorted(maj_val) else 0.0
            else:
                agree = 1.0 if vlm_val == maj_val else 0.0

            confidences.append(float(conf))
            agreements.append(agree)

    if len(confidences) < 10:
        return 0.5  # not enough data, neutral score

    # Pearson correlation
    n = len(confidences)
    mean_c = sum(confidences) / n
    mean_a = sum(agreements) / n
    cov = sum((c - mean_c) * (a - mean_a) for c, a in zip(confidences, agreements)) / n
    std_c = (sum((c - mean_c) ** 2 for c in confidences) / n) ** 0.5
    std_a = (sum((a - mean_a) ** 2 for a in agreements) / n) ** 0.5

    if std_c < 1e-9 or std_a < 1e-9:
        return 0.5  # no variance, neutral

    r = cov / (std_c * std_a)
    return max(0.0, min(1.0, (r + 1) / 2))  # map [-1,1] → [0,1]


# ── Weighted Score ────────────────────────────────────────────────────────

def compute_weighted_score(scores: dict) -> float:
    """Compute weighted total from 5 criteria (all normalized to [0,1])."""
    total = 0.0
    for criterion, weight in WEIGHTS.items():
        total += weight * scores.get(criterion, 0.0)
    return total


def normalize_speed(speed_scores: dict) -> dict:
    """Normalize speed to [0,1] relative to fastest VLM."""
    max_speed = max(speed_scores.values()) if speed_scores else 1.0
    if max_speed <= 0:
        return {m: 0.5 for m in speed_scores}
    return {m: v / max_speed for m, v in speed_scores.items()}


# ── Plot ──────────────────────────────────────────────────────────────────

def generate_comparison_plot(results: dict, winner: str, output_dir: Path = BAKEOFF_DIR):
    """Generate radar + bar chart comparing VLMs (.png + .pdf)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib/numpy not available, skipping plots")
        return

    models = list(results.keys())
    criteria = list(WEIGHTS.keys())
    criteria_labels = ["JSON Parse", "Agreement", "Speed", "Taxonomy", "Conf. Calibration"]

    # ── Bar chart ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: per-criterion grouped bar chart
    ax = axes[0]
    x = np.arange(len(criteria))
    width = 0.8 / len(models)
    colors = ["#00acc1", "#43a047", "#e53935"]

    for j, model in enumerate(models):
        scores = [results[model]["scores"][c] for c in criteria]
        bars = ax.bar(x + j * width, scores, width, label=model, color=colors[j % len(colors)])
        # Mark winner
        if model == winner:
            for bar in bars:
                bar.set_edgecolor("gold")
                bar.set_linewidth(2)

    ax.set_xlabel("Criterion")
    ax.set_ylabel("Score (0-1)")
    ax.set_title("VLM Bake-off: Per-Criterion Scores")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(criteria_labels, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Right: weighted total bar chart
    ax2 = axes[1]
    totals = [results[m]["weighted_total"] for m in models]
    bar_colors = ["gold" if m == winner else colors[i % len(colors)] for i, m in enumerate(models)]
    bars = ax2.bar(models, totals, color=bar_colors, edgecolor="black", linewidth=1.5)

    for bar, total in zip(bars, totals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{total:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax2.set_ylabel("Weighted Total")
    ax2.set_title(f"VLM Bake-off Winner: {winner}")
    ax2.set_ylim(0, max(totals) * 1.15)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "m04b_vlm_comparison.png"
    pdf_path = output_dir / "m04b_vlm_comparison.pdf"
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ── 2x2 Dashboard (same layout as m04c_sanity_compare) ───────────────────

def compute_dashboard_metrics(tags: list) -> dict:
    """Compute 4 visual-diagnostic metrics for one VLM's tags."""
    import numpy as np

    n = len(tags)

    # 1. Parse rate — scene_type is a single valid enum (no pipe-delimited dumps)
    valid_clips = [c for c in tags if "|" not in str(c.get("scene_type", ""))
                   and str(c.get("scene_type", "")) in VALID_SCENE_TYPES]
    parse_rate = len(valid_clips) / n * 100 if n > 0 else 0.0

    # 2. Scene diversity
    scene_types = [c["scene_type"] for c in valid_clips]
    scene_counts = {}
    for st in scene_types:
        scene_counts[st] = scene_counts.get(st, 0) + 1

    # 3. Confidence distribution (exclude all-zero garbage clips)
    all_confs = []
    for c in tags:
        clip_confs = [c.get(f, 0.0) for f in CONFIDENCE_FIELDS if f in c]
        if clip_confs and any(v > 0.0 for v in clip_confs):
            all_confs.extend(clip_confs)

    # 4. Objects: on-taxonomy vs off-taxonomy
    all_objects = set()
    for c in tags:
        objs = c.get("notable_objects", [])
        if isinstance(objs, list):
            all_objects.update(objs)
    all_objects = {o for o in all_objects if "subset of:" not in str(o)}
    on_taxonomy = all_objects & VALID_OBJECTS
    off_taxonomy = all_objects - VALID_OBJECTS

    return {
        "parse_rate": parse_rate,
        "valid_clips": len(valid_clips),
        "total_clips": n,
        "scene_counts": scene_counts,
        "conf_values": all_confs,
        "objects_on_taxonomy": len(on_taxonomy),
        "objects_off_taxonomy": len(off_taxonomy),
    }


def generate_dashboard_plot(all_tags: dict, model_names: list, output_dir: Path = BAKEOFF_DIR):
    """Generate 2x2 diagnostic dashboard (same layout as m04c_sanity_compare)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("WARNING: matplotlib/numpy not available, skipping dashboard")
        return

    # Compute dashboard metrics per VLM
    results = {m: compute_dashboard_metrics(all_tags[m]) for m in model_names}
    n_clips = len(all_tags[model_names[0]])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"VLM Bake-off Dashboard ({n_clips:,} clips each)",
                 fontsize=16, fontweight="bold", y=0.98)

    x_pos = np.arange(len(model_names))
    colors = [VLM_COLORS.get(m, "#888888") for m in model_names]
    labels = [MODEL_LABELS.get(m, m) for m in model_names]

    # ── Top-left: Parse rate ─────────────────────────────────────────────
    ax = axes[0, 0]
    rates = [results[m]["parse_rate"] for m in model_names]
    bars = ax.bar(x_pos, rates, color=colors, width=0.5, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Parse Rate (%)", fontsize=11)
    ax.set_title("JSON Parse Rate", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{rate:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # ── Top-right: Scene type distribution (stacked horizontal bar) ──────
    ax = axes[0, 1]
    all_scenes = sorted({s for m in model_names for s in results[m]["scene_counts"]})
    y_pos = np.arange(len(model_names))
    left_offsets = np.zeros(len(model_names))
    for scene in all_scenes:
        widths = [results[m]["scene_counts"].get(scene, 0) for m in model_names]
        color = SCENE_COLORS.get(scene, SCENE_COLORS["unknown"])
        ax.barh(y_pos, widths, left=left_offsets, height=0.5, color=color,
                edgecolor="white", linewidth=0.5, label=scene)
        left_offsets += np.array(widths)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Number of clips", fontsize=11)
    ax.set_title("Scene Type Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", ncol=2)

    # ── Bottom-left: Confidence distributions (box plot) ─────────────────
    ax = axes[1, 0]
    conf_data = [results[m]["conf_values"] or [0.0] for m in model_names]
    bp = ax.boxplot(conf_data, patch_artist=True, widths=0.4,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, m in zip(bp["boxes"], model_names):
        patch.set_facecolor(VLM_COLORS.get(m, "#888888"))
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Confidence Score", fontsize=11)
    ax.set_title("Confidence Distributions", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)

    # ── Bottom-right: Objects butterfly (on-taxonomy + / off-taxonomy -) ─
    ax = axes[1, 1]
    on_counts = [results[m]["objects_on_taxonomy"] for m in model_names]
    off_counts = [results[m]["objects_off_taxonomy"] for m in model_names]
    bar_w = 0.5
    bars_on = ax.bar(x_pos, on_counts, color=colors, width=bar_w,
                     edgecolor="white", linewidth=0.8, label="On-taxonomy")
    bars_off = ax.bar(x_pos, [-c for c in off_counts], color=colors, width=bar_w,
                      edgecolor="white", linewidth=0.8, alpha=0.45, label="Off-taxonomy",
                      hatch="//")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Unique Objects", fontsize=11)
    ax.set_title("Notable Objects: On vs Off Taxonomy", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=0.8)
    for bar, count in zip(bars_on, on_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, count in zip(bars_off, off_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() - 0.3,
                    str(count), ha="center", va="top", fontsize=10, fontweight="bold",
                    color="#d32f2f")
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = output_dir / "m04b_vlm_dashboard.png"
    pdf_path = output_dir / "m04b_vlm_dashboard.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="VLM bake-off selection")
    parser.add_argument("--SANITY", action="store_true",
                        help="Use SANITY tags (src/outputs/tags_sanity_*.json) instead of bakeoff")
    return parser.parse_args()


def main():
    args = parse_args()
    sanity = args.SANITY

    mode_label = "SANITY" if sanity else "BAKEOFF"
    output_dir = OUTPUTS_DIR if sanity else BAKEOFF_DIR
    output_json = output_dir / "m04b_vlm_comparison.json"

    print(f"VLM Bake-off Selection  (mode={mode_label})")
    print(f"Tag dir: {OUTPUTS_DIR if sanity else BAKEOFF_DIR}")
    print(f"Weights: {WEIGHTS}")
    print()

    # Load tags
    all_tags = load_bakeoff_tags(sanity=sanity)
    model_names = list(all_tags.keys())
    print(f"\nModels loaded: {model_names}")

    # Align clip counts (use minimum across all VLMs)
    min_clips = min(len(all_tags[m]) for m in model_names)
    for m in model_names:
        all_tags[m] = all_tags[m][:min_clips]
    print(f"Aligned to {min_clips:,} clips per VLM")

    # Build majority vote (cross-VLM consensus)
    majority = build_majority_vote(all_tags)
    print(f"Majority vote computed across {len(model_names)} VLMs")

    # Compute raw scores per VLM
    raw_speeds = {}
    results = {}

    for model in model_names:
        tags = all_tags[model]
        json_parse = compute_json_parse_rate(tags)
        agreement = compute_agreement(tags, majority)
        speed = compute_speed(tags)
        taxonomy = compute_taxonomy_compliance(tags)
        conf_cal = compute_confidence_calibration(tags, majority)

        raw_speeds[model] = speed
        results[model] = {
            "model_id": VLM_MODELS.get(model, model),
            "n_clips": len(tags),
            "raw_scores": {
                "json_parse": json_parse,
                "agreement": agreement,
                "speed_clips_per_sec": speed,
                "taxonomy": taxonomy,
                "confidence_cal": conf_cal,
            },
        }

    # Normalize speed to [0,1]
    norm_speeds = normalize_speed(raw_speeds)

    # Compute weighted totals
    for model in model_names:
        raw = results[model]["raw_scores"]
        scores = {
            "json_parse": raw["json_parse"],
            "agreement": raw["agreement"],
            "speed": norm_speeds[model],
            "taxonomy": raw["taxonomy"],
            "confidence_cal": raw["confidence_cal"],
        }
        results[model]["scores"] = scores
        results[model]["weighted_total"] = compute_weighted_score(scores)

    # Pick winner
    winner = max(model_names, key=lambda m: results[m]["weighted_total"])

    # Print results table
    print(f"\n{'='*70}")
    print(f"{'VLM ' + mode_label + ' RESULTS':^70}")
    print(f"{'='*70}")
    print(f"{'Criterion':<25} {'Weight':>6}  ", end="")
    for m in model_names:
        print(f"  {m:>12}", end="")
    print()
    print("-" * 70)

    for criterion, weight in WEIGHTS.items():
        label = criterion.replace("_", " ").title()
        print(f"{label:<25} {weight:>5.0%}  ", end="")
        for m in model_names:
            val = results[m]["scores"][criterion]
            marker = " *" if m == winner else "  "
            print(f"  {val:>10.3f}{marker}", end="")
        print()

    print("-" * 70)
    print(f"{'WEIGHTED TOTAL':<25} {'100%':>6}  ", end="")
    for m in model_names:
        val = results[m]["weighted_total"]
        marker = " *" if m == winner else "  "
        print(f"  {val:>10.3f}{marker}", end="")
    print()
    print(f"\n>>> WINNER: {winner} ({VLM_MODELS.get(winner, winner)}) <<<")

    # Save results
    output = {
        "mode": mode_label,
        "winner": winner,
        "winner_model_id": VLM_MODELS.get(winner, winner),
        "weights": WEIGHTS,
        "n_clips_compared": min_clips,
        "models": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_json}")

    # Generate plots
    generate_comparison_plot(results, winner, output_dir=output_dir)
    generate_dashboard_plot(all_tags, model_names, output_dir=output_dir)

    print(f"\n=== {mode_label} COMPLETE ===")
    print(f"Winner: {winner}")
    if sanity:
        print(f"Note: SANITY results on {min_clips} clips — use --BAKEOFF for statistical significance")
    else:
        print(f"Next: python -u src/m04_vlm_tag.py --model {winner} --FULL [--subset data/subset_10k.json]")


if __name__ == "__main__":
    main()
