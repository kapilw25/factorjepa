"""Path-based action-label derivation for m06d_* modules. CPU-only.
Mirrors m00b_fetch_durations.py extract_all_videos section schema.

USAGE (called by m06d_*.py — direct __main__ entry exists for self-test only):
    from utils.action_labels import (
        parse_action_from_clip_key, load_subset_with_labels,
        stratified_split, write_action_labels_json, load_action_labels,
        CLASS_NAMES_3CLASS, CLASS_NAMES_4CLASS,
    )

Self-test (CPU sanity):
    python -u src/utils/action_labels.py --eval-subset data/eval_10k.json
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.checkpoint import save_json_checkpoint, load_json_checkpoint


# ── Constants ────────────────────────────────────────────────────────

# Path activity (from clip_key) → semantic class. Must match m00b extract_all_videos buckets.
PATH_TO_CLASS_3CLASS: dict = {
    "walking":  "walking",
    "rain":     "walking",   # tier2 rain bucket = rainy walking tour
    "drive":    "driving",
    "drone":    "drone",
}

# Stable ID order — alphabetical → reproducible across runs.
CLASS_NAMES_3CLASS: list = ["driving", "drone", "walking"]                       # n=3
CLASS_NAMES_4CLASS: list = ["driving", "drone", "monument", "walking"]           # n=4

# scene_type override for 4-class mode — VLM tag value that triggers "monument".
HERITAGE_SCENE_TYPE: str = "heritage_tourist"


# ── Public API ───────────────────────────────────────────────────────

def parse_action_from_clip_key(clip_key, *, enable_monument, tags=None):
    """Derive 3- or 4-class action label from a clip_key (+ optional VLM tags).

    Args:
        clip_key: e.g. "tier1/mumbai/walking/<vid>/<vid>-007.mp4"
        enable_monument: if True, route monuments/* AND tags[clip_key].scene_type=="heritage_tourist" → "monument"
        tags: optional {clip_key: tag_record} lookup; required when enable_monument=True
              and the route is via VLM tag rather than path prefix.

    Returns:
        One of {"driving", "drone", "walking"} (3-class) or
        adds "monument" (4-class with enable_monument=True), or None for
        monuments/* clips when enable_monument=False (caller filters them out).

    Raises:
        ValueError: clip_key has unrecognized prefix or activity (FAIL-LOUD per CLAUDE.md).
    """
    if not clip_key:
        raise ValueError(f"Empty clip_key: {clip_key!r}")

    parts = clip_key.split("/")

    # Monument override: path-based first
    if enable_monument and parts[0] == "monuments":
        return "monument"
    # Monument override: VLM tag-based (heritage_tourist clips from any city)
    if enable_monument and tags is not None:
        rec = tags.get(clip_key)
        if isinstance(rec, dict) and rec.get("scene_type") == HERITAGE_SCENE_TYPE:
            return "monument"

    # monuments/ without enable_monument → return None (caller filters out)
    if parts[0] == "monuments":
        return None

    # Path-based 3-class derivation
    if parts[0] in ("tier1", "tier2"):
        if len(parts) < 3:
            raise ValueError(f"tier{{1,2}} clip_key missing activity segment: {clip_key!r}")
        activity = parts[2]
    elif parts[0] == "goa":
        if len(parts) < 2:
            raise ValueError(f"goa clip_key missing activity segment: {clip_key!r}")
        activity = parts[1]
    else:
        raise ValueError(f"Unrecognized clip_key prefix '{parts[0]}': {clip_key!r}")

    if activity not in PATH_TO_CLASS_3CLASS:
        raise ValueError(f"Unrecognized activity '{activity}' in clip_key: {clip_key!r}")
    return PATH_TO_CLASS_3CLASS[activity]


def load_subset_with_labels(subset_path, tags_path, *, enable_monument):
    """Load eval subset JSON + tags, return per-clip records with action labels.

    Returns: list of {"clip_key": str, "class": str, "class_id": int}.
    Drops clips with class=None silently (monuments/* with enable_monument=False).
    """
    subset_path = Path(subset_path)
    if not subset_path.exists():
        sys.exit(f"FATAL: --eval-subset not found: {subset_path}")
    subset = json.loads(subset_path.read_text())
    clip_keys = subset["clip_keys"]   # fail-loud — no .get(default)

    tags = None
    if enable_monument:
        if tags_path is None:
            sys.exit("FATAL: --tags-json required when --enable-monument-class is set")
        tags_path = Path(tags_path)
        if not tags_path.exists():
            sys.exit(f"FATAL: --tags-json not found: {tags_path}")
        tags_list = json.loads(tags_path.read_text())
        # tags.json schema: list of dicts with section/video_id/source_file fields
        # Build {clip_key: record} lookup matching m04_vlm_tag.py output.
        tags = {f"{t['section']}/{t['video_id']}/{t['source_file']}": t for t in tags_list}

    class_names = CLASS_NAMES_4CLASS if enable_monument else CLASS_NAMES_3CLASS
    class_to_id = {c: i for i, c in enumerate(class_names)}

    records = []
    for k in clip_keys:
        cls = parse_action_from_clip_key(k, enable_monument=enable_monument, tags=tags)
        if cls is None:
            continue
        records.append({"clip_key": k, "class": cls, "class_id": class_to_id[cls]})
    return records


def stratified_split(records, train_pct=0.70, val_pct=0.15, seed=99):
    """Stratified-by-class 70/15/15 split. Returns {clip_key: "train"|"val"|"test"}.

    Raises ValueError if any class has < 5 clips in any split (BCa CI floor).
    """
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["class"]].append(r["clip_key"])

    splits = {}
    for cls in sorted(by_class.keys()):       # deterministic class order
        keys = list(by_class[cls])
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)
        n_test = n - n_train - n_val
        if min(n_train, n_val, n_test) < 5:
            raise ValueError(
                f"Class '{cls}' has only n={n} → train={n_train}/val={n_val}/test={n_test}; "
                f"each split must have >=5 clips for BCa CI to be meaningful."
            )
        for k in keys[:n_train]:
            splits[k] = "train"
        for k in keys[n_train:n_train + n_val]:
            splits[k] = "val"
        for k in keys[n_train + n_val:]:
            splits[k] = "test"
    return splits


def write_action_labels_json(records, splits, output_path):
    """Atomically write {clip_key: {class, class_id, split}} + class_counts.json.

    Returns: the labels dict that was written (for in-memory chaining).
    """
    out = {}
    for r in records:
        k = r["clip_key"]
        out[k] = {"class": r["class"], "class_id": r["class_id"], "split": splits[k]}
    output_path = Path(output_path)
    save_json_checkpoint(out, output_path)

    # Diagnostic class-count table
    counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for k, info in out.items():
        counts[info["class"]][info["split"]] += 1
    save_json_checkpoint(dict(counts), output_path.parent / "class_counts.json")
    return out


def load_action_labels(labels_path):
    """Reverse of write_action_labels_json. Fail-loud if missing."""
    labels_path = Path(labels_path)
    if not labels_path.exists():
        sys.exit(f"FATAL: action_labels.json not found: {labels_path} -- run --stage labels first")
    return load_json_checkpoint(labels_path)


# ── Self-test (CLI entry point) ─────────────────────────────────────

def _self_test():
    p = argparse.ArgumentParser(description="Self-test: derive labels + print class counts")
    p.add_argument("--eval-subset", type=Path, required=True)
    p.add_argument("--tags-json", type=Path, default=None)
    p.add_argument("--enable-monument-class", action="store_true")
    args = p.parse_args()

    records = load_subset_with_labels(args.eval_subset, args.tags_json,
                                      enable_monument=args.enable_monument_class)
    splits = stratified_split(records)
    counts = Counter(r["class"] for r in records)
    print(f"Total: {len(records)} clips ({len(counts)} classes)")
    by_clip = {r["clip_key"]: r["class"] for r in records}
    for cls in sorted(counts.keys()):
        n = counts[cls]
        n_train = sum(1 for k, c in by_clip.items() if c == cls and splits[k] == "train")
        n_val   = sum(1 for k, c in by_clip.items() if c == cls and splits[k] == "val")
        n_test  = sum(1 for k, c in by_clip.items() if c == cls and splits[k] == "test")
        print(f"  {cls:10s}: total={n:>5d}  train={n_train:>5d}  val={n_val:>5d}  test={n_test:>5d}")


if __name__ == "__main__":
    _self_test()
