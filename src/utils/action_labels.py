"""Optical-flow-derived motion-class label derivation for probe_* modules. CPU-only.

Reads m04d_motion_features.py outputs (RAFT 13D per-clip flow features) and
bins them into PURE motion classes (mean_magnitude × dominant_direction = 16
classes) that cannot be solved from a single frame. Replaces the legacy
3/4-class path-derived action labels (saturated frozen V-JEPA at 0.94+).

USAGE (called by probe_*.py — direct __main__ entry exists for self-test only):
    from utils.action_labels import (
        parse_optical_flow_class, compute_magnitude_quartiles,
        load_subset_with_labels, stratified_split,
        write_action_labels_json, load_action_labels,
        MOTION_MAGNITUDE_BINS, MOTION_DIRECTION_BINS,
    )

Self-test (CPU sanity, prints per-class clip counts + split sizes):
    python -u src/utils/action_labels.py \\
        --eval-subset data/eval_10k.json \\
        --motion-features data/eval_10k_local/motion_features.npy
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

# Motion-magnitude bins: 4 quantile-derived buckets over the dataset's
# `mean_magnitude` column (vec[0] from m04d). Stable alphabetical-by-id order.
MOTION_MAGNITUDE_BINS: list = ["fast", "medium", "slow", "still"]

# Motion-direction bins: 4 grouped buckets over the 8-bin angle histogram
# (vec[3:11] from m04d). Bins 0,1 → rightward; 2,3 → upward; 4,5 → leftward;
# 6,7 → downward. Index ordering (0..3) is FIXED and matches np.argmax order;
# the public name list below is sorted for class_id stability.
MOTION_DIRECTION_BINS: list = ["downward", "leftward", "rightward", "upward"]
_DIRECTION_BIN_ORDER: list = ["rightward", "upward", "leftward", "downward"]   # argmax index → name

MOTION_SEPARATOR: str = "__"
MIN_CLIPS_PER_CLASS_DEFAULT: int = 34   # ≥34 → ≥5 per split at 70/15/15 (BCa CI floor)
MIN_PER_SPLIT_DEFAULT: int = 5          # BCa CI floor per split


# ── Public API ───────────────────────────────────────────────────────

def compute_magnitude_quartiles(flow_features_array: np.ndarray) -> list:
    """Return [Q1, Q2, Q3] cut-points over the dataset's mean_magnitude column.

    Args:
        flow_features_array: (N_clips, 13) float32 from m04d motion_features.npy
    Returns:
        [Q1, Q2, Q3] floats — global quartile cut-points so each magnitude bin
        receives ~25 % of clips.
    """
    mean_mags = flow_features_array[:, 0]
    return [float(np.quantile(mean_mags, q)) for q in (0.25, 0.5, 0.75)]


def parse_optical_flow_class(clip_key, flow_features_by_key, magnitude_quartiles):
    """Pure-motion class string = `<magnitude_bin>__<direction_bin>`,
    e.g. 'fast__rightward', 'still__downward', 'medium__upward'.

    Args:
        clip_key:               full clip key e.g. 'tier1/mumbai/walking/<vid>/<vid>-007.mp4'
        flow_features_by_key:   {clip_key: 13D float32 vec from m04d motion_features.npy}
        magnitude_quartiles:    [Q1, Q2, Q3] cut-points over dataset mean_magnitude column

    m04d FEATURE_NAMES indices:
        [0]   mean_magnitude
        [1]   magnitude_std
        [2]   max_magnitude
        [3-10] dir_hist_0..7  (8-bin angle histogram; bin i covers [-π + i·π/4, -π + (i+1)·π/4))
        [11]  camera_motion_x
        [12]  camera_motion_y

    Magnitude bins (4): still / slow / medium / fast — three quartile cut-points.
    Direction bins (4): rightward / upward / leftward / downward — argmax over
    the 4 grouped angle buckets:
        [0,1] → rightward;  [2,3] → upward;  [4,5] → leftward;  [6,7] → downward.

    Returns:
        Class name string (e.g. 'fast__rightward') or None if the clip has no
        motion-features record (caller filters None).
    """
    vec = flow_features_by_key.get(clip_key)
    if vec is None:
        return None

    mean_mag = float(vec[0])
    q1, q2, q3 = magnitude_quartiles
    if mean_mag < q1:
        mag_bin = "still"
    elif mean_mag < q2:
        mag_bin = "slow"
    elif mean_mag < q3:
        mag_bin = "medium"
    else:
        mag_bin = "fast"

    dir_hist = vec[3:11]
    grouped = np.array([
        dir_hist[0] + dir_hist[1],   # rightward
        dir_hist[2] + dir_hist[3],   # upward
        dir_hist[4] + dir_hist[5],   # leftward
        dir_hist[6] + dir_hist[7],   # downward
    ], dtype=np.float64)
    dir_bin = _DIRECTION_BIN_ORDER[int(np.argmax(grouped))]

    return f"{mag_bin}{MOTION_SEPARATOR}{dir_bin}"


def load_subset_with_labels(subset_path, motion_features_path, *,
                             min_clips_per_class=MIN_CLIPS_PER_CLASS_DEFAULT):
    """Load eval subset + m04d motion features, return per-clip records with
    optical-flow-derived motion-class labels.

    Args:
        subset_path:           data/eval_*.json with "clip_keys" list
        motion_features_path:  <local_data>/motion_features.npy from m04d (13D × N_clips)
        min_clips_per_class:   drop classes with fewer than this many clips (default 34
                               → ≥5 per split at 70/15/15)

    Returns:
        (records, class_names):
          records: list of {"clip_key": str, "class": str, "class_id": int}
          class_names: sorted list of surviving class strings (alphabetical →
                       deterministic class_id assignment across runs)

    Schema of m04d output:
        motion_features.npy        — (N_clips, 13) float32 RAFT optical-flow features
        motion_features.paths.npy  — (N_clips,) clip-key strings aligned by row
    """
    subset_path = Path(subset_path)
    if not subset_path.exists():
        sys.exit(f"FATAL: --eval-subset not found: {subset_path}")
    subset = json.loads(subset_path.read_text())
    clip_keys = subset["clip_keys"]   # fail-loud — no .get(default)

    motion_features_path = Path(motion_features_path)
    paths_path = motion_features_path.with_name(
        motion_features_path.stem + ".paths.npy")
    if not motion_features_path.exists():
        sys.exit(
            f"FATAL: motion_features.npy not found at {motion_features_path}.\n"
            f"  Run first: python -u src/m04d_motion_features.py --FULL "
            f"--subset {subset_path} --local-data <local_data> "
            f"--features-out {motion_features_path}"
        )
    if not paths_path.exists():
        sys.exit(f"FATAL: motion_features.paths.npy not found at {paths_path} "
                 f"(must be next to motion_features.npy)")
    flow_features = np.load(motion_features_path)                 # (N, 13)
    flow_paths = np.load(paths_path, allow_pickle=True)           # (N,) clip keys
    if flow_features.shape[0] != flow_paths.shape[0]:
        sys.exit(f"FATAL: motion_features.npy rows ({flow_features.shape[0]}) "
                 f"!= paths.npy rows ({flow_paths.shape[0]})")
    flow_features_by_key = {str(k): flow_features[i]
                            for i, k in enumerate(flow_paths)}

    # Compute global magnitude quartiles over the FULL motion-features set (not
    # just the eval subset) so the bin cut-points are dataset-wide stable.
    quartiles = compute_magnitude_quartiles(flow_features)
    print(f"  [motion-flow] magnitude quartiles: "
          f"Q1={quartiles[0]:.3f}  Q2={quartiles[1]:.3f}  Q3={quartiles[2]:.3f}")

    # Pass 1 — derive flow class per clip (None means no m04d record → filter)
    raw = []
    n_no_record = 0
    for k in clip_keys:
        cls = parse_optical_flow_class(k, flow_features_by_key, quartiles)
        if cls is None:
            n_no_record += 1
            continue
        raw.append((k, cls))
    if n_no_record:
        print(f"  [motion-flow] {n_no_record}/{len(clip_keys)} clip_keys had no "
              f"motion_features record (m04d may not have processed them yet)")

    # Pass 2 — filter sparse classes (>= min_clips_per_class)
    counts = Counter(cls for _, cls in raw)
    surviving = {cls for cls, n in counts.items() if n >= min_clips_per_class}
    if not surviving:
        sys.exit(f"FATAL: no flow class has >= {min_clips_per_class} clips. "
                 f"All counts: {dict(counts)}")

    # Pass 3 — alphabetical class_id assignment (deterministic across runs)
    class_names = sorted(surviving)
    class_to_id = {c: i for i, c in enumerate(class_names)}
    records = [{"clip_key": k, "class": cls, "class_id": class_to_id[cls]}
               for k, cls in raw if cls in surviving]
    n_dropped = len(raw) - len(records)
    print(f"  [motion-flow] {len(class_names)} classes after >= "
          f"{min_clips_per_class}-clip filter: {class_names}")
    print(f"  [motion-flow] kept {len(records)} clips, dropped {n_dropped} "
          f"in {len(counts) - len(surviving)} sparse classes "
          f"(dropped counts: "
          f"{dict((c, n) for c, n in counts.items() if c not in surviving)})")
    return records, class_names


def stratified_split(records, train_pct=0.70, val_pct=0.15, seed=99,
                     *, min_per_split=MIN_PER_SPLIT_DEFAULT):
    """Stratified-by-class 70/15/15 split. Returns {clip_key: "train"|"val"|"test"}.

    Raises ValueError if any class has < min_per_split clips in any split.
    Default min_per_split=5 (BCa CI floor). SANITY can pass a lower value if
    class data is sparse — at the cost of meaningless per-class CI.
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
        if min(n_train, n_val, n_test) < min_per_split:
            raise ValueError(
                f"Class '{cls}' has only n={n} → train={n_train}/val={n_val}/test={n_test}; "
                f"each split must have >={min_per_split} clips for BCa CI to be meaningful."
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
    p = argparse.ArgumentParser(description="Self-test: derive motion-flow labels + print class counts")
    p.add_argument("--eval-subset", type=Path, required=True)
    p.add_argument("--motion-features", type=Path, required=True,
                   help="m04d motion_features.npy path. Companion .paths.npy must exist next to it.")
    p.add_argument("--min-clips-per-class", type=int, default=MIN_CLIPS_PER_CLASS_DEFAULT)
    p.add_argument("--min-per-split", type=int, default=MIN_PER_SPLIT_DEFAULT)
    args = p.parse_args()

    records, class_names = load_subset_with_labels(
        args.eval_subset, args.motion_features,
        min_clips_per_class=args.min_clips_per_class)
    splits = stratified_split(records, min_per_split=args.min_per_split)
    counts = Counter(r["class"] for r in records)
    print(f"\nTotal: {len(records)} clips ({len(counts)} classes after filter)")
    by_clip = {r["clip_key"]: r["class"] for r in records}
    for cls in sorted(counts.keys()):
        n = counts[cls]
        n_train = sum(1 for k, c in by_clip.items() if c == cls and splits[k] == "train")
        n_val   = sum(1 for k, c in by_clip.items() if c == cls and splits[k] == "val")
        n_test  = sum(1 for k, c in by_clip.items() if c == cls and splits[k] == "test")
        print(f"  {cls:24s}: total={n:>5d}  train={n_train:>5d}  val={n_val:>5d}  test={n_test:>5d}")


if __name__ == "__main__":
    _self_test()
