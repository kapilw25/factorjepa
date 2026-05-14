"""Optical-flow-derived motion-class label derivation for probe_* modules. CPU-only.

Reads m04d_motion_features.py outputs (RAFT 23-D per-clip flow features (post-Phase-0)) and
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
# `fg_mean_mag` column (vec[13] from m04d, post-Phase-0). Stable alphabetical-by-id
# order. Phase 0 / iter15: switched from vec[0] (global, camera-contaminated) to
# vec[13] (foreground, camera-subtracted → agent motion only).
MOTION_MAGNITUDE_BINS: list = ["fast", "medium", "slow", "still"]

# Motion-direction bins: 4 grouped buckets over the FG 8-bin angle histogram
# (vec[15:23] from m04d, post-Phase-0). Bins 0,1 → rightward; 2,3 → upward;
# 4,5 → leftward; 6,7 → downward. Index ordering (0..3) is FIXED and matches
# np.argmax order; the public name list below is sorted for class_id stability.
MOTION_DIRECTION_BINS: list = ["downward", "leftward", "rightward", "upward"]
_DIRECTION_BIN_ORDER: list = ["rightward", "upward", "leftward", "downward"]   # argmax index → name

MOTION_SEPARATOR: str = "__"
MIN_CLIPS_PER_CLASS_DEFAULT: int = 34   # ≥34 → ≥5 per split at 70/15/15 (BCa CI floor)
MIN_PER_SPLIT_DEFAULT: int = 5          # BCa CI floor per split


# ── Public API ───────────────────────────────────────────────────────

def compute_magnitude_quartiles(flow_features_array: np.ndarray) -> list:
    """Return [Q1, Q2, Q3] cut-points over the dataset's FG (camera-subtracted)
    mean_magnitude column.

    Args:
        flow_features_array: (N_clips, 23) float32 from m04d motion_features.npy
            (post-Phase 0). Index [:, 13] = fg_mean_mag.
    Returns:
        [Q1, Q2, Q3] floats — agent-motion quartile cut-points so each magnitude
        bin receives ~25 % of clips.

    Phase 0 / iter15: switched from global mean_magnitude (vec[0],
    camera-contaminated) to foreground mean_magnitude (vec[13], camera-subtracted)
    so motion-class boundaries reflect AGENT motion, not camera-induced
    translation. Requires 23-D m04d output.
    """
    if flow_features_array.shape[1] < 23:
        sys.exit(
            f"FATAL: compute_magnitude_quartiles requires 23-D motion features "
            f"(Phase 0 m04d 13→23-D); got {flow_features_array.shape[1]}-D. "
            f"Rerun: CACHE_POLICY_ALL=2 python -u src/m04d_motion_features.py --FULL "
            f"--subset <subset.json> --local-data <local_data> "
            f"--features-out <local_data>/motion_features.npy"
        )
    fg_mean_mags = flow_features_array[:, 13]
    return [float(np.quantile(fg_mean_mags, q)) for q in (0.25, 0.5, 0.75)]


def parse_optical_flow_class(clip_key, flow_features_by_key, magnitude_quartiles):
    """Pure-motion class string = `<magnitude_bin>__<direction_bin>`,
    e.g. 'fast__rightward', 'still__downward', 'medium__upward'.

    Args:
        clip_key:               full clip key e.g. 'tier1/mumbai/walking/<vid>/<vid>-007.mp4'
        flow_features_by_key:   {clip_key: 23-D float32 vec from m04d motion_features.npy}
        magnitude_quartiles:    [Q1, Q2, Q3] cut-points over dataset fg_mean_mag column

    m04d FEATURE_NAMES indices (Phase 0 / iter15 — 23-D):
        [0]    mean_magnitude         (global flow — camera-contaminated)
        [1]    magnitude_std
        [2]    max_magnitude
        [3-10] dir_hist_0..7          (global 8-bin angle histogram)
        [11]   camera_motion_x
        [12]   camera_motion_y
        [13]   fg_mean_mag            (camera-subtracted — agent motion only) ← bin axis
        [14]   fg_max_mag
        [15-22] fg_dir_hist_0..7      (FG 8-bin angle histogram)             ← direction axis

    Phase 0 / iter15: switched magnitude binning from vec[0] (global, camera-
    contaminated) to vec[13] (fg_mean_mag, agent-only). Direction binning moved
    from vec[3:11] (global) to vec[15:23] (FG). Camera-induced global translation
    no longer dominates class assignments.

    Magnitude bins (4): still / slow / medium / fast — three quartile cut-points.
    Direction bins (4): rightward / upward / leftward / downward — argmax over
    the 4 grouped FG angle buckets:
        [0,1] → rightward;  [2,3] → upward;  [4,5] → leftward;  [6,7] → downward.

    Returns:
        Class name string (e.g. 'fast__rightward') or None if the clip has no
        motion-features record (caller filters None).
    """
    vec = flow_features_by_key.get(clip_key)
    if vec is None:
        return None

    fg_mean_mag = float(vec[13])
    q1, q2, q3 = magnitude_quartiles
    if fg_mean_mag < q1:
        mag_bin = "still"
    elif fg_mean_mag < q2:
        mag_bin = "slow"
    elif fg_mean_mag < q3:
        mag_bin = "medium"
    else:
        mag_bin = "fast"

    fg_dir_hist = vec[15:23]
    grouped = np.array([
        fg_dir_hist[0] + fg_dir_hist[1],   # rightward
        fg_dir_hist[2] + fg_dir_hist[3],   # upward
        fg_dir_hist[4] + fg_dir_hist[5],   # leftward
        fg_dir_hist[6] + fg_dir_hist[7],   # downward
    ], dtype=np.float64)
    dir_bin = _DIRECTION_BIN_ORDER[int(np.argmax(grouped))]

    return f"{mag_bin}{MOTION_SEPARATOR}{dir_bin}"


def load_subset_with_labels(subset_path, motion_features_path, *,
                             min_clips_per_class=MIN_CLIPS_PER_CLASS_DEFAULT):
    """Load eval subset + m04d motion features, return per-clip records with
    optical-flow-derived motion-class labels.

    Args:
        subset_path:           data/eval_*.json with "clip_keys" list
        motion_features_path:  <local_data>/motion_features.npy from m04d (23D × N_clips, post-Phase-0)
        min_clips_per_class:   drop classes with fewer than this many clips (default 34
                               → ≥5 per split at 70/15/15)

    Returns:
        (records, class_names):
          records: list of {"clip_key": str, "class": str, "class_id": int}
          class_names: sorted list of surviving class strings (alphabetical →
                       deterministic class_id assignment across runs)

    Schema of m04d output:
        motion_features.npy        — (N_clips, 23) float32, post-Phase-0 RAFT optical-flow features
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
    flow_features = np.load(motion_features_path)                 # (N, 23) post-Phase-0
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
    test_pct = 1.0 - train_pct - val_pct
    for cls in sorted(by_class.keys()):       # deterministic class order
        keys = list(by_class[cls])
        rng.shuffle(keys)
        n = len(keys)
        # iter13 v13 (2026-05-07): greedy allocation. Previously used
        # `int(n*pct)` floor for val + test which gave 0 when pct*n < 1
        # (e.g. n=5, val_pct=0.15 → val=0 → ValueError). The min_per_split
        # contract is ≥1 per split (BCa CI floor = 1 sample minimum) so
        # promote val + test to 1 and let train absorb the rounding loss.
        # For large n where int(n*pct) ≥ 1 the allocation is unchanged
        # (max(1, k) == k for k≥1), so paper-grade ratios are preserved.
        # Net effect: keeps small-n classes that otherwise FATAL'd, making
        # the probe HARDER (more classes to discriminate). Min keepable n=3
        # (val=1, test=1, train=1).
        n_val = max(1, int(n * val_pct))
        n_test = max(1, int(n * test_pct))
        n_train = n - n_val - n_test
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
