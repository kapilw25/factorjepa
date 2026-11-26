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
        --eval-subset data/eval_10k_local/eval_10k.json \\
        --motion-features data/eval_10k_local/m04d_motion_features/motion_features.npy
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
# iter16 M5 (2026-05-20): sklearn StratifiedGroupKFold powers the new
# video-disjoint stratified_split. scikit-learn>=1.3 is already pinned in
# requirements.txt; first sklearn import in src/ — adopt cleanly.
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.checkpoint import save_json_checkpoint, load_json_checkpoint
from utils.config import get_pipeline_config, get_probe_split   # iter17: single-source the split-threshold defaults
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)

# iter18 W7 (PLR2004): semantic named constants.
_MIN_KEY_PARTS = 3       # clip_key = <section>/<video_id>/<file> minimum depth


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
# iter17 (2026-05-27): single-sourced from pipeline.yaml probe.* (was literal 34/5
# duplicating the yaml values). Mode-keyed production values come from
# probe_action_labels.{min_clips_per_class,min_per_split}.<mode> via the callers;
# these mode-agnostic defaults back the CPU self-test (__main__) path.
_PROBE_CFG = get_pipeline_config()["probe"]
# iter18 (2026-06-20): the module fallback defaults read the CANONICAL mode-keyed
# probe_action_labels[*].full (100/5 = paper). The old scalar probe.min_clips_per_class:34 /
# min_per_split:5 was a STALE duplicate that conflicted with the mode-keyed block (it failed a
# standalone m04e SANITY run) and is deleted from pipeline.yaml. Production callers (m04e,
# probe_labels, run_eval) pass the per-MODE value explicitly; this is only the bare fallback.
_PAL_CFG = get_pipeline_config()["probe_action_labels"]
MIN_CLIPS_PER_CLASS_DEFAULT: int = _PAL_CFG["min_clips_per_class"]["full"]   # 100 (paper); callers pass mode-keyed
MIN_PER_SPLIT_DEFAULT: int = _PAL_CFG["min_per_split"]["full"]               # 5; BCa CI floor per split


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
    if flow_features_array.shape[1] < get_pipeline_config()["motion"]["feature_dim"]:
        sys.exit(
            f"FATAL: compute_magnitude_quartiles requires 23-D motion features "
            f"(Phase 0 m04d 13→23-D); got {flow_features_array.shape[1]}-D. "
            f"Rerun: CACHE_POLICY_ALL=2 python -u src/m04d_motion_features.py --FULL "
            f"--subset <subset.json> --local-data <local_data> "
            f"(writes to <local_data>/m04d_motion_features/ by default)"
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
                             min_clips_per_class=MIN_CLIPS_PER_CLASS_DEFAULT,
                             clip_keys=None, magnitude_quartiles=None,
                             class_names_override=None, edges_out=None):
    """Load eval subset + m04d motion features, return per-clip records with
    optical-flow-derived motion-class labels.

    Args:
        subset_path:           data/eval_*.json with "clip_keys" list
        motion_features_path:  <local_data>/m04d_motion_features/motion_features.npy from m04d (23D × N_clips, post-Phase-0)
        min_clips_per_class:   drop classes with fewer than this many clips (default → the
                               mode-keyed probe_action_labels.min_clips_per_class[mode])
        clip_keys:             iter16 M1 — when supplied, OVERRIDES the clip_keys read
                               from subset_path. Used by probe_action to inject a
                               clip_pool_ratio-derived subsample WITHOUT writing a
                               temp JSON file. subset_path existence is still checked
                               (it's the canonical reference for cache fingerprints).

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
    if clip_keys is None:
        subset = json.loads(subset_path.read_text())
        clip_keys = subset["clip_keys"]   # fail-loud — no .get(default)

    motion_features_path = Path(motion_features_path)
    paths_path = motion_features_path.with_name(
        motion_features_path.stem + ".paths.npy")
    if not motion_features_path.exists():
        sys.exit(
            f"FATAL: motion_features.npy not found at {motion_features_path}.\n"
            f"  Run first: python -u src/m04d_motion_features.py --FULL "
            f"--subset {subset_path} --local-data <local_data>\n"
            f"  (writes to <local_data>/m04d_motion_features/ by default; "
            f"override with --output-dir)"
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

    # Compute global magnitude quartiles over the FULL motion-features set (not just the eval
    # subset) so the bin cut-points are dataset-wide stable. iter18 cross-set: when
    # magnitude_quartiles is supplied (eval_10k's edges via --class-edges), REUSE them so subset_10k
    # bins 'still/slow/medium/fast' on the SAME boundaries the reused head was trained on.
    if magnitude_quartiles is not None:
        quartiles = list(magnitude_quartiles)
        print(f"  [motion-flow] REUSING supplied magnitude quartiles (cross-set): "
              f"Q1={quartiles[0]:.3f}  Q2={quartiles[1]:.3f}  Q3={quartiles[2]:.3f}")
    else:
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

    counts = Counter(cls for _, cls in raw)
    if class_names_override is not None:
        # iter18 cross-set: use eval_10k's FIXED class set + ID ORDER (preserve, do NOT re-sort) so
        # the reused head's output index i ↔ the SAME class. Drop subset clips whose class isn't in
        # eval_10k's set (the head can't score them). The min_clips filter is bypassed (set is fixed).
        class_names = list(class_names_override)
        surviving = set(class_names)
        print(f"  [motion-flow] REUSING supplied class set (cross-set, {len(class_names)} classes, "
              f"fixed ID order): {class_names}")
    else:
        # Pass 2 — filter sparse classes (>= min_clips_per_class)
        surviving = {cls for cls, n in counts.items() if n >= min_clips_per_class}
        if not surviving:
            sys.exit(f"FATAL: no flow class has >= {min_clips_per_class} clips. "
                     f"All counts: {dict(counts)}")
        # Pass 3 — alphabetical class_id assignment (deterministic across runs)
        class_names = sorted(surviving)
    class_to_id = {c: i for i, c in enumerate(class_names)}
    records = [{"clip_key": k, "class": cls, "class_id": class_to_id[cls]}
               for k, cls in raw if cls in surviving]
    if not records:
        sys.exit(f"FATAL: 0 clips matched the class set {sorted(surviving)} "
                 f"(class_names_override={class_names_override is not None}). "
                 f"Observed classes: {dict(counts)}")
    n_dropped = len(raw) - len(records)
    print(f"  [motion-flow] {len(class_names)} classes: {class_names}")
    print(f"  [motion-flow] kept {len(records)} clips, dropped {n_dropped} "
          f"(dropped counts: {dict((c, n) for c, n in counts.items() if c not in surviving)})")
    if edges_out is not None:
        edges_out["magnitude_quartiles"] = list(quartiles)
        edges_out["class_names"] = list(class_names)
    return records, class_names


def _extract_video_id(clip_key: str) -> str:
    """Extract video_id from a clip_key (variable section depth).

    Canonical formats observed in this repo:
      * tier1/tier2: ``<tier>/<city>/<action>/<video_id>/<file>.mp4`` (5 parts)
      * goa:        ``goa/<action>/<video_id>/<file>.mp4`` (4 parts)
      * monuments:  ``monuments/<site>/<video_id>/<file>.mp4`` (4 parts)

    The video_id is ALWAYS the directory immediately above the clip file →
    ``parts[-2]`` regardless of section depth.

    iter16 M5 (2026-05-20): introduced for video-disjoint stratified_split.

    FAIL LOUD per src/CLAUDE.md if the key has <3 parts (genuinely malformed
    — no plausible upstream manifest produces such a shape).
    """
    parts = clip_key.split("/")
    if len(parts) < _MIN_KEY_PARTS:
        raise ValueError(
            f"clip_key '{clip_key}' has only {len(parts)} parts; need ≥3 "
            f"(<section>/.../<video_id>/<file>.mp4). "
            f"Fix upstream manifest schema; do not silently fall through."
        )
    return parts[-2]


def subsample_manifest_for_mode(mode: str, clip_keys: list,
                                 motion_features_path,
                                 n_motion_classes: int) -> list:
    """Per-mode subsample of master manifest. iter16 M1 Option X (2026-05-21).

    Mode-keyed mechanism:
      SANITY → sorted(clip_keys)[:n]                        (code check; class-
                                                              imbalance OK)
      POC    → stratified_by_motion_class_subset()          (POC↔FULL parity
                                                              REQUIRED — every
                                                              motion class
                                                              represented)
      FULL   → identity (clip_pool_ratio.full = 1.0)        (no subsampling)

    Replaces the original Phase-1 ``sorted(clip_keys)[:n_clips]`` which
    violated POC↔FULL parity (alphabetical slicing yields 1-3 motion classes
    for POC instead of 8). Used by BOTH probe_action.run_labels_stage AND
    probe_labels.ensure_probe_labels_for_mode → symmetric subsampling
    across CLI and in-process bootstrap paths.

    Args:
        mode:                  "sanity" / "poc" / "full" (case-insensitive).
        clip_keys:             clip_keys list from the master manifest.
        motion_features_path:  m04d motion_features.npy (POC needs this for
                               class assignment; SANITY/FULL ignore it but it
                               must be a valid path either way for symmetry).
        n_motion_classes:      target class count (yaml-keyed, e.g. 8 post the
                               34-clip filter on the FULL pool).

    Returns:
        list of clip_keys (length depends on mode + clip_pool_ratio).

    Raises:
        KeyError: unknown mode (via clip_pool_ratio dict lookup).
    """
    # Lazy imports to avoid circular: action_labels ← config; eval_subset ←
    # action_labels (compute_magnitude_quartiles + parse_optical_flow_class).
    from utils.config import get_clip_pool_size
    mode = mode.lower()
    if mode == "full":
        return list(clip_keys)                            # identity (ratio=1.0)
    n_target = get_clip_pool_size(mode, len(clip_keys))
    if n_target >= len(clip_keys):                        # iter17: clip_pool_ratio≥1.0 → full
        return list(clip_keys)                            # corpus = identity (matches FULL). POC=1.0
                                                          # on eval_10k → the 8 robust classes, NOT
                                                          # the stratified booster's 13 (which lifted
                                                          # a 38-video class past the clip filter →
                                                          # video-disjoint split couldn't give ≥5/val)
    if mode == "sanity":
        return sorted(clip_keys)[:n_target]              # code check only
    # mode == "poc" (ratio<1.0) — stratified by motion class (POC↔FULL parity)
    from utils.eval_subset import stratified_by_motion_class_subset
    target_per_class = max(1, n_target // n_motion_classes)
    out = stratified_by_motion_class_subset(
        {"clip_keys": list(clip_keys)},
        motion_features_path,
        target_per_class,
    )
    return out["clip_keys"]


def stratified_split(records, train_pct=None, val_pct=None, seed=None,
                     *, min_per_split=MIN_PER_SPLIT_DEFAULT,
                     mode="full", test_all=False):
    """Class-stratified train/val/test split.

    iter18 (2026-06-20): None → the canonical probe_split[mode] (configs/pipeline.yaml);
    the old eval.action_split_* fallback was a stale duplicate, now deleted.

    Returns ``{clip_key: "train"|"val"|"test"}``.

    Mode-keyed split mechanism (iter16, 2026-05-21):
      - POC + FULL (paper-grade): VIDEO-DISJOINT via sklearn
        ``StratifiedGroupKFold`` (groups = video_id). Every clip from a
        given ``video_id`` lands in EXACTLY ONE split — no visual-style /
        background leakage. See M5 plan for full rationale.
      - SANITY (code-correctness only): clip-level stratified shuffle via
        ``sklearn.StratifiedShuffleSplit``. Per CLAUDE.md, SANITY validates
        code paths, not paper-grade splits; the video-disjoint guarantee is
        only enforced for POC/FULL where corpus size supports it. SANITY's
        tiny per-class video coverage (often 2-3 videos/class) cannot
        satisfy SGKF's n_splits constraint at iter16's val_pct=0.05.

    Algorithm (POC/FULL — video-disjoint):
      Phase 1: outer SGKF on (class_ids, video_ids) → peel off TEST set
      Phase 2: inner SGKF on remaining train+val pool, renormalised
               val_inner_pct = val_pct / (train_pct + val_pct)
      Phase 3: vectorised per-class per-split count + min_per_split assert
      Phase 4: defense-in-depth assert — NO video straddles splits

    Algorithm (SANITY — clip-level):
      Two sklearn StratifiedShuffleSplit passes: peel test set, then
      train/val from remainder. Stratifies by class_id only (groups
      ignored — no video-disjoint guarantee).

    FAIL LOUD on infeasibility (per src/CLAUDE.md): if any class has <
    min_per_split clips in any split, ValueError with per-class diagnostic.
    POC↔FULL parity rule applies to POC↔FULL, NOT to SANITY.

    iter18 CI-reduction: ``test_all=True`` → EVERY clip → the 'test' split (train/val empty),
    bypassing the SGKF split + the min_per_split asserts. For cross-set eval on a FULL disjoint
    corpus where NO head is trained here (it is loaded from the eval_10k run); the caller MUST skip
    the probe-training stages (this mode has no train data).
    """
    if test_all:
        clip_keys = [r["clip_key"] for r in records]
        n_v = len({_extract_video_id(k) for k in clip_keys})
        n_c = len({r["class_id"] for r in records})
        print(f"[stratified_split] TEST-ALL (cross-set eval): every clip → test; "
              f"N_clips={len(records)}, N_videos={n_v}, N_classes={n_c}; train/val/test = "
              f"0/0/{len(records)}")
        return {r["clip_key"]: "test" for r in records}
    _e = get_pipeline_config()["eval"]
    # iter18 (2026-06-20): the None-default for train/val reads the CANONICAL probe_split[mode]
    # (configs/pipeline.yaml) — the SAME source get_probe_split / m04e uses. The old
    # eval.action_split_train_pct/val_pct was a STALE duplicate ratio that conflicted with it; it's
    # deleted from pipeline.yaml. seed + k_tolerance stay under eval.action_split_*.
    if train_pct is None or val_pct is None:
        _ps = get_probe_split(mode)
        train_pct = _ps["train_pct"] if train_pct is None else train_pct
        val_pct = _ps["val_pct"] if val_pct is None else val_pct
    seed = _e["action_split_seed"] if seed is None else seed
    # SANITY fast path — clip-level stratified shuffle (no video-disjoint).
    if mode == "sanity":
        from sklearn.model_selection import StratifiedShuffleSplit
        clip_keys = [r["clip_key"] for r in records]
        class_ids = np.array([r["class_id"] for r in records])
        test_pct = 1.0 - train_pct - val_pct
        # Outer split: peel off test
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_pct,
                                          random_state=seed)
        trainval_idx, test_idx = next(sss_test.split(np.zeros_like(class_ids),
                                                       class_ids))
        # Inner split: train/val on remainder
        val_inner_pct = val_pct / (train_pct + val_pct)
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_inner_pct,
                                          random_state=seed + 1)
        inner_train, inner_val = next(sss_val.split(
            np.zeros_like(class_ids[trainval_idx]),
            class_ids[trainval_idx]))
        train_idx = trainval_idx[inner_train]
        val_idx   = trainval_idx[inner_val]
        splits = {}
        for i in train_idx: splits[clip_keys[i]] = "train"
        for i in val_idx:   splits[clip_keys[i]] = "val"
        for i in test_idx:  splits[clip_keys[i]] = "test"
        n_v = len({_extract_video_id(k) for k in clip_keys})
        print(f"[stratified_split] SANITY clip-level (NOT video-disjoint): "
              f"N_clips={len(records)}, N_videos={n_v}, "
              f"N_classes={len(set(class_ids))}; train/val/test = "
              f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
        return splits

    # POC/FULL — video-disjoint SGKF (existing M5 algorithm)
    # 1 — Build parallel arrays (vectorised on 115K records).
    clip_keys = np.array([r["clip_key"] for r in records])
    class_ids = np.array([r["class_id"] for r in records])
    video_ids = np.array([_extract_video_id(k) for k in clip_keys])

    n_classes = int(class_ids.max() + 1)
    n_videos  = len(np.unique(video_ids))
    n_total   = len(records)

    # 2 — Outer SGKF: carve out TEST.
    test_pct = 1.0 - train_pct - val_pct
    k_test   = int(round(1.0 / test_pct))           # 0.20 → 5 ; 0.15 → 7
    if abs(1.0 / k_test - test_pct) > get_pipeline_config()["eval"]["action_split_k_tolerance"]:
        raise ValueError(
            f"test_pct={test_pct:.3f} maps to k={k_test} (~{1/k_test:.3f}); "
            f"choose train_pct/val_pct such that (1 - train - val) = 1/k "
            f"for integer k (e.g. 0.10, 0.125, 0.15, 0.20, 0.25)."
        )
    sgkf_outer = StratifiedGroupKFold(n_splits=k_test, shuffle=True,
                                      random_state=seed)
    trainval_idx, test_idx = next(sgkf_outer.split(
        np.zeros_like(class_ids), class_ids, video_ids,
    ))

    # 3 — Inner SGKF on train+val pool (renormalised val ratio, seed+1).
    val_inner_pct = val_pct / (train_pct + val_pct)
    k_val = int(round(1.0 / val_inner_pct))
    sgkf_inner = StratifiedGroupKFold(n_splits=k_val, shuffle=True,
                                      random_state=seed + 1)
    inner_train, inner_val = next(sgkf_inner.split(
        np.zeros_like(class_ids[trainval_idx]),
        class_ids[trainval_idx],
        video_ids[trainval_idx],
    ))
    train_idx = trainval_idx[inner_train]
    val_idx   = trainval_idx[inner_val]

    # 4 — Build {clip_key: split} dict.
    splits = {}
    for i in train_idx: splits[clip_keys[i]] = "train"
    for i in val_idx:   splits[clip_keys[i]] = "val"
    for i in test_idx:  splits[clip_keys[i]] = "test"

    # 5 — Vectorised per-class per-split counts + min_per_split FAIL LOUD.
    split_id = np.full(n_total, -1, dtype=np.int8)
    split_id[train_idx], split_id[val_idx], split_id[test_idx] = 0, 1, 2
    counts = np.zeros((n_classes, 3), dtype=np.int64)
    np.add.at(counts, (class_ids, split_id), 1)

    infeasible = [(int(c), counts[c].tolist()) for c in range(n_classes)
                  if counts[c].min() < min_per_split]
    if infeasible:
        videos_per_class = {int(c): int(np.unique(video_ids[class_ids == c]).size)
                            for c in range(n_classes)}
        raise ValueError(
            f"video-disjoint stratified_split failed min_per_split="
            f"{min_per_split} for {len(infeasible)} class(es). "
            f"Per-class [train,val,test] counts: {infeasible}. "
            f"Videos-per-class: {videos_per_class}. "
            f"Fix options: (a) raise corpus N (sanity.default / poc.default_n "
            f"in pipeline.yaml), (b) raise min_clips_per_class so sparse "
            f"classes drop earlier, (c) lower min_per_split via CLI (only "
            f"for non-paper SANITY runs)."
        )

    # 6 — Defense-in-depth: NO video may straddle splits (Phase-4 invariant).
    video_to_splits = defaultdict(set)
    for k, s in splits.items():
        video_to_splits[_extract_video_id(k)].add(s)
    straddlers = {v: sp for v, sp in video_to_splits.items() if len(sp) > 1}
    assert not straddlers, (
        f"BUG: {len(straddlers)} videos straddling splits — {dict(list(straddlers.items())[:5])}"
    )

    print(f"[stratified_split] video-disjoint iter16+ mode: "
          f"N_clips={n_total}, N_videos={n_videos}, N_classes={n_classes}; "
          f"train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
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
    save_json_checkpoint(dict(counts), output_path.parent / artifact("class_counts"))
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
    by_clip = {r["clip_key"]: r["class"] for r in records}

    # iter16 M5: per-class videos count (helps diagnose infeasibility) +
    # per-class per-split CLIP and VIDEO counts (video-disjoint invariant proof).
    by_video_class = defaultdict(set)
    for r in records:
        by_video_class[r["class"]].add(_extract_video_id(r["clip_key"]))
    video_split = {}                                  # {video_id: split}
    for k, s in splits.items():
        video_split[_extract_video_id(k)] = s          # safe — disjoint by Phase-4 assert

    n_videos = len(video_split)
    print(f"\nTotal: {len(records)} clips · {n_videos} videos · "
          f"{len(counts)} classes after filter")
    print(f"  {'class':<24s}  {'clips':>10s}  {'train':>10s}  {'val':>10s}  {'test':>10s}  "
          f"{'│ vids':>7s}  {'tr_v':>5s}  {'va_v':>5s}  {'te_v':>5s}")
    for cls in sorted(counts.keys()):
        n = counts[cls]
        n_train = sum(1 for k, c in by_clip.items() if c == cls and splits[k] == "train")
        n_val   = sum(1 for k, c in by_clip.items() if c == cls and splits[k] == "val")
        n_test  = sum(1 for k, c in by_clip.items() if c == cls and splits[k] == "test")
        cls_videos = by_video_class[cls]
        v_train = sum(1 for v in cls_videos if video_split.get(v) == "train")
        v_val   = sum(1 for v in cls_videos if video_split.get(v) == "val")
        v_test  = sum(1 for v in cls_videos if video_split.get(v) == "test")
        print(f"  {cls:<24s}  {n:>10d}  {n_train:>10d}  {n_val:>10d}  {n_test:>10d}  "
              f"{len(cls_videos):>7d}  {v_train:>5d}  {v_val:>5d}  {v_test:>5d}")

    # Defense-in-depth re-assertion outside stratified_split (in case of future
    # refactor regression): verify no video_id appears in >1 split.
    vmap = defaultdict(set)
    for k, s in splits.items():
        vmap[_extract_video_id(k)].add(s)
    straddlers = {v: list(sp) for v, sp in vmap.items() if len(sp) > 1}
    if straddlers:
        print(f"\n  ❌ BUG: {len(straddlers)} videos straddle splits — "
              f"{dict(list(straddlers.items())[:5])}")
        sys.exit(1)
    print(f"\n  ✅ video-disjoint invariant: every video_id is in exactly ONE split "
          f"({n_videos}/{n_videos} videos clean)")


if __name__ == "__main__":
    _self_test()
