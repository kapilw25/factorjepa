"""Easy-to-hard sample ordering for curriculum learning (Bengio 2009).

Difficulty proxy: FOREGROUND motion magnitude (vec[13] = fg_mean_mag) from
Phase 0 m04d 23-D features. Camera motion is subtracted → ranks clips by AGENT
motion only, not by camera-induced global translation. Easy = still agent;
Hard = fast agent. Falls back to global mean_mag (vec[0]) with a loud WARN if
Phase 0 features are not yet built (still 13-D).

USAGE (consumed by m09a2_pretrain_head.py / m09c2_surgery_head.py):
    from utils.data_curriculum import sort_by_fg_magnitude, get_active_pool
    sorted_keys = sort_by_fg_magnitude(clip_keys, motion_features_path, order="ascending")
    active = get_active_pool(epoch, total_epochs, sorted_keys, pacing="linear")
"""
import sys
from pathlib import Path

import numpy as np


def sort_by_fg_magnitude(clip_keys, motion_features_path, order):
    """Sort clip_keys by FOREGROUND motion magnitude (vec[13]).

    Phase-0-aware: prefers fg_mean_mag at vec[13] (camera-subtracted, agent-only)
    when motion_features is 23-D; falls back to global mean_mag at vec[0] with a
    loud WARN if features are still 13-D (Phase 0 not run yet).

    Args:
        clip_keys: iterable of clip_key strings (e.g., from manifest).
        motion_features_path: path to m04d motion_features.npy (shape (N, 13) or (N, 23)).
        order: "ascending" (easy → hard) or "descending" (hard → easy, adversarial).

    Returns:
        list[str] — clip_keys sorted by the chosen difficulty axis.
    """
    if order not in ("ascending", "descending"):
        sys.exit(f"FATAL: sort_by_fg_magnitude order must be ascending|descending (got: {order})")

    features = np.load(motion_features_path)
    paths_npy = Path(motion_features_path).with_suffix(".paths.npy")
    if not paths_npy.exists():
        sys.exit(f"FATAL: sort_by_fg_magnitude — missing companion paths file {paths_npy}")
    paths = np.load(paths_npy, allow_pickle=True)
    key_to_idx = {Path(p).stem: i for i, p in enumerate(paths)}

    feat_dim = features.shape[1]
    if feat_dim >= 23:
        difficulty_col = 13
        print(f"  [curriculum] using Phase-0 FG magnitude (vec[13]) — "
              f"camera-subtracted, agent-only ({feat_dim}-D features)")
    elif feat_dim == 13:
        difficulty_col = 0
        print(f"  WARN [curriculum] Phase 0 features NOT available (still 13-D); "
              f"falling back to global mean_mag (vec[0]) — camera-motion-contaminated "
              f"difficulty signal. Run Phase 0 (m04d 13→23-D) for principled curriculum.")
    else:
        sys.exit(f"FATAL: unexpected motion_features shape {features.shape}; "
                 f"expected (N, 13) or (N, 23+)")

    pairs = []
    missing = 0
    for k in clip_keys:
        if k in key_to_idx:
            pairs.append((k, float(features[key_to_idx[k], difficulty_col])))
        else:
            missing += 1
    if missing > 0:
        print(f"  [curriculum] WARN: {missing}/{len(list(clip_keys))} clip_keys "
              f"have no motion_features row (will be dropped from curriculum)")
    pairs.sort(key=lambda x: x[1], reverse=(order == "descending"))
    return [k for k, _ in pairs]


def get_active_pool(epoch, total_epochs, sorted_keys, pacing):
    """Pacing function — fraction of sorted pool exposed at given epoch.

    Pacing modes (Bengio 2009 §3):
        linear  — frac grows linearly; full pool exposed at epoch = total_epochs//2.
        step    — bottom-50% for first half, full pool second half.
        log     — frac grows logarithmically (slow expansion early).

    Args:
        epoch: 0-indexed current epoch.
        total_epochs: total epoch budget for this run.
        sorted_keys: list[str] from sort_by_fg_magnitude (ascending = easy-first).
        pacing: "linear" | "step" | "log".

    Returns:
        list[str] — prefix of sorted_keys representing the active pool at this epoch.
    """
    half = max(1, total_epochs // 2)
    if pacing == "linear":
        frac = min(1.0, (epoch + 1) / half)
    elif pacing == "step":
        frac = 0.5 if epoch < half else 1.0
    elif pacing == "log":
        frac = min(1.0, float(np.log1p(epoch + 1)) / float(np.log1p(half)))
    else:
        sys.exit(f"FATAL: unknown pacing '{pacing}' (use linear|step|log)")
    n_active = max(1, int(frac * len(sorted_keys)))
    return sorted_keys[:n_active]
