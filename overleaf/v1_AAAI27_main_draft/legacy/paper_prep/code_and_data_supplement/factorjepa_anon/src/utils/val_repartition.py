"""Re-partition the held-out val/test splits: val → a STRATIFIED (motion-class-balanced) subset of
`target_val` clips; the leftover val clips move into TEST. TRAIN-POOL-PRESERVING BY CONSTRUCTION —
`val ∪ test` is unchanged (clips only move between the two held-out partitions), so clip_splits'
train_pool (= universe − (val ∪ test)) is identical and a mid-flight training run keeps a valid pool.

WHY (iter19 2026-07-05): the 5% val split (~5,750 of 116k) is far past the ~500-1,000 stability
plateau for a monitor / best-checkpoint signal, and preloading all of it OOM'd the box → the earlier
m09a1 cap took `sorted(val)[:1000]` = an alphabetically-clustered, NON-stratified slice that biased
best-ckpt selection. This computes the val subsample ONCE (CLAUDE.md SHARED-DERIVATION-VIA-CLI) as a
motion-class-balanced draw and hands EVERY trainer the same val_split.json; the freed ~4,750 clips
join the test set (tighter eval CIs) instead of being wasted. Idempotent (no-op if val ≤ target).

USAGE:
  python -u src/utils/val_repartition.py \
      --val-split  data/full_local/val_split.json \
      --test-split data/full_local/test_split.json \
      --motion-features data/full_local/m04d_motion_features/motion_features.npy \
      --target-val 1000
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.eval_subset import stratified_by_motion_class_subset


def _load_keys(p: Path) -> list:
    d = json.loads(p.read_text())
    if "clip_keys" not in d:
        sys.exit(f"FATAL val_repartition: {p} has no 'clip_keys' key (got {list(d)[:5]}).")
    return d["clip_keys"]


def _atomic_write(p: Path, keys: list, extra: dict) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"clip_keys": keys, "n_clips": len(keys), **extra}, indent=2))
    tmp.replace(p)


def _count_motion_classes(val_keys, motion_features_path) -> int:
    """Distinct m04d optical-flow motion classes present in val_keys — the SAME class fn the
    stratified sampler uses, so per_class = ceil(target / n_classes) lands the total near target."""
    import numpy as np
    from utils.action_labels import compute_magnitude_quartiles, parse_optical_flow_class
    mf = Path(motion_features_path)
    feats = np.load(mf)
    paths = np.load(mf.with_name(mf.stem + ".paths.npy"), allow_pickle=True)
    by_key = {str(k): feats[i] for i, k in enumerate(paths)}
    quartiles = compute_magnitude_quartiles(feats)
    classes = {parse_optical_flow_class(k, by_key, quartiles) for k in val_keys}
    classes.discard(None)
    if not classes:
        sys.exit(f"FATAL val_repartition: 0 val clips matched motion_features at {mf} — re-run m04d.")
    return len(classes)


def repartition(val_path: Path, test_path: Path, motion_features: str, target_val: int) -> None:
    val_keys = _load_keys(val_path)
    test_keys = _load_keys(test_path)
    union_before = set(val_keys) | set(test_keys)
    if len(val_keys) <= target_val:
        print(f"  [val-repartition] val={len(val_keys)} ≤ target={target_val} — no-op (already sized).",
              flush=True)
        return
    # per-class target so the stratified total ≈ target_val (balanced across the motion classes).
    n_cls = _count_motion_classes(val_keys, motion_features)
    per_class = max(1, math.ceil(target_val / n_cls))
    picked = stratified_by_motion_class_subset({"clip_keys": val_keys}, motion_features, per_class)
    new_val = picked["clip_keys"]
    picked_set = set(new_val)
    leftover = [k for k in val_keys if k not in picked_set]
    new_test = test_keys + leftover
    # INVARIANT (the whole point): val∪test unchanged → clip_splits train_pool identical → seed safe.
    if set(new_val) | set(new_test) != union_before:
        sys.exit("FATAL val_repartition: val∪test changed — would alter the train pool. Aborting.")
    if set(new_val) & set(new_test):
        sys.exit("FATAL val_repartition: val∩test non-empty after the move — splits must stay disjoint.")
    _atomic_write(val_path, new_val,
                  {"source": "val_repartition", "n_motion_classes": n_cls, "per_class": per_class})
    _atomic_write(test_path, new_test, {"source": "val_repartition_test_merge"})
    print(f"  [val-repartition] val {len(val_keys)} → {len(new_val)} (stratified · {n_cls} motion "
          f"classes × ~{per_class}/class) · test {len(test_keys)} → {len(new_test)} "
          f"(+{len(leftover)} moved) · val∪test INVARIANT held → train pool unchanged.", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Stratified val subsample + move the leftover val clips into test (train-pool-preserving).")
    ap.add_argument("--val-split", required=True, help="val_split.json to shrink (rewritten in place).")
    ap.add_argument("--test-split", required=True, help="test_split.json to grow (rewritten in place).")
    ap.add_argument("--motion-features", required=True,
                    help="<local_data>/m04d_motion_features/motion_features.npy (+ sibling .paths.npy).")
    ap.add_argument("--target-val", type=int, required=True,
                    help="target val size (= validation.max_val_clips; literature plateau ~500-1000).")
    args = ap.parse_args()
    repartition(Path(args.val_split), Path(args.test_split), args.motion_features, args.target_val)


if __name__ == "__main__":
    main()
