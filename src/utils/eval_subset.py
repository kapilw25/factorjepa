"""Stratified eval subset generator. CPU-only.

Picks N clips per **POV class** (walking/driving/drone — derived from clip_key
path prefix) from a source eval JSON, writing a derived JSON that keeps the
same path + TAR layout. POV diversity is preserved so any subset (SANITY,
ablation slice, debugging cut) shares codec/path/motion-pattern coverage with
FULL.

iter13 v12 (2026-05-06): the downstream probe_action.py now derives 16
optical-flow MOTION classes (magnitude × direction) from m04d, NOT this 3-class
POV. Why we still stratify by POV here:
  - POV class is path-derived → no GPU/m04d dependency; eval_subset.py runs
    BEFORE m04d in the orchestration order.
  - POV diversity (handheld walking / forward-sweep driving / aerial drone)
    naturally spreads the resulting 16-class motion distribution — drone clips
    populate fast__*, walking clips populate slow/still__*, etc.
  - SANITY's job is code-correctness validation; full motion-class coverage is
    FULL eval's job (10K clips, all 16 classes naturally represented).
The legacy `utils.action_labels.PATH_TO_CLASS_3CLASS` constant was deleted in
iter13 v12; this module now owns its own POV mapping (no cross-import).

USAGE (called by scripts/run_probe_eval.sh under MODE=SANITY; also standalone):
    python -u src/utils/eval_subset.py \\
        --eval-subset data/eval_10k.json \\
        --n-per-class 200 \\
        --output data/eval_10k_sanity.json

  Sizing notes (post-iter13-v12):
    - 200/POV × 3 = 600 clips → ~37 clips/motion-class avg → splits cleanly
      under run_probe_eval.sh's SANITY defaults (MIN_CLIPS_PER_CLASS=5,
      MIN_PER_SPLIT=1) even after several rare motion classes get filtered.
    - Lowering below ~150/POV (450 clips) risks all motion classes being
      filtered → action_labels.load_subset_with_labels FATALs.
"""
import argparse
import json
import sys
from pathlib import Path


# Path-prefix → POV class. Self-contained (no cross-import from action_labels;
# the legacy PATH_TO_CLASS_3CLASS constant there was deleted in iter13 v12).
PATH_TO_CLS = {
    "walking": "walking",
    "rain":    "walking",   # tier2 rain bucket = rainy walking tour
    "drive":   "driving",
    "drone":   "drone",
}

CLASSES = ("walking", "driving", "drone")


def _path_class(clip_key: str):
    """Return the POV class (walking|driving|drone) for a clip_key, or None to
    skip (monuments/* — no POV derivable; or unrecognized path prefix).
    """
    p = clip_key.split("/")
    if not p:
        return None
    if p[0] == "monuments":
        return None
    if p[0] in ("tier1", "tier2") and len(p) > 2:
        return PATH_TO_CLS.get(p[2])
    if p[0] == "goa" and len(p) > 1:
        return PATH_TO_CLS.get(p[1])
    return None


def stratified_subset(src: dict, n_per_class: int) -> dict:
    """Pure-function subset builder — takes the loaded source JSON dict + n_per_class,
    returns a new dict ready to json.dump. Order preserved within each class.
    """
    if n_per_class < 1:
        raise ValueError(f"n_per_class must be >= 1 (got {n_per_class})")
    keys = src["clip_keys"]   # fail-loud — no .get(default)
    buckets = {c: [] for c in CLASSES}
    for k in keys:
        cls = _path_class(k)
        if cls and len(buckets[cls]) < n_per_class:
            buckets[cls].append(k)
    out = dict(src)
    out["clip_keys"] = sum((buckets[c] for c in CLASSES), [])
    out["n_clips"] = len(out["clip_keys"])
    out["per_class_counts"] = {c: len(buckets[c]) for c in CLASSES}
    return out


def main():
    p = argparse.ArgumentParser(
        description="Generate a stratified eval subset (N per action class) from an eval JSON.")
    p.add_argument("--eval-subset", type=Path, required=True,
                   help="Source eval JSON (e.g. data/eval_10k.json).")
    p.add_argument("--n-per-class", type=int, required=True,
                   help="Clips to keep per action class.")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write the derived subset JSON.")
    args = p.parse_args()

    if not args.eval_subset.exists():
        sys.exit(f"FATAL: --eval-subset not found: {args.eval_subset}")

    src = json.loads(args.eval_subset.read_text())
    out = stratified_subset(src, args.n_per_class)
    out["source"] = f"stratified_subset_{args.n_per_class}_per_class_of_{args.eval_subset.name}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Generated {args.output}: {out['n_clips']} clips ({out['per_class_counts']})")


if __name__ == "__main__":
    main()
