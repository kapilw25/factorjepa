"""Stratified eval subset generator. CPU-only.

Picks N clips per action class from a source eval JSON, writing a derived JSON
that keeps the same path semantics + TAR layout as the source — so any subset
(SANITY, ablation slice, debugging cut) shares codec/path semantics with FULL.

USAGE (called by scripts/run_m06d.sh under MODE=SANITY; also a standalone CLI):
    python -u src/utils/eval_subset.py \\
        --eval-subset data/eval_10k.json \\
        --n-per-class 50 \\
        --output data/eval_10k_sanity.json

  Note: 50/class × 3 classes = 150 clips total. With 70/15/15 stratified split
  this gives 35/7/8 per class — clears action_labels.stratified_split's >=5/split
  floor with a 2-clip margin. Lowering below ~34/class WILL fail that floor.
"""
import argparse
import json
import sys
from pathlib import Path


# Path-prefix → semantic class. Mirrors utils/action_labels.PATH_TO_CLASS_3CLASS.
# Kept as a module-level constant so importers (and tests) can introspect it
# without having to re-derive from action_labels (avoids cyclic import risk).
PATH_TO_CLS = {
    "walking": "walking",
    "rain":    "walking",   # tier2 rain bucket = rainy walking tour
    "drive":   "driving",
    "drone":   "drone",
}

CLASSES = ("walking", "driving", "drone")


def _path_class(clip_key: str):
    """Return the semantic class for a clip_key, or None if it should be skipped
    (e.g. monuments/* in 3-class mode, unrecognized prefix).
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
