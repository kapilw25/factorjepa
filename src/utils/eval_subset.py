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

USAGE (called by scripts/run_eval.sh under MODE=SANITY; also standalone):
    python -u src/utils/eval_subset.py \\
        --eval-subset data/eval_10k.json \\
        --n-per-class 200 \\
        --output data/eval_10k_sanity.json

  Sizing notes (post-iter13-v12):
    - 200/POV × 3 = 600 clips → ~37 clips/motion-class avg → splits cleanly
      under run_eval.sh's SANITY defaults (MIN_CLIPS_PER_CLASS=5,
      MIN_PER_SPLIT=1) even after several rare motion classes get filtered.
    - Lowering below ~150/POV (450 clips) risks all motion classes being
      filtered → action_labels.load_subset_with_labels FATALs.
"""
import argparse
import json
import sys
from pathlib import Path

# Make sibling modules under src/utils/ importable when this file is invoked as a
# script (`python -u src/utils/eval_subset.py ...`). Without this, sys.path[0]
# would be src/utils/ and `from utils.action_labels import ...` (used inside
# stratified_by_motion_class_subset) would fail. No-op when imported as module.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


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


def first_n_subset(src: dict, n: int) -> dict:
    """First-N subset builder — takes the loaded source JSON dict + n total clips,
    returns a new dict with the FIRST n clip_keys verbatim (no stratification —
    downstream stratified_split applies). iter14 POC mode (2026-05-08).

    DEPRECATED 2026-05-09: caused 7-class POC label files (iter14 D₂ bug). Use
    stratified_by_motion_class_subset for POC instead. Kept for backward compat.
    """
    if n < 1:
        raise ValueError(f"--first-n must be >= 1 (got {n})")
    keys = src["clip_keys"]
    out = dict(src)
    out["clip_keys"] = list(keys[:n])
    out["n_clips"] = len(out["clip_keys"])
    return out


def stratified_by_motion_class_subset(src: dict, motion_features_path,
                                       target_per_class: int) -> dict:
    """Stratified subset by m04d optical-flow MOTION class — guarantees POC label
    file matches FULL schema (all 8 motion classes preserved).

    Replaces first_n_subset for POC. The legacy first-N path took clip_keys
    verbatim and could drop rare classes → POC labels file ends up with 7
    classes (vs FULL's 8) → motion_aux head is sized wrong → POC ↔ FULL parity
    violated (cf. src/CLAUDE.md POC↔FULL parity rule).

    Args:
        src: loaded eval JSON dict with "clip_keys" list.
        motion_features_path: m04d motion_features.npy (FULL-scale, ~9276 × 23D post-Phase-0).
            Must have a sibling motion_features.paths.npy.
        target_per_class: pick min(target_per_class, available) clips per motion
            class. Total output clips = sum(picked[c] for c in classes).

    Returns:
        out: dict with "clip_keys" replaced by stratified pick, plus
        "per_motion_class_counts" and "n_motion_classes" metadata.
    """
    if target_per_class < 1:
        raise ValueError(f"target_per_class must be >= 1 (got {target_per_class})")
    keys = src["clip_keys"]

    # Defer heavyweight imports — only this code path needs them. Keeps the
    # legacy POV-stratified path numpy-free.
    import numpy as np
    from utils.action_labels import (compute_magnitude_quartiles,
                                      parse_optical_flow_class)

    motion_features_path = Path(motion_features_path)
    paths_path = motion_features_path.with_name(
        motion_features_path.stem + ".paths.npy")
    if not motion_features_path.exists():
        sys.exit(
            f"FATAL: motion_features.npy not found at {motion_features_path}.\n"
            f"  Run m04d first: python -u src/m04d_motion_features.py --FULL "
            f"--subset {src.get('source', '<eval-subset>')} ..."
        )
    if not paths_path.exists():
        sys.exit(f"FATAL: motion_features.paths.npy not found at {paths_path} "
                 f"(must be next to motion_features.npy)")
    flow_features = np.load(motion_features_path)                 # (N, 23) post-Phase-0
    flow_paths = np.load(paths_path, allow_pickle=True)           # (N,) clip keys
    if flow_features.shape[0] != flow_paths.shape[0]:
        sys.exit(f"FATAL: motion_features rows={flow_features.shape[0]} != "
                 f"paths rows={flow_paths.shape[0]}")
    flow_features_by_key = {str(k): flow_features[i]
                            for i, k in enumerate(flow_paths)}

    # Global quartiles — same cut-points as load_subset_with_labels uses, so
    # POC's class assignment is identical to FULL's.
    quartiles = compute_magnitude_quartiles(flow_features)

    # Pass 1 — bucket eval-subset clips by motion class (sorted clip_key order
    # within each bucket → deterministic across runs)
    buckets: dict = {}
    n_no_record = 0
    for k in sorted(keys):
        cls = parse_optical_flow_class(k, flow_features_by_key, quartiles)
        if cls is None:
            n_no_record += 1
            continue
        buckets.setdefault(cls, []).append(k)

    if not buckets:
        sys.exit(
            f"FATAL: no clips in --eval-subset matched motion_features. "
            f"src={len(keys)} clips · motion_features rows={flow_features.shape[0]}. "
            f"Likely cause: motion_features.npy was generated for a different "
            f"eval pool. Re-run m04d on the current --eval-subset."
        )
    if n_no_record:
        print(f"  [stratified-motion] {n_no_record}/{len(keys)} clip_keys had "
              f"no motion_features record (skipped)")

    # Pass 2 — pick min(target_per_class, available) per bucket
    picked = {cls: clips[:target_per_class] for cls, clips in buckets.items()}

    out = dict(src)
    out["clip_keys"] = sum((picked[c] for c in sorted(picked)), [])
    out["n_clips"] = len(out["clip_keys"])
    out["per_motion_class_counts"] = {c: len(picked[c]) for c in sorted(picked)}
    out["n_motion_classes"] = len(picked)
    return out


def main():
    p = argparse.ArgumentParser(
        description="Generate a derived eval subset from a source eval JSON. "
                    "THREE modes: "
                    "(a) --n-per-class N → POV-stratified (SANITY); "
                    "(b) --stratified-by-motion-class → motion-class-stratified "
                    "    (POC, iter14, fixes 855/7-class bug); "
                    "(c) --first-n N → first-N verbatim (DEPRECATED, kept for back-compat).")
    p.add_argument("--eval-subset", type=Path, required=True,
                   help="Source eval JSON (e.g. data/eval_10k.json).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--n-per-class", type=int, default=None,
                   help="Stratified-by-POV mode: clips to keep per POV class "
                        "(walking/driving/drone). Used by SANITY.")
    g.add_argument("--first-n", type=int, default=None,
                   help="First-N mode: take the first N clip_keys verbatim. "
                        "DEPRECATED 2026-05-09 (caused iter14 D₂ 855/7 bug).")
    g.add_argument("--stratified-by-motion-class", action="store_true",
                   help="Stratified-by-motion-class mode (POC, iter14): use m04d "
                        "motion_features.npy to bucket clips by RAFT optical-flow "
                        "MOTION class, pick --target-per-class from each. "
                        "Guarantees POC labels match FULL schema (8 classes).")
    p.add_argument("--motion-features", type=Path, default=None,
                   help="Required when --stratified-by-motion-class is set. "
                        "Path to m04d motion_features.npy "
                        "(must have sibling motion_features.paths.npy).")
    p.add_argument("--target-per-class", type=int, default=None,
                   help="Required when --stratified-by-motion-class is set. "
                        "Per-class clip target (total ≈ target × n_motion_classes).")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write the derived subset JSON.")
    args = p.parse_args()

    if not args.eval_subset.exists():
        sys.exit(f"FATAL: --eval-subset not found: {args.eval_subset}")

    src = json.loads(args.eval_subset.read_text())

    if args.n_per_class is not None:
        out = stratified_subset(src, args.n_per_class)
        out["source"] = f"stratified_subset_{args.n_per_class}_per_class_of_{args.eval_subset.name}"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2))
        print(f"Generated {args.output}: {out['n_clips']} clips ({out['per_class_counts']})")
    elif args.stratified_by_motion_class:
        if args.motion_features is None or args.target_per_class is None:
            sys.exit("FATAL: --stratified-by-motion-class requires both "
                     "--motion-features and --target-per-class")
        out = stratified_by_motion_class_subset(
            src, args.motion_features, args.target_per_class)
        out["source"] = (f"stratified_by_motion_class_{args.target_per_class}_"
                          f"per_class_of_{args.eval_subset.name}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2))
        print(f"Generated {args.output}: {out['n_clips']} clips · "
              f"{out['n_motion_classes']} motion classes · "
              f"per-class counts: {out['per_motion_class_counts']}")
    else:
        out = first_n_subset(src, args.first_n)
        out["source"] = f"first_{args.first_n}_clips_of_{args.eval_subset.name}"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2))
        print(f"Generated {args.output}: first {out['n_clips']} clips of {args.eval_subset.name}")


if __name__ == "__main__":
    main()
