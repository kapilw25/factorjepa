"""m04e — Action (motion-flow) label derivation (ANNOTATION band). CPU-only.

Strict pipeline direction (iter16 §3.2): the `--stage labels` step of the old
probe_action.py was an ANNOTATION step (derives action_labels.json, consumed BOTH by
m09 TRAINING and by m12a_action EVALUATION) — so it is homed here in the m04 band,
BEFORE training. The downstream EVAL probe (features/train/paired_delta) lives in
src/m12a_action_top1.py. Both single-source the derivation via utils.action_labels — this
module is a thin orchestrator, NOT a re-implementation.

Derives per-clip MOTION-flow classes (RAFT optical-flow → <magnitude>__<direction>,
filtered to ≥min_clips_per_class) + a video-disjoint stratified train/val/test split
(utils.action_labels.stratified_split, StratifiedGroupKFold). Writes:
  <output-root>/action_labels.json   {clip_key: {class, class_id, split}}
  <output-root>/class_counts.json    {class: {train,val,test}}

USAGE:
    python -u src/m04e_action_labels.py --FULL \\
        --eval-subset data/eval_10k_local/eval_10k.json \\
        --motion-features data/eval_10k_local/m04d_motion_features/motion_features.npy \\
        --output-root outputs/full/probe_action \\
        --cache-policy 1 2>&1 | tee logs/m04e_action_labels_$(date +%Y%m%d_%H%M%S).log

Gold: motion-flow label scheme — iter13 v12 plan_code_dev.md Phase 2.
"""
import argparse
import json
import sys
from pathlib import Path

from utils.action_labels import (
    load_subset_with_labels,
    stratified_split,
    subsample_manifest_for_mode,
    write_action_labels_json,
)
from utils.cache_policy import (
    add_cache_policy_arg,
    guarded_delete,
    resolve_cache_policy_interactive,
)
from utils.checkpoint import load_json_checkpoint, save_json_checkpoint
from utils.config import get_pipeline_config, get_probe_split
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)

# iter18 W7 (PLR2004): semantic named constants.
_MIN_TRAIN_PER_CLASS = 30   # class viability floor for the probe

_PROBE_CFG = get_pipeline_config()["probe"]


def run_labels(args, wb) -> None:
    if args.eval_subset is None:
        sys.exit("FATAL: m04e requires --eval-subset")
    if args.motion_features is None:
        sys.exit("FATAL: m04e requires --motion-features "
                 "(m04d's motion_features.npy — see plan_code_dev.md Phase 2)")
    args.output_root.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_root / artifact("action_labels")
    if labels_path.exists() and args.cache_policy == "1":
        print(f"  [keep] {labels_path} present — skipping (--cache-policy 2 to redo)")
        return
    guarded_delete(labels_path, args.cache_policy, artifact("action_labels"))
    guarded_delete(args.output_root / artifact("class_counts"), args.cache_policy, artifact("class_counts"))

    # Mode-keyed clip-pool subsampling (POC↔FULL parity; SANITY = sorted()[:n] code check).
    mode = "sanity" if args.SANITY else ("poc" if args.POC else "full")
    # iter18 (2026-06-20): single-source the class-filter floors from the CANONICAL mode-keyed
    # probe_action_labels[mode] (same as probe_labels.py + run_eval.sh) when not overridden — was the
    # stale scalar probe.min_clips_per_class:34 that conflicted with poc/full=100 and failed standalone.
    _pal = get_pipeline_config()["probe_action_labels"]
    if args.min_clips_per_class is None:
        args.min_clips_per_class = _pal["min_clips_per_class"][mode]
    if args.min_per_split is None:
        args.min_per_split = _pal["min_per_split"][mode]
    manifest = json.loads(Path(args.eval_subset).read_text())
    n_motion_classes = get_pipeline_config()["probe_action_labels"]["n_motion_classes"]
    pool_keys = subsample_manifest_for_mode(
        mode, manifest["clip_keys"], args.motion_features, n_motion_classes)
    print(f"[M1 Opt X] mode={mode}: subsampled {len(pool_keys):,}/"
          f"{len(manifest['clip_keys']):,} clips "
          f"({len(pool_keys)/len(manifest['clip_keys']):.2%})")

    # iter18 cross-set: --class-edges <eval_10k class_edges.json> → REUSE eval_10k's motion-bin
    # quartiles + class→id ordering so a reused head (Stage B) scores against the SAME class
    # definition. Without it, m04e DERIVES fresh edges + writes class_edges.json (the reusable source).
    loaded_edges = json.loads(Path(args.class_edges).read_text()) if args.class_edges else None
    edges_capture = None if loaded_edges else {}
    records, class_names = load_subset_with_labels(
        args.eval_subset, args.motion_features,
        min_clips_per_class=args.min_clips_per_class,
        clip_keys=pool_keys,
        magnitude_quartiles=(loaded_edges["magnitude_quartiles"] if loaded_edges else None),
        class_names_override=(loaded_edges["class_names"] if loaded_edges else None),
        edges_out=edges_capture)
    print(f"Loaded {len(records)} labeled clips from {args.eval_subset} "
          f"({len(class_names)} motion-flow classes)"
          + (f" — REUSED edges from {args.class_edges}" if loaded_edges else ""))

    split_cfg = get_probe_split(mode)
    splits = stratified_split(records,
                              train_pct=split_cfg["train_pct"],
                              val_pct=split_cfg["val_pct"],
                              seed=args.seed,
                              min_per_split=args.min_per_split,
                              mode=mode,
                              test_all=(args.probe_split == "test-all"))
    write_action_labels_json(records, splits, labels_path)
    if edges_capture is not None:   # derived fresh → persist as the reusable cross-set edge source
        save_json_checkpoint(edges_capture, args.output_root / artifact("class_edges"))
        print(f"  [class-edges] wrote {args.output_root / artifact('class_edges')} "
              f"(quartiles + {len(class_names)} class_names) — reuse via --class-edges for cross-set eval")

    counts = load_json_checkpoint(args.output_root / artifact("class_counts"))
    if args.probe_split == "test-all":
        # test-all → every clip is TEST (train/val empty by design) → the per-split min_per_split
        # asserts don't apply. The probe-TRAIN stages (m12a/m12c Stage 3/3.5) have NO train data in
        # this mode and MUST be skipped; their heads are reused from eval_10k via --head-ckpt.
        n_test = sum(c["test"] for c in counts.values())
        print(f"  [probe-split=test-all] every clip → TEST (train=0, val=0, test={n_test}); "
              f"per-split min_per_split asserts SKIPPED.")
    else:
        for cls, c in counts.items():
            if c["test"] < args.min_per_split or c["val"] < args.min_per_split:
                if mode == "sanity":
                    print(f"  WARN [SANITY]: class '{cls}' val={c['val']}/"
                          f"test={c['test']} (need >= {args.min_per_split} each "
                          f"for POC/FULL; SANITY only validates code paths)")
                else:
                    sys.exit(f"FATAL: class '{cls}' val={c['val']}/test={c['test']} "
                             f"(need >= {args.min_per_split} each)")
            if c["train"] < _MIN_TRAIN_PER_CLASS:
                print(f"  WARN: class '{cls}' train={c['train']} (recommended >=30)")
    log_metrics(wb, {"n_clips_labeled": len(records), "n_classes": len(counts)})
    print(f"Wrote: {labels_path}  +  class_counts.json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m04e — action motion-flow label derivation (annotation)")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--eval-subset", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--motion-features", type=Path, default=None,
                   help="m04d motion_features.npy (RAFT optical-flow 23D × N_clips). "
                        "Companion .paths.npy must sit next to it.")
    p.add_argument("--min-clips-per-class", type=int, default=None,
                   help="Drop motion-flow classes with fewer than this many clips. Default (None) → "
                        "the mode-keyed probe_action_labels.min_clips_per_class[mode] (single source; "
                        "sanity 5 / poc 100 / full 100).")
    p.add_argument("--min-per-split", type=int, default=None,
                   help="Min clips per split per class for the BCa CI floor. Default (None) → "
                        "probe_action_labels.min_per_split[mode] (sanity 1 / poc 5 / full 5).")
    p.add_argument("--seed", type=int, default=_PROBE_CFG["seed"])
    p.add_argument("--probe-split", choices=["stratified", "test-all"], default="stratified",
                   help="'stratified' (default): video-disjoint train/val/test (training + standard "
                        "eval). 'test-all': every clip → the TEST split (train/val empty) for cross-set "
                        "CI-reduction eval on a FULL disjoint corpus (iter18); the probe-TRAIN stages "
                        "must then be skipped (no train data) — heads are reused via --head-ckpt.")
    p.add_argument("--class-edges", type=Path, default=None,
                   help="iter18 cross-set: reuse another corpus's motion-bin definition "
                        "(eval_10k's class_edges.json = {magnitude_quartiles, class_names}) so a "
                        "reused probe head (Stage B) scores against the SAME classes. Default (None) "
                        "→ derive fresh + write class_edges.json (the reusable source).")
    add_cache_policy_arg(p)
    add_wandb_args(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        sys.exit("ERROR: specify --SANITY, --POC, or --FULL")
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)
    args.output_root.mkdir(parents=True, exist_ok=True)
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb = init_wandb("m04e_action_labels", mode, config=vars(args), enabled=not args.no_wandb)
    try:
        run_labels(args, wb)
    finally:
        finish_wandb(wb)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        print(f"\n❌ FATAL: {Path(__file__).name} crashed — see traceback below", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
