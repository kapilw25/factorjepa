"""m04f — Taxonomy (VLM-tag) label derivation (ANNOTATION band). CPU-only.

Strict pipeline direction (iter16 §3.2): the `--stage labels` step of the old
probe_taxonomy.py is an ANNOTATION step (derives taxonomy_labels.json from VLM tags +
the tag_taxonomy spec) — homed here in the m04 band. The downstream EVAL probe
(train/paired_delta/plot over 15 taxonomy dims) lives in src/m12c_taxonomy_f1.py. Both
single-source the derivation via utils.taxonomy_labels — this module is a thin
orchestrator, NOT a re-implementation.

Writes <output-root>/taxonomy_labels.json = {dims, labels} where
  dims:   {dim_name: {type, values, n_classes, default}}   (13 single + 2 multi-label)
  labels: {clip_key: {dim_name: int | multi-hot list}}

USAGE:
    python -u src/m04f_taxonomy_labels.py --FULL \\
        --tags-json data/eval_10k_local/tags.json \\
        --tag-taxonomy configs/tag_taxonomy.json \\
        --eval-subset data/eval_10k_local/eval_10k.json \\
        --output-root outputs/full/probe_taxonomy \\
        --cache-policy 1 2>&1 | tee logs/m04f_taxonomy_labels_$(date +%Y%m%d_%H%M%S).log
"""
import argparse
import sys
from pathlib import Path

from utils.cache_policy import (
    add_cache_policy_arg, guarded_delete, resolve_cache_policy_interactive,
)
from utils.checkpoint import save_json_checkpoint
from utils.taxonomy_labels import derive_taxonomy_labels
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)


def run_labels(args, wb) -> None:
    if any(p is None for p in (args.tags_json, args.tag_taxonomy, args.eval_subset)):
        sys.exit("FATAL: m04f requires --tags-json + --tag-taxonomy + --eval-subset")
    args.output_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_root / artifact("taxonomy_labels")
    if out_path.exists() and args.cache_policy == "1":
        print(f"  [keep] {out_path} present — skipping (--cache-policy 2 to redo)")
        return
    guarded_delete(out_path, args.cache_policy, artifact("taxonomy_labels"))

    labels_by_clip, dims = derive_taxonomy_labels(
        args.tags_json, args.tag_taxonomy, args.eval_subset)
    save_json_checkpoint({"dims": dims, "labels": labels_by_clip}, out_path)
    print(f"Wrote: {out_path}  ({len(dims)} dims, {len(labels_by_clip)} labeled clips)")
    log_metrics(wb, {"n_clips_labeled": len(labels_by_clip), "n_dims": len(dims)})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="m04f — taxonomy VLM-tag label derivation (annotation)")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--tags-json", type=Path, default=None)
    p.add_argument("--tag-taxonomy", type=Path, default=None)
    p.add_argument("--eval-subset", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
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
    wb = init_wandb("m04f_taxonomy_labels", mode, config=vars(args), enabled=not args.no_wandb)
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
