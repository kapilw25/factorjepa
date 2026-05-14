"""Extract a per-split subset JSON from probe's action_labels.json. CPU-only.

probe Stage 1 (probe_action.run_labels_stage) produces action_labels.json:
    {"<clip_key>": {"class": "...", "class_id": N, "split": "train"|"val"|"test"}, ...}

P2 (m09a1_pretrain_encoder) and P3 (m09c1_surgery_encoder) need a flat {"clip_keys": [...]} subset
JSON to feed their --subset flag. This util reads action_labels.json, filters by
split, and writes the subset. Train-split → encoder training; val-split →
training-time validation; test-split → held out for probe Stage 4 paired-Δ gate.

USAGE (called by scripts/run_train.sh; also a standalone CLI):
    python -u src/utils/probe_train_subset.py \\
        --action-labels outputs/full/probe_action/action_labels.json \\
        --split train \\
        --output data/eval_10k_train_split.json
"""
import argparse
import json
import sys
from pathlib import Path


VALID_SPLITS = ("train", "val", "test")


def split_subset(action_labels: dict, split: str) -> dict:
    """Filter action_labels by split, return a flat {"clip_keys": [...]} dict."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS} (got {split!r})")
    keys = [k for k, v in action_labels.items() if v["split"] == split]
    return {
        "split": split,
        "clip_keys": keys,
        "n_clips": len(keys),
    }


def main():
    p = argparse.ArgumentParser(
        description="Extract a per-split subset JSON from probe's action_labels.json.")
    p.add_argument("--action-labels", type=Path, required=True,
                   help="probe Stage 1 output (e.g. outputs/full/probe_action/action_labels.json).")
    p.add_argument("--split", type=str, required=True, choices=VALID_SPLITS,
                   help="Which split to extract.")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to write the derived subset JSON.")
    args = p.parse_args()

    if not args.action_labels.exists():
        sys.exit(f"FATAL: --action-labels not found: {args.action_labels}")

    labels = json.loads(args.action_labels.read_text())
    out = split_subset(labels, args.split)
    out["source"] = f"probe_train_subset_{args.split}_of_{args.action_labels.name}"

    if out["n_clips"] == 0:
        sys.exit(f"FATAL: 0 clips in split={args.split!r} — labels file may be empty or split name mismatched")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"Generated {args.output}: {out['n_clips']} clips (split={args.split})")


if __name__ == "__main__":
    main()
