"""Per-category difficulty subset builder for iter11 v3 hard-tier pivot. CPU-only.

Builds 9 subset JSONs in data/ from ALL 115K clips (no val_1k/eval_10k exclusion —
each category gets its OWN train/val/eval splits, disjoint by seed). Each subset
matches the data/subset_10k.json schema (clip_keys list of canonical paths).

For the headline iter11 v3 ultra_hard subset, ALSO emits 3 sub-splits:
    ultra_hard_<N>_train.json  (default 2452 clips — m09c training input)
    ultra_hard_<N>_val.json    (default 306 clips — mid-training probe)
    ultra_hard_<N>_eval.json   (default 308 clips — paired-BCa decision gate)
All 3 disjoint, seed=42 deterministic. Sum must equal --ultra-target-n (default 3066,
the actual count of clips meeting ≥4 hard triggers AND ≥4 Indian-specific objects).

USAGE:
    python -u src/m00f_category_subsets.py \\
        --tags data/full_local/tags.json \\
        --output-dir data/ \\
        --ultra-target-n 3066 \\
        --ultra-train-n 2452 --ultra-val-n 306 --ultra-eval-n 308 \\
        2>&1 | tee logs/m00f_category_subsets.log
"""
import argparse
import json
import random
import sys
from pathlib import Path

# Indian-specific subset of `notable_objects` (9 of 15 in tag_taxonomy.json's notable_objects).
# Excludes universal {bus, bike, car, truck, pedestrian, skyscraper}.
INDIAN_SPECIFIC = {
    "auto_rickshaw", "cycle_rickshaw", "handcart", "street_vendor",
    "sacred_cow", "stray_dog", "religious_shrine", "overhead_wires", "signage",
}
CONFIDENCE_FLOOR = 0.7


def is_high_conf(t: dict, field: str) -> bool:
    return t.get(f"confidence_{field}", 0.0) >= CONFIDENCE_FLOOR


def n_indian(t: dict) -> int:
    return sum(1 for o in (t.get("notable_objects") or []) if o in INDIAN_SPECIFIC)


def hard_trigger_count(t: dict) -> int:
    """Count of {0..5} Hard composite triggers firing for this clip (m00e definition)."""
    n = 0
    if t.get("crowd_density") == "high" and is_high_conf(t, "crowd_density"):
        n += 1
    if t.get("traffic_density") == "high" and is_high_conf(t, "traffic_density"):
        n += 1
    if t.get("traffic_mix") in {"mixed_all", "pedestrian_dominant"} and is_high_conf(t, "traffic_mix"):
        n += 1
    if t.get("road_encroachment") == "heavy" and is_high_conf(t, "road_encroachment"):
        n += 1
    if n_indian(t) >= 3:
        n += 1
    return n


def clip_key(t: dict) -> str:
    return f"{t['section']}/{t['video_id']}/{t['source_file']}"


# Categories — 8 single-condition (no expected counts: full-corpus run, no exclusion).
CATEGORIES = [
    ("ge3_indian_objects",
     "N(Indian-specific notable_objects) >= 3",
     lambda t: n_indian(t) >= 3),
    ("crowd_high",
     "crowd_density=high AND confidence>=0.7",
     lambda t: t.get("crowd_density") == "high" and is_high_conf(t, "crowd_density")),
    ("traffic_mix_pedestrian_dominant",
     "traffic_mix=pedestrian_dominant AND confidence>=0.7",
     lambda t: t.get("traffic_mix") == "pedestrian_dominant" and is_high_conf(t, "traffic_mix")),
    ("traffic_high",
     "traffic_density=high AND confidence>=0.7",
     lambda t: t.get("traffic_density") == "high" and is_high_conf(t, "traffic_density")),
    ("traffic_mix_mixed_all",
     "traffic_mix=mixed_all AND confidence>=0.7",
     lambda t: t.get("traffic_mix") == "mixed_all" and is_high_conf(t, "traffic_mix")),
    ("ge4_indian_objects",
     "N(Indian-specific notable_objects) >= 4",
     lambda t: n_indian(t) >= 4),
    ("road_encroachment_heavy",
     "road_encroachment=heavy AND confidence>=0.7",
     lambda t: t.get("road_encroachment") == "heavy" and is_high_conf(t, "road_encroachment")),
    ("ge5_indian_objects",
     "N(Indian-specific notable_objects) >= 5 (long tail 5..9)",
     lambda t: n_indian(t) >= 5),
]


def write_subset(out_path: Path, name: str, criterion: str, matched: list,
                 source_tags: str, exclusions: list, n_excluded: int) -> None:
    out = {
        "n": len(matched),
        "tier": name,
        "criterion": criterion,
        "source_tags": source_tags,
        "exclusions": exclusions,
        "n_excluded": n_excluded,
        "indian_specific_objects": sorted(INDIAN_SPECIFIC),
        "confidence_floor": CONFIDENCE_FLOOR,
        "clip_keys": [clip_key(t) for t in matched],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", type=str, required=True,
                        help="m04 tags.json (LIST of per-clip tag dicts)")
    parser.add_argument("--exclude", type=str, nargs="*", default=[],
                        help="(optional) Subset JSONs whose clip_keys are EXCLUDED. "
                             "Default: empty (split all 115K). iter11 v3 splits ALL clips "
                             "and carves train/val/eval inside the ultra_hard category.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to write subset JSONs")
    parser.add_argument("--ultra-target-n", type=int, default=3066,
                        help="Top-N kept for ultra_hard subset (default 3066 = "
                             "actual count meeting ≥4 hard triggers AND ≥4 Indian-specific objects)")
    parser.add_argument("--ultra-train-n", type=int, default=2452,
                        help="ultra_hard training split size (default 2452 = 80% of 3066)")
    parser.add_argument("--ultra-val-n", type=int, default=306,
                        help="ultra_hard mid-training probe split (default 306 = 10% of 3066)")
    parser.add_argument("--ultra-eval-n", type=int, default=308,
                        help="ultra_hard paired-BCa decision-gate split (default 308 = remainder)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true",
                        help="(no-op flag for pipeline consistency)")
    args = parser.parse_args()

    if args.ultra_train_n + args.ultra_val_n + args.ultra_eval_n != args.ultra_target_n:
        raise SystemExit(
            f"FATAL: --ultra-train-n + --ultra-val-n + --ultra-eval-n "
            f"({args.ultra_train_n}+{args.ultra_val_n}+{args.ultra_eval_n}="
            f"{args.ultra_train_n + args.ultra_val_n + args.ultra_eval_n}) "
            f"must equal --ultra-target-n ({args.ultra_target_n})")

    tags_path = Path(args.tags)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tags_path.is_file():
        raise SystemExit(f"FATAL: tags not found: {tags_path}")

    # 1) Load tags + exclusion clip_keys
    print(f"Loading {tags_path} ...")
    tags = json.load(open(tags_path))
    if not isinstance(tags, list):
        raise SystemExit(f"FATAL: {tags_path} is type {type(tags).__name__}; expected list")
    print(f"  {len(tags)} clip tags loaded")

    excl: set[str] = set()
    for p in args.exclude:
        d = json.load(open(p))
        keys = d.get("clip_keys") if isinstance(d, dict) else None
        if not keys:
            raise SystemExit(f"FATAL: {p} has no clip_keys list")
        excl.update(keys)
        print(f"  + excluding {len(keys)} keys from {p}")
    print(f"  total excluded: {len(excl)}")

    # 2) Filter to eligible candidates (disjoint from val_1k + eval_10k)
    candidates = [t for t in tags if clip_key(t) not in excl]
    print(f"  eligible candidates: {len(candidates)} (= {len(tags)} − {len(tags) - len(candidates)})")
    print()

    # 3) Build 8 single-condition subsets — print counts (no expected check; full-corpus run)
    print(f"{'category':>35} | {'count':>8}")
    print("-" * 50)
    for name, criterion, pred in CATEGORIES:
        matched = [t for t in candidates if pred(t)]
        n = len(matched)
        print(f"{name:>35} | {n:>8}")
        write_subset(
            out_dir / f"{name}.json", name, criterion, matched,
            source_tags=str(tags_path), exclusions=args.exclude, n_excluded=len(excl),
        )

    # 4) Build ultra_hard_<N>: intersection of ≥4 triggers AND ≥4 Indian-specific objects
    print()
    ultra_raw = [t for t in candidates
                 if hard_trigger_count(t) >= 4 and n_indian(t) >= 4]
    ultra_sorted = sorted(ultra_raw,
                          key=lambda t: (-hard_trigger_count(t), -n_indian(t)))
    ultra_kept = ultra_sorted[:args.ultra_target_n]
    print(f"ultra_hard intersection (≥4 triggers AND ≥4 Indian objects): "
          f"{len(ultra_raw)} raw → top-{len(ultra_kept)} kept")

    out_path = out_dir / f"ultra_hard_{args.ultra_target_n}.json"
    out = {
        "n": len(ultra_kept),
        "tier": f"ultra_hard_{args.ultra_target_n}",
        "criterion": ("INTERSECTION: hard_trigger_count>=4 AND "
                      "N(Indian-specific notable_objects)>=4 (all conf>=0.7); "
                      "ranked by (n_triggers, n_indian) desc; top-N kept"),
        "raw_candidate_count": len(ultra_raw),
        "ultra_target_n": args.ultra_target_n,
        "source_tags": str(tags_path),
        "exclusions": args.exclude,
        "n_excluded": len(excl),
        "indian_specific_objects": sorted(INDIAN_SPECIFIC),
        "confidence_floor": CONFIDENCE_FLOOR,
        "hard_triggers_definition": (
            "crowd=high OR traffic=high OR traffic_mix in {mixed_all, "
            "pedestrian_dominant} OR road_encroachment=heavy OR "
            "N(Indian-specific notable_objects)>=3 (each conf>=0.7)"
        ),
        "clip_keys": [clip_key(t) for t in ultra_kept],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✅ wrote {out_path}")

    # 5) Train/Val/Eval split INSIDE ultra_hard (disjoint, seeded). All 3 share the
    # same difficulty distribution → apples-to-apples within-category comparison.
    # Per user direction (2026-04-25): no random val_1k/eval_10k — eval on same category.
    if len(ultra_kept) < args.ultra_target_n:
        print(f"  ⚠️  raw ultra_hard ({len(ultra_kept)}) < ultra_target_n ({args.ultra_target_n}); "
              f"shrinking train/val/eval proportionally")
        scale = len(ultra_kept) / args.ultra_target_n
        n_train = int(args.ultra_train_n * scale)
        n_val = int(args.ultra_val_n * scale)
        n_eval = len(ultra_kept) - n_train - n_val
    else:
        n_train, n_val, n_eval = args.ultra_train_n, args.ultra_val_n, args.ultra_eval_n

    rng = random.Random(args.seed)
    shuffled = list(ultra_kept)
    rng.shuffle(shuffled)
    train_clips = shuffled[:n_train]
    val_clips = shuffled[n_train:n_train + n_val]
    eval_clips = shuffled[n_train + n_val:n_train + n_val + n_eval]

    split_meta_common = {
        "parent_subset": str(out_path),
        "parent_n": len(ultra_kept),
        "split_seed": args.seed,
        "criterion": ("INTERSECTION: hard_trigger_count>=4 AND "
                      "N(Indian-specific notable_objects)>=4 (all conf>=0.7)"),
        "indian_specific_objects": sorted(INDIAN_SPECIFIC),
        "confidence_floor": CONFIDENCE_FLOOR,
        "source_tags": str(tags_path),
        "split_disjoint_from": "the other two ultra_hard splits (train ∩ val ∩ eval = ∅ by random.shuffle)",
    }

    print()
    print(f"Train/Val/Eval split (seed={args.seed}, all from {len(ultra_kept)} ultra_hard clips):")
    for tag, clips, n in [("train", train_clips, n_train),
                          ("val",   val_clips,   n_val),
                          ("eval",  eval_clips,  n_eval)]:
        sub_path = out_dir / f"ultra_hard_{args.ultra_target_n}_{tag}.json"
        meta = dict(split_meta_common)
        meta["n"] = len(clips)
        meta["tier"] = f"ultra_hard_{args.ultra_target_n}_{tag}"
        meta["role"] = {"train": "m09c training input",
                        "val":   "mid-training motion-flow probe (best_ckpt + val-loss plateau)",
                        "eval":  "paired-BCa decision gate (m05+probe_*)"}[tag]
        meta["clip_keys"] = [clip_key(t) for t in clips]
        with open(sub_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  ✅ wrote {sub_path} ({len(clips)} clips)")

    # 6) Summary
    print()
    print("Output files (data/<name>.json):")
    for name, *_ in CATEGORIES:
        p = out_dir / f"{name}.json"
        if p.is_file():
            print(f"  {p}")
    print(f"  {out_path}")
    for tag in ("train", "val", "eval"):
        print(f"  {out_dir}/ultra_hard_{args.ultra_target_n}_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
