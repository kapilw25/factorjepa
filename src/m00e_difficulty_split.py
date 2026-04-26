"""Difficulty-stratified subset splitter for the 115K WalkIndia corpus. CPU-only.

Hypothesis (iter11 v3): Frozen V-JEPA's Western pretraining covers Easy clips fine
(Goa beaches, drone aerials, empty highways) → Δ Surgery vs Frozen ≈ 0 there.
The Δ signal lives on the Hard tail (Indian-specific agent-rich scenes — bazaars,
chowks, ghats with auto_rickshaw + cycle_rickshaw + handcart + sacred_cow + street_vendor
co-occurring). Random eval_10k mixes both → dilution-saturated Δ ≈ 0.

This splitter buckets every clip into Easy / Medium / Hard using m04 VLM tags,
emits 3 disjoint subset files compatible with `data/subset_10k.json` schema,
and excludes any clip that overlaps with `data/val_1k.json` or `data/eval_10k.json`
(so train subsets stay disjoint from probe + decision-gate sets).

Output (default target_n=25000 per bucket):
    data/easy_<N>.json   (~30-50K candidates, clipped to N)
    data/medium_<N>.json (~30-50K candidates, clipped to N)
    data/hard_<N>.json   (~15-30K candidates, clipped to N)
    data/difficulty_stats.json (histogram + per-condition trigger counts + audit trail)

USAGE:
    python -u src/m00e_difficulty_split.py --tags data/full_local/tags.json \\
        --target-n 25000 --output-dir data/ --no-wandb 2>&1 | tee logs/m00e_difficulty_split.log
"""
import argparse
import json
import random
import sys
from pathlib import Path
from collections import Counter

# 17-cat agent taxonomy from configs/train/surgery_*.yaml + tag_taxonomy.json.
# Indian-specific subset of `notable_objects` — the values where Western-pretrained
# V-JEPA is most likely to fail. Excludes universal objects (bus, bike, car, truck,
# pedestrian, skyscraper) which appear across all distributions.
INDIAN_SPECIFIC_OBJECTS = {
    "auto_rickshaw",
    "cycle_rickshaw",
    "handcart",
    "street_vendor",
    "sacred_cow",
    "stray_dog",
    "religious_shrine",
    "overhead_wires",
    "signage",
}

# Confidence threshold below which we treat the VLM tag as unreliable
# (fails open: low-conf tag does not contribute to Hard or Easy classification).
CONFIDENCE_FLOOR = 0.7


def is_high_conf(tag: dict, field: str) -> bool:
    """True iff the VLM's confidence for `field` is ≥ CONFIDENCE_FLOOR."""
    conf_key = f"confidence_{field}"
    return tag.get(conf_key, 0.0) >= CONFIDENCE_FLOOR


def hard_conditions(tag: dict) -> list[str]:
    """List of per-condition Hard triggers active for this clip."""
    hits = []
    if tag.get("crowd_density") == "high" and is_high_conf(tag, "crowd_density"):
        hits.append("crowd_high")
    if tag.get("traffic_density") == "high" and is_high_conf(tag, "traffic_density"):
        hits.append("traffic_high")
    if tag.get("traffic_mix") in {"mixed_all", "pedestrian_dominant"} and is_high_conf(tag, "traffic_mix"):
        hits.append(f"traffic_mix_{tag['traffic_mix']}")
    if tag.get("road_encroachment") == "heavy" and is_high_conf(tag, "road_encroachment"):
        hits.append("encroachment_heavy")
    objs = tag.get("notable_objects") or []
    n_indian = sum(1 for o in objs if o in INDIAN_SPECIFIC_OBJECTS)
    if n_indian >= 3:
        hits.append(f"indian_objects_{n_indian}")
    return hits


def easy_conditions(tag: dict) -> bool:
    """All Easy conditions must hold (all high-conf)."""
    if not (tag.get("crowd_density") == "low" and is_high_conf(tag, "crowd_density")):
        return False
    if not (tag.get("traffic_density") == "low" and is_high_conf(tag, "traffic_density")):
        return False
    if not (tag.get("traffic_mix") == "motorized_only" and is_high_conf(tag, "traffic_mix")):
        return False
    if not (tag.get("road_encroachment") == "clear" and is_high_conf(tag, "road_encroachment")):
        return False
    objs = tag.get("notable_objects") or []
    if len(objs) > 2:
        return False
    return True


def clip_key_from_tag(tag: dict) -> str:
    """Reconstruct the canonical clip_key used by data/subset_10k.json + val_1k.json.

    Format: <section>/<video_id>/<source_file>
        e.g., 'tier1/delhi/drive/abc123XYZ/abc123XYZ-007.mp4'
              'goa/walking/U2hl1v8xxlE/U2hl1v8xxlE-092.mp4'
    """
    return f"{tag['section']}/{tag['video_id']}/{tag['source_file']}"


def hard_score(tag: dict) -> int:
    """Continuous Hard intensity = number of Hard conditions firing (0..5).
    Used to sub-sample if raw Hard count exceeds target_n.
    """
    return len(hard_conditions(tag))


def main():
    parser = argparse.ArgumentParser(description="Difficulty-stratified subset splitter")
    parser.add_argument("--tags", type=str, required=True,
                        help="Path to m04 tags.json (LIST of per-clip tag dicts)")
    parser.add_argument("--target-n", type=int, required=True,
                        help="Target clip count per bucket (e.g., 25000)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to write {easy,medium,hard}_<N>.json")
    parser.add_argument("--exclude", type=str, nargs="*", default=[],
                        help="Optional path(s) to subset JSON whose clip_keys must be EXCLUDED "
                             "(e.g., --exclude data/val_1k.json data/eval_10k.json)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true",
                        help="(no-op flag for pipeline consistency)")
    args = parser.parse_args()

    tags_path = Path(args.tags)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # 1) Load tags
    print(f"Loading {tags_path} ...")
    tags = json.load(open(tags_path))
    if not isinstance(tags, list):
        raise SystemExit(f"FATAL: {tags_path} is type {type(tags).__name__}; expected list of clip-tag dicts")
    print(f"  loaded {len(tags)} clip tags")

    # 2) Load exclusion sets
    exclude_keys: set[str] = set()
    for excl_path in args.exclude:
        excl = json.load(open(excl_path))
        keys = excl.get("clip_keys") if isinstance(excl, dict) else None
        if not keys:
            raise SystemExit(f"FATAL: {excl_path} has no clip_keys list")
        exclude_keys.update(keys)
        print(f"  excluding {len(keys)} clip_keys from {excl_path}")
    print(f"  total excluded: {len(exclude_keys)} unique clip_keys")

    # 3) Bucket every clip
    easy: list[tuple[str, dict]] = []
    medium: list[tuple[str, dict]] = []
    hard: list[tuple[str, dict, int]] = []  # also retain hard_score for ranked sub-sampling
    n_excluded = 0
    n_dropped_no_section = 0
    cond_counter: Counter = Counter()
    for tag in tags:
        # Skip if missing the fields needed to build a clip_key.
        if not all(tag.get(f) for f in ("section", "video_id", "source_file")):
            n_dropped_no_section += 1
            continue
        key = clip_key_from_tag(tag)
        if key in exclude_keys:
            n_excluded += 1
            continue
        h_hits = hard_conditions(tag)
        if h_hits:
            for c in h_hits:
                cond_counter[c] += 1
            hard.append((key, tag, len(h_hits)))
        elif easy_conditions(tag):
            easy.append((key, tag))
        else:
            medium.append((key, tag))

    print()
    print(f"  raw Easy   candidates: {len(easy):>6}")
    print(f"  raw Medium candidates: {len(medium):>6}")
    print(f"  raw Hard   candidates: {len(hard):>6}")
    print(f"  excluded (val/eval overlap): {n_excluded}")
    print(f"  dropped (missing section/video_id/source_file): {n_dropped_no_section}")
    print()
    print("  Hard condition trigger counts:")
    for cond, n in sorted(cond_counter.items(), key=lambda x: -x[1]):
        print(f"    {cond:>30s}: {n:>6}")

    # 4) Sub-sample to target_n (Hard: rank-by-intensity desc; Easy/Medium: random)
    n = args.target_n

    # Hard: take top-N by hard_score (ties broken random, seeded)
    hard.sort(key=lambda x: (-x[2], rng.random()))
    hard_final = hard[:n]
    rng.shuffle(easy)
    rng.shuffle(medium)
    easy_final = easy[:n]
    medium_final = medium[:n]

    # 5) Write 3 subset files
    metadata_common = {
        "seed": args.seed,
        "source_tags": str(tags_path),
        "exclusions": args.exclude,
        "n_excluded": n_excluded,
        "schema": "matches data/subset_10k.json (clip_keys list of '<section>/<video_id>/<source_file>')",
        "indian_specific_objects": sorted(INDIAN_SPECIFIC_OBJECTS),
        "confidence_floor": CONFIDENCE_FLOOR,
    }

    for name, bucket, raw_count, extra in [
        ("easy", easy_final, len(easy), {
            "definition": "ALL of: crowd=low AND traffic=low AND traffic_mix=motorized_only "
                          "AND road_encroachment=clear AND len(notable_objects)<=2 (all conf>=0.7)"}),
        ("medium", medium_final, len(medium), {
            "definition": "Residual — neither Hard nor Easy"}),
        ("hard", hard_final, len(hard), {
            "definition": "ANY of: crowd=high OR traffic=high OR traffic_mix in {mixed_all, "
                          "pedestrian_dominant} OR road_encroachment=heavy OR "
                          "N(Indian-specific notable_objects)>=3 (all conf>=0.7); "
                          "ranked by # conditions firing, top-N kept",
            "trigger_counts": dict(cond_counter)}),
    ]:
        out_path = output_dir / f"{name}_{n}.json"
        out = {
            "n": len(bucket),
            "tier": name,
            "raw_candidate_count": raw_count,
            **metadata_common,
            **extra,
            "clip_keys": [k for k, *_ in bucket],
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  ✅ wrote {out_path} ({len(bucket)} clip_keys)")

    # 6) Stats summary
    stats_path = output_dir / "difficulty_stats.json"
    stats = {
        "n_total_clips": len(tags),
        "n_excluded_overlap": n_excluded,
        "n_dropped_no_section": n_dropped_no_section,
        "raw_counts": {"easy": len(easy), "medium": len(medium), "hard": len(hard)},
        "final_counts": {"easy": len(easy_final), "medium": len(medium_final), "hard": len(hard_final)},
        "hard_trigger_counts": dict(cond_counter),
        "indian_specific_objects": sorted(INDIAN_SPECIFIC_OBJECTS),
        "confidence_floor": CONFIDENCE_FLOOR,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✅ wrote {stats_path}")
    print()
    print(f"Done. Use these as --train-subset / --eval-subset in iter11 v3:")
    for name in ("easy", "medium", "hard"):
        print(f"    {output_dir}/{name}_{n}.json")


if __name__ == "__main__":
    main()
