#!/usr/bin/env python3
"""Stratified val_500 / test_500 splits from val_1k by (city, tour_type) (H2 #76).

v10 observation: val_500 best Prec@K = 30.33 vs test_500 Prec@K = 27.97 → 2.36 pp
gap exceeds test_500's CI_half (2.35 pp). val/test are not drawn from equally-difficult
strata, which threatens the paper's decision-gate validity.

Fix: stratified round-robin split on (city, tour_type) joint — the only two tag
columns that correlate with Prec@K difficulty in iter8/iter9 logs. Singleton strata
(n=1) round-robin-assigned to val_500 (deterministic under seed=42).

Writes to NEW filenames (data/val_500_stratified.json / test_500_stratified.json)
so the user can diff against the existing val_500.json/test_500.json before
swapping. To adopt: `mv data/val_500_stratified.json data/val_500.json` (and
similarly for test_500).

USAGE:
    python -u scripts/build_stratified_splits.py 2>&1 | tee logs/build_stratified_splits.log
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


SEED = 42
VAL_1K_JSON = Path("data/val_1k.json")
TAGS_JSON = Path("data/val_1k_local/tags.json")
OUT_VAL = Path("data/val_500_stratified.json")
OUT_TEST = Path("data/test_500_stratified.json")
STRATA_COLS = ("city", "tour_type")


def main() -> None:
    val_1k = json.load(open(VAL_1K_JSON))
    all_keys = list(val_1k["clip_keys"])
    print(f"val_1k clip_keys: {len(all_keys)}")

    tags_list = json.load(open(TAGS_JSON))
    tags = {r["source_file"]: r for r in tags_list}
    print(f"tags records (keyed by source_file): {len(tags)}")

    def _tag_key(clip_key: str) -> str:
        return Path(clip_key).name

    missing = [k for k in all_keys if _tag_key(k) not in tags]
    if missing:
        raise RuntimeError(
            f"{len(missing)} val_1k keys missing tags (e.g. {missing[:3]}). "
            f"Cannot stratify — fail loud per CLAUDE.md §5."
        )

    buckets: dict[tuple, list[str]] = defaultdict(list)
    for k in all_keys:
        t = tags[_tag_key(k)]
        stratum = tuple(t[col] for col in STRATA_COLS)
        buckets[stratum].append(k)

    print(f"strata: {len(buckets)} (by {' × '.join(STRATA_COLS)})")

    rng = random.Random(SEED)
    val_keys: list[str] = []
    test_keys: list[str] = []

    for stratum in sorted(buckets.keys()):
        keys = list(buckets[stratum])
        rng.shuffle(keys)
        for i, k in enumerate(keys):
            (val_keys if i % 2 == 0 else test_keys).append(k)

    print(f"before trim: val={len(val_keys)} test={len(test_keys)}")

    if len(val_keys) + len(test_keys) != len(all_keys):
        raise RuntimeError(
            f"split lost clips: val+test={len(val_keys) + len(test_keys)} != {len(all_keys)}"
        )

    target = 500
    if len(val_keys) > target:
        rng.shuffle(val_keys)
        moved, val_keys = val_keys[target:], val_keys[:target]
        test_keys.extend(moved)
    elif len(test_keys) > target:
        rng.shuffle(test_keys)
        moved, test_keys = test_keys[target:], test_keys[:target]
        val_keys.extend(moved)

    rng.shuffle(val_keys)
    rng.shuffle(test_keys)
    val_keys = val_keys[:target]
    test_keys = test_keys[:target]

    print(f"after trim:  val={len(val_keys)} test={len(test_keys)}")

    overlap = set(val_keys) & set(test_keys)
    if overlap:
        raise RuntimeError(f"val/test overlap: {len(overlap)} keys (e.g. {list(overlap)[:3]})")

    def dist(keys: list[str]) -> dict[tuple, int]:
        out: dict[tuple, int] = defaultdict(int)
        for k in keys:
            t = tags[_tag_key(k)]
            out[tuple(t[col] for col in STRATA_COLS)] += 1
        return dict(out)

    dv, dt = dist(val_keys), dist(test_keys)
    print("\nstratum                             val    test")
    print("-" * 55)
    for stratum in sorted(set(dv) | set(dt)):
        label = " × ".join(stratum)
        print(f"  {label:32s}  {dv.get(stratum, 0):5d}  {dt.get(stratum, 0):5d}")

    for out_path, keys in [(OUT_VAL, val_keys), (OUT_TEST, test_keys)]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(val_1k)
        payload["clip_keys"] = keys
        payload["stratified_by"] = list(STRATA_COLS)
        payload["stratified_seed"] = SEED
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {out_path} (N={len(keys)})")

    print("\nNext: diff against existing data/val_500.json / test_500.json,")
    print("then `mv data/val_500_stratified.json data/val_500.json` (and test) when ready.")


if __name__ == "__main__":
    main()
