"""Curate top-20 spot-check PNGs/PDFs from m10 and m11 verify dirs. Technique-agnostic.

Called automatically from m11_factor_datasets.main() post-run (both verify dirs are
pruned in one pass since m11 owns the factor_manifest.json ranking signal — m10 alone
cannot rank because it doesn't yet have per-clip D_I tube counts). Also standalone:

    python -u src/utils/curate_verify.py --outputs-dir outputs/full --delete-originals

Parses factor_manifest.json to rank clips by quality + ensures unique-video coverage
(greedy dedup by video_id = clip_stem.rsplit('-', 1)[0]) + diversity floor (≥3 cities,
≥3 activities). Copies selected 20 clips' PNGs/PDFs from m10_overlay_verify +
m11_per_clip_verify into sibling *_top20/ dirs. Optionally deletes the bulk originals
(saves ~10 GB at 10K, ~115 GB at FULL 115K).
"""
import argparse
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# Composite quality score per clip — weights sum to 1.0
WEIGHTS = {
    "has_D_I": 0.40,    # D_I is the hardest to get right; having it = good mask quality
    "n_tubes": 0.20,    # more interaction tubes = richer content
    "has_D_A": 0.20,    # agent factor present
    "agent_pct": 0.20,  # sweet spot 0.01-0.15 (tiny = ghost masks, full = layout invisible)
}
AGENT_PCT_SWEET_MIN = 0.01
AGENT_PCT_SWEET_MAX = 0.15
TOP_N = 20
PRE_SELECT_N = 100        # cap on per-clip verify plots written by m10/m11 upfront
MIN_DISTINCT_CITIES = 3
MIN_DISTINCT_ACTIVITIES = 3


def parse_clip_key(clip_key: str) -> dict:
    """Extract city, activity, video_id from clip_key.

    Example: "tier1/mumbai/walking/DFQO7Zn-KM4-1000/DFQO7Zn-KM4-1000.mp4"
        → {tier: "tier1", city: "mumbai", activity: "walking",
           video_id: "DFQO7Zn-KM4", stem: "DFQO7Zn-KM4-1000"}
    """
    parts = clip_key.split("/")
    tier = parts[0] if len(parts) > 3 else ""
    # Handle both tier-prefixed ("tier1/city/activity/video/clip.mp4")
    # and tier-less ("goa/walking/video/clip.mp4") layouts
    if tier.startswith("tier"):
        city, activity = parts[1], parts[2]
    else:
        city, activity = tier, parts[1]
        tier = ""
    stem = Path(parts[-1]).stem  # "DFQO7Zn-KM4-1000"
    # video_id = strip trailing "-NNN" clip index (handles multi-dash video IDs)
    video_id = stem.rsplit("-", 1)[0] if "-" in stem else stem
    return {"tier": tier, "city": city, "activity": activity,
            "video_id": video_id, "stem": stem}


def score_clip(entry: dict) -> float:
    """Composite quality score from factor_manifest entry."""
    s = 0.0
    s += WEIGHTS["has_D_I"] * (1.0 if entry.get("has_D_I") else 0.0)
    s += WEIGHTS["has_D_A"] * (1.0 if entry.get("has_D_A") else 0.0)
    # n_tubes normalized — log-scale then clamp to [0, 1] at ~e^5 = 148 tubes
    n = entry.get("n_interaction_tubes", 0)
    s += WEIGHTS["n_tubes"] * min(1.0, math.log1p(n) / 5.0)
    # agent_pct: bump sweet-spot, penalize tails
    pct = entry.get("agent_pct", 0.0)
    if AGENT_PCT_SWEET_MIN <= pct <= AGENT_PCT_SWEET_MAX:
        s += WEIGHTS["agent_pct"] * 1.0
    elif pct > 0:
        # Gaussian-like falloff outside sweet spot
        dist = max(AGENT_PCT_SWEET_MIN - pct, pct - AGENT_PCT_SWEET_MAX)
        s += WEIGHTS["agent_pct"] * max(0.0, 1.0 - 10 * dist)
    return s


def curate(manifest: dict) -> list:
    """Rank clips + greedy unique-video dedup + diversity floor.

    Returns list of TOP_N clip_keys in descending score order.
    """
    scored = []
    for clip_key, entry in manifest.items():
        meta = parse_clip_key(clip_key)
        meta["clip_key"] = clip_key
        meta["score"] = score_clip(entry)
        scored.append(meta)
    scored.sort(key=lambda m: m["score"], reverse=True)

    # Greedy: unique video_id first
    selected = []
    seen_videos = set()
    seen_cities = set()
    seen_activities = set()
    for m in scored:
        if m["video_id"] in seen_videos:
            continue
        selected.append(m)
        seen_videos.add(m["video_id"])
        seen_cities.add(m["city"])
        seen_activities.add(m["activity"])
        if len(selected) >= TOP_N:
            break

    # Diversity floor — if we didn't hit min cities/activities, swap in
    # the highest-scoring clip from under-represented bucket
    for required_field, min_count, seen_set in [
        ("city", MIN_DISTINCT_CITIES, seen_cities),
        ("activity", MIN_DISTINCT_ACTIVITIES, seen_activities),
    ]:
        while len(seen_set) < min_count:
            # Find highest-scored clip NOT in any seen bucket of this field
            missing = [m for m in scored
                       if m[required_field] not in seen_set
                       and m["video_id"] not in seen_videos]
            if not missing:
                print(f"  WARN: can't reach min {required_field}={min_count} "
                      f"(have {len(seen_set)}: {seen_set}) — dataset too narrow")
                break
            swap_in = missing[0]
            # Drop the lowest-scoring over-represented entry
            selected.sort(key=lambda m: m["score"])
            for i, existing in enumerate(selected):
                if sum(1 for s in selected if s[required_field] == existing[required_field]) > 1:
                    seen_videos.discard(existing["video_id"])
                    selected.pop(i)
                    break
            selected.append(swap_in)
            seen_videos.add(swap_in["video_id"])
            seen_set.add(swap_in[required_field])
            selected.sort(key=lambda m: m["score"], reverse=True)

    return selected[:TOP_N]


def copy_pngs(selected: list, src_dir: Path, dst_dir: Path, exts=(".png", ".pdf")) -> int:
    """Copy selected clips' verify plots from src_dir → dst_dir. Returns n_copied."""
    if not src_dir.exists():
        print(f"  SKIP: {src_dir} does not exist")
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    n_missing = 0
    for m in selected:
        # Filename pattern: clip_key with "/" → "__" + ".mp4" + ext
        # e.g. "tier1/mumbai/walking/DFQO7Zn-KM4-1000/DFQO7Zn-KM4-1000.mp4"
        #   → "tier1__mumbai__walking__DFQO7Zn-KM4-1000__DFQO7Zn-KM4-1000.mp4.png"
        stem = m["clip_key"].replace("/", "__")
        for ext in exts:
            fname = f"{stem}{ext}"
            src = src_dir / fname
            if src.exists():
                shutil.copy2(src, dst_dir / fname)
                n_copied += 1
            else:
                n_missing += 1
    print(f"  {src_dir.name}: copied {n_copied} files to {dst_dir} "
          f"({n_missing} missing — OK if exts partial)")
    return n_copied


def select_verify_clips(clip_keys, n_target: int = PRE_SELECT_N, seed: int = 42) -> set:
    """Metadata-only pre-selection of ~100 clip_keys for m10/m11 verify plots.
    Runs BEFORE m10 writes any PNG (no factor_manifest.json needed).

    Strategy:
        1. Group clip_keys by (city, activity) bucket.
        2. Dedupe within bucket by video_id (one clip per video).
        3. Round-robin across buckets → even coverage across cities+activities.
        4. Cap at n_target. If dataset has < n_target distinct videos, returns fewer.

    Determinism: same (clip_keys, n_target, seed) → same set across calls, so m10
    and m11 independently pre-filter to the SAME clips (paired plots for spot-check).

    Args:
        clip_keys: iterable of clip_keys (e.g. "tier1/mumbai/walking/XYZ/XYZ-001.mp4")
        n_target: hard cap on selected count (default 100).
        seed: RNG seed for within-bucket shuffle (default 42 to match data.seed).

    Returns:
        set of selected clip_keys.
    """
    rng = random.Random(seed)
    # Bucket by (city, activity); dedupe by video_id within each
    buckets = defaultdict(list)          # (city, activity) -> [(video_id, clip_key)]
    for ck in clip_keys:
        meta = parse_clip_key(ck)
        buckets[(meta["city"], meta["activity"])].append((meta["video_id"], ck))
    # Shuffle + dedupe inside each bucket
    deduped = {}
    for bk, items in buckets.items():
        rng.shuffle(items)
        seen = set()
        picked = []
        for vid, ck in items:
            if vid not in seen:
                seen.add(vid)
                picked.append(ck)
        deduped[bk] = picked
    # Round-robin across buckets
    bucket_keys = sorted(deduped.keys())
    selected = set()
    seen_videos = set()
    while len(selected) < n_target:
        any_added = False
        for bk in bucket_keys:
            if not deduped[bk] or len(selected) >= n_target:
                continue
            ck = deduped[bk].pop(0)
            meta = parse_clip_key(ck)
            if meta["video_id"] in seen_videos:
                continue
            selected.add(ck)
            seen_videos.add(meta["video_id"])
            any_added = True
        if not any_added:
            break
    return selected


def curate_and_prune(outputs_dir, delete_originals: bool = False) -> dict:
    """Top-level entry point. Reads factor_manifest.json, selects top-20 clips
    with unique-video coverage + diversity floor, copies PNGs/PDFs to *_top20/
    dirs alongside m10 + m11 outputs, writes `verify_top20_manifest.json`, and
    optionally deletes the bulk verify dirs to free disk (10K: ~10 GB, FULL: ~115 GB).

    Args:
        outputs_dir: mode dir (e.g. `outputs/poc` or `outputs/full`) — must contain
            `m10_sam_segment/` and `m11_factor_datasets/` sub-directories.
        delete_originals: if True, `rm -rf` the bulk verify dirs after top-20 copy.

    Returns:
        dict with `n_selected`, `coverage`, `freed_gb`, `selection_json_path` for logging.
    """
    out = Path(outputs_dir)
    manifest_path = out / "m11_factor_datasets" / "factor_manifest.json"
    if not manifest_path.exists():
        print(f"[curate_verify] SKIP: {manifest_path} not found — run m11 first")
        return {"n_selected": 0, "skipped": True}

    manifest = json.load(open(manifest_path))
    print(f"[curate_verify] Loaded {len(manifest)} clips from {manifest_path.name}")

    selected = curate(manifest)
    if len(selected) < TOP_N:
        print(f"[curate_verify] WARN: curation produced only {len(selected)} clips "
              f"(expected {TOP_N}) — dataset may be too narrow")

    print(f"[curate_verify] Top {len(selected)} (score-ranked, unique videos):")
    for i, m in enumerate(selected, 1):
        print(f"  {i:2d}. score={m['score']:.3f}  "
              f"{m['city']:12s} {m['activity']:10s} {m['video_id']:30s}")

    seen_cities = sorted({m["city"] for m in selected})
    seen_activities = sorted({m["activity"] for m in selected})
    print(f"[curate_verify] Coverage: {len(seen_cities)} cities {seen_cities}, "
          f"{len(seen_activities)} activities {seen_activities}")

    m10_src = out / "m10_sam_segment" / "m10_overlay_verify"
    m11_src = out / "m11_factor_datasets" / "m11_per_clip_verify"
    m10_dst = out / "m10_sam_segment" / "m10_overlay_verify_top20"
    m11_dst = out / "m11_factor_datasets" / "m11_per_clip_verify_top20"
    copy_pngs(selected, m10_src, m10_dst)
    copy_pngs(selected, m11_src, m11_dst)

    selection_json = out / "verify_top20_manifest.json"
    with open(selection_json, "w") as f:
        json.dump({
            "n_selected": len(selected),
            "weights": WEIGHTS,
            "agent_pct_sweet_range": [AGENT_PCT_SWEET_MIN, AGENT_PCT_SWEET_MAX],
            "diversity_floor": {"cities": MIN_DISTINCT_CITIES, "activities": MIN_DISTINCT_ACTIVITIES},
            "coverage": {"cities": seen_cities, "activities": seen_activities},
            "selected": [{k: m[k] for k in ("clip_key", "video_id", "city", "activity", "score")}
                         for m in selected],
        }, f, indent=2)
    print(f"[curate_verify] Saved selection manifest: {selection_json}")

    freed_gb = 0.0
    if delete_originals:
        for d in [m10_src, m11_src]:
            if d.exists():
                size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                freed_gb += size / 1e9
                shutil.rmtree(d)
                print(f"[curate_verify] DELETED: {d}")
        print(f"[curate_verify] Freed ~{freed_gb:.1f} GB disk")
    else:
        print(f"[curate_verify] Originals KEPT. Pass delete_originals=True to free disk.")

    return {
        "n_selected": len(selected),
        "coverage": {"cities": seen_cities, "activities": seen_activities},
        "freed_gb": freed_gb,
        "selection_json_path": str(selection_json),
    }


def _main_cli():
    """Standalone CLI for debugging / one-off curation outside m11."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--outputs-dir", type=str, required=True,
                   help="e.g. outputs/poc or outputs/full (contains m10_sam_segment/ + m11_factor_datasets/)")
    p.add_argument("--delete-originals", action="store_true",
                   help="After copying top-20, rm -rf the bulk verify dirs (saves disk)")
    args = p.parse_args()
    result = curate_and_prune(args.outputs_dir, delete_originals=args.delete_originals)
    if result.get("skipped"):
        sys.exit(1)


if __name__ == "__main__":
    _main_cli()
