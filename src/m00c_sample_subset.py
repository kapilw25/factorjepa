"""
Video-level uniform sampling of 10K clips from 115K for POC runs.
Each of the 714 original videos contributes ~14 clips (10000/714).

USAGE:
    python -u src/m00c_sample_subset.py --FULL 2>&1 | tee logs/m00c_sample_subset.log
    python -u src/m00c_sample_subset.py --FULL --n 5000 2>&1 | tee logs/m00c_sample_subset_5k.log
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import PROJECT_ROOT, OUTPUTS_DATA_PREP_DIR

# Paths
CLIP_DURATIONS_JSON = OUTPUTS_DATA_PREP_DIR / "clip_durations.json"
OUTPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_N = 10000
SEED = 42


def load_clips_by_video(clip_durations: dict) -> dict:
    """
    Group all clips by original video_id.

    Returns:
        {video_id: [clip_dict, ...]}
        where clip_dict has: key, section, video_id, file, duration_sec, size_mb
    """
    sections = clip_durations.get("sections", {})
    by_video = {}

    for section, section_data in sections.items():
        for video_id, video_clips in section_data.get("videos", {}).items():
            if video_id not in by_video:
                by_video[video_id] = []
            for clip in video_clips:
                by_video[video_id].append({
                    "key": f"{section}/{video_id}/{clip['file']}",
                    "section": section,
                    "video_id": video_id,
                    "file": clip["file"],
                    "duration_sec": clip["duration_sec"],
                    "size_mb": clip["size_mb"],
                })

    return by_video


def uniform_sample(by_video: dict, n: int, seed: int) -> list:
    """
    Video-level uniform sampling: each video contributes ~floor(n/num_videos) clips.

    Steps:
        1. base_per_video = floor(n / num_videos)
        2. For videos with fewer clips than base → take all, track shortfall
        3. Redistribute shortfall round-robin to videos that still have headroom
        4. Remaining slots (n - allocated) → round-robin to largest videos
    """
    rng = random.Random(seed)
    num_videos = len(by_video)
    total_clips = sum(len(clips) for clips in by_video.values())

    print(f"Total videos: {num_videos}")
    print(f"Total clips:  {total_clips:,}")
    print(f"Target:       {n:,}")
    print(f"Base per video: floor({n}/{num_videos}) = {math.floor(n / num_videos)}")

    if n >= total_clips:
        print("WARNING: n >= total clips, returning all clips")
        all_clips = []
        for clips in by_video.values():
            all_clips.extend(clips)
        return all_clips

    base = math.floor(n / num_videos)

    # Sort video_ids for determinism
    video_ids = sorted(by_video.keys())

    # Pass 1: allocate base per video, cap at available
    allocation = {}
    shortfall = 0
    for vid in video_ids:
        available = len(by_video[vid])
        if available <= base:
            allocation[vid] = available
            shortfall += (base - available)
        else:
            allocation[vid] = base

    # Pass 2: redistribute shortfall + remainder to videos with headroom
    allocated = sum(allocation.values())
    remaining = n - allocated

    if remaining > 0:
        # Videos sorted by headroom (most headroom first)
        headroom_vids = sorted(
            [v for v in video_ids if allocation[v] < len(by_video[v])],
            key=lambda v: len(by_video[v]) - allocation[v],
            reverse=True
        )
        idx = 0
        while remaining > 0 and headroom_vids:
            vid = headroom_vids[idx % len(headroom_vids)]
            if allocation[vid] < len(by_video[vid]):
                allocation[vid] += 1
                remaining -= 1
            else:
                # This video is exhausted, remove from rotation
                headroom_vids.remove(vid)
                if not headroom_vids:
                    break
                idx = idx % len(headroom_vids)
                continue
            idx += 1

    # Pass 3: sample clips from each video
    sampled = []
    for vid in video_ids:
        clips = by_video[vid]
        k = allocation[vid]
        if k >= len(clips):
            chosen = clips[:]
        else:
            chosen = rng.sample(clips, k)
        # Sort chosen by file name for deterministic ordering within video
        chosen.sort(key=lambda c: c["file"])
        sampled.extend(chosen)

    return sampled


def print_summary(sampled: list, by_video: dict):
    """Print stratification summary."""
    # Group sampled by various dimensions
    tier_counts = {}
    type_counts = {}
    city_counts = {}
    videos_represented = set()

    for clip in sampled:
        section = clip["section"]
        parts = section.split("/")
        videos_represented.add(clip["video_id"])

        # Tier
        tier = parts[0]
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Tour type (last part of section path)
        tour_type = parts[-1] if len(parts) >= 2 else "unknown"
        type_counts[tour_type] = type_counts.get(tour_type, 0) + 1

        # City (second part for tier1/tier2)
        if len(parts) >= 2 and parts[0] in ("tier1", "tier2"):
            city = parts[1]
        elif parts[0] == "goa":
            city = "goa"
        elif parts[0] == "monuments":
            city = "monuments"
        else:
            city = parts[0]
        city_counts[city] = city_counts.get(city, 0) + 1

    print(f"\n{'='*60}")
    print(f"SUBSET SUMMARY: {len(sampled):,} clips")
    print(f"{'='*60}")

    print(f"\nVideos represented: {len(videos_represented)} / {len(by_video)}")

    # Per-video clip count distribution
    per_video = {}
    for clip in sampled:
        vid = clip["video_id"]
        per_video[vid] = per_video.get(vid, 0) + 1
    counts = sorted(per_video.values())
    print(f"Clips per video: min={counts[0]}, median={counts[len(counts)//2]}, max={counts[-1]}")

    print(f"\nBy Tier:")
    for tier in sorted(tier_counts.keys()):
        orig_clips = sum(
            len(by_video[v]) for v in by_video
            for c in by_video[v][:1]
            if c["section"].startswith(tier)
        )
        print(f"  {tier:15s}  {tier_counts[tier]:5d} clips")

    print(f"\nBy Tour Type:")
    for tt in sorted(type_counts.keys()):
        print(f"  {tt:15s}  {type_counts[tt]:5d} clips")

    print(f"\nBy City:")
    for city in sorted(city_counts.keys()):
        print(f"  {city:20s}  {city_counts[city]:5d} clips")

    # Duration / size stats
    total_dur = sum(c["duration_sec"] for c in sampled)
    total_size = sum(c["size_mb"] for c in sampled)
    print(f"\nTotal duration: {total_dur/3600:.1f} hours")
    print(f"Total size:     {total_size/1024:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Video-level uniform subset sampling for POC")
    parser.add_argument("--FULL", action="store_true", required=True,
                        help="Run full sampling (required flag for consistency)")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"Number of clips to sample (default: {DEFAULT_N})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Random seed for reproducibility (default: {SEED})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: data/subset_<n>k.json)")
    args = parser.parse_args()

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = OUTPUT_DIR / f"subset_{args.n // 1000}k.json"

    print(f"Input:  {CLIP_DURATIONS_JSON}")
    print(f"Output: {output_path}")
    print(f"N:      {args.n}")
    print(f"Seed:   {args.seed}")

    # Load
    if not CLIP_DURATIONS_JSON.exists():
        print(f"ERROR: {CLIP_DURATIONS_JSON} not found")
        print("Run m02b_scene_fetch_duration.py first")
        sys.exit(1)

    with open(CLIP_DURATIONS_JSON) as f:
        clip_durations = json.load(f)

    # Group clips by video
    by_video = load_clips_by_video(clip_durations)
    print(f"Videos: {len(by_video)}")

    # Uniform sample
    sampled = uniform_sample(by_video, args.n, args.seed)

    # Summary
    print_summary(sampled, by_video)

    # Output
    output_data = {
        "n": len(sampled),
        "seed": args.seed,
        "source": str(CLIP_DURATIONS_JSON),
        "sampling": "video-level uniform",
        "clips_per_video": f"~{math.floor(args.n / len(by_video))}",
        "num_videos": len(by_video),
        "clip_keys": [c["key"] for c in sampled],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nWritten: {output_path} ({len(sampled):,} clip keys)")


if __name__ == "__main__":
    main()
