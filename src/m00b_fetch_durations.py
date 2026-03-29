"""
Fetch YouTube video durations without downloading. Uses yt-dlp parallel metadata extraction.
Outputs storage estimates for download planning.

USAGE:
    python -u src/m00b_fetch_durations.py --SANITY 2>&1 | tee logs/m00b_fetch_durations_sanity.log
    python -u src/m00b_fetch_durations.py --FULL 2>&1 | tee logs/m00b_fetch_durations_full.log
    python -u src/m00b_fetch_durations.py --FULL --resolution 480 2>&1 | tee logs/m00b_fetch_durations_480p.log
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import YT_VIDEOS_JSON
from utils.config import get_sanity_clip_limit, get_pipeline_config

# Paths
from utils.config import OUTPUTS_DATA_PREP_DIR
INPUT_JSON = YT_VIDEOS_JSON
OUTPUT_DIR = OUTPUTS_DATA_PREP_DIR
OUTPUT_JSON = OUTPUT_DIR / "video_durations.json"

# Defaults
DEFAULT_WORKERS = get_pipeline_config()["data"]["data_prep_workers"]
SANITY_LIMIT = get_sanity_clip_limit("data_prep")
AVG_CLIP_DURATION = 7.85  # seconds (measured from 500 sampled clips across 168 videos)


def extract_all_videos(data: dict) -> list:
    """Extract all (id, title, section) tuples from parsed JSON."""
    videos = []

    # Drive tours
    for city, vids in data.get("drive_tours", {}).items():
        for v in vids:
            if v.get("id"):
                videos.append({"id": v["id"], "title": v["title"], "url": v["url"],
                                "section": f"tier1/{city}/drive"})

    # Drone views
    for city, vids in data.get("drone_views", {}).items():
        for v in vids:
            if v.get("id"):
                videos.append({"id": v["id"], "title": v["title"], "url": v["url"],
                                "section": f"tier1/{city}/drone"})

    # Walking tours
    for city, vids in data.get("walking_tours", {}).items():
        for v in vids:
            if v.get("id"):
                videos.append({"id": v["id"], "title": v["title"], "url": v["url"],
                                "section": f"tier1/{city}/walking" if city != "goa" else "goa/walking"})

    # Tier 2
    for city, city_data in data.get("tier2_cities", {}).items():
        for tour_type in ["drive", "walking", "drone", "rain"]:
            for v in city_data.get(tour_type, []):
                if v.get("id"):
                    videos.append({"id": v["id"], "title": v["title"], "url": v["url"],
                                    "section": f"tier2/{city}/{tour_type}"})

    # Monuments
    for m in data.get("monuments", []):
        for tour_type in ["walking_tours", "drive_tours", "drone_views"]:
            for v in m.get(tour_type, []):
                if v.get("id"):
                    videos.append({"id": v["id"], "title": v["title"], "url": v["url"],
                                    "section": f"monument/{m.get('name', '')}"})

    return videos


def fetch_duration_ytdlp(video: dict, resolution: int = 0) -> dict:
    """Fetch single video duration using yt-dlp metadata (no download)."""
    vid_id = video["id"]
    url = video["url"]

    try:
        cmd = ["yt-dlp", "-j", "--skip-download", "--no-warnings"]
        if resolution > 0:
            cmd += ["-f", f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]"]
        cmd.append(url)
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            meta = json.loads(result.stdout)
            duration = meta.get("duration", 0)
            resolution = meta.get("height", 0)
            filesize = meta.get("filesize_approx") or meta.get("filesize") or 0
            fps = meta.get("fps", 0)
            import math
            expected_clips = math.ceil(duration / AVG_CLIP_DURATION) if duration > 0 else 0
            return {
                "id": vid_id,
                "title": video["title"],
                "section": video["section"],
                "duration_sec": duration,
                "duration_str": f"{int(duration)//60}:{int(duration)%60:02d}" if duration else "0:00",
                "resolution": resolution,
                "fps": fps,
                "filesize_approx_mb": round(filesize / (1024 * 1024), 1) if filesize else 0,
                "expected_clips": expected_clips,
                "status": "ok"
            }
        else:
            return {
                "id": vid_id, "title": video["title"], "section": video["section"],
                "duration_sec": 0, "duration_str": "0:00",
                "resolution": 0, "fps": 0, "filesize_approx_mb": 0,
                "expected_clips": 0,
                "status": f"error: {result.stderr[:100]}"
            }
    except subprocess.TimeoutExpired:
        return {
            "id": vid_id, "title": video["title"], "section": video["section"],
            "duration_sec": 0, "duration_str": "0:00",
            "resolution": 0, "fps": 0, "filesize_approx_mb": 0,
            "expected_clips": 0,
            "status": "timeout"
        }
    except Exception as e:
        return {
            "id": vid_id, "title": video["title"], "section": video["section"],
            "duration_sec": 0, "duration_str": "0:00",
            "resolution": 0, "fps": 0, "filesize_approx_mb": 0,
            "expected_clips": 0,
            "status": f"exception: {str(e)[:80]}"
        }


def fetch_all_durations(videos: list, workers: int = 10, resolution: int = 0) -> list:
    """Fetch durations for all videos using parallel yt-dlp calls."""
    results = []
    total = len(videos)
    ok_count = 0
    fail_count = 0

    res_label = f"{resolution}p" if resolution > 0 else "best"
    print(f"Fetching metadata for {total} videos ({workers} parallel workers, format={res_label})...")
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_video = {executor.submit(fetch_duration_ytdlp, v, resolution): v for v in videos}

        for i, future in enumerate(as_completed(future_to_video), 1):
            result = future.result()
            results.append(result)

            if result["status"] == "ok":
                ok_count += 1
            else:
                fail_count += 1

            # Progress every 25 videos or at the end
            if i % 25 == 0 or i == total:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"  [{i:>4}/{total}] ok={ok_count} fail={fail_count} "
                      f"rate={rate:.1f}v/s ETA={int(eta)}s")

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed:.1f}s ({total/elapsed:.1f} videos/sec)")
    print(f"Success: {ok_count}, Failed: {fail_count}")

    return results


def print_summary(results: list) -> dict:
    """Print duration summary tables and storage estimates. Returns summary dict."""
    ok_results = [r for r in results if r["status"] == "ok" and r["duration_sec"] > 0]
    failed = [r for r in results if r["status"] != "ok"]

    if not ok_results:
        print("ERROR: No successful results")
        return

    total_sec = sum(r["duration_sec"] for r in ok_results)
    total_hours = total_sec / 3600
    total_min = total_sec / 60
    avg_sec = total_sec / len(ok_results)
    durations = [r["duration_sec"] for r in ok_results]

    # Overall
    print("\n" + "=" * 70)
    print("DURATION SUMMARY")
    print("=" * 70)
    print(f"Videos with duration: {len(ok_results)}/{len(results)}")
    print(f"Total duration:       {total_hours:.1f} hours ({total_min:.0f} min)")
    print(f"Average per video:    {avg_sec:.0f}s ({avg_sec/60:.1f} min)")
    print(f"Shortest:             {min(durations):.0f}s")
    print(f"Longest:              {max(durations):.0f}s ({max(durations)/60:.1f} min)")

    # Real filesize stats
    total_filesize_mb = sum(r["filesize_approx_mb"] for r in ok_results)
    total_filesize_gb = total_filesize_mb / 1024
    has_filesize = [r for r in ok_results if r["filesize_approx_mb"] > 0]
    no_filesize = len(ok_results) - len(has_filesize)

    # Per-category breakdown
    print("\n" + "=" * 80)
    print("DURATION & SIZE BY CATEGORY (real filesize from yt-dlp)")
    print("=" * 80)
    print(f"{'Category':<25} {'Videos':>6} {'Total(h)':>9} {'Avg(min)':>9} {'Size(GB)':>9} {'Avg(MB)':>9}")
    print("-" * 80)

    categories = {}
    for r in ok_results:
        parts = r["section"].split("/")
        cat = parts[0]
        if cat == "tier1":
            cat_key = f"tier1/{parts[2]}" if len(parts) > 2 else "tier1"
        elif cat == "tier2":
            cat_key = f"tier2/{parts[2]}" if len(parts) > 2 else "tier2"
        else:
            cat_key = cat

        if cat_key not in categories:
            categories[cat_key] = {"count": 0, "total_sec": 0, "total_mb": 0}
        categories[cat_key]["count"] += 1
        categories[cat_key]["total_sec"] += r["duration_sec"]
        categories[cat_key]["total_mb"] += r["filesize_approx_mb"]

    grand_gb = 0
    for cat_key in sorted(categories.keys()):
        c = categories[cat_key]
        hrs = c["total_sec"] / 3600
        avg_min = (c["total_sec"] / c["count"]) / 60
        size_gb = c["total_mb"] / 1024
        avg_mb = c["total_mb"] / c["count"]
        grand_gb += size_gb
        print(f"{cat_key:<25} {c['count']:>6} {hrs:>9.1f} {avg_min:>9.1f} {size_gb:>9.1f} {avg_mb:>9.0f}")

    print("-" * 80)
    avg_mb_all = total_filesize_mb / len(ok_results) if ok_results else 0
    print(f"{'TOTAL':<25} {len(ok_results):>6} {total_hours:>9.1f} "
          f"{(avg_sec/60):>9.1f} {grand_gb:>9.1f} {avg_mb_all:>9.0f}")

    # Storage estimates from REAL data
    print("\n" + "=" * 80)
    print("STORAGE ESTIMATES (from real yt-dlp filesize_approx)")
    print("=" * 80)
    if no_filesize > 0:
        print(f"  WARNING: {no_filesize}/{len(ok_results)} videos missing filesize, using avg to fill")
        avg_filesize_mb = total_filesize_mb / len(has_filesize) if has_filesize else 0
        filled_total_gb = (total_filesize_mb + no_filesize * avg_filesize_mb) / 1024
        print(f"  Adjusted total (best quality): {filled_total_gb:.1f} GB")
    else:
        filled_total_gb = total_filesize_gb
    print(f"  Total download size (best quality): {filled_total_gb:.1f} GB")

    # Resolution breakdown
    res_counts = {}
    for r in ok_results:
        res = r.get("resolution", 0)
        label = f"{res}p" if res else "unknown"
        if label not in res_counts:
            res_counts[label] = {"count": 0, "total_mb": 0}
        res_counts[label]["count"] += 1
        res_counts[label]["total_mb"] += r["filesize_approx_mb"]
    print(f"\n  By resolution:")
    for label in sorted(res_counts.keys(), key=lambda x: res_counts[x]["count"], reverse=True):
        rc = res_counts[label]
        print(f"    {label:<10} {rc['count']:>4} videos  {rc['total_mb']/1024:>8.1f} GB")

    # Clip estimates (using measured avg from real scene detection)
    est_clips = total_sec / AVG_CLIP_DURATION
    total_expected_clips = sum(r.get("expected_clips", 0) for r in ok_results)
    # Real bitrate = total_filesize / total_duration
    real_bitrate_mbs = total_filesize_mb / total_sec if total_sec > 0 else 4
    clips_gb = est_clips * (AVG_CLIP_DURATION * real_bitrate_mbs) / 1024
    print(f"\n  Real avg bitrate:               {real_bitrate_mbs:.1f} MB/s")
    print(f"  Avg clip duration (measured):   {AVG_CLIP_DURATION}s")
    print(f"  Estimated clips:                ~{int(est_clips):,} clips")
    print(f"  Sum(expected_clips):            {total_expected_clips:,} clips")
    print(f"  Clips total size:               ~{clips_gb:.0f} GB (same as source)")

    # Per-video streaming estimate
    filesizes_mb = [r["filesize_approx_mb"] for r in ok_results if r["filesize_approx_mb"] > 0]
    max_vid_mb = max(filesizes_mb) if filesizes_mb else 0
    max_vid_entry = next((r for r in ok_results if r["filesize_approx_mb"] == max_vid_mb), None)
    print(f"\n  Stream-split-upload strategy:")
    print(f"    Largest video: {max_vid_mb/1024:.1f} GB", end="")
    if max_vid_entry:
        print(f" ({max_vid_entry['title'][:40]}...)")
    else:
        print()
    print(f"    Peak disk (video + clips): ~{max_vid_mb*2/1024:.1f} GB")
    print(f"    Safe buffer needed:        ~{max(5, int(max_vid_mb*2.5/1024))} GB")

    # Failed videos
    if failed:
        print(f"\n{'='*70}")
        print(f"FAILED VIDEOS ({len(failed)})")
        print("=" * 70)
        for r in failed[:20]:
            print(f"  [{r['id']}] {r['title'][:50]}  → {r['status']}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

    # Build summary dict
    filesizes_mb = [r["filesize_approx_mb"] for r in ok_results if r["filesize_approx_mb"] > 0]
    max_vid_mb = max(filesizes_mb) if filesizes_mb else 0
    return {
        "total_videos": len(results),
        "ok_videos": len(ok_results),
        "failed_videos": len(failed),
        "total_duration_hours": round(total_hours, 1),
        "total_duration_sec": round(total_sec, 0),
        "avg_duration_sec": round(avg_sec, 0),
        "min_duration_sec": min(durations),
        "max_duration_sec": max(durations),
        "total_filesize_gb": round(total_filesize_gb, 1),
        "avg_clip_duration_sec": AVG_CLIP_DURATION,
        "estimated_total_clips": int(est_clips),
        "sum_expected_clips": total_expected_clips,
        "estimated_clips_size_gb": round(clips_gb, 0),
        "real_avg_bitrate_mbs": round(real_bitrate_mbs, 1),
        "largest_video_mb": round(max_vid_mb, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube video durations without downloading")
    parser.add_argument("--SANITY", action="store_true", help=f"Fetch durations for {SANITY_LIMIT} videos only")
    parser.add_argument("--FULL", action="store_true", help="Fetch durations for all videos")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--resolution", type=int, default=0, help="Target resolution height (e.g. 480 for 480p). Default: best quality")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    # Load input JSON
    if not INPUT_JSON.exists():
        print(f"ERROR: {INPUT_JSON} not found. Run m00_data_prep.py --FULL first.")
        sys.exit(1)

    print(f"=== Reading {INPUT_JSON} ===")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract all videos
    videos = extract_all_videos(data)
    # Deduplicate by ID (keep first occurrence)
    seen_ids = set()
    unique_videos = []
    for v in videos:
        if v["id"] not in seen_ids:
            seen_ids.add(v["id"])
            unique_videos.append(v)
    videos = unique_videos

    print(f"Total unique videos: {len(videos)}")

    if args.SANITY:
        videos = videos[:SANITY_LIMIT]
        print(f"SANITY MODE: Fetching {len(videos)} videos")

    # Check yt-dlp
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        print(f"yt-dlp version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)

    # Fetch durations
    results = fetch_all_durations(videos, workers=args.workers, resolution=args.resolution)

    # Sort by section for consistent output
    results.sort(key=lambda r: r["section"])

    # Print summary (returns summary dict)
    summary = print_summary(results)

    # Save as {"summary": {...}, "videos": [...]}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_data = {"summary": summary, "videos": results}
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {OUTPUT_JSON}")

    mode = "SANITY" if args.SANITY else "FULL"
    print(f"\n{mode} COMPLETED")


if __name__ == "__main__":
    main()
