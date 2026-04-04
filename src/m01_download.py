"""
Download all 714 YouTube videos at 480p from YT_videos_raw.json.
CPU-only script (M1 compatible). Skips existing files. Uses aria2c + Chrome cookies.

USAGE:
    python -u src/m01_download.py --SANITY 2>&1 | tee logs/m01_download_sanity.log
    python -u src/m01_download.py --POC 2>&1 | tee logs/m01_download_poc.log
    python -u src/m01_download.py --FULL 2>&1 | tee logs/m01_download_full.log
    python -u src/m01_download.py --FULL --res 720 2>&1 | tee logs/m01_download_720p.log
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.progress import make_pbar
from utils.config import VIDEOS_DIR, YT_VIDEOS_JSON, OUTPUTS_DATA_PREP_DIR
from utils.config import get_sanity_clip_limit, get_pipeline_config

# Paths
INPUT_JSON = YT_VIDEOS_JSON
DURATIONS_JSON = OUTPUTS_DATA_PREP_DIR / "video_durations.json"

# Defaults
# 480p for prototyping: fits on M1 Mac (70 GB) + HF Private (100 GB limit)
# V-JEPA input=256x256, Qwen3-VL input=360x420 → 480p is sufficient for model accuracy
# Switch to 2160 (4K) for final HF Public release after arxiv
DEFAULT_RESOLUTION = get_pipeline_config()["data"]["download_resolution"]
SANITY_LIMIT = get_sanity_clip_limit("download")


def check_ytdlp_version():
    """Warn if yt-dlp is older than 90 days."""
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_str = result.stdout.strip()
            try:
                version_date = datetime.strptime(version_str, "%Y.%m.%d")
                age_days = (datetime.now() - version_date).days
                if age_days > 90:
                    print(f"WARNING: yt-dlp version {version_str} is {age_days} days old")
                    print("         Update with: pip install -U yt-dlp")
                else:
                    print(f"yt-dlp version: {version_str} (OK)")
            except ValueError:
                print(f"yt-dlp version: {version_str}")
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)


def check_aria2c():
    """Check if aria2c is available for fast parallel downloads."""
    if shutil.which("aria2c"):
        print("aria2c: available (fast parallel downloads enabled)")
        return True
    else:
        print("aria2c: not found (using default downloader)")
        print("        Install for faster downloads: brew install aria2")
        return False


def extract_all_videos(data: dict) -> list:
    """Extract all (id, url, section) from parsed JSON. Deduplicates by ID."""
    videos = []
    seen_ids = set()

    def add_videos(vids, section):
        for v in vids:
            vid_id = v.get("id", "")
            if vid_id and vid_id not in seen_ids:
                seen_ids.add(vid_id)
                videos.append({"id": vid_id, "url": v["url"], "title": v["title"], "section": section})

    for city, vids in data.get("drive_tours", {}).items():
        add_videos(vids, f"tier1/{city}/drive")
    for city, vids in data.get("drone_views", {}).items():
        add_videos(vids, f"tier1/{city}/drone")
    for city, vids in data.get("walking_tours", {}).items():
        section = f"tier1/{city}/walking" if city != "goa" else "goa/walking"
        add_videos(vids, section)
    for city, city_data in data.get("tier2_cities", {}).items():
        for tour_type in ["drive", "walking", "drone", "rain"]:
            add_videos(city_data.get(tour_type, []), f"tier2/{city}/{tour_type}")
    for m in data.get("monuments", []):
        for tour_type in ["walking_tours", "drive_tours", "drone_views"]:
            add_videos(m.get(tour_type, []), f"monument/{m.get('name', '')}")

    return videos


def download_video(url: str, output_path: Path, max_resolution: int = 480,
                   max_duration: int = None, use_aria2c: bool = True,
                   force: bool = False) -> bool:
    """Download a video at specified max resolution."""
    if output_path.exists() and not force:
        print(f"  SKIP: {output_path.name} (exists)")
        return True

    # Format string: best video up to max_resolution + best audio, fallback to combined
    fmt = (f"bestvideo[height<={max_resolution}][ext=mp4]+bestaudio[ext=m4a]/"
           f"bestvideo[height<={max_resolution}]+bestaudio/"
           f"best[height<={max_resolution}]/best")

    cmd = ["yt-dlp", "-f", fmt, "-o", str(output_path), "--merge-output-format", "mp4",
           "--js-runtimes", "node"]

    # Read fresh cookies from Chrome's cookie store (avoids YouTube cookie rotation)
    cmd.extend(["--cookies-from-browser", "chrome"])

    if use_aria2c and shutil.which("aria2c"):
        cmd.extend(["--downloader", "aria2c", "--downloader-args", "aria2c:-x 16 -s 16 -k 1M"])

    if max_duration is not None:
        cmd.extend(["--download-sections", f"*0-{max_duration}"])

    cmd.append(url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  OK: {output_path.name} ({size_mb:.0f} MB)")
            return True
        else:
            stderr_short = result.stderr[-200:] if result.stderr else "unknown error"
            print(f"  FAIL: {stderr_short}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {output_path.name} (>10 min)")
        return False
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        return False


def estimate_disk_usage(videos: list, max_resolution: int) -> float:
    """Estimate total download size in GB using duration data if available."""
    if not DURATIONS_JSON.exists():
        return 0

    with open(DURATIONS_JSON, 'r') as f:
        durations = json.load(f)

    # Build ID → filesize map (durations is {"summary":..., "videos":[...]})
    id_to_info = {d["id"]: d for d in durations.get("videos", []) if d.get("status") == "ok"}

    # Bitrate ratios relative to best quality (from real data: 4K avg = 2.3 MB/s)
    res_ratio = {360: 0.07, 480: 0.12, 720: 0.25, 1080: 0.5, 1440: 0.75, 2160: 1.0}
    ratio = res_ratio.get(max_resolution, 0.12)

    total_mb = 0
    for v in videos:
        info = id_to_info.get(v["id"])
        if info and info.get("filesize_approx_mb", 0) > 0:
            total_mb += info["filesize_approx_mb"] * ratio
        elif info and info.get("duration_sec", 0) > 0:
            # Fallback: estimate from duration
            total_mb += info["duration_sec"] * 2.3 * ratio
    return total_mb / 1024


def main():
    parser = argparse.ArgumentParser(description="Download all 714 YouTube videos at specified resolution")
    parser.add_argument("--SANITY", action="store_true", help=f"Download {SANITY_LIMIT} videos (30s each) for testing")
    parser.add_argument("--POC", action="store_true", help="10K subset")
    parser.add_argument("--FULL", action="store_true", help="Download all videos (full length)")
    parser.add_argument("--res", type=int, default=DEFAULT_RESOLUTION, help=f"Max resolution height (default: {DEFAULT_RESOLUTION})")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--no-aria2c", action="store_true", help="Disable aria2c acceleration")
    parser.add_argument("--start", type=int, default=0, help="Start from video index N (for resuming)")
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    # Load input JSON
    if not INPUT_JSON.exists():
        print(f"ERROR: {INPUT_JSON} not found. Run m00_data_prep.py --FULL first.")
        sys.exit(1)

    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    videos = extract_all_videos(data)
    print(f"Total unique videos: {len(videos)}")

    # Check dependencies
    print("\n=== Checking dependencies ===")
    check_ytdlp_version()
    has_aria2c = check_aria2c()
    use_aria2c = has_aria2c and not args.no_aria2c

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    # Count existing
    existing = sum(1 for v in videos if (VIDEOS_DIR / f"{v['id']}.mp4").exists())
    print(f"\nAlready downloaded: {existing}/{len(videos)}")

    # Estimate disk usage
    est_gb = estimate_disk_usage(videos, args.res)
    if est_gb > 0:
        print(f"Estimated total size at {args.res}p: ~{est_gb:.0f} GB")

    if args.SANITY:
        print(f"\n=== SANITY MODE: Download {SANITY_LIMIT} videos (30s each, {args.res}p) ===")
        videos = videos[:SANITY_LIMIT]
        max_duration = 30
    else:
        print(f"\n=== FULL MODE: Download {len(videos)} videos at {args.res}p ===")
        if args.start > 0:
            print(f"Resuming from index {args.start}")
            videos = videos[args.start:]
        max_duration = None

    # Download loop
    ok_count = 0
    fail_count = 0
    skip_count = 0
    start_time = time.time()
    pbar = make_pbar(total=len(videos), desc="m01_download", unit="video")

    for i, video in enumerate(videos, 1):
        vid_id = video["id"]
        output_path = VIDEOS_DIR / f"{vid_id}.mp4"

        elapsed = time.time() - start_time
        rate = (ok_count + skip_count) / elapsed if elapsed > 0 else 0
        remaining = len(videos) - i
        eta = remaining / rate if rate > 0 else 0

        print(f"\n[{i}/{len(videos)}] {video['title'][:60]}")
        print(f"  Section: {video['section']} | ID: {vid_id} | ETA: {int(eta/60)}m")

        success = download_video(
            video["url"], output_path,
            max_resolution=args.res,
            max_duration=max_duration,
            use_aria2c=use_aria2c,
            force=args.force
        )

        if success:
            if output_path.exists() and "SKIP" not in str(success):
                ok_count += 1
            else:
                skip_count += 1
        else:
            fail_count += 1
        pbar.update(1)

    pbar.close()

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DOWNLOAD SUMMARY ({args.res}p)")
    print(f"{'='*60}")
    print(f"Downloaded: {ok_count}")
    print(f"Skipped:    {skip_count} (already existed)")
    print(f"Failed:     {fail_count}")
    print(f"Time:       {elapsed/60:.1f} min")

    # Check actual disk usage
    total_size = sum(f.stat().st_size for f in VIDEOS_DIR.glob("*.mp4"))
    print(f"Disk usage: {total_size / (1024**3):.1f} GB in {VIDEOS_DIR}")

    if fail_count > 0:
        print(f"\nFailed videos (re-run with same command to retry):")
        for v in videos:
            if not (VIDEOS_DIR / f"{v['id']}.mp4").exists():
                print(f"  [{v['id']}] {v['title'][:50]}")

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    print(f"\n{mode} {'COMPLETED' if fail_count == 0 else 'COMPLETED WITH ERRORS'}")


if __name__ == "__main__":
    main()
