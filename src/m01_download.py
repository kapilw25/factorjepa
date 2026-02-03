"""
Download videos from @walkinginindia YouTube channel using yt-dlp.
CPU-only script (M1 compatible). Skips existing files by default.

USAGE:
    python -u src/m01_download.py --SANITY 2>&1 | tee logs/m01_download_sanity.log
    python -u src/m01_download.py --FULL 2>&1 | tee logs/m01_download_full.log
    python -u src/m01_download.py --FULL --force 2>&1 | tee logs/m01_download_full.log  # re-download
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import VIDEOS_DIR, VIDEO_URLS


def download_video(url: str, output_path: Path, max_duration: int = 600, force: bool = False) -> bool:
    """
    Download a video using yt-dlp with duration limit.

    Args:
        url: YouTube video URL
        output_path: Path to save the video
        max_duration: Max duration in seconds (default 10 min = 600s)
        force: If True, re-download even if file exists

    Returns:
        True if successful, False otherwise
    """
    # Skip if already downloaded
    if output_path.exists() and not force:
        print(f"SKIP: {output_path.name} already exists (use --force to re-download)")
        return True

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--download-sections", f"*0-{max_duration}",
        "-o", str(output_path),
        url
    ]

    print(f"Downloading: {url}")
    print(f"Output: {output_path}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"SUCCESS: Downloaded {output_path.name}")
            return True
        else:
            print(f"ERROR: {result.stderr}")
            return False
    except FileNotFoundError:
        print("ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download videos from YouTube")
    parser.add_argument("--SANITY", action="store_true", help="Download 1 video (30s) for testing")
    parser.add_argument("--FULL", action="store_true", help="Download all 3 videos (10 min each)")
    parser.add_argument("--url", type=str, help="Custom YouTube URL to download")
    parser.add_argument("--name", type=str, default="custom", help="Name for custom video")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL or args.url):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --FULL, or --url")
        sys.exit(1)

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if args.SANITY:
        print("=== SANITY MODE: Download 1 video (30s) ===")
        # Download first video, 30 seconds only
        scene_type = list(VIDEO_URLS.keys())[0]
        url = VIDEO_URLS[scene_type]
        output_path = VIDEOS_DIR / f"{scene_type}.mp4"
        success = download_video(url, output_path, max_duration=30, force=args.force)
        print(f"\nSANITY {'PASSED' if success else 'FAILED'}")

    elif args.FULL:
        print("=== FULL MODE: Download all 3 videos (10 min each) ===")
        results = []
        for scene_type, url in VIDEO_URLS.items():
            output_path = VIDEOS_DIR / f"{scene_type}.mp4"
            success = download_video(url, output_path, max_duration=600, force=args.force)
            results.append(success)

        print(f"\nFULL: {sum(results)}/{len(results)} videos downloaded")

    elif args.url:
        print(f"=== CUSTOM MODE: Download {args.url} ===")
        output_path = VIDEOS_DIR / f"{args.name}.mp4"
        success = download_video(args.url, output_path, max_duration=600, force=args.force)
        print(f"\nCUSTOM {'PASSED' if success else 'FAILED'}")


if __name__ == "__main__":
    main()
