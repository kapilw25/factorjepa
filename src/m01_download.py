"""
Download complete YouTube videos using yt-dlp with aria2c for fast parallel downloads.
CPU-only script (M1 compatible). Skips existing files by default.

USAGE:
    python -u src/m01_download.py --SANITY 2>&1 | tee logs/m01_download_sanity.log
    python -u src/m01_download.py --FULL 2>&1 | tee logs/m01_download_full.log
    python -u src/m01_download.py --url "https://youtube.com/watch?v=XXX" --name video_name 2>&1 | tee logs/m01_download_custom.log
"""
import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import VIDEOS_DIR, VIDEO_URLS


def check_ytdlp_version():
    """Warn if yt-dlp is older than 90 days."""
    try:
        result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_str = result.stdout.strip()  # e.g., "2024.12.13"
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


def download_video(url: str, output_path: Path, max_duration: int = None,
                   use_aria2c: bool = True, force: bool = False) -> bool:
    """
    Download a video using yt-dlp with optional aria2c acceleration.

    Args:
        url: YouTube video URL
        output_path: Path to save the video
        max_duration: Max duration in seconds (None = full video)
        use_aria2c: Use aria2c for faster parallel downloads
        force: If True, re-download even if file exists

    Returns:
        True if successful, False otherwise
    """
    if output_path.exists() and not force:
        print(f"SKIP: {output_path.name} already exists (use --force to re-download)")
        return True

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", str(output_path),
    ]

    # Add aria2c for fast parallel downloads (bypasses throttling)
    if use_aria2c and shutil.which("aria2c"):
        cmd.extend(["--downloader", "aria2c", "--downloader-args", "aria2c:-x 16 -s 16 -k 1M"])

    # Add duration limit only if specified (SANITY mode)
    if max_duration is not None:
        cmd.extend(["--download-sections", f"*0-{max_duration}"])

    cmd.append(url)

    print(f"Downloading: {url}")
    print(f"Output: {output_path}")
    if max_duration:
        print(f"Duration limit: {max_duration}s")

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
    parser = argparse.ArgumentParser(description="Download YouTube videos (full length)")
    parser.add_argument("--SANITY", action="store_true", help="Download 1 video (30s) for testing")
    parser.add_argument("--FULL", action="store_true", help="Download all videos (complete length)")
    parser.add_argument("--url", type=str, help="Custom YouTube URL to download")
    parser.add_argument("--name", type=str, default="custom", help="Name for custom video")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--no-aria2c", action="store_true", help="Disable aria2c acceleration")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL or args.url):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --FULL, or --url")
        sys.exit(1)

    # Check dependencies
    print("=== Checking dependencies ===")
    check_ytdlp_version()
    has_aria2c = check_aria2c()
    use_aria2c = has_aria2c and not args.no_aria2c
    print()

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if args.SANITY:
        print("=== SANITY MODE: Download 1 video (30s) ===")
        scene_type = list(VIDEO_URLS.keys())[0]
        url = VIDEO_URLS[scene_type]
        output_path = VIDEOS_DIR / f"{scene_type}.mp4"
        success = download_video(url, output_path, max_duration=30,
                                 use_aria2c=use_aria2c, force=args.force)
        print(f"\nSANITY {'PASSED' if success else 'FAILED'}")

    elif args.FULL:
        print("=== FULL MODE: Download all videos (complete length) ===")
        results = []
        for scene_type, url in VIDEO_URLS.items():
            output_path = VIDEOS_DIR / f"{scene_type}.mp4"
            success = download_video(url, output_path, max_duration=None,
                                     use_aria2c=use_aria2c, force=args.force)
            results.append(success)

        print(f"\nFULL: {sum(results)}/{len(results)} videos downloaded")

    elif args.url:
        print(f"=== CUSTOM MODE: Download {args.url} ===")
        output_path = VIDEOS_DIR / f"{args.name}.mp4"
        success = download_video(args.url, output_path, max_duration=None,
                                 use_aria2c=use_aria2c, force=args.force)
        print(f"\nCUSTOM {'PASSED' if success else 'FAILED'}")


if __name__ == "__main__":
    main()
