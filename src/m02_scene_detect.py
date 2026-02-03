"""
Split videos into 4-5 second clips using PySceneDetect.
CPU-only script (M1 compatible).

USAGE:
    python -u src/m02_scene_detect.py --SANITY 2>&1 | tee logs/m02_scene_detect_sanity.log
    python -u src/m02_scene_detect.py --FULL 2>&1 | tee logs/m02_scene_detect_full.log
"""
import argparse
import sys
from pathlib import Path

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import VIDEOS_DIR, CLIPS_DIR, CLIP_MIN_DURATION, CLIP_MAX_DURATION

try:
    from scenedetect import open_video, SceneManager, split_video_ffmpeg
    from scenedetect.detectors import ContentDetector
except ImportError:
    print("ERROR: scenedetect not found. Install with: pip install scenedetect[opencv]")
    sys.exit(1)


def split_video_to_clips(video_path: Path, output_dir: Path, min_duration: float = 4.0) -> int:
    """
    Split a video into clips using content-aware scene detection.

    Args:
        video_path: Path to input video
        output_dir: Directory to save clips
        min_duration: Minimum clip duration in seconds

    Returns:
        Number of clips created
    """
    print(f"Processing: {video_path.name}")

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=int(min_duration * video.frame_rate)))

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    print(f"Detected {len(scene_list)} scenes")

    if len(scene_list) == 0:
        print("WARNING: No scenes detected, using fixed-duration splits")
        # Fallback: split into fixed 5-second clips
        return split_fixed_duration(video_path, output_dir, duration=5.0)

    # Create output directory for this video
    video_clip_dir = output_dir / video_path.stem
    video_clip_dir.mkdir(parents=True, exist_ok=True)

    # Split video at scene boundaries
    split_video_ffmpeg(
        str(video_path),
        scene_list,
        output_dir=str(video_clip_dir),
        output_file_template="$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4",
        show_progress=True
    )

    clips = list(video_clip_dir.glob("*.mp4"))
    print(f"Created {len(clips)} clips in {video_clip_dir}")
    return len(clips)


def split_fixed_duration(video_path: Path, output_dir: Path, duration: float = 5.0) -> int:
    """
    Split video into fixed-duration clips using ffmpeg.

    Args:
        video_path: Path to input video
        output_dir: Directory to save clips
        duration: Clip duration in seconds

    Returns:
        Number of clips created
    """
    import subprocess

    video_clip_dir = output_dir / video_path.stem
    video_clip_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(video_clip_dir / f"{video_path.stem}-%03d.mp4")

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-c", "copy",
        "-map", "0",
        "-segment_time", str(duration),
        "-f", "segment",
        "-reset_timestamps", "1",
        output_pattern
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: ffmpeg failed: {result.stderr}")
        return 0

    clips = list(video_clip_dir.glob("*.mp4"))
    print(f"Created {len(clips)} clips (fixed {duration}s) in {video_clip_dir}")
    return len(clips)


def main():
    parser = argparse.ArgumentParser(description="Split videos into clips using PySceneDetect")
    parser.add_argument("--SANITY", action="store_true", help="Process 1 video only")
    parser.add_argument("--FULL", action="store_true", help="Process all videos")
    parser.add_argument("--video", type=str, help="Path to specific video to process")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL or args.video):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --FULL, or --video")
        sys.exit(1)

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    if args.SANITY:
        print("=== SANITY MODE: Process 1 video ===")
        videos = list(VIDEOS_DIR.glob("*.mp4"))
        if not videos:
            print(f"ERROR: No videos found in {VIDEOS_DIR}")
            sys.exit(1)

        video = videos[0]
        num_clips = split_video_to_clips(video, CLIPS_DIR, CLIP_MIN_DURATION)
        print(f"\nSANITY: Created {num_clips} clips from {video.name}")

    elif args.FULL:
        print("=== FULL MODE: Process all videos ===")
        videos = list(VIDEOS_DIR.glob("*.mp4"))
        if not videos:
            print(f"ERROR: No videos found in {VIDEOS_DIR}")
            sys.exit(1)

        total_clips = 0
        for video in videos:
            num_clips = split_video_to_clips(video, CLIPS_DIR, CLIP_MIN_DURATION)
            total_clips += num_clips

        print(f"\nFULL: Created {total_clips} clips from {len(videos)} videos")

    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"ERROR: Video not found: {video_path}")
            sys.exit(1)

        num_clips = split_video_to_clips(video_path, CLIPS_DIR, CLIP_MIN_DURATION)
        print(f"\nCreated {num_clips} clips from {video_path.name}")


if __name__ == "__main__":
    main()
