"""
Split videos into 4-5 second clips using PySceneDetect with intelligent re-splitting.
CPU-only script (M1 compatible).

USAGE:
    python -u src/m02_scene_detect.py --SANITY 2>&1 | tee logs/m02_scene_detect_sanity.log
    python -u src/m02_scene_detect.py --FULL 2>&1 | tee logs/m02_scene_detect_full.log
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    VIDEOS_DIR, CLIPS_DIR, CLIP_MIN_DURATION, CLIP_MAX_DURATION,
    get_all_videos, get_video_duration, check_output_exists
)

try:
    from scenedetect import open_video, SceneManager, split_video_ffmpeg
    from scenedetect.detectors import ContentDetector, AdaptiveDetector
except ImportError:
    print("ERROR: scenedetect not found. Install with: pip install scenedetect[opencv]")
    sys.exit(1)


def detect_scenes_in_clip(clip_path: Path, detector_type: str = "content", threshold: float = 15.0) -> list:
    """
    Detect scenes within a clip using specified detector.

    Args:
        clip_path: Path to video clip
        detector_type: "content" or "adaptive"
        threshold: Detection threshold

    Returns:
        List of scene boundaries (start_time, end_time) in seconds
    """
    try:
        video = open_video(str(clip_path))
        scene_manager = SceneManager()

        if detector_type == "adaptive":
            scene_manager.add_detector(AdaptiveDetector(
                adaptive_threshold=threshold,
                min_scene_len=int(CLIP_MIN_DURATION * video.frame_rate)
            ))
        else:
            scene_manager.add_detector(ContentDetector(
                threshold=threshold,
                min_scene_len=int(CLIP_MIN_DURATION * video.frame_rate)
            ))

        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        # Convert to time tuples
        scenes = []
        for start, end in scene_list:
            scenes.append((start.get_seconds(), end.get_seconds()))

        return scenes
    except Exception as e:
        print(f"    Detection error: {e}")
        return []


def intelligent_resplit(clip_path: Path, max_duration: float) -> list:
    """
    Intelligently re-split a long clip using multiple detection passes.

    Pass 1: ContentDetector with low threshold (15.0) - detects subtle changes
    Pass 2: AdaptiveDetector (2.0) - adapts to content, more sensitive
    Pass 3: Fixed duration split (fallback)

    Args:
        clip_path: Path to the long clip
        max_duration: Maximum allowed duration in seconds

    Returns:
        List of new clip paths created
    """
    duration = get_video_duration(clip_path)
    if duration <= max_duration:
        return [clip_path]

    parent_dir = clip_path.parent
    stem = clip_path.stem
    new_clips = []

    print(f"  Intelligent re-split: {clip_path.name} ({duration:.1f}s)")

    # PASS 1: ContentDetector with lower threshold
    print(f"    Pass 1: ContentDetector(threshold=15.0)")
    scenes = detect_scenes_in_clip(clip_path, "content", threshold=15.0)

    if len(scenes) > 1:
        print(f"    → Found {len(scenes)} scenes (avg {duration/len(scenes):.1f}s each)")
        # Split at detected boundaries
        new_clips = split_at_scenes(clip_path, scenes, parent_dir, stem, max_duration)
        if new_clips:
            clip_path.unlink()
            print(f"    ✓ Final: {len(new_clips)} clips from {duration:.1f}s source")
            return new_clips

    # PASS 2: AdaptiveDetector (more sensitive)
    print(f"    Pass 2: AdaptiveDetector(threshold=2.0)")
    scenes = detect_scenes_in_clip(clip_path, "adaptive", threshold=2.0)

    if len(scenes) > 1:
        print(f"    → Found {len(scenes)} scenes (avg {duration/len(scenes):.1f}s each)")
        new_clips = split_at_scenes(clip_path, scenes, parent_dir, stem, max_duration)
        if new_clips:
            clip_path.unlink()
            print(f"    ✓ Final: {len(new_clips)} clips from {duration:.1f}s source")
            return new_clips

    # PASS 3: Fixed duration split (fallback)
    print(f"    Pass 3: Fixed {max_duration}s split (fallback)")
    new_clips = split_fixed_chunks(clip_path, parent_dir, stem, max_duration)

    if new_clips:
        clip_path.unlink()
        print(f"    ✓ Final: {len(new_clips)} clips (fixed {max_duration}s chunks)")

    return new_clips


def split_at_scenes(clip_path: Path, scenes: list, parent_dir: Path, stem: str, max_duration: float) -> list:
    """
    Split clip at detected scene boundaries.
    If any resulting scene is still too long, recursively split it.
    """
    new_clips = []
    further_split_count = 0

    for i, (start, end) in enumerate(scenes):
        scene_duration = end - start

        # Skip very short scenes
        if scene_duration < CLIP_MIN_DURATION:
            continue

        output_path = parent_dir / f"{stem}-scene{i:03d}.mp4"

        cmd = [
            "ffmpeg", "-y", "-i", str(clip_path),
            "-ss", str(start),
            "-t", str(scene_duration),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output_path.exists():
            # If scene is still too long, split it further
            if scene_duration > max_duration:
                sub_clips = split_fixed_chunks(output_path, parent_dir, f"{stem}-scene{i:03d}", max_duration)
                if sub_clips:
                    output_path.unlink()
                    new_clips.extend(sub_clips)
                    further_split_count += 1
                else:
                    new_clips.append(output_path)
            else:
                new_clips.append(output_path)

    if further_split_count > 0:
        print(f"    → {further_split_count} scenes were >5s, further split into {len(new_clips)} clips")

    return new_clips


def split_fixed_chunks(clip_path: Path, parent_dir: Path, stem: str, max_duration: float) -> list:
    """
    Split clip into fixed-duration chunks, each ≤ max_duration.
    """
    import math

    duration = get_video_duration(clip_path)
    if duration <= max_duration:
        return [clip_path]

    # Calculate number of chunks using ceil to ensure all chunks ≤ max_duration
    num_chunks = math.ceil(duration / max_duration)
    chunk_duration = duration / num_chunks

    # If chunks would be too short, reduce number of chunks
    if chunk_duration < CLIP_MIN_DURATION and num_chunks > 1:
        num_chunks = max(1, int(duration / CLIP_MIN_DURATION))
        chunk_duration = duration / num_chunks

    new_clips = []

    for i in range(num_chunks):
        start_time = i * chunk_duration
        output_path = parent_dir / f"{stem}-chunk{i:03d}.mp4"

        cmd = [
            "ffmpeg", "-y", "-i", str(clip_path),
            "-ss", str(start_time),
            "-t", str(chunk_duration),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output_path.exists():
            new_clips.append(output_path)

    return new_clips


def resplit_all_long_clips(clips_dir: Path, max_duration: float) -> int:
    """
    Find and intelligently re-split all clips that exceed max_duration.

    Args:
        clips_dir: Directory containing clip subdirectories
        max_duration: Maximum allowed duration in seconds

    Returns:
        Total number of clips after re-splitting
    """
    print(f"\n=== Intelligent re-splitting (clips > {max_duration}s) ===")
    print(f"Strategy: ContentDetector(15.0) → AdaptiveDetector(2.0) → Fixed split")

    total_clips = 0
    resplit_count = 0

    for video_dir in clips_dir.iterdir():
        if not video_dir.is_dir():
            continue

        print(f"\nProcessing: {video_dir.name}/")
        clips = list(video_dir.glob("*.mp4"))

        for clip in clips:
            # Skip already-processed clips
            if "-scene" in clip.stem or "-chunk" in clip.stem:
                total_clips += 1
                continue

            duration = get_video_duration(clip)
            if duration > max_duration:
                new_clips = intelligent_resplit(clip, max_duration)
                total_clips += len(new_clips)
                resplit_count += 1
            else:
                total_clips += 1

    print(f"\n=== Re-split Summary ===")
    print(f"Long clips processed: {resplit_count}")
    print(f"Total clips: {total_clips}")

    # Verify final clip durations
    print(f"\n=== Final Clip Duration Check ===")
    all_durations = []
    clips_over_5s = 0
    clips_under_4s = 0

    for video_dir in clips_dir.iterdir():
        if not video_dir.is_dir():
            continue
        for clip in video_dir.glob("*.mp4"):
            dur = get_video_duration(clip)
            all_durations.append(dur)
            if dur > max_duration:
                clips_over_5s += 1
            if dur < CLIP_MIN_DURATION:
                clips_under_4s += 1

    if all_durations:
        import statistics
        print(f"Total clips: {len(all_durations)}")
        print(f"Duration range: {min(all_durations):.1f}s - {max(all_durations):.1f}s")
        print(f"Mean duration: {statistics.mean(all_durations):.1f}s")
        print(f"Median duration: {statistics.median(all_durations):.1f}s")
        print(f"Clips >5s: {clips_over_5s} (should be 0)")
        print(f"Clips <4s: {clips_under_4s}")

        if clips_over_5s > 0:
            print(f"\nWARNING: {clips_over_5s} clips still exceed 5s!")
            # Show which clips
            for video_dir in clips_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                for clip in video_dir.glob("*.mp4"):
                    dur = get_video_duration(clip)
                    if dur > max_duration:
                        print(f"  {clip.relative_to(clips_dir)}: {dur:.1f}s")

    return total_clips


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

    print(f"Detected {len(scene_list)} scenes (threshold=27.0)")

    if len(scene_list) == 0:
        print("WARNING: No scenes detected, using fixed-duration splits")
        return split_fixed_duration(video_path, output_dir, duration=CLIP_MAX_DURATION)

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
    parser = argparse.ArgumentParser(description="Split videos into 4-5s clips with intelligent scene detection")
    parser.add_argument("--SANITY", action="store_true", help="Process 1 video only")
    parser.add_argument("--FULL", action="store_true", help="Process all videos")
    parser.add_argument("--video", type=str, help="Path to specific video to process")
    parser.add_argument("--resplit-only", action="store_true", help="Only re-split existing long clips")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL or args.video or args.resplit_only):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --FULL, --video, or --resplit-only")
        sys.exit(1)

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if clips already exist (skip for resplit-only mode)
    if not args.resplit_only:
        clip_dirs = [d for d in CLIPS_DIR.iterdir() if d.is_dir()] if CLIPS_DIR.exists() else []
        if clip_dirs:
            if not check_output_exists(clip_dirs, "clips"):
                print("Using cached clips.")
                return

    # Re-split only mode
    if args.resplit_only:
        print("=== RE-SPLIT ONLY MODE ===")
        resplit_all_long_clips(CLIPS_DIR, CLIP_MAX_DURATION)
        return

    if args.SANITY:
        print("=== SANITY MODE: Process 1 video ===")
        videos = get_all_videos()
        if not videos:
            print(f"ERROR: No videos found in {VIDEOS_DIR}")
            sys.exit(1)

        video = videos[0]
        num_clips = split_video_to_clips(video, CLIPS_DIR, CLIP_MIN_DURATION)
        print(f"\nScene detection: {num_clips} clips from {video.name}")

        # Intelligent re-split
        total = resplit_all_long_clips(CLIPS_DIR, CLIP_MAX_DURATION)
        print(f"\nSANITY COMPLETE: {total} clips (4-5s each)")

    elif args.FULL:
        print("=== FULL MODE: Process all videos ===")
        videos = get_all_videos()
        if not videos:
            print(f"ERROR: No videos found in {VIDEOS_DIR}")
            sys.exit(1)

        total_clips = 0
        for video in videos:
            num_clips = split_video_to_clips(video, CLIPS_DIR, CLIP_MIN_DURATION)
            total_clips += num_clips

        print(f"\nScene detection: {total_clips} clips from {len(videos)} videos")

        # Intelligent re-split
        total = resplit_all_long_clips(CLIPS_DIR, CLIP_MAX_DURATION)
        print(f"\nFULL COMPLETE: {total} clips (4-5s each)")

    elif args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"ERROR: Video not found: {video_path}")
            sys.exit(1)

        num_clips = split_video_to_clips(video_path, CLIPS_DIR, CLIP_MIN_DURATION)
        print(f"\nScene detection: {num_clips} clips from {video_path.name}")

        # Intelligent re-split
        total = resplit_all_long_clips(CLIPS_DIR, CLIP_MAX_DURATION)
        print(f"\nCOMPLETE: {total} clips (4-5s each)")


if __name__ == "__main__":
    main()
