"""
Greedy scene-aware splitting: detect all boundaries once, pick best cut in [4s, 10s]
window, encode each clip once with libx264 CRF 28. No double encoding, no resplitting.

USAGE:
    python -u src/m02_scene_detect.py --SANITY 2>&1 | tee logs/m02_scene_detect_sanity.log
    python -u src/m02_scene_detect.py --POC 2>&1 | tee logs/m02_scene_detect_poc.log
    caffeinate -s python -u src/m02_scene_detect.py --FULL 2>&1 | tee logs/m02_scene_detect_full.log
    caffeinate -s python -u src/m02_scene_detect.py --FULL --keyframes 2>&1 | tee logs/m02_scene_detect_full_kf.log
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.progress import make_pbar
from utils.config import (
    VIDEOS_DIR, CLIPS_DIR, CLIP_MIN_DURATION, CLIP_MAX_DURATION, REENCODE_CRF,
    get_all_videos, get_video_duration,
    build_video_section_map, get_processed_video_ids, mark_video_processed
)

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
except ImportError:
    print("ERROR: scenedetect not found. Install with: pip install scenedetect[opencv]")
    sys.exit(1)


def detect_all_boundaries(video_path: Path) -> list:
    """Run PySceneDetect once on full video, return boundary timestamps in seconds."""
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=15.0))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    # Extract boundary points (each scene end = next scene start)
    boundaries = []
    for start, end in scene_list:
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        if start_sec > 0 and start_sec not in boundaries:
            boundaries.append(start_sec)
        if end_sec not in boundaries:
            boundaries.append(end_sec)

    # Remove 0.0 and video end (not useful as cut points)
    video_dur = get_video_duration(video_path)
    boundaries = sorted(set(b for b in boundaries if 0 < b < video_dur))
    return boundaries


def greedy_split_plan(video_duration: float, boundaries: list,
                      min_dur: float = 4.0, max_dur: float = 10.0) -> list:
    """
    Greedy algorithm: walk through video, pick best scene boundary in [4s, 10s] window.
    Returns list of (start, end) tuples. Pure logic, no I/O.
    """
    if video_duration <= 0:
        return []
    if video_duration <= max_dur:
        return [(0.0, video_duration)]

    clips = []
    pos = 0.0

    while pos < video_duration:
        remaining = video_duration - pos

        # Last segment too short to be its own clip — extend previous
        if remaining < min_dur:
            if clips:
                prev_start, _ = clips[-1]
                clips[-1] = (prev_start, video_duration)
            else:
                clips.append((0.0, video_duration))
            break

        window_start = pos + min_dur
        window_end = pos + max_dur

        # Find scene boundaries in [pos+4, pos+10]
        candidates = [b for b in boundaries if window_start <= b <= min(window_end, video_duration)]

        if candidates:
            cut = max(candidates)  # closest to 10s = longest clip
        else:
            cut = min(window_end, video_duration)

        # If cutting here would leave <4s remainder, absorb it into this clip
        if 0 < (video_duration - cut) < min_dur:
            cut = video_duration

        clips.append((pos, cut))
        pos = cut

    return clips


def encode_clip(video_path: Path, start: float, duration: float,
                output_path: Path) -> bool:
    """Encode one clip with libx264 CRF 28. -ss before -i for fast seeking."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264", "-crf", str(REENCODE_CRF), "-preset", "medium",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-loglevel", "error",
        str(output_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0 and output_path.exists()
    except Exception as e:
        print(f"    Encode error: {e}")
        return False


def extract_keyframe(clip_path: Path) -> bool:
    """Extract 1 keyframe (middle frame) from a clip as JPEG. Output: same dir, same name .jpg."""
    output_path = clip_path.with_suffix(".jpg")
    if output_path.exists():
        return True
    dur = get_video_duration(clip_path)
    if dur <= 0:
        return False
    mid = dur / 2.0
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(mid),
        "-i", str(clip_path),
        "-frames:v", "1",
        "-q:v", "2",
        "-loglevel", "error",
        str(output_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and output_path.exists()
    except Exception:
        return False


def verify_video_clips(video_id: str, section_map: dict, clips_dir: Path) -> int:
    """Count actual clips on disk for a video. Returns count."""
    if section_map and video_id in section_map:
        video_clip_dir = clips_dir / section_map[video_id]
    else:
        video_clip_dir = clips_dir / "unsorted"
    if not video_clip_dir.exists():
        return 0
    return len(list(video_clip_dir.glob(f"{video_id}-*.mp4")))


def process_video(video: Path, clips_dir: Path, section_map: dict = None,
                   keyframes: bool = False) -> int:
    """Detect scenes → greedy split plan → encode each clip once. Returns clip count."""
    video_id = video.stem
    print(f"\nProcessing: {video.name}")

    video_duration = get_video_duration(video)
    if video_duration <= 0:
        print(f"  ERROR: Cannot get duration")
        return 0
    print(f"  Duration: {video_duration:.1f}s")

    # 1. Detect all scene boundaries (one pass)
    try:
        boundaries = detect_all_boundaries(video)
        print(f"  Scene boundaries: {len(boundaries)}")
    except Exception as e:
        print(f"  Scene detection failed: {e}, using fixed splits")
        boundaries = []

    # 2. Compute greedy split plan
    intervals = greedy_split_plan(video_duration, boundaries,
                                  CLIP_MIN_DURATION, CLIP_MAX_DURATION)
    print(f"  Split plan: {len(intervals)} clips")

    # 3. Determine output directory
    if section_map and video_id in section_map:
        video_clip_dir = clips_dir / section_map[video_id]
    else:
        video_clip_dir = clips_dir / "unsorted"
    video_clip_dir.mkdir(parents=True, exist_ok=True)

    # 4. Encode each clip
    encoded_clips = []
    start_time = time.time()
    for i, (start, end) in enumerate(intervals):
        duration = end - start
        output_path = video_clip_dir / f"{video_id}-{i:03d}.mp4"

        if encode_clip(video, start, duration, output_path):
            encoded_clips.append(output_path)
        else:
            print(f"    WARNING: Failed to encode clip {i:03d} ({start:.1f}s-{end:.1f}s)")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = int((len(intervals) - i - 1) / rate) if rate > 0 else 0
            print(f"    [{i+1}/{len(intervals)}] rate={rate:.1f}/s ETA={eta}s")

    encode_time = time.time() - start_time
    print(f"  Encoded {len(encoded_clips)}/{len(intervals)} clips in {encode_time:.0f}s")

    # 5. Extract keyframes if requested
    if keyframes and encoded_clips:
        kf_ok = sum(1 for c in encoded_clips if extract_keyframe(c))
        print(f"  Keyframes: {kf_ok}/{len(encoded_clips)} extracted")

    # 6. Print per-clip durations for log monitoring
    violations = 0
    print(f"  --- {video_id}: {len(encoded_clips)} clips ---")
    for clip in encoded_clips:
        dur = get_video_duration(clip)
        if dur > CLIP_MAX_DURATION + 1.0:
            print(f"    {clip.name}  {dur:.2f}s  *** VIOLATION >11s")
            violations += 1
        elif dur < CLIP_MIN_DURATION - 1.0:
            print(f"    {clip.name}  {dur:.2f}s  * <3s")
            violations += 1
        else:
            print(f"    {clip.name}  {dur:.2f}s")

    if violations:
        print(f"  WARNING: {violations} clips outside range!")
    else:
        print(f"  OK: all {len(encoded_clips)} clips within [3s, 11s]")

    return len(encoded_clips)


def main():
    parser = argparse.ArgumentParser(description="Greedy scene-aware split → encode → upload")
    parser.add_argument("--SANITY", action="store_true", help="Process 1 video only")
    parser.add_argument("--POC", action="store_true", help="10K subset")
    parser.add_argument("--FULL", action="store_true", help="Process all videos (auto-polls if m01 still running)")
    parser.add_argument("--keyframes", action="store_true", help="Extract 1 middle-frame JPEG per clip")
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    POLL_INTERVAL = 300  # 5 min between polls

    section_map = build_video_section_map()
    if not section_map:
        # iter13 v13 FIX-16 (2026-05-07): FAIL HARD per CLAUDE.md (no
        # WARN-without-exit). Without a section map, every clip falls into
        # 'unsorted/' which breaks downstream subset JSON organization
        # (m00c_sample_subset, m00f_category_subsets etc. select BY section).
        # Better to surface the upstream failure than ship corrupt data.
        print("FATAL: No section map available — build_video_section_map() returned empty.")
        print("  Without per-video section mapping, clips would be dumped into 'unsorted/'")
        print("  and miss every downstream section-organized subset (m00c/m00e/m00f).")
        print("  Investigate the YT_videos_raw.md → JSON conversion (m00_data_prep).")
        sys.exit(1)
    print(f"Section map: {len(section_map)} videos mapped to hierarchical dirs")

    if args.SANITY:
        print("=== SANITY MODE: Process 1 video ===")
        videos = get_all_videos()
        if not videos:
            print(f"ERROR: No videos found in {VIDEOS_DIR}")
            sys.exit(1)

        processed = get_processed_video_ids()
        vid = None
        for v in videos:
            if v.stem not in processed:
                vid = v
                break
            actual = verify_video_clips(v.stem, section_map, CLIPS_DIR)
            expected = processed[v.stem]
            if expected == -1:
                if actual == 0:
                    vid = v
                    break
            elif actual < expected or actual == 0:
                vid = v
                break
        if vid is None:
            print("All videos already processed. Nothing to SANITY test.")
            sys.exit(0)

        print(f"  Selected: {vid.stem} (unprocessed)")
        clip_count = process_video(vid, CLIPS_DIR, section_map, keyframes=args.keyframes)
        mark_video_processed(vid.stem, clip_count)
        print(f"\nSANITY COMPLETE")

    elif args.POC or args.FULL:
        mode = "POC" if args.POC else "FULL"
        print(f"=== {mode} MODE: Process all videos (auto-polls for new downloads) ===")
        print(f"Videos dir: {VIDEOS_DIR}")
        print(f"Clips dir:  {CLIPS_DIR}")

        total_processed = 0
        cycle = 0

        try:
            while True:
                cycle += 1
                all_videos = get_all_videos()
                processed = get_processed_video_ids()

                new_videos = []
                for v in all_videos:
                    if v.stem not in processed:
                        new_videos.append(v)
                    else:
                        actual = verify_video_clips(v.stem, section_map, CLIPS_DIR)
                        expected = processed[v.stem]
                        if expected == -1:
                            if actual == 0:
                                new_videos.append(v)
                        elif actual < expected or actual == 0:
                            print(f"  STALE: {v.stem} (expected {expected}, found {actual})")
                            new_videos.append(v)

                if new_videos:
                    print(f"\n[Cycle {cycle}] {len(new_videos)} new/stale videos "
                          f"(downloaded: {len(all_videos)}, processed: {len(processed)})")
                    batch_clips = 0
                    pbar_scene = make_pbar(total=len(new_videos), desc="m02_scene", unit="video")
                    for i, video in enumerate(new_videos, 1):
                        print(f"\n  [{i}/{len(new_videos)}] {video.name}")
                        clip_count = process_video(video, CLIPS_DIR, section_map, keyframes=args.keyframes)
                        batch_clips += clip_count
                        mark_video_processed(video.stem, clip_count)
                        pbar_scene.update(1)

                    pbar_scene.close()
                    total_processed += len(new_videos)
                    print(f"\n[Cycle {cycle}] Done: {len(new_videos)} videos -> {batch_clips} clips")
                else:
                    print(f"[Cycle {cycle}] No new videos. "
                          f"Downloaded: {len(all_videos)}, processed: {len(processed)}")

                    if len(all_videos) > 0 and total_processed > 0:
                        print(f"\nAll {len(all_videos)} videos processed. Finalizing...")
                        break

                    if len(all_videos) == 0:
                        print(f"Waiting for m01 to start downloading... ({POLL_INTERVAL}s)")

                print(f"Polling again in {POLL_INTERVAL}s (Ctrl+C to stop and finalize)...")
                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            processed = get_processed_video_ids()
            print(f"\n\nStopped early. Processed {len(processed)} videos. Finalizing...")

        print(f"\n{mode} COMPLETE")


if __name__ == "__main__":
    main()
