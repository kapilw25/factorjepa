"""
Scan all clips in src/data/clips/ and output per-clip durations to JSON.
Flags clips outside [4s, 10s] range. JSON mirrors the directory hierarchy.

USAGE:
    python -u src/m02b_scene_fetch_duration.py --SANITY 2>&1 | tee logs/m02b_scene_fetch_duration_sanity.log
    python -u src/m02b_scene_fetch_duration.py --FULL 2>&1 | tee logs/m02b_scene_fetch_duration_full.log
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    CLIPS_DIR, CLIP_MIN_DURATION, CLIP_MAX_DURATION, get_video_duration,
    PROJECT_ROOT,
)

OUTPUT_JSON = PROJECT_ROOT / "outputs_data_prep" / "clip_durations.json"


def extract_video_id(clip_stem: str) -> str:
    """Extract video_id from clip filename stem."""
    if "-Scene-" in clip_stem:
        return clip_stem.split("-Scene-")[0]
    # fixed-duration pattern: {video_id}-{NNN}
    m = re.match(r"^(.+)-\d{3}$", clip_stem)
    if m:
        return m.group(1)
    return clip_stem


def probe_clip(clip_path: Path) -> dict:
    """Get duration and size for one clip."""
    dur = get_video_duration(clip_path)
    size_bytes = clip_path.stat().st_size
    if dur > CLIP_MAX_DURATION:
        status = "too_long"
    elif dur < CLIP_MIN_DURATION:
        status = "too_short"
    else:
        status = "ok"
    return {
        "file": clip_path.name,
        "duration_sec": round(dur, 2),
        "size_mb": round(size_bytes / (1024 ** 2), 2),
        "status": status,
        "_section": str(clip_path.parent.relative_to(CLIPS_DIR)),
        "_video_id": extract_video_id(clip_path.stem),
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch durations of all clips")
    parser.add_argument("--SANITY", action="store_true", help="Process 1 section only")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    all_clips = sorted(CLIPS_DIR.rglob("*.mp4"))
    if not all_clips:
        print(f"ERROR: No clips found in {CLIPS_DIR}")
        sys.exit(1)

    # Group by section (leaf dir relative to CLIPS_DIR)
    section_clips = {}
    for c in all_clips:
        sec = str(c.parent.relative_to(CLIPS_DIR))
        section_clips.setdefault(sec, []).append(c)

    if args.SANITY:
        first = sorted(section_clips.keys())[0]
        section_clips = {first: section_clips[first]}
        all_clips = section_clips[first]
        print(f"SANITY: section '{first}' only ({len(all_clips)} clips)")

    print(f"Scanning {len(all_clips)} clips across {len(section_clips)} sections ...")

    # Probe all clips in parallel
    start = time.time()
    clip_infos = []
    done = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(probe_clip, c): c for c in all_clips}
        for fut in as_completed(futures):
            clip_infos.append(fut.result())
            done += 1
            if done % 5000 == 0 or done == len(all_clips):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed else 1
                eta = int((len(all_clips) - done) / rate) if rate else 0
                print(f"  [{done:>6}/{len(all_clips)}] rate={rate:.0f}/s ETA={eta}s")

    # Organize into hierarchy: section → video_id → [clips]
    tree = {}
    for info in clip_infos:
        sec = info.pop("_section")
        vid = info.pop("_video_id")
        tree.setdefault(sec, {}).setdefault(vid, []).append(info)

    # Sort clips within each video
    for sec in tree:
        for vid in tree[sec]:
            tree[sec][vid].sort(key=lambda x: x["file"])

    # Build per-section summaries
    total_ok = total_long = total_short = 0
    total_dur = total_size = 0.0
    sections_out = {}

    for sec in sorted(tree.keys()):
        videos = tree[sec]
        s_ok = s_long = s_short = 0
        for vid, clips in sorted(videos.items()):
            for c in clips:
                if c["status"] == "ok":
                    s_ok += 1
                elif c["status"] == "too_long":
                    s_long += 1
                else:
                    s_short += 1
                total_dur += c["duration_sec"]
                total_size += c["size_mb"]

        total_ok += s_ok
        total_long += s_long
        total_short += s_short

        sections_out[sec] = {
            "total_clips": s_ok + s_long + s_short,
            "ok": s_ok,
            "too_long": s_long,
            "too_short": s_short,
            "videos": {v: clips for v, clips in sorted(videos.items())},
        }

    n = len(all_clips)
    output = {
        "summary": {
            "total_clips": n,
            "in_range_4_10s": total_ok,
            "too_long_above_10s": total_long,
            "too_short_below_4s": total_short,
            "violation_pct": round((total_long + total_short) / n * 100, 1) if n else 0,
            "total_duration_hours": round(total_dur / 3600, 1),
            "total_size_gb": round(total_size / 1024, 1),
            "clip_min_duration": CLIP_MIN_DURATION,
            "clip_max_duration": CLIP_MAX_DURATION,
        },
        "sections": sections_out,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    # Generate metadata.jsonl in each leaf directory (for HF dataset viewer)
    meta_count = 0
    for sec, sec_data in sorted(sections_out.items()):
        sec_dir = CLIPS_DIR / sec
        if not sec_dir.exists():
            continue
        lines = []
        for vid, clips in sorted(sec_data["videos"].items()):
            for c in clips:
                lines.append(json.dumps({
                    "file_name": c["file"],
                    "video_id": vid,
                    "section": sec,
                    "duration_sec": c["duration_sec"],
                    "size_mb": c["size_mb"],
                }))
        if lines:
            (sec_dir / "metadata.jsonl").write_text("\n".join(lines) + "\n")
            meta_count += 1

    print(f"\nGenerated {meta_count} metadata.jsonl files in {CLIPS_DIR}")

    elapsed = time.time() - start
    print(f"\n=== Clip Duration Summary ===")
    print(f"Total clips:      {n}")
    print(f"In range [4-10s]: {total_ok} ({total_ok / n * 100:.1f}%)")
    print(f"Too long  (>10s): {total_long} ({total_long / n * 100:.1f}%)")
    print(f"Too short (<4s):  {total_short} ({total_short / n * 100:.1f}%)")
    print(f"Total duration:   {total_dur / 3600:.1f} hours")
    print(f"Total size:       {total_size / 1024:.1f} GB")
    print(f"Scan time:        {elapsed:.0f}s")
    print(f"Output:           {OUTPUT_JSON}")

    # Print top 20 worst violations
    if total_long > 0:
        print(f"\n=== Top violations (>10s) ===")
        violations = []
        for sec in sections_out:
            for vid, clips in sections_out[sec]["videos"].items():
                for c in clips:
                    if c["status"] == "too_long":
                        violations.append((sec, c["file"], c["duration_sec"]))
        violations.sort(key=lambda x: -x[2])
        for sec, fname, dur in violations[:20]:
            print(f"  {sec}/{fname} → {dur}s")
        if len(violations) > 20:
            print(f"  ... and {len(violations) - 20} more")


if __name__ == "__main__":
    main()
