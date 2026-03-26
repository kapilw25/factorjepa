"""
Scan all clips in src/data/clips/ and output per-clip durations to JSON.
Flags clips outside [4s, 10s] range. JSON mirrors the directory hierarchy.

USAGE:
    python -u src/m02b_scene_fetch_duration.py --SANITY 2>&1 | tee logs/m02b_scene_fetch_duration_sanity.log
    python -u src/m02b_scene_fetch_duration.py --FULL 2>&1 | tee logs/m02b_scene_fetch_duration_full.log
    
    # Print clips-per-city table from existing JSON (no scanning)
    python -u src/m02b_scene_fetch_duration.py --stats
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
    OUTPUTS_DATA_PREP_DIR, PROJECT_ROOT,
)

OUTPUT_JSON = OUTPUTS_DATA_PREP_DIR / "clip_durations.json"


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


def print_clips_per_city():
    """Read existing clip_durations.json and print clips-per-city tables."""
    if not OUTPUT_JSON.exists():
        print(f"ERROR: {OUTPUT_JSON} not found. Run --FULL first.")
        sys.exit(1)

    with open(OUTPUT_JSON, "r") as f:
        data = json.load(f)

    sections = data.get("sections", {})
    summary = data.get("summary", {})

    def get_clips(prefix):
        return sum(v["total_clips"] for k, v in sections.items() if k.startswith(prefix))

    def get_dur_gb(prefix):
        dur = 0.0
        size = 0.0
        for k, v in sections.items():
            if k.startswith(prefix):
                for _, clips in v.get("videos", {}).items():
                    for c in clips:
                        dur += c.get("duration_sec", 0)
                        size += c.get("size_mb", 0)
        return dur / 3600, size / 1024

    tier1_cities = ["kolkata", "chennai", "bangalore", "mumbai", "delhi", "hyderabad"]
    tier2_cities = ["jaipur", "varanasi", "lucknow", "ahmedabad", "pune", "kochi",
                    "chandigarh", "indore", "bhopal", "coimbatore", "nagpur",
                    "visakhapatnam", "surat", "thiruvananthapuram", "mysuru"]

    total_clips = summary.get("total_clips", 0)
    total_hrs = summary.get("total_duration_hours", 0)
    total_gb = summary.get("total_size_gb", 0)

    print(f"\n{'=' * 90}")
    print(f"CLIPS PER CITY ({total_clips:,} total clips | {total_hrs:.1f} hours | {total_gb:.1f} GB)")
    print(f"{'=' * 90}")

    # ===== TIER 1 =====
    print(f"\n--- Tier 1 Cities (6 metros) ---")
    print(f"{'City':<20} {'Drive':>8} {'Walk':>8} {'Drone':>8} {'Total':>8} {'Hrs':>7} {'GB':>7}")
    print("-" * 75)

    t1 = {"drive": 0, "walk": 0, "drone": 0, "total": 0}
    t1_dur = t1_size = 0.0
    for city in sorted(tier1_cities, key=lambda c: -get_clips(f"tier1/{c}")):
        d = get_clips(f"tier1/{city}/drive")
        w = get_clips(f"tier1/{city}/walking")
        dr = get_clips(f"tier1/{city}/drone")
        total = d + w + dr
        hrs, gb = get_dur_gb(f"tier1/{city}")
        print(f"{city.capitalize():<20} {d:>8,} {w:>8,} {dr:>8,} {total:>8,} {hrs:>7.1f} {gb:>7.1f}")
        t1["drive"] += d; t1["walk"] += w; t1["drone"] += dr; t1["total"] += total
        t1_dur += hrs; t1_size += gb

    print("-" * 75)
    print(f"{'Tier 1 Total':<20} {t1['drive']:>8,} {t1['walk']:>8,} {t1['drone']:>8,} {t1['total']:>8,} {t1_dur:>7.1f} {t1_size:>7.1f}")

    # ===== GOA =====
    goa_w = get_clips("goa/walking")
    goa_hrs, goa_gb = get_dur_gb("goa")
    print(f"\n{'Goa':<20} {'':>8} {goa_w:>8,} {'':>8} {goa_w:>8,} {goa_hrs:>7.1f} {goa_gb:>7.1f}")

    # ===== TIER 2 =====
    print(f"\n--- Tier 2 Cities (15 cities) ---")
    print(f"{'City':<20} {'Drive':>8} {'Walk':>8} {'Drone':>8} {'Rain':>8} {'Total':>8} {'Hrs':>7} {'GB':>7}")
    print("-" * 85)

    t2 = {"drive": 0, "walk": 0, "drone": 0, "rain": 0, "total": 0}
    t2_dur = t2_size = 0.0
    for city in sorted(tier2_cities, key=lambda c: -get_clips(f"tier2/{c}")):
        d = get_clips(f"tier2/{city}/drive")
        w = get_clips(f"tier2/{city}/walking")
        dr = get_clips(f"tier2/{city}/drone")
        r = get_clips(f"tier2/{city}/rain")
        total = d + w + dr + r
        hrs, gb = get_dur_gb(f"tier2/{city}")
        print(f"{city.capitalize():<20} {d:>8,} {w:>8,} {dr:>8,} {r:>8,} {total:>8,} {hrs:>7.1f} {gb:>7.1f}")
        t2["drive"] += d; t2["walk"] += w; t2["drone"] += dr; t2["rain"] += r; t2["total"] += total
        t2_dur += hrs; t2_size += gb

    print("-" * 85)
    print(f"{'Tier 2 Total':<20} {t2['drive']:>8,} {t2['walk']:>8,} {t2['drone']:>8,} {t2['rain']:>8,} {t2['total']:>8,} {t2_dur:>7.1f} {t2_size:>7.1f}")

    # ===== MONUMENTS =====
    mon_clips = get_clips("monuments")
    mon_hrs, mon_gb = get_dur_gb("monuments")
    mon_sections = {k: v for k, v in sections.items() if k.startswith("monuments")}
    if mon_sections:
        print(f"\n--- Monuments ---")
        print(f"{'Monument':<40} {'Clips':>8} {'Hrs':>7} {'GB':>7}")
        print("-" * 65)
        for k in sorted(mon_sections.keys()):
            name = k.replace("monuments/", "").replace("_", " ").title()
            c = mon_sections[k]["total_clips"]
            h, g = get_dur_gb(k)
            print(f"{name:<40} {c:>8,} {h:>7.1f} {g:>7.1f}")
        print("-" * 65)
        print(f"{'Monuments Total':<40} {mon_clips:>8,} {mon_hrs:>7.1f} {mon_gb:>7.1f}")

    # ===== GRAND TOTAL =====
    grand = t1["total"] + goa_w + t2["total"] + mon_clips
    grand_hrs = t1_dur + goa_hrs + t2_dur + mon_hrs
    grand_gb = t1_size + goa_gb + t2_size + mon_gb

    print(f"\n{'=' * 90}")
    print(f"{'GRAND TOTAL':<20} {'':>42} {grand:>8,} {grand_hrs:>7.1f} {grand_gb:>7.1f}")
    print(f"{'=' * 90}")

    # ===== Export docs/static/stats.json for webpage =====
    from datetime import date

    def city_row(city, prefix, has_rain=False):
        d = get_clips(f"{prefix}/drive")
        w = get_clips(f"{prefix}/walking")
        dr = get_clips(f"{prefix}/drone")
        r = get_clips(f"{prefix}/rain") if has_rain else 0
        total = d + w + dr + r
        hrs, gb = get_dur_gb(prefix)
        row = {"city": city, "drive": d, "walk": w, "drone": dr, "total": total,
               "hours": round(hrs, 1), "gb": round(gb, 1)}
        if has_rain:
            row["rain"] = r
        return row

    stats = {
        "summary": {
            "total_clips": grand,
            "total_videos": 714,
            "total_cities": len(tier1_cities) + len(tier2_cities) + 1,
            "total_hours": round(grand_hrs, 1),
            "total_gb": round(grand_gb, 1),
            "taxonomy_fields": 16,
            "last_updated": str(date.today()),
        },
        "tier1": {
            "label": "Tier 1 Cities",
            "description": "6 metros",
            "cities": [city_row(c.capitalize(), f"tier1/{c}") for c in sorted(tier1_cities, key=lambda c: -get_clips(f"tier1/{c}"))],
            "totals": {"drive": t1["drive"], "walk": t1["walk"], "drone": t1["drone"],
                       "total": t1["total"], "hours": round(t1_dur, 1), "gb": round(t1_size, 1)},
        },
        "goa": {
            "walk": goa_w, "total": goa_w,
            "hours": round(goa_hrs, 1), "gb": round(goa_gb, 1),
        },
        "tier2": {
            "label": "Tier 2 Cities",
            "description": "15 cities",
            "cities": [city_row(c.capitalize(), f"tier2/{c}", has_rain=True) for c in sorted(tier2_cities, key=lambda c: -get_clips(f"tier2/{c}"))],
            "totals": {"drive": t2["drive"], "walk": t2["walk"], "drone": t2["drone"],
                       "rain": t2["rain"], "total": t2["total"],
                       "hours": round(t2_dur, 1), "gb": round(t2_size, 1)},
        },
        "monuments": {
            "sites": [],
            "totals": {"total": mon_clips, "hours": round(mon_hrs, 1), "gb": round(mon_gb, 1)},
        },
    }

    for k in sorted(mon_sections.keys()):
        name = k.replace("monuments/", "").replace("_", " ").title()
        c = mon_sections[k]["total_clips"]
        h, g = get_dur_gb(k)
        stats["monuments"]["sites"].append({"name": name, "clips": c, "hours": round(h, 1), "gb": round(g, 1)})

    stats_path = PROJECT_ROOT / "docs" / "static" / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nExported: {stats_path}")


if __name__ == "__main__":
    # Check for --stats before main argparse (no scanning needed)
    if "--stats" in sys.argv:
        print_clips_per_city()
    else:
        main()
