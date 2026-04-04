"""
Pack clips into WebDataset TAR shards and upload to HuggingFace.
Streaming: create shard -> upload -> delete local -> next. Only ~1GB on disk at a time.

USAGE:
    tmux
    caffeinate -s python -u src/m03_pack_shards.py --SANITY 2>&1 | tee logs/m03_pack_shards_sanity.log
    caffeinate -s python -u src/m03_pack_shards.py --POC 2>&1 | tee logs/m03_pack_shards_poc.log
    caffeinate -s python -u src/m03_pack_shards.py --FULL 2>&1 | tee logs/m03_pack_shards_full.log
"""
import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.progress import make_pbar
from utils.config import (
    CLIPS_DIR, SHARDS_DIR, HF_DATASET_REPO, OUTPUTS_DATA_PREP_DIR,
)
from utils.config import get_pipeline_config
from utils.hf_utils import _setup_hf_env, _get_token, generate_readme, upload_readme

CLIP_DURATIONS_JSON = OUTPUTS_DATA_PREP_DIR / "clip_durations.json"
CLIPS_PER_SHARD = get_pipeline_config()["data"]["clips_per_shard"]


def build_manifest() -> list:
    """Build sorted clip manifest from clip_durations.json.
    Returns list of (clip_path, metadata_dict) sorted by section/video/clip."""
    if not CLIP_DURATIONS_JSON.exists():
        print(f"ERROR: {CLIP_DURATIONS_JSON} not found")
        print("Run: python -u src/m02b_scene_fetch_duration.py --FULL")
        sys.exit(1)

    with open(CLIP_DURATIONS_JSON) as f:
        data = json.load(f)

    manifest = []
    for section, sec_data in sorted(data["sections"].items()):
        # Parse tier/city/tour_type from section path
        parts = section.split("/")
        if parts[0] in ("tier1", "tier2"):
            tier, city = parts[0], parts[1]
            tour_type = parts[2] if len(parts) > 2 else "unknown"
        elif parts[0] == "goa":
            tier, city, tour_type = "goa", "goa", parts[1] if len(parts) > 1 else "walking"
        elif parts[0] == "monuments":
            tier, city, tour_type = "monuments", parts[1] if len(parts) > 1 else "unknown", "monument"
        else:
            tier, city, tour_type = parts[0], parts[0], "unknown"

        for video_id, clips in sorted(sec_data["videos"].items()):
            for clip in clips:
                clip_path = CLIPS_DIR / section / clip["file"]
                # Handle kolkata/walking split into part1/part2/part3
                if not clip_path.exists():
                    for part in ["part1", "part2", "part3"]:
                        alt = CLIPS_DIR / section / part / clip["file"]
                        if alt.exists():
                            clip_path = alt
                            break

                metadata = {
                    "video_id": video_id,
                    "section": section,
                    "tier": tier,
                    "city": city,
                    "tour_type": tour_type,
                    "duration_sec": clip["duration_sec"],
                    "size_mb": clip["size_mb"],
                    "source_file": clip["file"],
                }
                manifest.append((clip_path, metadata))

    return manifest


def create_shard(shard_idx: int, clips: list, global_offset: int) -> Path:
    """Create a single TAR shard from a list of (clip_path, metadata) tuples.
    Returns path to the created tar file."""
    shard_name = f"train-{shard_idx:05d}.tar"
    shard_path = SHARDS_DIR / shard_name

    shard_size = 0
    with tarfile.open(shard_path, "w") as tar:
        for i, (clip_path, metadata) in enumerate(clips):
            global_idx = global_offset + i
            prefix = f"{global_idx:06d}"

            # Add mp4
            if not clip_path.exists():
                print(f"  WARN: missing {clip_path}, skipping")
                continue
            tar.add(str(clip_path), arcname=f"{prefix}.mp4")
            shard_size += clip_path.stat().st_size

            # Add json sidecar
            meta_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo(name=f"{prefix}.json")
            info.size = len(meta_bytes)
            tar.addfile(info, io.BytesIO(meta_bytes))

    return shard_path


def main():
    parser = argparse.ArgumentParser(description="Pack clips into WebDataset TAR shards and upload to HF")
    parser.add_argument("--SANITY", action="store_true", help="Process first 2 shards only")
    parser.add_argument("--POC", action="store_true", help="10K subset")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    # Step 1: Build manifest
    print("Step 1: Building clip manifest from clip_durations.json...")
    manifest = build_manifest()
    print(f"  Total clips in manifest: {len(manifest):,}")

    # Verify clips exist
    missing = sum(1 for p, _ in manifest if not p.exists())
    if missing > 0:
        print(f"  WARNING: {missing} clips missing on disk")

    # Step 2+3: Create shards and upload (streaming)
    _setup_hf_env()
    token = _get_token()
    if not token:
        print("ERROR: HF_TOKEN not found in .env")
        sys.exit(1)

    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)

    repo_id = HF_DATASET_REPO

    # Create/ensure repo
    try:
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True, private=True)
        print(f"  Repository: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"  Repo note: {e}")

    # Calculate shards
    total_clips = len(manifest)
    total_shards = (total_clips + CLIPS_PER_SHARD - 1) // CLIPS_PER_SHARD

    if args.SANITY:
        total_shards = min(2, total_shards)
        manifest = manifest[:total_shards * CLIPS_PER_SHARD]
        total_clips = len(manifest)
        print(f"  SANITY mode: {total_shards} shards, {total_clips} clips")

    print(f"\nStep 2+3: Create {total_shards} shards → upload → delete (streaming)")
    print(f"  Clips per shard: {CLIPS_PER_SHARD}")
    print(f"  Total clips: {total_clips:,}")

    total_size_uploaded = 0
    total_clips_uploaded = 0
    start = time.time()
    pbar = make_pbar(total=total_shards, desc="m03_shards", unit="shard")

    for shard_idx in range(total_shards):
        shard_start = time.time()
        offset = shard_idx * CLIPS_PER_SHARD
        batch = manifest[offset:offset + CLIPS_PER_SHARD]

        # Create shard
        shard_path = create_shard(shard_idx, batch, offset)
        shard_size_mb = shard_path.stat().st_size / (1024 ** 2)

        # Upload shard
        shard_name = shard_path.name
        try:
            api.upload_file(
                path_or_fileobj=str(shard_path),
                path_in_repo=f"data/{shard_name}",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            total_size_uploaded += shard_size_mb
            total_clips_uploaded += len(batch)

            # Delete local shard
            shard_path.unlink()

            elapsed = time.time() - start
            shard_dur = time.time() - shard_start
            rate_mbps = shard_size_mb / shard_dur if shard_dur > 0 else 0
            remaining = total_shards - (shard_idx + 1)
            eta_min = remaining * (elapsed / (shard_idx + 1)) / 60 if shard_idx > 0 else 0

            print(f"  [{shard_idx + 1}/{total_shards}] {shard_name} "
                  f"({shard_size_mb:.0f} MB, {len(batch)} clips) "
                  f"@ {rate_mbps:.1f} MB/s | "
                  f"total={total_size_uploaded / 1024:.1f} GB | "
                  f"ETA={eta_min:.0f} min")

        except Exception as e:
            print(f"  [{shard_idx + 1}/{total_shards}] ERROR uploading {shard_name}: {e}")
            print(f"  Shard kept at: {shard_path}")
            print(f"  Re-run to retry (already-uploaded shards will be overwritten)")
        pbar.update(1)

    pbar.close()

    # Step 4: Upload README
    print(f"\nStep 4: Uploading README.md...")
    num_videos = len(set(m["video_id"] for _, m in manifest))
    total_gb = sum(m["size_mb"] for _, m in manifest) / 1024
    readme = generate_readme(total_clips, num_videos, total_gb, num_shards=total_shards)
    upload_readme(repo_id, token, readme)

    # Summary
    elapsed = time.time() - start
    print(f"\n{'=' * 50}")
    print(f"=== UPLOAD COMPLETE ===")
    print(f"Shards uploaded: {total_shards}")
    print(f"Clips: {total_clips_uploaded:,}")
    print(f"Size: {total_size_uploaded / 1024:.1f} GB")
    print(f"Duration: {elapsed / 3600:.1f} hours")
    print(f"Dataset: https://huggingface.co/datasets/{repo_id}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
