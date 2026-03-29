"""
Pre-download clips from HF to local WebDataset TAR shards. CPU-only.

USAGE:
    python -u src/m00d_download_subset.py --SANITY --subset data/subset_10k.json 2>&1 | tee logs/m00d_sanity.log
    python -u src/m00d_download_subset.py --POC --subset data/subset_10k.json 2>&1 | tee logs/m00d_poc.log
    python -u src/m00d_download_subset.py --FULL 2>&1 | tee logs/m00d_full.log
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import HF_DATASET_REPO, load_subset, add_subset_arg, PROJECT_ROOT
from utils.config import get_sanity_clip_limit, get_pipeline_config
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb

CLIPS_PER_SHARD = get_pipeline_config()["data"]["clips_per_shard"]
SANITY_CLIP_LIMIT = get_sanity_clip_limit("download_subset")


def _output_dir_from_subset(subset_path: str) -> Path:
    """Derive local data dir from subset filename: data/subset_10k.json → data/subset_10k_local/"""
    p = Path(subset_path)
    return p.parent / f"{p.stem}_local"


def _load_manifest(manifest_path: Path) -> dict:
    """Load existing manifest for resume. Returns empty dict if not found."""
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def _save_manifest(manifest_path: Path, manifest: dict):
    """Atomically write manifest (write to tmp, then rename)."""
    tmp = manifest_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp.rename(manifest_path)


def _write_shard(output_dir: Path, shard_idx: int,
                 buffer: list) -> Path:
    """Write a TAR shard from buffer of (key, mp4_bytes, json_bytes) tuples."""
    shard_name = f"subset-{shard_idx:05d}.tar"
    shard_path = output_dir / shard_name

    with tarfile.open(shard_path, "w") as tar:
        for hf_key, mp4_bytes, json_bytes in buffer:
            mp4_info = tarfile.TarInfo(name=f"{hf_key}.mp4")
            mp4_info.size = len(mp4_bytes)
            tar.addfile(mp4_info, io.BytesIO(mp4_bytes))

            json_info = tarfile.TarInfo(name=f"{hf_key}.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, io.BytesIO(json_bytes))

    return shard_path


def _get_clip_key_from_tar(json_bytes: bytes) -> str:
    """Extract clip key from JSON sidecar bytes."""
    try:
        meta = json.loads(json_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    section = meta.get("section", "")
    video_id = meta.get("video_id", "")
    source_file = meta.get("source_file", "")
    return f"{section}/{video_id}/{source_file}"


def download_subset(args):
    """Download HF TAR shards via CDN. With --subset: filter clips. Without: download all."""
    from huggingface_hub import HfApi, hf_hub_download
    from tqdm import tqdm

    # Load HF_TOKEN from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Load subset keys (None = no filtering = download all)
    subset_keys = load_subset(args.subset) if args.subset else set()
    target_keys = set(subset_keys) if subset_keys else None
    full_mode = target_keys is None

    if args.SANITY:
        clip_limit = SANITY_CLIP_LIMIT
    elif subset_keys:
        clip_limit = len(subset_keys)
    else:
        clip_limit = None  # download all shards for full corpus

    # Output directory
    if args.subset:
        output_dir = _output_dir_from_subset(args.subset)
    else:
        output_dir = PROJECT_ROOT / "data" / "full_local"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    mode_label = "FULL CORPUS" if full_mode else f"SUBSET ({len(subset_keys):,} keys)"
    print(f"=== m00d: Pre-download {mode_label} to local WebDataset ===")
    print(f"  Method: Direct TAR shard download (CDN, no streaming API)")
    print(f"  Subset: {args.subset} ({len(subset_keys):,} keys)")
    print(f"  Output: {output_dir}")
    print(f"  Clip limit: {f'{clip_limit:,}' if clip_limit else 'ALL'}")
    print(f"  Clips per shard: {CLIPS_PER_SHARD}")

    # Resume: load manifest to find already-saved keys
    manifest = _load_manifest(manifest_path)
    saved_keys = set(manifest.get("saved_keys", []))
    processed_hf_shards = set(manifest.get("processed_hf_shards", []))
    if saved_keys:
        print(f"  [resume] {len(saved_keys):,} clips already saved, skipping")
        print(f"  [resume] {len(processed_hf_shards)} HF shards already processed")

    # Init wandb
    wb_run = init_wandb("m00d", "download_subset",
                        {"subset": args.subset, "clip_limit": clip_limit,
                         "method": "cdn_tar_download"},
                        enabled=not args.no_wandb)

    # List all TAR shards in HF repo
    api = HfApi()
    all_files = api.list_repo_files(HF_DATASET_REPO, repo_type="dataset")
    tar_files = sorted([f for f in all_files if f.endswith(".tar")])
    print(f"\nHF repo: {HF_DATASET_REPO}")
    print(f"  TAR shards: {len(tar_files)}")

    # Remaining keys to find
    if target_keys is not None:
        remaining_keys = target_keys - saved_keys
        print(f"  Remaining clips to find: {len(remaining_keys):,}")
    else:
        remaining_keys = None  # full mode: keep all clips
        print(f"  Full mode: downloading ALL clips from ALL shards")

    found = len(saved_keys)
    scanned = 0
    out_shard_idx = manifest.get("next_shard_idx", 0)
    shard_buffer = []
    shards_written = list(manifest.get("shards", []))
    all_saved_keys = list(saved_keys)
    start_time = time.time()

    pbar = tqdm(tar_files, desc="Processing HF shards", unit="shard")

    try:
        for tar_name in pbar:
            # Skip already-processed HF shards
            if tar_name in processed_hf_shards:
                pbar.set_postfix({"found": found, "status": "skip"})
                continue

            # Check if we've found enough (None = no limit)
            if clip_limit is not None and found >= clip_limit:
                print(f"\n  Reached clip limit ({clip_limit:,})")
                break

            # Download this shard via CDN (to temp dir, deleted after processing)
            pbar.set_postfix({"found": found, "status": "downloading"})
            tmp_dir = Path("/tmp/hf_shard_tmp")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            local_tar = hf_hub_download(
                HF_DATASET_REPO,
                filename=tar_name,
                repo_type="dataset",
                local_dir=tmp_dir,
            )

            # Scan the TAR for matching clips
            pbar.set_postfix({"found": found, "status": "scanning"})
            shard_matches = 0
            try:
                with tarfile.open(local_tar, "r") as tar:
                    # Group entries by key prefix (each clip = .mp4 + .json pair)
                    entries = {}
                    for member in tar.getmembers():
                        base = member.name.rsplit(".", 1)[0]
                        ext = member.name.rsplit(".", 1)[-1] if "." in member.name else ""
                        if base not in entries:
                            entries[base] = {}
                        entries[base][ext] = member

                    for base, parts in entries.items():
                        if "json" not in parts or "mp4" not in parts:
                            continue

                        scanned += 1

                        # Read JSON to get clip key
                        json_bytes = tar.extractfile(parts["json"]).read()
                        clip_key = _get_clip_key_from_tar(json_bytes)

                        # Filter: subset mode checks remaining_keys, full mode keeps all
                        if remaining_keys is not None and clip_key not in remaining_keys:
                            continue
                        if clip_key in saved_keys:
                            continue

                        # Match! Extract mp4
                        mp4_bytes = tar.extractfile(parts["mp4"]).read()
                        if not mp4_bytes:
                            continue

                        shard_buffer.append((base, mp4_bytes, json_bytes))
                        found += 1
                        shard_matches += 1
                        all_saved_keys.append(clip_key)
                        if remaining_keys is not None:
                            remaining_keys.discard(clip_key)

                        # Write output shard when buffer full
                        if len(shard_buffer) >= CLIPS_PER_SHARD:
                            shard_path = _write_shard(output_dir, out_shard_idx, shard_buffer)
                            shards_written.append(shard_path.name)
                            out_shard_idx += 1
                            shard_buffer = []
                            print(f"  [checkpoint] shard {out_shard_idx-1}: {len(all_saved_keys):,} clips saved")

                            if wb_run:
                                log_metrics(wb_run, {"clips_saved": len(all_saved_keys),
                                                     "scanned": scanned}, step=scanned)

                        if clip_limit is not None and found >= clip_limit:
                            break

            except Exception as e:
                print(f"  WARN: error reading {tar_name}: {e}")

            # Delete downloaded shard to free disk (only need ~1.5GB at a time)
            try:
                os.remove(local_tar)
            except OSError:
                pass

            # Mark this HF shard as processed
            processed_hf_shards.add(tar_name)

            # Update manifest after each HF shard
            manifest = {
                "n": len(all_saved_keys),
                "shards": shards_written,
                "subset_file": str(args.subset),
                "next_shard_idx": out_shard_idx,
                "saved_keys": all_saved_keys,
                "processed_hf_shards": sorted(processed_hf_shards),
            }
            _save_manifest(manifest_path, manifest)

            elapsed = time.time() - start_time
            pbar.set_postfix({
                "found": found,
                "matches": shard_matches,
                "elapsed": f"{elapsed/60:.0f}m",
            })

    except KeyboardInterrupt:
        print("\n  Interrupted! Saving partial progress...")

    pbar.close()

    # Write final partial shard
    if shard_buffer:
        shard_path = _write_shard(output_dir, out_shard_idx, shard_buffer)
        shards_written.append(shard_path.name)
        out_shard_idx += 1

    # Final manifest
    manifest = {
        "n": len(all_saved_keys),
        "shards": shards_written,
        "subset_file": str(args.subset),
        "next_shard_idx": out_shard_idx,
        "saved_keys": all_saved_keys,
        "processed_hf_shards": sorted(processed_hf_shards),
    }
    _save_manifest(manifest_path, manifest)

    elapsed = time.time() - start_time
    print(f"\n=== Download complete ===")
    limit_str = f"{clip_limit:,}" if clip_limit else "ALL"
    print(f"  Clips saved: {len(all_saved_keys):,}/{limit_str}")
    print(f"  Shards written: {len(shards_written)}")
    print(f"  HF shards processed: {len(processed_hf_shards)}/{len(tar_files)}")
    print(f"  Clips scanned: {scanned:,}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Output: {output_dir}")

    # Disk usage
    total_bytes = sum((output_dir / s).stat().st_size for s in shards_written
                      if (output_dir / s).exists())
    print(f"  Disk: {total_bytes / 1e9:.2f} GB")

    # Clean up temp dir
    import shutil
    tmp_dir = Path("/tmp/hf_shard_tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("  Cleaned temp dir")

    if wb_run:
        log_metrics(wb_run, {"total_clips": len(all_saved_keys),
                              "total_shards": len(shards_written),
                              "elapsed_min": elapsed / 60,
                              "disk_gb": total_bytes / 1e9})
    finish_wandb(wb_run)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-download subset clips from HF to local WebDataset TAR shards")
    parser.add_argument("--SANITY", action="store_true",
                        help=f"Download first {SANITY_CLIP_LIMIT} matching clips only")
    parser.add_argument("--FULL", action="store_true",
                        help="Download all subset clips (default behavior)")
    add_subset_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not args.subset:
        parser.print_help()
        print("\nERROR: --subset is required")
        sys.exit(1)

    download_subset(args)


if __name__ == "__main__":
    main()
