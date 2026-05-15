"""
Pre-download clips from HF to local WebDataset TAR shards. CPU-only.
Parallel download (8 workers) — see DOWNLOAD_WORKERS.

USAGE:
    python -u src/m00d_download_subset.py --SANITY \
        --subset data/subset_10k_local/subset_10k.json \
        --master-tags data/full_local/tags.json \
        2>&1 | tee logs/m00d_sanity.log
    python -u src/m00d_download_subset.py --FULL \
        --subset data/ultra_hard_3066.json \
        --master-tags data/full_local/tags.json \
        --no-wandb 2>&1 | tee logs/m00d_ultra_hard.log
    python -u src/m00d_download_subset.py --FULL \
        --master-tags data/full_local/tags.json \
        --no-wandb 2>&1 | tee logs/m00d_full.log
    # Skip tags filter (rare — m10 will FATAL on tags.json absence):
    python -u src/m00d_download_subset.py --FULL \
        --subset data/val_1k_local/val_1k.json \
        --master-tags data/full_local/tags.json --no-tags-filter \
        --no-wandb 2>&1 | tee logs/m00d_val_1k.log
"""
import argparse
import io
import json
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Disable hf_transfer: its multi-stream per file conflicts with parallel workers
# (N workers × 8+ streams = CDN throttling/timeouts). Worker-level parallelism is better.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import HF_DATASET_REPO, load_subset, add_subset_arg, PROJECT_ROOT
from utils.config import get_sanity_clip_limit, get_pipeline_config
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb

CLIPS_PER_SHARD = get_pipeline_config()["data"]["clips_per_shard"]
SANITY_CLIP_LIMIT = get_sanity_clip_limit("download_subset")
DOWNLOAD_WORKERS = 8  # parallel HF shard downloads (8 × ~400MB = ~3.2GB temp disk)


def _download_one_shard(tar_name: str, tmp_dir: Path) -> str:
    """Download a single HF shard to its own temp dir. Returns local path. Thread-safe."""
    from huggingface_hub import hf_hub_download
    # Each worker gets a unique subdir to avoid HF Hub cache race conditions
    # (parallel workers sharing one local_dir causes temp file collisions)
    shard_stem = Path(tar_name).stem  # e.g. "train-00042"
    worker_dir = tmp_dir / shard_stem
    worker_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        HF_DATASET_REPO,
        filename=tar_name,
        repo_type="dataset",
        local_dir=str(worker_dir),
    )


def _scan_shard(local_tar: str, remaining_keys, saved_keys) -> list:
    """Scan a downloaded TAR, extract matching clips. Returns (base, mp4_bytes, json_bytes, clip_key) list."""
    matches = []
    with tarfile.open(local_tar, "r") as tar:
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

            json_bytes = tar.extractfile(parts["json"]).read()
            clip_key = _get_clip_key_from_tar(json_bytes)

            if remaining_keys is not None and clip_key not in remaining_keys:
                continue
            if clip_key in saved_keys:
                continue

            mp4_bytes = tar.extractfile(parts["mp4"]).read()
            if not mp4_bytes:
                continue

            matches.append((base, mp4_bytes, json_bytes, clip_key))

    return matches


def _output_dir_from_subset(subset_path: str) -> Path:
    """Derive local data dir from subset filename.

    Two layouts supported (iter15 Phase B, 2026-05-15):
      legacy: data/subset_10k.json                  → data/subset_10k_local/
      new:    data/subset_10k_local/subset_10k.json → data/subset_10k_local/
    """
    p = Path(subset_path)
    # New layout: subset file is already INSIDE its <stem>_local/ dir.
    if p.parent.name == f"{p.stem}_local":
        return p.parent
    return p.parent / f"{p.stem}_local"


def _filter_master_tags(saved_keys: list, master_tags_path: Path,
                        output_path: Path) -> None:
    """Filter the master m04 tags.json to a subset's saved clip_keys; write
    `<output_dir>/tags.json` so downstream m10 (line 142 FATAL guard),
    m06 (per #77 auto-derive), and m09c (--probe-tags) find it.

    Convention: every `*_local/` dir produced by m00d MUST contain a tags.json
    matching its TAR shards' clip set. Mirrors data/{val_1k,subset_10k,eval_10k}_local/.

    Args:
        saved_keys: list of clip_keys actually saved to the local TARs (from manifest)
        master_tags_path: data/full_local/tags.json (master m04 VLM-tag list of 115,687 dicts)
        output_path: <output_dir>/tags.json — filtered LIST (same schema as master)
    """
    if not master_tags_path.is_file():
        print(f"FATAL: master tags.json missing at {master_tags_path}.")
        print(f"  Run ./git_pull.sh or python -u src/utils/hf_outputs.py download-data first")
        print(f"  to fetch m04 VLM tags from anonymousML123/factorjepa-outputs.")
        sys.exit(1)

    with open(master_tags_path) as f:
        master = json.load(f)
    if not isinstance(master, list):
        print(f"FATAL: {master_tags_path} is type {type(master).__name__}; expected list "
              f"of clip-tag dicts (m04 VLM output schema).")
        sys.exit(1)

    saved_set = set(saved_keys)

    def _key(t: dict) -> str:
        return f"{t.get('section', '')}/{t.get('video_id', '')}/{t.get('source_file', '')}"

    filtered = [t for t in master if _key(t) in saved_set]

    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"  Filtered tags.json: {len(filtered)}/{len(master)} master entries -> {output_path}")
    n_missing = len(saved_set) - len(filtered)
    if n_missing > 0:
        print(f"  WARNING: {n_missing} saved clip_keys lack a tag in master "
              f"(m04 may not have run on those clips; downstream m10/m06 will skip them)")


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
    """Download HF TAR shards via CDN with parallel workers + hf_transfer."""
    from huggingface_hub import HfApi
    from tqdm import tqdm

    # Check hf_transfer availability
    try:
        import hf_transfer  # noqa: F401
        hf_xfer = True
    except ImportError:
        hf_xfer = False

    # Load HF_TOKEN from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        hf_xfer = hf_xfer  # no-op to avoid bare except

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
    print(f"  Method: Parallel TAR download ({DOWNLOAD_WORKERS} workers, CDN)")
    print(f"  hf_transfer: {'enabled (Rust multi-stream)' if hf_xfer else 'not installed'}")
    print(f"  Subset: {args.subset} ({len(subset_keys):,} keys)")
    print(f"  Output: {output_dir}")
    print(f"  Clip limit: {f'{clip_limit:,}' if clip_limit else 'ALL'}")
    print(f"  Clips per shard: {CLIPS_PER_SHARD}")

    # Resume: load manifest to find already-saved keys
    manifest = _load_manifest(manifest_path)
    # Verify claimed shards actually exist on disk (guards against stale manifest)
    claimed_shards = manifest.get("shards", [])
    missing = [s for s in claimed_shards if not (output_dir / s).exists()]
    if missing:
        print(f"  [resume] STALE MANIFEST: {len(missing)}/{len(claimed_shards)} TARs missing on disk. Resetting.")
        manifest = {}
    saved_keys = set(manifest.get("saved_keys", []))
    processed_hf_shards = set(manifest.get("processed_hf_shards", []))
    if saved_keys:
        print(f"  [resume] {len(saved_keys):,} clips already saved, skipping")
        print(f"  [resume] {len(processed_hf_shards)} HF shards already processed")

    # Init wandb
    wb_run = init_wandb("m00d", "download_subset",
                        {"subset": args.subset, "clip_limit": clip_limit,
                         "method": "parallel_cdn_download",
                         "workers": DOWNLOAD_WORKERS, "hf_transfer": hf_xfer},
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

    # Filter to shards that still need processing
    pending_shards = [t for t in tar_files if t not in processed_hf_shards]
    print(f"  Shards to process: {len(pending_shards)} (skipping {len(tar_files) - len(pending_shards)} cached)")

    found = len(saved_keys)
    scanned = 0
    out_shard_idx = manifest.get("next_shard_idx", 0)
    shard_buffer = []
    shards_written = list(manifest.get("shards", []))
    all_saved_keys = list(saved_keys)
    start_time = time.time()
    download_errors = 0

    # Parallel download + sequential scan pipeline
    # Each worker holds 1 shard on disk (~400MB), so ~1.6GB temp at peak (4 workers)
    tmp_dir = Path("/tmp/hf_shard_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(pending_shards), desc="Processing HF shards",
                unit="shard", initial=0)

    try:
        # Process in batches of DOWNLOAD_WORKERS to maintain backpressure
        for batch_start in range(0, len(pending_shards), DOWNLOAD_WORKERS):
            if clip_limit is not None and found >= clip_limit:
                print(f"\n  Reached clip limit ({clip_limit:,})")
                break

            batch = pending_shards[batch_start:batch_start + DOWNLOAD_WORKERS]

            # Submit parallel downloads
            download_results = {}
            with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
                futures = {
                    pool.submit(_download_one_shard, tar_name, tmp_dir): tar_name
                    for tar_name in batch
                }

                # Collect results as downloads complete (fastest first)
                for future in as_completed(futures):
                    tar_name = futures[future]
                    try:
                        local_tar = future.result()
                        download_results[tar_name] = local_tar
                    except Exception as e:
                        print(f"  FATAL: download failed for {tar_name}: {e}")
                        download_errors += 1

            # Scan each downloaded shard sequentially (safe for shared state)
            for tar_name, local_tar in download_results.items():
                if not os.path.exists(local_tar):
                    print(f"  FATAL: downloaded file vanished: {local_tar}")
                    download_errors += 1
                    continue
                matches = _scan_shard(local_tar, remaining_keys, saved_keys)

                for base, mp4_bytes, json_bytes, clip_key in matches:
                    if clip_limit is not None and found >= clip_limit:
                        break

                    shard_buffer.append((base, mp4_bytes, json_bytes))
                    found += 1
                    all_saved_keys.append(clip_key)
                    saved_keys.add(clip_key)
                    if remaining_keys is not None:
                        remaining_keys.discard(clip_key)

                    # Write output shard when buffer full
                    if len(shard_buffer) >= CLIPS_PER_SHARD:
                        shard_path = _write_shard(output_dir, out_shard_idx, shard_buffer)
                        shards_written.append(shard_path.name)
                        out_shard_idx += 1
                        shard_buffer = []

                        if wb_run:
                            log_metrics(wb_run, {"clips_saved": len(all_saved_keys),
                                                 "scanned": scanned}, step=scanned)

                scanned += len(matches)

                # Delete downloaded shard to free disk
                if os.path.exists(local_tar):
                    os.remove(local_tar)

                # Mark this HF shard as processed
                processed_hf_shards.add(tar_name)
                pbar.update(1)

                elapsed = time.time() - start_time
                pbar.set_postfix({"found": found, "elapsed": f"{elapsed/60:.0f}m"})

            # Checkpoint after each batch of parallel downloads
            manifest = {
                "n": len(all_saved_keys),
                "shards": shards_written,
                "subset_file": str(args.subset),
                "next_shard_idx": out_shard_idx,
                "saved_keys": all_saved_keys,
                "processed_hf_shards": sorted(processed_hf_shards),
            }
            _save_manifest(manifest_path, manifest)

    except KeyboardInterrupt:
        print("\n  Interrupted! Saving partial progress...")

    pbar.close()

    if download_errors > 0:
        print(f"FATAL: {download_errors} shard download(s) failed")
        sys.exit(1)

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

    # Filter master m04 tags.json into output_dir/tags.json so downstream m10/m06/m09c
    # find it at the canonical `<*_local>/tags.json` path. Skip if --no-tags-filter.
    if not getattr(args, "no_tags_filter", False):
        master_tags = Path(args.master_tags)
        out_tags = output_dir / "tags.json"
        print(f"\nFiltering master tags ({master_tags}) -> {out_tags} ...")
        _filter_master_tags(all_saved_keys, master_tags, out_tags)

    elapsed = time.time() - start_time
    print(f"\n=== Download complete ===")
    limit_str = f"{clip_limit:,}" if clip_limit else "ALL"
    print(f"  Clips saved: {len(all_saved_keys):,}/{limit_str}")
    print(f"  Shards written: {len(shards_written)}")
    print(f"  HF shards processed: {len(processed_hf_shards)}/{len(tar_files)}")
    print(f"  Clips scanned: {scanned:,}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Output: {output_dir}")
    print(f"  hf_transfer: {'yes' if hf_xfer else 'no'}")
    print(f"  Workers: {DOWNLOAD_WORKERS}")

    # Disk usage
    total_bytes = sum((output_dir / s).stat().st_size for s in shards_written
                      if (output_dir / s).exists())
    print(f"  Disk: {total_bytes / 1e9:.2f} GB")

    # Clean up temp dir
    shutil.rmtree("/tmp/hf_shard_tmp", ignore_errors=True)
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
    parser.add_argument("--master-tags", required=True,
                        help="Master m04 tags.json to filter into <output_dir>/tags.json. "
                             "Required by m10 (FATAL guard L142), m06 (#77 auto-derive), m09c probe. "
                             "Pass explicit path (e.g., data/full_local/tags.json) — no default per CLAUDE.md FAIL-LOUD rule. "
                             "Pair with --no-tags-filter to opt out.")
    parser.add_argument("--no-tags-filter", action="store_true",
                        help="Skip filtering master tags.json into output_dir (rare; m10 will then FATAL). "
                             "If set, --master-tags value is ignored but still required by argparse.")
    add_subset_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not args.subset and not args.FULL:
        parser.print_help()
        print("\nERROR: --subset is required (or use --FULL to download entire corpus)")
        sys.exit(1)

    download_subset(args)


if __name__ == "__main__":
    main()
