"""
Pre-download subset clips from HF to local WebDataset TAR shards.
Eliminates producer starvation bug (8.4% hit rate → 100% hit rate).
CPU-only — no GPU needed.

USAGE:
    python -u src/m00d_download_subset.py --subset data/subset_10k.json 2>&1 | tee logs/m00d_download.log
    python -u src/m00d_download_subset.py --subset data/subset_10k.json --SANITY 2>&1 | tee logs/m00d_sanity.log
"""
import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import HF_DATASET_REPO, load_subset, add_subset_arg, PROJECT_ROOT
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def get_clip_key(example: dict) -> str:
    """Reconstruct clip key from HF WebDataset example metadata.
    Same logic as m05_vjepa_embed.get_clip_key (duplicated to avoid torch import chain)."""
    meta = example.get("json", {})
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta) if meta else {}
    section = meta.get("section", "")
    video_id = meta.get("video_id", "")
    source_file = meta.get("source_file", "")
    return f"{section}/{video_id}/{source_file}"

CLIPS_PER_SHARD = 1000
SANITY_CLIP_LIMIT = 20


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


def download_subset(args):
    """Stream HF dataset, save matching subset clips to local WebDataset TARs."""
    if not HAS_DATASETS:
        print("ERROR: datasets not installed. Run: pip install datasets")
        sys.exit(1)

    if not args.subset:
        print("ERROR: --subset is required (e.g., --subset data/subset_10k.json)")
        sys.exit(1)

    # Load subset keys
    subset_keys = load_subset(args.subset)
    if not subset_keys:
        print("ERROR: subset is empty")
        sys.exit(1)

    clip_limit = SANITY_CLIP_LIMIT if args.SANITY else len(subset_keys)
    target_keys = subset_keys  # full set for matching

    # Output directory
    output_dir = _output_dir_from_subset(args.subset)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    print(f"=== m00d: Pre-download subset to local WebDataset ===")
    print(f"  Subset: {args.subset} ({len(subset_keys):,} keys)")
    print(f"  Output: {output_dir}")
    print(f"  Clip limit: {clip_limit:,}")
    print(f"  Clips per shard: {CLIPS_PER_SHARD}")

    # Resume: load manifest to find already-saved keys
    manifest = _load_manifest(manifest_path)
    saved_keys = set(manifest.get("saved_keys", []))
    if saved_keys:
        print(f"  [resume] {len(saved_keys):,} clips already saved, skipping")

    # Init wandb
    wb_run = init_wandb("m00d", "download_subset",
                        {"subset": args.subset, "clip_limit": clip_limit},
                        enabled=not args.no_wandb)

    # Load HF_TOKEN from .env for private dataset access
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Stream HF dataset
    print(f"\nStreaming from: {HF_DATASET_REPO}")
    ds = load_dataset(HF_DATASET_REPO, split="train", streaming=True)
    ds = ds.decode(False)

    # Collect matching clips into shard-sized batches
    from tqdm import tqdm

    found = 0
    scanned = 0
    shard_idx = manifest.get("next_shard_idx", 0)
    shard_buffer = []  # list of (key, mp4_bytes, json_bytes)
    shards_written = list(manifest.get("shards", []))
    all_saved_keys = list(saved_keys)
    start_time = time.time()

    pbar = tqdm(desc="Scanning HF dataset", unit=" clips",
                postfix={"found": 0, "hit_rate": "0%"})

    try:
        for example in ds:
            scanned += 1
            pbar.update(1)

            clip_key = get_clip_key(example)

            # Skip non-subset clips
            if clip_key not in target_keys:
                continue

            # Skip already-saved clips (resume)
            if clip_key in saved_keys:
                found += 1
                pbar.set_postfix({"found": found, "hit_rate": f"{found/scanned*100:.1f}%"})
                continue

            # Extract bytes
            mp4_data = example.get("mp4", b"")
            mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
            if not mp4_bytes:
                print(f"  WARN: empty mp4 for {clip_key}, skipping")
                continue

            json_data = example.get("json", b"")
            if isinstance(json_data, dict):
                json_bytes = json.dumps(json_data, ensure_ascii=False).encode("utf-8")
            elif isinstance(json_data, bytes):
                json_bytes = json_data
            elif isinstance(json_data, str):
                json_bytes = json_data.encode("utf-8")
            else:
                json_bytes = b"{}"

            hf_key = example.get("__key__", f"{found:06d}")
            shard_buffer.append((hf_key, mp4_bytes, json_bytes))
            found += 1
            all_saved_keys.append(clip_key)

            pbar.set_postfix({"found": found, "hit_rate": f"{found/scanned*100:.1f}%"})

            # Write shard when buffer is full
            if len(shard_buffer) >= CLIPS_PER_SHARD:
                shard_path = _write_shard(output_dir, shard_idx, shard_buffer)
                shards_written.append(shard_path.name)
                shard_idx += 1
                shard_buffer = []

                # Update manifest (checkpoint)
                manifest = {
                    "n": len(all_saved_keys),
                    "shards": shards_written,
                    "subset_file": str(args.subset),
                    "next_shard_idx": shard_idx,
                    "saved_keys": all_saved_keys,
                }
                _save_manifest(manifest_path, manifest)
                print(f"  [checkpoint] shard {shard_idx-1}: {len(all_saved_keys):,} clips saved")

                if wb_run:
                    log_metrics(wb_run, {"clips_saved": len(all_saved_keys),
                                         "scanned": scanned}, step=scanned)

            # Check clip limit
            if found >= clip_limit:
                print(f"\n  Reached clip limit ({clip_limit:,})")
                break

    except KeyboardInterrupt:
        print("\n  Interrupted! Saving partial progress...")

    pbar.close()

    # Write final partial shard
    if shard_buffer:
        shard_path = _write_shard(output_dir, shard_idx, shard_buffer)
        shards_written.append(shard_path.name)
        shard_idx += 1

    # Final manifest
    manifest = {
        "n": len(all_saved_keys),
        "shards": shards_written,
        "subset_file": str(args.subset),
        "next_shard_idx": shard_idx,
        "saved_keys": all_saved_keys,
    }
    _save_manifest(manifest_path, manifest)

    elapsed = time.time() - start_time
    print(f"\n=== Download complete ===")
    print(f"  Clips saved: {len(all_saved_keys):,}/{clip_limit:,}")
    print(f"  Shards: {len(shards_written)}")
    print(f"  Scanned: {scanned:,} HF examples")
    print(f"  Hit rate: {len(all_saved_keys)/max(scanned,1)*100:.1f}%")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Output: {output_dir}")

    # Disk usage
    total_bytes = sum((output_dir / s).stat().st_size for s in shards_written
                      if (output_dir / s).exists())
    print(f"  Disk: {total_bytes / 1e9:.2f} GB")

    if wb_run:
        log_metrics(wb_run, {"total_clips": len(all_saved_keys),
                              "total_shards": len(shards_written),
                              "elapsed_min": elapsed / 60,
                              "disk_gb": total_bytes / 1e9})
    finish_wandb(wb_run)


def _write_shard(output_dir: Path, shard_idx: int,
                 buffer: list) -> Path:
    """Write a TAR shard from buffer of (key, mp4_bytes, json_bytes) tuples."""
    shard_name = f"subset-{shard_idx:05d}.tar"
    shard_path = output_dir / shard_name

    with tarfile.open(shard_path, "w") as tar:
        for hf_key, mp4_bytes, json_bytes in buffer:
            # mp4 entry
            mp4_info = tarfile.TarInfo(name=f"{hf_key}.mp4")
            mp4_info.size = len(mp4_bytes)
            tar.addfile(mp4_info, io.BytesIO(mp4_bytes))

            # json sidecar
            json_info = tarfile.TarInfo(name=f"{hf_key}.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, io.BytesIO(json_bytes))

    return shard_path


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
