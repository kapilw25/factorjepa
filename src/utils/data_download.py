"""
Auto-download WebDataset shards from HF if local data dir is missing.
Parallel TAR iterator for GPU-starved image encoders (CLIP/DINOv2).
Reuses m00d_download_subset.py logic. Called by GPU scripts before data streaming.

Usage:
    from utils.data_download import ensure_local_data, iter_clips_parallel
    ensure_local_data(args)
    for clip_key, mp4_bytes in iter_clips_parallel(local_data, num_readers=8):
        ...
"""
import io
import json
import queue
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _derive_local_dir(subset_path: str) -> Path:
    """Derive local data dir from subset filename: data/subset_10k.json → data/subset_10k_local/"""
    p = Path(subset_path)
    return p.parent / f"{p.stem}_local"


def ensure_local_data(args) -> str:
    """Ensure local WebDataset shards exist. Auto-download from HF if missing.

    Checks args.local_data (or derives from args.subset). If the directory
    exists with a manifest.json, returns the path (no-op). Otherwise,
    invokes m00d_download_subset.py to download from HF CDN.

    Args:
        args: Parsed argparse namespace with .local_data, .subset, .SANITY attributes.

    Returns:
        str: Path to local data directory (guaranteed to exist after call).
    """
    local_data = getattr(args, "local_data", None)

    # Derive local_data path from --subset if not explicitly provided
    if not local_data and getattr(args, "subset", None):
        local_data = str(_derive_local_dir(args.subset))
        args.local_data = local_data

    # For --FULL without --subset, use data/full_local
    if not local_data and not getattr(args, "subset", None):
        local_data = "data/full_local"
        args.local_data = local_data

    # SANITY mode can stream (small clip count) — no download needed
    if getattr(args, "SANITY", False):
        return local_data

    local_path = Path(local_data)
    manifest = local_path / "manifest.json"

    # Already downloaded — no-op
    if local_path.exists() and manifest.exists():
        return local_data

    # Auto-download
    print(f"\n{'='*60}")
    print(f"  AUTO-DOWNLOAD: {local_data} not found")
    print(f"  Downloading WebDataset shards from HuggingFace CDN...")
    print(f"  This is a one-time operation (~50-60 min for 10K, ~60 min for 115K)")
    print(f"{'='*60}\n")

    # Build m00d args
    import argparse
    m00d_args = argparse.Namespace(
        SANITY=False,
        FULL=True,
        subset=getattr(args, "subset", None),
        no_wandb=True,
    )

    # Import and run m00d download
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from m00d_download_subset import download_subset
    download_subset(m00d_args)

    # Verify download succeeded
    if not manifest.exists():
        print(f"FATAL: Download failed — {manifest} not created")
        sys.exit(1)

    print(f"\nDownload complete: {local_data}")
    return local_data


# ═════════════════════════════════════════════════════════════════════════
# PARALLEL TAR READER — feeds GPU-starved image encoders (CLIP/DINOv2)
# ═════════════════════════════════════════════════════════════════════════

TAR_READER_THREADS = 8  # concurrent TAR file readers (I/O-bound, GIL-safe)


def _get_clip_key(json_bytes: bytes) -> str:
    """Extract clip key from JSON sidecar bytes."""
    try:
        meta = json.loads(json_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    return "{}/{}/{}".format(
        meta.get("section", ""), meta.get("video_id", ""), meta.get("source_file", ""))


def _read_one_tar(tar_path: Path, out_q: queue.Queue,
                  subset_keys: set, processed_keys: set,
                  stop_event: threading.Event):
    """Read a single TAR, extract (clip_key, mp4_bytes) pairs into shared queue. Thread-safe.

    iter13 v13 FIX-27 (2026-05-07): logs per-tar emit/drop/filter counts at exit.
    Previously the four `continue` branches silently dropped clips — at FULL v1
    (2026-05-07) 703 of 10,000 manifest-listed clips never reached m10 because
    of `missing_part` (corrupt TAR member) or `empty_mp4` (0-byte body) anomalies,
    and there was no way to tell from the log. Now visible as a one-liner per TAR.
    """
    drops = {"missing_part": 0, "subset_miss": 0, "resume_skip": 0, "empty_mp4": 0}
    n_emitted = 0
    try:
        with tarfile.open(tar_path, "r") as tar:
            entries = {}
            for member in tar.getmembers():
                base = member.name.rsplit(".", 1)[0]
                ext = member.name.rsplit(".", 1)[-1] if "." in member.name else ""
                if base not in entries:
                    entries[base] = {}
                entries[base][ext] = member

            for base, parts in entries.items():
                if stop_event.is_set():
                    return
                # Anomaly: TAR member is missing one of the json/mp4 pair (corrupt
                # or partial m00d save). Counted as DATA DROP (not filter).
                if "json" not in parts or "mp4" not in parts:
                    drops["missing_part"] += 1
                    continue

                json_bytes = tar.extractfile(parts["json"]).read()
                clip_key = _get_clip_key(json_bytes)

                # Filter: caller-requested subset filter (intentional, not an anomaly).
                if subset_keys and clip_key not in subset_keys:
                    drops["subset_miss"] += 1
                    continue
                # Filter: caller-requested resume skip (intentional).
                if clip_key in processed_keys:
                    drops["resume_skip"] += 1
                    continue

                mp4_bytes = tar.extractfile(parts["mp4"]).read()
                # Anomaly: TAR member exists but body is 0-byte (m00d wrote a
                # placeholder for a failed YouTube fetch). DATA DROP.
                if not mp4_bytes:
                    drops["empty_mp4"] += 1
                    continue

                out_q.put((clip_key, mp4_bytes))
                n_emitted += 1
    except (tarfile.TarError, OSError) as e:
        print(f"  ERROR: failed reading {tar_path.name}: {e}")

    n_anomaly = drops["missing_part"] + drops["empty_mp4"]
    n_filtered = drops["subset_miss"] + drops["resume_skip"]
    print(f"  [tar reader] {tar_path.name}: emitted={n_emitted}, "
          f"dropped={n_anomaly} (missing_part={drops['missing_part']}, "
          f"empty_mp4={drops['empty_mp4']}), "
          f"filtered={n_filtered} (subset={drops['subset_miss']}, "
          f"resume={drops['resume_skip']})")


def iter_clips_parallel(local_data: str, subset_keys: set = None,
                        processed_keys: set = None,
                        num_readers: int = TAR_READER_THREADS,
                        max_queue: int = 256) -> "queue.Queue":
    """Start parallel TAR readers, return queue of (clip_key, mp4_bytes) pairs.

    Reads num_readers TARs concurrently (I/O-bound, GIL released during read).
    Returns (queue, stop_event, reader_thread) — caller must set stop_event when done.

    Usage:
        clip_q, stop, reader = iter_clips_parallel("data/full_local", num_readers=8)
        while True:
            item = clip_q.get(timeout=60)
            if item is None: break  # sentinel: all TARs exhausted
            clip_key, mp4_bytes = item
    """
    tar_files = sorted(Path(local_data).glob("*.tar"))
    clip_q = queue.Queue(maxsize=max_queue)
    stop_event = threading.Event()

    def _reader_main():
        """Dispatch TAR files to thread pool, send sentinel when done."""
        with ThreadPoolExecutor(max_workers=num_readers) as pool:
            futures = []
            for tar_path in tar_files:
                if stop_event.is_set():
                    break
                f = pool.submit(_read_one_tar, tar_path, clip_q,
                                subset_keys or set(), processed_keys or set(),
                                stop_event)
                futures.append(f)
            # Wait for all readers to finish
            for f in futures:
                f.result()
        clip_q.put(None)  # sentinel: all TARs exhausted

    reader = threading.Thread(target=_reader_main, daemon=True)
    reader.start()

    print(f"  Parallel TAR reader: {num_readers} threads, {len(tar_files)} TARs, "
          f"queue={max_queue}")

    return clip_q, stop_event, reader
