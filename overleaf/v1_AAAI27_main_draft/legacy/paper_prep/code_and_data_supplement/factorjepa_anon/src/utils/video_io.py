"""Shared video I/O functions — clip key reconstruction, HF streaming, video decoding.

Moved from m05_vjepa_embed.py to eliminate cross-imports between m*.py scripts.
All pipeline scripts (m04d, m05, m05b, m05c, m09, m10, m11) import from here.

USAGE:
    from utils.video_io import get_clip_key, decode_video_bytes, create_stream
"""
import contextlib
import fcntl
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from utils.config import HF_DATASET_REPO

# Video decoder backend — env-selectable (iter18 2026-06-12, MEASURED on the 4×96GB box,
# 64-frame decode of 10s 480p TAR-extracted clips, steady-state):
#   VIDEO_DECODER=av               PyAV CPU — long-standing default          395 ms/clip
#   VIDEO_DECODER=torchcodec       torchcodec CPU — BIT-IDENTICAL to PyAV    338 ms/clip (1.17×)
#   VIDEO_DECODER=torchcodec_cuda  NVDEC — pixels DIFFER (mean 0.66/255, max 2/255) and it
#                                  measured SLOWER here (415 ms/clip: per-clip decoder init
#                                  dominates short 480p files). Only worth it for long/hi-res
#                                  videos, and only applied UNIFORMLY from clip #1 of a run —
#                                  the decoder is part of the pixel source; mixing backends
#                                  across encoders mid-run manufactures a confound.
# The 2026-04-13 "torchcodec segfaults on TAR-extracted mp4s" incident is FIXED in torchcodec
# 0.14 (verified 2026-06-12 on a real subset-*.tar clip, CPU + CUDA, no SIGSEGV).
# NOTE: the eval frame cache below outruns every decoder (55 ms/clip hit) — backend choice
# only matters on cache MISSES.
_DECODER = os.environ.get("VIDEO_DECODER", "av")
VideoDecoder = None
if _DECODER in ("torchcodec", "torchcodec_cuda"):
    try:
        from torchcodec.decoders import VideoDecoder
    except ImportError:
        print(f"FATAL: VIDEO_DECODER={_DECODER} requested but torchcodec is not importable")
        sys.exit(1)
    print(f"  [video_io] decoder backend: {_DECODER} (default av — pixel-parity notes in video_io.py)")
elif _DECODER != "av":
    print(f"FATAL: unknown VIDEO_DECODER={_DECODER!r} — choices: av | torchcodec | torchcodec_cuda")
    sys.exit(1)
try:
    import av
except ImportError:
    if _DECODER == "av":
        print("FATAL: PyAV not available for video decoding")
        print("Install with: pip install av")
        sys.exit(1)
    av = None   # torchcodec backend selected — av genuinely unused


def get_clip_key(example: dict) -> str:
    """Reconstruct clip key from HF WebDataset example metadata.

    Key format: "{section}/{video_id}/{source_file}"
    Example: "goa/walking/U2hl1v8xxlE/U2hl1v8xxlE-092.mp4"
    """
    meta = example.get("json", {})
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta) if meta else {}
    section = meta.get("section", "")
    video_id = meta.get("video_id", "")
    source_file = meta.get("source_file", "")
    return f"{section}/{video_id}/{source_file}"


def create_stream(skip_count: int = 0, local_data: str = None):
    """Create streaming dataset from HF or local WebDataset shards.

    Args:
        skip_count: Number of examples to skip (for resume).
        local_data: Path to local WebDataset dir (from m00d). If None, streams from HF.

    Returns:
        IterableDataset: HuggingFace streaming dataset (undecoded bytes).
    """
    from datasets import load_dataset

    if local_data:
        from utils.data_paths import video_shard_glob  # iter17: subdir-or-root subset-*.tar
        ds = load_dataset("webdataset", data_files=video_shard_glob(local_data),
                          split="train", streaming=True)
    else:
        ds = load_dataset(HF_DATASET_REPO, split="train", streaming=True)
    ds = ds.decode(False)
    if skip_count > 0:
        ds = ds.skip(skip_count)
    return ds


def _load_torchcodec(video_path: str, num_frames: int) -> torch.Tensor:
    """Load video via torchcodec — CPU (bit-identical to PyAV, measured 06-12) or NVDEC when
    VIDEO_DECODER=torchcodec_cuda (different pixels — see the backend block above)."""
    decoder = VideoDecoder(video_path, device="cuda" if _DECODER == "torchcodec_cuda" else "cpu")
    total_frames = decoder.metadata.num_frames if hasattr(decoder.metadata, "num_frames") else 200
    frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    frames = decoder.get_frames_at(indices=frame_indices.tolist())
    video_tensor = frames.data  # (T, C, H, W)
    if video_tensor.is_cuda:
        video_tensor = video_tensor.cpu()   # downstream contract: CPU uint8 tensor
    del decoder
    if video_tensor.shape[0] < num_frames:
        pad_size = num_frames - video_tensor.shape[0]
        last_frame = video_tensor[-1:].repeat(pad_size, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, last_frame], dim=0)
    return video_tensor[:num_frames]


def _load_av(video_path: str, num_frames: int) -> torch.Tensor:
    """Load video using PyAV — the DEFAULT backend (not a fallback; backend choice is
    explicit via VIDEO_DECODER and never silently switches)."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.codec_context.thread_count = 1
    total_frames = stream.frames
    if total_frames == 0:
        if stream.duration and stream.average_rate:
            total_frames = int(float(stream.duration * stream.time_base) * float(stream.average_rate))
        else:
            total_frames = 200
    indices = set(np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int).tolist())
    frames = []
    frame_idx = 0
    for frame in container.decode(video=0):
        if frame_idx in indices:
            img = frame.to_ndarray(format="rgb24")
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            frames.append(img_tensor)
        frame_idx += 1
        if len(frames) >= num_frames:
            break
    container.close()
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1].clone())
        else:
            frames.append(torch.zeros((3, 256, 256), dtype=torch.uint8))
    return torch.stack(frames[:num_frames])


# ── Eval frame cache (iter17) ─────────────────────────────────────────────
# Every eval module (m12a action, m12d future, m12e predictor-temporal, m12f
# encoder-temporal) reaches frames through decode_to_tensor → THIS decode_video_bytes,
# which is also the only decode in frozen_features. The scheduler runs one run_eval.sh
# PER encoder, so 8 processes decode the SAME ~1.8k clips concurrently (~7 passes each),
# pinning the CPU (load 467/256) while the GPUs idle. When EVAL_FRAME_CACHE_DIR is set
# (run_eval.sh + the build_probe_clips decode SUBPROCESSES since iter18 — never the
# training stream/producer, whose full-pool decode over many epochs would blow
# the cache), the decoded native (T,C,H,W) uint8 tensor is memoized to disk keyed by
# (clip_key, num_frames). Decode is deterministic (linspace sampling + fixed decoder), so
# a hit returns a byte-identical tensor; decode_to_tensor still runs its crop/normalize
# AFTER, so metrics are unchanged. Unset env → every line below is skipped → zero effect
# on training / data-prep / the in-flight run (which uses the old run_eval.sh).
_FRAME_CACHE_ENV = "EVAL_FRAME_CACHE_DIR"
_FRAME_CACHE_MIN_FREE_ENV = "EVAL_FRAME_CACHE_MIN_FREE_GB"


def _frame_cache_path(key: str, num_frames: int):
    """On-disk cache path for (clip_key, num_frames), or None when the cache is disabled."""
    cache_dir = os.environ.get(_FRAME_CACHE_ENV)
    if not cache_dir:
        return None
    safe = key.replace("/", "_").replace("\\", "_")
    if safe.endswith(".mp4"):
        safe = safe[:-4]
    return os.path.join(cache_dir, f"{safe}__nf{int(num_frames)}.npy")


def _frame_cache_load(cache_path: str):
    """Cached (T,C,H,W) uint8 tensor, or None on miss / unreadable. Unreadable = a
    half-written file from a peer still mid-decode → treat as a miss and decode fresh
    (correctness-preserving: a bad cache file never contaminates the returned frames)."""
    if not os.path.exists(cache_path):
        return None
    try:
        return torch.from_numpy(np.load(cache_path))
    except Exception:
        return None  # partial/corrupt cache file → fall through to a fresh decode


# ── Bounded LRU cap (iter19 2026-07-06) ────────────────────────────────────────
# The old min_free_gb "free-disk floor" RESERVED NOTHING — it only stopped the cache from ADDING
# below the floor, so checkpoint/eval writes ate straight through the reserve and the FULL run
# crashed on ENOSPC (the cache had ballooned to 512 G). The permanent fix: cap the cache's OWN
# footprint. EVAL_FRAME_CACHE_MAX_GB is a HARD ceiling on the cache dir's total .npy bytes; a store
# that would exceed it LRU-evicts oldest entries first (least-recently-STORED — robust under
# noatime). The cache therefore can NEVER fill the disk — the rest is guaranteed-free for tars +
# checkpoints + eval. Eviction is serialized across the concurrent writers (diheavy ∥ peft ∥ evals)
# by an flock, and batch-evicts to a low-water mark so the O(N) scan is amortized; a running
# .cache_bytes counter keeps the common path O(1) and each eviction recomputes truth → self-heals.
_FRAME_CACHE_MAX_ENV = "EVAL_FRAME_CACHE_MAX_GB"
_EVICT_LOW_WATER = 0.85   # on overflow, evict down to this fraction of the cap (amortizes the scan)


def _cache_npy_bytes(cache_dir: str) -> int:
    """True total bytes of the cached .npy files (scandir — the source of truth)."""
    total = 0
    with contextlib.suppress(OSError):
        with os.scandir(cache_dir) as it:
            for e in it:
                if e.name.endswith(".npy"):
                    with contextlib.suppress(OSError):
                        total += e.stat().st_size
    return total


def _evict_to(cache_dir: str, target_bytes: float) -> int:
    """Delete oldest-STORED .npy files (by mtime) until the dir is <= target_bytes; return the true
    post-eviction total. Caller holds the flock so concurrent writers never double-evict."""
    entries = []
    with contextlib.suppress(OSError):
        with os.scandir(cache_dir) as it:
            for e in it:
                if not e.name.endswith(".npy"):
                    continue
                with contextlib.suppress(OSError):
                    st = e.stat()
                    entries.append((st.st_mtime, st.st_size, e.path))
    total = sum(sz for _, sz, _ in entries)
    entries.sort(key=lambda t: t[0])   # oldest store-time first = least-recently-stored
    for _mt, sz, path in entries:
        if total <= target_bytes:
            break
        with contextlib.suppress(OSError):
            os.unlink(path)
            total -= sz
    return total


def _frame_cache_store(cache_path: str, frames) -> None:
    """Atomically memoize the decoded tensor under a HARD size cap (EVAL_FRAME_CACHE_MAX_GB) with
    LRU eviction; falls back to the legacy free-disk floor (EVAL_FRAME_CACHE_MIN_FREE_GB) only when
    no cap is set. The cache is a pure speed optimization, so a write/evict failure WARNs and
    continues — the decode already returned the correct frames. Atomic = unique temp + os.replace."""
    try:
        cache_dir = os.path.dirname(cache_path)
        os.makedirs(cache_dir, exist_ok=True)
        arr = frames.numpy() if hasattr(frames, "numpy") else np.asarray(frames)
        new_bytes = int(arr.nbytes)

        max_gb = os.environ.get(_FRAME_CACHE_MAX_ENV)
        if max_gb is not None:
            # HARD CAP + LRU — bound the cache's own footprint so it can NEVER fill the disk.
            max_bytes = float(max_gb) * (1024 ** 3)
            if new_bytes > max_bytes:
                return  # a single clip bigger than the whole cap (never in practice) → skip
            counter = os.path.join(cache_dir, ".cache_bytes")
            with open(os.path.join(cache_dir, ".cache.lock"), "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    try:
                        with open(counter) as cf:
                            total = int(cf.read().strip())
                    except (OSError, ValueError):
                        total = _cache_npy_bytes(cache_dir)   # missing/corrupt → recompute truth
                    if total + new_bytes > max_bytes:
                        total = _evict_to(cache_dir, _EVICT_LOW_WATER * max_bytes - new_bytes)
                    with contextlib.suppress(OSError):
                        with open(counter, "w") as cf:
                            cf.write(str(total + new_bytes))
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
        else:
            # Legacy fallback: free-disk floor (kept for callers that don't set the cap).
            min_free_gb = float(os.environ.get(_FRAME_CACHE_MIN_FREE_ENV, "20"))
            if shutil.disk_usage(cache_dir).free < min_free_gb * (1024 ** 3):
                return  # disk getting tight → stop adding entries (existing ones still serve hits)

        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".npy.tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, cache_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    except Exception as e:
        print(f"  WARN: frame-cache store failed ({cache_path}): {e}")


def decode_video_bytes(mp4_bytes: bytes, tmp_dir: str, key: str,
                       num_frames: int, cache_frames: bool = True) -> torch.Tensor:
    """Write mp4 bytes to temp file, decode frames, delete temp file.

    Args:
        mp4_bytes: Raw MP4 bytes from WebDataset TAR.
        tmp_dir: Temp directory for writing intermediate files.
        key: Clip key (used for safe filename).
        num_frames: Number of frames to extract (uniformly sampled).

    Returns:
        torch.Tensor: (T, C, H, W) uint8 tensor, or None if decode fails.

    When EVAL_FRAME_CACHE_DIR is set, the (key, num_frames) decode is memoized to disk
    (see the eval-frame-cache note above) so concurrent per-encoder evals decode each clip
    ONCE instead of re-decoding it for every encoder/metric.
    """
    cache_path = _frame_cache_path(key, num_frames)
    if cache_path is not None:
        cached = _frame_cache_load(cache_path)
        if cached is not None:
            return cached
    safe_key = key.replace("/", "_").replace("\\", "_")
    # HF stream clip keys end in ".mp4" (e.g. "tier1/.../abc-119.mp4"); strip
    # before re-appending so we don't write/read "abc-119.mp4.mp4". The doubled
    # extension is consistent across write+read in the same call (path is reused),
    # so it's not the ENOENT cause — but the doubled name made the explora v10
    # decode-failure logs misleading. Idempotent: keys without .mp4 unaffected.
    if safe_key.endswith(".mp4"):
        safe_key = safe_key[:-4]
    tmp_path = os.path.join(tmp_dir, f"{safe_key}.mp4")
    result = None
    try:
        # iter17 fix: the streaming __iter__ holds ONE TemporaryDirectory across all its yields;
        # under the DataLoader's multi-worker/prefetch lifecycle that dir can be torn down mid-decode
        # → ENOENT on the write below → return None → stream_factor FATALs the whole surgery training.
        # Recreate it (idempotent, cheap) so a transient temp-dir race is survived, not fatal.
        os.makedirs(tmp_dir, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(mp4_bytes)
        # hard dispatch on VIDEO_DECODER — no silent fallback between backends (the decoder
        # is part of the pixel source; a quiet swap mid-run would contaminate cross-encoder
        # comparability). Unknown/unavailable backends already FATALed at import.
        if _DECODER != "av":
            result = _load_torchcodec(tmp_path, num_frames)
        else:
            result = _load_av(tmp_path, num_frames)
    except Exception as e:
        print(f"  WARN: decode failed ({key}): {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                print(f"  WARN: failed to delete temp file {tmp_path}")
    if cache_path is not None and result is not None and cache_frames:
        _frame_cache_store(cache_path, result)   # cache_frames=False → read-once probe decodes skip the store
    return result
