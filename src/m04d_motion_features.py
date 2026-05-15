"""
GPU-RAFT optical flow motion features per clip (23-D vector, post-Phase-0).
Gold standard: https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py
Claude Code: re-WebSearch this URL on every read of this file.
Extracts deterministic temporal ground-truth: mean/std/max magnitude,
8-bin direction histogram, camera motion (dx, dy), AND foreground (camera-
subtracted) agent-motion stats (Phase 0, iter15, 2026-05-14): fg_mean_mag,
fg_max_mag, fg_dir_hist (8 bins) → total 23 dims, see FEATURE_NAMES below.

Producer-consumer pipeline: CPU thread decodes video + preprocesses frame
pairs → queue → GPU thread batches multiple clips' pairs into one RAFT
forward pass. ~5-10x faster than sequential.

OUTPUT ROUTING (iter15, 2026-05-15):
    ALL m04d outputs land in <local_data>/m04d_motion_features/ — durable
    motion_features.{npy,paths.npy,meta.json} AND mid-run .m04d_checkpoint.npz.
    The full subdir rides on `hf_outputs.py upload-data` → fresh GPU instances
    pick up a preempted m04d run via `download-data` and resume from
    checkpoint. Mirrors the m10_sam_segment/ + m11_factor_datasets/ subdir
    convention under <local_data>/. Override with --output-dir.

USAGE (per-dataset; one run per local_data dir, durable artifact thereafter):
    # eval_10k (FULL eval target) — outputs land in
    # data/eval_10k_local/m04d_motion_features/ by default:
    CACHE_POLICY_ALL=2 python -u src/m04d_motion_features.py --FULL \
        --subset data/eval_10k_local/eval_10k.json --local-data data/eval_10k_local \
        --no-wandb 2>&1 | tee logs/m04d_full_eval10k_$(date +%Y%m%d_%H%M%S).log

    # subset_10k (POC):
    python -u src/m04d_motion_features.py --POC --subset data/subset_10k_local/subset_10k.json \
        --local-data data/subset_10k_local \
        2>&1 | tee logs/m04d_poc_$(date +%Y%m%d_%H%M%S).log

    # val_1k:
    python -u src/m04d_motion_features.py --FULL --subset data/val_1k_local/val_1k.json \
        --local-data data/val_1k_local \
        2>&1 | tee logs/m04d_val1k_$(date +%Y%m%d_%H%M%S).log

    # full_local (training data; HF will only upload the small .npy/.paths.npy,
    # NOT the multi-TB TAR shards — see hf_outputs.py:upload_data()):
    python -u src/m04d_motion_features.py --FULL --local-data data/full_local \
        2>&1 | tee logs/m04d_full_$(date +%Y%m%d_%H%M%S).log

    # SANITY (smoke test):
    python -u src/m04d_motion_features.py --SANITY --subset data/subset_10k_local/subset_10k.json \
        --local-data data/subset_10k_local \
        2>&1 | tee logs/m04d_sanity_$(date +%Y%m%d_%H%M%S).log
"""
import argparse
import gc
import io
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# iter13 v12 fix (2026-05-05): pin torch._inductor's kernel-compile pool to a
# single thread. The default (`= num_cpu`) parallel compile was racing with
# our producer's ThreadPoolExecutor — torch.compile's pool shutdown set the
# global `concurrent.futures._shutdown=True` flag prematurely, so the next
# producer batch raised `RuntimeError: cannot schedule new futures after
# interpreter shutdown` mid-run (logs/m04d_full_eval10k_$(date +%Y%m%d_%H%M%S).log:19). MUST be set
# BEFORE any torch import. Costs ~30s extra one-time on warm-up, free
# afterwards. Belt-and-suspenders alongside the persistent-pool fix below.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# iter15 Phase 3 fix (2026-05-14): cap OpenMP / BLAS thread teams to 1 worker
# per process. PyAV's sws_scale color conversion (yuv420p → bgr24) and ffmpeg's
# internal libs use libgomp, which defaults to NUM_THREADS = nproc. On
# many-core / containerized hosts (e.g., 96-core box with cgroup pids.max=1024),
# 16 decode workers × 96 OMP threads = 1536 transient threads → instant
# "libgomp: Thread creation failed: Resource temporarily unavailable" crash on
# the first frame decode. Caps below force sequential color conversion (the
# work is already pipelined behind GPU RAFT inference, so wall-time is unchanged).
# Use setdefault so an explicit OMP_NUM_THREADS=4 env still wins for debugging.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import av
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    HF_DATASET_REPO, check_gpu,
    add_subset_arg, add_local_data_arg, load_subset,
    get_pipeline_config, get_sanity_clip_limit, get_total_clips,
)
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)
from utils.gpu_batch import add_gpu_mem_arg, AdaptiveBatchSizer, cleanup_temp
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog

# Lazy imports — torch + torchvision loaded after check_gpu()
torch = None
raft_large = None
Raft_Large_Weights = None

FEATURE_DIM = 23
FEATURE_NAMES = [
    # Existing 13-D global flow (indices 0-12)
    "mean_magnitude", "magnitude_std", "max_magnitude",
    "dir_hist_0", "dir_hist_1", "dir_hist_2", "dir_hist_3",
    "dir_hist_4", "dir_hist_5", "dir_hist_6", "dir_hist_7",
    "camera_motion_x", "camera_motion_y",
    # Phase 0 / iter15 — foreground / camera-subtracted flow (indices 13-22)
    # Computed as flow MINUS per-pair camera motion → agent-only motion.
    # Powers vec[13]-based action-class binning + data_curriculum difficulty.
    "fg_mean_mag", "fg_max_mag",
    "fg_dir_hist_0", "fg_dir_hist_1", "fg_dir_hist_2", "fg_dir_hist_3",
    "fg_dir_hist_4", "fg_dir_hist_5", "fg_dir_hist_6", "fg_dir_hist_7",
]
N_FRAME_PAIRS = 16
_pcfg = get_pipeline_config()
CHECKPOINT_INTERVAL = _pcfg["eval"]["motion_checkpoint_every"]
PRODUCER_QUEUE_SIZE = _pcfg["streaming"]["producer_queue_motion"]
CLIPS_PER_GPU_BATCH = _pcfg["gpu"]["motion_batch_size"]
DECODE_WORKERS = _pcfg["streaming"]["decode_workers_motion"]


# ── HF Streaming (reuse m05 pattern) ────────────────────────────────

def _create_stream(skip_count: int = 0, local_data: str = None):
    """Create streaming dataset from HF or local WebDataset shards."""
    from datasets import load_dataset
    if local_data:
        ds = load_dataset("webdataset", data_files=f"{local_data}/*.tar",
                          split="train", streaming=True)
    else:
        ds = load_dataset(HF_DATASET_REPO, split="train", streaming=True)
    ds = ds.decode(False)
    if skip_count > 0:
        ds = ds.skip(skip_count)
    return ds


def get_clip_key(example: dict) -> str:
    """Reconstruct clip key from HF example metadata (matches m05 pattern)."""
    meta = example.get("json", {})
    if isinstance(meta, bytes):
        meta = json.loads(meta)
    section = meta.get("section", "")
    video_id = meta.get("video_id", "")
    source_file = meta.get("source_file", "")
    return f"{section}/{video_id}/{source_file}"


# ── Video Decode (PyAV from bytes — no temp file, 5x faster than cv2 seek) ──

def decode_video_frames(video_bytes: bytes, n_pairs: int = N_FRAME_PAIRS):
    """Decode video bytes to evenly-spaced consecutive frame pairs via PyAV.

    Single sequential pass — avoids cv2's slow H.264 keyframe seeking.
    Returns list of (prev_frame, curr_frame) as uint8 BGR numpy arrays (H, W, 3).
    """
    container = av.open(io.BytesIO(video_bytes))
    stream = container.streams.video[0]

    # Get total frame count (stream.frames may be 0 for some containers)
    total_frames = stream.frames
    if total_frames == 0 and stream.duration and stream.time_base:
        total_frames = int(float(stream.duration * stream.time_base)
                          * float(stream.average_rate or 30))
    if total_frames < 2:
        container.close()
        return []

    # Compute which frame indices we need
    stride = max(1, (total_frames - 1) // n_pairs)
    pair_starts = []
    needed = set()
    for i in range(n_pairs):
        s = i * stride
        if s + 1 < total_frames:
            pair_starts.append(s)
            needed.add(s)
            needed.add(s + 1)
    max_needed = max(needed) if needed else 0

    # Single sequential decode pass — collect only needed frames
    frames = {}
    for idx, frame in enumerate(container.decode(video=0)):
        if idx in needed:
            frames[idx] = frame.to_ndarray(format="bgr24")
        if idx >= max_needed:
            break
    container.close()

    pairs = []
    for s in pair_starts:
        prev = frames.get(s)
        curr = frames.get(s + 1)
        if prev is not None and curr is not None:
            pairs.append((prev, curr))
    return pairs


# ── RAFT Optical Flow ────────────────────────────────────────────────

def load_raft_model(device):
    """Load pretrained RAFT-Large from torchvision."""
    global torch, raft_large, Raft_Large_Weights
    import torch as _torch
    torch = _torch
    from torchvision.models.optical_flow import raft_large as _raft_large
    from torchvision.models.optical_flow import Raft_Large_Weights as _weights
    raft_large = _raft_large
    Raft_Large_Weights = _weights

    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights).to(device).eval()
    # iter13 v12 fix (2026-05-05): torch.compile DISABLED for RAFT.
    # Reason: AdaptiveBatchSizer preemptively shrinks the batch (e.g. 512 → 511)
    # when post-batch VRAM > cap, which triggers torch._inductor recompilation
    # under dynamic shapes. Recompile hits a known PyTorch bug:
    #   InductorError: CantSplit: 1123200*s11+1123200*s15 not divisible by 96*s11+96*s15
    # (see logs/m04d_full_eval10k_v1_$(date +%Y%m%d_%H%M%S).log:96). torch.compile is also what pushes
    # VRAM to the cap in the first place (compile workspace ~70 GB on Blackwell),
    # so disabling it ALSO eliminates the trigger for AdaptiveBatch shrink.
    # Cost: ~1.5-2× slower RAFT inference. Acceptable: m04d is one-time-per-
    # dataset (durable artifact, runs once on each <local_data>/).
    transforms = weights.transforms()
    print(f"RAFT-Large loaded on {device} (weights: C_T_SKHT_V2, eager, fp16)")
    return model, transforms


def _preprocess_pairs(frame_pairs, transforms):
    """Preprocess frame pairs into batched tensors for RAFT.

    Returns (prev_batch, curr_batch) each of shape (N, 3, 360, 520).
    """
    h, w = 360, 520
    assert h % 8 == 0 and w % 8 == 0, f"RAFT requires h,w divisible by 8, got {h}x{w}"
    prev_list, curr_list = [], []
    for prev_frame, curr_frame in frame_pairs:
        prev_resized = cv2.resize(prev_frame, (w, h))
        curr_resized = cv2.resize(curr_frame, (w, h))
        # BGR→RGB, HWC→CHW, uint8→float32
        prev_t = torch.from_numpy(prev_resized[:, :, ::-1].copy()).permute(2, 0, 1).float()
        curr_t = torch.from_numpy(curr_resized[:, :, ::-1].copy()).permute(2, 0, 1).float()
        prev_t, curr_t = transforms(prev_t, curr_t)
        prev_list.append(prev_t)
        curr_list.append(curr_t)
    return torch.stack(prev_list), torch.stack(curr_list)


def _aggregate_flow(flow_np, n_pairs):
    """Aggregate RAFT flow output for one clip into 23-D feature vector.

    Args:
        flow_np: numpy array of shape (n_pairs, 2, H, W)
        n_pairs: number of frame pairs for this clip
    Returns:
        np.ndarray of shape (23,) float32 — first 13 dims are global flow stats
        (unchanged from pre-Phase-0); last 10 dims are foreground (camera-subtracted)
        agent-motion stats added in iter15 Phase 0.
    """
    dx_all = flow_np[:, 0]  # (N, H, W)
    dy_all = flow_np[:, 1]  # (N, H, W)
    mag_all = np.sqrt(dx_all**2 + dy_all**2)
    ang_all = np.arctan2(dy_all, dx_all)

    flat_mag = mag_all.flatten()
    mean_mag = float(np.mean(flat_mag))
    std_mag = float(np.std(flat_mag))
    max_mag = float(np.max(flat_mag))

    # 8-bin direction histogram (normalized) — global flow
    flat_ang = ang_all.flatten()
    hist, _ = np.histogram(flat_ang, bins=8, range=(-np.pi, np.pi))
    hist = hist.astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum

    # Camera motion: median flow per pair, then median across pairs
    per_pair_dx = np.median(dx_all.reshape(n_pairs, -1), axis=1)
    per_pair_dy = np.median(dy_all.reshape(n_pairs, -1), axis=1)
    cam_x = float(np.median(per_pair_dx))
    cam_y = float(np.median(per_pair_dy))

    # Phase 0 / iter15: foreground motion = flow MINUS per-pair camera motion.
    # Removes camera-induced global translation → captures agent/object motion only.
    cam_dx_per_pair = per_pair_dx.reshape(n_pairs, 1, 1)            # (N, 1, 1) broadcast
    cam_dy_per_pair = per_pair_dy.reshape(n_pairs, 1, 1)
    fg_dx = dx_all - cam_dx_per_pair                                # (N, H, W)
    fg_dy = dy_all - cam_dy_per_pair
    fg_mag = np.sqrt(fg_dx**2 + fg_dy**2)
    fg_ang = np.arctan2(fg_dy, fg_dx)

    fg_mean_mag = float(fg_mag.mean())
    fg_max_mag = float(fg_mag.max())
    fg_hist, _ = np.histogram(fg_ang.flatten(), bins=8, range=(-np.pi, np.pi))
    fg_hist = fg_hist.astype(np.float32)
    fg_hist_sum = fg_hist.sum()
    if fg_hist_sum > 0:
        fg_hist = fg_hist / fg_hist_sum

    return np.array([
        mean_mag, std_mag, max_mag, *hist, cam_x, cam_y,            # existing 13 dims
        fg_mean_mag, fg_max_mag, *fg_hist,                          # Phase 0 — 10 new dims
    ], dtype=np.float32)


# ── Producer Thread ──────────────────────────────────────────────────

def _producer_thread(q: queue.Queue, stop_event: threading.Event,
                     transforms, n_pairs: int, clip_limit: int,
                     subset_keys: set, processed_keys: set,
                     local_data: str = None):
    """Stream clips, parallel-decode video via PyAV, preprocess, enqueue for GPU.

    Each queue item is ("batch", prev_tensors, curr_tensors, keys, n_pairs_per_clip)
    where prev/curr are (total_pairs, 3, H, W) tensors ready for GPU .to(device).
    Uses DECODE_WORKERS threads for parallel PyAV decode (releases GIL).
    """
    produced = 0
    skipped = 0
    errors = 0

    def _decode_one(mp4_bytes, key, n_p):
        """Decode + preprocess one clip. Called from thread pool."""
        pairs = decode_video_frames(mp4_bytes, n_pairs=n_p)
        if not pairs:
            return None, key
        prev_b, curr_b = _preprocess_pairs(pairs, transforms)
        return (prev_b, curr_b, prev_b.shape[0]), key

    tar_stop = None

    # iter13 v12 fix (2026-05-05): single persistent ThreadPoolExecutor for
    # the producer's entire lifetime. The previous per-batch pool creation
    # (`with ThreadPoolExecutor(...) as pool:` inside _flush_batch) raced with
    # torch._inductor's internal compile pool — at the boundary between
    # batches, concurrent.futures' global shutdown state was sometimes
    # already True, raising RuntimeError "cannot schedule new futures after
    # interpreter shutdown". Persistent pool sidesteps that race AND avoids
    # spawning 8 workers per batch (~32 batches/min × 8 spawns = 256/min).
    pool = ThreadPoolExecutor(max_workers=DECODE_WORKERS,
                              thread_name_prefix="m04d-decode")

    def _flush_batch(batch):
        """Parallel-decode a batch and enqueue for GPU."""
        nonlocal produced, errors
        pending_prevs, pending_currs = [], []
        pending_keys, pending_n_pairs = [], []
        futures = [pool.submit(_decode_one, b, k, n_pairs) for b, k in batch]
        for fut in futures:
            result, key = fut.result()
            if result is None:
                errors += 1
            else:
                prev_b, curr_b, n_p = result
                pending_prevs.append(prev_b)
                pending_currs.append(curr_b)
                pending_keys.append(key)
                pending_n_pairs.append(n_p)
        if pending_keys:
            cat_prev = torch.cat(pending_prevs, dim=0)
            cat_curr = torch.cat(pending_currs, dim=0)
            q.put(("batch", cat_prev, cat_curr,
                    pending_keys[:], pending_n_pairs[:]))
            produced += len(pending_keys)

    try:
        batch_bytes = []  # (mp4_bytes, clip_key)

        if local_data:
            # Fast path: parallel TAR readers (8 threads, skip processed keys at TAR level)
            clip_q, tar_stop, _reader = iter_clips_parallel(
                local_data, subset_keys=subset_keys, processed_keys=processed_keys)
            while not stop_event.is_set() and produced < clip_limit:
                item = clip_q.get(timeout=120)
                if item is None:
                    break
                clip_key, mp4_bytes = item
                if not mp4_bytes or len(mp4_bytes) < 1000:
                    errors += 1
                    continue
                batch_bytes.append((mp4_bytes, clip_key))
                if len(batch_bytes) >= CLIPS_PER_GPU_BATCH:
                    _flush_batch(batch_bytes)
                    batch_bytes = []
        else:
            # Fallback: sequential HF streaming
            ds = _create_stream(local_data=local_data)
            for example in ds:
                if stop_event.is_set() or produced >= clip_limit:
                    break
                clip_key = get_clip_key(example)
                if subset_keys and clip_key not in subset_keys:
                    skipped += 1
                    continue
                if clip_key in processed_keys:
                    skipped += 1
                    continue
                mp4_data = example.get("mp4", b"")
                mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
                if isinstance(mp4_bytes, str):
                    mp4_bytes = mp4_bytes.encode()
                if not mp4_bytes or len(mp4_bytes) < 1000:
                    errors += 1
                    continue
                batch_bytes.append((mp4_bytes, clip_key))
                if len(batch_bytes) >= CLIPS_PER_GPU_BATCH:
                    _flush_batch(batch_bytes)
                    batch_bytes = []

        # Flush remaining partial batch
        if batch_bytes and not stop_event.is_set():
            _flush_batch(batch_bytes)

    except Exception as e:
        print(f"\n  PRODUCER ERROR: {e}")
    finally:
        if tar_stop:
            tar_stop.set()
        pool.shutdown(wait=True)
        q.put(("done", errors, skipped))


# ── Checkpoint ───────────────────────────────────────────────────────

def save_checkpoint(features_list, keys_list, checkpoint_file):
    """Atomic checkpoint save."""
    if not features_list:
        return
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = checkpoint_file.with_name(checkpoint_file.stem + "_tmp.npz")
    np.savez(tmp_file,
             features=np.stack(features_list).astype(np.float32),
             keys=np.array(keys_list, dtype=object))
    os.replace(tmp_file, checkpoint_file)


def load_checkpoint(checkpoint_file):
    """Load checkpoint. Returns (features_list, keys_list, count)."""
    if not checkpoint_file.exists():
        return [], [], 0
    try:
        data = np.load(checkpoint_file, allow_pickle=True)
        feat_list = list(data["features"])
        keys_list = list(data["keys"])
        print(f"Checkpoint loaded: {len(feat_list):,} clips from {checkpoint_file.name}")
        return feat_list, keys_list, len(feat_list)
    except Exception as e:
        print(f"  WARN: checkpoint corrupt ({e}), starting fresh")
        return [], [], 0


# ── Main ─────────────────────────────────────────────────────────────

def main():
    cleanup_temp()
    parser = argparse.ArgumentParser(
        description="GPU-RAFT optical flow motion features (13D per clip)")
    parser.add_argument("--SANITY", action="store_true",
                        help="Process 20 clips only")
    parser.add_argument("--POC", action="store_true", help="10K subset")
    parser.add_argument("--FULL", action="store_true",
                        help="Process all clips")
    parser.add_argument("--n-pairs", type=int, default=N_FRAME_PAIRS,
                        help=f"Frame pairs per clip (default: {N_FRAME_PAIRS})")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    add_gpu_mem_arg(parser)
    # iter15 (2026-05-15): single output dir holds EVERYTHING m04d produces —
    # motion_features.{npy,paths.npy,meta.json} + .m04d_checkpoint.npz. Default
    # routes under <local_data>/m04d_motion_features/ so the whole subdir rides
    # on `hf_outputs.py upload-data` → fresh GPU instances pick up a preempted
    # run via download-data + resume from checkpoint. Mirrors m10_sam_segment/
    # + m11_factor_datasets/ subdir convention. The retired --features-out arg
    # is gone — pass --output-dir if you need to override (e.g., ablation).
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for ALL m04d outputs (motion_features.npy + "
                             ".paths.npy + .meta.json + .m04d_checkpoint.npz). "
                             "Default: <local_data>/m04d_motion_features/.")
    # Cache-policy gate (iter11): every destructive delete in this module must route
    # through utils.cache_policy.guarded_delete(path, args.cache_policy, ...).
    # --cache-policy defaults to 1 (keep) so overnight re-runs never destroy cache.
    from utils.cache_policy import add_cache_policy_arg, resolve_cache_policy_interactive
    add_cache_policy_arg(parser)
    args = parser.parse_args()

    # Cache-policy prompt — shells stay thin (CLAUDE.md DELETE PROTECTION).
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    ensure_local_data(args)
    check_gpu()

    # iter15 Phase 3 (2026-05-14): cgroup envelope + OOM watchdog.
    # Container hosts cap RAM via cgroup memory.limit_in_bytes (NOT `free -h`).
    # When usage hits the limit, the kernel SIGKILLs the process with no
    # Python traceback — log ends mid-tqdm with empty shell prompt. The
    # watchdog thread prints LOUD warnings at 80/90/97% so the LAST log line
    # before SIGKILL shows the run-up to OOMKill. Forensic only; doesn't
    # PREVENT the kill — for that, tune decode_workers/producer_queue in
    # pipeline.yaml per the scaling table.
    print_cgroup_header(prefix="[m04d]")
    start_oom_watchdog(prefix="[m04d-oom-watchdog]")

    # Output routing — iter15 (2026-05-15): single subdir holds EVERYTHING m04d
    # produces (durable .npy + .paths.npy + .meta.json AND resumable
    # .m04d_checkpoint.npz). Default = <local_data>/m04d_motion_features/ so the
    # whole subdir rides on `hf_outputs.py upload-data` → fresh GPU instances
    # pick up a preempted run via download-data + resume from checkpoint. The
    # previous split (durables in <local_data>/, checkpoint in
    # outputs/<mode>/m04d_motion_features/) is retired: cache-policy=2 wipes the
    # checkpoint unconditionally, and cache-policy=1 skips m04d entirely when
    # the .npy is present (no half-written re-upload risk).
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.local_data is not None:
        output_dir = Path(args.local_data) / "m04d_motion_features"
    else:
        sys.exit("FATAL: --output-dir not supplied and --local-data missing — "
                 "cannot resolve m04d output location.")
    output_dir.mkdir(parents=True, exist_ok=True)
    features_file   = output_dir / "motion_features.npy"
    paths_file      = output_dir / "motion_features.paths.npy"
    meta_file       = output_dir / "motion_features.meta.json"
    checkpoint_file = output_dir / ".m04d_checkpoint.npz"
    print(f"Motion-features output: {features_file}")
    print(f"Paths output:           {paths_file}")
    print(f"Meta output:            {meta_file}")
    print(f"Mid-run checkpoint:     {checkpoint_file}")

    # Skip-or-regenerate guard, routed through utils.cache_policy.
    # iter15 Phase 3 fix (2026-05-14a): the original unconditional skip-if-exists
    # violated the DELETE PROTECTION contract — CACHE_POLICY_ALL=2 silently
    # skipped (e.g., Phase 0 13→23-D upgrade couldn't regenerate).
    # iter15 Phase 3 fix (2026-05-14d): also wipe stale checkpoint on --cache-policy 2
    # — the 14a fix only triggered when BOTH .npy files existed. If a prior run
    # crashed mid-stream (e.g., cgroup OOMKilled at clip 224), it left a
    # checkpoint.npz but the .npy outputs were already deleted by its own
    # cache-policy=2 prelude. Restarting with cache-policy=2 would then SKIP the
    # guard entirely (because .npy files were missing) and resume from the stale
    # checkpoint instead of starting fresh. Now: cache-policy=2 wipes ALL three
    # artifacts unconditionally, regardless of which ones exist.
    from utils.cache_policy import guarded_delete, is_recompute
    if is_recompute(args.cache_policy):
        # cache-policy=2 → authorize wipe of all stale artifacts (whichever exist).
        # Each guarded_delete is a no-op if the file is missing.
        guarded_delete(features_file, args.cache_policy,
                       label="m04d motion_features.npy")
        guarded_delete(paths_file, args.cache_policy,
                       label="m04d motion_features.paths.npy")
        guarded_delete(checkpoint_file, args.cache_policy,
                       label="m04d mid-run checkpoint")
    elif features_file.exists() and paths_file.exists() and not checkpoint_file.exists():
        # cache-policy=1 → keep existing finished output; skip if no checkpoint
        # is in flight (meaning the prior run finished cleanly).
        print(f"Motion features already exist: {features_file}")
        print("  Skipping (--cache-policy=1/keep; rerun with --cache-policy 2 "
              "to regenerate, e.g., after FEATURE_DIM change).")
        return

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    clip_limit = get_sanity_clip_limit("motion") if args.SANITY else None
    wb_run = init_wandb("m04d", mode, config=vars(args),
                        enabled=not args.no_wandb)

    print(f"Mode: {mode}")
    print(f"Output: {output_dir}")
    print(f"Frame pairs/clip: {args.n_pairs}")

    # Load subset keys
    subset_keys = load_subset(args.subset) if args.subset else set()
    if subset_keys:
        print(f"Subset: {len(subset_keys):,} keys")
        if clip_limit is None:
            clip_limit = len(subset_keys)
    if clip_limit is None:
        clip_limit = get_total_clips(local_data=getattr(args, 'local_data', None))
        if clip_limit == 0:
            print("FATAL: Cannot determine clip count. Use --subset or --local-data with manifest.json")
            sys.exit(1)

    # Checkpoint (already resolved above; load resume state if present).
    features_list, keys_list, start_count = load_checkpoint(checkpoint_file)
    processed_keys = set(keys_list)

    if start_count > 0:
        print(f"Resuming: {start_count:,} clips already processed")

    # Load RAFT model
    import torch as _torch
    device = _torch.device("cuda")
    model, transforms = load_raft_model(device)

    # Batch sizer for sub-batching within GPU forward (OOM recovery).
    # VRAM ceiling from universal pipeline.yaml key `gpu_memory_target` (#47).
    sizer = AdaptiveBatchSizer(
        initial_size=CLIPS_PER_GPU_BATCH * args.n_pairs,
        min_size=args.n_pairs,
        max_size=CLIPS_PER_GPU_BATCH * args.n_pairs,
        memory_cap=get_pipeline_config()["gpu"]["gpu_memory_target"])
    print(f"RAFT batch sizer: {sizer}")
    print(f"Producer-consumer: queue={PRODUCER_QUEUE_SIZE}, "
          f"clips/GPU_batch={CLIPS_PER_GPU_BATCH}")

    # ── Launch producer thread ──
    q = queue.Queue(maxsize=PRODUCER_QUEUE_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_producer_thread,
        args=(q, stop_event, transforms, args.n_pairs,
              clip_limit, subset_keys, processed_keys,
              getattr(args, 'local_data', None)),
        daemon=True)
    producer.start()

    # ── GPU consumer loop ──
    total = clip_limit if clip_limit < 999_999_999 else 0
    pbar = tqdm(total=total, initial=start_count,
                desc="m04d motion features", unit="clip")
    processed = start_count
    t_start = time.time()
    last_window_count = start_count
    last_window_time = t_start
    errors = 0
    producer_errors = 0
    producer_skipped = 0

    while True:
        try:
            item = q.get(timeout=300)  # 5 min timeout
        except queue.Empty:
            print("\n  TIMEOUT: producer stalled for 5 min, stopping")
            break

        if item[0] == "done":
            producer_errors = item[1]
            producer_skipped = item[2]
            break

        _, prev_all, curr_all, batch_keys, n_pairs_list = item
        # prev_all/curr_all: (total_pairs, 3, H, W) on CPU
        total_pairs = prev_all.shape[0]

        # GPU forward — sub-batch if needed for OOM safety
        batch_size = sizer.size
        all_flows = []
        i = 0
        oom_failed = False

        while i < total_pairs:
            end = min(i + batch_size, total_pairs)
            prev_sub = prev_all[i:end].contiguous().to(device)
            curr_sub = curr_all[i:end].contiguous().to(device)

            try:
                # cuDNN grid_sample overflows at large batch (PyTorch#88380)
                # fp16 autocast: ~2x speedup, ~40% VRAM reduction for RAFT CNN
                with _torch.no_grad(), _torch.backends.cudnn.flags(enabled=False), \
                     _torch.amp.autocast("cuda", dtype=_torch.float16):
                    flows = model(prev_sub, curr_sub)
                all_flows.append(flows[-1].float().cpu())  # back to fp32 for numpy
                sizer.after_batch_success()
                i = end
            except _torch.cuda.OutOfMemoryError:
                gc.collect()
                _torch.cuda.empty_cache()
                if sizer.on_oom():
                    batch_size = sizer.size
                else:
                    oom_failed = True
                    break

        if oom_failed or not all_flows:
            errors += len(batch_keys)
            continue

        # Concat flows and split per clip
        flow_cat = _torch.cat(all_flows, dim=0).numpy()  # (total_pairs, 2, H, W)

        offset = 0
        for clip_key, n_p in zip(batch_keys, n_pairs_list):
            clip_flow = flow_cat[offset:offset + n_p]
            offset += n_p

            feat = _aggregate_flow(clip_flow, n_p)
            features_list.append(feat)
            keys_list.append(clip_key)
            processed_keys.add(clip_key)
            processed += 1
            pbar.update(1)

        # Checkpoint
        if processed % CHECKPOINT_INTERVAL < len(batch_keys):
            save_checkpoint(features_list, keys_list, checkpoint_file)
            now = time.time()
            window_clips = processed - last_window_count
            window_time = now - last_window_time
            rate = window_clips / window_time if window_time > 0 else 0
            last_window_count = processed
            last_window_time = now
            pbar.set_postfix({"rate": f"{rate:.1f}/s",
                              "err": errors + producer_errors})
            log_metrics(wb_run, {"processed": processed, "errors": errors,
                                 "rate": rate}, step=processed)

    pbar.close()
    stop_event.set()
    producer.join(timeout=10)
    elapsed = time.time() - t_start
    total_errors = errors + producer_errors

    # Final save
    if not features_list:
        print("FATAL: No clips processed")
        finish_wandb(wb_run)
        sys.exit(1)

    features_arr = np.stack(features_list).astype(np.float32)
    keys_arr = np.array(keys_list, dtype=object)

    # features_file + paths_file + meta_file all live inside output_dir (set
    # above to <local_data>/m04d_motion_features/ by default). The subdir rides
    # on hf_outputs.py upload-data so the meta describing .npy shape +
    # feature_names + n_frame_pairs travels with the .npy — never orphaned in
    # scratch outputs/.
    np.save(features_file, features_arr)
    np.save(paths_file, keys_arr)
    print(f"\nSaved: {features_file} ({features_arr.shape})")
    print(f"Saved: {paths_file} ({keys_arr.shape})")

    meta = {
        "n_clips": len(features_list),
        "feature_dim": FEATURE_DIM,
        "feature_names": FEATURE_NAMES,
        "n_frame_pairs": args.n_pairs,
        "compute_time_sec": round(elapsed, 1),
        "errors": total_errors,
        "producer_errors": producer_errors,
        "producer_skipped": producer_skipped,
        "mode": mode,
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_file}")

    rate = (processed - start_count) / elapsed if elapsed > 0 else 0
    print(f"\n{'='*60}")
    print(f"m04d COMPLETE: {processed:,} clips, {elapsed:.0f}s ({rate:.1f} clips/s)")
    print(f"  Errors: {total_errors} (GPU: {errors}, producer: {producer_errors})")
    print(f"  Skipped: {producer_skipped}")
    print(f"{'='*60}")

    log_metrics(wb_run, {"total_clips": processed, "total_time_sec": elapsed,
                         "errors": total_errors}, step=processed)

    # iter11 META-fix: gate checkpoint cleanup through --cache-policy (default=1/keep).
    from utils.cache_policy import guarded_delete
    guarded_delete(checkpoint_file, args.cache_policy,
                   label="m04d motion_features checkpoint")

    finish_wandb(wb_run)

    # Force exit: torch.compile + CUDA atexit cleanup deadlocks on futex_wait_queue
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
