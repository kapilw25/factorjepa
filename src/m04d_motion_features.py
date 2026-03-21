"""
GPU-RAFT optical flow motion features per clip (13D vector).
Extracts deterministic temporal ground-truth: mean/std/max magnitude,
8-bin direction histogram, camera motion (dx, dy).

USAGE:
    python -u src/m04d_motion_features.py --SANITY --subset data/subset_10k.json \
        --local-data data/subset_10k_local 2>&1 | tee logs/m04d_sanity.log
    python -u src/m04d_motion_features.py --FULL --subset data/subset_10k.json \
        --local-data data/subset_10k_local 2>&1 | tee logs/m04d_motion.log
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    HF_DATASET_REPO, check_gpu,
    add_subset_arg, add_local_data_arg, get_output_dir, load_subset,
)
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)
from utils.gpu_batch import add_gpu_mem_arg, AdaptiveBatchSizer

# Lazy imports — torch + torchvision loaded after check_gpu()
torch = None
raft_large = None
Raft_Large_Weights = None

FEATURE_DIM = 13
FEATURE_NAMES = [
    "mean_magnitude", "magnitude_std", "max_magnitude",
    "dir_hist_0", "dir_hist_1", "dir_hist_2", "dir_hist_3",
    "dir_hist_4", "dir_hist_5", "dir_hist_6", "dir_hist_7",
    "camera_motion_x", "camera_motion_y",
]
N_FRAME_PAIRS = 16
CHECKPOINT_INTERVAL = 200


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


# ── Video Decode ─────────────────────────────────────────────────────

def decode_video_frames(video_bytes: bytes, n_pairs: int = N_FRAME_PAIRS):
    """Decode video bytes to evenly-spaced consecutive frame pairs.

    Returns list of (prev_frame, curr_frame) as uint8 numpy arrays (H, W, 3).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2:
            cap.release()
            return []

        # Sample n_pairs evenly-spaced consecutive pairs
        stride = max(1, (total_frames - 1) // n_pairs)
        pair_starts = [i * stride for i in range(n_pairs)]
        pair_starts = [s for s in pair_starts if s + 1 < total_frames]

        pairs = []
        frame_cache = {}
        for start in pair_starts:
            for idx in [start, start + 1]:
                if idx not in frame_cache:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    frame_cache[idx] = frame if ret else None
            prev = frame_cache.get(start)
            curr = frame_cache.get(start + 1)
            if prev is not None and curr is not None:
                pairs.append((prev, curr))
        cap.release()
        return pairs
    finally:
        os.unlink(tmp.name)


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
    transforms = weights.transforms()
    print(f"RAFT-Large loaded on {device} (weights: C_T_SKHT_V2)")
    return model, transforms


def _preprocess_pairs(frame_pairs, transforms):
    """Preprocess frame pairs into batched tensors for RAFT.

    Returns (prev_batch, curr_batch) each of shape (N, 3, 360, 520).
    """
    h, w = 360, 520
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


def compute_clip_motion(model, transforms, frame_pairs, device, sizer=None):
    """Extract 13D motion feature from frame pairs via batched GPU-RAFT.

    Batches all frame pairs into a single forward pass. Uses AdaptiveBatchSizer
    for OOM recovery (halves batch on OOM).

    Returns np.ndarray of shape (13,) float32, or None if no valid pairs.
    """
    if not frame_pairs:
        return None

    # Preprocess all pairs into tensors (CPU)
    prev_batch, curr_batch = _preprocess_pairs(frame_pairs, transforms)
    n_total = len(frame_pairs)

    # Determine sub-batch size
    batch_size = sizer.size if sizer else n_total
    all_flows = []

    i = 0
    while i < n_total:
        end = min(i + batch_size, n_total)
        prev_sub = prev_batch[i:end].to(device)
        curr_sub = curr_batch[i:end].to(device)

        oom = False
        try:
            with torch.no_grad():
                flows = model(prev_sub, curr_sub)
            # flows[-1] = last refinement iteration, shape (B, 2, H, W)
            all_flows.append(flows[-1].cpu())
            if sizer:
                sizer.after_batch_success()
            i = end
        except torch.cuda.OutOfMemoryError:
            oom = True

        if oom:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if sizer:
                if not sizer.on_oom():
                    return None  # give up on this clip
                batch_size = sizer.size
            else:
                # No sizer — fall back to sequential
                batch_size = max(1, batch_size // 2)
                if batch_size == 0:
                    return None
            # Retry from same position
            continue

    # Concat all sub-batch flows → (N, 2, H, W)
    flow_all = torch.cat(all_flows, dim=0).numpy()  # (N, 2, H, W)

    dx_all = flow_all[:, 0]  # (N, H, W)
    dy_all = flow_all[:, 1]  # (N, H, W)
    mag_all = np.sqrt(dx_all**2 + dy_all**2)
    ang_all = np.arctan2(dy_all, dx_all)

    # Aggregate across all frame pairs
    flat_mag = mag_all.flatten()
    mean_mag = float(np.mean(flat_mag))
    std_mag = float(np.std(flat_mag))
    max_mag = float(np.max(flat_mag))

    # 8-bin direction histogram (normalized)
    flat_ang = ang_all.flatten()
    hist, _ = np.histogram(flat_ang, bins=8, range=(-np.pi, np.pi))
    hist = hist.astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum

    # Camera motion: median flow per pair, then median across pairs
    per_pair_dx = np.median(dx_all.reshape(n_total, -1), axis=1)
    per_pair_dy = np.median(dy_all.reshape(n_total, -1), axis=1)
    cam_x = float(np.median(per_pair_dx))
    cam_y = float(np.median(per_pair_dy))

    return np.array([mean_mag, std_mag, max_mag, *hist, cam_x, cam_y],
                    dtype=np.float32)


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
    parser = argparse.ArgumentParser(
        description="GPU-RAFT optical flow motion features (13D per clip)")
    parser.add_argument("--SANITY", action="store_true",
                        help="Process 20 clips only")
    parser.add_argument("--FULL", action="store_true",
                        help="Process all clips")
    parser.add_argument("--n-pairs", type=int, default=N_FRAME_PAIRS,
                        help=f"Frame pairs per clip (default: {N_FRAME_PAIRS})")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    add_gpu_mem_arg(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    check_gpu()

    # Output routing
    output_dir = get_output_dir(args.subset, sanity=args.SANITY)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "SANITY" if args.SANITY else ("POC" if args.subset else "FULL")
    clip_limit = 20 if args.SANITY else None
    wb_run = init_wandb("m04d", mode, config=vars(args),
                        enabled=not args.no_wandb)

    print(f"Mode: {mode}")
    print(f"Output: {output_dir}")
    print(f"Frame pairs/clip: {args.n_pairs}")

    # Load subset keys
    subset_keys = load_subset(args.subset)
    if subset_keys:
        print(f"Subset: {len(subset_keys):,} keys")
        if clip_limit is None:
            clip_limit = len(subset_keys)

    # Checkpoint
    checkpoint_file = output_dir / ".m04d_checkpoint.npz"
    features_list, keys_list, start_count = load_checkpoint(checkpoint_file)
    processed_keys = set(keys_list)

    if start_count > 0:
        print(f"Resuming: {start_count:,} clips already processed")

    # Load RAFT model
    import torch as _torch
    device = _torch.device("cuda")
    model, transforms = load_raft_model(device)

    # Batch sizer: start with all n_pairs in one forward, OOM halves
    sizer = AdaptiveBatchSizer(initial_size=args.n_pairs, min_size=1,
                               max_size=args.n_pairs, memory_cap=0.85)
    print(f"RAFT batch sizer: {sizer}")

    # Stream clips
    ds = _create_stream(local_data=getattr(args, 'local_data', None))

    total = clip_limit or 0
    pbar = tqdm(total=total, initial=start_count,
                desc="m04d motion features", unit="clip")
    processed = start_count
    skipped = 0
    t_start = time.time()
    errors = 0

    for example in ds:
        if clip_limit and processed >= clip_limit:
            break

        # Subset filter
        clip_key = get_clip_key(example)
        if subset_keys and clip_key not in subset_keys:
            skipped += 1
            continue

        # Resume dedup
        if clip_key in processed_keys:
            skipped += 1
            continue

        # Decode video
        mp4_bytes = example.get("mp4", b"")
        if isinstance(mp4_bytes, str):
            mp4_bytes = mp4_bytes.encode()
        if not mp4_bytes or len(mp4_bytes) < 1000:
            errors += 1
            continue

        try:
            frame_pairs = decode_video_frames(mp4_bytes, n_pairs=args.n_pairs)
            if not frame_pairs:
                errors += 1
                continue

            features = compute_clip_motion(model, transforms, frame_pairs, device,
                                         sizer=sizer)
            if features is None:
                errors += 1
                continue

            features_list.append(features)
            keys_list.append(clip_key)
            processed_keys.add(clip_key)
            processed += 1
            pbar.update(1)

            # Checkpoint
            if processed % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(features_list, keys_list, checkpoint_file)
                elapsed = time.time() - t_start
                rate = (processed - start_count) / elapsed if elapsed > 0 else 0
                pbar.set_postfix({"rate": f"{rate:.1f}/s", "err": errors,
                                  "skip": skipped})
                log_metrics(wb_run, {"processed": processed, "errors": errors,
                                     "rate": rate}, step=processed)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\n  ERROR [{clip_key}]: {e}")
            continue

    pbar.close()
    elapsed = time.time() - t_start

    # Final save
    if not features_list:
        print("FATAL: No clips processed")
        finish_wandb(wb_run)
        sys.exit(1)

    features_arr = np.stack(features_list).astype(np.float32)
    keys_arr = np.array(keys_list, dtype=object)

    features_file = output_dir / "motion_features.npy"
    paths_file = output_dir / "motion_features.paths.npy"
    meta_file = output_dir / "motion_features_meta.json"

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
        "errors": errors,
        "skipped": skipped,
        "mode": mode,
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_file}")

    rate = (processed - start_count) / elapsed if elapsed > 0 else 0
    print(f"\n{'='*60}")
    print(f"m04d COMPLETE: {processed:,} clips, {elapsed:.0f}s ({rate:.1f} clips/s)")
    print(f"  Errors: {errors}, Skipped: {skipped}")
    print(f"{'='*60}")

    log_metrics(wb_run, {"total_clips": processed, "total_time_sec": elapsed,
                         "errors": errors}, step=processed)

    # Cleanup checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"Removed checkpoint: {checkpoint_file.name}")

    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
