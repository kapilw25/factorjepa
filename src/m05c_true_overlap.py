"""
Generate augmented V-JEPA embeddings for True Overlap@K measurement.
GPU-only. Produces overlap_augA.npy and overlap_augB.npy from same clips.

USAGE:
    python -u src/m05c_true_overlap.py --SANITY 2>&1 | tee logs/m05c_overlap_sanity.log
    python -u src/m05c_true_overlap.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m05c_overlap_poc.log
"""
import argparse
import os
import queue
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    VJEPA_FRAMES_PER_CLIP, check_gpu, check_output_exists,
    load_subset, add_subset_arg, add_local_data_arg, get_output_dir,
)
from utils.gpu_batch import compute_batch_sizes, add_gpu_mem_arg
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, log_artifact, finish_wandb

from m05_vjepa_embed import (
    get_clip_key, _create_stream, decode_video_bytes, get_batch_embeddings,
    save_checkpoint, load_checkpoint,
    DECODE_WORKERS, MAX_STREAM_RETRIES, CHECKPOINT_EVERY, PREFETCH_QUEUE_SIZE,
)

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ── Augmentation Transforms (BYOL/DINO protocol) ─────────────────────

def _augment_clip_consistent(video_tensor: torch.Tensor, view: str,
                              seed: int) -> torch.Tensor:
    """Vectorized BYOL/DINO augmentation on full (T,C,H,W) tensor.

    Replaces per-frame loop with batch tensor ops (64x fewer kernel launches).
    """
    T_frames, C, H, W = video_tensor.shape
    rng = torch.Generator()
    rng.manual_seed(seed)

    if view == "A":
        # Large crop (0.4-1.0 of area)
        i, j, h, w = T.RandomResizedCrop.get_params(
            video_tensor[0], scale=(0.4, 1.0), ratio=(0.75, 1.33))
        do_flip = torch.rand(1, generator=rng).item() < 0.5
        # Color jitter params (fixed for all frames)
        brightness = 1.0 + (torch.rand(1, generator=rng).item() - 0.5) * 0.8
        contrast = 1.0 + (torch.rand(1, generator=rng).item() - 0.5) * 0.8
        saturation = 1.0 + (torch.rand(1, generator=rng).item() - 0.5) * 0.4
        hue = (torch.rand(1, generator=rng).item() - 0.5) * 0.2
    else:
        # Small crop (0.2-0.6 of area)
        i, j, h, w = T.RandomResizedCrop.get_params(
            video_tensor[0], scale=(0.2, 0.6), ratio=(0.75, 1.33))
        do_flip = torch.rand(1, generator=rng).item() < 0.5

    # ── Vectorized: crop + resize all T frames in one shot ──
    video = video_tensor.float() / 255.0          # (T, C, H, W)
    video = video[:, :, i:i+h, j:j+w]             # spatial crop (single slice)
    video = torch.nn.functional.interpolate(       # single F.interpolate vs 64
        video, size=(384, 384), mode='bilinear', align_corners=False)

    if do_flip:
        video = video.flip(-1)

    if view == "A":
        # All TF functions support (T, C, H, W) — T treated as batch dim
        video = TF.adjust_brightness(video, brightness)
        video = TF.adjust_contrast(video, contrast)
        video = TF.adjust_saturation(video, saturation)
        if abs(hue) > 0.01:
            video = TF.adjust_hue(video, hue)
    else:
        # gaussian_blur supports (*, C, H, W) via conv2d with groups=C
        video = TF.gaussian_blur(video, kernel_size=23, sigma=1.0)

    video = (video * 255).clamp(0, 255).to(torch.uint8)
    return video


# ── Producer ──────────────────────────────────────────────────────────

def _producer_overlap(processor, batch_size: int, tmp_dir: str,
                       q: queue.Queue, stop_event: threading.Event,
                       clip_limit: int, subset_keys: set,
                       processed_keys: set, num_frames: int,
                       local_data: str = None):
    """Stream clips (HF or local shards), produce two augmented views per clip."""
    produced = 0
    retries = 0

    while produced < clip_limit and not stop_event.is_set():
        try:
            ds = _create_stream(0, local_data=local_data)
            pending_bytes = []
            pending_keys = []

            for example in ds:
                if stop_event.is_set():
                    break

                clip_key = get_clip_key(example)
                if subset_keys and clip_key not in subset_keys:
                    continue
                if clip_key in processed_keys:
                    continue

                mp4_data = example.get("mp4", b"")
                mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
                if not mp4_bytes:
                    continue

                pending_bytes.append(mp4_bytes)
                pending_keys.append(clip_key)

                if len(pending_bytes) >= batch_size:
                    _decode_augment_enqueue(
                        pending_bytes, pending_keys, tmp_dir, num_frames,
                        processor, q)
                    produced += len(pending_bytes)
                    pending_bytes = []
                    pending_keys = []
                    retries = 0
                    if produced >= clip_limit:
                        break

            if pending_bytes and not stop_event.is_set():
                _decode_augment_enqueue(
                    pending_bytes, pending_keys, tmp_dir, num_frames,
                    processor, q)
                produced += len(pending_bytes)

            break  # stream exhausted

        except (ConnectionError, TimeoutError, OSError) as e:
            retries += 1
            if retries > MAX_STREAM_RETRIES:
                print(f"  ERROR: stream failed after {MAX_STREAM_RETRIES} retries: {e}")
                break
            wait = min(2 ** retries, 60)
            print(f"  WARN: stream error ({e}), retry {retries}/{MAX_STREAM_RETRIES} in {wait}s")
            time.sleep(wait)
        except Exception as e:
            print(f"  ERROR: unexpected producer error: {e}")
            import traceback
            traceback.print_exc()
            break

    q.put(("done", None, None, None))


def _decode_augment_enqueue(pending_bytes, pending_keys, tmp_dir,
                              num_frames, processor, q):
    """Decode batch, create 2 augmented views, process, enqueue.

    Decode is threaded (I/O-bound). Augmentation is sequential but vectorized
    (64x fewer kernel launches per clip). No threading for augment/processor
    to avoid ATen thread-pool oversubscription (8 workers × 80 ATen threads
    = 640+ threads → scheduler thrash → 0% throughput).
    """
    # 1) Parallel decode (I/O-bound, GIL-free — threading helps)
    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
        futures = [
            pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
            for b, k in zip(pending_bytes, pending_keys)
        ]
        results = [(f.result(), k) for f, k in zip(futures, pending_keys)]

    views_a = []
    views_b = []
    keys = []

    # 2) Sequential augmentation (vectorized per-clip: 1 F.interpolate vs 64)
    for tensor, key in results:
        if tensor is None:
            continue
        seed = hash(key) % (2**31)
        aug_a = _augment_clip_consistent(tensor, "A", seed)
        aug_b = _augment_clip_consistent(tensor, "B", seed + 1)
        views_a.append(aug_a)
        views_b.append(aug_b)
        keys.append(key)

    if not views_a:
        return

    # 3) Sequential processor (normalization — fast after vectorized augment)
    pixels_a = torch.cat([processor(v, return_tensors="pt")["pixel_values_videos"]
                          for v in views_a], dim=0)
    pixels_b = torch.cat([processor(v, return_tensors="pt")["pixel_values_videos"]
                          for v in views_b], dim=0)
    q.put(("batch", pixels_a, pixels_b, keys[:]))


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate augmented V-JEPA embeddings for True Overlap@K")
    parser.add_argument("--SANITY", action="store_true", help="Process 5 clips only")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    parser.add_argument("--batch-size", type=int, default=None)
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
    device = "cuda"

    output_dir = get_output_dir(args.subset, sanity=args.SANITY)
    output_dir.mkdir(parents=True, exist_ok=True)
    aug_a_file = output_dir / "overlap_augA.npy"
    aug_b_file = output_dir / "overlap_augB.npy"
    keys_file = output_dir / "overlap_keys.npy"
    checkpoint_file = output_dir / ".m05c_checkpoint.npz"

    if aug_a_file.exists() and aug_b_file.exists():
        if not check_output_exists([aug_a_file, aug_b_file], "overlap embeddings"):
            print("Using cached augmented embeddings.")
            return

    subset_keys = load_subset(args.subset) if args.subset else set()

    # Use deduped keys from V-JEPA embeddings.paths.npy instead of full subset.
    # m05 deduplicates at cosine sim > 0.95 (e.g. 10K → 5,105). Clips outside
    # this set have no base embedding — m06 can't use their overlap data.
    deduped_paths_file = output_dir / "embeddings.paths.npy"
    if deduped_paths_file.exists() and not args.SANITY:
        deduped_keys = set(np.load(deduped_paths_file, allow_pickle=True).tolist())
        original_count = len(subset_keys) if subset_keys else 115_687
        subset_keys = (subset_keys & deduped_keys) if subset_keys else deduped_keys
        print(f"[DEDUP] Target: {len(subset_keys):,} deduped keys "
              f"(from {original_count:,} in subset → {len(deduped_keys):,} after V-JEPA dedup)")
    elif not args.SANITY:
        print("WARNING: embeddings.paths.npy not found — processing full subset "
              "(run m05 first for dedup optimization)")

    clip_limit = 5 if args.SANITY else (len(subset_keys) if subset_keys else 115_687)

    batch_size = args.batch_size or compute_batch_sizes(gpu_vram_gb=args.gpu_mem)["vjepa"]
    if args.SANITY:
        batch_size = min(batch_size, 2)

    # Load V-JEPA model
    from transformers import AutoModel, AutoVideoProcessor
    from utils.config import ENCODER_REGISTRY

    model_id = ENCODER_REGISTRY["vjepa"]["model_id"]
    print(f"Loading V-JEPA: {model_id}")
    processor = AutoVideoProcessor.from_pretrained(model_id)
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print("FATAL: flash-attn not installed.")
        sys.exit(1)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map="auto", attn_implementation="flash_attention_2",
    )
    model.eval()
    model = torch.compile(model)
    print(f"V-JEPA loaded for augmented inference")

    mode = "SANITY" if args.SANITY else ("POC" if args.subset else "FULL")
    wb_run = init_wandb("m05c", mode, config=vars(args), enabled=not args.no_wandb)

    # Resume from checkpoint (stores interleaved A,B embeddings)
    all_emb_a = []
    all_emb_b = []
    all_keys = []
    processed_keys = set()

    if checkpoint_file.exists():
        data = np.load(checkpoint_file, allow_pickle=True)
        all_emb_a = list(data["emb_a"])
        all_emb_b = list(data["emb_b"])
        all_keys = list(data["keys"])
        processed_keys = set(all_keys)
        print(f"Checkpoint loaded: {len(all_keys):,} clips")

    tmp_base = output_dir / "tmp_m05c"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)

    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_producer_overlap,
        args=(processor, batch_size, tmp_dir, q, stop_event,
              clip_limit, subset_keys, processed_keys, VJEPA_FRAMES_PER_CLIP,
              getattr(args, 'local_data', None)),
        daemon=True,
    )
    producer.start()

    start_time = time.time()
    try:
        while True:
            try:
                msg_type, pixels_a, pixels_b, batch_keys = q.get(timeout=600)
            except queue.Empty:
                print("\nProducer timeout (10 min). Saving checkpoint...")
                break

            if msg_type == "done":
                break

            emb_a = get_batch_embeddings(model, pixels_a, device)
            emb_b = get_batch_embeddings(model, pixels_b, device)

            for ea, eb, k in zip(emb_a, emb_b, batch_keys):
                all_emb_a.append(ea)
                all_emb_b.append(eb)
                all_keys.append(k)
                processed_keys.add(k)

            elapsed = time.time() - start_time
            throughput = len(all_keys) / elapsed if elapsed > 0 else 0
            print(f"  [{len(all_keys):,}/{clip_limit:,}] {throughput:.1f} clips/s (2 views each)")

            if len(all_keys) % CHECKPOINT_EVERY < batch_size:
                _save_overlap_checkpoint(all_emb_a, all_emb_b, all_keys, checkpoint_file)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        stop_event.set()
    finally:
        stop_event.set()
        _save_overlap_checkpoint(all_emb_a, all_emb_b, all_keys, checkpoint_file)
        producer.join(timeout=10)
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            tmp_base.rmdir()
        except OSError:
            pass

    if not all_emb_a:
        print("ERROR: No embeddings collected.")
        sys.exit(1)

    arr_a = np.stack(all_emb_a).astype(np.float32)
    arr_b = np.stack(all_emb_b).astype(np.float32)

    # Filter to deduped keys only (checkpoint may contain extra clips from prior runs)
    if subset_keys and len(all_keys) > len(subset_keys):
        keep = np.array([k in subset_keys for k in all_keys])
        n_before = len(all_keys)
        arr_a = arr_a[keep]
        arr_b = arr_b[keep]
        all_keys = [k for k, m in zip(all_keys, keep) if m]
        print(f"  Filtered: {n_before:,} → {len(all_keys):,} (removed {n_before - len(all_keys):,} non-deduped)")

    np.save(aug_a_file, arr_a)
    np.save(aug_b_file, arr_b)
    np.save(keys_file, np.array(all_keys, dtype=object))

    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print(f"\nSaved: {aug_a_file} ({arr_a.shape})")
    print(f"Saved: {aug_b_file} ({arr_b.shape})")
    print(f"Saved: {keys_file} ({len(all_keys)} keys)")
    print(f"\nNext: python -u src/m06_faiss_metrics.py --true-overlap --FULL --subset data/subset_10k.json")

    log_metrics(wb_run, {"total_clips": len(all_keys), "embedding_dim": arr_a.shape[1]})
    log_artifact(wb_run, "overlap_augA", str(aug_a_file))
    log_artifact(wb_run, "overlap_augB", str(aug_b_file))
    finish_wandb(wb_run)


def _save_overlap_checkpoint(emb_a, emb_b, keys, checkpoint_file):
    """Save overlap checkpoint (both views + keys)."""
    if not emb_a:
        return
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = checkpoint_file.with_name(checkpoint_file.stem + "_tmp.npz")
    np.savez(tmp,
             emb_a=np.stack(emb_a).astype(np.float32),
             emb_b=np.stack(emb_b).astype(np.float32),
             keys=np.array(keys, dtype=object))
    os.replace(tmp, checkpoint_file)


if __name__ == "__main__":
    main()
