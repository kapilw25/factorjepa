"""
Generate baseline embeddings: Random, DINOv2, CLIP, Shuffled V-JEPA.
GPU-only for DINOv2/CLIP/Shuffled (no CPU fallback). Random is CPU-safe.

USAGE:
    python -u src/m05b_baselines.py --encoder all --SANITY 2>&1 | tee logs/m05b_all_sanity.log
    python -u src/m05b_baselines.py --encoder all --POC --subset data/subset_10k.json --local-data data/subset_10k_local 2>&1 | tee logs/m05b_all_poc.log
    python -u src/m05b_baselines.py --encoder all --FULL --local-data data/full_local 2>&1 | tee logs/m05b_all_full.log
"""
import argparse
import io
import json
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    ENCODER_REGISTRY, VJEPA_EMBEDDING_DIM, VJEPA_FRAMES_PER_CLIP,
    check_gpu, check_output_exists, load_subset, add_subset_arg, add_local_data_arg,
    get_output_dir, get_encoder_files,
    get_sanity_clip_limit, get_total_clips, get_pipeline_config,
)
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.gpu_batch import compute_batch_sizes, add_gpu_mem_arg, cuda_cleanup, cleanup_temp
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, log_artifact, finish_wandb

# Reuse HF streaming + video decode from m05
from m05_vjepa_embed import (
    get_clip_key, _create_stream, decode_video_bytes,
    save_checkpoint, load_checkpoint,
    DECODE_WORKERS, MAX_STREAM_RETRIES, CHECKPOINT_EVERY, PREFETCH_QUEUE_SIZE,
)

# Lazy imports (GPU-only libs)
torch = None
_TORCH_IMPORTED = False


def _import_torch():
    global torch, _TORCH_IMPORTED
    if not _TORCH_IMPORTED:
        import torch as _torch
        torch = _torch
        _TORCH_IMPORTED = True
    return torch


# ── Random Baseline (CPU-safe) ───────────────────────────────────────

def generate_random(clip_keys: list, output_dir: Path, args):
    """Task 16: Random L2-normalized vectors. CPU-only, deterministic."""
    dim = VJEPA_EMBEDDING_DIM  # 1408 — match V-JEPA for apples-to-apples
    n = len(clip_keys)
    print(f"Generating {n:,} random embeddings (dim={dim}, seed=42)")

    rng = np.random.RandomState(42)
    embeddings = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings /= norms

    files = get_encoder_files("random", output_dir)
    np.save(files["embeddings"], embeddings)
    np.save(files["paths"], np.array(clip_keys, dtype=object))

    print(f"Saved: {files['embeddings']} ({embeddings.shape})")
    print(f"Saved: {files['paths']} ({len(clip_keys)} keys)")
    return embeddings, clip_keys


# ── Image Encoder Baselines (DINOv2, CLIP) ───────────────────────────

def _extract_middle_frame(video_tensor, num_frames: int):
    """Extract middle frame from (T, C, H, W) tensor -> PIL Image."""
    from PIL import Image

    mid_idx = video_tensor.shape[0] // 2
    frame = video_tensor[mid_idx]  # (C, H, W)
    if hasattr(frame, 'numpy'):
        frame = frame.numpy()
    # (C, H, W) -> (H, W, C)
    frame = np.transpose(frame, (1, 2, 0))
    return Image.fromarray(frame.astype(np.uint8))


def _decode_and_extract_frame(mp4_bytes: bytes, clip_key: str,
                               tmp_dir: str, num_frames: int):
    """Extract middle frame via decord (single-frame seek, no full decode).

    5-10x faster than PyAV full decode for image encoders (CLIP/DINOv2)
    that only need 1 frame per clip. Falls back to PyAV on decord failure.
    """
    from PIL import Image

    try:
        from decord import VideoReader
        vr = VideoReader(io.BytesIO(mp4_bytes))
        mid_idx = len(vr) // 2
        frame = vr[mid_idx].asnumpy()  # (H, W, C) uint8
        return Image.fromarray(frame), clip_key
    except Exception:
        # Fallback: full PyAV decode (slower but more robust)
        tensor = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames)
        if tensor is None:
            return None, clip_key
        return _extract_middle_frame(tensor, num_frames), clip_key


def _producer_image_baseline(processor, batch_size: int, tmp_dir: str,
                              q: queue.Queue, stop_event: threading.Event,
                              clip_limit: int, subset_keys: set,
                              processed_keys: set, num_frames: int,
                              local_data: str = None):
    """Parallel TAR reading + parallel decode for image encoders (CLIP/DINOv2).

    8 threads read TARs concurrently → shared clip queue → DECODE_WORKERS threads
    decode + extract middle frame → processor batches → GPU consumer queue.
    """
    produced = 0

    if local_data:
        # Parallel TAR reader: 8 threads reading different TARs simultaneously
        clip_q, tar_stop, reader = iter_clips_parallel(
            local_data, subset_keys=subset_keys, processed_keys=processed_keys)
    else:
        # Fallback: HF streaming (no parallel TAR reading)
        clip_q, tar_stop, _reader = None, None, None

    try:
        if clip_q is not None:
            # Fast path: parallel TAR reading → parallel decode → batch
            pending = []  # list of (mp4_bytes, clip_key)

            while produced < clip_limit and not stop_event.is_set():
                try:
                    item = clip_q.get(timeout=120)
                except queue.Empty:
                    print("  Parallel TAR reader timeout (2 min), flushing remaining")
                    break

                if item is None:  # sentinel: all TARs exhausted
                    break

                clip_key, mp4_bytes = item
                pending.append((mp4_bytes, clip_key))

                if len(pending) >= batch_size:
                    # Parallel decode + frame extraction
                    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                        futures = [
                            pool.submit(_decode_and_extract_frame, b, k, tmp_dir, num_frames)
                            for b, k in pending
                        ]
                        results = [f.result() for f in futures]

                    frames = [img for img, _ in results if img is not None]
                    keys = [k for img, k in results if img is not None]

                    if frames:
                        inputs = processor(images=frames, return_tensors="pt")
                        q.put(("batch", inputs, keys[:]))
                        produced += len(frames)

                    pending = []

                    if produced >= clip_limit:
                        break

            # Final partial batch
            if pending and not stop_event.is_set():
                with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                    futures = [
                        pool.submit(_decode_and_extract_frame, b, k, tmp_dir, num_frames)
                        for b, k in pending
                    ]
                    results = [f.result() for f in futures]

                frames = [img for img, _ in results if img is not None]
                keys = [k for img, k in results if img is not None]

                if frames:
                    inputs = processor(images=frames, return_tensors="pt")
                    q.put(("batch", inputs, keys[:]))
                    produced += len(frames)

        else:
            # HF streaming fallback (no local data)
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
                            with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                                futures = [
                                    pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
                                    for b, k in zip(pending_bytes, pending_keys)
                                ]
                                results = [(f.result(), k) for f, k in zip(futures, pending_keys)]

                            frames = [_extract_middle_frame(t, num_frames) for t, k in results if t is not None]
                            keys = [k for t, k in results if t is not None]

                            if frames:
                                inputs = processor(images=frames, return_tensors="pt")
                                q.put(("batch", inputs, keys[:]))
                                produced += len(frames)

                            pending_bytes, pending_keys = [], []
                            if produced >= clip_limit:
                                break

                    break  # stream exhausted

                except (ConnectionError, TimeoutError, OSError) as e:
                    retries += 1
                    if retries > MAX_STREAM_RETRIES:
                        print(f"  ERROR: stream failed after {MAX_STREAM_RETRIES} retries: {e}")
                        break
                    wait = min(2 ** retries, 60)
                    print(f"  WARN: stream error ({e}), retry {retries}/{MAX_STREAM_RETRIES}")
                    time.sleep(wait)

    except Exception as e:
        print(f"  ERROR: unexpected producer error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if tar_stop:
            tar_stop.set()

    q.put(("done", None, None))


def generate_dinov2(output_dir: Path, args, clip_limit: int, subset_keys: set):
    """DINOv2 ViT-g/14 — middle frame, CLS token, 1536-dim."""
    torch = _import_torch()
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda"
    model_id = ENCODER_REGISTRY["dinov2"]["model_id"]
    files = get_encoder_files("dinov2", output_dir)
    checkpoint_file = output_dir / ".m05b_dinov2_checkpoint.npz"

    print(f"Loading model: {model_id}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16,
                                      device_map="auto",
                                      attn_implementation="flash_attention_2")
    model.eval()
    print("Applying torch.compile (first batch will be slow due to compilation)...")
    model = torch.compile(model)
    dim = ENCODER_REGISTRY["dinov2"]["dim"]
    print(f"DINOv2 loaded (dim={dim}, dtype=float16, FA2+compiled)")

    all_embeddings, all_keys, resume_count = load_checkpoint(checkpoint_file)
    processed_keys = set(all_keys)

    # Read from: 1) CLI arg, 2) profiler JSON, 3) pipeline.yaml, 4) linear estimate
    batch_size = None
    if args.batch_size:
        batch_size = args.batch_size
    else:
        _prof_path = Path("outputs/profile/inference/dinov2/profile_data.json")
        if _prof_path.exists():
            _prof = json.load(open(_prof_path))
            if "optimal_bs" in _prof:
                batch_size = _prof["optimal_bs"]
                print(f"Batch size: {batch_size} (from inference/dinov2 profiler)")
        if batch_size is None:
            batch_size = get_pipeline_config()["gpu"].get("inference_dinov2_bs",
                         compute_batch_sizes(gpu_vram_gb=args.gpu_mem)["image_encoder"])
    if args.SANITY:
        batch_size = min(batch_size, 4)

    tmp_base = output_dir / "tmp_m05b"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)
    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_producer_image_baseline,
        args=(processor, batch_size, tmp_dir, q, stop_event, clip_limit,
              subset_keys, processed_keys, VJEPA_FRAMES_PER_CLIP,
              getattr(args, 'local_data', None)),
        daemon=True,
    )
    producer.start()

    start_time = time.time()
    last_window_count = len(all_embeddings)
    last_window_time = start_time
    pbar = tqdm(total=clip_limit, initial=resume_count,
                desc="m05b dinov2", unit="clip")
    try:
        while True:
            try:
                msg_type, batch_inputs, batch_keys = q.get(timeout=60)
            except queue.Empty:
                print("\nProducer timeout (10 min). Saving checkpoint...")
                break

            if msg_type == "done":
                break

            inputs = {k: v.to(device) for k, v in batch_inputs.items() if hasattr(v, 'to')}
            with torch.no_grad():
                outputs = model(**inputs)
                # DINOv2: CLS token is first position
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)

            # L2-normalize
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb /= norms

            # Enforce clip limit: truncate batch if it would overshoot
            remaining = clip_limit - len(all_embeddings)
            for e, k in zip(emb[:remaining], batch_keys[:remaining]):
                all_embeddings.append(e)
                all_keys.append(k)
                processed_keys.add(k)

            pbar.update(min(len(batch_keys), remaining))

            if len(all_embeddings) >= clip_limit:
                stop_event.set()
                break

            now = time.time()
            window_clips = len(all_embeddings) - last_window_count
            window_time = now - last_window_time
            throughput = window_clips / window_time if window_time > 0 else 0
            last_window_count = len(all_embeddings)
            last_window_time = now
            pbar.set_postfix({"rate": f"{throughput:.1f}/s"})

            if len(all_embeddings) % CHECKPOINT_EVERY < batch_size:
                save_checkpoint(list(all_embeddings), list(all_keys), checkpoint_file)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        stop_event.set()
    finally:
        pbar.close()
        stop_event.set()
        save_checkpoint(all_embeddings, all_keys, checkpoint_file)
        producer.join(timeout=10)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _finalize(all_embeddings, all_keys, files, checkpoint_file)
    return np.stack(all_embeddings).astype(np.float32), all_keys


def generate_clip(output_dir: Path, args, clip_limit: int, subset_keys: set):
    """Task 19: CLIP ViT-L/14 — middle frame, image features, 768-dim."""
    torch = _import_torch()
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda"
    model_id = ENCODER_REGISTRY["clip"]["model_id"]
    files = get_encoder_files("clip", output_dir)
    checkpoint_file = output_dir / ".m05b_clip_checkpoint.npz"

    print(f"Loading model: {model_id}")
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16,
                                      device_map="auto",
                                      attn_implementation="sdpa")
    model.eval()
    print("Applying torch.compile (first batch will be slow due to compilation)...")
    model = torch.compile(model)
    print(f"CLIP loaded (dim=768, dtype=float16, SDPA+compiled)")

    all_embeddings, all_keys, resume_count = load_checkpoint(checkpoint_file)
    processed_keys = set(all_keys)

    batch_size = args.batch_size or compute_batch_sizes(gpu_vram_gb=args.gpu_mem)["image_encoder"]
    if args.SANITY:
        batch_size = min(batch_size, 4)

    tmp_base = output_dir / "tmp_m05b"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)
    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_producer_image_baseline,
        args=(processor, batch_size, tmp_dir, q, stop_event, clip_limit,
              subset_keys, processed_keys, VJEPA_FRAMES_PER_CLIP,
              getattr(args, 'local_data', None)),
        daemon=True,
    )
    producer.start()

    start_time = time.time()
    last_window_count = len(all_embeddings)
    last_window_time = start_time
    pbar = tqdm(total=clip_limit, initial=resume_count,
                desc="m05b clip", unit="clip")
    try:
        while True:
            try:
                msg_type, batch_inputs, batch_keys = q.get(timeout=60)
            except queue.Empty:
                print("\nProducer timeout (10 min). Saving checkpoint...")
                break

            if msg_type == "done":
                break

            inputs = {k: v.to(device) for k, v in batch_inputs.items() if hasattr(v, 'to')}
            with torch.no_grad():
                emb = model.get_image_features(**inputs).cpu().numpy().astype(np.float32)

            # L2-normalize
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb /= norms

            # Enforce clip limit: truncate batch if it would overshoot
            remaining = clip_limit - len(all_embeddings)
            for e, k in zip(emb[:remaining], batch_keys[:remaining]):
                all_embeddings.append(e)
                all_keys.append(k)
                processed_keys.add(k)

            pbar.update(min(len(batch_keys), remaining))

            if len(all_embeddings) >= clip_limit:
                stop_event.set()
                break

            now = time.time()
            window_clips = len(all_embeddings) - last_window_count
            window_time = now - last_window_time
            throughput = window_clips / window_time if window_time > 0 else 0
            last_window_count = len(all_embeddings)
            last_window_time = now
            pbar.set_postfix({"rate": f"{throughput:.1f}/s"})

            if len(all_embeddings) % CHECKPOINT_EVERY < batch_size:
                save_checkpoint(list(all_embeddings), list(all_keys), checkpoint_file)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        stop_event.set()
    finally:
        pbar.close()
        stop_event.set()
        save_checkpoint(all_embeddings, all_keys, checkpoint_file)
        producer.join(timeout=10)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _finalize(all_embeddings, all_keys, files, checkpoint_file)
    return np.stack(all_embeddings).astype(np.float32), all_keys


# ── Shuffled V-JEPA ──────────────────────────────────────────────────

def _producer_shuffled_vjepa(processor, batch_size: int, tmp_dir: str,
                              q: queue.Queue, stop_event: threading.Event,
                              clip_limit: int, subset_keys: set,
                              processed_keys: set, num_frames: int,
                              local_data: str = None):
    """Stream from HF (or local shards), decode, shuffle frame order, process for V-JEPA."""
    torch = _import_torch()
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
                    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                        futures = [
                            pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
                            for b, k in zip(pending_bytes, pending_keys)
                        ]
                        results = [(f.result(), k) for f, k in zip(futures, pending_keys)]

                    batch_tensors = []
                    batch_keys = []
                    for tensor, key in results:
                        if tensor is not None:
                            # Shuffle frames: deterministic per clip key
                            seed = hash(key) % (2**31)
                            rng = torch.Generator()
                            rng.manual_seed(seed)
                            perm = torch.randperm(tensor.shape[0], generator=rng)
                            tensor = tensor[perm]
                            batch_tensors.append(tensor)
                            batch_keys.append(key)

                    if batch_tensors:
                        processed_list = []
                        for vt in batch_tensors:
                            processed = processor(vt, return_tensors="pt")
                            processed_list.append(processed["pixel_values_videos"])
                        batched_pixels = torch.cat(processed_list, dim=0)
                        q.put(("batch", batched_pixels, batch_keys[:]))
                        produced += len(batch_keys)

                    pending_bytes = []
                    pending_keys = []
                    retries = 0

                    if produced >= clip_limit:
                        break

            # Final partial batch
            if pending_bytes and not stop_event.is_set():
                with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                    futures = [
                        pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
                        for b, k in zip(pending_bytes, pending_keys)
                    ]
                    results = [(f.result(), k) for f, k in zip(futures, pending_keys)]

                batch_tensors = []
                batch_keys = []
                for tensor, key in results:
                    if tensor is not None:
                        seed = hash(key) % (2**31)
                        rng = torch.Generator()
                        rng.manual_seed(seed)
                        perm = torch.randperm(tensor.shape[0], generator=rng)
                        tensor = tensor[perm]
                        batch_tensors.append(tensor)
                        batch_keys.append(key)

                if batch_tensors:
                    processed_list = []
                    for vt in batch_tensors:
                        processed = processor(vt, return_tensors="pt")
                        processed_list.append(processed["pixel_values_videos"])
                    batched_pixels = torch.cat(processed_list, dim=0)
                    q.put(("batch", batched_pixels, batch_keys[:]))
                    produced += len(batch_keys)

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

    q.put(("done", None, None))


def generate_shuffled_vjepa(output_dir: Path, args, clip_limit: int, subset_keys: set):
    """Task 18: Shuffled V-JEPA — same model, shuffled frame order, 1408-dim."""
    torch = _import_torch()
    from transformers import AutoModel, AutoVideoProcessor
    from m05_vjepa_embed import get_batch_embeddings

    device = "cuda"
    model_id = ENCODER_REGISTRY["vjepa_shuffled"]["model_id"]
    files = get_encoder_files("vjepa_shuffled", output_dir)
    checkpoint_file = output_dir / ".m05b_vjepa_shuffled_checkpoint.npz"

    print(f"Loading model: {model_id}")
    processor = AutoVideoProcessor.from_pretrained(model_id)
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        print("FATAL: flash-attn not installed. V-JEPA 2 ViT-G requires Flash Attention 2.")
        sys.exit(1)
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map="auto", attn_implementation="flash_attention_2",
    )
    model.eval()
    model = torch.compile(model)
    print(f"V-JEPA loaded for shuffled inference (dim=1408, dtype={next(model.parameters()).dtype})")

    all_embeddings, all_keys, resume_count = load_checkpoint(checkpoint_file)
    processed_keys = set(all_keys)

    # Read from: 1) CLI arg, 2) profiler JSON, 3) pipeline.yaml, 4) linear estimate
    batch_size = None
    if args.batch_size:
        batch_size = args.batch_size
    else:
        profile_path = Path("outputs/profile/inference/vjepa2/profile_data.json")
        if profile_path.exists():
            _profile = json.load(open(profile_path))
            if "optimal_bs" in _profile:
                batch_size = _profile["optimal_bs"]
                print(f"Batch size: {batch_size} (from inference/vjepa2 profiler)")
        if batch_size is None:
            batch_size = get_pipeline_config()["gpu"].get("inference_vjepa_bs",
                         compute_batch_sizes(gpu_vram_gb=args.gpu_mem)["vjepa"])
    if args.SANITY:
        batch_size = min(batch_size, 2)

    tmp_base = output_dir / "tmp_m05b"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)

    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_producer_shuffled_vjepa,
        args=(processor, batch_size, tmp_dir, q, stop_event,
              clip_limit, subset_keys, processed_keys, VJEPA_FRAMES_PER_CLIP,
              getattr(args, 'local_data', None)),
        daemon=True,
    )
    producer.start()

    start_time = time.time()
    last_window_count = len(all_embeddings)
    last_window_time = start_time
    pbar = tqdm(total=clip_limit, initial=resume_count,
                desc="m05b vjepa_shuffled", unit="clip")
    try:
        while True:
            try:
                msg_type, batched_pixels, batch_keys = q.get(timeout=60)
            except queue.Empty:
                print("\nProducer timeout (10 min). Saving checkpoint...")
                break

            if msg_type == "done":
                break

            emb = get_batch_embeddings(model, batched_pixels, device)

            for e, k in zip(emb, batch_keys):
                all_embeddings.append(e)
                all_keys.append(k)
                processed_keys.add(k)

            pbar.update(len(batch_keys))
            now = time.time()
            window_clips = len(all_embeddings) - last_window_count
            window_time = now - last_window_time
            throughput = window_clips / window_time if window_time > 0 else 0
            last_window_count = len(all_embeddings)
            last_window_time = now
            pbar.set_postfix({"rate": f"{throughput:.1f}/s"})

            if len(all_embeddings) % CHECKPOINT_EVERY < batch_size:
                save_checkpoint(list(all_embeddings), list(all_keys), checkpoint_file)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        stop_event.set()
    finally:
        pbar.close()
        stop_event.set()
        save_checkpoint(all_embeddings, all_keys, checkpoint_file)
        producer.join(timeout=10)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            tmp_base.rmdir()
        except OSError:
            pass

    _finalize(all_embeddings, all_keys, files, checkpoint_file)
    return np.stack(all_embeddings).astype(np.float32), all_keys


# ── Shared Finalize ──────────────────────────────────────────────────

def _finalize(all_embeddings: list, all_keys: list, files: dict, checkpoint_file: Path):
    """Save final output, clean checkpoint."""
    if not all_embeddings:
        print("ERROR: No embeddings collected.")
        sys.exit(1)

    embeddings = np.stack(all_embeddings).astype(np.float32)
    files["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(files["embeddings"], embeddings)
    np.save(files["paths"], np.array(all_keys, dtype=object))

    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print(f"\nSaved: {files['embeddings']} ({embeddings.shape})")
    print(f"Saved: {files['paths']} ({len(all_keys)} keys)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    cleanup_temp()
    parser = argparse.ArgumentParser(
        description="Generate baseline embeddings: Random, DINOv2, CLIP, Shuffled V-JEPA")
    ALL_BASELINES = ["random", "dinov2", "clip", "vjepa_shuffled"]
    parser.add_argument("--encoder", required=True,
                        choices=ALL_BASELINES + ["all"],
                        help="Baseline encoder to run ('all' = run all 4 sequentially)")
    parser.add_argument("--SANITY", action="store_true", help="Process 5 clips only")
    parser.add_argument("--POC", action="store_true", help="10K subset")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (auto-computed if omitted)")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    add_gpu_mem_arg(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    ensure_local_data(args)
    encoders_to_run = ALL_BASELINES if args.encoder == "all" else [args.encoder]

    if args.encoder == "all":
        print(f"{'='*60}")
        print(f"RUNNING ALL 4 BASELINES SEQUENTIALLY")
        print(f"Order: {' → '.join(encoders_to_run)}")
        print(f"{'='*60}\n")

    for enc_idx, encoder in enumerate(encoders_to_run):
        if len(encoders_to_run) > 1:
            print(f"\n{'#'*60}")
            print(f"# [{enc_idx+1}/{len(encoders_to_run)}] {encoder}")
            print(f"{'#'*60}\n")

        _run_single_encoder(encoder, args)

        # Cleanup CUDA between encoders to prevent cross-encoder fragmentation
        if len(encoders_to_run) > 1:
            cuda_cleanup()

    if args.encoder == "all":
        print(f"\n{'='*60}")
        print(f"ALL 4 BASELINES COMPLETE")
        print(f"Next: python -u src/m06_faiss_metrics.py --encoder <enc> --FULL --subset data/subset_10k.json")
        print(f"{'='*60}")


def _run_single_encoder(encoder: str, args):
    """Run a single baseline encoder end-to-end."""
    info = ENCODER_REGISTRY[encoder]
    output_dir = get_output_dir(args.subset, sanity=args.SANITY)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = get_encoder_files(encoder, output_dir)

    print(f"{'='*60}")
    print(f"BASELINE: {encoder}")
    print(f"Model:    {info['model_id'] or 'N/A (synthetic)'}")
    print(f"Dim:      {info['dim']}")
    print(f"Type:     {info['type']}")
    print(f"Output:   {files['embeddings']}")
    print(f"{'='*60}")

    # Output-exists guard
    from utils.output_guard import verify_or_skip
    if verify_or_skip(output_dir, {
        "embeddings": files["embeddings"],
        "paths": files["paths"],
    }, label=f"m05b {encoder}"):
        return

    # Determine clip limit + subset
    subset_keys = load_subset(args.subset) if args.subset else set()
    if args.SANITY:
        clip_limit = get_sanity_clip_limit("embed")
    elif subset_keys:
        clip_limit = len(subset_keys)
    else:
        clip_limit = get_total_clips(local_data=getattr(args, 'local_data', None))
        if clip_limit == 0:
            print("FATAL: Cannot determine clip count. Use --subset or --local-data with manifest.json")
            sys.exit(1)

    # Random is CPU-safe, others need GPU
    if encoder == "random":
        # Load reference clip keys from V-JEPA embeddings.paths.npy
        vjepa_files = get_encoder_files("vjepa", output_dir)
        if vjepa_files["paths"].exists():
            ref_keys = list(np.load(vjepa_files["paths"], allow_pickle=True))
            print(f"Using {len(ref_keys):,} clip keys from V-JEPA embeddings.paths.npy")
        else:
            print(f"WARNING: {vjepa_files['paths']} not found. Generating {clip_limit} keys.")
            ref_keys = [f"clip_{i:06d}" for i in range(clip_limit)]

        if args.SANITY:
            ref_keys = ref_keys[:get_sanity_clip_limit("embed")]

        wb_run = init_wandb("m05b", f"random_{'SANITY' if args.SANITY else 'POC' if args.POC else 'FULL'}",
                            config=vars(args), enabled=not args.no_wandb)
        embeddings, keys = generate_random(ref_keys, output_dir, args)
        log_metrics(wb_run, {"total_clips": len(keys), "embedding_dim": embeddings.shape[1]})
        log_artifact(wb_run, f"embeddings_{encoder}", str(files["embeddings"]))
        finish_wandb(wb_run)
    else:
        check_gpu()
        mode = f"{encoder}_{'SANITY' if args.SANITY else 'POC' if args.POC else 'FULL'}"
        wb_run = init_wandb("m05b", mode, config=vars(args), enabled=not args.no_wandb)

        if encoder == "dinov2":
            embeddings, keys = generate_dinov2(output_dir, args, clip_limit, subset_keys)
        elif encoder == "clip":
            embeddings, keys = generate_clip(output_dir, args, clip_limit, subset_keys)
        elif encoder == "vjepa_shuffled":
            embeddings, keys = generate_shuffled_vjepa(output_dir, args, clip_limit, subset_keys)

        log_metrics(wb_run, {"total_clips": len(keys), "embedding_dim": embeddings.shape[1]})
        log_artifact(wb_run, f"embeddings_{encoder}", str(files["embeddings"]))
        finish_wandb(wb_run)

    print(f"\n=== BASELINE COMPLETE: {encoder} ===")
    print(f"Clips:     {len(keys):,}")
    print(f"Dim:       {embeddings.shape[1]}")
    print(f"Next step: python -u src/m06_faiss_metrics.py --encoder {encoder} --FULL --subset data/subset_10k.json")

    # Force exit: torch.compile + CUDA atexit cleanup deadlocks on futex_wait_queue
    # (hangs indefinitely at 0% GPU after all output files are saved)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
