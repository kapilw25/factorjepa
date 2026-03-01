"""
Generate V-JEPA 2 embeddings via HF WebDataset streaming + producer-consumer GPU inference.
GPU-only (Nvidia CUDA required, no CPU fallback). Streams from HF — no local clips needed.

USAGE:
    python -u src/m05_vjepa_embed.py --SANITY 2>&1 | tee logs/m05_vjepa_embed_sanity.log
    python -u src/m05_vjepa_embed.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m05_vjepa_embed_poc.log
    python -u src/m05_vjepa_embed.py --FULL 2>&1 | tee logs/m05_vjepa_embed_full.log
"""
import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    EMBEDDINGS_FILE, VJEPA_MODEL_ID, HF_DATASET_REPO,
    VJEPA_FRAMES_PER_CLIP, VJEPA_EMBEDDING_DIM,
    check_gpu, DEFAULT_BATCH_SIZE,
    check_output_exists, load_subset, add_subset_arg, get_output_dir,
)
from utils.gpu_batch import compute_batch_sizes, add_gpu_mem_arg
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, log_artifact, finish_wandb

try:
    import torch
    from transformers import AutoModel, AutoVideoProcessor
    from datasets import load_dataset
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install torch transformers datasets")
    sys.exit(1)

# Try torchcodec first (faster), fallback to av
USE_TORCHCODEC = False
try:
    from torchcodec.decoders import VideoDecoder
    USE_TORCHCODEC = True
    print("Using torchcodec (fast video decoding)")
except (ImportError, RuntimeError, OSError):
    USE_TORCHCODEC = False

if not USE_TORCHCODEC:
    try:
        import av
        print("Using PyAV for video decoding")
    except ImportError:
        print("ERROR: Neither torchcodec nor av available")
        print("Install with: pip install av")
        sys.exit(1)

# Constants
DEDUPE_THRESHOLD = 0.95
MAX_STREAM_RETRIES = 5
PREFETCH_QUEUE_SIZE = 2
CHECKPOINT_EVERY = 500
TOTAL_CLIPS = 115_687
DECODE_WORKERS = 4
ENGINE_RESTART_EVERY = 10_000


# ── HF Streaming Helpers ──────────────────────────────────────────────

def get_clip_key(example: dict) -> str:
    """Reconstruct clip key from HF WebDataset example metadata."""
    meta = example.get("json", {})
    if isinstance(meta, (bytes, str)):
        meta = json.loads(meta) if meta else {}
    section = meta.get("section", "")
    video_id = meta.get("video_id", "")
    source_file = meta.get("source_file", "")
    return f"{section}/{video_id}/{source_file}"


def _create_stream(skip_count: int = 0):
    """Create HF streaming dataset, optionally skipping already-processed examples."""
    ds = load_dataset(HF_DATASET_REPO, split="train", streaming=True)
    ds = ds.decode(False)
    if skip_count > 0:
        ds = ds.skip(skip_count)
    return ds


# ── Video Decoding (from bytes, not local paths) ──────────────────────

def _load_torchcodec(video_path: str, num_frames: int) -> torch.Tensor:
    """Load video using torchcodec (faster, GPU-accelerated decode)."""
    decoder = VideoDecoder(video_path)
    metadata = decoder.metadata
    total_frames = metadata.num_frames if hasattr(metadata, 'num_frames') else 200
    frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    frames = decoder.get_frames_at(indices=frame_indices.tolist())
    video_tensor = frames.data  # (T, C, H, W)
    if video_tensor.shape[0] < num_frames:
        pad_size = num_frames - video_tensor.shape[0]
        last_frame = video_tensor[-1:].repeat(pad_size, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, last_frame], dim=0)
    return video_tensor[:num_frames]


def _load_av(video_path: str, num_frames: int) -> torch.Tensor:
    """Load video using PyAV (fallback)."""
    container = av.open(video_path)
    stream = container.streams.video[0]
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


def decode_video_bytes(mp4_bytes: bytes, tmp_dir: str, key: str,
                       num_frames: int):
    """Write mp4 bytes to temp file, decode frames, delete temp file. Returns tensor or None."""
    safe_key = key.replace("/", "_").replace("\\", "_")
    tmp_path = os.path.join(tmp_dir, f"{safe_key}.mp4")
    try:
        with open(tmp_path, "wb") as f:
            f.write(mp4_bytes)
        if USE_TORCHCODEC:
            return _load_torchcodec(tmp_path, num_frames)
        else:
            return _load_av(tmp_path, num_frames)
    except Exception as e:
        print(f"  WARN: decode failed ({key}): {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Checkpoint ─────────────────────────────────────────────────────────

def save_checkpoint(embeddings_list: list, keys_list: list, checkpoint_file: Path):
    """Atomic checkpoint save: embeddings + keys to .npz."""
    if not embeddings_list:
        return
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    # np.savez auto-appends .npz, so use _tmp.npz to get a predictable path
    tmp_file = checkpoint_file.with_name(checkpoint_file.stem + "_tmp.npz")
    np.savez(tmp_file,
             embeddings=np.stack(embeddings_list).astype(np.float32),
             keys=np.array(keys_list, dtype=object))
    os.replace(tmp_file, checkpoint_file)


def load_checkpoint(checkpoint_file: Path) -> tuple:
    """Load checkpoint. Returns (embeddings_list, keys_list, count)."""
    if not checkpoint_file.exists():
        return [], [], 0
    try:
        data = np.load(checkpoint_file, allow_pickle=True)
        emb_list = list(data["embeddings"])
        keys_list = list(data["keys"])
        print(f"Checkpoint loaded: {len(emb_list):,} embeddings from {checkpoint_file.name}")
        return emb_list, keys_list, len(emb_list)
    except Exception as e:
        print(f"  WARN: checkpoint corrupt ({e}), starting fresh")
        return [], [], 0


# ── Producer Thread (HF Stream → Decode → Queue) ──────────────────────

def _process_and_enqueue(processor, batch_tensors, batch_keys, q):
    """Process video tensors through processor and enqueue for GPU inference."""
    processed_list = []
    for vt in batch_tensors:
        processed = processor(vt, return_tensors="pt")
        processed_list.append(processed["pixel_values_videos"])
    batched_pixels = torch.cat(processed_list, dim=0)
    q.put(("batch", batched_pixels, batch_keys[:]))


def _producer_thread(processor, batch_size: int, tmp_dir: str,
                     q: queue.Queue, stop_event: threading.Event,
                     clip_limit: int, subset_keys: set, num_frames: int,
                     processed_keys: set):
    """Stream from HF, filter by subset, decode video tensors in parallel, enqueue."""
    produced = 0
    skipped = 0
    retries = 0

    while produced < clip_limit and not stop_event.is_set():
        try:
            ds = _create_stream(0)
            # Collect a batch of (bytes, key) pairs, then decode in parallel
            pending_bytes = []
            pending_keys = []

            for example in ds:
                if stop_event.is_set():
                    break

                clip_key = get_clip_key(example)

                # Subset filtering
                if subset_keys and clip_key not in subset_keys:
                    skipped += 1
                    continue

                # Skip already-processed clips (checkpoint resume)
                if clip_key in processed_keys:
                    continue

                # Extract video bytes
                mp4_data = example.get("mp4", b"")
                mp4_bytes = mp4_data["bytes"] if isinstance(mp4_data, dict) else mp4_data
                if not mp4_bytes:
                    continue

                pending_bytes.append(mp4_bytes)
                pending_keys.append(clip_key)

                if len(pending_bytes) >= batch_size:
                    # Parallel decode batch
                    with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                        futures = [
                            pool.submit(decode_video_bytes, b, tmp_dir, k, num_frames)
                            for b, k in zip(pending_bytes, pending_keys)
                        ]
                        results = [(f.result(), k) for f, k in zip(futures, pending_keys)]

                    batch_tensors = [t for t, k in results if t is not None]
                    batch_keys = [k for t, k in results if t is not None]

                    if batch_tensors:
                        _process_and_enqueue(processor, batch_tensors, batch_keys, q)
                        produced += len(batch_tensors)

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

                batch_tensors = [t for t, k in results if t is not None]
                batch_keys = [k for t, k in results if t is not None]

                if batch_tensors:
                    _process_and_enqueue(processor, batch_tensors, batch_keys, q)
                    produced += len(batch_tensors)

            break  # stream exhausted normally

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


# ── Deduplication ──────────────────────────────────────────────────────

def deduplicate_embeddings(embeddings: np.ndarray, clip_keys: list, threshold: float = 0.95) -> tuple:
    """Remove near-duplicate clips based on cosine similarity (chunked to avoid OOM)."""
    print(f"\n=== Deduplicating clips (cosine sim > {threshold}) ===")
    n = len(embeddings)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms

    keep_mask = np.ones(n, dtype=bool)
    CHUNK = 2048  # ~2048 x 115K x 4 bytes = ~900 MB per chunk (fits in VRAM)
    use_gpu = torch.cuda.is_available()

    for i_start in range(0, n, CHUNK):
        i_end = min(i_start + CHUNK, n)
        # Only process rows still marked as kept
        active_rows = [i for i in range(i_start, i_end) if keep_mask[i]]
        if not active_rows:
            continue

        if use_gpu:
            chunk = torch.from_numpy(normalized[active_rows]).cuda()
        else:
            chunk = normalized[active_rows]

        # Compare against all columns AFTER the chunk start (avoid double-counting)
        for j_start in range(i_start, n, CHUNK):
            j_end = min(j_start + CHUNK, n)
            if use_gpu:
                target = torch.from_numpy(normalized[j_start:j_end]).cuda()
                sim = torch.mm(chunk, target.T).cpu().numpy()
            else:
                sim = normalized[active_rows] @ normalized[j_start:j_end].T

            for ci, i in enumerate(active_rows):
                if not keep_mask[i]:
                    continue
                # Only mark duplicates with index > i
                col_start = max(0, i + 1 - j_start)
                if col_start >= sim.shape[1]:
                    continue
                dupes = np.where(sim[ci, col_start:] > threshold)[0] + j_start + col_start
                keep_mask[dupes] = False

    keep_indices = np.where(keep_mask)[0]
    deduped_embeddings = embeddings[keep_indices]
    deduped_keys = [clip_keys[i] for i in keep_indices]
    num_removed = n - len(keep_indices)

    print(f"Original clips: {n}")
    print(f"Removed duplicates: {num_removed}")
    print(f"Remaining clips: {len(deduped_keys)}")

    return deduped_embeddings, deduped_keys, num_removed


# ── GPU Inference ──────────────────────────────────────────────────────

def get_batch_embeddings(model, batched_pixels: torch.Tensor, device: str) -> np.ndarray:
    """Get V-JEPA 2 embeddings for a batch of processed videos."""
    pixel_values = batched_pixels.to(device)
    with torch.no_grad():
        outputs = model(pixel_values_videos=pixel_values, skip_predictor=True)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


# ═════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR / WORKER (subprocess pattern for HF stream resilience)
# ═════════════════════════════════════════════════════════════════════════

def orchestrator_main(args):
    """Spawn worker subprocesses every ENGINE_RESTART_EVERY clips.

    Each worker gets fresh HF connections + GPU state. On stream stall
    (10-min producer timeout), worker exits, orchestrator respawns from checkpoint.
    """
    output_dir = get_output_dir(args.subset)
    embeddings_file = output_dir / "embeddings.npy" if args.subset else EMBEDDINGS_FILE
    checkpoint_file = output_dir / ".m05_checkpoint.npz"

    print(f"Output: {embeddings_file}")
    if args.subset:
        print(f"[POC] Subset: {args.subset}")

    # Check if embeddings already exist
    if embeddings_file.exists():
        if not check_output_exists([embeddings_file, embeddings_file.with_suffix('.paths.npy')], "embeddings"):
            print("Using cached embeddings.")
            return

    # Determine total clips
    subset_keys = load_subset(args.subset) if args.subset else set()
    if args.SANITY:
        total_clips = 5
    elif subset_keys:
        total_clips = len(subset_keys)
    else:
        total_clips = TOTAL_CLIPS

    print(f"Clip limit: {total_clips:,}")
    print(f"Streaming from: {HF_DATASET_REPO}")

    # Load checkpoint to determine progress
    _, _, skip_count = load_checkpoint(checkpoint_file)
    if skip_count >= total_clips:
        print(f"All clips processed ({skip_count:,}/{total_clips:,}). Running post-processing...")
    else:
        if skip_count > 0:
            print(f"Resuming from checkpoint: {skip_count:,}/{total_clips:,} clips")

        segment_idx = 0
        while skip_count < total_clips:
            segment_size = min(ENGINE_RESTART_EVERY, total_clips - skip_count)
            segment_idx += 1
            print(f"\n{'='*60}")
            print(f"WORKER {segment_idx}: embeddings {skip_count:,} → {skip_count + segment_size:,}")
            print(f"{'='*60}")

            cmd = [
                sys.executable, "-u", os.path.abspath(__file__),
                "--_worker",
                "--start-from", str(skip_count),
                "--process-count", str(segment_size),
            ]
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(args.batch_size)])
            if args.SANITY:
                cmd.append("--SANITY")
            if args.FULL:
                cmd.append("--FULL")
            if args.subset:
                cmd.extend(["--subset", args.subset])
            if args.no_wandb:
                cmd.append("--no-wandb")
            if args.gpu_mem is not None:
                cmd.extend(["--gpu-mem", str(args.gpu_mem)])
            if args.model != VJEPA_MODEL_ID:
                cmd.extend(["--model", args.model])

            result = subprocess.run(cmd)

            _, _, new_count = load_checkpoint(checkpoint_file)
            if new_count > skip_count:
                skip_count = new_count
                print(f"Worker done. Progress: {skip_count:,}/{total_clips:,}")
            elif result.returncode != 0:
                print(f"Worker failed (exit {result.returncode}). Resume with same command.")
                break
            else:
                skip_count = total_clips

    # ── Post-processing: dedupe + save final output ──
    all_embeddings, all_keys, final_count = load_checkpoint(checkpoint_file)
    if not all_embeddings:
        print("ERROR: No embeddings collected.")
        sys.exit(1)

    embeddings = np.stack(all_embeddings).astype(np.float32)
    clip_keys = all_keys

    print(f"\n=== Processing Stats ===")
    print(f"Total clips:     {len(clip_keys):,}")
    print(f"Embedding shape: {embeddings.shape}")

    if not args.no_dedupe:
        embeddings, clip_keys, num_removed = deduplicate_embeddings(
            embeddings, clip_keys, args.threshold
        )

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, embeddings)
    np.save(embeddings_file.with_suffix('.paths.npy'), np.array(clip_keys, dtype=object))

    if checkpoint_file.exists():
        checkpoint_file.unlink()

    print(f"\n=== EMBEDDING COMPLETE ===")
    print(f"Saved: {embeddings_file}")
    print(f"Shape: {embeddings.shape}")
    print(f"Unique clips: {len(clip_keys)}")

    wb_run = init_wandb("m05", "COMPLETE", config=vars(args),
                        enabled=not args.no_wandb)
    log_metrics(wb_run, {
        "total_clips": len(clip_keys),
        "embedding_dim": embeddings.shape[1],
    })
    log_artifact(wb_run, "embeddings", str(embeddings_file))
    log_artifact(wb_run, "paths", str(embeddings_file.with_suffix('.paths.npy')))
    finish_wandb(wb_run)


def worker_main(args):
    """Worker subprocess: load V-JEPA, process segment, save checkpoint, exit."""
    check_gpu()
    device = "cuda"

    if args.batch_size is None:
        batch_sizes = compute_batch_sizes(gpu_vram_gb=args.gpu_mem)
        args.batch_size = batch_sizes["vjepa"]
    print(f"Batch size: {args.batch_size}")

    mode = "SANITY" if args.SANITY else ("POC" if args.subset else "FULL")
    wb_run = init_wandb("m05", mode,
                        config={"start_from": args.start_from,
                                "process_count": args.process_count},
                        enabled=not args.no_wandb)

    output_dir = get_output_dir(args.subset)
    checkpoint_file = output_dir / ".m05_checkpoint.npz"
    subset_keys = load_subset(args.subset) if args.subset else set()

    # Load checkpoint for resume (processed_keys used by producer to skip done clips)
    all_embeddings, all_keys, resume_count = load_checkpoint(checkpoint_file)
    processed_keys = set(all_keys)

    clip_limit = args.process_count
    if clip_limit <= 0:
        print("No clips to process.")
        finish_wandb(wb_run)
        return

    if args.SANITY:
        args.batch_size = min(args.batch_size, 2)

    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        processor = AutoVideoProcessor.from_pretrained(args.model)
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("FATAL: flash-attn not installed.")
            print("V-JEPA 2 ViT-G requires Flash Attention 2 for memory-efficient inference.")
            print("")
            print("Install via setup_env_uv.sh --gpu (downloads pre-built wheel), or manually:")
            print("  pip install flash-attn --no-build-isolation")
            sys.exit(1)
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        model.eval()
        print("Applying torch.compile (first batch will be slow due to compilation)...")
        model = torch.compile(model)
        print(f"Model loaded on {device} (dtype: {next(model.parameters()).dtype})")
    except Exception as e:
        print(f"FATAL: Model load failed: {e}")
        sys.exit(1)

    # Producer-consumer setup
    print(f"\n=== Streaming Config ===")
    print(f"batch_size:    {args.batch_size}")
    print(f"video_decoder: {'torchcodec (fast)' if USE_TORCHCODEC else 'PyAV'}")
    print(f"prefetch:      {PREFETCH_QUEUE_SIZE} batches")
    print(f"checkpoint:    every {CHECKPOINT_EVERY} clips")

    tmp_base = output_dir / "tmp_m05"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)

    q = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_producer_thread,
        args=(processor, args.batch_size, tmp_dir, q, stop_event,
              clip_limit, subset_keys, VJEPA_FRAMES_PER_CLIP, processed_keys),
        daemon=True,
    )
    producer.start()

    start_time = time.time()
    failed_count = 0
    checkpoint_thread = None

    try:
        while True:
            try:
                msg_type, batched_pixels, batch_keys = q.get(timeout=600)
            except queue.Empty:
                print("\nProducer timeout (10 min). Saving checkpoint...")
                break

            if msg_type == "done":
                break

            embeddings = get_batch_embeddings(model, batched_pixels, device)

            for emb, key in zip(embeddings, batch_keys):
                all_embeddings.append(emb)
                all_keys.append(key)
                processed_keys.add(key)

            elapsed = time.time() - start_time
            clips_this_run = len(all_embeddings) - resume_count
            total_target = clip_limit + resume_count
            throughput = clips_this_run / elapsed if elapsed > 0 else 0
            print(f"  [{len(all_embeddings):,}/{total_target:,}] "
                  f"{throughput:.1f} clips/s | failed={failed_count}")
            log_metrics(wb_run, {
                "clips_processed": len(all_embeddings),
                "throughput_clips_per_s": throughput,
                "failed": failed_count,
            })

            if len(all_embeddings) % CHECKPOINT_EVERY < args.batch_size:
                if checkpoint_thread and checkpoint_thread.is_alive():
                    checkpoint_thread.join()
                emb_snapshot = list(all_embeddings)
                keys_snapshot = list(all_keys)
                checkpoint_thread = threading.Thread(
                    target=save_checkpoint,
                    args=(emb_snapshot, keys_snapshot, checkpoint_file),
                    daemon=True,
                )
                checkpoint_thread.start()

    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        stop_event.set()
    finally:
        stop_event.set()
        if checkpoint_thread and checkpoint_thread.is_alive():
            checkpoint_thread.join(timeout=30)
        save_checkpoint(all_embeddings, all_keys, checkpoint_file)
        producer.join(timeout=10)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        try:
            tmp_base.rmdir()
        except OSError:
            pass

    clips_this_run = len(all_embeddings) - resume_count
    if clips_this_run > 0:
        elapsed = time.time() - start_time
        print(f"\nSegment done: {clips_this_run:,} clips in {elapsed:.0f}s "
              f"({clips_this_run/elapsed:.2f} clips/s)")

    finish_wandb(wb_run)
    print("Worker exiting (GPU memory will be released).")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate V-JEPA 2 embeddings (GPU-only, HF WebDataset streaming)")
    parser.add_argument("--SANITY", action="store_true", help="Process 5 clips only")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    parser.add_argument("--model", type=str, default=VJEPA_MODEL_ID, help="Model ID")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size (auto-computed if omitted)")
    parser.add_argument("--no-dedupe", action="store_true", help="Skip deduplication")
    parser.add_argument("--threshold", type=float, default=DEDUPE_THRESHOLD,
                        help="Dedupe threshold")
    add_subset_arg(parser)
    add_wandb_args(parser)
    add_gpu_mem_arg(parser)

    # Internal worker args (spawned by orchestrator)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start-from", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--process-count", type=int, default=ENGINE_RESTART_EVERY,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Worker mode (spawned by orchestrator)
    if args._worker:
        worker_main(args)
        return

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    orchestrator_main(args)


if __name__ == "__main__":
    main()
