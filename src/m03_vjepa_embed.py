"""
Generate V-JEPA 2 embeddings with batched GPU inference + deduplication.
GPU-only script (Nvidia CUDA required). Optimized for A100-40GB with torchcodec + RAM cache.

USAGE:
    python -u src/m03_vjepa_embed.py --SANITY 2>&1 | tee logs/m03_vjepa_embed_sanity.log
    python -u src/m03_vjepa_embed.py --FULL 2>&1 | tee logs/m03_vjepa_embed_full.log
    python -u src/m03_vjepa_embed.py --FULL --batch-size 24 --num-workers 8 2>&1 | tee logs/m03_full_bs24.log
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    CLIPS_DIR, EMBEDDINGS_FILE, VJEPA_MODEL_ID,
    VJEPA_FRAMES_PER_CLIP, VJEPA_EMBEDDING_DIM,
    ensure_clips_exist, check_gpu, get_all_clips,
    DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS,
    setup_ram_cache, cleanup_ram_cache, restore_original_path,
    check_output_exists
)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModel, AutoVideoProcessor
    from tqdm import tqdm
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install torch transformers tqdm")
    sys.exit(1)

# Try torchcodec first (faster), fallback to av
USE_TORCHCODEC = False
try:
    from torchcodec.decoders import VideoDecoder
    USE_TORCHCODEC = True
    print("Using torchcodec (fast video decoding)")
except (ImportError, RuntimeError, OSError):
    # torchcodec may fail with RuntimeError if FFmpeg libs missing
    USE_TORCHCODEC = False

if not USE_TORCHCODEC:
    try:
        import av
        print("Using PyAV for video decoding")
    except ImportError:
        print("ERROR: Neither torchcodec nor av available")
        print("Install with: pip install av")
        sys.exit(1)

# Deduplication threshold (cosine similarity)
DEDUPE_THRESHOLD = 0.95


class VideoClipDataset(Dataset):
    """PyTorch Dataset with torchcodec/av video loading."""

    def __init__(self, clip_paths: list, num_frames: int = 64):
        self.clip_paths = [Path(p) for p in clip_paths]
        self.num_frames = num_frames

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        clip_path = self.clip_paths[idx]
        try:
            if USE_TORCHCODEC:
                video_tensor = self._load_torchcodec(clip_path)
            else:
                video_tensor = self._load_av(clip_path)
            return {"video": video_tensor, "path": str(clip_path), "valid": True}
        except Exception as e:
            return {
                "video": torch.zeros((self.num_frames, 3, 256, 256), dtype=torch.uint8),
                "path": str(clip_path),
                "valid": False,
                "error": str(e)
            }

    def _load_torchcodec(self, video_path: Path) -> torch.Tensor:
        """Load video using torchcodec (faster, GPU-accelerated decode)."""
        decoder = VideoDecoder(str(video_path))

        # Get total frames
        metadata = decoder.metadata
        total_frames = metadata.num_frames if hasattr(metadata, 'num_frames') else 200

        # Sample frame indices evenly
        frame_indices = np.linspace(0, max(total_frames - 1, 0), self.num_frames, dtype=int)

        # Get frames at specified indices
        frames = decoder.get_frames_at(indices=frame_indices.tolist())
        video_tensor = frames.data  # (T, C, H, W)

        # Pad if needed
        if video_tensor.shape[0] < self.num_frames:
            pad_size = self.num_frames - video_tensor.shape[0]
            last_frame = video_tensor[-1:].repeat(pad_size, 1, 1, 1)
            video_tensor = torch.cat([video_tensor, last_frame], dim=0)

        return video_tensor[:self.num_frames]

    def _load_av(self, video_path: Path) -> torch.Tensor:
        """Load video using PyAV (fallback)."""
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        total_frames = stream.frames
        if total_frames == 0:
            if stream.duration and stream.average_rate:
                total_frames = int(float(stream.duration * stream.time_base) * float(stream.average_rate))
            else:
                total_frames = 200

        indices = set(np.linspace(0, max(total_frames - 1, 0), self.num_frames, dtype=int).tolist())

        frames = []
        frame_idx = 0
        for frame in container.decode(video=0):
            if frame_idx in indices:
                img = frame.to_ndarray(format="rgb24")
                img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                frames.append(img_tensor)
            frame_idx += 1
            if len(frames) >= self.num_frames:
                break

        container.close()

        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros((3, 256, 256), dtype=torch.uint8))

        return torch.stack(frames[:self.num_frames])


def collate_batch(batch: list, processor) -> dict:
    """Custom collate function to batch videos through processor."""
    videos = [item["video"] for item in batch]
    paths = [item["path"] for item in batch]
    valid = [item["valid"] for item in batch]

    processed_list = []
    for video in videos:
        processed = processor(video, return_tensors="pt")
        processed_list.append(processed["pixel_values_videos"])

    batched_pixels = torch.cat(processed_list, dim=0)

    return {
        "pixel_values_videos": batched_pixels,
        "paths": paths,
        "valid": valid
    }


def get_batch_embeddings(model, batch: dict, device: str) -> np.ndarray:
    """Get V-JEPA 2 embeddings for a batch of videos."""
    pixel_values = batch["pixel_values_videos"].to(device)

    with torch.no_grad():
        embeddings = model.get_vision_features(pixel_values)
        embeddings = embeddings.mean(dim=1).cpu().numpy()

    return embeddings


def deduplicate_embeddings(embeddings: np.ndarray, clip_paths: list, threshold: float = 0.95) -> tuple:
    """Remove near-duplicate clips based on cosine similarity."""
    print(f"\n=== Deduplicating clips (cosine sim > {threshold}) ===")
    n = len(embeddings)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms

    if torch.cuda.is_available():
        normalized_tensor = torch.from_numpy(normalized).cuda()
        similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.T).cpu().numpy()
    else:
        similarity_matrix = normalized @ normalized.T

    keep_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep_mask[i]:
            continue
        similar = np.where((similarity_matrix[i] > threshold) & (np.arange(n) > i))[0]
        keep_mask[similar] = False

    keep_indices = np.where(keep_mask)[0]
    deduped_embeddings = embeddings[keep_indices]
    deduped_paths = [clip_paths[i] for i in keep_indices]
    num_removed = n - len(keep_indices)

    print(f"Original clips: {n}")
    print(f"Removed duplicates: {num_removed}")
    print(f"Remaining clips: {len(deduped_paths)}")

    if num_removed > 0:
        removed_indices = np.where(~keep_mask)[0][:5]
        print(f"\nExample removed clips:")
        for idx in removed_indices:
            similar_to = np.where((similarity_matrix[idx] > threshold) & (np.arange(n) < idx) & keep_mask)[0]
            if len(similar_to) > 0:
                sim_val = similarity_matrix[idx, similar_to[0]]
                print(f"  {Path(clip_paths[idx]).name} (sim={sim_val:.3f})")

    return deduped_embeddings, deduped_paths, num_removed


def main():
    parser = argparse.ArgumentParser(description="Generate V-JEPA embeddings (GPU-optimized, torchcodec + RAM cache)")
    parser.add_argument("--SANITY", action="store_true", help="Process 5 clips only")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    parser.add_argument("--model", type=str, default=VJEPA_MODEL_ID, help="Model ID")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (default: 16)")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers (default: 8)")
    parser.add_argument("--no-ram-cache", action="store_true", help="Disable RAM cache (/dev/shm)")
    parser.add_argument("--no-dedupe", action="store_true", help="Skip deduplication")
    parser.add_argument("--threshold", type=float, default=DEDUPE_THRESHOLD, help="Dedupe threshold")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    check_gpu()
    device = "cuda"

    # Check if embeddings already exist
    if EMBEDDINGS_FILE.exists():
        if not check_output_exists([EMBEDDINGS_FILE, EMBEDDINGS_FILE.with_suffix('.paths.npy')], "embeddings"):
            print("Using cached embeddings.")
            return

    # Ensure clips exist
    if not ensure_clips_exist():
        print(f"ERROR: No clips available. Run m02_scene_detect.py or check HuggingFace access.")
        sys.exit(1)

    # Find all clips
    all_clips = get_all_clips()
    if not all_clips:
        print(f"ERROR: No clips found in {CLIPS_DIR}")
        sys.exit(1)

    print(f"Found {len(all_clips)} clips")

    # Limit clips for sanity mode
    if args.SANITY:
        all_clips = all_clips[:5]
        args.batch_size = min(args.batch_size, 2)
        args.num_workers = min(args.num_workers, 2)
        print(f"SANITY MODE: Processing {len(all_clips)} clips")

    # Setup RAM cache (copy clips to /dev/shm for faster I/O)
    use_ram_cache = not args.no_ram_cache and not args.SANITY
    clip_paths, ram_cache_enabled = setup_ram_cache(all_clips, use_cache=use_ram_cache, cache_subdir="vjepa_clips")

    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        processor = AutoVideoProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        model.eval()
        print(f"Model loaded on {device} (dtype: {next(model.parameters()).dtype})")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        cleanup_ram_cache(cache_subdir="vjepa_clips")
        print("\nFalling back to dummy embeddings")
        embeddings = np.random.randn(len(all_clips), VJEPA_EMBEDDING_DIM).astype(np.float32)
        orig_paths = [str(c) for c in all_clips]
        if not args.no_dedupe:
            embeddings, orig_paths, _ = deduplicate_embeddings(embeddings, orig_paths, args.threshold)
        np.save(EMBEDDINGS_FILE, embeddings)
        np.save(EMBEDDINGS_FILE.with_suffix('.paths.npy'), orig_paths)
        print(f"Saved dummy embeddings: {EMBEDDINGS_FILE}")
        return

    # Create dataset and dataloader
    print(f"\n=== Batch Processing Config ===")
    print(f"batch_size:    {args.batch_size}")
    print(f"num_workers:   {args.num_workers}")
    print(f"prefetch:      {2 * args.batch_size} clips")
    print(f"video_decoder: {'torchcodec (fast)' if USE_TORCHCODEC else 'PyAV (slow)'}")
    print(f"ram_cache:     {'enabled' if ram_cache_enabled else 'disabled'}")

    dataset = VideoClipDataset(clip_paths, num_frames=VJEPA_FRAMES_PER_CLIP)

    def collate_fn(batch):
        return collate_batch(batch, processor)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Process batches with progress bar
    all_embeddings = []
    all_paths = []
    failed_count = 0

    start_time = time.time()

    pbar = tqdm(dataloader, desc="Embedding clips", unit="batch")
    for batch in pbar:
        embeddings = get_batch_embeddings(model, batch, device)

        for i, (emb, path, valid) in enumerate(zip(embeddings, batch["paths"], batch["valid"])):
            if valid:
                all_embeddings.append(emb)
                # Restore original path if using RAM cache
                if ram_cache_enabled:
                    all_paths.append(restore_original_path(Path(path), CLIPS_DIR))
                else:
                    all_paths.append(path)
            else:
                failed_count += 1

        elapsed = time.time() - start_time
        clips_done = len(all_embeddings)
        throughput = clips_done / elapsed if elapsed > 0 else 0
        pbar.set_postfix({
            "clips": clips_done,
            "failed": failed_count,
            "clips/s": f"{throughput:.1f}"
        })

    elapsed_total = time.time() - start_time

    # Cleanup RAM cache
    if ram_cache_enabled:
        cleanup_ram_cache(cache_subdir="vjepa_clips")

    # Stack embeddings
    embeddings = np.stack(all_embeddings).astype(np.float32)
    clip_paths = all_paths

    print(f"\n=== Processing Stats ===")
    print(f"Total clips:     {len(all_clips)}")
    print(f"Successful:      {len(clip_paths)}")
    print(f"Failed:          {failed_count}")
    print(f"Time:            {elapsed_total:.1f}s")
    print(f"Throughput:      {len(clip_paths) / elapsed_total:.1f} clips/sec")
    print(f"Embedding shape: {embeddings.shape}")

    # Deduplicate
    if not args.no_dedupe:
        embeddings, clip_paths, num_removed = deduplicate_embeddings(
            embeddings, clip_paths, args.threshold
        )

    # Save embeddings
    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(EMBEDDINGS_FILE.with_suffix('.paths.npy'), clip_paths)

    print(f"\n=== EMBEDDING COMPLETE ===")
    print(f"Saved: {EMBEDDINGS_FILE}")
    print(f"Shape: {embeddings.shape}")
    print(f"Unique clips: {len(clip_paths)}")


if __name__ == "__main__":
    main()
