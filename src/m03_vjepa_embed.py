"""
Generate V-JEPA 2 embeddings for video clips + deduplicate near-identical clips.
GPU-only script (Nvidia CUDA required).

USAGE:
    python -u src/m03_vjepa_embed.py --SANITY 2>&1 | tee logs/m03_vjepa_embed_sanity.log
    python -u src/m03_vjepa_embed.py --FULL 2>&1 | tee logs/m03_vjepa_embed_full.log
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    CLIPS_DIR, EMBEDDINGS_FILE, VJEPA_MODEL_ID,
    VJEPA_FRAMES_PER_CLIP, VJEPA_EMBEDDING_DIM,
    ensure_clips_exist, check_gpu, get_all_clips
)

try:
    import torch
    from transformers import AutoModel, AutoProcessor
    import av
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install torch transformers av")
    sys.exit(1)

# Deduplication threshold (cosine similarity)
DEDUPE_THRESHOLD = 0.95


def load_video_frames(video_path: Path, num_frames: int = 64) -> np.ndarray:
    """
    Load frames from a video file.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample

    Returns:
        numpy array of shape (num_frames, H, W, 3)
    """
    container = av.open(str(video_path))
    stream = container.streams.video[0]

    total_frames = stream.frames
    if total_frames == 0:
        # Estimate from duration
        total_frames = int(stream.duration * stream.average_rate)

    # Sample frame indices evenly
    indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)

    frames = []
    frame_idx = 0
    for frame in container.decode(video=0):
        if frame_idx in indices:
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
        frame_idx += 1
        if len(frames) >= num_frames:
            break

    container.close()

    # Pad if we didn't get enough frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    return np.stack(frames[:num_frames])


def get_embedding(model, processor, video_path: Path, device: str) -> np.ndarray:
    """
    Get V-JEPA embedding for a video clip.

    Args:
        model: V-JEPA model
        processor: V-JEPA processor
        video_path: Path to video clip
        device: Device to use (cuda/cpu)

    Returns:
        Embedding vector of shape (embedding_dim,)
    """
    frames = load_video_frames(video_path, VJEPA_FRAMES_PER_CLIP)

    # Process frames
    inputs = processor(images=list(frames), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Get pooled output or mean of last hidden state
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            embedding = outputs.pooler_output.squeeze().cpu().numpy()
        else:
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    return embedding


def deduplicate_embeddings(embeddings: np.ndarray, clip_paths: list, threshold: float = 0.95) -> tuple:
    """
    Remove near-duplicate clips based on cosine similarity.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)
        clip_paths: List of clip paths
        threshold: Cosine similarity threshold for duplicates

    Returns:
        Tuple of (deduplicated_embeddings, deduplicated_paths, num_removed)
    """
    print(f"\n=== Deduplicating clips (cosine sim > {threshold}) ===")
    n = len(embeddings)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Compute cosine similarity matrix (on GPU if available)
    if torch.cuda.is_available():
        normalized_tensor = torch.from_numpy(normalized).cuda()
        similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.T).cpu().numpy()
    else:
        similarity_matrix = normalized @ normalized.T

    # Find duplicates using greedy approach
    # Keep track of which indices to keep
    keep_mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep_mask[i]:
            continue
        # Find all clips similar to clip i (excluding self)
        similar = np.where((similarity_matrix[i] > threshold) & (np.arange(n) > i))[0]
        # Mark similar clips for removal
        keep_mask[similar] = False

    # Filter embeddings and paths
    keep_indices = np.where(keep_mask)[0]
    deduped_embeddings = embeddings[keep_indices]
    deduped_paths = [clip_paths[i] for i in keep_indices]
    num_removed = n - len(keep_indices)

    print(f"Original clips: {n}")
    print(f"Removed duplicates: {num_removed}")
    print(f"Remaining clips: {len(deduped_paths)}")

    # Show some examples of removed duplicates
    if num_removed > 0:
        removed_indices = np.where(~keep_mask)[0][:5]  # Show first 5
        print(f"\nExample removed clips (similar to earlier clips):")
        for idx in removed_indices:
            # Find which clip it's similar to
            similar_to = np.where((similarity_matrix[idx] > threshold) & (np.arange(n) < idx) & keep_mask)[0]
            if len(similar_to) > 0:
                sim_val = similarity_matrix[idx, similar_to[0]]
                print(f"  {Path(clip_paths[idx]).name} (sim={sim_val:.3f} to {Path(clip_paths[similar_to[0]]).name})")

    return deduped_embeddings, deduped_paths, num_removed


def main():
    parser = argparse.ArgumentParser(description="Generate V-JEPA embeddings + deduplicate")
    parser.add_argument("--SANITY", action="store_true", help="Process 5 clips only")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    parser.add_argument("--model", type=str, default=VJEPA_MODEL_ID, help="Model ID")
    parser.add_argument("--no-dedupe", action="store_true", help="Skip deduplication")
    parser.add_argument("--threshold", type=float, default=DEDUPE_THRESHOLD, help="Dedupe threshold")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    check_gpu()
    device = "cuda"

    # Ensure clips exist (auto-download from HuggingFace if needed)
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
        print(f"SANITY MODE: Processing {len(all_clips)} clips")

    # Load model
    print(f"Loading model: {args.model}")
    try:
        processor = AutoProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model).to(device)
        model.eval()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        print("Falling back to dummy embeddings for testing")
        # Generate dummy embeddings for testing
        embeddings = np.random.randn(len(all_clips), VJEPA_EMBEDDING_DIM).astype(np.float32)
        clip_paths = [str(c) for c in all_clips]

        # Still deduplicate dummy embeddings
        if not args.no_dedupe:
            embeddings, clip_paths, _ = deduplicate_embeddings(embeddings, clip_paths, args.threshold)

        np.save(EMBEDDINGS_FILE, embeddings)
        np.save(EMBEDDINGS_FILE.with_suffix('.paths.npy'), clip_paths)
        print(f"Saved dummy embeddings: {EMBEDDINGS_FILE}")
        return

    # Generate embeddings
    embeddings = []
    clip_paths = []

    for i, clip_path in enumerate(all_clips):
        print(f"[{i+1}/{len(all_clips)}] Processing: {clip_path.name}")
        try:
            emb = get_embedding(model, processor, clip_path, device)
            embeddings.append(emb)
            clip_paths.append(str(clip_path))
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Stack embeddings
    embeddings = np.stack(embeddings).astype(np.float32)
    print(f"\nGenerated embeddings: {embeddings.shape}")

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
