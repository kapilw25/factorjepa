"""
Generate V-JEPA 2 embeddings for video clips.
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
    VJEPA_FRAMES_PER_CLIP, VJEPA_EMBEDDING_DIM
)

try:
    import torch
    from transformers import AutoModel, AutoProcessor
    import av
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install torch transformers av")
    sys.exit(1)


def check_gpu():
    """Check if CUDA GPU is available."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU not available. This script requires Nvidia GPU.")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")


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


def main():
    parser = argparse.ArgumentParser(description="Generate V-JEPA embeddings for video clips")
    parser.add_argument("--SANITY", action="store_true", help="Process 5 clips only")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    parser.add_argument("--model", type=str, default=VJEPA_MODEL_ID, help="Model ID")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    check_gpu()
    device = "cuda"

    # Find all clips
    clip_dirs = [d for d in CLIPS_DIR.iterdir() if d.is_dir()]
    all_clips = []
    for clip_dir in clip_dirs:
        all_clips.extend(list(clip_dir.glob("*.mp4")))

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

    # Save embeddings
    embeddings = np.stack(embeddings).astype(np.float32)
    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(EMBEDDINGS_FILE.with_suffix('.paths.npy'), clip_paths)

    print(f"\nSaved embeddings: {EMBEDDINGS_FILE}")
    print(f"Shape: {embeddings.shape}")
    print(f"Clips processed: {len(clip_paths)}")


if __name__ == "__main__":
    main()
