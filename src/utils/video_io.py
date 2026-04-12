"""Shared video I/O functions — clip key reconstruction, HF streaming, video decoding.

Moved from m05_vjepa_embed.py to eliminate cross-imports between m*.py scripts.
All pipeline scripts (m04d, m05, m05b, m05c, m09, m10, m11) import from here.

USAGE:
    from utils.video_io import get_clip_key, decode_video_bytes, create_stream
"""
import json
import os
import sys

import numpy as np
import torch

from utils.config import HF_DATASET_REPO

# Video decoder: prefer torchcodec (GPU NVDEC), fallback to PyAV
try:
    from torchcodec.decoders import VideoDecoder
    _USE_TORCHCODEC = True
except ImportError:
    VideoDecoder = None
    _USE_TORCHCODEC = False

if not _USE_TORCHCODEC:
    try:
        import av
    except ImportError:
        print("FATAL: Neither torchcodec nor av available for video decoding")
        print("Install with: pip install av")
        sys.exit(1)


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
        ds = load_dataset("webdataset", data_files=f"{local_data}/*.tar",
                          split="train", streaming=True)
    else:
        ds = load_dataset(HF_DATASET_REPO, split="train", streaming=True)
    ds = ds.decode(False)
    if skip_count > 0:
        ds = ds.skip(skip_count)
    return ds


def _load_torchcodec(video_path: str, num_frames: int) -> torch.Tensor:
    """Load video using torchcodec (GPU NVDEC, 5-10x faster than PyAV CPU)."""
    decoder = VideoDecoder(video_path)
    total_frames = decoder.metadata.num_frames if hasattr(decoder.metadata, "num_frames") else 200
    frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    frames = decoder.get_frames_at(indices=frame_indices.tolist())
    video_tensor = frames.data  # (T, C, H, W)
    del decoder
    if video_tensor.shape[0] < num_frames:
        pad_size = num_frames - video_tensor.shape[0]
        last_frame = video_tensor[-1:].repeat(pad_size, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, last_frame], dim=0)
    return video_tensor[:num_frames]


def _load_av(video_path: str, num_frames: int) -> torch.Tensor:
    """Load video using PyAV (CPU fallback)."""
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


def decode_video_bytes(mp4_bytes: bytes, tmp_dir: str, key: str,
                       num_frames: int) -> torch.Tensor:
    """Write mp4 bytes to temp file, decode frames, delete temp file.

    Args:
        mp4_bytes: Raw MP4 bytes from WebDataset TAR.
        tmp_dir: Temp directory for writing intermediate files.
        key: Clip key (used for safe filename).
        num_frames: Number of frames to extract (uniformly sampled).

    Returns:
        torch.Tensor: (T, C, H, W) uint8 tensor, or None if decode fails.
    """
    safe_key = key.replace("/", "_").replace("\\", "_")
    tmp_path = os.path.join(tmp_dir, f"{safe_key}.mp4")
    try:
        with open(tmp_path, "wb") as f:
            f.write(mp4_bytes)
        if _USE_TORCHCODEC:
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
                print(f"  WARN: failed to delete temp file {tmp_path}")
