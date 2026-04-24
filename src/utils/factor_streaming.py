"""On-demand factor generation primitives — streaming variant of m11 disk-write path.

Generates D_L / D_A factors from (raw_mp4_bytes, m10_mask_npz) pairs at training
time instead of materializing .npy files to disk. Unlocks 10K → 50K → 115K scale
ladder on a single 500 GB vast.ai instance (vs 500 GB → 3 TB → 5 TB without).

Bitwise-parity contract with m11: `stream_factor(mp4_bytes, mask_npz_path, 'D_L',
cfg, ...)` returns the SAME (T, H, W, C) uint8 array that
`np.load(m11_outputs/D_L/<clip>.npy)` yields for the same inputs. Tested in
scripts/tests_streaming/test_parity.py.

This module is pure: no globals, no side effects, no RNG. Deterministic given
(mp4_bytes, mask_npz_path, factor_type, factor_cfg).
"""
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent.parent))
from m11_factor_datasets import (
    make_layout_only,
    make_agent_only,
    make_interaction_tubes_from_bboxes,
    make_interaction_tubes_from_centroids,
)
from utils.video_io import decode_video_bytes


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _align_mask(
    mask: np.ndarray,
    target_T: int,
    target_hw: Tuple[int, int],
) -> np.ndarray:
    """Temporal + spatial align mask to frame grid — mirrors m11:652-664 EXACTLY.

    Args:
        mask: (T_mask, H_mask, W_mask) bool
        target_T: desired temporal length (matches frames.shape[0])
        target_hw: (H, W) desired spatial shape (matches frames.shape[1:3])
    Returns: (target_T, H, W) bool — same dtype, same alignment logic as m11.
    """
    T_mask = mask.shape[0]
    if T_mask != target_T:
        idx = np.linspace(0, T_mask - 1, target_T, dtype=int)
        mask = mask[idx]
    if mask.shape[1:] != target_hw:
        H, W = target_hw
        aligned = np.zeros((target_T, H, W), dtype=bool)
        for t in range(target_T):
            aligned[t] = np.array(
                PILImage.fromarray(mask[t]).resize((W, H), PILImage.NEAREST)
            )
        mask = aligned
    return mask


def stream_factor(
    mp4_bytes: bytes,
    mask_npz_path: Path,
    factor_type: str,
    factor_cfg: dict,
    num_frames: int,
    tmp_dir: str,
    clip_key: str,
) -> np.ndarray:
    """Generate D_L or D_A factor array on-demand. Returns (T, H, W, C) uint8.

    Args:
        mp4_bytes: raw MP4 bytes (same source as m11 reads via iter_clips_parallel)
        mask_npz_path: path to m10 .npz with 'agent_mask', 'layout_mask' keys
        factor_type: "D_L" (layout-only, agents blurred) or "D_A" (agent-only, BG matte)
        factor_cfg: flattened m11 config dict with keys:
            layout_method, blur_sigma, feather_sigma      # for D_L
            agent_method, matte_factor, feather_sigma      # for D_A
        num_frames: temporal count (default 16, matches m10/m11)
        tmp_dir: per-worker scratch dir for MP4 decode
        clip_key: for error context + decode_video_bytes tmp name

    Output shape/dtype is bitwise-identical to what m11 writes at
    src/m11_factor_datasets.py:675 (D_L) or :681 (D_A) given the same inputs.
    """
    if factor_type not in ("D_L", "D_A"):
        # D_I is handled by the separate `stream_interaction_tubes()` entrypoint
        # below (returns List[np.ndarray]) — not routed through this scalar function.
        raise ValueError(
            f"stream_factor: factor_type must be 'D_L' or 'D_A', got {factor_type!r}. "
            f"For D_I, call stream_interaction_tubes() instead.")

    frames_tensor = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames=num_frames)
    if frames_tensor is None:
        raise RuntimeError(
            f"stream_factor: decode_video_bytes returned None for clip_key={clip_key!r}. "
            f"Upstream MP4 is corrupt or unreadable.")
    frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
    if frames_np.max() <= 1.0:
        frames_np = (frames_np * 255).astype(np.uint8)
    else:
        frames_np = frames_np.astype(np.uint8)

    data = np.load(mask_npz_path)
    if factor_type == "D_L":
        agent_mask = data["agent_mask"]
        agent_mask = _align_mask(
            agent_mask,
            target_T=frames_np.shape[0],
            target_hw=(frames_np.shape[1], frames_np.shape[2]),
        )
        return make_layout_only(
            frames_np, agent_mask,
            method=factor_cfg["layout_method"],
            blur_sigma=factor_cfg["blur_sigma"],
            feather_sigma=factor_cfg["feather_sigma"],
        )

    layout_mask = data["layout_mask"]
    layout_mask = _align_mask(
        layout_mask,
        target_T=frames_np.shape[0],
        target_hw=(frames_np.shape[1], frames_np.shape[2]),
    )
    return make_agent_only(
        frames_np, layout_mask,
        method=factor_cfg["agent_method"],
        matte_factor=factor_cfg["matte_factor"],
        feather_sigma=factor_cfg["feather_sigma"],
    )


def stream_interaction_tubes(
    mp4_bytes: bytes,
    mask_npz_path: Path,
    interaction_cfg: dict,
    num_frames: int = 16,
    tmp_dir: str = None,
    clip_key: str = "",
) -> List[np.ndarray]:
    """Generate D_I interaction tubes on-demand from (raw MP4 bytes + m10 mask.npz).

    Bitwise-parity contract with m11's legacy disk path at
    src/m11_factor_datasets.py:_process_one_clip lines 643-691 (the regen_di_only
    branch). Follows the same 4-step pipeline:
      1. np.load(mask_npz_path) → interactions_json + per_object_bboxes_json +
         centroids_json + obj_id_to_cat_json
      2. decode_video_bytes(mp4_bytes, ...) → frames_np (T, H, W, C) uint8
      3. Apply category_pair_blacklist filter (order-insensitive, per #77)
      4. If per_object_bboxes present: make_interaction_tubes_from_bboxes(...)
         Elif centroids present:       make_interaction_tubes_from_centroids(...)
         Else:                         empty list

    Returns list of (T_tube, H, W, C) uint8 arrays — one per non-filtered
    interaction event with ≥4 valid frames. Empty list when all filtered out or
    no interactions present in mask.npz.

    interaction_cfg must contain:
      - `tube_margin_pct`: float, box expansion margin (e.g. 0.15)
      - `category_pair_blacklist`: list of [cat_a, cat_b] pairs to drop
      - `enabled`: bool (when False returns empty list — matches m11 behavior)

    This is a pure function: deterministic given (mp4_bytes, mask_npz, cfg). No
    globals, no RNG. Called per-step from StreamingFactorDataset; the caller
    picks a random tube via its seeded RNG.
    """
    if not interaction_cfg.get("enabled", False):
        return []

    data = np.load(mask_npz_path, allow_pickle=False)
    interactions = (json.loads(str(data["interactions_json"]))
                    if "interactions_json" in data.files else [])
    if not interactions:
        return []

    centroids = (json.loads(str(data["centroids_json"]))
                 if "centroids_json" in data.files else {})
    per_object_bboxes = (json.loads(str(data["per_object_bboxes_json"]))
                         if "per_object_bboxes_json" in data.files else {})
    obj_id_to_cat = (json.loads(str(data["obj_id_to_cat_json"]))
                     if "obj_id_to_cat_json" in data.files else {})

    # Blacklist filter: match m11:661-670 EXACTLY for parity.
    blacklist = {tuple(sorted(pair))
                 for pair in interaction_cfg["category_pair_blacklist"]}
    filtered: list = []
    for ev in interactions:
        ca = ev.get("cat_a") or obj_id_to_cat.get(str(ev["obj_a"]))
        cb = ev.get("cat_b") or obj_id_to_cat.get(str(ev["obj_b"]))
        if ca is not None and cb is not None and tuple(sorted((ca, cb))) in blacklist:
            continue
        ev = dict(ev)
        ev["cat_a"], ev["cat_b"] = ca, cb
        filtered.append(ev)

    if not filtered:
        return []

    frames_tensor = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames=num_frames)
    if frames_tensor is None:
        raise RuntimeError(
            f"stream_interaction_tubes: decode_video_bytes returned None for "
            f"clip_key={clip_key!r}. Upstream MP4 is corrupt or unreadable.")
    frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
    frames_np = ((frames_np * 255).astype(np.uint8)
                 if frames_np.max() <= 1.0 else frames_np.astype(np.uint8))

    tube_margin = interaction_cfg["tube_margin_pct"]
    if per_object_bboxes:
        return make_interaction_tubes_from_bboxes(
            frames_np, filtered, per_object_bboxes, tube_margin)
    if centroids:
        return make_interaction_tubes_from_centroids(
            frames_np, filtered, centroids, tube_margin)
    return []


def tensor_from_factor_array(
    frames_uint8: np.ndarray,
    num_frames: int,
    crop_size: int,
) -> torch.Tensor:
    """(T, H, W, C) uint8 → (T, C, H, W) float32 ImageNet-normalized.

    Shared normalization path between legacy disk loader (load_factor_clip) and
    streaming dataloader — guarantees bitwise parity across paths.

    Mirrors src/utils/training.py:load_factor_clip body (lines 1154-1172),
    minus the `np.load(path)` disk read.
    """
    frames = frames_uint8
    if frames.shape[1] != crop_size or frames.shape[2] != crop_size:
        resized = []
        for t in range(frames.shape[0]):
            img = PILImage.fromarray(frames[t])
            img = img.resize((crop_size, crop_size), PILImage.BILINEAR)
            resized.append(np.array(img))
        frames = np.stack(resized)
    if frames.shape[0] > num_frames:
        indices = np.linspace(0, frames.shape[0] - 1, num_frames, dtype=int)
        frames = frames[indices]
    elif frames.shape[0] < num_frames:
        pad = np.repeat(frames[-1:], num_frames - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor
