"""Generate factor datasets D_L (layout-only) + D_A (agent-only) from m10 masks. CPU-only.

USAGE (every path arg required — CLAUDE.md no-default rule):
    python -u src/m11_factor_datasets.py --SANITY \
        --train-config configs/train/surgery_3stage_DI.yaml \
        --local-data data/val_1k_local --no-wandb 2>&1 | tee logs/m11_sanity.log
    python -u src/m11_factor_datasets.py --POC \
        --train-config configs/train/surgery_3stage_DI.yaml \
        --local-data data/val_1k_local --no-wandb 2>&1 | tee logs/m11_poc.log
    python -u src/m11_factor_datasets.py --FULL --streaming \
        --train-config configs/train/surgery_3stage_DI.yaml \
        --local-data data/full_local --no-wandb 2>&1 | tee logs/m11_full.log
    python -u src/m11_factor_datasets.py --SANITY --plot \
        --train-config configs/train/surgery_3stage_DI.yaml    # re-plot only
"""
import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait as futures_wait
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json

import av
import cv2
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    add_subset_arg, add_local_data_arg, get_module_output_dir,
    load_subset,
)
from utils.checkpoint import save_json_checkpoint, load_json_checkpoint
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.plots import init_style, save_fig, COLORS
from utils.progress import make_pbar
from utils.video_io import decode_video_bytes
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb
from utils.curate_verify import select_verify_clips, curate_and_prune
from utils.cache_policy import (
    add_cache_policy_arg, resolve_cache_policy_interactive, wipe_output_dir,
)

import yaml


# ── Factor Patching ──────────────────────────────────────────────────

def make_layout_only(frames: np.ndarray, agent_mask: np.ndarray,
                     method: str, blur_sigma: float,
                     feather_sigma: float = 3.0) -> np.ndarray:
    """D_L: suppress agents, preserve layout. Feathered mask edges. Returns (T, H, W, C) uint8."""
    # Feather mask: soft blend instead of hard binary (prevents shortcut learning)
    alpha = gaussian_filter(agent_mask.astype(np.float32),
                            sigma=(0, feather_sigma, feather_sigma))
    alpha_4d = alpha[:, :, :, None]  # (T, H, W, 1)

    if method == "blur":
        blurred = np.stack([
            gaussian_filter(frames[t].astype(np.float32), sigma=(blur_sigma, blur_sigma, 0))
            for t in range(frames.shape[0])
        ], axis=0)
        patched = alpha_4d * blurred + (1.0 - alpha_4d) * frames.astype(np.float32)
        return patched.astype(np.uint8)
    elif method == "zero":
        patched = frames.astype(np.float32) * (1.0 - alpha_4d)
        return patched.astype(np.uint8)
    else:
        print(f"FATAL: Unknown layout_patch_method: {method}")
        sys.exit(1)


def make_agent_only(frames: np.ndarray, layout_mask: np.ndarray,
                    method: str, matte_factor: float,
                    feather_sigma: float = 3.0) -> np.ndarray:
    """D_A: suppress background, preserve agents. Feathered mask edges. Returns (T, H, W, C) uint8."""
    # Feather mask: soft edges prevent shortcut learning at mask boundaries
    alpha = gaussian_filter(layout_mask.astype(np.float32),
                            sigma=(0, feather_sigma, feather_sigma))
    alpha_4d = alpha[:, :, :, None]  # (T, H, W, 1)

    if method == "soft_matte":
        patched = frames.astype(np.float32)
        patched = patched * (1.0 - alpha_4d * (1.0 - matte_factor))
        return patched.astype(np.uint8)
    elif method == "hard_zero":
        patched = frames.astype(np.float32) * (1.0 - alpha_4d)
        return patched.astype(np.uint8)
    else:
        print(f"FATAL: Unknown agent_patch_method: {method}")
        sys.exit(1)


def make_interaction_tubes_from_centroids(frames: np.ndarray, interactions: list,
                                          centroids: dict, margin_pct: float) -> list:
    """D_I: crop interaction tubes using centroid bounding boxes.

    Uses centroid pairs to define a crop region (no per-object masks needed).
    Simpler than full mask-based tubes but sufficient for POC.

    Args:
        frames: (T, H, W, C) uint8
        interactions: [{obj_a, obj_b, frames: [int]}] from m10
        centroids: {obj_id: {t: [cy, cx]}} from m10
        margin_pct: box expansion margin

    Returns:
        List of (T_tube, H_crop, W_crop, C) uint8 arrays
    """
    H, W = frames.shape[1], frames.shape[2]
    crop_size = int(max(H, W) * 0.3)
    tubes = []

    for event in interactions:
        obj_a, obj_b = event["obj_a"], event["obj_b"]
        event_frames = event["frames"]
        tube_frames = []

        for t in event_frames:
            if t >= frames.shape[0]:
                continue
            ca = centroids.get(str(obj_a), {}).get(str(t))
            cb = centroids.get(str(obj_b), {}).get(str(t))
            if ca is None or cb is None:
                continue

            cy = int((ca[0] + cb[0]) / 2)
            cx = int((ca[1] + cb[1]) / 2)
            half = int(crop_size * (1 + margin_pct) / 2)
            y1 = max(0, cy - half)
            y2 = min(H, cy + half)
            x1 = max(0, cx - half)
            x2 = min(W, cx + half)

            if y2 - y1 > 10 and x2 - x1 > 10:
                tube_frames.append(frames[t, y1:y2, x1:x2, :])

        if len(tube_frames) >= 4:
            target_h = int(np.median([f.shape[0] for f in tube_frames]))
            target_w = int(np.median([f.shape[1] for f in tube_frames]))
            resized = []
            for f in tube_frames:
                img = PILImage.fromarray(f)
                img = img.resize((target_w, target_h), PILImage.BILINEAR)
                resized.append(np.array(img))
            tubes.append(np.stack(resized))

    return tubes


def make_interaction_tubes_from_bboxes(frames: np.ndarray, interactions: list,
                                       per_object_bboxes: dict, margin_pct: float) -> list:
    """D_I: crop interaction tubes — tight union bbox of interacting agent pair + margin.

    Preferred over `make_interaction_tubes_from_centroids` (fixed 30% centroid square):
    uses per-object tight bboxes saved by m10 to compute a union bbox of the two specific
    interacting agents, preserving aspect ratio + agent identity region + scene context
    immediately around the pair. Still a square-ish crop fed to V-JEPA (not a full
    tubelet-token architecture), but strictly tighter and agent-aware vs the centroid fallback.

    Args:
        frames: (T, H, W, C) uint8
        interactions: [{obj_a, obj_b, frames: [int]}] from m10 mine_interactions
        per_object_bboxes: {obj_id: {t: [y1, x1, y2, x2]}} from m10 .npz per_object_bboxes_json
        margin_pct: box expansion margin (e.g., 0.15 = 15%)

    Returns:
        List of (T_tube, H_crop, W_crop, C) uint8 arrays — one per interaction event
        with ≥4 valid frames (matches centroid-version filter for parity).
    """
    H, W = frames.shape[1], frames.shape[2]
    tubes = []

    for event in interactions:
        obj_a, obj_b = str(event["obj_a"]), str(event["obj_b"])
        event_frames = event["frames"]

        tube_frames = []
        for t in event_frames:
            t_str = str(t)
            ba = per_object_bboxes.get(obj_a, {}).get(t_str)  # [y1, x1, y2, x2] or None
            bb = per_object_bboxes.get(obj_b, {}).get(t_str)
            if ba is None and bb is None:
                continue

            # Tight union bbox of the two agents' bboxes (ignore missing one)
            boxes = [b for b in (ba, bb) if b is not None]
            y1 = min(b[0] for b in boxes)
            x1 = min(b[1] for b in boxes)
            y2 = max(b[2] for b in boxes)
            x2 = max(b[3] for b in boxes)

            # Expand by margin
            h_margin = int((y2 - y1) * margin_pct)
            w_margin = int((x2 - x1) * margin_pct)
            y1 = max(0, y1 - h_margin)
            y2 = min(H, y2 + h_margin)
            x1 = max(0, x1 - w_margin)
            x2 = min(W, x2 + w_margin)

            if y2 - y1 > 10 and x2 - x1 > 10 and t < frames.shape[0]:
                tube_frames.append(frames[t, y1:y2, x1:x2, :])

        if len(tube_frames) >= 4:
            target_h = int(np.median([f.shape[0] for f in tube_frames]))
            target_w = int(np.median([f.shape[1] for f in tube_frames]))
            resized = []
            for f in tube_frames:
                img = PILImage.fromarray(f).resize((target_w, target_h), PILImage.BILINEAR)
                resized.append(np.array(img))
            tubes.append(np.stack(resized))

    return tubes


# ── Paper Visualizations ─────────────────────────────────────────────

def plot_factor_samples(dl_dir: Path, da_dir: Path, output_dir: Path,
                        n_samples: int = 6):
    """Paper-quality grid: D_L (agents blurred) | D_A (background suppressed) for n_samples clips.

    Shows middle frame of each patched clip. Saved as m11_factor_samples.png/.pdf.
    """
    init_style()

    dl_files = sorted(dl_dir.glob("*.npy"))[:n_samples]
    if not dl_files:
        return

    fig, axes = plt.subplots(len(dl_files), 2, figsize=(10, 3 * len(dl_files)))
    if len(dl_files) == 1:
        axes = axes[np.newaxis, :]

    for i, dl_file in enumerate(dl_files):
        da_file = da_dir / dl_file.name
        if not da_file.exists():
            continue

        dl_frames = np.load(dl_file)  # (T, H, W, C) uint8
        da_frames = np.load(da_file)
        mid = dl_frames.shape[0] // 2

        axes[i, 0].imshow(dl_frames[mid])
        axes[i, 0].set_title("D_L: layout-only (agents blurred)", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(da_frames[mid])
        axes[i, 1].set_title("D_A: agent-only (background suppressed)", fontsize=10)
        axes[i, 1].axis("off")

    save_fig(fig, str(output_dir / "m11_factor_samples"))


def plot_interaction_samples(di_dir: Path, manifest: dict, output_dir: Path,
                            n_samples: int = 8):
    """Paper-quality grid of D_I interaction tubes. 2 columns: frame 0 | middle frame.

    Shows spatial crop around interacting agent pairs across time.
    Saved as m11_interaction_samples.png/.pdf.
    """
    init_style()

    tube_files = sorted(di_dir.glob("*.npy"))[:n_samples]
    if not tube_files:
        print("  No D_I tubes found — skipping interaction visualization")
        return

    fig, axes = plt.subplots(len(tube_files), 3, figsize=(14, 3 * len(tube_files)))
    if len(tube_files) == 1:
        axes = axes[np.newaxis, :]

    for i, tube_file in enumerate(tube_files):
        tube = np.load(tube_file)  # (T_tube, H_crop, W_crop, C) uint8
        T = tube.shape[0]
        mid = T // 2

        clip_name = tube_file.stem.rsplit("_tube", 1)[0].replace("__", "/")
        tube_id = tube_file.stem.rsplit("_tube", 1)[1] if "_tube" in tube_file.stem else "?"

        axes[i, 0].imshow(tube[0])
        axes[i, 0].set_title(f"D_I tube {tube_id} — frame 0", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(tube[mid])
        axes[i, 1].set_title(f"frame {mid}/{T}", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(tube[-1])
        axes[i, 2].set_title(f"frame {T-1} (last)", fontsize=9)
        axes[i, 2].axis("off")

        axes[i, 0].set_ylabel(clip_name[:30], fontsize=7, rotation=0, labelpad=80, va="center")

    fig.suptitle(f"D_I Interaction Tubes ({len(tube_files)} samples)", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, str(output_dir / "m11_interaction_samples"))

    # Print D_I summary stats
    clips_with_tubes = sum(1 for v in manifest.values() if v.get("n_interaction_tubes", 0) > 0)
    total_tubes = sum(v.get("n_interaction_tubes", 0) for v in manifest.values())
    tube_counts = [v.get("n_interaction_tubes", 0) for v in manifest.values() if v.get("n_interaction_tubes", 0) > 0]
    print(f"  D_I quality: {clips_with_tubes}/{len(manifest)} clips have tubes "
          f"({100*clips_with_tubes/max(len(manifest),1):.0f}%), "
          f"{total_tubes} total tubes"
          f"{f', median {np.median(tube_counts):.0f}/clip' if tube_counts else ''}")


def plot_factor_per_clip(dl_dir: Path, da_dir: Path, di_dir: Path,
                        masks_dir: Path, output_dir: Path):
    """2x2 per-clip grid: original | D_L (layout) | D_A (agent) | D_I (interaction).

    Pre-filters to ~100 clips via `utils.curate_verify.select_verify_clips()` — paired
    with m10's overlay selection (same seed=42 → identical 100 clip_keys for m10 ↔ m11
    side-by-side spot-check). At 1K writes 100/1000 PNGs; at 10K writes 100/10000
    PNGs (~100× less CPU + disk than the old "write-all-then-curate" path).

    Reads mid_frame_rgb from m10 .npz masks for the original frame.
    Saves individual images per clip to output_dir/m11_per_clip_verify/.
    """
    init_style()
    verify_dir = output_dir / "m11_per_clip_verify"
    verify_dir.mkdir(parents=True, exist_ok=True)

    dl_files = sorted(dl_dir.glob("*.npy"))
    # Recover clip_keys from safe_key filenames (safe_key uses "__" as "/" replacement)
    all_clip_keys = [f.stem.replace("__", "/") + ".mp4" if not f.stem.endswith(".mp4") else f.stem.replace("__", "/")
                     for f in dl_files]
    # Normalize: saved as "<tier>__<city>__<activity>__<video>__<video>-<idx>.mp4" → drop final ".mp4"
    # for consistency with parse_clip_key which expects a trailing ".mp4" path component.
    all_clip_keys = [f.stem.replace("__", "/") for f in dl_files]
    selected = select_verify_clips(all_clip_keys, n_target=100, seed=42)
    print(f"  [m11_per_clip_verify] pre-selected {len(selected)}/{len(dl_files)} clips "
          f"(paired with m10 via seed=42, round-robin city/activity)")

    count = 0
    for dl_file in dl_files:
        safe_key = dl_file.stem
        # Reconstruct clip_key for membership test (matches what select_verify_clips returned)
        clip_key = safe_key.replace("__", "/")
        if clip_key not in selected:
            continue
        da_file = da_dir / f"{safe_key}.npy"
        mask_file = masks_dir / f"{safe_key}.npz"

        dl_frames = np.load(dl_file)
        mid = dl_frames.shape[0] // 2

        # Original frame from m10 masks (saved as mid_frame_rgb)
        original = None
        if mask_file.exists():
            data = np.load(mask_file)
            if "mid_frame_rgb" in data:
                original = data["mid_frame_rgb"]

        # D_A
        da_frame = None
        if da_file.exists():
            da_frames = np.load(da_file)
            da_frame = da_frames[min(mid, da_frames.shape[0] - 1)]

        # D_I — find first tube for this clip
        di_frame = None
        di_tubes = sorted(di_dir.glob(f"{safe_key}_tube*.npy"))
        if di_tubes:
            tube = np.load(di_tubes[0])
            di_frame = tube[tube.shape[0] // 2]

        # 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        clip_name = safe_key.replace("__", "/")

        if original is not None:
            axes[0, 0].imshow(original)
        axes[0, 0].set_title("Original", fontsize=11)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(dl_frames[mid])
        axes[0, 1].set_title("D_L: Layout (agents blurred)", fontsize=11)
        axes[0, 1].axis("off")

        if da_frame is not None:
            axes[1, 0].imshow(da_frame)
            axes[1, 0].set_title("D_A: Agents (background suppressed)", fontsize=11)
        else:
            axes[1, 0].text(0.5, 0.5, "D_A: skipped\n(agent area < threshold)",
                           ha="center", va="center", fontsize=12, transform=axes[1, 0].transAxes)
            axes[1, 0].set_title("D_A: Agents", fontsize=11)
        axes[1, 0].axis("off")

        if di_frame is not None:
            axes[1, 1].imshow(di_frame)
            axes[1, 1].set_title(f"D_I: Interaction tube ({len(di_tubes)} tubes)", fontsize=11)
        else:
            axes[1, 1].text(0.5, 0.5, "D_I: no interactions\nfound for this clip",
                           ha="center", va="center", fontsize=12, transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("D_I: Interaction", fontsize=11)
        axes[1, 1].axis("off")

        fig.suptitle(f"{clip_name}", fontsize=12, y=0.98)
        plt.tight_layout()
        save_fig(fig, str(verify_dir / safe_key))
        plt.close(fig)
        count += 1

    print(f"  Saved: {verify_dir}/ ({count} clips, 2x2 grids)")


def plot_factor_per_videoclip(manifest: dict, dl_dir: Path, da_dir: Path,
                              di_dir: Path, output_dir: Path,
                              local_data: str, top_n: int = 20):
    """2x2 video grid (Original | D_L | D_A | D_I) per clip — for top N best clips.

    Output: H.264-encoded MP4s (web-browser compatible) at
    `output_dir/m11_per_Videoclip_verify/{safe_key}.mp4`. For human eyeballing and
    for embedding in the project website (`docs/index.html`) as `<video controls>`.

    Selection: top N by score = (has_D_A + has_D_I + n_interaction_tubes + agent_pct*100)
    so we get clips with all 3 factors populated and dense agent activity.
    Cost: ~1s/clip encoding × N clips + re-decode top-N MP4s from val_1k_local.
    """
    PANEL_H, PANEL_W = 270, 480
    GRID_H, GRID_W = PANEL_H * 2, PANEL_W * 2
    T = 16
    FPS = 6

    # Score + select top N
    def _score(v):
        return (
            int(v.get("has_D_A", False)) * 1000
            + int(v.get("has_D_I", False)) * 1000
            + v.get("n_interaction_tubes", 0) * 10
            + v.get("agent_pct", 0.0) * 100
        )
    # Streaming-aware: manifest flags (`has_D_L=True`) cover all eligible clips,
    # but under `--streaming` only the ~100 verify clips actually have .npy on
    # disk. Restrict ranking to materialized files to avoid FileNotFoundError
    # when the top-scored manifest entry was short-circuited. Same code path
    # works for legacy (non-streaming) where all manifest `has_D_L` clips have .npy.
    def _has_materialized_dl(clip_key: str) -> bool:
        return (dl_dir / f"{clip_key.replace('/', '__')}.npy").exists()

    ranked = sorted(manifest.items(), key=lambda kv: -_score(kv[1]))
    top = [(k, v) for k, v in ranked
           if v.get("has_D_L") and _has_materialized_dl(k)][:top_n]
    if not top:
        print("  No clips qualify for video grid (need at least D_L .npy materialized on disk)")
        return

    verify_dir = output_dir / "m11_per_Videoclip_verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    top_keys = {k for k, _ in top}

    # Re-decode original MP4 bytes for top N (only ~20 clips, runs in ~30s)
    print(f"  Generating video grids for top {len(top)} clips...")
    with tempfile.TemporaryDirectory(prefix="m11_vid_") as tmp_dir:
        clip_q, tar_stop, _reader = iter_clips_parallel(
            local_data=local_data, subset_keys=top_keys)
        clip_bytes = {}
        n_done = 0
        while n_done < len(top_keys):
            try:
                item = clip_q.get(timeout=120)
            except Exception:
                break
            if item is None:
                break
            clip_key, mp4_bytes = item
            if clip_key in top_keys and clip_key not in clip_bytes:
                clip_bytes[clip_key] = mp4_bytes
                n_done += 1
        tar_stop.set()

        for clip_key, info in top:
            safe_key = clip_key.replace("/", "__")
            out_path = verify_dir / f"{safe_key}.mp4"
            mp4 = clip_bytes.get(clip_key)
            if mp4 is None:
                print(f"  SKIP {safe_key}: original MP4 not retrievable")
                continue

            # Decode original (T, C, H, W) → (T, H, W, 3) uint8
            t = decode_video_bytes(mp4, tmp_dir, clip_key, num_frames=T)
            if t is None:
                print(f"  SKIP {safe_key}: decode failed")
                continue
            orig = t.permute(0, 2, 3, 1).numpy()
            if orig.max() <= 1.0:
                orig = (orig * 255).astype(np.uint8)
            else:
                orig = orig.astype(np.uint8)

            # Load factor arrays — D_L always exists; D_A/D_I may be missing.
            dl = np.load(dl_dir / f"{safe_key}.npy")
            da = np.load(da_dir / f"{safe_key}.npy") if (da_dir / f"{safe_key}.npy").exists() else None
            di_files = sorted(di_dir.glob(f"{safe_key}_tube*.npy"))
            di = np.load(di_files[0]) if di_files else None

            _write_grid_video(
                out_path, orig, dl, da, di, info,
                T=T, panel_h=PANEL_H, panel_w=PANEL_W,
                grid_h=GRID_H, grid_w=GRID_W, fps=FPS,
            )

    n_written = len(list(verify_dir.glob("*.mp4")))
    print(f"  Saved: {verify_dir}/ ({n_written} videos, 960x540 @ {FPS}fps, H.264)")


def _fit_panel(frame: np.ndarray, panel_h: int, panel_w: int) -> np.ndarray:
    """Resize frame to fit panel preserving aspect ratio; pad rest with black."""
    h, w = frame.shape[:2]
    scale = min(panel_h / h, panel_w / w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    y0 = (panel_h - new_h) // 2
    x0 = (panel_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _write_grid_video(out_path: Path, orig: np.ndarray, dl: np.ndarray,
                     da: np.ndarray, di: np.ndarray, info: dict,
                     T: int, panel_h: int, panel_w: int,
                     grid_h: int, grid_w: int, fps: int):
    """Encode 2x2 grid MP4 (H.264, yuv420p) for one clip. Browser-compatible."""
    n_tubes = info.get("n_interaction_tubes", 0)
    pct = info.get("agent_pct", 0.0) * 100

    def _label(panel: np.ndarray, text: str):
        # White text with black outline for legibility
        cv2.putText(panel, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(panel, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    container = av.open(str(out_path), mode='w')
    stream = container.add_stream('h264', rate=fps)
    stream.width = grid_w
    stream.height = grid_h
    stream.pix_fmt = 'yuv420p'
    stream.options = {'preset': 'fast', 'crf': '23'}

    for t in range(T):
        # Cycle each panel's source if shorter than T (D_I tube is variable-length)
        o_t = _fit_panel(orig[t % orig.shape[0]], panel_h, panel_w)
        l_t = _fit_panel(dl[t % dl.shape[0]], panel_h, panel_w)
        if da is not None:
            a_t = _fit_panel(da[t % da.shape[0]], panel_h, panel_w)
        else:
            a_t = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        if di is not None:
            i_t = _fit_panel(di[t % di.shape[0]], panel_h, panel_w)
        else:
            i_t = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

        _label(o_t, "Original")
        _label(l_t, "D_L Layout (agents blurred)")
        _label(a_t, "D_A Agents (BG suppressed)" if da is not None else "D_A skipped (low agent area)")
        _label(i_t, f"D_I Tube ({n_tubes} tubes)" if di is not None else "D_I empty")

        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        grid[:panel_h, :panel_w] = o_t
        grid[:panel_h, panel_w:] = l_t
        grid[panel_h:, :panel_w] = a_t
        grid[panel_h:, panel_w:] = i_t

        # Frame counter + clip stats banner across the bottom
        cv2.putText(grid, f"frame {t+1}/{T}", (grid_w - 130, grid_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(grid, f"frame {t+1}/{T}", (grid_w - 130, grid_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(grid, f"agent_pct={pct:.1f}%", (8, grid_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(grid, f"agent_pct={pct:.1f}%", (8, grid_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        frame = av.VideoFrame.from_ndarray(grid, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def plot_factor_stats(manifest: dict, output_dir: Path):
    """Histogram of agent pixel ratio across clips. Paper figure."""
    init_style()

    if not manifest:
        return

    ratios = [v["agent_pct"] * 100 for v in manifest.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=30, color=COLORS["blue"], alpha=0.85, edgecolor="black")
    ax.set_xlabel("Agent pixel ratio (%)")
    ax.set_ylabel("Number of clips")
    ax.set_title("Distribution of Agent Coverage Across Clips")
    ax.axvline(np.mean(ratios), color=COLORS["red"], linewidth=2.5,
               linestyle="--", label=f"Mean: {np.mean(ratios):.1f}%")
    ax.legend()

    save_fig(fig, str(output_dir / "m11_factor_stats"))


# ── Parallel worker ──────────────────────────────────────────────────

def _get_cpu_workers(cap: int = 32) -> int:
    """Cgroup/container-aware usable CPU count. Linux: sched_getaffinity(0)."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    return max(1, min(cap, n))


def _process_one_clip(args: tuple) -> tuple:
    """Worker: generate D_L/D_A/D_I for one clip. Returns (clip_key, manifest_entry | None)."""
    (clip_key, mp4_bytes, seg_entry, masks_dir_s,
     dl_dir_s, da_dir_s, di_dir_s, cfg) = args

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    masks_dir = Path(masks_dir_s)
    dl_dir = Path(dl_dir_s)
    da_dir = Path(da_dir_s)
    di_dir = Path(di_dir_s)

    safe_key = clip_key.replace("/", "__")
    mask_file = masks_dir / f"{safe_key}.npz"
    if not mask_file.exists():
        return (clip_key, None)

    dl_file = dl_dir / f"{safe_key}.npy"
    da_file = da_dir / f"{safe_key}.npy"
    if dl_file.exists() and da_file.exists():
        n_tubes = len(sorted(di_dir.glob(f"{safe_key}_tube*.npy")))
        return (clip_key, {
            "has_D_L": True, "has_D_A": True,
            "has_D_I": n_tubes > 0,
            "n_interaction_tubes": n_tubes,
            "agent_pct": seg_entry["agent_pixel_ratio"],
            "tube_category_pairs": [],   # #77: unknown for pre-existing .npy cache
        })

    # --regen-D_I: rebuild D_I tubes only, leave D_L/D_A .npy untouched.
    # Reads cached mask.npz (centroids + interactions_json populated by
    # m10 --interactions-only), decodes MP4 for tube crop, writes tubes to di_dir.
    if cfg.get("regen_di_only", False):
        data = np.load(mask_file, allow_pickle=True)
        interactions = json.loads(str(data["interactions_json"])) if "interactions_json" in data else []
        centroids = json.loads(str(data["centroids_json"])) if "centroids_json" in data else {}
        per_object_bboxes = json.loads(str(data["per_object_bboxes_json"])) if "per_object_bboxes_json" in data else {}
        obj_id_to_cat = json.loads(str(data["obj_id_to_cat_json"])) if "obj_id_to_cat_json" in data else {}
        with tempfile.TemporaryDirectory(prefix="m11_di_") as tmp:
            frames_tensor = decode_video_bytes(mp4_bytes, tmp, clip_key, num_frames=16)
        if frames_tensor is None:
            return (clip_key, None)
        frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
        frames_np = (frames_np * 255).astype(np.uint8) if frames_np.max() <= 1.0 else frames_np.astype(np.uint8)
        n_tubes = 0
        tube_category_pairs: list = []
        if interactions and cfg["interaction_mining_enabled"]:
            blacklist = {tuple(sorted(pair)) for pair in cfg["interaction_category_blacklist"]}
            filtered: list = []
            for ev in interactions:
                ca = ev.get("cat_a") or obj_id_to_cat.get(str(ev["obj_a"]))
                cb = ev.get("cat_b") or obj_id_to_cat.get(str(ev["obj_b"]))
                if ca is not None and cb is not None and tuple(sorted((ca, cb))) in blacklist:
                    continue
                ev = dict(ev)
                ev["cat_a"], ev["cat_b"] = ca, cb
                filtered.append(ev)
            if per_object_bboxes:
                tubes = make_interaction_tubes_from_bboxes(
                    frames_np, filtered, per_object_bboxes, cfg["tube_margin"])
            elif centroids:
                tubes = make_interaction_tubes_from_centroids(
                    frames_np, filtered, centroids, cfg["tube_margin"])
            else:
                tubes = []
            for i, tube in enumerate(tubes):
                np.save(di_dir / f"{safe_key}_tube{i}.npy", tube)
            n_tubes = len(tubes)
            tube_category_pairs = [[ev["cat_a"], ev["cat_b"]] for ev in filtered[:n_tubes]]
        agent_pct = seg_entry["agent_pixel_ratio"]
        return (clip_key, {
            "has_D_L": agent_pct <= cfg["max_agent_pct"],
            "has_D_A": agent_pct >= cfg["min_agent_pct"],
            "has_D_I": n_tubes > 0,
            "n_interaction_tubes": n_tubes,
            "agent_pct": agent_pct,
            "tube_category_pairs": tube_category_pairs,
        })

    # Streaming mode: short-circuit for non-verify clips. D_L/D_A/D_I will be
    # generated on-demand inside m09c DataLoader from (raw_mp4, mask.npz).
    # Manifest entries derive from seg_entry agent_pct thresholds + mined
    # interactions count — no mask load, no MP4 decode, no scipy blur, no tube
    # writes. ~90% m11 wall-time reduction at 10K+; 57 GB D_I disk eliminated.
    # iter10 2026-04-22: has_D_I / n_interaction_tubes now populate from
    # seg_entry["n_interactions"] (mined by m10 or m10 --interactions-only)
    # so StreamingFactorDataset knows which clips have D_I without disk scan.
    streaming_skip = (cfg.get("streaming", False)
                      and cfg.get("verify_100_set") is not None
                      and clip_key not in cfg["verify_100_set"])
    if streaming_skip:
        agent_pct = seg_entry["agent_pixel_ratio"]
        n_ints = seg_entry.get("n_interactions", 0)
        return (clip_key, {
            "has_D_L": agent_pct <= cfg["max_agent_pct"],
            "has_D_A": agent_pct >= cfg["min_agent_pct"],
            "has_D_I": n_ints > 0 and cfg.get("interaction_mining_enabled", False),
            "n_interaction_tubes": n_ints,
            "agent_pct": agent_pct,
            "tube_category_pairs": [],
        })

    data = np.load(mask_file, allow_pickle=True)
    agent_mask = data["agent_mask"]
    layout_mask = data["layout_mask"]
    interactions = json.loads(str(data["interactions_json"])) if "interactions_json" in data else []
    centroids = json.loads(str(data["centroids_json"])) if "centroids_json" in data else {}
    per_object_bboxes = json.loads(str(data["per_object_bboxes_json"])) if "per_object_bboxes_json" in data else {}
    # #77: obj_id → canonical 17-cat taxonomy (may be absent for pre-#77 mask caches).
    obj_id_to_cat = json.loads(str(data["obj_id_to_cat_json"])) if "obj_id_to_cat_json" in data else {}

    with tempfile.TemporaryDirectory(prefix="m11_w_") as tmp:
        frames_tensor = decode_video_bytes(mp4_bytes, tmp, clip_key, num_frames=16)
    if frames_tensor is None:
        return (clip_key, None)
    frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
    frames_np = (frames_np * 255).astype(np.uint8) if frames_np.max() <= 1.0 else frames_np.astype(np.uint8)

    T_vid, T_mask = frames_np.shape[0], agent_mask.shape[0]
    if T_mask != T_vid:
        idx = np.linspace(0, T_mask - 1, T_vid, dtype=int)
        agent_mask = agent_mask[idx]
        layout_mask = layout_mask[idx]
    if agent_mask.shape[1:] != frames_np.shape[1:3]:
        H, W = frames_np.shape[1], frames_np.shape[2]
        nm = np.zeros((T_vid, H, W), dtype=bool)
        nl = np.zeros_like(nm)
        for t in range(T_vid):
            nm[t] = np.array(PILImage.fromarray(agent_mask[t]).resize((W, H), PILImage.NEAREST))
            nl[t] = np.array(PILImage.fromarray(layout_mask[t]).resize((W, H), PILImage.NEAREST))
        agent_mask, layout_mask = nm, nl

    agent_pct = seg_entry["agent_pixel_ratio"]
    has_dl = agent_pct <= cfg["max_agent_pct"]
    has_da = agent_pct >= cfg["min_agent_pct"]

    if has_dl:
        dl_frames = make_layout_only(frames_np, agent_mask,
                                     method=cfg["layout_method"],
                                     blur_sigma=cfg["blur_sigma"],
                                     feather_sigma=cfg["feather_sigma"])
        np.save(dl_file, dl_frames)
    if has_da:
        da_frames = make_agent_only(frames_np, layout_mask,
                                    method=cfg["agent_method"],
                                    matte_factor=cfg["matte_factor"],
                                    feather_sigma=cfg["feather_sigma"])
        np.save(da_file, da_frames)

    # D_I (interaction tubes) gated on interaction_mining.enabled yaml flag.
    # 2-stage iter9+ recipe retires Stage 3 → D_I is unused training data →
    # skipping here avoids ~26% of m11 disk waste (1.1 GB @ 10K, ~5 GB @ 50K,
    # ~12 GB @ 115K). m10 still mines interactions into mask .npz (reversible);
    # re-enable by flipping yaml flag → m11 rerun produces D_I from cached masks.
    n_tubes = 0
    tube_category_pairs: list = []
    if interactions and cfg["interaction_mining_enabled"]:
        # #77: category-pair blacklist (order-insensitive). Empty list = accept all.
        # Events inherit cat_a / cat_b from m10's mine_interactions annotation; we
        # rebuild via obj_id_to_cat as a fail-loud fallback when reading a pre-#77
        # mask cache that lacks the inline annotation.
        blacklist = {tuple(sorted(pair)) for pair in cfg["interaction_category_blacklist"]}
        filtered: list = []
        for ev in interactions:
            ca = ev.get("cat_a") or obj_id_to_cat.get(str(ev["obj_a"]))
            cb = ev.get("cat_b") or obj_id_to_cat.get(str(ev["obj_b"]))
            if ca is not None and cb is not None and tuple(sorted((ca, cb))) in blacklist:
                continue
            ev = dict(ev)
            ev["cat_a"], ev["cat_b"] = ca, cb
            filtered.append(ev)

        if per_object_bboxes:
            tubes = make_interaction_tubes_from_bboxes(
                frames_np, filtered, per_object_bboxes, cfg["tube_margin"])
        elif centroids:
            tubes = make_interaction_tubes_from_centroids(
                frames_np, filtered, centroids, cfg["tube_margin"])
        else:
            tubes = []
        for i, tube in enumerate(tubes):
            np.save(di_dir / f"{safe_key}_tube{i}.npy", tube)
        n_tubes = len(tubes)
        tube_category_pairs = [[ev["cat_a"], ev["cat_b"]] for ev in filtered[:n_tubes]]

    return (clip_key, {
        "has_D_L": has_dl, "has_D_A": has_da,
        "has_D_I": n_tubes > 0,
        "n_interaction_tubes": n_tubes,
        "agent_pct": agent_pct,
        "tube_category_pairs": tube_category_pairs,   # #77 typed D_I → paper narrative
    })


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate factor datasets D_L (layout) + D_A (agent) from m10 masks. CPU-only.")
    parser.add_argument("--SANITY", action="store_true", help="20 clips")
    parser.add_argument("--POC", action="store_true", help="1K clips")
    parser.add_argument("--FULL", action="store_true", help="All clips")
    parser.add_argument("--plot", action="store_true",
                        help="Re-generate plots only from existing outputs (no data processing)")
    parser.add_argument("--train-config", required=True,
                        help="Patching params YAML (e.g., configs/train/surgery_3stage_DI.yaml)")
    parser.add_argument("--input-dir", default=None,
                        help="m10 output dir (default: {output_dir}/factors/)")
    parser.add_argument("--output-dir", default=None, help="Override output dir")
    parser.add_argument("--streaming", action="store_true",
                        help="Skip D_L/D_A .npy writes for non-verify clips (100 curated); "
                             "m09c reads factors on-demand via utils.factor_streaming. "
                             "~90%% m11 wall-time reduction + ~340 GB disk saved at 10K. "
                             "See iter/iter9/plan_code_dev.md.")
    parser.add_argument("--regen-D_I", dest="regen_di", action="store_true",
                        help="DEPRECATED (iter10 2026-04-22): D_I tubes are now streamed "
                             "on-demand inside m09c DataLoader via utils/factor_streaming."
                             "stream_interaction_tubes(). No disk writes needed — saves "
                             "57 GB @ 10K / 655 GB @ 115K. Flag kept as no-op for "
                             "backward-compat; emits a warning when used. Run m10 "
                             "--interactions-only to populate interactions_json in mask.npz, "
                             "then m11 --streaming (no --regen-D_I) is enough.")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    # Cache-policy gate (iter11): every destructive delete in this module must route
    # through utils.cache_policy.guarded_delete(path, args.cache_policy, ...).
    # --cache-policy defaults to 1 (keep) so overnight re-runs never destroy cache.
    add_cache_policy_arg(parser)
    args = parser.parse_args()

    # Cache-policy prompt — shells stay thin (CLAUDE.md DELETE PROTECTION).
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    # iter11 v3 (2026-04-26): cache-policy=2 nukes the WHOLE output_dir at startup.
    # m11 single-owns outputs/full/m11_factor_datasets/ — safe to wipe. Closes
    # prompt-trigger ≠ delete-target asymmetry (stale verify/, factor_manifest.json
    # without per-clip .npy, etc).
    _m11_out = Path(args.output_dir) if args.output_dir else get_module_output_dir(
        "m11_factor_datasets", args.subset, sanity=args.SANITY, poc=args.POC)
    wipe_output_dir(_m11_out, args.cache_policy, label=f"output_dir ({_m11_out.name})")

    # iter10 2026-04-22: --regen-D_I is deprecated. D_I streams on-demand
    # inside m09c via stream_interaction_tubes() — no disk writes required.
    if args.regen_di:
        print("\n[DEPRECATED] --regen-D_I is a no-op as of iter10. D_I tubes are "
              "streamed on-demand by m09c's StreamingFactorDataset.")
        print("             Continuing as if --regen-D_I was not passed.")
        args.regen_di = False

    # --plot: re-generate plots from existing outputs (no data processing)
    if args.plot:
        output_dir = Path(args.output_dir) if args.output_dir else get_module_output_dir(
            "m11_factor_datasets", args.subset, sanity=args.SANITY, poc=args.POC)
        m10_dir = get_module_output_dir("m10_sam_segment", args.subset,
                                        sanity=args.SANITY, poc=args.POC)
        masks_dir = (Path(args.input_dir) if args.input_dir else m10_dir) / "masks"
        dl_dir, da_dir, di_dir = output_dir / "D_L", output_dir / "D_A", output_dir / "D_I"
        manifest_file = output_dir / "factor_manifest.json"
        if not manifest_file.exists():
            print(f"FATAL: {manifest_file} not found. Run without --plot first.")
            sys.exit(1)
        manifest = json.load(open(manifest_file))
        print(f"Re-generating plots from {output_dir} ({len(manifest)} clips)...")
        plot_factor_samples(dl_dir, da_dir, output_dir)
        plot_interaction_samples(di_dir, manifest, output_dir)
        plot_factor_per_clip(dl_dir, da_dir, di_dir, masks_dir, output_dir)
        plot_factor_stats(manifest, output_dir)
        # Top-20 video grid (re-decodes original MP4s — needs --local-data)
        local_data_for_video = getattr(args, "local_data", None)
        if local_data_for_video:
            plot_factor_per_videoclip(manifest, dl_dir, da_dir, di_dir,
                                      output_dir, local_data_for_video, top_n=20)
        else:
            print("  Skipping video grid (--plot mode needs --local-data to re-decode originals)")
        print("Done (--plot).")
        return

    ensure_local_data(args)

    # Load config
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    factor_cfg = train_cfg["factor_datasets"]

    # Directories — m11 reads masks from m10, writes factors to its own dir
    m10_dir = get_module_output_dir("m10_sam_segment", args.subset,
                                    sanity=args.SANITY, poc=args.POC)
    input_dir = Path(args.input_dir) if args.input_dir else m10_dir
    output_dir = Path(args.output_dir) if args.output_dir else get_module_output_dir(
        "m11_factor_datasets", args.subset, sanity=args.SANITY, poc=args.POC)
    masks_dir = input_dir / "masks"

    dl_dir = output_dir / "D_L"
    da_dir = output_dir / "D_A"
    di_dir = output_dir / "D_I"
    dl_dir.mkdir(parents=True, exist_ok=True)
    da_dir.mkdir(parents=True, exist_ok=True)
    di_dir.mkdir(parents=True, exist_ok=True)

    manifest_file = output_dir / "factor_manifest.json"
    if args.regen_di:
        print(f"\n[--regen-D_I] Will rebuild D_I tubes only "
              f"(D_L/D_A .npy preserved on disk). Existing manifest: {manifest_file.exists()}")

    # Load segments metadata from m10 (MANDATORY)
    segments_file = input_dir / "segments.json"
    if not segments_file.exists():
        print(f"FATAL: {segments_file} not found. Run m10_sam_segment.py first.")
        sys.exit(1)
    segments = load_json_checkpoint(segments_file)
    if not segments:
        print(f"FATAL: {segments_file} is empty. Run m10_sam_segment.py first.")
        sys.exit(1)

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb_run = init_wandb("m11", mode, config=vars(args), enabled=not args.no_wandb)

    layout_method = factor_cfg["layout_patch_method"]
    agent_method = factor_cfg["agent_patch_method"]
    matte_factor = factor_cfg["soft_matte_factor"]
    blur_sigma = factor_cfg["blur_sigma"]
    feather_sigma = factor_cfg["feather_sigma"]
    min_agent_pct = factor_cfg["min_agent_area_pct"]
    max_agent_pct = factor_cfg["max_agent_area_pct"]
    interaction_cfg = train_cfg["interaction_mining"]
    tube_margin = interaction_cfg["tube_margin_pct"]
    interaction_mining_enabled = interaction_cfg["enabled"]
    interaction_category_blacklist = interaction_cfg["category_pair_blacklist"]   # #77

    local_data = getattr(args, "local_data", None)
    subset_keys = load_subset(args.subset) if args.subset else None

    # Effective work count: TAR reader yields only clips in subset_keys (if given),
    # so pbar + target must reflect that — not len(segments) from m10's full mask cache.
    # Avoids a ~10-min timeout stall at end of Step B when subset < segments (F1 fix
    # for val_1k leak dropped subset_10k from 10000 → 9566). errors_N_fixes #70.
    n_to_process = len(subset_keys) if subset_keys else len(segments)

    print(f"\n{'='*60}")
    print(f"Factor Dataset Generation — {mode}")
    print(f"Input masks: {masks_dir}")
    print(f"Output D_L: {dl_dir}")
    print(f"Output D_A: {da_dir}")
    print(f"Output D_I: {di_dir}")
    print(f"Layout method: {layout_method} | Agent method: {agent_method} (factor={matte_factor})")
    print(f"Clips to process: {n_to_process}")
    print(f"{'='*60}\n")

    manifest = {}
    pbar = make_pbar(total=n_to_process, desc="m11 factors", unit="clip")
    t0 = time.time()

    # Cgroup-aware worker count. Cap at 32 to bound memory (each worker holds
    # ~30 MB of frames + masks during gaussian_filter). On a 24-core cgroup we
    # get 24; on a host-shared 192-core machine we get 192 → clamped to 32.
    n_workers = _get_cpu_workers(cap=32)
    print(f"CPU workers: {n_workers} (os.sched_getaffinity-based; cap=32)\n")

    # Streaming mode: pre-select 100 verify clips (metadata-only, seed=42) that
    # STILL get .npy written so plot_factor_per_clip keeps working unchanged.
    # All other clips short-circuit → m09c streams D_L/D_A from (raw_mp4, mask.npz).
    # Pool for verify selection must match what the TAR reader will actually yield
    # (subset_keys under --subset, else all segment keys). Otherwise some verify
    # picks land on clips the TAR skips → fewer than 100 .npy materialized.
    verify_100_set = None
    if args.streaming:
        verify_pool = list(subset_keys) if subset_keys else list(segments.keys())
        verify_100 = select_verify_clips(verify_pool, n_target=100, seed=42)
        verify_100_set = set(verify_100)
        print(f"Streaming mode: {len(verify_100_set)} verify clips materialized; "
              f"{n_to_process - len(verify_100_set)} short-circuited (no D_L/D_A .npy).")

    worker_cfg = {
        "layout_method": layout_method,
        "agent_method": agent_method,
        "matte_factor": matte_factor,
        "blur_sigma": blur_sigma,
        "feather_sigma": feather_sigma,
        "min_agent_pct": min_agent_pct,
        "max_agent_pct": max_agent_pct,
        "tube_margin": tube_margin,
        "interaction_mining_enabled": interaction_mining_enabled,
        "interaction_category_blacklist": interaction_category_blacklist,   # #77
        "streaming": args.streaming,
        "verify_100_set": verify_100_set,
        "regen_di_only": args.regen_di,
    }
    path_args = (str(masks_dir), str(dl_dir), str(da_dir), str(di_dir))

    with tempfile.TemporaryDirectory(prefix="m11_") as _tmp_dir:
        segment_keys = set(segments.keys())
        clip_q, tar_stop, _reader = iter_clips_parallel(
            local_data=local_data, subset_keys=subset_keys or segment_keys)

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures: dict = {}
            submitted = 0
            n_done = 0
            target = n_to_process

            while n_done < target:
                # Fill pool up to 2× n_workers outstanding to keep decode pipeline warm
                while len(futures) < 2 * n_workers and submitted < target:
                    try:
                        item = clip_q.get(timeout=600)
                    except Exception:
                        break
                    if item is None:
                        break
                    clip_key, mp4_bytes = item
                    if clip_key not in segments:
                        continue
                    safe_key = clip_key.replace("/", "__")
                    mask_file = masks_dir / f"{safe_key}.npz"
                    if not mask_file.exists():
                        print(f"  FATAL: mask file missing for {clip_key}: {mask_file}")
                        sys.exit(1)
                    fut = pool.submit(
                        _process_one_clip,
                        (clip_key, mp4_bytes, segments[clip_key], *path_args, worker_cfg),
                    )
                    futures[fut] = clip_key
                    submitted += 1

                if not futures:
                    break
                done, _pending = futures_wait(list(futures), return_when=FIRST_COMPLETED)
                for fut in done:
                    clip_key = futures.pop(fut)
                    _, entry = fut.result()
                    if entry is None:
                        print(f"  FATAL: worker returned None for {clip_key} (decode or mask load failed)")
                        sys.exit(1)
                    manifest[clip_key] = entry
                    n_done += 1
                    pbar.update(1)

        tar_stop.set()

    pbar.close()

    # Save manifest
    save_json_checkpoint(manifest, manifest_file)

    elapsed = time.time() - t0
    n_tubes_total = sum(v["n_interaction_tubes"] for v in manifest.values())
    print(f"\nDone: {len(manifest)} clips → D_L + D_A + D_I in {elapsed:.0f}s")
    print(f"  D_L: {dl_dir} ({len(list(dl_dir.glob('*.npy')))} files)")
    print(f"  D_A: {da_dir} ({len(list(da_dir.glob('*.npy')))} files)")
    print(f"  D_I: {di_dir} ({len(list(di_dir.glob('*.npy')))} tubes from {n_tubes_total} interactions)")

    # Paper visualizations (D_L vs D_A, D_I tubes, stats, per-clip 2x2 grids)
    plot_factor_samples(dl_dir, da_dir, output_dir)
    plot_interaction_samples(di_dir, manifest, output_dir)
    plot_factor_per_clip(dl_dir, da_dir, di_dir, masks_dir, output_dir)
    plot_factor_stats(manifest, output_dir)
    # Top-20 video grid (2x2 MP4s for human eyeballing + docs/index.html embed)
    plot_factor_per_videoclip(manifest, dl_dir, da_dir, di_dir,
                              output_dir, local_data, top_n=20)

    # Top-20 paper-figures sub-selection — the pre-filter in plot_overlay_per_clip
    # + plot_factor_per_clip already wrote only ~100 per-clip PNGs (metadata-deduped).
    # curate_and_prune here further picks the best-20 by factor_manifest quality
    # score (D_I presence + tube count + agent_pct sweet spot) for paper headline
    # figures. Never deletes originals now — the 100 IS the curated supplementary
    # set, and the 20 lives inside *_top20/ sibling dirs for easy citation.
    try:
        mode_outputs_dir = output_dir.parent   # outputs/<mode>/
        curate_and_prune(mode_outputs_dir, delete_originals=False)
    except Exception as e:
        print(f"[curate_verify] WARN: curation failed (non-fatal): {e}")

    log_metrics(wb_run, {"n_clips": len(manifest), "elapsed": elapsed})
    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
