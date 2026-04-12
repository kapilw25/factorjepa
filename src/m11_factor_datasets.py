"""Generate factor datasets D_L (layout-only) + D_A (agent-only) from m10 masks. CPU-only.
    python -u src/m11_factor_datasets.py --SANITY --local-data data/val_1k_local 2>&1 | tee logs/m11_sanity.log
    python -u src/m11_factor_datasets.py --POC --local-data data/val_1k_local 2>&1 | tee logs/m11_poc.log
    python -u src/m11_factor_datasets.py --FULL --local-data data/full_local 2>&1 | tee logs/m11_full.log
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    add_subset_arg, add_local_data_arg, get_output_dir,
    load_subset,
)
from utils.checkpoint import save_json_checkpoint, load_json_checkpoint
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.output_guard import verify_or_skip
from utils.plots import init_style, save_fig, COLORS
from utils.progress import make_pbar
from utils.video_io import get_clip_key, decode_video_bytes
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb

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


def make_interaction_tubes(frames: np.ndarray, interactions: list,
                           per_object: dict, margin_pct: float) -> list:
    """D_I: crop interaction tubes — bounding box of interacting agent pair + margin.

    Args:
        frames: (T, H, W, C) uint8
        interactions: list from mine_interactions() [{obj_a, obj_b, frames}]
        per_object: {obj_id: {t: mask}} per-object masks
        margin_pct: box expansion margin (e.g., 0.15 = 15%)

    Returns:
        List of (T_tube, H_crop, W_crop, C) uint8 arrays — one per interaction event
    """
    H, W = frames.shape[1], frames.shape[2]
    tubes = []

    for event in interactions:
        obj_a, obj_b = event["obj_a"], event["obj_b"]
        event_frames = event["frames"]

        tube_frames = []
        for t in event_frames:
            # Union bounding box of both agents in this frame
            mask_a = per_object.get(obj_a, {}).get(t)
            mask_b = per_object.get(obj_b, {}).get(t)
            if mask_a is None and mask_b is None:
                continue

            combined = np.zeros((H, W), dtype=bool)
            if mask_a is not None:
                combined |= mask_a
            if mask_b is not None:
                combined |= mask_b

            if not combined.any():
                continue

            ys, xs = np.where(combined)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())

            # Expand by margin
            h_margin = int((y2 - y1) * margin_pct)
            w_margin = int((x2 - x1) * margin_pct)
            y1 = max(0, y1 - h_margin)
            y2 = min(H, y2 + h_margin)
            x1 = max(0, x1 - w_margin)
            x2 = min(W, x2 + w_margin)

            tube_frames.append(frames[t, y1:y2, x1:x2, :])

        if tube_frames:
            # Resize all frames to same size (use the median crop dimensions)
            target_h = int(np.median([f.shape[0] for f in tube_frames]))
            target_w = int(np.median([f.shape[1] for f in tube_frames]))
            resized = []
            for f in tube_frames:
                img = PILImage.fromarray(f)
                img = img.resize((target_w, target_h), PILImage.BILINEAR)
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


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate factor datasets D_L (layout) + D_A (agent) from m10 masks. CPU-only.")
    parser.add_argument("--SANITY", action="store_true", help="20 clips")
    parser.add_argument("--POC", action="store_true", help="1K clips")
    parser.add_argument("--FULL", action="store_true", help="All clips")
    parser.add_argument("--train-config", default="configs/train/ch11_surgery.yaml",
                        help="Patching params YAML")
    parser.add_argument("--input-dir", default=None,
                        help="m10 output dir (default: {output_dir}/factors/)")
    parser.add_argument("--output-dir", default=None, help="Override output dir")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    ensure_local_data(args)

    # Load config
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    factor_cfg = train_cfg["factor_datasets"]

    # Directories
    base_dir = get_output_dir(args.subset, sanity=args.SANITY, poc=args.POC)
    input_dir = Path(args.input_dir) if args.input_dir else base_dir / "factors"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    masks_dir = input_dir / "masks"

    dl_dir = output_dir / "D_L"
    da_dir = output_dir / "D_A"
    di_dir = output_dir / "D_I"
    dl_dir.mkdir(parents=True, exist_ok=True)
    da_dir.mkdir(parents=True, exist_ok=True)
    di_dir.mkdir(parents=True, exist_ok=True)

    # Skip if done
    manifest_file = output_dir / "factor_manifest.json"
    if verify_or_skip(output_dir, {"manifest": manifest_file},
                      label="m11 factor_datasets"):
        return

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

    local_data = getattr(args, "local_data", None)
    subset_keys = load_subset(args.subset) if args.subset else None

    print(f"\n{'='*60}")
    print(f"Factor Dataset Generation — {mode}")
    print(f"Input masks: {masks_dir}")
    print(f"Output D_L: {dl_dir}")
    print(f"Output D_A: {da_dir}")
    print(f"Output D_I: {di_dir}")
    print(f"Layout method: {layout_method} | Agent method: {agent_method} (factor={matte_factor})")
    print(f"Clips to process: {len(segments)}")
    print(f"{'='*60}\n")

    manifest = {}
    pbar = make_pbar(total=len(segments), desc="m11 factors", unit="clip")
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="m11_") as tmp_dir:
        for example in iter_clips_parallel(local_data=local_data, subset_keys=subset_keys):
            clip_key = get_clip_key(example)
            if clip_key not in segments:
                continue

            safe_key = clip_key.replace("/", "__")
            mask_file = masks_dir / f"{safe_key}.npz"
            if not mask_file.exists():
                print(f"  FATAL: mask file missing for {clip_key}: {mask_file}")
                sys.exit(1)

            # Skip if already generated
            dl_file = dl_dir / f"{safe_key}.npy"
            da_file = da_dir / f"{safe_key}.npy"
            if dl_file.exists() and da_file.exists():
                manifest[clip_key] = {
                    "has_D_L": True,
                    "has_D_A": True,
                    "agent_pct": segments[clip_key]["agent_pixel_ratio"],
                }
                pbar.update(1)
                continue

            # Load masks + interaction data
            data = np.load(mask_file, allow_pickle=True)
            agent_mask = data["agent_mask"]
            layout_mask = data["layout_mask"]

            # Load per-object masks + interactions (for D_I)
            interactions = []
            centroids = {}
            if "interactions_json" in data:
                interactions = json.loads(str(data["interactions_json"]))
            if "centroids_json" in data:
                centroids = json.loads(str(data["centroids_json"]))

            # Decode original frames
            mp4_bytes = example["mp4"]
            if isinstance(mp4_bytes, str):
                mp4_bytes = mp4_bytes.encode()
            frames_tensor = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames=16)
            if frames_tensor is None:
                print(f"  FATAL: decode failed for {clip_key}")
                sys.exit(1)
            frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
            if frames_np.max() <= 1.0:
                frames_np = (frames_np * 255).astype(np.uint8)
            else:
                frames_np = frames_np.astype(np.uint8)

            # Quality filters: skip degenerate samples (proposal Sec 11.7)
            agent_pct = segments[clip_key]["agent_pixel_ratio"]
            has_dl = agent_pct <= max_agent_pct
            has_da = agent_pct >= min_agent_pct

            # Generate D_L (layout-only: blur agents, feathered edges)
            if has_dl:
                dl_frames = make_layout_only(frames_np, agent_mask,
                                             method=layout_method, blur_sigma=blur_sigma,
                                             feather_sigma=feather_sigma)
                np.save(dl_file, dl_frames)

            # Generate D_A (agent-only: suppress background, feathered edges)
            if has_da:
                da_frames = make_agent_only(frames_np, layout_mask,
                                            method=agent_method, matte_factor=matte_factor,
                                            feather_sigma=feather_sigma)
                np.save(da_file, da_frames)

            # Generate D_I (interaction tubes from centroids)
            n_tubes = 0
            if interactions and centroids:
                tubes = make_interaction_tubes_from_centroids(
                    frames_np, interactions, centroids, tube_margin)
                for tube_idx, tube in enumerate(tubes):
                    tube_file = di_dir / f"{safe_key}_tube{tube_idx}.npy"
                    np.save(tube_file, tube)
                n_tubes = len(tubes)

            manifest[clip_key] = {
                "has_D_L": has_dl,
                "has_D_A": has_da,
                "has_D_I": n_tubes > 0,
                "n_interaction_tubes": n_tubes,
                "agent_pct": agent_pct,
            }
            pbar.update(1)

    pbar.close()

    # Save manifest
    save_json_checkpoint(manifest, manifest_file)

    elapsed = time.time() - t0
    n_tubes_total = sum(v["n_interaction_tubes"] for v in manifest.values())
    print(f"\nDone: {len(manifest)} clips → D_L + D_A + D_I in {elapsed:.0f}s")
    print(f"  D_L: {dl_dir} ({len(list(dl_dir.glob('*.npy')))} files)")
    print(f"  D_A: {da_dir} ({len(list(da_dir.glob('*.npy')))} files)")
    print(f"  D_I: {di_dir} ({len(list(di_dir.glob('*.npy')))} tubes from {n_tubes_total} interactions)")

    # Paper visualizations
    plot_factor_samples(dl_dir, da_dir, output_dir)
    plot_factor_stats(manifest, output_dir)

    log_metrics(wb_run, {"n_clips": len(manifest), "elapsed": elapsed})
    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
