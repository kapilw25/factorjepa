"""SAM 3.1 text-prompted video segmentation → agent/layout masks. GPU-only.
    python -u src/m10_sam_segment.py --SANITY --local-data data/val_1k_local --no-wandb 2>&1 | tee logs/m10_sanity.log
    python -u src/m10_sam_segment.py --POC --local-data data/val_1k_local --no-wandb 2>&1 | tee logs/m10_poc.log
    python -u src/m10_sam_segment.py --FULL --local-data data/full_local --no-wandb 2>&1 | tee logs/m10_full.log
    python -u src/m10_sam_segment.py --SANITY --plot    # re-generate plots only (no GPU, reads existing outputs)
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    check_gpu, add_subset_arg, add_local_data_arg, get_output_dir, get_module_output_dir,
    load_subset, get_sanity_clip_limit, get_total_clips,
)
from utils.checkpoint import save_json_checkpoint, load_json_checkpoint
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.gpu_batch import cleanup_temp
from utils.output_guard import verify_or_skip
from utils.plots import init_style, save_fig, COLORS, SCENE_COLORS
from utils.progress import make_pbar
from utils.video_io import decode_video_bytes
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb

import yaml
from sam3 import build_sam3_predictor


# ── SAM 3.1 Model Loading ───────────────────────────────────────────

def load_sam3(model_id: str):
    """Load SAM 3 / 3.1 video predictor. Uses build_sam3_predictor (unified entry point)."""
    version = "sam3.1" if "3.1" in model_id else "sam3"
    # use_fa3=False: SAM 3.1 defaults to FA3 (flash_attn_interface), a separate package
    # from FA2 (flash_attn 2.8.3). We have FA2; FA3 not installed. Falls back to SDPA.
    return build_sam3_predictor(version=version, use_fa3=False)


# ── Per-Clip Tags → Agent Prompt ─────────────────────────────────────

def load_tags_lookup(tags_path: Path) -> dict:
    """Load tags.json and index by clip path. FATAL if missing."""
    if not tags_path.exists():
        print(f"FATAL: tags.json not found at {tags_path}")
        print("  Download: python -u src/utils/hf_outputs.py download-data")
        print("  Or filter from full: see iter/iter8/plan_code_development.md")
        sys.exit(1)
    tags = json.load(open(tags_path))
    lookup = {}
    for t in tags:
        clip_path = f"{t['section']}/{t['video_id']}/{t['source_file']}"
        lookup[clip_path] = t
    print(f"Tags loaded: {len(lookup)} clips from {tags_path}")
    return lookup


def get_agent_prompts(clip_key: str, tags_lookup: dict) -> list:
    """Get per-object contextual text prompts from tags.json. Returns list or None.

    SAM 3.1 requires ONE add_prompt call per object category (Meta benchmark pattern).
    Contextual prompts ("bus on road in market") outperform bare nouns ("bus") because
    SAM3's text encoder disambiguates with scene context. All prompts stay under 30 BPE
    tokens (SAM3 context_length=32, minus SOT/EOT). Context from Qwen3-VL tags.
    """
    if clip_key not in tags_lookup:
        print(f"  SKIP: clip {clip_key} not found in tags.json")
        return None
    tag = tags_lookup[clip_key]
    objects = tag.get("notable_objects", [])
    if not objects:
        print(f"  SKIP: clip {clip_key} has empty notable_objects")
        return None

    # Build context suffix from existing Qwen3-VL tags (no extra VLM call)
    scene = tag.get("scene_type", "").replace("_", " ")
    road = tag.get("road_surface", "")
    context = f"on {road} road in {scene}" if scene and road else (f"in {scene}" if scene else "")

    prompts = []
    for obj in objects:
        name = obj.replace("_", " ")
        prompt = f"{name} {context}".strip() if context else name
        prompts.append(prompt)
    return prompts


# ── Frame I/O for SAM 3.1 ───────────────────────────────────────────

def save_frames_as_jpeg(frames_np: np.ndarray, frame_dir: str) -> str:
    """Save clip frames as JPEG folder for SAM 3.1 video predictor input."""
    os.makedirs(frame_dir, exist_ok=True)
    for i in range(frames_np.shape[0]):
        Image.fromarray(frames_np[i]).save(os.path.join(frame_dir, f"{i:05d}.jpg"))
    return frame_dir


# ── Per-Clip Segmentation ───────────────────────────────────────────

def segment_clip(predictor, frame_dir: str, agent_prompts: list,
                 dilation_px: int, min_confidence: float,
                 min_mask_area_pct: float = 0.001) -> dict:
    """Run SAM 3.1 text-prompted segmentation on a video clip.

    Calls add_prompt ONCE PER OBJECT CATEGORY (Meta benchmark pattern).
    Each call detects all instances of that category, then propagates.
    Results from all categories are merged into unified agent/layout masks.

    Returns:
        agent_mask: (T, H, W) bool — union of all agent instances per frame
        layout_mask: (T, H, W) bool — complement
        n_agents: int — total distinct agent instances across all categories
        agent_pixel_ratio: float — mean agent pixel fraction across frames
        centroids: {obj_id: {t: (cy, cx)}} for interaction mining
        per_object: {obj_id: {t: mask}} per-object masks
    """
    masks_per_frame = {}
    per_object = {}
    centroids = {}
    accepted_probs = []  # track confidence of accepted masks for quality gate
    n_agents = 0
    H, W = 0, 0  # set from first detected mask
    obj_id_offset = 0

    for prompt in agent_prompts:
        # Start fresh session per category (Meta benchmark pattern: prompt → propagate → reset)
        response = predictor.handle_request(dict(
            type="start_session",
            resource_path=frame_dir,
        ))
        session_id = response["session_id"]

        # Text prompt on frame 0 → detect all instances of this category
        prompt_resp = predictor.handle_request(dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        ))

        # Capture frame 0 masks
        out0 = prompt_resp["outputs"]
        n_this = len(out0["out_obj_ids"])

        def _accept_mask(mask_raw, prob):
            """Check mask, set H/W on first detection, resize if needed, filter by area."""
            nonlocal H, W
            m = np.asarray(mask_raw)
            if H == 0:
                H, W = m.shape  # set target resolution from first detected mask
            if m.shape != (H, W):
                m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (W, H), Image.NEAREST)) > 127
            min_area = int(H * W * min_mask_area_pct)
            if m.any() and prob >= min_confidence and m.sum() >= min_area:
                return m
            return None

        for i, oid in enumerate(out0["out_obj_ids"].tolist()):
            m = _accept_mask(out0["out_binary_masks"][i], float(out0["out_probs"][i]))
            if m is not None:
                global_id = obj_id_offset + int(oid)
                masks_per_frame.setdefault(0, {})[global_id] = m
                accepted_probs.append(float(out0["out_probs"][i]))
                n_agents += 1

        # Propagate across all frames
        for resp in predictor.handle_stream_request(dict(
            type="propagate_in_video",
            session_id=session_id,
        )):
            fidx = resp["frame_index"]
            out = resp["outputs"]
            for i, oid in enumerate(out["out_obj_ids"].tolist()):
                m = _accept_mask(out["out_binary_masks"][i], float(out["out_probs"][i]))
                if m is not None:
                    global_id = obj_id_offset + int(oid)
                    masks_per_frame.setdefault(fidx, {})[global_id] = m
                    accepted_probs.append(float(out["out_probs"][i]))

        # Close session before next category
        predictor.handle_request(dict(type="close_session", session_id=session_id))

        # Offset obj_ids so categories don't collide
        obj_id_offset += max(n_this, 1) * 100

    # Build per-frame A_t (agent union) + per-object tracking
    if H == 0:
        H, W = 384, 384  # fallback if no masks detected by any prompt
    T = max(masks_per_frame.keys()) + 1 if masks_per_frame else 1
    agent_mask = np.zeros((T, H, W), dtype=bool)

    for t, frame_masks in masks_per_frame.items():
        frame_union = np.zeros((H, W), dtype=bool)
        for obj_id, mask in frame_masks.items():
            m = np.asarray(mask, dtype=bool)
            frame_union |= m
            per_object.setdefault(obj_id, {})[t] = m
            if m.any():
                ys, xs = np.where(m)
                centroids.setdefault(obj_id, {})[t] = (float(ys.mean()), float(xs.mean()))
        if dilation_px > 0:
            struct = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=bool)
            frame_union = binary_dilation(frame_union, structure=struct)
        agent_mask[t] = frame_union

    layout_mask = ~agent_mask
    agent_pixel_ratio = float(agent_mask.sum()) / max(T * H * W, 1)

    mean_mask_confidence = float(np.mean(accepted_probs)) if accepted_probs else 0.0

    return {
        "agent_mask": agent_mask,
        "layout_mask": layout_mask,
        "n_agents": n_agents,
        "agent_pixel_ratio": agent_pixel_ratio,
        "mean_mask_confidence": mean_mask_confidence,
        "centroids": centroids,
        "per_object": per_object,
    }


# ── Save ─────────────────────────────────────────────────────────────

def mine_interactions(centroids: dict, frame_width: int,
                      max_dist_frac: float, min_frames: int) -> list:
    """Find agent pairs close enough for long enough (D_I interaction mining).

    Proposal Sec 11.2: pairs of agent tracklets that stay within d_max
    for >= min_frames consecutive frames.

    Args:
        centroids: {obj_id: {t: (cy, cx)}} from segment_clip()
        frame_width: W pixels (for distance normalization)
        max_dist_frac: d_max as fraction of frame_width (default 0.20)
        min_frames: minimum consecutive frames for interaction (default 4)

    Returns:
        List of interaction events:
        [{"obj_a": str, "obj_b": str, "frames": [int], "bboxes": [(y1,x1,y2,x2)]}]
    """
    d_max = max_dist_frac * frame_width
    obj_ids = list(centroids.keys())
    interactions = []

    # Pre-extract centroid arrays per object for vectorized distance
    obj_frames = {}
    obj_coords = {}
    for oid in obj_ids:
        times = sorted(centroids[oid].keys())
        obj_frames[oid] = np.array(times)
        obj_coords[oid] = np.array([centroids[oid][t] for t in times])  # (n_frames, 2)

    for i in range(len(obj_ids)):
        for j in range(i + 1, len(obj_ids)):
            a, b = obj_ids[i], obj_ids[j]
            # Vectorized shared frame detection
            shared = np.intersect1d(obj_frames[a], obj_frames[b])
            if len(shared) < min_frames:
                continue

            # Vectorized distance computation across all shared frames
            idx_a = np.searchsorted(obj_frames[a], shared)
            idx_b = np.searchsorted(obj_frames[b], shared)
            coords_a = obj_coords[a][idx_a]  # (n_shared, 2)
            coords_b = obj_coords[b][idx_b]  # (n_shared, 2)
            dists = np.linalg.norm(coords_a - coords_b, axis=1)  # (n_shared,)

            close_mask = dists < d_max
            close_frames = shared[close_mask].tolist()

            if len(close_frames) < min_frames:
                continue

            # Find consecutive runs of length >= min_frames
            runs = []
            current_run = [close_frames[0]]
            for k in range(1, len(close_frames)):
                if close_frames[k] == close_frames[k - 1] + 1:
                    current_run.append(close_frames[k])
                else:
                    if len(current_run) >= min_frames:
                        runs.append(current_run)
                    current_run = [close_frames[k]]
            if len(current_run) >= min_frames:
                runs.append(current_run)

            for run in runs:
                interactions.append({
                    "obj_a": str(a),
                    "obj_b": str(b),
                    "frames": run,
                })

    return interactions


def save_clip_masks(clip_key: str, result: dict, interactions: list,
                    masks_dir: Path, mid_frame_rgb: np.ndarray = None):
    """Save per-clip masks + centroids + interactions + middle frame as compressed .npz."""
    safe_key = clip_key.replace("/", "__")
    out_path = masks_dir / f"{safe_key}.npz"
    centroids_json = json.dumps({str(k): {str(t): list(v) for t, v in frames.items()}
                                  for k, frames in result["centroids"].items()})
    interactions_json = json.dumps(interactions)
    save_dict = dict(
        agent_mask=result["agent_mask"],
        layout_mask=result["layout_mask"],
        centroids_json=np.array(centroids_json),
        interactions_json=np.array(interactions_json),
    )
    if mid_frame_rgb is not None:
        save_dict["mid_frame_rgb"] = mid_frame_rgb  # (H, W, 3) uint8 for overlay plots
    np.savez_compressed(out_path, **save_dict)


# ── Paper Visualizations ─────────────────────────────────────────────

def plot_overlay_per_clip(segments: dict, masks_dir: Path, tags_lookup: dict,
                         output_dir: Path):
    """Save per-clip overlay images: original | agent mask (red) | layout (blue)."""
    overlay_dir = output_dir / "m10_overlay_verify"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for clip_key in segments:
        safe_key = clip_key.replace("/", "__")
        mask_file = masks_dir / f"{safe_key}.npz"
        if not mask_file.exists():
            continue

        data = np.load(mask_file)
        if "mid_frame_rgb" not in data:
            continue

        frame_rgb = data["mid_frame_rgb"]  # (H, W, 3) uint8
        mid = min(data["agent_mask"].shape[0] - 1, frame_rgb.shape[0] // 2 if frame_rgb.ndim == 3 else 0)
        agent_mask = data["agent_mask"][mid]

        # Resize mask to frame if needed
        fh, fw = frame_rgb.shape[:2]
        mh, mw = agent_mask.shape
        if (mh, mw) != (fh, fw):
            from PIL import Image as PILImage
            agent_mask = np.array(PILImage.fromarray(agent_mask).resize((fw, fh), PILImage.NEAREST))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].imshow(frame_rgb)
        axes[0].set_title("Original", fontsize=10)
        axes[0].axis("off")

        overlay = frame_rgb.astype(float).copy()
        overlay[agent_mask > 0] = [255, 50, 50]
        blended = (0.6 * frame_rgb.astype(float) + 0.4 * overlay).astype(np.uint8)
        n_ag = segments[clip_key]["n_agents"]
        pct = segments[clip_key]["agent_pixel_ratio"]
        scene = tags_lookup.get(clip_key, {}).get("scene_type", "unknown")
        axes[1].imshow(blended)
        axes[1].set_title(f"Agents (red): {n_ag} det, {pct:.0%} px", fontsize=10)
        axes[1].axis("off")

        layout_mask = ~(agent_mask > 0)
        overlay2 = frame_rgb.astype(float).copy()
        overlay2[layout_mask] = [50, 80, 220]
        blended2 = (0.7 * frame_rgb.astype(float) + 0.3 * overlay2).astype(np.uint8)
        axes[2].imshow(blended2)
        axes[2].set_title("Layout (blue)", fontsize=10)
        axes[2].axis("off")

        fig.suptitle(f"{scene} — {clip_key.split('/')[-1]}", fontsize=11)
        plt.tight_layout()
        save_fig(fig, str(overlay_dir / safe_key))
        plt.close(fig)

    print(f"  Saved: {overlay_dir}/ ({len(list(overlay_dir.glob('*.png')))} clips)")


def plot_agent_stats(segments: dict, tags_lookup: dict, output_dir: Path):
    """Dual-axis grouped bar: agent count (left) + pixel ratio % (right) by scene_type."""
    init_style()
    if not segments:
        return

    by_scene = {}
    for k, s in segments.items():
        scene = tags_lookup.get(k, {}).get("scene_type", "unknown")
        by_scene.setdefault(scene, []).append(s)

    scenes = sorted(by_scene.keys())
    mean_agents = [np.mean([s["n_agents"] for s in by_scene[sc]]) for sc in scenes]
    mean_pct = [np.mean([s["agent_pixel_ratio"] for s in by_scene[sc]]) * 100 for sc in scenes]
    n_clips = [len(by_scene[sc]) for sc in scenes]

    # Sort by agent count descending
    idx = np.argsort(mean_agents)[::-1]
    scenes = [scenes[i] for i in idx]
    mean_agents = [mean_agents[i] for i in idx]
    mean_pct = [mean_pct[i] for i in idx]
    n_clips = [n_clips[i] for i in idx]

    x = np.arange(len(scenes))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - w / 2, mean_agents, w, label="Mean agents/clip", color=COLORS["blue"], alpha=0.85)
    b2 = ax2.bar(x + w / 2, mean_pct, w, label="Agent pixel ratio (%)", color=COLORS["red"], alpha=0.85)

    ax1.set_ylabel("Mean agents per clip", color=COLORS["blue"])
    ax2.set_ylabel("Agent pixel ratio (%)", color=COLORS["red"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(scenes, n_clips)],
                        fontsize=9, rotation=30, ha="right")
    ax1.set_title(f"SAM 3.1 Agent Detection by Scene Type ({len(segments)} clips)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    save_fig(fig, str(output_dir / "m10_agent_stats"))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    cleanup_temp()
    parser = argparse.ArgumentParser(
        description="SAM 3.1 text-prompted video segmentation → agent/layout masks. GPU-only.")
    parser.add_argument("--SANITY", action="store_true", help="20 clips")
    parser.add_argument("--POC", action="store_true", help="1K clips (val_1k_local)")
    parser.add_argument("--FULL", action="store_true", help="All clips")
    parser.add_argument("--plot", action="store_true",
                        help="Re-generate plots only from existing outputs (no GPU needed)")
    parser.add_argument("--train-config", default="configs/train/ch11_surgery.yaml",
                        help="Factor dataset params YAML")
    parser.add_argument("--output-dir", default=None,
                        help="Override output dir (used by train_surgery.sh)")
    parser.add_argument("--tags-json", default=None,
                        help="Path to tags.json (auto-detected from output dir if omitted)")
    add_subset_arg(parser)
    add_local_data_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    # --plot: re-generate plots from existing outputs (no GPU, no SAM3)
    if args.plot:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = get_module_output_dir("m10_sam_segment", args.subset,
                                               sanity=args.SANITY, poc=args.POC)
        masks_dir = output_dir / "masks"
        segments_file = output_dir / "segments.json"
        if not segments_file.exists():
            print(f"FATAL: {segments_file} not found. Run segmentation first (without --plot).")
            sys.exit(1)
        segments = json.load(open(segments_file))
        # Load tags
        if args.tags_json:
            tags_path = Path(args.tags_json)
        else:
            local_data = getattr(args, "local_data", None)
            if local_data and Path(local_data).joinpath("tags.json").exists():
                tags_path = Path(local_data) / "tags.json"
            else:
                base = get_output_dir(args.subset, sanity=args.SANITY, poc=args.POC)
                tags_path = base / "tags.json"
        tags_lookup = load_tags_lookup(tags_path)
        print(f"Re-generating plots from {output_dir} ({len(segments)} clips)...")
        plot_overlay_per_clip(segments, masks_dir, tags_lookup, output_dir)
        plot_agent_stats(segments, tags_lookup, output_dir)
        print("Done (--plot).")
        return

    ensure_local_data(args)
    check_gpu()

    # Load factor_datasets config
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    factor_cfg = train_cfg["factor_datasets"]
    dilation_px = factor_cfg["agent_dilation_pixels"]
    min_confidence = factor_cfg["min_confidence"]
    min_mask_area_pct = factor_cfg["min_mask_area_pct"]
    interaction_cfg = train_cfg["interaction_mining"]

    # Output routing
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_module_output_dir("m10_sam_segment", args.subset,
                                           sanity=args.SANITY, poc=args.POC)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Skip if done
    if verify_or_skip(output_dir, {"segments": output_dir / "segments.json"},
                      label="m10 SAM3.1"):
        return

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")

    # Clip limit. POC + FULL derive from --subset/--local-data. No yaml fallback (#poc_simplified
    # removed from ch11_surgery.yaml 2026-04-17 when Phase 2b moved to 1K val_1k).
    if args.SANITY:
        clip_limit = get_sanity_clip_limit("default")
    else:
        clip_limit = get_total_clips(
            local_data=getattr(args, "local_data", None),
            subset_file=args.subset)
        if clip_limit == 0:
            print("FATAL: POC/FULL require explicit --subset or --local-data/manifest.json (no yaml fallback)")
            sys.exit(1)

    # Load tags.json (MANDATORY — per-clip agent prompts)
    if args.tags_json:
        tags_path = Path(args.tags_json)
    else:
        # Auto-detect: look in local-data dir first, then output dir parent
        local_data = getattr(args, "local_data", None)
        if local_data and Path(local_data).joinpath("tags.json").exists():
            tags_path = Path(local_data) / "tags.json"
        else:
            # Look in m04's output dir, then base dir (backward compat)
            m04_dir = get_module_output_dir("m04_vlm_tag", args.subset,
                                            sanity=args.SANITY, poc=args.POC)
            if (m04_dir / "tags.json").exists():
                tags_path = m04_dir / "tags.json"
            else:
                tags_path = get_output_dir(args.subset, sanity=args.SANITY, poc=args.POC) / "tags.json"

    tags_lookup = load_tags_lookup(tags_path)

    wb_run = init_wandb("m10", mode, config=vars(args), enabled=not args.no_wandb)

    # Load SAM 3.1
    print(f"\nLoading SAM 3.1 ({factor_cfg['sam_model']})...")
    predictor = load_sam3(factor_cfg["sam_model"])
    print(f"SAM 3.1 loaded ({factor_cfg['sam_model']})")

    # Resume checkpoint
    ckpt_file = output_dir / ".m10_checkpoint.json"
    ckpt = load_json_checkpoint(ckpt_file, default={"processed_keys": []})
    processed_keys = set(ckpt["processed_keys"])
    segments = load_json_checkpoint(output_dir / "segments.json", default={})

    # Iterate clips
    subset_keys = load_subset(args.subset) if args.subset else None
    local_data = getattr(args, "local_data", None)

    print(f"\n{'='*60}")
    print(f"SAM 3.1 Segmentation — {mode}")
    print(f"Clip limit: {clip_limit}")
    print(f"Tags: {tags_path} ({len(tags_lookup)} clips)")
    print(f"Output: {output_dir}")
    print(f"Resume: {len(processed_keys)} already done")
    print(f"{'='*60}\n")

    pbar = make_pbar(total=clip_limit, desc="m10 SAM3.1", unit="clip",
                     initial=len(processed_keys))
    n_processed = len(processed_keys)
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="m10_") as tmp_dir:
        clip_q, tar_stop, _reader = iter_clips_parallel(
            local_data=local_data, subset_keys=subset_keys,
            processed_keys=processed_keys)
        while n_processed < clip_limit:
            item = clip_q.get(timeout=600)
            if item is None:
                break
            clip_key, mp4_bytes = item

            # Decode → numpy frames (T, H, W, C) uint8
            frames_tensor = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames=16)
            if frames_tensor is None:
                print(f"  SKIP: decode failed for {clip_key}")
                continue
            frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
            if frames_np.max() <= 1.0:
                frames_np = (frames_np * 255).astype(np.uint8)
            else:
                frames_np = frames_np.astype(np.uint8)

            # Save as JPEG folder for SAM 3.1
            frame_dir = os.path.join(tmp_dir, "clip_frames")
            if os.path.exists(frame_dir):
                shutil.rmtree(frame_dir)
            save_frames_as_jpeg(frames_np, frame_dir)

            # Per-clip agent prompts from tags.json (one per object category)
            agent_prompts = get_agent_prompts(clip_key, tags_lookup)
            if agent_prompts is None:
                n_processed += 1
                pbar.update(1)
                continue

            # Segment (one add_prompt call per category, Meta benchmark pattern)
            result = segment_clip(predictor, frame_dir, agent_prompts, dilation_px, min_confidence,
                                 min_mask_area_pct=min_mask_area_pct)

            # Mine interactions (D_I)
            interactions = mine_interactions(
                result["centroids"],
                frame_width=frames_np.shape[2],
                max_dist_frac=interaction_cfg["max_distance_frame_fraction"],
                min_frames=interaction_cfg["min_overlap_frames"],
            )

            # Save masks + centroids + interactions + middle frame for overlay plots
            mid_idx = frames_np.shape[0] // 2
            save_clip_masks(clip_key, result, interactions, masks_dir,
                            mid_frame_rgb=frames_np[mid_idx])
            # Concept recall: detected agents vs expected from tags.json
            expected_objects = tags_lookup[clip_key]["notable_objects"]
            n_expected = len(expected_objects)
            n_detected = result["n_agents"]
            concept_recall = n_detected / max(n_expected, 1)

            segments[clip_key] = {
                "n_agents": n_detected,
                "n_expected": n_expected,
                "concept_recall": round(concept_recall, 3),
                "n_frames": frames_np.shape[0],
                "agent_pixel_ratio": result["agent_pixel_ratio"],
                "mean_mask_confidence": result["mean_mask_confidence"],
                "agent_prompt": ", ".join(agent_prompts),
                "n_interactions": len(interactions),
            }

            processed_keys.add(clip_key)
            n_processed += 1
            pbar.update(1)

            # Checkpoint every 10 clips
            if n_processed % 10 == 0:
                save_json_checkpoint({"processed_keys": list(processed_keys)}, ckpt_file)
                save_json_checkpoint(segments, output_dir / "segments.json")

    pbar.close()

    # Final save
    save_json_checkpoint(segments, output_dir / "segments.json")
    elapsed = time.time() - t0
    n_interactions_total = sum(s["n_interactions"] for s in segments.values())
    concept_recalls = [s["concept_recall"] for s in segments.values()]
    mean_concept_recall = float(np.mean(concept_recalls)) if concept_recalls else 0
    pixel_ratios = [s["agent_pixel_ratio"] for s in segments.values()]
    mean_pixel_ratio = float(np.mean(pixel_ratios)) if pixel_ratios else 0
    mask_confs = [s["mean_mask_confidence"] for s in segments.values() if s["mean_mask_confidence"] > 0]
    mean_mask_confidence = float(np.mean(mask_confs)) if mask_confs else 0
    clips_with_agents = sum(1 for s in segments.values() if s["n_agents"] > 0)
    clips_with_agents_pct = clips_with_agents / max(len(segments), 1)

    # Composite quality gate — checks 4 failure modes:
    gate_checks = {
        "pixel_ratio_min": mean_pixel_ratio >= 0.02,       # not empty masks
        "pixel_ratio_max": mean_pixel_ratio <= 0.50,       # not everything masked
        "mask_confidence": mean_mask_confidence >= 0.4,     # SAM is confident
        "clips_with_agents": clips_with_agents_pct >= 0.5,  # >=50% clips have agents
    }
    quality_gate = all(gate_checks.values())

    summary = {
        "n_clips": len(segments),
        "n_total_agents": sum(s["n_agents"] for s in segments.values()),
        "n_total_interactions": n_interactions_total,
        "mean_agent_pixel_ratio": mean_pixel_ratio,
        "mean_concept_recall": mean_concept_recall,
        "mean_mask_confidence": mean_mask_confidence,
        "clips_with_agents_pct": round(clips_with_agents_pct, 3),
        "min_confidence_threshold": min_confidence,
        "min_mask_area_pct": min_mask_area_pct,
        "elapsed_sec": elapsed,
        "sam_model": factor_cfg["sam_model"],
        "quality_gate": "PASS" if quality_gate else "FAIL",
        "quality_gate_checks": {k: "PASS" if v else "FAIL" for k, v in gate_checks.items()},
    }
    save_json_checkpoint(summary, output_dir / "summary.json")

    # Cleanup checkpoint
    if ckpt_file.exists():
        ckpt_file.unlink()

    # Paper visualizations
    plot_overlay_per_clip(segments, masks_dir, tags_lookup, output_dir)
    plot_agent_stats(segments, tags_lookup, output_dir)

    # Quality gate: FATAL if composite check fails (Rule 33: quality gates in Python, not shell)
    if not quality_gate:
        failed = [k for k, v in gate_checks.items() if not v]
        print(f"FATAL: Quality gate FAILED — {len(failed)} check(s):")
        print(f"  pixel_ratio_min:  mean={mean_pixel_ratio:.3f} (need >=0.02) {'FAIL' if not gate_checks['pixel_ratio_min'] else 'OK'}")
        print(f"  pixel_ratio_max:  mean={mean_pixel_ratio:.3f} (need <=0.50) {'FAIL' if not gate_checks['pixel_ratio_max'] else 'OK'}")
        print(f"  mask_confidence:  mean={mean_mask_confidence:.3f} (need >=0.40) {'FAIL' if not gate_checks['mask_confidence'] else 'OK'}")
        print(f"  clips_with_agents: {clips_with_agents_pct:.0%} (need >=50%) {'FAIL' if not gate_checks['clips_with_agents'] else 'OK'}")
        os._exit(1)  # os._exit to kill SAM3 async threads (sys.exit hangs)

    log_metrics(wb_run, summary)
    finish_wandb(wb_run)
    print(f"\nDone: {len(segments)} clips segmented in {elapsed:.0f}s")
    print(f"  Agents detected: {summary['n_total_agents']}")
    print(f"  Interactions mined: {n_interactions_total}")
    print(f"  Mean agent pixel ratio: {summary['mean_agent_pixel_ratio']:.2%}")

    # Force exit: SAM 3.1 spawns async frame-loading threads (async_loading_frames=True)
    # that keep the process alive after main() completes. No shutdown() method exists.
    # Without os._exit, train_surgery.sh hangs indefinitely after m10 step.
    del predictor
    import gc
    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL (unhandled): {e}")
        import traceback
        traceback.print_exc()
        os._exit(1)  # guarantee exit even with SAM3 async threads