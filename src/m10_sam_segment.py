"""SAM 3.1 text-prompted video segmentation → agent/layout masks. GPU-only.
    python -u src/m10_sam_segment.py --SANITY --local-data data/val_1k_local 2>&1 | tee logs/m10_sanity.log
    python -u src/m10_sam_segment.py --POC --local-data data/val_1k_local 2>&1 | tee logs/m10_poc.log
    python -u src/m10_sam_segment.py --FULL --local-data data/full_local 2>&1 | tee logs/m10_full.log
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
    check_gpu, add_subset_arg, add_local_data_arg, get_output_dir,
    load_subset, get_sanity_clip_limit, get_total_clips,
)
from utils.checkpoint import save_json_checkpoint, load_json_checkpoint
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.gpu_batch import cleanup_temp
from utils.output_guard import verify_or_skip
from utils.plots import init_style, save_fig, COLORS, SCENE_COLORS
from utils.progress import make_pbar
from utils.video_io import get_clip_key, decode_video_bytes
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb

import yaml


# ── SAM 3.1 Model Loading ───────────────────────────────────────────

def load_sam3(model_id: str, gpus: list = None):
    """Load SAM 3 / 3.1 video predictor. Dispatches builder by version."""
    import torch as _torch
    if gpus is None:
        gpus = list(range(_torch.cuda.device_count()))
    if "3.1" in model_id:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        return build_sam3_multiplex_video_predictor(gpus_to_use=gpus)
    else:
        from sam3.model_builder import build_sam3_video_predictor
        return build_sam3_video_predictor(gpus_to_use=gpus)


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


def get_agent_prompt(clip_key: str, tags_lookup: dict) -> str:
    """Build SAM 3.1 text prompt from per-clip notable_objects. FATAL if clip not in tags."""
    if clip_key not in tags_lookup:
        print(f"FATAL: clip {clip_key} not found in tags.json")
        print("  tags.json must contain ALL clips being processed")
        sys.exit(1)
    objects = tags_lookup[clip_key]["notable_objects"]
    if not objects:
        print(f"FATAL: clip {clip_key} has empty notable_objects in tags.json")
        sys.exit(1)
    return ", ".join(objects)


# ── Frame I/O for SAM 3.1 ───────────────────────────────────────────

def save_frames_as_jpeg(frames_np: np.ndarray, frame_dir: str) -> str:
    """Save clip frames as JPEG folder for SAM 3.1 video predictor input."""
    os.makedirs(frame_dir, exist_ok=True)
    for i in range(frames_np.shape[0]):
        Image.fromarray(frames_np[i]).save(os.path.join(frame_dir, f"{i:05d}.jpg"))
    return frame_dir


# ── Per-Clip Segmentation ───────────────────────────────────────────

def segment_clip(predictor, frame_dir: str, agent_prompt: str,
                 dilation_px: int) -> dict:
    """Run SAM 3.1 text-prompted segmentation on a video clip.

    Uses SAM 3.1 streaming API (handle_stream_request + propagate_in_video).
    Output format per frame: {out_obj_ids: arr, out_binary_masks: arr, out_probs: arr}.

    Returns:
        agent_mask: (T, H, W) bool — union of all agent instances per frame
        layout_mask: (T, H, W) bool — complement
        n_agents: int — distinct agent instances detected on frame 0
        agent_pixel_ratio: float — mean agent pixel fraction across frames
        centroids: {obj_id: {t: (cy, cx)}} for interaction mining
        per_object: {obj_id: {t: mask}} per-object masks
    """
    # 1. Start SAM 3.1 video session
    response = predictor.handle_request(dict(
        type="start_session",
        resource_path=frame_dir,
    ))
    session_id = response["session_id"]

    # 2. Text prompt on frame 0 → detect all agent instances
    prompt_resp = predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=agent_prompt,
    ))

    # 3. Capture frame 0 masks from prompt response
    # SAM 3.1 output: {out_obj_ids: (N,), out_binary_masks: (N, H, W), out_probs: (N,)}
    masks_per_frame = {}
    out0 = prompt_resp["outputs"]
    n_agents = len(out0["out_obj_ids"])
    masks_per_frame[0] = {
        int(oid): out0["out_binary_masks"][i]
        for i, oid in enumerate(out0["out_obj_ids"].tolist())
        if out0["out_binary_masks"][i].any()
    }

    # Infer spatial dims from first mask
    if n_agents > 0:
        H, W = out0["out_binary_masks"][0].shape
    else:
        H, W = 384, 384

    # 4. Propagate across all frames (streaming — SAM 3.1 API)
    for resp in predictor.handle_stream_request(dict(
        type="propagate_in_video",
        session_id=session_id,
    )):
        fidx = resp["frame_index"]
        out = resp["outputs"]
        frame_masks = {}
        for i, oid in enumerate(out["out_obj_ids"].tolist()):
            if out["out_binary_masks"][i].any():
                frame_masks[oid] = out["out_binary_masks"][i]
        masks_per_frame[fidx] = frame_masks

    # 5. Build per-frame A_t (agent union) + per-object tracking
    T = max(masks_per_frame.keys()) + 1 if masks_per_frame else 1
    agent_mask = np.zeros((T, H, W), dtype=bool)
    per_object = {}
    centroids = {}

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

    # 6. Close session (SAM 3.1 API: "close_session", NOT "end_session")
    predictor.handle_request(dict(type="close_session", session_id=session_id))

    return {
        "agent_mask": agent_mask,
        "layout_mask": layout_mask,
        "n_agents": n_agents,
        "agent_pixel_ratio": agent_pixel_ratio,
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


def save_clip_masks(clip_key: str, result: dict, interactions: list, masks_dir: Path):
    """Save per-clip masks + centroids + interactions as compressed .npz."""
    safe_key = clip_key.replace("/", "__")
    out_path = masks_dir / f"{safe_key}.npz"
    # Serialize centroids and interactions as JSON strings (numpy can't store nested dicts)
    centroids_json = json.dumps({str(k): {str(t): list(v) for t, v in frames.items()}
                                  for k, frames in result["centroids"].items()})
    interactions_json = json.dumps(interactions)
    np.savez_compressed(out_path,
                        agent_mask=result["agent_mask"],
                        layout_mask=result["layout_mask"],
                        centroids_json=np.array(centroids_json),
                        interactions_json=np.array(interactions_json))


# ── Paper Visualizations ─────────────────────────────────────────────

def plot_segmentation_samples(segments: dict, masks_dir: Path, tags_lookup: dict,
                              output_dir: Path, n_samples: int = 8):
    """Paper-quality segmentation grid: agent mask (red) | layout mask (blue) per scene_type."""
    init_style()
    clip_keys = list(segments.keys())
    if not clip_keys:
        return

    by_scene = {}
    for k in clip_keys:
        if k in tags_lookup:
            by_scene.setdefault(tags_lookup[k]["scene_type"], []).append(k)
    sampled = []
    for scene_keys in by_scene.values():
        sampled.append(scene_keys[0])
        if len(sampled) >= n_samples:
            break
    while len(sampled) < min(n_samples, len(clip_keys)):
        sampled.append(clip_keys[len(sampled)])

    fig, axes = plt.subplots(len(sampled), 2, figsize=(10, 2.5 * len(sampled)))
    if len(sampled) == 1:
        axes = axes[np.newaxis, :]

    agent_cmap = ListedColormap(["#f0f0f0", COLORS["red"]])
    layout_cmap = ListedColormap(["#f0f0f0", COLORS["blue"]])

    for i, clip_key in enumerate(sampled):
        safe_key = clip_key.replace("/", "__")
        mask_file = masks_dir / f"{safe_key}.npz"
        if not mask_file.exists():
            continue

        data = np.load(mask_file)
        mid = data["agent_mask"].shape[0] // 2
        agent_frame = data["agent_mask"][mid]
        layout_frame = data["layout_mask"][mid]

        scene = tags_lookup[clip_key]["scene_type"] if clip_key in tags_lookup else "unknown"
        n_ag = segments[clip_key]["n_agents"]
        pct = segments[clip_key]["agent_pixel_ratio"]

        axes[i, 0].imshow(agent_frame, cmap=agent_cmap)
        axes[i, 0].set_title(f"{scene} — {n_ag} agents ({pct:.0%} pixels)", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(layout_frame, cmap=layout_cmap)
        axes[i, 1].set_title("Layout (complement)", fontsize=10)
        axes[i, 1].axis("off")

    save_fig(fig, str(output_dir / "m10_segmentation_samples"))


def plot_agent_stats(segments: dict, tags_lookup: dict, output_dir: Path):
    """Bar chart: agent count + pixel ratio by scene_type. Paper figure."""
    init_style()
    if not segments:
        return

    by_scene = {}
    for k, s in segments.items():
        scene = tags_lookup[k]["scene_type"] if k in tags_lookup else "unknown"
        by_scene.setdefault(scene, []).append(s)

    scenes = sorted(by_scene.keys())
    mean_agents = [np.mean([s["n_agents"] for s in by_scene[sc]]) for sc in scenes]
    mean_pct = [np.mean([s["agent_pixel_ratio"] for s in by_scene[sc]]) * 100 for sc in scenes]
    bar_colors = [SCENE_COLORS.get(sc, COLORS["gray"]) for sc in scenes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.barh(scenes, mean_agents, color=bar_colors, alpha=0.85)
    ax1.set_xlabel("Mean agents per clip")
    ax1.set_title("Agent Count by Scene Type")

    ax2.barh(scenes, mean_pct, color=bar_colors, alpha=0.85)
    ax2.set_xlabel("Agent pixel ratio (%)")
    ax2.set_title("Agent Coverage by Scene Type")

    save_fig(fig, str(output_dir / "m10_agent_stats"))


# ── Main ─────────────────────────────────────────────────────────────

def main():
    cleanup_temp()
    parser = argparse.ArgumentParser(
        description="SAM 3.1 text-prompted video segmentation → agent/layout masks. GPU-only.")
    parser.add_argument("--SANITY", action="store_true", help="20 clips")
    parser.add_argument("--POC", action="store_true", help="1K clips (val_1k_local)")
    parser.add_argument("--FULL", action="store_true", help="All clips")
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

    ensure_local_data(args)
    check_gpu()

    # Load factor_datasets config
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)
    factor_cfg = train_cfg["factor_datasets"]
    dilation_px = factor_cfg["agent_dilation_pixels"]
    interaction_cfg = train_cfg["interaction_mining"]

    # Output routing
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base = get_output_dir(args.subset, sanity=args.SANITY, poc=args.POC)
        output_dir = base / "factors"
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Skip if done
    if verify_or_skip(output_dir, {"segments": output_dir / "segments.json"},
                      label="m10 SAM3.1"):
        return

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")

    # Clip limit
    if args.SANITY:
        clip_limit = get_sanity_clip_limit("default")
    elif args.POC:
        clip_limit = train_cfg["poc_simplified"]["n_clips"]
    else:
        clip_limit = get_total_clips(
            local_data=getattr(args, "local_data", None),
            subset_file=args.subset)
        if clip_limit == 0:
            print("FATAL: Cannot determine clip count. Use --subset or --local-data with manifest.json")
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
            base_out = get_output_dir(args.subset, sanity=args.SANITY, poc=args.POC)
            tags_path = base_out / "tags.json"

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
        for example in iter_clips_parallel(local_data=local_data, subset_keys=subset_keys):
            if n_processed >= clip_limit:
                break

            clip_key = get_clip_key(example)
            if clip_key in processed_keys:
                continue
            if subset_keys and clip_key not in subset_keys:
                continue

            mp4_bytes = example["mp4"]
            if isinstance(mp4_bytes, str):
                mp4_bytes = mp4_bytes.encode()

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

            # Per-clip agent prompt from tags.json
            agent_prompt = get_agent_prompt(clip_key, tags_lookup)

            # Segment
            result = segment_clip(predictor, frame_dir, agent_prompt, dilation_px)

            # Mine interactions (D_I)
            interactions = mine_interactions(
                result["centroids"],
                frame_width=frames_np.shape[2],
                max_dist_frac=interaction_cfg["max_distance_frame_fraction"],
                min_frames=interaction_cfg["min_overlap_frames"],
            )

            # Save masks + centroids + interactions
            save_clip_masks(clip_key, result, interactions, masks_dir)
            segments[clip_key] = {
                "n_agents": result["n_agents"],
                "n_frames": frames_np.shape[0],
                "agent_pixel_ratio": result["agent_pixel_ratio"],
                "agent_prompt": agent_prompt,
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
    summary = {
        "n_clips": len(segments),
        "n_total_agents": sum(s["n_agents"] for s in segments.values()),
        "n_total_interactions": n_interactions_total,
        "mean_agent_pixel_ratio": float(np.mean([s["agent_pixel_ratio"] for s in segments.values()])) if segments else 0,
        "elapsed_sec": elapsed,
        "sam_model": factor_cfg["sam_model"],
    }
    save_json_checkpoint(summary, output_dir / "summary.json")

    # Cleanup checkpoint
    if ckpt_file.exists():
        ckpt_file.unlink()

    # Paper visualizations
    plot_segmentation_samples(segments, masks_dir, tags_lookup, output_dir)
    plot_agent_stats(segments, tags_lookup, output_dir)

    log_metrics(wb_run, summary)
    # Shutdown SAM 3.1 predictor (releases GPU worker processes)
    if hasattr(predictor, "shutdown"):
        predictor.shutdown()

    finish_wandb(wb_run)
    print(f"\nDone: {len(segments)} clips segmented in {elapsed:.0f}s")
    print(f"  Agents detected: {summary['n_total_agents']}")
    print(f"  Interactions mined: {n_interactions_total}")
    print(f"  Mean agent pixel ratio: {summary['mean_agent_pixel_ratio']:.2%}")


if __name__ == "__main__":
    main()
