"""Grounded-SAM video segmentation: Grounding DINO + HF Sam3TrackerVideoModel. GPU-only.
Gold standard: https://github.com/IDEA-Research/Grounded-SAM-2 | https://github.com/facebookresearch/sam3
Claude Code: re-WebSearch both URLs on every read of this file.

4-anchor re-seed: DINO detects boxes on frames [0, 4, 8, 12]; HF Sam3TrackerVideoModel
propagates per-anchor within its 4-frame segment via `max_frame_num_to_track=3` (raw sam3
pkg couldn't — errors #33/#34/#35). Measured ~11 s/clip = 4.21× faster than raw sam3 pkg,
→ FULL 115K ETA ~14.7 days on 24GB. See `iter/iter8/plan_TODO.md` for architecture.

USAGE (every path arg required — CLAUDE.md no-default rule):
    python -u src/m10_sam_segment.py --SANITY \
        --train-config configs/train/surgery_3stage_DI_encoder.yaml \
        --local-data data/val_1k_local --no-wandb 2>&1 | tee logs/m10_sanity.log
    python -u src/m10_sam_segment.py --POC \
        --train-config configs/train/surgery_3stage_DI_encoder.yaml \
        --subset data/val_1k_local/sanity_100_dense.json --local-data data/val_1k_local --no-wandb \
        2>&1 | tee logs/m10_poc.log
    python -u src/m10_sam_segment.py --FULL \
        --train-config configs/train/surgery_3stage_DI_encoder.yaml \
        --local-data data/full_local --no-wandb 2>&1 | tee logs/m10_full.log
    python -u src/m10_sam_segment.py --SANITY --plot \
        --train-config configs/train/surgery_3stage_DI_encoder.yaml    # re-generate plots only (no GPU)

HF model download: `facebook/sam3` (~12 GB, first-run only) is pre-cached by `setup_env_uv.sh` step [10/10]
via `hf download` which respects `HF_HUB_ENABLE_HF_TRANSFER=1` (Rust multi-stream, 1.5-3× per file).
"""
import argparse
import gc
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Load .env early so HF_TOKEN, HF_HOME, TRANSFORMERS_CACHE are set before
# any transformers/huggingface_hub import reads them.
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    check_gpu, add_subset_arg, add_local_data_arg, get_output_dir, get_module_output_dir,
    load_subset, get_sanity_clip_limit, get_poc_clip_limit, get_total_clips,
    load_train_config_with_extends,
)
from utils.checkpoint import save_json_checkpoint, load_json_checkpoint
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.gpu_batch import cleanup_temp
from utils.plots import init_style, save_fig, COLORS
from utils.progress import make_pbar
from utils.video_io import decode_video_bytes
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb
from utils.curate_verify import select_verify_clips
from utils.cache_policy import (
    add_cache_policy_arg, resolve_cache_policy_interactive, wipe_output_dir,
)
# iter13 v13 FIX-18 (2026-05-07): observability metrics M1/M5/M6 — see plan_code_dev.md
# Layer C. M2 is reused from existing mean_mask_confidence (line 1037).
from utils.mask_metrics import (
    stability_score, temporal_iou_per_object, compactness, aggregate_percentiles,
)


# ── iter13 v12+ Task 3 (2026-05-06): output co-located with input ──
# m10 writes outputs INSIDE the --local-data directory so hf_outputs.upload_data()
# ships clips + masks + factors as one self-describing bundle. No
# outputs/<mode>/m10_sam_segment/ default — LOCAL_DATA dir IS the source of truth.
def _resolve_output_dir(args) -> Path:
    """Return m10's output dir: --output-dir > <--local-data>/m10_sam_segment/ > FATAL."""
    if getattr(args, "output_dir", None):
        return Path(args.output_dir)
    if getattr(args, "local_data", None):
        return Path(args.local_data) / "m10_sam_segment"
    print("FATAL: m10_sam_segment requires either --output-dir or --local-data")
    print("  USAGE: python -u src/m10_sam_segment.py --FULL "
          "--train-config <yaml> --local-data data/eval_10k_local")
    sys.exit(2)


import torch
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
)


# ── Fixed Agent Taxonomy ─────────────────────────────────────────────
# Clip-independent taxonomy sent to Grounding DINO. 17 categories chosen to
# maximize recall on Indian streets (see iter/iter8/plan_code_dev.md).
# Source of truth: configs/train/ch11_surgery.yaml > factor_datasets.grounding_dino.agent_taxonomy

# DINO_TO_TAG maps DINO's finer taxonomy back to tag_taxonomy.json's coarser
# notable_objects vocabulary for concept_recall reporting. None = bonus detection
# (counts in n_agents but doesn't inflate recall vs VLM expectation).
DINO_TO_TAG = {
    "pedestrian":     "pedestrian",
    "person":         "pedestrian",
    "vendor":         "street_vendor",
    "policeman":      None,
    "bus":            "bus",
    "car":            "car",
    "truck":          "truck",
    "motorcycle":     "bike",
    "bicycle":        "bike",
    "scooter":        "bike",
    "auto rickshaw":  "auto_rickshaw",
    "cycle rickshaw": "cycle_rickshaw",
    "cart":           "handcart",
    "cow":            "sacred_cow",
    "bull":           "sacred_cow",
    "buffalo":        "sacred_cow",
    "dog":            "stray_dog",
}


# ── Model Loading ───────────────────────────────────────────────────

def load_sam3_hf(model_id: str = "facebook/sam3"):
    """HF Sam3TrackerVideoModel (box/point/mask-prompted video tracker). Returns (model, processor).

    Replaces raw `sam3.build_sam3_predictor` to unlock `max_frame_num_to_track` +
    `start_frame_idx` in `propagate_in_video_iterator` (raw pkg crashed #34/#35 with
    empty-tensor). bfloat16 for throughput; model cached at $HF_HOME after first call.
    """
    model = Sam3TrackerVideoModel.from_pretrained(
        model_id, dtype=torch.bfloat16,
    ).to("cuda").eval()
    processor = Sam3TrackerVideoProcessor.from_pretrained(model_id)
    return model, processor


def load_grounding_dino(model_id: str):
    """Load Grounding DINO in fp32 (see errors_N_fixes.md #37).

    transformers 5.x removed auto-cast across DINO's cross-modal encoder:
    pixel_values can be cast to fp16, but the text branch (input_ids → embedding
    → text_enhancer_layer) produces fp32 activations that hit fp16 Linear weights
    → crash. Full autocast wrapping is too invasive; fp32 DINO (+500 MB VRAM,
    negligible vs SAM3's 12 GB) avoids all dtype issues for 4 forwards/clip.
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda").eval()
    return processor, model


def build_compound_prompt(taxonomy: list) -> str:
    """Build Grounding DINO compound prompt: 'cat1. cat2. cat3.' format."""
    return ". ".join(taxonomy) + "."


# ── Per-Clip Tags (metadata only, no longer drives prompting) ────────

def load_tags_lookup(tags_path: Path) -> dict:
    """Load tags.json and index by clip path. FATAL if missing.

    Used for: (1) concept_recall diagnostic (DINO vs VLM-expected), (2) scene_type in plots.
    No longer drives DINO prompts — fixed taxonomy in ch11_surgery.yaml does that.
    """
    if not tags_path.exists():
        print(f"FATAL: tags.json not found at {tags_path}")
        print("  Download: python -u src/utils/hf_outputs.py download-data")
        sys.exit(1)
    tags = json.load(open(tags_path))
    lookup = {}
    for t in tags:
        clip_path = f"{t['section']}/{t['video_id']}/{t['source_file']}"
        lookup[clip_path] = t
    print(f"Tags loaded: {len(lookup)} clips from {tags_path}")
    return lookup


def get_expected_agent_tags(clip_key: str, tags_lookup: dict) -> set:
    """Get VLM-expected agents (for concept_recall). Filters out layout tags."""
    if clip_key not in tags_lookup:
        return set()
    objects = tags_lookup[clip_key].get("notable_objects", []) or []
    # Layout objects (excluded from agent recall): overhead_wires, signage, religious_shrine, skyscraper
    vlm_agent_tags = set(DINO_TO_TAG.values()) - {None}
    return {obj for obj in objects if obj in vlm_agent_tags}


# ── Frame I/O for SAM 3.1 ───────────────────────────────────────────

def save_frames_as_jpeg(frames_np: np.ndarray, frame_dir: str) -> str:
    """Save clip frames as JPEG folder for SAM 3.1 video predictor input."""
    os.makedirs(frame_dir, exist_ok=True)
    for i in range(frames_np.shape[0]):
        Image.fromarray(frames_np[i]).save(os.path.join(frame_dir, f"{i:05d}.jpg"))
    return frame_dir


# ── Per-Clip Segmentation ───────────────────────────────────────────

def detect_boxes_grounding_dino(dino_processor, dino_model,
                                frame_rgb: np.ndarray, compound_prompt: str,
                                box_threshold: float,
                                text_threshold: float) -> dict:
    """Run Grounding DINO on frame 0 → boxes grouped by detected category.

    Returns:
        {
          "boxes_by_cat": {"pedestrian": [[x0,y0,x1,y1], ...], "car": [...], ...},
          "scores_by_cat": {"pedestrian": [0.78, ...], ...},
          "detected_cats": list[str] (categories with >=1 box),
          "H": int, "W": int (frame shape)
        }
    """
    H, W = frame_rgb.shape[:2]
    pil = Image.fromarray(frame_rgb)
    inputs = dino_processor(images=pil, text=compound_prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    # transformers 5.x: processor returns fp32 pixel_values but model weights are fp16.
    # Explicit cast required (4.x auto-cast was removed). int tensors (attention masks,
    # input_ids) must stay int — only cast floating tensors.
    model_dtype = next(dino_model.parameters()).dtype
    for k, v in inputs.items():
        if v.dtype.is_floating_point and v.dtype != model_dtype:
            inputs[k] = v.to(model_dtype)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    # post_process returns boxes in (x_min, y_min, x_max, y_max) absolute pixels.
    # transformers>=4.51: kwarg renamed box_threshold→threshold; `labels` now holds int IDs,
    # `text_labels` holds the string spans matched from the compound prompt.
    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(H, W)],
    )[0]

    boxes_by_cat = {}
    scores_by_cat = {}
    for box, label, score in zip(results["boxes"].cpu().tolist(),
                                 results["text_labels"],
                                 results["scores"].cpu().tolist()):
        # `label` is the matched span from compound_prompt (lowercased substring).
        # Normalize to canonical taxonomy key (handle DINO returning "auto" instead of "auto rickshaw" etc.)
        cat = _canonicalize_label(label)
        if cat is None:
            continue
        boxes_by_cat.setdefault(cat, []).append(box)
        scores_by_cat.setdefault(cat, []).append(score)

    return {
        "boxes_by_cat": boxes_by_cat,
        "scores_by_cat": scores_by_cat,
        "detected_cats": list(boxes_by_cat.keys()),
        "H": H,
        "W": W,
    }


def _canonicalize_label(raw_label: str) -> str:
    """Map Grounding DINO's raw matched-span label to canonical taxonomy key.

    DINO returns the longest matched substring from compound_prompt; for multi-word
    categories like 'auto rickshaw' it may return 'auto' or 'rickshaw' depending on
    which tokens triggered. We match against known taxonomy keys to canonicalize.
    """
    if not raw_label:
        return None
    label = raw_label.strip().lower()
    # Exact hit against taxonomy key
    if label in DINO_TO_TAG:
        return label
    # Substring match for multi-word categories (e.g., "auto" → "auto rickshaw")
    for cat in DINO_TO_TAG:
        if " " in cat and (label in cat.split() or label in cat):
            # Avoid over-matching "cycle" to "bicycle" — prefer "cycle rickshaw"
            # when clear; fallback to longer match if multiple hits.
            return cat
    return None


def segment_clip(sam_model, sam_processor, dino_processor, dino_model,
                 frames_np: np.ndarray,
                 compound_prompt: str,
                 dilation_px: int, min_confidence: float,
                 min_mask_area_pct: float,
                 box_threshold: float, text_threshold: float,
                 min_stability_score: float = 0.0) -> dict:
    """Grounded-SAM segmentation (HF backend): DINO boxes → Sam3TrackerVideoModel tracks.

    Multi-anchor re-seed at frames [0,4,8,12] for T=16. Each anchor × detected-category
    opens an HF inference session, seeds boxes on that anchor frame, propagates forward
    within the 4-frame segment via `propagate_in_video_iterator(start_frame_idx=anchor,
    max_frame_num_to_track=segment_size-1)` — unlike raw sam3 pkg, these params work
    correctly in HF wrapper (#36), giving ~10× speedup with tight per-segment tracking.

    iter13 v13 FIX-18 (2026-05-07): added `min_stability_score` post-filter (Layer A)
    + per-mask M1/M6 metric collection + per-clip M5 from `per_object` (Layer C).
    Quality block returned for downstream summary aggregation.

    Returns: agent_mask, layout_mask, n_agents, agent_pixel_ratio, mean_mask_confidence,
             centroids, per_object, detected_categories, obj_id_to_cat, quality
    """
    # Multi-anchor: DINO on [0,4,8,12] (T=16), each anchor owns its 4-frame segment.
    # Max tracking drift per frame is ~2 frames instead of ~15 (see plan_TODO.md Level 2).
    T = frames_np.shape[0]
    n_anchors = 4
    segment_size = max(1, T // n_anchors)
    anchors = [a * segment_size for a in range(n_anchors)]
    anchor_segments = {}
    for i, a in enumerate(anchors):
        end = anchors[i + 1] if i + 1 < len(anchors) else T
        anchor_segments[a] = set(range(a, end))

    # Step 1: DINO on each anchor frame
    dino_per_anchor = {}
    H = W = 0
    all_detected_categories = set()
    for a in anchors:
        dino_out = detect_boxes_grounding_dino(
            dino_processor, dino_model, frames_np[a], compound_prompt,
            box_threshold, text_threshold,
        )
        H, W = dino_out["H"], dino_out["W"]
        dino_per_anchor[a] = dino_out["boxes_by_cat"]
        all_detected_categories.update(dino_out["boxes_by_cat"].keys())

    if not any(dino_per_anchor[a] for a in anchors):
        empty_mask = np.zeros((T, H if H else 384, W if W else 384), dtype=bool)
        return {
            "agent_mask": empty_mask, "layout_mask": ~empty_mask,
            "n_agents": 0, "agent_pixel_ratio": 0.0, "mean_mask_confidence": 0.0,
            "centroids": {}, "per_object": {}, "detected_categories": [],
            "obj_id_to_cat": {},   # #77: always present (contract with main()/save_clip_masks)
            # iter13 v13 FIX-21 (2026-05-07): emit empty quality block on no-
            # detection clips so the segments[clip_key] schema is uniform
            # across all clips. Without this, main() at the segments dict
            # build site KeyError'd on the first clip that DINO found nothing
            # for (POC v1: clip #5 of 100 had no detections → run aborted).
            "quality": {
                "stability_score": aggregate_percentiles([]),
                "object_score":    aggregate_percentiles([]),
                "compactness":     aggregate_percentiles([]),
                "temporal_iou_m5": 0.0,
                "n_filtered_by_stability": 0,
            },
        }

    # Step 2: one HF video session per clip (reused across anchor × category sub-calls).
    # frames_np is (T, H, W, 3) uint8 — HF processor accepts numpy video directly.
    session = sam_processor.init_video_session(
        video=frames_np, inference_device="cuda", dtype=torch.bfloat16,
    )

    masks_per_frame = {}
    per_object = {}
    centroids = {}
    obj_id_to_cat = {}
    accepted_probs = []
    # iter13 v13 FIX-18 (2026-05-07): per-mask quality accumulators (Layer C).
    # `accepted_stab` collects M1 (stability_score) for each accepted mask;
    # `accepted_compact` collects M6 (compactness). Both are aggregated to
    # percentiles in the return-dict's `quality` block; aggregate over all clips
    # lands in summary.json["quality_aggregate"]. M5 is computed once at end of
    # this function from `per_object`. M2 reuses `accepted_probs` (existing).
    accepted_stab = []
    accepted_compact = []
    n_filtered_by_stability = 0
    n_agents = 0
    obj_id_offset = 0  # reserve 100 ids per (anchor,category) session so D_I can tell them apart

    def _accept_mask(mask_np, prob):
        """Accept/reject a candidate mask + record per-mask quality if accepted.

        iter13 v13 FIX-18: gates on (a) min_confidence (existing), (b) min_area
        (existing), (c) min_stability_score (NEW Layer A — SAM-style perturbation
        IoU; Meta default 0.92). When accepted, also computes M6 compactness
        on the RAW mask (before agent_dilation_pixels expansion) so the metric
        reflects SAM's natural mask shape, not our morphology overlay.

        Mutates outer-scope counters: accepted_stab, accepted_compact,
        n_filtered_by_stability.
        """
        nonlocal n_filtered_by_stability
        m = np.asarray(mask_np, dtype=bool)
        if m.shape != (H, W):
            m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize(
                (W, H), Image.NEAREST)) > 127
        min_area = int(H * W * min_mask_area_pct)
        if not (m.any() and prob >= min_confidence and m.sum() >= min_area):
            return None
        # M1 stability filter — added in Layer A.
        if min_stability_score > 0:
            stab = stability_score(m)
            if stab < min_stability_score:
                n_filtered_by_stability += 1
                return None
            accepted_stab.append(stab)
        else:
            accepted_stab.append(stability_score(m))   # collect for observability even when filter off
        accepted_compact.append(compactness(m))        # M6 (collect on raw, pre-dilation mask)
        return m

    for anchor in anchors:
        boxes_by_cat = dino_per_anchor[anchor]
        segment_frames = anchor_segments[anchor]
        if not boxes_by_cat:
            continue

        for category, boxes in boxes_by_cat.items():
            # Clamp DINO boxes to frame bounds (#28), keep xyxy pixel coords — HF accepts pixel xyxy.
            boxes_xyxy = []
            for x0, y0, x1, y1 in boxes:
                x0c = max(0.0, min(float(W), x0))
                y0c = max(0.0, min(float(H), y0))
                x1c = max(0.0, min(float(W), x1))
                y1c = max(0.0, min(float(H), y1))
                if x1c - x0c <= 0 or y1c - y0c <= 0:
                    continue
                boxes_xyxy.append([x0c, y0c, x1c, y1c])
            if not boxes_xyxy:
                continue

            # HF API: one obj_id per box. `input_boxes` expected_depth=3:
            # [image level, box level, box coordinates]. Points are depth-4 (extra
            # object level), but boxes are flat per image. Parallel with obj_ids.
            local_ids = list(range(len(boxes_xyxy)))
            input_boxes = [boxes_xyxy]  # [[[x1,y1,x2,y2], ...]]
            sam_processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=anchor,
                obj_ids=local_ids,
                input_boxes=input_boxes,
            )

            # Propagate forward within this anchor's segment only — max_frame_num_to_track=3
            # covers frames [anchor..anchor+3]; HF wrapper clips correctly (unlike raw sam3 #35).
            max_track = max(1, segment_size - 1)
            try:
                for output in sam_model.propagate_in_video_iterator(
                    inference_session=session,
                    start_frame_idx=anchor,
                    max_frame_num_to_track=max_track,
                    reverse=False,
                ):
                    fidx = int(output.frame_idx)
                    if fidx not in segment_frames:
                        continue
                    # HF returns pred_masks as logits at model resolution — post_process_masks
                    # resizes to original (H, W) and applies sigmoid + threshold.
                    masks_full = sam_processor.post_process_masks(
                        [output.pred_masks], original_sizes=[(H, W)], binarize=True,
                    )[0]  # (n_objs, 1, H, W) bool
                    # Mask confidence from object_score_logits (sigmoid → [0,1]).
                    # Dataclass attr is `object_score_logits`, NOT `iou_scores` (that's SAM2's name).
                    probs = torch.sigmoid(
                        output.object_score_logits.detach().float()
                    ).squeeze().cpu().numpy()
                    if probs.ndim == 0:
                        probs = probs.reshape(1)
                    for i in range(masks_full.shape[0]):
                        mask_np = masks_full[i, 0].detach().cpu().numpy()
                        prob = float(probs[i])
                        m = _accept_mask(mask_np, prob)
                        if m is not None:
                            global_id = obj_id_offset + i
                            masks_per_frame.setdefault(fidx, {})[global_id] = m
                            if fidx == anchor:
                                obj_id_to_cat[global_id] = category
                                n_agents += 1
                            accepted_probs.append(prob)
            except RuntimeError as e:
                print(f"  WARN: HF Sam3Tracker propagate anchor={anchor} cat={category}: {e}")

            # Reset prompts + obj_ids + tracklets between (anchor × category) pairs.
            # `reset_tracking_data()` keeps the encoded video cache (expensive); use that
            # instead of `reset_inference_session()` which would force re-encoding for every
            # category. Method lives on the session object, not the processor.
            session.reset_tracking_data()
            obj_id_offset += 100

    boxes_by_cat = {c: [] for c in sorted(all_detected_categories)}

    # ── Step 3: Build agent_mask union + centroids ─────────────────
    T = frames_np.shape[0]
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

    # iter13 v13 FIX-18 (2026-05-07): per-clip quality block (Layer C).
    # M1 = SAM stability_score percentiles across accepted masks
    # M2 = sigmoid(object_score_logits) percentiles — resurfaces existing data
    # M5 = mean adjacent-frame IoU per object_id, averaged across objects
    # M6 = isoperimetric compactness percentiles across accepted masks
    # n_filtered_by_stability: count of masks rejected by min_stability_score
    quality = {
        "stability_score": aggregate_percentiles(accepted_stab),       # M1
        "object_score":    aggregate_percentiles(accepted_probs),       # M2 (resurfaced)
        "compactness":     aggregate_percentiles(accepted_compact),     # M6
        "temporal_iou_m5": temporal_iou_per_object(per_object),         # M5 (scalar)
        "n_filtered_by_stability": int(n_filtered_by_stability),
    }

    return {
        "agent_mask": agent_mask,
        "layout_mask": layout_mask,
        "n_agents": n_agents,
        "agent_pixel_ratio": agent_pixel_ratio,
        "mean_mask_confidence": mean_mask_confidence,
        "centroids": centroids,
        "per_object": per_object,
        "detected_categories": list(boxes_by_cat.keys()),
        # #77 (2026-04-20): {obj_id → canonical 17-cat taxonomy key} populated on each
        # anchor frame (line 397). Downstream (mine_interactions / D_I tube builder /
        # factor_manifest) can now label interactions by category pair — previously
        # obj_ids were opaque strings so D_I was category-agnostic.
        "obj_id_to_cat": obj_id_to_cat,
        "quality": quality,   # iter13 v13 FIX-18 — Layer C observability
    }


# ── Save ─────────────────────────────────────────────────────────────

def mine_interactions(centroids: dict, frame_width: int,
                      max_dist_frac: float, min_frames: int,
                      obj_id_to_cat: dict = None) -> list:
    """Find agent pairs close enough for long enough (D_I interaction mining).

    Proposal Sec 11.2: pairs of agent tracklets that stay within d_max
    for >= min_frames consecutive frames.

    Args:
        centroids: {obj_id: {t: (cy, cx)}} from segment_clip()
        frame_width: W pixels (for distance normalization)
        max_dist_frac: d_max as fraction of frame_width (default 0.20)
        min_frames: minimum consecutive frames for interaction (default 4)
        obj_id_to_cat: optional {obj_id: canonical_category_key} from segment_clip();
            when provided, each interaction event is annotated with `cat_a` + `cat_b`
            for downstream category-pair filtering / typed D_I tubes (#77, 2026-04-20).

    Returns:
        List of interaction events:
        [{"obj_a": str, "obj_b": str, "frames": [int],
          "cat_a": str|None, "cat_b": str|None}]  (cat_* = None when obj_id_to_cat absent)
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

            cat_a = obj_id_to_cat.get(a) if obj_id_to_cat else None
            cat_b = obj_id_to_cat.get(b) if obj_id_to_cat else None
            for run in runs:
                interactions.append({
                    "obj_a": str(a),
                    "obj_b": str(b),
                    "cat_a": cat_a,
                    "cat_b": cat_b,
                    "frames": run,
                })

    return interactions


def save_clip_masks(clip_key: str, result: dict, interactions: list,
                    masks_dir: Path, mid_frame_rgb: np.ndarray = None):
    """Save masks + centroids + per-object bboxes + obj→cat + interactions + middle frame as compressed .npz.

    `per_object_bboxes_json` ({obj_id: {t: [y1,x1,y2,x2]}}) enables m11 tight-union-bbox
    D_I tubes instead of fixed 30% centroid squares. ~5 KB/clip vs ~130 MB for full masks.
    `obj_id_to_cat_json` ({obj_id: canonical_17cat_key}) lets m11 filter D_I tubes by
    category pair + paper narrative can describe typed interactions (#77, 2026-04-20).
    """
    safe_key = clip_key.replace("/", "__")
    out_path = masks_dir / f"{safe_key}.npz"
    centroids_json = json.dumps({str(k): {str(t): list(v) for t, v in frames.items()}
                                  for k, frames in result["centroids"].items()})
    interactions_json = json.dumps(interactions)

    per_object_bboxes = {}
    for oid, frames_dict in result.get("per_object", {}).items():
        per_frame = {}
        for t, mask in frames_dict.items():
            m = np.asarray(mask, dtype=bool)
            if not m.any():
                continue
            ys, xs = np.where(m)
            per_frame[str(t)] = [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]
        if per_frame:
            per_object_bboxes[str(oid)] = per_frame
    per_object_bboxes_json = json.dumps(per_object_bboxes)

    # #77 stringify obj_ids for JSON compat (match centroids_json pattern above).
    obj_id_to_cat = result["obj_id_to_cat"]
    obj_id_to_cat_json = json.dumps({str(k): v for k, v in obj_id_to_cat.items()})

    save_dict = dict(
        agent_mask=result["agent_mask"],
        layout_mask=result["layout_mask"],
        centroids_json=np.array(centroids_json),
        interactions_json=np.array(interactions_json),
        per_object_bboxes_json=np.array(per_object_bboxes_json),
        obj_id_to_cat_json=np.array(obj_id_to_cat_json),
    )
    if mid_frame_rgb is not None:
        save_dict["mid_frame_rgb"] = mid_frame_rgb
    np.savez_compressed(out_path, **save_dict)


# ── Paper Visualizations ─────────────────────────────────────────────

def plot_overlay_per_clip(segments: dict, masks_dir: Path, tags_lookup: dict,
                         output_dir: Path):
    """Save per-clip overlay images: original | agent mask (red) | layout (blue).

    Pre-filters to ~100 clips via `utils.curate_verify.select_verify_clips()` — writes
    one plot per unique video_id, round-robin'd across (city, activity) buckets.
    At 1K runs 100/1000 PNGs; at 10K runs 100/10000 PNGs (~100× less disk + CPU).
    Uses seed=42 so m10 and m11 independently pick the SAME clip_keys for paired
    spot-check (m10 mask overlay ↔ m11 factor grid on the same clips).
    """
    overlay_dir = output_dir / "m10_overlay_verify"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    selected = select_verify_clips(list(segments.keys()), n_target=100, seed=42)
    print(f"  [m10_overlay_verify] pre-selected {len(selected)}/{len(segments)} clips "
          f"(unique video_id × round-robin city/activity, seed=42)")

    n_written = 0
    for clip_key in segments:
        if clip_key not in selected:
            continue
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
            agent_mask = np.array(Image.fromarray(agent_mask).resize((fw, fh), Image.NEAREST))

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
        n_written += 1

    print(f"  Saved: {overlay_dir}/ ({n_written} clips — pre-filtered from {len(segments)} via curate_verify.select_verify_clips)")


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

    ax1.bar(x - w / 2, mean_agents, w, label="Mean agents/clip", color=COLORS["blue"], alpha=0.85)
    ax2.bar(x + w / 2, mean_pct, w, label="Agent pixel ratio (%)", color=COLORS["red"], alpha=0.85)

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
    parser.add_argument("--interactions-only", action="store_true",
                        help="Skip SAM3 inference; read cached masks/*.npz and re-run "
                             "mine_interactions() on cached centroids + obj_id_to_cat, "
                             "updating segments.json + each .npz's interactions_json. "
                             "For iter10 v15c safer-interactions: repopulates D_I when "
                             "the prior m10 run had interaction_mining.enabled=false.")
    parser.add_argument("--train-config", required=True,
                        help="Factor dataset params YAML (e.g., configs/train/surgery_3stage_DI_encoder.yaml)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output dir. iter13 Task 3 default: "
                             "<--local-data>/m10_sam_segment/ (co-located with input).")
    parser.add_argument("--tags-json", default=None,
                        help="Path to tags.json (auto-detected from output dir if omitted)")
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
    # m10 single-owns outputs/full/m10_sam_segment/ — safe to wipe (no shared state
    # with other variants/modules). Mirror of m09b/c wipe_output_dir() — closes the
    # prompt-trigger ≠ delete-target asymmetry where partial-run state (stale
    # masks/, .m10_checkpoint.json without segments.json) survived recompute.
    _m10_out = _resolve_output_dir(args)
    wipe_output_dir(_m10_out, args.cache_policy, label=f"output_dir ({_m10_out.name})")

    # --plot: re-generate plots from existing outputs (no GPU, no SAM3)
    if args.plot:
        output_dir = _resolve_output_dir(args)
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

    # --interactions-only: iter10 v15c. CPU-only pass over cached masks/*.npz.
    # Reads centroids + obj_id_to_cat from each .npz, runs mine_interactions(),
    # rewrites each .npz with fresh interactions_json, and updates segments.json.
    # No SAM3/DINO/GPU — prior run's segmentation is reused verbatim.
    if args.interactions_only:
        # iter13 v13 FIX-10 (2026-05-07): use extends-resolving loader so the
        # variant yaml inherits factor_datasets + interaction_mining from
        # surgery_base.yaml (these blocks are NOT redefined per-variant).
        train_cfg = load_train_config_with_extends(args.train_config)
        interaction_cfg = train_cfg["interaction_mining"]
        output_dir = _resolve_output_dir(args)
        masks_dir = output_dir / "masks"
        segments_file = output_dir / "segments.json"
        if not segments_file.exists():
            print(f"FATAL: {segments_file} not found. Run m10 (without --interactions-only) first.")
            sys.exit(1)
        segments = json.load(open(segments_file))
        max_dist_frac = interaction_cfg["max_distance_frame_fraction"]
        min_frames = interaction_cfg["min_overlap_frames"]
        print(f"\n{'='*60}")
        print(f"m10 --interactions-only: mining on {len(segments)} cached clips")
        print(f"  max_distance_frame_fraction={max_dist_frac} · min_overlap_frames={min_frames}")
        print(f"  masks_dir={masks_dir}")
        print(f"{'='*60}\n")
        pbar = make_pbar(total=len(segments), desc="m10_mine_only", unit="clip")
        n_updated = 0
        n_missing_mask = 0
        n_total_interactions = 0
        for clip_key in list(segments.keys()):
            safe_key = clip_key.replace("/", "__")
            mask_file = masks_dir / f"{safe_key}.npz"
            if not mask_file.exists():
                n_missing_mask += 1
                pbar.update(1)
                continue
            with np.load(mask_file, allow_pickle=False) as data:
                save_dict = {k: np.asarray(data[k]) for k in data.files}
            centroids_raw = json.loads(str(save_dict["centroids_json"]))
            centroids = {int(k): {int(t): tuple(v) for t, v in frames.items()}
                         for k, frames in centroids_raw.items()}
            obj_id_to_cat = {}
            if "obj_id_to_cat_json" in save_dict:
                o2c_raw = json.loads(str(save_dict["obj_id_to_cat_json"]))
                obj_id_to_cat = {int(k): v for k, v in o2c_raw.items()}
            frame_width = int(save_dict["agent_mask"].shape[-1])
            interactions = mine_interactions(
                centroids,
                frame_width=frame_width,
                max_dist_frac=max_dist_frac,
                min_frames=min_frames,
                obj_id_to_cat=obj_id_to_cat,
            )
            save_dict["interactions_json"] = np.array(json.dumps(interactions))
            np.savez_compressed(mask_file, **save_dict)
            segments[clip_key]["n_interactions"] = len(interactions)
            n_updated += 1
            n_total_interactions += len(interactions)
            pbar.update(1)
        pbar.close()
        with open(segments_file, "w") as f:
            json.dump(segments, f, indent=2)
        print("\n✅ m10 --interactions-only complete")
        print(f"   Clips updated: {n_updated}/{len(segments)} (missing masks: {n_missing_mask})")
        print(f"   Total interactions mined: {n_total_interactions:,}")
        print(f"   Updated: {segments_file}")
        return

    ensure_local_data(args)
    check_gpu()

    # Load factor_datasets config — iter13 v13 FIX-10: extends-resolving loader.
    train_cfg = load_train_config_with_extends(args.train_config)
    factor_cfg = train_cfg["factor_datasets"]
    dilation_px = factor_cfg["agent_dilation_pixels"]
    min_confidence = factor_cfg["min_confidence"]
    # iter13 v13 FIX-18 (2026-05-07): SAM stability post-filter (Layer A). Pairs
    # with the lowered DINO thresholds in surgery_base.yaml. Disabled by setting
    # to 0.0 in yaml. Default 0.92 = Meta's SAM auto-mask-generator default.
    # iter15 audit (2026-05-15): FAIL LOUD on missing yaml key (CLAUDE.md
    # "No .get(key, default) on YAML"). 0.0 stays a valid yaml value meaning
    # "filter disabled"; missing key now KeyErrors instead of silently disabling.
    min_stability_score = factor_cfg["min_stability_score"]
    min_mask_area_pct = factor_cfg["min_mask_area_pct"]
    interaction_cfg = train_cfg["interaction_mining"]

    # Grounding DINO config
    dino_cfg = factor_cfg["grounding_dino"]
    dino_model_id = dino_cfg["model_id"]
    box_threshold = dino_cfg["box_threshold"]
    text_threshold = dino_cfg["text_threshold"]
    agent_taxonomy = dino_cfg["agent_taxonomy"]
    compound_prompt = build_compound_prompt(agent_taxonomy)

    # iter13 Task 3: output co-located with input via _resolve_output_dir.
    output_dir = _resolve_output_dir(args)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")

    # Clip limit (computed before guard so the completeness check knows the target).
    # iter13 v13 FIX-19 (2026-05-07): POC = mid-tier validation (default 100 from
    # pipeline.yaml poc.factor_prep). Bridges SANITY (20 — code-correctness) and
    # FULL (10K — paper). No --subset JSON required: producer iterates local_data
    # TAR shards in deterministic order, consumer stops at clip_limit. Provides
    # statistically meaningful quality percentile distributions for the new
    # factor_manifest_quality.json without ~10× FULL wall cost.
    # FULL still derives count from --subset (explicit clip_keys list) or
    # --local-data (manifest.json). Fail loud if neither works — no yaml fallback.
    # Previously POC read `train_cfg["poc_simplified"]["n_clips"]` which silently
    # capped 1000-clip val_1k runs at 100 regardless of --subset (Phase 2a
    # 100-dense tier leftover). Removed 2026-04-17 when Phase 2b moved to 1K val_1k.
    if args.SANITY:
        clip_limit = get_sanity_clip_limit("default")
    elif args.POC:
        clip_limit = get_poc_clip_limit("factor_prep")
    else:
        clip_limit = get_total_clips(
            local_data=getattr(args, "local_data", None),
            subset_file=args.subset)
        if clip_limit == 0:
            print("FATAL: FULL requires explicit --subset (JSON with clip_keys list) or "
                  "--local-data (directory with manifest.json). No yaml fallback.")
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

    # Load Grounding DINO (text → boxes)
    print(f"\nLoading Grounding DINO ({dino_model_id})...")
    dino_processor, dino_model = load_grounding_dino(dino_model_id)
    print(f"Grounding DINO loaded. Compound prompt ({len(agent_taxonomy)} cats): {compound_prompt[:120]}...")

    # Load HF Sam3TrackerVideoModel (box → mask + temporal propagation, `max_frame_num_to_track` works)
    sam_model_id = factor_cfg["sam_hf_model"]
    print(f"\nLoading HF Sam3TrackerVideoModel ({sam_model_id})...")
    sam_model, sam_processor = load_sam3_hf(sam_model_id)
    print(f"HF Sam3TrackerVideoModel loaded ({sam_model_id})")

    # Resume checkpoint
    ckpt_file = output_dir / ".m10_checkpoint.json"
    ckpt = load_json_checkpoint(ckpt_file, default={"processed_keys": []})
    processed_keys = set(ckpt["processed_keys"])
    segments = load_json_checkpoint(output_dir / "segments.json", default={})

    # Iterate clips
    subset_keys = load_subset(args.subset) if args.subset else None
    local_data = getattr(args, "local_data", None)

    print(f"\n{'='*60}")
    print(f"Grounded-SAM (DINO + SAM 3.1) — {mode}")
    print(f"Clip limit: {clip_limit}")
    print(f"Taxonomy: {len(agent_taxonomy)} categories (box>={box_threshold}, text>={text_threshold})")
    print(f"Tags: {tags_path} ({len(tags_lookup)} clips — concept_recall diagnostic only)")
    print(f"Output: {output_dir}")
    print(f"Resume: {len(processed_keys)} already done")
    print(f"{'='*60}\n")

    pbar = make_pbar(total=clip_limit, desc="m10 grounded-sam", unit="clip",
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

            # HF API takes the numpy video array directly — no JPEG frame dir needed.
            result = segment_clip(
                sam_model, sam_processor, dino_processor, dino_model,
                frames_np,
                compound_prompt=compound_prompt,
                dilation_px=dilation_px,
                min_confidence=min_confidence,
                min_mask_area_pct=min_mask_area_pct,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                min_stability_score=min_stability_score,
            )

            # Mine interactions (D_I) — now annotated with cat_a/cat_b for typed filtering (#77)
            interactions = mine_interactions(
                result["centroids"],
                frame_width=frames_np.shape[2],
                max_dist_frac=interaction_cfg["max_distance_frame_fraction"],
                min_frames=interaction_cfg["min_overlap_frames"],
                obj_id_to_cat=result["obj_id_to_cat"],
            )

            # Save masks + centroids + interactions + middle frame for overlay plots
            mid_idx = frames_np.shape[0] // 2
            save_clip_masks(clip_key, result, interactions, masks_dir,
                            mid_frame_rgb=frames_np[mid_idx])

            # Concept recall: detected agents (mapped to VLM taxonomy) vs VLM expected agents
            expected_agent_tags = get_expected_agent_tags(clip_key, tags_lookup)
            detected_vlm_tags = {
                DINO_TO_TAG.get(cat) for cat in result["detected_categories"]
            } - {None}
            n_expected = len(expected_agent_tags)
            n_detected_matched = len(expected_agent_tags & detected_vlm_tags)
            concept_recall = n_detected_matched / max(n_expected, 1)

            segments[clip_key] = {
                "n_agents": result["n_agents"],
                "n_expected": n_expected,
                "concept_recall": round(concept_recall, 3),
                "n_frames": frames_np.shape[0],
                "agent_pixel_ratio": result["agent_pixel_ratio"],
                "mean_mask_confidence": result["mean_mask_confidence"],
                "detected_categories": result["detected_categories"],
                "n_interactions": len(interactions),
                # iter13 v13 FIX-18 (2026-05-07): per-clip mask-quality observability.
                # Block produced by segment_clip's accumulators; aggregate-of-aggregates
                # lands in summary.json["quality_aggregate"] below.
                "quality": result["quality"],
            }

            processed_keys.add(clip_key)
            n_processed += 1
            pbar.update(1)

            # Checkpoint every 10 clips
            if n_processed % 10 == 0:
                save_json_checkpoint({"processed_keys": list(processed_keys)}, ckpt_file)
                save_json_checkpoint(segments, output_dir / "segments.json")
                # Periodic cache release — m10 doesn't use AdaptiveBatchSizer (per-clip
                # SAM3 sessions, not batched) but still benefits from compaction every
                # 10 clips to prevent fragmentation buildup over a 100-1000 clip run (#47).
                torch.cuda.empty_cache()

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

    # Composite quality gate — checks 4 failure modes. Thresholds calibrated for the
    # Grounded-SAM output distribution: tight precise masks, high mask confidence,
    # ~0.5-1.0% mean pixel ratio per clip with agents + ~35% legitimately-empty scenes
    # (monuments, deserted lanes). Old 0.02 pixel_ratio_min was calibrated for noisy
    # SAM3-text pipeline (avg 3-5% from false positives); deprecated here.
    gate_checks = {
        "pixel_ratio_min": mean_pixel_ratio >= 0.002,      # pipeline producing some mask
        "pixel_ratio_max": mean_pixel_ratio <= 0.50,       # not everything masked
        "mask_confidence": mean_mask_confidence >= 0.4,     # SAM is confident
        "clips_with_agents": clips_with_agents_pct >= 0.5,  # >=50% clips have agents
    }
    quality_gate = all(gate_checks.values())

    # iter13 v13 FIX-18 (2026-05-07): aggregate per-clip quality means into
    # corpus-level percentiles (Layer C). Each clip's "quality" block holds
    # per-mask aggregates; here we further aggregate the per-clip MEANS so
    # summary.json shows the distribution of clip qualities (e.g. "is the
    # bottom 10% of clips below 0.5 stability?"). Total filtered count tells
    # how many masks the new Layer A stability gate rejected corpus-wide.
    per_clip_stab_means    = [s["quality"]["stability_score"]["mean"] for s in segments.values() if "quality" in s]
    per_clip_obj_means     = [s["quality"]["object_score"]["mean"]    for s in segments.values() if "quality" in s]
    per_clip_compact_means = [s["quality"]["compactness"]["mean"]     for s in segments.values() if "quality" in s]
    per_clip_tiou          = [s["quality"]["temporal_iou_m5"]         for s in segments.values() if "quality" in s]
    n_filtered_total       = sum(s["quality"].get("n_filtered_by_stability", 0)
                                  for s in segments.values() if "quality" in s)
    quality_aggregate = {
        "stability_score": aggregate_percentiles(per_clip_stab_means),    # M1
        "object_score":    aggregate_percentiles(per_clip_obj_means),     # M2
        "compactness":     aggregate_percentiles(per_clip_compact_means), # M6
        "temporal_iou_m5": aggregate_percentiles(per_clip_tiou),          # M5
        "min_stability_score_filter": min_stability_score,
        "n_masks_filtered_by_stability": int(n_filtered_total),
    }

    summary = {
        "n_clips": len(segments),
        "n_total_agents": sum(s["n_agents"] for s in segments.values()),
        "n_total_interactions": n_interactions_total,
        "mean_agent_pixel_ratio": mean_pixel_ratio,
        "mean_concept_recall": mean_concept_recall,
        "mean_mask_confidence": mean_mask_confidence,
        "clips_with_agents_pct": round(clips_with_agents_pct, 3),
        "min_confidence_threshold": min_confidence,
        "min_stability_score": min_stability_score,   # iter13 v13 FIX-18
        "min_mask_area_pct": min_mask_area_pct,
        "elapsed_sec": elapsed,
        "pipeline": "grounded-sam",
        "dino_model": dino_model_id,
        "dino_box_threshold": box_threshold,
        "dino_text_threshold": text_threshold,
        "sam_model": factor_cfg["sam_model"],
        "agent_taxonomy": agent_taxonomy,
        "quality_gate": "PASS" if quality_gate else "FAIL",
        "quality_gate_checks": {k: "PASS" if v else "FAIL" for k, v in gate_checks.items()},
        # iter13 v13 FIX-18 (2026-05-07): corpus-wide aggregate of per-clip quality
        # means. Read this to assess factor-quality regression across re-runs.
        "quality_aggregate": quality_aggregate,
    }
    save_json_checkpoint(summary, output_dir / "summary.json")

    # Cleanup checkpoint
    if ckpt_file.exists():
        ckpt_file.unlink()

    # Paper visualizations
    plot_overlay_per_clip(segments, masks_dir, tags_lookup, output_dir)
    plot_agent_stats(segments, tags_lookup, output_dir)

    # iter13 v13 FIX-25 (2026-05-07): TAR packing moved out of m10. m10 produces
    # only raw masks/*.npz here. `python src/utils/hf_outputs.py upload-data`
    # tars + uploads + cleans up raws in one shot (single source of truth for
    # HF transit format). m11 reads masks/*.npz directly at runtime; the tar
    # shards never need to exist mid-pipeline.

    # Quality gate: FATAL if composite check fails (Rule 33: quality gates in Python, not shell)
    if not quality_gate:
        failed = [k for k, v in gate_checks.items() if not v]
        print(f"FATAL: Quality gate FAILED — {len(failed)} check(s):")
        print(f"  pixel_ratio_min:  mean={mean_pixel_ratio:.3f} (need >=0.002) {'FAIL' if not gate_checks['pixel_ratio_min'] else 'OK'}")
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

    # Force exit: HF Sam3 video sessions + DINO hold GPU buffers.
    del sam_model, sam_processor, dino_model, dino_processor
    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL (unhandled): {e}")
        traceback.print_exc()
        os._exit(1)  # guarantee exit even with SAM3 async threads
