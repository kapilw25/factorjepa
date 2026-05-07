"""Mask-quality metrics for m10/m11 — no ground-truth required.

iter13 v13 FIX-18 (2026-05-07): observability pack for factor-quality auditing.

Why this exists
---------------
Visual inspection showed m10 detected only 9 agents on a chennai walking-tour clip
with ~6 visible humans, leaving D_A nearly black and D_L indistinguishable from the
original. The factor decomposition is degenerate when detection recall is low — and
without ground-truth labels, there's no automatic way to flag the bad clips.

This module supplies four cheap, GT-free signals so segments.json + factor_manifest.json
can carry per-clip quality alongside the existing agent counts:

    M1 — stability_score(mask)         : robustness to ±N-pixel dilate/erode (SAM-style)
    M5 — temporal_iou_per_object(...)  : IoU(mask[t], mask[t+1]) per tracked agent
    M6 — compactness(mask)             : 4π·area / perimeter² (isoperimetric)
    +   aggregate_percentiles(values)  : standard {mean, p10, p50, p90, n} reducer

M2 (per-mask object-confidence) is REUSED from m10's existing `mean_mask_confidence`
which already collects sigmoid(object_score_logits) — see m10_sam_segment.py:412.
M3 (VLM-tag-recall proxy) and M4 (cross-detector ensemble) are explicitly out of scope.

All metrics are pure numpy + cv2 — no new dependencies.
"""
from __future__ import annotations

import cv2
import numpy as np


def stability_score(mask: np.ndarray, dilation_offsets: tuple = (1, 2)) -> float:
    """SAM-style mask stability under threshold-cutoff perturbation.

    Computes mean IoU(mask, dilate/erode(mask, ±N pixels)) across the supplied
    offsets. Higher = mask survives binarization-cutoff jitter; Meta uses ≥ 0.92
    as the default acceptance filter in SAM's `automatic_mask_generator.py`.

    Source: SAM `mask_under_threshold_perturbation_iou` semantics — see
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py

    Args:
        mask: (H, W) bool / 0-1 numpy array. Empty masks return 0.0.
        dilation_offsets: pixel radii to perturb at. Default (1, 2) → 4 IoU
            samples per mask (dilate at 1, erode at 1, dilate at 2, erode at 2).

    Returns:
        float ∈ [0, 1]. NaN-safe (zero-area masks → 0.0).
    """
    if mask is None or mask.sum() == 0:
        return 0.0
    m = mask.astype(np.uint8)
    ious = []
    for off in dilation_offsets:
        k = np.ones((2 * off + 1, 2 * off + 1), np.uint8)
        for variant in (cv2.dilate(m, k), cv2.erode(m, k)):
            inter = int(np.logical_and(m, variant).sum())
            union = int(np.logical_or(m, variant).sum())
            if union > 0:
                ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def temporal_iou_per_object(per_object: dict) -> float:
    """Mean adjacent-frame IoU averaged across tracked objects.

    For each obj_id with masks at frames {t0, t1, ...}, compute IoU(mask[t_i],
    mask[t_{i+1}]) over consecutive pairs, average over the object, then average
    across objects. Stable agents have IoU > 0.7 frame-to-frame; jittery /
    fragmented detections trend toward 0.

    Args:
        per_object: {obj_id: {frame_idx: mask_ndarray}} — direct match for m10's
            tracker dict at m10_sam_segment.py:449.

    Returns:
        float ∈ [0, 1]. Empty input → 0.0. Single-frame objects skipped.
    """
    object_means = []
    for _obj_id, by_frame in per_object.items():
        frames = sorted(by_frame.keys())
        if len(frames) < 2:
            continue
        pair_ious = []
        for i in range(len(frames) - 1):
            m1 = np.asarray(by_frame[frames[i]], dtype=bool)
            m2 = np.asarray(by_frame[frames[i + 1]], dtype=bool)
            inter = int(np.logical_and(m1, m2).sum())
            union = int(np.logical_or(m1, m2).sum())
            if union > 0:
                pair_ious.append(inter / union)
        if pair_ious:
            object_means.append(float(np.mean(pair_ious)))
    return float(np.mean(object_means)) if object_means else 0.0


def compactness(mask: np.ndarray) -> float:
    """Isoperimetric compactness = 4π · area / perimeter².

    Sums area + perimeter across all connected components in the mask.
    Pedestrian masks typically score 0.3–0.6; fragmented / noisy boundaries
    score < 0.1. Range capped at 1.0 (perfect circle).

    Args:
        mask: (H, W) bool / 0-1 numpy array. Empty masks return 0.0.

    Returns:
        float ∈ (0, 1].
    """
    if mask is None or mask.sum() == 0:
        return 0.0
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    area = float(sum(cv2.contourArea(c) for c in contours))
    perim = float(sum(cv2.arcLength(c, True) for c in contours))
    if perim < 1e-6 or area < 1e-6:
        return 0.0
    return float(min(1.0, 4 * np.pi * area / (perim ** 2)))


def aggregate_percentiles(values) -> dict:
    """Standard {mean, p10, p50, p90, n} summary used by both m10 + m11.

    Empty inputs return zeros so downstream JSON consumers see a uniform schema.

    Args:
        values: any iterable of floats (numpy array, list, tuple).

    Returns:
        dict with keys mean, p10, p50, p90, n. All floats except n (int).
    """
    if values is None:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "n": 0}
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "n": 0}
    return {
        "mean": float(arr.mean()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "n": int(arr.size),
    }
