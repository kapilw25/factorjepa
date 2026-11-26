"""Metric #6 — Temporal-Order Sensitivity (predictor latent-L1).

Predict the LAST temporal slot from all earlier slots, twice: once with frames in correct
temporal order, once with the frames temporally SHUFFLED (same mask positions). Per-clip
ΔL1 = shuffled_L1 − ordered_L1 = how much the predictor RELIES on correct temporal order
(>0 → order-dependent; ~0 → order-blind). Discriminative-temporal, orthogonal to MSE.
Gold: Shuffle&Learn (Misra ECCV16); arrow-of-time insensitivity studies.

GPU-VALIDATION REQUIRED post-eval (§3.1).
"""
import numpy as np  # noqa: F401
import torch

from utils.predictor_eval import (
    PT_SEED, expand_mask, masked_predict_l1, temporal_token_idx, to_pixel, token_grid,
)


def compute(encoder, predictor, batch, num_frames) -> "np.ndarray":
    """batch (B,T,3,H,W) cpu → per-clip ΔL1 (shuffled − ordered) (B,)."""
    Tp, _, _, _ = token_grid(num_frames)
    pixel = to_pixel(batch)
    b = pixel.shape[0]
    m_enc = expand_mask(temporal_token_idx(num_frames, range(0, Tp - 1)), b)
    m_pred = expand_mask(temporal_token_idx(num_frames, [Tp - 1]), b)

    l1_ord, _, _ = masked_predict_l1(encoder, predictor, pixel, m_enc, m_pred)

    # Shuffle the temporal frame order (axis=2 of pixel (B,3,T,H,W)); fixed perm for determinism.
    g = torch.Generator().manual_seed(PT_SEED)
    perm_t = torch.randperm(pixel.shape[2], generator=g)
    pixel_sh = pixel[:, :, perm_t, :, :].contiguous()
    l1_sh, _, _ = masked_predict_l1(encoder, predictor, pixel_sh, m_enc, m_pred)

    return l1_sh - l1_ord
