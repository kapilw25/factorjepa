"""Metric #2 — Causal Future-Block prediction (predictor latent-L1).

Mask the FUTURE temporal half only (m_pred = slots [Tp/2, Tp)) and predict it from the PAST
half (m_enc = slots [0, Tp/2)). Strictly causal — orthogonal to future_mse's random spacetime
mask (direction-agnostic). Per-clip mean-L1; LOWER = better causal prediction.
Gold: Arrow-of-Time (Wei CVPR18); V-JEPA causal masking.

GPU-VALIDATION REQUIRED post-eval (§3.1) — predicting a full temporal half is a larger m_pred
than future_mse's small blocks; confirm the predictor handles it + tune if OOM.
"""
import numpy as np  # noqa: F401  (kept for dtype parity with siblings)

from utils.predictor_eval import (
    expand_mask, masked_predict_l1, temporal_token_idx, to_pixel, token_grid,
)


def compute(encoder, predictor, batch, num_frames, h_full=None) -> "np.ndarray":
    """batch (B,T,3,H,W) cpu → per-clip causal-future L1 (B,).

    h_full: optional batch-shared full_target_h(encoder, pixel) (m12e --metric all reuses one encode across
    metrics; bit-identical). None → masked_predict_l1 encodes internally (current path)."""
    Tp, _, _, _ = token_grid(num_frames)
    half = Tp // 2
    pixel = to_pixel(batch)
    b = pixel.shape[0]
    m_enc = expand_mask(temporal_token_idx(num_frames, range(0, half)), b)
    m_pred = expand_mask(temporal_token_idx(num_frames, range(half, Tp)), b)
    l1, _, _ = masked_predict_l1(encoder, predictor, pixel, m_enc, m_pred, h_full=h_full)
    return l1
