"""Metric #1 — Multi-step Rollout Drift (predictor latent-L1). ★ top surgery-win bet.

Free-running iterated rollout: predict slot k from {0..k-1}, feeding the PREDICTED features
back as context so errors COMPOUND. Per-clip OLS slope of L1 vs horizon h = drift slope (how
fast the world-model's error grows over the rollout). LOWER slope = more stable dynamics.
Orthogonal to future_mse (single masked step) and tdist (single-shot, no feedback).
Gold: V-JEPA 2-AC rollout loss (Meta 2025); LIVE / Rolling-Forcing drift (2025).

GPU-VALIDATION REQUIRED post-eval (§3.1) — confirm L1 grows monotonically with horizon.
"""
import numpy as np

from utils.predictor_eval import perclip_slope, rollout_l1_per_horizon, to_pixel


def compute(encoder, predictor, batch, num_frames, h_full=None) -> np.ndarray:
    """batch (B,T,3,H,W) cpu → per-clip drift slope (B,).

    h_full: optional batch-shared full_target_h (m12e --metric all reuses one encode; bit-identical).
    None → rollout_l1_per_horizon encodes internally (current path)."""
    pixel = to_pixel(batch)
    L = rollout_l1_per_horizon(encoder, predictor, pixel, num_frames, free_running=True, h_full=h_full)  # (B, Tp-1)
    horizons = list(range(1, L.shape[1] + 1))
    return perclip_slope(L, horizons)
