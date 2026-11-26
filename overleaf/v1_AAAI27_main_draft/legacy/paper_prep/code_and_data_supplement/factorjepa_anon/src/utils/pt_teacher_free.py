"""Metric #4 — Teacher-vs-Free-run Gap (predictor latent-L1). ★ top surgery-win bet.

Run the rollout twice — teacher-forced (REAL context re-encoded each step) vs free-running
(the predictor's OWN predicted context) — and take the per-clip gap (free − teacher) averaged
over horizons = EXPOSURE BIAS: how much error inflates when the predictor must consume its own
mistakes. LOWER gap = recovers better / less compounding. Orthogonal to raw drift (#1 measures
the free-run growth rate; this ISOLATES the free-vs-teacher difference).
Gold: Scheduled Sampling (Bengio '15); BAgger (2025).

GPU-VALIDATION REQUIRED post-eval (§3.1).
"""
import numpy as np

from utils.predictor_eval import full_target_h, rollout_l1_per_horizon, to_pixel


def compute(encoder, predictor, batch, num_frames, h_full=None) -> np.ndarray:
    """batch (B,T,3,H,W) cpu → per-clip mean(free − teacher) gap over horizons (B,).

    h_full: optional batch-shared full_target_h (m12e --metric all reuses one encode; bit-identical).
    None → compute per-metric (current path)."""
    pixel = to_pixel(batch)
    if h_full is None:                       # iter19: reuse a batch-shared h across metrics if passed
        h_full = full_target_h(encoder, pixel)  # h-memo: free + teacher rollouts share one target encode (None when off)
    free = rollout_l1_per_horizon(encoder, predictor, pixel, num_frames, free_running=True, h_full=h_full)
    teach = rollout_l1_per_horizon(encoder, predictor, pixel, num_frames, free_running=False, h_full=h_full)
    return (free - teach).mean(axis=1)
