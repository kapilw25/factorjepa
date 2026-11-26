"""Temporal Cycle-Consistency encoder-temporal metric (Dwibedi et al. CVPR19). TRAINING-FREE.

Mechanism: per-frame embeddings from each encoder → soft-NN alignment between same-action
clip pairs → cycle-back error (A→B→A) + Kendall's τ on hard-NN frame ordering. NO trainable
head — pure geometry. surgery≫pretrain hypothesis: lower cycle-back error and higher Kendall's
τ = encoder produces feature trajectories where corresponding frames of the same action align
across clip pairs, indicating temporal-structure awareness.

Pairing strategy (orchestrator m12f):
  - Pair test clips that share the same action label (loaded from action_labels.json).
  - Skip singleton classes (need ≥2 clips per action to form a pair).
  - For each pair (A, B): compute_pair returns (cycle_back_err, kendalls_tau). Per-clip score
    = mean over all pairs that clip participates in.

Outputs (m12f, --metric tcc):
  per_clip_tcc_cycle.npy   ← per-test-clip mean cycle-back error (lower = better)
  per_clip_tcc_tau.npy     ← per-test-clip mean Kendall's τ      (higher = better)
  aggregate_tcc.json       ← both metrics' mean/std/BCa CI + n_test + n_pairs + temperature

Gold:
  - Dwibedi et al. "Temporal Cycle-Consistency Learning" CVPR 2019
    github.com/google-research/google-research/tree/master/tcc (TF original)
  - June01/tcc_Temporal_Cycle_Consistency (PyTorch port)
  - Kendall, M.G. "A New Measure of Rank Correlation" Biometrika 1938

GPU-VALIDATION REQUIRED post-eval. CPU-checked (py_compile + ruff F,E9) only.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

from utils.per_frame_features import forward_per_frame

# iter18 W7 (PLR2004): contracts.
_FEATS_RANK = 2
_MIN_UNIQUE = 2


@torch.no_grad()
def compute_per_frame(encoder, batch, num_frames, tubelet_size):
    """(B, T, C, H, W) → (B, T_eff, D) per-tubelet-step embeddings."""
    return forward_per_frame(encoder, batch, num_frames, tubelet_size).float().cpu()


def _soft_nn_indices(feats_a, feats_b, *, temperature):
    """For each frame i in A, return the soft-NN frame index in B.
    soft_idx_i = Σ_j softmax_j(<a_i, b_j> / τ) · j   (Dwibedi eq. 1).
    feats_a, feats_b: (T_eff, D). Returns (T_eff,) soft indices."""
    if feats_a.ndim != _FEATS_RANK or feats_b.ndim != _FEATS_RANK:
        raise RuntimeError(
            f"_soft_nn_indices expects 2-D per-clip; got A {feats_a.shape}, B {feats_b.shape}")
    if temperature <= 0:
        raise RuntimeError(f"temperature must be > 0; got {temperature}")
    sim = feats_a @ feats_b.T / temperature                   # (T_a, T_b)
    w = F.softmax(sim, dim=1)                                 # (T_a, T_b)
    j_range = torch.arange(feats_b.shape[0], device=feats_a.device, dtype=feats_a.dtype)
    return (w * j_range.unsqueeze(0)).sum(dim=1)              # (T_a,)


def cycle_back_error(feats_a, feats_b, *, temperature):
    """A→B→A cycle: for each frame i in A, soft-NN to B → soft-NN back to A. Mean |i - cycle_i|.
    Returns a scalar (Python float). Lower = better temporal correspondence."""
    soft_b_idx = _soft_nn_indices(feats_a, feats_b, temperature=temperature)
    # Round to nearest B-frame index for the A-back step (Dwibedi uses soft both ways but the
    # straightforward variant is fine for the scaffold; promotion to full-soft is a §3.3 knob).
    b_idx_hard = soft_b_idx.round().long().clamp_(0, feats_b.shape[0] - 1)
    selected_b = feats_b[b_idx_hard]                          # (T_a, D)
    soft_a_idx = _soft_nn_indices(selected_b, feats_a, temperature=temperature)
    i_range = torch.arange(feats_a.shape[0], device=feats_a.device, dtype=feats_a.dtype)
    return float((soft_a_idx - i_range).abs().mean().item())


def kendalls_tau_alignment(feats_a, feats_b):
    """Hard-NN alignment from A → B (one B-index per A-frame), then Kendall's τ vs the
    monotonic identity ordering. Higher τ (range [-1, 1]) = more order-preserving alignment.
    Returns a Python float, or NaN for a DEGENERATE alignment (all A-frames map to one B-index):
    Kendall's τ is mathematically UNDEFINED when a ranking has zero variance — scipy.kendalltau
    itself returns NaN for constant input. A degenerate pair (the encoder produced collinear
    per-frame features — expected for static clips, especially FROZEN encoders) is a valid DATA
    edge case, NOT a code bug, so it must NOT abort the whole encoder's eval. The m12f orchestrator
    excludes NaN-τ pairs via nanmean and REPORTS the degenerate count so it stays visible."""
    sim = feats_a @ feats_b.T                                # (T_a, T_b)
    hard_b_idx = sim.argmax(dim=1).cpu().numpy()
    if np.unique(hard_b_idx).size < _MIN_UNIQUE:
        return float("nan")                                  # τ undefined for a constant ranking
    tau, _p = kendalltau(np.arange(hard_b_idx.size), hard_b_idx)
    return float(tau)


def compute_pair(feats_a, feats_b, *, temperature):
    """One pair → (cycle_back_err, kendalls_tau). Both metrics from the same (A, B) features."""
    if feats_a.shape != feats_b.shape:
        raise RuntimeError(
            f"compute_pair: feats_a {feats_a.shape} vs feats_b {feats_b.shape} shape mismatch")
    return cycle_back_error(feats_a, feats_b, temperature=temperature), \
           kendalls_tau_alignment(feats_a, feats_b)


def _kendall_tau_b_vs_arange(x):
    """Batched Kendall's τ-b of each ROW of x (M, T) against the identity ranking arange(T) —
    torch-native replica of scipy.stats.kendalltau(arange(T), row) (iter18 #8):
      τ_b = S / √((n0 − n1)·(n0 − n2)) ; S = Σ_{i<j} sign(x_j − x_i)·sign(j − i)
      n0 = T(T−1)/2 ; n1 = tied pairs in x ; n2 = tied pairs in arange = 0
    sign(j − i) = +1 for i<j, so S reduces to Σ sign(x_j − x_i). A constant row has n1 = n0 →
    denominator 0 → NaN, exactly scipy's behaviour for a zero-variance ranking (and the
    unique<2 guard in kendalls_tau_alignment)."""
    if x.ndim != _FEATS_RANK:
        raise RuntimeError(f"_kendall_tau_b_vs_arange expects (M, T); got {tuple(x.shape)}")
    T = x.shape[1]
    iu = torch.triu_indices(T, T, offset=1, device=x.device)
    dij = x[:, iu[1]].float() - x[:, iu[0]].float()          # x_j − x_i for all i<j → (M, n0)
    s = torch.sign(dij).sum(dim=1)
    n1 = (dij == 0).sum(dim=1).float()
    n0 = float(iu.shape[1])
    denom = ((n0 - n1) * n0).sqrt()
    return torch.where(denom > 0, s / denom, torch.full_like(denom, float("nan")))


@torch.no_grad()
def compute_pairs_batched(feats_a, feats_b, *, temperature):
    """Batched compute_pair over M pairs: (M, T, D) × (M, T, D) → (cycle (M,), tau (M,)).
    Identical math to the scalar path — the per-pair T×T similarity / softmax / gather chain is
    stacked into ONE batched matmul sequence (iter18 #8; the reference TCC implementations
    compute soft-NN alignment batched the same way — Dwibedi CVPR19 eq. 1,
    github.com/google-research/google-research/tree/master/tcc). Degenerate (constant-ranking)
    pairs → τ = NaN, matching kendalls_tau_alignment; cycle-back is never NaN.
    Parity-tested against compute_pair (scipy τ-b included) in the m12f CPU suite."""
    if feats_a.shape != feats_b.shape or feats_a.ndim != _FEATS_RANK + 1:
        raise RuntimeError(
            f"compute_pairs_batched expects matching (M,T,D); got {tuple(feats_a.shape)} vs "
            f"{tuple(feats_b.shape)}")
    if temperature <= 0:
        raise RuntimeError(f"temperature must be > 0; got {temperature}")
    M, T, D = feats_a.shape
    j = torch.arange(T, dtype=feats_a.dtype, device=feats_a.device)
    sim_ab = feats_a @ feats_b.transpose(1, 2)                            # (M,T,T) UNSCALED (τ uses raw)
    soft_b = (F.softmax(sim_ab / temperature, dim=2) * j).sum(dim=2)      # (M,T) soft index into B
    b_hard = soft_b.round().long().clamp_(0, T - 1)
    sel_b = torch.gather(feats_b, 1, b_hard.unsqueeze(-1).expand(-1, -1, D))   # (M,T,D)
    soft_a = (F.softmax(sel_b @ feats_a.transpose(1, 2) / temperature, dim=2) * j).sum(dim=2)
    cycle = (soft_a - j).abs().mean(dim=1)                                # (M,)
    tau = _kendall_tau_b_vs_arange(sim_ab.argmax(dim=2))                  # hard-NN, no temperature
    return cycle, tau


__all__ = [
    "compute_per_frame", "cycle_back_error", "kendalls_tau_alignment", "compute_pair",
    "compute_pairs_batched",
]
