"""Frame-resolved features + temporal transforms for the m12f encoder-temporal metric suite.

Encoder-temporal metrics (TOV / AoT / Pace / TCC — plan_metrics_temporal.md §6) read the
ENCODER's feature space, not the predictor's outputs. V-JEPA's 3D attention mixes time across
frames, so the temporal-order signal cannot be recovered AFTER the encoder — every transform
(reverse / permute / stride-resample) MUST be applied to the (B, T, C, H, W) frame tensor
BEFORE the encoder forward.

Public API:
  - forward_per_frame(encoder, batch, num_frames, tubelet_size) → (B, T_eff, D)
      spatial-mean per tubelet step (T_eff = num_frames // tubelet_size). Required by et_tcc
      for soft-NN alignment; et_aot/et_tov/et_pace further mean over T_eff to feed the
      classifier head with one (B, D) vector per clip variant.
  - reverse_frames(batch)         → (B, T, C, H, W)  time-reversed copy        (et_aot)
  - permute_frames(batch, idx)    → (B, T, C, H, W)  frame-permuted            (et_tov)
  - stride_indices(num_frames, stride, source_frames) → LongTensor[num_frames] (et_pace)

Eval-safe: NEW file. Not imported by any live-eval script (run_eval.sh / probe_*.py /
existing utils). CPU-checked; GPU validation deferred to post-eval (host RAM ~97% from
the live eval blocks torch import now).

Gold:
  - V-JEPA 2.1 (Bardes et al. 2025) — joint 3D tubelet encoder, T frames → T_eff tubelet steps
  - TCC (Dwibedi et al. CVPR19) — per-frame embeddings for soft-NN cycle-back alignment
  - Shuffle&Learn (Misra et al. ECCV16) / VCOP (Xu et al. CVPR19) — permuted-frame discriminators
  - Arrow-of-Time (Wei et al. CVPR18) — forward-vs-reversed binary discriminator
  - Pace (Wang et al. ECCV20) — playback-rate discriminator over oversampled stride windows
"""
import torch

from utils.predictor_eval import to_pixel

# iter18 W7 (PLR2004): tensor-rank contracts.
_VIDEO_RANK = 5   # (B, T, C, H, W)
_FEATS_RANK = 3   # (B, tokens, D)


@torch.no_grad()
def forward_per_frame(encoder, batch, num_frames, tubelet_size):
    """Encode (B, T, C, H, W) → per-tubelet-step features (B, T_eff, D).

    Uses `utils.predictor_eval.to_pixel` to cast → bf16 and permute (B,T,C,H,W) → (B,C,T,H,W)
    (channels-first for the encoder's 3D conv patch-embed) — single-source the dtype/layout
    with m12e (no per-module re-derivation per src/CLAUDE.md "SHARED DERIVATION").
    T_eff = num_frames // tubelet_size. Spatial patch tokens are mean-pooled per tubelet step;
    the temporal axis is preserved (required by TCC; collapsed downstream by AoT/TOV/Pace).
    Output cast back to float32 for the downstream head (.float()) — bf16 outputs cause silent
    precision loss in the CrossEntropy / Kendall's-τ stages.
    FAIL LOUD if the token count doesn't factor into (T_eff × n_spatial) — that indicates an
    encoder/tubelet config mismatch.
    """
    return pool_per_frame(forward_tokens(encoder, batch), num_frames, tubelet_size)


def forward_tokens(encoder, batch):
    """The shared encoder forward: (B,T,C,H,W) → (B, N_tok, D) fp32 on the encoder's device (CUDA).
    iter19 2026-07-09: factored OUT of forward_per_frame so ONE forward can feed BOTH the per-tubelet
    pool (below) AND the opt-in probe_action per-frame cache emit — so tcc's per-frame features equal
    that cache BYTE-FOR-BYTE (same tokens, same device: the pool must stay on CUDA — a CPU reduction of
    the same tokens differs in the last ULP). Byte-identical to the old inline forward (pure split)."""
    if batch.ndim != _VIDEO_RANK:
        raise RuntimeError(f"forward_tokens expects (B,T,C,H,W); got {tuple(batch.shape)}")
    pixel = to_pixel(batch)                 # (B,C,T,H,W) bf16 on CUDA
    out = encoder(pixel).float()
    if not torch.is_tensor(out) or out.ndim != _FEATS_RANK:
        raise RuntimeError(
            f"encoder forward must return 3-D (B,N_tok,D); got {type(out).__name__} "
            f"shape={getattr(out, 'shape', None)}")
    return out


def pool_per_frame(tokens, num_frames, tubelet_size):
    """(B, N_tok, D) encoder tokens → (B, T_eff, D) per-tubelet spatial-mean, ON THE TOKENS' DEVICE.
    T_eff = num_frames // tubelet_size. FAIL LOUD if N doesn't factor into (T_eff × n_spatial)."""
    if num_frames % tubelet_size != 0:
        raise RuntimeError(
            f"num_frames={num_frames} not divisible by tubelet_size={tubelet_size}")
    B, N, D = tokens.shape
    t_eff = num_frames // tubelet_size
    if N % t_eff != 0:
        raise RuntimeError(
            f"token count {N} not divisible by T_eff={t_eff} (num_frames={num_frames}, "
            f"tubelet_size={tubelet_size}) — check encoder patch/tubelet alignment")
    n_spatial = N // t_eff
    return tokens.view(B, t_eff, n_spatial, D).mean(dim=2).contiguous()


def reverse_frames(batch):
    """(B, T, C, H, W) → time-reversed copy. For Arrow-of-Time (Wei CVPR18)."""
    if batch.ndim != _VIDEO_RANK:
        raise RuntimeError(f"reverse_frames expects 5-D (B,T,C,H,W); got {tuple(batch.shape)}")
    return batch.flip(dims=[1]).contiguous()


def permute_frames(batch, perm_idx):
    """(B, T, C, H, W) → frame-permuted along T using an explicit (T,) LongTensor.
    Caller supplies the permutation (m12f orchestrator generates one per permutation-id for
    TOV/VCOP). Same permutation applied to every clip in the batch."""
    if batch.ndim != _VIDEO_RANK:
        raise RuntimeError(f"permute_frames expects 5-D (B,T,C,H,W); got {tuple(batch.shape)}")
    if not torch.is_tensor(perm_idx) or perm_idx.ndim != 1:
        raise RuntimeError(f"perm_idx must be 1-D LongTensor; got {perm_idx!r}")
    T = batch.shape[1]
    if perm_idx.numel() != T:
        raise RuntimeError(f"perm_idx length {perm_idx.numel()} != T={T}")
    return batch.index_select(1, perm_idx.to(batch.device).long()).contiguous()


def stride_indices(num_frames, stride, source_frames):
    """Index tensor selecting `num_frames` frames at effective `stride` from a `source_frames`
    -long source clip. For Pace classification: source clip is decoded with OVERSAMPLE frames
    (T_src = num_frames × max_stride) so every stride fits non-wrap. If source is too short,
    indices cycle via modulo (the orchestrator logs the cyclic wrap; not the gold-standard
    Pace setup, but a defined fallback)."""
    if stride < 1:
        raise RuntimeError(f"stride must be >= 1; got {stride}")
    if num_frames < 1 or source_frames < 1:
        raise RuntimeError(
            f"num_frames ({num_frames}) and source_frames ({source_frames}) must be >= 1")
    idx = torch.arange(num_frames) * stride
    if int(idx.max().item()) >= source_frames:
        idx = idx % source_frames
    return idx.long()


__all__ = ["forward_per_frame", "forward_tokens", "pool_per_frame",
           "reverse_frames", "permute_frames", "stride_indices"]
