"""Shared V-JEPA encoder+predictor eval primitive for the m12e predictor-temporal suite.

NON-STANDALONE helper (imported by src/utils/pt_*.py + src/m12e_predictor_temporal.py).
Self-contained NOW (copies probe_future_mse.py's module-local loaders + masked-predict L1
core) — the dedup refactor of probe_future_mse → import this is DEFERRED post-eval (editing a
live-invoked script). See iter/iter16_ablations/plan_metrics_temporal.md §2.1.

Gold standard: facebookresearch/vjepa2/app/vjepa_2_1 (encoder hierarchical + predictor).
GPU-VALIDATION REQUIRED post-eval (§3.1) — written but not yet GPU-smoked (eval holds the GPU).

PUBLIC API
    load_encoder_predictor(ckpt_path, num_frames) -> (encoder, predictor, embed_dim_concat)
    build_mask_gen(num_frames, *, npred=None, spatial_scale=None, ...) -> _MaskGenerator
    token_grid(num_frames) -> (Tp, Hp, Wp, spatial_per_frame)
    temporal_token_idx(num_frames, t_slots) -> LongTensor of token indices for those temporal slots
    to_pixel(batch) -> (B,3,T,H,W) bf16 cuda
    masked_predict_l1(encoder, predictor, pixel, m_enc, m_pred) -> (per_clip_l1 (B,), out, h_target)
    bootstrap_ci  (re-exported from utils.bootstrap)
"""
import contextlib
import os
import sys

import numpy as np
import torch

from utils.bootstrap import bootstrap_ci  # noqa: F401  (re-export for the pt_*.py + orchestrator)
from utils.config import (
    get_model_config,
    get_pipeline_config,
    load_train_config_with_extends,
)
from utils.frozen_features import (
    resolve_encoder_state_dict,
    resolve_predictor_state_dict,
)
from utils.gpu_batch import cuda_cleanup
from utils.vjepa2_imports import (
    get_apply_masks,
    get_mask_generator,
    get_vit_by_arch,
    get_vit_predictor,
    get_vit_predictor_2_1,
)

# iter18 W7 (PLR2004): semantic named constants.
_PRED_TUPLE_LEN = 2      # predictor returns (z_pred, z_context) when predict_all
_MIN_TEMPORAL_SLOTS = 2  # rollout needs ≥2 temporal slots
_MIN_LOADED_FRAC = 0.5   # ckpt param-coverage floor — below = random init garbage

# iter18 2026-06-07 — h-memoization (V-JEPA "amortize target computation"): when PT_H_MEMO=1, the
# mask-INDEPENDENT full-forward target h=encoder(pixel) is computed ONCE per batch (full_target_h)
# and reused across a metric's mask sweep (tdist Δt, maskratio r, teacher_free's 2 rollouts) instead
# of recomputed each masked_predict_l1 call. Bit-identical (deterministic eval/no_grad forward of the
# same pixel) — but ships FLAG-GATED OFF until a GPU parity smoke confirms it; default = exact prior path.
_H_MEMO = os.environ.get("PT_H_MEMO", "0") == "1"

# ── Constants (single-sourced from yaml — copied from probe_future_mse.py) ──
_PCFG = get_pipeline_config()
_MODEL_CFG = get_model_config(None)["model"]               # None → default vjepa2_1.yaml
NUM_FRAMES_DEFAULT = _PCFG["probe"]["num_frames"]
# Geometry constants are IDENTICAL across all predictor-capable backbones (vitG/vitg/2.0_vitg
# = crop 384, patch 16, tubelet 2) → safe at module level. Per-model dims (arch/embed_dim/
# pred_*/num_mask_tokens/n_output_distillation/predict_all) are read PER-CALL from model_cfg in
# the loaders below (iter17 WS-B3 — these were ViT-G-hardcoded here, crashing non-G backbones).
PATCH_SIZE = _MODEL_CFG["patch_size"]
TUBELET_SIZE = _MODEL_CFG["tubelet_size"]
CROP = _MODEL_CFG["crop_size"]

_MASK0 = load_train_config_with_extends(_PCFG["config_paths"]["base_optimization"])["mask"][0]   # iter18 H1: yaml pointer
DEFAULT_SPATIAL_SCALE = tuple(_MASK0["spatial_scale"])
DEFAULT_TEMPORAL_SCALE = tuple(_MASK0["temporal_scale"])
DEFAULT_ASPECT_RATIO = tuple(_MASK0["aspect_ratio"])
DEFAULT_NUM_BLOCKS = _MASK0["num_blocks"]

# iter16 §3.4: m12e predictor-temporal sweep params single-sourced from pipeline.yaml
# (were hardcoded _DELTAS/_RATIOS/_SEED in pt_tdist/pt_maskratio/pt_order).
_PT = _PCFG["probe"]["predictor_temporal"]   # yaml KEY (rewriter had aliased it via artifact() — same string, wrong semantics)
PT_DELTAS = tuple(_PT["deltas"])              # pt_tdist Δt sweep
PT_MASK_RATIOS = tuple(_PT["mask_ratios"])    # pt_maskratio sweep
PT_SEED = _PT["seed"]                         # pt_maskratio + pt_order deterministic-partition seed


# ── Loaders (encoder hierarchical + predictor) ─────────────────────────
def load_encoder_only(ckpt_path, num_frames, model_cfg=None):
    """Load ONLY the V-JEPA encoder (NO predictor build) from a .pt; arch/dims from model_cfg.

    iter17 WS-B3: was ViT-G-hardcoded; now reads arch/crop/embed_dim/n_output_distillation from
    get_model_config(model_cfg) (None → vjepa2_1.yaml ViT-G, back-compat). Returns
    (encoder, ckpt, embed_dim_concat). 2.1 (n_distill>1) → deep-sup hierarchical concat =
    embed_dim*n_distill; 2.0 (n_distill=1) → single-output encoder, concat = embed_dim. `ckpt` is
    returned so load_encoder_predictor reuses this single torch.load. Used encoder-only by m12f.
    """
    from pathlib import Path
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        sys.exit(f"FATAL: ckpt not found: {ckpt_path}")
    mc = get_model_config(model_cfg)["model"]
    crop, n_distill = mc["crop_size"], mc["n_output_distillation"]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)

    enc_sd = resolve_encoder_state_dict(ckpt)
    enc_sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in enc_sd.items()}
    encoder = get_vit_by_arch(mc["arch"])(
        img_size=(crop, crop), patch_size=mc["patch_size"], num_frames=num_frames,
        tubelet_size=mc["tubelet_size"], use_sdpa=True, use_silu=False, wide_silu=True,
        uniform_power=False, use_rope=mc["use_rope"],
    )
    msg = encoder.load_state_dict(enc_sd, strict=False)
    loaded = len(enc_sd) - len(msg.unexpected_keys)
    total = len(list(encoder.state_dict().keys()))
    if loaded < total * 0.9:
        sys.exit(f"FATAL: only {loaded}/{total} encoder params loaded — key mismatch")
    if n_distill > 1:
        # 2.1 deep supervision: encoder returns the hierarchical 4-layer concat.
        encoder.return_hierarchical = True
        embed_dim_concat = encoder.embed_dim * len(encoder.hierarchical_layers)
        if embed_dim_concat != mc["embed_dim"] * n_distill:
            sys.exit(f"FATAL: hierarchical concat {embed_dim_concat} != {mc['embed_dim']}*{n_distill}")
    else:
        # 2.0: no deep supervision → single-output encoder.
        embed_dim_concat = mc["embed_dim"]
    encoder = encoder.to(device="cuda", dtype=torch.bfloat16).eval()
    torch.backends.cuda.sdp_kernel = contextlib.nullcontext
    return encoder, ckpt, embed_dim_concat


def load_encoder_predictor(ckpt_path, num_frames, model_cfg=None):
    """Load V-JEPA encoder + predictor from one .pt; arch/dims from model_cfg (iter17 WS-B3).

    Returns (encoder, predictor, embed_dim_concat). Mirrors training.build_student_predictor:
    pred ctor = get_vit_predictor_2_1() if predict_all (2.1 dense loss) else get_vit_predictor()
    (2.0 base); the build kwargs are identical across versions, only the ctor + return_all_tokens
    differ. None model_cfg → ViT-G (back-compat). Both predictor forwards accept the shared
    masked_predict_l1 call signature (z, [m_enc], [m_pred], mask_index=0) — 2.1's `mod` defaults
    to "video". iter16 §3.3: encoder half delegated to load_encoder_only (one torch.load).
    """
    encoder, ckpt, embed_dim_concat = load_encoder_only(ckpt_path, num_frames, model_cfg)
    mc = get_model_config(model_cfg)["model"]
    crop = mc["crop_size"]

    # --- predictor (same .pt) ---
    pred_sd = resolve_predictor_state_dict(ckpt)
    if pred_sd is None:
        sys.exit(f"FATAL: ckpt has no predictor key. top-level: {list(ckpt.keys())[:6]}")
    pred_ctor = get_vit_predictor_2_1() if mc["predict_all"] else get_vit_predictor()
    predictor = pred_ctor(
        img_size=(crop, crop), patch_size=mc["patch_size"], num_frames=num_frames,
        tubelet_size=mc["tubelet_size"], embed_dim=mc["embed_dim"],
        predictor_embed_dim=mc["pred_embed_dim"], depth=mc["pred_depth"],
        num_heads=mc["pred_num_heads"], use_mask_tokens=True,
        num_mask_tokens=mc["num_mask_tokens"], zero_init_mask_tokens=True,
        use_rope=mc["use_rope"], uniform_power=False, use_sdpa=True, use_silu=False,
        wide_silu=True, use_activation_checkpointing=False, return_all_tokens=mc["predict_all"],
    )
    pred_sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in pred_sd.items()}
    pmsg = predictor.load_state_dict(pred_sd, strict=False)
    p_total = len(list(predictor.state_dict().keys()))
    p_loaded = p_total - len(pmsg.missing_keys)
    if p_loaded / max(p_total, 1) < _MIN_LOADED_FRAC:
        sys.exit(f"FATAL: predictor only {p_loaded}/{p_total} loaded — random init = garbage")
    predictor = predictor.to(device="cuda", dtype=torch.bfloat16).eval()
    # iter19 2026-07-07: torch.compile TESTED + REJECTED here — on this V-JEPA RoPE/SDPA arch it compiles and
    # is ~1.38× faster BUT changes the encoder output (max|diff|≈1.9-2.4 vs eager, both with AND without the
    # #130098 SDPA monkey-patch) → it would silently corrupt every predictor-temporal metric. Do NOT re-add.
    return encoder, predictor, embed_dim_concat


def build_mask_gen(num_frames, *, npred=None, spatial_scale=None,
                   temporal_scale=None, aspect_ratio=None):
    """_MaskGenerator with V-JEPA 2.1 small-block defaults; npred/scales overridable
    (the #5 mask-ratio metric sweeps npred)."""
    return get_mask_generator()(
        crop_size=(CROP, CROP), num_frames=num_frames,
        spatial_patch_size=(PATCH_SIZE, PATCH_SIZE), temporal_patch_size=TUBELET_SIZE,
        spatial_pred_mask_scale=spatial_scale or DEFAULT_SPATIAL_SCALE,
        temporal_pred_mask_scale=temporal_scale or DEFAULT_TEMPORAL_SCALE,
        aspect_ratio=aspect_ratio or DEFAULT_ASPECT_RATIO,
        npred=npred if npred is not None else DEFAULT_NUM_BLOCKS,
    )


# ── Token-grid helpers (temporal-slot ↔ token-index mapping) ───────────
def token_grid(num_frames):
    """Token layout after patch_embed: Tp temporal × (Hp×Wp) spatial. Index order is
    t*(Hp*Wp) + h*Wp + w, so temporal slot t owns indices [t*S, (t+1)*S), S=Hp*Wp."""
    Tp = num_frames // TUBELET_SIZE
    Hp = CROP // PATCH_SIZE
    Wp = CROP // PATCH_SIZE
    return Tp, Hp, Wp, Hp * Wp


def temporal_token_idx(num_frames, t_slots) -> torch.Tensor:
    """LongTensor of token indices belonging to the given temporal slot ids."""
    _, _, _, S = token_grid(num_frames)
    # iter19 2026-07-09 (CPU→GPU audit): vectorized the O(len(t_slots)·S) python double-loop (built up to
    # ~18k elems element-by-element) into deterministic index arithmetic. Same t*S+j row-major order →
    # byte-identical (smoke: torch.equal vs the old comprehension across single/multi/empty t_slots).
    t = torch.as_tensor(list(t_slots), dtype=torch.long)
    return (t.unsqueeze(1) * S + torch.arange(S, dtype=torch.long)).reshape(-1)


def to_pixel(batch: torch.Tensor) -> torch.Tensor:
    """(B,T,3,H,W) cpu → (B,3,T,H,W) bf16 cuda (matches probe_future_mse)."""
    return batch.to("cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3, 4).contiguous()


# ── Shared OOM-backoff runner for pt_* metric fns ───────────────────────
# iter18 (2026-06-05): moved here from m12e_predictor_temporal._safe_metric so
# the in-training probe (utils/probe_trio.py) and the §2.2 eval (m12e) run the
# SAME implementation — single source, no drift.
def safe_metric(fn, encoder, predictor, batch, num_frames, min_bs=1, h_full=None):
    """Run a metric on `batch`, sub-batching with OOM backoff (mirrors AdaptiveBatchSizer
    intent). Returns concatenated per-clip array.
    The retry runs OUTSIDE the except block (iter18 2026-06-12, found via the m12f smoke):
    an exception's __traceback__ pins the FAILED forward's activations, so recursing inside
    `except` accumulates every failed attempt's VRAM — on a small card the halvings then OOM
    all the way down to min_bs. Leaving the except scope releases the pinned tensors before
    the retry; the 96 GB boxes never surfaced this only because of headroom.
    h_full: optional batch-shared full_target_h (m12e --metric all). None → call fn WITHOUT it
    (backward-compat for metrics that don't take it: pt_order + every m12f/probe_trio metric).
    On OOM sub-batching it is split alongside the clip batch (h_full[:mid]/[mid:])."""
    oom = False
    try:
        if h_full is None:
            return fn(encoder, predictor, batch, num_frames)
        return fn(encoder, predictor, batch, num_frames, h_full)
    except torch.cuda.OutOfMemoryError:
        if batch.shape[0] <= min_bs:
            raise
        oom = True
    assert oom
    cuda_cleanup()
    mid = batch.shape[0] // 2
    hlo = None if h_full is None else h_full[:mid]
    hhi = None if h_full is None else h_full[mid:]
    lo = safe_metric(fn, encoder, predictor, batch[:mid], num_frames, min_bs, h_full=hlo)
    hi = safe_metric(fn, encoder, predictor, batch[mid:], num_frames, min_bs, h_full=hhi)
    return np.concatenate([lo, hi], axis=0)


# ── h-memoization helper (V-JEPA target amortization) ──────────────────
@torch.no_grad()
def full_target_h(encoder, pixel):
    """The mask-INDEPENDENT full-forward target h = encoder(pixel) (hierarchical concat if the
    encoder returns a list), computed ONCE per batch and passed to masked_predict_l1 /
    rollout_l1_per_horizon so a metric's mask sweep reuses one target encode instead of redoing it.
    Returns None when PT_H_MEMO is off → callers recompute h inline (exact current behavior, no
    wasted forward). Bit-identical: deterministic eval/no_grad forward of the same pixel."""
    if not _H_MEMO:
        return None
    h = encoder(pixel)
    if isinstance(h, (list, tuple)):
        h = torch.cat(list(h), dim=-1)
    return h


# ── The shared masked-predict L1 core (from probe_future_mse._forward_one_batch) ──
@torch.no_grad()
def masked_predict_l1(encoder, predictor, pixel, m_enc, m_pred, h_full=None):
    """pixel (B,3,T,H,W) bf16 cuda ; m_enc/m_pred (B,n) long cuda (custom per-metric).
    Returns (per_clip_l1 (B,) np.float32, out, h_target). Same path as future_mse.
    h_full: optional precomputed full-forward target (full_target_h). When given, the
    mask-INDEPENDENT `h = encoder(pixel)` is REUSED instead of recomputed (h-memoization) —
    bit-identical, since h does not depend on the mask. None → recompute inline (current path).
    """
    apply_masks = get_apply_masks()
    z = encoder(pixel, masks=[m_enc])
    if isinstance(z, (list, tuple)):
        z = torch.cat(list(z), dim=-1)
    if h_full is not None:
        h = h_full
    else:
        h = encoder(pixel)
        if isinstance(h, (list, tuple)):
            h = torch.cat(list(h), dim=-1)
    h_target = apply_masks(h, [m_pred])
    out = predictor(z, [m_enc], [m_pred], mask_index=0)
    if isinstance(out, tuple) and len(out) == _PRED_TUPLE_LEN:
        out = out[0]
    if out.shape != h_target.shape:
        sys.exit(f"FATAL: predictor out {out.shape} != h_target {h_target.shape}")
    per_clip_l1 = (out.float() - h_target.float()).abs().mean(dim=(1, 2))
    return per_clip_l1.cpu().numpy().astype(np.float32), out, h_target


@torch.no_grad()
def rollout_l1_per_horizon(encoder, predictor, pixel, num_frames, *, free_running, h_full=None):
    """Iterated temporal rollout: predict slot k from the context of slots {0..k-1},
    for k=1..Tp-1; return per-clip L1 at each horizon → (B, Tp-1).

    free_running=True  : context bank starts as encoder(slot0); each predicted slot's OUTPUT
                         features are APPENDED to the bank (errors compound → drift).
    free_running=False : teacher-forced — at each k re-encode the REAL visible slots {0..k-1}.

    Targets are the full-forward encoder features per slot (same target space as future_mse's
    h_target). Both branches are the SAME predictor call; only the context differs.

    GPU-VALIDATION REQUIRED post-eval (§3.1): the free-running bank injects predicted (6656-dim)
    features as predictor context — the exact rollout semantics need a GPU smoke + sanity that
    L1 grows monotonically with horizon. Design per plan §2.1.
    """
    Tp, _, _, S = token_grid(num_frames)
    if Tp < _MIN_TEMPORAL_SLOTS:
        raise ValueError(f"rollout needs Tp>=2 temporal slots; got {Tp}")
    b = pixel.shape[0]
    if h_full is not None:                               # h-memoization: reuse the shared target
        h = h_full
    else:
        h = encoder(pixel)                               # full forward → per-slot targets
        if isinstance(h, (list, tuple)):
            h = torch.cat(list(h), dim=-1)               # (B, N, 6656)

    l1s = []
    if free_running:
        bank = encoder(pixel, masks=[expand_mask(temporal_token_idx(num_frames, [0]), b)])
        if isinstance(bank, (list, tuple)):
            bank = torch.cat(list(bank), dim=-1)         # (B, S, 6656) slot-0 context
        bank_idx = temporal_token_idx(num_frames, [0])
        for k in range(1, Tp):
            m_enc = expand_mask(bank_idx, b)
            m_pred = expand_mask(temporal_token_idx(num_frames, [k]), b)
            out = predictor(bank, [m_enc], [m_pred], mask_index=0)
            if isinstance(out, tuple) and len(out) == _PRED_TUPLE_LEN:
                out = out[0]
            tgt = h[:, k * S:(k + 1) * S, :]
            l1s.append((out.float() - tgt.float()).abs().mean(dim=(1, 2)).cpu().numpy())
            bank = torch.cat([bank, out], dim=1)         # APPEND predicted features (drift)
            bank_idx = torch.cat([bank_idx, temporal_token_idx(num_frames, [k])])
    else:
        for k in range(1, Tp):
            m_enc = expand_mask(temporal_token_idx(num_frames, range(0, k)), b)
            z = encoder(pixel, masks=[m_enc])            # re-encode REAL visible slots
            if isinstance(z, (list, tuple)):
                z = torch.cat(list(z), dim=-1)
            m_pred = expand_mask(temporal_token_idx(num_frames, [k]), b)
            out = predictor(z, [m_enc], [m_pred], mask_index=0)
            if isinstance(out, tuple) and len(out) == _PRED_TUPLE_LEN:
                out = out[0]
            tgt = h[:, k * S:(k + 1) * S, :]
            l1s.append((out.float() - tgt.float()).abs().mean(dim=(1, 2)).cpu().numpy())
    return np.stack(l1s, axis=1).astype(np.float32)      # (B, Tp-1)


def perclip_slope(L: np.ndarray, x) -> np.ndarray:
    """Per-clip OLS slope of L (B, k) against x (k,). Vectorised cov(x,L)/var(x).
    Used by the slope metrics (#1 rollout, #3 tdist, #5 maskratio)."""
    x = np.asarray(x, dtype=np.float64)
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    if denom == 0:
        raise ValueError("perclip_slope: x has zero variance (need ≥2 distinct sweep points)")
    return ((L.astype(np.float64) - L.mean(axis=1, keepdims=True)) * xc).sum(axis=1) / denom


def expand_mask(idx: torch.Tensor, B: int) -> torch.Tensor:
    """(n,) token-index LongTensor → (B, n) on cuda (same mask broadcast across the batch)."""
    # iter19 2026-07-09 (CPU→GPU audit): move the (n,) index to cuda BEFORE the B× expand+contiguous, so
    # the host→device copy is n longs (~0.15 MB) not B·n (~4.2 MB at B=32, n≈16.5k) — the expand then
    # materializes on-GPU. Byte-identical integer broadcast (device placement doesn't change values).
    return idx.to("cuda").unsqueeze(0).expand(B, -1).contiguous()


__all__ = [
    "load_encoder_predictor", "build_mask_gen", "token_grid", "temporal_token_idx",
    "to_pixel", "masked_predict_l1", "full_target_h", "rollout_l1_per_horizon",
    "perclip_slope", "expand_mask", "bootstrap_ci",
    "NUM_FRAMES_DEFAULT", "CROP", "TUBELET_SIZE", "PATCH_SIZE",
]
