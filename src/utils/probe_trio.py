"""Mid-training probe trio: top-1 (kNN-centroid LOOCV) + intra-inter cosine + future-frame L1.

This is the iter13 mid-training diagnostic that mirrors the post-training
m08d_plot_m06d trio (see iter/iter13_motion_probe_eval/m06d_encoder_comparison.png).
Computed at every m09a/m09c validation checkpoint to give a trajectory of the
exact metrics the paper reports — replaces the iter11 legacy retrieval probe
and the kNN-centroid action probe.

Why a single function vs running three scripts (probe_action, probe_motion_cos,
probe_future_mse) back-to-back: those three each do their own encoder forward.
Combined, that's 4 encoder forwards + 1 predictor forward per batch. This shared
pass does 2 encoder forwards + 1 predictor forward — the unmasked encoder pass
gets reused as both the pooling source (top-1, motion-cos) and the L1 target.
~25% wall-clock saved per call. Cost: ~1-1.5 min on 1000 probe clips on
Blackwell GPU. See iter/iter13_motion_probe_eval/plan_code_dev.md for the design
trade-offs (Choice A: share forward; Choice B: drop run_probe_acc_eval).

Mid-training trajectory uses the VAL split. Paper-final m06d uses the TEST
split — numbers may differ; this is by design (trajectory is a sanity check,
not the verdict).

USAGE:
    from utils.probe_trio import compute_metric_trio

    trio = compute_metric_trio(
        student, predictor, probe_clips, probe_labels,
        mask_gen=mask_generators[0],
        cfg=cfg, device=device,
        dist_layers=cfg["model"]["n_output_distillation"],
    )
    # → {"top1": 0.83, "motion_cos": 0.05, "future_l1": 0.55, "n_clips": 925, "n_classes": 3}
"""
from __future__ import annotations

import numpy as np
import torch

from utils.config import get_pipeline_config
from utils.gpu_batch import AdaptiveBatchSizer, cuda_cleanup
from utils.vjepa2_imports import get_apply_masks


# ─── Math helpers (vectorised) ────────────────────────────────────────────

def _per_clip_motion_score(emb: np.ndarray, labels: np.ndarray) -> tuple:
    """Vectorised intra-inter cosine. Mirrors src/probe_motion_cos.py:178-200.

    Returns (motion_score (N,), pos_mean (N,), neg_mean (N,)).
    No Python loop — `S = emb_n @ emb_n.T` then class-mask reductions.
    """
    if emb.ndim != 2:
        raise ValueError(f"emb must be (N, D); got {emb.shape}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
    emb_n = emb / norms
    S = emb_n @ emb_n.T
    same = (labels[:, None] == labels[None, :])
    np.fill_diagonal(same, False)
    diff = (~same)
    np.fill_diagonal(diff, False)
    same_count = same.sum(1).clip(min=1)
    diff_count = diff.sum(1).clip(min=1)
    pos = (S * same).sum(1) / same_count
    neg = (S * diff).sum(1) / diff_count
    return (pos - neg).astype(np.float32), pos.astype(np.float32), neg.astype(np.float32)


def _knn_centroid_loocv(feats: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 via leave-one-out kNN-centroid classification. Mirrors
    utils/training.py:run_probe_acc_eval but vectorised + bootstrap-CI dropped.

    For each clip i: per-class centroid = mean(features[j != i, y[j] == c]).
    Predict argmax cosine(feat_i, centroids). Returns top-1 accuracy in [0, 1].
    """
    feats_n = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-12)
    class_ids = sorted(set(labels.tolist()))
    correct = 0
    for i in range(len(feats_n)):
        # Per-class centroid excluding clip i. With ~1000 clips and ~3 classes,
        # this loop is ~30 ms — fine. If we ever need to scale to 10K clips,
        # cache class sums and subtract feats[i] / N for O(N) total.
        centroids = []
        for c in class_ids:
            mask = (labels == c) & (np.arange(len(labels)) != i)
            if mask.sum() > 0:
                centroids.append(feats_n[mask].mean(axis=0))
            else:
                centroids.append(np.zeros_like(feats_n[i]))
        centroids = np.array(centroids)
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-12)
        sims = feats_n[i] @ centroids.T
        pred = class_ids[int(sims.argmax())]
        correct += int(pred == labels[i])
    return correct / max(len(feats_n), 1)


# ─── Main entry point ─────────────────────────────────────────────────────

@torch.no_grad()
def _encoder_signature(encoder) -> tuple:
    """iter15 D15 (2026-05-16): O(1) cheap signature of encoder weights.
    Hashes the first / middle / last param tensors' means. Used to detect
    when the caller asserts encoder_frozen=True but encoder weights actually
    changed since the cache was populated → cache contract violation →
    FAIL LOUD instead of silently serving stale results. ~1 ms cost per call.

    FAIL LOUD: empty params list is structurally impossible for a trained
    encoder; raise instead of returning a hardcoded default tuple (silent
    fallback would let two uninitialized encoders share a cache slot →
    stale results). See src/CLAUDE.md "No DEFAULT, no FALLBACK".
    """
    params = list(encoder.parameters())
    if not params:
        raise RuntimeError(
            "_encoder_signature: encoder has zero parameters — cannot compute "
            "signature for cache contract verification. Encoder appears "
            "uninitialized; check the caller's build_model() path.")
    return (
        float(params[0].data.float().mean().item()),
        float(params[len(params) // 2].data.float().mean().item()),
        float(params[-1].data.float().mean().item()),
    )


def compute_metric_trio(
    student,
    predictor,
    probe_clips: list,
    probe_labels: dict,
    mask_gen,
    cfg: dict,
    device: torch.device,
    dist_layers: int = 4,
    motion_aux_head=None,                # iter15 Phase 6 C1 (2026-05-16)
    encoder_cache=None,                  # iter15 D15 (2026-05-16): persistent dict, caller-owned
    encoder_frozen: bool = False,         # iter15 D15: caller asserts encoder is not changing across calls
) -> dict:
    """Top-1 + intra-inter cosine + future-L1 in a single shared forward pass.

    Args:
        student:      encoder module. We toggle student.return_hierarchical=True
                      so the predictor receives the 4-layer-concat (6656-dim).
                      Restored to original on return.
        predictor:    V-JEPA 2.1 predictor (returns (z_pred, z_context) tuple
                      when predict_all=True). Required — top-1 + motion-cos
                      alone don't need it, but we keep the signature uniform
                      so callers can run all three metrics with one call.
        probe_clips:  list of (clip_key, tag_dict, tensor[T,C,H,W]) from
                      utils.training.build_probe_clips.
        probe_labels: {clip_key → {"class_id": int, ...}} from
                      utils.action_labels.load_action_labels.
        mask_gen:     callable(B) → (masks_enc, masks_pred). Stateful; we
                      let it advance — caller's training step gets a different
                      mask sample but this is acceptable for an additive
                      diagnostic that runs every val cycle.
        cfg:          merged training config (used for batch-sizer wiring).
        device:       cuda device.
        dist_layers:  number of hierarchical taps (4 for V-JEPA 2.1 ViT-G).

    Returns dict (no bootstrap CI — trajectory mode):
      {"top1": float, "motion_cos": float, "future_l1": float,
       "n_clips": int, "n_classes": int}

    Cost: ~1-1.5 min on 1000 probe clips on Blackwell.
    Raises ValueError if too few labeled clips (< 10) for stable LOOCV.
    """
    apply_masks = get_apply_masks()

    # 1. Filter to labeled clips. probe_clips items are (key, tags, tensor).
    keyed = [(c[0], c[2]) for c in probe_clips if c[0] in probe_labels]
    if len(keyed) < 10:
        raise ValueError(
            f"compute_metric_trio: too few labeled probe clips ({len(keyed)}) — "
            f"need >= 10 for stable LOOCV top-1.")

    keys = [k for k, _ in keyed]
    labels = np.array([probe_labels[k]["class_id"] for k in keys], dtype=np.int64)
    n_classes = int(labels.max() + 1)

    # iter15 D15 (2026-05-16): cache HIT branch.
    # When the caller asserts encoder_frozen=True AND has provided an encoder_cache
    # that's already populated, skip the entire encoder-forward + predictor-forward
    # loop. Only the motion_aux head output is recomputed (it's the thing being
    # trained). Saves ~10 min per val checkpoint on head cells. Encoder cells
    # always cache-miss because encoder_frozen=False (encoder updates each step).
    # FAIL LOUD if the encoder signature changed since cache populated — that means
    # the caller violated the frozen contract and we'd serve stale results.
    cache_hit = (
        encoder_cache is not None
        and encoder_frozen
        and "pooled_no_ma" in encoder_cache
    )
    if cache_hit:
        current_sig = _encoder_signature(student)
        cached_sig = encoder_cache["enc_sig"]
        if current_sig != cached_sig:
            raise RuntimeError(
                f"compute_metric_trio (D15): caller asserted encoder_frozen=True "
                f"but encoder signature changed since cache populated.\n"
                f"  cached_sig:  {cached_sig}\n"
                f"  current_sig: {current_sig}\n"
                f"Frozen contract violated — refusing to serve stale cached "
                f"top1/motion_cos/future_l1. Caller must either (a) re-initialize "
                f"encoder_cache={{}} after any encoder weight change, OR (b) pass "
                f"encoder_frozen=False to disable caching.")
        pooled_no_ma_cpu = encoder_cache["pooled_no_ma"]   # (N, D=1664) cpu fp32
        per_clip_l1_cpu = encoder_cache["per_clip_l1"]     # (N,) cpu fp32
        # Recompute MA augment on cached pooled features (only the head is training).
        if motion_aux_head is not None:
            from utils.motion_aux_loss import forward_motion_aux_concat
            pooled_gpu = pooled_no_ma_cpu.to(device)
            ma_concat = forward_motion_aux_concat(motion_aux_head, pooled_gpu)
            pooled_full_cpu = torch.cat([pooled_gpu, ma_concat], dim=-1).cpu()
        else:
            pooled_full_cpu = pooled_no_ma_cpu
        pooled_np = pooled_full_cpu.numpy()
        future_l1 = float(per_clip_l1_cpu.numpy().mean())
        print(f"  [probe-trio][D15 cache HIT] reused {len(pooled_np)} pooled + "
              f"{len(per_clip_l1_cpu)} per-clip L1 — skipped encoder forward.",
              flush=True)
        top1 = _knn_centroid_loocv(pooled_np, labels)
        score, _, _ = _per_clip_motion_score(pooled_np, labels)
        motion_cos = float(score.mean())
        return {
            "top1":       round(top1, 6),
            "motion_cos": round(motion_cos, 6),
            "future_l1":  round(future_l1, 6),
            "n_clips":    len(pooled_np),
            "n_classes":  n_classes,
        }
    # ── Cache MISS path below: full compute, optionally populate cache at end ──
    # (n_classes already computed pre-branch — shared by HIT + MISS return paths)

    # 2. Save state — toggle hierarchical ON for predictor input. Restore
    #    on return so caller's downstream code (e.g. probe-acc using last-layer)
    #    sees the same toggle it set.
    was_train = student.training
    was_pred_train = predictor.training
    had_hier = getattr(student, "return_hierarchical", None)
    student.eval()
    predictor.eval()
    if had_hier is not None:
        student.return_hierarchical = True

    # 3. AdaptiveBatchSizer wiring — same as run_probe_acc_eval.
    _gpu = get_pipeline_config()["gpu"]
    sizer = AdaptiveBatchSizer(
        initial_size=_gpu["training_adapted_probe_bs"],
        min_size=1,
        max_size=_gpu["training_adapted_probe_bs"],
        memory_cap=_gpu["gpu_memory_target"],
    )

    # 4. dtype matches student's parameters (typically bf16 in our pipeline).
    model_dtype = next(student.parameters()).dtype
    embed_dim = cfg["model"]["embed_dim"]   # 1664 for ViT-G

    # iter15 D15 (2026-05-16): split pooled_chunks_no_ma (cacheable — encoder
    # output is frozen for head cells) from pooled_chunks_full (post-MA augment,
    # changes every val cycle because MA head is trained). Both populated in the
    # same loop. The _no_ma list goes into the cache; the _full list feeds top1
    # + motion_cos this call. For encoder-cell mode (encoder_cache=None) only
    # _full is meaningful.
    pooled_chunks_no_ma: list = []
    pooled_chunks_full: list = []
    l1_chunks: list = []
    n = len(keyed)
    i = 0
    print(f"  [probe-trio] encoding {n} clips...", flush=True)
    try:
        while i < n:
            end = min(i + sizer.size, n)
            # iter13 (2026-05-05): V-JEPA's patch_embed is a Conv3d that expects
            # (B, C, T, H, W). decode_to_tensor returns (T, C, H, W) per clip and
            # torch.stack gives us (B, T, C, H, W) — must permute T↔C BEFORE the
            # forward, exactly like forward_vjepa (utils/frozen_features.py:197).
            # Without this, F.conv3d raises "expected input to have 3 channels,
            # but got 16 channels instead" (because the 16-frame T axis lands
            # where C should be).
            batch_tensors = torch.stack([t for (_, t) in keyed[i:end]]).to(
                device, dtype=model_dtype, non_blocking=True)
            batch_tensors = batch_tensors.permute(0, 2, 1, 3, 4).contiguous()
            B = batch_tensors.shape[0]

            # 4a. Unmasked forward — used both for pooling AND L1 target.
            #     With return_hierarchical=True, returns (B, N, 4*embed_dim).
            try:
                h_full = student(batch_tensors, training=True)
            except torch.cuda.OutOfMemoryError:
                cuda_cleanup()
                sizer.on_oom()
                continue
            if isinstance(h_full, (list, tuple)):
                h_concat = torch.cat(list(h_full), dim=-1)
            else:
                h_concat = h_full   # already (B, N, 4*embed_dim) per return_hierarchical

            # 4b. Pool LAST embed_dim columns of the concat → semantically same
            #     as run_probe_acc_eval's last-layer mean-pool. This keeps top-1
            #     and motion-cos comparable to the post-training m06d trio.
            last_layer = h_concat[..., -embed_dim:]
            pooled_no_ma = last_layer.mean(dim=1).float()                 # (B, D) GPU
            # iter15 D15 (2026-05-16): persist the pre-augment pooled tensor so
            # cache HIT path can recompute MA augment without rerunning encoder.
            pooled_chunks_no_ma.append(pooled_no_ma.cpu())
            # iter15 Phase 6 C1 (2026-05-16): augment with motion_aux head output
            # so top1 + motion_cos reflect the trained head (head cells: encoder
            # frozen → without this the metrics are identical across head cells).
            # future_l1 (computed below from masked predictor forward) is NOT
            # affected — predictor sees encoder features only, by design.
            if motion_aux_head is not None:
                from utils.motion_aux_loss import forward_motion_aux_concat
                ma_concat = forward_motion_aux_concat(motion_aux_head, pooled_no_ma)
                pooled = torch.cat([pooled_no_ma, ma_concat], dim=-1)
            else:
                pooled = pooled_no_ma
            pooled_chunks_full.append(pooled.cpu())

            # 4c. Mask sample for L1. Mirrors probe_future_mse._forward_one_batch
            #     but reuses h_concat as the prediction target (no extra forward).
            m_enc_raw, m_pred_raw = mask_gen(B)
            m_enc = (torch.stack(m_enc_raw, dim=0).to(device)
                     if isinstance(m_enc_raw, list) else m_enc_raw.to(device))
            m_pred = (torch.stack(m_pred_raw, dim=0).to(device)
                      if isinstance(m_pred_raw, list) else m_pred_raw.to(device))

            # Context forward (with mask) — separate from h_concat which is unmasked.
            try:
                z_ctx = student(batch_tensors, masks=[m_enc], training=True)
            except torch.cuda.OutOfMemoryError:
                cuda_cleanup()
                sizer.on_oom()
                continue
            if isinstance(z_ctx, (list, tuple)):
                z_concat = torch.cat(list(z_ctx), dim=-1)
            else:
                z_concat = z_ctx

            # Reuse h_concat as L1 target → predictor predicts target tokens
            # at the m_pred positions.
            h_target = apply_masks(h_concat, [m_pred])
            out = predictor(z_concat, [m_enc], [m_pred], mod="video", mask_index=0)
            if isinstance(out, tuple) and len(out) == 2:
                out = out[0]   # (z_pred, z_context) → keep z_pred only
            if out.shape != h_target.shape:
                # Defensive: skip the L1 batch but keep pooled (top1+motion still valid)
                print(f"  [probe-trio] WARN: predictor shape {out.shape} != "
                      f"h_target {h_target.shape} for batch {i}:{end} — skipping L1",
                      flush=True)
                i = end
                sizer.after_batch_success()
                continue
            per_clip_l1 = (out.float() - h_target.float()).abs().mean(dim=(1, 2))
            l1_chunks.append(per_clip_l1.cpu())

            i = end
            sizer.after_batch_success()
    finally:
        # Restore state regardless of exception.
        if was_train:
            student.train()
        if was_pred_train:
            predictor.train()
        if had_hier is not None:
            student.return_hierarchical = had_hier
        cuda_cleanup()

    # 5. Aggregate.
    pooled_np = torch.cat(pooled_chunks_full, dim=0).numpy()
    if l1_chunks:
        per_clip_l1_cpu = torch.cat(l1_chunks, dim=0)
        future_l1 = float(per_clip_l1_cpu.numpy().mean())
    else:
        per_clip_l1_cpu = None
        future_l1 = float("nan")   # all batches OOM'd or shape-skipped — flag clearly

    # iter15 D15 (2026-05-16): populate cache if caller wants it + asserted frozen.
    # FAIL LOUD if l1_chunks empty — every-batch OOM in a freshly-extracted run is
    # not a "let's cache None" situation. Caller should debug VRAM, not silently
    # skip the L1 axis on subsequent val cycles.
    if encoder_cache is not None and encoder_frozen:
        if per_clip_l1_cpu is None:
            raise RuntimeError(
                "compute_metric_trio (D15): all probe-trio batches OOM'd / shape-"
                "skipped → per_clip_l1 is empty. Refusing to populate cache with a "
                "NaN future_l1. Fix the underlying VRAM/shape issue, then re-run.")
        encoder_cache["pooled_no_ma"] = torch.cat(pooled_chunks_no_ma, dim=0)
        encoder_cache["per_clip_l1"] = per_clip_l1_cpu
        encoder_cache["enc_sig"] = _encoder_signature(student)
        print(f"  [probe-trio][D15 cache POPULATED] stored "
              f"pooled_no_ma={tuple(encoder_cache['pooled_no_ma'].shape)} + "
              f"per_clip_l1={tuple(encoder_cache['per_clip_l1'].shape)} for "
              f"reuse across val checkpoints.", flush=True)

    # Top-1 — vectorised LOOCV per-class centroid.
    top1 = _knn_centroid_loocv(pooled_np, labels)

    # Motion-cos — intra-inter mean.
    score, _, _ = _per_clip_motion_score(pooled_np, labels)
    motion_cos = float(score.mean())

    return {
        "top1":       round(top1, 6),
        "motion_cos": round(motion_cos, 6),
        "future_l1":  round(future_l1, 6) if not np.isnan(future_l1) else float("nan"),
        "n_clips":    len(pooled_np),
        "n_classes":  n_classes,
    }
