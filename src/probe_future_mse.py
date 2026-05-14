"""Future-frame latent prediction MSE. V-JEPA-only diagnostic. GPU-only.

For each clip: sample (m_enc, m_pred) via V-JEPA's MaskGenerator. Encode the
context (visible tokens via m_enc) and the full clip (target tokens via m_pred).
Predictor predicts target tokens from context tokens. Per-clip L1 distance
between predicted and target tokens — matches V-JEPA's training objective.

DINOv2 has no future-frame predictor head — it would require a fresh predictor
trained under the wrong objective. So this module is a HEALTH CHECK for V-JEPA
on Indian video, NOT a paired V-JEPA-vs-DINOv2 test.

Stages:
  forward             — encode + predict, dump per-clip MSE [GPU, ~30 min/encoder]
  paired_per_variant  — pairwise BCa Δ across {frozen, explora, surgery_*} [CPU]

USAGE (priority 1: forward on V-JEPA frozen only — DINOv2 path intentionally absent):
    python -u src/probe_future_mse.py --FULL \\
        --stage forward --variant vjepa_2_1_frozen \\
        --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \\
        --action-probe-root outputs/full/probe_action \\
        --local-data data/eval_10k_local \\
        --output-root outputs/full/probe_future_mse \\
        --cache-policy 1 2>&1 | tee logs/probe_future_mse_forward_vjepa.log

    python -u src/probe_future_mse.py --FULL \\
        --stage paired_per_variant \\
        --output-root outputs/full/probe_future_mse --cache-policy 1
"""
import argparse
import contextlib
import json
import os
import queue
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils.action_labels import load_action_labels
from utils.bootstrap import bootstrap_ci, paired_bca
from utils.cache_policy import (
    add_cache_policy_arg,
    guarded_delete,
    resolve_cache_policy_interactive,
)
from utils.checkpoint import save_array_checkpoint, save_json_checkpoint
from utils.config import add_local_data_arg, check_gpu, get_pipeline_config
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.frozen_features import (
    decode_to_tensor,
    resolve_encoder_state_dict,
)
from utils.gpu_batch import AdaptiveBatchSizer, cleanup_temp, cuda_cleanup
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog
from utils.progress import make_pbar
from utils.vjepa2_imports import (
    get_apply_masks,
    get_mask_generator,
    get_vit_gigantic_xformers,
    get_vit_predictor_2_1,
)
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics


# ── Constants ─────────────────────────────────────────────────────────

_PCFG = get_pipeline_config()
NUM_FRAMES_DEFAULT = 16
PATCH_SIZE = 16
TUBELET_SIZE = 2
CROP = 384                          # V-JEPA 2.1 ViT-G/384 (must match encoder)
CHECKPOINT_EVERY = _PCFG["streaming"]["checkpoint_every"]

# V-JEPA 2.1 predictor architecture (from configs/model/vjepa2_1.yaml).
PRED_EMBED_DIM = 384
PRED_DEPTH = 24
PRED_NUM_HEADS = 12
NUM_MASK_TOKENS = 2

# Default mask config — matches V-JEPA 2.1 small-block convention (8 × ~15% spatial).
# Single mask spec (no large-block component) is sufficient for a forward-only
# diagnostic; the large-block path is a training-time optimization not an objective.
DEFAULT_SPATIAL_SCALE = (0.15, 0.15)
DEFAULT_TEMPORAL_SCALE = (1.0, 1.0)
DEFAULT_ASPECT_RATIO = (0.75, 1.5)
DEFAULT_NUM_BLOCKS = 8

# Variants understood by --paired_per_variant. P1 ships frozen only; P2 adds the
# m09a continual-pretrain output; P3 adds the m09c factor-surgery output. m09b
# ExPLoRA was dropped per iter13 pivot — see plan_code_dev.md §"P2 + P3 Coding plan".
KNOWN_VARIANTS = (
    "vjepa_2_1_frozen",
    "vjepa_2_1_pretrain",            # P2 — m09a continual SSL pretrain
    "vjepa_2_1_surgical_3stage_DI",  # P3-A — m09c 3-stage WITH interaction tubes (D_I)
    "vjepa_2_1_surgical_noDI",       # P3-B — m09c 2-stage WITHOUT D_I (skepticism test)
)


# ── Encoder / predictor / mask-generator loaders ──────────────────────

def _load_vjepa_2_1_encoder_hierarchical(ckpt_path: Path, num_frames: int):
    """V-JEPA 2.1 ViT-G encoder with deep-supervision hierarchical output ON.

    Why this lives here and NOT in utils/frozen_features.py:
    `load_vjepa_2_1_frozen` returns the LAST layer (1664-dim) — what every other
    m*.py probe needs. The predictor in this module is built with
    `predictor_embed_dim_in = embed_dim * len(hierarchical_layers) = 6656`
    (deps/vjepa2/app/vjepa_2_1/models/predictor.py:85-89), so it expects the
    4-layer concat. Setting `return_hierarchical=True` makes the encoder's
    own forward path do the concat (vision_transformer.py:335-337) → returns
    a single (B, N, 6656) tensor that lines up with the predictor input.

    Returns (model, crop=384, embed_dim_concat=6656).
    """
    if not ckpt_path.exists():
        sys.exit(f"FATAL: encoder ckpt not found: {ckpt_path}")
    print(f"Loading V-JEPA 2.1 ViT-G hierarchical (vit_gigantic_xformers, crop={CROP}, T={num_frames}) ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = resolve_encoder_state_dict(ckpt)
    state_dict = {k.replace("module.", "").replace("backbone.", ""): v
                  for k, v in state_dict.items()}

    vit_constructor = get_vit_gigantic_xformers()
    model = vit_constructor(
        img_size=(CROP, CROP), patch_size=PATCH_SIZE, num_frames=num_frames,
        tubelet_size=TUBELET_SIZE, use_sdpa=True, use_silu=False, wide_silu=True,
        uniform_power=False, use_rope=True,
    )
    msg = model.load_state_dict(state_dict, strict=False)
    loaded = len(state_dict) - len(msg.unexpected_keys)
    total = len(list(model.state_dict().keys()))
    print(f"  Loaded {loaded}/{total} params  (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")
    if loaded < total * 0.9:
        sys.exit(f"FATAL: only {loaded}/{total} V-JEPA params loaded — key mismatch")

    # Flip to hierarchical: encoder forward returns (B, N, 1664*4=6656) instead of (B, N, 1664).
    model.return_hierarchical = True

    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    torch.backends.cuda.sdp_kernel = contextlib.nullcontext
    embed_dim_concat = model.embed_dim * len(model.hierarchical_layers)
    return model, CROP, embed_dim_concat


def _load_predictor_2_1(ckpt_path: Path, num_frames: int):
    """Load V-JEPA 2.1 predictor from the same .pt that holds the encoder.
    Returns predictor on cuda, eval-mode, bf16. Mirrors m09a1_pretrain_encoder.py:268-309.
    """
    if not ckpt_path.exists():
        sys.exit(f"FATAL: predictor ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "predictor" not in ckpt:
        sys.exit("FATAL: ckpt has no 'predictor' key — V-JEPA 2.1 official ckpt should include it. "
                 f"Top-level keys: {list(ckpt.keys())[:6]}")

    pred_constructor = get_vit_predictor_2_1()
    predictor = pred_constructor(
        img_size=(CROP, CROP),
        patch_size=PATCH_SIZE,
        num_frames=num_frames,
        tubelet_size=TUBELET_SIZE,
        embed_dim=1664,                                      # encoder hidden dim
        predictor_embed_dim=PRED_EMBED_DIM,
        depth=PRED_DEPTH,
        num_heads=PRED_NUM_HEADS,
        use_mask_tokens=True,
        num_mask_tokens=NUM_MASK_TOKENS,
        zero_init_mask_tokens=True,
        use_rope=True,
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,                  # eval-only — disable
        return_all_tokens=True,                              # V-JEPA 2.1 dense path
    )
    pred_state = {k.replace("module.", "").replace("backbone.", ""): v
                  for k, v in ckpt["predictor"].items()}
    msg = predictor.load_state_dict(pred_state, strict=False)
    pred_total = len(list(predictor.state_dict().keys()))
    pred_loaded = pred_total - len(msg.missing_keys)
    pct = pred_loaded / max(pred_total, 1) * 100
    print(f"  Predictor loaded {pred_loaded}/{pred_total} keys ({pct:.0f}%)")
    if pct < 50:
        sys.exit(f"FATAL: predictor only {pct:.0f}% loaded — random init would produce garbage MSE")

    predictor = predictor.to(device="cuda", dtype=torch.bfloat16).eval()
    return predictor


def _build_mask_gen(num_frames: int):
    """Build a single _MaskGenerator with V-JEPA 2.1 small-block defaults."""
    _MaskGenerator = get_mask_generator()
    return _MaskGenerator(
        crop_size=(CROP, CROP),
        num_frames=num_frames,
        spatial_patch_size=(PATCH_SIZE, PATCH_SIZE),
        temporal_patch_size=TUBELET_SIZE,
        spatial_pred_mask_scale=DEFAULT_SPATIAL_SCALE,
        temporal_pred_mask_scale=DEFAULT_TEMPORAL_SCALE,
        aspect_ratio=DEFAULT_ASPECT_RATIO,
        npred=DEFAULT_NUM_BLOCKS,
    )


# ── Per-batch forward ─────────────────────────────────────────────────

@torch.no_grad()
def _forward_one_batch(encoder, predictor, mask_gen, batch: torch.Tensor) -> np.ndarray:
    """For each clip in batch, sample masks → context+target → predict → per-clip L1.

    Returns: (B,) float32 numpy array of per-clip mean-L1 across (n_pred_tokens, D).

    Pipeline mirrors utils/training.py:528-534 (forward step for jepa loss):
        1. mask_gen(B) → (m_enc, m_pred) tensors of token indices
        2. z = encoder(pixel, masks=[m_enc])         # context features at visible positions
        3. h = encoder(pixel)                         # full forward (no mask) for target
        4. h_target = apply_masks(h, [m_pred])       # ground-truth target tokens
        5. out = predictor(z, [m_enc], [m_pred], mask_index=0)  # predicted target tokens
        6. per-clip L1 = mean |out − h_target| over (n_pred_tokens, D) per row
    """
    apply_masks = get_apply_masks()
    B = batch.shape[0]
    device = "cuda"

    # Mask gen returns lists; we collate to tensors of shape (B, n_tokens) and move to GPU.
    # _MaskGenerator(B) typically returns (masks_enc_list, masks_pred_list); each list
    # element is a single tensor of token indices for one clip in the batch. We stack.
    m_enc_raw, m_pred_raw = mask_gen(B)
    m_enc = torch.stack(m_enc_raw, dim=0).to(device) if isinstance(m_enc_raw, list) else m_enc_raw.to(device)
    m_pred = torch.stack(m_pred_raw, dim=0).to(device) if isinstance(m_pred_raw, list) else m_pred_raw.to(device)

    pixel = batch.to(device, dtype=torch.bfloat16).permute(0, 2, 1, 3, 4).contiguous()  # (B,3,T,H,W)

    # 1) Context features (encoder applied with m_enc → only visible tokens).
    # V-JEPA 2.1 with n_output_distillation=4 returns a list of hierarchical-layer
    # outputs (4 × 1664-dim each). predictor_embed is built as a Sequential whose
    # first Linear maps `embed_dim * len(hierarchical_layers) = 1664*4 = 6656` →
    # 1664 (deps/vjepa2/app/vjepa_2_1/models/predictor.py:85-89), so the predictor
    # expects ALL 4 layers concatenated along the feature dim — NOT just the last
    # layer. Taking z[-1] gave a 1664-dim tensor that crashed the 6656-dim Linear.
    z = encoder(pixel, masks=[m_enc])
    if isinstance(z, (list, tuple)):
        z = torch.cat(list(z), dim=-1)                 # (B, N_vis, embed_dim * n_distill = 6656)

    # 2) Target features — full forward (no mask) → all tokens; mask post-hoc.
    # Same concat: predictor_proj outputs 6656-dim per token (predictor.py:178-184),
    # so h_target must be the 4-layer-concat as well for the L1 loss to align.
    h = encoder(pixel)
    if isinstance(h, (list, tuple)):
        h = torch.cat(list(h), dim=-1)                 # (B, N, 6656)
    h_target = apply_masks(h, [m_pred])               # (B, n_pred_tokens, 6656)

    # 3) Predict.
    out = predictor(z, [m_enc], [m_pred], mask_index=0)
    if isinstance(out, tuple) and len(out) == 2:
        out = out[0]                                   # discard projected-context aux output

    if out.shape != h_target.shape:
        sys.exit(f"FATAL: predictor output {out.shape} != h_target {h_target.shape}")

    # 4) Per-clip L1: mean |Δ| over (n_pred_tokens, D).
    per_clip_l1 = (out.float() - h_target.float()).abs().mean(dim=(1, 2))  # (B,)
    return per_clip_l1.cpu().numpy()


# ── Stage 1 — forward on test split ───────────────────────────────────

def _save_mse_ckpt(ckpt_file: Path, mse_acc, keys_acc) -> None:
    """Atomic .npz checkpoint. Suffix is .tmp.npz (not .npz.tmp) —
    np.savez auto-appends .npz when the path doesn't end in .npz, which
    would silently rename our tmp behind our back. See errors_N_fixes.md #82.
    """
    if not mse_acc:
        return
    tmp = ckpt_file.with_suffix(".tmp.npz")
    np.savez(tmp,
             mse=np.array(mse_acc, dtype=np.float32),
             keys=np.array(keys_acc, dtype=object))
    os.replace(tmp, ckpt_file)


def _flush_batch_forward(pending_tensors, pending_keys, encoder, predictor, mask_gen,
                         sizer, mse_acc, keys_acc, pbar):
    """Sub-batch with OOM-retry. Mirrors frozen_features._flush_batch."""
    batch = torch.stack(pending_tensors)
    n_total = batch.shape[0]
    i = 0
    while i < n_total:
        sub = batch[i:i + sizer.size]
        try:
            per_clip_l1 = _forward_one_batch(encoder, predictor, mask_gen, sub)
        except torch.cuda.OutOfMemoryError:
            cuda_cleanup()
            if not sizer.on_oom():
                raise
            continue
        for r in range(per_clip_l1.shape[0]):
            mse_acc.append(float(per_clip_l1[r]))
            keys_acc.append(pending_keys[i + r])
            pbar.update(1)
        sizer.after_batch_success()
        i += sub.shape[0]


def run_forward_stage(args, wb) -> None:
    """V-JEPA encoder+predictor forward over test split, dump per-clip L1.

    Outputs:
        <output_root>/<variant>/per_clip_mse.npy        (N_test,) float32
        <output_root>/<variant>/clip_keys.npy           (N_test,) str
        <output_root>/<variant>/aggregate_mse.json      {mse_mean, mse_std, mse_ci, n}
    """
    if args.variant is None:
        sys.exit("FATAL: --stage forward requires --variant")
    if not args.variant.startswith("vjepa"):
        sys.exit(f"FATAL: --stage forward only supports vjepa_* variants (DINOv2 has no future predictor); got {args.variant!r}")
    if args.encoder_ckpt is None:
        sys.exit("FATAL: --stage forward requires --encoder-ckpt (loads encoder + predictor + EMA target_encoder)")
    if args.local_data is None:
        sys.exit("FATAL: --stage forward requires --local-data")
    if args.action_probe_root is None:
        sys.exit("FATAL: --stage forward requires --action-probe-root (for action_labels.json test split)")

    check_gpu()
    # iter15 (2026-05-14): cgroup envelope + OOM watchdog (utils/cgroup_monitor.py)
    print_cgroup_header(prefix="[probe_future_mse]")
    start_oom_watchdog(prefix="[probe_future_mse]-oom-watchdog")
    cleanup_temp()
    ensure_local_data(args)

    out_dir = args.output_root / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mse = out_dir / "per_clip_mse.npy"
    out_keys = out_dir / "clip_keys.npy"
    out_agg = out_dir / "aggregate_mse.json"
    if out_mse.exists() and out_keys.exists() and args.cache_policy == "1":
        print(f"  [keep] {out_mse} present -- skipping (--cache-policy 2 to redo)")
        return
    guarded_delete(out_mse, args.cache_policy, "per_clip_mse")
    guarded_delete(out_keys, args.cache_policy, "clip_keys")
    guarded_delete(out_agg, args.cache_policy, "aggregate_mse")

    labels = load_action_labels(args.action_probe_root / "action_labels.json")
    test_keys = [k for k, info in labels.items() if info["split"] == "test"]
    print(f"Forward on {len(test_keys)} test clips, variant={args.variant}")

    # Encoder + predictor + mask-gen
    encoder, crop, embed_dim_concat = _load_vjepa_2_1_encoder_hierarchical(
        args.encoder_ckpt, args.num_frames)
    if crop != CROP:
        sys.exit(f"FATAL: encoder crop {crop} != module CROP {CROP}; predictor was built for {CROP}")
    # predictor_embed.0 is Linear(1664*n_distill -> 1664) = Linear(6656 -> 1664).
    if embed_dim_concat != 1664 * 4:
        sys.exit(f"FATAL: hierarchical concat dim {embed_dim_concat} != 6656; predictor expects 1664*4")
    predictor = _load_predictor_2_1(args.encoder_ckpt, args.num_frames)
    mask_gen = _build_mask_gen(args.num_frames)
    print(f"  predictor: {sum(p.numel() for p in predictor.parameters()) / 1e6:.1f}M params")

    # Resume
    ckpt_file = out_dir / ".probe_future_mse_ckpt.npz"
    mse_acc, keys_acc, processed = [], [], set()
    if ckpt_file.exists():
        try:
            data = np.load(ckpt_file, allow_pickle=True)
            mse_acc = list(data["mse"].tolist())
            keys_acc = list(data["keys"])
            processed = set(keys_acc)
            print(f"  Resume: {len(processed)} clips already cached")
        except Exception as e:
            print(f"  WARN: resume ckpt corrupt ({e}), starting fresh")

    target_keys = set(test_keys) - processed
    if not target_keys:
        print("  All clips already processed -- skipping forward")
    else:
        sizer = AdaptiveBatchSizer(
            initial_size=4, min_size=1, max_size=8,
            memory_cap=_PCFG["gpu"]["gpu_memory_target"],
        )
        print(f"  AdaptiveBatchSizer: start={sizer.size}, max=8 (encoder + predictor heavier than encoder-alone)")

        pbar = make_pbar(total=len(test_keys), desc="probe_future_mse",
                         unit="clip", initial=len(processed))
        tmp_base = out_dir / "tmp_decode"
        tmp_base.mkdir(parents=True, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(dir=tmp_base)
        clip_q, tar_stop, _reader = iter_clips_parallel(
            local_data=args.local_data, subset_keys=target_keys, processed_keys=processed)

        pending_tensors, pending_keys = [], []
        n_since_ckpt = 0
        try:
            while True:
                try:
                    item = clip_q.get(timeout=300)
                except queue.Empty:
                    print("  WARN: clip queue timeout (5 min) -- flushing pending")
                    break
                if item is None:
                    break
                clip_key, mp4_bytes = item
                t = decode_to_tensor(mp4_bytes, tmp_dir, clip_key, args.num_frames, CROP)
                if t is None:
                    print(f"  SKIP (decode fail): {clip_key}")
                    continue
                pending_tensors.append(t)
                pending_keys.append(clip_key)

                if len(pending_tensors) >= 8:
                    _flush_batch_forward(pending_tensors, pending_keys,
                                         encoder, predictor, mask_gen,
                                         sizer, mse_acc, keys_acc, pbar)
                    n_since_ckpt += len(pending_keys)
                    pending_tensors, pending_keys = [], []
                    if n_since_ckpt >= CHECKPOINT_EVERY:
                        _save_mse_ckpt(ckpt_file, mse_acc, keys_acc)
                        n_since_ckpt = 0
            if pending_tensors:
                _flush_batch_forward(pending_tensors, pending_keys,
                                     encoder, predictor, mask_gen,
                                     sizer, mse_acc, keys_acc, pbar)
        finally:
            tar_stop.set()
            pbar.close()

    if not mse_acc:
        sys.exit("FATAL: 0 clips produced an MSE — pipeline failure")

    mse_arr = np.asarray(mse_acc, dtype=np.float32)
    save_array_checkpoint(mse_arr, out_mse)
    np.save(out_keys, np.array(keys_acc, dtype=object))
    mse_ci = bootstrap_ci(mse_arr.astype(np.float64))
    save_json_checkpoint({
        "variant": args.variant, "n_test": int(len(mse_arr)),
        "mse_mean": round(float(mse_arr.mean()), 6),
        "mse_std":  round(float(mse_arr.std()),  6),
        "mse_ci":   mse_ci,
        "num_frames": args.num_frames,
        "predictor_loaded_from": str(args.encoder_ckpt),
    }, out_agg)
    print(f"  per_clip_mse: mean={mse_arr.mean():.4f}  std={mse_arr.std():.4f}  N={len(mse_arr)}")
    log_metrics(wb, {f"{args.variant}_mse_mean": float(mse_arr.mean()),
                     f"{args.variant}_mse_std":  float(mse_arr.std()),
                     f"{args.variant}_n_test":   int(len(mse_arr))})


# ── Stage 2 — paired Δ across V-JEPA variants (CPU) ──────────────────

def run_paired_per_variant_stage(args, wb) -> None:
    """Pairwise BCa Δ across discovered V-JEPA variants on the test split.

    Output: <output_root>/probe_future_mse_per_variant.json
    Priority 1: only vjepa_2_1_frozen exists, so the JSON shows that single
    variant + 'dinov2: n/a' explicitly. Priorities 2/3 fill in explora / surgery.
    """
    by_variant = {}
    for v in KNOWN_VARIANTS:
        var_dir = args.output_root / v
        mse_path = var_dir / "per_clip_mse.npy"
        keys_path = var_dir / "clip_keys.npy"
        agg_path = var_dir / "aggregate_mse.json"
        if not (mse_path.exists() and keys_path.exists() and agg_path.exists()):
            by_variant[v] = None
            continue
        agg = json.loads(agg_path.read_text())
        by_variant[v] = {
            "mse_mean": agg["mse_mean"],
            "mse_std":  agg["mse_std"],
            "mse_ci":   agg["mse_ci"],
            "n":        agg["n_test"],
            "_per_clip_mse": np.load(mse_path).astype(np.float64),
            "_keys":         np.load(keys_path, allow_pickle=True),
        }

    available = [v for v, e in by_variant.items() if e is not None]
    print(f"Variants found: {available}")
    pairwise_deltas = {}
    if len(available) >= 2:
        for a, b in [(available[i], available[j])
                     for i in range(len(available))
                     for j in range(i + 1, len(available))]:
            ka = by_variant[a]["_keys"]
            kb = by_variant[b]["_keys"]
            if not np.array_equal(ka, kb):
                # iter14 recipe-v2 (2026-05-09): FAIL LOUD per CLAUDE.md. Small
                # intersection drift (<5%) is OK (e.g., one encoder failed to
                # decode a few clips), but >5% drift indicates broader pipeline
                # mismatch (different eval subset, stale cache, etc).
                a_idx = {k: i for i, k in enumerate(ka)}
                shared = [k for k in ka if k in set(kb)]
                drop_pct_a = 1.0 - (len(shared) / max(len(ka), 1))
                drop_pct_b = 1.0 - (len(shared) / max(len(kb), 1))
                max_drop = max(drop_pct_a, drop_pct_b)
                if max_drop > 0.05:
                    print(f"❌ FATAL [probe_future_mse]: {a} vs {b} keys disagree by "
                          f"{max_drop:.1%} (>5% threshold).", file=sys.stderr)
                    print(f"   {a}: {len(ka)} keys  |  {b}: {len(kb)} keys  |  "
                          f"intersection: {len(shared)}", file=sys.stderr)
                    print("   Re-extract per-clip MSE on the same eval-subset for both encoders.",
                          file=sys.stderr)
                    sys.exit(3)
                a_arr = np.array([by_variant[a]["_per_clip_mse"][a_idx[k]] for k in shared])
                b_idx = {k: i for i, k in enumerate(kb)}
                b_arr = np.array([by_variant[b]["_per_clip_mse"][b_idx[k]] for k in shared])
                print(f"  [probe_future_mse] {a} vs {b} keys disagree by {max_drop:.1%} "
                      f"— re-aligned to {len(shared)} shared clips (under 5% threshold)")
            else:
                a_arr = by_variant[a]["_per_clip_mse"]
                b_arr = by_variant[b]["_per_clip_mse"]
            delta = a_arr - b_arr             # >0 means a worse than b on MSE (lower=better)
            bca = paired_bca(delta)
            pairwise_deltas[f"{a}_minus_{b}"] = {
                "n": int(len(delta)),
                "delta_mean": round(float(delta.mean()), 6),
                "delta_ci_lo": round(float(bca["ci_lo"]), 6),
                "delta_ci_hi": round(float(bca["ci_hi"]), 6),
                "delta_ci_half": round(float(bca["ci_half"]), 6),
                "p_value": float(bca["p_value_vs_zero"]),
                "interpretation": (
                    f"{a} - {b} > 0 means {a} has HIGHER MSE (worse future-frame "
                    f"prediction) than {b}; lower MSE = better."
                ),
            }

    # Strip ndarray fields before serialising
    serial_by_variant = {}
    for v, entry in by_variant.items():
        if entry is None:
            serial_by_variant[v] = None
        else:
            serial_by_variant[v] = {
                "mse_mean": entry["mse_mean"],
                "mse_std":  entry["mse_std"],
                "mse_ci":   entry["mse_ci"],
                "n":        entry["n"],
            }
    serial_by_variant["dinov2"] = "n/a — no future-frame predictor"

    out = {
        "metric": "future_frame_l1_loss",
        "by_variant": serial_by_variant,
        "pairwise_deltas": pairwise_deltas,
    }
    save_json_checkpoint(out, args.output_root / "probe_future_mse_per_variant.json")
    log_metrics(wb, {"n_variants_with_data": len(available)})
    print(json.dumps(out, indent=2))


# ── CLI ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Future-frame latent prediction MSE (probe future_mse — V-JEPA-only diagnostic)")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--stage", required=True, choices=["forward", "paired_per_variant"])
    p.add_argument("--variant", choices=list(KNOWN_VARIANTS), default=None,
                   help="V-JEPA variant whose ckpt is loaded for forward stage")
    p.add_argument("--encoder-ckpt", type=Path, default=None,
                   help="V-JEPA .pt holding encoder + predictor (target_encoder + predictor keys)")
    add_local_data_arg(p)
    p.add_argument("--action-probe-root", type=Path, default=None,
                   help="probe_action output dir (provides action_labels.json test split)")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES_DEFAULT)
    p.add_argument("--seed", type=int, default=99)
    add_cache_policy_arg(p)
    add_wandb_args(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not (args.SANITY or args.POC or args.FULL):
        sys.exit("ERROR: specify --SANITY, --POC, or --FULL")
    args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)
    args.output_root.mkdir(parents=True, exist_ok=True)
    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb = init_wandb(f"probe_future_mse_{args.stage}", mode,
                    config=vars(args), enabled=not args.no_wandb)
    try:
        torch.manual_seed(args.seed)
        if args.stage == "forward":
            t0 = time.time()
            run_forward_stage(args, wb)
            print(f"forward stage: {time.time() - t0:.0f}s")
        elif args.stage == "paired_per_variant":
            run_paired_per_variant_stage(args, wb)
    finally:
        finish_wandb(wb)


if __name__ == "__main__":
    # Fail-fast: any uncaught exception → traceback + sys.exit(1) so the
    # parent shell (run_eval.sh under `set -e`) sees non-zero and aborts the
    # chain. Mirrors m10_sam_segment.py pattern (errors_N_fixes #14/#16).
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        print(f"\n❌ FATAL: {Path(__file__).name} crashed — see traceback below", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
