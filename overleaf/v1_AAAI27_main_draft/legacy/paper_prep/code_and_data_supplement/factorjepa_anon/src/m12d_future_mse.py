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
    python -u src/m12d_future_mse.py --FULL \\
        --stage forward --variant vjepa_2_1_frozen \\
        --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \\
        --action-probe-root outputs/full/probe_action \\
        --local-data data/eval_10k_local \\
        --output-root outputs/full/probe_future_mse \\
        --cache-policy 1 2>&1 | tee logs/probe_future_mse_forward_vjepa.log

    python -u src/m12d_future_mse.py --FULL \\
        --stage paired_per_variant \\
        --output-root outputs/full/probe_future_mse --cache-policy 1
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils.action_labels import load_action_labels
from utils.bootstrap import bootstrap_ci, parallel_bca
from utils.cache_policy import (
    add_cache_policy_arg,
    guarded_delete,
    resolve_cache_policy_interactive,
)
from utils.checkpoint import save_array_checkpoint, save_json_checkpoint
from utils.config import add_local_data_arg, check_gpu, get_pipeline_config
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.decode_feeder import BatchFeeder, DecodeFeeder
from utils.frozen_features import (
    ENCODERS,
    decode_to_tensor,
)
from utils.predictor_eval import (
    build_mask_gen,
    load_encoder_predictor,
    masked_predict_l1,
    to_pixel,
)
from utils.gpu_batch import AdaptiveBatchSizer, cleanup_temp, cuda_cleanup
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog
from utils.progress import make_pbar
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics


# ── Constants ─────────────────────────────────────────────────────────

_PCFG = get_pipeline_config()
# iter16 §3.2 DEDUP: the encoder/predictor/mask-gen builders + the masked-predict-L1
# core moved to utils/predictor_eval.py (single source, shared with m12e). The
# predictor-arch (PRED_*, NUM_MASK_TOKENS, PATCH/TUBELET) + mask-scale (DEFAULT_*)
# constants that ONLY those builders used now live there too — removed here to keep
# one source of truth. Constants still used by THIS module's forward loop stay below.
from utils.config import get_model_config as _get_mcfg
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)

# iter18 W7 (PLR2004): semantic named constants.
_PENDING_FLUSH_N = 8         # GPU->CPU transfer batching (matches utils.frozen_features)
_MIN_COMPARABLE = 2
_MAX_KEY_DISAGREE_FRAC = 0.05   # FATAL if encoder key sets diverge beyond this
_MODEL_CFG     = _get_mcfg(None)["model"]              # None → default vjepa2_1.yaml
NUM_FRAMES_DEFAULT = _PCFG["probe"]["num_frames"]
CROP           = _MODEL_CFG["crop_size"]                # used by decode_to_tensor (V-JEPA 2.1 384)
CHECKPOINT_EVERY = _PCFG["streaming"]["checkpoint_every"]

# Variants understood by --paired_per_variant. P1 ships frozen only; P2 adds the
# m09a continual-pretrain output; P3 adds the m09c factor-surgery output. m09b
# ExPLoRA was dropped per iter13 pivot — see plan_code_dev.md §"P2 + P3 Coding plan".
# iter15 Phase 8 (2026-05-17): runtime-discovered V-JEPA variant list. Single
# source of truth is configs/eval/probe_encoders.yaml (loaded by
# utils.frozen_features._load_encoders_registry → ENCODERS dict). Filtered to
# kind=="vjepa" because Stage 9 future_mse requires a JEPA predictor (dinov2
# has none). Per src/CLAUDE.md "No hardcoded values in Python — YAML or
# runtime discovery only" — replaces the prior hardcoded tuple that silently
# dropped vjepa_2_1_pretrain_2X_encoder from Δ3 paired-Δ aggregation when the
# variant was added to the YAML registry but the tuple was not updated.
# Adding a new variant: edit configs/eval/probe_encoders.yaml — this list
# and the argparse --variant choices below pick it up automatically.
KNOWN_VARIANTS = tuple(
    name for name, spec in ENCODERS.items() if spec.get("kind") == "vjepa"
)


# ── Encoder / predictor / mask-generator loaders → utils/predictor_eval.py ──
# iter16 §3.2 DEDUP: _load_vjepa_2_1_encoder_hierarchical + _load_predictor_2_1 +
# _build_mask_gen lived here as a copy of predictor_eval's builders. They are now
# imported from utils.predictor_eval (load_encoder_predictor / build_mask_gen) —
# single source shared with m12e. The construction args + FATAL guards are
# byte-identical (verified by line diff); only the duplication was removed.


# ── Per-batch forward ─────────────────────────────────────────────────

@torch.no_grad()
def _forward_one_batch(encoder, predictor, mask_gen, batch: torch.Tensor):
    """Sample masks for the batch, then delegate the encode→predict→L1 core to
    utils.predictor_eval.masked_predict_l1 (iter16 §3.2 DEDUP — single source shared
    with m12e). This wrapper keeps the (encoder, predictor, mask_gen, batch) signature
    its callers use and the (per_clip_l1, out, h_target) 3-tuple the motion_aux path
    consumes; only the duplicated forward body was removed.

    Returns: (per_clip_l1 (B,) np.float32, out, h_target).
    """
    B = batch.shape[0]
    # _MaskGenerator(B) → (masks_enc_list, masks_pred_list); stack each to (B, n_tokens).
    m_enc_raw, m_pred_raw = mask_gen(B)
    m_enc = torch.stack(m_enc_raw, dim=0).to("cuda") if isinstance(m_enc_raw, list) else m_enc_raw.to("cuda")
    m_pred = torch.stack(m_pred_raw, dim=0).to("cuda") if isinstance(m_pred_raw, list) else m_pred_raw.to("cuda")
    pixel = to_pixel(batch)                                       # (B,3,T,H,W) bf16 cuda
    return masked_predict_l1(encoder, predictor, pixel, m_enc, m_pred)


# ── Stage 1 — forward on test split ───────────────────────────────────

def _save_mse_ckpt(ckpt_file: Path, mse_acc, keys_acc, ma_acc=None) -> None:
    """Atomic .npz checkpoint. Suffix is .tmp.npz (not .npz.tmp) —
    np.savez auto-appends .npz when the path doesn't end in .npz, which
    would silently rename our tmp behind our back. See errors_N_fixes.md #82.

    D2 (2026-05-16): when `ma_acc` is provided, also persists the parallel
    motion_aux-augmented per-clip L1 accumulator so resume keeps the two
    streams in lockstep length-wise. Mismatch → FATAL on next flush.
    """
    if not mse_acc:
        return
    tmp = ckpt_file.with_suffix(".tmp.npz")
    payload = dict(
        mse=np.array(mse_acc, dtype=np.float32),
        keys=np.array(keys_acc, dtype=object),
    )
    if ma_acc is not None:
        if len(ma_acc) != len(mse_acc):
            sys.exit(f"FATAL: ma_acc length {len(ma_acc)} != mse_acc length {len(mse_acc)} at ckpt save")
        payload["ma"] = np.array(ma_acc, dtype=np.float32)
    np.savez(tmp, **payload)
    os.replace(tmp, ckpt_file)


def _flush_batch_forward(batch, pending_keys, encoder, predictor, mask_gen,
                         sizer, mse_acc, keys_acc, pbar,
                         ma_head=None, ma_acc=None, embed_dim=None):
    """Sub-batch with OOM-retry. Mirrors frozen_features._flush_batch.

    D2 (2026-05-16): when `ma_head` is provided, also accumulates per-clip L1
    in motion_aux-augmented feature space (mean-pool over tokens → ma_head →
    L1 between pooled-augmented pred + target). Appends to `ma_acc`.

    iter19 2026-07-09 rungs 2+3: `batch` arrives pre-STACKED + pinned from BatchFeeder (the
    0.1-0.2s main-thread torch.stack moved to the builder thread) and is uploaded ONCE here —
    the sizer's sub-batches become cuda views and to_pixel's .to("cuda") a no-op move.
    Values byte-identical: stack + slice are pure data movement.
    """
    batch = batch.to("cuda", non_blocking=True)
    n_total = batch.shape[0]
    i = 0
    while i < n_total:
        sub = batch[i:i + sizer.size]
        try:
            per_clip_l1, out, h_target = _forward_one_batch(encoder, predictor, mask_gen, sub)
        except torch.cuda.OutOfMemoryError:
            cuda_cleanup()
            if not sizer.on_oom():
                raise
            continue
        # D2: motion_aux-augmented per-clip L1 (parallel metric, not replacing the above).
        per_clip_ma_l1 = None
        if ma_head is not None and embed_dim is not None:
            # Take LAST embed_dim columns of the hierarchical concat (matches
            # probe_trio.py:225 convention) → (B, N, D) → mean over tokens → (B, D).
            out_last = out[..., -embed_dim:].float().mean(dim=1)               # (B, D)
            tgt_last = h_target[..., -embed_dim:].float().mean(dim=1)          # (B, D)
            with torch.no_grad():
                out_logits, out_vec = ma_head(out_last)
                tgt_logits, tgt_vec = ma_head(tgt_last)
            ma_pred = torch.cat([out_logits, out_vec], dim=-1)                 # (B, K+n_dims)
            ma_tgt  = torch.cat([tgt_logits, tgt_vec], dim=-1)                 # (B, K+n_dims)
            per_clip_ma_l1 = (ma_pred - ma_tgt).abs().mean(dim=-1).cpu().numpy()  # (B,)
        for r in range(per_clip_l1.shape[0]):
            mse_acc.append(float(per_clip_l1[r]))
            if ma_acc is not None and per_clip_ma_l1 is not None:
                ma_acc.append(float(per_clip_ma_l1[r]))
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
    print_cgroup_header(prefix="[m12d_future_mse]")
    start_oom_watchdog(prefix="[m12d_future_mse]-oom-watchdog")
    cleanup_temp()
    ensure_local_data(args)

    out_dir = args.output_root / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mse = out_dir / artifact("per_clip_mse")
    out_keys = out_dir / artifact("clip_keys")
    out_agg = out_dir / artifact("aggregate_mse")
    if out_mse.exists() and out_keys.exists() and args.cache_policy == "1":
        print(f"  [keep] {out_mse} present -- skipping (--cache-policy 2 to redo)")
        return
    guarded_delete(out_mse, args.cache_policy, "per_clip_mse")
    guarded_delete(out_keys, args.cache_policy, "clip_keys")
    guarded_delete(out_agg, args.cache_policy, "aggregate_mse")

    labels = load_action_labels(args.action_probe_root / artifact("action_labels"))
    test_keys = [k for k, info in labels.items() if info["split"] == "test"]
    print(f"Forward on {len(test_keys)} test clips, variant={args.variant}")

    # Encoder + predictor + mask-gen
    # iter16 §3.2 DEDUP: one combined load (encoder hierarchical + predictor) from the
    # single source in utils.predictor_eval (shared with m12e) — was two separate calls.
    encoder, predictor, embed_dim_concat = load_encoder_predictor(
        args.encoder_ckpt, args.num_frames, args.model_config)
    # concat-dim validity (== embed_dim * n_output_distillation) is checked per-config inside
    # load_encoder_only (WS-B3) — no ViT-G-hardcoded re-check here.
    mask_gen = build_mask_gen(args.num_frames)
    print(f"  predictor: {sum(p.numel() for p in predictor.parameters()) / 1e6:.1f}M params")

    # D2 (2026-05-16): optional motion_aux head augment → parallel per-clip L1
    # in (K+n_dims) space. None → original encoder+predictor JEPA-only path.
    ma_head = None
    embed_dim = _get_mcfg(args.model_config)["model"]["embed_dim"]   # last-layer dim (slice hierarchical concat)
    if args.motion_aux_head is not None:
        from utils.motion_aux_loss import load_motion_aux_head
        ma_head = load_motion_aux_head(args.motion_aux_head, device="cuda")
        print(f"  [motion_aux] augment ON: head={args.motion_aux_head} "
              f"(K={ma_head.n_motion_classes}, n_dims={ma_head.n_motion_dims}) → "
              f"will emit per_clip_motion_aux_l1.npy alongside per_clip_mse.npy")

    # Resume
    ckpt_file = out_dir / ".probe_future_mse_ckpt.npz"
    mse_acc, keys_acc, processed = [], [], set()
    ma_acc = [] if ma_head is not None else None    # D2: parallel ma-L1 accumulator (None when augment OFF)
    if ckpt_file.exists():
        try:
            data = np.load(ckpt_file, allow_pickle=True)
            mse_acc = list(data["mse"].tolist())
            keys_acc = list(data["keys"])
            processed = set(keys_acc)
            print(f"  Resume: {len(processed)} clips already cached")
            if ma_head is not None:
                if "ma" not in data.files:
                    sys.exit("FATAL: --motion-aux-head set but resume ckpt has no 'ma' array — "
                             "delete .probe_future_mse_ckpt.npz to redo from scratch (FAIL LOUD: cache_policy).")
                ma_acc = list(data["ma"].tolist())
                if len(ma_acc) != len(mse_acc):
                    sys.exit(f"FATAL: resume ckpt mse({len(mse_acc)}) != ma({len(ma_acc)}) — corrupt cache")
        except SystemExit:
            raise
        except Exception as e:
            print(f"  WARN: resume ckpt corrupt ({e}), starting fresh")
            mse_acc, keys_acc, processed = [], [], set()
            ma_acc = [] if ma_head is not None else None

    target_keys = set(test_keys) - processed
    if not target_keys:
        print("  All clips already processed -- skipping forward")
    else:
        sizer = AdaptiveBatchSizer(
            initial_size=4, min_size=1, max_size=8,
            memory_cap=_PCFG["gpu"]["gpu_memory_target"],
        )
        print(f"  AdaptiveBatchSizer: start={sizer.size}, max=8 (encoder + predictor heavier than encoder-alone)")

        pbar = make_pbar(total=len(test_keys), desc="m12d_future_mse",
                         unit="clip", initial=len(processed))
        tmp_base = out_dir / "tmp_decode"
        tmp_base.mkdir(parents=True, exist_ok=True)
        tmp_dir = tempfile.mkdtemp(dir=tmp_base)
        clip_q, tar_stop, _reader = iter_clips_parallel(
            local_data=args.local_data, subset_keys=target_keys, processed_keys=processed)

        n_since_ckpt = 0
        # iter19 2026-07-09: zero-idle DecodeFeeder (utils/decode_feeder — SHARED by m12d/e/f).
        # m12d was the last SERIAL decoder: one decode_to_tensor per clip in the main thread
        # (~150ms/clip even on cache hits — the torch resize runs on every read) between predictor
        # forwards. Feeder workers decode continuously; pixels BYTE-IDENTICAL (same call, same args).
        feeder = DecodeFeeder(
            clip_q,
            decode_one=lambda ck, mp4: decode_to_tensor(mp4, tmp_dir, ck, args.num_frames, CROP),
            n_workers=max(1, args.decode_workers), ready_depth=_PENDING_FLUSH_N * 3, timeout_s=300)
        # rung 2 (07-09): builder thread stacks+pins batch k+1 while k forwards; tail batch
        # published as-is, so the old pending-list accumulation + post-loop flush both go away.
        batches = BatchFeeder(feeder, batch_size=_PENDING_FLUSH_N, depth=2, pin_memory=True)
        try:
            for pending_keys, batch_t in batches:
                _flush_batch_forward(batch_t, pending_keys,
                                     encoder, predictor, mask_gen,
                                     sizer, mse_acc, keys_acc, pbar,
                                     ma_head=ma_head, ma_acc=ma_acc, embed_dim=embed_dim)
                n_since_ckpt += len(pending_keys)
                if n_since_ckpt >= CHECKPOINT_EVERY:
                    _save_mse_ckpt(ckpt_file, mse_acc, keys_acc, ma_acc=ma_acc)
                    n_since_ckpt = 0
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

    # D2 (2026-05-16): persist motion_aux-augmented per-clip L1 alongside the
    # JEPA per_clip_mse.npy. Paired-Δ stage below will key off this file when
    # `--motion-aux-head` was set on the upstream `forward` invocation.
    if ma_acc is not None:
        if len(ma_acc) != len(mse_acc):
            sys.exit(f"FATAL: ma_acc length {len(ma_acc)} != mse_acc length {len(mse_acc)} at final save")
        ma_arr = np.asarray(ma_acc, dtype=np.float32)
        out_ma = out_dir / artifact("per_clip_motion_aux_l1")
        save_array_checkpoint(ma_arr, out_ma)
        ma_ci = bootstrap_ci(ma_arr.astype(np.float64))
        save_json_checkpoint({
            "variant": args.variant, "n_test": int(len(ma_arr)),
            "ma_l1_mean": round(float(ma_arr.mean()), 6),
            "ma_l1_std":  round(float(ma_arr.std()),  6),
            "ma_l1_ci":   ma_ci,
            "num_frames": args.num_frames,
            "predictor_loaded_from": str(args.encoder_ckpt),
            "motion_aux_head_loaded_from": str(args.motion_aux_head),
            "augment_space_dim": int(ma_head.n_motion_classes + ma_head.n_motion_dims),
        }, out_dir / artifact("aggregate_motion_aux_l1"))
        print(f"  per_clip_motion_aux_l1: mean={ma_arr.mean():.4f}  std={ma_arr.std():.4f}  N={len(ma_arr)}")
        log_metrics(wb, {f"{args.variant}_ma_l1_mean": float(ma_arr.mean()),
                         f"{args.variant}_ma_l1_std":  float(ma_arr.std())})


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
        mse_path = var_dir / artifact("per_clip_mse")
        keys_path = var_dir / artifact("clip_keys")
        agg_path = var_dir / artifact("aggregate_mse")
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
    if len(available) >= _MIN_COMPARABLE:
        _units, _slots = [], []   # iter19: collect deltas → ONE parallel_bca (byte-identical fixed-seed BCa)
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
                if max_drop > _MAX_KEY_DISAGREE_FRAC:
                    print(f"❌ FATAL [m12d_future_mse]: {a} vs {b} keys disagree by "
                          f"{max_drop:.1%} (>5% threshold).", file=sys.stderr)
                    print(f"   {a}: {len(ka)} keys  |  {b}: {len(kb)} keys  |  "
                          f"intersection: {len(shared)}", file=sys.stderr)
                    print("   Re-extract per-clip MSE on the same eval-subset for both encoders.",
                          file=sys.stderr)
                    sys.exit(3)
                a_arr = np.array([by_variant[a]["_per_clip_mse"][a_idx[k]] for k in shared])
                b_idx = {k: i for i, k in enumerate(kb)}
                b_arr = np.array([by_variant[b]["_per_clip_mse"][b_idx[k]] for k in shared])
                print(f"  [m12d_future_mse] {a} vs {b} keys disagree by {max_drop:.1%} "
                      f"— re-aligned to {len(shared)} shared clips (under 5% threshold)")
            else:
                a_arr = by_variant[a]["_per_clip_mse"]
                b_arr = by_variant[b]["_per_clip_mse"]
            delta = a_arr - b_arr             # >0 means a worse than b on MSE (lower=better)
            _units.append(delta)
            _slots.append((a, b, int(len(delta)), round(float(delta.mean()), 6)))
        for (a, b, n, dmean), bca in zip(_slots, parallel_bca(_units)):
            pairwise_deltas[f"{a}_minus_{b}"] = {
                "n": n,
                "delta_mean": dmean,
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
    save_json_checkpoint(out, args.output_root / artifact("probe_future_mse_per_variant"))
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
    p.add_argument("--model-config", type=str, default=None,
                   help="configs/model/<backbone>.yaml — encoder+predictor arch/dims (WS-B3 arch-aware). "
                        "Required for --stage forward; unused by --stage paired_per_variant (None → ViT-G).")
    # D2 fix (2026-05-16): wire motion_aux head for SYMMETRY with the rest of
    # the eval pipeline (probe_action / probe_motion_cos / probe_future_regress
    # all accept this flag). When provided, ADDS a parallel metric:
    # per_clip_motion_aux_l1.npy = ||ma_head(pool(out)) − ma_head(pool(target))||_1
    # in augmented (K + n_dims) space. The original per_clip_mse.npy is
    # UNCHANGED (encoder+predictor JEPA-only path preserved → iter14
    # reproducibility intact). Downstream paired-Δ stages can use either file.
    p.add_argument("--motion-aux-head", type=Path, default=None,
                   help="Path to motion_aux_head.pt. Emits parallel "
                        "per_clip_motion_aux_l1.npy alongside the JEPA per_clip_mse.npy. "
                        "None → encoder+predictor JEPA-only path (iter14 baseline).")
    add_local_data_arg(p)
    p.add_argument("--action-probe-root", type=Path, default=None,
                   help="probe_action output dir (provides action_labels.json test split)")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES_DEFAULT)
    p.add_argument("--decode-workers", type=int,
                   default=_PCFG["probe"]["future_mse"]["decode_workers"],
                   help="parallel clip-decode threads feeding the forward loop (iter19 zero-idle "
                        "DecodeFeeder). Default: pipeline.yaml probe.future_mse.decode_workers.")
    p.add_argument("--seed", type=int, default=_PCFG["probe"]["seed"])   # was 99 literal
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
    wb = init_wandb(f"m12d_future_mse_{args.stage}", mode,
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
