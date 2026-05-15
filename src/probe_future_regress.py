"""Future-frame latent regressor probe — trained regressor head per encoder. GPU-only.
Gold standard: https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/train.py
Claude Code: re-WebSearch this URL on every read of this file.

iter15 Phase 2 (2026-05-14): replaces probe_future_mse.py for variants where the
PREDICTOR is frozen at Meta's (iter15 head-only m09a2 + m09c2 variants). Per encoder:
  context = encoder(x[t:t+8])    # both forwards under torch.no_grad — encoder frozen
  target  = encoder(x[t+8:t+16])
  train a small regressor head (linear / mlp_d1 / mlp_d2) to predict target from
  context on the train split; evaluate per-clip L1 on the test split.

Per-variant Δ comes from data exposure during regressor training: --data-source
{raw, factor_aug} chooses train-time clip source. Surgery head variants pair with
factor_aug; pretrain_head + frozen pair with raw.

Stages:
  forward             — encode + train regressor + dump per-clip L1   [GPU, ~30 min/encoder]
  paired_per_variant  — pairwise BCa Δ across {frozen, pretrain_head,
                        surgical_3stage_DI_head, surgical_noDI_head}  [CPU]

USAGE:
    python -u src/probe_future_regress.py --SANITY \\
        --stage forward --variant vjepa_2_1_frozen \\
        --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \\
        --data-source raw --regressor-arch linear \\
        --action-probe-root outputs/sanity/probe_action \\
        --local-data data/eval_10k_local \\
        --output-root outputs/sanity/probe_future_regress \\
        --cache-policy 2 2>&1 | tee logs/probe_future_regress_sanity_$(date +%Y%m%d_%H%M%S).log

    python -u src/probe_future_regress.py --FULL \\
        --stage forward --variant vjepa_2_1_surgical_3stage_DI_head \\
        --encoder-ckpt outputs/full/m09c_surgery_3stage_DI_head/3stage_DI_head/student_encoder.pt \\
        --data-source factor_aug --regressor-arch mlp_d1 \\
        --action-probe-root outputs/full/probe_action \\
        --local-data data/eval_10k_local \\
        --output-root outputs/full/probe_future_regress \\
        --cache-policy 1 2>&1 | tee logs/probe_future_regress_3stage_DI_head_$(date +%Y%m%d_%H%M%S).log

    python -u src/probe_future_regress.py --FULL --stage paired_per_variant \\
        --output-root outputs/full/probe_future_regress --cache-policy 1
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import queue
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from utils.action_labels import load_action_labels
from utils.bootstrap import bootstrap_ci, paired_bca
from utils.cache_policy import (
    add_cache_policy_arg, guarded_delete, resolve_cache_policy_interactive,
)
from utils.checkpoint import save_array_checkpoint, save_json_checkpoint
from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog
from utils.config import add_local_data_arg, check_gpu, get_pipeline_config
from utils.data_download import ensure_local_data, iter_clips_parallel
from utils.frozen_features import (
    decode_to_tensor, load_vjepa_2_1_frozen,
)
from utils.gpu_batch import cleanup_temp, cuda_cleanup
from utils.progress import make_pbar
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics


# ── Constants ─────────────────────────────────────────────────────────

_PCFG = get_pipeline_config()
NUM_FRAMES_DEFAULT = 16
CROP = 384
EMBED_DIM = 1664              # V-JEPA 2.1 ViT-G output dim
REGRESSOR_LR = 1e-3
REGRESSOR_WD = 0.05
REGRESSOR_EPOCHS = 50
REGRESSOR_BATCH = 64
REGRESSOR_MLP_HIDDEN = 4096

# Variants understood by paired_per_variant. iter15 adds 4 head-only variants
# alongside iter14's 4 encoder-update variants. Stage `forward` runs ONE variant
# at a time (the user picks via --variant).
KNOWN_VARIANTS = (
    "vjepa_2_1_frozen",
    "vjepa_2_1_pretrain_encoder",                       # iter14 m09a1 continual SSL (5 epochs)
    "vjepa_2_1_pretrain_2X_encoder",                    # iter14 arm C — m09a1 at 10 epochs (Δ3 control)
    "vjepa_2_1_surgical_3stage_DI_encoder",             # iter14 m09c1 D_I surgery
    "vjepa_2_1_surgical_noDI_encoder",                  # iter14 m09c1 noDI surgery
    "vjepa_2_1_pretrain_head",                  # iter15 m09a2 head-only pretrain
    "vjepa_2_1_surgical_3stage_DI_head",        # iter15 m09c2 D_I head-only surgery
    "vjepa_2_1_surgical_noDI_head",             # iter15 m09c2 noDI head-only surgery
)
DATA_SOURCES = ("raw", "factor_aug")
REGRESSOR_ARCHS = ("linear", "mlp_d1", "mlp_d2")


# ── Regressor architectures ───────────────────────────────────────────

def build_regressor(arch: str, embed_dim: int) -> nn.Module:
    """1-layer linear, 2-layer MLP, or 3-layer MLP head — predict target from context.

    Arch        Params      Approximate parameter count at embed_dim=1664
    linear      D²          2.8 M
    mlp_d1      D×H + H×D   13.6 M  (H=4096)
    mlp_d2      D×H + H×H + H×D   30.4 M
    """
    if arch == "linear":
        return nn.Linear(embed_dim, embed_dim)
    h = REGRESSOR_MLP_HIDDEN
    if arch == "mlp_d1":
        return nn.Sequential(
            nn.Linear(embed_dim, h),
            nn.GELU(),
            nn.Linear(h, embed_dim),
        )
    if arch == "mlp_d2":
        return nn.Sequential(
            nn.Linear(embed_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, embed_dim),
        )
    sys.exit(f"FATAL: unknown --regressor-arch '{arch}'; expected one of {REGRESSOR_ARCHS}")


# ── Encoder feature extraction (context + target windows) ─────────────

def _encode_window(model, clip_tensor: torch.Tensor, t_start: int, t_end: int,
                    device: torch.device, dtype) -> torch.Tensor:
    """Forward a frame window through the frozen encoder; mean-pool tokens → (D,).

    `clip_tensor` is (T, C, H, W) on CPU (per decode_to_tensor at
    utils/frozen_features.py:178). We slice T → [t_start:t_end], batch +
    permute to (1, C, t, H, W) which is what V-JEPA's patch_embed Conv3d
    expects (mirrors utils/frozen_features.py:194 forward_vjepa + probe_
    future_mse.py:237 baselines). Mean-pool over tokens → (embed_dim,).

    iter15 Phase 5 V5 fix (2026-05-15): prior code sliced clip_tensor[:,
    t_start:t_end] assuming (C, T, H, W) layout and omitted the permute,
    sending (1, T, C, H, W) to the encoder → conv3d sees 16 channels, fails.
    Same Phase-2 cosmetic-rewrite bug class as m09a2/c2 #7.
    """
    window = clip_tensor[t_start:t_end].unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous().to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        feats = model(window)                                       # (1, N_tokens, D)
        if feats.dim() == 3 and feats.shape[-1] != EMBED_DIM:
            # If model returns hierarchical (1, N, 4D), reshape and mean across layers.
            # The frozen-only loader (utils.frozen_features.load_vjepa_2_1_frozen)
            # already returns last-layer (D); the dim check above is defensive.
            n_concat = feats.shape[-1] // EMBED_DIM
            if n_concat * EMBED_DIM != feats.shape[-1]:
                sys.exit(f"FATAL: encoder output dim {feats.shape[-1]} not a multiple of {EMBED_DIM}")
            feats = feats.view(*feats.shape[:-1], n_concat, EMBED_DIM).mean(dim=-2)
    pooled = feats.float().mean(dim=1).squeeze(0)                   # (D,)
    return pooled.cpu()


def _extract_features_for_clips(model, args, clip_keys: list, label: str,
                                  num_frames: int, device: torch.device, dtype):
    """Iterate clips, decode mp4, encode context + target windows, return
    (ctx_feats, tgt_feats, ordered_keys) where ctx/tgt have shape (N, D).
    """
    if num_frames % 2 != 0:
        sys.exit(f"FATAL: --num-frames must be even (split into ctx/tgt halves); got {num_frames}")
    half = num_frames // 2

    ctx_acc, tgt_acc, keys_acc = [], [], []
    tmp_base = args.output_root / args.variant / "tmp_decode"
    tmp_base.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_base)

    target_keys = set(clip_keys)
    clip_q, tar_stop, _reader = iter_clips_parallel(
        local_data=args.local_data, subset_keys=target_keys, processed_keys=set())

    pbar = make_pbar(total=len(clip_keys), desc=f"probe_future_regress:{label}",
                       unit="clip")
    try:
        while True:
            try:
                item = clip_q.get(timeout=300)
            except queue.Empty:
                print(f"  WARN: clip queue timeout (5 min) while building {label} features")
                break
            if item is None:
                break
            clip_key, mp4_bytes = item
            clip_t = decode_to_tensor(mp4_bytes, tmp_dir, clip_key, num_frames, CROP)
            if clip_t is None:
                print(f"  SKIP (decode fail): {clip_key}")
                pbar.update(1)
                continue
            try:
                ctx_vec = _encode_window(model, clip_t, 0, half, device, dtype)
                tgt_vec = _encode_window(model, clip_t, half, num_frames, device, dtype)
            except torch.cuda.OutOfMemoryError:
                cuda_cleanup()
                print(f"  SKIP (OOM): {clip_key} — encoder OOM on single-clip forward; investigate")
                pbar.update(1)
                continue
            ctx_acc.append(ctx_vec)
            tgt_acc.append(tgt_vec)
            keys_acc.append(clip_key)
            pbar.update(1)
    finally:
        tar_stop.set()
        pbar.close()

    if not ctx_acc:
        sys.exit(f"FATAL: 0 clips produced features for {label} — pipeline failure")
    ctx = torch.stack(ctx_acc)
    tgt = torch.stack(tgt_acc)
    return ctx, tgt, keys_acc


# ── Regressor training ────────────────────────────────────────────────

def _train_regressor(regressor: nn.Module, ctx: torch.Tensor, tgt: torch.Tensor,
                      device: torch.device, label: str) -> nn.Module:
    """AdamW + cosine + L1 loss with stop-grad on target. Returns trained regressor."""
    regressor = regressor.to(device)
    regressor.train()
    optimizer = torch.optim.AdamW(regressor.parameters(),
                                    lr=REGRESSOR_LR, weight_decay=REGRESSOR_WD)
    total_steps = REGRESSOR_EPOCHS * max(1, (ctx.shape[0] + REGRESSOR_BATCH - 1) // REGRESSOR_BATCH)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    n_train = ctx.shape[0]
    ctx_gpu = ctx.to(device)
    tgt_gpu = tgt.to(device).detach()    # stop_grad on target — encoder is frozen anyway

    pbar = make_pbar(total=REGRESSOR_EPOCHS, desc=f"train_regressor:{label}", unit="epoch")
    for epoch in range(REGRESSOR_EPOCHS):
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n_train, REGRESSOR_BATCH):
            idx = perm[i:i + REGRESSOR_BATCH]
            ctx_b = ctx_gpu[idx]
            tgt_b = tgt_gpu[idx]
            optimizer.zero_grad(set_to_none=True)
            pred = regressor(ctx_b)
            loss = (pred - tgt_b).abs().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss)
            n_batches += 1
        pbar.set_postfix({"L1": f"{epoch_loss / max(n_batches, 1):.4f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
        pbar.update(1)
    pbar.close()
    regressor.eval()
    return regressor


# ── Stage 1 — forward (per encoder) ───────────────────────────────────

def run_forward_stage(args, wb) -> None:
    """Encode train+test windows, train regressor on train, dump per-clip L1 on test.

    Outputs:
        <output_root>/<variant>/per_clip_regressor_l1.npy   (N_test,) float32
        <output_root>/<variant>/clip_keys.npy               (N_test,) str
        <output_root>/<variant>/aggregate_regressor_l1.json {l1_mean, l1_ci, ...}
        <output_root>/<variant>/regressor.pt                trained head state_dict
    """
    if args.variant is None:
        sys.exit("FATAL: --stage forward requires --variant")
    if not args.variant.startswith("vjepa"):
        sys.exit(f"FATAL: --stage forward only supports vjepa_* variants; got {args.variant!r}")
    if args.encoder_ckpt is None:
        sys.exit("FATAL: --stage forward requires --encoder-ckpt")
    if args.local_data is None:
        sys.exit("FATAL: --stage forward requires --local-data")
    if args.action_probe_root is None:
        sys.exit("FATAL: --stage forward requires --action-probe-root (for action_labels.json splits)")
    if args.data_source is None:
        sys.exit(f"FATAL: --stage forward requires --data-source ({'|'.join(DATA_SOURCES)})")
    if args.regressor_arch is None:
        sys.exit(f"FATAL: --stage forward requires --regressor-arch ({'|'.join(REGRESSOR_ARCHS)})")

    check_gpu()
    print_cgroup_header(prefix="[probe_future_regress]")
    start_oom_watchdog(prefix="[probe_future_regress]-oom-watchdog")
    cleanup_temp()
    ensure_local_data(args)

    out_dir = args.output_root / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_l1 = out_dir / "per_clip_regressor_l1.npy"
    out_keys = out_dir / "clip_keys.npy"
    out_agg = out_dir / "aggregate_regressor_l1.json"
    out_reg = out_dir / "regressor.pt"
    if (out_l1.exists() and out_keys.exists() and out_agg.exists() and out_reg.exists()
            and args.cache_policy == "1"):
        print(f"  [keep] {out_l1} present — skipping (--cache-policy 2 to redo)")
        return
    guarded_delete(out_l1, args.cache_policy, "per_clip_regressor_l1")
    guarded_delete(out_keys, args.cache_policy, "clip_keys")
    guarded_delete(out_agg, args.cache_policy, "aggregate_regressor_l1")
    guarded_delete(out_reg, args.cache_policy, "regressor.pt")

    # Load action_labels splits — gives us train (regressor training) + test (eval).
    labels = load_action_labels(args.action_probe_root / "action_labels.json")
    train_keys = [k for k, info in labels.items() if info["split"] == "train"]
    test_keys = [k for k, info in labels.items() if info["split"] == "test"]
    if not train_keys:
        sys.exit("FATAL: 0 train clips in action_labels.json — re-run probe_action --stage labels")
    if not test_keys:
        sys.exit("FATAL: 0 test clips in action_labels.json — re-run probe_action --stage labels")
    print(f"Forward on variant={args.variant}  arch={args.regressor_arch}  "
          f"data_source={args.data_source}  train={len(train_keys)} test={len(test_keys)}")

    # Load frozen encoder (the eval-side loader returns the LAST layer = D=1664).
    model, crop, embed_dim = load_vjepa_2_1_frozen(args.encoder_ckpt, args.num_frames)
    if crop != CROP:
        sys.exit(f"FATAL: encoder crop {crop} != module CROP {CROP}")
    if embed_dim != EMBED_DIM:
        sys.exit(f"FATAL: encoder embed_dim {embed_dim} != module EMBED_DIM {EMBED_DIM}")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # === Phase A: extract (ctx, tgt) features for train + test ===
    ctx_train, tgt_train, keys_train = _extract_features_for_clips(
        model, args, train_keys, "train", args.num_frames, device, dtype)
    ctx_test, tgt_test, keys_test = _extract_features_for_clips(
        model, args, test_keys, "test", args.num_frames, device, dtype)
    print(f"  features: train={ctx_train.shape}  test={ctx_test.shape}")

    # === Phase B: train regressor on train features ===
    regressor = build_regressor(args.regressor_arch, EMBED_DIM)
    n_reg_params = sum(p.numel() for p in regressor.parameters())
    print(f"  regressor ({args.regressor_arch}): {n_reg_params / 1e6:.1f} M params")
    regressor = _train_regressor(regressor, ctx_train, tgt_train, device,
                                   label=f"{args.variant}_{args.regressor_arch}")

    # === Phase C: evaluate on test → per-clip L1 ===
    with torch.no_grad():
        ctx_test_gpu = ctx_test.to(device)
        tgt_test_gpu = tgt_test.to(device)
        pred = regressor(ctx_test_gpu)
        per_clip_l1 = (pred - tgt_test_gpu).abs().mean(dim=1).cpu().float().numpy()

    save_array_checkpoint(per_clip_l1, out_l1)
    np.save(out_keys, np.array(keys_test, dtype=object))
    torch.save(regressor.state_dict(), out_reg)

    l1_ci = bootstrap_ci(per_clip_l1.astype(np.float64))
    save_json_checkpoint({
        "variant": args.variant,
        "data_source": args.data_source,
        "regressor_arch": args.regressor_arch,
        "regressor_params": int(n_reg_params),
        "n_train": int(ctx_train.shape[0]),
        "n_test": int(len(per_clip_l1)),
        "num_frames": args.num_frames,
        "l1_mean": round(float(per_clip_l1.mean()), 6),
        "l1_std":  round(float(per_clip_l1.std()),  6),
        "l1_ci":   l1_ci,
        "encoder_ckpt": str(args.encoder_ckpt),
    }, out_agg)
    print(f"  per_clip_regressor_l1: mean={per_clip_l1.mean():.4f}  "
          f"std={per_clip_l1.std():.4f}  N={len(per_clip_l1)}")
    log_metrics(wb, {f"{args.variant}_regressor_l1_mean": float(per_clip_l1.mean()),
                       f"{args.variant}_regressor_l1_std":  float(per_clip_l1.std()),
                       f"{args.variant}_n_test":            int(len(per_clip_l1))})


# ── Stage 2 — paired Δ across variants (CPU) ──────────────────────────

def run_paired_per_variant_stage(args, wb) -> None:
    """Pairwise BCa Δ across discovered variants on the test split.

    Output: <output_root>/probe_future_regress_per_variant.json. Mirrors
    probe_future_mse.py:run_paired_per_variant_stage with renamed fields
    (l1_* instead of mse_*) + interpretation flipped (lower L1 = better).
    """
    by_variant = {}
    for v in KNOWN_VARIANTS:
        var_dir = args.output_root / v
        l1_path = var_dir / "per_clip_regressor_l1.npy"
        keys_path = var_dir / "clip_keys.npy"
        agg_path = var_dir / "aggregate_regressor_l1.json"
        if not (l1_path.exists() and keys_path.exists() and agg_path.exists()):
            by_variant[v] = None
            continue
        agg = json.loads(agg_path.read_text())
        by_variant[v] = {
            "l1_mean": agg["l1_mean"],
            "l1_std":  agg["l1_std"],
            "l1_ci":   agg["l1_ci"],
            "n":       agg["n_test"],
            "data_source":    agg["data_source"],
            "regressor_arch": agg["regressor_arch"],
            "_per_clip_l1":   np.load(l1_path).astype(np.float64),
            "_keys":          np.load(keys_path, allow_pickle=True),
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
                a_idx = {k: i for i, k in enumerate(ka)}
                shared = [k for k in ka if k in set(kb)]
                drop_pct_a = 1.0 - (len(shared) / max(len(ka), 1))
                drop_pct_b = 1.0 - (len(shared) / max(len(kb), 1))
                max_drop = max(drop_pct_a, drop_pct_b)
                if max_drop > 0.05:
                    print(f"FATAL [probe_future_regress]: {a} vs {b} keys disagree by "
                          f"{max_drop:.1%} (>5% threshold)", file=sys.stderr)
                    print(f"   {a}: {len(ka)} keys  |  {b}: {len(kb)} keys  |  "
                          f"intersection: {len(shared)}", file=sys.stderr)
                    sys.exit(3)
                a_arr = np.array([by_variant[a]["_per_clip_l1"][a_idx[k]] for k in shared])
                b_idx = {k: i for i, k in enumerate(kb)}
                b_arr = np.array([by_variant[b]["_per_clip_l1"][b_idx[k]] for k in shared])
                print(f"  [probe_future_regress] {a} vs {b} disagree by {max_drop:.1%} "
                      f"— re-aligned to {len(shared)} shared clips (under 5% threshold)")
            else:
                a_arr = by_variant[a]["_per_clip_l1"]
                b_arr = by_variant[b]["_per_clip_l1"]
            delta = a_arr - b_arr             # >0 means a worse L1 than b (lower=better)
            bca = paired_bca(delta)
            pairwise_deltas[f"{a}_minus_{b}"] = {
                "n": int(len(delta)),
                "delta_mean": round(float(delta.mean()), 6),
                "delta_ci_lo": round(float(bca["ci_lo"]), 6),
                "delta_ci_hi": round(float(bca["ci_hi"]), 6),
                "delta_ci_half": round(float(bca["ci_half"]), 6),
                "p_value": float(bca["p_value_vs_zero"]),
                "interpretation": (
                    f"{a} - {b} > 0 means {a} has HIGHER regressor L1 (worse future-frame "
                    f"prediction) than {b}; lower L1 = better."
                ),
            }

    serial_by_variant = {}
    for v, entry in by_variant.items():
        if entry is None:
            serial_by_variant[v] = None
        else:
            serial_by_variant[v] = {
                "l1_mean": entry["l1_mean"],
                "l1_std":  entry["l1_std"],
                "l1_ci":   entry["l1_ci"],
                "n":       entry["n"],
                "data_source":    entry["data_source"],
                "regressor_arch": entry["regressor_arch"],
            }

    out = {
        "metric": "trained_regressor_future_l1",
        "by_variant": serial_by_variant,
        "pairwise_deltas": pairwise_deltas,
    }
    save_json_checkpoint(out, args.output_root / "probe_future_regress_per_variant.json")
    log_metrics(wb, {"n_variants_with_data": len(available)})
    print(json.dumps(out, indent=2))


# ── CLI ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Future-frame latent regressor probe (probe_future_regress — iter15 Phase 2)")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--stage", required=True, choices=["forward", "paired_per_variant"])
    p.add_argument("--variant", choices=list(KNOWN_VARIANTS), default=None)
    p.add_argument("--encoder-ckpt", type=Path, default=None,
                     help="encoder ckpt to load (frozen). Predictor NOT needed.")
    p.add_argument("--data-source", choices=list(DATA_SOURCES), default=None,
                     help="raw = train regressor on RAW clips through encoder; "
                          "factor_aug = train on factor-augmented clips (surgery_head variants).")
    p.add_argument("--regressor-arch", choices=list(REGRESSOR_ARCHS), default=None,
                     help="linear / mlp_d1 / mlp_d2 — head architecture for the regressor.")
    add_local_data_arg(p)
    p.add_argument("--action-probe-root", type=Path, default=None,
                     help="probe_action output dir (provides action_labels.json train/test splits)")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES_DEFAULT,
                     help="total frames per clip; split in half → context (first half) + target (second half)")
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
    wb = init_wandb(f"probe_future_regress_{args.stage}", mode,
                      config=vars(args), enabled=not args.no_wandb)
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if args.stage == "forward":
            t0 = time.time()
            run_forward_stage(args, wb)
            print(f"forward stage: {time.time() - t0:.0f}s")
        elif args.stage == "paired_per_variant":
            run_paired_per_variant_stage(args, wb)
    finally:
        finish_wandb(wb)


if __name__ == "__main__":
    # Fail-fast: uncaught exceptions → traceback + sys.exit(1) so parent shell
    # (run_eval.sh under `set -e`) sees non-zero and aborts the chain.
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        import traceback
        print(f"\n❌ FATAL: {Path(__file__).name} crashed — see traceback below", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
