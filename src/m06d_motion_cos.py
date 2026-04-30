"""Per-clip motion-feature cosine similarity to held-out same-action neighbors.

Proxy for the SSv2-style motion test on Indian clips. GPU (features) + CPU (cosine math).

For each test clip q with class c(q):
    pos_sim(q) = mean cos(emb_q, emb_n)  for n in test, c(n) == c(q), n != q
    neg_sim(q) = mean cos(emb_q, emb_n)  for n in test, c(n) != c(q)
    motion_score(q) = pos_sim(q) - neg_sim(q)        in [-2, 2]   (higher = better)

Stages:
  features      — extract MEAN-POOLED frozen features (1 vec per clip) [GPU]
                  --share-features tries to mean-pool m06d_action_probe features instead.
  cosine        — vectorised intra/inter cosine on test split (CPU, ~1 min)
  paired_delta  — paired BCa Δ = motion_score_vjepa − motion_score_dinov2 (CPU)

USAGE (priority 1 only, paths required):
    # Stage 1: features per encoder (GPU, or --share-features to reuse Module 1's cache)
    python -u src/m06d_motion_cos.py --FULL \\
        --stage features --encoder vjepa_2_1_frozen \\
        --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \\
        --eval-subset data/eval_10k.json --local-data data/eval_10k_local \\
        --action-probe-root outputs/full/m06d_action_probe \\
        --output-root outputs/full/m06d_motion_cos \\
        --share-features --cache-policy 1 2>&1 | tee logs/m06d_motion_cos_features_vjepa.log

    # Stage 2: cosine + per-clip score (CPU, ~1 min)
    python -u src/m06d_motion_cos.py --FULL \\
        --stage cosine --encoder vjepa_2_1_frozen \\
        --action-probe-root outputs/full/m06d_action_probe \\
        --output-root outputs/full/m06d_motion_cos --cache-policy 1

    # Stage 3: paired Δ (CPU, BCa 10K bootstrap)
    python -u src/m06d_motion_cos.py --FULL \\
        --stage paired_delta \\
        --output-root outputs/full/m06d_motion_cos --cache-policy 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.action_labels import load_action_labels
from utils.bootstrap import bootstrap_ci, paired_bca
from utils.cache_policy import (
    add_cache_policy_arg,
    guarded_delete,
    resolve_cache_policy_interactive,
)
from utils.checkpoint import save_array_checkpoint, save_json_checkpoint
from utils.config import add_local_data_arg, check_gpu
from utils.data_download import ensure_local_data
from utils.frozen_features import (
    ENCODERS,
    extract_features_for_keys,
    load_dinov2_frozen,
    load_vjepa_2_1_frozen,
)
from utils.gpu_batch import cleanup_temp
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics


# ── Stage 1 — features ────────────────────────────────────────────────

def _meanpool_action_probe_features(action_probe_root: Path, encoder: str,
                                    enc_dir: Path) -> tuple:
    """Reuse m06d_action_probe features by mean-pooling over the n_tokens axis.

    m06d_action_probe writes (N, n_tokens, D) per split. Motion-cos only needs (N, D).
    Returns (success: bool, log_message: str).
    """
    src_dir = action_probe_root / encoder
    src_features = src_dir / "features_test.npy"
    src_keys     = src_dir / "clip_keys_test.npy"
    if not (src_features.exists() and src_keys.exists()):
        return False, f"  [share-features] action_probe cache not found at {src_dir} -- falling back to fresh extract"

    feats = np.load(src_features)           # (N, T_tokens, D)
    keys  = np.load(src_keys, allow_pickle=True)
    if feats.ndim != 3:
        return False, f"  [share-features] expected (N, T, D), got shape {feats.shape} -- falling back"
    pooled = feats.mean(axis=1).astype(np.float32)        # (N, D)
    save_array_checkpoint(pooled, enc_dir / "pooled_features_test.npy")
    np.save(enc_dir / "clip_keys_test.npy", keys)
    return True, f"  [share-features] mean-pooled {feats.shape} -> {pooled.shape}; reused {len(keys)} clip keys"


def _fresh_extract_pooled(args, enc_dir):
    """Path B: GPU re-extract via utils.frozen_features, then mean-pool over n_tokens.

    Required when --no-share-features is set OR action_probe cache is missing.
    Reuses the SAME loaders + extractor as m06d_action_probe (no DRY violation).
    Output: pooled_features_test.npy (N, D), clip_keys_test.npy (N,) str.
    """
    if args.encoder_ckpt is None and ENCODERS[args.encoder]["kind"] == "vjepa":
        sys.exit("FATAL: fresh-extract path for V-JEPA requires --encoder-ckpt")
    if args.local_data is None:
        sys.exit("FATAL: fresh-extract path requires --local-data")
    if args.action_probe_root is None:
        sys.exit("FATAL: fresh-extract path requires --action-probe-root (for action_labels.json test split)")

    check_gpu()
    cleanup_temp()
    ensure_local_data(args)

    labels = load_action_labels(args.action_probe_root / "action_labels.json")
    test_keys = [k for k, info in labels.items() if info["split"] == "test"]
    print(f"  Fresh-extract: {len(test_keys)} test clips for encoder={args.encoder}")

    enc_kind = ENCODERS[args.encoder]["kind"]
    if enc_kind == "vjepa":
        model, crop, embed_dim = load_vjepa_2_1_frozen(args.encoder_ckpt, args.num_frames)
    elif enc_kind == "dinov2":
        model, _processor, crop, embed_dim = load_dinov2_frozen()
    else:
        sys.exit(f"FATAL: unknown encoder kind '{enc_kind}'")

    feats, ordered_keys = extract_features_for_keys(
        args, model, enc_kind, crop, embed_dim,
        test_keys, enc_dir, label="motion_cos_test",
    )
    if feats.size == 0:
        sys.exit("FATAL: fresh-extract returned 0 clips — no MP4s decoded")
    pooled = feats.mean(axis=1).astype(np.float32)            # (N, D)
    save_array_checkpoint(pooled, enc_dir / "pooled_features_test.npy")
    np.save(enc_dir / "clip_keys_test.npy", np.array(ordered_keys, dtype=object))
    print(f"  Fresh-extract: mean-pooled {feats.shape} -> {pooled.shape}")


def run_features_stage(args, wb) -> None:
    """Extract / reuse pooled (N, D) features for `args.encoder` test split.

    Path A (default): mean-pool m06d_action_probe's cached features (--share-features).
    Path B (fresh):   GPU re-extract via utils.frozen_features (--no-share-features).
    """
    if args.encoder is None:
        sys.exit("FATAL: --stage features requires --encoder")
    if args.action_probe_root is None:
        sys.exit("FATAL: --action-probe-root required (--share-features mean-pools its cache; "
                 "fresh-extract path needs its action_labels.json for the test split)")

    enc_dir = args.output_root / args.encoder
    enc_dir.mkdir(parents=True, exist_ok=True)
    out_pooled = enc_dir / "pooled_features_test.npy"
    out_keys   = enc_dir / "clip_keys_test.npy"

    if out_pooled.exists() and out_keys.exists() and args.cache_policy == "1":
        print(f"  [keep] {out_pooled} present -- skipping (--cache-policy 2 to redo)")
        return
    guarded_delete(out_pooled, args.cache_policy, "pooled_features_test")
    guarded_delete(out_keys,   args.cache_policy, "clip_keys_test")

    if args.share_features:
        ok, msg = _meanpool_action_probe_features(args.action_probe_root, args.encoder, enc_dir)
        print(msg)
        if ok:
            log_metrics(wb, {
                f"{args.encoder}_pool_method": "share_features",
                f"{args.encoder}_n_test": int(np.load(out_keys, allow_pickle=True).shape[0]),
            })
            return
        # Fall-through: action_probe cache absent → automatic fresh extract (no surprise sys.exit).
        print("  [share-features fallback] action_probe cache unavailable -> fresh GPU extract")

    # --no-share-features OR --share-features fallback path
    _fresh_extract_pooled(args, enc_dir)
    log_metrics(wb, {
        f"{args.encoder}_pool_method": "fresh_extract",
        f"{args.encoder}_n_test": int(np.load(out_keys, allow_pickle=True).shape[0]),
    })


# ── Stage 2 — vectorised intra/inter cosine ───────────────────────────

def _per_clip_motion_score(emb: np.ndarray, labels: np.ndarray) -> tuple:
    """Vectorised pos/neg cosine. emb (N, D), labels (N,) int.

    Returns (motion_score (N,), pos_mean (N,), neg_mean (N,)).
    No Python loop — `S = emb_n @ emb_n.T` then class-mask reductions.
    """
    if emb.ndim != 2:
        raise ValueError(f"emb must be (N, D); got {emb.shape}")
    if len(emb) != len(labels):
        raise ValueError(f"emb N={len(emb)} != labels N={len(labels)}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
    emb_n = emb / norms                              # L2 normalise
    S = emb_n @ emb_n.T                              # (N, N)
    same = (labels[:, None] == labels[None, :])
    np.fill_diagonal(same, False)
    diff = (~same)
    np.fill_diagonal(diff, False)
    same_count = same.sum(1).clip(min=1)
    diff_count = diff.sum(1).clip(min=1)
    pos = (S * same).sum(1) / same_count             # (N,)
    neg = (S * diff).sum(1) / diff_count             # (N,)
    score = pos - neg
    return score.astype(np.float32), pos.astype(np.float32), neg.astype(np.float32)


def run_cosine_stage(args, wb) -> None:
    """Compute per-clip motion_score on the test split for one encoder.

    Outputs:
      <output_root>/<encoder>/per_clip_motion_cos.npy  (N_test,) per-clip motion_score
      <output_root>/<encoder>/intra_inter_ratio.json   {pos_mean, neg_mean, score_mean+ci}
    """
    if args.encoder is None:
        sys.exit("FATAL: --stage cosine requires --encoder")
    if args.action_probe_root is None:
        sys.exit("FATAL: --stage cosine requires --action-probe-root (for action_labels.json)")

    enc_dir = args.output_root / args.encoder
    pooled_path = enc_dir / "pooled_features_test.npy"
    keys_path   = enc_dir / "clip_keys_test.npy"
    if not (pooled_path.exists() and keys_path.exists()):
        sys.exit(f"FATAL: pooled features missing at {enc_dir} -- run --stage features first")

    out_per_clip = enc_dir / "per_clip_motion_cos.npy"
    out_ratio    = enc_dir / "intra_inter_ratio.json"
    if out_per_clip.exists() and out_ratio.exists() and args.cache_policy == "1":
        print(f"  [keep] cosine cache present at {enc_dir} -- skipping")
        return
    guarded_delete(out_per_clip, args.cache_policy, "per_clip_motion_cos")
    guarded_delete(out_ratio, args.cache_policy, "intra_inter_ratio")

    labels = load_action_labels(args.action_probe_root / "action_labels.json")
    emb = np.load(pooled_path).astype(np.float32)
    keys = np.load(keys_path, allow_pickle=True)
    y = np.array([labels[str(k)]["class_id"] for k in keys], dtype=np.int64)

    # Preflight B75: shape match
    if emb.shape[0] != len(y):
        sys.exit(f"FATAL: pooled features N={emb.shape[0]} != labels N={len(y)} (--encoder mismatch?)")

    print(f"Cosine over N={len(y)} clips, D={emb.shape[-1]}, classes={len(np.unique(y))}")
    t0 = time.time()
    score, pos, neg = _per_clip_motion_score(emb, y)
    elapsed = time.time() - t0
    print(f"  computed in {elapsed:.1f}s   pos_mean={pos.mean():.4f}  neg_mean={neg.mean():.4f}  score_mean={score.mean():.4f}")

    save_array_checkpoint(score, out_per_clip)
    score_ci = bootstrap_ci(score.astype(np.float64))
    save_json_checkpoint({
        "encoder": args.encoder,
        "n_test":   int(len(y)),
        "pos_mean": round(float(pos.mean()), 6),
        "neg_mean": round(float(neg.mean()), 6),
        "score_mean": round(float(score.mean()), 6),
        "score_ci": score_ci,
        "elapsed_sec": round(elapsed, 2),
    }, out_ratio)
    log_metrics(wb, {f"{args.encoder}_motion_score_mean": float(score.mean()),
                     f"{args.encoder}_pos_mean": float(pos.mean()),
                     f"{args.encoder}_neg_mean": float(neg.mean())})


# ── Stage 3 — paired BCa Δ ────────────────────────────────────────────

def run_paired_delta_stage(args, wb) -> None:
    """Paired BCa Δ_motion = motion_score(vjepa) − motion_score(dinov2).

    Output: <output_root>/m06d_motion_cos_paired.json
    """
    vj_dir = args.output_root / "vjepa_2_1_frozen"
    dn_dir = args.output_root / "dinov2"
    for d in (vj_dir, dn_dir):
        if not (d / "per_clip_motion_cos.npy").exists():
            sys.exit(f"FATAL: {d}/per_clip_motion_cos.npy missing -- run --stage cosine first")

    vj = np.load(vj_dir / "per_clip_motion_cos.npy").astype(np.float64)
    dn = np.load(dn_dir / "per_clip_motion_cos.npy").astype(np.float64)
    vj_keys = np.load(vj_dir / "clip_keys_test.npy", allow_pickle=True)
    dn_keys = np.load(dn_dir / "clip_keys_test.npy", allow_pickle=True)
    if not np.array_equal(vj_keys, dn_keys):
        sys.exit("FATAL: clip_keys disagree between encoders -- re-run --stage features for both")

    delta = vj - dn
    bca = paired_bca(delta)
    vj_ci = bootstrap_ci(vj)
    dn_ci = bootstrap_ci(dn)
    out = {
        "metric": "intra_minus_inter_cosine",
        "n_test": int(len(delta)),
        "vjepa_score_mean": round(float(vj.mean()), 6),
        "vjepa_score_ci":   vj_ci,
        "dinov2_score_mean": round(float(dn.mean()), 6),
        "dinov2_score_ci":   dn_ci,
        "delta_mean": round(float(delta.mean()), 6),
        "delta_ci_lo": round(float(bca["ci_lo"]), 6),
        "delta_ci_hi": round(float(bca["ci_hi"]), 6),
        "delta_ci_half": round(float(bca["ci_half"]), 6),
        "p_value": float(bca["p_value_vs_zero"]),
        "gate_pass": bool(bca["ci_lo"] > 0),
    }
    save_json_checkpoint(out, args.output_root / "m06d_motion_cos_paired.json")
    log_metrics(wb, out)
    print(json.dumps(out, indent=2))


# ── CLI ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-clip motion cosine similarity (m06d motion_cos -- priority 1 secondary)")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--stage", required=True,
                   choices=["features", "cosine", "paired_delta"])
    p.add_argument("--encoder", choices=list(ENCODERS), default=None)
    p.add_argument("--encoder-ckpt", type=Path, default=None,
                   help="V-JEPA ckpt (required for fresh-extract path on V-JEPA encoder)")
    add_local_data_arg(p)
    p.add_argument("--action-probe-root", type=Path, default=None,
                   help="m06d_action_probe output dir (provides action_labels.json + share-features cache)")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--num-frames", type=int, default=16,
                   help="Frames per clip (must match action_probe when --share-features)")
    p.add_argument("--share-features", action="store_true", default=True,
                   help="Default ON: mean-pool m06d_action_probe features. Falls back to fresh extract if cache absent.")
    p.add_argument("--no-share-features", dest="share_features", action="store_false",
                   help="Force fresh GPU re-extract (independent of m06d_action_probe cache)")
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
    wb = init_wandb(f"m06d_motion_cos_{args.stage}", mode,
                    config=vars(args), enabled=not args.no_wandb)
    try:
        if args.stage == "features":
            # check_gpu/cleanup_temp deferred to _fresh_extract_pooled (only fires if needed).
            run_features_stage(args, wb)
        elif args.stage == "cosine":
            run_cosine_stage(args, wb)
        elif args.stage == "paired_delta":
            run_paired_delta_stage(args, wb)
    finally:
        finish_wandb(wb)


if __name__ == "__main__":
    main()
