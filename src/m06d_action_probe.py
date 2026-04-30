"""Indian action attentive probe (Priority 1 gate). GPU-only.

Stages:
  labels        — derive 3/4-class action labels + 70/15/15 split (CPU)
  features      — extract frozen spatiotemporal token features per encoder (GPU)
  train         — train AttentiveClassifier head on cached features (GPU)
  paired_delta  — paired BCa Δ between V-JEPA and DINOv2 (CPU)

USAGE (sequence — every path arg required, no defaults):
    # Stage 1: labels (CPU, ~1 min)
    python -u src/m06d_action_probe.py --SANITY \\
        --stage labels --eval-subset data/eval_10k.json \\
        --tags-json data/eval_10k_local/tags.json \\
        --output-root outputs/sanity/m06d_action_probe \\
        --cache-policy 1 2>&1 | tee logs/m06d_action_probe_labels_sanity.log

    # Stage 2: features (GPU, ~30 min × 2 encoders)
    python -u src/m06d_action_probe.py --FULL \\
        --stage features --encoder vjepa_2_1_frozen \\
        --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \\
        --eval-subset data/eval_10k.json --local-data data/eval_10k_local \\
        --output-root outputs/full/m06d_action_probe \\
        --cache-policy 1 2>&1 | tee logs/m06d_action_probe_features_vjepa.log

    # Stage 3: train probe (GPU, ~15 min × 2 encoders)
    python -u src/m06d_action_probe.py --FULL \\
        --stage train --encoder vjepa_2_1_frozen \\
        --output-root outputs/full/m06d_action_probe --cache-policy 1

    # Stage 4: paired Δ (CPU, ~5 min, BCa 10K bootstrap)
    python -u src/m06d_action_probe.py --FULL \\
        --stage paired_delta \\
        --output-root outputs/full/m06d_action_probe --cache-policy 1
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from utils.action_labels import (
    CLASS_NAMES_3CLASS,
    CLASS_NAMES_4CLASS,
    load_action_labels,
    load_subset_with_labels,
    stratified_split,
    write_action_labels_json,
)
from utils.bootstrap import bootstrap_ci, paired_bca
from utils.cache_policy import (
    add_cache_policy_arg,
    guarded_delete,
    resolve_cache_policy_interactive,
)
from utils.checkpoint import (
    load_json_checkpoint,
    save_array_checkpoint,
    save_json_checkpoint,
)
from utils.config import (
    add_local_data_arg,
    check_gpu,
)
from utils.data_download import ensure_local_data
from utils.frozen_features import (
    ENCODERS,
    extract_features_for_keys,
    load_dinov2_frozen,
    load_vjepa_2_1_frozen,
)
from utils.gpu_batch import cleanup_temp
from utils.vjepa2_imports import get_attentive_classifier
from utils.wandb_utils import add_wandb_args, finish_wandb, init_wandb, log_metrics


# ── Constants ─────────────────────────────────────────────────────────

NUM_FRAMES_DEFAULT = 16


# ── Probe head training ──────────────────────────────────────────────

def _make_probe(d_in: int, n_classes: int, depth: int):
    """AttentiveClassifier (Meta's reference) with V-JEPA 2.1 protocol defaults."""
    AC = get_attentive_classifier()
    return AC(embed_dim=d_in, num_classes=n_classes,
              depth=depth, num_heads=16, mlp_ratio=4.0,
              complete_block=True, use_activation_checkpointing=False)


def _train_attentive_classifier(probe, X_tr, y_tr, X_val, y_val, args, jsonl_path: Path, wb):
    """AdamW + cosine LR + 10 % warmup + cross-entropy. Best-by-val-acc model selection.
    Returns (best_val_acc, best_state_dict).
    """
    device = "cuda"
    probe = probe.to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr, weight_decay=args.probe_wd)

    n_tr = len(y_tr)
    bs = max(8, min(args.train_batch_size, n_tr))
    steps_per_epoch = math.ceil(n_tr / bs)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * 0.10))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    rng = np.random.default_rng(args.seed)
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).long()
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()

    best_val_acc = -1.0
    best_state = None
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(jsonl_path, "w")

    step = 0
    for epoch in range(args.epochs):
        probe.train()
        idx = rng.permutation(n_tr)
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for s in range(0, n_tr, bs):
            sub = idx[s:s + bs]
            xb = X_tr_t[sub].to(device, non_blocking=True)
            yb = y_tr_t[sub].to(device, non_blocking=True)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            step += 1
            ep_loss += float(loss.item()) * yb.size(0)
            ep_correct += int((logits.argmax(-1) == yb).sum().item())
            ep_total += yb.size(0)

        val_acc = _eval_top1(probe, X_val_t, y_val_t, bs)
        train_acc = ep_correct / ep_total
        train_loss = ep_loss / ep_total
        cur_lr = optim.param_groups[0]["lr"]
        rec = {"epoch": epoch, "step": step, "train_loss": round(train_loss, 6),
               "train_acc": round(train_acc, 6), "val_acc": round(val_acc, 6),
               "lr": round(cur_lr, 8)}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        os.fsync(log_f.fileno())
        log_metrics(wb, rec)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:>3d}/{args.epochs}: loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={cur_lr:.2e}")
    log_f.close()
    return best_val_acc, best_state


def _eval_top1(probe, X_t: torch.Tensor, y_t: torch.Tensor, bs: int) -> float:
    """Top-1 accuracy (scalar). No-grad, eval mode."""
    probe.eval()
    correct = 0
    with torch.no_grad():
        for s in range(0, len(y_t), bs):
            xb = X_t[s:s + bs].to("cuda", non_blocking=True)
            yb = y_t[s:s + bs].to("cuda", non_blocking=True)
            correct += int((probe(xb).argmax(-1) == yb).sum().item())
    return correct / len(y_t)


def _eval_per_clip(probe, X_t: torch.Tensor, y_t: torch.Tensor, bs: int):
    """Per-clip {0,1} correctness vector + scalar acc + BCa CI dict."""
    probe.eval()
    correct = np.zeros(len(y_t), dtype=np.int8)
    with torch.no_grad():
        for s in range(0, len(y_t), bs):
            xb = X_t[s:s + bs].to("cuda", non_blocking=True)
            yb = y_t[s:s + bs].to("cuda", non_blocking=True)
            correct[s:s + bs] = (probe(xb).argmax(-1) == yb).cpu().numpy().astype(np.int8)
    acc = float(correct.mean())
    ci = bootstrap_ci(correct.astype(np.float64))
    return correct, acc, ci


# ── Stage dispatchers ─────────────────────────────────────────────────

def run_labels_stage(args, wb) -> None:
    if args.eval_subset is None:
        sys.exit("FATAL: --stage labels requires --eval-subset")
    args.output_root.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_root / "action_labels.json"
    if labels_path.exists() and args.cache_policy == "1":
        print(f"  [keep] {labels_path} present — skipping (--cache-policy 2 to redo)")
        return
    guarded_delete(labels_path, args.cache_policy, "action_labels.json")
    guarded_delete(args.output_root / "class_counts.json", args.cache_policy, "class_counts.json")

    records = load_subset_with_labels(args.eval_subset, args.tags_json,
                                      enable_monument=args.enable_monument_class)
    print(f"Loaded {len(records)} labeled clips from {args.eval_subset}")
    splits = stratified_split(records, seed=args.seed)
    write_action_labels_json(records, splits, labels_path)

    counts = load_json_checkpoint(args.output_root / "class_counts.json")
    for cls, c in counts.items():
        if c["test"] < 5 or c["val"] < 5:
            sys.exit(f"FATAL: class '{cls}' val={c['val']}/test={c['test']} (need >=5 each)")
        if c["train"] < 30:
            print(f"  WARN: class '{cls}' train={c['train']} (recommended >=30)")
    log_metrics(wb, {"n_clips_labeled": len(records), "n_classes": len(counts)})
    print(f"Wrote: {labels_path}  +  class_counts.json")


def run_features_stage(args, wb) -> None:
    if args.encoder is None:
        sys.exit("FATAL: --stage features requires --encoder")
    if args.eval_subset is None or args.local_data is None:
        sys.exit("FATAL: --stage features requires --eval-subset and --local-data")
    if ENCODERS[args.encoder]["kind"] == "vjepa" and args.encoder_ckpt is None:
        sys.exit("FATAL: V-JEPA encoder requires --encoder-ckpt")

    check_gpu()
    cleanup_temp()
    ensure_local_data(args)

    labels = load_action_labels(args.output_root / "action_labels.json")
    enc_dir = args.output_root / args.encoder
    enc_dir.mkdir(parents=True, exist_ok=True)

    enc_kind = ENCODERS[args.encoder]["kind"]
    if enc_kind == "vjepa":
        model, crop, embed_dim = load_vjepa_2_1_frozen(args.encoder_ckpt, args.num_frames)
    elif enc_kind == "dinov2":
        model, _processor, crop, embed_dim = load_dinov2_frozen()
    else:
        sys.exit(f"FATAL: unknown encoder kind '{enc_kind}'")

    by_split = {"train": [], "val": [], "test": []}
    for k, info in labels.items():
        by_split[info["split"]].append(k)

    for split, keys in by_split.items():
        out_features = enc_dir / f"features_{split}.npy"
        out_keys     = enc_dir / f"clip_keys_{split}.npy"
        if out_features.exists() and out_keys.exists() and args.cache_policy == "1":
            print(f"  [keep] {split}: features cached -> skipping")
            continue
        guarded_delete(out_features, args.cache_policy, f"features_{split}")
        guarded_delete(out_keys, args.cache_policy, f"clip_keys_{split}")

        print(f"\n=== Stage 2 features: {split} ({len(keys)} clips, encoder={args.encoder}) ===")
        t0 = time.time()
        feats, ordered_keys = extract_features_for_keys(
            args, model, enc_kind, crop, embed_dim,
            keys, enc_dir, label=f"features_{split}",
        )
        elapsed = time.time() - t0
        save_array_checkpoint(feats, out_features)
        np.save(out_keys, np.array(ordered_keys, dtype=object))
        print(f"  Saved {out_features} {feats.shape}  ({elapsed:.0f}s)")
        log_metrics(wb, {f"features_{split}_n": int(feats.shape[0]),
                         f"features_{split}_dim": int(feats.shape[-1]),
                         f"features_{split}_sec": round(elapsed, 1)})


def run_train_stage(args, wb) -> None:
    if args.encoder is None:
        sys.exit("FATAL: --stage train requires --encoder")
    check_gpu()
    cleanup_temp()

    enc_dir = args.output_root / args.encoder
    labels = load_action_labels(args.output_root / "action_labels.json")
    class_names = CLASS_NAMES_4CLASS if args.enable_monument_class else CLASS_NAMES_3CLASS

    feats_train = np.load(enc_dir / "features_train.npy")
    feats_val   = np.load(enc_dir / "features_val.npy")
    feats_test  = np.load(enc_dir / "features_test.npy")
    keys_train  = np.load(enc_dir / "clip_keys_train.npy", allow_pickle=True)
    keys_val    = np.load(enc_dir / "clip_keys_val.npy", allow_pickle=True)
    keys_test   = np.load(enc_dir / "clip_keys_test.npy", allow_pickle=True)
    y_train = np.array([labels[str(k)]["class_id"] for k in keys_train], dtype=np.int64)
    y_val   = np.array([labels[str(k)]["class_id"] for k in keys_val], dtype=np.int64)
    y_test  = np.array([labels[str(k)]["class_id"] for k in keys_test], dtype=np.int64)

    d_in = int(feats_train.shape[-1])
    n_classes = len(class_names)
    print(f"Probe: AttentiveClassifier(embed_dim={d_in}, n_classes={n_classes}, depth={args.probe_depth})")
    probe = _make_probe(d_in, n_classes, args.probe_depth)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"  params: {n_params/1e6:.2f}M")

    best_val_acc, best_state = _train_attentive_classifier(
        probe, feats_train, y_train, feats_val, y_val,
        args, jsonl_path=enc_dir / "train_log.jsonl", wb=wb)
    torch.save(best_state, enc_dir / "probe.pt")
    print(f"Saved best probe → {enc_dir / 'probe.pt'} (val_acc={best_val_acc:.4f})")

    probe.load_state_dict(best_state)
    X_test_t = torch.from_numpy(feats_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    test_correct, test_acc, test_ci = _eval_per_clip(probe, X_test_t, y_test_t, args.train_batch_size)

    np.save(enc_dir / "test_predictions.npy", test_correct)
    np.save(enc_dir / "test_clip_keys.npy", keys_test)
    save_json_checkpoint({
        "encoder": args.encoder, "n_classes": n_classes, "class_names": class_names,
        "n_test": int(len(test_correct)), "top1_acc": test_acc, "top1_ci": test_ci,
        "best_val_acc": float(best_val_acc),
    }, enc_dir / "test_metrics.json")
    print(f"Test top-1 acc: {test_acc:.4f}  ±{test_ci['ci_half']:.4f}  (95% BCa CI)")
    log_metrics(wb, {"test_top1_acc": test_acc, "test_top1_ci_half": test_ci["ci_half"]})


def run_paired_delta_stage(args, wb) -> None:
    vj_dir = args.output_root / "vjepa_2_1_frozen"
    dn_dir = args.output_root / "dinov2"
    for d in (vj_dir, dn_dir):
        if not (d / "test_predictions.npy").exists():
            sys.exit(f"FATAL: {d}/test_predictions.npy missing — run --stage train first")

    vj = np.load(vj_dir / "test_predictions.npy").astype(np.float32)
    dn = np.load(dn_dir / "test_predictions.npy").astype(np.float32)
    vj_keys = np.load(vj_dir / "test_clip_keys.npy", allow_pickle=True)
    dn_keys = np.load(dn_dir / "test_clip_keys.npy", allow_pickle=True)
    if not np.array_equal(vj_keys, dn_keys):
        sys.exit("FATAL: test_clip_keys disagree between encoders — re-run features stage")

    delta = vj - dn                                    # (N,) ∈ {-1, 0, +1}
    bca = paired_bca(delta)
    out = {
        "metric": "top1_accuracy",
        "n_clips_test": int(len(delta)),
        "vjepa_acc_pct":   round(float(vj.mean()) * 100, 4),
        "dinov2_acc_pct":  round(float(dn.mean()) * 100, 4),
        "delta_pp":        round(float(delta.mean()) * 100, 4),
        "ci_lo_pp":        round(float(bca["ci_lo"]) * 100, 4),
        "ci_hi_pp":        round(float(bca["ci_hi"]) * 100, 4),
        "ci_half_pp":      round(float(bca["ci_half"]) * 100, 4),
        "p_value":         float(bca["p_value_vs_zero"]),
        "gate_pass":       bool(bca["ci_lo"] > 0),
    }
    save_json_checkpoint(out, args.output_root / "m06d_paired_delta.json")
    log_metrics(wb, out)
    print(json.dumps(out, indent=2))


# ── CLI ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Indian action attentive probe (m06d action_probe — priority 1 gate)")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--stage", required=True,
                   choices=["labels", "features", "train", "paired_delta"])
    p.add_argument("--encoder", choices=list(ENCODERS), default=None)
    p.add_argument("--encoder-ckpt", type=Path, default=None)
    p.add_argument("--eval-subset", type=Path, default=None)
    p.add_argument("--tags-json", type=Path, default=None)
    add_local_data_arg(p)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--enable-monument-class", action="store_true")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES_DEFAULT)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--probe-lr", type=float, default=5e-4)
    p.add_argument("--probe-wd", type=float, default=0.05)
    p.add_argument("--probe-depth", type=int, default=4,
                   help="N attentive-pool layers (V-JEPA 2.1 published: 4)")
    p.add_argument("--train-batch-size", type=int, default=64)
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
    wb = init_wandb(f"m06d_action_probe_{args.stage}", mode,
                    config=vars(args), enabled=not args.no_wandb)
    try:
        if args.stage == "labels":
            run_labels_stage(args, wb)
        elif args.stage == "features":
            run_features_stage(args, wb)
        elif args.stage == "train":
            run_train_stage(args, wb)
        elif args.stage == "paired_delta":
            run_paired_delta_stage(args, wb)
    finally:
        finish_wandb(wb)


if __name__ == "__main__":
    main()
