"""Motion-flow attentive probe (Priority 1 gate). GPU-only.

iter13 v12 (2026-05-05): replaces saturated 3-class path-derived action labels
with optical-flow-derived motion classes (16 classes of magnitude×direction
from m04d motion_features.npy). See plan_code_dev.md Phase 2.

Stages:
  labels        — derive motion-flow class labels + 70/15/15 split (CPU)
  features      — extract frozen spatiotemporal token features per encoder (GPU)
  train         — train AttentiveClassifier head on cached features (GPU)
  paired_delta  — paired BCa Δ between encoders (CPU)

USAGE (sequence — every path arg required, no defaults):
    # Stage 1: labels (CPU, ~1 min). --motion-features is m04d's output,
    # default location <local_data>/motion_features.npy.
    python -u src/probe_action.py --SANITY \\
        --stage labels --eval-subset data/eval_10k.json \\
        --motion-features data/eval_10k_local/motion_features.npy \\
        --output-root outputs/sanity/probe_action \\
        --cache-policy 1 2>&1 | tee logs/probe_action_labels_sanity.log

    # Stage 2: features (GPU, ~30 min × 2 encoders)
    python -u src/probe_action.py --FULL \\
        --stage features --encoder vjepa_2_1_frozen \\
        --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \\
        --eval-subset data/eval_10k.json --local-data data/eval_10k_local \\
        --output-root outputs/full/probe_action \\
        --cache-policy 1 2>&1 | tee logs/probe_action_features_vjepa.log

    # Stage 3: train probe (GPU, ~15 min × 2 encoders)
    python -u src/probe_action.py --FULL \\
        --stage train --encoder vjepa_2_1_frozen \\
        --output-root outputs/full/probe_action --cache-policy 1

    # Stage 4: paired Δ (CPU, ~5 min, BCa 10K bootstrap)
    python -u src/probe_action.py --FULL \\
        --stage paired_delta \\
        --output-root outputs/full/probe_action --cache-policy 1
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
    # Meta's published recipe (deps/vjepa2/configs/eval/vitg-384/ssv2.yaml) uses
    # warmup=0.0 with the 5-LR multihead sweep. Our default is 0.10 (single-LR
    # variant) for stability at higher peak LR; FULL can override to 0.0 for
    # paper-faithful comparison.
    warmup_steps = max(0, int(total_steps * args.warmup_pct))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
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

    # Adaptive micro-batch (gradient accumulation): macro batch stays = bs so AdamW
    # + cosine schedule are unchanged; micro-batch shrinks on OOM. Fixes 24 GB OOM
    # at fc1(x) inside AttentiveClassifier blocks (depth=4 × 4608 tokens × 1664 dim).
    micro_bs = bs

    step = 0
    for epoch in range(args.epochs):
        probe.train()
        idx = rng.permutation(n_tr)
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for s in range(0, n_tr, bs):
            sub = idx[s:s + bs]
            xb = X_tr_t[sub].to(device, non_blocking=True)
            yb = y_tr_t[sub].to(device, non_blocking=True)
            macro_n = xb.shape[0]
            optim.zero_grad(set_to_none=True)
            macro_loss_sum, macro_correct = 0.0, 0
            i = 0
            while i < macro_n:
                end = min(i + micro_bs, macro_n)
                sub_xb, sub_yb = xb[i:end], yb[i:end]
                try:
                    sub_logits = probe(sub_xb)
                    # Scale by micro/macro so sum-of-grads = full-batch grads.
                    sub_loss = F.cross_entropy(sub_logits, sub_yb) * (sub_xb.shape[0] / macro_n)
                    sub_loss.backward()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    new_micro = max(1, micro_bs // 2)
                    if new_micro == micro_bs:
                        # Already at min; can't shrink further → re-raise.
                        raise
                    print(f"  OOM at micro_bs={micro_bs} → shrinking to {new_micro}, retrying chunk")
                    micro_bs = new_micro
                    optim.zero_grad(set_to_none=True)  # discard partial grads
                    macro_loss_sum, macro_correct = 0.0, 0
                    i = 0  # restart the macro batch from scratch
                    continue
                # Track unscaled loss + accuracy for logging.
                macro_loss_sum += float(sub_loss.item()) * macro_n  # unscale for sum
                macro_correct += int((sub_logits.argmax(-1) == sub_yb).sum().item())
                i = end
            optim.step()
            sched.step()
            step += 1
            ep_loss += macro_loss_sum
            ep_correct += macro_correct
            ep_total += macro_n

        val_acc = _eval_top1(probe, X_val_t, y_val_t, micro_bs)
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


def _eval_forward_oom_safe(probe, X_t: torch.Tensor, bs: int) -> tuple:
    """Forward CPU tensor `X_t` with adaptive micro-batch shrinking on OOM.

    CRITICAL: slices on CPU then moves chunk-by-chunk to GPU. Moving the WHOLE
    tensor first would OOM at FULL scale (e.g. 1492 val clips × 4608 × 1664 fp32
    = 45.7 GiB — exceeds even a 96 GB GPU). Returns (preds_np, used_bs).
    """
    n = X_t.shape[0]
    preds = np.empty(n, dtype=np.int64)
    micro = bs
    i = 0
    while i < n:
        end = min(i + micro, n)
        sub_xb = None
        try:
            with torch.no_grad():
                sub_xb = X_t[i:end].to("cuda", non_blocking=True)
                preds[i:end] = probe(sub_xb).argmax(-1).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            del sub_xb
            torch.cuda.empty_cache()
            new_micro = max(1, micro // 2)
            if new_micro == micro:
                raise
            print(f"  eval OOM at micro={micro} → {new_micro}, retrying chunk")
            micro = new_micro
            continue
        i = end
    return preds, micro


def _eval_top1(probe, X_t: torch.Tensor, y_t: torch.Tensor, bs: int) -> float:
    """Top-1 accuracy (scalar). No-grad, eval mode. OOM-safe (CPU-side slicing)."""
    probe.eval()
    preds, _ = _eval_forward_oom_safe(probe, X_t, bs)
    correct = (preds == y_t.numpy()).sum()
    return float(correct) / len(y_t)


def _eval_per_clip(probe, X_t: torch.Tensor, y_t: torch.Tensor, bs: int):
    """Per-clip {0,1} correctness vector + scalar acc + BCa CI dict. OOM-safe."""
    probe.eval()
    preds, _ = _eval_forward_oom_safe(probe, X_t, bs)
    correct = (preds == y_t.numpy()).astype(np.int8)
    acc = float(correct.mean())
    ci = bootstrap_ci(correct.astype(np.float64))
    return correct, acc, ci


# ── Stage dispatchers ─────────────────────────────────────────────────

def run_labels_stage(args, wb) -> None:
    if args.eval_subset is None:
        sys.exit("FATAL: --stage labels requires --eval-subset")
    if args.motion_features is None:
        sys.exit("FATAL: --stage labels requires --motion-features "
                 "(m04d's motion_features.npy — see plan_code_dev.md Phase 2)")
    args.output_root.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_root / "action_labels.json"
    if labels_path.exists() and args.cache_policy == "1":
        print(f"  [keep] {labels_path} present — skipping (--cache-policy 2 to redo)")
        return
    guarded_delete(labels_path, args.cache_policy, "action_labels.json")
    guarded_delete(args.output_root / "class_counts.json", args.cache_policy, "class_counts.json")

    # iter13 v12 (2026-05-05): MOTION-flow labels (RAFT optical-flow → 16 classes
    # of <magnitude>__<direction>, filter to ≥34 clips/class). Drops the legacy
    # path-derived 3-class action probe (saturated frozen V-JEPA at 0.94+).
    records, class_names = load_subset_with_labels(
        args.eval_subset, args.motion_features,
        min_clips_per_class=args.min_clips_per_class)
    print(f"Loaded {len(records)} labeled clips from {args.eval_subset} "
          f"({len(class_names)} motion-flow classes)")
    splits = stratified_split(records, seed=args.seed,
                              min_per_split=args.min_per_split)
    write_action_labels_json(records, splits, labels_path)

    counts = load_json_checkpoint(args.output_root / "class_counts.json")
    for cls, c in counts.items():
        if c["test"] < args.min_per_split or c["val"] < args.min_per_split:
            sys.exit(f"FATAL: class '{cls}' val={c['val']}/test={c['test']} "
                     f"(need >= {args.min_per_split} each)")
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

    # iter13 lazy-feature-extraction (2026-05-05): only save splits the user
    # explicitly listed in --features-splits (default ["test"]). train+val are
    # extracted in-process by Stage 3 (run_train_stage) — the .probe_features_
    # <split>_ckpt.npz resume side-effect doubles as a cross-LR cache. See
    # iter/iter13_motion_probe_eval/plan_code_dev.md and
    # /root/.claude/plans/wiggly-sniffing-muffin.md for design rationale.
    splits_to_save = list(args.features_splits)
    by_split = {s: [] for s in splits_to_save}
    for k, info in labels.items():
        if info["split"] in splits_to_save:
            by_split[info["split"]].append(k)
    if splits_to_save != ["train", "val", "test"]:
        skipped = [s for s in ("train", "val", "test") if s not in splits_to_save]
        print(f"  [features] saving splits {splits_to_save}; lazy splits "
              f"(extracted by Stage 3): {skipped}")

    for split, keys in by_split.items():
        out_features = enc_dir / f"features_{split}.npy"
        out_keys     = enc_dir / f"clip_keys_{split}.npy"
        if out_features.exists() and out_keys.exists() and args.cache_policy == "1":
            print(f"  [keep] {split}: features cached -> skipping")
            continue
        guarded_delete(out_features, args.cache_policy, f"features_{split}")
        guarded_delete(out_keys, args.cache_policy, f"clip_keys_{split}")

        print(f"\n=== Stage 2 features: {split} ({len(keys)} clips, encoder={args.encoder}, "
              f"pool_tokens={args.pool_tokens or 'OFF'}) ===")
        t0 = time.time()
        feats, ordered_keys = extract_features_for_keys(
            args, model, enc_kind, crop, embed_dim,
            keys, enc_dir, label=f"features_{split}",
            pool_tokens=(args.pool_tokens if args.pool_tokens > 0 else None),
        )
        elapsed = time.time() - t0
        save_array_checkpoint(feats, out_features)
        np.save(out_keys, np.array(ordered_keys, dtype=object))
        print(f"  Saved {out_features} {feats.shape}  ({elapsed:.0f}s)")
        log_metrics(wb, {f"features_{split}_n": int(feats.shape[0]),
                         f"features_{split}_dim": int(feats.shape[-1]),
                         f"features_{split}_sec": round(elapsed, 1)})


def _cleanup_lazy_feature_caches(enc_dir: Path) -> None:
    """Remove transient .probe_features_{train,val}_ckpt.npz after Stage 3 done.

    The lazy-extraction path (iter13 plan_code_dev.md) writes train+val
    feature cache .npz files via extract_features_for_keys' resume mechanism.
    Those caches are PER-RUN scratch — they're not consumed by Stage 4
    (paired_delta) or Stage 5 (motion_cos), only by Stage 3 train+val LR sweeps.
    Once Stage 3's best-LR is selected, the caches become dead disk weight
    (~30 GB per encoder per the v2 ENOSPC measurement).

    Called from end of run_train_stage (single-LR) AND
    run_select_best_lr_stage (multi-LR). test ckpt is preserved — Stage 4 +
    Stage 5 may consume it as a fallback if features_test.npy was wiped.
    """
    for split in ("train", "val"):
        for path in (enc_dir / f".probe_features_{split}_ckpt.npz",
                      enc_dir / f".probe_features_{split}_ckpt.tmp.npz"):
            if path.exists():
                try:
                    sz_gb = path.stat().st_size / 1e9
                    path.unlink()
                    print(f"  [lazy-cleanup] removed {path.name} ({sz_gb:.1f} GB)")
                except OSError as e:
                    print(f"  [lazy-cleanup] WARN: could not remove {path}: {e}")


def run_train_stage(args, wb) -> None:
    if args.encoder is None:
        sys.exit("FATAL: --stage train requires --encoder")
    if ENCODERS[args.encoder]["kind"] == "vjepa" and args.encoder_ckpt is None:
        sys.exit("FATAL: V-JEPA encoder requires --encoder-ckpt (lazy extract needs the model)")
    if args.local_data is None:
        sys.exit("FATAL: --stage train requires --local-data (lazy extract reads the TARs)")
    check_gpu()
    cleanup_temp()
    ensure_local_data(args)

    # Read features from canonical <encoder>/, write probe outputs to optional subdir.
    enc_dir = args.output_root / args.encoder
    out_dir = enc_dir / args.output_subdir if args.output_subdir else enc_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_action_labels(args.output_root / "action_labels.json")
    # iter13 v12 (2026-05-05): runtime-derived class_names from labels (motion-flow
    # classes are dataset-driven, K varies). Sorted alphabetically — matches
    # load_subset_with_labels' deterministic class_id assignment so class_id i ↔
    # class_names[i] is invariant.
    class_names = sorted({info["class"] for info in labels.values()})

    # iter13 lazy-extract (2026-05-05): for any split whose features_<split>.npy
    # is missing on disk, extract in-memory via extract_features_for_keys. The
    # function's resume side-effect (.probe_features_<split>_ckpt.npz) acts as
    # a cross-LR cache for multi-LR sweeps. See plan_code_dev.md.
    enc_kind = ENCODERS[args.encoder]["kind"]
    by_split = {"train": [], "val": [], "test": []}
    for k, info in labels.items():
        by_split[info["split"]].append(k)

    # iter13 (2026-05-05): stream-and-discard branch. Skips ALL feature persistence
    # AND the lazy-extract RAM cache; per-batch fwd-pool-head-loss-backward,
    # discard. Use when pooled features still don't fit RAM (>1M clips at
    # pool_tokens=128) OR for ablating multiple pool_tokens settings without
    # re-extracting. Single LR (no sweep) — for sweeps stick with the
    # disk-cached lazy path.
    if args.stream_train:
        from utils.probe_stream import stream_train_attentive_probe
        if enc_kind == "vjepa":
            model, crop, embed_dim = load_vjepa_2_1_frozen(args.encoder_ckpt, args.num_frames)
        elif enc_kind == "dinov2":
            model, _proc, crop, embed_dim = load_dinov2_frozen()
        else:
            sys.exit(f"FATAL: unknown encoder kind '{enc_kind}'")

        labels_by_clip = {k: info["class_id"] for k, info in labels.items()}
        n_classes = len(class_names)
        print(f"[stream-train] encoder={args.encoder}, embed_dim={embed_dim}, "
              f"pool_tokens={args.pool_tokens}, n_classes={n_classes}")
        best_val_acc, best_state, test_correct, test_keys = stream_train_attentive_probe(
            args, model, enc_kind, crop, embed_dim,
            by_split, labels_by_clip, n_classes, out_dir, wb)
        # Persist probe + test predictions in canonical layout (matches lazy path).
        torch.save(best_state, out_dir / "probe.pt")
        np.save(out_dir / "test_predictions.npy", test_correct)
        np.save(out_dir / "test_clip_keys.npy", np.array(test_keys, dtype=object))
        test_acc = float(test_correct.mean()) if test_correct.size else 0.0
        test_ci = bootstrap_ci(test_correct.astype(np.float64))
        save_json_checkpoint({
            "encoder": args.encoder, "n_classes": n_classes, "class_names": class_names,
            "n_test": int(test_correct.size), "top1_acc": test_acc, "top1_ci": test_ci,
            "best_val_acc": float(best_val_acc),
            "probe_lr": float(args.probe_lr), "warmup_pct": float(args.warmup_pct),
            "epochs": int(args.epochs),
            "stream_train": True,
        }, out_dir / "test_metrics.json")
        print(f"[stream-train] test top-1: {test_acc:.4f} ±{test_ci['ci_half']:.4f} (95% BCa CI)")
        log_metrics(wb, {"test_top1_acc": test_acc,
                         "test_top1_ci_half": test_ci["ci_half"],
                         "stream_train": 1})
        # Free encoder; no lazy caches to clean up (none were written).
        del model
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()
        return

    # Detect missing on-disk features → load encoder lazily (only when needed).
    needs_extract = [s for s in ("train", "val", "test")
                     if not (enc_dir / f"features_{s}.npy").exists()]
    model = crop = embed_dim = None
    if needs_extract:
        print(f"  [lazy-extract] missing splits on disk: {needs_extract} — "
              f"loading {args.encoder} encoder for in-memory extraction")
        if enc_kind == "vjepa":
            model, crop, embed_dim = load_vjepa_2_1_frozen(args.encoder_ckpt, args.num_frames)
        elif enc_kind == "dinov2":
            model, _proc, crop, embed_dim = load_dinov2_frozen()
        else:
            sys.exit(f"FATAL: unknown encoder kind '{enc_kind}'")

    def _load_or_extract(split: str):
        """Return (features (N,T,D) np.float32, ordered_keys np.array) for `split`.
        Disk-first; falls back to extract_features_for_keys with resume cache.
        """
        npy_path  = enc_dir / f"features_{split}.npy"
        keys_path = enc_dir / f"clip_keys_{split}.npy"
        if npy_path.exists() and keys_path.exists():
            return np.load(npy_path), np.load(keys_path, allow_pickle=True)
        feats, ordered_keys = extract_features_for_keys(
            args, model, enc_kind, crop, embed_dim,
            by_split[split], enc_dir, label=f"features_{split}",
            pool_tokens=(args.pool_tokens if args.pool_tokens > 0 else None),
        )
        return feats, np.array(ordered_keys, dtype=object)

    feats_train, keys_train = _load_or_extract("train")
    feats_val,   keys_val   = _load_or_extract("val")
    feats_test,  keys_test  = _load_or_extract("test")
    # iter13 (2026-05-05): free encoder GPU memory BEFORE probe-head training.
    # The encoder (~7.5 GB ViT-G weights + 1.5 GB activation buffers) is no
    # longer needed once features are extracted — keeping it resident competes
    # with the probe-head attention matrix and triggers spurious OOM-shrink at
    # micro_bs=64 (see iter13 v3 log L297). gc.collect() is required because
    # `del model` only drops the local reference; the actual frees happen when
    # Python finalises the underlying nn.Modules — which means the dangling
    # references inside torch.cuda allocator block reuse until gc runs.
    if model is not None:
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    y_train = np.array([labels[str(k)]["class_id"] for k in keys_train], dtype=np.int64)
    y_val   = np.array([labels[str(k)]["class_id"] for k in keys_val], dtype=np.int64)
    y_test  = np.array([labels[str(k)]["class_id"] for k in keys_test], dtype=np.int64)

    d_in = int(feats_train.shape[-1])
    n_classes = len(class_names)
    print(f"Probe: AttentiveClassifier(embed_dim={d_in}, n_classes={n_classes}, depth={args.probe_depth})")
    probe = _make_probe(d_in, n_classes, args.probe_depth)
    n_params = sum(p.numel() for p in probe.parameters())
    print(f"  params: {n_params/1e6:.2f}M  → output_dir={out_dir}")

    best_val_acc, best_state = _train_attentive_classifier(
        probe, feats_train, y_train, feats_val, y_val,
        args, jsonl_path=out_dir / "train_log.jsonl", wb=wb)
    torch.save(best_state, out_dir / "probe.pt")
    print(f"Saved best probe → {out_dir / 'probe.pt'} (val_acc={best_val_acc:.4f})")

    probe.load_state_dict(best_state)
    X_test_t = torch.from_numpy(feats_test).float()
    y_test_t = torch.from_numpy(y_test).long()
    test_correct, test_acc, test_ci = _eval_per_clip(probe, X_test_t, y_test_t, args.train_batch_size)

    np.save(out_dir / "test_predictions.npy", test_correct)
    np.save(out_dir / "test_clip_keys.npy", keys_test)
    save_json_checkpoint({
        "encoder": args.encoder, "n_classes": n_classes, "class_names": class_names,
        "n_test": int(len(test_correct)), "top1_acc": test_acc, "top1_ci": test_ci,
        "best_val_acc": float(best_val_acc),
        "probe_lr": float(args.probe_lr), "warmup_pct": float(args.warmup_pct),
        "epochs": int(args.epochs),
    }, out_dir / "test_metrics.json")
    print(f"Test top-1 acc: {test_acc:.4f}  ±{test_ci['ci_half']:.4f}  (95% BCa CI)")
    log_metrics(wb, {"test_top1_acc": test_acc, "test_top1_ci_half": test_ci["ci_half"]})

    # iter13 lazy-extract cleanup: for single-LR runs (no output_subdir), the
    # train+val .probe_features_<split>_ckpt.npz caches are dead weight after
    # this point. For multi-LR sweeps (output_subdir="lr_xxx"), DEFER cleanup
    # to run_select_best_lr_stage — sibling LR runs still need the cache.
    if not args.output_subdir:
        _cleanup_lazy_feature_caches(enc_dir)


def run_select_best_lr_stage(args, wb) -> None:
    """Multi-LR sweep best-by-val-acc selection (post-hoc, idempotent).

    Scans `<output_root>/<encoder>/lr_*/test_metrics.json`, picks the subdir with
    highest `best_val_acc`, and symlinks its outputs to the canonical
    `<output_root>/<encoder>/<file>` paths so paired_delta reads the winner with
    no further code changes. Mirrors the spirit of Meta's multihead probe sweep
    (`deps/vjepa2/configs/eval/vitg-384/ssv2.yaml` selects best-of-N at eval time).
    Idempotent: re-running just re-symlinks. No-op if only one lr_* subdir exists.
    """
    if args.encoder is None:
        sys.exit("FATAL: --stage select_best_lr requires --encoder")
    enc_dir = args.output_root / args.encoder
    if not enc_dir.exists():
        sys.exit(f"FATAL: encoder dir not found: {enc_dir}")
    lr_dirs = sorted(enc_dir.glob("lr_*"))
    if not lr_dirs:
        print(f"  no lr_* subdirs under {enc_dir} — nothing to select (single-LR run, canonical paths already populated)")
        return
    best_dir, best_acc = None, -1.0
    for sub in lr_dirs:
        mf = sub / "test_metrics.json"
        if not mf.exists():
            continue
        val = json.loads(mf.read_text()).get("best_val_acc", -1.0)
        if val > best_acc:
            best_acc, best_dir = val, sub
    if best_dir is None:
        sys.exit(f"FATAL: no lr_*/test_metrics.json found under {enc_dir}")
    print(f"Best LR for {args.encoder}: {best_dir.name} (best_val_acc={best_acc:.4f}) "
          f"out of {len(lr_dirs)} swept LRs")
    n_linked = 0
    for fname in ("probe.pt", "test_predictions.npy", "test_clip_keys.npy",
                  "test_metrics.json", "train_log.jsonl"):
        if not (best_dir / fname).exists():
            continue
        target = enc_dir / fname
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(best_dir.name + "/" + fname)
        print(f"  symlink: {target.name} -> {best_dir.name}/{fname}")
        n_linked += 1
    log_metrics(wb, {f"{args.encoder}_best_lr": best_dir.name,
                     f"{args.encoder}_best_val_acc": best_acc,
                     f"{args.encoder}_n_lrs_swept": len(lr_dirs),
                     f"{args.encoder}_n_files_linked": n_linked})

    # iter13 lazy-extract cleanup: all sibling LR runs are done — caches are dead.
    _cleanup_lazy_feature_caches(enc_dir)


def run_paired_delta_stage(args, wb) -> None:
    """N-way paired-Δ across all encoders that have probe outputs on disk.

    Auto-discovers encoders by scanning <output_root>/*/ for the 3 expected files
    (test_predictions.npy + test_clip_keys.npy + test_metrics.json). For each
    pair (a, b) computes Δ = a_acc − b_acc on the shared test-clip intersection
    with BCa 95 % CI. Schema:
        {
          "metric": "top1_accuracy",
          "by_encoder":      {<enc>: {"acc_pct", "n", "top1_ci"}, ...},
          "pairwise_deltas": {"<a>_minus_<b>": {n_shared, delta_pp, ci_*_pp,
                                                 p_value, gate_pass, interpretation},
                              ...}
        }
    Plus legacy top-level keys (vjepa_acc_pct/dinov2_acc_pct/delta_pp/etc.) when
    BOTH vjepa_2_1_frozen and dinov2 are present — kept so probe_plot.py
    (which reads those directly at lines 213-302) doesn't break on the N-way
    upgrade. Same algorithmic template as probe_future_mse.py:447-529.
    """
    enc_data = {}
    for enc_dir in sorted(args.output_root.iterdir()):
        if not enc_dir.is_dir():
            continue
        if all((enc_dir / f).exists() for f in
               ("test_predictions.npy", "test_clip_keys.npy", "test_metrics.json")):
            enc_data[enc_dir.name] = {
                "preds": np.load(enc_dir / "test_predictions.npy").astype(np.float32),
                "keys":  [str(k) for k in np.load(enc_dir / "test_clip_keys.npy", allow_pickle=True)],
                "agg":   json.loads((enc_dir / "test_metrics.json").read_text()),
            }
    available = sorted(enc_data.keys())
    if len(available) < 2:
        sys.exit(f"FATAL: need >=2 encoders with probe outputs, found: {available} "
                 f"(run --stage train per-encoder first)")
    print(f"Encoders found: {available}")

    by_encoder = {e: {"acc_pct": round(float(d["preds"].mean()) * 100, 4),
                      "n":       len(d["keys"]),
                      "top1_ci": d["agg"]["top1_ci"]}
                  for e, d in enc_data.items()}

    # Pairwise alignment: per-encoder key order is non-deterministic (Stage 2
    # iter_clips_parallel uses N concurrent TAR readers), so we intersect keys
    # for each pair separately. Pattern ported from probe_future_mse.py:447-529.
    pairwise_deltas = {}
    for i, a in enumerate(available):
        for b in available[i + 1:]:
            ka, kb = enc_data[a]["keys"], enc_data[b]["keys"]
            shared = sorted(set(ka) & set(kb))
            if not shared:
                print(f"  WARN: {a} vs {b} have ZERO shared test clips — skipping")
                continue
            ai = {k: idx for idx, k in enumerate(ka)}
            bi = {k: idx for idx, k in enumerate(kb)}
            a_arr = np.array([enc_data[a]["preds"][ai[k]] for k in shared], dtype=np.float32)
            b_arr = np.array([enc_data[b]["preds"][bi[k]] for k in shared], dtype=np.float32)
            delta = a_arr - b_arr
            bca = paired_bca(delta)
            pairwise_deltas[f"{a}_minus_{b}"] = {
                "n_shared":       int(len(shared)),
                "delta_pp":       round(float(delta.mean()) * 100, 4),
                "ci_lo_pp":       round(float(bca["ci_lo"]) * 100, 4),
                "ci_hi_pp":       round(float(bca["ci_hi"]) * 100, 4),
                "ci_half_pp":     round(float(bca["ci_half"]) * 100, 4),
                "p_value":        float(bca["p_value_vs_zero"]),
                "gate_pass":      bool(bca["ci_lo"] > 0),
                "interpretation": f"{a} - {b} > 0 means {a} more accurate than {b}",
            }

    out = {"metric": "top1_accuracy",
           "by_encoder": by_encoder,
           "pairwise_deltas": pairwise_deltas}
    save_json_checkpoint(out, args.output_root / "probe_paired_delta.json")
    log_metrics(wb, {"n_encoders_compared": len(available),
                     "n_pairwise_deltas":   len(pairwise_deltas)})
    print(json.dumps(out, indent=2))


# ── CLI ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Indian action attentive probe (probe action_probe — priority 1 gate)")
    p.add_argument("--SANITY", action="store_true")
    p.add_argument("--POC",    action="store_true")
    p.add_argument("--FULL",   action="store_true")
    p.add_argument("--stage", required=True,
                   choices=["labels", "features", "train", "select_best_lr", "paired_delta"])
    p.add_argument("--encoder", choices=list(ENCODERS), default=None)
    p.add_argument("--encoder-ckpt", type=Path, default=None)
    p.add_argument("--output-subdir", type=str, default="",
                   help="Optional sub-directory under <output_root>/<encoder>/ for nesting per-LR-sweep "
                        "probe outputs. Empty (default) writes probe.pt + test_predictions.npy at the "
                        "canonical location. Set to e.g. 'lr_5e-4' to enable multi-LR sweeps without collision.")
    p.add_argument("--eval-subset", type=Path, default=None)
    add_local_data_arg(p)
    p.add_argument("--output-root", type=Path, required=True)
    # iter13 v12 (2026-05-05): MOTION-flow class derivation (Phase 2). Replaces
    # legacy 3/4-class path-derived action labels (saturated). --motion-features
    # is required for --stage labels; --min-clips-per-class + --min-per-split
    # gate sparse-class filtering. See plan_code_dev.md.
    p.add_argument("--motion-features", type=Path, default=None,
                   help="m04d motion_features.npy path (RAFT optical-flow 13D × N_clips). "
                        "Required for --stage labels. Companion .paths.npy must exist next to it. "
                        "Default location: <local_data>/motion_features.npy.")
    p.add_argument("--min-clips-per-class", type=int, default=34,
                   help="Drop motion-flow classes with fewer than this many clips. "
                        "Default 34 = ≥5 clips per split at 70/15/15 stratification.")
    p.add_argument("--min-per-split", type=int, default=5,
                   help="Minimum clips per split (train/val/test) per class for BCa CI floor. "
                        "Default 5. SANITY runs may need a lower value if class data is sparse.")
    p.add_argument("--num-frames", type=int, default=NUM_FRAMES_DEFAULT)
    # iter13 (2026-05-05): lazy-feature-extraction. Stage 2 saves only the listed
    # splits as features_<split>.npy. train+val are NOT durable by default —
    # Stage 3 lazily re-extracts them in-memory (with disk-resume across LR
    # sweeps via .probe_features_<split>_ckpt.npz). Saves ~80 GB per FULL eval.
    # Override with --features-splits train val test for the legacy all-3 layout.
    p.add_argument("--features-splits", nargs="+", default=["test"],
                   choices=["train", "val", "test"],
                   help="Which splits Stage 2 (--stage features) saves to disk as "
                        "features_<split>.npy. Default: 'test' only — train/val are "
                        "extracted lazily in Stage 3 to save ~80 GB durable disk per FULL eval.")
    p.add_argument("--pool-tokens", type=int, default=16,
                   help="Adaptive-avg-pool encoder output to N tokens BEFORE storage / probe. "
                        "Default 16 (V-JEPA paper §4 attentive-probe regime; 290× smaller .npy "
                        "and 290× less probe-head attention compute). Use 0 to disable pooling "
                        "(legacy 4608-token storage; needs ~21 MB/clip and 43 GB attention matrix "
                        "at BS=64 — see iter13 OOM diagnosis).")
    p.add_argument("--stream-train", action="store_true",
                   help="Stream-and-discard Stage 3 training: no .npy persistence; per-batch "
                        "TAR-decode → encoder forward (no_grad) → pool → head → loss → backward. "
                        "Use for 100k+ clip datasets where even pooled features don't fit RAM. "
                        "Cost: encoder runs once per epoch (~1× the lazy-extract cost). Disk: 0 GB.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--probe-lr", type=float, default=5e-4)
    p.add_argument("--probe-wd", type=float, default=0.05)
    p.add_argument("--warmup-pct", type=float, default=0.10,
                   help="Fraction of total_steps for linear LR warmup. "
                        "0.10 = our single-LR default. 0.0 = Meta's published recipe (deps/vjepa2/configs/eval/vitg-384/ssv2.yaml).")
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
    wb = init_wandb(f"probe_action_{args.stage}", mode,
                    config=vars(args), enabled=not args.no_wandb)
    try:
        if args.stage == "labels":
            run_labels_stage(args, wb)
        elif args.stage == "features":
            run_features_stage(args, wb)
        elif args.stage == "train":
            run_train_stage(args, wb)
        elif args.stage == "select_best_lr":
            run_select_best_lr_stage(args, wb)
        elif args.stage == "paired_delta":
            run_paired_delta_stage(args, wb)
    finally:
        finish_wandb(wb)


if __name__ == "__main__":
    # Fail-fast: any uncaught exception → traceback + sys.exit(1) so the
    # parent shell (run_probe_eval.sh under `set -e`) sees non-zero and aborts the
    # chain. Mirrors m10_sam_segment.py pattern (errors_N_fixes #14/#16).
    try:
        main()
    except SystemExit:
        raise  # respect explicit sys.exit(N) calls inside main()
    except BaseException:
        import traceback
        print(f"\n❌ FATAL: {Path(__file__).name} crashed — see traceback below", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
