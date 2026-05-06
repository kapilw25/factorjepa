"""Stream-and-discard attentive-probe training (zero feature persistence). GPU-only.

Per-batch pipeline: TAR-decode → encoder forward (no_grad, frozen) → adaptive
token pool → AttentiveClassifier forward → cross-entropy → backward → discard.
Features never touch disk and never accumulate in RAM. Re-encodes train+val each
epoch (encoder dominates per-step time vs. probe head), so this is the right
choice when pooled features still don't fit RAM (≥1M clips at pool_tokens=128)
OR when comparing multiple pool_tokens settings without re-extracting. For
typical 10k–200k-clip frozen-encoder evals at pool_tokens=16, the disk-cached
lazy-extract path (probe_action.run_train_stage default) is faster — pooled
features fit in <10 GB RAM and the probe trains over them in seconds/epoch.

PUBLIC API
    stream_train_attentive_probe(args, model, encoder_kind, crop, embed_dim,
                                  by_split, labels_by_clip, n_classes, output_dir, wb)
        -> (best_val_acc, best_state_dict, test_correct (N_test,), test_keys list[str])
"""
import json
import math
import os
import queue
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.data_download import iter_clips_parallel
from utils.frozen_features import (
    _pool_tokens,
    decode_to_tensor,
    forward_dinov2,
    forward_vjepa,
)
from utils.gpu_batch import AdaptiveBatchSizer, cuda_cleanup
from utils.progress import make_pbar
from utils.vjepa2_imports import get_attentive_classifier
from utils.wandb_utils import log_metrics


def _make_probe(d_in: int, n_classes: int, depth: int):
    """AttentiveClassifier with V-JEPA 2.1 protocol defaults. Mirrors probe_action._make_probe."""
    AC = get_attentive_classifier()
    return AC(embed_dim=d_in, num_classes=n_classes,
              depth=depth, num_heads=16, mlp_ratio=4.0,
              complete_block=True, use_activation_checkpointing=False)


def _encode_pool(model, encoder_kind: str, batch: torch.Tensor,
                 num_frames: int, pool_tokens: int) -> torch.Tensor:
    """(B, T, 3, H, W) → (B, n_pool, D) on CUDA. Frozen encoder; no_grad inherited
    from forward_vjepa / forward_dinov2 (both decorated with @torch.no_grad).
    """
    if encoder_kind == "vjepa":
        feats = forward_vjepa(model, batch)
    else:
        feats = forward_dinov2(model, batch, num_frames)
    if pool_tokens is not None and pool_tokens > 0:
        feats = _pool_tokens(feats, pool_tokens)
    return feats.float().to("cuda", non_blocking=True)


def _stream_encoded_batches(args, model, encoder_kind: str, crop: int,
                            keys: list, labels_by_clip: dict, bs: int,
                            tmp_dir_root: Path, label: str):
    """Generator: yields (xb_cuda (B, n_pool, D), yb_cuda (B,) long, key_list).

    Each yield = one macro-batch. Encoder is frozen; the producer-side
    AdaptiveBatchSizer protects encoder fwd from OOM by sub-batching the macro
    batch through the encoder, then concatenating pooled outputs. The probe-
    head fwd+backward operates on the full macro batch downstream (caller).
    """
    target_keys = set(keys)
    clip_q, tar_stop, _reader = iter_clips_parallel(
        local_data=args.local_data, subset_keys=target_keys, processed_keys=set())

    # AdaptiveBatchSizer for encoder fwd. Initial value matches extract_features_for_keys.
    from utils.config import get_pipeline_config
    pcfg = get_pipeline_config()
    if encoder_kind == "vjepa":
        initial_bs = pcfg["gpu"].get("inference_vjepa_probe_initial_bs",
                                     pcfg["gpu"]["inference_vjepa_initial_bs"])
        max_bs = pcfg["gpu"]["inference_adapted_probe_bs"]
    else:
        initial_bs = pcfg["gpu"]["inference_dinov2_initial_bs"]
        max_bs = pcfg["gpu"]["inference_dinov2_bs"]
    sizer = AdaptiveBatchSizer(initial_size=initial_bs, min_size=1, max_size=max_bs,
                               memory_cap=pcfg["gpu"]["gpu_memory_target"])

    tmp_dir_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir_root)

    pending_tensors, pending_keys, pending_labels = [], [], []
    pbar = make_pbar(total=len(target_keys), desc=f"stream_{label}", unit="clip")
    try:
        while True:
            try:
                item = clip_q.get(timeout=300)
            except queue.Empty:
                print(f"  [stream-{label}] clip queue timeout — flushing partial batch and exiting")
                break
            if item is None:
                break
            clip_key, mp4_bytes = item
            if clip_key not in labels_by_clip:
                continue
            t = decode_to_tensor(mp4_bytes, tmp_dir, clip_key, args.num_frames, crop)
            if t is None:
                continue
            pending_tensors.append(t)
            pending_keys.append(clip_key)
            pending_labels.append(labels_by_clip[clip_key])
            if len(pending_tensors) >= bs:
                xb = _encode_macro(pending_tensors, model, encoder_kind, args.num_frames,
                                   args.pool_tokens, sizer)
                yb = torch.tensor(pending_labels, dtype=torch.long, device="cuda")
                pbar.update(len(pending_tensors))
                yield xb, yb, list(pending_keys)
                pending_tensors, pending_keys, pending_labels = [], [], []
        # Final partial batch
        if pending_tensors:
            xb = _encode_macro(pending_tensors, model, encoder_kind, args.num_frames,
                               args.pool_tokens, sizer)
            yb = torch.tensor(pending_labels, dtype=torch.long, device="cuda")
            pbar.update(len(pending_tensors))
            yield xb, yb, list(pending_keys)
    finally:
        tar_stop.set()
        pbar.close()


def _encode_macro(pending_tensors, model, encoder_kind: str, num_frames: int,
                  pool_tokens: int, sizer) -> torch.Tensor:
    """Encode a macro batch, sub-batching through encoder under AdaptiveBatchSizer.
    Returns (n_macro, n_pool, D) on CUDA. Encoder fwd is no_grad (frozen).
    """
    batch = torch.stack(pending_tensors)
    n_total = batch.shape[0]
    chunks = []
    i = 0
    while i < n_total:
        sub = batch[i:i + sizer.size]
        try:
            feats = _encode_pool(model, encoder_kind, sub, num_frames, pool_tokens)
        except torch.cuda.OutOfMemoryError:
            cuda_cleanup()
            if not sizer.on_oom():
                raise
            continue
        chunks.append(feats)
        sizer.after_batch_success()
        i += sub.shape[0]
    return torch.cat(chunks, dim=0)


def _eval_accuracy(probe, args, model, encoder_kind: str, crop: int,
                   keys: list, labels_by_clip: dict, bs: int,
                   tmp_dir_root: Path, label: str):
    """Returns (per_clip_correct (N,) int8, ordered_keys list[str])."""
    probe.eval()
    correct_chunks, key_chunks = [], []
    with torch.no_grad():
        for xb, yb, batch_keys in _stream_encoded_batches(
                args, model, encoder_kind, crop, keys, labels_by_clip, bs,
                tmp_dir_root, label):
            preds = probe(xb).argmax(-1)
            correct = (preds == yb).to(torch.int8).cpu().numpy()
            correct_chunks.append(correct)
            key_chunks.extend(batch_keys)
    if not correct_chunks:
        return np.empty(0, dtype=np.int8), []
    return np.concatenate(correct_chunks, axis=0), key_chunks


def stream_train_attentive_probe(args, model, encoder_kind: str, crop: int, embed_dim: int,
                                 by_split: dict, labels_by_clip: dict, n_classes: int,
                                 output_dir: Path, wb):
    """Single-LR stream-and-discard attentive probe trainer.

    Returns:
        best_val_acc: float
        best_state:   dict (probe state_dict, on cpu)
        test_correct: np.ndarray (N_test,) int8
        test_keys:    list[str] aligned with test_correct
    """
    device = "cuda"
    probe = _make_probe(embed_dim, n_classes, args.probe_depth).to(device)
    optim = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr,
                              weight_decay=args.probe_wd)

    n_train = len(by_split["train"])
    bs = max(8, min(args.train_batch_size, n_train))
    steps_per_epoch = math.ceil(n_train / bs)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(0, int(total_steps * args.warmup_pct))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "train_log.jsonl"
    log_f = open(jsonl_path, "w")
    tmp_dir_root = output_dir / "tmp_decode_stream"

    best_val_acc, best_state = -1.0, None
    step = 0
    t0 = time.time()
    print(f"[stream-train] {args.epochs} epochs × ~{steps_per_epoch} steps "
          f"(bs={bs}, n_train={n_train}, warmup={warmup_steps})")

    for epoch in range(args.epochs):
        probe.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for xb, yb, _ in _stream_encoded_batches(
                args, model, encoder_kind, crop,
                by_split["train"], labels_by_clip, bs,
                tmp_dir_root, f"ep{epoch}_train"):
            optim.zero_grad(set_to_none=True)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optim.step()
            sched.step()
            step += 1
            ep_loss += float(loss.item()) * yb.shape[0]
            ep_correct += int((logits.argmax(-1) == yb).sum().item())
            ep_total += yb.shape[0]

        if ep_total == 0:
            sys.exit(f"FATAL: stream-train epoch {epoch} processed 0 train clips — "
                     f"check --local-data + iter_clips_parallel TAR queue")

        val_correct, _ = _eval_accuracy(
            probe, args, model, encoder_kind, crop,
            by_split["val"], labels_by_clip, bs,
            tmp_dir_root, f"ep{epoch}_val")
        val_acc = float(val_correct.mean()) if val_correct.size else 0.0
        train_acc = ep_correct / ep_total
        train_loss = ep_loss / ep_total
        cur_lr = optim.param_groups[0]["lr"]
        rec = {"epoch": epoch, "step": step,
               "train_loss": round(train_loss, 6),
               "train_acc":  round(train_acc, 6),
               "val_acc":    round(val_acc, 6),
               "lr":         round(cur_lr, 8),
               "mode":       "stream"}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        os.fsync(log_f.fileno())
        log_metrics(wb, rec)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"  ep {epoch:>3d}/{args.epochs}: loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={cur_lr:.2e}")
    log_f.close()

    # Final test pass with best probe
    if best_state is not None:
        probe.load_state_dict(best_state)
    test_correct, test_keys = _eval_accuracy(
        probe, args, model, encoder_kind, crop,
        by_split["test"], labels_by_clip, bs,
        tmp_dir_root, "test_final")
    elapsed = time.time() - t0
    print(f"[stream-train] done in {elapsed:.0f}s; best_val_acc={best_val_acc:.4f}, "
          f"n_test={len(test_keys)}")
    return best_val_acc, best_state, test_correct, test_keys


__all__ = ["stream_train_attentive_probe"]
