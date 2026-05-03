---
name: Hardware split — what runs on 24 GB vs 96 GB
description: Empirically validated 2026-05-03; ironclad — don't try to train on 24 GB
type: project
---

# Hardware split — 24 GB SANITY vs 96 GB FULL

> Validated 2026-05-03 via `logs/probe_pretrain_sanity_v6.log`. Don't burn time re-validating — the OOM is structural (FIXED memory > 24 GB before any activations).

## Pipeline → hardware matrix

| Pipeline | 24 GB SANITY | 96 GB FULL | Notes |
|---|---|---|---|
| `run_probe_eval.sh --sanity` (10 stages × 150 clips) | ✅ ~6-8 min | ✅ trivially | Only needs ONE encoder forward; no teacher copy; no optimizer state |
| `run_probe_eval.sh --FULL` (10 stages × ~9.9k clips) | ⚠️ ~4 h feasible per spec | ✅ ~2.5 GPU-h | 96 GB recommended for headroom |
| `run_probe_train.sh pretrain --SANITY` | ❌ OOM at sub-batch=1 | ✅ ~3-5 min | Code-path validation only useful on 96 GB |
| `run_probe_train.sh pretrain --FULL` | ❌ same OOM | ✅ ~3 GPU-h | Produces `m09a_ckpt_best.pt` (~15 GB) |
| `run_probe_train.sh surgery_3stage_DI --SANITY` | ❌ same OOM | ✅ ~10 min | — |
| `run_probe_train.sh surgery_3stage_DI --FULL` | ❌ same OOM | ✅ ~6-8 GPU-h | Produces `m09c_ckpt_best.pt` (~15 GB) |
| `run_probe_train.sh surgery_noDI --SANITY` | ❌ same OOM | ✅ ~7 min | — |
| `run_probe_train.sh surgery_noDI --FULL` | ❌ same OOM | ✅ ~4-6 GPU-h | Produces `m09c_ckpt_best.pt` (~15 GB) |

## Why 24 GB can't train V-JEPA ViT-G (memory math)

| Component | bf16 / fp32 size |
|---|---|
| Student ViT-G (1.84B params) — fp32 weights on GPU | ~7.4 GB |
| Teacher ViT-G EMA copy (deepcopy of student) — fp32 | ~7.4 GB |
| Predictor (60M params) — fp32 | ~0.24 GB |
| Multi-task probe head (140K params) | <1 MB |
| Subtotal — model parameters | **~15 GB** |
| 8-bit Adam state (bitsandbytes paged — partial GPU) | ~1-2 GB |
| Master fp32 + AdamW state for trainable params | ~7-9 GB |
| PyTorch allocator overhead + bitsandbytes workspaces | ~2 GB |
| Subtotal — optimizer + framework | **~10-13 GB** |
| **FIXED total before any activations** | **~25 GB** |

That's already ≥ 24 GB. There's negative budget for activations, autograd graph, batch tensors, val_batches, mask gen, multi-task forward — hence sub-batch=1 + 8 frames + grad-ckpt + 8-bit-Adam + paged-optim + fragmentation-empty_cache stack still OOMs.

## Why eval works on 24 GB

`probe_action.py --stage features` does ONE encoder forward at a time, no teacher copy, no optimizer state, no autograd graph (inference mode). Per-clip activation budget is ~1-2 GB → easily fits with the ~17 GB freed up by not having teacher + optimizer.

## What to run on a fresh 96 GB instance

See [next_actions.md](next_actions.md). The TL;DR:
```bash
# Stage-1-only label bootstrap (CPU, ~2 min)
SKIP_STAGES="2,3,4,5,6,7,8,9,10" CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL
# Train all 3 V-JEPA variants (3 + 6-8 + 4-6 GPU-h, semicolon-chained NOT && — independent failures)
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain --FULL ; \
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL ; \
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI --FULL
# 4-encoder eval (~2.5 GPU-h)
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL
```

## Don't try

- ❌ Reducing num_frames at SANITY (tried 16→8 in `probe_pretrain_sanity_v6.log` — no help, FIXED memory dominates)
- ❌ Reducing batch_size at SANITY (sub-batch already at 1; macro batch_size doesn't change FIXED memory)
- ❌ Disabling multi-task at SANITY (multi-task forward is AFTER JEPA succeeds; JEPA itself OOMs)
- ❌ Switching to V-JEPA 2.0 ViT-g (1408-dim) — paper claim is 2.1; would invalidate framing
- ❌ Adding more memory savers — already running 8-bit Adam + paged optim + grad-ckpt + sub-batch=1 + bf16
