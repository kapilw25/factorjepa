---
name: Module map — what consumes what
description: Pipeline layout for FactorJEPA iter13 — script → module → output → consumer
type: project
---

# Pipeline layout — module map (iter13)

## Top-level scripts (`scripts/`)

| Script | Subcommand / mode | Modules invoked | Output namespace |
|---|---|---|---|
| `run_probe_eval.sh` | `--sanity` / `--FULL` | probe_action.py + probe_taxonomy.py + probe_motion_cos.py + probe_future_mse.py + probe_plot.py | `outputs/{sanity,full}/probe_*/` |
| `run_probe_train.sh` | `pretrain --SANITY/--POC/--FULL` | m09a_pretrain.py + (auto-gen) probe_taxonomy.py + utils/probe_train_subset.py | `outputs/{mode}/probe_pretrain/` |
| `run_probe_train.sh` | `surgery_3stage_DI --SANITY/--POC/--FULL` | m09c_surgery.py | `outputs/{mode}/probe_surgery_3stage_DI/` |
| `run_probe_train.sh` | `surgery_noDI --SANITY/--POC/--FULL` | m09c_surgery.py | `outputs/{mode}/probe_surgery_noDI/` |
| `run_factor_prep.sh` | `--FULL` | m10_sam_segment.py + m11_factor_datasets.py | `outputs/full/m10_*/` + `outputs/full/m11_factor_datasets/` (m09c surgery prereq) |

## Eval pipeline stages (`run_probe_eval.sh`)

```
Stage 1 (CPU, ~1-2 min)
   probe_action.py --stage labels   → outputs/<mode>/probe_action/action_labels.json (3-class, 70/15/15 split)
   probe_taxonomy.py --stage labels → outputs/<mode>/probe_taxonomy/taxonomy_labels.json (16-dim)

Stage 2 (GPU, ~1 h × N encoders)
   probe_action.py --stage features → outputs/<mode>/probe_action/<encoder>/features_{train,val,test}.npy
                                       (uses utils.frozen_features.load_vjepa_2_1_frozen
                                        which calls resolve_encoder_state_dict() — handles all 4 ckpt schemas)

Stage 3 (GPU, ~30 min × N encoders)
   probe_action.py --stage train    → outputs/<mode>/probe_action/<encoder>/probe.pt + test_metrics.json
                                       (Meta's AttentiveClassifier, 50 ep SANITY / 20 ep FULL,
                                        AdamW lr=5e-4 wd=0.05 + cosine 10% warmup, best-by-val-acc)

Stage 4 (CPU, ~5 min) — 🔥 P1 GATE
   probe_action.py --stage paired_delta → outputs/<mode>/probe_action/probe_paired_delta.json
                                          (N-way pairwise BCa Δ across encoders + by_encoder summary
                                           + legacy back-compat keys for m08d_plot consumers)

Stage 5 (CPU, ~2 min × N encoders)
   probe_motion_cos.py --stage features → outputs/<mode>/probe_motion_cos/<encoder>/pooled_features_test.npy
                                          (mean-pool over patch tokens; reuses Stage 2 cache via --share-features)

Stage 6 (CPU, ~1 min × N encoders)
   probe_motion_cos.py --stage cosine   → outputs/<mode>/probe_motion_cos/<encoder>/per_clip_motion_cos.npy
                                          (intra-class − inter-class cos)

Stage 7 (CPU, ~1 min)
   probe_motion_cos.py --stage paired_delta → outputs/<mode>/probe_motion_cos/probe_motion_cos_paired.json

Stage 8 (GPU, ~30 min × V-JEPA variants only)
   probe_future_mse.py --stage forward → outputs/<mode>/probe_future_mse/<variant>/per_clip_mse.npy + aggregate_mse.json
                                          (encoder + predictor; reads m09{a,c}_ckpt_best.pt
                                           via resolve_encoder_state_dict() + _load_predictor_2_1)
                                          DINOv2 has no predictor — auto-skipped.

Stage 9 (CPU, ~1 min)
   probe_future_mse.py --stage paired_per_variant → outputs/<mode>/probe_future_mse/probe_future_mse_per_variant.json

Stage 10 (CPU, ~5 s)
   probe_plot.py → outputs/<mode>/probe_plot/{probe_action_loss,probe_action_acc,probe_encoder_comparison}.{png,pdf}
                   (NaN-safe ylim handles degenerate BCa CIs at small N)
```

## Training pipeline (`run_probe_train.sh`)

```
Pre-flights (CPU, instant)
   1. action_labels.json must exist (else sends user to: SKIP_STAGES="2,3,4,5,6,7,8,9,10" run_probe_eval.sh)
   2. bitsandbytes import-check (only if --SANITY mode wants 8-bit Adam)
   3. taxonomy_labels.json — auto-generates via probe_taxonomy.py --stage labels if missing

Auto-generation
   utils/probe_train_subset.py → data/eval_10k_train_split.json (~6,966 keys)
                              → data/eval_10k_val_split.json    (~1,492 keys)

Dispatch (one trainer per invocation)
   pretrain          → m09a_pretrain.py    --train-config configs/train/probe_pretrain.yaml
   surgery_3stage_DI → m09c_surgery.py     --train-config configs/train/surgery_3stage_DI.yaml      --factor-dir outputs/full/m11_factor_datasets
   surgery_noDI      → m09c_surgery.py     --train-config configs/train/surgery_2stage_noDI.yaml    --factor-dir outputs/full/m11_factor_datasets

Outputs per trainer (consumed by run_probe_eval.sh Stages 2-9)
   outputs/<mode>/probe_pretrain/student_encoder.pt        (~7 GB, encoder only — Stages 2-7)
   outputs/<mode>/probe_pretrain/m09a_ckpt_best.pt         (~15 GB, full ckpt — Stage 8 future_mse)
   outputs/<mode>/probe_pretrain/multi_task_head.pt        (~1 MB, mt_head state)
   outputs/<mode>/probe_pretrain/loss_log.{jsonl,csv}      (per-step training loss + multi-task per-dim)
   outputs/<mode>/probe_pretrain/probe_history.jsonl       (mid-train kNN-centroid probe top-1 at every val cadence)
   (same 5-file pattern for probe_surgery_3stage_DI/ and probe_surgery_noDI/, with m09c_ckpt_best.pt instead)
```

## Shared utils (`src/utils/`)

| Module | Purpose | Consumers |
|---|---|---|
| `multi_task_loss.py` | MultiTaskProbeHead + compute_multi_task_probe_loss + 5 integration helpers (merge_config / build_head / attach_optim / run_step / export) | m09a + m09c |
| `frozen_features.py` | ENCODERS registry + `load_vjepa_2_1_frozen` + `load_dinov2_frozen` + `extract_features_for_keys` + **`resolve_encoder_state_dict`** (4-schema dispatch — Bug fix iter13) | probe_action, probe_motion_cos (via mean-pool), probe_future_mse |
| `training.py` | Shared SSL training primitives — load_config, producer_thread, build_mask_generators, _train_step_grad_accum, build_optimizer, save/load_training_checkpoint, export_student_for_eval, build_probe_clips, run_probe_acc_eval, FactorSampler, StreamingFactorDataset, **technique-agnostic** per #49 contract | m09a + m09c |
| `action_labels.py` | 3-class (walking/driving/drone) and 4-class action label derivation from path + tags | probe_action.py + m09a/m09c probe_history |
| `probe_train_subset.py` | Extract per-split clip_keys from action_labels.json | run_probe_train.sh |
| `gpu_batch.py` | AdaptiveBatchSizer + cuda_cleanup | m09a + m09c (training); probe_action features extraction |
| `vjepa2_imports.py` | Shim around `deps/vjepa2/` to avoid src/ namespace collision | everywhere V-JEPA models are constructed |
| `cache_policy.py` | DELETE PROTECTION machinery: add_cache_policy_arg + resolve_cache_policy_interactive + guarded_delete + wipe_output_dir | every m*.py + probe_*.py |
| `bootstrap.py` | bootstrap_ci + paired_bca (BCa 95% CIs) | probe_action.py + probe_motion_cos.py + probe_future_mse.py |
| `wandb_utils.py` | add_wandb_args + init_wandb + log_metrics + finish_wandb (no-op when --no-wandb) | every m*.py + probe_*.py |

## Encoder registry (`utils/frozen_features.py:ENCODERS`)

```python
{
  "vjepa_2_1_frozen":              {"kind": "vjepa",  "arch": "vit_gigantic_xformers", "crop": 384, "embed_dim": 1664},
  "vjepa_2_1_pretrain":            {"kind": "vjepa",  "arch": "vit_gigantic_xformers", "crop": 384, "embed_dim": 1664},
  "vjepa_2_1_surgical_3stage_DI":  {"kind": "vjepa",  "arch": "vit_gigantic_xformers", "crop": 384, "embed_dim": 1664},
  "vjepa_2_1_surgical_noDI":       {"kind": "vjepa",  "arch": "vit_gigantic_xformers", "crop": 384, "embed_dim": 1664},
  "dinov2":                        {"kind": "dinov2", "model_id": "facebook/dinov2-with-registers-giant", "crop": 224, "embed_dim": 1536},
}
```

The 4 V-JEPA variants share the loader (`load_vjepa_2_1_frozen`) which dispatches state_dict resolution via `resolve_encoder_state_dict` for all 4 ckpt schemas.

## Per-encoder ckpt resolvers in `run_probe_eval.sh`

```bash
encoder_ckpt_for() { ... }              # → student_encoder.pt for Stages 2-7 (encoder-only)
encoder_predictor_ckpt_for() { ... }    # → m09{a,c}_ckpt_best.pt for Stage 8 (full ckpt with predictor)
```

Two functions because Stages 2/3 only need encoder, but Stage 8 future_mse also needs the predictor.
