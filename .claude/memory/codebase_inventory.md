---
name: Codebase inventory — canonical scripts + m*.py + utils + configs + data
description: One-line purpose per file across scripts/, src/m*.py, src/utils/, configs/, data/. Refresh when files added/renamed.
type: project
---

# Codebase inventory (project state, 2026-05-08)

> Pure file-level reference. For module-flow + stage map, see [pipeline_layout.md](pipeline_layout.md). For the canonical script flow, this file's "Canonical scripts" section is sufficient — pipeline_layout covers their stage-by-stage internals.

## 🟢 Canonical scripts (the 4 production wrappers)

```
scripts/
├── run_probe_eval.sh         — 10-stage paper-evidence pipeline; produces P1 GATE JSON
├── run_probe_train.sh        — m09a/m09c trainer dispatch (3 subcommands)
├── run_factor_prep.sh        — serial m10 → m11 (factor-data prep for surgery)
└── run_factor_prep_parallel.sh — N-worker m10 + merge + m11 (≈2× speedup at N=4)
```

Contract for ALL four (per `src/CLAUDE.md`):
- THIN wrapper — all logic in Python; no `python -c` inline math; no shell-level `rm`
- Cache-policy gathered UPFRONT (one prompt per output dir), propagated as `--cache-policy` to each `.py`
- `set -euo pipefail` + ERR trap; `;` not `&&` between independent runs
- Mode-gated yaml reads via `scripts/lib/yaml_extract.py` (no hardcoded numerics in shells)
- Hardcoded canonical paths in shells ARE allowed (CLAUDE.md no-default rule targets `src/m0*.py`)

### Per-script flow (one-line summary)

| Script | Subcommand / mode | Inputs | Outputs |
|---|---|---|---|
| `run_probe_eval.sh` | `--FULL` (or `--sanity`) | trainer ckpts + `eval_10k.json` + `motion_features.npy` + `vjepa2_1_vitG_384.pt` | `outputs/{mode}/probe_{action,taxonomy,motion_cos,future_mse,plot}/` (the 🔥 P1 GATE JSON) |
| `run_probe_train.sh` | `pretrain` / `surgery_3stage_DI` / `surgery_noDI` × `--SANITY/--POC/--FULL` | yaml + train/val/test splits + tags + motion_features | `outputs/<mode>/m09{a,c}_*/{student_encoder.pt,m09{a,c}_ckpt_best.pt,loss_log.{jsonl,csv},probe_history.jsonl}` |
| `run_factor_prep.sh` | `<factor-yaml>` × mode | `data/eval_10k_local/` + `surgery_3stage_DI.yaml` (recommended) | `<local_data>/m10_sam_segment/` + `<local_data>/m11_factor_datasets/D_{L,A,I}/` |
| `run_factor_prep_parallel.sh` | `<factor-yaml>` × N-workers × mode | same as serial | same as serial; per-worker scratch in `m10_sam_segment_w{i}/` merged into canonical |

### Pre-flights in `run_probe_eval.sh` (load-bearing)

1. **Drop encoders without `student_encoder.pt`** (line 304-341) — pipeline continues with whatever exists
2. **Build `STAGE8_ENCODERS`** subset whose `m09{a,c}_ckpt_best.pt` (predictor-bearing) exists — Stage 8 uses this subset; Stages 2-7 use the full ENCODERS list
3. **Drop `m09{a,c}_ckpt_latest.pt`** (~29 GB resume anchors) once `student_encoder.pt` exists — opt-out via `EVAL_KEEP_LATEST=1`

### Per-encoder fused loop (Stages 2/3/3.5/5/6/8) in `run_probe_eval.sh`

- Stage 2 features (~31 GB fp32 / ~16 GB fp16) freed after Stage 5 motion_cos consumes them — bounds peak disk by ONE encoder's footprint, not N
- Stage 8 in-loop check is WARN+continue (not FATAL) — defense-in-depth against `SKIP_STAGES` bypass
- Lazy-extract: train+val features cached in `.probe_features_<split>_ckpt.npz`; reused across LR sweep runs

---

## 🐍 `src/m*.py` — compute pipeline (28 modules)

> Numeric prefix `m{NN}` avoids import collision (CLAUDE.md rule 32). Suffixed variants (`m04b`, `m09a/c`) signal related-but-isolated modules at the same numeric stage.

| Module | Purpose | GPU? |
|---|---|---|
| `m00_data_prep.py` | Top-level data prep orchestrator | CPU |
| `m00b_fetch_durations.py` | YouTube duration scraper | CPU |
| `m00c_sample_subset.py` | Stratified sampling from manifest | CPU |
| `m00d_download_subset.py` | MP4 download + tar-shard packing (`subset-*.tar`) | CPU |
| `m00e_difficulty_split.py` | Hard/medium/easy clip stratification | CPU |
| `m00f_category_subsets.py` | Category-level subset generation | CPU |
| `m01_download.py` | yt-dlp wrapper | CPU |
| `m02_scene_detect.py` | PySceneDetect → clip MP4s | CPU |
| `m02b_scene_fetch_duration.py` | Per-scene duration metadata | CPU |
| `m03_pack_shards.py` | webdataset-style tar packing | CPU |
| `m04_vlm_tag.py` | Qwen-VL tagging via transformers → `tags.json` | GPU |
| `m04_vlm_tag_vllm.py` | Same via vLLM (faster offline batch — currently SKIPPED, see vLLM_plan_Blackwell.md) | GPU |
| `m04b_vlm_select.py` | Tag-quality dedup | CPU |
| `m04c_sanity_compare.py` | VLM tag sanity vs labels | CPU |
| `m04d_motion_features.py` | RAFT optical flow → `motion_features.npy` (13-D × N clips) — Phase 5 will extend to 23-D FG motion | GPU |
| `m05_vjepa_embed.py` | V-JEPA frozen embeddings (legacy retrieval baseline) | GPU |
| `m05b_baselines.py` | DINOv2 / CLIP baseline embeddings | GPU |
| `m05c_true_overlap.py` | True-overlap eval (legacy) | CPU |
| `m07_umap.py` | UMAP projection (cuML GPU) | GPU |
| `m09a_pretrain.py` | 🟢 **Continual SSL pretrain** — JEPA L1 + drift L2 + motion_aux. v12 anchor (probe_top1=0.808). | GPU |
| `m09c_surgery.py` | 🟢 **Factor surgery** — 3-stage (D_L→D_A→D_I) or 2-stage. Phase 4 wires motion_aux; iter14 will add `--init-from-ckpt`. | GPU |
| `m10_sam_segment.py` | Grounded-SAM (DINO + SAM 3.1) per-clip masks | GPU |
| `m11_factor_datasets.py` | Build `D_L/D_A/D_I` factor tubes (streaming or disk mode) | GPU |
| `probe_action.py` | 🎯 ACTION PROBE — labels (motion-flow 8-class) / features / train (AttentiveClassifier) / paired_delta (P1 GATE) / select_best_lr | GPU + CPU |
| `probe_taxonomy.py` | 16-dim taxonomy probe — labels / train / paired_delta / plot | GPU + CPU |
| `probe_motion_cos.py` | Mean-pooled features → intra-class − inter-class cosine | CPU |
| `probe_future_mse.py` | Future-frame L1 (encoder + predictor); V-JEPA only | GPU |
| `probe_plot.py` | Plotter; reads JSONs from probe_* dirs → 3 PNG/PDF pairs | CPU |

---

## 🛠️ `src/utils/` — shared library (35 .py + 2 data files)

| Module | Purpose |
|---|---|
| `__init__.py` | Package marker |
| `action_labels.py` | Action-label derivation; `parse_optical_flow_class` (8-class motion-flow), `compute_magnitude_quartiles`, `stratified_split` |
| `bootstrap.py` | `bootstrap_ci` + `paired_bca` (BCa 95% CIs, 10K iter) |
| `cache_policy.py` | DELETE PROTECTION: `add_cache_policy_arg` + `resolve_cache_policy_interactive` + `guarded_delete` |
| `checkpoint.py` | `save/load_training_checkpoint(full=True/False)` for m09a/m09c |
| `config.py` | `load_merged_config(model_yaml, train_yaml)` deep-merge + `get_pipeline_config()` |
| `curate_verify.py` | Top-N curation filter (FIX-27b: top-N restricted to renderable video_ids) |
| `data_download.py` | HF dataset download + `_read_one_tar` (FIX-27a drop logging) |
| `eval_subset.py` | SANITY 200-clip stratified subset builder for `data/eval_10k_sanity.json` |
| `export_metadata.py` | Run-metadata export (config, git sha) |
| `factor_streaming.py` | `StreamingFactorDataset` + `stream_factor` (on-the-fly D_L/D_A from raw MP4 + mask.npz) |
| `frozen_features.py` | `ENCODERS` registry, `load_vjepa_2_1_frozen`, `load_dinov2_frozen`, **`resolve_encoder_state_dict`** (4-schema dispatch — Bug-iter13 fix) |
| `gpu_batch.py` | `AdaptiveBatchSizer` + `cuda_cleanup` (OOM-resilient training/inference) |
| `gpu_watchdog.py` | OOM/hang detector wrapper |
| `hf_finetuned_push.py` | Push trained encoder to HF model repo (used 2026-05-06 to push v12 anchor → `anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep`) |
| `hf_outputs.py` | HF dataset IO: `upload outputs/full` (auto-pack tars per FIX-25), `download outputs` (auto-unpack + delete tars per FIX-28), `_mirror_cleanup` |
| `hf_utils.py` | Generic HF helpers (auth, retry) |
| `live_debug.py` | Mid-train debugging hooks |
| `m09_common.py` | Shared m09a/m09c primitives (val sampling) |
| `m10_merge.py` | Union N-worker `m10_sam_segment_w{i}/` into canonical `m10_sam_segment/` |
| `m10_split_subset.py` | Split clip universe into N disjoint subsets for parallel m10 |
| `mask_metrics.py` | Mask quality metrics (IoU, area, persistence) |
| `motion_aux_loss.py` | 🎯 **iter14 MOTION_AUX**: `MotionAuxHead` (CE+MSE) + 5 helpers (merge_config / build_head / attach_optim / run_step / export). v12 lift +37 pp top1. |
| `multi_task_loss.py` | iter13 RETIRED pivot: `MultiTaskProbeHead` (16-dim CE+BCE) + 5 helpers (mirrors motion_aux contract) |
| `plots.py` | Shared matplotlib utilities |
| `probe_history.py` | Mid-train probe — emits `probe_history.jsonl` |
| `probe_labels.py` | Additional motion-flow class refinement helpers |
| `probe_stream.py` | Streaming probe-feature extraction (used during training) |
| `probe_train_subset.py` | Read `action_labels.json` → emit per-split clip-key JSONs |
| `probe_trio.py` | Mid-train triple metric (top1 + motion_cos + future_l1) on 1000-clip sample |
| `profile_vram.py` | Pre-run VRAM profiling → `profile_data.json` for auto-batch sizing |
| `progress.py` | `make_pbar` (tqdm wrapper with throughput windowing) |
| `tar_shard.py` | `pack_dir_to_shards` / `unpack_shards_to_dir` — used by `hf_outputs.py` for HF 10K-file cap workaround |
| `training.py` | Shared SSL primitives — load_config, producer_thread, build_mask_generators, _train_step_grad_accum, build_optimizer, save/load ckpt, export_student_for_eval, FactorSampler, StreamingFactorDataset (technique-agnostic per #49) |
| `video_io.py` | PyAV/decord MP4 read helpers |
| `vjepa2_imports.py` | Shim for `deps/vjepa2/` namespace (avoids src/ collision) |
| `wandb_utils.py` | `add_wandb_args` + `init_wandb` + `log_metrics` (no-op when `--no-wandb`) |
| `tag_taxonomy.json` | (data) 15-dim tag schema — duplicate of `configs/tag_taxonomy.json` for utils-internal use |
| `YT_videos_raw.json` | (data) raw YouTube URL list |

---

## 📋 `configs/` — yaml + json

| File | Purpose |
|---|---|
| `pipeline.yaml` | 🟢 Clip limits per mode, decode/streaming workers, GPU defaults (`gpu_memory_target`, batch sizes), eval params (`probe_head_train.{mode}.epochs/warmup_pct/lr_sweep`), encoder registry, `data.max_tar_shard_gb`, `action_probe_train.early_stop_*` |
| `probe_encoders.yaml` | Extension over `frozen_features.ENCODERS` (per-encoder crop, embed_dim, ckpt overrides) |
| `tag_taxonomy.json` | 15-dim taxonomy schema — multi-task probe label source |
| `YT_videos_raw.json` | Raw YouTube video URL list (m01 input) |
| `model/vjepa2_0.yaml` | 🔴 LEGACY V-JEPA 2.0 ViT-g (1B, 1408-dim) |
| `model/vjepa2_1.yaml` | 🟢 PRIMARY V-JEPA 2.1 ViT-G (2B, 1664-dim, depth=48, 384 crop) |
| `train/base_optimization.yaml` | 🟢 Shared base — data, masking, augmentation, AdamW, EMA, mp, probe, multi_task_probe (default `enabled: false` per mode) |
| `train/probe_pretrain.yaml` | 🟢 P2 m09a continual SSL — extends base; declares `motion_aux` block (lines 178-191), disables multi_task_probe (v12) |
| `train/surgery_base.yaml` | 🟢 Shared surgery scaffolding — `drift_control.lambda_reg`, mask/aug overrides for surgery |
| `train/surgery_3stage_DI.yaml` | 🟢 P3a — extends surgery_base; 3 stages (D_L → D_A → D_I, 40/30/30), interaction_mining=true. Phase 4 will copy motion_aux block here. |
| `train/surgery_2stage_noDI.yaml` | 🟢 P3b — 2 stages (D_L → D_A, 50/50), no D_I. Phase 4 will copy motion_aux block here. |
| `legacy2/ch10_pretrain.yaml` | 🔴 RETIRED iter10 pretrain (drift control + λ sweep) |
| `legacy2/explora.yaml` | 🔴 RETIRED ExPLoRA recipe (LoRA rank=16 + 2-block unfreeze) |
| `legacy2/surgery_2stage_loud_agent.yaml` | 🔴 RETIRED loud-agent variant |
| `legacy2/surgery_2stage_noDI_multitask.yaml` | 🔴 RETIRED iter13 multi_task surgery noDI |
| `legacy2/surgery_3stage_DI_multitask.yaml` | 🔴 RETIRED iter13 multi_task surgery 3stage_DI |

---

## 💾 `data/`

| File / Dir | Purpose |
|---|---|
| `eval_10k.json` | 🟢 PRIMARY eval set — 10K hard-mode Indian urban clip key list |
| `eval_10k_sanity.json` | SANITY subset (200 clips/action) — auto-generated by `src/utils/eval_subset.py` |
| `eval_10k_train_split.json` | 70% train (~6490 clips after motion-flow filter) |
| `eval_10k_val_split.json` | 15% val (~1388 clips) |
| `eval_10k_test_split.json` | 15% test (~1398 clips) |
| `eval_10k_local/` | 📁 Co-located storage for the 10K eval set |
| ↳ `manifest.json` | Master clip metadata |
| ↳ `tags.json` | VLM (Qwen-VL) tags per clip |
| ↳ `motion_features.npy` + `motion_features.paths.npy` | (9297, 13) RAFT optical-flow features — m04d output + per-row clip-key index |
| ↳ `subset-{00000..00009}.tar` | 10 webdataset tars holding raw MP4 clips |
| ↳ `verify_top20_manifest.json` | Top-20 visual verification subset |
| ↳ `m10_sam_segment/` | masks (`*.npz`) + `segments.json` + plots — m10 output |
| ↳ `m11_factor_datasets/` | `D_L/D_A/D_I/*.npy` + `factor_manifest.json` — m11 output |
| `subset_10k.json` + `subset_10k_local/` | 10K training subset (manifest + 10 tars + tags) |
| `val_1k.json` + `val_1k_local/` | 1K validation set (manifest + 1 tar + tags) |
| `val_500.json` / `test_500.json` | 500-clip validation/test lists (legacy) |
| `sanity_100_dense.json` | 100-clip dense sanity smoke set |
| `full_local/tags.json` | Tags for full 200K WalkIndia set (videos NOT local) |

---

## 🔴 Legacy (pointer-only, do not edit)

- `scripts/legacy/` — pre-iter11 .sh + .py (m00→m04 chain, train_explora, run_iter9_10k, etc.)
- `scripts/legacy/tests_streaming/` — iter9 v15c streaming parity + wall-time tests
- `scripts/legacy2/` — iter11/12 .sh (run_eval, run_paired_eval_10k, run_train)
- `src/legacy/` — m06_faiss_metrics, m06b_temporal_corr, m06c_temporal_projection, m08_plot, m08b_compare, m09b_explora, utils/eval_suite.py
- `configs/legacy2/` — ch10_pretrain, explora, multi_task surgery yamls (iter13 v1 retired by v12 motion_aux)

Retiring policy: `mv` to `legacy/` sibling, **never `rm`** (CLAUDE.md DELETE PROTECTION).

---

## 🎯 iter14 edit sites (next to touch)

Per `iter/iter14_surgery_on_pretrain/plan_surgery_on_pretrain.md` § T3 (~120 LoC across 7 files):

| # | File | Type | Edit |
|---|---|---|---|
| 1 | `configs/train/probe_pretrain_2X.yaml` | NEW (~12 LoC) | extends probe_pretrain.yaml; `max_epochs.full: 10` (was 5) — long-pretrain control arm C |
| 2 | `configs/train/surgery_3stage_DI_iter14.yaml` + `surgery_2stage_noDI_iter14.yaml` | NEW (~10 LoC each) | extends surgery_*; `max_epochs.full: 5`; `drift_control.lambda_reg: 0.005`; `anchor_to: pretrain` |
| 3 | `src/m09c_surgery.py` (lines 259-276 + ~line 1370) | EDIT (~25 LoC) | argparse `--init-from-ckpt`; `_load_init_state` dispatcher (priority: `student_state_dict` → `student` → `state_dict`); anchor loss in train_step |
| 4 | `scripts/run_probe_train.sh` (line ~50 + dispatch) | EDIT (~30 LoC) | accept `pretrain_2X` subcommand; thread `--init-from-ckpt` through surgery; repoint surgery to `*_iter14.yaml` |
| 5 | `scripts/run_probe_eval.sh` (line 143, 178, 188, 404) | EDIT (~15 LoC) | add `vjepa_2_1_pretrain_2X` to ENCODERS + 3 resolver cases |
| 6 | `src/probe_action.py --stage paired_delta` | EDIT (~20 LoC) | emit `iter14_paper_deltas.{delta_1_pretrain_vs_frozen, delta_2_surgical_vs_pretrain, delta_3_surgical_vs_pretrain_2X}` with BCa CIs |
| 7 | `iter/iter14_surgery_on_pretrain/runbook.md` | NEW (~50 LoC) | canonical sequence: surgery_3stage_DI → surgery_noDI → pretrain_2X → run_probe_eval → inspect Δ1/Δ2/Δ3 |

🔑 The schema branch (`student_state_dict` priority in step 3) is **empirically verified** against the live HF endpoint (`anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep`, 1.84 B params, 588 keys).

T4 BLOCKED on the 3 approval gates (epoch budget · anchor λ · HF push) per [project_pulse.md](project_pulse.md). HF push ✅; awaiting user reply on the other two.
