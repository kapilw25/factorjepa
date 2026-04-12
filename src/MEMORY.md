# WalkIndia-200K Project Memory

## Snapshot Sync Rule
When you update this file OR `src/CLAUDE.md`, copy the updated version to its snapshot:
- This file → `src/MEMORY.md`
- `src/CLAUDE.md` → already in-repo, no extra copy needed
Both snapshots are version-tracked in git. Keep them in sync.

## Project Overview
Research benchmark testing if V-JEPA 2 (Meta's video foundation model, trained on Western data) transfers to Indian street scenes. Pipeline: YouTube videos → scene-split clips → WebDataset shards (HF) → VLM tagging → V-JEPA embeddings → FAISS metrics → UMAP → plots → continual pretraining → surgery fine-tuning.

## Pipeline Modules

### m00-m03: Data Pipeline (CPU, completed)
- **m00_data_prep.py**: Parse YT_videos_raw.md → JSON, word freq, city matrix.
- **m00b_fetch_durations.py**: yt-dlp metadata fetch. Workers from `configs/pipeline.yaml`.
- **m00c_sample_subset.py**: Video-level uniform 10K subset → data/subset_10k.json. Default N from `configs/pipeline.yaml`.
- **m00d_download_subset.py**: Pre-download subset to local WebDataset TARs. Clips/shard from `configs/pipeline.yaml`.
- **m01_download.py**: Download videos via yt-dlp + aria2c. Resolution from `configs/pipeline.yaml`.
- **m02_scene_detect.py**: PySceneDetect → greedy [4-10s] split → ffmpeg encode.
- **m02b_scene_fetch_duration.py**: ffprobe scan clips → clip_durations.json.
- **m03_pack_shards.py**: Pack clips → WebDataset TARs. Clips/shard from `configs/pipeline.yaml`.

### m04-m08b: Evaluation Pipeline (Ch9, completed)
- **m04_vlm_tag.py** (~1416 lines): 3 VLM backends (Qwen/VideoLLaMA3/LLaVA). Orchestrator/worker pattern. All streaming params from `configs/pipeline.yaml`.
- **m04b_vlm_select.py**: CPU-only. 5-criterion weighted bake-off.
- **m04c_sanity_compare.py**: CPU-only. 4-metric table + dashboard.
- **m04d_motion_features.py**: GPU-RAFT optical flow → 13D motion features. All params from `configs/pipeline.yaml`.
- **m05_vjepa_embed.py** (~700 lines): V-JEPA 2 ViT-G (1B, fp16, FA2). Orchestrator/worker pattern. Supports `--encoder` flag for per-lambda unique output files. Adapted models: skip torch.compile (dtype conflict), use autocast, permute (B,T,C,H,W)→(B,C,T,H,W). No deduplication (circular reasoning removed). All streaming params from `configs/pipeline.yaml`.
- **m05b_baselines.py**: 4 baselines (Random, DINOv2, CLIP, Shuffled V-JEPA). Clip limits from `configs/pipeline.yaml`.
- **m05c_true_overlap.py**: Augmented V-JEPA embeddings for True Overlap@K. Workers from `configs/pipeline.yaml`.
- **m06_faiss_metrics.py** (~1275 lines): FAISS-GPU kNN → 9 metrics Easy/Hard + bootstrap CI. Exclusion window and clip duration from `configs/pipeline.yaml`. Hard fail on missing data (no silent fallbacks). Dynamic encoder names via `get_encoder_info()` fallback.
- **m06b_temporal_corr.py**: 5 temporal metrics per encoder. Sample pairs and clusters from `configs/pipeline.yaml`.
- **m07_umap.py**: cuML GPU UMAP.
- **m08_plot.py**: CPU-only matplotlib. Reads pre-computed .npy files.
- **m08b_compare.py**: CPU-only. Auto-scans m06/m06b JSON. Hard fail on missing temporal scores.

### m09: Training — ExPLoRA + Ch10 Pretrain + Surgery (Ch10/Ch11)
- **m09_pretrain.py** (~940 lines): V-JEPA 2 student-teacher JEPA with EMA, L1 latent prediction, drift control. Epoch-based training (not step-based). Key features:
  - vjepa2 imports via `utils/vjepa2_imports.py` shim (namespace collision fix)
  - Epoch geometry: `steps_per_epoch = n_train // batch_size`, `total_steps = steps_per_epoch * max_epochs`
  - Epochs per mode from YAML: `configs/train/ch10_pretrain.yaml` → `optimization.max_epochs.{sanity,poc,full,winner}`
  - SANITY clip limits from `configs/pipeline.yaml` → `sanity.pretrain_train/val`
  - No SANITY subset: collects first N clip keys from data stream for filtering
  - Full dataset size: discovered from `--subset` (JSON) or `--local-data` manifest.json. FATAL if undiscoverable.
  - Producer thread: multi-epoch loop (no `break`), `torch.set_num_threads(1)`, decode workers from config
  - No double LayerNorm on teacher output (ViT already normalizes)
  - GradScaler disabled for bfloat16 (unnecessary, bfloat16 has full dynamic range)
  - init_params stored on CPU (saves ~4GB VRAM), moved to GPU per-parameter in drift loss
  - Light checkpoints (no optimizer, ~8GB) for periodic saves; full checkpoints for resume only
  - `cleanup_old_checkpoints(keep_n)` deletes oldest step files
  - Post-training: exports `student_encoder.pt`, deletes ALL intermediate checkpoints
  - CSV loss log per step + wandb per step
  - LR warmup capped at 10% of total steps
  - `--max-epochs` CLI override for winner deep run

### m10-m11: Surgery Factor Datasets (Ch11, BUILT — untested on GPU)
- **m10_sam_segment.py** (~595 lines): SAM 3.1 text-prompted segmentation → agent/layout masks + interaction mining. Per-clip notable_objects from tags.json. Streaming API (`handle_stream_request` → `propagate_in_video`). Multiplex builder for 3.1. Quality gate: FATAL if mean_concept_recall < 0.3. Saves `.npz` masks + `segments.json` + `summary.json` + sample plots.
- **m11_factor_datasets.py** (~515 lines): CPU-only. Generate D_L (blur agents σ=15, feathered edges σ=3), D_A (suppress background, soft matte 10% residual), D_I (interaction tubes from centroids, ≥4 frame runs). Quality filters (min 2% / max 70% agent area). Saves `.npy` per clip + `factor_manifest.json`. All params from ch11_surgery.yaml.
- **m09 surgery mode** (IMPLEMENTED, untested on GPU): `train_surgery()` function (~270 lines). 3-stage progressive prefix unfreezing: builds `FactorSampler` from `factor_manifest.json`, loads `.npy` factor clips, per-stage optimizer rebuild + warmup. Stages from `ch11_surgery.yaml`: stage1 (layers 0-25%, 100% D_L), stage2 (0-50%, 90% D_A + 10% D_L), stage3 (0-75%, 85% D_I + 10% D_A + 5% D_L). Same V-JEPA 2.1 dense loss.

### Scripts
- **scripts/train_explora.sh**: ExPLoRA pipeline: m05(frozen baseline) → m09(ExPLoRA LoRA+unfreeze) → m05(re-embed adapted) → m06(metrics). 3 modes (SANITY/POC/FULL). Auto batch size from profiler, GPU pre-flight, signal trap for clean interrupt. Sources `lib/common.sh`.
- **scripts/train_surgery.sh**: Ch11 surgery pipeline: m10(SAM3) → m11(factor datasets) → m09(surgery) → m05(re-embed) → m06(eval). Same 3 modes. Factor dir at `${OUT_DIR}/factors/`.
- **scripts/run_eval.sh**: Standalone eval (reusable across Ch9/Ch10/Ch11). Auto-detects encoders from `embeddings*.npy`. Delegates to `eval_suite.py` which runs m06→m06b→m07→m08→m08b.
- **scripts/run_embed.sh**: Standalone embedding extraction. Auto-detects adapted models from `m09_*/student_encoder.pt`. Routes to m05 or m05b per encoder type.
- **scripts/prep_data.sh** (aka run_evaluate.sh): Ch9 data pipeline (m04 tags + m04d motion). Pre-flight GPU checks, optional vLLM backend.
- **scripts/lib/common.sh**: Shared infrastructure: `log()`, `banner()`, `run_step()` (PIPESTATUS capture, auto HF upload), `bg_upload()`, `start_watchdog()`, `finalize()`.
- **scripts/run_pretrain.sh** (Ch10, OLD): 4-lambda ablation → winner → 5-epoch deep run. Per-lambda encoder names.
- **scripts/profile_vram.py**: VRAM profiler for ViT-g training. Sweeps batch sizes [1..256]. Generates 5 diagnostic plots.

### Utils
- **utils/config.py** (~600 lines): Paths, constants, `get_pipeline_config()` (cached YAML reader), `get_sanity_clip_limit(module)`, `get_total_clips(local_data, subset_file)`. ENCODER_REGISTRY with dynamic fallback for lambda variants. `FAISS_K_NEIGHBORS`, `DEFAULT_BATCH_SIZE`, `DEFAULT_NUM_WORKERS`, `BAKEOFF_CLIP_COUNT` all from YAML.
- **utils/vjepa2_imports.py**: Import shim for vjepa2 modules. Temporarily changes CWD to /tmp and isolates sys.path/modules to avoid `src/utils/` namespace collision. Exposes `get_vit_giant_xformers()`, `get_vit_predictor()`, `get_mask_generator()`, `get_apply_masks()`.
- **utils/bootstrap.py**: BCa bootstrap 95% CI via scipy.stats.bootstrap.
- **utils/gpu_batch.py**: compute_batch_sizes(vram), AdaptiveBatchSizer class.
- **utils/hf_utils.py**: HF auth, upload helpers.
- **utils/wandb_utils.py**: Shared wandb integration. All functions no-op when run=None.
- **utils/export_metadata.py**: Convert tags.json → per-directory metadata.jsonl.
- **utils/output_guard.py**: Output verification before GPU work. `verify_or_skip()` for per-script guards (non-interactive, shape-validated). `verify_training_output()` for m09 (epoch-aware). `preflight_pipeline()` for shell scripts (checks ALL steps' inputs/outputs at pipeline start, interactive confirm). CLI: `python -u src/utils/output_guard.py preflight_pretrain|preflight_evaluate <args>`.

### Config Files
- **configs/pipeline.yaml**: Single source of truth for clip limits (SANITY per module), streaming params (retries, checkpoint interval, decode workers, prefetch queues), GPU defaults (batch sizes), eval params (FAISS K, temporal pairs), verification thresholds (MIN_CLIPS), data processing (clips/shard, resolution). All `src/m*.py` read from here via `get_pipeline_config()`.
- **configs/train/ch10_pretrain.yaml**: V-JEPA 2 ViT-g training config. Data (frames, crop, patch, val split, sanity clips), model (arch, embed dim, pred depth, mask tokens, RoPE, activation checkpointing), masking (8 small + 2 large blocks), augmentation, optimization (LR, EMA, epochs per mode, warmup, grad clip, loss_exp), drift control (lambda), validation (interval, Cycle@K), checkpointing (saves_per_epoch, keep_last_n), mixed precision (bfloat16).

## ENCODER_REGISTRY
| Encoder | Model | Dim | Suffix | Type |
|---------|-------|-----|--------|------|
| vjepa | vjepa2-vitg-fpc64-384 | 1408 | "" | video |
| random | — | 1408 | _random | synthetic |
| dinov2 | dinov2-vitl14 | 1024 | _dinov2 | image (middle frame) |
| clip | clip-vit-large-patch14 | 768 | _clip | image (middle frame) |
| vjepa_shuffled | vjepa2-vitg-fpc64-384 | 1408 | _vjepa_shuffled | video (shuffled) |
| vjepa_adapted | — | 1408 | _vjepa_adapted | video (Ch10 adapted) |
| vjepa_lambda* | — | 1408 | _vjepa_lambda* | video (per-lambda, dynamic fallback) |

## Cross-Module Dependencies
```
Ch9 Eval (run_evaluate.sh):
m00d → m04 → m05 → m05b → m05c → m04d → m06 → m06b → m07 → m08 → m08b

Ch10 Pretrain (run_pretrain.sh):
Phase 1: For each lambda: m09 train only (select winner by jepa_loss from JSON)
Phase 2: Winner → m09 deep train (5ep) → m05 re-embed → m06 metrics
Phase 3: m06b temporal → m05 shuffled adapted → m06 shuffled → m07 UMAP → m08 plots → m08b compare (7 encoders)
```

## Current Status (updated 2026-04-12)
- **Ch9: COMPLETE** — 5-encoder comparison on 10K POC. Baseline: Prec@K=36.1% (frozen V-JEPA)
  - **Key finding: shuffled > normal V-JEPA by 2.4x** → temporal interference (temporal encoding HURTS spatial retrieval)
- **Ch10 (10K POC): DONE** — Pipeline validated, Prec@K 36.14% vs 36.09% (**noise**, 10K insufficient)
- **Ch10 (115K FULL): CATASTROPHIC FORGETTING** (2026-04-05)
  - λ=0.001 → Prec@K crashed 36.1% → **14.3% (random-level, −21.8pp)**
  - Diagnosis: drift penalty 1000x smaller than JEPA loss → zero regularization
  - Gold standard audit found **12 discrepancies** (4 CRITICAL, 7 HIGH) — `iter/iter8/plan_training.md`
- **Strategic pivot (2026-04-10):**
  - Ch11 runs **directly on frozen encoder** (no Ch10 prerequisite) — clean attribution
  - V-JEPA 2.1 ViT-G (2B, 1664-dim) = PRIMARY target (not 2.0)
  - Temporal interference projection (30 min CPU) = potential paper centerpiece
  - λ=100 Ch10 = parallel ablation, NOT prerequisite
  - Idea Critic verdict: **PURSUE** (upgraded from REFINE)
  - Full plan: `iter/iter8/plan_training.md` | Action items: `iter/iter8/next_steps.md`
- **Ch11: ALL CODE BUILT, UNTESTED ON GPU** (2026-04-12)
  - m10 SAM 3.1 segmentation: DONE (595 lines)
  - m11 factor datasets: DONE (515 lines)
  - m09 surgery mode (`train_surgery()`): DONE (~270 lines, `FactorSampler` + `load_factor_clip` + 3-stage loop)
  - m09 ExPLoRA mode: DONE (LoRA injection + block freeze)
  - train_explora.sh + train_surgery.sh: DONE (full pipelines with common.sh infra)
  - **GPU instance provisioned**: RTX PRO 6000 Blackwell 96GB, env setup complete, HF data synced
  - **Next**: Execute runbook Steps A-E (SANITY), then POC — per `iter/iter8/runbook.md`

## Data Download Times (measured, RTX PRO 6000 instance, April 2026)

| Command | What it does | Time |
|---------|-------------|:----:|
| `m00d --FULL --no-wandb` | Downloads ALL 116 TARs from walkindia-200k, keeps all 115K clips | **24 min** |
| `m00d --FULL --subset data/subset_10k.json` | Downloads ALL 116 TARs, opens each, filters to 10K clips | **~50 min** (NOT 5 min — scans all 116 TARs) |
| `m00d --FULL --subset data/val_1k.json` | Downloads ALL 116 TARs, opens each, filters to 1K clips | **~50 min** (15 min if TARs cached) |
| `rsync data/ from Mac` | Transfers pre-filtered 10K (10 TARs) + val 1K (1 TAR) | **~17 min** (network limited) |
| `hf_outputs.py download-data` | Downloads poc 10K + val 1K from factorjepa-outputs | **~3 min** (measured, pre-filtered 11 TARs) |

**Key insight:** m00d always downloads ALL 116 TARs regardless of subset size. For POC/val data, rsync from Mac or hf_outputs download-data is 10-25x faster because the TARs are pre-filtered.

## Lessons Learned (Ch10 debugging, 2026-03-28/29)
1. **vjepa2 namespace collision**: Our `src/utils/__init__.py` shadows vjepa2's `src/utils/` (regular pkg wins over namespace). Fixed with CWD-based import shim.
2. **Disk management**: Each full checkpoint = 16GB (student+teacher+predictor+optimizer). 4 lambdas × 4 checkpoints = 256GB → disk full. Fixed: light checkpoints + cleanup.
3. **Lambda dir naming**: `str(0.0)` → `"0.0"` → `"0_0"` not `"0"`. Fixed: `f"{lam:g}"`.
4. **torch.compile + float16**: dynamo traces with float32 fake tensors → dtype mismatch. Fixed: skip compile for adapted models.
5. **Tensor permutation**: HF processor outputs (B,T,C,H,W), native vjepa2 expects (B,C,T,H,W). Fixed in `get_batch_embeddings()`.
6. **GradScaler + bfloat16**: GradScaler is a no-op for bfloat16 (same dynamic range as fp32). Disabled.
7. **Producer epoch loop**: `break` after stream exhaust → training stops after 1 pass. Fixed: removed break, added epoch counter.
8. **CUDA fragmentation**: 13.8GB reserved-but-unallocated after 30+ steps. Fixed: `expandable_segments:True`.
9. **Epoch vs step training**: Fixed BS × fixed steps = variable clips processed. Fixed: epoch-based (clips = n_train × epochs, independent of BS).
10. **Per-lambda file paths**: All lambdas wrote to same `embeddings_vjepa_adapted.npy`. Fixed: per-lambda encoder names via `--encoder vjepa_lambda*`.
11. **Winner stdout parsing**: `2>&1 | tee` mixes stderr into shell variable. Fixed: JSON-to-JSON via `ablation_winner.json`.
12. **Unguarded checkpoint deletion**: `run_pretrain.sh` deleted 5-epoch student_encoder.pt (3h GPU) without checking epoch count. Fixed: `verify_training_output(min_epochs)` + shell epoch guard + `protect-checkpoints.sh` hook.
13. **Output preflight**: Pipeline discovered missing inputs 3h into a run. Fixed: `preflight_pipeline()` in `output_guard.py` checks ALL steps' inputs/outputs at pipeline start, before any GPU work. Interactive confirm/abort.
14. **m05 re-embed dominates pipeline time**: 1.7h per 10K clips (1.55 clips/s). Skipped Phase 1 m05/m06 — winner selected by jepa_loss from JSON instead of Cycle@K. Full m05/m06 only for winner in Phase 2.

## User Preferences
- Never be a yes-man — give pros/cons like a Sr. AI/ML Research Engineer
- Be brutally honest. Disagree when wrong, never hallucinate
- Git: provide commit message text only, never run git commands (enforced by hook)
- GPU time is expensive — keep GPU busy, no idle waste
- When auditing, SHOW grep output as proof — user does not trust "I checked" claims
- No hardcoded values in Python — YAML configs or runtime discovery only
- Fail hard in research — silent errors cost paper rejections
