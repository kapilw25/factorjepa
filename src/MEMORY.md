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

### m09: Continual Pretraining (Ch10, in progress)
- **m09_pretrain.py** (~940 lines): V-JEPA 2 student-teacher JEPA with EMA, L1 latent prediction, drift control. Epoch-based training (not step-based). Key features:
  - vjepa2 imports via `utils/vjepa2_imports.py` shim (namespace collision fix)
  - Epoch geometry: `steps_per_epoch = n_train // batch_size`, `total_steps = steps_per_epoch * max_epochs`
  - Epochs per mode from YAML: `configs/pretrain/vitg16_indian.yaml` → `optimization.max_epochs.{sanity,poc,full,winner}`
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

### m10-m12: Surgery Fine-Tuning (Ch11, NOT BUILT)
- **m10_sam_segment.py**: SAM3 segmentation → tracklets → agent/layout separation (TODO)
- **m11_factor_datasets.py**: Create D_L/D_A/D_I factor datasets (TODO)
- **m12_surgery.py**: Progressive prefix unfreezing training (TODO)

### Scripts
- **scripts/run_evaluate.sh**: Ch9 eval pipeline (m04→m05→m05b→m05c→m04d→m06→m06b→m07→m08→m08b). Hard fail on all errors.
- **scripts/run_pretrain.sh**: Ch10 training pipeline. Two phases:
  - Phase 1: 4-lambda ablation sweep (1 epoch each) → m09 train → m05 re-embed → m06 metrics per lambda → m08b compare
  - Phase 2: Winner selection (best Cycle@K from JSON) → 5-epoch deep run → re-embed → metrics
  - Auto-detect BS from profiler (75% VRAM, `gpu_gb - headroom`)
  - Auto-run profiler if `profile_data.json` missing
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for fragmentation
  - Hard fail everywhere (no WARNING, no continue, no fallback)
  - Per-lambda encoder names: `vjepa_lambda0`, `vjepa_lambda0_001`, etc.
  - Winner saved to `ablation_winner.json` (JSON-to-JSON, no stdout parsing)
  - MIN_CLIPS from `configs/pipeline.yaml`
- **scripts/run_surgery.sh**: Ch11 surgery pipeline (empty stub).
- **scripts/profile_vram.py**: VRAM profiler for ViT-g training. Sweeps batch sizes [1..256], measures with/without gradient checkpointing. Generates 5 diagnostic plots. Uses CWD trick for vjepa2 imports.

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
- **configs/pretrain/vitg16_indian.yaml**: V-JEPA 2 ViT-g training config. Data (frames, crop, patch, val split, sanity clips), model (arch, embed dim, pred depth, mask tokens, RoPE, activation checkpointing), masking (8 small + 2 large blocks), augmentation, optimization (LR, EMA, epochs per mode, warmup, grad clip, loss_exp), drift control (lambda), validation (interval, Cycle@K), checkpointing (saves_per_epoch, keep_last_n), mixed precision (bfloat16).

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

## Current Status
- **Ch9: COMPLETE** — 5-encoder comparison on 10K POC. Baseline: Prec@K=36.1% (frozen V-JEPA)
- **Ch10: COMPLETE (10K POC)** — Pipeline validated end-to-end. Key results:
  - Winner: λ=0.001 (lowest jepa_loss=1.4914, 5 epochs)
  - Adapted vs Frozen: Prec@K 36.14% vs 36.09% (Δ=+0.05%, **noise**)
  - Cycle@K: 75.31% vs 76.01% (slight regression)
  - **Conclusion: 10K clips insufficient for 1B model adaptation. 115K full corpus needed.**
  - Total GPU time: ~8h (training + re-embedding + metrics)
  - student_encoder.pt for winner (λ=0.001) was lost due to unguarded deletion — rebuilt with epoch-count protection
- **Ch10: NEXT** — 115K full corpus (1 epoch, λ=0.001 only, skip ablation sweep)
- **Ch11: NOT BUILT** — m10/m11/m12 planned, code not started

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
