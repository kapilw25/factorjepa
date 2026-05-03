# WalkIndia-200K Project Memory

## Snapshot Sync Rule
When you update this file OR `src/CLAUDE.md`, copy the updated version to its snapshot:
- This file → `src/MEMORY.md`
- `src/CLAUDE.md` → already in-repo, no extra copy needed
Both snapshots are version-tracked in git. Keep them in sync.

## Project Overview
Research benchmark testing whether V-JEPA 2.1 (Meta's video foundation model, trained on Western data) transfers to Indian street scenes. Pipeline: YouTube videos → scene-split clips → WebDataset shards (HF) → VLM tagging → V-JEPA embeddings → FAISS metrics → UMAP → plots → ExPLoRA / Surgery / Multi-task fine-tuning.

## Pipeline Modules

### m00-m03: Data Pipeline (CPU)
- **m00_data_prep.py**: Parse YT_videos_raw.json → word freq, city × tour-type matrix.
- **m00b_fetch_durations.py**: yt-dlp metadata fetch (parallel, no download).
- **m00c_sample_subset.py**: Video-level uniform stratified sampling → `data/subset_*.json`.
- **m00d_download_subset.py**: WebDataset TAR pre-download from HF (8 workers, streaming, resume-safe).
- **m00e_difficulty_split.py**: Easy/Medium/Hard bucket stratification using m04 VLM-tag triggers + confidence ≥ 0.7.
- **m00f_category_subsets.py**: 8 single-condition subsets + `ultra_hard_3066` intersection (≥4 hard triggers AND ≥4 Indian objects); 80/10/10 train/val/eval split via seeded shuffle (2,452 / 306 / 308 clips).
- **m01_download.py**: yt-dlp + aria2c at 480p, resumable.
- **m02_scene_detect.py**: PySceneDetect → greedy [4-10s] cuts → libx264 CRF 28.
- **m02b_scene_fetch_duration.py**: ffprobe scan → `clip_durations.json` + `docs/static/stats.json`.
- **m03_pack_shards.py**: Streaming TAR packing → HF upload → delete; generates README on HF.

### m04-m08b: Evaluation Pipeline (probe trio added in iter13 for the motion-centric P1 gate)
- **m04_vlm_tag.py**: 3 VLM backends (Qwen3-VL default / VideoLLaMA3 / LLaVA-NeXT). Orchestrator/worker for VRAM, AdaptiveBatchSizer, checkpoint/resume. 16 closed-vocab fields per clip.
- **m04_vlm_tag_vllm.py**: vLLM (Qwen3-VL only) variant for full 115K — separate `venv_vllm`.
- **m04b_vlm_select.py**: CPU 5-criterion bake-off scoring (parse rate, agreement, speed, taxonomy fit, confidence cal).
- **m04c_sanity_compare.py**: CPU 4-metric sanity dashboard between VLM backends.
- **m04d_motion_features.py**: GPU-RAFT optical flow → 13D motion vector per clip. AdaptiveBatchSizer.
- **m05_vjepa_embed.py**: V-JEPA 2.1 ViT-G (2B, bf16, FA2, torch.compile) via `_resolve_model(user_model)` (#42). RoPE Q/K cast to V.dtype before SDPA (#44). AdaptiveBatchSizer (#46/#47). Fingerprinted `.m05_checkpoint_*_<fp>.npz` (#75) — variants do not collide. Partial-tolerance `failed_clip_keys_*.json` if ≥80% but ≤95% decode (#72). HF AutoModel flash_attention_2 for 2.0 path. Permute (B,T,C,H,W) → (B,C,T,H,W). No deduplication.
- **m05b_baselines.py**: 5 baselines (Random, Oracle = multi-hot tag-vector ceiling, DINOv2-giant 1536-dim, CLIP-L 768-dim, Shuffled V-JEPA 1664-dim). AdaptiveBatchSizer.
- **m05c_true_overlap.py**: Augmented V-JEPA embeddings (paired aug A vs B) for True Overlap@K ground-truth.
- **m06_faiss_metrics.py**: FAISS-GPU kNN → 9 metrics × Easy/Hard + bootstrap 95% CI. Exclusion window + clip duration from `pipeline.yaml`. **Always-recomputes** (no cache_policy). Hard-fail on missing inputs.
- **m06b_temporal_corr.py**: 5 temporal metrics per encoder (Spearman embedding-vs-motion). k-means motion clusters from `pipeline.yaml`.
- **m06c_temporal_projection.py**: CPU PCA: project out (normal − shuffled) subspace from V-JEPA embeddings; 30-min experiment, potential paper centerpiece.
- **probe_action.py** (iter13 P1, ~412 LoC): 4 stages (`labels` → `features` → `train` → `paired_delta`). Uses Meta's exact `AttentiveClassifier` from `deps/vjepa2/src/models/attentive_pooler.py` via `utils.vjepa2_imports.get_attentive_classifier()`. AdamW lr=5e-4 wd=0.05 + cosine 10 % warmup + 50 ep + cross-entropy + best-by-val-acc. Per-clip `test_predictions.npy` ∈ {0,1} for paired-bootstrap. `paired_delta` runs BCa Δ V-JEPA − DINOv2 with `gate_pass = ci_lo > 0`.
- **probe_motion_cos.py** (iter13 P1 secondary, ~353 LoC): 3 stages (`features` → `cosine` → `paired_delta`). Default `--share-features`: mean-pools probe_action's cached `(N, n_tokens, D)` features over the n_tokens axis (no GPU re-extract). Vectorised intra-class − inter-class cosine via `S = emb_n @ emb_n.T` + class-mask reduction. Per-clip `motion_score = pos − neg`. BCa Δ paired test V-JEPA − DINOv2.
- **probe_future_mse.py** (iter13 P1 health check, V-JEPA-only, ~519 LoC): 2 stages (`forward` → `paired_per_variant`). Loads encoder + predictor + MaskGenerator from same V-JEPA `.pt` (target_encoder + predictor keys). 8-block ~15 % spatial mask via `_MaskGenerator(num_frames=…)`. Per-clip L1 between predictor output and EMA-target tokens on masked positions. AdaptiveBatchSizer + resumable `.probe_future_mse_ckpt.npz`. `paired_per_variant` does pairwise BCa across `vjepa_2_1_{frozen,explora,surgical}` (DINOv2 reported as `n/a — no future-frame predictor`).
- **m07_umap.py**: cuML GPU UMAP → 2D.
- **m08_plot.py**: CPU matplotlib UMAP scatter + kNN confusion matrix + kNN grids. Reads pre-computed `.npy`.
- **m08b_compare.py**: CPU multi-encoder grouped bars / radar / paired-Δ plots / LaTeX table with 95% CI. **Always-recomputes**. Auto-scans m06/m06b JSON.

### m09a / m09b / m09c: Training (post-#49 split, 2026-04-15)
**Why split**: m09_pretrain.py monolith (2164 lines, 3 entangled techniques via flags) risked silent regression; user-directed split into 3 physically isolated files sharing only `src/utils/training.py` primitives (#49 contract: zero `if cfg["technique"]` branches).

- **m09a_pretrain.py** (~1176 lines): Ch10 continual pretraining — full-param or layer-frozen V-JEPA 2.1 student-teacher JEPA with EMA, L1 latent prediction, drift control (λ·‖θ-θ₀‖²), λ ablation sweep. Epoch-based. AdaptiveBatchSizer + `_train_step_grad_accum` (#48). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at main() (#53). Per-λ encoder names.
- **m09b_explora.py** (~1049 lines): ExPLoRA — LoRA r=16 α=32 on blocks 2-47 + unfreeze blocks 0-1 + LayerNorm. Same JEPA loss; no drift control (LoRA params auto-excluded from `init_params`). Probe infrastructure: Prec@K trajectory + best-by-Prec@K + (disabled) early-stop triggers. Within-step OOM retry (#55).
- **m09c_surgery.py** (~685 lines): **PAPER NOVELTY** — N-stage progressive prefix unfreezing on FROZEN V-JEPA 2.1 with factor datasets. `FactorSampler(factor_manifest.json)` per-stage `mode_mixture`. `set_trainable_prefix(n_layers)` rebuilds optimizer between stages. Stages from variant yaml (`surgery_2stage_noDI` / `loud_agent` / `3stage_DI` / `*_multitask`). Streaming DataLoader when `factor_streaming.enabled`. Multi-task loss (α JEPA + β InfoNCE + γ TCC) with optional Kendall 2018 Uncertainty Weighting. Inter-stage cleanup (#58). Mode-gated 8-bit optim / grad-checkpointing / paged optim (sanity=true, poc/full=false). Per-stage warmup via `warmup_pct: 0.20` (#61). Within-step retry + 0-step fail-hard (#55). Per-stage pre-init of loss vars (#54).

### m10 / m11: Surgery Factor Datasets (Ch11)
- **m10_sam_segment.py** (~996 lines): Grounded-SAM pipeline — Grounding DINO-base (Swin-B, fp32 #37) does open-vocab text→box on 4 anchor frames `[0,4,8,12]` (multi-anchor #32) → HF `Sam3TrackerVideoModel` (transformers ≥5.5.4, #36) does box→mask refinement + propagation. Box clamp before xywh-normalize (#28). 17-cat fixed agent taxonomy from `surgery_base.yaml`. `load_dotenv()` at top (#21). `os._exit(0)` on success / `os._exit(1)` on crash (#14/#16). Outputs per-clip `.npz` (agent_mask, layout_mask, centroids/interactions JSON, mid-frame RGB) + `segments.json` + `summary.json` + `per_object_bboxes_json` (~5KB/clip). Composite quality gate.
- **m11_factor_datasets.py** (~853 lines): CPU. D_L (feathered Gaussian σ=15 blur on agents + feather σ=3), D_A (soft matte BG ×0.1), D_I (bbox-adaptive tubes via `make_interaction_tubes_from_bboxes` reading m10's `per_object_bboxes_json`). Temporal interpolation (`np.linspace(0, T_mask-1, T_vid)`, #17). Quality filters: `min_agent_area_pct=0.003` (#26), `max_agent_area_pct=0.70`. Outputs per-clip `.npy` + `_tube{i}.npy` + `factor_manifest.json`. Plots: `m11_factor_samples.png`, `m11_interaction_samples.png`, `m11_per_Videoclip_verify/*.mp4`. **`--streaming`** (iter9): manifest-only for non-verify clips → ~90% wall-time + ~340 GB disk saved at 10K. D_L/D_A generated on-demand at training time via `utils/factor_streaming.py::stream_factor` (bitwise parity with legacy disk path).

### Scripts (all THIN wrappers — logic in `.py`)
**Active (`scripts/`):**
- **`scripts/run_factor_prep.sh`**: `<factor-yaml>` → m10 (Grounded-SAM) → m11 (`--streaming`). Cache-policy prompts up-front (`m10`, `m11`); m10 recompute auto-invalidates m11.
- **`scripts/legacy2/run_train.sh`**: `<train-yaml1>...<train-yamlN>` → loops; dispatches `m09b` / `m09c` based on yaml's `data.module`. NO `set -e`. Per-variant cache-policy prompt against `${OUT_DIR}/*` glob.
- **`scripts/legacy2/run_eval.sh`**: `<train-yaml1>...<train-yamlN>` → shared frozen baseline (m05+m06, **flock mutex** at `outputs/full/m05_vjepa_embed/.frozen.lock`) → per-variant m05+m06+m08b at `<output_dir>/eval/`. `INCLUDE_BASELINES=1` adds random/oracle/dinov2/clip/vjepa_shuffled sweep. Final aggregate m08b at `outputs/full/m08b_aggregate/`.
- **`scripts/legacy2/run_paired_eval_10k.sh`**: iter10-era full eval_10k loop. 3-tier idempotency gates (G1/G2/G3). Frozen archive cp/restore hooks. Schema-drift kill (greps `missing: [1-9]+`).
- **`scripts/lib/common.sh`**: Shared helpers — `log`, `banner`, `prompt_cache` (with `CACHE_POLICY_ALL` + non-TTY fallbacks), `run_step` (PIPESTATUS-correct), `verify`, `bg_upload`, `start/stop_watchdog`, `finalize`.
- **`scripts/lib/yaml_extract.py`**: `<yaml> <dotted.key>` → scalar; walks recursive `extends:` chain with deep-merge; fail-loud on missing key.

**Legacy (`scripts/legacy/`)**: pre-iter11 `prep_data.sh` / `prep_eval_10k.sh` / `run_embed.sh` / `run_eval.sh` / `train_*.sh` / `run_iter9_10k.sh` / `run_iter10_overnight.sh` + parity tests `tests_streaming/test_parity{,_D_I}.py` + `test_wall_time.py` + `tests_probe_m09b/test_sanity.sh`.

### Root-level Scripts
- **setup_env_uv.sh**: UV-based env setup. `--mac` (CPU/lint), `--gpu --from-wheels` (10-step install from prebuilt sm_120 wheels release `sm120-cu128-py312`): PyTorch 2.12+cu128 → verify → requirements_gpu → FA2 → FAISS-GPU (source-built sm_120) → cuML → wandb → SAM 3.1 (`--no-deps` to preserve numpy>=2.3 per #5) → Grounding DINO pre-cache → `facebook/sam3` HF (~12 GB parallel HF_TRANSFER=1). Auto-downloads V-JEPA 2.1 ckpt via `aria2c -x 16 -s 16` (#7b). Idempotent vjepa2 RoPE patch heredoc (#44/#59).
- **git_pull.sh**: `git fetch + reset --hard origin/main + clean -fd` (preserves .gitignored data). `--code-only` skips HF.
- **git_push.sh**: User-invoked (Claude blocked from git state by hook).
- **build_faiss_sm120.sh**: Source-build FAISS-GPU 1.14.1 for Blackwell sm_120. `--install` reuses `/tmp/faiss_build`.

### Utils (`src/utils/`)
- **`action_labels.py`** (218 LoC, iter13) — `parse_action_from_clip_key` (path-prefix `tier1/<city>/<activity>`; `rain → walking`; optional 4-class monument override via `tags.scene_type=="heritage_tourist"`). `load_subset_with_labels`, `stratified_split` (rejects classes <5 per split — BCa CI floor), `write_action_labels_json` + `class_counts.json`. CPU self-test CLI. Used by probe_action + probe_motion_cos + probe_future_mse.
- **`frozen_features.py`** (327 LoC, iter13) — Shared encoder loaders + extractor for the probe trio. `ENCODERS = {vjepa_2_1_frozen, dinov2}`. `load_vjepa_2_1_frozen` (mirrors m05:629-670 — bf16, RoPE patch, SDPA monkey-patch). `load_dinov2_frozen` (fp16, FA2). `decode_to_tensor` (PyAV + ImageNet normalize + resize-shorter-side + center-crop). `forward_vjepa` (B,T,3,H,W → permute B,3,T,H,W; deep-supervision unwrap to last layer). `forward_dinov2` (B*T flatten → tile + concat over T, matches V-JEPA 2 paper §4.1 "tile + temporal pool"). `extract_features_for_keys` producer-consumer with AdaptiveBatchSizer + atomic `.probe_<label>_ckpt.npz` resume.
- **`bootstrap.py`** — BCa 10K-iter `scipy.stats.bootstrap`, vectorized.
- **`cache_policy.py`** — **SOLE cross-module destructive-delete guard**. `add_cache_policy_arg`, `resolve_cache_policy_interactive`, `guarded_delete`. Replaces removed `output_guard.py` (2026-04-26).
- **`checkpoint.py`** — Atomic `os.replace` save/load for `.npy` / `.npz` / `.json`. Returns empty defaults on missing/corrupt; never raises.
- **`config.py`** (792 LoC) — `load_merged_config(model_yaml, train_yaml)` deep-merge with recursive `extends:`. Path/constant single source of truth. Dynamic ENCODER_REGISTRY fallback. `get_pipeline_config()` cached.
- **`curate_verify.py`** — Composite-quality ranker for m10/m11 verify clips with diversity floor (≥3 cities, ≥3 activities).
- **`data_download.py`** — `ensure_local_data()` + `iter_clips_parallel()` returns `(queue, stop_event, reader_thread)` — NOT iterable (B8 footgun).
- **`eval_suite.py`** — Subprocess orchestrator chaining m06 → m06b → m07 → m08 → m08b per encoder.
- **`export_metadata.py`** — tags.json → per-directory `metadata.jsonl` for HF dataset upload.
- **`factor_streaming.py`** (157 LoC, iter9) — `stream_factor(mp4_bytes, mask_npz_path, factor_type, ...)` on-demand D_L/D_A generation. Bitwise parity with legacy m11 disk path (`scripts/legacy/tests_streaming/test_parity.py` PASSED). Shared ImageNet normalization between disk and streaming paths.
- **`gpu_batch.py`** — `compute_batch_sizes(vram)`, `AdaptiveBatchSizer` (grow toward `gpu_memory_target=0.85`, halve+cooldown on OOM), `cuda_cleanup()`.
- **`gpu_watchdog.py`** — Background util-monitor daemon (email on low-util streak); not in pipeline.
- **`hf_outputs.py`** — HF Hub mirror sync to `anonymousML123/factorjepa-outputs`. `_mirror_cleanup` deletes stale, `_stale_checkpoint_ignores` skips active ckpts (<120s mtime). Batched `delete_files` (~30s for 11k deletes). `HF_HUB_ENABLE_HF_TRANSFER=1`. Has `sys.path.insert` for CLI use (#3).
- **`hf_utils.py`** — HF auth (`_setup_hf_env`, `_get_token`) + dataset README/citation generator.
- **`live_debug.py`** — `faulthandler` SIGUSR1/SIGUSR2 stack-dump for ptrace-restricted containers; called by m09a/b/c main().
- **`plots.py`** — Tier-1 paper rcParams (bold serif, 300 DPI). `init_style`, `save_fig` (.png+.pdf), `plot_training_curves` (stage-segmented), `plot_val_loss_curves`, COLORS / ENCODER_COLORS palettes.
- **`probe_history.py`** — Robust reader preferring atomic `training_summary.json.probe_history` over jsonl (avoids torn records from concurrent writers).
- **`profile_vram.py`** — Empirical V-JEPA 2 ViT-G VRAM cost model → `outputs/profile/training/profile_data.json`.
- **`progress.py`** — `make_pbar(...)` tqdm factory with consistent ETA format.
- **`training.py`** (787 LoC, post-#49) — 20 technique-agnostic helpers shared by m09a/b/c. ZERO `if cfg["technique"]` branches. `build_model`, `build_optimizer` (`use_8bit_optim` → `bnb.optim.AdamW8bit` or `PagedAdamW8bit` per `paged_optim`), `build_mask_generators` (reads `cfg["model"]` per #51), `enable_gradient_checkpointing` (block-wise `torch.utils.checkpoint(use_reentrant=False)`), `_train_step_grad_accum` (#48), `compute_drift_loss`, `update_teacher_ema`, `export_student_for_eval`, `render_training_plots` (iter11 extraction). Houses `StreamingFactorDataset` (IterableDataset, fork-safe per-worker TAR cache + RNG), `UncertaintyWeights`, `compute_multitask_loss`.
- **`vjepa2_imports.py`** — CWD-based shim avoiding `src/utils/` namespace collision with vjepa2's own `src/utils/`. `finally:` block restores ALL saved `src.*` modules (#50). Exposes `get_vit_by_arch`, `get_vit_predictor`, `get_vit_predictor_2_1`, `get_mask_generator`, `get_apply_masks`, `get_attentive_classifier` (iter13 — Meta's exact `AttentiveClassifier` from `deps/vjepa2/src/models/attentive_pooler.py`, used by probe_action).
- **`video_io.py`** — `_USE_TORCHCODEC = False` (#10 SIGSEGV on Blackwell sm_120); PyAV is the active path. `decode_video_bytes`, `get_clip_key`, `create_stream`.
- **`wandb_utils.py`** — All functions no-op when `run=None`.

### Config Files
- **`configs/pipeline.yaml`**: Single source of truth — SANITY clip caps per module, streaming params (prefetch 8, decode_workers 8/16), GPU defaults (RTX Pro 6000 96GB profiled), **universal `gpu_memory_target: 0.85`**, per-module `*_initial_bs`, eval params (FAISS K=6, temporal pairs 100K), ENCODER_REGISTRY, VLM IDs, scene-detection thresholds, verify mins.
- **`configs/tag_taxonomy.json`**: VLM taxonomy v3 — 16 closed-vocab fields (scene_type 13 vals, time_of_day day/night, weather, crowd/traffic_density, traffic_mix, pedestrian_vehicle_separation, road_layout multi, road_surface 8 vals, infrastructure_quality, road_encroachment, notable_objects 15 vals, vegetation, lighting, video_quality).
- **`configs/YT_videos_raw.json`**: 714 YouTube IDs across drive/walk tours (Delhi/Mumbai/Bangalore/Chennai/Hyderabad/Goa/Jaipur).
- **`configs/model/vjepa2_0.yaml`**: Legacy V-JEPA 2.0 ViT-g (1B, 1408-dim, depth 40, pred_depth 12, masked-only L1, HF model `facebook/vjepa2-vitg-fpc64-384`).
- **`configs/model/vjepa2_1.yaml`**: **PRIMARY** V-JEPA 2.1 ViT-G (2B, 1664-dim, depth 48, 26 heads, pred_depth 24, `predict_all=true`, `weight_distance_loss=true`, `n_output_distillation=4`). Native ckpt at `checkpoints/vjepa2_1_vitG_384.pt` (no HF release).
- **`configs/train/base_optimization.yaml`**: Shared inheritance root. Data block (ultra_hard_3066 splits 2452/306/308 + factor_dir), masking (8 small + 2 large blocks ~85% mask), augmentation, AdamW (lr=5e-5, BS=32, 15 epochs FULL), cosine schedule (warmup_pct=15), grad_clip=1.0, `nan_tolerance=2`, EMA τ=0.99925, bf16 mixed precision. Multi-task loss block (α/β/γ + UW + tcc_enabled / tcc_scale). Probe block: val-loss plateau is the ONLY active early-stop trigger (kill_switch / prec_plateau / bwt all disabled — operate below CI_half noise floor).
- **`configs/legacy2/ch10_pretrain.yaml`**: Ch10 recipe. lr=1e-6, EWC drift λ ∈ [10, 100, 1000] (post-audit), FIM-weighted L2, VICReg, freeze_below=20.
- **`configs/legacy2/explora.yaml`**: ExPLoRA recipe. lr=5e-5 (inherited), unfreeze blocks 0-1 + LayerNorm, LoRA rank=16 α=32 on `qkv + proj`, `use_peft: true`, drift disabled. `module: m09b`, `adapted_encoder: vjepa_2_1_explora`.
- **`configs/train/surgery_base.yaml`**: Surgery shared root. `module: m09c`, `train_val_split` (POC only), `factor_streaming` (sanity/poc=false, full=true; 16 workers, persistent), Grounded-SAM (DINO 0.20/0.18, 17-cat agent taxonomy), interaction_mining default `enabled: false` + thresholds (min_overlap=4, d_max=0.20), surgery `warmup_pct=0.20`, drift L2 λ=1.0. Mode-gated `use_8bit_optim` / `gradient_checkpointing` / `paged_optim` (sanity=true, poc/full=false).
- **`configs/legacy2/surgery_2stage_noDI.yaml`**: `vjepa_2_1_surgical_noDI`. Stage 1 unfreeze 0.25 → `L:1.00`; Stage 2 unfreeze 0.50 → `L:0.30, A:0.70`. 50/50 budget.
- **`configs/legacy2/surgery_2stage_loud_agent.yaml`**: `vjepa_2_1_surgical_loud_agent` (iter10 v15b). Stage 2 mix `L:0.15, A:0.85`.
- **`configs/train/surgery_3stage_DI.yaml`**: `vjepa_2_1_surgical_3stage_DI`. `interaction_mining.enabled: true`. 0.40/0.30/0.30 budget. Stage 3 unfreeze stays 0.50 (NOT prior 0.75 BWT=−0.33 over-destabilization). Mix `L:0.15, A:0.15, I:0.70`.
- **`configs/legacy2/surgery_2stage_noDI_multitask.yaml`** + **`surgery_3stage_DI_multitask.yaml`**: Activate Kendall UW (`uncertainty_weighting: true`); `tcc_enabled: false` (v2 fix per iter12 #81 — TCC raw=6.6 vs JEPA=0.5 consumed 80% gradient).

### Data Files (`data/`)
- **`sanity_100_dense.json`**: 100-clip density-scored subset (73 tier1 + 26 tier2 + 1 goa).
- **`subset_10k.json`**: 10K (9,566 actual) video-level uniform subset, seed=42, from m00c.
- **`val_1k.json`**: 1K validation subset, seed=99.
- **`val_500.json`** / **`test_500.json`**: Stratified val/test splits (iter9 H2 fix).
- **`eval_10k.json`**: 10K paired-BCa eval set, seed=99 (used by `run_paired_eval_10k.sh`).
- **`val_1k_local/tags.json`** / **`subset_10k_local/tags.json`**: Per-clip 16-field VLM tags.
- **`subset_10k_local/manifest.json`**: 10-TAR WebDataset manifest.
- **`eval_10k_local/manifest.json`**: 10-TAR manifest for eval_10k. (No tags — m06 derives via subset.)
- **`full_local/manifest.json`**: 116-TAR manifest for full 115,687-clip corpus.
- **(Optional, materialized at run time)** `data/ultra_hard_3066{,_train,_val,_eval}.json` + `data/ultra_hard_3066_local/` — produced by `m00f_category_subsets.py`. iter11 v3 hard-pivot tier (2,452 / 306 / 308 80/10/10 split inside ultra_hard subset).

## ENCODER_REGISTRY (live in `configs/pipeline.yaml`)
| Encoder | Dim | Suffix | Type |
|---------|----:|--------|------|
| vjepa (legacy 2.0) | 1408 | "" | video |
| vjepa_2_0_frozen | 1408 | _vjepa_2_0_frozen | video |
| random | 1408 | _random | synthetic |
| oracle | (tag-vocab) | _oracle | synthetic upper bound |
| dinov2 | 1536 | _dinov2 | image (middle frame) |
| clip | 768 | _clip | image (middle frame) |
| vjepa_shuffled | 1408 | _vjepa_shuffled | video (shuffled) |
| vjepa_adapted | 1408 | _vjepa_adapted | video (Ch10 adapted) |
| vjepa_2_1_frozen | 1664 | _vjepa_2_1_frozen | video |
| vjepa_2_1_frozen_shuffled | 1664 | _vjepa_2_1_frozen_shuffled | video (shuffled) |
| vjepa_2_1_explora | 1664 | _vjepa_2_1_explora | video (adapted) |
| vjepa_2_1_surgical | 1664 | _vjepa_2_1_surgical | video (adapted) |

> Variant-specific `data.adapted_encoder` keys (e.g. `vjepa_2_1_surgical_noDI` / `_loud_agent` / `_3stage_DI` / `_3stage_DI_multitask` / `_noDI_multitask`) are dynamically resolved via `get_encoder_info()` and inherit dim=1664 / type=video_adapted from the base `vjepa_2_1_surgical` registry entry. m05/m06 outputs do not collide across variants.

## Cross-Module Dependencies
```
Ch9 Eval pipeline (legacy/prep_data.sh + run_embed.sh + legacy/run_eval.sh):
  m00d → m04 → m04d → m05 → m05b → m05c → m06 → m06b → m07 → m08 → m08b

Ch10 brute-force (legacy/train_pretrain.sh):  m09a (λ ablation) → m05 re-embed → m06
ExPLoRA (run_train.sh + run_eval.sh):          m05 frozen → m09b → m05 re-embed → m06 → m08b
Surgery (run_factor_prep.sh → run_train.sh →   m10 → m11 → m09c → m05 re-embed → m06 → m08b
         run_eval.sh):
Multi-task (same surgery flow, alt yaml):      m10 → m11 → m09c (UW) → m05 re-embed → m06 → m08b
```

## Current Status (updated 2026-04-29)

### Empirical record
| Tier | Run | Δ Prec@K vs Frozen | Verdict |
|---|---|---|---|
| **Ch9 baseline (10K POC)** | frozen V-JEPA 2.0 | — | Prec@K 36.1 % (ref) |
| **Ch10 (115K λ=0.001)** | m09a vanilla | **−21.8 pp** ✅ sig | ❌ catastrophic forgetting |
| **iter8 1K POC Surgery** | m09c 3-stage | +0.17 pp | ❌ N=100 underpowered (CI ±4.5) |
| **iter9 v10/v13/v14** | m09c 2-stg / 3-stg | +0.07 to +0.14 pp on test_500 | ❌ <+3 pp gate; 20× below NeurIPS bar |
| **paired_eval_10k v10/v13** | N=9,297 BCa | −0.01 to −0.08 pp p>0.6 | 🟡 saturated |
| **iter11 v3 ultra_hard_3066 C** | surgery_3stage_DI | **+0.87 pp p=0.0038** | 🟠 first stat-sig but tiny |
| **iter12 multitask E v3** | UW 2-task drop TCC | +0.05 pp vs v1 | ❌ loss balance fixed; Prec@K still flat |
| **115K full m09a pretrain** | full encoder all layers | no improvement | ❌ scale + capacity also don't unlock |

### Pivot (iter13 — 2026-04-28, refined to 2-track plan 2026-04-29)

Five distinct recipes (small surgery / 115K full pretrain / multi-task / staged factor unfreeze / loss-balance fix) all failed to lift Prec@K meaningfully. Encoder is at equilibrium on this data; no eval metric will rescue a no-shift training problem.

**iter13 = adopt Meta's actual V-JEPA 2 transfer recipe: frozen encoder + small new head, evaluated on motion-centric tasks (Meta's published bench), not cross-clip retrieval.**

| | Track 1 (motion-centric backbone sanity gate) | Track 2 (Indian factor-conditioned probe) |
|---|---|---|
| Goal | 🥇 P1: `vjepa_frozen > dinov2_frozen` on a SSv2-style action probe | 🥈🥉 P2/P3: `vjepa_explora > vjepa_frozen`, then `vjepa_surgical > vjepa_explora`, on the same probe |
| Modules | `probe_action.py` + `probe_motion_cos.py` + `probe_future_mse.py` (+ utils `action_labels.py`, `frozen_features.py`); orchestrated by `scripts/run_probe_eval.sh` | `m12_action_labels.py` + `m13_probe_train.py` + `m13b_factor_probe_train.py` (NOT YET BUILT) |
| Data | `data/eval_10k.json` + `data/eval_10k_local/` (3-class default; 4-class with `--enable-monument-class`) | `data/ultra_hard_3066{,_train,_val,_eval}.json` + D_L/D_A/D_I from m10/m11 |
| Gate | Δ accuracy V-JEPA − DINOv2 (BCa, gate=ci_lo>0 ∧ p<0.05) | Δ accuracy surgical_probe − vanilla_probe (same paired BCa) |
| Budget | ~2.5 GPU-h | ~10 GPU-h |
| Decision | If Δ ≤ 0 → cancel P2/P3, diff `m05` vs `vjepa2_demo.ipynb`, pivot harder | Worst case = dataset + pipeline + negative finding paper |

### Active research artifacts
- iter12 ultra_hard_3066 5-variant comparison done (logged in `iter/utils/experiment_log.md`).
- Variants F (3stage_DI_multitask) **killed by user** at step 45/1140 — predicted to add 0 over E v3 stage 2 = +0 D_A lift.
- Iter13 plan in `iter/iter13_motion_probe_eval/{plan_training.md, analysis.md, runbook.md, errors_N_fixes.md, plan_code_dev.md}`.

## Env Stack (pinned, 2026-04-30)
PyTorch **2.12.0.dev20260407+cu128** (rotated twice off CDN — see setup_env_uv.sh:27-38 for 2026-04-30 bumps), CUDA 12.8, FA2 2.8.3, FAISS-GPU 1.14.1 (source-built sm_120), cuML 26.04, SAM 3.1 (raw pkg via `--no-deps`), transformers **5.5.4** (`Sam3TrackerVideoModel`), bitsandbytes 0.49.2, Python 3.12, UV. Release tag: `sm120-cu128-py312`.

## Reference Paths (Iteration Docs — iter13 is ACTIVE)
- HIGH: `iter/iter13_motion_probe_eval/plan_training.md` — Track 1 backbone sanity gate + Track 2 factor-conditioned probe (BLOCKED on Track 1 PASS). Supersedes iter12 multi-task plan.
- LOW: `iter/iter13_motion_probe_eval/runbook.md` — terminal commands for `scripts/run_probe_eval.sh`. Iter12 runbook still useful for m13/m13b pattern (`iter/iter12_multitask_LOSS/runbook.md`); iter11 runbook at `iter/iter11_epoch15/runbook.md`.
- DEV: `iter/iter13_motion_probe_eval/plan_code_dev.md` — per-module specs + LoC budget for the probe trio.
- ERROR: `iter/iter13_motion_probe_eval/errors_N_fixes.md` — **80** entries iter8 → iter12 (rolled-up history; header still says "iter8 → iter11"). NOTE: `iter/iter10/` only contains `logs/`; no errors file.
- ANALYSIS: `iter/iter13_motion_probe_eval/analysis.md` — Q2 wrong-metric finding + Q7 Framing B pivot (probe-head options A/B/C/D, outcome matrix).
- LIVE: `iter/utils/experiment_log.md` — append-only post-completion experiment log.
- FALLBACKS: `iter/utils/literarure_survey.md` — 24 JEPA variants (SIGReg / VLA-JEPA / temporal projection).
- PROPOSAL (outdated, kept for reference): `iter/utils/FactorJEPA.md` Sections 8-11 — original surgery proposal; §10 (continual pretrain) and §11 (progressive unfreezing) retired by iter13.

## Lessons Learned (selected — full history in `iter/iter13_motion_probe_eval/errors_N_fixes.md` #1-80)
1. **vjepa2 namespace collision**: `src/utils/__init__.py` shadows vjepa2's `src/utils/`. CWD-based import shim + `finally:` restores ALL saved `src.*` modules (#50).
2. **SAM3 `--no-deps` undeclared deps**: Every runtime import must be in `requirements_gpu.txt` (pycocotools, einops, iopath, ftfy). Pattern: #5/#6/#7/#18/#19.
3. **transformers 4.57 → 5.5.4 migration**: `torch_dtype=` → `dtype=` (#37/#43); DINO text fp16 crashes → fp32 (#37); `box_threshold` → `threshold` + `labels` → `text_labels` (#24); `Sam3TrackerVideoProcessor.add_inputs_to_inference_session` wants depth-3 boxes (#38); reset on session not processor (#39); `output.object_score_logits` not `iou_scores` (#40).
4. **SAM 3.1 multiplex tracking**: needs `text + boxes` hybrid (#27). HF `Sam3TrackerVideoModel` unlocks 4.21× speedup (#36). Multi-anchor DINO re-seed every 4 frames caps drift (#32).
5. **V-JEPA 2.1 torch.compile + bf16**: RoPE Q/K emerge fp32 but V is bf16 → SDPA crashes under inductor. `q = q.to(v.dtype); k = k.to(v.dtype)` before SDPA. Idempotent heredoc in `setup_env_uv.sh` because `deps/vjepa2/` is wiped on every fresh clone (#44/#59).
6. **m05 `_resolve_model`**: routes `--model None + hf_model_id null` to native-frozen branch (unpacks `target_encoder`/`encoder`); never reassign `args.model = VJEPA_CHECKPOINT_PATH` (#42).
7. **m05 fingerprinted ckpt path**: `_checkpoint_fingerprint(model, mtime, size)` (#75) prevents cross-variant collision; multiple surgical variants can share `outputs/full/m05_vjepa_embed/` safely.
8. **m05 partial-tolerance** (#72): if 80% ≤ embedded < 95%, save partial `.npy` + `failed_clip_keys_*.json`, exit 0 — downstream m06/m08b consume successes only.
9. **m09 monolith split** (#49): `utils/training.py` is technique-agnostic; ZERO `if cfg["technique"]` branches.
10. **AdaptiveBatchSizer is universal infra**: wired into every GPU forward loop. `memory_cap = pipeline.yaml gpu_memory_target` (universal 0.85), `initial_size = per-module *_initial_bs` (#46/#47).
11. **Gradient accumulation preserves research integrity**: effective BS stays = `cfg.optimization.batch_size`; sub-batch loss scaled by `(micro/macro)` (#48).
12. **Within-step OOM retry + 0-step fail-hard** (#55): never export a misleading checkpoint produced from 0 successful optimizer steps.
13. **Cache-policy is the SOLE guard**: `output_guard.py` removed 2026-04-26. Each `.py` prompts via `input()` if `--cache-policy` not on CLI; shells gather prompts UPFRONT so chains run unattended (`CACHE_POLICY_ALL=1|2`).
14. **No shell-level `rm`** (iter11 META-fix): destructive deletes live in `.py` behind `--cache-policy` + `guarded_delete()`; shells use merge-`cp -rf` and atomic `os.replace` instead.
15. **flock-coordinated frozen baseline** (`run_eval.sh`): one process computes m05/m06 frozen; others block on `outputs/full/m05_vjepa_embed/.frozen.lock` then read cache. Same-host only (separate cloud instances each compute).
16. **YAML training-config sanity is load-bearing**: `max_epochs.poc=1` silently ran 3 total optimizer steps (#60); `warmup_steps>stage_steps` capped LR (#61). Preflight B38/B39 guard at CPU-time.
17. **torchcodec SIGSEGV on Blackwell**: `_USE_TORCHCODEC = False` in `video_io.py` (#10).
18. **SAM3 async-thread shutdown**: `os._exit(0)` on success + `os._exit(1)` on crash (#14/#16).
19. **vjepa2 patches re-apply on fresh clone** (#59): `setup_env_uv.sh` does `rm -rf deps/vjepa2 && git clone` every provision; RoPE patch must be idempotent heredoc.
20. **Hardware upgrade can beat code complexity** (#58 postscript): v8 teacher-CPU-offload superseded by 24GB → 96GB Blackwell migration (~$0.60/h delta, cents per SANITY).
21. **iter11 v3 anti-corr finding**: JEPA L1 anti-correlated with downstream Prec@K/mAP@K/Cycle@K (Pearson r = −0.21 to −0.68 across 11/12 cells). Motivated multi-task loss; multi-task didn't rescue Prec@K either.
22. **iter12 #81 — TCC dominated gradient**: raw TCC=6.6 vs JEPA=0.5 → TCC consumed ~80 % of signal, starving InfoNCE (~11 %). UW couldn't rebalance fast enough at LR=5e-5. Fix: `tcc_enabled: false` + 2-task UW [JEPA, InfoNCE] gives InfoNCE ~85 % gradient share.
23. **#79 — `getattr(args, key, default)` is BANNED** alongside `\|\| true`, `.get(k, non_None_default)`, bare `except: pass`. `_finalize` referenced `args.cache_policy` via inferred lexical scope; would NameError on first invocation, but dormant code paths hid it for months. Fix: positional kwargs everywhere (no defaults). codified in `src/CLAUDE.md` FAIL HARD bullet.
24. **#80 — long-lived shared `tmp_dir` causes tmpfs ENOENT after ~2M write/unlink cycles** (~5 h of m09b producer-thread). Fix: per-epoch `tempfile.mkdtemp(prefix=f"m09_e{epoch}_")` rotation in `producer_thread`. Also: `safe_key.endswith(".mp4")` strip in `decode_video_bytes` (doubled `.mp4.mp4` was misleading log noise, not the ENOENT cause). Lesson: any tmpfs handle held >1 h on a busy `/tmp` needs rotation cadence (per-epoch / per-stage / per-N-clips).

## User Preferences
- Never be a yes-man — give pros/cons like a Sr. AI/ML Research Engineer.
- Be brutally honest. Disagree when wrong, never hallucinate.
- Git: provide commit message text only, never run git commands (enforced by hook).
- GPU time is expensive — keep GPU ≥85% busy.
- When auditing, SHOW grep output as proof — user does not trust "I checked" claims.
- No hardcoded values in Python — YAML configs or runtime discovery only.
- Fail hard in research — silent errors cost paper rejections.
- WEBSEARCH before recommending any fix that trades off accuracy or throughput; cite ≥2 sources.
