# WalkIndia-200K Project Memory

## Snapshot Sync Rule
When you update this file OR `src/CLAUDE.md`, copy the updated version to its snapshot:
- This file → `src/MEMORY.md`
- `src/CLAUDE.md` → already in-repo, no extra copy needed
Both snapshots are version-tracked in git. Keep them in sync.

## Project Overview
Research benchmark testing if V-JEPA 2 (Meta's video foundation model, trained on Western data) transfers to Indian street scenes. Pipeline: YouTube videos → scene-split clips → WebDataset shards (HF) → VLM tagging → V-JEPA embeddings → FAISS metrics → UMAP → plots → ExPLoRA/Surgery fine-tuning.

## Pipeline Modules

### m00-m03: Data Pipeline (CPU, completed)
- **m00_data_prep.py**: Parse YT_videos_raw.json → word freq, city matrix.
- **m00b_fetch_durations.py**: yt-dlp metadata fetch. Workers from `configs/pipeline.yaml`.
- **m00c_sample_subset.py**: Video-level uniform 10K subset → data/subset_10k.json.
- **m00d_download_subset.py**: Pre-download subset to local WebDataset TARs.
- **m01_download.py**: Download videos via yt-dlp + aria2c.
- **m02_scene_detect.py**: PySceneDetect → greedy [4-10s] split → ffmpeg encode.
- **m02b_scene_fetch_duration.py**: ffprobe scan → clip_durations.json.
- **m03_pack_shards.py**: Pack clips → WebDataset TARs.

### m04-m08b: Evaluation Pipeline (Ch9, completed)
- **m04_vlm_tag.py**: 3 VLM backends (Qwen3-VL default / VideoLLaMA3 / LLaVA-NeXT). Orchestrator/worker. AdaptiveBatchSizer wired (#47).
- **m04_vlm_tag_vllm.py**: vLLM backend variant for Qwen3-VL.
- **m04b_vlm_select.py**: CPU-only 5-criterion weighted VLM bake-off.
- **m04c_sanity_compare.py**: CPU-only 4-metric comparison dashboard.
- **m04d_motion_features.py**: GPU-RAFT optical flow → 13D motion features. AdaptiveBatchSizer wired (#47).
- **m05_vjepa_embed.py**: V-JEPA 2.1 ViT-G (2B, bf16, FA2, torch.compile) via `_resolve_model(user_model)` helper (#42): `None → (None, is_adapted=False) → native-frozen branch` unpacks `target_encoder`/`encoder` from .pt; `.pt path → adapted-student branch`. RoPE Q/K cast to V.dtype before SDPA (#44). AdaptiveBatchSizer (#46/#47). HF AutoModel flash_attention_2 for 2.0 path. Permute (B,T,C,H,W)→(B,C,T,H,W). No deduplication.
- **m05b_baselines.py**: 4 baselines (Random, DINOv2-giant 1536-dim, CLIP-L 768-dim, Shuffled V-JEPA). AdaptiveBatchSizer wired.
- **m05c_true_overlap.py**: Augmented V-JEPA embeddings for True Overlap@K. Paired-forward AdaptiveBatchSizer (initial = vjepa/2).
- **m06_faiss_metrics.py**: FAISS-GPU kNN → 9 metrics Easy/Hard + bootstrap 95% CI. Exclusion window + clip duration from pipeline.yaml. Hard fail on missing data.
- **m06b_temporal_corr.py**: 5 temporal metrics per encoder. k-means motion clusters from pipeline.yaml.
- **m06c_temporal_projection.py**: CPU-only PCA on (normal-shuffled) V-JEPA embeddings → project out temporal-interference subspace. 30-min experiment, potential paper centerpiece.
- **m07_umap.py**: cuML GPU UMAP.
- **m08_plot.py**: CPU-only matplotlib. Reads pre-computed .npy files.
- **m08b_compare.py**: CPU-only. Auto-scans m06/m06b JSON. Hard fail on missing temporal scores.

### m09a / m09b / m09c: Training (post-#49 split, 2026-04-15)
**Why split**: m09_pretrain.py monolith (2164 lines, 3 entangled techniques via `--explora`/`--surgery` flags) risked silent regression in unrelated technique during iteration. User-directed split into 3 physically isolated files, each with its own full training loop sharing only `src/utils/training.py` primitives.

- **m09a_pretrain.py** (1176 lines): Ch10 continual pretraining — full-param or layer-frozen V-JEPA 2 student-teacher JEPA with EMA, L1 latent prediction, drift control (λ·‖θ-θ₀‖²), lambda ablation sweep. Epoch-based. AdaptiveBatchSizer + `_train_step_grad_accum` (#48). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at main() (#53). Per-lambda encoder output names.
- **m09b_explora.py** (1049 lines): ExPLoRA — LoRA rank=16 α=32 injected on blocks 2-47 + unfreeze blocks 0-1 + LayerNorm. No drift control (LoRA params auto-excluded from `init_params`). Same JEPA loss. Hardcoded ExPLoRA mode (no --explora flag). Within-step OOM retry loop (#55).
- **m09c_surgery.py** (685 lines): **PAPER NOVELTY** — 3-stage progressive prefix unfreezing on FROZEN V-JEPA 2.1 with factor datasets. `FactorSampler(factor_manifest.json)` samples per-stage `mode_mixture` weights. `set_trainable_prefix(n_layers)` freezes all → unfreezes blocks [0, n_layers) + norm layers → rebuilds optimizer. Stages from `ch11_surgery.yaml`: stage1 (25% depth, 100% D_L), stage2 (50%, 90% D_A + 10% D_L), stage3 (75%, 85% D_I + 10% D_A + 5% D_L). Hardcoded surgery mode (no --surgery flag). Inter-stage cleanup: `optimizer=scheduler=sampler=None; gc.collect(); empty_cache(); ipc_collect()` (#58). Mode-gated memory savers: `use_8bit_optim` / `gradient_checkpointing` / `paged_optim` (sanity=true / poc=false / full=false). Per-stage warmup via `warmup_pct: 0.20` fraction (#61 — replaced fixed `warmup_steps: 200` which exceeded POC's 99 stage-steps → LR never reached target). Within-step retry + 0-step fail-hard (#55). Per-stage pre-init of loss vars (#54). **iter9 2-stage recipe** (2026-04-19): Stage 3 dropped post-1K-POC diagnosis (BWT=−0.33); Stage 2 replay 10%→30% (CLEAR rec). **iter9 streaming DataLoader**: when `cfg["factor_streaming"]["enabled"]`, per-stage builds `StreamingFactorDataset(mp4_index, mask_index, mode_mixture, ...)` + `DataLoader(num_workers, persistent_workers, prefetch_factor, pin_memory, worker_init_fn=_streaming_worker_init)`; per-step `next(stream_iter)["tensor"]` replaces legacy `sampler.sample() + load_factor_clip`. CLI `--factor-streaming` / `--no-factor-streaming` overrides yaml mode gate. **Status 2026-04-19**: streaming refactor landed + Tier-1 bitwise parity 10/10 PASSED + Tier-2 integration 4/4 PASSED + Tier-3 wall-time 4.77h projection ✅ under 10h budget. 10K FULL run is next GPU spend.

### m10 / m11: Surgery Factor Datasets (Ch11, validated 2026-04-15 on 100 dense clips)
- **m10_sam_segment.py** (996 lines): Grounded-SAM pipeline — Grounding DINO-base (Swin-B, fp32 per #37) does open-vocab text→box on 4 re-seed anchor frames `[0,4,8,12]` (multi-anchor per #32) → HF `Sam3TrackerVideoModel` (transformers 5.5.4 per #36) does box→mask refinement + propagation within each anchor's 4-frame segment. API calls: `init_video_session(video=frames_np)` per clip; per anchor × category `add_inputs_to_inference_session(frame_idx=anchor, obj_ids=[...], input_boxes=[boxes_xyxy])` (depth-3 boxes per #38), `session.reset_tracking_data()` between categories (not processor per #39), `propagate_in_video_iterator(start_frame_idx=anchor, max_frame_num_to_track=segment_size-1, reverse=False)`, `post_process_masks([output.pred_masks], original_sizes=[(H,W)], binarize=True)`, mask confidence from `sigmoid(output.object_score_logits)` (not iou_scores per #40). Box clamp before xywh-normalize (#28). Guards: empty `add_prompt` out_obj_ids (#30), `RuntimeError "No points are provided"` (#31). Saves per-clip `.npz` (agent_mask, layout_mask, centroids_json, interactions_json, mid_frame_rgb) + `segments.json` + `summary.json` + `per_object_bboxes_json` (~5KB/clip for m11 bbox-adaptive tubes). 17-category fixed agent taxonomy from `ch11_surgery.yaml` (replaces per-clip VLM `notable_objects`). `load_dotenv()` at top (#21). `os._exit(0)` on success / `os._exit(1)` on crash (#14/#16). Composite quality gate (4 checks). Measured on dense100: 11.02 s/clip (4.21× faster than raw sam3), 6146 agents, 8723 interactions.
- **m10_sam_segment_legacy.py** (742 lines): Pre-2026-04-15 raw `sam3` package path. Kept as fallback reference.
- **m11_factor_datasets.py** (853 lines): CPU-only. D_L (feathered Gaussian σ=15 blur on agents + feather σ=3), D_A (soft matte BG x0.1 residual), D_I (bbox-adaptive tubes via `make_interaction_tubes_from_bboxes` reading m10's `per_object_bboxes_json`; fallback to fixed centroid-30% crops for legacy `.npz`). Temporal interpolation for mask-frame/video-frame mismatches: `np.linspace(0, T_mask-1, T_vid)` (#17). Spatial resize if shapes differ. Quality filters: `min_agent_area_pct=0.003` (tuned for Grounded-SAM tight masks per #26), `max_agent_area_pct=0.70`. Saves per-clip `.npy` + `_tube{i}.npy` + `factor_manifest.json`. Plots: `m11_factor_samples.png` (D_L|D_A grid), `m11_interaction_samples.png`, `m11_per_Videoclip_verify/*.mp4` (20 top videos 2×2 grids 960×540 @ 6fps), `m11_factor_stats.png`. Measured on dense100: 91/100 clips have D_I, 8723 tubes, 5659 unique bbox shapes, median 65 tubes/clip. **iter9 `--streaming` flag** (2026-04-19): skips MP4 decode + scipy blur + `np.save` for non-verify clips (only the 100 `select_verify_clips(seed=42)` curated clips get full processing for `plot_factor_per_clip`). Manifest `has_D_L`/`has_D_A` still computed from `seg_entry["agent_pixel_ratio"]` thresholds. Net: ~90% m11 wall-time + ~340 GB disk saved at 10K. D_L/D_A generated on-demand at training time via `utils/factor_streaming.py::stream_factor` (bitwise parity with legacy `np.save` output — `scripts/tests_streaming/test_parity.py` 10/10 PASSED).
- **utils/factor_streaming.py** (157 lines, iter9): streaming factor-gen primitives. `stream_factor(mp4_bytes, mask_npz_path, factor_type, factor_cfg, num_frames, tmp_dir, clip_key) → (T,H,W,C) uint8` — imports `make_layout_only`/`make_agent_only` from m11 (no duplication); mirrors m11:652-664 mask align (temporal linspace + PIL NEAREST). `tensor_from_factor_array(frames_uint8, num_frames, crop_size) → (T,C,H,W) float32 ImageNet-normalized` — shared normalization between legacy `load_factor_clip` (disk) and streaming (memory) → guarantees bitwise parity. `_align_mask(mask, target_T, target_hw)` private helper.
- **utils/training.py::StreamingFactorDataset** (iter9, +~215 LoC): `IterableDataset` subclass for `DataLoader`. Per-worker sharding `clip_keys[worker_id::num_workers]`, per-worker `np.random.default_rng` seeded by `(base_seed * 1_000_003) ^ (worker_id * 2_718_281)`, per-worker TAR cache (fork-safe — each worker opens its own `tarfile.open`). Yields `{"tensor", "factor_type", "clip_key"}` dicts (default collate stacks tensors, lists strings). D_I falls through to legacy `di_legacy_index`. Paired: `build_streaming_indices(manifest_path, masks_dir, local_data) → (mp4_index, mask_index, manifest)` one-shot TAR scan in main process. `_streaming_worker_init(worker_id)` caps `torch.set_num_threads(1)` + PyAV log level. See `iter/iter9/plan_code_dev.md` for architecture + Tier-1/2/3 test results.

### Scripts
- **scripts/train_pretrain.sh**: Ch10 brute-force baseline (4-lambda ablation → winner → 5-epoch deep). Routes to `m09a_pretrain.py` post-#49.
- **scripts/train_explora.sh**: ExPLoRA pipeline. Steps [0]-[3]: m05 frozen → m06 frozen → m09b (no `--explora` flag, hardcoded) → m05 re-embed `vjepa_2_1_explora` → m06 metrics. AdaptiveBatchSizer from profiler `outputs/profile/training/profile_data.json`; YAML fallback. Checkpoint preserved on INT/TERM (trap). Sources `lib/common.sh`.
- **scripts/train_surgery.sh**: Ch11 surgery pipeline. Steps [0]-[4]: m10 Grounded-SAM → m11 factor datasets → m09c (no `--surgery` flag, hardcoded) `--factor-dir` → m05 re-embed `vjepa_2_1_surgical` → m06 metrics.
- **scripts/run_eval.sh**: Standalone eval (reusable across Ch9/Ch10/Ch11). Auto-detects encoders from `embeddings*.npy`. Delegates to `eval_suite.py` (m06→m06b→m07→m08→m08b).
- **scripts/run_embed.sh**: Standalone embedding extraction. Glob covers both legacy (m09_pretrain) and post-#49 (m09a/b/c) student_encoder.pt paths.
- **scripts/prep_data.sh**: Ch9 data pipeline (m04 tags + m04d motion). Pre-flight GPU checks, optional vLLM backend.
- **scripts/migrate_m09_outputs.sh**: Phase 3 of #49 — idempotent `mv` migration of legacy `outputs/.../m09_pretrain/{ablation,lambda*,explora,surgery}` into `m09a_pretrain/` / `m09b_explora/` / `m09c_surgery/`.
- **scripts/lib/common.sh**: Shared infra — `log()`, `banner()`, `run_step(num, name, est_time, log_file, cmd...)` (PIPESTATUS capture, auto HF upload on success), `bg_upload()`, `verify()`, `start_watchdog()`/`stop_watchdog()`, `finalize(pipeline_name)`.

### Root-level Scripts
- **setup_env_uv.sh**: UV-based env setup. `--mac` (CPU/lint), `--gpu --from-wheels` (10-step install from prebuilt sm_120 wheels release `sm120-cu128-py312`): PyTorch 2.12+cu128 → verify → requirements_gpu → FA2 → FAISS-GPU (source-built sm_120) → cuML → wandb → SAM 3.1 (`--no-deps` to preserve numpy>=2.3 per #5) → Grounding DINO pre-cache → `facebook/sam3` HF (~12 GB parallel HF_TRANSFER=1). Auto-downloads V-JEPA 2.1 ckpt (~28 GB) via `aria2c -x 16 -s 16` (#7b).
- **git_pull.sh**: `git fetch + reset --hard origin/main + clean -fd` (preserves .gitignored data). `--code-only` skips HF. Else `hf_outputs.py download outputs` + `download-data`.
- **git_push.sh**: User-invoked (Claude blocked from git state by hook). `--code-only` skips HF upload.
- **build_faiss_sm120.sh**: Source-build FAISS-GPU 1.14.1 for Blackwell sm_120 (pip wheels only sm_70+sm_80). Injects .so files into pip wheel, rewrites WHEEL + RECORD. `--install` reuses `/tmp/faiss_build`.

### HF Sync
- **src/utils/hf_outputs.py**: HF Hub sync to `anonymousML123/factorjepa-outputs` (public+gated). `_mirror_cleanup()` deletes stale remote files, `_stale_checkpoint_ignores()` skips ckpts modified <120s ago. `_UPLOAD_EXTENSIONS = {npy, npz, json, csv, png, pdf, tex, pt}`. `HF_HUB_ENABLE_HF_TRANSFER=1`. Has `sys.path.insert` for CLI use (#3).

### Utils
- **utils/config.py** (792 lines): Paths, constants, `load_merged_config(model_yaml, train_yaml)` deep-merge (handles `extends: base_optimization.yaml`), `get_pipeline_config()` cached reader, `get_sanity_clip_limit(module)`, `get_total_clips(local_data, subset_file)`. ENCODER_REGISTRY with dynamic lambda fallback.
- **utils/training.py** (787 lines, NEW #49): 20 technique-agnostic helpers shared by m09a/b/c. `build_model`, `build_optimizer` (reads `use_8bit_optim` → `bnb.optim.AdamW8bit` or `PagedAdamW8bit` per `paged_optim`), `build_mask_generators` (reads `cfg["model"]["crop_size/patch_size/tubelet_size"]` per #51), `enable_gradient_checkpointing` (block-wise `torch.utils.checkpoint(use_reentrant=False)`), `_train_step_grad_accum` (#48), `compute_drift_loss`, `update_teacher_ema`, `export_student_for_eval`. Contract: ZERO `if args.explora`/`if cfg["technique"]` branches.
- **utils/vjepa2_imports.py**: Import shim for vjepa2 modules. CWD-based isolation to avoid `src/utils/` namespace collision. `_ensure_loaded_2_1()` finally-block restores ALL saved `src.*` modules (not just `src` + `src.utils`) per #50. Exposes `get_vit_by_arch`, `get_vit_predictor`, `get_mask_generator`, `get_apply_masks`.
- **utils/bootstrap.py**: BCa bootstrap 95% CI via scipy.stats.bootstrap (10K iter).
- **utils/gpu_batch.py**: `compute_batch_sizes(vram)`, `AdaptiveBatchSizer` class (grow to `memory_cap`, halve on OOM, cooldown).
- **utils/hf_utils.py**: HF auth, upload helpers.
- **utils/wandb_utils.py**: All functions no-op when run=None.
- **utils/export_metadata.py**: tags.json → per-directory metadata.jsonl.
- **utils/output_guard.py**: `verify_or_skip(dir, min_clips=N)` returns bool (caller must `if`). JSON branch now sets `clip_count = len(data)` for completeness check (#29). `preflight_pipeline()` for shell scripts — interactive confirm before GPU work.
- **utils/video_io.py**: `_USE_TORCHCODEC = False` (#10 SIGSEGV on Blackwell sm_120); PyAV fallback is the active path.
- **utils/data_download.py**: `iter_clips_parallel(local_dir, processed_keys=...)` returns `(queue, stop_event, reader_thread)`. Resume-safe.

### Config Files
- **configs/pipeline.yaml**: Single source of truth. SANITY per-module clip limits, streaming params (prefetch queues 8, decode workers 8-16), GPU defaults, profiled batch sizes for RTX Pro 6000 96GB, **universal `gpu_memory_target: 0.85`** + per-module `*_initial_bs`, eval params (FAISS K=6, temporal pairs 100K), ENCODER_REGISTRY, VLM IDs, verify thresholds.
- **configs/model/vjepa2_0.yaml**: Legacy V-JEPA 2.0 ViT-g (1B, 1408-dim, 40 blocks, pred_depth=12, masked-only L1).
- **configs/model/vjepa2_1.yaml**: **PRIMARY** V-JEPA 2.1 ViT-G (2B, 1664-dim, 48 blocks, 26 heads, pred_depth=24, `predict_all=true`, `weight_distance_loss=true`, `n_output_distillation=4`). `hf_model_id: null` → `checkpoint_path: checkpoints/vjepa2_1_vitG_384.pt`.
- **configs/train/base_optimization.yaml**: Shared AdamW betas, EMA τ=0.99925, warmup 500 capped at 10%, LR schedule constant, grad_clip=10.0, bfloat16 mixed precision. `use_8bit_optim=false` + `gradient_checkpointing=false` defaults (SAFE for m09a/b).
- **configs/train/ch10_pretrain.yaml**: Ch10 recipe. lr=1e-6, 5 epochs + cooldown, freeze_below=20, λ ∈ [10, 100, 1000] (post-audit), EWC + VICReg.
- **configs/train/explora.yaml**: ExPLoRA recipe. lr=1e-5, unfreeze blocks 0-1 + LayerNorm, LoRA rank=16 α=32 on `qkv + proj`, `use_peft: true`, drift disabled.
- **configs/train/ch11_surgery.yaml**: Surgery recipe. lr=1e-6 constant, layer_wise_lr_decay=0.75, 3 stages `unfreeze_below ∈ {0.25, 0.50, 0.75}`. Mode-gated `use_8bit_optim`/`gradient_checkpointing`/`paged_optim` (sanity=true, poc=false, full=false). `factor_datasets`: Grounded-SAM (DINO box=0.15, text=0.12), 17-cat agent taxonomy, `min_agent_area_pct=0.003`. D_I mining: `min_overlap_frames=4`, `max_distance_frame_fraction=0.20`. Light L2 drift λ=1.0.
- **configs/YT_videos_raw.json**: 714 YouTube video IDs + titles + URLs across Delhi/Mumbai/Bangalore/Goa drive & walk tours.
- **configs/tag_taxonomy.json**: VLM tag taxonomy v3 (scene_type 13 values, crowd/traffic density, traffic_mix, 11 notable_objects, road_encroachment, video_quality).

### Data Files (`data/`)
- **sanity_100_dense.json**: 100-clip density-scored subset (73 tier1 + 26 tier2 + 1 goa), traffic+crowd+agent scored. Used across Steps A-E so comparisons are apples-to-apples.
- **subset_10k.json**: 10K video-level uniform subset, seed=42, from m00c.
- **val_1k.json**: 1K validation subset, seed=99.
- **val_1k_local/{manifest.json, tags.json}**: WebDataset manifest (1 TAR) + per-clip VLM tags.
- **subset_10k_local/{manifest.json, tags.json}**: WebDataset manifest (10 TARs) + tags.
- **full_local/manifest.json**: Full 115K corpus manifest (116 TARs).

## ENCODER_REGISTRY (live in `configs/pipeline.yaml`)
| Encoder | Dim | Suffix | Type |
|---------|----:|--------|------|
| vjepa (legacy 2.0) | 1408 | "" | video |
| vjepa_2_0_frozen | 1408 | _vjepa_2_0_frozen | video |
| random | 1408 | _random | synthetic |
| dinov2 | 1536 | _dinov2 | image (middle frame) |
| clip | 768 | _clip | image (middle frame) |
| vjepa_shuffled | 1408 | _vjepa_shuffled | video (shuffled) |
| vjepa_adapted | 1408 | _vjepa_adapted | video (Ch10 adapted) |
| vjepa_2_1_frozen | 1664 | _vjepa_2_1_frozen | video |
| vjepa_2_1_frozen_shuffled | 1664 | _vjepa_2_1_frozen_shuffled | video (shuffled) |
| vjepa_2_1_explora | 1664 | _vjepa_2_1_explora | video (adapted) |
| vjepa_2_1_surgical | 1664 | _vjepa_2_1_surgical | video (adapted) |

## Cross-Module Dependencies
```
Ch9 Eval (prep_data.sh + run_embed.sh + run_eval.sh):
m00d → m04 → m04d → m05 → m05b → m05c → m06 → m06b → m07 → m08 → m08b

Ch10 brute-force (train_pretrain.sh): m09a train (λ-ablation) → m05 re-embed → m06 metrics
ExPLoRA  (train_explora.sh):          m05 frozen → m09b → m05 re-embed → m06 metrics
Surgery  (train_surgery.sh):          m10 → m11 → m09c → m05 re-embed → m06 metrics
```

## Current Status (updated 2026-04-17)
- **Ch9: COMPLETE** — 5-encoder comparison on 10K POC. Baseline: Prec@K=36.1% (frozen V-JEPA 2.0). Key finding: shuffled > normal V-JEPA by 2.4× → temporal interference.
- **Ch10 (115K FULL): CATASTROPHIC FORGETTING** (2026-04-05). λ=0.001 → Prec@K crashed 36.1% → 14.3%. Drift penalty 1000× smaller than JEPA loss. Gold-standard audit found 12 discrepancies. Demoted to comparison arm.
- **Strategic pivot (2026-04-10)**: Ch11 runs directly on frozen V-JEPA 2.1 (no Ch10 prerequisite). V-JEPA 2.1 ViT-G = PRIMARY. Temporal interference projection (30 min CPU) = potential paper centerpiece. Idea Critic verdict: PURSUE.
- **Ch11 upstream (m10 + m11 + m05 frozen) VALIDATED on 96GB 2026-04-17**:
  - Step A (m10 Grounded-SAM): **6.13 s/clip on 96GB** (11.02 s/clip on 24GB), 6141 agents, 8712 interactions, quality_gate PASS.
  - Step B (m11 bbox-tubes, 32-worker ProcessPool): 47 s total on 100 dense clips (5.7× speedup vs single-thread). 91/100 clips with D_I, 8712 tubes, median 65 tubes/clip.
  - Step C (m05 V-JEPA 2.1 frozen embed): 100 clips × 1664-dim in **423 s on 96GB** (4.23 s/clip bf16 with torch.compile + AdaptiveBatchSizer). RoPE Q/K cast durable via setup_env_uv.sh heredoc (#44/#59).
  - 115K FULL ETA: ~8.2 d single-stream m10 / ~2 d at batch ×4 on 96GB.
- **Ch11 m09c Surgery SANITY: ✅ RESOLVED on 96GB 2026-04-17** — all 3 stages PASS (loss=0.4870/0.4901/**0.4806** — first ever successful Stage 3 completion). Stage 3 post-cleanup VRAM 19.9 / 102 GB = 80 GB headroom. v8 teacher-offload plan NOT needed — 96GB hardware migration resolved #58 for free. PagedAdamW8bit + grad-checkpointing kept mode-gated to sanity for 24GB backward-compat.
- **Ch11 m09c Surgery POC (100-dense) debugged + retired 2026-04-17**:
  - v1: `max_epochs.poc: 1` → 3 total steps across 3 stages (silent near-no-op, #60). Fixed → 100.
  - v2: `warmup_steps: 200 > stage_steps: 99` per stage → LR never reached target, loss 0.503→0.476 warmup-truncated (#61). Fixed → `warmup_pct: 0.20` auto-scaling.
  - 🚚 Tier retired — 3200 visits/clip at 100 scale produces overfitting pressure that wouldn't replicate at FULL scale. Unpublishable.
- **🎯 NEXT (Phase 2b): 1K val_1k POC**. Full pipeline ~10 h on 96GB (m10 ~102 min + m11 ~8 min + m05 frozen ~70 min + m09c Surgery ~2.7 h + m05 surgical ~70 min + m06 ~5 min + m09b ExPLoRA arm ~2.75 h). 20 visits/clip = publishable scale. Decision gate: Surgery > Frozen with non-overlapping 95% CIs → scale to FULL 115K; else follow `literature_survey.md` fallback ladder.

## Env Stack (pinned, 2026-04-15)
PyTorch 2.12.0.dev20260228+cu128, CUDA 12.8, FA2 2.8.3, FAISS-GPU 1.14.1 (source-built sm_120), cuML 26.04, SAM 3.1 (raw pkg via `--no-deps`), transformers **5.5.4** (`Sam3TrackerVideoModel`), bitsandbytes 0.49.2, Python 3.12, UV. Release tag: `sm120-cu128-py312`.

## NeurIPS 2026-05-04 Deadline
- Budget: ~27h remaining (of original ~38 h, ~11 h spent on Phase 1 SANITY + Phase 2a 100-dense debug).
- Phase 1 (24GB→96GB SANITY): ✅ DONE.
- Phase 2a (100-dense POC discovery tier): ✅ DONE + 🚚 retired. Caught #60 max_epochs + #61 warmup_pct.
- Phase 2b (1K val_1k POC, publishable tier): 🎯 NEXT — ~10 h full pipeline D.2 → E.3.
- Phase 3 (FULL 115K + ablations): ⬜ ~36 h on 96GB batch×4 if 1K Prec@K shows Surgery > Frozen cleanly.
- Decision gate: Surgery > ExPLoRA > Frozen on Prec@K with non-overlapping 95% CIs on 1K val_1k clips (moved up from 100-dense).
- Fallback: `iter/utils/literarure_survey.md` — 24 JEPA variants, 3 top techniques (SIGReg, VLA-JEPA leakage-free, temporal straightening).
- Best-paper reframe: "Temporal Interference in Video Foundation Models" — shuffled > normal by 2.4× as paper centerpiece.

## Data Download Times (measured, April 2026)
| Command | What it does | Time |
|---------|-------------|:----:|
| `m00d --FULL --no-wandb` | Downloads ALL 116 TARs, keeps all 115K clips | 24 min |
| `m00d --FULL --subset data/subset_10k.json` | Downloads ALL 116 TARs, filters to 10K | ~50 min |
| `rsync data/ from Mac` | Transfers pre-filtered 10K + val 1K | ~17 min |
| `hf_outputs.py download-data` | Downloads poc 10K + val 1K from factorjepa-outputs | ~3 min |

**Key insight**: m00d always downloads ALL 116 TARs regardless of subset. rsync or hf_outputs download-data is 10-25× faster for POC/val.

## Lessons Learned (selected — full history in iter/iter8/errors_N_fixes.md #1-#61)
1. **vjepa2 namespace collision**: `src/utils/__init__.py` shadows vjepa2's `src/utils/`. Fixed with CWD-based import shim + `_ensure_loaded_2_1()` finally-block restores ALL saved `src.*` modules (#50).
2. **SAM3 `--no-deps` undeclared dependencies**: Every runtime import must be explicitly declared in `requirements_gpu.txt` (pycocotools, einops, iopath, ftfy). Pattern: #5, #6, #7, #18, #19.
3. **transformers 4.57 → 5.5.4 migration**: `torch_dtype=` → `dtype=` (#37/#43), DINO text branch crashes under fp16 → load fp32 (#37), `box_threshold` → `threshold` + `labels` → `text_labels` (#24), `Sam3TrackerVideoProcessor.add_inputs_to_inference_session` wants depth-3 boxes not depth-4 (#38), session-reset methods on session not processor (#39), `output.object_score_logits` not `iou_scores` (#40).
4. **SAM 3.1 multiplex tracking**: needs `text + boxes` hybrid — boxes-only loses tracking (#27). `max_frame_num_to_track` unusable on raw sam3 pkg (#33/#35). Multi-anchor DINO re-seed every 4 frames caps drift (#32). HF `Sam3TrackerVideoModel` unlocks 4.21× speedup and works correctly (#36).
5. **V-JEPA 2.1 torch.compile + bf16**: RoPE Q/K emerge in fp32 but V stays fp16 → SDPA crashes under inductor. Cast Q, K to V.dtype before SDPA (`q = q.to(v.dtype); k = k.to(v.dtype)`); use bf16 over fp16 for wider dynamic range. Zero accuracy impact (#44/#45).
6. **m05 `is_adapted` branch selector**: `_resolve_model(None)` helper routes `--model=None + hf_model_id=null` to native-frozen branch which unpacks `target_encoder`/`encoder` keys. Never set `args.model = VJEPA_CHECKPOINT_PATH` (triggers wrong adapted-student branch) (#42).
7. **m09 monolith split (#49)**: Zero cross-technique contamination. m09a/b/c each own full training loop; utils/training.py shared and technique-agnostic (no `if cfg["technique"]` branches).
8. **AdaptiveBatchSizer is universal infra**: Wire into every GPU forward loop (m04/m05/m05b/m05c/m09). `memory_cap=pipeline.yaml gpu_memory_target` (universal 0.85), `initial_size=per-module *_initial_bs` (#46/#47).
9. **Gradient accumulation preserves research integrity**: Effective BS must stay = `cfg.optimization.batch_size` so optimizer dynamics are bit-identical to a static-BS run. Micro-batch sub-size is adaptive; scale each micro's loss by `(micro/macro)` (#48).
10. **m09c Stage 3 memory on 24GB**: fp32 master weights + 8-bit m1/m2 + CUDA context overshoots. Fixes compound: inter-stage optimizer cleanup (None-ref + empty_cache + ipc_collect) + bnb.PagedAdamW8bit (CPU-paged) + gradient_checkpointing + (v8 planned) teacher CPU offload (#56/#57/#58).
11. **Silent failures are garbage metrics**: Within-step retry loop on OOM (shrink sub-batch, retry SAME macro batch, raise if at min_size); post-loop fail-hard if 0 successful steps (refuse to export misleading checkpoint) (#55).
12. **verify_or_skip completeness**: Must check output count not just existence. Both JSON and .npy branches set `clip_count = len(data)` (#29).
13. **torchcodec SIGSEGV on Blackwell**: `_USE_TORCHCODEC = False` in `video_io.py`; PyAV fallback is active. Silent C-extension crashes bypass try/except (#10).
14. **SAM3 async thread shutdown**: `os._exit(0)` on success + `os._exit(1)` on crash — `return` leaks async frame-loading threads that hold VRAM (#14/#16).
15. **vjepa2 patches must re-apply on fresh clone (#59)**: `setup_env_uv.sh` does `rm -rf deps/vjepa2 && git clone` on every provision — any local edit to `deps/vjepa2/` is wiped. The RoPE Q/K dtype cast (#44) must live as an idempotent heredoc in `setup_env_uv.sh` (anchor-match + SystemExit if anchor missing), not as a one-off file edit.
16. **YAML training-config sanity is load-bearing (#60, #61)**: `max_epochs.poc: 1` silently ran only 3 total optimizer steps; `warmup_steps: 200 > stage_steps: 99` silently capped LR at <50% of target. Both produced exported students that looked OK downstream but carried no training signal. Fail-hard preflight checks (B38 max_epochs sanity, B39 warmup_pct) now guard these at CPU-time before GPU spend.
17. **Hardware upgrade can beat code complexity (#58 postscript)**: v8 teacher-CPU-offload patch (~3 h code + test + maintain) was superseded by moving SANITY from 24GB RTX Pro 4000 → 96GB RTX Pro 6000 Blackwell (~$0.60/hr delta, cents per SANITY run). Compare "code fix" vs "hardware upgrade" as equal options, not hardware as last resort.

## User Preferences
- Never be a yes-man — give pros/cons like a Sr. AI/ML Research Engineer.
- Be brutally honest. Disagree when wrong, never hallucinate.
- Git: provide commit message text only, never run git commands (enforced by hook).
- GPU time is expensive — keep GPU ≥85% busy, no idle waste.
- When auditing, SHOW grep output as proof — user does not trust "I checked" claims.
- No hardcoded values in Python — YAML configs or runtime discovery only.
- Fail hard in research — silent errors cost paper rejections.
- WEBSEARCH before recommending any fix that trades off accuracy or throughput; cite ≥2 sources.
