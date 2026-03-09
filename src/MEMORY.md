# WalkIndia-200K Project Memory

## Snapshot Sync Rule
When you update this file OR `src/CLAUDE.md`, copy the updated version to its snapshot:
- This file → `src/MEMORY.md`
- `src/CLAUDE.md` → already in-repo, no extra copy needed
Both snapshots are version-tracked in git. Keep them in sync.

## Project Overview
Research benchmark testing if V-JEPA 2 (Meta's video foundation model, trained on Western data) transfers to Indian street scenes. Pipeline: YouTube videos → scene-split clips → WebDataset shards (HF) → VLM tagging → V-JEPA embeddings → FAISS metrics → UMAP → plots.

## Pipeline Modules

### m00-m03: Data Pipeline (CPU, completed)
- **m00_data_prep.py**: Parse YT_videos_raw.md → JSON, word freq, city matrix
- **m00b_fetch_durations.py**: yt-dlp metadata fetch (no download)
- **m00c_sample_subset.py**: Video-level uniform 10K subset → data/subset_10k.json (seed=42)
- **m00d_download_subset.py**: Pre-download subset to local WebDataset TARs (~11 min, ~10.7 GB). CPU-only. Fixes producer starvation (8.4%→100% hit rate). Output: `data/subset_10k_local/`
- **m01_download.py**: Download 714 videos at 480p via yt-dlp + aria2c
- **m02_scene_detect.py**: Greedy scene-aware split [4-10s] + CRF28 encode
- **m02b_scene_fetch_duration.py**: Scan clips → clip_durations.json
- **m03_pack_shards.py**: Pack clips → WebDataset TARs → stream-upload to HF

### m04-m08b: GPU Pipeline
- **m04_vlm_tag.py**: 3 VLM backends (Qwen/VideoLLaMA3/LLaVA). Checkpoint every 500. `--local-data` support.
- **m04b_vlm_select.py**: CPU-only. 5-criterion weighted bake-off comparison.
- **m04c_sanity_compare.py**: CPU-only. Reads 3 sanity JSONs, computes 4 metrics, 2x2 dashboard.
- **m05_vjepa_embed.py**: V-JEPA 2 ViT-G (1B, frozen, fp16, FA2, torch.compile). Producer-consumer. `--local-data` support.
- **m05b_baselines.py**: 4 baselines — `--encoder random|dinov2|clip|vjepa_shuffled|all`. `--local-data` support.
- **m05c_true_overlap.py**: Augmented V-JEPA embeddings (BYOL/DINO multi-crop). `--local-data` support.
- **m06_faiss_metrics.py**: FAISS-GPU kNN → 9 metrics Easy/Hard. `--encoder` flag (dim-agnostic).
- **m07_umap.py**: cuML GPU UMAP (1408→2D). `--encoder` flag.
- **m08_plot.py**: CPU-only matplotlib. UMAP scatter, confusion matrix, kNN grid.
- **m08b_compare.py**: CPU-only. Multi-encoder comparison: bar chart, radar, LaTeX table.

## Critical Constants (config.py)
- HF_DATASET_REPO = "anonymousML123/walkindia-200k"
- VJEPA: facebook/vjepa2-vitg-fpc64-384, 64 frames, 1408-dim embeddings
- VLMs: qwen (winner, 0.919), videollama, llava — all transformers sequential
- ENCODER_REGISTRY: vjepa, random, dinov2, clip, vjepa_shuffled (suffix-based file paths)
- FAISS_K_NEIGHBORS = 6, BAKEOFF_CLIP_COUNT = 2500
- POC subset: 10K clips (video-level uniform, seed=42)
- Taxonomy: v3 (`tag_taxonomy.json`) — 16 fields (13 single + 2 multi + 1 changelog)

## ENCODER_REGISTRY
| Encoder | Model | Dim | Suffix | Type |
|---------|-------|-----|--------|------|
| vjepa | vjepa2-vitg-fpc64-384 | 1408 | "" | video |
| random | — | 1408 | _random | synthetic |
| dinov2 | dinov2-vitl14 | 1024 | _dinov2 | image (middle frame) |
| clip | clip-vit-large-patch14 | 768 | _clip | image (middle frame) |
| vjepa_shuffled | vjepa2-vitg-fpc64-384 | 1408 | _vjepa_shuffled | video (shuffled) |

## Output Paths
- POC: src/outputs_poc/ (--subset data/subset_10k.json)
- Full: src/outputs/ (no --subset)
- Bakeoff: src/data/bakeoff/ (--BAKEOFF)

## Current Status (Ch9)
- **10K POC: COMPLETE** (Mar 9, 2026). 48 outputs verified, 0 errors. Pushed to GitHub.
- Pipeline: `run_ch9_overnight.sh --FULL` — 1h 36m total (Steps 1-3 cached, m05c 93min, m06-m08 2.5min)
- GPU env: VERIFIED on RTX PRO 6000 Blackwell (102GB). All components pass.
- Next: Analyze V-JEPA results (underperforms DINOv2/CLIP on Prec@K — needs investigation)

## Ch9 10K POC Results (5 encoders, Easy mode)

| Encoder | Prec@K | mAP@K | Cycle@K | nDCG@K | Overlap@K | Silhouette |
|---------|--------|-------|---------|--------|-----------|------------|
| **dinov2** | **50.5%** | **0.4271** | 66.8% | 0.9577 | 60.9% (dim) | -0.0574 |
| clip | 46.0% | 0.3816 | 65.2% | **0.9583** | 47.1% (dim) | -0.0470 |
| vjepa_shuffled | 35.3% | 0.2724 | **76.2%** | 0.9500 | 35.3% (dim) | -0.2245 |
| vjepa | 14.6% | 0.0792 | **78.7%** | 0.9032 | 10.5% (true) | -0.2503 |
| random | 12.2% | 0.0608 | 55.0% | 0.8978 | 0.0% (dim) | -0.0206 |

- V-JEPA wins Cycle@K (78.7%) but lags on Prec@K/mAP@K (14.6%/0.079 vs DINOv2 50.5%/0.427)
- Only V-JEPA uses true multi-crop Overlap@K (10.5%); others use dim-split approximation
- V-JEPA operates on 5,105 deduped clips; baselines on 10,000 (independent embedding spaces)

## Architecture Gotchas
- GPU scripts save .npy → CPU scripts read them (never duplicate GPU compute in plotting)
- embeddings.paths.npy stores clip keys (not local paths) — used for Hard mode ±30s exclusion
- Tags↔embeddings alignment via __key__ field
- FAISS IVFFlat (not IVF-PQ) — dim-agnostic (d = embeddings.shape[1])
- Encoder suffix system: vjepa="" (backward compat), others="_encodername"
- m05c reads embeddings.paths.npy (deduped keys) not subset_10k.json — ordering dependency: m05 must complete before m05c
- m05 dedup (cosine sim > 0.95): 10K → 5,105 unique clips. This is by design, NOT a stream failure

## Lessons Learned (Ch9)
- **Throughput metric `total/elapsed` is misleading with checkpoints**: includes checkpoint clips at t=0, so reported rate declines monotonically even when per-batch rate is steady. Fix: report `new_clips / elapsed` or per-batch rate.
- **ThreadPoolExecutor for CPU-bound PyTorch ops = disaster**: 8 workers × ~80 ATen internal threads = 640+ threads → OS scheduler thrash → 0% throughput. Keep threading for I/O-bound decode only. See Bug #4 in `iter/iter6/plan_batch_speedup.md`.
- **Vectorized augmentation**: Replace per-frame TF.resized_crop loop (64 calls) with single tensor slice + F.interpolate on (T,C,H,W) → 64x fewer kernel launches per clip.
- **Checkpoint key coverage ≠ target key coverage**: A checkpoint with N clips from streaming order does NOT guarantee it covers all M target keys (M < N). Keys are spread across the full stream.
- **GitHub file size**: .npy files > 50MB trigger warnings. At 115K scale, will exceed 100MB hard limit → need Git LFS.

## Batch Size Auto-Scaling (gpu_batch.py)
- Baseline: A100-40GB, scale = actual_vram / 40
- RTX PRO 4000 (24GB): vjepa=9, image_encoder=36, transformers=2
- RTX PRO 6000 (96GB): vjepa=38, image_encoder=152, transformers=9

## VLM Strategy
- 10K POC: Qwen3-VL-8B via transformers (validated, 0.919 score)
- 115K FULL: Qwen3.5-9B via vLLM (transformers video BROKEN — GitHub Issue #58)
- See `iter/iter6/vLLM_plan_Blackwell.md` for deployment plan

## Known Active Issue
- **Qwen3.5-9B video via transformers is BROKEN**: `StopIteration` bug in `get_rope_index` (GitHub Issue #58). vLLM is the ONLY working path. Impacts 115K FULL run only.

## User Preferences
- Never be a yes-man — give pros/cons like a Sr. AI/ML Research Engineer
- Be brutally honest. Disagree when wrong, never hallucinate
- Git: provide commit message text only, never run git commands (enforced by hook)
- GPU time is expensive — keep GPU busy, no idle waste

## Setup Scripts
- `setup_env_uv.sh --mac` (M1 CPU) / `--gpu` (Nvidia) / `--gpu --from-wheels` (prebuilt sm_120 wheels)
- `build_faiss_sm120.sh` — source-build FAISS for Blackwell sm_120
- `build_wheels_sm120.sh` — build FA2+FAISS wheels + upload to GitHub Release
- Prebuilt wheels: GitHub Release tag `sm120-cu128-py312` → FA2 2.8.3 + FAISS 1.14.1
- FAISS wheel fix (Mar 9): setup_env_uv.sh auto-installs libopenblas-dev + patchelf RPATH fix
