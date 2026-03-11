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
- **m04f_motion_features.py**: (TODO) CPU-only. RAFT/Farneback optical flow → mean magnitude, direction histogram, camera motion estimate per clip. Temporal ground-truth for retrieval evaluation.
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

## Current Status
- **Ch9: COMPLETE** (48 outputs, 0 errors, ~6h 35m clean on RTX PRO 6000)
  - 5-encoder comparison: DINOv2 (50.5% Prec@K) > CLIP (46.0%) > shuffled (35.3%) > vjepa (14.6%) > random (12.2%)
  - V-JEPA wins Cycle@K (78.7%) but lags on spatial retrieval accuracy
  - Key finding: shuffled > normal V-JEPA → temporal encoding HURTS spatial scene classification
  - Key finding: taxonomy measures ONLY spatial features, 0 temporal fields → evaluation gap
  - External validation: arXiv:2509.21595 confirms same DINOv3 > V-JEPA spatial tradeoff
- **Temporal eval extension: TODO** (pre-Ch10 requirement)
  - m04f: optical flow motion features (CPU, Approach A — recommended first)
  - VLM temporal tags (Approach B — supplementary, uncertain quality)
  - Expected: V-JEPA >> image baselines on temporal metrics (reversal of spatial result)
- **Ch10: NOT BUILT** (m09 — continual pretraining)
- **Ch11: NOT BUILT** (m10/m11/m12 — SAM + factors + surgery)

## Architecture Gotchas
- GPU scripts save .npy → CPU scripts read them (never duplicate GPU compute in plotting)
- embeddings.paths.npy stores clip keys (not local paths) — used for Hard mode ±30s exclusion
- Tags↔embeddings alignment via __key__ field
- FAISS IVFFlat (not IVF-PQ) — dim-agnostic (d = embeddings.shape[1])
- Encoder suffix system: vjepa="" (backward compat), others="_encodername"

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
