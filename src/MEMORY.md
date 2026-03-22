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
- **m00_data_prep.py** (~732 lines): Parse YT_videos_raw.md → JSON, word freq, city matrix. State machine parser handles tier1/tier2/monuments/goa sections.
  - `python -u src/m00_data_prep.py --SANITY 2>&1 | tee logs/m00_sanity.log`
  - `python -u src/m00_data_prep.py --FULL 2>&1 | tee logs/m00_full.log`
- **m00b_fetch_durations.py** (~398 lines): yt-dlp metadata fetch (no download). ThreadPoolExecutor(10 workers). 30s timeout per video.
  - `python -u src/m00b_fetch_durations.py --SANITY 2>&1 | tee logs/m00b_sanity.log`
  - `python -u src/m00b_fetch_durations.py --FULL 2>&1 | tee logs/m00b_full.log`
- **m00c_sample_subset.py** (~271 lines): Video-level uniform 10K subset → data/subset_10k.json (seed=42). 3-pass algorithm: floor allocation → shortfall redistribution → per-video sampling.
  - `python -u src/m00c_sample_subset.py --FULL 2>&1 | tee logs/m00c_full.log`
- **m00d_download_subset.py** (~333 lines): Pre-download subset to local WebDataset TARs. CPU-only. Uses hf_hub_download (CDN, not streaming). Resume via manifest.json tracking processed HF shards.
  - `python -u src/m00d_download_subset.py --SANITY --subset data/subset_10k.json 2>&1 | tee logs/m00d_sanity.log`
  - `python -u src/m00d_download_subset.py --subset data/subset_10k.json 2>&1 | tee logs/m00d_download.log`
- **m01_download.py** (~281 lines): Download 714 videos at 480p via yt-dlp + aria2c (16 parallel connections). Chrome cookies for auth.
  - `python -u src/m01_download.py --SANITY 2>&1 | tee logs/m01_sanity.log`
  - `python -u src/m01_download.py --FULL 2>&1 | tee logs/m01_full.log`
- **m02_scene_detect.py** (~355 lines): PySceneDetect ContentDetector(threshold=15) → greedy [4-10s] split → ffmpeg libx264 CRF28 encode. Hierarchical output: clips/section/video_id-NNN.mp4.
  - `python -u src/m02_scene_detect.py --SANITY 2>&1 | tee logs/m02_sanity.log`
  - `python -u src/m02_scene_detect.py --FULL 2>&1 | tee logs/m02_full.log`
- **m02b_scene_fetch_duration.py** (~398 lines): ffprobe scan clips → clip_durations.json. 8 workers parallel. Reports violations (>10s / <4s).
  - `python -u src/m02b_scene_fetch_duration.py --SANITY 2>&1 | tee logs/m02b_sanity.log`
  - `python -u src/m02b_scene_fetch_duration.py --FULL 2>&1 | tee logs/m02b_full.log`
- **m03_pack_shards.py** (~228 lines): Pack clips → WebDataset TARs (1000 clips/shard) → stream-upload to HF. Global indexing: {idx:06d}.mp4 + .json per clip.
  - `python -u src/m03_pack_shards.py --SANITY 2>&1 | tee logs/m03_sanity.log`
  - `python -u src/m03_pack_shards.py --FULL 2>&1 | tee logs/m03_full.log`

### m04-m08b: GPU Pipeline
- **m04_vlm_tag.py** (~1416 lines): 3 VLM backends (Qwen/VideoLLaMA3/LLaVA). Orchestrator/worker subprocess pattern (VRAM management). Producer-consumer with ThreadPoolExecutor decode. Checkpoint every 500 clips (`.tags_checkpoint.npz`, atomic save via os.replace). Resume via `processed_keys` set. Builds prompt from tag_taxonomy.json (16 fields). JSON parse fallback to dummy. OOM: halves batch via AdaptiveBatchSizer.
  - `python -u src/m04_vlm_tag.py --model qwen --SANITY 2>&1 | tee logs/m04_sanity_qwen.log`
  - `python -u src/m04_vlm_tag.py --model qwen --FULL --subset data/subset_10k.json --local-data data/subset_10k_local 2>&1 | tee logs/m04_full_qwen_poc.log`
- **m04b_vlm_select.py** (~649 lines): CPU-only. 5-criterion weighted bake-off (json_parse=0.30, agreement=0.25, speed=0.20, taxonomy=0.15, confidence_cal=0.10). Majority vote consensus.
  - `python -u src/m04b_vlm_select.py --SANITY 2>&1 | tee logs/m04b_sanity.log`
- **m04c_sanity_compare.py** (~263 lines): CPU-only. 4-metric table + 2x2 dashboard for 20-clip sanity comparison.
  - `python -u src/m04c_sanity_compare.py 2>&1 | tee logs/m04c_compare.log`
- **m04d_motion_features.py** (~456 lines): GPU-RAFT optical flow → 13D motion features (mean/std/max magnitude, 8-bin direction histogram, camera_motion_x/y). RAFT-Large ViT (C_T_SKHT_V2 weights). 16 frame pairs per clip. Checkpoint every 200 clips (`.m04d_checkpoint.npz`). Resume via `processed_keys`.
  - `python -u src/m04d_motion_features.py --SANITY --subset data/subset_10k.json --local-data data/subset_10k_local 2>&1 | tee logs/m04d_sanity.log`
  - `python -u src/m04d_motion_features.py --FULL --subset data/subset_10k.json --local-data data/subset_10k_local 2>&1 | tee logs/m04d_motion.log`
- **m05_vjepa_embed.py** (~699 lines): V-JEPA 2 ViT-G (1B, frozen, fp16, FA2, torch.compile). Orchestrator/worker subprocess pattern (ENGINE_RESTART_EVERY=10K clips). Producer-consumer: CPU thread decodes (torchcodec or PyAV fallback) + preprocesses → GPU forwards. Checkpoint every 500 clips (`.m05_checkpoint.npz`). Post-processing: cosine dedup (threshold=0.95, chunked GPU). Output: embeddings.npy + embeddings.paths.npy.
  - `python -u src/m05_vjepa_embed.py --SANITY 2>&1 | tee logs/m05_sanity.log`
  - `python -u src/m05_vjepa_embed.py --FULL --subset data/subset_10k.json --local-data data/subset_10k_local 2>&1 | tee logs/m05_vjepa_embed_poc.log`
- **m05b_baselines.py** (~733 lines): 4 baselines. Random (1408D, CPU, L2-normed uniform, seed=42). DINOv2 (ViT-L/14, 1024D, middle frame, FA2). CLIP (ViT-L/14, 768D, middle frame, SDPA). Shuffled V-JEPA (same model, permuted frames, seed=hash(key)). Per-encoder checkpoint (`.m05b_<enc>_checkpoint.npz`). Auto-skips if output exists.
  - `python -u src/m05b_baselines.py --encoder all --SANITY 2>&1 | tee logs/m05b_sanity.log`
  - `python -u src/m05b_baselines.py --encoder all --FULL --subset data/subset_10k.json --local-data data/subset_10k_local 2>&1 | tee logs/m05b_all_poc.log`
- **m05c_true_overlap.py** (~406 lines): Augmented V-JEPA embeddings for True Overlap@K. BYOL/DINO protocol: View A (large crop 0.4-1.0, color jitter), View B (small crop 0.2-0.6, Gaussian blur). Vectorized augmentation (F.interpolate once for all T frames). Reads deduped keys from m05's embeddings.paths.npy (~5K). Checkpoint: `.m05c_checkpoint.npz`.
  - `python -u src/m05c_true_overlap.py --SANITY 2>&1 | tee logs/m05c_sanity.log`
  - `python -u src/m05c_true_overlap.py --FULL --subset data/subset_10k.json --local-data data/subset_10k_local 2>&1 | tee logs/m05c_overlap_poc.log`
- **m06_faiss_metrics.py** (~1275 lines): FAISS-GPU kNN → 9 metrics Easy/Hard + bootstrap 95% CI. IndexFlatL2 on GPU. Hard mode: ±30s exclusion mask (same video). 9 metrics: Cycle@K, Overlap@K, Prec@K, mAP@K, nDCG@K, Silhouette, per-scene purity, confidence sweep (7 thresholds), macro/micro averages. `--true-overlap` loads augA/augB from m05c. No checkpoint (read-only post-processing).
  - `python -u src/m06_faiss_metrics.py --encoder vjepa --true-overlap --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_vjepa_poc.log`
- **m06b_temporal_corr.py** (~466 lines): CPU-only. 5 temporal metrics per encoder: (1) Spearman rho (embedding vs motion distance), (2) Temporal Prec@K (motion quartile matching), (3) Motion Retrieval mAP (KMeans 4 clusters), (4) Order Sensitivity (normal vs shuffled distance), (5) Temporal Locality (intra vs inter-video ratio). All with bootstrap 95% CI. Metrics 1-3 need m04d, 4-5 use existing embeddings only.
  - `python -u src/m06b_temporal_corr.py --encoder vjepa --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06b_vjepa.log`
- **m07_umap.py** (~99 lines): cuML GPU UMAP (N×D → N×2). No CPU fallback. Default: n_neighbors=15, min_dist=0.1.
  - `python -u src/m07_umap.py --encoder vjepa --FULL --subset data/subset_10k.json 2>&1 | tee logs/m07_umap_vjepa_poc.log`
- **m08_plot.py** (~592 lines): CPU-only matplotlib. Reads pre-computed .npy files. Plots: UMAP scatter, confusion matrix, kNN grid (ffmpeg frame extraction with placeholder fallback), UMAP grid (3×3), confusion grid (3×3). All with encoder suffix.
  - `python -u src/m08_plot.py --encoder vjepa --FULL --subset data/subset_10k.json 2>&1 | tee logs/m08_plot_vjepa.log`
- **m08b_compare.py** (~510 lines): CPU-only. Auto-scans m06_metrics_*.json + m06b_temporal_corr_*.json. Generates: grouped bar (Easy/Hard), radar (5 spatial metrics), spatial-temporal bar, tradeoff scatter, LaTeX table with ±CI.
  - `python -u src/m08b_compare.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m08b_compare.log`

### Utils
- **config.py** (~605 lines): Paths, constants, argparse helpers, load_subset(), get_output_dir(), check_gpu(), build_video_section_map(), setup_ram_cache(), get_deduplicated_clips(). ENCODER_REGISTRY with suffix system.
- **bootstrap.py** (~138 lines): BCa bootstrap 95% CI via scipy.stats.bootstrap (10K iterations). Per-clip score functions: per_clip_prec_at_k, per_clip_map_at_k, per_clip_cycle_at_k, per_clip_ndcg_at_k.
- **gpu_batch.py** (~167 lines): compute_batch_sizes(vram) → {"vjepa", "image_encoder", "transformers", "transformers_batch"}. AdaptiveBatchSizer class: VRAM monitoring, OOM halving, success adjustment.
- **hf_utils.py** (~243 lines): HF auth (_get_token from .env), generate_readme(), upload_readme(), upload_metadata().
- **wandb_utils.py** (~95 lines): add_wandb_args(), init_wandb(), log_metrics(), log_image(), log_artifact(), finish_wandb(). All no-op when run=None.
- **export_metadata.py** (~90 lines): Convert tags.json → per-directory metadata.jsonl for HF dataset viewer.

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
- SANITY: src/outputs_sanity/ (--SANITY, 5-20 clips)
- POC: src/outputs_poc/ (--FULL --subset data/subset_10k.json)
- Full: src/outputs/ (--FULL, no --subset)
- Bakeoff: src/data/bakeoff/ (--BAKEOFF)

## Cross-Module Dependencies (run_evaluate.sh order)
```
m00d (pre-download) → m04 (VLM tags) → m05 (V-JEPA embed) → m05b (baselines) → m05c (overlap) → m04d (motion)
                                                                                                      ↓
m06 (FAISS metrics, ×5 enc) → m06b (temporal, ×5 enc) → m07 (UMAP, ×5 enc) → m08 (plots, ×5 enc) → m08b (compare)
```
- m04 → m06/m08: tags required for metric computation
- m05 → m05b, m05c, m06, m07: embeddings required; m05c uses deduped keys from m05
- m04d → m06b: motion features required for temporal metrics 1-3
- m05c → m06 (optional): augmented embeddings enable True Overlap@K (--true-overlap flag)
- m06 + m06b → m08b: metrics JSON files required for comparison visualization

## Checkpoint Files (for resume/debugging)
| Module | Checkpoint | Resume Key | Interval |
|--------|-----------|------------|----------|
| m04_vlm_tag | .tags_checkpoint.npz | processed_keys set | 500 clips |
| m04d_motion | .m04d_checkpoint.npz | processed_keys set | 200 clips |
| m05_vjepa | .m05_checkpoint.npz | processed_keys set | 500 clips |
| m05b_baselines | .m05b_<enc>_checkpoint.npz | processed_keys set | 500 clips |
| m05c_overlap | .m05c_checkpoint.npz | processed_keys set | 500 clips |
| m00d_download | manifest.json | processed_hf_shards list | per shard |
| m06-m08b | None | Read-only post-processing | N/A |

## Current Status
- **Ch9: COMPLETE** (48 outputs, 0 errors, ~6h 35m clean on RTX PRO 6000)
  - 5-encoder comparison: DINOv2 (50.5% Prec@K) > CLIP (46.0%) > shuffled (35.3%) > vjepa (14.6%) > random (12.2%)
  - V-JEPA wins Cycle@K (78.7%) but lags on spatial retrieval accuracy
  - Key finding: shuffled > normal V-JEPA → temporal encoding HURTS spatial scene classification
  - Key finding: taxonomy measures ONLY spatial features, 0 temporal fields → evaluation gap
  - External validation: arXiv:2509.21595 confirms same DINOv3 > V-JEPA spatial tradeoff
- **Temporal eval extension: TODO** (pre-Ch10 requirement)
  - m04d: GPU-RAFT optical flow motion features (Approach A — confirmed)
  - m06b: temporal correlation analysis (CPU, per encoder)
  - VLM temporal tags (Approach B — confirmed unreliable out-of-the-box; needs fine-tuning on ~1,400 clips if pursued)
  - Expected: V-JEPA >> image baselines on temporal metrics (reversal of spatial result)
- **Ch10: NOT BUILT** (m09 — continual pretraining)
- **Ch11: NOT BUILT** (m10/m11/m12 — SAM + factors + surgery)

## Architecture Gotchas
- **NEVER use HF streaming for subset runs** — GPU sits idle 90%+ while scanning 115K clips for 10K matches (8.4% hit rate). Always pre-download via m00d → `--local-data data/subset_10k_local` (100% hit rate)
- GPU scripts save .npy → CPU scripts read them (never duplicate GPU compute in plotting)
- embeddings.paths.npy stores clip keys (not local paths) — used for Hard mode ±30s exclusion
- Tags↔embeddings alignment via __key__ field
- FAISS IndexFlatL2 (not IVF-PQ) — dim-agnostic (d = embeddings.shape[1])
- Encoder suffix system: vjepa="" (backward compat), others="_encodername"
- `src/data/` is legacy local storage (gitignored). Zero Python scripts reference it. All pipeline scripts read from HF or `data/subset_10k_local/`
- m05_vjepa_embed uses orchestrator/worker subprocess pattern to manage VRAM leaks (restart every 10K clips)
- m04_vlm_tag uses same orchestrator/worker pattern (ENGINE_RESTART_EVERY clips)
- m05c reads deduped keys from m05's embeddings.paths.npy (~5K) NOT subset_10k.json (10K) — ordering dependency
- Cosine dedup in m05 uses chunked GPU computation (chunk=2048) to avoid OOM on similarity matrix
- run_evaluate.sh `cd`s into scripts/ dir — all paths in the script are relative to scripts/, not repo root

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
