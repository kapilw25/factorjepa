# Codebase Architecture

## Pipeline Modules (sequential, numbered)

### m00-m03: Data Pipeline (Mac CPU, already completed)
- **m00_data_prep.py**: Parse YT_videos_raw.md → JSON, word freq, city matrix. Reads from Literature/Prev_work4/
- **m00b_fetch_durations.py**: yt-dlp metadata fetch (no download). Outputs outputs_data_prep/video_durations.json
- **m00c_sample_subset.py**: Video-level uniform 10K subset → data/subset_10k.json (seed=42). Reads outputs_data_prep/clip_durations.json
- **m01_download.py**: Download 714 videos at 480p via yt-dlp + aria2c. Output: src/data/videos/
- **m02_scene_detect.py**: Greedy scene-aware split [4-10s] + CRF28 encode. Uses PySceneDetect ContentDetector(threshold=15). Auto-polls for m01. Output: src/data/clips/{section}/
- **m02b_scene_fetch_duration.py**: Scan clips → outputs_data_prep/clip_durations.json + per-dir metadata.jsonl
- **m03_pack_shards.py**: Pack clips → WebDataset TARs (1000 clips/shard) → stream-upload to HF. Deletes local shard after upload.

### m04-m08: GPU Pipeline (current scope)
- **m04_vlm_tag.py**: 3 VLM backends (Qwen/transformers, VideoLLaMA3/transformers, LLaVA-NeXT-Video/transformers). Orchestrator/worker subprocess pattern (restart every 10K clips for VRAM leak). Producer-consumer HF streaming. Checkpoint every 500 clips (atomic os.replace). Modes: --SANITY (20), --BAKEOFF (2500), --FULL.
  - Output BAKEOFF: src/data/bakeoff/tags_{model}.json
  - Output FULL+subset: src/outputs_poc/tags.json
  - Output FULL: src/outputs/tags.json
  - env vars: OMP_NUM_THREADS=1

- **m04b_vlm_select.py**: CPU-only. 5-criterion weighted comparison (JSON parse 30%, agreement 25%, speed 20%, taxonomy 15%, conf calibration 10%). Reads 3 bakeoff JSONs. Output: src/data/bakeoff/vlm_comparison.json + .png/.pdf + vlm_dashboard.{png,pdf} (2x2 diagnostic)

- **m04c_sanity_compare.py**: CPU-only. Reads 3 tags_sanity_{model}.json files, computes 4 metrics (parse rate, scene diversity, confidence calibration, on/off-taxonomy objects), generates 2x2 dashboard. Output: src/outputs/m04c_sanity_compare.{png,pdf}

- **m05_vjepa_embed.py**: V-JEPA 2 ViT-G (1B params, frozen, float16, FA2, torch.compile). Requires flash-attn (FATAL exit if missing). Producer-consumer: HF stream → parallel decode (torchcodec or PyAV) → GPU inference. Async checkpoints (.npz). Cosine dedup (threshold 0.95). Output: embeddings.npy + embeddings.paths.npy

- **m06_faiss_metrics.py**: FAISS-GPU IndexFlatL2 (<1000) or IndexIVFFlat (≥1000). 9 metrics in Easy/Hard mode. Hard = exclude ±30s same video. Metrics: Cycle@K, Overlap@K (dim-split approx), Silhouette, Prec@K, mAP@K, nDCG@K, per-scene purity, multi-attr slices, confidence sweep. Saves knn_indices.npy for m08.

- **m07_umap.py**: cuML GPU UMAP (n_components=2, n_neighbors=15, min_dist=0.1). Output: umap_2d.npy

- **m08_plot.py**: CPU-only matplotlib. Reads pre-computed .npy + tags.json. UMAP scatter (colored by scene_type), confusion matrix, kNN grid (8 rows, thumbnail extraction via ffmpeg). No GPU compute.

## Utils
- **config.py**: All path constants, VLM_MODELS dict, load_subset(), get_output_dir(), check_gpu(), add_subset_arg(), build_video_section_map(), check_output_exists()
- **gpu_batch.py**: compute_batch_sizes(gpu_vram_gb) — auto-scales from A100-40GB baseline. Profiles: vjepa(16,64), transformers(4,16).
- **wandb_utils.py**: add_wandb_args, init_wandb, log_metrics, log_image, log_artifact, finish_wandb. All no-op when run=None.
- **export_metadata.py**: tags.json → per-dir metadata.jsonl for HF upload
- **hf_utils.py**: HF auth, token, README gen, upload

## Key Patterns
- All GPU scripts: --SANITY/--BAKEOFF/--FULL + --subset + --no-wandb + --gpu-mem
- POC mode: --subset data/subset_10k.json → outputs_poc/
- Full mode: no --subset → outputs/
- Checkpoint/resume: all long-running scripts (m04, m05) have atomic checkpoint + recovery
- Producer-consumer: m04 (preprocess→Queue→GPU infer), m05 (decode→Queue→GPU embed)
- Fail loud: check_gpu() exits if no CUDA, FAISS exits if no GPU, cuML exits if not installed
