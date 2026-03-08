# Codebase Architecture

## Pipeline Modules (sequential, numbered)

### m00-m03: Data Pipeline (Mac CPU, completed)
- **m00_data_prep.py**: Parse YT_videos_raw.md → JSON, word freq, city matrix
- **m00b_fetch_durations.py**: yt-dlp metadata fetch (no download)
- **m00c_sample_subset.py**: Video-level uniform 10K subset → data/subset_10k.json (seed=42)
- **m01_download.py**: Download 714 videos at 480p via yt-dlp + aria2c
- **m02_scene_detect.py**: Greedy scene-aware split [4-10s] + CRF28 encode
- **m02b_scene_fetch_duration.py**: Scan clips → clip_durations.json
- **m03_pack_shards.py**: Pack clips → WebDataset TARs → stream-upload to HF

### m04-m08b: GPU Pipeline (current scope)
- **m04_vlm_tag.py**: 3 VLM backends (Qwen/VideoLLaMA3/LLaVA). Orchestrator/worker subprocess restart every 10K clips. Checkpoint every 500.
- **m04b_vlm_select.py**: CPU-only. 5-criterion weighted bake-off comparison.
- **m04c_sanity_compare.py**: CPU-only. Reads 3 sanity JSONs, computes 4 metrics, 2x2 dashboard.
- **m05_vjepa_embed.py**: V-JEPA 2 ViT-G (1B params, frozen, float16, FA2, torch.compile). Producer-consumer HF stream → parallel decode → GPU inference. Cosine dedup (0.95).
- **m05b_baselines.py**: 4 baseline encoders — `--encoder random|dinov2|clip|vjepa_shuffled|all`. `all` runs all 4 sequentially. Random=CPU, others=GPU. Shared producer-consumer + checkpoint pattern. Optimized: DINOv2=FA2+torch.compile, CLIP=SDPA+torch.compile, shuffled=FA2+torch.compile. Producer pre-processes (processor runs in CPU thread, GPU never waits).
- **m05c_true_overlap.py**: Augmented V-JEPA embeddings (BYOL/DINO multi-crop). View A: large crop + color jitter. View B: small crop + gaussian blur. Output: overlap_augA/B.npy.
- **m06_faiss_metrics.py**: FAISS-GPU kNN → 9 metrics Easy/Hard. `--encoder` flag (dim-agnostic). `--true-overlap` for True Overlap@K.
- **m07_umap.py**: cuML GPU UMAP (1408→2D). `--encoder` flag. Output: umap_2d{sfx}.npy.
- **m08_plot.py**: CPU-only matplotlib. UMAP scatter, confusion matrix, kNN grid.
- **m08b_compare.py**: CPU-only. Multi-encoder comparison: grouped bar chart, radar plot, LaTeX table. Reads m06_metrics_*.json.

## Utils
- **config.py**: Path constants, VLM_MODELS, ENCODER_REGISTRY (5 encoders with suffix-based file paths), load_subset(), get_output_dir(), get_encoder_files(), check_gpu(), add_subset_arg(), add_encoder_arg()
- **gpu_batch.py**: compute_batch_sizes(gpu_vram_gb) — returns `{"vjepa", "image_encoder", "transformers", "transformers_batch"}`. V-JEPA: linear from A100-40GB baseline. Image encoder: 4x vjepa (cap 256) for DINOv2/CLIP. VLM: VRAM-based cost model + AdaptiveBatchSizer
- **wandb_utils.py**: add_wandb_args, init_wandb, log_metrics, log_image, log_artifact, finish_wandb. All no-op when run=None
- **tag_taxonomy.json**: v3 taxonomy — 16 fields (traffic_mix, ped_vehicle_separation, road_encroachment, video_quality + 12 others)

## ENCODER_REGISTRY (config.py)
| Encoder | Model | Dim | Suffix | Type |
|---------|-------|-----|--------|------|
| vjepa | vjepa2-vitg-fpc64-384 | 1408 | "" (backward compat) | video |
| random | — | 1408 | _random | synthetic |
| dinov2 | dinov2-vitl14 | 1024 | _dinov2 | image (middle frame) |
| clip | clip-vit-large-patch14 | 768 | _clip | image (middle frame) |
| vjepa_shuffled | vjepa2-vitg-fpc64-384 | 1408 | _vjepa_shuffled | video (shuffled) |

## Key Patterns
- All GPU scripts: --SANITY/--FULL + --subset + --no-wandb + --gpu-mem
- Checkpoint/resume: atomic checkpoint + recovery (m04, m05, m05b, m05c)
- Producer-consumer: m04 (preprocess→Queue→GPU), m05/m05b/m05c (decode→preprocess→Queue→GPU). Producer pre-processes tensors on CPU thread so GPU never waits for preprocessing
- Fail loud: check_gpu() exits if no CUDA
- m05b `--encoder all`: runs random→dinov2→clip→vjepa_shuffled sequentially, skips if output exists
- GPU optimizations per encoder: V-JEPA/shuffled=FA2+torch.compile, DINOv2=FA2+torch.compile, CLIP=SDPA+torch.compile

## Overnight Automation
- **run_ch9_overnight.sh**: Chains m04→m05→m05b→m05c→m06→m07→m08→m08b with pre-flight checks, per-step verification, timing, and final output audit. Two modes: `--SANITY` (~15 min) and `--FULL` (~14-17h)
