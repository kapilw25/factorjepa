# WalkIndia-200K Project Memory

## Project Overview
Research benchmark testing if V-JEPA (Meta's video foundation model, trained on Western data) transfers to Indian street scenes. Pipeline: YouTube videos → scene-split clips → WebDataset shards (HF) → VLM tagging → V-JEPA embeddings → FAISS metrics → UMAP → plots.

## Key Files
- See [codebase.md](codebase.md) for full module-by-module details
- See [debugging.md](debugging.md) for known issues and fixes

## GPU Environment
- Debug/SANITY GPU: RTX PRO 4000 Blackwell (24GB VRAM, sm_120, ~$0.2/hr)
- Full/BAKEOFF GPU: RTX PRO 6000 Blackwell (96GB VRAM, sm_120, ~$0.8/hr)
- PyTorch 2.12.0.dev+cu128 (nightly), CUDA 12.8, FA2 (built from source for sm_120)
- cuML 26.02, FAISS-GPU 1.13.2 (built from source for sm_120 via build_faiss_sm120.sh), wandb 0.25
- transformers >=4.57.0,<5.0 (pinned), Python 3.12.12, UV package manager

## Critical Path Constants (config.py)
- PROJECT_ROOT = src/../.. (repo root)
- DATA_DIR = src/data/, OUTPUTS_DIR = src/outputs/, OUTPUTS_POC_DIR = src/outputs_poc/
- BAKEOFF_DIR = src/data/bakeoff/
- HF_DATASET_REPO = "anonymousML123/walkindia-200k"
- VJEPA: facebook/vjepa2-vitg-fpc64-384, 64 frames, 1408-dim embeddings
- VLMs: qwen (transformers), videollama (transformers), llava (transformers) — all use transformers sequential inference
- FAISS_K_NEIGHBORS = 6, BAKEOFF_CLIP_COUNT = 2500

## Known Fixes Applied
- PyTorch 2.8.0+ renamed `total_mem` → `total_memory` (fixed in setup_env_uv.sh + plan_execution.md)
