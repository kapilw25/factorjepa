# Debugging Notes

## Known Issues & Fixes

### 1. PyTorch 2.8.0+ API change: total_mem → total_memory
- PyTorch 2.8.0+ renamed `torch.cuda.get_device_properties(0).total_mem` → `.total_memory`
- Status: FIXED

### 2. Blackwell GPU (sm_120) requires source builds
- Flash-Attention 2: `FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=4 pip wheel /tmp/flash-attention-build --no-build-isolation`
- FAISS-GPU 1.13.2: build from source with `-DCMAKE_CUDA_ARCHITECTURES="120"` (script: build_faiss_sm120.sh)
- PyTorch nightly cu128 required (not stable cu124)
- Prebuilt wheels cached to GitHub Release (`build_wheels_sm120.sh`), install with `--from-wheels`
- Status: FIXED

### 3. vLLM abandoned for 10K POC → all 3 VLMs use transformers
- vLLM 0.11.0 OOMs on 24GB (model=16.78GB + profiling > 24GB)
- All 3 POC VLMs now use transformers sequential inference
- 115K FULL: Qwen3.5-9B via vLLM (different model, works)
- Status: FIXED

### 4. AutoModelForCausalLM fails for Qwen3-VL
- Use `Qwen3VLForConditionalGeneration` explicitly (vision-language, not pure causal LM)
- Also: `torch_dtype` deprecated → use `dtype` instead
- Status: FIXED

### 5. SANITY mode output collision
- Fixed by adding `is_sanity` param → model-specific files: `tags_sanity_{model}.json`
- Status: FIXED

### 6. Orchestrator batch-size override bug
- Orchestrator hardcoded `--batch-size 4`, overriding gpu_batch auto-compute
- Fix: only pass --batch-size when user explicitly sets it via CLI
- Status: FIXED

### 7. transformers version pinned >=4.57.0,<5.0
- Keye-VL dropped (4 cascading errors). Replaced with LLaVA-NeXT-Video-7B-hf
- VideoLLaMA3: inject `typing.Any` into `image_utils.VideoInput` before import
- Status: FIXED

### 8. HF streaming reliability
- Producer threads retry with exponential backoff (1s→60s, max 5 retries)
- Status: IMPLEMENTED

### 9. FAISS-GPU sm_120 — no prebuilt kernel
- `faiss-gpu-cu12` only ships sm_70+sm_80 kernels
- Fix: build FAISS 1.13.2 from source (build_faiss_sm120.sh, ~10 min on 96-core)
- Status: FIXED

### 10. UV venv has no pip binary
- `uv venv` creates minimal venvs without pip
- Fix: use `uv pip` with VIRTUAL_ENV env var
- Status: FIXED

### 11. Qwen3.5-9B video via transformers is BROKEN
- `StopIteration` bug in `get_rope_index` (GitHub Issue #58, fix PR transformers#44474 pending)
- vLLM is the ONLY working path for Qwen3.5-9B video inference
- Impacts: 115K FULL run must use vLLM, not transformers
- 10K POC stays with Qwen3-VL-8B via transformers (validated, working)
- Status: KNOWN — workaround is vLLM

## Architecture Gotchas
- m06 saves knn_indices.npy → m08 reads it (no FAISS needed for plotting)
- m07 saves umap_2d.npy → m08 reads it (no cuML needed for plotting)
- embeddings.paths.npy stores clip keys (not local paths) — used for Hard mode ±30s exclusion
- Tags↔embeddings alignment verified via __key__ field
- FAISS uses IVFFlat (not IVF-PQ) — simpler, sufficient at 10K-115K scale
- Encoder suffix system: vjepa="" (backward compat), others="_encodername"
- FAISS is dimension-agnostic (d = embeddings.shape[1]) — metrics comparable across encoders with different dims

## Batch Size Auto-Scaling
- gpu_batch.py baseline: A100-40GB, scale = actual_vram / 40
- 4 profiles: vjepa, image_encoder (4x vjepa, cap 256), transformers, transformers_batch
- RTX PRO 4000 (24GB): scale ≈ 0.60 → vjepa=9, image_encoder=36, transformers=2
- RTX PRO 6000 (96GB): scale ≈ 2.40 → vjepa=38, image_encoder=152, transformers=9
- image_encoder profile: for DINOv2 (300M, 1 frame) and CLIP (400M, 1 frame) — much cheaper per clip than V-JEPA (1B, 64 frames)

## Setup Script Flow (setup_env_uv.sh)
- `--mac`: CPU-only packages from requirements.txt
- `--gpu`: PyTorch nightly cu128 + requirements_gpu.txt + FA2 + FAISS + cuML + wandb
- `--gpu --from-wheels`: Downloads prebuilt FA2+FAISS wheels from GitHub Release first, then installs
- Wheel precedence: local wheels/ → GitHub Release (--from-wheels) → build from source
