# Debugging Notes

## Known Issues & Fixes

### 1. PyTorch 2.8.0+ API change: total_mem → total_memory
- **Files affected**: setup_env_uv.sh (line 278), plan_execution.md (line 20)
- **Fix**: `torch.cuda.get_device_properties(0).total_mem` → `.total_memory`
- **Status**: FIXED

### 2. Blackwell GPU (sm_120) requires special handling
- Flash-Attention 2 must be built from source (no prebuilt wheel for sm_120)
- PyTorch nightly cu128 required (not stable cu124)
- Build command: FLASH_ATTN_CUDA_ARCHS=120 MAX_JOBS=4 uv pip install /tmp/flash-attention-build --no-build-isolation

### 3. vLLM abandoned → all 3 VLMs now use transformers
- vLLM 0.11.0 V1 engine OOMs on 24GB RTX PRO 4000 (model=16.78GB + profiling overhead > 24GB)
- Hit 4 cascading errors: V0/V1 routing, transformers 5.x compat, warmup OOM, KV cache OOM
- **Decision**: Switched QwenBackend from vLLM to transformers (matching VideoLLaMA3/LLaVA-NeXT-Video)
- Removed `vllm>=0.6.0` from requirements_gpu.txt
- All 3 VLMs now use same pattern: AutoModelForCausalLM + sequential inference
- Qwen-specific: uses `qwen_vl_utils.process_vision_info()` for video preprocessing
- transformers pin `>=4.57.0,<5.0` kept (Qwen3-VL needs 4.57+)

### 4. AutoModelForCausalLM fails for Qwen3-VL
- `ValueError: Unrecognized configuration class Qwen3VLConfig for AutoModelForCausalLM`
- Qwen3-VL is a vision-language model, not a pure causal LM
- **Fix**: Use `Qwen3VLForConditionalGeneration` explicitly (VideoLLaMA3 works with AutoModelForCausalLM; LLaVA-NeXT-Video uses native LlavaNextVideoForConditionalGeneration)
- Also: `torch_dtype` deprecated in newer transformers → use `dtype` instead
- **Status**: FIXED

### 5. SANITY mode output collision — all 3 VLMs wrote to same file
- `get_tags_file()` had no `is_sanity` parameter — all models in `--SANITY` mode wrote to `outputs/tags.json`
- Running VideoLLaMA3 sanity prompted to delete Qwen's sanity results
- **Fix**: Added `is_sanity` param → model-specific files: `tags_sanity_qwen.json`, `tags_sanity_videollama.json`, `tags_sanity_llava.json`
- BAKEOFF mode was already correct (`tags_{model}.json`); FULL mode is intentionally shared (only winner runs)
- **Status**: FIXED

### 6. Orchestrator batch-size override bug
- Orchestrator always passed `--batch-size 4` (hardcoded default) to worker subprocesses
- This overrode `gpu_batch.compute_batch_sizes()` auto-compute in the worker (which calculated `transformers=2` for 24GB)
- **Fix**: Only pass `--batch-size` to worker when user explicitly sets it via CLI
- **Status**: FIXED

### 7. transformers version: pinned >=4.57.0,<5.0
- Keye-VL dropped (4 cascading errors: PytorchGELUTanh, SlidingWindowCache, pad_token_id, ROPE_INIT_FUNCTIONS). Replaced with LLaVA-NeXT-Video-7B-hf (native transformers, zero patches)
- VideoLLaMA3 remote code imports `VideoInput` from `transformers.image_utils` but it doesn't exist there in 4.57.x (only a type hint, not runtime). Fix: inject `typing.Any` into `image_utils.VideoInput` before import
- Qwen3-VL: natively supported >=4.57.0. LLaVA-NeXT-Video: native >=4.42.0
- **Fix**: Pin `transformers>=4.57.0,<5.0`, 1 minimal type-hint patch in VideoLLaMA3Backend.load_model()
- **Status**: FIXED

### 8. HF streaming reliability
- Producer threads retry with exponential backoff (1s→60s, max 5 retries)
- Both m04 and m05 have this pattern

### 9. FAISS-GPU sm_120 (Blackwell) — no prebuilt kernel
- `faiss-gpu-cu12` pip package only ships sm_70+sm_80 CUDA kernels
- Runtime: `CUDA error 209 (no kernel image is available for execution on the device)`
- Confirmed via `cuobjdump` — no sm_120 in the .so
- **Fix**: Build FAISS 1.13.2 from source with `-DCMAKE_CUDA_ARCHITECTURES="120"`
- Script: `build_faiss_sm120.sh` (full build ~10 min on 96-core, `--install` for re-install only)
- Requires: `apt-get install libopenblas-dev swig` before cmake
- **Status**: FIXED

### 10. UV venv has no pip binary
- `uv venv` creates minimal venvs without `pip` or `python -m pip`
- `build_faiss_sm120.sh` originally used `${VENV_DIR}/bin/pip` which didn't exist
- **Fix**: Use `uv pip` with `VIRTUAL_ENV` env var set to target the correct venv
- **Status**: FIXED

## Architecture Gotchas
- m06 saves knn_indices.npy so m08 doesn't need FAISS (CPU-only plotting)
- m07 saves umap_2d.npy so m08 doesn't need cuML
- embeddings.paths.npy stores clip keys (not local paths) — used for Hard mode exclusion parsing
- Tags alignment with embeddings: verified via __key__ field in tags and clip paths
- FAISS IVFFlat (not IVF-PQ as mentioned in plan) — simpler, sufficient for 10K-115K scale

## Batch Size Auto-Scaling
- gpu_batch.py: baseline A100-40GB, scale = actual_vram / 40
- RTX PRO 4000 (25GB): scale ≈ 0.62 → vjepa batch=9, transformers batch=2
- All 3 VLMs use transformers batch sizing (no more vLLM exception)
