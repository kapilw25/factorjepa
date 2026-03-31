# vLLM + Qwen3.5-9B on NVIDIA Blackwell (sm_120) — WalkIndia-200K

## Context

Video tagging pipeline (`m04_vlm_tag.py`) runs VLMs on Indian street clips.
- **10K POC**: Qwen3-VL-8B via transformers (validated, 0.919 weighted score). Completed.
- **115K FULL**: **Qwen3.5-9B** via vLLM (native multimodal, MoE efficiency, live video optimized).

Current backend: transformers v3, batched `model.generate()`, 1.70 clips/s on 96GB RTX PRO 6000.
Goal: vLLM + Qwen3.5-9B for continuous batching → estimated 5-15 clips/s for 115K full run.

### Why Qwen3.5-9B over Qwen3-VL-8B for 115K?

| Feature | Qwen3-VL-8B (POC) | Qwen3.5-9B (FULL) |
|---------|-------------------|-------------------|
| Architecture | Vision encoder + LM decoder (two-stage) | **Unified native multimodal** (single model) |
| Video support | Requires specific `-VL` variant | All models natively handle video |
| Context | ~256K tokens | 262K native, extendable to **1M** |
| Efficiency | Dense, 8B params all active | **Gated DeltaNet + MoE** — lower VRAM/RAM per token |
| Speed | Standard | Optimized for **live video** (<200ms latency) |
| Performance | Good for its generation | **9B beats older 30B-VL** on all benchmarks |
| Transformers video? | Works | **BROKEN** ([issue #58](https://github.com/QwenLM/Qwen3.5/issues/58)) — vLLM only |

**NOTE:** Qwen3.5-9B video inference ONLY works via vLLM/OpenAI API. Transformers `model.generate()` with video hits a `StopIteration` bug in `get_rope_index` (PR [transformers#44474](https://github.com/huggingface/transformers/pull/44474) pending). This makes vLLM migration MANDATORY for the 115K run.

**Hardware:**
- Debug/SANITY: RTX PRO 4000 Blackwell (24GB VRAM, sm_120, ~$0.2/hr)
- Production/FULL: RTX PRO 6000 Blackwell (96GB VRAM, sm_120, ~$0.8/hr)

**Reference:** [tutorial_vLLM_Qwen3.md](tutorial_vLLM_Qwen3.md) (official Ascend NPU docs — API examples are universal, deployment setup is NPU-specific)

---

## Why vLLM Failed Previously (10 Root Causes)

### Round 1: vLLM 0.11.0 (original attempt → abandoned for transformers)

| # | Root Cause | Details |
|:---:|:---|:---|
| 1 | **BF16 model too large** | Qwen3-VL-8B BF16 = 17.5 GB weights. With vLLM overhead (KV cache + 4GB multimodal cache + CUDA graphs) → exceeds 24GB |
| 2 | **No sm_120 in prebuilt wheels** | vLLM PyPI wheels lack Blackwell sm_120 CUDA kernels → runtime crash. [Issue #35432](https://github.com/vllm-project/vllm/issues/35432) |
| 3 | **PyTorch version mismatch** | vLLM expects stable PyTorch (2.9.1). Blackwell requires nightly (2.12.0.dev+cu128). `pip install vllm` = incompatible |

### Round 2: vLLM 0.18.1 nightly on 96GB RTX PRO 6000 (resolved March 2026)

| # | Root Cause | Fix |
|:---:|:---|:---|
| 4 | **`setup_env_uv.sh` skipped install when venv existed** | Old check: `if [ -d "venv_vllm" ]` → skipped even when vLLM wasn't installed. Fix: check `from vllm import LLM` importability |
| 5 | **`No module named pip` in uv venv** | `uv venv` creates pip-less envs. `python -m pip install` fails. Fix: use `uv pip install --python venv_vllm/bin/python` |
| 6 | **`no such option: --torch-backend`** | `--torch-backend=auto` is a vLLM custom installer flag, NOT a pip/uv flag. pip 26.0.1 rejects it. Fix: remove flag, install torch first via `requirements_gpu_vllm.txt` |
| 7 | **Install order: vLLM before torch** | vLLM needs torch present at install time. Old script tried vLLM first (failed), then Qwen deps (which pulled torch). Fix: Qwen+torch deps first, then vLLM |
| 8 | **`spawn` multiprocessing crash** | vLLM v0.18+ uses `spawn` (not `fork`). `smoke_test_vllm.py` had `LLM()` at module level → child re-imports → infinite recursion. Fix: `if __name__ == '__main__':` guard |
| 9 | **`total_mem` AttributeError** | `torch.cuda.get_device_properties(0).total_mem` wrong. Fix: `.total_memory` |
| 10 | **`KeyError: checkpoint_every_vlm`** | `m04_vlm_tag_vllm.py` used non-existent `pipeline.yaml` key `checkpoint_every_vlm`. Correct key: `checkpoint_every`. Smoke test passed but `run_evaluate.sh --vllm` would crash at module-level import. Fix: `_pcfg["streaming"]["checkpoint_every"]` |

---

## What Changed Since (vLLM 0.11 → 0.17)

| Change | Impact |
|:---|:---|
| **Qwen3.5-9B released (Feb 2026)** | Native multimodal (no `-VL` variant needed). MoE architecture = lower VRAM per token. 516K+ HF downloads. Video only via vLLM. |
| **Official FP8 model released** | `Qwen3-VL-8B-Instruct-FP8` = 10.6 GB (saves ~7GB vs BF16). Qwen3.5-9B FP8 TBD. |
| **Community source-build guides** | Documented path for Blackwell: [vllm-blackwell-setup](https://github.com/Audible83/vllm-blackwell-setup) |
| **Memory conservation params** | `--enforce-eager`, `--mm-processor-cache-gb`, `--limit-mm-per-prompt video=1` |
| **FP8 kernel fixes for Blackwell** | NVFP4/FP8 workstation GPU fixes merged in vLLM 0.13+ |
| **Video inference documented** | [Qwen3-VL vLLM Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html) — fps control, video_pad tokens |

---

## Feasibility Verdict (VRAM Analysis)

| Config | Weight Size | Total w/ vLLM | Fits 24GB? | Fits 96GB? | Headroom (24/96 GB) |
|:---|:---:|:---:|:---:|:---:|:---|
| **Qwen3.5-9B BF16** | ~18 GB | ~22-25 GB | **TIGHT** | YES | 0-2 / 71-74 GB |
| **Qwen3.5-9B (MoE active)** | ~18 GB load, ~6-8 GB active | ~15-20 GB | **YES** | YES | 4-9 / 76-81 GB |
| Qwen3-VL-8B BF16 (old) | 17.5 GB | ~22-24 GB | **NO** | YES | 0-2 / 72-74 GB |
| Qwen3-VL-8B FP8 (old) | 10.6 GB | ~15-17 GB | YES | YES | 7-9 / 79-81 GB |

**Decision: Use `Qwen/Qwen3.5-9B` (BF16). MoE architecture keeps active VRAM low despite 9B total params. On 96GB, ~74GB headroom for KV cache → high concurrency. On 24GB, use `--enforce-eager` + `--max-model-len 4096`.**

> **Note:** Qwen3.5-9B uses "Gated DeltaNet + MoE" — all 9B params are loaded (~18GB) but only a subset of experts are active per token (~6-8GB active). This gives dense-model quality with sparse-model inference cost.

---

## Build Strategy

### Why Source Build is Required

1. vLLM PyPI wheels compiled **without** sm_120 → crash on Blackwell
2. PyTorch stable (2.9.1) lacks sm_120 → only nightly has it
3. Must build PyTorch nightly + vLLM from source in an isolated venv

### Environment Isolation

```
venv_walkindia/   ← existing: PyTorch nightly + FAISS-GPU + cuML + FA2 (m04-m08 pipeline)
venv_vllm/        ← NEW: PyTorch nightly + vLLM (source-built) + qwen-vl-utils (m04 vLLM only)
```

**CRITICAL: Never install vLLM into venv_walkindia. Never install FAISS/cuML into venv_vllm.**

### Install Script: `setup_vllm_env.sh`

```bash
#!/bin/bash
set -euo pipefail

echo "=== Creating venv_vllm ==="
uv venv venv_vllm --python 3.12

echo "=== Installing PyTorch nightly (cu128, sm_120 support) ==="
source venv_vllm/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

echo "=== Verify PyTorch + CUDA ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name()}')"

echo "=== Building vLLM from source (sm_120) ==="
# Option A: Try pip install first (may work if vLLM 0.17+ ships cu128 wheel)
uv pip install vllm>=0.17.0 || {
    echo "Prebuilt wheel failed, building from source..."
    uv pip install git+https://github.com/vllm-project/vllm.git@main
}

echo "=== Installing Qwen dependencies ==="
uv pip install qwen-vl-utils transformers accelerate

echo "=== Verify vLLM ==="
python -c "from vllm import LLM; print('vLLM OK')"

echo "=== DONE ==="
echo "Activate: source venv_vllm/bin/activate"
```

**Estimated build time: 1-4 hours** (PyTorch CUDA compilation is the bottleneck).

Ref: [vllm-blackwell-setup](https://github.com/Audible83/vllm-blackwell-setup), [vLLM on RTX5090 forum](https://discuss.vllm.ai/t/vllm-on-rtx5090-working-gpu-setup-with-torch-2-9-0-cu128/1492)

---

## Launch Configs

### Debug / SANITY (24GB RTX PRO 4000)

Tight VRAM — single-request, low FPS, no CUDA graphs.

```bash
source venv_vllm/bin/activate
python -u src/m04_vlm_tag_vllm.py --model qwen --backend vllm --SANITY 2>&1 | tee logs/m04_vllm_sanity.log
```

LLM constructor params for 24GB:
```python
llm = LLM(
    model="Qwen/Qwen3.5-9B",
    max_model_len=4096,              # short context (video tagging prompts are ~500 tokens)
    gpu_memory_utilization=0.92,     # use 92% of 24GB = 22.1 GB
    max_num_seqs=1,                  # single-request (no concurrent batching on 24GB)
    enforce_eager=True,              # skip CUDA graphs, saves ~1-2 GB
    limit_mm_per_prompt={"video": 1},
    mm_processor_kwargs={"fps": 1.0},  # 1 frame/sec to minimize video tokens
)
```

**Expected VRAM**: ~15-18 GB total, ~6-9 GB headroom.

### Production / FULL (96GB RTX PRO 6000)

Full VRAM — continuous batching, higher concurrency, CUDA graphs enabled.

```bash
source venv_vllm/bin/activate
python -u src/m04_vlm_tag_vllm.py --model qwen --backend vllm --FULL --subset data/subset_10k.json 2>&1 | tee logs/m04_vllm_full_poc.log
```

LLM constructor params for 96GB:
```python
llm = LLM(
    model="Qwen/Qwen3.5-9B",
    max_model_len=8192,              # more context headroom
    gpu_memory_utilization=0.90,     # 86.4 GB available
    max_num_seqs=16,                 # concurrent batching (continuous)
    limit_mm_per_prompt={"video": 1},
    mm_processor_kwargs={"fps": 1.0},
    # enforce_eager=False (default) — CUDA graphs enabled for throughput
)
```

**Expected VRAM**: ~25-35 GB total, ~60 GB headroom for KV cache.
**Expected throughput**: 5-10 clips/s (vs 1.70 clips/s transformers v3).

### 115K Full Run (96GB, no --subset)

```bash
source venv_vllm/bin/activate
python -u src/m04_vlm_tag_vllm.py --model qwen --backend vllm --FULL 2>&1 | tee logs/m04_vllm_full_115k.log

# Then switch back for m05-m08 (FAISS/cuML need venv_walkindia)
source venv_walkindia/bin/activate
python -u src/m05_vjepa_embed.py --FULL 2>&1 | tee logs/m05_full_115k.log
python -u src/m06_faiss_metrics.py --FULL 2>&1 | tee logs/m06_full_115k.log
python -u src/m07_umap.py --FULL 2>&1 | tee logs/m07_full_115k.log
python -u src/m08_plot.py --FULL 2>&1 | tee logs/m08_full_115k.log
```

**Estimated time for 115K**: ~3-6 hours at 5-10 clips/s (vs ~19 hours with transformers).

---

## Video Inference Details

### Option A: OpenAI-compatible API (recommended for Qwen3.5-9B)

Qwen3.5-9B model card documents video via OpenAI API served by vLLM:

```python
from openai import OpenAI
client = OpenAI()  # connects to local vLLM server

messages = [{"role": "user", "content": [
    {"type": "video_url", "video_url": {"url": "/path/to/clip.mp4"}},
    {"type": "text", "text": TAG_PROMPT},
]}]

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=messages,
    max_tokens=512,
    temperature=0.1,
    extra_body={
        "mm_processor_kwargs": {"fps": 1.0, "do_sample_frames": True},
    },
)
tag_json = response.choices[0].message.content
```

**vLLM server launch:**
```bash
source venv_vllm/bin/activate
vllm serve Qwen/Qwen3.5-9B \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16 \
    --limit-mm-per-prompt video=1 \
    --media-io-kwargs '{"video": {"num_frames": -1}}'
```

### Option B: Direct vLLM LLM() API (if server mode not desired)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3.5-9B", ...)
# Use process_vision_info or direct multimodal input
```

**Key differences from transformers backend:**
- No `AdaptiveBatchSizer` — vLLM handles batching via PagedAttention
- No producer-consumer queue — vLLM has internal preemptive scheduler
- No OOM recovery — vLLM manages KV cache allocation
- No manual left-padding — vLLM handles variable-length sequences
- **Qwen3.5-9B specific:** natively multimodal, no separate vision processor needed

Ref: [Qwen3.5-9B Model Card — Video Input](https://huggingface.co/Qwen/Qwen3.5-9B#video-input)

---

## VRAM Conservation Params Reference

| Parameter | What it does | 24GB | 96GB |
|:---|:---|:---:|:---:|
| `--max-model-len` | Limits max context length (tokens). Smaller = less KV cache | 4096 | 8192 |
| `--gpu-memory-utilization` | Fraction of VRAM vLLM can use (default 0.9) | 0.92 | 0.90 |
| `--enforce-eager` | Disables CUDA graphs, saves ~1-2 GB (costs ~10-20% throughput) | YES | NO |
| `--max-num-seqs` | Max concurrent batch size | 1 | 16 |
| `--limit-mm-per-prompt video=1` | Limits to 1 video per prompt | YES | YES |
| `--mm-processor-cache-gb` | Multimodal processor cache (default 4 GiB) | 1 | 4 |
| `--kv-cache-dtype fp8_e5m2` | FP8 KV cache (halves KV memory) | If needed | NO |

Ref: [vLLM Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory/), [vLLM Optimization](https://docs.vllm.ai/en/stable/configuration/optimization/)

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|:---|:---:|:---|
| **Source build fails on sm_120** | High | Follow [vllm-blackwell-setup](https://github.com/Audible83/vllm-blackwell-setup). Fallback: wait for prebuilt sm_120 wheel |
| **OOM on 24GB with video** | Medium | FP8 model + `enforce_eager` + `max_model_len=4096` + `max_num_seqs=1`. Fallback: `Qwen3-VL-4B-Instruct-FP8` |
| **VRAM leak over time** | Medium | [Issue #28230](https://github.com/vllm-project/vllm/issues/28230). Mitigated by `ENGINE_RESTART_EVERY` subprocess pattern (already in m04) |
| **FP8 quality degradation** | Low | Official FP8 model "nearly identical" to BF16 per [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B). Verify on SANITY (20 clips) |
| **PyTorch nightly breaks vLLM** | Medium | Pin exact nightly version in `setup_vllm_env.sh`. Test before production run |
| **FPS too low for tagging quality** | Low | fps=1.0 gives 4-10 frames for 4-10s clips. Sufficient for scene-level tagging (not action recognition) |

---

## Execution Plan

```
Phase 1 ✅: transformers v3 (1.70 clips/s, 91% GPU util) — DONE, m04_vlm_tag.py works
Phase 2:    Build venv_vllm on Blackwell (source build PyTorch + vLLM)
Phase 3:    Smoke test — m04_vllm_smoke.py --SANITY (20 clips, 24GB)
Phase 4:    Copy m04 → m04_vlm_tag_vllm.py with VLLMQwenBackend
Phase 5:    Benchmark — BAKEOFF (2500 clips) vLLM vs transformers
Phase 6:    Production — 115K clips on 96GB RTX PRO 6000
```

---

## Model Weights Reference

| Model | Size | Precision | Architecture | HuggingFace |
|:---|:---:|:---:|:---|:---|
| **Qwen3.5-9B (FULL)** | **~18 GB** | **BF16** | **Gated DeltaNet + MoE (native multimodal)** | [Link](https://huggingface.co/Qwen/Qwen3.5-9B) |
| Qwen3.5-9B-Base | ~18 GB | BF16 | Gated DeltaNet + MoE (pre-trained) | [Link](https://huggingface.co/Qwen/Qwen3.5-9B-Base) |
| Qwen3-VL-8B-Instruct (POC) | 17.5 GB | BF16 | Dense (vision encoder + LM) | [Link](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) |
| Qwen3-VL-8B-Instruct-FP8 | 10.6 GB | FP8 (w8a8) | Dense | [Link](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8) |

---

## Sources

- [Qwen3.5-9B Model Card](https://huggingface.co/Qwen/Qwen3.5-9B) — video input via vLLM, 516K+ downloads
- [Qwen3.5-9B video bug — GitHub Issue #58](https://github.com/QwenLM/Qwen3.5/issues/58) — transformers video broken, vLLM works
- [Transformers fix PR #44474](https://github.com/huggingface/transformers/pull/44474) — pending fix for transformers video
- [vLLM GitHub Releases](https://github.com/vllm-project/vllm/releases) — v0.17.0 (latest as of March 2026)
- [vLLM Prebuilt wheels fail on Blackwell — Issue #35432](https://github.com/vllm-project/vllm/issues/35432)
- [vLLM on RTX5090 — Forum Guide](https://discuss.vllm.ai/t/vllm-on-rtx5090-working-gpu-setup-with-torch-2-9-0-cu128/1492)
- [vLLM Blackwell Setup Guide](https://github.com/Audible83/vllm-blackwell-setup)
- [PyTorch sm_120 support — Issue #164342](https://github.com/pytorch/pytorch/issues/164342)
- [Qwen3-VL vLLM Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [vLLM Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)
- [vLLM VRAM Leak — Issue #28230](https://github.com/vllm-project/vllm/issues/28230)
- [vLLM FP8 Blackwell Fixes — Issue #31085](https://github.com/vllm-project/vllm/issues/31085)
- [PyTorch + vLLM Integration Blog](https://pytorch.org/blog/pytorch-vllm-%E2%99%A5%EF%B8%8F/)
