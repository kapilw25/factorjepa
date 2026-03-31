# vLLM + Qwen for 115K Video Tagging

> Ref: `iter/utils/vLLM_plan_Blackwell.md` (detailed build strategy + VRAM analysis)

---

## Step 0: Smoke Test (do this FIRST)

```bash
# Separate venv — NEVER install vLLM into venv_walkindia
uv venv venv_vllm --python 3.12
source venv_vllm/bin/activate
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
uv pip install qwen-vl-utils transformers

# Run smoke test
python scripts/smoke_test_vllm.py 2>&1 | tee logs/vllm_smoke.log
```

If this passes (image + video inference work), proceed. If not, debug before touching m04.

---

## Path A vs Path B

| | Path A: Offline Batch | Path B: Online Server |
|---|---|---|
| **How** | `LLM.generate(all_inputs)` in Python | `vllm serve` + OpenAI client |
| **Batching** | Pass all inputs at once, vLLM schedules internally | Fire concurrent HTTP requests |
| **Simplest for us?** | **YES** — single script, no server process | No — need server + client + coordination |
| **Structured JSON** | `StructuredOutputsParams(json=schema)` | `extra_body={"guided_json": schema}` |
| **Checkpoint/resume** | We handle it (same as current m04) | We handle it |
| **Video support** | `multi_modal_data={"video": frames}` via `process_vision_info` | `video_url` in messages |
| **VRAM control** | `gpu_memory_utilization`, `max_num_seqs` | Same, set at server launch |

**Decision: Path A (offline batch).** Simpler, no server process, same checkpoint pattern as current m04.

---

## Qwen3-VL-8B vs Qwen3.5-9B

| | Qwen3-VL-8B-Instruct | Qwen3.5-9B |
|---|---|---|
| **Architecture** | Dense (vision encoder + LM) | Gated DeltaNet + MoE (native multimodal) |
| **Params** | 8B all active | 9B total, ~6-8B active (MoE) |
| **VRAM (BF16)** | 17.5 GB | ~18 GB load, ~15 GB active |
| **Video via transformers** | Works | **BROKEN** ([issue #58](https://github.com/QwenLM/Qwen3.5/issues/58)) |
| **Video via vLLM** | Works | Works |
| **FP8 variant** | Yes (10.6 GB) | TBD |
| **POC validated?** | **YES** (0.919 weighted score on 10K) | No |
| **Context** | ~256K tokens | 262K, extendable to 1M |
| **HF downloads** | High | 516K+ |

**Decision for 115K: Start with Qwen3-VL-8B** (already validated on POC). Switch to Qwen3.5-9B only if 8B throughput is too slow or quality is insufficient. Don't risk an untested model on a $35+ GPU run.

---

## Current m04 vs vLLM

| | Current (transformers) | vLLM |
|---|---|---|
| Inference | Sequential (1 clip/call) | Continuous batching (auto) |
| Speed | ~3 clips/min (10K POC) | Expected ~10-20 clips/min |
| OOM handling | AdaptiveBatchSizer | PagedAttention (automatic) |
| JSON output | Parse + dummy fallback (5%+ failure rate) | Structured output guarantees valid JSON |
| Reproducibility | `do_sample=False` (fixed today) | `temperature=0` |
| **115K est. time** | **~35h ($35)** | **~6-12h ($6-12)** |

---

## Structured JSON (eliminates dummy tags)

```python
from pydantic import BaseModel
from vllm.sampling_params import StructuredOutputsParams

class SceneTag(BaseModel):
    scene_type: str
    weather: str
    # ... all 16 taxonomy fields

sampling_params = SamplingParams(
    temperature=0, max_tokens=512,
    structured_outputs=StructuredOutputsParams(json=SceneTag.model_json_schema()))
```

Guarantees valid JSON — no `parse_json_output()` → `get_dummy_tag()` fallback needed.

**Caveat:** Structured output + VLM not explicitly documented as supported. Validate on SANITY (20 clips) before committing.

---

## Install

```bash
# On GPU instance (separate venv)
uv venv venv_vllm --python 3.12
source venv_vllm/bin/activate
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
uv pip install qwen-vl-utils transformers
```

Requirements: `vllm >= 0.11.0`, `qwen-vl-utils >= 0.0.14`.

---

## Execution Plan

```
1. Smoke test: scripts/smoke_test_vllm.py (image + video, 5 min)
2. If pass: build m04_vlm_tag_vllm.py (Path A offline batch)
3. SANITY (20 clips): validate structured JSON + tag quality
4. BAKEOFF (2500 clips): compare vLLM vs transformers quality
5. FULL (115K): production run
```

---

## Previous Failures (why it didn't work before)

| # | Root Cause | Status Now |
|---|---|---|
| 1 | BF16 too large for 24GB GPU | 96GB GPU — solved |
| 2 | No sm_120 in prebuilt wheels | `--extra-index-url https://wheels.vllm.ai/nightly` — nightly has sm_120 |
| 3 | PyTorch version mismatch | `--torch-backend=auto` handles this |

See `iter/utils/vLLM_plan_Blackwell.md` for detailed build strategy.
