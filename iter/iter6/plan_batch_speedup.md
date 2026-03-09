# Batched VLM Inference Refactor — Implementation Plan

## Context

Qwen bakeoff runs at **0.09 clips/s** and **32% GPU utilization** on RTX PRO 6000 (96 GB).
Root cause: `model.generate()` called sequentially per clip, CPU preprocessing interleaved in GPU thread.

**Goal:** true batched `model.generate()` → **achieved 1.70 clips/s, 91% GPU util (18.3× speedup).**

---

## Architecture

### GPU Utilization vs GPU Memory

> **GPU SM utilization** ≠ **GPU memory pressure** (what causes OOM).

Monitor VRAM via `torch.cuda.mem_get_info()` (CUDA driver-level), not SM utilization.

### AdaptiveBatchSizer — Core Logic

```
after_batch_success():
    VRAM > 85%  →  shrink by 1       (proactive, avoids OOM)
    VRAM < 65%  →  grow by 1         (only if zero OOM history)
    65%-85%     →  hold steady        (sweet spot)

on_oom():
    halve sub-batch size              (geometric backoff: 64→32→16→8→4→2→1)
    at min size → return False        (dummy-tag this sub-batch, continue run)
```

OOM cleanup **outside `except` block** (fairseq pattern — exception holds stack frame → prevents tensor dealloc).

### VRAM-Based Auto Batch Sizing (gpu_batch.py v3)

Replaced hardcoded baselines with direct VRAM calculation:

```
max_batch = (VRAM × 80% − model_overhead) / marginal_per_clip
```

Empirical cost model: Qwen3-VL-8B uses ~16 GB fixed (weights) + ~0.78 GB/clip marginal.

| GPU | VRAM | max_batch | initial sub-batch | est. VRAM |
|-----|------|-----------|-------------------|-----------|
| RTX PRO 4000 | 24 GB | 4 | 3 | 19 GB (80%) |
| A100-40 | 40 GB | 20 | 16 | 32 GB (79%) |
| A100-80 | 80 GB | 61 | 48 | 64 GB (79%) |
| RTX PRO 6000 | 96 GB | 64 (cap) | 51 | 66 GB (69%) |

`--gpu-mem` override if auto-detection fails. AdaptiveBatchSizer fine-tunes at runtime.

### Producer/Consumer Pipeline

```
Producer (ThreadPoolExecutor)      Consumer (GPU, adaptive sub-batching)
┌───────────────────────────┐      ┌──────────────────────────────────┐
│ preprocess N clips (CPU)  │─queue─▶│ model.generate() sub-batches of M │
│ (decord decode, tokenize) │      │ M adapts based on VRAM pressure    │
└───────────────────────────┘      └──────────────────────────────────┘
```

`PREFETCH_QUEUE_SIZE = 4` batches buffered. Producer batch N = consumer max M (both = `transformers` profile).

---

## Backend Implementations (all ✅ DONE)

### Step 1 — QwenBackend

- `preprocess_one()`: mp4 → `process_vision_info(fps=1.0)` (decord, thread-safe) → `apply_chat_template(tokenize=False)` → delete temp file. Returns `{text, video_inputs, video_kwargs, key}`. Runs in `ThreadPoolExecutor(6)`.
- `_generate_batched()`: single `processor(text=[...], videos=[...], padding=True, padding_side="left")` → single `model.generate()` → `batch_decode` → parse JSON.
- OOM → re-raises to AdaptiveBatchSizer. Non-OOM → `_generate_per_clip()` fallback (reuses decoded tensors).

### Step 2 — LLaVANextBackend

**Cleanest batching** — `LlavaNextVideoProcessor` natively supports batched input.
- `preprocess_one()`: PyAV decode to numpy `(16,H,W,3)`. No temp files.
- `_generate_batched()`: `processor(text=[prompt]*N, videos=[clips], padding=True)` → stacked 5D pixel_values (fixed `LLAVA_NUM_FRAMES=16`) → single `model.generate()`.
- `padding_side="left"` already default for LlamaTokenizerFast.

### Step 3 — VideoLLaMA3Backend

Processor does **not** support batched input. CPU prep stays sequential on GPU thread.
- `preprocess_one()`: write mp4 + validate (lightweight).
- `_generate_batched()`: per-sample `processor()` calls (sequential) → manual left-pad `input_ids`/`attention_mask` → concat vision tensors → single `model.generate()`.
- Temp files preserved across OOM retries (processor needs to re-read from disk).

### 3-Backend Comparison

| Aspect | **Qwen** | **LLaVA** | **VideoLLaMA3** |
|--------|----------|-----------|-----------------|
| Processor batching | Yes | **Native** (cleanest) | **No** (manual pad) |
| Vision tensors | Concatenated (variable grid) | Stacked (fixed frames) | Concatenated + flattened modals |
| `preprocess_one()` | Heavy (decord + tokenize) | Medium (PyAV decode) | Light (file I/O) |
| Speedup source | CPU parallel + GPU batch | GPU batch | GPU batch only |

---

## CRITICAL: Two Bugs Discovered During 10K POC FULL Run (Mar 8, 2026)

### Context: What Happened

Ran `./run_ch9_overnight.sh --FULL` on RTX PRO 6000 (96GB). Step 1 (m04 VLM tagging) started at 11:13 AM, still running at 6:30 AM next day. Only 3,844/10,000 clips tagged in ~7 hours. Throughput declined continuously:

```
clip   320:  1.16 clips/s  (GPU warm, queue pre-filled)
clip 1,344:  0.95 clips/s
clip 2,880:  0.67 clips/s
clip 3,456:  0.59 clips/s
clip 3,712:  0.51 clips/s  ← still declining, no sign of plateau
```

GPU utilization: **0% most of the time** (nvtop confirmed). VRAM: 62%. The GPU was idle, waiting for the producer thread to deliver batches.

---

### BUG 1: Producer Starvation — HF Streaming with Subset Filtering

**Root cause**: The 10K POC subset is 8.4% of the 115K dataset. To fill ONE batch of 64 clips, the producer must:
1. Stream ~744 clips from HuggingFace (network I/O)
2. Check each clip key against the subset (CPU)
3. Discard ~680 non-matching clips
4. Decode the 64 matching clips via decord (CPU)
5. Queue the batch for GPU inference

Network I/O + filtering + decode became slower than GPU inference → GPU starves.

**This affects ALL 4 HF-streaming steps, not just m04:**

| Step | Module | Streams from HF? | Same bottleneck? | Est. time at 0.5 clips/s |
|------|--------|:-:|:-:|:-:|
| 1 | m04 (VLM tagging) | Yes | **YES** | ~5.5h |
| 2 | m05 (V-JEPA embed) | Yes | **YES** | ~5.5h |
| 3 | m05b (4 baselines) | Yes | **YES** × 4 encoders | ~22h |
| 4 | m05c (true overlap) | Yes | **YES** | ~5.5h |
| 5-7 | m06/m07/m08 | No (reads .npy) | No | ~30 min total |
| | | | **TOTAL** | **~39h** |

At current rates, the FULL pipeline would take **~39 hours** — unacceptable on an $0.80/hr GPU.

**Benchmark confirming HF itself is fast** (tested during the run):
```
Raw HF streaming (no decode): 171 clips/s, 43.5 clips/s with full bytes
Subset hit rate: 8.4% (168 matches in 2000 clips sampled)
Avg clip size: 1.1 MB
10K subset total: ~10.7 GB
Stream-through all 115K to find 10K matches: ~11 min
```

HF is not dying — the producer is just doing 12× more work than it needs to (streaming 115K to find 10K).

---

### BUG 2: Resume Creates Duplicate Tags with Subset Filtering

**The bug**: When the pipeline restarts (kill + rerun, or ENGINE_RESTART_EVERY), the orchestrator sets `--start-from` to the checkpoint count (number of matched clips). But this count != the raw HF dataset position.

**How it works now (BROKEN for subset mode):**

```
Orchestrator:
  all_tags, skip_count = load_checkpoint(tags_file)    # skip_count = 3,844 (matched clips)
  cmd = [..., "--start-from", str(skip_count), ...]     # --start-from 3844

Worker:
  _create_stream(start_from=3844)                       # ds.skip(3844) in raw dataset
  → Stream starts at raw dataset position 3844
```

**The math that shows the bug:**

```
10K subset is uniformly sampled from 115K → hit rate = 8.4%
To find 3,844 matching clips, the stream scanned through:
  3,844 / 0.084 ≈ 45,762 raw dataset positions (positions 0 → ~45,762)

On restart, stream starts at position 3,844 (not 45,762!)
  → Positions 3,844 → 45,762 contain ~(45,762-3,844) × 0.084 ≈ 3,521 ALREADY-TAGGED clips
  → These get re-processed and APPENDED to all_tags → DUPLICATE ENTRIES

Result: tags.json would contain ~3,521 duplicate clips out of 10,000.
```

**Why this didn't matter before:** Without subset filtering (full 115K mode), every dataset item gets processed, so `skip_count == raw position`. The bug only manifests with `--subset`.

**Why a simple kill+restart would make things WORSE:**
1. Duplicates corrupt tags.json (downstream m06/m08 would compute wrong metrics)
2. Speed would temporarily improve (fresh connection, early stream position) then degrade again at the same point
3. The worker re-does GPU inference on already-tagged clips — pure waste

---

### SOLUTION: Pre-Download Subset to Local Disk (`m00d_download_subset.py`)

> **STATUS: CODE COMPLETE (Mar 8, 2026).** All 7 files implemented + verified (45/45 plan requirements pass on M1 Mac via `py_compile` + AST). **Pending: SANITY test + FULL POC run on RTX PRO 6000 (96GB).**

Instead of streaming 115K clips from HF and filtering to 10K on every step, download the 10K subset clips ONCE to local disk. All subsequent steps read locally → 100% hit rate, no network I/O.

#### Files Modified (all verified `py_compile` + AST on M1 Mac)

| File | Change | Status |
|------|--------|--------|
| `src/utils/config.py` | Added `add_local_data_arg(parser)` shared helper | DONE |
| `src/m00d_download_subset.py` | **NEW** — CPU-only pre-download (~275 lines) | DONE |
| `src/m05_vjepa_embed.py` | `_create_stream(local_data=)` + `--local-data` arg + orchestrator passthrough | DONE |
| `src/m04_vlm_tag.py` | `_create_stream(local_data=)` + `already_tagged_keys` dedup + `--local-data` arg + orchestrator passthrough | DONE |
| `src/m05b_baselines.py` | `local_data` param to both producer functions + `--local-data` arg | DONE |
| `src/m05c_true_overlap.py` | `local_data` param to producer + `--local-data` arg | DONE |
| `run_ch9_overnight.sh` | Step 0 pre-download + `$LOCAL_FLAG` to Steps 1-4 + updated time estimates | DONE |

#### Pending GPU Validation

```bash
# SANITY test (20 clips, ~5 min):
python -u src/m00d_download_subset.py --subset data/subset_10k.json --SANITY 2>&1 | tee logs/m00d_sanity.log
# Verify: data/subset_10k_local/ has TARs + manifest.json

# Then test --local-data flag propagation:
python -u src/m05b_baselines.py --encoder random --SANITY --local-data data/subset_10k_local 2>&1

# FULL POC (10K clips, ~12 min download + ~8-12h pipeline):
./run_ch9_overnight.sh --FULL
```

Implementation details in the code itself (7 files modified, 45/45 plan requirements verified).

---

### Impact Summary

| Metric | Current (HF stream + filter) | After fix (local pre-download) |
|--------|-----|------|
| m04 throughput | 0.51 clips/s (declining) | ~1.5-2.0 clips/s (stable) |
| m04 ETA (10K) | ~5.5h+ | ~1.5-2h |
| m05 ETA | ~5.5h | ~1-2h |
| m05b ETA (4 enc) | ~22h | ~4-6h |
| m05c ETA | ~5.5h | ~1-2h |
| **Total pipeline** | **~39h** | **~8-12h** |
| GPU idle time | ~90% (waiting for producer) | ~10% (normal batch gaps) |
| Resume correctness | **BROKEN** (duplicate tags) | **FIXED** (dedup by __key__) |
| Pre-download cost | N/A | ~11 min one-time |

### Implementation Order

```
1. ✅ config.py: add_local_data_arg() shared helper
2. ✅ m05_vjepa_embed.py: _create_stream(local_data=) + --local-data arg + orchestrator passthrough
3. ✅ m04_vlm_tag.py: _create_stream(local_data=) + already_tagged_keys dedup + --local-data arg
4. ✅ m05b_baselines.py: local_data param to both producers + --local-data arg
5. ✅ m05c_true_overlap.py: local_data param to producer + --local-data arg
6. ✅ m00d_download_subset.py: new CPU-only script (~275 lines)
7. ✅ run_ch9_overnight.sh: Step 0 pre-download + $LOCAL_FLAG to Steps 1-4
── ALL CODE COMPLETE (Mar 8, 2026). Verified: py_compile + AST (45/45 requirements). ──
8. ⏳ SANITY test on RTX PRO 6000 (20 clips, ~5 min)
9. ⏳ FULL POC run on RTX PRO 6000 (10K clips, ~8-12h)
```

---

## Future Work: HF Stream Resilience

`ENGINE_RESTART_EVERY` already handles HF stream deaths (checkpoint + fresh worker subprocess). Observed at clip ~8K: stream stall → auto-recovery, zero data lost. **TODO for 115K:** add a per-worker timeout (e.g. 10 min no progress → kill + restart) to avoid hanging on dead sockets that never trigger subprocess exit.

---

## Measured Performance (RTX PRO 6000, 96 GB)

| Metric | v1 (sequential) | v2 (batched, hardcoded) | v3 (VRAM-auto + decord) |
|--------|----------------|--------------------------|-------------------------|
| batch_size (producer) | 9 | 35 | **64** |
| sub-batch (generate) | 1 (per-clip) | 35 (grew 28→35) | **64** (grew 51→64) |
| clips/s | 0.09 | 0.49 | **1.70** |
| GPU util | 32% | 71% | **91%** |
| VRAM usage | 25% (24 GB) | 37% (36 GB) | **63% (61 GB)** |
| Power draw | ~200 W | ~350 W | **475 W / 600 W** |
| 10K ETA | ~30 h | ~5.7 h | **~1.7 h** |
| Speedup vs v1 | 1× | 5.4× | **18.3×** |

### What changed across versions

- **v1 → v2:** Batched `model.generate()` + AdaptiveBatchSizer + producer-consumer pipeline
- **v2 → v3:** VRAM-based auto-calculation (`gpu_batch.py` rewrite) + uninstalled broken torchcodec → restored decord

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **OOM** | AdaptiveBatchSizer geometric backoff (halve on OOM, never cliff to 1) + proactive shrink at >85% VRAM |
| **OOM cleanup** | Cleanup outside `except` block (fairseq pattern) |
| **HF stream stall** | ENGINE_RESTART_EVERY + atomic checkpoint + fresh worker subprocess |
| **torchcodec ABI** | Uninstalled — decord is default video reader. torchcodec needs stable PyTorch (not nightly 2.12.0) |
| **Batched quality drift** | Minor padding effects for Qwen3-VL. Acceptable for tagging |
| **VideoLLaMA3 batch incompat** | Degrades to sub-batch=1 gracefully |

---

## Production: 115K Full Run — vLLM Migration

### Why vLLM for 115K

transformers v3 maxes out at **1.70 clips/s**. At that rate: **115K = ~19 h** — too slow for iteration.

| Feature | transformers (current) | vLLM |
|---------|----------------------|------|
| Batching | **Static** — GPU idles between batches | **Continuous** — no idle time |
| KV cache | Pre-allocated, fragmented | **PagedAttention** — near-zero fragmentation |
| Scheduling | Producer-consumer queue stalls | **Preemptive** — priority-based |
| Throughput | 1.70 clips/s (measured) | **5–10 clips/s (estimated)** |
| 115K ETA | ~19 h | **~3–6 h** |

Ref: Qwen2.5-VL-7B on A100 via vLLM: 7.35 req/s for video ([vLLM #24728](https://github.com/vllm-project/vllm/issues/24728)).

### Blocker: PyTorch Nightly vs Stable

Our venv uses PyTorch 2.12.0.dev (nightly) for sm_120 (Blackwell) + FA2 + FAISS source builds. vLLM needs stable PyTorch (2.8.x tested with vLLM 0.11).

### Installation Options

| Option | Approach | Risk | Notes |
|--------|----------|------|-------|
| **A (recommended)** | Separate `venv_vllm` | Zero | `uv venv venv_vllm && uv pip install vllm>=0.11.0 qwen-vl-utils==0.0.14` |
| B | Build vLLM from source against nightly | High | May not compile. 20-30 min build. Could break pipeline |
| C | Wait for stable PyTorch w/ sm_120 | None | Blocks production until PyTorch 2.8.0 stable ships |

### Code Changes

Add `VLLMQwenBackend` — dramatically simpler than transformers backend:

```python
from vllm import LLM, SamplingParams

class VLLMQwenBackend:
    def __init__(self):
        self.llm = LLM(model="Qwen/Qwen3-VL-8B-Instruct", dtype="bfloat16",
                        max_model_len=4096, gpu_memory_utilization=0.85)
        self.sampling_params = SamplingParams(max_tokens=512, temperature=0)

    def generate_batch(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [out.outputs[0].text for out in outputs]
```

No AdaptiveBatchSizer, no producer-consumer queue, no manual padding, no OOM recovery — vLLM handles all of it via PagedAttention.

### Execution Plan

```
Phase 1 ✅: 10K POC with transformers v3 (1.70 clips/s, 91% GPU util)
Phase 2:    m05 → m08 pipeline on 10K POC outputs
Phase 3:    Install vLLM (Option A), benchmark --SANITY (20 clips)
Phase 4:    115K production: m04 (vLLM) → m05 → m06 → m07 → m08
```

```bash
# Phase 4 commands
source venv_vllm/bin/activate
python -u src/m04_vlm_tag.py --model qwen --FULL 2>&1 | tee logs/m04_full_115k.log

source venv_walkindia/bin/activate  # switch back for FAISS/cuML
python -u src/m05_vjepa_embed.py --FULL 2>&1 | tee logs/m05_full_115k.log
python -u src/m06_faiss_metrics.py --FULL 2>&1 | tee logs/m06_full_115k.log
python -u src/m07_umap.py --FULL 2>&1 | tee logs/m07_full_115k.log
python -u src/m08_plot.py --FULL 2>&1 | tee logs/m08_full_115k.log
```
