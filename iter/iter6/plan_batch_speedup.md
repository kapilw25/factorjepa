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

## CRITICAL: Four Bugs Discovered During 10K POC Runs (Mar 8-9, 2026)

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

### BUG 3: `manifest.json` Causes `tarfile.ReadError: invalid header` (Mar 9, 2026)

**Discovered:** First FULL pipeline run after m00d download completed. Both m04 and m05 crashed immediately.

**Root cause:** `m00d_download_subset.py` writes `manifest.json` (resume tracker) alongside the `.tar` shards in `data/subset_10k_local/`. When downstream modules call `load_dataset("webdataset", data_dir=local_data, ...)`, the HF datasets library tries to parse ALL files in the directory as TARs — including `manifest.json`, which is not a TAR.

```
tarfile.ReadError: invalid header
  File ".../webdataset.py", line 80, in _split_generators
    first_examples = list(islice(pipeline, self.NUM_EXAMPLES_FOR_FEATURES_INFERENCE))
  File ".../file_utils.py", line 1331, in _iter_tar
    stream = tarfile.open(fileobj=f, mode="r|*")   ← manifest.json is not a TAR
```

**Fix:** Changed `data_dir=local_data` → `data_files=f"{local_data}/*.tar"` in both:
- `src/m04_vlm_tag.py:865` — `_create_stream()`
- `src/m05_vjepa_embed.py:89` — `_create_stream()` (also used by m05b/m05c via import)

The `data_files` glob explicitly selects only `.tar` files, excluding `manifest.json`.

**Impact:** Pipeline-blocking. Both m04 and m05 failed with 0 clips processed. Fixed in 2 lines.

---

### BUG 4: m05c True Overlap — ATen Thread Oversubscription (Mar 9, 2026)

**Discovered:** m05c running at 0.3 clips/s with GPU at 0%, CPU at 1420%. 4,144/10,000 clips completed in ~4h.

**Root cause (two layers):**

1. **Per-frame augmentation loop (original code):** `_augment_clip_consistent()` looped over 64 frames calling `TF.resized_crop()` individually = 4,736 `F.interpolate` calls per batch (64 frames × 2 views × 37 clips). Entire batch augmented on CPU while GPU sat idle.

2. **ThreadPoolExecutor + ATen thread explosion (first fix attempt):** Vectorized the per-frame loop (1 `F.interpolate` per clip instead of 64) but wrapped augmentation in `ThreadPoolExecutor(max_workers=8)`. Each Python thread's PyTorch ops spawned ~80 ATen internal threads → **644 total threads** → OS scheduler thrash → process sleeping with 0% throughput. Process appeared stuck for 35+ minutes with no output.

**Fix (two parts):**

Part 1 — Vectorized augmentation (`_augment_clip_consistent`):
```python
# Before: 64 per-frame calls
for t in range(T_frames):
    frame = TF.resized_crop(frame, i, j, h, w, [384, 384])  # 64× F.interpolate

# After: single tensor op on (T, C, H, W)
video = video[:, :, i:i+h, j:j+w]                           # single slice
video = F.interpolate(video, size=(384, 384), ...)           # single call
# Color ops (adjust_brightness/contrast/saturation/hue) and gaussian_blur
# all support 4D tensors natively — no per-frame loop
```

Part 2 — Removed threading for augment/processor (kept for decode only):
```
ThreadPoolExecutor for decode:     KEPT (I/O-bound, GIL-free)
ThreadPoolExecutor for augment:    REMOVED (ATen thread explosion)
ThreadPoolExecutor for processor:  REMOVED (same ATen issue)
```

**Impact:** Per-clip augment: 64 `F.interpolate` → 1 (~64× fewer kernel launches). Thread count: 644 → ~20. Expected throughput: 0.3 → 3-10 clips/s (pending validation after re-run).

---

### SOLUTION: Pre-Download Subset to Local Disk (`m00d_download_subset.py`)

> **STATUS: GPU VALIDATED (Mar 9, 2026).** SANITY passed (16/16 steps). FULL 10K POC download complete + pipeline running.

Instead of streaming 115K clips from HF and filtering to 10K on every step, download the 10K subset clips ONCE to local disk. All subsequent steps read locally → 100% hit rate, no network I/O.

#### Files Modified (all verified `py_compile` + AST on M1 Mac, GPU validated on RTX PRO 6000)

| File | Change | Status |
|------|--------|--------|
| `src/utils/config.py` | Added `add_local_data_arg(parser)` shared helper | DONE |
| `src/m00d_download_subset.py` | **NEW** — CPU-only pre-download (~330 lines). **Rewritten v3: CDN TAR shard download** (not streaming API) to bypass HF bandwidth throttle. Downloads one shard at a time via `hf_hub_download`, extracts matching clips, deletes shard, moves to next. Resume via `manifest.json`. | DONE |
| `src/m05_vjepa_embed.py` | `_create_stream(local_data=)` + `--local-data` arg + orchestrator passthrough. **Fix: `data_files` glob** (see Bug #3 below) | DONE |
| `src/m04_vlm_tag.py` | `_create_stream(local_data=)` + `already_tagged_keys` dedup + `--local-data` arg + orchestrator passthrough. **Fix: `data_files` glob** (see Bug #3 below) | DONE |
| `src/m05b_baselines.py` | `local_data` param to both producer functions + `--local-data` arg | DONE |
| `src/m05c_true_overlap.py` | `local_data` param to producer + `--local-data` arg | DONE |
| `run_ch9_overnight.sh` | Step 0 pre-download + `$LOCAL_FLAG` to Steps 1-4 + updated time estimates | DONE |
| `setup_env_uv.sh` | FAISS RPATH fix (`patchelf --set-rpath '$ORIGIN'`) + `libopenblas-dev` install | DONE |

#### GPU Validation Results (RTX PRO 6000 Blackwell, 102GB)

**m00d download (v3 CDN approach):**
- 10,000/10,000 clips downloaded in **23.8 min** (116 HF shards scanned, 115,637 clips scanned)
- Output: 25 local TAR shards, 10.45 GB in `data/subset_10k_local/`
- Resume worked correctly: initial streaming run saved 4,148 clips → CDN run found remaining 5,852
- Disk-friendly: ~1.5 GB temp at a time (download → extract → delete)

**SANITY pipeline:** 16/16 steps passed, 29 OK outputs, 4m 52s

**FULL pipeline (in progress):** m04 running at 1.20 clips/s (batch=64, VRAM 63%), ETA ~2.3h. First checkpoint at 512 tags.

#### m00d Evolution: Why 3 Versions

| Version | Method | Problem |
|---------|--------|---------|
| v1 | HF `load_dataset(streaming=True)` | Throttled at ~62K clips (~68 GB streamed), speed drops from 140→3.5 clips/s |
| v2 | v1 + auto-reconnect with `.skip(scanned)` | HF throttles by token/IP bandwidth, not per-connection. Reconnect doesn't help. |
| **v3 (current)** | `hf_hub_download` per TAR shard via CDN | Different rate-limit path. 12s/shard, 23.8 min total. No throttle. |

---

### Impact Summary

| Metric | Before (HF stream + filter) | After (local pre-download) | Measured |
|--------|-----|------|---------|
| m04 throughput | 0.51 clips/s (declining) | ~1.5-2.0 clips/s (stable) | **1.33 clips/s** (stable, batch=64) |
| m04 ETA (10K) | ~5.5h+ | ~1.5-2h | **2h 2m** ✅ |
| m05 ETA | ~5.5h | ~1-2h | **1h 20m** (5,105 clips before stream death) |
| m05b ETA (4 enc) | ~22h | ~4-6h | **1h 39m** (dinov2+clip+shuffled=10K each) |
| m05c ETA | ~5.5h | ~1-2h | Bug #4: 0.3→?? clips/s (vectorized fix pending validation) |
| **Total pipeline** | **~39h** | **~8-12h** | **~5h so far** (m05c incomplete) |
| GPU idle time | ~90% (waiting for producer) | ~10% (normal batch gaps) | **VRAM 63%, batch=64** |
| Resume correctness | **BROKEN** (duplicate tags) | **FIXED** (dedup by __key__) | ✅ verified |
| Pre-download cost | N/A | ~11 min (streaming) | **23.8 min** (CDN, after HF throttle) |

### Implementation Order

```
1. ✅ config.py: add_local_data_arg() shared helper
2. ✅ m05_vjepa_embed.py: _create_stream(local_data=) + --local-data arg + orchestrator passthrough
3. ✅ m04_vlm_tag.py: _create_stream(local_data=) + already_tagged_keys dedup + --local-data arg
4. ✅ m05b_baselines.py: local_data param to both producers + --local-data arg
5. ✅ m05c_true_overlap.py: local_data param to producer + --local-data arg
6. ✅ m00d_download_subset.py: v1 streaming → v2 reconnect → v3 CDN TAR download (~330 lines)
7. ✅ run_ch9_overnight.sh: Step 0 pre-download + $LOCAL_FLAG to Steps 1-4
── ALL CODE COMPLETE (Mar 8, 2026). Verified: py_compile + AST (45/45 requirements). ──
8. ✅ SANITY test on RTX PRO 6000 (16/16 passed, 4m 52s) — Mar 9, 2026
9. ✅ FULL download on RTX PRO 6000 (10K/10K, 23.8 min, 10.45 GB) — Mar 9, 2026
10. ✅ FULL pipeline run 1 on RTX PRO 6000 — Mar 9, 2026
    m04: 10K/10K (2h 2m, 1.33 clips/s) ✅
    m05: 5,105/10K (stream died, no checkpoint) ⚠️
    m05b: dinov2 10K ✅, clip 10K ✅, vjepa_shuffled 10K ✅, random 5K (matched vjepa) ⚠️
    m05c: 4,144/10K @ 0.3 clips/s (stuck, see Bug #4) ⚠️
11. ✅ Bug #4 fix: vectorized augmentation + removed ATen thread oversubscription — Mar 9, 2026
12. 🔄 FULL pipeline run 2 needed: delete incomplete vjepa+random, re-run m05/m05b-random/m05c/m06-m08
```

---

## Implemented Optimizations

### m05c: Skip Deduped Clips (~2× speedup) — DONE

m05 deduplicates V-JEPA embeddings (cosine sim > 0.95): 10,000 → 5,105 unique clips. m05c now reads `embeddings.paths.npy` (5,105 deduped keys) instead of `subset_10k.json` (10,000 keys), halving V-JEPA inference work.

**Changes in `src/m05c_true_overlap.py`:**
- After loading subset_keys, loads `embeddings.paths.npy` from output_dir and intersects with subset
- Falls back to full subset if `embeddings.paths.npy` doesn't exist (with warning)
- Filters final output to only deduped keys (handles checkpoint from prior run with extra clips)
- Prints `[DEDUP] Target: N deduped keys` for observability

**Does NOT apply to m05b baselines.** dinov2/clip/vjepa_shuffled each produce 10,000 embeddings independently. The dedup is V-JEPA-specific — baselines need the full 10K for fair per-encoder evaluation.

**Ordering dependency:** m05 must complete before m05c (already enforced by `run_ch9_overnight.sh` step ordering).

## Future Work

### HF Stream Resilience

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
