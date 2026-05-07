# 🔬 Factor-Quality Observability + DINO Recall Lift  ·  m10 / m11

> 📅 2026-05-07  ·  iter13 v13 FIX-18..21  ·  ~290 LoC across 5 files

---

## 🎯 Why this exists

🚨 **The problem:**  Visual inspection of the chennai walking-tour clip
(`VX_bCHlvFaw-171.mp4`) shows ~6 visible humans, but m10 detected only **9 agents
at 2 % pixel coverage**.  Result:  `D_A` is nearly black, `D_L` is
indistinguishable from the original, and the factor decomposition is **degenerate
on poor-recall clips**.

🎯 **Why it matters for the paper:**

> **Goal**:  `vjepa_surgery > vjepa_pretrain > vjepa_frozen` on motion / temporal features

Surgery's win condition depends on the encoder receiving **clean factor signal**.
With ~50 % agent recall, surgery's paired-Δ over frozen risks collapsing into
the 95 % CI overlap zone → no statistical separation → **no paper**.

📚 **Citations:**

- 🟢 [Disentangled World Models (2025)](https://arxiv.org/html/2503.08751) — clean factors required for transfer
- 🟡 [Locatello et al., JMLR 2020](https://www.jmlr.org/papers/volume21/19-976/19-976.pdf) — unsupervised disentanglement does **not** universally help unless factors are clean
- 🥇 [SAHI, Akyon ICIP 2022](https://arxiv.org/pdf/2203.04799) — pedestrian recall **31.8 % → 86.4 %** with sliced inference

---

## 🛠️ The four layers

| 🎚️ Layer | 🔧 What it does | 📏 LoC | ⏱️ When |
|:---:|---|---:|---|
| 🟢 **A** | Lower DINO thresholds `0.20/0.18 → 0.15/0.12` + SAM `stability_score ≥ 0.92` post-filter | ~15 | always on |
| 🟡 **B** | 🥇 SAHI tile inference for DINO (opt-in flag `--sahi-slicing`) | ~50 | A/B vs Layer A |
| 🟢 **C** | 4 free metrics in m10 → `segments.json["quality"]` + `summary.json["quality_aggregate"]` | ~150 | always on |
| 🟢 **D** | m11 propagates m10 quality + adds D_L blur-completeness + D_A signal-to-bg | ~80 | always on |

🔥 **Layer A is mandatory + free**.  🥇 **Layer B is the heavy hitter** but costs ~10× FULL m10 wall.

---

## 📊 The four metrics  (M1, M2, M5, M6  —  M3/M4 explicitly skipped)

| 🏷️ ID | 📊 Metric | 🎯 What it tests | 💰 Cost | 📄 Source |
|:---:|---|---|:---:|---|
| **M1** | 🔬 **stability_score** per mask (mean + p10/p50/p90) | Mask robustness to ±1 px / ±2 px dilation / erosion.  Meta default filter ≥ 0.92. | 🟢 free | [SAM auto_mask_generator.py](https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py) |
| **M2** | 🎯 **object_score** per mask  *(resurfaced from existing `mean_mask_confidence`)* | SAM3's per-mask object-score logits → sigmoid → [0,1] | 🟢 free | [Mask-Scoring R-CNN, Huang CVPR 2019](https://arxiv.org/abs/1903.00241) |
| **M5** | ⏱️ **temporal_iou** per object | Mean IoU(mask[t], mask[t+1]) across consecutive frames.  Stable agents > 0.7. | 🟢 free | uses `per_object` dict already in m10:449 |
| **M6** | 📐 **compactness** per mask | `4π·area / perimeter²` ∈ (0,1].  Pedestrians 0.3–0.6;  fragmented < 0.1. | 🟢 free | isoperimetric inequality |

❌ **M3** (VLM-tag-recall proxy) — user noted: m04-tag-driven prompts already tried, recall worse than current `DINO_TO_TAG`.  Skipped.

❌ **M4** (cross-detector ensemble) — costly (~2× DINO time).  Not on user's list.

---

## 📂 Change surface — file by file

### 1️⃣ `configs/train/surgery_base.yaml` ✏️ EDIT  (~10 LoC)

```yaml
factor_datasets:
  grounding_dino:
    box_threshold: 0.15        # 🔻 was 0.20 (iter13 v11) — lift recall on small pedestrians
    text_threshold: 0.12       # 🔻 was 0.18

  min_confidence: 0.3
  min_stability_score: 0.92    # ➕ NEW — Meta default for SAM auto-filter (Layer A)
  min_mask_area_pct: 0.001

  # ➕ NEW — SAHI tile inference (opt-in via --sahi-slicing CLI flag)
  sahi:
    slice_height: 640
    slice_width: 640
    overlap: 0.2
    nms_iou: 0.5
```

### 2️⃣ `src/utils/mask_metrics.py` 🆕 NEW  (~90 LoC)

Pure numpy/cv2 helpers — no new dependencies.  Functions:

- 🔬 `stability_score(mask, dilation_offsets=(1,2))` → float ∈ [0,1]  *(M1)*
- ⏱️ `temporal_iou_per_object(per_object)` → float ∈ [0,1]  *(M5)*
- 📐 `compactness(mask)` → float ∈ (0,1]  *(M6)*
- 📊 `aggregate_percentiles(values)` → `{mean, p10, p50, p90, n}`  *(used by both m10 + m11)*

### 3️⃣ `src/m10_sam_segment.py` ✏️ MODIFY  (~120 LoC)

| 📍 Where | ✏️ What |
|---|---|
| Imports | Add `from utils.mask_metrics import stability_score, temporal_iou_per_object, compactness, aggregate_percentiles` |
| `_accept_mask` (line 350) | 🟢 Compute `stab = stability_score(m)`.  Gate: `prob ≥ min_confidence AND stab ≥ min_stability_score AND area ≥ min_area`.  Track count of stability-rejected masks. |
| `segment_clip` accumulator (line 346) | Add `accepted_stability = []`, `accepted_compactness = []` lists |
| Post-tracker loop (~line 460) | Compute `temporal_iou_m5 = temporal_iou_per_object(per_object)` |
| Return dict (~line 462) | Add `quality = {stability_score: aggregate_percentiles(accepted_stability), object_score: aggregate_percentiles(accepted_probs), compactness: aggregate_percentiles(accepted_compactness), temporal_iou_m5: float, n_filtered_by_stability: int}` |
| `segments[clip_key]` (line 1037) | Add `"quality": result["quality"]` field |
| `summary` dict (line 1089) | Add `quality_aggregate = aggregate_percentiles([s["quality"]["stability_score"]["mean"] for s in segments.values()])` etc. |
| 🆕 argparse (line 770) | Add `--sahi-slicing` flag |
| DINO call (line 319) | Wrap with `if args.sahi_slicing: ... else detect_boxes_grounding_dino(...)` |

### 4️⃣ `src/m11_factor_datasets.py` ✏️ MODIFY  (~80 LoC)

| 📍 Where | ✏️ What |
|---|---|
| Imports | Add `from utils.mask_metrics import aggregate_percentiles` + `from skimage.metrics import structural_similarity as ssim` |
| `_process_one_clip` worker (line 786) | After `make_layout_only`: compute `blur_completeness = 1 - ssim(blurred_agent_region, original_agent_region)` ∈ [0,1].  Higher = more blur applied to agent pixels. |
| Same worker (line 792) | After `make_agent_only`: compute `signal_to_bg_ratio = mean(D_A on agent pixels) / max(mean(D_A on bg pixels), 1e-6)`.  Higher = cleaner agent isolation (≥ 3.0 = good). |
| Return entry (line 835) | Add `"m10_quality": seg_entry.get("quality")`, `"quality_l": {"blur_completeness": …}`, `"quality_a": {"signal_to_bg_ratio": …}` |
| End of main (after line 1144) | Compute aggregate over all manifest entries' D_L/D_A metrics → write `factor_manifest_quality.json` |

### 5️⃣ `requirements_gpu.txt` ✏️ EDIT  (Layer B only)

```diff
+ sahi>=0.11.0          # iter13 v13 FIX-18: opt-in tile inference (DINO recall lift)
+ scikit-image>=0.22.0  # iter13 v13 FIX-18: ssim() for D_L blur-completeness
```

---

## 🧩 Reused helpers  (don't reinvent)

| 🔧 Function | 📂 Path | ♻️ Why reuse |
|---|---|---|
| `per_object[obj_id][t] = mask` | `m10_sam_segment.py:449` | Already populated by SAM3 tracker — direct M5 input |
| `mean_mask_confidence` | `m10_sam_segment.py:1037` | Already computed — direct M2 source |
| `agent_pixel_ratio` | `m10_sam_segment.py:1037` | Already in segments.json — kept for back-compat |
| `load_train_config_with_extends` | `utils/config.py:130` (FIX-10) | Resolves new `sahi:` block from yaml inheritance |
| `save_json_checkpoint` | `utils/checkpoint.py` | Atomic write for new `factor_manifest_quality.json` |
| `make_layout_only` (blur path) | `m11_factor_datasets.py:82-100` | Hook D_L blur-completeness here |
| `make_agent_only` (matte path) | `m11_factor_datasets.py:106-124` | Hook D_A signal-to-bg-ratio here |

---

## ⚠️ Risks + 🩹 mitigations

| 🚨 Risk | 🩹 Mitigation |
|---|---|
| 🔻 Lower DINO thresholds (0.15/0.12) flood with false positives | 🛡️ SAM `stability_score ≥ 0.92` post-filter sweeps weak masks.  SANITY A/B confirms net recall gain on chennai walking clip. |
| 🔪 SAHI breaks SAM3 tracker (overlapping tile boxes confuse propagation) | 🛡️ NMS merge at `sahi.nms_iou: 0.5` before passing to SAM3.  Tracker sees deduplicated unique boxes. |
| 📦 `HuggingfaceDetectionModel` may not support DINO text prompts | 🛡️ Fallback: manual tile loop (~30 LoC explicit tiling + NMS) |
| 🐌 SSIM on 384×384 frames | 🛡️ ~1 ms/frame · 16 frames/clip = 16 ms.  Negligible. |
| 📊 segments.json bloat from new `quality` block | 🛡️ ~10 floats/clip × 10K × 8 bytes = **800 KB**.  Trivial. |
| 📐 compactness ≈ 0.7 always (m10 dilates by 4 px) | 🛡️ Compute on **raw** SAM mask (in `per_object`) BEFORE dilation.  Current code keeps raw masks. |
| 💾 Schema change → cache invalidation | 🛡️ Schema is **additive** (new keys only).  No cache wipe required. |

---

## ✅ Verification ladder

### 🪜 Step 1 — 3-check after each edit
```bash
python -m py_compile src/utils/mask_metrics.py src/m10_sam_segment.py src/m11_factor_datasets.py
ruff check --select F,E9 src/utils/mask_metrics.py src/m10_sam_segment.py src/m11_factor_datasets.py
```

### 🪜 Step 2 — SANITY (Layers A + C + D, no SAHI)
```bash
CACHE_POLICY_ALL=2 ./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml --SANITY \
  2>&1 | tee logs/sanity_factor_quality_v1.log
```
✅ **Expect:**
- 🔬 m10 log: `[m10] stability_score filter: kept N/M masks (rejected K below 0.92)`
- 📁 `data/eval_10k_local/m10_sam_segment/segments.json` has new `"quality": {…}` per clip
- 📁 `data/eval_10k_local/m10_sam_segment/summary.json` has `"quality_aggregate"`
- 📁 `data/eval_10k_local/m11_factor_datasets/factor_manifest.json` entries have `"m10_quality"` + `"quality_l"` + `"quality_a"`
- 📁 `data/eval_10k_local/m11_factor_datasets/factor_manifest_quality.json` written
- 🔢 Numerical: `stability ∈ [0,1]`, `temporal_iou ∈ [0,1]`, `compactness ∈ [0,1]`, `signal_to_bg_ratio > 0`

### 🪜 Step 3 — 👀 Visual regression on chennai walking clip
```bash
xdg-open data/eval_10k_local/m10_sam_segment/m10_overlay_verify_top20/tier1__chennai__walking__VX_bCHlvFaw__VX_bCHlvFaw-171.mp4.png
```
✅ **Expect**: agent count jumps from 9 → **15-20+** (recovery from 0.20 → 0.15 box threshold).

### 🪜 Step 4 — 🅰️ A/B  SAHI on / SAHI off
```bash
# baseline (Layer A + C + D, no SAHI)
CACHE_POLICY_ALL=2 ./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml --SANITY 2>&1 \
  | tee logs/sanity_factor_no_sahi.log
N_NO_SAHI=$(jq '[.[].n_agents] | add' data/eval_10k_local/m10_sam_segment/segments.json)

# with SAHI
CACHE_POLICY_ALL=2 ./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml --SANITY --sahi-slicing 2>&1 \
  | tee logs/sanity_factor_sahi.log
N_SAHI=$(jq '[.[].n_agents] | add' data/eval_10k_local/m10_sam_segment/segments.json)

echo "n_agents lift: $N_NO_SAHI → $N_SAHI"
```
✅ **Expect**: SAHI run ≥ **1.5×** more agents on average.  Decide if ~10× FULL wall cost is worth it.

### 🪜 Step 5 — 🚀 FULL gate  (deferred)
After A/B verdict, re-run FULL m10 with chosen config.  Expected wall: 1 hr (no SAHI) vs ~10 hr (with SAHI).

---

## 📏 LoC budget

| 📂 File | ➕ Added | ➖ Removed | 📐 Net |
|---|---:|---:|---:|
| `src/utils/mask_metrics.py` 🆕 | ~90 | 0 | **+90** |
| `src/m10_sam_segment.py` | ~120 | ~5 | **+115** |
| `src/m11_factor_datasets.py` | ~80 | ~5 | **+75** |
| `configs/train/surgery_base.yaml` | ~10 | ~2 | **+8** |
| `requirements_gpu.txt` | +2 | 0 | **+2** |
| **Total** | ~302 | ~12 | **+290** |

---

## 🚫 Out of scope  (explicit non-goals)

- ❌ **M3 VLM-tag-recall proxy** — m04 tag-driven prompts tried, recall worse than current `DINO_TO_TAG`
- ❌ **M4 cross-detector ensemble** — costly (~2× DINO time)
- ❌ **Multi-frame DINO union** (Rank 5) — m10 already runs DINO on 4 anchor frames `[0,4,8,12]`
- ❌ **Grounding DINO 1.5 Pro** (Rank 6) — defer until Layer A/B/C/D insufficient
- ❌ **m04 VLM-tag-aware dynamic prompts** — explicitly retired (recall worse than hardcoded)
- ❌ **Visualization of quality scores in overlay PNGs** — defer; JSON suffices

---

## 🗺️ Implementation order  (~5 hours dev + 1 hour SANITY)

| 🎚️ # | 📋 Task | ⏱️ Est | 🎯 Validates |
|:---:|---|---:|---|
| 1 | 📂 yaml threshold edit + add `min_stability_score` + `sahi:` block | 10 min | Layer A config |
| 2 | 🆕 `src/utils/mask_metrics.py` — 4 helpers | 30 min | Layer C foundation |
| 3 | ✏️ m10 stability_score filter + quality block + summary aggregate | 90 min | Layers A + C |
| 4 | ✏️ m11 propagation + D_L/D_A metrics + factor_manifest_quality.json | 60 min | Layer D |
| 5 | 🥇 SAHI wrapper + `--sahi-slicing` flag + requirements update | 90 min | Layer B |
| 6 | 🪜 SANITY verification + 👀 visual regression on chennai clip | 30 min | end-to-end |
| 7 | 🅰️ A/B SAHI on/off + lock final config for FULL | 15 min | decision |

---

## 🔥 The bottom line

> Higher-quality factor decomposition translates **directly** to surgery's
> paired-Δ magnitude.  This plan adds **observability** (so degeneracy is
> measurable per-clip) **and** **two recall interventions** (lower thresholds
> + opt-in SAHI) so the chennai-walking failure mode becomes both visible and
> fixable.

**🎯 Net effect: surgery sees clean factor signal → win condition reachable.**
