# Pipeline Improvement Plan

> **Goal**: Improve object detection quality and instruction generation for 3D scene understanding

---

## Table of Contents

1. [🔍 Current Pipeline](#1-current-pipeline)
2. [❌ Problems Identified](#2-problems-identified)
3. [✅ Proposed Pipeline v2](#3-proposed-pipeline-v2)
4. [🤖 VLM Candidates Comparison](#4-vlm-candidates-comparison)
5. [⚔️ Grounded-SAM 2 vs VLM Comparison](#5-grounded-sam-2-vs-vlm-comparison)
6. [📊 Evaluation Pipeline](#6-evaluation-pipeline)
7. [🔧 Code Changes Required](#7-code-changes-required)
8. [💰 Cost Analysis](#8-cost-analysis)

---

## 1. 🔍 Current Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      CURRENT PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   _isometric.png          _topdown.png                          │
│        │                       │                                │
│        └───────────┬───────────┘                                │
│                    ▼                                            │
│         ┌──────────────────────┐                                │
│         │  LLaVA-1.5-7B (4-bit)│  ◄── BOTTLENECK                │
│         │  SceneUnderstanding  │                                │
│         └──────────┬───────────┘                                │
│                    ▼                                            │
│     Scene Description + OBJECT_LIST                             │
│     (often returns "unknown")                                   │
│                    │                                            │
│                    ▼                                            │
│         ┌──────────────────────┐                                │
│         │  Llama-3-8B-Instruct │                                │
│         │  (HF Router API)     │                                │
│         └──────────┬───────────┘                                │
│                    ▼                                            │
│         Robot Navigation Tasks                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. ❌ Problems Identified

| # | Problem | Impact |
|---|---------|--------|
| ❌ 1 | LLaVA-1.5-7B + 4-bit quantization | Poor object detection (1/7 benchmark tests) |
| ❌ 2 | Regex extraction of `OBJECT_LIST` | Brittle parsing, fails silently |
| ❌ 3 | 512x512 image resize | Loses small objects and details |
| ❌ 4 | No validation/evaluation | Can't measure quality |
| ❌ 5 | "unknown" objects cascade | Useless instructions generated |

### Evidence from Current Output

```
# ✅ Good output (34_Pedestrian_mall.4.txt)
START: Near table | END: Near chair | INTERACT: person, table

# ❌ Bad output (35_Bus_loop.3.txt)
START: Near unknown | END: Near unknown | INTERACT: unknown, unknown
```

---

## 3. ✅ Proposed Pipeline v2

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROPOSED PIPELINE v2                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   _isometric.png              _topdown.png                      │
│   (1024x1024)                 (1024x1024)    ◄── Higher res     │
│        │                           │                            │
│        └───────────┬───────────────┘                            │
│                    ▼                                            │
│         ┌──────────────────────┐                                │
│         │  Qwen3-VL-32B        │  ◄── Dense: Full 32B active    │
│         │  (8-bit, ~40GB VRAM) │      Best quality              │
│         └──────────┬───────────┘                                │
│                    ▼                                            │
│         JSON Output:                                            │
│         {                                                       │
│           "objects": [                                          │
│             {"name": "bench", "confidence": 0.95},              │
│             {"name": "tree", "confidence": 0.92}                │
│           ],                                                    │
│           "scene_type": "outdoor_mall"                          │
│         }                                                       │
│                    │                                            │
│                    ▼                                            │
│         ┌──────────────────────┐                                │
│         │  Object Validation   │  ◄── Filter conf < 0.7         │
│         │  + Deduplication     │                                │
│         └──────────┬───────────┘                                │
│                    ▼                                            │
│                    │                                            │
│         ┌──────────┴──────────┐                                 │
│         │  🔄 UNLOAD VLM      │  ◄── Sequential loading         │
│         │  (free ~19GB VRAM)  │      (A100-80GB constraint)     │
│         └──────────┬──────────┘                                 │
│                    ▼                                            │
│         ┌─────────────────────────────────┐                     │
│         │  Llama-3.1-70B-Instruct (8-bit) │  ◄── 128K context   │
│         │  (~70GB VRAM)                   │                     │
│         └──────────┬──────────────────────┘                     │
│                    ▼                                            │
│         Robot Navigation Tasks                                  │
│                    │                                            │
│                    ▼                                            │
│         ┌──────────────────────┐                                │
│         │  EVALUATION PIPELINE │  ◄── NEW                       │
│         │  (VLM-as-Judge)      │                                │
│         └──────────────────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 💾 Memory Management: Sequential Loading

Since both models exceed A100-80GB when loaded together:

| Step | Action | VRAM Used |
|------|--------|-----------|
| 1️⃣ | Load Qwen3-VL-32B (8-bit) | ~40GB |
| 2️⃣ | Process images → JSON output | ~40GB |
| 3️⃣ | **Unload VLM** (`del model; torch.cuda.empty_cache()`) | ~0GB |
| 4️⃣ | Load Llama-3.1-70B-Instruct (8-bit) | ~70GB |
| 5️⃣ | Generate instructions from text | ~70GB |

**Why Llama-3.1-70B instead of Llama-3-72B?**

| Model | Context Window | VRAM (8-bit) |
|-------|----------------|--------------|
| ❌ Llama-3-72B | 8K tokens | ~72GB |
| ✅ Llama-3.1-70B | **128K tokens** | ~70GB |

Llama-3.1 has 16x longer context + slightly smaller footprint.

---

## 4. 🤖 VLM Candidates Comparison

### Option A: Vision-Language Models (VLMs)

| Model | Params | Active | VRAM (bf16) | Object Detection | Strength |
|-------|--------|--------|-------------|------------------|----------|
| **LLaVA-1.5-7B** (current) | 7B | 7B | ~14GB | ❌ Poor (1/7) | Baseline |
| **Qwen2.5-VL-7B** | 7B | 7B | ~29GB | ✅ Good | Best accuracy/VRAM |
| **Gemma-3-27B-it** | 27B | 27B | ~54GB | ✅ Good | 140+ languages |
| **Qwen3-VL-30B-A3B** | 30B | 3B | ~19GB (Q4) | ✅ Very Good | MoE efficiency |
| **Qwen3-VL-32B** 🔥 | 32B | **32B** | ~40GB (8-bit) | ✅ Excellent | Best quality |

### Recommendation: **Qwen3-VL-32B (8-bit)** 🔥

**Why?**
- ✅ Dense model: Full 32B parameters active (not sparse MoE)
- ✅ Fits on A100-80GB with 8-bit quantization (~40GB VRAM)
- ✅ Best benchmark scores for object detection
- ✅ Native JSON + bounding box output

---

## 5. ⚔️ Grounded-SAM 2 vs VLM Comparison

### Option B: Grounded-SAM 2 Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  GROUNDED-SAM 2 PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Image Input                                                   │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────┐                                           │
│   │ Grounding DINO  │  ◄── Open-set object detector             │
│   │ (text prompts)  │      "detect: bench, tree, person"        │
│   └────────┬────────┘                                           │
│            │ bounding boxes                                     │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │     SAM 2       │  ◄── Segment Anything Model 2             │
│   │  (segmentation) │      Precise masks for each object        │
│   └────────┬────────┘                                           │
│            │ masks + labels                                     │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │  VLM (optional) │  ◄── Still needed for scene description   │
│   │  for description│                                           │
│   └────────┬────────┘                                           │
│            ▼                                                    │
│   Objects + Masks + Scene Description                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Option C: SAM 3D Pipeline (if 3D reconstruction needed)

```
┌─────────────────────────────────────────────────────────────────┐
│                      SAM 3D PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Single 2D Image                                               │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────┐                                           │
│   │     SAM 3D      │  ◄── Meta's 3D reconstruction model       │
│   │   Objects       │      Outputs: 3D mesh + pose + texture    │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   3D Mesh with spatial relationships                            │
│   (useful for robot navigation planning)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Head-to-Head Comparison

| Criteria | VLM Only (Qwen3-VL-32B) 🔥 | Grounded-SAM 2 | SAM 3D |
|----------|---------------------|----------------|--------|
| **Object Detection** | ✅ Excellent | ✅ Excellent (48.7 mAP) | ⚠️ N/A (reconstruction) |
| **Semantic Labels** | ✅ Yes (native) | ✅ Yes (Grounding DINO) | ⚠️ Limited |
| **Bounding Boxes** | ✅ Yes | ✅ Yes | ❌ N/A |
| **Segmentation Masks** | ❌ No | ✅ Yes (precise) | ✅ 3D mesh |
| **Scene Description** | ✅ Yes (native) | ❌ No (needs VLM) | ❌ No |
| **3D Understanding** | ⚠️ Inferred from 2 views | ❌ No | ✅ Yes (from single image) |
| **Pipeline Complexity** | ✅ Single model | ⚠️ 2-stage pipeline | ✅ Single model |
| **VRAM Required** | ✅ ~40GB (8-bit) | ⚠️ ~12GB + ~8GB | ✅ ~16GB |
| **Inference Speed** | ⚠️ Moderate | ✅ Fast (real-time) | ✅ 30ms on H200 |
| **For Your Use Case** | ✅ **Recommended** | ⚠️ Overkill | 🔮 Future work |

### Why NOT Grounded-SAM 2 for this project?

| Reason | Explanation |
|--------|-------------|
| ❌ **Masks not needed** | You need object names for instructions, not pixel-perfect masks |
| ❌ **Two-stage complexity** | Grounding DINO → SAM 2 adds pipeline complexity |
| ❌ **No scene description** | Still need a VLM for "outdoor mall with S-shaped walkway" |
| ❌ **Research shows poor combo** | LLaVA+SAM2 achieved only 0.291 mIoU (2025 study) |

### When WOULD you use Grounded-SAM 2? 🎯

- ✅ If you need precise object boundaries for robot grasping
- ✅ If you're doing instance segmentation for counting
- ✅ If you need to track objects across video frames

### When WOULD you use SAM 3D? 🔮 (Future Work)

- ✅ If you need actual 3D reconstruction from single images
- ✅ If robot needs to plan paths in 3D space
- ✅ Future enhancement: combine 3D mesh with navigation planning

**Potential SAM 3D Applications (Out of Scope for Now):**

```
┌─────────────────────────────────────────────────────────────────┐
│  FUTURE: SAM 3D Integration                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Option A: Generate 3D from 2D images                           │
│  ─────────────────────────────────────                          │
│  _isometric.png → SAM 3D → 3D mesh → Navigation simulation      │
│                                                                 │
│  Option B: Navigate existing 3D environments                    │
│  ─────────────────────────────────────────────                  │
│  3D environment → SAM 3D (object extraction) → Path planning    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

| Use Case | SAM 3D Role | Benefit |
|----------|-------------|---------|
| 🔮 2D→3D reconstruction | Generate mesh from `_isometric.png` | True spatial reasoning |
| 🔮 3D navigation | Extract objects with pose/depth | Collision-aware paths |

> ⚠️ **Current Focus**: Improve instruction quality from existing 2D image pairs first. SAM 3D integration is Phase 2.

---

## 6. 📊 Evaluation Pipeline

### Metrics

#### Metric 1: Object Grounding Score

```python
score = (instructions_with_valid_objects) / (total_instructions)

# Current: 35_Bus_loop.3 → 0% (all "unknown")
# Target: > 90%
```

#### Metric 2: Instruction Quality (VLM-as-Judge)

| Criterion | Score (1-5) | Question |
|-----------|-------------|----------|
| 🎯 Object Accuracy | 1-5 | Are objects visible in image? |
| 🗺️ Spatial Coherence | 1-5 | Does path make sense? |
| 📝 Task Clarity | 1-5 | Is instruction unambiguous? |
| 📊 Difficulty Alignment | 1-5 | Does complexity match level? |
| 🤖 Executability | 1-5 | Could robot perform this? |

#### Metric 3: Object Detection Recall

```python
# With ground truth:
recall = len(detected ∩ GT) / len(GT)

# Without ground truth (cross-model consensus):
# Run 3 VLMs, objects detected by ≥2 = pseudo-GT
```

### Hybrid Evaluation Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                   EVALUATION STRATEGY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: Baseline (N=100)                                      │
│  ├── Judge: GPT-4o API                                          │
│  ├── Cost: ~$1-2                                                │
│  └── Purpose: Establish gold standard scores                    │
│                                                                 │
│  PHASE 2: Calibration                                           │
│  ├── Run Prometheus-Vision-13B on same 100 samples              │
│  ├── Compute Pearson correlation with GPT-4o                    │
│  └── If r > 0.75, proceed to scale                              │
│                                                                 │
│  PHASE 3: Scale (N=10K+)                                        │
│  ├── Judge: Prometheus-Vision-13B (self-hosted)                 │
│  ├── Cost: ~$40-75 on A100                                      │
│  └── Spot-check 1% with GPT-4o                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Output Format

```json
{
  "scene_id": "34_Pedestrian_mall.4",
  "object_grounding_score": 0.92,
  "instruction_quality": {
    "level_1": {"avg": 4.2, "scores": [4, 5, 4, 4, 4]},
    "level_2": {"avg": 3.8, "scores": [4, 3, 4, 4, 4]},
    "level_3": {"avg": 3.5, "scores": [3, 4, 3, 4, 3]}
  },
  "detected_objects": ["bench", "tree", "kiosk"],
  "judge_model": "prometheus-vision-13b"
}
```

---

## 7. 🔧 Code Changes Required

| Component | Current | Proposed | Change |
|-----------|---------|----------|--------|
| VLM Model | ❌ `llava-hf/llava-1.5-7b-hf` | ✅ `Qwen/Qwen3-VL-32B-Instruct` (8-bit) | 🔄 Replace |
| Quantization | ❌ 4-bit NF4 | ✅ bf16 | 🔄 Remove BitsAndBytesConfig |
| Image Size | ❌ 512x512 | ✅ 1024x1024 | 🔄 Increase |
| Object Extraction | ❌ Regex `OBJECT_LIST:` | ✅ Native JSON | 🔄 Rewrite prompt |
| Validation | ❌ None | ✅ Confidence filter + dedup | ➕ Add class |
| Evaluation | ❌ None | ✅ VLM-as-Judge | ➕ Add module |

### Code Comparison

```python
# BEFORE (current)
class SceneUnderstanding:
    model = "llava-hf/llava-1.5-7b-hf"
    quantization = "4-bit"
    output = "free-form text + regex OBJECT_LIST"

# AFTER (proposed)
class SceneUnderstanding:
    model = "Qwen/Qwen3-VL-32B-Instruct"
    quantization = "8-bit"  # ~40GB VRAM, full 32B active
    output = "structured JSON with bbox + confidence"

# NEW (evaluation)
class InstructionEvaluator:
    baseline_judge = "gpt-4o"              # 100 samples
    scale_judge = "prometheus-vision-13b"  # 10K+ samples
    metrics = ["object_grounding", "instruction_quality", "detection_recall"]
```

---

## 8. Cost Analysis 💰

### Object Detection (per 10K images)

| Method | VRAM | Time | Cost |
|--------|------|------|------|
| ❌ LLaVA-1.5-7B (current) | 4GB | ~5 hrs | ~$8 |
| ✅ Qwen3-VL-32B (8-bit) 🔥 | 40GB | ~12 hrs | ~$24 |
| ⚠️ Grounded-SAM 2 | 20GB | ~8 hrs | ~$16 |

### Evaluation (per 10K images)

| Method | Cost |
|--------|------|
| ⚠️ GPT-4o API only | $80 |
| ✅ GPT-4o-mini API | $5 |
| ✅ Prometheus-13B (A100) | $40-75 |
| ✅ Hybrid (recommended) 🔥 | ~$50 |

---

## Summary: Recommended Architecture 🏆

```
┌─────────────────────────────────────────────────────────────────┐
│              FINAL RECOMMENDED PIPELINE  🔥                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Object Detection:     Qwen3-VL-32B (8-bit, ~40GB)   ✅         │
│  Instruction Gen:      Llama-3.1-70B-Instruct (8-bit)✅         │
│  Memory Strategy:      Sequential loading (unload→load)✅       │
│  Evaluation Baseline:  GPT-4o (100 samples)          ✅         │
│  Evaluation Scale:     Prometheus-Vision-13B (10K+)  ✅         │
│  Hardware:             A100-80GB (sufficient for all)✅         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Decision Matrix

| Approach | Object Detection | Complexity | Cost | Recommendation |
|----------|------------------|------------|------|----------------|
| ✅ VLM (Qwen3-VL-32B) 🔥 | ✅ Excellent | ✅ Low | ✅ Low | ✅ **Recommended** |
| ⚠️ Grounded-SAM 2 | ✅ Excellent | ⚠️ Medium | ⚠️ Medium | ⚠️ Overkill |
| 🔮 SAM 3D | ⚠️ N/A (3D recon) | ✅ Low | ✅ Low | 🔮 Future work |
| ❌ Hybrid (VLM + SAM) | ✅ Excellent | ❌ High | ⚠️ Medium | ❌ Not recommended |
