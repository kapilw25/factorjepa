# WalkIndia-50 POC (WITH FAISS + Qwen3-VL)

---

## Progress Status

| Module | Status | Location |
|--------|--------|----------|
| `m01_download.py` | ✅ DONE | M1 Mac |
| `m02_scene_detect.py` | ✅ DONE | M1 Mac |
| `m02b_upload_hf.py` | ✅ DONE | M1 Mac |
| `m03_vjepa_embed.py` | ⏳ PENDING | GPU Server |
| `m04_qwen_tag.py` | ⏳ PENDING | GPU Server |
| `m05_faiss_metrics.py` | ⏳ PENDING | GPU Server |
| `m06_umap_plot.py` | ⏳ PENDING | M1 Mac (after GPU) |

**Dataset**: https://huggingface.co/datasets/anonymousML123/walkindia-50-clips (337 clips uploaded)

---

## Pipeline Diagram

```
════════════════════════════════════════ WalkIndia-200K Pipeline ════════════════════════════════════════

M1 MAC (CPU)                          │ GPU SERVER (Nvidia CUDA)                    │ M1 MAC (CPU)
──────────────────────────────────────┼─────────────────────────────────────────────┼──────────────────
                                      │                                             │
[m01]─►[m02]─►[m02b]─►HuggingFace────►│─►clips(337)─┬─►[m03]─►embeddings.npy──┬────►│─►[m06]─►umap.png
YouTube  scene  upload                │             │   VJEPA    (N×768)      │     │   UMAP   umap.pdf
  (3)    detect                       │             │                         ▼     │
   │       │                          │             │                    [m05]──────│
   ▼       ▼                          │             │                    FAISS      │
videos  clips                         │             │                       │       │
 (3)    (337)                         │             └─►[m04]─►tags.json─────┤       │
                                      │                Qwen3VL              │       │
                                      │                                     ▼       │
                                      │            ◄───PARALLEL───►    metrics.json │
                                      │                                             │

EXECUTION: m01 → m02 → m02b → [m03 ∥ m04] → m05 → m06
```

## Module I/O Details

| Module | Input | Output | Notes |
|--------|-------|--------|-------|
| `m01_download` | YouTube URLs (3) | `data/videos/*.mp4` (3 videos, 10min ea) | yt-dlp, CPU |
| `m02_scene_detect` | `data/videos/*.mp4` | `data/clips/**/*.mp4` (337 clips, 4-5s ea) | PySceneDetect, CPU |
| `m02b_upload_hf` | `data/clips/**/*.mp4` | HuggingFace Dataset | huggingface_hub, CPU |
| `m03_vjepa_embed` | `data/clips/**/*.mp4` | `data/embeddings.npy` (N×768), `data/embeddings.paths.npy` | V-JEPA 2, dedupe, **GPU** |
| `m04_qwen_tag` | `data/clips/**/*.mp4` | `data/tags.json` | Qwen3-VL-8B, **GPU** |
| `m05_faiss_metrics` | `embeddings.npy` + `tags.json` | `outputs/metrics.json` | FAISS kNN, **GPU** |
| `m06_umap_plot` | `embeddings.npy` + `tags.json` + `metrics.json` | `outputs/poc_umap.png`, `outputs/poc_umap.pdf` | UMAP 2D, CPU |

### Output File Formats

**embeddings.npy** (m03)
```
shape: (N, 768)  # N = unique clips after deduplication
dtype: float32
```

**tags.json** (m04)
```json
[
  {
    "clip_path": "data/clips/temple/clip001.mp4",
    "scene_type": "temple|market|junction|lane|highway|residential|commercial",
    "crowd_density": "low|med|high",
    "traffic_density": "low|med|high",
    "time_of_day": "morning|afternoon|evening|night",
    "weather": "clear|cloudy|rain|fog",
    "notable_objects": ["pedestrian", "vehicle", ...]
  }
]
```

**metrics.json** (m05)
```json
{
  "self_consistency": 72.5,
  "cluster_purity": 65.3,
  "k_neighbors": 6,
  "num_clips": 337,
  "pass": true
}
```

---

# PART 1: EXECUTION PLAN

## Terminal Commands

### Step 1: M1 Macbook (CPU/API) - Data Preparation ✅
```bash
# Download 3 videos from @walkinginindia YouTube
python -u src/m01_download.py --SANITY 2>&1 | tee logs/m01_download_sanity.log
python -u src/m01_download.py --FULL 2>&1 | tee logs/m01_download_full.log

# Split videos into 4-5s clips using PySceneDetect
python -u src/m02_scene_detect.py --SANITY 2>&1 | tee logs/m02_scene_detect_sanity.log
python -u src/m02_scene_detect.py --FULL 2>&1 | tee logs/m02_scene_detect_full.log

# Upload clips to HuggingFace (for GPU server access)
python -u src/m02b_upload_hf.py --SANITY 2>&1 | tee logs/m02b_upload_hf_sanity.log
python -u src/m02b_upload_hf.py --FULL 2>&1 | tee logs/m02b_upload_hf_full.log
```

### Step 2: Nvidia GPU Server - Inference ⏳
```bash
# Generate V-JEPA embeddings (requires CUDA)
python -u src/m03_vjepa_embed.py --SANITY 2>&1 | tee logs/m03_vjepa_embed_sanity.log
python -u src/m03_vjepa_embed.py --FULL 2>&1 | tee logs/m03_vjepa_embed_full.log

# Generate Qwen3-VL tags (requires CUDA)
python -u src/m04_qwen_tag.py --SANITY 2>&1 | tee logs/m04_qwen_tag_sanity.log
python -u src/m04_qwen_tag.py --FULL 2>&1 | tee logs/m04_qwen_tag_full.log

# Compute FAISS metrics (requires CUDA)
python -u src/m05_faiss_metrics.py --SANITY 2>&1 | tee logs/m05_faiss_metrics_sanity.log
python -u src/m05_faiss_metrics.py --FULL 2>&1 | tee logs/m05_faiss_metrics_full.log
```

### Step 3: M1 Macbook (CPU) - Visualization ⏳
```bash
# Generate UMAP plot
python -u src/m06_umap_plot.py --SANITY 2>&1 | tee logs/m06_umap_plot_sanity.log
python -u src/m06_umap_plot.py --FULL 2>&1 | tee logs/m06_umap_plot_full.log
```

---

## Module Summary

| Module | Purpose | GPU Requirement | Status |
|--------|---------|-----------------|--------|
| `m01_download.py` | yt-dlp download | CPU/API (M1 OK) | ✅ |
| `m02_scene_detect.py` | PySceneDetect split | CPU (M1 OK) | ✅ |
| `m02b_upload_hf.py` | HuggingFace upload | CPU/API (M1 OK) | ✅ |
| `m03_vjepa_embed.py` | V-JEPA embeddings | **Nvidia GPU only** | ⏳ |
| `m04_qwen_tag.py` | Qwen3-VL tagging | **Nvidia GPU only** | ⏳ |
| `m05_faiss_metrics.py` | FAISS + metrics | **Nvidia GPU only** | ⏳ |
| `m06_umap_plot.py` | UMAP visualization | CPU (M1 OK) | ⏳ |

---

## Directory Structure

```
src/
├── m01_download.py      # ✅
├── m02_scene_detect.py  # ✅
├── m02b_upload_hf.py    # ✅
├── m03_vjepa_embed.py   # ⏳
├── m04_qwen_tag.py      # ⏳
├── m05_faiss_metrics.py # ⏳
├── m06_umap_plot.py     # ⏳
├── utils/
│   ├── __init__.py
│   └── config.py        # shared utilities
├── data/
│   ├── videos/          # 3 downloaded videos
│   ├── clips/           # 337 clips (4-5s each)
│   ├── embeddings.npy   # 337 x 768 (pending)
│   └── tags.json        # Qwen3-VL structured tags (pending)
└── outputs/
    ├── poc_umap.png     # UMAP visualization
    ├── poc_umap.pdf     # UMAP visualization (PDF)
    └── metrics.json     # Self-Consistency, Cluster Purity
```

---

## Success Criteria

```
POC PASSES IF:
├── Self-Consistency > 60%
├── Cluster Purity > 50%
├── UMAP shows visible clustering by scene_type
└── Qwen3-VL tags look reasonable (spot-check 10 clips)

THEN → Proceed with full WalkIndia-200K
ELSE → Debug: Is V-JEPA bad? Or Qwen3-VL tags wrong?
```

---
---

# PART 2: SYSTEM DESIGN PLAN

## Architecture Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                         POC: WalkIndia-50 (4 Hours)                                               ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   YouTube    │       │ PySceneDetect│       │   ~50 clips  │
│   3 videos   │ ════► │  (4-5s cuts) │ ════► │   (4-5s)     │ ════════════════════════════════════════════╗
│   10 min ea  │       │              │       │              │                                             ║
└──────────────┘       └──────────────┘       └──────────────┘                                             ║
                                                                                                           ▼
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╩════════╗
║                                              PARALLEL PROCESSING                                                  ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                   ║
║  V-JEPA BRANCH:                                                                                                   ║
║  ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐            ║
║  │    V-JEPA 2      │       │      FAISS       │       │ Self-Consistency │       │ UMAP + matplotlib│            ║
║  │  clip ➔ 64 frm   │ ════► │   IndexFlatL2    │ ════► │   %              │ ════► │   scatter plot   │ ═══════╗  ║
║  │  ➔ ViT-L (frozen)│       │   (exact kNN)    │       │   Cluster Purity │       │   colored by tag │       ║  ║
║  └──────────────────┘       └──────────────────┘       └──────────────────┘       └──────────────────┘       ║  ║
║                                                                                                              ║  ║
║  QWEN3-VL BRANCH:                                                                                            ║  ║
║  ┌───────────────────────────────────────────────┐       ┌───────────────────────────────────────┐           ║  ║
║  │           Qwen3-VL-8B (via API/local)         │       │         Structured Tags JSON          │           ║  ║
║  │  • scene_type: market|temple|junction|...     │ ════► │         tags.json (50 clips)          │ ══════════╣  ║
║  │  • crowd_density: low|med|high                │       │                                       │           ║  ║
║  │  • time_of_day: morning|afternoon|evening     │       │                                       │           ║  ║
║  └───────────────────────────────────────────────┘       └───────────────────────────────────────┘           ║  ║
║                                                                                                              ║  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╩══╝
                                                                                                               ║
                                                                                                               ▼
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                POC OUTPUT                                                         ║
╠═════════════════════════╦═════════════════════════════════════════════════════════════════════════════════════════╣
║ 1. UMAP plot            ║ poc_umap.png (colored by Qwen3-VL scene_type)                                           ║
║ 2. Self-Consistency %   ║ Single number: XX% (target > 60%)                                                       ║
║ 3. Cluster Purity %     ║ % of kNN neighbors with same scene_type (target > 50%)                                  ║
║ 4. tags.json            ║ Qwen3-VL structured tags for all 50 clips                                               ║
║ 5. Answer               ║ "V-JEPA clusters Indian scenes" → YES / NO                                              ║
╚═════════════════════════╩═════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## What's IN (Full POC with FAISS + Qwen)

| Component | Status | Notes |
|-----------|--------|-------|
| YouTube videos | ✅ 3 videos | 1 market, 1 temple, 1 junction |
| PySceneDetect | ✅ | Split to 4-5s clips |
| V-JEPA 2 | ✅ | `facebook/vjepa2-vitl-fpc64-256` |
| **FAISS** | ✅ | `IndexFlatL2` (GPU-accelerated) |
| **Qwen3-VL** | ✅ | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Self-Consistency | ✅ | Label-free metric |
| Cluster Purity | ✅ | Uses Qwen3-VL tags |
| UMAP | ✅ | matplotlib scatter |

---

## What's OUT (Skip for POC)

| Component | Why Skip |
|-----------|----------|
| FiftyOne | Setup overhead, matplotlib enough |
| VideoLLaMA3 | Cross-VLM not needed for POC |
| InternVL2.5 | Cross-VLM not needed for POC |
| Transform Stability | Extra augmentation processing |
| Recall@k | Cluster Purity is simpler |

---

## Key Algorithms

### Self-Consistency Metric
```
For each clip A:
  1. Find nearest neighbor B = kNN(A)[1]
  2. Check if A is in kNN(B)
  3. Count consistent pairs

Self-Consistency % = consistent / total × 100
Target: > 60%
```

### Cluster Purity Metric
```
For each clip A with scene_type T:
  1. Get k nearest neighbors
  2. Count neighbors with same scene_type T

Cluster Purity % = correct / total × 100
Target: > 50%
```

---

## Models & Libraries

| Component | Model/Library | Version |
|-----------|---------------|---------|
| Video Embeddings | `facebook/vjepa2-vitl-fpc64-256` | 0.3B params |
| VLM Tagging | `Qwen/Qwen2.5-VL-7B-Instruct` | 7B params |
| Similarity Search | `faiss-gpu` | GPU-accelerated |
| Scene Detection | `scenedetect` | PySceneDetect |
| Visualization | `umap-learn`, `matplotlib` | CPU |
