# WalkIndia-200K

> **A Large-Scale Benchmark for Evaluating Video Foundation Models on Non-Western Urban Scenes**

---

## The Big Question

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│   "Can an AI model trained on WESTERN videos understand INDIAN streets?"        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   V-JEPA was trained on YouTube videos (mostly Western content).                │
│   We test if it can recognize that:                                             │
│                                                                                 │
│   • Two Indian market scenes are SIMILAR                                        │
│   • A market scene is DIFFERENT from a temple scene                             │
│   • Night traffic looks DIFFERENT from morning traffic                          │
│                                                                                 │
│   WITHOUT teaching it anything about India!                                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Research Novelty

| Rank | Novelty | Strength | Why Novel |
|------|---------|----------|-----------|
| 1 | **Geographic transfer evaluation** | STRONG | No one has tested if V-JEPA's "world model" transfers to Indian streets |
| 2 | **Label-free video evaluation metrics** | STRONG | Self-consistency & stability metrics are new for video (only done for images) |
| 3 | **Indian urban video dataset** | MEDIUM | New dataset contribution (~200K clips from 700 videos) |
| 4 | **Qwen3-VL pseudo-labeling pipeline** | MEDIUM | Practical contribution for structured video tagging |

### Research Gap (Validated via Web Search)

- NO evaluation of V-JEPA on Indian/non-Western street videos
- NO large-scale Indian urban walking video dataset
- NO "self-consistency + stability" metrics applied to VIDEO embeddings
- NO study on cultural/geographical transfer of video world models

### Honest Limitations

| Limitation | Mitigation |
|------------|------------|
| Qwen3-VL tags are pseudo-labels, not ground truth | Human eval on 500 clips + cross-VLM validation |
| Circular bias: Western models validating Western models | Include DINOv2/random baselines + human judgment |
| Video artifacts (blur, shake) may confound clustering | Quality filtering + stratified analysis |

---

## Pipeline Summary

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                       WalkIndia-200K Benchmark                                                            ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   YouTube    │       │ PySceneDetect│       │  ~200K clips │
│  700 videos  │ ════► │ ~ 10 second  │ ════► │   (4-5s)     │ ════╗
│  20-30 min   │       │    cuts      │       │              │     ║
└──────────────┘       └──────────────┘       └──────────────┘     ║
                                                                   ▼
╔══════════════════════════════════════════════════════════════════▼════════════════════════════════════════════════════════════════════════╗
║                                          PARALLEL PROCESSING                                                                              ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                                           ║
║  V-JEPA BRANCH:                                                                                                                           ║
║  ┌──────────────────┐       ┌──────────────────┐       ┌─────────────────────────────┐       ┌──────────────────────────────┐             ║
║  │    V-JEPA 2      │       │      FAISS       │       │         METRICS             │       │     UMAP + FiftyOne          │             ║
║  │  clip ➔ 64 frm   │ ════► │     IVF-PQ       │ ════► │  Recall@k | Self-Consist.  │ ════► │   2D/3D visualization        ═│═════╗       ║
║  │  ➔ ViT-L (frozen)│       │  GPU-accelerated │       │  Transform Stability       │       │   Interactive exploration     │     ║       ║
║  └──────────────────┘       └──────────────────┘       └─────────────────────────────┘       └──────────────────────────────┘     ║       ║
║                                                                                                                                   ║       ║
║  MULTI-VLM BRANCH:                                                                                                                ║       ║                
║  ┌───────────────────────────────────────────────┐       ┌─────────────────────────────────────┐                                  ║       ║                
║  │  • Qwen3-VL-8B    (MLVU: 75.3, Hindi OCR)    │       │         Cross-VLM Agreement          │                                  ║       ║                
║  │  • VideoLLaMA3-7B (#1 VideoMME, 128 frames)  │ ════► │  Qwen3 ∩ VideoLLaMA3 ∩ InternVL2.5   │ ═════════════════════════════════╣       ║                
║  │  • InternVL2.5-8B (different architecture)   │       │  >90% = high conf | <70% = discard   │                                  ║       ║                
║  └───────────────────────────────────────────────┘       └─────────────────────────────────────┘                                  ║       ║                
║                                                                                                                                   ║       ║              
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╩═══════╝
                                                                                                                                    ║
                                                                                                                                    ▼
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                           DELIVERABLES                                                                    ║
╠═════════════════════╦═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ 1. Benchmark        ║ Self-consistency %, Recall@k, Transform Stability %                                                                 ║
║ 2. Dataset          ║ WalkIndia-200K (clips + embeddings + multi-VLM labels)                                                              ║
║ 3. Paper Finding    ║ Does V-JEPA transfer to Indian streets? (Yes/No + evidence)                                                         ║
║ 4. Baselines        ║ Random embeddings, DINOv2, shuffled V-JEPA                                                                          ║
║ 5. Cross-VLM        ║ % clips where Qwen3, VideoLLaMA3, InternVL2.5 agree                                                                 ║
╚═════════════════════╩═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Clarification: "Label-Free" Claim

| Metric | Truly Label-Free? | Explanation |
|--------|-------------------|-------------|
| **Self-Consistency** | YES | If A's nearest neighbor is B, does B point back to A? No labels needed. |
| **Transform Stability** | YES | Same clip with different crops → similar neighbors? No labels needed. |
| **Recall@k Retrieval** | YES | Retrieval ranking computed from embeddings only. |
| **Semantic Clustering** | **NO** | Uses Qwen3-VL pseudo-labels for validation. Honest about this. |

### Addressing Circular Bias

| Concern | Mitigation |
|---------|------------|
| V-JEPA + single VLM both Western-trained | Cross-VLM agreement: Qwen3 ∩ VideoLLaMA3 ∩ InternVL2.5 |
| Single VLM bias | Use 3 different VLMs, only trust labels where all 3 agree |
| Video artifacts vs semantics | Filter by quality score, stratify analysis by blur/shake |
| Models may share training data biases | Report agreement % as confidence metric |

---

## Step 1: Data Collection

| Source | @walkinginindia YouTube |
|--------|-------------------------|
| Videos | ~700 videos × 20-30 min |
| Total  | 14,000-21,000 minutes   |
| Content| Markets, Junctions, Temples, Beach roads, Lanes |
| Tool   | `yt-dlp` / `youtube-dl` |
| Output | Raw `.mp4` files        |

---

## Step 2: Scene Detection

**Library**: `PySceneDetect` ([scenedetect.com](https://scenedetect.com))

```
30-min video ──→ Content Detection ──→ [clip1][clip2][clip3]...
                                        4-5s   4-5s   4-5s
```

| Input  | 700 videos              |
|--------|-------------------------|
| Output | ~200,000 short clips    |
| Method | Content-aware splitting |

---

## Step 3: V-JEPA 2 Embedding

**Model**: `facebook/vjepa2-vitl-fpc64-256` (0.3B params, frozen)

```
4-5s clip ──→ Sample 64 frames ──→ ViT-L Encoder ──→ Embedding Vector
                                   (NO TRAINING)     [dim: hidden_size]
```

| Property | Value |
|----------|-------|
| Frames   | 64 per clip |
| Params   | 0.3B (frozen) |
| Training | None required |

---

## Step 4: Auto-Tagging

**Model**: `Qwen3-VL-8B-Instruct`

**Why Qwen3-VL over Video-LLaVA for Indian urban scenes:**
| Feature | Qwen3-VL | Video-LLaVA |
|---------|----------|-------------|
| Text/signage reading | Yes (Hindi/English) | Limited |
| Frame sampling | Dynamic FPS (adapts to motion) | Fixed 64 frames |
| Scene layouts | Stronger | Moderate |
| JSON output | Native support | Needs prompting |

Structured tags per clip (NOT free-form captions):

```json
{
  "scene_type":      "market|junction|lane|promenade|transit|temple|highway|alley",
  "time_of_day":     "morning|afternoon|evening|night",
  "weather":         "clear|rain|fog",
  "crowd_density":   "low|med|high",
  "traffic_density": "low|med|high",
  "notable_objects": ["bus","rickshaw","bike","vendor","police","signage","animals"],
  "road_layout":     "intersection|narrow_lane|wide_road|sidewalk|median"
}
```

---

## Step 5: FAISS Indexing

**Library**: `FAISS` (Facebook AI Similarity Search)

**Why FAISS instead of naive kNN:**
| Metric | Naive kNN | FAISS (IVF-PQ) |
|--------|-----------|----------------|
| Time complexity | O(n²) | O(n log n) |
| 200K clips search | ~hours | ~seconds |
| Memory | All in RAM | Compressed (PQ) |
| GPU support | No | Yes (5-10x faster) |

```
200K embeddings ──→ FAISS Index (IVF/HNSW) ──→ Fast Approximate kNN
```

**Recommended Index:**
```python
import faiss

d = 768  # V-JEPA embedding dimension
nlist = 1000  # clusters for IVF

# IVF + PQ: fast, memory-efficient
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, 16, 8)
index.train(embeddings)
index.add(embeddings)

# Search k nearest neighbors
distances, indices = index.search(query, k=10)
```

---

## Step 6: UMAP Visualization

**Library**: `UMAP` (Uniform Manifold Approximation and Projection)

**Why UMAP:**
| Feature | UMAP | t-SNE |
|---------|------|-------|
| Speed | Fast | Slow |
| Global structure | Preserved | Lost |
| Scalability | 200K+ points | ~10K points |
| Clustering-friendly | Yes (works with HDBSCAN) | Limited |

```
768-dim embeddings ──→ UMAP ──→ 2D/3D scatter plot
```

**Use cases:**
- Paper figures showing cluster separation
- Debug embedding quality
- Validate if scene types actually cluster

```python
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
embedding_2d = reducer.fit_transform(embeddings)

# Plot with scene_type colors from Qwen3-VL tags
plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=scene_type_colors)
```

---

## Step 7: FiftyOne Exploration

**Library**: `FiftyOne` (Voxel51) - Open-source dataset curation tool

**Why FiftyOne:**
| Feature | Custom Scripts | FiftyOne |
|---------|----------------|----------|
| Interactive UI | No | Yes (web-based) |
| UMAP built-in | Manual | One-click |
| Filter by tags | Code | Visual |
| Find outliers | Hard | Easy |
| Share with team | Difficult | URL link |

```
clips + embeddings + tags ──→ FiftyOne Dataset ──→ Interactive Web UI
```

**Use cases:**
- Browse 200K clips visually
- Click on UMAP points to view clips
- Filter by scene_type, crowd_density, etc.
- Find mislabeled samples

```python
import fiftyone as fo

dataset = fo.Dataset("walkindia-200k")
for clip_path, embedding, tags in zip(clips, embeddings, all_tags):
    sample = fo.Sample(filepath=clip_path)
    sample["embedding"] = embedding.tolist()
    sample["scene_type"] = tags["scene_type"]
    sample["crowd_density"] = tags["crowd_density"]
    dataset.add_sample(sample)

# Launch interactive UI
session = fo.launch_app(dataset)
```

---

## Step 8: Evaluation - Quantitative Metrics

### 8.1 Label-Free Metrics (Core Contribution)

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Self-Consistency** | % of clips where kNN(A)=B implies kNN(B)=A | Embedding neighborhood stability |
| **Transform Stability** | IoU of kNN(crop1) vs kNN(crop2) for same clip | Robustness to view changes |
| **Recall@k** | % of same-scene clips in top-k neighbors | Retrieval quality |

### 8.2 Pseudo-Label Validation (Ablation Only)

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Cluster Purity** | % of kNN neighbors with same Qwen3-VL scene_type | Semantic coherence |
| **Cross-VLM Agreement** | % where Qwen3-VL, VideoLLaMA3, InternVL2.5 agree | Reduces single-model bias |

### 8.3 Baselines (Required for Fair Comparison)

| Baseline | Purpose |
|----------|---------|
| **Random embeddings** | Lower bound - should have ~0% consistency |
| **Shuffled V-JEPA** | Tests if temporal order matters |
| **DINOv2 (image-only)** | Tests if video understanding adds value |
| **CLIP** | Tests text-vision alignment baseline |

### 8.4 Cross-VLM Agreement (Reduces Single-Model Bias)

| VLM | Size | Why Selected | Strength |
|-----|------|--------------|----------|
| **Qwen3-VL-8B** | 8B | Best on MLVU (75.3), LVBench (56.2) | Long video, Hindi text/signage |
| **VideoLLaMA3-7B** | 7B | #1 on VideoMME (7B class), 128 frames | Motion understanding |
| **InternVL2.5-8B** | 8B | Different architecture, Chinese training | Architectural diversity |

**Agreement Metric:**
```
Cross-VLM Agreement % = |clips where Qwen3 ∩ VideoLLaMA3 ∩ InternVL agree| / total clips
```

| Agreement Level | Interpretation | Action |
|-----------------|----------------|--------|
| > 90% | High confidence | Use as ground truth |
| 70-90% | Moderate confidence | Review edge cases |
| < 70% | Low confidence | Discard or manual review |

### 8.5 Confounder Analysis

| Confounder | Mitigation |
|------------|------------|
| Motion blur | Filter clips by blur score > threshold |
| Camera shake | Filter clips by optical flow variance |
| Lighting changes | Stratify analysis: day vs night |
| Video quality | Report metrics separately for high/low quality |

### Success Criteria

> V-JEPA transfers well if: (1) Self-consistency > 70%, (2) Cross-VLM agreement > 80%, (3) Outperforms DINOv2 baseline on Indian data.

---

## Key Libraries

| Step | Library | Purpose |
|------|---------|---------|
| 2 | PySceneDetect | Split videos into clips |
| 3 | V-JEPA 2 | Frozen video embeddings |
| 4 | Qwen3-VL-8B | Structured auto-tagging |
| 5 | FAISS | Fast similarity search (GPU) |
| 6 | UMAP | Dimensionality reduction & visualization |
| 7 | FiftyOne | Interactive dataset exploration |

---

## Optional Improvements

The following tools are **not required** for the current pipeline but may be useful for future extensions.

---

### 1. SAM3 (Segment Anything Model 3)

**Purpose**: Pixel-level object segmentation masks

| Aspect | Details |
|--------|---------|
| **Pros** | Precise object boundaries, exact object counts, track objects across frames |
| **Cons** | High compute cost, slow inference, requires GPU |
| **Current Redundancy** | HIGH - Qwen3-VL already captures `notable_objects` list |
| **Future Use Case** | Object-level features for fine-grained retrieval, counting pedestrians/vehicles |

```
[OPTIONAL] clip frames ──→ SAM3 ──→ pixel masks per object
```

---

### 2. DINOv2 (Multi-Encoder Ensemble)

**Purpose**: Add image-based embeddings alongside V-JEPA video embeddings

| Aspect | Details |
|--------|---------|
| **Pros** | Strong static appearance features, well-established baseline, can ensemble with V-JEPA |
| **Cons** | 2x compute cost, requires embedding fusion strategy |
| **Current Redundancy** | HIGH - V-JEPA 2 already trained on images+videos, covers both motion & appearance |
| **Future Use Case** | Ablation study comparing V-JEPA vs DINOv2 vs ensemble on Indian data |

```
[OPTIONAL] clip ──→ DINOv2 ──→ image embedding ──┐
                └──→ V-JEPA ──→ video embedding ──┴──→ concat/fuse
```

---

### 3. TransNetV2 (Neural Scene Detection)

**Purpose**: Neural network-based scene boundary detection (replace PySceneDetect)

| Aspect | Details |
|--------|---------|
| **Pros** | Higher accuracy on hard cuts, better on gradual transitions, trained on real boundaries |
| **Cons** | Requires GPU, slower than PySceneDetect, marginal improvement |
| **Current Redundancy** | MEDIUM - PySceneDetect's `detect-adaptive` already good enough |
| **Future Use Case** | If scene splits are poor quality, switch to TransNetV2 |

```
[OPTIONAL] video ──→ TransNetV2 ──→ more accurate scene boundaries
```

---

### 4. Autodistill (Zero-Annotation Object Detection)

**Purpose**: Auto-label objects using foundation models (GroundingDINO + SAM)

| Aspect | Details |
|--------|---------|
| **Pros** | Precise bounding boxes, object counts, no manual labeling needed |
| **Cons** | Pipeline complexity, requires multiple models, slow inference |
| **Current Redundancy** | HIGH - Qwen3-VL sufficient for scene-level tagging, we don't need boxes |
| **Future Use Case** | If you need object-level ground truth for training downstream models |

```
[OPTIONAL] clip ──→ GroundingDINO ──→ bounding boxes ──→ object counts
```

---

### 5. Weak Supervision / LLM Validator

**Purpose**: Use LLM (GPT-4) to auto-correct/validate Qwen3-VL tags

| Aspect | Details |
|--------|---------|
| **Pros** | Catches tagging errors, improves ground truth quality, industry standard |
| **Cons** | API costs (GPT-4), adds latency, premature optimization |
| **Current Redundancy** | HIGH - Only needed if Qwen3-VL tags have many errors (test first) |
| **Future Use Case** | Production-grade dataset curation, if Qwen3-VL accuracy drops below 90% |

```
[OPTIONAL] clip ──→ Qwen3-VL ──→ tags ──→ GPT-4 validator ──→ cleaned tags
```

---

## Optional Summary Table

| Tool | Redundancy | Add When? |
|------|------------|-----------|
| SAM3 | HIGH | Need object-level features |
| DINOv2 | HIGH | Ablation study / ensemble experiments |
| TransNetV2 | MEDIUM | Scene splits are poor quality |
| Autodistill | HIGH | Need bounding box annotations |
| Weak Supervision | HIGH | Qwen3-VL accuracy < 90% |
