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
│   • Chaotic mixed traffic is DIFFERENT from orderly motorized traffic           │
│   • Shared pedestrian-vehicle space is DIFFERENT from separated sidewalks       │
│                                                                                 │
│   WITHOUT teaching it anything about India!  (Ch 9)                             │
│   Then: Can we TEACH it Indian patterns?     (Ch 10 + Ch 11)                   │
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
| 4 | **VLM bake-off pipeline** | MEDIUM | 3-VLM comparison → consensus-based winner selection |
| 5 | **Domain adaptation via surgery fine-tuning** | STRONG | Factor-decomposed self-supervised adaptation (layout/agent/interaction) |

### Research Gap (Validated via Web Search)

- NO evaluation of V-JEPA on Indian/non-Western street videos
- NO large-scale Indian urban walking video dataset
- NO "self-consistency + stability" metrics applied to VIDEO embeddings
- NO study on cultural/geographical transfer of video world models
- NO factor-decomposed self-supervised domain adaptation for video encoders

### Honest Limitations

| Limitation | Mitigation |
|------------|------------|
| VLM tags are pseudo-labels, not ground truth | VLM bake-off (3-way consensus on 2.5K clips) + per-field confidence + confidence sweep |
| Circular bias: Western models validating Western models | Include DINOv2/random baselines; primary metrics are label-free (Cycle@K, Overlap@K) |
| Video artifacts (blur, shake) may confound clustering | Quality filtering + stratified analysis |

---

## POC-First Strategy

**Run the entire 4-chapter pipeline on a 10K video-level uniform subset before scaling to 115K.**

| Aspect | Detail |
|--------|--------|
| **Subset** | 10,000 clips from 115K (video-level uniform, seed=42) |
| **Tool** | `m00c_sample_subset.py` → `data/subset_10k.json` |
| **Flag** | All scripts accept `--subset data/subset_10k.json` |
| **Output** | `outputs_poc/` (separate from `outputs/`) |
| **Scale** | After POC validates → drop `--subset`, same scripts run on 115K |

### POC Timeline (Sequential, on RTX PRO 6000 96GB)

| Week | Chapter | GPU Hours | Deliverable | Status |
|:----:|---------|:---------:|-------------|--------|
| 1 | Ch 8+9 (data + eval + baselines) | ~15h | metrics_frozen.json + 15 plots + baselines | **90% DONE** |
| 2 | Ch 10 (continual pretraining) | ~20h | metrics_adapted.json (frozen vs adapted) | NEXT |
| 3-4 | Ch 11 (surgery fine-tuning) | ~54h | metrics_surgical.json (frozen vs adapted vs surgical) | FUTURE |

---

## Full Pipeline (Ch 8 → 9 → 10 → 11)

```mermaid
flowchart TB
    subgraph DATA ["Ch 8 · Data Pipeline (Mac CPU)"]
        direction LR
        A["YouTube<br>700 vids"] --> B["PySceneDetect<br>4-10s cuts"] --> C["~115K clips<br>121 GB"] --> D["WebDataset<br>116 TAR · HF"]
    end

    SUB(["m00c · 10K uniform · seed=42"])

    subgraph VLM ["Ch 8 · VLM Bake-off (2.5K)"]
        direction LR
        I1["Qwen3-VL-8B"] --> S["m04b · 5-criterion<br>consensus"]
        I2["VideoLLaMA3-7B"] --> S
        I3["LLaVA-NeXT-1.5-8B"] --> S
        S --> W["Winner → tags.json<br>16 fields × denseworld"]
    end

    subgraph CH9 ["Ch 9 · Evaluate Frozen V-JEPA"]
        direction TB
        E["V-JEPA 2 ViT-G<br>frozen · 1408-dim"] --> F["FAISS kNN<br>k=6 · Hard/Easy"]
        BL["m05b · 4 Baselines<br>Random · DINOv2<br>Shuffled · CLIP"] --> F
        F --> G["9 Metrics<br>+ true Overlap@K<br>+ quality strat.<br>+ 16-field slices"]
        G --> H["UMAP + plots<br>× 5 encoders"]
    end

    subgraph CH10 ["Ch 10 · Continual Pretraining"]
        direction TB
        J1["Student encoder<br>init = frozen"] --> J2["JEPA loss<br>mask 20%"] --> J3["Teacher EMA<br>τ = 0.999"] --> J4["V-JEPA<br>(adapted)"]
    end

    subgraph CH11 ["Ch 11 · Surgery Fine-Tuning"]
        direction TB
        K1["SAM3 masks<br>→ tracklets"] --> K2["3 Factor Datasets<br>D_L · D_A · D_I"] --> K3["3-Stage Prefix<br>Unfreezing"] --> K4["V-JEPA<br>(surgical)"]
    end

    subgraph FINAL ["PAPER · 3-WAY COMPARISON"]
        direction TB
        L1["Re-run m05→m08<br>all 3 encoders"] --> L2["frozen vs adapted<br>vs surgical<br>× 15 taxonomy keys"]
    end

    DATA --> SUB
    SUB --> VLM
    SUB --> CH9
    W -->|"tags for eval"| CH9
    W -->|"stratified<br>batch"| CH10
    W -->|"stratified<br>batch"| CH11
    CH9 -->|"frozen<br>baseline"| FINAL
    CH10 --> FINAL
    CH11 --> FINAL

    style A fill:#1e88e5,color:#fff,font-weight:bold,font-size:28px
    style B fill:#8e24aa,color:#fff,font-weight:bold,font-size:28px
    style C fill:#00897b,color:#fff,font-weight:bold,font-size:28px
    style D fill:#5e35b1,color:#fff,font-weight:bold,font-size:28px
    style SUB fill:#fdd835,color:#000,font-weight:bold,font-size:28px
    style E fill:#43a047,color:#fff,font-weight:bold,font-size:28px
    style F fill:#e53935,color:#fff,font-weight:bold,font-size:28px
    style G fill:#d81b60,color:#fff,font-weight:bold,font-size:28px
    style H fill:#795548,color:#fff,font-weight:bold,font-size:28px
    style BL fill:#546e7a,color:#fff,font-weight:bold,font-size:28px
    style I1 fill:#00acc1,color:#fff,font-weight:bold,font-size:28px
    style I2 fill:#00838f,color:#fff,font-weight:bold,font-size:28px
    style I3 fill:#006064,color:#fff,font-weight:bold,font-size:28px
    style S fill:#ff6f00,color:#fff,font-weight:bold,font-size:28px
    style W fill:#00acc1,color:#fff,font-weight:bold,font-size:28px
    style J1 fill:#7b1fa2,color:#fff,font-weight:bold,font-size:28px
    style J2 fill:#7b1fa2,color:#fff,font-weight:bold,font-size:28px
    style J3 fill:#9c27b0,color:#fff,font-weight:bold,font-size:28px
    style J4 fill:#4a148c,color:#fff,font-weight:bold,font-size:28px
    style K1 fill:#1565c0,color:#fff,font-weight:bold,font-size:28px
    style K2 fill:#1565c0,color:#fff,font-weight:bold,font-size:28px
    style K3 fill:#1565c0,color:#fff,font-weight:bold,font-size:28px
    style K4 fill:#0d47a1,color:#fff,font-weight:bold,font-size:28px
    style L1 fill:#bf360c,color:#fff,font-weight:bold,font-size:28px
    style L2 fill:#b71c1c,color:#fff,font-weight:bold,font-size:28px
```

---

## Taxonomy: v1 → v2 (Denseworld)

Tags serve **two purposes only**: (1) stratified batching for Ch10/Ch11, (2) slice-wise evaluation. They are NEVER training labels.

| Field | v1 (Western) | v2 (Denseworld — Indian) | Change |
|-------|-------------|--------------------------|--------|
| `scene_type` | market, junction, residential_lane, promenade, transit, temple_tourist, highway, **alley**(n=14), commercial, **construction**(n=53) | market, **bazaar**, junction, residential_lane, promenade, transit, temple_tourist, highway, commercial, **ghat**, **flyover_underpass** | Removed dead categories, added Indian scenes |
| `time_of_day` | morning, afternoon, evening, night | **day, night** | Collapsed — pollution haze makes day subdivisions indistinguishable |
| `traffic_mix` | *(missing)* | **motorized_only, mixed_motorized, mixed_all, pedestrian_dominant** | **NEW** — THE Indian differentiator |
| `ped_vehicle_separation` | *(missing)* | **separated, partial, shared_space** | **NEW** — Western vs Indian infrastructure gap |
| `road_encroachment` | *(missing)* | **clear, partial, heavy** | **NEW** — informal road use |
| `road_layout` | 5 values | + **speed_breaker, open_drain** | Indian infrastructure |
| `notable_objects` | 11 Western-generic | 14 Indian-specific: + **cycle_rickshaw, handcart, sacred_cow, stray_dog, overhead_wires, religious_shrine** − police, construction_barrier, animals | India-specific objects |
| `video_quality` | *(missing)* | **clean, blur, shake** | **NEW** — quality stratification for confounder analysis |
| weather, crowd_density, traffic_density, road_surface, infrastructure_quality, vegetation, lighting | *(unchanged)* | *(unchanged)* | Universal fields |

**Total: 11 → 16 fields** (13 single + 2 multi + 1 changelog). File: `src/utils/tag_taxonomy_denseworld.json`

---

## Ch 9: Evaluate Frozen V-JEPA (DONE — 90%)

### Code built vs Proposal (FactorJEPA Ch 9)

| # | Proposal Step | Status | Module → Evidence |
|:---:|:---|:---:|:---|
| | **9.1 Step-by-step evaluation protocol** | | |
| 1 | Clip bank + leakage prevention | ✅ BUILT | m02 (4-10s scene-aware cuts) + m06 (±30s exclusion mask, video_id grouping) |
| 2 | Embedding extraction (frozen) | ✅ BUILT | m05 (V-JEPA 2 ViT-G, 1408-dim, mean-pool, near-dedup) |
| 3 | Build kNN index | ✅ BUILT | m06 (FAISS-GPU, k=6, Easy + Hard modes) |
| 4 | Evaluation subsets via tags | ✅ BUILT | m04 (dynamic prompt from taxonomy, 16 fields) + m06 (confidence sweep 7 thresholds) |
| | **9.2 Overall (label-free) evaluation** | | |
| 5 | Qualitative kNN grids | ✅ BUILT | m08 create_knn_grid() (query + k neighbors, green/red borders) |
| 6 | Cycle consistency | ✅ BUILT | m06 compute_cycle_at_k() Easy + Hard |
| 7 | Overlap@K (augmentation) | ⚠️ APPROX | m06 compute_overlap_at_k() = dim-split approximation, NOT true multi-crop |
| 8 | Clustering diagnostics | ✅ BUILT | m06 compute_silhouette() per 13 single-val keys (tags as labels, no k-means) |
| | **9.3 Class-wise evaluation using weak tags** | | |
| 9 | Prec@K per class | ✅ BUILT | m06 compute_prec_at_k() + compute_per_scene_purity() per-value |
| 10 | mAP@K / nDCG@K | ✅ BUILT | m06 compute_map_at_k() + compute_ndcg_at_k() (graded multi-field) |
| 11 | Multi-attribute slices | ✅ BUILT | m06 compute_multi_attribute_slices() (8 slice fields, all taxonomy keys) |
| 12 | Confusion analysis | ✅ BUILT | m08 create_confusion_matrix() + 3x3 grid per taxonomy key |
| | **9.4 Reporting: overall vs class-wise** | | |
| 13 | Macro/micro aggregation | ✅ BUILT | m06 compute_macro_micro_avg() (macro + count-weighted micro) |
| 14 | Confidence sweep + Hard/Easy | ✅ BUILT | m06 compute_confidence_sweep() (7 thresholds) + Hard mode throughout |
| 15 | Student-friendly protocol | — OPTIONAL | (not needed for paper) |
| | **Critical for paper (not in original 15 steps)** | | |
| 16 | Baseline — Random embeddings | ❌ NOT BUILT | Lower bound: random 1408-dim vectors → same m06 metrics pipeline |
| 17 | Baseline — DINOv2 (image) | ❌ NOT BUILT | Image-only encoder on middle frame → exposes video vs image gap |
| 18 | Baseline — Shuffled V-JEPA | ❌ NOT BUILT | Frame-shuffled clips → re-embed → proves temporal order matters |
| 19 | Baseline — CLIP (text-vision) | ❌ NOT BUILT | Text-aligned encoder → tests if semantic alignment helps retrieval |
| 20 | True Overlap@K (multi-crop) | ❌ NOT BUILT | Requires video augmentation pipeline (crop/resize → re-embed → compare kNN) |
| 21 | VLM re-tag with denseworld | ❌ NOT RUN | Code built (m04), but POC used old 9-key taxonomy; denseworld adds 4 fields |
| 22 | UMAP visualization | ✅ BUILT | m07 GPU cuML UMAP (1408→2D) + m08 scatter plots per key; umap_2d.npy exists |

### GAP: Baselines (CRITICAL — must do before paper)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Ch 9 BASELINE COMPARISON (MISSING)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Without baselines, "Prec@K = 18.73%" is meaningless.                      │
│  Is that good? Compared to WHAT?                                           │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   Random     │  │  DINOv2     │  │  Shuffled   │  │    CLIP     │       │
│  │  embeddings  │  │ (image-only)│  │   V-JEPA    │  │(text-vision)│       │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤       │
│  │ 1408-dim    │  │ ViT-L/14    │  │ shuffle frms│  │ ViT-L/14    │       │
│  │ random vecs │  │ frozen      │  │ re-embed    │  │ frozen      │       │
│  │ L2-normed   │  │ 1 frame/clip│  │ same V-JEPA │  │ 1 frame/clip│       │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤       │
│  │ LOWER BOUND │  │ IMAGE vs    │  │ TEMPORAL    │  │ TEXT-VISION │       │
│  │ ~10% Prec@K │  │ VIDEO test  │  │ ORDER test  │  │ ALIGNMENT   │       │
│  │ (1/10 types)│  │             │  │             │  │ test        │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                    │                                        │
│                                    ▼                                        │
│                    ALL re-run through SAME m06/m07/m08                      │
│                    SAME 5,105 clips, SAME k=6, SAME tags                   │
│                    → side-by-side comparison table in paper                 │
│                                                                             │
│  PRIORITY:                                                                  │
│  ■ Random baseline    — ~30 min (generate random vectors, run m06)         │
│  ■ DINOv2 baseline    — ~2-3h GPU (embed 5K clips, run m06)               │
│  ■ Shuffled V-JEPA    — ~2h GPU (temporal order test)                      │
│  ■ CLIP baseline      — ~2h GPU (text-vision alignment)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Ch 9 Key Findings (from report.md)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Ch 9 FINDINGS SUMMARY                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. V-JEPA organizes by ILLUMINATION, not SCENE SEMANTICS                  │
│     lighting mAP@K = 0.66  >>>  scene_type mAP@K = 0.11  (6x gap)         │
│     Only lighting has positive silhouette (+0.0007)                         │
│                                                                             │
│  2. High nDCG (0.90) + Low Prec@K (18.7%) = CONTEXTUAL retrieval          │
│     Neighbors share lighting + weather + crowd but DIFFER in scene type    │
│     "Similar vibes, different places"                                       │
│                                                                             │
│  3. Neighborhoods are STABLE: Cycle@K ~79% (77-81% band)                   │
│     Uniform across ALL scene types — model quality is consistent            │
│                                                                             │
│  4. Easy/Hard gap < 0.6pp — data pipeline prevents temporal leakage        │
│                                                                             │
│  5. VLM confidence UNCALIBRATED: 99.84% of clips ≥ 0.9 confidence         │
│     Confidence sweep is flat — threshold filtering uninformative            │
│                                                                             │
│  IMPLICATION FOR Ch 10-11:                                                  │
│  V-JEPA needs domain adaptation to understand Indian SCENE SEMANTICS.       │
│  It already captures lighting/weather — adaptation should target            │
│  traffic_mix, pedestrian_vehicle_separation, scene_type (denseworld keys)  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Ch 10: Continual Self-Supervised Pretraining

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     Ch 10: CONTINUAL PRETRAINING ON INDIAN CLIPS                        │
│                     Same JEPA loss, no labels, just Indian data                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  INPUTS FROM Ch 9:                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                                  │
│  │ tags.json    │  │ embeddings   │  │ metrics      │                                  │
│  │ (denseworld  │  │ .npy         │  │ _frozen.json │                                  │
│  │  15 fields)  │  │ (frozen      │  │ (baseline to │                                  │
│  │              │  │  encoder)    │  │  beat)       │                                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                                  │
│         │                 │                 │                                            │
│    stratified         validation        compare                                         │
│    batching           retrieval         before/after                                     │
│         │                 │                 │                                            │
│  ═══════╪═════════════════╪═════════════════╪══════════════════════════════════           │
│         ▼                 ▼                 ▼                                            │
│                                                                                         │
│  ┌─ ONE TRAINING STEP ─────────────────────────────────────────────────────────┐        │
│  │                                                                              │        │
│  │  1. SAMPLE batch (uniform by video_id, stratified by denseworld tags)       │        │
│  │     ┌───────────────────────────────────────────────────────────────┐        │        │
│  │     │ Ensure mix of: day/night, traffic_mix values,                │        │        │
│  │     │ pedestrian_vehicle_separation, scene_type diversity          │        │        │
│  │     │ Tags used for SAMPLING ONLY — never as supervised targets    │        │        │
│  │     └───────────────────────────────────────────────────────────────┘        │        │
│  │                                                                              │        │
│  │  2. DECODE clip → T frames (16 or 32) → 224×224                            │        │
│  │     Apply video-consistent augmentations (one crop for ALL frames)          │        │
│  │                                                                              │        │
│  │  3. MASK 20% of spatiotemporal patches (2-6 rectangular blocks)             │        │
│  │     ┌──┬──┬──┬──┬──┬──┬──┬──┐                                              │        │
│  │     │  │  │██│██│  │  │  │  │  ██ = hidden (target)                         │        │
│  │     ├──┼──┼──┼──┼──┼──┼──┼──┤  □  = visible (context)                      │        │
│  │     │  │  │██│██│  │  │  │  │                                               │        │
│  │     ├──┼──┼──┼──┼──┼──┼──┼──┤                                              │        │
│  │     │  │  │  │  │  │  │  │  │                                               │        │
│  │     └──┴──┴──┴──┴──┴──┴──┴──┘                                              │        │
│  │                                                                              │        │
│  │  4. FORWARD PASS                                                            │        │
│  │     ┌─────────────────┐          ┌─────────────────┐                        │        │
│  │     │  STUDENT f_θ    │          │  TEACHER f_θ̄   │                        │        │
│  │     │  (trainable)    │          │  (EMA copy)     │                        │        │
│  │     │  sees: visible  │          │  sees: masked   │                        │        │
│  │     │  patches only   │          │  patches only   │                        │        │
│  │     └────────┬────────┘          └────────┬────────┘                        │        │
│  │              │                            │ (no gradients)                   │        │
│  │              ▼                            ▼                                  │        │
│  │     ┌─────────────────┐          ┌─────────────────┐                        │        │
│  │     │  PREDICTOR g_φ  │          │  Teacher targets │                        │        │
│  │     │  (trainable)    │─── MSE ──│  T = sg(f_θ̄)    │                        │        │
│  │     │  predicts T̂     │   loss   │  (stop gradient) │                        │        │
│  │     └─────────────────┘          └─────────────────┘                        │        │
│  │                                                                              │        │
│  │  5. UPDATE                                                                  │        │
│  │     θ ← θ − lr·∇L_JEPA(θ,φ)           (student + predictor by gradient)    │        │
│  │     θ̄ ← 0.999·θ̄ + 0.001·θ             (teacher by EMA, no gradient)       │        │
│  │     Optional: + λ·‖θ − θ₀‖²            (drift stabilizer)                  │        │
│  │                                                                              │        │
│  └──────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                         │
│  CHECKPOINT SELECTION (every 2K-5K steps):                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐                │
│  │  1. Extract embeddings on validation subset (held-out video_ids)    │                │
│  │  2. Build FAISS index → compute Cycle@K (Hard mode)                │                │
│  │  3. Pick checkpoint with best Cycle@K                              │                │
│  │     (primary: label-free, no tag dependency)                        │                │
│  │  4. Also log: Prec@K per denseworld key (diagnostic, not selection) │                │
│  └──────────────────────────────────────────────────────────────────────┘                │
│                                                                                         │
│  OUTPUT: V-JEPA (adapted) = student encoder f_θ at best checkpoint                      │
│                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  HYPERPARAMETERS (runnable defaults)                                                    │
│  Clip: 10s, T=16 frames, 224px │ Mask: 20%, 2-6 blocks │ EMA τ: 0.996→0.999 warmup    │
│  Optimizer: AdamW │ LR: small (backbone) + larger (predictor) │ Grad clip: 1.0          │
│  Drift: λ tuned in ablation │ Checkpoint: every 2K steps │ Mixed precision (bf16)       │
│  Est GPU: ~20h on RTX PRO 6000 (96GB) for 10K clips                                    │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Ch 10 Evaluation (re-run SAME pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Ch 10 EVALUATION FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  V-JEPA (adapted)                                                           │
│  student checkpoint                                                         │
│        │                                                                    │
│        ▼                                                                    │
│  m05_vjepa_embed.py ──→ embeddings_adapted.npy (re-embed ALL 5K clips)     │
│        │                                                                    │
│        ▼                                                                    │
│  m06_faiss_metrics.py ──→ metrics_adapted.json (SAME 9 metrics)            │
│        │                     + per-key breakdown on 15 denseworld fields    │
│        ▼                                                                    │
│  m07_umap.py ──→ umap_2d_adapted.npy                                      │
│        │                                                                    │
│        ▼                                                                    │
│  m08_plot.py ──→ SIDE-BY-SIDE plots: frozen vs adapted                     │
│                                                                             │
│  KEY COMPARISON TABLE (the paper result):                                   │
│  ┌──────────────────────┬──────────────┬──────────────┬─────────┐          │
│  │ Metric               │ Frozen (Ch9) │ Adapted(Ch10)│ Delta   │          │
│  ├──────────────────────┼──────────────┼──────────────┼─────────┤          │
│  │ scene_type mAP@K     │ 0.11         │ ???          │ +???    │          │
│  │ traffic_mix mAP@K    │ (new field)  │ ???          │ (new)   │          │
│  │ ped_veh_sep mAP@K    │ (new field)  │ ???          │ (new)   │          │
│  │ lighting mAP@K       │ 0.66         │ ???          │ ±???    │          │
│  │ Cycle@K              │ 78.96%       │ ???          │ ±???    │          │
│  │ nDCG@K               │ 0.90         │ ???          │ ±???    │          │
│  └──────────────────────┴──────────────┴──────────────┴─────────┘          │
│                                                                             │
│  EXPECTED OUTCOMES:                                                         │
│  • scene_type mAP improves modestly (0.11 → 0.15-0.20)                    │
│  • traffic_mix/ped_veh_sep show Indian-specific learning                    │
│  • lighting mAP stays stable (already good at 0.66)                        │
│  • IF no improvement → motivates Ch 11 (surgery needed, not just data)     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Code built vs Proposal (FactorJEPA Ch 10)

| # | Proposal Step | Status | Module → Evidence |
|:---:|:---|:---:|:---|
| | **10.1 Training data and sampling** | | |
| 1 | Train/val split by video_id (avoid leakage) | ❌ NOT BUILT | m09 — need video_id-level split (no clip from same video in both sets) |
| 2 | Decode normalization (fixed FPS, T frames, 224px) | ❌ NOT BUILT | m09 — consistent T={16,32} frames, fixed spatial resize+crop pipeline |
| 3 | Stratified sampling (uniform by video_id + tag mix) | ❌ NOT BUILT | m09 — denseworld tags for batch balancing (day/night, traffic_mix, scene_type) |
| | **10.2 Model components** | | |
| 4 | Student encoder (init from frozen V-JEPA, trainable) | ❌ NOT BUILT | m09 — load facebook/vjepa2-vitg-fpc64-384, set requires_grad=True |
| 5 | Teacher encoder (EMA copy, non-trainable) | ❌ NOT BUILT | m09 — deepcopy of student, no gradients, updated only by EMA |
| 6 | Predictor network (student→teacher space, trainable) | ❌ NOT BUILT | m09 — small network g_phi, trained jointly with student |
| | **10.3 Continual JEPA objective** | | |
| 7 | Two views per clip (context + target views) | ❌ NOT BUILT | m09 — video-consistent augments (one crop for ALL frames in clip) |
| 8 | Spatiotemporal masking (15-30%, 2-6 block sampling) | ❌ NOT BUILT | m09 — sample M_t (target tokens), M_c = complement (context tokens) |
| 9 | Latent regression loss (MSE, stop-gradient on T) | ❌ NOT BUILT | m09 — L_JEPA = E[norm(T_hat - sg(T))^2] over masked tokens + minibatch |
| 10 | Teacher EMA update (tau warmup 0.996 → 0.999) | ❌ NOT BUILT | m09 — theta_bar = tau * theta_bar + (1-tau) * theta after each step |
| | **10.4 Optimization** | | |
| 11 | AdamW + LR schedule (small backbone, larger predictor) | ❌ NOT BUILT | m09 — warmup + grad clip 1.0, mixed precision bf16 |
| 12 | Conservative drift control (L2 anchor to theta_0) | ❌ NOT BUILT | m09 — optional R_stab = lambda * norm(theta - theta_0)^2, lambda tuned |
| | **10.5 Training loop** | | |
| 13 | Full training step (sample→decode→augment→mask→fwd→loss→update→EMA) | ❌ NOT BUILT | m09 — complete loop, uniform video_id sampling |
| 14 | Checkpointing (student + teacher, every 2K-5K steps) | ❌ NOT BUILT | m09 — save both weights, student = official checkpoint |
| | **10.6 Validation and model selection** | | |
| 15 | Fast validation subset (held-out video_ids, 5-10K) | ❌ NOT BUILT | m09 — fixed val set, cheap retrieval metrics per checkpoint |
| 16 | Checkpoint selection (best Cycle@K hard mode) | ❌ NOT BUILT | m09 — primary: label-free Cycle@K, diagnostic: per-key Prec@K |
| | **10.7 Reporting and ablations** | | |
| 17 | Ablations (steps, aug strength, EMA tau, stabilizer lambda) | ❌ NOT BUILT | m09 — sweep 4 hyperparams, report overall + slice metrics |
| 18 | Evaluation (re-run m05→m08, frozen vs adapted table) | ❌ NOT BUILT | m05+m06+m07+m08 — re-embed with adapted encoder, side-by-side comparison |

---

## Ch 11: Surgery Fine-Tuning

### Factor Dataset Creation (SAM → Tracklets → 3 Datasets)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     Ch 11: FACTOR DATASET CREATION                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  RAW CLIP (10s, Delhi market):                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐        │
│  │  🏪🏪  🛺 🐄  👤👤👤  🏪🏪  🛺  👤  🏪  overhead wires ~~~~            │        │
│  │  shops  auto cow  people  shops auto person  road surface ════             │        │
│  └─────────────────────────────────────────────────────────────────────────────┘        │
│                    │                                                                    │
│                    ▼                                                                    │
│  STEP 1: SAM3 (every frame) → instance masks {m_t,k} with confidence                   │
│  STEP 2: Track across frames → greedy IoU matching (δ_iou=0.3, gap=1 frame)            │
│  STEP 3: Classify tracklets → motion score (centroid displacement)                      │
│           moving (≥4 frames above threshold) = AGENT                                    │
│           static = LAYOUT / BACKGROUND                                                  │
│                    │                                                                    │
│          ┌─────────┼────────────────────┐                                               │
│          ▼         ▼                    ▼                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────────────┐                 │
│  │ D_L: LAYOUT  │ │ D_A: AGENT  │ │ D_I: INTERACTION                 │                 │
│  │              │ │              │ │                                  │                 │
│  │ Suppress all │ │ Suppress    │ │ Mine pairs of agent tracklets    │                 │
│  │ agents       │ │ background  │ │ that are:                        │                 │
│  │ (blur/       │ │ (zeros/     │ │  • close (d < 0.2 × frame_w)    │                 │
│  │  inpaint)    │ │  matte)     │ │  • persistent (≥4 frames)       │                 │
│  │              │ │              │ │  • with motion cue:             │                 │
│  │ Keeps:       │ │ Keeps:      │ │    approach/retreat/cross/follow │                 │
│  │ roads,       │ │ autos,      │ │                                  │                 │
│  │ buildings,   │ │ cows,       │ │ Extract spatiotemporal tube      │                 │
│  │ wires,       │ │ rickshaws,  │ │ (bounding box + margin around    │                 │
│  │ drains,      │ │ people,     │ │  both agents across event)       │                 │
│  │ speed bumps  │ │ dogs,       │ │                                  │                 │
│  │              │ │ handcarts   │ │ Anti-shortcut perturbations:     │                 │
│  │              │ │              │ │  • tube jitter (±5-15%)         │                 │
│  │              │ │              │ │  • margin randomization          │                 │
│  │              │ │              │ │  • raw vs masked mixing (50/50) │                 │
│  │              │ │              │ │  • mask noise (dilation/erosion) │                 │
│  └──────────────┘ └──────────────┘ └──────────────────────────────────┘                 │
│                                                                                         │
│  EVAL MAPPING (denseworld taxonomy → factor):                                           │
│  ┌──────────────────────────────────────────────────────────────┐                       │
│  │ Layout (D_L):  road_layout, road_surface, infrastructure_   │                       │
│  │                quality, road_encroachment                    │                       │
│  │ Agent (D_A):   notable_objects, traffic_mix,                │                       │
│  │                pedestrian_vehicle_separation, crowd_density  │                       │
│  │ Interaction:   mined from SAM (not VLM tags)                │                       │
│  └──────────────────────────────────────────────────────────────┘                       │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3-Stage Progressive Prefix Unfreezing

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     Ch 11: 3-STAGE SURGERY SCHEDULE                                     │
│                     Same JEPA loss throughout — only input + trainable depth change      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  V-JEPA encoder: L transformer layers (e.g. L=40)                                       │
│                                                                                         │
│  STAGE 1: LAYOUT (learn Indian road geometry)                                           │
│  ┌─────────────────────────────────────────────────────────────┐                        │
│  │  Layers 0-10   ████████████ TRAINABLE                      │                        │
│  │  Layers 11-39  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░ FROZEN        │                        │
│  │                                                             │                        │
│  │  Input: 100% layout-only clips (D_L)                       │                        │
│  │  Learns: narrow lanes, open drains, speed breakers,         │                        │
│  │          overhead wires, road widths — Indian road features  │                        │
│  │  Duration: ~5K steps + short warmup                         │                        │
│  └─────────────────────────────────────────────────────────────┘                        │
│                              │                                                          │
│                              ▼                                                          │
│  STAGE 2: AGENTS (learn Indian vehicles/people/animals)                                 │
│  ┌─────────────────────────────────────────────────────────────┐                        │
│  │  Layers 0-20   ████████████████████████ TRAINABLE          │                        │
│  │  Layers 21-39  ░░░░░░░░░░░░░░░░░░ FROZEN                  │                        │
│  │                                                             │                        │
│  │  Input: 90% agent-only (D_A) + 10% layout replay (D_L)    │                        │
│  │  Learns: auto-rickshaws, cycle-rickshaws, sacred cows,      │                        │
│  │          handcarts, stray dogs — Indian agent vocabulary     │                        │
│  │  Duration: ~5K steps + short warmup for newly-unfrozen      │                        │
│  └─────────────────────────────────────────────────────────────┘                        │
│                              │                                                          │
│                              ▼                                                          │
│  STAGE 3: INTERACTIONS (learn agent-agent relationships)                                │
│  ┌─────────────────────────────────────────────────────────────┐                        │
│  │  Layers 0-30   ████████████████████████████████ TRAINABLE  │                        │
│  │  Layers 31-39  ░░░░░░░░ FROZEN                             │                        │
│  │                                                             │                        │
│  │  Input: 85% interaction (D_I) + 10% agent + 5% layout     │                        │
│  │  Learns: auto dodging cow, pedestrian crossing through      │                        │
│  │          mixed traffic, rickshaw following pedestrian        │                        │
│  │  Duration: ~5K steps + short warmup for newly-unfrozen      │                        │
│  └─────────────────────────────────────────────────────────────┘                        │
│                              │                                                          │
│                              ▼                                                          │
│  OUTPUT: V-JEPA (surgical) = student encoder at best checkpoint                         │
│                                                                                         │
│  WHY PROGRESSIVE (not all-at-once):                                                     │
│  • Shallow layers learn low-level Indian textures FIRST (roads, surfaces)               │
│  • Mid layers learn mid-level Indian objects NEXT (agents, vehicles)                    │
│  • Deep layers learn high-level Indian relationships LAST (interactions)                │
│  • Replay mixing prevents catastrophic forgetting of earlier stages                     │
│  • Frozen output layers preserve compatibility with downstream tasks                    │
│                                                                                         │
│  SANITY CHECK: Run evaluation on RAW (unpatched) clips.                                 │
│  If gains only on patched clips → model learned artifacts, not Indian patterns. FAIL.   │
│                                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  Est GPU: ~54h on RTX PRO 6000 (96GB) for 10K clips                                    │
│  SAM3 masks: can run in parallel with Ch10 training (~10h GPU)                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Code built vs Proposal (FactorJEPA Ch 11)

| # | Proposal Step | Status | Module → Evidence |
|:---:|:---|:---:|:---|
| | **11.1 Factor datasets from SAM segmentation** | | |
| 1 | SAM3 instance segmentation (every frame → masks) | ❌ NOT BUILT | m10 — run SAM3 per frame, store as RLE/PNG, with confidence scores |
| 2 | Greedy IoU tracklet matching (delta_iou=0.3, gap=1) | ❌ NOT BUILT | m10 — associate masks across frames, max IoU matching, short gap tolerance |
| 3 | Agent vs layout classification (motion filter) | ❌ NOT BUILT | m10 — centroid displacement per tracklet, agent if motion > thresh for >=4 frames |
| 4 | Per-frame mask generation (A_t, B_t) | ❌ NOT BUILT | m10 — A_t = union of agent masks, B_t = complement, optional dilation for thin structures |
| | **11.1 Derived datasets** | | |
| 5 | D_L: layout-only (suppress agents via blur/inpaint) | ❌ NOT BUILT | m11 — preserve B_t (roads, buildings, wires), suppress A_t pixels |
| 6 | D_A: agent-only (suppress background via zeros/matte) | ❌ NOT BUILT | m11 — preserve A_t (vehicles, people, animals), suppress B_t pixels |
| | **11.2 Mining interaction events for D_I** | | |
| 7 | Candidate pairs (overlapping agent tracklets >=4 frms) | ❌ NOT BUILT | m11 — enumerate (tau_a, tau_b) pairs with temporal co-occurrence >= r frames |
| 8 | Distance + persistence filter (d < d_max, >=4 consec) | ❌ NOT BUILT | m11 — centroid distance < 0.15-0.25 x frame_w for >=r consecutive frames |
| 9 | Relative motion cue (approach/retreat/cross/follow) | ❌ NOT BUILT | m11 — direction vectors, approach=decreasing d, crossing >45 deg, following=similar velocity |
| 10 | Interaction tube extraction (bbox + 10-20% margin) | ❌ NOT BUILT | m11 — per-frame box enclosing both agents, expand margin, crop spatiotemporal tube |
| 11 | D_I: interaction dataset (raw vs masked rendering) | ❌ NOT BUILT | m11 — raw tube crop + soft-matte masked crop, mix 50/50 |
| | **11.3 Selective factor patching** | | |
| 12 | Anti-shortcut perturbations (6 types) | ❌ NOT BUILT | m11 — tube jitter +-5-15%, margin rand, raw/masked mixing, boundary blend, mask noise, artifact realism |
| | **11.4 Training objective (same JEPA loss)** | | |
| 13 | JEPA loss on patched clips (MSE, stop-grad, EMA) | ❌ NOT BUILT | m12 — identical loss to Ch10, only input distribution changes (patched clips) |
| | **11.5 Progressive prefix unfreezing** | | |
| 14 | Prefix boundary implementation (freeze layers > n_s) | ❌ NOT BUILT | m12 — requires_grad=False, exclude from optimizer param groups, no state update |
| 15 | Stage 1 — Layout (n1 ~ 0.25L, p(L)=1.0) | ❌ NOT BUILT | m12 — shallow layers trainable, 100% D_L input, ~5K steps + warmup |
| 16 | Stage 2 — Agent (n2 ~ 0.50L, p(A)=0.9, p(L)=0.1) | ❌ NOT BUILT | m12 — mid layers unfrozen, 90% D_A + 10% D_L replay, ~5K steps + warmup |
| 17 | Stage 3 — Interaction (n3 ~ 0.75L, p(I)=0.85, p(A)=0.10, p(L)=0.05) | ❌ NOT BUILT | m12 — deep layers unfrozen, 85% D_I + replay mix, ~5K steps + warmup |
| 18 | Layer-wise LR decay (smaller LR early, larger at boundary) | ❌ NOT BUILT | m12 — within unfrozen prefix, reduces risk of destroying low-level filters |
| | **11.6 Stage-wise training loop** | | |
| 19 | Per-stage init (increase n_s, rebuild optimizer, warmup) | ❌ NOT BUILT | m12 — newly-unfrozen layers get fresh optimizer state, short warmup avoids spikes |
| 20 | Full training iteration per stage | ❌ NOT BUILT | m12 — sample mode m → clip → P_m(x) → views/masks → fwd → loss → backprop unfrozen → EMA |
| 21 | Checkpointing (student + teacher per stage) | ❌ NOT BUILT | m12 — student checkpoint = final "V-JEPA (surgical)" model |
| | **11.7 Quality filters and defaults** | | |
| 22 | Quality filters (drop empty/degenerate samples) | ❌ NOT BUILT | m11 — drop agent-only if A_t ~ 0, layout-only if A_t covers >80%, broken tracklets |
| | **11.8 Verification** | | |
| 23 | Overall retrieval eval (kNN grids, Cycle@K, Overlap@K) | ❌ NOT BUILT | m05+m06+m07+m08 — re-embed with surgical encoder, same pipeline as Ch9 |
| 24 | Factor-sliced retrieval (query D_L/D_A/D_I separately) | ❌ NOT BUILT | m06 — per-factor neighborhoods: layout→layout, agent→agent, interaction→interaction |
| 25 | Sanity check — raw vs patched clips | ❌ NOT BUILT | m06 — gains must transfer to RAW clips; patched-only gains = artifact learning (FAIL) |
| 26 | Final 3-way comparison (frozen vs adapted vs surgical) | ❌ NOT BUILT | m05+m06+m07+m08 — x 15 denseworld keys, side-by-side table for paper |

---

## Final Comparison (The Paper's Punchline)

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     3-WAY COMPARISON: frozen vs adapted vs surgical                     │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  SAME evaluation pipeline for all 3 encoders:                                           │
│                                                                                         │
│  m05 (embed) → m06 (FAISS metrics) → m07 (UMAP) → m08 (plots)                         │
│  SAME 5,105 clips, SAME tags, SAME k=6, SAME Hard/Easy modes                           │
│                                                                                         │
│  ┌────────────────────┬──────────────┬──────────────┬──────────────┐                    │
│  │ Metric             │ Ch 9: Frozen │ Ch 10: Adapt │ Ch 11: Surg  │                    │
│  ├────────────────────┼──────────────┼──────────────┼──────────────┤                    │
│  │ scene_type mAP@K   │ 0.11         │              │              │                    │
│  │ traffic_mix mAP@K  │ (retag)      │              │              │                    │
│  │ ped_veh_sep mAP@K  │ (retag)      │              │              │                    │
│  │ road_encroach mAP  │ (retag)      │              │              │                    │
│  │ lighting mAP@K     │ 0.66         │              │              │                    │
│  │ Cycle@K            │ 78.96%       │              │              │                    │
│  │ nDCG@K             │ 0.90         │              │              │                    │
│  │ Silhouette (scene) │ -0.061       │              │              │                    │
│  │ Prec@K (scene)     │ 18.73%       │              │              │                    │
│  ├────────────────────┼──────────────┼──────────────┼──────────────┤                    │
│  │ DINOv2 baseline    │ ???          │     —        │     —        │                    │
│  │ Random baseline    │ ???          │     —        │     —        │                    │
│  └────────────────────┴──────────────┴──────────────┴──────────────┘                    │
│                                                                                         │
│  PAPER STORY (what each outcome means):                                                 │
│                                                                                         │
│  IF Ch10 improves + Ch11 improves more:                                                 │
│  → "Self-supervised adaptation helps, structured surgery helps MORE.                    │
│     Factor decomposition (layout→agent→interaction) is the right                        │
│     inductive bias for adapting video world models to new domains."                     │
│                                                                                         │
│  IF Ch10 improves + Ch11 ≈ Ch10:                                                       │
│  → "Basic domain data is sufficient. Surgery adds complexity but                        │
│     not value — simpler continual pretraining is recommended."                          │
│                                                                                         │
│  IF Ch10 no improvement + Ch11 improves:                                                │
│  → "Unstructured data exposure fails. The model needs GUIDED                            │
│     exposure to layout/agent/interaction factors separately."                            │
│                                                                                         │
│  IF neither improves:                                                                   │
│  → "Self-supervised adaptation is insufficient for cross-domain                         │
│     transfer. Supervised fine-tuning or architectural changes needed."                  │
│     (Still a publishable negative result!)                                              │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Dependency Graph & Parallelization

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                     DEPENDENCY GRAPH (what blocks what)                                  │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  WEEK 1: Fill Ch 9 gaps                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                                             │
│  │ Random baseline  │  │ DINOv2 baseline  │  (independent, can run in parallel)         │
│  │ ~30 min CPU      │  │ ~3h GPU          │                                             │
│  └────────┬─────────┘  └────────┬─────────┘                                             │
│           │                     │                                                        │
│           └──────────┬──────────┘                                                        │
│                      ▼                                                                   │
│           ┌──────────────────────┐                                                       │
│           │ Re-tag 10K clips     │                                                       │
│           │ with denseworld      │                                                       │
│           │ taxonomy (Qwen)      │                                                       │
│           │ ~5.3h GPU            │                                                       │
│           └──────────┬───────────┘                                                       │
│                      │                                                                   │
│  ════════════════════╪═══════════════════════════════════════════════════                 │
│                      │                                                                   │
│  WEEK 2: Ch 10       │                                                                   │
│  ┌───────────────────▼───────────┐  ┌─────────────────────────────────┐                  │
│  │ Ch 10: Continual pretraining  │  │ Ch 11 PREP (parallel on CPU):  │                  │
│  │ Student-teacher JEPA          │  │ • Write SAM3 pipeline script   │                  │
│  │ ~20h GPU                      │  │ • Write tracklet mining        │                  │
│  │                               │  │ • Write factor dataset builder │                  │
│  │ ALSO (parallel on GPU):       │  │ • Test on SANITY (20 clips)   │                  │
│  │ • SAM3 masks on 10K clips     │  │                               │                  │
│  │   (~10h, can interleave)      │  │                               │                  │
│  └───────────────┬───────────────┘  └─────────────────────────────────┘                  │
│                  │                                                                       │
│                  ▼                                                                       │
│  ┌───────────────────────────────┐                                                       │
│  │ Ch 10 Evaluation              │                                                       │
│  │ Re-run m05→m08                │                                                       │
│  │ Compare: frozen vs adapted    │                                                       │
│  │ ~3h GPU                       │                                                       │
│  └───────────────┬───────────────┘                                                       │
│                  │                                                                       │
│  ════════════════╪═══════════════════════════════════════════════════                     │
│                  │                                                                       │
│  WEEK 3-4: Ch 11 │                                                                       │
│  ┌───────────────▼───────────────┐                                                       │
│  │ Ch 11: Surgery fine-tuning    │                                                       │
│  │ Starts from Ch10 checkpoint   │                                                       │
│  │ (NOT from frozen V-JEPA)      │                                                       │
│  │                               │                                                       │
│  │ Stage 1: Layout  (~18h GPU)   │                                                       │
│  │ Stage 2: Agent   (~18h GPU)   │                                                       │
│  │ Stage 3: Interact (~18h GPU)  │                                                       │
│  └───────────────┬───────────────┘                                                       │
│                  │                                                                       │
│                  ▼                                                                       │
│  ┌───────────────────────────────┐                                                       │
│  │ FINAL Evaluation              │                                                       │
│  │ Re-run m05→m08 on all 3      │                                                       │
│  │ 3-way comparison table        │                                                       │
│  │ ~3h GPU                       │                                                       │
│  └───────────────────────────────┘                                                       │
│                                                                                         │
│  TOTAL: ~65h GPU + ~5h CPU over 4 weeks                                                │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Artifact Flow (What Each Chapter Produces & Consumes)

```mermaid
flowchart LR
    subgraph CH9 ["Ch 9 Artifacts ✅"]
        direction TB
        E1["embeddings.npy<br>frozen · 1408-dim"]
        T1["tags.json<br>denseworld · 16 fields"]
        M1["metrics_frozen.json<br>9 metrics · baseline"]
        P1["15 plots (.png/.pdf)"]
        B1["baselines: Random ·<br>DINOv2 · Shuffled · CLIP"]
    end

    subgraph CH10 ["Ch 10 Artifacts"]
        direction TB
        CK2["checkpoint_adapted.pt<br>student encoder"]
        E2["embeddings_adapted.npy"]
        M2["metrics_adapted.json"]
    end

    subgraph CH11 ["Ch 11 Artifacts"]
        direction TB
        CK3["checkpoint_surgical.pt<br>student encoder"]
        E3["embeddings_surgical.npy"]
        M3["metrics_surgical.json"]
    end

    subgraph PAPER ["Paper Deliverables"]
        direction TB
        D1["3-way comparison table<br>frozen vs adapted vs surgical<br>× 15 denseworld keys"]
        D2["Key finding:<br>lighting ≫ scene semantics<br>Surgery → traffic_mix gain"]
        D1 --> D2
    end

    E1 -->|"val retrieval"| CH10
    T1 -->|"strat. batch"| CH10
    T1 -->|"strat. batch"| CH11
    M1 -->|"baseline"| CH10
    CK2 -->|"init weights"| CH11

    M1 --> D1
    M2 --> D1
    M3 --> D1
    B1 --> D1

    style E1 fill:#43a047,color:#fff,font-weight:bold,font-size:28px
    style T1 fill:#00acc1,color:#fff,font-weight:bold,font-size:28px
    style M1 fill:#d81b60,color:#fff,font-weight:bold,font-size:28px
    style P1 fill:#795548,color:#fff,font-weight:bold,font-size:28px
    style B1 fill:#ff6f00,color:#fff,font-weight:bold,font-size:28px
    style E2 fill:#7b1fa2,color:#fff,font-weight:bold,font-size:28px
    style M2 fill:#7b1fa2,color:#fff,font-weight:bold,font-size:28px
    style CK2 fill:#4a148c,color:#fff,font-weight:bold,font-size:28px
    style E3 fill:#1565c0,color:#fff,font-weight:bold,font-size:28px
    style M3 fill:#1565c0,color:#fff,font-weight:bold,font-size:28px
    style CK3 fill:#0d47a1,color:#fff,font-weight:bold,font-size:28px
    style D1 fill:#bf360c,color:#fff,font-weight:bold,font-size:28px
    style D2 fill:#b71c1c,color:#fff,font-weight:bold,font-size:28px
```

---

## Module Numbering (Existing + Proposed)

| Module | Chapter | Purpose | Status |
|--------|---------|---------|--------|
| m00-m03 | Ch 8 | Data pipeline (YouTube → clips → shards → HF) | DONE (Mac CPU) |
| m00c | Ch 8 | Video-level uniform 10K subset | DONE |
| m04 | Ch 8 | VLM tagging (Qwen/VideoLLaMA/LLaVA) | DONE |
| m04b | Ch 8 | VLM bake-off comparison | DONE |
| m04c | Ch 8 | Sanity comparison dashboard | DONE |
| m05 | Ch 9 | V-JEPA 2 embedding extraction | DONE |
| m06 | Ch 9 | FAISS kNN + 9 metrics | DONE |
| m07 | Ch 9 | cuML GPU UMAP | DONE |
| m08 | Ch 9 | CPU matplotlib plots | DONE |
| **m05b** | **Ch 9** | **Baseline embeddings (random, DINOv2, shuffled, CLIP)** | **TODO** |
| **m09** | **Ch 10** | **Continual pretraining (student-teacher JEPA)** | **TODO** |
| **m10** | **Ch 11** | **SAM3 segmentation + tracklet mining** | **TODO** |
| **m11** | **Ch 11** | **Factor dataset creation (D_L, D_A, D_I)** | **TODO** |
| **m12** | **Ch 11** | **Surgery fine-tuning (3-stage progressive unfreezing)** | **TODO** |

---

## Success Criteria

| Milestone | Criteria | When |
|-----------|----------|------|
| **Ch 9 complete** | Baselines done. V-JEPA Prec@K significantly above random. | Week 1 |
| **Ch 10 POC** | Adapted Cycle@K ≥ frozen. scene_type mAP improves. traffic_mix/ped_veh_sep show signal. | Week 2 |
| **Ch 11 POC** | Surgical > adapted on factor-specific metrics. Gains transfer to RAW (unpatched) clips. | Week 4 |
| **Paper-ready** | 3-way comparison table with baselines. 15-key denseworld breakdown. All plots reproducible. | Week 5 |
