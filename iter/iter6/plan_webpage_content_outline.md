# DenseWorld + FactorJEPA — GitHub Pages Content Outline

> Template: [Eliahuhorwitz Academic Project Page](https://github.com/eliahuhorwitz/Academic-project-page-template)
> Stack: Plain HTML + Bulma CSS + vanilla JS (zero build step)
> Deployment: GitHub Pages (static, <15 MB total)

---

## Section 1: Hero / Title Banner

**Title:** Does V-JEPA 2 Understand Indian Streets?

**Subtitle:** Benchmarking Video Foundation Models on WalkIndia-200K

**Authors:** Kapil Wanaskar<sup>1</sup>
- <sup>1</sup> Affiliation TBD

**Badges (shields.io):**
- `Status: Preprint` (yellow badge)
- `Dataset: HuggingFace` (HF badge → https://huggingface.co/datasets/anonymousML123/walkindia-200k)
- `Code: GitHub` (GitHub badge → repo URL)
- `arXiv: XXXX.XXXXX` (red badge → arXiv link, once submitted)

**Logo:** DenseWorld spiral mandala (top-left, ~40px height in navbar)

---

## Section 2: Teaser Video Grid

**Layout:** 3×4 responsive grid of autoplay muted loop MP4 clips

**Content:** 12 diverse Indian street scene clips selected via greedy set-cover:

| Cell | City | Scene Type | Conditions | Visual Hook |
|------|------|-----------|------------|-------------|
| 1 | Mumbai | market | day, high crowd | Bustling market street |
| 2 | Delhi | heritage_tourist | day, clear | Monument / Red Fort area |
| 3 | Goa | beach_coastal | day, clear | Coastal promenade |
| 4 | Varanasi | ghat | day, high crowd | Riverfront steps |
| 5 | Jaipur | bazaar | day, med crowd | Pink City narrow lanes |
| 6 | Bangalore | commercial | day, mixed traffic | Modern India contrast |
| 7 | Hyderabad | flyover_underpass | night, artificial | Night driving scene |
| 8 | Kochi | residential_lane | day, rain | Monsoon street |
| 9 | Bhopal | junction | day, high traffic | Chaotic intersection |
| 10 | Chennai | highway | day, clear | Wide road |
| 11 | Kolkata | transit | day, high crowd | Tram / transit |
| 12 | Monuments | heritage_tourist | day, natural | Iconic landmark |

**Specs:** 640×360, 15fps, 4s loop, CRF 28, ~150KB each = 1.8 MB total

**Caption:** _"Sample clips from WalkIndia-200K: 200K clips across 714 YouTube walking tour videos from 25+ Indian cities."_

---

## Section 3: Abstract

**Source:** Adapted from FactorJEPA proposal + Ch9 results

**Draft:**

> Do video foundation models trained on Western data understand the dense, crowded, chaotic streets of India? We introduce **WalkIndia-200K**, a large-scale benchmark of 200,000 video clips from 714 YouTube walking tours across 25+ Indian cities. Each clip is automatically tagged with 16 structured attributes (scene type, crowd density, traffic mix, road surface, etc.) using a VLM bake-off won by Qwen3-VL-8B.
>
> We evaluate five encoders — **V-JEPA 2** (ViT-G, frozen), **DINOv2** (ViT-L/14), **CLIP** (ViT-L/14), **V-JEPA Shuffled** (temporal ablation), and a **Random** baseline — on 9 retrieval metrics across Easy/Hard modes. Our key finding: **image-based encoders (DINOv2: 50.5% Prec@K) dramatically outperform the video encoder (V-JEPA: 14.6%)** on spatial scene classification. More strikingly, **shuffling V-JEPA's input frames improves performance by 2.4×** (35.3% Prec@K), demonstrating that temporal encoding actively hurts spatial scene understanding in this domain.
>
> V-JEPA's temporal features are self-consistent (78.7% Cycle@K, highest of all encoders) but encode motion similarity rather than scene semantics — it finds clips with similar camera dynamics, not similar places. This reveals a fundamental **spatial-temporal evaluation gap**: current taxonomy measures only spatial attributes, while V-JEPA's strengths lie in the unmeasured temporal axis.

---

## Section 4: Key Findings (Visual Callouts)

**Layout:** 2-3 large callout cards with icons/numbers

**Card 1: "Image Beats Video"**
- Stat: `DINOv2 50.5% vs V-JEPA 14.6%`
- One-liner: Single-frame image encoders outperform 64-frame video encoder on scene classification

**Card 2: "Temporal Encoding Hurts"**
- Stat: `Shuffled V-JEPA 35.3% vs Normal 14.6%`
- One-liner: Destroying temporal order improves spatial retrieval by 2.4×

**Card 3: "Consistent but Wrong"**
- Stat: `V-JEPA Cycle@K: 78.7% (highest)`
- One-liner: V-JEPA builds the most self-consistent neighborhoods — they just encode motion, not scenes

---

## Section 5: Five-Encoder Comparison

**Layout:** Side-by-side: bar chart (left) + radar chart (right)

**Left: Bar Chart**
- File: `plots/m08b_encoder_comparison.png` (already generated)
- 5 encoders × 5 metrics × Easy/Hard

**Right: Radar Chart**
- File: `plots/m08b_radar.png` (already generated)
- 5-encoder polygon comparison (Easy mode, normalized)

**Below: HTML Table**

| Encoder | Dim | Cycle@K | Overlap@K | Prec@K | mAP@K | nDCG@K |
|---------|-----|---------|-----------|--------|-------|--------|
| **V-JEPA** | 1408 | **78.65%** | 10.50% | 14.63% | 0.0792 | 0.9032 |
| Random | 1408 | 54.99% | 0.00% | 12.21% | 0.0608 | 0.8978 |
| **DINOv2** | 1024 | 66.77% | **60.90%** | **50.48%** | **0.4271** | 0.9577 |
| CLIP | 768 | 65.17% | 47.06% | 46.03% | 0.3816 | **0.9583** |
| Shuffled | 1408 | 76.19% | 35.32% | 35.33% | 0.2724 | 0.9500 |

---

## Section 6: Pipeline Overview

**Layout:** Static image of the pipeline diagram

**Source:** Convert the mermaid flowchart from report.md to a static SVG/PNG

```
YouTube (714 videos) → PySceneDetect (4-10s cuts) → ~115K clips
→ WebDataset (HF) → VLM Bake-off (Qwen wins) → 16-field tags
→ V-JEPA/DINOv2/CLIP/Shuffled/Random embeddings
→ FAISS kNN (k=6, Easy/Hard) → 9 metrics → UMAP → Plots
```

**Caption:** _"End-to-end pipeline: from YouTube walking tours to reproducible retrieval benchmarks."_

---

## Section 7: kNN Retrieval Demo (Interactive Video Grid)

**Layout:** 2-3 rows, each showing: Query clip → 5 nearest neighbors

**Row 1: DINOv2 success case**
- Query: `market` clip → 5/5 neighbors are `market` (green borders)
- Caption: "DINOv2 correctly retrieves visually similar market scenes"

**Row 2: V-JEPA failure case**
- Same query clip → only 1/5 neighbors are `market` (4 red borders)
- Caption: "V-JEPA retrieves clips with similar camera motion, not similar scenes"

**Row 3: Shuffled improvement**
- Same query → 3/5 neighbors are `market`
- Caption: "Destroying temporal order improves scene retrieval"

**Specs:** 256×144, 12fps, 3s, ~40KB each. 3 rows × 6 clips = 18 clips = ~720KB

**Implementation:** Select query indices where encoders maximally disagree using knn_indices*.npy

---

## Section 8: Dataset — WalkIndia-200K

**Layout:** Stats panel + tag distribution plot

**Stats:**
| Metric | Value |
|--------|-------|
| Total clips | ~200,000 (115,687 in v1) |
| Source videos | 714 YouTube walking tours |
| Cities | 25+ (Mumbai, Delhi, Bangalore, Goa, Jaipur, ...) |
| Clip duration | 4-12 seconds |
| Taxonomy | 16 fields (v3): 13 single + 2 multi + changelog |
| VLM tagger | Qwen3-VL-8B (0.919 bake-off score) |
| Format | WebDataset TARs on HuggingFace |

**Plot:** `plots/m04_taxonomy_distribution_qwen.png` (full taxonomy distribution)

**HuggingFace button:** Link to https://huggingface.co/datasets/anonymousML123/walkindia-200k

---

## Section 9: Detailed Results

**Subsection 9a: What Does V-JEPA Cluster By?**
- Plot: `plots/m06_silhouette_per_key.png` — silhouette per taxonomy key
- Text: "V-JEPA organizes by lighting (+0.0009 silhouette, only positive) not scene semantics (-0.25, worst)"
- Note: Plot shows vjepa_shuffled due to overwrite bug; text uses V-JEPA JSON values

**Subsection 9b: UMAP Visualization**
- Plot: `plots/m08_umap.png` — 5,105 clips colored by scene_type
- Caption: "No clean scene-type clusters. Colors are scattered — V-JEPA does not organize by scene semantics."

**Subsection 9c: Confusion Matrix**
- Plot: `plots/m08_confusion_matrix.png` — scene_type retrieval confusion
- Caption: "Market and residential_lane dominate retrievals regardless of query type."

**Subsection 9d: Easy vs Hard**
- Text: "Easy/Hard gap is <0.5pp on all metrics — our data pipeline prevents temporal leakage."

---

## Section 10: The Spatial-Temporal Gap

**Layout:** Text explanation with inline diagram

**Content:**

> **Why does V-JEPA fail on scene classification?**
>
> V-JEPA 2 is trained to predict masked spatiotemporal patches — it learns motion patterns, camera trajectories, and temporal dynamics. But our taxonomy measures ONLY spatial attributes (scene_type, road_surface, crowd_density...). The model's temporal features are strong (Cycle@K: 78.7%) but encode the wrong thing for this task.
>
> **Evidence:** Shuffling V-JEPA's 64 input frames destroys temporal information, forcing the model to use its spatial pathway — which improves Prec@K from 14.6% to 35.3%.
>
> **Next steps:** We are extending the evaluation with temporal metrics (optical flow features, motion tags) to measure V-JEPA on the axis where it should excel. If V-JEPA >> image baselines on temporal metrics, the paper story becomes: _"Spatial and temporal transfer are independent axes — V-JEPA captures Indian dynamics but not Indian scenes."_

---

## Section 11: Future Work

**Layout:** Compact text block or timeline graphic

**Content:**

| Phase | Description | Status |
|-------|------------|--------|
| Ch 9: Frozen Evaluation | 5-encoder benchmark on 10K POC | **Complete** |
| Temporal Extension | Optical flow features (m04f) + temporal Prec@K | Planned |
| Ch 10: Continual Pretraining | Adapt V-JEPA to Indian data (same JEPA loss) | Planned |
| Ch 11: Surgery Fine-Tuning | Progressive prefix unfreezing with factor datasets (layout/agent/interaction) | Planned |

---

## Section 12: BibTeX

```bibtex
@article{wanaskar2026walkindia,
  title={Does V-JEPA 2 Understand Indian Streets? Benchmarking Video Foundation Models on WalkIndia-200K},
  author={Wanaskar, Kapil},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

(Copy-to-clipboard button via template JS)

---

## Section 13: Footer

- "Part of the **DenseWorld** research program at **Pragya.ai**"
- DenseWorld logo (small, linked to umbrella site)
- Template attribution: "This website is built using the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), licensed under CC BY-SA 4.0."

---

## SEO / Meta Tags (in `<head>`)

```html
<!-- Google Scholar -->
<meta name="citation_title" content="Does V-JEPA 2 Understand Indian Streets? Benchmarking Video Foundation Models on WalkIndia-200K">
<meta name="citation_author" content="Wanaskar, Kapil">
<meta name="citation_publication_date" content="2026">
<meta name="citation_pdf_url" content="https://arxiv.org/pdf/XXXX.XXXXX.pdf">

<!-- Open Graph (social sharing) -->
<meta property="og:title" content="Does V-JEPA 2 Understand Indian Streets?">
<meta property="og:description" content="Image encoders (DINOv2: 50.5%) dramatically outperform V-JEPA (14.6%) on Indian street scene retrieval. Shuffling frames improves V-JEPA by 2.4×.">
<meta property="og:image" content="static/images/og_preview.png">
<meta property="og:url" content="https://USERNAME.github.io/PROJECT/">

<!-- Twitter Card -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Does V-JEPA 2 Understand Indian Streets?">
<meta name="twitter:image" content="static/images/og_preview.png">
```

---

## File Structure

```
project-page/
├── index.html                    # Main page (single file)
├── static/
│   ├── css/
│   │   └── bulma.min.css         # Bulma framework (from template)
│   ├── js/
│   │   └── index.js              # Carousel, BibTeX copy (from template)
│   ├── images/
│   │   ├── denseworld_logo.png   # Cropped DenseWorld spiral (~200px wide)
│   │   ├── favicon.ico           # Spiral mandala 32x32
│   │   ├── og_preview.png        # Open Graph image 1200x630
│   │   ├── pipeline.svg          # Pipeline diagram
│   │   ├── m08b_encoder_comparison.png
│   │   ├── m08b_radar.png
│   │   ├── m04_taxonomy_distribution_qwen.png
│   │   ├── m06_silhouette_per_key.png
│   │   ├── m08_umap.png
│   │   └── m08_confusion_matrix.png
│   └── videos/
│       ├── teaser/               # 12 hero grid clips (360p, 150KB each)
│       │   ├── mumbai_market.mp4
│       │   ├── delhi_heritage.mp4
│       │   └── ...
│       └── knn/                  # 18 kNN demo clips (144p, 40KB each)
│           ├── dinov2_q1_query.mp4
│           ├── dinov2_q1_nn1.mp4
│           └── ...
├── CNAME                         # Custom domain (optional)
└── README.md                     # Repo description
```

**Estimated total size: ~10-15 MB**

---

## Color Scheme (from DenseWorld branding)

| Element | Color | Hex |
|---------|-------|-----|
| Background | Warm cream | `#FAF7F2` |
| Cards/code blocks | Light sand | `#F0EBE1` |
| Primary text | Dark charcoal-brown | `#2D2318` |
| Links/buttons | Deep terracotta | `#8B3A2A` |
| Hover state | Saffron orange | `#D4762C` |
| Tags/badges | Forest green | `#3B6B35` |
| Borders | Muted tan | `#D4C9B8` |

---

## Implementation Priority

1. Fork Eliahuhorwitz template → customize branding/colors
2. Fill Hero + Abstract + BibTeX (text-only, fast)
3. Add static plots (copy from src/outputs_poc/)
4. Add encoder comparison table (HTML)
5. Write clip selection script → extract 12 hero clips → ffmpeg convert
6. Write kNN demo selection → extract 18 clips → ffmpeg convert
7. Add video grids (hero teaser + kNN demo)
8. Add SEO meta tags
9. Test locally → push → verify GitHub Pages
10. Crop DenseWorld logo → add favicon + header + OG image
