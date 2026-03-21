# FactorJEPA: Does V-JEPA 2 Understand Indian Streets?

**Benchmarking Video Foundation Models on DenseWorld-200K**

[![Project Page](https://img.shields.io/badge/Project-Page-8B3A2A)](https://kapilw25.github.io/factorjepa/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-ffd21e?logo=huggingface)](https://huggingface.co/datasets/anonymousML123/walkindia-200k)
[![Status](https://img.shields.io/badge/Status-Preprint-yellow)]()

> **115,687 clips** | 714 videos | 22 cities | 276 hours | 121 GB

**[See 192 video clips from all cities and taxonomy categories on the project page](https://kapilw25.github.io/factorjepa/)**

---

## Key Finding

| Encoder | Prec@K | mAP@K | Cycle@K |
|---------|--------|-------|---------|
| **DINOv2** (image) | **50.5%** | **0.427** | 66.8% |
| CLIP (image) | 46.0% | 0.382 | 65.2% |
| V-JEPA Shuffled | 35.3% | 0.272 | 76.2% |
| **V-JEPA 2** (video) | 14.6% | 0.079 | **78.7%** |
| Random | 12.2% | 0.061 | 55.0% |

**Image beats video on spatial metrics.** DINOv2 outperforms V-JEPA by 3.5x on scene classification. Shuffling V-JEPA's frames improves it by 2.4x — temporal encoding hurts spatial understanding. But V-JEPA wins Cycle@K (78.7%), the most temporally-sensitive metric.

---

## Setup

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
./setup_env_uv.sh --gpu          # Nvidia GPU server (installs PyTorch, FAISS-GPU, cuML, FA2)
# or: ./setup_env_uv.sh --mac    # M1 Mac (CPU-only, for development/testing)
source venv_walkindia/bin/activate
```

---

## Pipeline

Three stages, each a single command. All stages use checkpoint/resume — safe to interrupt and restart.

### Evaluate frozen encoders (~7h GPU)

5 encoders (V-JEPA 2 + DINOv2 + CLIP + Shuffled + Random), 9 spatial metrics + 5 temporal metrics, bootstrap 95% CI, comparison plots.

```bash
./scripts/run_evaluate.sh --FULL
```

Covers: `m00d` (data download) &#8594; `m04` (VLM tagging) &#8594; `m05/m05b/m05c` (embeddings) &#8594; `m04d` (RAFT motion features) &#8594; `m06/m06b` (spatial + temporal metrics) &#8594; `m07` (UMAP) &#8594; `m08/m08b` (plots + comparison)

### Continual pretraining (TODO, ~20h GPU)

Self-supervised JEPA loss on Indian clips. Student-teacher with EMA, stratified sampling by v3 taxonomy.

```bash
./scripts/run_pretrain.sh --FULL
```

Covers: `m09` (continual pretraining) &#8594; re-run evaluation pipeline with adapted encoder

### Representation surgery (TODO, ~54h GPU)

Progressive prefix unfreezing with factor datasets (Layout &#8594; Agent &#8594; Interaction) from SAM3 segmentation.

```bash
./scripts/run_surgery.sh --FULL
```

Covers: `m10` (SAM3 + tracklets) &#8594; `m11` (factor datasets) &#8594; `m12` (3-stage surgery) &#8594; re-run evaluation

---

## Dataset

| Tier | Cities | Clips | Hours | GB |
|------|--------|-------|-------|----|
| Tier 1 | 6 metros | 68,614 | 161h | 74 |
| Goa | 1 | 5,835 | 14h | 6 |
| Tier 2 | 15 cities | 40,743 | 99h | 41 |
| Monuments | 3 | 495 | 1h | 1 |
| **Total** | **22** | **115,687** | **276h** | **121** |

---

## Code Structure

```
src/
├── m00-m03          # Data pipeline (YouTube → clips → WebDataset → HF)
├── m04              # VLM tagging (Qwen3-VL-8B, 16-field taxonomy)
├── m04d             # GPU-RAFT optical flow (13D motion features)
├── m05/m05b/m05c    # Embeddings (V-JEPA + 4 baselines + True Overlap)
├── m06/m06b         # Spatial metrics (FAISS) + temporal correlation
├── m07              # UMAP (cuML GPU)
├── m08/m08b         # Plots + multi-encoder comparison
└── utils/           # Config, bootstrap CI, gpu_batch, wandb
```

---

## Authors

Kapil Wanaskar¹, Gaytri Jena⁴, Vinija Jain², Aman Chadha³, Amitava Das⁴

¹Canva Research, USA · ²Google, USA · ³Apple, USA · ⁴Pragya Lab, BITS Pilani Goa, India

Part of the **DenseWorld** research program — *World Models for Populous, Crowded, and Chaotic Global South*

## Citation

```bibtex
@article{wanaskar2026factorjepa,
  title={Does V-JEPA 2 Understand Indian Streets? Benchmarking Video Foundation Models on DenseWorld-200K},
  author={Wanaskar, Kapil and Jena, Gaytri and Jain, Vinija and Chadha, Aman and Das, Amitava},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
