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

**Image beats video.** DINOv2 (50.5%) outperforms V-JEPA (14.6%) by 3.5x on scene classification. Shuffling V-JEPA's frames improves it by **2.4x** — temporal encoding hurts spatial understanding.

---

## Reproduce: Full Evaluation Pipeline

### 1. Setup

```bash
# GPU server (Nvidia)
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
./setup_env_uv.sh --gpu
source venv_walkindia/bin/activate
```

### 2. Pre-download 10K subset (CPU, ~11 min)

```bash
python -u src/m00d_download_subset.py --subset data/subset_10k.json 2>&1 | tee logs/m00d_download.log
```

### 3. VLM tagging — Qwen3-VL-8B (GPU, ~2h)

```bash
python -u src/m04_vlm_tag.py --model qwen --FULL --subset data/subset_10k.json \
    --local-data data/subset_10k_local 2>&1 | tee logs/m04_full_qwen_poc.log
```

### 4. Embeddings — V-JEPA 2 + 4 baselines (GPU, ~4h)

```bash
# V-JEPA 2 ViT-G (1408-dim, ~80 min)
python -u src/m05_vjepa_embed.py --FULL --subset data/subset_10k.json \
    --local-data data/subset_10k_local 2>&1 | tee logs/m05_vjepa_embed_poc.log

# 4 baselines: Random + DINOv2 + CLIP + Shuffled (~100 min)
python -u src/m05b_baselines.py --encoder all --FULL --subset data/subset_10k.json \
    --local-data data/subset_10k_local 2>&1 | tee logs/m05b_all_poc.log

# True Overlap@K augmented embeddings (~90 min)
python -u src/m05c_true_overlap.py --FULL --subset data/subset_10k.json \
    --local-data data/subset_10k_local 2>&1 | tee logs/m05c_overlap_poc.log
```

### 5. FAISS metrics — 5 encoders (GPU, ~3 min)

```bash
python -u src/m06_faiss_metrics.py --encoder vjepa --true-overlap \
    --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_vjepa_poc.log

for enc in random dinov2 clip vjepa_shuffled; do
    python -u src/m06_faiss_metrics.py --encoder $enc \
        --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_${enc}_poc.log
done
```

### 6. UMAP + Plots (GPU + CPU, ~2 min)

```bash
# UMAP (GPU — cuML)
for enc in vjepa random dinov2 clip vjepa_shuffled; do
    python -u src/m07_umap.py --encoder $enc --FULL --subset data/subset_10k.json \
        2>&1 | tee logs/m07_umap_${enc}_poc.log
done

# Plots (CPU — matplotlib)
python -u src/m08_plot.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m08_plot_poc.log
python -u src/m08b_compare.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m08b_compare.log
```

### Total: ~6.5 hours on RTX PRO 6000 (96GB)

---

## Dataset Stats

```bash
python -u src/m02b_scene_fetch_duration.py --stats
```

| Tier | Cities | Clips | Hours | GB |
|------|--------|-------|-------|----|
| Tier 1 | 6 metros | 68,614 | 161h | 74 |
| Goa | 1 | 5,835 | 14h | 6 |
| Tier 2 | 15 cities | 40,743 | 99h | 41 |
| Monuments | 3 | 495 | 1h | 1 |
| **Total** | **22** | **115,687** | **276h** | **121** |

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
