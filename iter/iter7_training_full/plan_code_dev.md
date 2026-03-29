# Ch10 Full (115K): Terminal Commands & Duration Estimates

> All code is ready. Run on a fresh GPU instance with ≥200GB disk.

---

## System Design: What Calls What

### run_evaluate.sh --FULL (Ch9: frozen encoder evaluation)

```
run_evaluate.sh --FULL
│
├── PREFLIGHT: output_guard.py preflight_evaluate
│
├── m00d_download_subset.py ──→ data/full_local/ (120GB, one-time)
│
├── m04_vlm_tag.py ──→ tags.json                ⚠️ BIGGEST BOTTLENECK
│
├── m05_vjepa_embed.py ──→ embeddings.npy        ⚠️ 2ND BOTTLENECK
│
├── m05b_baselines.py ──→ embeddings_{dinov2,clip,random,shuffled}.npy
│
├── m05c_true_overlap.py ──→ overlap_augA/B.npy
│
├── m04d_motion_features.py ──→ motion_features.npy
│
├── FOR EACH encoder (vjepa, random, dinov2, clip, vjepa_shuffled):
│   ├── m06_faiss_metrics.py ──→ m06_metrics_{enc}.json
│   ├── m06b_temporal_corr.py ──→ m06b_temporal_corr_{enc}.json
│   ├── m07_umap.py ──→ umap_2d_{enc}.npy
│   └── m08_plot.py ──→ plots
│
└── m08b_compare.py ──→ m08b_radar.png (5 frozen encoders)
```

### run_pretrain.sh --FULL (Ch10: continual pretraining + eval)

```
run_pretrain.sh --FULL
│
├── PREFLIGHT: output_guard.py preflight_pretrain
│
├── PHASE 1: Train λ=0.001 (fixed, no ablation — POC proved all 4 identical)
│   └── m09_pretrain.py --lambda-reg 0.001 --max-epochs 5
│       └── student_encoder.pt + training_summary.json
│
├── PHASE 2: Re-embed + Metrics
│   ├── m05_vjepa_embed.py --model adapted --encoder vjepa_lambda0_001
│   │   └── embeddings_vjepa_lambda0_001.npy          ⚠️ BOTTLENECK
│   └── m06_faiss_metrics.py --encoder vjepa_lambda0_001
│       └── m06_metrics_vjepa_lambda0_001.json
│
└── PHASE 3: Full Evaluation
    ├── m06b_temporal_corr.py (adapted)
    ├── m05_vjepa_embed.py --shuffle (shuffled adapted) ⚠️ BOTTLENECK
    ├── m06_faiss_metrics.py (shuffled adapted)
    ├── m07_umap.py (adapted)
    ├── m08_plot.py (adapted)
    └── m08b_compare.py (7 encoders) ──→ m08b_radar.png
```

---

## Combined Duration Table (115K FULL)

> **Basis:** All GPU times extrapolated from 10K POC measured rates.
> m05: 1.55 clips/s (measured). m09: 25s/step (measured). m04: ~3 clips/min (measured).

### run_evaluate.sh --FULL (must run FIRST)

| # | Script | GPU? | What | Rate (measured on 10K) | 115K Est. | Cumulative |
|:-:|--------|:----:|------|:----------------------:|:---------:|:----------:|
| 0 | m00d_download_subset.py | CPU | Download 116 TARs | ~30 MB/s | **1.1h** | 1.1h |
| 1 | m04_vlm_tag.py | GPU | Qwen3-VL tagging | ~3 clips/min | **~35h** | 36h |
| 2 | m05_vjepa_embed.py | GPU | V-JEPA frozen embed | 1.55 clips/s | **~20.6h** | 57h |
| 3a | m05b (random) | CPU | Random embeddings | instant | <1 min | 57h |
| 3b | m05b (dinov2) | GPU | DINOv2 embed | ~5 clips/s | **~6.4h** | 63h |
| 3c | m05b (clip) | GPU | CLIP embed | ~8 clips/s | **~4h** | 67h |
| 3d | m05b (shuffled) | GPU | Shuffled V-JEPA embed | 1.55 clips/s | **~20.6h** | 88h |
| 4 | m05c_true_overlap.py | GPU | Augmented A+B embeds | 1.55 clips/s × 2 | **~11.5h** | 99h |
| 4.5 | m04d_motion_features.py | GPU | RAFT optical flow | ~4 clips/s | **~8h** | 107h |
| 5 | m06_faiss_metrics.py ×5 | GPU | FAISS kNN metrics | fast | ~25 min | 107.5h |
| 5.5 | m06b_temporal_corr.py ×5 | CPU | Temporal correlation | fast | ~50 min | 108h |
| 6 | m07_umap.py ×5 | GPU | cuML UMAP | fast | ~75 min | 109h |
| 7 | m08_plot.py ×5 + m08b | CPU | Plots + compare | fast | ~30 min | **110h** |

### run_pretrain.sh --FULL (run AFTER evaluate)

| # | Script | GPU? | What | Rate (measured on 10K) | 115K Est. | Cumulative |
|:-:|--------|:----:|------|:----------------------:|:---------:|:----------:|
| 1 | m09_pretrain.py | GPU | Train λ=0.001, 5 ep | 25s/step, 929 steps/ep | **~32h** | 32h |
| 2a | m05_vjepa_embed.py | GPU | Re-embed adapted | 1.55 clips/s | **~20.6h** | 53h |
| 2b | m06_faiss_metrics.py | GPU | Adapted metrics | fast | ~5 min | 53h |
| 3a | m06b_temporal_corr.py | CPU | Adapted temporal | fast | ~10 min | 53h |
| 3b | m05_vjepa_embed.py --shuffle | GPU | Shuffled adapted embed | 1.55 clips/s | **~20.6h** | 74h |
| 3c | m06_faiss_metrics.py | GPU | Shuffled adapted metrics | fast | ~5 min | 74h |
| 3d | m07_umap.py | GPU | Adapted UMAP | fast | ~15 min | 74h |
| 3e | m08_plot.py | CPU | Adapted plots | fast | ~5 min | 74h |
| 3f | m08b_compare.py | CPU | 7-encoder radar | fast | ~30s | **74h** |

### Grand Total

| Pipeline | Duration |
|----------|:--------:|
| run_evaluate.sh --FULL | **~110h** |
| run_pretrain.sh --FULL | **~74h** |
| **Sequential total** | **~184h (~7.7 days)** |
| **Parallel (if 2 GPUs)** | **~110h (~4.6 days)** |

### Top 5 Bottlenecks (sorted by time)

| Rank | Script | Time | % of Total |
|:----:|--------|:----:|:----------:|
| 1 | m04_vlm_tag.py (115K tags) | **35h** | 19% |
| 2 | m09_pretrain.py (5 epochs) | **32h** | 17% |
| 3 | m05_vjepa_embed.py (×4 runs total) | **~82h** | 45% |
| 4 | m05c_true_overlap.py | **11.5h** | 6% |
| 5 | m04d_motion_features.py | **8h** | 4% |

**m05 V-JEPA embedding (1.55 clips/s) dominates at 45% of total time.** Any speedup here (torch.compile fix, larger BS, torchcodec) would have the biggest impact.

---

## Terminal Commands

### Step 0: Setup (one-time, ~10 min)

```bash
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
```

### Step 1: Download 115K corpus (~1h, ~120GB)

```bash
python -u src/m00d_download_subset.py --FULL 2>&1 | tee logs/m00d_full.log
```

### Step 2: Ch9 Eval on 115K (~110h)

```bash
tmux new -s ch9
./scripts/run_evaluate.sh --FULL 2>&1 | tee logs/ch9_full.log
# Ctrl+B, D to detach
```

### Step 3: Ch10 Pretrain on 115K (~74h)

```bash
tmux new -s ch10
./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log
```

### Step 4: RSYNC to Mac

```bash
# From Mac:
rsync -avz --progress gpu:/workspace/factorjepa/{src,scripts,configs,.claude,docs,iter} ~/factorjepa/
rsync -avz --progress gpu:/workspace/factorjepa/outputs/full/ ~/factorjepa/outputs/full/
rsync -avz --progress gpu:/workspace/factorjepa/logs/ ~/factorjepa/logs/
```

---

## Disk Budget

| Item | Size |
|------|:----:|
| data/full_local/ (115K clips) | ~120 GB |
| checkpoints/vitg-384.pt | 16 GB |
| Training peak (student+teacher+optimizer) | ~40 GB |
| Embeddings (7 encoders × 115K × 1408 × 4B) | ~4.2 GB |
| outputs/full/ (metrics, plots, UMAP) | ~5 GB |
| tags.json (115K × 16 fields) | ~50 MB |
| **Total needed** | **~200 GB** |
