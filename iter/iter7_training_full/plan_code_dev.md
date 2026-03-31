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

### Step 1: Spin up GPU, clone repo

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
```

### Step 2: Setup venv (NOT automatic — scripts fail without it)

```bash
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
```

### Step 3: Rsync data from Mac (FROM YOUR MAC terminal, ~17 min)

```bash
# Transfers POC 10K (9.7GB) + val 1K (0.9GB) + JSON manifests
rsync -avhP data/ vast_RTXpro6000_96GB:/workspace/factorjepa/data/
```

### Step 3.5: vLLM smoke test (on GPU, ~5 min)

`setup_env_uv.sh` (Step 2) already creates `venv_vllm` automatically. Just verify it works:

```bash
# Smoke test: 1 image + 1 video (~3 min)
source venv_vllm/bin/activate
python scripts/smoke_test_vllm.py 2>&1 | tee logs/vllm_smoke.log
```

If pass → `--vllm` flag is available for `run_evaluate.sh` (3-5x faster m04 tagging).
If fail → debug from error, see `iter/iter7_training_full/plan_vLLM_Qwen.md`.
If vLLM never works → no problem, `run_evaluate.sh` without `--vllm` uses transformers.

### Step 4: Sanity check (on GPU, ~5 min each)

```bash
source venv_walkindia/bin/activate
./scripts/run_evaluate.sh --SANITY 2>&1 | tee logs/ch9_sanity.log
./scripts/run_pretrain.sh --SANITY 2>&1 | tee logs/ch10_sanity.log
```

### Step 5: POC lambda ablation — m09 only (~2h on GPU)

```bash
source venv_walkindia/bin/activate

# Train 4 lambdas on 10K clips (1 epoch each, ~30 min per lambda)
for LAM in 0 0.001 0.01 0.1; do
  LAM_STR=$(echo $LAM | tr '.' '_')
  python -u src/m09_pretrain.py --config configs/pretrain/vitg16_indian.yaml \
    --POC --subset data/subset_10k.json --val-subset data/val_1k.json \
    --local-data data/subset_10k_local --val-local-data data/val_1k_local \
    --lambda-reg $LAM 2>&1 | tee logs/m09_poc_lambda${LAM_STR}.log
done

# Compare val/jepa_loss across 4 lambdas (check WandB or training_summary.json)
# Pick the lambda with lowest val loss → use for --FULL
```

Output: `outputs/poc/m09_lambda{X}/training_summary.json` → compare `final_val_loss` across 4 runs.

### Step 6: Full 115K run on GPU

**Option A: with vLLM (if smoke test passed) — ~75h total**

```bash
tmux new -s full
source venv_walkindia/bin/activate

# --vllm: uses venv_vllm/bin/python for m04 tagging (3-5x faster)
# If vLLM fails, auto-falls back to transformers
./scripts/run_evaluate.sh --FULL --vllm 2>&1 | tee logs/ch9_full.log
./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log
# Ctrl+B, D to detach | tmux attach -t full to reconnect
```

**Option B: without vLLM (if vLLM fails) — ~184h total**

```bash
tmux new -s full
source venv_walkindia/bin/activate

# Default: uses m04_vlm_tag.py (transformers) — always works
./scripts/run_evaluate.sh --FULL 2>&1 | tee logs/ch9_full.log
./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log
```

Note: `--vllm` calls `venv_vllm/bin/python src/m04_vlm_tag_vllm.py` for m04 only.
All other steps (m05-m08b) use `venv_walkindia` python regardless.
If vLLM fails at runtime, `run_evaluate.sh` auto-falls back to transformers.

### Step 7: Push results to Mac via git

```bash
# On GPU:
./git_push.sh "Ch9+Ch10 full 115K results"
# On Mac:
git pull
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
