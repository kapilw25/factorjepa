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

> **Basis:** Measured rates on 115K FULL run (March 2026, RTX PRO 6000 102GB).
> m00d: 24 min measured (4 workers). m04: 1.15 clips/s measured (transformers, adaptive batch 64).
> m05: 1.55 clips/s (POC measured). m09: 25s/step (POC measured).

### run_evaluate.sh --FULL (Ch9)

| | # | Script | What | Rate | 115K Est. | Cumul. |
|:-:|:-:|--------|------|:----:|:---------:|:------:|
| ✅ | 0 | m00d_download_subset.py | Download 116 TARs (8 workers) | 12.3 s/shard (measured) | **24 min** | 0.4h |
| 🔄 | 1 | m04_vlm_tag.py | Qwen3-VL tagging (transformers, BS=64) | 1.15 clips/s (measured) | **~28h** | 28h |
| ⏳ | 2 | m05_vjepa_embed.py | V-JEPA frozen embed | 1.55 clips/s (est. from 10K) | **~20.6h** | 49h |
| ⏳ | 3a | m05b (random) | Random embeddings | instant | <1 min | 49h |
| ⏳ | 3b | m05b (dinov2) | DINOv2 embed | ~5 clips/s (est. from 10K) | **~6.4h** | 55h |
| ⏳ | 3c | m05b (clip) | CLIP embed | ~8 clips/s (est. from 10K) | **~4h** | 59h |
| ⏳ | 3d | m05b (shuffled) | Shuffled V-JEPA embed | 1.55 clips/s (est. from 10K) | **~20.6h** | 80h |
| ⏳ | 4 | m05c_true_overlap.py | Augmented A+B embeds | 1.55 clips/s × 2 (est. from 10K) | **~11.5h** | 91h |
| ⏳ | 4.5 | m04d_motion_features.py | RAFT optical flow | ~4 clips/s (est. from 10K) | **~8h** | 99h |
| ⏳ | 5 | m06_faiss_metrics.py ×5 | FAISS kNN metrics | fast | ~25 min | 100h |
| ⏳ | 5.5 | m06b_temporal_corr.py ×5 | Temporal correlation | fast | ~50 min | 100h |
| ⏳ | 6 | m07_umap.py ×5 | cuML UMAP | fast | ~75 min | 101h |
| ⏳ | 7 | m08_plot.py ×5 + m08b | Plots + compare | fast | ~30 min | **102h** |

### run_pretrain.sh --FULL (Ch10)

| | # | Script | What | Rate | 115K Est. | Cumul. |
|:-:|:-:|--------|------|:----:|:---------:|:------:|
| ⏳ | 1 | m09_pretrain.py | Train λ=0.001, 5 ep | 25s/step (est. from 10K) | **~32h** | 32h |
| ⏳ | 2a | m05_vjepa_embed.py | Re-embed adapted | 1.55 clips/s (est. from 10K) | **~20.6h** | 53h |
| ⏳ | 2b | m06_faiss_metrics.py | Adapted metrics | fast | ~5 min | 53h |
| ⏳ | 3a | m06b_temporal_corr.py | Adapted temporal | fast | ~10 min | 53h |
| ⏳ | 3b | m05_vjepa_embed.py --shuffle | Shuffled adapted embed | 1.55 clips/s (est. from 10K) | **~20.6h** | 74h |
| ⏳ | 3c | m06_faiss_metrics.py | Shuffled adapted metrics | fast | ~5 min | 74h |
| ⏳ | 3d | m07_umap.py | Adapted UMAP | fast | ~15 min | 74h |
| ⏳ | 3e | m08_plot.py | Adapted plots | fast | ~5 min | 74h |
| ⏳ | 3f | m08b_compare.py | 7-encoder radar | fast | ~30s | **74h** |

### Grand Total (revised with measured m04 rate)

| Pipeline | Duration |
|----------|:--------:|
| run_evaluate.sh --FULL | **~102h** |
| run_pretrain.sh --FULL | **~74h** |
| **Sequential total** | **~176h (~7.3 days)** |

### Top 5 Bottlenecks (sorted by time)

| Rank | Script | Time | % of Total |
|:----:|--------|:----:|:----------:|
| 1 | m05_vjepa_embed.py (×4 runs total) | **~82h** | 47% |
| 2 | m09_pretrain.py (5 epochs) | **~32h** | 18% |
| 3 | m04_vlm_tag.py (115K tags) | **~28h** | 16% |
| 4 | m05c_true_overlap.py | **~11.5h** | 7% |
| 5 | m04d_motion_features.py | **~8h** | 5% |

**m05 V-JEPA embedding (1.55 clips/s) dominates at 47% of total time.** m04 tagging dropped from #1 to #3 after switching from vLLM (70h) to transformers (28h).

---

## Shared Setup (both chapters)

### Step 1: Spin up GPU, clone repo

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
```

### Step 2: Setup venv ✅

```bash
mkdir -p logs && \
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
```

### Step 3: Data download ✅

```bash
# 3a: Rsync POC 10K + Val 1K from Mac
rsync -avhP data/ vast_RTXpro6000_96GB:/workspace/factorjepa/data/

# 3b: Download full 115K corpus (~24 min measured, 8 parallel workers)
source venv_walkindia/bin/activate
python -u src/m00d_download_subset.py --FULL --no-wandb 2>&1 | tee logs/m00d_full.log
```

Output: `data/full_local/` (115,687 clips, 116 shards, 130 GB).

### Step 4: vLLM smoke test ✅

```bash
source venv_vllm/bin/activate
python scripts/smoke_test_vllm.py 2>&1 | tee logs/vllm_smoke.log
```

**Status (March 2026):** Passed. vLLM 0.18.1 + Qwen3-VL-8B on RTX PRO 6000 102GB.
Video: 4232 toks/s input, 31.8 toks/s output. 66 GiB KV cache free.

---

## Ch9: Frozen Encoder Evaluation (~110h)

### Ch9-1: SANITY ✅ (26/26 passed, 0 failed)

```bash
source venv_walkindia/bin/activate

# With vLLM — 26/26 PASSED (March 2026)
./scripts/run_evaluate.sh --SANITY --vllm 2>&1 | tee logs/ch9_sanity_vllm.log

# Without vLLM (validates transformers fallback)
./scripts/run_evaluate.sh --SANITY 2>&1 | tee logs/ch9_sanity.log
```

### Ch9-2: FULL 115K run

**Option A: with vLLM (recommended)**

```bash
tmux new -s ch9
cd factorjepa/
source venv_walkindia/bin/activate
./scripts/run_evaluate.sh --FULL --vllm 2>&1 | tee logs/ch9_full.log
# Ctrl+B, D to detach | tmux attach -t ch9 to reconnect
```

**Option B: without vLLM**

```bash
tmux new -s ch9
source venv_walkindia/bin/activate
./scripts/run_evaluate.sh --FULL 2>&1 | tee logs/ch9_full.log
```

Note: `--vllm` calls `venv_vllm/bin/python src/m04_vlm_tag_vllm.py` for m04 only.
All other steps (m05-m08b) use `venv_walkindia` python regardless.
If vLLM fails at runtime, `run_evaluate.sh` auto-falls back to transformers.

### Ch9-3: Push results

```bash
./git_push.sh "Ch9 full 115K frozen encoder evaluation"
```

---

## Ch10: Continual Pretraining (~74h)

> **Prerequisites:**
> - Ch9 FULL complete (Ch9-2) — `outputs/full/m06_metrics.json` needed for m08b comparison
> - `data/full_local/` + `data/val_1k_local/` exist

### Ch10-1: SANITY (~5 min)

```bash
source venv_walkindia/bin/activate
./scripts/run_pretrain.sh --SANITY 2>&1 | tee logs/ch10_sanity.log
```

### Ch10-2: POC lambda ablation (~2h on GPU)

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

### Ch10-3: FULL 115K run

> **Prerequisites:**
> - Ch9 FULL complete (Ch9-2) — `outputs/full/m06_metrics.json` needed for m08b comparison
> - Ch10-2 POC ablation complete — confirms λ=0.001 as winner

```bash
tmux new -s ch10
source venv_walkindia/bin/activate
./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log
# Ctrl+B, D to detach | tmux attach -t ch10 to reconnect
```

### Ch10-4: Push results

```bash
./git_push.sh "Ch10 full 115K continual pretraining + evaluation"
```

---

## Disk Budget

| Item | Size |
|------|:----:|
| data/full_local/ (115,687 clips) | 130 GB (measured) |
| checkpoints/vitg-384.pt | 16 GB |
| Training peak (student+teacher+optimizer) | ~40 GB |
| Embeddings (7 encoders × 115K × 1408 × 4B) | ~4.2 GB |
| outputs/full/ (metrics, plots, UMAP) | ~5 GB |
| tags.json (115K × 16 fields) | ~50 MB |
| **Total needed** | **~200 GB** |
