# Ch9 + Ch10 Full (115K): Terminal Commands & Duration Estimates

> All code is ready. Run on a fresh GPU instance with ≥250GB disk.

---

## System Design: What Calls What

### run_evaluate.sh --FULL (Ch9: frozen encoder evaluation)

```
run_evaluate.sh --FULL
│
├── PREFLIGHT: output_guard.py preflight_evaluate
│
├── m00d_download_subset.py ──→ data/full_local/ (130GB, one-time)
│
├── m04_vlm_tag.py ──→ tags.json                (17.6h measured)
│
├── m05_vjepa_embed.py ──→ embeddings.npy        ⚠️ BIGGEST BOTTLENECK (17.4h)
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
├── PHASE 1: Train with winner λ (from Ch10-2 POC ablation)
│   └── m09_pretrain.py --lambda-reg <winner> --max-epochs 1
│       └── student_encoder.pt + training_summary.json
│
├── PHASE 2: Re-embed + Metrics
│   ├── m05_vjepa_embed.py --model adapted --encoder vjepa_<winner>
│   │   └── embeddings_vjepa_<winner>.npy             ⚠️ BOTTLENECK
│   └── m06_faiss_metrics.py --encoder vjepa_<winner>
│       └── m06_metrics_vjepa_<winner>.json
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

> **Basis:** Measured rates on 115K FULL run (April 2026, RTX PRO 6000 102GB).
> m00d: 24 min (8 workers). m04: 1.43 clips/s (transformers, BS=64).
> m05: 1.80 clips/s (measured on FULL). m09: 25s/step (est. from 10K).

### run_evaluate.sh --FULL (Ch9)

| | # | Script | What | Rate | 115K Est. | Cumul. |
|:-:|:-:|--------|------|:----:|:---------:|:------:|
| ✅ | 0 | m00d_download_subset.py | Download 116 TARs (8 workers) | 12.3 s/shard | **24 min** | 0.4h |
| ✅ | 1 | m04_vlm_tag.py | Qwen3-VL tagging (transformers, BS=64) | 1.43 clips/s | **17h 36m** | 18h |
| ✅ | 2 | m05_vjepa_embed.py | V-JEPA frozen embed (12 workers) | 1.80 clips/s | **17h 24m** | 35h |
| ✅ | 3a | m05b (random) | Random embeddings | instant | **14s** | 35h |
| ✅ | 3b | m05b (dinov2) | DINOv2 embed | 7.9 clips/s | **4h 4m** | 39h |
| ✅ | 3c | m05b (clip) | CLIP embed (decord + parallel TAR, 21x speedup) | 72 clips/s | **~27 min** (was 4h) | 39.5h |
| 🔄 | 3d | m05b (shuffled) | Shuffled V-JEPA embed (GPU-bound) | 1.8 clips/s | **~17.8h** | 57h |
| ⏳ | 4 | m05c_true_overlap.py | Augmented A+B embeds | ~1.8 clips/s × 2 | **~10h** | 67h |
| ⏳ | 4.5 | m04d_motion_features.py | RAFT optical flow | ~4 clips/s (est. from 10K) | **~8h** | 75h |
| ⏳ | 5 | m06_faiss_metrics.py ×5 | FAISS kNN metrics | fast | ~25 min | 75h |
| ⏳ | 5.5 | m06b_temporal_corr.py ×5 | Temporal correlation | fast | ~50 min | 76h |
| ⏳ | 6 | m07_umap.py ×5 | cuML UMAP | fast | ~75 min | 77h |
| ⏳ | 7 | m08_plot.py ×5 + m08b | Plots + compare | fast | ~30 min | **77h** |

### run_pretrain.sh --FULL (Ch10)

| | # | Script | What | Rate | 115K Est. | Cumul. |
|:-:|:-:|--------|------|:----:|:---------:|:------:|
| ⏳ | 1 | m09_pretrain.py | Train λ=&lt;winner&gt;, **1 ep** | 25s/step (est. from 10K), 3615 steps | **~25h** | 25h |
| ⏳ | 2a | m05_vjepa_embed.py | Re-embed adapted | ~1.80 clips/s | **~17.8h** | 43h |
| ⏳ | 2b | m06_faiss_metrics.py | Adapted metrics | fast | ~5 min | 43h |
| ⏳ | 3a | m06b_temporal_corr.py | Adapted temporal | fast | ~10 min | 43h |
| ⏳ | 3b | m05_vjepa_embed.py --shuffle | Shuffled adapted embed | ~1.80 clips/s | **~17.8h** | 61h |
| ⏳ | 3c | m06_faiss_metrics.py | Shuffled adapted metrics | fast | ~5 min | 61h |
| ⏳ | 3d | m07_umap.py | Adapted UMAP | fast | ~15 min | 61h |
| ⏳ | 3e | m08_plot.py | Adapted plots | fast | ~5 min | 61h |
| ⏳ | 3f | m08b_compare.py | 7-encoder radar (CPU, needs Ch9 metrics) | fast | ~30s | **61h** |

### Grand Total (all measured except Ch10)

| Pipeline | Duration |
|----------|:--------:|
| run_evaluate.sh --FULL (Ch9) | **~77h** |
| run_pretrain.sh --FULL (Ch10, 1 epoch) | **~61h** |
| **Sequential total** | **~138h (~5.8 days)** |

### Code Hardening (April 3 2026)

| Fix | Severity | What |
|-----|:--------:|------|
| Duplicate `model:` key in YAML | **CRITICAL** | `yaml.safe_load()` silently overwrites first block. All arch params (embed_dim, depth, etc.) lost. Merged into single block. |
| 19 `.get(key, default)` removed | HIGH | Every `.get()` on YAML config hid missing keys behind hardcoded defaults — violates no-hardcoded-values rule. Changed to direct `[]` access. |
| Duplicate `sanity_val_clips` | MEDIUM | `data.sanity_val_clips=100` vs `validation.sanity_val_clips=50`. Unified to `data.sanity_val_clips` only. |
| Hardcoded lambdas in `select_ablation_winner()` | MEDIUM | Default `["0", "0.001", "0.01", "0.1"]` → reads from `drift_control.ablation_lambdas` in YAML. |
| Missing manifest key `"n"` | MEDIUM | `manifest.get("n_clips")` missed FULL mode key `"n"`. Now checks all 3 variants. |
| Added `final_weight_decay: 0.04` | LOW | Eliminates `.get("final_weight_decay", initial_wd)` — explicit in YAML, defaults to fixed WD. |
| CLAUDE.md rule 15.1 | — | New rule: no `.get(key, default)` on YAML config. `.get()` allowed only on runtime data. |

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
source venv_walkindia/bin/activate

# 3a: Download POC 10K + Val 1K from HF (~3 min, measured)
python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/download_poc_val.log

# 3b: Download full 115K corpus (~24 min, 8 parallel workers)
python -u src/m00d_download_subset.py --FULL --no-wandb 2>&1 | tee logs/m00d_full.log
```

Output: `data/full_local/` (115,687 clips, 116 shards, 130 GB).

---

> **Note on vLLM:** vLLM 0.18.1 was tested and integrated (see `iter/utils/vLLM_plan_Blackwell.md`
> for 14 root causes found + fixed). However, transformers is **2.5x faster** for offline batch
> tagging (1.43 clips/s vs 0.45 clips/s) due to vLLM's double-preprocessing overhead.
> Use transformers (`run_evaluate.sh --FULL` without `--vllm`) for production runs.

## Ch9: Frozen Encoder Evaluation (~77h)

### Ch9-1: SANITY ✅ (26/26 passed, 0 failed)

```bash
source venv_walkindia/bin/activate
./scripts/run_evaluate.sh --SANITY 2>&1 | tee logs/ch9_sanity.log
```

### Ch9-2: FULL 115K run

```bash
tmux new -s ch9
source venv_walkindia/bin/activate
./scripts/run_evaluate.sh --FULL 2>&1 | tee logs/ch9_full.log
# Ctrl+B, D to detach | tmux attach -t ch9 to reconnect
```

### Ch9-3: Push results

```bash
./git_push.sh "Ch9 full 115K frozen encoder evaluation"
```

---

## Ch10: Continual Pretraining (~61h)

> **Prerequisites:**
> - `data/full_local/` + `data/val_1k_local/` exist
> - Ch9's `m06_metrics.json` needed only for m08b at the end (NOT for training)

### Ch10-1: SANITY (~5 min)

```bash
source venv_walkindia/bin/activate
./scripts/run_pretrain.sh --SANITY 2>&1 | tee logs/ch10_sanity.log
```

### Ch10-2 + Ch10-3: Ablation + FULL training (automated)

`run_pretrain.sh --FULL` auto-runs ablation if `ablation_winner.json` is missing:
1. Trains 4 lambdas [0, 0.001, 0.01, 0.1] on `data/subset_10k_local` (10K clips, 1 epoch each, ~2h)
2. Selects winner by lowest `best_val_loss` → saves `ablation_winner.json`
3. Continues with winner lambda on full 115K data (1 epoch, ~25h)

> **Prerequisites:**
> - `data/full_local/` + `data/subset_10k_local/` + `data/val_1k.json` exist
> - Ch9's `m06_metrics.json` needed only for m08b at the end

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

## 2-GPU Parallel Strategy

Ch10 training (m09) does NOT depend on Ch9 outputs. Only the final comparison (m08b) does.
Spin up a 2nd GPU instance to overlap Ch10 work while Ch9 is still running.

### Dependency Map

```
GPU 1:  Ch9 [m04 → m05 → m05b → m05c → m04d → m06 → m06b → m07 → m08] → upload to HF
GPU 2:  Ch10 [SANITY → POC ablation → ablation_winner.json → m09 FULL → m05 → m06] → upload to HF

CPU (Mac or any machine):
  download both Ch9 + Ch10 outputs from HF
  └── m08b compare (CPU-only, needs m06_metrics from BOTH chapters)
```

### GPU 2 Setup + Run Order

```bash
# 1. Clone + setup (~15 min)
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log

# 2. Copy .env (credentials — not in git)
# FROM GPU 1 or local Mac:
scp .env gpu2:/workspace/factorjepa/

# 3. Download POC + val data from HF (~3 min, measured)
source venv_walkindia/bin/activate
python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/download_poc_val.log

# 4. Download full 115K corpus from HF (~24 min, 8 parallel workers)
python -u src/m00d_download_subset.py --FULL --no-wandb 2>&1 | tee logs/m00d_full.log

# 5. Ch10-1: SANITY (~5 min)
./scripts/run_pretrain.sh --SANITY 2>&1 | tee logs/ch10_sanity.log

# 6. Ch10-2+3: FULL (auto-ablation on 10K + train winner on 115K, ~61h total)
tmux new -s ch10
./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log
# Ablation (~2h) + winner training (~25h) + re-embed (~18h) + metrics
# bg_upload pushes outputs to HF after each step
```

### CPU: Final Comparison (Mac or any machine, no GPU needed)

After BOTH GPU 1 (Ch9) and GPU 2 (Ch10) finish and upload to HF:

```bash
# Download all outputs from HF
python -u src/utils/hf_outputs.py download outputs/full 2>&1 | tee logs/hf_download_full.log

# Run m08b comparison with all 7 encoders (CPU-only, ~30s)
source venv_walkindia/bin/activate
python -u src/m08b_compare.py --FULL 2>&1 | tee logs/m08b_full.log
```

---

## Disk Budget

### Input Data

| Item | Size |
|------|:----:|
| data/full_local/ (115,687 clips, 116 TARs) | 130 GB (measured) |
| data/subset_10k_local/ (10K clips) | 10.5 GB |
| data/val_1k_local/ (1K clips) | 0.9 GB |
| checkpoints/vjepa2_vitg384.pt (V-JEPA 2 pretrained) | 16 GB |
| HF model cache (Qwen3-VL-8B + V-JEPA + DINOv2 + CLIP) | ~40 GB |

### Ch9 outputs (run_evaluate.sh)

| Script | Output files | Size (115K) |
|--------|-------------|:----------:|
| m04_vlm_tag.py | tags.json (115K × 42 fields) | ~150 MB |
| m05_vjepa_embed.py | embeddings.npy + paths.npy | ~620 MB |
| m05b (random) | embeddings_random.npy + paths | ~620 MB |
| m05b (dinov2) | embeddings_dinov2.npy + paths (1536d) | ~680 MB |
| m05b (clip) | embeddings_clip.npy + paths (768d) | ~340 MB |
| m05b (shuffled) | embeddings_vjepa_shuffled.npy + paths | ~620 MB |
| m05c_true_overlap.py | overlap_augA/B.npy + keys | ~1.2 GB |
| m04d_motion_features.py | motion_features.npy + paths (13d) | ~6 MB |
| m06_faiss_metrics.py ×5 | m06_metrics_*.json + knn_indices_*.npy | ~2.7 GB |
| m06b_temporal_corr.py ×5 | m06b_temporal_corr_*.json | <1 MB |
| m07_umap.py ×5 | umap_2d_*.npy | ~5 MB |
| m08_plot.py ×5 + m08b | .png + .pdf plots | ~50 MB |
| **Ch9 outputs subtotal** | | **~7 GB** |

### Ch10 outputs (run_pretrain.sh)

| Script | Output files | Size (115K) |
|--------|-------------|:----------:|
| m09_pretrain.py | student_encoder.pt + loss_log.csv | ~3.8 GB |
| m09 training peak (student+teacher+optimizer+scaler) | temp, cleaned after | ~40 GB peak |
| m05 re-embed (adapted) | embeddings_vjepa_lambda*.npy | ~620 MB |
| m05 re-embed (shuffled adapted) | embeddings_*_shuffled.npy | ~620 MB |
| m06/m06b/m07/m08/m08b (adapted) | metrics + plots | ~1.5 GB |
| **Ch10 outputs subtotal** | | **~6.5 GB** |

### Total Disk

| Category | Size |
|----------|:----:|
| Input data + models | ~197 GB |
| Ch9 outputs | ~7 GB |
| Ch10 outputs | ~6.5 GB |
| Ch10 training peak (temporary) | ~40 GB |
| **Total needed** | **~250 GB** |
| **Available on current instance** | **299 GB free** |
