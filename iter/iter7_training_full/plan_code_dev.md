# Ch9 + Ch10 Full (115K): Terminal Commands & Duration Estimates

> All code is ready. Run on a fresh GPU instance with ≥250GB disk.

---

## System Design: What Calls What

### Architecture: 4 scripts, clear responsibilities

```
scripts/
├── run_frozen.sh     → Ch9:  Tags + Embeddings (5 frozen encoders + motion)
├── run_pretrain.sh   → Ch10: Training + Embeddings (2 adapted encoders)
├── run_surgery.sh    → Ch11: Surgical training + Embeddings (TBD)
└── run_eval.sh       → ALL:  Evaluation (auto-detects all available encoders)
```

No cross-chapter dependency for evaluation — run_eval.sh evaluates whatever
embeddings exist, in one shot. m08b radar includes all available encoders.

### run_frozen.sh --FULL (Ch9: tags + embeddings only)

```
run_frozen.sh --FULL
│
├── PREFLIGHT
├── m00d_download_subset.py ──→ data/full_local/ (130GB, one-time)
├── m04_vlm_tag.py ──→ tags.json                (17.6h)
├── m05_vjepa_embed.py ──→ embeddings.npy        (17.4h)
├── m05b_baselines.py ──→ embeddings_{dinov2,clip,random,shuffled}.npy
└── m04d_motion_features.py ──→ motion_features.npy
```

### run_pretrain.sh --FULL (Ch10: training + embeddings only)

```
run_pretrain.sh --FULL
│
├── PREFLIGHT
├── PHASE 1: m09 train (ablation + winner)
│   └── student_encoder.pt + training_summary.json
└── PHASE 2: m05 embeddings (both use same adapted model)
    ├── m05 adapted ──→ embeddings_vjepa_<winner>.npy          (~2.2h)
    └── m05 shuffled ──→ embeddings_vjepa_<winner>_shuffled.npy (~2.2h)
```

### run_eval.sh --FULL (ALL chapters: evaluation)

```
run_eval.sh --FULL
│
├── Auto-detect all embeddings in outputs/full/
│   (vjepa, random, dinov2, clip, vjepa_shuffled, vjepa_lambda0_001, ...)
│
└── FOR EACH encoder:
    ├── m06_faiss_metrics.py ──→ m06_metrics_{enc}.json
    ├── m06b_temporal_corr.py ──→ m06b_temporal_corr_{enc}.json
    ├── m07_umap.py ──→ umap_2d_{enc}.npy
    └── m08_plot.py ──→ plots
    └── m08b_compare.py ──→ m08b_radar.png (all available encoders)
```

---

## Combined Duration Table (115K FULL)

> **Basis:** Measured rates on 115K FULL run (April 2026, RTX PRO 6000 102GB).
> m00d: 24 min (8 workers). m04: 1.43 clips/s (transformers, BS=64).
> m05: 1.80 clips/s (measured on FULL). m09: 25s/step (est. from 10K).

### run_evaluate.sh --FULL (Ch9)

| | # | Script | What | Rate | 115K Time | Cumul. |
|:-:|:-:|--------|------|:----:|:---------:|:------:|
| ✅ | 0 | m00d_download_subset.py | Download 116 TARs (8 workers) | 12.3 s/shard | **24 min** | 0.4h |
| ✅ | 1 | m04_vlm_tag.py | Qwen3-VL tagging (transformers, BS=64) | 1.43 clips/s | **17h 36m** | 18h |
| ✅ | 2 | m05_vjepa_embed.py | V-JEPA frozen embed (12 workers) | 1.80 clips/s | **17h 24m** | 35h |
| ✅ | 3a | m05b (random) | Random embeddings | instant | **14s** | 35h |
| ✅ | 3b | m05b (dinov2) | DINOv2 embed | 7.9 clips/s | **4h 4m** | 39h |
| ✅ | 3c | m05b (clip) | CLIP embed (decord + parallel TAR, 21x speedup) | 72 clips/s | **27 min** | 39.5h |
| ✅ | 3d | m05b (shuffled) | Shuffled V-JEPA embed (GPU-bound) | 1.02 clips/s | **18h 56m** | 58.5h |
| ✅ | 4.5 | m04d_motion_features.py | RAFT optical flow | 3.5→0.9 clips/s (producer starvation at 90%+) | **10h 55m** | **69.4h** |

### run_pretrain.sh --FULL (Ch10: training + embeddings only)

> **Measured (April 3-4 2026, RTX PRO 6000 102GB, BS=112):**
> Ablation: 4 lambdas × 281 steps × 6.2s/step = 2h 44m total (BS=32, fix applied for BS=112 next run).
> Winner: λ=0.001 (best_val_loss=1.6263). Winner training: 1,023 steps, ~6.2s/step.

| | # | Phase | Script | What | Rate | 115K Est. | Cumul. |
|:-:|:-:|:-----:|--------|------|:----:|:---------:|:------:|
| ✅ | 0 | 1 | m09 ablation (4λ on 10K) | Lambda sweep → winner | 6.2s/step, BS=32 | **2h 44m** (measured) | 2.7h |
| ✅ | 1 | 1 | m09_pretrain.py | Train λ=0.001, **1 ep**, 115K | ~21s/step, 1023 steps, BS=112 | **~6h** (measured, incl OOM recovery) | 8.7h |
| ✅ | 2a | 2 | m05_vjepa_embed.py | Re-embed adapted | 14.7 clips/s (measured, torch.compile + sdp_kernel patch) | **~2.2h** | 11h |
| ⏳ | 2b | 2 | m05_vjepa_embed.py --shuffle | Shuffled adapted embed | ~14.7 clips/s | **~2.2h** | **13.2h** |

### run_eval.sh --FULL (evaluation for ALL available encoders)

| | # | Script | What | Rate | Est. (7 encoders) | Cumul. |
|:-:|:-:|--------|------|:----:|:---------:|:------:|
| ⏳ | 1 | m06 ×7 | FAISS kNN metrics | vectorized (was 113m/enc pre-fix) | **~35-105 min** | 1.5h |
| ⏳ | 2 | m06b ×7 | Temporal correlation | fast | ~70 min | 2.5h |
| ⏳ | 3 | m07 ×7 | cuML UMAP | fast | ~100 min | 4h |
| ⏳ | 4 | m08 ×7 | Plots | fast | ~35 min | 4.5h |
| ⏳ | 5 | m08b | Radar comparison (all encoders) | fast | ~30s | **4.5h** |

### Grand Total (measured + estimated)

| Pipeline | Duration |
|----------|:--------:|
| run_frozen.sh --FULL (Ch9: tags + embeddings) | **~69h** |
| run_pretrain.sh --FULL (Ch10: train + embeddings) | **~13h** (was ~40h before num_frames fix + sdp_kernel patch) |
| run_eval.sh --FULL (ALL: eval 7 encoders) | **~4.5h** |
| **Sequential total** | **~87h (~3.6 days)** |
| **Parallel (2×GPU shared disk)** | **~69h (~2.9 days)** (Ch9 ∥ Ch10, then eval) |

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

### Step 2: Setup venv 

```bash
mkdir -p logs && \
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
```

### Step 3: Data download 

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

## SANITY: Both Chapters Together (~10 min)

> **IMPORTANT:** Run Ch9 + Ch10 SANITY sequentially in one shot.
> Ch10's m06 metrics step requires tags from Ch9's m04 — if run separately,
> stale/mismatched outputs cause `FATAL: N embeddings vs M tags` errors.
> Running both back-to-back from a clean `outputs/sanity/` guarantees
> aligned clip counts across the entire pipeline (m04 tags → m05 embeds →
> m06 metrics → m09 train → m05 re-embed → m06 adapted metrics).

```bash
source venv_walkindia/bin/activate
rm -rf outputs/sanity/
./scripts/run_evaluate.sh --SANITY 2>&1 | tee logs/ch9_sanity.log && \
./scripts/run_pretrain.sh --SANITY 2>&1 | tee logs/ch10_sanity.log
```

---

## Ch9: Frozen Encoder Evaluation (~77h)

### Ch9-1: SANITY ✅ (included in combined SANITY above)

### Ch9-2: FULL 115K run (tags + embeddings only)

```bash
tmux new -s ch9
source venv_walkindia/bin/activate
./scripts/run_frozen.sh --FULL 2>&1 | tee logs/ch9_full.log
# Ctrl+B, D to detach | tmux attach -t ch9 to reconnect
```

---

## Ch10: Continual Pretraining (training + embeddings only)

> **Prerequisites:**
> - `data/full_local/` + `data/val_1k_local/` exist
> - No Ch9 dependency for training or embedding (only run_eval.sh needs tags)

### Ch10-1: SANITY ✅ (included in combined SANITY above)

### Ch10-2 + Ch10-3: Ablation + FULL training (automated)

`run_pretrain.sh --FULL` auto-runs ablation if `ablation_winner.json` is missing:
1. Trains 4 lambdas [0, 0.001, 0.01, 0.1] on `data/subset_10k_local` (10K clips, 1 epoch each, ~2h)
2. Selects winner by lowest `best_val_loss` → saves `ablation_winner.json`
3. Continues with winner lambda on full 115K data (1 epoch, ~25h)
4. Creates adapted + shuffled adapted embeddings (~4.4h)

```bash
tmux new -s ch10
source venv_walkindia/bin/activate
./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log
# Ctrl+B, D to detach | tmux attach -t ch10 to reconnect
```

---

## Evaluation: All Encoders (~4.5h)

> **Prerequisites:** Ch9 (run_frozen.sh) + Ch10 (run_pretrain.sh) complete.
> run_eval.sh auto-detects all available embeddings — no encoder list needed.

```bash
tmux new -s eval
source venv_walkindia/bin/activate
./scripts/run_eval.sh --FULL 2>&1 | tee logs/eval_full.log
# Evaluates all 7 encoders: m06→m06b→m07→m08→m08b
```

### Push all results

```bash
./git_push.sh "Full 115K evaluation: 7 encoders, metrics + plots + radar"
```

---

## 2×GPU Shared-Disk Strategy (Recommended)

> **Why shared disk over 2 separate 1×GPU instances:**
> - Separate disks require git_push/pull + HF upload/download to sync code and outputs — the #1 source of bugs (stale manifests, wrong output dirs, code divergence).
> - Shared disk: `outputs/full/` is ONE directory. Ch9 writes tags/motion, Ch10 reads them directly. Zero sync overhead.
> - Parallel on 2×GPU finishes in ~68h vs ~108h sequential. At $1.20/hr per GPU: parallel costs $1.20×2×68h=$163 vs sequential $1.20×108h=$130. **$33 more but 40h faster.**
> - Code fixes apply to both pipelines instantly — no "pull while running" risk.

### Architecture (shared disk)

```
┌──────────────────────────────────────────────┐
│            2×GPU Instance (shared disk)       │
│                                              │
│  GPU 0 (Ch9):  run_frozen.sh --FULL          │
│  GPU 1 (Ch10): run_pretrain.sh --FULL        │
│  After both:   run_eval.sh --FULL (1 GPU)    │
│                                              │
│  Shared: outputs/full/, data/full_local/     │
│          configs/, src/ (one codebase)        │
└──────────────────────────────────────────────┘
```

### Cross-Chapter Dependencies (simplified)

With the 3-script architecture, dependencies are clean:

```
run_frozen.sh:   m04(tags) → m05/m05b(embeds) → m04d(motion)   [NO eval dependency]
run_pretrain.sh: m09(train) → m05(adapted embeds)               [NO eval dependency]
run_eval.sh:     reads ALL embeddings + tags + motion → eval     [runs AFTER both]
```

Ch9 and Ch10 are fully independent — they only produce embeddings.
run_eval.sh needs Ch9's tags.json + motion_features.npy for m06/m06b metrics.
No cross-chapter dependency during GPU-heavy embedding work.

### Dependency Map

```
GPU 0 (Ch9):  m04 → m05 → m05b → m04d
              ↓ writes tags.json, embeddings, motion_features.npy

GPU 1 (Ch10): m09(ablation) → m09(115K train) → m05(adapted) → m05(shuffled)
              ↓ writes adapted embeddings

After both:   run_eval.sh → m06 ×7 → m06b ×7 → m07 ×7 → m08 ×7 → m08b
              ↑ reads ALL embeddings from same outputs/full/ directory
```

### Setup + Run (2×GPU shared-disk instance)

```bash
# 1. Clone + setup (~30 min)
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log

# 2. Copy .env (credentials — not in git)
scp .env <from-old-instance>:/workspace/factorjepa/

# 3. Download data
source venv_walkindia/bin/activate
python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/download_poc_val.log
python -u src/m00d_download_subset.py --FULL --no-wandb 2>&1 | tee logs/m00d_full.log

# 4. Verify both GPUs visible
nvidia-smi -L
# GPU 0: NVIDIA RTX PRO 6000 ...
# GPU 1: NVIDIA RTX PRO 6000 ...

# 5. Run Ch9 + Ch10 in parallel (CUDA_VISIBLE_DEVICES pins each to its own GPU)
tmux new -s ch9
CUDA_VISIBLE_DEVICES=0 ./scripts/run_frozen.sh --FULL 2>&1 | tee logs/ch9_full.log
# Ctrl+B, D to detach

tmux new -s ch10
CUDA_VISIBLE_DEVICES=1 ./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log
# Ctrl+B, D to detach

# Monitor: tmux attach -t ch9 / tmux attach -t ch10

# 6. After BOTH complete: run evaluation on all 7 encoders
./scripts/run_eval.sh --FULL 2>&1 | tee logs/eval_full.log
```

### SANITY (run before FULL to validate code paths)

```bash
source venv_walkindia/bin/activate
rm -rf outputs/sanity/
CUDA_VISIBLE_DEVICES=0 ./scripts/run_evaluate.sh --SANITY 2>&1 | tee logs/ch9_sanity.log && \
CUDA_VISIBLE_DEVICES=0 ./scripts/run_pretrain.sh --SANITY 2>&1 | tee logs/ch10_sanity.log
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
| ~~m05c_true_overlap.py~~ | ~~overlap_augA/B.npy + keys~~ | ~~1.2 GB~~ (SKIPPED — dim-split in m06 sufficient) |
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
