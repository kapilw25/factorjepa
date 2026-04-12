# FactorJEPA Runbook — Week 1 Experiments

> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**
> Ch11 surgery is the PRIMARY path. ExPLoRA is the baseline to beat. Ch10 brute-force deferred.
>
> **Key files for new Claude sessions:**
> - `iter/iter8/next_steps.md` — 3 action items for Week 1
> - `iter/iter8/plan_code_development.md` — implementation plan + completed work
> - `iter/iter8/plan_training.md` — full research plan (training recipe, audit fixes, paper strategy)
> - `src/CLAUDE.md` — codebase rules (31 rules, hook-enforced)
> - `src/MEMORY.md` — project state, pipeline modules, encoder registry

---

## Scripts

```
scripts/
├── lib/common.sh           # Shared: log, run_step, verify, watchdog, bg_upload
├── prep_data.sh            # Ch9: m04(tags) + m04d(motion)
├── train_pretrain.sh       # Ch10: m09(continual pretraining) — DEFERRED
├── train_explora.sh        # Step 1b: ExPLoRA (LoRA + unfreeze) — NEW
├── train_surgery.sh        # Step 2: Ch11 factor surgery — NEW (m10/m11 NOT BUILT)
├── run_embed.sh            # ALL: m05/m05b embedding (auto-detects encoders)
└── run_eval.sh             # ALL: m06→m08b evaluation (auto-detects encoders)
```

---

## Setup (one-time, on GPU instance)

```bash
# 1. Clone + setup
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
source venv_walkindia/bin/activate

# 2. Download V-JEPA 2.1 (2B) checkpoint (~8 GB)
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt -P checkpoints/

# 3. Download pre-filtered data (POC 10K + val 1K, ~3 min)
python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/download_poc_val.log

# 4. (Optional) Download full 115K corpus (~24 min)
python -u src/m00d_download_subset.py --FULL --no-wandb 2>&1 | tee logs/m00d_full.log
```

---

## Sanity Check (~10 min)

```bash
rm -rf outputs/sanity/
./scripts/prep_data.sh --SANITY && \
./scripts/train_explora.sh --SANITY && \
./scripts/run_embed.sh --SANITY && \
./scripts/run_eval.sh --SANITY
```

---

## Week 1 Experiment Sequence

### Step 1a: Temporal Interference Projection [~2.5h GPU + 30 min CPU]

Requires frozen 2.1 embeddings (from Step 0) + shuffled 2.1 embeddings.

```bash
# 1. Generate shuffled V-JEPA 2.1 embeddings (~2h GPU, one-time)
python -u src/m05b_baselines.py --encoder vjepa_2_1_frozen_shuffled \
    --model-config configs/model/vjepa2_1.yaml \
    --FULL --subset data/subset_10k.json --local-data data/subset_10k_local --no-wandb \
    2>&1 | tee logs/m05b_vjepa_2_1_shuffled.log

# 2. Temporal projection: PCA on (frozen - shuffled), sweep k, re-run Prec@K (30 min CPU)
python -u src/m06c_temporal_projection.py --FULL --subset data/subset_10k.json \
    --normal-encoder vjepa_2_1_frozen --shuffled-encoder vjepa_2_1_frozen_shuffled \
    2>&1 | tee logs/m06c.log

# Output: outputs/poc/m06c_projection_results.json
# Compare: vjepa_2_1_frozen Prec@K vs projected Prec@K for each k
# If Prec@K jumps → paper novelty (temporal interference is removable linear subspace)
```

### Step 1b: ExPLoRA Baseline [~7h GPU total]

Simplest V-JEPA 2.1 domain adaptation. Sets the bar that Ch11 surgery must beat.
`train_explora.sh` handles ALL sub-steps automatically:

```bash
# Single command — handles frozen baseline + ExPLoRA train + re-embed + eval
./scripts/train_explora.sh --POC 2>&1 | tee logs/explora_poc.log

# What it does internally:
#   Step 0: frozen 2.1 embed + eval (~2h, skips if cached)
#   Step 1: ExPLoRA training (~3h)
#   Step 2: re-embed adapted model (~2h)
#   Step 3: evaluate adapted Prec@K

# Compare:
#   outputs/poc/m06_metrics_vjepa_2_1_frozen.json    ← frozen baseline
#   outputs/poc/m06_metrics_vjepa_2_1_explora.json   ← ExPLoRA adapted
#   If ExPLoRA improves over frozen 2.1 → publishable result
```

**Embedding files generated:**
```
outputs/poc/
├── embeddings.npy                              (10000, 1408) ← V-JEPA 2.0 frozen (existing, KEEP)
├── embeddings_vjepa_2_1_frozen.npy             (10000, 1664) ← V-JEPA 2.1 frozen
├── embeddings_vjepa_2_1_frozen_shuffled.npy    (10000, 1664) ← V-JEPA 2.1 shuffled (from Step 1a)
├── embeddings_vjepa_2_1_explora.npy            (10000, 1664) ← V-JEPA 2.1 ExPLoRA adapted
```

### Step 2: Ch11 Factor Surgery POC [~3h GPU + SAM3 prep]

THE experiment. Factor-decomposed inputs + progressive unfreezing on frozen V-JEPA 2.1.
POC uses `--poc-simple`: 100 clips, 2 factors (D_L + D_A), skip D_I.

```bash
# SAM3 segmentation + factor datasets + 2-stage surgery + re-embed + eval
./scripts/train_surgery.sh --POC 2>&1 | tee logs/surgery_poc.log

# What it does:
#   Step 0: m10 SAM3 segmentation on 100 clips → masks → tracklets (~30 min GPU)
#   Step 1: m11 factor datasets D_L + D_A (~5 min CPU)
#   Step 2: m09 surgery training, 2-stage progressive unfreezing (~2.5h GPU)
#   Step 3: m05 re-embed surgical model (~2h GPU)
#   Step 4: m06 evaluate

# THE KEY COMPARISON:
#   outputs/poc/m06_metrics_vjepa_2_1_frozen.json     ← frozen baseline
#   outputs/poc/m06_metrics_vjepa_2_1_explora.json    ← ExPLoRA (from Step 1b)
#   outputs/poc/m06_metrics_vjepa_2_1_surgical.json   ← surgery adapted
#   Surgery must beat ExPLoRA to justify complexity
```

---

## Decision Gate (end of Week 1)

| Step 1a projection | Step 1b ExPLoRA | Step 2 Surgery | Action |
|---|---|---|---|
| Prec@K jumps | ExPLoRA improves | Surgery > ExPLoRA | **Strongest: all 3 + surgery wins** |
| Any | ExPLoRA improves | Surgery = ExPLoRA | **Publish ExPLoRA, surgery adds no value** |
| Any | No change | Surgery improves | **Best novelty: standard fails, surgery succeeds** |
| Any | ExPLoRA improves | Surgery < ExPLoRA | **Publish ExPLoRA, drop surgery** |
| Any | No change | No change | **Debug: reverse factor order, more clips, LoRA fallback** |

---

## Config Architecture (new, this session)

```
configs/
├── pipeline.yaml                    # Shared: clip limits, encoders, streaming, eval
├── model/
│   ├── vjepa2_0.yaml                # Legacy (1B, 1408-dim)
│   └── vjepa2_1.yaml                # PRIMARY (2B, 1664-dim)
└── train/
    ├── base_optimization.yaml       # Shared: masking, augmentation, AdamW, EMA
    ├── ch10_pretrain.yaml           # Drift control + lambda sweep
    ├── explora.yaml                 # LoRA + unfreeze 1-2 blocks
    └── ch11_surgery.yaml            # 3-stage progressive unfreezing + factor datasets
```

Training scripts merge: `pipeline.yaml` + `model/*.yaml` + `train/*.yaml` via `load_merged_config()`.

---

## Key Configs

| Config | Value | Source |
|---|---|---|
| Model | V-JEPA 2.1 ViT-G (2B, 1664-dim) | `configs/model/vjepa2_1.yaml` |
| Checkpoint | `checkpoints/vjepa2_1_vitG_384.pt` | Download from Meta |
| Training frames | 16 | `configs/train/base_optimization.yaml` |
| Eval frames | 64 | `configs/pipeline.yaml: gpu.eval_frames_per_clip` |
| ExPLoRA LoRA rank | 16 | `configs/train/explora.yaml` |
| ExPLoRA unfreeze | 2 blocks | `configs/train/explora.yaml` |
| Surgery stages | 3 (layout → agent → interaction) | `configs/train/ch11_surgery.yaml` |
| Grad clip | 10.0 (post-audit, was 1.0) | `configs/train/base_optimization.yaml` |
| LR schedule | Constant (post-audit, was cosine) | `configs/train/base_optimization.yaml` |

---

## Previous Results (Ch10, April 5, 2026)

| Metric | Frozen | Ch10 Adapted (λ=0.001) | Delta |
|---|---|---|---|
| Prec@K | 36.1 +/-0.6 | 14.3 +/-0.3 | **-21.8pp (FAILED)** |
| nDCG@K | 0.950 +/-0.001 | 0.906 +/-0.001 | **-0.045 (FAILED)** |

Diagnosis: λ=0.001 drift penalty 1000x smaller than JEPA loss → catastrophic forgetting.
Gold standard audit found 12 discrepancies → all fixed in new configs.
Full details: `iter/utils/experiment_log.md`

---

## Acceptance Bar (NeurIPS-grade)

- 1% Prec@K from a single run is noise (arXiv:2511.19794)
- Need: non-overlapping 95% bootstrap CIs on nDCG@K + majority (5/8) metrics improved
- For paper: 3-5 training seeds, propagated CI on delta
- POC shortcut: bootstrap CIs on 10K subset (r=0.84 with full)
