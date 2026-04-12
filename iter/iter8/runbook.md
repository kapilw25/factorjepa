# FactorJEPA Runbook — Week 1 Experiments

> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**
> **Short-term: Show Surgery > ExPLoRA > Frozen on 1K val clips (~70 min GPU).**
> Ch11 surgery is the PRIMARY path. ExPLoRA is the baseline to beat.
>
> **Key files for new Claude sessions:**
> - `iter/iter8/next_steps.md` — action items for Week 1
> - `iter/iter8/plan_code_development.md` — implementation plan + completed work
> - `iter/iter8/plan_training.md` — full research plan (system design diagrams, paper strategy)
> - `src/CLAUDE.md` — codebase rules (32 rules, hook-enforced)
> - `src/MEMORY.md` — project state, pipeline modules, encoder registry

---

## GPU Instance Setup (one-time)

```bash
# 1. Pull latest code
git pull origin main

# 2. Setup environment
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
source venv_walkindia/bin/activate

# 3. Download V-JEPA 2.1 (2B) checkpoint (~8 GB)
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt -P checkpoints/

# 4. Install SAM 3.1 (gated — need HF access approval first)
huggingface-cli login
pip install git+https://github.com/facebookresearch/sam3.git

# 5. Install PEFT for ExPLoRA
pip install peft>=0.13.0

# 6. Download data (POC 1K val + 10K subset, ~3 min)
python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/download_poc_val.log
```

---

## Sanity Check (~5 min)

```bash
rm -rf outputs/sanity/
./scripts/train_explora.sh --SANITY 2>&1 | tee logs/sanity_explora.log
```

---

## Week 1: Fast Signal on 1K Val Clips (~70 min total)

### Step 1b: ExPLoRA Baseline [~1h GPU]

Single command — handles frozen 2.1 baseline + ExPLoRA training + re-embed + eval:

```bash
./scripts/train_explora.sh --POC 2>&1 | tee logs/explora_poc.log

# What it does:
#   Step 0: frozen V-JEPA 2.1 embed on 1K clips + eval (~12 min, skips if cached)
#   Step 1: ExPLoRA training — LoRA rank=16, unfreeze blocks 0-1 (~20 min)
#   Step 2: re-embed adapted model (~12 min)
#   Step 3: evaluate Prec@K

# Compare:
#   outputs/poc/m06_metrics_vjepa_2_1_frozen.json    ← frozen baseline
#   outputs/poc/m06_metrics_vjepa_2_1_explora.json   ← ExPLoRA adapted
```

### Step 2: Ch11 Factor Surgery [~30 min GPU]

THE experiment — SAM 3.1 text-prompted segmentation + 3-factor progressive unfreezing:

```bash
./scripts/train_surgery.sh --POC 2>&1 | tee logs/surgery_poc.log

# What it does:
#   Step 0: m10 SAM 3.1 on 1K clips — per-clip text prompt from tags.json notable_objects
#           → agent masks + layout masks + interaction mining (~5 min GPU)
#   Step 1: m11 factor datasets D_L + D_A + D_I (~3 min CPU)
#   Step 2: m09 surgery — 3-stage progressive unfreezing (~15 min GPU)
#           Stage 1: layers 0-12, 100% D_L (layout)
#           Stage 2: layers 0-24, 90% D_A + 10% D_L replay
#           Stage 3: layers 0-36, 85% D_I + 10% D_A + 5% D_L replay
#   Step 3: m05 re-embed surgical model (~12 min GPU)
#   Step 4: m06 evaluate Prec@K

# THE KEY COMPARISON:
#   outputs/poc/m06_metrics_vjepa_2_1_frozen.json     ← frozen baseline
#   outputs/poc/m06_metrics_vjepa_2_1_explora.json    ← ExPLoRA (from Step 1b)
#   outputs/poc/m06_metrics_vjepa_2_1_surgical.json   ← surgery adapted
```

### Step 1a (optional): Temporal Interference Projection [30 min CPU]

Run after Step 1b (needs frozen 2.1 + shuffled embeddings):

```bash
# 1. Generate shuffled V-JEPA 2.1 embeddings (~12 min GPU on 1K clips)
python -u src/m05b_baselines.py --encoder vjepa_2_1_frozen_shuffled \
    --model-config configs/model/vjepa2_1.yaml \
    --POC --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05b_vjepa_2_1_shuffled.log

# 2. Temporal projection (30 min CPU)
python -u src/m06c_temporal_projection.py --POC \
    --normal-encoder vjepa_2_1_frozen --shuffled-encoder vjepa_2_1_frozen_shuffled \
    2>&1 | tee logs/m06c.log
```

---

## Embedding Files Generated

```
outputs/poc/
├── embeddings.npy                              (10000, 1408) ← V-JEPA 2.0 frozen (existing)
├── embeddings_vjepa_2_1_frozen.npy             (1000, 1664)  ← V-JEPA 2.1 frozen
├── embeddings_vjepa_2_1_frozen_shuffled.npy    (1000, 1664)  ← V-JEPA 2.1 shuffled
├── embeddings_vjepa_2_1_explora.npy            (1000, 1664)  ← ExPLoRA adapted
├── embeddings_vjepa_2_1_surgical.npy           (1000, 1664)  ← Surgery adapted
├── factors/
│   ├── masks/{clip_key}.npz                    ← SAM 3.1 agent/layout masks
│   ├── D_L/{clip_key}.npy                      ← layout-only (agents blurred)
│   ├── D_A/{clip_key}.npy                      ← agent-only (background suppressed)
│   ├── D_I/{clip_key}_tube{idx}.npy            ← interaction tubes
│   ├── segments.json                           ← per-clip segmentation metadata
│   └── factor_manifest.json                    ← D_L/D_A/D_I availability per clip
```

---

## Decision Gate (end of Week 1)

| Step 1b ExPLoRA | Step 2 Surgery | Action |
|---|---|---|
| ExPLoRA improves | Surgery > ExPLoRA | **Strongest: surgery wins** |
| ExPLoRA improves | Surgery = ExPLoRA | **Publish ExPLoRA, surgery adds no value** |
| No change | Surgery improves | **Best novelty: standard fails, surgery succeeds** |
| ExPLoRA improves | Surgery < ExPLoRA | **Publish ExPLoRA, drop surgery** |
| No change | No change | **Debug: reverse factor order, more clips, LoRA fallback** |

If Surgery > ExPLoRA on 1K → scale to 10K for paper (statistically significant Prec@K with 95% CI).

---

## Scripts Architecture

```
scripts/
├── lib/common.sh           # Shared: log, run_step, verify, watchdog, bg_upload
├── prep_data.sh            # Ch9: m04(tags) + m04d(motion) — DONE
├── train_explora.sh        # Step 1b: frozen embed + ExPLoRA train + re-embed + eval
├── train_surgery.sh        # Step 2: m10 SAM 3.1 + m11 factors + m09 surgery + eval
├── train_pretrain.sh       # Ch10: legacy continual pretraining — DEFERRED
├── run_embed.sh            # ALL: m05/m05b embedding (auto-detects encoders)
└── run_eval.sh             # ALL: m06→m08b evaluation (auto-detects encoders)
```

---

## Config Architecture

```
configs/
├── pipeline.yaml                    # Shared: clip limits, encoders, streaming, eval
├── tag_taxonomy.json                # VLM tag schema (notable_objects → SAM 3.1 prompts)
├── model/
│   ├── vjepa2_0.yaml                # Legacy (1B, 1408-dim)
│   └── vjepa2_1.yaml                # PRIMARY (2B, 1664-dim)
└── train/
    ├── base_optimization.yaml       # Shared: masking, augmentation, AdamW, EMA
    ├── ch10_pretrain.yaml           # Drift control + lambda sweep — DEFERRED
    ├── explora.yaml                 # LoRA rank=16 + unfreeze 2 blocks
    └── ch11_surgery.yaml            # 3-stage unfreezing + SAM 3.1 + factor datasets
```

---

## Previous Results (Ch10, April 5, 2026)

| Metric | Frozen | Ch10 Adapted (lambda=0.001) | Delta |
|---|---|---|---|
| Prec@K | 36.1 +/-0.6 | 14.3 +/-0.3 | **-21.8pp (FAILED)** |
| nDCG@K | 0.950 +/-0.001 | 0.906 +/-0.001 | **-0.045 (FAILED)** |

Diagnosis: lambda=0.001 drift penalty 1000x smaller than JEPA loss. Gold standard audit found 12 discrepancies — all fixed in new configs.

---

## Acceptance Bar (NeurIPS-grade)

- 1% Prec@K from a single run is noise (arXiv:2511.19794)
- Need: non-overlapping 95% bootstrap CIs on nDCG@K + majority (5/8) metrics improved
- POC shortcut: bootstrap CIs on 1K val subset as fast signal
- For paper: 3-5 training seeds on 10K+ clips
