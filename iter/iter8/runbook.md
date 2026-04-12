# FactorJEPA Runbook — GPU Instance Commands

> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**
> **Short-term: Show Surgery > ExPLoRA > Frozen on 1K val clips (~70 min GPU).**
>
> **Key files for new Claude sessions:**
> - `iter/iter8/next_steps.md` — what to run + implementation status
> - `iter/iter8/plan_code_development.md` — code plan for remaining work
> - `iter/iter8/plan_training.md` — system design diagrams + paper strategy
> - `src/CLAUDE.md` — codebase rules (32 rules, hook-enforced)
> - `src/MEMORY.md` — project state, pipeline modules, encoder registry

---

## GPU Instance Setup

```bash
# 1. Pull latest code
git pull origin main

# 2. Setup (clones vjepa2 + downloads 2.1 checkpoint + installs deps)
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
source venv_walkindia/bin/activate

# 3. Install SAM 3.1 (gated — accept access at hf.co/facebook/sam3.1 first)
huggingface-cli login
pip install git+https://github.com/facebookresearch/sam3.git

# 4. Install PEFT for ExPLoRA
pip install peft>=0.13.0

# 5. Download data (POC 1K val, ~3 min)
python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/download_poc_val.log
```

---

## Step 1b: ExPLoRA [~1h GPU — READY TO RUN NOW]

```bash
./scripts/train_explora.sh --POC 2>&1 | tee logs/explora_poc.log

# What it does:
#   Step 0: frozen V-JEPA 2.1 embed on 1K clips + eval (~12 min, skips if cached)
#   Step 1: ExPLoRA training — LoRA rank=16, unfreeze blocks 0-1 (~20 min)
#   Step 2: re-embed adapted model (~12 min)
#   Step 3: evaluate Prec@K
#
# Compare:
#   outputs/poc/m06_metrics_vjepa_2_1_frozen.json    ← frozen baseline
#   outputs/poc/m06_metrics_vjepa_2_1_explora.json   ← ExPLoRA adapted
```

---

## Step 2: Surgery [~35 min GPU — READY TO RUN]

```bash
./scripts/train_surgery.sh --POC 2>&1 | tee logs/surgery_poc.log

# What it does:
#   Step 0: m10 SAM 3.1 on 1K clips — per-clip text prompt from tags.json (~5 min GPU)
#   Step 1: m11 factor datasets D_L + D_A + D_I with feathered edges (~3 min CPU)
#   Step 2: m09 --surgery — 3-stage progressive prefix unfreezing (~15 min GPU)
#           Stage 1: layers 0-12, 100% D_L (layout)
#           Stage 2: layers 0-24, 90% D_A + 10% D_L replay
#           Stage 3: layers 0-36, 85% D_I + 10% D_A + 5% D_L replay
#   Step 3: m05 re-embed surgical model (~12 min GPU)
#   Step 4: m06 evaluate Prec@K
#
# THE KEY COMPARISON:
#   outputs/poc/m06_metrics_vjepa_2_1_frozen.json     ← frozen baseline
#   outputs/poc/m06_metrics_vjepa_2_1_explora.json    ← ExPLoRA (from Step 1b)
#   outputs/poc/m06_metrics_vjepa_2_1_surgical.json   ← surgery adapted
```

---

## Step 1a: Temporal Projection [30 min CPU, optional]

After frozen 2.1 embeddings exist (from Step 1b Step 0):

```bash
python -u src/m05b_baselines.py --encoder vjepa_2_1_frozen_shuffled \
    --model-config configs/model/vjepa2_1.yaml \
    --POC --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05b_shuffled.log

python -u src/m06c_temporal_projection.py --POC \
    --normal-encoder vjepa_2_1_frozen --shuffled-encoder vjepa_2_1_frozen_shuffled \
    2>&1 | tee logs/m06c.log
```

---

## Gold Standard Fixes Applied (this session)

| Fix | What | Impact |
|---|---|---|
| Dense loss | Context loss on ALL tokens, lambda=0.5 | Doubles training signal (was masked-only) |
| Deep supervision | 4-layer hierarchical output (6656-dim) | 4x supervision depth |
| Predictor LR 1x | Same LR for encoder + predictor | Matches Meta (was 10x) |
| return_hierarchical | Student + teacher produce 6656-dim | Prevents mat1/mat2 crash |
| 2.1 imports | `app/vjepa_2_1/models/` not `src/models/` | Correct ViT with deep supervision |
| SAM 3.1 API | `propagate_in_video`, `handle_stream_request`, `close_session` | Fixes 4 crash bugs |
| Mask feathering | Gaussian blur on mask before blend (sigma=3) | Prevents shortcut learning |
| Quality filters | min 2% / max 70% agent area | Removes degenerate samples |
| Cooldown config | 64f, linear decay, in base_optimization.yaml | V-JEPA 2.1 recipe |

---

## Config Architecture

```
configs/
├── pipeline.yaml                    # Shared: clip limits, encoders, streaming, eval
├── tag_taxonomy.json                # VLM tag schema → SAM 3.1 per-clip prompts
├── YT_videos_raw.json               # Video metadata
├── model/
│   ├── vjepa2_0.yaml                # Legacy (1B, 1408-dim)
│   └── vjepa2_1.yaml                # PRIMARY (2B, 1664-dim, 4-layer deep supervision)
└── train/
    ├── base_optimization.yaml       # Shared: masking, augmentation, AdamW, EMA, pred_lr 1x, cooldown
    ├── ch10_pretrain.yaml           # Ch10 drift control — DEFERRED
    ├── explora.yaml                 # LoRA rank=16 + unfreeze 2 blocks
    └── ch11_surgery.yaml            # 3-stage unfreezing + SAM 3.1 + factor datasets + feathering
```

---

## Encoder Names

| Encoder | Dim | File suffix | What |
|---|---|---|---|
| `vjepa` | 1408 | (none) | V-JEPA 2.0 frozen (existing Ch9) |
| `vjepa_2_0_frozen` | 1408 | `_vjepa_2_0_frozen` | Explicit 2.0 frozen |
| `vjepa_2_1_frozen` | 1664 | `_vjepa_2_1_frozen` | V-JEPA 2.1 frozen baseline |
| `vjepa_2_1_frozen_shuffled` | 1664 | `_vjepa_2_1_frozen_shuffled` | Shuffled (temporal diagnostic) |
| `vjepa_2_1_explora` | 1664 | `_vjepa_2_1_explora` | ExPLoRA adapted |
| `vjepa_2_1_surgical` | 1664 | `_vjepa_2_1_surgical` | Surgery adapted |

---

## Previous Results (Ch10, April 5, 2026)

| Metric | Frozen | Ch10 Adapted (lambda=0.001) | Delta |
|---|---|---|---|
| Prec@K | 36.1 +/-0.6 | 14.3 +/-0.3 | **-21.8pp (FAILED)** |

Catastrophic forgetting. Gold standard audit found 12 discrepancies — all fixed in new configs.
