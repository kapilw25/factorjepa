# FactorJEPA Runbook — Terminal Commands

> **Current goal:** Build pretraining params/algorithm that show improved metrics over frozen baseline.
> Embed + eval on 10K (POC) only. No 115K runs until POC shows positive results.
>
> **Acceptance bar (NeurIPS-grade):**
> - 1% Prec@K from a single run is noise, not signal (arxiv 2511.19794)
> - Need: non-overlapping 95% bootstrap CIs on **nDCG@K** (primary, MTEB standard)
>   + majority (5/8) of all metrics improved
> - For paper: 3-5 training seeds, propagated CI on delta
> - POC shortcut: bootstrap CIs on 10K subset are a proxy (r=0.84 with full)

---

## Scripts

```
scripts/
├── prep_data.sh        → Ch9:  m04(tags) + m04d(motion)
├── train_pretrain.sh   → Ch10: m09(training) only
├── train_surgery.sh    → Ch11: m09(surgical training) — TODO
├── run_embed.sh        → ALL:  m05/m05b embedding (auto-detects encoders)
└── run_eval.sh         → ALL:  m06→m08b evaluation (auto-detects encoders)
```

---

## Iteration Cycle (~7h per cycle)

Train on 115K, embed + eval on 10K. Repeat until metrics improve.

```bash
# 1. Train on 115K (6h, cached after first run)
./scripts/train_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log

# 2. Embed adapted on 10K only (~75 min)
./scripts/run_embed.sh --FULL --subset data/subset_10k.json \
    --local-data data/subset_10k_local --encoders vjepa_lambda0_001

# 3. Eval on 10K (~10 min) — choose [1] fresh start when prompted
./scripts/run_eval.sh --POC

# 4. Compare frozen vs adapted (8 metrics, 95% CI):
#    Look at: outputs/poc/m08b_spatial_temporal_bar.png (CI error bars)
#    Primary: nDCG@K (MTEB standard) — CI must not overlap frozen
#    Secondary: 5/8 metrics improved with non-overlapping CIs → positive signal
#    If nDCG@K CI overlaps frozen → change params, retrain, repeat
#    JSON: outputs/poc/m06_metrics.json vs m06_metrics_vjepa_lambda0_001.json
```

---

## Setup (one-time)

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
source venv_walkindia/bin/activate

# Data
python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/download_poc_val.log
python -u src/m00d_download_subset.py --FULL --no-wandb 2>&1 | tee logs/m00d_full.log
```

## Sanity Check (~10 min)

```bash
rm -rf outputs/sanity/
./scripts/prep_data.sh --SANITY && \
./scripts/train_pretrain.sh --SANITY && \
./scripts/run_embed.sh --SANITY && \
./scripts/run_eval.sh --SANITY
```

## Data Prep (Ch9, one-time)

```bash
./scripts/prep_data.sh --FULL 2>&1 | tee logs/ch9_full.log
```

---

## Current Status (April 5, 2026)

| Step | Status | Notes |
|---|---|---|
| Ch9 data (tags + motion) | ✅ Done | 115K clips tagged, motion features extracted |
| Ch9 frozen embeddings (115K) | ✅ Done | 5 encoders at 64f |
| Ch9 frozen eval (115K) | ✅ Done | m06→m08b, radar plot generated |
| Ch10 training (λ=0.001) | ✅ Done | 1 epoch, BS=112, 16f, loss 0.497→0.476 |
| Ch10 adapted embedding (10K) | ✅ Done | 64f, BS=44, 2.2 clips/s, 76 min |
| Ch10 POC eval | ✅ Done | 25/25 steps passed, 10.5 min |
| Ch10 result | ❌ **FAILED** | Prec@K 14.3% vs frozen 36.1% (catastrophic forgetting) |
| Val data (1K clips) | ✅ Fixed | Was 326/1000, re-downloaded to 1000/1000 |

## Result: λ=0.001 catastrophic forgetting

| Metric | Frozen | Adapted | Delta | Sig? |
|---|---|---|---|---|
| Prec@K | 36.1 ±0.6 | 14.3 ±0.3 | **-21.8pp** | YES |
| nDCG@K | 0.950 ±0.001 | 0.906 ±0.001 | **-0.045** | YES |

Adapted collapsed to random baseline (12.2%). λ=0.001 drift penalty is 1000x smaller than JEPA loss.
Full details: `iter/utils/experiment_log.md`

## Next Iteration

```bash
# 1. Update ablation_lambdas in vitg16_indian.yaml to [1.0, 10.0, 100.0]
# 2. Delete old training outputs
rm -rf outputs/full/m09_lambda0_001/ outputs/full/ablation/
# 3. Retrain
./scripts/train_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log
# 4. Re-embed + re-eval
./scripts/run_embed.sh --FULL --subset data/subset_10k.json \
    --local-data data/subset_10k_local --encoders vjepa_lambda<winner>
./scripts/run_eval.sh --POC
```

## Key Configs

| Config | Value | Location |
|---|---|---|
| Training frames | 16 | `configs/pretrain/vitg16_indian.yaml: num_frames` |
| Eval frames | 64 | `configs/pipeline.yaml: eval_frames_per_clip` |
| Training BS | 112 | `configs/pipeline.yaml: training_batch_size` |
| Adapted inference BS | 44 | `configs/pipeline.yaml: inference_adapted_bs` |
| Frozen inference BS | 176 | `configs/pipeline.yaml: inference_vjepa_bs` |
| Lambda | **0.001 (FAILED)** → next: [1.0, 10.0, 100.0] | `configs/pretrain/vitg16_indian.yaml: ablation_lambdas` |
| LR | 1e-5, pred 10x | `configs/pretrain/vitg16_indian.yaml: optimization.lr` |

## Critical Fixes Applied

| Fix | What |
|---|---|
| ImageNet normalization | m09 augmentation was [0,1], model expects [-2,2.6] |
| 16f train / 64f eval | Meta's recipe: 95% of training at 16f, eval at 64f |
| Bootstrap BCa→percentile | BCa jackknife OOM at 115K (100GB array) |
| sdp_kernel nullcontext | torch.compile graph breaks → 89GB VRAM |
| Orchestrator 95% guard | Prevents saving incomplete embeddings as final output |
| Val data 326→1000 | Incomplete TAR download, re-downloaded |
| --POC routing fix | get_output_dir() wasn't wired to --POC flag → wrong output dir |
| ls\|pipefail fix | ls glob with set -eo pipefail silently killed run_eval.sh |
