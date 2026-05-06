# 🚀 iter13 Runbook

> 4 V-JEPA variants (`vjepa_2_1_{frozen, pretrain, surgical_3stage_DI, surgical_noDI}`); DINOv2 skipped. Pre-flight auto-drops variants whose `student_encoder.pt` is missing. Replace `<MODE>` with `SANITY` (lower-cased `sanity`) or `FULL` (`full`). m09c reads `outputs/full/m11_factor_datasets` regardless of mode.

## 📋 Pipeline (one table, in run-order)

| # | Step | What it does | Wall (SANITY \| FULL) | Required by |
|---|---|---|---|---|
| 0a | m04d motion features (eval_10k_local) | RAFT optical-flow → 13D / clip → durable `data/eval_10k_local/motion_features.npy` (other datasets per `m04d` USAGE docstring) | n/a \| ~57 min | step 1, step 5 |
| 0b | factor prep (m10 → m11 streaming) | builds D_L/D_A/D_I tubes for surgery; `surgery_3stage_DI.yaml` toggles D_I on; `surgery_noDI` reuses but ignores D_I | ~30 min \| ~6-8 GPU-h | steps 3, 4 |
| 1 | eval frozen-only (`SKIP_STAGES=4,7,9,10`) | also emits `action_labels.json` for trainers | ~6-8 min \| ~30-40 min | steps 2, 3, 4 |
| 2 | train pretrain (m09a, λ=0, motion_aux ON in v12+) | continual SSL + joint CE+MSE motion supervision | ~3 min \| **~7.7 hr** (1010 steps × 27.6 s, observed in v12) | step 5 |
| 3 | train surgery_3stage_DI (m09c with D_I) | factor-progressive unfreezing | ~10 min \| ~6-8 GPU-h | step 5 |
| 4 | train surgery_noDI (m09c without D_I) | ablation control | ~7 min \| ~4-6 GPU-h | step 5 |
| 5 | eval all 4 V-JEPA variants (paired-Δ) | 🔥 P1 gate — `probe_paired_delta.json` + per-dim taxonomy + motion_cos + future_mse | ~10 min \| ~3 GPU-h | — |

> **Steps 2/3/4 are sequential on a single GPU**; total FULL wall ≈ 13-19 GPU-h training + 3-4 GPU-h eval.

---

## 🖥️ Terminal commands

### 0a — m04d motion features (one-time, durable)
```bash
CACHE_POLICY_ALL=2 python -u src/m04d_motion_features.py --FULL \
    --subset data/eval_10k.json --local-data data/eval_10k_local \
    --features-out data/eval_10k_local/motion_features.npy \
    --no-wandb 2>&1 | tee logs/m04d_full_eval10k_v1.log

python -c "import numpy as np; v=np.load('data/eval_10k_local/motion_features.npy'); print('shape:', v.shape, 'mean_mag pct:', np.percentile(v[:,0], [0,25,50,75,100]))"

python -u src/utils/hf_outputs.py upload-data 2>&1 | tee logs/upload_motion_features.log
```

### 0b — factor prep (m10 + m11 streaming)
```bash
# SANITY (~30 min) — required for steps 1.3-style runs only when surgery is in scope
python -u src/m10_sam_segment.py --SANITY \
    --train-config configs/train/surgery_3stage_DI.yaml \
    --subset data/eval_10k_sanity.json --local-data data/eval_10k_local \
    --output-dir outputs/full/m10_sam_segment \
    --cache-policy 2 --no-wandb 2>&1 | tee logs/factor_prep_m10_sanity_v1.log

python -u src/m11_factor_datasets.py --SANITY --streaming \
    --train-config configs/train/surgery_3stage_DI.yaml \
    --subset data/eval_10k_sanity.json --local-data data/eval_10k_local \
    --input-dir outputs/full/m10_sam_segment \
    --output-dir outputs/full/m11_factor_datasets \
    --cache-policy 2 --no-wandb 2>&1 | tee logs/factor_prep_m11_sanity_v1.log

# FULL (~6-8 GPU-h) — same commands with --FULL --subset data/eval_10k.json + tee *_full_v1.log
```

### 1 — eval frozen-only
```bash
# SANITY
SKIP_STAGES="4,7,9,10" CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --sanity \
  2>&1 | tee logs/probe_eval_sanity_frozen_v1.log
# FULL
SKIP_STAGES="4,7,9,10" CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/probe_eval_full_frozen_v1.log
```

### 2 — train pretrain (m09a, motion_aux v12+)
```bash
# SANITY
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain --SANITY \
  2>&1 | tee logs/probe_train_pretrain_sanity_v1.log
# FULL (tmux for resilience)
tmux new -s p2_pretrain -d "
  CACHE_POLICY_ALL=1 ./scripts/run_probe_train.sh pretrain --FULL \
    2>&1 | tee logs/probe_train_pretrain_full_v12.log
"
```

### 3 — train surgery_3stage_DI (m09c with D_I)
```bash
# SANITY
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --SANITY \
  2>&1 | tee logs/probe_train_surgery_3stage_DI_sanity_v1.log
# FULL
tmux new -s p3_3DI -d "
  CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL \
    2>&1 | tee logs/probe_train_surgery_3stage_DI_full_v1.log
"
```

### 4 — train surgery_noDI (m09c control)
```bash
# SANITY
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI --SANITY \
  2>&1 | tee logs/probe_train_surgery_noDI_sanity_v1.log
# FULL
tmux new -s p3_noDI -d "
  CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI --FULL \
    2>&1 | tee logs/probe_train_surgery_noDI_full_v1.log
"
```

### 5 — eval all 4 V-JEPA variants
```bash
# SANITY
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --sanity \
  2>&1 | tee logs/probe_eval_sanity_all4_v1.log
# FULL
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/probe_eval_full_all4_v1.log
```

### Read the gates
```bash
jq '.pairwise_deltas.vjepa_2_1_pretrain_minus_vjepa_2_1_frozen' \
  outputs/full/probe_action/probe_paired_delta.json                                  # 🔥 P1 motion-flow

jq '.pairwise_deltas.vjepa_2_1_surgical_3stage_DI_minus_vjepa_2_1_pretrain' \
  outputs/full/probe_action/probe_paired_delta.json                                  # surgery vs continual SSL

jq '.pairwise_deltas.vjepa_2_1_surgical_3stage_DI_minus_vjepa_2_1_surgical_noDI' \
  outputs/full/probe_action/probe_paired_delta.json                                  # does D_I help?

jq '.by_variant' outputs/full/probe_future_mse/probe_future_mse_per_variant.json     # JEPA's native obj (lower=better)

ls -lh outputs/full/probe_plot/probe_encoder_comparison.{png,pdf}                    # final 4-encoder bar plot
```

### Verify per-variant training success
```bash
for v in probe_pretrain probe_surgery_3stage_DI probe_surgery_noDI; do
  echo "── $v ──"
  ls -lh outputs/full/$v/{student_encoder,m09{a,c}_ckpt_best,multi_task_head,motion_aux_head}.pt 2>/dev/null
  wc -l outputs/full/$v/loss_log.jsonl
done
```

### tmux ops
```bash
tmux ls                                    # list
tmux attach -t p2_pretrain                 # attach (Ctrl-b d detach)
tmux kill-session -t p2_pretrain           # kill
```
