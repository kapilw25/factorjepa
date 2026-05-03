---
name: Next concrete actions on a fresh 96 GB instance
description: Copy-paste-safe command sequence to resume iter13 from current state
type: project
---

# Next actions — pick up iter13 on a fresh 96 GB GPU instance

> Prerequisites: repo cloned, `setup_env_uv.sh` run, `checkpoints/vjepa2_1_vitG_384.pt` (29 GB) downloaded, `data/eval_10k_local/` (~9.9k clips, ~80 GB) downloaded. If any missing, see `iter/utils/setup_*.md`.

## Step 0 — verify environment

```bash
cd /workspace/factorjepa
source venv_walkindia/bin/activate
nvidia-smi | head -10                                # confirm 96 GB Blackwell
ls -lh checkpoints/vjepa2_1_vitG_384.pt              # ~29 GB; from setup_env_uv.sh aria2c
ls data/eval_10k_local/manifest.json data/eval_10k.json   # eval clip pool
ls data/eval_10k_local/tags.json                     # required for taxonomy_labels gen
ls configs/tag_taxonomy.json                         # 15-dim taxonomy spec
python -c "import bitsandbytes; print('bnb', bitsandbytes.__version__)"   # required for 8-bit Adam paths
```

If any FAIL → run `./setup_env_uv.sh` first.

## Step 1 — bootstrap labels at FULL scale (CPU only, ~2 min)

```bash
# Generates outputs/full/probe_action/action_labels.json (Stage 1 P1 split — 70/15/15 stratified)
# AND outputs/full/probe_taxonomy/taxonomy_labels.json (16-dim multi-task supervision source)
SKIP_STAGES="2,3,4,5,6,7,8,9,10" CACHE_POLICY_ALL=2 \
  ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/run_probe_eval_full_stage1_only.log

# Verify both label files exist
jq -r '. | length' outputs/full/probe_action/action_labels.json       # should be ~9,951
jq -r '.dims | keys | length' outputs/full/probe_taxonomy/taxonomy_labels.json   # should be 16
jq -r '.labels | keys | length' outputs/full/probe_taxonomy/taxonomy_labels.json # should be ~9,500-9,951
```

## Step 2 — train all 3 V-JEPA variants on 96 GB (sequential, ~13-19 GPU-h total)

```bash
# tmux for each, OR semicolon-chained for fire-and-forget. Semicolons NOT && — failure
# in one trainer must not kill the next. Each writes independent artifacts.

tmux new -s p2_pretrain -d '
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain --FULL \
  2>&1 | tee logs/probe_pretrain_full_v1.log
'
# ~3 GPU-h. Watch for:
#   "[multi-task] enabled: 16 dims, ~7000 clips labeled, 139,860 head params"
#   loss_log.jsonl growing (was 0 bytes pre-fix; now > 0)
#   "Exported student encoder: outputs/full/probe_pretrain/student_encoder.pt"  (~7 GB)
#   "New best val loss: ..."                                                     (writes ~15 GB _best.pt)
#   "Exported predictor-bearing best ckpt: ...m09a_ckpt_best.pt"  (would be the future_mse Stage 8 input)

tmux new -s p3_surgery_3stage_DI -d '
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL \
  2>&1 | tee logs/probe_surgery_3stage_DI_full_v1.log
'
# ~6-8 GPU-h. Three stages: D_L (12/48 layers) → D_A (24/48) → D_I (24/48 with interaction tubes)
# Watch for "[best] Promoted student_best.pt → student_encoder.pt"
# Then "Exported predictor-bearing best ckpt: ...m09c_ckpt_best.pt"

tmux new -s p3_surgery_noDI -d '
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI --FULL \
  2>&1 | tee logs/probe_surgery_noDI_full_v1.log
'
# ~4-6 GPU-h. Two stages: D_L → D_A only (no D_I — controlled comparison vs 3stage_DI).
```

If any trainer raises `M09A FAILED: 0 successful training steps` or `M09C FAILED: ...`, that's the iter13 fail-hard guard — investigate the log, but don't expect this on 96 GB (the fail-hard is for 24 GB OOM cases).

### Mid-run health checks (per tmux)

```bash
# tail the live JSONL (crash-safe writes)
tail -f outputs/full/probe_pretrain/loss_log.jsonl

# multi-task loss diagnostic — should see jepa_loss + multi_task_loss BOTH dropping
jq -r 'select(.loss_jepa != null) | "step=\(.step) jepa=\(.loss_jepa) mt=\(.loss_multi_task // \"\")"' \
  outputs/full/probe_pretrain/loss_log.jsonl | tail -20

# probe_acc trajectory (mid-train kNN-centroid eval at every val cadence)
tail -10 outputs/full/probe_pretrain/probe_history.jsonl
```

## Step 3 — full eval (4 V-JEPA + DINOv2, ~2.5 GPU-h)

```bash
tmux new -s probe_eval_full -d '
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/run_src_probe_full_v1.log
'

# Pre-flight should show all 5 encoders kept:
#   ✓ vjepa_2_1_frozen: external (no trainer needed)
#   ✓ vjepa_2_1_pretrain: outputs/full/probe_pretrain/student_encoder.pt
#   ✓ vjepa_2_1_surgical_3stage_DI: outputs/full/probe_surgery_3stage_DI/student_encoder.pt
#   ✓ vjepa_2_1_surgical_noDI: outputs/full/probe_surgery_noDI/student_encoder.pt
#
# Stage 8 ENCODERS should include all 4 V-JEPA variants:
#   ✓ vjepa_2_1_frozen
#   ✓ vjepa_2_1_pretrain
#   ✓ vjepa_2_1_surgical_3stage_DI
#   ✓ vjepa_2_1_surgical_noDI
```

## Step 4 — read the gates

```bash
# P1 — V-JEPA frozen vs DINOv2
jq '.pairwise_deltas.vjepa_2_1_frozen_minus_dinov2' \
  outputs/full/probe_action/probe_paired_delta.json
# PASS: delta_pp ≥ +20, ci_lo_pp > 0, p_value < 0.05, gate_pass: true

# P2 — pretrain vs frozen (does multi-task supervision help over frozen?)
jq '.pairwise_deltas.vjepa_2_1_pretrain_minus_vjepa_2_1_frozen' \
  outputs/full/probe_action/probe_paired_delta.json
# PASS: delta_pp > 0, ci_lo_pp > 0, p_value < 0.05

# P3 — surgical_3stage_DI vs pretrain (does factor surgery help over continual SSL?)
jq '.pairwise_deltas.vjepa_2_1_surgical_3stage_DI_minus_vjepa_2_1_pretrain' \
  outputs/full/probe_action/probe_paired_delta.json

# Sanity check — D_I helps or hurts?
jq '.pairwise_deltas.vjepa_2_1_surgical_3stage_DI_minus_vjepa_2_1_surgical_noDI' \
  outputs/full/probe_action/probe_paired_delta.json
# Δ > 0 → D_I helps; Δ ≤ 0 → D_I noisy (matches iter9 v15c BWT=-0.33 finding)

# Future MSE (V-JEPA's native objective; lower better)
jq '.by_variant | to_entries | map({k: .key, mse: .value.mse_mean})' \
  outputs/full/probe_future_mse/probe_future_mse_per_variant.json

# Final 4-encoder bar chart
ls -lh outputs/full/probe_plot/
# probe_action_loss.{png,pdf}        — Stage 3 train_loss curves per encoder
# probe_action_acc.{png,pdf}         — Stage 3 val_acc curves per encoder
# probe_encoder_comparison.{png,pdf} — 3-panel: action acc + motion cos + future MSE bars
```

## Step 5 — interpret + iterate

If P1 PASSES (V-JEPA frozen > DINOv2) AND P2 PASSES (pretrain > frozen) AND P3 PASSES (surgical > pretrain), this is the publishable result. Update `iter/utils/experiment_log.md` with the verdict and write the paper.

If P2 or P3 FAILS, re-read `iter/iter13_motion_probe_eval/analysis.md` Q7 for the iter13 Framing B fallback (factor-conditioned PROBE HEAD, NOT continual encoder pretrain) — but that requires iter14 architectural changes. iter13 has shipped its design.

## Useful Bash one-liners

```bash
# Wipe ALL outputs (DANGEROUS — only when starting fresh)
# Per CLAUDE.md DELETE PROTECTION: never use shell-level `rm`; use --cache-policy 2 on every script.

# Check disk usage by trainer output dir
du -sh outputs/full/probe_*/

# Find the predictor-bearing ckpt for each variant (Stage 8 input)
for v in probe_pretrain probe_surgery_3stage_DI probe_surgery_noDI; do
  find outputs/full/$v/ -name "m09[ac]_ckpt_best.pt" -ls
done

# Tail all 4 trainer logs in one terminal
multitail -f logs/probe_pretrain_full_v1.log \
          -f logs/probe_surgery_3stage_DI_full_v1.log \
          -f logs/probe_surgery_noDI_full_v1.log \
          -f logs/run_src_probe_full_v1.log
```
