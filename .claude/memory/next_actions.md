---
name: Next concrete actions on a fresh 96 GB instance
description: Copy-paste-safe sequence to resume iter14 from current state (Phase 4 motion_aux into surgery)
type: project
---

# Next actions — pick up iter14 on a fresh 96 GB GPU instance

> Prerequisites: repo cloned, `setup_env_uv.sh` run, `checkpoints/vjepa2_1_vitG_384.pt` (29 GB) downloaded. The v12 pretrain student + factor data come from HF, **not** from a fresh re-train.

## Step 0 — verify environment

```bash
cd /workspace/factorjepa
source venv_walkindia/bin/activate
nvidia-smi | head -10                                # confirm 96 GB Blackwell
ls -lh checkpoints/vjepa2_1_vitG_384.pt              # ~29 GB; from setup_env_uv.sh aria2c
ls configs/tag_taxonomy.json                         # 15-dim taxonomy (still consumed by tagging path)
python -c "import bitsandbytes; print('bnb', bitsandbytes.__version__)"   # required for 8-bit Adam paths
```

If any FAIL → run `./setup_env_uv.sh` first.

## Step 1 — pull v12 pretrain student + factor data from HF (~10-15 min, network-bound)

```bash
# Pulls outputs/full/ + data/eval_10k_local/ from HF. Auto-unpacks tars, then auto-deletes them (FIX-28).
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/hf_download_data.log
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py download outputs/full 2>&1 | tee logs/hf_download_outputs.log

# Verify the v12 anchor lands intact
ls -lh outputs/full/probe_pretrain/student_encoder.pt           # ~6.9 GB — the v12 trained encoder
ls -lh outputs/full/probe_pretrain/m09a_ckpt_best.pt            # ~14 GB — predictor-bearing for Stage 8
tail -1 outputs/full/probe_pretrain/probe_history.jsonl | jq    # expect probe_top1 ≈ 0.808
ls -lh data/eval_10k_local/m10_sam_segment/segments.json        # m10 factor segments
ls -lh data/eval_10k_local/m11_factor_datasets/factor_manifest.json   # m11 D_L/D_A/D_I manifest
ls -lh data/eval_10k_local/motion_features.npy                  # 13-D motion features (9297, 13)
```

If the v12 anchor doesn't reproduce (`probe_top1` far from 0.808), that's a HF roundtrip bug — investigate before re-training.

## Step 2 — RESOLVE THE 3 APPROVAL GATES first (no code changes until user replies)

Per `iter/iter14_surgery_on_pretrain/plan_surgery_on_pretrain.md` lines ~308-315:

1. **Epoch budget** — 🅰️ "5+5 vs 10" (~$33, recommended) OR 🅱️ "5+15 vs 20" (~$57)
2. **Anchor `λ`** — `drift_control.lambda_reg = 0.005` (literature default) OR 3-point sweep `{0.001, 0.005, 0.01}` (3× surgery cost)
3. **HF push of pretrain** — ✅ DONE

User reply form: `"go: 🅰️, λ=0.005"`. **Do not start T4 until this lands.**

## Step 3 — execute T4 (Phase 4 code edits, ~50 LoC across 5 files)

Per `plan_motion_aux_to_surgery.md`:

```
1. configs/train/surgery_base.yaml         — λ audit: set drift_control.lambda_reg per gate-2 reply
2. configs/train/surgery_3stage_DI.yaml    — disable multi_task_probe, enable motion_aux block (mirror probe_pretrain.yaml:176-189)
3. configs/train/surgery_2stage_noDI.yaml  — same as #2
4. src/m09c_surgery.py                     — 9 call sites mirroring m09a v12 (imports / argparse / merge / build / per-stage optim re-attach / run_step / step_record / wb_metrics / export)
5. scripts/run_probe_train.sh              — thread --motion-features-path "${LOCAL_DATA}/motion_features.npy" in surgery_3stage_DI + surgery_noDI cases
```

3-check + ruff + bash -n after each edit:
```bash
python -m py_compile src/m09c_surgery.py && \
  python -c "import ast; ast.parse(open('src/m09c_surgery.py').read())" && \
  ruff check --select F,E9 src/m09c_surgery.py && \
  bash -n scripts/run_probe_train.sh
```

## Step 4 — T5 SANITY smoke (~25 min)

```bash
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --SANITY \
  2>&1 | tee logs/probe_train_surgery_3stage_DI_sanity_v1.log

# Pass criteria:
#   - "[motion_aux] enabled: 8 classes ..." appears 3× (once per stage rebuild — confirms per-stage optimizer re-attach works)
#   - loss_log.jsonl has loss_motion_aux + loss_motion_aux_ce + loss_motion_aux_mse rows
#   - motion_aux_head.pt written next to student_encoder.pt
```

## Step 5 — T6 FULL iter14 training arms (gated on T5 green)

Three arms, semicolon-chained NOT && — per `;`-not-`&&` rule (overnight independence):

```bash
# Arm 1: surgery_3stage_DI on pretrain init (~6-8 GPU-h)
tmux new -s p4_3DI -d "
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL \
  2>&1 | tee logs/probe_train_surgery_3stage_DI_full_v14.log
"

# Arm 2: surgery_noDI on pretrain init (~4-6 GPU-h, after Arm 1 if disk-bound)
tmux new -s p4_noDI -d "
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI --FULL \
  2>&1 | tee logs/probe_train_surgery_noDI_full_v14.log
"

# Arm 3: long-pretrain compute control (~12-15 GPU-h depending on gate-1 budget)
# (specific command lives in plan_surgery_on_pretrain.md after gate-1 resolves)
```

## Step 6 — full eval (4 V-JEPA + DINOv2, ~3 GPU-h)

```bash
tmux new -s probe_eval_full -d "
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/probe_eval_full_all4_v6.log
"
```

Expect Stage 8 to include `vjepa_2_1_pretrain` + both surgical variants + frozen + dinov2 (DINOv2 has no predictor → auto-skipped at Stage 8).

## Step 7 — read the iter14 gates

```bash
# P3 — surgery vs pretrain (the iter14 paper claim)
jq '.pairwise_deltas.vjepa_2_1_surgical_3stage_DI_minus_vjepa_2_1_pretrain' \
  outputs/full/probe_action/probe_paired_delta.json

# Compute-control: surgery vs long-pretrain (causal attribution)
jq '.pairwise_deltas.vjepa_2_1_surgical_3stage_DI_minus_vjepa_2_1_long_pretrain' \
  outputs/full/probe_action/probe_paired_delta.json

# Phase 5 trigger
# If Δ (surgery_3DI − pretrain) < +5 pp → escalate to Phase 5 (FG motion features)
# Plan: iter/iter14_surgery_on_pretrain/plan_phase5_fg_motion_features.md
```

## Step 8 — interpret + iterate

| Outcome | Action |
|---|---|
| `surgery_3DI ≫ pretrain ≫ frozen` (≥ +5 pp each) AND surgery > long_pretrain | 🏆 publishable — write the paper. Push final outputs to HF. |
| `surgery_3DI ≈ long_pretrain` | Gain is from extra steps, not factor patching. Drop surgery claim. |
| Δ (surgery_3DI − pretrain) < +5 pp | Phase 5 escalation: extend m04d to 23-D FG motion (~25 GPU-h). See `plan_phase5_fg_motion_features.md`. |
| `surgery_noDI ≈ surgery_3DI` | D_I tubes don't help; drop D_I from paper (matches iter9 v15c BWT=-0.33 finding). |

## Useful one-liners

```bash
# Tail all 3 trainer tmux logs at once
multitail -f logs/probe_train_surgery_3stage_DI_full_v14.log \
          -f logs/probe_train_surgery_noDI_full_v14.log \
          -f logs/probe_train_long_pretrain_full_v1.log

# Mid-train motion_aux health check
jq -r 'select(.loss_motion_aux != null) | "step=\(.step) jepa=\(.loss_jepa) ma=\(.loss_motion_aux) ce=\(.loss_motion_aux_ce) mse=\(.loss_motion_aux_mse)"' \
  outputs/full/probe_surgery_3stage_DI/loss_log.jsonl | tail -20

# Push iter14 outputs back to HF on completion (mirrors stale by default — see logs/hf_upload_outputs_full.log pattern)
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload outputs/full \
  2>&1 | tee logs/hf_upload_outputs_full_iter14.log
```
