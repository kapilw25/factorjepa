# 🚀 iter14 surgery-on-pretrain — command sheet

## ✅ Prerequisites

- venv: `source venv_walkindia/bin/activate`
- `.env` at repo root with `HF_TOKEN` (FAIL LOUD if missing)
- `data/eval_10k_local/{motion_features.npy, m11_factor_datasets/}` on disk → else `python -u src/utils/hf_outputs.py download-data`
- HF endpoints live: `huggingface.co/anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep`

---

## 🧪 1️⃣ Boot smoke test — 24 GB (~$0, ~2 min) ✅ **DONE 2026-05-09**

> Verified end-to-end: 5 recipe-v2 wiring claims confirmed by GPU execution.
> Evidence log: `logs/m09c_surgery_3stage_DI_sanity.log` (Stage 0 ran head-only at 0/48 blocks, loss=0.5111; Stage 1 ran at 12/48 blocks, loss=0.5072; Stage 2 OOM at 24/48 blocks — expected per 24 GB ceiling).
> Skip this section on future runs unless flag wiring changes.

---

## 🧬 2️⃣ Recipe-v2 POC sweep — 96 GB Blackwell (~$1.20, ~90 min, 4 cells)

> ⚠️ Requires RTX Pro 6000 Blackwell (96 GB). 24 GB will OOM at first training batch.

```bash
tmux new -s iter14_poc_sweep
cd /workspace/factorjepa && source venv_walkindia/bin/activate

# Cell A — EMA × LP-FT off (legacy baseline; reproduces iter14 POC v3)
CACHE_POLICY_ALL=2 TEACHER_MODE_OVERRIDE=EMA    LP_FT_OVERRIDE=off \
  ./scripts/run_probe_train.sh surgery_3stage_DI --POC 2>&1 \
  | tee logs/iter14_poc_recipe_v2_ema_lpft-off.log
mv outputs/poc/m09c_surgery_3stage_DI outputs/poc/m09c_surgery_3stage_DI__ema_lpft-off

# Cell B — EMA × LP-FT on
CACHE_POLICY_ALL=2 TEACHER_MODE_OVERRIDE=EMA    LP_FT_OVERRIDE=on  \
  ./scripts/run_probe_train.sh surgery_3stage_DI --POC 2>&1 \
  | tee logs/iter14_poc_recipe_v2_ema_lpft-on.log
mv outputs/poc/m09c_surgery_3stage_DI outputs/poc/m09c_surgery_3stage_DI__ema_lpft-on

# Cell C — FROZEN × LP-FT off (SALT alone)
CACHE_POLICY_ALL=2 TEACHER_MODE_OVERRIDE=FROZEN LP_FT_OVERRIDE=off \
  ./scripts/run_probe_train.sh surgery_3stage_DI --POC 2>&1 \
  | tee logs/iter14_poc_recipe_v2_frozen_lpft-off.log
mv outputs/poc/m09c_surgery_3stage_DI outputs/poc/m09c_surgery_3stage_DI__frozen_lpft-off

# Cell D — FROZEN × LP-FT on (🥇 full recipe v2)
CACHE_POLICY_ALL=2 TEACHER_MODE_OVERRIDE=FROZEN LP_FT_OVERRIDE=on  \
  ./scripts/run_probe_train.sh surgery_3stage_DI --POC 2>&1 \
  | tee logs/iter14_poc_recipe_v2_frozen_lpft-on.log
mv outputs/poc/m09c_surgery_3stage_DI outputs/poc/m09c_surgery_3stage_DI__frozen_lpft-on
```

### 📊 Aggregate the 4 cells' trio top-1 trajectories

```bash
for cell in ema_lpft-off ema_lpft-on frozen_lpft-off frozen_lpft-on; do
  echo "=== Cell: $cell ==="
  grep -oE "step=[0-9]+ stage=\S+ N=[0-9]+ top-1=[0-9.]+ motion_cos=[0-9.\-]+" \
    logs/iter14_poc_recipe_v2_${cell}.log | tail -5
done

# Best final top-1 across cells
for cell in ema_lpft-off ema_lpft-on frozen_lpft-off frozen_lpft-on; do
  best=$(grep -oE "top-1=[0-9.]+" logs/iter14_poc_recipe_v2_${cell}.log | sort -t= -k2 -nr | head -1)
  echo "  $cell  →  $best"
done
```

### 🚦 Decision rule (per `plan_surgery_wins.md` § 7.5)

| Best cell trio top-1 | ➡️ Branch |
|---|---|
| 🟢 ≥ 0.808 AND projected test-Δ ≥ +5 pp | wire fixes via `plan_no_discrepancy.md` Phases A→D, then 3️⃣ FULL |
| 🟡 0.81–0.83 (marginal) | run Phase 5 FG-feature m04d (`plan_phase5_fg_motion_features.md`) |
| 🔴 all 4 cells regress | Path 2: relax m10 thresholds (data-scale fix) |

---

## 🚀 3️⃣ FULL training — 96 GB (post-POC 🟢, ~37 GPU-h ≈ $30)

> Fires only on 🟢 branch. Use `;` not `&&` (independent failures shouldn't cancel queue).

```bash
tmux new -s iter14_train

CACHE_POLICY_ALL=2 TEACHER_MODE_OVERRIDE=FROZEN LP_FT_OVERRIDE=on \
  ./scripts/run_probe_train.sh surgery_3stage_DI --FULL 2>&1 | tee logs/iter14_surgery_3stage_DI.log ; \
CACHE_POLICY_ALL=2 TEACHER_MODE_OVERRIDE=FROZEN LP_FT_OVERRIDE=on \
  ./scripts/run_probe_train.sh surgery_noDI      --FULL 2>&1 | tee logs/iter14_surgery_noDI.log ; \
CACHE_POLICY_ALL=2 \
  ./scripts/run_probe_train.sh pretrain_2X       --FULL 2>&1 | tee logs/iter14_pretrain_2X.log
```

---

## 🧪 4️⃣ FULL 5-encoder eval (~4 h, ~$3)

```bash
tmux new -s iter14_eval
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL 2>&1 | tee logs/iter14_probe_eval.log
```

---

## 🎯 Inspect Δ1/Δ2/Δ3 paper deltas

```bash
jq '.iter14_paper_deltas' outputs/full/probe_action/probe_paired_delta.json
```

| Delta | Pass when |
|---|---|
| Δ1 (`pretrain − frozen`) | `delta_1_pretrain_vs_frozen.ci_lo_pp > 0` |
| Δ2 (`surgical − pretrain`) | `delta_2_surgical_vs_pretrain.ci_lo_pp > 0` |
| Δ3 (`surgical − pretrain_2X`) ⭐ | `delta_3_surgical_vs_pretrain_2X.ci_lo_pp > 0` |

---

## 🔍 Mid-run diagnostics

```bash
# Probe trajectory of currently-running cell
tail outputs/poc/m09c_surgery_3stage_DI/probe_history.jsonl \
  | jq '{step, probe_top1, motion_cos, future_l1}'

# Drift loss firing (drift_control.lambda_reg=0.005; should be 0 on FROZEN cells with L2 dropped)
jq -r 'select(.loss_drift != null) | "step=\(.step) jepa=\(.loss_jepa) drift=\(.loss_drift)"' \
  outputs/poc/m09c_surgery_3stage_DI/loss_log.jsonl | tail -10

# Watch all 4 cell logs simultaneously (during sweep)
multitail -f logs/iter14_poc_recipe_v2_*.log
```
