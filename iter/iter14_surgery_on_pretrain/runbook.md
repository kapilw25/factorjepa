# 🚀 iter14 surgery-on-pretrain — command sheet (recipe-v3, 2026-05-09)

## ✅ Prerequisites

- venv: `source venv_walkindia/bin/activate`
- `.env` at repo root with `HF_TOKEN` (FAIL LOUD if missing)
- `data/eval_10k_local/{motion_features.npy, m11_factor_datasets/}` on disk → else run the download command below
- HF endpoints live: `huggingface.co/anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep`

```bash
# Bootstrap data (only if data/eval_10k_local/ is missing)
python -u src/utils/hf_outputs.py download-data 2>&1 \
  | tee logs/iter14_prereq_data_download.log
```

```bash
# Verify POC labels = 8 motion classes (T1 sampler fix landed)
jq -r '[.[] | select(.class != null) | .class] | unique | length' \
  outputs/poc/probe_action/action_labels.json 2>&1 \
  | tee logs/iter14_prereq_poc_label_classes.log    # expect → 8
```

---

## 🧪 1️⃣ Boot smoke test — 24 GB (~$0, ~2 min) ✅ **DONE 2026-05-09**

> Verified end-to-end: 5 recipe-v2 wiring claims confirmed by GPU execution.
> Evidence log: `logs/m09c_surgery_3stage_DI_sanity.log` (Stage 0 ran head-only at 0/48 blocks, loss=0.5111; Stage 1 ran at 12/48 blocks, loss=0.5072; Stage 2 OOM at 24/48 blocks — expected per 24 GB ceiling).
> Skip this section on future runs unless flag wiring changes.

---

## 🧬 2️⃣ Recipe-v3 POC sweep — 96 GB Blackwell (~$1.60, ~2 hr, 7 cells drop-one)

> ⚠️ Requires RTX Pro 6000 Blackwell (96 GB). 24 GB will OOM at first training batch.
> 🚦 Drop-one ablation across all 5 interventions (#1 frozen teacher · #2 LP-FT · #3 surgical subset · #4 SPD · #5 CLEAR replay) + audit A2/A4. Yields per-intervention contribution = R1 − R(i) for the paper ablation table.

### 🚀 How to run on Blackwell

```bash
tmux new -s iter14_recipe_v3
cd /workspace/factorjepa && source venv_walkindia/bin/activate
```

```bash
# Full 7-cell drop-one ablation (~2 hr · ~$1.60)
# Note: each cell ALSO writes its own log (logs/iter14_poc_recipe_v3_<cell>.log)
# from inside scripts/run_probe_train.sh; the outer tee captures the orchestrator log.
./scripts/run_recipe_v3_sweep.sh 2>&1 | tee logs/iter14_recipe_v3_sweep_orchestrator_v1.log
```

```bash
# OR — just the full recipe-v3 cell (~17 min · ~$0.23) — gate check first
./scripts/run_recipe_v3_sweep.sh R1 2>&1 | tee logs/iter14_recipe_v3_R1_only_orchestrator_v1.log
```

```bash
# OR — specific subset for quick re-run
./scripts/run_recipe_v3_sweep.sh R0 R1 R5 2>&1 \
  | tee logs/iter14_recipe_v3_subset_orchestrator.log
```

### 🔬 Sweep matrix (7 drop-one cells)

| Cell | TEACHER | LPFT | SUBSET    | WARMUP    | SALIENCY | SPD | REPLAY | ⏱️       |
|------|---------|------|-----------|-----------|----------|-----|--------|----------|
| R0_baseline      | EMA    | off | legacy    | per_stage | off | off | off | 13 min   |
| R1_recipe_v3 ⭐ | FROZEN | on  | recipe_v3 | single    | on  | on  | on  | 17 min   |
| R2_minus_frozen  | EMA    | on  | recipe_v3 | single    | on  | on  | on  | 17 min   |
| R3_minus_lpft    | FROZEN | off | recipe_v3 | single    | on  | on  | on  | 16 min   |
| R4_minus_subset  | FROZEN | on  | legacy    | single    | on  | on  | on  | 17 min   |
| R5_minus_spd     | FROZEN | on  | recipe_v3 | single    | on  | off | on  | 17 min   |
| R6_minus_replay  | FROZEN | on  | recipe_v3 | single    | on  | on  | off | 16 min   |

### 📊 Aggregate the 7 cells' trio top-1 trajectories

```bash
# Tail of each cell's per-stage probe-trio readings
for cell in R0_baseline R1_recipe_v3 R2_minus_frozen R3_minus_lpft \
            R4_minus_subset R5_minus_spd R6_minus_replay; do
  echo "=== Cell: $cell ==="
  grep -oE "step=[0-9]+ stage=\S+ N=[0-9]+ top-1=[0-9.]+ motion_cos=[0-9.\-]+" \
    logs/iter14_poc_recipe_v3_${cell}.log 2>/dev/null | tail -5
done 2>&1 \
  | tee logs/iter14_aggregate_trajectories.log
```

```bash
# Best top-1 per cell + drop-one delta vs R1
venv_walkindia/bin/python <<'EOF' 2>&1 \
  | tee logs/iter14_aggregate_drop_one_deltas.log
import re, glob
def best(p):
    txt = open(p).read()
    vals = [float(m) for m in re.findall(r"top-1=([0-9.]+)", txt)]
    return max(vals) if vals else None
results = {}
for f in sorted(glob.glob("logs/iter14_poc_recipe_v3_R*.log")):
    cell = f.split("recipe_v3_")[1].rsplit(".log", 1)[0]
    results[cell] = best(f)
print(f"{'Cell':<22s} best_top1   Δ_vs_R1")
ref = results.get("R1_recipe_v3")
for c, v in results.items():
    delta = "n/a" if (ref is None or v is None) else f"{(v - ref) * 100:+.2f} pp"
    print(f"{c:<22s} {v if v else 'n/a':<10}  {delta}")
EOF
```

### 🚦 Decision rule (per `plan_surgery_wins.md` § 7.5 + § 12.4)

| R1 best trio top-1                          | ➡️ Branch                                                                 |
|---------------------------------------------|---------------------------------------------------------------------------|
| 🟢 ≥ 0.808 AND projected test-Δ ≥ +5 pp     | wire fixes via `plan_no_discrepancy.md` Phases A→D, then 3️⃣ FULL surgery  |
| 🟡 0.81–0.83 (marginal, Δ < +5 pp)          | run Phase 5 FG-feature m04d (`plan_phase5_fg_motion_features.md`)         |
| 🔴 all 7 cells regress (R1 < 0.78)          | Path 2: relax m10 thresholds (data-scale fix, ~$50–60)                    |

---

## 🚀 3️⃣ FULL training — 96 GB (post-POC 🟢, ~37 GPU-h ≈ $30)

> Fires only on 🟢 branch. Uses the SAME env-vars as R1 (recipe-v3 full stack).
> `;` not `&&` — independent failures shouldn't cancel the queue.

```bash
tmux new -s iter14_train
cd /workspace/factorjepa && source venv_walkindia/bin/activate
```

```bash
# 3stage_DI surgery (recipe-v3 full stack)
CACHE_POLICY_ALL=2 \
  TEACHER_MODE_OVERRIDE=FROZEN LP_FT_OVERRIDE=on \
  SUBSET_OVERRIDE=recipe_v3 WARMUP_OVERRIDE=single \
  SALIENCY_OVERRIDE=on SPD_OVERRIDE=on REPLAY_OVERRIDE=on \
  ./scripts/run_probe_train.sh surgery_3stage_DI --FULL 2>&1 \
  | tee logs/iter14_surgery_3stage_DI.log
```

```bash
# noDI ablation (same recipe-v3, different variant)
CACHE_POLICY_ALL=2 \
  TEACHER_MODE_OVERRIDE=FROZEN LP_FT_OVERRIDE=on \
  SUBSET_OVERRIDE=recipe_v3 WARMUP_OVERRIDE=single \
  SALIENCY_OVERRIDE=on SPD_OVERRIDE=on REPLAY_OVERRIDE=on \
  ./scripts/run_probe_train.sh surgery_noDI --FULL 2>&1 \
  | tee logs/iter14_surgery_noDI.log
```

```bash
# pretrain_2X (10 ep compute control — NO factor patching, NO recipe-v3 overrides)
CACHE_POLICY_ALL=2 \
  ./scripts/run_probe_train.sh pretrain_2X --FULL 2>&1 \
  | tee logs/iter14_pretrain_2X.log
```

```bash
# OR — chain all three back-to-back with `;` (each command logs separately)
CACHE_POLICY_ALL=2 \
  TEACHER_MODE_OVERRIDE=FROZEN LP_FT_OVERRIDE=on \
  SUBSET_OVERRIDE=recipe_v3 WARMUP_OVERRIDE=single \
  SALIENCY_OVERRIDE=on SPD_OVERRIDE=on REPLAY_OVERRIDE=on \
  ./scripts/run_probe_train.sh surgery_3stage_DI --FULL 2>&1 \
  | tee logs/iter14_surgery_3stage_DI.log ; \
CACHE_POLICY_ALL=2 \
  TEACHER_MODE_OVERRIDE=FROZEN LP_FT_OVERRIDE=on \
  SUBSET_OVERRIDE=recipe_v3 WARMUP_OVERRIDE=single \
  SALIENCY_OVERRIDE=on SPD_OVERRIDE=on REPLAY_OVERRIDE=on \
  ./scripts/run_probe_train.sh surgery_noDI --FULL 2>&1 \
  | tee logs/iter14_surgery_noDI.log ; \
CACHE_POLICY_ALL=2 \
  ./scripts/run_probe_train.sh pretrain_2X --FULL 2>&1 \
  | tee logs/iter14_pretrain_2X.log
```

---

## 🧪 4️⃣ FULL 5-encoder eval (~4 h, ~$3)

```bash
tmux new -s iter14_eval
cd /workspace/factorjepa && source venv_walkindia/bin/activate
```

```bash
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL 2>&1 \
  | tee logs/iter14_probe_eval.log
```

---

## 🎯 Inspect Δ1/Δ2/Δ3 paper deltas

```bash
jq '.iter14_paper_deltas' outputs/full/probe_action/probe_paired_delta.json 2>&1 \
  | tee logs/iter14_inspect_paper_deltas.log
```

| Delta                            | Pass when                                                |
|----------------------------------|----------------------------------------------------------|
| Δ1 (`pretrain − frozen`)         | `delta_1_pretrain_vs_frozen.ci_lo_pp > 0`                |
| Δ2 (`surgical − pretrain`)       | `delta_2_surgical_vs_pretrain.ci_lo_pp > 0`              |
| Δ3 (`surgical − pretrain_2X`) ⭐ | `delta_3_surgical_vs_pretrain_2X.ci_lo_pp > 0`           |

---

## 🔍 Mid-run diagnostics

```bash
# Probe trajectory of currently-running R1 (or whichever cell)
tail outputs/poc/m09c_surgery_3stage_DI/probe_history.jsonl \
  | jq '{step, probe_top1, motion_cos, future_l1}' 2>&1 \
  | tee logs/iter14_diag_probe_history_tail.log
```

```bash
# SPD optimizer logs (recipe-v3 R1 / R2 / R3 / R4 / R6 — anywhere SPD=on)
grep "Optimizer: SPDAdamW" logs/iter14_poc_recipe_v3_*.log 2>&1 \
  | tee logs/iter14_diag_spd_active_cells.log
```

```bash
# Raw-replay activation (recipe-v3 R1 / R2 / R3 / R4 / R5 — anywhere REPLAY=on)
grep "raw-replay ENABLED" logs/iter14_poc_recipe_v3_*.log 2>&1 \
  | tee logs/iter14_diag_replay_active_cells.log
```

```bash
# Single-warmup activation (recipe-v3 cells with WARMUP=single)
grep -E "warmup [0-9]+" logs/iter14_poc_recipe_v3_*.log | head -20 2>&1 \
  | tee logs/iter14_diag_warmup_active_cells.log
```

```bash
# Drift-loss should be 0 on SPD cells (SPD replaces L2 anchor)
jq -r 'select(.loss_drift != null) | "step=\(.step) jepa=\(.loss_jepa) drift=\(.loss_drift)"' \
  outputs/poc/m09c_surgery_3stage_DI__R1_recipe_v3/loss_log.jsonl | tail -10 2>&1 \
  | tee logs/iter14_diag_drift_loss_R1.log
```

```bash
# Watch all 7 cell logs simultaneously (interactive — no tee, multitail handles its own UI)
multitail -f logs/iter14_poc_recipe_v3_*.log
```

---

## 🛡️ Recipe-v3 env-var reference (canonical)

| Env var                      | Values            | What it overrides                                         |
|------------------------------|-------------------|-----------------------------------------------------------|
| `TEACHER_MODE_OVERRIDE`      | `EMA` / `FROZEN`  | `surgery.teacher_mode`                                    |
| `LP_FT_OVERRIDE`             | `on` / `off`      | `surgery.lp_ft_stage0.enabled`                            |
| `SUBSET_OVERRIDE`            | `legacy` / `recipe_v3` | `surgery.stages[*].unfreeze_below` (legacy=12/24/24, recipe_v3=4/8/8) |
| `WARMUP_OVERRIDE`            | `per_stage` / `single` | `surgery.warmup_mode`                              |
| `SALIENCY_OVERRIDE`          | `on` / `off`      | `optimization.loss.saliency_weighting`                    |
| `SPD_OVERRIDE`               | `on` / `off`      | `optimization.spd.enabled` (alpha_spd=0.05)               |
| `REPLAY_OVERRIDE`            | `on` / `off`      | `replay.raw_pretrain_pct` (on=0.5)                        |
| `CACHE_POLICY_ALL`           | `1` / `2`         | wipe-then-recompute when `2` (else keep cache)            |

All env-vars unset → m09c uses yaml defaults → bit-identical pre-iter14 path (modulo T2 yaml change which moves unfreeze_below from 12/24/24 → 4/8/8 — restore via `SUBSET_OVERRIDE=legacy`).
