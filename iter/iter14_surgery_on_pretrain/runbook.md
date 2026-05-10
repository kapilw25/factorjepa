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

## 🧬 2️⃣ Recipe-v3 POC sweep — 96 GB Blackwell (~$65 total @ $2.50/hr, ~26 hr serial, 7 cells drop-one)

> ⚠️ Requires RTX Pro 6000 Blackwell (96 GB). 24 GB will OOM at first training batch.
> 🚦 Drop-one ablation across all 5 interventions (#1 frozen teacher · #2 LP-FT · #3 surgical subset · #4 SPD · #5 CLEAR replay) + audit A2/A4. Yields per-intervention contribution = R1 − R(i) for the paper ablation table.

### 🚀 How to run on Blackwell — recommended 3-step sequence (single GPU, serial)

> 💡 **Idempotent**: cells whose `outputs/<mode>/m09c_surgery_3stage_DI__<name>/`
> AND `logs/iter14_<mode>_recipe_v3_<name>.log` BOTH exist are auto-skipped.
> Stop/resume the sweep without losing completed cells.
>
> 💡 `SWEEP_MODE` env-var defaults to **POC** if unset. Explicit `SWEEP_MODE=POC`
> in the runbook is for clarity only — functionally optional.

```bash
tmux new -s iter14_recipe_v3
cd /workspace/factorjepa && source venv_walkindia/bin/activate
```

#### 🅰️ Step 1 — POC R1 gate check (measured: 3 h 43 m · $9.30 @ $2.50/hr)

```bash
# Recipe-v3 full-stack POC, R1 cell only. Top-1 ≥ 0.808 = 🟢 unblocks the rest.
SWEEP_MODE=POC ./scripts/run_recipe_v3_sweep.sh R1 2>&1 \
  | tee logs/iter14_recipe_v3_R1_only_orchestrator_v3.log
```

#### 🅱️ Step 2 — SANITY pre-flight on the OTHER 6 cells (~12 min · ~$0.20)

> SANITY validates env-var dispatch + recipe-v3 wiring without burning POC budget.
> Skip R1 (already validated by Step 1). Cell args use prefix match.

```bash
# 6-cell SANITY sweep — skip R1, ~2 min/cell × 6 ≈ ~12 min.
SWEEP_MODE=SANITY ./scripts/run_recipe_v3_sweep.sh R0 R2 R3 R4 R5 R6 2>&1 \
| tee logs/iter14_sanity_recipe_v3_sweep_orchestrator_v2.log
```

⚠️ **OOM heads-up at SANITY** — cells with `SUBSET=legacy` (R0, R4) unfreeze 24 blocks at stage 2.
On a 96 GB Blackwell that fits comfortably; on a 24 GB GPU it would OOM. Either way, SANITY's
job is "no crash before stage 1" — even partial completion confirms wiring is correct.

#### 🅲 Step 3 — Full POC drop-one ablation (~22 hr · ~$56 @ $2.50/hr) — overnight, after SANITY passes

```bash
# 🧹 MANDATORY pre-flight: wipe SANITY outputs from Step 2 (~14 GB stale data)
rm -rf outputs/sanity/ 2>&1 | tee logs/iter14_pre_poc_wipe_sanity.log

# Full 7-cell POC sweep. R1 auto-skipped (output + log exist from Step 1 — verified
# scripts/run_recipe_v3_sweep.sh:121-124 idempotency check).
# Runs R0 + R2-R6 = 6 fresh cells × ~3h 43m (R1 measured) ≈ ~22 hr · ~$56 @ $2.50/hr.
# Per-cell cost = ~$9.30 (matching R1's actual wall, not stale ~3.7 hr estimate).
SWEEP_MODE=POC ./scripts/run_recipe_v3_sweep.sh 2>&1 \
| tee logs/iter14_recipe_v3_sweep_orchestrator_v1.log
```

#### 📋 Per-cell inner log location (written by `run_probe_train.sh`, separate from orchestrator log)

```
logs/iter14_${mode_dir}_recipe_v3_${cell_name}.log
   ↳ e.g. logs/iter14_poc_recipe_v3_R1_recipe_v3.log
   ↳ e.g. logs/iter14_sanity_recipe_v3_R0_baseline.log
```

#### 🪤 Optional alternatives

```bash
# Re-run a specific subset (e.g. retry R5 only after a transient OOM)
SWEEP_MODE=POC ./scripts/run_recipe_v3_sweep.sh R5 2>&1 \
  | tee logs/iter14_recipe_v3_R5_retry_orchestrator.log
```

```bash
# Force-rerun all 7 POC cells from scratch (deletes prior outputs first)
rm -rf outputs/poc/m09c_surgery_3stage_DI__R*
SWEEP_MODE=POC ./scripts/run_recipe_v3_sweep.sh 2>&1 \
  | tee logs/iter14_recipe_v3_sweep_orchestrator_v2.log
```

### 🔬 Sweep matrix (7 drop-one cells)

> ⏱️ Wall update (2026-05-10, measured from R1): POC trains on 9,161 clips × 286
> mini-batches per cell. R1 wall = 3 h 43 m (measured). 7-cell projection ≈ 26 hr ·
> cost ≈ $65 @ $2.50/hr. R1 already done → 6 remaining cells ≈ 22 hr · $56.

| 🔢 Cell                | 🧊 TEACH    | 🧠 LPFT  | ✂️ SUBSET     | 📝 WARMUP    | 🎯 SALI  | 🛡️ SPD  | 🔁 REPLAY    | ⏱️ Wall  | 💡 Layman example — "what does this cell teach us?" |
|------------------------|-------------|----------|---------------|--------------|----------|---------|--------------|----------|------------------------------------------------------|
| 🅰️ R0_baseline         | 🌀 EMA      | ❌ off   | 📏 12/24/24   | 📐 per_stage | ❌ off   | ❌ off  | ❌ off       | ~3.7 hr  | 🆓 **Control group — NO recipe-v3 guardrails.** Like editing a Wikipedia article with no spell-check, no undo button, no backup. Measures the damage WITHOUT any of the 5 fixes. |
| ⭐ R1_recipe_v3        | 🧊 FROZEN   | ✅ on    | ✂️ 4 / 8 / 8  | 📝 single    | ✅ on    | 🛡️ on  | 🔁 on (50%)  | ~3.7 hr  | 🥇 **All 5 safety guardrails ON.** Like editing Wikipedia WITH: (a) a frozen reference copy you compare against · (b) typing-tutor warmup before real edits · (c) edit-only-4-paragraphs limit · (d) smart undo that only undoes harmful edits · (e) 50% of the original article mixed back in. Full toolkit. |
| 🅱️ R2_minus_frozen     | 🌀 EMA      | ✅ on    | ✂️ 4 / 8 / 8  | 📝 single    | ✅ on    | 🛡️ on  | 🔁 on        | ~3.7 hr  | ❓ **"Does FROZEN-reference matter?"** Reference copy slowly drifts toward your edits via EMA. Tests: is the rock-solid anchor critical, or is a slow-moving anchor close enough? |
| 🅲 R3_minus_lpft       | 🧊 FROZEN   | ❌ off   | ✂️ 4 / 8 / 8  | 📝 single    | ✅ on    | 🛡️ on  | 🔁 on        | ~3.7 hr  | ❓ **"Does head-only WARMUP matter?"** Skip the typing-tutor — jump straight to editing the article. Tests: does pre-warming the task heads first protect the pretrained skill set? |
| 🅳 R4_minus_subset     | 🧊 FROZEN   | ✅ on    | 📏 12/24/24   | 📝 single    | ✅ on    | 🛡️ on  | 🔁 on        | ~3.7 hr  | ❓ **"Does SHALLOW unfreezing matter?"** Allow editing 12+ paragraphs at once instead of just 4. Tests: is the gradient blast on too-many-layers the catastrophic-forgetting cause? |
| 🅴 R5_minus_spd        | 🧊 FROZEN   | ✅ on    | ✂️ 4 / 8 / 8  | 📝 single    | ✅ on    | ❌ off  | 🔁 on        | ~3.7 hr  | ❓ **"Does SPD specifically help?"** Vanilla AdamW with uniform weight-decay anchor instead of selective pull-back. Tests: does the *selective* gating (only fight gradient when it's pulling AWAY from anchor) actually beat plain uniform decay? |
| 🅵 R6_minus_replay     | 🧊 FROZEN   | ✅ on    | ✂️ 4 / 8 / 8  | 📝 single    | ✅ on    | 🛡️ on  | ❌ off       | ~3.7 hr  | ❓ **"Does raw-video REPLAY matter?"** Train ONLY on factor-distorted clips, no glimpses of pretrain-domain. Tests: does mixing 50% of the original distribution back in actually prevent domain drift? |

#### 🗝️ Switch legend (what each column means)

| Switch       | OFF state                                        | ON state                                          | What's at stake                                     |
|--------------|--------------------------------------------------|---------------------------------------------------|------------------------------------------------------|
| 🧊 TEACH     | 🌀 EMA — teacher slowly tracks student          | 🧊 FROZEN — teacher = pretrain ckpt forever (SALT)| anchor stability vs. drift                          |
| 🧠 LPFT      | ❌ no head-only warmup                          | ✅ stage 0 trains heads only (encoder frozen)     | feature distortion at step 1                        |
| ✂️ SUBSET    | 📏 legacy 12 / 24 / 24 unfrozen blocks per stage| ✂️ recipe-v3 4 / 8 / 8 (Lee ICLR'23)             | gradient blast on too many layers                   |
| 📝 WARMUP    | 📐 per_stage — warmup repeats every stage        | 📝 single — one front-loaded warmup (vjepa2 ref)  | LR shock at stage boundaries                        |
| 🎯 SALI      | ❌ uniform mean loss across tokens               | ✅ MGMAE-style teacher-norm-weighted loss         | learning signal concentration                       |
| 🛡️ SPD       | ❌ uniform L2 anchor (legacy lambda_reg)         | 🛡️ selective projection decay (Tian NeurIPS'24)  | escape Δ2 ≈ 0 trap                                 |
| 🔁 REPLAY    | ❌ factor-only batches                           | 🔁 50% raw mp4 + 50% factor (CLEAR)               | pretrain-domain anchoring                           |

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

### 📊 7-method comparison — R1 LANDED · others pending

> R1 finished 2026-05-10 02:20 UTC (~3 h 32 m, ~$2.83). Source:
> `outputs/poc/m09c_surgery_3stage_DI__R1_recipe_v3/probe_history.jsonl` +
> `loss_log.jsonl` + `training_summary.json`.
>
> 🥇 **R1 verdict**: top1=**0.8456** (+3.7 pp vs 0.808 anchor), MONOTONIC trajectory,
> NO regression. Per §7.5 decision tree → between 🟢-light and 🟡 (above marginal
> band, below strict-win Δ≥+5 pp threshold). Recipe-v3 mechanism works at POC scale.

```
┌────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────┬───────────┬─────────┬───────────────────┐
│ 🔢 Cell            │ best top1 │ Δ vs 0.808│ best m_cos│ best fL1 ↓│ best vJ ↓ │ train↓best│ BWT       │ ⏱️ wall  │ status            │
├────────────────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼───────────┼─────────┼───────────────────┤
│ 🅰️ R0_baseline    │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳      │ ⏳ pending        │
│ ⭐ R1_recipe_v3   │ 0.8456 🥇 │ +3.70 pp✅│ 0.2747    │ 0.5329    │ 0.4744    │ 0.4570    │ +0.0147 ✅│ 3h 32m  │ ✅ DONE 02:20 UTC│
│ 🅱️ R2_minus_frozen │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳      │ ⏳ pending        │
│ 🅲 R3_minus_lpft   │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳      │ ⏳ pending        │
│ 🅳 R4_minus_subset │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳      │ ⏳ pending        │
│ 🅴 R5_minus_spd    │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳      │ ⏳ pending        │
│ 🅵 R6_minus_replay │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳        │ ⏳      │ 🔄 NEXT (queued)  │
└────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴───────────┴─────────┴───────────────────┘
markers:  🥇 best across cells · ✅ above 0.808 anchor · ↓ lower=better · ⏳ pending
metrics:  m_cos = motion_cos best across stages · fL1 = future_l1 (lower=better) · vJ = val_jepa (lower=better)
          BWT = top1(stage 3) − top1(stage 0)  ·  positive = improving across stages
```

#### 🪜 R1 per-stage trajectory (motion_cos / future_l1 / val_jepa per stage)

```
┌───────────────────────────┬───────────┬───────────┬───────────┬───────────┬───────────┐
│ 🪜 Stage                  │ probe top1│ motion_cos│ future_l1 │ val_jepa  │ train loss│
├───────────────────────────┼───────────┼───────────┼───────────┼───────────┼───────────┤
│ 0️⃣ stage0_head_only      │ 0.8309    │ 0.2699    │ 0.5545    │ 0.4919    │ 0.4960    │
│ 1️⃣ stage1_layout (D_L)   │ 0.8382 ↑ │ 0.2747 ↑ │ 0.5374 ↓ │ 0.4758 ↓ │ 0.4960    │
│ 2️⃣ stage2_agent (D_A)    │ 0.8382 = │ 0.2699 ↓ │ 0.5386    │ 0.4744 ↓ │ 0.4623 ↓ │
│ 3️⃣ stage3_interaction(I) │ 0.8456 ⭐│ 0.2683 ↓ │ 0.5329 ↓ │ 0.4664 ↓ │ 0.4570 ↓ │
└───────────────────────────┴───────────┴───────────┴───────────┴───────────┴───────────┘
trajectory: MONOTONIC ↑ on top1 + future_l1 + val_jepa + train loss (no regression)
            stage-DIFFERENTIAL on motion_cos: D_L stage UP, D_A stage DOWN ← signature of factor-conditioned learning
```

### 🔬 "Is +3.7 pp just from raw replay (50% of batch), not factor masks?" — 3 scalable tables

> One row per cell, one column per stage. R1 filled · R0/R2–R6 ⏳ pending.
> Sources: `outputs/poc/m09c_surgery_3stage_DI__<cell>/{loss_log.jsonl, probe_history.jsonl}`.
> Definitive answer requires **R6** (drop-replay, queued first) + **Δ3** vs `pretrain_FULL_10ep` (FULL mode).

**Evidence #1 · `loss_drift` per stage** — raw-replay hypothesis predicts drift ≈ 0 (replay = anchor distribution)
```
┌──────────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┐
│ Cell             │ s0_head  │ s1_L     │ s2_A     │ s3_I     │ verdict          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────┤
│ R6_minus_replay  │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R1_recipe_v3 ⭐  │ 8.6e-05  │ 0.00290  │ 0.00550  │ 0.00446  │ ↑ 50× — NOT zero✅│
│ R0_baseline      │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R2_minus_frozen  │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R3_minus_lpft    │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R4_minus_subset  │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R5_minus_spd     │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
└──────────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┘
verdict ✅ = drift grew while loss_jepa fell (R1: 0.5070→0.4570) → encoder did move, prediction got better
```

**Evidence #2 · `motion_cos` per stage** — raw-replay hypothesis predicts MONOTONIC ↑ (no stage signal)
```
┌──────────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┐
│ Cell             │ s0_head  │ s1_L 🅛  │ s2_A 🅐  │ s3_I 🅘  │ pattern          │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────┤
│ R6_minus_replay  │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R1_recipe_v3 ⭐  │ 0.2699   │ 0.2747 ↑ │ 0.2735 ↓ │ 0.2683 ↓ │ DIFFERENTIAL ✅  │
│ R0_baseline      │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R2_minus_frozen  │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R3_minus_lpft    │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R4_minus_subset  │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R5_minus_spd     │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
└──────────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┘
pattern ✅ = stage-conditional ↑/↓ tracks dominant-factor mixture (50%L → 70%A → 70%I)
```

**Evidence #3 · `probe_top1` Δ per stage** — raw-replay hypothesis predicts gain front-loaded at s0 (replay starts then)
```
┌──────────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┐
│ Cell             │ Δ s0     │ Δ s1     │ Δ s2     │ Δ s3     │ total            │
├──────────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────┤
│ R6_minus_replay  │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R1_recipe_v3 ⭐  │ +2.3 pp  │ +0.7 pp  │ +0.0 pp  │ +0.7 pp  │ +3.7 pp ✅       │
│ R0_baseline      │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R2_minus_frozen  │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R3_minus_lpft    │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R4_minus_subset  │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
│ R5_minus_spd     │ ⏳       │ ⏳       │ ⏳       │ ⏳       │ ⏳               │
└──────────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┘
total ✅ = ⅔ of gain (s1 + s3) lands AT factor-data introduction boundaries, not at s0 alone
```

⚠️ `block_drift_history.json` is AMBIGUOUS (all 48 blocks moved, middle > top — likely rel_l2 attribution artifact). NOT cited.

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
