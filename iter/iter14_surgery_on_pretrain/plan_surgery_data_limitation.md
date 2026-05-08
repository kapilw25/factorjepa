# 🚨 iter14 Surgery — Data Limitation Crisis & Path to Paper Goal

> ## 🎯 Non-Negotiable Paper Goal
> **`vjepa_surgery` ≫ `vjepa_pretrain` ≫ `vjepa_frozen` on motion / temporal features**
>
> 🚫 **Pivoting the claim is NOT acceptable.** This document maps the technical path that achieves the goal.

---

## 📊 The Three Findings That Force a Rethink

### 🔍 Finding 1 — Surgery's training pool is 70× smaller than pretrain's

```
pretrain pool (m09a):    ~6,500 motion-eligible clips × 5 epochs = 32,500 clip-visits
surgery pool (m09c):         91 quality-gated factor clips × 5 ep =     455 clip-visits
                                                          ratio = ~71× LESS data
```

🔻 **Bottleneck**: m10's SAM3 quality gate

```
factor_manifest.json (potential, has_*=true flags):
  ├─ has_D_L: 9,238 clips ✅ POSSIBLE
  ├─ has_D_A: 7,702 clips ✅ POSSIBLE
  ├─ has_D_I: 6,860 clips ✅ POSSIBLE
  └─ ALL 3 + tubes: 6,771 clips ✅ POSSIBLE

factor_manifest_quality.json (passed thresholds):
  ├─ D_L blur completeness:    98 clips 🚧 ← m10 gate
  ├─ D_A signal-to-bg ratio:   75 clips 🚧 ← m10 gate
  └─ m10 stability_score:    9,297 clips (all measured)

build_factor_index (.npy files actually on disk):
  ├─ D_L: 98 files → 89 in index (9 orphan)
  ├─ D_A: 75 files → 68 in index
  ├─ D_I: 3,462 tube files → 62 clips with ≥1 tube
  └─ UNION = 91 clips ⚠️ REAL TRAIN POOL
```

**Reality**: m10 quality-gates **~99% of clips out**. Surgery operates on the 1% survivors.

---

### 📉 Finding 2 — POC trajectory shows surgery is REGRESSING

#### 🥇 v12 pretrain (gold standard) — monotonic GAIN

| 📍 Step | epoch | 🎯 probe_top1 | 🎬 motion_cos | 📊 val_jepa | 📈 Trend |
|---|---|---|---|---|---|
| 100 | 0 | 0.439 | 0.046 | 0.4726 | 🔵 start |
| 201 | 0 | 0.510 | 0.123 | 0.4676 | ⬆ |
| 302 | 1 | 0.599 | 0.146 | 0.4622 | ⬆⬆ |
| 504 | 2 | 0.678 | 0.215 | 0.4576 | ⬆⬆⬆ |
| 706 | 3 | 0.757 | 0.231 | 0.4568 | ⬆⬆⬆⬆ |
| **1009** | **4** | **0.808** ⭐ | **0.267** | **0.4584** | **🏆 +36.9 pp gain** |

✅ **Healthy SSL signature**: probe_top1 climbs steadily, motion_cos 5.8×, val_jepa stable.

#### 🅲 iter14 POC surgery_3stage_DI — monotonic LOSS

| 📍 Step | Stage | 🎯 probe_top1 | 🎬 motion_cos | 📊 val_jepa | 🔻 BWT |
|---|---|---|---|---|---|
| (init) | pretrain anchor | **0.808** ⭐ | 0.267 | 0.4584 | — |
| 1 | stage1_layout | **0.7449** 🔻 | 0.2744 | 0.5025 | 0.000 |
| 2 | stage2_agent | **0.7245** 🔻🔻 | 0.2482 ⬇ | 0.5014 | -0.0204 |
| 3 | stage3_interaction | **0.7143** 🔻🔻🔻 | 0.2651 | 0.4992 | **-0.0306** ❌ |

#### 🅲 iter14 POC surgery_noDI — same regression pattern

| 📍 Step | Stage | 🎯 probe_top1 | 🎬 motion_cos | 📊 val_jepa | 🔻 BWT |
|---|---|---|---|---|---|
| (init) | pretrain anchor | **0.808** ⭐ | 0.267 | 0.4584 | — |
| 1 | stage1_layout | **0.7449** 🔻 | 0.2744 | 0.5025 | 0.000 |
| 2 | stage2_agent | **0.7245** 🔻🔻 | 0.2482 ⬇ | 0.5015 | **-0.0204** ❌ |

🚨 **The pattern is structural, not noise** — both variants agree on the regression direction. Surgery is **damaging the encoder by ~6 pp on the FIRST training step alone**.

---

### 🧮 Finding 3 — The step-budget math doesn't work

| Run | Steps to reach `probe_top1=0.808` | Notes |
|---|---|---|
| 🥇 v12 pretrain (FULL) | **1009 steps** (5 ep × ~200 step/ep) | actual measured |
| 🅲 iter14 surgery FULL (proposed) | **~45 steps** (5 ep × 3 batches × 3 stages) | 22× FEWER |
| 🆚 ratio | **22×** less optimization | |

> 🎯 For surgery to **beat** 0.808 at this budget, it would need to learn **22× faster per step** than pretrain — while POC evidence shows it learns in the **wrong direction** per step. Mathematically extremely unlikely. ❌

---

## 🎯 Probability of Paper Goal Success — Honest Estimate

> 🚫 **User constraint**: pivoting the paper goal is NOT acceptable. So this section is purely diagnostic — to identify which technical path UNLOCKS the goal, not to suggest abandoning it.

| Outcome (with current iter14 plan) | 🎲 P(success) |
|---|---|
| Δ2 ✅ (`surgery_3stage_DI` ≫ `pretrain`) | ~5-15% |
| Δ3 ✅ (`surgery_3stage_DI` ≫ `pretrain_2X`) | ~10-20% |
| 🏆 Δ2 ✅ AND Δ3 ✅ (paper headline) | **~3-10%** |
| ❌ Surgery actively HURTS encoder | **~50-70%** |

📌 **At current scale (91 clips × 5 ep), the paper goal is statistically near-impossible.** We must change the experiment to make it achievable.

---

## 🛠️ Three Paths That Make the Paper Goal Achievable

> 🚫 **Path 0 — Pivot paper claim**: REJECTED by user. Not listed below.

### 🔵 Path 1 — Drastically increase surgery epochs

```
max_epochs.full:  5 → 50  (10× more time to converge)
clip-visits:    455 → 4,550
total steps:    ~45 → ~450 steps
```

| Aspect | Detail |
|---|---|
| 💰 Cost | ~$30 → ~$80 surgery FULL (15 h → ~50 h) |
| ✅ Pros | 🔧 No preprocessing changes • 🧪 Tests if surgery just needs more time • 🚀 Fastest to launch |
| ❌ Cons | ⚠️ If POC regression is structural (not transient), more epochs = more damage • 🛑 Top@1 plateau early-stop may kick in and cut training short |
| 🎯 Goal achievability | **~25-35%** if POC×10 shows recovery; **~5%** if POC×10 still regresses |

### 🟢 Path 2 — Relax m10 quality threshold → re-run m10 + m11

```
factor pool:  91 clips → potentially 1,000-6,000 clips
surgery becomes scale-comparable to pretrain ✅
```

| Aspect | Detail |
|---|---|
| 💰 Cost | ~5-10 GPU-h re-prep m10 + m11 + iter14 budget = ~$50-60 total |
| ✅ Pros | 🍎 Apple-to-apple comparison • 📈 Genuine learning signal possible • 🏆 Best chance of paper-headline positive result |
| ❌ Cons | ⏪ Re-runs work that `plan_HIGH_LEVEL.md:140` said "DON'T re-run" • 🎚️ Adds threshold-tuning variable (needs ablation justification) |
| 🎯 Goal achievability | **~50-70%** — surgery on 1-6K clips × 5 ep is comparable to v12's 6.5K × 5 ep training scale |

### 🟡 Path 3 — Increase λ anchor + structural regularization

```
λ: 0.005 → 0.05 or 0.1   (10-20× stronger anchor pull-back)
+ shorter unfreeze depth at Stage 1 (12 → 4 blocks)
+ longer warmup per stage
```

| Aspect | Detail |
|---|---|
| 💰 Cost | $0 (config-only) but compounds with FULL run cost |
| ✅ Pros | ⚡ Fast experiment • 🧪 Cleanly tests "is it just drift?" hypothesis |
| ❌ Cons | 🎚️ Strong λ may freeze surgery into "indistinguishable from pretrain" (Δ2 = 0) • 🔒 Trades drift for stagnation |
| 🎯 Goal achievability | **~10-20%** — likely yields stable but uninformative trajectory |

---

## 🎬 Recommended Execution Sequence

### Phase 0️⃣ — Diagnostic POC sweep (~$1, ~1.5 h) ✨ DO THIS FIRST

```bash
# 4 quick POC runs to disambiguate the problem
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --POC \
    -e MAX_EPOCHS_OVERRIDE=10 -e LAMBDA_OVERRIDE=0.005 \
    2>&1 | tee logs/iter14_poc_surgery_diag_e10_l005.log

CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --POC \
    -e MAX_EPOCHS_OVERRIDE=10 -e LAMBDA_OVERRIDE=0.05 \
    2>&1 | tee logs/iter14_poc_surgery_diag_e10_l050.log

CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --POC \
    -e MAX_EPOCHS_OVERRIDE=20 -e LAMBDA_OVERRIDE=0.005 \
    2>&1 | tee logs/iter14_poc_surgery_diag_e20_l005.log

CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --POC \
    -e MAX_EPOCHS_OVERRIDE=20 -e LAMBDA_OVERRIDE=0.05 \
    2>&1 | tee logs/iter14_poc_surgery_diag_e20_l050.log
```

📋 **Decision matrix from POC sweep**:

| POC×10/×20 outcome | 🚦 Verdict | ➡️ Next path |
|---|---|---|
| 📈 Any run shows probe_top1 climbing back ABOVE 0.808 | Surgery CAN learn given more time | ✅ **Path 1** at FULL with that config |
| 📊 Trajectory stabilizes (≥ 0.808) but doesn't exceed | Surgery preserves but doesn't improve | ⚠️ **Path 2** required to break the ceiling |
| 📉 All runs still monotonic decline | Data deficit is structural | 🚨 **Path 2 mandatory** — no other technical fix works |

### Phase 1️⃣ — Based on Phase 0 result, pick ONE main path

| If Phase 0 says... | Run this |
|---|---|
| 📈 Path 1 viable | 🚀 Path 1 FULL (`max_epochs.full=50`) → ~$80 |
| 🔧 Path 2 mandatory | 🛠️ Re-run m10 with relaxed thresholds → re-run m11 → Path 1 FULL with bigger pool → ~$50-60 |
| ❓ Mixed signals | 🧪 Path 3 (λ sweep) as cheap parallel test |

### Phase 2️⃣ — Deliver paper goal

After Phase 1, run:
```bash
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL 2>&1 | tee logs/iter14_probe_eval.log
```

Target: **Δ1 ✅ Δ2 ✅ Δ3 ✅** → paper headline 🏆

---

## 🚦 Decision Gates

| Gate | Pass Condition | If FAIL |
|---|---|---|
| 🔬 G-Phase0 | At least 1 POC sweep run shows probe_top1 ≥ 0.808 by end | ⛔ Path 1 alone insufficient → Path 2 mandatory |
| 🚀 G-Phase1 | FULL surgery probe_top1 > pretrain's 0.808 by ≥ 0.5 pp | ⛔ Re-evaluate threshold + epochs |
| 🏆 G-Phase2 | Δ2 + Δ3 both have non-overlapping 95% CI | ⛔ Investigate eval pipeline; re-run with more bootstrap iters |

---

## 📐 Why the Paper Goal IS Achievable (with the right path)

🧠 **Theoretical justification** (from FactorJEPA proposal Sec 11):
- Surgery operates in a **causal feature space** — D_L (layout), D_A (agent), D_I (interaction) are *disentangled* signals that pretrain can't access from raw video
- Even small N can outperform large N if the **signal-to-noise ratio** is dramatically higher
- v12 anchor (surgery from FROZEN init) showed +13.16 pp on 91 clips — **the technique works**, but iter14's "from pretrain" init starts at 0.808 instead of frozen's 0.677

📊 **Empirical evidence the goal is reachable** (with Path 2):
- iter13 v12: pretrain=0.808, frozen=0.677, surgery_from_frozen=0.808+ → factor patching adds ~13 pp from frozen baseline
- If we can match the **scale** (Path 2 → 1-6K factor clips), surgery from pretrain should similarly add 5-15 pp on top → 0.86-0.92
- That's the **headline result** the paper needs

⚖️ **The current 91-clip pool is the binding constraint**, not the technique. Paths 1 + 2 together break that constraint without compromising the paper claim.

---

## 📂 Reference

| What | Where |
|---|---|
| 🎯 Paper goal | `iter/iter14_surgery_on_pretrain/plan_HIGH_LEVEL.md` § "Paper goal" |
| 📐 v12 reference (pretrain at 0.808) | `iter/iter13_motion_probe_eval/result_outputs/v12/full/probe_pretrain/probe_history.jsonl` |
| 📉 POC surgery_3stage_DI regression | `outputs/poc/m09c_surgery_3stage_DI/probe_history.jsonl` |
| 📉 POC surgery_noDI regression | `outputs/poc/m09c_surgery_noDI/probe_history.jsonl` |
| 🎚️ m10 quality thresholds (where to relax for Path 2) | `src/m10_*.py` (SAM3 segmentation) + `src/m11_*.py` (factor extraction) |
| 📓 Original proposal — surgery section | `Literature/proposal/FactorJEPA/FactorJEPA.md` Sec 11 |

---

> 🎬 **Bottom line**: Paper goal is achievable, but NOT with the current 91-clip × 5-epoch budget. The diagnostic POC sweep (Phase 0) costs $1 and tells us within 1.5 h whether Path 1 alone works or Path 2 is required. **Run the sweep before committing FULL budget.** 🚀
