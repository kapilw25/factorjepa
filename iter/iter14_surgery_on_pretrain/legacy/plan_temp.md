# 📊 iter14 POC Recipe-v2 — 4-Cell Sweep Results & 🚨 Diagnosis

> 🎬 **4 cells**: `{🌀 EMA, 🧊 FROZEN teacher} × {🅰️ LP-FT off, 🅱️ LP-FT on}` + 🅳 D v1 ablation
> 🎯 **Pass gate** (`plan_surgery_wins.md §6.0`): best cell trio top-1 ≥ **0.808** (pretrain anchor)
> 🚨 **Headline**: every cell **regresses** across stages — recipe-v2 (only **2/5 interventions** deployed) is **insufficient**

---

## 🗺️ LEGEND

```
┌─────────┬───────────────┬─────────────┬─────────────────────────────────────────────┐
│  🔠 ID  │  🧊/🌀 Teacher │  🧠 LP-FT  │  🌀 motion_aux                              │
├─────────┼───────────────┼─────────────┼─────────────────────────────────────────────┤
│  🅰️  A  │  🌀 EMA       │  🅰️ off    │  ✅ ON (9272 clips, 8 cls — full POC labels) │
│  🅱️  B  │  🌀 EMA       │  🅱️ on     │  ✅ ON (9272 clips, 8 cls — full POC labels) │
│  🅲  C  │  🧊 FROZEN    │  🅰️ off    │  ✅ ON (9272 clips, 8 cls — full POC labels) │
│  🅳  D₂ │  🧊 FROZEN    │  🅱️ on     │  ⚠️  ON but 855 clips, 7 cls (rm-rf recovery)│
│  🅳  D₁ │  🧊 FROZEN    │  🅱️ on     │  ❌ OFF (silent rm-rf bug — historical)      │
└─────────┴───────────────┴─────────────┴─────────────────────────────────────────────┘
```

---

## 🪜 Table 1 — Per-stage probe trio top-1 trajectory

```
┌───────────────────────────┬─────────────┬─────────────┬───────────────┬──────────────────┐
│  🪜 Stage                 │   🅰️ A     │   🅱️ B     │    🅲 C       │    🅳 D₂         │
├───────────────────────────┼─────────────┼─────────────┼───────────────┼──────────────────┤
│  0️⃣  stage0_head_only    │      —      │  0.7840 ⭐  │       —       │   0.7840 ⭐      │
│  1️⃣  stage1_layout       │   0.7520    │   0.7520    │    0.7520     │   0.7600         │
│  2️⃣  stage2_agent        │   0.7360    │   0.7200    │    0.7360     │   0.7680         │
│  3️⃣  stage3_interaction  │   0.7440    │   0.7680    │    0.7440     │   0.7360 🔻      │
└───────────────────────────┴─────────────┴─────────────┴───────────────┴──────────────────┘
```
🔠 **Markers**: ⭐ best within run (saved as `student_best.pt`) · 🔻 below stage-1
🚨 **Observation**: every cell ends **BELOW** its peak — stages 1→3 are net-destructive

---

## 📊 Table 2 — Top-line metrics across all 4 cells

```
┌──────────────────────────────────┬───────────┬───────────┬───────────┬───────────┬──────────────────────────────┐
│  📊 Metric                       │   🅰️ A   │   🅱️ B   │    🅲 C  │   🅳 D₂  │  🚩 Winner                  │
├──────────────────────────────────┼───────────┼───────────┼───────────┼───────────┼──────────────────────────────┤
│  🥇 Best top-1                   │  0.7520   │  0.7840   │  0.7520   │  0.7840   │  🤝 B = D₂  (tied @ stage 0) │
│  🏁 Final top-1                  │  0.7440   │  0.7680   │  0.7440   │  0.7360   │  🅱️ B                       │
│  🛡️  BWT (final − step1)         │ -0.0080   │ -0.0160   │ -0.0080   │ -0.0480 🔥│  🅰️=🅲 (smallest swing)     │
│  📏 max_drop within run          │  0.0160   │  0.0320   │  0.0160   │  0.0320   │  🅰️ = 🅲                    │
│  🌀 motion_cos best              │  0.2606   │  0.2623   │  0.2606   │  0.2616   │  🅱️ B                       │
│  🌀 motion_cos final stage       │  0.2529   │  0.2561   │  0.2529   │  0.1949 🔥│  🅱️  (D₂ collapses stage 3) │
│  🔮 future_l1 best (lower=bttr)  │  0.5561   │  0.5558   │  0.5563   │  0.5458 ⭐│  🅳 D₂                       │
│  📉 val_jepa best (lower=bttr)   │  0.5000   │  0.4987   │  0.5002   │  0.5004   │  🅱️ B                       │
│  🎬 train loss best              │  0.5054   │  0.4933   │  0.5054   │  0.5023   │  🅱️ B                       │
│  ⏱️  Wall time                   │ 12m 41s   │ 15m 49s   │ 12m 33s   │ 16m 32s   │  🅰️/🅲 (no aux + no LP-FT)  │
│  🎯 ≥ 0.808 gate vs pretrain     │ -5.6 pp 🔴│ -2.4 pp 🔴│ -5.6 pp 🔴│ -2.4 pp 🔴│  ❌ none clear it           │
└──────────────────────────────────┴───────────┴───────────┴───────────┴───────────┴──────────────────────────────┘
```
🔠 **Markers**: ⭐ best across cells · 🔥 worst / severe collapse · 🔴 fails 0.808 gate

---

## 🧪 Table 3 — D v1 (motion_aux **OFF**) vs D v2 (motion_aux **ON**, 855-clip POC) sandbag

```
┌──────────────────────────┬──────────────────┬──────────────────┬────────────────────┐
│  🪜 Stage / Metric       │  🅳 D₁ (no aux)  │  🅳 D₂ (aux 855) │  Δ (v2 − v1)       │
├──────────────────────────┼──────────────────┼──────────────────┼────────────────────┤
│  0️⃣ stage0_head_only    │     0.7840       │     0.7840       │  +0.000            │
│  1️⃣ stage1_layout       │     0.7920 ⭐    │     0.7600       │  −3.20 pp 🔻       │
│  2️⃣ stage2_agent        │     0.7200       │     0.7680       │  +4.80 pp ✅       │
│  3️⃣ stage3_interaction  │     0.7680       │     0.7360       │  −3.20 pp 🔻       │
│  🥇 Best top-1           │     0.7920 ⭐    │     0.7840       │  −0.80 pp          │
│  🛡️  BWT                 │    -0.0160       │    -0.0480       │  worse (3× swing)  │
│  🌀 motion_cos final     │     0.2561       │     0.1949       │  −24% collapse 🔥  │
└──────────────────────────┴──────────────────┴──────────────────┴────────────────────┘
```
⚠️ **Verdict**: 855-clip POC-scale motion_aux is **DESTRUCTIVE** — removes the only **+0.8 pp Stage-1 win** (D₁) seen across all cells. Likely cause: noisy 7-class gradient direction at small N conflicts with frozen-teacher JEPA signal.

---

## 🚦 Table 4 — `0.808` gate decision matrix (per `plan_surgery_wins.md §7.5` decision tree)

```
┌──────────────────────────────────────┬──────────────────────────────┬──────────────────────────────┬──────────────────────────────────────────────┐
│  🚦 Bucket                           │  📏 Threshold               │  🥇 Best result             │  🚩 Verdict                                  │
├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────────────────────────────┤
│  🟢 strict win                       │ ≥ 0.808 + Δ ≥ +5 pp         │ A/B/C/D₂ = 0.7840           │ ❌ FAIL: −2.4 pp short                       │
│  🟢 strict win (incl. D₁ ablation)  │ same                         │ D₁ peak = 0.7920            │ ❌ FAIL: −1.6 pp short                       │
│  🟡 marginal (within-run gain)       │ Stage-1 backbone gain > 0   │ D₁: +0.8 pp; rest flat/neg  │ 🟡 ONLY D₁ shows healthy backbone slope     │
│  🔴 all regress                      │ all cells < 0.808           │ All cells ≤ 0.7920          │ 🔴 TRUE — every cell below pretrain anchor  │
└──────────────────────────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────────────────────────────────┘
```

🚥 **Per `plan_surgery_wins.md §7.5` decision tree → this bucket = 🔴 fall back to Path 2 (data scale)**.
   ⚠️ But that's premature: see § 🚨 below — recipe-v2 is **only 2 of 5 interventions**.

---

# 🚨 §X — Why all 4 cells continue to regress (the real problem)

## 🧬 Root cause analysis (cross-referencing `plan_surgery_wins.md` §2 + §4 + §11.6)

### 🅰️ Cause 1: Only **2 of 5** literature-grounded interventions deployed

```
┌────┬────────────────────────────────────────┬────────────┬───────────────────┬──────────────────────────────────┐
│ #  │  🛠️  Intervention                     │  Status    │  Evidence (grep)  │  Expected effect when deployed   │
├────┼────────────────────────────────────────┼────────────┼───────────────────┼──────────────────────────────────┤
│ 1  │ 🧊 Frozen teacher (SALT)              │ ✅ DEPLOYED│ in C, D — verified│ already in C/D — saw no win 🤔  │
│ 2  │ 🧠 LP-FT Stage 0                      │ ✅ DEPLOYED│ in B, D — verified│ +3.2 pp head-only spike (B,D)   │
│ 3  │ ✂️  Surgical layer subset (4/8 blocks)│ ❌ MISSING │ yaml = 12/24/24   │ blocks gradient blast → less    │
│    │                                        │            │ blocks (NOT 4/8)  │ pretrained-feature destruction  │
│ 4  │ 🛡️  Selective Projection Decay (SPD) │ ❌ MISSING │ 0 grep hits       │ replaces uniform L2 anchor →    │
│    │                                        │            │ across configs/+  │ stops Δ2≈0 trap from L2 anchor  │
│    │                                        │            │ utils/            │                                  │
│ 5  │ 🔁 50% raw-video CLEAR replay         │ ❌ MISSING │ only WITHIN-factor│ injects pretrain-domain signal  │
│    │                                        │            │ mode_mixture used │ to anchor encoder vs forgetting │
│    │                                        │            │ (NOT raw replay)  │                                  │
└────┴────────────────────────────────────────┴────────────┴───────────────────┴──────────────────────────────────┘
```

### 🅱️ Cause 2: **0 of 4** audit fixes from `§11.6` deployed

```
┌────┬─────────────────────────────────────────────┬────────────┬─────────────────────┐
│ #  │  🔧 Audit fix                               │  Status    │  Evidence           │
├────┼─────────────────────────────────────────────┼────────────┼─────────────────────┤
│ A1 │ 📐 Scheduled EMA momentum (cosine 0.998→1) │ ❌ MISSING │ no momentum_schedule│
│    │   (relevant only to EMA cells A/B)         │            │ in utils/training.py│
│ A2 │ 🎯 Saliency-weighted JEPA loss             │ ❌ MISSING │ no cal_loss_mask    │
│ A4 │ 📝 SINGLE warmup over total budget         │ ❌ MISSING │ surgery_base.yaml   │
│    │   (instead of per-stage warmup_pct=0.20)   │            │ still warmup_pct=0.20│
│    │                                             │            │ per-stage           │
│ A5 │ 🧊 Replace L2 anchor with frozen teacher   │ ✅ DEPLOYED│ subsumed by §4 #1   │
└────┴─────────────────────────────────────────────┴────────────┴─────────────────────┘
```

### 🅲 Cause 3: **POC step budget is structurally degenerate**

```
┌──────────────────────────────────────────┬─────────────────┬────────────────────────────┐
│  🪜 Stage budget breakdown (POC)        │  steps          │  warmup steps              │
├──────────────────────────────────────────┼─────────────────┼────────────────────────────┤
│  0️⃣ stage0_head_only (LP-FT)           │  1              │  1 (warmup_pct = 1.0)      │
│  1️⃣ stage1_layout                      │  1              │  1 (warmup_pct = 0.20→1)   │
│  2️⃣ stage2_agent                       │  1              │  1 (warmup_pct = 0.20→1)   │
│  3️⃣ stage3_interaction                 │  1              │  1 (warmup_pct = 0.20→1)   │
│  ───────────────────────────────────────┼─────────────────┼────────────────────────────┤
│  ⚠️  Effective REAL-LR backbone steps   │  ≈ 0            │  every step is in warmup   │
└──────────────────────────────────────────┴─────────────────┴────────────────────────────┘
```
🚨 **At POC, every backbone step is consumed by warmup**. The encoder never sees the configured base LR. The "trajectory" we measure is mostly noise + warmup artifacts — not the recipe's actual behavior.

### 🅳 Cause 4: **Per-stage unfreeze is too aggressive** (cf. `§4 #3`)

```
┌──────────────────────────┬───────────────────┬────────────────────┬─────────────────────────┐
│  🪜 Stage                │  Current (yaml)   │  Recipe-v2 spec    │  Δ                      │
├──────────────────────────┼───────────────────┼────────────────────┼─────────────────────────┤
│  1️⃣ stage1_layout       │  12/48 blocks     │   4/48 blocks     │  3× too many trainable  │
│  2️⃣ stage2_agent        │  24/48 blocks     │   8/48 blocks     │  3× too many trainable  │
│  3️⃣ stage3_interaction  │  24/48 blocks     │   8/48 blocks     │  3× too many trainable  │
└──────────────────────────┴───────────────────┴────────────────────┴─────────────────────────┘
```
🚨 With **24 trainable blocks** on a 48-block ViT-G **and** only 1 backbone step, the gradient blast updates **half the encoder** in one shot. That's the textbook recipe for catastrophic forgetting in a continual-SSL setting. Surgical FT (Lee ICLR'23, `§4 #3`) prescribes ≤4 blocks per stage.

---

## 🔧 §X+1 — Recipe-v3 spec (deploy ALL remaining interventions)

```
┌──────────────────────────────────────┬─────────────────┬─────────────┬─────────────────────────────────────────┐
│  🛠️  Action                          │  💸 LoC        │  ⏱️ Effort  │  📍 File                                │
├──────────────────────────────────────┼─────────────────┼─────────────┼─────────────────────────────────────────┤
│  ✂️  Subset: 0–3 / 0–7 blocks       │  yaml only     │  5 min      │  configs/train/surgery_3stage_DI.yaml   │
│      (unfreeze_below 0.083 / 0.167) │                 │             │                                          │
│  🛡️  SPD optimizer wrapper           │  ~80 LoC       │  2 hrs      │  src/utils/spd_optimizer.py (NEW)       │
│      drop-in replacement for AdamW  │                 │             │  + m09c hook                            │
│  🔁 50% raw-video CLEAR replay      │  ~80 LoC       │  2 hrs      │  src/utils/factor_streaming.py          │
│      (mix raw m11 mp4 50/50 with    │                 │             │  + dataloader factory in m09c           │
│      factor views per step)         │                 │             │                                          │
│  📝 SINGLE warmup over total budget │  ~15 LoC       │  20 min     │  src/utils/training.py                  │
│      (warmup_pct on TOTAL, not      │                 │             │  + surgery_base.yaml                    │
│      per-stage)                     │                 │             │                                          │
│  🎯 Saliency-weighted JEPA loss     │  ~15 LoC       │  30 min     │  src/utils/training.py                  │
│      (port MGMAE cal_loss_mask)     │                 │             │                                          │
│  📐 Scheduled EMA momentum          │  ~20 LoC       │  20 min     │  src/utils/training.py                  │
│      (only relevant to A/B; skip    │                 │             │  (deferred — A/B already lose to C/D₁) │
│      if FROZEN wins)                │                 │             │                                          │
└──────────────────────────────────────┴─────────────────┴─────────────┴─────────────────────────────────────────┘
                                                                                        Total: ~210 LoC, ~6 hrs eng
```

---

## 🚦 §X+2 — Next-step branch decision

```
┌─────┬────────────────────────────────────────────────────┬────────────────────┬───────────────────────────────────────────────┬───────────────────────────────────────────────────┐
│ 🅿️  │  🎯 Option                                         │  💰 Cost           │  ✅ Pros                                       │  ❌ Cons                                          │
├─────┼────────────────────────────────────────────────────┼────────────────────┼───────────────────────────────────────────────┼───────────────────────────────────────────────────┤
│ 🅰️  │ Quick: re-POC w/ **subset (0-3/0-7) + single     │ $0.20, ~10 min     │ 🎯 cheapest test of 2 high-leverage fixes    │ still misses SPD + replay (interventions 4 & 5)  │
│     │  warmup A4** (no SPD/replay yet)                   │                    │   from the 5; no eng overhead                 │                                                    │
│ 🅱️  │ ⭐ **FULL recipe-v3**: deploy all 5 interventions │ ~$1, ~6 hrs eng + │ 🎯 actually tests `plan_surgery_wins.md §4`  │ 6 hrs eng work before next data point            │
│     │  + A2/A4 audit fixes; re-POC                      │ ~90 min POC GPU   │   stack as designed; clean signal             │                                                    │
│ 🅲  │ Skip POC, go straight to **FULL with recipe-v3**  │ ~$80, ~50 GPU-h   │ 🎯 45 backbone steps means warmup is no       │ 🚨 commits big GPU spend before validating fix   │
│     │  (single warmup gives ~36 real-LR steps)          │                    │   longer dominating; real recipe assessment   │   stack at POC scale                              │
│ 🅳  │ 🔴 Fall back to **Path 2** (relax m10 → 91→1-6K  │ $50–60             │ addresses data scale; simple; no eng work     │ 🚨 PREMATURE — recipe is only 2/5 deployed; we   │
│     │  clips)                                            │                    │                                               │   haven't actually tested the recipe argument     │
│ 🅴  │ Phase 5 (FG/BG motion features)                    │ $5–10              │ harder probe → larger surgery gap             │ doesn't fix regression; orthogonal               │
└─────┴────────────────────────────────────────────────────┴────────────────────┴───────────────────────────────────────────────┴───────────────────────────────────────────────────┘
```

🥇 **Recommendation**: 🅱️ FULL recipe-v3 → re-POC.
   📐 **Reasoning**: per `plan_surgery_wins.md §4` table, `P(unblock Δ2)` for #1+#2 stacked alone ≈ 50%; with #3+#4+#5 stacked ≈ **~85%**. We're sitting on a research result that's gated by ~6 hours of engineering. Skipping straight to 🅲 FULL without 🅱️ POC is risky ($80 commitment); falling to 🅳 Path 2 is **premature** (the framework's own §7.5 fork was conditioned on recipe-v2 being **fully deployed**, which it isn't).

   📝 **Defer 🅴 Phase 5 + scheduled EMA A1** until recipe-v3 POC reads.

---

## 🛠️ §X+3 — Concrete execution order for 🅱️ FULL recipe-v3

```
┌─────┬────────────────────────────────────────────────────┬─────────────┬──────────────────────────────────────────────┐
│ 🪜  │  Action                                            │  ⏱️ Effort │  Verification                                 │
├─────┼────────────────────────────────────────────────────┼─────────────┼──────────────────────────────────────────────┤
│ 1️⃣  │ Edit yaml: unfreeze_below 0.083/0.167 (4/8 blocks)│  5 min      │ grep "0.083" surgery_3stage_DI.yaml          │
│ 2️⃣  │ Edit yaml: warmup_pct on TOTAL, not per-stage    │  20 min     │ surgery_base.yaml has total_warmup_pct: 0.10│
│ 3️⃣  │ Implement src/utils/spd_optimizer.py + hook      │  2 hrs      │ unit test: SPD reduces to AdamW when α=0    │
│ 4️⃣  │ Add 50% raw replay via factor_streaming         │  2 hrs      │ batch sample shows 50% raw mp4 + 50% factor │
│ 5️⃣  │ Port cal_loss_mask from MGMAE (saliency loss)   │  30 min     │ grep cal_loss_mask in src/utils/training.py │
│ 6️⃣  │ 3-check gate (py_compile + ast.parse + ruff F,E9)│ auto       │ post-edit-lint.sh hook                       │
│ 7️⃣  │ Re-run 4-cell POC sweep (all 5 deployed)         │  90 min GPU │ logs/iter14_poc_recipe_v3_*.log              │
│ 8️⃣  │ Apply `plan_surgery_wins.md §7.5` decision tree  │  reading    │ 🟢/🟡/🔴 verdict from re-run                 │
└─────┴────────────────────────────────────────────────────┴─────────────┴──────────────────────────────────────────────┘
```

---

> 🎬 **Bottom line**: regression continues because we tested **2/5 of the recipe**. The framework prescribes 5 stacked interventions; we skipped 3. Before falling to data-scale (Path 2), deploy the missing 3 (SPD + CLEAR replay + surgical subset) plus audit A2/A4 — this is **~6 hrs of eng for $1 of GPU**. Don't pivot the experiment until we've actually run the experiment we wrote down.
