# 🏆 iter14 Surgery — Making Surgery Outperform Pretrain & Frozen

> 🎯 **Non-negotiable goal**: `vjepa_surgery` ≫ `vjepa_pretrain` ≫ `vjepa_frozen` on motion / temporal features
> 🚫 No claim pivot. We change the experiment.

---

## 🎬 0. MASTER action items — full sequence (every fix in this file)

> 🗺️ Single-table summary spanning §1–§11. Each row maps to: where it's argued in this file ✚ which `plan_no_discrepancy.md` phase implements it ✚ cost. Read this table first.

| #️⃣ | 🎯 Action | 📍 Section ref | 🚦 plan_no_discrepancy.md gate | 💰 Cost | ⏱️ Effort | 🚥 Status |
|---|---|---|---|---|---|---|
| 1️⃣ | 🧊 **Frozen teacher (SALT)** — replace `teacher = deepcopy(student)` EMA loop with iter13 v12 pretrain encoder, never updated. **Subsumes audit A5.** | §4 #1 + §11.6 A5 | 🔵 **Phase A** — hook contract MUST expose `teacher_mode={EMA, FROZEN}` + `teacher_forward()` separate from `student_forward()` | $0 | ~50 LoC, 1 day | 🆕 NEXT |
| 2️⃣ | 🧠 **LP-FT Stage 0** — head-only warmup (predictor + motion_aux) on FROZEN encoder, 0.5 ep, before backbone unfreeze | §4 #2 | 🔵 **Phase A** — hook supports `head_only_step` (skip backbone grad) | $0 | ~30 LoC + yaml | 🆕 NEXT |
| 3️⃣ | ✂️ **Surgical layer subset** — Stage 1 unfreeze 0–3 (was 0–11), Stage 2/3 unfreeze 0–7 max | §4 #3 | 🔵 **Phase A** — yaml-only inside `surgery.stages[*].unfreeze_below` | $0 | yaml only | 🆕 |
| 4️⃣ | 🛡️ **Selective Projection Decay (SPD)** — drop-in optimizer wrapper replacing uniform L2 anchor | §4 #4 | 🔵 **Phase A** — wrap `build_optimizer` with SPD | $0 | drop-in (~10 LoC) | 🆕 |
| 5️⃣ | 🔁 **50/50 pretrain replay (CLEAR)** — mix 50% raw-video pretrain clips with 50% factor-views per step | §4 #5 | 🔵 **Phase A** — hook supports `aux_data_iter` for replay batches | $0 | ~40 LoC dataloader | 🆕 |
| 6️⃣ | 📐 **Scheduled EMA momentum** — replace fixed τ=0.99925 with cosine schedule (vjepa2 reference) | §11.6 A1 | 🔵 **Phase A** — replace `update_teacher_ema` body | $0 | ~20 LoC | 🆕 |
| 7️⃣ | 🎯 **Saliency-weighted JEPA loss** — port MGMAE's `loss × cal_loss_mask / mask.sum()` weighting | §11.6 A2 | 🔵 **Phase A** — extend `compute_jepa_loss` | $0 | ~15 LoC | 🆕 |
| 8️⃣ | 📝 **Unify warmup across stages** — single warmup over total budget (not per-stage 0.1) | §11.6 A4 | 🔵 **Phase A** — yaml + scheduler factory | $0 | yaml + 5 LoC | 🆕 |
| 9️⃣ | 📚 **Document pixel-mask paradigm in paper §11** — cite Hide-and-Seek + ForAug as nearest analogs to D_L/D_A/D_I | §11.6 A3 | 📖 docs only — independent of phases | $0 | ~1 hour | 🆕 |
| 🔟 | 🧪 **Phase 0 POC sweep** — `{EMA, FROZEN teacher} × {LP-FT y/n}` = 4 runs | §6 | 🔵 **BEFORE Phase A** — POC validates recipe v2 BEFORE locking the hook contract (per §7 step 1–2) | ~$1 | 1.5 GPU-h | 🆕 NEXT |
| 1️⃣1️⃣ | 🏗️ **Refactor m09a + m09c → utils/training_loop.py** with all hooks above wired in | §7 step 3 + plan_no_discrepancy.md | 🔵 **Phases A → B → C → D** (full rollout, ±0.5 pp gates per phase) | — | 1 day eng | ⏸️ blocked on POC |
| 1️⃣2️⃣ | 🧬 **Phase 5 — Harder FG/BG motion features in m04d** (CONDITIONAL) — replaces 13-D summary stats with 23-D FG-camera-subtracted motion → widens surgery-vs-pretrain gap when probe is saturated | §7.5 + [plan_phase5_fg_motion_features.md](./plan_phase5_fg_motion_features.md) | 🟡 **CONDITIONAL** — fires only if recipe v2 lands surgery at 0.81–0.83 (**marginal win**, test-Δ < +5 pp) | ~$5 GPU + $3 re-eval | ~10 LoC + ~57 min m04d + 4 GPU-h re-eval | ⏸️ conditional |
| 1️⃣3️⃣ | 🔧 **Path 2 — relax m10 thresholds** (CONDITIONAL) | §7 step 4 + Path 2 | 🔴 **CONDITIONAL** — fires only if all 4 POC cells regress (recipe-mechanism not the issue → data-scale is) | $50–60 | 5–10 GPU-h | ⏸️ conditional |
| 1️⃣4️⃣ | 🚀 **Path 1 — FULL surgery (50 ep)** | §7 step 5 + Path 1 | 🟢 **AFTER Phase D** with healed recipe (and Phase 5 if it fired) | ~$80 | ~50 GPU-h | ⏸️ |
| 1️⃣5️⃣ | 🏆 **FULL eval — Δ1/Δ2/Δ3 paper deltas** | §7 step 6 | 🟢 **Final** (after #11–#14) | ~$3 | ~4 GPU-h | ⏸️ |

### 🟢 What's already done (don't re-run)

| ✅ Done | Where | Evidence |
|---|---|---|
| Δ1 (`pretrain > frozen`) on 3 metrics | iter13 v12 / v5 | p=0.0 across motion-flow probe top-1, motion_cos, future_mse |
| iter14 implementation (E1–E24, 23 edits) | plan_HIGH_LEVEL.md §🛠️ | 3-check gate green; HF backed up |
| Gold-standard URLs added to docstrings | src/m09a, m09c, m10, m11, m04d, probe_action | verified live 2026-05-09 |
| `src/CLAUDE.md` rule: "Training scripts MUST cite official gold-standard repo URL in docstring" | line 22 | committed |

### 📊 Table reading guide

| Symbol | Meaning |
|---|---|
| 🆕 NEXT | execute on next GPU rental |
| 🆕 | scheduled in next refactor pass |
| ⏸️ | blocked on prior step |
| 🔵 Phase A | shared training-loop refactor (utils/training_loop.py) |
| 🔵 Phase B | m09a migration (gold-standard, low-risk) |
| 🔵 Phase C | m09c migration (with recipe v2 baked in) |
| 🔵 Phase D | end-to-end POC + parity verification |
| 🟡 conditional (FG features) | fires if POC marginally clears 0.808 but test-Δ < +5 pp → Phase 5 |
| 🔴 conditional (data scale) | fires if all 4 POC cells regress → Path 2 (relax m10 thresholds) |
| 🟢 | unblocked after Phase D |

---

## 📊 1. Three findings forcing a rethink

### 🔻 1.1 Data: surgery pool is 70× smaller than pretrain

| Pool | Size | Multiplier vs pretrain |
|---|---|---|
| 🥇 m09a pretrain (motion-eligible × 5 ep) | ~6,500 clips × 5 = **32,500 clip-visits** | 1× |
| 🅲 m09c surgery (m10-quality-gated × 5 ep) | 91 clips × 5 = **455 clip-visits** | **0.014× (≈70× LESS)** |

### 🔻 1.2 m10 SAM3 quality gate funnel

| Source | Count | Status |
|---|---|---|
| `factor_manifest.json` `has_D_L=true` | 9,238 clips | ✅ possible |
| `factor_manifest.json` `has_D_A=true` | 7,702 clips | ✅ possible |
| `factor_manifest.json` `has_D_I=true` | 6,860 clips | ✅ possible |
| `factor_manifest.json` ALL 3 + tubes | 6,771 clips | ✅ possible |
| 🚧 **D_L blur completeness gate** | **98 clips** | ❌ funnel |
| 🚧 **D_A signal-to-bg gate** | **75 clips** | ❌ funnel |
| `m10 stability_score` measured | 9,297 clips | ✅ |
| Disk: D_L `.npy` files | 98 → 89 indexed | — |
| Disk: D_A `.npy` files | 75 → 68 indexed | — |
| Disk: D_I tube files | 3,462 → 62 clips with ≥1 tube | — |
| 🎯 **UNION on disk = REAL train pool** | **91 clips** | 🚨 binding constraint |

### 🔻 1.3 POC trajectory — surgery monotonically REGRESSES

| Run | init | step 1 | step 2 | step 3 | BWT |
|---|---|---|---|---|---|
| 🥇 v12 pretrain (gold) | n/a | 0.439 | 0.510 | 0.599→**0.808@1009** | **+36.9 pp** ✅ |
| 🅲 surgery_3stage_DI POC | **0.808** ⭐ | 0.7449 🔻 | 0.7245 🔻🔻 | 0.7143 🔻🔻🔻 | **−3.06 pp** ❌ |
| 🅲 surgery_noDI POC | **0.808** ⭐ | 0.7449 🔻 | 0.7245 🔻🔻 | (no stage 3) | **−2.04 pp** ❌ |

### 🔻 1.4 Step-budget math

| Run | Steps to 0.808 | Notes |
|---|---|---|
| 🥇 v12 pretrain FULL | 1009 | 5 ep × ~200 steps |
| 🅲 iter14 surgery FULL (proposed) | ~45 | 5 ep × 3 batches × 3 stages |
| 🆚 ratio | **22× FEWER** | per-step direction is currently negative |

---

## 🧬 2. Mechanism diagnosis (recipe, NOT just data)

> ⚠️ A −6 pp drop on **step 1** (32 clips × 1 batch) cannot be explained by "70× less data". After ONE optimizer update, you've barely seen anything. **The recipe is the bottleneck.**

| 🧬 Mechanism | 📘 Source | 💥 How it produces step-1 drop |
|---|---|---|
| Feature distortion (LP-FT) | Kumar et al., **ICLR'22** ([arXiv:2202.10054](https://arxiv.org/abs/2202.10054)) | Untrained head → large losses → backbone gradient distorts pretrained features |
| EMA teacher decay | Apple **SALT '25** ([arXiv:2509.24317](https://arxiv.org/abs/2509.24317)) | `teacher = deepcopy(student)` → teacher tracks regressed student → loss target decays |
| Foundation-model "concept forgetting" | Mukhoti / CVPR'24 PEFT-ViT ([arXiv:2404.17245](https://arxiv.org/abs/2404.17245)) | DINO ViT-B/16 loses 70% ImageNet acc in 10 fine-tune iters |
| Sharp init (0.808 peak) + tiny step budget | LayerLock '25 ([arXiv:2509.10156](https://arxiv.org/html/2509.10156)) | Any direction = downhill from a peak; recovery requires N steps that don't exist |

### 🅰️ Why m09a/v12 worked despite the same primitives

| Aspect | 🅰️ m09a v12 (works) | 🅲 m09c POC (regresses) |
|---|---|---|
| Total optimizer steps | 1,010 | 3 (POC) / ~45 (FULL) |
| Effective warmup (10%) | ~100 steps ✅ | 0.3 step / ~4.5 step ❌ |
| Trainable blocks | 28/48 (`[20,48)`) | 12→24/48 |
| Starting point | 🟢 flat (Meta init) | 🔴 sharp peak (0.808) |
| Recovery budget after first-step dip | 1009 steps | 2–44 steps |
| Aux gradient (motion_aux) | 9,276 clips (strong) | 91 clips (noisy) |
| Loss landscape on inputs | raw video (intact info) | factor-views (info-destructive blur/suppress) |
| Outcome | 🥇 +36.9 pp | ❌ −3.06 pp |

🟡 **The honest read**: warmup itself isn't THE issue (m09a's 100-step warmup was real). The deeper issue is **sharp-init + tiny-budget + EMA-decay + info-destructive views** — and the literature interventions still apply.

---

## 🎯 3. P(paper goal) — current vs recipe-fixed

| Outcome | Current iter14 plan (Path 1/2/3) | Recipe v2 (frozen teacher + LP-FT + SPD + replay) |
|---|---|---|
| Δ2 ✅ (`surgery ≫ pretrain`) | ~5–15% | **30–50%** |
| Δ3 ✅ (`surgery ≫ pretrain_2X`) | ~10–20% | **35–55%** |
| 🏆 Δ2 ✅ AND Δ3 ✅ (headline) | **~3–10%** | **~25–40%** |
| ❌ Surgery actively HURTS encoder | ~50–70% | <10% |

---

## 🛠️ 4. Five literature-grounded interventions (orthogonal to data scale)

| # | 🛠️ Intervention | 📘 Source | 💸 Cost (LoC) | 🎲 P(unblock Δ2) | 🔗 Repo |
|---|---|---|---|---|---|
| 🥇 1 | **Frozen teacher (SALT)** — `teacher` = v12 pretrain encoder, never EMA-updated | Apple [arXiv:2509.24317](https://arxiv.org/abs/2509.24317) | ~50 LoC | **40–60%** ⭐ | (paper-only as of '25) |
| 🥈 2 | **LP-FT Stage 0** — head-only warmup before backbone unfreeze | Kumar [arXiv:2202.10054](https://arxiv.org/abs/2202.10054) | ~30 LoC + yaml | 30–50% standalone, 70%+ stacked | [AnanyaKumar/transfer_learning](https://github.com/AnanyaKumar/transfer_learning) |
| 🥉 3 | **Surgical layer subset** — Stage 1 unfreeze 4 blocks, not 12 | Lee [arXiv:2210.11466](https://arxiv.org/abs/2210.11466) | yaml-only | 20–30% standalone | — |
| 🏅 4 | **Selective Projection Decay** — replaces uniform L2 anchor | Tian [arXiv:2411.01713](https://arxiv.org/abs/2411.01713) | drop-in optim wrapper | modest, reliable stacked | [GT-RIPL/SPD](https://github.com/gt-ripl/selective-projection-decay) |
| 🎖️ 5 | **Pretrain-domain replay (CLEAR)** — 50% raw-video pretrain clips per step | Rolnick NeurIPS'18 + [arXiv:2305.13622](https://arxiv.org/html/2305.13622v2) | ~40 LoC dataloader | 25–35% standalone | — |

---

## 🧪 5. Recommended stacked recipe (no Path 2 re-prep needed)

| Stage | Trainable blocks | Steps allocation | LR | Mixture | Notes |
|---|---|---|---|---|---|
| 0️⃣ head-only (LP-FT) | 0/48 (encoder FROZEN) | 0.5 ep | 5e-4 (predictor + motion_aux only) | factor + raw mix | 🆕 fixes step-1 distortion |
| 1️⃣ stage1_layout | 0–3 (4/48) | 1.5 ep | base 1e-5, LLRD 0.9 | {L:1.0} | was 12 blocks → now 4 |
| 2️⃣ stage2_agent | 0–7 (8/48) | 1.5 ep | base 1e-5, LLRD 0.9 | {L:0.3, A:0.7} | was 24 blocks → now 8 |
| 3️⃣ stage3_interaction | 0–7 (8/48) | 1.5 ep | base 1e-5, LLRD 0.9 | {L:0.15, A:0.15, I:0.7} | was 24 blocks → now 8 |

| Cross-cutting setting | Old (POC v3) | New (Recipe v2) | Source |
|---|---|---|---|
| Teacher | EMA `deepcopy(student)`, τ=0.99925 | **FROZEN v12 pretrain encoder** | SALT |
| L2 anchor λ | 0.005 uniform | **DROPPED** (frozen teacher IS the anchor) + **SPD** for weight decay | SALT + SPD |
| Replay | within-factor 30% D_L | **50% raw-video pretrain clips** | CLEAR |
| Warmup_pct | 0.1 of stage steps | 0.20 of stage steps | LLRD |
| EMA | on student weights only | unchanged (probe stability) | — |

---

## 🚦 6. Phase-0 POC sweep — re-spec (compete with the plan's λ × epochs sweep)

### 📢 6.0 Research-group share-card (single table)

| 🔖 Aspect | 🎯 Detail |
|---|---|
| 🧪 **Recipe fix #1** | 🧊 **Frozen teacher (SALT)** — anchor encoder to v12 pretrain weights, no EMA drift |
| 🧪 **Recipe fix #2** | 🧠 **LP-FT** — warm up task heads on frozen backbone before unfreezing (V-JEPA / fine-tuning literature) |
| 🔬 **POC sweep** | 4 runs · `{🌀 EMA, 🧊 FROZEN}` × `{🅰️ LP-FT off, 🅱️ LP-FT on}` |
| 💰 **Cost** | ~**$1** / ~**90 min** GPU |
| 🎯 **Pass gate** | 🥇 any cell reaches **top-1 ≥ 0.808** (pretrain baseline) |
| ✅ **If pass** | 🔧 wire both fixes into shared training loop, then scale to FULL surgery |
| ⚠️ **If all fail** | 🚧 bottleneck is data scale — only **91 SAM-quality-gated factor clips** today → relax SAM thresholds to grow the pool |

### 🧮 6.1 Sweep axes (full spec)

| Sweep axis A | Sweep axis B | Why this axis | Cost |
|---|---|---|---|
| `teacher ∈ {EMA, FROZEN}` | `lp_ft ∈ {off, 0.5 ep on}` | Tests SALT + LP-FT directly | $1, ~1.5 h, 4 runs |

| 🚦 POC outcome | ➡️ Verdict |
|---|---|
| 📈 FROZEN+LP_FT climbs ≥ 0.808 by step 5 | ✅ Recipe v2 unblocks → go FULL |
| 📈 FROZEN-only climbs but LP_FT-only doesn't | ✅ SALT alone sufficient |
| 📉 Both still regress | 🚨 fall back to plan's Path 2 (relax m10 thresholds) |

---

## 🚀 7. Execution order (overrides plan's "ready to execute")

| 🪜 Step | Action | Cost | Decision |
|---|---|---|---|
| 1️⃣ | POC: `frozen_teacher: true` + LP-FT Stage 0 + LLRD 0.9 in `surgery_3stage_DI.yaml` | $0 (~80 LoC) | If trio top-1 ≥ 0.808 in first 3 steps → ✅ go FULL |
| 2️⃣ | Add 50/50 pretrain replay; re-POC | +$1, 1.5 h | Stacks with #1 |
| 3️⃣ | Refactor m09a/m09c via `plan_no_discrepancy.md` (**only after** hook contract is informed by POC) | 1 day eng | Phases A→B→C→D, ±0.5 pp gate |
| 4️⃣ | Path 2 (relax m10 → 1–6K clips) | $50–60 | Only if recipe v2 still regresses |
| 5️⃣ | Path 1 (50 ep) | $80, 50 h | Only with healed recipe |
| 6️⃣ | FULL eval, Δ1/Δ2/Δ3 | $3 | 🏆 paper headline |

---

## 🌳 7.5 Phase-5 conditional fork — decision tree (no paralysis, just causality)

> 🎯 Where [`plan_phase5_fg_motion_features.md`](./plan_phase5_fg_motion_features.md) gates relative to recipe v2. Phase 5 is **NOT** a parallel track — it's a follow-up that fires only when recipe v2 reveals the probe metric is saturated.

| 🪜 | Action | 💰 Cost | 🚦 Decision rule |
|---|---|---|---|
| 1️⃣ | Run **recipe v2 POC sweep** (4 cells: `{🌀 EMA, 🧊 FROZEN} × {🅰️ LP-FT off, 🅱️ LP-FT on}`) | $1, 90 min | If best cell trio top-1 ≥ 0.808 **AND** projected test-Δ ≥ +5 pp → ✅ wire fixes; **STOP** (no Phase 5 needed) |
| 2️⃣ | If recipe v2 lands surgery at **0.81–0.83** (marginal win, test-Δ < +5 pp) | — | 🟡 **Now Phase 5's gate fires** → run FG-feature m04d (~57 min) + Stage 1 re-label + re-eval (~$5 + $3) → re-test surgery on harder probe |
| 3️⃣ | If recipe v2 **fully fails** (all 4 cells regress) | — | 🔴 recipe-mechanism is **not** the issue → fall back to **Path 2** (relax m10 thresholds, grow factor pool 91 → 1–6K clips) |

### 🌐 Decision flow visual

```
                    Run recipe v2 POC ($1 / 90 min)
                                │
       ┌────────────────────────┼────────────────────────┐
       ▼                        ▼                        ▼
  🟢 top-1 ≥ 0.808       🟡 top-1 0.81–0.83       🔴 all 4 regress
  AND Δ ≥ +5 pp          (marginal, Δ < +5 pp)    (recipe doesn't help)
       │                        │                        │
       ▼                        ▼                        ▼
  ✅ Wire fixes →         🧬 Phase 5: harder        🔧 Path 2: relax
   refactor → FULL          FG/BG features          m10 thresholds
                            (~$8, ~6 GPU-h)         (~$50–60)
       │                        │                        │
       └────────────────────────┴────────────────────────┘
                                ▼
                         🏆 FULL eval Δ1/Δ2/Δ3
```

---

## 🛑 8. Decision gates

| Gate | Pass condition | If FAIL |
|---|---|---|
| 🔬 G-Phase0 | Any sweep cell shows trio top-1 ≥ 0.808 | ⛔ recipe insufficient → 🔴 Path 2 (data-scale fix) |
| 🧬 G-Phase5 (NEW) | If G-Phase0 passes but **projected test-Δ < +5 pp**, then after FG-feature re-eval surgery test-Δ ≥ +5 pp | ⛔ both metric saturation AND recipe insufficient → revisit Path 2 OR pivot to Path 1 (50 ep) on harder features |
| 🔁 G-Refactor | m09a POC top-1 ∈ [0.4535, 0.4635] AND m09c POC ∈ [0.6949, 0.7949] | ⛔ git revert, investigate hook |
| 🚀 G-Phase1 | FULL surgery probe_top1 > 0.808 + 0.5 pp (or post-Phase-5 equivalent on harder probe) | ⛔ re-evaluate threshold + epochs |
| 🏆 G-Phase2 | Δ2 + Δ3 BCa 95% CIs both non-overlapping | ⛔ re-bootstrap; check eval pipeline |

---

## 📌 9. Where the original plan framing is incomplete

| Plan claim | ✅ True | ❌ Incomplete because |
|---|---|---|
| "Both DI and noDI agree → structural, not noise" | structural ✅ | structure is **recipe** (EMA decay + sharp init), not "data deficit" |
| "22× fewer steps → mathematically near-impossible" | given current recipe ✅ | SALT recipes converge in ↓ student steps; Apple's scaling curves dominate V-JEPA's |
| "Path 3 (λ↑) freezes surgery into pretrain-equivalent" | for **uniform** L2 ✅ | **SPD** (NeurIPS'24) is selective — escapes Δ2≈0 trap |
| Anti-forget table marks LLRD/LR-cap as "🟡 partial" | factually correct | LLRD is **prerequisite** in ULMFiT canon, not nice-to-have |

---

## 🔍 10. Gold-standard repo registry (verified live 2026-05-09)

| 🅼 Module | 🥇 Gold-standard repo(s) | 🟢 What it covers |
|---|---|---|
| `m09c_surgery.py` | 1️⃣ [facebookresearch/vjepa2 / app/vjepa_2_1/train.py](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/train.py)<br>2️⃣ [MCG-NJU/MGMAE](https://github.com/MCG-NJU/MGMAE) (ICCV'23)<br>3️⃣ [MCG-NJU/VideoMAE](https://github.com/MCG-NJU/VideoMAE) (NeurIPS'22) | Training-loop primitives + mask-conditioned video SSL paradigm + foundational video MAE |
| `m10_sam_segment.py` | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) + [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | 4-anchor DINO + SAM3 video tracking |
| `m11_factor_datasets.py` | Closest analog: [ForAug](https://arxiv.org/html/2503.09399) (foreground/background recombine via masks) + [Hide-and-Seek](https://arxiv.org/abs/1811.02545) (mask-driven pixel manipulation) | The pixel-augmentation pattern; {D_L, D_A, D_I} taxonomy is FactorJEPA-novel |
| `m04d_motion_features.py` | [pytorch/vision/.../optical_flow/raft.py](https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py) | RAFT optical-flow inference |
| `m09a_pretrain.py` | [facebookresearch/vjepa2 / app/vjepa_2_1/train.py](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/train.py) | Continual SSL training loop |
| `probe_action.py` | [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) (`configs/eval/vitg-384/ssv2.yaml`) | Attentive-classifier probe |

🆕 **Codified in `src/CLAUDE.md` § CODE STANDARDS**: *"Training scripts MUST cite official gold-standard repo URL in docstring."* (9 words, added 2026-05-09)

---

## 🧪 11. Audit m09c against the 3 gold-standards (verification matrix)

> 🎯 **Purpose**: structural sanity-check before / during / after iter14 implementation. Each row is a concrete claim the user (or Claude Code) can verify by reading the cited file.

### 🅰️ 11.1 Repo-structure parity

| Concept | 🅲 m09c (ours) | 🅶¹ vjepa2 train.py | 🅶² MGMAE | 🅶³ VideoMAE |
|---|---|---|---|---|
| Training entry-point | `src/m09c_surgery.py` (1717 LoC) | `app/vjepa_2_1/train.py` (~835 LoC) | `run_mgmae_pretraining.py` | `run_mae_pretraining.py` |
| Training loop body | `src/utils/training.py:_train_step_grad_accum` | inline in train.py | `engine_for_mgmae.py:train_one_epoch` | `engine_for_pretraining.py:train_one_epoch` |
| Dataset / loader | `src/utils/factor_streaming.py` + `FactorSampler`/`StreamingFactorDataset` | `app/vjepa_2_1/wrappers.py` | `dataset/` + `flow_utils/` | `datasets.py` + `kinetics.py`/`ssv2.py` |
| Masking module | n/a — masks live in **pixels** (m11) | tube-mask scheduler in `transforms.py` | `engine_for_mgmae.py:get_build_mask_volume_func` | `masking_generator.py` |
| Model architecture | `vjepa2_imports.get_vit_by_arch` | `app/vjepa_2_1/models/` | `models/` | `modeling_pretrain.py` |
| Optimizer factory | `utils/training.build_optimizer` | inline `optim_factory` style | `optim_factory.py` | `optim_factory.py` |

### 🅱️ 11.2 Training-loop primitives — vs `vjepa2/app/vjepa_2_1/train.py`

| Primitive | 🅲 m09c (ours) | 🅶¹ vjepa2 ref | ✅ match / ⚠️ diverge |
|---|---|---|---|
| Distributed setup | not used (single-GPU) | `init_distributed`, DDP | ⚠️ ours single-GPU only |
| Optimizer | AdamW (utils.training.build_optimizer) | AdamW | ✅ |
| Mixed precision | bf16 autocast + scaler | bf16/fp16 autocast | ✅ |
| EMA target encoder | `update_teacher_ema(τ=0.99925)` | momentum schedule (cosine 0.998 → 1.0) | ⚠️ ours uses fixed τ; vjepa2 uses **scheduled** momentum |
| Loss function | `compute_jepa_loss` = SmoothL1 (loss_exp=1) on predictor↔teacher latents at masked positions | same SmoothL1 latent prediction | ✅ |
| Predict-all flag | `cfg["model"]["predict_all"]=True` (Dense Predictive Loss) | V-JEPA 2.1 = same (DPL) | ✅ |
| Deep supervision | `n_output_distillation=4` | V-JEPA 2.1 = 4 levels | ✅ |
| Drift L2 anchor | `compute_drift_loss(student, init_params, λ=0.005)` | **NOT present** in vanilla vjepa2 | 🆕 ours-only (from continual-SSL literature) |
| Motion-aux head | 8-cls CE + 13-D MSE on RAFT features (weight=0.1) | not present | 🆕 ours-only (from iter12 v3) |
| Multi-task losses (InfoNCE/TCC) | gated off by yaml; available in code | not present | 🆕 ours-only (gated off in iter14) |
| Gradient accumulation | `_train_step_grad_accum` micro-batches | yes via `accum_iter` | ✅ |
| Adaptive batch sizer | `gpu_batch.AdaptiveBatchSizer` (OOM recovery) | not present | 🆕 ours-only |

### 🅲 11.3 Mask-conditioning paradigm — vs MGMAE (ICCV 2023)

| Aspect | 🅶² MGMAE | 🅲 m09c (ours) | 🚨 KEY divergence |
|---|---|---|---|
| Mask source | optical-flow motion volume (computed from RAFT) | SAM3 segmentation (m10) | external signal in both — **paradigm match ✅** |
| Mask generation timing | **on-the-fly per batch** (warps base mask via flow) | **pre-baked offline** by m11 (D_L/D_A/D_I .npy) OR streaming (factor_streaming.stream_factor) | both options exist |
| Mask APPLICATION POINT | **TOKEN level** — masks decide which tokens are visible to predictor | **PIXEL level** — masks decide which pixels get blurred / suppressed in the input video | 🚨 **fundamentally different mechanism** |
| Loss function | MSE on masked patches (pixels) | SmoothL1 on masked latents (V-JEPA style) | different |
| What the model "sees" | unmodified video tokens, but only the unmasked subset | factor-modified pixels (blurred / suppressed), all tokens visible | 🚨 ours = data augmentation; MGMAE = objective masking |
| Loss weighting by mask | yes (loss × motion_volume / mask_sum) | uniform (no per-pixel weighting) | ⚠️ ours could borrow this |
| Empirical takeaway from MGMAE | motion-volume mask + standard MAE → SOTA on K400/SSv2 | factor-pixel-mod + V-JEPA → POC regresses | ours' divergence may be the bug |

> 🚨 **The audit's biggest finding**: m09c's "mask-conditioning" is closer to **input data augmentation** (pixel manipulation) than to **objective masking** (token visibility). The literature for that exact pattern is **Hide-and-Seek + ForAug**, NOT MGMAE/VideoMAE. m09c hybridizes the two without explicit prior. **This is the load-bearing structural choice that has no canonical reproduction target.**

### 🅳 11.4 Foundational video masked SSL — vs VideoMAE (NeurIPS 2022)

| Aspect | 🅶³ VideoMAE | 🅲 m09c (ours) | Notes |
|---|---|---|---|
| Loss | MSE on masked **pixel patches** (reconstruct pixels) | SmoothL1 on masked **latents** (predict embeddings) | ours is JEPA-flavored, not MAE |
| Masking ratio | **90–95 %** | factor-views modify ~30–60 % of pixels (blur strength, matte_factor, crop) | not directly comparable |
| Mask shape | **tube** (same spatial mask across all frames) | per-frame (T, H, W) from SAM tracking | OK — tracking provides temporal coherence |
| Pretraining target | reconstruct masked pixels | predict teacher's latents at masked positions | V-JEPA paradigm |
| Decoder | yes (small) | no (predictor head only, like vjepa2) | match V-JEPA |

### 🅴 11.5 Concrete verification checklist (can be executed in 1 hour)

| # | Action | File to read / command | Expected finding |
|---|---|---|---|
| 1 | Confirm SmoothL1 vs MSE | `grep -n "smooth_l1_loss\|mse_loss\|F\.l1_loss" src/utils/training.py` | smooth_l1_loss (V-JEPA standard) — **diverges from MGMAE/VideoMAE MSE** |
| 2 | Confirm fixed vs scheduled EMA | `grep -nE "ema_momentum\|momentum_scheduler" src/utils/training.py configs/train/surgery_base.yaml` | fixed τ=0.99925 — vjepa2 uses cosine schedule → consider porting |
| 3 | Confirm masks-on-pixels (not tokens) | Read `src/utils/factor_streaming.py:stream_factor` lines 75–145 | Confirms `make_layout_only` blurs pixels via Gaussian; no token-level mask passed to predictor |
| 4 | Find optical-flow loss-weighting | grep `cal_loss_mask\|mask_volume` in `src/utils/training.py` | NOT present — opportunity to port from MGMAE |
| 5 | Compare drift-loss to vjepa2 | WebFetch vjepa2/app/vjepa_2_1/train.py and search "drift\|anchor\|reg_loss" | NOT present in vjepa2 → ours adds this from continual-SSL literature |
| 6 | Confirm motion_aux is m09c-only or shared with m09a | `grep -n "motion_aux" src/m09a_pretrain.py src/m09c_surgery.py` | Both have it (iter12 v3) — verify hyperparams match |
| 7 | Verify `predict_all=True` (Dense Predictive Loss) | `grep -nE "predict_all" configs/model/vjepa2_1.yaml configs/train/surgery_base.yaml` | Should be true (matches V-JEPA 2.1 paper) |
| 8 | Compare per-stage warmup vs vjepa2 | `grep -nE "warmup_pct\|surgery.*warmup" configs/train/surgery_base.yaml configs/train/surgery_3stage_DI.yaml` | Per-stage warmup_pct=0.1 — vjepa2 uses one continuous warmup |

### 🅵 11.6 Issues surfaced by the audit (action items)

| # | Issue | Fix-to (cited gold-standard) | Effort |
|---|---|---|---|
| A1 | Fixed EMA momentum (τ=0.99925) instead of scheduled | port `momentum_schedule` from vjepa2 train.py | 20 LoC |
| A2 | No loss-weighting by mask saliency | port `loss × cal_loss_mask` pattern from MGMAE engine | 15 LoC |
| A3 | Pixel-level mask conditioning has no exact prior — **document this as a design choice in the paper** with Hide-and-Seek + ForAug as nearest analogs | Citation in paper §11 | docs only |
| A4 | Per-stage warmup is *much* shorter than gold-standard continuous warmup → first-step shock | use **single** warmup over the entire 3-stage budget (Section 7's recipe-v2) | yaml only |
| A5 | Drift L2 anchor is project-local (no vjepa2/MGMAE/VideoMAE prior) | replace with **frozen pretrain teacher (SALT)** — Section 4 #1 | ~50 LoC |

### 🅶 11.7 What to KEEP (validated by audit)

| Element | Validation source |
|---|---|
| ✅ V-JEPA 2.1 architecture (vit_gigantic_xformers, 48 blocks, predict_all, deep supervision n=4) | matches vjepa2 |
| ✅ SmoothL1 latent prediction loss | matches vjepa2 (V-JEPA paper) |
| ✅ EMA target encoder | matches vjepa2 (modulo schedule) |
| ✅ AdamW + bf16 autocast | matches vjepa2 |
| ✅ Gradient accumulation across micro-batches | matches vjepa2 (`accum_iter`) |
| ✅ DINO + SAM3 4-anchor pipeline (m10) | matches Grounded-SAM-2 tracking demo |

---

## 📚 Sources (compact)

| Topic | Link |
|---|---|
| SALT (frozen teacher) | https://arxiv.org/abs/2509.24317 · https://machinelearning.apple.com/research/rethinking-jepa |
| LP-FT | https://arxiv.org/abs/2202.10054 · https://github.com/AnanyaKumar/transfer_learning |
| Surgical Fine-Tuning | https://arxiv.org/abs/2210.11466 |
| Block Expansion / PEFT-ViT | https://arxiv.org/abs/2404.17245 · https://github.com/rezaakb/peft-vit |
| Selective Projection Decay (SPD) | https://arxiv.org/abs/2411.01713 · https://github.com/gt-ripl/selective-projection-decay |
| LayerLock | https://arxiv.org/html/2509.10156 |
| Strong Experience Replay / CLEAR | https://arxiv.org/html/2305.13622v2 · https://arxiv.org/html/2404.12526v1 |
| ULMFiT (LLRD + STLR) | https://arxiv.org/pdf/1801.06146 |
| V-JEPA 2 | https://arxiv.org/abs/2506.09985 |
| V-JEPA 2.1 | https://arxiv.org/html/2603.14482v1 |
| **vjepa2 train.py (m09a/m09c training-loop gold)** | https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_2_1/train.py |
| **MGMAE — Motion-Guided Masking (mask-conditioned video SSL gold)** | https://github.com/MCG-NJU/MGMAE · https://arxiv.org/abs/2308.10794 |
| **VideoMAE (foundational video masked SSL)** | https://github.com/MCG-NJU/VideoMAE |
| MotionMAE (motion-aware MAE) | https://github.com/happy-hsy/MotionMAE |
| Text-Guided Video MAE (saliency masking via captions) | https://arxiv.org/abs/2408.00759 |
| ForAug (foreground/background recombine via masks) | https://arxiv.org/html/2503.09399 |
| Hide-and-Seek (mask-driven pixel hide augmentation) | https://arxiv.org/abs/1811.02545 |
| RAFT (torchvision optical flow) | https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py |
| Grounded-SAM-2 | https://github.com/IDEA-Research/Grounded-SAM-2 |
| SAM 3 | https://github.com/facebookresearch/sam3 |

---

> 🎬 **Bottom line**: regression is recipe-mechanism, not data-deficit. Run the **frozen-teacher × LP-FT** POC for $1 — its result picks the next branch:
> &nbsp;&nbsp;🟢 **clear win** (Δ ≥ +5 pp) → wire fixes, refactor, FULL surgery
> &nbsp;&nbsp;🟡 **marginal win** (0.81–0.83) → Phase 5 harder FG features (`plan_phase5_fg_motion_features.md`)
> &nbsp;&nbsp;🔴 **full fail** (all 4 regress) → Path 2 relax m10 thresholds
> 💡 **No analysis paralysis**: phase5's own gate ("only proceed if Δ < +5 pp") makes this sequence the cheapest causal path. 🚀
