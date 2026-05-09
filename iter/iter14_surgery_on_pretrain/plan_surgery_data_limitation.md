# 🚨 iter14 Surgery — Data Limitation + Recipe Mechanism (Table-only)

> 🎯 **Non-negotiable goal**: `vjepa_surgery` ≫ `vjepa_pretrain` ≫ `vjepa_frozen` on motion / temporal features
> 🚫 No claim pivot. We change the experiment.

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

## 🛑 8. Decision gates

| Gate | Pass condition | If FAIL |
|---|---|---|
| 🔬 G-Phase0 | Any sweep cell shows trio top-1 ≥ 0.808 | ⛔ recipe insufficient → Path 2 |
| 🔁 G-Refactor | m09a POC top-1 ∈ [0.4535, 0.4635] AND m09c POC ∈ [0.6949, 0.7949] | ⛔ git revert, investigate hook |
| 🚀 G-Phase1 | FULL surgery probe_top1 > 0.808 + 0.5 pp | ⛔ re-evaluate threshold + epochs |
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

## 🔍 10. m09c gold-standard fidelity (TASK 2 finding)

| 🅼 Module | 🥇 Gold-standard repo | 🟢 Coverage | 🚨 Gap |
|---|---|---|---|
| `m09c_surgery.py` | [vjepa2](https://github.com/facebookresearch/vjepa2) + LP-FT + SALT + SPD | training primitives ✅ | **3-stage factor surgery composition is novel — no canonical clone target** |
| `m10_sam_segment.py` | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) + [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | 4-anchor pattern matches Grounded-SAM-2 ✅ | quality gates (stab/obj/compact) are local |
| `m11_factor_datasets.py` | [SOLV (kuis-ai/solv)](https://kuis-ai.github.io/solv/) — closest only | object-centric exists | **{D_L, D_A, D_I} factorization has no precedent — VIBE-CODED** |

🆕 **Codified in `src/CLAUDE.md` § CODE STANDARDS**: *"Training scripts MUST cite official gold-standard repo URL in docstring."* (9 words, added 2026-05-09)

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
| Grounded-SAM-2 | https://github.com/IDEA-Research/Grounded-SAM-2 |
| SAM 3 | https://github.com/facebookresearch/sam3 |

---

> 🎬 **Bottom line**: regression is recipe-mechanism, not data-deficit. Run the **frozen-teacher × LP-FT** POC for $1 BEFORE committing $50–100 to Path 2 or 1 day-eng to the m09a/m09c refactor. 🚀
