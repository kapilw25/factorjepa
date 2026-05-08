# 🧪 iter14 — Surgery on Pretrain (causal-attribution experiment)

> ## 🎯 Paper goal:  `vjepa_surgery` ≫ `vjepa_pretrain` ≫ `vjepa_frozen` on motion / temporal features
>
> **iter14 thesis** — surgery composes ON TOP OF a continual-pretrain checkpoint (Option A in `plan_surgery_on_pretrain.md`); a compute-matched **long-pretrain (10 ep)** ablation arm is added so the gain is **causally attributable to factor patching**, not to the extra training steps.

---

## 🟢 Status carried over from iter13 (already proven, do not re-run)

| ✅ Artifact | Where | Evidence |
|---|---|---|
| `pretrain (5 ep)` student encoder | `outputs/full/m09a_pretrain/student_encoder.pt` (6.9 GB) | `probe_top1=0.808` · `motion_cos↑5.8×` (0.046→0.267) |
| `pretrain (5 ep)` predictor ckpt | `outputs/full/m09a_pretrain/m09a_ckpt_best.pt` (14 GB) | carries `predictor` key for Stage 8 future_mse |
| **`pretrain > frozen` on `future_mse`** | `outputs/full/probe_future_mse/probe_future_mse_per_variant.json` | Δ = **+0.0027**, CI [0.0017, 0.0037], **p = 0.0** ✅ |
| Factor data (m10/m11) | `data/eval_10k_local/{m10_sam_segment, m11_factor_datasets}` | 9,297 clips · 277 K agents · 356 K interactions · `quality_gate=PASS` |
| Motion-flow probe gate | `outputs/full/probe_action/` | 16-class motion_flow probe trained at FULL |

🟢 Half of the strict ordering (`pretrain > frozen`) is already statistically established. iter14 = prove the **second half** (`surgery > pretrain`) cleanly.

---

## 🧭 Three-arm experimental design

### 🎯 The 4 encoders eval will compare

| # | Encoder ID | Init from | Training | Status |
|---|---|---|---|---|
| 0️⃣ | `vjepa_2_1_frozen` | Meta V-JEPA 2.1 ViT-G | none (zero-shot) | ✅ ready |
| 1️⃣ | `vjepa_2_1_pretrain` | Meta V-JEPA 2.1 ViT-G | continual SSL **5 ep** on Indian video | ✅ DONE |
| 2️⃣ | `vjepa_2_1_pretrain_2X` | Meta V-JEPA 2.1 ViT-G | continual SSL **10 ep** (compute-matched control, NO factor patching) | 🆕 TO RUN |
| 3️⃣A | `vjepa_2_1_surgical_3stage_DI` | **`vjepa_2_1_pretrain` student** ← 🆕 | factor surgery (D_L → D_A → D_I), **5 ep** | 🆕 TO RUN |
| 3️⃣B | `vjepa_2_1_surgical_noDI` | **`vjepa_2_1_pretrain` student** ← 🆕 | factor surgery (D_L → D_A only), **5 ep** | 🆕 TO RUN |

### 🔄 Sequential composition (encoder 3 = encoder 1 + factor patching)

```text
                                                  (factor labels from m10/m11)
                                                              ▼
Meta V-JEPA 2.1 ──▶ pretrain (5 ep) ─────────────▶ surgery_3stage_DI (5 ep)
                       │                              ┃
                       │                              ┗━▶ surgery_noDI    (5 ep)
                       │
                       └──▶ pretrain_2X (10 ep)  ←─── ❎ NO factor patching (control)
```

---

## 🧮 Three paired-Δ tests after eval

| Δ | Comparison | What it proves | Already done? |
|---|---|---|---|
| **Δ1** | `pretrain (5)` vs `frozen` | continual SSL adapts to Indian video | ✅ p=0.0 |
| **Δ2** | `surgical (5+5)` vs `pretrain (5)` | factor patching adds value on top of SSL | 🆕 |
| **Δ3** | `surgical (5+5)` vs `pretrain_2X (10)` | **factor patching is CAUSAL — not just extra steps** ⭐ | 🆕 |

### 🎯 Decision matrix (paired BCa 10K-resample CI; significance = non-overlapping 95 % CI / p < 0.05)

| Δ2 outcome | Δ3 outcome | Reading | Paper framing |
|---|---|---|---|
| ✅ | ✅ | **Strongest claim** — factor patching beats both pretrain and compute-matched long-pretrain | 🏆 "Factor-patching causally lifts V-JEPA 2.1 motion features beyond continual-SSL alone" |
| ✅ | ❌ | factor patching wins, but extra-steps confound — could also be timing, not factors | 🟡 weaker claim; still publishable as "factor patching ≥ extra SSL steps" |
| ❌ | ✅ | unlikely — surgery loses to pretrain but beats long-pretrain (would imply long-pretrain itself overfits/forgets) | ⚠️ pivot: report long-pretrain regression as separate finding |
| ❌ | ❌ | factor patching adds nothing beyond more SSL on the same data | 🔴 negative result; pivot to dataset + pipeline contribution per fallback in `iter/utils/literarure_survey.md` |

---

## 🛠️ Implementation diffs needed (detailed in T3 plan response)

| File | Change | LoC est |
|---|---|---|
| `configs/train/probe_pretrain_2X.yaml` (NEW) | extends `probe_pretrain.yaml`; overrides `max_epochs.full: 10` | ~12 |
| `configs/train/surgery_3stage_DI_iter14.yaml` (NEW, optional) | extends `surgery_3stage_DI.yaml`; pins `max_epochs.full: 5` (ensures 5+5=10 budget) | ~10 |
| `configs/train/surgery_2stage_noDI_iter14.yaml` (NEW, optional) | mirror of above for noDI | ~10 |
| `src/m09c_surgery.py` | add `--init-from-ckpt <path>` so surgery loads `student_encoder.pt` from `pretrain` instead of Meta V-JEPA URL (current default at line 259) | ~25 |
| `scripts/run_probe_train.sh` | add `pretrain_2X` subcommand; thread `--init-from-ckpt` through `surgery_*` subcommands | ~30 |
| `scripts/run_probe_eval.sh` | add `vjepa_2_1_pretrain_2X` to default `ENCODERS`; new `encoder_ckpt_for` + `encoder_predictor_ckpt_for` cases | ~15 |
| `src/probe_action.py` (paired_delta stage) | extend the pairwise table to emit Δ2 + Δ3 explicitly (currently emits all O(N²) pairs but not labeled as Δ1/Δ2/Δ3) | ~20 |

🧮 **Total** ≈ ~120 LoC, all surgical (no architectural rewrites).

---

## ⏱️ Wall-clock + GPU-budget plan

| Run | Wall (RTX Pro 6000 Blackwell, 96 GB) | $-cost (~$0.8/h) |
|---|---|---|
| pretrain (5 ep) | DONE — 10h 16m | $8.20 ✅ paid |
| `pretrain_2X` (10 ep) | ~20 h (linear 2× of pretrain) | ~$16 |
| `surgery_3stage_DI` (5 ep on pretrain) | ~10 h (similar to pretrain compute, 3 stages × shorter epochs) | ~$8 |
| `surgery_noDI` (5 ep on pretrain) | ~7 h | ~$5.60 |
| `run_probe_eval.sh --FULL` (4 encoders, 10 stages) | ~4 h | ~$3.20 |
| **Total iter14** (incremental) | **~41 h** | **~$33** |

### 📋 Recommended execution order (max early-signal)

1. 🥇 **Surgery (5+5) first** — kicks off off pretrain ckpt; if `surgery > pretrain` already shows up at eval, that's Δ2 ✅ → motivates running long-pretrain control.
2. 🥈 **`pretrain_2X (10 ep)`** in parallel/next — only needed once we have any signal on Δ2.
3. 🥉 **Surgery_noDI** last — ablation arm; only needed if Δ2/Δ3 land.

If Δ2 fails outright, abort long-pretrain to save GPU time.

---

## 🧠 Anti-forgetting safeguards in surgery (per Q1.2 / Q1.3 in `plan_surgery_on_pretrain.md`)

🛡️ **Already in proposal Sec 10.6 / 11.5 (re-applied for surgery init from pretrain):**

- ⚓ **L2 anchor loss** `λ‖θ − θ_pretrain‖²`, `λ ∈ [0.001, 0.01]`
- 📐 **Layer-wise LR decay 0.7–0.9** across the unfrozen prefix
- 🚦 **Backbone LR cap ≤ 1e-5** (vs predictor 1e-4)
- 🔁 **EMA τ ≥ 0.99**
- 🔥 **Short 100–500-step warmup** at each stage boundary

🆕 **New for iter14 (recommended additions):**

- 📊 **Per-block CKA similarity** vs `θ_pretrain` logged every checkpoint (extends existing `m09a_block_drift.png`)
- 🏷️ **"Old probe" retention metric** — frozen probe trained on pretrain features, applied to surgery checkpoints; drop = forgetting
- 🛑 **Early-abort surgery Stage-1** if `val_jepa` on pretrain-val rises > 5 %

---

## 📂 Reference paths

| What | Where |
|---|---|
| 📒 Q&A on Option A vs B | `iter/iter14_surgery_on_pretrain/plan_surgery_on_pretrain.md` |
| 📍 This file | `iter/iter14_surgery_on_pretrain/plan_HIGH_LEVEL.md` |
| 🛤️ iter13 final state (pretrain+factor-prep) | `iter/iter13_motion_probe_eval/result_outputs/v12/` (37 GB checkpoints already deleted; metrics + plots retained) |
| 📓 Original FactorJEPA proposal | `Literature/proposal/FactorJEPA/FactorJEPA.md` (Sec 10 = continual pretrain, Sec 11 = surgery) |
| 🐛 Bug log | `iter/iter12_multitask_LOSS/errors_N_fixes.md` (#1–#81 carry over) |
| 🚨 Fallback techniques | `iter/utils/literarure_survey.md` (24 JEPA variants if iter14 also fails) |

---

## ❌ What iter14 explicitly does NOT do

| Cancelled | Why |
|---|---|
| ❌ Re-run iter13 pretrain (5 ep) | already produced `pretrain > frozen` at p=0.0 — reuse the checkpoint |
| ❌ Re-run m10/m11 factor-prep | DONE 9,297 clips · `quality_gate=PASS` — reuse |
| ❌ Surgery from FROZEN init | Option B in `plan_surgery_on_pretrain.md` — literature + architectural-design analysis ranks it ~30–45 % P vs ~55–65 % P for Option A |
| ❌ Encoder-only fine-tuning recipes (iter9–iter12) | 5 distinct recipes failed to lift Prec@K; iter13 motion-flow probe gate is the validated metric |
| ❌ Multi-task probe loss sweep | iter12 E v3 already showed flat — bottleneck isn't loss balance |
| ❌ DINOv2 / CLIP / SigLIP swap | constraint: paper is V-JEPA 2.1 only |

---

## 🔓 Open questions

1. 🔍 **Surgery epoch budget** — keep 5 epochs total (matched to "(5+5)" framing) OR keep current 15-epoch surgery (3 stages × 5 ep) and match long-pretrain to 20 ep? `plan_surgery_on_pretrain.md` Q3.1 uses the 5+5 framing; this is the cheaper experiment but may underexpose surgery. Decision for T3.
2. 📏 **Anchor `λ` value** — start at `λ = 0.005` (literature default for sequential SSL) or run a 3-point sweep `{0.001, 0.005, 0.01}`?
3. 🤗 **Pretrain HF push timing** — push `student_encoder.pt` BEFORE surgery starts (so surgery downloads from HF for reproducibility) OR after iter14 completes (single bundled upload of all 4 encoders)?
