# Experiment Log — FactorJEPA Continual Pretraining

> **Hard cap: 1000 lines** (budget: 50 iterations × ~18 lines/run + cross-run table + spec). Append-only, POST-completion only.
> Live state lives in `plan_TODO.md` / `errors_N_fixes.md` / `git log` / `training_summary.json` — **not here**.
> Per-run entries **omit** Provenance (§1) and Environment Hash (§15) — ephemeral, recoverable from logs/git. Hypothesis + Config Delta (§2) is **promoted to the cross-run table below** so every experiment is comparable at a glance.

---

## 📊 iter9/iter10 10K Cross-Run Outcome Table (Ch11 Surgery)

> Source: `iter/iter10/logs/{iter9_10k_overnight_v10,v13,v14}.log`, `iter10_v15{a,b}.log`, `v15c_retry.log`. All rows use `data/subset_10k.json` (9,566 clips after filter) · val_500/test_500 stratified splits.

| log | recipe | 🚂 train / 🧪 val / 🎯 test | 🅻L · 🅰A · 🅸I | SAM rec | stg | n_agents | Δ E | Δ H | p (E/H) | wall | verdict | takeaway |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 🅰️ v10 | iter9 early · broad taxonomy 🦜 · 3-stg schedule · stage3 L=33/A=67 renorm | 9,566 / 500 / 500 | ✅ · ✅ · ❌ (D_I=0) | **0.655** ✅ | 3️⃣ | **312,103** 🏋️ | ➕0.14 🟢 | ➕0.24 🟢 | — | 3h17m ⚡ | ⚠️ gate crash (`KeyError: precision_at_k`) | 🧯 biggest Δ seen · but ≠ apples-apples (broad taxonomy + 88% clip coverage) |
| 🅱️ v13 | narrow taxonomy 🪶 · 2-stg · m10 cold re-mine · H1 lr×10 · H4 DINO 0.20/0.18 · H5 plateau-kill | 9,566 / 500 / 500 | ✅ · ⚠️ (S2 killed step 1) · ❌ | 0.632 ⚠️ | 2️⃣ | 223,826 | 0.00 🟡 | 0.00 🟡 | — | **9h45m** 🐢 | ❌ FAIL | 🛠️ long wall = m10 cold re-mine; D_A never tested (H5 state-across-stages bug → v14 H10 fix) |
| 🅲 v14 | narrow · 2-stg lean 🏆 · S2 L=30/A=70 · H10 per-stage plateau reset | 9,566 / 500 / 500 | ✅ · ✅ · ❌ | 0.632 ⚠️ | 2️⃣ | 223,826 | ➕0.07 🟢 | ➕0.13 🟢 | — | 3h07m ⚡ | ❌ (<+3pp 🎯) | 👑 apples-apples leader · 20× below NeurIPS bar · dataset-limited |
| 🅳 v15a | more-laps 🔄 · 2-stg · max_epochs=3 · S2 L=30/A=70 | 9,566 / 500 / 500 | ✅ · ✅ · ❌ | 0.632 ⚠️ | 2️⃣ | 223,826 | 0.00 🟡 | ➖0.07 🔴 | — | 5h36m 🐌 | ❌ FAIL | 📉 3× epochs → 0 · no data-starvation at 10K |
| 🅴 v15b | louder-agent 📢 · 2-stg · S2 L=15/A=85 | 9,566 / 500 / 500 | ✅ · ✅ · ❌ | 0.632 ⚠️ | 2️⃣ | 223,826 | ➖0.10 🔴 | ➖0.07 🔴 | — | 3h07m ⚡ | ❌ FAIL | ⚖️ 70/30 mix optimal · tilting toward A overfits agent-noise |
| 🅵 v15c | safer-int 🛡️ · 3-stg schedule · stage3 L=50/A=50 · D_I streaming code live | 9,566 / 500 / 500 | ✅ · ✅ · ❌ (interaction_mining disabled, D_I=0) | 0.632 ⚠️ | 3️⃣ | 223,826 | ➖0.13 🔴 | 0.00 🟡 | 0.375 / 0.975 🟡 | 3h17m ⚡ | ❌ FAIL | 🎭 Easy hurt; Hard flat — stage3 L/A re-balance adds noise on narrow taxonomy |
| 🧭 **META** | iter10 hypothesis space exhausted 🔚 · **v15c withdrawn** (contaminated ckpt deleted) | — | — | — | — | — | — | — | — | **30h 13m** 🕐 from v1–v3 (v3 FATAL'd at frozen m05 9297/10k · 703-clip PyAV decode). v4/v5 killed mid-ramp; **v6 = clean resume** started 2026-04-22 18:45 w/ N=5 variants (v15c skipped), ETA +15h 15m | 🚀 escalate to 50K | 🔑 **iter11 @ 50K** — NOT clean-v15c (9h m10 `--interactions-only` + 3h m09c re-train confirmed silently re-balanced to L=50/A=50 at 10K, Δ unlikely to separate from 0 given v10-v15b all cluster near 0). Instead port v10's **broad-taxonomy + 3-stg** recipe to 50K where data×5 may lift the +0.14 Δ seen at 10K. **D_I still untested** (0 tubes × all 6 iter10 runs) — defer its clean test to post-50K-baseline. **v15c contamination root cause** (documented, not re-run): `scripts/run_iter9_10k.sh` m11 invocation hardcoded base yaml → `interaction_mining.enabled` never propagated → segments.json had `n_interactions=0` → `StreamingFactorDataset` silently renormalized `{L:0.15, A:0.15, I:0.70}` → `{L:0.50, A:0.50}`. Fail-loud raise now in `utils/training.py` (#3/#4 fixed this session) + B50/B51 preflight added. Paired-BCa p-values for v10/v13/v14/v15a/v15b now unblocked via m05 `--stuck_clips` partial-tolerance patch (iter10 2026-04-22, 3-tier 80%/95% gate). |

**Key shift in conclusion** v14 is only the leader **among narrow-taxonomy runs**. v10 (broad taxonomy, 312K agents, 3-stage schedule) had a larger raw Δ (+0.14/+0.24) but gate-crashed on `KeyError: precision_at_k` → never statistically confirmed. Before defaulting to v14's 2-stage recipe for iter11, re-run v10's recipe at 50K with fixed gate + paired BCa. Also note: **no run ever trained with D_I tubes** (D_I=0 across all 6) — the "3-stg" label is a schedule, not a factor set; stage 3 in v10 and v15c ran L/A-only re-mixtures.

---

## 📊 iter10 `paired_eval_10k` Progress Table — N=9,297 paired-BCa (LIVE SNAPSHOT @ 2026-04-23 21:18, v10 log)

> Source: `logs/run_paired_eval_10k_v10.log` · subset `data/eval_10k.json` (10,000 requested → **9,297 embedded** after 703 stuck_clips PyAV-decode failures, both frozen+surgical converge on same 9,297). CI_half at N=9,297 ≈ ±0.42 pp vs ±2.35 pp at N=500 from iter9 test_500 → ~5.6× tighter confidence. v10 + v13 cells are FINAL (archived); v14 m05 in-flight; v15a/v15b queued.

| Stage / metric | **Frozen** (baseline) | **v10** | **v13** | **v14** | **v15a** | **v15b** |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| 📦 m09c ckpt staged | — (no surgery) | ✅ | ✅ | ✅ | ✅ | ✅ |
| 🎞️ m05 embed | ✅ 9,297/10,000 (703 stuck_clips, #77) | ✅ G2-staged (9,297, skipped re-embed) | ✅ 9,297/10,000 PARTIAL | 🟡 **7,568/10,000 (76 %)** running (G3 resumed from fingerprinted ckpt) | ⬜ queued | ⬜ queued |
| 📈 m06 Prec@K **Easy** | **40.45 %** | 40.44 % | **40.37 %** ± 0.66 | — | — | — |
| 📈 m06 Prec@K **Hard** | **38.83 %** | 38.78 % | **38.74 %** ± 0.66 | — | — | — |
| 📈 m06 mAP@K **Easy** | 0.3278 | 0.3271 | 0.3271 ± 0.0067 | — | — | — |
| 📈 m06 mAP@K **Hard** | 0.3086 | 0.3078 | 0.3084 ± 0.0066 | — | — | — |
| 📈 m06 Cycle@K **Easy** | 79.80 % | 80.14 % | 79.60 % | — | — | — |
| 📈 m06 Cycle@K **Hard** | 77.12 % | 77.84 % | 76.61 % | — | — | — |
| 📈 m06 nDCG@K **Easy** | 0.9580 | 0.9576 | 0.9584 ± 0.0008 | — | — | — |
| 📈 m06 nDCG@K **Hard** | 0.9556 | 0.9550 | 0.9558 ± 0.0008 | — | — | — |
| ⚖️ m08b intersect gate (#79) | — (ref) | ✅ 9,297∩9,297=9,297 (dropped 0/0) | ✅ 9,297∩9,297=9,297 (dropped 0/0) | — | — | — |
| 🎯 m08b Δ Prec@K **Easy** (CI, p) | 0 (ref) | **−0.0143 ± 0.4178** · p=0.9556 🟡 | **−0.0843 ± 0.4102** · p=0.6856 🟡 | — | — | — |
| 🎯 m08b Δ Prec@K **Hard** (CI, p) | 0 (ref) | **−0.0502 ± 0.4141** · p=0.8232 🟡 | **−0.0843 ± 0.4078** · p=0.6848 🟡 | — | — | — |
| 🎯 m08b Δ mAP@K **Easy** | 0 (ref) | −0.0008 ± 0.0040 · p=0.7114 | −0.0007 ± 0.0039 · p=0.7216 | — | — | — |
| 🎯 m08b Δ mAP@K **Hard** | 0 (ref) | −0.0008 ± 0.0039 · p=0.6888 | −0.0002 ± 0.0039 · p=0.9206 | — | — | — |
| 🎯 m08b Δ Cycle@K **Easy** | 0 (ref) | +0.0034 ± 0.0086 · p=0.4296 | −0.0020 ± 0.0083 · p=0.6376 | — | — | — |
| 🎯 m08b Δ Cycle@K **Hard** | 0 (ref) | **+0.0072 ± 0.0087** · p=0.1066 (closest to sig) | −0.0052 ± 0.0085 · p=0.2350 | — | — | — |
| 🎯 m08b Δ nDCG@K **Easy** | 0 (ref) | −0.0004 ± 0.0007 · p=0.2892 | +0.0004 ± 0.0007 · p=0.2750 | — | — | — |
| 🎯 m08b Δ nDCG@K **Hard** | 0 (ref) | −0.0006 ± 0.0008 · p=0.1146 | +0.0002 ± 0.0008 · p=0.6180 | — | — | — |
| 📥 archive landed | ✅ `frozen_eval10k/` | ✅ `v10_eval10k/paired_bootstrap_results.json` @ 16:45:59 | ✅ `v13_eval10k/paired_bootstrap_results.json` @ 19:17:45 | — | — | — |
| ⏱️ wall (this v10 run) | ~2.3 h (m05 re-embed from 0) | 16 s (G2 m08b only) | ~2 h 32 m (16:45→19:17) | in-flight ~32 m elapsed | — | — |

**Headline read at v10 + v13 (both FINAL):**
- Both surgical variants show **Δ Prec@K ≈ 0.0 pp at tight N=9,297 CI (±0.42 pp)** — all 8 paired BCa cells for Prec@K are non-significant, p≥0.68 everywhere.
- v13 is STRICTLY WORSE than v10 on Easy (Δ=−0.0843 vs −0.0143) — the `H1 lr×10 + H4 DINO-up + H5 plateau-kill` recipe added noise, not signal.
- v10 remains the only variant with ANY metric trending positive (Δ Cycle@K Hard +0.0072, p=0.1066), but still far from significance.
- **v14 ETA ≈ 39 min** (7,568/10,000 @ 1.0 clip/s + m06 ~7 min + m08b ~10 s) · **v15a/v15b ETA ≈ 5 h each**.

---

## 📊 Legacy Cross-Run Outcome Table (pre-iter9, non-comparable scales)

| # | Date | Run name | Scale | N_train / N_val / N_test | Stages | max_ep | LR | Key config delta | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-03-29 | 10K POC fake | 10K | 10K / 1K / — | — | 5 | 1e-5 | ImageNet norm=NO (bug) | 🩺 FALSE-POS |
| 2 | 2026-04-05 | 115K λ=0.001 1ep | 115K | 114,576 / 1K / — | — | 1 | 1e-5 | L2 drift λ=0.001, ImageNet YES | ❌ FORGETTING (Prec@K −21.8 pp) |
| 3 | 2026-04-19 | iter8 1K POC Surgery | 1K | 900 / 100 / — | 3 (L/A/I) | 5 | 1e-6 | 3-stage, internal 90/10 split, val=val_1k itself | ❌ gate FAILED (N=100 underpowered, CI ±4.5) |

---

## 📋 Per-Run Format (≤ 18 lines)

Each run entry has 5 parts: (1) `## Run` header + 1-line headline (wall/cost/steps/best/BWT), (2) Gate table (m06 `test_X.json`, 3-5 metric rows), (3) Probe trajectory (3 key rows: S1-best / S2-end / last), (4) Stages summary line (per-stage wall + Δloss + best-ckpt count + m10/m11 totals), (5) Diagnosis + `errors_N_fixes #NN` fixes landed.
Source files (cross-referenced, not duplicated): `outputs/<mode>/m09c_surgery/{training_summary.json, probe_history.jsonl, loss_log.jsonl}`, `outputs/<mode>/m06_faiss_metrics/m06_metrics_*.json`, orchestrator log `logs/iter<N>_*.log`.

---

## Run 2026-03-29 · 10K POC fake (ImageNet norm bug) · 🩺 FALSE POSITIVE

**Train** λ=0.001, 5 ep, 10K × BS=112, 16f, **ImageNet norm=NO (bug)**, jepa_loss stuck at 1.49 (never dropped from init)

| Gate (10K POC hold-out) | Frozen | Adapted | Δ | Note |
|---|---|---|---|---|
| Prec@K | 36.14 | 36.09 | −0.05 | noise — weights barely moved (training ineffective) |
| Overlap@K | baseline | slight↑ | small | JEPA-loss-neighbor metric; coincidental lift |

**Diagnosis** Input range [0,1] vs expected [-2.1, 2.6] → noise gradients → model stayed near frozen init → "close to frozen" misread as "good". Radar had no min-max normalization so adapted ≈ frozen visually. False positive unmasked by 2026-04-05 run which fixed norm → real gradients → catastrophic forgetting.

---

## Run 2026-04-05 · 115K λ=0.001 1 epoch · ❌ CATASTROPHIC FORGETTING

**Train** 115K × BS=112, 1023 steps, λ=0.001 L2 drift (drift_loss=0.00047), ImageNet YES, jepa_loss 0.497 → 0.476, val=1.648

| Gate (10K POC hold-out, BCa 95 % CI) | Frozen | Adapted | Δ | Sig? |
|---|---|---|---|---|
| Prec@K | 36.1 ±0.6 | 14.3 ±0.3 | **−21.8** | ✅ YES |
| mAP@K | 0.278 ±0.006 | 0.080 ±0.002 | −0.198 | ✅ YES |
| nDCG@K | 0.950 ±0.001 | 0.906 ±0.001 | −0.045 | ✅ YES |
| Cycle@K | 76.0 ±0.8 | 75.5 ±0.8 | −0.5 | ❌ |

**Diagnosis** λ=0.001 drift penalty (0.00047) was **1000× smaller** than JEPA loss (0.476) → effective λ ≈ 0 → unbounded feature drift. Val_loss 1.648 ≫ train 0.476 = overfit/forgetting. Prec@K collapsed to random level (12.2 %). EWC literature uses λ=10²–10⁹ (arXiv 2505.05946, 2603.18596). **Fixes landed** λ sweep [10, 100, 1000] planned but superseded by Ch11 factor-surgery pivot (iter8+).

---

## Run 2026-04-19 · iter8 1K POC Surgery (3-stage, 5 epoch) · ❌ GATE FAILED (N=100 underpowered)

**Wall** A→F 3h 40m · **Train** 139 steps × 3 stages · **BWT** −0.33 pp (noise at N=100 CI±4.5) · **Best** Prec@K=20.50 step 12 (Stage 1)

| Gate (val_1k 100-hold-out, BCa 95 % CI) | Frozen | Surgical | Δ | Sig? |
|---|---|---|---|---|
| Prec@K | 20.17 ±4.50 | 20.33 ±4.67 | +0.17 | ❌ overlap |
| mAP@K | 0.1299 ±0.045 | 0.1313 ±0.046 | +0.0014 | ❌ |
| Cycle@K | 0.69 ±0.09 | 0.69 ±0.09 | 0 | ❌ |

| Probe (100-val, 70 probes) | step | Prec@K | val_jepa | note |
|---|---|---|---|---|
| S1 best | 12 | **20.50** | 0.4854 | auto-promoted to student_encoder.pt |
| S2 end | 93 | 20.33 | 0.4735 | −0.17 vs S1 best |
| S3 end | 139 | 19.83–20.00 | 0.4749 | never re-gained; Cycle@K drifted 63→62 |

**Stages** wall A=1h00m20s · B=11m43s · C=2h27m (63 s/step) · D=1m58s · E=4m45s · F=~1m · **Disk** 71 GB (55 GB D_{L,A,I} → retired by streaming refactor)
**Diagnosis** N=100 CI ±4.5 pp cannot resolve sub-pp deltas. Stage 3 net-useless at 1K (multi-signal): val_jepa 4× slower, 0 best-ckpt events in 27 probes, Cycle@K monotone drift. **Fixes landed** #63 probe infra · post-POC: 2-stage recipe + replay 10→30 % D_L + plateau/BWT early-stop + use_permanent_val + val_500/test_500 split (#66) + streaming refactor (#65, 340 GB → 3.5 GB disk).

---

## Run 2026-04-20 17:51 · iter9 10K v10 (3-stage yaml drift, D_I=0) · 🟡 SATURATED

**Wall** 3h 16m 49s · **$** ~$2.62 · **Train** 297 steps × 3 stages (YAML drift #71, D_I=0 renorm to L=33%/A=67%) · **Best** Prec@K=30.33 step 58 (S1) · **BWT** −0.40 pp

| Gate (test_500, N=500, BCa 95 % CI ±2.35) | Frozen | Surgical | Δ | Sig? |
|---|---|---|---|---|
| Prec@K | 27.83 ±2.35 | 27.97 ±2.35 | **+0.14** | ❌ overlap |
| mAP@K | 0.1921 ±0.023 | 0.1927 ±0.023 | +0.0006 | ❌ |
| Cycle@K | 69.60 | 69.20 | −0.40 | — |
| nDCG@K | 0.9485 | 0.9482 | −0.0003 | ❌ |

| Probe (val_500, 13 probes) | step | Prec@K ±CI | val_jepa | Cycle@K | BWT |
|---|---|---|---|---|---|
| S1 best | 58 | **30.33** ±2.40 | 0.4828 | 71.6 | +0.03 |
| S2 end | 196 | 30.13 ±2.40 | 0.4770 | 72.6 | −0.17 |
| S3 end | 297 | 29.90 ±2.42 | 0.4740 | 73.4 | **−0.40** |

**Stages** S1=49m35s (Δloss −0.010, 2 best-ckpt) · S2=45m42s (−0.005, 0) · S3=54m15s (−0.009, 0) · m10=PASS · m11 9566 manifest (D_L 9501 / D_A 7977 / D_I 0) · peak VRAM 52 GB / 102 GB
**Diagnosis** val_jepa ↓ & Cycle@K **+1.8 pp** BUT Prec@K monotone ↓ from step 58 → **representation decoupling**. val_500 (30.33) vs test_500 (27.97) gap = 2.36 pp > CI_half → val/test splits not equally-difficult (methodology concern for paper). **Fixes landed** #71 yaml 3→2 (v11+) · #72 live-plot · #73 BWT Option C-adapted (ci_frac×abs_floor × patience=3) · #74 gate key `prec_at_k` · #75 m08b `m06_faiss_metrics/` path

---

## Run 2026-04-21 00:10 · iter9 10K v13 (H1+H4+H5 + stratified splits) · 🟡 FLAT (Δ=0.00 pp)

| Gate (test_500, N=500, BCa 95 % CI) | Frozen | Surgical | Δ | Sig? |
|---|---|---|---|---|
| Prec@K | 29.93 ±2.38 | 29.93 ±2.37 | **0.00** | ❌ identical |
| mAP@K | 0.2129 | 0.2118 | −0.0011 | ❌ |
| Cycle@K | 71.40 | 71.80 | +0.40 | — |
| nDCG@K | 0.9471 | 0.9466 | −0.0005 | ❌ |

| Probe (val_500, 7 probes) | step | Prec@K ±CI | val_jepa | Cycle@K | BWT |
|---|---|---|---|---|---|
| S1 best | 116 | **29.6 ±2.39** | 0.4670 | 70.8 | +0.10 |
| S1 end | 149 | 29.6 ±2.38 | 0.4647 | 70.8 | +0.10 |
| S2 step 1 (killed) | 150 | 29.6 ±2.38 | 0.4641 | 71.0 | +0.10 |

**Stages** S1=1h37m (Δloss −0.021, 3 best-ckpt) · S2=**1 step** (H5 kill at first S2 probe) · m10 PASS (224k agents · recall 0.632 · mask_conf 0.986; H4 cost: −28 % agents, −0.023 recall vs v10) · m11 9566 manifest (D_I=0)
**Diagnosis** H1+H4 produced a no-op on test_500 — surgical Prec@K identical to frozen. val→test gap collapsed 2.36 pp (v10) → 0.33 pp (v13) → **H2 stratified splits worked**, but surgery itself contributed ∼0 pp over pretrained V-JEPA 2.1. BWT +0.10 + zero drop confirm no forgetting — but because H5's `prec_plateau_state` accumulated across stages, S1's flat-Prec@K window killed at S2 entry → D_A never meaningfully tested. Effectively an accidental H3 (`stage1_only`). **Fixes queued** · **H10** per-stage reset of `prec_plateau_state` + within-stage `≥patience+1` probe requirement · **H11** uncap `max_epochs:1` · **H12** S2 mixture 70/30 → 85/15 D_A · #78 m08b Easy-only + auto-Y + CI bars + 3-panel probe_trajectory.

---

## 🔬 3-experiment comparison (iter8 POC · v10 · v13)

| Axis | iter8 POC (1K, 2026-04-19) | v10 (iter9 10K, 2026-04-20) | v13 (iter9 10K, 2026-04-21, H1+H4+H5) |
|---|---|---|---|
| Train / val split | 900 / 100 | 9566 / 500 | 9566 / 500 |
| LR | 1.0e-6 | 1.0e-6 | **1.0e-5 (H1, 10×)** |
| DINO thresholds (box/text) | 0.15 / 0.12 | 0.15 / 0.12 | **0.20 / 0.18 (H4)** |
| Stages executed | 1L + 2A + 3I | 1L + 2A + 3I | 1L + 1 step of 2A (killed) |
| Total steps | 139 | 297 | 150 |
| Stage-1 / 2 / 3 steps | 46 / 46 / 47 | 99 / 98 / 100 | 149 / 1 / 0 |
| Trainable blocks used | up to 36/48 | up to 36/48 | only 12/48 (Stage 1) |
| n_probes | 70 | 13 | 7 |
| Best Prec@K (val) | 20.5 @ step 12 (S1) | 30.33 @ step 58 (S1) | 29.6 @ step 116 (S1) |
| Final Prec@K (last probe) | 20.0 | 29.9 | 29.6 |
| **BWT** (last − first) | −0.33 | −0.40 | **+0.10 ✅** |
| Max drop from peak | −0.17 | −0.17 | **0.00** |
| Monotonic trajectory? | ❌ decayed in S3 | ❌ decayed after S1 | ✅ flat |
| Best-ckpt stage | stage1_layout | stage1_layout | stage1_layout |
| Early-stop trigger | none (5pp floor) | none (BWT legacy never fired) | ✅ prec_at_k_plateau (new H5) |
| Final train loss | 0.452 | 0.472 | 0.467 |
| Test-500 Prec@K (Frozen) | — | 27.83 ±2.35 | **29.93 ±2.38** (+2.10 pp, H2 artefact) |
| Test-500 Prec@K (Surgical) | — | 27.97 ±2.35 | **29.93 ±2.37** |
| Test-500 Prec@K Δ (Surgical − Frozen) | ❌ +0.17 (CI overlap @ N=100) | 🟡 +0.14 (SATURATED @ N=500) | 🟡 **0.00 pp** (identical, CI overlap) |
| val→test Prec@K gap | — | 2.36 pp (random split) | **0.33 pp (stratified, H2 fix)** |
| Paper-narrative shape | "3-factor curriculum" | "3-factor curriculum" | **"layout-factor surgery" (D_L-only de facto)** |

**Honest 1-line read:** v13 cleanest trajectory (BWT +0.10, zero drop, val↔test aligned) but surgery = frozen on test_500 (29.93 vs 29.93, Δ=0). H2 methodology fix worked; H1+H4 at S1-only produced a no-op; D_A never tested because H5 state spans stages. **v14 = H10** (per-stage H5 reset + within-stage patience) is the one decisive next action to give D_A a real shot under H1+H4.

---

## ⏱️ Wall-time per pipeline step (Ch11 surgery pipeline, A → G)

Derived from `═══ HH:MM:SS · Step X ═══` stamps in each run's orchestrator log. 2026-03-29 fake-pos and 2026-04-05 catastrophic run used pre-Ch11 single-phase pipeline (no A→G stamps) — noted but not timed below.

| Step · .py | **iter8 POC** (1K, 139 steps) | **v10** (10K, 297 steps) | **v13** (10K, 150 steps, H5 kill) | Notes |
|---|---|---|---|---|
| A · `m10_sam_segment.py` | 1h 00m 20s | **~7h 45m** (fresh, m11 errors forced rerun) | **7h 45m 06s** (fresh, H4 rerun) | both v10 & v13 ran m10 end-to-end; v10 log stamp is from the post-crash restart session, not the full cold-start clock |
| B · `m11_factor_datasets.py` (`--streaming`) | 11m 43s | ~5 min (fresh, streaming) | 4m 39s | |
| C · `m09c_surgery.py` | 2h 27m (63 s/step × 139) | **2h 57m 23s** (36 s/step × 297) | **1h 37m 30s** (39 s/step × 150, H5 kill) | v13 ~45 % shorter than v10 because H5 halted at step 150 |
| D · `m05_vjepa_embed.py` (frozen) | 1m 58s | 8m 27s | 8m 29s | 5× longer @ 10K vs 1K (test_500 vs 100 hold-out) |
| E · `m05_vjepa_embed.py` (surgical) | 4m 45s | 10m 28s | 9m 03s | Surgical ~2× frozen (torch.compile overhead on adapted model) |
| F · `m06_faiss_metrics.py` (×2) | ~1m | 23s | 22s | FAISS-GPU kNN + BCa bootstrap |
| G · `m08b_compare.py` | — | — | 2s | Step G added post-v10 (#75) — 2 encoders, plots only |
| **Total A→G wall (end-to-end cold)** | **~3h 47m** | **~11h** (A+B fresh + C→F logged 3h 17m) | **9h 45m 12s** | |
| **GPU cost** @ $0.80/h (96GB Blackwell) | ~$3.03 | **~$8.80** (cold end-to-end) | ~$7.80 | |

**Key timing insights:**
1. **Step A (m10) dominates cold-start** at 10K (~7h 45m, 80 % of total wall for BOTH v10 and v13). At 50K this projects to ~39 h; at 115K ~89 h — m10 is the scale-ladder bottleneck once any factor-generation config (thresholds, prompts, mask params) changes.
2. **Step C (m09c) is scale-linear in steps** (~60 s/step at ViT-G 2B, BS=32, fp32 AdamW) — H5 saving 147 steps cut Step C by 1h 20m in v13 vs v10.
3. **Steps D+E+F+G** are negligible (~18-20 min combined at 10K, independent of training budget).
4. **Pre-Ch11 runs (2026-03-29 fake-pos, 2026-04-05 catastrophic)** used single-phase continual-pretrain pipeline (ch10, no factor generation), wall dominated by 1023 training steps × BS=112 — not directly comparable to the Ch11 A→G timings above.
