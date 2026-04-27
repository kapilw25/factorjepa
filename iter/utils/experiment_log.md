# Experiment Log — FactorJEPA Continual Pretraining

> **Hard cap: 1000 lines.** Append-only, POST-completion only. Live state lives in `plan_TODO.md` / `errors_N_fixes.md` / `git log` / `training_summary.json`.

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

---

## 📊 iter10 `paired_eval_10k` Progress Table — N=9,297 paired-BCa (FINAL @ 2026-04-23)

> Source: `logs/run_paired_eval_10k_v10.log` · subset `data/eval_10k.json` (10,000 requested → **9,297 embedded** after 703 stuck_clips PyAV-decode failures). CI_half at N=9,297 ≈ ±0.42 pp.

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

---

## 📊 Legacy Cross-Run Outcome Table (pre-iter9, non-comparable scales)

| # | Date | Run name | Scale | N_train / N_val / N_test | Stages | max_ep | LR | Key config delta | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-03-29 | 10K POC fake | 10K | 10K / 1K / — | — | 5 | 1e-5 | ImageNet norm=NO (bug) | 🩺 FALSE-POS |
| 2 | 2026-04-05 | 115K λ=0.001 1ep | 115K | 114,576 / 1K / — | — | 1 | 1e-5 | L2 drift λ=0.001, ImageNet YES | ❌ FORGETTING (Prec@K −21.8 pp) |
| 3 | 2026-04-19 | iter8 1K POC Surgery | 1K | 900 / 100 / — | 3 (L/A/I) | 5 | 1e-6 | 3-stage, internal 90/10 split, val=val_1k itself | ❌ gate FAILED (N=100 underpowered, CI ±4.5) |

---

## Run 2026-03-29 · 10K POC fake (ImageNet norm bug) · 🩺 FALSE POSITIVE

| Gate (10K POC hold-out) | Frozen | Adapted | Δ | Note |
|---|---|---|---|---|
| Prec@K | 36.14 | 36.09 | −0.05 | noise — weights barely moved (training ineffective) |
| Overlap@K | baseline | slight↑ | small | JEPA-loss-neighbor metric; coincidental lift |

---

## Run 2026-04-05 · 115K λ=0.001 1 epoch · ❌ CATASTROPHIC FORGETTING

| Gate (10K POC hold-out, BCa 95 % CI) | Frozen | Adapted | Δ | Sig? |
|---|---|---|---|---|
| Prec@K | 36.1 ±0.6 | 14.3 ±0.3 | **−21.8** | ✅ YES |
| mAP@K | 0.278 ±0.006 | 0.080 ±0.002 | −0.198 | ✅ YES |
| nDCG@K | 0.950 ±0.001 | 0.906 ±0.001 | −0.045 | ✅ YES |
| Cycle@K | 76.0 ±0.8 | 75.5 ±0.8 | −0.5 | ❌ |

---

## Run 2026-04-19 · iter8 1K POC Surgery (3-stage, 5 epoch) · ❌ GATE FAILED (N=100 underpowered)

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

---

## Run 2026-04-20 17:51 · iter9 10K v10 (3-stage yaml drift, D_I=0) · 🟡 SATURATED

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

---

## ⏱️ Wall-time per pipeline step (Ch11 surgery pipeline, A → G)

> Derived from `═══ HH:MM:SS · Step X ═══` stamps in each run's orchestrator log.

| Step · .py | **iter8 POC** (1K, 139 steps) | **v10** (10K, 297 steps) | **v13** (10K, 150 steps, H5 kill) | Notes |
|---|---|---|---|---|
| A · `m10_sam_segment.py` | 1h 00m 20s | **~7h 45m** (fresh, m11 errors forced rerun) | **7h 45m 06s** (fresh, H4 rerun) | both v10 & v13 ran m10 end-to-end |
| B · `m11_factor_datasets.py` (`--streaming`) | 11m 43s | ~5 min (fresh, streaming) | 4m 39s | |
| C · `m09c_surgery.py` | 2h 27m (63 s/step × 139) | **2h 57m 23s** (36 s/step × 297) | **1h 37m 30s** (39 s/step × 150, H5 kill) | v13 ~45 % shorter (H5 halt) |
| D · `m05_vjepa_embed.py` (frozen) | 1m 58s | 8m 27s | 8m 29s | 5× longer @ 10K vs 1K |
| E · `m05_vjepa_embed.py` (surgical) | 4m 45s | 10m 28s | 9m 03s | Surgical ~2× frozen (torch.compile) |
| F · `m06_faiss_metrics.py` (×2) | ~1m | 23s | 22s | FAISS-GPU kNN + BCa bootstrap |
| G · `m08b_compare.py` | — | — | 2s | Step G added post-v10 (#75) |
| **Total A→G wall (end-to-end cold)** | **~3h 47m** | **~11h** (A+B fresh + C→F logged 3h 17m) | **9h 45m 12s** | |
| **GPU cost** @ $0.80/h (96GB Blackwell) | ~$3.03 | **~$8.80** (cold end-to-end) | ~$7.80 | |

---

## 📊 iter11 v3 `ultra_hard_3066` Cross-Run Comparison (2026-04-27, BCa CI ±~3.7 pp at N=306 val)

- Source: `iter/iter11/outputs/epoch5_LR1e5/full/surgery_2stage_noDI/probe_history.jsonl` (A baseline, 27 probes) · `iter/iter11/outputs/epoch15_LR5e5/full/{surgery_2stage_noDI,surgery_3stage_DI}/probe_history.jsonl` (B,C done at 1140 steps, 76 + 78 probes) · `outputs/full/explora/probe_history.jsonl` (D in-flight, 36 probes / 50 % done @ step 555).
- Color rule: 🟢 ≥5 % real progress · 🟡 3–5 % · 🟠 0–3 % inside CI noise · 🔴 ≤0 regression. **All Δ in this table are vs A (5ep LR1e-5 noDI baseline).**

| Metric | A: 5ep LR1e-5 noDI | B: 15ep LR5e-5 noDI | C: 15ep LR5e-5 DI | D: explora v10 (50 %) |
|---|---|---|---|---|
| Prec@K (%) end | 75.22 | 75.11  🔴 −0.1 % | 75.05  🔴 −0.2 % | **76.03**  🟠 +1.1 % |
| mAP@K (%) end | 70.18 | 69.93  🔴 −0.4 % | 70.02  🔴 −0.2 % | **71.37**  🟠 +1.7 % |
| Cycle@K (%) end | 77.45 | 77.12  🔴 −0.4 % | 75.49  🔴 −2.5 % | **79.41**  🟠 +2.5 % |
| val_jepa best ↓ | 0.4663 | **0.4545**  🟠 −2.5 % | **0.4532**  🟠 −2.8 % | 0.4606  🔴 +1.6 % worse |
| within-run val_jepa Δ (end−first) | −5.7 % | −7.2 % | −6.7 % | −6.5 % (mid-flight) |
| Probes recorded | 27 | 76 | 78 | 36 (running) |
| Trainable surface | full prefix (~hundreds M) | full prefix | full prefix | 4.6 % (LoRA + 2 blocks + LN) |
| GPU-hours | ~5 h | ~15 h | ~15 h | ~7.6 h projected |
| Paired-eval Δ Prec@K vs Frozen (Easy, N=308) | n/a | +0.32 ± 0.57 · p=0.31 🟠 not sig | **+0.87 ± 0.60 · p=0.0038 ✅ first-ever-significant** 🟠 | TBD |
| ROI (pp paired-Δ per GPU-hour) | n/a | ~0 🔴 | ~0.06 🟠 | TBD |
| Trajectory verdict | flat plateau | flat plateau | flat plateau | flat plateau (slightly higher band) |

**Verdict** — Is explora v10 learning better than the 3 surgery variants? Marginally — it's the only run that moves probes in the right direction (+1.1 to +2.5 %), but still entirely inside the ±3 % noise band 🟠 and val_jepa is 1.6 % worse than the best surgery 🔴; **no run earns a single green emoji on probes** — bottleneck is the loss function (JEPA L1 reconstruction is not a retrieval surrogate), not the technique.
