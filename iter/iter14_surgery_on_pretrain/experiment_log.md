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

## 📊 iter11 v3 + iter12 multi-task `ultra_hard_3066` Cross-Run Comparison (2026-04-27, BCa CI ±~3.7 pp at N=306 val)

Color rule (Δ vs A baseline; val_jepa inverted: lower=better): 🟢 ≥5 % · 🟡 3–5 % · 🟠 0–3 % (CI noise) · 🔴 ≤0 (regression).

| Metric | A: 5ep LR1e-5 noDI | B: 15ep LR5e-5 noDI | C: 15ep LR5e-5 DI | D: explora v10 | E v3: noDI_multitask UW 2-task | F: 3stage_DI_multitask UW 2-task |
|---|---|---|---|---|---|---|
| Status | done 297 steps | done 1140 steps | done 1140 steps | killed 792/1140 | done 1140/1140 | killed 45/1140 (user) |
| Prec@K best (%) | 75.22 (ref) | 75.11 🔴 −0.15 % | 75.05 🔴 −0.23 % | 75.87 🟠 +0.86 % | 75.87 🟠 +0.86 % (stage1; stage2 best 75.82 = no D_A lift) | n/a |
| mAP@K best (%) | 70.18 (ref) | 69.93 🔴 −0.36 % | 70.02 🔴 −0.23 % | 71.15 🟠 +1.38 % | 70.79 🟠 +0.87 % | n/a |
| Cycle@K best (%) | 77.45 (ref) | 77.12 🔴 −0.43 % | 75.49 🔴 −2.53 % | 80.07 🟡 +3.38 % | 81.70 🟡 +5.49 % (final probe, step 1140) | n/a |
| val_jepa min ↓ | 0.4663 (ref) | 0.4545 🟠 −2.53 % | 0.4532 🟠 −2.81 % | 0.4528 🟠 −2.90 % | 0.4997 🔴 +7.16 % | n/a |
| Probes | 27 | 76 | 78 | 51 | 76 (38 stage1 + 38 stage2) | 0 |
| GPU-h | 5 | 15 | 15 | 5.55 | ~10 | 0.4 (killed) |
| Paired-Δ Prec@K vs Frozen (Easy, N=308) | n/a | +0.32 ± 0.57 · p=0.31 🟠 | +0.87 ± 0.60 · p=0.0038 🟠 | +0.27 ± 0.49 · p=0.29 🟠 | TBD (eval pending) | n/a |
| ROI pp/GPU-h | n/a | ~0 🔴 | 0.06 🟠 | 0.049 🟠 | TBD | n/a |
| UW final w_jepa : w_infonce | n/a | n/a | n/a | n/a | 1.053 : 1.016 (step 1139) | n/a |

---

## 📊 iter13 m09a continual SSL — `eval_10k` v5+v6+v7 Diagnostic (2026-05-04, **STUCK ENCODER**)

iter13's first attempt at continual SSL pretrain on the 10K eval-pool. Multi-attempt run across 3 wall-clock days due to disk-full crashes; v6 patched cleanup, v7 patched best.pt slimming. Final state — **encoder didn't move in any of the 3 attempts**. Empirical proof that LR=1e-5 is below the threshold where ViT-G moves while a 60M-param predictor competes for gradient.

| Metric | v5 (initial) | v6 (post-cleanup-fix) | v7 (post-best.pt-slim) | Combined verdict |
|---|---|---|---|---|
| Status | crashed step 86 (disk-full) | crashed step 744 (disk-full) | done 1085/1085 | 3 attempts → 1 completion |
| LR (peak / floor) | 1e-5 / 1e-6 | 1e-5 / 1e-6 | 1e-5 / 1e-6 | uniform (the bug) |
| Resume start step | 0 | 86 | 731 | resumed forward |
| top1 (kNN-centroid, mid-train) | 0.8259 ↔ 0.8281 over 8 probes | 0.8249 ↔ 0.8281 over 7 probes | 0.8249 ↔ 0.8270 over 6 probes | **🔴 plateau ±3 clips of 925 (0.32 pp = pure noise)** |
| val_jepa best | 0.4641 | 0.4604 | 0.4597 | 🟠 −0.95% over 860 steps (CI noise band) |
| Per-block weight rel_l2 vs Meta init | (not measured) | (not measured) | **~2e-5 across ALL 48 blocks** | 🔴 **100-1000× too small for active fine-tune** |
| Frozen vs trainable layer drift signature | n/a | n/a | indistinguishable | layer_freeze yaml directive was a SILENT NO-OP |
| Probes | 8 | 7 | 6 | 21 total — none showed encoder movement |
| GPU-h | ~0.5 (killed early) | ~3.5 (killed) | ~5.5 (completed) | ~9.5 wasted |

**Root-cause diagnosis** (audit at `iter/iter13_motion_probe_eval/audit.md`):
1. **LR too low** — yaml's lr=1e-5 is 10× lower than Meta's continual SSL peak (1e-4). For ViT-G's 1.84B params, this falls below the threshold where the encoder moves visibly within 1000 steps while a 60M-param predictor also competes for gradient.
2. **`layer_freeze` ignored** — yaml said `enabled: true, freeze_below: 20` but m09a never read it. All 48 blocks were nominally trainable; in practice none of them moved.
3. **MultiSeqWrapper missing** — minor stylistic divergence from Meta's reference, semantically a no-op for our setup.

**Three fixes committed before v8 launch**: (1) `lr 1e-5 → 1e-4` + `warmup 500 → 1500` + `min_lr 1e-6 → 1e-5`; (2) `layer_freeze` actually wired; (3) explicit `training=True` + `mod="video"` to align with Meta's call style. v8 is the diagnostic run — if encoder still doesn't move after Fix #1+#2+#3, deeper rewrite needed.

---

## 📊 iter13 v10 m09a continual SSL pretrain — `eval_10k` (2026-05-05, 🔻 ANCHOR-SATURATION COLLAPSE)

> Source: `outputs/full/probe_pretrain/{loss_log.jsonl,probe_history.jsonl,training_summary.json,m09_block_drift.png}` + v4 eval JSONs (`probe_action/probe_paired_delta_3class.json`, `probe_future_mse/probe_future_mse_per_variant.json`, `probe_motion_cos/probe_motion_cos_paired.json`). Recipe: lr=1e-4 (Meta continual peak, post-v7 LR-bump), `drift_control: l2_uniform, lambda_reg=1.0`, layer_freeze blocks 0-19, 5 epochs / 1085 steps / 34,720 clips_seen. best_state by `val_loss` (NOT probe_top1).

| Aspect | Value |
|---|---|
| **🏃 Training (1085 steps, 5 epochs, 34,720 clips_seen)** | |
| `block_drift_mean` trajectory | 9.7e-5 (s107) → **🔺 1.26e-4 (s215, peak)** → 1.16e-4 (s323) → 9.2e-5 → 8.0e-5 → 6.5e-5 → 4.9e-5 → **🔻 3.2e-5 (s863, min)** → 3.5e-5 (s1079) — **rise-then-fall = anchor-saturation signature** |
| `probe_top1` trajectory | 0.8249 (s107) → 0.8270 → **🔺 0.8281 (s323, peak)** → 0.8270 → 0.8259 (s755-971) → 0.8270 (s1079) — peaked at 1.5 epochs, regressed |
| `val_drift_loss` shrinkage | 0.0205 (s107) → 0.0196 → 0.0106 → 0.00914 → ... → **0.00108 (s1079) — 95% drop** = anchor had nothing to penalize because weights were back at init |
| training_summary | `lambda_reg: 1.0`, `final_jepa_loss: 0.4461`, `best_val_loss: 0.4486` (s863, **collapsed**), `final_lr: 1e-5`, `batch_size: 32` |
| best_state failure | `val_loss` selection picked s863 (collapsed) instead of s323 peak; `student_encoder.pt` exported from collapsed ckpt |
| **🎯 Eval (N=1394 walkindia test, 3-class action probe, BCa 95% CI 10K-iter)** | |
| 🔥 Stage 4 action top-1 | frozen 94.48 % [93.19, 95.55]; pretrain 95.34 % [94.12, 96.34]; **Δ +0.86 pp** [−0.07, +1.87], p=0.095, ❌ CI crosses 0 (saturated 3-class probe) |
| Stage 7 motion_cos (intra−inter) | frozen 0.03806; pretrain 0.03816; Δ +9.6e−5 [+8.8e−5, +1.0e−4], p=0.000, ❌ trivial Δ |
| Stage 9 future_mse (L1 ↓ better) | frozen 0.5576; pretrain **0.5272**; **Δ −0.0304** [−0.0312, −0.0297], p=0.000, ✅ **pretrain wins (−5.5 % rel, only positive signal)** |
| **🩹 v11 fixes wired (code in; RUN pending)** | `drift_control.enabled: false`, `lambda_reg: 0.0` (`probe_pretrain.yaml:119,121`); best_state by `probe_top1` not val_loss (`m09a_pretrain.py:1192-1247`); `keep_last_n: 2 → 5` (`base_optimization.yaml:239`); Phase 2 16-class motion-flow probe (eval-side); Phase 3 motion_aux CE+MSE loss in m09a (training-side) |

**Verdict**: 🔻 **diagnostic-only, no paper signal**. v10 conclusively proves anchor-saturation collapse — encoder peaked at s323 then reverted to init by s863. The lone positive signal (future_mse −5.5 %) is invisible to the saturated 3-class probe. ~$5.20 GPU spent (~$2.40 train + ~$2.40 eval + buffer); locks in v11 + Phase 2 + Phase 3 design.
