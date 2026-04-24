# TODO — iter11

> **Final GOAL: Surgery > ExPLoRA > Frozen on Prec@K, 115K clips.**
> **Immediate GOAL (iter11 v2): 4-variant apples-to-apples at 10K with 5-epoch unified budget, val-loss-plateau-only early-stop.**
> **m10/m11 Goal = maximize D_A/D_L/D_I accuracy for Prec@K**
> **Deadline: NeurIPS May 04 (10 days remaining from 2026-04-24).**
> **If surgery doesn't improve:** `iter/utils/literarure_survey.md` — 24 JEPA variants, 3 fallback techniques.
> **Live code plan: `iter/iter11/plan_code_dev.md` (rename yamls → train.sh → eval.sh).**

---

## ✅ DONE: Grounded-SAM Pivot (2026-04-14, on GPU directly)

**Architecture (locked in):**
```
Grounded-SAM Path D:  fixed 17-cat agent taxonomy → Grounding DINO (text → boxes on frame 0)
                      → SAM 3.1 add_prompt(text=cat, boxes=DINO_xywh_norm, box_labels=[1]*N)
                      → SAM 3.1 propagates masks across all 16 frames (text drives tracking,
                        boxes refine frame 0). Per-category sessions preserve obj_id ranges
                        (offset += 100) for D_I cross-category mining.
```

**Verified on SANITY v5 (20 clips, 24GB RTX PRO 4000 Blackwell):**
- 12/20 clips with detected agents (8 truly-empty Goa/monument scenes correctly skipped)
- 82 total agent detections, mean mask confidence 0.93
- 39 interaction tubes from 9/20 clips (vs 0 with text-only or boxes-only)
- Per-clip 2x2 verify grids show clean D_L blur, D_A isolation, D_I crops
- m10 quality gate PASS, m11 manifest produced for all 20 clips

**Done tasks:**
- ✅ Modified `m10_sam_segment.py`: DINO box detection + SAM 3.1 text+boxes hybrid (Path D)
- ✅ Added `iopath`, `ftfy` to `requirements_gpu.txt` (SAM3 `--no-deps` undeclared deps)
- ✅ Added `load_dotenv()` to m10 (HF_HOME, HF_TOKEN propagation)
- ✅ Fixed transformers 4.57 API renames (`box_threshold`→`threshold`, `labels`→`text_labels`)
- ✅ Fixed SAM 3.1 box format (xyxy→normalized xywh, paired box_labels=[1]*N)
- ✅ Tuned thresholds Option C: DINO box=0.15, text=0.12; m11 min_agent_area_pct=0.003
- ✅ Updated `runbook.md` paths: `outputs/sanity/factors/` → `outputs/sanity/m10_sam_segment/` etc.
- ✅ Fixed taxonomy: 17 agent categories in `ch11_surgery.yaml` (not VLM tags) for accuracy
- ✅ Added step [9/9] to `setup_env_uv.sh` to pre-cache Grounding DINO weights

**Documented in `errors_N_fixes.md` entries #18-27.**

---

## 🏗️ Current m10/m11 Architecture (2026-04-14, Level 2)

**Pipeline:** Grounded-SAM (DINO detection + SAM 3.1 mask refinement + tracking), multi-anchor re-seed.

| Stage | Tool | Frames | Purpose |
|---|---|---|---|
| Detection | Grounding DINO base (17-cat taxonomy) | Anchors `[0, 4, 8, 12]` | Open-vocab text→box on frame 0 of each 4-frame segment |
| Refinement + tracking | SAM 3.1 multiplex, text+boxes hybrid | Per anchor, segment `[a..a+3]` | `add_prompt(frame_index=anchor, text=category, boxes=...)` → propagate within 4 frames |
| D_I mining | `mine_interactions()` geometry | Full 16 frames | Centroid pair runs ≥4 frames within 20% frame width |

**Why multi-anchor (Level 2)**: single-anchor drifted 25%→5% by mid-frame; 4 anchors cap drift at ≤2 frames. 4× runtime for +334% agents / +51% D_I. See `errors_N_fixes.md` #32.

**D_I tube builder (2026-04-14)**: m10 now saves `per_object_bboxes_json` (~5 KB/clip); m11 prefers tight union-bbox crops over fixed 30% centroid squares (graceful fallback for legacy .npz).

### 📊 POC dense100 measurements — Level 1 → Level 2 → **v2_HF + bbox-tubes** (current, 2026-04-15)

| Metric | Level 1 (1 anchor, raw sam3) | Level 2 (4 anchors, raw sam3) | **v2_HF (4 anchors, HF Sam3Tracker + bbox tubes)** |
|---|---:|---:|---:|
| m10 throughput | 12.86 s/clip | 46.38 s/clip | **11.02 s/clip** ✅ |
| m10 total agents | 1286 | 5581 | **6146** |
| m10 D_I interactions | 1759 | 2659 | **8723** |
| m10 mean pixel_ratio | 17.42% | 21.66% | 18.61% (tighter masks) |
| m11 D_L / D_A / D_I present | — | 100 / 93 / 88 | **100 / 94 / 91** |
| m11 total tubes | — | 2659 (fixed 30 % squares) | **8723 (5659 unique bbox shapes)** |
| m11 median tubes/clip | — | 21.5 | **65** |
| 115K ETA (24GB, m10 only) | ~15 days | ~61 days | **~14.7 days** |
| 115K ETA (96GB, batch ×4) | — | — | **~3.7 days** |

Sources: `logs/m10_v2HF_dense100_probe5_v5.log`, `logs/m11_dense100_level2_v5.log` (m11 over v2_HF masks).
Verdict: Path B achieved 4.21× speedup AND +228 % D_I tubes AND tighter agent masks — a clean Pareto win.

**Decision log:**
- ✅ 17-category agent taxonomy in `configs/train/ch11_surgery.yaml > factor_datasets.grounding_dino.agent_taxonomy` (fixed, not per-clip VLM tags) — accuracy-first for D_L/D_A/D_I
- ✅ Option C thresholds: DINO `box=0.15, text=0.12` (aggressive recall), m11 `min_agent_area_pct=0.003`
- ✅ Path D text+boxes hybrid in SAM 3.1 `add_prompt` (not boxes-only which drops tracking)
- ✅ Box clamp before xywh-normalization (errors #25, #28)
- ✅ Guards on empty add_prompt output (#30) + SAM 3.1 state inconsistency (#31)
- ✅ `verify_or_skip` completeness check (#29) — partial output now resumes instead of skipping

---

## ✅ Done (iter10 Phase 2c — v15a/v15b archived, v15c WITHDRAWN, paired_eval complete)

### ✅ iter9 v10/v13/v14 — prior results (archived)

| Run | Completed | Frozen | Surgical | Δ (test_500) | Verdict |
|---|---|---|---|---|---|
| v10 | 2026-04-20 | 27.83 ±2.35 | 27.97 ±2.35 | 🟡 **+0.14 pp** | SATURATED (LR 1e-6 too low, grad_norm decayed 15×) |
| v13 | 2026-04-21 | 29.93 ±2.38 | 29.93 ±2.37 | 🟡 **0.00 pp** | FLAT (H5 plateau bug killed S2 at step 1 — D_A untested) |
| v14 | 2026-04-21 | 29.93 proj | ~30.06 proj | 🟡 **~+0.13 pp** | FLAT (plateau-fix #79 worked, but D_A carried almost no signal) |

**v13 → v14 handoff insight**: H2 stratified splits collapsed val→test gap 2.36 pp → 0.33 pp ✅. Plateau-fix #79 per-stage buffer reset validated. But D_L+D_A at 10K-scale ≈ pretrained baseline.

### ✅ iter10 v15a/v15b results (completed 2026-04-21 → 2026-04-22)

Three-way overnight ran `scripts/run_iter10_overnight.sh`. Reality diverged from the original TODO plan — split each hypothesis into its OWN run (v15a/b/c) rather than combining.

| Run | Keyword | Config diff vs v14 | Frozen | Surgical | Δ | Verdict |
|---|---|---|---|---|---|---|
| **v15a** | more-laps | `max_epochs: 1 → 3` (894 steps planned) | 29.87 | 29.87 | 🔴 **0.00 pp** | REJECTED — early-stopped at 609/894 via real S2 plateau; best val 29.87 @ S1 step 232 then drifted down. val_jepa still dropped 3.6% but retrieval decoupled. 3× epochs = no retrieval signal. |
| **v15b** | louder-agent | S2 `L:0.30→0.15 A:0.70→0.85` | 29.93 | 29.83 | 🔴 **−0.10 pp** ❌ | REJECTED *and reverse-effect*. 15% L-replay sits below CLEAR 2018's 20% noise floor → lost anti-forgetting cushion → Stage 2 drifted down from peak 29.83 @ step 174 to 29.47 @ step 290. 30% L-replay wasn't dilution; it was scaffolding. |

**Note on "FAILED" labels**: `run_iter10_overnight.sh` logged v15a + v15b as exit=2 due to the `ls` verify bug (errors_N_fixes #71) — but m09c/m05/m06/m08b ran cleanly. Results archived as `outputs_versioned/v15{a,b}_FAILED_*` → renamed to `v15{a,b}_*` post-fix.

### 🗑️ iter10 v15c — WITHDRAWN

v15c was invalidated by `errors_N_fixes.md #73` (silent `StreamingFactorDataset` L/A renorm: yaml requested `{L:0.15, A:0.15, I:0.70}` but empty D_I list dropped I silently → actual `{L:0.50, A:0.50}`). Ckpt deleted; v15c *recipe* replaced by iter11 `surgery_3stage_DI.yaml` (post-rename) with fail-loud per-factor preflight from #73 fix.

### ✅ iter10 paired_eval_10k (N=9,297, CI_half ±0.42 pp) — complete for v10/v13/v14/v15a/v15b

All 5 non-withdrawn iter10 variants cluster at Δ Prec@K ≈ 0 on eval_10k (see `iter/utils/experiment_log.md` cross-run table). None cleared the +3 pp gate. v14 remained leader-among-narrow-taxonomy (Δ=+0.07/+0.13 on test_500 at N=500) but Δ ≈ 0 with p ≥ 0.68 under paired BCa at N=9,297.

### 🚀 iter11 v2 — next active run (4-variant apples-to-apples at 10K)

Design lives in `iter/iter11/plan_code_dev.md`. Key changes from iter10:
- 🎯 **Unified 5-epoch budget** across all 4 variants (yamls already edited: `max_epochs.full: 5`, `saves_per_epoch: 5`)
- 🧹 **Only val-loss plateau trigger active** (kill_switch/prec_plateau/bwt all disabled — each fires below CI noise floor, see `feedback_only_val_loss_early_stop.md`)
- 🏷️ **Semantic yaml naming**: `surgery_2stage_noDI` / `surgery_2stage_loud_agent` / `surgery_3stage_DI` + `explora`
- 📂 **Per-config output dir** (`outputs/full/<config_name>/`, no more `outputs_versioned/<tag>_m09c_surgery/` archive shuffle)
- 🚀 **New thin wrappers**: `scripts/train.sh` (4 trainings) + `scripts/eval.sh` (adapted from `run_paired_eval_10k.sh`)
- ⏱️ **Budget**: ~30-60 h train + ~10 h eval ≈ ~$32-56 GPU

### 🔒 Conditional escalation (if iter11 v2 also saturates)

| Result at iter11 v2 | Interpretation | Next action |
|---|---|---|
| Any variant Δ ≥ +3 pp on eval_10k (paired BCa CI_lo > 0) | ✅ publishable recipe | Ship + ExPLoRA ablation + 50K ladder |
| Best Δ ∈ [+0.3, +3) pp | 🟡 marginal signal | 50K scale-up with leader recipe (~52 h / ~$42) |
| All Δ < +0.3 pp on eval_10k | 🔴 10K dataset-limited | **Concede tier**: "layout-factor surgery at NOISE FLOOR" paper pitch; D_L/D_A/D_I → ablation table; BWT ≈ 0 across scales becomes headline |

### 🔒 v12-era Meta-paper audit queue (deferred, not blocking v14-v16)

Source: 2026-04-20 audit of `configs/train_2_1/vitG16/*.yaml` (Meta official). These are paper-methodology fixes, not expected to unlock Δ ≥ 3 pp on their own — queue behind the D_A signal test.

| # | Keyword | Intervention | Status |
|---|---|---|---|
| — | **no-rewarmup** | Single scheduler spans all stages (build LambdaLR before stage loop) | 🔒 queued |
| — | **linear-decay** | `lr_schedule: constant → linear_decay` (start 1e-5 → final 1e-6), matches Meta cooldown | 🔒 queued |
| — | **clean-yaml** | Delete dead `warmup_steps=500` + `warmup_cap_pct=10` from base_optimization.yaml (surgery uses `surgery.warmup_pct` instead) | 🔒 queued |
| — | **unfreeze-sweep** | 1D coordinate sweep on n₁:n₂=1:2 diagonal `{0.20/0.40, 0.25/0.50, 0.30/0.60}` — publishable ablation | 🔒 queued |

### ✅ v11/v13 interventions — completed 2026-04-20 → 2026-04-21

| # | Keyword | Status | Outcome |
|---|---|---|---|
| **lr-up** | lr 1e-6 → 1e-5 (linear-BS-scaled from Meta 6e-4) | ✅ landed yaml | Did not degrade Prec@K but did not lift it either |
| **dino-tight** | DINO box 0.15→0.20, text 0.12→0.18 | ✅ landed yaml + m10 rerun | n_agents −28 %, recall 0.655 → 0.632, mask_conf +0.002 |
| **stratified-splits** | val_500/test_500 by (city × tour_type, seed=42) | ✅ landed 2026-04-20 | val→test gap 2.36 pp → 0.33 pp ✅ (worked) |
| **prec-plateau-kill** | Prec@K plateau trigger (min_delta=0.3pp, patience=5) | ✅ landed m09c | Fired correctly BUT premature (v13 bug) → `plateau-fix` |
| **typed-interactions** | `obj_id → cat` persistence + D_I cat-pair filter | ✅ landed #77 | Dormant under iter9 (D_I disabled), ready for v16 |
| **robust-probe-reader** | `utils/probe_history.py:read_probe_history_robust()` | ✅ landed | Uses `training_summary.json.probe_history` canonical path |
| **plot-polish** | Bar-chart Easy-only + auto-Y + CI visible · 3-panel probe_trajectory · radar-skip when n<3 | ✅ landed #78 | paper-ready visualizations |

### ✅ Streaming refactor + val/test split landed pre-v10 (2026-04-19 → 2026-04-20)

### ✅ 1K POC complete (2026-04-19 morning→evening) — Surgery ≈ Frozen on N=100 val-split

- ✅ Step A (m10 1K): 1000/1000 in 1h00m20s, quality_gate PASS
- ✅ Step B (m11 1K): 1000/1000 factor-gen in 11m43s
- ✅ Step C (m09c Surgery v3, 3-stage, 5 ep): 139 steps / 2h27m; **best Prec@K=20.50 @ step 12 (Stage 1)**; BWT = −0.33 (Stage 3 hurt); kill-switch not triggered (max_drop=0.17 pp < 5 pp floor); val_jepa 0.4854 → 0.4749 (−2.2 %).
- ✅ Steps D/E (m05 frozen + surgical on 100-val hold-out): 1m58s + 4m45s
- ❌ **Step F decision gate: FAILED** — Frozen 20.17 ±4.5 pp vs Surgical 20.33 ±4.67 pp → Δ +0.17 pp, CIs overlap. Surgery ≈ Frozen at N=100 (CI too wide to resolve sub-pp delta).

### 🩺 Diagnosis — Stage 3 is net-useless at 1K scale

Multi-signal agreement (see `experiment_log.md` 2026-04-19 entry for numbers):
1. **val_jepa Δ/step**: Stage 1 −0.5 %, Stage 2 −1.4 %, Stage 3 **−0.3 %** (4× slower).
2. **Cycle@K drift**: 63 → 62 → 61/62 (Stage 3 never re-gains).
3. **Best-ckpt events**: Stage 1 = 1 new-max, Stage 2 = 0, **Stage 3 = 0** in 27 probes.
4. **BWT = −0.33** (net-negative on held-out Prec@K).

### ✅ Code landed 2026-04-19 post-POC — 2-stage recipe + early-stop suite

- **yaml** (`ch11_surgery.yaml`): Stage 3 removed (stages=[L, A]); Stage 2 replay 10 % → **30 %** D_L (closer to CLEAR 50/50); `max_epochs.{sanity,poc,full}: 1` (scale DATASET, not epochs); `batch_size: 32` research-LOCKED (comment updated); new probe keys: `plateau_enabled`/`plateau_min_delta=1e-3`/`plateau_patience=5`, `bwt_trigger_enabled`/`bwt_tolerance_pct=0.5`/`bwt_patience=10`, `use_permanent_val` (FULL=true → val = permanent `data/val_1k`, no internal split).
- **`src/m09c_surgery.py`**: `create_train_val_split` supports `use_permanent_val` branch (fail-loud on overlap), 3 early-stop triggers wired into `_run_probe_at_step`, unified `kill_state.reason` for catastrophic_forgetting / val_loss_plateau / negative_bwt, `training_summary.json.early_stop` block.
- **`src/utils/plots.py`**: `m09_val_loss.png` skipped when no val data in loss_log (fixes overwrite of probe-based val plot); `m09_train_loss.png` gains dual x-axis (bottom = optimizer step, top = `n_unique × n_epochs` training-samples-seen).
- **`src/m09c_surgery.py` live-plot helper**: `_render_live_plots()` refreshes trajectory/forgetting/val_loss PNGs every probe (silent during run, verbose at end).
- **`src/m06_faiss_metrics.py`**: reads embeddings from `outputs/<mode>/m05_vjepa_embed/`, `--local-data` flag for tags path (pre-existing bug #64 where m06 looked in mode-root not module-dir).
- Status md at `iter/iter8/status_1k_poc_run.md` captures the full 1K POC run (numbers + plots + 3 sample verify videos).

### ✅ Code landed 2026-04-19 → 2026-04-20 pre-launch

**Streaming refactor** (9 files, ~1014 LoC incl. tests) — see `iter/iter9/plan_code_dev.md`:
- `src/utils/factor_streaming.py` NEW — `stream_factor()` + `tensor_from_factor_array()`
- `src/utils/training.py` — `StreamingFactorDataset(IterableDataset)` + `build_streaming_indices` + `_streaming_worker_init` + legacy `load_factor_clip` refactored to share normalization
- `src/m11_factor_datasets.py` — `--streaming` flag short-circuits factor-gen for non-verify clips (~90 % m11 wall reduction)
- `src/m09c_surgery.py` — DataLoader wire-in (`num_workers=16, persistent_workers, prefetch_factor=4, pin_memory`) + `--[no-]factor-streaming` CLI override
- `configs/train/ch11_surgery.yaml` — `factor_streaming:` block (sanity=F, poc=F, full=T)
- `scripts/tests_streaming/test_parity.py` → **10/10 bitwise PASSED** on iter8 POC D_L+D_A .npy (5 clips × 2 factors, `np.array_equal` uint8)
- `scripts/tests_streaming/test_sanity_end_to_end.sh` → **4/4 PASSED** (CLI override, TAR scan, DataLoader @ nw={0,2}, legacy path)
- `scripts/tests_streaming/test_wall_time.py` → **4.77 h projected 10K × 1 ep @ nw=16**, well under 10 h budget

**Val/Test split policy** (methodology fix vs iter8):
- `data/val_500.json` (probe) + `data/test_500.json` (gate) — disjoint, seed=42, both share `data/val_1k_local/` TARs
- Fixes best-of-K selection bias (~2.2 pp at K=50 probes, σ=0.77 pp at N=1000)
- CI half-width widens ±1.5 pp → ±2.1 pp (still < 3 pp gate threshold at N=500)

**`scripts/run_iter9_10k.sh`** (173 LoC, ~13 h overnight wrapper) — runbook A→F verbatim + assertion-based verify blocks after every step. Launched in tmux `terminal1` 2026-04-20 00:43.

**Disk cleanup (freed ~123 GB before launch):**
- `rm -rf outputs/poc/` (67 GB) — iter8 1K POC bulk; archived to HF as `anonymousML123/factorjepa-outputs` pre-delete
- `rm -rf iter/iter8/outputs/poc/m11_factor_datasets/D_{L,A,I}` (56 GB) — .npy bulk permanently retired by streaming
- `git gc --prune=now --aggressive` — compacted `.git/objects` pack

### ✅ iter10 gate criteria — CLOSED

All iter10 10K gates closed: Δ ≈ 0 pp on eval_10k for v10/v13/v14/v15a/v15b (v15c withdrawn). Decision was "pivot to iter11 v2 with unified budget + disabled-noise-triggers" rather than 50K-scale jump; 50K deferred to post-iter11-v2 escalation.

### ✅ ExPLoRA arm landed (iter11 #step3)

m09b_explora.py ported probe infra from m09c (shared `render_training_plots` in `utils/training.py`) + `val_split.json` writer + `--cache-policy` gate. Variant now part of `scripts/eval.sh` (and previously `run_paired_eval_10k.sh:L72`).

### 📈 Step H: Scale-ladder — plateau-seeking 50K → 115K (conditional on 10K gate-pass)

> **Why laddered, not direct-to-115K**: each scale × val_1k Prec@K = a datapoint on a **scaling curve** (n_clips vs Prec@K with 95 % CIs) = publishable figure. Plateau/BWT early-stop auto-halts GPU spend when signal saturates, so worst-case ≈ best-case when training flattens.
> **Reuse rule**: each scale is a fresh A→F pipeline. m10/m11 outputs are clip-keyed to the subset manifest → CANNOT reuse 10K artifacts at 50K+.

**Gate rules (auto-escalation from 10K Step F result):**

| 10K Δ (surgery − frozen, val_1k) | Action |
|---|---|
| ≥ 2 pp (non-overlap CI) | → H.1 (50K) |
| ∈ [0.5, 2) pp (marginal) | → BWT Option B first, re-run 10K; decide ladder on 2nd result |
| < 0.5 pp (saturated) | → 10K is publishable tier; skip ladder; Step G (ExPLoRA) only |
| ≤ 0 pp | → BWT Options B → C; no scale spend |

**Per-tier requirements & budget:**

| Tier | Subset file | Local-data dir (HF pull) | Disk ≥ (post-streaming) | m10 | m11 | m09c (1 ep / BS=32) | m05+m06 on test_500 | Total GPU | Gate to next |
|---|---|---|---|---|---|---|---|---|---|
| H.1 50K | `data/subset_50k.json` | `data/subset_50k_local` (~55 GB, ~2 h) | **250 GB** (was 3 TB) | ~38 h | ~30 min | ~12.5 h (1562 steps) | ~40 min | **~52 h** (was ~68 h) | Δ ≥ 0.5 pp vs 10K → H.2 |
| H.2 115K | `data/subset_115k.json` | `data/full_local` (already in repo) | **500 GB** (was 5 TB) | ~86 h (96 GB batch×4 ≈ 58 h) | ~1 h | ~29 h (3594 steps) | ~40 min | **~117 h** (was ~98 h — streaming kills disk blocker, not m10 wall) | Δ ≥ 0.5 pp vs 50K → paper headline |

Plateau/BWT early-stop likely halts m09c well before full budget at 50K+ (signal saturates on curve knee). Budget row = worst case.

**Blocker trade-off at 115K**: ✅ RESOLVED 2026-04-19 — streaming refactor landed, cuts disk need 5 TB → 500 GB. 115K now fits on same instance tier as 10K.

**Scaling-curve figure (triggered by any H-completion):**

```python
# 4 data points (1k, 10k, 50k, 115k) × 2 arms (frozen, surgical) with BCa 95% CIs
# Plot Prec@K vs log10(n_clips); annotate plateau knee; non-overlapping CI = win
```

**Publishable tiers (paper narrative):**
- 1K POC → "method works but noisy" (iter8 artifact, archived)
- 10K → smallest scale where signal resolves (CI ±1.5 pp)
- 50K → intermediate curve point
- 115K → asymptotic headline number (only if signal still climbing past 50K)



### 🩹 Backward Transfer (BWT to measure catastrophic forgetting) resolution queue — contingent on 10K run result

> 📖 **BWT defined**: Backward Transfer measures catastrophic forgetting in model training by evaluating the **drop in accuracy on earlier tasks after the model learns new, subsequent tasks**. A **negative** BWT = the model forgot previous knowledge (interference); a **positive** BWT = it benefited from earlier learning. Our implementation: `BWT_t = Prec@K[current_probe] − Prec@K[first_probe]` in `src/m09c_surgery.py:_run_probe_at_step`. See also `training_summary.json.probe_trajectory_stats.bwt_prec_at_k`.

**Root cause (identified 2026-04-19)**: two mechanisms drove BWT=−0.33 on 1K POC:
1. **Re-unfreezing + dilute replay** — Stage 2/3 unlock Meta-pretrained layers (13-24, 25-36); old 10-15 % replay fraction sat below noise, new-factor gradients drowned old-factor retention.
2. **Weak drift anchor** — `drift_control.lambda_reg: 1.0` with `l2_uniform` produces drift penalty ≈ 0.01, JEPA gradient ≈ 0.48 → 100× mismatch. L2 anchor barely tugs weights back.

**Fix options (queued, NOT yet applied):**

| Option | Pros | Cons |
|---|---|---|
| **A** ✅ already applied (2026-04-19): drop Stage 3 + replay 10 → 30 % D_L | Removes biggest BWT source + cushions Stage 2; 0 code change | Doesn't fix the underlying weak drift anchor |
| **B** ⬜ bump `drift_control.lambda_reg: 1.0 → 50` (yaml-only) | Literature precedent — EWC uses λ=10²–10⁹ (plan_training.md:458 row 11, ref arXiv 2210.16365, 2603.18596); 1-line edit; scales drift penalty to match JEPA gradient | Too-strong anchor → model can't learn; need to re-tune LR; may zero out improvement too |
| **C** ⬜ switch `l2_uniform` → EWC-weighted L2 (FIM-based) | Gold standard (EWC Done Right, arXiv 2603.18596); penalizes only "important" pretrained weights, lets noise drift freely | 2-3 h to implement + 30 min pre-training FIM computation per run; adds code complexity |

**Trigger rule (auto-escalation based on 10K gate outcome):**
- 10K run plateau/BWT kill-switch **does not fire** AND Surgery > Frozen at N=1000 (non-overlapping CI) → **no fix needed**, scale to 50K/115K with A only.
- 10K BWT still < −0.5 pp (early-stop #3 triggers) OR Surgery ≤ Frozen with flat val_jepa → **apply B** (λ=50), re-run 10K.
- 10K with B still negative-BWT, OR B over-penalizes (val_jepa stuck flat from step 1) → **apply C** (EWC-weighted), re-run 10K. Budget: ~2-3 h impl + 30 min FIM pre-compute + ~3 h 10K training.

---

### ✅ m09c SANITY training — RESOLVED 2026-04-17 on 96GB Blackwell

**Iterations:** v0 → v7 on 24GB, v8-hw on 96GB. Errors & fixes catalogued in `errors_N_fixes.md` #50-#58.

| Version | Furthest point reached | Blocker | Fixed by |
|---|---|---|---|
| v0 | `build_model` | `KeyError 'src.models.predictor'` | #50 (vjepa2_imports finally-block restores all saved_modules) |
| v1 | `build_model:125` | `KeyError 'patch_size'` on cfg["data"] | #51 (sed `data_cfg[X]` → `model_cfg[X]` for crop_size/patch_size/tubelet_size) |
| v2 | `train_surgery:351` | `TypeError 'int' not subscriptable` on max_epochs | #52 (drop redundant `[mode_key]` — merge already flattened) |
| v3 | Stage 1 step 0 | OOM at first forward (2.25 GiB need, 1.87 free on 24GB) | #53 (AdaptiveBatchSizer + `_train_step_grad_accum` wired + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) |
| v4 | Stage 1 summary | `UnboundLocalError jepa_val` (step OOMed → loop ended before any value assigned) | #54 (pre-init loss vars per-stage before inner for-loop) |
| v5 | All 3 stages printed "complete" | Silent fail: 0 successful steps, exported unmodified student | #55 (within-step retry on OOM + fail-hard when sizer at min) |
| v6 | Stages 1+2 ✅ (loss=0.4841, 0.4874), Stage 3 OOM | Stage 3: 36/48 trainable blocks — fp32 master (5.5 GB) + 8-bit m1/m2 (3 GB) + model + activations overflowed 24 GB | #56 (grad checkpointing + bnb AdamW8bit) + #57 (mode-gated yaml: savers ON for SANITY, OFF for POC/FULL) |
| v7 | Stages 1+2 ✅, Stage 3 OOM at min sub-batch=1 | fp32 master + 8-bit m1/m2 + activation spike exceeded 24GB even with PagedAdamW8bit | #58 (inter-stage cleanup + PagedAdamW8bit — helped but didn't close the gap on 24GB) |
| ✅ **v8-hw (2026-04-17 noon)** | **ALL 3 STAGES PASS on 96GB** (Stage 1=0.4870, Stage 2=0.4901, **Stage 3=0.4806**), student_encoder.pt exported | — (resolved by 96GB hardware migration; post-cleanup VRAM 19.9 GB / 102 GB after Stage 2) | 🏁 Hardware upgrade; "v8 teacher CPU offload" patch from #58's follow-up plan NOT needed. |
| 🐛 **v9 (2026-04-17 dinner-time POC)** | D.2 POC 100-dense "SURGERY COMPLETE" in 60 s — but only 3 optimizer steps total 😱 | `max_epochs.poc: 1` (yaml comment "1 epoch per stage" ≠ code "1 epoch total"). `stage_steps = int(3 × 0.33) = 0` → clamped to 1/stage. Silent near-no-op. | #60 — bumped `max_epochs.poc: 1 → 100`, yaml comment corrected. |
| 🐛 **v10 (2026-04-17 late POC)** | D.2 POC 100-dense 300 steps completed — loss dropped only 0.50 → 0.476 (∆ −5.4 %) 📉 | `warmup_steps: 200` per stage > `stage_steps: 99` → LR never reached target (~50 % max). Fresh scheduler per stage restarts warmup from 0. | #61 — replaced fixed `warmup_steps` with `warmup_pct: 0.20` in yaml, auto-scales to 19/41/718 for 100/1K/115K. |
| 🎯 **v11 (1K val_1k POC, NEXT)** | (pending) | — | — |

**Closure:** SANITY loop resolved via hardware (v7→v8). POC loop debugged via config-fix cascade (#60 → #61). Real Prec@K signal will come from v11 at 1K scale.

### 🛑 v7 Stage 3 OOM — detailed root cause

**What v7 did accomplish:** `PagedAdamW8bit` (unified-memory CPU paging) wired in via `#58`; inter-stage optimizer cleanup (`None`-ref + `empty_cache` + `ipc_collect`) working. Stage 2 → Stage 3 transition log shows `17.7 GB used / 25.2 GB total after releasing Stage 2 state` — cleanup IS happening correctly. Stage 3's optimizer built successfully. Forward at sub-batch=1 immediately OOMed.

**Memory accounting for Stage 3 on 24 GB (actual, from v7 log):**

| Component | Size | Cumulative |
|---|---:|---:|
| CUDA context + PyTorch reserved | ~2.0 GB | 2.0 GB |
| Student fp16 (48 blocks × 2B params) | 3.7 GB | 5.7 GB |
| Teacher fp16 (frozen EMA copy) | 3.7 GB | 9.4 GB |
| Predictor fp16 (60M) | 0.12 GB | 9.5 GB |
| Mixed-precision buffers, mask generators, etc. | ~0.5 GB | 10.0 GB |
| **Post-cleanup baseline ← matches log "17.7 GB"** (7.7 GB extra accounted for by partial optimizer pre-allocation) | — | 17.7 GB |
| Stage 3 optimizer: fp32 master (1.38B × 4 bytes) | 5.5 GB | — |
| Stage 3 optimizer: 8-bit m1+m2 (quantized) | ~3.0 GB | — |
| PagedAdamW8bit pages SOME of above to CPU RAM | -3 to -5 GB | ~20 GB after build |
| **Forward spike at sub-batch=1** (activations + grads + intermediate) | **+5-6 GB** | **~25-26 GB → OOM** |

**Why PagedAdamW8bit alone wasn't enough:** paging is *reactive* (pages on pressure, not proactively). The forward-pass spike is fast — allocator requests activation tensors, there's no free slot, it returns OOM *before* paging can swap anything out. Paging helps steady-state memory but not burst peaks.

### ✅ Proposed fix for v8 — teacher CPU offload

**Idea:** Teacher is 3.7 GB permanently resident on GPU but only *used* during a single `torch.no_grad()` forward per sub-batch (inside `_train_step_grad_accum`). For 95%+ of each step the teacher sits idle consuming premium GPU memory. Move teacher to CPU by default, swap to GPU only for that one forward call, move back to CPU after.

**Expected memory freed:** 3.7 GB on GPU permanently.
**Expected throughput hit:** ~200-500ms per sub-batch for 3.7 GB PCIe transfer (× 2: to-GPU then back). At SANITY's scale (1 step per stage), ~1 second added per step total. Negligible.
**Accuracy impact:** ZERO — teacher runs at full fp16 precision on GPU during its forward; the only change is residency between calls.

**Implementation sketch** (surgery-only per scope discipline):
1. Add mode-gated yaml: `teacher_offload: {sanity: true, poc: false, full: false}` in `ch11_surgery.yaml` + flatten in `m09c_surgery.merge_config_with_args`.
2. In `_train_step_grad_accum` (or a surgery-only wrapper around it), if offload flag is True:
   ```python
   teacher = teacher.to(device, non_blocking=True)
   with torch.no_grad():
       h = teacher(bc)
   teacher = teacher.to("cpu", non_blocking=True)
   torch.cuda.synchronize()  # ensure copy-out completes before student forward
   ```
3. Store teacher on CPU after initial build in m09c `build_model` (conditional).
4. EMA update (`update_teacher_ema`) also needs teacher on GPU briefly — swap pattern same as above.

**Gold-standard reference:** HF Accelerate's `cpu_offload_with_hook` pattern, DeepSpeed ZeRO-Infinity's weight offloading, FAIR vissl's `param_offload_to_cpu`. All use the same "move to GPU on use, back on done" semantics.

**Expected v8 Stage 3 memory:** 17.7 GB - 3.7 GB teacher = 14.0 GB post-cleanup → +8.5 GB Stage 3 optimizer (partially paged) ≈ 20 GB → +5-6 GB forward spike ≈ 25-26 GB. Still tight. **May need a second fix.**

### 📋 Fallback fixes if teacher offload alone doesn't fit v8

Ordered by preference (least invasive first):

1. **Reduce teacher to fp32→bf16 master only + fp16 weights** (already in mixed precision, marginal)
2. **Disable teacher hierarchical output (4-level deep supervision) for SANITY Stage 3** — saves ~1.5 GB in teacher forward activations. Config flag `cfg[model][n_output_distillation]=1` for SANITY. Changes training loss slightly but SANITY is code-only.
3. **Reduce SANITY `num_frames` from 16 → 8** — halves token count, halves activation memory. Add mode-gate to `data.num_frames`. Clean yaml-only change.
4. **Predictor CPU offload** (predictor is small, 0.12 GB — low ROI)
5. **Gradient accumulation at sub-micro-batch level** — split sub-batch=1 forward into time-sliced chunks. Non-trivial code change.

### ✅ Status at 2026-04-17 (RESOLVED on 96 GB)

- All SANITY code paths validated end-to-end on 96 GB Blackwell: Stage 1 loss=0.4870, Stage 2 loss=0.4901, **Stage 3 loss=0.4806** (first successful measurement — first ever Stage 3 completion).
- Stage 3 peak post-cleanup VRAM: 19.9 GB / 102 GB — ~80 GB of headroom. No OOM, no fallback needed.
- **v8 teacher-CPU-offload patch was NOT landed** — prediction from 2026-04-15 ("may not fit even with offload") was superseded by the cheaper option (just run on 96 GB). Leaves code clean (no offload complexity to maintain) and matches plan_training's SANITY→POC hardware progression.
- **POC D.2 unblocked** — all savers will auto-flip OFF (mode-gated yaml #57) for clean fp32 AdamW training, the research-quality recipe for the Prec@K comparison.

---

## ⏩ SPEEDUP (TODO) for FULL (115K clips) mode

Central registry for speedups across `src/m*.py` + `scripts/*.sh`. Add one row per option. "Status" = ✓ done / 🔬 tested, failed / ⬜ pending.

Measured baseline on POC dense100 (46 s/clip with v5 forward-only): **115K naive projection = ~61 days on 24GB GPU**. Path A alone is NOT sufficient for FULL — Path B (or equivalent) is mandatory.

| Module | Path | Effort | Speedup | 115K ETA (24GB) | Status |
|---|---|---|---|---|---|
| m10 | **A. `propagation_direction="forward"`** (skip backward SAM3 call) | done | 1.83× measured | ~61 days | ✓ #35 (unblocks POC, not FULL) |
| m10 | **B. HF `Sam3TrackerVideoModel` (replaced `m10_sam_segment.py`)** — requires `transformers==5.5.4` | done | **4.21× measured** | **~14.7 days** | ✅ #36-#40 validated 2026-04-15 on dense100 |
| m10 | B+96GB. Path B + larger batch on 96GB GPU | +0h | ~4× on top | **~3.7 days** | ⬜ (preferred for FULL) |
| m10 | B'. P-3a probe: `Sam3VideoModel` text-only (stripped from m10 code, kept in git history) | post-paper | +dropping DINO ~2× | ~1.5 days | ⬜ (backlog — not on critical path) |
| m10 | C. Streaming mode (HF only) — disables hotstart heuristics (quality risk) | ~3h | 10× | ~6 days | ⬜ (not recommended) |
| m10 | D. Density-filter FULL to ~30-40K multi-agent clips only | ~30min | — | ~2-4 days with B+96GB | ⬜ (paper-valid if stratified) |
| m10 | — `max_frame_num_to_track=3` in raw sam3 pkg | tried | would be 10× | — | 🔬 #33/#35 (SAM3 bug: empty tensor, reverted) |

**Default plan**: POC dense100 finishes with Path A (validates Level 2 quality). Then implement Path B BEFORE FULL — at current speed, 115K is ~61 days on 24GB GPU. Path B+96GB drops FULL to ~1.5 days, which fits the deadline.

---

## 🔬 Phase 3: Ablations (if POC positive)

> **Paper-grade metric for every ablation below: downstream Prec@K from surgery training.** Upstream proxies (tube area, mask confidence, concept_recall) are useful for debugging but only Prec@K validates any upstream change. Bootstrap 95% CI mandatory (`utils/bootstrap.py`).

| Ablation | What it shows | GPU time |
|---|---|---|
| **A1: Stage contribution** | Stage 1 only, 1+2, 1+2+3 — does each stage add value? | 3 × ~40 min |
| **A2: Factor type** | D_L only, D_A only, D_I only — which factor matters most? | 3 × ~40 min |
| **A2b: `min_overlap_frames` 4 vs 8** | 4/16 (2659 tubes, noisier) vs 8/16 (~1000-1500, cleaner) — switch if +Prec@K | 2 × ~40 min |
| **A3: Surgery vs naive fine-tune** | Same layers unfrozen, raw clips (no factors) — is factoring the key? | ~40 min |
| **A4: Random seeds** | 3-5 seeds of best config — statistical significance | 3 × ~40 min |
| **A5: D_I tube crop type** | Centroid-30%-square vs tight-union-bbox (m10 `per_object_bboxes_json`) — does identity-aware cropping help? | 2 × ~40 min |

Priority if time-constrained: **A3** (proves factoring matters) then **A4** (NeurIPS rigor).

---

## 📋 Backlog

- ⬜ 🟡 Paper figures: per-clip segmentation samples (m08 CPU-only)
- ⬜ 🟡 Verification videos: MP4 with mask overlay for temporal consistency
- ⬜ 🟡 Output dir restructure: verify all cross-references after per-module migration
- ⬜ 🟡 **D_I gold-standard architecture** (post-deadline): current tight-union-bbox crop is still a POC shortcut. Gold standard (Social-Fabric ICCV'21, Video-HOI NeurIPS'22): per-agent tubelets + RoIAlign on scene features + pair transformer. Requires V-JEPA forward-pass change → out of NeurIPS scope, log for v2.
- ⬜ 🟡 **m11 GPU rewrite** (post-POC): replace `scipy.ndimage.gaussian_filter` (σ=15 blur, ~2.4s/clip CPU bottleneck) with `kornia.filters.gaussian_blur2d` on GPU — projects 3.5s/clip → ~5ms/clip compute, FULL 115K from 5.6d → ~10h. Needs kornia in requirements_gpu, ≥40dB PSNR diff-test vs scipy, `--gpu` CPU-optional flag.
- ✅ **m11 streaming refactor** (landed 2026-04-19, ~1014 LoC, 10/10 bitwise parity): `--streaming` flag + `StreamingFactorDataset(IterableDataset)` generates D_L/D_A on-demand from `(raw_mp4, mask.npz)` pairs. **10K disk: 380 GB → 52 GB · 50K: 1.7 TB → 150 GB · 115K: 4 TB → 345 GB** — full ladder fits on single 500 GB instance (no 3 TB / 5 TB rent needed). See `iter/iter9/plan_code_dev.md` for design + all 3 test tiers green.
- ⬜ 🟢 `hf_outputs.py` upload: `git_push.sh` doesn't `source .env`
- ⬜ 🟢 `setup_env_uv.sh`: cuML/SAM3 version ping-pong
- ⬜ 🟢 FA3 installation: only if SAM3 bottleneck on FULL
- ⬜ 🟢 `output_guard.py` absolute-path → repo-relative (errors_N_fixes.md #23): stops noisy HF 404 + URL-encoded `%2Fworkspace%2F...` log spam on every m10/m11 run.

---

## 🔧 Troubleshooting

| Problem | Fix | Time |
|---|---|---|
| Grounded-SAM box quality poor | Try YOLO-World + SAM instead | 2h |
| V-JEPA 2.1 shape mismatch | `state_dict.keys()` vs model params | 1h |
| LoRA target modules wrong | `print(model)` → find attn module names | 30 min |
| Surgery loss NaN | Lower LR, check grad norms | 1h |
| D_I: 0% clips have tubes | Lower `max_distance_frame_fraction` in YAML | 15 min |
| D_I: 100% clips have tubes | Raise `min_overlap_frames`, lower `tube_margin_pct` | 15 min |

---

## 🔮 Future (post-paper)

- ✅ Split m09 (2164 lines) → m09a_pretrain.py + m09b_explora.py + m09c_surgery.py + utils/training.py (2026-04-15, #49)
- WebDataset TARs for factor datasets (.npy won't scale to 115K)
- 6 interaction perturbations (tube jitter, margin random, raw/masked mixing)
- Patch shortcut sanity check (eval raw vs patched clips)
- Cooldown (64f) implementation

---

## ✅ Completed

### 2026-04-15 (~2h GPU): Path B speedup + bbox-tubes + m10 consolidation
- 5 bugs found & fixed: #37 DINO fp16 text-branch crash (fp32 default), #38 Sam3Tracker box depth=3 (not 4), #39 session.reset_tracking_data (not processor), #40 silent bug — object_score_logits not iou_scores, #41 add_text_prompt kwarg `text=` not `prompts=`
- transformers 4.57.6 → **5.5.4** (setup_env_uv.sh steps [9/10] DINO + [10/10] facebook/sam3 ~12 GB HF_TRANSFER parallel)
- HF `Sam3TrackerVideoModel` integrated; `max_frame_num_to_track` now works (raw sam3 pkg #33/#35 unfixable)
- m10 v2_HF merged back into `m10_sam_segment.py` (P-3a probe stripped); `train_surgery.sh` unchanged
- m11 D_I upgrade: `per_object_bboxes_json` saved by m10; `make_interaction_tubes_from_bboxes` replaces fixed 30% centroid square
- setup_env_uv.sh: added non-fatal `uv pip check` with allowlist (sam3/numpy, sam3/ftfy, torch/cuda-bindings, decord)
- preflight skill extended B16-B20 for transformers 5.x regression guards
- Measured on dense100: **11.02 s/clip (4.21× faster), 6146 agents (+10 %), 8723 D_I tubes (+228 %), 91 % clips have tubes**
- 115K FULL ETA: 61 days → **14.7 days on 24GB**, **3.7 days on 96GB+batch×4**

### 2026-04-14 (~5h GPU): Grounded-SAM Pivot + Level 2 multi-anchor
- 32 bugs found and fixed (see `errors_N_fixes.md` #18-32)
- Architecture pivot: SAM3-text-only → Grounded-SAM Path D (DINO + SAM3.1 text+boxes hybrid)
- **Level 2 upgrade**: single-anchor → multi-anchor DINO re-seed (4 anchors, drift capped at ≤2 frames)
- Fixed 17-cat agent taxonomy in `ch11_surgery.yaml` replacing per-clip VLM `notable_objects`
- DINO weights pre-cached via `setup_env_uv.sh` step [9/9] (~1.8 GB at HF_HOME)
- Density-scored 100-clip subset: `data/sanity_100_dense.json` (74 tier1 + 25 tier2)
- Top-20 2x2 MP4 video grid added to m11 for human eyeballing + website
- Verified D_L/D_A/D_I quality on 20 SANITY clips (39 D_I tubes from 9 clips)
- Tuned thresholds Option C for new mask distribution (recall-first)
- `verify_or_skip` completeness check fixed — partial runs now resume properly

### 2026-04-12/13 (~6h GPU): Initial SANITY infrastructure
- 17 initial bugs fixed (env, SAM3 integration, torchcodec SIGSEGV)
- Per-module output dirs: `outputs/{mode}/{module_name}/` for all m04-m11
- m10 overlay verification images + m11 2x2 per-clip grids implemented
- Composite quality gate (4 checks: pixel ratio, mask confidence, clips with agents)
- `--plot` flag on m10 and m11 for CPU-only plot regeneration

---

## ⏱️ Time Budget (updated 2026-04-24)

| Phase | Hours | Status |
|---|---|---|
| Phase 0-1: pivot + SANITY (24GB→96GB) | ~15h | ✅ |
| Phase 2a-2b: POC 100-dense + 1K val_1k (iter8) | ~8h | ✅ recipe landed |
| Phase 2c: iter9/iter10 10K × 6 variants + paired_eval | ~45h | ✅ all closed Δ ≈ 0 pp |
| Phase 2d: iter11 v1 (10K ExPLoRA, halted by prec-plateau at 174/298) | ~3h | 🗑️ INVALIDATED (trigger was below noise floor; ckpts deleted) |
| **Phase 2e (active): iter11 v2 — 4 variants × 5 epochs × only val-loss plateau** | **~30-60h train + ~10h eval projected** | 🟢 yamls edited; plan in `plan_code_dev.md`; train.sh/eval.sh pending |
| Phase 3a: 50K scale-up (conditional) | ~52h projected | 🔒 contingent on iter11 v2 Δ ≥ +0.3 pp on eval_10k |
| Phase 3b: 115K (conditional) | ~117h projected | 🔒 contingent on 50K Δ ≥ 0.5 pp |
| Phase 4: Paper writing | ~14h | ⬜ |
| Buffer | 2h | |

---

## 📁 Key Files

| File | What |
|---|---|
| `src/CLAUDE.md` | Codebase rules |
| `iter/iter11/plan_code_dev.md` | 🚀 LIVE iter11 v2 code-dev plan (rename + train.sh + eval.sh) |
| `iter/iter11/plan_training.md` | System design, architecture, literature |
| `iter/iter11/errors_N_fixes.md` | Bugs catalogued (iter8 → iter11) |
| `configs/train/ch11_surgery.yaml` (→ `surgery_2stage_noDI.yaml` post-rename) | Surgery base config |
| `configs/train/explora.yaml` | ExPLoRA config (LoRA rank=16 + 2 unfrozen blocks) |
| `configs/model/vjepa2_1.yaml` | V-JEPA 2.1 model config |
| `iter/utils/literarure_survey.md` | 24 JEPA variants (fallback) |
| `iter/utils/experiment_log.md` | POST-completion experiment results only (iter10 cross-run table) |
| `scripts/train.sh` (pending) / `scripts/eval.sh` (pending) | iter11 v2 thin wrappers |
