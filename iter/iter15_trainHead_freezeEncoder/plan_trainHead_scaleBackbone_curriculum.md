# 🚀 iter15 — Three-Pillar Plan: Head-Only Freeze ❄️ + Backbone Scaling 🏗️ + Curriculum Learning 📚

> 📅 **Date locked:** 2026-05-11 · **expanded** 2026-05-11 (added pillars 2 & 3)
> 🎯 **Paper goal:** `vjepa_surgery ≫ vjepa_pretrain ≫ vjepa_frozen` on motion / temporal features
> 🧑‍🔬 **Triggered by:** research lead directive (DINOv2 + Llama 3/4 author) — meeting notes 2026-05-10

---

## 🏗️ Three-pillar scope (+ Phase 0 prerequisite)

```
┌────────────────────────────────────┬──────────────────────────────────────────────────────────────────┐
│ Phase / Pillar                      │ What                                                              │
├────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 🧬 0️⃣ Phase 0 (PREREQ)              │ Extend m04d motion features 13-D → 23-D (foreground / camera-   │
│   📍 Was plan_phase5_fg_motion_*    │ subtracted). Powers BOTH richer motion-class labels AND data    │
│                                    │ curriculum difficulty axis. ~10 LoC, ~57 min m04d rerun.        │
├────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ ❄️ 1️⃣ Head-Only Freeze (dual track) │ Encoder + predictor frozen; only motion_aux head trains          │
│   📍 Meeting note §1                 │ m09a2_pretrain_head + m09c2_surgery_head (3stage_DI + noDI)     │
│                                    │ Linear/MLP eval protocol — Chen 2020 + Caron 2021 DINO §A.2      │
├────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 🏗️ 2️⃣ Backbone Scaling              │ ViT-G/14 (1.84B params, 1664-dim) vs ViT-H/14 (~632M, 1280-dim)  │
│   📍 Meeting note §1.1               │ Model-size sweep (NOT an ablation — no component removed)         │
│                                    │ Same head-only protocol applied to both backbones                │
├────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ 📚 3️⃣ Curriculum Learning           │ Parameter curriculum (Lee ICLR'23) — ALREADY in m09c1 stages    │
│   📍 Meeting note §2                 │ Data curriculum (Bengio 2009) — NEW · easy→hard FG motion mag   │
│                                    │ Uses Phase 0's vec[13] (fg_mean_mag, camera-subtracted)         │
└────────────────────────────────────┴──────────────────────────────────────────────────────────────────┘
```

> 📦 **Phase 0 was extracted from** `legacy/plan_phase5_fg_motion_features.md` (now archived). Its content is fully integrated below — Option A (FG-only, +10 LoC) is the minimum viable scope. Options B/C/D/E/F (acceleration, multi-scale, FG/BG via SAM3, etc.) are deferred.

---

## 📋 Context — why this iter exists

```
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🧪 Empirical trigger (iter14 R6 stage 0)                                                       │
├───────────────────────────────────────────────────────────────────────────────────────────────┤
│   Frozen encoder + 28-step motion_aux head training on factor data → top-1 = 0.8382           │
│   R1 full surgery (all 5 anti-forgetting mechanisms ON)         → top-1 = 0.8456              │
│   Δ                                                              = +0.74 pp = 1 clip @ N=136  │
│   → Most of R1's gain is recoverable WITHOUT encoder updates ✨                                │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 🎬 Decision (user-approved **Option C — dual track**)

Don't pivot, **add**. Preserve current encoder-update scripts AND add head-only siblings. Then eval ALL variants in one pass. The dual track yields a clean **3×2 ablation grid**:

```
                    ┌─────────────────────┬─────────────────────┐
                    │ 🔥 ENCODER-UPDATE   │ ❄️ HEAD-ONLY         │
┌───────────────────┼─────────────────────┼─────────────────────┤
│ 🆎 frozen baseline│ vjepa_2_1_frozen    │ vjepa_2_1_frozen    │
│ 🥈 pretrain       │ vjepa_2_1_pretrain  │ pretrain_head 🆕     │
│ 🥇 surgery        │ surgical_3stage_DI  │ surgical_*_head 🆕   │
└───────────────────┴─────────────────────┴─────────────────────┘
```

### 🔒 Locked decisions (from user clarification 2026-05-11)

| # | Decision | Reason |
|---|----------|--------|
| 1 | 🧊 **STRICT predictor freeze** — encoder AND predictor both frozen | Closest to DINOv2 protocol; ZERO catastrophic-forgetting risk |
| 2 | 🔮 **Build standalone `probe_future_regress.py`** | Frozen predictor makes `probe_future_mse.py` constant across head variants — need a head-trained regressor |
| 3 | 🔬 **m09c2 covers both `3stage_DI_head` AND `noDI_head`** | Tests whether D_I (interaction tubes) carries signal beyond D_L + D_A — addresses uncertainty about agent-extraction quality |

---

## 🎯 Goal — 13-variant comparison in a single eval pass

```
┌─────────────────────────────────────────────┬─────────────────────────┬───────────────────────────────┬──────────────┐
│ 🏷️ Variant                                  │ 🧬 Encoder source        │ 🛤️ Track                       │ 🏗️ Backbone   │
├─────────────────────────────────────────────┼─────────────────────────┼───────────────────────────────┼──────────────┤
│ vjepa_2_1_frozen                             │ Meta ViT-G direct       │ 🆎 Baseline                   │ ViT-G/14     │
│ vjepa_2_1_pretrain                           │ m09a1 output            │ 🔥 Encoder-update · cont. SSL │ ViT-G/14     │
│ vjepa_2_1_pretrain_2X                        │ m09a1 · 10 epochs       │ 🔥 Causal control             │ ViT-G/14     │
│ vjepa_2_1_pretrain_head 🆕                   │ m09a2 (≡ Meta)          │ ❄️ Head-only · raw            │ ViT-G/14     │
│ vjepa_2_1_pretrain_head_curriculum 🆕        │ m09a2 + data curriculum │ ❄️ Head-only · easy→hard      │ ViT-G/14     │
│ vjepa_2_1_surgical_3stage_DI                 │ m09c1 (3 stages)        │ 🔥 Encoder-update · surgery   │ ViT-G/14     │
│ vjepa_2_1_surgical_3stage_DI_head 🆕         │ m09c2 (1 stage)         │ ❄️ Head-only · factor-aug     │ ViT-G/14     │
│ vjepa_2_1_surgical_3stage_DI_head_curr 🆕    │ m09c2 + data curriculum │ ❄️ Head-only · factor + curr  │ ViT-G/14     │
│ vjepa_2_1_surgical_noDI_head 🆕              │ m09c2 (noDI)            │ ❄️ Head-only · D_L+D_A only   │ ViT-G/14     │
├─────────────────────────────────────────────┼─────────────────────────┼───────────────────────────────┼──────────────┤
│ vjepa_2_1_vith_frozen 🏗️                     │ Meta ViT-H direct       │ 🆎 Baseline                   │ ViT-H/14 🆕  │
│ vjepa_2_1_vith_pretrain_head 🏗️              │ m09a2 on ViT-H          │ ❄️ Head-only · raw            │ ViT-H/14 🆕  │
│ vjepa_2_1_vith_surgical_3stage_DI_head 🏗️    │ m09c2 on ViT-H          │ ❄️ Head-only · factor-aug     │ ViT-H/14 🆕  │
│ vjepa_2_1_vith_surgical_noDI_head 🏗️         │ m09c2 on ViT-H (noDI)   │ ❄️ Head-only · D_L+D_A        │ ViT-H/14 🆕  │
└─────────────────────────────────────────────┴─────────────────────────┴───────────────────────────────┴──────────────┘
```

---

## 🗺️ File changes at a glance

```
┌───────────────────────────────────────────────────────────────────────┬─────────┐
│ Action                                                                 │ Files   │
├───────────────────────────────────────────────────────────────────────┼─────────┤
│ 🧬 PHASE 0 — FG Motion Features (13-D → 23-D)                                     │
│   🔧 MODIFY src/m04d_motion_features.py (_aggregate_flow + FEATURE_DIM)│ 1       │
│   🔧 MODIFY src/utils/action_labels.py (parse_optical_flow_class)      │ 1       │
│   🔧 MODIFY src/utils/motion_aux_loss.py (n_motion_dims 13→23)         │ 1       │
│   🔄 RE-RUN m04d to regenerate motion_features.npy (~57 min)           │ —       │
├───────────────────────────────────────────────────────────────────────┼─────────┤
│ ❄️ PILLAR 1 — Head-Only Freeze                                                    │
│   🏷️ RENAME existing scripts (no logic change)                        │ 2       │
│   🆕 CREATE new training scripts (head-only siblings)                  │ 2       │
│   🆕 CREATE new probe script (future_regress)                          │ 1       │
│   🆕 CREATE new yaml configs                                            │ 3       │
│   🔧 MODIFY shell wrappers + probe_action.py + utils/training.py       │ 4       │
├───────────────────────────────────────────────────────────────────────┼─────────┤
│ 🏗️ PILLAR 2 — Backbone Scaling                                                    │
│   🆕 DOWNLOAD ViT-H/14 V-JEPA 2 checkpoint                              │ 1       │
│   🆕 CREATE configs/model/vjepa2_1_vitH.yaml                           │ 1       │
│   🔧 MODIFY utils/frozen_features.py (multi-backbone loader)           │ 1       │
│   🔧 MODIFY run_probe_eval.sh (add encoder_model_config_for helper)    │ 1       │
├───────────────────────────────────────────────────────────────────────┼─────────┤
│ 📚 PILLAR 3 — Curriculum Learning                                                 │
│   🆕 CREATE src/utils/data_curriculum.py (uses Phase 0's vec[13])       │ 1       │
│   🔧 MODIFY m09a2 + m09c2 (wire pacing into epoch loop)                │ 2       │
│   🔧 MODIFY base_optimization.yaml (add data_curriculum block)         │ 1       │
│   🔧 MODIFY run_probe_train.sh (CURRICULUM_OVERRIDE env-var)           │ 1       │
├───────────────────────────────────────────────────────────────────────┼─────────┤
│ TOTAL                                                                  │ 24      │
└───────────────────────────────────────────────────────────────────────┴─────────┘
```

---

## 🧬 Phase 0 — FG Motion Features (13-D → 23-D) — PREREQUISITE

> **🎯 Goal**: widen the surgery-vs-pretrain gap by replacing summary-statistical motion features (current 13-D) with **camera-subtracted foreground motion** (23-D) so factor curriculum's agent/interaction inductive bias has something to express. ALSO provides the principled difficulty axis (`vec[13] = fg_mean_mag`) needed by Pillar 3's data curriculum.

> **🔄 Sourced from**: `legacy/plan_phase5_fg_motion_features.md` — content integrated below; original archived.

### ❓ Why this exists

Current `m04d` output is too summary-statistical: `mean_mag` + `dir_hist` argmax → 8 classes recoverable from 2-3 numbers. Pretrain's `top1=0.808` proves it learned the summary stats, but surgery's factor curriculum (D_L / D_A / D_I) has no opportunity to express its agent / interaction inductive bias. **Harder features → larger surgery vs pretrain gap.** Phase 0 is the **prerequisite** that unlocks both Pillar 1's surgery-vs-pretrain delta AND Pillar 3's principled curriculum sort.

### 📋 Feature axes ranked by surgery-vs-pretrain gap-widening (from archived plan)

```
┌────┬──────────────────────────────────┬──────────────────────────┬──────┬───────────────────┐
│ 🆔 │ Feature axis                      │ What m04d adds            │ LoC  │ Δ-gap expected    │
├────┼──────────────────────────────────┼──────────────────────────┼──────┼───────────────────┤
│ 🏆 1│ D — FG/BG flow via SAM3 masks    │ FG (13-D) + BG (13-D)    │ ~50  │ +10–20 pp 🏆🏆     │
│    │   (escalation path)               │ Requires m10 SAM3 masks   │      │ Surgery's D_A/D_I │
│    │                                   │                          │      │ aligned 1:1       │
├────┼──────────────────────────────────┼──────────────────────────┼──────┼───────────────────┤
│ ✅ 2│ A — Foreground motion (camera-   │ flow MINUS camera motion │ ~10  │ +5–10 pp ← CHOICE │
│    │   subtracted) ← PHASE 0 SCOPE     │ fg_mean_mag, fg_max_mag, │      │ Cheap subset of D │
│    │                                   │ fg_dir_hist (10 dims)    │      │                   │
├────┼──────────────────────────────────┼──────────────────────────┼──────┼───────────────────┤
│ 🥈 3│ B — Acceleration (2nd-order)     │ mean_accel, accel_std,   │ ~5   │ +3–5 pp           │
│    │   (deferred)                      │ peak_accel_idx (3 dims)  │      │                   │
├────┼──────────────────────────────────┼──────────────────────────┼──────┼───────────────────┤
│ 🥉 4│ E — Divergence/curl              │ div + curl per pair      │ ~10  │ +2–5 pp           │
│    │   (deferred)                      │ (4 dims)                 │      │                   │
├────┼──────────────────────────────────┼──────────────────────────┼──────┼───────────────────┤
│ 🥉 5│ F — Motion frequency (FFT)       │ Top-3 spectral peaks +   │ ~5   │ +1–3 pp           │
│    │   (deferred)                      │ dominant freq (4 dims)   │      │                   │
├────┼──────────────────────────────────┼──────────────────────────┼──────┼───────────────────┤
│ 🥉 6│ C — Multi-scale temporal flow    │ RAFT on (i,i+1),(i,i+2), │ ~30  │ +1–3 pp           │
│    │   (deferred)                      │ (i,i+4) — 3 scales       │      │                   │
└────┴──────────────────────────────────┴──────────────────────────┴──────┴───────────────────┘
```

**🎯 Phase 0 scope = Option A only** (cheapest gap-widener, no SAM3 dependency). Decision matrix: if gap < +10 pp after Pillar 1 lands, escalate to Option D (full FG/BG via SAM3). Options B/C/E/F deferred to follow-up iter.

### 🔨 Step 0.1 — `src/m04d_motion_features.py:217-261` — extend `_aggregate_flow`

```python
def _aggregate_flow(flow_np, n_pairs):
    dx_all  = flow_np[:, 0]                                    # (N, H, W)
    dy_all  = flow_np[:, 1]
    mag_all = np.sqrt(dx_all**2 + dy_all**2)
    ang_all = np.arctan2(dy_all, dx_all)

    flat_mag = mag_all.flatten()
    mean_mag = float(np.mean(flat_mag))
    std_mag  = float(np.std(flat_mag))
    max_mag  = float(np.max(flat_mag))

    # 8-bin direction histogram (normalized) — global flow
    flat_ang = ang_all.flatten()
    hist, _  = np.histogram(flat_ang, bins=8, range=(-np.pi, np.pi))
    hist     = hist.astype(np.float32)
    if hist.sum() > 0:
        hist = hist / hist.sum()

    # Camera motion: median flow per pair, then median across pairs
    per_pair_dx = np.median(dx_all.reshape(n_pairs, -1), axis=1)
    per_pair_dy = np.median(dy_all.reshape(n_pairs, -1), axis=1)
    cam_x = float(np.median(per_pair_dx))
    cam_y = float(np.median(per_pair_dy))

    # 🧬 NEW (Phase 0): foreground motion = flow MINUS per-pair camera motion.
    # Removes camera-induced global translation → captures agent/object motion only.
    cam_dx_per_pair = per_pair_dx[:, None, None]               # (N, 1, 1) broadcast
    cam_dy_per_pair = per_pair_dy[:, None, None]
    fg_dx = dx_all - cam_dx_per_pair                           # (N, H, W)
    fg_dy = dy_all - cam_dy_per_pair
    fg_mag = np.sqrt(fg_dx**2 + fg_dy**2)
    fg_ang = np.arctan2(fg_dy, fg_dx)

    fg_mean_mag = float(fg_mag.mean())
    fg_max_mag  = float(fg_mag.max())
    fg_hist, _  = np.histogram(fg_ang.flatten(), bins=8, range=(-np.pi, np.pi))
    fg_hist     = fg_hist.astype(np.float32)
    if fg_hist.sum() > 0:
        fg_hist = fg_hist / fg_hist.sum()

    return np.array([
        mean_mag, std_mag, max_mag, *hist, cam_x, cam_y,        # existing 13 dims
        fg_mean_mag, fg_max_mag, *fg_hist,                      # 🆕 10 dims (total 23-D)
    ], dtype=np.float32)
```

Also update at module top (lines ~84-90):

```python
FEATURE_DIM = 23        # was 13
FEATURE_NAMES = [
    # Existing global flow (13 dims)
    "mean_mag", "std_mag", "max_mag",
    *[f"dir_hist_{i}" for i in range(8)],
    "cam_x", "cam_y",
    # 🆕 Phase 0 — foreground / camera-subtracted (10 dims)
    "fg_mean_mag", "fg_max_mag",
    *[f"fg_dir_hist_{i}" for i in range(8)],
]
```

### 🔨 Step 0.2 — `src/utils/action_labels.py:66-115` — `parse_optical_flow_class`

Switch the magnitude binning from `vec[0]` (global `mean_mag`, camera-contaminated) to `vec[13]` (`fg_mean_mag`, agent-only):

```python
def parse_optical_flow_class(clip_key, flow_features_by_key, magnitude_quartiles):
    vec = flow_features_by_key.get(clip_key)
    if vec is None:
        return None

    # 🧬 Phase 0: bin on FOREGROUND magnitude (vec[13]) instead of global (vec[0])
    # Foreground = flow with camera motion subtracted → captures AGENT motion classes.
    fg_mean_mag = float(vec[13])
    q1, q2, q3 = magnitude_quartiles
    if   fg_mean_mag < q1: mag_bin = "still"
    elif fg_mean_mag < q2: mag_bin = "slow"
    elif fg_mean_mag < q3: mag_bin = "medium"
    else:                  mag_bin = "fast"

    # 🆕 FG direction histogram lives at vec[15:23] (after fg_mean_mag, fg_max_mag).
    fg_dir_hist = vec[15:23]
    grouped = np.array([
        fg_dir_hist[0] + fg_dir_hist[1],   # rightward
        fg_dir_hist[2] + fg_dir_hist[3],   # upward
        fg_dir_hist[4] + fg_dir_hist[5],   # leftward
        fg_dir_hist[6] + fg_dir_hist[7],   # downward
    ], dtype=np.float64)
    dir_bin = _DIRECTION_BIN_ORDER[int(np.argmax(grouped))]

    return f"{mag_bin}{MOTION_SEPARATOR}{dir_bin}"
```

Also update `compute_magnitude_quartiles` (`action_labels.py:53-63`) to use `flow_features_array[:, 13]` instead of `[:, 0]` so quartile boundaries reflect FG magnitude distribution.

### 🔨 Step 0.3 — `src/utils/motion_aux_loss.py:75-100` — `MotionAuxHead.n_motion_dims`

Bump `n_motion_dims` from 13 → 23 in `MotionAuxHead.__init__` default + the wiring in `build_motion_aux_head_from_cfg`. Z-norm buffers (`vec_mean`, `vec_std`) auto-resize via `flow_features.mean(axis=0)` (already shape-agnostic).

### 🚀 Step 0.4 — Launch sequence

```bash
# 1. Audit current m04d output dim
python -c "import numpy as np; print(np.load('data/eval_10k_local/motion_features.npy').shape)"
# expect: (9297, 13) BEFORE Phase 0; (9297, 23) AFTER

# 2. Apply Step 0.1, 0.2, 0.3 edits (~10-15 LoC across 3 files)

# 3. Lint
python -m py_compile src/m04d_motion_features.py \
                     src/utils/action_labels.py \
                     src/utils/motion_aux_loss.py && \
    ruff check --select F,E9 src/m04d_motion_features.py \
                              src/utils/action_labels.py \
                              src/utils/motion_aux_loss.py

# 4. Re-run m04d on eval_10k_local (~57 min)
CACHE_POLICY_ALL=2 python -u src/m04d_motion_features.py --FULL \
    --subset data/eval_10k.json --local-data data/eval_10k_local \
    --features-out data/eval_10k_local/motion_features.npy \
    --no-wandb 2>&1 | tee logs/iter15_phase0_m04d.log

# 5. Verify post-Phase-0 shape
python -c "import numpy as np; \
  feats = np.load('data/eval_10k_local/motion_features.npy'); \
  print(f'shape={feats.shape}'); \
  assert feats.shape[1] == 23, 'FATAL: Phase 0 did not extend to 23-D'; \
  print(f'✅ Phase 0 features ready — vec[13]=fg_mean_mag range: [{feats[:,13].min():.3f}, {feats[:,13].max():.3f}]')"

# 6. Action labels auto-regenerate via probe_action.py --stage labels next time
#    any training script runs (uses new vec[13] FG magnitude for class binning).
```

### 🎲 Step 0.5 — Decision matrix (after Pillar 1 + 2 + 3 land)

```
┌──────────────────────────────────────┬───────────────────────────────────────────────┐
│ Outcome (surgery − pretrain gap)     │ Next                                           │
├──────────────────────────────────────┼───────────────────────────────────────────────┤
│ widens ≥ +10 pp 🏆                    │ Cite as paper claim; Phase 0 sufficient        │
│ widens +5 to +10 pp                  │ Escalate to Option D (FG/BG via SAM3 masks)   │
│ widens < +5 pp                       │ Escalate to Option D — full SAM3-conditioned   │
│ surgery REGRESSES 🚨                  │ Investigate Phase 4 λ + Stage-3 unfreeze       │
└──────────────────────────────────────┴───────────────────────────────────────────────┘
```

### 💰 Phase 0 cost

```
m04d rerun on eval_10k:  ~57 min · ~$2.40 @ $2.50/hr · disk +1.5 MB (23 floats × 9297 clips)
```

---

## 🔧 Phase 1 — RENAMES (no logic change)

```bash
# Rename current scripts — preserves them as the encoder-update track
git mv src/m09a_pretrain.py  src/m09a1_pretrain_encoder.py
git mv src/m09c_surgery.py   src/m09c1_surgery_encoder.py

# Update every caller across the repo
sed -i 's|src/m09a_pretrain\.py|src/m09a1_pretrain_encoder.py|g' \
    scripts/*.sh iter/iter*/*.md configs/train/*.yaml 2>/dev/null

sed -i 's|src/m09c_surgery\.py|src/m09c1_surgery_encoder.py|g' \
    scripts/*.sh iter/iter*/*.md configs/train/*.yaml 2>/dev/null

# Verify zero broken callers
grep -rnE 'm09a_pretrain\.py|m09c_surgery\.py' src/ scripts/ configs/ iter/  # expect: zero
```

> ⏰ **Timing**: defer until iter14 R5 sweep finishes (~01:00 UTC May 11). Python imports modules by name at process start, so renaming the file mid-run doesn't perturb already-loaded code — but to be safe, wait for R5.

---

## 🆕 Phase 2 — NEW FILES

### 🧠 `src/m09a2_pretrain_head.py` · ~200 LoC · ~80% reuse from m09a1

> **Purpose**: Train motion_aux MLP head on FROZEN Meta encoder + FROZEN Meta predictor + Indian eval_10k_train_split with motion-class CE + 13-D RAFT MSE loss.
> **Output**: trained motion_aux head; encoder + predictor bit-identical to Meta init.

#### 🔨 Function-level changes from m09a1

**In `merge_config_with_args(cfg, args)`** — force these overrides:

```python
cfg["layer_freeze"]["enabled"] = True
cfg["layer_freeze"]["freeze_below"] = 48                    # ALL ViT-G blocks frozen
cfg["drift_control"]["enabled"] = False                      # nothing drifts when frozen
cfg["optimization"]["lr"] = 5.0e-4                           # probe-head LR (head has only 432K params)
cfg["loss"]["weight_jepa"] = 0.0                             # skip JEPA — predictor frozen
cfg["loss"]["weight_motion_aux"] = 1.0                       # motion_aux is the sole signal
```

**In `build_model(cfg, device)`** — after predictor construction (extends `m09a1_pretrain_encoder.py:201-395`):

```python
for p in student.parameters():     p.requires_grad = False
for p in predictor.parameters():   p.requires_grad = False
for p in teacher.parameters():     p.requires_grad = False    # already frozen in m09a1

from utils.training import assert_encoder_frozen
assert_encoder_frozen(student)                                 # fail-loud guard

print("[m09a2 STRICT HEAD-ONLY] encoder+predictor frozen; "
      "trainable params = motion_aux head only")
```

**In `train(cfg, args)`** — reuses m09a1.train() verbatim EXCEPT:

```
- 🚫 skip JEPA L1 loss path        (predictor frozen → no gradient signal)
- 🚫 skip EMA teacher update        (teacher_mode=FROZEN; no parameter updates)
- 🚫 skip compute_drift_loss        (frozen params have no drift to penalize)
- ✅ keep motion_aux loss (CE + 13-D MSE) as the SOLE training signal
```

**Optimizer** — `build_optimizer` call at m09a1:547 works unchanged:
- `split_params` returns empty lists for student/predictor (requires_grad=False)
- Only motion_aux head's param group is active via `attach_motion_aux_to_optimizer`

**Checkpoint export** — outputs:
- `student_encoder.pt` — COPY of Meta's encoder (bit-identical) 📄
- `m09a_ckpt_best.pt` — Meta encoder + Meta predictor + TRAINED motion_aux head 📦
- `motion_aux_head.pt` — trained head only (~432K params, ~2 MB) 🧠

> 💡 **Why COPY not SYMLINK** for `student_encoder.pt`: avoids breaking `du -h` checks in `run_probe_eval.sh:267` and `pretrain-cleanup` logic at L427-471 which expects regular files.

---

### 🔬 `src/m09c2_surgery_head.py` · ~250 LoC · ~85% reuse from m09c1

> **Purpose**: Same freeze policy as m09a2 + factor-augmented data path. Single training stage (no progressive unfreezing — encoder frozen always). Trains motion_aux head on Meta encoder features from factor-decomposed clips.

#### 🔨 Function-level changes from m09c1

**In `merge_config_with_args(cfg, args)`** — force single head-only stage:

```python
cfg["surgery"]["stages"] = [{
    "name": "stage0_head_only_factor",
    "unfreeze_below": 0.0,                                      # ALL blocks frozen
    "max_epochs_pct": 1.0,                                       # consume entire epoch budget
    "mode_mixture": <inherited from yaml>
        # 3stage_DI variant → {L: 0.15, A: 0.15, I: 0.70}
        # noDI variant      → {L: 0.50, A: 0.50, I: 0.00}
}]
cfg["drift_control"]["enabled"] = False
cfg["loss"]["weight_jepa"] = 0.0
# IGNORE: recipe-v3 flags that no-op under freeze (SPD, saliency, lp_ft_stage0 — all moot)
# KEEP:   recipe-v3 flags still meaningful: CLEAR raw replay (controls factor-vs-raw mixture)
```

**In `build_model`** — same freeze pattern as m09a2 + `assert_encoder_frozen(student)`.

**In `train`** — inherits m09c1's stage loop verbatim:
- Single stage with `unfreeze_below=0.0` → `set_trainable_prefix(student, 0)` (m09c1:1190) freezes all
- `StreamingFactorDataset` (m09c1:1237-1262) works unchanged — factor data → frozen encoder → features → motion_aux head → CE+MSE loss
- 🚫 JEPA loss + drift loss both skipped

#### 🎛️ CLI flag behavior under STRICT freeze

```
┌──────────────────┬─────────────────────────────────────────────────────────┐
│ Flag              │ Effect under head-only freeze                            │
├──────────────────┼─────────────────────────────────────────────────────────┤
│ --teacher-mode    │ 🚫 MOOT — encoder doesn't move, EMA target unused        │
│ --lp-ft-stage0    │ 🚫 MOOT — we already force a single head-only stage      │
│ --subset-mode     │ 🚫 MOOT — no encoder blocks to unfreeze                  │
│ --warmup-mode     │ ✅ ACTIVE — controls LR warmup for head training        │
│ --saliency        │ 🚫 MOOT — saliency operates on encoder loss              │
│ --spd             │ 🚫 MOOT — SPD anchors encoder weights                    │
│ --replay          │ ✅ ACTIVE — controls raw-vs-factor data mixture          │
└──────────────────┴─────────────────────────────────────────────────────────┘
```

**Outputs per variant**:

```
outputs/<mode>/m09c_surgery_3stage_DI_head/
    📄 student_encoder.pt        # COPY of Meta's encoder
    📦 m09c_ckpt_best.pt         # Meta enc + Meta pred + trained motion_aux head
    🧠 motion_aux_head.pt

outputs/<mode>/m09c_surgery_noDI_head/   # same structure for noDI variant
```

---

### 🔮 `src/probe_future_regress.py` · ~280 LoC · NEW module

> **Purpose**: Replace `probe_future_mse.py` for variants where the predictor is frozen at Meta's. Trains a small MLP head (linear or 2-layer) to predict `encoder(x[t+8:t+16])` from `encoder(x[t:t+8])` on each variant.
> **Per-variant Δ comes from**: data exposure during regressor head training (raw vs factor-aug).

#### 🖥️ CLI surface (mirrors `probe_future_mse.py`)

```
--stage {forward, paired_per_variant}
--variant <encoder_name>
--encoder-ckpt <path>             # encoder-only, no predictor needed
--data-source {raw, factor_aug}   # 🆕 selects training data
--regressor-arch {linear, mlp_d1, mlp_d2}
--action-probe-root, --local-data, --output-root, --num-frames, --cache-policy
```

#### 🔄 Stage `forward` — per encoder

```python
# 1) Load frozen encoder via utils.frozen_features.load_vjepa_2_1_frozen
#    (probe_action.py:314 pattern)
# 2) Extract per-clip (16-frame) features split into:
#    context = encoder(x[t:t+8])
#    target  = encoder(x[t+8:t+16])
#    BOTH forwards under torch.no_grad() — encoder is permanently frozen

# 3) Build regressor head per --regressor-arch:
arch_map = {
    "linear":  nn.Linear(1664, 1664),
    "mlp_d1":  nn.Sequential(nn.Linear(1664, 4096), nn.GELU(), nn.Linear(4096, 1664)),
    "mlp_d2":  nn.Sequential(
                   nn.Linear(1664, 4096), nn.GELU(),
                   nn.Linear(4096, 4096), nn.GELU(),
                   nn.Linear(4096, 1664)),
}

# 4) Train regressor — AdamW(lr=1e-3, wd=0.05), cosine schedule, 50 epochs
#    Loss: L1(regressor_output, target)  # stop-grad on target

# 5) Evaluate on test_split — per-clip L1 averaged across tokens

# 6) Outputs (probe_future_mse-compatible):
#    outputs/<mode>/probe_future_regress/<variant>/
#        📄 per_clip_regressor_l1.npy     # (N_test,) float32
#        📄 clip_keys.npy                  # (N_test,) object array
#        📄 aggregate_regressor_l1.json    # {variant, n_test, l1_mean, l1_std, l1_ci, regressor_arch}
#        📦 regressor.pt                   # trained head weights
```

#### 🔄 Stage `paired_per_variant` — aggregator

Mirror `probe_future_mse.py:450-547`:
- Auto-discover variants by scanning `output_root/*/`
- For each pair (a, b): intersect clip_keys, compute per-clip delta = a_l1 − b_l1
- `paired_bca(delta)` from `utils/bootstrap.py`
- Emit `outputs/<mode>/probe_future_regress/probe_future_regress_per_variant.json`

> 📝 **`--data-source` flag wiring**: surgery_head variants use `factor_aug` (StreamingFactorDataset); pretrain_head + frozen use `raw`.

---

### 📜 `src/utils/training.py` — add ~10 LoC

```python
def assert_encoder_frozen(student) -> None:
    """🚨 Fail-loud guard: every block param must have requires_grad=False.

    Norms can remain trainable (Meta convention; tiny param count).
    Called from m09a2/m09c2.train() to prevent silent unfreezing bugs.
    """
    block_params_trainable = sum(
        p.numel()
        for blk in student.blocks
        for p in blk.parameters()
        if p.requires_grad
    )
    if block_params_trainable > 0:
        sys.exit(
            f"❌ FATAL: assert_encoder_frozen — {block_params_trainable:,} block params "
            f"have requires_grad=True. STRICT head-only mode requires all 48 blocks frozen. "
            f"Check freeze_encoder wiring in build_model()."
        )
```

---

## 📄 Phase 2b — NEW YAML configs

### 🧠 `configs/train/probe_pretrain_head.yaml` · ~25 lines

```yaml
# 🧠 m09a2 — head-only pretraining
# Frozen Meta encoder + frozen Meta predictor + trained motion_aux head
extends: probe_pretrain.yaml

data:
  adapted_encoder: vjepa_2_1_pretrain_head
  output_dir: outputs/full/m09a_pretrain_head

# ❄️ Force ALL ViT-G blocks frozen (overrides parent's freeze_below: 20)
layer_freeze:
  enabled: true
  freeze_below: 48

# 🚫 No drift anchor — nothing drifts (encoder + predictor frozen)
drift_control:
  enabled: false
  lambda_reg: 0.0

# 🚫 Skip JEPA loss (no trainable encoder/predictor)
# ✅ motion_aux head supervised loss is the SOLE training signal
loss:
  weight_jepa: 0.0
  weight_motion_aux: 1.0

# 🎛️ Lower LR for head-only training (432K params total)
optimization:
  lr: 5.0e-4              # standard probe-head LR (vs 1e-4 for encoder fine-tuning)
  warmup_pct: 0.10
  weight_decay: 0.05
```

### 🔬 `configs/train/surgery_3stage_DI_head.yaml` · ~30 lines

```yaml
# 🔬 m09c2 — head-only surgery with D_L + D_A + D_I curriculum
# Single stage, factor-aug data, encoder + predictor both frozen
extends: surgery_3stage_DI.yaml

data:
  adapted_encoder: vjepa_2_1_surgical_3stage_DI_head
  output_dir: outputs/full/m09c_surgery_3stage_DI_head

# ❄️ Force single head-only stage; inherits D_I-rich mixture from parent stage 3
surgery:
  warmup_mode: single
  total_warmup_pct: 0.10
  lp_ft_stage0:
    enabled: false                            # 🚫 MOOT — we already force head-only stage
  stages:
    - name: stage0_head_only_DI
      unfreeze_below: 0.0                      # ❄️ ALL 48 blocks frozen
      mode_mixture: {L: 0.15, A: 0.15, I: 0.70}
      max_epochs_pct: 1.0                      # consume entire epoch budget

# 🚫 Drift OFF — encoder + predictor frozen, nothing to anchor
drift_control:
  enabled: false
  lambda_reg: 0.0

# 🚫 No JEPA loss — same reason as m09a2
loss:
  weight_jepa: 0.0
  weight_motion_aux: 1.0

optimization:
  lr: 5.0e-4
  warmup_pct: 0.10
  weight_decay: 0.05
```

### 🔬 `configs/train/surgery_2stage_noDI_head.yaml` · ~30 lines

Same as `surgery_3stage_DI_head.yaml` but:
- `extends: surgery_2stage_noDI.yaml`
- `mode_mixture: {L: 0.50, A: 0.50, I: 0.00}` (no D_I)
- Tests whether D_I tubes carry signal beyond D_L + D_A 🧪

---

## 🔌 Phase 3 — Wiring changes

### 🐚 `scripts/run_probe_train.sh`

Extend `SUBCMD` dispatch at L50-53 + L221-225:

```bash
case "$SUBCMD" in
    pretrain|pretrain_2X|pretrain_head|\
    surgery_3stage_DI|surgery_3stage_DI_head|\
    surgery_noDI|surgery_noDI_head) ;;
    *) echo "FATAL: ..." >&2; exit 2 ;;
esac

case "$SUBCMD" in
    pretrain_head)
        OUT_DIR="outputs/${mode_dir}/m09a_pretrain_head"
        TRAIN_CFG="configs/train/probe_pretrain_head.yaml"
        SRC_SCRIPT="src/m09a2_pretrain_head.py"
        # Dispatch identical to pretrain branch (L258-274)
        ;;
    surgery_3stage_DI_head)
        TRAIN_CFG="configs/train/surgery_3stage_DI_head.yaml"
        VARIANT_TAG="3stage_DI_head"
        SRC_SCRIPT="src/m09c2_surgery_head.py"
        OUT_DIR="outputs/${mode_dir}/m09c_surgery_3stage_DI_head"
        ;;
    surgery_noDI_head)
        TRAIN_CFG="configs/train/surgery_2stage_noDI_head.yaml"
        VARIANT_TAG="noDI_head"
        SRC_SCRIPT="src/m09c2_surgery_head.py"
        OUT_DIR="outputs/${mode_dir}/m09c_surgery_noDI_head"
        ;;
esac
```

### 🐚 `scripts/run_probe_eval.sh` — 3 changes

**Change 1** — Default ENCODERS list (L158):

```bash
ENCODERS="${ENCODERS:-vjepa_2_1_frozen \
                     vjepa_2_1_pretrain \
                     vjepa_2_1_pretrain_2X \
                     vjepa_2_1_pretrain_head \
                     vjepa_2_1_surgical_3stage_DI \
                     vjepa_2_1_surgical_3stage_DI_head \
                     vjepa_2_1_surgical_noDI_head}"
```

**Change 2** — `encoder_ckpt_for()` (L198-207), add 3 cases:

```bash
vjepa_2_1_pretrain_head)
    echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_head/student_encoder.pt" ;;
vjepa_2_1_surgical_3stage_DI_head)
    echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_3stage_DI_head/student_encoder.pt" ;;
vjepa_2_1_surgical_noDI_head)
    echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_noDI_head/student_encoder.pt" ;;
```

**Change 3** — Add NEW STAGE 9b — `probe_future_regress` (after Stage 9):

```bash
# 🔮 STAGE 9b — probe_future_regress paired_per_variant (CPU, BCa across N variants)
if ! should_skip 9b; then
    stamp "STAGE 9b · probe_future_regress paired_per_variant (CPU)"
    python -u src/probe_future_regress.py "--$MODE" \
        --stage paired_per_variant \
        --output-root "$OUTPUT_FUTURE_REGRESS" \
        --cache-policy "$P_MSE" \
        --no-wandb 2>&1 | tee logs/probe_future_regress_paired.log
fi
```

> 📝 Inside the per-encoder loop (around current Stage 8 at L709-731), add **Stage 8b** for `probe_future_regress` forward pass.

### 🐍 `src/probe_action.py` — extend `ITER14_DELTAS` (L684-694)

Append after Δ3:

```python
("delta_4_pretrain_vs_pretrain_head",
 "vjepa_2_1_pretrain", "vjepa_2_1_pretrain_head",
 "Δ4: pretrain > pretrain_head (encoder updates from continual SSL beat head-only baseline)"),

("delta_5_surgical_3stage_DI_vs_surgical_3stage_DI_head",
 "vjepa_2_1_surgical_3stage_DI", "vjepa_2_1_surgical_3stage_DI_head",
 "Δ5: surgery > surgery_head (factor surgery's gain is NOT just from head-level factor exposure) ⭐"),

("delta_6_surgery_head_vs_pretrain_head",
 "vjepa_2_1_surgical_3stage_DI_head", "vjepa_2_1_pretrain_head",
 "Δ6: surgery_head > pretrain_head (factor-aug data improves head training even with frozen encoder)"),

("delta_7_surgery_3stage_DI_head_vs_surgery_noDI_head",
 "vjepa_2_1_surgical_3stage_DI_head", "vjepa_2_1_surgical_noDI_head",
 "Δ7: 3stage_DI_head > noDI_head (D_I interaction tubes carry signal independent of D_L/D_A)"),
```

#### 🔍 Δ-interpretation guide (Pillar 1 deltas)

```
┌─────┬──────────────────────────────────────────────────────────────────────────────────┐
│ Δ   │ What it answers                                                                   │
├─────┼──────────────────────────────────────────────────────────────────────────────────┤
│ Δ1  │ Does continual SSL on Indian data help? (pretrain vs frozen)                     │
│ Δ2  │ Does factor surgery help beyond continual SSL? (surgery vs pretrain)             │
│ Δ3  │ Causal isolation — is surgery's gain from factor patching, not just more steps?  │
│ Δ4  │ Is encoder fine-tuning needed in the pretrain track?                              │
│ Δ5  │ ⭐ Is encoder fine-tuning needed in the surgery track? (THE KEY PAPER CLAIM)      │
│ Δ6  │ Does factor-aug data improve head training EVEN with frozen encoder?             │
│ Δ7  │ Do D_I tubes carry signal beyond D_L + D_A? (addresses agent-extraction concern) │
└─────┴──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Phase 4 — Backbone Scaling (ViT-G/14 vs ViT-H/14)

> **Goal**: replicate the head-only protocol on a smaller backbone (ViT-H/14 ≈ 632M params, 3× smaller than ViT-G's 1.84B) to measure how downstream motion-flow top-1 scales with encoder size.
> **Why this is a sweep, NOT an ablation**: no component is removed — both backbones run the SAME head-only protocol. This is the V-JEPA 2 paper §4 scaling-study design (their Fig. 5 reports ViT-L/H/g/G).

### 🔨 Step 4.1 — Download V-JEPA 2 ViT-H/14 checkpoint

```bash
# Need confirmed HF URI from V-JEPA 2 release. Candidates:
#   facebook/vjepa2-vith16-fpc64-256
#   facebook/vjepa2-vith16-fpc64-384
# Verify via HF model card before downloading.

mkdir -p checkpoints/vjepa2_1_vitH
huggingface-cli download facebook/<CONFIRMED_VITH_URI> \
    --local-dir checkpoints/vjepa2_1_vitH

# Verify
ls -lh checkpoints/vjepa2_1_vitH/  # expect ~2.5 GB ckpt
```

### 🔨 Step 4.2 — New model config

```yaml
# configs/model/vjepa2_1_vitH.yaml
# V-JEPA 2.1 ViT-H/14 — scaling comparison vs ViT-G/14
model:
  arch: vit_huge_xformers           # confirm exact name in utils/vjepa2_imports.get_vit_by_arch
  embed_dim: 1280                    # ViT-H/14 hidden dim
  n_blocks: 32                       # vs ViT-G's 48
  num_heads: 16
  patch_size: 14
  tubelet_size: 2
  use_rope: true                     # V-JEPA 2.1 convention
```

### 🔨 Step 4.3 — Extend `src/utils/frozen_features.py`

```python
# Add ViT-H entries to ENCODERS dict at frozen_features.py top:
ENCODERS = {
    "vjepa_2_1_frozen":                       {"kind": "vjepa", "cfg": "vjepa2_1.yaml"},
    "vjepa_2_1_pretrain":                     {"kind": "vjepa", "cfg": "vjepa2_1.yaml"},
    # ... existing ViT-G entries ...
    "vjepa_2_1_vith_frozen":                  {"kind": "vjepa", "cfg": "vjepa2_1_vitH.yaml"},
    "vjepa_2_1_vith_pretrain_head":           {"kind": "vjepa", "cfg": "vjepa2_1_vitH.yaml"},
    "vjepa_2_1_vith_surgical_3stage_DI_head": {"kind": "vjepa", "cfg": "vjepa2_1_vitH.yaml"},
    "vjepa_2_1_vith_surgical_noDI_head":      {"kind": "vjepa", "cfg": "vjepa2_1_vitH.yaml"},
}

# Modify load_vjepa_2_1_frozen() signature to accept model_config:
def load_vjepa_2_1_frozen(ckpt_path, num_frames, model_config="vjepa2_1.yaml"):
    cfg = load_yaml(f"configs/model/{model_config}")
    arch = cfg["model"]["arch"]
    embed_dim = cfg["model"]["embed_dim"]
    n_blocks = cfg["model"]["n_blocks"]
    # ... build model via get_vit_by_arch(arch) ...
    return model, crop, embed_dim
```

### 🔨 Step 4.4 — Wire into `run_probe_eval.sh`

```bash
# Add encoder_model_config_for() helper next to encoder_ckpt_for():
encoder_model_config_for() {
    case "$1" in
        vjepa_2_1_vith_*)   echo "vjepa2_1_vitH.yaml" ;;
        vjepa_2_1_*)        echo "vjepa2_1.yaml"     ;;
    esac
}

# encoder_ckpt_for() — add ViT-H cases:
vjepa_2_1_vith_frozen)
    echo "checkpoints/vjepa2_1_vitH/<ckpt-name>.pt" ;;
vjepa_2_1_vith_pretrain_head)
    echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_head_vith/student_encoder.pt" ;;
vjepa_2_1_vith_surgical_3stage_DI_head)
    echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_3stage_DI_head_vith/student_encoder.pt" ;;
vjepa_2_1_vith_surgical_noDI_head)
    echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_noDI_head_vith/student_encoder.pt" ;;

# Extend default ENCODERS list to include ViT-H variants
```

### 🔨 Step 4.5 — Train ViT-H head-only variants

```bash
# m09a2/m09c2 already work with frozen encoder — just point them at ViT-H ckpt
# via a new --model-config CLI arg + yaml override:

./scripts/run_probe_train.sh pretrain_head_vith --FULL              # m09a2 + ViT-H
./scripts/run_probe_train.sh surgery_3stage_DI_head_vith --FULL     # m09c2 + ViT-H
./scripts/run_probe_train.sh surgery_noDI_head_vith --FULL           # m09c2 + ViT-H (noDI)
```

> 💡 **Design choice**: ViT-H variants share m09a2/m09c2 scripts; backbone differs ONLY via `--model-config configs/model/vjepa2_1_vitH.yaml` CLI arg. No new training-script copies.

### 🔍 Δ-interpretation guide (Pillar 2 deltas)

```
┌─────┬──────────────────────────────────────────────────────────────────────────────────┐
│ Δ   │ What it answers                                                                   │
├─────┼──────────────────────────────────────────────────────────────────────────────────┤
│ Δ8  │ ViT-H vs ViT-G frozen baseline (raw scaling effect)                              │
│ Δ9  │ ViT-H vs ViT-G after head-only pretrain (does scaling effect survive head adapt)│
│ Δ10 │ ViT-H surgery_head vs ViT-G surgery_head (factor-aug × model-size interaction)  │
└─────┴──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Phase 5 — Curriculum Learning

### 📜 Sub-pillar 5a — Parameter curriculum (Lee ICLR'23)

**Status**: ✅ **ALREADY IMPLEMENTED** in `m09c1_surgery_encoder.py`. The 3-stage progressive prefix unfreezing in `surgery_3stage_DI.yaml`:

```
Stage 1: unfreeze_below = 0.083 →  4 of 48 blocks trainable  (D_L only)
Stage 2: unfreeze_below = 0.167 →  8 of 48 blocks trainable  (D_L + D_A mix)
Stage 3: unfreeze_below = 0.167 →  8 of 48 blocks trainable  (+ D_I)
```

This **IS** Lee ICLR'23's surgical schedule. The meeting note's "parameter curriculum: progressive layer unfreezing" is satisfied by the existing m09c1 design. No new code needed here.

### 🔨 Sub-pillar 5b — Data curriculum (Bengio 2009) — NEW

**Goal**: feed easy clips first (low motion magnitude → still / slow classes), then progressively expose harder clips (fast classes). Mitigates catastrophic forgetting by giving the head time to learn stable motion-class boundaries before exposure to high-magnitude noise.

#### 🔨 Step 5b.1 — New utility `src/utils/data_curriculum.py` (~80 LoC)

> 🧬 **Uses Phase 0's `fg_mean_mag` (vec[13])** as the difficulty axis — camera-subtracted, agent-only motion. Falls back to global `mean_mag` (vec[0]) with a loud WARN if Phase 0 features are not yet built (still 13-D).

```python
"""Easy-to-hard sample ordering for curriculum learning (Bengio 2009).

Difficulty proxy: FOREGROUND motion magnitude (vec[13] = fg_mean_mag) from
Phase 0 m04d 23-D features. Camera motion is subtracted → ranks clips by AGENT
motion only, not by camera-induced global translation.

Easy = still agent (regardless of camera);   Hard = fast agent.
"""
import sys
import numpy as np
from pathlib import Path


def sort_by_fg_magnitude(clip_keys, motion_features_path, order="ascending"):
    """Sort by FOREGROUND magnitude (vec[13]) — Phase-0-aware.

    Fallback: if motion_features is still 13-D (Phase 0 not yet run), uses
    global mean_mag (vec[0]) with a loud WARN. Curriculum is then
    camera-motion-contaminated and less principled.
    """
    features = np.load(motion_features_path)                              # (N, 13) or (N, 23+)
    paths    = np.load(Path(motion_features_path).with_suffix(".paths.npy"),
                       allow_pickle=True)
    key_to_idx = {Path(p).stem: i for i, p in enumerate(paths)}

    feat_dim = features.shape[1]
    if feat_dim >= 23:
        difficulty_col = 13                                                # fg_mean_mag (Phase 0)
        print(f"  [curriculum] using Phase-0 FG magnitude (vec[13]) "
              f"— camera-subtracted, agent-only")
    elif feat_dim == 13:
        difficulty_col = 0                                                 # mean_mag (fallback)
        print(f"  ⚠️ [curriculum] Phase 0 features NOT available (still 13-D); "
              f"falling back to global mean_mag (vec[0]) — camera-motion-contaminated "
              f"difficulty signal. Run Phase 0 to enable principled curriculum.")
    else:
        sys.exit(f"❌ FATAL: unexpected motion_features shape {features.shape}")

    pairs = [(k, float(features[key_to_idx[k], difficulty_col]))
             for k in clip_keys if k in key_to_idx]
    pairs.sort(key=lambda x: x[1], reverse=(order == "descending"))
    return [k for k, _ in pairs]


def get_active_pool(epoch, total_epochs, sorted_keys, pacing="linear"):
    """Pacing function — fraction of sorted pool exposed at given epoch.

    Pacing modes:
        linear  — frac = min(1.0, (epoch+1) / (total_epochs/2))
        step    — bottom-50% for first half, full pool second half
        log     — frac grows logarithmically (slow expansion early)
    """
    if pacing == "linear":
        frac = min(1.0, (epoch + 1) / max(1, total_epochs // 2))
    elif pacing == "step":
        frac = 0.5 if epoch < total_epochs // 2 else 1.0
    elif pacing == "log":
        frac = min(1.0, np.log1p(epoch + 1) / np.log1p(total_epochs // 2))
    else:
        sys.exit(f"❌ FATAL: unknown pacing '{pacing}' (use linear|step|log)")
    n_active = max(1, int(frac * len(sorted_keys)))
    return sorted_keys[:n_active]
```

#### 🔨 Step 5b.2 — Wire into m09a2 / m09c2 train loops

```python
# In merge_config_with_args() — read data_curriculum config:
curr_cfg = cfg.get("data_curriculum", {})
curr_enabled = curr_cfg.get("enabled", {}).get(args.mode_key, False)
if curr_enabled:
    from utils.data_curriculum import sort_by_fg_magnitude
    sorted_clip_keys = sort_by_fg_magnitude(
        train_clip_keys,
        curr_cfg["motion_features_path"],
        order=curr_cfg["order"],                          # "ascending" = easy → hard
    )
    cfg["_runtime"]["sorted_clip_keys"] = sorted_clip_keys
    # If Phase 0 features (23-D) are present → curriculum uses fg_mean_mag (camera-subtracted).
    # Else falls back to global mean_mag with a loud WARN.

# Inside epoch loop (m09a2:train + m09c2:train, around the epoch boundary):
from utils.data_curriculum import get_active_pool
for epoch in range(total_epochs):
    if curr_enabled:
        active_keys = get_active_pool(
            epoch, total_epochs,
            cfg["_runtime"]["sorted_clip_keys"],
            pacing=curr_cfg["pacing"],
        )
        train_loader.dataset.set_clip_keys(active_keys)   # NEW: dataset-level setter
        print(f"[curriculum] epoch {epoch}: active pool = {len(active_keys)}/"
              f"{len(cfg['_runtime']['sorted_clip_keys'])} clips")
    # ... existing train step ...
```

**Note on `set_clip_keys`**: requires adding a setter on `StreamingFactorDataset` (m09c2) and the legacy DataLoader subset (m09a2). Both should accept a new clip_keys list and rebuild the per-worker shards. ~15 LoC change.

#### 🔨 Step 5b.3 — Yaml flag (per-mode dict)

Append to `configs/train/base_optimization.yaml`:

```yaml
# 📚 Data curriculum (Bengio 2009) — easy → hard motion-magnitude ordering
# Used by m09a2 (head-only) and m09c2 (head-only + factor). NOT applied to m09a1/m09c1
# (encoder-update track already uses parameter curriculum via Lee ICLR'23 progressive unfreeze).
data_curriculum:
  enabled:
    sanity: false                            # smoke test: skip curriculum to validate single-epoch path
    poc:    true                             # POC enables curriculum for early Δ measurement
    full:   true
  motion_features_path: data/eval_10k_local/motion_features.npy
  order: ascending                           # easy → hard  (descending = hard → easy for adversarial test)
  pacing: linear                             # {linear, step, log}
```

#### 🔨 Step 5b.4 — CLI env-var override

```bash
# In scripts/run_probe_train.sh (next to other recipe-v3 env vars at L306-392):
if [ -n "${CURRICULUM_OVERRIDE:-}" ]; then
    case "$CURRICULUM_OVERRIDE" in
        off|linear|step|log)
            RECIPE_V2_ARGS+=(--data-curriculum "$CURRICULUM_OVERRIDE")
            ;;
        *)
            echo "❌ FATAL: CURRICULUM_OVERRIDE must be off|linear|step|log (got: $CURRICULUM_OVERRIDE)" >&2
            exit 2
            ;;
    esac
fi
```

CLI flag in m09a2/m09c2:

```python
parser.add_argument("--data-curriculum", choices=["off", "linear", "step", "log"], default=None,
                    help="iter15 §5b: data-curriculum pacing mode. Overrides yaml.")
```

### 🔍 Δ-interpretation guide (Pillar 3 deltas)

```
┌─────┬──────────────────────────────────────────────────────────────────────────────────┐
│ Δ   │ What it answers                                                                   │
├─────┼──────────────────────────────────────────────────────────────────────────────────┤
│ Δ11 │ Does data curriculum (easy→hard) help pretrain_head? (vs raw shuffled order)     │
│ Δ12 │ Does data curriculum help surgery_head? (vs raw shuffled)                        │
│ Δ13 │ Parameter × data curriculum interaction — m09c1 (encoder) vs m09c2_curr (head)   │
└─────┴──────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✅ Verification plan — SANITY-mode end-to-end

> 🎯 Each step is **code-correctness**, not model-quality. Total wall ≈ 35 min on Blackwell.

### 🧪 V0 — Phase 0 motion-features shape check (~5 sec, run AFTER Phase 0)

```bash
python -c "
import numpy as np
feats = np.load('data/eval_10k_local/motion_features.npy')
assert feats.shape[1] == 23, f'❌ Phase 0 did not extend features to 23-D (got {feats.shape[1]})'
fg_mag = feats[:, 13]
print(f'✅ Phase 0 features ready')
print(f'   shape: {feats.shape}')
print(f'   vec[13] fg_mean_mag range: [{fg_mag.min():.4f}, {fg_mag.max():.4f}]')
print(f'   quartile boundaries: q1={np.percentile(fg_mag, 25):.4f}, '
      f'q2={np.percentile(fg_mag, 50):.4f}, q3={np.percentile(fg_mag, 75):.4f}')
"
```

### 🧪 V1 — Static checks (~5 sec)

```bash
python -c "import py_compile; \
  py_compile.compile('src/m04d_motion_features.py', doraise=True); \
  py_compile.compile('src/utils/action_labels.py', doraise=True); \
  py_compile.compile('src/utils/motion_aux_loss.py', doraise=True); \
  py_compile.compile('src/utils/data_curriculum.py', doraise=True); \
  py_compile.compile('src/m09a2_pretrain_head.py', doraise=True); \
  py_compile.compile('src/m09c2_surgery_head.py', doraise=True); \
  py_compile.compile('src/probe_future_regress.py', doraise=True); \
  py_compile.compile('src/utils/training.py', doraise=True); \
  print('✅ all compile clean')"
```

### 🧪 V2 — m09a2 SANITY (~3 min)

```bash
./scripts/run_probe_train.sh pretrain_head --SANITY 2>&1 \
  | tee logs/iter15_v1_m09a2_sanity.log
```

**Pass criteria** 🎯:
- ✅ Log contains `[m09a2 STRICT HEAD-ONLY]`
- ✅ `assert_encoder_frozen` passed
- ✅ `outputs/sanity/m09a_pretrain_head/{student_encoder.pt, m09a_ckpt_best.pt}` produced

### 🧪 V3 — m09c2 SANITY both variants (~10 min total)

```bash
./scripts/run_probe_train.sh surgery_3stage_DI_head --SANITY 2>&1 \
  | tee logs/iter15_v1_m09c2_3stage_DI_head_sanity.log

./scripts/run_probe_train.sh surgery_noDI_head --SANITY 2>&1 \
  | tee logs/iter15_v1_m09c2_noDI_head_sanity.log
```

**Pass criteria** 🎯:
- ✅ Each produces `student_encoder.pt` + `m09c_ckpt_best.pt`
- ✅ Single-stage loop confirmed in log (no Stage 1/2/3 transitions)

### 🧪 V4 — Encoder invariance check (~10 sec)

```bash
python -c "
import torch
from pathlib import Path

meta = torch.load('checkpoints/vjepa2_1_vitG_384.pt', map_location='cpu')

for variant in ['m09a_pretrain_head',
                'm09c_surgery_3stage_DI_head',
                'm09c_surgery_noDI_head']:
    p = Path(f'outputs/sanity/{variant}/student_encoder.pt')
    if not p.exists():
        print(f'⚠️  {variant}: not produced')
        continue
    head_out = torch.load(p, map_location='cpu')
    n_match, n_total = 0, 0
    for k in head_out:
        if k.startswith('blocks.'):
            n_total += 1
            if k in meta and torch.allclose(head_out[k], meta[k]):
                n_match += 1
    print(f'{variant}: {n_match}/{n_total} blocks identical to Meta')
    assert n_match == n_total, f'❌ FAIL — {variant} has updated blocks (encoder not frozen)'
print('✅ all 3 head-only variants have bit-identical encoder weights to Meta')
"
```

### 🧪 V5 — probe_future_regress SANITY (~5 min)

```bash
python -u src/probe_future_regress.py --SANITY \
    --stage forward \
    --variant vjepa_2_1_frozen \
    --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \
    --data-source raw \
    --regressor-arch linear \
    --action-probe-root outputs/sanity/probe_action \
    --local-data data/eval_10k_local \
    --output-root outputs/sanity/probe_future_regress \
    --cache-policy 2 2>&1 | tee logs/iter15_v1_probe_future_regress_sanity.log
```

**Pass criteria** 🎯:
- ✅ `outputs/sanity/probe_future_regress/vjepa_2_1_frozen/` contains:
  - `per_clip_regressor_l1.npy`
  - `aggregate_regressor_l1.json`
  - `regressor.pt`

### 🧪 V6 — Full eval SANITY with all 7 variants (~15 min)

```bash
./scripts/run_probe_eval.sh --sanity 2>&1 \
  | tee logs/iter15_v1_full_eval_sanity.log
```

**Pass criteria** 🎯:
- ✅ STAGE 4 paired_delta emits `probe_paired_delta.json` with Δ4/Δ5/Δ6/Δ7 keys
- ✅ STAGE 9b emits `probe_future_regress_per_variant.json`
- ✅ ZERO STAGE failures across all 7 variants

---

## 💰 Compute budget (FULL mode)

```
┌──────────────────────────────────────────┬──────────────┬────────────────────┐
│ 🏃 Training run                           │ ⏰ Wall (FULL)│ 💵 Cost @ $2.50/h  │
├──────────────────────────────────────────┼──────────────┼────────────────────┤
│ 🧬 PHASE 0 — m04d 13-D → 23-D rerun       │ ~57 min      │ ~$2.40             │
├──────────────────────────────────────────┼──────────────┼────────────────────┤
│ ❄️ PILLAR 1 — Head-Only Freeze                                                   │
│   🧠 m09a2_pretrain_head                   │ ~1.5 GPU-h   │ ~$3.75             │
│   🔬 m09c2_3stage_DI_head                  │ ~2.0 GPU-h   │ ~$5.00             │
│   🔬 m09c2_noDI_head                       │ ~1.5 GPU-h   │ ~$3.75             │
│   🔮 probe_future_regress × 7 variants     │ ~30 min × 7  │ ~$8.75             │
├──────────────────────────────────────────┼──────────────┼────────────────────┤
│ 🏗️ PILLAR 2 — Backbone Scaling (ViT-H/14)                                        │
│   m09a2 + m09c2 × 2 noDI/3stage variants  │ ~3-4 GPU-h   │ ~$10               │
│   probe_future_regress × 4 ViT-H variants  │ ~20 min × 4  │ ~$3.33             │
├──────────────────────────────────────────┼──────────────┼────────────────────┤
│ 📚 PILLAR 3 — Curriculum (re-uses Pillar 1 budget — no extra compute)            │
│   2 curriculum variants (pretrain+surg)    │ included     │ —                  │
├──────────────────────────────────────────┼──────────────┼────────────────────┤
│ 🎯 TOTAL NEW COMPUTE                       │ ~13-15 GPU-h │ ~$37               │
└──────────────────────────────────────────┴──────────────┴────────────────────┘

Plus already-spent iter14 sweep ($65). 🏁 Grand total to land 13-variant paper figure: ~$102.
```

---

## 🚨 Open notes / risks

### 🟡 Risk 1 — m09a2's expected NULL result on probe_action top-1

By construction, `vjepa_2_1_pretrain_head`'s probe_action top-1 should EQUAL `vjepa_2_1_frozen`'s (same encoder; both retrain AttentiveClassifier in probe_action Stage 3).

**Δ6_pretrain_head_vs_frozen ≈ 0 is THE SANITY CHECK** 🔍 — if its CI excludes 0, something leaked:
- 🚨 norm params drifted under layer_freeze
- 🚨 AttentiveClassifier seed/data nondeterminism

This is the lead's baseline null result, NOT a paper failure.

### 🟡 Risk 2 — Label/clip domain shift for m09c2

m09c2 motion_aux head trains on factor-aug clips but test labels come from RAW clips. Factor-augmented clips have different motion statistics:
- 🌫️ D_L blurs agents → less motion energy on motion-rich regions
- 🎭 D_A suppresses background → motion mostly from agents
- 🤝 D_I = interaction tubes → spatially restricted

**Training on D_A features with labels from raw clips creates a train-test domain shift.** This is INTENTIONAL — the paper claim is that exposure to factor-decomposed data IS the contribution. If this hurts the head, that's a finding (factor augmentation doesn't help head training without encoder updates).

### 🟡 Risk 3 — probe_future_regress with identical encoders

Encoder is IDENTICAL (Meta's) for `frozen` + `pretrain_head` + `surgery_head` + `surgery_noDI_head`. Input features to the regressor are IDENTICAL.

Per-variant Δ comes ONLY from regressor head's TRAINING DATA exposure:
- `frozen` + `pretrain_head` (both raw train_split) → regressor outputs identical to seed noise
- `surgery_head` variants (factor-aug) → different feature pairs → meaningful Δ

### 🟡 Risk 4 — Norm parameters

`set_trainable_prefix(student, 0)` keeps norms trainable per Meta convention. ViT-G has ~50K norm params. **These WILL update during m09a2/m09c2 training.**

Verify (V4 check) that BLOCK params are frozen — norm drift is acceptable per Meta convention.

### 🟡 Risk 5 — Loss weighting safety

Forcing `weight_jepa=0.0` requires `compute_jepa_loss` to handle this gracefully:
- ✅ Multiplication by 0 is well-defined
- 🚨 Check there's no division by `weight_jepa` anywhere downstream (e.g., normalization, logging)

### 🟡 Risk 6 — Rename safety

iter14 R5 cell is currently running and depends on `src/m09c_surgery.py`.

**Defer the rename until R5 finishes (~01:00 UTC May 11).** After rename, validate via:

```bash
grep -rnE 'm09a_pretrain|m09c_surgery' src/ scripts/ configs/ iter/  # expect: zero
```

---

## 📌 Critical files reference

### 🏷️ To RENAME

```
src/m09a_pretrain.py   →  src/m09a1_pretrain_encoder.py
src/m09c_surgery.py    →  src/m09c1_surgery_encoder.py
```

### 🆕 To CREATE

```
src/m09a2_pretrain_head.py                  🧠 head-only pretraining
src/m09c2_surgery_head.py                   🔬 head-only factor surgery
src/probe_future_regress.py                 🔮 future-prediction probe (replaces probe_future_mse)
configs/train/probe_pretrain_head.yaml      📄 yaml for m09a2
configs/train/surgery_3stage_DI_head.yaml   📄 yaml for m09c2 (D_I variant)
configs/train/surgery_2stage_noDI_head.yaml 📄 yaml for m09c2 (noDI variant)
```

### 🔧 To MODIFY

```
🧬 PHASE 0 (FG motion features)
src/m04d_motion_features.py   🔧 _aggregate_flow (L217-261) + FEATURE_DIM 13→23
src/utils/action_labels.py    🔧 parse_optical_flow_class (L66-115) — use vec[13]
src/utils/motion_aux_loss.py  🔧 MotionAuxHead.n_motion_dims 13→23 (L75-100)

❄️ PILLAR 1 + 🏗️ PILLAR 2 + 📚 PILLAR 3
src/utils/training.py        🔧 add assert_encoder_frozen() helper (~L1450)
src/utils/frozen_features.py 🔧 multi-backbone loader (ViT-G + ViT-H) — Pillar 2
scripts/run_probe_train.sh   🔧 add 3 subcommands + CURRICULUM_OVERRIDE env-var
scripts/run_probe_eval.sh    🔧 extend ENCODERS + encoder_ckpt_for + Stage 8b/9b
                                + encoder_model_config_for() helper
src/probe_action.py          🔧 extend ITER14_DELTAS at L684-694 with Δ4-Δ13
configs/train/base_optimization.yaml  🔧 add data_curriculum block
```

### 🆕 To CREATE (additional, from pillars 2 & 3)

```
configs/model/vjepa2_1_vitH.yaml      🏗️ Pillar 2 — ViT-H/14 model config
checkpoints/vjepa2_1_vitH/<*.pt>      🏗️ Pillar 2 — downloaded HF ckpt
src/utils/data_curriculum.py          📚 Pillar 3 — sort_by_fg_magnitude + pacing
```

### ♻️ REUSED (no changes)

```
src/utils/training.py        🔧 build_optimizer, compute_jepa_loss,
                                set_trainable_prefix, StreamingFactorDataset
src/utils/motion_aux_loss.py 🧠 attach_motion_aux_to_optimizer
src/utils/bootstrap.py       📊 paired_bca, bootstrap_ci
```

### 📦 ARCHIVED

```
iter/iter15_trainHead_freezeEncoder/legacy/plan_phase5_fg_motion_features.md
  → Content integrated as Phase 0 above; original moved to legacy/ for provenance.
```

---

> 🏁 **Execution order** (when iter14 R5 finishes ~01:00 UTC May 11):
> **Phase 0** (motion features, ~1 hr) → V0 shape check → **Phase 1** (renames) → V1 static checks → **Phase 2** (m09a2/m09c2/probe_future_regress) → V2/V3/V4 trainer SANITY → **Phase 3** (wiring) → **Phase 4** (ViT-H scaling) → **Phase 5** (curriculum) → V5/V6 eval SANITY → FULL training across 13 variants.
