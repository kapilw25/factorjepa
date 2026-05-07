# 🧬 Phase 5 — Harder motion features (FG/BG flow in `m04d`)

> **🎯 Goal**: widen the surgery-vs-pretrain gap by replacing summary-statistical motion features (current 13-D) with **camera-subtracted foreground motion** (23-D) so factor curriculum's agent/interaction inductive bias has something to express.

> **🚦 Gate**: ⚠️ **DO NOT START** until Phase 4 ([`plan_motion_aux_to_surgery.md`](./plan_motion_aux_to_surgery.md)) finishes. Only proceed if **Δ (surgery_3DI − pretrain) < +5 pp** on the motion-flow probe.

> **Sister docs**: [`plan_HIGH_LEVEL.md`](./plan_HIGH_LEVEL.md) · [`plan_motion_aux_to_surgery.md`](./plan_motion_aux_to_surgery.md) (Phase 4 — pre-req)

---

## ❓ Why this exists

Current `m04d` output is too summary-statistical: `mean_mag` + `dir_hist` argmax → 8 classes recoverable from 2-3 numbers. Pretrain's `top1=0.808` proves it learned the summary stats, but surgery's factor curriculum (D_L/D_A/D_I) has no opportunity to express its agent / interaction inductive bias. **Harder features → larger surgery vs pretrain gap.**

---

## 📋 Table 3 — Feature axes ranked by surgery-vs-pretrain gap-widening

| 🆔 | Feature axis | What `m04d` adds | LoC | Effort | Δ-gap expected | Why surgery wins |
|---|---|---|---|---|---|---|
| 🏆 1 | **D — FG/BG flow via SAM3 masks** | Compute flow stats SEPARATELY for person/vehicle pixels (m10 SAM3) vs background (extra 13 dims FG, kept 13 BG = 26-D vec) | ~50 | m04d wall + ~6 GPU-h m10 prereq | **+10–20 pp** | 🏆🏆 Surgery's D_A/D_I tubes ARE agent/interaction segmentations → 1:1 alignment. Pretrain has zero structural reason to encode FG vs BG separately |
| 🏆 2 | **A — Foreground motion (camera-subtracted)** | Per-pair flow MINUS `camera_motion_{x,y}` → `fg_mean_mag`, `fg_max_mag`, `fg_dir_hist` (extra 10 dims) | ~10 | re-run m04d (~57 min) | **+5–10 pp** | Cheap subset of D (no SAM3). Surgery's D_A trains on agent-only crops → encoder learns agent motion independent of camera |
| 🥈 3 | **B — Acceleration** (2nd-order temporal) | Time-derivative of per-pair magnitudes across 16 pairs → `mean_accel`, `accel_std`, `peak_accel_idx` (extra 3 dims) | ~5 | re-run m04d | +3–5 pp | Surgery's 3-stage curriculum sees temporally-windowed factor masks → encoder learns DYNAMICS not averages |
| 🥉 4 | **E — Divergence/curl** (vector-field invariants) | `∂dx/∂x + ∂dy/∂y` (divergence) + `∂dy/∂x − ∂dx/∂y` (curl) per pair, aggregated (extra 4 dims) | ~10 | re-run m04d | +2–5 pp | Curl is camera-rotation-independent → surgery's D_L (Layout) preserves layout under camera motion |
| 🥉 5 | **F — Motion frequency** (FFT) | Top-3 spectral peaks + dominant freq of 16-pair magnitude series (extra 4 dims) | ~5 | re-run m04d | +1–3 pp | Captures rhythmic motion (gait ~2 Hz, blinker, arm swing). Both encoders likely learn similarly |
| 🥉 6 | **C — Multi-scale temporal flow** | RAFT on (i, i+1), (i, i+2), (i, i+4) pairs → 3 scales (extra 6 dims) | ~30 | ~3× m04d wall (~3 hr) | +1–3 pp | Distinguishes 5 m/s sustained vs 1 m/s jittery (same `mean_mag`). Helps pretrain too — less surgery advantage |

🎯 **Recommendation**: **Option A first** (cheapest gap-widener, no SAM3 dependency). If gap < +10 pp, escalate to **Option D** (full FG/BG via SAM3 — uses m10 masks already produced for surgery → natural progression).

---

## 🛠️ Phase 5 code plan (Option A — foreground motion, ~10 LoC)

### 📂 File 1 — `src/m04d_motion_features.py:217-261` (extend `_aggregate_flow`)

```python
def _aggregate_flow(flow_np, n_pairs):
    dx_all = flow_np[:, 0]                                    # (N, H, W)
    dy_all = flow_np[:, 1]
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
    if hist.sum() > 0: hist = hist / hist.sum()

    # Camera motion: median flow per pair, then median across pairs
    per_pair_dx = np.median(dx_all.reshape(n_pairs, -1), axis=1)
    per_pair_dy = np.median(dy_all.reshape(n_pairs, -1), axis=1)
    cam_x = float(np.median(per_pair_dx))
    cam_y = float(np.median(per_pair_dy))

    # iter13 v13 (NEW): foreground motion = flow MINUS per-pair camera motion.
    # Removes camera-induced global translation → captures agent/object motion only.
    cam_dx_per_pair = per_pair_dx[:, None, None]              # (N, 1, 1) for broadcast
    cam_dy_per_pair = per_pair_dy[:, None, None]
    fg_dx = dx_all - cam_dx_per_pair                          # (N, H, W)
    fg_dy = dy_all - cam_dy_per_pair
    fg_mag = np.sqrt(fg_dx**2 + fg_dy**2)
    fg_ang = np.arctan2(fg_dy, fg_dx)

    fg_mean_mag = float(fg_mag.mean())
    fg_max_mag  = float(fg_mag.max())
    fg_hist, _  = np.histogram(fg_ang.flatten(), bins=8, range=(-np.pi, np.pi))
    fg_hist     = fg_hist.astype(np.float32)
    if fg_hist.sum() > 0: fg_hist = fg_hist / fg_hist.sum()

    return np.array([
        mean_mag, std_mag, max_mag, *hist, cam_x, cam_y,      # existing 13 dims
        fg_mean_mag, fg_max_mag, *fg_hist,                    # NEW 10 dims (total 23-D)
    ], dtype=np.float32)
```

Also update `FEATURE_DIM = 13 → 23` and `FEATURE_NAMES` list at module top (lines ~84-90).

### 📂 File 2 — `src/utils/action_labels.py:66-115` (`parse_optical_flow_class`)

```python
def parse_optical_flow_class(clip_key, flow_features_by_key, magnitude_quartiles):
    vec = flow_features_by_key.get(clip_key)
    if vec is None: return None

    # iter13 v13: bin on FOREGROUND magnitude (vec[13]) instead of global (vec[0])
    # Foreground = flow with camera motion subtracted → captures agent motion classes.
    fg_mean_mag = float(vec[13])
    q1, q2, q3 = magnitude_quartiles
    if   fg_mean_mag < q1: mag_bin = "still"
    elif fg_mean_mag < q2: mag_bin = "slow"
    elif fg_mean_mag < q3: mag_bin = "medium"
    else:                  mag_bin = "fast"

    # FG direction histogram lives at vec[15:23] (after fg_mean_mag, fg_max_mag).
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

Update `compute_magnitude_quartiles` to use `flow_features_array[:, 13]` instead of `[:, 0]`.

### 📂 File 3 — `src/utils/motion_aux_loss.py:75-100` (`MotionAuxHead.n_motion_dims`)

Bump `n_motion_dims` from 13 → 23 in `MotionAuxHead.__init__` default + the wiring in `build_motion_aux_head_from_cfg`. Z-norm buffers (`vec_mean`, `vec_std`) auto-resize via `flow_features.mean(axis=0)` (already shape-agnostic).

---

## 🚀 Phase 5 launch sequence

```bash
# 0. Audit current m04d output dim
python -c "import numpy as np; print(np.load('data/eval_10k_local/motion_features.npy').shape)"
# expect: (9297, 13) → after rebuild: (9297, 23)

# 1. Apply Phase 5 edits (Option A) — ~10-15 LoC across 3 files

# 2. Lint
python -m py_compile src/m04d_motion_features.py src/utils/action_labels.py src/utils/motion_aux_loss.py && \
  ruff check --select F,E9 src/m04d_motion_features.py src/utils/action_labels.py src/utils/motion_aux_loss.py

# 3. Re-run m04d on eval_10k_local (~57 min)
CACHE_POLICY_ALL=2 python -u src/m04d_motion_features.py --FULL \
    --subset data/eval_10k.json --local-data data/eval_10k_local \
    --features-out data/eval_10k_local/motion_features.npy \
    --no-wandb 2>&1 | tee logs/m04d_full_eval10k_v2.log

# 4. Re-run v13 pretrain with new motion_features (~7.7 hr — m09a auto-bootstraps action_labels.json with new FG-binned classes)
tmux new -s p5_pretrain -d "
  CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain --FULL \
    2>&1 | tee logs/probe_train_pretrain_full_v13.log
"

# 5. Re-run surgery on v13 motion_features (Phase 4 wiring assumed in place)
tmux new -s p5_3DI -d "
  CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL \
    2>&1 | tee logs/probe_train_surgery_3stage_DI_full_v15.log
"

# 6. Final 4-encoder eval (~3 GPU-h)
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/probe_eval_full_all4_v7.log
```

💰 **Total Phase 5 cost**: ~57 min m04d + 7.7 hr pretrain + 6-8 hr surgery × 2 + ~3 hr eval = **~25 GPU-h ≈ $20**.

---

## 🎲 Phase 5 decision matrix

| Outcome (vs Phase 4 baseline gap) | Interpretation | Next |
|---|---|---|
| `surgery − pretrain` widens by ≥ +10 pp | 🏆 FG-motion classes expose factor-curriculum advantage | Cite as "surgery dominates pretrain on agent-motion" — paper |
| Widens +5 to +10 pp | Modest improvement, consistent with hypothesis | Escalate to Option D (FG/BG via SAM3) |
| Widens < +5 pp | Camera-subtraction not sufficient discrimination | Escalate to Option D — full SAM3 mask-conditioned flow |
| Surgery REGRESSES | Surgery's encoder couldn't learn agent motion (training instability) | Investigate: Phase 4 λ audit + Stage-3 unfreeze ablation first |

---

## 📚 References

- m04d current state: `src/m04d_motion_features.py:217-261` (`_aggregate_flow`)
- 13D feature names: `src/m04d_motion_features.py:84-90` (`FEATURE_DIM`, `FEATURE_NAMES`)
- Class derivation: `src/utils/action_labels.py:66-115` (`parse_optical_flow_class`) + `:53-63` (`compute_magnitude_quartiles`)
- motion_aux head wiring: `src/utils/motion_aux_loss.py:75-100` (`MotionAuxHead.__init__`)
- m10 SAM3 masks (for Option D escalation): `outputs/full/m10_sam_segment/masks/*.npz`
