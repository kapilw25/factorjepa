# 🎬 iter13 — Phase 4 code plan: motion_aux in m09c surgery

> **Sister docs**: [`plan_v12_motion_aux.md`](./plan_v12_motion_aux.md) · [`runbook.md`](./runbook.md)
  Recommend: train surgery on CURRENT 8-class motion features first. Phase 5 only if Δ < +5 pp.                          
                                                          
  ┌─────────────────────────────────┬────────────────────────────────┬─────────────────────────────┬────────────────┐    
  │              Path               │             GPU-h              │            Risk             │   Recommend    │ 
  ├─────────────────────────────────┼────────────────────────────────┼─────────────────────────────┼────────────────┤ 
  │ A) Train surgery NOW on 8-class │ 6-8 (3DI) + 4-6 (noDI) + 3     │ If pretrain (0.808) is near │                │ 
  │  motion-flow probe              │ (eval) = ~15 GPU-h             │  ceiling, surgery may       │ 🏆 DO FIRST    │ 
  │                                 │                                │ saturate at +1-2 pp         │                │    
  ├─────────────────────────────────┼────────────────────────────────┼─────────────────────────────┼────────────────┤
  │ B) Build Phase 5 first          │ 0.95 (m04d) + 7.7 (v13         │ Sunk cost if surgery        │ Conditional    │    
  │ (FG-motion features,            │ pretrain) + 15 (surgery+eval)  │ already wins on current     │ escalation     │    
  │ re-pretrain v13, then surgery)  │ = ~24 GPU-h                    │ features                    │                │    
  └─────────────────────────────────┴────────────────────────────────┴─────────────────────────────┴────────────────┘    
                                                                              
  Reasoning: surgery_3stage_DI may already widen the gap on current features (D_A/D_I tubes give factor inductive bias
  that pretrain lacks). Don't pay Phase 5's 9 extra GPU-h until we know.                                                 
                                                          

---

## 📋 Table 1 — Recipe diff (m09a v12 vs m09c current)

| Axis | 🟢 m09a v12 (done, 0.808) | 🛠️ m09c surgery_3stage_DI (current) | Phase 4 target |
|---|---|---|---|
| Data | raw `eval_10k_train_split.json` (6.5 K) | factor variants D_L / D_A / D_I (m10 + m11 streaming) | unchanged |
| Schedule | flat 5 epochs | 3-stage 40/30/30 % w/ unfreeze 25→50→50 % | unchanged |
| Mode mixture | n/a | s1: 100 L · s2: 30 L + 70 A · s3: 15 L + 15 A + 70 I | unchanged |
| Aux loss | **motion_aux** (CE+MSE) | `multi_task_probe` (16 retrieval dims) ❌ | **swap → motion_aux** |
| λ drift anchor | 0 | inherits `surgery_base.yaml` — **AUDIT NEEDED** | drop to 0 if not already |
| best_state | probe_top1 | probe_top1 (assume) — verify | verify |

---

## 📋 Table 2 — Phase 4 levers (ranked by expected gain)

| # | Lever | Files | LoC | GPU | Expected gain |
|---|---|---|---|---|---|
| 1 | **Enable motion_aux in `surgery_3stage_DI.yaml` + `surgery_2stage_noDI.yaml` + `surgery_base.yaml`** + wire 9 call sites in `m09c_surgery.py` (mirror m09a's wiring exactly) | `configs/train/surgery_*.yaml`, `src/m09c_surgery.py` | ~50 | 6-8 GPU-h surgery run × 2 variants | 🏆 **LARGE** — paper-grade combined result |
| 2 | **λ audit**: read `configs/train/surgery_base.yaml` `drift_control.lambda_reg`. If > 0, set to 0.0 + `enabled: false`. Per-yaml + per-mode. | `configs/train/surgery_base.yaml` | ~3 | 0 (config only) | MEDIUM — possibly avoids anchor saturation |
| 3 | **Stage 3 unfreeze 50 → 75 %** (iter8 originally hurt with 0.75; motion_aux may stabilize) | `surgery_3stage_DI.yaml:56` | ~1 | 6-8 GPU-h ablation | UNCERTAIN — fire only as v15 ablation if v14 plateaus |

---

## 🛠️ Phase 4 code plan (Levers 1 + 2)

### File 1 — `configs/train/surgery_base.yaml` (audit + fix λ)
```bash
# Audit current λ (1-line check)
grep -nE "drift_control|lambda_reg|enabled" configs/train/surgery_base.yaml | head -10
```
**Edit (if λ != 0)**: mirror `probe_pretrain.yaml:113-132` block — set `drift_control.enabled: false` + `lambda_reg: 0.0`.

### File 2 — `configs/train/surgery_3stage_DI.yaml` (swap aux loss)
```yaml
# REPLACE this block (current lines ~60-71):
multi_task_probe:
  enabled: {sanity: true, poc: true, full: true}

# WITH:
multi_task_probe:
  enabled: {sanity: false, poc: false, full: false}   # disabled — retrieval gradient stole capacity in v11

motion_aux:                                            # mirror probe_pretrain.yaml:176-189
  enabled:
    sanity: true
    poc:    true
    full:   true
  motion_features_path: data/eval_10k_local/motion_features.npy
  action_labels_path:   outputs/full/probe_action/action_labels.json
  weight_motion: 0.1
  weight_ce:     1.0
  weight_mse:    1.0
  head:
    hidden_dim: 256
    dropout:    0.1
  head_lr_multiplier: 10.0
```

### File 3 — `configs/train/surgery_2stage_noDI.yaml` (same swap, ablation control)
Same edit as File 2.

### File 4 — `src/m09c_surgery.py` (9 call sites mirroring m09a v12)
```python
# IMPORTS (~line 109, next to motion_aux_loss already imported by m09a):
from utils.motion_aux_loss import (
    merge_motion_aux_config, build_motion_aux_head_from_cfg,
    attach_motion_aux_to_optimizer, run_motion_aux_step,
    export_motion_aux_head,
)

# CLI args (next to --no-multi-task in m09c's build_parser):
parser.add_argument("--motion-features-path", type=Path, default=None)
parser.add_argument("--no-motion-aux", action="store_true")

# After cfg parse (next to merge_multi_task_config call):
merge_motion_aux_config(cfg, args, mode_key)

# After build_multi_task_head_from_cfg (find line):
ma_head, ma_lookup, ma_cfg = build_motion_aux_head_from_cfg(cfg, device)

# CRITICAL — surgery rebuilds optimizer per stage. Re-attach motion_aux head
# inside the per-stage build_optimizer block (mirrors attach_head_to_optimizer call):
attach_motion_aux_to_optimizer(optimizer, ma_head, ma_cfg, base_lr=cfg["optimization"]["lr"])

# In training loop, after run_multi_task_step (find line ~920):
try:
    ma_loss_val, ma_per_branch = run_motion_aux_step(
        student, ma_head, ma_cfg, ma_lookup,
        batch_clips, batch_keys, scaler, mp_cfg, dtype, device)
except torch.cuda.OutOfMemoryError:
    optimizer.zero_grad()
    print(f"  OOM at step {step} (motion_aux fwd): macro discarded, retrying")
    torch.cuda.empty_cache()
    continue

# step_record + wb_metrics logging (mirror m09a v12):
if ma_head is not None:
    step_record["loss_motion_aux"]     = round(ma_loss_val, 6)
    step_record["loss_motion_aux_ce"]  = round(ma_per_branch.get("ce", 0.0), 6)
    step_record["loss_motion_aux_mse"] = round(ma_per_branch.get("mse", 0.0), 6)
    step_record["motion_aux_n_kept"]   = ma_per_branch.get("n_kept", 0)
    wb_metrics["loss/motion_aux"]      = ma_loss_val
    wb_metrics["loss/motion_aux/ce"]   = ma_per_branch.get("ce", 0.0)
    wb_metrics["loss/motion_aux/mse"]  = ma_per_branch.get("mse", 0.0)

# End-of-train (next to export_multi_task_head):
export_motion_aux_head(ma_head, output_dir / "motion_aux_head.pt")
```

### File 5 — `scripts/run_probe_train.sh` (thread `--motion-features-path` for surgery)
```bash
# In `surgery_3stage_DI` and `surgery_noDI` cases (line ~212), add:
--motion-features-path "${LOCAL_DATA}/motion_features.npy" \
```

---

## 🚀 Phase 4 launch sequence

```bash
# 0. Wait for v5 eval to finish + read paired-Δ gate (m09a v12 vs frozen)
jq '.pairwise_deltas.vjepa_2_1_pretrain_minus_vjepa_2_1_frozen' \
  outputs/full/probe_action/probe_paired_delta.json

# 1. Apply Phase 4 edits (Levers 1 + 2) — ~50 LoC, ~30 min coding
# 2. Lint
python -m py_compile src/m09c_surgery.py && \
  ruff check --select F,E9 src/m09c_surgery.py && \
  bash -n scripts/run_probe_train.sh

# 3. SANITY (~10 min) — verify motion_aux fires through 3-stage optimizer rebuilds
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --SANITY \
  2>&1 | tee logs/probe_train_surgery_3stage_DI_sanity_v1.log
# expect: [motion_aux] enabled: 8 classes ... line × 3 (once per stage rebuild)

# 4. v14 surgery_3stage_DI FULL (~6-8 GPU-h)
tmux new -s p4_3DI -d "
  CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL \
    2>&1 | tee logs/probe_train_surgery_3stage_DI_full_v14.log
"

# 5. v14 surgery_noDI FULL (~4-6 GPU-h, fires after v14 3DI completes)
tmux new -s p4_noDI -d "
  CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI --FULL \
    2>&1 | tee logs/probe_train_surgery_noDI_full_v14.log
"

# 6. Final 4-encoder eval (~3 GPU-h)
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/probe_eval_full_all4_v6.log

# 7. Read paper-grade gate
jq '.pairwise_deltas | to_entries[] | select(.key | contains("surgical"))' \
  outputs/full/probe_action/probe_paired_delta.json
```

---

## 🎲 Decision matrix after Phase 4

| Outcome | Interpretation | Paper claim |
|---|---|---|
| `surgery_3DI ≫ pretrain ≫ frozen` (≥ +5 pp each) | 🏆 factor curriculum + motion supervision both contribute | "Factor surgery with motion-aware aux loss" |
| `surgery_3DI ≈ pretrain ≫ frozen` | motion_aux is the lever; surgery curriculum redundant | Cite m09a pretrain only |
| `surgery_noDI ≈ surgery_3DI` | D_I tubes don't help (iter9 v15c finding holds) | Drop D_I from paper |
| `surgery_3DI < pretrain` | surgery hurts on motion task (over-destabilization) | Investigate λ + Stage 3 unfreeze |

---

## 📚 References

- v12 motion_aux design: [`plan_v12_motion_aux.md`](./plan_v12_motion_aux.md)
- v12 final result: `outputs/full/probe_pretrain/probe_history.jsonl` (top1 0.439 → 0.808 across 10 vals; val_jepa stable)
- m09a wiring template: `src/m09a_pretrain.py` (lines ~120, 245, 530, 618, 973, 1053, 1369 — 9 call sites)
- m09c current state: `src/m09c_surgery.py` + `configs/train/surgery_*.yaml`

---

# 🧬 Phase 5 — Harder motion features in m04d (widen surgery vs pretrain gap)

> **Why**: current m04d output is too summary-statistical (`mean_mag` + `dir_hist` argmax → 8 classes recoverable from 2-3 numbers). Pretrain's 0.808 top-1 means it learned the summary stats, but surgery's factor curriculum (D_L/D_A/D_I) has no opportunity to express its agent/interaction inductive bias. Harder features → larger surgery vs pretrain gap.

## 📋 Table 3 — Feature axes ranked by surgery-vs-pretrain gap-widening

| 🆔 | Feature axis | What m04d adds | LoC | Effort | Δ-gap expected | Why surgery wins |
|---|---|---|---|---|---|---|
| 🏆 1 | **D — FG/BG flow via SAM3 masks** | Compute flow stats SEPARATELY for person/vehicle pixels (m10 SAM3) vs background (extra 13 dims FG, kept 13 BG = 26-D vec) | ~50 | m04d wall + ~6 GPU-h m10 prereq | **+10–20 pp** | 🏆🏆 Surgery's D_A/D_I tubes ARE agent/interaction segmentations → 1:1 alignment. Pretrain has zero structural reason to encode FG vs BG separately |
| 🏆 2 | **A — Foreground motion (camera-subtracted)** | Per-pair flow MINUS `camera_motion_{x,y}` → `fg_mean_mag`, `fg_max_mag`, `fg_dir_hist` (extra 10 dims) | ~10 | re-run m04d (~57 min) | **+5–10 pp** | Cheap subset of D (no SAM3). Surgery's D_A trains on agent-only crops → encoder learns agent motion independent of camera |
| 🥈 3 | **B — Acceleration** (2nd-order temporal) | Time-derivative of per-pair magnitudes across 16 pairs → `mean_accel`, `accel_std`, `peak_accel_idx` (extra 3 dims) | ~5 | re-run m04d | +3–5 pp | Surgery's 3-stage curriculum sees temporally-windowed factor masks → encoder learns DYNAMICS not averages |
| 🥉 4 | **E — Divergence/curl** (vector-field invariants) | `∂dx/∂x + ∂dy/∂y` (divergence) + `∂dy/∂x − ∂dx/∂y` (curl) per pair, aggregated (extra 4 dims) | ~10 | re-run m04d | +2–5 pp | Curl is camera-rotation-independent → surgery's D_L (Layout) preserves layout under camera motion |
| 🥉 5 | **F — Motion frequency** (FFT) | Top-3 spectral peaks + dominant freq of 16-pair magnitude series (extra 4 dims) | ~5 | re-run m04d | +1–3 pp | Captures rhythmic motion (gait ~2 Hz, blinker, arm swing). Both encoders likely learn similarly |
| 🥉 6 | **C — Multi-scale temporal flow** | RAFT on (i, i+1), (i, i+2), (i, i+4) pairs → 3 scales (extra 6 dims) | ~30 | ~3× m04d wall (~3 hr) | +1–3 pp | Distinguishes 5 m/s sustained vs 1 m/s jittery (same `mean_mag`). Helps pretrain too — less surgery advantage |

**🎯 Recommendation**: **Option A first** (cheapest gap-widener, no SAM3 dependency). If gap < +10 pp, escalate to **Option D** (full FG/BG via SAM3 — uses m10 masks already produced for surgery → natural progression).

---

## 🛠️ Phase 5 code plan (Option A — foreground motion)

### File 1 — `src/m04d_motion_features.py:217-261` (extend `_aggregate_flow`)

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

### File 2 — `src/utils/action_labels.py:66-115` (`parse_optical_flow_class`)

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

### File 3 — `src/utils/motion_aux_loss.py:75-100` (`MotionAuxHead.n_motion_dims`)

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

# 5. After Phase 4 (surgery + motion_aux) is also coded — re-run surgery on v13 motion_features
tmux new -s p5_3DI -d "
  CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL \
    2>&1 | tee logs/probe_train_surgery_3stage_DI_full_v15.log
"

# 6. Final 4-encoder eval (~3 GPU-h)
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/probe_eval_full_all4_v7.log
```

**Total Phase 5 cost**: ~57 min m04d + 7.7 hr pretrain + 6-8 hr surgery × 2 + ~3 hr eval = **~25 GPU-h ≈ $20**.

---

## 🎲 Phase 5 decision matrix

| Outcome (vs Phase 4 baseline gap) | Interpretation | Next |
|---|---|---|
| `surgery − pretrain` widens by ≥ +10 pp | 🏆 FG-motion classes expose factor-curriculum advantage | Cite as "surgery dominates pretrain on agent-motion" — paper |
| Widens +5 to +10 pp | Modest improvement, consistent with hypothesis | Escalate to Option D (FG/BG via SAM3) |
| Widens < +5 pp | Camera-subtraction not sufficient discrimination | Escalate to Option D — full SAM3 mask-conditioned flow |
| Surgery REGRESSES | Surgery's encoder couldn't learn agent motion (training instability) | Investigate: Phase 4 λ audit + Stage-3 unfreeze ablation first |

---

## 📚 Phase 5 references

- m04d current state: `src/m04d_motion_features.py:217-261` (`_aggregate_flow`)
- 13D feature names: `src/m04d_motion_features.py:84-90` (`FEATURE_DIM`, `FEATURE_NAMES`)
- Class derivation: `src/utils/action_labels.py:66-115` (`parse_optical_flow_class`) + `:53-63` (`compute_magnitude_quartiles`)
- motion_aux head wiring: `src/utils/motion_aux_loss.py:75-100` (`MotionAuxHead.__init__`)
- m10 SAM3 masks (for Option D): `outputs/full/m10_sam_segment/masks/*.npz`
