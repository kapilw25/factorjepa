# 🎬 Phase 4 — Wire `motion_aux` into `m09c_surgery.py`

> **🎯 Goal**: extend the `motion_aux` aux-loss recipe that lifted `m09a_pretrain` to `probe_top1 = 0.808` into the `m09c` surgery pipeline so the **factor curriculum × motion supervision** combination feeds the iter14 `surgery > pretrain > frozen` paper claim.

> **Sister docs**: [`plan_HIGH_LEVEL.md`](./plan_HIGH_LEVEL.md) · [`plan_surgery_on_pretrain.md`](./plan_surgery_on_pretrain.md) · [`plan_phase5_fg_motion_features.md`](./plan_phase5_fg_motion_features.md) (conditional escalation)

---

## 🛣️ Path decision (Phase 4 first, Phase 5 only on null result)

| Path | GPU-h | Risk | Recommend |
|---|---|---|---|
| **A — Train surgery NOW on existing 8-class motion-flow probe** | 6-8 (3DI) + 4-6 (noDI) + 3 (eval) = **~15 GPU-h** | If pretrain (0.808) is near ceiling, surgery may saturate at +1-2 pp | 🏆 **DO FIRST** |
| B — Build Phase 5 first (FG-motion features, re-pretrain v13, then surgery) | 0.95 (m04d) + 7.7 (v13 pretrain) + 15 (surgery+eval) = ~24 GPU-h | Sunk cost if surgery already wins on current features | Conditional escalation |

📝 **Rationale**: surgery_3stage_DI may already widen the gap on current features (D_A/D_I tubes give factor inductive bias that pretrain lacks). Don't pay Phase 5's 9 extra GPU-h until we know.

---

## 📋 Table 1 — Recipe diff (m09a v12 vs current m09c)

| Axis | 🟢 m09a v12 (done, 0.808) | 🛠️ m09c surgery_3stage_DI (current) | Phase 4 target |
|---|---|---|---|
| Data | raw `eval_10k_train_split.json` (6.5 K) | factor variants D_L / D_A / D_I (m10 + m11 streaming) | unchanged |
| Schedule | flat 5 epochs | 3-stage 40/30/30 % w/ unfreeze 25→50→50 % | unchanged |
| Mode mixture | n/a | s1: 100 L · s2: 30 L + 70 A · s3: 15 L + 15 A + 70 I | unchanged |
| Aux loss | **`motion_aux` (CE+MSE)** | `multi_task_probe` (16 retrieval dims) ❌ | **swap → `motion_aux`** |
| λ drift anchor | 0 | inherits `surgery_base.yaml` — **AUDIT NEEDED** | drop to 0 if not already |
| `best_state` | probe_top1 | probe_top1 (assume) — verify | verify |

---

## 📋 Table 2 — Phase 4 levers (ranked by expected gain)

| # | Lever | Files | LoC | GPU | Expected gain |
|---|---|---|---|---|---|
| 1️⃣ | **Enable `motion_aux`** in `surgery_3stage_DI.yaml` + `surgery_2stage_noDI.yaml` + `surgery_base.yaml` + wire 9 call sites in `m09c_surgery.py` (mirror m09a's wiring exactly) | `configs/train/surgery_*.yaml`, `src/m09c_surgery.py` | ~50 | 6-8 GPU-h × 2 variants | 🏆 **LARGE** — paper-grade combined result |
| 2️⃣ | **λ audit**: read `configs/train/surgery_base.yaml` `drift_control.lambda_reg`. If > 0, set to 0.0 + `enabled: false`. Per-yaml + per-mode. | `configs/train/surgery_base.yaml` | ~3 | 0 (config only) | MEDIUM — possibly avoids anchor saturation |
| 3️⃣ | Stage 3 unfreeze 50 → 75 % (iter8 originally hurt at 0.75; motion_aux may stabilize) | `surgery_3stage_DI.yaml:56` | ~1 | 6-8 GPU-h ablation | UNCERTAIN — fire only as v15 ablation if v14 plateaus |

---

## 🛠️ Phase 4 code plan (Levers 1 + 2)

### 📂 File 1 — `configs/train/surgery_base.yaml` (audit + fix λ)

```bash
grep -nE "drift_control|lambda_reg|enabled" configs/train/surgery_base.yaml | head -10
```

**Edit (if λ != 0)**: mirror `probe_pretrain.yaml:113-132` block — set `drift_control.enabled: false` + `lambda_reg: 0.0`.

### 📂 File 2 — `configs/train/surgery_3stage_DI.yaml` (swap aux loss)

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

### 📂 File 3 — `configs/train/surgery_2stage_noDI.yaml` (mirror File 2)

Same edit as File 2.

### 📂 File 4 — `src/m09c_surgery.py` (9 call sites mirroring m09a v12)

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

### 📂 File 5 — `scripts/run_probe_train.sh` (thread `--motion-features-path` for surgery)

```bash
# In `surgery_3stage_DI` and `surgery_noDI` cases (line ~212), add:
--motion-features-path "${LOCAL_DATA}/motion_features.npy" \
```

---

## 🚀 Phase 4 launch sequence

```bash
# 0. Confirm v5 eval Δ (m09a v12 vs frozen) is on disk
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
# expect: "[motion_aux] enabled: 8 classes ..." line × 3 (once per stage rebuild)

# 4. v14 surgery_3stage_DI FULL (~6-8 GPU-h)
tmux new -s p4_3DI -d "
  CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL \
    2>&1 | tee logs/probe_train_surgery_3stage_DI_full_v14.log
"

# 5. v14 surgery_noDI FULL (~4-6 GPU-h, after v14 3DI completes)
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
| **Δ (surgery_3DI − pretrain) < +5 pp** | current motion features may be too summary-statistical | ⏭️ **Escalate to [Phase 5](./plan_phase5_fg_motion_features.md)** |

---

## 📚 References

- v12 motion_aux design: [`plan_v12_motion_aux.md`](./plan_v12_motion_aux.md)
- v12 final result: `outputs/full/m09a_pretrain/probe_history.jsonl` (top1 0.439 → 0.808 across 10 vals)
- m09a wiring template: `src/m09a_pretrain.py` (lines ~120, 245, 530, 618, 973, 1053, 1369 — 9 call sites)
- m09c current state: `src/m09c_surgery.py` + `configs/train/surgery_*.yaml`
