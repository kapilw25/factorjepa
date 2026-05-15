# 🏗️ Plan — Extract Shared Training-Loop Body to `utils/training_loop.py`

> 🎯 **Goal**: m09a and m09c become **THIN dispatchers** over a shared loop with technique-specific hooks. Forgetting to backport to one module becomes structurally **impossible**.

---

## 📋 Context — Why This Refactor Now

`m09a_pretrain.py` (1593 LoC) and `m09c_surgery.py` (1717 LoC) each hand-roll their own ~600-700 LoC `train()` body. Per-iter additions to m09a (the **gold standard** 🥇) silently miss m09c, accumulating divergence:

### 🐛 4 Root-Cause Bug Classes

| 🚨 Bug class | 📍 Where it lives | 💥 Examples this iter |
|---|---|---|
| **Schema drift** (CSV cols, step_record keys) | Per-module step builder | `loss_total` KeyError ❌, `monitoring` KeyError ❌, missing `block_drift_mean` ⚠️, missing `throughput` ⚠️ |
| **Plot rendering forks** | Per-module render call site | `m09c_val_loss.png` 3-panel legacy 🗑️, `m09c_drift_loss.png` redundant 🗑️, `val_split.json` artifact 🗑️, missing `probe_trajectory_trio` 🆕 |
| **Import path drift** | Per-module top-of-file imports | `compute_block_drift` from wrong module 📦 (m09a uses local import; my m09c fix used wrong top-level path) |
| **Best-ckpt / early-stop forks** | Per-module probe-cycle logic | E8 top@1-only refactor required touching **4 files** 📁📁📁📁 |

> 📊 **Two full audits this session caught 30+ surface divergences.** Each fix has been a leaf-level patch when the trunk needed surgery. **Per-iter, the gap GROWS.** 📈

---

## 🏛️ Architecture: Hook-Based Shared Loop

### ❌ NOT a `TrainingLoop` Base Class
m09a (single-pass) and m09c (multi-stage) are **orthogonal control flows**. Inheritance would create `if isinstance(self, Surgery)` smells (CLAUDE.md rule 49 violation 🚫).

### ✅ YES a Function with Hooks Dict

```python
run_training_loop(state, cfg, hooks, args)
```

🧬 **Mechanism:**
- 📦 `state` = mutable dict (model, optimizer, scheduler, data_iter, probe_history, …)
- 🪝 `hooks` = dict of callables for technique-specific logic
- 🅰️ m09a passes `stages=[None]` (single-pass)
- 🅲 m09c passes `stages=cfg["surgery"]["stages"]` (2 or 3 stages)
- 🔄 m09a's `on_stage_begin` is a no-op; m09c's mutates state to install per-stage optimizer/data/trainable-prefix
- ➰ Same `for stage in stages: for step in range(...)` shape works for **both**; m09a degenerates to one outer iteration

### 🧩 Core Loop Skeleton

```python
# src/utils/training_loop.py — NEW (~450 LoC)
def run_training_loop(state: dict, cfg: dict, hooks: dict, args) -> dict:
    """Generic training loop. Iterates stages × steps, dispatches to hooks at
    lifecycle events. Returns final summary dict."""
    stages = state.get("stages", [None])  # m09a: [None]; m09c: list of stage_cfgs
    global_step = state.get("start_step", 0)
    summary = {"steps": 0, "stages_run": 0, "early_stopped": False}

    hooks["on_training_start"](state, global_step)

    for stage_idx, stage_cfg in enumerate(stages):
        # 🎬 Stage setup (m09c: rebuild optimizer + data_iter + trainable prefix)
        # m09a: no-op (single stage)
        hooks["on_stage_begin"](state, stage_idx, stage_cfg)

        stage_steps = state["stage_steps"]
        for local_step in range(stage_steps):
            # 📦 Get batch (technique-specific iterator)
            batch_clips, batch_keys = next(state["data_iter"])

            # ⚙️ Shared step body (calls _train_step_grad_accum + MT + MA + optim)
            loss_dict = _run_step_inner(state, batch_clips, batch_keys, cfg)

            # 📝 Step record (canonical schema; replaces hand-rolled per-module versions)
            step_record = build_step_record(
                global_step=global_step, stage_idx=stage_idx,
                stage_name=stage_cfg["name"] if stage_cfg else None,
                loss_dict=loss_dict, state=state, cfg=cfg)
            hooks["on_step_end"](state, step_record, global_step)

            # 🚨 NaN guard (shared)
            if not _check_loss_finite(loss_dict, state):
                if hooks["on_nan_detected"](state, loss_dict, global_step):
                    return _emergency_summary(state, summary)

            # 🔬 Probe cadence
            if hooks["should_run_probe"](state, global_step, stage_idx,
                                         is_stage_boundary=False):
                probe_record = _run_probe_cycle(state, cfg, global_step, stage_idx)
                hooks["on_val_cycle"](state, probe_record, global_step, stage_idx)
                if hooks["should_trigger_early_stop"](state, probe_record):
                    summary["early_stopped"] = True
                    return summary

            global_step += 1

        # 🏁 Stage end (m09c: forced boundary probe + per-stage ckpt; m09a: no-op)
        hooks["on_stage_end"](state, stage_idx, global_step)

    summary["steps"] = global_step
    summary["stages_run"] = len(stages)
    hooks["on_training_end"](state, summary)
    return summary
```

---

## 🪝 Hook Contract — 8 Essentials

> 🎯 The audit agent proposed 30 hooks. Consolidated to **8** that cover every observed m09a/m09c divergence. Anything finer-grained can be added later.

### 1️⃣ `on_training_start`
- 📡 **Signature**: `(state, global_step) → None`
- 🅰️ **m09a**: print banner + setup wandb run name with λ
- 🅲 **m09c**: print banner + factor preflight summary

### 2️⃣ `on_stage_begin`
- 📡 **Signature**: `(state, stage_idx, stage_cfg) → None`
- 🅰️ **m09a**: First call: build optimizer + scheduler + data_producer **once**. Subsequent calls: skip (m09a has 1 stage = `[None]`)
- 🅲 **m09c**: Per stage: 🧹 GC cleanup if idx>0, ♻️ rebuild optimizer (only trainable params), 🔄 rebuild scheduler (per-stage warmup), 🆕 rebuild data sampler (FactorSampler or StreamingFactorDataset), 🧊 set_trainable_prefix

### 3️⃣ `on_step_end`
- 📡 **Signature**: `(state, step_record, global_step) → None`
- 🅰️ **m09a**: append to JSONL + CSV; tqdm postfix; periodic GC; periodic ckpt save (every ckpt_interval)
- 🅲 **m09c**: same JSONL + CSV; tqdm postfix; periodic GC. **NO** per-step ckpt — m09c uses stage-boundary ckpt only

### 4️⃣ `on_nan_detected`
- 📡 **Signature**: `(state, loss_dict, global_step) → bool` ⚠️ (return True = emergency exit)
- 🅰️ **m09a**: track nan_strikes; FATAL after 3 consecutive 💀
- 🅲 **m09c**: same

### 5️⃣ `should_run_probe`
- 📡 **Signature**: `(state, global_step, stage_idx, is_stage_boundary) → bool`
- 🅰️ **m09a**: `(global_step + 1) % val_interval == 0`
- 🅲 **m09c**: `global_step % probe_every == 0` OR `is_stage_boundary`

### 6️⃣ `on_val_cycle`
- 📡 **Signature**: `(state, probe_record, global_step, stage_idx) → None`
- 🅰️ **m09a**: 🎨 plot suite (training_curves + combined_losses + probe_trajectory_trio + val_loss_with_kill_switch_overlay); update best_state via `apply_val_cycle_triggers`
- 🅲 **m09c**: ✨ same plot suite (now identical via post-iter14 unification); same `apply_val_cycle_triggers`

### 7️⃣ `should_trigger_early_stop`
- 📡 **Signature**: `(state, probe_record) → bool`
- 🅰️ **m09a**: `kill_state["triggered"]` (top@1 plateau)
- 🅲 **m09c**: same

### 8️⃣ `on_stage_end`
- 📡 **Signature**: `(state, stage_idx, global_step) → None`
- 🅰️ **m09a**: 🚫 no-op (single-stage)
- 🅲 **m09c**: 💾 save `m09c_ckpt_stage{N}.pt`, 🧹 cleanup_stage_checkpoints, 🔬 force boundary probe

### 9️⃣ `on_training_end` *(bonus)*
- 📡 **Signature**: `(state, summary) → None`
- 🅰️ **m09a**: ❄️ cooldown phase (frame switch + linear LR decay); export_student; finalize_training; write training_summary.json
- 🅲 **m09c**: 🏆 best-ckpt promotion (move student_best.pt → student_encoder.pt); export_student; finalize_training; trajectory_stats; write training_summary.json

---

## 🔧 New Shared Utilities

> 📌 Add to `src/utils/training_loop.py` or fold into `utils/training.py` (TBD by reviewer).

```python
def make_step_logger(jsonl_path: Path, csv_writer) -> callable:
    """🪵 Returns _log_step closure with fsync.
    Extracted from m09a:743 == m09c:767, byte-identical."""

def build_step_record(*, global_step, stage_idx, stage_name,
                      loss_dict, state, cfg) -> dict:
    """📋 Canonical schema. Includes all keys both m09a + m09c need:
    {step, stage, epoch, loss_jepa, loss_masked, loss_context, loss_drift,
     loss_total, loss_multi_task, loss_motion_aux, lr, grad_norm, throughput,
     block_drift_mean, uw_w_jepa, uw_w_infonce, uw_w_tcc}.
    Conditional keys (mt/ma) added when corresponding head exists."""

def build_training_csv_header(technique: str) -> list[str]:
    """📊 Returns canonical column list per technique.
    Fixes 9-col vs 12-col drift."""

def _run_step_inner(state: dict, batch_clips, batch_keys, cfg) -> dict:
    """⚙️ Shared inner step: _train_step_grad_accum + MT + MA + optim.step + EMA.
    Replaces ~150 LoC duplicated in m09a:854-936 and m09c:1215-1317."""

def _run_probe_cycle(state, cfg, global_step, stage_idx) -> dict:
    """🔬 Shared probe cycle: run_validation OR run_probe_val_loss
    + run_trio_at_val + track_block_drift_at_val + append to probe_history."""
```

---

## 🅰️ Migration: m09a as Thin Dispatcher

📉 **Before**: `def train(cfg, args):` ~1000 LoC of hand-rolled loop
📈 **After**: `def train(cfg, args):` ~150 LoC — most of it building `state` + `hooks`, then calling shared loop

```python
# src/m09a1_pretrain_encoder.py:train() — POST refactor (~150 LoC vs current ~1000)
def train(cfg, args):
    # 1️⃣ Setup: model, data_subset, optimizer, scheduler, producer, probe (~80 LoC)
    state = _setup_m09a_state(cfg, args)
    # state["stages"] = [None]   (single-pass)
    # state["data_iter"] = ProducerIter(producer_thread)   (m09a-specific)

    # 2️⃣ Hooks (~50 LoC of small bindings)
    hooks = {
        "on_training_start":         lambda s, gs: _m09a_print_banner(s, cfg),
        "on_stage_begin":            _m09a_on_stage_begin,
        "on_step_end":               _m09a_on_step_end,
        "on_nan_detected":           _m09a_on_nan,
        "should_run_probe":          lambda s, gs, si, sb: (gs + 1) % cfg["val_interval"] == 0,
        "on_val_cycle":              _m09a_on_val_cycle,
        "should_trigger_early_stop": lambda s, pr: s["kill_state"]["triggered"],
        "on_stage_end":              lambda s, si, gs: None,   # 🚫 no-op
        "on_training_end":           _m09a_cooldown_and_finalize,
    }

    # 3️⃣ Run shared loop
    return run_training_loop(state, cfg, hooks, args)
```

📊 **Lines removed from m09a**: ~600
📊 **Lines added (hooks + bindings)**: ~200
📊 **Net m09a**: ~993 LoC (down from 1593) ⬇️

---

## 🅲 Migration: m09c as Thin Dispatcher

```python
# src/m09c1_surgery_encoder.py:train() — POST refactor (~200 LoC vs current ~1100)
def train(cfg, args):
    state = _setup_m09c_state(cfg, args)
    state["stages"] = cfg["surgery"]["stages"]  # 2 or 3 stages
    state["data_iter_factory"] = lambda stage_cfg: build_factor_iter(stage_cfg, ...)

    hooks = {
        "on_training_start":         _m09c_print_banner,
        "on_stage_begin":            _m09c_on_stage_begin,
        "on_step_end":               _m09c_on_step_end,
        "on_nan_detected":           _m09c_on_nan,
        "should_run_probe":          _m09c_should_run_probe,
        "on_val_cycle":              _m09c_on_val_cycle,
        "should_trigger_early_stop": _m09c_should_early_stop,
        "on_stage_end":              _m09c_on_stage_end,         # 💾 stage ckpt
        "on_training_end":           _m09c_finalize,             # 🏆 best promotion
    }

    return run_training_loop(state, cfg, hooks, args)
```

📊 **Lines removed from m09c**: ~700
📊 **Lines added**: ~250
📊 **Net m09c**: ~1267 LoC (down from 1717) ⬇️

---

## 📁 Files to Modify

| 📄 File | 🛠️ Action | ➕➖ LoC Δ |
|---|---|---|
| 🆕 `src/utils/training_loop.py` | **NEW** — `run_training_loop()` + 4 helpers | **+450** |
| ✏️ `src/utils/training.py` | Add `build_step_record()`, `build_training_csv_header()`, `make_step_logger()` extracted from m09a/m09c | **+80** |
| 🅰️ `src/m09a1_pretrain_encoder.py` | Refactor `train()` to thin dispatcher; keep `_m09a_*` hooks in same file | **-600** |
| 🅲 `src/m09c1_surgery_encoder.py` | Refactor `train()` to thin dispatcher; keep `_m09c_*` hooks in same file | **-450** |
| 🗄️ `src/legacy/m09b_explora.py` | NO CHANGE (legacy) | **0** |
| 📚 `iter/iter14_surgery_on_pretrain/plan_No_discrepancy.md` | Update to reference new architecture | **+20** |

> 🎯 **Net codebase**: −520 LoC (5847 → ~5300), but the structural-bug surface drops to **near-zero** ✨

---

## ♻️ Existing Utilities to Reuse (Don't Reinvent)

> ✅ These already exist in `src/utils/` and the new orchestrator should call them:

### From `utils.training`
- ⚙️ `_train_step_grad_accum()` — forward + backward (line 492-613)
- 🧮 `compute_total_loss()` — canonical loss formula (line 477) ✨ *added this iter*
- 🧠 `update_teacher_ema()` — EMA update (line 615)
- 🛠️ `build_optimizer()` — optimizer factory (line 632)
- 📉 `build_scheduler()` — LR scheduler (line 727)
- 🧪 `run_validation()` — m09a's val (line 771)
- 🧪 `run_probe_val_loss()` — m09c's val (line 2101)
- 🎯 `run_trio_at_val()` — probe-trio (line 2533)
- 📏 `track_block_drift_at_val()` — drift metric (line 2452)
- 🚨 `apply_val_cycle_triggers()` — early-stop dispatcher (line 1252)
- 💾 `save_training_checkpoint()` / `cleanup_old_checkpoints()` / `cleanup_stage_checkpoints()` (lines 981, 1058, 1086)
- 🧊 `set_trainable_prefix()` — m09c stage unfreezing (line 1368)
- 🏁 `export_student_for_eval()` / `finalize_training()` (lines 1135, 1316)

### From `utils.plots`
- 📈 `plot_training_curves` / `plot_combined_losses` / `plot_probe_trajectory_trio` / `plot_val_loss_with_kill_switch_overlay` / `compute_block_drift` ✨ *all iter14-unified*

---

## 🔒 Decisions (Locked by User)

> ✅ **Rollout scope**: Full **4-phase A→B→C→D** in this session/iter (highest dedup ~520 LoC; structurally fixes the bug class)
>
> ✅ **Numerical fidelity gate**: **±0.5 pp tolerance** on `probe_top1` between pre-refactor (current POC: m09a=0.4585, m09c=0.7449) and post-refactor. Anything outside ±0.5 pp blocks merge to next phase.

---

## 🚀 Phased Rollout — 4 Phases with Verification Gates

> 🆕 **Recipe-v2 wiring (2026-05-09)** — see `plan_surgery_wins.md` § 0 master action items table. The 9 recipe-v2 fixes (frozen teacher, LP-FT, surgical layer subset, SPD, replay, scheduled EMA, saliency-weighted loss, unified warmup, paper docs) are sequenced into the phases below. The hook contract MUST be designed to support all 9 BEFORE Phase A code is written.

### 🟣 Phase 0 — Recipe-v2 POC (NEW, BEFORE Phase A)
- 🧪 Run `{teacher: EMA vs FROZEN} × {LP-FT: off vs on}` 4-cell sweep on m09c POC ($1, ~1.5 GPU-h)
- 🚦 **Gate**: at least one cell shows trio top-1 ≥ 0.808 by step 5 → recipe v2 unblocks paper goal
- ⛔ If all 4 cells regress → STOP this refactor; the regression is data-deficit (Path 2 territory), not recipe — refactoring won't help the paper goal

### 🔵 Phase A — Build Core Foundation (with recipe-v2 hooks)
- 🛠️ Create `utils/training_loop.py` with `run_training_loop()` + helpers
- ➕ Add `build_step_record`, `build_training_csv_header`, `make_step_logger` to `utils/training.py`
- 🆕 **Hook contract MUST expose** (driven by Phase-0 outcome):
  - `teacher_mode: {"EMA", "FROZEN"}` + separate `teacher_forward()` (action item 1️⃣ in plan_surgery_wins.md)
  - `head_only_step` mode that skips backbone gradient (action item 2️⃣ LP-FT)
  - `aux_data_iter` slot for pretrain_encoder-replay batches (action item 5️⃣ CLEAR)
  - Scheduled `momentum_schedule(step)` instead of fixed τ (action item 6️⃣)
  - Optional `mask_volume` arg in `compute_jepa_loss` for saliency weighting (action item 7️⃣)
- ➕ Add SPD optimizer wrapper (action item 4️⃣)
- 🚫 NO m09a/m09c body changes yet
- 🚦 **Gate**: ✅ py_compile clean + ✅ ruff F,E9 clean across all 5 files + ✅ unit tests for each new hook surface

### 🟢 Phase B — Migrate m09a (Lower Risk 🥇)
- 🅰️ Refactor m09a's `train()` to call `run_training_loop`
- 📦 Keep all `_m09a_*` hook bodies in `m09a_pretrain.py`
- 🚦 **Gate**:
  - ✅ SANITY pretrain_2X_encoder passes (1 step)
  - ✅ POC pretrain_2X_encoder postrefactor produces **18-file output list** identical to current POC v1 (modulo timestamps)
  - ✅ `probe_top1` within **±0.5 pp** of current POC v1 final (0.4585) → must land in **[0.4535, 0.4635]**

### 🟡 Phase C — Migrate m09c (Higher Risk; Multi-Stage 🅲) + recipe-v2 baked in
- 🅲 Refactor m09c's `train()` to call `run_training_loop` with `stages=cfg["surgery"]["stages"]`
- 📦 Keep `_m09c_*` hook bodies in `m09c_surgery.py`
- 🆕 **Wire recipe-v2 actions into m09c's hook bindings**:
  - 1️⃣ Frozen teacher: `state["teacher_mode"] = "FROZEN"` + load v12 pretrain_encoder encoder as static target
  - 2️⃣ LP-FT: prepend Stage 0 (head-only, 0.5 ep) to `cfg["surgery"]["stages"]`
  - 3️⃣ Surgical layer subset: yaml change — Stage 1 unfreeze 0–3, Stage 2/3 unfreeze 0–7
  - 5️⃣ Replay: `_m09c_on_stage_begin` hooks builds 50/50 factor + raw-video iterator
  - 8️⃣ Unified warmup: scheduler factory takes total budget, not per-stage
- 🚦 **Gate**:
  - ✅ SANITY surgery_3stage_DI_encoder + surgery_noDI_encoder **both** pass
  - ✅ POC surgery_3stage_DI_encoder postrefactor produces **18 files** matching m09a's set (modulo `m09a_*` → `m09c_*` prefix)
  - ✅ Stage 1 `probe_top1` within **±0.5 pp** of current POC v3 final (0.7449) → must land in **[0.6949, 0.7949]** *(this gate is for refactor parity; recipe-v2 IMPROVEMENT is verified in Phase D)*

### 🟤 Phase E (NEW, 2026-05-09) — Move shell-side bootstrap into m09a/m09c (truly thin shells)

> 🎯 **Goal**: kill the redundant shell-level bootstrap of action_labels +
> probe_train_subset.py invocations in `scripts/run_train.sh`. m09a/m09c
> already call `utils.probe_labels.ensure_probe_labels_for_mode(cfg=cfg)` at
> startup (in-process, yaml-driven, recipe-v3-aware). The shell-level
> orchestration is pure tech debt — kept only because `probe_train_subset.py`
> at lines ~93-97 of run_train.sh runs BEFORE m09a/m09c and consumes
> ACTION_LABELS via --subset / --val-subset CLI args.

#### 🛠️ What needs to change (~5 file edits, ~2 hrs Mac)

```
┌─────┬──────────────────────────────────────────────────────────────────┬───────────────────────────────────┐
│  #  │  File                                                              │  Change                          │
├─────┼──────────────────────────────────────────────────────────────────┼───────────────────────────────────┤
│  1  │ src/utils/m09_common.py                                            │ Make --val-subset OPTIONAL when  │
│     │                                                                    │ require_val_data=True; default=None│
│  2  │ src/m09a1_pretrain_encoder.py:471                                          │ When args.subset is None, derive │
│     │                                                                    │ train_keys in-process via         │
│     │                                                                    │ utils.probe_train_subset.        │
│     │                                                                    │ split_subset(action_labels,"train")│
│  3  │ src/m09a1_pretrain_encoder.py:472                                          │ Same for args.val_subset →       │
│     │                                                                    │ val_key_set                      │
│  4  │ src/m09c1_surgery_encoder.py (subset-load site)                            │ Same fallback for both           │
│  5  │ scripts/run_train.sh:62-140                                 │ DELETE the ACTION_LABELS         │
│     │                                                                    │ bootstrap block (~70 LoC)        │
│  6  │ scripts/run_train.sh:88-97                                  │ DELETE the probe_train_subset.py │
│     │                                                                    │ invocations (3 calls × 3 splits) │
│  7  │ scripts/run_train.sh m09a/m09c invocations                  │ DROP --subset / --val-subset CLI │
│     │                                                                    │ args (m09a/m09c derive in-process)│
│  8  │ scripts/run_recipe_v3_sweep.sh                                    │ Already has the redundant pre-   │
│     │                                                                    │ flight comment — no further change│
└─────┴──────────────────────────────────────────────────────────────────┴───────────────────────────────────┘
```

#### 📐 Architecture (post-Phase-E)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Pre-Phase-E (current — tech debt)                                       │
│                                                                            │
│  shell                            m09a/m09c                                │
│  ──┬──                            ──┬──                                  │
│    │ bootstrap action_labels        │                                     │
│    │ (probe_action.py subprocess)   │                                     │
│    │ ↓                              │                                     │
│    │ probe_train_subset.py × 3      │                                     │
│    │ → train/val/test_split.json    │                                     │
│    │ ↓                              │                                     │
│    │ python m09a/m09c               │                                     │
│    │ --subset $TRAIN_SPLIT          ──→ ensure_probe_labels_for_mode      │
│    │ --val-subset $VAL_SPLIT          (idempotent — no-ops because shell  │
│                                       already generated action_labels)    │
│                                                                            │
│  Post-Phase-E (clean)                                                     │
│                                                                            │
│  shell                            m09a/m09c                                │
│  ──┬──                            ──┬──                                  │
│    │ python m09a/m09c (no args)    ──→ ensure_probe_labels_for_mode      │
│                                       (in-process: writes action_labels) │
│                                       ↓                                    │
│                                       split_subset() × 3 in-memory        │
│                                       (no JSON files written)             │
│                                       ↓                                    │
│                                       train(...)                           │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 🚦 Phase E gates

```
┌──────────────────────────────────────────────────────────────────────────┐
│  ✅ utils/m09_common.py: --val-subset becomes type=str default=None       │
│  ✅ m09a + m09c: when args.subset is None → derive in-process via         │
│     split_subset(); same for args.val_subset                              │
│  ✅ scripts/run_train.sh:62-97 deleted (~80 LoC removed)            │
│  ✅ scripts/run_train.sh m09a/m09c invocations: drop --subset       │
│     and --val-subset args                                                  │
│  ✅ Smoke test: shellcheck + bash -n; m09a/m09c --POC e2e                 │
│  ✅ Numerical fidelity gate: top-1 within ±0.5 pp of pre-Phase-E baseline │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 🛡️ Risk assessment

```
┌──────────────────────────────────────────────────────────────────┬────────────┬──────────────────────────────┐
│ Risk                                                               │ Likelihood │ Mitigation                   │
├──────────────────────────────────────────────────────────────────┼────────────┼──────────────────────────────┤
│ --subset arg semantics differ across m09a/m09c                    │ MEDIUM     │ Parallel changes both files  │
│   (m09a uses for output_dir derivation + leakage gate;             │            │ in same commit; exhaustive   │
│   m09c uses for output_dir name only)                              │            │ unit test of args.subset=None│
│ Downstream consumers reading data/eval_10k_*_split.json            │ LOW        │ Verified earlier: only       │
│                                                                    │            │ run_train.sh reads     │
│                                                                    │            │ them (no run_eval.sh   │
│                                                                    │            │ reference). Safe to remove   │
│                                                                    │            │ the disk artifacts.          │
│ utils/m09_common.py changes affect m09b legacy too                 │ LOW        │ m09b in src/legacy/ — frozen │
│                                                                    │            │ per CLAUDE.md; require_val   │
│                                                                    │            │ change is back-compat (None  │
│                                                                    │            │ default) → no breakage       │
│ POC↔FULL parity drift                                              │ NONE       │ ensure_probe_labels_for_mode │
│                                                                    │            │ already drives ALL paths +   │
│                                                                    │            │ numbers from yaml; m09a +    │
│                                                                    │            │ m09c just thread args.subset │
│                                                                    │            │ → in-memory derivation       │
└──────────────────────────────────────────────────────────────────┴────────────┴──────────────────────────────┘
```

#### 📌 Why deferred to Phase E (not done in iter14 today)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Iter14 today: focus is recipe-v3 7-cell drop-one ablation sweep.             │
│  Phase E touches m09_common + m09a + m09c shared CLI surface — invasive +     │
│  needs its own SANITY/POC parity gate (±0.5 pp tolerance) BEFORE recipe-v3.   │
│  Right ordering: ship recipe-v3 sweep FIRST → get research signal → THEN      │
│  Phase E refactor → THEN any FULL training. The structural cleanup follows    │
│  the research result, not vice versa.                                         │
│                                                                                │
│  Pre-existing tech debt (acknowledged):                                        │
│    1. scripts/run_train.sh:62-140 has hardcoded numbers                  │
│       (MIN_CLIPS_BOOTSTRAP=34, MIN_SPLIT_BOOTSTRAP=5) — duplicates the         │
│       cfg["probe_action_labels"][...]  yaml block but the shell can't easily  │
│       read yaml without a yaml_extract subprocess. Phase E removes the entire │
│       block, so the hardcoding goes away naturally.                            │
│    2. probe_train_subset.py invocation in shell creates 3 transient JSON      │
│       files that m09a/m09c then read back. After Phase E: in-memory only.     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

### 🔴 Phase D — Verify FULL Eligibility + recipe-v2 paper-goal POC
- 🧪 Run full POC suite (pretrain_2X_encoder + surgery_3stage_DI_encoder + surgery_noDI_encoder + eval)
- ✅ Confirm every divergence flagged in audits is now structurally absent
- 🆕 **Recipe-v2 paper-goal POC** (re-runs Phase 0's frozen+LP-FT cell, but on the refactored code path):
  - 🎯 Acceptance: surgery POC trio top-1 reaches **≥ 0.808** (the v12 pretrain_encoder anchor) by stage 1 → unblocks FULL surgery (action items 🔟 → 1️⃣3️⃣ in plan_surgery_wins.md)
  - ⛔ If still regresses despite Phase A hooks being correct → fall back to Path 2 (relax m10 thresholds, action item 1️⃣2️⃣)
- 🚦 **Gate**:
  - ✅ All 4 POC commands pass
  - ✅ `diff <(ls outputs/poc/m09a_pretrain_2X/) <(ls outputs/poc/m09c_surgery_3stage_DI/ | sed 's/m09c_/m09a_/g')` produces **EMPTY output**
  - ✅ Recipe-v2 m09c POC trio top-1 ≥ 0.808 (paper-goal feasibility gate)

---

## 🧪 Verification Commands

> 🖥️ Run from project root after each phase

### 🟢 Phase B verify (m09a only)
```bash
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_2X_encoder --SANITY 2>&1 | tee logs/iter14_sanity_pretrain_2X_postrefactor.log
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_2X_encoder --POC 2>&1 | tee logs/iter14_poc_pretrain_2X_postrefactor.log
# 👀 Compare new vs old POC v1: file list, training_summary.json key set, final probe_top1 within ±0.5 pp
```

### 🟡 Phase C verify (m09c surgery)
```bash
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_encoder --SANITY 2>&1 | tee logs/iter14_sanity_surgery_3stage_DI_postrefactor.log
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_encoder      --SANITY 2>&1 | tee logs/iter14_sanity_surgery_noDI_postrefactor.log
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_encoder --POC    2>&1 | tee logs/iter14_poc_surgery_3stage_DI_postrefactor.log
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_encoder      --POC    2>&1 | tee logs/iter14_poc_surgery_noDI_postrefactor.log
```

### 🔴 Phase D parity check
```bash
# 🎯 File lists MUST be identical modulo prefix
diff \
  <(ls outputs/poc/m09a_pretrain_2X/        | sed 's/m09a_//g' | sort) \
  <(ls outputs/poc/m09c_surgery_3stage_DI/  | sed 's/m09c_//g' | sort)
# ✅ Expected: only algo-diffs (val_split.json gone; drift_loss should NOT appear in either)

# 🧪 Phase D eval pipeline
CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --poc 2>&1 | tee logs/iter14_poc_eval_postrefactor.log
```

---

## ⚠️ Risk Mitigation

| 🎲 Risk | 📊 Likelihood | 🛡️ Mitigation |
|---|---|---|
| 💥 Refactor breaks m09a (gold standard currently working) | 🟡 MEDIUM | 🔵 Phase B happens **FIRST** in isolation; SANITY before POC; ↩️ ROLLBACK = `git revert` if SANITY fails. Keep current m09a behavior bit-exact (no LR/seed changes) |
| 🪝 Hook contract too narrow → can't express all m09c logic | 🟡 MEDIUM | 📋 Audit pre-identified all 30 lifecycle events; consolidated to 8 essentials. ➕ If a hook is missing, ADD it before merging Phase B |
| 📊 Unintended m09a metric drift (final probe_top1 ≠ 0.4585) | 🟢 LOW-MED | 🔒 **Locked tolerance: ±0.5 pp** (per user). If postrefactor probe_top1 < 0.4535 or > 0.4635 → **FATAL**, investigate before Phase C. Also verify val_jepa final stays within similar band (current 0.4716) |
| 📊 Unintended m09c metric drift (final probe_top1 ≠ 0.7449 best) | 🟢 LOW-MED | 🔒 Same ±0.5 pp tolerance; m09c POC v3 best probe_top1 was 0.7449 at Stage 1 step 1. Post-refactor must produce **0.6949–0.7949** |
| 📏 `utils/training_loop.py` grows too big (>800 LoC) | 🟢 LOW | ✂️ Cap at 500. If exceeds, split helpers (`_run_step_inner`, `_run_probe_cycle`) into separate utils files |
| 🧊 `state` dict becomes a god-object | 🟡 MEDIUM | 📜 Document expected keys in run_training_loop docstring; 💥 FAIL LOUD on missing keys (no `.get(key, default)` per CLAUDE.md) |
| ❄️ Cooldown phase migration (m09a-only) | 🟢 LOW | ℹ️ Audit found cooldown is currently a partial implementation (only schedules LR, doesn't run steps). Refactor preserves status; tracked as deferred fix |
| 🤗 HF push parity | ⚪ NONE | ✅ Audit confirmed neither module performs HF push; downstream m05 re-embed handles this |

---

## 💾 Memory Updates After Merge

📌 Add to `~/.claude/projects/-workspace-factorjepa/memory/MEMORY.md`:

> 📝 **New entry**: `feedback_training_loop_dispatcher.md` —
> *"When extending training behavior (new step_record key, new probe metric, new plot, new ckpt cadence), modify `utils/training_loop.py` or its hook list — **NEVER edit m09a or m09c's `train()` body alone**. Module hooks should be thin (≤30 LoC each) — anything >30 LoC is a sign that logic should move into utils."*

---

## 📦 Out of Scope (Deferred)

> ⏭️ Not in this iter — tracked for future work

| ❌ Item | 📐 LoC | 🤔 Rationale |
|---|---|---|
| 🏗️ `build_model_and_heads` factory | ~200 | CLAUDE.md rule 49 explicitly justifies per-module isolation for model construction. Cost > benefit this session |
| 📝 CSV append mode unification | ~10 | Resume not used at POC/FULL scale; defer until resume is needed |
| 🗄️ m09b ExPLoRA migration | ~? | Legacy code, not on active path. 4th phase if/when m09b returns from legacy |
| ❄️ Cooldown full implementation | ~50 | Audit found m09a's cooldown is a partial stub (schedules but doesn't loop). Fixing requires producer restart logic. Tracked separately |

---

## 🎯 Success Criteria Summary

After Phase D, this should hold:

1. ✅ `diff outputs/poc/m09a_*/  outputs/poc/m09c_*/` (modulo prefix) → **EMPTY**
2. ✅ Adding a new `step_record` key requires editing **1 file** (`utils/training_loop.py`), not 2
3. ✅ Adding a new plot requires editing **1 file** (`utils/training_loop.py` hook), not 2
4. ✅ m09a probe_top1 ∈ [0.4535, 0.4635] (within ±0.5 pp of 0.4585)
5. ✅ m09c probe_top1 ∈ [0.6949, 0.7949] (within ±0.5 pp of 0.7449)
6. ✅ m09a LoC: 1593 → ~993 (-37%)
7. ✅ m09c LoC: 1717 → ~1267 (-26%)
8. ✅ utils/training_loop.py: 0 → ~450 LoC (new, technique-agnostic)
9. ✅ Net codebase: 5847 → ~5300 LoC (-520, **−9%**)
10. ✅ Structural bug class **eliminated** — schema/import/plot drift cannot recur

---

> 🎬 **Ready to execute when user says "go".** ▶️
