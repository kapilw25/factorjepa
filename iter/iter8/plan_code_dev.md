# 🏗️ Code Dev Plan — Split `m09_pretrain.py` into m09a/b/c + utils/training.py

> **Status legend**: 📋 planned · 🔄 WIP · ⏸️ blocked · ✅ done · ⚠️ needs review · 🚨 danger · 🎯 goal · 🔍 verify

---

## 🎯 Why split now (Context)

`src/m09_pretrain.py` is **2092 lines** and houses **3 distinct training techniques** entangled via `--explora` / `--surgery` argparse flags:

| Technique | Status | Function | Lines |
|---|---|---|---:|
| **Vanilla Ch10 pretrain** | proven (Ch10 catastrophic-forgetting baseline) | `train()` | ~700 |
| **ExPLoRA (LoRA + 2 unfrozen blocks)** | 🆕 unproven (Step D) | `train()` w/ `--explora` flag | ~50 LoRA-specific |
| **Ch11 surgery (factor datasets)** | 🆕 unproven (Step E) | `train_surgery()` | ~270 |

🚨 **Risk in current monolith**: any iteration on one technique can silently regress the other two. Same failure mode that bit Ch10 (`iter/iter7_training_full/runbook.md`) where intermixed logging masked divergence.

🎯 **Outcome**: physical separation so iteration on m09b can't break a matured m09c (and vice versa). Future technique additions (BitFit / IA³ / adapters per `iter/iter8/FactorJEPA-Alternatives*.pdf`) become NEW files, never edits to validated ones.

---

## ✅ User-confirmed decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Full isolation** — each m09a/b/c has its OWN complete training loop body (~500 LOC each) | Max safety. No shared `run_loop()` function. ~500 LOC duplication accepted. |
| 2 | **Output dir migration** — `outputs/.../m09_pretrain/{ablation,lambda*,explora,surgery}/` → `outputs/.../m09a_pretrain/`, `m09b_explora/`, `m09c_surgery/` | Code module ↔ output dir parity. Requires `scripts/run_embed.sh:90` refactor + 1-time migration script. |
| 3 | **Numerical bit-equivalence validation** between phases (tolerance 1e-6) | Catches accidental behavior change introduced by extraction. Loss values must match Phase 0 baseline. |

---

## 🏛️ Final architecture

```
src/
├── m09a_pretrain.py       ~750 lines  📋 vanilla Ch10 (drift control, lambda sweep)
├── m09b_explora.py        ~700 lines  📋 LoRA on 46/48 blocks + unfreeze first 2, no drift
├── m09c_surgery.py        ~900 lines  📋 3-stage progressive unfreeze + factor datasets
└── utils/
    └── training.py        ~700 lines  📋 PRIMITIVES ONLY (no full training loops here)
```

Each `m09[abc]` module owns: `main()`, argparse, technique-specific `build_<X>_model()`, the **FULL** training-loop body, CSV/wandb logging, NaN/Inf guards, checkpoint orchestration.

---

## 📦 What goes in `utils/training.py`

🔒 **CRITICAL CONTRACT**: every function in `utils/training.py` has **ZERO** `if args.explora`, `if args.surgery`, `if cfg["technique"]` branches. Mode-specific behavior is configured via explicit parameters (`init_params=None`, `drift_cfg=None`, `explora_enabled=False`).

| 📋 Function | Source line in m09 | Purpose | Used by | Status |
|---|---:|---|---|---|
| `producer_thread`, `_decode_batch` | 197-301 | Background WebDataset/TAR decode | m09a, m09b | 📋 |
| `compute_jepa_loss` | 529-567 | Dense masked + context loss | all 3 | 📋 |
| `compute_drift_loss` | 570-578 | L2 param regularizer | m09a only (pure fn → utils) | 📋 |
| `_train_step_grad_accum` | 581-666 | Adaptive grad-accum forward+backward | all 3 (already technique-agnostic per #48) | 📋 |
| `update_teacher_ema` | 667-682 | EMA teacher momentum | all 3 | 📋 |
| `build_optimizer`, `build_scheduler`, `update_weight_decay` | 684-743 | Optimizer/LR plumbing | all 3 | 📋 |
| `run_validation` | 745-819 | Held-out val pass | m09a, m09b (m09c has no held-out val) | 📋 |
| `save_training_checkpoint`, `load_training_checkpoint`, `cleanup_old_checkpoints` | 821-866 | Checkpoint I/O | all 3 | 📋 |
| `export_student_for_eval` | 869-888 | Export student .pt for m05 (handles LoRA merge via `explora_enabled` param) | all 3 | 📋 |
| `set_trainable_prefix` | 890-903 | Freeze all + unfreeze first N layers | m09c (also reusable by future techniques) | 📋 |
| `augment_clip_consistent` | 167-194 | Spatial augmentation (consistent across frames) | all 3 | 📋 |
| `build_mask_generators` | 504-527 | Mask config → generators | all 3 | 📋 |
| `load_config`, `merge_config_with_args`, `load_val_subset` | 96-164 | Config I/O | all 3 | 📋 |
| `FactorSampler` (class), `build_factor_index`, `load_factor_clip` | 906-997 | Factor dataset (D_L/D_A/D_I) | m09c (reusable for future ablations) | 📋 |
| `_resolve_train_sizer` (🆕 helper) | — | Reads yaml `training_initial_bs` + `gpu_memory_target` + module max BS → returns `AdaptiveBatchSizer` | all 3 | 📋 |

---

## 🚨 Danger zones (from Phase 1 audit) and mitigations

| # | 🚨 Danger | Where in m09 | 🛡️ Fix in split | Status |
|---|---|---:|---|---|
| 1 | `main._nan_strikes` global counter (function attribute) | line 1727 | Replace with local `nan_strikes = 0` in each m09[abc] training loop | 📋 |
| 2 | `init_params` differs per technique (vanilla = all params, ExPLoRA = filter LoRA, surgery = not used) | lines 444, 487 | Each m09[abc] computes its own `init_params` after model build; passes (or omits) to `_train_step_grad_accum` | 📋 |
| 3 | `drift_cfg` only used by m09a | lines 1693, 652-656 | m09b/m09c never construct/pass `drift_cfg`; helper sig has default `drift_cfg=None` | 📋 |
| 4 | `explora_enabled` flag must thread to `export_student_for_eval()` for LoRA merge | lines 875-877, 1906 | m09b passes `explora_enabled=True`; m09a + m09c pass `False` (default) | 📋 |
| 5 | `producer_thread` only used by m09a/m09b; m09c uses FactorSampler directly | line 1570 | m09c imports `FactorSampler` from utils, never instantiates a producer thread | 📋 |
| 6 | `run_validation` only called by m09a/m09b; m09c has no held-out val | line 1805 | m09c never imports `run_validation` | 📋 |
| 7 | Per-step loss-component logging schemas differ (m09a logs `loss_drift`, m09c doesn't) | lines 1747-1760, 1208-1222 | Each module's own `_log_step` function; no shared logger schema | 📋 |
| 8 | Ablation lambda sweep (subprocess fork) is m09a-only | lines 2032-2053 | `select_ablation_winner()` + ablation-sweep code stays in m09a, not extracted | 📋 |

---

## 🗂️ Output directory migration

| Before | After | Status |
|---|---|---|
| `outputs/{sanity,poc,full}/m09_pretrain/lambda*/` | `outputs/.../m09a_pretrain/lambda*/` | 📋 |
| `outputs/{sanity,poc,full}/m09_pretrain/ablation/` | `outputs/.../m09a_pretrain/ablation/` | 📋 |
| `outputs/{sanity,poc,full}/m09_pretrain/explora/` | `outputs/.../m09b_explora/` | 📋 |
| `outputs/{sanity,poc,full}/m09_pretrain/surgery/` | `outputs/.../m09c_surgery/` | 📋 |

🆕 **Migration script** (`scripts/migrate_m09_outputs.sh`, throwaway, idempotent):
- For each `outputs/{sanity,poc,full}/m09_pretrain/{lambda*,ablation}/` → `mv` to `outputs/.../m09a_pretrain/`
- For `m09_pretrain/explora/` → `m09b_explora/`
- For `m09_pretrain/surgery/` → `m09c_surgery/`

🔧 **`scripts/run_embed.sh:90`** — currently `for model_dir in "$OUT_DIR"/m09_pretrain/lambda*/ "$OUT_DIR"/m09_lambda*/`. After split:
```bash
for model_dir in "$OUT_DIR"/m09a_pretrain/lambda*/ "$OUT_DIR"/m09a_pretrain/ablation/lambda*/ \
                 "$OUT_DIR"/m09b_explora/ "$OUT_DIR"/m09c_surgery/; do
```

---

## 📁 Files inventory (16 changed + 7 new + 1 deleted)

### 🆕 NEW (7)

| File | Status | Purpose |
|---|---|---|
| `src/m09a_pretrain.py` | 📋 | Vanilla Ch10 pretrain |
| `src/m09b_explora.py` | 📋 | ExPLoRA training |
| `src/m09c_surgery.py` | 📋 | Ch11 surgery |
| `src/utils/training.py` | 📋 | Shared primitives |
| `scripts/migrate_m09_outputs.sh` | 📋 | Throwaway output-dir migration |
| `tests/m09_baseline.json` | 📋 | Phase 0 5-step loss reference |
| `tests/diff_loss_baseline.py` | 📋 | Bit-equivalence checker |

### 🗑️ DELETED (1)

| File | Status | Reason |
|---|---|---|
| `src/m09_pretrain.py` | 📋 | Content split across new files |

### 🔧 EDITED (16)

| # | File | Change | Status |
|---:|---|---|---|
| 1 | `src/utils/profile_vram.py:62` | `from m09_pretrain import load_config` → `from utils.training import load_config` | 📋 |
| 2 | `scripts/train_pretrain.sh:275, 285` | `src/m09_pretrain.py` → `src/m09a_pretrain.py` | 📋 |
| 3 | `scripts/train_explora.sh:193` | `src/m09_pretrain.py --explora` → `src/m09b_explora.py` (drop `--explora` flag) | 📋 |
| 4 | `scripts/train_surgery.sh:197` | `src/m09_pretrain.py --surgery --factor-dir ...` → `src/m09c_surgery.py --factor-dir ...` | 📋 |
| 5 | `scripts/run_embed.sh:90` | Glob refactor (see "Output directory migration" above) | 📋 |
| 6 | `configs/train/ch10_pretrain.yaml:10` | Docstring command update | 📋 |
| 7 | `configs/train/explora.yaml:14` | Docstring update (drop `--explora`) | 📋 |
| 8 | `configs/train/ch11_surgery.yaml:19` | Docstring update (drop `--surgery`) | 📋 |
| 9 | `iter/iter8/runbook.md:108-130` | Steps D + E command examples → new module names | 📋 |
| 10 | `iter/iter8/plan_TODO.md:178` | Mark "Split m09" item as ✅ done | 📋 |
| 11 | `iter/iter7_training_full/plan_training.md:421, 507` | Architecture mermaid `m09_pretrain.py` → 3 nodes | 📋 |
| 12 | `iter/iter7_training_full/configs/pretrain/vitg16_indian.yaml:4-8` | Docstring command examples | 📋 |
| 13 | `src/utils/plots.py:157`, `src/utils/output_guard.py:143`, `src/utils/config.py:763` | Docstring path examples (`m09_pretrain` → `m09a_pretrain`) | 📋 |
| 14 | `.claude/skills/cost-check/SKILL.md:38`, `.claude/skills/audit-against-gold/SKILL.md:69` | Docstring updates | 📋 |
| 15 | `.claude/hooks/protect-checkpoints.sh:21,24` | Extend pattern to also match `m09a_*`/`m09b_*`/`m09c_*` | 📋 |
| 16 | `src/CLAUDE.md` | Update PROJECT STRUCTURE + GPU PIPELINE CHECKLIST + REFERENCE; add rule "utils/training.py functions MUST be technique-agnostic" | 📋 |

---

## 🚦 Phased execution (Phase 0 + 4 phases)

### 📍 Phase 0 — Baseline capture (~10 min) · 📋

**Goal**: capture per-step loss values for each technique on current m09 as bit-equivalence reference.

```bash
# Run each technique for 5 steps in SANITY mode, dump loss values to JSON
for technique in pretrain explora surgery; do
  python -u src/m09_pretrain.py --SANITY \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/<technique>.yaml \
    --max-steps 5 --no-wandb \
    2>&1 | tee logs/baseline_${technique}_5step.log
done
```

🔍 **Verify**: extract `loss_jepa`, `loss_masked`, `loss_context`, `loss_drift` (where applicable) per step → save to `tests/m09_baseline.json`.

---

### 📍 Phase 1 — Extract shared helpers to `utils/training.py` (~3 hours) · 📋

**Goal**: pure copy of 17 primitives into `utils/training.py`. NO behavior change.

| Step | Action | Status |
|---|---|---|
| 1.1 | Create `src/utils/training.py` with the 17 primitives (copied from m09) | 📋 |
| 1.2 | Update m09_pretrain.py to `from utils.training import ...` | 📋 |
| 1.3 | m09_pretrain.py shrinks ~2092 → ~1100 lines (only `train()`, `train_surgery()`, `main()`, `select_ablation_winner()` remain) | 📋 |
| 1.4 | Lint: `py_compile + ruff F821/F841/F811` | 📋 |
| 1.5 | 🔍 **Validation**: re-run 5-step SANITY for each technique. Diff vs Phase 0 baseline. Tolerance 1e-6. | 📋 |

🚨 **If divergence detected** in step 1.5: bisect by reverting individual helpers; most likely cause is a closure capturing wrong scope or shared state.

---

### 📍 Phase 2 — Split into m09a/m09b/m09c (~6 hours) · 📋

| Step | Action | Status |
|---|---|---|
| 2.1 | Create `src/m09a_pretrain.py` (~750 lines): `main()` + `build_pretrain_model()` + full vanilla `train()` loop body + `select_ablation_winner()`. Drops `--explora`/`--surgery` flags. Output paths → `outputs/.../m09a_pretrain/`. Logs `loss_drift` | 📋 |
| 2.2 | Create `src/m09b_explora.py` (~700 lines): `main()` + `build_explora_model()` w/ LoRA injection + full training loop COPIED from `train()` (drift removed: `init_params=None`, `drift_cfg=None`). `export_student_for_eval(student, path, explora_enabled=True)`. Output paths → `outputs/.../m09b_explora/`. No `loss_drift` in logs | 📋 |
| 2.3 | Create `src/m09c_surgery.py` (~900 lines): `main()` + `build_surgery_model()` + full `train_surgery()` body + 3-stage progressive unfreeze + per-stage optimizer rebuild. Uses `FactorSampler` from utils. No `producer_thread`, no `run_validation`. Output paths → `outputs/.../m09c_surgery/` | 📋 |
| 2.4 | 🗑️ Delete `src/m09_pretrain.py` | 📋 |
| 2.5 | Lint: `py_compile + ruff F821/F841/F811` on all 3 new files | 📋 |
| 2.6 | 🔍 **Validation**: re-run 5-step SANITY against Phase 0 baseline (each via NEW module name). Loss values must match 1e-6. | 📋 |

---

### 📍 Phase 3 — Update consumers (scripts, configs, hooks) (~1 hour) · 📋

| Step | Action | Status |
|---|---|---|
| 3.1 | Run `scripts/migrate_m09_outputs.sh` to mv existing outputs to new dir naming | 📋 |
| 3.2 | Update `scripts/train_pretrain.sh:275`, `scripts/train_explora.sh:193`, `scripts/train_surgery.sh:197` to call new module names | 📋 |
| 3.3 | Update `scripts/run_embed.sh:90` to glob new output dirs | 📋 |
| 3.4 | Update `src/utils/profile_vram.py:62` import path | 📋 |
| 3.5 | `bash -n` syntax check on all touched scripts | 📋 |
| 3.6 | 🔍 **Validation**: dry-run `./scripts/train_explora.sh --SANITY 2>&1 | tee logs/dry_run_explora.log`; verify it invokes `src/m09b_explora.py` and produces output under `outputs/sanity/m09b_explora/` | 📋 |

---

### 📍 Phase 4 — Documentation (~1 hour) · 📋

| Step | Action | Status |
|---|---|---|
| 4.1 | `src/CLAUDE.md`: PROJECT STRUCTURE adds m09a/b/c convention; REFERENCE updated; new rule "utils/training.py MUST be technique-agnostic" | 📋 |
| 4.2 | `iter/iter8/plan_TODO.md:178`: mark split as ✅ done | 📋 |
| 4.3 | `iter/iter8/runbook.md` Steps D + E: command examples updated | 📋 |
| 4.4 | `iter/iter8/errors_N_fixes.md`: log as #49 with the 8-item danger-zone catalogue | 📋 |
| 4.5 | `iter/iter7_training_full/plan_training.md`: mermaid diagrams `m09_pretrain.py` node → 3 nodes for m09a/b/c | 📋 |

---

## 🔍 End-to-end verification (after all phases land)

```bash
# 1. Lint all new files
source venv_walkindia/bin/activate
python3 -m py_compile src/m09a_pretrain.py src/m09b_explora.py src/m09c_surgery.py src/utils/training.py
ruff check --select F821,F841,F811 src/m09[abc]_*.py src/utils/training.py

# 2. Verify utils/training.py has NO technique branches
grep -E "args\.(explora|surgery)|cfg\[.technique.\]" src/utils/training.py
# MUST return 0 matches

# 3. SANITY each technique end-to-end
for module_pair in "m09a_pretrain ch10_pretrain" "m09b_explora explora" "m09c_surgery ch11_surgery"; do
  set -- $module_pair
  python -u src/$1.py --SANITY \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/$2.yaml \
    --max-steps 5 --no-wandb \
    2>&1 | tee logs/postsplit_${1}_5step.log
done

# 4. Numerical bit-equivalence vs Phase 0 baseline
python3 tests/diff_loss_baseline.py logs/baseline_pretrain_5step.log logs/postsplit_m09a_pretrain_5step.log
python3 tests/diff_loss_baseline.py logs/baseline_explora_5step.log  logs/postsplit_m09b_explora_5step.log
python3 tests/diff_loss_baseline.py logs/baseline_surgery_5step.log  logs/postsplit_m09c_surgery_5step.log
# Each MUST report: max abs diff < 1e-6

# 5. Verify scripts wire to new modules
bash -n scripts/train_pretrain.sh scripts/train_explora.sh scripts/train_surgery.sh scripts/run_embed.sh
grep -rE "src/m09_pretrain\.py" scripts/   # MUST return 0 matches

# 6. Verify output dir migration succeeded
ls outputs/poc/m09a_pretrain/  outputs/poc/m09b_explora/  outputs/poc/m09c_surgery/  2>/dev/null

# 7. Smoke test profile_vram.py still works after import path change
python3 -c "from utils.training import load_config; print('OK')"
```

---

## ♻️ Existing utilities to reuse (no re-implementation)

| Utility | Path | Used by |
|---|---|---|
| `AdaptiveBatchSizer` (#47) | `src/utils/gpu_batch.py` | All 3 via `_resolve_train_sizer` |
| `get_pipeline_config`, `get_model_config`, `load_merged_config` | `src/utils/config.py` | All 3 |
| `iter_clips_parallel` | `src/utils/data_download.py` | `producer_thread` (in utils/training) |
| `add_wandb_args`, `init_wandb`, `log_metrics`, `finish_wandb` | `src/utils/wandb_utils.py` | All 3 |
| `verify_or_skip` | `src/utils/output_guard.py` | All 3 |
| `save_json_checkpoint`, `load_json_checkpoint` | `src/utils/checkpoint.py` | All 3 |
| `get_vit_by_arch` | `src/utils/vjepa2_imports.py` | All `build_*_model` variants |
| `get_clip_key`, `create_stream`, `decode_video_bytes` | `src/utils/video_io.py` | `producer_thread` |
| `peft.LoraConfig, get_peft_model` | external | `m09b_explora` only |

---

## ⚠️ Risk register

| Risk | Likelihood | Mitigation | Status |
|---|---|---|---|
| Phase 1 numerical divergence (helper extraction subtly changes behavior) | Medium | Phase 0 baseline; bit-equivalence at 1e-6; bisect on failure | 📋 |
| Phase 2 introduces a per-module bug (e.g., m09b drift not zeroed → trains with stale gradient) | Medium | Same baseline check; lint guard `grep "drift" src/m09b_explora.py` should return 0 | 📋 |
| Output dir migration breaks existing checkpoint resume | Low | Migration script idempotent + reversible (`mv` not `rm`) | 📋 |
| `scripts/run_embed.sh` glob refactor misses an output | Low | Test by listing all expected paths after dry-run | 📋 |
| Documentation drift (CLAUDE.md not updated to reflect new convention) | High if rushed | Phase 4 explicitly enforces; final grep `grep -rE "src/m09_pretrain\.py" --include="*.md"` MUST be 0 | 📋 |

---

## ⏱️ Time estimate

| Phase | Time | Status |
|---|---:|---|
| Phase 0 (baseline capture) | 10 min | 📋 |
| Phase 1 (extract to utils/training.py) | 3 h | 📋 |
| Phase 2 (split into m09a/b/c) | 6 h | 📋 |
| Phase 3 (scripts + configs migration) | 1 h | 📋 |
| Phase 4 (docs) | 1 h | 📋 |
| **Total** | **~11 h** | 📋 |

User explicitly: *"Time spent on code development is not constraint. HIGH ACCURACY and HIGH THROUGHPUT is the top priority."* 11h refactor pays back over 5+ technique iterations on Steps D + E.

---

## 📜 Live progress log

> **Status icons**: 📋 planned · 🔄 WIP · ⏸️ blocked · ✅ done · ⚠️ needs review · 🚨 risk hit

| Time | Phase | Step | Action | Status | Notes |
|---|---|---|---|---|---|
| 2026-04-15 | — | — | Plan written + decisions captured | ✅ | Full isolation, output migration, bit-equiv validation |
| 2026-04-15 | 0 | — | Skip Phase 0 baseline capture | ⚠️ | `--max-steps` flag doesn't exist; m09 hasn't been validated under transformers 5.5.4 anyway. Switched to: Phase 1 pure-mechanical extraction verified by diff + smoke test; Phase 2 split verified same way |
| 2026-04-15 | 1 | 1.1-1.5 | Extract 20 symbols to utils/training.py | ✅ | m09 2164→1520 lines; utils/training.py 715 lines; lint clean (F821/F841/F811); 0 `args.explora`/`args.surgery` in utils/training.py; backup at /tmp/m09_pretrain_backup.py. 6 defs remain in m09 (merge_config_with_args, build_model, train_surgery, train, main, select_ablation_winner) |
| 2026-04-15 | 2 | 2.1-2.3 | Create m09a / m09b / m09c in parallel | 🔄 | 3 agents launched |
| _next_ | 2 | 2.4 | Delete src/m09_pretrain.py | 📋 | after 2.1-2.3 complete |
| _next_ | 3 | — | Update scripts + configs + utils imports | 📋 | |
| _next_ | 4 | — | Docs | 📋 | |

---

## 📝 Notes / discoveries (append as we go)

_(to be populated during execution)_
