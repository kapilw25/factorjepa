---
name: Known bugs and their fixes (iter13)
description: Bug-class catalog for FactorJEPA iter13 — symptom, root cause, fix location, prevention guard
type: project
---

# Bug log — iter13 fixes (2026-05-03)

> Each entry follows: **Symptom** → **Root cause** → **Fix location** → **Prevention guard** (preflight B-rule or REPL).

## Bug A — `m09a_ckpt_best.pt` missing predictor key

**Symptom**: Stage 8 future_mse FATAL: `KeyError 'predictor'` when loading m09a's best checkpoint.

**Root cause**: `m09a_pretrain.py:1098` saved `_best.pt` with `full=False`, which writes only `{"student", "step", "best_metric", "is_full"}` — no predictor / teacher / optim. `probe_future_mse._load_predictor_2_1` requires the `predictor` key.

**Fix**: Flip `full=False` → `full=True`. Best.pt now ~15 GB (was ~7 GB) but symmetric with surgery_base convention. **Location**: `src/m09a_pretrain.py:1098`.

**Prevention**: When introducing any "best ckpt" save, decide: is this a downstream-eval artifact (full=True needed for predictor-bearing eval paths) or a resume-only anchor (full=True for optimizer state)? Either way, FULL is almost always the right answer for V-JEPA-style trainers.

---

## Bug B — pretrain SANITY exits with 0 successful training steps (silent Meta-weight export)

**Symptom**: `outputs/<mode>/m09a_pretrain/loss_log.jsonl` is 0 bytes; `student_encoder.pt` is the un-modified Meta weights re-exported. Downstream eval shows `vjepa_2_1_pretrain` numerically identical to `vjepa_2_1_frozen`.

**Root cause**: m09a's OOM handler used `optimizer.zero_grad(); continue` to advance to the next for-loop iteration. With SANITY `total_steps=1`, that single OOM exits the for-loop with 0 successful steps, but training proceeds to `export_student_for_eval` and writes the un-trained weights as if training had succeeded.

**Fix**: Two-part —
1. Replace `continue` with a `while not step_succeeded:` retry-same-macro loop (mirrors m09c #55 pattern). On OOM, sub-batch shrinks via `train_sizer.on_oom()` (called inside `_train_step_grad_accum`) and the macro retries at the new size. Fail-hard if sub-batch hits min and still OOMs.
2. Post-train guard: if `step + 1 == start_step` (no successful step), raise `RuntimeError("M09A FAILED: 0 successful training steps...")` BEFORE `export_student_for_eval` runs.

**Location**: `src/m09a_pretrain.py:836-866` (retry loop) + `:1166-1177` (post-train guard).

**Prevention**: Per CLAUDE.md "Silent failures = garbage metrics". Any path that would emit a downstream artifact MUST verify the artifact is meaningful before writing. The guard pattern is also in `m09c_surgery.py:1295` (same incident class).

---

## Bug R8 — m09c writes `student_best.pt`, eval expects `m09c_ckpt_best.pt`

**Symptom**: Stage 8 future_mse FATAL on missing `m09c_ckpt_best.pt` for surgery variants. m09c writes `student_best.pt` (encoder-only) which gets promoted to `student_encoder.pt`; the only full ckpts are stage rollbacks (`m09c_ckpt_stage<N>.pt`) that get nuked by `cleanup_stage_checkpoints(keep_n=0)` at line 1424.

**Root cause**: Naming mismatch + cleanup-glob deletion. The original orchestration plan assumed `m09c_ckpt_best.pt` would be a full ckpt; m09c actually writes `student_best.pt` (encoder-only) for the best-Prec@K tracker, and full ckpts only exist as stage rollbacks that get cleaned up.

**Fix**: After best-promotion (line 1370) and BEFORE `cleanup_stage_checkpoints`, add an explicit `save_training_checkpoint(_best.pt, ..., full=True, uw=uw_module)` call. The new file's name (`m09c_ckpt_best.pt`) doesn't match the cleanup glob (`{prefix}_stage*.pt`), so it survives.

**Location**: `src/m09c_surgery.py:1372-1385`.

**Prevention**: When wiring eval-side dependency on a training artifact, run a glob-safety REPL test that confirms the artifact survives the cleanup pattern. Example:
```python
import fnmatch; assert not fnmatch.fnmatch("m09c_ckpt_best.pt", "m09c_ckpt_stage*.pt")
```

---

## Bug OOM-frag — orphan tensors accumulate across OOM-retries

**Symptom**: Even with the Bug B fix (while-loop retry), JEPA OOMs successively at sub-batch 16 → 8 → 4 → 2 → 1 — all the way to min — even though sub-batch=1 should easily fit on the same hardware that fit sub-batch=8 in earlier runs.

**Root cause**: The OOM-retry handler called `optimizer.zero_grad()` to discard partial grads but did NOT call `gc.collect() + torch.cuda.empty_cache()`. Failed forward passes left orphan tensors held by autograd's interrupted graph; each successive sub-batch shrink started with LESS free VRAM than the previous → eventual exhaustion at sub-batch=1 even when the model fit at sub-batch=8 on a fresh allocator.

**Fix**: Add `gc.collect() + torch.cuda.empty_cache()` after `optimizer.zero_grad()` in the OOM handler, BEFORE the sub-batch-min check. Mirrors the existing `torch.cuda.empty_cache()` in m09a's multi-task OOM handler (line 853) which already had this fix.

**Location**: `src/m09a_pretrain.py:849-857` + `src/m09c_surgery.py:1170-1179`.

**Prevention**: Any try/except around CUDA forward+backward should release fragmented memory in the except path before continuing. `empty_cache()` is generally discouraged on the success path (defeats allocator reuse) but is correct on the failure path.

---

## Bug eval-ckpt-schema — `load_vjepa_2_1_frozen` doesn't recognize iter13 ckpt schemas

**Symptom**: Stage 2 features for `vjepa_2_1_pretrain` reports `Loaded 0/588 params (missing=588, unexpected=3)` then FATAL: `only 0/588 V-JEPA params loaded — key mismatch`.

**Root cause**: `utils/frozen_features.py:86` only walked Meta's `target_encoder` / `encoder` keys then fell through to raw dict. iter13's `student_encoder.pt` (written by `export_student_for_eval`) uses key `student_state_dict`; iter13's `m09a_ckpt_best.pt` (written by `save_training_checkpoint(full=True)`) uses key `student`. Neither was recognized → loader treated the wrapper dict (`{"student_state_dict", "model_id", "type"}`) as the state itself → 3 unexpected, 588 missing.

**Fix**: Add `resolve_encoder_state_dict(ckpt)` helper that walks 4 keys in priority order (`target_encoder` → `encoder` → `student_state_dict` → `student`) before falling through to raw. Used by both `load_vjepa_2_1_frozen` and `probe_future_mse._load_encoder_2_1_hierarchical` (which had the same 2-key fallback).

**Location**: `src/utils/frozen_features.py:74-95` + `src/probe_future_mse.py:53-56` (import) + `:121` (use).

**Prevention**: When introducing a new ckpt write convention (key name choice in `torch.save`), audit all eval-side loaders that read that file family AT THE SAME TIME, not as a separate session. m05_vjepa_embed.py:597-598 already had a `student_state_dict` branch (precedent), so this was a regression of an already-known pattern.

---

## Bug Stage 8 FATAL — eval-side hard-stop on missing predictor ckpt

**Symptom**: Stage 8 fails with `FATAL: predictor-bearing ckpt missing for vjepa_2_1_pretrain: outputs/sanity/m09a_pretrain/m09a_ckpt_best.pt`. `set -e + trap ERR` cascade kills Stages 9 + 10 — no plots produced.

**Root cause**: `run_probe_eval.sh:495` had a `[ -e "$CKPT" ] || { echo FATAL; exit 3; }` check inside the Stage 8 loop. When the predictor ckpt was missing for one V-JEPA variant (e.g. pretrain not yet trained), the entire pipeline aborted instead of skipping that one variant.

**Fix**: Two-layer fix —
1. **Pre-flight**: Build a `STAGE8_ENCODERS` subset list at startup (lines 289-326) — for each V-JEPA variant in `$ENCODERS`, check if `encoder_predictor_ckpt_for(ENC)` exists; if not, drop from `STAGE8_ENCODERS` with a loud WARN (mirrors the existing P2/P3 trainer-output pre-flight at lines 258-281).
2. **In-loop defense-in-depth**: Stage 8 loop iterates `STAGE8_ENCODERS` (not `ENCODERS`); per-iter check is WARN+continue (not FATAL). Defends against `SKIP_STAGES` bypassing the pre-flight.

**Location**: `scripts/run_probe_eval.sh:289-326` (pre-flight) + `:484-512` (loop).

**Prevention**: For any orchestrator stage that has a per-encoder dependency, the pre-flight should validate AT STARTUP (before any compute) — both to fail-fast on user error AND to give the user the actionable hint (e.g. "train via `./scripts/run_probe_train.sh pretrain --$MODE`"). Stage 9 (paired_per_variant) handles partial results natively by reading whatever `per_clip_mse.npy` files exist on disk → no further fix needed downstream.

---

## Bug plot-NaN — `_bar_with_ci` crashes on degenerate BCa CI

**Symptom**: Stage 10 `probe_plot.py` FATAL: `ValueError: Axis limits cannot be NaN or Inf` at `ax.set_ylim(...)`. The first 2 plots (action_loss, action_acc) save successfully; the 3rd (encoder_comparison) dies.

**Root cause**: At small N (SANITY n_test=22), V-JEPA hits perfect 100% accuracy → BCa formula divides by zero variance → `ci_half = NaN`. NaN propagates through `(real_v - real_e).min()` into `ax.set_ylim(NaN, NaN)`, which matplotlib explicitly rejects (other matplotlib paths like `bar(yerr=NaN)` silently no-op the error caps and don't crash).

**Fix**: Use `np.nan_to_num(real_e, nan=0.0)` before computing `lo`/`hi`/`pad` for ylim. Bars still draw (matplotlib ignores NaN in yerr); error caps just don't appear; ylim sits at `[data_min, data_max]` ± pad. Same NaN-substitution for the value-label placement (`v + er + pad*0.1` was placing labels at NaN → silently dropped).

**Location**: `src/probe_plot.py:196-218`.

**Prevention**: Whenever a downstream consumer ingests bootstrap CIs that could be NaN on degenerate inputs (perfect predictions / single-class / zero-variance), defend explicitly. The BCa formula's failure mode is well-known: `scipy.stats._resampling.py:155 RuntimeWarning: invalid value encountered in scalar divide` is the canary in the log.

---

## Pre-iter13 historical bugs (still load-bearing context)

These predate the iter13 session but inform current architecture:

- **#52** (`max_epochs[mode_key][mode_key]` double subscript) — guard B31 in preflight
- **#53** (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` missing in m09*) — guard B32
- **#54** (`jepa_val`/`masked_val`/`context_val` not pre-init in m09c) — guard B33
- **#55** (m09c retry-loop + 0-step raise — Bug B above mirrors this) — guard B34
- **#57** (mode-gated yaml dict not flattened by reader → `TypeError: 'dict' is not subscriptable`) — guard B59
- **#62** (YAML dead-cap `n_clips` / `clip_limit` silently downscopes `--subset`) — guard B55
- **#69** (data subset overlap with val_1k.json — 41% leak silently invalidated paired-eval) — guard B52
- **#74** (shell `rm -rf outputs/full/m05*` nukes hidden `.m05_checkpoint*.npz`) — guard B53
- **#79** (m09c BWT plateau buffer reset across stage transitions)
- **#80** (long-lived `tmp_dir` ENOENT after ~2M write/unlink — covered by B56)
- **#81** (latest pre-iter13 entry; see `errors_N_fixes.md` for full catalog)

Full catalog: `iter/iter13_motion_probe_eval/errors_N_fixes.md` (~125 KB, 81 entries iter8 → iter12).

## Preflight skill (`/preflight @<file>`)

Path: `.claude/skills/preflight/SKILL.md`. Run on any modified `.py` / `.yaml` / `.sh` to catch regressions of past bug classes. B1-B65 cover all the entries above plus generic fail-hard / wandb / cache-policy / config-schema patterns. Auto-runs partially via the `post-edit-lint.sh` hook (3-check: `py_compile` + `ast.parse` + `ruff check --select F,E9`).
