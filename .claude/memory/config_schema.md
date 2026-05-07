---
name: Config schema — per-mode flatten, opt-in pattern, ckpt schemas
description: How configs work in FactorJEPA — per-mode dicts, inheritance, ckpt-key dispatch
type: project
---

# Config schema — load-bearing patterns

## Inheritance chain

```
configs/model/vjepa2_1.yaml             ← architecture (embed_dim=1664, depth=48, etc.)
configs/train/base_optimization.yaml    ← shared: data, masking, augmentation, AdamW, EMA, mp, probe, multi_task_probe
    └─ extends: configs/train/probe_pretrain.yaml         ← P2 (m09a continual SSL)
    └─ extends: configs/train/surgery_base.yaml           ← shared surgery scaffolding
        └─ extends: configs/train/surgery_3stage_DI.yaml  ← P3a (D_L → D_A → D_I)
        └─ extends: configs/train/surgery_2stage_noDI.yaml  ← P3b (D_L → D_A only)
configs/train/legacy2/{ch10_pretrain,explora}.yaml      ← retired, kept for reference
configs/pipeline.yaml                                    ← clip limits, streaming, GPU defaults, encoder registry
configs/tag_taxonomy.json                                ← 15-dim taxonomy schema (multi-task labels source)
```

Loaded via `utils.config.load_merged_config(model_yaml, train_yaml)` which deep-merges train_yaml over model_yaml + base_optimization.yaml.

## The per-mode dict pattern (THE convention)

Many keys ship as `{sanity: X, poc: Y, full: Z}` dicts and are **flattened** to a scalar by `merge_config_with_args(cfg, args)` based on which mode flag (`--SANITY` / `--POC` / `--FULL`) is set:

```yaml
# In YAML
optimization:
  max_epochs: {sanity: 1, poc: 5, full: 50}
  use_8bit_optim: {sanity: true, poc: false, full: false}
  gradient_checkpointing: {sanity: true, poc: false, full: false}
  paged_optim: {sanity: true, poc: false, full: false}
probe:
  enabled: {sanity: false, poc: true, full: true}
  best_ckpt_enabled: {sanity: false, poc: true, full: true}
  ...
multi_task_probe:
  enabled: {sanity: true, poc: true, full: true}    # iter13 opt-in (per-config)
factor_streaming:
  sanity: false
  poc: true
  full: true
  num_workers: {sanity: 2, poc: 4, full: 8}
```

```python
# Python flatten in m09a/m09c merge_config_with_args
mode_key = "sanity" if args.SANITY else ("poc" if args.POC else "full")
cfg["optimization"]["max_epochs"] = cfg["optimization"]["max_epochs"][mode_key]
for k in ("use_8bit_optim", "gradient_checkpointing", "paged_optim"):
    if k in cfg["optimization"] and isinstance(cfg["optimization"][k], dict):
        cfg["optimization"][k] = cfg["optimization"][k][mode_key]
# ... and so on for probe + multi_task_probe (latter via utils.multi_task_loss.merge_multi_task_config)
```

**Rule**: Anything that genuinely differs by mode → per-mode dict. Anything mode-agnostic → scalar. Memory-savers + epochs are per-mode (24 GB SANITY ≠ 96 GB FULL); LR + batch_size are usually scalar (architecturally fixed).

### Why per-mode dicts and not multiple files

Single source of truth for the architecture (`probe_pretrain.yaml` is one file describing P2's recipe, with mode-specific knobs inline) — easier to grep, easier to diff modes, no risk of mode-specific files drifting.

### How to add a new per-mode key

1. Add the dict to the right YAML (`base_optimization.yaml` for shared defaults; per-config leaf for opt-in overrides).
2. Add the flatten in `merge_config_with_args` of EACH consumer (`m09a_pretrain.py`, `m09c_surgery.py`, etc.) — the membership-check pattern (`if k in cfg[...] and isinstance(cfg[...][k], dict)`) makes the flatten safe for legacy configs that ship a scalar.
3. Add CLI override flag (default `None` so unspecified CLI doesn't shadow YAML).

## The opt-in-per-config pattern (iter13 multi_task_probe — RETIRED in v12)

`multi_task_probe.enabled` ships as `{sanity: false, poc: false, full: false}` in `base_optimization.yaml`. iter13 leaf configs (`probe_pretrain.yaml`, `surgery_3stage_DI.yaml`, `surgery_2stage_noDI.yaml`) originally opted in by overriding:

```yaml
multi_task_probe:
  enabled: {sanity: true, poc: true, full: true}
```

**Status (v12 + iter14)**: `probe_pretrain.yaml` now sets `multi_task_probe.enabled: {sanity:false, poc:false, full:false}` (motion_aux replaced it). Phase 4 will do the same in surgery yamls. Legacy `configs/legacy2/ch10_pretrain.yaml` and `configs/legacy2/explora.yaml` don't override → inherit base default `false` for all modes. See [legacy/iter13_multi_task.md](legacy/iter13_multi_task.md) for retired-pivot history.

**Why opt-in per-config and not just enable in base**: scoping. The base file is shared; iter13's pivot shouldn't auto-enable for legacy training paths a future contributor might re-run.

## The motion_aux block (iter13 v12 + iter14 — declared per-config, NOT inherited)

Unlike `multi_task_probe`, **`motion_aux` is declared directly in each leaf config** that wants it — no entry in `base_optimization.yaml`. Trade-off: simpler (no opt-in indirection) but adding it to a new config requires copying the full ~14-line block, not flipping `enabled: true`.

```yaml
# configs/train/probe_pretrain.yaml:178-191 (live; v12 anchor)
# Phase 4 will copy this block into configs/train/surgery_3stage_DI.yaml
# and configs/train/surgery_2stage_noDI.yaml.
motion_aux:
  enabled:
    sanity: true                              # validate code path on 24 GB
    poc:    true
    full:   true                              # the actual research run
  motion_features_path: data/eval_10k_local/motion_features.npy   # m04d output, (9297, 13)
  action_labels_path:   outputs/full/probe_action/action_labels.json
  weight_motion: 0.1                          # λ_motion (overall scaler in L_total)
  weight_ce:     1.0                          # CE branch weight (8-class motion-flow)
  weight_mse:    1.0                          # MSE branch weight (13-D z-normed vec)
  head:
    hidden_dim: 256                           # ~430 K head params total
    dropout:    0.1
  head_lr_multiplier: 10.0                    # head LR = base_lr × 10
```

Per-mode flatten happens in `utils.motion_aux_loss.merge_motion_aux_config(cfg, args, mode_key)` (mirrors `merge_multi_task_config` contract). See [iter14_motion_aux_pivot.md](iter14_motion_aux_pivot.md) for the full design.

## The CLI-overrides-YAML pattern

Every CLI flag that overrides a YAML value has `default=None`. The flatten reads:
```python
if getattr(args, "<key>", None):
    cfg["<section>"]["<key>"] = args.<key>
```
This way, an unspecified CLI doesn't shadow the YAML value with `None`. **Never** put a numeric default in argparse — that's a hardcoded value (CLAUDE.md ban). See [feedback_no_hardcoded_defaults.md](feedback_no_hardcoded_defaults.md).

## Ckpt-schema dispatch (`utils.frozen_features.resolve_encoder_state_dict`)

V-JEPA-style checkpoints have one of 4 top-level key conventions. The eval-side loader walks them in priority order:

| Priority | Key | Source | Saved by |
|---|---|---|---|
| 1 | `target_encoder` | Meta's V-JEPA 2.1 frozen ckpt (EMA teacher = best quality) | Meta's pretrain |
| 2 | `encoder` | Older Meta convention | Older Meta releases |
| 3 | `student_state_dict` | iter13 export artifact (encoder-only) | `utils.training.export_student_for_eval` → `student_encoder.pt` |
| 4 | `student` | iter13 full ckpt (encoder + teacher + predictor + opt) | `utils.training.save_training_checkpoint(full=True)` → `m09{a,c}_ckpt_best.pt` |
| 5 | raw dict | Last-resort fallback (state_dict already at top level) | Manual exports |

Without this dispatch, iter13's `student_encoder.pt` would fall through to the raw-dict path and report `0/588 missing keys` (the wrapper dict's `{"student_state_dict", "model_id", "type"}` get treated as state). See [bug_log.md](bug_log.md) for the regression history.

## CACHE_POLICY contract

Every destructive `.py` (m04/m04d/m05/m05b/m05c/m06/m08/m08b/m09a/m09b/m09c/m10/m11) registers `--cache-policy {1,2}` via `utils.cache_policy.add_cache_policy_arg()`:
- `1` (default, Enter) — keep all caches; log a `[cache-policy=1/keep]` line per artifact
- `2` — recompute; route through `guarded_delete(path, policy, label)` which logs `[cache-policy=2/recompute] deleted ...`

When `--cache-policy` is unspecified, the .py prompts via `input("[1=keep / 2=recompute] (Enter=1): ")` — UNLESS the env var `CACHE_POLICY_ALL=1|2` is set (overnight runs). Shells stay THIN — they don't prompt; they pass `--cache-policy "$P_X"` per-call.

`run_probe_eval.sh` collects ALL cache-policy decisions UPFRONT in one block (lines 200-247) before any compute starts, then passes the resolved value to each .py.

## YAML field discoverability

Per CLAUDE.md "VERIFY-FIRST RECOMMENDATIONS" rule: never guess a YAML key. Always grep first:
```bash
grep -nE "^<key>:" configs/train/*.yaml configs/pipeline.yaml
grep -rE "cfg\[.<key>.\]|\.<key>\b" src/   # who reads it
```

A common bug class is **dead YAML fields** (declared but read by zero .py — operator believes one schedule, code runs another). The B61 preflight check catches these for `lr_schedule`, `schedule_type`, `optimizer_type`, `sampler_type`. If you add a new YAML key, also add a consumer in some .py the same session — don't defer.
