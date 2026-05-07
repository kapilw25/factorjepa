---
name: iter14 motion_aux pivot — design + state
description: motion_aux (CE+MSE on motion-flow classes + 13-D normalized vec) replaced multi_task_probe in v12 and is the iter14 surgery aux loss
type: project
---

# iter14 motion_aux pivot — design + state (2026-05-07 PDT)

## Why this exists (and why it replaced multi_task_probe)

iter13's first pivot was `multi_task_probe` (CE on 14 single-label dims + BCE on 2 multi-label dims, 16-D taxonomy from `tags.json`) — see `legacy/iter13_multi_task.md`. It plateaued at modest gains because the 16-D taxonomy is a *scene-tag* signal, not a *motion-temporal* signal, and the paper goal evaluates on motion-flow probe top-1.

**v12 pivot (2026-05-06)**: replace `multi_task_probe` with `motion_aux` — CE on 8 motion-flow classes (derived from optical-flow `mean_mag` quartile + 8-bin direction histogram via `utils/action_labels.parse_optical_flow_class`) + MSE regression on the 13-D z-normalized motion-feature vector itself.

**Empirical result on m09a pretrain (5 ep FULL)**:

| Metric | Pre-v12 (multi_task) | v12 (motion_aux) | Lift |
|---|---|---|---|
| `probe_top1` (motion-flow, 8-class) | ~0.44 | **0.808** | +37 pp |
| `motion_cos` vs frozen | ~1× | **5.8×** (0.046→0.267) | — |
| `future_mse` Δ vs frozen | ~0 | **+0.0027** (CI [0.0017, 0.0037], p=0.0) | statistically separated |

Source: `outputs/full/probe_pretrain/probe_history.jsonl` final row (step 1009, epoch 4, full=100%).

## The loss

```
total_loss = α · JEPA_L1
           + λ_drift · L2(student, init)        # iter14 anchor; was 0 in v12 pretrain
           + w_motion · (w_ce · CE(logits, motion_class) + w_mse · MSE(vec_pred, vec_target_normed))
            (w_motion = 0.1, w_ce = 1.0, w_mse = 1.0 — default)
```

| Component | Default | Where |
|---|---|---|
| `weight_motion` (overall scaler) | 0.1 | `cfg.motion_aux.weight_motion` |
| `weight_ce` | 1.0 | `cfg.motion_aux.weight_ce` |
| `weight_mse` | 1.0 | `cfg.motion_aux.weight_mse` |
| Head LR multiplier | 10× base_lr | `cfg.motion_aux.head_lr_multiplier` |
| Head hidden dim | 256 | `cfg.motion_aux.head.hidden_dim` |
| Head dropout | 0.1 | `cfg.motion_aux.head.dropout` |
| Z-norm buffers | `(mean, std)` of `motion_features.npy` over 9297 clips | computed once at head construction |

## The 5 integration helpers (`utils/motion_aux_loss.py`)

Mirrors the `utils/multi_task_loss.py` contract exactly so call sites in m09a/m09c read symmetrically:

| Helper | Signature | Purpose |
|---|---|---|
| `merge_motion_aux_config(cfg, args, mode_key)` | in-place mutate cfg | Per-mode flatten of `enabled` dict + CLI overrides (`--motion-features-path`, `--no-motion-aux`) |
| `build_motion_aux_head_from_cfg(cfg, device)` | → `(head, lookup_by_clip, ma_cfg)` | Construct MotionAuxHead + load motion_features.npy + action_labels.json + compute z-norm; silent-disable if files missing |
| `attach_motion_aux_to_optimizer(optimizer, ma_head, ma_cfg, base_lr)` | in-place mutate optimizer | Add head's params as separate group at `base_lr × head_lr_multiplier` |
| `run_motion_aux_step(student, ma_head, ma_cfg, ma_lookup, batch_clips, batch_keys, scaler, mp_cfg, dtype, device)` | → `(ma_loss_val, ma_per_branch)` | One forward+backward; re-raises `OutOfMemoryError` for caller's per-loop policy |
| `export_motion_aux_head(ma_head, path)` | side effect | Write `motion_aux_head.pt` next to student_encoder.pt |

All 5 helpers no-op when `ma_head is None`, so call sites never need a `if ma_head is not None:` guard around them.

## Per-config declaration (NOT inherited from base)

Unlike `multi_task_probe` (which lives in `base_optimization.yaml` with `enabled: false` defaults that leaf configs override), `motion_aux` is declared **directly in each leaf config** that wants it. This is simpler — there's no opt-in indirection — but the cost is that adding motion_aux to a new config means copying the full block, not just flipping `enabled: true`.

Currently declared in `configs/train/probe_pretrain.yaml:178-191`. Phase 4 adds the same block to `surgery_3stage_DI.yaml` + `surgery_2stage_noDI.yaml`:

```yaml
# probe_pretrain.yaml + (Phase 4) surgery_3stage_DI.yaml + surgery_2stage_noDI.yaml
motion_aux:
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

**Phase 4** = swap `multi_task_probe.enabled: {sanity:true, poc:true, full:true}` → `false` in surgery yaml + add the motion_aux block above + thread `--motion-features-path` in `run_probe_train.sh` for surgery cases.

## Critical wiring nuance for m09c surgery (vs m09a pretrain)

m09c rebuilds the optimizer **per stage** (3 stages for surgery_3stage_DI: D_L → D_A → D_I; 2 stages for surgery_noDI: D_L → D_A). Each stage rebuild must also re-attach the motion_aux head's params to the new optimizer:

```python
# Inside the per-stage build_optimizer block (m09c_surgery.py):
attach_motion_aux_to_optimizer(optimizer, ma_head, ma_cfg, base_lr=cfg["optimization"]["lr"])
```

m09a v12 only rebuilds the optimizer once (no stages), so it has the call-site only at the initial optimizer build.

## Data path: how clip_keys reach `run_motion_aux_step`

- **m09a** (continual SSL): `producer_thread` yields `(msg_type, batch_clips, batch_keys)` — same 3-tuple as multi_task_probe path.
- **m09c** (factor surgery): `StreamingFactorDataset` yields `{"tensor", "factor_type", "clip_key"}`; collator turns clip_key into a list. Threaded as `batch_keys = list(batch["clip_key"])`.

If `batch_keys` is empty (defensive), `run_motion_aux_step` short-circuits to `(0.0, {})`.

## Motion forward cost

Adds ONE extra full-batch unmasked encoder forward per macro batch (same as multi_task_probe path). With grad-ckpt + bf16 on V-JEPA ViT-G, this is ~6-8 GB on a single forward at 16-frame full-batch.

That's why this path is fine on 96 GB but stacks badly on 24 GB. See `hardware_split.md`.

## What's NOT yet implemented

- **Phase 5 — 23-D foreground motion** (`plan_phase5_fg_motion_features.md`): camera-subtracted FG flow + class rebin on `vec[13]` instead of `vec[0]`. Gated on Phase 4 outcome.
- **Phase 4 wiring in m09c** itself: 9 call sites + per-stage optimizer re-attach. The plan is in `iter/iter14_surgery_on_pretrain/plan_motion_aux_to_surgery.md`; T4 is blocked on the 3 approval gates in `plan_surgery_on_pretrain.md`.
- **mt_head warm-start at eval time** — `motion_aux_head.pt` is exported but eval-time probe doesn't load it. Could be a small lift.

## References

- v12 final result: `outputs/full/probe_pretrain/probe_history.jsonl` (step 1009 → top1 0.808)
- m09a v12 wiring: `src/m09a_pretrain.py` (~9 call sites: imports / argparse / merge_config / build_head / attach_optim / run_step / step_record / wb_metrics / export)
- Helpers: `src/utils/motion_aux_loss.py` (~340 LoC including MotionAuxHead class)
- Phase 4 plan: `iter/iter14_surgery_on_pretrain/plan_motion_aux_to_surgery.md`
- Phase 5 plan: `iter/iter14_surgery_on_pretrain/plan_phase5_fg_motion_features.md`
- iter13 predecessor (RETIRED): `legacy/iter13_multi_task.md`
