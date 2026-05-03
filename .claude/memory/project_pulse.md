---
name: Project pulse — FactorJEPA iter13
description: Where the work is, what's done, what's next, as of 2026-05-03 16:30 PDT
type: project
---

# FactorJEPA iter13 — current state (2026-05-03 16:30 PDT)

## What this project is

V-JEPA 2.1 ViT-G continual pretraining + factor-conditioned surgery on Indian-context urban driving / walking / monument video, evaluated against frozen V-JEPA + DINOv2 baselines on Meta's published frozen-encoder + 4-layer attentive-probe protocol applied to our own labeled clips. Three priorities, gated sequentially:

- **P1**: `vjepa_2_1_frozen` outperforms `dinov2` on top-1 action accuracy (target Δ ≥ +20 pp)
- **P2**: `vjepa_2_1_pretrain` (m09a continual SSL + multi-task probe loss) outperforms `vjepa_2_1_frozen`
- **P3**: `vjepa_2_1_surgical_3stage_DI` and/or `vjepa_2_1_surgical_noDI` (m09c factor surgery + multi-task) outperforms `vjepa_2_1_pretrain`

## What landed in iter13 (this session, 2026-05-03)

**The pivot**: pure-SSL m09a/m09c have ZERO gradient signal toward the 16-dim probe metric. Added multi-task probe-loss supervision so the encoder gets DIRECT gradient toward eval metric:
```
total_loss = α·JEPA_L1 + β·Σ_d (1/n_dims)·L_d + drift_L2
            (α=1.0)     (β=0.1)
```
where L_d is CrossEntropy for 14 single-label dims (action + 13 taxonomy) and BCEWithLogits for 2 multi-label dims (road_layout, notable_objects). Labels from `probe_taxonomy.py --stage labels`.

**Code shipped** (all 3-check + REPL smoke validated):
- `utils/multi_task_loss.py` — 5 helpers + MultiTaskProbeHead + compute_multi_task_probe_loss + load_taxonomy_labels_for_training + get_probe_head_param_groups
- `m09a_pretrain.py` + `m09c_surgery.py` — 21 LoC and 19 LoC respectively for full multi-task wiring (init / optimizer attach / forward+backward / export)
- `probe_action.py` Stage 1 emits `action_labels.json`; `probe_taxonomy.py --stage labels` emits `taxonomy_labels.json` (16 dims × clips)
- `scripts/run_probe_train.sh` and `scripts/run_probe_eval.sh` — auto-generate taxonomy labels if missing; 4-encoder pre-flight; Stage 8 graceful skip
- 7 bugs fixed: see [bug_log.md](bug_log.md)

## Where we are stuck

24 GB SANITY can run the **eval** pipeline (`run_probe_eval.sh --sanity`) end-to-end, BUT cannot run **training** (`run_probe_train.sh pretrain | surgery_*`) — V-JEPA ViT-G + EMA teacher + optimizer state OOMs at sub-batch=1 + 8-frame + all memory savers stacked. Empirical evidence in `logs/probe_pretrain_sanity_v6.log`. Fixed footprint ≈ 25 GB > 24 GB budget before activations. Reverted the 8-frame SANITY override on 2026-05-03 16:25 — won't help.

## What's next

**Move pretrain + surgery training to 96 GB FULL hardware.** Eval can stay on either. See [next_actions.md](next_actions.md) for the canonical command sequence and [hardware_split.md](hardware_split.md) for the validated split.

The actual paper question — does multi-task supervision move V-JEPA pretrain/surgical above frozen on probe top-1 — is unanswered until Phase B+C of `iter/iter13_motion_probe_eval/runbook.md` runs on 96 GB.

## Key files modified this session

```
configs/pipeline.yaml                              (no change — reverted action_probe_train block)
configs/train/base_optimization.yaml               + multi_task_probe block (default false all modes)
configs/train/probe_pretrain.yaml                  + multi_task_probe.enabled: {sanity:true, poc:true, full:true}
configs/train/surgery_3stage_DI.yaml               + same opt-in
configs/train/surgery_2stage_noDI.yaml             + same opt-in
src/utils/multi_task_loss.py                       NEW (~340 LoC: head class + loss fn + 5 helpers)
src/utils/frozen_features.py                       + resolve_encoder_state_dict() (4-schema dispatch)
src/probe_future_mse.py                            uses resolve_encoder_state_dict() at line 121
src/probe_plot.py                                  NaN-safe _bar_with_ci ylim + value labels
src/m09a_pretrain.py                               multi-task wiring + Bug A/B fixes + OOM-frag fix
src/m09c_surgery.py                                multi-task wiring + Bug R8 fix + OOM-frag fix
scripts/run_probe_train.sh                         auto-gen taxonomy_labels + threads --taxonomy-labels-json
scripts/run_probe_eval.sh                          Stage 1 emits 2 label artifacts + STAGE8_ENCODERS pre-flight
iter/iter13_motion_probe_eval/runbook.md           rewrote: SANITY → FULL canonical sequence
iter/iter13_motion_probe_eval/plan_training.md     prepended status snapshot
iter/iter13_motion_probe_eval/analysis.md          prepended state snapshot
.claude/memory/                                    NEW (this directory)
```

## Reference docs (canonical sources)

- **Plan**: `iter/iter13_motion_probe_eval/plan_training.md` (status header at top is current as of 2026-05-03)
- **Runbook**: `iter/iter13_motion_probe_eval/runbook.md` (Phase A SANITY → Phase B 96 GB train → Phase C FULL eval)
- **Deep research**: `iter/iter13_motion_probe_eval/analysis.md` (Q1-Q7 + iter13 design + state snapshot)
- **Bug log**: `iter/iter13_motion_probe_eval/errors_N_fixes.md`
- **Code-dev plan**: `iter/iter13_motion_probe_eval/plan_code_dev.md`
- **Onboarding**: `.claude/memory/MEMORY.md` (this file's index)
