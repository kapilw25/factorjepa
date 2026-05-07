---
name: Project pulse — FactorJEPA iter14 (surgery-on-pretrain)
description: Where the work is, what's done, what's next, as of 2026-05-07 PDT
type: project
---

# FactorJEPA iter14 — current state (2026-05-07 PDT)

## What this project is

V-JEPA 2.1 ViT-G (1664-dim, 48 layers) continual pretraining + factor-conditioned surgery on Indian-context urban driving / walking / monument video, evaluated against frozen V-JEPA + DINOv2 baselines on Meta's published frozen-encoder + 4-layer attentive-probe protocol applied to our own labeled clips.

**Paper goal**: `vjepa_surgery ≫ vjepa_pretrain ≫ vjepa_frozen` on motion / temporal features (16-class motion-flow probe).

## What landed in iter13 (DONE — do not re-run)

The iter13 pivot evolved twice:
1. **multi_task_probe** (16-dim CE+BCE on taxonomy) — wired in m09a/m09c, plateaued at modest gains. See `legacy/iter13_multi_task.md`.
2. **motion_aux (v12)** — CE on 8 motion-flow classes + MSE on 13-D normalized motion-feature vector. **REPLACED** multi_task_probe (m09a v12 disables `multi_task_probe` and runs motion_aux only).

### v12 motion_aux empirical anchor (2026-05-06)

| Metric | Value | Where |
|---|---|---|
| `probe_top1` (motion-flow) | **0.808** at step 1009 / 5 epochs | `outputs/full/probe_pretrain/probe_history.jsonl` |
| `motion_cos` lift vs frozen | **5.8×** (0.046 → 0.267) | same file |
| `future_mse` Δ vs frozen | **+0.0027** (CI [0.0017, 0.0037], p=0.0) | `outputs/full/probe_future_mse/probe_future_mse_per_variant.json` |
| Clips processed | 9,297 / 10,000 (703 silently dropped at TAR-reader; FIX-27a now logs them) | — |

**Half of the strict ordering is statistically established**: `pretrain > frozen` on all three eval axes. iter14 = prove the second half (`surgery > pretrain`) cleanly.

### Durable artifacts now on Hugging Face

| Endpoint | What | Wall delta vs local |
|---|---|---|
| `anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep` (model repo) | `student_encoder.pt` (6.9 GB) + `m09a_ckpt_best.pt` (14 GB) | uploaded 2026-05-06 |
| `anonymousML123/factorjepa-outputs` (dataset repo) | `outputs/full/` paper-evidence (probe_taxonomy 15 GB + probe_action 1.5 GB + probe_motion_cos 18 MB + probe_future_mse 0.3 MB + probe_plot 828 KB); 8875 stale → ~150 final after mirror cleanup | uploaded 2026-05-07 21:38 PDT, 95s wall |

**GPU teardown is safe** — both model checkpoint and paper-evidence outputs are durable.

## What iter14 is

**Surgery on top of the v12 pretrain student**, with a compute-matched **long-pretrain (10 ep)** ablation arm so any gain is causally attributable to factor patching, not extra training steps.

Two phases planned:

| Phase | Goal | Plan doc |
|---|---|---|
| **Phase 4** | Wire motion_aux (CE+MSE recipe that lifted m09a to 0.808) into `m09c_surgery.py` — swap `multi_task_probe` → `motion_aux` in `surgery_3stage_DI.yaml` + `surgery_2stage_noDI.yaml`; mirror m09a's 9 call-sites in m09c (with stage-aware optimizer re-attach) | `iter/iter14_surgery_on_pretrain/plan_motion_aux_to_surgery.md` |
| **Phase 5** | (CONDITIONAL) — extend `m04d_motion_features.py` to 23-D foreground motion (camera-subtracted) + rebin classes on FG vec[13]. **Gated**: only run if Phase 4 yields Δ (surgery − pretrain) < +5 pp | `iter/iter14_surgery_on_pretrain/plan_phase5_fg_motion_features.md` |

The high-level + detail plans are in `iter/iter14_surgery_on_pretrain/plan_HIGH_LEVEL.md` + `plan_surgery_on_pretrain.md`.

## Where we are stuck — three approval gates blocking T4 (code edits)

Per `plan_surgery_on_pretrain.md` lines ~308-315:

1. **Epoch budget**: 🅰️ "5+5 vs 10" (~$33, recommended) OR 🅱️ "5+15 vs 20" (~$57)?
2. **Anchor `λ`**: `drift_control.lambda_reg = 0.005` (literature default) OR 3-point sweep `{0.001, 0.005, 0.01}` (3× surgery cost)?
3. **HF push of pretrain**: ✅ DONE (no decision needed).

User's reply form: `"go: 🅰️, λ=0.005"`. Until that lands, T4 (7 file edits across configs/ + src/ + scripts/) is blocked.

## Files modified across iter13 v12 + iter14 plan (recent commits)

```
src/m09a_pretrain.py                               motion_aux wiring (9 call sites)
src/m09c_surgery.py                                M (modified) — Phase 4 wiring NOT YET applied
src/utils/motion_aux_loss.py                       NEW (~340 LoC) — head + loss + 5 helpers (mirrors multi_task_loss.py contract)
src/utils/probe_labels.py                          NEW (untracked) — likely motion-flow class derivation
src/m04d_motion_features.py                        produces motion_features.npy (13-D); Phase 5 will extend to 23-D
configs/train/probe_pretrain.yaml                  motion_aux opt-in
src/utils/hf_outputs.py                            FIX-25 + FIX-28 (auto-pack on upload, auto-unpack-and-cleanup on download)
src/utils/data_download.py                         FIX-27a (drop logging in _read_one_tar)
src/utils/curate_verify.py                         FIX-27b (top-N filter to renderable video_ids)
iter/iter14_surgery_on_pretrain/                   NEW dir: HIGH_LEVEL + surgery_on_pretrain + motion_aux_to_surgery + phase5_fg_motion_features + run_factor_prep_parallel
iter/iter14_surgery_on_pretrain/legacy/            plan_next_steps.md (split into Phase 4 + Phase 5 plans, original archived per never-rm rule)
src/CLAUDE.md                                      added "Retiring files: mv to legacy/ — never rm" rule under DELETE PROTECTION
.claude/memory/                                    refreshed for iter14 (this file is the new pulse)
```

## Reference docs (canonical sources)

- **iter14 high-level**: `iter/iter14_surgery_on_pretrain/plan_HIGH_LEVEL.md` (4-encoder design, paper goal, status carryover)
- **iter14 detail**: `iter/iter14_surgery_on_pretrain/plan_surgery_on_pretrain.md` (3 approval gates at top, code-diff in body)
- **Phase 4 plan**: `iter/iter14_surgery_on_pretrain/plan_motion_aux_to_surgery.md` (the 5 file edits)
- **Phase 5 plan**: `iter/iter14_surgery_on_pretrain/plan_phase5_fg_motion_features.md` (gated on Phase 4 outcome)
- **iter13 deep research**: `iter/iter13_motion_probe_eval/analysis.md` (Q1-Q7 + iter13 design + state snapshot)
- **Bug catalog**: `iter/iter14_surgery_on_pretrain/errors_N_fixes.md` (carries forward from iter13 + adds iter13/14 entries)
- **Onboarding**: `.claude/memory/MEMORY.md` (this file's index)
