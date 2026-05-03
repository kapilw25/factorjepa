# iter13 Runbook — terminal commands for SANITY → FULL

> **Hardware split** (validated 2026-05-03):
> - **Eval pipeline** (`run_probe_eval.sh`) — runs on **24 GB** SANITY OR 96 GB FULL.
> - **Training** (`run_probe_train.sh pretrain | surgery_*`) — requires **96 GB FULL** (V-JEPA ViT-G + EMA teacher + optimizer state OOMs at 24 GB even with all mitigations stacked: 8-bit Adam + paged optim + grad-ckpt + sub-batch=1 + 8 frames). See `probe_pretrain_sanity_v6.log` for the OOM evidence.
>
> **Encoder roster** (4 V-JEPA + 1 DINOv2):
> - `vjepa_2_1_frozen` — Meta's checkpoint (P1 control)
> - `vjepa_2_1_pretrain` — m09a continual SSL with multi-task probe loss (P2)
> - `vjepa_2_1_surgical_3stage_DI` — m09c factor surgery WITH interaction tubes (P3a)
> - `vjepa_2_1_surgical_noDI` — m09c factor surgery WITHOUT D_I (P3b)
> - `dinov2` — frozen DINOv2 ViT-G/14 (P1 baseline)

---

## Phase A — SANITY on 24 GB (eval-side validation; ~6-8 min)

> Validates the eval pipeline end-to-end on 150 stratified clips (50/class). Numbers are NOT meaningful (n_test ≈ 22 after stratified split → BCa CIs degenerate to NaN on perfect predictions). Purpose: code-path correctness, NOT model performance.

```bash
# Full SANITY eval pipeline. Drops trainer-output encoders that aren't on disk
# (vjepa_2_1_frozen + dinov2 always available; m09a/m09c outputs only if Phase B already ran).
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --sanity \
  2>&1 | tee logs/run_src_probe_sanity_v1.log

# Re-run only Stage 10 (plotting) without recomputing Stages 1-9:
SKIP_STAGES="1,2,3,4,5,6,7,8,9" CACHE_POLICY_ALL=1 \
  ./scripts/run_probe_eval.sh --sanity \
  2>&1 | tee logs/run_src_probe_sanity_plot_only.log
```

**Verify SANITY pass:**
```bash
ls outputs/sanity/probe_plot/    # → 3 PNGs + 3 PDFs
jq '.pairwise_deltas | keys'  outputs/sanity/probe_action/probe_paired_delta.json
jq '.by_encoder | keys'       outputs/sanity/probe_motion_cos/probe_motion_cos_paired.json
jq '.by_variant | keys'       outputs/sanity/probe_future_mse/probe_future_mse_per_variant.json
```

**Expected outcomes at SANITY** (per `run_src_probe_sanity_v4.log`):
- All 10 stages complete (Stage 8 will SKIP V-JEPA variants whose `m09{a,c}_ckpt_best.pt` isn't on disk; that's by-design)
- DINOv2 ≈ 95.45% top-1; V-JEPA frozen = 100% (saturation on N=22; expected per CLAUDE.md SANITY rule)
- BCa CIs may report `NaN` for perfect-prediction encoders — `probe_plot._bar_with_ci` is NaN-safe (iter13 fix)

---

## Phase B — Training on 96 GB FULL (~10-15 GPU-h)

> All three trainers ship their `m09{a,c}_ckpt_best.pt` (predictor-bearing) artifact that probe Stage 8 (future_mse) consumes. Multi-task probe loss is `enabled: {sanity:true, poc:true, full:true}` in each per-config YAML, so multi-task supervision runs by default at FULL.

### B.1 — Bootstrap labels first (CPU, ~2 min, only needed once)

```bash
# Generates outputs/full/probe_action/action_labels.json (P1 split source)
# AND outputs/full/probe_taxonomy/taxonomy_labels.json (multi-task supervision source).
# Both consumed by Phase B trainers and Phase C eval.
SKIP_STAGES="2,3,4,5,6,7,8,9,10" CACHE_POLICY_ALL=2 \
  ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/run_probe_eval_full_stage1_only.log
```

### B.2 — Train all 3 V-JEPA variants (sequential or parallel via tmux)

```bash
tmux new -s p2_pretrain
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain --FULL \
  2>&1 | tee logs/probe_pretrain_full_v1.log
# ~3 GPU-h on 96 GB; produces:
#   outputs/full/probe_pretrain/student_encoder.pt   (~7 GB, encoder only)
#   outputs/full/probe_pretrain/m09a_ckpt_best.pt    (~15 GB, encoder+predictor for Stage 8)
#   outputs/full/probe_pretrain/multi_task_head.pt   (~1 MB, mt_head state)

# Detach from tmux: Ctrl-b d, then re-attach: tmux attach -t p2_pretrain

tmux new -s p3_surgery_3stage_DI
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL \
  2>&1 | tee logs/probe_surgery_3stage_DI_full_v1.log
# ~6-8 GPU-h on 96 GB; produces:
#   outputs/full/probe_surgery_3stage_DI/student_encoder.pt
#   outputs/full/probe_surgery_3stage_DI/m09c_ckpt_best.pt
#   outputs/full/probe_surgery_3stage_DI/multi_task_head.pt

tmux new -s p3_surgery_noDI
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI --FULL \
  2>&1 | tee logs/probe_surgery_noDI_full_v1.log
# ~4-6 GPU-h on 96 GB (no Stage 3 D_I); produces:
#   outputs/full/probe_surgery_noDI/student_encoder.pt
#   outputs/full/probe_surgery_noDI/m09c_ckpt_best.pt
#   outputs/full/probe_surgery_noDI/multi_task_head.pt
```

**Verify training success per variant:**
```bash
for v in probe_pretrain probe_surgery_3stage_DI probe_surgery_noDI; do
  echo "── $v ──"
  ls -lh outputs/full/$v/{student_encoder,m09{a,c}_ckpt_best,multi_task_head}.pt 2>/dev/null
  wc -l outputs/full/$v/loss_log.jsonl       # should be > 0 (silent-Meta-export Bug B is fixed)
done
```

### B.3 — Optional: SANITY smoke for each trainer on 96 GB before committing FULL hours

If you want a 5-min code-path verify per trainer before kicking the multi-hour FULL runs:

```bash
# Each ~3-10 min on 96 GB; useful only on 96 GB hardware (24 GB OOMs per Phase A note above)
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain --SANITY          2>&1 | tee logs/probe_pretrain_sanity_smoke.log
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --SANITY 2>&1 | tee logs/probe_surgery_3stage_DI_sanity_smoke.log
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI --SANITY      2>&1 | tee logs/probe_surgery_noDI_sanity_smoke.log
```

---

## Phase C — Eval all 4 V-JEPA variants + DINOv2 on FULL (~2.5 GPU-h)

> Once Phase B has produced all `m09{a,c}_ckpt_best.pt` artifacts, run the eval pipeline at FULL scale (eval_10k = ~9.9k clips → ~6,966 train / ~1,492 val / ~1,492 test).

```bash
tmux new -s probe_eval_full
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/run_src_probe_full_v1.log
```

**Read the gates:**
```bash
# P1 GATE — Δ V-JEPA frozen vs DINOv2 frozen on top-1 acc (target +20 pp)
jq '.pairwise_deltas.dinov2_minus_vjepa_2_1_frozen' \
  outputs/full/probe_action/probe_paired_delta.json

# P2 GATE — Δ pretrain vs frozen (target Δ > 0, p < 0.05)
jq '.pairwise_deltas.vjepa_2_1_pretrain_minus_vjepa_2_1_frozen' \
  outputs/full/probe_action/probe_paired_delta.json

# P3 GATE — Δ surgical_3stage_DI vs surgical_noDI (D_I helps?)
jq '.pairwise_deltas.vjepa_2_1_surgical_3stage_DI_minus_vjepa_2_1_surgical_noDI' \
  outputs/full/probe_action/probe_paired_delta.json

# Future MSE — V-JEPA's native objective (lower is better)
jq '.by_variant | to_entries | map(select(.value != null and .value != "n/a — no future-frame predictor"))' \
  outputs/full/probe_future_mse/probe_future_mse_per_variant.json

# Final 4-encoder bar comparison
xdg-open outputs/full/probe_plot/probe_encoder_comparison.png 2>/dev/null \
  || echo "Plots: outputs/full/probe_plot/{probe_action_loss,probe_action_acc,probe_encoder_comparison}.{png,pdf}"
```

---

## End-to-end one-liner (Phase A → B → C, ~12-18 GPU-h on 96 GB)

```bash
# 1. SANITY smoke (~6-8 min on 24 GB OR 96 GB)
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --sanity \
  2>&1 | tee logs/run_src_probe_sanity.log

# 2. Bootstrap labels at FULL scale (~2 min)
SKIP_STAGES="2,3,4,5,6,7,8,9,10" CACHE_POLICY_ALL=2 \
  ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/run_probe_eval_full_stage1_only.log

# 3. Train all 3 V-JEPA variants (semicolons — independent, no early-abort cascade)
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain --FULL          2>&1 | tee logs/probe_pretrain_full_v1.log ; \
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL 2>&1 | tee logs/probe_surgery_3stage_DI_full_v1.log ; \
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI --FULL      2>&1 | tee logs/probe_surgery_noDI_full_v1.log

# 4. FULL eval — all 4 V-JEPA + DINOv2
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL \
  2>&1 | tee logs/run_src_probe_full_v1.log
```

> Why semicolons (not `&&`) between trainers: per CLAUDE.md "OVERNIGHT CHAINS — `;` NOT `&&`". A failure in one trainer must not cancel the next; each writes independent artifacts that the eval pipeline can consume independently.

---

## Failure recovery

| Symptom | Diagnosis | Fix |
|---|---|---|
| Training OOMs at sub-batch=1 | Hardware too small (24 GB) | Move to 96 GB; SANITY pretrain/surgery are 96-GB-only |
| Stage 8 FATAL: `m09{a,c}_ckpt_best.pt missing` | Pre-iter13 trainer didn't save full ckpt | iter13 Bug A/R8 fixes ensure `_best.pt` is `full=True`; re-train with `CACHE_POLICY_ALL=2` |
| Pretrain `loss_log.jsonl` is 0 bytes | Bug B (silent 0-step exit) | iter13 fix raises hard with `M09A FAILED: 0 successful training steps`; re-run on bigger GPU |
| `probe_encoder_comparison.png` missing / FATAL ValueError "Axis limits cannot be NaN" | Degenerate BCa CI (perfect predictions on tiny test set) | iter13 NaN-safe ylim in `probe_plot.py:_bar_with_ci`; only fires at SANITY n=22 |
| `taxonomy_labels.json` missing → multi-task auto-disabled | Phase B.1 not run | Run Phase B.1 once; `run_probe_train.sh` auto-generates if sources are present |

---

## Reference

- **Plan**: `iter/iter13_motion_probe_eval/plan_training.md`
- **Deep research**: `iter/iter13_motion_probe_eval/analysis.md`
- **Code-dev plan**: `iter/iter13_motion_probe_eval/plan_code_dev.md`
- **Bug log**: `iter/iter13_motion_probe_eval/errors_N_fixes.md`
- **Onboarding**: `.claude/memory/MEMORY.md` (iter13 state for fresh Claude Code sessions)
