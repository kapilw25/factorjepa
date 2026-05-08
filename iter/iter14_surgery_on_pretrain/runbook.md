# 🚀 iter14 surgery-on-pretrain — runbook

- Canonical command sequence for the 4-arm experiment from `plan_surgery_on_pretrain.md`.
- Reference: Δ1 (pretrain > frozen) · Δ2 (surgery > pretrain) · Δ3 (surgery > pretrain_2X — the **causal** claim).

## ✅ Prerequisites

- 96 GB RTX Pro 6000 Blackwell instance (24 GB cannot train V-JEPA ViT-G — see [hardware_split.md](../../.claude/memory/hardware_split.md))
- venv active: `source venv_walkindia/bin/activate`
- `.env` at project root with `HF_TOKEN` (m09c hf_hub_download depends on this; FAIL LOUD if missing)
- `data/eval_10k_local/{motion_features.npy, m11_factor_datasets/}` on disk
  (auto-pulled via `python -u src/utils/hf_outputs.py download-data` if missing)
- HF endpoint live: `https://huggingface.co/anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep`
  (m09c surgery downloads `student_encoder.pt` from here on first invocation, caches in `HF_HOME`)

## 🧪 SANITY smoke (~25 min, MUST run before FULL)

```bash
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain_2X       --SANITY 2>&1 | tee logs/iter14_sanity_pretrain_2X_v1.log
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --SANITY 2>&1 | tee logs/iter14_sanity_surgery_3stage_DI_v1.log
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI      --SANITY 2>&1 | tee logs/iter14_sanity_surgery_noDI_v1.log
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --sanity 2>&1 | tee logs/iter14_sanity_eval_v1.log
```

## 🧪 POC dry-run (~2-4 GPU-h, intermediate validation between SANITY and FULL)

POC = small-scale research validation: probe ON, eval_10k.json subset (paper-grade), 1-2 epochs, no 24GB memory savers. Useful for catching plateau / convergence issues before committing to FULL's $30 budget. POC eval reads from `outputs/poc/` (matches POC training output dir).

```bash
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain_2X       --POC 2>&1 | tee logs/iter14_poc_pretrain_2X_v1.log
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --POC 2>&1 | tee logs/iter14_poc_surgery_3stage_DI_v1.log
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI      --POC 2>&1 | tee logs/iter14_poc_surgery_noDI_v1.log
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh                    --poc 2>&1 | tee logs/iter14_poc_eval_v1.log
```

**Pass criteria** (verify before kicking FULL):

| Check | Where | Expected |
|---|---|---|
| HF download fired | `logs/iter14_sanity_surgery.log` | `[iter14] HF download: anonymousML123/factorjepa-pretrain-vjepa21-vitg-5ep/student_encoder.pt` |
| Schema accepted | same | `[iter14] Schema: student_state_dict (588 keys)` |
| Anchor loss firing | `outputs/sanity/m09c_surgery_3stage_DI/loss_log.jsonl` | `loss_drift` non-zero (drift_control.lambda_reg=0.005) |
| Δ1/Δ2/Δ3 keys emitted | `outputs/sanity/probe_action/probe_paired_delta.json` | `iter14_paper_deltas.{delta_1_*, delta_2_*, delta_3_*}` present (CIs may be NaN at SANITY n=200; just need keys) |
| Trainer artifacts | each output dir | `student_encoder.pt` + `m09{a,c}_ckpt_best.pt` |

## 🚀 FULL training arms (~37 GPU-h ≈ $30 at $0.8/hr)

**Use `;` not `&&` between arms** — independent failures shouldn't cancel the queue (CLAUDE.md OVERNIGHT CHAINS).

```bash
tmux new -s iter14_train

# Arm B-1: surgery_3stage_DI on pretrain init (~10 h, ~$8)
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_3stage_DI --FULL 2>&1 | tee logs/iter14_surgery_3stage_DI.log ; \
# Arm B-2: surgery_noDI on pretrain init (~7 h, ~$6, ablation for D_I contribution)
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh surgery_noDI      --FULL 2>&1 | tee logs/iter14_surgery_noDI.log ; \
# Arm C: long-pretrain compute control (~20 h, ~$16)
# ⚠️ ONLY launch if Arm B-1 shows signal — quick check after Arm B-1:
#    tail outputs/full/m09c_surgery_3stage_DI/probe_history.jsonl | jq .probe_top1
#    If probe_top1 plateaus < 0.808 (the v12 pretrain anchor), abort Arm C.
CACHE_POLICY_ALL=2 ./scripts/run_probe_train.sh pretrain_2X       --FULL 2>&1 | tee logs/iter14_pretrain_2X.log
```

## 🧪 5-encoder FULL eval (~4 h, ~$3)

```bash
tmux new -s iter14_eval
CACHE_POLICY_ALL=2 ./scripts/run_probe_eval.sh --FULL 2>&1 | tee logs/iter14_probe_eval.log
```

Encoders evaluated: `vjepa_2_1_frozen` + `vjepa_2_1_pretrain` + `vjepa_2_1_pretrain_2X` + `vjepa_2_1_surgical_3stage_DI` + `vjepa_2_1_surgical_noDI`.

## 🎯 Inspect Δ1/Δ2/Δ3 paper deltas

```bash
jq '.iter14_paper_deltas' outputs/full/probe_action/probe_paired_delta.json
```

| Delta | Comparison | Pass when |
|---|---|---|
| Δ1 | pretrain − frozen | `delta_1_pretrain_vs_frozen.ci_lo_pp > 0` |
| Δ2 | surgical_3stage_DI − pretrain | `delta_2_surgical_vs_pretrain.ci_lo_pp > 0` |
| Δ3 | surgical_3stage_DI − pretrain_2X | `delta_3_surgical_vs_pretrain_2X.ci_lo_pp > 0` ⭐ causal claim |

## 🚦 Decision matrix

| Outcome | Action |
|---|---|
| Δ1 ✅ + Δ2 ✅ + Δ3 ✅ | 🏆 **publishable** — strict ordering proven; write paper |
| Δ1 ✅ + Δ2 ✅ + Δ3 ❌ | Weaker claim publishable: "factor patching ≥ extra training steps" |
| Δ1 ✅ + Δ2 ❌ | Surgery doesn't beat pretrain → drop surgery claim; report negative result; consider Phase 5 (FG motion features, see `plan_phase5_fg_motion_features.md`) |
| Δ1 ❌ | Unexpected — already proven by v12 anchor (0.808 top1, +13.16 pp vs frozen). Investigate eval pipeline regression |

## 🔍 Mid-run diagnostics

```bash
# Watch all 3 trainer tmux logs at once
multitail -f logs/iter14_surgery_3stage_DI.log -f logs/iter14_surgery_noDI.log -f logs/iter14_pretrain_2X.log

# Check anchor loss is firing (iter14 marker)
jq -r 'select(.loss_drift != null) | "step=\(.step) jepa=\(.loss_jepa) drift=\(.loss_drift)"' \
  outputs/full/m09c_surgery_3stage_DI/loss_log.jsonl | tail -20

# Check probe trajectory
tail outputs/full/m09c_surgery_3stage_DI/probe_history.jsonl | jq '{step, probe_top1, motion_cos, future_l1}'
```
