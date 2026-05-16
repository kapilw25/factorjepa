# 🚀 iter15 — Runbook: Phase 5 SANITY + Phase 6 POC

📚 Sibling docs: [`plan_trainHead_scaleBackbone_curriculum.md`](./plan_trainHead_scaleBackbone_curriculum.md) · [`planCODE_html.md`](./planCODE_html.md)

📅 Status (2026-05-15):
- ✅ Phase 5 head-only SANITY — complete (24 GB Pro 4000)
- ✅ Phase 5b Cell A (m09a1 pretrain_encoder) — complete (96 GB Blackwell, 92 sec wall)
- ✅ Phase 5b Cell B (m09c1 surgery_3stage_DI_encoder) — complete (96 GB Blackwell, 4 min wall)
- ✅ Phase 5b Cell C (m09c1 surgery_noDI_encoder) — complete (96 GB Blackwell, 1.5 min wall, recipe-v3 parity)
- ✅ Phase 5b Cell D (m09a1 pretrain_2X_encoder) — complete (96 GB Blackwell, 1.5 min wall, iter15 fallback fired)
- 🎉 **Phase 5b ALL 4 CELLS DONE** — every encoder-update code path validated end-to-end
- 🔥 **NEXT** → Phase 6 POC (head-only 3 cells + encoder-update 3 cells)

---

## ✅ Phase 5 head-only SANITY — Complete (2026-05-15, commit 2318de5)

```
┌─────────┬───────────────────────────────────────┬──────────┬──────────────────────────────┐
│ 🚦 Gate  │ 📝 Description                         │ ⏳ Wall   │ 🎯 Result                     │
├─────────┼───────────────────────────────────────┼──────────┼──────────────────────────────┤
│ V0      │ 🧪 CPU yaml + Δ count                  │ 30 sec    │ ✅                            │
│ V2      │ 🧊 m09a2 pretrain_head SANITY           │ 137 sec   │ ✅ 3 ckpts                    │
│ V3a     │ 🔬 m09c2 3stage_DI_head SANITY          │ 175 sec   │ ✅ 3 ckpts                    │
│ V3b     │ 🔬 m09c2 noDI_head SANITY               │ 219 sec   │ ✅ 3 ckpts                    │
│ V4      │ 🧮 encoder bit-invariance (CPU)         │ 10 sec    │ ✅ 576/576 × 3 variants       │
│ V5      │ 📈 probe_future_regress SANITY          │ 128 sec   │ ✅ regressor L1=0.31          │
│ V6      │ 🏁 full eval pipeline SANITY            │ 30 min    │ ✅ 7-Δ grid emitted           │
├─────────┼───────────────────────────────────────┼──────────┼──────────────────────────────┤
│ TOTAL   │ 🎉 Phase 5 SANITY                       │ ~50 min   │ 🐛 15 bugs caught + fixed     │
└─────────┴───────────────────────────────────────┴──────────┴──────────────────────────────┘
```

📦 Artifacts: `outputs/sanity/m09{a,c}_*_head/{student_encoder,m09{a,c}_ckpt_best,motion_aux_head}.pt` · `outputs/sanity/probe_action/probe_paired_delta.json` (Δ6 + Δ7 computed; Δ1-Δ5 skipped pending encoder-update SANITY below).

---

## 🔓 Phase 5b — m09a1/m09c1 ENCODER-UPDATE SANITY (96 GB Blackwell only)

> 🚨 Encoder-update training OOMs on 24 GB. Run on 96 GB Blackwell to validate code paths before Phase 6 POC.

```
┌────────┬──────────────────────────────────────┬──────────┬──────────────────────────────────────┐
│ 🚦 Cell │ 📝 Description                        │ ⏳ Wall   │ 🎯 Result                             │
├────────┼──────────────────────────────────────┼──────────┼──────────────────────────────────────┤
│ 🟢 A    │ 🧠 m09a1 pretrain_encoder             │ 92 sec    │ ✅ best.pt 14 GB (iter15 fallback)    │
│ 🟢 B    │ 🔬 m09c1 surgery_3stage_DI_encoder    │ 4 min     │ ✅ 3 stages · best.pt 14 GB           │
│ 🟢 C    │ 🔬 m09c1 surgery_noDI_encoder         │ 1.5 min   │ ✅ 2 stages · recipe-v3 parity        │
│ 🟢 D    │ 🧠 m09a1 pretrain_2X_encoder          │ 1.5 min   │ ✅ best.pt 14 GB (iter15 fallback)    │
└────────┴──────────────────────────────────────┴──────────┴──────────────────────────────────────┘
```

```bash
# ✅ Cell A — DONE 2026-05-15 21:17 (m09a1 iter15 fallback fired correctly)
# CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_encoder --SANITY  # already complete

# ✅ Cell B — DONE 2026-05-15 21:38 (HF init from iter14 pretrain, 3 stages, 4 min wall)
# CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_3stage_DI_encoder --SANITY  # already complete

# ✅ Cell C — DONE 2026-05-15 22:01 (2 stages, 1.5 min wall, recipe-v3 parity applied)
# CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_noDI_encoder --SANITY  # already complete

# ✅ Cell D — DONE 2026-05-15 21:51 (1.5 min wall, iter15 fallback fired, same code path as Cell A)
# CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_2X_encoder --SANITY  # already complete

# 🔬 Post-check (all 4 cells complete): ckpts at FLAT path (run_eval.sh:210-211 + 226-227)
ls -la outputs/sanity/m09a_pretrain_encoder/{student_encoder.pt,m09a_ckpt_best.pt}
ls -la outputs/sanity/m09c_surgery_3stage_DI_encoder/{student_encoder.pt,m09c_ckpt_best.pt}
ls -la outputs/sanity/m09c_surgery_noDI_encoder/{student_encoder.pt,m09c_ckpt_best.pt}
grep -E 'Stage:|FATAL|IMMINENT SIGKILL' logs/iter15_v*enc_*.log | head -20
```

🎯 Pass criteria per cell:
- ✅ 3 ckpts written at the correct `*_encoder` path (m09a1 uses iter15 fallback if probe disabled; m09c1 always saves)
- 📉 `train_loss` decreasing across epochs (real backward — unlike head-only's trivial-loss possibility)
- ⛔ NO `FATAL` lines, NO `IMMINENT SIGKILL`
- 📊 `[lambda_reg]` drift-control logs visible (m09a1) OR `Stage: stage{1,2,3}_*` progression logs (m09c1)

---

## 🏆 Phase 6 — POC head-only + encoder-update (96 GB Blackwell, ~3-4 hr, ~$3 total)

> 🧹 **REMEMBER**: `rm -rf outputs/sanity` before Phase 6 to free disk space (~73 GB)

📊 3 head-only 🧊 + 3 encoder-update 🔓 cells. On 96 GB all 6 run sequentially in ~3-4 hr wall (head-only ~30-45 min each, encoder-update ~30-60 min each); parallel-in-3 cuts to ~2 hr.

```bash
# ── 🧊 HEAD-ONLY POC (3 cells — frozen encoder, head trains only) ───────────
CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_head --POC 2>&1 \
    | tee logs/iter15_poc_m09a2_pretrain_head_$(date +%Y%m%d_%H%M%S).log

CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_3stage_DI_head --POC 2>&1 \
    | tee logs/iter15_poc_m09c2_3stage_DI_head_$(date +%Y%m%d_%H%M%S).log

CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_noDI_head --POC 2>&1 \
    | tee logs/iter15_poc_m09c2_noDI_head_$(date +%Y%m%d_%H%M%S).log

# ── 🔓 ENCODER-UPDATE POC (3 cells — paired-Δ counterpart for paper claim) ──
CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_encoder --POC 2>&1 \
    | tee logs/iter15_poc_m09a1_pretrain_encoder_$(date +%Y%m%d_%H%M%S).log

CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_3stage_DI_encoder --POC 2>&1 \
    | tee logs/iter15_poc_m09c1_3stage_DI_encoder_$(date +%Y%m%d_%H%M%S).log

CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_noDI_encoder --POC 2>&1 \
    | tee logs/iter15_poc_m09c1_noDI_encoder_$(date +%Y%m%d_%H%M%S).log
```

🎯 Pass criteria per cell — 🧊 head-only:
- ✅ `student_encoder.pt` bit-identical to Meta (rerun V4 against POC outputs)
- ✅ `m09{a,c}_ckpt_best.pt` has `motion_aux_head_state_dict` + `best_val_loss`
- 📉 `train_loss` DECREASING across epochs (early: high; late: low)
- 📊 Final `motion_aux` val_loss < frozen baseline anchor (~0.47)
- ⛔ NO `IMMINENT SIGKILL`

🎯 Pass criteria per cell — 🔓 encoder-update:
- 🧠 `student_encoder.pt` ≠ Meta (encoder DID drift — measured via V4 helper, expected > 1e-4 rel_l2)
- 📊 `lambda_reg` drift-control prints reasonable values (m09a1)
- 🔬 Surgery stages emit in order (stage1 → stage2 → stage3 for 3stage_DI; stage1 → stage2 for noDI)

---

## 🏆 Phase 6 post-POC — Δ5 paper claim probe (~30 min)

```bash
# 🎬 Run the full 13-stage eval pipeline on POC ckpts
CACHE_POLICY_ALL=1 ./scripts/run_eval.sh --POC 2>&1 \
    | tee logs/iter15_post_poc_eval_$(date +%Y%m%d_%H%M%S).log

# 🎯 Δ5 = surgery_3stage_DI_encoder − surgery_3stage_DI_head  (paper claim)
#   🟢 |Δ5| < 0.01 + CI contains 0 → head-only WINS (1/40× GPU savings unlock)
#   🔵 Δ5 > 0.01                    → encoder-update wins by margin
#   🔴 Δ5 < -0.01                   → head-only outperforms (investigate)
python -c "
import json
d = json.load(open('outputs/poc/probe_action/probe_paired_delta.json'))['iter14_paper_deltas']
d5 = d.get('delta_5_surgical_vs_surgical_head')
if not d5 or d5.get('skipped'):
    print('Δ5 not yet available — POC cells incomplete')
else:
    print(f'Δ5 mean: {d5[\"delta_mean\"]:+.4f}')
    print(f'Δ5 95% CI: [{d5[\"delta_ci_lo\"]:+.4f}, {d5[\"delta_ci_hi\"]:+.4f}]')
    print(f'p_value:   {d5[\"p_value\"]:.4f}')
    print(f'interpretation: {d5[\"interpretation\"]}')"
```

---

## 🛑 HARD STOP — triage guide (if any Phase 5b / Phase 6 cell fails)

```bash
# 🐛 Common failure modes:
#   🚨 OOM at BS=1                  → 24 GB ceiling reached; check VRAM math
#   🚨 assert_encoder_frozen FATAL  → freeze wiring broken (m09a2/c2 build_model)
#   🚨 motion_aux REQUIRED FATAL    → action_labels.json missing
#   🚨 factor_manifest.json missing → run scripts/run_factor_prep.sh first
#   🚨 Producer stalled 10 min      → reduce factor_streaming.num_workers in pipeline.yaml
#   🚨 encoder NOT drifted (V4 fail on m09a1/c1) → drift-control or LR misconfigured

# 📋 Health sweep across all Phase 5b/6 logs:
echo "📊 Phase 6 review:"
for log in logs/iter15_v*enc_*.log logs/iter15_poc_*.log; do
    [ -f "$log" ] || continue
    FATAL=$(grep -c 'FATAL' "$log")
    OOM=$(grep -c 'IMMINENT\|OutOfMemoryError' "$log")
    DONE=$(grep -c 'DONE\|COMPLETE\|Saved' "$log")
    echo "  $log: FATAL=$FATAL  OOM=$OOM  DONE=$DONE"
done
```

---

## 📊 ETA monitoring (optional, second tmux pane)

```bash
# ⚡ Live GPU + memory + tqdm watcher (refresh every 5 sec)
watch -n 5 '
echo "=== 🔥 GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
echo ""
echo "=== 🧠 cgroup memory ==="
cat /sys/fs/cgroup/memory/memory.current 2>/dev/null \
    | awk "{printf \"%.1f GB used\\n\", \$1/1024/1024/1024}" \
    || cat /sys/fs/cgroup/memory.current | awk "{printf \"%.1f GB used\\n\", \$1/1024/1024/1024}"
echo ""
echo "=== 📈 latest tqdm line ==="
ls -t logs/iter15_*.log 2>/dev/null | head -1 | xargs tail -1 2>/dev/null
'
```

---

## 🧹 Cleanup / 📤 HF upload (post-Phase-6 only — SANITY artifacts are throw-away)

```bash
# 🔬 Verify all 6 POC training cells produced their ckpts:
find outputs/poc/m09a_pretrain_head outputs/poc/m09a_pretrain_encoder \
     outputs/poc/m09c_surgery_3stage_DI_head outputs/poc/m09c_surgery_3stage_DI_encoder \
     outputs/poc/m09c_surgery_noDI_head outputs/poc/m09c_surgery_noDI_encoder \
    -name 'student_encoder.pt' -o -name 'm09a_ckpt_best.pt' \
    -o -name 'm09c_ckpt_best.pt' -o -name 'motion_aux_head.pt' \
    | sort

# 📤 Upload POC outputs to HF (skip SANITY — code-correctness tier, not paper-grade)
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload outputs/poc 2>&1 \
    | tee logs/upload_outputs_poc_$(date +%Y%m%d_%H%M%S).log
```
