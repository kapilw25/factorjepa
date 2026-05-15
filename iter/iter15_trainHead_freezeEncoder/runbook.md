# 🚀 iter15 — Runbook: Phase 5 SANITY + Phase 6 POC

Sibling docs: [`plan_trainHead_scaleBackbone_curriculum.md`](./plan_trainHead_scaleBackbone_curriculum.md) (design + status) · [`planCODE_html.md`](./planCODE_html.md) (HTML refactor)

---

## Pre-flight (~30 sec, CPU)

```bash
# === VENV ACTIVATION — must be venv_walkindia (NOT /venv/main) ===
# Setup script (scripts/setup_env_gpu.sh) creates venv_walkindia with numpy/torch/
# cuML/FAISS/Flash-Attn. Fresh shells default to /venv/main which lacks numpy →
# every command below fails with ModuleNotFoundError if this is skipped.
source venv_walkindia/bin/activate
python -c "import numpy, torch; print(f'numpy={numpy.__version__}  torch={torch.__version__}  cuda={torch.cuda.is_available()}')" \
    || { echo "FATAL: venv_walkindia not viable"; exit 1; }

# === CGROUP ENVELOPE — fail loud if cap < 48 GB (head-only needs ~10 GB GPU + ~30 GB host RAM peak) ===
cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null \
    | awk '{printf "cgroup memory cap: %.1f GB\n", $1/1024/1024/1024}' \
    || cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/pids/pids.max 2>/dev/null \
    || cat /sys/fs/cgroup/pids.max

# === GPU sanity ===
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# === CPU + disk ===
nproc && free -h | head -2 && df -h /workspace 2>/dev/null | tail -1

# === Confirm 23-D motion features are on disk (Phase 3 prereq) ===
# iter15 (2026-05-15): m04d outputs now land under
# <local_data>/m04d_motion_features/ (was: <local_data>/ root) so the whole
# subdir — including .m04d_checkpoint.npz — rides on hf_outputs.py upload-data.
python -c "
import numpy as np
f = np.load('data/eval_10k_local/m04d_motion_features/motion_features.npy')
assert f.shape == (9297, 23), f'expected (9297, 23), got {f.shape} — rerun Phase 3 m04d'
print(f'motion_features.npy: {f.shape}  vec[13] range: [{f[:,13].min():.3f}, {f[:,13].max():.3f}]')"

# === Confirm action_labels.json was regenerated on 23-D taxonomy ===
# If file MISSING or older than motion_features.npy → regenerate (CPU, ~5 sec):
#   python -u src/probe_action.py --SANITY --stage labels \
#       --eval-subset data/eval_10k_local/eval_10k.json \
#       --motion-features data/eval_10k_local/m04d_motion_features/motion_features.npy \
#       --output-root outputs/sanity/probe_action
ls -la outputs/sanity/probe_action/action_labels.json 2>/dev/null \
    || echo "MISSING: regenerate with probe_action --stage labels (see comment above)"
```

---

## Phase 5 V0 — CPU preflight (~30 sec, no GPU)

```bash
# === Yaml parse + extends-chain resolution ===
PYTHONPATH=src python -c "
import sys; sys.path.insert(0, 'src')
from utils.config import load_merged_config
for y in ['configs/train/pretrain_head.yaml',
          'configs/train/surgery_3stage_DI_head.yaml',
          'configs/train/surgery_2stage_noDI_head.yaml']:
    cfg = load_merged_config('configs/model/vjepa2_1.yaml', y)
    print(f'OK {y}')"

# === merge_config_with_args contract enforcement (head-only forces) ===
# m09a2 forces: layer_freeze.freeze_below=48 + drift_control.enabled=False + weight_jepa=0.0
# m09c2 forces: stages[0].unfreeze_below=0.0 + factor_streaming.enabled=True
# (full V0 preflight assertion suite ran 2026-05-14 — see commit 63a9ffb)

# === ITER14_DELTAS has 7 entries (Δ1-Δ7) ===
PYTHONPATH=src python -c "
import re
src = open('src/probe_action.py').read()
m = re.search(r'ITER14_DELTAS = \[(.*?)\n    \]', src, re.DOTALL)
n = m.group(1).count('(\"delta_') if m else 0
assert n == 7, f'expected 7, got {n}'
print(f'ITER14_DELTAS: {n} entries')"
```

---

## Phase 5 V2 — m09a2 SANITY (~5 min, ~$0.02)

```bash
# === Head-only continual SSL on FROZEN encoder + FROZEN predictor ===
# Expected log signatures (in order):
#   1. [m09a2] cgroup envelope: memory: X.X GB / X.X GB
#   2. [m09a2-oom-watchdog] started (warn=80%, crit=90%, imminent=97%, ...)
#   3. [m09a2 STRICT HEAD-ONLY] encoder FROZEN: 0 trainable block params (asserted)
#   4. Predictor: ~600M params (FROZEN)
#   5. Trainable params: motion_aux head = ~432,XXX
#   6. SANITY → 1 epoch wall ~2-3 min → 3 ckpts produced
./scripts/run_train.sh pretrain_head --SANITY 2>&1 \
    | tee logs/iter15_v2_m09a2_sanity_$(date +%Y%m%d_%H%M%S).log

# === Post-check: 3 ckpts produced ===
ls -la outputs/sanity/m09a_pretrain_head/{student_encoder.pt,m09a_ckpt_best.pt,motion_aux_head.pt}

# === Pass criteria checked via grep ===
# - all 3 ckpts exist (above)
# - log contains [m09a2 STRICT HEAD-ONLY] banner
# - log contains 'Trainable params: motion_aux head = 432' (~432K head params)
# - NO 'FATAL' lines
# - NO 'IMMINENT SIGKILL' from cgroup watchdog
grep -E 'STRICT HEAD-ONLY|FATAL|IMMINENT SIGKILL|Trainable params: motion_aux head' \
    logs/iter15_v2_m09a2_sanity_*.log | tail -5
```

---

## Phase 5 V3 — m09c2 SANITY both variants (~10 min total)

```bash
# === V3a: 3stage_DI head-only (factor-aug data, mode_mixture {L:0.15, A:0.15, I:0.70}) ===
./scripts/run_train.sh surgery_3stage_DI_head --SANITY 2>&1 \
    | tee logs/iter15_v3a_m09c2_3stage_DI_head_sanity_$(date +%Y%m%d_%H%M%S).log

# === V3b: noDI head-only ({L:0.50, A:0.50, I:0.00}) ===
./scripts/run_train.sh surgery_noDI_head --SANITY 2>&1 \
    | tee logs/iter15_v3b_m09c2_noDI_head_sanity_$(date +%Y%m%d_%H%M%S).log

# === Post-check: both variants produce ckpts in their own subdirs ===
ls -la outputs/sanity/m09c_surgery_3stage_DI_head/3stage_DI_head/{student_encoder.pt,m09c_ckpt_best.pt,motion_aux_head.pt}
ls -la outputs/sanity/m09c_surgery_noDI_head/noDI_head/{student_encoder.pt,m09c_ckpt_best.pt,motion_aux_head.pt}

# === Pass criteria:
#   - both variants produce 3 ckpts
#   - logs show 'Stage: stage0_head_only_*' (single stage, no Stage 1/2/3 transitions)
#   - mode_mixture differs between the two variants
grep -E 'Stage: stage0_head_only|mode_mixture|STRICT HEAD-ONLY' \
    logs/iter15_v3a_*.log logs/iter15_v3b_*.log | head -10
```

---

## Phase 5 V4 — Encoder invariance check (~10 sec, CPU)

```bash
# === CRITICAL: assert all 3 head-only variants kept the encoder BIT-IDENTICAL to Meta init ===
# This is the contract validation. If any block param drifted, the freeze wiring is broken.
python -c "
import torch
from pathlib import Path
meta_full = torch.load('checkpoints/vjepa2_1_vitG_384.pt', map_location='cpu', weights_only=False)
meta = meta_full.get('target_encoder', meta_full.get('encoder', meta_full))
meta = {k.replace('module.', '').replace('backbone.', ''): v for k, v in meta.items()}
for variant_path in [
    'outputs/sanity/m09a_pretrain_head/student_encoder.pt',
    'outputs/sanity/m09c_surgery_3stage_DI_head/3stage_DI_head/student_encoder.pt',
    'outputs/sanity/m09c_surgery_noDI_head/noDI_head/student_encoder.pt',
]:
    p = Path(variant_path)
    if not p.exists():
        print(f'SKIP {variant_path} — not produced (run V2/V3 first)')
        continue
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    sd = ckpt.get('student_state_dict', ckpt)
    n_match = n_total = 0
    for k, v in sd.items():
        if k.startswith('blocks.') and k in meta and v.shape == meta[k].shape:
            n_total += 1
            if torch.allclose(v, meta[k]):
                n_match += 1
    print(f'{p.parts[-2]}/{p.name}: {n_match}/{n_total} block params bit-identical to Meta')
    assert n_match == n_total, f'FATAL: {variant_path} has {n_total - n_match} drifted blocks'
print('PASS: all 3 head-only variants have bit-identical encoder weights to Meta')"
```

---

## Phase 5 V5 — probe_future_regress SANITY (~5 min)

```bash
# === Train a regressor head on FROZEN encoder ctx/tgt features ===
# Uses linear arch + raw data source (simplest cell). MLP variants tested in Phase 6.
python -u src/probe_future_regress.py --SANITY \
    --stage forward \
    --variant vjepa_2_1_frozen \
    --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \
    --data-source raw \
    --regressor-arch linear \
    --action-probe-root outputs/sanity/probe_action \
    --local-data data/eval_10k_local \
    --output-root outputs/sanity/probe_future_regress \
    --cache-policy 2 2>&1 \
    | tee logs/iter15_v5_probe_future_regress_sanity_$(date +%Y%m%d_%H%M%S).log

# === Post-check: 4 outputs produced ===
ls -la outputs/sanity/probe_future_regress/vjepa_2_1_frozen/{per_clip_regressor_l1.npy,clip_keys.npy,aggregate_regressor_l1.json,regressor.pt}

# === Sanity check on aggregate JSON ===
cat outputs/sanity/probe_future_regress/vjepa_2_1_frozen/aggregate_regressor_l1.json | head -15
```

---

## Phase 5 V6 — Full eval pipeline SANITY (~20 min)

```bash
# === Run all 8 encoders through run_eval.sh ===
# Validates: probe_action --stage features + train + eval + paired_delta (Δ1-Δ7),
# probe_future_mse (Stage 8), probe_future_regress (Stage 9b).
# Δ4-Δ7 will be SKIPPED (skipped=true) for any encoder not yet trained — but the
# JSON should still emit those keys.
./scripts/run_eval.sh --sanity 2>&1 \
    | tee logs/iter15_v6_full_eval_sanity_$(date +%Y%m%d_%H%M%S).log

# === Pass criteria ===
# - probe_paired_delta.json has 7 delta_* keys (Δ1-Δ7), some may have skipped=true
# - probe_future_regress_per_variant.json emits for available variants
# - zero STAGE FAIL across all 7 variants × all 10+ stages
grep -E 'STAGE.*FAIL|ALL STAGES' logs/iter15_v6_*.log | tail -5
python -c "
import json
d = json.load(open('outputs/sanity/probe_action/probe_paired_delta.json'))
deltas = [k for k in d if k.startswith('delta_')]
print(f'Found {len(deltas)} deltas:', deltas)
assert len(deltas) == 7, f'expected 7, got {len(deltas)}'"
```

---

## HARD STOP — review V0-V6 before Phase 6

```bash
# === If ANY V step FAILED on 24 GB → investigate before Phase 6 ===
# Triage common failures:
#   OOM        → AdaptiveBatchSizer should have shrunk BS; if it OOM'd at BS=1,
#                that's a 24 GB ceiling we did not predict. Check VRAM math.
#                Grep the log for 'OOM' or 'IMMINENT SIGKILL'.
#   FATAL: assert_encoder_frozen — N block params trainable
#                → freeze wiring broken. Check m09a2/c2 build_model().
#   FATAL: motion_aux REQUIRED
#                → action_labels.json missing or motion_aux disabled in yaml.
#                  Regenerate labels: probe_action --stage labels.
#   FATAL: factor_manifest.json missing (m09c2 only)
#                → run scripts/run_factor_prep.sh first.
#   Producer stalled 10 min
#                → DataLoader workers crashed; reduce num_workers in
#                  pipeline.yaml factor_streaming.num_workers.

echo "Phase 5 review checklist:"
for log in logs/iter15_v2_*.log logs/iter15_v3a_*.log logs/iter15_v3b_*.log \
           logs/iter15_v5_*.log logs/iter15_v6_*.log; do
    [ -f "$log" ] || continue
    FATAL=$(grep -c 'FATAL' "$log")
    OOM=$(grep -c 'IMMINENT\|OutOfMemoryError' "$log")
    DONE=$(grep -c 'DONE\|COMPLETE\|Saved' "$log")
    echo "  $log: FATAL=$FATAL  OOM_signal=$OOM  DONE_lines=$DONE"
done
```

---

## Phase 6 — POC head-only training (~24 hr GPU total, ~$5)

```bash
# === Three independent POC cells — run serially or in 3 tmux sessions ===
# Each is ~6-8 hr on 24 GB Pro 4000 at $0.20/hr → ~$5 total.

# === Cell 1: m09a2 head-only continual pretrain ===
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_head --POC 2>&1 \
    | tee logs/iter15_poc_m09a2_pretrain_head_$(date +%Y%m%d_%H%M%S).log

# === Cell 2: m09c2 head-only surgery (3stage_DI) ===
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_head --POC 2>&1 \
    | tee logs/iter15_poc_m09c2_3stage_DI_head_$(date +%Y%m%d_%H%M%S).log

# === Cell 3: m09c2 head-only surgery (noDI) ===
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_head --POC 2>&1 \
    | tee logs/iter15_poc_m09c2_noDI_head_$(date +%Y%m%d_%H%M%S).log

# === Pass criteria per cell ===
# - student_encoder.pt bit-identical to Meta (rerun V4 check on POC outputs)
# - m09c_ckpt_best.pt has motion_aux_head_state_dict + best_val_loss
# - log shows train_loss DECREASING across epochs (early epochs: high; late: low)
# - Final motion_aux val_loss < frozen baseline anchor (~0.47)
# - NO IMMINENT SIGKILL lines in any log
```

---

## Phase 6 post-POC — refresh probe + ckpt sweep (~30 min, ~$0.10)

```bash
# === After all 3 POC cells finish, re-run probe pipeline on the new ckpts ===
# This produces Δ4-Δ7 with REAL paired-BCa CIs (not skipped).
CACHE_POLICY_ALL=1 ./scripts/run_eval.sh --POC 2>&1 \
    | tee logs/iter15_post_poc_eval_$(date +%Y%m%d_%H%M%S).log

# === Inspect the key paper claim — Δ5 = surgery_3stage_DI − surgery_3stage_DI_head ===
# If |Δ5| < 0.01 with 95% CI containing 0 → head-only WINS (1/40× GPU savings unlock).
# If Δ5 > 0.01 → encoder-update still wins by margin; head-only is a worse approximation.
# If Δ5 < -0.01 → head-only actually OUTPERFORMS (surprising; investigate).
python -c "
import json
d = json.load(open('outputs/poc/probe_action/probe_paired_delta.json'))
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

## ETA monitoring (optional, run in a second tmux pane)

```bash
# === Watch GPU + memory + tqdm progress every 5 sec ===
watch -n 5 '
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv
echo ""
echo "=== cgroup memory ==="
cat /sys/fs/cgroup/memory/memory.current 2>/dev/null \
    | awk "{printf \"%.1f GB used\\n\", \$1/1024/1024/1024}" \
    || cat /sys/fs/cgroup/memory.current | awk "{printf \"%.1f GB used\\n\", \$1/1024/1024/1024}"
echo ""
echo "=== latest tqdm line ==="
ls -t logs/iter15_*.log 2>/dev/null | head -1 | xargs tail -1 2>/dev/null
'
```

---

## Cleanup / handoff

```bash
# === Verify outputs are durable + ready for HF upload ===
find outputs/poc/m09a_pretrain_head outputs/poc/m09c_surgery_3stage_DI_head outputs/poc/m09c_surgery_noDI_head \
    -name 'student_encoder.pt' -o -name 'm09a_ckpt_best.pt' -o -name 'm09c_ckpt_best.pt' -o -name 'motion_aux_head.pt' \
    | sort

# === Upload to HF for the paper (optional) ===
HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload outputs/poc 2>&1 \
    | tee logs/upload_outputs_poc_$(date +%Y%m%d_%H%M%S).log
```
