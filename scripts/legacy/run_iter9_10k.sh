#!/usr/bin/env bash
# Thin wrapper: chains iter/iter9/runbook.md Steps A→F verbatim.
# Designed for overnight unattended execution.
#
# USAGE:
#   tmux new -s terminal1
#   ./scripts/run_iter9_10k.sh 2>&1 | tee logs/iter9_10k_overnight_v11.log
#   # detach: Ctrl+B then D · reattach: tmux attach -t iter9
#
# iter10 additions (2026-04-21):
#   ./scripts/run_iter9_10k.sh --train-config configs/train/ch11_surgery_v15a.yaml
#   (default train-config = configs/train/ch11_surgery.yaml; Step C only — m10/m11
#    still use their default yaml, which is fine since they read only the
#    factor_datasets / interaction_mining sections that don't change per variant.)
set -euo pipefail

cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs

# Default Step C train-config. Override via --train-config <path>.
TRAIN_CONFIG="configs/train/ch11_surgery.yaml"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-config) TRAIN_CONFIG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done
echo "Step C train-config: $TRAIN_CONFIG"

T0=$(date +%s)
stamp() { echo -e "\n═══ $(date '+%H:%M:%S') · $1 ═══"; }

# ── Step A: m10 Grounded-SAM (10K, ~10h GPU) ────────────────────────
stamp "Step A — m10 10K"
# rm -rf outputs/full/m10_sam_segment/ # VERY COSTLY process, so be very careful about deleting m10's artifacts
python -u src/m10_sam_segment.py --FULL \
    --subset data/subset_10k.json \
    --local-data data/subset_10k_local --no-wandb \
    2>&1 | tee logs/m10_10k_v1.log

stamp "Verify A — m10 summary + disk"
cat outputs/full/m10_sam_segment/summary.json | python3 -m json.tool
du -sh outputs/full/m10_sam_segment/
python3 -c "
import json
s = json.load(open('outputs/full/m10_sam_segment/summary.json'))
qg = s.get('quality_gate', 'UNKNOWN')
assert qg == 'PASS', f'Step A quality_gate={qg} (expected PASS) — halting'
print(f'✅ quality_gate=PASS · n_agents={s.get(\"n_total_agents\")} · clips_with_agents={s.get(\"clips_with_agents_pct\", 0)*100:.0f}%')
"

# ── Step B: m11 factor datasets --streaming (~10 min CPU) ───────────
stamp "Step B — m11 --streaming"
# rm -rf outputs/full/m11_factor_datasets/
python -u src/m11_factor_datasets.py --FULL --streaming \
    --subset data/subset_10k.json \
    --local-data data/subset_10k_local --no-wandb \
    2>&1 | tee logs/m11_10k_v1.log

stamp "Verify B — m11 manifest + verify-clip .npy count"
python3 -c "
import json
m = json.load(open('outputs/full/m11_factor_datasets/factor_manifest.json'))
s = json.load(open('data/subset_10k.json'))
expected = len(s['clip_keys'])  # dynamic: subset_10k is authoritative (9566 after val_1k filter)
print(f'clips in manifest: {len(m)} (expected: {expected} from data/subset_10k.json)')
tubes = [v.get('n_interaction_tubes', 0) for v in m.values()]
print(f'D_I tubes: {sum(tubes)} (expect 0 — interaction_mining disabled)')
dl = sum(1 for v in m.values() if v['has_D_L'])
da = sum(1 for v in m.values() if v['has_D_A'])
print(f'D_L eligible: {dl}/{len(m)}; D_A eligible: {da}/{len(m)}')
assert len(m) == expected, f'manifest has {len(m)} clips (expected {expected} from subset_10k.json)'
assert sum(tubes) == 0, f'D_I tubes should be 0 under iter9 2-stage recipe'
print('✅ manifest OK')
"
ls outputs/full/m11_factor_datasets/D_L/ | wc -l   # ~100 verify clips only
du -sh outputs/full/m11_factor_datasets/

# ── Step C: m09c Surgery — probe on val_500 (~2.5h GPU) ─────────────
stamp "Step C — m09c Surgery (probe=val_500)"
# iter11: no shell-level deletion — m09c owns its output dir lifecycle via --cache-policy.
python -u src/m09c_surgery.py --FULL \
    --subset data/subset_10k.json \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config "$TRAIN_CONFIG" \
    --factor-dir outputs/full/m11_factor_datasets/ \
    --local-data data/subset_10k_local \
    --probe-subset data/val_500.json \
    --probe-local-data data/val_1k_local \
    --probe-tags data/val_1k_local/tags.json \
    --no-wandb \
    2>&1 | tee logs/m09c_10k_v1.log

stamp "Verify C — student_encoder.pt + training summary"
ls -lh outputs/full/m09c_surgery/student_encoder.pt outputs/full/m09c_surgery/val_split.json
python3 -c "
import json
s = json.load(open('outputs/full/m09c_surgery/training_summary.json'))
print('split:     ', s['train_val_split'])
print('best_ckpt: ', s['best_ckpt'])
print('early_stop:', s['early_stop'])
print('BWT:       ', s['probe_trajectory_stats']['bwt_prec_at_k'])
assert s['train_val_split']['val'] == 500, f'val split should be 500 (val_500), got {s[\"train_val_split\"][\"val\"]}'
print('✅ training summary OK')
"

# ── Step D: m05 frozen embed on test_500 (~11 min GPU) ──────────────
stamp "Step D — m05 frozen on test_500"
# iter11: no shell-level deletion — m05 output_guard handles cached-skip vs recompute.
python -u src/m05_vjepa_embed.py --FULL \
    --subset data/test_500.json \
    --model-config configs/model/vjepa2_1.yaml \
    --encoder vjepa_2_1_frozen \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_test500_frozen_v1.log

stamp "Verify D — frozen embedding shape"
python3 -c "
import numpy as np
from pathlib import Path
npy = next(Path('outputs/full/m05_vjepa_embed').glob('embeddings_vjepa_2_1_frozen*.npy'))
a = np.load(npy, mmap_mode='r')
print(f'frozen {npy.name}: shape={a.shape} dtype={a.dtype}')
assert a.shape == (500, 1664), f'expected (500, 1664), got {a.shape}'
print('✅ frozen shape OK')
"

# ── Step E: m05 surgical embed on test_500 (~25 min GPU) ────────────
stamp "Step E — m05 surgical on test_500"
# iter11: no shell-level deletion of surgical artifacts; m05's cache-policy gate owns cleanup.
python -u src/m05_vjepa_embed.py --FULL \
    --subset data/test_500.json \
    --model-config configs/model/vjepa2_1.yaml \
    --model outputs/full/m09c_surgery/student_encoder.pt \
    --encoder vjepa_2_1_surgical \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_test500_surgical_v1.log

stamp "Verify E — surgical embedding shape"
python3 -c "
import numpy as np
from pathlib import Path
npy = next(Path('outputs/full/m05_vjepa_embed').glob('embeddings_vjepa_2_1_surgical*.npy'))
a = np.load(npy, mmap_mode='r')
print(f'surgical {npy.name}: shape={a.shape} dtype={a.dtype}')
assert a.shape == (500, 1664), f'expected (500, 1664), got {a.shape}'
print('✅ surgical shape OK')
"

# ── Step F: m06 Prec@K gate on test_500 (~2 min GPU) ────────────────
stamp "Step F — m06 gate (frozen + surgical) on test_500"
# iter11: no shell-level deletion — m06 overwrites atomically; output_guard handles skip.
python -u src/m06_faiss_metrics.py --FULL \
    --subset data/test_500.json --encoder vjepa_2_1_frozen --no-wandb \
    2>&1 | tee logs/m06_test500_frozen.log

python -u src/m06_faiss_metrics.py --FULL \
    --subset data/test_500.json --encoder vjepa_2_1_surgical --no-wandb \
    2>&1 | tee logs/m06_test500_surgical.log

stamp "Verify F — both m06 metrics JSONs present"
ls -lh outputs/full/m06_faiss_metrics/m06_metrics_vjepa_2_1_{frozen,surgical}.json

# ── Step G: m08b multi-encoder comparison (radar + heatmap + ablation plots) ──
stamp "Step G — m08b compare (frozen vs surgical plots + LaTeX table)"
python -u src/m08b_compare.py --FULL \
    --subset data/test_500.json \
    --encoders vjepa_2_1_frozen,vjepa_2_1_surgical \
    --no-wandb \
    2>&1 | tee logs/m08b_test500.log

stamp "Verify G — m08b plot set present"
# 2026-04-21 fix: radar + adaptation_ablation are OPTIONAL — radar skipped when
# n_encoders<3, adaptation_ablation skipped when vjepa_shuffled absent. Only the
# 3 always-present plots must exist (fail-hard if missing); optional plots are
# listed for visibility with explicit "skipped" note when absent.
ls -lh outputs/full/m08b_compare/m08b_{encoder_comparison,heatmap,spatial_temporal_bar}.png
for opt in radar adaptation_ablation; do
    opt_path="outputs/full/m08b_compare/m08b_${opt}.png"
    if [ -f "$opt_path" ]; then
        ls -lh "$opt_path"
    else
        echo "  [optional] m08b_${opt}.png SKIPPED (expected when n_encoders<3 or vjepa_shuffled absent)"
    fi
done

# ── 🏆 Decision gate printout ───────────────────────────────────────
# m06 JSON schema (confirmed 2026-04-20): m['easy']['prec_at_k'] is a float,
# 95% CI bands live in m['easy']['ci']['prec_at_k'] = {ci_half, ci_lo, ci_hi}.
# Older iter8 schema keyed under 'precision_at_k' — renamed, hence #74 fix.
stamp "🏆 DECISION GATE (train=subset_10k / test=test_500, N=500, CI ~±2.4 pp)"
python3 -c "
import json
rows = []
for name in ['frozen', 'surgical']:
    m = json.load(open(f'outputs/full/m06_faiss_metrics/m06_metrics_vjepa_2_1_{name}.json'))
    e = m['easy']
    prec_mean = e['prec_at_k']
    prec_ci = e['ci']['prec_at_k']
    rows.append((name, prec_mean, prec_ci['ci_half'], prec_ci['ci_lo'], prec_ci['ci_hi']))
    print(f'{name:12s}: Prec@K = {prec_mean:.2f}% +/- {prec_ci[\"ci_half\"]:.2f}  '
          f'(CI: [{prec_ci[\"ci_lo\"]:.2f}, {prec_ci[\"ci_hi\"]:.2f}])')
delta = rows[1][1] - rows[0][1]
overlap = not (rows[1][3] > rows[0][4] or rows[0][3] > rows[1][4])
print(f'\nΔ = {delta:+.2f} pp | CIs overlap: {overlap}')
if delta >= 3.0 and not overlap:   print('✅ PASS — proceed to Step G / H.1 (50K)')
elif delta >= 1.0:                   print('🟡 MARGINAL — apply BWT Option B (λ=50), re-run C/D/E/F')
elif delta >= 0.0:                   print('🟡 SATURATED — 10K is publishable tier, skip ladder')
else:                                 print('❌ FAIL — apply BWT Options B → C in plan_TODO.md')
"

DUR=$(( $(date +%s) - T0 ))
echo -e "\n⏱️  Total wall: $(( DUR / 3600 ))h $(( (DUR % 3600) / 60 ))m"
df -h /workspace
