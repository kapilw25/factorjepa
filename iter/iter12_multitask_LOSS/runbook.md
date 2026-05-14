# 🚀 FactorJEPA Runbook — iter12 multi-task LOSS (terminal commands only)

> 2-variant multi-task training (`surgery_2stage_noDI_multitask` + `surgery_3stage_DI_multitask`) on the SAME `data/ultra_hard_3066_*` splits as iter11 v3, REUSING m10/m11 factor outputs + frozen-baseline cache. Loss = `α·JEPA + β·InfoNCE + γ·TCC` with α/β/γ LEARNED via Kendall 2018 Uncertainty Weighting.
> Architecture / decisions / errors → `plan_training.md` / `plan_TODO.md` / `errors_N_fixes.md`.

---

## 0. Pre-flight (1-shot verify — multitask code wired + iter11 cache present)

```bash
cd /workspace/factorjepa
source venv_walkindia/bin/activate

# (a) iter11 v3 prerequisites cached (data, m10/m11, frozen baseline, 3 standard variants)
ls -d data/ultra_hard_3066_local data/ultra_hard_3066_{train,val,eval}.json
ls outputs/full/m10_sam_segment/segments.json \
   outputs/full/m11_factor_datasets/factor_manifest.json \
   outputs/full/m05_vjepa_embed/embeddings_vjepa_2_1_frozen.npy \
   outputs/full/m06_faiss_metrics/per_clip_vjepa_2_1_frozen_easy.npz

# (b) Both multitask yamls exist + UW=true
for y in configs/train/surgery_{2stage_noDI,3stage_DI}_multitask.yaml; do
    [ -f "$y" ] && echo "✓ $y"
done
python3 -c "
import sys; sys.path.insert(0, 'src')
from utils.config import load_merged_config
from utils.training import UncertaintyWeights, compute_multitask_loss  # import smoke
for y in ['configs/train/surgery_2stage_noDI_multitask.yaml',
          'configs/train/surgery_3stage_DI_multitask.yaml']:
    cfg = load_merged_config('configs/model/vjepa2_1.yaml', y)
    assert cfg['optimization']['loss']['uncertainty_weighting'], f'{y}: UW not enabled'
    print(f'  ✓ {y.split(\"/\")[-1]}: UW active, encoder={cfg[\"data\"][\"adapted_encoder\"]}')
print('  ✓ UncertaintyWeights + compute_multitask_loss importable')
"
echo "✓ Pre-flight passed — safe to launch Phase 1"
```

---

## 1. Phase 1 — SANITY (1-step code-path validation, ~5 min each) ⭐ STRONGLY RECOMMENDED FIRST

> First training run with the new UW path on 2B-param V-JEPA — fail-loud at 5 min is much cheaper than fail at hour 6 of the FULL run. SANITY catches NaN, OOM, missing UW banner, missing CSV columns, stuck UW weights, and missing student export. Per `src/CLAUDE.md`: "SANITY validates code correctness (no crashes), NOT model performance."

> **Note** — `train_subset`, `val_subset`, `eval_subset`, and their `_local_data` paths
> all come from the YAML's `data:` block (inherited via
> `surgery_*_multitask.yaml → surgery_*.yaml → surgery_base.yaml → base_optimization.yaml`).
> m09c's CLI only exposes `--subset` / `--local-data` (general overrides), `--probe-subset`,
> `--probe-local-data`, `--probe-tags`, `--no-probe`, `--factor-dir`, `--output-dir`,
> `--batch-size`, `--max-epochs`, `--cache-policy`, `--no-wandb`, and the mode flags.
> Do NOT pass `--train-subset` / `--eval-subset` / `--eval-local-data` — argparse
> will reject them as unrecognized.

```bash
tmux new -s iter12_sanity

# Variant A — surgery_2stage_noDI multitask (SANITY, ~5 min)
venv_walkindia/bin/python3 -u src/m09c1_surgery_encoder.py --SANITY \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/surgery_2stage_noDI_multitask.yaml \
    --factor-dir outputs/full/m11_factor_datasets \
    --output-dir outputs/sanity/surgery_2stage_noDI_multitask \
    --cache-policy 2 --no-wandb \
    2>&1 | tee logs/sanity_2stage_noDI_multitask.log

# Variant B — surgery_3stage_DI multitask (SANITY, ~5 min)
venv_walkindia/bin/python3 -u src/m09c1_surgery_encoder.py --SANITY \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/surgery_3stage_DI_multitask.yaml \
    --factor-dir outputs/full/m11_factor_datasets \
    --output-dir outputs/sanity/surgery_3stage_DI_multitask \
    --cache-policy 2 --no-wandb \
    2>&1 | tee logs/sanity_3stage_DI_multitask_v1.log
```

**Quick verify after SANITY** (catches all 6 failure modes — missing log file, no crash, NaN, missing UW banner, missing CSV columns, stuck/Inf UW weights, missing student export):
```bash
# Auto-detects latest versioned log (sanity_*.log, _v1.log, _v2.log, ...) so the
# verify never silently passes when log filenames have version suffixes — that
# silent-pass mode hit us once already (mtime-glob fix).
for v in 2stage_noDI 3stage_DI; do
    log=$(ls -t logs/sanity_${v}_multitask*.log 2>/dev/null | head -1)
    out="outputs/sanity/surgery_${v}_multitask"
    echo "=== ${v}_multitask  (log=$(basename "$log"))"
    if [ -z "$log" ] || [ ! -f "$log" ]; then
        echo "  ❌ log file not found — re-run SANITY"; continue
    fi
    grep -q "SURGERY COMPLETE" "$log" && echo "  ✓ no crash" || echo "  ❌ training did NOT complete"
    grep -q "Uncertainty Weighting ENABLED" "$log" && echo "  ✓ UW active" || echo "  ❌ UW banner missing"
    grep -qiE "loss.*nan|=nan" "$log" && echo "  ❌ NaN" || echo "  ✓ no NaN"
    head -1 "$out/loss_log.csv" 2>/dev/null | grep -q "uw_w_jepa" \
        && echo "  ✓ UW columns in CSV" || echo "  ❌ UW columns missing"
    [ -f "$out/student_encoder.pt" ] \
        && echo "  ✓ student exported ($(ls -lh $out/student_encoder.pt | awk '{print $5}'))" \
        || echo "  ❌ student missing"
    venv_walkindia/bin/python3 -c "
import csv, math
rows = list(csv.DictReader(open('$out/loss_log.csv')))
finite = all(math.isfinite(float(r[k])) for r in rows for k in ['uw_w_jepa','uw_w_infonce','uw_w_tcc'])
print(f'  ✓ UW weights all finite across {len(rows)} step(s)' if finite else '  ❌ UW NaN/Inf')
last = rows[-1]
print(f'  final UW: w_jepa={float(last[\"uw_w_jepa\"]):.4f} '
      f'w_infonce={float(last[\"uw_w_infonce\"]):.4f} w_tcc={float(last[\"uw_w_tcc\"]):.4f}')
"
done
```

**If all ✓ green → safe to launch Phase 2 FULL.** If anything ❌ red → investigate the SANITY log before burning 14 GPU-h on a broken UW path.

---

## 2. Phase 2 — train 2 variants (FULL, ~14 GPU-h sequential)

```bash
tmux new -s iter12_train

./scripts/run_train.sh \
    configs/train/surgery_2stage_noDI_multitask.yaml \
    configs/train/surgery_3stage_DI_multitask.yaml \
    2>&1 | tee logs/run_train_iter12_multitask_v1.log
```

**Verify per variant:**
```bash
for v in surgery_2stage_noDI_multitask surgery_3stage_DI_multitask; do
    f="outputs/full/$v/student_encoder.pt"
    if [ -f "$f" ]; then
        echo "✅ $v: $(ls -lh "$f" | awk '{print $5}')"
        python3 -c "
import json, csv
s = json.load(open('outputs/full/$v/training_summary.json'))
print(f'  best_ckpt: {s.get(\"best_ckpt\")}  early_stop: {s.get(\"early_stop\")}')
print(f'  BWT Prec@K: {s.get(\"probe_trajectory_stats\", {}).get(\"bwt_prec_at_k\")}')
last = list(csv.DictReader(open('outputs/full/$v/loss_log.csv')))[-1]
print(f'  UW final weights: w_jepa={float(last[\"uw_w_jepa\"]):.3f} '
      f'w_infonce={float(last[\"uw_w_infonce\"]):.3f} '
      f'w_tcc={float(last[\"uw_w_tcc\"]):.3f}')
" 2>/dev/null || echo "  (training_summary.json or loss_log.csv missing/unparseable)"
    else
        echo "❌ $v: student_encoder.pt missing → check logs/run_train_${v}*.log"
    fi
done
```

---

## 3. Phase 3 — EVAL

### 3.1 Evaluate 5 variants (3 standard from iter11 v3 + 2 new multitask)

```bash
./scripts/run_eval.sh \
    configs/train/explora.yaml \
    configs/train/surgery_2stage_noDI.yaml \
    configs/train/surgery_3stage_DI.yaml \
    configs/train/surgery_2stage_noDI_multitask.yaml \
    configs/train/surgery_3stage_DI_multitask.yaml \
    2>&1 | tee logs/run_eval_iter12_multitask_v1.log
```

> Expects ~5–10 min wall: m05_frozen + per_clip_*.npz + 3 standard m05 outputs are CACHED from iter11 v3 (`run_eval_all_v3.log`); only 2 new multitask m05 + m06 cycles run. m08b always recomputes (per docstring) → fresh per-variant `<variant>/eval/` + aggregate `outputs/full/m08b_aggregate/` plots.

**Verify aggregate output (paired-Δ chart is the headline):**
```bash
ls -lh outputs/full/m08b_aggregate/m08b_paired_delta.{png,pdf} \
       outputs/full/m08b_aggregate/paired_bootstrap_results.json

python3 -c "
import json
r = json.load(open('outputs/full/m08b_aggregate/paired_bootstrap_results.json'))
print(f'Frozen baseline: {r[\"frozen\"]}')
print(f'{\"Adapted encoder\":<48s} {\"Δ Prec@K (Easy)\":>16s} {\"p_vs_0\":>8s}  Verdict')
print('-' * 86)
for c in sorted(r['comparisons'],
                key=lambda c: -c['modes']['easy']['metrics']['prec_at_k']['delta_mean']):
    pk = c['modes']['easy']['metrics']['prec_at_k']
    sig = '✅' if pk['delta_ci_lo'] > 0 or pk['delta_ci_hi'] < 0 else '🟡'
    print(f'{c[\"adapted\"]:<48s} {pk[\"delta_mean\"]:>+13.4f} pp {pk[\"p_value_vs_zero\"]:>8.4f}  {sig}')
"
```

PASS if either multitask variant's Δ Prec@K (Easy) beats its pure-JEPA counterpart from iter11 v3 (`surgery_3stage_DI` baseline = +0.87 ± 0.62 ✅).
