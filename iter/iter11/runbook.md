# 🚀 FactorJEPA Runbook — iter11 v2 (terminal commands only)

> 4-variant chain (`explora` + `surgery_2stage_noDI` + `surgery_2stage_loud_agent` + `surgery_3stage_DI`) at 10K, paired BCa on `data/eval_10k.json` (N=9,297, CI_half ≈ ±0.42 pp).
> Architecture / decisions / errors → `plan_TODO.md` / `plan_training.md` / `errors_N_fixes.md`.

---

## 0. Setup (one-time on fresh GPU instance)

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
mkdir -p logs && ./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
source venv_walkindia/bin/activate
chmod +x git_push.sh git_pull.sh scripts/*.sh scripts/lib/yaml_extract.py
./git_pull.sh 2>&1 | tee logs/git_pull.log
df -h /workspace                                                       # need ≥ 200 GB
```

**Verify:**
```bash
for y in configs/train/explora.yaml configs/train/surgery_2stage_noDI.yaml \
         configs/train/surgery_2stage_loud_agent.yaml configs/train/surgery_3stage_DI.yaml; do
    mod=$(scripts/lib/yaml_extract.py "$y" data.module)
    enc=$(scripts/lib/yaml_extract.py "$y" data.adapted_encoder)
    od=$(scripts/lib/yaml_extract.py "$y" data.output_dir)
    echo "  $(basename "$y" .yaml): module=$mod  encoder=$enc  output_dir=$od"
done
ls scripts/run_factor_prep.sh scripts/run_train.sh scripts/run_eval.sh scripts/lib/yaml_extract.py
```

---

## 1. Phase 1 — factor prep (m10 + m11), ~10 h GPU

```bash
tmux new -s iter11_v2
./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml \
    2>&1 | tee logs/run_factor_prep.log
```

**Verify A (m10):**
```bash
cat outputs/full/m10_sam_segment/summary.json | python3 -m json.tool
du -sh outputs/full/m10_sam_segment/
python3 -c "
import json
s = json.load(open('outputs/full/m10_sam_segment/summary.json'))
qg = s.get('quality_gate', 'UNKNOWN')
assert qg == 'PASS', f'quality_gate={qg} (expected PASS)'
print(f'✅ quality_gate=PASS · n_agents={s.get(\"n_total_agents\")} · clips_with_agents={s.get(\"clips_with_agents_pct\", 0)*100:.0f}%')
"
```

**Verify B (m11):**
```bash
python3 -c "
import json
m = json.load(open('outputs/full/m11_factor_datasets/factor_manifest.json'))
print(f'clips: {len(m)}')
tubes = [v.get('n_interaction_tubes', 0) for v in m.values()]
print(f'D_I tubes: {sum(tubes)} (expect > 0 — interaction_mining=true in surgery_3stage_DI.yaml)')
dl = sum(1 for v in m.values() if v.get('has_D_L'))
da = sum(1 for v in m.values() if v.get('has_D_A'))
print(f'D_L eligible: {dl}/{len(m)}; D_A eligible: {da}/{len(m)}')
assert sum(tubes) > 0, 'D_I tubes should be > 0 for the maximal factor config'
print('✅ m11 manifest OK')
"
ls outputs/full/m11_factor_datasets/D_L/ | wc -l   # ~100 verify clips
du -sh outputs/full/m11_factor_datasets/
```

---

## 2. Phase 2 — train 4 variants, ~30-60 h GPU

```bash
./scripts/run_train.sh \
    configs/train/explora.yaml \
    configs/train/surgery_2stage_noDI.yaml \
    configs/train/surgery_2stage_loud_agent.yaml \
    configs/train/surgery_3stage_DI.yaml \
    2>&1 | tee logs/run_train_iter11_v2.log
```

**Verify per variant:**
```bash
for v in explora surgery_2stage_noDI surgery_2stage_loud_agent surgery_3stage_DI; do
    f="outputs/full/$v/student_encoder.pt"
    if [ -f "$f" ]; then
        echo "✅ $v: $(ls -lh "$f" | awk '{print $5}')"
        python3 -c "
import json
s = json.load(open('outputs/full/$v/training_summary.json'))
print(f'  best_ckpt: {s.get(\"best_ckpt\")}')
print(f'  early_stop: {s.get(\"early_stop\")}')
print(f'  BWT: {s.get(\"probe_trajectory_stats\", {}).get(\"bwt_prec_at_k\")}')
" 2>/dev/null || echo "  (training_summary.json missing/unparseable)"
    else
        echo "❌ $v: student_encoder.pt missing → check logs/run_train_${v}.log"
    fi
done
```

---

## 3. Phase 3 — paired eval, ~12 h GPU

```bash
./scripts/run_eval.sh \
    configs/train/explora.yaml \
    configs/train/surgery_2stage_noDI.yaml \
    configs/train/surgery_2stage_loud_agent.yaml \
    configs/train/surgery_3stage_DI.yaml \
    2>&1 | tee logs/run_eval_iter11_v2.log
```

**Verify shared frozen baseline:**
```bash
ls -lh outputs/full/m05_vjepa_embed/embeddings_vjepa_2_1_frozen.npy \
       outputs/full/m06_faiss_metrics/m06_metrics_vjepa_2_1_frozen.json \
       outputs/full/m06_faiss_metrics/per_clip_vjepa_2_1_frozen_easy.npz \
       outputs/full/m06_faiss_metrics/per_clip_vjepa_2_1_frozen_hard.npz
```

**Verify per variant (m05 surgical + m06 + m08b):**
```bash
for v in explora surgery_2stage_noDI surgery_2stage_loud_agent surgery_3stage_DI; do
    echo "=== $v ==="
    enc=$(scripts/lib/yaml_extract.py "configs/train/${v}.yaml" data.adapted_encoder)
    ls -lh "outputs/full/m05_vjepa_embed/embeddings_${enc}.npy" 2>/dev/null \
        || echo "  ❌ m05 ${enc} embeddings missing"
    ls -lh "outputs/full/m06_faiss_metrics/m06_metrics_${enc}.json" 2>/dev/null \
        || echo "  ❌ m06 ${enc} metrics missing"
    ls -lh "outputs/full/m08b_compare/paired_bootstrap_results.json" 2>/dev/null \
        || echo "  ❌ m08b paired_bootstrap_results.json missing"
done
```

---

## 4. 🏆 Final decision gate

```bash
python3 -c "
import json
from pathlib import Path
print(f'{\"variant\":>30} | {\"Frozen\":>8} | {\"Adapted\":>8} | {\"Δ pp\":>8} | {\"CI_half\":>8} | {\"p\":>8} | sig?')
print('-' * 105)
for v in ['explora', 'surgery_2stage_noDI', 'surgery_2stage_loud_agent', 'surgery_3stage_DI']:
    p = Path(f'outputs/full/{v}/eval10k/paired_bootstrap_results.json')
    if not p.exists():
        print(f'{v:>30} | {\"—\":>8} | {\"—\":>8} | {\"—\":>8} | {\"—\":>8} | {\"—\":>8} |')
        continue
    d = json.load(open(p))
    pk = d['modes']['easy']['metrics']['prec_at_k']
    sig = '✅' if (pk['delta_ci_lo'] > 0 or pk['delta_ci_hi'] < 0) else '🟡'
    print(f'{v:>30} | {pk[\"frozen_mean\"]*100:>7.2f}% | {pk[\"surgical_mean\"]*100:>7.2f}% | '
          f'{pk[\"delta_mean\"]*100:>+7.2f} | ±{pk[\"delta_ci_half\"]*100:>6.2f} | {pk[\"p_value_vs_zero\"]:>8.4f} | {sig}')
print()
print('PASS gate: Δ ≥ +3 pp · delta_ci_lo > 0 → publishable + 50K ladder')
print('MARGINAL:  Δ ∈ [+0.3, +3) pp → 50K scale-up only')
print('SATURATED: Δ < +0.3 pp → concede tier (ablation-table reframe)')
"
```

---

## 5. Resume / cache notes

- **m10 / m11 / m05 / m06 / m08b** all own per-module `output_guard` + `--cache-policy {1=keep,2=recompute}`. Re-running `run_factor_prep.sh` / `run_train.sh` / `run_eval.sh` is idempotent — completed steps short-circuit.
- **m05 surgical checkpoint** is keyed by `_checkpoint_fingerprint(model_path, mtime, size)` (#75) — variants do not collide.
- **m05 frozen partial-tolerance** (#72): if `eval_10k` decode fails on >95% but ≥80% clips, m05 saves partial `.npy` + `failed_clip_keys_*.json` and exits 0 (downstream m06/m08b use whichever clips succeeded).
- **`m08b` overwrite caveat**: `outputs/full/m08b_compare/paired_bootstrap_results.json` overwrites per variant. To preserve per-variant: `cp outputs/full/m08b_compare/paired_bootstrap_results.json outputs/full/<v>/eval10k/` between variants (or after each `run_eval` call).
