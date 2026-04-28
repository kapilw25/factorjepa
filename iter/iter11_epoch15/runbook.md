# 🚀 FactorJEPA Runbook — iter11 v3 (terminal commands only)

> 4-variant chain (`explora` + `surgery_2stage_noDI` +  `surgery_3stage_DI`) on the ultra_hard 3,066-clip tier, paired BCa on `data/ultra_hard_3066_eval.json` (N=308, CI_half ≈ ±2.4 pp at p=0.5). Iter11 v3 hard-pivot replaces v2's 10K/eval_10k random tier (which saturated to Δ≈0).
> Architecture / decisions / errors → `plan_TODO.md` / `plan_training.md` / `errors_N_fixes.md`.

---

## 0. Setup (one-time on fresh GPU instance)

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
mkdir -p logs && ./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
source venv_walkindia/bin/activate
chmod +x git_push.sh git_pull.sh scripts/*.sh scripts/lib/yaml_extract.py
./git_pull.sh 2>&1 | tee logs/git_pull.log
df -h /workspace                                                       # need ≥ 50 GB for Strategy A; ≥ 150 GB for Strategy B
```

**Setup sanity (run on EVERY new instance — `/workspace` is shared but `/usr/lib` is PER-INSTANCE, so all 6 apt steps from setup_env_uv.sh must complete locally; otherwise crashes mid-training).**
```bash
# Catches all 6 system-pkg gaps that setup_env_uv.sh installs (paths in setup_env_uv.sh):
ldconfig -p | grep -q "libopenblas.so.0" \
    || { echo "FATAL: libopenblas-dev missing (line 411) — apt-get install -y libopenblas-dev"; exit 1; }
command -v nvcc &>/dev/null \
    || { echo "FATAL: cuda-toolkit missing (line 369) — apt-get install -y cuda-toolkit-12-8"; exit 1; }
command -v python3.12 &>/dev/null \
    || { echo "FATAL: python3.12 missing (line 125) — see deadsnakes PPA in setup_env_uv.sh"; exit 1; }
for cmd in jq tmux wget curl aria2c; do
    command -v "$cmd" &>/dev/null || echo "WARN: $cmd missing (line 113/266) — convenience tool, m00d falls back"
done
# Python-import smoke (each line maps to a runtime crash class):
python -c "import torch; assert torch.cuda.is_available(); print(f'  ✓ torch {torch.__version__} · CUDA {torch.version.cuda} · {torch.cuda.get_device_name(0)}')"
python -c "import faiss; print(f'  ✓ faiss {faiss.__version__}')"                # libopenblas
python -c "import scipy; print(f'  ✓ scipy {scipy.__version__}')"                  # libopenblas (m11)
python -c "import cv2; print(f'  ✓ opencv-python {cv2.__version__}')"              # libGL/libgthread (m04d/m11)
python -c "import av; print(f'  ✓ pyav {av.__version__}')"                          # FFmpeg (m05/m11/m04d decode)
python -c "from transformers import AutoModel, Sam3TrackerVideoModel; print('  ✓ transformers + SAM3 importable')"  # m10
echo "✓ Setup sanity passed — runbook chain safe to launch"
```

---

## 0b. Data download — Strategies A & B (run ONCE, no GPU)

> Verified disk math (HF dataset = 130 GB / 115,687 clips = **1.13 MB/clip** measured; m00d log line 25: 3,066 clips → 3.96 GB on disk in 7.5 min).

### Strategy A — single download, share across train/val/eval ⭐ RECOMMENDED
Download `data/ultra_hard_3066.json` ONCE (3,066 clips, ~4 GB, ~7-15 min) → all 3 splits read from the same `data/ultra_hard_3066_local/` via `--subset` filtering at `iter_clips_parallel(subset_keys=...)`.

```bash
python -u src/m00d_download_subset.py --FULL \
    --subset data/ultra_hard_3066.json --no-wandb 2>&1 | tee logs/m00d_ultra_hard.log
# → data/ultra_hard_3066_local/  (parent TAR dir for ALL 3 splits + 9 single-condition subsets that overlap)
```

Then in each train yaml's `data:` block:
```yaml
data:
  train_subset:    data/ultra_hard_3066_train.json   # 2,452 clips
  train_local_data: data/ultra_hard_3066_local        # ← shared
  val_subset:      data/ultra_hard_3066_val.json     # 306 clips
  val_local_data:   data/ultra_hard_3066_local        # ← shared
  eval_subset:     data/ultra_hard_3066_eval.json    # 308 clips
  eval_local_data:  data/ultra_hard_3066_local        # ← shared
```

### Strategy B — also download the 8 single-condition categories (ablation matrix)
Run the chain once per category (each invocation re-walks 116 HF TARs but writes only matched clips):

```bash
for sub in ultra_hard_3066 ge3_indian_objects crowd_high traffic_mix_pedestrian_dominant \
           traffic_high traffic_mix_mixed_all ge4_indian_objects road_encroachment_heavy ge5_indian_objects; do
    python -u src/m00d_download_subset.py --FULL \
        --subset "data/${sub}.json" --no-wandb 2>&1 | tee "logs/m00d_${sub}.log" ;
done
```

### Disk + wall time (measured + projected)

| Action | Clips | Disk | Wall time |
|---|---:|---:|---:|
| **A**: ultra_hard_3066 parent only | 3,066 | **~4 GB** (measured) | **~7.5 min** (measured) |
| **B**: all 9 categories union | ~99K unique | **~109 GB** (1.13 MB × 99K) | **~70 min** (9 × 7.5 min) |
| Full 115K corpus reference | 115,687 | **~127 GB** | ~7 min/116-TAR-walk |
| GPU cost | — | — | **0 h GPU** (CPU+network only) |

---

**Verify:**
```bash
for y in configs/train/explora.yaml \
        configs/train/surgery_2stage_noDI.yaml \
        configs/train/surgery_3stage_DI.yaml; do
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
tmux new -s iter11_v3
./scripts/run_factor_prep.sh configs/train/surgery_3stage_DI.yaml \
    2>&1 | tee logs/run_factor_prep_v3.log
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

**Verify C (per-variant sufficiency gate — confirms m10/m11 outputs cover the train_subset all 3 surgery variants share, AND that surgery_3stage_DI's Stage 3 has interactions to consume):**
```bash
# C1. m10 covers every clip in the shared train_subset
python3 -c "
import json
seg = json.load(open('outputs/full/m10_sam_segment/segments.json'))
sub = json.load(open('data/ultra_hard_3066_train.json'))['clip_keys']
covered = set(sub) & set(seg.keys())
missing = set(sub) - set(seg.keys())
n_with_inter = sum(1 for k in covered if seg[k]['n_interactions'] > 0)
print(f'm10 covered: {len(covered)}/{len(sub)} ({100*len(covered)/len(sub):.1f}%)')
print(f'  with interactions: {n_with_inter}/{len(covered)}')
assert not missing, f'm10 missed {len(missing)} train clips — all 3 variants will FATAL'
assert n_with_inter > 0, 'D_I tubes will be empty → surgery_3stage_DI Stage 3 has no data'
print('✅ m10 covers all train clips + has interactions')
"

# C2. m11 manifest covers all train clips with proper has_D_L / has_D_A / has_D_I flags
python3 -c "
import json
m = json.load(open('outputs/full/m11_factor_datasets/factor_manifest.json'))
sub = set(json.load(open('data/ultra_hard_3066_train.json'))['clip_keys'])
covered = sub & set(m.keys())
n_DL = sum(1 for k in covered if m[k].get('has_D_L'))
n_DA = sum(1 for k in covered if m[k].get('has_D_A'))
n_DI = sum(1 for k in covered if m[k].get('has_D_I'))
n_tubes = sum(m[k].get('n_interaction_tubes', 0) for k in covered)
print(f'manifest covers: {len(covered)}/{len(sub)} train clips')
print(f'  has_D_L: {n_DL} ({100*n_DL/max(len(covered),1):.1f}%) — needed by ALL 3 surgery variants')
print(f'  has_D_A: {n_DA} ({100*n_DA/max(len(covered),1):.1f}%) — needed by ALL 3')
print(f'  has_D_I: {n_DI} ({100*n_DI/max(len(covered),1):.1f}%) + {n_tubes} tubes — only surgery_3stage_DI uses')
assert n_DL > len(covered) * 0.5, 'D_L coverage <50% — Stage 1 starved'
assert n_DA > len(covered) * 0.3, 'D_A coverage <30% — Stage 2 starved'
assert n_DI > 0, 'D_I tubes empty — surgery_3stage_DI Stage 3 has no input'
print('✅ m11 manifest provides D_L+D_A for all 3 variants and D_I for surgery_3stage_DI')
"

# C3. Per-variant gate — confirms each yaml's train_subset is covered + D-requirements met
for variant in surgery_2stage_noDI surgery_3stage_DI; do
python3 -c "
import yaml, json
y = yaml.safe_load(open('configs/train/${variant}.yaml'))
sub = set(json.load(open(y['data']['train_subset']))['clip_keys'])
m = json.load(open('outputs/full/m11_factor_datasets/factor_manifest.json'))
covered = sub & set(m.keys())
needs_DI = y['interaction_mining']['enabled']
n_DL = sum(1 for k in covered if m[k].get('has_D_L'))
n_DA = sum(1 for k in covered if m[k].get('has_D_A'))
n_DI = sum(1 for k in covered if m[k].get('has_D_I'))
status = '✅' if (n_DL > 0 and n_DA > 0 and (not needs_DI or n_DI > 0)) else '❌'
print(f'{status} ${variant}: covered={len(covered)}/{len(sub)} D_L={n_DL} D_A={n_DA} D_I={n_DI} needs_DI={needs_DI}')
"
done
echo "All ✅ → safe to launch Phase 2. Any ❌ → that variant will FATAL at training start."
```

---

## 2. Phase 2 — train 4 variants, ~30-60 h GPU

```bash
./scripts/run_train.sh \
    configs/train/explora.yaml \
    configs/train/surgery_2stage_noDI.yaml \
    configs/train/surgery_3stage_DI.yaml \
    2>&1 | tee logs/run_train_v1.log
```

**Verify per variant:**
```bash
for v in explora surgery_2stage_noDI surgery_3stage_DI; do
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

## 3. Phase 3 — paired eval, ~12 h GPU (sequential) or ~5 h (4-way parallel)

### Option A — sequential (single instance, all 4 variants)
```bash
./scripts/run_eval.sh \
    configs/train/explora.yaml \
    configs/train/surgery_2stage_noDI.yaml \
    configs/train/surgery_3stage_DI.yaml \
    2>&1 | tee logs/run_eval_v1.log
```

### Option B — 4-way parallel (N instances or N processes on same machine) ⭐ FAST
Same-machine flock mutex (lock at `outputs/full/m05_vjepa_embed/.frozen.lock`) ensures **exactly ONE** process computes the frozen baseline; the other 3 block on the lock and reuse the cache. Per-variant `--output-dir <yaml.data.output_dir>/eval/` keeps `paired_bootstrap_results.json` + 8 plots + `metrics_table.tex` from overwriting. Per-yaml `data.adapted_encoder` (now unique: `vjepa_2_1_explora` / `_surgical_noDI` / `_surgical_loud_agent` / `_surgical_3stage_DI`) keeps m05/m06 outputs separate.

```bash
# Same time on all 4 (CACHE_POLICY_ALL=1 bypasses prompts for unattended overnight):
CACHE_POLICY_ALL=1 ./scripts/run_eval.sh configs/train/explora.yaml                   # Instance/proc A
CACHE_POLICY_ALL=1 ./scripts/run_eval.sh configs/train/surgery_2stage_noDI.yaml       # Instance/proc B
CACHE_POLICY_ALL=1 ./scripts/run_eval.sh configs/train/surgery_3stage_DI.yaml         # Instance/proc C
```

**Important — disk topology:** the flock + shared frozen baseline only coordinate processes that see the SAME inode. On this layout `/workspace` is `/dev/md127` (local RAID per-instance), so the 4-way parallel above only collapses on a SINGLE cloud instance with multiple GPUs/processes (e.g., 4 tmux panes). For 4 SEPARATE cloud instances each with own `/workspace`, each computes its own frozen baseline (~2.3 h × 4 GPU-hours) — to dedupe, mount a shared NFS for `outputs/full/m05_vjepa_embed/` + `outputs/full/m06_faiss_metrics/` first.

**Verify shared frozen baseline:**
```bash
ls -lh outputs/full/m05_vjepa_embed/embeddings_vjepa_2_1_frozen.npy \
       outputs/full/m06_faiss_metrics/m06_metrics_vjepa_2_1_frozen.json \
       outputs/full/m06_faiss_metrics/per_clip_vjepa_2_1_frozen_easy.npz \
       outputs/full/m06_faiss_metrics/per_clip_vjepa_2_1_frozen_hard.npz
```

**Verify per variant (m05 surgical + m06 + m08b at per-variant location):**
```bash
for v in explora surgery_2stage_noDI surgery_3stage_DI; do
    echo "=== $v ==="
    enc=$(scripts/lib/yaml_extract.py "configs/train/${v}.yaml" data.adapted_encoder)
    od=$(scripts/lib/yaml_extract.py  "configs/train/${v}.yaml" data.output_dir)
    ls -lh "outputs/full/m05_vjepa_embed/embeddings_${enc}.npy" 2>/dev/null \
        || echo "  ❌ m05 ${enc} embeddings missing"
    ls -lh "outputs/full/m06_faiss_metrics/m06_metrics_${enc}.json" 2>/dev/null \
        || echo "  ❌ m06 ${enc} metrics missing"
    ls -lh "${od}/eval/paired_bootstrap_results.json" 2>/dev/null \
        || echo "  ❌ m08b ${v} paired_bootstrap_results.json missing"
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
for v in ['explora', 'surgery_2stage_noDI', 'surgery_3stage_DI']:
    p = Path(f'outputs/full/{v}/eval/paired_bootstrap_results.json')
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

- **`utils.cache_policy.guarded_delete()` is the SOLE cross-module cache guard** (output_guard removed 2026-04-26). Default `--cache-policy 1` (keep) preserves all caches; pass `--cache-policy 2` to authorize destructive deletes. Per-script intra-run resume (m05's `_checkpoint_fingerprint`, m10's `.m10_checkpoint.json`, m11's manifest, m09c's epoch ckpts) handles "skip already-finished clips" — that is resume, not a guard.
- **m05 surgical checkpoint** is keyed by `_checkpoint_fingerprint(model_path, mtime, size)` (#75) — variants do not collide.
- **m05 frozen partial-tolerance** (#72): if `eval_10k` decode fails on >95% but ≥80% clips, m05 saves partial `.npy` + `failed_clip_keys_*.json` and exits 0 (downstream m06/m08b use whichever clips succeeded).
- **`m08b` per-variant output (2026-04-26)**: `run_eval.sh` now passes `--output-dir <yaml.data.output_dir>/eval/` to m08b so `paired_bootstrap_results.json` + 8 plots + `metrics_table.tex` no longer overwrite when 4 variants run in parallel on the same disk. Old shared `outputs/full/m08b_compare/` is reachable only when m08b is invoked manually without `--output-dir`.
- **Frozen baseline flock (2026-04-26)**: `run_eval.sh` opens fd 200 on `outputs/full/m05_vjepa_embed/.frozen.lock`; first process to acquire (non-blocking) computes m05+m06 frozen, others block on `flock 200` then read the freshly-written cache. 13 ms wake-up after release in tests. Coordinates same-host processes only — separate cloud instances with their own `/workspace` each compute independently.
