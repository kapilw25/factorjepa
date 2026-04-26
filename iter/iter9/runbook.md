# 🚀 FactorJEPA Runbook — iter9 (10K scale-up on 500+ GB vast.ai instance)

> 🏆 **High-level goal**: Surgery > ExPLoRA > Frozen on Prec@K at 115K clips.
> 🎯 **iter9 goal**: Surgery > Frozen on Prec@K at **10K training clips**, evaluated on **disjoint 500-clip test_500** held-out set (CI ±2.1 pp — 2× tighter than 1K POC's ±4.5 pp).
> ⚖️ **Val/Test split policy**: `val_1k.json` → `val_500.json` (m09c probe / best-ckpt / BWT / early-stop) + `test_500.json` (m06 decision gate, touched **once**). Eliminates best-of-K selection bias (~2.2 pp) that would have invalidated iter8's single-set design. Disjoint, seed=42. Files already generated: `data/val_500.json`, `data/test_500.json`.
> 🧪 SANITY (code-check only): `--SANITY` on 20 clips.

> Run on GPU. Verify each step before moving on.
> **Architecture + decisions** → `plan_TODO.md`. **Error history** → `errors_N_fixes.md`. **1K POC results + hyperparams + wall-times + disk footprint** → `iter/utils/experiment_log.md` ("Run 2026-04-19" entry).

---
> 💡 **Streaming savings**: removing the 340 GB m11 D_L/D_A write at 10K (7× reduction: 380 → 52 GB). Unlocks 50K on ~150 GB + 115K on ~345 GB — the full scale-ladder fits on a single 200 GB instance.

**Rent a vast.ai instance with `/workspace` disk ≥ 200 GB** (4× headroom over 52 GB actual). Verify on spin-up:
```bash
df -h /workspace    # must show ≥ 200 GB capacity BEFORE Step A
```

If you already have iter8 1K POC artifacts on disk (`outputs/poc/m11_factor_datasets/D_{L,A}/*.npy` ≈ 340 GB), back them up to HF via `./git_push.sh "snapshot iter8"` then delete:
```bash
rm -rf outputs/poc/m11_factor_datasets/D_L outputs/poc/m11_factor_datasets/D_A
```

---

## 🛠️ GPU Setup (one-time on fresh 200 GB instance)

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
mkdir -p logs && ./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
source venv_walkindia/bin/activate
chmod +x git_push.sh git_pull.sh
./git_push.sh --code-only "updating"
./git_pull.sh 2>&1 | tee logs/git_pull.log   # pulls data/subset_10k_local (~11 GB) + data/val_1k_local
df -h /workspace
```

---

## ⚙️ One-time yaml edit before Step B (skip D_I generation — Stage 3 retired)

```bash
# Disable interaction mining in m11 (D_L and D_A still generated — 2-stage recipe needs them)
sed -i 's/^  enabled: true.*# Set false for POC/  enabled: false                     # Set true only if re-enabling Stage 3 ablation/' configs/train/ch11_surgery.yaml

grep -A1 "^interaction_mining:" configs/train/ch11_surgery.yaml | head -3
# Expected:
#   interaction_mining:
#     enabled: false                     # Set true only if re-enabling Stage 3 ablation
```

Saves ~210 GB disk at 10K. D_L + D_A generation unchanged.

> **Why m11 is still needed**: Stage 1 input = 100 % D_L (layout-only), Stage 2 input = 70 % D_A + 30 % D_L (agents + layout replay). Only D_I (interaction tubes, Stage 3) is skipped.

---

## 🟢 Step A: m10 Grounded-SAM — 10K clips, GPU, ~10 h

```bash
rm -rf outputs/full/m10_sam_segment/
python -u src/m10_sam_segment.py --FULL \
    --subset data/subset_10k.json \
    --local-data data/subset_10k_local --no-wandb \
    2>&1 | tee logs/m10_10k_v1.log
```

**Verify:**
```bash
cat outputs/full/m10_sam_segment/summary.json | python3 -m json.tool
du -sh outputs/full/m10_sam_segment/
```

| Check | Expect (10K, linear from 1K's 3.62 s/clip) |
|---|---|
| Wall time | **~10 h** |
| `n_total_agents` | ~400K-600K |
| `clips_with_agents_pct` | ≥ 0.80 |
| `quality_gate` | `"PASS"` (4 checks) |
| Disk written | ~30 GB (masks + overlay PNGs) |

---

## ⏳ Step B: m11 factor datasets (streaming — manifest + 100 verify only) — CPU, ~10 min

```bash
rm -rf outputs/full/m11_factor_datasets/
python -u src/m11_factor_datasets.py --FULL --streaming \
    --subset data/subset_10k.json \
    --local-data data/subset_10k_local --no-wandb \
    2>&1 | tee logs/m11_10k_v1.log
```

> `--streaming` short-circuits factor-gen + `np.save` for ~9,900 non-verify clips (manifest entries still written from `agent_pixel_ratio` thresholds). Only the 100 `select_verify_clips(seed=42)` curated clips get full D_L/D_A processing for `plot_factor_per_clip`. D_L/D_A feed into m09c on-demand via `utils.factor_streaming.stream_factor`. Bitwise parity verified — see `scripts/tests_streaming/test_parity.py` (10/10 PASSED).

**Verify:**
```bash
python3 -c "
import json
m = json.load(open('outputs/full/m11_factor_datasets/factor_manifest.json'))
print(f'clips in manifest: {len(m)}')
tubes = [v.get('n_interaction_tubes', 0) for v in m.values()]
print(f'D_I tubes: {sum(tubes)} (expect 0 — interaction_mining disabled)')
dl = sum(1 for v in m.values() if v['has_D_L'])
da = sum(1 for v in m.values() if v['has_D_A'])
print(f'D_L eligible: {dl}/{len(m)}; D_A eligible: {da}/{len(m)}')
"
ls outputs/full/m11_factor_datasets/D_L/ | wc -l   # ~100 verify clips only
du -sh outputs/full/m11_factor_datasets/
```

| Check | Expect (10K, --streaming, D_I off) |
|---|---|
| Wall time | ~10 min (~9,900 short-circuited × ~0.001 s + 100 full-process × ~2 s) |
| `factor_manifest.json` | 10,000 entries; `n_interaction_tubes: 0` all rows |
| D_L eligible | 10,000 (manifest flag) |
| D_A eligible | ≥ 8,000 (80 % ratio) |
| D_I present | 0 (disabled) |
| `D_L/*.npy` file count | ~100 (verify clips only) |
| Disk written | **~0.5 GB** (was ~340 GB pre-streaming — 680× reduction) |

---

## 🎯 Step C: m09c Surgery — 2-stage, 1 epoch — GPU, ~2.5 h

> **2-stage recipe** (Stage 3 dropped post-1K-POC): Stage 1 (0-25 % layers, 100 % D_L) + Stage 2 (0-50 % layers, 70 % D_A + 30 % D_L).
> **`--FULL` mode** → `use_permanent_val=true` → trains on all 10K clips, validates on `data/val_500` (N=500, CI ±2.1 pp) via `--probe-subset`.
> **Early-stop suite auto-active**: plateau (Δval_jepa < 1e-3 × 5 probes), BWT (< −0.5 pp × 10 probes), catastrophic (−5 pp × 3 probes) — trigger reason in `training_summary.json.early_stop.reason`.
> **Streaming DataLoader auto-active**: yaml default `factor_streaming.full: true` → builds `StreamingFactorDataset` + `DataLoader(num_workers=16, persistent_workers, prefetch_factor=4)`. One-shot TAR scan at stage setup (~10 sec). D_L/D_A generated per-step on CPU workers. Override: `--no-factor-streaming` forces legacy .npy path (requires Step B to have run WITHOUT `--streaming`).

```bash
rm -rf outputs/full/m09c_surgery/
python -u src/m09c_surgery.py --FULL \
    --subset data/subset_10k.json \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/ch11_surgery.yaml \
    --factor-dir outputs/full/m11_factor_datasets/ \
    --local-data data/subset_10k_local \
    --probe-subset data/val_500.json \
    --probe-local-data data/val_1k_local \
    --probe-tags data/val_1k_local/tags.json \
    --no-wandb \
    2>&1 | tee logs/m09c_10k_v1.log
```

> `--probe-subset data/val_500.json` routes the probe to the 500-clip val split (disjoint from test_500). `student_best.pt` is now selected on val_500 only — test_500 stays fully held-out for Step F.

**Verify:**
```bash
ls -lh outputs/full/m09c_surgery/student_encoder.pt outputs/full/m09c_surgery/val_split.json
python3 -c "
import json
s = json.load(open('outputs/full/m09c_surgery/training_summary.json'))
print('split:', s['train_val_split'])
print('best_ckpt:', s['best_ckpt'])
print('early_stop:', s['early_stop'])
print('BWT:', s['probe_trajectory_stats']['bwt_prec_at_k'])"
```

| Check | Expect (10K × 1 epoch / BS=32 ≈ 312 steps / 2 stages ≈ 156 per stage) |
|---|---|
| Wall time | ~2.5 h (full budget) — shorter if plateau/BWT fires early |
| `early_stop.reason` | `null` (healthy) OR `val_loss_plateau` / `negative_bwt` / `catastrophic_forgetting` |
| `best_ckpt.prec_at_k` | > 20 (iter8 1K POC baseline = 20.50) |
| BWT (`bwt_prec_at_k`) | > 0 ideally (iter8 was −0.33 pre-patch) |
| `val_split.json` | 500 keys from `data/val_500.json` (probe split) |
| `val_split.split_strategy` | `"permanent"` |
| Probe evaluated on | `val_500` (not val_1k, not test_500) — test_500 stays held-out for Step F |

---

## ⏳ Step D: m05 frozen embed on test_500 — GPU, ~11 min

```bash
rm -rf outputs/full/m05_vjepa_embed/
python -u src/m05_vjepa_embed.py --FULL \
    --subset data/test_500.json \
    --model-config configs/model/vjepa2_1.yaml \
    --encoder vjepa_2_1_frozen \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_test500_frozen_v1.log
```

| Check | Expect |
|---|---|
| Wall time | ~11 min (1.27 s/clip × 500) |
| Shape | (500, 1664) |

---

## ⏳ Step E: m05 surgical embed on test_500 — GPU, ~25 min

```bash
rm -f outputs/full/m05_vjepa_embed/embeddings_vjepa_2_1_surgical*.npy \
      outputs/full/m05_vjepa_embed/.m05_checkpoint_vjepa_2_1_surgical.npz
python -u src/m05_vjepa_embed.py --FULL \
    --subset data/test_500.json \
    --model-config configs/model/vjepa2_1.yaml \
    --model outputs/full/m09c_surgery/student_encoder.pt \
    --encoder vjepa_2_1_surgical \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_test500_surgical_v1.log
```

> Adapted-model compile adds ~2.4× overhead (iter8 measured 2.85 s/clip adapted vs 1.18 s/clip frozen) → ~25 min for 500 clips.

---

## 🏆 Step F: m06 FAISS Prec@K decision gate — GPU, ~2 min

```bash
rm -f outputs/full/m06_faiss_metrics/m06_metrics_vjepa_2_1_frozen.json outputs/full/m06_faiss_metrics/knn_indices_vjepa_2_1_frozen.npy
python -u src/m06_faiss_metrics.py --FULL \
    --subset data/test_500.json \
    --encoder vjepa_2_1_frozen \
    --no-wandb \
    2>&1 | tee logs/m06_test500_frozen.log

rm -f outputs/full/m06_faiss_metrics/m06_metrics_vjepa_2_1_surgical.json outputs/full/m06_faiss_metrics/knn_indices_vjepa_2_1_surgical.npy
python -u src/m06_faiss_metrics.py --FULL \
    --subset data/test_500.json \
    --encoder vjepa_2_1_surgical \
    --no-wandb \
    2>&1 | tee logs/m06_test500_surgical.log
```

**Decision:**
```bash
python3 -c "
import json
for name in ['frozen', 'surgical']:
    m = json.load(open(f'outputs/full/m06_faiss_metrics/m06_metrics_vjepa_2_1_{name}.json'))
    pk = m['easy']['precision_at_k']
    print(f'{name:12s}: Prec@K = {pk[\"mean\"]:.2f}% +/- {pk[\"ci\"][\"ci_half\"]:.2f}')
"
```

| Outcome | Next action |
|---|---|
| Δ ≥ 3 pp (non-overlap @ N=500, CI ~±2.1 pp) | → Step G (ExPLoRA arm); scale-ladder per `plan_TODO.md` §Step H |
| Δ ∈ [1, 3) pp (marginal) | → BWT Option B (λ=50) in `plan_TODO.md`; re-run C/D/E/F only |
| Δ < 1 pp (saturated) | → 10K is publishable tier; skip ladder; Step G only |
| Surgery ≤ Frozen | → BWT Options B → C in `plan_TODO.md` |
| Plateau/BWT kill fired early in C (on val_500) | → inspect `m09_forgetting.png` + `probe_trajectory.png`; BWT queue in `plan_TODO.md` |

---

## 🔒 Step G: ExPLoRA comparison arm — conditional on F gate-pass

```bash
rm -rf outputs/full/m09b_explora/
python -u src/m09b_explora.py --FULL \
    --subset data/subset_10k.json \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/explora.yaml \
    --local-data data/subset_10k_local --no-wandb \
    2>&1 | tee logs/m09b_10k_explora.log

# m05 + m06 for explora arm (same pattern as D/E/F on --subset data/test_500.json,
# --encoder vjepa_2_1_explora) — stays on test_500 for apples-to-apples Prec@K
```

---

## 🏁 Final 3-arm comparison (after Step G)

```bash
python3 -c "
import json
for name in ['frozen', 'explora', 'surgical']:
    f = f'outputs/full/m06_faiss_metrics/m06_metrics_vjepa_2_1_{name}.json'
    try:
        m = json.load(open(f))
        pk = m['easy']['precision_at_k']
        print(f'{name:12s}: Prec@K = {pk[\"mean\"]:.1f}% +/- {pk[\"ci\"][\"ci_half\"]:.1f}')
    except FileNotFoundError:
        print(f'{name:12s}: NOT YET RUN')
"
```

**Win conditions:**
- Surgery > ExPLoRA > Frozen, non-overlapping 95 % CIs → paper headline, scale 50K → 115K.
- Surgery > Frozen, ExPLoRA ≈ Frozen → strong paper (surgery is the novelty).
- Surgery ≈ ExPLoRA > Frozen → publishable but weakens "surgery is special" claim.
- Surgery ≤ Frozen → pivot to temporal-interference-projection diagnostic paper (`iter/utils/literarure_survey.md`).

---

## 📜 iter8 1K POC (archived)

Full 1K POC run — commands, results, wall-times (~3h40m A→F), disk footprint (71 GB), stage-by-stage diagnosis, and the decision that retired Stage 3 + bumped replay 10→30 % — lives in `iter/utils/experiment_log.md` entry dated 2026-04-19. 1K tier **RETIRED** (N=100 val-split ±4.5 pp CI too wide to resolve sub-pp Surgery-vs-Frozen delta).
