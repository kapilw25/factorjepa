# FactorJEPA Runbook
> **Final GOAL: Surgery > ExPLoRA > Frozen on Prec@K, 115K clips.**
> **Immediate GOAL: Surgery > ExPLoRA > Frozen on Prec@K, 1K clips from @data/val_1k_local/manifest.json**
> **m10/m11 Goal = maximize D_A/D_L/D_I accuracy for Prec@K**

> Run these commands on GPU. Verify each step before moving to the next.
> Architecture + decisions: `plan_TODO.md`. Error history: `errors_N_fixes.md`.

---

## GPU Setup (one-time, bare instance)

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
mkdir -p logs && ./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_env_gpu.log
# Step [9/9] pre-caches Grounding DINO weights (~1.8 GB) so m10 runs offline.
source venv_walkindia/bin/activate
./git_push.sh --code-only "updating"  # sync code only (no HF upload, use on both Mac and GPU)
./git_pull.sh 2>&1 | tee logs/git_pull.log  # downloads outputs + data from HF
```

---

## Step A: Grounded-SAM Segmentation — DINO + HF Sam3TrackerVideo ✅ validated 2026-04-15

```bash
rm -rf outputs/poc/m10_sam_segment/ outputs/poc/m11_factor_datasets/
python -u src/m10_sam_segment.py --POC \
    --subset data/sanity_100_dense.json \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m10_dense100.log
```

**Verify:** `cat outputs/poc/m10_sam_segment/summary.json | python3 -m json.tool`

| Check | Expect (100 dense clips, measured 2026-04-15) |
|---|---|
| Wall time | ~1100 s (11 s/clip — 4.21× faster than raw sam3 pkg) |
| `n_total_agents` | ~6100 |
| `n_total_interactions` | ~8700 |
| `mean_agent_pixel_ratio` | ~0.18 |
| `mean_mask_confidence` | ≥ 0.85 |
| `clips_with_agents_pct` | ≥ 0.95 |
| `quality_gate` | `"PASS"` (4 checks) |
| `m10_overlay_verify/*.png` | Red masks on real agents, no FPs on wires/signage |

---

## Step B: Factor Datasets (D_L + D_A + D_I with tight-bbox tubes) ✅ validated 2026-04-15

```bash
python -u src/m11_factor_datasets.py --POC \
    --subset data/sanity_100_dense.json \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m11_dense100.log
```

**Verify:**

| Check | Expect (100 dense clips, measured 2026-04-15) |
|---|---|
| Wall time | ~270 s (2.7 s/clip, CPU-bound) |
| `D_L present` | 100/100 |
| `D_A present` | ≥ 93/100 |
| `D_I present` | ≥ 90/100 |
| Total D_I tubes | ~8700 (bbox-adaptive, ~5600 unique shapes, not fixed 30% squares) |
| Median tubes/clip | ~65 (max ~400 for very dense Mumbai clips) |
| `m11_factor_samples.png` | D_L: agents blurred, layout sharp. D_A: agents bright, BG dimmed to 10% |
| `m11_per_Videoclip_verify/*.mp4` | 20 top videos, 2×2 H.264 grids, 960×540 @ 6fps |

```bash
python3 -c "
import json
m = json.load(open('outputs/poc/m11_factor_datasets/factor_manifest.json'))
tubes = [v['n_interaction_tubes'] for v in m.values()]
clips_with = sum(1 for t in tubes if t > 0)
print(f'D_I: {clips_with}/{len(tubes)} clips ({100*clips_with/len(tubes):.0f}%)  Total tubes: {sum(tubes)}')
"
```

---

## Step C: V-JEPA 2.1 Frozen Embedding — 100-clip dense subset

```bash
python -u src/m05_vjepa_embed.py --POC \
    --subset data/sanity_100_dense.json \
    --model-config configs/model/vjepa2_1.yaml \
    --encoder vjepa_2_1_frozen \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_dense100_frozen.log
```

**Verify:**
```bash
python3 -c "
import numpy as np
e = np.load('outputs/poc/embeddings_vjepa_2_1_frozen.npy')
print(f'Shape: {e.shape}')   # expect (100, 1664)
print(f'Range: [{e.min():.2f}, {e.max():.2f}]')
"
```

---

## Step D: ExPLoRA Training — 100-clip dense subset

```bash
python -u src/m09_pretrain.py --POC \
    --subset data/sanity_100_dense.json \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/explora.yaml \
    --explora --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09_dense100_explora.log
```

**Verify:** `ls -lh outputs/poc/m09_pretrain/explora/student_encoder.pt` — exists, ~8 GB

---

## Step E: Surgery Training — 100-clip dense subset

```bash
python -u src/m09_pretrain.py --POC \
    --subset data/sanity_100_dense.json \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/ch11_surgery.yaml \
    --surgery --factor-dir outputs/poc/m11_factor_datasets/ \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09_dense100_surgery.log
```

**Verify:** `ls -lh outputs/poc/m09_pretrain/surgery/student_encoder.pt` — exists, ~8 GB

**Subset = 100 clips from `data/sanity_100_dense.json`** (density-scored: traffic + crowd + agent tags, 73 tier1 + 26 tier2 + 1 goa). Same subset across Steps A-E so Prec@K comparisons (frozen vs ExPLoRA vs surgical) are apples-to-apples.

---

## POC (1K clips) — after all SANITY steps pass

```bash
./scripts/train_explora.sh --POC 2>&1 | tee logs/explora_poc.log
./scripts/train_surgery.sh --POC 2>&1 | tee logs/surgery_poc.log
```

**Result:**
```bash
python3 -c "
import json
for name in ['frozen', 'explora', 'surgical']:
    f = f'outputs/poc/m06_metrics_vjepa_2_1_{name}.json'
    try:
        m = json.load(open(f))
        pk = m['easy']['precision_at_k']
        print(f'{name:12s}: Prec@K = {pk[\"mean\"]:.1f}% +/- {pk[\"ci\"][\"ci_half\"]:.1f}')
    except FileNotFoundError:
        print(f'{name:12s}: NOT YET RUN')
"
```
