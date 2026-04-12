# FactorJEPA Runbook

> Run these commands on GPU. Verify each step before moving to the next.
> Future work, ablations, troubleshooting: see `next_steps.md`

---

## GPU Setup (one-time, bare instance)

```bash
git clone https://github.com/kapilw25/factorjepa.git && cd factorjepa
./setup_env_uv.sh --gpu --from-wheels 2>&1 | tee logs/setup_gpu.log
source venv_walkindia/bin/activate
./git_pull.sh                          # downloads outputs + data from HF
```

---

## Step A: SAM 3.1 Segmentation

```bash
python -u src/m10_sam_segment.py --SANITY \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m10_sanity.log
```

**Verify:** `cat outputs/sanity/factors/summary.json | python3 -m json.tool`

| Check | Expect |
|---|---|
| `quality_gate` | `"pass"` |
| `mean_agents_per_clip` | 2-15 |
| `mean_agent_pixel_ratio` | 5-40% |
| `mean_concept_recall` | > 0.5 |
| `m10_segmentation_samples.png` | Red masks on people/vehicles, blue on roads |

---

## Step B: Factor Datasets (D_L + D_A + D_I)

```bash
python -u src/m11_factor_datasets.py --SANITY \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m11_sanity.log
```

**Verify:**

| Check | Expect |
|---|---|
| `m11_factor_samples.png` | D_L: people blurred, roads sharp. D_A: people sharp, BG dark |
| `m11_interaction_samples.png` | Tight crops around 2+ agents near each other (see D_I note below) |
| `m11_factor_stats.png` | Agent ratio bell curve ~10-25% |
| Console: `D_I quality` line | 30-60% clips have tubes |

**D_I note:** D_I is NOT a SAM masking problem. SAM produces the same agent masks for all 3 factors. D_I finds pairs of agents whose centroids come within 20% of frame width for 4+ consecutive frames, then crops a bounding box around both. This is geometry logic in `m10:mine_interactions()`, not SAM quality. If D_I tubes look wrong, tune `max_distance_frame_fraction` and `min_overlap_frames` in `configs/train/ch11_surgery.yaml` — 15 second fix.

```bash
python3 -c "
import json
m = json.load(open('outputs/sanity/factors/factor_manifest.json'))
tubes = [v['n_interaction_tubes'] for v in m.values()]
clips_with = sum(1 for t in tubes if t > 0)
print(f'D_I: {clips_with}/{len(tubes)} clips have tubes ({100*clips_with/len(tubes):.0f}%)')
"
```

---

## Step C: V-JEPA 2.1 Frozen Embedding

```bash
python -u src/m05_vjepa_embed.py --SANITY \
    --model-config configs/model/vjepa2_1.yaml \
    --encoder vjepa_2_1_frozen \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_sanity_2_1.log
```

**Verify:**
```bash
python3 -c "
import numpy as np
e = np.load('outputs/sanity/embeddings_vjepa_2_1_frozen.npy')
print(f'Shape: {e.shape}')   # (20, 1664)
print(f'Range: [{e.min():.2f}, {e.max():.2f}]')
"
```

---

## Step D: ExPLoRA Training

```bash
python -u src/m09_pretrain.py --SANITY \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/explora.yaml \
    --explora --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09_explora_sanity.log
```

**Verify:** `ls -lh outputs/sanity/student_encoder.pt` — exists, ~8 GB

---

## Step E: Surgery Training

```bash
python -u src/m09_pretrain.py --SANITY \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/ch11_surgery.yaml \
    --surgery --factor-dir outputs/sanity/factors/ \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09_surgery_sanity.log
```

**Verify:** `ls -lh outputs/sanity/student_encoder.pt` — exists, ~8 GB

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
