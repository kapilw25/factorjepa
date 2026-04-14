# FactorJEPA Runbook

> Run these commands on GPU. Verify each step before moving to the next.
> Architecture: Grounded-SAM (Grounding DINO box detection + SAM 3.1 mask refinement & propagation), Path D text+boxes hybrid. See `errors_N_fixes.md` #20-27 for pivot history.

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

## Step A: Grounded-SAM Segmentation (DINO + SAM 3.1)

```bash
rm -rf outputs/sanity/m10_sam_segment/
python -u src/m10_sam_segment.py --SANITY \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m10_sanity_v5.log
```

**Verify:** `cat outputs/sanity/m10_sam_segment/summary.json | python3 -m json.tool`

| Check | Expect (Grounded-SAM v5 distribution) |
|---|---|
| `quality_gate` | `"PASS"` |
| `quality_gate_checks` | All 4 checks PASS |
| `mean_agent_pixel_ratio` | 0.5-15% (gate: >=0.2%, <=50%) — varies by scene density |
| `mean_mask_confidence` | >= 0.4 (typical: 0.85-0.95 with DINO box anchoring) |
| `clips_with_agents_pct` | >= 50% (8/20 truly-empty Goa/monument clips correctly skip) |
| `n_total_interactions` | > 0 (typical: 30-50 across 20 SANITY clips) |
| `pipeline` | `"grounded-sam"` (confirms Path D, not legacy SAM3-text) |
| `m10_overlay_verify/*.png` | Red masks on real agents (people/vehicles), no FPs on wires/signage |

---

## Step B: Factor Datasets (D_L + D_A + D_I)

```bash
rm -rf outputs/sanity/m11_factor_datasets/
python -u src/m11_factor_datasets.py --SANITY \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m11_sanity_v5.log
```

**Verify:**

| Check | Expect (Grounded-SAM v5) |
|---|---|
| `m11_factor_samples.png` | D_L: agents visibly blurred, layout (signage/buildings) sharp. D_A: agents bright, BG dimmed to 10% |
| `m11_interaction_samples.png` | Tight crops around 2+ agents (cross-category pairs valuable: pedestrian × motorcycle, etc.) |
| `m11_factor_stats.png` | Agent ratio bell curve, mode at 1-5% (precise tight masks) |
| `m11_per_clip_verify/*.png` | 2x2 grids: Original \| D_L (blurred agents) \| D_A (isolated agents) \| D_I (tube crop) |
| Console: `D_I quality` line | 30-60% clips have tubes (45% on SANITY v5) |
| Console: `D_A: N files` | 50-65% of clips (12/20 on SANITY v5; the rest are truly-empty scenes) |

**D_I note:** D_I depends on SAM 3.1 cross-frame TRACKING (now working via Path D text+boxes hybrid). If D_I returns 0 tubes after a fresh m10 run, check `segments.json[clip]["n_interactions"]` — if 0 there too, agents weren't tracked across ≥4 consecutive frames. Tune `max_distance_frame_fraction` and `min_overlap_frames` in `configs/train/ch11_surgery.yaml` — 15 second fix.

```bash
python3 -c "
import json
m = json.load(open('outputs/sanity/m11_factor_datasets/factor_manifest.json'))
tubes = [v['n_interaction_tubes'] for v in m.values()]
clips_with = sum(1 for t in tubes if t > 0)
print(f'D_I: {clips_with}/{len(tubes)} clips have tubes ({100*clips_with/len(tubes):.0f}%)')
print(f'Total tubes: {sum(tubes)}')
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
    --surgery --factor-dir outputs/sanity/m11_factor_datasets/ \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09_surgery_sanity.log
```

**Verify:** `ls -lh outputs/sanity/m09_pretrain/surgery/student_encoder.pt` — exists, ~8 GB

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
