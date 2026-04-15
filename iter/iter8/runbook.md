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

## Step A: Grounded-SAM Segmentation (DINO + SAM 3.1)

```bash
rm -rf outputs/poc/m10_sam_segment/ outputs/poc/m11_factor_datasets/
python -u src/m10_sam_segment.py --POC \
    --subset data/sanity_100_dense.json \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m10_dense100_level2_v1.log
```

**Verify:** `cat outputs/poc/m10_sam_segment/summary.json | python3 -m json.tool`

| Check | Expect (100 dense clips) |
|---|---|
| `quality_gate` | `"PASS"` |
| `quality_gate_checks` | All 4 checks PASS |
| `mean_agent_pixel_ratio` | 2-15% |
| `mean_mask_confidence` | >= 0.85 |
| `clips_with_agents_pct` | >= 90% |
| `n_total_interactions` | > 200 |
| `pipeline` | `"grounded-sam"` |
| `m10_overlay_verify/*.png` | Red masks on real agents, no FPs on wires/signage |

## Step A.2:
```bash
./setup_env_uv.sh --gpu --from-wheels 
# Then smoke-test 3 load paths in the upgraded venv before running v2_HF on POC:
source venv_walkindia/bin/activate && python3 -c "
from transformers import Sam3TrackerVideoModel, Sam3VideoModel, AutoModelForZeroShotObjectDetection, Qwen3VLForConditionalGeneration
print('Sam3Tracker OK, Sam3Video OK, DINO OK, Qwen3VL OK')" 

# If those 4 imports succeed, we're clear to run: 
                         
python -u src/m10_sam_segment_v2_HF.py --POC \
--subset data/sanity_100_dense.json \
--local-data data/val_1k_local \
--no-wandb --probe-p3a 5 \
2>&1 | tee logs/m10_v2HF_dense100_probe5.log
```

---

## Step B: Factor Datasets (D_L + D_A + D_I)

```bash
python -u src/m11_factor_datasets.py --POC \
    --subset data/sanity_100_dense.json \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m11_dense100_level2_v1.log
```

**Verify:**

| Check | Expect (100 dense clips) |
|---|---|
| `m11_factor_samples.png` | D_L: agents visibly blurred, layout (signage/buildings) sharp. D_A: agents bright, BG dimmed to 10% |
| `m11_interaction_samples.png` | Tight crops around 2+ agents (cross-category pairs: pedestrian × motorcycle, car × bus, etc.) |
| `m11_factor_stats.png` | Agent ratio bell curve, mode at 2-8% (denser scenes than random val_1k) |
| `m11_per_clip_verify/*.png` | 2x2 stills — Original \| D_L (blurred agents) \| D_A (isolated agents) \| D_I (tube crop) |
| `m11_per_Videoclip_verify/*.mp4` | 2x2 H.264 videos, top 20 clips, 16 frames animated, 960x540 @ 6fps |
| Console: `D_I quality` line | >= 90% clips have tubes |
| Console: `D_A: N files` | >= 90 of 100 clips |
| Mid-frame coverage | Masks 15-25% consistently across all 16 frames |

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
    --surgery --factor-dir outputs/poc/m11_factor_datasets/ \
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
