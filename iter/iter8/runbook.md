# FactorJEPA Runbook
> **High level Goal: Surgery > ExPLoRA > Frozen on Prec@K, 115K clips.**
> **Mid level Goal: Surgery > ExPLoRA > Frozen on Prec@K, 1K clips (val_1k) — the POC decision gate.**
> **Low level Goal (SANITY smoke test only): 3 stages pass end-to-end on 20 clips with non-NaN loss.**

> **Tier ladder (2026-04-17):** SANITY (20 clips, ~60 s code check) → POC (1K clips / val_1k, ~6-7 h full pipeline) → FULL (115K clips, ~1.5 d on 96GB). The 100-dense-clip intermediate tier was validated on 2026-04-17 and then retired — 3200 visits-per-clip at 100 scale produced overfitting-pressure Prec@K that wouldn't replicate at FULL. 1K × 20 epochs = 20 visits/clip is the cheapest scale where Prec@K signal is publishable.

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

## Step A: Grounded-SAM Segmentation — DINO + HF Sam3TrackerVideo

POC tier = 1000 clips from `data/val_1k.json`. Pipeline code validated on 100-dense-clip tier 2026-04-17 (6141 agents, 8712 interactions, quality_gate PASS at 6.13 s/clip); now scaled up.

```bash
rm -rf outputs/poc/m10_sam_segment/ outputs/poc/m11_factor_datasets/
python -u src/m10_sam_segment.py --POC \
    --subset data/val_1k.json \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m10_1k_v1.log
```

**Verify:**
```bash
cat outputs/poc/m10_sam_segment/summary.json | python3 -m json.tool
```

| Check | Expect (1000 clips val_1k, projected from 6.13 s/clip on 96GB Blackwell) |
|---|---|
| Wall time | ~6100 s = ~**102 min** (1000 × 6.13 s/clip; random-sample clips may run slightly faster than dense subset) |
| `n_total_agents` | ~40K-60K (extrapolated from 6141 on 100 dense; val_1k is random so lighter than dense subset) |
| `n_total_interactions` | ~60K-90K |
| `mean_agent_pixel_ratio` | 0.10-0.18 (lower than the 0.186 dense measurement — val_1k has some empty aerial/monument clips) |
| `mean_mask_confidence` | ≥ 0.85 |
| `clips_with_agents_pct` | ≥ 0.80 (val_1k random sample; some truly empty clips get correctly skipped) |
| `quality_gate` | `"PASS"` (4 checks) |
| `m10_overlay_verify/*.png` | 1000 saved; spot-check 20 — red masks on real agents, no FPs on wires/signage |

---

## Step B: Factor Datasets (D_L + D_A + D_I with tight-bbox tubes) ✅ validated 2026-04-15

```bash
rm -rf outputs/poc/m11_factor_datasets/
python -u src/m11_factor_datasets.py --POC \
    --subset data/sanity_100_dense.json \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m11_dense100_v1.log
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

## Step C: V-JEPA 2.1 Frozen Embedding — 100-clip dense subset ✅ validated 2026-04-15

```bash
python -u src/m05_vjepa_embed.py --POC \
    --subset data/sanity_100_dense.json \
    --model-config configs/model/vjepa2_1.yaml \
    --encoder vjepa_2_1_frozen \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_dense100_frozen_v1.log
```

**Verify:**
```bash
python3 -c "
import numpy as np
e = np.load('outputs/poc/m05_vjepa_embed/embeddings_vjepa_2_1_frozen.npy')
print(f'Shape: {e.shape}')   # expect (100, 1664)
print(f'Range: [{e.min():.2f}, {e.max():.2f}]')
print(f'L2 norm mean: {np.linalg.norm(e, axis=1).mean():.2f}')"
```

| Check | Expect (100 dense clips, measured 2026-04-15) |
|---|---|
| Wall time (model load + 100 clips) | ~232 s (0.43 clips/s on 24GB Blackwell) |
| Per-clip rate | ~2.3 s/clip steady-state (after compile warmup) |
| `embeddings_vjepa_2_1_frozen.npy` shape | `(100, 1664)` |
| AdaptiveBatchSizer growth | 8 → 14 (VRAM 36% → 65%, target=85%, max=44) |
| OOM events | 0 (sizer kept VRAM headroom) |
| Compile path | `torch.compile` ✅ (RoPE Q/K cast to V.dtype, #44) |
| Model dtype | `bfloat16` (V-JEPA 2.1 native, #44) |
| Branch routing | `frozen-native` (loads `target_encoder` from .pt, #42) |

---

## Step D: Surgery Training — `src/m09c_surgery.py` 🎯 PRIMARY PATH (paper novelty)

> m09c = 3-stage progressive prefix unfreezing + factor datasets (D_L → D_A → D_I). No drift, no held-out val.
>
> **Why Step D (not E)**: immediate goal is `Surgery > Frozen` on Prec@K. D_L/D_A/D_I factors already built in Step B. Test this path FIRST — if it works, we have the paper result. ExPLoRA (Step E) is the comparison arm run AFTER Surgery is validated.

### D.1 — SANITY ✅ validated 2026-04-17 on 96 GB Blackwell

**Result:** 3 stages passed end-to-end in ~60 s (Stage 1 loss=0.4870, Stage 2 loss=0.4901, Stage 3 loss=0.4806). `student_encoder.pt` exported. Stage 3 — which OOMed on 24 GB at v7 — used only 19.9 / 102 GB VRAM on 96 GB, confirming errors_N_fixes.md #58's "no v8 patch needed, move to 96 GB" decision.

> Requires `outputs/sanity/m11_factor_datasets/` from Step B run in `--SANITY` mode. If not present, re-run Step B with `--SANITY`.

```bash
rm -rf outputs/sanity/m09c_surgery/
python -u src/m09c_surgery.py --SANITY \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/ch11_surgery.yaml \
    --factor-dir outputs/sanity/m11_factor_datasets/ \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09c_sanity_surgery.log
```

**Verify:** 
```bash
ls -lh outputs/sanity/m09c_surgery/student_encoder.pt
``` 
— exists, ~8 GB. Check log for 3 stage transitions (`Stage 1/2/3`), 3 optimizer rebuilds, non-NaN losses across stages.

### D.2 — POC (100-clip dense subset, ~100 min on 96GB, real training signal)

```bash
rm -rf outputs/poc/m09c_surgery/
python -u src/m09c_surgery.py --POC \
    --subset data/sanity_100_dense.json \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/ch11_surgery.yaml \
    --factor-dir outputs/poc/m11_factor_datasets/ \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09c_dense100_surgery_v1.log
```

**Verify:** `ls -lh outputs/poc/m09c_surgery/student_encoder.pt` — exists, ~8 GB. Check `m09_train_loss.png` for 3 color-segmented stage curves (green → orange → purple) + transition vlines at steps 99, 198.

### D.3 — m05 re-embed on surgical student (~8 min on 96GB)

Apply surgery-trained V-JEPA 2.1 to the same 100 dense clips so Prec@K is directly comparable with the frozen baseline (Step C).

```bash
python -u src/m05_vjepa_embed.py --POC \
    --subset data/sanity_100_dense.json \
    --model-config configs/model/vjepa2_1.yaml \
    --model outputs/poc/m09c_surgery/student_encoder.pt \
    --encoder vjepa_2_1_surgical \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m05_surgical_poc.log
```

**Verify:**
```bash
python3 -c "
import numpy as np
e = np.load('outputs/poc/m05_vjepa_embed/embeddings_vjepa_2_1_surgical.npy')
print(f'Shape: {e.shape}')                             # expect (100, 1664)
print(f'L2 norm mean: {np.linalg.norm(e, axis=1).mean():.2f}')"
```

### D.4 — m06 FAISS Prec@K metrics — decision gate (~2 min, FAISS-GPU)

Two m06 runs (frozen + surgical) to close the 🎯 decision gate. m06 takes `--encoder` (NOT `--local-data` — chain script error, see fix in #60). Per-encoder JSON lands at `outputs/poc/m06_metrics_vjepa_2_1_*.json`.

```bash
# Frozen baseline (re-run — already computed in Step C flow but m06 may not have been invoked)
python -u src/m06_faiss_metrics.py --POC \
    --subset data/sanity_100_dense.json \
    --encoder vjepa_2_1_frozen \
    --no-wandb \
    2>&1 | tee logs/m06_frozen_poc.log

# Surgical — the paper-arm result
python -u src/m06_faiss_metrics.py --POC \
    --subset data/sanity_100_dense.json \
    --encoder vjepa_2_1_surgical \
    --no-wandb \
    2>&1 | tee logs/m06_surgical_poc.log
```

**Verify — decision gate:**
```bash
python3 -c "
import json
for name in ['frozen', 'surgical']:
    try:
        m = json.load(open(f'outputs/poc/m06_metrics_vjepa_2_1_{name}.json'))
        pk = m['easy']['precision_at_k']
        print(f'{name:12s}: Prec@K = {pk[\"mean\"]:.2f}% +/- {pk[\"ci\"][\"ci_half\"]:.2f}')
    except FileNotFoundError:
        print(f'{name:12s}: NOT YET RUN')
"
```

- **Surgery > Frozen (non-overlapping 95 % CIs)** → proceed to Step E (ExPLoRA comparison arm).
- **Surgery ≤ Frozen** → pause Step E; debug m10/m11 factor quality or revisit training config (warmup, epoch count — see #60).

---

## Step E: ExPLoRA Training — `src/m09b_explora.py` (comparison arm)

> m09b = LoRA on blocks 2-47 + unfreeze blocks 0-1, no drift. Hardcoded ExPLoRA mode (no `--explora` flag).
>
> **Why Step E (not D)**: ExPLoRA is the adaptation-baseline comparator for `Surgery > ExPLoRA > Frozen`. Only valuable AFTER Step D Surgery passed — it completes the comparison triangle. If Surgery already ≤ Frozen, pause and debug factor quality before spending GPU on ExPLoRA.

### E.1 — SANITY (20 clips, ~10 min, code smoke test before spending POC GPU time)

```bash
rm -rf outputs/sanity/m09b_explora/
python -u src/m09b_explora.py --SANITY \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/explora.yaml \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09b_sanity_explora.log
```

**Verify:** `ls -lh outputs/sanity/m09b_explora/student_encoder.pt` — exists, ~8 GB. Check log for `LoRA injection`, non-NaN `loss_jepa`, clean exit.

### E.2 — POC (100-clip dense subset, ~1.5h, real training signal)

```bash
rm -rf outputs/poc/m09b_explora/
python -u src/m09b_explora.py --POC \
    --subset data/sanity_100_dense.json \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/explora.yaml \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09b_dense100_explora.log
```

**Verify:** `ls -lh outputs/poc/m09b_explora/student_encoder.pt` — exists, ~8 GB

**Subset = 100 clips from `data/sanity_100_dense.json`** (density-scored: traffic + crowd + agent tags, 73 tier1 + 26 tier2 + 1 goa). Same subset across Steps A-E so Prec@K comparisons (frozen vs ExPLoRA vs surgical) are apples-to-apples.

> **SANITY vs POC rationale**: SANITY (D.1 / E.1) validates the code path (no crashes, non-NaN losses, checkpoint saves) on 20 clips in minutes — cheap smoke test. POC (D.2 / E.2) on 100 dense clips is the actual training signal where loss curves and Prec@K deltas become interpretable. Run SANITY first, then POC only if SANITY passes.
>
> **🎯 Decision gate after D.2 (Surgery POC)**: if Prec@K(surgery) > Prec@K(frozen) with non-overlapping 95 % CIs → we have the paper result; proceed to E.1/E.2 for the full `Surgery > ExPLoRA > Frozen` comparison + ablations. If Surgery ≤ Frozen → **pause Step E** and debug factor quality (m10/m11 re-examination) FIRST — ExPLoRA offers no insight into a broken factoring signal.

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
