# FactorJEPA Runbook
> **High level Goal: Surgery > ExPLoRA > Frozen on Prec@K, 115K clips.**
> **Mid level Goal: Surgery > ExPLoRA > Frozen on Prec@K, 1K clips**
> **Low level Goal: Surgery > Frozen on Prec@K with non-overlapping 95 % CIs, 100 dense clips**

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

## Step C: V-JEPA 2.1 Frozen Embedding — 100-clip dense subset ✅ validated 2026-04-15

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

### D.1 — SANITY (20 clips, ~15 min, code smoke test + multi-stage transitions)

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

**Verify:** `ls -lh outputs/sanity/m09c_surgery/student_encoder.pt` — exists, ~8 GB. Check log for 3 stage transitions (`Stage 1/2/3`), 3 optimizer rebuilds, non-NaN losses across stages.

### D.2 — POC (100-clip dense subset, ~3h, real training signal)

```bash
rm -rf outputs/poc/m09c_surgery/
python -u src/m09c_surgery.py --POC \
    --subset data/sanity_100_dense.json \
    --model-config configs/model/vjepa2_1.yaml \
    --train-config configs/train/ch11_surgery.yaml \
    --factor-dir outputs/poc/m11_factor_datasets/ \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m09c_dense100_surgery.log
```

**Verify:** `ls -lh outputs/poc/m09c_surgery/student_encoder.pt` — exists, ~8 GB

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
