# Next Steps (Week 1)

> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**
> **Short-term: Show Surgery > ExPLoRA > Frozen on 1K val clips (~70 min total GPU).**
> Ch11 surgery is the PRIMARY path. ExPLoRA is the baseline to beat. Ch10 brute-force deferred.
> **Full plan:** `iter/iter8/plan_training.md` | **Code plan:** `iter/iter8/plan_code_development.md`

---

## Step 0: Frozen V-JEPA 2.1 Baseline [~12 min GPU, prerequisite]

- Generate frozen V-JEPA 2.1 (2B, 1664-dim) embeddings on 1K val clips
- Evaluate frozen 2.1 Prec@K — this is the NEW baseline to beat
- **Handled automatically by `train_explora.sh` Step 0** (skips if cached)
- Data: `data/val_1k_local/` (1000 clips, ready)

## Step 1a: Temporal Interference Projection [30 min CPU, after Step 0]

- Generate `vjepa_2_1_frozen_shuffled` embeddings (same 2.1 model, frames shuffled)
- PCA on `(vjepa_2_1_frozen - vjepa_2_1_frozen_shuffled)` for 1K clips
- Project V-JEPA 2.1 embeddings orthogonal to top components
- Re-run Prec@K on projected embeddings
- **Diagnostic** — if Prec@K recovers → paper novelty

## Step 1b: ExPLoRA on V-JEPA 2.1 [~1h GPU on 1K clips]

- `./scripts/train_explora.sh --POC` with `--local-data data/val_1k_local`:
  - Step 0: frozen 2.1 embeddings + eval (~12 min, skips if cached)
  - Step 1: ExPLoRA training — LoRA rank=16, unfreeze blocks 0-1 (~20 min)
  - Step 2: re-embed adapted model (~12 min)
  - Step 3: evaluate adapted Prec@K
- Compare: `m06_metrics_vjepa_2_1_frozen.json` vs `m06_metrics_vjepa_2_1_explora.json`
- **Sets the bar** — Ch11 surgery must beat this

## Step 2: Ch11 Factor Surgery POC on V-JEPA 2.1 [~30 min GPU + SAM3.1 prep]

- **SAM 3.1 text-prompted segmentation** (`src/m10_sam_segment.py`) on 1K val clips
  - Per-clip prompt from `data/val_1k_local/tags.json` → `notable_objects` field
  - Example: clip tagged `[auto_rickshaw, pedestrian]` → SAM prompt `"auto_rickshaw, pedestrian"`
  - SAM 3.1 multiplexing tracks all agents across 16 frames
- **Factor datasets** (`src/m11_factor_datasets.py`): D_L (blur agents), D_A (suppress background)
- **2-stage progressive prefix unfreezing** (`--poc-simple`):
  - Stage 1: unfreeze layers 0→25%L, train on D_L (layout)
  - Stage 2: unfreeze layers 0→50%L, train on 90% D_A + 10% D_L replay
- Compare Prec@K: frozen 2.1 vs ExPLoRA vs Surgery
- **THE KEY COMPARISON: does factor decomposition beat ExPLoRA?**

### If Step 2 fails: debug sequence

1. **Reverse order (D_A → D_L)** — agents first (most visually different)
2. **All factors simultaneously** — isolates ordering vs decomposition
3. **Add D_I (interactions)** — 3-stage with interaction tubes
4. **Scale to 10K** — 1K may be insufficient for 2B model

### If Step 2 succeeds: scale up

- Full pipeline on 10K clips → statistically significant Prec@K with 95% CI

---

## Decision Gate (end of Week 1)

| Step 1a projection | Step 1b ExPLoRA | Step 2 Surgery | Action |
|---|---|---|---|
| Prec@K jumps | ExPLoRA improves | Surgery > ExPLoRA | **Strongest: all 3 + surgery wins** |
| Any | ExPLoRA improves | Surgery = ExPLoRA | **Publish ExPLoRA, surgery adds no value** |
| Any | No change | Surgery improves | **Best novelty: standard fails, surgery succeeds** |
| Any | ExPLoRA improves | Surgery < ExPLoRA | **Publish ExPLoRA, drop surgery** |
| Any | No change | No change | **Debug: reverse order, more clips, LoRA fallback** |
