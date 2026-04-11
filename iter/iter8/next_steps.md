# Next Steps (Week 1)

> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**
> Ch11 surgery is the PRIMARY path. ExPLoRA is the baseline to beat. Ch10 brute-force is deferred.
> **Full plan:** `iter/iter8/plan_training.md`

---

## Step 1a (parallel, during SAM3 prep): Temporal Interference Projection [30 min CPU]

- PCA on `(normal_embedding - shuffled_embedding)` for 10K clips
- Project V-JEPA 2.1 embeddings orthogonal to top components
- Re-run Prec@K on projected embeddings
- **Parallel diagnostic** — run while SAM3 processes 100 clips for Step 2
- If Prec@K recovers → paper novelty (Contribution 2)

## Step 1b (parallel): ExPLoRA on V-JEPA 2.1 [3h GPU — baseline to beat]

- Freeze all blocks EXCEPT blocks 0-1. Add LoRA (rank 8-16) to all other layers
- Continue JEPA self-supervised pretraining on 10K Indian clips
- Compare Prec@K: frozen vs ExPLoRA-adapted
- **Sets the bar** — Ch11 surgery must beat this to justify its complexity
- Ref: [ExPLoRA (ICML 2025)](https://arxiv.org/abs/2406.10973) — +8% on DINOv2 domain shift with <10% params

## Step 2: Ch11 Factor Surgery POC on V-JEPA 2.1 [3h GPU + SAM3 prep]

**Simplified POC: 2 factors (D_L + D_A), skip D_I for now.** Interaction mining is the most complex part and least likely to affect spatial Prec@K. Add D_I later if 2-factor POC works.

- **SAM3 segmentation** on 100 clips → instance masks → tracklets → agent vs layout separation (motion filter)
- **Factor datasets**: D_L (layout-only, blur agents), D_A (agent-only, suppress background)
- **2-stage progressive prefix unfreezing** (simplified from Sec 11.5):
  - Stage 1: unfreeze layers 0→25%L, train on 100% D_L (layout — roads, buildings, wires)
  - Stage 2: unfreeze layers 0→50%L, train on 90% D_A + 10% D_L replay (agents — people, vehicles, animals)
- Dense loss (all tokens) + deep self-supervision (4 layers)
- Compare Prec@K: frozen vs Ch11-adapted vs ExPLoRA-adapted (from Step 1b)
- **THE KEY COMPARISON: does factor decomposition beat standard ExPLoRA?**

### If Step 2 fails: debug sequence (before declaring surgery dead)

1. **Reverse order (D_A → D_L)** — adapts agent features first (most visually different from Western data)
2. **All factors simultaneously** (no ordering) — isolates whether the problem is ordering vs factor decomposition itself
3. **Add D_I (interactions)** — 3-stage with interaction tubes (Sec 11.2-11.3 of proposal)
4. **More clips** (1K instead of 100) — POC may need more data for 2B model

### If Step 2 succeeds: scale up

- Full Ch11 pipeline on 10K clips: SAM3 → D_L + D_A + D_I → 3-stage progressive unfreezing
- **Target: statistically significant Prec@K improvement with 95% CI**

---

## Decision Gate (end of Week 1)

| Step 1a projection | Step 1b ExPLoRA | Step 2 Surgery | Action |
|---|---|---|---|
| Prec@K jumps | ExPLoRA improves | Surgery > ExPLoRA | **Strongest: all 3 methods + factor surgery wins** |
| Any | ExPLoRA improves | Surgery = ExPLoRA | **Publish ExPLoRA result, surgery adds no value** |
| Any | No change | Surgery improves | **Best novelty: standard fails, surgery succeeds** |
| Any | ExPLoRA improves | Surgery < ExPLoRA | **Publish ExPLoRA, drop surgery** |
| Any | No change | No change | **Debug: reverse order, simultaneous factors, more clips** |

---

> **Why both ExPLoRA AND Surgery?** ExPLoRA is the current SOTA for domain adaptation (<10% params, ICML 2025). If surgery can't beat ExPLoRA, it's not worth the complexity. If surgery beats ExPLoRA, THAT is the paper.
>
> **Why D_L→D_A order?** Curriculum learning: simple (layout) before complex (agents). Standard transfer learning says the opposite (adapt high-level first). The order is an ablation concern — the FIRST question is whether ANY factor decomposition helps. If POC fails, reverse the order before giving up.
