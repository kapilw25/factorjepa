# Temporal Evaluation Extension + Bug Fix — Code Plan

> **Context**: Ch9 spatial metrics all favor DINOv2. V-JEPA only wins Cycle@K. Taxonomy has zero temporal fields. The problem isn't V-JEPA — **our metrics don't measure what V-JEPA is good at.**

---

## Priority 2: m06/m08 Plot Overwrite Bug — DONE

- [x] `m06_faiss_metrics.py` — `sfx` on all 9 plot filenames
- [x] `m08_plot.py` — `--encoder` flag + `enc_sfx` on all 5 plot functions
- [ ] `run_ch9_overnight.sh` — pass `--encoder` to m08

---

## Priority 1: Temporal Evaluation — DONE

- [x] `m04d_motion_features.py` — GPU-RAFT 13D features, AdaptiveBatchSizer
- [x] `m06b_temporal_corr.py` — 5 temporal metrics (3 motion-based + 2 embedding-only)
- [x] `m08b_compare.py` — grouped bar + tradeoff scatter + error bars + ±CI LaTeX

---

## Bootstrap 95% CI — DONE

- [x] `utils/bootstrap.py` — `scipy.stats.bootstrap` BCa 10K iters
- [x] `m06` — CI on Prec@K, mAP@K, Cycle@K, nDCG@K in JSON
- [x] `m06b` — CI on Spearman rho
- [x] `m08b` — error bars + `±CI` in LaTeX table
- [x] `CLAUDE.md` rule 7.4 — mandatory CI enforcement

---

## m08b Visualization — DONE

4 output figures (all with 95% CI where applicable):

| Figure | File | What |
|--------|------|------|
| Grouped bar (spatial vs temporal) | `m08b_spatial_temporal_bar.png` | Spatial metrics left, temporal right, per encoder |
| Tradeoff scatter | `m08b_tradeoff_scatter.png` | X=spatial aggregate, Y=temporal aggregate, per encoder |
| Original bar (Easy vs Hard) | `m08b_encoder_comparison.png` | Existing, now with error bars |
| Radar (combined) | `m08b_radar.png` | Existing spatial axes |

---

## Execution Order

```
Priority 2 (bug fix)         → DONE
Step 1 (m04d)                → BUILT, needs GPU run
Step 2 (m06b 5 metrics)      → DONE (metrics 4-5 run now, 1-3 after m04d)
Step 3 (VLM tags)            → BLOCKED (unreliable)
Step 4 (m08b plots)          → DONE (grouped bar + scatter + CI)
Bootstrap CI                 → DONE (all metrics)
```

### What's left before GPU run

- [ ] `run_ch9_overnight.sh` — add m04d step + m06b loop + pass `--encoder` to m08

---

## TODO: Rename V-JEPA files to use `_vjepa` suffix (post-Ch9)

V-JEPA files use empty suffix (`embeddings.npy`, `knn_indices.npy`, `umap_2d.npy`) while all other encoders use `_encodername` suffix. This is legacy backward compat from when V-JEPA was the only encoder.

**Scope**: ~10 files — `config.py` ENCODER_REGISTRY, `m05_vjepa_embed.py`, `m05c_true_overlap.py`, `m06_faiss_metrics.py`, `m06b_temporal_corr.py`, `m07_umap.py`, `m08_plot.py`, `m08b_compare.py`, `run_evaluate.sh` verify blocks, `plan_execution.md` verify scripts.

**When**: After no-dedup re-run validates. Not during active pipeline runs — one missed reference = silent file-not-found.

**Change**: In `config.py` ENCODER_REGISTRY, change `vjepa` suffix from `""` to `"_vjepa"`. Update all file I/O that constructs paths with suffix.

---

## Honest Assessment

**Risk (original)**: 480p compressed video may limit RAFT flow quality.

**Risk mitigation (verified)**: Risk is overstated. Local clips are 854x480 H.264 CRF28, 30fps — within RAFT's benchmarked sweet spot (Sintel 1024x436, KITTI 1242x375). Three factors mitigate:
1. **Resolution is fine** — RAFT was trained/benchmarked at similar resolution
2. **CRF28 compression artifacts** are smoothed by the 854x480→520x360 downsampling in m04d before RAFT processes them
3. **Aggregate features** (mean/std/max over 360x520 pixels × 16 pairs) average out per-pixel noise — we're not using per-pixel flow maps

If temporal correlation is weak across ALL encoders, it means the encoders genuinely don't capture temporal structure — not a data quality problem.

**Bet**: V-JEPA >> DINOv2/CLIP on temporal metrics 1-3, significantly higher on 4-5. Gap may be smaller than expected — shuffled > normal V-JEPA suggests temporal encoding confused by Indian street chaos.
