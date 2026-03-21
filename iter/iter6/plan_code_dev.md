# Temporal Evaluation Extension + Bug Fix — Code Plan

> **Context**: Ch9 spatial metrics (Prec@K, mAP@K, Overlap@K, nDCG@K) all favor DINOv2. V-JEPA only wins Cycle@K. But the taxonomy has **zero temporal fields** — V-JEPA's proven strengths (SSv2 77% vs 54%, robotics 65-80% vs 15%) are untested. The problem isn't V-JEPA — **the problem is our metrics don't measure what V-JEPA is good at.**

---

## Priority 2: m06/m08 Plot Overwrite Bug — DONE

- [x] `m06_faiss_metrics.py` — added `sfx` to all 9 plot filenames + `generate_plots(sfx=sfx)`
- [x] `m08_plot.py` — added `--encoder` flag, `get_encoder_files()` for inputs, `enc_sfx` in all 5 plot functions
- [ ] `run_ch9_overnight.sh` — pass `--encoder` to m08 in per-encoder loop

---

## Priority 1: Temporal Evaluation Extension

### What we're testing

V-JEPA encodes 64 frames temporally. DINOv2/CLIP see 1 frame. If V-JEPA's temporal encoding is meaningful, it should outperform image baselines on metrics that require temporal understanding.

### Modules

```
m04d (BUILT)        → motion_features.npy per clip (GPU-RAFT, AdaptiveBatchSizer)
m06b (BUILT)        → temporal correlation metrics per encoder (CPU)
m08b extension      → add temporal axis to radar/bar/LaTeX (TODO)
```

---

### Step 1: `m04d_motion_features.py` — DONE

GPU-RAFT optical flow → 13D features per clip. Batched inference via `AdaptiveBatchSizer`.

```bash
python -u src/m04d_motion_features.py --SANITY --subset data/subset_10k.json \
    --local-data data/subset_10k_local 2>&1 | tee logs/m04d_sanity.log
python -u src/m04d_motion_features.py --FULL --subset data/subset_10k.json \
    --local-data data/subset_10k_local 2>&1 | tee logs/m04d_motion.log
```

Output: `motion_features.npy` (N, 13) + `motion_features.paths.npy` + `motion_features_meta.json`

---

### Step 2: `m06b_temporal_corr.py` — BUILT, needs 2 new metrics

Currently computes 3 metrics per encoder:

| Metric | What it measures | Needs m04d? |
|--------|-----------------|-------------|
| `spearman_rho` | Correlation: embedding distance vs motion distance | Yes |
| `temporal_prec_at_k` | % kNN neighbors in same motion quartile | Yes |
| `motion_retrieval_map` | mAP where relevant = same motion cluster | Yes |

**Two new metrics to add** (zero GPU cost, use existing embeddings):

| Metric | What it measures | Needs m04d? | Expected result |
|--------|-----------------|-------------|-----------------|
| `temporal_order_sensitivity` | L2 distance between normal V-JEPA and shuffled V-JEPA embeddings per clip | **No** | V-JEPA: LARGE (order matters). DINOv2/CLIP: ~0 (single frame) |
| `temporal_locality` | Intra-video embedding coherence vs inter-video distance | **No** | V-JEPA: HIGH (temporal continuity). DINOv2: LOW (single frame) |

**Why these matter**: Metrics 1-3 require m04d to run first (GPU). Metrics 4-5 use only existing embeddings — they can validate temporal encoding immediately with zero new compute.

```bash
python -u src/m06b_temporal_corr.py --encoder vjepa --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m06b_vjepa.log

for enc in random dinov2 clip vjepa_shuffled; do
    python -u src/m06b_temporal_corr.py --encoder $enc --FULL --subset data/subset_10k.json \
        2>&1 | tee logs/m06b_${enc}.log
done
```

---

### Step 3: VLM Temporal Tags — BLOCKED (VLM temporal unreliable)

VLM temporal understanding confirmed unreliable: MotionBench <60%, CameraBench ~50% AP, SpookyBench 0%.

**If pursued**: fine-tune Qwen3-VL on ~1,400 annotated clips (doubles AP per CameraBench). Never out-of-the-box.

---

### Step 4: m08b Extension — TODO (after Step 2 validates)

Add temporal metrics to radar chart + bar chart + LaTeX table.

---

---

## Bootstrap 95% CI — DONE

All metrics now include publication-quality 95% CI (BCa method, 10K bootstrap iterations via `scipy.stats.bootstrap`).

### `utils/bootstrap.py` (NEW)

Shared utility. Resamples queries (clips), not pairs (standard IR practice per Sakai SIGIR 2006).

- `bootstrap_ci(per_query_scores)` → `{mean, ci_lo, ci_hi, ci_half}`
- `per_clip_prec_at_k()`, `per_clip_map_at_k()`, `per_clip_cycle_at_k()`, `per_clip_ndcg_at_k()` — per-clip score arrays

### `m06_faiss_metrics.py`

- CI computed in `compute_all_metrics()` on Prec@K, mAP@K, Cycle@K, nDCG@K
- Stored in JSON: `easy.ci.prec_at_k.{mean, ci_lo, ci_hi, ci_half}`
- Terminal: `Prec@K: 50.5% ± 2.1`

### `m06b_temporal_corr.py`

- Bootstrap CI on Spearman rho (resample distance pairs, recompute correlation)
- Stored as `spearman_rho_ci.{mean, ci_lo, ci_hi, ci_half}`

### `m08b_compare.py`

- Bar chart: error bars via `yerr=ci_half` + `capsize=3`
- LaTeX table: `50.5{\tiny$\pm$2.1}` format
- Reads CI from `m06_metrics{sfx}.json` → `easy.ci.{metric}.ci_half`

### Enforcement

Added to `src/CLAUDE.md` rule 7.4: every metric in JSON/plots/tables MUST include 95% CI. No point estimates without CI.

---

## Execution Order

```
Priority 2 (bug fix)         → DONE (m06 suffix + m08 --encoder)
Step 1 (m04d)                → BUILT, needs GPU run
Step 2 (m06b)                → BUILT, needs 2 new metrics (temporal_order_sensitivity, temporal_locality)
Step 3 (VLM tags)            → BLOCKED (unreliable without fine-tuning)
Step 4 (m08b plots)          → TODO (after Step 2 results)
```

---

## Honest Assessment

**Risk**: Even with GPU-RAFT, 480p compressed video may limit flow quality. If correlation is weak across ALL encoders, either (a) flow features are too noisy, or (b) none of these encoders truly capture temporal structure.

**Bet**: V-JEPA will show higher temporal correlation than DINOv2/CLIP on metrics 1-3 (motion correlation) and significantly higher on metrics 4-5 (temporal order + locality). But the gap may be smaller than expected — Ch9 showed shuffled > normal V-JEPA, suggesting temporal encoding is confused by Indian street chaos.
