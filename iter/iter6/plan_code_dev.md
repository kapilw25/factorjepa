# Temporal Evaluation Extension + Bug Fix — Code Plan

> **Context**: Ch9 taxonomy v3 has **zero temporal fields** — V-JEPA's temporal encoding is untested. This plan adds temporal evaluation to establish if V-JEPA recovers its advantage on motion-aware metrics.

---

## Priority 2: m06/m08 Plot Overwrite Bug

**Root cause**: m06 saves 10 plots with hardcoded names (`m06_map_per_key.png`, etc.) — no encoder suffix. When `run_ch9_overnight.sh` loops through 5 encoders, the last one (vjepa_shuffled) overwrites all previous plots. Same issue in m08 (no `--encoder` flag at all).

Data files are fine — `m06_metrics{sfx}.json`, `knn_indices{sfx}.npy` already use suffixes via `get_encoder_files()`. **Only plots are broken.**

### Fix Plan

| File | Change | Lines affected |
|------|--------|----------------|
| `m06_faiss_metrics.py` | Add `sfx` to all 10 plot filenames: `m06_distance_hist{sfx}{ext}` | ~10 savefig lines |
| `m08_plot.py` | Add `--encoder` flag, use `get_encoder_files()` for inputs, add suffix to 7-9 plot filenames | ~15 lines |
| `run_ch9_overnight.sh` | Pass `--encoder` to m08 in the per-encoder loop (or run m08 in a loop) | ~5 lines |

**Effort**: ~1 hour.

---

## Priority 1: Temporal Evaluation Extension

### Architecture Overview

```
m04f (NEW)          → motion_features.npy per clip (GPU-RAFT optical flow via torchvision)
m06b (NEW)          → temporal correlation: embedding distance vs motion distance per encoder
m08b extension      → add temporal axis to radar/bar/LaTeX
```

### Step 1: `m04f_motion_features.py` (NEW, ~300 lines)

**Purpose**: Extract deterministic, ground-truth temporal features from video clips using GPU-accelerated RAFT optical flow.

#### Why GPU-RAFT (not CPU-Farneback)?

| Criterion | GPU-RAFT (`torchvision`) | CPU-Farneback (OpenCV) |
|-----------|--------------------------|------------------------|
| Per-pixel accuracy | **SOTA** (Sintel EPE 2.855, 30% over prior best) | Not competitive on modern benchmarks |
| Motion blur handling | **Strong** (12 recurrent refinement iterations) | Weak (single-pass polynomial expansion) |
| Speed at 480p | ~50-100 fps on GPU | ~125 fps on CPU |
| 10K clips (16 pairs each) | **~30-50 min** on RTX PRO 6000 | ~2-3 hours on CPU |
| VRAM | ~2-4 GB (negligible on 96GB card) | 0 (CPU-only) |
| Dependency | `torchvision.models.optical_flow.raft_large()` — already installed | `cv2.calcOpticalFlowFarneback` — already installed |

**Decision**: GPU-RAFT. Walking videos have frequent camera shake/motion blur — exactly where RAFT outperforms classical methods. GPU is idle during m04f otherwise. 3x faster. Zero new deps.

> **Fallback**: If RAFT is noisy on compressed 480p, Farneback is a one-line swap. Keep `compute_clip_motion()` flow-engine-agnostic.

#### Features per clip (12D vector)

| Feature | Dim | Description |
|---------|-----|-------------|
| `mean_magnitude` | 1 | Mean optical flow magnitude across frame pairs |
| `magnitude_std` | 1 | Std dev of flow magnitude (motion consistency) |
| `max_magnitude` | 1 | Peak flow magnitude (captures sudden motion) |
| `direction_histogram` | 8 | 8-bin normalized histogram of flow directions (0-360) |
| `camera_motion_x` | 1 | Median horizontal flow (camera pan estimate) |
| `camera_motion_y` | 1 | Median vertical flow (camera tilt estimate) |

**Total**: 13D per clip (compact, interpretable, sufficient for correlation)

#### Frame Sampling Strategy

- Sample 16 evenly-spaced consecutive frame pairs per clip (stride = total_frames / 16)
- Resize frames to 520x360 for RAFT (recommended input size for `raft_large`)
- Compute RAFT flow on each pair (12 refinement iterations, pretrained `C_T_SKHT_V2` weights)
- Aggregate: mean/std/max over all 16 pairs for magnitudes, accumulate direction histogram

#### CLI

```bash
# 10K POC with local data (~30-50 min on RTX PRO 6000)
python -u src/m04f_motion_features.py --FULL --subset data/subset_10k.json \
    --local-data data/subset_10k_local 2>&1 | tee logs/m04f_motion.log

# Sanity (20 clips, ~1 min)
python -u src/m04f_motion_features.py --SANITY --subset data/subset_10k.json \
    --local-data data/subset_10k_local 2>&1 | tee logs/m04f_sanity.log
```

#### Output Files

```
outputs_poc/
├── motion_features.npy          # shape (N, 13), float32
├── motion_features.paths.npy    # shape (N,), clip keys (dtype=object)
└── motion_features_meta.json    # {n_clips, feature_names, frame_pairs, compute_time_sec}
```

#### Implementation Skeleton

```python
# m04f_motion_features.py
import torchvision.models.optical_flow as of_models
from torchvision.models.optical_flow import Raft_Large_Weights

def load_raft_model(device):
    """Load pretrained RAFT-Large from torchvision."""
    weights = Raft_Large_Weights.C_T_SKHT_V2  # Sintel+KITTI fine-tuned
    model = of_models.raft_large(weights=weights).to(device).eval()
    transforms = weights.transforms()
    return model, transforms

def compute_clip_motion(model, transforms, frame_pairs, device) -> np.ndarray:
    """Extract 13D motion feature from frame pairs via GPU-RAFT."""
    # 1. For each (prev, curr) pair: resize to 520x360, tensorize, apply transforms
    # 2. RAFT forward: flow = model(prev_tensor, curr_tensor)[-1]  # last iteration
    # 3. flow shape: (1, 2, H, W) -> magnitude + angle per pixel
    # 4. Aggregate across all 16 pairs:
    #    mean_mag, std_mag, max_mag, direction_hist(8), median_dx, median_dy
    # 5. Return 13D float32 vector

def main():
    # 1. check_gpu() — GPU required, no CPU fallback
    # 2. Load RAFT model
    # 3. Argparse: --SANITY, --FULL, --subset, --local-data, --no-wandb, --gpu-mem
    # 4. Stream clips (reuse _create_stream pattern from m05)
    # 5. Checkpoint/resume via .m04f_checkpoint.npz
    # 6. tqdm progress bar + wandb logging
    # 7. Save motion_features.npy + motion_features.paths.npy + motion_features_meta.json
```

#### Mandatory Checklist (per CLAUDE.md rule 44)

- [x] tqdm progress bar
- [x] Auto-resume from checkpoint
- [x] tee logging (CLI docstring)
- [x] wandb integration (via shared utils)

**Dependencies**: `torchvision>=0.12` (already installed — `raft_large` model), `torch`, `opencv-python` (video decode), `numpy`, `tqdm`. **No new packages.**

---

### Step 2: `m06b_temporal_corr.py` (NEW, ~250 lines)

**Purpose**: Measure how well each encoder's embedding space captures temporal/motion information.

#### Core Metric — Motion-Embedding Correlation

1. Load `motion_features.npy` (N, 13) and `embeddings{sfx}.npy` (N, D) for a given encoder
2. Align by keys (`motion_features.paths.npy` <-> `embeddings{sfx}.paths.npy`)
3. Sample P random clip pairs (P=100K, sufficient for stable correlation)
4. For each pair: compute L2 distance in embedding space + L2 distance in motion feature space
5. Compute **Spearman rank correlation** (rho) between the two distance vectors
6. Higher rho = encoder better captures temporal/motion structure

#### Additional Metrics

| Metric | Description | Expected ranking |
|--------|-------------|------------------|
| `spearman_rho` | Rank correlation (embedding dist vs motion dist) | V-JEPA >> DINOv2 > CLIP |
| `temporal_prec_at_k` | % of kNN neighbors with similar motion (within top quartile) | V-JEPA >> others |
| `motion_retrieval_map` | mAP where "relevant" = same motion cluster | V-JEPA >> others |

#### CLI

```bash
# Per encoder
python -u src/m06b_temporal_corr.py --encoder vjepa --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m06b_vjepa.log

for enc in random dinov2 clip vjepa_shuffled; do
    python -u src/m06b_temporal_corr.py --encoder $enc --FULL --subset data/subset_10k.json \
        2>&1 | tee logs/m06b_${enc}.log
done
```

#### Output Files

```
outputs_poc/
├── m06b_temporal_corr.json              # vjepa (no suffix)
├── m06b_temporal_corr_random.json
├── m06b_temporal_corr_dinov2.json
├── m06b_temporal_corr_clip.json
└── m06b_temporal_corr_vjepa_shuffled.json
```

#### Implementation Sketch

```python
def compute_temporal_correlation(embeddings, motion_features, n_pairs=100000):
    """Spearman correlation between embedding distance and motion distance."""
    idx_a = np.random.randint(0, N, n_pairs)
    idx_b = np.random.randint(0, N, n_pairs)
    emb_dist = np.linalg.norm(embeddings[idx_a] - embeddings[idx_b], axis=1)
    mot_dist = np.linalg.norm(motion_features[idx_a] - motion_features[idx_b], axis=1)
    rho, pval = scipy.stats.spearmanr(emb_dist, mot_dist)
    return rho, pval

def compute_temporal_prec_at_k(knn_indices, motion_features, k=6):
    """% of kNN neighbors in same motion quartile as query."""
    quartiles = np.digitize(motion_features[:, 0], np.percentile(motion_features[:, 0], [25, 50, 75]))
    matches = 0
    for i in range(len(knn_indices)):
        query_q = quartiles[i]
        neighbor_qs = quartiles[knn_indices[i, 1:k+1]]
        matches += np.sum(neighbor_qs == query_q)
    return matches / (len(knn_indices) * k) * 100
```

---

### Step 3: VLM Temporal Tags (Optional, depends on 1-2 results)

Add 3 temporal fields to taxonomy v4:

| Field | Type | Values |
|-------|------|--------|
| `camera_motion` | single | static, pan_left, pan_right, tilt_up, tilt_down, zoom_in, zoom_out, tracking |
| `dominant_traffic_flow` | single | static, left_to_right, right_to_left, towards_camera, away_from_camera, mixed |
| `crowd_dynamics` | single | static, flowing, dispersing, converging, chaotic |

**Pros**: Richer temporal signal, human-interpretable

**Cons**: VLM temporal understanding is **confirmed unreliable** for fine-grained motion features (MotionBench CVPR 2025: <60%, CameraBench NeurIPS 2025: ~50% AP, SpookyBench: 0% on pure temporal patterns). Out-of-the-box Qwen3-VL will NOT reliably tag camera_motion or traffic_flow.

**If pursued** (after Steps 1-2 validate the framework):
- **Option A**: Fine-tune Qwen3-VL-8B on ~1,400 manually annotated WalkIndia clips (CameraBench showed this doubles camera motion AP)
- **Option B**: Try Molmo 2 (8B, Ai2 Dec 2025) — strongest open-weight video temporal scores
- **Option C**: Try TimeLens-8B — fine-tuned Qwen3-VL for temporal grounding

**Recommendation**: Steps 1-2 first. VLM temporal tags only after fine-tuning, never out-of-the-box.

---

### Step 4: Temporal Prec@K in m06 (extend after Step 2 validates)

Add `--temporal` flag to m06 that loads motion features and adds:
- `temporal_prec_at_k_easy` / `temporal_prec_at_k_hard` to metrics JSON
- **Control**: if DINOv2 scores well on temporal metrics -> tags are spatial proxies (not true temporal)

---

### Step 5: m08b Extension (extend after Step 4)

Add temporal metrics to radar chart + bar chart:
- `spearman_rho` to comparison metrics
- `temporal_prec_at_k` to bar chart
- Update LaTeX table with temporal columns

---

## Execution Order & Dependencies

```
Priority 2 (bug fix)         → immediate, no deps
  ├── m06 suffix fix
  └── m08 encoder flag + suffix

Priority 1 Step 1 (m04f)     → immediate, no deps
Priority 1 Step 2 (m06b)     → depends on m04f + existing m06 outputs
Priority 1 Steps 3-5         → depends on Step 2 results (data-driven)
```

---

## Honest Assessment

**What this plan validates**: Whether V-JEPA's embedding space captures motion/temporal structure better than image baselines. If the Spearman correlation is significantly higher for V-JEPA than DINOv2/CLIP, that's the missing piece from Ch9.

**Risk**: Even with GPU-RAFT, 480p compressed video may limit flow quality. If correlation is weak across ALL encoders, it could mean (a) flow features are too noisy at this resolution, or (b) none of these encoders truly capture temporal structure.

**What I'd bet on**: V-JEPA will show higher temporal correlation than DINOv2/CLIP (it IS a temporal encoder), but the gap may be smaller than expected because Ch9 showed shuffled > normal, suggesting V-JEPA's temporal signal is confused by Indian street chaos.
