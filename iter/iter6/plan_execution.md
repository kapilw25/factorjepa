# Execution Plan: Ch9 Complete Pipeline (m04 → m08b)

STATUS: m00-m03 COMPLETED (Mac CPU). Steps 0-7 (collapsed pipeline — each module runs once).
Dataset: https://huggingface.co/datasets/anonymousML123/walkindia-200k (private, HF_TOKEN in .env)
VLM: **Qwen3-VL-8B** (winner — 0.919 weighted score via 5-criterion bake-off). Transformers batched inference with AdaptiveBatchSizer.

### GPU Strategy & Clip Selection

| Mode | Command Flags | GPU | VRAM | Input Pool | Selection Method | Output Count | Output Path |
|------|---------------|-----|------|------------|------------------|--------------|-------------|
| **SANITY** | `--SANITY` | RTX Pro 4000 | 24GB | 115K stream | First 20 from stream | 20 | `outputs/tags_sanity_qwen.json` |
| **FULL (POC)** | `--FULL --subset` | RTX Pro 6000 | 96GB | 10K subset | All 10,000 matching subset keys | 10,000 | `outputs_poc/tags.json` |
| **FULL (prod)** | `--FULL` | RTX Pro 6000 | 96GB | 115K stream | All clips, no filter | 115,687 | `outputs/tags.json` |

---

## Pre-flight: GPU Server Setup

```bash
git clone <repo> && cd LLM_asAgent_3D_SR
./setup_env_uv.sh --gpu
source venv_walkindia/bin/activate
```

### Verify setup

```bash
# GPU + packages
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB')"
python -c "import faiss; print(f'FAISS GPUs: {faiss.get_num_gpus()}')"
python -c "import flash_attn; print(f'Flash-Attn:   {flash_attn.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import cuml; print(f'cuML: {cuml.__version__}')"
python -c "import wandb; print(f'wandb: {wandb.__version__}')"

# HF auth (private repo)
python -c "from dotenv import load_dotenv; load_dotenv(); import os; t=os.getenv('HF_TOKEN'); print(f'HF_TOKEN: {t[:10]}...' if t else 'MISSING')"

# Subset file exists
python -c "import json; d=json.load(open('data/subset_10k.json')); print(f'Subset: {d[\"n\"]} clips from {d[\"num_videos\"]} videos')"
```

---

## Step 0: VLM Selection — COMPLETED

**Winner: Qwen3-VL-8B** (weighted score: **0.919** out of 1.0)

5-criterion bake-off on 20-clip SANITY + 2,500-clip partial runs:

| Criterion | Weight | Qwen | VideoLLaMA3 | LLaVA |
|-----------|--------|------|-------------|-------|
| JSON Parse | 30% | 1.00 | 0.83 | 1.00 |
| Agreement | 25% | 0.89 | 0.67 | 0.50 |
| Speed | 20% | 1.00 | 0.10 | 0.31 |
| Taxonomy | 15% | 1.00 | 0.88 | 0.85 |
| Conf. Calibration | 10% | 0.47 | 0.47 | 0.64 |
| **Weighted Total** | | **0.919** | **0.615** | **0.676** |

See `iter/iter6/plots/m04b_vlm_comparison.png` for full visualization.

**Status:**
- [x] SANITY (3 VLMs x 20 clips): DONE
- [x] Sanity comparison (`m04c_sanity_compare.png`): DONE
- [x] Winner selection: **Qwen** (0.919 — dominates on parse, speed, taxonomy)

---

## Step 1: Qwen tags 10K clips (v3 taxonomy) — RTX Pro 6000 (96GB)

Qwen3-VL tags all 10K POC clips using v3 taxonomy (`src/utils/tag_taxonomy.json` — 16 fields including `traffic_mix`, `ped_vehicle_separation`, `road_encroachment`, `video_quality`).

```bash
python -u src/m04_vlm_tag.py --model qwen --FULL --subset data/subset_10k.json 2>&1 | tee logs/m04_full_qwen_poc.log
```

### Verify Step 1 output

```bash
python -c "
import json
t = json.load(open('src/outputs_poc/tags.json'))
print(f'tags.json: {len(t)} clips, {len(t[0].keys())} fields')
for field in ['traffic_mix', 'ped_vehicle_separation', 'road_encroachment', 'video_quality']:
    vals = set(t[i].get(field, 'MISSING') for i in range(min(20, len(t))))
    print(f'  {field}: {vals}')
"
```

**Expected:** `outputs_poc/tags.json` with ~10K clips x 40+ fields (16 tags + 16 confidences + provenance).
**Est. time (96GB, batch=36):** ~2-3h at 0.7-1.0 clips/s. Checkpoint saves every 500 clips.

**Status:**
- [ ] Qwen FULL POC (10K, v3 taxonomy): pending

---

## Step 2: V-JEPA 2 embeddings (streams from HF)

V-JEPA 2 ViT-G (1B params, frozen) encodes each clip → 1408-dim embedding. Producer-consumer pipeline with torch.compile.

```bash
# SANITY — RTX Pro 4000 (24GB) — validate model loading + embedding output
python -u src/m05_vjepa_embed.py --SANITY 2>&1 | tee logs/m05_sanity.log

# FULL — RTX Pro 6000 (96GB) — embed 10K clips
python -u src/m05_vjepa_embed.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m05_vjepa_embed_poc.log
```

### Verify Step 2 output

```bash
python -c "
import numpy as np
emb = np.load('src/outputs_poc/embeddings.npy')
paths = np.load('src/outputs_poc/embeddings.paths.npy', allow_pickle=True)
print(f'embeddings.npy:       {emb.shape} (expect ~10000 x 1408)')
print(f'embeddings.paths.npy: {len(paths)} clip keys')
print(f'Shape match: {emb.shape[0] == len(paths)}')
"
```

**Expected:** `embeddings.npy` (~10K × 1408), `embeddings.paths.npy` (10K keys). ~2h GPU.

**SANITY Status:**
- [x] m05 SANITY: PASSED (6 clips streamed, 2 dupes removed → 4 unique embeddings, shape (4, 1408), 96.7s, 0.1 clips/s)

---

### Encoder Comparison Table

> V-JEPA metrics alone are meaningless without baselines — "18.73% compared to what?" Steps 3-4 add 4 baselines + True Overlap@K. Steps 5-7 run m06/m07/m08 once each on ALL 5 encoders.
>
> All baseline scripts built and verified (`py_compile` + AST) on M1 Mac. See `iter/iter6/plan_ch9_baselines.md` for architecture details.

| Encoder | Script | Model | Dim | Type | GPU? | Attention | Batch Profile |
|---------|--------|-------|-----|------|------|-----------|---------------|
| V-JEPA (Step 2) | `m05_vjepa_embed.py` | `vjepa2-vitg-fpc64-384` | 1408 | video (all frames) | GPU | FA2 + compile | `["vjepa"]` |
| Random | `m05b_baselines.py` | — | 1408 | synthetic | **CPU** | — | — |
| DINOv2 | `m05b_baselines.py` | `dinov2-vitl14` | 1024 | image (middle frame) | GPU | FA2 + compile | `["image_encoder"]` (4x vjepa) |
| CLIP | `m05b_baselines.py` | `clip-vit-large-patch14` | 768 | image (middle frame) | GPU | SDPA + compile | `["image_encoder"]` (4x vjepa) |
| Shuffled V-JEPA | `m05b_baselines.py` | `vjepa2-vitg-fpc64-384` | 1408 | video (shuffled frames) | GPU | FA2 + compile | `["vjepa"]` |

**Native dims, no projection.** FAISS is dimension-agnostic (`d = embeddings.shape[1]`). Metrics are dimensionless ratios — comparable across dims.

---

## Step 3: Baseline embeddings — m05b (CPU + GPU)

**Prerequisite:** Step 2 must be complete (V-JEPA `embeddings.npy` + `embeddings.paths.npy` exist).

Generate embeddings for all 4 baseline encoders sequentially (Random → DINOv2 → CLIP → Shuffled V-JEPA). Random is CPU-safe; others require GPU. Each encoder auto-skips if output already exists.

**GPU Optimizations (matching m05_vjepa_embed.py patterns):**
- DINOv2: FA2 (`attn_implementation="flash_attention_2"`) + `torch.compile(model)` + producer pre-processes tensors on CPU thread
- CLIP: SDPA (`attn_implementation="sdpa"`) + `torch.compile(model)` + producer pre-processes tensors on CPU thread
- Shuffled V-JEPA: FA2 + `torch.compile(model)` + producer pre-processes (same V-JEPA model)
- Batch size: `compute_batch_sizes()["image_encoder"]` for DINOv2/CLIP (4x vjepa, cap 256 — single-frame models are 10-50x cheaper per clip than V-JEPA)

```bash
# SANITY — validate all 4 encoders load + produce output (5 clips each)
python -u src/m05b_baselines.py --encoder all --SANITY 2>&1 | tee logs/m05b_all_sanity.log

# FULL — embed 10K clips × 4 encoders (~6-8h GPU total)
python -u src/m05b_baselines.py --encoder all --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m05b_all_poc.log
```

> **Debug individual encoder:** `python -u src/m05b_baselines.py --encoder dinov2 --SANITY 2>&1 | tee logs/m05b_dinov2_sanity.log`
> Choices: `random`, `dinov2`, `clip`, `vjepa_shuffled`, `all`

### Verify Step 3 output

```bash
python -c "
import numpy as np
for enc, sfx, dim in [('random','_random',1408), ('dinov2','_dinov2',1024),
                       ('clip','_clip',768), ('vjepa_shuffled','_vjepa_shuffled',1408)]:
    emb = np.load(f'src/outputs_poc/embeddings{sfx}.npy')
    paths = np.load(f'src/outputs_poc/embeddings{sfx}.paths.npy', allow_pickle=True)
    ok = '✓' if emb.shape[1] == dim and emb.shape[0] == len(paths) else '✗'
    print(f'{ok} {enc:20s} embeddings{sfx}.npy: {emb.shape} (expect ~10K x {dim}), paths: {len(paths)}')
"
```

**Expected per encoder:**

| Encoder | File | Shape | Time |
|---------|------|-------|------|
| Random | `embeddings_random.npy` | (10K, 1408) | ~1 min CPU |
| DINOv2 | `embeddings_dinov2.npy` | (10K, 1024) | ~2-3h GPU |
| CLIP | `embeddings_clip.npy` | (10K, 768) | ~2h GPU |
| Shuffled V-JEPA | `embeddings_vjepa_shuffled.npy` | (10K, 1408) | ~2h GPU |

**Status:**
- [ ] SANITY (`--encoder all --SANITY`): pending
- [ ] FULL POC (`--encoder all --FULL --subset`): pending

---

## Step 4: True Overlap@K augmented embeddings — RTX Pro 6000 (96GB)

**Prerequisite:** Step 2 must be complete (V-JEPA model loads, embeddings exist).

Generate two augmented V-JEPA embedding sets (BYOL/DINO multi-crop protocol) for True Overlap@K measurement.

- View A: `RandomResizedCrop(scale=0.4-1.0)` + ColorJitter
- View B: `RandomResizedCrop(scale=0.2-0.6)` + GaussianBlur
- Same crop params for all T frames per clip (temporal consistency)

```bash
python -u src/m05c_true_overlap.py --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m05c_overlap_poc.log
```

### Verify Step 4 output

```bash
python -c "
import numpy as np
a = np.load('src/outputs_poc/overlap_augA.npy')
b = np.load('src/outputs_poc/overlap_augB.npy')
k = np.load('src/outputs_poc/overlap_keys.npy', allow_pickle=True)
print(f'overlap_augA.npy: {a.shape} (expect ~10K x 1408)')
print(f'overlap_augB.npy: {b.shape} (expect ~10K x 1408)')
print(f'overlap_keys.npy: {len(k)} keys')
print(f'Shape match: {a.shape == b.shape and a.shape[0] == len(k)}')
"
```

**Expected:** `overlap_augA.npy` + `overlap_augB.npy` (10K x 1408 each) + `overlap_keys.npy` (10K keys).
**Est. time:** ~3-4h GPU (2x V-JEPA inference + augmentation overhead).

**Status:**
- [ ] True Overlap augmented embeddings: pending

---

## Step 5: FAISS 9-metric evaluation — ALL 5 encoders — RTX Pro 6000 (96GB)

**Prerequisite:** Steps 1 (v3 tags), 2 (vjepa), 3a-3d (all baselines), 4 (augmented embeddings).

**Blackwell (sm_120) prerequisite:** `faiss-gpu-cu12` pip package only ships sm_70+sm_80 kernels → CUDA error 209 at runtime. Must build from source first:
```bash
./build_faiss_sm120.sh 2>&1 | tee logs/build_faiss_sm120.log
# Re-install only (build artifacts cached): ./build_faiss_sm120.sh --install
```

FAISS-GPU kNN index → 9 metrics in Easy/Hard mode + confidence sweep + multi-attribute slices. Run m06 once per encoder. V-JEPA gets `--true-overlap` for True Overlap@K (using augmented embeddings from Step 4). Tags are shared across all encoders (same `tags.json`).

### 5a: V-JEPA (with True Overlap@K)

```bash
python -u src/m06_faiss_metrics.py --encoder vjepa --true-overlap \
    --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_vjepa_poc.log
```

### 5b: Random

```bash
python -u src/m06_faiss_metrics.py --encoder random \
    --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_random_poc.log
```

### 5c: DINOv2

```bash
python -u src/m06_faiss_metrics.py --encoder dinov2 \
    --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_dinov2_poc.log
```

### 5d: CLIP

```bash
python -u src/m06_faiss_metrics.py --encoder clip \
    --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_clip_poc.log
```

### 5e: Shuffled V-JEPA

```bash
python -u src/m06_faiss_metrics.py --encoder vjepa_shuffled \
    --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_shuffled_poc.log
```

### Verify Step 5 output

```bash
python -c "
import json
for enc, sfx in [('vjepa',''), ('random','_random'), ('dinov2','_dinov2'),
                  ('clip','_clip'), ('vjepa_shuffled','_vjepa_shuffled')]:
    try:
        m = json.load(open(f'src/outputs_poc/m06_metrics{sfx}.json'))
        ov = m['easy'].get('overlap_method', 'dim_split')
        print(f'✓ {enc:20s} Prec@K={m[\"easy\"][\"prec_at_k\"]:5.1f}%  mAP={m[\"easy\"][\"map_at_k\"]:.3f}  '
              f'Cycle={m[\"easy\"][\"cycle_at_k\"]:5.1f}%  overlap={ov}')
    except FileNotFoundError:
        print(f'✗ {enc:20s} m06_metrics{sfx}.json NOT FOUND')
"
```

**Expected per encoder:**

| Encoder | Output JSON | kNN Indices | Time |
|---------|-------------|-------------|------|
| V-JEPA | `m06_metrics.json` | `knn_indices.npy` | ~5 min |
| Random | `m06_metrics_random.json` | `knn_indices_random.npy` | ~5 min |
| DINOv2 | `m06_metrics_dinov2.json` | `knn_indices_dinov2.npy` | ~5 min |
| CLIP | `m06_metrics_clip.json` | `knn_indices_clip.npy` | ~5 min |
| Shuffled V-JEPA | `m06_metrics_vjepa_shuffled.json` | `knn_indices_vjepa_shuffled.npy` | ~5 min |

**Note:** V-JEPA's `m06_metrics.json` will have `"overlap_method": "true_multi_crop"` (from Step 4). All others use dim-split approximation.

**SANITY Status:**
- [x] m06 SANITY: PASSED (4 clips, FAISS-GPU sm_120 from source build, Easy Prec@K=50%, knn_indices (4,4), 3 plots saved)

**Status:**
- [ ] V-JEPA (with True Overlap): pending
- [ ] Random: pending
- [ ] DINOv2: pending
- [ ] CLIP: pending
- [ ] Shuffled V-JEPA: pending

---

## Step 6: UMAP for ALL 5 encoders — RTX Pro 6000 (96GB)

**Prerequisite:** Steps 2 + 3a-3d (all encoder embeddings exist).

cuML GPU UMAP: N × D → N × 2 for each encoder. Produces `umap_2d{sfx}.npy` for CPU plotting.

```bash
# V-JEPA
python -u src/m07_umap.py --encoder vjepa --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m07_umap_vjepa_poc.log

# Random
python -u src/m07_umap.py --encoder random --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m07_umap_random_poc.log

# DINOv2
python -u src/m07_umap.py --encoder dinov2 --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m07_umap_dinov2_poc.log

# CLIP
python -u src/m07_umap.py --encoder clip --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m07_umap_clip_poc.log

# Shuffled V-JEPA
python -u src/m07_umap.py --encoder vjepa_shuffled --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m07_umap_shuffled_poc.log
```

### Verify Step 6 output

```bash
python -c "
import numpy as np
for enc, sfx in [('vjepa',''), ('random','_random'), ('dinov2','_dinov2'),
                  ('clip','_clip'), ('vjepa_shuffled','_vjepa_shuffled')]:
    try:
        u = np.load(f'src/outputs_poc/umap_2d{sfx}.npy')
        print(f'✓ {enc:20s} umap_2d{sfx}.npy: {u.shape}')
    except FileNotFoundError:
        print(f'✗ {enc:20s} umap_2d{sfx}.npy NOT FOUND')
"
```

**Expected:** 5 files × (10K, 2). ~2 min each GPU.

**SANITY Status:**
- [x] m07 SANITY: PASSED (4 clips, cuML GPU UMAP in 0.6s, n_neighbors=3, umap_2d.npy shape (4, 2))

**Status:**
- [ ] V-JEPA UMAP: pending
- [ ] Random UMAP: pending
- [ ] DINOv2 UMAP: pending
- [ ] CLIP UMAP: pending
- [ ] Shuffled V-JEPA UMAP: pending

---

## Step 7: Visualization + multi-encoder comparison (CPU — no GPU needed)

**Prerequisite:** Steps 5a-5e (all 5 encoder metrics JSON files exist) + Step 6 (UMAP).

### 7a: Per-encoder V-JEPA plots (m08)

```bash
python -u src/m08_plot.py --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m08_plot_poc.log
```

### 7b: Multi-encoder comparison (m08b)

Reads `m06_metrics_*.json` for all available encoders. Generates grouped bar chart, radar plot, LaTeX table, and terminal summary.

```bash
python -u src/m08b_compare.py --FULL --subset data/subset_10k.json \
    2>&1 | tee logs/m08b_compare.log
```

### Verify Step 7 output

```bash
# Per-encoder plots (m08)
ls -la src/outputs_poc/m08_umap.{png,pdf}
ls -la src/outputs_poc/m08_confusion_matrix.{png,pdf}
ls -la src/outputs_poc/m08_knn_grid.{png,pdf}

# Multi-encoder comparison (m08b)
ls -la src/outputs_poc/m08b_encoder_comparison.{png,pdf}
ls -la src/outputs_poc/m08b_radar.{png,pdf}
ls -la src/outputs_poc/m08b_comparison_table.tex
```

**Expected outputs from m08b:**

| File | Description |
|------|-------------|
| `m08b_encoder_comparison.{png,pdf}` | Grouped bar chart: 5 metrics x 5 encoders, Easy vs Hard |
| `m08b_radar.{png,pdf}` | Radar plot: one polygon per encoder, normalized |
| `m08b_comparison_table.tex` | Paper-ready LaTeX table |

**Status:**
- [ ] m08 per-encoder plots: pending
- [ ] m08b multi-encoder comparison: pending

---

## Final Verification: All Ch9 Outputs

```bash
echo "=== TAGS (v3 taxonomy) ==="
python -c "
import json; t=json.load(open('src/outputs_poc/tags.json'))
print(f'tags.json: {len(t)} clips, {len(t[0].keys())} fields')
v3_fields = ['traffic_mix','ped_vehicle_separation','road_encroachment','video_quality']
present = [f for f in v3_fields if f in t[0]]
print(f'v3 fields: {len(present)}/{len(v3_fields)} present')
"

echo ""
echo "=== EMBEDDINGS (5 encoders) ==="
python -c "
import numpy as np
for enc, sfx, dim in [('vjepa','',1408), ('random','_random',1408), ('dinov2','_dinov2',1024),
                       ('clip','_clip',768), ('vjepa_shuffled','_vjepa_shuffled',1408)]:
    try:
        e = np.load(f'src/outputs_poc/embeddings{sfx}.npy')
        p = np.load(f'src/outputs_poc/embeddings{sfx}.paths.npy', allow_pickle=True)
        ok = '✓' if e.shape[1]==dim and e.shape[0]==len(p) else '✗'
        print(f'{ok} {enc:20s} {e.shape}')
    except: print(f'✗ {enc:20s} MISSING')
"

echo ""
echo "=== METRICS (5 encoders) ==="
python -c "
import json
print(f'{\"Encoder\":20s} {\"Prec@K\":>8s} {\"mAP@K\":>8s} {\"Cycle@K\":>8s} {\"nDCG@K\":>8s} {\"Silhouet\":>8s}')
print('-' * 62)
for enc, sfx in [('vjepa',''), ('random','_random'), ('dinov2','_dinov2'),
                  ('clip','_clip'), ('vjepa_shuffled','_vjepa_shuffled')]:
    try:
        m = json.load(open(f'src/outputs_poc/m06_metrics{sfx}.json'))
        e = m['easy']
        print(f'{enc:20s} {e[\"prec_at_k\"]:7.1f}% {e[\"map_at_k\"]:8.4f} {e[\"cycle_at_k\"]:7.1f}% {e[\"ndcg_at_k\"]:8.4f} {e[\"silhouette\"]:8.4f}')
    except: print(f'{enc:20s} MISSING')
"

echo ""
echo "=== TRUE OVERLAP@K ==="
python -c "
import json
try:
    m = json.load(open('src/outputs_poc/m06_metrics.json'))
    method = m['easy'].get('overlap_method', 'dim_split')
    ov = m['easy'].get('overlap_at_k', 'N/A')
    print(f'Overlap@K: {ov}% (method: {method})')
except: print('MISSING')
"

echo ""
echo "=== UMAP (5 encoders) ==="
python -c "
import numpy as np
for enc, sfx in [('vjepa',''), ('random','_random'), ('dinov2','_dinov2'),
                  ('clip','_clip'), ('vjepa_shuffled','_vjepa_shuffled')]:
    try:
        u = np.load(f'src/outputs_poc/umap_2d{sfx}.npy')
        print(f'✓ {enc:20s} {u.shape}')
    except: print(f'✗ {enc:20s} MISSING')
"

echo ""
echo "=== PLOTS ==="
ls -la src/outputs_poc/m06_*.png src/outputs_poc/m08_*.png src/outputs_poc/m08b_*.png 2>/dev/null
ls -la src/outputs_poc/m08b_comparison_table.tex 2>/dev/null
```

---

## Timeline

### RTX PRO 6000 Blackwell (96GB VRAM — production GPU, batched inference)

| Step | Module | GPU? | Est. Time | Actual |
|------|--------|------|-----------|--------|
| 0 | VLM Selection (3 VLMs bake-off) | GPU | — | **DONE** (Qwen 0.919) |
| 1 | m04 Qwen tags 10K (v3 taxonomy) | GPU | ~2-3h | |
| 2 | m05 V-JEPA embed (10K) | GPU | ~2h | |
| 3 | m05b `--encoder all` (random+DINOv2+CLIP+shuffled) | GPU | ~6-8h | |
| 4 | m05c True Overlap augmented (10K) | GPU | ~3-4h | |
| 5 | m06 FAISS metrics (x5 encoders) | GPU | ~25 min | |
| 6 | m07 UMAP (x5 encoders) | GPU | ~10 min | |
| 7 | m08 + m08b plots + comparison | CPU | ~5 min | |
| **Grand Total** | | | **~14-17h GPU + ~10 min CPU** | |

### GPU Parallelization Opportunities

Steps 3b, 3c, 3d can run on **different GPUs** in parallel (independent encoder embeddings). On single GPU, run sequentially. Step 4 (m05c) can run in parallel with Steps 3b-3d if a second GPU is available.

### Vast.ai: Use Datacenter-Backed Instances Only

> **IMPORTANT:** When renting GPU instances on vast.ai, always select machines tagged **`datacenter`** (not `host`-only). Datacenter machines support **shared storage volumes** — multiple GPU instances can mount the same disk. This enables spinning up 2-4 parallel instances on the same volume to run baselines concurrently:
>
> | Instance | Step | Encoder | Shared Volume |
> |----------|------|---------|---------------|
> | GPU #1 | 3b | DINOv2 | `/workspace` (shared) |
> | GPU #2 | 3c | CLIP | `/workspace` (shared) |
> | GPU #3 | 3d | Shuffled V-JEPA | `/workspace` (shared) |
> | GPU #4 | 4 | m05c True Overlap | `/workspace` (shared) |
>
> All instances read from the same `data/subset_10k.json` and `embeddings.paths.npy`, and write to separate output files (`embeddings_dinov2.npy`, `embeddings_clip.npy`, etc.) — no write conflicts.
>
> **How to set up:**
> 1. Create a **local volume** on a datacenter (e.g. `datacenter:120840` Texas or `datacenter:18` Alberta)
> 2. Rent first instance with "Create local volume" → this provisions the shared disk
> 3. Rent additional instances on the **same datacenter** → attach the **existing volume**
> 4. All instances see the same `/workspace` with repo, venv, and HF cache
>
> **Cost savings:** Running 4 baselines in parallel (~2-3h each) costs the same total GPU-hours as sequential (~10h on 1 GPU), but finishes in ~3h wall-clock instead of ~10h. At ~$1/hr per RTX PRO 6000, that's $4 x 3h = $12 parallel vs $1 x 10h = $10 sequential — marginal cost increase for 3x faster turnaround.

---

## Dependency Graph

```
Step 0: VLM Selection ✅ DONE (Qwen 0.919)
         │
         ↓
Step 1: m04 Qwen tags 10K (v3 taxonomy) ──────────────────────────────────────────┐
         │                                                                         │
         ↓                                                                         │
Step 2: m05 V-JEPA embed ──┬──────────────────────────────────────────────────┐    │
                            │                                                  │    │
                            ├→ Step 3a: m05b random  (CPU) ──┐                │    │
                            │                                 │                │    │
                            ├→ Step 3b: m05b DINOv2  (GPU) ──┤                │    │
                            │                                 │                │    │
                            ├→ Step 3c: m05b CLIP    (GPU) ──┤                │    │
                            │                                 │                │    │
                            └→ Step 3d: m05b shuffled (GPU) ──┤                │    │
                                                              │                │    │
                            Step 4: m05c true overlap ────────┤                │    │
                                                              │                │    │
                               ┌──────────────────────────────┘                │    │
                               │  All 5 encoder embeddings                    │    │
                               │  + augmented embeddings                      │    │
                               │  + v3 tags                                   │    │
                               ▼                                              ▼    ▼
                     Step 5: m06 FAISS metrics (x5 encoders + --true-overlap)
                               │
                               ├→ Step 6: m07 UMAP (x5 encoders)
                               │
                               └→ Step 7: m08 + m08b plots + comparison + LaTeX table
```

**Key dependencies:**
- Steps 3a-3d are independent of each other (can parallelize on multiple GPUs)
- Step 4 is independent of Steps 3a-3d (can parallelize)
- Step 5 needs ALL of: Step 1 (tags) + Steps 2+3a-3d (embeddings) + Step 4 (augmented)
- Steps 6-7 only need Step 5 output
- Steps 2-4 have NO dependency on Step 1 (tags). Embeddings can be generated IN PARALLEL with tagging. Only Step 5 (metrics) needs both embeddings AND tags.

All clips stream from HF — no local data/clips needed on GPU server.
