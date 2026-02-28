# Execution Plan: BAKEOFF Experiment (m04 → m08)

STATUS: m00-m03 COMPLETED (Mac CPU). m04-m08 on GPU.
Dataset: https://huggingface.co/datasets/anonymousML123/walkindia-200k (private, HF_TOKEN in .env)
Backend: All 3 VLMs use transformers sequential inference (vLLM removed — OOMs on ≤24GB GPUs).

### GPU Strategy
| Mode | GPU | VRAM | Purpose |
|------|-----|------|---------|
| `--SANITY` (20 clips) | RTX Pro 4000 | 24GB | Debug: validate model loading, inference, JSON parsing |
| `--BAKEOFF` (2500 clips) | RTX Pro 6000 | 96GB | Production: 3-VLM comparison |
| `--FULL` (10K-115K clips) | RTX Pro 6000 | 96GB | Production: winner tags full dataset |

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
python -c "import cuml; print(f'cuML: {cuml.__version__}')"
python -c "import wandb; print(f'wandb: {wandb.__version__}')"

# HF auth (private repo)
python -c "from dotenv import load_dotenv; load_dotenv(); import os; t=os.getenv('HF_TOKEN'); print(f'HF_TOKEN: {t[:10]}...' if t else 'MISSING')"

# Subset file exists
python -c "import json; d=json.load(open('data/subset_10k.json')); print(f'Subset: {d[\"n\"]} clips from {d[\"num_videos\"]} videos')"
```

---

## Step 0: SANITY checks — RTX Pro 4000 (24GB) — debug/validate before bake-off

Run each VLM on 20 clips to verify model loading, inference, JSON parsing, and output format.
Delete stale checkpoints before each sanity run if needed.

```bash
# Clean stale outputs (only if re-running)
rm -f src/outputs/tags.json src/outputs/tags.json.tmp

# Qwen3-VL sanity (PASSED — 20/20 clips, 0.08 clips/s, 260s)
python -u src/m04_vlm_tag.py --model qwen --SANITY 2>&1 | tee logs/m04_sanity_qwen.log

# VideoLLaMA3 sanity
python -u src/m04_vlm_tag.py --model videollama --SANITY 2>&1 | tee logs/m04_sanity_videollama.log

# LLaVA-NeXT-Video sanity
python -u src/m04_vlm_tag.py --model llava --SANITY 2>&1 | tee logs/m04_sanity_llava.log
```

### Verify Step 0 output

```bash
# Check each model's sanity output (SANITY creates tags_sanity_{model}.json, not tags.json)
for model in qwen videollama llava; do
  python -c "
import json
t = json.load(open('src/outputs/tags_sanity_${model}.json'))
print(f'tags_sanity_${model}.json: {len(t)} clips, {len(t[0].keys())} fields')
print(f'  scene_type: {t[0].get(\"scene_type\", \"MISSING\")}')
print(f'  confidence: {t[0].get(\"confidence_scene_type\", \"MISSING\")}')
"
done
```

**Expected:** 20 clips, 34 fields each, valid JSON with all 11 tag fields + 11 confidence fields + provenance.

**Status:**
- [x] Qwen3-VL: PASSED (20/20, 0.08 clips/s, all JSON parsed)
- [x] VideoLLaMA3: PASSED (20/20, 0.10 clips/s, all JSON parsed)
- [x] LLaVA-NeXT-Video: PASSED (20/20, 0.08 clips/s, 17/20 valid — 3 returned enum dumps)

**Note:** `m04b_vlm_select.py` (Step 2) is expected to fail at this stage — it requires Step 1 bakeoff data (`src/data/bakeoff/tags_{model}.json`) which doesn't exist until bakeoff runs complete.

### Compare SANITY results (CPU — no GPU needed)

```bash
python -u src/m04c_sanity_compare.py 2>&1 | tee logs/m04c_sanity_compare.log
ls -la src/outputs/m04c_sanity_compare.{png,pdf}
```

**Output:** `m04c_sanity_compare.png` + `.pdf` — 2x2 dashboard (parse rate, scene distribution, confidence boxplot, on/off-taxonomy objects butterfly chart).

---

## Step 1: VLM Bake-off — RTX Pro 6000 (96GB) — 3 VLMs × 2,500 clips each

Each VLM tags the first 2,500 clips from the 10K subset (streams from HF, no local clips).

```bash
# Clean stale bakeoff checkpoints (only if re-running from scratch)
rm -f src/data/bakeoff/tags_qwen.json src/data/bakeoff/tags_videollama.json src/data/bakeoff/tags_llava.json

python -u src/m04_vlm_tag.py --model qwen --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_qwen_poc.log

python -u src/m04_vlm_tag.py --model videollama --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_videollama_poc.log

python -u src/m04_vlm_tag.py --model llava --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_llava_poc.log
```

### Verify Step 1 output

```bash
# 3 bakeoff tag files exist, each with 2500 clips
for model in qwen videollama llava; do
  python -c "import json; t=json.load(open('src/data/bakeoff/tags_${model}.json')); print(f'tags_${model}.json: {len(t)} clips')"
done
```

**Expected:** 3 files × 2,500 clips each.
**Est. time (24GB GPU):** ~8-9h per VLM at ~0.08 clips/s. ~26h total (sequential).
**Est. time (96GB GPU):** ~1-2h per VLM. ~4-7h total.

---

## Step 2: Pick winner VLM (CPU — no GPU needed)

5-criterion weighted comparison: JSON parse (30%), cross-VLM agreement (25%), speed (20%), taxonomy compliance (15%), confidence calibration (10%).

```bash
python -u src/m04b_vlm_select.py 2>&1 | tee logs/m04b_vlm_select.log
```

### Verify Step 2 output

```bash
# Winner selected + comparison report
python -c "import json; d=json.load(open('src/data/bakeoff/vlm_comparison.json')); print(f'Winner: {d[\"winner\"]} (score: {d[\"models\"][d[\"winner\"]][\"weighted_total\"]:.3f})')"
ls -la src/data/bakeoff/vlm_comparison.{json,png,pdf}
ls -la src/data/bakeoff/vlm_dashboard.{png,pdf}
```

**Expected:** `vlm_comparison.json` with winner name + scores. `vlm_comparison.{png,pdf}` (weighted scores) + `vlm_dashboard.{png,pdf}` (2x2 diagnostic: parse rate, scene distribution, confidence, on/off-taxonomy objects).

---

## Step 3: Winner tags remaining 7,500 clips — RTX Pro 6000 (96GB)

Winner VLM runs --FULL on the 10K subset. Resumes from bake-off checkpoint (already has 2,500 tagged), so only ~7,500 new clips processed.

```bash
# Replace <winner> with output from Step 2 (qwen, videollama, or llava)
python -u src/m04_vlm_tag.py --model <winner> --FULL --subset data/subset_10k.json 2>&1 | tee logs/m04_full_poc.log
```

### Verify Step 3 output

```bash
python -c "import json; t=json.load(open('src/outputs_poc/tags.json')); print(f'tags.json: {len(t)} clips, fields: {len(t[0].keys())}')"
# Expect: ~10,000 clips, 33 fields each
```

**Expected:** `outputs_poc/tags.json` with ~10K clips × 33 fields.
**Est. time (24GB):** ~26h. **Est. time (96GB):** ~2-4h.

---

## Step 4: V-JEPA 2 embeddings (streams from HF)

V-JEPA 2 ViT-G (1B params, frozen) encodes each clip → 1408-dim embedding. Producer-consumer pipeline with torch.compile.

```bash
# SANITY — RTX Pro 4000 (24GB) — validate model loading + embedding output
python -u src/m05_vjepa_embed.py --SANITY 2>&1 | tee logs/m05_sanity.log

# FULL — RTX Pro 6000 (96GB) — embed 10K clips
python -u src/m05_vjepa_embed.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m05_vjepa_embed_poc.log
```

### Verify Step 4 output

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

---

## Step 5: FAISS 9-metric evaluation (requires Step 3 + Step 4)

FAISS-GPU kNN index → 9 metrics in Easy/Hard mode + confidence sweep + multi-attribute slices. Saves knn_indices.npy for downstream plotting.

```bash
# SANITY — RTX Pro 4000 (24GB) — validate FAISS index + metric computation
python -u src/m06_faiss_metrics.py --SANITY 2>&1 | tee logs/m06_sanity.log

# FULL — RTX Pro 6000 (96GB) — compute 9 metrics on 10K clips
python -u src/m06_faiss_metrics.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_faiss_metrics_poc.log
```

### Verify Step 5 output

```bash
# Metrics JSON
python -c "
import json
m = json.load(open('src/outputs_poc/m06_metrics.json'))
print(f'Easy Cycle@K:   {m[\"easy\"][\"cycle_at_k\"]:.1f}%')
print(f'Easy Prec@K:    {m[\"easy\"][\"prec_at_k\"]:.1f}%')
print(f'Hard Cycle@K:   {m[\"hard\"][\"cycle_at_k\"]:.1f}%')
print(f'Hard Prec@K:    {m[\"hard\"][\"prec_at_k\"]:.1f}%')
print(f'Silhouette:     {m[\"easy\"][\"silhouette\"]:.3f}')
print(f'Conf sweep pts: {len(m[\"confidence_sweep\"])}')
"

# kNN indices + plots exist
ls -la src/outputs_poc/knn_indices.npy
ls -la src/outputs_poc/m06_*.png src/outputs_poc/m06_*.pdf
```

**Expected:** `m06_metrics.json` (9 metrics × 2 modes), `knn_indices.npy`, 4 plots (.png + .pdf). ~5 min GPU.

---

## Step 6: UMAP dimensionality reduction (GPU cuML)

cuML GPU UMAP: 10K × 1408 → 10K × 2. Saves umap_2d.npy for CPU plotting.

```bash
# SANITY — RTX Pro 4000 (24GB) — validate cuML UMAP loading + output shape
python -u src/m07_umap.py --SANITY 2>&1 | tee logs/m07_sanity.log

# FULL — RTX Pro 6000 (96GB) — UMAP on 10K embeddings
python -u src/m07_umap.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m07_umap_poc.log
```

### Verify Step 6 output

```bash
python -c "import numpy as np; u = np.load('src/outputs_poc/umap_2d.npy'); print(f'umap_2d.npy: {u.shape} (expect ~10000 x 2)')"
```

**Expected:** `umap_2d.npy` (10K × 2). ~2 min GPU.

---

## Step 7: Visualization (CPU — no GPU needed)

Reads pre-computed .npy files (embeddings, knn_indices, umap_2d) + tags.json → UMAP scatter, confusion matrix, kNN grid.

```bash
python -u src/m08_plot.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m08_plot_poc.log
```

### Verify Step 7 output

```bash
ls -la src/outputs_poc/m08_umap.{png,pdf}
ls -la src/outputs_poc/m08_confusion_matrix.{png,pdf}
ls -la src/outputs_poc/m08_knn_grid.{png,pdf}
```

**Expected:** 3 plots × 2 formats = 6 files. ~5 min CPU.

---

## Final Verification: All POC Outputs

```bash
echo "=== BAKEOFF OUTPUTS ==="
ls -lh src/data/bakeoff/tags_*.json
ls -lh src/data/bakeoff/vlm_comparison.*

echo ""
echo "=== POC OUTPUTS ==="
ls -lh src/outputs_poc/tags.json
ls -lh src/outputs_poc/embeddings.npy
ls -lh src/outputs_poc/embeddings.paths.npy
ls -lh src/outputs_poc/knn_indices.npy
ls -lh src/outputs_poc/umap_2d.npy
ls -lh src/outputs_poc/m06_metrics.json
ls -lh src/outputs_poc/m06_*.png
ls -lh src/outputs_poc/m08_*.png

echo ""
echo "=== METRICS SUMMARY ==="
python -c "
import json
m = json.load(open('src/outputs_poc/m06_metrics.json'))
print(f'Clips:          {m[\"num_clips\"]:,}')
print(f'k_neighbors:    {m[\"k_neighbors\"]}')
print(f'')
print(f'          Easy     Hard')
print(f'Cycle@K   {m[\"easy\"][\"cycle_at_k\"]:5.1f}%   {m[\"hard\"][\"cycle_at_k\"]:5.1f}%')
print(f'Prec@K    {m[\"easy\"][\"prec_at_k\"]:5.1f}%   {m[\"hard\"][\"prec_at_k\"]:5.1f}%')
print(f'mAP@K     {m[\"easy\"][\"map_at_k\"]:5.3f}    {m[\"hard\"][\"map_at_k\"]:5.3f}')
print(f'nDCG@K    {m[\"easy\"][\"ndcg_at_k\"]:5.3f}    {m[\"hard\"][\"ndcg_at_k\"]:5.3f}')
print(f'Silhouet  {m[\"easy\"][\"silhouette\"]:5.3f}    {m[\"hard\"][\"silhouette\"]:5.3f}')
"
```

---

## Timeline

### RTX PRO 4000 (24GB VRAM — debug GPU)

| Step | Module | GPU? | Est. Time |
|------|--------|------|-----------|
| 0 | m04 SANITY (3 VLMs × 20) | GPU | ~15 min each |
| 1 | m04 BAKEOFF (3 VLMs × 2.5K) | GPU | ~8-9h each, ~26h total |
| 2 | m04b select winner | CPU | ~1 min |
| 3 | m04 FULL (winner × 7.5K) | GPU | ~26h |
| 4 | m05 V-JEPA embed (10K) | GPU | ~2h |
| 5 | m06 FAISS metrics | GPU | ~5 min |
| 6 | m07 UMAP (cuML) | GPU | ~2 min |
| 7 | m08 plots | CPU | ~5 min |
| **Total** | | | **~54h GPU + ~10 min CPU** |

### RTX PRO 6000 Blackwell (96GB VRAM — production GPU)

| Step | Module | GPU? | Est. Time |
|------|--------|------|-----------|
| 0 | m04 SANITY (3 VLMs × 20) | GPU | ~5 min each |
| 1 | m04 BAKEOFF (3 VLMs × 2.5K) | GPU | ~1-2h each, ~4-7h total |
| 2 | m04b select winner | CPU | ~1 min |
| 3 | m04 FULL (winner × 7.5K) | GPU | ~2-4h |
| 4 | m05 V-JEPA embed (10K) | GPU | ~2h |
| 5 | m06 FAISS metrics | GPU | ~5 min |
| 6 | m07 UMAP (cuML) | GPU | ~2 min |
| 7 | m08 plots | CPU | ~5 min |
| **Total** | | | **~10-15h GPU + ~10 min CPU** |

---

## Dependency Graph

```
Step 0: m04 --SANITY (qwen, videollama, llava) → validate before bake-off
                    │
                    ↓
Step 1: m04 --BAKEOFF (qwen)  ─┐
Step 1: m04 --BAKEOFF (videollama) ─┤→ Step 2: m04b (pick winner) → Step 3: m04 --FULL (winner)
Step 1: m04 --BAKEOFF (llava)  ─┘                                            │
                                                                             ↓
Step 4: m05 V-JEPA embed ──────────────────────────────────────────→ Step 5: m06 FAISS metrics
                                                                             │
                                                                             ↓
                                                                     Step 6: m07 UMAP
                                                                             │
                                                                             ↓
                                                                     Step 7: m08 plots
```

NOTE: Step 4 (m05) has NO dependency on Steps 1-3. It can run in PARALLEL with the bake-off if you have 2 GPUs. On single GPU, run sequentially as listed above.

All clips stream from HF — no local data/clips needed on GPU server.
