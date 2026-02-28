# Execution Plan: BAKEOFF Experiment (m04 → m08)

STATUS: m00-m03 COMPLETED (Mac CPU). m04-m08 code ready for GPU.
Dataset: https://huggingface.co/datasets/anonymousML123/walkindia-200k (private, HF_TOKEN in .env)

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
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB')"
python -c "import faiss; print(f'FAISS GPUs: {faiss.get_num_gpus()}')"
python -c "import cuml; print(f'cuML: {cuml.__version__}')"
python -c "import wandb; print(f'wandb: {wandb.__version__}')"

# HF auth (private repo)
python -c "from dotenv import load_dotenv; load_dotenv(); import os; t=os.getenv('HF_TOKEN'); print(f'HF_TOKEN: {t[:10]}...' if t else 'MISSING')"

# Subset file exists
python -c "import json; d=json.load(open('data/subset_10k.json')); print(f'Subset: {d[\"n\"]} clips from {d[\"num_videos\"]} videos')"
```

---

## Step 1: VLM Bake-off — 3 VLMs × 2,500 clips each

Each VLM tags the first 2,500 clips from the 10K subset (streams from HF, no local clips).

```bash
python -u src/m04_vlm_tag.py --model qwen --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_qwen_poc.log
python -u src/m04_vlm_tag.py --model videollama --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_videollama_poc.log
python -u src/m04_vlm_tag.py --model keye --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_keye_poc.log
```

### Verify Step 1 output

```bash
# 3 bakeoff tag files exist, each with 2500 clips
for model in qwen videollama keye; do
  python -c "import json; t=json.load(open('src/data/bakeoff/tags_${model}.json')); print(f'tags_${model}.json: {len(t)} clips')"
done
```

**Expected:** 3 files × 2,500 clips each. ~1h total GPU.

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
```

**Expected:** `vlm_comparison.json` with winner name + scores. `vlm_comparison.png` + `.pdf` plots.

---

## Step 3: Winner tags remaining 7,500 clips

Winner VLM runs --FULL on the 10K subset. Resumes from bake-off checkpoint (already has 2,500 tagged), so only ~7,500 new clips processed.

```bash
# Replace <winner> with output from Step 2 (qwen, videollama, or keye)
python -u src/m04_vlm_tag.py --model <winner> --FULL --subset data/subset_10k.json 2>&1 | tee logs/m04_full_poc.log
```

### Verify Step 3 output

```bash
python -c "import json; t=json.load(open('src/outputs_poc/tags.json')); print(f'tags.json: {len(t)} clips, fields: {len(t[0].keys())}')"
# Expect: ~10,000 clips, 33 fields each
```

**Expected:** `outputs_poc/tags.json` with ~10K clips × 33 fields. ~45 min GPU.

---

## Step 4: V-JEPA 2 embeddings (streams from HF)

V-JEPA 2 ViT-G (1B params, frozen) encodes each clip → 1408-dim embedding. Producer-consumer pipeline with torch.compile.

```bash
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

## Timeline (RTX Pro 4000 — 24GB VRAM)

| Step | Module | GPU? | Est. Time |
|------|--------|------|-----------|
| 1 | m04 BAKEOFF (3 VLMs × 2.5K) | GPU | ~1h |
| 2 | m04b select winner | CPU | ~1 min |
| 3 | m04 FULL (winner × 7.5K) | GPU | ~45 min |
| 4 | m05 V-JEPA embed (10K) | GPU | ~2h |
| 5 | m06 FAISS metrics | GPU | ~5 min |
| 6 | m07 UMAP (cuML) | GPU | ~2 min |
| 7 | m08 plots | CPU | ~5 min |
| **Total** | | | **~4h GPU + ~10 min CPU** |

---

## Dependency Graph

```
Step 1: m04 --BAKEOFF (qwen)  ─┐
Step 1: m04 --BAKEOFF (videollama) ─┤→ Step 2: m04b (pick winner) → Step 3: m04 --FULL (winner)
Step 1: m04 --BAKEOFF (keye)  ─┘                                            │
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
