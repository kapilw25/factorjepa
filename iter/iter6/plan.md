# Execution Plan: WalkIndia-200K Pipeline

STATUS: m00-m03 COMPLETED (Mac CPU). m04-m07 ready for GPU.
Dataset: https://huggingface.co/datasets/anonymousML123/walkindia-200k

## Module Tree

```
src/
├── m00_data_prep.py              # Parse YT_videos_raw.md → JSON, word freq, city matrix
├── m00b_fetch_durations.py       # Fetch YT video durations via yt-dlp metadata (no download)
├── m00c_sample_subset.py         # Video-level uniform 10K subset sampling for POC runs
├── m01_download.py               # Download 714 YT videos at 480p (Mac, aria2c)
├── m02_scene_detect.py           # Greedy scene-aware split → ffmpeg encode clips (optional --keyframes)
├── m02b_scene_fetch_duration.py  # Scan all clips, output clip_durations.json
├── m03_pack_shards.py            # Pack clips into WebDataset TAR shards → upload to HF
├── m04_vlm_tag.py                # [GPU] VLM tagging (--model qwen|videollama|keye, --BAKEOFF|--FULL)
├── m04b_vlm_select.py            # [CPU] Bake-off comparison → pick winner VLM
├── m05_vjepa_embed.py            # [GPU] V-JEPA 2 embeddings (ViT-G, 1408-dim, HF streaming)
├── m06_faiss_metrics.py          # FAISS kNN: 9 metrics + Hard/Easy mode
├── m07_umap_plot.py              # UMAP visualization + kNN grids
└── utils/
    ├── __init__.py
    ├── config.py                 # Paths, constants, shared utility functions
    ├── export_metadata.py        # tags.json → metadata.jsonl per leaf dir
    ├── hf_utils.py               # HF auth, README gen, metadata upload
    └── tag_taxonomy.json         # 11 tag fields + confidence schema for VLMs
```

---

## Phase 1: Data Preparation (Mac CPU) — COMPLETED

m00→m00b→m01→m02→m02b→m03: 714 videos → 115,687 clips → 116 TAR shards → HF upload (121.5 GB).
m00c: Video-level uniform sampling → `data/subset_10k.json` (10K clip keys, seed=42).

---

## Phase 2: POC Pipeline on 10K Subset (GPU Server)

### Hardware: A100-40GB (sufficient)

| Component | Size |
|---|---|
| Python venv + packages | ~15 GB |
| HF model cache (4 models) | ~50 GB |
| Outputs (embeddings, tags, plots) | ~1 GB |
| Local clips needed | **0 GB** (all HF streaming) |
| **Total disk volume** | **100 GB** |

### Setup

```bash
git clone <repo> && cd LLM_asAgent_3D_SR
./setup_env_uv.sh --gpu
source venv_walkindia/bin/activate
```

### GPU Execution Sequence (all sequential, peak VRAM ~16GB)

**Step 1: Bake-off — 3 VLMs × 2,500 clips each (streams from HF)**

| | Path |
|---|---|
| INPUT | HF WebDataset stream, `src/utils/tag_taxonomy.json`, `data/subset_10k.json` |
| OUTPUT | `src/data/bakeoff/tags_{qwen,videollama,keye}.json` |

```bash
python -u src/m04_vlm_tag.py --model qwen --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_qwen_poc.log
python -u src/m04_vlm_tag.py --model videollama --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_videollama_poc.log
python -u src/m04_vlm_tag.py --model keye --BAKEOFF --subset data/subset_10k.json 2>&1 | tee logs/m04_bakeoff_keye_poc.log
```

**Step 2: Pick winner (CPU)**

| | Path |
|---|---|
| INPUT | `src/data/bakeoff/tags_{qwen,videollama,keye}.json`, `src/utils/tag_taxonomy.json` |
| OUTPUT | `src/data/bakeoff/vlm_comparison.{json,png,pdf}` |

```bash
python -u src/m04b_vlm_select.py 2>&1 | tee logs/m04b_vlm_select.log
```

**Step 3: Winner tags remaining 7.5K clips (streams from HF)**

| | Path |
|---|---|
| INPUT | HF WebDataset stream, `src/utils/tag_taxonomy.json`, `data/subset_10k.json` |
| OUTPUT | `src/outputs_poc/tags.json` (~10K clips, 33 fields each) |

```bash
python -u src/m04_vlm_tag.py --model <winner> --FULL --subset data/subset_10k.json 2>&1 | tee logs/m04_full_poc.log
```

**Step 4: V-JEPA 2 embeddings (streams from HF)**

| | Path |
|---|---|
| INPUT | HF WebDataset stream, `data/subset_10k.json` |
| OUTPUT | `src/outputs_poc/embeddings.npy` (10K × 1408), `src/outputs_poc/embeddings.paths.npy` |

```bash
python -u src/m05_vjepa_embed.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m05_vjepa_embed_poc.log
```

**Step 5: FAISS metrics (requires Step 3 + Step 4 done)**

| | Path |
|---|---|
| INPUT | `src/outputs_poc/embeddings.npy`, `src/outputs_poc/tags.json`, `src/utils/tag_taxonomy.json` |
| OUTPUT | `src/outputs_poc/m06_metrics.json`, `src/outputs_poc/m06_*.{png,pdf}` |

```bash
python -u src/m06_faiss_metrics.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06_faiss_metrics_poc.log
```

**Step 6: UMAP + kNN plots (CPU)**

| | Path |
|---|---|
| INPUT | `src/outputs_poc/embeddings.npy`, `src/outputs_poc/tags.json`, `src/outputs_poc/m06_metrics.json` |
| OUTPUT | `src/outputs_poc/m07_umap.{png,pdf}`, `src/outputs_poc/m07_confusion_matrix.{png,pdf}`, `src/outputs_poc/m07_knn_grid.{png,pdf}` |

```bash
python -u src/m07_umap_plot.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m07_umap_plot_poc.log
```

**Estimated time (A100-40GB):** ~4h GPU + 25min CPU for POC (10K subset).

---

## Phase 3: Full Scale (115K, after POC validates) — FUTURE

Same commands as Phase 2 but **without** `--subset` flag. Outputs go to `src/outputs/` instead of `src/outputs_poc/`.

```bash
python -u src/m04_vlm_tag.py --model qwen --BAKEOFF 2>&1 | tee logs/m04_bakeoff_qwen.log
python -u src/m04_vlm_tag.py --model videollama --BAKEOFF 2>&1 | tee logs/m04_bakeoff_videollama.log
python -u src/m04_vlm_tag.py --model keye --BAKEOFF 2>&1 | tee logs/m04_bakeoff_keye.log
python -u src/m04b_vlm_select.py 2>&1 | tee logs/m04b_vlm_select.log
python -u src/m04_vlm_tag.py --model <winner> --FULL 2>&1 | tee logs/m04_full.log
python -u src/m05_vjepa_embed.py --FULL 2>&1 | tee logs/m05_vjepa_embed_full.log
python -u src/m06_faiss_metrics.py --FULL 2>&1 | tee logs/m06_faiss_metrics_full.log
python -u src/m07_umap_plot.py --FULL 2>&1 | tee logs/m07_umap_plot_full.log
```

---

## Execution Order (Dependency)

```
m00 → m00b → m01 → m02 → m02b → m03     [COMPLETED — Mac CPU]
                                  ↓
                               m00c        [COMPLETED — data/subset_10k.json]
                                  ↓
                    ┌─────────────┼──────────────┐
                    ↓             ↓              ↓
               m04 BAKEOFF    m04 BAKEOFF    m05 V-JEPA
               (qwen)        (videollama)    (parallel)
               m04 BAKEOFF
               (keye)
                    ↓
                  m04b          (CPU — pick winner)
                    ↓
               m04 --FULL      (winner on remaining)
                    ↓
                    └─────────→ m06 ←───────┘
                                 ↓
                                m07
```

All clips stream from HF — no local data/clips needed on GPU server.
All architectural details, design decisions, and diagrams → see `plan_HIGH_LEVEL.md`
