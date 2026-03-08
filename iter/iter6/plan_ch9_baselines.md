# Plan: Ch9 Baselines + True Overlap@K (Tasks 16-20)

## Context

Ch9 evaluates frozen V-JEPA on 10K Indian urban clips. Current results (Prec@K=18.73%, mAP@K=0.106) are **meaningless without baselines** — "18.73% compared to what?" We need 4 baselines + True Overlap@K to complete Ch9 for the paper. All code development is on M1 Mac (CPU), GPU runs later.

---

## Architecture Decision: Keep Native Dims (no projection)

| Encoder | Model | Dim | Type |
|---------|-------|-----|------|
| V-JEPA (existing) | `vjepa2-vitg-fpc64-384` | 1408 | video (all frames) |
| Random | — | 1408 | synthetic |
| DINOv2 | `dinov2-vitl14` | 1024 | image (middle frame) |
| CLIP | `clip-vit-large-patch14` | 768 | image (middle frame) |
| Shuffled V-JEPA | `vjepa2-vitg-fpc64-384` | 1408 | video (shuffled frames) |

**No zero-padding or projection.** FAISS is already dimension-agnostic (`d = embeddings.shape[1]`). Metrics (Prec@K, mAP@K, etc.) are dimensionless ratios — comparable across dims. This follows DINOv2/CLIP/V-JEPA paper conventions.

---

## Files: 3 New + 3 Modified

### NEW FILES

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `src/m05b_baselines.py` | Generate embeddings for all 4 baselines | ~500 |
| `src/m05c_true_overlap.py` | Augmented V-JEPA embeddings for True Overlap@K | ~300 |
| `src/m08b_compare.py` | CPU-only multi-encoder comparison plots + LaTeX table | ~250 |

### MODIFIED FILES

| File | Change | Scope |
|------|--------|-------|
| `src/utils/config.py` | Add `ENCODER_REGISTRY`, `get_encoder_files()`, `add_encoder_arg()` | ~40 lines added |
| `src/m06_faiss_metrics.py` | Add `--encoder` flag + `--true-overlap` flag + `compute_true_overlap_at_k()` | ~30 lines changed |
| `src/m07_umap.py` | Add `--encoder` flag, encoder-aware file paths | ~8 lines changed |

---

## Phase 1: `src/utils/config.py` — Encoder Registry

Add after existing constants:

```python
ENCODER_REGISTRY = {
    "vjepa":           {"model_id": "facebook/vjepa2-vitg-fpc64-384",    "dim": 1408, "type": "video",           "suffix": ""},
    "random":          {"model_id": None,                                 "dim": 1408, "type": "synthetic",       "suffix": "_random"},
    "dinov2":          {"model_id": "facebook/dinov2-vitl14",            "dim": 1024, "type": "image",            "suffix": "_dinov2"},
    "clip":            {"model_id": "openai/clip-vit-large-patch14",     "dim": 768,  "type": "image",            "suffix": "_clip"},
    "vjepa_shuffled":  {"model_id": "facebook/vjepa2-vitg-fpc64-384",   "dim": 1408, "type": "video_shuffled",   "suffix": "_vjepa_shuffled"},
}

def get_encoder_files(encoder: str, output_dir: Path) -> dict:
    """Return file paths for a given encoder. suffix="" for vjepa (backward compat)."""
    sfx = ENCODER_REGISTRY[encoder]["suffix"]
    return {
        "embeddings": output_dir / f"embeddings{sfx}.npy",
        "paths":      output_dir / f"embeddings{sfx}.paths.npy",
        "metrics":    output_dir / f"m06_metrics{sfx}.json",
        "knn_indices": output_dir / f"knn_indices{sfx}.npy",
        "umap_2d":    output_dir / f"umap_2d{sfx}.npy",
    }

def add_encoder_arg(parser):
    parser.add_argument("--encoder", default="vjepa", choices=list(ENCODER_REGISTRY))
```

> **Key:** `suffix=""` for vjepa ensures backward compatibility with existing `embeddings.npy`, `m06_metrics.json`.

---

## Phase 2: `src/m05b_baselines.py` — 4 Baseline Generators

**CLI:**
```bash
python -u src/m05b_baselines.py --encoder random --SANITY 2>&1 | tee logs/m05b_random.log
python -u src/m05b_baselines.py --encoder dinov2 --FULL --subset data/subset_10k.json 2>&1 | tee logs/m05b_dinov2.log
python -u src/m05b_baselines.py --encoder clip --FULL --subset data/subset_10k.json 2>&1 | tee logs/m05b_clip.log
python -u src/m05b_baselines.py --encoder vjepa_shuffled --FULL --subset data/subset_10k.json 2>&1 | tee logs/m05b_shuffled.log
```

**4 generator functions:**

1. **`generate_random()`** (Task 16) — CPU-only. Load `embeddings.paths.npy` for clip keys -> `np.random.RandomState(42).randn(N, 1408)` -> L2-normalize. No HF streaming needed.
2. **`generate_dinov2()`** (Task 17) — GPU. HF stream -> decode video -> extract middle frame -> `AutoImageProcessor` + `AutoModel("facebook/dinov2-vitl14")` -> `last_hidden_state[:, 0, :]` (CLS token) -> 1024-dim -> L2-normalize.
3. **`generate_clip()`** (Task 19) — GPU. HF stream -> decode video -> extract middle frame -> `CLIPProcessor` + `CLIPModel` -> `model.get_image_features()` -> 768-dim -> L2-normalize.
4. **`generate_shuffled_vjepa()`** (Task 18) — GPU. HF stream -> decode video -> `torch.randperm(T)` shuffle frames -> V-JEPA processor -> `model(pixel_values_videos=, skip_predictor=True)` -> `last_hidden_state.mean(dim=1)` -> 1408-dim.

**Shared infrastructure:**
- HF streaming: import `_create_stream`, `get_clip_key`, `decode_video_bytes` from `m05_vjepa_embed`
- Clip key alignment: load existing `embeddings.paths.npy` as reference key order, reorder results to match
- Producer-consumer pattern for GPU encoders (DINOv2/CLIP/Shuffled)
- Checkpoint/resume: `.m05b_{encoder}_checkpoint.npz` every 500 clips
- tqdm + wandb + tee logging (CLAUDE.md mandatory checklist)

**Output:** `embeddings_{encoder}.npy` + `embeddings_{encoder}.paths.npy` in `outputs_poc/`

---

## Phase 3: `src/m06_faiss_metrics.py` — Add `--encoder` flag

**Changes (~30 lines):**

1. Import `add_encoder_arg`, `get_encoder_files`, `ENCODER_REGISTRY` from config
2. Add `add_encoder_arg(parser)` to argparse (line ~925)
3. Replace hardcoded file resolution (lines ~951-959) with:
   ```python
   files = get_encoder_files(args.encoder, output_dir)
   emb_file = files["embeddings"]
   metrics_output = files["metrics"]
   knn_output = files["knn_indices"]
   # tags_file stays shared (same tags for all encoders)
   ```
4. Print encoder info: `Encoder: {args.encoder} (dim={info['dim']})`
5. Add `--true-overlap` flag (Phase 5)

> **No other logic changes.** `build_faiss_index()` already reads `d = embeddings.shape[1]`. All metric functions are dimension-agnostic.

---

## Phase 4: `src/m07_umap.py` — Add `--encoder` flag

**Changes (~8 lines):**

1. Import `add_encoder_arg`, `get_encoder_files` from config
2. Add `add_encoder_arg(parser)` (after line 37)
3. Replace line 55 (`emb_file = ...`) with:
   ```python
   files = get_encoder_files(args.encoder, output_dir)
   emb_file = files["embeddings"]
   ```
4. Replace line 84 (`out_path = output_dir / "umap_2d.npy"`) with:
   ```python
   out_path = files["umap_2d"]
   ```

---

## Phase 5: `src/m05c_true_overlap.py` + m06 integration (Task 20)

**m05c generates two augmented embedding sets:**
- Stream same clips -> apply 2 different augmentations -> V-JEPA embed each
- View A: `RandomResizedCrop(scale=0.4-1.0)` + `ColorJitter`
- View B: `RandomResizedCrop(scale=0.2-0.6)` + `GaussianBlur`
- Augmentation applied per-frame consistently within a clip (same crop params for all T frames)
- Output: `overlap_augA.npy` (N, 1408) + `overlap_augB.npy` (N, 1408)

**m06 integration — add `--true-overlap` flag:**
```python
def compute_true_overlap_at_k(output_dir: Path, k: int) -> float:
    """Load augA/augB, build FAISS on each, compute kNN IoU."""
    aug_a = np.load(output_dir / "overlap_augA.npy")
    aug_b = np.load(output_dir / "overlap_augB.npy")
    # For each clip i: kNN_A(i) vs kNN_B(i), compute IoU
    # Return mean IoU * 100
```

> When `--true-overlap` is passed, this replaces the dim-split approximation in the output JSON. The existing dim-split is kept as fallback when augmented embeddings don't exist.

---

## Phase 6: `src/m08b_compare.py` — Multi-Encoder Comparison

**CPU-only matplotlib.** Reads `m06_metrics_*.json` for all available encoders.

```bash
python -u src/m08b_compare.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m08b_compare.log
```

**Generates:**
1. **Grouped bar chart** — 6 metrics x 5 encoders, Easy vs Hard
2. **Radar plot** — one polygon per encoder, 5 normalized metrics
3. **LaTeX table** — paper-ready, saved as `m08b_comparison_table.tex`
4. **Terminal summary** — pretty-printed table

---

## Execution Order (on M1 Mac — code only)

| Step | File | What | Testable on Mac? |
|------|------|------|-----------------|
| 1 | `config.py` | Add ENCODER_REGISTRY + helpers | `py_compile` |
| 2 | `m05b_baselines.py` | Write all 4 generators | `py_compile` + `--help` + `--encoder random --SANITY` (CPU) |
| 3 | `m06_faiss_metrics.py` | Add `--encoder` flag | `py_compile` + `--help` |
| 4 | `m07_umap.py` | Add `--encoder` flag | `py_compile` + `--help` |
| 5 | `m05c_true_overlap.py` | Write augmentation pipeline | `py_compile` + `--help` |
| 6 | `m06` add `--true-overlap` | True Overlap@K integration | `py_compile` + `--help` |
| 7 | `m08b_compare.py` | Write comparison plots | `py_compile` + `--help` |
| 8 | Dependencies | Update `requirements_gpu.txt` if needed | verify |

---

## Dependencies Check

No new pip packages needed:
- DINOv2 + CLIP: both in `transformers` (already pinned >=4.57.0)
- `torchvision.transforms`: already installed with PyTorch
- `faiss-gpu`: already installed
- `cuml`: already installed (for m07)

---

## Verification (M1 Mac)

For each new/modified file:
```bash
source venv_walkindia/bin/activate
python3 -m py_compile src/<file>.py
python3 src/<file>.py --help
# Random baseline only (CPU-safe):
python3 src/m05b_baselines.py --encoder random --SANITY
```
