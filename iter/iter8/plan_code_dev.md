# m10 Grounded-SAM Pivot — Code Development Plan

## Architecture change

```
CURRENT (broken):   tags.json notable_objects → SAM3.1 text grounding → pixel masks
                    SAM3.1 does BOTH detection AND segmentation. Bad at detection for
                    Indian objects (Evidence: 10/15 SANITY clips had n_agents=0,
                    concept_recall=0.0, wires/signage masked instead of agents).

NEW (Grounded-SAM): fixed taxonomy → Grounding DINO text detection → boxes →
                    SAM3.1 box-prompted mask refinement + temporal propagation.
                    DINO: Objects365+GoldG (365 categories, strong open-vocab).
                    SAM3.1: box → pixel-perfect mask (its strength, always has been).
```

## Model choice

- **Grounding DINO**: `IDEA-Research/grounding-dino-base` (Swin-B, 233M params, ~1.2 GB VRAM). Loaded from HuggingFace Hub — no new pip dep (ships in `transformers>=4.38`).
- **SAM3.1**: unchanged (`build_sam3_predictor(version="sam3.1", use_fa3=False)`).
- **Thresholds**: `box_threshold=0.25`, `text_threshold=0.20` (below HF defaults of 0.35/0.25 — optimize for recall, since D_A quality filter at m11 catches false positives).

## Fixed agent taxonomy (17 categories, accuracy-first)

Goal = Surgery > ExPLoRA > Frozen on Prec@K. Accuracy requires high recall of agents (missed agents leak into D_L; poor category separation breaks D_I cross-category interaction mining). Extended beyond tag_taxonomy.json's 11 agent items with 6 finer-grained categories that DINO distinguishes in Objects365.

```
"pedestrian. person. vendor. policeman. bus. car. truck. motorcycle. bicycle. scooter. auto rickshaw. cycle rickshaw. cart. cow. dog. bull. buffalo."
```

**Rationale per category:**
- `person` (distinct from `pedestrian`): Objects365 has 1M+ annotations; covers drivers, passengers, seated people. Recall boost.
- `policeman`: distinct uniform, low FP. High-value D_I pairs (policeman × vehicle).
- `motorcycle` + `bicycle` + `scooter` (split from tag_taxonomy's coarse `bike`): DINO's three separate Objects365 classes. Scooters are 30%+ of Indian two-wheelers.
- `cart` (for tag_taxonomy's `handcart`): DINO-friendly generic English.
- `vendor` (for `street_vendor`), `cow` (for `sacred_cow`), `dog` (for `stray_dog`): generic English names DINO was trained on.
- `bull`, `buffalo`: visually distinct from cow, common in rural WalkIndia clips (Goa, Kerala).
- Dropped from initial proposal: `child` (DINO's `person` covers it), `van` (redundant with truck/car), `rickshaw` (redundant with auto/cycle rickshaw).

**Alias map for `concept_recall` reporting** (DINO → VLM coarser taxonomy):
```python
DINO_TO_TAG = {
    "pedestrian":     "pedestrian",
    "person":         "pedestrian",
    "vendor":         "street_vendor",
    "policeman":      None,              # bonus detection — no VLM equivalent
    "bus":            "bus",
    "car":            "car",
    "truck":          "truck",
    "motorcycle":     "bike",
    "bicycle":        "bike",
    "scooter":        "bike",
    "auto rickshaw":  "auto_rickshaw",
    "cycle rickshaw": "cycle_rickshaw",
    "cart":           "handcart",
    "cow":            "sacred_cow",
    "bull":           "sacred_cow",
    "buffalo":        "sacred_cow",
    "dog":            "stray_dog",
}
```
`concept_recall = |VLM-expected ∩ {mapped DINO detections}| / |VLM-expected|`. `None` entries are bonus agents (count in `n_agents` but don't inflate recall).

## Code changes in `src/m10_sam_segment.py`

### 1. `get_agent_prompts()` → rewrite to `get_fixed_taxonomy()`
- Drop per-clip VLM object list. Taxonomy is clip-independent.
- Returns `(compound_prompt: str, category_list: list[str])` read from `ch11_surgery.yaml`.
- Called ONCE in `main()`, not per-clip.
- tags.json still loaded but only for `scene_type` (plots) and expected-object set (concept_recall diagnostic).

### 2. New `load_grounding_dino(model_id: str)` function
- Mirrors `load_sam3()` pattern.
- Loads `AutoProcessor.from_pretrained(model_id)` + `AutoModelForZeroShotObjectDetection.from_pretrained(model_id, torch_dtype=float16)`.
- Moves to `cuda`, `.eval()`, `torch.compile(model)`.
- Returns `(processor, model)`.
- Called ONCE in `main()` before the clip loop.

### 3. Rewrite `segment_clip()` signature
```
segment_clip(
    sam_predictor, dino_processor, dino_model,
    frame_dir, compound_prompt, category_list,
    dilation_px, min_confidence, min_mask_area_pct,
    box_threshold, text_threshold,
) → dict
```
Body (replaces current lines 113-233):

1. Load frame 0 PIL from `frame_dir/00000.jpg`.
2. DINO forward on frame 0:
   - `inputs = dino_processor(images=frame_0, text=compound_prompt, return_tensors="pt").to("cuda")`
   - `outputs = dino_model(**inputs)`
   - `results = dino_processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold, text_threshold, target_sizes=[(H,W)])[0]`
   - Returns `boxes` (abs pixel coords), `labels` (category strings matched from compound_prompt), `scores`.
3. If `len(boxes) == 0` → return early with `n_agents=0, agent_pixel_ratio=0, centroids={}, per_object={}`.
4. Group boxes by category label. For each category with ≥1 box:
   - Start SAM3.1 session on `frame_dir`.
   - `sam.add_prompt(session_id, frame_index=0, bounding_boxes=boxes_for_this_cat)`.
   - Propagate via `sam.handle_stream_request(type="propagate_in_video", session_id=...)` (unchanged).
   - Accept masks via existing `_accept_mask()` (unchanged).
   - Close session.
   - Increment `obj_id_offset` by 100 (reserves per-category ID ranges for D_I mining).
5. Build `agent_mask` union + `per_object` dict + `centroids` dict — unchanged from current code.

### 4. `concept_recall` + tag alias
Add module-level constant:
```python
DINO_TO_TAG = {
    "motorcycle":     "bike",
    "bicycle":        "bike",
    "auto rickshaw":  "auto_rickshaw",
    "cycle rickshaw": "cycle_rickshaw",
    "cart":           "handcart",
    "vendor":         "street_vendor",
    "cow":            "sacred_cow",
    "dog":            "stray_dog",
}
```
In `main()`'s clip loop, convert DINO-detected categories back to tag_taxonomy names before computing `concept_recall = n_matched / max(n_expected, 1)` where `n_expected = len(tags[clip_key]["notable_objects"] ∩ agents_only)`.

### 5. `main()` updates
- Load DINO once before clip loop (~lines 557-561, next to SAM3 load).
- Pass `(dino_processor, dino_model, compound_prompt, category_list)` into `segment_clip()`.
- Read `box_threshold`, `text_threshold`, `agent_taxonomy` from `ch11_surgery.yaml` (factor_datasets.grounding_dino block).

## Config additions: `configs/train/ch11_surgery.yaml`

```yaml
factor_datasets:
  grounding_dino:
    model_id: "IDEA-Research/grounding-dino-base"
    box_threshold: 0.25
    text_threshold: 0.20
    agent_taxonomy:
      - pedestrian
      - person
      - vendor
      - policeman
      - bus
      - car
      - truck
      - motorcycle
      - bicycle
      - scooter
      - auto rickshaw
      - cycle rickshaw
      - cart
      - cow
      - dog
      - bull
      - buffalo
  # existing fields kept:
  sam_model: ...
  agent_dilation_pixels: ...
  min_confidence: ...
  min_mask_area_pct: ...
```

## What does NOT change (keep diff reviewable)

- `mine_interactions()` — geometry-only, category-aware via `per_object` obj_id ranges (already supported).
- `_accept_mask()` — same min area + confidence filters.
- `.npz` save format — `agent_mask`, `layout_mask`, `centroids_json`, `interactions_json`, `mid_frame_rgb` all unchanged.
- `plot_overlay_per_clip()`, `plot_agent_stats()` — unchanged.
- Quality gate (4 checks: pixel ratio, mask confidence, clips with agents) — unchanged.
- `src/m11_factor_datasets.py` — untouched. Consumes m10's same output format.
- `src/m09_pretrain.py` — untouched.

## Requirements

No `requirements_gpu.txt` edit needed. `transformers>=4.57.0,<5.0` already covers Grounding DINO. Weights auto-download from HF Hub on first `from_pretrained()` call (`HF_HUB_ENABLE_HF_TRANSFER=1` already set → fast).

## Validation

After implementing, re-run SANITY on 15 clips (same data as before):
```bash
rm -rf outputs/sanity/m10_sam_segment/
python -u src/m10_sam_segment.py --SANITY \
    --local-data data/val_1k_local --no-wandb \
    2>&1 | tee logs/m10_sanity_grounded_v1.log
```

Expected improvements over SAM3.1-text-only:
- `mean_concept_recall` ≥ 0.6 (vs 0.0 on bad clip before)
- `mean_agent_pixel_ratio` in [5%, 40%] (vs 1.2% before)
- `clips_with_agents_pct` ≥ 80% (vs current quality gate threshold 50%)
- Per-clip overlay images (`m10_overlay_verify/*.png`) show red masks on people/vehicles, not roofs/walls.

If validation passes → unblock Step B (m11), then C/D/E per `runbook.md`.
