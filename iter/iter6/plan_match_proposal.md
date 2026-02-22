⏺ Here's the comparison. I'll structure it as: what the proposal says vs what the plans actually implement.                                                                                 
                                                                                                                                                                                        
---                                                                                                                                                                                       
  Chapter 8 (Automatic Annotations) Discrepancies

1. Tag field count: 7 vs 11 >> LET's KEEP IT as per plan_HIGH_LEVEL.md or plan.md

┌────────────────────────┬───────────────────┬─────────────────────────────┐
│         Field          │ Proposal Table 11 │ plan.md (tag_taxonomy.json) │
├────────────────────────┼───────────────────┼─────────────────────────────┤
│ scene_type             │ yes               │ yes                         │
├────────────────────────┼───────────────────┼─────────────────────────────┤
│ time_of_day            │ yes               │ yes                         │
├────────────────────────┼───────────────────┼─────────────────────────────┤
│ weather                │ yes               │ yes                         │
├────────────────────────┼───────────────────┼─────────────────────────────┤
│ crowd_density          │ yes               │ yes                         │
├────────────────────────┼───────────────────┼─────────────────────────────┤
│ traffic_density        │ yes               │ yes                         │
├────────────────────────┼───────────────────┼─────────────────────────────┤
│ road_layout            │ yes               │ yes                         │
├────────────────────────┼───────────────────┼─────────────────────────────┤
│ notable_objects        │ yes               │ yes                         │
├────────────────────────┼───────────────────┼─────────────────────────────┤
│ road_surface           │ NO                │ yes                         │
├────────────────────────┼───────────────────┼─────────────────────────────┤
│ infrastructure_quality │ NO                │ yes                         │
├────────────────────────┼───────────────────┼─────────────────────────────┤
│ vegetation             │ NO                │ yes                         │
├────────────────────────┼───────────────────┼─────────────────────────────┤
│ lighting               │ NO                │ yes                         │
└────────────────────────┴───────────────────┴─────────────────────────────┘

plan.md added 4 extra fields (road_surface, infrastructure_quality, vegetation, lighting) that don't exist in the proposal.

2. Clip length: fixed 10s vs variable 4--10s  >> LET's KEEP IT as per plan_HIGH_LEVEL.md or plan.md

- Proposal: "fixed-length clips (typically ~10 seconds)", "10-second clips"
- plan.md: greedy scene-aware split producing [4s, 10s] range clips
- plan_HIGH_LEVEL.md: "4-10s cuts" (mermaid), "4-5s" (ASCII art)                                                                                                                                         
                                                

3. Quality control: proposed but not implemented

┌───────────────┬───────────────────────────────────────────────────────────────────────────────────────┬─────────────┬────────────────────────┐
│ Proposal Step │                                     What it says                                      │ In plan.md? │ In plan_HIGH_LEVEL.md? │
├───────────────┼───────────────────────────────────────────────────────────────────────────────────────┼─────────────┼────────────────────────┤
│ Step 7        │ Dual-prompt self-consistency (run 2 paraphrased prompts, accept only if they agree)   │ NO          │ NO                     │
├───────────────┼───────────────────────────────────────────────────────────────────────────────────────┼─────────────┼────────────────────────┤
│ Step 8        │ Per-field confidence thresholds (stricter for scene_type, looser for notable_objects) │ NO          │ NO                     │
├───────────────┼───────────────────────────────────────────────────────────────────────────────────────┼─────────────┼────────────────────────┤
│ Step 9        │ Human spot-checking audit set                                                         │ NO          │ NO                     │
└───────────────┴───────────────────────────────────────────────────────────────────────────────────────┴─────────────┴────────────────────────┘

plan_HIGH_LEVEL.md substitutes Cross-VLM agreement (3 VLMs) as quality control instead -- a different approach entirely.

4. Per-field confidence scores: missing from implementation

- Proposal: Each tag has confidence.* in [0,1]
- plan.md / plan_HIGH_LEVEL.md: No per-field confidence in the tag output schema

5. Provenance tracking: missing from implementation

- Proposal: Stores model version, prompt version, timestamp per clip
- plan.md: Not in the enriched JSON sidecar (only 8 metadata + 11 tags = 19 fields, no provenance)

6. Keyframe export: missing from implementation

- Proposal Step 3: "export 1--3 keyframes per clip for qualitative inspection and debugging"
- plan.md: m02_scene_detect outputs mp4 clips only, no keyframe extraction

---
Chapter 9 (Evaluating V-JEPA) Discrepancies

7. Metric naming: same concept, different names

┌────────────────────┬──────────────────────────────────────────┬────────────────────────────────────────────────────────┐
│   Proposal name    │                Plan name                 │                      Same metric?                      │
├────────────────────┼──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Cycle@k (Step 6)   │ Self-Consistency (plan.md m06)           │ YES -- "if A's NN is B, does B's NN include A?"        │
├────────────────────┼──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Overlap@K (Step 7) │ Transform Stability (plan_HIGH_LEVEL.md) │ YES -- "same clip, different crops -> same neighbors?" │
├────────────────────┼──────────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Prec@K (Step 9)    │ Cluster Purity (plan.md m06)             │ YES -- "% of kNN neighbors sharing same scene_type"    │
└────────────────────┴──────────────────────────────────────────┴────────────────────────────────────────────────────────┘

8. Many proposed metrics not implemented

┌──────────────────────────────────────┬────────────────────────────────────┬─────────────────────────┐
│           Proposal metric            │       In plan.md (m06/m07)?        │ In plan_HIGH_LEVEL.md?  │
├──────────────────────────────────────┼────────────────────────────────────┼─────────────────────────┤
│ Cycle@k / Self-Consistency           │ YES                                │ YES                     │
├──────────────────────────────────────┼────────────────────────────────────┼─────────────────────────┤
│ Overlap@K / Transform Stability      │ NO                                 │ YES (named differently) │
├──────────────────────────────────────┼────────────────────────────────────┼─────────────────────────┤
│ Prec@K / Cluster Purity              │ YES                                │ YES                     │
├──────────────────────────────────────┼────────────────────────────────────┼─────────────────────────┤
│ mAP@K (Step 10)                      │ NO                                 │ NO                      │
├──────────────────────────────────────┼────────────────────────────────────┼─────────────────────────┤
│ nDCG@K (Step 10)                     │ NO                                 │ NO                      │
├──────────────────────────────────────┼────────────────────────────────────┼─────────────────────────┤
│ Multi-attribute slices (Step 11)     │ NO                                 │ NO                      │
├──────────────────────────────────────┼────────────────────────────────────┼─────────────────────────┤
│ Confusion-style analysis (Step 12)   │ Partial (m07 has confusion matrix) │ NO                      │
├──────────────────────────────────────┼────────────────────────────────────┼─────────────────────────┤
│ Silhouette score (Step 8)            │ NO                                 │ NO                      │
├──────────────────────────────────────┼────────────────────────────────────┼─────────────────────────┤
│ Confidence threshold sweep (Step 14) │ NO                                 │ NO                      │
├──────────────────────────────────────┼────────────────────────────────────┼─────────────────────────┤
│ kNN neighbor grids (Step 5)          │ NO                                 │ NO                      │
├──────────────────────────────────────┼────────────────────────────────────┼─────────────────────────┤
│ Macro/micro averaging (Step 13)      │ NO                                 │ NO                      │
└──────────────────────────────────────┴────────────────────────────────────┴─────────────────────────┘

9. Hard vs Easy mode: not implemented

- Proposal: Defines exclusion window $\Delta t$ (±30s within same video) for "hard mode" retrieval. Reports both modes.
- plan.md: No exclusion window, no hard/easy distinction
- plan_HIGH_LEVEL.md: No mention

10. Train/val/test splits: not defined

- Proposal Step 1: "train/val/test splits by video_id" to prevent leakage
- plan.md: No split strategy defined
- plan_HIGH_LEVEL.md: No split strategy

11. Cross-VLM agreement: added by plan, not in proposal

- Proposal Ch 9: Uses a single VLM's tags
- plan_HIGH_LEVEL.md: Adds Cross-VLM agreement (Qwen3 + VideoLLaMA3 + InternVL2.5) as a key deliverable  >> LET's KEEP IT as per plan_HIGH_LEVEL.md or plan.md

12. Baselines: added by plan, not in proposal Ch 9

- Proposal: No explicit baselines in Ch 9
- plan_HIGH_LEVEL.md: Random embeddings, Shuffled V-JEPA, DINOv2, CLIP  >> LET's KEEP IT as per plan_HIGH_LEVEL.md or plan.md

---

Implementation Plan: Resolving Discrepancies 3--10
===================================================

Items [1, 2, 11, 12] → KEEP as per plan_HIGH_LEVEL.md / plan.md (no changes needed)
Items [3--10] → addressed below

---

### #3  Quality Control: Dual-Prompt / Thresholds / Human Audit

Proposal wants 3 QC steps. plan_HIGH_LEVEL.md substitutes Cross-VLM agreement (#11, KEPT).
Since Cross-VLM agreement (Qwen3 + VideoLLaMA3 + InternVL2.5) is already a stronger QC mechanism
than dual-prompt self-consistency (same model, paraphrased prompt), we do NOT add dual-prompt.

Action:
- Step 7 (dual-prompt): SKIP — Cross-VLM (#11) supersedes this
- Step 8 (per-field thresholds): IMPLEMENT via per-field confidence (#4 below)
- Step 9 (human audit): SKIP — Cross-VLM agreement across 3 VLMs is a stronger signal than manual spot-checking

Files touched: none (all QC handled by Cross-VLM pipeline in #11)

---

### #4  Per-Field Confidence Scores

Proposal: each tag field has `confidence.*` in [0, 1].
Current m04: outputs only the tag value, no confidence.

Action:
- Modify `build_tag_prompt()` in m04_qwen_tag.py to ask Qwen to output confidence per field
- Output schema changes from:
  ```json
  {"scene_type": "market", "weather": "clear", ...}
  ```
  to:
  ```json
  {
    "scene_type": "market", "confidence_scene_type": 0.92,
    "weather": "clear",    "confidence_weather": 0.85,
    ...
  }
  ```
- Update tag_taxonomy.json: add `"confidence": true` flag per field
- Enriched sidecar goes from 19 fields → 19 + 11 confidence = 30 fields
- m06_faiss_metrics.py: add `--confidence-threshold` sweep (proposal Step 14) using these scores

Files touched:
- src/utils/tag_taxonomy.json (add confidence flag)
- src/m04_qwen_tag.py (modify prompt + JSON parse to extract confidence)
- src/m06_faiss_metrics.py (consume confidence for threshold sweep, see #8)

---

### #5  Provenance Tracking

Proposal: stores model_version, prompt_version, timestamp per clip.
Current m04: no provenance in output.

Action:
- Add 3 provenance fields to each clip's tag dict in tags.json:
  ```json
  {
    "scene_type": "market", ...,
    "_model": "Qwen/Qwen3-VL-8B-Instruct",
    "_prompt_version": "v1.0",
    "_tagged_at": "2026-02-21T14:30:00Z"
  }
  ```
- Prefix with `_` to separate from tag fields (convention already used in tag_taxonomy.json)
- Sidecar: 30 fields + 3 provenance = 33 fields

Files touched:
- src/m04_qwen_tag.py (add provenance dict to each clip's output)
- No config change needed (model ID already in config.py as QWEN_MODEL_ID)

---

### #6  Keyframe Export

Proposal Step 3: "export 1--3 keyframes per clip for qualitative inspection and debugging"
Current m02: outputs mp4 only.

Action:
- Add `--keyframes` flag to m02_scene_detect.py
- For each clip, extract 1 keyframe (middle frame) via ffmpeg:
  `ffmpeg -ss <mid> -i clip.mp4 -vframes 1 -q:v 2 keyframe.jpg`
- Save alongside clips: `clips/<section>/<video_id>_<clip_idx>.jpg`
- Optional: also export in m04 for debugging VLM outputs (show keyframe + tags side by side)

Files touched:
- src/m02_scene_detect.py (add --keyframes flag + ffmpeg extraction)

---

### #7  Metric Naming Alignment

Same metrics, different names. Align code to use BOTH names (proposal name as primary, plan name as alias).

Current → Aligned:
- `compute_self_consistency()` → `compute_cycle_at_k()` (alias: self_consistency)
- `compute_cluster_purity()` → `compute_prec_at_k()` (alias: cluster_purity)
- Transform Stability stays as `compute_overlap_at_k()` (new function, see #8)

Action:
- Rename functions in m06_faiss_metrics.py
- Update metrics JSON keys: `"cycle_at_k"`, `"prec_at_k"` (keep old keys as aliases for backward compat)
- Update m07_umap_plot.py plot labels

Files touched:
- src/m06_faiss_metrics.py (rename functions + output keys)
- src/m07_umap_plot.py (update labels)

---

### #8  Missing Metrics — ADD to m06_faiss_metrics.py

6 new metric functions to implement:

| Metric                      | Proposal Step | Implementation                                                       | Difficulty |
|-----------------------------|---------------|----------------------------------------------------------------------|------------|
| Overlap@K (Transform Stab.) | Step 7        | Augment clip (crop/flip), embed both, compare kNN sets               | HIGH (needs m05 re-embed with augmented clips) |
| mAP@K                       | Step 10       | Mean Average Precision: ranked retrieval with tag-based relevance     | LOW (pure numpy over existing kNN indices)     |
| nDCG@K                      | Step 10       | Normalized DCG: graded relevance from multi-field tag overlap         | LOW (pure numpy)                               |
| Silhouette score            | Step 8        | sklearn.metrics.silhouette_score on embeddings + scene_type labels    | LOW (1 function call)                          |
| Multi-attribute slices      | Step 11       | Compute Prec@K grouped by (time_of_day, weather, crowd_density, etc) | MEDIUM (loop over tag fields)                  |
| kNN neighbor grids          | Step 5        | For N random queries, show query + K neighbors as image grid         | MEDIUM (keyframe export needed → depends on #6)|

Also add from #4:
| Confidence threshold sweep  | Step 14       | Vary confidence cutoff, plot Prec@K vs coverage curve                | LOW (filter + recompute)                       |

Also add:
| Macro/micro averaging       | Step 13       | Report both macro (per-class avg) and micro (global) for all metrics | LOW (already have per_scene_purity, extend)    |

Architecture change in m06:
```
Current:  main() → load → build index → search → 2 metrics → save
Modified: main() → load → build index → search → 9 metrics → save
                                                 ├── cycle_at_k (renamed)
                                                 ├── prec_at_k (renamed)
                                                 ├── overlap_at_k (NEW)
                                                 ├── map_at_k (NEW)
                                                 ├── ndcg_at_k (NEW)
                                                 ├── silhouette (NEW)
                                                 ├── multi_attribute_slices (NEW)
                                                 ├── confidence_sweep (NEW, needs #4)
                                                 └── macro_micro_avg (NEW)
```

kNN neighbor grids → move to m07_umap_plot.py (visualization, not metric)

Files touched:
- src/m06_faiss_metrics.py (6 new functions + confidence sweep + macro/micro)
- src/m07_umap_plot.py (kNN neighbor grids visualization)
- src/utils/config.py (add AUGMENTED_EMBEDDINGS_FILE for Overlap@K)

---

### #9  Hard vs Easy Mode (Exclusion Window)

Proposal: exclusion window Δt = ±30s within same video for "hard mode" retrieval.
"Easy mode" = default (no exclusion). Report both.

Action:
- Add `build_video_section_map()` usage in m06 to know which clips come from same video
- Add `--exclusion-window` flag (default 30s) to m06_faiss_metrics.py
- For hard mode: after kNN search, mask out neighbors within ±30s of same video_id
  before computing metrics
- Output both `"hard"` and `"easy"` variants in metrics JSON:
  ```json
  {
    "easy": {"cycle_at_k": 72.1, "prec_at_k": 58.3, ...},
    "hard": {"cycle_at_k": 41.5, "prec_at_k": 35.2, ...}
  }
  ```
- This requires clip metadata (video_id, start_time) to be available in tags.json
  (already stored: video_id is in the 8 metadata fields from m03)

Files touched:
- src/m06_faiss_metrics.py (add exclusion window logic + dual reporting)
- src/utils/config.py (add EXCLUSION_WINDOW_SEC = 30)

---

### #10  Train/Val/Test Splits

Proposal Step 1: "train/val/test splits by video_id" to prevent leakage.

**User concern: "it is a pure evaluation project >> training not yet started"**

Analysis: The user is correct that for pure kNN evaluation (no model training/fine-tuning),
traditional train/val/test splits are NOT needed. kNN retrieval evaluates the ENTIRE
embedding space — there is no training set to overfit on.

HOWEVER, the exclusion window (#9) already handles the leakage concern (clips from same
video being neighbors). This is the correct approach for evaluation-only settings.

Action: SKIP — #9 (exclusion window) handles the leakage concern without needing formal splits.
If/when continual pretraining (Ch 10) begins, splits can be added then.

Files touched: none

---

Modified Project Overview (after all changes)
==============================================

Module changes:
```
src/
├── m02_scene_detect.py        ← ADD --keyframes flag (#6)
├── m04_qwen_tag.py            ← ADD confidence scores (#4), provenance (#5), --audit (#3)
├── m06_faiss_metrics.py       ← ADD 7 metrics (#8), rename 2 (#7), hard/easy mode (#9)
├── m07_umap_plot.py           ← ADD kNN neighbor grids (#8), update labels (#7)
└── utils/
    ├── config.py              ← ADD EXCLUSION_WINDOW_SEC, AUGMENTED_EMBEDDINGS_FILE
    └── tag_taxonomy.json      ← ADD confidence flag per field (#4)
```

No new Python scripts needed. All changes fit into existing modules.

Tag output schema (after #4 + #5):
```
8 metadata + 11 tags + 11 confidence + 3 provenance = 33 fields per clip
```

Metrics output schema (after #7 + #8 + #9):
```json
{
  "easy": {
    "cycle_at_k": 72.1,
    "prec_at_k": 58.3,
    "map_at_k": 0.45,
    "ndcg_at_k": 0.52,
    "silhouette": 0.31,
    "per_scene": {"..."},
    "multi_attribute_slices": {"..."},
    "macro_avg": {"..."},
    "micro_avg": {"..."}
  },
  "hard": {
    "cycle_at_k": 41.5,
    "prec_at_k": 35.2,
    "..."
  },
  "confidence_sweep": [
    {"threshold": 0.5, "coverage": 0.95, "prec_at_k": 56.1},
    {"threshold": 0.7, "coverage": 0.80, "prec_at_k": 62.3},
    "..."
  ],
  "k_neighbors": 6,
  "num_clips": 115687,
  "exclusion_window_sec": 30
}
```

Priority order:
1. #7 (metric renaming) — 30 min, no logic change
2. #8 (mAP, nDCG, silhouette, macro/micro) — 2 hr, pure numpy
3. #4 (confidence scores) — 2 hr, prompt engineering + parse
4. #5 (provenance) — 30 min, trivial
5. #9 (hard/easy mode) — 1 hr, masking logic
6. #8 (multi-attribute slices) — 1 hr, loop over fields
7. #6 (keyframe export) — 1 hr, ffmpeg call
8. #8 (Overlap@K) — 4 hr, needs augmented re-embedding
9. #8 (kNN grids) — 1 hr, needs keyframes from #6
