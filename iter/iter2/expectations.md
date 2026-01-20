# INPUT/OUTPUT Specification for src/m0*.py scripts

## Quick Start on A100-80GB
```bash
# 1. Setup environment
./setup_env.sh --gpu

# 2. Sanity test each module (FIRST TIME ONLY - see below)

# 3. Run full pipeline (after sanity tests pass)
python -u src/m04_pipeline_orchestrator.py --full --input_dir Literature/Prev_work2/dataset/3DReasoningProject_images 2>&1 | tee logs/m04_full.log

# 4. Compare with baseline
python -u src/m05_comparison.py --compare 2>&1 | tee logs/m05_compare.log
```

---

## First-Time Sanity Testing (Sequential)

Run each module with `--sanity` before full pipeline. Debug failures at each step.

```bash
# Step 1: Test VLM object detection (Qwen2.5-VL-32B)
python -u src/m01_scene_understanding.py --sanity --input_dir Literature/Prev_work2/dataset/3DReasoningProject_images 2>&1 | tee logs/m01_sanity.log
# Expected: outputs/m01_scenes/<scene_id>.json with objects list

# Step 2: Test LLM instruction generation (Llama-3.1-70B)
python -u src/m02_instruction_generator.py --sanity 2>&1 | tee logs/m02_sanity.log
# Expected: outputs/m02_instructions/<scene_id>.json with level_1/2/3 instructions

# Step 3: Test GPT-4o evaluation (already tested for baseline - SKIP if needed)
python -u src/m03_evaluator.py --sanity 2>&1 | tee logs/m03_sanity.log
# Expected: outputs/m03_evaluations/<scene_id>.json + metrics.csv

# Step 4: Test full orchestrator (runs m01→m02→m03 sequentially)
python -u src/m04_pipeline_orchestrator.py --sanity --input_dir Literature/Prev_work2/dataset/3DReasoningProject_images 2>&1 | tee logs/m04_sanity.log
# Expected: All outputs from steps 1-3

# Step 5: Test comparison (requires both baseline + proposed metrics.csv)
python -u src/m05_comparison.py --compare 2>&1 | tee logs/m05_sanity.log
# Expected: outputs_comparison/*.pdf plots
```

### Sanity Test Checklist
| Module | GPU | Dependency | Success Criteria |
|--------|-----|------------|------------------|
| m01 | ~40GB | None | `outputs/m01_scenes/*.json` exists with `objects` list |
| m02 | ~70GB | m01 output | `outputs/m02_instructions/*.json` with L1/L2/L3 |
| m03 | API only | m01+m02 output | `outputs/metrics.csv` + 5 plots |
| m04 | Sequential | Images + .env | All m01/m02/m03 outputs generated |
| m05 | CPU only | Two metrics.csv | `outputs_comparison/*.pdf` plots |

---

## m01_scene_understanding.py
**Purpose**: Object detection from scene images using Qwen2.5-VL-32B (8-bit) — ~40GB VRAM

### INPUT
```
<input_dir>/
└── <scene_id>/
    ├── grid_panorama.png          # Isometric view (1024x1024 resize)
    └── grid_panorama_topdown.png  # Top-down view (1024x1024 resize)
```

### OUTPUT
```
outputs/m01_scenes/<scene_id>.json
{
  "scene_id": "35_Bus_loop.3",
  "isometric_path": "/abs/path/grid_panorama.png",
  "topdown_path": "/abs/path/grid_panorama_topdown.png",
  "objects": [
    {"name": "bench", "confidence": 0.95},
    {"name": "tree", "confidence": 0.92}
  ],
  "scene_type": "outdoor_mall",
  "layout_description": "Open plaza with benches..."
}
```

### Commands
```bash
python -u src/m01_scene_understanding.py --sanity 2>&1 | tee logs/m01_sanity.log
python -u src/m01_scene_understanding.py --full --input_dir data/images 2>&1 | tee logs/m01_full.log
python -u src/m01_scene_understanding.py --metadata_only --input_dir data/images 2>&1 | tee logs/m01_metadata.log
```

---

## m02_instruction_generator.py
**Purpose**: Robot navigation instruction generation using Llama-3.1-70B-Instruct (8-bit) — ~70GB VRAM

### INPUT
```
outputs/m01_scenes/<scene_id>.json  # Must contain: objects, scene_type, layout_description
```

### OUTPUT
```
outputs/m02_instructions/<scene_id>.json
{
  "level_1": [
    {"task": "Navigate from bench to fountain", "start": "Near bench", "end": "Near fountain", "interact": []}
  ],
  "level_2": [
    {"task": "Pick up package and deliver", "start": "Near table", "end": "Near chair", "interact": ["package"]}
  ],
  "level_3": [
    {"task": "Collect items, check display, deliver", "start": "Near shelf", "end": "Near counter", "interact": ["shelf", "display", "items"]}
  ]
}
```

### Commands
```bash
python -u src/m02_instruction_generator.py --sanity 2>&1 | tee logs/m02_sanity.log
python -u src/m02_instruction_generator.py --full 2>&1 | tee logs/m02_full.log
```

---

## m03_evaluator.py
**Purpose**: VLM-as-Judge evaluation using GPT-4o API (CPU + API, no GPU)

### INPUT
```
outputs/
├── m01_scenes/<scene_id>.json       # Image paths
└── m02_instructions/<scene_id>.json # Instructions to evaluate
```
+ `.env` file with `OPENAI_API_KEY=sk-...`

### OUTPUT
```
outputs/
├── m03_evaluations/<scene_id>.json  # Detailed scores per instruction
├── metrics.csv                      # Aggregated metrics (all scenes)
└── m03_plots/                       # 5 research-grade plots
    ├── bar_avg_scores_by_level.png
    ├── radar_metric_comparison.png
    ├── heatmap_scene_metrics.png
    ├── violin_score_distribution.png
    └── summary_dashboard.png
```

### metrics.csv columns
```
scene_id, L1_object_accuracy, L1_spatial_coherence, L1_task_clarity, L1_difficulty_alignment, L1_executability,
          L2_object_accuracy, L2_spatial_coherence, L2_task_clarity, L2_difficulty_alignment, L2_executability,
          L3_object_accuracy, L3_spatial_coherence, L3_task_clarity, L3_difficulty_alignment, L3_executability
```

### Commands
```bash
python -u src/m03_evaluator.py --sanity 2>&1 | tee logs/m03_sanity.log
python -u src/m03_evaluator.py --full 2>&1 | tee logs/m03_full.log
python -u src/m03_evaluator.py --plot_only 2>&1 | tee logs/m03_plot.log
python -u src/m03_evaluator.py --metadata_only 2>&1 | tee logs/m03_metadata.log
```

---

## m04_pipeline_orchestrator.py
**Purpose**: Full pipeline with sequential model loading for A100-80GB

### Memory Strategy
```
Step 1: Load Qwen2.5-VL-32B (~40GB) → process all scenes → unload
Step 2: Load Llama-3.1-70B (~70GB) → process all scenes → unload
Step 3: GPT-4o API (no GPU)
```

### INPUT
```
<input_dir>/
└── <scene_id>/
    ├── grid_panorama.png
    └── grid_panorama_topdown.png
```
+ `.env` with `HF_TOKEN=hf_...` and `OPENAI_API_KEY=sk-...`

### OUTPUT
```
outputs/
├── m01_scenes/          # From Step 1
├── m02_instructions/    # From Step 2
├── m03_evaluations/     # From Step 3
├── metrics.csv          # Aggregated
└── m03_plots/           # 5 plots
```

### Commands
```bash
# Sanity (1 sample, all steps)
python -u src/m04_pipeline_orchestrator.py --sanity 2>&1 | tee logs/m04_sanity.log

# Full run without evaluation
python -u src/m04_pipeline_orchestrator.py --full --skip_eval 2>&1 | tee logs/m04_full_noeval.log

# Full run with evaluation
python -u src/m04_pipeline_orchestrator.py --full 2>&1 | tee logs/m04_full.log

# Evaluation only (requires existing m01/m02 outputs)
python -u src/m04_pipeline_orchestrator.py --eval_only 2>&1 | tee logs/m04_eval.log
```

---

## m05_comparison.py
**Purpose**: Baseline vs Proposed comparison plots with statistical significance testing (CPU only)

### INPUT
```
outputs_baseline/metrics.csv   # Baseline pipeline results
outputs/metrics.csv            # Proposed pipeline results (same scenes)
```

### OUTPUT
```
outputs_comparison/
├── comparison_by_metric.pdf   # Side-by-side bar chart
├── comparison_by_level.pdf    # Performance by difficulty
├── improvement_heatmap.pdf    # % improvement per scene/metric
└── comparison_dashboard.pdf   # Summary with significance tests
```

### Commands
```bash
python -u src/m05_comparison.py --compare 2>&1 | tee logs/m05_compare.log
python -u src/m05_comparison.py --compare --baseline outputs_v1 --proposed outputs_v2
```

---

## File Structure
```
src/
├── m01_scene_understanding.py    # Qwen2.5-VL-32B (GPU ~40GB)
├── m02_instruction_generator.py  # Llama-3.1-70B (GPU ~70GB)
├── m03_evaluator.py              # GPT-4o API (CPU + API)
├── m04_pipeline_orchestrator.py  # Sequential orchestration (GPU)
├── m05_comparison.py             # Statistical comparison (CPU)
└── utils/
    ├── __init__.py
    ├── io_handler.py             # JSON/CSV file operations
    ├── image_utils.py            # resize, encode, pair-finding
    └── plotting.py               # Research-grade matplotlib/seaborn

outputs_baseline/                 # Baseline results (already generated)
outputs/                          # Proposed (Qwen-32B + Llama-70B) results
outputs_comparison/               # Comparison plots
logs/                             # All execution logs
```

---

## Environment Variables (.env)
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

## Execution Order on A100-80GB
1. `./setup_env.sh --gpu`
2. `m04_pipeline_orchestrator.py --full` → runs m01→m02→m03 sequentially
3. `m05_comparison.py --compare` → generates comparison with baseline
