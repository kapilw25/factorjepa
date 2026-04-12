# Code Development Plan — Week 1 Experiments

> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**
> **Short-term target: Show Surgery > ExPLoRA > Frozen on 1K val clips (~1h total).**

---

## Completed (previous session)

| Task | File | What |
|---|---|---|
| Config restructure | `configs/model/` + `configs/train/` | 6 YAML files |
| Config merging | `src/utils/config.py` | `load_merged_config()`, `get_model_config()` |
| Zero hardcoding | `src/utils/config.py` + `configs/pipeline.yaml` | All model IDs/dims from YAML |
| V-JEPA 2.1 import | `src/utils/vjepa2_imports.py` | `get_vit_gigantic_xformers()` + `get_vit_by_arch()` |
| m05 V-JEPA 2.1 | `src/m05_vjepa_embed.py` | `--model-config`, native 2.1 loading (adapted/frozen/HF) |
| m09 ExPLoRA | `src/m09_pretrain.py` | `--model-config` + `--train-config` + `--explora` (LoRA + block freeze) |
| Temporal projection | `src/m06c_temporal_projection.py` | PCA sweep, verify_or_skip, resume, vectorized |
| Checkpoint utils | `src/utils/checkpoint.py` | Atomic save/load (embedding, array, JSON) |
| Shell scripts | `scripts/train_explora.sh` + `scripts/train_surgery.sh` | Full orchestration pipelines |
| CLAUDE.md rules | `src/CLAUDE.md` | Rules 29-31 (GPU infra, --model-config, get_vit_by_arch) |
| 3-check gate | `.claude/hooks/post-edit-lint.sh` | py_compile + AST + ruff F+E9 |

---

## Fastest Path: 1K Val Clips (~1h total GPU)

`data/val_1k_local/` has 1000 clips ready. Instead of 10K POC (7h), run on 1K for fast signal:

| Step | What | Time |
|---|---|---|
| Frozen 2.1 embed | `m05 --encoder vjepa_2_1_frozen` on 1K clips | ~12 min |
| ExPLoRA train | `m09 --explora` on 1K clips, 5 epochs | ~20 min |
| SAM 3.1 segment | `m10` on 1K × 16 frames | ~5 min |
| Factor datasets | `m11` D_L + D_A from masks | ~3 min |
| Surgery train | `m09 --train-config ch11_surgery.yaml` on 1K, 2 stages | ~15 min |
| Re-embed + eval | m05 + m06 for each adapted model | ~15 min |
| **Total** | | **~70 min** |

If Surgery > ExPLoRA on 1K → scale to 10K for paper.
If Surgery = ExPLoRA on 1K → debug factor order / add clips before investing more.

---

## SAM 3.1 + Tag Taxonomy Integration

`m04_vlm_tag.py` reads `configs/tag_taxonomy.json` (schema) → generates `outputs/poc/tags.json` (per-clip tags).

SAM 3.1 text prompt uses per-clip `notable_objects` from `tags.json` for precise agent detection.

`outputs/full/tags.json` (115K clips, all tagged) is available on HF and downloaded by `src/utils/hf_outputs.py` via `git_pull.sh`. This is NOT optional — per-clip prompts are mandatory.

**How it works in m10:**

```python
# Load tags.json (already exists for all clips)
tags = json.load(open(output_dir / "tags.json"))
tags_by_key = {t["__key__"]: t for t in tags}

# Per-clip: read notable_objects → build SAM 3.1 text prompt
clip_tags = tags_by_key[clip_key]
objects = clip_tags["notable_objects"]  # e.g., ["auto_rickshaw", "pedestrian", "sacred_cow"]
agent_prompt = ", ".join(objects)        # → "auto_rickshaw, pedestrian, sacred_cow"
```

**Why mandatory (not generic prompt):**
- Generic prompt "person, vehicle, animal" misses India-specific objects (auto_rickshaw, handcart, sacred_cow)
- Per-clip prompt is MORE PRECISE — only segments objects actually present (from VLM tagger)
- `tags.json` already computed (0 extra cost) — no reason to use a worse generic prompt
- Fallback for clips without tags: use full `notable_objects` list from `configs/tag_taxonomy.json`

**Tags data (DONE — downloaded and filtered this session):**

| File | Clips | Status |
|---|---|---|
| `outputs/full/tags.json` | 115,687 | Downloaded from HF |
| `data/val_1k_local/tags.json` | 1,000 | Filtered from full (100% match) |
| `data/subset_10k_local/tags.json` | 10,000 | Filtered from full (100% match) |

Key matching: `tag["section"]/tag["video_id"]/tag["source_file"]` → manifest `saved_keys` path.

m10 loads per-clip `notable_objects` → builds SAM 3.1 text prompt per clip. Example:
- Clip `goa/walking/.../market` → `notable_objects: [bus, auto_rickshaw, pedestrian, ...]`
- SAM prompt: `"bus, auto_rickshaw, pedestrian, bike, car, truck, handcart, street_vendor, signage"`

---

## Why SAM 3.1 (not SAM 2)

| Feature | SAM 2 | SAM 3.1 |
|---|---|---|
| Object detection | Click/box prompts per object | **Text prompts**: "person", "vehicle" |
| Multi-object tracking | Sequential per object (linear cost) | **Multiplexing**: all objects together |
| Dense scenes | Struggles with crowds | **Designed for crowded, fast-moving** |
| Agent identification | Custom motion-based filter (~150 lines) | **Text prompt directly** (~30 lines) |
| API | `sam2.build_sam2_video_predictor` | `sam3.model_builder.build_sam3_video_predictor` |
| HuggingFace | `facebook/sam2.1-hiera-large` | `facebook/sam3.1` (gated, need `hf auth login`) |
| Paper | arXiv:2408.00714 | arXiv:2511.16719 + SAM 3.1 blog (Mar 2026) |

Refs: [SAM 3.1 Blog](https://ai.meta.com/blog/segment-anything-model-3/), [GitHub](https://github.com/facebookresearch/sam3), [HuggingFace](https://huggingface.co/facebook/sam3.1)

---

## TODO List (10 tasks, ~140 min)

| Phase | # | Task | File | Est. |
|---|---|---|---|---|
| **A: Refactor** | 1 | Create `src/utils/video_io.py` — move `get_clip_key`, `decode_video_bytes`, `_create_stream` from m05 | `utils/video_io.py` (NEW) | 15 min |
| | 2 | Update 5 importers → `from utils.video_io import ...` | m05, m05b, m05c, m09, profile_vram | 10 min |
| | 3 | Add CLAUDE.md rule 32: no cross-imports between m*.py | `src/CLAUDE.md` | 2 min |
| **B: Config** | 4 | Move `tag_taxonomy.json` + `YT_videos_raw.json` to `configs/` | `configs/`, config.py, 8 importers | 10 min |
| | 5 | Update `ch11_surgery.yaml`: `sam_model → facebook/sam3.1` + `agent_prompt` from tag_taxonomy | `configs/train/ch11_surgery.yaml` | 2 min |
| | 6 | Add `sam3` to `requirements_gpu.txt` | `requirements_gpu.txt` | 2 min |
| **C: Build** | 7 | Build `src/m10_sam_segment.py` — SAM 3.1 text-prompted segmentation | NEW (~300 lines) | 45 min |
| | 8 | Build `src/m11_factor_datasets.py` — generate D_L + D_A from masks | NEW (~200 lines) | 30 min |
| **D: Verify** | 9 | 3-check gate on ALL modified files (py_compile + ruff + AST) | ~10 files | 10 min |
| **E: Plan** | 10 | Update `next_steps.md` + `runbook.md`: 1K val clips as fast signal | 2 files | 10 min |

### JSON files to move to `configs/`

| File | Currently at | Move to | Used by |
|---|---|---|---|
| `tag_taxonomy.json` | `src/utils/tag_taxonomy.json` | `configs/tag_taxonomy.json` | m04, m04b, m04c, m04_vllm, m06, m08 |
| `YT_videos_raw.json` | `src/utils/YT_videos_raw.json` | `configs/YT_videos_raw.json` | m00, m01, config.py |

### Prerequisite refactor: `src/utils/video_io.py`

All shared functions must live in `src/utils/`. Currently these are in m05 and cross-imported by 5 files:

| Function | Currently in | Move to | Used by |
|---|---|---|---|
| `get_clip_key(example)` | m05:79 | `utils/video_io.py` | m05, m05b, m05c, m09, m10, m11 |
| `_create_stream(skip, local_data)` | m05:90 | `utils/video_io.py` | m05, m05b, m05c, m09 |
| `decode_video_bytes(mp4, tmp, key, n)` | m05:152 | `utils/video_io.py` | m05, m05b, m05c, m09, m10, m11, profile_vram |
| `save_checkpoint` / `load_checkpoint` | m05:177 | `utils/checkpoint.py` | **Already done** |

---

## m10: `src/m10_sam_segment.py` — Detailed Design (~300 lines)

```
USAGE:
    python -u src/m10_sam_segment.py --SANITY --local-data data/val_1k_local 2>&1 | tee logs/m10_sanity.log
    python -u src/m10_sam_segment.py --POC --local-data data/val_1k_local 2>&1 | tee logs/m10_poc.log
    python -u src/m10_sam_segment.py --FULL --local-data data/full_local 2>&1 | tee logs/m10_full.log
```

### CLI Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--SANITY` | bool | false | 20 clips |
| `--POC` | bool | false | 1K clips (val_1k_local for fast signal) |
| `--FULL` | bool | false | All clips |
| `--subset` | str | None | Subset JSON |
| `--local-data` | str | None | Local WebDataset dir |
| `--train-config` | str | `configs/train/ch11_surgery.yaml` | Factor dataset params |
| `--output-dir` | str | None | Override (used by train_surgery.sh) |
| `--agent-prompt` | str | from YAML | Override text prompt |
| `--no-wandb` | bool | false | Disable wandb |

### Mode-specific behavior

| Mode | Clip limit | Output dir | Notes |
|---|---|---|---|
| SANITY | 20 | `outputs/sanity/factors/` | Validate SAM loads + masks save |
| POC | 1000 | `outputs/poc/factors/` | Fast signal on val_1k_local |
| FULL | 10K+ | `outputs/{poc\|full}/factors/` | Scale-up for paper |

### Output

```
outputs/poc/factors/
├── masks/
│   └── {clip_key_safe}.npz     # agent_mask (T,H,W) bool + layout_mask (T,H,W) bool
├── segments.json               # {clip_key: {n_agents, n_frames, agent_pixel_ratio}}
├── summary.json                # {n_clips, n_total_agents, elapsed_sec}
└── .m10_checkpoint.json        # Resume: {processed_keys: [...]}
```

### Infrastructure — ALL imports from `src/utils/` (Rule 29 + Rule 32)

| Function | Import from | Purpose in m10 |
|---|---|---|
| `check_gpu()` | `utils.config` | FATAL if no CUDA |
| `cleanup_temp()` | `utils.gpu_batch` | Clean /tmp at start |
| `verify_or_skip()` | `utils.output_guard` | Skip if `segments.json` exists |
| `save_json_checkpoint()` | `utils.checkpoint` | Resume every 10 clips |
| `load_json_checkpoint()` | `utils.checkpoint` | Load resume state |
| `ensure_local_data()` | `utils.data_download` | Validate local TARs |
| `iter_clips_parallel()` | `utils.data_download` | Parallel TAR reading |
| `get_clip_key()` | `utils.video_io` (NEW) | Construct clip key from HF example |
| `decode_video_bytes()` | `utils.video_io` (NEW) | Decode MP4 → tensor frames |
| `make_pbar()` | `utils.progress` | Progress bar |
| `init_wandb()` / `log_metrics()` / `finish_wandb()` | `utils.wandb_utils` | Experiment tracking |
| `get_output_dir()` | `utils.config` | Mode-aware output routing |
| `add_subset_arg()` / `add_local_data_arg()` | `utils.config` | Argparse helpers |
| `add_wandb_args()` | `utils.wandb_utils` | `--no-wandb` flag |
| `load_subset()` | `utils.config` | Load subset JSON |
| `get_sanity_clip_limit()` | `utils.config` | SANITY clip limit |
| `get_total_clips()` | `utils.config` | Clip count from manifest |
| `binary_dilation()` | `scipy.ndimage` | Dilate agent masks |
| **SAM 3.1** | `sam3.model_builder` | `build_sam3_video_predictor()` |
| **Config params** | `configs/train/ch11_surgery.yaml` | `factor_datasets.*` (all numeric params) |
| **Agent categories** | `configs/tag_taxonomy.json` | `notable_objects` → text prompt |

### Key Functions

```python
def load_sam3(device: str = "cuda"):
    """Load SAM 3.1 video predictor from facebook/sam3.1."""

def save_clip_as_frames(frames: np.ndarray, tmp_dir: str) -> str:
    """Save frames as JPEG folder for SAM 3.1 input. Returns frame_dir path."""

def segment_clip(predictor, frame_dir: str, agent_prompt: str, dilation_px: int) -> dict:
    """SAM 3.1 text prompt → agent/layout masks per frame.
    Returns {agent_mask: (T,H,W) bool, layout_mask: (T,H,W) bool, n_agents: int, agent_pixel_ratio: float}."""

def save_clip_masks(clip_key: str, result: dict, masks_dir: Path):
    """Save compressed .npz (bool arrays → ~10 KB/clip)."""

def main():
    """Parse args → load SAM 3.1 → iterate clips → segment → save."""
```

---

## m11: `src/m11_factor_datasets.py` — Detailed Design (~200 lines)

```
USAGE:
    python -u src/m11_factor_datasets.py --SANITY --local-data data/val_1k_local 2>&1 | tee logs/m11_sanity.log
    python -u src/m11_factor_datasets.py --POC --local-data data/val_1k_local 2>&1 | tee logs/m11_poc.log
```

**CPU-only script.** No GPU needed.

### CLI Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--SANITY` | bool | false | 20 clips |
| `--POC` | bool | false | 1K clips |
| `--FULL` | bool | false | All clips |
| `--subset` | str | None | Subset JSON |
| `--local-data` | str | None | Local WebDataset dir (load original frames) |
| `--input-dir` | str | None | m10 output dir (default: `{output_dir}/factors/`) |
| `--output-dir` | str | None | Override |
| `--train-config` | str | `configs/train/ch11_surgery.yaml` | Patching params |
| `--no-wandb` | bool | false | Disable wandb |

### Output

```
outputs/poc/factors/
├── D_L/
│   └── {clip_key_safe}.npy    # (T, H, W, C) uint8 — layout-only (agents blurred)
├── D_A/
│   └── {clip_key_safe}.npy    # (T, H, W, C) uint8 — agent-only (background suppressed)
└── factor_manifest.json       # {clip_key: {has_D_L, has_D_A, agent_pct}}
```

### Infrastructure — ALL imports from `src/utils/` (CPU-only subset of Rule 29)

| Function | Import from | Purpose in m11 |
|---|---|---|
| `verify_or_skip()` | `utils.output_guard` | Skip if `factor_manifest.json` exists |
| `save_json_checkpoint()` | `utils.checkpoint` | Save manifest atomically |
| `load_json_checkpoint()` | `utils.checkpoint` | Load m10's `segments.json` |
| `ensure_local_data()` | `utils.data_download` | Validate local TARs |
| `iter_clips_parallel()` | `utils.data_download` | Read original clips |
| `decode_video_bytes()` | `utils.video_io` (NEW) | Decode MP4 → frames |
| `get_clip_key()` | `utils.video_io` (NEW) | Match clip keys |
| `make_pbar()` | `utils.progress` | Progress bar |
| `init_wandb()` / `log_metrics()` / `finish_wandb()` | `utils.wandb_utils` | Tracking |
| `get_output_dir()` | `utils.config` | Output routing |
| `add_subset_arg()` / `add_local_data_arg()` | `utils.config` | Argparse helpers |
| `add_wandb_args()` | `utils.wandb_utils` | `--no-wandb` |
| `load_subset()` | `utils.config` | Subset JSON |
| `get_sanity_clip_limit()` | `utils.config` | SANITY limit |
| `gaussian_filter()` | `scipy.ndimage` | Blur agents for D_L |
| **NOT used** (CPU-only) | `check_gpu`, `cleanup_temp`, `AdaptiveBatchSizer`, `cuda_cleanup` | — |
| **Config params** | `configs/train/ch11_surgery.yaml` | `layout_patch_method`, `agent_patch_method`, `soft_matte_factor` |

### Key Functions

```python
def make_layout_only(frames: np.ndarray, agent_mask: np.ndarray,
                     method: str = "blur", blur_sigma: float = 15.0) -> np.ndarray:
    """D_L: blur agents, preserve layout. Returns (T, H, W, C) uint8."""

def make_agent_only(frames: np.ndarray, layout_mask: np.ndarray,
                    method: str = "soft_matte", matte_factor: float = 0.1) -> np.ndarray:
    """D_A: suppress background, preserve agents. Returns (T, H, W, C) uint8."""

def main():
    """Parse args → load m10 masks → load original frames → patch → save D_L/D_A."""
```

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| SAM 3.1 gated access not approved | m10 blocked | Apply at `huggingface.co/facebook/sam3.1`, typically <1 day |
| SAM 3.1 text prompt misclassifies | Wrong agent/layout | Tune prompt, inspect on SANITY first |
| SAM 3.1 VRAM usage | OOM | SAM 3.1 designed for edge (~4GB). Process 1 clip at a time |
| sam3 package install conflicts | Broken env | Install from source, isolated venv test |
| 1K val clips insufficient signal | Inconclusive | Scale to 10K if 1K shows any trend |

---

## Verification

### Mac (syntax only)

```bash
source venv_walkindia/bin/activate
for f in src/utils/video_io.py src/m10_sam_segment.py src/m11_factor_datasets.py; do
    python3 -m py_compile "$f" && python3 -m ruff check --select F,E9 "$f" && echo "OK: $f"
done
```

### GPU (functional)

```bash
# 1. Install SAM 3.1
pip install git+https://github.com/facebookresearch/sam3.git
huggingface-cli login

# 2. SANITY on 20 clips
python -u src/m10_sam_segment.py --SANITY --local-data data/val_1k_local --no-wandb
python -u src/m11_factor_datasets.py --SANITY --local-data data/val_1k_local --no-wandb
ls outputs/sanity/factors/masks/ outputs/sanity/factors/D_L/ outputs/sanity/factors/D_A/

# 3. POC on 1K clips (fast signal)
./scripts/train_surgery.sh --POC
```
