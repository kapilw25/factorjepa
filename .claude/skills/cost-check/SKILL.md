---
name: cost-check
description: Before recommending any command on GPU, check if it can run on Mac (CPU) instead. GPU time costs $1.0/hr — never waste it on CPU-bound tasks.
disable-model-invocation: true
allowed-tools: Read, Grep
argument-hint: <command or task description>
---

# GPU vs CPU Cost Check

Before recommending ANY task to run on a GPU instance, answer these questions:

## Decision Tree

1. **Does the task use torch.cuda, FAISS-GPU, cuML, or model inference?**
   - YES → GPU required
   - NO → Run on Mac (free)

2. **CPU-bound tasks that MUST run on Mac (never GPU):**
   - `m00d_download_subset.py` — downloads from HF CDN, zero GPU usage
   - `m00c_sample_subset.py` — reads JSON, samples clips, zero GPU usage
   - `m00_data_prep.py`, `m00b_fetch_durations.py` — parse metadata
   - `m02b_scene_fetch_duration.py` — scan clip durations
   - `m03_pack_shards.py` — TAR packing
   - `rsync` / `scp` data transfers
   - `git push` / `git clone`
   - Any `pip install` / `uv pip install`
   - Generating subset JSONs (val_1k.json, subset_10k.json)

3. **GPU-required tasks (must run on GPU instance):**
   - `m04_vlm_tag.py` — VLM inference
   - `m05_vjepa_embed.py` — V-JEPA forward pass
   - `m05b_baselines.py` — DINOv2/CLIP inference
   - `m05c_true_overlap.py` — augmented V-JEPA inference
   - `m04d_motion_features.py` — RAFT optical flow
   - `m06_faiss_metrics.py` — FAISS-GPU kNN
   - `m07_umap.py` — cuML GPU UMAP
   - `m09_pretrain.py` — V-JEPA training
   - `scripts/profile_vram.py` — GPU memory profiling

4. **Can run on either (prefer Mac to save money):**
   - `m08_plot.py` — matplotlib CPU plotting
   - `m08b_compare.py` — comparison plots
   - `m06b_temporal_corr.py` — CPU correlation analysis

## Output

For each recommended command, prepend:
- `[MAC]` — run on Mac (free)
- `[GPU]` — must run on GPU ($0.80/hr)
- `[EITHER]` — prefer Mac unless already on GPU

**NEVER recommend running a [MAC] task on a GPU instance.**
