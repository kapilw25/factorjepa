# WalkIndia-200K Project Memory

## Project Overview
Research benchmark testing if V-JEPA 2 (Meta's video foundation model, trained on Western data) transfers to Indian street scenes. Pipeline: YouTube videos → scene-split clips → WebDataset shards (HF) → VLM tagging → V-JEPA embeddings → FAISS metrics → UMAP → plots.

## Key References
- [codebase.md](codebase.md) — full module-by-module architecture (m00-m08b + utils)
- [debugging.md](debugging.md) — 11 known issues & fixes, architecture gotchas, batch scaling

## Critical Constants (config.py)
- HF_DATASET_REPO = "anonymousML123/walkindia-200k"
- VJEPA: facebook/vjepa2-vitg-fpc64-384, 64 frames, 1408-dim embeddings
- VLMs: qwen (winner, 0.919), videollama, llava — all transformers sequential
- ENCODER_REGISTRY: vjepa, random, dinov2, clip, vjepa_shuffled (suffix-based file paths)
- FAISS_K_NEIGHBORS = 6, BAKEOFF_CLIP_COUNT = 2500
- POC subset: 10K clips (video-level uniform, seed=42)
- Taxonomy: v3 (`tag_taxonomy.json`) — 16 fields (13 single + 2 multi + 1 changelog)

## Output Paths
- POC: src/outputs_poc/ (--subset data/subset_10k.json)
- Full: src/outputs/ (no --subset)
- Bakeoff: src/data/bakeoff/ (--BAKEOFF)

## Current Status (Ch9)
- m00-m08: DONE (code built, SANITY passed for m05/m06/m07)
- m05b: DONE — 4 baselines (random/dinov2/clip/vjepa_shuffled), `--encoder all` runs all 4 sequentially
  - Optimized: DINOv2=FA2+torch.compile, CLIP=SDPA+torch.compile, shuffled=FA2+torch.compile
  - Producer pre-processes tensors on CPU thread (GPU never waits for preprocessing)
  - Batch profile: `compute_batch_sizes()["image_encoder"]` (4x vjepa, cap 256)
- m05c: DONE — augmented V-JEPA embeddings for True Overlap@K
- m08b: DONE — multi-encoder comparison (bar chart, radar, LaTeX table)
- m06/m07: updated with `--encoder` flag (dimension-agnostic)
- run_ch9_overnight.sh: overnight automation (--SANITY/--FULL), pre-flight + per-step verify + final audit
- GPU runs pending for all baselines + True Overlap@K

## VLM Strategy
- 10K POC: Qwen3-VL-8B via transformers (validated, 0.919 score)
- 115K FULL: Qwen3.5-9B via vLLM (transformers video BROKEN — GitHub Issue #58)
- See `iter/iter6/vLLM_plan_Blackwell.md` for deployment plan

## Enforced Rules (Hooks)
- **PreToolUse (Bash):** `.claude/hooks/enforce-dev-rules.sh` — blocks pip install, git state changes, bare python3 without venv
- **PostToolUse (Edit/Write):** `.claude/hooks/post-edit-lint.sh` — auto-runs `py_compile` on any `src/m*.py` edit, fails on syntax error
- **MANDATORY after editing src/m*.py:** (1) py_compile (auto via hook), (2) AST structural check (manual)
- Config: `.claude/settings.json` registers both hooks

## User Preferences
- Never be a yes-man — give pros/cons like a Sr. AI/ML Research Engineer
- Be brutally honest. Disagree when wrong, never hallucinate
- Git: provide commit message text only, never run git commands (enforced by hook)
- GPU time is expensive — keep GPU busy, no idle waste

## Setup Scripts
- `setup_env_uv.sh --mac` (M1 CPU) / `--gpu` (Nvidia) / `--gpu --from-wheels` (prebuilt sm_120 wheels)
- `build_faiss_sm120.sh` — source-build FAISS for Blackwell sm_120
- `build_wheels_sm120.sh` — build FA2+FAISS wheels + upload to GitHub Release
- Wheel flow: Machine 1 builds → uploads to GH Release → Machine 2+ uses `--from-wheels`
