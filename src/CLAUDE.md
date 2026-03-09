1) Modules: src/m00_<name>.py … src/m08_<name>.py — prefix "m" avoids import errors. Numbers must NOT repeat.
2) Utils: @src/utils/
3) GPU Hardware:
- Debug/SANITY: RTX Pro 4000 (24GB VRAM, ~$0.2/hr) — use --SANITY (20 clips) to validate model loading, inference, JSON parsing
- Full/BAKEOFF runs: RTX Pro 6000 Blackwell (96GB VRAM, ~$0.8/hr) — use --BAKEOFF (2500 clips) and --FULL (10K-115K clips)
- M1 Macbook: CPU/API ops + AST/lint only. No GPU fallback
- GPU Software: PyTorch 2.12.0.dev+cu128 nightly, CUDA 12.8, FA2 2.8.3 (prebuilt sm_120 wheel), cuML 26.02, FAISS-GPU 1.14.1 (prebuilt sm_120 wheel, needs patchelf RPATH fix + libopenblas-dev), Python 3.12.12, UV
- GPU scripts must FAIL LOUD — no silent CPU fallback (e.g. FAISS-CPU masking GPU fail, sklearn masking cuML fail)
- "No CPU fallback" applies to inference/compute scripts (m04/m05/m06/m07), NOT visualization/plotting scripts (m08)
4) Docstrings: max 2-line explanation + terminal commands only (--SANITY, --FULL args)
4.1) format: `python -u src/*.py --args arg_name 2>&1 | tee logs/<log_name>.log`
5) Dependencies: update @setup_env_uv.sh, @requirements.txt (CPU), @requirements_gpu.txt (GPU) — install via UV ONLY, no individual pip. **ENFORCED: `.claude/hooks/enforce-dev-rules.sh` blocks `pip install` / `uv pip install` via PreToolUse hook.**
6) Test on M1 `venv_walkindia` : `py_compile` + `--help` + `ast` — full GPU tests on cloud only. **ENFORCED: `.claude/hooks/enforce-dev-rules.sh` blocks bare `python3` without venv activation via PreToolUse hook.** Always: `source venv_walkindia/bin/activate && python3 -m module`
6.1) **MANDATORY after ANY edit to `src/m*.py`:** Run BOTH checks before moving on. No exceptions.
   - `source venv_walkindia/bin/activate && python3 -m py_compile src/<file>.py` — **ENFORCED: `.claude/hooks/post-edit-lint.sh` auto-runs via PostToolUse hook.**
   - AST structural check (verify functions, argparse choices, imports) — **MANUAL: Claude must run explicitly.**
7) Plots: both .png & .pdf. m08_plot.py = CPU-only (pure matplotlib, reads pre-computed .npy files)
7.1) GPU scripts save .npy artifacts (embeddings, knn_indices, umap_2d) → CPU scripts read them. NEVER duplicate GPU compute in CPU scripts (e.g. never rebuild FAISS index in plotting when m06 already saves knn_indices.npy)
7.2) embeddings.paths.npy stores clip keys (not local paths) — used for Hard mode ±30s exclusion. Tags↔embeddings alignment via __key__ field. FAISS uses IVFFlat (not IVF-PQ) — simpler, sufficient at 10K-115K scale
8) Devil's advocate: OOM, GPU underutil, data starvation, VRAM leaks, fp16 instability (use flash-attn-2), checkpoint corruption /auto-resume solution
9) GPU Optimizations:
- torch.compile(model) after model.eval() — warn about first-batch compile latency
- FAISS GPU: faiss.StandardGpuResources() + index_cpu_to_gpu(). Never CPU FAISS in GPU scripts
- cuML GPU: for iterative algorithms (UMAP, DBSCAN, KMeans, PCA) — 50-100x speedup. Install via `--extra-index-url https://pypi.nvidia.com`. For metrics (silhouette, accuracy, F1) keep sklearn/numpy on CPU — post-inference, not a bottleneck
- Auto batch sizing: src/utils/gpu_batch.py — compute_batch_sizes(gpu_vram_gb) auto-detects VRAM, returns `{"vjepa", "image_encoder", "transformers", "transformers_batch"}`. V-JEPA: linear from 40GB baseline. Image encoder: 4x vjepa (cap 256) for DINOv2/CLIP single-frame models. --gpu-mem arg to override. All 3 VLMs use transformers sequential inference
- Attention per encoder: V-JEPA/shuffled/DINOv2 = FA2 (`attn_implementation="flash_attention_2"`), CLIP = SDPA (`attn_implementation="sdpa"` — FA2 support uncertain for CLIPModel)
- Producer pre-processing: processor() runs in CPU producer thread, enqueues ready tensors → GPU thread only does .to(device) + forward pass. Applies to m05 (V-JEPA), m05b (all GPU baselines), m05c (augmented)
- transformers pinned >=4.57.0,<5.0 — all 3 VLMs work. LLaVA-NeXT-Video native (>=4.42). Keye-VL dropped (4 cascading compat errors with both 4.x and 5.x)
- wandb: shared src/utils/wandb_utils.py with: add_wandb_args(parser), init_wandb(module, mode, config, enabled), log_metrics(run, dict, step), log_image(run, key, path), log_artifact(run, name, path), finish_wandb(run). --no-wandb flag on every GPU module, all functions no-op when run=None
10) Each `print` statmenet must be `dynamic`. Remove/modify all false advertising / `static` prints from code. 

# RULES (MUST follow)
- you do not be have to be yes-man on my very demand >> behave like a  Sr. AI/ML Research engineer >> give me pros and cons of each of my demand
- Be brutally honest. Disagree when I'm wrong, but never hallucinate or lie.
- Devil's advocate does NOT mean fabricating bugs that don't exist. If code is correct, say so and move on.
- WEBSEARCH when needed to confirm universal AI/ML research practices.
- Git: provide commit message text only. NEVER run git commands. User handles all git ops via git_push.sh. **ENFORCED: `.claude/hooks/enforce-dev-rules.sh` blocks `git commit/push/add/reset` via PreToolUse hook.**
- NEVER draw conclusions from statistically insufficient data. A sanity check validates code correctness (no crashes, valid output), NOT model performance. Do not propose architectural changes, formula rewrites, or bottleneck diagnoses based on < 100 data points.
- Think like a Sr. Research Scientist: before making any recommendation, ask "do I have enough evidence for this claim?" If the answer is no, say so explicitly instead of speculating. Superficial pattern-matching on small samples is Jr-level analysis — dig into architecture, read the actual code paths, and ground every recommendation in either sufficient data or first-principles reasoning.
- GPU time is expensive — idle GPU = wasted money; idle user during GPU job = wasted time. Keep the GPU busy.
- Mandatory checklist for ANY new loop that makes GPU/API/VLM calls: (1) tqdm progress bar, (2) auto-resume from checkpoint (skip completed items on restart), (3) tee logging, (4) wandb integration. No exceptions. If you write a pipeline loop without all four, you have shipped broken code. **MANUAL: Run `/preflight <file>` skill to verify before running any new pipeline code.** See `.claude/skills/preflight/SKILL.md`.

# REFERENCE
- Bug history & batch speedup details: `iter/iter6/plan_batch_speedup.md`