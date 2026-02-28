1) Modules: src/m00_<name>.py … src/m08_<name>.py — prefix "m" avoids import errors. Numbers must NOT repeat.
2) Utils: @src/utils/
3) GPU Hardware:
- Debug: RTX Pro 4000 (24GB VRAM, ~$0.2/hr)
- Full runs: RTX Pro 6000 Blackwell (96GB VRAM, ~$0.8/hr)
- M1 Macbook: CPU/API ops + AST/lint only. No GPU fallback
- GPU scripts must FAIL LOUD — no silent CPU fallback (e.g. FAISS-CPU masking GPU fail, sklearn masking cuML fail)
- "No CPU fallback" applies to inference/compute scripts (m04/m05/m06/m07), NOT visualization/plotting scripts (m08)
4) Docstrings: max 2-line explanation + terminal commands only (--SANITY, --FULL args)
4.1) format: `python -u src/*.py --args arg_name 2>&1 | tee logs/<log_name>.log`
5) Dependencies: update @setup_env.sh, @requirements.txt (CPU), @requirements_gpu.txt (GPU) — install via venv ONLY, no individual pip
6) Test on M1 `venv_walkindia` : `py_compile` + `--help` + `ast` — full GPU tests on cloud only
7) Plots: both .png & .pdf. m08_plot.py = CPU-only (pure matplotlib, reads pre-computed .npy files)
7.1) GPU scripts save .npy artifacts (embeddings, knn_indices, umap_2d) → CPU scripts read them. NEVER duplicate GPU compute in CPU scripts (e.g. never rebuild FAISS index in plotting when m06 already saves knn_indices.npy)
8) Devil's advocate: OOM, GPU underutil, data starvation, VRAM leaks, fp16 instability (use flash-attn-2), checkpoint corruption /auto-resume solution
9) GPU Optimizations:
- torch.compile(model) after model.eval() — warn about first-batch compile latency
- FAISS GPU: faiss.StandardGpuResources() + index_cpu_to_gpu(). Never CPU FAISS in GPU scripts
- cuML GPU: for iterative algorithms (UMAP, DBSCAN, KMeans, PCA) — 50-100x speedup. Install via `--extra-index-url https://pypi.nvidia.com`. For metrics (silhouette, accuracy, F1) keep sklearn/numpy on CPU — post-inference, not a bottleneck
- Auto batch sizing: src/utils/gpu_batch.py — compute_batch_sizes(gpu_vram_gb) auto-detects VRAM, scales linearly from A100-40GB baseline. --gpu-mem arg to override. vLLM excluded (continuous batching)
- wandb: shared src/utils/wandb_utils.py with: add_wandb_args(parser), init_wandb(module, mode, config, enabled), log_metrics(run, dict, step), log_image(run, key, path), log_artifact(run, name, path), finish_wandb(run). --no-wandb flag on every GPU module, all functions no-op when run=None
10) Each `print` statmenet must be `dynamic`. Remove/modify all false advertising / `static` prints from code. 

# RULES (MUST follow)
- you do not be have to be yes-man on my very demand >> behave like a  Sr. AI/ML Research engineer >> give me pros and cons of each of my demand
- Be brutally honest. Disagree when I'm wrong, but never hallucinate or lie.
- Devil's advocate does NOT mean fabricating bugs that don't exist. If code is correct, say so and move on.
- WEBSEARCH when needed to confirm universal AI/ML research practices.
- Git: provide commit message text only. NEVER run git commands. User handles all git ops via git_push.sh.