1) Modules: src/m00_<name>.py … src/m09_<name>.py — prefix "m" avoids import errors. Numbers must NOT repeat.
2) Utils: @src/utils/
3) GPU Hardware:
- Debug/SANITY: RTX Pro 4000 (24GB VRAM, ~$0.2/hr) — use --SANITY to validate model loading, inference, JSON parsing
- Full/BAKEOFF runs: RTX Pro 6000 Blackwell (96GB VRAM, ~$0.8/hr) — use --BAKEOFF (2500 clips) and --FULL (10K-115K clips)
- M1 Macbook: CPU/API ops + AST/lint only. No GPU fallback
- GPU Software: PyTorch 2.12.0.dev+cu128 nightly, CUDA 12.8, FA2 2.8.3 (prebuilt sm_120 wheel), cuML 26.02, FAISS-GPU 1.14.1 (prebuilt sm_120 wheel, needs patchelf RPATH fix + libopenblas-dev), Python 3.12.12, UV
- GPU scripts must FAIL LOUD — no silent CPU fallback (e.g. FAISS-CPU masking GPU fail, sklearn masking cuML fail)
- "No CPU fallback" applies to inference/compute scripts (m04/m05/m06/m07/m09), NOT visualization/plotting scripts (m08)
4) Docstrings: max 2-line explanation + terminal commands only (--SANITY, --FULL args)
4.1) format: `python -u src/*.py --args arg_name 2>&1 | tee logs/<log_name>.log`
5) Dependencies: update @setup_env_uv.sh, @requirements.txt (CPU), @requirements_gpu.txt (GPU) — install via UV ONLY, no individual pip. **ENFORCED: `.claude/hooks/enforce-dev-rules.sh` blocks `pip install` / `uv pip install` via PreToolUse hook.**
6) Test on M1 `venv_walkindia` : `py_compile` + `--help` + `ast` — full GPU tests on cloud only. **ENFORCED: `.claude/hooks/enforce-dev-rules.sh` blocks bare `python3` without venv activation via PreToolUse hook.** Always: `source venv_walkindia/bin/activate && python3 -m module`
6.1) **MANDATORY after ANY edit to `src/m*.py`:** Run BOTH checks before moving on. No exceptions.
   - `source venv_walkindia/bin/activate && python3 -m py_compile src/<file>.py` — **ENFORCED: `.claude/hooks/post-edit-lint.sh` auto-runs via PostToolUse hook.**
   - AST structural check (verify functions, argparse choices, imports) — **ENFORCED: `.claude/hooks/post-edit-lint.sh`.**
   - `ruff check --select F821,F841,F811` (undefined names, unused vars, redefined vars) — **ENFORCED: `.claude/hooks/post-edit-lint.sh`.** Catches "used before assignment" bugs that py_compile misses.
7) Plots: both .png & .pdf. m08_plot.py = CPU-only (pure matplotlib, reads pre-computed .npy files)
7.1) GPU scripts save .npy artifacts (embeddings, knn_indices, umap_2d) → CPU scripts read them. NEVER duplicate GPU compute in CPU scripts (e.g. never rebuild FAISS index in plotting when m06 already saves knn_indices.npy)
7.4) **95% CI MANDATORY**: Every metric reported in JSON or displayed in plots/tables MUST include bootstrap 95% CI (BCa, 10K iterations via `scipy.stats.bootstrap`). Use `utils/bootstrap.py`: compute per-clip scores → `bootstrap_ci(scores)` → store `{"mean", "ci_lo", "ci_hi", "ci_half"}` under `"ci"` key in JSON. Plots: error bars via `yerr=ci_half`. LaTeX tables: `50.5{\tiny$\pm$2.1}` format. No point estimates without CI — this is a research paper.
7.2) embeddings.paths.npy stores clip keys (not local paths) — used for Hard mode ±30s exclusion. Tags↔embeddings alignment via __key__ field. FAISS uses IVFFlat (not IVF-PQ) — simpler, sufficient at 10K-115K scale
7.3) m05c reads embeddings.paths.npy (deduped keys, ~5K) instead of subset_10k.json (10K). Ordering dependency: m05 must complete before m05c (enforced by run_evaluate.sh step ordering)
8) Devil's advocate: OOM, GPU underutil, data starvation, VRAM leaks, fp16 instability (use flash-attn-2), checkpoint corruption /auto-resume solution
9) GPU Optimizations:
- torch.compile(model) after model.eval() — warn about first-batch compile latency. **Exception**: skip torch.compile for adapted (float16) models loaded from .pt files — dynamo traces with float32 fake tensors, causing dtype mismatch crash.
- FAISS GPU: faiss.StandardGpuResources() + index_cpu_to_gpu(). Never CPU FAISS in GPU scripts
- cuML GPU: for iterative algorithms (UMAP, DBSCAN, KMeans, PCA) — 50-100x speedup. For metrics (silhouette, accuracy, F1) keep sklearn/numpy on CPU — post-inference, not a bottleneck
- Auto batch sizing: `scripts/profile_vram.py` → `outputs/profile/profile_data.json` → auto-detect in `run_pretrain.sh` (75% VRAM threshold). If profile_data.json missing, profiler runs automatically (~5 min).
- Attention per encoder: V-JEPA/shuffled/DINOv2 = FA2 (`attn_implementation="flash_attention_2"`), CLIP = SDPA (`attn_implementation="sdpa"`)
- Producer pre-processing: processor() runs in CPU producer thread, enqueues ready tensors → GPU thread only does .to(device) + forward pass
- CUDA memory fragmentation: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in shell scripts. Prevents OOM from reserved-but-unallocated memory on long training runs.
- wandb: shared src/utils/wandb_utils.py with: add_wandb_args(parser), init_wandb(module, mode, config, enabled), log_metrics(run, dict, step), log_image(run, key, path), log_artifact(run, name, path), finish_wandb(run). --no-wandb flag on every GPU module, all functions no-op when run=None
10) Each `print` statement must be `dynamic`. Remove/modify all false advertising / `static` prints from code.
11) Throughput reporting: NEVER use `total_clips / elapsed` when checkpoint clips are loaded at t=0 — this inflates initial rate and shows fake decline. Use **windowed throughput** (`clips_this_window / window_time`) for display, `new_clips_this_run / elapsed` for final summary.
12) Threading: NEVER use ThreadPoolExecutor for CPU-bound PyTorch tensor ops (augmentation, processor). ATen spawns ~80 internal threads per worker → 8 workers = 640+ threads → OS scheduler thrash → 0% throughput. Threading is ONLY for I/O-bound ops (video decode, file read). **Exception**: call `torch.set_num_threads(1)` inside each worker function to prevent ATen OpenMP oversubscription (PyTorch #37259).
13) **tqdm progress bar MANDATORY** in every GPU script's main processing loop. Must show: total count, current progress, rate (clips/s or items/s), ETA. Use `tqdm(total=N, desc="module_name", unit="clip")` with `pbar.update(batch_size)` per batch.
14) **LR warmup capped at 10% of total steps** for all modes. Prevents warmup exceeding training length on short runs. Implemented in `build_scheduler()`: `warmup = min(cfg_warmup, total_steps // 10)`.
15) **No hardcoded dataset sizes, clip counts, or magic numbers in Python.** Put them in `configs/pipeline.yaml` (clip limits, streaming params, eval params) or `configs/pretrain/*.yaml` (training params), or discover at runtime via `get_total_clips()` / `get_sanity_clip_limit()` from `utils/config.py`. **ENFORCED: code review + `.claude/hooks/fail-hard-research.sh`.**
16) **FAIL HARD in research pipelines.** No `|| continue`, `|| true`, or `WARNING`-without-exit in shell scripts. No bare `except: pass` in Python. Every error must crash immediately. Silent failures produce garbage metrics. **ENFORCED: `.claude/hooks/fail-hard-research.sh` blocks error-swallowing patterns via PreToolUse hook.**
17) **vjepa2 imports via shim** (`src/utils/vjepa2_imports.py`). vjepa2's `from src.models...` collides with our `src/utils/__init__.py`. The shim temporarily isolates sys.path + CWD to import vjepa2 modules without collision. Use `get_vit_giant_xformers()`, `get_vit_predictor()`, `get_mask_generator()`, `get_apply_masks()`. NEVER `from models.` or `from src.models.` directly.
18) **Epoch-based training, not step-based.** m09_pretrain.py uses `max_epochs` from YAML (per mode: sanity/poc/full/winner). Steps computed as `n_train // batch_size`. Ensures same clips processed regardless of batch size. SANITY clip count from `configs/pipeline.yaml` → `data.sanity_train_clips`.
19) **Per-lambda encoder paths for ablation.** Each lambda gets unique encoder name (e.g., `vjepa_lambda0`, `vjepa_lambda0_001`) via dynamic fallback in `get_encoder_info()`. m05 `--encoder` flag, m06 `--encoder` flag. Prevents overwriting across lambdas.
20) **No V-JEPA deduplication.** Using V-JEPA's own cosine similarity to filter eval clips is circular reasoning (model decides what it's evaluated on). Dedup removed from m05. Hard mode ±30s exclusion in m06 handles temporal duplicates (metadata-based, not model-based).
21) **Checkpoint disk management.** m09 exports only `student_encoder.pt` (~3.8GB) as deliverable. All intermediate checkpoints (latest, step*, final) cleaned after training. Periodic checkpoints use `full=False` (no optimizer, ~8GB vs ~16GB). `keep_last_n` from YAML.

# CONFIG FILES
- `configs/pipeline.yaml` — clip limits (SANITY/BAKEOFF), streaming params, GPU defaults, eval params, verification thresholds. Single source of truth for all `src/m*.py`.
- `configs/pretrain/vitg16_indian.yaml` — training hyperparameters (LR, EMA, masking, augmentation, epochs per mode, drift control, checkpointing, mixed precision).

# RULES (MUST follow)
- you do not be have to be yes-man on my very demand >> behave like a Sr. AI/ML Research engineer >> give me pros and cons of each of my demand
- Be brutally honest. Disagree [challenge me] when I'm wrong, but never hallucinate or lie.
- Devil's advocate does NOT mean fabricating bugs that don't exist. If code is correct, say so and move on.
- WEBSEARCH when needed to confirm universal AI/ML research practices.
- Git: provide commit message text only. NEVER run git commands. User handles all git ops via git_push.sh. **ENFORCED: `.claude/hooks/enforce-dev-rules.sh` blocks `git commit/push/add/reset` via PreToolUse hook.**
- NEVER draw conclusions from statistically insufficient data. A sanity check validates code correctness (no crashes, valid output), NOT model performance.
- Think like a Sr. Research Scientist: before making any recommendation, ask "do I have enough evidence for this claim?" If the answer is no, say so explicitly instead of speculating.
- GPU time is expensive — idle GPU = wasted money; idle user during GPU job = wasted time. Keep the GPU busy.
- Mandatory checklist for ANY GPU pipeline script: (1) tqdm progress bar, (2) auto-resume from checkpoint, (3) tee logging, (4) wandb integration, (5) windowed throughput reporting, (6) **output-exists guard** — check if final output file exists BEFORE loading model; skip if exists. **ENFORCED: Claude MUST run `/preflight <file>` after ANY edit to `src/m*.py` that touches main().**
- When auditing for hardcoded values, SHOW the grep output as proof. User does not trust "I audited everything" claims without evidence.

# HOOKS
- `.claude/hooks/enforce-dev-rules.sh` (PreToolUse:Bash) — blocks pip install, git state changes, bare python3
- `.claude/hooks/post-edit-lint.sh` (PostToolUse:Edit,Write) — auto py_compile + + ruff check on src/m*.py
- `.claude/hooks/fail-hard-research.sh` (PreToolUse:Edit,Write) — blocks `|| continue`, `|| true`, WARNING-without-exit, bare `except: pass`

# REFERENCE
- Bug history & batch speedup details: `iter/iter6/plan_batch_speedup.md`
- Ch10 expected vs real errors: `iter/iter7_training/expected_errors.md`
- Training plan: `iter/iter7_training/plan_training.md`
