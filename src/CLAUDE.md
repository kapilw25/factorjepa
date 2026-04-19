# PROJECT STRUCTURE
- Modules: `src/m00_*.py` … `src/m11_*.py` — prefix "m" avoids import errors. Numbers must NOT repeat. **Suffixed variants (`m04b`, `m09a/b/c`) ARE allowed** and signal related-but-isolated modules (e.g., m09a=vanilla pretrain, m09b=ExPLoRA, m09c=surgery — all share `src/utils/training.py` primitives, but each has its own full training loop for isolation per #49).
- Utils: `src/utils/` — shared functions only. **No cross-imports between m*.py files** (rule 32). **`src/utils/training.py` (#49) contract**: every function MUST be technique-agnostic. ZERO `if args.explora`/`if args.surgery`/`if cfg["technique"]` branches. Mode-specific behavior is configured via explicit parameters (`init_params=None`, `drift_cfg=None`, `explora_enabled=False`).
- Configs: `configs/pipeline.yaml` (clip limits, streaming, eval), `configs/model/*.yaml` (architecture), `configs/train/*.yaml` (technique, inherits `base_optimization.yaml`). Use `load_merged_config()` to merge. **No hardcoded values in Python** — YAML or runtime discovery only. **No `.get(key, default)` on YAML** — use `cfg[key]` so missing keys crash.
- Plots: both .png & .pdf. GPU scripts save .npy → CPU scripts (m08) read them. Never duplicate GPU compute in CPU scripts.
- Shell scripts are THIN wrappers — all logic in Python. No `python -c` inline, no `bc -l` math in shell.

# GPU HARDWARE & SOFTWARE
- **SANITY**: RTX Pro 4000 (24GB, ~$0.2/hr). **FULL**: RTX Pro 6000 Blackwell (96GB, ~$0.8/hr). **Mac**: CPU/lint only.
- Stack: PyTorch 2.12.0+cu128 nightly, CUDA 12.8, FA2 2.8.3, FAISS-GPU 1.14.1, cuML 26.04, SAM 3.1, Python 3.12, UV.
- **GPU util ≥85% is TOP PRIORITY.** Idle GPU = wasted money. Fix I/O pipeline (parallelize TAR readers, increase DECODE_WORKERS/PREFETCH_QUEUE), not the model.
- **No CPU fallback** in inference/compute scripts (m04/m05/m06/m07/m09/m10). FATAL if GPU path missing. Exception: m08 (plotting, CPU-only).
- torch.compile after model.eval(). For adapted models, monkey-patch `torch.backends.cuda.sdp_kernel = contextlib.nullcontext` before compile (PyTorch #130098). FAISS: `index_cpu_to_gpu()`. cuML for iterative algos. Attention: V-JEPA/DINOv2=FA2, CLIP=SDPA.
- Auto batch sizing: `profile_vram.py` → `profile_data.json` → 75% VRAM threshold. Auto-run profiler if missing.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in shell scripts. Producer pre-processing in CPU thread; GPU thread only `.to(device)` + forward.
- Threading: NEVER ThreadPoolExecutor for CPU-bound tensor ops. `torch.set_num_threads(1)` inside workers.

# CODE STANDARDS
- **All imports at TOP.** Only exception: guarded `try/except ImportError` for optional deps.
- **Docstrings**: max 2-line + terminal commands. Format: `"""One-line. GPU-only.\n    python -u src/file.py --SANITY 2>&1 | tee logs/file.log\n"""`
- **Dependencies**: update `setup_env_uv.sh` + `requirements.txt`/`requirements_gpu.txt`. Install via `setup_env_uv.sh` ONLY for reproducibility purpose, no individual installation.
- **FAIL HARD.** No `|| continue`, `|| true`, WARNING-without-exit in shell. No bare `except: pass` in Python. Silent failures = garbage metrics.
- **95% CI MANDATORY**: Every metric needs bootstrap 95% CI (BCa, 10K iter via `utils/bootstrap.py`). No point estimates without CI.
- **Vectorize**: Replace Python for-loops with NumPy when iterating 1K+ items. Incident: m06 bootstrap 88 min → <1 min vectorized.
- **Organize outputs** into clean subdirectories. Each logical group owns its own dir + JSON. Never flatten structures.
- Each `print` must be dynamic (no static false advertising). Throughput: use windowed rate, not `total/elapsed`.
- vjepa2 imports via `utils/vjepa2_imports.py` shim. Use `get_vit_by_arch(arch)`. NEVER import `from models.` directly.

# GPU PIPELINE CHECKLIST
Every `src/m*.py` using GPU MUST have: (1) `check_gpu()`, (2) `cleanup_temp()`, (3) `verify_or_skip()`, (4) auto-batch from VRAM, (5) `save/load_checkpoint()`, (6) `iter_clips_parallel()`, (7) `make_pbar()` with total/rate/ETA, (8) `init_wandb()`+`--no-wandb`, (9) `get_output_dir()`, (10) output-exists guard before loading model, **(11) `AdaptiveBatchSizer` from `utils.gpu_batch` wired into the forward loop — sub-batch the producer batch, `try/except OutOfMemoryError` → `cuda_cleanup()` + `sizer.on_oom()` → retry; success → `sizer.after_batch_success()`. `memory_cap` ALWAYS reads `pipeline.yaml gpu_memory_target` (universal — never hardcoded). `initial_size` reads per-module `inference_*_initial_bs` from yaml. `max_size` reads the module's existing BS cap. Per-clip-session scripts like m10 instead empty_cache() every N clips for fragmentation hygiene. Training scripts (m09): per-epoch gc + empty_cache + ipc_collect; full mid-step sizer is research-coupled (changes optimizer dynamics) — log as deferred. See errors_N_fixes.md #46/#47.** CPU scripts (m06c/m08): skip GPU-specific items. **tqdm MANDATORY in EVERY `src/m*.py`** (not just GPU scripts).

# TRAINING RULES
- **Epoch-based, not step-based.** `max_epochs` from YAML per mode. Steps = `n_train // batch_size`.
- **LR warmup capped at 10%** of total steps. Predictor LR 1x (gold standard audit).
- **Crash-safe JSONL logging** with `os.fsync()` every write. CSV for backward compat only.
- **Checkpoint management**: export only `student_encoder.pt` (~3.8GB). Clean all intermediates after training. Periodic saves use `full=False` (~8GB, no optimizer). `keep_last_n` from YAML.
- **Per-lambda encoder paths** for ablation. Dynamic fallback in `get_encoder_info()`.
- **No V-JEPA deduplication** — circular reasoning. Hard mode ±30s exclusion is metadata-based.
- Use `--model-config` + `--train-config`. Model configs = architecture. Train configs = technique.

# TESTING & VALIDATION
- **3-check gate after ANY edit to `src/**/*.py`**: (1) `py_compile`, (2) `ast.parse`, (3) `ruff check --select F,E9`. Auto-enforced by `post-edit-lint.sh` hook.
- **End-to-end REPL test** before restarting pipelines. Test FULL code path with real data, not just import.
- **Trace data flow** after adding CLI flags: flag → argparse → `get_output_dir()` → correct directory. `shellcheck scripts/*.sh`.
- SANITY validates code correctness (no crashes), NOT model performance. Never draw conclusions from insufficient data.

# WORKFLOW RULES
- **Goal override**: #1 priority is research results. Every recommendation must maximize "Surgery(Prec@k) > Frozen(Prec@K) with non-overlapping 95 % CIs. Never filter by implementation effort.
- **Never sacrifice metric accuracy for speed.** Eval must match frozen baseline exactly (64 frames, same resolution, ImageNet norm).
- **Never choose the easy option. WEBSEARCH for the gold-standard solution that preserves BOTH highest accuracy AND highest throughput.** When hitting a runtime error, the first reflex (disable the offending feature, fallback to slower path, add a `|| true`, skip the hard case) is almost always wrong. BEFORE proposing any fix that trades off throughput (disabling torch.compile, falling back to eager, reducing batch size, downgrading dtype) OR accuracy (skipping a mask, lowering a threshold, dropping a factor), websearch how the community (HF, facebookresearch, Dao-AILab, PyTorch core) solved the exact same error on a comparable stack. Cite ≥2 sources. Example: disabling `torch.compile` on a 2B model to dodge a RoPE-induced Q/K/V dtype mismatch is lazy; the correct fix is a one-line `q, k = q.to(v.dtype), k.to(v.dtype)` before SDPA (the idiom used by HF Llama + naver-ai/rope-vit + HF PR #36065). See `iter/iter8/errors_N_fixes.md` #44.
- **Interrupt freely**: All GPU scripts have checkpoint resume. Don't say "let the run complete" when a fix is ready.
- Be a Sr. AI/ML Research Engineer — give pros/cons, disagree when wrong, never hallucinate. WEBSEARCH before recommending.
- Git: provide commit message text only. NEVER run git commands (enforced by hook). User handles via `git_push.sh`.
- When auditing for hardcoded values, SHOW grep output as proof. User does not trust "I checked" claims without evidence.
- wandb: shared `utils/wandb_utils.py`. `--no-wandb` on every module. All functions no-op when `run=None`.
- **Update CLAUDE.md + MEMORY.md** at end of every session with new results/pivots/decisions. Sync `src/MEMORY.md` → `~/.claude/projects/.../memory/MEMORY.md`.

# HOOKS
- `enforce-dev-rules.sh` (PreToolUse:Bash) — blocks pip install, git state changes, bare `python3` without venv activation
- `post-edit-lint.sh` (PostToolUse:Edit,Write) — auto py_compile + ruff on src/**/*.py
- `fail-hard-research.sh` (PreToolUse:Edit,Write) — blocks `|| continue`, `|| true`, bare `except: pass`

# CONFIGS
- `configs/pipeline.yaml` — clip limits, streaming, GPU defaults, eval params, encoder registry
- `configs/model/vjepa2_1.yaml` — PRIMARY (V-JEPA 2.1 ViT-G 2B, 1664-dim)
- `configs/model/vjepa2_0.yaml` — legacy (V-JEPA 2.0 ViT-g 1B, 1408-dim)
- `configs/train/base_optimization.yaml` — shared: masking, augmentation, AdamW, EMA, mixed precision
- `configs/train/ch10_pretrain.yaml` — continual pretraining (drift control, lambda sweep)
- `configs/train/explora.yaml` — ExPLoRA (LoRA rank=16 + unfreeze 2 blocks)
- `configs/train/ch11_surgery.yaml` — factor surgery (3-stage progressive unfreezing + SAM3 params)

# 📚 REFERENCE
- 🏗️ Training plan: `iter/iter8/plan_training.md` (HIGH level — system design, literature)
- 📋 TODO + status: `iter/iter8/plan_TODO.md` (MID level — kanban, m09c iteration table, time budget)
- 🚀 Runbook: `iter/iter8/runbook.md` (LOW level — GPU-ready commands + verify tables)
- 🐛 Bug log: `iter/iter8/errors_N_fixes.md` (ERROR level — 61 entries catalogued, #1-#61 as of 2026-04-17)
- 🛡️ Preflight CPU-side guards: `.claude/skills/preflight/SKILL.md` (B1-B42 static checks citing errors_N_fixes entries)
- 📖 Fallback techniques if Surgery fails: `iter/utils/literarure_survey.md` (24 JEPA variants, SIGReg / VLA-JEPA / temporal-projection)
