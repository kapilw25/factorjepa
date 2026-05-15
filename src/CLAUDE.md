# PROJECT STRUCTURE
- Modules: `src/m00_*.py` … `src/m11_*.py` — prefix "m" avoids import errors. Numbers must NOT repeat. **Suffixed variants (`m04b`, `m09a/b/c`) ARE allowed** and signal related-but-isolated modules (e.g., m09a=vanilla pretrain, m09b=ExPLoRA, m09c=surgery — all share `src/utils/training.py` primitives, but each has its own full training loop for isolation per #49).
- Utils: `src/utils/` — shared functions only. **No cross-imports between m*.py files** (rule 32). **`src/utils/training.py` (#49) contract**: every function MUST be technique-agnostic. ZERO `if args.explora`/`if args.surgery`/`if cfg["technique"]` branches. Mode-specific behavior is configured via explicit parameters (`init_params=None`, `drift_cfg=None`, `explora_enabled=False`).
- Configs: `configs/pipeline.yaml` (clip limits, streaming, eval), `configs/model/*.yaml` (architecture), `configs/train/*.yaml` (technique, inherits `base_optimization.yaml`). Use `load_merged_config()` to merge. **No hardcoded values in Python** — YAML or runtime discovery only. 
- **No DEFAULT, no hardcoded paths, no FALLBACK: — silent error leads to research paper rejection.** No `.get(key, default)` on YAML — use `cfg[key]` so missing keys crash. NO module-level path constants (`Path("data/...")`, `Path("outputs/...")`, `OUTPUT_DIR = ...`, etc.). Every `.py`/`.yaml`/`.pt`/`.json`/`.md`/`.npy`/`.npz`/`.csv`/`.tar` path arrives via `argparse.add_argument(..., required=True)` with NO `default=`. Each .py's docstring USAGE block shows the full sample command for `--SANITY` / `--POC` / `--FULL` with every previously-hardcoded path passed as an arg. Same in shell wrappers (`$1`/`$@` or `yaml_extract <yaml> <key>` — never inline). 
- FAIL LOUD. This is not a production project. 
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
- **Training scripts MUST cite official gold-standard repo URL in docstring.**
- **Dependencies**: update `setup_env_uv.sh` + `requirements.txt`/`requirements_gpu.txt`. Install via `setup_env_uv.sh` ONLY for reproducibility purpose, no individual installation.
- **FAIL HARD.** No `|| continue`, `|| true`, WARNING-without-exit in shell. No bare `except: pass` in Python. **No `getattr(args, key, default)`** — argparse `required=True` already guarantees presence; `getattr` defaults swallow the error and let `None` propagate to multi-h GPU runs that silently produce wrong `.npy`. Pass values as explicit function parameters instead. See errors_N_fixes #79. Silent failures = garbage metrics.
- **95% CI MANDATORY**: Every metric needs bootstrap 95% CI (BCa, 10K iter via `utils/bootstrap.py`). No point estimates without CI.
- **Vectorize**: Replace Python for-loops with NumPy when iterating 1K+ items. Incident: m06 bootstrap 88 min → <1 min vectorized.
- **Organize outputs** into clean subdirectories. Each logical group owns its own dir + JSON. Never flatten structures.
- Each `print` must be dynamic (no static false advertising). Throughput: use windowed rate, not `total/elapsed`.
- vjepa2 imports via `utils/vjepa2_imports.py` shim. Use `get_vit_by_arch(arch)`. NEVER import `from models.` directly.

# DELETE PROTECTION (iter11 — META-fix for v1→v10 cycle)
- **No shell-level `rm`.** All destructive deletes live in .py behind `--cache-policy`. Past pattern (script wipes `.npy` + m05 deletes its own checkpoint post-save) destroyed both durable artifacts every round-trip and rebuilt the same 9,297-clip frozen embeddings ~10 h across v1→v10. Scripts use merge-`cp -rf`, atomic `os.replace`, `: >` truncate — never `rm`.
- **`utils.cache_policy` is the SINGLE cross-module guard.** `output_guard` was removed (2026-04-26) because its silent skip masked `--cache-policy` and made re-runs surprising. Per-script intra-run resume (`load_checkpoint`, fingerprint files like `.m05_checkpoint_*_<fp>.npz`) handles "skip already-finished clips" — that is resume, not a guard.
- **CLI contract**: each .py (m04/m04d/m05/m05b/m05c/m06/m08/m08b/m09a/m09b/m09c/m10/m11) registers `--cache-policy {1,2}` via `utils.cache_policy.add_cache_policy_arg()` AND prompts the user via `input("[1=keep / 2=recompute] (Enter=1): ")` if the flag wasn't supplied (see `resolve_cache_policy_interactive` pattern below). `1=keep` (default, Enter) preserves all caches with a log line; `2=recompute` authorizes destruction through `guarded_delete(path, policy, label)`. No `.py` deletes anything without the flag reaching `2`.
- **Shell contract**: shells stay THIN. Wrappers (`scripts/*.sh`) do NOT prompt — they just forward CLI args to the .py. Overnight/tmux runs pass `--cache-policy 1` (or `2`) explicitly to bypass the .py's `input()` prompt; or set `CACHE_POLICY_ALL=1|2` env var which each .py honors before prompting.
- **Retiring files**: `mv` to sibling `legacy/` subdir — never `rm`.

# GPU PIPELINE CHECKLIST
Every `src/m*.py` using GPU MUST have: (1) `check_gpu()`, (2) `cleanup_temp()`, (3) `add_cache_policy_arg()` + interactive `input()` prompt for cache-policy, (4) auto-batch from VRAM, (5) `save/load_checkpoint()`, (6) `iter_clips_parallel()`, (7) `make_pbar()` with total/rate/ETA, (8) `init_wandb()`+`--no-wandb`, (9) `get_output_dir()`, (10) `guarded_delete()` for any destructive file ops, **(11) `AdaptiveBatchSizer` from `utils.gpu_batch` wired into the forward loop — sub-batch the producer batch, `try/except OutOfMemoryError` → `cuda_cleanup()` + `sizer.on_oom()` → retry; success → `sizer.after_batch_success()`. `memory_cap` ALWAYS reads `pipeline.yaml gpu_memory_target` (universal — never hardcoded). `initial_size` reads per-module `inference_*_initial_bs` from yaml. `max_size` reads the module's existing BS cap. Per-clip-session scripts like m10 instead empty_cache() every N clips for fragmentation hygiene. Training scripts (m09): per-epoch gc + empty_cache + ipc_collect; full mid-step sizer is research-coupled (changes optimizer dynamics) — log as deferred. See errors_N_fixes.md #46/#47.** CPU scripts (m06c/m08): skip GPU-specific items. **tqdm MANDATORY in EVERY `src/m*.py`** (not just GPU scripts).

# TRAINING RULES
- **Epoch-based, not step-based.** `max_epochs` from YAML per mode. Steps = `n_train // batch_size`.
- **LR warmup capped at 10%** of total steps. Predictor LR 1x (gold standard audit).
- **Crash-safe JSONL logging** with `os.fsync()` every write. CSV for backward compat only.
- **Checkpoint management**: export only `student_encoder.pt` (~3.8GB). Clean all intermediates after training. Periodic saves use `full=False` (~8GB, no optimizer). `keep_last_n` from YAML.
- **Per-lambda encoder paths** for ablation. Dynamic fallback in `get_encoder_info()`.
- **No V-JEPA deduplication** — circular reasoning. Hard mode ±30s exclusion is metadata-based.
- Use `--model-config` + `--train-config`. Model configs = architecture. Train configs = technique.
- **Factor streaming (iter9+)**: m09c has two factor-data paths that coexist. Legacy = `FactorSampler` + `load_factor_clip` reading m11 D_L/D_A `.npy` files (~340 GB @ 10K); streaming = `StreamingFactorDataset(IterableDataset)` + `DataLoader(num_workers, persistent_workers, prefetch_factor)` generating D_L/D_A on-demand from `(raw_mp4, mask.npz)` via `utils.factor_streaming.stream_factor` (~40 GB @ 10K). Mode-gated in `ch11_surgery.yaml > factor_streaming: {sanity,poc,full}`; CLI override `--factor-streaming` / `--no-factor-streaming` wins. Bitwise parity verified against legacy disk output — see `scripts/tests_streaming/test_parity.py`. Under streaming, `m11 --streaming` short-circuits all non-verify clips (manifest-only, no MP4 decode, no scipy blur, no np.save) → ~90% m11 wall-time reduction. `plot_factor_per_clip` keeps reading .npy → m11 still materializes the 100 `select_verify_clips(seed=42)` regardless.

# TESTING & VALIDATION
- **3-check gate after ANY edit to `src/**/*.py`**: (1) `py_compile`, (2) `ast.parse`, (3) `ruff check --select F,E9`. Auto-enforced by `post-edit-lint.sh` hook.
- **End-to-end REPL test** before restarting pipelines. Test FULL code path with real data, not just import.
- **Trace data flow** after adding CLI flags: flag → argparse → `get_output_dir()` → correct directory. `shellcheck scripts/*.sh`.
- SANITY validates code correctness (no crashes), NOT model performance. Never draw conclusions from insufficient data.
- **POC ↔ FULL parity (mandatory)**: POC must be a byte-identical scaled-down copy of FULL. The ONLY legitimate POC vs FULL deltas are `poc_total_clips` (subset size) and `max_epochs.poc` (epoch budget). Every other yaml/CLI flag — motion_aux on/off, multi_task on/off, label `n_classes`, head dim, optimizer, EMA τ, warmup type, augmentations — MUST match between POC and FULL. If POC produces degenerate inputs (missing classes, sparse labels, undersized samples), **fix the POC sampler** so its output schema matches FULL — never propose "disable feature X at POC, re-enable at FULL". 2026-05-09 incident: I recommended skipping motion_aux at POC because 855-clip / 7-class POC labels were degenerate; user invoked this rule — the right fix was to repair the POC label sampler to guarantee 8-class stratified output, not to branch the recipe.

# WORKFLOW RULES

## GOAL OVERRIDE
#1 priority is research results. Every recommendation must maximize "Surgery outperform Frozen on [Prec@K/mAP@K/Cycle@K] with non-overlapping 95 % CIs. Never filter by implementation effort. Never take shortcut at the cost of compromising results.

## ACCURACY OVER SPEED
Never sacrifice metric accuracy for speed. Eval must match frozen baseline exactly (64 frames, same resolution, ImageNet norm).

## NO LAZY FIX — WEBSEARCH FOR GOLD STANDARD
Never choose the easy option. WEBSEARCH for the gold-standard solution that preserves BOTH highest accuracy AND highest throughput. When hitting a runtime error, the first reflex (disable the offending feature, fallback to slower path, add a `|| true`, skip the hard case) is almost always wrong. BEFORE proposing any fix that trades off throughput (disabling torch.compile, falling back to eager, reducing batch size, downgrading dtype) OR accuracy (skipping a mask, lowering a threshold, dropping a factor), websearch how the community (HF, facebookresearch, Dao-AILab, PyTorch core) solved the exact same error on a comparable stack. Cite ≥2 sources. Example: disabling `torch.compile` on a 2B model to dodge a RoPE-induced Q/K/V dtype mismatch is lazy; the correct fix is a one-line `q, k = q.to(v.dtype), k.to(v.dtype)` before SDPA (the idiom used by HF Llama + naver-ai/rope-vit + HF PR #36065). See `iter/iter8/errors_N_fixes.md` #44.

## VERIFY-FIRST RECOMMENDATIONS (NO LAZY ANSWERS)
**Do NOT trust the first recommendation Claude generates. Claude is lazy and biased toward the least-token answer that "sounds right".** Every recommendation that the user could act on (cache-policy choices, "is X safe to run", "is the cache reusable", "did the run complete", "is this file the latest", "does Y break Z", "should I pick option A or B") MUST be backed by **direct evidence gathered in the same turn** — log greps, file shape/mtime checks, jsonl tails, code reads, WEBSEARCH cites — before being delivered to the user. Banned response patterns: "should be fine", "I think it's safe", "the cache should reuse", "no changes needed" — without a tool-call result quoted in the same response. Show the evidence inline, then state the recommendation. If you cannot gather evidence in the available tool budget, say "cannot verify" and ask the user, do NOT guess. Concrete incidents: (1) 2026-04-26 — claimed `m05_frozen` cache was reusable as "1=keep" without checking shape; the cached `.npy` was actually 9297 rows from a prior `eval_10k.json` run with **0/9297 overlap** with the current 308-clip `ultra_hard_3066_eval.json` — would have silently fed m06 the wrong embeddings. The 30-second `np.load(...).shape` + clip-key intersect check would have caught it. (2) 2026-04-20 — claimed `sed -i` on a live bash script was safe; silently failed and burned 4 h of GPU. **Verification budget is cheaper than retraction.**

## INTERRUPT FREELY
All GPU scripts have checkpoint resume. Don't say "let the run complete" when a fix is ready.

## OVERNIGHT CHAINS — `;` NOT `&&`
Use `;` not `&&` when running INDEPENDENT long-running commands back-to-back (e.g., `training.sh ; eval.sh`). `&&` aborts the chain if the first command exits non-zero — a single 703-clip decode failure in m05 killed the 8-hour paired_eval queue on 2026-04-22 because the 2nd command was gated behind the 1st's success. Both downstream commands have their own resume/checkpoint logic; a failed upstream command should NOT silently cancel the queue. Reserve `&&` ONLY for cases where the 2nd command literally cannot run without the 1st's output (and even then, prefer explicit `if [ -f output ]; then next.sh; else echo FAIL; fi`). Pattern: `./train.sh 2>&1 | tee train.log ; ./eval.sh 2>&1 | tee eval.log` (semicolon). **Devil's advocate scope extends to the enclosing shell / orchestrator / tmux layer — most glue-layer bugs sit one level above where script-internal review looks.**

## SR. AI/ML ENGINEER POSTURE
Be a Sr. AI/ML Research Engineer — give pros/cons, disagree when wrong, never hallucinate. WEBSEARCH before recommending.

## MANDATORY PROS/CONS GATE
**NO bogus / gamble solutions.** BEFORE running any tool call that changes state (Edit, Write, Bash mutations like `sed -i`/`cp`/`mv`/`rm`, HF push, process kill, git ops), AND before recommending an action to the user, list **≥ 3 pros AND ≥ 3 cons** for **every option considered** (minimum 2 options). "Do nothing / wait" counts as an option and must have its own pros/cons. Never skip this gate for "fast / small / obvious" fixes — the 2026-04-20 `sed -i` on a running bash script looked obvious, silently failed, and burned 4 h of overnight GPU. Rationale: every tool call is a commitment with hidden cost surface (inode races, process state, FD caching, silent no-ops). If you cannot articulate 3 cons for your chosen option, you haven't thought hard enough to act. If the user asks you to act without providing this analysis, write it out first, THEN wait for explicit "go" — do NOT silently execute.

## NO DEFER, NO TECH DEBT
**Fix at the right layer, now.** Language like "defer to iter10", "harmless, minor cost", "not worth fixing now", "revisit later", "acceptable at this scale" is BANNED framing. If a mismatch exists (stale yaml vs code, unused output generation, unchecked config flag, misaligned paths, silent no-op), it gets fixed **in this session** at the layer where it originates — not patched around, not added as a "backlog item", not shrugged off as negligible. Scalability rationale: 1.1 GB of unused D_I at 10K becomes ~5 GB at 50K and ~12 GB at 115K — same line of code, linearly worse over the scale ladder. A single deferred fix compounds: 3 deferrals and the codebase is a minefield of "harmless at current scale" traps that each cost a multi-hour incident when the next scale lights them up. When tempted to say "defer", instead: (a) identify the minimum surgical fix (usually one `if cfg[flag]:` guard, one yaml key, or one caller update), (b) apply it at the originating layer, (c) verify downstream callers still work. If the fix is genuinely out-of-scope (e.g., requires 500 LoC of new infrastructure), say so explicitly with LoC estimate and blocked work — do NOT hide it under a "minor" or "harmless" adjective. Every "it's fine, let's just ship it" framing is a future incident postponed, not avoided.

## GIT
Provide commit message text only. NEVER run git commands (enforced by hook). User handles via `git_push.sh`.

## EVIDENCE FOR HARDCODED-VALUE AUDITS
When auditing for hardcoded values, SHOW grep output as proof. User does not trust "I checked" claims without evidence.

## WANDB
Shared `utils/wandb_utils.py`. `--no-wandb` on every module. All functions no-op when `run=None`.

## SESSION-END SYNC
**Update CLAUDE.md + MEMORY.md** at end of every session with new results/pivots/decisions. Sync `src/MEMORY.md` → `~/.claude/projects/.../memory/MEMORY.md`.

## OUTPUT FORMATTING — TABLES, NOT LISTS  (with emojis for scannability)
**Default to ASCII box-drawing tables** (`┌─┬─┐` `│` `├─┼─┤` `└─┴─┘`) for ANY comparison spanning ≥2 columns × ≥2 rows. Markdown pipe tables (`| col | col |`) and bullet lists are BANNED for comparison data — Claude Code's CommonMark renderer flattens them into unaligned single-column "lists" in the user's terminal, which breaks the comparison and frustrates the user. **Use emojis LIBERALLY** in column headers, row identifiers, and status/marker columns for visual scannability — the user explicitly chose "eyeballable with emojis" over "perfectly aligned plain ASCII". Keep emojis OUT of pure-numeric cells (where width drift matters most) but in headers / status / verdict columns they're encouraged. Box-drawing borders MUST still be present so the table structure is visible even if cell widths drift slightly with emoji rendering. Always declare a marker legend below the table. POC ablation sweeps (Cell A/B/C/D × N metrics) MUST be a single box-drawn grid, not N grouped lists.

# HOOKS
- `enforce-dev-rules.sh` (PreToolUse:Bash) — blocks pip install, git state changes, bare `python3` without venv activation
- `post-edit-lint.sh` (PostToolUse:Edit,Write) — auto py_compile + ruff on src/**/*.py
- `fail-hard-research.sh` (PreToolUse:Edit,Write) — blocks `|| continue`, `|| true`, bare `except: pass`

# CONFIGS
- `configs/pipeline.yaml` — clip limits, streaming, GPU defaults, eval params, encoder registry
- `configs/model/vjepa2_1.yaml` — PRIMARY (V-JEPA 2.1 ViT-G 2B, 1664-dim)
- `configs/model/vjepa2_0.yaml` — legacy (V-JEPA 2.0 ViT-g 1B, 1408-dim)
- `configs/train/base_optimization.yaml` — shared: masking, augmentation, AdamW, EMA, mixed precision
- `configs/legacy2/ch10_pretrain.yaml` — continual pretraining (drift control, lambda sweep)
- `configs/legacy2/explora.yaml` — ExPLoRA (LoRA rank=16 + unfreeze 2 blocks)
- `configs/train/ch11_surgery.yaml` — factor surgery (2-stage progressive unfreezing, SAM3 params, early-stop triggers, `factor_streaming` block)

# 📚 REFERENCE
- 🏗️ Training plan: `iter/iter8/plan_training.md` (HIGH level — system design, literature)
- 📋 TODO + status: `iter/iter8/plan_TODO.md` (MID level — kanban, m09c iteration table, time budget)
- 🚀 Runbook: `iter/iter8/runbook.md` (LOW level — GPU-ready commands + verify tables)
- 🐛 Bug log: `iter/iter8/errors_N_fixes.md` (ERROR level — 63 entries catalogued, #1-#63 as of 2026-04-19: +#62 poc_simplified removed, +#63 probe infra landed)
- 🛡️ Preflight CPU-side guards: `.claude/skills/preflight/SKILL.md` (B1-B42 static checks citing errors_N_fixes entries)
- 📖 Fallback techniques if Surgery fails: `iter/utils/literarure_survey.md` (24 JEPA variants, SIGReg / VLA-JEPA / temporal-projection)
