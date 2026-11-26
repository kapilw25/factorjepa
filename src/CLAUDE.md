# PROJECT STRUCTURE
- Modules: `src/m00_*.py` … `src/m11_*.py` — prefix "m" avoids import errors. Numbers must NOT repeat. **Suffixed variants (`m04b`, `m09a/b/c`) ARE allowed** and signal related-but-isolated modules (e.g., m09a=vanilla pretrain, m09b=ExPLoRA, m09c=surgery — all share `src/utils/training.py` primitives, but each has its own full training loop for isolation per #49).
- Utils: `src/utils/` — shared functions only. **No cross-imports between m*.py files** (rule 32). **`src/utils/training.py` (#49) contract**: every function MUST be technique-agnostic. ZERO `if args.explora`/`if args.surgery`/`if cfg["technique"]` branches. Mode-specific behavior is configured via explicit parameters (`init_params=None`, `drift_cfg=None`, `explora_enabled=False`).
- Configs: `configs/pipeline.yaml` (clip limits, streaming, eval), `configs/model/*.yaml` (architecture), `configs/train/*.yaml` (technique, inherits `base_optimization.yaml`). Use `load_merged_config()` to merge. **No hardcoded values in Python** — YAML or runtime discovery only. 
- **No DEFAULT, no hardcoded paths, no FALLBACK: — silent error leads to research paper rejection.** No `.get(key, default)` on YAML — use `cfg[key]` so missing keys crash. NO module-level path constants (`Path("data/...")`, `Path("outputs/...")`, `OUTPUT_DIR = ...`, etc.). Every `.py`/`.yaml`/`.pt`/`.json`/`.md`/`.npy`/`.npz`/`.csv`/`.tar` path arrives via `argparse.add_argument(..., required=True)` with NO `default=`. Each .py's docstring USAGE block shows the full sample command for `--SANITY` / `--POC` / `--FULL` with every previously-hardcoded path passed as an arg. Same in shell wrappers (`$1`/`$@` or `yaml_extract <yaml> <key>` — never inline). 
- **SHARED DERIVATION VIA CLI — NO PER-MODULE RE-DERIVATION (iter17 2026-05-26).** Any value that (a) SELECTS or PARTITIONS data (clip pools, train/val/test splits, manifests, subsets, factor indices, mask dirs, label sets) OR (b) MUST be identical across sibling modules (`m09a1/a2/c1/c2`) is computed **ONCE** by a shared util (e.g. `src/utils/clip_splits.py`), invoked by the THIN shell, and passed to every `m*.py` as an explicit CLI arg / consumed artifact. **NEVER re-derive it independently inside each module.** Why this is a hard rule: iter15's two validity-killing bugs both came from per-module derivation — (1) `m09c1` built `train_keys = manifest − val` internally and *forgot test* → the eval-TEST clips leaked into surgery's SSL pool; (2) pretrain derived its pool from the ~1k labeled `train_split` while surgery derived from the ~9k broad manifest → an **8× data-volume confound**. Both were INVISIBLE for an entire iter because each trainer rolled its own `train_keys`. The fix: `run_train.sh` builds ONE leakage-safe `train_pool.json` (`= universe − val − test`) via `clip_splits.py` and feeds it to all trainers as `--subset`. **Rule of thumb: if two modules must agree on X, X is a CLI input from the shell / a shared-util output — not an in-module computation.** Divergent path conventions (e.g. one module deriving `factor_manifest` from `--factor-dir`, another from `--local-data / "m11_factor_datasets"`) are the same class of latent bug — route them through one source too.
- FAIL LOUD. This is not a production project. 
- Plots: both .png & .pdf. GPU scripts save .npy → CPU scripts (m08) read them. Never duplicate GPU compute in CPU scripts. **Analysis-phase figure styling (cell-annotated heatmaps/scorecards: all text BLACK+BOLD, numbers sized to fill the cell, ASCII glyphs only): `.claude/plotting.md`.**
- Shell scripts are THIN wrappers — all logic in Python. No `python -c` inline, no `bc -l` math in shell.

# GPU HARDWARE & SOFTWARE
- **SANITY**: RTX Pro 5000 (48GB, ~$1/hr). **FULL**: RTX Pro 6000 Blackwell (96GB, ~$1.5/hr). **Mac**: CPU/lint only.
- Stack: PyTorch 2.12.0+cu130, CUDA 13.0, FA2 2.8.3, FAISS-GPU 1.14.1, cuML 26.06, SAM 3.1, Python 3.12, UV.  (resolved by setup_env_uv.sh on the Blackwell sm_120 box, 2026-06-14)
- **GPU util ≥85% is TOP PRIORITY.** Idle GPU = wasted money. Fix I/O pipeline (parallelize TAR readers, increase DECODE_WORKERS/PREFETCH_QUEUE), not the model.
- **No CPU fallback** in inference/compute scripts (m04/m05/m06/m07/m09/m10). FATAL if GPU path missing. Exception: m08 (plotting, CPU-only).
- torch.compile after model.eval(). For adapted models, monkey-patch `torch.backends.cuda.sdp_kernel = contextlib.nullcontext` before compile (PyTorch #130098). FAISS: `index_cpu_to_gpu()`. cuML for iterative algos. Attention: V-JEPA/DINOv2=FA2, CLIP=SDPA.
- Auto batch sizing: `profile_vram.py` → `profile_data.json` → 75% VRAM threshold. Auto-run profiler if missing.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in shell scripts. Producer pre-processing in CPU thread; GPU thread only `.to(device)` + forward.
- Threading: NEVER ThreadPoolExecutor for CPU-bound tensor ops. `torch.set_num_threads(1)` inside workers.

# CODE STANDARDS
- **All imports at TOP.** Only exception: guarded `try/except ImportError` for optional deps.
- **Training scripts MUST cite official gold-standard repo URL in docstring.**
- **Dependencies**: update `setup_env_uv.sh` + `requirements.txt`/`requirements_gpu.txt`. Install via `setup_env_uv.sh` ONLY for reproducibility purpose, no individual installation.
- **FAIL HARD.** No `|| continue`, `|| true`, WARNING-without-exit in shell. No bare `except: pass` in Python. **No `getattr(args, key, default)`** — argparse `required=True` already guarantees presence; `getattr` defaults swallow the error and let `None` propagate to multi-h GPU runs that silently produce wrong `.npy`. Pass values as explicit function parameters instead. See errors_N_fixes #79. Silent failures = garbage metrics.
- **95% CI MANDATORY**: Every metric needs bootstrap 95% CI (BCa, 10K iter via `utils/bootstrap.py`). No point estimates without CI.
- **Vectorize**: Replace Python for-loops with NumPy when iterating 1K+ items. Incident: m06 bootstrap 88 min → <1 min vectorized.
- **Organize outputs** into clean subdirectories. Each logical group owns its own dir + JSON. Never flatten structures.
- **Timestamped logs.** Every log path ends with `_$(date +%Y%m%d_%H%M%S).log` (shell) / `time.strftime("%Y%m%d_%H%M%S")` (py) — never a static name; re-runs must NOT silently overwrite a prior log.
- Each `print` must be dynamic (no static false advertising). Throughput: use windowed rate, not `total/elapsed`.
- vjepa2 imports via `utils/vjepa2_imports.py` shim. Use `get_vit_by_arch(arch)`. NEVER import `from models.` directly.

# DELETE PROTECTION (iter11 — META-fix for v1→v10 cycle)
- **No shell-level `rm`.** All destructive deletes live in .py behind `--cache-policy`. Past pattern (script wipes `.npy` + m05 deletes its own checkpoint post-save) destroyed both durable artifacts every round-trip and rebuilt the same 9,297-clip frozen embeddings ~10 h across v1→v10. Scripts use merge-`cp -rf`, atomic `os.replace`, `: >` truncate — never `rm`.
- **`utils.cache_policy` is the SINGLE cross-module guard.** `output_guard` was removed (2026-04-26) because its silent skip masked `--cache-policy` and made re-runs surprising. Per-script intra-run resume (`load_checkpoint`, fingerprint files like `.m05_checkpoint_*_<fp>.npz`) handles "skip already-finished clips" — that is resume, not a guard.
- **CLI contract**: each .py (m04/m04d/m05/m05b/m05c/m06/m08/m08b/m09a/m09b/m09c/m10/m11) registers `--cache-policy {1,2}` via `utils.cache_policy.add_cache_policy_arg()` AND prompts the user via `input("[1=keep / 2=recompute] (Enter=1): ")` if the flag wasn't supplied (see `resolve_cache_policy_interactive` pattern below). `1=keep` (default, Enter) preserves all caches with a log line; `2=recompute` authorizes destruction through `guarded_delete(path, policy, label)`. No `.py` deletes anything without the flag reaching `2` — **ONE sanctioned exception**: `clear_anchors_on_completion()` deletes a FINISHED arm's resume anchors (`*_ckpt_{latest,stage,step}.pt`) UN-GATED on success. Rationale: resume anchors are single-arm training SCRATCH, not the durable cross-module CACHES this guard exists to protect (frozen embeddings, factor sets). Safe by construction — it refuses unless `student_encoder.pt` + `*ckpt_best*.pt` already exist (so a parked/interrupted arm keeps every anchor) and only ever removes latest/stage/step, never best/student.
- **Shell contract**: shells stay THIN. Wrappers (`scripts/*.sh`) do NOT prompt — they just forward CLI args to the .py. Overnight/tmux runs pass `--cache-policy 1` (or `2`) explicitly to bypass the .py's `input()` prompt; or set `CACHE_POLICY_ALL=1|2` env var which each .py honors before prompting.
- **Retiring files**: `mv` to sibling `legacy/` subdir — never `rm`.

# GPU PIPELINE CHECKLIST
Every `src/m*.py` using GPU MUST have: (1) `check_gpu()`, (2) `cleanup_temp()`, (3) `add_cache_policy_arg()` + interactive `input()` prompt for cache-policy, (4) auto-batch from VRAM, (5) `save/load_checkpoint()`, (6) `iter_clips_parallel()`, (7) `make_pbar()` with total/rate/ETA, (8) `init_wandb()`+`--no-wandb`, (9) `get_output_dir()`, (10) `guarded_delete()` for any destructive file ops, **(11) `AdaptiveBatchSizer` from `utils.gpu_batch` wired into the forward loop — sub-batch the producer batch, `try/except OutOfMemoryError` → `cuda_cleanup()` + `sizer.on_oom()` → retry; success → `sizer.after_batch_success()`. `memory_cap` ALWAYS reads `pipeline.yaml gpu_memory_target` (universal — never hardcoded). `initial_size` reads per-module `inference_*_initial_bs` from yaml. `max_size` reads the module's existing BS cap. Per-clip-session scripts like m10 instead empty_cache() every N clips for fragmentation hygiene. Training scripts (m09): per-epoch gc + empty_cache + ipc_collect; full mid-step sizer is research-coupled (changes optimizer dynamics) — log as deferred. See errors_N_fixes.md #46/#47.** CPU scripts (m06c/m08): skip GPU-specific items. **tqdm MANDATORY in EVERY `src/m*.py`** (not just GPU scripts).

# TRAINING RULES
- **Epoch-based, not step-based.** `max_epochs` from YAML per mode. Steps = `n_train // batch_size`.
- **LR warmup capped at 10%** of total steps. Predictor LR 1x (gold standard audit).
- **Crash-safe JSONL logging** with `os.fsync()` every write. CSV for backward compat only.
- **Checkpoint management**: export `student_encoder.pt` (~3.8GB encoder, eval Stages 2-7) + `*_ckpt_best.pt` (~8GB encoder+predictor, Stage-8 future-MSE/JEPA prediction). **Clean intermediates ON SUCCESS** — every m09 trainer calls `clear_anchors_on_completion(output_dir)` (`utils/clear_resume_anchors`) right after the final `export_student_for_eval` → deletes the resume anchors `*_ckpt_{latest,stage,step}.pt` (~15-50 GB/arm of pure scratch), keeping student + best. Without it they persist (iter18: 319 GB across 13 arms). Periodic saves use `full=False` (~8GB, no optimizer); `keep_last_n` (YAML) rotates `*_ckpt_step*` mid-run.
- **Per-lambda encoder paths** for ablation. Dynamic fallback in `get_encoder_info()`.
- **No V-JEPA deduplication** — circular reasoning. Hard mode ±30s exclusion is metadata-based.
- Use `--model-config` + `--train-config`. Model configs = architecture. Train configs = technique.
- **Factor streaming (iter9+)**: m09c has two factor-data paths that coexist. Legacy = `FactorSampler` + `load_factor_clip` reading m11 D_L/D_A `.npy` files (~340 GB @ 10K); streaming = `StreamingFactorDataset(IterableDataset)` + `DataLoader(num_workers, persistent_workers, prefetch_factor)` generating D_L/D_A on-demand from `(raw_mp4, mask.npz)` via `utils.factor_streaming.stream_factor` (~40 GB @ 10K). Mode-gated in `surgery_base.yaml > factor_streaming: {sanity,poc,full}`; CLI override `--factor-streaming` / `--no-factor-streaming` wins. Bitwise parity verified against legacy disk output — see `scripts/tests_streaming/test_parity.py`. Under streaming, `m11 --streaming` short-circuits all non-verify clips (manifest-only, no MP4 decode, no scipy blur, no np.save) → ~90% m11 wall-time reduction. `plot_factor_per_clip` keeps reading .npy → m11 still materializes the 100 `select_verify_clips(seed=42)` regardless.

# TESTING & VALIDATION
- **EDIT `src/` + `scripts/` ONLY via the Edit/Write tool — NEVER via Bash** (`python … .read_text()/.replace()/.write_text()`, `sed -i`, `echo >>`, `cat > file`, `perl -pi`). Bash-route edits (a) render **no inline diff card** for the user to review at edit-time, and (b) **bypass the `post-edit-lint.sh` hook** → no auto py_compile/ruff fires, so a syntax/F-error slips silently into a GPU run. Even a many-site mechanical change (e.g. flipping `full=True` across 5 trainers) MUST be N separate `Edit` calls (or one `replace_all`), not a shell loop. The user reviews via `git diff`, but the live diff card + the lint gate are the in-loop safety net — don't skip them for speed. 2026-06-14 incident: patched `full_resume_anchor` + the resume-guard across 5 trainers via `python str.replace` in Bash; the user couldn't see the change (no diff card) and the lint hook never ran.
- **3-check gate after ANY edit to `src/**/*.py`**: (1) `py_compile`, (2) `ast.parse`, (3) `ruff check --select F,E9`. Auto-enforced by `post-edit-lint.sh` hook.
- **Smallest-SANITY smoke after each code mod (when GPU available)**: after every edit touching `src/m*.py` or `src/utils/*.py` reachable from a GPU pipeline (m04/m04d/m05/m06/m09a/m09c/m10/m11, probe_*.py, frozen_features, gpu_batch, etc.), run the smallest possible `--SANITY` invocation of the affected stage on the local Pro 4000 24 GB **before marking the task complete**. Catches what 3-check misses: missing yaml keys (FAIL LOUD asserts), broken CLI wiring (`getattr` defaults silently `None`-propagating), dtype/shape mismatches, CUDA OOM on the new code path, downstream regressions in callers. Define "smallest" per-stage in the plan's verification block (e.g., iter16 M1's smoke was `probe_action.py --FULL --stage labels` on eval_10k; M2's was `yaml_extract.py "$cfg" optimization.max_epochs.${mode}` resolution across 5 train configs). Run-mode order: SANITY → POC → FULL — never propose POC/FULL until SANITY is green. If GPU is genuinely unavailable (Mac, CPU-only box, cloud GPU not provisioned), substitute the CPU-side import + helper-test (`python -c "from utils.X import Y; print(Y(...))"`) and explicitly note the substitution in the completion report. **NEVER skip both** — silent code changes accumulate into the iter11 v1→v10 cycle trap.
- **End-to-end REPL test** before restarting pipelines. Test FULL code path with real data, not just import.
- **Trace data flow** after adding CLI flags: flag → argparse → `get_output_dir()` → correct directory. `shellcheck scripts/*.sh`.
- SANITY validates code correctness (no crashes), NOT model performance. Never draw conclusions from insufficient data.
- **POC ↔ FULL parity (mandatory)**: POC must be a byte-identical scaled-down copy of FULL. The ONLY legitimate POC vs FULL deltas are `poc_total_clips` (subset size) and `max_epochs.poc` (epoch budget). Every other yaml/CLI flag — motion_aux on/off, multi_task on/off, label `n_classes`, head dim, optimizer, EMA τ, warmup type, augmentations — MUST match between POC and FULL. If POC produces degenerate inputs (missing classes, sparse labels, undersized samples), **fix the POC sampler** so its output schema matches FULL — never propose "disable feature X at POC, re-enable at FULL". 2026-05-09 incident: I recommended skipping motion_aux at POC because 855-clip / 7-class POC labels were degenerate; user invoked this rule — the right fix was to repair the POC label sampler to guarantee 8-class stratified output, not to branch the recipe.

# WORKFLOW RULES

## GOAL OVERRIDE
#1 priority is research results. Every recommendation must maximize "Surgery outperform Frozen on Downstream Eval metrics with non-overlapping 95 % CIs. Never filter by implementation effort. Never take shortcut at the cost of compromising results.

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
- **NO bogus / gamble solutions.** BEFORE running any tool call that changes state (Edit, Write, Bash mutations like `sed -i`/`cp`/`mv`/`rm`, HF push, process kill, git ops), AND before recommending an action to the user, list **≥ 3 pros AND ≥ 3 cons** for **every option considered** (minimum 2 options). 
- "Do nothing / wait" counts as an option and must have its own pros/cons. Never skip this gate for "fast / small / obvious" fixes — the 2026-04-20 `sed -i` on a running bash script looked obvious, silently failed, and burned 4 h of overnight GPU. 
- Rationale: every tool call is a commitment with hidden cost surface (inode races, process state, FD caching, silent no-ops). 
- If you cannot articulate 3 cons for your chosen option, you haven't thought hard enough to act. If the user asks you to act without providing this analysis, write it out first, THEN wait for explicit "go" — do NOT silently execute.

## NO DEFER, NO TECH DEBT
**Fix at the right layer, now.** Language like "defer to iter10", "harmless, minor cost", "not worth fixing now", "revisit later", "acceptable at this scale" is BANNED framing. If a mismatch exists (stale yaml vs code, unused output generation, unchecked config flag, misaligned paths, silent no-op), it gets fixed **in this session** at the layer where it originates — not patched around, not added as a "backlog item", not shrugged off as negligible. Scalability rationale: 1.1 GB of unused D_I at 10K becomes ~5 GB at 50K and ~12 GB at 115K — same line of code, linearly worse over the scale ladder. A single deferred fix compounds: 3 deferrals and the codebase is a minefield of "harmless at current scale" traps that each cost a multi-hour incident when the next scale lights them up. When tempted to say "defer", instead: (a) identify the minimum surgical fix (usually one `if cfg[flag]:` guard, one yaml key, or one caller update), (b) apply it at the originating layer, (c) verify downstream callers still work. If the fix is genuinely out-of-scope (e.g., requires 500 LoC of new infrastructure), say so explicitly with LoC estimate and blocked work — do NOT hide it under a "minor" or "harmless" adjective. Every "it's fine, let's just ship it" framing is a future incident postponed, not avoided.

## NO RATIONALIZING — FIX INFRA/OPS DISCREPANCIES, DON'T EXPLAIN THEM
**When the user flags an infra/ops asymmetry between code paths that should be symmetric (sibling modules, paired training/eval scripts, mode-gated branches, mirrored yaml configs, paired-comparison cells, etc.), DO NOT write a "why this is actually fine" paragraph. JUST FIX IT.** Rationalizing an observed asymmetry as a deliberate design choice gaslights the user about their own valid observation; if they noticed it and bothered to point it out, the symmetry expectation IS the requirement, not a misunderstanding to be educated away. **Banned response shape (recognize and abort if I catch myself writing this):** "Why the asymmetry: [path A] does [heavy/expensive work] so frequent [observability/checkpointing/validation] is useful. [path B] does [light/cheap work] so end-of-cycle [observability/checkpointing/validation] is sufficient." That's wrapping a deficiency in a beautiful bag. The fix is almost always trivial — make the second path read the same config field as the first. **Banned phrases — any one of these is a tell I'm about to defend instead of fix:** "by design", "is sufficient", "is cheap so we don't need to", "the asymmetry is fine because", "different operations on different graphs", "X-only's structure differs so Y is N/A", "is intentional", "moot for [path B]", "trajectory is monotonic so we don't need", "redundant for [cheaper path]". **Required behavior:** (a) Identify the symmetric pair and the exact line where they diverge (e.g., "path A reads `cfg[...][shared_field]` at line N; path B hardcodes the value at line M"). (b) Make the second path read the same field — single source of truth, no duplicated literals. (c) Verify with a smoke test that both paths now produce identical schema/cadence/output-file-set/exit-codes. (d) Report what changed in the response — NOT why the original asymmetry "made sense". The asymmetry-rationale belongs in the commit message footnote, NOT in the chat reply. **Escalation signal:** if the user uses the words "defending", "wrapping in a beautiful bag", "rationalizing", "excuses", "shit", "stop explaining", "just fix it", or any equivalent frustration trigger, assume every prior asymmetry-rationalization in this session was also wrong. Re-audit the full diff between every claimed-symmetric pair (configs, output dirs, plot sets, JSONL schemas, log fields, cleanup hooks, cli flags, env-var handling, etc.) and fix the bundle in one pass — not one item at a time across multiple turns. **TRUE-impossibility carve-out (rare):** if a path PHYSICALLY cannot match the other (the action itself doesn't exist, not "it's cheap so we skipped it"), state the technical constraint in ONE LINE, then propose the closest-possible-match fix anyway (e.g., emit a placeholder file, log the no-op explicitly, expose the same CLI flag with a documented N/A behavior). Never use the carve-out for performance or convenience — only for genuine architectural impossibility.

## GIT
Never run raw `git commit/push/add/reset/...` (enforced by hook). Claude runs `echo 1 | bash git_push.sh "<plain-text msg>"` itself after finished work, WITHOUT asking first (account 1 = Kapil; user, 2026-11-26: "keep pushing everything, no need to ask my permission"). Before every such push: `git status` + sizes (GitHub rejects >100 MB files), confirm `.env` is ignored, grep the new/changed text files for secrets and personal e-mails (co-author e-mails live in gitignored `overleaf/author_contacts.local.md`, never in tracked files; the repo is public). Repos outside this folder (e.g. `../cita_ecliptica`) are not writable from the sandbox: hand the user the command instead.

## EVIDENCE FOR HARDCODED-VALUE AUDITS
When auditing for hardcoded values, SHOW grep output as proof. User does not trust "I checked" claims without evidence.

## WANDB
Shared `utils/wandb_utils.py`. `--no-wandb` on every module. All functions no-op when `run=None`.

## SESSION-END SYNC
**Update CLAUDE.md + MEMORY.md** at end of every session with new results/pivots/decisions. Sync `src/MEMORY.md` → `~/.claude/projects/.../memory/MEMORY.md`.

## INSTANCE TEARDOWN — 3 BACKUPS BEFORE KILL/RECYCLE/DESTROY (NON-NEGOTIABLE)
**This is a Vast.ai container — `recycle`/`destroy` WIPES the whole filesystem** (`/workspace` is NOT a persistent volume here — `workspace_is_volume:false`; even `stop` still bills storage). Nothing on the box survives unless it is pushed off-box first. **Before the user closes / stops / kills / recycles / destroys the instance (or its disk), CLAUDE must remind the user to run ALL THREE** and only then declare the box safe to tear down. The user runs them (never me — GPU/secret commands are the user's). Each is independently resumable, so a half-finished one just re-runs.
1. **CODE → GitHub** (THE script — prompts for Kapil/Gaytri account):
   ```bash
   bash git_push.sh "<commit message>"
   ```
2. **OUTPUTS → HF** (resumable large-folder; `--exclude "**/.*"` keeps `.env`/secrets/`.cache` out; re-run to top-up):
   ```bash
   hf upload-large-folder anonymousML123/factorjepa-outputs . --repo-type dataset \
     --include "outputs/poc/**" --exclude "**/.*" \
     2>&1 | tee logs/upload_large_folder_outputs_poc_$(date +%Y%m%d_%H%M%S).log
   ```
   Verify the tail shows `committed N/N` + `Upload is complete!` (N = candidate count) before trusting it.
3. **SESSIONS + MEMORY → Mac** (run ON THE MAC, not the instance — pulls the `.claude` transcripts + `memory/*.md`):
   ```bash
   bash claude_session.sh --download --host <vast_host_alias>
   ```
**Teardown gate:** the box is safe to kill ONLY after all 3 succeed — code reconstructable from GitHub, outputs from HF, sessions/memory on the Mac. Regenerable caches (`*/m12_frame_cache`, `outputs/poc/**/*.pt` once mirrored on HF) do NOT need backup — they rebuild. NEVER bake `${WORKSPACE}/.env` (live HF/OpenAI/GitHub/WANDB secrets) into any uploaded/public artifact — the `--exclude "**/.*"` above is what keeps it out.

## HF PUSH/PULL — COMPLETE ONLY (never partial)
Always `download-data <dir>` and `upload-large-folder … --include "outputs/poc/**"` the FULL tree — NEVER `--ext`/narrow filters. A partial local ≠ HF hides stale/orphaned remote files undetectably + makes `verify` FAIL. Only sanctioned exclude: `--exclude "**/.*"` (regenerable scratch dotfiles). 2026-06-21 incident.

## OUTPUT FORMATTING — TABLES, NOT LISTS  (with emojis for scannability)
**Default to ASCII box-drawing tables** (`┌─┬─┐` `│` `├─┼─┤` `└─┴─┘`) for ANY comparison spanning ≥2 columns × ≥2 rows. Markdown pipe tables (`| col | col |`) and bullet lists are BANNED for comparison data — Claude Code's CommonMark renderer flattens them into unaligned single-column "lists" in the user's terminal, which breaks the comparison and frustrates the user. **Use emojis LIBERALLY** in column headers, row identifiers, and status/marker columns for visual scannability — the user explicitly chose "eyeballable with emojis" over "perfectly aligned plain ASCII". Keep emojis OUT of pure-numeric cells (where width drift matters most) but in headers / status / verdict columns they're encouraged. Box-drawing borders MUST still be present so the table structure is visible even if cell widths drift slightly with emoji rendering. Always declare a marker legend below the table. POC ablation sweeps (Cell A/B/C/D × N metrics) MUST be a single box-drawn grid, not N grouped lists.

## SPELL OUT METRIC NAMES IN PROSE — NO SHORT KEYS / UNDEFINED JARGON
The eval suite is **14 metrics**, so the short json/code keys (`act` `tax` `mcos` `fut` `rollout` `causal` `tdist` `teacher_free` `maskratio` `order` `aot` `tov` `pace` `tcc_tau` `tcc_cycle`) are UNREADABLE mixed into a sentence — the user cannot tell them apart at a glance, and they've explicitly asked to stop using them. When EXPLAINING results to the user (chat replies + result/plan `.md` docs), ALWAYS write the **FULL human metric name** — e.g. "Action top-1 accuracy", "taxonomy F1", "motion-cosine separation", "future-frame MSE", "rollout drift slope", "causal future-block L1", "L1-vs-Δt decay", "Arrow-of-Time accuracy", "temporal-order (frame-permutation) accuracy", "playback-pace accuracy", "TCC Kendall τ", "TCC cycle-back error", "frame-order sensitivity". The canonical full names are the `_CATALOG` ylabels/descriptions in `src/m13_eval_plot.py` (the single metric registry — use those, don't coin variants). The short key may appear ONCE in parentheses for a csv/json lookup, NEVER as the standalone reference. Same rule for undefined ML jargon (`Pareto`, `BCa`, `co-leader`, a bare `Δ`): say it in plain English ("a trade-off win — better on X without losing Y", "95% confidence band") or define it on first use. Short keys live in code/csv/json ONLY.

**GLOSSARY — the 15 eval keys → the full name to use in prose** (↑/↓ = higher/lower is better; group = the umbrella to use instead of vague words like "coherence"):
```text
key           full name (USE THIS in prose)              dir  group              what it measures (plain)
──────────────────────────────────────────────────────────────────────────────────────────────────────────
act           Action top-1 accuracy                      ↑    head/probe         classify the motion class from frozen features
tax           taxonomy F1                                 ↑    head/probe         linear read-off of VLM scene attributes
mcos          motion-cosine separation                   ↑    head/probe         same-motion clips sit closer than different-motion
fut           future-frame MSE                            ↓    predictor          JEPA predictor's L1 on the masked next frame
rollout       rollout drift slope                         ↓    predictor          how fast multi-step prediction error snowballs
causal        causal future-block L1                      ↓    predictor          predict the 2nd half of a clip from the 1st
tdist         L1-vs-Δt decay slope                        ↓    predictor          accuracy decay as it predicts further ahead
teacher_free  free-running exposure-bias gap              ↓    predictor          worse on its OWN predictions vs fed the truth
maskratio     mask-ratio robustness slope                ↓    predictor          graceful degradation as more video is hidden
order         frame-order sensitivity                     —    predictor          shuffled−ordered ΔL1 (does it actually use time?)
aot           Arrow-of-Time accuracy                      ↑    encoder-temporal   tell forwards from backwards playback
tov           temporal-order (frame-permutation) acc      ↑    encoder-temporal   put shuffled frames back in order
pace          playback-pace accuracy                      ↑    encoder-temporal   tell normal 1× from 2×/4× fast-forward
tcc_tau       TCC Kendall τ                               ↑    encoder-temporal   matching moments line up across two clips
tcc_cycle     TCC cycle-back error                        ↓    encoder-temporal   frame → its match → back: how far it lands off
```
Group umbrellas (use the group name, never invent one like "coherence"): **head/probe** = linear read-off of the frozen features · **predictor** = the JEPA predictor's quality · **encoder-temporal** = does the encoder preserve frame order / timing.

# HOOKS
- `enforce-dev-rules.sh` (PreToolUse:Bash) — blocks pip install, git state changes, bare `python3` without venv activation
- `post-edit-lint.sh` (PostToolUse:Edit,Write) — auto py_compile + ruff on src/**/*.py
- `fail-hard-research.sh` (PreToolUse:Edit,Write) — blocks `|| continue`, `|| true`, bare `except: pass`

# CONFIGS
- `configs/pipeline.yaml` — clip limits, streaming, GPU defaults, eval params, encoder registry
- `configs/arm_registry.yaml` — **SINGLE SOURCE for the arm roster** (arm→encoder→m09-dir + kind/group). Add a new arm = ONE entry here. Read by `run_train.sh`/`run_eval.sh`/`iter18_poc_ngpu.py` + the plot scripts via `src/utils/arm_registry.py` (python import + bash CLI). Replaced 8 re-typed lists that caused 6 whack-a-mole FATALs on 2026-06-14.
- `configs/model/vjepa2_1.yaml` — PRIMARY (V-JEPA 2.1 ViT-G 2B, 1664-dim)
- `configs/model/vjepa2_0.yaml` — legacy (V-JEPA 2.0 ViT-g 1B, 1408-dim)
- `configs/train/base_optimization.yaml` — shared: masking, augmentation, AdamW, EMA, mixed precision
- `configs/legacy2/ch10_pretrain.yaml` — continual pretraining (drift control, lambda sweep)
- `configs/legacy2/explora.yaml` — ExPLoRA (LoRA rank=16 + unfreeze 2 blocks)
- `configs/train/surgery_base.yaml` — factor surgery base (3-stage progressive unfreeze D_L→D_A→D_I, SAM3 params, early-stop triggers, `factor_streaming` block); per-arm `surgery_3stage_DI_*.yaml` / `surgery_2stage_noDI_*.yaml` extend it

# 📚 REFERENCE
- 🏗️ Roadmap: `iter/iter18_ablations_FTtechniues/plan_baselines_roadmap.md` (HIGH level — 13 FT-ablation arms + the 5 improvement arms)
- 📋 Improvement plan + status: `iter/iter18_ablations_FTtechniues/plan_outperform_FT.md` (MID level — each new arm → the eval gap it targets)
- 🚀 Runbook: `iter/iter18_ablations_FTtechniues/runbook.md` (LOW level — GPU-ready commands + verify tables)
- 🐛 Bug log: `iter/utils/high_level_errors_N_fixes.md` (ERROR level — ~#82 as of 2026-06; the old `iter/iter8/errors_N_fixes.md` was consolidated here)
- 🛡️ Preflight CPU-side guards: `.claude/skills/preflight/SKILL.md` (B1-B42 static checks citing errors_N_fixes entries)
- 📖 Fallback techniques if Surgery fails: `iter/utils/literarure_survey.md` (24 JEPA variants, SIGReg / VLA-JEPA / temporal-projection)
