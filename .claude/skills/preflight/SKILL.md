---
name: preflight
description: Pipeline code review checklist + automated checks (py_compile, AST, ruff) + call pattern + SIGSEGV detection. Run before any GPU job.
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash
argument-hint: [file-path or module-name]
---

# Pipeline Preflight Checklist

Review the specified file against these MANDATORY requirements.

## Part A: Automated Checks (run ALL 3)

**A1. py_compile:** `source venv_walkindia/bin/activate && python3 -m py_compile <file>`
**A2. AST check:** `main()` exists, `--SANITY`/`--FULL` in argparse, tqdm imported (GPU scripts), no orphan functions
**A3. ruff:** `ruff check --select F821,F841,F811 <file>` — undefined names, unused vars, redefined names

## Part B: Manual Checklist

**B1. tqdm:** Loop has `tqdm(desc=, unit=)`. Resume uses `initial=len(completed)`.
**B2. Checkpoint:** Periodic saves, load+skip on startup, output-exists guard BEFORE model loading.
**B3. Tee logging:** Docstring has `python -u src/*.py --args 2>&1 | tee logs/<name>.log`
**B4. wandb:** `add_wandb_args`, `init_wandb`, `log_metrics`, `finish_wandb`. No-op when `run=None`.
**B5. GPU fail-loud:** `check_gpu()` early. No silent CPU fallback. No bare `except: pass`.
**B6. Fail-hard:** No `.get(key, default)` on YAML configs. Shape/dim validation. NaN checks.
**B7. Reproducibility:** Seeds set (training only). `do_sample=False` (m04 only).

**B8. Call pattern validation** — grep for known footguns:
```bash
grep -n "for .* in iter_clips_parallel" src/m*.py   # MUST return 0 matches
```
Known mismatches:
- `iter_clips_parallel()` returns `(queue, event, thread)` not an iterable. MUST unpack + `clip_q.get(timeout=N)`.
- `verify_or_skip()` returns bool — caller must `if verify_or_skip(...): return`
- `get_output_dir()` returns Path — no raw string concat

**B9. Silent crash (SIGSEGV) detection** — C-extensions that segfault bypass `try/except`:
```bash
# Smoke test decode pipeline with real data (no GPU needed, catches SIGSEGV):
timeout 10 python3 -c "
import sys; sys.path.insert(0, 'src')
from utils.data_download import iter_clips_parallel
from utils.video_io import decode_video_bytes
import tempfile
q, s, _ = iter_clips_parallel('data/val_1k_local')
k, b = q.get(timeout=5)
with tempfile.TemporaryDirectory() as d:
    t = decode_video_bytes(b, d, k, 16)
    print(f'OK: {t.shape}') if t is not None else print('FAIL')
s.set()
"
# Exit 139 = SIGSEGV. Timeout = hang. Fix: disable broken backend in video_io.py.
```

## Output Format

```
=== PREFLIGHT: <filename> ===
AUTOMATED:  [A1] PASS/FAIL  [A2] PASS/FAIL  [A3] PASS/FAIL
MANUAL:     [B1-B9] PASS/FAIL/N/A each
TOTAL: X/12 passed. Y FAILs need fixing.
```

List all FAILs with line numbers and fix instructions.
