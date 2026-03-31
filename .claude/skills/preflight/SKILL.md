---
name: preflight
description: Pipeline code review checklist + 3 automated checks (py_compile, AST, ruff). Run before any GPU job. Missing any = broken code.
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash
argument-hint: [file-path or module-name]
---

# Pipeline Preflight Checklist

Review the specified file (or recently modified pipeline files) against these MANDATORY requirements.

## Part A: Automated Checks (run ALL 3, report PASS/FAIL)

### A1. py_compile (syntax)
Run: `source venv_walkindia/bin/activate && python3 -m py_compile <file>`
- PASS if exit code 0
- FAIL if any syntax error — fix before proceeding

### A2. AST structural check
Run via inline python:
- [ ] `main()` function exists
- [ ] `--SANITY` and/or `--FULL` in argparse
- [ ] `from tqdm import tqdm` present (GPU scripts m04-m07, m09 only)
- [ ] `validate_tag_fields` defined AND called (m04 only)
- [ ] No orphan functions (defined but never called anywhere in file)

### A3. ruff / pyflakes (undefined variables)
Run: `python3 -m ruff check --select F821,F841,F811 <file>`
If ruff not installed, use: `python3 -m pyflakes <file>`
If neither available, manually check:
- [ ] No undefined variable names (F821)
- [ ] No unused variables (F841)
- [ ] No redefined unused names (F811)
- Search for new functions/variables added — verify each is used

## Part B: Manual Checklist (verify each, report PASS/FAIL/N/A)

### B1. tqdm progress bar
- [ ] Loop wrapped in `tqdm(...)` with `desc=`, `unit=`
- [ ] If resuming, tqdm uses `initial=len(completed)` and `total=len(all_items)`

### B2. Auto-resume / checkpoint
- [ ] Completed items saved to disk periodically
- [ ] On startup, completed items loaded and skipped
- [ ] **OUTPUT-EXISTS GUARD**: at TOP of main(), BEFORE model loading, check if final output exists → skip entirely

### B3. Tee logging
- [ ] Docstring includes `python -u src/*.py --args 2>&1 | tee logs/<log_name>.log`

### B4. wandb integration
- [ ] `add_wandb_args(parser)`, `init_wandb()`, `log_metrics()`, `finish_wandb()`
- [ ] All wandb functions no-op when `run=None`

### B5. GPU fail-loud (m04/m05/m06/m07/m09 only)
- [ ] `check_gpu()` called early
- [ ] No silent CPU fallback
- [ ] No bare `except Exception: pass`

### B6. Fail-hard validation
- [ ] No `.get(key, default)` for config keys that MUST exist
- [ ] `strict=False` in load_state_dict has param count validation (>90%)
- [ ] Input shapes/dims validated before computation
- [ ] NaN check on model outputs

### B7. Reproducibility
- [ ] Random seeds set (random, numpy, torch) — training scripts only
- [ ] `do_sample=False` for VLM inference — m04 only

## Output Format

```
=== PREFLIGHT: <filename> ===

AUTOMATED:
  [A1] py_compile:  PASS/FAIL
  [A2] AST check:   PASS/FAIL (details)
  [A3] ruff/undef:  PASS/FAIL (details)

MANUAL:
  [B1] tqdm:        PASS/FAIL/N/A
  [B2] checkpoint:  PASS/FAIL/N/A
  [B3] tee logging: PASS/FAIL/N/A
  [B4] wandb:       PASS/FAIL/N/A
  [B5] GPU fail:    PASS/FAIL/N/A
  [B6] Fail-hard:   PASS/FAIL/N/A
  [B7] Repro seeds: PASS/FAIL/N/A

TOTAL: X/10 passed. Y FAILs need fixing.
```

List all FAILs with line numbers and fix instructions.
