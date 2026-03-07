---
name: preflight
description: Pipeline code review checklist. Use before running any new pipeline that makes API/VLM/LLM calls. Ensures checkpoint/resume, tee logging, and fail-fast validation are present. Missing any = broken code.
disable-model-invocation: true
allowed-tools: Read, Grep, Glob
argument-hint: [file-path or module-name]
---

# Pipeline Preflight Checklist

Review the specified file (or recently modified pipeline files) against these MANDATORY requirements.
Every pipeline loop that makes GPU/API/VLM calls MUST have ALL of these. Missing any = broken code.

## Checklist

For each file containing a GPU inference or API call loop, verify:

### 1. tqdm progress bar
- [ ] Loop is wrapped in `tqdm(...)` with `desc=`, `unit=` parameters
- [ ] If resuming, tqdm uses `initial=len(completed)` and `total=len(all_items)`
- Search for: `from tqdm import tqdm` and `tqdm(` in the file

### 2. Auto-resume / checkpoint
- [ ] Completed items are saved to disk (JSONL) after EACH iteration, with `f.flush()`
- [ ] On startup, completed items are loaded and skipped
- [ ] `--fresh` flag exists to clear checkpoints
- Search for: `checkpoint`, `completed`, `f.flush()`, `"a"` (append mode)

### 3. Tee logging
- [ ] Output goes to both terminal AND a log file
- [ ] Command format: `python -u src/*.py --args 2>&1 | tee logs/<log_name>.log`
- [ ] Docstring includes the exact terminal command with tee

### 4. wandb integration
- [ ] `add_wandb_args(parser)` adds `--no-wandb` flag
- [ ] `init_wandb(module, mode, config, enabled)` called after arg parse
- [ ] `log_metrics()` / `log_image()` / `log_artifact()` at key points
- [ ] `finish_wandb(run)` in finally block
- [ ] All wandb functions no-op when `run=None`
- Search for: `from src.utils.wandb_utils import` and `--no-wandb`

### 5. GPU fail-loud (m04/m05/m06/m07 only)
- [ ] Script calls `check_gpu()` or `torch.cuda.is_available()` early
- [ ] Exits with clear error if no GPU — NEVER silently falls back to CPU
- [ ] No `import faiss` without `faiss.StandardGpuResources()` (use GPU FAISS only)
- [ ] No `from sklearn` for iterative algorithms — use cuML instead
- Search for: `check_gpu`, `cuda`, `faiss`, `sklearn`, `cuml`

### 6. Auto batch sizing (m04/m05 only)
- [ ] Uses `compute_batch_sizes()` or `AdaptiveBatchSizer` from `src/utils/gpu_batch.py`
- [ ] `--gpu-mem` arg to override VRAM detection
- [ ] `--batch-size` arg to override computed batch size
- Search for: `gpu_batch`, `AdaptiveBatchSizer`, `--gpu-mem`, `--batch-size`

### 7. Dynamic prints only
- [ ] No static/hardcoded print statements that lie about runtime values
- [ ] All prints use f-strings or .format() with actual computed values
- [ ] Progress messages reflect real counts, not placeholder numbers
- Search for: `print(` — verify each is dynamic

## Output Format

For each check, report:
- PASS: requirement met (with line number evidence)
- FAIL: requirement missing (with what needs to be added)
- N/A: not applicable to this module type

Summarize: X/7 checks passed. List all FAILs with fix instructions.
