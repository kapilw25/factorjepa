---
name: preflight
description: Pipeline code review checklist + automated checks (py_compile, AST, ruff) + call pattern + SIGSEGV detection + SAM3/utils-specific regression guards derived from iter/iter8/errors_N_fixes.md. Catches known failure patterns BEFORE GPU run.
disable-model-invocation: true
allowed-tools: Read, Grep, Glob, Bash
argument-hint: [file-path or module-name]
---

# Pipeline Preflight Checklist

Review the specified file against these MANDATORY requirements. Every check is designed to be runnable on CPU without GPU access. Checks B10-B14 are regression guards derived directly from `iter/iter8/errors_N_fixes.md` — each one maps to a specific error we've already been bitten by.

## Part A: Automated Checks (run ALL 3)

**A1. py_compile:** `source venv_walkindia/bin/activate && python3 -m py_compile <file>`

**A2. AST check:** `main()` exists, `--SANITY`/`--FULL` in argparse, tqdm imported (GPU scripts), no orphan functions.
```bash
source venv_walkindia/bin/activate && python3 -c "
import ast, sys
src = open('<file>').read()
tree = ast.parse(src)
funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
assert 'main' in funcs, 'A2 FAIL: no main()'
for tok in ['--SANITY','--FULL']:
    assert tok in src, f'A2 FAIL: argparse missing {tok}'
print('A2 PASS')
"
```

**A3. ruff:** `ruff check --select F821,F841,F811 <file>` — undefined names, unused vars, redefined names.

## Part B: Manual Checklist (B1-B9 generic + B10-B14 regression guards)

**B1. tqdm:** Loop has `tqdm(desc=, unit=)` or `make_pbar(...)`. Resume uses `initial=len(completed)`.

**B2. Checkpoint:** Periodic saves, load+skip on startup, `verify_or_skip(...)` BEFORE model loading.

**B3. Tee logging:** Docstring has `python -u src/*.py --args 2>&1 | tee logs/<name>.log`

**B4. wandb:** `add_wandb_args`, `init_wandb`, `log_metrics`, `finish_wandb`. No-op when `run=None`.

**B5. GPU fail-loud:** `check_gpu()` early. No silent CPU fallback. No bare `except: pass`.

**B6. Fail-hard:** No `.get(key, default)` on YAML config dicts (`factor_cfg`, `train_cfg`, `model_cfg`, `dino_cfg`, `interaction_cfg`). Shape/dim validation. NaN checks.
```bash
grep -nE "\b(factor_cfg|train_cfg|model_cfg|dino_cfg|interaction_cfg|cfg\[.*\])\.get\(" <file>
# MUST return 0 matches. `.get()` on tags.json data or queue.Queue is fine.
```

**B7. Reproducibility:** Seeds set (training only). `do_sample=False` (m04 only).

**B8. Call-pattern footguns:**
```bash
# iter_clips_parallel returns (queue, event, thread), NOT iterable:
grep -nE "for\s+\w+\s+in\s+iter_clips_parallel" <file>   # MUST return 0 matches
# correct: clip_q, tar_stop, _reader = iter_clips_parallel(...); clip_q.get(timeout=N)

# verify_or_skip returns bool — must be in `if` condition:
grep -nE "^\s*verify_or_skip\(" <file>   # MUST return 0 (should be `if verify_or_skip(...)`)
```
Known mismatches:
- `iter_clips_parallel()` returns `(queue, event, thread)` not an iterable. MUST unpack + `clip_q.get(timeout=N)`.
- `verify_or_skip()` returns bool — caller must `if verify_or_skip(...): return`.
- `get_output_dir()` / `get_module_output_dir()` return Path — no raw string concat.

**B9. Silent crash (SIGSEGV) detection** — C-extensions that segfault bypass `try/except`:
```bash
source venv_walkindia/bin/activate && timeout 30 python3 -c "
import sys, os; sys.path.insert(0, 'src')
from utils.data_download import iter_clips_parallel
from utils.video_io import decode_video_bytes
import tempfile
q, s, _ = iter_clips_parallel('data/val_1k_local')
k, b = q.get(timeout=10)
with tempfile.TemporaryDirectory() as d:
    t = decode_video_bytes(b, d, k, 16)
    print(f'B9 PASS: {t.shape}') if t is not None else print('B9 FAIL: decode returned None')
s.set(); os._exit(0)
" 2>&1 | tail -3
# Exit 139 = SIGSEGV (torchcodec-style). Timeout after 30s = hang. Fix: disable broken backend in video_io.py.
```

---

## Part C: Regression Guards (B10-B14, derived from `iter/iter8/errors_N_fixes.md`)

Each B10-B14 check maps to a specific past failure. Keep these as one-shot bash commands — run them all in a single pass.

**B10. CLI utils import-path guard** (catches error #3: `hf_outputs.py` crashed with `ModuleNotFoundError: No module named 'utils'`):

Any file in `src/` or `src/utils/` that imports `from utils.X import ...` AND has an `if __name__ == "__main__"` block MUST have `sys.path.insert(0, ...)` set to the parent-of-utils directory BEFORE those imports. Otherwise, running as a CLI script blows up.
```bash
python3 -c "
import re, sys
src_text = open('<file>').read()
imports_utils = bool(re.search(r'^\s*from\s+utils\.', src_text, re.M))
has_main_block = '__main__' in src_text
has_syspath = 'sys.path.insert' in src_text
if imports_utils and has_main_block and not has_syspath:
    print('B10 FAIL: imports from utils.* as CLI but missing sys.path.insert. Add: sys.path.insert(0, str(Path(__file__).resolve().parent.parent))')
    sys.exit(1)
print('B10 PASS')
"
```

**B11. SAM3 undeclared-dependency scan** (catches errors #5, #6, #7, #18, #19 — all the same pattern: SAM3 installed `--no-deps` so every transitive runtime import must be in `requirements_gpu.txt`):

If the file imports `sam3`, AST-walk the installed `sam3` package's **unconditional top-level** imports (the ones that execute at `import sam3` time and actually crash if missing). Ignore imports inside `try/except`, function bodies, or `if`-gated blocks — those are optional/guarded and don't block package load. Skip `sam3/train/` and `sam3/agent/` subtrees (off inference path).
```bash
source venv_walkindia/bin/activate && python3 << 'EOF'
import ast, os, re, sys, pathlib
target = '<file>'
src = open(target).read()
if 'from sam3' not in src and 'import sam3' not in src:
    print('B11 SKIP: file does not import sam3'); sys.exit(0)

import sam3
sam3_root = pathlib.Path(sam3.__file__).parent
stdlib_top = set(sys.stdlib_module_names) | {'typing_extensions', 'pkg_resources'}

def top_level_unconditional_imports(py: pathlib.Path) -> set:
    """Collect only imports that execute at module load AND are not guarded by try/except or if-branches."""
    try:
        tree = ast.parse(py.read_text())
    except SyntaxError:
        return set()
    hits = set()
    # Only direct children of Module body — no descent into Try/FunctionDef/ClassDef/If
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                hits.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports (level > 0 means "from . import x" or "from .submod import x")
            if node.level == 0 and node.module:
                hits.add(node.module.split('.')[0])
    return hits

external = set()
off_path_subtrees = {'train', 'agent'}   # training / agent code, not imported by build_sam3_predictor
off_path_dirs = {'tests', 'test', 'benchmarks', 'examples'}  # never imported at package load
for py in sam3_root.rglob('*.py'):
    rel = py.relative_to(sam3_root)
    if rel.parts and rel.parts[0] in off_path_subtrees: continue
    if any(p in off_path_dirs for p in rel.parts): continue
    for top in top_level_unconditional_imports(py):
        if top in stdlib_top or top in {'sam3', '__future__'}: continue
        external.add(top)

declared = set()
for req_file in ['requirements_gpu.txt', 'requirements.txt']:
    if not os.path.exists(req_file): continue
    for line in open(req_file):
        line = line.split('#')[0].strip()
        if not line or line.startswith('-'): continue
        pkg = re.split(r'[<>=!\s]', line)[0].lower().replace('-', '_')
        declared.add(pkg)

# Known-transitive: pulled by things already declared, so safe.
transitive_ok = {
    'torch', 'torchvision', 'numpy', 'PIL', 'matplotlib', 'tqdm', 'pandas', 'cv2',
    'huggingface_hub', 'regex', 'scipy', 'skimage', 'sklearn', 'yaml', 'requests',
    'psutil', 'triton',  # transitive via torch
}
missing = sorted({m for m in external if m.lower().replace('-', '_') not in declared and m not in transitive_ok})
if missing:
    print(f'B11 FAIL: SAM3 unconditional top-level imports not in requirements_gpu.txt: {missing}')
    print('  Add each to requirements_gpu.txt, re-run ./setup_env_uv.sh --gpu --from-wheels')
    sys.exit(1)
print(f'B11 PASS: {len(external)} SAM3 top-level imports, all declared')
EOF
```

**B12. SAM3 Flash-Attention 3 guard** (catches error #9: `ModuleNotFoundError: flash_attn_interface` — SAM 3.1 defaults to FA3 which we don't have):

Any call to `build_sam3_predictor(...)` MUST pass `use_fa3=False`. Our stack ships FA2 (`flash_attn 2.8.3`), not FA3 (`flash_attn_interface`). Uses AST to avoid docstring false positives.
```bash
python3 -c "
import ast, sys
tree = ast.parse(open('<file>').read())
bad_lines = []
n_calls = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'build_sam3_predictor':
        n_calls += 1
        kws = {kw.arg for kw in node.keywords}
        has_false = any(kw.arg == 'use_fa3' and isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in node.keywords)
        if not has_false: bad_lines.append(node.lineno)
if bad_lines:
    print(f'B12 FAIL: build_sam3_predictor() at line(s) {bad_lines} missing use_fa3=False')
    sys.exit(1)
print(f'B12 PASS: {n_calls} build_sam3_predictor call(s), all have use_fa3=False') if n_calls else print('B12 SKIP')
"
```

**B13. SAM3 async-thread exit guard** (catches errors #14 + #16: process hangs after Done because SAM3 spawns async frame-loading threads with no shutdown method):

Any file that imports `sam3` AND has a `def main()` MUST:
- end `main()` with `os._exit(0)` (not `return` or `sys.exit`)
- wrap the `main()` call under `if __name__ == "__main__":` in a `try/except` that calls `os._exit(1)` on crash
```bash
python3 -c "
import re, sys
src = open('<file>').read()
if 'import sam3' not in src and 'from sam3' not in src:
    print('B13 SKIP: no sam3 import')
    sys.exit(0)
ok_exit = 'os._exit(0)' in src
ok_crash = re.search(r'if\s+__name__\s*==\s*[\"\\']__main__[\"\\']:\s*\n\s*try:.*?os\._exit\(1\)', src, re.DOTALL)
if not ok_exit:
    print('B13 FAIL: sam3 is imported but os._exit(0) not called in main() — will hang on async frame-loader threads (error #14)')
    sys.exit(1)
if not ok_crash:
    print('B13 FAIL: sam3 is imported but __main__ missing try/except ... os._exit(1) wrapper — unhandled exceptions hang (error #16)')
    sys.exit(1)
print('B13 PASS: os._exit(0) on success path, os._exit(1) on crash path')
"
```

**B14. SAM3 API unified-entry guard** (catches error #8: `build_sam3_multiplex_video_predictor() got unexpected keyword argument 'gpus_to_use'` — old 3.0 API removed in 3.1):

Any direct call to the old multiplex builder is a bug — SAM 3.1 requires `build_sam3_predictor(version="sam3.1")` unified entry.
```bash
python3 -c "
import re, sys
src = open('<file>').read()
bad = re.findall(r'build_sam3_multiplex_video_predictor\s*\(', src)
if bad:
    print(f'B14 FAIL: {len(bad)} call(s) to deprecated build_sam3_multiplex_video_predictor() — use build_sam3_predictor(version=\"sam3.1\", use_fa3=False)')
    sys.exit(1)
print('B14 PASS')
"
```

**B15. Decode backend sanity** (catches error #10: torchcodec SIGSEGV on TAR-extracted MP4 bytes with PyTorch nightly 2.12+cu128 on Blackwell):

`src/utils/video_io.py` must keep torchcodec disabled until upstream fix — the `_USE_TORCHCODEC` module-level flag MUST be `False`.
```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from utils.video_io import _USE_TORCHCODEC
if _USE_TORCHCODEC:
    print('B15 FAIL: _USE_TORCHCODEC is True — will SIGSEGV on Blackwell sm_120 (error #10). Set to False in src/utils/video_io.py')
    sys.exit(1)
print('B15 PASS: torchcodec disabled, PyAV fallback active')
"
```

---

## Part D: transformers 5.x Regression Guards (B16-B20)

Added 2026-04-14 during transformers 4.57 → 5.5.4 migration. Each maps to an error logged in `iter/iter8/errors_N_fixes.md` (#37, #38, #39, #40). All are AST/grep-based, run in <1 s.

**B16. Deprecated `torch_dtype=` kwarg** (catches error #37 — transformers 5.x renamed to `dtype=`):
```bash
source venv_walkindia/bin/activate && python3 -c "
import ast, sys
src = open('<file>').read()
tree = ast.parse(src)
bad = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'from_pretrained':
        for kw in node.keywords:
            if kw.arg == 'torch_dtype': bad.append(node.lineno)
if bad: print(f'B16 FAIL: deprecated torch_dtype= at line(s) {bad}. Use dtype= (errors_N_fixes.md #37)'); sys.exit(1)
print('B16 PASS')"
```

**B17. DINO must be fp32 under transformers 5.x** (catches error #37 — text-branch crash when DINO is fp16, no auto-cast):
```bash
source venv_walkindia/bin/activate && python3 -c "
import ast, sys
src = open('<file>').read()
tree = ast.parse(src)
bad = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'from_pretrained':
        recv = node.func.value
        if isinstance(recv, ast.Name) and recv.id == 'AutoModelForZeroShotObjectDetection':
            for kw in node.keywords:
                if kw.arg in ('dtype', 'torch_dtype'):
                    v = kw.value
                    if isinstance(v, ast.Attribute) and v.attr in ('float16', 'half', 'bfloat16'):
                        bad.append((node.lineno, v.attr))
if bad: print(f'B17 FAIL: Grounding DINO loaded non-fp32: {bad}. transformers 5.x text branch crashes fp16 (errors_N_fixes.md #37)'); sys.exit(1)
print('B17 PASS')"
```

**B18. Sam3TrackerVideoProcessor box-input depth=3, not 4** (catches error #38):
```bash
python3 -c "
import re, sys
src = open('<file>').read()
bad = [i+1 for i, line in enumerate(src.split(chr(10)))
       if re.search(r'\[\s*\[\s*\[\s*b\s*\]\s+for\s+b\s+in\s+', line)]
if bad: print(f'B18 FAIL: depth-4 box nesting at line(s) {bad}. Sam3TrackerVideoProcessor expects depth-3 [image,box,coords] for input_boxes (depth-4 is for input_points). See errors_N_fixes.md #38'); sys.exit(1)
print('B18 PASS')"
```

**B19. Session-reset methods live on session object, not processor** (catches error #39):
```bash
python3 -c "
import re, sys
src = open('<file>').read()
bad = []
for i, line in enumerate(src.split(chr(10)), 1):
    m = re.search(r'(\w*processor\w*)\.(reset_inference_session|reset_tracking_data|clear_all|remove_point_inputs|remove_mask_inputs)\s*\(', line)
    if m: bad.append((i, m.group(1), m.group(2)))
if bad: print(f'B19 FAIL: session-reset called on processor (should be on session): {bad}. See errors_N_fixes.md #39'); sys.exit(1)
print('B19 PASS')"
```

**B20. Sam3TrackerVideoSegmentationOutput attr names** (catches error #40 — silent bug: legacy SAM2 attr `iou_scores` etc. returns None with getattr-fallback, defeats mask confidence filter):
```bash
source venv_walkindia/bin/activate && python3 << 'PY'
import re, sys
src = open('<file>').read()
bad = []
for i, line in enumerate(src.split('\n'), 1):
    if re.search(r'\boutput\.(iou_scores|mask_logits|out_obj_ids|out_binary_masks|out_probs)\b', line):
        bad.append((i, line.strip()))
    if re.search(r"""getattr\(output,\s*['"]iou_scores['"]""", line):
        bad.append((i, line.strip()))
if bad: print(f'B20 FAIL: legacy SAM2/raw-sam3 output attrs. Sam3TrackerVideoSegmentationOutput exposes ONLY object_ids, pred_masks, object_score_logits, frame_idx. {bad}. See errors_N_fixes.md #40'); sys.exit(1)
print('B20 PASS')
PY
```

---

## Output Format

```
=== PREFLIGHT: <filename> ===
AUTOMATED:       [A1] PASS/FAIL  [A2] PASS/FAIL  [A3] PASS/FAIL
GENERIC MANUAL:  [B1] …  [B2] …  [B3] …  [B4] …  [B5] …  [B6] …  [B7] PASS/N/A  [B8] …  [B9] …
REGRESSION:      [B10] …  [B11] …  [B12] …  [B13] …  [B14] …  [B15] …
TX 5.x:          [B16] …  [B17] …  [B18] …  [B19] …  [B20] …
TOTAL: X/22 passed. Y FAILs need fixing. (B7 is N/A for non-training scripts.)
```

List all FAILs with line numbers and fix instructions referencing `iter/iter8/errors_N_fixes.md` entry numbers.

---

## Extending this Skill

When a new GPU failure hits and gets documented in `iter/iter8/errors_N_fixes.md`, add a matching **static** check here (B16, B17, …). Criteria for inclusion:
1. **Catchable on CPU** — no GPU required
2. **Deterministic** — grep/AST/import-chain scan, not runtime sampling
3. **One-shot** — runs in <5 seconds
4. **Cited** — each check names the `errors_N_fixes.md` entry it prevents

Checks that require GPU or runtime values (e.g., mask-resolution divergence in error #15, temporal mask mismatch in #17) belong in runtime assertions in `src/m*.py`, not here.
