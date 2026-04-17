---
name: preflight
description: Pipeline code review checklist + automated checks (py_compile, AST, ruff) + call pattern + SIGSEGV detection + SAM3/Grounded-SAM/m09-split regression guards derived from iter/iter8/errors_N_fixes.md #1-#58. Catches known failure patterns BEFORE GPU run.
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

## Part E: Grounded-SAM / Path D Guards (B21-B25)

Added 2026-04-14 during the SAM3-text-only → Grounded-SAM pivot. Each cites `iter/iter8/errors_N_fixes.md` entry.

**B21. `load_dotenv()` at top of SAM3/transformers scripts** (catches error #21 — m10 ran without HF_HOME/HF_TOKEN because it never sourced `.env`):

Any file that imports `sam3` OR any `transformers` auto-loader (`AutoModel`, `AutoProcessor`, `AutoModelFor*`) MUST call `load_dotenv()` BEFORE those imports — they read HF_HOME/TRANSFORMERS_CACHE at import time.
```bash
python3 -c "
import re, sys
src = open('<file>').read()
needs_env = bool(re.search(r'(^|\n)\s*(import sam3\b|from sam3\b|from transformers import.*Auto[A-Z]\w*)', src))
if not needs_env: print('B21 SKIP'); sys.exit(0)
lines = src.split(chr(10))
# Find earliest HF-reading import (sam3 or transformers Auto*)
first_env_import = next((i for i, l in enumerate(lines) if re.search(r'(import sam3\b|from sam3\b|from transformers import.*Auto[A-Z]\w*)', l)), None)
first_load_dotenv = next((i for i, l in enumerate(lines) if 'load_dotenv(' in l and not l.lstrip().startswith('#')), None)
if first_load_dotenv is None or first_load_dotenv > first_env_import:
    print(f'B21 FAIL: sam3/AutoModel imported at line {first_env_import+1} but load_dotenv() call not found before it (or missing entirely). Add load_dotenv(.env) at top BEFORE transformers/sam3 imports (errors_N_fixes.md #21)'); sys.exit(1)
print('B21 PASS')"
```

**B22. Grounding DINO post-process kwarg + label key** (catches error #24 — transformers >=4.51 renamed `box_threshold` → `threshold` and `labels` → `text_labels`):
```bash
python3 -c "
import ast, sys
tree = ast.parse(open('<file>').read())
bad_kwarg, bad_key = [], []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'post_process_grounded_object_detection':
        for kw in node.keywords:
            if kw.arg == 'box_threshold': bad_kwarg.append(node.lineno)
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == 'labels':
        if isinstance(node.value, ast.Name) and 'result' in node.value.id.lower():
            bad_key.append(node.lineno)
if bad_kwarg: print(f'B22 FAIL: post_process_grounded_object_detection(box_threshold=...) at line(s) {bad_kwarg}. Rename to threshold= (errors_N_fixes.md #24)'); sys.exit(1)
if bad_key: print(f'B22 WARN: result[\"labels\"] at line(s) {bad_key} — transformers 5.x returns int class ids; iterate results[\"text_labels\"] for str labels (#24)')
print('B22 PASS')"
```

**B23. SAM 3.1 text+boxes hybrid required for tracking** (catches error #27 — boxes-only lost frame>0 masks, leading to silent D_A/D_I degradation):

Any call to `add_prompt(...)` on a raw `sam3` predictor that passes `bounding_boxes=` MUST also pass `text=` (HF's `add_inputs_to_inference_session` is exempt — it uses `input_boxes=` without a text arg).
```bash
python3 -c "
import ast, sys
tree = ast.parse(open('<file>').read())
bad = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_prompt':
        kws = {kw.arg for kw in node.keywords}
        if 'bounding_boxes' in kws and 'text' not in kws:
            bad.append(node.lineno)
if bad: print(f'B23 FAIL: SAM 3.1 add_prompt(bounding_boxes=...) WITHOUT text= at line(s) {bad}. Tracking will drop after frame 0. Add text=category (errors_N_fixes.md #27)'); sys.exit(1)
print('B23 PASS')"
```

**B24. Propagation guards for empty frame-0 + "No points are provided"** (catches errors #30 + #31 — m10 crashes late in 100-clip runs without these guards):

If the file calls `propagate_in_video` (raw sam3) OR `propagate_in_video_iterator` (HF), it MUST (a) pre-check the add_prompt output for `len(out_obj_ids) == 0` and continue on empty, AND (b) wrap the propagate loop in a `try/except RuntimeError` that re-raises unless `"No points are provided" in str(e)`.
```bash
python3 -c "
import re, sys
src = open('<file>').read()
if 'propagate_in_video' not in src: print('B24 SKIP'); sys.exit(0)
has_empty_guard = bool(re.search(r'(n_frame0_objs|len\(.*out_obj_ids.*\))\s*==\s*0', src))
has_runtime_guard = bool(re.search(r'except\s+RuntimeError', src)) and 'No points are provided' in src
fails = []
if not has_empty_guard: fails.append('missing empty-add_prompt guard (len(out_obj_ids)==0) — error #30')
if not has_runtime_guard: fails.append('missing try/except RuntimeError for \"No points are provided\" — error #31')
if fails: print(f'B24 FAIL: {fails}'); sys.exit(1)
print('B24 PASS')"
```

**B25. Box clamp before xywh normalization** (catches error #28 — DINO can emit out-of-frame coords that trip SAM 3.1's `(0 <= boxes_xywh <= 1).all()` assertion):

If the file computes `x/W, y/H, w/W, h/H` to pass to SAM 3.1 `bounding_boxes=`, it MUST either (a) clamp each coord with `max(0, min(W, x))` patterns BEFORE division, or (b) use HF's `Sam3TrackerVideoProcessor` path which doesn't require normalization.
```bash
python3 -c "
import re, sys
src = open('<file>').read()
if 'bounding_boxes=' not in src: print('B25 SKIP'); sys.exit(0)
has_normalize = bool(re.search(r'/\s*[WH]\b|/\s*float\([WH]\)', src))
has_clamp = bool(re.search(r'max\s*\(\s*0[.,]?\s*,\s*min\s*\(\s*[WH]', src)) or 'clip(' in src or '.clamp(' in src
uses_hf = 'add_inputs_to_inference_session' in src or 'Sam3TrackerVideoProcessor' in src
if has_normalize and not has_clamp and not uses_hf:
    print('B25 FAIL: xywh normalization present without box clamp. DINO can emit out-of-frame boxes; SAM 3.1 asserts (0<=xywh<=1).all(). Clamp x/y to [0,W]/[0,H] before /W //H (errors_N_fixes.md #28)'); sys.exit(1)
print('B25 PASS')"
```

## Part F: Raw sam3 vs HF sam3 API Guards (B26-B27)

Added 2026-04-15 during the Path B speedup (HF `Sam3TrackerVideoModel` integration).

**B26. `max_frame_num_to_track=` banned on raw sam3** (catches errors #33 + #35 — raw sam3 pkg's multiplex model has an internal bug; bounded propagation crashes on both forward and reverse):

Raw `sam3` package `propagate_in_video` / `handle_stream_request(propagate_in_video)` MUST NOT receive `max_frame_num_to_track=` (any int value). HF's `propagate_in_video_iterator` is EXEMPT (different code path that correctly clips).
```bash
python3 -c "
import re, sys
src = open('<file>').read()
uses_raw_sam3 = bool(re.search(r'(build_sam3_predictor|handle_stream_request)', src))
uses_hf_iter = 'propagate_in_video_iterator' in src
if not uses_raw_sam3: print('B26 SKIP'); sys.exit(0)
# Find max_frame_num_to_track= in non-HF context (file uses raw sam3)
bad = [i+1 for i, line in enumerate(src.split(chr(10))) if 'max_frame_num_to_track=' in line and not line.lstrip().startswith('#')]
# If file uses BOTH paths, can't easily split — warn only when NOT pure HF
if bad and not uses_hf_iter:
    print(f'B26 FAIL: max_frame_num_to_track= at line(s) {bad} on raw sam3 path. SAM 3.1 multiplex model has a bug (empty tensor crash); remove and use propagation_direction=\"forward\" instead (errors_N_fixes.md #33, #35)'); sys.exit(1)
print('B26 PASS')"
```

**B27. `Sam3VideoProcessor` kwarg + postprocess-key regression** (catches error #41 — Meta model card docs diverged from installed 5.5.4 API):

If the file calls `Sam3VideoProcessor.add_text_prompt(...)` (text-only probe path), it MUST use kwarg `text=` not `prompts=`. If it reads `postprocess_outputs(...)` result, it MUST index `["masks"]` not `["pred_masks"]`.
```bash
python3 -c "
import re, sys
src = open('<file>').read()
if 'Sam3VideoProcessor' not in src and 'Sam3VideoModel' not in src: print('B27 SKIP'); sys.exit(0)
bad_kw = [i+1 for i, line in enumerate(src.split(chr(10))) if re.search(r'add_text_prompt\s*\([^)]*prompts\s*=', line)]
bad_key = [i+1 for i, line in enumerate(src.split(chr(10))) if re.search(r'\bpost\[[\\x22\\x27]pred_masks[\\x22\\x27]\]', line)]
fails = []
if bad_kw: fails.append(f'add_text_prompt(prompts=) at line(s) {bad_kw} — rename to text= (#41)')
if bad_key: fails.append(f'post[\"pred_masks\"] at line(s) {bad_key} — Sam3VideoProcessor returns post[\"masks\"] (#41)')
if fails: print(f'B27 FAIL: {fails}'); sys.exit(1)
print('B27 PASS')"
```

## Part G: V-JEPA 2.1 + config-schema guards (B28-B31)

**B28. Hardcoded `dtype=torch.float16` in m05** (catches error #45 — m05 path must read dtype from the loaded model at runtime, not assume fp16):

Input casts in `m05_vjepa_embed.py` inside `get_batch_embeddings` / autocast / warmup tensors MUST use runtime detection (`next(model.parameters()).dtype`), NOT hardcoded `torch.float16`.
```bash
python3 -c "
import os, re, sys
target = '<file>'
if os.path.basename(target) != 'm05_vjepa_embed.py': print('B28 SKIP'); sys.exit(0)
src = open(target).read()
# Allowed: single source-of-truth at model-load time. Disallowed: in the hot forward path.
# Hot path signals: near 'autocast', 'get_batch_embeddings', 'warmup'.
bad = []
for i, line in enumerate(src.split(chr(10)), 1):
    if re.search(r'dtype\s*=\s*torch\.float16\b', line) and any(ctx in src[max(0, src.find(line)-500):src.find(line)+500] for ctx in ['get_batch_embeddings', 'autocast', 'warmup']):
        bad.append(i)
if bad:
    print(f'B28 FAIL: hardcoded dtype=torch.float16 in m05 forward path at line(s) {bad}. Use runtime detection: dt = next(getattr(model, \"_orig_mod\", model).parameters()).dtype (errors_N_fixes.md #45)'); sys.exit(1)
print('B28 PASS')"
```

**B29. `vjepa2_imports._ensure_loaded_2_1` finally-block restores ALL saved modules** (catches error #50 — finally-block only restored `src` + `src.utils` on first implementation, leaking other `src.*` pops and breaking `get_vit_predictor`):

Only applies to `src/utils/vjepa2_imports.py`. The `finally` block inside `_ensure_loaded_2_1` MUST iterate `saved_modules` and restore each key (not just two hardcoded keys).
```bash
python3 -c "
import os, re, sys
target = '<file>'
if os.path.basename(target) != 'vjepa2_imports.py': print('B29 SKIP'); sys.exit(0)
src = open(target).read()
finally_blocks = re.findall(r'finally:\s*\n(.*?)(?=\n\S)', src, re.S)
for fb in finally_blocks:
    # Good pattern: loops over saved_modules OR assigns dict back into sys.modules in a loop
    if re.search(r'for\s+\w+\s*(,\s*\w+)?\s+in\s+saved_modules', fb) or 'sys.modules.update' in fb:
        print('B29 PASS'); sys.exit(0)
print('B29 FAIL: _ensure_loaded_2_1 finally-block does not restore all saved_modules. Must iterate and re-assign (errors_N_fixes.md #50)'); sys.exit(1)"
```

**B30. `cfg[\"data\"][\"patch_size|crop_size|tubelet_size\"]` schema bug** (catches error #51 — these keys live under `cfg[\"model\"]`):
```bash
python3 -c "
import re, sys
src = open('<file>').read()
bad = []
for i, line in enumerate(src.split(chr(10)), 1):
    if re.search(r'(data_cfg|cfg\[[\\x22\\x27]data[\\x22\\x27]\])\s*\[\s*[\\x22\\x27](crop_size|patch_size|tubelet_size)[\\x22\\x27]\s*\]', line):
        bad.append(i)
if bad: print(f'B30 FAIL: cfg[\"data\"][\"patch_size/crop_size/tubelet_size\"] at line(s) {bad} — these live under cfg[\"model\"] (errors_N_fixes.md #51)'); sys.exit(1)
print('B30 PASS')"
```

**B31. `cfg[\"optimization\"][\"max_epochs\"][mode_key]` double-subscript** (catches error #52 — `merge_config_with_args` already flattens the per-mode dict into an int):
```bash
python3 -c "
import re, sys
src = open('<file>').read()
bad = [i+1 for i, line in enumerate(src.split(chr(10))) if re.search(r'\[[\\x22\\x27]max_epochs[\\x22\\x27]\]\s*\[\s*mode_key\s*\]', line)]
if bad: print(f'B31 FAIL: max_epochs[mode_key] double-subscript at line(s) {bad} — merge_config_with_args already flattened (errors_N_fixes.md #52)'); sys.exit(1)
print('B31 PASS')"
```

## Part H: m09 split training regression guards (B32-B36)

Added 2026-04-15 post-#49 (m09 monolith → m09a/m09b/m09c split).

**B32. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` at main() of m09\*** (catches error #53 — fragmented blocks prevent OOM recovery):
```bash
python3 -c "
import os, re, sys
target = '<file>'
if not re.match(r'm09[abc]_.*\.py$', os.path.basename(target)): print('B32 SKIP'); sys.exit(0)
src = open(target).read()
has_setdefault = bool(re.search(r'os\.environ\.setdefault\([\\x22\\x27]PYTORCH_CUDA_ALLOC_CONF[\\x22\\x27]\s*,\s*[\\x22\\x27]expandable_segments:True', src))
if not has_setdefault:
    print(f'B32 FAIL: m09 training script missing os.environ.setdefault(\"PYTORCH_CUDA_ALLOC_CONF\", \"expandable_segments:True\") (errors_N_fixes.md #53)'); sys.exit(1)
print('B32 PASS')"
```

**B33. m09c per-stage pre-init of loss vars** (catches error #54 — OOM-only stages hit `UnboundLocalError: jepa_val` at the summary print):
```bash
python3 -c "
import os, re, sys
target = '<file>'
if os.path.basename(target) != 'm09c_surgery.py': print('B33 SKIP'); sys.exit(0)
src = open(target).read()
needed = ['jepa_val', 'masked_val', 'context_val']
# Require each var to be assigned 0.0 somewhere INSIDE a stage loop (looked for via a simple pre-init block pattern)
bad = [v for v in needed if not re.search(rf'\b{v}\s*=\s*0\.0\b', src)]
if bad: print(f'B33 FAIL: m09c missing per-stage pre-init for {bad}. Add v = 0.0 before inner for-loop in each stage (errors_N_fixes.md #54)'); sys.exit(1)
print('B33 PASS')"
```

**B34. m09b/m09c within-step retry + 0-step fail-hard** (catches error #55 — static-BS sub-batch=1 OOM with nowhere to shrink would silently export unmodified weights):
```bash
python3 -c "
import os, re, sys
target = '<file>'
if not re.match(r'm09[bc]_.*\.py$', os.path.basename(target)): print('B34 SKIP'); sys.exit(0)
src = open(target).read()
has_while = bool(re.search(r'while\s+not\s+step_succeeded', src))
has_fail_hard = bool(re.search(r'if\s+global_step\s*==\s*0\s*:\s*\n\s*raise\s+RuntimeError', src, re.M)) or 'm09b' in os.path.basename(target)  # m09b has equivalent inside-loop check
has_min_raise = bool(re.search(r'train_sizer\.size\s*<=\s*train_sizer\.min_size', src))
fails = []
if not has_while: fails.append('missing while not step_succeeded retry loop (#55)')
if not has_min_raise: fails.append('missing fail-hard when sizer.size <= sizer.min_size (#55)')
if 'm09c' in os.path.basename(target) and not has_fail_hard: fails.append('m09c missing post-loop if global_step == 0: raise (#55)')
if fails: print(f'B34 FAIL: {fails}'); sys.exit(1)
print('B34 PASS')"
```

**B35. AdaptiveBatchSizer wired in every GPU batch loop** (catches error #47 — sizer is universal INFRA, must be in m04/m05/m05b/m05c/m09*):
```bash
python3 -c "
import os, re, sys
target = '<file>'
modules_needing_sizer = {'m04_vlm_tag.py', 'm04d_motion_features.py', 'm05_vjepa_embed.py', 'm05b_baselines.py', 'm05c_true_overlap.py', 'm09a_pretrain.py', 'm09b_explora.py', 'm09c_surgery.py'}
if os.path.basename(target) not in modules_needing_sizer: print('B35 SKIP'); sys.exit(0)
src = open(target).read()
has_import = 'AdaptiveBatchSizer' in src or '_train_step_grad_accum' in src
has_memory_cap = bool(re.search(r'memory_cap\s*=\s*[\w\.\[\]\\x22\\x27]*gpu_memory_target', src)) or bool(re.search(r'memory_cap\s*=.*yaml', src, re.I))
fails = []
if not has_import: fails.append('AdaptiveBatchSizer/grad-accum helper not imported or used')
if not has_memory_cap: fails.append('memory_cap not read from pipeline.yaml gpu_memory_target (hardcoded 0.85 forbidden — #47)')
if fails: print(f'B35 FAIL: {fails} (errors_N_fixes.md #46, #47, #48)'); sys.exit(1)
print('B35 PASS')"
```

**B36. `requirements_gpu.txt` transformers pin is 5.5.4** (catches error #36 — `Sam3TrackerVideoModel` was added in transformers 5.x; 4.57 would ImportError):
```bash
source venv_walkindia/bin/activate && python3 -c "
import re, sys, os
if not os.path.exists('requirements_gpu.txt'): print('B36 SKIP: no requirements_gpu.txt'); sys.exit(0)
req = open('requirements_gpu.txt').read()
m = re.search(r'^\s*transformers\s*([<>=~!]+)\s*([\d.]+)', req, re.M)
if not m: print('B36 FAIL: transformers not pinned in requirements_gpu.txt (errors_N_fixes.md #36)'); sys.exit(1)
op, ver = m.group(1), m.group(2)
major = int(ver.split('.')[0])
if major < 5:
    print(f'B36 FAIL: transformers pinned at {op}{ver} — Sam3TrackerVideoModel needs >=5.0. Pin to ==5.5.4 (errors_N_fixes.md #36)'); sys.exit(1)
print(f'B36 PASS: transformers {op}{ver}')"
```

---

## Output Format

```
=== PREFLIGHT: <filename> ===
AUTOMATED:       [A1] PASS/FAIL  [A2] PASS/FAIL  [A3] PASS/FAIL
GENERIC MANUAL:  [B1] …  [B2] …  [B3] …  [B4] …  [B5] …  [B6] …  [B7] PASS/N/A  [B8] …  [B9] …
REGRESSION:      [B10] …  [B11] …  [B12] …  [B13] …  [B14] …  [B15] …
TX 5.x:          [B16] …  [B17] …  [B18] …  [B19] …  [B20] …
GROUNDED-SAM:    [B21] …  [B22] …  [B23] …  [B24] …  [B25] …
RAW-vs-HF SAM3:  [B26] …  [B27] …
VJEPA/CFG:       [B28] …  [B29] …  [B30] …  [B31] …
m09 SPLIT:       [B32] …  [B33] …  [B34] …  [B35] …  [B36] …
TOTAL: X/38 passed. Y FAILs need fixing. (Many checks SKIP when not applicable to the target file.)
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
