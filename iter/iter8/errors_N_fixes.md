# Errors & Fixes — iter8 GPU SANITY (2026-04-12/13)

## Environment Setup

| # | Error | Root Cause | Fix | File |
|---|---|---|---|---|
| 1 | `setup_env_uv.sh` crashes: `.env: line 34: heud: command not found` | `GMAIL_APP_PASSWORD=vtjn heud iupn wrlc` unquoted spaces — bash executes `heud` as command | Quote the value: `"vtjn heud iupn wrlc"` | `.env` |
| 2 | `git_pull.sh` skips HF download: `HF_TOKEN not found` | Ran `git_pull.sh` before `setup_env_uv.sh` — no venv, no `python-dotenv`, `os.getenv` returns None | Swap order in runbook: setup_env first, then git_pull | `runbook.md` |
| 3 | `hf_outputs.py` crashes: `ModuleNotFoundError: No module named 'utils'` | `from utils.progress import make_pbar` without `sys.path.insert` — fails when run as CLI script | Add `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` | `src/utils/hf_outputs.py` |
| 4 | `git_pull.sh` deletes log files mid-run | `git clean -fd` removes untracked files including `logs/*.log` being written by `tee` | Add `logs/*.log` to `.gitignore` + `git rm --cached logs/*.log` | `.gitignore` |
| 5 | SAM3 numpy downgrade breaks cuML: `numpy.core.multiarray failed to import` | `uv pip install sam3` pulls `numpy<2.0` (SAM3 metadata), but cuML/CuPy needs numpy 2.x | `--no-deps` on SAM3 install; pin `numpy>=2.3` in requirements.txt | `setup_env_uv.sh`, `requirements.txt` |
| 6 | SAM3 import fails: `No module named 'pycocotools'` | SAM3's import chain pulls training code (`coco_json_loaders.py`) at load time | Add `pycocotools>=2.0.0` to `requirements_gpu.txt` | `requirements_gpu.txt` |
| 7 | SAM3 import fails: `No module named 'einops'` | SAM3's `rope.py` uses `rearrange/repeat` from einops, not declared as dependency | Add `einops>=0.6.0` to `requirements_gpu.txt` | `requirements_gpu.txt` |

## Step A: m10 SAM 3.1 Segmentation

| # | Error | Root Cause | Fix | File |
|---|---|---|---|---|
| 8 | `build_sam3_multiplex_video_predictor() got unexpected keyword argument 'gpus_to_use'` | SAM 3.1 multiplex builder doesn't accept `gpus_to_use` (SAM 3.0 only) | Use `build_sam3_predictor(version="sam3.1")` unified entry point | `m10_sam_segment.py` |
| 9 | `ModuleNotFoundError: No module named 'flash_attn_interface'` | SAM 3.1 defaults to FA3 (`flash_attn_interface`), we have FA2 (`flash_attn`) | Pass `use_fa3=False` to `build_sam3_predictor()` | `m10_sam_segment.py` |
| 10 | SIGSEGV (exit 139) — silent crash, no traceback | `torchcodec` segfaults on TAR-extracted MP4 bytes | Disable torchcodec in `video_io.py`, use PyAV | `src/utils/video_io.py` |
| 11 | `iter_clips_parallel` misuse: `TypeError: '<' not supported between instances of 'dict' and 'int'` | `for example in iter_clips_parallel(...)` iterates over tuple `(queue, event, thread)`, not clips | Unpack: `clip_q, tar_stop, _reader = iter_clips_parallel(...)` + `clip_q.get()` loop | `m10_sam_segment.py`, `m11_factor_datasets.py` |
| 12 | `FATAL: clip has empty notable_objects` — crashes entire pipeline | Some clips (7%) have empty `notable_objects` in tags.json | Return `None` from `get_agent_prompts()`, skip clip in caller | `m10_sam_segment.py` |
| 13 | Mean concept recall = 0.22 (quality gate FAIL) | All objects comma-joined into one text prompt — SAM treats as single query, detects nothing | One `add_prompt` call per object category (Meta benchmark pattern) | `m10_sam_segment.py` |
| 14 | Process hangs after "Done" — never exits, holds 20GB VRAM | SAM3 spawns async frame-loading threads, no `shutdown()` method | `os._exit(0)` after cleanup (kills orphan threads) | `m10_sam_segment.py` |
