# Expected vs Real GPU Errors: Ch10 Training Pipeline

> **Context:** `run_pretrain.sh` and all `src/m*.py` scripts were developed on CPU (M1 Mac).
> First GPU test: RTX Pro 6000 Blackwell 96GB, 2026-03-28.

---

## Pipeline Overview

`run_pretrain.sh` orchestrates the **Ch10 lambda ablation sweep** (4 drift-control values x 3 steps):

```
For each lambda in [0, 0.001, 0.01, 0.1]:
  A) m09_pretrain.py    — train student encoder with that lambda
  B) m05_vjepa_embed.py — re-embed clips with the adapted encoder
  C) m06_faiss_metrics.py — compute retrieval metrics

Final: m08b_compare.py  — radar plot comparing frozen vs all lambdas
```

---

## Real Errors (from `logs/ch10_sanity.log`)

### Error R1 — `ModuleNotFoundError: No module named 'models'` (CRASH)

**NOT anticipated.** The `sys.path` added `deps/vjepa2/` but vjepa2 internally imports `from src.models...` (not `from models...`). Our `src/utils/__init__.py` shadows vjepa2's `src/utils/` because Python resolves regular packages over namespace packages.

**Fix:** Created `src/utils/vjepa2_imports.py` shim that temporarily isolates sys.path + CWD to import vjepa2 modules without collision.

### Error R2 — `RuntimeError: [enforce fail at inline_container.cc:672]` (CRASH, disk full)

**NOT anticipated.** Each full training checkpoint = 16GB (student 3.8G + teacher 3.8G + predictor 0.1G + optimizer 7.6G). Lambda=0 wrote 4 checkpoints (step25, step50, latest, final) = 64GB + pretrained 16GB = 80GB. Disk 100% full before lambda=0.001 started.

**Fix:** Implemented light checkpoints (no optimizer, ~8GB) for periodic saves, full checkpoints only for `latest` (resume). Added `cleanup_old_checkpoints(keep_n=2)` and post-training cleanup.

### Error R3 — Dir name mismatch: `m09_lambda0_0` vs `m09_lambda0` (SILENT)

**NOT anticipated.** `str(0.0).replace(".", "_")` = `"0_0"` (float string includes `.0`), but shell `LAMBDA_DIRS` expected `"lambda0"`. Result: verify step couldn't find `student_encoder.pt`, m05 re-embed was skipped, m06 metrics failed.

**Fix:** Changed to `f"{lam:g}".replace(".", "_")` which gives `"0"` for `0.0`, `"0_01"` for `0.01`.

### Error R4 — `run_step` exit code always "0" in FAILED log (MISREPORT)

**NOT anticipated.** Bash `if cmd; then..else.. fi` resets `$?` after evaluation. The "exit code 0" in `FAILED: ... exit code $?` was wrong — Python actually returned non-zero.

**Fix:** Capture exit code via `${PIPESTATUS[0]}` before the `if`.

### Error R5 — Pipeline continued after crash (`|| continue`)

**Anticipated** (by design) but **wrong for research.** Failed lambda=0.001/0.01/0.1 silently continued to next iteration, wasting time and masking the root cause (disk full).

**Fix:** Changed `|| continue` to `|| exit 1` (hard fail).

---

## Anticipated Errors: Status After SANITY Run

| # | Anticipated Bug | Status | Notes |
|---|----------------|--------|-------|
| 8 | Producer exhausts after 1 epoch | **NOT triggered** (SANITY uses only 5 clips, 50 steps x BS=2 = 100 draws, stream has enough) | **FIXED anyway** — removed `break`, added epoch loop. Will be critical for FULL. |
| 4 | Double LayerNorm on teacher | **PRESENT but silent** | **FIXED** — removed `F.layer_norm(h, ...)`. ViT already normalizes. |
| 7 | PyTorch ops in producer thread | **PRESENT but silent** (low throughput) | **FIXED** — added `torch.set_num_threads(1)`. |
| 6 | GradScaler with bfloat16 | **PRESENT, no warnings** (PyTorch silently no-ops) | **FIXED** — `enabled=False` when dtype is bfloat16. |
| 9 | `init_params` on GPU (+4GB) | **NOT triggered** (lambda=0 disables drift) | **FIXED** — stored on CPU, `.to(device)` in loss function. |
| 5 | Checkpoint key mismatch | **NOT an issue** — 484/484 keys loaded (100%) | No fix needed. Key stripping works correctly. |
| 10 | m05 checkpoint path mismatch | **NOT triggered** (re-embed was skipped due to R3) | **FIXED** — worker now uses suffix-aware checkpoint path. |
| 12 | m06 adapted embeddings path | **Triggered** (FATAL: embeddings not found) — downstream of R3 | Fixed by R3 fix (dir name alignment). |
| 13 | m08b lambda variant names | **Partially triggered** — only 1 encoder found, skipped comparison | Fixed by R3 fix. |

---

## Unanticipated Errors (Missed in Analysis)

| # | Error | Why Missed |
|---|-------|-----------|
| R1 | `src/` namespace collision | Analyzed vjepa2 API signatures but not Python import resolution. Our `src/utils/__init__.py` shadows vjepa2's `src/utils/` via namespace package rules. |
| R2 | Disk full (16GB x 4 checkpoints) | Anticipated OOM for VRAM but not disk. Checkpoint size = student + teacher + predictor + optimizer = 16GB was not calculated. |
| R3 | `str(0.0)` = `"0.0"` not `"0"` | Python float-to-string edge case. Should have tested `str(float)` for all lambda values. |
| R4 | `$?` reset in bash if/else | Bash semantics — `$?` after `if` is always 0 in else branch. Missed because `run_step` was developed on CPU where errors didn't trigger. |

---

## Summary of All Fixes Applied

| File | Fix | Lines Changed |
|------|-----|---------------|
| `src/utils/vjepa2_imports.py` | **NEW** — import shim for vjepa2 namespace isolation | New file |
| `src/m09_pretrain.py` | Lambda dir name: `f"{lam:g}"` | 1 line |
| `src/m09_pretrain.py` | Producer epoch loop: removed `break` | 3 lines |
| `src/m09_pretrain.py` | `torch.set_num_threads(1)` in producer | 1 line |
| `src/m09_pretrain.py` | Removed double LayerNorm on teacher | 1 line deleted |
| `src/m09_pretrain.py` | GradScaler disabled for bfloat16 | 2 lines |
| `src/m09_pretrain.py` | `init_params` stored on CPU | 2 lines |
| `src/m09_pretrain.py` | Light vs full checkpoints + `keep_last_n` cleanup | 30 lines |
| `src/m09_pretrain.py` | CSV loss logging (per-step) | 15 lines |
| `src/m09_pretrain.py` | WandB logging every step (not every 30s) | 10 lines |
| `src/m05_vjepa_embed.py` | Worker checkpoint path uses suffix | 4 lines |
| `scripts/run_pretrain.sh` | Hard fail (`exit 1` not `continue`) | 1 line |
| `scripts/run_pretrain.sh` | `${PIPESTATUS[0]}` for real exit code | 5 lines |
| `scripts/run_pretrain.sh` | Removed `--no-wandb` from all commands | 4 lines |
| `scripts/profile_vram.py` | Import shim for vjepa2 namespace | 15 lines |

---

## Disk Budget (Post-Fix)

| Phase | Peak Disk | After Cleanup |
|-------|-----------|---------------|
| Pretrained checkpoint | 16 GB | 16 GB (permanent) |
| 1 lambda training (active) | +40 GB | +20 GB (final + student) |
| 4 lambdas sequential | 96 GB total | 96 GB total |
| Re-embedding (m05) | +0.2 GB | +0.2 GB per lambda |
| **Total** | **97 GB** | **Fits in 150 GB** |
