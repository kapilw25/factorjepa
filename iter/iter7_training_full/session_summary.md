# Session Summary: Ch10 Continual Pretraining (2026-03-28/29)

> GPU: RTX Pro 6000 Blackwell 96GB, ~18h session, ~$14 GPU cost

---

## What Was Built

### New Files Created
- `src/m09_pretrain.py` — V-JEPA 2 student-teacher JEPA training with EMA, drift control, epoch-based
- `src/utils/vjepa2_imports.py` — Import shim for vjepa2 namespace collision
- `src/utils/output_guard.py` — Output verification + pipeline preflight (checks ALL steps before GPU work)
- `configs/pretrain/vitg16_indian.yaml` — Training hyperparameters (LR, EMA, masking, epochs per mode)
- `configs/pipeline.yaml` — Single source of truth for clip limits, streaming params, GPU defaults
- `scripts/profile_vram.py` — VRAM profiler (found BS=112 optimal with grad checkpointing)
- `.claude/hooks/fail-hard-research.sh` — Blocks silent error swallowing
- `.claude/hooks/protect-checkpoints.sh` — Blocks unguarded deletion of training outputs
- `docs/index.html` — Added "Foundations" tab + Ch10 results table

### Files Significantly Modified
- `scripts/run_pretrain.sh` — Full 3-phase pipeline (train → embed → eval), 3-mode (SANITY/POC/FULL)
- `scripts/run_evaluate.sh` — Added --POC flag, output preflight, local data routing
- `src/m05_vjepa_embed.py` — Adapted model support (permutation fix, autocast, --shuffle, --encoder)
- `src/m06_faiss_metrics.py` — Output guard, hard fail on missing data
- `src/m08b_compare.py` — Dynamic labels for lambda encoders, ENCODER_ORDER override
- All 20 `src/m*.py` files — 3-flag refactor (--SANITY/--POC/--FULL), no hardcoded values
- `src/utils/config.py` — `get_pipeline_config()`, `get_total_clips()`, `get_sanity_clip_limit()`

---

## 10K POC Results

| Metric | Frozen V-JEPA | Adapted (λ=0.001, 5ep) | Delta |
|--------|:---:|:---:|:---:|
| Prec@K (Easy) | 36.09% | 36.14% | +0.05% (noise) |
| Cycle@K (Easy) | 76.01% | 75.31% | -0.70% |
| Prec@K (Hard) | 34.70% | 34.70% | 0.00% |

**Conclusion:** 10K clips insufficient for 1B model adaptation. Zero meaningful change.

---

## Key Decisions Made

1. **Epoch-based training, not step-based** — same clips processed regardless of batch size
2. **Winner by jepa_loss, not Cycle@K** — skips 6.8h of m05 re-embedding per lambda
3. **No V-JEPA deduplication** — circular reasoning (model judges its own eval set)
4. **Lambda ablation removed for FULL** — POC proved all 4 lambdas identical, saves 25.6h
5. **Retrain winner from scratch** — confirmed by ML research consensus (Google Tuning Playbook, DINOv2, BYOL)
6. **Shuffled adapted encoder** — key ablation for temporal interference finding ($1.36 GPU cost, high paper value)
7. **CUDA expandable_segments:True** — fixes memory fragmentation OOM

---

## Bugs Found & Fixed (14 total)

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | vjepa2 src/ namespace collision | CRASH | CWD-based import shim |
| 2 | Disk full (16GB × 4 checkpoints) | CRASH | Light checkpoints + cleanup |
| 3 | Lambda dir name (`str(0.0)` → `"0_0"`) | SILENT | `f"{lam:g}"` formatting |
| 4 | torch.compile + float16 adapted | CRASH | Skip compile for .pt models |
| 5 | Tensor permutation (B,T,C,H,W) | CRASH | Permute in get_batch_embeddings |
| 6 | GradScaler with bfloat16 | WASTE | Disabled (bfloat16 has full range) |
| 7 | Producer break after 1 epoch | SILENT | Removed break, epoch loop |
| 8 | CUDA fragmentation (13.8GB) | OOM | expandable_segments:True |
| 9 | Fixed steps = variable clips | WRONG | Epoch-based training |
| 10 | All lambdas overwrite same file | SILENT | Per-lambda encoder names |
| 11 | Winner stdout parsing fragile | FRAGILE | JSON-to-JSON via ablation_winner.json |
| 12 | 5-epoch student deleted (3h GPU lost) | DATA LOSS | Epoch-count guard + hook |
| 13 | Missing inputs found mid-pipeline | WASTE | Preflight checks ALL steps at start |
| 14 | `drift_cfg` used before assignment | CRASH | ruff F821 check added to hook |

---

## GPU Time Consumed

| Activity | Duration | Outcome |
|----------|:--------:|---------|
| VRAM profiler | 10 min | BS=112 optimal |
| SANITY debugging (×6 attempts) | ~2h | Fixed 8 bugs |
| Lambda ablation (4 × 1 epoch) | 2.3h | All identical |
| Winner deep train (5 epochs) | 3h | student_encoder.pt (then lost, rebuilt) |
| m05 re-embed (10K adapted) | 1.8h | embeddings |
| Phase 3 eval (m06b, m07, m08, m08b) | 15 min | Radar plot |
| **Total useful** | **~9h** | |
| **Total wasted (bugs/reruns)** | **~5h** | |
| **Grand total** | **~14h** | |

---

## Next Steps (on new GPU instance, ≥200GB disk)

### Immediate (before any GPU work)
- [ ] `git pull` latest code (all 3-flag changes, output_guard, etc.)
- [ ] `./setup_env_uv.sh --gpu --from-wheels`
- [ ] Run profiler: `python scripts/profile_vram.py` (new GPU may have different optimal BS)

### Step 1: Download 115K (~1h)
```bash
python -u src/m00d_download_subset.py --FULL 2>&1 | tee logs/m00d_full.log
```

### Step 2: Ch9 Eval on 115K (~110h, run first)
```bash
tmux new -s ch9
./scripts/run_evaluate.sh --FULL 2>&1 | tee logs/ch9_full.log
```
**Bottleneck:** m04 VLM tagging (~35h). Consider vLLM for 10x speedup if available.

### Step 3: Ch10 Pretrain on 115K (~74h)
```bash
tmux new -s ch10
./scripts/run_pretrain.sh --FULL 2>&1 | tee logs/ch10_full.log
```
**No ablation sweep** — trains λ=0.001 directly (POC winner).

### Step 4: Analyze results
- Compare adapted vs frozen on 115K → this is the paper's main result
- If Prec@K improves → continual pretraining works, proceed to Ch11
- If Prec@K still flat → 1B model architecture can't adapt via JEPA loss alone

### Step 5: Ch11 Surgery (if Step 4 shows gain)
- m10_sam_segment.py, m11_factor_datasets.py, m12_surgery.py — NOT BUILT YET
- See `plan_training.md` for design

---

## Files to Commit

```bash
# Key new/modified files:
git add src/m09_pretrain.py src/utils/vjepa2_imports.py src/utils/output_guard.py
git add configs/pipeline.yaml configs/pretrain/vitg16_indian.yaml
git add scripts/run_pretrain.sh scripts/run_evaluate.sh scripts/profile_vram.py
git add .claude/hooks/fail-hard-research.sh .claude/hooks/protect-checkpoints.sh
git add .claude/hooks/post-edit-lint.sh .claude/settings.json
git add src/CLAUDE.md src/MEMORY.md docs/index.html requirements.txt
git add iter/iter7_training_full/

# All modified src/m*.py (3-flag refactor + output_guard + no hardcoding):
git add src/m0*.py src/m04*.py src/m05*.py src/m06*.py src/m07*.py src/m08*.py
```
