# Next Steps (Week 1)

> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**
> **Short-term: Show Surgery > ExPLoRA > Frozen on 1K val clips (~70 min GPU).**
> **If surgery doesn't improve:** See `iter/utils/literarure_survey.md` — 24 JEPA variants surveyed, 3 techniques to steal (SIGReg from LeJEPA, leakage-free from VLA-JEPA, temporal straightening diagnostic).

---

## Implementation Status

| Component | Status | Notes |
|---|---|---|
| V-JEPA 2.1 model loading | DONE | `get_vit_by_arch()` dispatches to 2.1 `app/vjepa_2_1/` modules |
| Dense loss (predict_all) | DONE | Context loss with lambda=0.5, doubles training signal |
| Deep supervision (4-layer) | DONE | `return_hierarchical=True`, per-chunk LayerNorm, 6656-dim output |
| Predictor LR 1x | DONE | Meta uses same LR (was 10x, gold standard audit fix) |
| ExPLoRA (LoRA + unfreeze) | DONE | `--explora` flag in m09, peft LoRA injection |
| m10 SAM 3.1 segmentation | DONE | `handle_stream_request`, `propagate_in_video`, multiplex builder |
| m11 factor datasets (D_L+D_A+D_I) | DONE | Feathered mask edges, quality filters, interaction tubes |
| train_explora.sh | DONE | Safety infra: batch size, pre-flight, FATAL enforcement, signal trap |
| train_surgery.sh | **DONE** | FATAL guard removed, --surgery --factor-dir wired up |
| Checkpoint download | ON GPU | `wget vjepa2_1_vitG_384.pt` (~8 GB) — in setup_env_uv.sh |

---

## GPU Execution Order

### Step 1b: ExPLoRA [~1h GPU — READY TO RUN]

```bash
./scripts/train_explora.sh --POC 2>&1 | tee logs/explora_poc.log
```

Handles: frozen 2.1 baseline embed → ExPLoRA training → re-embed → eval. All automated.

### Step 2: Surgery [~35 min GPU — READY TO RUN]

```bash
./scripts/train_surgery.sh --POC 2>&1 | tee logs/surgery_poc.log
```

Handles: m10 SAM 3.1 (5 min) → m11 factor D_L+D_A+D_I (3 min) → m09 --surgery 3-stage (15 min) → m05 re-embed (12 min) → m06 eval. All automated.

### Step 1a: Temporal Projection [30 min CPU — after frozen 2.1 embeddings exist]

```bash
python -u src/m05b_baselines.py --encoder vjepa_2_1_frozen_shuffled \
    --model-config configs/model/vjepa2_1.yaml --POC --local-data data/val_1k_local --no-wandb
python -u src/m06c_temporal_projection.py --POC \
    --normal-encoder vjepa_2_1_frozen --shuffled-encoder vjepa_2_1_frozen_shuffled
```

---

## Decision Gate

| Step 1b ExPLoRA | Step 2 Surgery | Action |
|---|---|---|
| ExPLoRA improves | Surgery > ExPLoRA | **Strongest: surgery wins** |
| ExPLoRA improves | Surgery = ExPLoRA | **Publish ExPLoRA** |
| No change | Surgery improves | **Best novelty: standard fails, surgery succeeds** |
| No change | No change | **Debug: reverse factor order, more clips** |

---

## Future Tasks (post-POC)

### Refactor: Split m09 into focused training scripts + utils/training.py

m09 is ~2000 lines with 3 training modes (Ch10, ExPLoRA, Surgery). After POC results confirm which mode wins, split into clean modules:

```
src/
├── utils/
│   └── training.py              # JEPA step, EMA update, masking, loss, checkpoint (~300 lines)
│                                 # Shared: compute_jepa_loss, update_teacher_ema, build_optimizer,
│                                 # build_scheduler, save/load_training_checkpoint, export_student_for_eval
├── m09_pretrain.py              # Ch10 only — drift control + lambda sweep (~400 lines)
├── m09b_explora.py              # ExPLoRA only — LoRA injection + block freeze (~300 lines)
├── m09c_surgery.py              # Surgery only — 3-stage prefix unfreezing + factor loading (~400 lines)
```

11 shared functions to move (from earlier audit):

| Function | Currently in m09 | Move to |
|---|---|---|
| `compute_jepa_loss()` | line 526 | `utils/training.py` |
| `compute_drift_loss()` | line 568 | `utils/training.py` |
| `update_teacher_ema()` | line 584 | `utils/training.py` |
| `build_optimizer()` | line 601 | `utils/training.py` |
| `build_scheduler()` | line 625 | `utils/training.py` |
| `update_weight_decay()` | line 641 | `utils/training.py` |
| `build_mask_generators()` | line 530 | `utils/training.py` |
| `save_training_checkpoint()` | line 727 | `utils/training.py` |
| `load_training_checkpoint()` | line 758 | `utils/training.py` |
| `cleanup_old_checkpoints()` | line 749 | `utils/training.py` |
| `export_student_for_eval()` | line 775 | `utils/training.py` |

**When:** After POC confirms surgery works (or doesn't). ~2-3 hour refactor.
**Why not now:** High breakage risk, doesn't improve Prec@K. GOAL OVERRIDE says research results first.

### Scale-up tasks (if POC positive)

| Task | When | What |
|---|---|---|
| Run on 10K clips | After POC positive | Scale factor datasets + training for paper-quality CI |
| 6 interaction perturbations | Before FULL | Tube jitter, margin random, raw/masked mixing (proposal Sec 11.3) |
| Patch shortcut sanity check | Before paper | Eval raw vs patched clips (proposal Sec 11.8) |
| 3-5 training seeds | Before paper | Propagated CI on delta for NeurIPS |
| Cooldown (64f) implementation | Before paper | Full producer restart at 64 frames |
| WebDataset TARs for factors | Before FULL | .npy per-file won't scale to 115K |

### Ch10 improvements (deferred — comparison arm only)

| Task | What |
|---|---|
| EMA momentum ramp (0.996 → 1.0) | Linear schedule matching Meta |
| Lambda progressive (context loss warmup) | Ramp from 0 to 0.5 over first 15K steps |
| Loss outlier regulation | Skip gradient step if loss > mean + N*std |
| Distance-weighted context loss | `compute_mask_distance()` weighting |
