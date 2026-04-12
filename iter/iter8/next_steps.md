# Next Steps

> **GOAL: Surgery > ExPLoRA > Frozen on Prec@K, 115K (full) clips.**
> **GOAL: Get V-JEPA 2.1 (2B) surgical adaptation to improve Prec@K over frozen baseline on WalkIndia-200K.**                              
> **Short-term: Show Surgery > ExPLoRA > Frozen on 1K val clips**      
> **Deadline: NeurIPS May 04. Budget: ~44h total (22 days x 2h/day).**
> **If surgery doesn't improve:** See `iter/utils/literarure_survey.md` — 24 JEPA variants surveyed, 3 techniques to steal.

---

## Risk Assessment

| Component | Lines | Tested on GPU? | Crash risk | Why |
|---|---|---|---|---|
| m10 SAM 3.1 segmentation | 580 | NO | **HIGH** | API from web search, never run |
| m11 factor datasets | 450 | NO | Medium | Pure NumPy, depends on m10 output format |
| m09 ExPLoRA mode | ~200 | NO | Medium | PEFT LoRA on V-JEPA 2.1 untested |
| m09 surgery mode | ~300 | NO | **HIGH** | 3-stage training + factor loading + deep supervision |
| V-JEPA 2.1 loading | ~100 | NO | Medium | `_ensure_loaded_2_1()` + `return_hierarchical` untested |
| m05 re-embed adapted | ~50 | Partial | Low | Frozen path works (Ch9). Adapted path new |
| m06 eval | 1275 | YES | Low | Battle-tested in Ch9/Ch10 |

**Probability of full pipeline working first try: ~5%.**
**Probability after 1 SANITY debug session: ~70%.**

---

## Phases

```
Phase 0: Mac validation (FREE, 30 min)       ← catch argparse/config bugs
Phase 1: GPU SANITY (20 clips, ~$0.50)       ← find and fix crashes
Phase 2: GPU POC (1K clips, ~$2-3)           ← the experiment: Surgery vs Frozen
  IF Surgery wins on 1K:
Phase 3: Scale to 115K + ablations (~$15-20)  ← paper-quality results
Phase 4: Paper writing                        ← only after results exist
```

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
| train_surgery.sh | DONE | FATAL guard removed, --surgery --factor-dir wired up |
| Checkpoint download | ON GPU | `wget vjepa2_1_vitG_384.pt` (~8 GB) — in setup_env_uv.sh |

---

## Decision Gate (after POC)

| POC result | Next action |
|---|---|
| Surgery > Frozen (statistically significant) | Phase 3: scale to 115K |
| Surgery > Frozen (within CI overlap) | Phase 3 with more epochs or tuning |
| Surgery = Frozen | Debug: check factor quality, try different stage order |
| Surgery < Frozen | Pivot: ExPLoRA-only paper, or temporal projection diagnostic paper |

---

## Phase 3: Scale to 115K + Ablations (~$15-20 GPU)

**Only if POC shows Surgery > Frozen.**

### 3a. Full-scale training

```bash
./scripts/train_surgery.sh --FULL 2>&1 | tee logs/surgery_full.log
./scripts/train_explora.sh --FULL 2>&1 | tee logs/explora_full.log
```

### 3b. Ablations for NeurIPS

| Ablation | What to run | What it shows | GPU time |
|---|---|---|---|
| **A1: Stage contribution** | Stage 1 only, Stage 1+2, Stage 1+2+3 | Does each stage add value? | 3 x ~40 min |
| **A2: Factor type** | D_L only, D_A only, D_I only | Which factor matters most? | 3 x ~40 min |
| **A3: Surgery vs naive fine-tune** | Unfreeze same layers, raw clips (no factors) | Factor decomposition vs just unfreezing? | ~40 min |
| **A4: 3-5 random seeds** | Re-run best config, different seeds | Statistical significance | 3 x ~40 min |

**Ablation priority if time-constrained:**
1. **A3** — proves factor decomposition matters, not just unfreezing
2. **A4** — NeurIPS requires statistical rigor

### 3c. Temporal Projection (30 min CPU, optional)

```bash
python -u src/m05b_baselines.py --encoder vjepa_2_1_frozen_shuffled \
    --model-config configs/model/vjepa2_1.yaml --POC --local-data data/val_1k_local --no-wandb
python -u src/m06c_temporal_projection.py --POC \
    --normal-encoder vjepa_2_1_frozen --shuffled-encoder vjepa_2_1_frozen_shuffled
```

---

## Troubleshooting

| Problem | Fix | Time |
|---|---|---|
| SAM 3.1 API crash | Read `sam3` package source, fix API calls | 1-2h |
| V-JEPA 2.1 shape mismatch | `state_dict.keys()` vs model params | 1h |
| LoRA target modules wrong | `print(model)` → find attn module names | 30 min |
| Surgery loss NaN | Lower LR, check grad norms | 1h |
| D_I: 0% clips have tubes | Lower `max_distance_frame_fraction` in YAML | 15 min |
| D_I: 100% clips have tubes (all giant) | Raise `min_overlap_frames`, lower `tube_margin_pct` | 15 min |

**D_I note:** D_I is NOT a SAM masking problem. SAM produces the same agent masks for all 3 factors. D_I finds pairs of agents whose centroids come within 20% of frame width for 4+ consecutive frames, then crops a bounding box around both. This is geometry logic in `m10:mine_interactions()`, not SAM quality. If D_I tubes look wrong, tune `max_distance_frame_fraction` and `min_overlap_frames` in `configs/train/ch11_surgery.yaml` — 15 second fix.
| ExPLoRA works, surgery doesn't | Submit ExPLoRA + factor analysis paper | 0h pivot |
| Nothing works | Frozen 2.1 + temporal projection diagnostic paper | 2h pivot |

---

## Time Budget (44h total)

| Phase | Hours | When |
|---|---|---|
| Phase 0: Mac validation | 1h | Apr 12 |
| Phase 1: GPU SANITY + debug | 6h | Apr 13-16 |
| Phase 2: GPU POC (1K) | 3h | Apr 17-18 |
| *Decision gate: does Surgery win?* | | |
| Phase 3a: Scale to 115K | 4h | Apr 19-21 |
| Phase 3b: Ablations (A1-A4) | 8h | Apr 22-26 |
| Phase 4: Paper writing | 14h | Apr 27-May 03 |
| Buffer | 8h | |

---

## Future Refactoring (post-paper)

### Split m09 into focused training scripts

m09 is ~2000 lines with 3 modes. After POC confirms which mode wins:

```
src/
├── utils/training.py           # Shared: JEPA step, EMA, loss, checkpoint (~300 lines)
├── m09_pretrain.py             # Ch10 only (~400 lines)
├── m09b_explora.py             # ExPLoRA only (~300 lines)
├── m09c_surgery.py             # Surgery only (~400 lines)
```

**When:** After paper. ~2-3 hour refactor.

### Scale-up tasks (if POC positive)

| Task | When |
|---|---|
| 6 interaction perturbations (tube jitter, margin random, raw/masked mixing) | Before FULL |
| Patch shortcut sanity check (eval raw vs patched clips) | Before paper |
| 3-5 training seeds | Before paper |
| Cooldown (64f) implementation | Before paper |
| WebDataset TARs for factors (.npy won't scale to 115K) | Before FULL |

---

## Key Files

| File | What |
|---|---|
| `src/CLAUDE.md` | 33 codebase rules |
| `iter/iter8/runbook.md` | Commands to run on GPU |
| `iter/iter8/plan_training.md` | System design, paper strategy |
| `configs/train/ch11_surgery.yaml` | Surgery config (thresholds, stages) |
| `configs/model/vjepa2_1.yaml` | V-JEPA 2.1 model config |
| `iter/utils/literarure_survey.md` | 24 JEPA variants (related work) |
