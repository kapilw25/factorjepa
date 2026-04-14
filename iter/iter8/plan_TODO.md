# TODO — iter8

> **Final GOAL: Surgery > ExPLoRA > Frozen on Prec@K, 115K clips.**
> **Immediate GOAL: Surgery > ExPLoRA > Frozen on Prec@K, 1K clips from @data/val_1k_local/manifest.json**
> **m10/m11 Goal = maximize D_A/D_L/D_I accuracy for Prec@K**
> **Deadline: NeurIPS May 04. Budget: ~38h remaining (22 days x 2h/day minus 6h spent).**
> **If surgery doesn't improve:** `iter/utils/literarure_survey.md` — 24 JEPA variants, 3 fallback techniques.

---

## ✅ DONE: Grounded-SAM Pivot (2026-04-14, on GPU directly)

**Architecture (locked in):**
```
Grounded-SAM Path D:  fixed 17-cat agent taxonomy → Grounding DINO (text → boxes on frame 0)
                      → SAM 3.1 add_prompt(text=cat, boxes=DINO_xywh_norm, box_labels=[1]*N)
                      → SAM 3.1 propagates masks across all 16 frames (text drives tracking,
                        boxes refine frame 0). Per-category sessions preserve obj_id ranges
                        (offset += 100) for D_I cross-category mining.
```

**Verified on SANITY v5 (20 clips, 24GB RTX PRO 4000 Blackwell):**
- 12/20 clips with detected agents (8 truly-empty Goa/monument scenes correctly skipped)
- 82 total agent detections, mean mask confidence 0.93
- 39 interaction tubes from 9/20 clips (vs 0 with text-only or boxes-only)
- Per-clip 2x2 verify grids show clean D_L blur, D_A isolation, D_I crops
- m10 quality gate PASS, m11 manifest produced for all 20 clips

**Done tasks:**
- ✅ Modified `m10_sam_segment.py`: DINO box detection + SAM 3.1 text+boxes hybrid (Path D)
- ✅ Added `iopath`, `ftfy` to `requirements_gpu.txt` (SAM3 `--no-deps` undeclared deps)
- ✅ Added `load_dotenv()` to m10 (HF_HOME, HF_TOKEN propagation)
- ✅ Fixed transformers 4.57 API renames (`box_threshold`→`threshold`, `labels`→`text_labels`)
- ✅ Fixed SAM 3.1 box format (xyxy→normalized xywh, paired box_labels=[1]*N)
- ✅ Tuned thresholds Option C: DINO box=0.15, text=0.12; m11 min_agent_area_pct=0.003
- ✅ Updated `runbook.md` paths: `outputs/sanity/factors/` → `outputs/sanity/m10_sam_segment/` etc.
- ✅ Fixed taxonomy: 17 agent categories in `ch11_surgery.yaml` (not VLM tags) for accuracy
- ✅ Added step [9/9] to `setup_env_uv.sh` to pre-cache Grounding DINO weights

**Documented in `errors_N_fixes.md` entries #18-27.**

---

## 🔥 Active (Phase 1: GPU SANITY) — Steps C/D/E next

- ✅ Step A: m10 Grounded-SAM segmentation — quality gate PASS, 12/20 clips with clean masks
- ✅ Step B: m11 factor datasets — D_L blurred, D_A isolated, D_I 39 tubes/9 clips, all 2x2 grids correct
- 🔥 Step C: m05 frozen V-JEPA 2.1 embedding
- 🔥 Step D: m09 ExPLoRA training
- 🔥 Step E: m09 Surgery training (uses `--factor-dir outputs/sanity/m11_factor_datasets/`)
- ⬜ Commit all fixes via `git_push.sh`

---

## 🚧 Phase 2: POC (96GB GPU, after SANITY passes on 24GB)

- ⬜ POC: `train_explora.sh --POC` + `train_surgery.sh --POC`
- ⬜ 🎯 Decision gate: compare Prec@K frozen vs ExPLoRA vs surgical

| POC result | Next action |
|---|---|
| Surgery > Frozen (significant) | Phase 3: scale to 115K |
| Surgery > Frozen (within CI) | More epochs or tuning |
| Surgery = Frozen | Debug factor quality, try different stage order |
| Surgery < Frozen | Pivot: ExPLoRA-only paper, or temporal projection diagnostic |
| ExPLoRA works, surgery doesn't | Submit ExPLoRA + factor analysis paper (0h pivot) |
| Nothing works | Frozen 2.1 + temporal projection diagnostic paper (2h pivot) |

---

## 🔬 Phase 3: Ablations (if POC positive)

| Ablation | What it shows | GPU time |
|---|---|---|
| **A1: Stage contribution** | Stage 1 only, 1+2, 1+2+3 — does each stage add value? | 3 × ~40 min |
| **A2: Factor type** | D_L only, D_A only, D_I only — which factor matters most? | 3 × ~40 min |
| **A3: Surgery vs naive fine-tune** | Unfreeze same layers, raw clips (no factors) — is factoring the key? | ~40 min |
| **A4: Random seeds** | 3-5 seeds of best config — statistical significance | 3 × ~40 min |

Priority if time-constrained: **A3** (proves factoring matters) then **A4** (NeurIPS rigor).

---

## 📋 Backlog

- ⬜ 🟡 Paper figures: per-clip segmentation samples (m08 CPU-only)
- ⬜ 🟡 Verification videos: MP4 with mask overlay for temporal consistency
- ⬜ 🟡 Output dir restructure: verify all cross-references after per-module migration
- ⬜ 🟢 `hf_outputs.py` upload: `git_push.sh` doesn't `source .env`
- ⬜ 🟢 `setup_env_uv.sh`: cuML/SAM3 version ping-pong
- ⬜ 🟢 FA3 installation: only if SAM3 bottleneck on FULL

---

## 🔧 Troubleshooting

| Problem | Fix | Time |
|---|---|---|
| Grounded-SAM box quality poor | Try YOLO-World + SAM instead | 2h |
| V-JEPA 2.1 shape mismatch | `state_dict.keys()` vs model params | 1h |
| LoRA target modules wrong | `print(model)` → find attn module names | 30 min |
| Surgery loss NaN | Lower LR, check grad norms | 1h |
| D_I: 0% clips have tubes | Lower `max_distance_frame_fraction` in YAML | 15 min |
| D_I: 100% clips have tubes | Raise `min_overlap_frames`, lower `tube_margin_pct` | 15 min |

---

## 🔮 Future (post-paper)

- Split m09 (~2000 lines) into m09_pretrain.py + m09b_explora.py + m09c_surgery.py
- WebDataset TARs for factor datasets (.npy won't scale to 115K)
- 6 interaction perturbations (tube jitter, margin random, raw/masked mixing)
- Patch shortcut sanity check (eval raw vs patched clips)
- Cooldown (64f) implementation

---

## ✅ Completed (2026-04-12/13, 6h GPU)

- 17 bugs found and fixed (see `errors_N_fixes.md`)
- Per-module output dirs: `outputs/{mode}/{module_name}/` for all m04-m11
- SAM3 integration validated: model loading, per-object prompting, async exit fix
- torchcodec SIGSEGV diagnosed → PyAV fallback
- m10 overlay verification images + m11 2x2 per-clip grids implemented
- Composite quality gate (4 checks: pixel ratio, mask confidence, clips with agents)
- `--plot` flag on m10 and m11 for CPU-only plot regeneration

---

## ⏱️ Time Budget

| Phase | Hours | Status |
|---|---|---|
| Phase 0: Mac (Grounded-SAM pivot) | 2-3h | ⬜ |
| Phase 1: GPU SANITY (24GB, $0.20/hr) | 6h spent + ~2h remaining | 🔄 A/B done, C/D/E pending |
| Phase 2: GPU POC (96GB, $0.80/hr) | 3h | ⬜ |
| Decision gate | — | ⬜ |
| Phase 3: Scale 115K + ablations | 12h | ⬜ |
| Phase 4: Paper writing | 14h | ⬜ |
| Buffer | 2h | |

---

## 📁 Key Files

| File | What |
|---|---|
| `src/CLAUDE.md` | Codebase rules |
| `iter/iter8/runbook.md` | GPU execution commands |
| `iter/iter8/plan_training.md` | System design, architecture, literature |
| `iter/iter8/errors_N_fixes.md` | 17 bugs catalogued |
| `configs/train/ch11_surgery.yaml` | Surgery config (stages, thresholds) |
| `configs/model/vjepa2_1.yaml` | V-JEPA 2.1 model config |
| `iter/utils/literarure_survey.md` | 24 JEPA variants (fallback) |
