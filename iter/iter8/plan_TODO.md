# TODO — iter8

> **GOAL: Surgery > ExPLoRA > Frozen on Prec@K, 115K clips.**
> **Deadline: NeurIPS May 04. Budget: ~38h remaining (22 days x 2h/day minus 6h spent).**
> **If surgery doesn't improve:** `iter/utils/literarure_survey.md` — 24 JEPA variants, 3 fallback techniques.

---

## 🛑 BLOCKING: Grounded-SAM Pivot (do on Mac, no GPU needed)

SAM3 native text grounding is too weak for Indian objects — masks roofs/walls instead of vehicles/people. Evidence: 10/15 clips had wrong or missing agent masks on SANITY.

- ⬜ 🔴 WebSearch: "Grounded-SAM 2026 video segmentation best practice" for latest API
- ⬜ 🔴 Add `groundingdino` to `requirements_gpu.txt` + `setup_env_uv.sh`
- ⬜ 🔴 Modify `m10_sam_segment.py`: DINO box detection → SAM mask refinement (replace text prompt path)
- ⬜ 🔴 py_compile + ruff all modified files on Mac
- ⬜ 🔴 Fix cross-reference paths from per-module dir restructure (verify m06/m08 read from m05/m04 module dirs)
- ⬜ 🔴 Update `runbook.md` paths: `outputs/sanity/factors/` → `outputs/sanity/m10_sam_segment/` etc.

---

## 🔥 Active (Phase 1: GPU SANITY) — resume on 24GB GPU after Mac fixes

- ✅ Step A: m10 SAM 3.1 segmentation — quality gate PASS, but masks noisy (pivot to Grounded-SAM)
- ✅ Step B: m11 factor datasets — D_L/D_A/D_I generated, per-clip 2x2 verify images working
- ⬜ Step A': Re-run m10 with Grounded-SAM → verify overlay images
- ⬜ Step B': Re-run m11 with new masks → verify 2x2 grids
- ⬜ Step C: m05 frozen V-JEPA 2.1 embedding
- ⬜ Step D: m09 ExPLoRA training
- ⬜ Step E: m09 Surgery training
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
