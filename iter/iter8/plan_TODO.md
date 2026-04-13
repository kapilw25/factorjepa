# TODO — iter8

## 🔥 Active (Phase 1: GPU SANITY)

- ✅ Step A: m10 SAM 3.1 segmentation — PASS (⚠️ noisy masks accepted, quality gate passed)
- ⬜ Step B: m11 factor datasets (D_L + D_A + D_I)
- ⬜ Step C: m05 frozen V-JEPA 2.1 embedding
- ⬜ Step D: m09 ExPLoRA training
- ⬜ Step E: m09 Surgery training
- ⬜ Commit all fixes via `git_push.sh`

## 🚧 Blocked on SANITY completion

- ⬜ POC: `train_explora.sh --POC` + `train_surgery.sh --POC`
- ⬜ 🎯 Decision gate: compare Prec@K frozen vs ExPLoRA vs surgical

## 📋 Backlog (post-POC)

- ⬜ 🔴 **Pivot m10 to Grounded-SAM (Grounding DINO + SAM)** — SAM3 native text grounding is too weak for Indian objects (masks roofs/walls instead of vehicles/people). DINO detects boxes, SAM refines masks. ~2-3h. Do this BEFORE scaling to POC/FULL.
- ⬜ 🟡 Paper figures: per-clip segmentation samples (original | agent | layout | D_I tubes) — m08 CPU-only
- ⬜ 🟡 Verification videos: MP4 with mask overlay for temporal consistency
- ⬜ 🟢 `hf_outputs.py` upload: `git_push.sh` skips HF upload (doesn't `source .env`)
- ⬜ 🟢 `setup_env_uv.sh`: cuML/SAM3 version ping-pong (~80MB wasted per run)
- ⬜ 🟢 FA3 installation: only if SAM3 bottleneck on FULL (115K clips)
- ⬜ 🟢 Quality gate: revisit per-category thresholds for rare Indian objects
