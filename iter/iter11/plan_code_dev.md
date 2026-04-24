# 🎯 iter11 v2 — Code Development Plan

Two blocker questions still pending at L247-250 (Task 5 Y/N, Task 6 A/B).

※ recap: Goal: build iter11 v2 infrastructure for a 4-variant apples-to-apples 10K comparison (ExPLoRA + 3 surgery recipes).  Current task: rename the yamls + build scripts/train.sh and scripts/eval.sh. Next action: answer the two blocker questions  (delete 4 superseded orchestrator scripts Y/N, and update tests_streaming/ refs A/B).

> **Goal**: Build the infrastructure for the iter11 v2 comparison:
> **🥇 Surgery > 🥈 ExPLoRA > 🥉 Frozen** on Prec@K at eval_10k (N=9,297, BCa CI ±0.42 pp).
> 4 training variants × same-budget (5 epochs × saves_per_epoch=5) × same data (subset_10k + val_1k held-out).

---

## 📋 Task Tracker (mirrors TaskList)

| # | Task | Status |
|---|---|---|
| 59 | 🏷️ Rename yamls + delete v15a | ⏳ pending |
| 60 | 🗂️ Route outputs via `--output-dir` CLI (no yaml changes) | ⏳ pending |
| 61 | 🚀 Build `scripts/train.sh` thin wrapper | ⏳ pending |
| 62 | 📊 Build `scripts/eval.sh` from `run_paired_eval_10k.sh` | ⏳ pending |

---

## 🏷️ Task 1 — Rename yamls + delete obsolete v15a

### Why
Post max-epochs unification (all full=5), `ch11_surgery_v15a.yaml` is identical to `ch11_surgery.yaml` base. The `v15*` naming is opaque; semantic names encode the actual diff (stages × agent-weight × D_I).

### File moves

| 🔴 Old | 🟢 New | ✏️ Action |
|---|---|---|
| `configs/train/explora.yaml` | *(unchanged)* | 🟰 keep |
| `configs/train/ch11_surgery.yaml` | `configs/train/surgery_2stage_noDI.yaml` | 📝 `git mv` |
| `configs/train/ch11_surgery_v15a.yaml` | — | 🗑️ delete |
| `configs/train/ch11_surgery_v15b.yaml` | `configs/train/surgery_2stage_loud_agent.yaml` | 📝 `git mv` |
| `configs/train/ch11_surgery_v15c.yaml` | `configs/train/surgery_3stage_DI.yaml` | 📝 `git mv` |

### Runtime-code updates (2 edits)

| File | Line | Change |
|---|---|---|
| `src/m09c_surgery.py` | 68 | `DEFAULT_TRAIN_CONFIG = "configs/train/surgery_2stage_noDI.yaml"` |
| `src/m09c_surgery.py` | 9 | Docstring: example `--train-config` path → new name |

### Left alone (cosmetic/deferred)

| Location | Reason |
|---|---|
| `src/m09c_surgery.py` L106/124/148/383/617/1345 comments | Cosmetic — no runtime impact |
| `src/m10_sam_segment.py`, `src/m11_factor_datasets.py` comments | Cosmetic |
| `src/CLAUDE.md`, `src/MEMORY.md` | Docs — sync at session end |
| `scripts/tests_streaming/*` | ⚠️ **Decision needed** (see Q2 below) |

---

## 🗂️ Task 2 — Route outputs via `--output-dir` CLI

### Why
4 variants cannot share `outputs/full/m09c_surgery/`. The `--output-dir` CLI flag already exists in both `m09b_explora.py:210` and `m09c_surgery.py:166` — **zero yaml/code changes needed** for this task.

### Target directory structure 📂

```
outputs/
├── full/
│   ├── explora/                       ⬅ m09b → explora.yaml
│   ├── surgery_2stage_noDI/           ⬅ m09c → surgery_2stage_noDI.yaml
│   ├── surgery_2stage_loud_agent/     ⬅ m09c → surgery_2stage_loud_agent.yaml
│   └── surgery_3stage_DI/             ⬅ m09c → surgery_3stage_DI.yaml
├── frozen_eval10k/                    ⬅ 🥇 shared Frozen baseline (reused from prior run)
├── m10_sam_segment/                   ⬅ 🧠 SAM3 masks (shared — critical cache, ~10 h if regen)
└── m11_factor_datasets/               ⬅ 🧩 factor datasets (shared — stages inputs for Surgery)
```

The `--output-dir outputs/full/<config_name>/` flag per invocation (set in `scripts/train.sh`) does the routing. 🟰 No yaml edits; wrapper-layer concern.

---

## 🚀 Task 3 — `scripts/train.sh` thin wrapper

### Design
Semicolon-chain (🚫 NOT `&&`, per overnight-chains memory rule) so a failure in one variant doesn't abort the rest.

### File skeleton

```bash
#!/usr/bin/env bash
# iter11 v2 trainer — chains 4 training runs back-to-back.
# Each variant writes to outputs/full/<config_name>/; no archive step.
# USAGE:
#   tmux new -s train
#   ./scripts/train.sh 2>&1 | tee logs/train_iter11_v2.log
set -uo pipefail   # note: NO -e — let individual failures not abort the chain
cd "$(dirname "$0")/.."
source venv_walkindia/bin/activate
mkdir -p logs outputs/full

SUBSET=data/subset_10k.json
LOCAL=data/subset_10k_local
VAL=data/val_1k.json
VAL_LOCAL=data/val_1k_local
MODEL_CFG=configs/model/vjepa2_1.yaml

# 🎓 m09b ExPLoRA
python -u src/m09b_explora.py --FULL \
    --model-config "$MODEL_CFG" \
    --train-config configs/train/explora.yaml \
    --subset "$SUBSET" --local-data "$LOCAL" \
    --val-subset "$VAL" --val-local-data "$VAL_LOCAL" \
    --output-dir outputs/full/explora \
    --no-wandb 2>&1 | tee logs/train_explora.log ;

# 🧬 m09c surgery — 3 variants
for CFG in surgery_2stage_noDI surgery_2stage_loud_agent surgery_3stage_DI ; do
    python -u src/m09c_surgery.py --FULL \
        --model-config "$MODEL_CFG" \
        --train-config "configs/train/${CFG}.yaml" \
        --subset "$SUBSET" --local-data "$LOCAL" \
        --factor-dir outputs/full/m11_factor_datasets/ \
        --probe-subset "$VAL" --probe-local-data "$VAL_LOCAL" \
        --probe-tags "$VAL_LOCAL/tags.json" \
        --output-dir "outputs/full/${CFG}" \
        --no-wandb 2>&1 | tee "logs/train_${CFG}.log" ;
done

echo "✅ iter11 v2 train chain complete — $(date)"
```

### ⚠️ Pre-flight assumptions

| Prereq | Check | If missing |
|---|---|---|
| `outputs/full/m10_sam_segment/segments.json` | Surgery needs SAM3 masks | Run `src/m10_sam_segment.py --FULL --subset ...` first (~10 h GPU) |
| `outputs/full/m11_factor_datasets/factor_manifest.json` | Surgery needs factor indices | Run `src/m11_factor_datasets.py --FULL --streaming --subset ...` (~10 min CPU) |
| `data/val_1k_local/tags.json` | Probe tag lookup | Regenerate via `m00d_download_subset.py` |

### Expected wall time ⏱️

| Variant | Est. wall | Notes |
|---|---|---|
| explora | 6–14 h | val-loss plateau may halt early |
| surgery_2stage_noDI | 6–14 h | 2-stage, no D_I |
| surgery_2stage_loud_agent | 6–14 h | S2 agent weight 0.7→0.85 |
| surgery_3stage_DI | 8–18 h | 3-stage, D_I re-enabled (+m10 interactions-only + m11 --regen-D_I prep, ~20 min CPU) |
| **Total** | **~30–60 h** | ~$24-48 on RTX Pro 6000 @ $0.8/h |

---

## 📊 Task 4 — `scripts/eval.sh` from `run_paired_eval_10k.sh`

### Design
Adapt `scripts/run_paired_eval_10k.sh` (511 LoC) with these changes:

| Change | Where | Why |
|---|---|---|
| `VARIANTS=(...)` list | L72 | Replace `v10 v13 v14 explora v15a v15b v15c` → `explora surgery_2stage_noDI surgery_2stage_loud_agent surgery_3stage_DI` |
| Student ckpt lookup | L50/L73/L144/L377 | Replace `outputs_versioned/${v}_m09c_surgery/student_encoder.pt` → `outputs/full/${v}/student_encoder.pt` |
| Per-variant archive | L310/L334/L438 | Replace `outputs_versioned/${v}_eval10k/` → `outputs/full/${v}/eval10k/` |
| Frozen archive restore hooks | L206–214, L245–259 | ✅ Keep as-is (still at `outputs_versioned/frozen_eval10k/`) — it's the shared Frozen baseline from prior run, reusable |
| `stage_from_iter9 v10 v13` block | L46–61 | 🗑️ Delete — iter9 legacy ckpts no longer in scope |
| Summary table | L479–504 | Update variant-list string literal |

### Cache-policy machinery — 🟰 keep verbatim
The `_check_and_prompt_any`, policy propagation, G1/G2/G3 gates, auto-stage trace-file hook all remain identical. These are technique-agnostic delete-protection (per DELETE PROTECTION section of `src/CLAUDE.md`).

### Frozen reuse 🟰

`outputs_versioned/frozen_eval10k/` survived the iter11 v1 cleanup — same `data/eval_10k.json` subset, same vjepa2_1.yaml architecture → **frozen baseline reusable, saves ~2.3 h GPU**. 🎉

### Expected wall time ⏱️

| Stage | Per variant | 4 variants |
|---|---|---|
| m05 surgical embed | ~2.3 h | 9.2 h |
| m06 surgical metrics | ~7 min | 28 min |
| m08b paired bootstrap | ~30 s | 2 min |
| **Total** | ~2.4 h | **~10 h** (~$8) |

---

## 🔀 Optional Task 5 — Delete superseded orchestrators (awaiting user Y/N)

These all become dead weight once `train.sh` + `eval.sh` land:

| 🗑️ Legacy | Superseded by |
|---|---|
| `scripts/run_iter9_10k.sh` | `train.sh` |
| `scripts/run_iter10_overnight.sh` | `train.sh` |
| `scripts/run_paired_eval_10k.sh` | `eval.sh` |
| `scripts/train_surgery.sh` | `train.sh` |

**Benefit**: removes 4 files hardcoding `ch11_surgery*.yaml` names → ends the rename blast radius. No one needs old orchestrators if new ones cover the same ground.

---

## ⚠️ Optional Task 6 — `scripts/tests_streaming/*` decision

Your memory rule **"D_I streaming path still in scope (deferred, not killed)"** says don't delete v15c yaml / tests_streaming during iter11 cleanups. After the rename, 5 files in `scripts/tests_streaming/` still reference `ch11_surgery_v15c.yaml`.

| Option | Action |
|---|---|
| **A** | 🔄 Update refs to new name `surgery_3stage_DI.yaml` — keeps tests runnable |
| **B** | 🟰 Leave broken — `grep` will catch them when D_I streaming work resumes |

📣 **Decision needed before Task 1 executes** (the rename).

---

## ✅ Verification checklist per task

### 🏷️ Task 1
- [ ] `ls configs/train/*.yaml` → explora + 3 surgery_* files present, no `ch11_*` or `v15*`
- [ ] `python3 -c "import yaml; yaml.safe_load(open('configs/train/surgery_2stage_noDI.yaml'))"` parses
- [ ] `grep -l ch11_surgery src/ scripts/` → only stale cosmetic comments (or empty if Task 5 deletes)
- [ ] `python3 -m py_compile src/m09c_surgery.py` OK

### 🚀 Task 3
- [ ] `shellcheck scripts/train.sh` → 0 errors
- [ ] Dry-run with SANITY: `bash scripts/train.sh --DRY` (add a flag) shows 4 planned invocations
- [ ] Each log path unique

### 📊 Task 4
- [ ] `shellcheck scripts/eval.sh` → 0 errors
- [ ] Variants list contains exactly 4 names matching Task 1 output dirs
- [ ] Frozen archive path still resolves to `outputs_versioned/frozen_eval10k/`
- [ ] Summary table reads from `outputs/full/<v>/eval10k/paired_bootstrap_results.json`

---

## 🧭 Execution order

```
Task 1 (rename)
   ↓
Task 2 (--output-dir is already supported — no code edit)
   ↓
Task 3 (train.sh — references new yaml names)
   ↓
Task 5 (delete old orchestrators — safe once train.sh exists)
   ↓
Task 4 (eval.sh — references new output dirs)
   ↓
Task 6 (tests_streaming decision)
   ↓
🚀 Launch: tmux new -s train ; ./scripts/train.sh ; ./scripts/eval.sh
```

---

## 📌 Standing questions (blockers)

1. ❓ **Task 5**: delete the 4 superseded orchestrator scripts? **Y/N**
2. ❓ **Task 6**: update `scripts/tests_streaming/*` refs to new yaml name? **A / B**

Once answered, tasks execute in the order above with verification per step.
