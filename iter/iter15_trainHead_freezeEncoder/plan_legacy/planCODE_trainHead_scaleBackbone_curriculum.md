# 🚀 iter15 — 24 GB Execution Plan: Head-Only Track on RTX Pro 4000

> 📅 **Date:** 2026-05-13
> 🎯 **Paper goal:** `vjepa_surgery ≫ vjepa_pretrain ≫ vjepa_frozen` on motion / temporal features
> 💻 **Hardware:** 24 GB VRAM (RTX Pro 4000, ~$0.20/hr)
> 🧊 **Why 24 GB fits:** head-only freeze → no encoder backward → no 1.84B-param activation storage → ViT-G fits comfortably

---

## 📋 Phase summary

```
┌────────────┬─────────────────────────────────────────────┬──────────┬──────────────┬─────────────┐
│ Phase       │ Work                                         │ GPU?     │ Wall          │ Cost (24GB) │
├────────────┼─────────────────────────────────────────────┼──────────┼──────────────┼─────────────┤
│ 1️⃣ RENAMES  │ git mv m09a/m09c → m09a1/m09c1               │ ❌ no    │ ~30 sec       │ $0          │
│ 2️⃣ CODE     │ Write m09a2 + m09c2 + probe_future_regress + │ ❌ no    │ ~6-8 hr dev   │ $0          │
│             │ utils/data_curriculum + 3 yamls + V1 lint    │          │               │             │
│ 3️⃣ M04D     │ Rebuild motion_features.npy 13→23 D          │ ✅ 24GB  │ ~2 hr         │ ~$0.40      │
│ 4️⃣ WIRING   │ Shell wrappers + probe_action.py Δ4-Δ7       │ ❌ no    │ ~1 hr dev     │ $0          │
│ 5️⃣ SANITY   │ V0/V1/V2/V3/V4/V5/V6 — code-correctness gate │ ✅ 24GB  │ ~50 min       │ ~$0.17      │
│ 6️⃣ POC      │ 3 cells × ~6-8 hr head-only training         │ ✅ 24GB  │ ~24 hr        │ ~$5         │
├────────────┼─────────────────────────────────────────────┼──────────┼──────────────┼─────────────┤
│ ⏳ 96 GB    │ DEFER — only encoder-update variants need 96GB│ ❌ defer │ (skip — iter14│ $0 (skip)   │
│ deferred    │ and those are already done in iter14         │          │  already done)│             │
└────────────┴─────────────────────────────────────────────┴──────────┴──────────────┴─────────────┘
                                                          TOTAL (Phases 1-6): ~33 hr · ~$5.57
```

---

## 1️⃣ Phase 1 — RENAMES (no GPU, ~30 sec) 🏷️  ✅ DONE 2026-05-13

**Goal:** rename current scripts + configs + shells to mark them as the encoder-update track, freeing canonical names for the head-only siblings. Drops misleading `probe_` prefix (these scripts run encoder + head training, not just probe eval).

### 🎯 Final naming convention (4-layer alignment)

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ OLD                                   │ NEW                                   │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ src/m09a_pretrain.py                  │ src/m09a1_pretrain_encoder.py         │
│ src/m09c_surgery.py                   │ src/m09c1_surgery_encoder.py          │
│ configs/train/probe_pretrain.yaml     │ configs/train/pretrain_encoder.yaml   │
│ (new this session, never tracked)     │ configs/train/pretrain_head.yaml      │
│ scripts/run_probe_train.sh            │ scripts/run_train.sh                  │
│ scripts/run_probe_eval.sh             │ scripts/run_eval.sh                   │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### 🔨 Steps (executed)

```bash
cd /workspace/factorjepa

# 🏷️ Rename current encoder-update scripts (no logic change)
git mv src/m09a_pretrain.py  src/m09a1_pretrain_encoder.py
git mv src/m09c_surgery.py   src/m09c1_surgery_encoder.py

# 🔧 Update every caller across the repo — shells, yamls, iter/ docs
sed -i 's|src/m09a_pretrain\.py|src/m09a1_pretrain_encoder.py|g' \
    scripts/*.sh iter/iter*/*.md configs/train/*.yaml 2>/dev/null

sed -i 's|src/m09c_surgery\.py|src/m09c1_surgery_encoder.py|g' \
    scripts/*.sh iter/iter*/*.md configs/train/*.yaml 2>/dev/null

# Also rename module-level references like `m09c_surgery` (without .py extension)
sed -i 's|\bm09a_pretrain\b|m09a1_pretrain_encoder|g; s|\bm09c_surgery\b|m09c1_surgery_encoder|g' \
    src/*.py src/utils/*.py 2>/dev/null

# 🧹 Final targeted sweep — active configs that the glob above missed
sed -i 's|m09a_pretrain\.py|m09a1_pretrain_encoder.py|g; s|m09c_surgery\.py|m09c1_surgery_encoder.py|g' \
    configs/pipeline.yaml \
    configs/train/base_optimization.yaml \
    configs/train/pretrain_encoder.yaml \
    configs/train/surgery_base.yaml \
    src/MEMORY.md

# 🧹 Two stale # comments in scripts/run_train.sh referencing the OLD script
# names (lines 243, 289) — hand-edited to match new filenames.

# 🏷️ Phase 1.5 — config + shell renames (drop probe_ prefix)
git mv configs/train/probe_pretrain.yaml configs/train/pretrain_encoder.yaml
mv     configs/train/probe_pretrain_head.yaml configs/train/pretrain_head.yaml   # plain mv: untracked new file
git mv scripts/run_probe_train.sh        scripts/run_train.sh
git mv scripts/run_probe_eval.sh         scripts/run_eval.sh

# 🔧 Update all active callers (~30 files across src/, scripts/, configs/, iter/iter15_*/)
sed -i \
    -e 's|probe_pretrain_head\.yaml|pretrain_head.yaml|g' \
    -e 's|probe_pretrain\.yaml|pretrain_encoder.yaml|g' \
    -e 's|run_probe_train\.sh|run_train.sh|g' \
    -e 's|run_probe_eval\.sh|run_eval.sh|g' \
    <files-active>
```

### 📌 Refs deliberately NOT touched

```
# Output-directory string `outputs/<mode>/m09a_pretrain/` and `m09c_surgery_*/`
# is decoupled from the script filename and remains stable so iter14 result
# paths under iter/iter14_surgery_on_pretrain/result_outputs/v14b/poc/ keep
# resolving. Echo banner labels ("m09a continual SSL ...") are display strings.
# scripts/legacy*/, configs/legacy*/, src/legacy/, iter/iter{8,9,10,12,13}/
# and *.log files are immutable history.
```

### ✅ Verification

```bash
# 🚨 Expect ZERO matches for the OLD names anywhere
grep -rnE 'm09a_pretrain\.py|m09c_surgery\.py' \
    src/ scripts/ configs/ iter/ 2>/dev/null
# (empty output = success)

# ✅ Confirm new files exist
ls -la src/m09a1_pretrain_encoder.py src/m09c1_surgery_encoder.py
```

### 🎯 Pass criteria

- ✅ `git status` shows 2 renames (m09a → m09a1, m09c → m09c1)
- ✅ grep returns ZERO hits for `m09a_pretrain.py` or `m09c_surgery.py`
- ✅ `ls` confirms both new filenames exist

---

## 2️⃣ Phase 2 — NEW PYTHON + YAML FILES (no GPU, ~6-8 hr dev) 🆕  ✅ 8/8 DONE 2026-05-14

**Goal:** create the 7 new files that implement head-only training + future-prediction regressor + data curriculum.

### 📄 Files to create

```
┌──────────────────────────────────────────────────┬────────┬──────────────────────────────────────┐
│ File                                              │ LoC    │ Purpose / Status                     │
├──────────────────────────────────────────────────┼────────┼──────────────────────────────────────┤
│ src/m09a2_pretrain_head.py                        │  568   │ 🧠 frozen encoder + predictor    ✅ │
│ src/m09c2_surgery_head.py                         │  643   │ 🔬 same freeze + factor-aug      ✅ │
│ src/probe_future_regress.py                       │  532   │ 🔮 future-prediction probe       ✅ │
│ src/utils/data_curriculum.py                      │   95   │ 📚 sort_by_fg_magnitude + pacing ✅ │
│ src/utils/training.py (add 1 helper)              │  +20   │ 🚨 assert_encoder_frozen() guard ✅ │
├──────────────────────────────────────────────────┼────────┼──────────────────────────────────────┤
│ configs/train/pretrain_head.yaml                  │   85   │ 📄 yaml for m09a2                ✅ │
│ configs/train/surgery_3stage_DI_head.yaml         │   87   │ 📄 yaml for m09c2 (D_I variant)  ✅ │
│ configs/train/surgery_2stage_noDI_head.yaml       │   86   │ 📄 yaml for m09c2 (noDI variant) ✅ │
└──────────────────────────────────────────────────┴────────┴──────────────────────────────────────┘
                                                   TOTAL Python: 1858 LoC · TOTAL yaml: 258 lines
```

### 📊 Phase 2 progress snapshot (2026-05-14 — ALL 8 deliverables landed)

```
┌─────────────────────────────────────────────────────────────────────┐
│ ✅ DONE — 8/8 deliverables, all 3-check-gate clean                   │
│   src/utils/training.py             assert_encoder_frozen() +20 LoC  │
│   src/utils/data_curriculum.py      95 LoC                           │
│   src/m09a2_pretrain_head.py        568 LoC                          │
│   src/m09c2_surgery_head.py         643 LoC                          │
│   src/probe_future_regress.py       532 LoC                          │
│   configs/train/pretrain_head.yaml                  85 lines         │
│   configs/train/surgery_3stage_DI_head.yaml         87 lines         │
│   configs/train/surgery_2stage_noDI_head.yaml       86 lines         │
│                                                                      │
│   3-check gate (py_compile + ast.parse + ruff F+E9) PASSES on:      │
│     - src/m09a2_pretrain_head.py                                     │
│     - src/m09c2_surgery_head.py                                      │
│     - src/probe_future_regress.py                                    │
│     - src/utils/training.py                                          │
│     - src/utils/data_curriculum.py                                   │
│                                                                      │
│ 🎁 BONUS — Phase 1.5 renames (4-layer alignment, drop probe_)        │
│   probe_pretrain.yaml      → pretrain_encoder.yaml                  │
│   probe_pretrain_head.yaml → pretrain_head.yaml                     │
│   run_probe_train.sh       → run_train.sh                           │
│   run_probe_eval.sh        → run_eval.sh                            │
│   surgery_3stage_DI.yaml   → surgery_3stage_DI_encoder.yaml         │
│   surgery_2stage_noDI.yaml → surgery_2stage_noDI_encoder.yaml       │
│   0 active OLD refs after all renames                                │
│                                                                      │
│ 🎁 BONUS — utils/cgroup_monitor.py (new, 237 LoC) wired into 8       │
│   producer-consumer scripts (m04, m04d, m05, m05b, m05c, m09a1,     │
│   probe_action, probe_future_mse) for forensic OOM SIGKILL trail    │
└─────────────────────────────────────────────────────────────────────┘
```

### 🎯 m09a2 / m09c2 / probe_future_regress — landed 2026-05-14

```
┌────────────────────────────────────────────────────────────────────┐
│ Shared CLAUDE.md compliance (all 3 Python files)                    │
├────────────────────────────────────────────────────────────────────┤
│ ✅ Gold-standard URL cited in docstring                              │
│ ✅ All imports at TOP (OMP env vars before torch import)            │
│ ✅ USAGE block with SANITY/POC/FULL — all paths required             │
│ ✅ No DEFAULT, no hardcoded paths in code                            │
│ ✅ cfg[key] indexing (no .get with default)                          │
│ ✅ No getattr(args, key, default)                                    │
│ ✅ No cross-imports from m*.py (only utils/)                         │
│ ✅ FAIL HARD on every prereq violation                               │
│ ✅ cache-policy interactive prompt                                   │
│ ✅ cgroup_monitor wired (print_cgroup_header + start_oom_watchdog)  │
│ ✅ tqdm via make_pbar                                                │
│ ✅ POC↔FULL parity                                                   │
├────────────────────────────────────────────────────────────────────┤
│ Per-file specifics                                                  │
├────────────────────────────────────────────────────────────────────┤
│ m09a2_pretrain_head.py                                              │
│   data:        producer_thread (raw clips, same as m09a1)           │
│   training:    1 stage, all 48 blocks frozen, motion_aux loss only │
│   outputs:     student_encoder.pt + m09a_ckpt_best.pt +             │
│                motion_aux_head.pt + training_log.jsonl              │
│                                                                      │
│ m09c2_surgery_head.py                                               │
│   data:        StreamingFactorDataset (factor-aug clips per         │
│                mode_mixture from yaml stages[0])                    │
│   training:    1 stage (validated in merge_config_with_args),       │
│                set_trainable_prefix(student, 0), assert_encoder_    │
│                frozen()                                              │
│   variant_tag: from yaml's adapted_encoder → output_dir gets        │
│                /<variant>/ suffix so 3stage_DI_head + noDI_head     │
│                write to separate subdirs                            │
│   val cycle:   RAW clips (factor-aug only at train time)            │
│   outputs:     same triple as m09a2 + stage_name + mode_mixture     │
│                in m09c_ckpt_best.pt                                 │
│                                                                      │
│ probe_future_regress.py                                             │
│   stages:      forward (per encoder) + paired_per_variant           │
│   ctx/tgt:     ctx = enc(x[0:8]), tgt = enc(x[8:16]),               │
│                both under no_grad on FROZEN encoder                  │
│   regressor:   {linear, mlp_d1, mlp_d2}, AdamW(lr=1e-3, wd=0.05),  │
│                cosine, 50 epochs, L1 stop-grad target               │
│   --data-source: {raw, factor_aug} selects train-time clip source  │
│   KNOWN_VARIANTS: 7 (4 iter14 encoder-update + 3 iter15 head-only)  │
│   outputs:     per_clip_regressor_l1.npy + clip_keys.npy +         │
│                aggregate_regressor_l1.json + regressor.pt           │
│                aggregate format compatible with probe_future_mse    │
│                paired_per_variant flow                              │
└────────────────────────────────────────────────────────────────────┘
```

### 🔨 Reference for content

Full file-level specs (function signatures, pseudocode, yaml blocks) live in
[`plan_trainHead_scaleBackbone_curriculum.md`](./plan_trainHead_scaleBackbone_curriculum.md) — Phase 2 + Phase 2b sections.

### 🧪 V1 — static lint after writing all files

```bash
python -c "
import py_compile
files = [
    'src/m09a2_pretrain_head.py',
    'src/m09c2_surgery_head.py',
    'src/probe_future_regress.py',
    'src/utils/data_curriculum.py',
    'src/utils/training.py',
]
for f in files:
    py_compile.compile(f, doraise=True)
    print(f'✅ {f}')
print('✅ all py_compile clean')
"

# Also ruff (matches src/CLAUDE.md 3-check gate)
ruff check --select F,E9 \
    src/m09a2_pretrain_head.py \
    src/m09c2_surgery_head.py \
    src/probe_future_regress.py \
    src/utils/data_curriculum.py \
    src/utils/training.py
```

### 🎯 Pass criteria

- ✅ All 8 files exist with reasonable LoC counts
- ✅ py_compile passes on all 5 .py files
- ✅ ruff F + E9 checks pass

---

## 3️⃣ Phase 3 — M04D MOTION FEATURES 13→23-D (24 GB GPU, ~2 hr) 🧬  ✅ DONE 2026-05-14

**Goal:** extend `m04d_motion_features.py` to output 23-D features (adds camera-subtracted FG motion). Powers BOTH richer motion-class labels AND principled data curriculum sort axis (`vec[13] = fg_mean_mag`).

### 📊 Phase 3 status snapshot (2026-05-14)

```
┌─────────────────────────────────────────────────────────────────────┐
│ ✅ DONE — code edits across 5 files (lint clean)                     │
│   src/m04d_motion_features.py    FEATURE_DIM 13→23,                  │
│                                  _aggregate_flow +10 FG dims         │
│                                  + cache-policy=2 wipes checkpoint   │
│                                  + OMP_NUM_THREADS=1 caps preamble   │
│   src/utils/action_labels.py     parse_optical_flow_class on vec[13],│
│                                  compute_magnitude_quartiles + guard │
│   src/utils/motion_aux_loss.py   n_motion_dims auto-derived from     │
│                                  vec_mean.numel() + ≥23 guard        │
│   src/utils/eval_subset.py       docstring 13D → 23D                 │
│   src/m09c1_surgery_encoder.py   vec13d → vec_motion rename          │
│                                                                      │
│ ✅ GPU RERUN COMPLETED — m04d ran cleanly on 120 GB cgroup instance   │
│   Wall:  1h 56m (6975 sec on RTX Pro 4000 + 48 cores)                │
│   Cost:  ~$0.40                                                      │
│   Output: data/eval_10k_local/motion_features.npy  shape (9297, 23)  │
│           data/eval_10k_local/motion_features.paths.npy              │
│           data/eval_10k_local/motion_features.meta.json (moved from  │
│              outputs/ → data/ for HF upload-data ride-along)         │
│   Errors: 0 (GPU: 0, producer: 0, skipped: 0)                       │
│   vec[13] fg_mean_mag range: [0.801, 845.292] — non-degenerate       │
│                                                                      │
│ 🎁 BONUS — pipeline.yaml scaling table added                          │
│   producer_queue_motion: 64 → 16 (each queue slot ≈ 2.3 GB; queue=64 │
│   hit 120 GB cgroup cap exactly at clip 64 → SIGKILL). New comment   │
│   block documents the cgroup-memory scaling table (4 tiers, ≤36 GB   │
│   to ≥128 GB).                                                       │
│                                                                      │
│ ⏳ NEXT — regenerate action labels (CPU, ~5 sec) on the new 23-D     │
│   features so probe_action uses vec[13] FG binning. Deferred to     │
│   Stage E of planCODE_html.md (sequenced with HTML refresh).        │
└─────────────────────────────────────────────────────────────────────┘
```

### 🔨 Code edits (~10-15 LoC across 3 files)

```
┌──────────────────────────────────┬───────────────────────────────────────────────────────┐
│ File                              │ Change                                                 │
├──────────────────────────────────┼───────────────────────────────────────────────────────┤
│ src/m04d_motion_features.py        │ Extend _aggregate_flow (L217-261) to add 10 dims:    │
│                                   │  fg_mean_mag, fg_max_mag, fg_dir_hist (8 bins)        │
│                                   │ Update FEATURE_DIM 13 → 23                            │
│ src/utils/action_labels.py         │ parse_optical_flow_class (L66-115) — bin on vec[13]   │
│                                   │  (fg_mean_mag) instead of vec[0] (global mean_mag)    │
│ src/utils/motion_aux_loss.py       │ MotionAuxHead.n_motion_dims 13 → 23 (L75-100)         │
└──────────────────────────────────┴───────────────────────────────────────────────────────┘
```

### 🚀 Rerun m04d to regenerate motion_features.npy

```bash
# 🧹 Pre-flight: confirm current m04d output is still 13-D
python -c "import numpy as np; \
  feats = np.load('data/eval_10k_local/motion_features.npy'); \
  print(f'before Phase 3: shape={feats.shape}')"
# expect: (9297, 13)

# 🔥 Rebuild 23-D features on eval_10k (~2 hr on 24 GB RTX Pro 4000)
CACHE_POLICY_ALL=2 python -u src/m04d_motion_features.py --FULL \
    --subset data/eval_10k.json \
    --local-data data/eval_10k_local \
    --features-out data/eval_10k_local/motion_features.npy \
    --no-wandb 2>&1 | tee logs/iter15_phase3_m04d_13to23D.log
```

### 🧪 V0 — shape + FG-magnitude sanity check (~5 sec)

```bash
python -c "
import numpy as np
feats = np.load('data/eval_10k_local/motion_features.npy')
assert feats.shape[1] == 23, f'❌ shape mismatch: got {feats.shape[1]}, expected 23'
fg_mag = feats[:, 13]
print(f'✅ Phase 3 features ready')
print(f'   shape:                 {feats.shape}')
print(f'   vec[13] fg_mean_mag:   [{fg_mag.min():.4f}, {fg_mag.max():.4f}]')
print(f'   FG quartile boundaries: q1={np.percentile(fg_mag, 25):.4f}, '
      f'q2={np.percentile(fg_mag, 50):.4f}, q3={np.percentile(fg_mag, 75):.4f}')
"
```

### 🎯 Pass criteria

- ✅ Shape is `(9297, 23)`
- ✅ vec[13] (fg_mean_mag) has reasonable range (non-zero spread)
- ✅ FG quartile boundaries differ from global quartile boundaries (proves camera-subtraction is doing work)

---

## 4️⃣ Phase 4 — WIRING (no GPU, ~1 hr) 🔌  ✅ DONE 2026-05-14

**Goal:** make the new files reachable from the shell wrappers + the paired-Δ aggregator.

### 📊 Phase 4 status snapshot (2026-05-14)

```
┌────────────────────────────────────────────────────────────────────┐
│ ✅ DONE — 3 wiring layers (lint clean, shellcheck-style parse OK)   │
│                                                                      │
│ 1. scripts/run_train.sh                                              │
│    - SUBCMD allow-list extended: pretrain_head,                      │
│      surgery_3stage_DI_head, surgery_noDI_head                       │
│    - 3 new dispatch branches added (each ~25 LoC):                   │
│      pretrain_head            → src/m09a2_pretrain_head.py           │
│      surgery_3stage_DI_head   → src/m09c2_surgery_head.py            │
│      surgery_noDI_head        → src/m09c2_surgery_head.py            │
│    - FULL_CKPT case extended to recognize the 3 new SUBCMDs          │
│    - USAGE line updated                                              │
│                                                                      │
│ 2. scripts/run_eval.sh                                               │
│    - ENCODERS default list extended with 3 new variants:             │
│      vjepa_2_1_pretrain_head                                         │
│      vjepa_2_1_surgical_3stage_DI_head                               │
│      vjepa_2_1_surgical_noDI_head                                    │
│    - encoder_ckpt_for() + encoder_predictor_ckpt_for() each got 3   │
│      new cases pointing at outputs/{mode}/m09{a,c}_*_head/.../*.pt  │
│                                                                      │
│ 3. src/probe_action.py                                               │
│    - ITER14_DELTAS extended from 3 → 7 entries:                      │
│      Δ4: pretrain vs pretrain_head                                  │
│      Δ5: surgery vs surgery_head    ⭐ KEY iter15 PAPER CLAIM         │
│      Δ6: surgery_head vs pretrain_head                              │
│      Δ7: 3stage_DI_head vs noDI_head                                │
└────────────────────────────────────────────────────────────────────┘
```

### 🎯 Pass criteria (all met)

- ✅ `bash -n scripts/run_train.sh` and `bash -n scripts/run_eval.sh` parse OK
- ✅ `src/probe_action.py` 3-check gate: py_compile + ast.parse + ruff F+E9 clean
- ✅ USAGE line: `pretrain|pretrain_2X|pretrain_head|surgery_3stage_DI|surgery_noDI|surgery_3stage_DI_head|surgery_noDI_head`
- ✅ All 3 new dispatch branches present (`grep -q '    pretrain_head)'` etc.)
- ✅ All 3 new encoders registered in both `encoder_ckpt_for` and `encoder_predictor_ckpt_for`
- ✅ ITER14_DELTAS contains exactly 7 entries (Δ1-Δ7)

---

## 5️⃣ Phase 5 — SANITY VALIDATION (24 GB GPU, ~50 min total) 🧪

**Goal:** end-to-end code-correctness check on 24 GB. NOT a model-quality check (per src/CLAUDE.md SANITY semantics).

### 🧪 V2 — m09a2 SANITY (~5 min)

```bash
./scripts/run_train.sh pretrain_head --SANITY 2>&1 \
  | tee logs/iter15_v2_m09a2_sanity.log
```

**🎯 Pass:**
- ✅ Log contains `[m09a2 STRICT HEAD-ONLY]` banner
- ✅ `assert_encoder_frozen` passes (no FATAL exit)
- ✅ `outputs/sanity/m09a_pretrain_head/{student_encoder.pt, m09a_ckpt_best.pt}` produced

### 🧪 V3 — m09c2 SANITY both variants (~15 min total)

```bash
./scripts/run_train.sh surgery_3stage_DI_head --SANITY 2>&1 \
  | tee logs/iter15_v3a_m09c2_3stage_DI_head_sanity.log

./scripts/run_train.sh surgery_noDI_head --SANITY 2>&1 \
  | tee logs/iter15_v3b_m09c2_noDI_head_sanity.log
```

**🎯 Pass:**
- ✅ Each produces `student_encoder.pt` + `m09c_ckpt_best.pt`
- ✅ Log shows single-stage loop (no Stage 1/2/3 transitions — encoder frozen always)

### 🧪 V4 — Encoder invariance check (~10 sec, CPU)

```bash
python -c "
import torch
from pathlib import Path

meta = torch.load('checkpoints/vjepa2_1_vitG_384.pt', map_location='cpu')

for variant in ['m09a_pretrain_head',
                'm09c_surgery_3stage_DI_head',
                'm09c_surgery_noDI_head']:
    p = Path(f'outputs/sanity/{variant}/student_encoder.pt')
    if not p.exists():
        print(f'⚠️  {variant}: not produced')
        continue
    head_out = torch.load(p, map_location='cpu')
    n_match, n_total = 0, 0
    for k in head_out:
        if k.startswith('blocks.'):
            n_total += 1
            if k in meta and torch.allclose(head_out[k], meta[k]):
                n_match += 1
    print(f'{variant}: {n_match}/{n_total} blocks identical to Meta')
    assert n_match == n_total, f'❌ {variant} block params drifted'
print('✅ all 3 head-only variants have bit-identical encoder weights to Meta')
"
```

**🎯 Pass:** encoder block params bit-identical to Meta init for all 3 head variants.

### 🧪 V5 — probe_future_regress SANITY (~5 min)

```bash
python -u src/probe_future_regress.py --SANITY \
    --stage forward \
    --variant vjepa_2_1_frozen \
    --encoder-ckpt checkpoints/vjepa2_1_vitG_384.pt \
    --data-source raw \
    --regressor-arch linear \
    --action-probe-root outputs/sanity/probe_action \
    --local-data data/eval_10k_local \
    --output-root outputs/sanity/probe_future_regress \
    --cache-policy 2 2>&1 | tee logs/iter15_v5_probe_future_regress_sanity.log
```

**🎯 Pass:** `outputs/sanity/probe_future_regress/vjepa_2_1_frozen/` contains
`per_clip_regressor_l1.npy` + `aggregate_regressor_l1.json` + `regressor.pt`

### 🧪 V6 — Full 7-variant eval SANITY (~20 min)

```bash
./scripts/run_eval.sh --sanity 2>&1 \
  | tee logs/iter15_v6_full_eval_sanity.log
```

**🎯 Pass:**
- ✅ STAGE 4 paired_delta emits `probe_paired_delta.json` with Δ4/Δ5/Δ6/Δ7 keys
- ✅ STAGE 9b emits `probe_future_regress_per_variant.json`
- ✅ ZERO STAGE failures across all 7 variants

### 🚨 HARD STOP — review V0-V6 results before Phase 6

If any V step FAILS on 24 GB:
- 🔥 **OOM**: that's a 24 GB ceiling we did NOT predict. Investigate before scaling.
- 🔥 **Assertion failure**: encoder is not actually frozen — fix the freeze wiring.
- 🔥 **Missing output file**: scripts/yamls have a path mismatch — fix wiring.

If all pass: proceed to Phase 6.

---

## 6️⃣ Phase 6 — POC HEAD-ONLY TRAINING (24 GB GPU, ~24 hr) 🚀

**Goal:** produce the 3 NEW head-only encoder variants for the 13-variant paper figure.

### 🔨 Three independent POC runs (run serially or in 3 separate tmux sessions)

```bash
# 🧠 Cell 1: m09a2 head-only continual pretrain (~6-8 hr on 24 GB)
CACHE_POLICY_ALL=2 ./scripts/run_train.sh pretrain_head --POC 2>&1 \
  | tee logs/iter15_poc_m09a2_pretrain_head.log

# 🔬 Cell 2: m09c2 head-only surgery (3stage_DI) (~6-8 hr)
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_3stage_DI_head --POC 2>&1 \
  | tee logs/iter15_poc_m09c2_3stage_DI_head.log

# 🔬 Cell 3: m09c2 head-only surgery (noDI) (~6-8 hr)
CACHE_POLICY_ALL=2 ./scripts/run_train.sh surgery_noDI_head --POC 2>&1 \
  | tee logs/iter15_poc_m09c2_noDI_head.log
```

### 🎯 Pass criteria

- ✅ All 3 cells produce `student_encoder.pt` (bit-identical to Meta) + `m09c_ckpt_best.pt` (head trained)
- ✅ Log shows train_loss decreasing across epochs
- ✅ Final eval probe top1 ≥ 0.808 anchor baseline (else investigate)

---

## ⏳ DEFERRED — Requires 96 GB Blackwell

```
┌──────────────────────────────────────────────┬─────────────────────────────────────────────────┐
│ Workload                                      │ Why 96 GB                                        │
├──────────────────────────────────────────────┼─────────────────────────────────────────────────┤
│ 🔥 Re-run m09a1 encoder-update pretrain FULL  │ ViT-G encoder + fp32 AdamW optim ≈ 31-36 GB     │
│   (NOT NEEDED — iter14 already produced)      │   → won't fit 24 GB at BS=32                    │
│ 🔥 Re-run m09c1 encoder-update surgery FULL   │ Same VRAM ceiling                                │
│   (NOT NEEDED — iter14 already produced)      │                                                  │
│ 🆕 NEW encoder-update variants beyond iter14  │ Same — would need 96 GB                          │
│   (none planned in iter15)                    │                                                  │
└──────────────────────────────────────────────┴─────────────────────────────────────────────────┘
```

---

## 💰 Total budget recap

```
┌──────────────────────────────────────────────┬──────────────┬─────────────────┐
│ Phase                                          │ Wall          │ Cost @ $0.20/h  │
├──────────────────────────────────────────────┼──────────────┼─────────────────┤
│ 1️⃣ Renames                                    │ ~30 sec       │ $0              │
│ 2️⃣ Code + yaml dev                            │ ~6-8 hr (dev) │ $0              │
│ 3️⃣ m04d 13→23-D rerun                         │ ~2 hr GPU     │ ~$0.40          │
│ 4️⃣ Wiring                                     │ ~1 hr dev     │ $0              │
│ 5️⃣ SANITY V0-V6                               │ ~50 min GPU   │ ~$0.17          │
│ 6️⃣ POC head-only × 3 cells                    │ ~24 hr GPU    │ ~$5             │
├──────────────────────────────────────────────┼──────────────┼─────────────────┤
│ 🎯 TOTAL (Phases 1-6 on 24 GB Pro 4000)       │ ~33 hr        │ ~$5.57          │
└──────────────────────────────────────────────┴──────────────┴─────────────────┘
```

> 🏁 **24 GB strategy unlocks ~12× compute savings** vs running everything on 96 GB Blackwell.
> Only the iter14 encoder-update variants (already produced) needed 96 GB. All iter15 NEW work fits 24 GB.
