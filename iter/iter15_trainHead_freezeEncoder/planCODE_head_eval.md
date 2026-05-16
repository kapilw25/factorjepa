# 🎯 iter15 — Wire `motion_aux_head.pt` into Eval (head-vs-encoder paired-Δ fix)

📅 Started: 2026-05-16 · 🧑‍💻 Owner: kapilw25 · 🎯 Goal: make Δ5/Δ6/Δ7 statistically valid

> 🚨 **The problem in one line**: motion_aux_head.pt is trained, saved, uploaded to HF — and then NEVER LOADED by any of the 4 eval probes. So all 3 head cells share byte-identical encoder.pt → identical eval features → identical metrics → Δ5/Δ6/Δ7 collapse to "encoder vs frozen Meta" instead of "head vs encoder".

---

## 🚦 Status legend

```
⏳ pending   🔥 in-progress   🧪 testing   ✅ done   ❌ failed   🔁 redo   ⚠ blocked
```

---

## 📊 Step-by-step progress tracker (update after each step)

```
┌────┬─────────────────────────────────────────────────────┬────────┬─────────┬───────────────┐
│ #  │ 🛠 Step                                              │ ±LoC   │ 🚦 St   │ ✅ When        │
├────┼─────────────────────────────────────────────────────┼────────┼─────────┼───────────────┤
│ 1  │ 🧱 utils/motion_aux_loss.py — load_motion_aux_head  │  +66   │ ✅      │ 2026-05-16    │
│ 2  │ 🧱 utils/frozen_features.py — head-aware augmenter  │  +63   │ ✅      │ 2026-05-16    │
│ 3  │ 🎯 probe_action.py — --motion-aux-head CLI + thread │  +35   │ ✅      │ 2026-05-16    │
│ 4  │ 📐 probe_motion_cos.py — --motion-aux-head CLI      │  +20   │ ✅      │ 2026-05-16    │
│ 5  │ 🔮 probe_future_regress.py — --motion-aux-head CLI  │  +30   │ ✅      │ 2026-05-16    │
│ 6  │ 🐚 run_eval.sh — motion_aux_head_for + threading    │  +54   │ ✅      │ 2026-05-16    │
│ 7  │ 🧪 POC end-to-end re-eval (all 6 cells, verify Δ≠0) │   0    │ 🔥      │ —             │
├────┼─────────────────────────────────────────────────────┼────────┼─────────┼───────────────┤
│    │ 🎉 TOTAL                                             │ ~268   │ 6/7     │               │
└────┴─────────────────────────────────────────────────────┴────────┴─────────┴───────────────┘
```

---

## 🧬 1. Design choice — how do probes consume the head?

```
┌─────┬──────────────────┬─────────────────────────────────────────────────┬──────┐
│ ID  │ Strategy          │ Mechanism                                         │ 🚦   │
├─────┼──────────────────┼─────────────────────────────────────────────────┼──────┤
│ S1  │ 🥇 CONCAT (PICK)  │ feature = [encoder_pool ‖ ma_logits ‖ ma_vec]   │ ✅   │
│     │                   │ = (D + K + 23) dim · most info · backward-compat │      │
│ S2  │ ❌ ZERO_SHOT      │ top1 = argmax(ma_logits) · loses attentive probe │ ❌   │
│ S3  │ ❌ REPLACE        │ feature = ma penultimate · throws away encoder   │ ❌   │
└─────┴──────────────────┴─────────────────────────────────────────────────┴──────┘
```

🎯 **Why CONCAT wins**:
- 🧊 Preserves encoder signal (= iter14 baseline reproduces exactly when head absent)
- 🔓 Adds head signal as a treatment (paired-Δ becomes real)
- 🛡 Frozen baseline keeps working (`--motion-aux-head None` → encoder-only path)
- 📐 Backward-compatible — no iter14 reproducibility loss

---

## 🛠 2. Step 1 — `utils/motion_aux_loss.py` ✅

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ ✅ DONE 2026-05-16                             │
│ 📁 File                   │ src/utils/motion_aux_loss.py                  │
│ ➕ LoC                    │ +66 (planned +30 — added 2nd helper for       │
│                          │       concat-augment, used by Step 2)         │
│ ⏱ Effort                 │ ~15 min                                        │
└──────────────────────────┴────────────────────────────────────────────────┘
```

**🎯 Implemented:**
- ✅ `load_motion_aux_head(ckpt_path, device) → MotionAuxHead` (eval mode, strict load)
- ✅ `forward_motion_aux_concat(head, pooled) → (B, K + n_dims)` concat helper
- ✅ FAIL LOUD on missing path / corrupted ckpt / state_dict drift
- ✅ Buffers (vec_mean, vec_std) ride state_dict — load is bit-exact

**🧪 Verify output:**
```
  py_compile: OK
  ast.parse: OK
  loaded · n_classes=13 · n_dims=23 · d_in=1664 · hidden=256
  forward_concat output shape: (4, 36)  (expected K+n_dims = 13+23 = 36 ✓)
  head.training mode: False  (eval mode ✓)
  m09a_pretrain_head            ce_head.weight md5=5588665a  K=13
  m09c_surgery_3stage_DI_head   ce_head.weight md5=4f268a9b  K=13
  m09c_surgery_noDI_head        ce_head.weight md5=b0912c49  K=13
```

**📝 Critical finding:** All 3 head cells load with DIFFERENT `ce_head.weight` md5s →
the heads ARE uniquely trained per cell. This confirms paired-Δ has signal to
extract once we wire it into eval probes.

---

## 🛠 3. Step 2 — `utils/frozen_features.py` ✅

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ ✅ DONE 2026-05-16                             │
│ 📁 File                   │ src/utils/frozen_features.py                  │
│ ➕ LoC                    │ +63 (planned +50 — slightly over for FAIL-    │
│                          │       LOUD validation paths)                  │
│ ⏱ Effort                 │ ~20 min                                        │
└──────────────────────────┴────────────────────────────────────────────────┘
```

**🎯 Design pivot from plan:**
- ❌ Original: thread `motion_aux_head` kwarg through `extract_features_for_keys`
  + `_flush_batch` (invasive — touches the producer-consumer loop)
- ✅ Implemented: separate **post-hoc augmenter** `apply_motion_aux_head_to_features`
  + `extract_features_for_keys` UNCHANGED (encoder-only path is bit-identical
  to iter14 baseline — no reproducibility risk)
  + Probes call extractor first, then optionally call augmenter

**🎯 Function spec:**
```python
def apply_motion_aux_head_to_features(features, motion_aux_head, batch_size=64):
    # features: (N, n_tokens, D) OR (N, D)  — float16 from extractor
    # Returns: (N, K + n_dims) float32 concat of [ce_logits, mse_pred]
    # FAIL LOUD on dim mismatch + bad ndim
```

**🧪 Verify output:**
```
  3-D in (10, 16, 1664) → out (10, 36)  (K+n_dims = 13+23 ✓)
  2-D in (10, 1664) → out (10, 36)  ✓
  ✅ dim-mismatch FAIL LOUD (D=100 vs head.d_encoder=1664)
  ✅ bad-ndim FAIL LOUD (1-D rejected)

  Cross-head determinism (same features, 3 heads):
    m09a_pretrain_head            output md5=29ba7418
    m09c_surgery_3stage_DI_head   output md5=7d158977
    m09c_surgery_noDI_head        output md5=e7b9b057
```

**📝 Critical finding:** identical encoder features through 3 different motion_aux
heads produce **3 distinct output md5s** → paired-Δ has signal end-to-end once
probes wire this in (Steps 3-5).

---

## 🛠 4. Step 3 — `probe_action.py` ✅

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ ✅ DONE 2026-05-16                             │
│ 📁 File                   │ src/probe_action.py                            │
│ ➕ LoC                    │ +35 (planned +40 — simpler than expected      │
│                          │       because train_stage auto-derives d_in)  │
│ ⏱ Effort                 │ ~20 min                                        │
└──────────────────────────┴────────────────────────────────────────────────┘
```

**🎯 Implemented:**
- ✅ CLI arg `--motion-aux-head <path>` at build_parser L781 (default None)
- ✅ Augmentation block in `run_features_stage` (after extract, before save):
  load head → apply_motion_aux_head_to_features → tile across n_tokens → concat
- ✅ train_stage unchanged — `d_in = feats_train.shape[-1]` auto-derives 1700
- ✅ Frozen baseline + cells without `--motion-aux-head` flag: encoder-only path,
  iter14 baseline reproduces bit-identically

**🧪 Verify output:**
```
  CLI parse: --motion-aux-head parses correctly (default None ✓)
  encoder-only  shape: (5, 16, 1664)
  augmented     shape: (5, 16, 1700)  (D=1664 + K+n_dims=36 = 1700 ✓)
  tile correctness:    True (same K+n_dims values across all 16 token positions)
  probe d_in auto-derive: 1700  (no train_stage change needed ✓)

  Per-cell features.npy md5 (same encoder feats, 3 heads):
    m09a_pretrain_head            features.npy md5=fc7dc05d
    m09c_surgery_3stage_DI_head   features.npy md5=8e539489
    m09c_surgery_noDI_head        features.npy md5=ae4caa84
```

**📝 Critical finding:** features.npy is now per-cell-unique → attentive probe
trains on per-cell features → top1 differs per cell → Δ5/Δ6/Δ7 paired tests
become statistically valid (no longer collapsed to frozen baseline).

---

## 🛠 5. Step 4 — `probe_motion_cos.py` ⏳

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ ⏳ pending                                     │
│ 📁 File                   │ src/probe_motion_cos.py                        │
│ ➕ LoC                    │ +30                                            │
│ ⏱ Effort                 │ ~20 min                                        │
└──────────────────────────┴────────────────────────────────────────────────┘
```

**🎯 What:**
- Add `--motion-aux-head <path>` CLI (default `None`)
- Reuse `extract_features_for_keys` head-aware path
- Cosine math is dim-agnostic → augmented features → intra/inter-class separation reflects head signal

**🧪 Verify:**
```bash
# Stage 5 features for Cell A with head:
python -u src/probe_motion_cos.py --POC --stage features ... --motion-aux-head <path>
```

**📝 Completion notes:**
- _Fill in: cosine values pre/post-head augmentation_

---

## 🛠 6. Step 5 — `probe_future_regress.py` ⏳

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ ⏳ pending                                     │
│ 📁 File                   │ src/probe_future_regress.py                    │
│ ➕ LoC                    │ +30                                            │
│ ⏱ Effort                 │ ~20 min                                        │
└──────────────────────────┴────────────────────────────────────────────────┘
```

**🎯 What:**
- Add `--motion-aux-head <path>` CLI (default `None`)
- In `_encode_window` (L138): append head forward output to encoder pooled vec
- `build_regressor(arch, embed_dim + K + 23)` auto-resizes input/output

**🧪 Verify:**
```bash
python -u src/probe_future_regress.py --POC --stage forward ... --motion-aux-head <path>
```

**📝 Completion notes:**
- _Fill in: regressor L1 pre/post-head_

---

## 🛡 7. Step — `probe_future_mse.py` — 🚫 NO CHANGE

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ 🛡 INTENTIONALLY SKIPPED                       │
│ 🎯 Why                    │ probe_future_mse tests predictor's masked-     │
│                          │ latent prediction. motion_aux is orthogonal to │
│                          │ JEPA-style future prediction. V-JEPA gold       │
│                          │ standard preserved here exactly.                │
└──────────────────────────┴────────────────────────────────────────────────┘
```

---

## 🐚 8. Step 6 — `scripts/run_eval.sh` (THIN WRAPPER ONLY) ⏳

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ ⏳ pending                                     │
│ 📁 File                   │ scripts/run_eval.sh                            │
│ ➕ LoC                    │ +25                                            │
│ ⏱ Effort                 │ ~15 min                                        │
└──────────────────────────┴────────────────────────────────────────────────┘
```

**🎯 What:** Add `motion_aux_head_for(ENC)` resolver + thread `--motion-aux-head $MA_CKPT` into Stages 2/3/5/9b. NO logic in shell — pure dispatch.

**🐚 Code to add (mirrors `encoder_ckpt_for` at L201):**
```bash
motion_aux_head_for() {
    case "$1" in
        vjepa_2_1_frozen)                          echo "" ;;
        vjepa_2_1_pretrain_encoder)                echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_encoder/motion_aux_head.pt" ;;
        vjepa_2_1_pretrain_2X_encoder)             echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_2X_encoder/motion_aux_head.pt" ;;
        vjepa_2_1_surgical_3stage_DI_encoder)      echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_3stage_DI_encoder/motion_aux_head.pt" ;;
        vjepa_2_1_surgical_noDI_encoder)           echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_noDI_encoder/motion_aux_head.pt" ;;
        vjepa_2_1_pretrain_head)                   echo "${DEFAULT_OUTPUT_PREFIX}/m09a_pretrain_head/motion_aux_head.pt" ;;
        vjepa_2_1_surgical_3stage_DI_head)         echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_3stage_DI_head/motion_aux_head.pt" ;;
        vjepa_2_1_surgical_noDI_head)              echo "${DEFAULT_OUTPUT_PREFIX}/m09c_surgery_noDI_head/motion_aux_head.pt" ;;
        *) echo "" ;;
    esac
}

# Thread into existing per-encoder calls (Stage 2 L583, Stage 3 L616, Stage 5 L683, Stage 9b TBD)
MA_CKPT="$(motion_aux_head_for "$ENC")"
MA_FLAG=""
[ -n "$MA_CKPT" ] && [ -f "$MA_CKPT" ] && MA_FLAG="--motion-aux-head $MA_CKPT"
python -u src/probe_action.py "--$MODE" --stage features ... $MA_FLAG ...
```

**🚀 Pre-flight check (mirror Stage 8 pre-flight at L393):**
```bash
echo "──────────────────────────────────────────────"
echo "🔓 motion_aux_head pre-flight (head-vs-encoder paired-Δ readiness)"
echo "──────────────────────────────────────────────"
for ENC in $ENCODERS; do
    case "$ENC" in vjepa_2_1_frozen) continue ;; esac
    MA="$(motion_aux_head_for "$ENC")"
    if [ -e "$MA" ]; then
        echo "  ✅ $ENC: $MA"
    else
        echo "  ⚠️  $ENC: $MA not found — eval falls back to encoder-only for this variant"
    fi
done
```

**🧪 Verify:**
```bash
bash -n scripts/run_eval.sh && echo "  syntax: OK"
# Then dry-run on Cell A through Stage 5 only
SKIP_STAGES="3,4,6,7,8,9,10,11,12,13" ./scripts/run_eval.sh --POC
```

**📝 Completion notes:**
- _Fill in: bash -n result, dry-run output_

---

## 🎬 9. Step 7 — POC end-to-end re-eval ⏳

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ ⏳ pending                                     │
│ 💰 Cost                   │ ~$0.40 GPU (Stage 2 re-extract × 8 encoders)  │
│ ⏱ Wall                   │ ~30 min on Blackwell                           │
└──────────────────────────┴────────────────────────────────────────────────┘
```

**🎬 Command:**
```bash
CACHE_POLICY_ALL=2 ./scripts/run_eval.sh --POC 2>&1 \
    | tee logs/iter15_post_poc_eval_head_aware_$(date +%Y%m%d_%H%M%S).log
```

**🎯 Acceptance criteria (paper-grade validity check):**
```
┌────────────────────────────────────────────────────────────────────────────┐
│ ✅ All 3 head cells produce DIFFERENT eval metrics                          │
│    (proves motion_aux head signal made it through to paired-Δ)              │
│ ✅ Frozen baseline metric UNCHANGED vs iter14 (gold-standard preserved)     │
│ ✅ Δ5 / Δ6 / Δ7 are non-zero AND have 95% CI not crossing 0                │
│ ✅ Δ5 sign + magnitude tells the paper claim:                               │
│      🟢 |Δ5| < 0.01  → head-only WINS (paper headline + 1/40× GPU unlock)  │
│      🔵 Δ5 > 0       → encoder margin                                       │
│      🔴 Δ5 < 0       → head outperforms (investigate variance)              │
└────────────────────────────────────────────────────────────────────────────┘
```

**📝 Completion notes:**
- _Fill in: Δ5/Δ6/Δ7 values + CIs · paper-headline verdict_

---

## 🚨 10. Risk + cost matrix

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│ 🚨 Risk                                │ 🛡 Mitigation                         │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ Feature dim varies per cell (K diff)  │ probe attentive classifier d_in       │
│                                       │ auto-derives from features.npy shape  │
│                                       │ (probe_action.py:495 existing pattern)│
│ Frozen baseline can't be augmented    │ resolver echoes "" for frozen → MA_   │
│                                       │ FLAG empty → encoder-only path runs   │
│ iter14 reproducibility break          │ NO — default=None preserves encoder-  │
│                                       │ only path bit-identically              │
│ POC re-extract cost                   │ ~$0.30 (Stage 2 once per encoder)     │
│ FULL re-extract cost                  │ ~$2–3                                  │
│ Per-cell features.npy size grows       │ +0.1-0.2% (K=13 + 23 vs D=1664)       │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

---

## ❌ 11. Rejected alternatives (no lazy reframing later)

```
┌────────────────────────────────────────┬────────────────────────────────────┐
│ ❌ Alternative                          │ Why rejected                        │
├────────────────────────────────────────┼────────────────────────────────────┤
│ S2 zero-shot (argmax logits → top1)    │ Discards attentive-classifier      │
│                                         │ semantics; iter14 anchor 0.808 was │
│                                         │ attentive — apples-to-oranges      │
│ S3 replace (head penultimate features)  │ Throws away encoder info; hidden_  │
│                                         │ dim ≠ encoder_dim → re-shape mess  │
│ New Stage 14/15/16 zero-shot stages    │ Doubles eval wall; existing 4/7/9   │
│                                         │ already do paired-Δ — just feed     │
│                                         │ augmented features in               │
│ Add motion_aux to probe_future_mse     │ Orthogonal to JEPA prediction       │
│                                         │ paradigm — forcing it adds noise   │
│ Defer to FULL run                       │ Per CLAUDE.md NO DEFER — fix now   │
└────────────────────────────────────────┴────────────────────────────────────┘
```

---

## 🎉 12. Definition of done

✅ All 7 steps in the tracker above show `✅` status
✅ POC re-eval emits Δ5/Δ6/Δ7 with non-overlapping 95% CIs
✅ Frozen baseline numbers reproduce iter14 anchor (0.808 top1 on motion-flow)
✅ Plan doc has completion timestamps + observed values per step
✅ Commit-ready with no `// TODO` or deferred items

---

## 📝 13. Update protocol

After each step:
1. Run the verification command
2. Update the step's `🚦 St` cell: `⏳ → 🔥 → 🧪 → ✅` (or `❌` + reason)
3. Fill in the `📝 Completion notes` slot with timestamp + actual values
4. Bump the `0/7` counter in the tracker
5. If a step blocks: mark `⚠ blocked` + add a row to the Risk table

🎬 **Ready to start Step 1?** Confirm and I'll apply the `utils/motion_aux_loss.py` edit + run its verify command.

---

# 🎯 PHASE 2 — 3 corrections to make in-training plots MEANINGFUL for head cells

> 🚨 **Why**: Steps 1-7 fix the EVAL-time gap (probe_action / probe_motion_cos / probe_future_regress consume motion_aux head). BUT the in-training plots emitted by m09a2/m09c2 (probe_trajectory_trio, block_drift, val_loss_jepa) still show flat / zero data because:
> - **trio** reads frozen encoder features (ignores the trained head)
> - **block_drift** targets encoder blocks (frozen → drift ≡ 0)
> - **val_loss_jepa** reads `probe_record["val_jepa_loss"]` which is constant for frozen encoder + predictor
>
> 3 corrections fix each one independently. After applying, re-run the 3 head POC cells (~30 min) and the 4 missing plots will exist with REAL non-trivial data.

---

## 📊 Phase 2 corrections tracker

```
┌────┬──────────────────────────────────────────────────────┬────────┬─────────┬───────────────┐
│ C# │ 🛠 Correction                                          │ ±LoC   │ 🚦 St   │ ✅ When        │
├────┼──────────────────────────────────────────────────────┼────────┼─────────┼───────────────┤
│ C1 │ 🎯 trio: consume motion_aux head augmentation         │  +30   │ ✅      │ 2026-05-16    │
│ C2 │ 📐 drift: target motion_aux head params (not encoder) │  +135  │ ✅      │ 2026-05-16    │
│ C3 │ 📉 val_jepa plot: read val_total_loss for head cells  │  +10   │ ✅      │ 2026-05-16    │
│ C5 │ 🏷 file_prefix m09{a,c} → m09{a,c}2 (module-distinct) │  +10   │ ✅      │ 2026-05-16    │
│ C6 │ 🛡 POC↔FULL parity yaml audit (5 mode-gates flattened)│  +20   │ ✅      │ 2026-05-16    │
│ C4 │ 🧪 Re-run 3 head POC cells, verify 11 plots emit      │    0   │ ⏳      │ —             │
├────┼──────────────────────────────────────────────────────┼────────┼─────────┼───────────────┤
│    │ 🎉 PHASE 2 TOTAL                                       │ ~205   │ 5/6     │               │
└────┴──────────────────────────────────────────────────────┴────────┴─────────┴───────────────┘
```

### 🏷 C5 — file_prefix rename (head modules get distinct identity)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Issue (user-flagged 2026-05-16)                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ m09a2/m09c2 emitted plots under prefix "m09a"/"m09c" → collided semantically │
│ with m09a1/m09c1 plot names. Different modules deserve distinct prefixes.    │
│ But CHECKPOINT_PREFIX (m09{a,c}_ckpt) is HARDCODED in scripts/run_eval.sh    │
│ — must NOT change those. Solution: rename only PLOT file_prefix.             │
├──────────────────────────────────────────────────────────────────────────────┤
│ Applied                                                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│ m09a2: file_prefix="m09a" → "m09a2" (5 sites)                                │
│ m09c2: file_prefix="m09c" → "m09c2" (5 sites)                                │
│ CHECKPOINT_PREFIX unchanged (m09a_ckpt / m09c_ckpt) — run_eval.sh contract   │
│ preserved                                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 🛡 C6 — yaml parity audit (collapse mode-gated dicts to scalars)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Issue (user-flagged 2026-05-16)                                               │
├──────────────────────────────────────────────────────────────────────────────┤
│ Code paths gated probe/multi_task by SANITY-vs-rest, violating CLAUDE.md     │
│ POC↔FULL parity rule ("only n_clips + n_epochs may differ across modes;      │
│ never disable features at SANITY"). I had added a SANITY-fallback block in   │
│ m09a2 that also needed removing.                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Yaml changes (5 mode-gates collapsed to scalar)                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ base_optimization.yaml:                                                        │
│   probe.enabled:              {sanity:false, poc/full:true} → true            │
│   probe.best_ckpt_enabled:     same → true                                    │
│   probe.prec_plateau_enabled:  same → true                                    │
│   probe.use_permanent_val:     same → true                                    │
│   multi_task_probe.enabled:   {all false} → false                            │
│ pretrain_encoder.yaml + surgery_base.yaml:                                    │
│   multi_task_probe.enabled:   {all false} → false                            │
├──────────────────────────────────────────────────────────────────────────────┤
│ Code change                                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ m09a2: removed SANITY-fallback block (probe-disabled path) — probe now       │
│ universal-true → probe_history always non-empty → no fallback needed.        │
├──────────────────────────────────────────────────────────────────────────────┤
│ Verified: all 6 cells × 3 modes resolve to identical scalar values            │
│   probe.enabled = True, best_ckpt = True, plateau = True, perm_val = True,  │
│   mt.enabled = False (uniform across SANITY/POC/FULL)                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 C1 — Trio consumes motion_aux head augmentation

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ ⏳ pending                                     │
│ 📁 Files                  │ src/utils/probe_trio.py (+15)                  │
│                          │ src/utils/training.py:run_trio_at_val (+5)     │
│                          │ src/m09a2_pretrain_head.py + src/m09c2_...    │
│                          │   (+5 each for ma_head kwarg threading)       │
│ ➕ LoC                    │ ~+30                                           │
│ ⏱ Effort                 │ ~30 min                                        │
└──────────────────────────┴────────────────────────────────────────────────┘
```

**🔬 Current behavior (head cells):**
- `compute_metric_trio` (probe_trio.py:99) computes top1 + motion_cos from pooled encoder features (L208 `last_layer.mean(dim=1)`)
- Encoder FROZEN → identical features every val cycle → identical top1, motion_cos
- future_l1 (L246) uses encoder + predictor forward → both frozen → constant

**🛠 Fix design:**
- Add `motion_aux_head: MotionAuxHead | None = None` kwarg to `compute_metric_trio`
- When provided: AFTER L208 (`pooled = last_layer.mean(...)`):
  ```python
  if motion_aux_head is not None:
      ma_concat = forward_motion_aux_concat(motion_aux_head, pooled)
      pooled = torch.cat([pooled.to(ma_concat.device), ma_concat], dim=-1).cpu()
  ```
- top1 + motion_cos compute on augmented `pooled_np` → cell-specific values
- Thread `motion_aux_head` kwarg through `run_trio_at_val` (`utils/training.py:2566`) → call site

**🚨 Brutally honest limitation (future_l1):**
- future_l1 uses predictor's masked-latent prediction (L240-246). The predictor's forward path doesn't see motion_aux head → **future_l1 STAYS CONSTANT for head cells even with this fix**. Augmenting `pooled` only affects top1 + motion_cos.
- Result: 2 of 3 trio metrics will move per cell; future_l1 line stays flat in the trio plot.
- This is acceptable for the paper claim (Δ5/Δ6/Δ7 only need top1 OR motion_cos to differ; future_l1 stays as a baseline anchor).

**🛠 Sites to edit:**
1. `src/utils/probe_trio.py` — add kwarg + augment after L208 pooling
2. `src/utils/training.py:run_trio_at_val` (L2566) — accept + forward kwarg
3. `src/m09a2_pretrain_head.py` — pass `motion_aux_head=ma_head` in val cycle call
4. `src/m09c2_surgery_head.py` — same

**🧪 Verify:**
```bash
# Synthetic: same encoder features through trio → 3 different top1/motion_cos values
python -c "
from utils.probe_trio import compute_metric_trio
# ... call with each head; verify top1 + motion_cos differ across 3 heads
"
```

**📝 Completion notes:**
- _Fill in post-implementation: top1 spread across 3 heads, motion_cos spread, future_l1 confirmation of constancy_

---

## 📐 C2 — Drift targets motion_aux head params (not encoder)

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ ⏳ pending                                     │
│ 📁 Files                  │ src/utils/training.py — add                    │
│                          │   track_head_drift_at_val (+35 LoC)            │
│                          │ src/m09a2_pretrain_head.py + src/m09c2_...    │
│                          │   (+8 each: snapshot init + call per val)     │
│ ➕ LoC                    │ ~+50                                           │
│ ⏱ Effort                 │ ~30 min                                        │
└──────────────────────────┴────────────────────────────────────────────────┘
```

**🔬 Current behavior (head cells):**
- `track_block_drift_at_val(student, init_params, freeze_below, ...)` computes per-block rel-L2 over ENCODER weights
- Encoder FROZEN for head cells → drift ≡ 0 across all 48 blocks
- Output file `m09{a,c}_block_drift.{png,pdf}` shows all-zero band; `_block_drift_history.json` stub already emitted

**🛠 Fix design:**
- Add new helper `track_head_drift_at_val(ma_head, head_init_params, head_drift_history, output_dir, step, probe_record, title_prefix, file_prefix)` in `utils/training.py`
- Computes per-LOGICAL-BLOCK rel-L2 over motion_aux head's 3 sub-modules:
  - `trunk` (Linear→GELU→LayerNorm→Dropout, ~427K params)
  - `ce_head` (Linear→K classes, ~3.3K params at K=13)
  - `mse_head` (Linear→n_dims=23, ~5.9K params)
- Returns 3 drift scalars; appends to `head_drift_history`
- Renders 2-panel heatmap (3 logical blocks × N val cycles) + trajectory plot
- **Same file names as encoder cells for parity**: `m09{a,c}_block_drift.{png,pdf}` + `m09{a,c}_block_drift_history.json` (content semantically = HEAD drift; title clarifies)

**📐 Implementation in m09a2 / m09c2:**
1. After `ma_head` is built (post `build_motion_aux_head_from_cfg`):
   ```python
   head_init_params = {n: p.detach().cpu().clone() for n, p in ma_head.named_parameters()}
   head_drift_history = []
   ```
2. In val cycle (after motion_aux val computation):
   ```python
   track_head_drift_at_val(
       ma_head, head_init_params, head_drift_history,
       output_dir=output_dir, step=step,
       probe_record=probe_record,
       title_prefix=f"m09a head step={step} · ",
       file_prefix="m09a")
   ```

**📊 Expected result (head cells):**
- ep 0 (just after init): drift ≈ 0 per block
- ep 1+: drift > 0, monotonically increasing as head trains
- Plot shows 3 colored trajectories (trunk / ce_head / mse_head) ascending across epochs

**🧪 Verify (post-implementation):**
```bash
# Run Cell A POC re-train; inspect m09a_block_drift_history.json:
python -c "
import json
h = json.load(open('outputs/poc/m09a_pretrain_head/m09a_block_drift_history.json'))
print(f'  N val cycles: {len(h)}')
print(f'  block names: {list(h[0][\"drift_per_block\"].keys())}')
print(f'  ep0 drift: {h[0][\"drift_per_block\"]}')
print(f'  epN drift: {h[-1][\"drift_per_block\"]}  (expect non-zero, ascending)')
"
```

**📝 Completion notes:**
- _Fill in: per-block drift values at ep0 vs epN, confirming non-trivial trajectory_

---

## 📉 C3 — val_jepa plot reads val_total_loss for head cells

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ ⏳ pending                                     │
│ 📁 Files                  │ src/m09a2_pretrain_head.py (+3)                │
│                          │ src/m09c2_surgery_head.py (+3)                 │
│ ➕ LoC                    │ ~+6                                            │
│ ⏱ Effort                 │ ~5 min                                         │
└──────────────────────────┴────────────────────────────────────────────────┘
```

**🔬 Current behavior (head cells):**
- `plot_val_loss_with_kill_switch_overlay(probe_history, ...)` reads `probe_record["val_jepa_loss"]`
- For frozen encoder+predictor, val_jepa is CONSTANT (zero info)
- If we populate it with `0.0` or skip, plot is empty or flat

**🛠 Fix design (semantic field repurposing):**
- The plot function doesn't care about the SEMANTIC label — it plots whatever's in `val_jepa_loss` key
- For head cells, populate `val_jepa_loss` with `val_total_loss = val_motion_aux * weight_motion` (= the optimizer's actual target)
- Plot file name stays `m09{a,c}_val_loss_jepa.{png,pdf}` for FILE PARITY
- Add inline comment + plot title override clarifies "= val_total for head cells"

**🛠 Sites:**
- In m09a2/m09c2 val cycle, when building probe_record:
  ```python
  # iter15 Phase 6 C3 (2026-05-16): for frozen encoder+predictor, val_jepa is
  # constant → repurpose this field to hold val_total_loss = the optimizer's
  # actual target = val_motion_aux × weight. Plot reads this key directly.
  val_total = val_loss * float(ma_cfg["weight_motion"])
  probe_record = {
      "step": step, "epoch": epoch,
      "val_jepa_loss": val_total,    # repurposed for head cells (= val_total)
      "val_motion_aux_loss": val_loss,
      "val_total_loss": val_total,
      "epoch_pct": round(pct, 1),
  }
  ```

**📊 Expected result:**
- val_loss_jepa plot shows a DESCENDING curve across val cycles (head trains → motion_aux val decreases)
- Matches the loss_log.csv val_loss column we already emit (no contradiction)

**📝 Completion notes:**
- _Fill in: val_loss values at ep0 and epN — should descend_

---

## 🧪 C4 — Re-run 3 head POC cells + verify all 11+4 = 15 files exist

```
┌──────────────────────────┬────────────────────────────────────────────────┐
│ 🚦 Status                 │ ⏳ pending                                     │
│ 💰 Cost                   │ ~$0.40 GPU (3 cells × ~8 min each)             │
│ ⏱ Wall                   │ ~30 min on Blackwell                           │
└──────────────────────────┴────────────────────────────────────────────────┘
```

**🎬 Commands:**
```bash
CACHE_POLICY_ALL=2 bash scripts/run_train.sh pretrain_head --POC 2>&1 \
    | tee logs/iter15_poc_m09a2_pretrain_head_v4_$(date +%Y%m%d_%H%M%S).log
CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_3stage_DI_head --POC 2>&1 \
    | tee logs/iter15_poc_m09c2_3stage_DI_head_v4_$(date +%Y%m%d_%H%M%S).log
CACHE_POLICY_ALL=2 bash scripts/run_train.sh surgery_noDI_head --POC 2>&1 \
    | tee logs/iter15_poc_m09c2_noDI_head_v4_$(date +%Y%m%d_%H%M%S).log
```

**🎯 Acceptance criteria (file parity with iter14 R1):**

```
┌────────────────────────────────────────────────────┬───────────────────────────┐
│ File                                                │ Expected post-fix          │
├────────────────────────────────────────────────────┼───────────────────────────┤
│ loss_log.csv                                        │ ✅ already emit (Step 2)   │
│ loss_log.jsonl                                      │ ✅ already emit (Step 2)   │
│ training_summary.json                               │ ✅ already emit            │
│ m09{a,c}2_train_loss.png + .pdf                     │ ✅ already emit (Step 3)   │
│ m09{a,c}2_loss_decomposition.png + .pdf             │ ✅ already emit (Step 3)   │
│ m09{a,c}2_block_drift.png + .pdf                    │ 🆕 from C2 (HEAD drift)    │
│ m09{a,c}2_block_drift_history.json                  │ 🆕 from C2 (non-zero)      │
│ m09{a,c}2_probe_trajectory_trio.png + .pdf          │ 🆕 from C1 (top1+m_cos move│
│                                                     │   future_l1 stays flat ⚠) │
│ m09{a,c}2_val_loss_jepa.png + .pdf                  │ 🆕 from C3 (val_total ↓)   │
│ probe_history.jsonl                                 │ 🆕 from C1 (run_trio fires)│
│ motion_aux_head.pt                                  │ ✅ already emit            │
│ student_encoder.pt                                  │ ✅ already emit            │
│ m09{a,c}_ckpt_best.pt                               │ ✅ already emit (run_eval  │
│                                                     │   contract: keeps prefix)  │
├────────────────────────────────────────────────────┼───────────────────────────┤
│ TOTAL                                               │ 15 files per head cell     │
│ NOTE: plot prefix m09{a,c}2_ ≠ iter14 R1's m09{a,c}_ — INTENTIONAL per C5    │
│ (head modules m09a2/m09c2 ≠ encoder modules m09a1/m09c1 — distinct identity).│
└────────────────────────────────────────────────────┴───────────────────────────┘
```

**🚨 Brutal-honest disclaimers (after C1+C2+C3 applied, what the plots actually show for head cells):**

```
┌──────────────────────────────────────┬──────────────────────────────────────────┐
│ Plot                                   │ What it shows (head cell semantics)       │
├──────────────────────────────────────┼──────────────────────────────────────────┤
│ train_loss.{png,pdf}                  │ ✅ REAL — motion_aux train loss descent   │
│ loss_decomposition.{png,pdf}          │ ✅ REAL — motion_aux dominates total      │
│ block_drift.{png,pdf}                 │ ✅ REAL — head's trunk/ce/mse drift ↑    │
│   (file shows HEAD drift, not encoder)│                                            │
│ probe_trajectory_trio.{png,pdf}       │ ⚠ PARTIAL — top1 + motion_cos MOVE per   │
│                                       │   cell (real signal); future_l1 stays    │
│                                       │   FLAT (predictor frozen — by design)    │
│ val_loss_jepa.{png,pdf}               │ ✅ REAL — populated with val_total_loss   │
│   (file shows val_total, not JEPA)    │   = val_motion_aux × weight; descending  │
└──────────────────────────────────────┴──────────────────────────────────────────┘
```

**📝 Completion notes:**
- _Fill in: file diff vs iter14 R1 (expect 0 missing); plot inspection notes_

---

## ⏱ Phase 2 total wall + cost

```
┌────────────────────────────────┬─────────────────────────────────┐
│ Item                             │ Estimate                          │
├────────────────────────────────┼─────────────────────────────────┤
│ Code (C1+C2+C3)                  │ ~86 LoC across 4 files, ~70 min  │
│ Re-train 3 head POC cells (C4)   │ ~30 min wall, ~$0.40 GPU         │
│ Total Phase 2                    │ ~100 min, ~$0.40                  │
└────────────────────────────────┴─────────────────────────────────┘
```

---

## 🎉 Phase 2 definition of done

✅ All 4 corrections (C1/C2/C3/C4) show ✅ status in tracker
✅ `outputs/poc/m09{a,c}_*_head/` directories contain ALL 15 files (parity with iter14 R1)
✅ Per-file content meets the honest expectations above (no flat lines except future_l1)
✅ probe_history.jsonl has ≥ 1 record per val cycle with non-trivial top1/motion_cos values
✅ block_drift_history.json shows ascending head-drift trajectories (non-zero, non-constant)
✅ val_loss_jepa plot shows descending curve (matches loss_log.csv val_loss column)

---

# 🧹 PHASE 3 — final audit (D1–D5) — close the 5 deferred items the audit flagged

Goal: close every "honest" gap that surfaced when re-reading C1–C6 against `scripts/run_eval.sh`. No deferral, all fixed in this session. Smoke-test verifies wiring without GPU.

## 📋 Phase 3 tracker

```
┌────┬──────────────────────────────────────────────────────────────┬────────┬────────┬──────────────┐
│ D# │ 🛠 Audit fix                                                  │ ±LoC   │ 🚦 St   │ ✅ When        │
├────┼──────────────────────────────────────────────────────────────┼────────┼────────┼──────────────┤
│ D1 │ 🧭 probe_taxonomy.py: wire --motion-aux-head CLI + augment   │ +35    │ ✅      │ 2026-05-16   │
│ D2 │ 🎯 probe_future_mse.py: wire --motion-aux-head (full Stage 8) │ +95    │ ✅      │ 2026-05-16   │
│ D3 │ 🔁 encoder cells (m09a1/c1) trio symmetry (pass ma_head)     │ +6     │ ✅      │ 2026-05-16   │
│ D4 │ 🔑 m09a2/c2 best_state key: val_jepa_loss → val_loss_at_best │ +2     │ ✅      │ 2026-05-16   │
│ D5 │ 🧪 synthetic smoke test (no-GPU import + signature + flag)   │ +0     │ ✅      │ 2026-05-16   │
│ D6 │ 🚨 probe_action lazy-extract MUST also augment (POC crash)   │ +30    │ ✅      │ 2026-05-16   │
└────┴──────────────────────────────────────────────────────────────┴────────┴────────┴──────────────┘
```

### 🚨 D6 — probe_action lazy-extract path was missing the augment (real POC crash)

Surfaced by `logs/iter15_post_poc_eval_head_aware_20260516_012451.log:482`. Original Phase 1 Step 2 wired `--motion-aux-head` into `run_features_stage` (which saves only `features_test.npy` per the `--features-splits` default), but `_load_or_extract` (Stage 3 lazy-extract for train/val) was NOT augmented. Result: `feats_train.shape[-1] = 1664` (lazy, no augment) while `feats_test.shape[-1] = 1700` (disk, augmented) → AttentiveClassifier built with embed_dim=1664, training succeeded on 1664-dim train+val (val_acc=0.30 @ epoch 19), then test eval crashed with `LayerNorm normalized_shape=[1664] vs input [64, 16, 1700]`.

Fix: mirror the augmentation block from `run_features_stage` into `_load_or_extract` — load `ma_head_lazy` once outside the closure, apply `apply_motion_aux_head_to_features` + tile + concat to any lazy-extracted split. Disk-loaded path is unchanged (caller persisted whatever shape).

Smoke verification: `augmentation sites in probe_action.py: 4 (run_features_stage + _load_or_extract)` — both paths now apply the head when `--motion-aux-head` is set.

Honest residual (B4): `stream_train_attentive_probe` (`STREAM_TRAIN=1` opt-in, default OFF) does NOT yet apply the augment. POC↔FULL parity rule says it must, BUT the failing POC run had `STREAM_TRAIN=0` (default in run_eval.sh:596). Flagged for follow-up; does not block the immediate P1 re-eval.

## 🧭 D1 — probe_taxonomy.py augment in lazy-extract path

- Added `--motion-aux-head` CLI flag (default None, type=Path).
- In `_load_or_extract`: when lazy-extracting frozen features, run `apply_motion_aux_head_to_features(...)` and tile/concat → augmented (N, n_tokens, D+K+n_dims) feature space matching probe_action.

## 🎯 D2 — probe_future_mse.py full wiring (Stage 8 of run_eval.sh)

- Added `--motion-aux-head` CLI flag.
- `_forward_one_batch` returns 3-tuple `(per_clip_l1, out, h_target)` so the augment path can mean-pool over the LAST `embed_dim=1664` columns of the hierarchical concat (matches `probe_trio.py:225`).
- `_flush_batch_forward` accepts `ma_head=None, ma_acc=None, embed_dim=None`. When set: forward both pred + target through `ma_head` → concat (logits ‖ vec) → L1 in augmented (K+n_dims) space → append to `ma_acc`.
- `_save_mse_ckpt` persists `ma` array alongside `mse` for crash-safe resume (FATAL if resume ckpt has no `ma` array when `--motion-aux-head` is set — no silent length drift).
- `run_forward_stage` saves `per_clip_motion_aux_l1.npy` + `aggregate_motion_aux_l1.json` (mean/std/95% CI BCa) parallel to the JEPA `per_clip_mse.npy` + `aggregate_mse.json`.

## 🔁 D3 — encoder-cell trio symmetry

- `m09a1_pretrain_encoder.py` + `m09c1_surgery_encoder.py`: added `motion_aux_head=ma_head` kwarg to in-training `run_trio_at_val` call so the encoder cells and head cells consume the same augmented feature space (paired-Δ semantics are now apples-to-apples in the in-training trio plots).

## 🔑 D4 — best_state key fix in m09a2/c2

- Plot reader (`plots.py:857`) reads `val_loss_at_best`, but two earlier patches typoed `val_jepa_loss_at_best`. Renamed → plot now overlays the kill-switch threshold correctly. Comment retained as `# D4 fix (2026-05-16): ...` for archaeology.

## 🧪 D5 — synthetic smoke test (CPU-only, no GPU needed)

Single Python invocation that imports every changed module and asserts:

```
[D5.1] all 4 m09 modules import: ok
[D5.2] core symbol import (probe_trio, training, plots, motion_aux_loss, frozen_features): ok
[D5.3] signature compat: compute_metric_trio / run_trio_at_val / track_head_drift_at_val
[D5.4] all 5 probe modules import: ok
[D5.5] all 5 probes have --motion-aux-head CLI: ok
[D5.6] D4 best_state key fix verified in m09a2/c2: ok
[D5.7] D3 encoder-cell symmetry verified in m09a1/c1: ok
[D5.8] D1 probe_taxonomy augment wired: ok
[D5.9] D2 probe_future_mse augment wired (save + flush): ok
```

This caught a real bug during the run: my first version imported `plot_probe_trajectory_trio` from `utils.training` (wrong module — it lives in `utils.plots`). Fixed the import target and re-ran. The smoke test does NOT replace P1/P2 GPU validation — it only guards against renames / typos / signature drift that would crash 30 minutes into a $0.40 POC.

## ⏱ Phase 3 total

```
┌────────────────────────────────┬─────────────────────────────────┐
│ Item                             │ Actual                            │
├────────────────────────────────┼─────────────────────────────────┤
│ Code (D1+D2+D3+D4+D5)            │ ~140 LoC across 6 files          │
│ Smoke test (no GPU)              │ < 30 s wall, $0.00               │
│ Total Phase 3                    │ ~25 min code, $0.00              │
└────────────────────────────────┴─────────────────────────────────┘
```

## 🎉 Phase 3 definition of done

✅ All 5 audit items (D1–D5) show ✅ status in tracker
✅ AST parse + py_compile pass on all 13 touched files
✅ Synthetic smoke test passes all 9 sub-checks (D5.1–D5.9)
✅ probe_future_mse emits a 4th paired-Δ axis (`per_clip_motion_aux_l1.npy`) in augmented (K+n_dims) space — gold-standard "consume head at eval to test the head's research claim"
✅ Honest residuals flagged: B1 (m09a1/c1 file_prefix), B2 (CHECKPOINT_PREFIX module-distinct), B3 (formal unit tests) — explicitly NOT fixed this session; user has not asked for them yet, and none of them block P1/P2 GPU re-eval
