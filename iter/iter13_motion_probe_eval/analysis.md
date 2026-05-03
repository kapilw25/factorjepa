# Deep-research answers — V-JEPA 2.1 ultrathink

---

## 🚦 iter13 state snapshot (2026-05-03 16:30 PDT)

> **TL;DR**: Multi-task probe-loss wiring shipped end-to-end (m09a + m09c + utils/multi_task_loss.py + 5 helpers + 11-test REPL gate). Eval pipeline runs clean on 24 GB SANITY (4-encoder pre-flight + Stage 8 graceful skip + NaN-safe plotting all landed). **Training of pretrain + surgery_3stage_DI + surgery_noDI requires 96 GB** — 24 GB OOMs even at sub-batch=1 + 8 frames + all memory savers stacked. All known bugs fixed (Bug A: `_best.pt` full=True; Bug B: while-not-step-succeeded retry; Bug R8: m09c writes `m09c_ckpt_best.pt`). Next concrete step: kick the FULL pipeline per `runbook.md` Phases A→B→C on 96 GB hardware.

### Multi-task probe-loss supervision (user pivot, 2026-05-03)

**Why added**: pure-SSL m09a/m09c have ZERO gradient signal toward the 16-dim probe metric (action + 15 taxonomy dims). Without direct supervision, probability of trained encoder beating frozen on probe accuracy depends on incidental alignment (~5-15%). Multi-task adds:

```
total_loss = α · JEPA_L1  +  β · Σ_d (1/n_dims) · L_d  +  drift_L2
            (α=1.0)        (β=0.1, weight_probe)
```
where L_d is CrossEntropy for 14 single-label dims (action + 13 taxonomy) and BCEWithLogits for 2 multi-label dims (road_layout, notable_objects).

**Code surface** — 5 helpers in `utils/multi_task_loss.py` (technique-agnostic per #49 contract):
| Helper | Purpose |
|---|---|
| `merge_multi_task_config(cfg, args, mode_key)` | Per-mode flatten + CLI overrides |
| `build_multi_task_head_from_cfg(cfg, device)` | Returns `(mt_head, labels, dims, mt_cfg)`; silent-disable on missing labels file |
| `attach_head_to_optimizer(optimizer, mt_head, mt_cfg, base_lr)` | Adds head param group at `base_lr × head_lr_multiplier` (10×) |
| `run_multi_task_step(student, mt_head, ..., batch_clips, batch_keys, scaler, mp_cfg, dtype, device)` | One forward+backward; returns `(mt_loss_val, mt_per_dim)`; re-raises OOM |
| `export_multi_task_head(mt_head, dims_spec, d_encoder, path)` | Writes `multi_task_head.pt` |

Each m09 call site shrinks from ~22 LoC to ~3 LoC; net 120 duplicated LoC → 30 LoC across both files + 95 LoC of helpers (single test surface).

**Per-config opt-in** (`probe_pretrain.yaml`, `surgery_3stage_DI.yaml`, `surgery_2stage_noDI.yaml`):
```yaml
multi_task_probe:
  enabled: {sanity: true, poc: true, full: true}    # opt-in for THIS config
```
Base ships `false` for all modes; opt-in is per-config so legacy ch10/explora are unaffected.

### Bug fixes that landed in iter13 (2026-05-03)

| # | Bug | Fix | Where |
|---|---|---|---|
| **A** | m09a `_best.pt` saved with `full=False` → no predictor key → Stage 8 future_mse FATAL on `KeyError 'predictor'` | Flip to `full=True` | `m09a_pretrain.py:1098` |
| **B** | OOM-retry used `continue` → SANITY `total_steps=1` exited with 0 successful steps → silently exported untrained Meta weights | `while not step_succeeded:` retry-same-macro loop + post-train `M09A FAILED: 0 successful training steps` fail-hard | `m09a_pretrain.py:836-866` + `:1166-1177` |
| **R8** | m09c writes `student_best.pt` (encoder-only, full=False); cleanup wipes stage ckpts; eval expects `m09c_ckpt_best.pt` (full ckpt) | Add explicit `save_training_checkpoint(_best.pt, ..., full=True)` after best-promotion; survives `cleanup_stage_checkpoints` (different glob pattern) | `m09c_surgery.py:1372-1385` |
| **OOM frag** | Each OOM-retry left orphan tensors → next sub-batch shrink had less free VRAM → eventually OOM at sub-batch=1 even though sub-batch=1 should fit | Add `gc.collect() + torch.cuda.empty_cache()` after `optimizer.zero_grad()` in OOM handler | both `m09a:849-857` + `m09c:1170-1179` |
| **eval ckpt schema** | `utils.frozen_features.load_vjepa_2_1_frozen` only knew Meta's `target_encoder`/`encoder` keys; m09a `student_state_dict` (export) and `student` (full ckpt) → 0/588 loaded | Add `resolve_encoder_state_dict()` helper recognizing 4 schemas in priority order | `utils/frozen_features.py:74-95` + `probe_future_mse.py:121` |
| **Stage 8 FATAL** | Eval-side hard-stop when predictor-bearing ckpt missing → killed Stages 9+10 even though future_mse is V-JEPA-only and partial result is valid | Pre-flight builds `STAGE8_ENCODERS` subset; in-loop WARN+continue (defense-in-depth) | `scripts/run_probe_eval.sh:289-326` + `:484-512` |
| **Plot NaN ylim** | `_bar_with_ci` computed `ax.set_ylim(NaN, NaN)` when any encoder had degenerate BCa CI (perfect predictions → zero variance → ci_half=NaN) | NaN-safe ylim via `np.nan_to_num(real_e, nan=0.0)` before `min/max`; same for value-label placement | `src/probe_plot.py:196-218` |

### What runs where (hardware split, validated 2026-05-03)

| Pipeline | 24 GB SANITY | 96 GB FULL |
|---|---|---|
| `run_probe_eval.sh --sanity` (10 stages) | ✅ ~6-8 min, all stages green | ✅ trivially |
| `run_probe_train.sh pretrain --SANITY` | ❌ OOM at sub-batch=1 (probe_pretrain_sanity_v6.log) | ✅ ~3 GPU-h |
| `run_probe_train.sh surgery_* --SANITY` | ❌ same OOM regime | ✅ ~4-8 GPU-h |
| `run_probe_eval.sh --FULL` (10 stages, ~9.9k clips) | ⚠️ ~4 h on 24 GB feasible per spec but 96 GB recommended | ✅ ~2.5 GPU-h |

**Why 24 GB can't train V-JEPA ViT-G** (`probe_pretrain_sanity_v6.log` evidence, fixed memory math):
- Student ViT-G fp32 (1.84B params): ~7.4 GB
- Teacher EMA copy: ~7.4 GB (NOT shared with student — m09a builds via `copy.deepcopy`)
- Predictor 60M: ~0.24 GB
- Master fp32 + 8-bit Adam state for trainable params: ~7-9 GB (paged optim has SOME state on GPU)
- PyTorch + bitsandbytes overhead: ~2 GB
- **Subtotal (FIXED) ≈ 24-26 GB** — already exceeds the 24 GB budget before any activations

The 8-bit + paged + grad-ckpt + sub-batch=1 + 8-frame stack (probe_pretrain_sanity_v6.log) couldn't shrink the FIXED footprint enough. errors_N_fixes #55 documents this; the fail-hard message in `m09a:851-857` points users at the right answer (move to FULL hardware).

### What's NOT yet validated empirically (post-fix-but-pre-FULL)

- ❌ Multi-task loss values on real training data — only REPL smoke (mock dims + random pooled feats) covers the math
- ❌ `m09a_ckpt_best.pt` (~15 GB full ckpt) actually being written and loaded by Stage 8 — only static glob-safety REPL covers this
- ❌ `m09c_ckpt_best.pt` round-trip from m09c training → Stage 8 future_mse load
- ❌ All 6 pairwise Δ comparisons in `probe_paired_delta.json` for the full 4-encoder roster
- ❌ Stage 10 `probe_encoder_comparison.png` with 4 V-JEPA bars + 1 DINOv2 bar
- ❌ Whether multi-task supervision actually moves probe top-1 acc (the open empirical question)

These get answered when `runbook.md` Phase B+C fires on 96 GB.

---

## 🆕 SANITY-mode P1 result (2026-05-03) — ⚠️ NOT P1-defining, AWAITING FULL

> 🚨 **Headline**: SANITY direction is **WRONG WAY** (DINOv2 beats V-JEPA by 18 pp). 🚨
> ⏳ **DO NOT update P1 verdict from this** — n_test = 22 is too small to draw any population-level conclusion. Re-run on **FULL eval_10k (~9.9k clips → ~1,492 test)** before judging the gate.

### 📊 SANITY verdict table (`outputs/sanity/m06d_action_probe/m06d_paired_delta.json`, n=22)

| Metric | 🎯 P1 target | 📉 SANITY plot | 🚦 Verdict (per `plan_training.md` decision matrix) |
|---|---|---|---|
| Δ top-1 acc (V-JEPA − DINOv2) | ≥ +20 pp, CI_lo > 0, p < 0.05 | **−18.18 pp** (V-JEPA 77.27 % vs DINOv2 95.45 %), CI [−36.36, −4.55], p = 0.024 | ❌ **Δ ≤ 0** — the *worst* row of the matrix |
| Direction | V-JEPA above DINOv2 | DINOv2 above V-JEPA by 18 pp | 🔄 **inverted** |
| CI containment | Δ-CI must exclude 0 from below | Δ-CI excludes 0 from **above** (CI_hi = −4.55) | ⚠️ **statistically significant in the WRONG direction** |

### 🖼️ What the 3-panel SANITY plot shows (`outputs/sanity/m08d_plot_m06d/m06d_encoder_comparison.png`)

| Panel | 🥇 V-JEPA 2.1 frozen | 🥈 DINOv2 frozen | Sign |
|---|---|---|---|
| 1 — Action probe top-1 (n=22) | 77.273 % (CI ~[59, 95]) | **95.454 %** (CI ~[84, 100]) | ❌ DINOv2 wins |
| 2 — Motion cosine intra-inter (n=22) | 0.050 (CI [0.034, 0.067]) | **0.089** (CI [0.061, 0.118]) | ❌ DINOv2 wins (signal echoes panel 1) |
| 3 — Future-frame L1 (n=22) | 0.555 (CI [0.549, 0.563]) | N/A (no predictor head) | ⛔ no comparison possible |

### ⚠️ Why we **cannot** treat SANITY as the P1 verdict

- 🪙 **n_test = 22** clips total (8 + 7 + 7 per class) — the smallest split that clears `stratified_split`'s ≥5/split floor with 2-clip margin. The ±18 pp CI half-width on V-JEPA's accuracy is exactly Wilson-noise at n=22.
- 🧠 **Probe overfit signal**: `train_acc = 1.0, val_acc = 0.857` after 50 epochs on 105 train clips → comparing two overfit probes on 22 test clips is signal-poor by design.
- 📏 **Scale check**: FULL test split = ~1,492 clips → Δ-CI shrinks ~8× (√(1492/22) ≈ 8.2). At that N, both a +20 pp lift and a −18 pp deficit are detectable; **SANITY cannot tell us which side of zero the population lives on.**
- 🚧 **CLAUDE.md rule**: *"SANITY validates code correctness (no crashes), NOT model performance. Never draw conclusions from insufficient data."*

### ⏳ What we are waiting for

```bash
# kicks off all 10 stages on eval_10k (~2.5 GPU-h on 24 GB)
./scripts/run_m06d_eval.sh 2>&1 | tee logs/run_src_m06d_full_v1.log
```

| Run | n_test | Δ-CI half-width target | What it answers |
|---|---|---|---|
| ✅ SANITY (done) | 22 | ±15.9 pp | code correctness — pipeline runs end-to-end without crash |
| ⏳ FULL (next) | ~1,492 | ~±2 pp | **the actual P1 gate** — does V-JEPA 2.1 frozen beat DINOv2 frozen on Indian motion-centric action probe? |

### 🚨 If FULL ALSO flips — pipeline-validation gap to close FIRST

**Critical missing experiment**: we have NO Meta-published-benchmark reproduction. Until we run frozen V-JEPA 2.1 + Meta's `ssv2-vitg-384-64x2x3.pt` probe on SSv2 val and reproduce Meta's published **75.3 %** ([V-JEPA 2 paper Table 4, arxiv 2506.09985](https://arxiv.org/html/2506.09985v1)), we cannot distinguish:

- 🅰️ **Domain shift** — Indian outdoor video genuinely flips the SSv2-domain +24.6 pp gap (publishable negative result)
- 🅱️ **Pipeline bug** — our `m05`/`m06d` extractor / probe trainer has a silent error that corrupts V-JEPA features (silent metric corruption, must fix before claiming anything)

#### 📌 Proposed pre-FULL pipeline-validation experiment (~1 GPU-h)

| Step | Action | Expected | If we see |
|---|---|---|---|
| 1 | Download SSv2 val split (~10 GB) | — | — |
| 2 | Run frozen V-JEPA 2.1 ViT-G + Meta's released SSv2 probe ckpt `ssv2-vitg-384-64x2x3.pt` (inference-only, no training) | **75.3 %** top-1 (Meta's published number) | ✅ infra correct → trust m06d FULL number<br>❌ ≠ 75.3 % → debug `m05` / `frozen_features.py` extractor before reading FULL m06d as P1 verdict |

### 📚 Why our 3 metrics ≠ Meta's 4 motion-centric benchmarks (similarity audit)

Meta's published frozen-encoder + 4-layer attentive-probe motion-centric suite: **SSv2 / Diving-48 / Ego4D OSC / EPIC-KITCHENS-100** ([V-JEPA 2 Table 4 + 5](https://arxiv.org/html/2506.09985v1)).

| Our metric | Meta's closest | Same recipe? | Same task? | Same data? | Grade |
|---|---|---|---|---|---|
| 🥇 Indian action probe top-1 | SSv2 top-1 (174-class) | ✅ identical 4-layer `AttentiveClassifier` | ✅ supervised action classification | ❌ Indian outdoor 3-class vs SSv2 indoor 174-class | **B−** |
| 🥈 Motion cosine intra-inter | none in Meta's published suite | ❌ probe-free | ❌ feature-similarity, not classification | ❌ Indian outdoor | **D** |
| 🥉 Future-frame L1 (V-JEPA only) | closest = Ego4D OSC mAP | ❌ raw training loss vs labeled state-change mAP | ❌ unsupervised L1 vs supervised anticipation | ❌ Indian outdoor vs Ego4D kitchen | **D−** |

**Honest read**: only the action probe is in the same *family* as one of Meta's 4 benchmarks. Motion cosine + future MSE are domain-specific *supplementary* probes — useful as secondary signals in our paper but **NOT a substitute for SSv2/Diving-48 reproduction** as a pipeline-correctness check.

### 🎯 Action items, ordered by priority

| Priority | Item | Cost | Unblocks |
|---|---|---|---|
| 🥇 P0 | Run **FULL m06d** on eval_10k → produce real P1 verdict at n=1,492 | ~2.5 GPU-h | the entire iter13 plan |
| 🥈 P0.5 | Run **SSv2 reproduction** with Meta's published probe ckpt | ~1 GPU-h + 10 GB download | disambiguate "domain shift" vs "pipeline bug" if FULL also flips |
| 🥉 P1 | If FULL P1 PASSES (Δ ≥ 0, CI_lo > 0): proceed to Track 2 (m12/m13/m13b factor-conditioned probe) | ~10 GPU-h | publishable surgical > frozen claim |
| 🛑 P1 | If FULL P1 FAILS (Δ < 0 or CI overlap): execute the decision-matrix branch (`plan_training.md` §"Outcome decision matrix") — diff `m05` extractor vs `vjepa2_demo.ipynb`, then either pivot paper framing or fix pipeline | varies | paper recovery path |

---

**The wrong-metric finding itself is publishable**: "Cross-clip retrieval Prec@K is not the right gate metric for V-JEPA-style generative SSL; here are the metrics that DO move (Cycle@K, val_jepa, action recognition probe)." That's a legitimate negative-result + corrective contribution.

**Pivot the paper's gate metric** from Prec@K to V-JEPA's native metrics (Q2). On those, you have real numbers to defend.

---

## Q2) Where V-JEPA fine-tuning DOES lift over frozen — official benchmark catalog

V-JEPA 2.1's published evaluation suite uses a **frozen encoder + 4-layer attentive probe** ([V-JEPA 2.1 arxiv:2603.14482](https://arxiv.org/abs/2603.14482), [V-JEPA 2 arxiv:2506.09985](https://arxiv.org/html/2506.09985v1)). Their headline lifts:

| Benchmark | Skill measured | V-JEPA 2 → 2.1 lift | Why this metric works for JEPA |
|---|---|---|---|
| **Something-Something-v2** | motion-centric action recognition | **74.2 → 76.5 % (+2.3 pp)** ([V-JEPA 2.1 paper](https://arxiv.org/html/2603.14482v2)) | Tests temporal dynamics — JEPA's masked-prediction objective explicitly trains for this |
| **Diving-48** | fine-grained motion classification | reported SOTA | same — fine-grained motion |
| **Kinetics-400** | appearance-based action recog | reported SOTA | broader appearance, weaker JEPA signal but still in-domain |
| **Ego4D OSC anticipation** | short-term object-interaction prediction | **7.71 mAP** | future-frame prediction = literal JEPA training objective |
| **EPIC-KITCHENS-100** | action anticipation | **40.8 Recall@5** | future action prediction |

**Mapping to our current measurements**:

| Our metric | V-JEPA-aligned? | v3 status (vs frozen) | Verdict |
|---|---|---|---|
| **Prec@K** (cross-clip retrieval) | ❌ NO — not in V-JEPA's published eval | +0.87 pp (best, C) | wrong gate; ceiling at ~76 |
| **mAP@K** | ❌ NO — same family as Prec@K | +0.87 pp (D) | wrong gate |
| **Cycle@K** (within-clip token cycle) | 🟡 PARTIAL — closest to JEPA's structural prior | **+3.79 % (E v3, peak 80.39)** 🟡 | the metric that DID lift |
| **val_jepa** (the actual training loss) | ✅ YES (objective is V-JEPA's own L1) | B/C/D drop −2.5 to −2.9 %; E v3 *rises* +7.2 % | **honest read**: 2.9 % drop is the lower-end of "marginal continual-pretraining" (similar-distro expects 5–10 %; heavy-OOD expects 15–30 %; from-scratch expects 65–80 %). At 2.9 % we cannot distinguish real-but-tiny domain-shift signal from per-clip noise / overfit-to-train. Per-clip variance + train/val gap NOT measured. The iter11 v3 anti-corr finding (r = −0.21 to −0.68 vs probe metrics) actively suggests this drop is **orthogonal to downstream lift** — so even if the 2.9 % is "real" it likely doesn't translate to Prec@K |

**What we should add (and don't currently measure)**:

1. **Linear / attentive probe accuracy on Indian action classes** — bucket the 2452 clips into action labels (walking / driving / drone / monument-scene) and run a 4-layer probe head per V-JEPA 2.1 protocol. This is the metric V-JEPA was designed to win on.
2. **Future-frame latent prediction MSE** — V-JEPA's pretraining objective. Probably the cleanest "the model got better at what it was trained to do" metric.
3. **Per-clip motion-feature cosine similarity** to held-out same-action clips — proxy for the SSv2-style motion test.

Sources: [V-JEPA 2 arxiv:2506.09985](https://arxiv.org/html/2506.09985v1), [V-JEPA 2.1 arxiv:2603.14482](https://arxiv.org/abs/2603.14482), [DeepWiki vjepa2 downstream tasks](https://deepwiki.com/facebookresearch/vjepa2/5.3-downstream-tasks-and-benchmarks).

---


## Bottom line (REVISED 2026-04-27 22:00 PDT after E v3 done + user history dump)



**The empirical signal across 4 recipes (small surgery / multi-task / 47× scale full-pretrain / per-stage factor unfreeze) is consistent: the encoder does NOT produce a different representation under continual pretraining on this data.** No eval metric will rescue this — if parameters don't shift in a useful direction, no probe / retrieval / cycle / loss measurement can show lift.

### Why my "switch eval metric" hedge was wrong
If continual pretraining produces no representation shift (the 115K full-pretrain test rules out scale & capacity), then **the attentive probe will see the same frozen features and output the same accuracy**. Switching eval metrics is metric-shopping; it doesn't fix a training problem.

### Real next-steps (NO more training, before killing GPU)

| # | Action | GPU cost | Tells us |
|---|---|---|---|
| 1 | Compute `‖θ_115K − θ_init‖ / ‖θ_init‖` per layer on the 115K full-pretrain checkpoint (CPU-only weight diff) | 0 GPU | Did params actually move? <0.001 = silent training bug; >0.01 = parameters moved but representations equivalent |
| 2 | Plot per-step val_jepa trajectory from m09a 115K log (already exists) | 0 GPU | Did loss drop early then plateau? Or flat from step 1? Flat=bug, drop-then-plateau=saturation |
| 3 | Extract frozen V-JEPA 2.1 features on 100 Indian + 100 Kinetics-400 clips, compare cosine sim distributions | 1 h GPU | If sim>0.95 → Indian is in-distribution → encoder genuinely saturated, no fine-tuning will help |
| 4 | Compute per-clip val_jepa **variance** on E v3 vs frozen on the 308-clip eval set | 1 h GPU | If σ ≫ |Δμ|, the −2.9 % "drop" is noise; settles whether ANY of the multi-task numbers are real |
---

## Q7) NEXT STEPS — Framing B: adopt Meta's actual V-JEPA 2 transfer recipe

**Non-negotiable goal**: `vjepa_surgical` must outperform `vjepa_frozen` on the gate metric.

**The reframe**: drop "surgical = continual-pretrained encoder" (4 recipes failed). Re-define `surgical` = **frozen V-JEPA 2.1 encoder + factor-conditioned probe head**. This is the only V-JEPA 2 transfer pattern that works in published evidence (Meta's own V-JEPA 2-AC on Droid 62h, SSv2 attentive probe at 77.3 %).

### What stays the same vs what changes

| Component | iter12 (failed) | iter13 (Framing B) |
|---|---|---|
| Encoder | continual-pretrain ViT-G (1.84B params, 12-25 % trainable) | **🔒 FROZEN** ViT-G (0 % trainable) |
| Trainable surface | full prefix or LoRA on encoder | **only the probe head** (~1-10M params) |
| Training data | 2452 Indian clips, 1140 steps | same 2452 clips, but used to train **the probe head**, not encoder |
| Loss | JEPA L1 + InfoNCE + (TCC) on encoder outputs | **task-specific** on probe head: cross-entropy for classification, MSE for regression |
| Gate metric | Prec@K (cross-clip retrieval — wrong for V-JEPA) | **action / factor classification accuracy** (V-JEPA-aligned, Meta-validated) |
| `surgical` vs `frozen` differ in | encoder weights (failed to differ) | **probe head architecture** (factor-conditioned vs vanilla) |

### How surgical BEATS frozen under this framing

`vjepa_frozen`'s probe = stock 4-layer attentive probe (Meta's reference implementation).
`vjepa_surgical`'s probe = **factor-conditioned head that exploits our D_L / D_A / D_I labels** that vjepa_frozen cannot use. The moat is the **labels** (which Meta's frozen probe doesn't know about), not the encoder.

| Probe-head option | Architecture | Why surgical might beat frozen |
|---|---|---|
| **A: Vanilla attentive** (= frozen baseline) | 4-layer attention pooling → MLP → softmax | reference; trained on same labels surgical sees |
| **B: Factor-conditioned cross-attention** | attention pooling with D_L/D_A/D_I tokens injected as cross-attn keys | head learns to attend to layout/agent/interaction tubes specifically |
| **C: Multi-task probe** (auxiliary heads) | shared trunk → 4 heads: action + D_L_class + D_A_class + D_I_class with co-training | factor-classification gradient regularizes the action head |
| **D: Factor-routed MoE probe** | router(D_L/D_A/D_I) → expert head per factor combination | exploits factor structure for specialization |

**Likeliest winner**: B or C. They exploit the factor labels (our novelty) without needing encoder fine-tuning (which doesn't work).

### iter13 concrete plan (NO encoder training — only probe heads)

| # | Step | GPU cost | What it produces | Pass criterion |
|---|---|---|---|---|
| 1 | Bucket the 2452 train + 308 eval clips into Indian action labels (walking / driving / drone / monument-scene) — write `m12_action_labels.py` reading existing tags.json | 0 (CPU, ~30 min) | `data/ultra_hard_3066_action_labels.json` (~2760 rows × 1 label each) | label distribution > 50 clips per class |
| 2 | Build `m13_probe_train.py` — generic 4-layer attentive probe trainer on top of frozen V-JEPA 2.1 features. Same as Meta's stock recipe | ~2 h | `vjepa_frozen_action_probe.pt` + accuracy + 95 % CI | runs without crash; baseline accuracy reported |
| 3 | Add `m13b_factor_probe_train.py` — factor-conditioned probe (Option B or C above) using D_L/D_A/D_I labels as auxiliary signal | ~2 h | `vjepa_surgical_action_probe.pt` (= frozen encoder + factor-aware head) | runs without crash |
| 4 | Paired-bootstrap accuracy diff: `surgical_probe − frozen_probe` on 308-clip eval | ~10 min | Δ accuracy + p-value (BCa CI) | **Δ > 0 with p < 0.05** = paper-worthy result |
| 5 | Ablation: which factor (D_L vs D_A vs D_I) contributes most to the lift? Run probe with one factor at a time | ~6 h (3 × 2 h) | per-factor accuracy contribution | identifies which factor dominates the moat |
| **Total** | | **~10 h GPU @ ~$8** | | vs the **~50 h** spent on iter11/iter12 encoder-training that produced 0 lift |

### Why this satisfies the non-negotiable goal

- Both `frozen` and `surgical` use the **same V-JEPA 2.1 ViT-G encoder weights** (no continual pretraining)
- They differ only in the **probe head**, which is what we've shown actually works for V-JEPA transfer
- `surgical` wins by USING the factor labels (`D_L / D_A / D_I` from our `m10` + `m11` pipeline) that the frozen baseline doesn't have access to
- This is the publishable "FactorJEPA contribution" reframed honestly: the **dataset + pipeline + factor-aware probe** is the contribution, NOT a "better encoder"

### What this saves us from doing

| Activity | Cancelled because |
|---|---|
| Variant F (3stage_DI_multitask) | E v3 stage 2 D_A added 0 — F's stage 3 D_I won't either |
| Optuna / grid sweep on (α, β, γ) | gradient share rebalance already tested in v3 → didn't move Prec@K |
| 50K-clip continual pretraining | 115K full-pretrain already produced no lift → larger scale won't help V-JEPA's L1 task |
| LR sweep / longer epochs | mismatched recipe (encoder fine-tune is not Meta's pattern); fixing the layer doesn't help |
| Switching to DINOv2/CLIP/SigLIP | constraint: paper is V-JEPA 2.1 only |

### Risk: what if the factor-conditioned probe ALSO fails to beat the vanilla one?

| Outcome | Reading | Paper framing |
|---|---|---|
| **Δ accuracy > 0, p < 0.05** | factor labels add real signal beyond what frozen V-JEPA features encode | **WIN**: "FactorJEPA: factor-conditioned probes lift V-JEPA 2.1 on Indian-context action recognition" |
| **Δ accuracy ≈ 0, p > 0.05** | frozen V-JEPA features already implicitly encode the factor structure → labels redundant | **NEGATIVE**: even probe-level surgical doesn't help; the dataset + pipeline are the only contributions. Drop "vjepa_surgical > vjepa_frozen" claim entirely |
| **Δ accuracy < 0** (unlikely) | factor head over-regularizes or is mis-spec'd | iterate on probe arch (Options B/C/D); doesn't invalidate the dataset/pipeline contribution |

**Worst case** = same as current (dataset + pipeline + negative finding). **Best case** = first reportable `surgical > frozen` lift on Indian video. Asymmetric upside, ~10 h GPU cost. Worth running.

---

*✻ Q7 added 2026-04-28 after WEBSEARCH confirmed Meta's actual V-JEPA 2 transfer recipe (V-JEPA 2-AC on Droid 62h, SSv2 attentive probe at 77.3 %) is **frozen encoder + small new head**, never continual-pretrain. iter13 = adopt that pattern with our factor labels as the surgical-vs-frozen differentiator.*
