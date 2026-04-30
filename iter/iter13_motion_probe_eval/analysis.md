# Deep-research answers — V-JEPA 2.1 ultrathink

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
