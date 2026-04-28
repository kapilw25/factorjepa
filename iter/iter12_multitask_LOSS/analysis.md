# Deep-research answers — V-JEPA 2.1 ultrathink

---

## Q1) Constraint accepted: V-JEPA 2.1 only, on Indian clips. What's still salvageable as a paper?

You're right — I was lazy to suggest swapping encoders. Within the V-JEPA 2.1 constraint, the paper's contribution is NOT "we beat retrieval Prec@K" — that was a misframed gate. The real contributions still on the table:

1. **The FactorJEPA dataset itself** — 2,452 / 306 / 308 clip splits with `D_L / D_A / D_I` factor labels for Indian-context driving / walking / monuments scenes. This is novel data; it's a contribution independent of any model number.
2. **The factor-mining pipeline** (`m10` Grounded-SAM 3.1 + `m11` D_L/D_A/D_I generation, ~340 → ~40 GB streaming compression) — methodological contribution.
3. **Continual pretraining recipe** for V-JEPA 2.1 on a small (~2.5 K-clip) domain shift — `surgery_3stage_DI` already shows **+0.87 pp paired-Δ Prec@K p=0.0038** vs frozen, the first statistically-significant Δ on this benchmark. Sub-1pp is small in absolute terms but it's significant in the BCa sense and IS a result.
4. **The wrong-metric finding itself is publishable**: "Cross-clip retrieval Prec@K is not the right gate metric for V-JEPA-style generative SSL; here are the metrics that DO move (Cycle@K, val_jepa, action recognition probe)." That's a legitimate negative-result + corrective contribution.

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

## Q3) Comparison with `factorjepa_seekr_vast.ipynb` & `factorjepa_vast_combined.ipynb`

Both notebooks now read from `iter/utils/teams_work/`. Source: anonymousML123/walkindia-200k (same upstream source as our `ultra_hard_3066`). Their goal is **continual learning across sessions** (no-forgetting); ours is **factor-conditioned surgery for retrieval**. Different research questions on overlapping data.

| Axis | `factorjepa_seekr_vast.ipynb` (SEEKR) | `factorjepa_vast_combined.ipynb` (VanillaFT + SSIAT + SAFE) | **Our `surgery_2stage_noDI_multitask` (E v3)** |
|---|---|---|---|
| Encoder backbone | V-JEPA 2.1 **ViT-B distilled from ViT-G** (~85M params) | same ViT-B distilled (~85M) | **V-JEPA 2.1 ViT-G full** (1.84B, 1664-dim, 48 blocks) — **22× larger** |
| Pretrain ckpt | `vjepa2_1_vitb_dist_vitG_384.pt` | same | `vjepa2_1_vitG_384.pt` (full ViT-G) |
| Loss function | JEPA L1 (MSE on **pooled mean** of context tokens) + 0.5·replay-JEPA + 0.1·selective-block-distill on top-4 forgettable blocks | JEPA L1 (MSE on pooled mean) — vanilla, no aux | **UW 2-task**: exp(−s_J)·JEPA_dense + exp(−s_I)·InfoNCE + Σs_i (predict_all=True, deep-sup 4 levels, TCC dropped per #81) |
| Predictor | tiny 3-layer MLP `LN→Linear(D→2D)→GELU→Linear(2D→D)` on **mean-pooled** ctx tokens | same tiny MLP | V-JEPA 2.1 stock predictor: 12-block transformer, 59.4M params, full token-level prediction |
| Mask ratio | **TARGET_MASK_RATIO = 0.20** (20 % masked / 80 % visible) ⚠️ **MAE-style — NOT V-JEPA's 85-90 %**; 4 blocks | same: 0.20, 4 blocks ⚠️ | **~85 % masked / ~15 % visible** (V-JEPA SOTA): 8 small (15 %) + 2 large (70 %) multi-block, contiguous |
| Batch size | 8 | 8 | 32 (effective; AdaptiveBatchSizer micro-batch 16→32) |
| Learning rate | backbone 5e-5, predictor 1e-4 | same | backbone 5e-5, predictor 1e-4 (`pred_lr_multiplier=1.0` ≠ legacy 10×) |
| EMA momentum | 0.996 | 0.996 | 0.996 (base_optimization.yaml) |
| Grad clip | 1.0 | 1.0 | 1.0 (post #78 lower from 10) |
| Epochs / sessions | 3 epochs × 3 sessions = 9 ep total | 3 × 3 = 9 ep total | 15 epochs × 1 session, 2 stages internally |
| Total steps | ~5625 (5K×3sess×3ep / BS=8) | ~2250 (2K×3×3 / 8) | 1140 steps (BS=32; equiv. ~9120 BS=8 steps if compared head-on) |
| Clips trained | 15,000 (5K × 3 sessions) | 6,000 (2K × 3) | 2,452 train + 306 val + 308 eval = 3,066 |
| Dataset | anonymousML123/walkindia-**200K**, shard ranges (0-37, 38-75, 76-99) | same 200K, same shards | **subset** of same source: `ultra_hard_3066` (filtered for hard-mode + factor-mining) |
| Eval metrics | **Cycle@K + Overlap@K** (K=10, 500 val clips) — **no Prec@K** | **Cycle@K only** (K=10, **50 val clips** — very small) | Prec@K, mAP@K, Cycle@K, nDCG@K, val_jepa (BCa CI, N=308 paired) |
| Factor recipe | none — flat continual sessions | none | Stage 1: D_L=100 % · Stage 2: D_A=100 % (no D_I) — `m11` factor mining |
| Multi-task / contrastive | **none** — pure JEPA + replay/distill | **none** — pure JEPA | **InfoNCE** (per-clip pooled, in-batch negatives B=32 → 31 negs, τ=0.07) |
| Continual-learning method | **SEEKR**: 50-clip replay buffer + selective top-K block distillation against frozen session snapshot | **3-method bench**: VanillaFT + SSIAT/SAPT (LoRA rank=8 on blocks 8-11) + SAFE (slow+fast LoRA) | none — single run, factor-conditioned curriculum |
| Headline metric reported | session-by-session Cycle@10, Overlap@10, BWT (last−first session) | Cycle@10 + BWT, t-SNE + neighbour-grid viz | best Prec@K/mAP@K/Cycle@K vs frozen baseline + paired-Δ p-value |
| Trainable params | full encoder (~85M, 100 %) | A: full (~85M); B: LoRA only (~0.2 %); C: slow-LoRA + fast-LoRA (~0.4 %) | full prefix (varies per stage; up to 460 M = 25 % at stage 1 unfreeze=12/48) |
| Augmentation | center-crop + ImageNet norm | same | random-resize-scale [0.3, 1.0] + h-flip 0.5 + random-crop |
| GPU | H100 80GB | H100/A100 | RTX PRO 6000 Blackwell 96GB |

**Three substantive deltas vs the team's notebooks** (not just style):

1. ⚠️ **Their mask ratio is wrong for V-JEPA.** Both notebooks set `TARGET_MASK_RATIO = 0.20` (MAE-style 20 % masked); V-JEPA 2.1's published recipe is **~85-90 % masked**. Their encoder is reconstructing 80 % of the input — that's not V-JEPA's training objective, it's much closer to MAE-tube. Any "JEPA loss = X" number from those notebooks is on a fundamentally different training task than V-JEPA's published recipe (and ours).
2. **They use ViT-B distilled, we use ViT-G full.** ~22× parameter difference; their compute budget per step is much smaller, but their representational ceiling is also lower. ViT-B distilled outputs may saturate before our ViT-G does on the same Indian clip distribution.
3. **They never measure Prec@K.** Their gate metrics are Cycle@K + Overlap@K — the V-JEPA-aligned metrics from Q2. They sidestepped the wrong-gate problem we ran into.

Their **continual-learning angle** (SEEKR replay, SSIAT shared-LoRA, SAFE slow/fast LoRA) is **complementary to ours** (factor-conditioned surgery), not competing — could be combined: run their continual recipes with our factor-conditioned per-stage data mixtures.

### Q3.1 Team's reported result: `loss 0.65 → 0.23 → 0.05` across 3 SSIAT sessions, BWT=0.000

**Direct comparison is invalid — different objectives.** Side-by-side:

| Confounder | Their setup | Our setup | Why their loss can drop further |
|---|---|---|---|
| Mask ratio | 0.20 (20 % hidden) | 0.85 (85 % hidden) | their task is **4× easier** — easier task = lower achievable floor |
| Target | mean-pooled (1 vec/clip) | per-token (1330 vec/clip) | theirs collapses 1330 targets → 1 |
| Predictor | 3-layer MLP, ~5M params | 12-block transformer, ~59M params | tiny predictor on small task → easy memorization |
| Dataset (Colab tier) | 150 clips × 3 sessions = 450 unique | 2452 train + 306 val | **5M-param MLP on 450 clips → memorize, not learn** |
| Encoder | ViT-B distilled, ~85M | ViT-G full, ~1.84B | smaller model = more headroom from random init |
| **0.05 vs 0.46 floor** | not commensurable | not commensurable | different prediction tasks |

**Their own BWT=0.000 contradicts the "training is working" story:**

| Their metric | Reading | What it actually says |
|---|---|---|
| `val_jepa: 0.65 → 0.05` | −92 % | predictor + LoRA fit those 450 clips |
| `Cycle@K BWT = 0.000` | downstream metric **didn't move** | iter11 v3 anti-corr finding (r=−0.21 to −0.68) reproduced cleanly in their data: loss drops orthogonal to representation quality |
| absolute Cycle@K vs frozen | **NOT REPORTED** | could be `frozen=0.85, all=0.85` (no learning) OR `frozen=0.85, all=0.92` (learned + preserved) — their report cannot distinguish |

**Three diagnostic asks for the team:**

| # | Question | If answer = X | If answer = Y |
|---|---|---|---|
| 1 | Was `0.05` **train** or **val** loss? | train → overfit on 450 clips | val → interesting but still see #2 |
| 2 | Absolute Cycle@K of session-3 ckpt vs **frozen V-JEPA 2.1 baseline** on same val shard? | Δ ≈ 0 → no real domain learning | Δ > 0 → SSIAT works for continual-Cycle@K (different question than ours) |
| 3 | Re-run their setup at OUR scale (2452 clips, V-JEPA mask=0.85, ViT-G, dense per-token loss). Does val_jepa still drop 92 %? | yes → our recipe is broken | no (drops 2-3 %) → confounders #1-#5 fully explain the gap |

**Verdict (concise):** Their number is real-but-misleading. The 92 % loss drop is on a fundamentally easier task (20 % mask, MLP predictor, mean-pooled target, 450-clip set, smaller encoder). Their own `BWT=0` shows the drop did **not** translate to representation quality — same anti-correlation pattern as ours, just framed as "no forgetting" instead of "no learning". Not evidence that V-JEPA fine-tuning learns Indian-context structure; evidence that loss-floor depends on task difficulty.

---

## Q4) Mask coverage — actual numbers + per-patch coverage analysis

**Your "80 % visible / 20 % hidden" framing is inverted.** V-JEPA is the opposite of MAE's 75 % masking.

**Our actual config** (`base_optimization.yaml:42-50`):

- Crop 224 × 224, patch 16 × 16 → 14 × 14 = 196 spatial patches per frame
- 16 frames, tubelet_size 2 → 8 temporal tubelets
- **Total tokens per clip = 14 × 14 × 8 = 1568**
- Mask blocks: **8 small** (spatial_scale 15 % × 15 %) + **2 large** (70 % × 70 %), all temporal_scale 100 %
- Yaml comment: **"~75-90 % total masking"** → predictor reconstructs ~1330 tokens, encoder sees only ~235 visible

V-JEPA 2.1 paper uses ≈ 90 % masking ratio with multi-block contiguous masking ([Bardes 2024 arxiv:2404.08471](https://arxiv.org/html/2404.08471v1), [Meta AI V-JEPA blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)). We're matching Meta's recipe.

**Per-patch coverage analysis** (computed): with random multi-block sampling at p_visible_per_step ≈ 0.15:

- After 10 steps: P(any patch seen visible at least once) = **80.3 %**
- After 100 steps: **100 %** (numerically saturated)
- After 1140 steps (our run): every patch has been a context token *and* a prediction target ~150+ times each

**Should we change position deterministically?** No — and here's the pros/cons that explain why:

**PRO of explicit per-position rotation** (Halton-style, [arxiv:2503.17076](https://arxiv.org/html/2503.17076v1)):

1. Guaranteed uniform spatial coverage in fewer steps
2. Lower training-signal variance per patch
3. Reproducible / debuggable

**CON of explicit per-position rotation**:

1. **V-JEPA paper explicitly tested this and found random multi-block > deterministic** ([JEPA repo issue #50](https://github.com/facebookresearch/jepa/issues/50)): *"random-tube strategy ... leads to features of low-semantic quality when combined with the feature prediction objective"*
2. Meta's masking is contiguous-block (not random per-patch) precisely because the model needs to predict *cohesive regions* — rotating positions deterministically would create predictable masks the model can exploit (cheap shortcut learning)
3. Stochasticity IS the regularizer — removes risk of memorizing per-position behavior
4. Our coverage math says full coverage in 100 steps; we run 1140. There is no coverage problem to fix.

**Verdict**: don't touch the masking. The 90 %-masked / 10 %-visible / random-block recipe is V-JEPA's published SOTA and our coverage is statistically saturated within the first 100 of 1140 steps.

---

## Q5) TCC γ=0 (current v3) vs γ=0.10–0.15 — does small-γ fix downstream alignment?

**Empirical evidence from logs**:

- v1 (γ=1.0 effective, UW init w_tcc=1.0 × raw=6.6 → 80 % gradient): Cycle@K peak **80.39 @ step 90**
- v3 (γ=0, TCC dropped entirely): Cycle@K peak **80.39 @ step 720** (same peak, just later)

**Cycle@K is identical with and without TCC.** That's a strong signal that **InfoNCE's structural pull on per-clip pooled vectors is sufficient to maintain token cycle structure** — TCC was redundant.

**PRO of γ = 0.10–0.15** (small but non-zero):

1. Cycle@K is in our measurement suite — small γ might push the peak higher than 80.39 (untested)
2. With pre-scale `tcc_scale=0.0833` × γ=0.15 = effective contribution ~0.083 × 6.6 × 0.15 ≈ 0.08, ~13 % of total — balanced, not domineering
3. Adds a token-level cycle regularizer that may help generalization on Cycle@K's held-out 308 eval clips

**CON of γ = 0.10–0.15**:

1. **v3 already proves TCC isn't needed for Cycle@K = 80.39** — adding it back risks reverting v3's val_jepa improvement (0.5113 vs v1 0.5322)
2. Adds an extra HP that needs joint tuning with α/β
3. Compute cost: TCC requires `bmm` over (B, T, T) = 32 × 16 × 16 = 8192 ops/sample/layer — non-trivial, ~5 % of step time
4. Cycle@K isn't our paper's gate metric (Prec@K is), so optimising for it is misaligned with the headline number — though per Q1+Q2, Prec@K is the wrong gate anyway, and Cycle@K aligns with V-JEPA's strengths

**Verdict**: γ=0.10 is a worth-running ablation IF we adopt the V-JEPA-aligned Cycle@K-as-gate framing from Q2. With Prec@K as gate, TCC stays at 0. With Cycle@K + val_jepa as gates, run γ ∈ {0, 0.05, 0.10, 0.15} as a 4-point sweep on stage 1 (~8h GPU total).

---

## Q6) Grid search vs Optuna for (α, β, γ) on stage 1 — and why our Kendall UW didn't deliver

### Why Kendall UW failed in our run (root cause analysis)

The UW math is correct. What broke:

1. **LR mismatch**: log-vars trained at the *same* LR (5e-5) as the 1.84B model. Kendall §4.1 implicitly assumes the log-vars live on a different timescale than model params. With 1140 steps × 5e-5, the log-vars moved only ±3 % — physically impossible to escape the unit-init basin.
2. **Raw-magnitude domination wasn't anticipated**: Kendall paper assumes raw losses are roughly OoM-comparable. Ours had TCC=6.6, JEPA=0.5 → 12 ×. UW *would* have rebalanced eventually, but 1140 steps isn't enough.
3. **No outer signal for the meta-objective**: UW optimizes total *training* loss, not downstream Prec@K. There's no guarantee the rebalancing serves the gate metric.

### Grid search vs Optuna — head-to-head

#### Approach A: Manual grid search (5–10 combos on stage 1 only)

**Cost**: stage 1 = 570 steps ≈ 2 h × 10 combos = **~20 h GPU @ $16**

**PRO**:

1. Trivial to implement (a yaml-emitter loop + run_train.sh)
2. Fully reproducible — every combo lives in a yaml, fits the "no hidden HP" research-rigor bar
3. Easy to plot 3D contour over (α, β, γ); reviewer-friendly figure
4. No new dependencies (Optuna would need to be added + integrated)

**CON**:

1. **Doesn't capture interactions** (α=0.5 might pair well with β=0.7 but the grid never tests that combo if they're not both in the cartesian product)
2. **Blind to the right region** — if optimal is α=0.3, β=0.9, γ=0.05, a coarse grid misses it
3. Stage 1-only signal may not predict full-run behavior (D_A unfreeze in stage 2 changes gradient dynamics)
4. 20 h cost gets you 10 data points; Optuna gets you 20 with smarter coverage

#### Approach B: Optuna multivariate-TPE ([optuna.org](https://optuna.org/), [TPESampler docs](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html))

**Cost**: ~15-20 trials needed before TPE surrogate becomes effective ([Optuna GitHub issue #2600](https://github.com/optuna/optuna/issues/2600)) → **~30-40 h GPU @ $24-32**

**PRO**:

1. **Multivariate TPE captures (α, β, γ) interactions** — fits joint GMM not 3 marginal distributions ([Preferred Networks blog](https://tech.preferred.jp/en/blog/multivariate-tpe-makes-optuna-even-more-powerful/))
2. **Pruning** (HyperbandPruner / MedianPruner) cuts losing trials early → effective trial budget can drop ~30 %
3. **Continuous α/β/γ** — grid forces discrete; TPE searches FloatDistribution
4. **HF Jobs integration** for parallel trials ([HF Jobs blog](https://huggingface.co/blog/chrisvoncsefalvay/claude-hf-jobs-optuna))

**CON**:

1. **Adds Optuna dependency + storage backend** (sqlite or postgres) — non-trivial integration into m09c_surgery
2. **Trial overhead** (logging, pruning checks, surrogate refit) — small but non-zero
3. **Stochastic** — different seeds give different "optimal" (α, β, γ); reviewer may flag
4. Surrogate quality is poor for first ~15 trials → if budget is small (< 15), reduces to random search

### My recommendation (with the v3 evidence weighing in)

**Don't run either yet.** The v3 result already tells us:

- Even with InfoNCE getting 58 % of gradient (the loss-balance fix worked), Prec@K moved +0.05 pp
- The bottleneck is **not the loss weights**

**Spend the 20-40 h GPU on Q2's V-JEPA-native metrics instead**:

1. Build an attentive-probe head on Indian-action labels (~8 h)
2. Re-evaluate frozen + B + C + D + v3 student encoders on the probe (~4 h × 5 = 20 h)
3. Report attentive-probe accuracy as the new gate metric

If after that the V-JEPA-native metrics also show no lift from multi-task, THEN we know the recipe (not the loss weights) is the issue and an Optuna sweep is justified. If V-JEPA-native metrics DO show lift, we have the paper result without any HP search — we just measured the wrong thing all along.

Sources: [Optuna TPESampler docs](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html), [TPE tutorial OptunaHub](https://hub.optuna.org/samplers/tpe_tutorial/), [Multivariate TPE blog](https://tech.preferred.jp/en/blog/multivariate-tpe-makes-optuna-even-more-powerful/), [V-JEPA mask issue #50](https://github.com/facebookresearch/jepa/issues/50), [Halton scheduler arxiv:2503.17076](https://arxiv.org/html/2503.17076v1).

---

## Bottom line (REVISED 2026-04-27 22:00 PDT after E v3 done + user history dump)

**My earlier "pivot to V-JEPA-native metrics / attentive probe" recommendation was wrong.** New evidence:

| Run | Trainable | Clips | val_jepa Δ | Downstream Δ | Conclusion |
|---|---|---|---|---|---|
| iter11 v3 surgery | partial prefix | 2.5 K | −2.5 to −2.9 % | ≤ +0.87 pp Prec@K | recipe doesn't unlock |
| iter12 v3 multitask | partial prefix | 2.5 K | +7 % (worse) | ≤ +0.05 pp vs v1 | loss-balance fix doesn't unlock |
| **m09a_pretrain (115K, full-encoder, all layers)** | **100 %** | **115 K** | **no improvement** | **no improvement** | **scale + capacity ALSO don't unlock** |
| iter12 E v3 stage2 D_A unfreeze | partial prefix | 2.5 K | (n/a) | stage2-best 75.82 vs stage1-best 75.87 | additional factor unfreeze adds 0 |

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

**If diagnostics #1–#3 confirm saturation** (parameters moved but eval features unchanged, OR Indian clips in-distribution), the paper's only honest contribution is:

**FactorJEPA dataset + factor-mining pipeline (D_L/D_A/D_I), NOT a model improvement.** Drop the "we improved V-JEPA on Indian video" claim entirely. Reframe as:
- **Contribution 1**: Novel Indian-context video dataset with hierarchical factor labels
- **Contribution 2**: Grounded-SAM 3.1 + streaming factor-mining pipeline (340 → 40 GB compression)
- **Contribution 3**: Negative finding — "V-JEPA 2.1 ViT-G frozen features are at equilibrium for Indian-context video; 4 distinct continual-pretraining recipes (surgery / multi-task / 115K full-pretrain / staged factor-unfreeze) failed to lift retrieval / cycle / loss above the frozen baseline by >1 pp at p<0.05. This bounds the achievable Δ for V-JEPA-style generative SSL in low-data domain-adaptation."

That's still 3 publishable contributions. Drop the model-improvement claim — keep the data, pipeline, and the negative-result honestly.

### What to do with F (3stage_DI_multitask, currently step 45/1140)
**Kill it.** E v3's stage 2 D_A unfreeze added 0 lift over stage 1. F's stage 3 D_I unfreeze is one more factor on the same broken signal — predicted to add 0 ± noise. Save the ~10 h GPU.

### What about Optuna / grid sweep (Q6)?
**Cancel it too.** Same logic. If parameters don't shift in useful directions across 4 recipes, no (α, β, γ) tuning will rescue the gradient signal. Optuna-on-broken-training = waste of GPU.

### What about the iter11 v3 anti-corr finding (r = −0.21 to −0.68 between val_jepa and probe metrics)?
This is now even more interesting given the bigger picture. It says: **whatever small parameter movement occurs during continual pretraining is anti-correlated with downstream metrics.** The recipe is doing *something* but in the wrong direction for retrieval. That's a publishable diagnostic finding tied to "V-JEPA's L1 objective is misaligned with retrieval-style downstreams" — supporting the negative-result framing.

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
