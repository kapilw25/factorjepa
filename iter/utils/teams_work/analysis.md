# Deep-research answers — V-JEPA 2.1 ultrathink

---

## Comparison with `factorjepa_seekr_vast.ipynb` & `factorjepa_vast_combined.ipynb`

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
