# 🔍 FactorJEPA / DENSEWORLD — paper claims vs. delivered artifacts

> **Audit date:** 2026-07-22 · **Evidence roots:**
> `outputs/poc/{vjepa_2_1_vitG_2B,vjepa_2_1_vitg_1B}/eval/eval_10k/…/metrics_watch/eval_metrics.json` ·
> `outputs/full/vjepa_2_1_vitg_1B/eval/full/…/metrics_watch/eval_metrics.json` ·
> `iter/iter17_ablations_model/result_outputs/v17a_frozen_eval/` · `configs/` · `src/`
>
> **Active paper files** (`0_main_AAAI.tex` / `0_main_neurips.tex`): `1_introduction`, `2_data`,
> `2_factor_jepa`, `8_conclusion`, `11_appendix`. `0_main.tex` additionally pulls `10_india` +
> `12_new_plan` — see §5.

---

## 1. ✅ What reproduces EXACTLY

Every headline number in `2_factor_jepa.tex` was recomputed from `eval_metrics.json` using the
`plot_forest` recipe in `src/m13_eval_plot.py`. **All 8 forest values and all 4 Spearman ρ match to
the printed precision.**

| Claim (paper) | §  | Recomputed | Verdict |
|---|---|---|---|
| Future-frame L1 vs frozen: 47.0× / 27.2× | L1374 | 47.0× / 27.2× | ✅ |
| Causal L1 vs frozen: 28.7× / 19.0× | L1375 | 28.7× / 19.0× | ✅ |
| Motion cosine vs frozen: 26.8× / 25.6× | L1376 | 26.8× / 25.6× | ✅ |
| Mask-ratio vs frozen: 22.3× / 15.7× | L1377 | 22.3× / 15.7× | ✅ |
| Future-frame L1 vs best comp: +1.7% / +1.4% (6.3× / 4.8×) | L1392 | +1.69% / +1.44% (6.3× / 4.8×) | ✅ |
| Causal L1 vs best comp: +0.9% / +1.1% (2.3× / 2.7×) | L1394 | +0.90% / +1.06% (2.3× / 2.7×) | ✅ |
| Mask-ratio vs best comp: +1.1% / +2.9% | L1396 | +1.15% / +2.90% | ✅ |
| Motion cosine trails full-FT by 44.9% / 33.4% | L1401 | −44.92% / −33.41% | ✅ |
| ρ = 0.978 / 0.952 / 0.938 / 0.895 | L1431 | 0.978 / 0.952 / 0.938 / 0.895 (n=14 arms) | ✅ |
| 3 diagnostics don't transfer | L1434 | tov −0.108, teacher_free −0.248, rollout −0.600 | ✅ |
| LoRA/DoRA matched rank+scaling | L113 | r=16, α=32, targets `qkv,proj,fc1,fc2`; DoRA = +`use_dora` | ✅ |
| 384×384, 16 frames, patch 16, tubelet 2 | app. Panel A | matches `configs/model/vjepa2_1.yaml` + `pipeline.yaml` | ✅ |
| 22 cities | L23 | docs site: 714 videos · 22 cities | ✅ |

⚠️ **Caveat on "12 of 15 transfer":** `playback-pace` is ρ=0.308 — a 4th weak transfer the sentence
implicitly counts as consistent.

---

## 2. ❌ NOT DELIVERED — remove or rewrite

### 2.1 🎬 Latent-to-RGB Future Decoding (the whole of §`sec:latent_to_rgb` + `app:rgb_decoder`)

| Paper element | Status |
|---|---|
| Cosmos-initialized decoder `G_ψC` | ❌ never built. What exists is `src/m15_pixel_decoder.py`: a per-token MLP `1408 → 2048 → 1536` that inverts a token to **its own tubelet** (2×16×16×3). No Cosmos component anywhere. |
| Transport operator `T_ω` (LN → W_in → reshape → B_ω → W_out) | ❌ no such module |
| Table `tab:latent_rgb` (PSNR/SSIM/LPIPS/Flow EPE/Agent F1) | ❌ **all 20 cells are literal `--`** (`2_factor_jepa.tex:1722-1725`) |
| Oracle-vs-forecast error decomposition | ❌ never computed |
| Factor-removed decoding `x̂^(−k)`, influence maps `Δ_k` | ❌ no code (requires the factor blocks, which also don't exist — §2.3) |
| `tab:rgb_decoder_contract` | ⚠️ 8 fields still say *"from log"* |
| Abstract sentence "We introduce a Cosmos-initialized latent-to-RGB decoder…" | ❌ unsupported |
| Conclusion sentence "a Cosmos-initialized latent-to-RGB decoder renders predicted futures, while an oracle control separates…" | ❌ unsupported |

**Action:** delete §`sec:latent_to_rgb`, `app:rgb_decoder`, `app:rgb_architecture`,
`fig:latent_to_rgb_pipeline`, `tab:latent_rgb`, the abstract paragraph, and the conclusion sentence.

### 2.2 🧑‍⚖️ The human audit split `Ψ_human` and the intervention-based Causal L1

| Paper element | Status |
|---|---|
| City-disjoint audit split `D_audit` with **human**-annotated agent masks, visibility states, interaction pairs, intervention regions | ❌ does not exist. Nothing in `src/` or `configs/` produces or consumes human annotations. `src/utils/audit_disjoint.py` is a **clip-key disjointness checker**, not an annotation pipeline. |
| `tab:evaluator_provenance` row "Audit factors/interventions — Human annotations" | ❌ fabricated |
| Causal L1 = `mean_a ‖Δ̂_a − Δ*_a‖₁ / ‖Δ*_a‖₁` over interventions `I_a` | ❌ **the shipped metric is completely different.** `src/utils/pt_causal.py`: mask the **future temporal half** (slots `[Tp/2, Tp)`), predict it from the past half, per-clip mean L1. No edit, no `Δ`, no annotation. |
| Headline claim "(ii) stronger intervention-sensitive structure" | ❌ rests on the above |

**Action:** rewrite Causal L1 honestly as *"causal future-block L1 — predict the second temporal half
from the first"* (this is exactly what `configs/metric_names.json` already calls it), drop
"intervention-sensitive" from abstract/conclusion/takeaway, and delete `sec:evaluator_independence`'s
`Ψ_human` equation + the audit row of `tab:evaluator_provenance`.

### 2.3 🧱 The FactorJEPA architecture itself

This is the largest gap. `2_factor_jepa.tex` L202–L1102 and `app:factor_architecture` describe an
architecture that **is not what was trained.**

| Paper component | In code? |
|---|---|
| Coordinate blocks `c_{n,L}, c_{n,A}, c_{n,I}` | ❌ |
| Soft visibility gate `v_n^(i) = σ(g_V(·))` + gated agent pooling | ❌ |
| Sparse interaction coords `w_n^(ij)·ē_n^(ij)` over pair set `E_n` | ❌ |
| Synthesis dictionaries `A_L, A_A, A_I`; factorization `Ŷ = C Aᵀ` | ❌ |
| Factor heads `P_k`, loss `L_factor` | ❌ |
| Block-diagonal covariance penalty `L_cov` | ❌ |
| Nonlinear kernel-alignment penalty `L_nlin`, `L_sep` | ❌ |
| Sparsity penalty `L_sparse` | ❌ |
| Visibility BCE `L_v` | ❌ |
| Final objective `L = L_JEPA + λ_sep L_sep + λ_sparse L_sparse + λ_v L_v + λ_sup L_factor` | ❌ |

**What was actually trained** (`src/m09c1_surgery_encoder.py` → `utils/training.compute_total_loss`):

```
L  =  L_JEPA  +  β · L_multitask(probe head)  +  γ · L_motion_aux(13-D motion MSE + K-class CE)  +  λ · L_drift(θ − θ_init anchor)
```

…delivered as a **3-stage progressive-unfreezing curriculum over factor-filtered *datasets***
`D_L → D_A → D_I` produced by `m10_sam_segment.py` (Grounding DINO + SAM 3) and
`m11_factor_datasets.py`. The predictor is the stock V-JEPA monolithic predictor throughout.

**Action:** this is a *data-curriculum* method, not an architectural factorization. Either rewrite
§`sec:factorjepa` to describe the delivered curriculum, or build the architecture. Presenting
`Ŷ = C Aᵀ` with five loss terms while shipping staged fine-tuning is not defensible under review.

### 2.4 📐 Other undelivered sections

| Paper element | Status |
|---|---|
| §`Cross-Factor Leakage Diagnostic` — probes `r_{k→k'}`, `Leak(k→k')` | ❌ no implementation, no data. (Needs §2.3's blocks.) |
| §`Depth-Wise Factor Realization` — `I_k(ℓ)` per-layer factor probes; `fig:factorjepa_channels` | ❌ no code. The nearest thing is `block_drift` = per-block **parameter** drift ‖θ_ℓ − θ_ℓ^init‖ — a different quantity. `factor_channels_depth.png` is not produced by this repo. |
| §`How Dense is DENSEWORLD?` + `app:five_axis_results` — BDD100K / nuScenes comparison on 5 axes | ❌ **zero** BDD100K or nuScenes code, config or output anywhere in the repo. `fig:denseworld_vs_west_density` is not generated by this pipeline. |
| RGB Agent F1 from "a frozen, independent detector" | ❌ not computed |
| 2B backbone at FULL 116k scale | ❌ never run — `outputs/full/` contains **only** `vjepa_2_1_vitg_1B` |
| `cassle_encoder`, `ewc_encoder` arms | ⚠️ trained, but `n_test = —` at **every** scale (never evaluated) |

---

## 3. 🔢 Numbers that are misaligned / over-committed

| # | Paper says | Delivered | Δ |
|---|---|---|---|
| 1 | **"approximately 1,000 hours"** (abstract, `2_data:23`, `8_conclusion:6`, `11_appendix:325`) | **275.8 hours** (`docs/index.html:1090`, our own live site: "115,687 clips \| 275.8 hours \| 121.2 GB") | 🚨 **3.6× over-claim** |
| 2 | "We partnered with **two professional video-collection companies** to record…" (`2_data:22`, `app:dataset_acquisition`) | **714 YouTube videos downloaded at 480p** (`src/m01_download.py:2` — *"Download all 714 YouTube videos at 480p from YT_videos_raw.json"*; `YT_videos_raw.json` metadata `grand_total: 718`) | 🚨 fabricated provenance |
| 3 | "all layout/agent/visibility/interaction targets produced using a fixed **DINOv2**-based pipeline"; §`DINOv2-Based Factor-Target Construction`; `F_DINOv2`; cites `oquab2024dinov2` | **Grounding DINO** (`IDEA-Research/grounding-dino-base`, Swin-B 233M) + **SAM 3** (`facebook/sam3`) — `src/m10_sam_segment.py`, `configs/train/surgery_base.yaml:170,183`. DINOv2 exists in this repo **only** as a frozen *baseline encoder* (`m05b`). | 🚨 wrong model family (DINO ≠ DINOv2) |
| 4 | "**80/10/10** train/val/test, stratified by city and capture mode" (`app:dataset_splits`) | train **86,831** / val **909** / test **23,106** = **75.1 / 0.8 / 20.0** (`training_summary.json`, `eval_metrics.json n_test`) | ⚠️ wrong ratios |
| 5 | "Frames 1:12 form context; frames 13:16 form the future target; horizon 1.0 s" (app. Panel A) | V-JEPA **random spacetime multiblock** mask: 8 blocks @ spatial 0.15 + 2 @ 0.70, `temporal_scale [1.0,1.0]` (`configs/train/base_optimization.yaml:72-80`). `m12d_future_mse.py` samples `(m_enc, m_pred)` from V-JEPA's `MaskGenerator`. | 🚨 the stated context/target split was never used |
| 6 | "**Future-frame L1** — L1 between predicted and target **future** embeddings" | random-spacetime **masked-token** L1 over the whole clip — not a future-prediction metric. (`configs/metric_names.json` calls it "future-frame MSE"; the file writes `future_frame_l1_loss`.) | ⚠️ name over-claims |
| 7 | "**Motion cosine** — cosine alignment between **linearly decoded** and target motion descriptors" | intra-class minus inter-class cosine **separation** on pooled frozen features: `score(q) = mean cos(q, same-class) − mean cos(q, other-class)` (`m12b_motion_cos.py`) | ⚠️ wrong definition |
| 8 | "Across **2B and 1B** backbones" (abstract, conclusion, `sec:experiments`) | true at **10k POC only**. At 116k FULL: **1B only, 3 arms**. | ⚠️ scope |
| 9 | `\ref{app:adaptation_details}`, `\ref{app:annotation_protocol}` ×2, `\ref{app:latent_rgb_details}`, `\ref{app:taxonomy_mapping}` | **0 of 4 labels are defined** → compiles to `??` | 🔧 broken refs |

### 3.1 🚨 The attribution problem — `Δ_sup` is negative everywhere

`app:attribution_primary` defines `Δ_sup|pkg = FactorJEPA − FactorJEPA-RAW` — *"isolates the
contribution of structured factor-target supervision."* Computed from our own data
(direction-normalized, **+ve = better**):

```text
             AutoRGN → FactorJEPA-RAW (surgery_raw) → FactorJEPA (surgical_3stage_DI)
┌──────────────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ metric                       │  d_pkg   │  d_sup   │  d_total │  d_pkg   │  d_sup   │
│                              │   2B     │   2B     │   2B     │   1B     │   1B     │
├──────────────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ future-frame L1              │ +0.02890 │ -0.00094 │ +0.02796 │ +0.02410 │ -0.00420 │
│ causal future-block L1       │ +0.02286 │ -0.00217 │ +0.02068 │ +0.02266 │ -0.00481 │
│ motion-cosine separation     │ -0.05523 │ -0.01468 │ -0.06990 │ -0.02041 │ -0.01238 │
│ mask-ratio robustness slope  │ +0.00472 │ -0.00037 │ +0.00435 │ +0.00968 │ -0.00123 │
│ Action top-1 accuracy        │ -0.01589 │ -0.01534 │ -0.03123 │ -0.01699 │ -0.00329 │
└──────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

**`d_sup` is negative on 5/5 metrics × 2/2 scales.** The factor-target curriculum
(`D_L → D_A → D_I`) is measurably *not* helping over the identical surgery run on raw clips. The
paper's central attribution claim is contradicted by its own evaluation.

### 3.2 🚨 The 8 headline numbers come from 4 different arms — none of them "FactorJEPA"

`plot_forest` takes a **max over every OURS-family arm** per metric. Which arm actually won:

```text
┌────────┬──────────────────────────┬───────────────────────────────────────────┬─────────────────┐
│ scale  │ metric                   │ arm that produced the headline number      │ is it FactorJEPA?│
├────────┼──────────────────────────┼───────────────────────────────────────────┼─────────────────┤
│ 2B     │ Future-frame L1  47.0x   │ surgical_3stage_DI_intervene_encoder       │ no — variant     │
│ 2B     │ Causal L1        28.7x   │ surgery_raw_encoder                        │ NO — this is RAW │
│ 2B     │ Motion cosine    26.8x   │ surgical_noDI_head  (encoder FROZEN,       │ NO — head-only,  │
│        │                          │   only a ~432K motion_aux head is trained) │   no D_I channel │
│ 2B     │ Mask-ratio       22.3x   │ surgical_3stage_DI_diheavy_encoder         │ no — variant     │
│ 1B     │ Future-frame L1  27.2x   │ surgery_raw_encoder                        │ NO — this is RAW │
│ 1B     │ Causal L1        19.0x   │ surgery_raw_encoder                        │ NO — this is RAW │
│ 1B     │ Motion cosine    25.6x   │ surgery_raw_encoder                        │ NO — this is RAW │
│ 1B     │ Mask-ratio       15.7x   │ surgical_3stage_DI_diheavy_encoder         │ no — variant     │
└────────┴──────────────────────────┴───────────────────────────────────────────┴─────────────────┘
```

Plain `surgical_3stage_DI_encoder` — the arm the paper *calls* FactorJEPA — wins **0 of 8**.
Four of eight are won by **FactorJEPA-RAW**, i.e. the ablation that has *no factor supervision at
all*, and one by a **head-only** arm whose encoder is bit-identical to Meta's release.

**Action:** either (a) redefine FactorJEPA as the *family* and say "best-of-family" explicitly with
the winning arm named per row (the forest figure already prints this in its right-hand margin), or
(b) fix a single arm and report its numbers. Option (a) is honest and cheap; the figure supports it.

---

## 4. 🧪 Ablation inventory (what we actually own)

### 4.1 1B vs 2B — ✅ delivered at 10k, ❌ absent at 116k

```text
POC 10k · n_test = 1825 · V-JEPA 2.1
┌──────────────────────────────┬───────────┬───────────┬──────────────────────────┐
│ metric                       │ 2B frozen │ 1B frozen │ best-OURS  2B  /  1B     │
├──────────────────────────────┼───────────┼───────────┼──────────────────────────┤
│ future-frame L1        (lo)  │  0.55735  │  0.63686  │  0.49552  /  0.58718     │
│ causal future-block L1 (lo)  │  0.58306  │  0.65515  │  0.52805  /  0.61057     │
│ motion-cosine sep.     (hi)  │  0.00910  │  0.00700  │  0.13461  /  0.05969     │
│ mask-ratio slope       (lo)  │  0.07195  │  0.06511  │  0.05309  /  0.05152     │
│ Action top-1           (hi)  │  0.44274  │  0.42466  │  0.50685  /  0.48000     │
└──────────────────────────────┴───────────┴───────────┴──────────────────────────┘
arms evaluated:  2B = 20 · 1B = 14   (the 6 missing at 1B: noDI_encoder, noDI_head,
3stage_DI_head, replay25, tccaux, wiseft)  →  the Spearman ρ is over the 14 common arms.
```

✅ **This is a real, publishable ablation.** ⚠️ But "across 2B and 1B" must be scoped to 10k.

### 4.2 10k POC vs 116k FULL — ⚠️ delivered for 1B / 3 arms only, and it is *confounded*

```text
1B ViT-g · the ONLY 3 arms present at both scales
┌──────────────────────────────┬──────────────────────┬───────────┬───────────┬──────────┐
│ metric                       │ arm                  │ POC 10k   │ FULL 116k │  delta   │
│                              │                      │ (n=1825)  │ (n=23106) │          │
├──────────────────────────────┼──────────────────────┼───────────┼───────────┼──────────┤
│ Action top-1          (hi)   │ frozen               │  0.42466  │  0.48810  │ +0.06344 │
│                              │ peft_lora            │  0.46685  │  0.57245  │ +0.10560 │
│                              │ OURS (diheavy)       │  0.48274  │  0.56739  │ +0.08465 │
├──────────────────────────────┼──────────────────────┼───────────┼───────────┼──────────┤
│ future-frame L1       (lo)   │ frozen               │  0.63686  │  0.64068  │ +0.00382 │
│                              │ peft_lora            │  0.59576  │  0.58924  │ -0.00652 │
│                              │ OURS (diheavy)       │  0.58986  │  0.57281  │ -0.01706 │
├──────────────────────────────┼──────────────────────┼───────────┼───────────┼──────────┤
│ causal future-block L1 (lo)  │ frozen               │  0.65515  │  0.65549  │ +0.00034 │
│                              │ peft_lora            │  0.61713  │  0.60698  │ -0.01015 │
│                              │ OURS (diheavy)       │  0.61385  │  0.59742  │ -0.01644 │
├──────────────────────────────┼──────────────────────┼───────────┼───────────┼──────────┤
│ motion-cosine sep.     (hi)  │ frozen               │  0.00700  │  0.00634  │ -0.00066 │
│                              │ peft_lora            │  0.06773  │  0.07767  │ +0.00995 │
│                              │ OURS (diheavy)       │  0.04811  │  0.09724  │ +0.04913 │
├──────────────────────────────┼──────────────────────┼───────────┼───────────┼──────────┤
│ mask-ratio slope       (lo)  │ frozen               │  0.06511  │  0.06540  │ +0.00029 │
│                              │ peft_lora            │  0.05419  │  0.05777  │ +0.00357 │
│                              │ OURS (diheavy)       │  0.05152  │  0.04750  │ -0.00402 │
└──────────────────────────────┴──────────────────────┴───────────┴───────────┴──────────┘
```

Two things this table says that the paper does **not**:

1. 🎉 **The "prediction–motion trade-off" evaporates at full scale.** At 116k, OURS beats LoRA on
   motion-cosine by **+25.2%** (0.0972 vs 0.0777, 20.0× CI) — at 10k it *lost* motion by 33–45%.
   The paper's §`Prediction–Motion Trade-off` and the conclusion's "consistent trade-off … replicates
   across scale" are **contradicted by the full-scale run**.
2. ⚠️ **LoRA overtakes OURS on Action top-1 at full scale** (57.25% vs 56.74%) — the POC ordering
   (48.27 vs 46.69) flips.

⚠️ **Confound to disclose:** POC and FULL do **not** share a test set (1,825 vs 23,106 clips), so
`delta` mixes training-set size with evaluation-corpus change. Frozen-arm deltas (e.g. Action top-1
+6.3 pp with *no training at all*) quantify how much is pure eval-set shift.

### 4.3 Frozen cross-family eval — ✅ **fully delivered, 100% absent from the paper**

`iter/iter17_ablations_model/result_outputs/v17a_frozen_eval/poc/` — 11 frozen encoder families,
same probe head, same 1,825 test clips:

```text
┌────────────────────────────────┬───────────┬────────┬──────────────┬──────────┐
│ frozen encoder                 │ Action    │  ±95%  │ motion-cos   │  ±95%    │
│                                │ top-1 %   │        │ separation   │          │
├────────────────────────────────┼───────────┼────────┼──────────────┼──────────┤
│ vjepa_2_1_pretrain_2X_encoder  │   53.15   │  2.27  │   0.168789   │ 0.004163 │
│ vjepa_2_1_surgical_noDI_enc    │   51.78   │  2.27  │   0.168132   │ 0.004549 │
│ vjepa_2_1_surgical_noDI_head   │   51.73   │  2.30  │   0.179114   │ 0.004775 │
│ vjepa_2_1_pretrain_encoder     │   51.67   │  2.27  │   0.159025   │ 0.004382 │
│ vjepa_2_1_surgical_3stage_head │   51.23   │  2.30  │   0.167397   │ 0.004530 │
│ vjepa_2_1_surgical_3stage_enc  │   50.30   │  2.27  │   0.163396   │ 0.004441 │
├────────────────────────────────┼───────────┼────────┼──────────────┼──────────┤
│ vjepa_2_1_frozen      (ViT-G)  │   44.38   │  2.27  │   0.009102   │ 0.000682 │
│ vjepa_2_1_vitL_frozen          │   44.22   │  2.30  │   0.003909   │ 0.000321 │
│ vjepa_2_1_pretrain_head        │   42.85   │  2.28  │   0.012688   │ 0.000779 │
│ vjepa_1_vitH_frozen            │   40.49   │  2.22  │   0.006783   │ 0.000583 │
│ lejepa_vitL_frozen             │   40.05   │  2.22  │   0.014352   │ 0.001507 │
│ vjepa_1_vitL_frozen            │   39.89   │  2.27  │   0.008279   │ 0.000642 │
│ ijepa_vitH14                   │   39.12   │  2.25  │   0.015554   │ 0.001421 │
│ vjepa_2_0_vitg_ssv2            │   38.85   │  2.27  │   0.007217   │ 0.000623 │
│ dinov2 (w/ registers, giant)   │   38.47   │  2.25  │   0.015817   │ 0.001459 │
│ vjepa_2_vitL_256_frozen        │   37.92   │  2.25  │   0.013029   │ 0.001142 │
│ ijepa_vitG16                   │   37.48   │  2.22  │   0.019025   │ 0.001492 │
└────────────────────────────────┴───────────┴────────┴──────────────┴──────────┘
```

🎯 **This is the strongest unused asset in the repo.** It is exactly the "existing JEPA formulations
struggle on DENSEWORLD" evidence the abstract asserts but never shows: **every** off-the-shelf
frozen encoder — image-JEPA, video-JEPA v1/v2.0/v2.1, DINOv2, LeJEPA, and even the SSv2
*supervised* fine-tune — lands in a 37–44% band, while any DENSEWORLD adaptation reaches 50–53%.

⚠️ Predictor-side metrics (`future_mse`, `causal`, `maskratio`) are V-JEPA-only in this sweep —
image encoders have no video predictor. State that explicitly.

---

## 5. 🧹 Stale content in `0_main.tex`

`0_main.tex` (unlike `0_main_AAAI.tex` / `0_main_neurips.tex`) still `\input`s two obsolete drafts:

| File | Problem |
|---|---|
| `10_india.tex` (1,065 ln) | A **different, older paper**: "IndianSuburb-**5K**", "SAM3-Only Annotation Plan", a Shampoo/K-FAC-Lite optimizer study, results R1–R4. Contradicts DENSEWORLD/116k throughout. |
| `12_new_plan.tex` (416 ln) | Draft artifacts: a literal `\textbf{:contentReference[oaicite:0]{index=0}}` where a citation should be, plus dangling *"produced by ."* and *"sourced from, e.g., the collection"*. |

Both are already commented out in the AAAI/NeurIPS mains — drop them from `0_main.tex` too.

---

## 6. 📣 What to market instead — the case built only on delivered evidence

### 6.1 DENSEWORLD 1.0 (dial the numbers to the truth)

> 115,687 clips · 275.8 hours · 121.2 GB · 22 Indian cities · 714 long-form source videos ·
> drive-through / walk-through / aerial · shot-aware segmentation (PySceneDetect `AdaptiveDetector`)
> · **86,831 clips carry automatic layout / agent / interaction targets** from a
> Grounding-DINO + SAM-3 pipeline · 23,106-clip held-out test split.

276 h across 22 cities with 87k factor-annotated clips is a strong dataset contribution **on its own
merits**. "1,000 hours" + "professional collection companies" is 3.6× inflation attached to a
provenance claim that the repo contradicts — it converts a real asset into a liability the moment a
reviewer opens the anonymized code link that the abstract itself provides.

### 6.2 Lead with the FULL 116k run, not the 10k POC

The 116k/1B run is the most defensible table we own — 12.7× the test clips, and every effect is
*larger*:

```text
FULL 116k · 1B ViT-g · n_test = 23,106
┌──────────────────────────────┬──────────┬──────────┬───────────┬──────────┬──────────┬──────────┐
│ metric                       │  OURS    │  frozen  │  ×CI vs   │  LoRA    │ ×CI vs   │ % vs     │
│                              │(diheavy) │          │  frozen   │          │  LoRA    │ LoRA     │
├──────────────────────────────┼──────────┼──────────┼───────────┼──────────┼──────────┼──────────┤
│ future-frame L1        (lo)  │ 0.572806 │ 0.640679 │   139.1x  │ 0.589242 │   33.2x  │  +2.79%  │
│ causal future-block L1 (lo)  │ 0.597417 │ 0.655490 │    87.8x  │ 0.606977 │   13.9x  │  +1.58%  │
│ motion-cosine sep.     (hi)  │ 0.097240 │ 0.006344 │   115.3x  │ 0.077673 │   20.0x  │ +25.19%  │
│ mask-ratio slope       (lo)  │ 0.047498 │ 0.065399 │    72.1x  │ 0.057766 │   43.3x  │ +17.78%  │
└──────────────────────────────┴──────────┴──────────┴───────────┴──────────┴──────────┴──────────┘
```

⚠️ **Disclose:** the FULL competitor field is LoRA only — `full_ft`, `lpft`, `dora`, `autorgn`,
`surgery_raw` were never evaluated at 116k. So "best competitor" is weaker here than at 10k, where
`full_ft` is the motion champion. Frame it as *"OURS vs frozen vs LoRA at full scale"*, not
*"OURS vs all baselines"*.

### 6.3 The three-legged story to keep

| Leg | Evidence | Where |
|---|---|---|
| 🌍 **DENSEWORLD is a distinct regime** | 11 frozen encoder families all collapse into 37–44% Action top-1; adaptation reaches 50–53% | §4.3 (new to the paper) |
| 🔧 **Predictor surgery on factor-filtered data beats generic adaptation** | 116k table above; 10k forest at both scales | §6.2 + `forest_plot_frozen_ci` |
| 📈 **The 1B backbone is a faithful screening proxy for 2B** | ρ = 0.895–0.978 on the 4 headline diagnostics over 14 common arms | §4.1 + `scale_replication` |

### 6.4 Claims to drop from abstract + conclusion

1. ❌ "1,000 hours" → **"276 hours"**
2. ❌ "Cosmos-initialized latent-to-RGB decoder"
3. ❌ "intervention-sensitive prediction (Causal L1)" → "causal future-block prediction"
4. ❌ "visibility gate and separated subspaces" (no such module)
5. ❌ "a reproducible motion-information trade-off" — it **reverses** at 116k
6. ⚠️ "across 2B and 1B backbones" → "at 2B and 1B on the 10k screening corpus, plus a full-scale
   116k run at 1B"
