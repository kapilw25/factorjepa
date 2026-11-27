# 📬 AAAI-27 Student Abstract: Reviews (Submission 30)

> **Venue:** AAAI-27 Student Abstract and Poster Program (EasyChair) · **Paper:** [v3_AAAI27_student_abstract/0_main_AAAI.pdf](../v3_AAAI27_student_abstract/0_main_AAAI.pdf) (2 pages) · **Supplement:** [1_supplement_full_paper.pdf](../v3_AAAI27_student_abstract/1_supplement_full_paper.pdf) (42 pages) · **Earlier main-track reviews:** [v1_AAAI27_main.md](v1_AAAI27_main.md)

> [!NOTE]
> The reviewers' text below is kept in full and unedited; only the layout changed. Claude Code's verification notes are in the last section, [🔎 Fact-check](#-fact-check-by-claude-code-not-the-reviewers).

**Legend:** 🟢 good · 🟡 fair or borderline · 🔴 weak or missing · ✅ confirmed · ⚠️ confirmed and worse than stated · 📍 page reference

## 📌 TL;DR

| | Summary |
|---|---|
| 🗓️ **Released** | Wed Nov 18, 2026, 2:27 PM, by the Student Abstract and Poster Program co-chairs (Tongliang Liu, Sirisha Rambhatla) |
| 🧾 **Submission** | #30, "FactorJEPA on DENSEWORLD-115k: Can a World Model Predict Crowded, Chaotic South Asian Streets?" |
| ⚖️ **Decision** | Not stated in this e-mail (it only says the reviews were released) |
| 🎯 **Overall scores** | 🟢 Review 1: **1 (weak accept)** · 🟡 Review 2: **0 (borderline paper)** |
| 🔑 **Core problem** | 🔴 The abstract and its supplement define the metrics differently (L1 vs MSE, two meanings of Causal L1), so Review 1 could not tell which definition the results use. |
| 💪 **What landed** | The dataset and setting (dense, occluded, heterogeneous traffic) and the clear layout / agent / interaction design; Review 2 calls it "a novel idea and a dataset with convincing preliminary results". |
| 🥇 **Most important fix** | One set of metric definitions in every version (abstract, supplement, arXiv, workshop paper), matching the code; drop the city-disjoint audit and DINOv2 claims. |

## 📊 Reviewer scorecard

| Criterion | 🧑‍⚖️ Review 1 | 🧑‍⚖️ Review 2 |
|---|---|---|
| 📚 Literature | 🟢 4 (good) | 🟢 4 (good) |
| 🔍 Clarity | 🟢 4 (good) | 🟡 3 (fair) |
| 🧪 Soundness | 🟡 3 (fair) | 🟢 4 (good) |
| 📈 Evaluation and Results | 🟡 3 (fair) | 🟡 3 (fair) |
| ⚖️ **Overall evaluation** | 🟢 **1 (weak accept)** | 🟡 **0 (borderline paper)** |

## 🧑‍⚖️ Review 1 · Overall 1 (weak accept)

### 📝 Summary

This paper introduces DENSEWORLD-115k, a dataset of crowded street scenes with heterogeneous agents and frequent occlusion, and proposes FactorJEPA, which decomposes future-feature prediction into layout, agent, and interaction channels. The problem is meaningful, and the method is well motivated by the characteristics of these scenes. The experiments identify both predictive gains and limitations in temporal consistency. However, inconsistencies in the evaluation definitions make some results difficult to interpret, and the comparisons provide limited support for conclusions about the effects of data scale and individual components.

### 💪 Strengths

| # | Strength | Reviewer's text |
|---|---|---|
| 🟢 S1 | The dataset addresses a meaningful evaluation setting. | Dense traffic, persistent occlusion, diverse agents, and poorly defined spatial boundaries present relevant challenges for predictive world models. DENSEWORLD-115k covers multiple cities, viewpoints, and scene types. Comparisons with existing datasets using agent density and occupancy provide quantitative support for the motivation. |
| 🟢 S2 | The method has a clear design that reflects the challenges of the target setting. | The layout, agent, and interaction channels, together with visibility gating, sparse interactions, and factor supervision, form a coherent approach to representing spatial structure, observation reliability, and relationships between agents. The method retains the basic V-JEPA framework and, according to the main paper, does not require an external DINOv2 or tracking pipeline at inference. |
| 🟢 S3 | The evaluation examines several capabilities and openly reports limitations. | The experiments assess future-feature prediction, robustness to masking, motion readout, and temporal consistency. Comparisons across several adaptation methods and two backbone sizes provide useful context in the smaller-data setting. The reported weaknesses on some motion and temporal metrics also help clarify where the method improves performance and where it remains limited. |

### ⚠️ Weaknesses

#### 🔴 W1. Inconsistent metric definitions and evaluation descriptions make the main results difficult to interpret.

The main paper describes future prediction error as L1, whereas pages 35–36 of the supplementary material explicitly define MSE and state that the scorecard reports this quantity. The main paper defines Causal L1 as the error when predicting the second half of a clip from the first, while the supplement defines it as the discrepancy between predicted and target feature changes under controlled input edits. These are different evaluation procedures, and it is unclear which definitions underlie the main results. The city-disjoint human-annotated audit set and the source-video-disjoint test set could coexist, but their respective roles in the reported evaluations are also insufficiently specified.

#### 🔴 W2. The comparisons do not fully support attributing the prediction–motion trade-off to data scale.

In Table 1, full fine-tuning is the strongest comparator on Motion cosine in the smaller-data setting, whereas the 115k setting includes only LoRA and the frozen backbone. Outperforming these methods at the larger scale therefore does not establish that the disadvantage relative to full fine-tuning has disappeared. The results support an advantage over the evaluated baselines at 115k, but the stronger conclusion that the trade-off is a data effect rather than a property of the factorization is not established by the current comparisons.

#### 🟡 W3. The ablations provide partial attribution but do not fully separate architecture, regularization, and supervision.

Comparing FactorJEPA-RAW with the complete model helps identify the incremental contribution of factor and visibility supervision within the same architecture. However, RAW differs from the monolithic predictor in several respects, including its factorized structure, prediction pathways, and regularization. The supplement also acknowledges that a monolithic predictor with matched auxiliary supervision was not evaluated. The experiments therefore support the effectiveness of the combined approach more clearly than the independent benefit of factorization or the necessity of individual components.

#### 🟡 W4. The visual decoding evaluation is incomplete, limiting the evidence for benefits beyond latent prediction.

The main results support improvements in latent-feature prediction, while several rollout and temporal-consistency metrics remain worse than the baselines. The supplement describes an evaluation using a shared decoder and a target-latent oracle, but the RGB results in Table 10 on page 40 remain placeholders. The current manuscript therefore does not provide complete evidence that the latent prediction gains translate into better visual reconstruction or preservation of agents and motion.

## 🧑‍⚖️ Review 2 · Overall 0 (borderline paper)

> In this paper the major focus is on the adapted model, FactorJEPA from V-JEPA. The proposed approach is able to achieve competetive result. In most of the cases, the rival (as mentioned in the paper) is outperforming the pruposed model. The authors could have highlighted the reason behind such outcome with more details. Overall, this is a good work as student abstract. The work presents a novel idea and a dataset with convincing preliminary results.

| Point | Type |
|---|---|
| Competitive results from the adapted V-JEPA model | 🟢 strength |
| "A novel idea and a dataset with convincing preliminary results" | 🟢 strength |
| The rival outperforms FactorJEPA in most cases | 🔴 weakness |
| The reason the rival wins is not explained in enough detail | 🔴 weakness |

## 🔎 Fact-check (by Claude Code, not the reviewers)

Checked on 2026-11-26 against the submitted files in [v3_AAAI27_student_abstract/](../v3_AAAI27_student_abstract/) and the code in `src/`.

### ✅ Confirmed

| Claim | Verdict | Evidence |
|---|---|---|
| R1-W1: abstract says L1, supplement says MSE | ✅ | Supplement 📍 pp. 35–36 defines "Future-frame MSE" and says the scorecard reports MSE, "not relabeled as L1". The code computes mean absolute error ([L1 code](../../src/m12d_future_mse.py)), so the **supplement text is the wrong one**. |
| R1-W1: two definitions of Causal L1 | ✅ | Abstract: "error when the second half of a clip is predicted from its first half". Supplement: renamed "Intervention L1", a controlled edit per annotated intervention region. The code predicts the second half from the first ([causal code](../../src/utils/pt_causal.py)), so again the **supplement is wrong**. |
| R1-W2: at 115k only LoRA and the frozen backbone were trained | ✅ | Table 1 caption of the abstract: "At 115k only LoRA and the frozen backbone were trained as rivals." Full fine-tuning, the Motion-cosine leader at 10k, has no 115k run. |
| R1-W3: RAW keeps the factorized predictor | ✅ | Supplement: FactorJEPA-RAW "retains the factorized predictor but removes factor-target supervision". The sentence about a matched monolithic predictor was not found by text search. |
| R1-W4: Table 10 is a placeholder | ✅ | Supplement 📍 p. 40: every PSNR, SSIM, LPIPS, Flow EPE and Agent F1 cell reads `[result]`. |

Code name for the first row: the L1 is computed by `masked_predict_l1`.

### ⚠️ Issues found (worse than the reviewers say)

| Claim | Verdict | Evidence |
|---|---|---|
| R1-W1: a "city-disjoint human-annotated audit set" exists | ⚠️ | The abstract says "Headline evaluation uses a city-disjoint audit split with independently annotated agents". No such split exists in the code: test clips are held out at the clip level ([split code](../../src/utils/clip_splits.py)), so clips from one source video can sit on both sides. |
| R1-S2: "does not require an external DINOv2 ... pipeline" | ⚠️ | The reviewer repeats our wording. The abstract says the factor targets come from a "frozen DINOv2 pipeline"; the code uses Grounding DINO plus SAM 3 ([segmentation code](../../src/m10_sam_segment.py)). |
| R2: "in most of the cases, the rival is outperforming" | 🟡 | Partly true. At 115k the best rival stays ahead on 6 diagnostics (rollout drift, exposure bias, L1 versus Δt decay, arrow of time, two TCC measures); FactorJEPA separates on the 4 predictive diagnostics. The abstract gives no reason for the split, which is Review 2's real request. |

### 🛠️ What to change, and where

| # | Fix | Student abstract (camera-ready) | Workshop paper [v4](../v4_AAAI_workshop/v1_Human_Centric/0_main.tex) | Supplement and arXiv |
|---|---|---|---|---|
| 1 | Future-frame error is L1 everywhere | ✅ already L1 | ✅ already L1 | 🔴 change MSE to L1 (📍 pp. 35–36) |
| 2 | Causal L1 = second half predicted from the first | ✅ already | ✅ already | 🔴 replace the intervention definition |
| 3 | Remove the city-disjoint audit claim; state the clip-level split | 🔴 still claims it | ✅ states the clip-level split | 🔴 check and fix |
| 4 | Factor targets = Grounding DINO + SAM 3, not DINOv2 | 🔴 still says DINOv2 | ✅ already | 🔴 check and fix |
| 5 | Drop "the trade-off is a data effect" (no full fine-tuning at 115k) | 🔴 still claims it | ✅ fixed 2026-11-26 in the v1 Human-Centric and v2 Cog2Soc papers | 🔴 check and fix |
| 6 | Say why the rival wins on the six temporal diagnostics | 🔴 missing | 🟡 one sentence ("does not yet keep long predicted futures more coherent") | 🟡 expand |
| 7 | Fill or remove the RGB decoding Table 10 | n/a | ✅ Limitations: embedding-space metrics only | 🔴 remove the placeholder table |
| 8 | State that factorization and factor supervision are not separated | 🔴 missing | ✅ in Limitations | 🟡 already acknowledged |
