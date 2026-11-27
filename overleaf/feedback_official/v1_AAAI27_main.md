# 📬 AAAI-27 Main Track: Decision and Reviews (Submission 43837)

> Source: OpenReview submission page, exported 2026-09-25. Every review sentence below is verbatim; only the OpenReview page chrome was removed and headings/tables added. Inline math was stripped by the export and is shown as `[math]`.

## 🗂️ Submission Facts

<!-- joined-words: off --> <!-- why: the Title row quotes the submitted title verbatim -->
| Field | Value |
|---|---|
| **Title** | FactorJEPA: Factorizing Monolithic Futures into Layout–Agent–Interaction Channels for Crowded and Chaotic Global South Urban Worlds |
| **Authors** | Kapil Wanaskar, Gaytri Jena, Aman Chadha, Vinija Jain, Vasu Sharma, Amitava Das |
| **Submitted** | 21 Jul 2026 (modified 20 Aug 2026), CC BY 4.0 |
| **Primary topic** | ML: World Models, Simulation & Environment Models |
| **Secondary topic** | CV: Datasets & Benchmarks for Vision |
| **Countries** | United States, India |
| **Supplements** | Reproducibility checklist (pdf), technical supplement (pdf), media supplement (zip), code and data supplement (zip) |
| **Reciprocal reviewer** | Aman Chadha |
| **TL;DR (as submitted)** | We introduce DENSEWORLD, a 1,000-hour benchmark from 22 cities, and FactorJEPA, which composes layout, agent, and interaction into visibility-aware, separated predictive channels to improve prediction in dense mixed traffic. |

<details>
<summary>📄 Abstract as submitted</summary>

World models have attracted significant attention for their ability to capture and predict the structure and dynamics of the physical world. Within this landscape, Joint Embedding Predictive Architectures (JEPA) offer a particularly compelling direction.

In this paper, we study a largely unexplored regime: populous, crowded, and chaotic Global South urban environments, which we call DENSEWORLD. Unlike the lower-density, lane-structured settings that dominate existing evaluations, these scenes exhibit soft spatial boundaries, extreme agent heterogeneity, persistent occlusion, and rapid social negotiation under mixed traffic. We introduce the first large-scale dataset for this regime, comprising 1,000 hours of drive-through, walk-through, and aerial video collected across 22 cities. The dataset spans diverse scene types, times of day, weather conditions, crowd and traffic densities, traffic mixtures, pedestrian–vehicle separation patterns, road layouts and surfaces, infrastructure quality, encroachment levels, locally distinctive objects, vegetation, lighting conditions, and video quality. Our evaluation reveals that existing JEPA formulations struggle to preserve dense interaction dynamics under heterogeneity and partial observability.

We introduce FactorJEPA, which makes world structure a first-class predictive primitive. Rather than encoding the future in a monolithic latent, it composes layout, entities, and interactions through visibility-aware, separated predictive channels that preserve partially observed agents and discourage cross-factor shortcuts. FactorJEPA improves (i) future-latent accuracy (Future-frame L1), (ii) intervention-sensitive prediction (Causal L1), and (iii) robustness to reduced visual evidence (Mask-ratio slope), while exposing (iv) a reproducible motion-information trade-off (Motion cosine). Method rankings replicate across 2B and 1B V-JEPA 2.1 backbones, with ρ = 0.895–0.978 across these four diagnostics.

</details>

## ⚖️ Decision: ❌ Reject (Phase 1)

**Program Chairs, 24 Sept 2026:**

> This paper did not advance to Phase 2 review for the AAAI-27 conference. The paper was reviewed by at least two human reviewers and at least one senior member of the program committee in Phase 1. In addition, papers with at least one positive review were examined by one program chair or associate program chair to ensure the rejection was justified. This decision is final. The provided reviews contain more detail, which we hope will be beneficial to the progress of this research.

## 📊 Reviewer Scorecard

| Reviewer | Date | Rating | Confidence | One-line verdict |
|---|---|---|---|---|
| 🧑‍⚖️ R1 (PC Csju) | 30 Aug 2026 | **5** marginally below threshold | 3 fairly confident | Promising benchmark and structure; inference-time pipeline, leakage evidence and external evaluation missing |
| 🧑‍⚖️ R2 (PC E856) | 29 Aug 2026 | **2** strong rejection | 4 confident | Raw-surgery ablation matches factorized variants; Causal L1 and dataset descriptions contradict the supplement; unfinished manuscript |
| 🤖 AI reviewer | 27 Jul 2026 | n/a | n/a | "Ours" is not one fixed model; sparsity and visibility-gate math do not do what the text claims; novelty and information boundary under-specified |

## 🧑‍⚖️ Review 1 (Program Committee Csju)

**Headline:** Promising structured JEPA formulation and dense-world benchmark, but key methodological and empirical evidence is incomplete

### 📝 Paper Summary

This paper studies predictive world modeling in crowded and heterogeneous urban environments, a regime the authors refer to as DENSEWORLD. The paper introduces DENSEWORLD 1.0, a dataset containing approximately 1,000 hours of drive-through, walk-through, and aerial video collected across 22 cities, with an emphasis on dense mixed traffic, heterogeneous agents, persistent occlusion, and weakly structured spatial boundaries. The methodological contribution is FactorJEPA, which replaces a monolithic JEPA predictor with explicitly factorized layout, agent, and interaction channels. The model additionally employs a soft visibility gate for agents, sparse interaction modeling, factor-specific supervision derived from a DINOv2-based preprocessing pipeline, and linear/nonlinear channel-separation regularization. The resulting factor coordinates are combined through learned synthesis dictionaries to reconstruct the future JEPA target representation. The authors evaluate FactorJEPA using V-JEPA 2.1 backbones at approximately 2B and 1B scales. The primary diagnostics measure future-latent prediction error, sensitivity to controlled interventions, robustness to increasing masking, and linearly accessible motion information. The paper reports improvements over several full and parameter-efficient adaptation baselines on several predictive diagnostics, with method rankings that are relatively stable across the two backbone scales.

### 💪 Strengths

1. **Important and underexplored problem setting.** The paper studies world modeling in crowded, heterogeneous, and weakly structured urban environments, where persistent occlusion, mixed traffic, and dense agent interactions pose challenges that are less emphasized in standard driving benchmarks. The proposed DENSEWORLD benchmark therefore targets a meaningful and relatively underexplored regime.
2. **Clear structural motivation for FactorJEPA.** The idea of decomposing future prediction into layout, agent, and interaction channels is intuitive and well motivated. The visibility gate and channel-separation objectives provide explicit inductive biases rather than relying on a fully entangled predictive latent.
3. **Thoughtful experimental design in several respects.** The paper evaluates the method across two V-JEPA backbone scales and multiple data regimes, compares against several adaptation strategies, and uses a city-disjoint audit set with independent annotations to reduce evaluator leakage from the DINOv2-derived supervision.

### ⚠️ Weaknesses

1. **The inference-time construction of the agent and interaction inputs is unclear.** The agent and interaction channels depend on per-agent representations `[math]` and candidate interaction pairs `[math]`, but the manuscript does not clearly explain how these quantities are obtained at inference time. This is particularly important because the DINOv2 pipeline is described as providing training supervision, while the relationship between that pipeline and the inference graph remains ambiguous.
2. **The central claim of factor separation is not directly validated.** The paper explicitly defines a cross-channel leakage diagnostic and argues that successful factorization should exhibit strong within-factor predictability and low off-diagonal leakage. However, I could not find the corresponding quantitative leakage results in the reported experiments. Since semantic separation of layout, agent, and interaction channels is the main methodological claim, this evidence appears important.
3. **The evidence for improved world modeling relies heavily on metrics tied to the JEPA latent space.** The main diagnostics primarily evaluate future-latent prediction, intervention responses in the same representation space, masking robustness, and motion readout. These results support improved JEPA prediction, but provide more limited evidence that FactorJEPA yields a generally stronger world model. An external predictive or downstream evaluation would make the broader claim substantially more convincing.

### 🧭 Justification of Recommendation

The paper addresses an important and underexplored world-modeling setting, and I find both the DENSEWORLD benchmark and the structural motivation behind FactorJEPA promising. The explicit decomposition into layout, agent, and interaction channels is intuitive, and the evaluation across multiple backbone scales and data regimes provides useful empirical evidence. My rating is 5 mainly because the current manuscript leaves several central aspects insufficiently established. First, the inference-time construction of the agent representations and interaction pairs is not sufficiently clear, making the complete prediction pipeline difficult to understand. Second, although semantic factor separation is a central methodological claim, the paper does not directly report the quantitative leakage results defined for validating this property. Third, most of the primary evidence is based on metrics closely tied to the JEPA latent prediction space, which makes the broader claim of improved world modeling less conclusive. These concerns currently outweigh the paper's strengths, but they do not appear necessarily fundamental. I would be more positive if the rebuttal can clearly explain the inference-time pipeline, provide direct evidence for channel separation, and clarify or strengthen the evidence for improved predictive modeling beyond the JEPA latent-space diagnostics.

### ❓ Specific Points for Rebuttal

1. How are the per-agent representations `[math]` and candidate interaction pairs `[math]` constructed at inference time? Please clarify whether they are produced internally by FactorJEPA or require the DINOv2 segmentation/tracking pipeline or another external module.
2. Can the authors provide quantitative evidence for the claimed separation between the layout, agent, and interaction channels? In particular, reporting the cross-channel leakage diagnostic defined in the paper would directly support the central factorization claim.
3. Is there any evaluation beyond the V-JEPA latent space that supports the broader claim of improved world modeling? For example, could the authors clarify the Cosmos-based future decoding experiment mentioned in the conclusion or provide another external predictive/downstream evaluation?

| Rating | Confidence |
|---|---|
| **5**: Marginally below acceptance threshold | **3**: The reviewer is fairly confident that the evaluation is correct |

## 🧑‍⚖️ Review 2 (Program Committee E856)

**Headline:** Review of FactorJEPA

### 📝 Paper Summary

This paper introduces FactorJEPA, an adaptation framework for video JEPA models in dense urban traffic environments. The proposed method decomposes the predicted future representation into three manually specified semantic components—layout, agents, and interactions—which are supervised using automatically generated targets. The paper also introduces a dataset referred to as DENSEWORLD and evaluates FactorJEPA against several frozen, full-finetuning, and parameter-efficient adaptation baselines using a collection of predictive, semantic, and temporal diagnostics.

### 💪 Strengths

The paper targets a relevant problem: learning predictive video representations in dense and highly interactive traffic scenes that are insufficiently represented by commonly used autonomous-driving datasets. The authors also make an effort to evaluate several aspects of the learned representations rather than relying on a single downstream metric. The submitted supplementary package exposes a substantial amount of experimental code and intermediate analysis, which is useful for examining the implementation.

### ⚠️ Weaknesses

Despite the messy and unusual arrangement of the paragraph, I have the following concerns:

1. **Factorization not shown to be responsible for the gains.** The paper does not establish that factorization is necessary or responsible for the reported improvements. Its own raw-surgery ablation performs similarly to or better than the factorized variants, suggesting that the gains may come from predictor surgery or training choices rather than semantic factorization.
2. **Paper contradicts the supplementary material.** There are also serious inconsistencies between the paper and the supplementary material. "Causal L1" is described as measuring responses to controlled interventions, but the implementation appears to be ordinary prediction of the second half of a video from the first half. The main paper describes approximately 1,000 hours of professionally collected footage, whereas the supplementary package describes a 276-hour YouTube-derived corpus.
3. **Manuscript appears unfinished.** Finally, the manuscript appears unfinished, with unresolved references such as "Figure ??," "Section," and "Appendix ??." The related-work discussion and positioning of the claimed V-JEPA 2.1 backbone are also insufficient.

### 🧭 Justification of Recommendation

I recommend rejection because the experiments do not isolate the paper's central contribution, and key descriptions of the evaluation and dataset do not match the submitted implementation. These issues directly affect the interpretation and credibility of the main results and cannot be resolved through minor clarification. This paper seems to be a unfinished one.

### ❓ Specific Points for Rebuttal

1. Why should the improvements be attributed to factorization when raw predictor surgery achieves comparable results?
2. Please reconcile the intervention-based definition of Causal L1 with its apparent temporal-completion implementation.
3. Please clarify the relationship between the claimed professionally collected 1,000-hour dataset and the 276-hour YouTube-derived corpus.
4. Please provide the exact provenance of V-JEPA 2.1 and explain the unresolved manuscript references.

| Rating | Confidence |
|---|---|
| **2**: Strong rejection | **4**: The reviewer is confident but not absolutely certain that the evaluation is correct |

## 🤖 AI Review (AI Reviewer, 27 Jul 2026)

### 📝 Synopsis of the Paper

The paper introduces DENSEWORLD, an urban-video benchmark focused on crowded mixed traffic, heterogeneous agents, occlusion, and weak spatial structure. It also proposes FactorJEPA, which replaces a monolithic video-JEPA predictor with layout, agent, and interaction channels supervised by automatically generated structural targets. The evaluation examines future-latent prediction, intervention sensitivity, robustness to masking, and motion accessibility across two backbone scales.

### 🧾 Summary of Review

The benchmark addresses an underrepresented setting, and the city-disjoint audit design provides useful separation between pseudo-supervision and evaluation. However, Figure 7 shows that FactorJEPA-RAW or different specialized FactorJEPA variants supply several reported wins, so the results do not isolate a fixed, semantically supervised FactorJEPA model. The stated interaction sparsity objective is also scale-degenerate, and the normalized visibility gate does not provide the absolute attenuation claimed in the text. The benchmark characterization and predictor information boundary require stronger evidence before the central structured-forecasting claims can be assessed.

### 💪 Strengths

1. **Concrete, underrepresented challenges.** The benchmark targets concrete challenges that conventional lane-structured driving data may underrepresent. The DENSEWORLD section identifies agent density, occupancy, occlusion, interaction pressure, and heterogeneity, while Figure 1 illustrates broad scene, infrastructure, and viewpoint coverage. This creates a potentially useful setting for studying prediction under mixed traffic and partial observability.
2. **Pseudo-labels separated from the test signal.** The evaluation separates pseudo-label generation from principal testing signals. The "Evaluator Independence and Audit Split" section specifies a city-disjoint, independently annotated audit set excluded from training, model selection, threshold selection, and tuning, with separate frozen evaluators for future embeddings and motion. This reduces the risk that agreement with DINOv2-derived training targets is mistaken for improved prediction.
3. **Confounders controlled.** The adaptation study controls several major experimental confounders. The "Protocol and Comparisons" section states that methods share clips, masking policies, optimization steps, evaluation splits, and metric implementations, and evaluates two V-JEPA scales. This makes the included adaptation comparisons more informative than comparisons based on separately tuned protocols.

### ⚠️ Weaknesses

1. **Headline evidence does not isolate one fixed FactorJEPA.** The headline evidence does not isolate improvement from the complete, semantically supervised FactorJEPA mechanism. The attribution chain defines FactorJEPA-RAW as architecture without factor supervision, yet Figure 7 reports RAW at `[math]` Causal L1 versus `[math]`–`[math]` for supervised variants; different specialized variants also lead the Future-frame and Mask-ratio panels. Figure 6 then reports singular "FactorJEPA" advantages without identifying a fixed variant, while its full-data panel provides only confidence-interval units rather than absolute scores.
2. **Interaction sparsity objective is scale-degenerate.** In the interaction and synthesis equations, replacing `[math]` by `[math]` while scaling `[math]` and `[math]` by `[math]` preserves both `[math]` and the interaction-head output but reduces `[math]`. Because the stated final objective imposes no dictionary-norm constraint, the weights need not encode sparse or meaningful interaction strength despite the paper's rescaling claim.
3. **Operating regime only partially demonstrated.** The "How Dense is DENSEWORLD?" section defines five axes, but Figure 2 reports only count density and occupancy and represents the stated BDD100K/nuScenes comparison with one "Representative West" series. Those measurements do not establish the subsequent claims of quantitatively higher occlusion pressure, interaction pressure, and heterogeneity.
4. **Normalized visibility gate does not suppress absolutely.** For `[math]`, uniformly reducing all `[math]` approximately leaves `[math]` unchanged when the denominator dominates `[math]`; with one such agent, `[math]`. Thus the equation primarily changes relative agent weighting and does not substantiate the stated mechanism of suppressing globally uncertain or occluded observations.
5. **Novelty under-positioned.** SlotFormer already predicts object-centric futures through spatiotemporal relations, while DriveWorld separates static scene information from temporal latent dynamics; neither work appears in the manuscript's body or references (Wu et al., 2023; Min et al., 2024). FactorJEPA may remain distinct through its JEPA-space dictionaries and dedicated interaction supervision, but those differences need to define the novelty rather than the broader idea of factorized future modeling.
6. **Predictor information boundary not specified.** The prediction equations consume agent representations `[math]` and edge sets `[math]`, but the main body does not state whether they are constructed solely from context frames, while the preceding DINOv2 pipeline associates regions over complete clips. Since these variables directly enter the future predictor, their context-only construction and inference-time availability are critical to the validity of the reported forecasting task.

### 💡 Suggestions for Improvement

1. **Scope India vs Global South.** The scope should distinguish evidence from India from claims about Global South urban environments generally. DENSEWORLD contains footage from 22 Indian cities, so the current title and framing extend beyond the geographic evidence presented. Positioning India as the studied instance and identifying cross-country validation as a separate generalization question would make the benchmark claim more precise.
2. **Linear contribution roadmap.** A linear contribution roadmap would make the two-part paper easier to assess. The benchmark evidence, canonical FactorJEPA configuration, training-only supervision, and experimental variants should be introduced before the detailed baseline equations and dense result figures. Consistent naming between the method and labels such as 3sDI, 3sDI·diH, and 3sDI·int would prevent readers from reconstructing the model identity from figure legends.
3. **Failure analysis by regime.** Failure analysis should connect model behavior to the benchmark's motivating phenomena. Breakdowns by density, occlusion, rare actor type, interaction pressure, viewpoint, and city would show whether performance changes occur specifically in the difficult scenes motivating the architecture rather than reflecting a dataset-wide latent-fitting effect.

### 🔍 Additional Details and Minor Issues

- The manuscript contains unresolved references including "Figure ??," "Section ," and "Appendix ??".
- The baseline objective encodes `[math]`, whereas the method encodes the full clip before applying `[math]`; these operations are not generally equivalent.
- Figure 4 depicts unnormalized channel sums, while the equations use normalized agent and interaction aggregates.
- The abstract reports an upper cross-scale correlation of `[math]`, while Figure 8 and the main text report `[math]`.
- The relationship between approximately 1,000 recorded hours and the full 115k-clip training corpus is not explained.
- The conclusion introduces a Cosmos decoder and oracle control that are not presented in the method or results.

### 📚 References cited by the AI reviewer

- Min, C., Zhao, D., Xiao, L., Zhao, J., Xu, X., Zhu, Z., Jin, L., Li, J., Guo, Y., Xing, J., Jing, L., Nie, Y., & Dai, B. (2024). DriveWorld: 4D pre-trained scene understanding via world models for autonomous driving. *CVPR 2024*, 15522–15533.
- Wu, Z., Dvornik, N., Greff, K., Kipf, T., & Garg, A. (2023). SlotFormer: Unsupervised visual dynamics simulation with object-centric models. *ICLR 2023*.

## 🎓 Learnings Applied to the AAAI-27 Student Abstract

Target: `overleaf/v3_AAAI27_student_abstract/0_main_AAAI.tex` (2 pages) + `1_supplement_full_paper.pdf`. Applied 2026-09-25.

| # | Reviewer finding | Raised by | Status | What changed |
|---|---|---|---|---|
| 1 | 1,000 h "professionally collected" vs 276 h YouTube corpus | R2, AI | ✅ fixed | Abstract + Problem: **276 hours, 714 public YouTube videos, 22 Indian cities, clips of 4 to 10 s** (from [clips.csv](../../src/hf_denseworld_115k/clips.csv): 115,687 clips, 275.8 h). Supplement patched to the same numbers everywhere (abstract, data section, conclusion, appendix construction text and stats table, limitations): 714 YouTube videos, 276 h, clips of 4 to 10 s; the two professional-collection sentences replaced by the v1 YouTube wording. |
| 2 | Causal L1 described as an intervention, implemented as second-half prediction | R2 | ✅ fixed | Diagnostics: "Causal L1: error when the second half of a clip is predicted from its first half" (matches the [causal code](../../src/utils/pt_causal.py)). "Intervention regions" dropped from the Protocol sentence. |
| 3 | "Ours" = RAW / 3sDI·int / WiSE-FT variants per cell, not one fixed model | R2, AI | ✅ fixed | Table 1 now reports **one fixed arm in every cell** (code name in the note under this table, the configuration the Method section describes). Every margin claim re-checked with that arm (table below): abstract and Findings text unchanged. |
| 4 | Inference-time construction of agent / interaction inputs unclear; information boundary | R1, AI | ✅ fixed | Method: "at inference the predictor reads nothing but the encoder's context tokens, with no DINOv2 pipeline, tracks, or agent proposals" (evaluation calls `predictor(z, [m_enc], [m_pred])` in the [predictor eval code](../../src/utils/predictor_eval.py)). |
| 5 | Global South claim beyond the Indian evidence | AI | ✅ fixed | Title and text say **South Asian streets / 22 Indian cities**; no Global South claim in the abstract. Supplement: abstract, DENSEWORLD section title and its opening sentence now say South Asian; the two Global South mentions left are the limitations paragraph that scopes the claim. |
| 6 | Novelty vs SlotFormer and DriveWorld not positioned | AI | ✅ fixed | One contrast sentence at the end of Method + both references added to `factorjepa.bib`. |
| 7 | Unresolved "Figure ??" / "Appendix ??" | R2, AI | ✅ verified | 0 "??" in the compiled abstract and supplement PDFs. |
| 8 | Only count density and occupancy measured, not all five axes | AI | ✅ scoped | Abstract claims only "consistently higher agent density and occupancy than BDD100K and nuScenes". |
| 9 | Naming 3sDI / 3sDI·diH / 3sDI·int | AI | ✅ avoided | The abstract names one model, FactorJEPA. |
| 10 | Cosmos decoder / oracle control announced but absent; ρ upper bound mismatch | AI (minor) | ✅ avoided | Neither appears; the abstract states only ρ ≥ 0.895. |
| 11 | V-JEPA 2.1 provenance | R2 | ✅ cited | `murlabadia2026vjepa21` cited at first mention. |
| 12 | Cross-channel leakage diagnostic never reported | R1 | ⬜ open | No leakage number exists in any tracked eval snapshot (`grep leakage` finds only split and data leakage code). Needs a new eval run on the audit split; not claimable in the abstract. |
| 13 | Raw-surgery ablation ≈ factorized variants (attribution) | R2, AI | ⬜ open | The abstract shows no ablation and the fixed arm removes `surgery_raw` from every cell, so the abstract no longer contradicts itself. A paired RAW versus 3sDI test is still needed for the full paper. |
| 14 | No evaluation outside the JEPA latent space | R1 | ⬜ open | Motion cosine (linear probe on the audit split) is the only external readout in the abstract. A downstream task needs new experiments. |
| 15 | Sparsity objective scale-degenerate; visibility gate is relative, not absolute | AI | ⬜ open | The abstract states only "sparse weighted sums" and "down-weights occluded agents" (relative wording, no absolute-suppression claim). The equation fix (dictionary-norm constraint) belongs to the full paper. |
| 16 | Failure analysis by density / occlusion / city | AI | ⬜ open | No space in 2 pages; full-paper item. |

Code name for row 3: the fixed arm is `surgical_3stage_DI_diheavy`.

### 📐 Margin check with the fixed arm (paired 95% BCa CI widths; ≥ 1 = separated, negative = rival ahead)

| Diagnostic | 2B · 10k | 1B · 10k | 1B · 115k |
|---|---|---|---|
| Future-frame L1 | 6.1 | 3.3 | 33.2 |
| Causal L1 | 1.9 | 1.4 | 13.9 |
| Mask-ratio slope | 0.9 | 1.9 | 43.3 |
| Motion cosine | -18.0 | -13.2 | 20.0 |

Findings text ("separates on Future-frame L1 and Causal L1 at both scales and on Mask-ratio slope at 1B; Motion cosine favors generic fine-tuning; at 115k all four separate by 13.9 to 43.3 widths") holds for the fixed arm exactly as it did for the per-cell best arm. Below the rule the fixed arm now loses most 10k rows to the rival (temporal order, pace, arrow-of-time, TCC, exposure bias, rollout drift, L1-vs-Δt at 2B), which the caption states.
