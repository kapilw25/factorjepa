> ChatGPT review of the [v5_Continual_Causality](../../../v5_Continual_Causality/0_main.pdf) draft for Continual Causality, from the judge prompt [prompt.md](../../input/prompt.md), on the 4 open-venue PDFs on GitHub main as of commit 52c2809c (PDFs unchanged since fd37d96b; ChatGPT received them as factorJEPA_<draft>.pdf). Saved Fri Nov 27 2026 ~6:50 PM PST. The review judged all 4 drafts in one answer: this file keeps the shared parts (opening, Section 0, Sections 5 to 10, final verdict) and Section 3 for v5; the other drafts' reviews are in [v1_Human_Centric](../v1_Human_Centric/by_chatgpt.md), [v3_Cog2Soc_fixes](../v3_Cog2Soc_fixes/by_chatgpt.md), [v4_Data2AgentEx](../v4_Data2AgentEx/by_chatgpt.md). Only the final review was saved, no Step 1 or Step 2 answers. Chat-page widget leftovers were turned into tables and repeated source chips into one line; every judge sentence and number is kept.

# **FactorJEPA — AAAI-27 Workshop & Bridge Review**

## **Executive verdict**

I evaluated the four PDFs as independent submissions, using the ten accepted papers from Step 1 as calibration references and adjusting the review criteria to each venue's scientific priorities.

The most important finding is that **FactorJEPA is already a technically substantial short paper. Its principal acceptance risk is not insufficient experimentation, but whether the experiments establish the scientific contribution expected by each venue.**

## **Current submission assessment**

Before any of the proposed improvements

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Venue and draft | Acceptance probability | Reviewer score |
| --- | --- | --- |
| Data2AgentEx — v4 | **79%** | 77/100 |
| Human-Centric AgMS — v1 | **77%** | 75/100 |
| Cog2Soc — v3 | **65%** | 68/100 |
| Continual Causality — v5 | **46%** | 60/100 |
| Typical AAAI-27 CV/embodied-AI workshop | **71%** | |

These are subjective reviewer probability estimates, not empirically measured acceptance rates. They assume that concurrent submission is permitted by the relevant venues and that anonymity and submission requirements are satisfied.

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Venue               | Technical substance | Thematic fit         | Current verdict |
| ------------------- | ------------------- | -------------------- | --------------- |
| Data2AgentEx        | 🟢 Strong           | 🟢 Excellent         | Accept          |
| Human-Centric AgMS  | 🟢 Strong           | 🟢 Excellent         | Accept          |
| Cog2Soc             | 🟢 Strong           | 🟡 Good but indirect | Weak accept     |
| Continual Causality | 🟢 Strong           | 🟠 Incomplete        | Borderline      |

The same experimental contribution has substantially different value depending on whether the reviewer is looking for data curation, mobility intelligence, behavioural modelling, or continual causal learning.

# **0. Shared scientific assessment across all four drafts**

Before evaluating the venue-specific versions, it is important to establish exactly what the common experiments demonstrate.

## **0.1 What FactorJEPA demonstrates**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Scientific finding                                                 | Evidence                                                                                      | Assessment                           |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------- | ------------------------------------ |
| DENSEWORLD-115k addresses an underrepresented traffic regime       | 115,687 clips, 276 hours, 714 source videos, 22 Indian cities; §2, pp. 1–2                    | 🟢 Strong resource contribution      |
| Dense scenes differ from the Western reference                     | Higher road-user counts and occupancy in all six compared scene categories; Fig. 2, p. 2      | 🟢 Supported on the measured axes    |
| Staged adaptation improves embedding prediction                    | Future-frame and causal-block L1 improve at both model scales; Table 1, p. 3                  | 🟢 Supported                         |
| Structured training views improve robustness to artificial masking | Mask-ratio slope improves over RAW by approximately 6% at 2B and 2% at 1B; Table 2, p. 4      | 🟢 Supported                         |
| The schedule explains most of the embedding-prediction gain        | Plain → RAW → Ours comparison; Table 2, p. 4                                                  | 🟢 Particularly valuable ablation    |
| Full-data adaptation improves four selected diagnostics            | Four advantages against LoRA and the frozen backbone, 13.9–43.3 interval widths; Fig. 3, p. 3 | 🟢 Supported within compared methods |
| Temporal coherence remains a weakness                              | Six clear rival advantages at 115k, spanning 8.3–146.5 interval widths; Fig. 3, p. 3          | 🟠 Important negative finding        |
| Real human interaction prediction improves                         | No human-validated behavioural prediction test; §6, p. 4                                      | 🔴 Not established                   |
| Continual knowledge retention improves                             | Original-distribution retention not evaluated; v5 §5, p. 4                                    | 🔴 Not established                   |
| Mobility planning or driving safety improves                       | No trajectory forecasting, planner or safety evaluation; §6, p. 4                             | 🔴 Not established                   |

*Sources cited by the judge:* `factorJEPA_v3_Cog2Soc.pdf`, `factorJEPA_v4_Data2AgentEx.pdf`, `factorJEPA_v5_Continual_Causality.pdf`

## **0.2 The most scientifically interesting result is Table 2, not Figure 3**

### **What contributes to FactorJEPA's performance?**

*Future-frame L1 · lower is better · Table 2, p. 4*

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Model and data | Plain | RAW | Ours |
| --- | --- | --- | --- |
| 2B model — 10k clips | 0.533 | 0.497 | 0.496 |
| 1B model — 10k clips | 0.607 | 0.587 | 0.590 |

At 2B, Ours versus RAW on this diagnostic is not statistically distinguishable at the reported 95% level. This is not evidence of equivalence. At 1B, RAW has lower future-frame error.

### **The distinctive utility of structured views**

*Mask-ratio slope · lower is better · Table 2, p. 4*

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Model and data | Plain | RAW | Ours |
| --- | --- | --- | --- |
| 2B model — 10k clips | 0.0625 | 0.0565 | 0.0531 |
| 1B model — 10k clips | 0.0586 | 0.0528 | 0.0515 |

Both structured-view improvements over RAW exclude zero at 95% under clip and source-video resampling, according to Table 2.

This establishes a more nuanced research contribution than simply claiming FactorJEPA outperforms other adaptation methods:

**The staged update schedule improves predictive accuracy, while structured data curation selectively improves robustness to missing visual information, with measurable costs in other capabilities.**

That is an interesting empirical finding even though the structured curriculum does not dominate every metric.

## **0.3 Six technical concerns affecting every submission**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Concern                                                                   | Severity                             | Evidence                                                           |
| ------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------ |
| Source-video overlap between training and testing                         | 🔴 High                              | §4, p. 3 and §6, p. 4 explicitly acknowledge clip-level splitting  |
| Human validity of automatic interaction labels                            | 🔴 High for Cog2Soc and Data2AgentEx | §2 and §6: proximity and relative motion, without human validation |
| Artificial token masking is not equivalent to natural road-user occlusion | 🟠 High                              | Mask-ratio slope definition, §4, p. 3                              |
| Only two rivals at full data scale                                        | 🟠 Moderate–high                     | Table 1 and §4, p. 3                                               |
| No independent training-seed repetitions                                  | 🟠 Moderate                          | §6, p. 4                                                           |
| Downstream task utility remains untested                                  | 🔴 Venue-dependent                   | §6, p. 4                                                           |

### **Important statistical distinction**

Table 2's source-video bootstrap over 127 videos is a valuable robustness check. However, **resampling videos is not the same as holding out videos during training**. It addresses dependence among evaluation samples, not possible source-specific information shared by the training and testing sets.

Likewise, the reported 95% bootstrap intervals characterize test-sample uncertainty conditional on the trained checkpoints. They do not establish variability across independent training runs.

I would also replace the unusual interval-width visualization in Figure 3 with ordinary paired effect estimates and confidence intervals. A gain of 43.3 interval widths is not equivalent to a 43.3-fold performance improvement, and the one-interval-width convention used in its legend is not itself a standard significance test.

# **3. v5 — Continual Causality Bridge**

*Staged Adaptation of a Pretrained Video World Model to Dense South Asian Streets: What It Gains and What It Costs*

AAAI-27 Bridge · Novel-work track · Deadline December 1 AoE · Approximately 4 days

## **PART A — Acceptance probability**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Continual Causality | Typical CV workshop | Reviewer score |
| --- | --- | --- |
| **46%** | **71%** | **60/100** |

### **Reviewer assessment**

This is the least naturally aligned submission. FactorJEPA studies continued pretraining under a domain shift, which is relevant to continual learning, but its three training stages are not demonstrated to represent a changing environment, nor is the retention of knowledge from the original distribution evaluated. More importantly, the paper explicitly states that it makes no causal claim (§5, p. 4). The accepted continual-causality papers from Step 1 had modest or nonexistent empirical evaluations, but they compensated through direct conceptual contributions involving independent mechanisms, local spurious features or causal reasoning. FactorJEPA has much stronger conventional experiments than those papers, yet currently lacks their central scientific connection. I would classify it as borderline rather than an outright rejection because the controlled adaptation study, frozen teacher, progressive unfreezing and partial-observability analysis provide a plausible basis for discussion.

### **Three strongest points**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Rank | Strength                                                                         | Evidence           |
| ---- | -------------------------------------------------------------------------------- | ------------------ |
| 🟢 1 | Empirical investigation of foundation-model adaptation under a domain shift      | §1, pp. 1–2; §4    |
| 🟢 2 | A staged update design that incorporates anchoring to starting weights           | §3, p. 2           |
| 🟢 3 | Controlled study of predictive accuracy versus other representation capabilities | Tables 1–2; Fig. 3 |

### **Three most likely rejection reasons**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Rank | Rejection reason                                                                                             | Evidence                                                            |
| ---- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| 🔴 1 | No demonstrated causal mechanism, intervention on an underlying causal variable or causal structure learning | §5 explicitly states no causal claim                                |
| 🔴 2 | No knowledge-retention or catastrophic-forgetting evaluation                                                 | Abstract and §5, p. 4 identify retention as future work             |
| 🔴 3 | The curriculum is static-domain adaptation, not a demonstrated non-stationary learning stream                | §3: layout → agent → interaction views derived from the same corpus |

*Source cited by the judge:* `factorJEPA_v5_Continual_Causality.pdf` (2 citations)

## **PART B — Ten reviewer rubrics**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Rubric                            | Why it matters here                                         | Score | Paper evidence                                    | Highest-value change                          |
| --------------------------------- | ----------------------------------------------------------- | ----- | ------------------------------------------------- | --------------------------------------------- |
| 1. Venue relevance                | Must connect continual learning with changing causal worlds | 🟡 6  | §1; §5                                            | Evaluate retention and structured changes     |
| 2. Causal-mechanism insight       | Central distinguishing theme of the bridge                  | 🔴 4  | §5 explicitly limits causal claims                | Controlled environment-factor perturbations   |
| 3. Continual retention            | Adaptation must preserve useful prior knowledge             | 🔴 3  | No original-distribution evaluation               | Source-domain retention experiment            |
| 4. Non-stationary evaluation      | Must distinguish ongoing learning from one-time adaptation  | 🔴 4  | §3: fixed training curriculum                     | Ordered domain stream                         |
| 5. Technical novelty              | Staged adaptation with anchoring is substantive             | 🟢 8  | §3; Table 2                                       | Stage-wise checkpoint analysis                |
| 6. Partial-observability analysis | Directly relevant to incomplete observations                | 🟢 8  | Mask-ratio slope; Tables 1–2                      | Natural and controlled occlusion              |
| 7. Shortcut/confounding analysis  | Causal learners should resist spurious evidence             | 🟡 6  | Structured masking, but no explicit shortcut test | Background-dependence perturbation            |
| 8. Experimental rigor             | Empirical conclusions must be controlled                    | 🟡 7  | Seven methods and raw ablation                    | Source-disjoint validation                    |
| 9. Statistical robustness         | Needed to distinguish adaptation trade-offs                 | 🟡 7  | Paired BCa intervals; Table 2                     | Source-level effects and seed replication     |
| 10. Conceptual clarity            | Causal and predictive claims must remain distinct           | 🟡 7  | §5 clearly distinguishes the two                  | Rename causal-block L1 and sharpen hypotheses |

**Total: 60/100**, Borderline

### **Comparison with Group C accepted papers**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Calibration paper                           | Estimated score | Comparison                                                                           |
| ------------------------------------------- | --------------- | ------------------------------------------------------------------------------------ |
| P6: From IID to Independent Mechanisms      | 80/100          | Direct conceptual connection plus synthetic continual-learning evidence              |
| P7: Spurious Features in Continual Learning | 72/100          | No empirical evaluation, but an original problem formulation directly within scope   |
| P8: Never Ending Reasoning and Learning     | 71/100          | Strong interdisciplinary causal-learning research agenda                             |
| FactorJEPA v5                               | **60/100**      | Stronger quantitative experiments but weaker causal and continual-learning substance |

For genuine position papers, rubrics concerning experiments are interpreted according to the contribution, rather than requiring model-training results. This is especially important when comparing P7 and P8 with FactorJEPA.

### **The critical distinction**

P6 investigates how independent mechanisms could help continual learners adapt when an underlying mechanism changes.

FactorJEPA currently organizes its training observations into semantic views. Those views are not necessarily independent mechanisms, and changing what is visible in an image is not necessarily an intervention on the causal process that generates the behaviour.

Do not claim causal disentanglement merely because the method is named FactorJEPA.

## **PART C — Improvements within approximately four days**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Rank | Improvement and what it establishes                                                                                                                                              | Resources                                                                                                         | Placement and cuts                                                                                       | Rubric gain                                      |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 1    | **Knowledge-retention evaluation.** Measure performance before and after adaptation on an untouched reference distribution, alongside DENSEWORLD performance.                    | Existing 1B frozen, Plain, RAW, Ours and LoRA checkpoints; approximately 4–8 GPU hours on a limited reference set | Continual-only §5 table, replacing most of its current discussion                                        | Retention +3; venue relevance +1                 |
| 2    | **Stage-wise adaptation analysis.** Evaluate layout, agent and interaction stage checkpoints on identical target and reference sets, if intermediate checkpoints exist.          | 4–8 GPU hours; no retraining if saved checkpoints are available                                                   | Combine with retention table in §5; compress repeated Table 2 interpretation                             | Non-stationary evidence +1; technical insight +1 |
| 3    | **Controlled input-factor perturbation.** Compare responses when background evidence, individual-road-user evidence or pairwise evidence is suppressed at matched mask coverage. | 1B FactorJEPA, RAW and LoRA on 300–500 clips; 4–8 GPU hours                                                       | Replace shared Fig. 3 or use a Continual-only compact plot in §5                                         | Shortcut analysis +2; causal relevance +1        |
| 4    | **Small sequential-domain pilot.** Adapt across two or three predefined scene groups, measuring performance after each transition on previous and new groups.                    | 1B staged versus plain adaptation; approximately 18–30 GPU hours for a capped, small pilot                        | Continual-only §5; use this instead of the stage-checkpoint table if it supplies stronger evidence       | Non-stationarity +2; retention +1                |
| 5    | **Reframe the scientific question.** Distinguish structured predictive adaptation, controlled visual perturbation and causal mechanism learning.                                 | 0 GPU hours                                                                                                       | Continual-only title, abstract, §1 and §5; rename causal-block L1 throughout all four drafts if feasible | Conceptual clarity +1; relevance +1              |

These experiments are alternatives where they compete for the same space. The most valuable combination is retention evaluation plus stage-wise analysis and a tightly scoped visual perturbation test.

### **Highest-value new table**

The following is a proposed table structure, not existing results.

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Checkpoint          | Reference future L1 ↓ | Dense future L1 ↓ | Reference motion ↑ | Dense motion ↑ |
| ------------------- | --------------------- | ----------------- | ------------------ | -------------- |
| Original V-JEPA 2.1 | Measure               | Measure           | Measure            | Measure        |
| Plain adaptation    | Measure               | Measure           | Measure            | Measure        |
| RAW staged          | Measure               | Measure           | Measure            | Measure        |
| FactorJEPA          | Measure               | Measure           | Measure            | Measure        |
| LoRA                | Measure               | Measure           | Measure            | Measure        |

For meaningful retention comparisons:

- Use exactly the same reference test examples and masking configurations for all checkpoints.
- Use a common target encoder for embedding-error comparisons, or clearly establish comparability.
- Fit motion probes under a consistent protocol with disjoint training and testing groups.
- If using BDD100K or nuScenes, describe them as external reference or proxy retention distributions unless their presence in original V-JEPA pretraining is established.
- Report degradation, improvements and uncertainty regardless of direction.

A reference-set experiment alone would not prove retention across the complete original pretraining distribution. It would establish a narrower, defensible retention finding.

### **Proposed title improvement**

**Structured Continual Adaptation of Video World Models under Distribution Shift and Partial Observability**

This is stronger than the current subtitle “What It Gains and What It Costs,” which makes the paper sound primarily like an adaptation report.

If a retention experiment is completed, the title can become more specific. Do not use “causal world model” in the title unless the corresponding causal claim is actually demonstrated.

### **What should be cut**

The existing §5 largely repeats the numerical results from Table 2. Replace that repetition with direct evidence of adaptation and retention. Compress the general traffic motivation in §1 and remove venue-irrelevant prospective social-force comparisons from this version's §6 if creating a venue-specific limitations section is permitted.

**Conditional target after feasible improvements: 72–76/100**

Estimated acceptance probability: **approximately 67–76%** if the retention and sequential-adaptation evidence is credible.

This submission has the largest potential improvement from a small, well-targeted experiment, but remains the most uncertain because visual perturbations alone do not establish causal mechanisms.

## **PART D — Double-blind check**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Location                       | Concern                                                             | Anonymous replacement                                 |
| ------------------------------ | ------------------------------------------------------------------- | ----------------------------------------------------- |
| p. 1, dataset link             | Searchable public project identity                                  | Anonymized artifact with verified neutral ownership   |
| p. 1, footnote 1               | Other three submissions, preprint and student abstract              | Confidential disclosure to chairs where policy allows |
| pp. 1–4                        | Exact dataset and method names can identify the arXiv preprint      | Policy-compliant neutral naming if necessary          |
| §3 and supplementary artifacts | Checkpoint filenames and training logs could expose author identity | Neutral checkpoint identifiers and sanitized logs     |
| p. 4, References               | No obvious identifying self-citation found                          | Retain ordinary references                            |

PDF metadata contains no author identity and no embedded files. The public preprint creates a potential discoverability issue, but public-preprint existence is not automatically a double-blind violation. The applicable venue policy determines what can be cited or linked.

The novel-work requirement also makes disclosure of the existing public preprint and pending student abstract important. Confirm eligibility with the organizers without revealing authorship to assigned reviewers unnecessarily.

## **PART E — Tone audit**

### **E.1 Venue-specific rewrites**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| #  | Page/section   | Quoted original                                                                                                                                                                                                            | Problem                               | Rewrite                                                                                                                                                                                                                  |
| -- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| K1 | p. 1, Title    | “What It Gains and What It Costs”                                                                                                                                                                                          | Engineering evaluation tone           | “Predictive Adaptation under Distribution Shift and Partial Observability”                                                                                                                                               |
| K2 | p. 1, Abstract | “A world model is often updated by continued pretraining when the world it models changes, and a new traffic culture is a sharp change: denser scenes, different road users, and many of them only partly visible.”        | Background-first                      | “We introduce a structured adaptation curriculum for video world models facing domain shifts characterized by increased density, heterogeneous road users, and partial visibility.”                                      |
| K3 | p. 1, Abstract | “An ablation on raw clips shows that the staged schedule carries most of the gain, while the structured clips make prediction degrade more slowly as more of the scene is hidden.”                                         | Contribution undersold                | “A controlled raw-clip ablation distinguishes the predictive benefits of staged updating from the additional masking robustness supplied by structured training views.”                                                  |
| K4 | p. 1, Abstract | “Competing methods still lead on temporal coherence, and retention on the original distribution is the next test.”                                                                                                         | Negative closing                      | “The evaluation identifies complementary requirements for continual adaptation: competing methods retain temporal-coherence advantages, and original-distribution retention remains to be evaluated.”                    |
| K5 | p. 1, §1       | “Adapting to it is a continual-learning problem under partial observability.”                                                                                                                                              | Overstates experimental setting       | “This domain shift provides a test case for structured continued pretraining and motivates continual-learning questions under partial observability.”                                                                    |
| K6 | p. 4, §5       | “Two results speak to the bridge’s question of keeping a world model accurate when its world changes and is only partly observed.”                                                                                         | Venue-directed rather than scientific | “The experiments isolate two properties relevant to continual world-model adaptation: the predictive effect of staged updates and robustness to incomplete visual evidence.”                                             |
| K7 | p. 4, §5       | “Both gains come with costs a continual learner has to weigh, in motion readout and in the coherence of long predicted futures, where the rivals still lead.”                                                              | Negative framing                      | “These gains characterize a capability trade-off: reduced motion readout and stronger rival performance on long-horizon temporal-coherence diagnostics remain important considerations for continual adaptation.”        |
| K8 | p. 4, §5       | “We make no causal claim: Causal L1 names the direction of prediction, from the first half of a clip to the second, not an identified cause.”                                                                              | Defensive, misleading metric name     | “Causal L1 denotes temporally directed embedding prediction from the first half of a clip to the second. The metric characterizes predictive structure rather than identifying causal relationships.”                    |
| K9 | p. 4, §5       | “Whether the adapted model keeps its accuracy on the original distribution is the next test, and the layout, agent, and interaction clips offer a way to ask which part of a changing scene a world model has to relearn.” | Negative and prospective              | “The layout, agent, and interaction representation establishes a framework for studying selective relearning across scene components. Evaluation of retention on the original distribution remains an open requirement.” |

The shared-section audit below also applies to v5.

### **E.2 Three rewrites that matter most**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Priority | Change                       | Why                                                                                               |
| -------- | ---------------------------- | ------------------------------------------------------------------------------------------------- |
| 1        | K5: continual-learning claim | Distinguishes domain adaptation from demonstrated continual learning                              |
| 2        | K8: causal terminology       | Prevents a causal-learning reviewer from interpreting the metric as a causal-identification claim |
| 3        | K9: retention                | Defines a central open question without implying that retention has been established              |

### **E.3 Rewritten abstract**

We introduce FactorJEPA, a structured adaptation curriculum for pretrained video world models encountering dense, partially observable street scenes. Our target distribution, DENSEWORLD-115k, comprises 115,687 clips (276 hours) from 714 public YouTube videos spanning 22 Indian cities, released as reconstructible metadata. FactorJEPA adapts V-JEPA 2.1 through successive layout, road-user, and interaction views, progressively unfreezing encoder blocks while anchoring updates through a frozen teacher and decay toward starting weights. Against seven adaptation methods, including continual self-supervised learning, under matched clips and budgets, a fixed configuration improves masked and future-content prediction at 2B and 1B scale. Following full 115k-clip training, it separates from the stronger of the two evaluated rivals on four predictive-structure diagnostics by 13.9 to 43.3 widths of the paired 95% confidence interval. A controlled raw-clip ablation identifies staged updating as the primary source of predictive gains, while structured views reduce error growth under increased masking. Competing methods retain advantages in temporal coherence. These results characterize adaptation trade-offs under distribution shift and partial visual observability, with retention on the original distribution remaining an open evaluation requirement.

# **5. Shared-section tone audit — Applies to ALL four PDFs**

Because §2, §3, §4, §6 and the figures and tables are shared, the following audit completes Part E.1 for **each** submission.

Implementing these rewrites in the shared source will improve all four papers simultaneously. I have consolidated the identical passages rather than reproducing the same table four times.

## **E.1 Shared passages requiring revision**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| #   | Page/section                       | Quoted original                                                                                                                                                                                                                                                                                          | Problem                                            | Research-centred rewrite                                                                                                                                                                                                                                                                                            |
| --- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S1  | p. 1, Contributions                | “wins and losses alike”                                                                                                                                                                                                                                                                                  | Informal, defensive                                | “including capability-specific improvements and trade-offs”                                                                                                                                                                                                                                                         |
| S2  | p. 1, §2                           | “For 86,831 clips an automatic pipeline finds the road users: Grounding DINO (Liu et al. 2024a) detects them on anchor frames, SAM 3 (Carion et al. 2025) tracks their masks through the clip, and pairs of road users that are close and move relative to each other are marked as interactions.”       | Engineering-first description                      | “For 86,831 clips, we construct structured road-user representations through anchor-frame detection with Grounding DINO (Liu et al. 2024a), temporal mask tracking with SAM 3 (Carion et al. 2025), and candidate interaction assignment based on proximity and relative motion.”                                   |
| S3  | p. 2, §2                           | “These two axes are the ones measured here; occlusion and interaction pressure are the natural next axes.”                                                                                                                                                                                               | Defensive limitation                               | “Road-user count and pixel occupancy establish two complementary dimensions of scene density, motivating subsequent evaluation of natural occlusion and interaction intensity.”                                                                                                                                     |
| S4  | p. 2, §2                           | “The gap lies in the data itself: a world model calibrated on the Western reference would meet denser scenes here on all six types.”                                                                                                                                                                     | Broad inference from limited evidence              | “The comparison establishes a density shift across all six matched scene categories, providing a targeted setting for evaluating world-model adaptation.”                                                                                                                                                           |
| S5  | p. 2, Responsible use              | “Rebuilt clips are the public videos as posted, so a video its creator removes can no longer be rebuilt, and rights holders can ask on the dataset page for a source to be removed.”                                                                                                                     | Operations tone                                    | “The metadata-based release preserves dependence on creator-hosted videos: removed videos are no longer reconstructible, and rights holders may request source removal through the dataset page.”                                                                                                                   |
| S6  | p. 2, Responsible use              | “The people filmed did not consent individually, so the data is meant for research on aggregate street behaviour, not on individuals.”                                                                                                                                                                   | Limitation requiring explicit retention            | “The corpus is intended for aggregate street-behaviour research rather than individual-level analysis, as individual consent from the people filmed was not obtained.”                                                                                                                                              |
| S7  | p. 2, §3                           | “FactorJEPA keeps V-JEPA 2.1’s encoder, predictor, masking policy, and prediction objective, and changes what the model is trained on and which blocks it updates.”                                                                                                                                      | Engineering configuration description              | “FactorJEPA isolates adaptation through training-view selection and staged parameter updating while retaining V-JEPA 2.1’s encoder, predictor, masking policy, and prediction objective.”                                                                                                                           |
| S8  | p. 2, §3                           | “At inference FactorJEPA is an ordinary V-JEPA model: no detector, tracker, or mask is needed.”                                                                                                                                                                                                          | Undersells method                                  | “FactorJEPA incorporates structured scene evidence through training while preserving V-JEPA’s inference formulation, without requiring detectors, trackers, or masks at inference.”                                                                                                                                 |
| S9  | p. 3, Table 1 caption              | “Above the rule: the four diagnostics of predictive structure; below: nine diagnostics of temporal coherence and semantics, where the rival leads on most rows.”                                                                                                                                         | Results framed as wins versus losses               | “The upper four rows characterize predictive structure; the lower nine characterize temporal coherence and semantics, with stronger rival performance on most of the latter diagnostics.”                                                                                                                           |
| S10 | p. 3, §4 Protocol                  | “Test clips are held out at the clip level (1,825 in the 10k regime, 23,106 in the full regime), so a source video can contribute clips to both training and test (Section 6).”                                                                                                                          | Necessary limitation, repeated later               | “The evaluation uses 1,825 held-out clips in the 10k regime and 23,106 in the full regime. The split evaluates clip-level generalization and permits shared source videos between training and testing (Section 6).”                                                                                                |
| S11 | p. 3, §4 Findings                  | “With limited data (Table 1), FactorJEPA separates from the best rival on Future-frame L1 and Causal L1 at both scales and on Mask-ratio slope at 1B, while Motion cosine favors full fine-tuning.”                                                                                                      | Results read as a performance report               | “The 10k-clip experiments identify a consistent predictive-accuracy advantage on Future-frame and Causal L1 at both model scales, together with separation on Mask-ratio slope at 1B; full fine-tuning retains stronger Motion cosine.”                                                                             |
| S12 | p. 3, §4 Findings                  | “Full fine-tuning, which leads on Motion cosine at 10k, was not trained at 115k, so whether more data closes that gap remains open.”                                                                                                                                                                     | Defensive limitation                               | “The full-data comparison establishes the Motion cosine advantage against LoRA and the frozen backbone. Full fine-tuning leads at 10k, and its relative performance at 115k remains untested.”                                                                                                                      |
| S13 | p. 3, §4 Findings                  | “On the diagnostics below the rule, the best rival leads at 115k on six (rollout drift, exposure bias, L1-versus-∆t decay, arrow-of-time, and the two TCC measures) by 8.3 to 146.5 interval widths.”                                                                                                    | Negative results presented as isolated losses      | “The full-data evaluation identifies a complementary temporal-reliability trade-off: the strongest rival retains advantages of 8.3 to 146.5 interval widths on six diagnostics, comprising rollout drift, exposure bias, L1-versus-∆t decay, arrow-of-time, and both TCC measures.”                                 |
| S14 | p. 3, §4 Findings                  | “FactorJEPA’s gain is in one-step prediction of crowded scenes; on the coherence of long predicted futures, the rivals still lead.”                                                                                                                                                                      | Narrow, defensive framing                          | “FactorJEPA specializes in improved one-step embedding prediction of crowded scenes, while the evaluated rivals retain advantages in longer-horizon temporal coherence.”                                                                                                                                            |
| S15 | p. 3, §4 Where the gain comes from | “The staged adaptation carries most of the gain: on raw clips alone it lowers Future-frame L1 by 0.036 (2B) and 0.020 (1B) below plain continued pretraining, and it already beats LoRA on Future-frame and Causal L1 at both scales.”                                                                   | Undersells discovery from ablation                 | “The ablation identifies staged updating as the primary source of predictive improvement: RAW reduces Future-frame L1 by 0.036 at 2B and 0.020 at 1B relative to plain continued pretraining, and achieves lower Future-frame and Causal L1 than LoRA at both scales.”                                              |
| S16 | p. 3, §4 Where the gain comes from | “The layout, agent, and interaction clips then flatten the Mask-ratio slope at both scales (by 6% at 2B and 2% at 1B), so prediction degrades more slowly as more of the scene is hidden, which is the situation of occluded road users.”                                                                | Artificial masking equated with natural occlusion  | “Structured layout, road-user, and interaction views provide additional robustness to missing visual evidence, reducing the Mask-ratio slope by 6% at 2B and 2% at 1B. Artificial masking provides a controlled proxy for incomplete observations, rather than direct validation on naturally occluded road users.” |
| S17 | p. 3, §4 Where the gain comes from | “They cost Motion cosine at both scales and a little prediction error: 0.001 in Causal L1 at 2B and about 0.003 in Future-frame and Causal L1 at 1B.”                                                                                                                                                    | Negative and “a little” minimizes measured effects | “Relative to RAW, structured views improve masking robustness while reducing Motion cosine at both scales and increasing Causal L1 by 0.001 at 2B and Future-frame and Causal L1 by approximately 0.003 at 1B.”                                                                                                     |
| S18 | pp. 3–4, §4                        | “Most of the gain therefore does not depend on the automatic labels, and what the labels buy is a specific property: robustness to missing evidence.”                                                                                                                                                    | Most damaging self-critical framing                | “The ablation distinguishes two complementary contributions: staged adaptation accounts for most predictive improvement, while automatically structured training views specifically strengthen robustness to missing visual evidence.”                                                                              |
| S19 | p. 4, Table 2 caption              | “Every RAW-minus-plain and Ours-minus-RAW difference excludes zero at 95% under both clip and source-video resampling (127 videos), except Future-frame L1 at 2B.”                                                                                                                                       | Statistically sound but can be clearer             | “All RAW-minus-plain and Ours-minus-RAW differences exclude zero at 95% under both clip and source-video resampling over 127 videos, with the exception of the 2B Future-frame L1 difference between Ours and RAW.”                                                                                                 |
| S20 | p. 4, §6                           | “Every diagnostic is measured in the frozen V-JEPA embedding space or with a linear probe on it.”                                                                                                                                                                                                        | Defensive scope statement                          | “The evaluation characterizes predictive representations using frozen V-JEPA embedding-space diagnostics and linear readouts.”                                                                                                                                                                                      |
| S21 | p. 4, §6                           | “Trajectory forecasting, a comparison with a calibrated social-force or agent-based model, and a check of the interaction clips against human judgements of who yields to whom are the next tests; none is reported here, and the interaction clips are chosen by a proximity and relative-motion rule.” | Long list of missing evidence                      | “The present contribution establishes representation-level evidence using candidate interaction views selected by proximity and relative motion. Trajectory forecasting, calibrated social-force or agent-based comparisons, and human validation of yielding behaviour remain outside the reported evaluation.”    |
| S22 | p. 4, §6                           | “The clip-level split lets a source video contribute to both training and test, and no city is held out.”                                                                                                                                                                                                | Serious methodological limitation                  | “The current protocol evaluates generalization across held-out clips, permitting source-video overlap and including no geographically held-out city.”                                                                                                                                                               |
| S23 | p. 4, §6                           | “Video-level and city-level splits, dense-traffic regions beyond India (all footage here is Indian), and repeated training runs (each configuration was trained once) are the natural extensions.”                                                                                                       | Ends with extensive future-work list               | “The current results characterize Indian street footage and one trained checkpoint per configuration. Independent source-video and city generalization, transfer beyond India, and training-run variability define complementary directions for extending this evaluation.”                                         |

The wording of the original passages is grounded in the shared sections of the four manuscripts.

*Sources cited by the judge:* `factorJEPA_v3_Cog2Soc.pdf`, `factorJEPA_v4_Data2AgentEx.pdf`, `factorJEPA_v5_Continual_Causality.pdf`, `factorJEPA_v1_Human_Centric.pdf`

Ordinary technical specifications that do not create a negative or operational narrative, such as the precise mixing ratios and number of unfrozen blocks, should be retained. They support reproducibility and do not need to be made artificially promotional.

## **5.1 Change the metric name across ALL four drafts**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Current terminology       | Recommended terminology                                                                       |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| Causal L1                 | Future-half prediction L1                                                                     |
| Causal-block L1           | Temporal-block prediction L1                                                                  |
| Interacting pairs         | Candidate interacting pairs, until independently validated                                    |
| Occlusion robustness      | Masking robustness, when referring to the current artificial-mask experiments                 |
| 13.9–43.3 interval widths | Retain the figure if needed, but supplement with effect sizes and actual confidence intervals |

This is especially important for Continual Causality. The word *causal* can reasonably describe one-directional temporal masking in an architectural context, but it has a much stronger interpretation in a causal-inference research community.

## **5.2 Figure 3 needs a statistical presentation improvement**

The current figure does not directly show the underlying confidence intervals, making readers interpret an unusual normalized statistic.

### **Current Figure 3: full-data capability profile**

*Chart axis:* Reported signed separation in widths of the paired 95% BCa interval

*Chart note:* Symmetric logarithmic-style visualization of the values already reported in Figure 3. These values are not percentage improvements.

> Note added when saving, not the judge's words: the chart was an interactive widget on the chat page, and only its axis label and note came through in the pasted text.

The result profile is interesting precisely because it distinguishes predictive gains from temporal limitations.

However, I would replace or supplement this figure with a compact table reporting:

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Diagnostic         | FactorJEPA | Best evaluated rival | Absolute difference, favourable direction | Paired 95% CI              |
| ------------------ | ---------- | -------------------- | ----------------------------------------- | -------------------------- |
| Future-frame L1 ↓  | 0.573      | 0.589                | +0.016                                    | Extract from saved results |
| Causal-block L1 ↓  | 0.597      | 0.607                | +0.010                                    | Extract from saved results |
| Mask-ratio slope ↓ | 0.047      | 0.058                | +0.011                                    | Extract from saved results |
| Motion cosine ↑    | 0.097      | 0.078                | +0.019                                    | Extract from saved results |
| TCC cycle ↓        | 2.806      | 2.132                | −0.674                                    | Extract from saved results |

Absolute differences above are calculated from the rounded Table 1 entries. Actual interval endpoints must come from the saved paired evaluations; they cannot be recovered reliably from the rounded values and interval-width ratios alone.

Also explain how the best rival was selected. If it was selected using the same test set, the comparison should be described as against an observed per-metric best baseline, with appropriate caution regarding selection and multiple comparisons.

## **5.3 Three most valuable shared rewrites**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Priority | Original                                                                     | Recommended framing                                                                           |
| -------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| 🥇 1     | S18: “Most of the gain therefore does not depend on the automatic labels...” | Separate the schedule's contribution from the specific robustness utility of structured views |
| 🥈 2     | S16: artificial masking described as occluded road users                     | Explicitly distinguish controlled masking from natural occlusion                              |
| 🥉 3     | S14: one-step gains but long-future losses                                   | Present capability-specific trade-offs rather than implying comprehensive superiority         |

These three changes improve the manuscript's scientific narrative without changing its experimental facts.

# **6. Cross-submission comparison**

## **6.1 Relevance to the four communities**

### **Scientific relevance by venue**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Venue | Scientific relevance |
| --- | --- |
| Data2AgentEx | **9/10** |
| Human-Centric AgMS | **9/10** |
| Cog2Soc | **8/10** |
| Continual Causality | **6/10** |

## **6.2 Which contribution matters most to each venue?**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Shared FactorJEPA contribution         | Cog2Soc      | Data2AgentEx | Continual Causality | Human-Centric                 |
| -------------------------------------- | ------------ | ------------ | ------------------- | ----------------------------- |
| 115,687 clips                          | 🟢 High      | 🟢 Very high | 🟡 Moderate         | 🟢 Very high                  |
| 22 Indian cities                       | 🟢 Very high | 🟢 High      | 🟡 Moderate         | 🟢 Very high                  |
| Layout, agent, interaction views       | 🟢 Very high | 🟢 Very high | 🟢 High             | 🟢 High                       |
| Progressive unfreezing                 | 🟡 Moderate  | 🟢 High      | 🟢 Very high        | 🟡 Moderate                   |
| Frozen teacher and anchored decay      | 🟡 Moderate  | 🟢 High      | 🟢 Very high        | 🟡 Moderate                   |
| Seven adaptation baselines             | 🟢 High      | 🟢 Very high | 🟢 High             | 🟢 High                       |
| Plain/RAW/Ours ablation                | 🟢 High      | 🟢 Very high | 🟢 Very high        | 🟢 High                       |
| Artificial masking robustness          | 🟢 High      | 🟢 Very high | 🟢 High             | 🟢 Very high                  |
| Temporal-coherence disadvantages       | 🟠 Concern   | 🟡 Trade-off | 🟠 Concern          | 🔴 Major concern for planning |
| Missing causal evaluation              | ⚪ Secondary  | ⚪ Secondary  | 🔴 Major gap        | ⚪ Secondary                   |
| Missing behavioural validation         | 🔴 Major gap | 🟠 Concern   | 🟡 Moderate         | 🟠 Concern                    |
| Missing retention evaluation           | ⚪ Secondary  | 🟡 Moderate  | 🔴 Major gap        | ⚪ Secondary                   |
| Missing downstream mobility evaluation | 🟠 Concern   | 🟠 Concern   | 🟡 Moderate         | 🔴 Major gap                  |

## **6.3 How FactorJEPA compares to the Step 1 acceptance bar**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Criterion                   | Accepted-paper calibration                             | FactorJEPA assessment                                                |
| --------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------- |
| Dataset scale               | P1 accepted with 30 images                             | 🟢 Considerably larger resource                                      |
| Number of model comparisons | P1 used four VLMs; P2 used three LLMs                  | 🟢 Strong                                                            |
| Controlled data comparisons | P3 compared five sampling compositions                 | 🟢 Competitive                                                       |
| Ablation                    | Several accepted papers lacked extensive ablations     | 🟢 Table 2 is a major strength                                       |
| Statistical reporting       | P3 reported 50 repeated fine-tuning experiments        | 🟡 Strong test-sample statistics but weaker training-run replication |
| Human validation            | P1 and P4 used human annotations                       | 🔴 Weaker                                                            |
| Downstream utility          | P4 showed feedback-driven agent improvement            | 🟠 Weaker                                                            |
| Preliminary contributions   | P5 demonstrated feasibility without retraining results | 🟢 More empirically complete                                         |
| Causal conceptual novelty   | P6–P8 made direct conceptual contributions             | 🔴 Weaker                                                            |
| Authentic mobility evidence | P9 and P10 evaluated real mobility outcomes            | 🟡 Strong source footage; weaker task-level outcomes                 |

**FactorJEPA is above the observed workshop acceptance bar in experimental breadth, but below the strongest matching accepted papers in several venue-specific evaluation dimensions.**

# **7. Optimal shared GPU and revision strategy**

You have one RTX PRO 6000 with 96 GB memory. Because the first three deadlines are close together, do not treat the GPU-hour budgets proposed for each venue as independent allocations.

Prioritize experiments that can improve multiple versions first.

## **7.1 Shared improvements ranked by total return**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Rank | Experiment                            | GPU hours (estimated) | Cog2Soc | Data2AgentEx | Continual | Human-Centric |
| ---- | ------------------------------------- | --------------------- | ------- | ------------ | --------- | ------------- |
| 1    | Human annotation audit                | 0–1                   | 🟢 +3   | 🟢 +3        | 🟡 +1     | 🟢 +2         |
| 2    | Density/scene-conditioned analysis    | 0–5                   | 🟢 +3   | 🟢 +2        | 🟡 +1     | 🟢 +2         |
| 3    | Source-disjoint checkpoint evaluation | 3–8                   | 🟢 +2   | 🟢 +3        | 🟢 +2     | 🟢 +2         |
| 4    | Reference-distribution retention      | 4–8                   | ⚪ 0     | 🟡 +1        | 🟢 +4     | 🟡 +1         |
| 5    | Stage-checkpoint analysis             | 4–8                   | 🟡 +1   | 🟢 +2        | 🟢 +2     | 🟡 +1         |
| 6    | Matched randomized-view control       | 12–24                 | 🟡 +1   | 🟢 +2        | 🟢 +2     | 🟡 +1         |
| 7    | Small sequential-domain pilot         | 18–30                 | ⚪ 0     | 🟡 +1        | 🟢 +3     | 🟡 +1         |
| 8    | Road-user trajectory probe            | 12–30                 | 🟢 +4   | 🟢 +3        | 🟡 +1     | 🟢 +4         |
| 9    | City-disjoint retraining              | 35–90                 | 🟢 +2   | 🟢 +3        | 🟢 +2     | 🟢 +3         |

Score gains are approximate marginal reviewer-score improvements. They overlap and should not be added mechanically.

## **7.2 Recommended execution sequence**

### **Single-GPU execution plan**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Dates | GPU hours | Focus | Work |
| --- | --- | --- | --- |
| **Nov 27–28** | 0–5 GPU h | Shared high-ROI analysis | Human audit begins; density-conditioned results; source-overlap inspection; rewrite shared sections. |
| **Nov 28–29** | 3–8 GPU h | Cog2Soc finalization | Unused-source inference if available, annotation agreement, behavioural framing, four-page compilation. |
| **Nov 29–30** | 8–16 GPU h | Continual retention experiments | Reference-domain inference and intermediate-stage checkpoint evaluation. |
| **Nov 30–Dec 1** | 12–24 GPU h | Data curation or sequential pilot | Choose randomized views or a capped sequential-domain experiment according to the highest-priority unresolved gap. |
| **Dec 1** | 0–3 GPU h | Data2AgentEx and Continual submission | Finalize tables, report all results, check anonymity, and preserve submission buffers. |
| **Dec 2–24** | 90–250 GPU h | Human-Centric extensions | Trajectory probe, source/city-disjoint validation, natural occlusion analysis and additional baselines if affordable. |

### **GPU estimate caveat**

The PDFs do not report training throughput, batch sizes, actual hardware utilization, clip-decoding throughput or wall-clock times. Consequently, the estimates above are provisional rather than benchmarked execution times.

Before starting a new training run, profile approximately 100 representative optimizer steps, including video loading, and extrapolate:

```math
T_{\mathrm{GPU}} \approx \frac{N_{\mathrm{steps}}\times t_{\mathrm{step}}}{3600}
```

Add sufficient allowance for evaluation, checkpointing, interruptions and manuscript integration. If the projected time exceeds the available budget, reduce the pilot size and clearly label the experiment as preliminary.

I would not commit to a new full 115k training run before the first three deadlines.

### **Priority if computational resources become constrained**

### **Submission experiment checklist**

Checked: 0/8

- [ ] Human annotation audit
- [ ] Density-conditioned analysis
- [ ] Source-overlap audit and unused-source evaluation
- [ ] Reference-domain retention evaluation
- [ ] Shared contribution-first rewriting
- [ ] Optional matched randomized-view pilot
- [ ] Optional sequential-domain pilot
- [ ] Human-Centric trajectory evaluation

# **8. Four-page space management**

All four manuscripts are currently four-page PDFs. References occupy much of page 4, meaning that, subject to each venue's formatting rules, there may be usable main-text capacity if references are allowed on subsequent pages.

You do not necessarily need to cut valuable experimental evidence simply to preserve a four-page total PDF when the venue permits four pages of main text excluding references. Nevertheless, the following replacement plan respects your stricter space constraint.

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Existing element                      | Action                          | New content                                                    |
| ------------------------------------- | ------------------------------- | -------------------------------------------------------------- |
| Fig. 1 scene-thumbnail strip          | Reduce                          | Human annotation validation summary                            |
| Fig. 1 three structured-view examples | 🟢 Keep                         | Essential visual explanation                                   |
| Fig. 2 density charts                 | Replace or compact              | Density and scene-conditioned evaluation                       |
| §3 implementation prose               | Compress                        | Concise reproducible parameter description                     |
| Table 1, thirteen diagnostics         | 🟢 Keep                         | Complete baseline and negative-result reporting                |
| Fig. 3 interval-width visualization   | Replace                         | Venue-relevant evidence with ordinary effect estimates and CIs |
| Table 2 ablation                      | 🟢 Keep                         | Central scientific contribution                                |
| §5 venue discussion                   | Rewrite                         | Specific research implications, not repeated results           |
| §6 limitations                        | Compress                        | Preserve all unresolved limitations                            |
| References                            | Move beyond page 4 if permitted | Preserve complete citations                                    |

**Never cut Table 2 to make space.** It is the single most important component for Data2AgentEx and also supports all three other venues.

Freeze each submitted version when its deadline passes. Subsequent improvements to the common source should apply only to drafts that are still open, unless the venue permits revisions.

# **9. Concurrent submission and double-blind risks**

This is a potentially serious non-technical issue.

You stated that:

- All four drafts use identical methods, data, results and tables.
- They have different titles and venue-specific framing.
- They disclose the other concurrent submissions in anonymous footnotes.
- The same work is a pending non-archival AAAI-27 student abstract and public arXiv preprint.

This transparency is preferable to concealing overlapping work. However, **non-archival status does not automatically establish that concurrent submission is permitted.**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Venue               | Status based on supplied descriptions                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| Human-Centric AgMS  | 🟢 Explicitly permits concurrent submissions                                                             |
| Data2AgentEx        | 🟡 Non-archival; concurrent-submission permission not explicitly established in the supplied description |
| Cog2Soc             | 🟡 Non-archival; concurrent-submission permission not explicitly established                             |
| Continual Causality | 🟠 Novel-work requirement makes eligibility clarification particularly important                         |

I have not independently verified the live submission and anonymity policies, and therefore the acceptance estimates assume eligibility. They should not be interpreted as confirmation that every concurrent submission complies with venue rules.

A generic anonymous disclosure, where permitted, would be:

This manuscript reports work concurrently submitted to other non-archival AAAI-27 workshops and bridges. Related versions include a public preprint and a non-archival student abstract. Details are available to the program chairs upon request.

If the venue specifically requires the other submission names, retain them or provide them through its designated confidential disclosure mechanism. Do not remove information required by the organizers.

Changing titles and introductions does not make these four independent research contributions. They remain venue-specific versions of substantially the same work.

# **10. Final ranking by acceptance probability**

### **Current versus conditional improved acceptance estimates**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Venue | Draft | Current draft | After successful improvements | Score |
| --- | --- | --- | --- | --- |
| **1. Data2AgentEx** | v4 | 79% | 91% | 77/100 → 85–89/100 |
| **2. Human-Centric AgMS** | v1 | 77% | 93% | 75/100 → 87–92/100 |
| **3. Cog2Soc** | v3 | 65% | 81% | 68/100 → 77–80/100 |
| **4. Continual Causality** | v5 | 46% | 72% | 60/100 → 72–76/100 |

Conditional estimates are illustrative midpoints of the improvement scenarios, not forecasts of actual reviewer decisions or unobserved experimental outcomes.

## **Single most valuable change for each submission**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Rank | Venue               | Current probability | Most valuable change                                             | Deadline   |
| ---- | ------------------- | ------------------- | ---------------------------------------------------------------- | ---------- |
| 🥇 1 | Data2AgentEx        | **79%**             | Human verification of automatic road-user and interaction labels | Dec 1      |
| 🥈 2 | Human-Centric AgMS  | **77%**             | Downstream road-user trajectory forecasting                      | Dec 25     |
| 🥉 3 | Cog2Soc             | **65%**             | Direct human-validated behavioural interaction analysis          | Nov 29 AoE |
| 4    | Continual Causality | **46%**             | Reference-distribution retention evaluation                      | Dec 1 AoE  |

## **10.1 Ranking after successful improvements**

<!-- joined-words: off --> <!-- why: verbatim ChatGPT review text, kept word for word -->
| Rank | Venue               | Current score | Conditional improved score | Conditional probability |
| ---- | ------------------- | ------------- | -------------------------- | ----------------------- |
| 🥇   | Human-Centric AgMS  | 75            | 87–92                      | 90–95%                  |
| 🥈   | Data2AgentEx        | 77            | 85–89                      | 88–93%                  |
| 🥉   | Cog2Soc             | 68            | 77–80                      | 78–83%                  |
| 4    | Continual Causality | 60            | 72–76                      | 67–76%                  |

Human-Centric overtakes Data2AgentEx in the improved scenario primarily because its deadline provides enough time to establish direct downstream utility.

# **Final PC verdict**

FactorJEPA has three research contributions that are competitive with the accepted papers in the calibration corpus:

1. **A geographically underrepresented, large-scale street-video resource.**
2. **A structured adaptation method evaluated across multiple models, baselines and data scales.**
3. **A controlled ablation identifying different roles for staged updating and structured data curation.**

The third is particularly important. The fact that RAW explains most of the predictive gain should not be concealed or treated as a failed experiment. It is an informative scientific result that reveals the more specific contribution of structured views.

Your immediate priorities should be the **100-frame, three-rater human validation and density-conditioned analysis**, followed by the retention experiment for Continual Causality. Reserve the more expensive trajectory forecasting and city-disjoint retraining experiments for Human-Centric.

I would submit all four, subject to confirmation of concurrent-submission eligibility. Data2AgentEx is currently the strongest scientific fit, while Continual Causality requires the most substantive new evidence rather than merely a more positive rewrite.
