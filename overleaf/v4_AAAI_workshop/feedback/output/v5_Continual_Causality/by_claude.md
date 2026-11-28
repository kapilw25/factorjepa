> Claude review (claude.ai document dated Nov 27, 2026, uploaded by the owner) of the [v5_Continual_Causality](../../../v5_Continual_Causality/0_main.pdf) draft for Continual Causality, from the judge prompt [prompt.md](../../input/prompt.md), on the 4 open-venue PDFs on GitHub main as of commit 52c2809c (PDFs unchanged since fd37d96b). Saved Fri Nov 27 2026 ~6:47 PM PST. The review judged all 4 drafts in one document: this file keeps the ranking, the shared findings and the v5 section; the other drafts' reviews are in [v1_Human_Centric](../v1_Human_Centric/by_claude.md), [v3_Cog2Soc_fixes](../v3_Cog2Soc_fixes/by_claude.md), [v4_Data2AgentEx](../v4_Data2AgentEx/by_claude.md). Text is word for word; only the byline line (date and author handle) was left out. Only the final review was saved, no Step 1 or Step 2 answers.

# FactorJEPA venue reviews: Cog2Soc, Data2AgentEx, Continual Causality, Human-Centric AgMS

## Ranking and the one change per venue

Data2AgentEx is the most likely acceptance and Continual Causality the least. The core work is the same in all four drafts; what differs is how well each venue's own question is answered. Probabilities are my judgment against the Step 1 calibration sheet, not a forecast from review data.

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| Rank | Draft and venue | Acceptance now | Typical CV / embodied-AI workshop | Single most valuable change before its deadline | Scope of that change |
| --- | --- | --- | --- | --- | --- |
| 1 | v4 Data2AgentEx (Dec 1, 9 PM PST) | 70% | 70% | S1 video-held-out evaluation from the saved 10k checkpoints (about 3 GPU-h for the key 1B models). It answers the bridge's named "training-test overlap" theme with a result. The S3 human check is already planned and fills the verification theme | All four drafts (shared Sec 4) |
| 2 | v1 Human-Centric AgMS (Dec 25) | 60% (75–80% after C1–C3) | 70% | C1 image-plane forecasting probe on frozen embeddings, using the SAM 3 tracks (about 10–15 GPU-h, about a week of code). It gives the title question a mobility answer | v1 (others only if done before Dec 1) |
| 3 | v3 Cog2Soc (Nov 29 AoE) | 55% | 65% | S3 human check reported as agreement between the interaction rule and human who-yields-to-whom judgements. It is the only behavioural grounding achievable in 2.4 days. Pair it with the free retitle (C rank 2) | All four (shared Sec 2) |
| 4 | v5 Continual Causality (Dec 1 AoE) | 35% | 65% | C1 retention and forgetting table on the Western reference clips and a general-video subset (inference, about 8–15 GPU-h). It replaces "retention ... is the next test" and makes the paper a continual-learning result | v5 framing; S4 data reused by v3 and v1 |

**Two things to do first, today, for all four drafts:** move the references to page 5, which frees about a column per draft (see Shared findings); and apply S5 (budget sentence, "Forward L1" rename, 2.5% rounding, a description of Figure 2's reference set).

**Suggested order of work on the GPU box**

1. Nov 28: S2 density split (CPU) and S1 for the 1B checkpoints (about 3 GPU-h); raters start S3.
2. Nov 29, before AoE: submit v3 with S3, S2 and the reframed title, Sec 1 and Sec 5.
3. Nov 30–Dec 1: S1 for the rest, S4 and C1 retention (about 15–25 GPU-h combined); submit v4 by 9 PM PST and v5 by Dec 1 AoE, with C2 if time allows.
4. Dec 2–25: C1 forecasting probe, the city-held-out retrain and the horizon curve for v1.

## Shared findings (apply to all four drafts)

Sections 2–4 (except the first paragraphs), Section 6, Tables 1–2 and Figures 1–3 are identical in all four drafts. Everything in this section therefore changes all four at once. The venue sections below add only what differs.

### Free space you are not using

In all four drafts the main text ends in the left column of page 4 and the references fill the right column (p.4). Every venue excludes references from the 4-page limit, so moving them to page 5 frees about one column (roughly half a page) per draft. Most additions below fit there without cutting anything.

### Evidence checks against Tables 1–2 and Figure 3

- **Numbers are consistent.** I re-derived 2.7% (0.573 vs 0.589), 19% (0.047 vs 0.058), 0.036 and 0.020 (Table 2), 13.9–43.3 and 8.3–146.5 widths (Figure 3), and the six red coherence rows. One rounding: the 1B Mask-ratio gain is 0.0528 → 0.0515, which is 2.5%, written as "2%" (Sec 4, "Where the gain comes from").
- **Name versus evidence.** The name and the v3 title credit the layout → road users → interactions clips, but Table 2 credits the staged schedule. RAW vs Ours on Future-frame L1 is 0.497 vs 0.496 at 2B (the one difference that does not exclude zero) and 0.587 vs 0.590 at 1B, where Ours is worse. The clips win only on Mask-ratio slope and lose on Motion cosine. The order of the stages is never ablated. Every reviewer will see this.
- **Budget ambiguity.** Sec 3 starts FactorJEPA "from a checkpoint already adapted to the corpus by plain continued pretraining", while Sec 4 says all methods have "identical clips, masking, optimization steps". Say in one sentence whether the rivals start from the same checkpoint. If they do not, FactorJEPA had extra steps.
- **Headline unit.** With 23,106 test clips the intervals are narrow, so 43.3 widths can be a small absolute effect (2.7% lower L1). Lead with absolute or relative differences and keep the widths as the significance statement.
- **Leakage.** The split is at clip level. Resampling 127 videos (Table 2) widens the intervals but does not remove shared-video leakage.
- **Training variance.** "each configuration was trained once" (Sec 6), so the intervals cover test sampling only.
- **115k rivals.** Only LoRA and the frozen backbone were trained there (Table 1 note).
- **Figure 2.** It compares 6 of the 10 scene types. The Western reference's size and viewpoint mix, and how agents were counted in it, are not stated; DENSEWORLD mixes drive-through, walk-through and aerial clips, while BDD100K and nuScenes are dash-cam.
- **Metric name.** "Causal L1" invites a causal reading; v5 has to disclaim it (Sec 5). "Forward L1" says what it measures.
- **References.** There are 10 references (11 in v3). Beyond one social-force citation in v3 there is no continual-learning, data-centric, mobility or behaviour-modelling literature.

### Shared improvements (each helps every venue)

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| ID | What it shows | Needs | Where it goes |
| --- | --- | --- | --- |
| S1 | A video-held-out test. The 10k models were trained on a stratified subset, so clips from source videos outside it give a video-level (possibly city-level) test with no retraining | Saved 10k checkpoints; inference only. 8 methods × 2 scales on about 2,000 clips at an assumed 0.5–1 GPU-h per checkpoint is 8–16 GPU-h; the 1B Ours, RAW, plain and LoRA alone take about 3 GPU-h. Scale from your own eval logs | One small table or 2 sentences in Sec 4; turns the leakage caveat in Sec 6 into a result |
| S2 | A density split. Bin test clips by agents per clip and by agent pixel share (from the 86,831 masks) and plot the FactorJEPA-minus-best-rival margin per bin. Tests the premise that the method helps in crowded, occluded scenes | Saved per-clip errors (the bootstrap implies they exist); CPU, a few hours | A small figure in the freed column |
| S3 | A human check of the labels (planned: 100 frames, 3 raters). Road-user precision and recall of Grounding DINO + SAM 3, and agreement on interaction pairs and on who yields to whom (Fleiss' κ) | Raters' time, about half a day each | A 2-row table plus 2 sentences in Sec 2 |
| S4 | Transfer and retention on the Western reference clips already used for Figure 2. Do the adapted checkpoints still predict sparse, lane-structured traffic? | Inference only, about 4–8 GPU-h for the 1B checkpoints | A small table in the discussion section; framed per venue (cross-cultural, retention, or deployment) |
| S5 | Writing fixes: one budget sentence, "Causal L1" renamed "Forward L1", 2.5% instead of 2%, Figure 2's reference set described | 1 hour | In place |

### Part D, double-blind check (shared)

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| # | Item (where) | Risk | Anonymous replacement |
| --- | --- | --- | --- |
| 1 | "https://huggingface.co/datasets/anonymousML123/denseworld-115k" (p.1, under the abstract) | The account name is anonymous, but the dataset card, its BibTeX, links to GitHub or arXiv, and the commit history can name the authors | Before submitting, strip the citation block and any arXiv or GitHub links from the card and check the commit author fields. Restore them after decisions |
| 2 | Footnote 1: "An earlier version is a non-archival AAAI-27 student abstract and a public preprint." (p.1) | Tells reviewers a searchable preprint exists; with distinctive names a search finds it with author names | Keep the disclosure the CFP requires, but word it neutrally, e.g. "Related non-archival versions of this work exist; details are disclosed to the chairs." Use OpenReview's disclosure field where available |
| 3 | The names DENSEWORLD-115k and FactorJEPA (title, abstract, body) | Identical to the public preprint, so one search de-anonymizes | AAAI venues generally allow arXiv, so accepting this risk is usual. The only full fix is placeholder names in the submission, which costs readability; I would keep the names and apply fix 2 |
| 4 | PDF metadata | Checked: Producer "pdfTeX-1.40.25", Creator "TeX", no Title, Author or XMP fields, no embedded link URIs | No change needed |
| 5 | File names "factorJEPA\_v3\_Cog2Soc.pdf" and so on | A file name can survive in some systems and ties the four submissions together | Upload as "paper.pdf" |
| 6 | Acknowledgements and self-citations | None found | No change needed |
| 7 | Figure 1, top strip (p.2) | Not an author leak, but it shows identifiable faces of people who did not consent (Sec 2 says so) | Blur faces in the figure; it also supports the responsible-use paragraph |

### Part E, shared tone rows (these sentences appear in all four drafts)

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| # | Page / section | Quoted original (exact) | Problem | Rewrite |
| --- | --- | --- | --- | --- |
| S-E1 | p.2, Sec 2 | "These two axes are the ones measured here; occlusion and interaction pressure are the natural next axes." | Negative: leads with what was not measured | On both measured axes DENSEWORLD is denser on every compared scene type, and occlusion and interaction pressure extend the same comparison. |
| S-E2 | p.2, Sec 3 | "The first stage unfreezes the first 4 of the encoder's 48 blocks and the later stages the first 8, following surgical fine-tuning (Lee et al. 2023); the predictor trains throughout against a frozen teacher, selective projection decay (Tian, Huang, and Kira 2024) keeps the weights near their starting point, and the fixed configuration splits its epochs 30/25/45 across the stages." | Engineering only: a recipe with no stated idea | Three anchors let the encoder absorb new structure without leaving its pretrained solution: it opens gradually (the first 4 of 48 blocks in stage one, the first 8 afterwards, following surgical fine-tuning (Lee et al. 2023)), the predictor learns against a frozen teacher, and selective projection decay (Tian, Huang, and Kira 2024) keeps weights near their start. One configuration, with epochs split 30/25/45 across the stages, produces every result. |
| S-E3 | p.3, Sec 4 Protocol | "Test clips are held out at the clip level (1,825 in the 10k regime, 23,106 in the full regime), so a source video can contribute clips to both training and test (Section 6)." | Criticism placed in the protocol with no mitigation | Test clips are held out at the clip level (1,825 in the 10k regime, 23,106 in the full regime); because a source video can contribute clips to both sides, Table 2 also resamples the 127 source videos, and Section 6 sets out video- and city-level splits. |
| S-E4 | p.3, Sec 4 Findings | "Full fine-tuning, which leads on Motion cosine at 10k, was not trained at 115k, so whether more data closes that gap remains open." | Negative framing of an open question | At 115k the trained rivals are LoRA and the frozen backbone; full fine-tuning, which leads on Motion cosine at 10k, was not trained at that scale and is the natural next rival there. |
| S-E5 | p.3, Sec 4 Findings | "FactorJEPA's gain is in one-step prediction of crowded scenes; on the coherence of long predicted futures, the rivals still lead." | Ends the results on a loss | The results separate two properties: FactorJEPA improves one-step prediction of crowded scenes, while the rivals lead on the coherence of long predicted futures, which marks long-horizon coherence as the next target. |
| S-E6 | p.3–4, Sec 4 | "They cost Motion cosine at both scales and a little prediction error: 0.001 in Causal L1 at 2B and about 0.003 in Future-frame and Causal L1 at 1B." | Self-critical phrasing ("a little prediction error") | This robustness has a measured price: Motion cosine is lower at both scales, Causal L1 rises by 0.001 at 2B, and Future-frame and Causal L1 rise by about 0.003 at 1B. |
| S-E7 | p.4, Sec 4 | "Most of the gain therefore does not depend on the automatic labels, and what the labels buy is a specific property: robustness to missing evidence." | Framed as a negative about the labels | Most of the gain therefore holds without relying on the automatic labels, and the labels add one specific, measured property: robustness to missing evidence. |
| S-E8 | p.4, Sec 6 | "Trajectory forecasting, a comparison with a calibrated social-force or agent-based model, and a check of the interaction clips against human judgements of who yields to whom are the next tests; none is reported here, and the interaction clips are chosen by a proximity and relative-motion rule." | Negative list ("none is reported here") | The paired diagnostics give one common scale for all methods, and three external tests build on them: trajectory forecasting, a comparison with a calibrated social-force or agent-based model, and a check of the rule-based interaction clips (proximity and relative motion) against human judgements of who yields to whom. None of these is reported in this paper. |
| S-E9 | p.4, Sec 6 | "The clip-level split lets a source video contribute to both training and test, and no city is held out. Video-level and city-level splits, dense-traffic regions beyond India (all footage here is Indian), and repeated training runs (each configuration was trained once) are the natural extensions." | Negative and self-critical, read as a weakness list | Generalisation tests follow the same plan: splits by source video and by city (the current split is at clip level, with no city held out), dense-traffic regions beyond India (all footage here is Indian), and repeated training runs (each configuration here was trained once). |

If you run S1 or S3, replace S-E3, S-E8 and S-E9 with the results rather than rewording them.

## v5 Continual Causality (bridge, novel-work track, due Dec 1 AoE = Wed Dec 2, 4:59 AM PST, about 4.4 days)

Title: "Staged Adaptation of a Pretrained Video World Model to Dense South Asian Streets: What It Gains and What It Costs". Matching accepted papers from Step 1: Group C (papers 6, 7 and 8, PMLR 208).

### Part A. Acceptance probability

**Continual Causality: 35%. Typical AAAI-27 CV or embodied-AI workshop: 65%.**

The bridge's question is how learning systems "continually build and maintain accurate causal world models when the world is non-stationary and only partly observed". v5 speaks well to two of its parts, a sharp distribution shift and partial observability (Mask-ratio slope, Table 2). It has nothing on the causal part and little on the continual part. Sec 5 states "We make no causal claim". The abstract defers retention: "retention on the original distribution is the next test". There is one adaptation step, not a sequence, and the 10 references include no continual-learning or causal work. Even the continual self-supervised learning rival is uncited. Evidence is not the problem: Group C accepted papers 7 and 8 with no experiments at all. What Group C rewarded was explicit bridging of continual learning and causality, and v5 bridges only one side. There is also a policy risk: the track requires that "works must be original", while the paper discloses a public preprint, a student abstract and three concurrent submissions.

**Most likely reasons to reject**

1. **No causal content.** "We make no causal claim: Causal L1 names the direction of prediction, from the first half of a clip to the second, not an identified cause." (Sec 5). The metric's name is the only causal element.
2. **Not yet continual learning.** One adaptation from general video to DENSEWORLD, with no forgetting measured (the abstract's last clause) and no sequence of shifts. A reviewer can call it domain adaptation.
3. **No positioning in either literature.** There are no references on catastrophic forgetting, continual pretraining, continual self-supervised learning, shortcut learning or causal representation learning. Papers 6 and 7 from this bridge's own 2023 volume are directly relevant and uncited.

**Strongest points**

1. A clean partial-observability result: the structured clips flatten the Mask-ratio slope (6% at 2B and 2.5% at 1B over RAW, Table 2; 19% flatter than LoRA at 115k, Sec 4). That is "prediction degrades more slowly as more of the scene is hidden".
2. Continual self-supervised learning is among the seven rivals, and in Table 1 it is never the strongest rival on the four predictive-structure rows (those are LoRA, DoRA or full fine-tuning). It is the strongest on several coherence rows.
3. Honest scoping: the no-causal-claim sentence, the stated costs, and "the update schedule matters" (Sec 5), backed by plain vs RAW in Table 2.

### Part B. Ten rubrics

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| Rubric | Why it matters here | Score | Evidence (page / section) | The one change that adds the most points |
| --- | --- | --- | --- | --- |
| Causal content | Half of the bridge's name; Group C papers 6 and 7 were accepted for it | 2 | Sec 5 "We make no causal claim" | An interventional-view test of shortcut reliance (C2 below) |
| Continual-learning framing | The other half; "non-stationary" in the CFP | 4 | One shift; Sec 2 "Adapting to it is a continual-learning problem under partial observability" asserted, not measured | Measure retention (C1) |
| Retention and forgetting | The first question a CL reviewer asks | 1 | Abstract: "retention on the original distribution is the next test" | C1 retention table |
| Partial observability | Named in the CFP | 7 | Mask-ratio slope, Table 2 and Figure 3 | S2 split by agent pixel share, as a proxy for occlusion |
| CL baselines | Reviewers expect CL methods | 6 | Continual SSL among the rivals (Sec 4); no EWC or replay | Cite and describe the rival properly; replay is the obvious missing baseline |
| Related-work positioning | Group C rewarded explicit bridging | 2 | No CL or causal references | 6–8 references in the freed column (C3) |
| Statistical rigor | Benchmarks and datasets are a listed topic | 8 | Paired BCa intervals, video resampling, Spearman ρ ≥ 0.895 | Absolute effect sizes first (S5) |
| Originality and policy fit | "Works must be original" | 6 | Footnote 1 discloses a preprint, a student abstract and three concurrent venues | Ask the chairs now (C5) |
| Claim calibration | This community values precise claims | 9 | No-causal-claim sentence; costs stated in Sec 5 | Rename "Causal L1" to "Forward L1" so the disclaimer is no longer needed |
| Discussion value | Bridges select for discussion | 6 | Sec 5: "which part of a changing scene a world model has to relearn" | Turn that into a measured question with C2 |

**Total: 51 / 100.** On these rubrics I would put Group C paper 6 at about 60 (explicit continual-plus-causal bridging, toy evidence), paper 7 at about 45 and paper 8 at about 40. v5 sits inside that range, but its points come from rigor, while the accepted papers' points came from the two topical rows where v5 scores lowest.

### Part C. Improvements before Dec 1 AoE

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| Rank | Improvement | What it shows | Needs | Where it goes | Rubric gain | Scope |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | C1 retention and forgetting | Stability versus plasticity: how much each method's four diagnostics degrade, relative to base V-JEPA 2.1, on (a) the Western reference clips (S4) and (b) a 1–2k clip subset of general video close to pretraining (e.g. Kinetics-400 validation) | Inference only on all 1B checkpoints plus the base model, about 8–15 GPU-h; dataset download time | A 5-row table in Sec 5, replacing "retention ... is the next test" | Retention +5, continual framing +2, discussion +1 | v5; S4 also serves v3 and v1 |
| 2 | C2 interventional-view shortcut test | Whether adaptation makes road-user prediction lean on background context: compare error on road-user tokens with the background present vs suppressed, and on layout tokens with road users present vs blurred, for each checkpoint. These are interventions on the input, which answers the CFP's "shortcut learning and confounding" without claiming causal identification | Inference on the 1,825 test clips × 4 views × about 6 checkpoints, about 4–8 GPU-h, plus a day of code | A short paragraph and a 2-column table in Sec 5 | Causal content +3, discussion +2 | v5 only |
| 3 | C3 positioning paragraph and rename | Places the work in CL and causal-learning literature. Candidates: catastrophic forgetting (Kirkpatrick et al. 2017), continued pretraining (Gururangan et al. 2020), continual self-supervised learning (e.g. Fini et al. 2022), shortcut learning (Geirhos et al. 2020), causal representation learning (Schölkopf et al. 2021), and this bridge's own Ostapenko et al. and Lesort (PMLR 208, 2023). Rename "Causal L1" to "Forward L1" | Writing, half a day | Freed column, end of Sec 1 | Related work +4, causal content +1 | References v5 only; rename in all four |
| 4 | S2 split by agent pixel share | Whether the Mask-ratio benefit is largest where road users are most hidden, a direct partial-observability result | CPU, a few hours | Small figure | Partial observability +2 | All four |
| 5 | C5 policy check | Avoids desk rejection | One email to the chairs today | n/a | Originality +3 | v5 |

A true sequential stream (for example, adapt on one group of cities and then another, and measure backward transfer for Ours, continual SSL and LoRA) would make this a continual-learning paper. It needs about 6 training runs at 1B·10k, so attempt it only if your logs show one run takes under about 12 GPU-h.

**Cut to make room:** move the references to page 5. Shorten the Sec 2 "The shift" paragraph, which repeats the abstract's first sentence (about 4 lines). Cut Sec 5's last clause once C1 reports retention.

### Part D. Double-blind check

All shared items apply. v5-specific: the CFP notes "Earlier editions were published in PMLR volumes 208 (2023) and 268 (2024)" and requires originality. Disclosing the preprint is honest but may trigger a policy query, so put the disclosure where the chairs ask for it, not only in a footnote. Footnote 1 names "the AAAI-27 Cog2Soc and Data2AgentEx bridges and the Human-Centric AgMS workshop".

### Part E. Tone audit (v5-specific rows; shared rows S-E1 to S-E9 also apply)

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| # | Page / section | Quoted original (exact) | Problem | Rewrite |
| --- | --- | --- | --- | --- |
| V5-1 | p.4, Sec 5 | "We make no causal claim: Causal L1 names the direction of prediction, from the first half of a clip to the second, not an identified cause." | Self-critical disclaimer about the bridge's own keyword | Causal L1 measures prediction in the forward direction, from the first half of a clip to the second; it is a test of temporal direction, and identifying causes lies outside the tests reported here. (Better: rename the metric "Forward L1" everywhere and drop the sentence.) |
| V5-2 | p.4, Sec 5 | "Both gains come with costs a continual learner has to weigh, in motion readout and in the coherence of long predicted futures, where the rivals still lead." | Negative | Both gains come with measured trade-offs that frame the stability-plasticity choice for a continual learner: motion readout, and the coherence of long predicted futures, where the rivals lead. |
| V5-3 | p.4, Sec 5 | "Whether the adapted model keeps its accuracy on the original distribution is the next test" | Negative, and the bridge's core question is left open | Retention on the original distribution is the next measurement. (Replace with the C1 result if you run it.) |
| V5-4 | p.1, abstract | "Competing methods still lead on temporal coherence, and retention on the original distribution is the next test." | Ends the abstract on two open items | See E.3 |
| V5-5 | p.1, Sec 2 | "Adapting to it is a continual-learning problem under partial observability: the model has to absorb a new distribution while predicting road users it can only partly see." | Not negative; it states the framing the paper must then measure | Keep it, and point to the retention table once C1 exists |

**E.2 The three rewrites that matter most:** V5-1 (it removes the sentence a causality reviewer will quote in a rejection), V5-2 (it recasts the costs in the community's own stability-plasticity terms), and S-E9 (the generalisation plan is what a CL reviewer reads last).

**E.3 Rewritten abstract (227 words; the original is 237):**

> We show that how a pretrained video world model is updated for a new world matters, and measure what each part of the update gains and costs. The new world is a sharp shift: DENSEWORLD-115k, 115,687 clips (276 hours) from 714 public YouTube videos of 22 Indian cities, released as reconstructible metadata, brings denser scenes, different road users, and many of them only partly visible. FactorJEPA adapts V-JEPA 2.1, pretrained on general video, in stages that follow the structure of a scene (the street layout, then the road users, then their interactions), unfreezing more of the encoder after the first stage while a frozen teacher and a decay toward the starting weights keep it anchored. Against seven adaptation methods, continual self-supervised learning among them, under identical clips and budgets, one fixed configuration predicts masked and future content better at both 2B and 1B scale, and after full 115k-clip training it separates from the stronger of the two rivals trained at that scale on all four diagnostics of predictive structure, by 13.9 to 43.3 widths of the paired 95% confidence interval. An ablation on raw clips shows that the staged schedule carries most of the gain, while the structured clips make prediction degrade more slowly as more of the scene is hidden. Competing methods lead on temporal coherence, and retention on the original distribution is the next test.

The opening claim is supported by Table 2 (plain vs RAW). If C1 runs, replace the last clause with the retention result.
