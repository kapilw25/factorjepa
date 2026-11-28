> Claude review (claude.ai document dated Nov 27, 2026, uploaded by the owner) of the [v4_Data2AgentEx](../../../v4_Data2AgentEx/0_main.pdf) draft for Data2AgentEx, from the judge prompt [prompt.md](../../input/prompt.md), on the 4 open-venue PDFs on GitHub main as of commit 52c2809c (PDFs unchanged since fd37d96b). Saved Fri Nov 27 2026 ~7:00 PM PST. The review judged all 4 drafts in one document: this file keeps the ranking, the shared findings and the v4 section; the other drafts' reviews are in [v1_Human_Centric](../v1_Human_Centric/by_claude.md), [v3_Cog2Soc_fixes](../v3_Cog2Soc_fixes/by_claude.md), [v5_Continual_Causality](../v5_Continual_Causality/by_claude.md). Text is word for word; only the byline line (date and author handle) was left out. Only the final review was saved, no Step 1 or Step 2 answers.

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

## v4 Data2AgentEx (bridge, due Tue Dec 1, 9:00 PM PST, about 4 days)

Title: "From Street Video to Training Signal: Automatic Layout, Agent, and Interaction Clips for Adapting a Video World Model". Matching accepted papers from Step 1: Group B (papers 3, 4 and 5).

### Part A. Acceptance probability

**Data2AgentEx: 70%. Typical AAAI-27 CV or embodied-AI workshop: 70%.**

This is the best fit of the four. The bridge asks how domain data becomes experience for learning agents, and v4 is written as exactly that: video to curated views to a curriculum (Sec 1, "The data problem"). Sec 5 answers the bridge's themes in turn: learning utility (the Table 2 ladder), training-test overlap (video resampling) and conditions for reuse (the rebuild script). Its evidence clears the Group B bar. Paper 3 was accepted with seeds and standard deviations, paper 4 with one run and a dataset, paper 5 with no numbers. The risk sits in the bridge's own third theme, verification. The labels that create the curated views are unchecked ("a human check ... is the next step", Sec 5). The ablation also shows the curated views add little to one-step prediction, so a data-centric reviewer may ask whether the data curation is the contribution at all.

**Most likely reasons to reject**

1. **Verification is the venue's theme, and the labels are unverified.** Sec 5: "Whether the automatic labels are valid: they come from a detector, a tracker, and a proximity rule; a human check of missed and false road users and of who yields to whom is the next step."
2. **The curated data's learning utility is narrow.** Table 2: RAW vs Ours Future-frame L1 is 0.497 vs 0.496 at 2B (not significant) and 0.587 vs 0.590 at 1B. The views win on Mask-ratio slope (6% and 2.5%) and lose on Motion cosine (0.133 → 0.116 at 2B). The title puts the clips first.
3. **Overlap is bounded, not removed.** Sec 4: "a source video can contribute clips to both training and test". The bridge names "training-test overlap" explicitly, and the Sec 5 opener, "Each question the bridge asks of agentic data has an answer here", overstates how fully it is answered.

**Strongest points**

1. A controlled learning-utility ladder (plain → RAW → Ours, Table 2) separates the schedule from the data, with intervals under both clip and video resampling. This is the bridge's "learning utility" theme in one table.
2. Failure analysis is visible and welcome here: Figure 3 shows six red coherence diagnostics, and the motion cost is stated in the abstract.
3. Governed reuse: metadata plus a deterministic rebuild script, CC BY 4.0, a removal route for rights holders, and a consent statement (p.2). This covers the "conditions for reuse" theme.

### Part B. Ten rubrics

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| Rubric | Why it matters here | Score | Evidence (page / section) | The one change that adds the most points |
| --- | --- | --- | --- | --- |
| Theme fit: domain data to agent experience | The bridge's central question | 9 | Sec 1 "The data problem"; Sec 5 maps each theme | Name which of the five themes each result serves in one line of Sec 1 |
| Learning utility, controlled | Theme 4 | 8 | Table 2 ladder; video resampling (127 videos) | S2 density split: shows where the curated views pay off |
| Verification of automatic labels | Theme 3 | 2 | Sec 5 "is the next step" | S3 human check: precision and recall of road users, κ on interaction pairs |
| Training-test overlap | Theme 4 names it | 4 | Clip-level split (Sec 4); resampling only (Table 2) | S1 video-held-out evaluation |
| Controlled evaluation of transfer | Theme 4 | 3 | No transfer test; all diagnostics on DENSEWORLD | S4 on the Western reference clips |
| Reuse and governance | Theme 5 | 8 | Rebuild script, licence, removal route, consent (p.2) | State expected rebuild loss: how many of the 714 videos are still online today |
| Resource documentation | Datasets are welcome, so they are judged as datasets | 6 | Totals and scene types given (Sec 2, Figure 1); no per-city, per-viewpoint or label-coverage breakdown beyond 86,831 clips | Add a composition table from the metadata |
| Failure analysis and negative results | Explicitly welcome | 8 | Figure 3 red rows; motion cost in the abstract | Say which clip types lose most on coherence, from saved results |
| Clarity to a cross-disciplinary audience | A review criterion | 6 | Table 1's 13 diagnostics with superscript rivals; "interval widths" unit | Lead with absolute differences (S5) |
| Statistical rigor | Reviewers weigh empirical substance | 7 | Paired BCa intervals; one training run per configuration (Sec 6) | State training-seed variance as a limitation alongside S1 |

**Total: 61 / 100.** On these rubrics I would put Group B paper 3 at about 65, paper 4 at about 50 and paper 5 at about 35. v4 is above the median of accepted papers. The verification row is the one an expert reviewer at this bridge will weigh most.

### Part C. Improvements before Dec 1, 9:00 PM PST

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| Rank | Improvement | What it shows | Needs | Where it goes | Rubric gain | Scope |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | S3 human check | Label precision and recall, and agreement on interaction pairs and yielding. It turns "next step" into a measured verification | 100 frames, 3 raters; κ | Replace the Sec 5 sentence on label validity with the result, plus a 2-row table | Verification +4, documentation +1 | All four (shared Sec 2) |
| 2 | S1 video-held-out evaluation | Utility without shared-video leakage | Saved 10k checkpoints; about 3 GPU-h for the 1B Ours, RAW, plain and LoRA, 8–16 GPU-h for all | 2 sentences or a small table in Sec 4; shorten Sec 6 accordingly | Overlap +4 | All four |
| 3 | S4 transfer to the Western reference clips | A controlled transfer test, the theme's own wording | Inference, about 4–8 GPU-h | Short table in Sec 5 | Transfer +3 | Data shared; framing v4-specific |
| 4 | Composition table | Corpus by city, viewpoint (drive, walk, aerial), scene type, and label coverage (86,831 of 115,687 clips, number of interaction pairs) | Metadata; CPU, 1–2 hours | Freed column, under Sec 2 | Documentation +2 | v4, optional for the others |
| 5 | S2 density split | Where the curated views help: whether the Mask-ratio benefit grows with agents per clip | Saved per-clip errors; CPU | Small figure in Sec 4 | Learning utility +1, failure analysis +1 | All four |

A per-view ablation (one 1B·10k run without the interaction stage) would measure each view's utility, which is the most data-centric experiment possible here. Start it only if a 1B·10k run takes under about 48 GPU-h on your box.

**Cut to make room:** move the references to page 5. Sec 6 repeats two Sec 5 points (label validity, clip-level overlap); keep them in one place and save about 6 lines. Drop the Sec 5 opener "Each question the bridge asks of agentic data has an answer here." It overclaims, and the themes are already named.

### Part D. Double-blind check

All shared items apply. v4-specific: footnote 1 names "the AAAI-27 Cog2Soc and Continual Causality bridges and the Human-Centric AgMS workshop"; the same advice as for v3 applies. Sec 5 says "the bridge", which is fine and contains no identity.

### Part E. Tone audit (v4-specific rows; shared rows S-E1 to S-E9 also apply)

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| # | Page / section | Quoted original (exact) | Problem | Rewrite |
| --- | --- | --- | --- | --- |
| V4-1 | p.4, Sec 5 | "Whether the automatic labels are valid: they come from a detector, a tracker, and a proximity rule; a human check of missed and false road users and of who yields to whom is the next step." | Negative, on the venue's key theme | Label validity: the labels come from a detector, a tracker, and a proximity rule, and the ladder shows that most of the gain holds without relying on them; a human check of missed and false road users and of who yields to whom is the next step. (Replace with the S3 result if you run it.) |
| V4-2 | p.4, Sec 5 | "Conditions for reuse: the release ships metadata and a rebuild script rather than video, so a video its creator removes can no longer be rebuilt." | Reads as a loss rather than a governance feature | Conditions for reuse: the release ships metadata and a rebuild script rather than video, so creators keep control: a video its creator removes can no longer be rebuilt. |
| V4-3 | p.1, abstract | "at a cost in motion readout. Competing methods still lead on temporal coherence." | Ends the abstract on two losses | See E.3: both facts kept, framed as the measured trade-off and the next target |
| V4-4 | p.1, Sec 1 | "Such video is unstructured: it does not mark the road layout, the individual road users, or who is interacting with whom, so a model trained on raw clips meets all of it at once." | Fine as a problem statement; no change needed | Keep |

**E.2 The three rewrites that matter most:** V4-1 (it sits on the bridge's verification theme), S-E3 (overlap is the bridge's named concern, and the rewrite shows the mitigation in the protocol), and S-E7 (it states the learning utility of the labels positively and exactly).

**E.3 Rewritten abstract (213 words; the original is 224):**

> We turn public street video, abundant domain data for agents that move among people, into staged training experience without human labels. DENSEWORLD-115k holds 115,687 clips (276 hours) from 714 public YouTube videos of 22 Indian cities, released as reconstructible metadata. For 86,831 of them, a detector, a tracker, and a proximity rule cut each clip into three curated views: the street layout with road users blurred, the road users with the background suppressed, and crops around interacting pairs. FactorJEPA trains the video world model V-JEPA 2.1 on these views as a three-stage curriculum. Against seven adaptation methods under identical clips and budgets, one fixed configuration predicts masked and future content better at both 2B and 1B scale, and after full 115k-clip training it separates from the stronger of the two rivals trained at that scale on all four diagnostics of predictive structure, by 13.9 to 43.3 widths of the paired 95% confidence interval. A controlled ablation that runs the same stages on raw clips measures the learning utility of each part: the staged schedule carries most of the gain, and the curated views make prediction more robust when more of the scene is hidden, at a cost in motion readout. Competing methods lead on temporal coherence, a clear target for the next curriculum.
