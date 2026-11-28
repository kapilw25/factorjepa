> Claude review (claude.ai document dated Nov 27, 2026, uploaded by the owner) of the [v3_Cog2Soc_fixes](../../../v3_Cog2Soc_fixes/0_main.pdf) draft for Cog2Soc, from the judge prompt [prompt.md](../../input/prompt.md), on the 4 open-venue PDFs on GitHub main as of commit 52c2809c (PDFs unchanged since fd37d96b). Saved Fri Nov 27 2026 ~6:47 PM PST. The review judged all 4 drafts in one document: this file keeps the ranking, the shared findings and the v3 section; the other drafts' reviews are in [v1_Human_Centric](../v1_Human_Centric/by_claude.md), [v4_Data2AgentEx](../v4_Data2AgentEx/by_claude.md), [v5_Continual_Causality](../v5_Continual_Causality/by_claude.md). Text is word for word; only the byline line (date and author handle) was left out. Only the final review was saved, no Step 1 or Step 2 answers.

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

## v3 Cog2Soc (bridge, due Sun Nov 29, 23:59 AoE, about 2.4 days)

Title: "Layout, Then Road Users, Then Interactions: Adapting a Video World Model to Dense South Asian Street Behaviour". Matching accepted papers from Step 1: Group A (papers 1 and 2).

### Part A. Acceptance probability

**Cog2Soc: 55%. Typical AAAI-27 CV or embodied-AI workshop: 65%.**

The draft ticks the three things the CFP requires of every paper. It explains the human phenomenon (p.1, "The human phenomenon"), states its approach, and connects to another community (social-force and agent-based models, Sec 1 and Sec 5). It also covers responsible use (p.2). Its rigor is far above the Group A papers: paper 1 was accepted with 30 author-scored images and one run, while this draft has paired 95% BCa intervals, 7 rivals and 2 model scales. The risk is the audience. Psychologists, sociologists and computational social scientists will look for a measured behaviour, and every result here is an embedding-space diagnostic (Sec 6). The paper's own analogy, building the model "in the same order" as agent-based models (Sec 1), is not tested, and Table 2 credits the staged schedule rather than the order. A non-archival bridge that scores "suitability for discussion" softens this. For a CV workshop the controlled comparison would count for more and the social framing for less.

**Most likely reasons to reject**

1. **No behaviour is measured.** "Every diagnostic is measured in the frozen V-JEPA embedding space or with a linear probe on it" (Sec 6). The phenomenon is defined by yielding, gap acceptance and passing distance (Sec 1), but none of these is measured.
2. **The framing analogy is untested and partly contradicted.** Table 2: on raw clips the same schedule gives Future-frame L1 0.497 vs Ours 0.496 at 2B (not significant) and 0.587 vs 0.590 at 1B, where Ours is worse. The stage order is never ablated, so the title's "Then ... Then" is a design choice, not a finding.
3. **The cross-cultural claim rests on one country and a density count.** "All footage here is Indian" (Sec 6). Figure 2 compares agent counts on 6 of 10 scene types against a reference whose size and viewpoints are not described.

**Strongest points**

1. The required elements are explicit and well written: the phenomenon (p.1), the connection to social-force modelling (Sec 5, Helbing and Molnár 1995), responsible use (p.2).
2. Candid, controlled evidence: 13 diagnostics with wins and losses (Figure 3), video-level resampling (Table 2), and a clear statement of where rivals lead.
3. A dense, non-Western resource for behaviour modellers: 115,687 clips from 22 cities, released in a copyright-respecting way (p.2).

### Part B. Ten rubrics

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| Rubric | Why it matters here | Score | Evidence (page / section) | The one change that adds the most points |
| --- | --- | --- | --- | --- |
| Human phenomenon explained | A CFP requirement for every paper | 8 | p.1 "The human phenomenon": negotiation, gap acceptance, small lateral distances | Cite 2–3 transport studies of heterogeneous, non-lane traffic for these behaviours |
| Connection to another community | A CFP requirement; reviewers come from five fields | 6 | Sec 5 links to agent-based and social-force models; one citation (Helbing and Molnár 1995) | Add how an ABM modeller would use the model (e.g. as a learned prior to compare with hand-set rules), with 2–3 ABM or CSS references |
| Behavioural grounding | The topics list empirical grounding and calibration | 3 | Sec 6: all diagnostics in embedding space; no yielding or trajectory result | S3 human check with who-yields-to-whom judgements, reported as agreement with the interaction rule |
| Cross-cultural evaluation | Named in the topics and in the v3 abstract | 5 | Figure 2 density comparison only; "all footage here is Indian" (Sec 6) | S4: evaluate the checkpoints on the Western reference clips |
| Claim matches evidence | Interdisciplinary reviewers judge the stated idea | 5 | Title and Sec 1 credit the order; Table 2 credits the schedule | Retitle and reword Sec 1 and Sec 5 to state what Table 2 shows |
| Statistical rigor and candour | Replication and uncertainty are listed topics | 8 | Paired BCa intervals (Sec 4), video resampling (Table 2), Spearman ρ ≥ 0.895 across scales; one training run per configuration (Sec 6) | Report absolute effect sizes first, widths second (S5) |
| Data resource and responsible use | Responsible use is a listed topic | 8 | Release and consent paragraph (p.2) | Blur faces in Figure 1 and state the corpus composition by city |
| Clarity for a non-ML audience | Mixed audience, scored on clarity of substance | 4 | Table 1 lists 13 diagnostics (TCC τ, arrow-of-time, exposure bias) with no behavioural meaning; "interval widths" as the main unit | Give each of the 4 main diagnostics a one-line behavioural reading; keep the 9 coherence rows in Figure 3 only |
| Discussion value | An explicit review criterion | 7 | Sec 5 asks "which kind of structure (layout, individuals, or interactions) a learned predictor needs" | Close Sec 5 with two open questions for the room |
| Scope of claims | Overclaims cost trust with social scientists | 6 | Sec 5 says results "already support the first part of that use"; Sec 2 says a Western-calibrated model "would meet denser scenes", which Figure 2 supports for the two measured axes only | Tie each discussion claim to its table or figure |

**Total: 60 / 100.** On these rubrics I would put Group A paper 1 at about 50 (strong phenomenon, weak rigor) and paper 2 at about 60. v3 is at the accepted level, stronger on rigor and resources, weaker on measuring the human behaviour itself.

### Part C. Improvements before Nov 29 AoE

All fit in the column freed by moving the references to page 5 (see Shared findings).

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| Rank | Improvement | What it shows | Needs | Where it goes | Rubric gain | Scope |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | S3 human check, including who yields to whom | Whether the interaction rule matches human judgements of yielding, which is behavioural grounding | 100 frames, 3 raters, about half a day each; κ computation | 2-row table plus 2 sentences at the end of Sec 2 | Behavioural grounding +3, connection +1 | All four drafts (shared Sec 2) |
| 2 | Retitle and reword Sec 1 and Sec 5 to the evidence | States the actual finding: staged adaptation improves one-step prediction, and the structured clips add robustness under occlusion | Writing, 2 hours | Title, last paragraph of Sec 1, first sentences of Sec 5. Example title: "Adapting a Video World Model to Dense South Asian Street Behaviour: Staged Training and Robustness to Occlusion" | Claim matches evidence +3, scope +1 | v3 only |
| 3 | S2 density split | Whether the margin over the best rival grows with crowding, the premise of the phenomenon | Saved per-clip errors and masks; CPU, a few hours | Small figure next to Figure 2 | Behavioural grounding +1, cross-cultural +1, discussion +1 | All four (shared Sec 4) |
| 4 | S4 Western-reference evaluation | Cross-cultural transfer: what adaptation to Indian streets does to prediction of sparser Western streets | Inference, about 4–8 GPU-h | 4-column table in Sec 5 | Cross-cultural +2 | Venue-framed; data shared with v5 and v1 |
| 5 | Plain-language diagnostics | Lets non-ML reviewers read Table 1 | Writing, 1 hour | Replace the nine lower rows of Table 1 with one sentence pointing to Figure 3, and add one behavioural reading per main diagnostic | Clarity +3 | v3 only; this changes a shared table, so do it in v3's copy only |

A stage-order ablation (one 1B·10k run with all clip types mixed from the start) would test the title's claim directly. Start it only if your logs show a 1B·10k FactorJEPA run finishes in under about 24 GPU-h; otherwise item 2 is the honest substitute.

**Cut to make room, if needed:** first move the references to page 5. Then cut the lower nine rows of Table 1, since Figure 3 shows all 13 diagnostics at 115k and the 10k values can go to an appendix if the bridge allows one. Then shorten the SlotFormer and DriveWorld sentence in Sec 3.

### Part D. Double-blind check

All shared items apply (see Shared findings, Part D). v3-specific: footnote 1 names "the AAAI-27 Data2AgentEx and Continual Causality bridges and the Human-Centric AgMS workshop". This does not identify the authors, but a reviewer who sits on two of these programmes can link the submissions. If the CFP allows it, keep the venue names in the OpenReview disclosure field rather than the PDF.

### Part E. Tone audit (v3-specific rows; shared rows S-E1 to S-E9 also apply)

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| # | Page / section | Quoted original (exact) | Problem | Rewrite |
| --- | --- | --- | --- | --- |
| V3-1 | p.1, abstract | "Competing methods still lead on temporal coherence. We discuss what this offers the modelling and cross-cultural evaluation of collective behaviour." | Negative close ("still lead"), then a vague promise | See E.3: the loss is kept and framed as the next target |
| V3-2 | p.4, Sec 5 | "Temporal coherence over longer rollouts, which a simulation needs, is where the rivals still lead." | Negative | Temporal coherence over longer rollouts, which a simulation needs, is where the rivals lead today, and it is the next target for staged adaptation. |
| V3-3 | p.4, Sec 5 | "and our results already support the first part of that use: one-step prediction of crowded scenes improves at every scale and data size tested" | Defensive ("already ... the first part") | Our results support the first step of that use directly: one-step prediction of crowded scenes improves at every scale and data size tested |
| V3-4 | p.1, Sec 1 | "Agent-based models are typically built in layers: the environment, then the individuals, then their interactions. We adapt V-JEPA 2.1 in the same order." | Not negative, but reads as a tested claim, and Table 2 does not test it | Agent-based models are typically built in layers: the environment, then the individuals, then their interactions. We adapt V-JEPA 2.1 in the same order and measure which part of that adaptation carries the gain (Table 2). |

**E.2 The three rewrites that matter most:** S-E8 (the Scope paragraph is the last thing a social scientist reads), V3-4 (it aligns the framing with Table 2 before a reviewer finds the gap), and S-E5 (it turns the coherence loss into a clean two-property finding).

**E.3 Rewritten abstract (209 words, the same as the original):**

> We contribute DENSEWORLD-115k, 115,687 clips (276 hours) from 714 public YouTube videos of 22 Indian cities, released as reconstructible metadata, and FactorJEPA, a curriculum that adapts the video world model V-JEPA 2.1 to dense street behaviour in the order agent-based models are built: first the street layout, then the road users, then their interactions. On these streets pedestrians, two-wheelers, auto-rickshaws, carts, and animals share one right of way and coordinate by negotiation rather than lane rules, often while partly hidden, unlike the sparser, lane-structured, mostly Western scenes on which such models are often calibrated. Against seven adaptation methods under identical clips and budgets, one fixed FactorJEPA configuration predicts masked and future content better at 2B and 1B scale, and after full 115k-clip training it separates from the stronger of the two rivals trained at that scale on all four diagnostics of predictive structure, by 13.9 to 43.3 widths of the paired 95% confidence interval. An ablation on raw clips shows that the staged adaptation carries most of this gain, while the layout, agent, and interaction clips make prediction more robust when more of the scene is hidden. Competing methods lead on temporal coherence, the next target. We discuss what this offers the modelling and cross-cultural evaluation of collective behaviour.
