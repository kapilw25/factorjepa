> Claude review (claude.ai document dated Nov 27, 2026, uploaded by the owner) of the [v1_Human_Centric](../../../v1_Human_Centric/0_main.pdf) draft for Human-Centric AgMS, from the judge prompt [prompt.md](../../input/prompt.md), on the 4 open-venue PDFs on GitHub main as of commit 52c2809c (PDFs unchanged since fd37d96b). Saved Fri Nov 27 2026 ~7:00 PM PST. The review judged all 4 drafts in one document: this file keeps the ranking, the shared findings and the v1 section; the other drafts' reviews are in [v3_Cog2Soc_fixes](../v3_Cog2Soc_fixes/by_claude.md), [v4_Data2AgentEx](../v4_Data2AgentEx/by_claude.md), [v5_Continual_Causality](../v5_Continual_Causality/by_claude.md). Text is word for word; only the byline line (date and author handle) was left out. Only the final review was saved, no Step 1 or Step 2 answers.

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

## v1 Human-Centric AgMS (workshop W4, due Fri Dec 25, 8:59 PM PST, about 4 weeks)

Title: "Can a World Model of Traffic Scenes Predict Crowded South Asian Streets?". Matching accepted papers from Step 1: Group D (papers 9 and 10).

### Part A. Acceptance probability

**Human-Centric AgMS: 60% as it stands, about 75–80% with improvements C1–C3. Typical AAAI-27 CV or embodied-AI workshop: 70%.**

The fit is direct. Theme A lists "world models of traffic scenes" and theme D asks for "benchmarks beyond driving metrics", which Sec 5 claims. The CFP welcomes early-stage work and negative results, and allows concurrent submissions. Group D's accepted bar was real mobility data plus one quantitative comparison: paper 9 had one before/after count on highD, paper 10 one 80/20 split. v1 clears that on rigor. The gaps are in the workshop's two adjectives. "Agentic": no mobility outcome is measured. The title asks whether the model can "predict crowded South Asian streets", but the answer is in embedding error only (Sec 6). "Human-centric": pedestrians, two-wheelers and animals are named but never analysed separately. Four weeks is enough to close both gaps.

**Most likely reasons to reject**

1. **No mobility task.** "Trajectory forecasting ... none is reported here" (Sec 6). A reviewer from autonomous driving will ask how embedding L1 relates to forecasting or planning.
2. **The human-centric content is thin.** No result is split by road-user type (pedestrian vs vehicle) or by crowding. The Sec 1 claim that road users "are hardest to predict where all three are tightly coupled" is stated, not tested.
3. **Deployment generalisation is untested.** Sec 5 argues the model suits "the many cities where such maps do not exist", but the split is at clip level with no city held out (Sec 6).

**Strongest points**

1. Theme fit, plus a deployment-friendly property: "At inference the model needs only monocular video, with no detector, HD map, or LiDAR" (Sec 5).
2. Robustness when road users are hidden: the Mask-ratio slope is 19% flatter than LoRA at 115k (Sec 4), and 6% and 2.5% flatter than RAW (Table 2). Occlusion is a safety-relevant property.
3. Real, diverse, governed data (22 cities, 3 viewpoints, 10 scene types) and an evaluation that reports where rivals lead on long-horizon coherence (Figure 3). Negative results are explicitly welcome here.

### Part B. Ten rubrics

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| Rubric | Why it matters here | Score | Evidence (page / section) | The one change that adds the most points |
| --- | --- | --- | --- | --- |
| Theme fit (A: traffic-scene world models; D: benchmarks) | The first screen | 8 | Sec 1 and Sec 5 name both | Name the themes in the last sentence of Sec 1 |
| Mobility-relevant outcome | "Agentic" means acting on predictions | 2 | Sec 6: forecasting not reported | C1 forecasting probe |
| Human-centric analysis (pedestrians, vulnerable road users) | The workshop's title | 4 | Road-user types named in the abstract and Sec 1; no per-type result | C2 split by road-user class and density |
| Real-world realism | Theme D, "real roads" | 7 | 22 cities, 480p public video, monocular, no map (Sec 2, Sec 5) | Report the viewpoint mix (drive, walk, aerial) |
| Robustness under occlusion | Safety-relevant | 7 | Mask-ratio slope (Table 2, Figure 3) | C2 split by agent pixel share |
| Long-horizon prediction (planning) | Planning needs seconds ahead | 3 | Six coherence diagnostics lost at 115k (Figure 3) | C4 horizon curve: up to what Δt FactorJEPA leads |
| Generalisation across cities and videos | Deployment claim in Sec 5 | 3 | Clip-level split, no city held out (Sec 6) | C3 video- and city-held-out evaluation |
| Statistical rigor | Workshop reviewers include ML researchers | 8 | Paired BCa intervals, 2 scales, 7 rivals | Absolute effect sizes first (S5) |
| Benchmark beyond driving metrics | Theme D asks for it; Sec 5 offers it | 6 | Sec 5: "DENSEWORLD-115k also gives a benchmark for this regime", but no protocol is defined | Define the protocol: video-level split, the 4 diagnostics, the forecasting probe |
| Claim clarity | A question title must be answered | 5 | Question title; untested coupling claim (Sec 1) | Declarative title (E, V1-5) |

**Total: 53 / 100.** On these rubrics I would put Group D paper 9 at about 45 and paper 10 at about 55. v1 is at the accepted level now. C1–C3 would add about 15 points, which would put it clearly above both.

### Part C. Improvements before Dec 25 (4 weeks)

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| Rank | Improvement | What it shows | Needs | Where it goes | Rubric gain | Scope |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | C1 image-plane forecasting probe | Answers the title with a mobility metric. From the SAM 3 tracks, predict each road user's mask centroid (or box) at +0.5, +1 and +2 s from frozen first-half embeddings pooled over its mask. Report ADE, FDE and miss rate per road-user class for Ours, RAW, plain, LoRA, frozen and full fine-tuning at 1B·10k, plus Ours vs LoRA at 115k | Feature extraction about 1–2 GPU-h per checkpoint, roughly 10–15 GPU-h; probes train in minutes; about a week of code | A new Table replacing the nine lower rows of Table 1 (Figure 3 already shows them at 115k), plus 3 sentences in Sec 4 | Mobility outcome +5, long-horizon +1, human-centric +1 | v1 now; add to v4 and v5 only if it is done before Dec 1 |
| 2 | C2 split by road-user class and density | Whether FactorJEPA's margin is larger for pedestrians, two-wheelers and animals, and in the densest, most occluded clips. Tests the Sec 1 coupling claim | Saved per-clip errors and masks; CPU. A per-class split needs per-token errors; if only per-clip errors were saved, bin clips by their dominant road-user class | Small figure in Sec 4 | Human-centric +3, occlusion +1, claim clarity +1 | All four (S2) |
| 3 | C3 video- and city-held-out evaluation | Whether gains survive unseen videos and cities, which the deployment claim needs. S1 first (inference); then, with the time available, one city-held-out retrain of Ours and LoRA at 1B·10k | S1: 8–16 GPU-h. Retrain: 2 runs at your logged 1B·10k cost | 2 sentences plus one table row group in Sec 4; shorten Sec 6 | Generalisation +4 | S1 for all; the retrain is v1 only |
| 4 | C4 horizon curve | Up to what prediction horizon FactorJEPA leads before the rivals' coherence advantage takes over. It turns "rivals lead on coherence" into a usable planning statement | The L1-vs-Δt diagnostic is already computed; plot the saved values; CPU | Small line figure next to Figure 3 | Long-horizon +2, benchmark +1 | v1, optionally all |
| 5 | S3 human check, with class labels | Label precision and recall per road-user class, validating the forecasting probe's targets | 100 frames, 3 raters | 2 rows in Sec 2 | Human-centric +1, benchmark +1 | All four |

**Cut to make room:** move the references to page 5 (about one column). Replace Table 1's lower nine rows with the forecasting table. Shorten the release paragraph (p.2) to 3 lines and point to the dataset card. Drop the Sec 5 sentence beginning "DENSEWORLD-115k also gives a benchmark" once C1 and C3 define the benchmark in Sec 4.

### Part D. Double-blind check

All shared items apply. v1-specific: footnote 1 names "the AAAI-27 Cog2Soc, Data2AgentEx, and Continual Causality bridges". This workshop allows concurrent submissions, so the disclosure is safe here. By Dec 25 the other three decisions (Dec 10, 11 and 13) will be known, so update the footnote then.

### Part E. Tone audit (v1-specific rows; shared rows S-E1 to S-E9 also apply)

<!-- joined-words: off --> <!-- why: verbatim Claude review text, kept word for word -->
| # | Page / section | Quoted original (exact) | Problem | Rewrite |
| --- | --- | --- | --- | --- |
| V1-1 | p.1, abstract | "Competing methods still lead on temporal coherence. We discuss what such a traffic-scene world model can and cannot yet offer mobility agents." | Negative close ("still", "cannot yet") | See E.3 |
| V1-2 | p.4, Sec 5 | "Our results support that use in part." | Self-critical opener for the venue section | Our results support two parts of that use directly and identify the third as the next target. |
| V1-3 | p.4, Sec 5 | "Temporal coherence over longer rollouts, which matters for planning several seconds ahead, is where the rivals still lead." | Negative | The third part is temporal coherence over longer rollouts, which matters for planning several seconds ahead and where the rivals lead today. |
| V1-4 | p.1, Sec 1 | "and they are hardest to predict where all three are tightly coupled." | Not negative, but an untested claim stated as fact (it breaks the no-untested-claims rule) | and we expect them to be hardest to predict where all three are tightly coupled. (Or test it with C2 and cite the result.) |
| V1-5 | p.1, title | "Can a World Model of Traffic Scenes Predict Crowded South Asian Streets?" | A question the paper answers only in embedding space | Adapting a Traffic-Scene World Model to Crowded South Asian Streets: Better One-Step Prediction and Robustness to Occlusion |

**E.2 The three rewrites that matter most:** V1-2 and V1-3 together (the venue section is where a mobility reviewer decides relevance), V1-5 (the title sets the expectation that C1 must meet), and S-E5 (it states the two-property finding a planner needs).

**E.3 Rewritten abstract (207 words; the original is 221):**

> We adapt a video world model to the crowded, mixed traffic that mobility agents must anticipate. DENSEWORLD-115k holds 115,687 clips (276 hours) cut from 714 public YouTube videos of 22 Indian cities, released as reconstructible metadata: streets where pedestrians, two-wheelers, auto-rickshaws, carts, and animals share one right of way, often partly hidden, while world models of traffic scenes are trained and evaluated mostly on lower-density, lane-structured traffic. FactorJEPA adapts V-JEPA 2.1 first on clips of the street layout, then on the road users, then on their interactions. Against seven adaptation methods under identical clips and budgets, one fixed configuration predicts masked and future content better at both 2B and 1B scale, and after full 115k-clip training it separates from the stronger of the two rivals trained at that scale on all four diagnostics of predictive structure, by 13.9 to 43.3 widths of the paired 95% confidence interval. An ablation on raw clips shows that the staged adaptation carries most of this gain, while the layout, agent, and interaction clips make prediction more robust when more of the scene is hidden, the situation of occluded road users. Competing methods lead on temporal coherence, which we discuss as the next step toward planning for mobility agents.
