> Fresh Claude judge review of the [v5_Continual_Causality](../../../v5_Continual_Causality/0_main.pdf) draft for Continual Causality, from the judge prompt [prompt.md](../../input/prompt.md). Judge: Claude (fresh helper, no chat context), run Fri Nov 27 2026, inputs = GitHub main at commit 52c2809c. Saved per venue Fri Nov 27 2026 ~7:00 PM PST from the full answers in [step1.md](../v3_open_venues/by_claude_fresh/step1.md), [step2.md](../v3_open_venues/by_claude_fresh/step2.md) and [step3.md](../v3_open_venues/by_claude_fresh/step3.md), which stay unchanged. This file keeps the v5 paragraph of Step 2 and, from Step 3, the shared findings, the v5 section and the ranking; Step 1 (the calibration sheet, shared by all 4) stays in step1.md. The other drafts' reviews are in [v1_Human_Centric](../v1_Human_Centric/by_claude_fresh.md), [v3_Cog2Soc_fixes](../v3_Cog2Soc_fixes/by_claude_fresh.md), [v4_Data2AgentEx](../v4_Data2AgentEx/by_claude_fresh.md). Text is word for word.

# STEP 2: What each venue's reviewers will look for first

**Venue 3, Continual Causality (bridge, novel-work track, due Dec 1 AoE).** Reviewers will look first for the causal and the continual parts, named explicitly: what is non-stationary, what is the causal structure or mechanism that the model should keep or learn, and what shift, confounder or shortcut is at play. Calibrated on Group C, they reward a crisp concept (independent mechanisms, local spurious features, evolving causal knowledge) and a formal statement over large tables; continual pretraining of a foundation model is in scope, but only if the paper argues it in causal and continual-learning language (forgetting, plasticity, mechanism shift, shortcut). They will also check originality, since "works must be original", and how the concurrent student abstract and preprint relate.

# STEP 3: Judgment of the 4 venue drafts

The four PDFs share Sections 2, 3, 4 and 6, Tables 1 and 2 and Figures 1 to 3 word for word; they differ in title, abstract, Section 1, Section 5 and footnote 1. To avoid saying the same thing four times, the shared problems are listed once as SH1 to SH10 and cited by number inside each draft's Parts A to E.

GPU-hour figures below are my rough estimates (the drafts report no training cost): about 6 to 12 GPU h for one 1B·10k training arm, 12 to 24 GPU h for one 2B·10k arm, 60 to 120 GPU h for one 1B·115k arm, and under 1 GPU h to embed or score the 1,825 test clips with one checkpoint. Scale them from your own training logs.

## Shared findings (all 4 drafts)

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
| # | Where | Issue | Why a strict reviewer cares |
|---|---|---|---|
| SH1 | p4, Table 2 and "Where the gain comes from" | RAW beats Ours on Future-frame L1 and Causal L1 at 1B (0.587 vs 0.590, 0.611 vs 0.614) and on Causal L1 at 2B (0.528 vs 0.529); the caption says these differences exclude zero (one exception). The scene-structured clips buy a 6% (2B) and 2% (1B) flatter Mask-ratio slope and cost Motion cosine (2B 0.133 to 0.116, 1B 0.060 to 0.048) | The component named in every title (layout, then road users, then interactions) is not what wins; reviewers will ask why the method is not RAW |
| SH2 | p2 Sec 3; p3 Protocol | FactorJEPA starts "From a checkpoint already adapted to the corpus by plain continued pretraining"; rivals run "under identical clips, masking, optimization steps, and splits", but their starting checkpoint is never stated, nor whether the plain checkpoint saw test clips | If rivals start from the original V-JEPA 2.1, FactorJEPA had more total training; if plain saw test clips, every number is contaminated |
| SH3 | p2 Sec 3; p4 Sec 6 | The predictor "trains throughout against a frozen teacher" and "Every diagnostic is measured in the frozen V-JEPA embedding space" | If the scoring space is the same frozen teacher FactorJEPA trains against while some rivals use a moving teacher, the L1 diagnostics favour FactorJEPA by construction; say which encoder gives the targets for every method |
| SH4 | p3 Protocol; Fig 3; Sec 6 | Clip-level split ("a source video can contribute clips to both training and test"); source-video resampling only for Table 2 at 10k; the 115k separations (13.9 to 43.3 widths) are over 23,106 clips with no stated clustering; "each configuration was trained once" | Layout clips can profit from seeing the same streets at test, which targets the novelty itself; the 115k widths likely shrink under video clustering; no seed variance |
| SH5 | Abstract; p3 Findings | Effects are given as "widths of the paired 95% confidence interval", a unit that grows with test-set size; the raw effects are a 2.7% lower Future-frame L1 and a 19% flatter Mask-ratio slope (p3 to p4) | Non-ML reviewers cannot read it; ML reviewers read it as inflation |
| SH6 | p3 Diagnostics; References | Nine of 13 diagnostics are named only (Table 1 below the rule); "action top-1" has no action labels anywhere in the data section; "clip motion" for Motion cosine is undefined; "Future-frame L1" scores V-JEPA mask blocks, not future frames. No citation for LP-FT, the continual self-supervised rival, TCC, arrow of time or the BCa bootstrap, nor for India road datasets such as IDD | Undefined metrics cannot be judged; uncited rivals look unreproducible; "sample this regime rarely" ignores existing India driving data |
| SH7 | p2 Fig 1 and Fig 2; p3 Fig 3 | Fig 1 bottom is "an illustration": panel (b) shows semantic layout labels (metro viaduct, zebra crossing, curb) and (c) flow arrows and coupling edges that the Sec 3 pipeline does not produce. Fig 2 compares 6 of the 10 scene types of Fig 1, with no clip counts or error bars, and mixes walk-through and aerial views against a vehicle-mounted reference. Fig 3 draws within-interval rival leads (pace -0.1×, action top-1 -0.6×) red like significant losses, draws 1.0× yellow though one width is the green threshold, and says "vJEPA2.1" | Reviewers may read Fig 1 as the model's input; Fig 2 invites a cherry-picking and viewpoint question; Fig 3's legend is inconsistent |
| SH8 | p4 layout | In all 4 drafts the main text ends in the left column of page 4; references fill the rest. All 4 venues exclude references from the limit | About half a page (one column) is free for new evidence before anything must be cut |
| SH9 | p4 Table 2 caption | "Every RAW-minus-plain and Ours-minus-RAW difference excludes zero ... except Future-frame L1 at 2B" does not say which of the two differences is the exception | Ambiguous significance statement on the key ablation |
| SH10 | p1 Sec 2; p3 Protocol | "stratified 10k-clip set" has no strata; 86,831 of 115,687 clips have masks with no reason given for the other 28,856 | Undocumented selection, a direct question at a data venue |

Shared strengths: Table 1 compares against the strongest of seven rivals per row, which is conservative; rankings replicate across scales (Spearman ρ ≥ 0.895, p3); losses are shown in full (Fig 3 has 8 red bars out of 13); release is metadata-only with a takedown path (p2). On statistics every draft is far above all ten Step 1 papers (only paper 3 had repeated runs; none reported a confidence interval or a significance test), so each decision will turn on venue fit and on SH1 to SH4.

## Draft v5: Continual Causality (due Dec 1 AoE, about 4 to 5 days)

Title: "Staged Adaptation of a Pretrained Video World Model to Dense South Asian Streets: What It Gains and What It Costs"

### PART A. Acceptance probability

**Continual Causality: 30%. Typical AAAI-27 computer-vision or embodied-AI workshop: 50%.**

The bridge asks how systems "continually build and maintain accurate causal world models when the world is non-stationary and only partially observed". The draft has the non-stationary and partially observed parts: a shift of traffic culture (p1 "The shift."), a continual self-supervised rival, and a partial-observability reading of the Mask-ratio slope. It has neither of the other two: it states outright "We make no causal claim" (Sec 5, p4), and it does not measure the continual-learning quantity every CL reviewer looks for first, retention on the earlier distribution ("retention on the original distribution is the next test", abstract). Group C was accepted with no or toy experiments because each paper made a causal concept precise (independent mechanisms, local spurious features, evolving causal knowledge; Step 1 pp. 34 to 46). This draft is the reverse: strong numbers, no causal or continual concept. The title's "What It Gains and What It Costs" is a good CL frame, but the costs reported are temporal coherence and motion readout, not forgetting.

Three things most likely to cause rejection:
1. No causal content: "We make no causal claim: Causal L1 names the direction of prediction, from the first half of a clip to the second, not an identified cause." (Sec 5)
2. No retention or forgetting measurement and a single shift step; reviewers will call it domain adaptation, not continual learning.
3. No CL or causal literature: the reference list has no continual-learning, continual SSL or causality work, and the continual SSL rival is unnamed and uncited (SH6).

Three strongest points:
1. Continual pretraining of a foundation model is a listed topic, and the paper compares against continual self-supervised learning under identical budgets (p3 Protocol).
2. The ablation separates the update schedule from the update data (Table 2), a clean "how vs what" result for continual updating.
3. Partial observability is measured (Mask-ratio slope, 19% flatter than LoRA at 115k), and costs are reported honestly (title, Sec 5, Fig 3).

### PART B. Ten rubrics

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
| Rubric | Why it matters here | Score | Evidence from the paper (page/section) | The one change that adds the most points |
|---|---|---|---|---|
| 1. Causal content | Defines the bridge | 2 | Sec 5: "We make no causal claim"; a metric named "Causal L1" | Test-time interventions: blur road users or suppress background on test clips and measure each model's error change |
| 2. Continual-learning evidence (retention, forgetting) | First thing CL reviewers check | 3 | Abstract: retention "is the next test" | Score all saved checkpoints on old-world clips (BDD100K, nuScenes reference; a general-video set) |
| 3. Non-stationarity and partial observability | Call topics | 6 | Fig 2 density shift; Mask-ratio slope in Table 2 | Plot error against hidden fraction 0.3 to 0.9 for plain, RAW, Ours instead of a slope only |
| 4. Conceptual framing | Group C rewards concepts | 3 | p1 "a continual-learning problem under partial observability"; no formal statement | One paragraph: layout, agents and interactions as separately shifting mechanisms; the curriculum adapts them in turn |
| 5. Grounding in CL and causal literature | Positioning | 2 | References: no CL, continual SSL or causal work | Cite the continual SSL method used, LP-FT, regularisation-based CL, sparse mechanism shift and independent mechanisms |
| 6. Evidence and statistics | Substance | 7 | 7 rivals, 2 scales, paired BCa; single runs; SH4 | Video-clustered intervals at 115k |
| 7. Honest stability and plasticity trade-off | CL values trade-offs | 7 | Title; Sec 5 "Both gains come with costs" | Make forgetting one of the measured costs |
| 8. Originality and concurrent disclosure | "works must be original" | 6 | Footnote 1 discloses 3 venues, a student abstract and a preprint | State in one line what is new in this version (retention test, CL framing) |
| 9. Clarity | Standard | 6 | Clear shift framing; "Causal L1" invites misreading at this venue; SH5, SH6 | Rename Causal L1 to "forward L1" in all drafts and drop the disclaimer |
| 10. Discussion value at the bridge | Review criterion | 5 | Sec 5 gives two results | Pose the question "which mechanism must be relearned when traffic culture changes", tied to Table 2 |

**Total: 47 / 100.** On the same rubrics the Group C papers score about 74 (paper 6, IM assumption with a toy experiment), about 68 (paper 7, local spurious features, formal definition) and about 62 (paper 8, NERL vision). The draft is far stronger on rubric 6 and far weaker on rubrics 1, 2, 4 and 5. With Part C items 1 to 4 it would reach about 62.

### PART C. Improvements before the deadline (about 4 to 5 days)

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
| Rank | Improvement and what it shows | What it needs | Where it goes (4-page budget) | Rubric gain | One venue or all |
|---|---|---|---|---|---|
| 1 | Retention test: score every saved checkpoint (frozen, plain, LoRA, DoRA, full fine-tuning, LP-FT, Auto-RGN, continual SSL, RAW, Ours; 2B and 1B at 10k, the three 1B·115k models) on the BDD100K and nuScenes reference clips of Fig 2 and on a few thousand general-video clips, with the four predictive diagnostics. Shows forgetting against gain, the CL trade-off | Inference only, about 6 to 12 GPU h | New compact Table 3 (gain on DENSEWORLD vs change on old data per method) in the free half page; replace "retention ... is the next test" in the abstract and Sec 5 | R2 +4, R7 +1, R10 +1 | Run once; framed here as retention, in v3 as cross-cultural |
| 2 | Interventional sensitivity: on test clips, blur all road users (layout only) or suppress the background (road users only) with the existing mask pipeline, and measure each model's error change on the untouched region. Shows how much each model relies on layout vs road users (shortcut reliance). State that this is an intervention on inputs, not causal identification | Masks for test clips exist for most clips; inference about 6 to 10 GPU h | 2 columns in Table 3 and 3 lines in Sec 5 replacing "We make no causal claim" | R1 +3, R4 +1 | Here |
| 3 | Concept paragraph: the scene as three mechanisms (layout, road users, interactions) that shift by different amounts between traffic cultures; the curriculum as adapting one mechanism at a time, stated as a hypothesis the interventions test; rename Causal L1 to "forward L1" | Writing, half a day | Sec 1 "Our approach." (+5 lines); rename in shared Sec 4, Tables, Fig 3 | R4 +2, R9 +1, R1 +1 | Rename for all; concept here |
| 4 | Literature and rival definitions: cite and name the continual SSL method, LP-FT, a regularisation-based CL method and independent mechanisms | Writing, 2 h | References; one Protocol sentence (shared) | R5 +3 | Citations for all; CL ones here |
| 5 | State what is new relative to the student abstract and confirm its archival status | Writing, 30 min | Footnote 1 | R8 +1 | Here |

Cut to make room: shorten "Release and responsible use" to two sentences in this draft only (about 4 lines); merge Fig 2 into one panel (about 10 lines); merge Sec 6 into Sec 5 (about 4 lines).

### PART D. Tone audit

Rows T2 to T9, T11 and T12 of the v3 table apply word for word (shared text). Rows specific to this draft:

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
| # | Page/section | Quoted original (exact) | Problem | Rewrite |
|---|---|---|---|---|
| C1 | p1 Abstract | "Competing methods still lead on temporal coherence, and retention on the original distribution is the next test." | Negative close on the key CL gap | "Competing methods lead on temporal coherence; retention on the original distribution is the next test." |
| C2 | p4 Sec 5 | "We make no causal claim: Causal L1 names the direction of prediction, from the first half of a clip to the second, not an identified cause." | Self-critical and fatal at a causality venue | "Causal L1 scores forward prediction, from the first half of a clip to the second; it measures temporal direction, not an identified cause." (better: rename the metric so no sentence is needed) |
| C3 | p4 Sec 5 | "Both gains come with costs a continual learner has to weigh, in motion readout and in the coherence of long predicted futures, where the rivals still lead." | Negative ("still") | "Both gains come with measured costs that a continual learner can weigh: motion readout and the coherence of long predicted futures, where the rivals lead." |
| C4 | p4 Sec 5 | "Whether the adapted model keeps its accuracy on the original distribution is the next test, and the layout, agent, and interaction clips offer a way to ask which part of a changing scene a world model has to relearn." | Negative (core CL result missing) | "The same diagnostics can next measure whether the adapted model keeps its accuracy on the original distribution, and the layout, agent, and interaction clips offer a way to ask which part of a changing scene a world model has to relearn." |

The 3 rewrites that matter most: C2 (it tells a causality bridge the paper has no causal content), C4 (it tells a CL bridge the CL measurement is missing; Part C item 1 should replace it with a result), T9 of v3 (it reads as "the structured data did not matter"; the rewrite states what it buys).

Rewritten abstract (226 words; current 237):

> We show that when a pretrained video world model is updated for a new world, the update schedule carries most of the gain, and scene-structured training clips make prediction degrade more slowly as more of the scene is hidden. The new world is a sharp change of traffic culture: denser scenes, different road users, and many of them only partly visible. We adapt V-JEPA 2.1, pretrained on general video, to DENSEWORLD-115k, 115,687 clips (276 hours) from 714 public YouTube videos of 22 Indian cities, released as reconstructible metadata. FactorJEPA adapts the model in stages that follow the structure of a scene, first the street layout, then the road users, then their interactions, unfreezing more of the encoder after the first stage while a frozen teacher and a decay toward the starting weights keep it anchored. Against seven adaptation methods, continual self-supervised learning among them, under identical clips and budgets, one fixed configuration predicts masked and future content better at both 2B and 1B scale, and after full 115k-clip training it separates from the stronger of the two rivals trained at that scale on all four diagnostics of predictive structure, by 13.9 to 43.3 widths of the paired 95% confidence interval. An ablation on raw clips separates the schedule from the clips. Competing methods lead on temporal coherence; retention on the original distribution is the next test.

### PART E. Double-blind check

Same items and replacements as the v3 table. Differences: footnote 1 names Cog2Soc, Data2AgentEx and Human-Centric AgMS. This venue states "works must be original", so the earlier-version footnote matters more here: keep it, but confirm with the chairs that a non-archival 4-page version of a work with a pending student abstract and a public preprint is eligible, and if past AAAI student abstracts were archival, say so precisely.

## Ranking by acceptance probability and the single most valuable change

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
| Rank | Draft and venue | Acceptance now | Single most valuable change before its deadline |
|---|---|---|---|
| 1 | v4, Data2AgentEx (Dec 1) | 60% | Retrain plain, RAW and Ours at 1B·10k on a video-disjoint split (about 18 to 36 GPU h) and put the result in Table 2 and Sec 5, so the bridge's "training-test overlap" question gets a measured answer |
| 2 | v1, Human-Centric AgMS (Dec 25) | 50% | Add a road-user forecasting probe (occupancy and path entry 1 to 2 s ahead, frozen embeddings, paired intervals) on a video-disjoint split, so the paper carries a mobility metric |
| 3 | v3, Cog2Soc (Nov 29) | 40% | Split the saved per-clip results by road users per clip (CPU only) and lead Sec 5 with how the gain changes with crowding, together with the 100-frame, 3-rater label check |
| 4 | v5, Continual Causality (Dec 1) | 30% | Score all saved checkpoints on the BDD100K and nuScenes reference clips and a general-video set (inference, about 6 to 12 GPU h) to report retention against gain, and replace "We make no causal claim" with the interventional test of Part C item 2 |

Changes that help all four at once (shared sections): state rival starting checkpoints and evaluation targets (SH2, SH3), source-video clustered intervals at 115k (SH4), plain deltas next to interval widths (SH5), one-line diagnostic definitions and the missing citations (SH6), real input frames in Fig 1 and a consistent Fig 3 legend (SH7), and the human label check. Because these edit shared files, do them first, before the Cog2Soc deadline, so every later venue inherits them.
