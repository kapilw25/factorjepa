> Fresh Claude judge review of the [v4_Data2AgentEx](../../../v4_Data2AgentEx/0_main.pdf) draft for Data2AgentEx, from the judge prompt [prompt.md](../../input/prompt.md). Judge: Claude (fresh helper, no chat context), run Fri Nov 27 2026, inputs = GitHub main at commit 52c2809c. Saved per venue Fri Nov 27 2026 ~6:47 PM PST from the full answers in [step1.md](../v3_open_venues/by_claude_fresh/step1.md), [step2.md](../v3_open_venues/by_claude_fresh/step2.md) and [step3.md](../v3_open_venues/by_claude_fresh/step3.md), which stay unchanged. This file keeps the v4 paragraph of Step 2 and, from Step 3, the shared findings, the v4 section and the ranking; Step 1 (the calibration sheet, shared by all 4) stays in step1.md. The other drafts' reviews are in [v1_Human_Centric](../v1_Human_Centric/by_claude_fresh.md), [v3_Cog2Soc_fixes](../v3_Cog2Soc_fixes/by_claude_fresh.md), [v5_Continual_Causality](../v5_Continual_Causality/by_claude_fresh.md). Text is word for word.

# STEP 2: What each venue's reviewers will look for first

**Venue 2, Data2AgentEx (bridge, due Tue Dec 1, 9:00 PM PST).** Reviewers will first look for the data-to-experience pipeline: what domain data was turned into what kind of experience, through which selection, curation or curriculum step, and how its lineage and quality were controlled. The theme "learning utility and transfer" names curricula, selection bias, training-test overlap and controlled evaluation of transfer, so they will check for a leak-free split, a matched control that isolates the data choice from the training recipe, and evidence that the curated experience transfers. Calibrated on Group B, a released dataset, benchmark or failure analysis carries weight; uncertainty in automatic labels (learned evaluators, evaluation shortcuts) and conditions for reuse of sensor data of people (consent, licences) will also be read early.

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

## Draft v4: Data2AgentEx (due Tue Dec 1, 9:00 PM PST, about 4 to 5 days)

Title: "From Street Video to Training Signal: Automatic Layout, Agent, and Interaction Clips for Adapting a Video World Model"

### PART A. Acceptance probability

**Data2AgentEx: 60%. Typical AAAI-27 computer-vision or embodied-AI workshop: 55%.**

This is the best fit of the four. The bridge asks how domain data becomes "grounded, informative, and responsibly governed experience", and the draft answers in its own first lines: "Public street video is abundant domain data ... We turn it into staged training experience without human labels." The pipeline (detector, tracker, proximity rule, three curated views, three-stage curriculum) is the contribution, and Table 2 is exactly the "controlled evaluation of transfer" the call names: it prices the schedule and the curated views separately and reports that the views cost motion readout. Section 5 walks through the bridge's questions one by one (learning utility, label validity, training-test overlap, conditions for reuse). Compared with Group B, it has a larger resource than paper 4 and far more evidence than paper 5. What keeps it from a clear accept: the theme it cites most directly, training-test overlap, is where the split is weakest (SH4); the label-validity question is answered with "the next step"; and the "agent" in agent learning is only motivation, since no agent acts or learns a policy.

Three things most likely to cause rejection:
1. Training-test overlap (SH4): clip-level split on a bridge that names "training-test overlap"; video resampling widens intervals but does not remove leakage, and layout clips could memorise streets seen at test.
2. Unverified automatic labels and undocumented selection: "a human check of missed and false road users and of who yields to whom is the next step" (Sec 5); the 28,856 clips without masks and the "stratified" 10k set are unexplained (SH10).
3. Weak agent link: "Agents that move among people, from delivery robots to driver assistants, need world models of crowded scenes" (p1) is the only agent content; every evaluation is an embedding diagnostic.

Three strongest points:
1. A data-to-experience pipeline with a controlled utility ablation (Table 2), including a mixed result, which a data-centric audience values.
2. A sizeable resource with governance: 115,687 clips, masks and interaction pairs for 86,831 clips, reconstructible metadata, CC BY 4.0 for metadata, takedown path (p1 bullets, p2 Release).
3. Section 5 maps the results to the bridge's own questions, which makes the discussion value obvious to reviewers.

### PART B. Ten rubrics

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
| Rubric | Why it matters here | Score | Evidence from the paper (page/section) | The one change that adds the most points |
|---|---|---|---|---|
| 1. Clarity of the pipeline from data to experience | The bridge's core question | 7 | Abstract; p1 "Our approach."; Sec 3 mixes 70/30 and 70/15/15; Fig 1 bottom is an illustration, not real inputs (SH7) | Replace Fig 1 bottom with real frames of each view as the model sees it, with view counts |
| 2. Learning utility, controlled | Theme 4 | 7 | Table 2 ladder at 2 scales; views flatten slope 6% and 2%, cost motion | Split the ladder by road users per clip (saved results) |
| 3. Training-test overlap and selection bias | Named in theme 4 | 4 | Clip-level split (p3); video resampling only in Table 2; SH10 | Retrain plain, RAW and Ours at 1B·10k on a video-disjoint split |
| 4. Verification of automatic labels | Theme 3 | 3 | Sec 5: human check "is the next step" | Report the 100-frame, 3-rater check |
| 5. Lineage, governance, reuse | Themes 2 and 5 | 7 | p2 Release; p1 bullet "with automatic road-user masks and interaction pairs" | A small data card table: per-city and per-viewpoint counts, masked share, label statistics |
| 6. Relevance to agent learning | The bridge pairs data with agents | 4 | p1 motivation only; no policy, planner or decision task | One agent-facing probe from frozen embeddings (road user entering the path in the next second) |
| 7. Evidence and statistics | Substance criterion | 7 | 7 rivals, 2 scales, paired BCa; single runs; 115k has only LoRA and frozen (Table 1 note) | Two more seeds of RAW and Ours at 1B·10k |
| 8. Resource value | Group B rewards resources | 7 | 276 h, 22 cities; Fig 2 density gap | State released splits and an evaluation script so others can reuse the protocol |
| 9. Clarity for a cross-disciplinary audience | Stated review criterion | 5 | Interval widths in abstract (SH5); undefined diagnostics (SH6); Sec 5 well organised | Plain deltas and one-line definitions (appendix allowed here) |
| 10. Failure analysis and discussion value | "failure analyses are welcome" | 6 | Fig 3 red bars; Table 2 costs; no label failure examples | A 3-example strip of label failures (missed rider, merged crowd, false pair) |

**Total: 57 / 100.** On the same rubrics the matched Step 1 papers score about 67 (paper 3, spatial composition: controlled data manipulation, 50 runs), about 67 (paper 4, human-labelled judge data, direct agent use) and about 48 (paper 5, proof of concept without numbers). The draft sits inside the accepted range, below the two empirical papers mainly on label verification and agent relevance. With Part C items 1 to 4 it would reach about 70.

### PART C. Improvements before the deadline (about 4 to 5 days)

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
| Rank | Improvement and what it shows | What it needs | Where it goes (4-page budget) | Rubric gain | One venue or all |
|---|---|---|---|---|---|
| 1 | Video-disjoint ladder: retrain plain, RAW and Ours at 1B·10k with no source video shared between training and test. Shows whether the Table 2 ladder survives without overlap | 3 arms, about 18 to 36 GPU h, plus under 1 GPU h of scoring; start now | 3 rows under the 1B block of Table 2; replace the "Training-test overlap" sentence of Sec 5 with the result | R3 +3, R7 +1 | Result for all; framing here |
| 2 | Human check of labels as the bridge's "verification": precision and recall of road users, agreement on interacting pairs | 100 frames, 3 raters, about 4 person-hours | Replace "a human check ... is the next step" in Sec 5 with the numbers; 2 lines in Sec 2 | R4 +3, R10 +1 | All |
| 3 | Data card: clips per city and per viewpoint, masked share and why 28,856 clips lack masks, the 10k strata, agents and pairs per clip | Saved metadata; CPU, about 3 h | Compact table in the free half page (SH8) or the appendix, which does not count here | R5 +2, R1 +1, R3 +1 | Here (appendix) |
| 4 | Density split of the ladder: do curated views help more in crowded clips? | Saved per-clip results; CPU, about 3 h | 2 lines in Sec 5 "Which experience helps the learner" | R2 +1, R6 +0.5 | All |
| 5 | Seeds: 2 more seeds of RAW and Ours at 1B·10k, after item 1 | 4 arms, about 24 to 48 GPU h | Add ± spread to Table 2's 1B rows | R7 +1 | All |

Cut to make room: Sec 6 repeats Sec 5 on overlap and labels; merge them and drop the Sec 6 heading (about 6 lines). Move the nine lower rows of Table 1 at 10k to the appendix if space is still short (the appendix does not count at this venue). Shorten the Sec 3 design sentence (T4 of v3).

### PART D. Tone audit

Rows T1 to T9, T11 and T12 of the v3 table apply word for word at the same places (shared text). Rows specific to this draft:

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
| # | Page/section | Quoted original (exact) | Problem | Rewrite |
|---|---|---|---|---|
| D1 | p4 Sec 5 | "Each question the bridge asks of agentic data has an answer here." | Overclaim (label validity is left to "the next step") | "Our results speak to four of the bridge's questions about agentic data." |
| D2 | p4 Sec 5 | "Whether the automatic labels are valid: they come from a detector, a tracker, and a proximity rule; a human check of missed and false road users and of who yields to whom is the next step." | Negative, answers a theme question with "next step" | "Whether the automatic labels are valid: they come from a detector, a tracker, and a proximity rule, and the release ships the masks and pairs so they can be audited; a human check of missed and false road users and of who yields to whom is the next step." |
| D3 | p1 Abstract | "Competing methods still lead on temporal coherence." | Negative ("still"), ends the abstract on a loss | "Competing methods lead on temporal coherence, a trade-off we report in full." |

The 3 rewrites that matter most: D2 (it is the theme-3 answer and now reads as a gap), T9 of v3 (in a data venue, "does not depend on the automatic labels" reads as "the data curation did not matter"; the rewrite keeps the fact and states what the labels buy), T12 of v3 (the overlap limitation is a theme-4 topic and should read as a protocol, not a confession).

Rewritten abstract (211 words; current 224):

> We turn public street video, abundant domain data for agents that move among people, into staged training experience for a video world model without human labels, and measure what each curation step adds. DENSEWORLD-115k holds 115,687 clips (276 hours) from 714 public YouTube videos of 22 Indian cities, released as reconstructible metadata. For 86,831 of them, a detector, a tracker, and a proximity rule cut each clip into three curated views: the street layout with road users blurred, the road users with the background suppressed, and crops around interacting pairs. FactorJEPA trains the video world model V-JEPA 2.1 on these views as a three-stage curriculum. Against seven adaptation methods under identical clips and budgets, one fixed configuration predicts masked and future content better at both 2B and 1B scale, and after full 115k-clip training it separates from the stronger of the two rivals trained at that scale on all four diagnostics of predictive structure, by 13.9 to 43.3 widths of the paired 95% confidence interval. An ablation that runs the same stages on raw clips prices each ingredient: the staged schedule carries most of the gain, and the curated views add robustness when more of the scene is hidden, at a cost in motion readout. Competing methods lead on temporal coherence.

### PART E. Double-blind check

Same items and replacements as the v3 table (dataset link, earlier-version footnote, names, figure thumbnails, Fig 3 header, metadata pass, no self-citations, upload file name). Differences: footnote 1 here names Cog2Soc, Continual Causality and Human-Centric AgMS. One addition: this draft says the masks and interaction pairs are released (p1 bullet), so check that the released mask files carry no user names or local paths in their headers or JSON fields before the reviewers can open the dataset page.

## Ranking by acceptance probability and the single most valuable change

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
| Rank | Draft and venue | Acceptance now | Single most valuable change before its deadline |
|---|---|---|---|
| 1 | v4, Data2AgentEx (Dec 1) | 60% | Retrain plain, RAW and Ours at 1B·10k on a video-disjoint split (about 18 to 36 GPU h) and put the result in Table 2 and Sec 5, so the bridge's "training-test overlap" question gets a measured answer |
| 2 | v1, Human-Centric AgMS (Dec 25) | 50% | Add a road-user forecasting probe (occupancy and path entry 1 to 2 s ahead, frozen embeddings, paired intervals) on a video-disjoint split, so the paper carries a mobility metric |
| 3 | v3, Cog2Soc (Nov 29) | 40% | Split the saved per-clip results by road users per clip (CPU only) and lead Sec 5 with how the gain changes with crowding, together with the 100-frame, 3-rater label check |
| 4 | v5, Continual Causality (Dec 1) | 30% | Score all saved checkpoints on the BDD100K and nuScenes reference clips and a general-video set (inference, about 6 to 12 GPU h) to report retention against gain, and replace "We make no causal claim" with the interventional test of Part C item 2 |

Changes that help all four at once (shared sections): state rival starting checkpoints and evaluation targets (SH2, SH3), source-video clustered intervals at 115k (SH4), plain deltas next to interval widths (SH5), one-line diagnostic definitions and the missing citations (SH6), real input frames in Fig 1 and a consistent Fig 3 legend (SH7), and the human label check. Because these edit shared files, do them first, before the Cog2Soc deadline, so every later venue inherits them.
