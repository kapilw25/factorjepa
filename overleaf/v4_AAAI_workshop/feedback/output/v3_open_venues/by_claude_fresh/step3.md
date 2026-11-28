Judge: Claude (fresh helper, no chat context), run Fri Nov 27 2026, inputs = GitHub main at commit 52c2809c

# STEP 3: Judgment of the 4 venue drafts

The four PDFs share Sections 2, 3, 4 and 6, Tables 1 and 2 and Figures 1 to 3 word for word; they differ in title, abstract, Section 1, Section 5 and footnote 1. To avoid saying the same thing four times, the shared problems are listed once as SH1 to SH10 and cited by number inside each draft's Parts A to E.

GPU-hour figures below are my rough estimates (the drafts report no training cost): about 6 to 12 GPU h for one 1B·10k training arm, 12 to 24 GPU h for one 2B·10k arm, 60 to 120 GPU h for one 1B·115k arm, and under 1 GPU h to embed or score the 1,825 test clips with one checkpoint. Scale them from your own training logs.

## Shared findings (all 4 drafts)

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


## Draft v3: Cog2Soc (due Sun Nov 29 AoE, about 2.5 days)

Title: "Layout, Then Road Users, Then Interactions: Adapting a Video World Model to Dense South Asian Street Behaviour"

### PART A. Acceptance probability

**Cog2Soc: 40%. Typical AAAI-27 computer-vision or embodied-AI workshop: 50%.**

The draft does what the call makes mandatory on page 1: a labelled "The human phenomenon." paragraph (negotiated road edges, yielding, gap acceptance, small lateral passing), "Our approach.", and a bridge to agent-based and social-force modelling (Helbing and Molnár 1995) that Section 5 develops. But nothing after page 1 measures a behaviour: every diagnostic is an embedding error or a linear probe (Sec 6: "Every diagnostic is measured in the frozen V-JEPA embedding space or with a linear probe on it"), and the behavioural tests are listed as not done ("Trajectory forecasting, a comparison with a calibrated social-force or agent-based model, and a check of the interaction clips against human judgements of who yields to whom are the next tests; none is reported here"). The Group A papers were accepted with tiny samples because the human phenomenon was the measured object (30 annotated images of intention, p. 2 of Step 1; persona effects on agents, p. 8). Here a psychology or computational social science reviewer sees a video-model adaptation paper with a social framing. The rigour helps on "substance", and the room may enjoy discussing whether learned world models can replace hand-set interaction rules, so it is not a clear reject; it sits below the Group A bar on relevance.

Three things most likely to cause rejection:
1. No behavioural evidence: the human phenomenon of p1 is never measured; Sec 6 concedes trajectories, social-force comparison and human judgements are "none ... reported here".
2. Thin bridge to the other community: one social-science citation; "cross-cultural evaluation" is promised (abstract, Sec 5) but no model is evaluated across cultures ("all footage here is Indian", Sec 6), and Fig 2 compares density only (SH7).
3. Attribution and validity (SH1, SH4): the title's three-stage clip order is not what drives the gains, interaction labels come from "a proximity and relative-motion rule" with no human check, and the split is clip-level with single runs.

Three strongest points:
1. Clean fit to the call's structure: phenomenon, approach and community link are each labelled on p1, and Sec 5 maps the curriculum to how agent-based models are built ("environment, individuals, interactions").
2. Evidence quality far above Group A: 7 rivals, 2 model scales, 2 data scales, strongest-rival comparison (Table 1), paired BCa intervals and a source-video-resampled ablation (Table 2).
3. A dense, non-Western behavioural corpus (115,687 clips, 22 cities, Fig 2 density gap on all six compared scene types) with a responsible release paragraph (p2) and candid reporting of where rivals lead (Fig 3).

### PART B. Ten rubrics

| Rubric | Why it matters here | Score | Evidence from the paper (page/section) | The one change that adds the most points |
|---|---|---|---|---|
| 1. Human phenomenon named and central | Mandatory item of the call | 6 | p1 "The human phenomenon." names yielding, gap acceptance, lateral passing; never measured later | Split the saved per-clip results by road users per clip and show where the gain sits |
| 2. Link to a second community | Mandatory item of the call | 5 | Helbing and Molnár 1995 is the only social-science reference; Sec 5 maps to agent-based layering | Add 3 to 4 behavioural citations (mixed-traffic gap acceptance, heterogeneous traffic simulation, cultural road norms) |
| 3. Behavioural validity of the measures | Reviewers ask what is learned about behaviour | 3 | Sec 6: all diagnostics in embedding space; trajectory and yield tests "none is reported here" | A linear probe on saved embeddings for per-clip interaction count, with intervals |
| 4. Empirical grounding of automatic labels | Call topic "empirical grounding, calibration" | 3 | p1 Sec 2: Grounding DINO, SAM 3 and a proximity rule; no validation | Report the planned 100-frame, 3-rater check (precision, recall, Fleiss kappa) |
| 5. Uncertainty and replication | Call topic | 7 | Paired BCa (p3); 127-video resampling (Table 2); ρ ≥ 0.895 (p3); single runs (Sec 6) | Recompute the 115k separations with source-video clustering (SH4) |
| 6. Cross-cultural evaluation | Call topic, promised in abstract and Sec 5 | 4 | Fig 2 density only; "all footage here is Indian" (Sec 6) | Score the saved checkpoints on the BDD100K and nuScenes reference clips |
| 7. Novelty and attribution | Title sells the clip order | 5 | Table 2: RAW at or below Ours on both L1s; clips give 6% and 2% flatter slope (SH1) | Lead with "the staged schedule wins; the structured clips add robustness to hidden road users" |
| 8. Responsible use | Call topic | 7 | p2 "Release and responsible use": metadata only, CC BY 4.0, takedown, no individual consent named | One sentence on faces and number plates in rebuilt clips and on excluded uses |
| 9. Clarity for a mixed audience | Psychology and sociology reviewers | 4 | Abstract "13.9 to 43.3 widths of the paired 95% confidence interval"; 9 undefined diagnostics (SH5, SH6) | Plain deltas in the abstract and a one-line definition per diagnostic |
| 10. Suitability for discussion | Stated review criterion | 7 | Sec 5: a learned predictor built in agent-based order; dense non-Western reference corpus | End Sec 5 with one concrete question for modellers, tied to Table 2 |

**Total: 51 / 100.** Scored on the same ten rubrics, the matched Step 1 papers come out at about 55 (paper 1, ToM VLM benchmark: human phenomenon measured, weak statistics) and about 61 (paper 2, persona effects on agents: a clean manipulation of a social variable). The draft beats both on rubric 5 by a wide margin but loses on rubrics 1, 3 and 4, which Cog2Soc reviewers weight most. With Part C items 1 to 4 done it would score about 63, above both.

### PART C. Improvements before the deadline (about 2.5 days)

| Rank | Improvement and what it shows | What it needs | Where it goes (4-page budget) | Rubric gain | One venue or all |
|---|---|---|---|---|---|
| 1 | Density split: Ours vs RAW vs LoRA on Future-frame L1 and Mask-ratio slope by tercile of road users per clip. Shows whether the gain grows with crowding, the collective phenomenon. Report it whichever way it falls | Saved per-clip test errors (the paired bootstrap implies they exist) plus per-clip agent counts from the Fig 2 pipeline; CPU, about 3 h | A 3-row inset or 3 lines after Table 2 in the free half page (SH8); no cut | R1 +2, R3 +1 | Analysis for all; placement in Sec 5 here |
| 2 | Human check of labels: road-user precision and recall, agreement on "interacting pair" and on who yields | 100 frames, 3 raters, about 4 person-hours; no GPU | 3 lines at the end of the Sec 2 pipeline paragraph; drop the human-check item from Sec 6 | R4 +3, R5 +0.5 | All (shared Sec 2) |
| 3 | Cross-cultural check: run the saved 2B·10k and 1B·10k checkpoints (frozen, plain, LoRA, full fine-tuning, RAW, Ours) on the BDD100K and nuScenes reference clips of Fig 2. Shows what adapting to dense streets does on sparse lane-structured scenes | Inference only, about 3 to 6 GPU h | 3 lines in Sec 5 and a 2-column addition to Table 2 | R6 +2, R3 +1 | Run once; framed here as cross-cultural, in v5 as retention |
| 4 | Rewrite for the room: plain deltas in the abstract (2.7% lower error, 19% flatter slope at 115k); 3 to 4 behavioural citations; one-line definitions of all 13 diagnostics; state rival starting checkpoints and evaluation targets (SH2, SH3) | Writing, about half a day | Abstract; +2 lines in Sec 1; +5 lines in Diagnostics using free space | R2 +2, R9 +2, R7 +1 | Definitions and SH2/SH3 for all; citations here |
| 5 | Source-video clustered intervals for the 115k numbers and Fig 3; fix the Fig 3 legend (SH7) | Saved per-clip results; CPU, about 1 h | Fig 3 caption and Findings sentence | R5 +1 | All |

Cut to make room if items 1 to 4 overflow the free half page: compress the Sec 3 design sentence (about 2 lines); delete the third repetition of "the rivals still lead" in Sec 5 (it already appears in the abstract and Findings); merge the two Fig 2 panels into one grouped bar chart (about 10 lines). Do not cut Table 1's lower rows: showing wins and losses alike is a selling point.

### PART D. Tone audit

| # | Page/section | Quoted original (exact) | Problem | Rewrite |
|---|---|---|---|---|
| T1 | p1 Abstract | "Competing methods still lead on temporal coherence." | Negative ("still") | "Competing methods lead on temporal coherence, a trade-off we report in full." |
| T2 | p2 Sec 2 | "These two axes are the ones measured here; occlusion and interaction pressure are the natural next axes." | Negative (what is not measured) | "We measure density on two axes, agents per clip and pixel share; occlusion and interaction pressure extend the same comparison." |
| T3 | p2 Release | "Rebuilt clips are the public videos as posted, so a video its creator removes can no longer be rebuilt, and rights holders can ask on the dataset page for a source to be removed." | Engineering and operations | "Rebuilding from the public source honours later removals: a video its creator takes down can no longer be rebuilt, and rights holders can request removal of a source on the dataset page." |
| T4 | p2 Sec 3 | "The first stage unfreezes the first 4 of the encoder's 48 blocks and the later stages the first 8, following surgical fine-tuning (Lee et al. 2023); the predictor trains throughout against a frozen teacher, selective projection decay (Tian, Huang, and Kira 2024) keeps the weights near their starting point, and the fixed configuration splits its epochs 30/25/45 across the stages." | Engineering (configuration dump) | "Three anchors keep each stage close to the pretrained model: only the earliest encoder blocks adapt (4 of 48 in the first stage, 8 later), following surgical fine-tuning (Lee et al. 2023); the predictor learns against a frozen teacher; and selective projection decay (Tian, Huang, and Kira 2024) holds the weights near their start. Epochs split 30/25/45 across the stages." |
| T5 | p3 Protocol | "Test clips are held out at the clip level (1,825 in the 10k regime, 23,106 in the full regime), so a source video can contribute clips to both training and test (Section 6)." | Self-critical aside in the protocol | "Test clips are held out at the clip level (1,825 in the 10k regime, 23,106 in the full regime); because a source video can contribute clips to both training and test, Table 2 also resamples the 127 source videos (Section 6)." |
| T6 | p3 Table 1 caption | "One fixed FactorJEPA model (Ours, identical in every cell) against its strongest rival on 13 diagnostics." | Imprecise (2B and 1B are different models) | "One fixed FactorJEPA configuration (Ours, the same settings in every column) against the strongest rival on each of 13 diagnostics." |
| T7 | p3 to p4 Findings | "Full fine-tuning, which leads on Motion cosine at 10k, was not trained at 115k, so whether more data closes that gap remains open." | Negative (gap left open) | "At 115k we trained LoRA and the frozen backbone as rivals; full fine-tuning, which leads on Motion cosine at 10k, was not trained at that scale, and training it there will show whether more data closes that gap." |
| T8 | p3 Findings | "FactorJEPA's gain is in one-step prediction of crowded scenes; on the coherence of long predicted futures, the rivals still lead." | Negative ("still") | "FactorJEPA's advantage lies in one-step prediction of crowded scenes, while the rivals lead on the coherence of long predicted futures." |
| T9 | p4 after Table 2 | "Most of the gain therefore does not depend on the automatic labels, and what the labels buy is a specific property: robustness to missing evidence." | Undercuts the contribution | "The staged schedule therefore delivers most of the gain without any labels, and the automatic labels add a distinct property on top: robustness to missing evidence." |
| T10 | p4 Sec 5 | "Temporal coherence over longer rollouts, which a simulation needs, is where the rivals still lead." | Negative ("still") | "The rivals lead on temporal coherence over longer rollouts, the property a simulation needs next." |
| T11 | p4 Sec 6 | "Trajectory forecasting, a comparison with a calibrated social-force or agent-based model, and a check of the interaction clips against human judgements of who yields to whom are the next tests; none is reported here, and the interaction clips are chosen by a proximity and relative-motion rule." | Negative list of what is missing | "The interaction clips come from a proximity and relative-motion rule, which makes three tests the direct next steps, none reported here: trajectory forecasting, comparison with a calibrated social-force or agent-based model, and a check of the interaction clips against human judgements of who yields to whom." |
| T12 | p4 Sec 6 | "The clip-level split lets a source video contribute to both training and test, and no city is held out. Video-level and city-level splits, dense-traffic regions beyond India (all footage here is Indian), and repeated training runs (each configuration was trained once) are the natural extensions." | Negative list | "Each configuration was trained once, on a clip-level split in which a source video can contribute to both training and test and no city is held out; all footage is Indian. The same protocol extends directly to repeated runs, video-level and city-level splits, and dense-traffic regions beyond India." |

The 3 rewrites that matter most: T9 (it currently tells the reader the paper's named idea is not what works), T11 (the Cog2Soc reader's first question is behaviour, and this sentence answers "none"), T10 (the simulation use is the bridge's hook; state the lead and the requirement without "still").

Rewritten abstract (207 words; current 209):

> We contribute DENSEWORLD-115k and FactorJEPA, a corpus and a curriculum for learning crowded South Asian street behaviour with a video world model. There, pedestrians, two-wheelers, auto-rickshaws, carts, and animals share one right of way and coordinate by negotiation rather than lane rules, often while partly hidden from each other; models of such collective behaviour are often calibrated on sparser, lane-structured, mostly Western scenes. DENSEWORLD-115k holds 115,687 clips (276 hours) from 714 public YouTube videos of 22 Indian cities, released as reconstructible metadata. FactorJEPA adapts V-JEPA 2.1 in the order agent-based models are built: the street layout, then the road users, then their interactions. Against seven adaptation methods under identical clips and budgets, one fixed configuration predicts masked and future content better at 2B and 1B scale; after full 115k-clip training it separates from the stronger of the two rivals trained at that scale on all four diagnostics of predictive structure, by 13.9 to 43.3 widths of the paired 95% confidence interval. A raw-clip ablation shows the staged adaptation carries most of this gain, while the layout, agent, and interaction clips add robustness when more of the scene is hidden. Competing methods lead on temporal coherence. We discuss what this offers modelling and cross-cultural evaluation of collective behaviour.

### PART E. Double-blind check

| Item | Where | Risk | Anonymous replacement |
|---|---|---|---|
| Dataset link | p1, below the abstract (Hugging Face, account "anonymousML123") | The account page can list other repositories, likes or organisations; the dataset card may carry a BibTeX entry, an arXiv ID or a code link; commit history shows the uploader | Keep only if the account is single-purpose and the card has no names, arXiv ID or code link; otherwise "Dataset: anonymized link in the supplementary material" |
| Footnote 1, earlier version | p1 | "An earlier version is a non-archival AAAI-27 student abstract and a public preprint." With the distinctive names DENSEWORLD-115k and FactorJEPA, one web search finds the preprint and its authors | Disclosure is legitimate and preprints are normally allowed; to lower the risk write "A preliminary version exists as a non-archival abstract and a preprint (withheld for review)". Also confirm the student abstract's archival status: past AAAI student abstracts appeared in the proceedings |
| Footnote 1, concurrent venues | p1 | Names Data2AgentEx, Continual Causality and Human-Centric AgMS; not author-identifying | Keep (honest disclosure) |
| Method and dataset names | Throughout | Same names as the public preprint | Renaming would break the release link; accept the residual risk |
| Figure images | p2 Fig 1 | Frames from public YouTube videos; at 220 dpi I saw no channel watermark; panel (b) shows a bank sign and regional script, which reveal a city, not authors | Check all thumbnails at full size for creator logos or overlays |
| Figure 3 header | p3 | "vJEPA2.1 (1B) Full 115k" looks like an internal plot title; not identifying | Rename "V-JEPA 2.1 (1B), full 115k" for consistency |
| PDF metadata | File | I checked: Creator "TeX", no Title, Author, Subject or Keywords, no hyperlinks, no embedded file paths | Pass |
| Self-citations, acknowledgements | Whole PDF | None | Pass |
| Upload file name | File | The file is named after the venue and a "fixes" revision | Upload as "paper.pdf" |


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


## Draft v1: Human-Centric AgMS (due Fri Dec 25, 8:59 PM PST, about 4 weeks)

Title: "Can a World Model of Traffic Scenes Predict Crowded South Asian Streets?"

### PART A. Acceptance probability

**Human-Centric AgMS: 50%. Typical AAAI-27 computer-vision or embodied-AI workshop: 60%.**

The title quotes the workshop's own theme A ("world models of traffic scenes"), the introduction opens on "Agentic mobility services, from driver assistants to delivery robots and fleet dispatchers", and Section 5 claims a benchmark "not tied to driving metrics" (theme D). The workshop welcomes early-stage work and negative results, which suits the candid Fig 3. But Group D papers were accepted on a mobility measurement: TTC conflicts on HighD (paper 9) and next-POI accuracy with baselines and a traveller-level split (paper 10). This draft has no mobility measurement: no trajectory, collision, occupancy or planning metric ("Trajectory forecasting ... none is reported here", Sec 6), and the answer to its title question is "in part" (Sec 5). With four weeks and one GPU this is the most fixable of the four drafts; as it stands it is a coin flip. A general CV or embodied-AI workshop would weight the dataset and the controlled comparison more and the missing mobility task less.

Three things most likely to cause rejection:
1. No downstream mobility task or domain metric (Sec 6), so "What This Offers Mobility Agents" (Sec 5) is argued, not shown.
2. Deployment generalisation untested: clip-level split, no held-out city, single country, single runs (Sec 6; SH4).
3. Benchmark claim without a benchmark: "DENSEWORLD-115k also gives a benchmark for this regime that is not tied to driving metrics" (Sec 5) with no task, released split or protocol; and the property planning needs, long-horizon coherence, is where rivals lead (Fig 3).

Three strongest points:
1. Direct theme fit (A, C, D) and a clear agent framing on p1.
2. Real, dense, heterogeneous traffic at scale, with a density gap against BDD100K and nuScenes on all six compared scene types (Fig 2).
3. Rigorous, honest evaluation that this workshop explicitly welcomes: 7 rivals, 2 scales, wins and losses alike.

### PART B. Ten rubrics

| Rubric | Why it matters here | Score | Evidence from the paper (page/section) | The one change that adds the most points |
|---|---|---|---|---|
| 1. Fit to theme A (world models of traffic scenes) | Primary theme | 7 | Title; p1 Introduction; Sec 5 | Name one agent use with a metric in Sec 1 (time until a road user enters the path) |
| 2. Road users and humans (theme C) | Human-centric focus | 5 | p1 lists pedestrians, two-wheelers, carts, animals; no result by road-user type | Split Table 2 and Fig 3 numbers by road-user mix or density (saved results) |
| 3. Downstream mobility metric | Group D rewards it | 2 | Sec 6: trajectory forecasting "none is reported here" | A forecasting probe of tracked road users 1 to 2 s ahead, with intervals |
| 4. Benchmark beyond driving metrics (theme D) | Claimed in Sec 5 | 4 | Sec 5 claim; no protocol | Define the benchmark: released splits, tasks, metrics, baseline numbers |
| 5. Real-world data realism | Group D rewards real data | 8 | 276 h, 22 cities, mixed viewpoints, Fig 2 | Match viewpoint in Fig 2 (drive-through only against the vehicle-mounted reference) |
| 6. Fair, complete comparison | Reviewers check baselines | 6 | 7 rivals; 115k has only LoRA and frozen; SH2, SH3 | Train full fine-tuning at 1B·115k and state starting checkpoints |
| 7. Generalisation and split hygiene | Deployment on real roads | 3 | Clip-level split, no held-out city (Sec 6) | Video- and city-disjoint retrain of the key arms |
| 8. Honest scope | Negative results welcome | 8 | Fig 3 red bars; Sec 5 "in part" | Keep, but phrase as findings (Part D) |
| 9. Clarity for transport and HCI readers | Mixed audience | 5 | Interval widths (SH5); undefined diagnostics (SH6) | Plain deltas and one-line definitions |
| 10. Safety and accountability | Theme D | 6 | p2 responsible use; no failure modes for deployment | A short failure analysis: scene types where errors are largest, with frames |

**Total: 54 / 100.** On the same rubrics the matched Step 1 papers score about 58 (paper 9, barrier certificates on HighD: domain metric and safety story, no baseline) and about 66 (paper 10, tourist trajectories: real field data, baselines, leak-free split). The draft is stronger on statistics and data scale and weaker on the mobility metric and split hygiene. With Part C items 1 to 4 it would reach about 70.

### PART C. Improvements before the deadline (about 4 weeks)

| Rank | Improvement and what it shows | What it needs | Where it goes (4-page budget) | Rubric gain | One venue or all |
|---|---|---|---|---|---|
| 1 | Road-user forecasting probe: from the first half of each test clip, predict the second-half occupancy grid of tracked road users and whether any road user enters the central lower region of the frame (the camera's path) within 1 to 2 s; frozen embeddings of every method, linear or small attentive probe, paired BCa intervals; validate the labels with the planned 3-rater check | Track-derived labels from the existing SAM 3 masks; embedding extraction for all methods about 10 to 20 GPU h; probes about 5 GPU h | New Table 3 in the free half page (SH8); Sec 5 first paragraph rewritten around it | R3 +4, R1 +1, R4 +1 | Here (could later feed v3) |
| 2 | Split hygiene and seeds: retrain LoRA, full fine-tuning, RAW and Ours at 1B·10k on a video-disjoint split with two held-out cities; 2 extra seeds of RAW, Ours and LoRA | About 24 to 48 GPU h for the disjoint arms and 36 to 72 GPU h for seeds | Replace the clip-level 1B column of Table 1 or add rows to Table 2; delete the split sentences of Sec 6 | R7 +3, R6 +1 | All (shared tables) |
| 3 | Close the open 115k question: train full fine-tuning (and continual SSL if time allows) at 1B·115k | About 60 to 120 GPU h per arm | Fig 3 and Table 1 1B·115k column; delete "was not trained at 115k" | R6 +1 | All |
| 4 | Define the benchmark and add road-user splits: released splits, the 4 predictive diagnostics plus the probe, baseline table; per-density and per-mix breakdown of the results | CPU plus release work, about 2 days | 4 lines in Sec 5; one line in Sec 2 Release | R4 +2, R2 +2 | Here |
| 5 | Deployment failure analysis: scene types where Ours errs most, 3 frames, tied to the red rows of Fig 3 | Saved per-clip results; CPU | 3 lines plus a thin image strip | R10 +1 | Here |

Total GPU for items 1 to 3: about 135 to 265 GPU h, which fits in four weeks on one box. Cut to make room: merge the Fig 2 panels (about 10 lines); compress the Sec 3 design sentence (about 2 lines); fold Sec 6 into Sec 5 once items 2 and 3 remove most of its content.

### PART D. Tone audit

Rows T2 to T9, T11 and T12 of the v3 table apply word for word (shared text). Rows specific to this draft:

| # | Page/section | Quoted original (exact) | Problem | Rewrite |
|---|---|---|---|---|
| H1 | p1 Abstract | "Competing methods still lead on temporal coherence. We discuss what such a traffic-scene world model can and cannot yet offer mobility agents." | Negative close ("still", "cannot yet") | "Competing methods lead on temporal coherence over longer rollouts. We discuss what this offers mobility agents today and what planning still requires." |
| H2 | p4 Sec 5 | "Our results support that use in part." | Self-limiting opener | "Our results support the first part of that use, one-step prediction of partly hidden road users." |
| H3 | p4 Sec 5 | "Temporal coherence over longer rollouts, which matters for planning several seconds ahead, is where the rivals still lead." | Negative ("still") | "For planning several seconds ahead, temporal coherence over longer rollouts matters, and there the rivals lead." |
| H4 | p4 Sec 5 | "DENSEWORLD-115k also gives a benchmark for this regime that is not tied to driving metrics, which we hope the workshop community will extend with interaction-level labels." | Overclaim plus hedge (no benchmark protocol exists in the paper) | "DENSEWORLD-115k and the 13 diagnostics used here are a starting point for a benchmark of this regime beyond driving metrics, which interaction-level labels would extend." |

The 3 rewrites that matter most: H2 (the mobility section should open on what works), H4 (removes an unsupported claim while keeping the offer), T11 of v3 (the missing trajectory test is the reviewer's main objection; state it as the next step, then replace it with Part C item 1).

Rewritten abstract (217 words; current 221):

> We contribute a world model of traffic scenes for crowded South Asian streets, where a mobility agent that drives, walks, or dispatches must anticipate pedestrians, two-wheelers, auto-rickshaws, carts, and animals, often while they are partly hidden. World models of traffic scenes are trained and evaluated mostly on lower-density, lane-structured traffic. We build DENSEWORLD-115k, 115,687 clips (276 hours) from 714 public YouTube videos of 22 Indian cities, released as reconstructible metadata, and FactorJEPA, a curriculum that adapts the video world model V-JEPA 2.1 first on clips of the street layout, then on the road users, then on their interactions, with no extra inputs at inference. Against seven adaptation methods under identical clips and budgets, one fixed FactorJEPA configuration predicts masked and future content better at both 2B and 1B scale, and after full 115k-clip training it separates from the stronger of the two rivals trained at that scale on all four diagnostics of predictive structure, by 13.9 to 43.3 widths of the paired 95% confidence interval. An ablation on raw clips shows that the staged adaptation carries most of this gain, and the layout, agent, and interaction clips make prediction more robust when more of the scene is hidden. Competing methods lead on temporal coherence. We discuss what this offers mobility agents today and what planning still requires.

### PART E. Double-blind check

Same items and replacements as the v3 table. Differences: footnote 1 names the Cog2Soc, Data2AgentEx and Continual Causality bridges. This workshop allows concurrent submissions, so the disclosure is safe; the four-week window also allows moving the dataset link to an anonymized supplementary file if the Hugging Face account cannot be confirmed as single-purpose.


## Ranking by acceptance probability and the single most valuable change

| Rank | Draft and venue | Acceptance now | Single most valuable change before its deadline |
|---|---|---|---|
| 1 | v4, Data2AgentEx (Dec 1) | 60% | Retrain plain, RAW and Ours at 1B·10k on a video-disjoint split (about 18 to 36 GPU h) and put the result in Table 2 and Sec 5, so the bridge's "training-test overlap" question gets a measured answer |
| 2 | v1, Human-Centric AgMS (Dec 25) | 50% | Add a road-user forecasting probe (occupancy and path entry 1 to 2 s ahead, frozen embeddings, paired intervals) on a video-disjoint split, so the paper carries a mobility metric |
| 3 | v3, Cog2Soc (Nov 29) | 40% | Split the saved per-clip results by road users per clip (CPU only) and lead Sec 5 with how the gain changes with crowding, together with the 100-frame, 3-rater label check |
| 4 | v5, Continual Causality (Dec 1) | 30% | Score all saved checkpoints on the BDD100K and nuScenes reference clips and a general-video set (inference, about 6 to 12 GPU h) to report retention against gain, and replace "We make no causal claim" with the interventional test of Part C item 2 |

Changes that help all four at once (shared sections): state rival starting checkpoints and evaluation targets (SH2, SH3), source-video clustered intervals at 115k (SH4), plain deltas next to interval widths (SH5), one-line diagnostic definitions and the missing citations (SH6), real input frames in Fig 1 and a consistent Fig 3 legend (SH7), and the human label check. Because these edit shared files, do them first, before the Cog2Soc deadline, so every later venue inherits them.
