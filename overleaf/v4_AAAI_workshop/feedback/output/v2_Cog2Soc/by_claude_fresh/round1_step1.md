# STEP 1: Calibration sheet from seven accepted AAAI-26 workshop papers

**Source PDF:** `overleaf/v4_AAAI_workshop/feedback/input/prev_acceptances/pdf/aaai26ws_00_combined_7papers.pdf` (50 pages, all read, including figures and tables). Every page number below is a page of that combined PDF.

Papers: (1) Text-to-Vector Conversion for Residential Plan Design, pp. 1 to 4; (2) Fine-tuning LLMs for Tourist Trajectory Prediction, pp. 5 to 9; (3) Multimodal Data Fusion ... Vulnerable Older Adults, pp. 10 to 15; (4) Vehicle Routing with Finite Time Horizon using Deep RL, pp. 16 to 23; (5) Attention Gathers, MLPs Compose, pp. 24 to 31; (6) Towards Human-AI Accessibility Mapping in India, pp. 32 to 40; (7) One Rule to Bring Them All, pp. 41 to 50.

## 1. Each paper in three lines

**(1) Text-to-Vector Conversion for Residential Plan Design (pp. 1 to 4, AI for Urban Planning)**
- **Core claim:** SDXL with a new "white background" loss (Eq. 1 and 2, p. 3) followed by a new Shi-Tomasi corner vectorizer produces clean SVG floor plans that beat raster-plus-vectorizer and LLM pipelines; the abstract claims "approximately 5%" better CLIPScore (p. 1).
- **Main evidence:** Table 1 (p. 3) compares seven pipelines on number of paths, time and CLIPScore: ours 12 paths, 23.3 s, 0.288 against the best rival at 0.268 (SDXL + EvoVec). Figure 1 (p. 3) is a pipeline sketch. The text mentions "about 30 experiments" (p. 3) and "About 20 prompts ... and 10 plans were generated based on each prompt" (pp. 3 to 4), but no table shows those runs.
- **Why accepted:** a concrete tool for planners, a new loss written as equations, a baseline table with recent generators and two LLMs, and public code (footnote 1, p. 1).

**(2) Fine-tuning LLMs for Tourist Trajectory Prediction using Field Experiment Data (pp. 5 to 9, AI4UP, footnote p. 5)**
- **Core claim:** QLoRA fine-tuning of Llama 3.1 8B on local trajectories predicts the next point of interest of tourists at Wakayama Castle Park with 49.1% accuracy, beating Markov models, an HMM, GPT-4o and Llama 3 Swallow (abstract p. 5; results p. 7).
- **Main evidence:** a field dataset of 566 trajectories (87 GPS, 479 QR stamp rally, p. 6); Figure 3 (p. 8) accuracy bars; Figure 4 (p. 8) n-gram and BLEU scores; a rainy-day subset of 12 test samples at 41.7% vs 8.3% for Markov (p. 7); a tourist-level 80/20 split (p. 7).
- **Why accepted:** real human behaviour from a field experiment, a behavioural framing (context-dependent choices, weather, fatigue), a clear baseline ladder with a large margin, and a stated policy use (counterfactual tests of mobility services).

**(3) Multimodal Data Fusion to Capture Dynamic Interactions between Built Environment and Vulnerable Older Adults (pp. 10 to 15, AI for Urban Planning, footnote p. 10)**
- **Core claim:** fusing eye tracking, IMU gait, EDA and HRV, GPS with SLAM, and first- and third-person video can represent how micro-scale built-environment features affect older adults with knee osteoarthritis or a fall history (abstract p. 10).
- **Main evidence:** qualitative only. Figure 1 sensing setup (p. 11), Figure 2 SLAM in GPS-denied spaces (p. 12), Figure 3 trajectories coloured by estimated walkway width and surface material (p. 13), Figure 4 dashboard (p. 14). No table, no participant count, no accuracy for the material classifier or the width estimate.
- **Why accepted:** a vulnerable human group, an ambitious and well illustrated sensing setup, clear planning value, and honest labelling of the section as "Preliminary Results" (p. 13).

**(4) Vehicle Routing with Finite Time Horizon using Deep Reinforcement Learning with Improved Network Embedding (pp. 16 to 23)**
- **Core claim:** a graph attention encoder with edge features plus a cross-attention global graph embedding that includes the remaining time horizon raises the share of customers served and cuts solution time (abstract p. 16).
- **Main evidence:** Tables 1 to 3 (p. 21), e.g., EMA-50 deterministic 86.6% vs 74.1% for DRL-Transformer; Table 4 (p. 21) solution time; Figure 3 (p. 22) horizon sweep; Table 5 (p. 22) ablation on EMA-50: 86.6 full vs 76.4 without the global embedding and 83.2 without edge features.
- **Why accepted:** clean MDP formulation (p. 18), a named new module, real road networks (EMA from OpenStreetMap, Vienna), five baselines, and an ablation.

**(5) Attention Gathers, MLPs Compose: A Causal Analysis of an Action-Outcome Circuit in VideoViT (pp. 24 to 31, Deployable AI)**
- **Core claim:** in a Kinetics-400 ViViT, the "success vs failure" difference between a strike and a gutter ball is built by an amplification cascade over layers 5 to 11, with attention heads gathering evidence and MLP blocks composing the concept (abstract p. 24).
- **Main evidence:** Figure 5 (p. 28) L2 norm of the activation delta rises from about 75 to about 270; Tables 1 and 2 (p. 29) top-10% token ablation barely moves the logits (-0.34 and -0.02); Table 3 and Figure 7 (pp. 29 to 30) activation patching recovers 37 to 54% (attention) vs 42 to 60% (MLP); Figure 1 (p. 26) attention overlays on real frames.
- **Why accepted:** mechanistic interpretability on a video model, which the paper says is rarely done (p. 24); causal interventions rather than heatmaps alone; a memorable thesis in the title; candid limitations.

**(6) Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh (pp. 32 to 40)**
- **Core claim:** Project Sidewalk, adapted to Chandigarh with localized labels, Indian example images and Gemini-generated mission guidance, supports useful annotation; an audit of about 40 km around 230 POIs flags 1,644 of 2,913 locations (abstract p. 32).
- **Main evidence:** Figure 2 and Table 2 (pp. 36 to 37) US to India label changes; Table 3 (p. 38) guidance examples; Table 4 (p. 38) ratings of Relevance 4.97, Accuracy 4.40, Usefulness 4.61 (N = 150); Table 5 (p. 39) Spearman ρ and weighted κ per rater pair; Figures 4 and 5 (p. 39) sector and category scores.
- **Why accepted:** a real deployment in an understudied Indian city, an explicit "context difference" section (p. 34), a human-in-the-loop evaluation with agreement statistics, and an actionable finding (education and transport score lowest, p. 39).

**(7) One Rule to Bring Them All: Investigating Transport Connectivity in Public Transport Route Generation for Equitable Access (pp. 41 to 50)**
- **Core claim:** adding a demand-weighted connectivity term to transit network design produces more robust and more equitable networks at a moderate operator cost (abstract p. 41).
- **Main evidence:** Table 2 (p. 46): algebraic connectivity 0.96 vs 0.77 on Mumford2; Figure 1 (p. 46) Pareto fronts; Figure 2 (p. 47) route sets; Tartu (Estonia) city test in Table 3 and Figure 3 (p. 48).
- **Why accepted:** an equity framing, a new metric, synthetic benchmarks plus one real city, honest discussion of modest gains and of planner interpretability (pp. 48 to 49), anonymized code links (footnotes on p. 44 and p. 47).

## 2. The common bar

| Element | What the accepted papers have |
|---|---|
| Framing | Every paper opens on a societal problem tied to the workshop theme: housing demand (1, p. 1), uneven tourist flows (2, p. 5), ageing and falls (3, p. 10), logistics (4, p. 16), trustworthy deployment (5, p. 24), disability access in India (6, p. 32), equitable transit (7, p. 41) |
| Novelty statement | An explicit "novel" claim or numbered contribution list in the abstract or introduction: (1) p. 1, (4) pp. 16 to 17, (5) p. 24 "threefold", (6) p. 33, (7) pp. 41 to 42 |
| Evidence level | One main quantitative comparison with named baselines (1, 2, 4, 7), a human rating study (6), or a causal intervention series (5); one paper (3) has no numbers at all |
| Statistics | Very light: no intervals or seeds in (1), (2), (4); a single fixed seed in (5) (p. 26) and (7) (p. 44); only (6) reports SD, Spearman ρ and κ (pp. 38 to 39) |
| Real data | Six of seven use real places or people: Wakayama (2), Singapore (3), EMA and Vienna (4), Chandigarh (6), Tartu (7); (5) uses two Kinetics videos |
| Length | 4 to 10 pages with references; the short ones carry about 3.3 to 4.5 pages of text |
| Figures | Every paper has at least one pipeline or setup figure; 1 to 7 figures each |
| Writing tone | Confident and positive ("substantially outperforming", p. 6; "successfully reverse-engineered", p. 30); limitations short or absent |

## 3. What these papers get away with

| Paper | Weakness that did not block acceptance | Where |
|---|---|---|
| (1) | Table 1 caption says "one text prompt" while the text mentions averages over about 20 prompts; no variance; the ImageReward metric was dropped because "its results are irrelevant"; no ablation of the new loss; no limitations | p. 3, p. 4 |
| (2) | GPT-4o numbers in the text (18.7% zero-shot, 35.2% fine-tuned) disagree with the Figure 3 bars (2.3% and 22.5%); the rainy-day claim rests on 12 test samples; one split and no intervals; 18% of persona attributes inferred by GPT-4o | p. 7, p. 8, p. 6 |
| (3) | No quantitative result; participant count never given; "sub-meter resolution" asserted without a measurement; a Grounding DINO plus SAM material classifier reported without any accuracy | pp. 12 to 13 |
| (4) | No seeds or variance; gains given as relative percentages of averages ("59.2%"); ablation on one instance family only; no limitations section | p. 21, p. 22 |
| (5) | The whole analysis uses one pair of videos and seed 42; data and code "available from the authors upon request"; a broad AI safety conclusion drawn from one pair | p. 25, p. 26, p. 30 |
| (6) | Three raters on 50 segments; Relevance κ not computable because ratings were nearly constant; one rater pair at κ = -0.113 on Accuracy; severity weights 0.2, 0.6 and 1.0 set by hand; no ground truth for label correctness | p. 37, pp. 38 to 39 |
| (7) | Single fixed seed; the main benchmark Tables 1 to 5 sit in "additional materials"; Figure 1 subpanels unreadable at print size; the headline "AC = 0.96 vs. 0.77" (p. 45) skips the passenger-only row of the same Table 2, which shows AC 1.01; Tartu connectivity gains are tiny (AC 0.020 vs 0.017) | p. 44, p. 45, p. 46, p. 48 |
| All | Typos and grammar slips, e.g., "irrelevant and unlikelihood" (p. 4) and "a a bigger number" (p. 41) | |

## 4. The one sentence per paper that best sells the novelty

| Paper | Quoted sentence | Where |
|---|---|---|
| (1) | "Our approach surpasses existing solutions by approximately 5% in CLIPScore-based visual quality, benefiting from its inherent handling of right angles and flexible settings." | Abstract, p. 1 |
| (2) | "This establishes LLMs as high-fidelity behavior models for context-dependent tourist prediction, providing groundwork for counterfactual analysis of mobility interventions." | Abstract, p. 5 |
| (3) | "This study presents a novel multimodal data-fusion approach that integrates wearable and environmental sensing to dynamically represent human-environment interactions and quantify the BE impacts on mobility among vulnerable older adults, specifically those with knee osteoarthritis (OA) or a history of falls." | Abstract, p. 10 |
| (4) | "We present a novel routing network embedding module which creates local node embedding vectors and a context-aware global graph representation." | Abstract, p. 16 |
| (5) | "Crucially, the existence of this sophisticated circuit for representing complex outcomes, even within a model trained only for simple classification, highlights the potential for models to develop forms of 'hidden knowledge' beyond their explicit task, underscoring the need for mechanistic oversight for building genuinely Explainable and Trustworthy AI systems intended for deployment." | Abstract, p. 24 |
| (6) | "In this work, we contextualize the Project Sidewalk user experience with locally-enriched tags (Figure 3) and examples thus enabling the first deployment of Project Sidewalk in India" | Introduction, p. 32 |
| (7) | "Our contribution is to introduce a transport connectivity-aware accessibility metric that bases optimization on principles of equitable accessibility rather than traditional trade-offs between passenger and operator costs." | Abstract, p. 41 |

Pattern: one sentence that names the artefact and the "first" or "novel" angle, usually with one number, placed in the abstract.

## 5. How they write about limitations and negative results

| Paper | Placement | Length | Tone and move |
|---|---|---|---|
| (1) | None | 0 | Conclusion is purely positive (p. 4) |
| (2) | Last two paragraphs of the Conclusion (p. 8) | About 10 lines | Matter of fact: "does not validate causal claims"; "Several limitations remain"; then "Despite these limitations, this work provides a foundation ..." |
| (3) | Only as future work (p. 14) | 3 lines | "Future work will expand the participant pool to investigate statistical association"; the missing statistics are never called a limitation |
| (4) | None | 1 line of future work | Fleet routing as next step (p. 22) |
| (5) | Own subsection 5.2 Limitations (p. 30) plus a results subsection on a failed probe (4.2, p. 28) | About 15 lines | Opens "Since the experiments are foundational, a primary limitation is the scope", then pivots to "the existence proof of 'hidden cognition' is sufficient to question the safety of deployed AI systems"; the failed linear probe (100% from layer 0, Figure 6) is told as the reason for the better method |
| (6) | Inline in the analysis (p. 39) | A few clauses | Weak agreement described neutrally ("R1's ratings differed slightly"); discussion ends "This marks the start of a journey" |
| (7) | Inside results (p. 45) and Discussion (pp. 48 to 49) | About 20 lines | "over-emphasizing C_w reveals important limitations" and "does not lead to dramatic performance gains", then turned into an argument for transparent post-evaluation of generated routes |

Pattern: limitations are short, placed at the end, and paired with a constructive next step or reframed as a finding. None of the seven lists five or more deficits back to back.

## 6. Short papers and how their bar differs

| Paper | Total pages | Text pages (approx.) | Figures and tables | What carried it |
|---|---|---|---|---|
| (1) | 4 | 3.3 | 1 figure, 1 table | One comparison table and a new loss |
| (2) | 5 | 4 | 4 figures, 0 tables | One accuracy figure on real field data |
| (3) | 6 | 4.5 | 4 figures, 0 tables | A demonstrated sensing system, no numbers |
| (4) | 8 | 7 | 3 figures, 5 tables | Multi-dataset tables plus ablation |
| (5) | 8 | 7 | 7 figures, 3 tables | Causal intervention series |
| (6) | 9 | 8 | 5 figures, 5 tables | Deployment plus rater statistics |
| (7) | 10 | 8.5 | 3 figures, 3 tables | Benchmarks plus one real city |

Short papers (1) to (3) clear the bar with one main result or one demonstrated system, no ablation, no statistics, and limitations optional. Longer papers add an ablation (4), several datasets (4, 7), human rating statistics (6), causal tests (5) and an explicit limitations passage (5, 7). A 4-page paper should be judged against (1) to (3): one clear result attached to a well framed applied problem.

## 7. What reviewers likely valued in (6) for Indian data and in (5) for a video model

**Paper (6), Indian data**
- It shows concretely why a Western tool breaks in India: shared carriageways, informal shoulders, vendor stalls, drainage channels and parked two-wheelers occupying footpaths (p. 35), with a side-by-side US vs India label and severity figure (Figure 2, p. 36) and local examples (Figure 3, p. 36).
- It grounds the study in place and law: Chandigarh's "7 Vs" road hierarchy and the RPwD Act 2016 and national accessibility guidelines (p. 34), and cites Indian evidence (Delhi PWD audit with 84% of footpaths failing, Bengaluru pedestrians at 32% of road deaths, p. 34).
- It puts humans in the loop and reports agreement honestly even when weak (Tables 4 and 5, pp. 38 to 39).
- It ends with an actionable local finding (p. 39).
- **Lesson for the next paper:** name what is specifically Indian in the scenes, show what a Western reference misses, cite local evidence, and validate automatic labels with even a small human check.

**Paper (5), video model**
- It looks inside the model instead of only scoring outputs, using a contrastive pair with the same class and a different outcome to isolate one variable (p. 25).
- It tests causally (top-K ablation, activation patching; Tables 1 to 3) rather than relying on attention maps, and shows layer-wise curves (Figures 2, 5, 7).
- It ties the claim to real frames (Figure 1, p. 26).
- It reports a failed probe and explains what it taught (Section 4.2, p. 28).
- **Lesson for the next paper:** for a factorized world model, show what the agent and interaction channels carry on real frames, test them causally (remove a channel and see which clips suffer), and present negative results as informative.

## Bar to carry into Step 2

1. A named human or societal problem tied to the venue, stated in the first paragraph.
2. One clear novelty sentence in the abstract.
3. At least one quantitative result against sensible baselines (the short papers need only one).
4. One pipeline or setup figure and one result figure or table.
5. Statistics are a bonus at this level: intervals, seeds or agreement scores put a paper above the calibration set, not at it.
6. Limitations short, at the end, paired with next steps.
7. For Indian data: show the local specifics and a human check. For a video model: show inside the model, causally if possible.
