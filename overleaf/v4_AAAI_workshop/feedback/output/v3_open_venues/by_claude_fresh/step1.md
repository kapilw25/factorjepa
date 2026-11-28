Judge: Claude (fresh helper, no chat context), run Fri Nov 27 2026, inputs = GitHub main at commit 52c2809c

# STEP 1: Calibration sheet from 10 accepted 4-page AAAI workshop and bridge papers

Page numbers are pages of the combined 55-page PDF.

## 1. Each paper in 3 lines

**(1) How Well Can VLMs Understand Humans' Intention? (ToM4AI, AAAI-25), pp. 1-5**
- Core claim: an open-ended question framework (intention, visual cue, future inference) shows GPT-4 best and GPT-4o-mini close, while VLMs fail on complex social scenes such as bullying and cheating, and small models can get the intention right from wrong cues (abstract, p. 1).
- Main evidence: Table 1 (p. 3), scores out of 30 for 4 VLMs x 3 categories (GPT4 27/27/28, LLaVA-7B 7.5/8.5/7); Fig. 1 (p. 3), 3 qualitative image examples; qualitative failure list (p. 4, cheating read as "focused" or "multitasking").
- Why accepted: squarely on the ToM theme; three explicit research questions Q1-Q3 (p. 2); a stated gap against binary-choice ToM work (Jin et al., p. 1); released data (GitHub link, p. 1); vivid human-interpretable failure cases.

**(2) From Biased Chatbots to Biased Agents (TrustAgent, AAAI-26), pp. 6-11**
- Core claim: demographic persona prompts that are irrelevant to the task shift LLM agent performance by up to 26.2% (abstract, p. 6).
- Main evidence: Table 2 (p. 8), 3 models x 5 benchmarks x 6 race/origin personas with coloured arrows; Fig. 2 gender radar (p. 8); Fig. 3 profession on ALFWorld (p. 9); Fig. 4 religion on Card Game (p. 9); Table 1 persona list (p. 7).
- Why accepted: timely trust question, one controlled manipulation (fixed two-turn prefix, p. 7), broad grid (23 personas, 5 benchmarks, 3 models), one memorable headline number, a "first systematic case study" claim (p. 6) and a clear teaser figure (Fig. 1, p. 6).

**(3) Spatial Distribution of Pre-training Data for Geospatial FMs (Good-Data, AAAI-25), pp. 12-22**
- Core claim: balanced, globally representative pre-training compositions give roughly equal downstream results and beat clustered compositions or no pre-training; the gaps shrink as fine-tuning data grows (abstract p. 12; p. 15).
- Main evidence: Table 1 (p. 15), F1 mean ± std over 50 runs, 6 continents x 2 tasks, cells 2% or more below UAR highlighted; Fig. 2 (p. 15) learning curves with bands; Figs. 3-9 and Table 2 in the appendix (pp. 18-22).
- Why accepted: a pure data-centric question for a data workshop; a controlled design (equal sample count per composition, Appendix B p. 17); the only paper in the set with 50 repeated runs and spread; actionable guidance for practitioners; "first study" claim (p. 13).

**(4) "Are We Done Yet?" Vision-Based Judge for Computer Use Agents (TrustAgent, AAAI-26), pp. 23-27**
- Core claim: a zero-shot VLM judge on the final screenshot classifies task completion with up to 73% accuracy, and feeding its rationale back raises CUA success by 27% relative on average (abstract, p. 23).
- Main evidence: Table 1 (p. 26), accuracy for 5 evaluators x 3 CUAs; Fig. 2 (p. 26), success before and after one retry; Fig. 1 pipeline (p. 25); dataset of 1,260 human-labelled tasks over 42 macOS apps (p. 24, Zenodo link).
- Why accepted: a new human-labelled resource plus a simple closed loop; directly about trust and control of agents; clear contribution bullets (p. 23).

**(5) Towards Adaptive Environment Generation for Training Embodied Agents (Embodied AI reliability bridge, AAAI-26), pp. 28-33**
- Core claim: a proof-of-concept closed-loop generator (LLM analysis F, LLM editor G over a ProcTHOR scene graph) can produce environments matched to an agent's weaknesses (abstract, p. 28).
- Main evidence: no quantitative result; Fig. 1 failure under layout perturbation (p. 28); Fig. 2 one worked loop (p. 30); Fig. 3 collision-aware placement (p. 31); objective J(G) (p. 29); prompts in the appendix (p. 33).
- Why accepted: a crisp, well-motivated idea ("diversity without feedback is inefficient", p. 28), a clean formalisation, polished figures, and openly stated preliminary status.

**(6) From IID to the Independent Mechanisms Assumption in CL (Continual Causality, AAAI-23, oral), pp. 34-38**
- Core claim: the independent mechanisms (IM) assumption should replace iid in continual learning, and modular causal learners can adapt sparsely under hidden shifts (p. 34).
- Main evidence: a taxonomy of 5 distribution shifts (p. 35); a toy MoE experiment on two arithmetic streams, Figs. 1-2 (p. 36), MSE vs EWC and a monolithic net, with a shaded band on Fig. 2.
- Why accepted: a position statement that joins causality and CL, the exact bridge theme; one small toy experiment that makes the argument concrete.

**(7) Spurious Features in Continual Learning (Continual Causality, AAAI-23, poster), pp. 39-42**
- Core claim: CL creates "local spurious features" that correlate with labels inside a task but not over the whole scenario, distinct from covariate-shift spurious features (p. 39).
- Main evidence: toy illustration Fig. 1 (p. 40); feature taxonomy Table 1 (p. 40); formal definitions Eqs. 1-2 (p. 41). No experiment.
- Why accepted: a named new concept with a formal definition that links CL forgetting to causal feature discovery.

**(8) Never Ending Reasoning and Learning (Continual Causality, AAAI-23, oral), pp. 43-46**
- Core claim: proposes the NERL paradigm uniting causality, CL, statistical relational AI and human-allied AI, motivated by adverse pregnancy outcomes (p. 43).
- Main evidence: Venn diagram Fig. 1 (p. 44); five challenges (p. 45). No data.
- Why accepted: a vision paper from senior researchers, a concrete high-stakes application, and an explicit call for cross-community work.

**(9) Formal Safety Guarantees for AVs using Barrier Certificates (Embodied AI reliability bridge, AAAI-26), pp. 47-50**
- Core claim: TTC-based barrier certificates verified with the Z3 SMT solver, plus a speed adjustment, cut TTC < 3 s events by up to 40% on HighD (abstract, p. 47).
- Main evidence: Algorithm 1 (p. 48); framework Fig. 1 (p. 49); per-lane conflict counts before and after adjustment, Fig. 3 (p. 50), for 300 and 3000 frames.
- Why accepted: bridges formal methods and driving (the bridge's theme), uses a real naturalistic dataset, and an interpretable safety metric.

**(10) Fine-tuning LLMs for Tourist Trajectory Prediction (AI for Urban Planning, AAAI-26), pp. 51-55**
- Core claim: QLoRA-fine-tuned Llama-3.1-8B reaches 49.1% next-POI accuracy on 566 field-experiment trajectories and generalises to rare contexts such as rainy days (abstract, p. 51).
- Main evidence: Fig. 3 accuracy bars vs Markov, HMM, GPT-4o and Swallow (p. 54); Fig. 4 n-gram and BLEU (p. 54); POI map Fig. 1 and text encoding Fig. 2 (p. 53); rainy-day subset of 12 samples, 41.7% vs 8.3% (p. 53).
- Why accepted: new real field data, a planning use case, several baselines, a tourist-level split "to prevent data leakage" (p. 53), and an honest causal caveat (p. 54).

## 2. The common bar for a 4-page AAAI workshop or bridge paper

| Element | What every accepted paper has | Examples |
|---|---|---|
| Framing | The venue's core concept appears in the title or first paragraph; a named gap in the first column | "remains underexplored" (p. 1); "largely unexplored" (p. 6); "relatively understudied" (p. 12) |
| Novelty statement | One sentence, often "first" or "we propose", in abstract and intro; 2 of 10 add contribution bullets | "first systematic case study" (p. 6); "first study" (p. 13); bullets pp. 13, 23 |
| Evidence level | From none (position papers 7, 8) to one qualitative demo (5) to one main table or figure of single-run numbers (1, 2, 4, 9, 10); only (3) runs 50 seeds | Table 1 p. 3; Table 2 p. 8; Table 1 p. 15 |
| Statistics | No significance test in any paper; only (3) reports std, (6) and (3) show bands; most report point numbers | Table 1 p. 15 (± std) |
| Figures | A teaser or pipeline figure in 8 of 10, plus 1 to 3 result figures | Fig. 1 pp. 6, 13, 25, 28, 49 |
| Size of claims | One headline number in the abstract | 26.2% (p. 6), 73% and 27% (p. 23), 40% (p. 47), 49.1% (p. 51) |
| Tone | Confident and constructive; limitations short, late, framed as future work | pp. 9, 31, 54 |
| Length | 3 to 4 pages of main text, often 1 table and 2 to 3 figures; appendices used freely (3 has 7 appendix pages) | pp. 16-22 |

## 3. What these papers get away with (weaknesses that did not block acceptance)

| Weakness | Where |
|---|---|
| Very small samples | 30 images, each author annotating 10, checked by one third-party evaluator (p. 2); rainy-day subset of 12 samples (p. 53); 566 trajectories (p. 51) |
| Single runs, no variance, no tests | (1), (2) deterministic decoding (p. 7), (4), (9), (10) |
| No baseline at all | (5) and (9): no rival controller or generator |
| No quantitative result | (5) states evaluation "remains as future work" (p. 31); (7) and (8) have no experiments |
| Manual scoring | VLM answers "manually compared with human annotations" by keyword relevance, 1 / 0.5 / 0 points, no agreement statistic (p. 3) |
| Text and figure disagree | (10): text gives zero-shot GPT-4o 18.7% and fine-tuned GPT-4o 35.2% (p. 53), Fig. 3 shows 2.3% and 22.5% (p. 54); (9): abstract claims "complete elimination of conflicts in some lanes" (p. 47) yet no lane in Fig. 3 reaches zero (p. 50), and its "up to 40% fewer events" sits beside a lane 3 drop of about half (192 to 94, 920 to 427); (4): "relative improvement of 27%" (p. 23 abstract) vs "27% relative percentage points" (p. 23 bullet) |
| Claim without shown result | (7): "Our results show that the problem of local spurious features also plays a major role" (p. 42) with no experiment in the paper |
| Causal reading of correlational data | (2) attributes shifts to stereotypes ("mirror human social stereotypes", p. 6) without a mechanism test |
| Near-tautological evaluation | (9) recomputes TTC after lowering the follower's speed (p. 49) |
| Heavy appendix | (3): 4 pages of main text plus 7 pages of appendix (pp. 16-22) |
| Inferred labels | (10) 18% of persona attributes inferred by GPT-4o (p. 52) |

## 4. The one sentence per paper that best sells its novelty

| # | Quote (exact) | Page |
|---|---|---|
| 1 | "We propose an open-ended question framework to evaluate VLMs' performance across diverse categories of ToM tasks." | p. 1 |
| 2 | "In this work, we present the first systematic case study showing that demographic-based persona assignments can alter LLM agents' behavior and degrade performance across diverse domains." | p. 6 |
| 3 | "To the best of the authors' knowledge, this is the first study to explore the impact of pre-training data distribution on downstream task performance for GFMs." | p. 13 |
| 4 | "We propose a methodology for autonomous evaluation of task completion, achieving up to 73% accuracy and improving success rate of CUAs by an average of 27% relative percentage points." | p. 23 |
| 5 | "To address this, we present a proof-of-concept for closed-loop environment generation that adapts difficulty to the agent's current capabilities." | p. 28 |
| 6 | "While there are several candidates, here we advocate that the independent mechanism assumption (IM) (Schölkopf et al., 2012) is a useful hypothesis for representing knowledge in a form, that makes it easy to adapt to new tasks in CL." | p. 34 |
| 7 | "This paper investigates the problem of spurious features (with covariate shift) in continual learning and shows that the continual learning setup leads to a specific type of spurious features that we call local spurious features (LSFs) (without covariate shift) in CL as shown in Fig. 1." | p. 39 |
| 8 | "Inspired by the motivation behind the Never-Ending Language Learner (NELL), a continual learning system that reads the web, we propose the never-ending reasoning and learning paradigm and one instance: the Never-Ending Reasoner and Learner (NERL), which continuously learns and reasons with causal models by actively interacting with domain experts." | p. 43 |
| 9 | "This work develops a formally verified safety framework for Connected and Autonomous Vehicles (CAVs) that integrates Barrier Certificates (BCs) with interpretable traffic conflict metrics, specifically Time-to-Collision (TTC) as a spatio-temporal safety metric." | p. 47 |
| 10 | "Our fine-tuned Llama-3.1-8B achieves 49.1% next POI accuracy and maintains strong performance on undersampled scenarios like rainy days, demonstrating effective generalization." | p. 51 |

## 5. How they write about their own limitations and negative results

| # | Placement | Length | Tone |
|---|---|---|---|
| 1 | No limitations section; one clause in the conclusion: VLMs "often need further fine-tuning to contextualize subtle cues" (p. 4) | 1 sentence | Model failures are presented as findings about VLMs, not as weaknesses of the study |
| 2 | "Limitations and Future Work" paragraph at the very end (p. 9) | about 7 lines | "broader coverage ... would further clarify"; purely forward-looking |
| 3 | One hedge in the results (p. 14), "preliminary insights" in the conclusion (p. 15), full list in Appendix E (p. 19) | 1 line main, 1 page appendix | "areas for future exploration"; the fading of gaps with more data is reported as a finding (p. 15) |
| 4 | No limitations section; "Future Work" (pp. 25-26) admits the binary metric and plans confidence intervals | 4 paragraphs | Every limitation is a next step |
| 5 | Titled paragraph "Limitations of frontier LLM Spatial Reasoning Capabilities" (p. 31) and "comprehensive evaluation ... remains as future work" | 1 paragraph | The limitation is placed on frontier LLMs, with a mitigation already tried |
| 6 | "Open challenges include ..." in the conclusion (p. 37); MCL's "much larger mean MSE at the beginning" stated plainly (p. 36) | 2 sentences | Neutral |
| 7 | None | 0 | Not discussed |
| 8 | The whole paper is challenges framed as "opportunities" (pp. 44-45) | n/a | Positive |
| 9 | None | 0 | Not discussed |
| 10 | "Several limitations remain" in the conclusion (p. 54), plus "does not validate causal claims" | about 8 lines | Honest and constructive, each limitation paired with a fix |

Pattern: limitations sit at the end, take at most one paragraph in the main text, never appear in the abstract, and are phrased as future work. Negative results are reported as discoveries ("struggled", "vulnerability", "converge") rather than as shortcomings of the method.

## 6. Evaluation evidence per paper

| # | Repeated runs | Variance or intervals | Ablations or factor sweeps | Failure analysis | Reports what did not work |
|---|---|---|---|---|---|
| 1 | No | No | No | Yes, qualitative (p. 4) | Yes, LLaVA and cheating failures (p. 4) |
| 2 | No, deterministic decoding (p. 7) | No | Persona categories as factors (Tables 1-2) | No per-error analysis | Partly: OS and Database "relatively stable, typically fluctuating within 2-5%" (p. 7) |
| 3 | 50 seeds (p. 14) | Mean ± std, shaded bands (p. 15) | Compositions, fine-tuning size sweep, 4 heads (Appendix D, p. 19) | Per-continent breakdown | Yes, gaps vanish with more data (p. 15) |
| 4 | No | No | 5 evaluators compared (Table 1) | No | No |
| 5 | n/a | n/a | No | One failure example (Fig. 1) | Yes, LLM spatial-reasoning limits (p. 31) |
| 6 | Toy | Shaded band in Fig. 2 (p. 36) | MCL vs EWC vs monolithic | No | Yes, MCL worse early (p. 36) |
| 7 | None | None | None | None | None |
| 8 | None | None | None | None | None |
| 9 | Two frame windows (300, 3000) | No | No | Per-lane breakdown (Fig. 3) | No |
| 10 | Single 80/20 tourist-level split | No | 6 baselines (Fig. 3) | Rainy-day and lunch subsets; 5.4% anomalous outputs (p. 54) | Yes, Swallow underperforms (p. 53) |

Summary: 1 of 10 has repeated runs with spread; 0 of 10 has a significance test or a confidence interval; 6 of 10 report at least one thing that did not work. A paper that brings paired intervals, a leak-free split and ablations is far above this bar on statistics; the bar is set instead by relevance and clarity.

## 7. What each community rewards, and how its bar differs

**Group A (behaviour of humans and agents; calibrates Cog2Soc).** Reviewers reward a clearly named human phenomenon (intention, social stereotype) and a finding phrased in human terms ("mirror human social stereotypes", p. 6). Human grounding counts for more than scale: 30 images annotated by the authors and checked by one third-party evaluator passed (p. 2) because the human label is the point. Statistics are not demanded. A paper here must say what it teaches about humans or about agents that model humans; a pure representation-learning result is off the bar.

**Group B (data for agent learning; calibrates Data2AgentEx).** Reviewers reward a data or evaluation resource (1,260 labelled tasks, p. 23; released code and data) or a controlled manipulation of the data itself (5 compositions at fixed size, p. 13; closed-loop environment edits, p. 28). The question "which data, selected how, helps the agent?" must be explicit. The bar on evidence is moderate (one table), but data lineage, leakage control and released artefacts carry weight. A proof-of-concept is acceptable when the data pipeline is the contribution (paper 5).

**Group C (continual learning and causality; calibrates Continual Causality).** These are vision and position papers with formal definitions (Eqs. 1-2, p. 41), taxonomies (p. 35) and toy experiments (p. 36). Reviewers reward an explicit causal concept (independent mechanisms, spurious vs causal features, evolving causal knowledge) tied to non-stationarity. The empirical bar is the lowest of the four groups, but the conceptual bar is the highest: the paper must name the causal structure and the shift. In one-column PMLR layout these are about 2 AAAI pages of text, so an empirical 4-page AAAI paper has room to spare but will be judged on whether it speaks causally, not on its statistics.

**Group D (mobility and traffic agents; calibrates Human-Centric AgMS).** Reviewers reward real mobility data (HighD, p. 49; a field experiment, p. 52), interpretable domain metrics (TTC thresholds, next-POI accuracy) and a deployment story (planners, CAV controllers). One dataset is enough; baselines are present in (10) but absent in (9). The bar differs from B in that domain realism and a human user (driver, tourist, planner) matter more than data methodology, and from C in that a concrete real-world measurement is expected.
