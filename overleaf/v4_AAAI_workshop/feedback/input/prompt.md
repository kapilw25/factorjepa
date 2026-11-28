# Judge prompt: FactorJEPA 4-page drafts for the 4 open AAAI-27 venues

Use a FRESH chat in each of ChatGPT, Claude and Gemini. Send STEP 1, wait for the answer, then STEP 2, wait, then STEP 3.
Attach the files with the chat's paperclip button. Every file is on the **Downloads** tab of the private [Workshop Deadlines](https://claude.ai/artifact/FemMusSPv4xxXJBTWGcw6b) page (open it while logged in to claude.ai):

- STEP 1 attachment: `aaai_ws_00_combined_10papers_4page.pdf` (10 accepted short papers, 55 pages, 1.8 MB): the **Save PDF** button on the Downloads tab. It is kept out of git because the papers are by other authors and the PMLR ones state no licence; this Mac keeps a copy at `prev_acceptances/pdf/` (gitignored).
- STEP 2 attachment: none (venue context only)
- STEP 3 attachments, one PDF per open venue (each is the draft for that venue; all built Fri Nov 27 2026, ~6:25 AM PST, from one shared source, all gates pass):
  1. Cog2Soc: [v3_Cog2Soc_fixes/0_main.pdf](https://github.com/kapilw25/factorjepa/blob/main/overleaf/v4_AAAI_workshop/v3_Cog2Soc_fixes/0_main.pdf) ([download](https://raw.githubusercontent.com/kapilw25/factorjepa/main/overleaf/v4_AAAI_workshop/v3_Cog2Soc_fixes/0_main.pdf))
  2. Data2AgentEx: [v4_Data2AgentEx/0_main.pdf](https://github.com/kapilw25/factorjepa/blob/main/overleaf/v4_AAAI_workshop/v4_Data2AgentEx/0_main.pdf) ([download](https://raw.githubusercontent.com/kapilw25/factorjepa/main/overleaf/v4_AAAI_workshop/v4_Data2AgentEx/0_main.pdf))
  3. Continual Causality: [v5_Continual_Causality/0_main.pdf](https://github.com/kapilw25/factorjepa/blob/main/overleaf/v4_AAAI_workshop/v5_Continual_Causality/0_main.pdf) ([download](https://raw.githubusercontent.com/kapilw25/factorjepa/main/overleaf/v4_AAAI_workshop/v5_Continual_Causality/0_main.pdf))
  4. Human-Centric: [v1_Human_Centric/0_main.pdf](https://github.com/kapilw25/factorjepa/blob/main/overleaf/v4_AAAI_workshop/v1_Human_Centric/0_main.pdf) ([download](https://raw.githubusercontent.com/kapilw25/factorjepa/main/overleaf/v4_AAAI_workshop/v1_Human_Centric/0_main.pdf))

Save each model's answers to `overleaf/v4_AAAI_workshop/feedback/output/v3_open_venues/by_<model>/step1.md`, `step2.md` and `step3.md` (`<model>` = chatgpt, claude, gemini).

The earlier one-venue prompts are in git history (Human-Centric) and in [prompt_v2_Cog2Soc.md](prompt_v2_Cog2Soc.md) (the fresh Claude judge loop on Cog2Soc).

## 📚 Calibration set (10 accepted short papers, grouped by the venue they calibrate)

Owner pick Fri Nov 27 2026 ~6:05 AM PST: venue-matched papers, each with at most 4 pages of main text, so the comparison is like for like (the earlier 7-paper AAAI-26 set had 4 papers of 7 to 8 pages and is retired; round 1 of the Cog2Soc judge used it). Every acceptance was checked on an official page; "main text" is measured to the References heading.

<!-- joined-words: off --> <!-- why: paper titles, venue names and licence names are quoted verbatim -->
| # | Pages in combined PDF | Paper | Accepted at | Main text | Source | Licence |
|---|---|---|---|---|---|---|
| 👥 1 | 1-5 | How Well Can Vision-Language Models Understand Humans' Intention? An Open-ended Theory of Mind Question Evaluation Benchmark | ToM4AI workshop, AAAI-25, poster ([list](https://sites.google.com/view/theory-of-mind-aaai-2025/poster-sessions)) | 3.8 p | [arXiv](https://arxiv.org/abs/2503.22093) | CC BY-SA 4.0 |
| 👥 2 | 6-11 | From Biased Chatbots to Biased Agents: Examining Role Assignment Effects on LLM Agent Robustness | TrustAgent workshop, AAAI-26, poster ([OpenReview](https://openreview.net/forum?id=WavZ34iiGB)) | 4.0 p | [arXiv](https://arxiv.org/abs/2602.12285) | CC BY 4.0 |
| 🧪 3 | 12-22 | How Does the Spatial Distribution of Pre-training Data Affect Geospatial Foundation Models? | Good-Data workshop, AAAI-25 ([OpenReview](https://openreview.net/forum?id=P4xkFMRyQJ)) | 4.0 p | [arXiv](https://arxiv.org/abs/2501.12535) | CC BY-SA 4.0 |
| 🧪 4 | 23-27 | "Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents | TrustAgent workshop, AAAI-26, poster ([OpenReview](https://openreview.net/forum?id=cYKcJGsgnA)) | 3.9 p | [arXiv](https://arxiv.org/abs/2511.20067) | CC BY 4.0 |
| 🧪 5 | 28-33 | Towards Adaptive Environment Generation for Training Embodied Agents | Bridge "Making Embodied AI Reliable with Testing and Formal Verification", AAAI-26 ([program](https://www.tacps.org/program/)) | 4.1 p | [arXiv](https://arxiv.org/abs/2602.06366) | CC BY 4.0 |
| 🔁 6 | 34-38 | From IID to the Independent Mechanisms assumption in continual learning | Continual Causality bridge, AAAI-23, oral ([list](https://www.continualcausality.com/23/papers)) | 3.4 p, one column | [PMLR 208](https://proceedings.mlr.press/v208/ostapenko23a/ostapenko23a.pdf) | not stated |
| 🔁 7 | 39-42 | Spurious Features in Continual Learning | Continual Causality bridge, AAAI-23, poster (same list) | 3.4 p, one column | [PMLR 208](https://proceedings.mlr.press/v208/lesort23a/lesort23a.pdf) | not stated |
| 🔁 8 | 43-46 | Never Ending Reasoning and Learning: Opportunities and Challenges | Continual Causality bridge, AAAI-23, oral (same list) | 3.1 p, one column | [PMLR 208](https://proceedings.mlr.press/v208/natarajan23a/natarajan23a.pdf) | not stated |
| 🚗 9 | 47-50 | Formal Safety Guarantees for Autonomous Vehicles using Barrier Certificates | Bridge "Making Embodied AI Reliable with Testing and Formal Verification", AAAI-26 ([program](https://www.tacps.org/program/)) | 2.9 p | [arXiv](https://arxiv.org/abs/2601.09740) | CC BY-NC-ND 4.0 |
| 🚗 10 | 51-55 | Fine-tuning LLMs for Tourist Trajectory Prediction using Field Experiment Data | AI for Urban Planning workshop, AAAI-26 ([list](https://ai-for-urban-planning.github.io/AAAI26-workshop/papers.html)) | 3.6 p | [arXiv](https://arxiv.org/abs/2608.20830) | CC BY 4.0 |

Legend: 👥 Cog2Soc · 🧪 Data2AgentEx · 🔁 Continual Causality · 🚗 Human-Centric AgMS.

Limits of the set, said plainly:
- Cog2Soc and Data2AgentEx are new bridges with no earlier edition, so their groups come from the nearest AAAI-25 and AAAI-26 workshops. No accepted short paper on collective behaviour or agent-based models could be downloaded, so the Cog2Soc group is theory of mind and agent behaviour.
- The Continual Causality group is that bridge's own first edition, but in PMLR's one-column layout: about 3.4 one-column pages is roughly 2 AAAI pages, less text than our drafts.
- Human-Centric has 2 papers. Every AAAI-25 autonomous-driving workshop paper found ran 6 to 7 pages of main text.
- arXiv copies were measured where the workshop version sits only on OpenReview, so a camera-ready may differ slightly.
- Dropped: a 2-page extended abstract (Rank-O-ToM, ToM4AI AAAI-25), by the 4-page rule.

## STEP 1 (attach the combined PDF of 10 accepted papers)

```text
You are a senior program-committee member for AAAI workshops and bridges on human behaviour modelling, agent data,
continual learning and mobility. The attached 55-page PDF joins 10 short papers ACCEPTED at AAAI workshops and bridges,
each with at most 4 pages of main text, grouped by the venue type they calibrate (page ranges are pages of the combined PDF):
Group A, behaviour of humans and agents (for a bridge on modelling human behaviour, cognition to society):
(1) How Well Can Vision-Language Models Understand Humans' Intention?, pp. 1-5 (ToM4AI, AAAI-25)
(2) From Biased Chatbots to Biased Agents, pp. 6-11 (TrustAgent, AAAI-26)
Group B, data for agent learning (for a bridge on turning domain data into agent experience):
(3) How Does the Spatial Distribution of Pre-training Data Affect Geospatial Foundation Models?, pp. 12-22 (Good-Data, AAAI-25)
(4) "Are We Done Yet?": A Vision-Based Judge for Computer Use Agents, pp. 23-27 (TrustAgent, AAAI-26)
(5) Towards Adaptive Environment Generation for Training Embodied Agents, pp. 28-33 (Embodied AI reliability bridge, AAAI-26)
Group C, continual learning and causality (earlier edition of the same bridge; PMLR one-column layout):
(6) From IID to the Independent Mechanisms assumption in continual learning, pp. 34-38 (Continual Causality, AAAI-23)
(7) Spurious Features in Continual Learning, pp. 39-42 (Continual Causality, AAAI-23)
(8) Never Ending Reasoning and Learning: Opportunities and Challenges, pp. 43-46 (Continual Causality, AAAI-23)
Group D, mobility and traffic agents:
(9) Formal Safety Guarantees for Autonomous Vehicles using Barrier Certificates, pp. 47-50 (Embodied AI reliability bridge, AAAI-26)
(10) Fine-tuning LLMs for Tourist Trajectory Prediction using Field Experiment Data, pp. 51-55 (AI for Urban Planning, AAAI-26)

Read all 10. Then write a calibration sheet I will use to judge a new paper:

1. For each paper, 3 lines: core claim, the main evidence (tables/figures), and what most
   likely got it accepted.
2. The common bar for acceptance at a 4-page AAAI workshop or bridge paper: what every accepted
   paper has (framing, novelty statement, evidence level, statistics, figures, writing tone).
3. What these papers get away with (weaknesses that did NOT block acceptance).
4. How each paper states its contribution in the abstract and introduction: quote the one
   sentence per paper that best sells its novelty.
5. How they write about their own limitations and negative results (tone, placement, length).
6. How each paper handles evaluation evidence: repeated runs, variance or intervals, ablations,
   failure analyses, and whether it reports what did not work.
7. For each group A to D: what that community's reviewers reward most, and how its bar differs
   from the other groups (for example, Group C papers are vision and position papers; Group B
   papers lead with data or evaluation resources).

Be concrete, cite page numbers. Do not judge any other paper yet; I will send the venues next.
```

## STEP 2 (no attachment: the 4 venues)

```text
Next I will send 4 versions of one paper, each tailored to one AAAI-27 venue that is still open. All 4 share the same
data, method, results and tables; each has its own title, abstract, introduction and one discussion section for its venue.
Read the 4 venue descriptions below and keep them in mind; do not judge anything yet. Reply with one short paragraph per
venue saying what its reviewers will look for first.

Venue 1. Modeling and Simulating Human Behavior in the Age of AI: From Cognition to Society (AAAI-27 Bridge, Cog2Soc)
- URL: https://cog2soc.github.io/ (OpenReview AAAI.org/2027/Bridge/Cog2Soc)
- Deadline: Sun Nov 29 2026, 23:59 AoE (Mon Nov 30, 3:59 AM PST). Notification Dec 10. Montreal, in person.
- About: brings AI, psychology, sociology, cognitive science, HCI and computational social science together to study
  human behaviour from individual cognition to collective dynamics, and to compare how each field builds and evaluates
  behavioural models. Topics: cognitive and behavioural mechanisms; individual differences and human-AI interaction;
  social networks, norms, institutions, cooperation and collective behaviour; generative agents, agent-based models and
  hybrid simulation; empirical grounding, calibration, uncertainty, replication, cross-cultural evaluation and
  responsible use. Each paper must explain the human phenomenon, its approach or research question, and its connection
  to another participating community. Reviews judge relevance, substance and suitability for discussion.
- Track used: up to 4 pages excluding references, AAAI two-column format, double-blind, non-archival.

Venue 2. From Domain Data to Agentic Experience: Bridging Data-Centric AI and Agent Learning (AAAI-27 Bridge, Data2AgentEx)
- URL: https://data2agentex.github.io/AAAI2027/ (OpenReview AAAI.org/2027/Bridge/Data2AgentEx)
- Deadline: Tue Dec 1 2026, 9:00 PM PST (OpenReview closes before the site's Dec 1 AoE). Notification Dec 13.
- About: "how can domain knowledge, data, and workflows become grounded, informative, and responsibly governed experience
  for learning and evaluating agents?" Five themes: workflow representation; experience production and management
  (demonstrations, logs, simulations, trace schemas, lineage); verification and evaluation (executable checks, expert
  review, learned evaluators, evaluation shortcuts, uncertainty); learning utility and transfer ("curricula, selection
  bias, training-test overlap, and controlled evaluation of transfer"); human and physical context (sensors, robots,
  consent, conditions for reuse). Reviews judge relevance, technical or empirical substance, clarity to a
  cross-disciplinary audience, and discussion value. Datasets, benchmarks and failure analyses are welcome.
- Track used: short paper, 4 pages (references and appendices not counted), AAAI-27 author kit, double-blind, non-archival.

Venue 3. Continual Causality Bridge, 4th edition, novel-work track (AAAI-27 Bridge)
- URL: https://www.continualcausality.com/cfp/ (OpenReview AAAI.org/2027/Bridge/Continual_Causality_Novel)
- Deadline: Dec 1 2026, AoE (Wed Dec 2, 4:59 AM PST). Notification Dec 11. Bridge Feb 17 2027, Montreal.
- About: "how learning systems can continually build and maintain accurate causal world models when the world is
  non-stationary and only partially observed." Suggested topics include continual learning in non-stationary
  environments; causal structure for continual and non-stationary RL and world models; agents in dynamic, confounded and
  partially observable worlds; shortcut learning and confounding in non-stationary data; foundation models at the
  intersection of causal and continual learning (including continual pretraining); datasets, benchmarks and applications.
  Earlier editions were published in PMLR volumes 208 (2023) and 268 (2024).
- Track used: short paper, 4 pages excluding references and optional appendices, AAAI format, double-blind,
  non-archival; "works must be original".

Venue 4. Human-Centric Agentic Mobility Services (AAAI-27 Workshop W4, Human-Centric AgMS)
- URL: https://agentic-mobility-workshop.github.io/cfp.html (OpenReview AAAI.org/2027/Workshop/Human-Centric_AgMS)
- Deadline: Fri Dec 25 2026, 8:59 PM PST. Results Sat Jan 2 2027. Montreal, Feb 22 or 23 2027, in person.
- About: agentic AI for vehicles, fleets and traveller assistants needs human-centred design to be reliable and
  accountable; agency covers goal interpretation, social interaction, tool use and ethical reasoning, not only autonomy.
  Themes: (A) agentic vehicles and in-cabin services, including grounding plans in perception and "world models of
  traffic scenes"; (B) fleets, transit and traffic, including multi-agent RL and generative-agent city simulation;
  (C) travellers, drivers and road users, including communication with pedestrians and accessibility; (D) trustworthy
  deployment on real roads, including "benchmarks beyond driving metrics". Early-stage work, benchmark proposals and
  negative results are welcome.
- Track used: up to 4 pages excluding references, AAAI-27 author kit, double-blind, non-archival, concurrent
  submissions allowed.

The same work is a pending AAAI-27 student abstract (non-archival) and a public arXiv preprint. Each draft carries an
anonymous page-1 footnote naming the other three venues, and each has its own title.
```

## STEP 3 (attach the 4 PDFs)

```text
Now judge the 4 attached PDFs. Each is the draft for one venue from Step 2: v3 = Cog2Soc, v4 = Data2AgentEx,
v5 = Continual Causality, v1 = Human-Centric. Use your Step 1 calibration sheet as the bar, adjusted for each venue
(start from the accepted papers that match that venue). Do PARTS A to E once per PDF, then finish with a ranking of the 4
by acceptance probability and the single most valuable change for each before its deadline.

Rules: be specific and honest. Cite page, section, table, figure or quoted sentence for every point. Do not invent
results the paper does not contain. Hard constraint for any suggestion: it must be doable before THAT VENUE's deadline
(Cog2Soc: about 2.5 days; Data2AgentEx and Continual Causality: about 4 to 5 days; Human-Centric: about 4 weeks), and the
paper must stay within 4 pages of main text (say what to cut for anything you add). Shared sections change all 4 drafts
at once, so say whether each change is for one venue or for all. One rented GPU box (1x RTX PRO 6000, 96 GB) is
available, so new training runs, inference and evaluation are allowed if they finish before the deadline (give rough GPU
hours), as are re-analysis of saved results (for example a split by scene type, by number of road users, or by source
video), new tables or figures, and rewriting. A small human check of the automatic road-user and interaction labels
(100 frames, 3 raters) is planned and may be cited as a suggestion.

PART A. Acceptance probability
- One number (0 to 100%) for acceptance at this venue, with a one-paragraph reason, and the same number for a typical
  AAAI-27 computer-vision or embodied-AI workshop.
- The 3 things most likely to make a reviewer reject, and the 3 strongest points.

PART B. Ten rubrics
Choose the 10 rubrics that you, as a reviewer for this venue, judge most important for this paper (do not use a generic
list; justify each choice in one line). Score each out of 10, as a table with columns:
rubric | why it matters here | score | evidence from the paper (page/section) | the one change that adds the most points
Then the total out of 100 and how it compares to the accepted papers from Step 1 that match this venue.

PART C. Improvements before the deadline (any resource: GPU training or inference, evaluation, analysis, writing)
List up to 5 improvements, ranked by expected score gain. For each: what it shows, what it needs (new run with model,
data and rough GPU hours, or saved results), where it goes (replace which element, so the paper stays within 4 pages),
which rubric it raises and by how much. Also list anything that should be CUT to make room.

PART D. Double-blind check
List anything in the PDF (links, dataset names, self-citations, acknowledgements, metadata wording) that could reveal the
authors, and the anonymous replacement for each.

PART E. Tone audit
Find every line or paragraph that reads as NEGATIVE and self-critical, or as an ENGINEERING/operations report, instead of
a RESEARCH paper that leads with its contribution and novelty. Output a table:
# | page/section | quoted original (exact) | problem (negative / criticism / engineering-only) | rewrite
The rewrite must be positive, constructive and centred on the contribution, BUT it must keep every fact and every number
true: never remove an interval, never turn a non-significant difference into a win or a tie, never hide a negative
result, never add a claim the paper does not test. No em dashes. 
Finish with: 
E.2) the 3 rewrites that matter most, and 
E.3) a rewritten abstract (same facts, at most the current length) that opens with the contribution.
```
