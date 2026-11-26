# Judge prompt: FactorJEPA (Human-Centric AgMS, AAAI-27 workshop, 4 pages)

Use a FRESH chat in each of ChatGPT, Claude and Gemini. Send STEP 1, wait for the answer, then send STEP 2.
Attach the files with the chat's paperclip button:

- STEP 1 attachment: `prev_acceptances/pdf/aaai26ws_00_combined_7papers.pdf` (on this Mac only: 50 pages, 2.0 MB, compressed).
  It is kept out of git because 3 of the 7 papers carry the arXiv non-exclusive licence, which does not allow redistribution.
  The papers' arXiv pages are listed in the table below.
- STEP 2 attachment: [v1_Human_Centric/0_main.pdf](https://github.com/kapilw25/factorjepa/blob/main/overleaf/v4_AAAI_workshop/v1_Human_Centric/0_main.pdf) ([download](https://raw.githubusercontent.com/kapilw25/factorjepa/main/overleaf/v4_AAAI_workshop/v1_Human_Centric/0_main.pdf)) (the anonymous 4-page paper to submit)

Save each model's answers to `overleaf/v4_AAAI_workshop/feedback/output/v1_Human_Centric/by_<model>/step1.md` and `step2.md` (`<model>` = chatgpt, claude, gemini).

## 📚 Calibration set (AAAI-26 workshop papers, all verified accepted)

| # | Pages in combined PDF | Paper | Workshop | arXiv | Licence |
|---|---|---|---|---|---|
| 1 | 1-4 (4 p) | Text-to-Vector Conversion for Residential Plan Design | AI4UP | [2602.10757](https://arxiv.org/abs/2602.10757) | CC BY-NC-SA 4.0 |
| 2 | 5-9 (5 p) | Fine-tuning LLMs for Tourist Trajectory Prediction using Field Experiment Data | AI4UP | [2608.20830](https://arxiv.org/abs/2608.20830) | CC BY 4.0 |
| 3 | 10-15 (6 p) | Multimodal Data Fusion to Capture Dynamic Interactions between Built Environment and Vulnerable Older Adults | AI4UP | [2601.11545](https://arxiv.org/abs/2601.11545) | CC BY 4.0 |
| 4 | 16-23 (8 p) | Vehicle Routing with Finite Time Horizon using Deep Reinforcement Learning with Improved Network Embedding | AI4UP | [2601.15131](https://arxiv.org/abs/2601.15131) | CC BY 4.0 |
| 5 | 24-31 (8 p) | Attention Gathers, MLPs Compose: A Causal Analysis of an Action-Outcome Circuit in VideoViT | DAI | [2603.11142](https://arxiv.org/abs/2603.11142) | arXiv non-exclusive |
| 6 | 32-40 (9 p) | Towards Human-AI Accessibility Mapping in India: VLM-Guided Annotations and POI-Centric Analysis in Chandigarh | AI4UP | [2602.09216](https://arxiv.org/abs/2602.09216) | arXiv non-exclusive |
| 7 | 41-50 (10 p) | One Rule to Bring Them All: Investigating Transport Connectivity in Public Transport Route Generation for Equitable Access | AI4UP | [2602.15053](https://arxiv.org/abs/2602.15053) | arXiv non-exclusive |

AI4UP = 2nd Workshop on AI for Urban Planning, AAAI-26 ([accepted papers](https://ai-for-urban-planning.github.io/AAAI26-workshop/papers.html)).
DAI = Workshop on Deployable AI, AAAI-26. AAAI-26 had no workshop on world models or driving and no predecessor of
Human-Centric AgMS, so the set calibrates the AAAI workshop acceptance bar, not world-model novelty.

## STEP 1 (attach the combined PDF of 7 accepted papers)

```text
You are a senior program-committee member for AAAI workshops on AI for transportation, urban
mobility and embodied perception. The attached 50-page PDF joins 7 papers ACCEPTED at AAAI-26
workshops, in this order (page ranges are pages of the combined PDF):
(1) Text-to-Vector Conversion for Residential Plan Design, pp. 1-4 (AI for Urban Planning)
(2) Fine-tuning LLMs for Tourist Trajectory Prediction, pp. 5-9 (AI for Urban Planning)
(3) Multimodal Data Fusion ... Vulnerable Older Adults, pp. 10-15 (AI for Urban Planning)
(4) Vehicle Routing with Finite Time Horizon using Deep RL, pp. 16-23 (AI for Urban Planning)
(5) Attention Gathers, MLPs Compose: ... Circuit in VideoViT, pp. 24-31 (Deployable AI)
(6) Towards Human-AI Accessibility Mapping in India, pp. 32-40 (AI for Urban Planning)
(7) One Rule to Bring Them All: Transport Connectivity ..., pp. 41-50 (AI for Urban Planning)

Read all 7. Then write a calibration sheet I will use to judge a new paper:

1. For each paper, 3 lines: core claim, the main evidence (tables/figures), and what most
   likely got it accepted.
2. The common bar for acceptance at this kind of workshop: what every accepted paper has
   (framing, novelty statement, evidence level, statistics, length, figures, writing tone).
3. What these papers get away with (weaknesses that did NOT block acceptance).
4. How each paper states its contribution in the abstract and introduction: quote the one
   sentence per paper that best sells its novelty.
5. How they write about their own limitations and negative results (tone, placement, length).
6. Which of the seven are short (about 4 to 5 pages) and how their bar differs from the
   longer ones.
7. What paper (6) does with data from India and paper (5) does with a video model that a
   reviewer valued, since the next paper uses Indian street video and a video world model.

Be concrete, cite page numbers. Do not judge any other paper yet; I will send it next.
```

## STEP 2 (attach the 4-page paper, 0_main.pdf)

```text
Now judge the attached anonymous paper, "FactorJEPA on DENSEWORLD-115k: Can a World Model
Predict Crowded, Chaotic South Asian Streets?" (4 pages plus references), to be submitted to
the AAAI-27 workshop "Human-Centric Agentic Mobility Services" (AgMS;
https://agentic-mobility-workshop.github.io/cfp.html ; OpenReview
AAAI.org/2027/Workshop/Human-Centric_AgMS). Use your Step 1 calibration sheet as the bar,
adjusted for this venue.

Workshop context (summary of the call): agentic AI for vehicles, fleets and traveller
assistants needs human-centred design to be reliable and accountable; agency covers goal
interpretation, social interaction, tool use and ethical reasoning, not only autonomy. Four
themes: (A) agentic vehicles and in-cabin services, including grounding plans in perception
and "world models of traffic scenes"; (B) fleets, transit and traffic, including multi-agent
RL and generative-agent city simulation; (C) travellers, drivers and road users, including
communication with pedestrians and accessibility; (D) trustworthy deployment on real roads,
including "benchmarks beyond driving metrics". Early-stage work, benchmark proposals and
negative results are welcome. Rules: research, position, benchmark, dataset, or systems and
demo papers; up to 4 pages excluding references; AAAI-27 author kit; double-blind; non-archival;
concurrent submissions allowed. The same work is a pending AAAI-27 student abstract and a public
arXiv preprint.
So a reviewer here asks: what does this paper give a mobility agent (a vehicle, a fleet or a
traveller assistant): a world model of traffic scenes, a benchmark, or both; how would an agent
use it; and is the evidence enough for a 4-page paper?

Rules: be specific and honest. Cite page, section, table, figure or quoted sentence for every
point. Do not invent results the paper does not contain. Hard constraint for any suggestion:
it must be doable before Fri Dec 25 2026, 8:59 PM PST, and the paper must stay within 4 pages
excluding references (say what to cut for anything you add). There is NO budget for new
training runs or new human annotation. Allowed: inference and evaluation with the existing
trained checkpoints, re-analysis of saved results (for example a split by source video or by
city), new tables or figures, and rewriting.

PART A. Acceptance probability
- One number (0 to 100%) for acceptance at this workshop, with a one-paragraph reason, and the
  same number for a typical AAAI-27 computer-vision or embodied-AI workshop.
- The 3 things most likely to make a reviewer reject, and the 3 strongest points.

PART B. Ten rubrics
Choose the 10 rubrics that you, as a reviewer for this workshop, judge most important for
this paper (do not use a generic list; justify each choice in one line). Score each out of
10, as a table with columns:
rubric | why it matters here | score | evidence from the paper (page/section) | the one change that adds the most points
Then the total out of 100 and how it compares to the 7 accepted papers (especially the short ones).

PART C. Improvements before the deadline (no new training or annotation)
List up to 5 improvements, ranked by expected score gain. For each: what it shows, what it
needs (existing checkpoints or saved results, rough GPU or CPU hours), where it goes (replace
which element, so the paper stays within 4 pages), which rubric it raises and by how much.
Also list anything that should be CUT to make room.

PART D. Tone audit
Find every line or paragraph that reads as NEGATIVE and self-critical, or as an
ENGINEERING/operations report, instead of a RESEARCH paper that leads with its contribution
and novelty. Output a table:
# | page/section | quoted original (exact) | problem (negative / criticism / engineering-only) | rewrite
The rewrite must be positive, constructive and centred on the contribution, BUT it must keep
every fact and every number true: never remove an interval, never turn a non-significant
difference into a win or a tie, never hide a negative result, never add a claim the paper
does not test. No em dashes. Finish with: the 3 rewrites that matter most, and a rewritten
abstract (same facts, at most the current length) that opens with the contribution.

PART E. Double-blind check
List anything in the PDF (links, dataset names, self-citations, acknowledgements, metadata
wording) that could reveal the authors, and the anonymous replacement for each.
```
