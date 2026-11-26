# Judge prompt: FactorJEPA (Cog2Soc, AAAI-27 bridge, 4 pages)

Same STEP 1 as [prompt.md](prompt.md); only STEP 2 changes (venue, deadline, framing).
Use a FRESH chat in each of ChatGPT, Claude and Gemini. Send STEP 1, wait for the answer, then send STEP 2.

- STEP 1 attachment: `prev_acceptances/pdf/aaai26ws_00_combined_7papers.pdf` (on this Mac only; calibration set listed in [prompt.md](prompt.md)).
- STEP 2 attachment: [v2_Cog2Soc/0_main.pdf](https://github.com/kapilw25/factorjepa/blob/main/overleaf/v4_AAAI_workshop/v2_Cog2Soc/0_main.pdf) ([download](https://raw.githubusercontent.com/kapilw25/factorjepa/main/overleaf/v4_AAAI_workshop/v2_Cog2Soc/0_main.pdf)) (the anonymous 4-page paper to submit)

Save each model's answers to `overleaf/v4_AAAI_workshop/feedback/output/v2_Cog2Soc/by_<model>/step1.md` and `step2.md` (`<model>` = chatgpt, claude, gemini).

## STEP 1 (attach the combined PDF of 7 accepted papers)

Copy STEP 1 from [prompt.md](prompt.md) unchanged.

## STEP 2 (attach the 4-page paper, 0_main.pdf)

```text
Now judge the attached anonymous paper, "FactorJEPA on DENSEWORLD-115k: Can a World Model
Predict Crowded, Chaotic South Asian Streets?" (4 pages plus references), to be submitted to
the AAAI-27 Bridge "Modeling and Simulating Human Behavior in the Age of AI: From Cognition to
Society" (Cog2Soc; https://cog2soc.github.io/ ; OpenReview AAAI.org/2027/Bridge/Cog2Soc).
Use your Step 1 calibration sheet as the bar, adjusted for this venue.

Venue context (summary of the call): the bridge brings AI, psychology, sociology, cognitive
science, HCI and computational social science together to study human behaviour from
individual cognition to collective dynamics, and to compare how each field builds and
evaluates behavioural models. Topics: cognitive and behavioural mechanisms; individual
differences and human-AI interaction; social networks, norms, institutions, cooperation and
collective behaviour; generative agents, agent-based models and hybrid simulation; empirical
grounding, calibration, uncertainty, replication, cross-cultural evaluation and responsible
use. Paper types: research findings, experience reports, methods, theory, datasets,
evaluation tools, demonstrations and position papers. Rules: up to 4 pages excluding
references, AAAI two-column format, double-blind, non-archival, at least two reviews on
relevance, technical or conceptual substance, and suitability for discussion. Each paper must
explain the human phenomenon, its approach or research question, and its connection to
another participating community. The same work is a pending AAAI-27 student abstract, is
under review at another non-archival AAAI-27 workshop, and is a public arXiv preprint.
So a reviewer here asks: what human phenomenon does the paper study (collective, negotiated
street behaviour), does the model or dataset tell a behavioural scientist anything about it,
how does it connect to agent-based or social-force modelling, and is the evidence enough for
a 4-page paper?

Rules: be specific and honest. Cite page, section, table, figure or quoted sentence for every
point. Do not invent results the paper does not contain. Hard constraint for any suggestion:
it must be doable before Sun Nov 29 2026, 23:59 AoE (Mon Nov 30, 3:59 AM PST), and the paper
must stay within 4 pages excluding references (say what to cut for anything you add). There
is NO budget for new training runs or new human annotation. Allowed: inference and evaluation
with the existing trained checkpoints, re-analysis of saved results (for example a split by
scene type or by city), new tables or figures, and rewriting.

PART A. Acceptance probability
- One number (0 to 100%) for acceptance at this bridge, with a one-paragraph reason, and the
  same number for a typical AAAI-27 computer-vision or embodied-AI workshop.
- The 3 things most likely to make a reviewer reject, and the 3 strongest points.

PART B. Ten rubrics
Choose the 10 rubrics that you, as a reviewer for this bridge, judge most important for this
paper (do not use a generic list; justify each choice in one line). Score each out of 10, as
a table with columns:
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
