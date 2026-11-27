# 🗺️ AAAI-27 workshop venues for FactorJEPA

AAAI-27 workshops and bridges with a deadline in the next 7 days (Fri Nov 27 to Thu Dec 3, 2026), as of 2026-11-26,
filtered from the OpenReview "Open for Submissions" list, plus the best fit overall, Human-Centric AgMS (Dec 25).
The table keeps only strong and weak fits, most urgent first; poor fits are one line under it. Beyond AAAI, only A*
conferences and their workshops were checked (owner pick 2026-11-26). Deadlines are the OpenReview ones, read from the
OpenReview API on 2026-11-26; several were extended past the Nov 20 date on the AAAI program page. "Results out" comes
from each call and may slip.

Already submitted: 🟡 AAAI-27 Student Abstract (EasyChair, non-archival; reviews in [../feedback_official/v3_AAAI27_student_abstract.md](../feedback_official/v3_AAAI27_student_abstract.md)). ❌ AAAI-27 main track (Submission 43837), rejected in Phase 1 on 2026-09-24 ([reviews](../feedback_official/v1_AAAI27_main.md)).

## 📋 Summary (most urgent first)

| 🏛️ Venue | 🔗 Links | ⏰ Due (PST) | 📬 Results out | 📄 Our track | 🎯 Why it fits |
|---|---|---|---|---|---|
| 👥 Cog2Soc (bridge) | 🔗 [site](https://cog2soc.github.io/) · [OpenReview](https://openreview.net/group?id=AAAI.org/2027/Bridge/Cog2Soc) | 🔹 Mon Nov 30, 3:59 AM (Nov 29 AoE) | 🔹 Thu Dec 10 | 🔹 ≤4 p excl. refs, [v3 folder](v3_Cog2Soc_fixes/) built | 🟢 best fit this week: collective behaviour, norms and cross-cultural evaluation are named topics, datasets welcome; DENSEWORLD vs the Western reference is a cross-cultural gap |
| 🧪 Data2AgentEx (bridge) | 🔗 [site](https://data2agentex.github.io/AAAI2027/) · [OpenReview](https://openreview.net/group?id=AAAI.org/2027/Bridge/Data2AgentEx) | 🔹 Tue Dec 1, 9:00 PM (OpenReview closes before the site's Dec 1 AoE) | 🔹 Sun Dec 13 | 🔹 short 4 p, [v4 folder](v4_Data2AgentEx/) built | 🟡 street video turned into automatic factor targets is a data to training signal story, but the call is about agent workflows and tool traces, not video |
| 🔁 Continual Causality, Novel track (bridge) | 🔗 [site](https://www.continualcausality.com/) · [OpenReview](https://openreview.net/group?id=AAAI.org/2027/Bridge/Continual_Causality_Novel) | 🔹 Wed Dec 2, 4:59 AM | 🔹 Fri Dec 11 | 🔹 short 4 p, [v5 folder](v5_Continual_Causality/) built | 🟡 world models under distribution shift fit, but "Causal L1" is only a metric name, and the student-abstract reviewers already questioned it |
| 🚗 Human-Centric AgMS (W4), beyond 7 days | 🔗 [site](https://agentic-mobility-workshop.github.io/cfp.html) · [OpenReview](https://openreview.net/group?id=AAAI.org/2027/Workshop/Human-Centric_AgMS) | 🔹 Fri Dec 25, 8:59 PM | 🔹 Sat Jan 2, 2027 | 🔹 ≤4 p excl. refs, [v1 folder](v1_Human_Centric/) built | 🟢 best fit overall: the call names "world models of traffic scenes" and asks for benchmarks; double-blind, non-archival, concurrent work allowed |

Legend: 🟢 strong fit · 🟡 weak fit (drafts built anyway, owner pick 2026-11-27) · 🔹 value from the call

🔴 Checked, poor fit (AAAI-27, due by Dec 3): MakingSense (Nov 27, sensors and wearables), MultiRobust (Nov 28, multi-objective optimisation), StreamingCL (Dec 3, archival single-column format), and 15 more with no video, driving, world-model or behaviour topic: Agentic Labs, A2NetOps, LoyalAgents, LAUT, EvalAIScientists, ANTHROLOGOS, NeSy-FM, ToM4AI, MATH4AI, PGAI4E, and the bridges AI4InclusiveEdu, BRAID, AIMedHealth, TGA, Bench2Evidence.

🔭 Outside AAAI, A* venues only (WACV and other tier-2 venues excluded): nothing due by Dec 3 fits. The in-window calls from other large conferences are the WSDM 2027 workshops Context4Agents and LLM4ECommerce (Dec 1; both on LLM agents, and [CORE 2023](https://portal.core.edu.au/conf-ranks/31) ranks WSDM A, not A*) and ICRA 2027's Community Building Day and Arts in Robotics calls (Nov 30 and Dec 1; not paper tracks for this work). The nearest A* deadline is [RSS 2027](https://roboticsconference.org/) (Robotics: Science and Systems; [CORE](https://portal.core.edu.au/conf-ranks/1709) A* in past rounds), Sat Dec 5, 3:59 AM PST, a full archival robotics paper: 🟡 weak, because FactorJEPA has no robot or control results.

Closed (checked 2026-11-26): GSR-VLM (W18, spatial reasoning in VLMs, Nov 20), ASTAD (W5, spatio-temporal anomaly detection, Nov 20, also bans dual submission), Embodied AI Meta-alignment (W35, Nov 24), Medical World Models (W41, Nov 15), W33 Orchestrating Human-AI to World Models (Nov 20), WMAC (W47, Nov 20).

Dual submission: all venues above are non-archival and review double-blind. Human-Centric allows concurrent work; Cog2Soc and Data2AgentEx do not state a rule; Continual Causality asks that "works must be original" (novel-work track). The OpenReview forms have no field for other submissions, so every draft carries an anonymous page-1 footnote naming the other three AAAI-27 venues, the student abstract and the preprint (owner pick 2026-11-27, `common/disclosure.tex`), and each draft has its own title.

Drafts (Fri Nov 27): one shared source in [common](common/) with venue parts per folder (paper-variants skill); `build_all.py` rebuilds all four and runs the page, anonymity, disclosure, drift and AI-signature gates. [v2 folder](v2_Cog2Soc/) is frozen as the version the judge scored in round 1.

## 👥 Cog2Soc (bridge): Modeling and Simulating Human Behavior in the Age of AI, From Cognition to Society

**About** (summary of the [call](https://cog2soc.github.io/)): brings AI, psychology, sociology, cognitive science, HCI and
computational social science together to study human behaviour from individual cognition to collective dynamics, and to
compare how each field builds and evaluates behavioural models.

**Topics of interest** (paraphrased):
1. Cognitive and behavioural mechanisms: memory, learning, emotion, motivation, bounded rationality, interpretable model discovery.
2. Individual differences, adaptation, and human-AI interaction.
3. Social networks, norms, institutions, cooperation, and collective behaviour.
4. Generative agents, agent-based models, and hybrid simulation.
5. Empirical grounding, calibration, uncertainty, replication, cross-cultural evaluation, and responsible use.

**Format:** research findings, experience reports, methods, theory, datasets, evaluation tools, demonstrations and position papers; up to 4 pages excluding references, AAAI two-column format; double-blind; at least two reviews on relevance, substance and suitability for discussion; non-archival. Each paper explains the human phenomenon, its approach or research question, and its connection to another participating community.

**Our version:** [v2_Cog2Soc/](v2_Cog2Soc/) is v1 reframed around negotiated, norm-governed street behaviour, with "The human phenomenon", "Our approach" and "Connection across communities" paragraphs, a social-force reference, and a discussion section for behaviour modellers. Same title, data and numbers as v1; it also drops the "data effect" claim flagged by student-abstract review W2. Judge prompt: [feedback/input/prompt_v2_Cog2Soc.md](feedback/input/prompt_v2_Cog2Soc.md).

## 🚗 Human-Centric AgMS (W4): Human-Centric Agentic Mobility Services

**About** (summary of the [call](https://agentic-mobility-workshop.github.io/cfp.html)): a one-day workshop arguing that agentic AI for
vehicles, fleets and traveller assistants needs human-centred design to be reliable and accountable. "Agency" is read
broadly: goal interpretation, social interaction, tool use and ethical reasoning, not only autonomy. Early-stage work,
benchmark proposals and negative results are welcome.

**Topics of interest** (paraphrased):
1. Theme A, agentic vehicles and in-cabin services: LLM, VLM and VLA copilots and driving agents; awareness of the cabin and road scene; grounding plans in perception; world models of traffic scenes; memory and tool use at vehicle latency.
2. Theme B, fleets, transit and traffic: robotaxi dispatch; orchestration of ride-hail, transit, charging and logistics; vehicle-infrastructure negotiation; multi-agent RL; generative-agent city simulation.
3. Theme C, travellers, drivers and road users: trip-goal dialogue; driver state; theory of mind; personalisation; trust calibration; accessibility; communication with pedestrians and other road users.
4. Theme D, trustworthy deployment on real roads: benchmarks beyond driving metrics; safety impact of hallucination and goal drift; verification and staged rollout; value alignment; privacy and accountability.

**Format:** research, position, benchmark, dataset, or systems and demo papers; at most 4 pages excluding references, AAAI-27 author kit; double-blind on OpenReview, at least 2 reviews; non-archival; work submitted elsewhere at the same time and recently published work are allowed; at least one author attends in person (Montreal, Feb 22 or 23, 2027); best papers are invited to the journal *Artificial Intelligence for Transportation*. Organisers, and anyone with a personal conflict with one, may not submit. Contact: the lead organiser at McGill (address on the call page).

**Open points (not stated in the call):** whether a public arXiv preprint of the same work is allowed under double-blind review (FactorJEPA is on arXiv), and whether a concurrent AAAI-27 student abstract counts as "submitted elsewhere" (the wording allows it, but it is not named). Both are worth one e-mail to the organisers. Both v1 and v2 dropped the "data effect" claim flagged by student-abstract review W2 (2026-11-26).

Sources: [AAAI-27 workshop list](https://aaai.org/conference/aaai/aaai-27/workshop-program/list/), [bridge list](https://aaai.org/conference/aaai/aaai-27/bridge-program-list/), the OpenReview API (`api2.openreview.net/invitations?id=<group>/-/Submission`), and each venue site linked above.
