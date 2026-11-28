> Archived Fri Nov 27 2026 ~7:15 PM PST (owner pick): replaced by the one plan for all 4 drafts, [TODOs.md](../../TODOs.md). Kept as it was, links updated for the new folder.

# ✅ TODOs: Cog2Soc from 51 to 80/100

**Target venue:** 👥 [Cog2Soc (bridge)](https://cog2soc.github.io/) · [OpenReview](https://openreview.net/group?id=AAAI.org/2027/Bridge/Cog2Soc) · ⏰ **Mon Nov 30, 3:59 AM PST** (Sun Nov 29 AoE) · 📬 Thu Dec 10 · 📄 ≤4 p excl. refs, double-blind · 🟢 collective street behaviour

**Built from:** the fresh Claude judge, round 1 on the [v2 paper](../../v2_Cog2Soc/0_main.pdf): **51/100**, acceptance 55% ([review](../../feedback/output/v2_Cog2Soc/by_claude_fresh/round1_step2.md), [calibration](../../feedback/output/v2_Cog2Soc/by_claude_fresh/round1_step1.md)), plus the official reviews: [AAAI-27 main track](../../../feedback_official/v1_AAAI27_main.md) (reject: 5, 2) and [student abstract](../../../feedback_official/v3_AAAI27_student_abstract.md) (weak accept, borderline). Written Fri Nov 27, 2026.

**Working copy:** [v3 fixes paper](../../v3_Cog2Soc_fixes/0_main.tex). v2 stays unchanged as the "before" the judge scored.

**Owner decisions (Nov 26 to 27):**
- **Loop:** focus on Cog2Soc until a fresh Claude judge scores it 80/100, then copy every fix into Human-Centric (due Fri Dec 25, 8:59 PM PST).
- **GPU:** one rented RTX PRO 6000 box, which Claude drives; training and inference allowed, highest-return fixes only.
- **Human check:** agent labels, 100 frames × 3 raters.
- **Page 1:** a new one-column hook figure.
- **Figures and title:** thumbnail faces blurred; Figure 2 shown at one-column width. Title: first kept, then (Nov 27, ~6:30 AM PST) one title per venue so four parallel submissions do not look like one paper sent four times; Cog2Soc's is "Layout, Then Road Users, Then Interactions: Adapting a Video World Model to Dense South Asian Street Behaviour".
- **Disclosure:** an anonymous page-1 footnote in every draft names the other three AAAI-27 venues, the student abstract and the preprint (the OpenReview forms have no field for it).
- **All 4 open venues get a draft** (Nov 27), built with the paper-variants skill from one shared source.
- **RAW result:** reported and reframed.

**Legend:** 🖥️ GPU or box task · 📝 writing or CPU task · ✅ done · ⏳ running · 🕒 queued · 👤 owner action · R1 to R10 = the judge's rubrics below

## 🧮 Judge rubrics (round 1)

| # | Rubric | Now | Target | Moved by |
|---|---|---|---|---|
| R1 | Human phenomenon, stated and observable | 6 | 8 | N4 |
| R2 | Behavioural insight from the results | 3 | 6 | G3, G4, C1 |
| R3 | Bridge to agent-based and social-force modelling | 5 | 7 | N1, C1 |
| R4 | Validity of automatic agent and interaction labels | 3 | 6 | C1 |
| R5 | Cross-cultural comparison design | 4 | 6 | G4, N6 |
| R6 | Controls and attribution of the gain | 5 | 8 | N1 |
| R7 | Statistical rigour and effect sizes | 6 | 8 | N2 |
| R8 | Claims match the evidence | 7 | 9 | N3 |
| R9 | Dataset contribution and responsible use | 6 | 8 | N5 |
| R10 | Clarity and discussion value in four pages | 6 | 8 | N3, N7 |
| | **Total** | **51** | **74** | |

> [!IMPORTANT]
> **80 is reachable only if the behavioural results cooperate.** The Target column adds up to 74. The last 6 points need R2 to reach 8 or more, so the human check (C1) has to show that raters agree with the automatic labels, and the agent-count split (G3) has to show where crowded prediction gets hard. Every score here is a projection from the judge's rubrics, and each result is reported whichever way it comes out.

## 🖥️ GPU and box tasks

| ID | Status | Task | Needs | ETA | Rubric |
|---|---|---|---|---|---|
| **B0** | 👤 | Rent 1× RTX PRO 6000 96 GB, 500 GB disk, paste the SSH line | vast.ai | owner | all below |
| **B1** | 🕒 | Box setup: clone, environment from wheels, download from HF the 10k and 115k test clips, masks and three 1B checkpoints | [setup script](../../../../setup_env_uv.sh) | ~1 to 2 h | all below |
| **G1** | 🕒 | Frames for the human check: 100 anchor frames stratified over the ten scene types, with the automatic boxes, masks and the strongest interaction edge, plus a short clip each | m10 masks on HF | ~1 h CPU | C1 |
| **G2** | 🕒 | Page-1 hook frame: one real crowded clip with its real agent and interaction overlays | same | ~0.5 h | N7 |
| **G3** | 🕒 | Split the 1B·115k per-clip errors by agents per clip and by scene type: does the FactorJEPA margin grow with density? | per-clip errors on HF, agent counts from m10 | ~2 h CPU | R2 +2, R5 +1 |
| **G4** | 🕒 | Run frozen, LoRA and FactorJEPA (1B·115k) on the BDD100K and nuScenes reference clips of Figure 2 | reference clips, 3 checkpoints | ~2 to 4 GPU h | R5 +2, R2 +0.5 |
| **G5** | 🕒 | Video-clustered intervals at 1B·115k for Figure 3 and effects in percent | per-clip arrays on HF | ~1 h CPU | R7 +2 |

**Not run, by owner choice of a 1-GPU box:** full fine-tuning at 1B·115k (~22 h training plus ~9 h evaluation, from iter19 step times). It would answer the main-track W2 and give a third rival at 115k.

## 📝 Writing and CPU tasks

| ID | Status | Task | Where | Rubric |
|---|---|---|---|---|
| **N1** | ✅ | Attribution ladder from saved 10k results: plain retrain, RAW, Ours, at 2B and 1B, paired 95% BCa and video-clustered intervals ([code](../../../../src/utils/paired_ladder.py), [2B](../../common/results/vjepa_2_1_vitG_2B.json), [1B](../../common/results/vjepa_2_1_vitg_1B.json)). RAW is the same staged recipe on raw clips. The staged adaptation carries most of the gain; the layout, agent and interaction clips add robustness to masking and cost motion readout. In v3 as Table 2 | Table 2 in Evaluation | R6 +3, R3 +1 |
| **N0** | ✅ | Method text matches the code (owner pick Fri Nov 27): the papers described a factorized predictor (dictionaries, visibility gate, covariance penalty) that was never implemented; v3 describes the 3-stage layout, agent, interaction clip curriculum that was trained. The false "faces and plates are blurred" sentence is replaced by a responsible-use paragraph | Abstract, Sec 1 to 6 | R8, honesty |
| **N2** | 🌓 | Video-clustered intervals: done for the 10k ladder (127 source videos behind 1,825 test clips; every separation survives); 115k pending (G5) | Protocol, Figure 3 | R7 +2 |
| **N3** | 🌓 | Tone rewrites from the judge's Part D (18 rows), the contribution-first abstract, "the stronger of the two rivals trained at that scale", "consistently" replaced by the exact scales | whole paper | R8 +1, R10 +1 |
| **N4** | 🕒 | Name observable behaviours (yielding, gap acceptance, close passing) and say which ones the data and labels capture | Sec 1 | R1 +2 |
| **N5** | 🕒 | Responsible use: consent, takedown, metadata licence; blur faces in the Figure 1 thumbnails (pixels only, nothing moves) | Release paragraph, Figure 1 | R9 +2 |
| **N6** | 🕒 | Figure 2 at one-column width (same file), clip counts in the caption | Sec 2 | R5 +0.5, room |
| **N7** | 🕒 | Page-1 hook: one-column Figure 1 with the real frame (G2) and a small result panel | page 1 | R10 +1 |
| **N8** | ✅ | PDF metadata without date or time zone (shared preamble; the build gate fails a PDF with a date or author field) | preamble | double-blind |
| **C1** | 🌓 | Human check page ([Agent Label Check](https://claude.ai/artifact/Kur76FrdFrR1NLFUuBNkuE), published empty, frames come from G1; share with Contributor access): 100 frames × 3 raters, missed and false agents, and whether the marked pair adjusts to each other; precision, recall, Fleiss κ | claude.ai page, Sec 2 | R4 +3, R3 +1 |
| **N9** | 🕒 | Build, 4-page check, AI-signature audit, paper-consistency against v1 Human-Centric | v3 folder | all |
| **N10** | 🕒 | "Judge Fixes" before and after page (v2 crops vs v3 crops) | claude.ai page | report |
| **N11** | 🕒 | Fresh judge round 2 on v3; repeat until 80 or the deadline | [judge folder](../../feedback/output/v2_Cog2Soc/by_claude_fresh/) | score |
| **N12** | ✅ | Every fix reaches every draft (Fri Nov 27): shared sections live once in [common](../../common/), each venue folder keeps only title, abstract, introduction and its discussion section; one [build script](../../build_all.py) rebuilds all 4 drafts with page, anonymity, disclosure, drift and AI-signature gates. Human-Centric now carries the corrected method text; Data2AgentEx and Continual Causality drafts are new | [build script](../../build_all.py) | all venues |
| **N14** | 👤 | Figure 1 labels print at 1.4 to 2.4 pt. Thickening in place (owner pick Fri Nov 27, ~4:45 PM PST; [code](../../../../src/utils/fig_teaser_embolden.py)) failed the visual audit: letter holes filled, grey marks on the legend keys, and no visible change at print size ([VM44](../../../../.claude/memory/visual_mistakes/VM44.md)). Undone at ~5:13 PM PST (5-minute default), so Figure 1 is unchanged. Readable labels need a redraw at 9 pt or more ([VM40](../../../../.claude/memory/visual_mistakes/VM40.md)) or the hook figure (N7): owner's call | Figure 1, all drafts | R10, page 1 |
| **N13** | 👤 | Submit on OpenReview: double-blind; disclose the student abstract, the Human-Centric submission and the arXiv preprint | [Cog2Soc OpenReview](https://openreview.net/group?id=AAAI.org/2027/Bridge/Cog2Soc) | submission |
