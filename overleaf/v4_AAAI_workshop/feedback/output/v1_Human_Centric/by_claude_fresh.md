> Fresh Claude judge review of the [v1_Human_Centric](../../../v1_Human_Centric/0_main.pdf) draft for Human-Centric AgMS, from the judge prompt [prompt.md](../../input/prompt.md). Judge: Claude (fresh helper, no chat context), run Fri Nov 27 2026, inputs = GitHub main at commit 52c2809c. Saved per venue Fri Nov 27 2026 ~6:47 PM PST from the full answers in [step1.md](../v3_open_venues/by_claude_fresh/step1.md), [step2.md](../v3_open_venues/by_claude_fresh/step2.md) and [step3.md](../v3_open_venues/by_claude_fresh/step3.md), which stay unchanged. This file keeps the v1 paragraph of Step 2 and, from Step 3, the shared findings, the v1 section and the ranking; Step 1 (the calibration sheet, shared by all 4) stays in step1.md. The other drafts' reviews are in [v3_Cog2Soc_fixes](../v3_Cog2Soc_fixes/by_claude_fresh.md), [v4_Data2AgentEx](../v4_Data2AgentEx/by_claude_fresh.md), [v5_Continual_Causality](../v5_Continual_Causality/by_claude_fresh.md). Text is word for word.

# STEP 2: What each venue's reviewers will look for first

**Venue 4, Human-Centric AgMS (workshop, due Fri Dec 25, 8:59 PM PST).** Reviewers will first ask where the human and the mobility service are: does the work help an agentic vehicle, fleet or traveller assistant perceive or predict road users, ground plans in perception, or build "world models of traffic scenes", and is it evaluated with "benchmarks beyond driving metrics"? Calibrated on Group D, they reward real road data, an interpretable metric tied to safety or behaviour of road users (pedestrians, two-wheelers, mixed traffic), and a deployment or accountability story. Early-stage work and negative results are welcome, so honesty about scope is safe, but the link to an agent that acts on roads must be concrete rather than implied.

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

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
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

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
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

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
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

<!-- joined-words: off --> <!-- why: verbatim fresh Claude judge review text, kept word for word -->
| Rank | Draft and venue | Acceptance now | Single most valuable change before its deadline |
|---|---|---|---|
| 1 | v4, Data2AgentEx (Dec 1) | 60% | Retrain plain, RAW and Ours at 1B·10k on a video-disjoint split (about 18 to 36 GPU h) and put the result in Table 2 and Sec 5, so the bridge's "training-test overlap" question gets a measured answer |
| 2 | v1, Human-Centric AgMS (Dec 25) | 50% | Add a road-user forecasting probe (occupancy and path entry 1 to 2 s ahead, frozen embeddings, paired intervals) on a video-disjoint split, so the paper carries a mobility metric |
| 3 | v3, Cog2Soc (Nov 29) | 40% | Split the saved per-clip results by road users per clip (CPU only) and lead Sec 5 with how the gain changes with crowding, together with the 100-frame, 3-rater label check |
| 4 | v5, Continual Causality (Dec 1) | 30% | Score all saved checkpoints on the BDD100K and nuScenes reference clips and a general-video set (inference, about 6 to 12 GPU h) to report retention against gain, and replace "We make no causal claim" with the interventional test of Part C item 2 |

Changes that help all four at once (shared sections): state rival starting checkpoints and evaluation targets (SH2, SH3), source-video clustered intervals at 115k (SH4), plain deltas next to interval widths (SH5), one-line diagnostic definitions and the missing citations (SH6), real input frames in Fig 1 and a consistent Fig 3 legend (SH7), and the human label check. Because these edit shared files, do them first, before the Cog2Soc deadline, so every later venue inherits them.
