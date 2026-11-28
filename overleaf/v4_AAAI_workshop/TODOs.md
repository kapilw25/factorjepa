# ✅ FactorJEPA TODOs (v1, v3, v4, v5)

Updated Fri Nov 27 2026, 9:22 PM PST. One TODO file for all four FactorJEPA workshop drafts (owner order Fri Nov 27 ~7:00 PM PST), next to [venues.md](venues.md). It replaces `v2_Cog2Soc/TODOs.md`, archived at [legacy/v2_Cog2Soc/TODOs.md](legacy/v2_Cog2Soc/TODOs.md) (owner pick ~7:15 PM PST). Submissions are tracked on the [Workshop Deadlines](https://claude.ai/artifact/FemMusSPv4xxXJBTWGcw6b) page, not here. Sources: the official AAAI-27 reviews ([main track](../feedback_official/v1_AAAI27_main.md), reject; [student abstract](../feedback_official/v3_AAAI27_student_abstract.md), weak accept and borderline) and, per draft, the reviews in `feedback/output/<draft>/by_chatgpt.md`, `by_claude.md` (claude.ai), `by_gemini.md` and `by_claude_fresh.md` (a fresh Claude Code helper). The shared sections live once in [common](common/), so a shared fix reaches all four drafts in one [build](build_all.py). Owner picks Fri Nov 27: draft fixes started once all three judges were in (~6:57 PM PST); a neutral disclosure footnote (~7:00 PM), then, as the student abstract was rejected, a footnote that names only the three concurrent venues, with the arXiv preprint told to the Continual Causality chairs only (~7:20 PM); "Causal L1" renamed "Forward L1", Figure 3 label included (~7:00 PM); v3 and v5 retitled to what Table 2 shows (~7:15 PM); the dataset card's six-author citation removed and its history squashed (~9:20 PM).

**Legend:** 🔴 open · ⏳ queued · 🟡 in progress · ✅ done · 👤 a person does it · ⚪ deferred · ⛔ blocked · 🟢 ChatGPT · 🟠 Claude (claude.ai) · 🔵 Gemini · 🟣 Claude fresh · 📜 official review

**💰 RoI** (what the task buys for acceptance, per hour spent): 🔥 high, it moves a headline result, a lead figure or a reject risk · 🟢 medium, it answers one reviewer ask in a sentence or a table · 🔸 low, polish or operations. Each table lists open rows first, ranked by acceptance gain, then done rows ranked by the RoI they delivered, then deferred rows.

## 🎯 Where each draft stands

| 📄 Draft | ⏰ Due (PST) | 🟢 ChatGPT | 🟠 Claude | 🔵 Gemini | 🟣 Claude fresh |
|---|---|---|---|---|---|
| 👥 [v3 Cog2Soc](v3_Cog2Soc_fixes/0_main.pdf) | 🔴 Mon Nov 30, 3:59 AM | 🔹 68, 65% | 🔹 60, 55% | 🔹 71, 62% | 🔹 51, 40% |
| 📦 [v4 Data2AgentEx](v4_Data2AgentEx/0_main.pdf) | 🟡 Tue Dec 1, 9:00 PM | 🔹 77, 79% | 🔹 61, 70% | 🔹 76, 84% | 🔹 57, 60% |
| 🔁 [v5 Continual Causality](v5_Continual_Causality/0_main.pdf) | 🟡 Wed Dec 2, 4:59 AM | 🔹 60, 46% | 🔹 51, 35% | 🔹 65, 54% | 🔹 47, 30% |
| 🚗 [v1 Human-Centric](v1_Human_Centric/0_main.pdf) | 🟢 Fri Dec 25, 8:59 PM | 🔹 75, 77% | 🔹 53, 60% | 🔹 76, 88% | 🔹 54, 50% |

Scores are out of 100, then the judge's acceptance probability. All four judges scored the drafts at commit 52c2809c, before the shared fixes of ecb38a46. All three owner-run judges name the same top fix for v3 and v4 (the human label check) and for v5 (retention on the Western reference clips).

ELI5: four homework papers, the first due Monday; four teachers graded each.

## 🖥️ GPU tasks

One rented RTX PRO 6000 (96 GB), driven over SSH from this session; the owner rents it. Nothing runs until the box is up, so every ETA below assumes the box by about 10 PM Fri Nov 27. "~" marks an estimate: GPU hours come from the judges' ranges until the first runs give measured speed.

| 🧩 Task | 📄 Draft | ⏰ ETA (PST) | 🚦 Status | 💰 RoI |
|---|---|---|---|---|
| 🖥️ Rent box, paste SSH line | 🔗 all | 👤 Fri Nov 27, now | 🔴 owner, reminder sent 6:43 PM | 🔥 high, every GPU row waits on it |
| 🧰 Box setup, data from HF | 🔗 all | ⏰ ~2 h after the SSH line | ⏳ queued, setup script ready | 🔥 high, unblocks every row below |
| 🎬 Video held-out test, saved 10k models | 🔗 all | ⏰ Sat Nov 28, ~10 AM | ⏳ ~3 GPU h for 1B Ours, RAW, plain, LoRA | 🔥 high, answers the overlap critique |
| 🌍 Western clips: transfer, retention | 🔁 v5, 👥 v3, 🚗 v1 | ⏰ Sat Nov 28, ~6 PM | ⏳ ~4 to 8 GPU h, 1B checkpoints | 🔥 high, v5 top fix, v3 cross-culture |
| 📚 General video retention subset | 🔁 v5 | ⏰ Sun Nov 29, ~6 PM | 🔴 open, 1 to 2k general clips | 🔥 high, v5 forgetting result |
| 🔀 Background swap shortcut test | 🔁 v5 | ⏰ Mon Nov 30, ~6 PM | 🔴 open, ~4 to 8 GPU h | 🟢 medium, v5 causal angle |
| 🎲 Random-view control, 1B·10k | 📦 v4 | ⏰ Tue Dec 1, ~9 AM | 🔴 open, ~12 to 24 GPU h | 🟢 medium, semantic views or masking |
| 🧩 Stage checkpoints, if saved | 🔁 v5 | ⏰ Mon Nov 30, ~noon | 🔴 open, check they exist | 🟢 medium, v5 stream evidence |
| 🚗 Road-user forecasting probe | 🚗 v1 | ⏰ Mon Dec 14 | 🔴 open, ~10 to 30 GPU h | 🔥 high, v1 top fix, mobility metric |
| 🏙️ City held-out retrain, 1B·10k | 🚗 v1 | ⏰ Fri Dec 18 | 🔴 open, ~35 to 90 GPU h | 🟢 medium, deployment generality |
| 🌱 Two more seeds, 1B·10k | 🚗 v1 | ⏰ Mon Dec 21 | 🔴 open, ~12 to 18 GPU h | 🟢 medium, run variance |
| 🛑 Stop box, rotate token | 🔗 all | 👤 after Dec 25 | 🔴 owner | 🔸 low, ops: spend, secret |
| 🏋️ Full fine-tuning at 1B·115k | 🚗 v1 | ⚪ not planned | ⚪ owner choice, 1 GPU box; ~31 GPU h | 🟢 medium, third rival at 115k |

ELI5: the big computer re-tests the saved models on new clips and teaches small probes.

## 💻 Non-GPU tasks

Writing and CPU work on the Mac. Each draft is rebuilt with all gates (pages, anonymity, disclosure, drift, AI signatures) after every change.

| 🧩 Task | 📄 Draft | ⏰ ETA (PST) | 🚦 Status | 💰 RoI |
|---|---|---|---|---|
| 📝 Tell Continual chairs: preprint | 🔁 v5 | 👤 before Wed Dec 2 | 🔴 owner | 🔥 high, originality rule |
| 📊 Density split of the margin | 🔗 all | ⏰ Sat Nov 28, ~6 PM | ⏳ waits box for per-clip data | 🔥 high, tests the crowding premise |
| 📏 115k clustered CIs, raw effects | 🔗 all | ⏰ Sat Nov 28, ~8 PM | ⏳ waits box for per-clip data | 🔥 high, Figure 3 in plain units |
| 🔁 Fresh judge round 2, v3 | 👥 v3 | ⏰ Sun Nov 29, ~8 PM | 🔴 open | 🔥 high, last error catch |
| 🖼️ Blur faces in Figure 1 | 🔗 all | 👤 owner call | ⛔ real frames or generated? | 🟢 medium, responsible use |
| 📈 Horizon curve, saved values | 🚗 v1 | ⏰ Mon Dec 7 | 🔴 open | 🟢 medium, v1 planning horizon |
| ✍️ v1 rewrites, abstract | 🚗 v1 | ⏰ Fri Dec 18 | 🔴 open | 🔥 high, first thing read |
| 🪝 Page-1 hook figure | 👥 v3, 🚗 v1 | ⏰ Sun Nov 29 if time | 🔴 open, needs a real frame | 🟢 medium, page-1 clarity |
| 🔍 Figure 3: 8 pt ticks, top gap | 🔗 all | 👤 owner call | 🔴 owner, audit found 4 old issues | 🔸 low, readability |
| 🔤 Figure 1 labels at 9 pt | 🔗 all | 👤 owner call | 🔴 owner, thickening failed | 🔸 low, readability |
| 📤 Upload as paper.pdf | 🔗 all | 👤 at each upload | 🔴 owner | 🔸 low, file name ties drafts |
| ✍️ v4, v5 rewrites, abstracts | 📦 v4, 🔁 v5 | ✅ Fri Nov 27, 9:32 PM | ✅ judges' rows, no still-lead | 🔥 high, first thing read |
| 📚 Continual, causal positioning | 🔁 v5 | ✅ Fri Nov 27, 9:32 PM | ✅ related work, 5 references | 🔥 high, v5 venue fit |
| 🏷️ Rival C named plainly | 🔗 all | ✅ Fri Nov 27, 9:32 PM | ✅ C is plain continued pretraining | 🔥 high, honest rival list |
| ✍️ v3 rewrites, abstract | 👥 v3 | ✅ Fri Nov 27, 9:25 PM | ✅ abstract, question, Sec 5 | 🔥 high, first thing read |
| 🔎 Dataset card: six names removed | 🔗 all | ✅ Fri Nov 27, 9:22 PM | ✅ owner OK, history squashed | 🔥 high, stopped a blind leak |
| 🔤 Rename Causal L1 to Forward L1 | 🔗 all | ✅ Fri Nov 27, 9:22 PM | ✅ text, tables, Figure 3 label | 🔥 high, removes a causal misreading |
| 📄 References to page 5 | 🔗 all | ✅ Fri Nov 27, 9:22 PM | ✅ page 4 right column freed | 🔥 high, frees a column for results |
| 🫥 Footnote: concurrent venues only | 🔗 all | ✅ Fri Nov 27, 7:22 PM | ✅ gate fails any preprint mention | 🔥 high, stopped a blind leak |
| 🏷️ Retitle v3, v5 to the evidence | 👥 v3, 🔁 v5 | ✅ Fri Nov 27, 7:16 PM | ✅ owner pick | 🔥 high, claim matches Table 2 |
| 🧷 One starting checkpoint stated | 🔗 all | ✅ Fri Nov 27, 7:05 PM | ✅ training clips only | 🔥 high, fairness of the comparison |
| 🎭 Masking called a proxy | 🔗 all | ✅ Fri Nov 27, 7:05 PM | ✅ not real occlusion | 🔥 high, construct validity |
| 🧮 Method text matches code | 🔗 all | ✅ Fri Nov 27, morning | ✅ 3-stage clip curriculum | 🔥 high, answers main-track R2 |
| 🪜 Attribution ladder, Table 2 | 🔗 all | ✅ Fri Nov 27, morning | ✅ paired and clustered CIs | 🔥 high, answers ablation critique |
| 🧾 Corpus composition table | 📦 v4 | ✅ Fri Nov 27, 9:32 PM | ✅ Table 3, from clips.csv | 🟢 medium, v4 documentation |
| ❓ Why rivals win on coherence | 🔗 all | ✅ Fri Nov 27, 9:22 PM | ✅ from saved ladder results | 🟢 medium, student-abstract reviewer ask |
| 📐 Plain diagnostic definitions | 🔗 all | ✅ Fri Nov 27, 9:22 PM | ✅ all 13; Motion cosine corrected | 🟢 medium, readers outside vision |
| 🗂️ Save the judges per draft | 🔗 all | ✅ Fri Nov 27, 6:57 PM | ✅ 16 files, word for word | 🟢 medium, one source per row |
| 🧾 Table 2 caption, one exception | 🔗 all | ✅ Fri Nov 27, 7:05 PM | ✅ Ours minus RAW, 2B | 🟢 medium, clear significance |
| 🔢 Slope gains 6.0%, 2.4% | 🔗 all | ✅ Fri Nov 27, 7:05 PM | ✅ from saved results | 🟢 medium, exact numbers |
| ✍️ Shared tone rewrites | 🔗 all | ✅ Fri Nov 27, 7:05 PM | ✅ every number kept | 🟢 medium, tone |
| 🧱 One shared source, gates | 🔗 all | ✅ Fri Nov 27, morning | ✅ all four PASS | 🟢 medium, no drift |
| 🔤 Thicken Figure 1 labels | 🔗 all | ⚪ undone Fri Nov 27, 5:13 PM | ⚪ failed audit (VM44) | 🔸 low, no visible gain |
| 🔗 Cross-channel leakage number | 🔗 all | ⚪ closed | ⚪ no factorized predictor | 🟢 medium, main-track R1 |
| 🎞️ RGB decoding results | 🔗 all | ⚪ not planned | ⚪ RoI: no room at 4 pages | 🔸 low, student-abstract W4 |

ELI5: fix words, tables and small pictures on the laptop.

## 🧑‍⚖️ Human-annotation tasks

Anything that cannot finish without a human rater lives here and in no other table. Owner pick (Nov 26): agent labels, 100 frames × 3 raters; Claude draws the frames and builds the page, people label.

| 🧩 Task | 📄 Draft | ⏰ ETA (PST) | 🚦 Status | 💰 RoI |
|---|---|---|---|---|
| ✍️ Label 100 frames, 3 raters | 🔗 all | 👤 Sun Nov 29, noon | 🔴 owner, waits frames | 🔥 high, top fix of all 3 judges |
| 📊 Precision, recall, Fleiss κ | 🔗 all | ⏰ Sun Nov 29, ~4 PM | 🔴 waits labels | 🔥 high, label validity in Sec 2 |
| 🖼️ Draw 100 frames, overlays | 🔗 all | ⏰ Sat Nov 28, ~1 AM | ⏳ waits box for masks | 🔥 high, enables the labels |
| 🏷️ Agent Label Check page | 🔗 all | ✅ Fri Nov 27 | ✅ [live](https://claude.ai/artifact/Kur76FrdFrR1NLFUuBNkuE), empty | 🟢 medium, enables the labels |

ELI5: real people look at 100 street pictures and say whether the computer's boxes are right.

## 🔎 Where each row comes from

| 🧩 Row | 🔗 Source |
|---|---|
| 🎬 Video held-out test | 🟠 S1 · 🔵 v4 C2 · 🟢 7.1 · 🟣 v4 |
| 🌍 Western clips | 🟢🟠🔵 v5 C1 · 🟠 S4 |
| 📚 General video retention | 🟠 v5 C1 |
| 🔀 Shortcut test | 🟠 v5 C2 · 🟢 v5 C3 |
| 🎲 Random-view control | 🟢 v4 C3 · 📜 abstract W3 |
| 🧩 Stage checkpoints | 🟢 v5 C2 |
| 🚗 Forecasting probe | 🟢🟠🔵 v1 C1 · 📜 main R1 |
| 🏙️ City held-out retrain | 🟢 v1 C2 · 🟠 v1 C3 |
| 🌱 Seeds | 🔵 v1 C4 · 🟢 0.3 |
| 🏋️ Full fine-tuning 115k | 🟢 v1 C4 · 📜 abstract W2 |
| 🔤 Forward L1 rename | 🟠 S5 · 🟢 5.1 · 📜 main R2 |
| 📄 References to page 5 | 🟠 shared · 🟢 8 · 🟣 SH8 |
| 📝 Tell Continual chairs | 🟢 9 · 🟠 v5 C5 · owner Nov 27 |
| 🏷️ Retitle v3, v5 | 🟠 v3 C2 · 🟢 v5 title |
| 🔎 Dataset card check | 🟢🟠🔵 Part D |
| 🔍 Figure 3 audit | visual audit, Nov 27 |
| 📊 Density split | 🟠 S2 · 🟢 v3 C2 · 🔵 v3 C2 · 📜 main AI |
| 📏 Clustered CIs, raw effects | 🟢 5.2 · 🟣 SH4, SH5 |
| 📐 Diagnostic definitions | 🔵 B10 · 🟠 v3 C5 · 🟣 SH6 |
| ❓ Why rivals win | 📜 abstract R2 · 🔵 v5 B9 |
| 🖼️ Blur faces | 🟠 Part D · owner Nov 26 |
| ✍️ Venue rewrites | 🟢🟠🔵 Part E |
| 📚 Continual positioning | 🟠 v5 C3 · 🔵 v5 C2 |
| 🧾 Composition table | 🟠 v4 C4 · 🔵 v4 B7 |
| 📈 Horizon curve | 🟠 v1 C4 |
| 🪝 Hook figure | owner Nov 26 |
| 📤 paper.pdf upload | 🟠 Part D |
| 🫥 Neutral footnote | 🟢🟠🔵 Part D |
| 🧷 Starting checkpoint | 🟣 SH2 · 🟠 shared |
| 🎭 Masking proxy | 🟢 S16, M7 |
| 🧾 Table 2 caption | 🟢 S19 · 🟣 SH9 |
| 🔢 Slope gains | 🟠 S5 |
| ✍️ Shared tone rewrites | 🟢 S1 to S23 · 🔵 Part E |
| 🧮 Method matches code | 📜 main R2, AI |
| 🪜 Attribution ladder | 📜 main R2 · 📜 abstract W3 |
| 🔗 Leakage number | 📜 main R1 |
| 🎞️ RGB decoding | 📜 abstract W4 |
| ✍️ Human label check | 🟢🟠🔵 v3, v4 C1 |

ELI5: every job on this list was asked for by a reviewer or a judge.
