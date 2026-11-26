# 🎯 Plan — Reproduce v1\_main Figures/Tables **Full-Width** in the v2 Supplement

> ⏰ **Deadline:** ~2 hours &nbsp;·&nbsp; 📄 **Deliverable:** `overleaf/2026_FactorJEPA_AAAI_v2/0_supplement_AAAI.tex`
> 🚫 Main upload is **CLOSED** — we can only touch the **supplement**.

---

## 📋 Task Tracker

| # | 🧩 Task | 📌 Status |
|:-:|---------|:---------:|
| 1 | 🖼️ Copy `demo_cards_teaser.png` from **v1 → v2** `figures/` | ✅ Done |
| 2 | 🧱 Create `A_v1main_floats.tex` — 8 floats, full-width, in v1\_main order | ✅ Done |
| 3 | 🔗 `\input` it at the **TOP** of the supplement + add a section header | ✅ Done |
| 4 | ⚙️ Compile the supplement (`latexmk -pdf`) | ✅ Done |
| 5 | 🔍 Verify **sequence == v1\_main** & each float is **both-column** | ✅ Done |
| 6 | 📸 Report page count + rendered snapshot for your check | ✅ Done |

> **Legend:** ⬜ Pending &nbsp; 🔄 In-progress &nbsp; ✅ Done &nbsp; ⚠️ Blocked

---

## 🧭 Context

- 🧑‍🏫 The professor already submitted the **MAIN** PDF (v2). Its upload deadline **expired** → the main is frozen.
- 😖 In that main, all **results / ablations** are squeezed into **one column** and are hard to read.
- 🎯 Fix: the **supplement** reproduces **every figure/table from v1's main**, in the **same order**, each spanning **BOTH columns** (full width) so they are big and legible.
- 👀 You will **literally check** the sequence + full-width against v1\_main.

---

## ❓ QQ1 — Which supplement base? → **v2** ✅

Build on **v2's supplement** so it **pairs with the submitted v2 main** (same story: `1,000 h · DINOv2 · RGB thread · Limitations`).
🚫 v1's supplement would **contradict** the submitted main (`276 h · DINO+SAM3 · no RGB`).

---

## 🔢 Exact v1\_main Float Sequence to Reproduce

> 📐 Order verified by source line number — **this is what you'll check against.**

| # | 🏷️ Label (v1\_main) | 🖼️ Asset | Now | 🔧 Action |
|:-:|--------------------|----------|-----|-----------|
| 1 | `fig:demo_cards` | `demo_cards_teaser.png` | 1-col | → `figure*` · 📥 **copy file v1→v2** |
| 2 | `fig:denseworld_scene_types` | `denseworld_scene_types_overview.png` | 1-col | → `figure*` |
| 3 | `fig:factorjepa_visual` | `factor_visual.png` | 1-col | → `figure*` |
| 4 | `fig:dino_factor_targets` | `segmented_scene.png` | 1-col | → `figure*` · ⚠️ caption says **"DINO-based"** (clashes w/ v2 DINOv2) |
| 5 | `tab:frozen_scorecard` | native LaTeX `table*` | ✅ 2-col | copy block **verbatim** |
| 6 | `fig:frozen_forest` | `forest_plot_best_ci_paper.pdf` | ✅ `figure*` | keep |
| 7 | `fig:eval_scorecard` | `eval_scorecard_winbars.pdf` | ✅ `figure*` | keep |
| 8 | `fig:scale_replication` | `scale_replication_single.pdf` | 1-col | → `figure*` |

- ✅ Every asset **except** `demo_cards_teaser.png` already lives in `v2/figures/` → **only 1 file copy**.
- ✅ Captions are conflict-free **except item 4** (the ⚠️ flag).

---

## 🛠️ Implementation

1. 📥 **Copy** `…/v1/figures/demo_cards_teaser.png` → `…/v2/figures/`.
2. 🧱 **Create** `…/v2/A_v1main_floats.tex` — the 8 floats in the order above:
   - 📋 Copy each figure/table block **+ caption verbatim** from the v1\_main source.
   - 🖥️ Wrap single-column figures as `\begin{figure*}[t!]` with `\includegraphics[width=\textwidth]` (or `0.95\textwidth`).
   - 🏷️ Give each a **new suffixed label** (e.g. `fig:demo_cards_full`, `tab:frozen_scorecard_full`) → no collisions.
   - 🔗 Neutralize any dangling `\ref` in captions (keep `\cite` — the bib resolves).
   - 🎨 Item 5: lift the whole `\begin{table*}…\end{table*}` block; v2's preamble already defines the `xcolor[table]` / `green!` / `red!` macros it needs.
3. 🔗 **Insert** `\input{A_v1main_floats}` at the **TOP** of the supplement body in `0_supplement_AAAI.tex` (right after `\maketitle`, before `\input{9_discussion}`) with a header:
   `\section{Main-Paper Figures and Tables (Full Size)}`.
4. 🧭 **Preserve order + full width:** place the 8 floats consecutively as `figure*`/`table*`; if LaTeX reorders, force it with `[t!]`/`[p]` or a `\clearpage` between floats.

---

## ⚠️ Assumption (correct me if wrong)

The 8 full-width floats go at the **TOP** of the supplement; the existing **Limitations + Appendix A–E stay below** them (non-destructive).
🗣️ *"start with those only"* → lead with these, keep the rest.

---

## ✅ Verification

- ⚙️ Compile `0_supplement_AAAI.tex` (`latexmk -pdf`).
- 👀 Render first pages → confirm the 8 floats appear in **exactly** v1\_main order and each spans **both columns**.
- 🚫 No `File not found` (esp. the copied teaser) · no duplicate-label errors.
- 📸 Report page count + a rendered snapshot for your **sequence check**.
