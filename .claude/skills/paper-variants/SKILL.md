---
name: paper-variants
description: One source, many venue drafts of one paper. Shared sections live once in a common/ folder that every venue draft pulls in with \input; each venue folder keeps only its venue parts (title, abstract, first paragraph, framing, venue switches). One build script rebuilds every unfrozen draft and runs the page, number, drift, anonymity and tone gates per draft, then one fresh checker agent per draft reads its PDF pages. Drafts at or past their deadline are frozen and never rebuilt by a shared edit. Use in any project with several venue drafts of one paper, whenever a judge fix or wording change should reach every draft, when creating a new venue draft, and when the owner says sync, all drafts, common parts, variants or venues.
---

# 🧬 paper-variants: one source, many venue drafts

**Project facts** (workshop folder, builder command, gates, frozen drafts, switches) are in the project's profile:
`.claude/skill-profiles/paper-variants.md`. Read it first. This file holds only the rules, the same in every project.

**Why (measured, Nov 26-27 2026):** judge rewrites had to be typed into two drafts one copy at a time, 4 hours before their deadline;
6 of 9 judge fixes made on one draft had never reached the other two. One shared file would have carried them in one edit.
**Pattern sources:** one source with venue switches ([ISI multi-version LaTeX](https://ant.isi.edu/~calvin/latex-multidoc.htm), CTAN
[optional](https://ctan.org/pkg/optional) and [tagging](https://ctan.org/pkg/tagging)); shared snippets plus a drift check
([Grafana shared content](https://grafana.com/docs/writers-toolkit/write/reuse-content/reuse-shared-content/)).
**Lesson:** when an audit finds a problem in a draft with a deadline, fix it in the same turn and show the proof; an explanation
without the fix is not an answer.

**Legend:** 🟢 shared once · 🟡 venue-only · 🔒 frozen · ✅ check passes · ❌ refused

## 📌 TL;DR

| # | Rule |
|---|---|
| 1 | 🟢 Text that is the same in every draft lives ONCE in `common/` (methods, setup, tables, figures, macros, refs, limits). |
| 2 | 🟡 A venue folder holds only `0_main.tex` with the venue parts plus `venue.tex` (switches). No copy of a common file, ever. |
| 3 | 🟡 A sentence that must differ for one venue goes behind a switch in `common/` (`\ifvenueX ... \fi`), not into a second copy. |
| 4 | 🔒 A draft whose deadline is under 6 h away, or already submitted, gets a `FROZEN` file; the build skips it and keeps its PDF. |
| 5 | ✅ One command rebuilds every unfrozen draft and runs every gate on each; one failing draft does not stop the others; one table at the end. |
| 6 | ✅ Then one fresh checker agent per draft reads only that draft's page pictures and returns PASS/FAIL per format check. |

## 1️⃣ Layout

```
<workshop dir>/                 (the profile names it)
  venues.md                     deadlines, page limits, review mode, status per venue
  common/                       🟢 written once: shared sections, preamble, numbers.tex (generated or macro-only), bib, figs
  v<N>_<Venue>/                 🟡 one folder per venue draft
    0_main.tex                  title, abstract, first paragraph, contribution framing, \input{../common/...}
    venue.tex                   \venueXtrue switches, venue page limit, track
    FROZEN                      🔒 optional: "submitted <date, time>" (build skips the folder)
  build_all.(py|sh)             rebuild every unfrozen draft + gates + summary table
```

| 🧩 Part | 🟢 common/ | 🟡 venue folder |
|---|---|---|
| 📝 Title, abstract, first paragraph | ❌ | ✅ |
| 🎯 Contribution framing for the venue's call | ❌ | ✅ |
| 🔬 Method, setup, tables, figures | ✅ | ❌ |
| 🔢 Numbers | ✅ macro files | ❌ never typed |
| 🧱 Limits, ethics, release, AI-use | ✅ | ❌ |
| 🔀 One sentence that differs for one venue | ✅ behind `\ifvenueX` | ❌ no copy |

## 2️⃣ The build (one command per project, named in the profile)

For every venue folder without `FROZEN`:

1. Compile inside the venue folder (`\input{../common/...}` resolves from there).
2. The project's per-draft gates: page limit (body pages, nothing spills into the reference page), numbers only as macros,
   tone, overclaim, AI signatures, anonymity for double-blind venues.
3. Drift gate: refuse (❌) a venue file whose name exists in `common/` (a stale copy), a 40+ word paragraph identical to one in
   `common/` (pasted instead of `\input`), and a measured number typed in a venue file.
4. One summary table: draft, pages, each gate PASS/FAIL. Exit non-zero if any unfrozen draft fails.

A FROZEN draft is listed as 🔒 with its last PDF's sha256, so a shared edit can never silently change a submitted paper.

## 3️⃣ The checker agent (one per draft, fresh, read-only)

After the build, spawn one agent per rebuilt draft, in parallel, each given ONLY that draft's page pictures (96 dpi) and the
venue's page rule from `venues.md`. Brief:

> Read every page picture of this draft. Report PASS/FAIL with page and position for: body within the page limit; nothing spills
> past the last body page; no overfull line or text off the margin; every figure and table readable at 100% zoom and referenced in
> the text; no orphan heading at a column bottom; anonymous (no names, no affiliations, no identifying links); references start
> after the body; title and abstract match the venue's theme. Do not suggest content changes. Do not edit files.

A FAIL is fixed in the venue folder (cut venue words) or with a switch in `common/`, then the drafts are rebuilt and re-checked.

## 4️⃣ Editing rules

| ⚠️ | Rule |
|---|---|
| 🟢 Fix to a shared sentence | Edit `common/` once, run the build, read every draft's page gate. A draft pushed over its limit is fixed in its venue parts, never by forking the shared file. |
| 🟡 Fix for one venue only | Venue folder, or a switch in `common/`. Say which in the commit message. |
| 🔒 Within 6 h of a deadline | Write `FROZEN` in that draft first. Unfreeze only by owner order. |
| 🔢 Numbers | Only from macro files; a typed measured number in a venue file is a gate FAIL. |
| 🧩 LaTeX switches | A control word eats the space after it: write `\fi{}` and `\nName{}` mid-sentence. The array package does not expand a macro inside a column spec: switch the whole `\begin{tabular}{...}` through a macro. |
| 🧾 Gates are per draft | A gate that passes on one draft says nothing about another; every draft runs every gate. |
| 🗳️ Review mode | Each `venue.tex` records the review mode read from the venue's own submission page, never guessed. |

## 5️⃣ Migrating a project (one time, after its nearest deadline)

1. Freeze every draft due within 6 h and copy its PDF aside.
2. Pick the most-judged draft as the base. Move its non-venue sections into `common/`; replace them with `\input{../common/...}`.
3. For each other draft: diff its sections against `common/`. Identical text becomes an `\input`; a sentence that differs on
   purpose becomes a switch; an accidental difference takes the `common/` text, and the commit says which fix it brings.
4. Rebuild all, run every gate, and compare each draft's pdftotext with its pre-migration PDF: every difference must be intended
   and every number identical. Prove the sync once: a sentinel sentence in one `common/` file must appear in every PDF; a planted
   pasted paragraph, stale copy and typed number must each be refused.
5. Checker agents on every draft; push only when all are PASS.

Related: [audit-judge-pattern](../audit-judge-pattern/SKILL.md), [audit-robotic-tone](../audit-robotic-tone/SKILL.md) (abstracts are
venue files, so both run per venue).
