---
name: audit-judge-pattern
description: Read every AI judge's feedback on a paper (step2.md files by Claude, ChatGPT, Gemini), extract the patterns the judges repeat (contribution first, 2-3 findings, implication close, no number list, no definitions or confessions in the abstract, judge "Before" sentences still in the paper), check the current PDF against them with judge_pattern_check.py, and FIX every finding in the same turn with wording that also passes the project's gates. Use after any judge round, before any submission, whenever a draft is rebuilt near a deadline, on /audit-judge-pattern, and when the owner says feedback, judges, pattern, engineering report, boring, robotic, tone, contribution or novelty.
---

# 🧭 /audit-judge-pattern: the judges' pattern, checked and fixed

**Project facts** (where the judge files are, the run command, the build, the hard limits a rewrite must keep) are in the
project's profile: `.claude/skill-profiles/audit-judge-pattern.md`. Read it first. This file holds only the method.

**Why (measured, Nov 27 2026):** two abstracts kept a 25-28 number list and a term definition for 2 days while all 3 judges' top
rewrite was the abstract; a third paper's judges named the same pattern ("a contribution or a lab report", 4/10). The other audit
skills score pages against a fixed checklist; none read the judges' files.

**Legend:** ✅ pass · ❌ fail, fixed in the same turn · 📝 decision row with reason · 🔁 rebuild

## 📌 TL;DR

| # | Step |
|---|---|
| 1 | 📥 Collect every `step2.md` of the draft's judges (all rounds) and their "rewrites that matter most" / "Before / After" parts. |
| 2 | 🧪 Run `python3 ${CLAUDE_SKILL_DIR}/judge_pattern_check.py <paper.pdf> <step2.md ...>` from the folder the profile names (`--selftest` gives 3 of 3). |
| 3 | ❌ Each J1-J5 FAIL is rewritten NOW, in the venue's own theme, every number from the macro files, then 🔁 rebuild and rerun every gate. |
| 4 | 📝 Each J6 line (a judge's "Before" sentence still in the paper) is rewritten, or gets a decision row in the draft's TODOs with its reason. |
| 5 | 📸 Screenshot every changed spot and send it; push. Explaining without fixing is not done. |

The script finds the abstract by its first five words from the `.tex` next to the PDF (two-column text order can put other text
right after the "Abstract" heading), and stops at the profile's `abstract_end`.

## 🧹 Part D sweep: every row, not only the top items

- **What went wrong once:** each judge's step2.md has a Part D tone table (# | page | quoted original | problem | rewrite). Only
  the top items were applied; 21 judge-quoted sentences stayed word for word in two drafts.
- **The rule:** after every judge round, walk EVERY Part D row of EVERY judge. For each row whose quoted original is still in the
  PDF, apply the rewrite (in `common/` when shared, so all drafts get it), keeping every number and fact, or write a decision row
  with its reason. Done when J6 lists only rows with a decision.
- **Pinned sentences:** if the project pins reviewer-answer sentences (a coverage ledger), a judge rewrite of one needs the owner's yes.
- J6 prefix matching ignores punctuation, so a rewrite that keeps the judge's opening words can still be listed.

## 1️⃣ The patterns

| Code | Pattern the judges repeat | Check |
|---|---|---|
| 🚪 J1 | the work is presented in the first 3 sentences | "We present", "we introduce", or a profile `work_names` name with a verb |
| 🔢 J2 | findings, not a number list | no 3 number-sentences in a row; at most 10 numbers per 100 words |
| 📖 J3 | no definitions in the abstract | no "is one model", "means that", "is defined as" |
| 🏁 J4 | close on what a reader should do | last sentence: should, points to, suggests, guides, so that |
| 🙅 J5 | never close on a confession or a tie | last sentence: no "not tested", "unverifiable", "tie" |
| 🧾 J6 | the judges' own "Before" sentences | still word for word in the PDF = rewrite or decision row |

## 2️⃣ Rewriting so the gates pass too

- 🧭 Sentence 1 says why the problem matters, sentence 2 is "We present ...".
- 🎯 Pick the 2-3 findings that fit the venue's call, not the judges' venue.
- 🛡️ Keep the project's hard limits (profile): "consistent with", never "confirming"; pinned sentences stay; page limit; anonymity.
- 🔢 Numbers only from macros; a new number needs its own source.

## 3️⃣ The agent brief (fresh, read-only, when a second opinion is wanted)

> Read these judge files: <paths>. List the patterns at least 2 judges repeat about framing, tone and structure (quote each). Then
> read this paper's abstract, introduction's first paragraph and conclusion from the page pictures <paths>. For each pattern: PASS
> or FAIL with the quoted sentence, and one rewrite that keeps every number. Do not edit files.

Related: [audit-robotic-tone](../audit-robotic-tone/SKILL.md) (voice, manual only), [paper-variants](../paper-variants/SKILL.md)
(abstracts are venue files; shared fixes go in `common/`).
