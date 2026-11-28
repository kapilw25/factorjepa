---
name: audit-robotic-tone
description: Manual only, never run on your own - the owner calls /audit-robotic-tone <venue>/0_main.tex [--all]. Lists every sentence of a paper's abstract (and body with --all) that reads as an engineering report instead of a researcher excited about a finding, says why line by line, then gives a fresh judge with veto power one question. Reports only; it changes nothing unless the owner says so.
---

# /audit-robotic-tone - find the lines that read like a results log

**Project facts** (where to run, the codebase's jargon words, words that are research words here, names whose digits are not
numbers, the build) are in the project's profile: `.claude/skill-profiles/audit-robotic-tone.md`. The script reads its ```json
block on its own. This file holds only the method.

**Why (owner, Sat Nov 28 2026 ~4:20 AM PST, after two abstracts shipped as number dumps):** "build SKILL, which I know will NOT stop
you, but if call it manually will SPIT out your ROBOTIC tones and filter each line which needs to be replaced"; the veto: "does this
sound like a researcher excited about a finding?". Root cause: Claude writes defensively to survive the next judge (every claim gets
a number, an interval and a hedge), and no other skill measures voice.

## Step 1 - the counted symptoms (script)

```bash
# from the folder the profile names; activate the project venv first if a hook requires it
python3 ${CLAUDE_SKILL_DIR}/check_tone.py <venue>/0_main.tex          # abstract only
python3 ${CLAUDE_SKILL_DIR}/check_tone.py <venue>/0_main.tex --all    # plus the body, from the built PDF
```

The abstract comes from the .tex with every macro number filled in (`\input` paths resolve from the profile's `macro_roots`); the
body comes from the PDF next to it (table rows skipped). A `NOTE:` line lists any macro it could not fill. Exit 1 = ROBOTIC.

| Level | Signal | Fires when |
|---|---|---|
| abstract | number load | more than `--max-abstract-numbers` (default 3) numbers in the whole abstract |
| abstract | spec opener | sentence 1 holds a number or two acronyms instead of a problem a reader cares about |
| abstract | no finding | sentences 1-3 never state a finding as one (we show, we find, surprisingly, only two ...) |
| abstract | number ending | the last sentence ends on a number without saying why it matters |
| abstract | hedging | 4 or more hedges (no resolved, at most, not yet, uncorrected, descriptive ...) |
| sentence | number dump | 2+ numbers (abstract) or 3+ (body) |
| sentence | interval in prose | [a, b], "95% interval", "(a to b)" |
| sentence | parenthesis stack | 2+ parentheses |
| sentence | too long | more than 35 words |
| sentence | semicolon chain | 2+ semicolons |
| sentence | hedge stack | 2+ hedges |
| sentence | engineering words | 2+ of the base list plus the profile's `engineering_add`, minus its `engineering_drop` |
| sentence | acronym soup | 4+ acronyms |
| sentence | telegraph style | 2+ colons |

## Step 2 - the veto (a fresh judge, one question)

Spawn ONE fresh agent (Agent tool, general-purpose). Give it ONLY the abstract text copied from the PDF and this prompt, word for
word. Never add Claude's opinion, the script output, the judges' earlier reviews or the paper's history: a judge handed a conclusion
returns it.

> You are a senior reviewer at a workshop and you have read only this abstract. Answer first, on its own line, YES or NO: does this
> sound like a researcher excited about a finding? Then quote every sentence that sounds like an engineering report, each with the
> reason in at most 12 words. Then name the at most 3 numbers you would keep, and the one sentence a reader should remember.

A NO is a veto: the abstract is not done, whatever the script says. A YES with a ROBOTIC script verdict still needs the flagged
lines fixed. With `--all`, run the judge on the introduction's first paragraph as well.

## Step 3 - one table to the owner

| # | Sentence (quoted) | Script signals | Judge says | Replacement |
|---|---|---|---|---|

Rules for every replacement:
- Story order: the problem a reader cares about, then the finding in words, then why it matters. At most 3 numbers in the abstract.
- Keep every claim true. A caveat or interval the evidence needs moves to the body; it is never deleted.
- Active voice with an agent: "we show", "we find". One job per sentence.
- No new claim, no number the body does not already carry. Numbers stay macros.
- A replacement must still pass the project's gates (profile of [paper-variants](../paper-variants/SKILL.md)): page limit, numbers,
  anonymity, AI signatures.

## Step 4 - stop

Report the table and stop. Apply a replacement only when the owner says which ones. This skill cannot make Claude write with
enthusiasm; it only shows where it did not, and the judge's NO (or the owner's) is the veto.

Sources: Mensh and Kording 2017, Ten simple rules for structuring papers,
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619; Nanda 2025, Highly Opinionated Advice on How to
Write ML Papers, https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers.
Related: [audit-judge-pattern](../audit-judge-pattern/SKILL.md) (judge patterns, J1-J6).
