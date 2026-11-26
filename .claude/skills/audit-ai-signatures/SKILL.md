---
name: audit-ai-signatures
description: Audit a LaTeX paper for prose that reads as machine-written (em-dashes, stock LLM vocabulary such as delve/pivotal/seamless/underscore, Moreover/Furthermore openers, heavy bold and italics, mixed curly and straight quotes) and fix it before any submission, arXiv upload, or hand-off. Use whenever paper prose is written or edited, before calling a paper version final, and when the user asks to audit "AI signatures" or "AI tells".
---

# /audit-ai-signatures - no machine fingerprints in the prose

Reviewers and readers now scan for the habits of LLM-written text. The em-dash is the
loudest one and is banned outright in this project (user rule, 2026-09); the others are
warnings that need a human look, because each has legitimate uses.

## Step 1 - run the scan

```bash
python3 .claude/skills/audit-ai-signatures/check_ai_signatures.py <main.tex>
# e.g. Overleaf_draft/v5_arxiv_v2_SwiPO/0_main.tex, Overleaf_draft/v6_SC4AI27_workshop/0_main.tex
```

It follows every `\input`, drops comments (they never reach the PDF), and prints one ASCII table:

```text
check       | status         | what it means
emdash      | FAIL           | '---' or an em dash in prose; must be zero
phrase: x   | WARN           | stock LLM wording with file:line, most frequent first
connectors  | WARN > 4/1k    | Moreover / Furthermore / Additionally / Notably sentence openers
emphasis    | WARN > 25/1k   | \textbf + \emph + \textit commands per 1,000 words
quotes      | WARN           | curly and straight apostrophes mixed (a paste-from-chat sign)
```

## Step 2 - fix by meaning, never by blind replace

- Em-dash: a comma for an aside, a colon before an explanation or list, parentheses for a
  paired aside, a semicolon between two clauses. In tables, a '---' cell meaning "not
  applicable" becomes `n/a`. Quoted user or model text that is a verbatim transcript keeps
  its punctuation; check the outputs folder before treating a quote as authored.
- Stock phrases: replace with the plain claim ("underscores X" becomes "shows X"), or cut
  the sentence if it only adds emphasis.
- Emphasis: keep bold for terms defined once; remove bold and italics used for stress.
- Fix with exact-substring replacements that must match once each, then rebuild the PDF and
  confirm zero em-dash glyphs in `pdftotext` output.

## Step 3 - re-run until emdash is PASS

Also re-run `~/.claude/skills/paper-consistency` afterwards: a prose fix in one version of a
paper usually needs the same fix in its other versions.
