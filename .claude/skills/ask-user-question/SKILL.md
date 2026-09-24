---
name: ask-user-question
description: When confused or blocked on a decision that is genuinely the user's to make (target venue, scope, which of several valid designs, a trade-off between cost/quality/time), FIRST run WebSearch to ground the options in current facts, THEN call the AskUserQuestion tool with 4 options (1 marked Recommended) so the user picks with one click. Use instead of guessing, and instead of asking open-ended questions in chat.
allowed-tools: WebSearch, WebFetch, AskUserQuestion, Read, Grep, Glob, Bash
argument-hint: <what you are unsure about>
---

# ask-user-question — search first, then ask with options

The user wants to be ASKED, not guessed at, when a real decision is theirs.
But a bare "what do you want?" wastes a round trip. So: research first, then
hand them a short menu with a clear recommendation they can accept in one click.

## When to use it

```text
┌──────────────────────────────────────────────┬──────────────────────────────────┐
│ USE IT                                       │ DO NOT USE IT                    │
├──────────────────────────────────────────────┼──────────────────────────────────┤
│ Two+ valid paths, and the choice changes     │ The answer is in the repo, a     │
│ what you build next                          │ config, git, or a doc: go read it│
│ Target / venue / deadline / page limit       │ A convention already decides it  │
│ Scope: how much to cut, keep, or rewrite     │ (house style, CLAUDE.md, memory) │
│ Trade-offs: speed vs quality vs cost         │ "Is my plan OK?" / "Proceed?"    │
│ Anything destructive or hard to reverse      │ (use plan mode instead)          │
│ You caught yourself about to GUESS           │ Pure facts you can WebSearch     │
└──────────────────────────────────────────────┴──────────────────────────────────┘
```

## Step 1 — WebSearch before asking (mandatory)

Options must be grounded in what is true TODAY, not in memory.

- Run 1-3 `WebSearch` queries on the facts the options depend on: deadlines,
  page limits, API/library versions, current best practice, pricing.
- `WebFetch` the primary source when a search snippet is not enough.
- Also read the local state the options depend on (repo files, git, configs).
- A search that kills an option is a win: drop options that are already
  impossible (e.g. a deadline that has passed) instead of offering them.

## Step 2 — Show context in chat (max 5 lines), then ask

Before the tool call, print what is ambiguous and the key facts you found,
each fact with its source as a markdown link. Plain English, no jargon.

## Step 3 — Call AskUserQuestion

```text
┌───────────────────┬─────────────────────────────────────────────────────────┐
│ field             │ rule                                                    │
├───────────────────┼─────────────────────────────────────────────────────────┤
│ options           │ exactly 4. The tool caps options at 4 and auto-adds     │
│                   │ "Other" (free text) → the user sees 5 choices.          │
│                   │ Never pass 5: the call is rejected.                     │
│ recommended       │ put it FIRST and end its label with " (Recommended)".   │
│                   │ Exactly one per question. Pick it from the evidence.    │
│ label             │ 1-5 words, the choice itself                            │
│ description       │ the consequence: what you will build, what it costs,    │
│                   │ the deadline or fact behind it. Why the recommended one │
│                   │ wins goes in ITS description.                           │
│ header            │ max 12 chars (e.g. "Target", "Scope", "Figure")         │
│ questions         │ 1-4 per call. Batch related decisions into one call.    │
│ multiSelect       │ true only when choices are not mutually exclusive       │
│ preview           │ for layouts / code / diagrams the user must compare     │
│                   │ side by side (single-select only)                       │
└───────────────────┴─────────────────────────────────────────────────────────┘
```

Rules for good options:

- Mutually exclusive, and each one leads to a DIFFERENT next action.
- Cover the realistic space; the 4th slot can be the conservative / minimal path.
- No option that search already proved impossible.
- Never ask what you can find out yourself.

## Step 4 — After the answer

- Act on it immediately. Do not re-ask the same thing.
- "Other" text is the user's own words: follow it literally.
- If the answer opens a NEW real decision, repeat Steps 1-3 for that one only.

## Example (from this repo)

Ambiguity: "create v3 of the AAAI paper" but v2 is an 8-page main-track draft.

1. WebSearch → AAAI-27 main track closed Jul 28, 2026; Student Abstract open,
   due Sep 28, 2026, 2 pages INCLUDING references, camera-ready style.
2. Context line + source links in chat.
3. AskUserQuestion, header "Target":
   - Student Abstract (2pp) (Recommended) — only open AAAI-27 track, 8 days left
   - Kit-clean v2 (8pp) — keep the format, fix rule violations only
   - Camera-ready main track — only if v2 was accepted
   - Workshop paper — format varies per workshop
   (+ automatic "Other")
