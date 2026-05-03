# FactorJEPA — Claude Code session-onboarding memory

> Fresh Claude Code sessions on a new GPU instance: read this file FIRST before any code action. Index is one-line-per-entry; click into the linked .md for full detail.

## Project pulse

- [project_pulse.md](project_pulse.md) — current iter (iter13), where the work is, what's done, what's next
- [hardware_split.md](hardware_split.md) — what runs on 24 GB vs 96 GB (ironclad — confirmed empirically 2026-05-03)
- [next_actions.md](next_actions.md) — concrete commands to run next on a fresh 96 GB instance

## Architecture

- [pipeline_layout.md](pipeline_layout.md) — module map (m04 → m11, scripts/, utils/), what consumes what
- [iter13_multi_task.md](iter13_multi_task.md) — multi-task probe-loss wiring (5 helpers + 16-dim taxonomy + 4-encoder eval) — the iter13 pivot
- [config_schema.md](config_schema.md) — per-mode YAML flatten convention, opt-in pattern, ckpt-schema dispatch

## Operating notes (copy-paste safe)

- [bug_log.md](bug_log.md) — known bugs with their fixes (A/B/R8/OOM-frag/eval-ckpt-schema/Stage-8/plot-NaN)
- [conventions.md](conventions.md) — CLAUDE.md's load-bearing rules condensed (no hardcoded defaults, fail-hard, cache-policy contract, semicolon-not-&&)

## Active feedback memories

- [feedback_no_hardcoded_defaults.md](feedback_no_hardcoded_defaults.md) — every numeric default lives in YAML, not as Python literal or argparse default
