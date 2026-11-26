"""Robust probe-history reader (Fix1 #76).

Post-hoc analysis scripts should read `training_summary.json.probe_history`
(canonical: written once, atomically, at end-of-training by m09c) rather
than streaming the live `probe_history.jsonl`. The jsonl is append-only
with per-record fsync but is a SINGLE-WRITER file — any concurrent external
writer (e.g. a backfill script) produces torn records (two JSON dicts on
one line without newline). This happened in v10 post-hoc analysis, 2026-04-20.

Policy: live tooling → training_summary.json. jsonl is a crash-recovery
safety net only, not a public read surface.
"""

from __future__ import annotations

import json
from pathlib import Path


def read_probe_history_robust(training_summary_path: str | Path,
                              fallback_jsonl_path: str | Path | None = None) -> list[dict]:
    """Return probe history as a list of probe-record dicts.

    Prefers the canonical `training_summary.json.probe_history` (single-write,
    atomic). If that file is missing or lacks the key, falls back to parsing
    the jsonl line-by-line with torn-record detection (skips lines where
    json.loads raises, logs count to stderr).

    Raises FileNotFoundError if neither source is available.
    """
    summary_path = Path(training_summary_path)
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        if "probe_history" in summary:
            return summary["probe_history"]

    if fallback_jsonl_path is None:
        raise FileNotFoundError(
            f"{summary_path} missing or has no 'probe_history' key, "
            f"and no fallback_jsonl_path was provided."
        )

    jsonl_path = Path(fallback_jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Neither {summary_path} (with probe_history key) nor "
            f"{jsonl_path} exists."
        )

    records: list[dict] = []
    torn = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                torn += 1
    if torn:
        import sys
        print(
            f"[probe_history] WARN: skipped {torn} torn record(s) in "
            f"{jsonl_path} — external writer race suspected. "
            f"Prefer training_summary.json.probe_history.",
            file=sys.stderr,
        )
    return records
