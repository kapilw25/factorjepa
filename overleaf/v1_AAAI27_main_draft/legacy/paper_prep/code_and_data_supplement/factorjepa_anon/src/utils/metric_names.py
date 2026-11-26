#!/usr/bin/env python3
"""Single accessor for metric DISPLAY names (configs/metric_names.json).

WHY: the human metric names ("free-running exposure-bias gap", "future-frame MSE", ...)
used to be re-typed in FOUR separate lists inside src/m13_eval_plot.py alone
(_VALIDITY_PLAIN, _MW_EVAL_METRICS, _XB_METRICS, _XB_ALL15) plus the shell wrappers.
They drifted: the forest plot said "Teacher-free gap" while the scorecard said
"free-running exposure-bias gap" for the SAME metric (2026-07-08). This module is the
ONE place every consumer reads them — python via `import`, bash via the CLI below —
so a rename in configs/metric_names.json propagates to every plot + doc at once.

Python:
    from utils.metric_names import names, name, direction, group
    NAMES = names()                      # {key: full display name} — the tick-label dict
    lbl   = name("teacher_free")         # "free-running exposure-bias gap"

Bash (run_eval.sh / plot wrappers):
    python src/utils/metric_names.py name teacher_free      # -> free-running exposure-bias gap
    python src/utils/metric_names.py dir  teacher_free      # -> lower
    python src/utils/metric_names.py group teacher_free     # -> predictor
    python src/utils/metric_names.py names                  # key<TAB>name, one per line
"""
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
# THE canonical registry path (mirrors utils.arm_registry's REGISTRY_PATH). An env override
# keeps it testable without leaking a hardcoded data path into a GPU/plot run.
REGISTRY_PATH = Path(os.environ.get("METRIC_NAMES_PATH", _REPO / "configs" / "metric_names.json"))


def _load() -> dict:
    if not REGISTRY_PATH.exists():
        sys.exit(f"FATAL: metric-name registry missing: {REGISTRY_PATH}")
    metrics = (json.loads(REGISTRY_PATH.read_text()) or {}).get("metrics")
    if not metrics:
        sys.exit(f"FATAL: metric-name registry {REGISTRY_PATH} has no 'metrics' block")
    return metrics


def names() -> dict:
    """{key: full display name} — THE single source for every figure tick label / panel title / prose name."""
    return {k: r["name"] for k, r in _load().items()}


def ordered_keys() -> list:
    """Every metric key in canonical DISPLAY order (== the scorecard panel order + the scale-grid order).
    The plot code iterates THIS instead of re-typing an ordered (key, dir) list."""
    return list(_load().keys())


def forest_keys() -> list:
    """The keys shown as forest-plot rows (metrics with forest=true) in canonical order — 13 of 15
    (tax + order are omitted). FAIL LOUD (KeyError) if any entry lacks the required 'forest' flag."""
    return [k for k, r in _load().items() if r["forest"]]


def name(key: str) -> str:
    """Full display name for one metric key. FAIL LOUD on an unknown key (no silent fallback)."""
    rec = _load().get(key)
    if rec is None:
        sys.exit(f"FATAL: metric key {key!r} not in {REGISTRY_PATH} (keys: {sorted(_load())})")
    return rec["name"]


def direction(key: str) -> str:
    """better-direction: 'higher' | 'lower' | 'signed'. FAIL LOUD via KeyError on an unknown key."""
    return _load()[key]["dir"]


def group(key: str) -> str:
    """umbrella group: 'head/probe' | 'predictor' | 'encoder-temporal'."""
    return _load()[key]["group"]


def meta(key: str) -> dict:
    """Full record {name, dir, group, plain} for one key (copy — callers must not mutate the cache)."""
    return dict(_load()[key])


def _cli(argv):
    if not argv:
        sys.exit("usage: metric_names.py {name|dir|group|plain} <key> | {names|ordered-keys|forest-keys}")
    cmd, *rest = argv
    if cmd == "names":
        for k, r in _load().items():
            print(f"{k}\t{r['name']}")
    elif cmd == "ordered-keys":
        print(" ".join(ordered_keys()))
    elif cmd == "forest-keys":
        print(" ".join(forest_keys()))
    elif cmd in ("name", "dir", "group", "plain"):
        if not rest:
            sys.exit(f"FATAL: metric_names.py {cmd} needs a <key>")
        field = {"name": "name", "dir": "dir", "group": "group", "plain": "plain"}[cmd]
        rec = _load().get(rest[0])
        if rec is None:
            sys.exit(f"FATAL: metric key {rest[0]!r} not in {REGISTRY_PATH}")
        print(rec[field])
    else:
        sys.exit(f"FATAL: unknown metric_names command '{cmd}'")


if __name__ == "__main__":
    _cli(sys.argv[1:])
