---
name: No hardcoded scalar defaults in src/*.py — YAML is the single source of truth
description: Even "trivial" hyperparameter defaults (patience=5, min_delta=1e-4, lr=1e-5) must live in configs/, never as Python literals — including new code added mid-session
type: feedback
---

# Feedback memory — no hardcoded scalar defaults in `src/*.py`

When adding new tunable knobs to `src/m*.py` or `src/utils/*.py` in this project, every numeric default (patience, min_delta, lr, weight_decay, threshold, etc.) MUST live in the corresponding `configs/train/*.yaml` or `configs/pipeline.yaml`, NOT as a Python literal in the .py file (and not as an `argparse` `default=...` either — argparse default IS a hardcoded value).

## Why

CLAUDE.md (`src/CLAUDE.md`) says "No hardcoded values in Python — YAML or runtime discovery only." User caught me embedding `early_stop_min_delta = 1.0e-4` as a Python literal default while adding early-stopping to `probe_action.py` and rejected the edit with: "as per src/CLAUDE.md, do not set hardcoded values in *.py files, put '1.0e-4' in respectives configs/ file."

The rule applies even to "small" / "obvious" / "trivial-default" values that I'd be tempted to inline for brevity. The user has been bitten by silent-default drift in past iterations and is religious about this.

## How to apply

When adding a new tunable knob:

1. **Add the default to the appropriate YAML**:
   - `configs/train/base_optimization.yaml` for SSL/probe-train shared defaults
   - `configs/pipeline.yaml` for eval-time / mode-agnostic knobs (faiss_k, etc.)
   - Per-config files for opt-in overrides

2. **Read it via direct subscript** (`cfg["section"]["key"]`) — never `cfg.get("key", default)` (also banned per CLAUDE.md §5 "Config .get() ban").

3. **CLI flags should override YAML, but their argparse `default=None`** so an unspecified CLI doesn't shadow the YAML value with `None`.

4. **The Python file should never contain the actual scalar** — `grep` `<file>` for the literal should return zero matches outside docstrings/comments.

## Pattern (the right way)

```python
# In configs/pipeline.yaml:
action_probe_train:
  early_stop_patience:    0           # 0 = OFF
  early_stop_min_delta:   1.0e-4

# In src/probe_action.py:
parser.add_argument("--early-stop-patience", type=int, default=None,
                    help="Default from pipeline.yaml action_probe_train.early_stop_patience")
parser.add_argument("--early-stop-min-delta", type=float, default=None,
                    help="Default from pipeline.yaml action_probe_train.early_stop_min_delta")

# In _train_attentive_classifier:
_aptcfg = get_pipeline_config()["action_probe_train"]
early_stop_patience = (args.early_stop_patience
                       if args.early_stop_patience is not None
                       else _aptcfg["early_stop_patience"])
early_stop_min_delta = (args.early_stop_min_delta
                        if args.early_stop_min_delta is not None
                        else _aptcfg["early_stop_min_delta"])
```

## Quick self-check

After any new knob lands, run:
```bash
# Should return ZERO matches outside docstrings/comments
grep -nE "(early_stop_patience|early_stop_min_delta)\s*=\s*[0-9]" <file>
```

If a literal scalar default is on the LHS or default= side, it's a violation. The exception is when the knob is *intentionally* hardcoded (e.g. `nan_strikes = 0` is a counter, not a configurable knob). Rule of thumb: if you can imagine a future paper-quality run wanting to tune the value, it goes in YAML.
