"""Cache-policy gate for all GPU .py files.

Context — the v1→v10 paired_eval cycle lost ~10 h rebuilding the same 9,297-clip
frozen embeddings because two sides independently destroyed durable state:

    1. m05 unlinked its .m05_checkpoint_*.npz after saving final .npy
       (treated .npy as durable).
    2. run_paired_eval_10k.sh L85-88 wiped .npy (treated .npy as ephemeral).

Net effect: no durable state survived a round-trip. This module is the
user-authored permission blocker — every destructive delete inside a .py must
route through guarded_delete(path, policy), which refuses to delete unless
policy=2 ("recompute") — i.e. the user explicitly typed 2 into the .sh prompt.
Default policy=1 ("keep") short-circuits ALL deletions with a fail-loud log.

UX — one-digit prompt (per user request):
    1 = keep (do not delete)     <-- default if user just presses Enter
    2 = recompute (authorize delete)

Orchestrator pattern (scripts/*.sh):
    read -p "m05_frozen cache [1=keep / 2=recompute] (Enter=1): " P
    P_M05_FROZEN="${P:-1}"
    python -u src/m05_vjepa_embed.py ... --cache-policy "$P_M05_FROZEN"

Python pattern (src/m*.py):
    from utils.cache_policy import add_cache_policy_arg, guarded_delete
    add_cache_policy_arg(parser)          # argparse wiring
    ...
    guarded_delete(ckpt_file, args.cache_policy, label="m05 checkpoint")
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional, Union

# Accept both digit (user-facing) and word (back-compat / self-documenting code).
_KEEP = {"1", "keep"}
_RECOMPUTE = {"2", "recompute"}
_VALID = sorted(_KEEP | _RECOMPUTE)


def add_cache_policy_arg(parser) -> None:
    """Register --cache-policy on an argparse.ArgumentParser.

    Default is None so each .py main() can detect "user did not pass the flag"
    and prompt interactively via input(). Per src/CLAUDE.md DELETE PROTECTION:
    shells stay THIN (no read -p), .py owns the prompt. Overnight/tmux runs
    bypass the prompt by passing `--cache-policy 1|2` or setting env var
    `CACHE_POLICY_ALL=1|2`.
    """
    parser.add_argument(
        "--cache-policy",
        type=str,
        choices=_VALID,
        default=None,
        help=(
            "1 = keep (preserve cached checkpoints/intermediates; delete call-sites "
            "no-op with a log line). 2 = recompute (authorize destructive deletes "
            "via guarded_delete). If omitted: each .py prompts via input() in TTY, "
            "honors CACHE_POLICY_ALL env var, or falls back to '1' in non-TTY. "
            "Word aliases keep/recompute also accepted."
        ),
    )


def resolve_cache_policy_interactive(value: Optional[str]) -> str:
    """Resolve --cache-policy after parse_args. Returns one of '1','2','keep','recompute'.

    Call from each m*.py main() right after parse_args:
        args.cache_policy = resolve_cache_policy_interactive(args.cache_policy)

    Resolution order (high → low priority):
        1. CLI flag (`value`) — if not None, return as-is (post-validate).
        2. CACHE_POLICY_ALL env var — FATAL if set to invalid value (FAIL LOUD).
        3. input() TTY prompt — only when stdin is a TTY.
        4. '1' (keep) — non-TTY silent fallback (overnight/CI/subprocess).

    EOFError on input() (e.g. Ctrl-D) is caught → silent fallback to '1'.
    """
    if value is not None:
        _validate(value)
        return value
    env = os.environ.get("CACHE_POLICY_ALL", "").strip()
    if env:
        if env not in _VALID:
            print(f"FATAL: invalid CACHE_POLICY_ALL={env!r} (must be one of {_VALID})")
            sys.exit(1)
        return env
    if not sys.stdin.isatty():
        return "1"
    try:
        ans = input("  [cache-policy] [1=keep / 2=recompute] (Enter=1): ").strip()
    except EOFError:
        ans = ""
    return ans if ans in _VALID else "1"


def is_recompute(policy: str) -> bool:
    """True iff the user explicitly authorized recompute (typed 2)."""
    _validate(policy)
    return policy in _RECOMPUTE


def guarded_delete(path: Union[str, Path], policy: str, label: str = "cache") -> bool:
    """Delete path only if policy=2 (recompute); else skip with a log line.

    Accepts files, directories, and missing paths (no-op). Returns True iff a
    delete actually happened.
    """
    _validate(policy)
    p = Path(path)
    if not p.exists():
        return False
    if policy in _KEEP:
        print(f"  [cache-policy=1/keep] preserved {label}: {p} — "
              f"rerun with --cache-policy 2 to delete")
        return False
    # policy in _RECOMPUTE
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()
    print(f"  [cache-policy=2/recompute] deleted {label}: {p}")
    return True


def guarded_delete_many(paths: Iterable[Union[str, Path]], policy: str,
                        label: str = "cache") -> int:
    """Batch guarded_delete. Returns count of paths actually deleted."""
    n = 0
    for p in paths:
        if guarded_delete(p, policy, label=label):
            n += 1
    return n


def wipe_output_dir(output_dir: Union[str, Path], policy: str,
                    label: str = "output_dir") -> bool:
    """Nuke the entire output_dir when policy=2 ("recompute"). No-op when policy=1.

    iter11 v3 (2026-04-26): semantic upgrade — "recompute" used to mean
    "enforce keep_last_n eviction of stale periodic ckpts during training",
    which still resumed from the latest ckpt. That defeated the user-intent
    of "fresh start". Now `cache-policy=2` deletes the WHOLE output_dir at
    startup so the next load_checkpoint() finds nothing → step 0 fresh run.

    Call this from m09a/m09b/m09c main() right after args.cache_policy is
    resolved and BEFORE any checkpoint discovery. The directory is recreated
    empty so downstream writes (logs, plots, ckpts) succeed.

    Args:
        output_dir: Variant results directory (e.g. outputs/full/explora).
        policy:     '1'/'keep' or '2'/'recompute'.
        label:      Log prefix shown to the user (default 'output_dir').

    Returns True iff the directory was actually wiped.
    """
    _validate(policy)
    p = Path(output_dir)
    if policy in _KEEP:
        if p.exists():
            print(f"  [cache-policy=1/keep] preserved {label}: {p} — "
                  f"rerun with --cache-policy 2 to delete (whole folder wipe)")
        return False
    # policy in _RECOMPUTE
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        return False
    n_files = sum(1 for _ in p.rglob("*") if _.is_file())
    print(f"  [cache-policy=2/recompute] WIPING {label}: {p} ({n_files} file(s))")
    shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    print(f"  [cache-policy=2/recompute] {label} recreated empty: {p}")
    return True


def guarded_glob_delete(parent: Union[str, Path], pattern: str, policy: str,
                        label: str = "cache") -> int:
    """Delete every file in parent matching pattern. Parent must exist."""
    _validate(policy)
    parent = Path(parent)
    if not parent.exists():
        return 0
    matches = list(parent.glob(pattern))
    if not matches:
        return 0
    if policy in _KEEP:
        print(f"  [cache-policy=1/keep] preserved {len(matches)} {label} file(s) "
              f"matching {parent}/{pattern} — rerun with --cache-policy 2 to delete")
        return 0
    n = 0
    for m in matches:
        if m.is_dir():
            shutil.rmtree(m)
        else:
            m.unlink()
        n += 1
    print(f"  [cache-policy=2/recompute] deleted {n} {label} file(s) matching "
          f"{parent}/{pattern}")
    return n


def _validate(policy: str) -> None:
    if policy not in _VALID:
        raise ValueError(
            f"invalid cache-policy={policy!r}; must be one of {_VALID}. "
            f"Orchestrator .sh script should default to '1' when user "
            f"pressed Enter at the read -p prompt."
        )
