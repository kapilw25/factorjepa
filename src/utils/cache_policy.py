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

import shutil
from pathlib import Path
from typing import Iterable, Union

# Accept both digit (user-facing) and word (back-compat / self-documenting code).
_KEEP = {"1", "keep"}
_RECOMPUTE = {"2", "recompute"}
_VALID = sorted(_KEEP | _RECOMPUTE)


def add_cache_policy_arg(parser) -> None:
    """Register --cache-policy on an argparse.ArgumentParser.

    Non-interactive. Accepts 1 (keep, default) or 2 (recompute). The .sh
    orchestrator is the only place that prompts the user; .py callees never
    block on input so overnight/tmux runs proceed unattended.
    """
    parser.add_argument(
        "--cache-policy",
        type=str,
        choices=_VALID,
        default="1",
        help=(
            "1 = keep (default, preserve all cached checkpoints/intermediates; "
            "deletion call-sites become no-ops with a log line). "
            "2 = recompute (authorize destructive deletes — user must type "
            "`2` at the .sh read -p prompt before a run can wipe cache). "
            "Word aliases `keep`/`recompute` also accepted."
        ),
    )


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
