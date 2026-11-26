#!/usr/bin/env python3
"""Single accessor for the iter18 arm roster (configs/arm_registry.yaml).

WHY: the arm roster + the arm→encoder→m09-dir mapping used to be re-typed in
scripts/run_train.sh, scripts/run_eval.sh, scripts/iter18_poc_ngpu.py and the plot
scripts. A new arm meant editing 6+ lists; missing one = a FATAL deep in a GPU run
(2026-06-14: six such whack-a-mole aborts in one session). This module is the ONE
place every consumer reads it — python via `import`, bash via the CLI below.

Python:
    from utils.arm_registry import arm2enc, arm2dir, group_for_token
    ARM2ENC = arm2enc()          # scheduler==true arms only (== the old ARM2ENC literal)

Bash (run_train.sh / run_eval.sh):
    python src/utils/arm_registry.py dir-for-token  surgical_3stage_DI_encoder
    python src/utils/arm_registry.py train-name-for-token surgical_3stage_DI_encoder
    python src/utils/arm_registry.py eval-tokens         # space-separated, all arms
    python src/utils/arm_registry.py train-names         # space-separated, kind!=merge
    python src/utils/arm_registry.py kind-for surgery_3stage_DI_replay25_encoder
    python src/utils/arm_registry.py label-for  peft_dora_encoder        # -> peft_dora
    python src/utils/arm_registry.py label-table                         # every arm's plot label
"""
import os
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
# THE canonical registry path (mirrors utils.config's pipeline.yaml constant). An env
# override keeps it testable without a hardcoded data path leaking into a GPU run.
REGISTRY_PATH = Path(os.environ.get("ARM_REGISTRY_PATH", _REPO / "configs" / "arm_registry.yaml"))


def _doc() -> dict:
    if not REGISTRY_PATH.exists():
        sys.exit(f"FATAL: arm registry missing: {REGISTRY_PATH}")
    return yaml.safe_load(REGISTRY_PATH.read_text()) or {}


def _load() -> dict:
    arms = _doc().get("arms")
    if not arms:
        sys.exit(f"FATAL: arm registry {REGISTRY_PATH} has no 'arms:' block")
    return arms


# ── plot labels ──────────────────────────────────────────────────────────────────────────
# THE single label rule for every figure (m13 axes, legends, forest winner column, hero WINNER
# row, scale-replication legend). Full arm name, redundant suffix stripped — NOT an abbreviation.
# Derived, so it cannot drift from the roster and a new arm needs no edit anywhere.
_LABEL_STRIP = ("_encoder",)


def reference_encoders() -> dict:
    """{token: {label, group}} for the non-arm encoders plots must still name (frozen, dinov2)."""
    return _doc().get("reference_encoders") or {}


def label_for(token: str) -> str:
    """Display label for an arm, accepting EITHER namespace: the run_train train-name
    ('surgery_3stage_DI_encoder') or the eval encoder-token ('surgical_3stage_DI_encoder').

    Both resolve through the registry to the SAME encoder-token, so the two namespaces can never
    print two different labels for one arm (they did until 2026-07-22).

    Resolution order:
      1. `reference_encoders:`  — the non-arm encoders (frozen, dinov2)
      2. the arm's optional `label:` field — an EXPLICIT override for arms whose token reads badly
         on a figure (e.g. pretrain_encoder → "VANILLA CONT-SSL", so the plots match the paper's
         prose instead of printing the bare directory token)
      3. derived — the full arm name minus '_encoder'
    Because the override is optional, a new arm still gets a correct label with no edit."""
    arms = _load()
    rec = arms.get(token) or next((r for r in arms.values() if r["encoder"] == token), None)
    enc = rec["encoder"] if rec else token                            # train-name → encoder-token
    ref = reference_encoders()
    if enc in ref:
        return ref[enc]["label"]
    if rec and rec.get("label"):
        return rec["label"]
    for suffix in _LABEL_STRIP:
        if enc.endswith(suffix) and enc != suffix:
            return enc[: -len(suffix)]
    return enc


def label_table() -> list:
    """(encoder_token, label) for ALL arms — eyeball the full label set in one call."""
    return [(r["encoder"], label_for(r["encoder"])) for r in _load().values()]


def paper_scorecard_group(token: str) -> str:
    """Style group for the paper-scorecard panels, for an arm token OR a reference encoder.

    Replaces m13's hand-written _PS_ARMS. FAILS LOUD (KeyError) on a registry group that
    configs/arm_registry.yaml `paper_scorecard_group:` does not list, so a newly-introduced group
    can never be silently rendered as a fine-tuning baseline."""
    ref = reference_encoders()
    if token in ref:
        return ref[token]["group"]
    arms = _load()
    rec = arms.get(token) or next((r for r in arms.values() if r["encoder"] == token), None)
    if rec is None:
        sys.exit(f"paper_scorecard_group: {token!r} is neither an arm nor a reference encoder")
    alias = _doc().get("paper_scorecard_group")
    if not alias:
        sys.exit(f"FATAL: {REGISTRY_PATH} has no 'paper_scorecard_group:' block")
    return alias[rec["group"]]


def arm2enc() -> dict:
    """{train_name: encoder_token} for scheduler arms — byte-equal to the old ARM2ENC literal."""
    return {a: r["encoder"] for a, r in _load().items() if r["scheduler"]}


def arm2dir() -> dict:
    """{train_name: m09_dir} for scheduler arms — byte-equal to the old ARM2DIR literal."""
    return {a: r["dir"] for a, r in _load().items() if r["scheduler"]}


def merge_arms() -> set:
    """train_names of all kind=merge arms — the scheduler builds their post-hoc merge job (not a train)."""
    return {a for a, r in _load().items() if r["kind"] == "merge"}


def merge_recipe(arm: str) -> dict:
    """{base, alpha, predictor} merge recipe for a kind=merge arm — the SINGLE source for the scheduler's
    wiseft_merge command. FAIL LOUD on a non-merge arm or a missing field (KeyError, no .get default)."""
    r = _load()[arm]
    if r["kind"] != "merge":
        sys.exit(f"merge_recipe: {arm!r} is kind={r['kind']!r}, not a merge arm")
    return {"base": r["merge_base"], "alpha": r["merge_alpha"], "predictor": r["merge_predictor"]}


def _by_token() -> dict:
    """{encoder_token: record} across ALL arms (incl. legacy non-scheduler ones)."""
    return {r["encoder"]: {"train": a, **r} for a, r in _load().items()}


def dir_for_token(token: str) -> str:
    """m09 dir for an eval encoder-token (run_eval _arm_dir). '' if unknown (caller decides)."""
    return _by_token().get(token, {}).get("dir", "")


def train_name_for_token(token: str) -> str:
    """run_train SUBCMD for an eval encoder-token (run_eval preflight 'train via' hint). '' if unknown."""
    return _by_token().get(token, {}).get("train", "")


def eval_tokens() -> list:
    """Every eval encoder-token (run_eval _split_enc longest-match list)."""
    return [r["encoder"] for r in _load().values()]


def train_names(exclude_merge: bool = True) -> list:
    """Every run_train SUBCMD (allowlist). Drops kind=merge (e.g. WiSE-FT — scheduler-only)."""
    return [a for a, r in _load().items() if not (exclude_merge and r["kind"] == "merge")]


def kind_for(train_name: str) -> str:
    rec = _load().get(train_name)
    return rec["kind"] if rec else ""


def group_for_token(token: str) -> str:
    return _by_token().get(token, {}).get("group", "")


def token_table() -> list:
    """(encoder_token, dir, train_name) for ALL arms — bash builds its assoc-arrays from this in ONE call."""
    return [(r["encoder"], r["dir"], a) for a, r in _load().items()]


def display_arms(include_merge: bool = True, include_heads: bool = True) -> list:
    """[(train_name, encoder_token, group, kind), ...] for SCHEDULER arms + ALL merge arms, in registry
    order — the plot/eval_metrics scripts build their roster from this (so a new arm auto-appears) and map
    group→color locally. A kind=merge arm is a post-hoc RESULT (built by wiseft_merge.py, not the scheduler),
    so it displays even when scheduler:false — display is decoupled from scheduler orchestration (arm2enc)."""
    out = []
    for a, r in _load().items():
        if not r["scheduler"] and r["kind"] != "merge":   # legacy non-scheduler, non-merge arm → never display
            continue
        if not include_merge and r["kind"] == "merge":
            continue
        if not include_heads and r["kind"] in ("surgery_head", "pretrain_head"):
            continue
        out.append((a, r["encoder"], r["group"], r["kind"]))
    return out


def _cli(argv):
    if not argv:
        sys.exit("usage: arm_registry.py {dir-for-token|train-name-for-token|kind-for} <arg> "
                 "| {eval-tokens|train-names}")
    cmd, *rest = argv
    if cmd == "eval-tokens":
        print(" ".join(eval_tokens()))
    elif cmd == "train-names":
        print(" ".join(train_names()))
    elif cmd == "dir-for-token":
        print(dir_for_token(rest[0]))
    elif cmd == "train-name-for-token":
        print(train_name_for_token(rest[0]))
    elif cmd == "kind-for":
        print(kind_for(rest[0]))
    elif cmd == "token-table":
        for tok, d, n in token_table():
            print(f"{tok}\t{d}\t{n}")
    elif cmd == "label-for":
        print(label_for(rest[0]))
    elif cmd == "label-table":
        for tok, lbl in label_table():
            print(f"{tok}\t{lbl}")
    else:
        sys.exit(f"FATAL: unknown arm_registry command '{cmd}'")


if __name__ == "__main__":
    _cli(sys.argv[1:])
