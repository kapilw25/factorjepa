"""Canonical split loading + leakage-safe training-pool builder (shared by m09a1/a2/c1/c2). GPU-free.
    python -u src/utils/clip_splits.py --manifest M.json --train-split T.json --val-split V.json --test-split E.json --universe broad_manifest --out train_pool.json
"""
# WHY THIS MODULE EXISTS — single source of truth for how EVERY m09* trainer picks
# its training clips, so the iter15 validity bugs cannot recur:
#   • TEST LEAKAGE     — m09c1/m09c2 trained on (manifest − val) → eval TEST clips
#                        were inside the surgery pool (only val was subtracted).
#   • UNIVERSE ASYMMETRY — pretrain trained on the ~1k labeled train_split while
#                        surgery trained on the ~9k broad manifest (8× data confound).
# Both collapse to one rule, enforced here: train_pool = universe − (val ∪ test),
# with `universe` chosen ONCE from cfg["training_pool"]["universe"] and applied
# identically by all trainers. FAIL LOUD on missing/overlapping splits.
import argparse
import json
import os
import sys
from pathlib import Path


def _read_keys(path: Path) -> set:
    """Load a split JSON → set of clip keys ({"clip_keys"|"saved_keys"|"keys":[...]}, bare
    list, or dict keyed by clip). FAIL LOUD if absent — no silent partial split."""
    if not path.is_file():
        raise FileNotFoundError(
            f"FATAL clip_splits: split file missing at {path}. Generate via "
            f"src/utils/probe_train_subset.py (run_train.sh does this from action_labels.json)."
        )
    d = json.load(open(path))
    if isinstance(d, list):
        return set(d)
    for k in ("clip_keys", "saved_keys", "keys"):
        if k in d and isinstance(d[k], list):
            return set(d[k])
    return set(d.keys())


def load_canonical_splits(train_path, val_path, test_path) -> tuple:
    """Load the ONE pinned (train, val, test) split set shared by every encoder in a run.

    All three paths are caller-supplied (run_train.sh pins them; no in-Python defaults
    per CLAUDE.md). Returns (train_keys, val_keys, test_keys) as sets. FAIL LOUD if any
    file is missing OR the three splits are not mutually disjoint (leakage tripwire).
    """
    train_keys = _read_keys(Path(train_path))
    val_keys = _read_keys(Path(val_path))
    test_keys = _read_keys(Path(test_path))
    tv, tt, vt = train_keys & val_keys, train_keys & test_keys, val_keys & test_keys
    if tv or tt or vt:
        raise ValueError(
            f"FATAL clip_splits: splits NOT disjoint (train∩val={len(tv)}, "
            f"train∩test={len(tt)}, val∩test={len(vt)}) — overlapping splits defeat "
            f"leakage exclusion. Regenerate the split."
        )
    return train_keys, val_keys, test_keys


def build_training_pool(universe_keys, val_keys, test_keys, *,
                        universe_label, log_prefix="") -> list:
    """THE single leakage-safe training-pool builder; every trainer MUST use this.

    train_pool = universe_keys − (val_keys ∪ test_keys) — always excludes BOTH (the
    iter15 bug excluded val only). `universe_label` (required) is provenance for the
    log line; `log_prefix` is an optional cosmetic line prefix. Order preserved.
    """
    val_set, test_set = set(val_keys), set(test_keys)
    holdout = val_set | test_set
    pool = [k for k in universe_keys if k not in holdout]
    n_val_rm = sum(1 for k in universe_keys if k in val_set)
    n_test_rm = sum(1 for k in universe_keys if k in test_set)
    print(f"{log_prefix}[clip_splits] universe={universe_label}: {len(universe_keys)} "
          f"− val({n_val_rm}) − test({n_test_rm}) = {len(pool)} train clips "
          f"(leakage-safe: val∪test excluded)")
    if not pool:
        raise ValueError(
            f"FATAL clip_splits: training pool EMPTY after excluding val∪test from "
            f"universe={universe_label} ({len(universe_keys)} keys). Check universe + splits."
        )
    return pool


def resolve_universe_keys(cfg: dict, train_split_keys, corpus_manifest_path) -> tuple:
    """Resolve the shared training universe from cfg['training_pool']['universe'].

    Returns (universe_keys: list, universe_label: str). Strict cfg lookup + explicit
    manifest path (no defaults/fallbacks). broad_manifest = all corpus clips (SSL needs
    no labels → max data, shared by pretrain AND surgery so the only difference is the
    recipe); labeled_train = the labeled train_split only.
    """
    label = cfg["training_pool"]["universe"]
    if label == "broad_manifest":
        p = Path(corpus_manifest_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"FATAL clip_splits: universe=broad_manifest needs corpus manifest at {p} "
                f"— not found."
            )
        # FIX (iter17 2026-05-26): use _read_keys (extracts manifest["clip_keys"]) — NOT
        # json.load(...).keys() which returned the manifest's TOP-LEVEL metadata keys
        # ("n","seed","source","sampling","clips_per_video","num_videos","clip_keys") → a
        # 7-key garbage pool that matched ZERO corpus clips → m09a producer spun forever
        # (epoch 1000+, GPU 0%). See logs/m09a_pretrain_encoder_sanity.log.
        return sorted(_read_keys(p)), label
    if label == "labeled_train":
        return list(train_split_keys), label
    raise ValueError(
        f"FATAL clip_splits: unknown training_pool.universe={label!r} "
        f"(expected broad_manifest | labeled_train)."
    )


def main() -> None:
    """CLI: build the ONE leakage-safe train pool consumed by every trainer's --subset.

    run_train.sh calls this ONCE per run so all encoders share an identical, test-free
    pool (the iter15 bugs came from each trainer deriving its own pool internally).
    """
    ap = argparse.ArgumentParser(description="Build leakage-safe training pool (val∪test excluded).")
    ap.add_argument("--manifest", required=True, help="Corpus manifest (universe for broad_manifest).")
    ap.add_argument("--train-split", required=True, help="train_split.json (universe for labeled_train).")
    ap.add_argument("--val-split", required=True, help="val_split.json (held out).")
    ap.add_argument("--test-split", required=True, help="test_split.json (held out).")
    ap.add_argument("--universe", required=True, choices=["broad_manifest", "labeled_train"])
    ap.add_argument("--pool-ratio", type=float, required=True,
                    help="Fraction of the universe to keep. The per-mode VALUES live ONLY in "
                         "configs/pipeline.yaml clip_pool_ratio[mode] (single source) — run_train.sh "
                         "yaml_extracts them; never hardcode here. 1.0 = no subsample; <1.0 = "
                         "deterministic seeded subsample so all matrix cells + re-runs share one pool.")
    ap.add_argument("--seed", type=int, required=True,
                    help="Deterministic subsample seed (data.seed). Identical across the 7-cell "
                         "matrix so every trainer draws the same ratio-scaled pool.")
    ap.add_argument("--out", required=True, help="Output train_pool.json ({\"clip_keys\": [...]}).")
    args = ap.parse_args()

    if not (0.0 < args.pool_ratio <= 1.0):
        sys.exit(f"FATAL clip_splits: --pool-ratio must be in (0, 1]; got {args.pool_ratio}.")

    train_keys, val_keys, test_keys = load_canonical_splits(
        args.train_split, args.val_split, args.test_split)
    universe_keys, label = resolve_universe_keys(
        {"training_pool": {"universe": args.universe}}, train_keys, args.manifest)
    # iter17 (2026-05-26): scale the universe by clip_pool_ratio[mode] BEFORE excluding val/test.
    # The per-mode fractions live ONLY in configs/pipeline.yaml clip_pool_ratio (single source);
    # ratio=1.0 → no subsample, ratio<1.0 → deterministic seeded subsample. The fraction scales
    # with corpus size on BOTH eval_10k_local (10k) AND full_local (115k) — no fixed in-trainer cap.
    if args.pool_ratio < 1.0:
        import random
        n_full = len(universe_keys)
        n_target = max(1, round(n_full * args.pool_ratio))
        uk = sorted(universe_keys)
        random.Random(args.seed).shuffle(uk)
        universe_keys = uk[:n_target]
        print(f"  [clip_splits] clip_pool_ratio={args.pool_ratio}: universe subsampled "
              f"{n_full} → {n_target} (seed={args.seed}, deterministic)", flush=True)
    pool = build_training_pool(universe_keys, val_keys, test_keys,
                               universe_label=label, log_prefix="  ")
    # per-PID tmp + atomic os.replace → concurrency-safe: the 4 parallel POC trains each rebuild this shared
    # pool when the pretrain seed is resume-skipped; a raw write_text raced read-vs-write (2026-06-23).
    _out = Path(args.out)
    _tmp = _out.with_suffix(f".tmp{os.getpid()}{_out.suffix}")
    _tmp.write_text(json.dumps({"clip_keys": pool}, indent=2))
    os.replace(_tmp, _out)
    print(f"  [clip_splits] wrote {len(pool)} train-pool keys → {args.out}", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
