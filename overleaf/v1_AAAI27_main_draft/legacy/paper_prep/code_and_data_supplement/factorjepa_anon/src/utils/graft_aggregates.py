#!/usr/bin/env python3
"""Graft a REFERENCE run's per-backbone AGGREGATED metrics into the CURRENT run's eval JSONs, so a
backbone evaluated in a prior iter (whose raw per-clip predictions were cleaned up) appears as a
NATIVE backbone in m13's plots — rendered with the CURRENT code (same methodology + formatting), not
pasted as a stale image.

WHY aggregates, not per-encoder dirs: the reference run kept only the aggregated test_metrics.json /
*_per_variant.json (the per-clip *.npy were deleted to save disk). m13 reads the aggregates, so we
merge those directly. NOTE: m12*'s paired_delta CANNOT reproduce the backbone (no per-clip arrays),
so if a paired_delta stage re-runs it will DROP the grafted backbone — re-run this graft after it.

Grafts encoders present in --source but ABSENT in --target (e.g. the champion arms), plus the
within-grafted pairwise_deltas (champion-arm vs champion-frozen). External baselines shared by both
runs are left untouched. Idempotent: re-running is a no-op once grafted.

USAGE:
  python -u src/utils/graft_aggregates.py \\
    --target outputs/poc \\
    --source iter/iter16_metrics_temporal/result_outputs/poc
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)

# (relative path, structure kind, per-encoder container key). 'flat' = top-level container; 'dims' /
# 'metrics' = one container per taxonomy-dim / predictor-metric.
# iter18 H1: dir + file names composed from pipeline.yaml artifacts.* — no literals.
_FILES = [
    (f"{artifact('probe_action_dir')}/{artifact('probe_paired_delta')}", "flat", "by_encoder"),
    (f"{artifact('probe_motion_cos_dir')}/{artifact('probe_motion_cos_paired')}", "flat", "by_encoder"),
    (f"{artifact('probe_future_mse_dir')}/{artifact('probe_future_mse_per_variant')}", "flat", "by_variant"),
    (f"{artifact('probe_taxonomy_dir')}/{artifact('per_dim_acc')}", "dims", "by_encoder"),
    (f"{artifact('predictor_temporal_dir')}/{artifact('predictor_temporal_per_variant')}", "metrics", "by_variant"),
]


def _graft_block(tcont, scont, tpair, spair, gset):
    """Copy the graft-set encoders' entries from scont into tcont (OVERWRITE so the champion matches
    the source exactly), and copy every within-graft-set pairwise_delta into tpair. The graft-set is
    computed ONCE from the action file (where the champion is cleanly source-only) and applied to
    EVERY file — so a metric whose champion entry already existed in the target still gets its
    paired-deltas grafted (without which _family_verdict drops that metric). Returns #entries copied."""
    n = 0
    for k in gset:
        if k in scont and isinstance(scont[k], dict):
            tcont[k] = scont[k]
            n += 1
    for pk, pv in (spair or {}).items():
        if "_minus_" in pk:
            a, b = pk.split("_minus_", 1)
            if a in gset and b in gset:
                tpair[pk] = pv
    return n


def _atomic_write(path, obj):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, str(path))


_ARM_SUFFIXES = ("_frozen", "_pretrain_2X_encoder", "_pretrain_encoder", "_pretrain_head",
                 "_surgical_3stage_DI_encoder", "_surgical_noDI_encoder",
                 "_surgical_3stage_DI_head", "_surgical_noDI_head")


def _stem(enc):
    """Backbone stem of a trained-arm encoder (strip the arm suffix), else None (not a trained arm)."""
    for suf in _ARM_SUFFIXES:
        if enc.endswith(suf):
            return enc[:-len(suf)]
    return None


def _graft_set(source_root):
    """Encoders to graft from the source = every arm of a FULLY-TRAINED backbone (a stem that has at
    least one pretrain/surgical arm — e.g. the champion), NOT standalone frozen baselines. Derived from
    the SOURCE alone → idempotent: re-running after a graft still picks the same champion set."""
    sa = json.loads((Path(source_root) / _FILES[0][0]).read_text()).get("by_encoder", {})
    stems = {_stem(k) for k in sa if ("surgical" in k or "pretrain" in k) and _stem(k)}
    return {k for k in sa if _stem(k) in stems and isinstance(sa[k], dict)}


def graft_run(target_root, source_root):
    tr, sr = Path(target_root), Path(source_root)
    gset = _graft_set(source_root)
    print(f"  graft-set ({len(gset)} encoders): {sorted(gset)}")
    total = 0
    for rel, kind, cont in _FILES:
        tp, sp = tr / rel, sr / rel
        if not (tp.exists() and sp.exists()):
            print(f"  [skip] {rel}: missing target or source")
            continue
        T, S = json.loads(tp.read_text()), json.loads(sp.read_text())
        added = 0
        if kind == "flat":
            T.setdefault("pairwise_deltas", {})
            added += _graft_block(T.setdefault(cont, {}), S.get(cont, {}),
                                  T["pairwise_deltas"], S.get("pairwise_deltas", {}), gset)
        else:  # 'dims' or 'metrics' — one container per sub-key
            top = "dims" if kind == "dims" else "metrics"
            for sub, sblk in S.get(top, {}).items():
                tblk = T.setdefault(top, {}).setdefault(sub, {cont: {}, "pairwise_deltas": {}})
                tblk.setdefault("pairwise_deltas", {})
                added += _graft_block(tblk.setdefault(cont, {}), sblk.get(cont, {}),
                                      tblk["pairwise_deltas"], sblk.get("pairwise_deltas", {}), gset)
        _atomic_write(tp, T)
        total += added
        print(f"  [graft] {rel}: {added} entries copied (incl. paired-deltas)")
    print(f"grafted {total} entries total")
    return total


def main():
    p = argparse.ArgumentParser(
        description="graft a reference run's source-only backbone aggregates into the current run's JSONs")
    p.add_argument("--target", required=True, help="current run output root (e.g. outputs/poc)")
    p.add_argument("--source", required=True, help="reference run output root (e.g. iter16 .../poc)")
    a = p.parse_args()
    print(f"grafting source-only encoders: {a.source} -> {a.target}")
    graft_run(a.target, a.source)


if __name__ == "__main__":
    main()
