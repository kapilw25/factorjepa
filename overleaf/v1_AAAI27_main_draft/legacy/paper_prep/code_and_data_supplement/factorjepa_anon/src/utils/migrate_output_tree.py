"""migrate_output_tree — one-off: move the flat outputs/<mode>/ tree into the iter18 backbone-first tree
   outputs/<mode>/<backbone>_<size>/{train, eval/<corpus>/<metric>/<enc>}.  Spec: plan_output_restructure.md.

DRY-RUN by default (lists every move + checks collisions, moves NOTHING). Pass --execute to perform them
via os.rename (same-filesystem → instant, no data copy). mv-only, never rm; FATAL on any dest collision.

Routing:
  • outputs/<mode>/<backbone>/             (encoder dir)  → <backbone>_<size>/train/            (one move, all arms)
  • outputs/<mode>/<metric>/<enc>/         (per-enc eval) → <bb_of_enc>_<size>/eval/<corpus>/<metric>/<enc>/
  • outputs/<mode>/<metric>/<file>         (agg/labels)   → <primary>_<size>/eval/<corpus>/<metric>/<file>
  • outputs/<mode>/probe_plot/             (plots)        → <primary>_<size>/eval/<corpus>/probe_plot/

The backbone of an eval-encoder leaf: non-champions keep their full name as a prefix (vjepa_2_1_vitg_<arm>);
the champion (vjepa_2_1_vitG) dropped its size tag (enc_prefix) → its leaves are the bare 'vjepa_2_1_<arm>'.

USAGE:
    python -u src/utils/migrate_output_tree.py --mode poc                 # dry-run (review)
    python -u src/utils/migrate_output_tree.py --mode poc --verbose       # dry-run + every move listed
    python -u src/utils/migrate_output_tree.py --mode poc --execute       # perform the moves
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config import get_pipeline_config  # noqa: E402
from utils.output_paths import eval_dir, output_root, train_dir  # noqa: E402


def _backbones():
    return get_pipeline_config()["backbone_size_labels"]   # {raw_name: size_label}


def backbone_from_enc(enc: str, champion: str):
    """Map an eval-encoder leaf to its backbone (None if it doesn't look like an encoder dir)."""
    for bb in _backbones():
        if bb != champion and enc.startswith(bb + "_"):        # non-champion keeps its full name
            return bb
    champ_prefix = champion.rsplit("_", 1)[0] + "_"            # champion's dropped-tag prefix, e.g. 'vjepa_2_1_'
    if enc.startswith(champ_prefix):
        return champion
    return None


def build_plan(mode: str, corpus: str, champion: str):
    root = Path(output_root(mode))
    bbs = _backbones()
    raw_names, new_names = set(bbs), {f"{bb}_{sz}" for bb, sz in bbs.items()}
    moves = []   # (src, dst, kind)
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name in new_names:
            continue                                           # skip files + already-migrated <bb>_<size> dirs
        if d.name in raw_names:                                # encoder dir → train/ (whole dir, all arms)
            moves.append((d, Path(train_dir(mode, d.name)), "encoders"))
        elif d.name == "probe_plot":                           # plots → primary backbone (regenerated per-bb)
            moves.append((d, Path(eval_dir(mode, champion, corpus, "probe_plot")), "plots"))
        else:                                                  # a metric family: split per-enc by backbone
            for item in sorted(d.iterdir()):
                if item.is_dir():
                    bb = backbone_from_enc(item.name, champion)
                    if bb is None:
                        sys.exit(f"FATAL: {item} — cannot map encoder '{item.name}' to a backbone")
                    moves.append((item, Path(eval_dir(mode, bb, corpus, d.name)) / item.name, "eval"))
                else:                                          # aggregate / label file → primary backbone
                    moves.append((item, Path(eval_dir(mode, champion, corpus, d.name)) / item.name, "agg"))
    return moves


def main():
    ap = argparse.ArgumentParser(description="migrate outputs/<mode> → backbone-first tree (dry-run default)")
    ap.add_argument("--mode", required=True, help="poc / sanity / full")
    ap.add_argument("--corpus", default="eval_10k", help="score corpus the existing eval results belong to")
    ap.add_argument("--primary-backbone", default="vjepa_2_1_vitG",
                    help="champion backbone that owns corpus-level aggregates/labels/plots")
    ap.add_argument("--verbose", action="store_true", help="list every move (dry-run prints a summary otherwise)")
    ap.add_argument("--execute", action="store_true", help="perform the moves (default: dry-run)")
    args = ap.parse_args()

    moves = build_plan(args.mode, args.corpus, args.primary_backbone)
    if not moves:
        print("Nothing to migrate (tree already backbone-first).")
        return

    # summary grouped by (destination <bb>_<size>, kind)
    groups = {}
    for _, dst, kind in moves:
        bb_seg = dst.parts[2] if len(dst.parts) > 2 else "?"
        groups.setdefault((bb_seg, kind), 0)
        groups[(bb_seg, kind)] += 1
    print(f"[migrate {args.mode} · corpus={args.corpus}] {len(moves)} moves:")
    for (bb_seg, kind), n in sorted(groups.items()):
        print(f"  {bb_seg:24s} {kind:9s} {n:4d}")
    if args.verbose:
        print("  --- full plan ---")
        for s, dst, kind in moves:
            print(f"  [{kind:8s}] {s}  →  {dst}")

    collisions = [dst for _, dst, _ in moves if dst.exists()]
    if collisions:
        print(f"\n❌ FATAL: {len(collisions)} destination(s) already exist — refusing to overwrite:")
        for dst in collisions[:20]:
            print(f"    {dst}")
        sys.exit(1)

    if not args.execute:
        print(f"\n[dry-run] {len(moves)} moves, 0 collisions. Re-run with --execute to perform them "
              f"(add --verbose to see every move).")
        return

    moved = 0
    for s, dst, _ in moves:
        if not s.exists():                                     # idempotent: a prior partial run already moved it
            print(f"  [skip] {s} (already moved)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(s, dst)
        moved += 1
    print(f"\n✅ migrated {moved}/{len(moves)} items into the backbone-first tree.")


if __name__ == "__main__":
    main()
