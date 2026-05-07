"""Split m10 remaining clips into N disjoint subset JSONs for parallel workers.

iter13 v13 FIX-26 (2026-05-07): helper for scripts/run_factor_prep_parallel.sh.

USAGE:
    python -u src/utils/m10_split_subset.py \
        --manifest data/eval_10k_local/manifest.json \
        --existing-segments data/eval_10k_local/m10_sam_segment/segments.json \
        --n-workers 4 \
        --out-dir /tmp/m10_parallel_subsets

Reads `manifest["saved_keys"]` (full clip universe), subtracts already-done keys
from existing segments.json, round-robin splits remaining keys into N subset JSONs
(`{"clip_keys": [...]}`) compatible with `utils/config.load_subset()`.

Round-robin (not contiguous) so high-agent-count clips (variance ~10x in SAM3
wall-time) are balanced across workers — contiguous slicing risks the last
worker getting all the heavy clips and finishing 2-3x slower.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help="Path to <local-data>/manifest.json with saved_keys list")
    ap.add_argument("--existing-segments", required=True,
                    help="Path to existing segments.json (already-done keys filtered out). "
                         "Pass /dev/null or non-existent path to skip.")
    ap.add_argument("--n-workers", type=int, required=True,
                    help="Number of disjoint subsets to produce")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write subset_w{i}.json files")
    args = ap.parse_args()

    if args.n_workers < 1:
        print(f"FATAL: --n-workers must be >= 1 (got {args.n_workers})")
        sys.exit(2)

    with open(args.manifest) as f:
        manifest = json.load(f)
    if "saved_keys" not in manifest:
        print(f"FATAL: {args.manifest} has no 'saved_keys' field")
        sys.exit(2)
    all_keys = manifest["saved_keys"]
    if not isinstance(all_keys, list) or not all_keys:
        print("FATAL: manifest['saved_keys'] is empty or wrong type")
        sys.exit(2)

    done_keys = set()
    seg_path = Path(args.existing_segments)
    if seg_path.exists() and seg_path.stat().st_size > 0:
        try:
            with open(seg_path) as f:
                seg = json.load(f)
            if isinstance(seg, dict):
                done_keys = set(seg.keys())
        except Exception as e:
            print(f"  WARN: could not read {seg_path}: {e} (treating as empty)")

    remaining = [k for k in all_keys if k not in done_keys]
    print(f"Total clips:  {len(all_keys):,}")
    print(f"Already done: {len(done_keys):,}")
    print(f"Remaining:    {len(remaining):,}")

    if not remaining:
        print("Nothing remaining — all clips already processed.")
        sys.exit(0)

    chunks = [[] for _ in range(args.n_workers)]
    for i, k in enumerate(remaining):
        chunks[i % args.n_workers].append(k)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        p = out_dir / f"subset_w{i}.json"
        with open(p, "w") as f:
            json.dump({"clip_keys": chunk}, f)
        print(f"  worker {i}: {len(chunk):,} keys → {p}")

    print(f"\n✅ Split complete: {len(remaining):,} clips → {args.n_workers} subsets")


if __name__ == "__main__":
    main()
