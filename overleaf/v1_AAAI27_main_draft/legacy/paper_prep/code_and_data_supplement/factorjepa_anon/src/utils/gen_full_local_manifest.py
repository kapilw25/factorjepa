"""Generate the master manifest JSON for a local data dir. CPU-only.

Reads `<local_data_dir>/tags.json` (list of per-clip dicts with `section` /
`video_id` / `source_file`), composes one clip_key per entry as
`{section}/{video_id}/{source_file}`, and writes a flat `{"clip_keys": [...]}`
master manifest under `--output`. Sister artifact of `data/eval_10k_local/
eval_10k.json` for the iter16 FULL corpus.

iter16 M3 (2026-05-21): single-shot utility — runs once per local data dir
when tags.json lands (e.g., after `m00d_download_subset.py --FULL`). Output
filename matches `data.master_manifest_name` in `configs/pipeline.yaml` so
the M9-keyed pipeline picks up the new master without code edits.

USAGE:
    python -u src/utils/gen_full_local_manifest.py \\
        --tags-json data/full_local/tags.json \\
        --output    data/full_local/full_local.json \\
        2>&1 | tee logs/gen_full_local_manifest_$(date +%Y%m%d_%H%M%S).log

Output schema (matches data/eval_10k_local/eval_10k.json):
    {
      "n":                <int>,   # number of clip_keys
      "seed":             99,
      "source":           <str>,   # absolute path to tags.json
      "sampling":         <str>,   # "all clips (full corpus, from <name>)"
      "clips_per_video":  <str>,   # "~N" average
      "num_videos":       <int>,
      "clip_keys":        [...],   # sorted ascending for determinism
    }
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags-json", required=True, type=Path,
                        help="Path to input tags.json (list of per-clip dicts).")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to output master manifest JSON.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed stamp written to manifest (cosmetic; clip_keys are "
                             "sorted deterministically). Default: pipeline.yaml data.manifest_seed.")
    args = parser.parse_args()
    # iter18 H2: None → pipeline.yaml data.manifest_seed (single source).
    if args.seed is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from utils.config import get_pipeline_config
        args.seed = get_pipeline_config()["data"]["manifest_seed"]

    if not args.tags_json.is_file():
        print(f"FATAL: --tags-json not found: {args.tags_json}", file=sys.stderr)
        return 3

    print(f"Reading {args.tags_json} ...")
    tags = json.loads(args.tags_json.read_text())
    if not isinstance(tags, list) or not tags:
        print(f"FATAL: --tags-json must be a non-empty list of dicts; "
              f"got {type(tags).__name__} of len={len(tags) if hasattr(tags, '__len__') else '?'}",
              file=sys.stderr)
        return 4

    # Compose clip_keys. FAIL LOUD on missing fields (no .get() per CLAUDE.md).
    clip_keys = sorted(
        f"{t['section']}/{t['video_id']}/{t['source_file']}" for t in tags
    )
    num_videos = len({t["video_id"] for t in tags})

    out_payload = {
        "n":               len(clip_keys),
        "seed":            args.seed,
        "source":          str(args.tags_json.resolve()),
        "sampling":        f"all clips (full corpus, from {args.tags_json.name})",
        "clips_per_video": f"~{len(clip_keys) // max(num_videos, 1)}",
        "num_videos":      num_videos,
        "clip_keys":       clip_keys,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_payload, indent=2))
    print(f"Wrote {args.output} — n={len(clip_keys):,}, "
          f"num_videos={num_videos:,}, {out_payload['clips_per_video']} clips/video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
