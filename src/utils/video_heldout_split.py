"""Video-disjoint held-out test set for models trained on an eval-subset (e.g. eval_10k), from the full corpus. CPU-only.

A source video is HELD OUT when none of its clips is in the training pool or the validation split of the trained
models. Training pool = eval-subset clips minus the labelled val and test clips (same rule as
utils/clip_splits.py: pool = universe - (val | test), unlabelled clips included). Source videos are identified by
clips.csv `base_video_id`, the true YouTube id (the `video_id` part of a clip key is a chunk id for long videos).
From held-out videos the test set takes NEW clips (not in the eval-subset), at most --per-video per video, sampled
with --seed. Output: {"clip_keys": [...], ...} (the eval-subset schema m04e reads), plus per-step counts.

Owner TODO 'Video held-out test, saved 10k models' (FactorJEPA workshop drafts, Fri Nov 27 2026).
USAGE:
    python -u src/utils/video_heldout_split.py \\
        --eval-subset data/eval_10k_local/eval_10k.json \\
        --action-labels outputs/poc/vjepa_2_1_vitg_1B/eval/eval_10k/probe_action/action_labels.json \\
        --corpus data/full_local/full_local.json \\
        --clips-csv src/hf_denseworld_115k/clips.csv \\
        --per-video 15 --seed 42 \\
        --output data/heldout_video/heldout_video.json 2>&1 | tee logs/video_heldout_split_$(date +%Y%m%d_%H%M%S).log
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def build(eval_subset: Path, action_labels: Path, corpus: Path, clips_csv: Path, per_video: int, seed: int) -> dict:
    sub = set(json.loads(eval_subset.read_text())["clip_keys"])
    labels = json.loads(action_labels.read_text())
    val = {k for k, v in labels.items() if v["split"] == "val"}
    test = {k for k, v in labels.items() if v["split"] == "test"}
    missing = (val | test) - sub
    if missing:
        raise RuntimeError(f"{len(missing)} labelled val/test clips are not in {eval_subset}")
    pool = sub - val - test
    base = {}
    with open(clips_csv) as f:
        for r in csv.DictReader(f):
            base[r["clip_key"]] = r["base_video_id"]
    corpus_keys = json.loads(corpus.read_text())["clip_keys"]
    unknown = [k for k in list(sub) + corpus_keys if k not in base]
    if unknown:
        raise RuntimeError(f"{len(unknown)} clip keys have no base_video_id in {clips_csv}, e.g. {unknown[:2]}")
    seen = {base[k] for k in pool | val}
    by_video = defaultdict(list)
    for k in corpus_keys:
        if base[k] not in seen and k not in sub:
            by_video[base[k]].append(k)
    rng = random.Random(seed)
    picked = []
    for vid in sorted(by_video):
        ks = sorted(by_video[vid])
        picked += sorted(rng.sample(ks, min(per_video, len(ks))))
    test_videos = {base[k] for k in test}
    return {
        "clip_keys": picked,
        "n": len(picked),
        "seed": seed,
        "per_video": per_video,
        "sampling": "video-disjoint: new corpus clips from source videos with no clip in the training pool or val",
        "counts": {
            "eval_subset_clips": len(sub), "train_pool_clips": len(pool), "val_clips": len(val), "test_clips": len(test),
            "videos_seen_in_training_or_val": len(seen), "heldout_videos": len(by_video),
            "heldout_videos_that_were_test_videos": len(set(by_video) & test_videos),
            "heldout_candidate_clips": sum(len(v) for v in by_video.values()),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval-subset", type=Path, required=True)
    p.add_argument("--action-labels", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--clips-csv", type=Path, required=True)
    p.add_argument("--per-video", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = build(a.eval_subset, a.action_labels, a.corpus, a.clips_csv, a.per_video, a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["counts"], indent=1))
    print(f"wrote {a.output}: {out['n']} clips from {out['counts']['heldout_videos']} held-out videos")


if __name__ == "__main__":
    main()
