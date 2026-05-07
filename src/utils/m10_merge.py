"""Merge N worker m10 output dirs back into the canonical m10_sam_segment/ dir.

iter13 v13 FIX-26 (2026-05-07): companion to scripts/run_factor_prep_parallel.sh.

USAGE:
    python -u src/utils/m10_merge.py \
        --canonical-dir data/eval_10k_local/m10_sam_segment \
        --worker-dirs data/eval_10k_local/m10_sam_segment_w0 \
                      data/eval_10k_local/m10_sam_segment_w1 \
                      data/eval_10k_local/m10_sam_segment_w2 \
                      data/eval_10k_local/m10_sam_segment_w3

Steps (idempotent, safe to re-run):
  1. Union segments.json: existing canonical + each worker's segments.json.
  2. Move masks/*.npz from each worker dir into canonical masks/ (skip dups).
  3. Recompute summary.json including quality_aggregate from utils.mask_metrics.
  4. Caller decides whether to delete worker dirs (default kept; pass --clean-up).

After merge, run m11 streaming separately to regenerate factor_manifest +
plots. m10 plots in the canonical dir become stale; user can run
`python src/m10_sam_segment.py --plot ...` to regenerate from merged segments.json.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.mask_metrics import aggregate_percentiles


def _load_json(p: Path, default):
    if not p.exists() or p.stat().st_size == 0:
        return default
    with open(p) as f:
        return json.load(f)


def _atomic_write_json(obj, target: Path):
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--canonical-dir", required=True,
                    help="data/eval_10k_local/m10_sam_segment (final merge target)")
    ap.add_argument("--worker-dirs", required=True, nargs="+",
                    help="data/eval_10k_local/m10_sam_segment_w{i} (one per worker)")
    ap.add_argument("--clean-up", action="store_true",
                    help="Delete worker dirs after successful merge (default OFF — "
                         "user's no-rm rule; verify merge first, then mv to legacy "
                         "or delete manually).")
    args = ap.parse_args()

    canonical = Path(args.canonical_dir)
    canonical_masks = canonical / "masks"
    canonical_masks.mkdir(parents=True, exist_ok=True)
    canonical_segments_path = canonical / "segments.json"

    merged = _load_json(canonical_segments_path, default={})
    print(f"Canonical existing: {len(merged)} clips")

    n_moved = 0
    n_dups = 0
    for wd_str in args.worker_dirs:
        wd = Path(wd_str)
        if not wd.is_dir():
            print(f"  WARN: worker dir missing: {wd}")
            continue

        wseg_path = wd / "segments.json"
        ws = _load_json(wseg_path, default={})
        n_overlap = len(set(ws.keys()) & set(merged.keys()))
        for k, v in ws.items():
            merged[k] = v
        print(f"  {wd.name}: {len(ws):,} clips (overlap with canonical: {n_overlap})")

        wmasks = wd / "masks"
        if wmasks.is_dir():
            for npz in sorted(wmasks.glob("*.npz")):
                tgt = canonical_masks / npz.name
                if tgt.exists():
                    n_dups += 1
                    continue
                shutil.move(str(npz), str(tgt))
                n_moved += 1

    print(f"\nMerged segments.json: {len(merged):,} total clips")
    print(f"Moved masks: {n_moved:,} new + {n_dups:,} dup-skipped")

    _atomic_write_json(merged, canonical_segments_path)
    print(f"Wrote {canonical_segments_path}")

    n_total_agents = sum(s.get("n_agents", 0) for s in merged.values())
    n_total_interactions = sum(s.get("n_interactions", 0) for s in merged.values())
    pixel_ratios = [s["agent_pixel_ratio"] for s in merged.values()
                    if "agent_pixel_ratio" in s]
    mean_pixel_ratio = (sum(pixel_ratios) / len(pixel_ratios)
                        if pixel_ratios else 0.0)
    mask_confs = [s["mean_mask_confidence"] for s in merged.values()
                  if s.get("mean_mask_confidence", 0) > 0]
    mean_mask_confidence = (sum(mask_confs) / len(mask_confs)
                            if mask_confs else 0.0)
    concept_recalls = [s["concept_recall"] for s in merged.values()
                       if "concept_recall" in s]
    mean_concept_recall = (sum(concept_recalls) / len(concept_recalls)
                           if concept_recalls else 0.0)
    clips_with_agents = sum(1 for s in merged.values()
                            if s.get("n_agents", 0) > 0)
    clips_with_agents_pct = clips_with_agents / max(len(merged), 1)

    per_clip_stab = [s["quality"]["stability_score"]["mean"]
                     for s in merged.values()
                     if "quality" in s and "stability_score" in s["quality"]]
    per_clip_obj = [s["quality"]["object_score"]["mean"]
                    for s in merged.values()
                    if "quality" in s and "object_score" in s["quality"]]
    per_clip_compact = [s["quality"]["compactness"]["mean"]
                        for s in merged.values()
                        if "quality" in s and "compactness" in s["quality"]]
    per_clip_tiou = [s["quality"]["temporal_iou_m5"]
                     for s in merged.values()
                     if "quality" in s and "temporal_iou_m5" in s["quality"]]
    n_filtered_total = sum(s["quality"].get("n_filtered_by_stability", 0)
                           for s in merged.values()
                           if "quality" in s)

    quality_aggregate = {
        "stability_score": aggregate_percentiles(per_clip_stab),
        "object_score":    aggregate_percentiles(per_clip_obj),
        "compactness":     aggregate_percentiles(per_clip_compact),
        "temporal_iou_m5": aggregate_percentiles(per_clip_tiou),
        "n_masks_filtered_by_stability": int(n_filtered_total),
    }

    gate_checks = {
        "pixel_ratio_min":   mean_pixel_ratio >= 0.002,
        "pixel_ratio_max":   mean_pixel_ratio <= 0.50,
        "mask_confidence":   mean_mask_confidence >= 0.4,
        "clips_with_agents": clips_with_agents_pct >= 0.5,
    }
    quality_gate = all(gate_checks.values())

    summary = {
        "n_clips": len(merged),
        "n_total_agents": n_total_agents,
        "n_total_interactions": n_total_interactions,
        "mean_agent_pixel_ratio": mean_pixel_ratio,
        "mean_concept_recall": mean_concept_recall,
        "mean_mask_confidence": mean_mask_confidence,
        "clips_with_agents_pct": round(clips_with_agents_pct, 3),
        "quality_gate": "PASS" if quality_gate else "FAIL",
        "quality_gate_checks": {k: "PASS" if v else "FAIL"
                                for k, v in gate_checks.items()},
        "quality_aggregate": quality_aggregate,
        "merged_from_workers": [str(p) for p in args.worker_dirs],
        "merge_method": "union segments + move masks; recomputed via m10_merge.py",
    }
    _atomic_write_json(summary, canonical / "summary.json")
    print(f"Wrote {canonical}/summary.json")
    print(f"  quality_gate: {summary['quality_gate']}")
    for k, v in gate_checks.items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")

    if args.clean_up:
        for wd_str in args.worker_dirs:
            wd = Path(wd_str)
            if wd.is_dir():
                shutil.rmtree(wd)
                print(f"  Cleaned: {wd}")
    else:
        print("\n[no-rm rule] Worker dirs KEPT. After verifying merged outputs:")
        for wd_str in args.worker_dirs:
            print(f"  rm -rf {wd_str}    # or mv to legacy/")

    print("\n✅ merge complete")


if __name__ == "__main__":
    main()
