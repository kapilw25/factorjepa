"""Paired comparisons between saved per-clip eval arrays, clip-level BCa and video-clustered.

Built 2026-11-27 for the AAAI-27 workshop papers (judge round 1 on Cog2Soc):
  * attribution ladder at 10k: plain JEPA retrain -> FactorJEPA-RAW (factorized, no factor
    labels) -> FactorJEPA (factor labels), per diagnostic, both scales;
  * video-clustered intervals (C4): the split is clip-level, so clips of one source video can
    sit in train and test; resampling source videos instead of clips gives the honest interval.

Inputs are the arrays the eval modules already save (per_clip_*.npy plus the matching clip-key
array). Nothing is recomputed on a GPU. Metric specs come from the CLI, so this file holds no
path literals; n_resamples / ci_level / seed come from configs/pipeline.yaml via utils.bootstrap.

Usage (one metric spec per --metric, NAME=PERCLIP:KEYS:DIRECTION, {arm} is substituted):
  python -u src/utils/paired_ladder.py --root <eval_dir> \
      --arm-prefix vjepa_2_1_vitg_ --arms pretrain_encoder surgery_raw_encoder ... \
      --pairs pretrain_encoder:surgery_raw_encoder surgery_raw_encoder:surgical_3stage_DI_diheavy_encoder \
      --metric future=probe_future_mse/{arm}/per_clip_mse.npy:probe_future_mse/{arm}/clip_keys.npy:lower \
      --out <json>
Delta is always B minus A for a pair A:B; "B better" means lower for "lower" metrics.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.bootstrap import paired_bca, N_BOOTSTRAP, CI_LEVEL, _DEFAULT_SEED  # noqa: E402
from utils.action_labels import _extract_video_id  # noqa: E402


def load_metric(root: Path, spec: str, arm: str) -> dict:
    """Return {clip_key: value} for one arm, failing loud on a length mismatch."""
    per_clip_t, keys_t, _ = spec.split(":")
    vals = np.load(root / per_clip_t.format(arm=arm), allow_pickle=False)
    # clip-key arrays are object (str) arrays written by our own eval modules -> pickle needed
    keys = np.load(root / keys_t.format(arm=arm), allow_pickle=True)
    if len(vals) != len(keys):
        raise RuntimeError(f"{arm}: {len(vals)} values vs {len(keys)} keys for {per_clip_t}")
    return {str(k): float(v) for k, v in zip(keys, vals)}


def aligned_delta(a: dict, b: dict) -> tuple:
    """Per-clip B minus A on the shared clip keys, plus those keys (sorted, so runs repeat)."""
    shared = sorted(set(a) & set(b))
    if not shared:
        raise RuntimeError("no shared clip keys between the two arms")
    d = np.array([b[k] - a[k] for k in shared], dtype=np.float64)
    return d, shared


def video_clustered_ci(deltas: np.ndarray, keys: list, n_boot: int, ci: float, seed: int) -> dict:
    """Percentile interval of the mean delta when whole source videos are resampled."""
    vids = np.array([_extract_video_id(k) for k in keys])
    uniq, inv = np.unique(vids, return_inverse=True)
    sums = np.bincount(inv, weights=deltas, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.quantile(boot, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
            "ci_half": round(float(hi - lo) / 2, 6), "n_videos": int(len(uniq))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--arm-prefix", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="A:B, delta = B - A")
    ap.add_argument("--metric", action="append", required=True, help="NAME=PERCLIP:KEYS:lower|higher")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    out = {"root": str(args.root), "n_resamples": N_BOOTSTRAP, "ci_level": CI_LEVEL,
           "seed": _DEFAULT_SEED, "means": {}, "pairs": {}}
    for m in args.metric:
        name, spec = m.split("=", 1)
        direction = spec.split(":")[2]
        if direction not in ("lower", "higher"):
            raise ValueError(f"{name}: direction must be lower or higher, got {direction}")
        data = {a: load_metric(args.root, spec, args.arm_prefix + a) for a in args.arms}
        out["means"][name] = {a: round(float(np.mean(list(v.values()))), 6) for a, v in data.items()}
        for p in args.pairs:
            a, b = p.split(":")
            d, keys = aligned_delta(data[a], data[b])
            clip = paired_bca(d)
            vid = video_clustered_ci(d, keys, N_BOOTSTRAP, CI_LEVEL, _DEFAULT_SEED)
            better = (clip["ci_hi"] < 0) if direction == "lower" else (clip["ci_lo"] > 0)
            worse = (clip["ci_lo"] > 0) if direction == "lower" else (clip["ci_hi"] < 0)
            out["pairs"].setdefault(name, {})[p] = {
                "delta_mean": clip["mean"], "clip_bca": clip, "video_clustered": vid,
                "relative_pct": round(100 * clip["mean"] / float(np.mean([data[a][k] for k in keys])), 3),
                "verdict_clip": "B better" if better else ("B worse" if worse else "not separable"),
            }
            print(f"{name:10s} {p:60s} delta={clip['mean']:+.5f} clip[{clip['ci_lo']:+.5f},{clip['ci_hi']:+.5f}] "
                  f"video[{vid['ci_lo']:+.5f},{vid['ci_hi']:+.5f}] n={clip['n']} vids={vid['n_videos']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
