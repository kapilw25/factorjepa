"""CPU-only probe: is a 'walks straight vs turns a corner' label auto-derivable from optical flow?

Motivation (iter20 demo hunt). Our motion labels (magnitude__direction from camera-subtracted
flow) are not eye-verifiable: a viewer cannot see whether a clip is "fast-upward". VM30 in
.claude/memory/visual_mistakes.md established that the NET flow pan is ~0 for typical straight
DenseWorld walking (radial flow), which killed the arrow/direction demos. The hypothesis this
script tests is that a camera TURN produces SUSTAINED horizontal flow that is both automatically
derivable at scale and obviously visible to a human watching the clip.

Method: per-consecutive-frame-pair coarse global translation minimising SAD on downsampled
grayscale -- the SAME estimator as src/m16_retrieval_demo.motion_descriptor, but kept as a
TRAJECTORY (dx over time) instead of mean-pooled, because a turn is defined by sustained
same-sign dx rather than by the clip-mean.

Emits per clip: cumulative dx, largest same-sign dx run, mean motion energy, and the raw dx
trajectory, so a caller can re-threshold without re-decoding.

NO model, NO CUDA -- pure ffmpeg + numpy, safe to run alongside a training job.

Usage
-----
    python -m utils.scratchpad.scan_turns \\
        --tars data/demo_src/data/train-0000{0,30}.tar \\
        --clip-dir <scratch>/clips \\
        --out <scratch>/raw_flow.json \\
        --per-shard 110
"""
import argparse
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.demo_video import decode_frames_select, probe_n_frames


def gray_small(frames_uint8, gw, gh):
    """(T,H,W,3) uint8 -> (T,gh,gw) float32 gray, cheap resize by strided sampling."""
    _T, H, W, _ = frames_uint8.shape
    ys = np.linspace(0, H - 1, gh).astype(int)
    xs = np.linspace(0, W - 1, gw).astype(int)
    return frames_uint8[:, ys][:, :, xs].astype(np.float32).mean(-1)


def dx_trajectory(gray, sx, sy):
    """(T,h,w) -> (dxs, dys, energies), one entry per consecutive pair.

    dx is the horizontal shift (in downsampled px) that best aligns frame t+1 onto frame t, found
    by exhaustive SAD over +/-sx px. dx > 0 means image content moved RIGHT, i.e. the camera
    yawed LEFT. energy is mean |frame diff| (moving vs still).
    """
    T, h, w = gray.shape
    dxs, dys, en = [], [], []
    for t in range(T - 1):
        a = gray[t, sy:h - sy, sx:w - sx]
        b = gray[t + 1]
        best, bdx, bdy = 1e18, 0, 0
        for dy in range(-sy, sy + 1, 2):
            for dx in range(-sx, sx + 1, 1):
                bb = b[sy + dy:h - sy + dy, sx + dx:w - sx + dx]
                sad = float(np.abs(a - bb).mean())
                if sad < best:
                    best, bdx, bdy = sad, dx, dy
        dxs.append(bdx)
        dys.append(bdy)
        en.append(float(np.abs(gray[t] - gray[t + 1]).mean()))
    return np.array(dxs, np.float32), np.array(dys, np.float32), np.array(en, np.float32)


def max_same_sign_run(dxs, dead):
    """Largest |sum| over a maximal contiguous run of same-signed dx.

    Pairs with |dx| < dead are treated as neutral and do NOT break the run -- a real turn often
    has a stalled frame or two mid-corner, and breaking on those would under-report it.
    """
    best, cur, sign = 0.0, 0.0, 0
    for v in dxs:
        s = 0 if abs(v) < dead else (1 if v > 0 else -1)
        if s == 0:
            continue
        if s == sign:
            cur += v
        else:
            sign, cur = s, v
        best = max(best, abs(cur))
    return float(best)


def _scan_one(job):
    cid, mp4, stride, gw, gh, sx, sy, dead, min_pairs = job
    p = Path(mp4)
    try:
        n = probe_n_frames(p)
        idx = list(range(0, n, stride))
        if len(idx) < min_pairs:
            return None
        fr = decode_frames_select(p, idx)
        g = gray_small(fr, gw, gh)
        dxs, dys, en = dx_trajectory(g, sx, sy)
        return dict(
            clip_path=str(p),
            clip_id=cid,
            cum_dx=float(dxs.sum()),
            abs_cum_dx=float(abs(dxs.sum())),
            max_run=max_same_sign_run(dxs, dead),
            energy=float(en.mean()),
            cum_dy=float(dys.sum()),
            n_pairs=int(len(dxs)),
            dxs=[float(v) for v in dxs],
        )
    except Exception as e:  # noqa: BLE001 - one bad clip must not kill a 400-clip scan
        print(f"[skip] {cid}: {type(e).__name__} {e}")
        return None


def extract_clips(tars, clip_dir, per_shard, seed):
    """Pull `per_shard` random .mp4 members out of each webdataset tar into clip_dir."""
    clip_dir = Path(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    jobs = []
    for tp in tars:
        shard = Path(tp).stem.split("-")[-1]
        with tarfile.open(tp) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".mp4")]
            take = rng.choice(len(members), min(per_shard, len(members)), replace=False)
            for mi in sorted(take):
                mem = members[mi]
                cid = f"{shard}_{Path(mem.name).stem}"
                dst = clip_dir / f"{cid}.mp4"
                if not dst.exists():
                    dst.write_bytes(tf.extractfile(mem).read())
                jobs.append((cid, str(dst)))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tars", nargs="+", required=True, help="webdataset shard tars")
    ap.add_argument("--clip-dir", required=True, help="where extracted .mp4s land (scratch)")
    ap.add_argument("--out", required=True, help="output raw_flow.json")
    ap.add_argument("--per-shard", type=int, default=110, help="clips sampled per tar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=6,
                    help="decode every Nth frame; 6 == 5 fps pairs on 30 fps DenseWorld clips")
    ap.add_argument("--gray-w", type=int, default=64, help="downsampled gray width")
    ap.add_argument("--gray-h", type=int, default=36, help="downsampled gray height")
    ap.add_argument("--search-x", type=int, default=10,
                    help="SAD search radius in x (px of gray-w); caps measurable pan per pair")
    ap.add_argument("--search-y", type=int, default=4, help="SAD search radius in y")
    ap.add_argument("--dead", type=float, default=0.5,
                    help="|dx| below this is neutral for the same-sign-run accumulator")
    ap.add_argument("--min-pairs", type=int, default=8, help="skip clips shorter than this")
    ap.add_argument("--workers", type=int, default=0, help="0 == os.cpu_count() capped at 24")
    args = ap.parse_args()

    jobs = extract_clips(args.tars, args.clip_dir, args.per_shard, args.seed)
    jobs = [(cid, mp4, args.stride, args.gray_w, args.gray_h, args.search_x, args.search_y,
             args.dead, args.min_pairs) for cid, mp4 in jobs]
    print(f"{len(jobs)} clips extracted -> {args.clip_dir}")

    workers = args.workers or min(24, os.cpu_count())
    recs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(_scan_one, jobs), total=len(jobs), desc="flow"):
            if r:
                recs.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(recs))
    print(f"OK {len(recs)}/{len(jobs)} records -> {out}")


if __name__ == "__main__":
    main()
