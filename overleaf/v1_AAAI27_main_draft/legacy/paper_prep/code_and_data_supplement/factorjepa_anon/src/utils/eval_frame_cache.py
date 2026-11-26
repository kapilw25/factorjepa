#!/usr/bin/env python3
"""Eval frame-cache pre-warm — decode the eval clips ONCE into the shared frame cache so every
per-encoder eval re-run gets cache HITS in decode_video_bytes (CPU-decode → GPU-bound = fast).

DECODE-ONLY: no GPU, no encoder forward. Video decode is a CPU job (PyAV; torchcodec/NVDEC is
disabled in video_io because it segfaults on TAR mp4 bytes) — that is exactly why the GPUs idle
during eval. Run this AFTER interrupting the scheduler so all CPU cores are free. It writes the
exact (clip_key, num_frames) keys decode_video_bytes uses, into cache_dir == EVAL_FRAME_CACHE_DIR.

Importable by any eval orchestrator:
    from utils.eval_frame_cache import prewarm_frame_cache
    prewarm_frame_cache("outputs/poc/probe_action/action_labels.json",
                        "data/eval_10k_local", "outputs/poc/_frame_cache", [16],
                        min_free_gb=25, workers=96)

Or as a CLI (run_eval.sh / runbook), full universe at nf16 (~28 GB, disk-safe):
    PYTHONPATH=src python -u src/utils/eval_frame_cache.py \
        --keys outputs/poc/probe_action/action_labels.json \
        --local-data data/eval_10k_local --cache-dir outputs/poc/_frame_cache \
        --num-frames 16 --min-free-gb 25 --workers 96
"""
import argparse
import fcntl
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

from utils.data_download import iter_clips_parallel
from utils.config import get_pipeline_config
from utils.video_io import (
    _frame_cache_load, _frame_cache_path, _frame_cache_store, decode_video_bytes)


def get_or_build_frames(mp4_bytes, tmp_dir, clip_key, num_frames):
    """The cache-aware decode EVERY eval module should call (via decode_to_tensor). No manual
    pre-build needed: it self-manages the shared frame cache.

        cache EXISTS  → return the cached (T,C,H,W) tensor (no decode).
        cache MISSING → take a per-CLIP lock so the 8 concurrent eval processes never decode the
                        SAME clip twice (no thundering-herd cold-wave), decode ONCE, store, return.
        EVAL_FRAME_CACHE_DIR unset → plain decode, no cache (training / data-prep path).

    The lock is an flock on a per-clip `.lock` file — held only for that one clip's decode and
    auto-released by the kernel if the holder dies (fd closes on exit), so a crashed peer can never
    wedge the cache. After acquiring the lock we RE-CHECK the cache: a peer may have built it while
    we waited, in which case we just read it (double-checked locking)."""
    cache_path = _frame_cache_path(clip_key, num_frames)
    if cache_path is None:                                    # cache disabled → behave like a raw decode
        return decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames)
    cached = _frame_cache_load(cache_path)
    if cached is not None:
        return cached                                         # ── cache EXISTS → use it
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    lock_fd = os.open(cache_path + ".lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)                   # ── only ONE process builds this clip
        cached = _frame_cache_load(cache_path)               # peer may have built it while we waited
        if cached is not None:
            return cached
        frames = decode_video_bytes(mp4_bytes, tmp_dir, clip_key, num_frames)  # ── MISSING → generate
        if frames is not None:
            _frame_cache_store(cache_path, frames)
        return frames
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _load_keys(path):
    """Keys from a .json (dict→its keys, or list) or a .npy array."""
    if path.endswith(".json"):
        d = json.load(open(path))
        return [str(k) for k in (d.keys() if isinstance(d, dict) else d)]
    return [str(k) for k in np.load(path, allow_pickle=True)]


def _prewarm_one(base_tmp, clip_key, mp4, nfs):
    """Decode one clip at each num_frames. The two nfs of the SAME clip run sequentially (shared
    tmp path is safe sequentially); DISTINCT clips run in distinct tasks with their OWN tmp subdir
    → no cross-clip temp-file race under high concurrency."""
    td = tempfile.mkdtemp(dir=base_tmp, prefix="pw_")
    try:
        return sum(get_or_build_frames(mp4, td, clip_key, nf) is not None for nf in nfs)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def prewarm_frame_cache(keys, local_data, cache_dir, num_frames, min_free_gb, workers, limit=0):
    """Decode `keys` clips at each num_frames into cache_dir (the EVAL_FRAME_CACHE_DIR) so a re-run
    of run_eval.sh gets cache hits. `keys` is a path (.json labels dict / .npy keys array) OR an
    iterable of keys; `num_frames` is a comma string ("16" / "16,64") or a list. Returns a summary
    dict. FAIL LOUD if no keys / no clips matched."""
    torch.set_num_threads(1)
    os.environ["EVAL_FRAME_CACHE_DIR"] = str(cache_dir)
    os.environ["EVAL_FRAME_CACHE_MIN_FREE_GB"] = str(min_free_gb)
    os.environ["EVAL_FRAME_CACHE_MAX_GB"] = str(  # PRIMARY hard cap (single source) -> LRU-evict, never fills disk
        get_pipeline_config()["probe"]["eval_frame_cache"]["max_cache_gb"])
    os.makedirs(cache_dir, exist_ok=True)
    nfs = [int(x) for x in (num_frames.split(",") if isinstance(num_frames, str) else num_frames)]
    key_list = _load_keys(keys) if isinstance(keys, str) else [str(k) for k in keys]
    if not key_list:
        raise SystemExit(f"FATAL: 0 keys from {keys}")
    if limit:
        key_list = key_list[:limit]
    print(f"pre-warm: {len(key_list)} clips × num_frames={nfs} → {cache_dir} "
          f"(workers={workers}, CPU decode, no GPU)", flush=True)

    clip_q, tar_stop, _r = iter_clips_parallel(
        local_data=local_data, subset_keys=set(key_list), processed_keys=set())
    items = []
    while True:
        it = clip_q.get(timeout=300)
        if it is None:
            break
        items.append(it)
    tar_stop.set()
    if not items:
        raise SystemExit(f"FATAL: iter_clips_parallel yielded 0 of {len(key_list)} keys from {local_data}")
    print(f"  read {len(items)}/{len(key_list)} clips from TARs; decoding on {workers} workers…", flush=True)

    t0, done, ok, n = time.time(), 0, 0, len(items)
    base_tmp = tempfile.mkdtemp(prefix="prewarm_")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_prewarm_one, base_tmp, k, mp4, nfs) for (k, mp4) in items]
            for f in as_completed(futs):
                ok += f.result()
                done += 1
                if done % 250 == 0 or done == n:
                    rate = done / max(time.time() - t0, 1e-9)
                    print(f"  {done}/{n} clips · {rate:.0f}/s · ETA {(n - done) / max(rate, 1e-9) / 60:.1f} min",
                          flush=True)
    finally:
        shutil.rmtree(base_tmp, ignore_errors=True)
    mins = (time.time() - t0) / 60
    print(f"DONE: {done}/{n} clips, {ok} (clip×nf) cached, {n * len(nfs) - ok} decode-fails, "
          f"{mins:.1f} min → {cache_dir}", flush=True)
    return {"clips": done, "cached": ok, "minutes": round(mins, 1), "cache_dir": str(cache_dir)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True, help="action_labels.json (universe) or a *_clip_keys.npy")
    ap.add_argument("--local-data", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--num-frames", required=True, help="comma list, e.g. 16 or 16,64")
    ap.add_argument("--min-free-gb", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only this many clips (0 = all)")
    args = ap.parse_args()
    prewarm_frame_cache(args.keys, args.local_data, args.cache_dir, args.num_frames,
                        args.min_free_gb, args.workers, args.limit)


if __name__ == "__main__":
    main()
