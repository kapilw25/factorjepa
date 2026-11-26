"""Runtime CPU-worker auto-tune for streaming dataloaders (iter18, 2026-06-06).

WHY: configs/train/surgery_base.yaml used to FIX factor_streaming.num_workers per mode
(poc: 4). On the 4× box, 3-4 concurrent arms each capped at 4 workers starved their GPUs
(60-140 s/step vs 27 s/step solo; box load 5.7 of 128 cores — the per-arm cap was the
bottleneck, not the machine). A fixed number can never fit 1×/2×/4×/8× boxes at once.

Train yamls now declare `num_workers: auto` (or keep an explicit int as an override).
This resolver turns "auto" into a number from the box's LIVE resources:

    workers = clamp( cores // concurrency − reserve_per_proc, min_workers, max_workers )
    workers = min( workers,  cgroup_ram × ram_headroom_pct / concurrency / ram_per_worker_gb )

· cores        = os.cpu_count() (container-visible CPUs)
· concurrency  = how many sibling training processes share the box:
                 NGPU_CONCURRENCY env (fed by scripts/iter18_poc_ngpu.py per job)
                 → else the box's GPU count (one arm per GPU is this repo's pattern)
                 → else 1 (solo run)
· all tuning constants live in configs/pipeline.yaml streaming.worker_autotune —
  NO numbers in this file (CLAUDE.md no-hardcode).

USAGE (each m09* trainer, right after flattening the mode-gated yaml value):
    from utils.stream_autotune import resolve_stream_workers
    fs_cfg["num_workers"] = resolve_stream_workers(fs_cfg["num_workers"], label="m09c1")
"""
import os
import shutil
import subprocess
from pathlib import Path

from utils.config import get_cgroup_memory_gb, get_pipeline_config
from utils.video_io import _FRAME_CACHE_ENV, _FRAME_CACHE_MIN_FREE_ENV, _FRAME_CACHE_MAX_ENV


def _detect_concurrency() -> int:
    """How many sibling trainers share this box. Scheduler-fed env wins; else one arm
    per GPU (this repo's launch pattern); else solo."""
    env = os.environ.get("NGPU_CONCURRENCY", "").strip()
    if env:
        return max(int(env), 1)
    smi = shutil.which("nvidia-smi")
    if smi:
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=30)
        n = sum(1 for line in out.stdout.splitlines() if line.startswith("GPU "))
        if n:
            return n
    return 1


def enable_train_frame_cache(local_data: str) -> None:
    """iter18 (2026-06-06): memoize the DETERMINISTIC decode (video_io.decode_video_bytes —
    linspace frame sampling + fixed decoder → bit-identical on re-decode) for EVERY trainer
    decode path: factor-stream D_L/D_A/D_I source clips, raw-replay, the m09a producer,
    val-collect, probe decode. All randomness (D_I tube pick, augmentation, replay coin)
    applies AFTER decode on the tensor → training data is byte-identical either way.

    WHY (measured 06-06, 4× box): the streaming loader re-decodes the SAME ~7k-clip pool
    every epoch, every stage, every arm — shallow loader-bound stages ran 38.9 → 189 s/step
    under 4-arm contention. With the cache, epoch ≥2 / stage ≥2 / arms 2..N read ~20 MB .npy
    served from the page cache (POC: ~138 G total, fits 483 G RAM) instead of PyAV-decoding
    (the disk-backed "stochastic caching" pattern: charl-ai.github.io/blog/dataloaders).

    Called from m09_common.merge_m09_common_config — the one seam every m09 trainer passes
    through — so DataLoader fork-workers + producer threads inherit the env. Same dir +
    (clip, nf) keys as probe.eval_frame_cache → shared with probe decode + the eval jobs.
    Knobs: pipeline.yaml streaming.train_frame_cache (enabled, min_free_gb floor — a partial
    cache at FULL scale stays correct: miss → fresh decode)."""
    _pcfg = get_pipeline_config()
    tfc = _pcfg["streaming"]["train_frame_cache"]
    if not tfc["enabled"]:
        print("  [train-frame-cache] disabled (streaming.train_frame_cache.enabled=false)",
              flush=True)
        return
    _max_gb = _pcfg["probe"]["eval_frame_cache"]["max_cache_gb"]   # single source: the SHARED-dir hard cap
    cache_dir = str(Path(local_data) / _pcfg["probe"]["eval_frame_cache"]["subdir"])
    os.environ[_FRAME_CACHE_ENV] = cache_dir
    os.environ[_FRAME_CACHE_MIN_FREE_ENV] = str(tfc["min_free_gb"])
    os.environ[_FRAME_CACHE_MAX_ENV] = str(_max_gb)   # PRIMARY: LRU-evict to stay under -> can never fill disk
    print(f"  [train-frame-cache] ON → {cache_dir} (max_cache={_max_gb}G, LRU-evict) — "
          f"deterministic decode memoized; shared across epochs/stages/arms/evals", flush=True)


def resolve_stream_workers(cfg_value, label: str) -> int:
    """Resolve a yaml num_workers value: int = explicit override (returned as-is),
    "auto" = computed from live cores/concurrency/RAM. FAIL LOUD on anything else."""
    if isinstance(cfg_value, int):
        return cfg_value
    if cfg_value != "auto":
        raise ValueError(
            f"[{label}] factor_streaming.num_workers must be an int or 'auto', "
            f"got {cfg_value!r} — declare it in the train yaml.")

    at = get_pipeline_config()["streaming"]["worker_autotune"]
    # iter18 (2026-06-07) affinity-aware: under the scheduler's taskset cpuset
    # (NGPU_CPUSET set) this arm OWNS its core slice — sched_getaffinity returns it
    # and the CPU term divides by 1, not by sibling count. RAM stays shared → its
    # clamp keeps dividing by concurrency either way.
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:          # non-Linux fallback
        cores = os.cpu_count()
    if not cores:
        raise RuntimeError(f"[{label}] no usable CPU count — cannot auto-tune; "
                           f"set an explicit num_workers int in the train yaml.")
    conc = _detect_concurrency()
    pinned = bool(os.environ.get("NGPU_CPUSET"))
    cpu_div = 1 if pinned else conc

    by_cpu = cores // cpu_div - at["reserve_per_proc"]
    workers = max(min(by_cpu, at["max_workers"]), at["min_workers"])
    ram_gb = get_cgroup_memory_gb()
    if ram_gb != float("inf"):
        by_ram = int(ram_gb * at["ram_headroom_pct"] / conc / at["ram_per_worker_gb"])
        workers = max(min(workers, by_ram), at["min_workers"])
    print(f"  [stream-autotune {label}] cores={cores}{' (pinned cpuset)' if pinned else ''} · "
          f"concurrency={conc} "
          f"(src={'env' if os.environ.get('NGPU_CONCURRENCY') else 'gpu-count'}) · "
          f"cgroup_ram={'inf' if ram_gb == float('inf') else f'{ram_gb:.0f}G'} "
          f"→ num_workers={workers}", flush=True)
    return workers
