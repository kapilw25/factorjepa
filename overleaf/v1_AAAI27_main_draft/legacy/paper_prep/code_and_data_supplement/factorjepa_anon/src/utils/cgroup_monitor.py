"""cgroup memory + PID monitor — forensic trail for silent SIGKILLs.

Container hosts cap RAM via cgroup memory.limit_in_bytes (NOT `free -h` —
ulimit -u and /proc/meminfo lie about the actual limit visible to the
process). When usage hits the limit, the kernel SIGKILLs the process with
no Python traceback. The classic symptom is a log that ends mid-tqdm
update with no error message and an empty shell prompt returning instantly.

This module gives the operator a forensic breadcrumb trail so the LAST
log line before SIGKILL shows the run-up to OOMKill:

    [m04d] cgroup envelope:
    [m04d]   memory: 2.2 GB / 36.0 GB (6%)
    [m04d]   pids:   405 / 1024 (40%)
    [m04d-oom-watchdog] started (warn=80%, crit=90%, imminent=97%, interval=10s)
    ... (training proceeds) ...
    [m04d-oom-watchdog] ⚠️  warn: memory 28.8 GB / 36.0 GB (80.0%)
    [m04d-oom-watchdog] 🚨 CRITICAL: memory 32.5 GB / 36.0 GB (90.3%) — reduce workers/queue NOW
    [m04d-oom-watchdog] 🔥 IMMINENT SIGKILL: memory 35.0 GB / 36.0 GB (97.2%)
    <process dies — no Python traceback, kernel sent SIGKILL>

Even though SIGKILL bypasses Python, the watchdog's last line is on disk
already (flush=True), so the operator can see WHY the run died.

USAGE (called from any producer-consumer GPU script — m04d, m05, m09):
    from utils.cgroup_monitor import print_cgroup_header, start_oom_watchdog
    print_cgroup_header(prefix="[m04d]")
    start_oom_watchdog(prefix="[m04d-oom-watchdog]")

Self-test (CPU sanity — prints envelope + 30s of watchdog ticks):
    python -u src/utils/cgroup_monitor.py
"""
import sys
import threading
import time
from pathlib import Path

_LOW_PID_LIMIT = 2048   # iter18 W7: below this, decode-worker thread pools risk EAGAIN


# cgroup v1 paths (Docker, k8s default)
_V1_MEM_LIMIT  = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_V1_MEM_USAGE  = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_V1_PIDS_MAX   = "/sys/fs/cgroup/pids/pids.max"
_V1_PIDS_CUR   = "/sys/fs/cgroup/pids/pids.current"

# cgroup v2 paths (unified hierarchy — newer systemd / Fedora / k8s 1.25+)
_V2_MEM_MAX    = "/sys/fs/cgroup/memory.max"
_V2_MEM_CUR    = "/sys/fs/cgroup/memory.current"
_V2_PIDS_MAX   = "/sys/fs/cgroup/pids.max"
_V2_PIDS_CUR   = "/sys/fs/cgroup/pids.current"

# memory.stat exposes the reclaimable/unreclaimable split. 'anon' (v2) / 'total_rss' (v1) is the
# UNRECLAIMABLE portion the kernel OOM-killer enforces against; 'file' is reclaimable page cache
# (evicted before any SIGKILL). memory.current/usage counts BOTH → useless as a proximity-to-kill
# signal on cache-heavy streaming runs (iter19 2026-07-04: 99% current = 91 G cache + 35 G anon).
_V1_MEM_STAT   = "/sys/fs/cgroup/memory/memory.stat"
_V2_MEM_STAT   = "/sys/fs/cgroup/memory.stat"

# A v1 cap is "effectively unlimited" when set to 2^63-1-ish (the kernel
# sentinel). Treat anything ≥ 1 EiB as unlimited.
_V1_UNLIMITED_SENTINEL = 1 << 60


def _read_int(path: str):
    """Return integer contents of `path`, or None if missing/unreadable/'max'."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        val = p.read_text().strip()
    except OSError:
        return None
    if val == "max":
        return None  # cgroup v2 sentinel for unlimited
    try:
        return int(val)
    except ValueError:
        return None


def read_cgroup_memory_limit():
    """Return cgroup memory cap in bytes, or None if unlimited / no cgroup."""
    v1 = _read_int(_V1_MEM_LIMIT)
    if v1 is not None and v1 < _V1_UNLIMITED_SENTINEL:
        return v1
    return _read_int(_V2_MEM_MAX)


def read_cgroup_memory_usage():
    """Return cgroup memory usage in bytes, or None if no cgroup.

    NOTE: this is the TOTAL (anon + reclaimable page cache). For proximity-to-SIGKILL use
    read_cgroup_memory_anon() instead — see its docstring."""
    v1 = _read_int(_V1_MEM_USAGE)
    if v1 is not None:
        return v1
    return _read_int(_V2_MEM_CUR)


def _read_stat_field(path: str, field: str):
    """Return the int value of `field` in a cgroup memory.stat file, or None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        for line in p.read_text().splitlines():
            k, _, v = line.partition(" ")
            if k == field:
                return int(v)
    except (OSError, ValueError):
        return None
    return None


def read_cgroup_memory_anon():
    """Return cgroup ANONYMOUS memory in bytes (RSS-like, UNRECLAIMABLE) — the metric the kernel
    OOM-killer actually enforces against. v2 'anon' / v1 'total_rss'. None if no cgroup.

    Use this, NOT read_cgroup_memory_usage(): memory.current/usage_in_bytes ALSO count reclaimable
    file-backed page cache, which the kernel evicts long before any SIGKILL. Thresholding the total
    false-alarms on cache-heavy (data-streaming) runs — iter19 2026-07-04: a HEALTHY seed sat at 99%
    memory.current (91 G of reclaimable tar-read cache) with only 35 G real anon and ~90 G of true
    headroom, yet the watchdog screamed 'IMMINENT SIGKILL' and nearly triggered a panic-kill."""
    v1 = _read_stat_field(_V1_MEM_STAT, "total_rss")
    if v1 is not None:
        return v1
    return _read_stat_field(_V2_MEM_STAT, "anon")


def read_cgroup_pids_limit():
    """Return cgroup PID cap, or None if unlimited / no cgroup."""
    v1 = _read_int(_V1_PIDS_MAX)
    if v1 is not None:
        return v1
    return _read_int(_V2_PIDS_MAX)


def read_cgroup_pids_current():
    """Return current PID count in this cgroup, or None if no cgroup."""
    v1 = _read_int(_V1_PIDS_CUR)
    if v1 is not None:
        return v1
    return _read_int(_V2_PIDS_CUR)


def _fmt_gb(nbytes):
    if nbytes is None:
        return "unlimited"
    return f"{nbytes / 1024 / 1024 / 1024:.1f} GB"


def print_cgroup_header(prefix: str = "[cgroup]") -> dict:
    """Print cgroup limits + current usage at script startup.

    Returns a dict of the measured values (for callers that want to make
    runtime decisions, e.g., auto-tune decode_workers from the cap).

    Output format:
        [m04d] cgroup envelope:
        [m04d]   memory: 2.2 GB / 36.0 GB (6%)
        [m04d]   pids:   405 / 1024 (40%)
        [m04d]   ⚠️  cgroup memory < 48 GB — tune pipeline.yaml decode/queue knobs
    """
    mem_limit = read_cgroup_memory_limit()
    mem_usage = read_cgroup_memory_usage()
    pid_limit = read_cgroup_pids_limit()
    pid_cur   = read_cgroup_pids_current()

    mem_pct = (100.0 * mem_usage / mem_limit) if (mem_limit and mem_usage) else 0.0
    pid_pct = (100.0 * pid_cur   / pid_limit) if (pid_limit and pid_cur)   else 0.0

    print(f"{prefix} cgroup envelope:")
    print(f"{prefix}   memory: {_fmt_gb(mem_usage)} / {_fmt_gb(mem_limit)} "
          f"({mem_pct:.0f}%)")
    print(f"{prefix}   pids:   {pid_cur if pid_cur is not None else '?'} / "
          f"{pid_limit if pid_limit is not None else 'unlimited'} "
          f"({pid_pct:.0f}%)")

    if mem_limit is not None and mem_limit < 48 * (1024 ** 3):
        print(f"{prefix}   ⚠️  cgroup memory < 48 GB — tune pipeline.yaml "
              f"decode/queue knobs (see scaling table at "
              f"configs/pipeline.yaml:streaming:decode_workers_motion)")
    if pid_limit is not None and pid_limit < _LOW_PID_LIMIT:
        print(f"{prefix}   ⚠️  cgroup pids < 2048 — keep OMP_NUM_THREADS=1 "
              f"caps in m04d preamble (libgomp Thread creation failure risk)")

    return {
        "memory_limit_bytes": mem_limit,
        "memory_usage_bytes": mem_usage,
        "memory_pct": mem_pct,
        "pids_limit": pid_limit,
        "pids_current": pid_cur,
        "pids_pct": pid_pct,
    }


def start_oom_watchdog(threshold_warn: float = None,
                       threshold_crit: float = None,
                       threshold_imminent: float = None,
                       interval_sec: int = None,
                       prefix: str = "[cgroup-oom-watchdog]") -> threading.Thread:
    """Start a daemon thread that prints LOUD warnings as memory approaches the cap.

    iter18 H6: None → pipeline.yaml train_guards.oom_watchdog.* (single source).

    Forensic trail: when the kernel SIGKILLs the process at the cap, Python's
    traceback handler doesn't run — but each print here has flush=True, so
    the LAST log line on disk shows how close to the cap the process was at
    the moment of death (e.g., "memory at 97.2% → IMMINENT SIGKILL").

    Operators reading the log can then diagnose the silent kill instantly
    without dmesg / cgroup inspection.

    Returns the watchdog Thread (already started). Caller does not need to
    join — it's a daemon and dies with the process. Returns None if there's
    no cgroup memory limit (host runs unlimited).
    """
    from utils.config import get_pipeline_config
    _wd = get_pipeline_config()["train_guards"]["oom_watchdog"]
    threshold_warn = _wd["threshold_warn"] if threshold_warn is None else threshold_warn
    threshold_crit = _wd["threshold_crit"] if threshold_crit is None else threshold_crit
    threshold_imminent = _wd["threshold_imminent"] if threshold_imminent is None else threshold_imminent
    interval_sec = _wd["interval_sec"] if interval_sec is None else interval_sec
    mem_limit = read_cgroup_memory_limit()
    if mem_limit is None:
        print(f"{prefix} no cgroup memory limit detected — watchdog disabled")
        return None

    # Idempotent: if a watchdog thread is already running (e.g., a multi-stage
    # script like probe_action.py calls check_gpu() + start_oom_watchdog at
    # both --stage train and --stage eval entry points), skip re-spawning.
    for t in threading.enumerate():
        if t.name == "cgroup-oom-watchdog" and t.is_alive():
            print(f"{prefix} already running (PID {t.ident}) — skipping duplicate spawn")
            return t

    state = {"last_pct": 0.0}

    def _watch():
        while True:
            # iter19 (2026-07-04): threshold on ANON (unreclaimable), NOT memory.current. The OOM-killer
            # enforces against anon; page cache (memory.current − anon) is reclaimed before any SIGKILL,
            # so thresholding the total false-alarms on cache-heavy streaming runs. Show reclaimable cache
            # as context so a high TOTAL no longer reads as danger.
            anon = read_cgroup_memory_anon()
            if anon is None:
                return
            total = read_cgroup_memory_usage()   # anon + reclaimable cache — context only
            pct = anon / mem_limit
            cache = (total - anon) if (total is not None and total >= anon) else None
            ctx = f" [+{_fmt_gb(cache)} reclaimable cache]" if cache else ""
            # Only emit when crossing a threshold upward, to keep log signal/noise high.
            if pct >= threshold_imminent and state["last_pct"] < threshold_imminent:
                print(f"\n{prefix} 🔥 IMMINENT SIGKILL: anon(unreclaimable) "
                      f"{_fmt_gb(anon)} / {_fmt_gb(mem_limit)} ({100*pct:.1f}%){ctx} "
                      f"— kernel may SIGKILL any moment", flush=True)
            elif pct >= threshold_crit and state["last_pct"] < threshold_crit:
                print(f"\n{prefix} 🚨 CRITICAL: anon(unreclaimable) "
                      f"{_fmt_gb(anon)} / {_fmt_gb(mem_limit)} ({100*pct:.1f}%){ctx} "
                      f"— reduce decode_workers / producer_queue / max_val_clips NOW", flush=True)
            elif pct >= threshold_warn and state["last_pct"] < threshold_warn:
                print(f"\n{prefix} ⚠️  warn: anon(unreclaimable) "
                      f"{_fmt_gb(anon)} / {_fmt_gb(mem_limit)} ({100*pct:.1f}%){ctx} "
                      f"— approaching cgroup cap", flush=True)
            # Re-arm downward too so a later real climb re-triggers (was upward-only).
            state["last_pct"] = pct
            time.sleep(interval_sec)

    t = threading.Thread(target=_watch, daemon=True, name="cgroup-oom-watchdog")
    t.start()
    print(f"{prefix} started (warn={int(100*threshold_warn)}%, "
          f"crit={int(100*threshold_crit)}%, "
          f"imminent={int(100*threshold_imminent)}%, "
          f"interval={interval_sec}s)")
    return t


# ── Self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_cgroup_header(prefix="[selftest]")
    print()
    print("Starting watchdog for 30 sec (low thresholds so it triggers on idle)...")
    start_oom_watchdog(threshold_warn=0.0001, threshold_crit=0.0002,
                        threshold_imminent=0.0003, interval_sec=2,
                        prefix="[selftest-watchdog]")
    time.sleep(30)
    print("\nSelf-test complete.", file=sys.stderr)
