"""iter11 live-debug hooks — shared across m09a/m09b/m09c (any GPU training script).

Container GPU servers typically lack CAP_SYS_PTRACE → py-spy, gdb, strace CANNOT
attach to a running process from outside. The workaround: Python's built-in
`faulthandler` module registers signal handlers THAT SELF-DUMP stacks from INSIDE
the target process when triggered. No external privilege required.

USAGE from within a training module (m09a/m09b/m09c):
    from utils.live_debug import install_debug_handlers
    install_debug_handlers()   # call once at import / start of main()

Then while the job runs, in a separate terminal:
    PID=$(pgrep -f m09b_explora.py | head -1)    # or whichever m09x
    # One-shot stack dump of all threads to stderr (captured by `tee`):
    kill -USR1 $PID
    # Same, but uses the SIGUSR2 Python handler for extra context + PID banner:
    kill -USR2 $PID
    # On actual crash (segfault, abort), faulthandler prints tb to stderr
    # automatically — no action needed.

Rationale for two signals:
    SIGUSR1 → native C-level `faulthandler.register` — minimal, signal-safe,
              always-available even during torch.compile / CUDA stalls.
    SIGUSR2 → Python-level handler with banner (PID, thread count, timestamp).
              May be delayed if Python GIL is held by a long CUDA-sync call,
              but adds context that raw USR1 lacks.

Written iter11 (2026-04-24) after m09b POC v3 stall triage where py-spy was
blocked by container capabilities; the self-dump path unblocked debugging.
"""
import faulthandler
import os
import signal
import sys
import threading
import time


def install_debug_handlers(log_pid: bool = True) -> None:
    """Install SIGUSR1/SIGUSR2 stack-dump handlers + crash traceback hook.

    Args:
        log_pid: If True (default), prints a banner line with the PID and
                 the two `kill -USR{1,2} <PID>` commands so user sees
                 exactly what to type.
    """
    # 1. Native faulthandler for crash-time tb (segfault, abort, OOM-killed)
    faulthandler.enable()

    # 2. SIGUSR1 → native dump (signal-safe, works during CUDA-sync / compile)
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

    # 3. SIGUSR2 → Python-level dump with banner (may be delayed by GIL)
    def _sigusr2_dump(signum, frame):
        ts = time.strftime("%H:%M:%S")
        n_threads = threading.active_count()
        print(f"\n[live-debug] {ts} SIGUSR2 — dumping {n_threads} threads "
              f"(pid={os.getpid()}):", file=sys.stderr, flush=True)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        print(f"[live-debug] {ts} dump complete\n", file=sys.stderr, flush=True)

    signal.signal(signal.SIGUSR2, _sigusr2_dump)

    if log_pid:
        pid = os.getpid()
        print(f"[live-debug] installed — pid={pid}. Force a stack dump anytime via:",
              file=sys.stderr, flush=True)
        print(f"[live-debug]   kill -USR1 {pid}   # signal-safe native dump",
              file=sys.stderr, flush=True)
        print(f"[live-debug]   kill -USR2 {pid}   # Python dump with banner",
              file=sys.stderr, flush=True)
