"""Shared tqdm progress bar factory for all src/m*.py scripts.

Every script must show: total count, current progress, rate, ETA.
Use make_pbar() instead of raw tqdm() to ensure consistent format.

USAGE:
    from utils.progress import make_pbar
    pbar = make_pbar(total=115687, desc="m06_metrics", unit="clip")
    for batch in data:
        process(batch)
        pbar.update(len(batch))
    pbar.close()

═══════════════════════════════════════════════════════════════════════════════
TQDM AUDIT — known gaps & how this module addresses them (iter16, 2026-05-23)
═══════════════════════════════════════════════════════════════════════════════
Gap                                  │ Status                                  │
─────────────────────────────────────┼─────────────────────────────────────────┤
1. smoothing=0.3 default → EWMA over │ ✅ FIXED via `smoothing=0` arg below.    │
   recent samples hides true rate    │ Forces total/elapsed (truthful aggregate)│
   on bursty workloads (tqdm docs).  │ for all 25 make_pbar callers.            │
─────────────────────────────────────┼─────────────────────────────────────────┤
2. start_t = bar_creation_time, NOT  │ ✅ FIXED via `pbar.start_t = _PROC_START`│
   process_spawn_time — excludes     │ override below. _PROC_START is captured  │
   warmup (model load + torch.compile│ at module-import time; per CLAUDE.md     │
   + producer queue cold-start) from │ "All imports at TOP", that's effectively │
   elapsed (tqdm #656).              │ process-spawn time.                      │
─────────────────────────────────────┼─────────────────────────────────────────┤
3. unpause() after long stalls       │ ⏭️ N/A — that pattern resets the rate    │
   (tqdm #861)                        │ counter to MASK real delays. We WANT    │
                                      │ delays counted (smoothing=0 truth model).│
                                      │ Calling unpause() would lie about elapsed│
─────────────────────────────────────┼─────────────────────────────────────────┤
4. Concurrent-process ETA bias on    │ ✅ Handled by smoothing=0. Issue #1161   │
   first-task-completes (tqdm #1161) │ describes EWMA-only instability; total/  │
                                      │ elapsed is monotonic and stable from clip│
                                      │ 1 onward.                                │
─────────────────────────────────────┼─────────────────────────────────────────┤
5. refresh() reports wrong elapsed   │ ⏭️ N/A — we don't call pbar.refresh()    │
   (tqdm #269)                        │ anywhere. .update(n) is the only mutator.│
─────────────────────────────────────┼─────────────────────────────────────────┤
6. Negative ETA on checkpoint resume │ ✅ Handled by tqdm's `initial=` semantics│
   (pytorch-lightning #13493)         │ — rate computes from (n-initial)/elapsed,│
                                      │ which is non-negative. We pass initial=  │
                                      │ start_count on every resume.             │
─────────────────────────────────────┴─────────────────────────────────────────┘
"""
import time
from tqdm import tqdm

_MIN_RATE_POINTS = 2   # iter18 W7: windowed rate needs a pair

# iter16 (2026-05-23): freeze the process-spawn baseline at module import. Per
# src/CLAUDE.md "All imports at TOP", `from utils.progress import make_pbar`
# runs at process spawn → _PROC_START ≈ true process spawn time (within ~ms).
# Used below to override tqdm's start_t so its `elapsed` counter reflects
# wall_clock_since_spawn (including warmup), NOT wall_clock_since_bar_creation.
# Prior 2026-05-22 m10 Stage 3 FULL run: tqdm showed ETA 75 hr while real wall
# was tracking ~133 hr — 3.5 hr of pre-bar warmup (model load + torch.compile +
# producer queue cold-start) was being excluded from elapsed. This baseline
# closes that gap for all ~25 make_pbar callers in one place.
_PROC_START = time.time()


class _RecentRateBar(tqdm):
    """tqdm subclass that shows BOTH whole-run avg AND recent-window rate.

    iter16 (2026-05-23): the 2026-05-22 Stage 3 FULL run exposed that tqdm's
    default {rate_fmt} field (with our smoothing=0) is whole-run total/elapsed
    — it's dragged DOWN by an initial ~3.5 hr warmup, so post-warmup the
    headline rate stays at 16.9 s/clip even though the steady-state rate is
    ~5.5 s/clip. The user manually computed the delta from disk to find the
    truth — that observation should be in the bar itself.

    This subclass tracks a sliding window of recent (n, t) samples (default 60 s)
    and writes `recent=Xs/{unit} ETA{H}h` to the postfix on every update().
    The MAIN bar still shows whole-run avg + ETA (so the operator sees BOTH
    the conservative aggregate and the post-warmup current rate side by side).

    Trade-offs:
    • Callers that call set_postfix(...) themselves (e.g., m04d's manual
      rate display at m04d_motion_features.py:749) will have their postfix
      OVERWRITTEN on the next update(). For m04d this is fine — its postfix
      and ours both show "recent rate", just different windows; ours wins.
      For any future caller that needs a dedicated postfix, switch back to
      raw `tqdm()` instead of `make_pbar()`.
    • Memory: the sliding window grows with throughput; trimmed to 60-sec
      cutoff so size is bounded by clips/min × 60.
    """

    def __init__(self, *args, recent_window_sec: float = None, **kwargs):
        super().__init__(*args, **kwargs)
        # iter18 H6: None → pipeline.yaml plots.progress_recent_window_sec.
        if recent_window_sec is None:
            from utils.config import get_pipeline_config
            recent_window_sec = get_pipeline_config()["plots"]["progress_recent_window_sec"]
        self._rwin_sec = recent_window_sec
        self._rwin = []  # list of (n, time) tuples; oldest first

    def update(self, n: int = 1):
        ret = super().update(n)
        now = time.time()
        self._rwin.append((self.n, now))
        cutoff = now - self._rwin_sec
        # Trim entries older than the sliding window.
        while self._rwin and self._rwin[0][1] < cutoff:
            self._rwin.pop(0)
        # Need ≥2 samples spanning some wall time to compute rate.
        if len(self._rwin) >= _MIN_RATE_POINTS:
            n0, t0 = self._rwin[0]
            n1, t1 = self._rwin[-1]
            dt = t1 - t0
            dn = n1 - n0
            if dt > 0 and dn > 0:
                sec_per_unit = dt / dn
                recent_eta_hr = max(0.0, (self.total - self.n) * sec_per_unit / 3600.0)
                # set_postfix() with kwargs renders ", recent=...". The leading
                # comma + space is auto-prepended by tqdm. `refresh=False` lets
                # tqdm's own mininterval/maxinterval throttle the actual draw.
                self.set_postfix_str(
                    f"recent={sec_per_unit:.1f}s/{self.unit} ETA{recent_eta_hr:.1f}h",
                    refresh=False,
                )
        return ret


def make_pbar(total: int, desc: str, unit: str = "item",
              initial: int = 0,
              recent_window_sec: float = None) -> tqdm:  # iter18 H6: None → plots.progress_recent_window_sec
    """Create a standardized tqdm progress bar with truthful ETA.

    Args:
        total: Total number of items to process.
        desc: Short description (e.g., "m06_metrics", "m05_vjepa").
        unit: Unit name (e.g., "clip", "step", "video", "shard").
        initial: Starting count (for checkpoint resume).
        recent_window_sec: Sliding-window size (seconds) for the recent-rate
                           postfix. Default 60 sec — large enough to absorb
                           tqdm update jitter, small enough to react quickly
                           to a regime change.

    Returns:
        tqdm subclass with main bar = whole-run avg, postfix = recent window.
        Caller must call .update(n) and .close().

    Gap-coverage matrix: see module docstring TQDM AUDIT block above.
    """
    pbar = _RecentRateBar(
        total=total,
        initial=initial,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        smoothing=0,              # Gap 1: total/elapsed (not EWMA)
        # Bar shows aggregate; {postfix} appends recent-window rate + ETA.
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        ),
        recent_window_sec=recent_window_sec,
    )
    pbar.start_t = _PROC_START    # Gap 2: include warmup in elapsed
    return pbar
