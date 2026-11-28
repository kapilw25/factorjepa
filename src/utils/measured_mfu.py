"""Measured-FLOP MFU for eval stages whose GPU work is not one encoder pass (m12d, m12e).

Why measured, not analytic: utils.mfu_calculator's analytic 2N + 4·L·H·T count fits a single
encoder forward (m12a/m12b). m12d runs encoder + predictor with masks, and m12e runs several masked
encodes and predictor passes per clip (Mask-ratio sweeps 4 ratios), so no closed form matches.
torch.utils.flop_counter.FlopCounterMode counts the FLOPs of every matmul / conv / SDPA call the
batch actually dispatches. It is entered for the FIRST batch only (FLOPs per clip do not depend on
the clip: same frames, crop and masks), so the timed run pays its dispatch overhead once.

MFU = (FLOPs per clip × clips per second) / peak, with the peak MEASURED by the same dense bf16
GEMM benchmark as utils.mfu_calculator (pipeline.yaml mfu block). Under concurrent processes the
per-process peak benchmark sees a shared GPU, so the line also prints achieved TFLOP/s, which sums
across processes against an idle-GPU peak.

Refs: https://pytorch.org/docs/stable/_modules/torch/utils/flop_counter.html (FlopCounterMode);
PaLM App. B and stas00 ml-engineering for the MFU definition (achieved / peak FLOP/s).

USAGE (inside an m12 forward loop):
    mfu = MeasuredMFU(_PCFG, label="m12e maskratio")
    for batch in batches:
        with mfu.measure(n_clips=len(batch_keys)):
            ...GPU work for the batch...
    print(mfu.report())
"""
import contextlib
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from utils.mfu_calculator import measured_peak_flops


class MeasuredMFU:
    """FLOPs per clip from the first batch (FlopCounterMode), wall rate over every batch."""

    def __init__(self, pipeline_cfg: dict, label: str):
        self.label = label
        self.peak = measured_peak_flops(pipeline_cfg, torch.device("cuda"))
        self.flops_per_clip = None
        self.n_clips = 0
        self.t_first = None
        self.t_last = None

    @contextlib.contextmanager
    def measure(self, n_clips: int):
        if n_clips < 1:
            raise ValueError(f"MeasuredMFU.measure: n_clips={n_clips} must be >= 1")
        if self.t_first is None:
            self.t_first = time.perf_counter()
        if self.flops_per_clip is None:
            with FlopCounterMode(display=False) as fc:
                yield
            total = fc.get_total_flops()
            if total <= 0:
                raise RuntimeError(f"MeasuredMFU({self.label}): FlopCounterMode counted {total} FLOPs "
                                   f"for {n_clips} clips; no matmul/SDPA ran inside measure()")
            self.flops_per_clip = total / n_clips
        else:
            yield
        torch.cuda.synchronize()
        self.n_clips += n_clips
        self.t_last = time.perf_counter()

    def report(self) -> str:
        if self.flops_per_clip is None or self.t_last is None or self.t_last <= self.t_first:
            raise RuntimeError(f"MeasuredMFU({self.label}): report() before any measured batch")
        rate = self.n_clips / (self.t_last - self.t_first)
        achieved = self.flops_per_clip * rate
        return (f"  [mfu] {self.label}: {self.flops_per_clip / 1e12:.3f} TFLOP/clip x {rate:.3f} clip/s "
                f"= {achieved / 1e12:.1f} TFLOP/s achieved; peak {self.peak / 1e12:.0f} TFLOP/s "
                f"(dense bf16 GEMM, measured now) -> mfu={achieved / self.peak:.3f} "
                f"over {self.n_clips} clips")
