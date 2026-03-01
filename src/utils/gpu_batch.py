"""Auto-compute batch sizes based on available GPU VRAM. GPU-only, no CPU fallback."""
import argparse
import sys

# ═════════════════════════════════════════════════════════════════════════
# VRAM COST MODEL — empirical from Qwen3-VL-8B on RTX PRO 6000 (96 GB)
# Measured: 9 clips = 23 GB total → model ~16 GB fixed + ~0.78 GB/clip
# ═════════════════════════════════════════════════════════════════════════
_VLM_MODEL_OVERHEAD_GB = 16.0       # Model weights (bf16)
_VLM_MARGINAL_GB_PER_CLIP = 0.78    # Per-clip: input tensors + KV cache + activations
_VLM_VRAM_TARGET = 0.80             # Target 80% utilization (AdaptiveBatchSizer fine-tunes)
_VLM_BATCH_CAP = 64                 # Hard cap regardless of VRAM

# V-JEPA: forward-only ViT-G (no KV cache) — linear scaling from 40 GB baseline
_VJEPA_BASELINE_VRAM_GB = 40
_VJEPA_PROFILE = (16, 64)           # (baseline_batch_at_40GB, cap)


def compute_batch_sizes(gpu_vram_gb: float | None = None) -> dict[str, int]:
    """
    Auto-compute batch sizes from detected (or --gpu-mem overridden) GPU VRAM.

    V-JEPA: linear scaling from A100-40GB baseline (no KV cache, different cost model).
    VLM: VRAM-based: max_batch = (vram × 80% − model_overhead) / marginal_per_clip.
         AdaptiveBatchSizer adjusts at runtime based on actual VRAM pressure.

    Args:
        gpu_vram_gb: Override VRAM in GB. If None, auto-detects via torch.cuda.

    Returns:
        {"vjepa": int, "transformers": int, "transformers_batch": int}
    """
    if gpu_vram_gb is None:
        try:
            import torch
            if not torch.cuda.is_available():
                print("FATAL: No CUDA GPU detected. Cannot compute batch sizes.")
                sys.exit(1)
            gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except ImportError:
            print("FATAL: torch not installed. Cannot detect GPU VRAM.")
            sys.exit(1)

    # ── V-JEPA: linear scaling (forward-only, no KV cache) ──
    scale = gpu_vram_gb / _VJEPA_BASELINE_VRAM_GB
    baseline, cap = _VJEPA_PROFILE
    vjepa = max(1, min(int(baseline * scale), cap))

    # ── VLM (transformers): VRAM-based auto-calculation ──
    usable_for_clips = gpu_vram_gb * _VLM_VRAM_TARGET - _VLM_MODEL_OVERHEAD_GB
    if usable_for_clips <= 0:
        vlm_max = 1
    else:
        vlm_max = max(1, min(int(usable_for_clips / _VLM_MARGINAL_GB_PER_CLIP), _VLM_BATCH_CAP))
    vlm_initial = max(1, int(vlm_max * 0.80))  # Conservative start, sizer grows

    sizes = {
        "vjepa": vjepa,
        "transformers": vlm_max,
        "transformers_batch": vlm_initial,
    }

    est_vram = _VLM_MODEL_OVERHEAD_GB + vlm_max * _VLM_MARGINAL_GB_PER_CLIP
    print(f"GPU VRAM: {gpu_vram_gb:.0f} GB | "
          f"VLM cost model: {_VLM_MODEL_OVERHEAD_GB:.0f}GB fixed + "
          f"{_VLM_MARGINAL_GB_PER_CLIP}GB/clip")
    print(f"  vjepa={vjepa} (linear, scale={scale:.2f}x from {_VJEPA_BASELINE_VRAM_GB}GB)")
    print(f"  transformers={vlm_max} (max batch, est {est_vram:.0f}GB = "
          f"{est_vram/gpu_vram_gb:.0%} VRAM)")
    print(f"  sub-batch={vlm_initial} (initial, AdaptiveBatchSizer grows to {vlm_max})")
    return sizes


# ═════════════════════════════════════════════════════════════════════════
# ADAPTIVE BATCH SIZER — VRAM-aware sub-batching with geometric OOM backoff
# ═════════════════════════════════════════════════════════════════════════

class AdaptiveBatchSizer:
    """VRAM-aware adaptive sub-batch sizing for batched model.generate().

    Monitors GPU memory via torch.cuda.mem_get_info() (CUDA driver-level)
    and adjusts sub-batch size to stay under memory_cap. On OOM, halves
    sub-batch size (geometric backoff) instead of falling to sequential.

    Usage:
        sizer = AdaptiveBatchSizer(initial_size=7, max_size=9)
        # In generate loop:
        sub_size = sizer.size
        try:
            results = model.generate(**sub_batch)
        except torch.cuda.OutOfMemoryError:
            oom = True
        # Cleanup OUTSIDE except block (critical — exception holds stack frame refs)
        if oom:
            gc.collect(); torch.cuda.empty_cache()
            sizer.on_oom()  # halves sub-batch
        else:
            sizer.after_batch_success()  # adjusts based on VRAM pressure
    """

    def __init__(self, initial_size: int, min_size: int = 1,
                 max_size: int | None = None, memory_cap: float = 0.85):
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size or initial_size
        self.memory_cap = memory_cap
        self._oom_count = 0

    @property
    def size(self) -> int:
        return self.current_size

    def after_batch_success(self):
        """After successful sub-batch: adjust size based on VRAM pressure."""
        import torch
        free, total = torch.cuda.mem_get_info(0)
        used_ratio = 1.0 - (free / total)

        if used_ratio > self.memory_cap:
            # Above cap — shrink by 1
            new = max(self.min_size, self.current_size - 1)
            if new != self.current_size:
                print(f"  AdaptiveBatch: VRAM {used_ratio:.0%} > {self.memory_cap:.0%} "
                      f"cap → sub-batch {self.current_size} → {new}")
                self.current_size = new
        elif used_ratio < self.memory_cap - 0.20 and self._oom_count == 0:
            # Well below cap (< 65%) AND no OOM history — grow by 1
            new = min(self.max_size, self.current_size + 1)
            if new != self.current_size:
                print(f"  AdaptiveBatch: VRAM {used_ratio:.0%} < {self.memory_cap - 0.20:.0%} "
                      f"→ sub-batch {self.current_size} → {new}")
                self.current_size = new

    def on_oom(self) -> bool:
        """On OOM: halve sub-batch size. Returns False if already at min (give up)."""
        self._oom_count += 1
        new = max(self.min_size, self.current_size // 2)
        if new == self.current_size and self.current_size <= self.min_size:
            print(f"  AdaptiveBatch: OOM #{self._oom_count} at min "
                  f"sub-batch={self.min_size} — giving up on this sub-batch")
            return False
        print(f"  AdaptiveBatch: OOM #{self._oom_count}! "
              f"sub-batch {self.current_size} → {new}")
        self.current_size = new
        return True

    def __repr__(self) -> str:
        return (f"AdaptiveBatchSizer(size={self.current_size}, "
                f"range=[{self.min_size}, {self.max_size}], "
                f"cap={self.memory_cap:.0%}, ooms={self._oom_count})")


def add_gpu_mem_arg(parser: argparse.ArgumentParser):
    """Add --gpu-mem override to any argparse parser."""
    parser.add_argument("--gpu-mem", type=float, default=None,
                        help="Override GPU VRAM in GB (auto-detected if omitted)")
