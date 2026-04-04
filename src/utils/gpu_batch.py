"""Auto-compute batch sizes based on available GPU VRAM. GPU-only, no CPU fallback."""
import argparse
import gc
import json
import shutil
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

# Image encoders (DINOv2-L ~300M, CLIP-L ~400M): single frame, 3-4x smaller than V-JEPA
# Per-clip cost ~10-50x cheaper (1 frame × 224² vs 64 frames × 384²)
_IMAGE_ENC_MULTIPLIER = 4           # 4x the V-JEPA batch size
_IMAGE_ENC_CAP = 256                # Hard cap


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

    # ── Image encoders (DINOv2, CLIP): 4x V-JEPA batch (much cheaper per clip) ──
    image_encoder = max(1, min(vjepa * _IMAGE_ENC_MULTIPLIER, _IMAGE_ENC_CAP))

    sizes = {
        "vjepa": vjepa,
        "image_encoder": image_encoder,
        "transformers": vlm_max,
        "transformers_batch": vlm_initial,
    }

    est_vram = _VLM_MODEL_OVERHEAD_GB + vlm_max * _VLM_MARGINAL_GB_PER_CLIP
    print(f"GPU VRAM: {gpu_vram_gb:.0f} GB | "
          f"VLM cost model: {_VLM_MODEL_OVERHEAD_GB:.0f}GB fixed + "
          f"{_VLM_MARGINAL_GB_PER_CLIP}GB/clip")
    print(f"  vjepa={vjepa} (linear, scale={scale:.2f}x from {_VJEPA_BASELINE_VRAM_GB}GB)")
    print(f"  image_encoder={image_encoder} (4x vjepa, cap {_IMAGE_ENC_CAP})")
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

    OOM_COOLDOWN = 50  # consecutive successes needed to reset _oom_count

    def __init__(self, initial_size: int, min_size: int = 1,
                 max_size: int | None = None, memory_cap: float = 0.85):
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size or initial_size
        self.memory_cap = memory_cap
        self._oom_count = 0
        self._consecutive_ok = 0

    @property
    def size(self) -> int:
        return self.current_size

    def after_batch_success(self):
        """After successful sub-batch: adjust size based on VRAM pressure."""
        import torch
        free, total = torch.cuda.mem_get_info(0)
        used_ratio = 1.0 - (free / total)

        # OOM cooldown: after N consecutive successes, allow growth again
        self._consecutive_ok += 1
        if self._oom_count > 0 and self._consecutive_ok >= self.OOM_COOLDOWN:
            print(f"  AdaptiveBatch: {self._consecutive_ok} consecutive OK batches "
                  f"→ resetting OOM count (was {self._oom_count})")
            self._oom_count = 0
            self._consecutive_ok = 0

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
        self._consecutive_ok = 0
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


def cuda_cleanup():
    """Force CUDA memory cleanup. Call between encoder runs or after OOM recovery."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def cleanup_temp():
    """Clean stale temp files from prior steps. Call at start of every GPU script main()."""
    from pathlib import Path
    for d in Path("/tmp").glob("hf_*"):
        shutil.rmtree(d, ignore_errors=True)
    for d in Path("/tmp").glob("m0*"):
        shutil.rmtree(d, ignore_errors=True)


def add_gpu_mem_arg(parser: argparse.ArgumentParser):
    """Add --gpu-mem override to any argparse parser."""
    parser.add_argument("--gpu-mem", type=float, default=None,
                        help="Override GPU VRAM in GB (auto-detected if omitted)")


def get_optimal_batch_size(profile_json: str = "outputs/profile/profile_data.json",
                           vram_pct: float = 0.75) -> int:
    """Read profiler results → max batch size at ≤vram_pct of GPU VRAM.

    USAGE (CLI):
        python -u src/utils/gpu_batch.py optimal-bs
        python -u src/utils/gpu_batch.py optimal-bs --profile-json outputs/profile/profile_data.json
    """
    d = json.load(open(profile_json))
    gpu_gb = d["gpu_total_gb"]
    target = gpu_gb * vram_pct
    best = 4
    for bs, info in sorted(d.get("grad_ckpt", {}).items(), key=lambda x: int(x[0])):
        if info["peak_gb"] > target:
            break
        best = int(bs)
    return best


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -u src/utils/gpu_batch.py optimal-bs [--profile-json PATH] [--vram-pct 0.75]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "optimal-bs":
        pj = "outputs/profile/profile_data.json"
        vp = 0.75
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--profile-json":
                pj = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--vram-pct":
                vp = float(sys.argv[i + 1])
                i += 2
            else:
                i += 1
        print(get_optimal_batch_size(pj, vp))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
