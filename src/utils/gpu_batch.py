"""Auto-compute batch sizes based on available GPU VRAM. GPU-only, no CPU fallback."""
import argparse
import sys

# Baseline: tuned on A100-40GB
_BASELINE_VRAM_GB = 40

# (baseline_value, cap) — only for manual-batch pipelines
_PROFILES = {
    "vjepa":        (16, 64),    # V-JEPA ViT-G embedding batch
    "transformers": (4,  16),    # Qwen3-VL / VideoLLaMA3 / LLaVA-NeXT via transformers (sequential)
}


def compute_batch_sizes(gpu_vram_gb: float | None = None) -> dict[str, int]:
    """
    Auto-compute batch sizes scaled to detected (or overridden) GPU VRAM.

    Args:
        gpu_vram_gb: Override VRAM in GB. If None, auto-detects via torch.cuda.

    Returns:
        {"vjepa": int, "transformers": int}
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

    scale = gpu_vram_gb / _BASELINE_VRAM_GB

    sizes = {}
    for key, (baseline, cap) in _PROFILES.items():
        sizes[key] = max(1, min(int(baseline * scale), cap))

    print(f"GPU VRAM: {gpu_vram_gb:.0f} GB | scale={scale:.2f}x (baseline={_BASELINE_VRAM_GB}GB)")
    print(f"  batch sizes: vjepa={sizes['vjepa']}, transformers={sizes['transformers']}")
    return sizes


def add_gpu_mem_arg(parser: argparse.ArgumentParser):
    """Add --gpu-mem override to any argparse parser."""
    parser.add_argument("--gpu-mem", type=float, default=None,
                        help="Override GPU VRAM in GB (auto-detected if omitted)")
