# Compute Environment

## Hardware (training and evaluation)
- Up to 4x NVIDIA RTX Pro 6000 Blackwell GPUs, 96 GB VRAM per GPU, single node.

## Software
- PyTorch 2.12 (nightly build `dev20260408`, cu128 wheels)
- CUDA 13.0 toolkit; cuDNN
- Python 3.12
- FlashAttention-2
- BF16 mixed precision; scaled-dot-product attention (SDPA)

See `setup_env_uv.sh` for the exact, reproducible install steps and
`requirements_gpu.txt` / `requirements.txt` for the pinned dependency set.

## Runs and statistics
- A single training run per method (fixed random seed 17).
- All reported point estimates are that run's held-out score.
- All confidence intervals are 95% BCa bootstrap over the held-out
  evaluation set (resampling cities, source videos, and clips); the
  source-video group is the lowest independently resampled unit.
