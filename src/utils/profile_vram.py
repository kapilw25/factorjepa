"""VRAM profiler for V-JEPA 2 ViT-g. GPU-only.

python -u src/utils/profile_vram.py --training 2>&1 | tee logs/profile_vram_training.log && \
python -u src/utils/profile_vram.py --inference 2>&1 | tee logs/profile_vram_inference.log && \
python -u src/utils/profile_vram.py --dinov2 2>&1 | tee logs/profile_vram_dinov2.log
    
    
  Training VRAM at BS=112: 62.5 GB (from profile_data.json)
  - Student: ~3.8 GB (1B params × 2 bytes bf16)
  - Teacher: ~3.8 GB (frozen copy)
  - Optimizer: ~7.5 GB (Adam m+v in fp32)
  - Predictor: ~0.1 GB
  - Activations + backward: ~47 GB

  Embedding VRAM at BS=112 (inference only):
  - Student: ~3.8 GB (only model loaded)
  - No teacher, no predictor, no optimizer, no backward
  - Activations (forward only, no grad): ~10-15 GB estimate
  - Total: ~15-20 GB — meaning BS=256+ should fit in 95 GB
    
"""

import gc
import json
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Import vjepa2 modules (CWD trick avoids src/ namespace collision with our src/)
import os as _os
import importlib as _il
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_saved_cwd = _os.getcwd()
_os.chdir("/tmp")
_saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "src" or k.startswith("src.")}
sys.path.insert(0, str(REPO_ROOT / "deps" / "vjepa2"))
_il.invalidate_caches()
_il.import_module("src.models.vision_transformer")
_il.import_module("src.models.predictor")
_il.import_module("src.masks.multiseq_multiblock3d")
_il.import_module("src.masks.utils")
_os.chdir(_saved_cwd)
for _k in ["src", "src.utils"]:
    if _k in _saved_mods:
        sys.modules[_k] = _saved_mods[_k]
_il.invalidate_caches()
vit_giant_xformers = sys.modules["src.models.vision_transformer"].vit_giant_xformers
vit_predictor = sys.modules["src.models.predictor"].vit_predictor

# Shared video I/O (Rule 32: no cross-imports between m*.py)
from utils.video_io import decode_video_bytes
# Phase 3 of #49: m09_pretrain.py was split into m09a/b/c. load_config now lives in
# utils.training. DEFAULT_CONFIG was a legacy name (never defined post-rename to
# DEFAULT_TRAIN_CONFIG); define locally here to preserve the one consumer at line 625.
from utils.training import load_config
DEFAULT_CONFIG = "configs/legacy2/ch10_pretrain.yaml"  # profiler defaults to Ch10 pretrain config
_MaskGenerator = sys.modules["src.masks.multiseq_multiblock3d"]._MaskGenerator
apply_masks = sys.modules["src.masks.utils"].apply_masks

# Import shared plot utilities (after vjepa2 sys.path hack is cleaned up)
sys.path.insert(0, str(REPO_ROOT / "src"))
from utils.plots import init_style, save_fig, COLORS

# Number of repeats per batch size for reliable throughput measurement
N_REPEATS = 3

# ── Config (matches plan_code_dev.md + Q1-Q5 corrections) ────────────
CROP_SIZE = 384
PATCH_SIZE = 16
NUM_FRAMES = 16
TUBELET_SIZE = 2
EMBED_DIM = 1408
LOSS_EXP = 1.0
EMA_MOMENTUM = 0.99925

# Token grid: (384/16)^2 * (16/2) = 24*24*8 = 4608
SPATIAL_GRID = CROP_SIZE // PATCH_SIZE  # 24
TEMPORAL_GRID = NUM_FRAMES // TUBELET_SIZE  # 8
TOTAL_TOKENS = SPATIAL_GRID * SPATIAL_GRID * TEMPORAL_GRID  # 4608

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 136, 144, 148, 152, 156, 160, 192, 224, 256]
DTYPE = torch.bfloat16


def build_models(device, use_activation_checkpointing=False):
    """Build student, teacher, predictor on device."""
    encoder_kwargs = dict(
        img_size=(CROP_SIZE, CROP_SIZE),
        patch_size=PATCH_SIZE,
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        use_activation_checkpointing=use_activation_checkpointing,
    )

    student = vit_giant_xformers(**encoder_kwargs).to(device)

    teacher = vit_giant_xformers(**encoder_kwargs).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    pred = vit_predictor(
        img_size=(CROP_SIZE, CROP_SIZE),
        patch_size=PATCH_SIZE,
        num_frames=NUM_FRAMES,
        tubelet_size=TUBELET_SIZE,
        embed_dim=EMBED_DIM,
        predictor_embed_dim=384,
        depth=12,
        num_heads=12,
        use_mask_tokens=True,
        num_mask_tokens=2,      # 2 mask generators (8-small + 2-large)
        zero_init_mask_tokens=True,
        use_rope=True,
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=use_activation_checkpointing,
    ).to(device)

    return student, teacher, pred


def build_mask_generators():
    """Build the 2 mask generators (8 small blocks + 2 large blocks)."""
    mg_small = _MaskGenerator(
        crop_size=(CROP_SIZE, CROP_SIZE),
        num_frames=NUM_FRAMES,
        spatial_patch_size=(PATCH_SIZE, PATCH_SIZE),
        temporal_patch_size=TUBELET_SIZE,
        spatial_pred_mask_scale=(0.15, 0.15),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5),
        npred=8,
    )
    mg_large = _MaskGenerator(
        crop_size=(CROP_SIZE, CROP_SIZE),
        num_frames=NUM_FRAMES,
        spatial_patch_size=(PATCH_SIZE, PATCH_SIZE),
        temporal_patch_size=TUBELET_SIZE,
        spatial_pred_mask_scale=(0.7, 0.7),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.75, 1.5),
        npred=2,
    )
    return [mg_small, mg_large]


def profile_batch(student, teacher, pred, mask_generators, batch_size, device,
                   real_batch=None):
    """Run one forward+backward step, return detailed profiling dict.
    Uses real video data if provided, else falls back to synthetic."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    waterfall = {}  # phase → VRAM snapshot (GB)

    def snap(label):
        torch.cuda.synchronize()
        waterfall[label] = torch.cuda.memory_allocated(device) / (1024 ** 3)

    snap("baseline")

    # Real or synthetic video: (B, C, T, H, W)
    if real_batch is not None and real_batch.shape[0] >= batch_size:
        clip = real_batch[:batch_size].to(device=device, dtype=DTYPE)
        # Ensure (B, C, T, H, W) format
        if clip.ndim == 5 and clip.shape[2] in (1, 3):
            clip = clip.permute(0, 2, 1, 3, 4)
    else:
        clip = torch.randn(batch_size, 3, NUM_FRAMES, CROP_SIZE, CROP_SIZE,
                            device=device, dtype=DTYPE)
    snap("data_loaded")

    # Generate masks from each generator
    all_masks_enc, all_masks_pred = [], []
    n_visible_tokens = []
    for mg in mask_generators:
        m_enc, m_pred = mg(batch_size)
        all_masks_enc.append(m_enc.to(device))
        all_masks_pred.append(m_pred.to(device))
        n_visible_tokens.append(m_enc.shape[1])
    snap("masks_generated")

    # Forward
    with torch.amp.autocast("cuda", dtype=DTYPE):
        # Teacher: all tokens, no grad
        with torch.no_grad():
            h = teacher(clip)  # (B, N_total, D)
            h = F.layer_norm(h, (h.size(-1),))
        snap("teacher_forward")

        # Student + Predictor: one pass per mask generator
        total_loss = torch.tensor(0.0, device=device)
        for i, (m_enc, m_pred) in enumerate(zip(all_masks_enc, all_masks_pred)):
            z = student(clip, masks=[m_enc])          # (B, N_visible, D)
            snap(f"student_forward_mg{i}")
            p = pred(z, [m_enc], [m_pred], mask_index=i)  # (B, N_masked, D_out)
            snap(f"predictor_forward_mg{i}")

            # L1 loss at masked positions
            h_masked = apply_masks(h, [m_pred])       # (B, N_masked, D)
            loss = torch.mean(torch.abs(p - h_masked) ** LOSS_EXP) / LOSS_EXP
            total_loss = total_loss + loss

        total_loss = total_loss / len(mask_generators)
    snap("loss_computed")

    # Backward
    total_loss.backward()
    snap("backward_done")

    # Read peak and stats
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    stats = torch.cuda.memory_stats(device)

    return {
        "peak_gb": peak_gb,
        "waterfall": waterfall,
        "n_visible_tokens": n_visible_tokens,
        "batch_size": batch_size,
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
    }


def generate_plots(results_no_ckpt, results_ckpt, gpu_name, gpu_total_gb,
                    n_student_params, out_dir):
    """Generate 5 diagnostic plots from profiling data."""
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
        "axes.grid": True, "grid.alpha": 0.3,
    })

    # ── Plot 1: Batch Size Scaling ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bs_no = sorted(results_no_ckpt.keys())
    bs_ck = sorted(results_ckpt.keys())
    peak_no = [results_no_ckpt[b]["peak_gb"] for b in bs_no]
    peak_ck = [results_ckpt[b]["peak_gb"] for b in bs_ck]

    ax.plot(bs_no, peak_no, "o-", linewidth=2.2, markersize=8,
            label="No grad checkpointing", color="#E53935", zorder=3)
    ax.plot(bs_ck, peak_ck, "s--", linewidth=2.2, markersize=8,
            label="Gradient checkpointing", color="#1E88E5", zorder=3)
    ax.axhline(y=gpu_total_gb, color="#757575", linestyle=":", linewidth=1.5,
               label=f"VRAM limit ({gpu_total_gb:.0f} GB)")
    ax.axhline(y=gpu_total_gb * 0.90, color="#BDBDBD", linestyle=":",
               linewidth=1, label="90% safe zone")
    for b, p in zip(bs_no, peak_no):
        ax.annotate(f"{p:.1f}G", (b, p), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#C62828")
    for b, p in zip(bs_ck, peak_ck):
        ax.annotate(f"{p:.1f}G", (b, p), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8, color="#1565C0")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title(f"Batch Size Scaling — ViT-g 1B on {gpu_name}")
    ax.set_xticks(BATCH_SIZES[:max(len(bs_no), len(bs_ck))])
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / "plot1_batch_scaling.png", dpi=150)
    plt.close(fig)
    print("  [1/5] Saved plot1_batch_scaling.png")

    # ── Plot 2: Grad Checkpointing Savings (Grouped Bar) ─────────────
    common_bs = sorted(set(bs_no) & set(bs_ck))
    if common_bs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(common_bs))
        w = 0.35
        vals_no = [results_no_ckpt[b]["peak_gb"] for b in common_bs]
        vals_ck = [results_ckpt[b]["peak_gb"] for b in common_bs]
        savings = [(1 - ck / no) * 100 for no, ck in zip(vals_no, vals_ck)]

        bars1 = ax1.bar(x - w/2, vals_no, w, label="No checkpointing", color="#E53935")
        bars2 = ax1.bar(x + w/2, vals_ck, w, label="Grad checkpointing", color="#1E88E5")
        ax1.bar_label(bars1, fmt="%.1f", fontsize=8, padding=2)
        ax1.bar_label(bars2, fmt="%.1f", fontsize=8, padding=2)
        ax1.axhline(y=gpu_total_gb, color="#757575", linestyle=":", linewidth=1)
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Peak VRAM (GB)")
        ax1.set_title("VRAM: With vs Without Checkpointing")
        ax1.set_xticks(x)
        ax1.set_xticklabels(common_bs)
        ax1.legend()

        ax2.bar(x, savings, 0.5, color="#43A047")
        ax2.bar_label(ax2.containers[0], fmt="%.1f%%", fontsize=9, padding=2)
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("VRAM Savings (%)")
        ax2.set_title("Memory Saved by Gradient Checkpointing")
        ax2.set_xticks(x)
        ax2.set_xticklabels(common_bs)
        ax2.set_ylim(0, max(savings) * 1.3 if savings else 50)

        plt.tight_layout()
        fig.savefig(out_dir / "plot2_ckpt_savings.png", dpi=150)
        plt.close(fig)
        print("  [2/5] Saved plot2_ckpt_savings.png")

    # ── Plot 3: Memory Breakdown (Stacked Bar) ───────────────────────
    # Estimate components from the model params
    param_bytes_bf16 = n_student_params * 2  # bf16
    student_gb = param_bytes_bf16 / (1024**3)
    teacher_gb = student_gb  # frozen copy
    optimizer_gb = n_student_params * 8 / (1024**3)  # Adam: m + v in fp32
    predictor_gb = 0.05  # ~22M params, negligible

    fig, ax = plt.subplots(figsize=(9, 5))
    if common_bs:
        categories = ["Params\n(student)", "Params\n(teacher)", "Optimizer\n(Adam m+v)",
                       "Predictor", "Activations\n+ buffers"]
        for i, bs in enumerate(common_bs[:4]):  # Plot up to 4 batch sizes
            measured_peak = results_ckpt[bs]["peak_gb"]
            fixed = student_gb + teacher_gb + optimizer_gb + predictor_gb
            activations = max(0, measured_peak - fixed)
            vals = [student_gb, teacher_gb, optimizer_gb, predictor_gb, activations]
            colors = ["#1565C0", "#546E7A", "#FF8F00", "#7B1FA2", "#C62828"]
            bottom = 0
            for j, (v, c) in enumerate(zip(vals, colors)):
                label = categories[j] if i == 0 else None
                ax.bar(i, v, bottom=bottom, color=c, label=label, edgecolor="white",
                       linewidth=0.5, width=0.6)
                bottom += v
            ax.text(i, bottom + 0.3, f"{measured_peak:.1f}G", ha="center",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(common_bs[:4])))
        ax.set_xticklabels([f"BS={b}" for b in common_bs[:4]])
        ax.set_ylabel("VRAM (GB)")
        ax.set_title("Memory Breakdown (Grad Ckpt ON)")
        ax.legend(loc="upper left", fontsize=9)
        plt.tight_layout()
        fig.savefig(out_dir / "plot3_breakdown.png", dpi=150)
        plt.close(fig)
        print("  [3/5] Saved plot3_breakdown.png")

    # ── Plot 4: Memory Waterfall (single step, largest common BS) ────
    ref_bs = common_bs[-1] if common_bs else (bs_ck[-1] if bs_ck else None)
    if ref_bs and ref_bs in results_ckpt:
        wf = results_ckpt[ref_bs]["waterfall"]
        phases = list(wf.keys())
        mem_vals = list(wf.values())

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(phases))
        ax.fill_between(x, mem_vals, step="mid", alpha=0.25, color="#1E88E5")
        ax.step(x, mem_vals, where="mid", linewidth=2.2, color="#1565C0")
        ax.scatter(x, mem_vals, zorder=5, color="#1565C0", s=50, edgecolors="white")
        peak_idx = mem_vals.index(max(mem_vals))
        ax.annotate(f"PEAK\n{max(mem_vals):.1f}G",
                    xy=(peak_idx, max(mem_vals)),
                    xytext=(peak_idx, max(mem_vals) + 1),
                    ha="center", fontsize=10, fontweight="bold", color="#C62828",
                    arrowprops=dict(arrowstyle="->", color="#C62828"))
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_", "\n") for p in phases],
                           rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("VRAM Allocated (GB)")
        ax.set_title(f"Memory Waterfall — Single Step (BS={ref_bs}, Grad Ckpt)")
        plt.tight_layout()
        fig.savefig(out_dir / "plot4_waterfall.png", dpi=150)
        plt.close(fig)
        print("  [4/5] Saved plot4_waterfall.png")

    # ── Plot 5: Visible Tokens vs Peak VRAM ──────────────────────────
    # Unique to JEPA: masking ratio directly affects student memory
    if results_ckpt:
        fig, ax = plt.subplots(figsize=(9, 5))
        for bs, info in sorted(results_ckpt.items()):
            vis_tokens = info["n_visible_tokens"]
            avg_vis = sum(vis_tokens) / len(vis_tokens)
            mask_pct = (1 - avg_vis / TOTAL_TOKENS) * 100
            ax.scatter(avg_vis, info["peak_gb"], s=bs * 30 + 50, zorder=3,
                       edgecolors="white", linewidth=1)
            ax.annotate(f"BS={bs}\n{mask_pct:.0f}% masked",
                        (avg_vis, info["peak_gb"]),
                        textcoords="offset points", xytext=(12, 0),
                        fontsize=8, va="center")

        ax.axhline(y=gpu_total_gb, color="#757575", linestyle=":", linewidth=1,
                   label=f"VRAM limit ({gpu_total_gb:.0f} GB)")
        ax.axvline(x=TOTAL_TOKENS, color="#FF9800", linestyle="--", linewidth=1,
                   label=f"Full tokens ({TOTAL_TOKENS})")
        ax.set_xlabel(f"Avg Visible Tokens (of {TOTAL_TOKENS} total)")
        ax.set_ylabel("Peak VRAM (GB)")
        ax.set_title("JEPA Masking Effect — Visible Tokens vs VRAM")
        ax.legend()
        plt.tight_layout()
        fig.savefig(out_dir / "plot5_masking_effect.png", dpi=150)
        plt.close(fig)
        print("  [5/5] Saved plot5_masking_effect.png")

    print(f"\nAll plots saved to {out_dir}/")


def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available. Run this on the GPU machine.")
        print("       This script is for profiling only — no CPU fallback.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    gpu_total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name} ({gpu_total_gb:.1f} GB)")
    print(f"Config: ViT-g (1B), {CROP_SIZE}px, {NUM_FRAMES}f, bf16")
    print(f"Token grid: {SPATIAL_GRID}x{SPATIAL_GRID}x{TEMPORAL_GRID} = {TOTAL_TOKENS} tokens")
    print()

    # Load real video clips for realistic profiling
    max_bs = max(BATCH_SIZES)
    print(f"Loading {max_bs} real video clips from data/full_local...")
    try:
        real_batch = _load_real_video_batch(max_bs)
        print(f"Real batch: {real_batch.shape}")
    except Exception as e:
        print(f"WARN: Could not load real data ({e}), falling back to synthetic")
        real_batch = None

    # ── Run 1: Without gradient checkpointing ────────────────────────
    print("=" * 60)
    print("Run 1: WITHOUT gradient checkpointing")
    print("=" * 60)
    student, teacher, pred = build_models(device, use_activation_checkpointing=False)

    n_student = sum(p.numel() for p in student.parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_pred = sum(p.numel() for p in pred.parameters())
    print(f"Student: {n_student / 1e9:.2f}B params")
    print(f"Teacher: {n_teacher / 1e9:.2f}B params (frozen)")
    print(f"Predictor: {n_pred / 1e6:.1f}M params")
    print()

    mask_generators = build_mask_generators()

    # Build optimizer (needed for realistic memory footprint)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(pred.parameters()),
        lr=1e-5, weight_decay=0.04,
    )

    def run_sweep():
        """Run batch-size sweep, return {bs: profile_dict}."""
        nonlocal student, teacher, pred, optimizer, mask_generators
        print(f"{'Batch':>6}  {'Peak VRAM':>10}  {'Visible':>10}  {'Status':>8}  {'Time':>7}")
        print("-" * 55)
        results = {}
        for bs in BATCH_SIZES:
            optimizer.zero_grad()
            try:
                # Run N_REPEATS times, take median time
                times = []
                info = None
                for _ in range(N_REPEATS):
                    optimizer.zero_grad()
                    t0 = time.time()
                    info = profile_batch(student, teacher, pred, mask_generators, bs, device,
                                         real_batch=real_batch)
                    times.append(time.time() - t0)
                dt = sorted(times)[len(times) // 2]  # median
                peak = info["peak_gb"]
                vis = info["n_visible_tokens"]
                status = "OK" if peak < gpu_total_gb * 0.95 else "TIGHT"
                print(f"{bs:>6}  {peak:>9.1f}G  {str(vis):>10}  {status:>8}  {dt:>6.1f}s")
                info["time_s"] = dt
                results[bs] = info
            except torch.cuda.OutOfMemoryError:
                print(f"{bs:>6}  {'OOM':>10}  {'—':>10}  {'FAILED':>8}")
                torch.cuda.empty_cache()
                gc.collect()
                break
        return results

    results_no_ckpt = run_sweep()

    # Cleanup
    del student, teacher, pred, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    # ── Run 2: With gradient checkpointing ───────────────────────────
    print()
    print("=" * 60)
    print("Run 2: WITH gradient checkpointing")
    print("=" * 60)
    student, teacher, pred = build_models(device, use_activation_checkpointing=True)

    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(pred.parameters()),
        lr=1e-5, weight_decay=0.04,
    )

    results_ckpt = run_sweep()

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"GPU: {gpu_name} ({gpu_total_gb:.1f} GB)")
    print()

    max_bs_no_ckpt = max(results_no_ckpt.keys()) if results_no_ckpt else 0
    max_bs_ckpt = max(results_ckpt.keys()) if results_ckpt else 0

    print(f"Max batch (no grad ckpt):   {max_bs_no_ckpt}  "
          f"({results_no_ckpt[max_bs_no_ckpt]['peak_gb']:.1f}G peak)" if max_bs_no_ckpt else "")
    print(f"Max batch (grad ckpt):      {max_bs_ckpt}  "
          f"({results_ckpt[max_bs_ckpt]['peak_gb']:.1f}G peak)" if max_bs_ckpt else "")
    print()
    print("Recommended for m09_pretrain.py:")
    if max_bs_ckpt >= 8:
        rec = min(max_bs_ckpt, 16)
        print(f"  batch_size: {rec}  (with gradient checkpointing)")
    elif max_bs_no_ckpt >= 4:
        rec = min(max_bs_no_ckpt, 8)
        print(f"  batch_size: {rec}  (without gradient checkpointing)")
    else:
        print("  WARNING: ViT-g may not fit at batch=4. Consider ViT-L for POC.")

    # ── Save raw data + generate plots ───────────────────────────────
    out_dir = REPO_ROOT / "outputs" / "profile" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute throughput for training results
    for results_dict in [results_no_ckpt, results_ckpt]:
        for bs, info in results_dict.items():
            info["throughput"] = bs / info["time_s"] if info["time_s"] > 0 else 0

    # Find optimal training BS: largest BS under 75% VRAM
    # Training clips/s is flat across BS (~2.5 clips/s) — larger BS = fewer optimizer steps
    train_threshold = gpu_total_gb * 0.75
    safe_ckpt = {bs: info for bs, info in results_ckpt.items() if info["peak_gb"] < train_threshold}
    if safe_ckpt:
        train_optimal = max(safe_ckpt.keys())
    else:
        train_optimal = max(results_ckpt.keys()) if results_ckpt else 1
    train_tput = results_ckpt[train_optimal]["throughput"] if train_optimal in results_ckpt else 0

    raw = {
        "gpu": gpu_name, "gpu_total_gb": gpu_total_gb,
        "config": {"crop": CROP_SIZE, "frames": NUM_FRAMES, "tokens": TOTAL_TOKENS},
        "training": {
            "optimal_bs": train_optimal,
            "optimal_throughput": train_tput,
            "peak_gb_at_optimal": results_ckpt[train_optimal]["peak_gb"] if train_optimal in results_ckpt else 0,
            "selection_method": "max_throughput_under_75pct_vram",
        },
        "no_ckpt": {str(k): {"peak_gb": v["peak_gb"], "time_s": v["time_s"],
                              "throughput": v["throughput"],
                              "visible_tokens": v["n_visible_tokens"],
                              "waterfall": v["waterfall"]}
                    for k, v in results_no_ckpt.items()},
        "grad_ckpt": {str(k): {"peak_gb": v["peak_gb"], "time_s": v["time_s"],
                                "throughput": v["throughput"],
                                "visible_tokens": v["n_visible_tokens"],
                                "waterfall": v["waterfall"]}
                      for k, v in results_ckpt.items()},
    }
    with open(out_dir / "profile_data.json", "w") as f:
        json.dump(raw, f, indent=2)

    generate_plots(results_no_ckpt, results_ckpt, gpu_name, gpu_total_gb, n_student, out_dir)

    # Training BS vs VRAM vs Throughput (uses shared plots.py for publication quality)
    for tag, res_dict in [("no_ckpt", results_no_ckpt), ("grad_ckpt", results_ckpt)]:
        if not res_dict:
            continue
        batch_sizes = sorted(res_dict.keys())
        vram = [res_dict[bs]["peak_gb"] for bs in batch_sizes]
        tput = [res_dict[bs]["throughput"] for bs in batch_sizes]
        label_step = max(1, len(batch_sizes) // 12)

        init_style()
        fig, ax1 = plt.subplots(figsize=(12, 6))
        bar_w = max(1, (batch_sizes[-1] - batch_sizes[0]) // len(batch_sizes)) * 0.6
        ax1.bar(batch_sizes, vram, width=bar_w, color=COLORS["blue"], alpha=0.5, label="Peak VRAM (GB)")
        ax1.axhline(y=gpu_total_gb, color="#757575", linestyle=":", linewidth=1.5,
                    label=f"VRAM limit ({gpu_total_gb:.0f} GB)")
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Peak VRAM (GB)")
        ax1.set_xticks([batch_sizes[i] for i in range(0, len(batch_sizes), label_step)])
        ax1.set_xticklabels([str(batch_sizes[i]) for i in range(0, len(batch_sizes), label_step)],
                            rotation=45, ha="right", fontsize=10)

        ax2 = ax1.twinx()
        ax2.plot(batch_sizes, tput, "s-", color=COLORS["red"], linewidth=3.0, markersize=5,
                 label="Throughput (steps/s)", zorder=10)
        ax2.set_ylabel("Throughput (steps/s)")

        if tag == "grad_ckpt" and train_optimal in batch_sizes:
            ax2.axvline(x=train_optimal, color=COLORS["green"], linestyle="--", linewidth=2,
                        label=f"Optimal BS={train_optimal} ({train_tput:.2f} steps/s)")

        ckpt_str = "Grad Checkpointing" if tag == "grad_ckpt" else "No Checkpointing"
        ax1.set_title(f"Training Profiling ({ckpt_str}) — ViT-g 1B")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=11)
        plt.subplots_adjust(bottom=0.25)
        save_fig(fig, str(out_dir / f"plot_training_profile_{tag}"))


def _load_real_video_batch(n_clips: int, local_data: str = "data/full_local"):
    """Load real videos using the pipeline's video decode + config functions."""
    from transformers import AutoVideoProcessor
    from utils.config import VJEPA_MODEL_ID

    cfg = load_config(str(REPO_ROOT / DEFAULT_CONFIG))
    num_frames = cfg["data"]["num_frames"]
    crop_size = cfg["data"]["crop_size"]

    processor = AutoVideoProcessor.from_pretrained(
        VJEPA_MODEL_ID, size={"height": crop_size, "width": crop_size})

    tar_files = sorted(Path(local_data).glob("*.tar"))
    clips = []
    tmp_dir = tempfile.mkdtemp(prefix="profile_")

    for tar_path in tar_files:
        if len(clips) >= n_clips:
            break
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".mp4"):
                    continue
                mp4_bytes = tar.extractfile(member).read()
                tensor = decode_video_bytes(mp4_bytes, tmp_dir, member.name, num_frames)
                if tensor is not None:
                    clips.append(tensor)
                if len(clips) >= n_clips:
                    break

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not clips:
        print("FATAL: No video clips found in local data for profiling")
        sys.exit(1)

    processed = processor(clips, return_tensors="pt")
    return processed["pixel_values_videos"]


def profile_inference():
    """Profile V-JEPA inference with REAL video data + torch.compile (realistic VRAM).
        python -u src/utils/profile_vram.py --inference 2>&1 | tee logs/profile_vram_inference.log
    """
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    gpu_total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name} ({gpu_total_gb:.1f} GB)")
    print("=== INFERENCE PROFILING (REAL video data + torch.compile) ===")
    print()

    # Load real video clips for realistic profiling
    max_bs = 256  # max we'll test
    print(f"Loading {max_bs} real video clips from data/full_local...")
    real_batch = _load_real_video_batch(max_bs)
    print(f"Real batch shape: {real_batch.shape} (dtype={real_batch.dtype})")

    # Build model exactly like m05 does
    student = vit_giant_xformers(
        img_size=(CROP_SIZE, CROP_SIZE), patch_size=PATCH_SIZE,
        num_frames=NUM_FRAMES, tubelet_size=TUBELET_SIZE,
        use_sdpa=True, use_silu=False, wide_silu=True,
        uniform_power=False, use_rope=True,
    ).to(device=device, dtype=torch.float16)
    student.eval()
    compiled = torch.compile(student)

    n_params = sum(p.numel() for p in student.parameters())
    print(f"Student: {n_params / 1e9:.2f}B params (float16), torch.compile=ON")

    # Warmup: compile the graph at max test BS (traces the inductor graph at full size).
    # IMPORTANT: compile at max BS, NOT a small BS like 4. torch.compile generates
    # different inductor graphs (different intermediate buffer sizes) depending on the
    # first input shape. Compiling at BS=4 then measuring BS=224 underestimates VRAM
    # because the inductor reuses the BS=4 graph. m05 compiles fresh at the target BS,
    # so the profiler must match that by compiling at the largest BS it will recommend.
    warmup_bs = min(max_bs, real_batch.shape[0])
    print(f"Warming up torch.compile with real data (BS={warmup_bs}, first batch = compilation)...")
    warmup_batch = real_batch[:warmup_bs].to(device=device, dtype=torch.float16)
    if warmup_batch.ndim == 5 and warmup_batch.shape[2] in (1, 3):
        warmup_batch = warmup_batch.permute(0, 2, 1, 3, 4)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            _ = compiled(warmup_batch)
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        # If max BS OOMs during warmup, retry with half
        torch.cuda.empty_cache()
        warmup_bs = warmup_bs // 2
        print(f"  OOM at BS={max_bs}, retrying warmup at BS={warmup_bs}")
        warmup_batch = real_batch[:warmup_bs].to(device=device, dtype=torch.float16)
        if warmup_batch.ndim == 5 and warmup_batch.shape[2] in (1, 3):
            warmup_batch = warmup_batch.permute(0, 2, 1, 3, 4)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            _ = compiled(warmup_batch)
        torch.cuda.synchronize()
    del warmup_batch
    torch.cuda.empty_cache()
    print("Warmup done — inductor graph cached, now profiling real throughput\n")

    INFER_BATCH_SIZES = [
        8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
        144, 160, 176, 192, 208, 224, 240, 256,
    ]
    print(f"{'Batch':>6}  {'Peak VRAM':>10}  {'Status':>8}  {'Time':>7}")
    print("-" * 50)

    results = {}
    for bs in INFER_BATCH_SIZES:
        if bs > real_batch.shape[0]:
            break
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        gc.collect()

        clip = real_batch[:bs].to(device=device, dtype=torch.float16)
        if clip.ndim == 5 and clip.shape[2] in (1, 3):
            clip = clip.permute(0, 2, 1, 3, 4)
        try:
            # Run N_REPEATS times, take median throughput (eliminates warmup noise)
            times = []
            for _ in range(N_REPEATS):
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.time()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    out = compiled(clip)
                    _ = out.mean(dim=1)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            dt = sorted(times)[len(times) // 2]  # median
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            status = "OK" if peak < gpu_total_gb * 0.85 else "TIGHT"
            throughput = bs / dt
            print(f"{bs:>6}  {peak:>9.1f}G  {status:>8}  {dt:>6.1f}s  {throughput:>5.1f} clip/s")
            results[bs] = {"peak_gb": peak, "time_s": dt, "throughput": throughput}
            del clip, out
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6}  {'OOM':>10}  {'FAILED':>8}")
            torch.cuda.empty_cache()
            gc.collect()
            break

    # Find optimal: BS with best throughput in the stable plateau region.
    # When throughput is flat (GPU compute-saturated), small BS can win by noise.
    # Fix: find peak throughput, then pick the LARGEST BS within 10% of peak.
    # This gives stable throughput + more clips per checkpoint cycle.
    all_tputs = [results[bs]["throughput"] for bs in sorted(results.keys())]
    peak_tput_raw = max(all_tputs)
    threshold_95pct = peak_tput_raw * 0.90
    # Among all BS within 5% of peak throughput, pick the largest
    plateau_bs = [bs for bs in results if results[bs]["throughput"] >= threshold_95pct]
    optimal = max(plateau_bs) if plateau_bs else max(results.keys(), key=lambda bs: results[bs]["throughput"])
    peak_tput = results[optimal]["throughput"]

    print("\n=== INFERENCE PROFILING RESULTS ===")
    print(f"  Optimal BS (max throughput): {optimal}")
    print(f"  Peak VRAM at BS={optimal}: {results[optimal]['peak_gb']:.1f}G / {gpu_total_gb:.0f}G")
    print(f"  Throughput at BS={optimal}: {peak_tput:.1f} clips/s")

    # Save to own directory
    out_dir = REPO_ROOT / "outputs" / "profile" / "inference" / "vjepa2"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / "profile_data.json"

    data = {
        "gpu": gpu_name, "gpu_total_gb": gpu_total_gb,
        "optimal_bs": optimal,
        "optimal_throughput": peak_tput,
        "selection_method": "plateau_within_10pct",
        "results": {str(k): v for k, v in results.items()},
    }
    with open(profile_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {profile_path}")

    # Plot: BS vs VRAM + Throughput (publication quality, no overlapping text)
    init_style()
    batch_sizes = sorted(results.keys())
    vram = [results[bs]["peak_gb"] for bs in batch_sizes]
    tput = [results[bs]["throughput"] for bs in batch_sizes]

    # Show every Nth label to avoid x-axis crowding
    n_labels = min(15, len(batch_sizes))
    label_step = max(1, len(batch_sizes) // n_labels)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Left y-axis: Peak VRAM (bars)
    ax1.bar(batch_sizes, vram, width=12, color=COLORS["blue"], alpha=0.5, label="Peak VRAM (GB)")
    ax1.axhline(y=gpu_total_gb, color="#757575", linestyle=":", linewidth=1.5,
                label=f"VRAM limit ({gpu_total_gb:.0f} GB)")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Peak VRAM (GB)")
    ax1.set_xticks([batch_sizes[i] for i in range(0, len(batch_sizes), label_step)])
    ax1.set_xticklabels([str(batch_sizes[i]) for i in range(0, len(batch_sizes), label_step)],
                        rotation=45, ha="right", fontsize=10)

    # Right y-axis: Throughput (line)
    ax2 = ax1.twinx()
    ax2.plot(batch_sizes, tput, "s-", color=COLORS["red"], linewidth=3.0, markersize=5,
             label="Throughput (clips/s)", zorder=10)
    ax2.set_ylabel("Throughput (clips/s)")

    # Mark optimal BS with vertical line
    if optimal in batch_sizes:
        ax2.axvline(x=optimal, color=COLORS["green"], linestyle="--", linewidth=2,
                    label=f"Optimal BS={optimal} ({peak_tput:.1f} clips/s)")

    ax1.set_title("Inference Batch Size Profiling — ViT-g 1B")

    # Combined legend — placed outside plot to avoid overlap
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11)

    plt.subplots_adjust(bottom=0.22)
    save_fig(fig, str(out_dir / "plot_inference_profile"))

    del student, compiled
    torch.cuda.empty_cache()
    gc.collect()

    return optimal


def profile_dinov2():
    """Profile DINOv2-giant inference VRAM (single image, not video).
        python -u src/utils/profile_vram.py --dinov2 2>&1 | tee logs/profile_vram_dinov2.log
    """
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    gpu_total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name} ({gpu_total_gb:.1f} GB)")
    print("=== DINOv2-GIANT INFERENCE PROFILING (REAL images) ===")

    from transformers import AutoModel, AutoImageProcessor

    # Load real middle frames from video clips
    max_bs = 256  # enough for profiling — DINOv2 saturates throughput at BS=96
    print(f"Loading {max_bs} real video clips for middle-frame extraction...")
    try:
        real_videos = _load_real_video_batch(max_bs)
        # Extract middle frame from each video: (B, T, C, H, W) → (B, C, H, W)
        mid_idx = real_videos.shape[1] // 2
        real_images_raw = real_videos[:, mid_idx]  # (B, C, H, W)
        print(f"Extracted {real_images_raw.shape[0]} middle frames: {real_images_raw.shape}")
        # Process through DINOv2 processor
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
        processed = processor(images=[img for img in real_images_raw], return_tensors="pt")
        real_images = processed["pixel_values"]  # (B, 3, 224, 224)
        print(f"Processed images: {real_images.shape}")
    except Exception as e:
        print(f"WARN: Could not load real data ({e}), falling back to synthetic")
        real_images = None

    model = AutoModel.from_pretrained(
        "facebook/dinov2-giant", torch_dtype=torch.float16,
        device_map="auto", attn_implementation="flash_attention_2")
    model.eval()
    compiled = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DINOv2-giant: {n_params / 1e9:.2f}B params (float16)")

    # Warmup torch.compile with real data
    warmup = (real_images[:4] if real_images is not None else torch.randn(4, 3, 224, 224)).to(device=device, dtype=torch.float16)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        _ = compiled(warmup)
    torch.cuda.synchronize()
    del warmup
    torch.cuda.empty_cache()
    print("torch.compile warmup done")

    DINOV2_BATCH_SIZES = [
        16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256,
    ]
    print(f"\n{'Batch':>6}  {'Peak VRAM':>10}  {'Status':>8}  {'Time':>7}")
    print("-" * 50)

    results = {}
    for bs in DINOV2_BATCH_SIZES:
        if real_images is not None and bs > real_images.shape[0]:
            break
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        gc.collect()

        if real_images is not None:
            pixel_values = real_images[:bs].to(device=device, dtype=torch.float16)
        else:
            pixel_values = torch.randn(bs, 3, 224, 224, device=device, dtype=torch.float16)
        try:
            # Run N_REPEATS times, take median throughput
            times = []
            for _ in range(N_REPEATS):
                torch.cuda.reset_peak_memory_stats(device)
                t0 = time.time()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    out = compiled(pixel_values)
                    _ = out.last_hidden_state[:, 0]
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            dt = sorted(times)[len(times) // 2]  # median
            peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            throughput = bs / dt
            status = "OK" if peak < gpu_total_gb * 0.85 else "TIGHT"
            print(f"{bs:>6}  {peak:>9.1f}G  {status:>8}  {dt:>6.1f}s  {throughput:>6.1f} img/s")
            results[bs] = {"peak_gb": peak, "time_s": dt, "throughput": throughput}
            del pixel_values, out
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>6}  {'OOM':>10}  {'FAILED':>8}")
            torch.cuda.empty_cache()
            gc.collect()
            break

    # Same plateau logic: largest BS within 10% of peak throughput
    all_tputs_d = [results[bs]["throughput"] for bs in sorted(results.keys())]
    peak_tput_d = max(all_tputs_d)
    thresh_d = peak_tput_d * 0.90
    plateau_d = [bs for bs in results if results[bs]["throughput"] >= thresh_d]
    optimal = max(plateau_d) if plateau_d else max(results.keys(), key=lambda bs: results[bs]["throughput"])
    peak_tput = results[optimal]["throughput"]
    print("\n=== DINOv2 INFERENCE RESULTS ===")
    print(f"  Optimal BS (max throughput): {optimal}")
    print(f"  Peak VRAM at BS={optimal}: {results[optimal]['peak_gb']:.1f}G / {gpu_total_gb:.0f}G")
    print(f"  Throughput at BS={optimal}: {peak_tput:.1f} img/s")

    # Save to own directory
    out_dir = REPO_ROOT / "outputs" / "profile" / "inference" / "dinov2"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / "profile_data.json"
    data = {
        "gpu": gpu_name, "gpu_total_gb": gpu_total_gb,
        "optimal_bs": optimal, "optimal_throughput": peak_tput,
        "selection_method": "plateau_within_10pct",
        "results": {str(k): v for k, v in results.items()},
    }
    with open(profile_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {profile_path}")

    # Plot
    init_style()
    batch_sizes = sorted(results.keys())
    vram = [results[bs]["peak_gb"] for bs in batch_sizes]
    tput = [results[bs]["throughput"] for bs in batch_sizes]
    label_step = max(1, len(batch_sizes) // 12)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.bar(batch_sizes, vram, width=max(1, (batch_sizes[-1] - batch_sizes[0]) // len(batch_sizes)) * 0.6,
            color=COLORS["blue"], alpha=0.5, label="Peak VRAM (GB)")
    ax1.axhline(y=gpu_total_gb, color="#757575", linestyle=":", linewidth=1.5,
                label=f"VRAM limit ({gpu_total_gb:.0f} GB)")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Peak VRAM (GB)")
    ax1.set_xticks([batch_sizes[i] for i in range(0, len(batch_sizes), label_step)])
    ax1.set_xticklabels([str(batch_sizes[i]) for i in range(0, len(batch_sizes), label_step)],
                        rotation=45, ha="right", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(batch_sizes, tput, "s-", color=COLORS["red"], linewidth=3.0, markersize=5,
             label="Throughput (img/s)", zorder=10)
    ax2.set_ylabel("Throughput (img/s)")
    ax2.axvline(x=optimal, color=COLORS["green"], linestyle="--", linewidth=2,
                label=f"Optimal BS={optimal} ({peak_tput:.1f} img/s)")

    ax1.set_title("DINOv2-Giant Inference Profiling")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=11)
    plt.subplots_adjust(bottom=0.25)
    save_fig(fig, str(out_dir / "plot_dinov2_profile"))

    del model, compiled
    torch.cuda.empty_cache()
    gc.collect()
    return optimal


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("--training", "--inference", "--dinov2"):
        print("Usage:")
        print("  python -u src/utils/profile_vram.py --training 2>&1 | tee logs/profile_vram_training.log")
        print("  python -u src/utils/profile_vram.py --inference 2>&1 | tee logs/profile_vram_inference.log")
        print("  python -u src/utils/profile_vram.py --dinov2 2>&1 | tee logs/profile_vram_dinov2.log")
        sys.exit(1)
    elif sys.argv[1] == "--training":
        main()
    elif sys.argv[1] == "--inference":
        profile_inference()
    elif sys.argv[1] == "--dinov2":
        profile_dinov2()
