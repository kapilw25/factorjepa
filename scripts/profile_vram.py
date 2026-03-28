"""
VRAM profiler for V-JEPA 2 ViT-g continual pretraining (Ch10 Q6).

Builds real student + teacher + predictor, runs forward+backward with
synthetic data at increasing batch sizes, reports peak VRAM.
Generates 5 diagnostic plots in outputs/profile/.

USAGE (on GPU machine only):
    python scripts/profile_vram.py 2>&1 | tee logs/profile_vram.log

Prerequisites:
    - deps/vjepa2 cloned
    - CUDA GPU available
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Add vjepa2 to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "deps" / "vjepa2"))

from src.models.vision_transformer import vit_giant_xformers
from src.models.predictor import vit_predictor
from src.masks.multiseq_multiblock3d import _MaskGenerator
from src.masks.utils import apply_masks

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

BATCH_SIZES = [1, 2, 4, 8, 16, 32]
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


def profile_batch(student, teacher, pred, mask_generators, batch_size, device):
    """Run one forward+backward step, return detailed profiling dict."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    gc.collect()

    waterfall = {}  # phase → VRAM snapshot (GB)

    def snap(label):
        torch.cuda.synchronize()
        waterfall[label] = torch.cuda.memory_allocated(device) / (1024 ** 3)

    snap("baseline")

    # Synthetic video: (B, C, T, H, W)
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

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
    print(f"  [1/5] Saved plot1_batch_scaling.png")

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
        print(f"  [2/5] Saved plot2_ckpt_savings.png")

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
        print(f"  [3/5] Saved plot3_breakdown.png")

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
        print(f"  [4/5] Saved plot4_waterfall.png")

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
        print(f"  [5/5] Saved plot5_masking_effect.png")

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
                t0 = time.time()
                info = profile_batch(student, teacher, pred, mask_generators, bs, device)
                dt = time.time() - t0
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
    out_dir = REPO_ROOT / "outputs" / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON for reproducibility
    raw = {
        "gpu": gpu_name, "gpu_total_gb": gpu_total_gb,
        "config": {"crop": CROP_SIZE, "frames": NUM_FRAMES, "tokens": TOTAL_TOKENS},
        "no_ckpt": {str(k): {"peak_gb": v["peak_gb"], "time_s": v["time_s"],
                              "visible_tokens": v["n_visible_tokens"],
                              "waterfall": v["waterfall"]}
                    for k, v in results_no_ckpt.items()},
        "grad_ckpt": {str(k): {"peak_gb": v["peak_gb"], "time_s": v["time_s"],
                                "visible_tokens": v["n_visible_tokens"],
                                "waterfall": v["waterfall"]}
                      for k, v in results_ckpt.items()},
    }
    with open(out_dir / "profile_data.json", "w") as f:
        json.dump(raw, f, indent=2)

    generate_plots(results_no_ckpt, results_ckpt, gpu_name, gpu_total_gb, n_student, out_dir)


if __name__ == "__main__":
    main()
