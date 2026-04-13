"""Temporal interference projection: PCA on (normal - shuffled), project out, re-run Prec@K. CPU-only.
    python -u src/m06c_temporal_projection.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m06c.log
    python -u src/m06c_temporal_projection.py --SANITY 2>&1 | tee logs/m06c_sanity.log
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import get_output_dir, get_module_output_dir, add_subset_arg
from utils.checkpoint import save_array_checkpoint, save_json_checkpoint
from utils.gpu_batch import cleanup_temp
from utils.output_guard import verify_or_skip
from utils.progress import make_pbar
from utils.wandb_utils import add_wandb_args, init_wandb, log_metrics, finish_wandb


K_SWEEP = [5, 10, 25, 50, 100, 200]


def compute_temporal_subspace(emb_normal: np.ndarray, emb_shuffled: np.ndarray,
                              n_components: int) -> tuple:
    """PCA on (normal - shuffled) difference vectors. Returns (components, explained_variance_ratio)."""
    diffs = emb_normal - emb_shuffled           # (N, D)
    diffs_centered = diffs - diffs.mean(axis=0)  # Center before SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(diffs_centered)
    return svd.components_, svd.explained_variance_ratio_


def project_out(embeddings: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project embeddings orthogonal to subspace defined by components.

    P_perp = I - V @ V.T  (orthogonal complement projector)
    projected = embeddings - embeddings @ V @ V.T
    Output stays at full dim (not reduced).
    """
    V = components.T                                     # (D, k)
    projected = embeddings - (embeddings @ V) @ V.T      # (N, D)
    return projected.astype(np.float32)


def load_and_verify(output_dir: Path) -> tuple:
    """Load vjepa normal + shuffled embeddings, verify alignment."""
    emb_file = output_dir / "embeddings.npy"
    shuf_file = output_dir / "embeddings_vjepa_shuffled.npy"
    paths_file = output_dir / "embeddings.paths.npy"
    paths_shuf = output_dir / "embeddings_vjepa_shuffled.paths.npy"

    for f in [emb_file, shuf_file, paths_file, paths_shuf]:
        if not f.exists():
            print(f"FATAL: {f} not found. Run m05 + m05b first.")
            sys.exit(1)

    emb_normal = np.load(emb_file).astype(np.float32)
    emb_shuffled = np.load(shuf_file).astype(np.float32)
    paths_n = np.load(paths_file, allow_pickle=True)
    paths_s = np.load(paths_shuf, allow_pickle=True)

    if emb_normal.shape != emb_shuffled.shape:
        print(f"FATAL: Shape mismatch: normal {emb_normal.shape} vs shuffled {emb_shuffled.shape}")
        sys.exit(1)

    if not np.array_equal(paths_n, paths_s):
        print("FATAL: Clip key mismatch between normal and shuffled embeddings")
        sys.exit(1)

    print(f"Loaded embeddings: {emb_normal.shape[0]} clips, {emb_normal.shape[1]}-dim")
    print("Clip keys aligned: OK")
    return emb_normal, emb_shuffled, paths_n


def run_m06_for_encoder(encoder_name: str, mode_flag: str, subset_arg: list):
    """Run m06_faiss_metrics.py for a projected encoder via subprocess."""
    cmd = [
        sys.executable, "-u", "src/m06_faiss_metrics.py",
        "--encoder", encoder_name,
        mode_flag,
        "--no-wandb",
    ] + subset_arg
    print(f"  Running m06 --encoder {encoder_name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: m06 failed for {encoder_name}")
        print(f"  stderr: {result.stderr[-500:]}")
        return False
    return True


def load_prec_at_k(metrics_file: Path) -> dict:
    """Load Prec@K from m06 metrics JSON."""
    if not metrics_file.exists():
        return {}
    with open(metrics_file) as f:
        data = json.load(f)
    result = {}
    for mode in ["easy", "hard"]:
        if mode in data:
            result[f"prec_at_k_{mode}"] = data[mode].get("prec_at_k", 0)
            ci = data[mode].get("ci", {}).get("prec_at_k", {})
            result[f"prec_at_k_{mode}_ci"] = ci.get("ci_half", 0)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Temporal interference projection: PCA on (normal - shuffled), "
                    "project out top-k components, re-run Prec@K")
    parser.add_argument("--SANITY", action="store_true", help="Run on sanity embeddings")
    parser.add_argument("--POC", action="store_true", help="Run on POC embeddings")
    parser.add_argument("--FULL", action="store_true", help="Run on full embeddings")
    parser.add_argument("--k-sweep", type=str, default=None,
                        help=f"Comma-separated n_components values (default: {K_SWEEP})")
    add_subset_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    mode_flag = f"--{mode}"
    subset_arg = ["--subset", args.subset] if args.subset else []

    input_dir = get_output_dir(args.subset, sanity=args.SANITY, poc=args.POC)
    output_dir = get_module_output_dir("m06c_temporal_projection", args.subset, sanity=args.SANITY, poc=args.POC)
    wb_run = init_wandb("m06c", mode, config=vars(args), enabled=not args.no_wandb)

    # Clean stale temp files (consistent with GPU scripts)
    cleanup_temp()

    k_values = K_SWEEP
    if args.k_sweep:
        k_values = [int(x.strip()) for x in args.k_sweep.split(",")]

    # Output-exists guard — skip full run if summary already exists
    summary_file = output_dir / "m06c_projection_results.json"
    if verify_or_skip(output_dir, {
        "projection summary": summary_file,
    }, label="m06c temporal_projection"):
        return

    print(f"\n{'='*60}")
    print(f"Temporal Interference Projection — {mode} mode")
    print(f"Output dir: {output_dir}")
    print(f"k sweep: {k_values}")
    print(f"{'='*60}\n")

    # ── Load embeddings ─────────────────────────────────────────
    t0 = time.time()
    emb_normal, emb_shuffled, _paths = load_and_verify(input_dir)
    N, D = emb_normal.shape

    # ── Compute difference stats ────────────────────────────────
    diffs = emb_normal - emb_shuffled
    diff_norm = np.linalg.norm(diffs, axis=1)
    print(f"Difference norms: mean={diff_norm.mean():.4f}, std={diff_norm.std():.4f}, "
          f"max={diff_norm.max():.4f}")
    print(f"Relative to embedding norm: {diff_norm.mean() / np.linalg.norm(emb_normal, axis=1).mean():.2%}")

    # ── Sweep n_components ──────────────────────────────────────
    max_k = max(k_values)
    if max_k > min(N, D):
        max_k = min(N, D) - 1
        k_values = [k for k in k_values if k <= max_k]
        print(f"Capped k_values to {k_values} (N={N}, D={D})")

    # Compute SVD once at max_k, slice for smaller k values
    print(f"\nComputing SVD with {max_k} components ...")
    components_full, explained_full = compute_temporal_subspace(emb_normal, emb_shuffled, max_k)
    cumulative_explained = np.cumsum(explained_full)
    print(f"Explained variance: top-5={cumulative_explained[4]:.2%}, "
          f"top-50={cumulative_explained[min(49, max_k-1)]:.2%}, "
          f"top-{max_k}={cumulative_explained[-1]:.2%}")

    # Precompute vectorized stats for original (used in every k iteration)
    emb_norm_vec = np.linalg.norm(emb_normal, axis=1)
    shuf_norm_vec = np.linalg.norm(emb_shuffled, axis=1)
    cos_before_vec = np.sum(emb_normal * emb_shuffled, axis=1) / (emb_norm_vec * shuf_norm_vec + 1e-8)
    orig_embedding_norm = float(emb_norm_vec.mean())
    cos_before_mean = float(cos_before_vec.mean())

    results = []
    pbar = make_pbar(total=len(k_values), desc="m06c_projection", unit="k")

    for k in k_values:
        suffix = f"_temporal_proj_k{k}"
        proj_emb_file = output_dir / f"embeddings{suffix}.npy"
        proj_paths_file = output_dir / f"embeddings{suffix}.paths.npy"

        # Resume: load cached projection if shape matches (corrupt file -> recompute)
        projected = None
        if proj_emb_file.exists():
            try:
                cached = np.load(proj_emb_file)
            except Exception as e:
                print(f"  k={k:3d}: cached file corrupt ({e}), recomputing")
                cached = None
            if cached is not None:
                if cached.shape == emb_normal.shape:
                    projected = cached
                else:
                    print(f"  k={k:3d}: cached shape {cached.shape} != expected {emb_normal.shape}, recomputing")

        if projected is None:
            components_k = components_full[:k]                 # (k, D)
            projected = project_out(emb_normal, components_k)  # (N, D)
            save_array_checkpoint(projected, proj_emb_file)    # atomic
            if not proj_paths_file.exists():
                shutil.copy(input_dir / "embeddings.paths.npy", proj_paths_file)
            assert projected.shape == emb_normal.shape, \
                f"Shape changed: {projected.shape} vs {emb_normal.shape}"
            action = "saved "
        else:
            action = "cached"

        # Vectorized cosine after projection
        proj_norm_vec = np.linalg.norm(projected, axis=1)
        cos_after_vec = np.sum(projected * emb_shuffled, axis=1) / (proj_norm_vec * shuf_norm_vec + 1e-8)

        results.append({
            "k": k,
            "explained_variance_cumulative": float(cumulative_explained[k - 1]),
            "cosine_normal_shuffled_before": cos_before_mean,
            "cosine_normal_shuffled_after": float(cos_after_vec.mean()),
            "embedding_norm_before": orig_embedding_norm,
            "embedding_norm_after": float(proj_norm_vec.mean()),
        })

        print(f"  k={k:3d}: explained={cumulative_explained[k-1]:.2%}, "
              f"cos(n,s) {cos_before_mean:.4f} -> {cos_after_vec.mean():.4f}, "
              f"{action} {proj_emb_file.name}")
        pbar.update(1)

    pbar.close()

    # ── Run m06 on each projected encoder ────��──────────────────
    print(f"\n{'='*60}")
    print("Running m06_faiss_metrics on projected embeddings ...")
    print(f"{'='*60}\n")

    for entry in results:
        k = entry["k"]
        encoder_name = f"temporal_proj_k{k}"
        success = run_m06_for_encoder(encoder_name, mode_flag, subset_arg)
        if success:
            m06_dir = get_module_output_dir("m06_faiss_metrics", args.subset, sanity=args.SANITY, poc=args.POC)
            metrics_file = m06_dir / f"m06_metrics_{encoder_name}.json"
            prec = load_prec_at_k(metrics_file)
            entry.update(prec)

    # ── Load original vjepa Prec@K for comparison ───────────────
    m06_dir = get_module_output_dir("m06_faiss_metrics", args.subset, sanity=args.SANITY, poc=args.POC)
    orig_metrics = load_prec_at_k(m06_dir / "m06_metrics.json")

    # ── Print comparison table ──────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"TEMPORAL PROJECTION RESULTS ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"\nOriginal V-JEPA Prec@K(Easy): {orig_metrics.get('prec_at_k_easy', '?'):.2f}%"
          f" +/- {orig_metrics.get('prec_at_k_easy_ci', '?'):.2f}")
    print(f"Original V-JEPA Prec@K(Hard): {orig_metrics.get('prec_at_k_hard', '?'):.2f}%"
          f" +/- {orig_metrics.get('prec_at_k_hard_ci', '?'):.2f}")
    print()
    print(f"{'k':>5} | {'Explained':>10} | {'Prec@K Easy':>12} | {'Prec@K Hard':>12} | {'Delta Easy':>11}")
    print(f"{'-'*5}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*11}")

    for entry in results:
        k = entry["k"]
        expl = entry["explained_variance_cumulative"]
        prec_e = entry.get("prec_at_k_easy", 0)
        prec_h = entry.get("prec_at_k_hard", 0)
        ci_e = entry.get("prec_at_k_easy_ci", 0)
        delta_e = prec_e - orig_metrics.get("prec_at_k_easy", 0)
        sign = "+" if delta_e >= 0 else ""
        print(f"{k:5d} | {expl:9.2%} | {prec_e:6.2f}+/-{ci_e:.2f} | {prec_h:10.2f}% | {sign}{delta_e:9.2f}pp")

    # ── Save summary JSON ───────────────────────────────────────
    summary = {
        "original_metrics": orig_metrics,
        "projection_results": results,
        "n_clips": N,
        "embedding_dim": D,
        "k_sweep": k_values,
        "elapsed_seconds": elapsed,
    }
    summary_file = output_dir / "m06c_projection_results.json"
    save_json_checkpoint(summary, summary_file)  # atomic + fsync
    print(f"\nSaved: {summary_file}")

    # ── Wandb logging ────────────────────────────────────────��──
    if wb_run:
        for entry in results:
            log_metrics(wb_run, {
                "k": entry["k"],
                "explained_variance": entry["explained_variance_cumulative"],
                "prec_at_k_easy": entry.get("prec_at_k_easy", 0),
                "prec_at_k_hard": entry.get("prec_at_k_hard", 0),
            }, step=entry["k"])
    finish_wandb(wb_run)

    # ── Final verdict ───────────────────────────────────────────
    best = max(results, key=lambda x: x.get("prec_at_k_easy", 0))
    best_delta = best.get("prec_at_k_easy", 0) - orig_metrics.get("prec_at_k_easy", 0)
    print(f"\nBest k={best['k']}: Prec@K(Easy) = {best.get('prec_at_k_easy', 0):.2f}% "
          f"(delta = {'+' if best_delta >= 0 else ''}{best_delta:.2f}pp)")
    if best_delta > 5:
        print("RESULT: Temporal projection RECOVERS significant Prec@K — paper centerpiece!")
    elif best_delta > 1:
        print("RESULT: Modest improvement — temporal interference partially linear")
    else:
        print("RESULT: No improvement — temporal interference is NOT a removable linear subspace")


if __name__ == "__main__":
    main()
