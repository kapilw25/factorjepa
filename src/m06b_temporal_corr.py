"""
Temporal correlation analysis: embedding distance vs motion feature distance.
Measures how well each encoder captures temporal/motion structure.
CPU-only — reads pre-computed .npy files from m04d + m05/m05b.

USAGE:
    python -u src/m06b_temporal_corr.py --encoder vjepa --FULL \
        --subset data/subset_10k.json 2>&1 | tee logs/m06b_vjepa.log
    python -u src/m06b_temporal_corr.py --encoder dinov2 --FULL \
        --subset data/subset_10k.json 2>&1 | tee logs/m06b_dinov2.log
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    ENCODER_REGISTRY, FAISS_K_NEIGHBORS,
    add_encoder_arg, add_subset_arg, get_output_dir, get_encoder_files,
)
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, finish_wandb,
)

N_SAMPLE_PAIRS = 100_000
N_MOTION_CLUSTERS = 4


# ── Key alignment ────────────────────────────────────────────────────

def align_by_keys(emb: np.ndarray, emb_keys: np.ndarray,
                  motion: np.ndarray, motion_keys: np.ndarray):
    """Align embeddings and motion features by shared clip keys.

    Returns aligned (embeddings, motion_features, n_aligned).
    """
    emb_key_to_idx = {k: i for i, k in enumerate(emb_keys)}
    shared_indices_emb = []
    shared_indices_mot = []

    for mot_idx, key in enumerate(motion_keys):
        if key in emb_key_to_idx:
            shared_indices_emb.append(emb_key_to_idx[key])
            shared_indices_mot.append(mot_idx)

    if not shared_indices_emb:
        return None, None, 0

    aligned_emb = emb[shared_indices_emb]
    aligned_mot = motion[shared_indices_mot]
    return aligned_emb, aligned_mot, len(shared_indices_emb)


# ── Metrics ──────────────────────────────────────────────────────────

def compute_spearman_correlation(embeddings, motion_features,
                                 n_pairs=N_SAMPLE_PAIRS, seed=42,
                                 n_boot=1000):
    """Spearman rank correlation between embedding and motion distance.

    Returns (rho, pval, n_actual_pairs, ci_dict).
    ci_dict has bootstrap 95% CI: {"mean", "ci_lo", "ci_hi", "ci_half"}.
    """
    N = len(embeddings)
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, N, n_pairs)
    idx_b = rng.integers(0, N, n_pairs)

    # Filter self-pairs
    mask = idx_a != idx_b
    idx_a = idx_a[mask]
    idx_b = idx_b[mask]

    emb_dist = np.linalg.norm(embeddings[idx_a] - embeddings[idx_b], axis=1)
    mot_dist = np.linalg.norm(motion_features[idx_a] - motion_features[idx_b],
                               axis=1)

    rho, pval = stats.spearmanr(emb_dist, mot_dist)

    # Bootstrap 95% CI on rho
    boot_rhos = np.empty(n_boot)
    n_dist = len(emb_dist)
    for i in range(n_boot):
        idx = rng.integers(0, n_dist, n_dist)
        boot_rhos[i], _ = stats.spearmanr(emb_dist[idx], mot_dist[idx])

    ci_lo = float(np.percentile(boot_rhos, 2.5))
    ci_hi = float(np.percentile(boot_rhos, 97.5))
    ci_dict = {
        "mean": round(float(rho), 6),
        "ci_lo": round(ci_lo, 6),
        "ci_hi": round(ci_hi, 6),
        "ci_half": round((ci_hi - ci_lo) / 2, 6),
    }

    return float(rho), float(pval), len(idx_a), ci_dict


def compute_temporal_prec_at_k(knn_indices, motion_features, k=6):
    """% of kNN neighbors in same motion quartile as query.

    Uses mean_magnitude (feature index 0) for quartile binning.
    """
    mean_mag = motion_features[:, 0]
    quartile_edges = np.percentile(mean_mag, [25, 50, 75])
    quartiles = np.digitize(mean_mag, quartile_edges)

    N = min(len(knn_indices), len(motion_features))
    matches = 0
    total = 0
    for i in range(N):
        query_q = quartiles[i]
        neighbors = knn_indices[i, 1:k + 1]  # skip self (col 0)
        valid = neighbors[neighbors < N]
        if len(valid) == 0:
            continue
        neighbor_qs = quartiles[valid]
        matches += int(np.sum(neighbor_qs == query_q))
        total += len(valid)

    return float(matches / total * 100) if total > 0 else 0.0


def compute_motion_retrieval_map(knn_indices, motion_features, k=6,
                                  n_clusters=N_MOTION_CLUSTERS):
    """mAP where 'relevant' = same motion cluster (KMeans on motion features)."""
    from sklearn.cluster import KMeans

    N = min(len(knn_indices), len(motion_features))
    # Cluster motion features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(motion_features[:N])

    ap_scores = []
    for i in range(N):
        query_label = labels[i]
        neighbors = knn_indices[i, 1:k + 1]
        valid = neighbors[neighbors < N]
        if len(valid) == 0:
            continue
        neighbor_labels = labels[valid]
        # Compute AP for this query
        relevant = (neighbor_labels == query_label)
        if not np.any(relevant):
            ap_scores.append(0.0)
            continue
        cumsum = np.cumsum(relevant).astype(np.float64)
        precision_at = cumsum / np.arange(1, len(relevant) + 1)
        ap = np.sum(precision_at * relevant) / np.sum(relevant)
        ap_scores.append(float(ap))

    return float(np.mean(ap_scores)) if ap_scores else 0.0


def compute_temporal_order_sensitivity(emb_normal, emb_shuffled,
                                       keys_normal, keys_shuffled):
    """Per-clip L2 distance between normal and shuffled V-JEPA embeddings.

    If V-JEPA truly encodes temporal order, shuffling frames should produce
    a LARGE embedding change. For image encoders (single frame), this is N/A.
    """
    # Align by shared keys
    normal_map = {k: i for i, k in enumerate(keys_normal)}
    shared_n, shared_s = [], []
    for s_idx, key in enumerate(keys_shuffled):
        if key in normal_map:
            shared_n.append(normal_map[key])
            shared_s.append(s_idx)

    if not shared_n:
        return None, 0

    n_emb = emb_normal[shared_n]
    s_emb = emb_shuffled[shared_s]
    per_clip_dist = np.linalg.norm(n_emb - s_emb, axis=1)

    return {
        "mean": float(np.mean(per_clip_dist)),
        "std": float(np.std(per_clip_dist)),
        "median": float(np.median(per_clip_dist)),
        "n_clips": len(shared_n),
    }, len(shared_n)


def compute_temporal_locality(embeddings, clip_keys, n_sample=50000, seed=42):
    """Ratio of intra-video to inter-video embedding distance.

    Same-video clips should cluster closer if encoder captures temporal continuity.
    Lower ratio = stronger temporal locality.
    """
    # Parse video_id from clip keys: "section/video_id/source_file"
    video_ids = []
    for key in clip_keys:
        parts = str(key).split("/")
        vid = parts[-2] if len(parts) >= 3 else str(key)
        video_ids.append(vid)
    video_ids = np.array(video_ids)

    # Build per-video clip indices
    from collections import defaultdict
    vid_to_indices = defaultdict(list)
    for i, vid in enumerate(video_ids):
        vid_to_indices[vid].append(i)

    # Only videos with 2+ clips can contribute intra-video pairs
    multi_clip_vids = {v: idxs for v, idxs in vid_to_indices.items() if len(idxs) >= 2}
    if not multi_clip_vids:
        return None

    rng = np.random.default_rng(seed)

    # Sample intra-video pairs
    intra_dists = []
    all_multi_indices = []
    for idxs in multi_clip_vids.values():
        all_multi_indices.extend(idxs)
    all_multi_indices = np.array(all_multi_indices)

    for vid, idxs in multi_clip_vids.items():
        if len(idxs) < 2:
            continue
        idxs = np.array(idxs)
        # Sample pairs within this video
        n_pairs_vid = min(len(idxs) * 2, 100)
        a = rng.choice(idxs, n_pairs_vid)
        b = rng.choice(idxs, n_pairs_vid)
        mask = a != b
        if mask.sum() == 0:
            continue
        dists = np.linalg.norm(embeddings[a[mask]] - embeddings[b[mask]], axis=1)
        intra_dists.extend(dists.tolist())

    if not intra_dists:
        return None

    # Sample inter-video pairs (random clips from different videos)
    n_inter = min(n_sample, len(embeddings) * 5)
    idx_a = rng.integers(0, len(embeddings), n_inter)
    idx_b = rng.integers(0, len(embeddings), n_inter)
    mask = (idx_a != idx_b) & (video_ids[idx_a] != video_ids[idx_b])
    inter_dists = np.linalg.norm(embeddings[idx_a[mask]] - embeddings[idx_b[mask]], axis=1)

    intra_mean = float(np.mean(intra_dists))
    inter_mean = float(np.mean(inter_dists))
    ratio = intra_mean / inter_mean if inter_mean > 0 else 0.0

    return {
        "intra_video_dist": round(intra_mean, 4),
        "inter_video_dist": round(inter_mean, 4),
        "ratio": round(ratio, 4),
        "n_intra_pairs": len(intra_dists),
        "n_inter_pairs": int(mask.sum()),
        "n_multi_clip_videos": len(multi_clip_vids),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Temporal correlation: embedding distance vs motion distance (CPU)")
    parser.add_argument("--SANITY", action="store_true",
                        help="Placeholder for consistency")
    parser.add_argument("--FULL", action="store_true",
                        help="Process all available data")
    parser.add_argument("--k", type=int, default=FAISS_K_NEIGHBORS,
                        help=f"kNN neighbors (default: {FAISS_K_NEIGHBORS})")
    parser.add_argument("--n-pairs", type=int, default=N_SAMPLE_PAIRS,
                        help=f"Random pairs for Spearman (default: {N_SAMPLE_PAIRS:,})")
    add_encoder_arg(parser)
    add_subset_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    output_dir = get_output_dir(args.subset, sanity=args.SANITY)
    mode = "SANITY" if args.SANITY else ("POC" if args.subset else "FULL")
    wb_run = init_wandb("m06b", mode, config=vars(args),
                        enabled=not args.no_wandb)

    enc_info = ENCODER_REGISTRY[args.encoder]
    enc_files = get_encoder_files(args.encoder, output_dir)
    sfx = enc_info["suffix"]

    print(f"Encoder: {args.encoder} (dim={enc_info['dim']}, suffix='{sfx}')")
    print(f"Output: {output_dir}")

    # ── Load motion features (optional — metrics 4+5 work without) ──
    motion_file = output_dir / "motion_features.npy"
    motion_keys_file = output_dir / "motion_features.paths.npy"
    motion_features = None
    motion_keys = None

    if motion_file.exists() and motion_keys_file.exists():
        motion_features = np.load(motion_file).astype(np.float32)
        motion_keys = np.load(motion_keys_file, allow_pickle=True)
        print(f"Motion features: {motion_features.shape}")
    else:
        print("  WARN: motion_features.npy not found — skipping motion-based metrics (1-3)")
        print("  Run m04d_motion_features.py first for Spearman/TemporalPrec/MotionMAP")

    # ── Load embeddings ───────────────────────────────────────────
    emb_file = enc_files["embeddings"]
    emb_keys_file = enc_files["paths"]

    for f, desc in [(emb_file, "embeddings"), (emb_keys_file, "embedding keys")]:
        if not f.exists():
            print(f"FATAL: {desc} not found: {f}")
            print(f"  Run m05/m05b with --encoder {args.encoder} first.")
            sys.exit(1)

    embeddings = np.load(emb_file).astype(np.float32)
    emb_keys = np.load(emb_keys_file, allow_pickle=True)
    print(f"Embeddings: {embeddings.shape}")

    # ── Load kNN indices (optional, for temporal_prec_at_k) ───────
    knn_file = enc_files["knn_indices"]
    knn_indices = None
    if knn_file.exists():
        knn_indices = np.load(knn_file)
        print(f"kNN indices: {knn_indices.shape}")
    else:
        print(f"  WARN: {knn_file.name} not found — skipping temporal_prec_at_k")

    t_start = time.time()

    # ── Motion-based metrics (1-3): require m04d ──────────────────
    rho, pval, actual_pairs, rho_ci = None, None, 0, None
    n_aligned = 0
    temporal_prec = None
    motion_map = None

    if motion_features is not None:
        aligned_emb, aligned_mot, n_aligned = align_by_keys(
            embeddings, emb_keys, motion_features, motion_keys)

        if n_aligned == 0:
            print("  WARN: No shared keys between embeddings and motion features")
        else:
            print(f"Aligned: {n_aligned:,} clips "
                  f"(emb={len(emb_keys):,}, motion={len(motion_keys):,})")

            # Normalize motion features for distance computation
            mot_std = aligned_mot.std(axis=0, keepdims=True)
            mot_std[mot_std == 0] = 1.0
            aligned_mot_norm = aligned_mot / mot_std

            # Metric 1: Spearman correlation + bootstrap CI
            print("\nComputing Spearman correlation (+ bootstrap 95% CI)...")
            rho, pval, actual_pairs, rho_ci = compute_spearman_correlation(
                aligned_emb, aligned_mot_norm, n_pairs=args.n_pairs)
            print(f"  Spearman rho = {rho:.4f} ± {rho_ci['ci_half']:.4f} "
                  f"(p = {pval:.2e}, {actual_pairs:,} pairs)")

            # Metrics 2-3: Temporal Prec@K + Motion mAP
            if knn_indices is not None:
                knn_n = min(len(knn_indices), n_aligned)
                print(f"\nComputing Temporal Prec@K (k={args.k})...")
                temporal_prec = compute_temporal_prec_at_k(
                    knn_indices[:knn_n], aligned_mot[:knn_n], k=args.k)
                print(f"  Temporal Prec@K = {temporal_prec:.1f}%")

                print(f"\nComputing Motion Retrieval mAP (k={args.k}, clusters={N_MOTION_CLUSTERS})...")
                motion_map = compute_motion_retrieval_map(
                    knn_indices[:knn_n], aligned_mot_norm[:knn_n], k=args.k)
                print(f"  Motion Retrieval mAP = {motion_map:.4f}")

    # ── Temporal Order Sensitivity (V-JEPA only) ─────────────────
    order_sensitivity = None
    if args.encoder in ("vjepa", "vjepa_shuffled"):
        # Load the counterpart: if running vjepa, load shuffled; if shuffled, load normal
        vjepa_files = get_encoder_files("vjepa", output_dir)
        shuffled_files = get_encoder_files("vjepa_shuffled", output_dir)
        normal_emb_file = vjepa_files["embeddings"]
        shuffled_emb_file = shuffled_files["embeddings"]
        normal_keys_file = vjepa_files["paths"]
        shuffled_keys_file = shuffled_files["paths"]

        if normal_emb_file.exists() and shuffled_emb_file.exists():
            print("\nComputing Temporal Order Sensitivity...")
            n_emb = np.load(normal_emb_file).astype(np.float32)
            s_emb = np.load(shuffled_emb_file).astype(np.float32)
            n_keys = np.load(normal_keys_file, allow_pickle=True)
            s_keys = np.load(shuffled_keys_file, allow_pickle=True)
            order_sensitivity, n_os = compute_temporal_order_sensitivity(
                n_emb, s_emb, n_keys, s_keys)
            if order_sensitivity:
                print(f"  Order sensitivity: mean={order_sensitivity['mean']:.2f}, "
                      f"median={order_sensitivity['median']:.2f} ({n_os:,} clips)")
        else:
            print("  WARN: Need both vjepa + vjepa_shuffled embeddings for order sensitivity")

    # ── Temporal Locality (all encoders) ───────────────────────────
    print("\nComputing Temporal Locality...")
    locality = compute_temporal_locality(embeddings, emb_keys)
    if locality:
        print(f"  Intra-video dist: {locality['intra_video_dist']:.4f}")
        print(f"  Inter-video dist: {locality['inter_video_dist']:.4f}")
        print(f"  Ratio (lower=better): {locality['ratio']:.4f} "
              f"({locality['n_multi_clip_videos']} videos)")

    elapsed = time.time() - t_start

    # ── Save results ──────────────────────────────────────────────
    results = {
        "encoder": args.encoder,
        "encoder_dim": enc_info["dim"],
        "encoder_type": enc_info["type"],
        "n_clips_aligned": n_aligned,
        "k_neighbors": args.k,
        "compute_time_sec": round(elapsed, 1),
        "mode": mode,
    }
    if rho is not None:
        results["spearman_rho"] = round(rho, 6)
        results["spearman_rho_ci"] = rho_ci
        results["spearman_pval"] = pval
        results["n_pairs_sampled"] = actual_pairs
        results["motion_clusters"] = N_MOTION_CLUSTERS
    if temporal_prec is not None:
        results["temporal_prec_at_k"] = round(temporal_prec, 2)
    if motion_map is not None:
        results["motion_retrieval_map"] = round(motion_map, 6)
    if order_sensitivity is not None:
        results["temporal_order_sensitivity"] = order_sensitivity
    if locality is not None:
        results["temporal_locality"] = locality

    results_file = output_dir / f"m06b_temporal_corr{sfx}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_file}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"m06b TEMPORAL CORRELATION — {args.encoder}")
    print(f"{'='*60}")
    if rho is not None:
        print(f"  Spearman rho:          {rho:.4f} ± {rho_ci['ci_half']:.4f}")
    if temporal_prec is not None:
        print(f"  Temporal Prec@K:       {temporal_prec:.1f}%")
    if motion_map is not None:
        print(f"  Motion Retrieval mAP:  {motion_map:.4f}")
    if order_sensitivity is not None:
        print(f"  Order Sensitivity:     {order_sensitivity['mean']:.2f} (mean L2)")
    if locality is not None:
        print(f"  Temporal Locality:     {locality['ratio']:.4f} (intra/inter ratio)")
    print(f"  Clips aligned:         {n_aligned:,}")
    print(f"  Compute time:          {elapsed:.1f}s")
    print(f"{'='*60}")

    log_metrics(wb_run, results)
    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
