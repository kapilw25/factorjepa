"""
Build FAISS index and compute evaluation metrics (Self-Consistency, Cluster Purity).
Uses CPU FAISS for small datasets (<1000 vectors). Requires embeddings.npy and tags.json.

USAGE:
    python -u src/m05_faiss_metrics.py 2>&1 | tee logs/m05_faiss_metrics.log
    python -u src/m05_faiss_metrics.py --cpu 2>&1 | tee logs/m05_faiss_metrics_cpu.log
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    EMBEDDINGS_FILE, TAGS_FILE, METRICS_FILE, OUTPUTS_DIR,
    FAISS_K_NEIGHBORS, VJEPA_EMBEDDING_DIM,
    check_gpu, load_embeddings_and_tags, check_output_exists
)

try:
    import torch
    import faiss
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install torch faiss-gpu")
    sys.exit(1)


def build_faiss_index_cpu(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index on CPU (fast enough for <10K vectors).

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)

    Returns:
        FAISS CPU index
    """
    d = embeddings.shape[1]
    print(f"Building FAISS CPU index: {embeddings.shape[0]} vectors, dim={d}")

    # For small datasets (<1000), use exact search (IndexFlatL2)
    if embeddings.shape[0] < 1000:
        index = faiss.IndexFlatL2(d)
    else:
        # IVF index for larger datasets
        nlist = min(100, embeddings.shape[0] // 10)
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(embeddings)

    index.add(embeddings)
    print(f"FAISS CPU index built with {index.ntotal} vectors")
    return index


def build_faiss_index_gpu(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index on GPU.

    Args:
        embeddings: numpy array of shape (n_samples, embedding_dim)

    Returns:
        FAISS GPU index
    """
    d = embeddings.shape[1]
    print(f"Building FAISS GPU index: {embeddings.shape[0]} vectors, dim={d}")

    # Check if FAISS GPU is available
    if faiss.get_num_gpus() == 0:
        print("WARNING: No FAISS GPU available, falling back to CPU")
        return build_faiss_index_cpu(embeddings)

    # Create GPU resources
    res = faiss.StandardGpuResources()

    # For small datasets (<1000), use exact search (IndexFlatL2)
    if embeddings.shape[0] < 1000:
        index_cpu = faiss.IndexFlatL2(d)
    else:
        # IVF index for larger datasets
        nlist = min(100, embeddings.shape[0] // 10)
        quantizer = faiss.IndexFlatL2(d)
        index_cpu = faiss.IndexIVFFlat(quantizer, d, nlist)
        index_cpu.train(embeddings)

    # Move to GPU
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    index_gpu.add(embeddings)

    print(f"FAISS GPU index built with {index_gpu.ntotal} vectors")
    return index_gpu


def compute_self_consistency(indices: np.ndarray, k: int = 5) -> float:
    """
    Compute self-consistency metric.
    If kNN(A)=B, does kNN(B) include A?

    Args:
        indices: kNN indices array of shape (n_samples, k+1), includes self at index 0
        k: Number of neighbors to consider (excluding self)

    Returns:
        Self-consistency percentage (0-100)
    """
    n = indices.shape[0]
    consistent = 0

    for i in range(n):
        nearest = indices[i, 1]  # index 0 is self
        nearest_neighbors = indices[nearest, 1:k+1]
        if i in nearest_neighbors:
            consistent += 1

    return (consistent / n) * 100


def compute_cluster_purity(indices: np.ndarray, tags: list, k: int = 5) -> float:
    """
    Compute cluster purity metric.
    What % of kNN neighbors have the same scene_type?

    Args:
        indices: kNN indices array of shape (n_samples, k+1)
        tags: List of tag dictionaries with 'scene_type' field
        k: Number of neighbors to consider

    Returns:
        Cluster purity percentage (0-100)
    """
    correct = 0
    total = 0

    for i, neighbors in enumerate(indices):
        my_type = tags[i].get("scene_type", "unknown")
        for j in neighbors[1:k+1]:
            neighbor_type = tags[j].get("scene_type", "unknown")
            if neighbor_type == my_type:
                correct += 1
            total += 1

    return (correct / total) * 100 if total > 0 else 0


def compute_per_scene_purity(indices: np.ndarray, tags: list, k: int = 5) -> dict:
    """Compute cluster purity for each scene type."""
    scene_stats = {}

    for i, neighbors in enumerate(indices):
        my_type = tags[i].get("scene_type", "unknown")
        if my_type not in scene_stats:
            scene_stats[my_type] = {"correct": 0, "total": 0}

        for j in neighbors[1:k+1]:
            neighbor_type = tags[j].get("scene_type", "unknown")
            if neighbor_type == my_type:
                scene_stats[my_type]["correct"] += 1
            scene_stats[my_type]["total"] += 1

    # Compute percentages
    result = {}
    for scene, stats in scene_stats.items():
        if stats["total"] > 0:
            result[scene] = {
                "purity": round(stats["correct"] / stats["total"] * 100, 2),
                "count": stats["total"] // k  # number of clips
            }
    return result


def generate_plots(distances: np.ndarray, indices: np.ndarray, tags: list, k: int, output_dir: Path,
                   self_consistency: float = None, cluster_purity: float = None):
    """Generate diagnostic plots for FAISS metrics."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    total_clips = len(tags)

    # Plot 1: Per-scene purity breakdown
    per_scene = compute_per_scene_purity(indices, tags, k=k-1)

    fig, ax = plt.subplots(figsize=(10, 6))
    scenes = sorted(per_scene.keys(), key=lambda x: per_scene[x]["purity"], reverse=True)
    purities = [per_scene[s]["purity"] for s in scenes]
    counts = [per_scene[s]["count"] for s in scenes]

    bars = ax.bar(scenes, purities, color=['#4CAF50' if p > 50 else '#F44336' for p in purities])

    # Add count labels above bars - all black and bold
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

    ax.axhline(y=50, color='orange', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Scene Type', fontsize=11)
    ax.set_ylabel('Cluster Purity (%)', fontsize=11)
    ax.set_title(f'V-JEPA kNN Retrieval Purity by Scene Type (k={k-1}, n={total_clips})', fontsize=12)

    # Add overall metrics text box (top-right, bars decrease left-to-right)
    if self_consistency is not None and cluster_purity is not None:
        sc_status = "PASS" if self_consistency > 60 else "FAIL"
        cp_status = "PASS" if cluster_purity > 50 else "FAIL"
        overall = "PASS" if (self_consistency > 60 and cluster_purity > 50) else "FAIL"
        metrics_text = (
            f"Self-Consistency: {self_consistency:.1f}% {sc_status}\n"
            f"Cluster Purity:   {cluster_purity:.1f}% {cp_status}\n"
            f"{'─'*28}\n"
            f"Overall: {overall}"
        )
        bbox_color = '#d4edda' if overall == "PASS" else '#f8d7da'
        ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', fontfamily='monospace',
                color='black', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=bbox_color, edgecolor='gray', alpha=0.95))

    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 50, 60, 80, 100])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plot_path_png = output_dir / "m05_purity_by_scene.png"
    plot_path_pdf = output_dir / "m05_purity_by_scene.pdf"
    plt.savefig(plot_path_png, dpi=300)
    plt.savefig(plot_path_pdf)
    plt.close()
    print(f"Saved: {plot_path_png}")

    # Plot 2: kNN distance distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    # distances[:, 0] is self (always 0), use [:, 1:] for actual neighbors
    neighbor_distances = distances[:, 1:].flatten()

    ax.hist(neighbor_distances, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(x=np.median(neighbor_distances), color='red', linestyle='--',
               label=f'Median: {np.median(neighbor_distances):.2f}')
    ax.set_xlabel('L2 Distance to Neighbor')
    ax.set_ylabel('Count')
    ax.set_title(f'kNN Distance Distribution (k={k-1})')
    ax.legend()
    plt.tight_layout()

    plot_path_png = output_dir / "m05_distance_hist.png"
    plot_path_pdf = output_dir / "m05_distance_hist.pdf"
    plt.savefig(plot_path_png, dpi=150)
    plt.savefig(plot_path_pdf)
    plt.close()
    print(f"Saved: {plot_path_png}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index and compute metrics")
    parser.add_argument("--k", type=int, default=FAISS_K_NEIGHBORS, help="Number of neighbors")
    parser.add_argument("--cpu", action="store_true", help="Use CPU FAISS (faster for small datasets, avoids GPU issues)")
    args = parser.parse_args()

    # Check if metrics already exist
    output_files = [
        METRICS_FILE,
        OUTPUTS_DIR / "m05_purity_by_scene.png",
        OUTPUTS_DIR / "m05_distance_hist.png"
    ]
    existing = [f for f in output_files if f.exists()]
    if existing:
        if not check_output_exists(existing, "metrics/plots"):
            print("Using cached metrics.")
            return

    # GPU check - exit if no CUDA (unless --cpu)
    if not args.cpu:
        check_gpu()

    # Load embeddings and tags
    embeddings, tags = load_embeddings_and_tags()

    # Build FAISS index (CPU or GPU)
    k = args.k
    if args.cpu or embeddings.shape[0] < 1000:
        # CPU is faster for small datasets, avoids faiss-gpu compatibility issues
        if embeddings.shape[0] < 1000:
            print(f"Small dataset ({embeddings.shape[0]} vectors) - using CPU FAISS")
        index = build_faiss_index_cpu(embeddings)
    else:
        index = build_faiss_index_gpu(embeddings)

    # Search for k nearest neighbors
    print(f"Searching for k={k} nearest neighbors on GPU...")
    D, I = index.search(embeddings, k + 1)

    # Compute metrics
    print("\n=== METRICS ===")

    self_consistency = compute_self_consistency(I, k=k-1)
    print(f"Self-Consistency: {self_consistency:.2f}%")

    cluster_purity = compute_cluster_purity(I, tags, k=k-1)
    print(f"Cluster Purity:   {cluster_purity:.2f}%")

    # Save metrics
    metrics = {
        "self_consistency": round(self_consistency, 2),
        "cluster_purity": round(cluster_purity, 2),
        "k_neighbors": k,
        "num_clips": len(tags),
        "embedding_dim": embeddings.shape[1],
        "thresholds": {
            "self_consistency_target": 60,
            "cluster_purity_target": 50
        },
        "pass": self_consistency > 60 and cluster_purity > 50
    }

    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics: {METRICS_FILE}")

    # Generate diagnostic plots
    print("\n=== GENERATING PLOTS ===")
    generate_plots(D, I, tags, k, OUTPUTS_DIR, self_consistency, cluster_purity)

    # Summary
    print("\n=== POC STATUS ===")
    print(f"Self-Consistency: {self_consistency:.2f}% (target > 60%) {'PASS' if self_consistency > 60 else 'FAIL'}")
    print(f"Cluster Purity:   {cluster_purity:.2f}% (target > 50%) {'PASS' if cluster_purity > 50 else 'FAIL'}")
    print(f"Overall: {'PASS' if metrics['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
