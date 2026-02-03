"""
Build FAISS index and compute evaluation metrics (Self-Consistency, Cluster Purity).
Nvidia GPU required (faiss-gpu).

USAGE:
    python -u src/m05_faiss_metrics.py --SANITY 2>&1 | tee logs/m05_faiss_metrics_sanity.log
    python -u src/m05_faiss_metrics.py --FULL 2>&1 | tee logs/m05_faiss_metrics_full.log
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    EMBEDDINGS_FILE, TAGS_FILE, METRICS_FILE,
    FAISS_K_NEIGHBORS, VJEPA_EMBEDDING_DIM
)

try:
    import torch
    import faiss
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install torch faiss-gpu")
    sys.exit(1)


def check_gpu():
    """Check if CUDA GPU is available. Exit if not."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU not available.")
        print("This script requires Nvidia GPU. No CPU fallback.")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")


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


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index and compute metrics")
    parser.add_argument("--SANITY", action="store_true", help="Quick test with existing data")
    parser.add_argument("--FULL", action="store_true", help="Full computation")
    parser.add_argument("--k", type=int, default=FAISS_K_NEIGHBORS, help="Number of neighbors")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    # GPU check - exit if no CUDA
    check_gpu()

    # Load embeddings
    if not EMBEDDINGS_FILE.exists():
        print(f"ERROR: Embeddings not found: {EMBEDDINGS_FILE}")
        print("Run m03_vjepa_embed.py first")
        sys.exit(1)

    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load tags
    if not TAGS_FILE.exists():
        print(f"ERROR: Tags not found: {TAGS_FILE}")
        print("Run m04_qwen_tag.py first")
        sys.exit(1)

    with open(TAGS_FILE, 'r') as f:
        tags = json.load(f)
    print(f"Loaded tags: {len(tags)} clips")

    # Verify alignment
    if len(tags) != embeddings.shape[0]:
        print(f"WARNING: Mismatch - {embeddings.shape[0]} embeddings vs {len(tags)} tags")
        min_len = min(len(tags), embeddings.shape[0])
        embeddings = embeddings[:min_len]
        tags = tags[:min_len]

    # Build FAISS GPU index
    k = args.k
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

    # Summary
    print("\n=== POC STATUS ===")
    print(f"Self-Consistency: {self_consistency:.2f}% (target > 60%) {'PASS' if self_consistency > 60 else 'FAIL'}")
    print(f"Cluster Purity:   {cluster_purity:.2f}% (target > 50%) {'PASS' if cluster_purity > 50 else 'FAIL'}")
    print(f"Overall: {'PASS' if metrics['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
