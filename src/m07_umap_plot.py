"""
Generate UMAP visualization and kNN confusion matrix for V-JEPA embeddings.
CPU-based visualization script (M1 compatible). Requires embeddings.npy and tags.json.

USAGE:
    python -u src/m07_umap_plot.py 2>&1 | tee logs/m07_umap_plot.log
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from collections import defaultdict
from utils.config import (
    EMBEDDINGS_FILE, TAGS_FILE, METRICS_FILE, OUTPUTS_DIR,
    UMAP_PLOT_PNG, UMAP_PLOT_PDF, FAISS_K_NEIGHBORS, load_embeddings_and_tags,
    check_output_exists
)

try:
    import umap
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install umap-learn matplotlib")
    sys.exit(1)


# Color palette for scene types (matches tag_taxonomy.json scene_type values)
SCENE_COLORS = {
    "market": "#e41a1c",
    "junction": "#4daf4a",
    "residential_lane": "#984ea3",
    "promenade": "#377eb8",
    "transit": "#999999",
    "temple_tourist": "#ff7f00",
    "highway": "#ffff33",
    "alley": "#a65628",
    "commercial": "#17becf",
    "construction": "#f781bf",
    "unknown": "#666666",
}


def create_confusion_matrix(
    embeddings: np.ndarray,
    tags: list,
    output_dir: Path,
    k: int = 5
):
    """
    Create kNN retrieval confusion matrix showing per-class accuracy.
    """
    try:
        import faiss
    except ImportError:
        print("WARNING: faiss not available, skipping confusion matrix")
        return

    print(f"Building FAISS index for confusion matrix (k={k})...")

    # Build CPU FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    D, I = index.search(embeddings, k + 1)  # +1 for self

    # Get unique scene types
    scene_types = sorted(set(t.get("scene_type", "unknown") for t in tags))

    # Build confusion counts
    conf = defaultdict(lambda: defaultdict(int))
    for i, neighbors in enumerate(I):
        query_type = tags[i].get("scene_type", "unknown")
        for j in neighbors[1:k+1]:  # skip self
            neighbor_type = tags[j].get("scene_type", "unknown")
            conf[query_type][neighbor_type] += 1

    # Convert to percentage matrix
    matrix = np.zeros((len(scene_types), len(scene_types)))
    for i, qt in enumerate(scene_types):
        row_total = sum(conf[qt].values())
        for j, nt in enumerate(scene_types):
            matrix[i, j] = conf[qt][nt] / row_total * 100 if row_total > 0 else 0

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='Blues')

    ax.set_xticks(range(len(scene_types)))
    ax.set_yticks(range(len(scene_types)))
    ax.set_xticklabels(scene_types, rotation=45, ha='right')
    ax.set_yticklabels(scene_types)
    ax.set_xlabel('Retrieved Scene Type')
    ax.set_ylabel('Query Scene Type')
    ax.set_title(f'V-JEPA kNN Retrieval Accuracy (k={k})\nDiagonal = Correct Retrievals')

    # Add text annotations
    for i in range(len(scene_types)):
        for j in range(len(scene_types)):
            color = 'white' if matrix[i, j] > 50 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.0f}%', ha='center', va='center', color=color, fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% of Retrievals')

    plt.tight_layout()

    # Save both formats
    png_path = output_dir / "m07_confusion_matrix.png"
    pdf_path = output_dir / "m07_confusion_matrix.pdf"
    plt.savefig(png_path, dpi=150)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Saved: {png_path}")

    # Print per-class accuracy
    print("\nPer-class retrieval accuracy:")
    for i, st in enumerate(scene_types):
        print(f"  {st}: {matrix[i, i]:.1f}%")


def create_umap_plot(
    embeddings: np.ndarray,
    tags: list,
    output_png: Path,
    output_pdf: Path,
    n_neighbors: int = 15,
    min_dist: float = 0.1
):
    """
    Create UMAP visualization of embeddings colored by scene type.
    Saves both PNG and PDF versions.
    """
    print(f"Computing UMAP projection (n_neighbors={n_neighbors}, min_dist={min_dist})...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=True
    )
    embedding_2d = reducer.fit_transform(embeddings)
    print(f"UMAP projection shape: {embedding_2d.shape}")

    # Get colors based on scene type
    scene_types = [t.get("scene_type", "unknown") for t in tags]
    colors = [SCENE_COLORS.get(st, SCENE_COLORS["unknown"]) for st in scene_types]

    # Count scene types
    scene_counts = {}
    for st in scene_types:
        scene_counts[st] = scene_counts.get(st, 0) + 1

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=colors,
        s=80,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.5
    )

    # Create legend
    legend_patches = []
    for scene_type, color in SCENE_COLORS.items():
        if scene_type in scene_counts:
            count = scene_counts[scene_type]
            patch = mpatches.Patch(color=color, label=f"{scene_type} (n={count})")
            legend_patches.append(patch)

    ax.legend(handles=legend_patches, loc='upper right', fontsize=10)

    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('V-JEPA Embeddings of Indian Street Videos\n(Colored by Scene Type from Qwen3-VL)', fontsize=14)

    # Add metrics if available
    if METRICS_FILE.exists():
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        metrics_text = (
            f"Self-Consistency: {metrics.get('self_consistency', 'N/A')}%\n"
            f"Cluster Purity: {metrics.get('cluster_purity', 'N/A')}%"
        )
        ax.text(
            0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

    plt.tight_layout()

    # Save both PNG and PDF (as per src/CLAUDE.md rule 8)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate UMAP visualization and confusion matrix")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    args = parser.parse_args()

    # Check if plots already exist
    output_files = [
        UMAP_PLOT_PNG,
        OUTPUTS_DIR / "m07_confusion_matrix.png"
    ]
    existing = [f for f in output_files if f.exists()]
    if existing:
        if not check_output_exists(existing, "plots"):
            print("Using cached plots.")
            return

    # Load embeddings and tags
    embeddings, tags = load_embeddings_and_tags()

    # Adjust UMAP parameters for small datasets
    n_neighbors = min(args.n_neighbors, len(tags) - 1)
    if n_neighbors < 2:
        print("ERROR: Need at least 3 samples for UMAP")
        sys.exit(1)

    # Generate UMAP plot
    create_umap_plot(
        embeddings,
        tags,
        UMAP_PLOT_PNG,
        UMAP_PLOT_PDF,
        n_neighbors=n_neighbors,
        min_dist=args.min_dist
    )

    # Generate confusion matrix
    create_confusion_matrix(
        embeddings,
        tags,
        OUTPUTS_DIR,
        k=FAISS_K_NEIGHBORS - 1  # k=5 neighbors (excluding self)
    )

    print("\n=== VISUALIZATION COMPLETE ===")
    print(f"PNG: {UMAP_PLOT_PNG}")
    print(f"PDF: {UMAP_PLOT_PDF}")


if __name__ == "__main__":
    main()
