"""
Generate UMAP visualization of V-JEPA embeddings colored by scene type.
CPU-based visualization script (M1 compatible - no GPU inference).

USAGE:
    python -u src/m06_umap_plot.py --SANITY 2>&1 | tee logs/m06_umap_plot_sanity.log
    python -u src/m06_umap_plot.py --FULL 2>&1 | tee logs/m06_umap_plot_full.log
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
    UMAP_PLOT_PNG, UMAP_PLOT_PDF
)

try:
    import umap
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install umap-learn matplotlib")
    sys.exit(1)


# Color palette for scene types
SCENE_COLORS = {
    "market": "#e41a1c",
    "temple": "#377eb8",
    "junction": "#4daf4a",
    "lane": "#984ea3",
    "highway": "#ff7f00",
    "residential": "#ffff33",
    "commercial": "#a65628",
    "unknown": "#999999",
}


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
    parser = argparse.ArgumentParser(description="Generate UMAP visualization")
    parser.add_argument("--SANITY", action="store_true", help="Quick test")
    parser.add_argument("--FULL", action="store_true", help="Full computation")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

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

    print("\n=== VISUALIZATION COMPLETE ===")
    print(f"PNG: {UMAP_PLOT_PNG}")
    print(f"PDF: {UMAP_PLOT_PDF}")


if __name__ == "__main__":
    main()
