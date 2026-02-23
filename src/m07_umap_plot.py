"""
UMAP visualization + kNN confusion matrix + kNN neighbor grids + macro/micro reporting.
CPU-only (M1 compatible). Reads embeddings.npy, tags.json, and optionally m06_metrics.json.

USAGE:
    python -u src/m07_umap_plot.py --SANITY 2>&1 | tee logs/m07_umap_plot_sanity.log
    python -u src/m07_umap_plot.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m07_umap_plot_poc.log
    python -u src/m07_umap_plot.py --FULL 2>&1 | tee logs/m07_umap_plot_full.log
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    EMBEDDINGS_FILE, TAGS_FILE, METRICS_FILE, OUTPUTS_DIR,
    CLIPS_DIR, FAISS_K_NEIGHBORS,
    add_subset_arg, get_output_dir,
)

try:
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install umap-learn matplotlib")
    sys.exit(1)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# ── Scene type color palette ─────────────────────────────────────────────
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

N_KNN_GRID_ROWS = 8  # number of query clips to show in kNN grid


# ── Video frame extraction ───────────────────────────────────────────────

def extract_frame(clip_path: str, size: int = 128) -> np.ndarray:
    """Extract middle frame from a video clip via ffmpeg. Returns RGB array or None."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(clip_path),
            "-vf", f"select=eq(n\\,5),scale={size}:{size}:force_original_aspect_ratio=decrease,pad={size}:{size}:(ow-iw)/2:(oh-ih)/2",
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode == 0 and len(result.stdout) == size * size * 3:
            return np.frombuffer(result.stdout, dtype=np.uint8).reshape(size, size, 3)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def make_placeholder(label: str, color: str, size: int = 128) -> np.ndarray:
    """Generate a colored placeholder thumbnail with text."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(1, 1), dpi=size)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(color)
    ax.text(0.5, 0.5, label[:12], ha='center', va='center', fontsize=8,
            color='white', fontweight='bold', transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(size, size, 4)[:, :, :3]


# ── UMAP Scatter Plot ────────────────────────────────────────────────────

def create_umap_plot(embeddings: np.ndarray, tags: list, output_dir: Path,
                     metrics_data: dict, n_neighbors: int = 15, min_dist: float = 0.1):
    """UMAP scatter colored by scene_type with metrics overlay."""
    print(f"Computing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=42, verbose=True)
    emb_2d = reducer.fit_transform(embeddings)

    scene_types = [t.get("scene_type", "unknown") for t in tags]
    colors = [SCENE_COLORS.get(st, SCENE_COLORS["unknown"]) for st in scene_types]
    scene_counts = defaultdict(int)
    for st in scene_types:
        scene_counts[st] += 1

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=max(5, 80 - len(tags) // 200),
               alpha=0.7, edgecolors='white', linewidths=0.3)

    legend_patches = [mpatches.Patch(color=c, label=f"{st} (n={scene_counts[st]})")
                      for st, c in SCENE_COLORS.items() if st in scene_counts]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(f'V-JEPA Embeddings (n={len(tags):,}, colored by scene_type)', fontsize=14)

    # Overlay metrics from m06 if available
    if metrics_data:
        easy = metrics_data.get("easy", {})
        lines = []
        for m in ["cycle_at_k", "prec_at_k", "silhouette"]:
            v = easy.get(m)
            if v is not None:
                label = m.replace("_", " ").title()
                lines.append(f"{label}: {v}")
        macro = easy.get("macro_avg", {}).get("prec_at_k")
        if macro is not None:
            lines.append(f"Macro Prec@K: {macro}%")
        if lines:
            ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m07_umap{ext}", dpi=150 if ext == ".png" else None,
                    bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'm07_umap.png'}")


# ── Confusion Matrix ─────────────────────────────────────────────────────

def create_confusion_matrix(embeddings: np.ndarray, tags: list, output_dir: Path,
                            k: int = 5):
    """kNN retrieval confusion matrix (CPU FAISS)."""
    if not HAS_FAISS:
        print("WARNING: faiss not available, skipping confusion matrix")
        return

    print(f"Building CPU FAISS for confusion matrix (k={k})...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    _, I = index.search(embeddings, k + 1)

    scene_types = sorted(set(t.get("scene_type", "unknown") for t in tags))

    conf = defaultdict(lambda: defaultdict(int))
    for i, neighbors in enumerate(I):
        qt = tags[i].get("scene_type", "unknown")
        for j in neighbors[1:k + 1]:
            conf[qt][tags[j].get("scene_type", "unknown")] += 1

    matrix = np.zeros((len(scene_types), len(scene_types)))
    for i, qt in enumerate(scene_types):
        total = sum(conf[qt].values())
        for j, nt in enumerate(scene_types):
            matrix[i, j] = conf[qt][nt] / total * 100 if total > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(len(scene_types)))
    ax.set_yticks(range(len(scene_types)))
    ax.set_xticklabels(scene_types, rotation=45, ha='right')
    ax.set_yticklabels(scene_types)
    ax.set_xlabel('Retrieved Scene Type')
    ax.set_ylabel('Query Scene Type')
    ax.set_title(f'kNN Retrieval Confusion (k={k})')

    for i in range(len(scene_types)):
        for j in range(len(scene_types)):
            color = 'white' if matrix[i, j] > 50 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.0f}%', ha='center', va='center',
                    color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='% of Retrievals')
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m07_confusion_matrix{ext}",
                    dpi=150 if ext == ".png" else None)
    plt.close()
    print(f"Saved: {output_dir / 'm07_confusion_matrix.png'}")

    # Per-class accuracy
    print("\nPer-class retrieval accuracy:")
    for i, st in enumerate(scene_types):
        print(f"  {st}: {matrix[i, i]:.1f}%")

    return I  # return indices for kNN grid


# ── kNN Neighbor Grid ────────────────────────────────────────────────────

def create_knn_grid(indices: np.ndarray, tags: list, clip_paths: list,
                    output_dir: Path, k: int = 5, n_rows: int = N_KNN_GRID_ROWS):
    """
    Visual grid: each row = [query clip] → [k nearest neighbors].
    Selects diverse queries across scene types for representative display.
    """
    n = len(tags)
    scene_types = sorted(set(t.get("scene_type", "unknown") for t in tags))

    # Select diverse queries: pick from each scene type round-robin
    rng = np.random.RandomState(42)
    by_scene = defaultdict(list)
    for i, t in enumerate(tags):
        by_scene[t.get("scene_type", "unknown")].append(i)

    selected = []
    scene_cycle = list(scene_types)
    rng.shuffle(scene_cycle)
    cycle_idx = 0
    while len(selected) < n_rows and cycle_idx < n_rows * len(scene_types):
        st = scene_cycle[cycle_idx % len(scene_cycle)]
        candidates = by_scene.get(st, [])
        if candidates:
            pick = rng.choice(candidates)
            if pick not in selected:
                selected.append(pick)
        cycle_idx += 1

    if not selected:
        print("WARNING: No queries selected for kNN grid")
        return

    # Build grid: n_rows x (1 + k) thumbnails
    n_cols = 1 + k
    thumb_size = 128

    fig, axes = plt.subplots(len(selected), n_cols, figsize=(n_cols * 1.8, len(selected) * 1.8))
    if len(selected) == 1:
        axes = [axes]

    for row, qi in enumerate(selected):
        query_scene = tags[qi].get("scene_type", "unknown")
        query_color = SCENE_COLORS.get(query_scene, "#666666")

        for col in range(n_cols):
            ax = axes[row][col] if n_cols > 1 else axes[row]
            idx = qi if col == 0 else indices[qi, col]  # col 0=self, 1..k=neighbors

            if idx < 0 or idx >= n:
                ax.set_facecolor("#333333")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            scene = tags[idx].get("scene_type", "unknown")
            color = SCENE_COLORS.get(scene, "#666666")

            # Try to extract video frame
            thumb = None
            if clip_paths and idx < len(clip_paths):
                p = Path(clip_paths[idx])
                if p.exists():
                    thumb = extract_frame(str(p), thumb_size)

            if thumb is not None:
                ax.imshow(thumb)
            else:
                placeholder = make_placeholder(scene, color, thumb_size)
                ax.imshow(placeholder)

            ax.set_xticks([])
            ax.set_yticks([])

            # Border: green if same scene as query, red if different
            if col == 0:
                for spine in ax.spines.values():
                    spine.set_color(query_color)
                    spine.set_linewidth(3)
                ax.set_ylabel(query_scene, fontsize=7, rotation=0, labelpad=60,
                              va='center', fontweight='bold')
            else:
                match = scene == query_scene
                border_color = '#4CAF50' if match else '#F44336'
                for spine in ax.spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(2)

            if col == 0:
                ax.set_title("Query", fontsize=8, fontweight='bold')
            elif row == 0:
                ax.set_title(f"NN-{col}", fontsize=8)

    plt.suptitle(f'kNN Neighbor Grid (k={k})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m07_knn_grid{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'm07_knn_grid.png'}")


# ── Macro/Micro Summary ─────────────────────────────────────────────────

def print_macro_micro(metrics_data: dict):
    """Print macro/micro summary from m06 metrics."""
    if not metrics_data:
        print("No m06 metrics available for macro/micro reporting")
        return

    for mode in ["easy", "hard"]:
        m = metrics_data.get(mode, {})
        macro = m.get("macro_avg", {})
        micro = m.get("micro_avg", {})
        print(f"\n{mode.upper()} mode:")
        print(f"  Macro Prec@K: {macro.get('prec_at_k', 'N/A')}%")
        print(f"  Micro Prec@K: {micro.get('prec_at_k', 'N/A')}%")
        for metric in ["cycle_at_k", "prec_at_k", "silhouette", "map_at_k", "ndcg_at_k"]:
            v = m.get(metric)
            if v is not None:
                print(f"  {metric.replace('_', ' ').title()}: {v}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="UMAP visualization + kNN confusion matrix + kNN grids")
    parser.add_argument("--SANITY", action="store_true", help="Run on first 200 clips")
    parser.add_argument("--FULL", action="store_true", help="Run on all clips")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--k", type=int, default=FAISS_K_NEIGHBORS,
                        help=f"kNN neighbors (default: {FAISS_K_NEIGHBORS})")
    parser.add_argument("--no-grid", action="store_true", help="Skip kNN grid (faster)")
    add_subset_arg(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    # Output routing
    output_dir = get_output_dir(args.subset)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {output_dir}")
    if args.subset:
        print(f"[POC] Subset: {args.subset}")

    # Load embeddings + tags (respect POC paths)
    if args.subset:
        emb_file = output_dir / "embeddings.npy"
        tags_file = output_dir / "tags.json"
        paths_file = output_dir / "embeddings.paths.npy"
        metrics_file = output_dir / "m06_metrics.json"
    else:
        emb_file = EMBEDDINGS_FILE
        tags_file = TAGS_FILE
        paths_file = EMBEDDINGS_FILE.with_suffix('.paths.npy')
        metrics_file = METRICS_FILE

    for f, desc in [(emb_file, "embeddings"), (tags_file, "tags")]:
        if not f.exists():
            print(f"FATAL: {desc} not found: {f}")
            sys.exit(1)

    embeddings = np.load(emb_file).astype(np.float32)
    with open(tags_file) as f:
        tags = json.load(f)

    # Clip paths
    clip_paths = []
    if paths_file.exists():
        clip_paths = np.load(paths_file, allow_pickle=True).tolist()

    # Align
    n = min(embeddings.shape[0], len(tags))
    embeddings = embeddings[:n]
    tags = tags[:n]
    if clip_paths:
        clip_paths = clip_paths[:n]

    if args.SANITY:
        n = min(200, n)
        embeddings = embeddings[:n]
        tags = tags[:n]
        clip_paths = clip_paths[:n] if clip_paths else []
        print(f"SANITY MODE: {n} clips")

    print(f"Loaded: {n:,} clips, dim={embeddings.shape[1]}")

    # Load m06 metrics if available
    metrics_data = {}
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics_data = json.load(f)
        print(f"Loaded m06 metrics: {metrics_file.name}")

    # UMAP params
    n_neighbors = min(args.n_neighbors, n - 1)
    if n_neighbors < 2:
        print("ERROR: Need at least 3 clips for UMAP")
        sys.exit(1)

    k = min(args.k, n - 1)

    # 1. UMAP scatter
    create_umap_plot(embeddings, tags, output_dir, metrics_data,
                     n_neighbors=n_neighbors, min_dist=args.min_dist)

    # 2. Confusion matrix (returns kNN indices)
    I = create_confusion_matrix(embeddings, tags, output_dir, k=k)

    # 3. kNN neighbor grid
    if not args.no_grid and I is not None:
        create_knn_grid(I, tags, clip_paths, output_dir, k=k)

    # 4. Macro/micro summary
    print("\n=== MACRO/MICRO REPORTING ===")
    print_macro_micro(metrics_data)

    print(f"\n=== VISUALIZATION COMPLETE ===")
    print(f"Outputs in: {output_dir}")


if __name__ == "__main__":
    main()
