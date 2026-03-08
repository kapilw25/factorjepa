"""
CPU-only visualization: UMAP scatter + kNN confusion matrix + kNN grids.
Reads pre-computed outputs: umap_2d.npy, knn_indices.npy, tags.json, m06_metrics.json.
No FAISS, no cuML, no torch. Runs on M1 Mac.

USAGE:
    python -u src/m08_plot.py --SANITY 2>&1 | tee logs/m08_plot_sanity.log
    python -u src/m08_plot.py --FULL --subset data/subset_10k.json 2>&1 | tee logs/m08_plot_poc.log
    python -u src/m08_plot.py --FULL 2>&1 | tee logs/m08_plot_full.log
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    EMBEDDINGS_FILE, TAGS_FILE, METRICS_FILE, OUTPUTS_DIR,
    FAISS_K_NEIGHBORS,
    add_subset_arg, get_output_dir,
)

TAXONOMY_FILE = Path(__file__).parent / "utils" / "tag_taxonomy.json"
SINGLE_VALUE_KEYS = []
if TAXONOMY_FILE.exists():
    with open(TAXONOMY_FILE) as _f:
        _tax = json.load(_f)
    SINGLE_VALUE_KEYS = [k for k, v in _tax.items()
                         if not k.startswith("_") and v.get("type") == "single"]
from utils.wandb_utils import (
    add_wandb_args, init_wandb, log_metrics, log_image, finish_wandb,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install matplotlib")
    sys.exit(1)

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

N_KNN_GRID_ROWS = 8


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


# ── Dynamic color palette ────────────────────────────────────────────────

_TABLEAU20 = [
    "#e41a1c", "#4daf4a", "#984ea3", "#377eb8", "#999999",
    "#ff7f00", "#ffff33", "#a65628", "#17becf", "#f781bf",
    "#1f77b4", "#aec7e8", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f", "#dbdb8d",
]

def _get_color_map(values: list) -> dict:
    """Build a value→color map. Uses SCENE_COLORS for scene_type values, else auto-assigns."""
    unique = sorted(set(values))
    cmap = {}
    for i, v in enumerate(unique):
        cmap[v] = SCENE_COLORS.get(v, _TABLEAU20[i % len(_TABLEAU20)])
    return cmap


# ── UMAP Scatter Plot ────────────────────────────────────────────────────

def create_umap_plot(emb_2d: np.ndarray, tags: list, output_dir: Path,
                     metrics_data: dict, field: str = "scene_type"):
    """UMAP scatter colored by a taxonomy field with metrics overlay."""
    values = [t.get(field, "unknown") for t in tags]
    color_map = _get_color_map(values)
    colors = [color_map.get(v, "#666666") for v in values]
    counts = defaultdict(int)
    for v in values:
        counts[v] += 1

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=max(5, 80 - len(tags) // 200),
               alpha=0.7, edgecolors='white', linewidths=0.3)

    legend_patches = [mpatches.Patch(color=color_map[v], label=f"{v} (n={counts[v]})")
                      for v in sorted(color_map) if v in counts]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    label = field.replace("_", " ").title()
    ax.set_title(f'V-JEPA Embeddings (n={len(tags):,}, colored by {field})', fontsize=14)

    if metrics_data and field == "scene_type":
        easy = metrics_data.get("easy", {})
        lines = []
        for m in ["cycle_at_k", "prec_at_k", "silhouette"]:
            v = easy.get(m)
            if v is not None:
                ml = m.replace("_", " ").title()
                lines.append(f"{ml}: {v}")
        macro = easy.get("macro_avg", {}).get("prec_at_k")
        if macro is not None:
            lines.append(f"Macro Prec@K: {macro}%")
        if lines:
            ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.tight_layout()
    suffix = f"_{field}" if field != "scene_type" else ""
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08_umap{suffix}{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / f'm08_umap{suffix}.png'}")


# ── Confusion Matrix ─────────────────────────────────────────────────────

def create_confusion_matrix(knn_indices: np.ndarray, tags: list, output_dir: Path,
                            k: int = 5, field: str = "scene_type"):
    """kNN retrieval confusion matrix from pre-computed indices (no FAISS needed)."""
    field_values = sorted(set(t.get(field, "unknown") for t in tags))

    conf = defaultdict(lambda: defaultdict(int))
    for i, neighbors in enumerate(knn_indices):
        qt = tags[i].get(field, "unknown")
        for j in neighbors[1:k + 1]:  # skip self (col 0)
            if 0 <= j < len(tags):
                conf[qt][tags[j].get(field, "unknown")] += 1

    matrix = np.zeros((len(field_values), len(field_values)))
    for i, qt in enumerate(field_values):
        total = sum(conf[qt].values())
        for j, nt in enumerate(field_values):
            matrix[i, j] = conf[qt][nt] / total * 100 if total > 0 else 0

    label = field.replace("_", " ").title()
    fig, ax = plt.subplots(figsize=(max(8, len(field_values) * 0.9 + 2),
                                    max(6, len(field_values) * 0.8 + 2)))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(len(field_values)))
    ax.set_yticks(range(len(field_values)))
    ax.set_xticklabels(field_values, rotation=45, ha='right')
    ax.set_yticklabels(field_values)
    ax.set_xlabel(f'Retrieved {label}')
    ax.set_ylabel(f'Query {label}')
    ax.set_title(f'kNN Retrieval Confusion — {label} (k={k})')

    fontsize = 9 if len(field_values) <= 10 else 7
    for i in range(len(field_values)):
        for j in range(len(field_values)):
            color = 'white' if matrix[i, j] > 50 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.0f}%', ha='center', va='center',
                    color=color, fontsize=fontsize)

    plt.colorbar(im, ax=ax, label='% of Retrievals')
    plt.tight_layout()
    suffix = f"_{field}" if field != "scene_type" else ""
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08_confusion_matrix{suffix}{ext}",
                    dpi=150 if ext == ".png" else None)
    plt.close()
    print(f"Saved: {output_dir / f'm08_confusion_matrix{suffix}.png'}")

    if field == "scene_type":
        print("\nPer-class retrieval accuracy:")
        for i, st in enumerate(field_values):
            print(f"  {st}: {matrix[i, i]:.1f}%")


# ── Combined 3x3 UMAP Grid ──────────────────────────────────────────────

def create_umap_grid(emb_2d: np.ndarray, tags: list, output_dir: Path,
                     plot_keys: list):
    """3x3 grid of UMAP scatters, one per taxonomy key."""
    n_panels = len(plot_keys)
    n_cols, n_rows = 3, (n_panels + 2) // 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes_flat = axes.flatten()
    pt_size = max(2, 30 - len(tags) // 300)

    for idx, field in enumerate(plot_keys):
        ax = axes_flat[idx]
        values = [t.get(field, "unknown") for t in tags]
        color_map = _get_color_map(values)
        colors = [color_map.get(v, "#666666") for v in values]
        counts = defaultdict(int)
        for v in values:
            counts[v] += 1

        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=pt_size,
                   alpha=0.6, edgecolors='none')
        legend_patches = [mpatches.Patch(color=color_map[v],
                          label=f"{v} ({counts[v]})")
                          for v in sorted(color_map) if v in counts]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=6,
                  framealpha=0.8, handlelength=1, handletextpad=0.3)
        label = field.replace("_", " ").title()
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=8)
        ax.set_ylabel('UMAP 2', fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f'V-JEPA UMAP — All Taxonomy Keys (n={len(tags):,})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08_umap_grid{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'm08_umap_grid.png'}")


# ── Combined 3x3 Confusion Matrix Grid ─────────────────────────────────

def create_confusion_matrix_grid(knn_indices: np.ndarray, tags: list,
                                 output_dir: Path, k: int, plot_keys: list):
    """3x3 grid of confusion matrices, one per taxonomy key."""
    n_panels = len(plot_keys)
    n_cols, n_rows = 3, (n_panels + 2) // 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes_flat = axes.flatten()

    for idx, field in enumerate(plot_keys):
        ax = axes_flat[idx]
        field_values = sorted(set(t.get(field, "unknown") for t in tags))

        conf = defaultdict(lambda: defaultdict(int))
        for i, neighbors in enumerate(knn_indices):
            qt = tags[i].get(field, "unknown")
            for j in neighbors[1:k + 1]:
                if 0 <= j < len(tags):
                    conf[qt][tags[j].get(field, "unknown")] += 1

        matrix = np.zeros((len(field_values), len(field_values)))
        for i, qt in enumerate(field_values):
            total = sum(conf[qt].values())
            for j, nt in enumerate(field_values):
                matrix[i, j] = conf[qt][nt] / total * 100 if total > 0 else 0

        im = ax.imshow(matrix, cmap='Blues')
        ax.set_xticks(range(len(field_values)))
        ax.set_yticks(range(len(field_values)))
        ax.set_xticklabels(field_values, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(field_values, fontsize=7)
        label = field.replace("_", " ").title()
        ax.set_title(label, fontsize=11, fontweight='bold')

        fontsize = 7 if len(field_values) <= 6 else 5
        for i in range(len(field_values)):
            for j in range(len(field_values)):
                color = 'white' if matrix[i, j] > 50 else 'black'
                ax.text(j, i, f'{matrix[i, j]:.0f}%', ha='center', va='center',
                        color=color, fontsize=fontsize)

    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f'kNN Retrieval Confusion — All Taxonomy Keys (k={k})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in [".png", ".pdf"]:
        plt.savefig(output_dir / f"m08_confusion_matrix_grid{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'm08_confusion_matrix_grid.png'}")


# ── kNN Neighbor Grid ────────────────────────────────────────────────────

def create_knn_grid(knn_indices: np.ndarray, tags: list, clip_paths: list,
                    output_dir: Path, k: int = 5, n_rows: int = N_KNN_GRID_ROWS):
    """Visual grid: each row = [query clip] -> [k nearest neighbors]."""
    n = len(tags)
    scene_types = sorted(set(t.get("scene_type", "unknown") for t in tags))

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
            idx = qi if col == 0 else knn_indices[qi, col]

            if idx < 0 or idx >= n:
                ax.set_facecolor("#333333")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            scene = tags[idx].get("scene_type", "unknown")
            color = SCENE_COLORS.get(scene, "#666666")

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
        plt.savefig(output_dir / f"m08_knn_grid{ext}",
                    dpi=150 if ext == ".png" else None, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'm08_knn_grid.png'}")


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
        description="CPU-only visualization: UMAP scatter + confusion matrix + kNN grids")
    parser.add_argument("--SANITY", action="store_true", help="Run on first 200 clips")
    parser.add_argument("--FULL", action="store_true", help="Run on all clips")
    parser.add_argument("--k", type=int, default=FAISS_K_NEIGHBORS,
                        help=f"kNN neighbors (default: {FAISS_K_NEIGHBORS})")
    parser.add_argument("--no-grid", action="store_true", help="Skip kNN grid (faster)")
    add_subset_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    output_dir = get_output_dir(args.subset, sanity=args.SANITY)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = "SANITY" if args.SANITY else ("POC" if args.subset else "FULL")
    wb_run = init_wandb("m08", mode, config=vars(args), enabled=not args.no_wandb)

    print(f"Output dir: {output_dir}")

    # Resolve file paths (all relative to output_dir)
    tags_file = output_dir / "tags.json"
    paths_file = output_dir / "embeddings.paths.npy"
    metrics_file = output_dir / "m06_metrics.json"
    umap_file = output_dir / "umap_2d.npy"
    knn_file = output_dir / "knn_indices.npy"

    # Check required files
    for f, desc, prereq in [
        (umap_file, "UMAP 2D coords", "m07_umap.py"),
        (knn_file, "kNN indices", "m06_faiss_metrics.py"),
        (tags_file, "tags", "m04_vlm_tag.py"),
    ]:
        if not f.exists():
            print(f"FATAL: {desc} not found: {f}")
            print(f"  Run {prereq} first.")
            sys.exit(1)

    emb_2d = np.load(umap_file)
    knn_indices = np.load(knn_file)
    with open(tags_file) as f:
        tags = json.load(f)

    clip_paths = []
    if paths_file.exists():
        clip_paths = np.load(paths_file, allow_pickle=True).tolist()

    # Align
    n = min(emb_2d.shape[0], knn_indices.shape[0], len(tags))
    emb_2d = emb_2d[:n]
    knn_indices = knn_indices[:n]
    tags = tags[:n]
    if clip_paths:
        clip_paths = clip_paths[:n]

    if args.SANITY:
        n = min(200, n)
        emb_2d = emb_2d[:n]
        knn_indices = knn_indices[:n]
        tags = tags[:n]
        clip_paths = clip_paths[:n] if clip_paths else []
        print(f"SANITY MODE: {n} clips")

    print(f"Loaded: {n:,} clips")

    # Load m06 metrics
    metrics_data = {}
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics_data = json.load(f)
        print(f"Loaded m06 metrics: {metrics_file.name}")

    k = min(args.k, n - 1)

    # Taxonomy keys to iterate (scene_type first, then remaining single-value keys)
    plot_keys = ["scene_type"]
    for fk in SINGLE_VALUE_KEYS:
        if fk != "scene_type" and fk not in plot_keys:
            plot_keys.append(fk)
    print(f"Taxonomy keys for plots: {plot_keys}")

    # 1. UMAP scatters (individual per key + combined 3x3 grid)
    for fk in plot_keys:
        create_umap_plot(emb_2d, tags, output_dir, metrics_data, field=fk)
    create_umap_grid(emb_2d, tags, output_dir, plot_keys)

    # 2. Confusion matrices (individual per key + combined 3x3 grid)
    for fk in plot_keys:
        create_confusion_matrix(knn_indices, tags, output_dir, k=k, field=fk)
    create_confusion_matrix_grid(knn_indices, tags, output_dir, k=k,
                                 plot_keys=plot_keys)

    # 3. kNN neighbor grid (scene_type only — visual design is scene-specific)
    if not args.no_grid:
        create_knn_grid(knn_indices, tags, clip_paths, output_dir, k=k)

    # 4. Macro/micro summary
    print("\n=== MACRO/MICRO REPORTING ===")
    print_macro_micro(metrics_data)

    # wandb
    if metrics_data:
        for prefix in ["easy", "hard"]:
            m = metrics_data.get(prefix, {})
            macro = m.get("macro_avg", {}).get("prec_at_k")
            micro = m.get("micro_avg", {}).get("prec_at_k")
            if macro is not None:
                log_metrics(wb_run, {f"{prefix}/macro_prec_at_k": macro})
            if micro is not None:
                log_metrics(wb_run, {f"{prefix}/micro_prec_at_k": micro})

    for fk in plot_keys:
        suffix = f"_{fk}" if fk != "scene_type" else ""
        log_image(wb_run, f"umap{suffix}", str(output_dir / f"m08_umap{suffix}.png"))
        log_image(wb_run, f"confusion_matrix{suffix}",
                  str(output_dir / f"m08_confusion_matrix{suffix}.png"))
    log_image(wb_run, "umap_grid", str(output_dir / "m08_umap_grid.png"))
    log_image(wb_run, "confusion_matrix_grid",
              str(output_dir / "m08_confusion_matrix_grid.png"))
    if not args.no_grid:
        log_image(wb_run, "knn_grid", str(output_dir / "m08_knn_grid.png"))

    print(f"\n=== VISUALIZATION COMPLETE ===")
    print(f"Outputs in: {output_dir}")
    finish_wandb(wb_run)


if __name__ == "__main__":
    main()
