"""
GPU-only UMAP dimensionality reduction on V-JEPA embeddings via cuML. Outputs umap_2d.npy.

USAGE:
    python -u src/m07_umap.py --SANITY 2>&1 | tee logs/m07_umap_sanity.log
    python -u src/m07_umap.py --POC --subset data/subset_10k_local/subset_10k.json 2>&1 | tee logs/m07_umap_poc.log
    python -u src/m07_umap.py --FULL 2>&1 | tee logs/m07_umap_full.log
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.progress import make_pbar
from utils.config import (
    check_gpu, get_pipeline_config,
    add_subset_arg, get_output_dir, get_module_output_dir,
    add_encoder_arg, get_encoder_files,
)
from utils.wandb_utils import add_wandb_args, init_wandb, log_artifact, finish_wandb

# iter18 W7 (PLR2004): semantic named constants.
_MIN_NEIGHBORS = 2   # UMAP needs >=2 neighbors

try:
    from cuml.manifold import UMAP as cuUMAP
except ImportError:
    print("FATAL: cuML not installed. GPU UMAP required (no CPU fallback).")
    print("Install via setup_env_uv.sh --gpu")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GPU UMAP reduction on V-JEPA embeddings (cuML)")
    parser.add_argument("--SANITY", action="store_true", help="First 200 clips only")
    parser.add_argument("--POC", action="store_true", help="POC subset (~10K clips)")
    parser.add_argument("--FULL", action="store_true", help="All clips")
    # iter18 H2: defaults live in pipeline.yaml eval.umap_* (None → yaml).
    parser.add_argument("--n-neighbors", type=int, default=None,
                        help="UMAP n_neighbors (default: pipeline.yaml eval.umap_n_neighbors)")
    parser.add_argument("--min-dist", type=float, default=None,
                        help="UMAP min_dist (default: pipeline.yaml eval.umap_min_dist)")
    add_encoder_arg(parser)
    add_subset_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()
    # iter18 H2: None → pipeline.yaml eval.umap_* (single source).
    _ecfg = get_pipeline_config()["eval"]
    if args.n_neighbors is None:
        args.n_neighbors = _ecfg["umap_n_neighbors"]
    if args.min_dist is None:
        args.min_dist = _ecfg["umap_min_dist"]

    if not (args.SANITY or args.POC or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY, --POC, or --FULL")
        sys.exit(1)

    check_gpu()

    input_dir = get_output_dir(args.subset, sanity=args.SANITY, poc=args.POC)
    output_dir = get_module_output_dir("m07_umap", args.subset, sanity=args.SANITY, poc=args.POC)

    mode = "SANITY" if args.SANITY else ("POC" if args.POC else "FULL")
    wb_run = init_wandb("m07", mode, config=vars(args), enabled=not args.no_wandb)

    # Load embeddings (encoder-aware paths from upstream)
    input_enc_files = get_encoder_files(args.encoder, input_dir)
    enc_files = get_encoder_files(args.encoder, output_dir)
    emb_file = input_enc_files["embeddings"]

    print(f"Encoder: {args.encoder}")
    if not emb_file.exists():
        print(f"FATAL: embeddings not found: {emb_file}")
        sys.exit(1)

    embeddings = np.load(emb_file).astype(np.float32)

    if args.SANITY:
        n = min(200, embeddings.shape[0])
        embeddings = embeddings[:n]
        print(f"SANITY MODE: {n} clips")

    print(f"Loaded: {embeddings.shape[0]:,} clips, dim={embeddings.shape[1]}")

    n_neighbors = min(args.n_neighbors, embeddings.shape[0] - 1)
    if n_neighbors < _MIN_NEIGHBORS:
        print("ERROR: Need at least 3 clips for UMAP")
        sys.exit(1)

    # Run cuML GPU UMAP
    print(f"cuML UMAP (n_neighbors={n_neighbors}, min_dist={args.min_dist})...")
    pbar = make_pbar(total=1, desc="m07_umap", unit="run")
    t0 = time.time()
    reducer = cuUMAP(n_components=2, n_neighbors=n_neighbors,
                     min_dist=args.min_dist, random_state=42, verbose=True)
    result = reducer.fit_transform(embeddings)
    emb_2d = result.get() if hasattr(result, 'get') else np.asarray(result)
    emb_2d = emb_2d.astype(np.float32)  # ensure float32 for downstream compatibility
    elapsed = time.time() - t0
    pbar.update(1)
    pbar.close()

    # Save (encoder-aware path)
    out_path = enc_files["umap_2d"]
    np.save(out_path, emb_2d)
    print(f"Saved: {out_path} ({emb_2d.shape})")
    print(f"UMAP completed in {elapsed:.1f}s")

    log_artifact(wb_run, "umap_2d", str(out_path))
    finish_wandb(wb_run)

    # Force exit: CUDA atexit cleanup can deadlock on futex_wait_queue
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
