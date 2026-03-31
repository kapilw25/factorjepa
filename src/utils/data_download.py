"""
Auto-download WebDataset shards from HF if local data dir is missing.
Reuses m00d_download_subset.py logic. Called by GPU scripts before data streaming.

Usage:
    from utils.data_download import ensure_local_data
    ensure_local_data(args)  # downloads if --local-data dir missing, no-op if exists
"""
import sys
from pathlib import Path


def _derive_local_dir(subset_path: str) -> Path:
    """Derive local data dir from subset filename: data/subset_10k.json → data/subset_10k_local/"""
    p = Path(subset_path)
    return p.parent / f"{p.stem}_local"


def ensure_local_data(args) -> str:
    """Ensure local WebDataset shards exist. Auto-download from HF if missing.

    Checks args.local_data (or derives from args.subset). If the directory
    exists with a manifest.json, returns the path (no-op). Otherwise,
    invokes m00d_download_subset.py to download from HF CDN.

    Args:
        args: Parsed argparse namespace with .local_data, .subset, .SANITY attributes.

    Returns:
        str: Path to local data directory (guaranteed to exist after call).
    """
    local_data = getattr(args, "local_data", None)

    # Derive local_data path from --subset if not explicitly provided
    if not local_data and getattr(args, "subset", None):
        local_data = str(_derive_local_dir(args.subset))
        args.local_data = local_data

    # For --FULL without --subset, use data/full_local
    if not local_data and not getattr(args, "subset", None):
        local_data = "data/full_local"
        args.local_data = local_data

    # SANITY mode can stream (small clip count) — no download needed
    if getattr(args, "SANITY", False):
        return local_data

    local_path = Path(local_data)
    manifest = local_path / "manifest.json"

    # Already downloaded — no-op
    if local_path.exists() and manifest.exists():
        return local_data

    # Auto-download
    print(f"\n{'='*60}")
    print(f"  AUTO-DOWNLOAD: {local_data} not found")
    print(f"  Downloading WebDataset shards from HuggingFace CDN...")
    print(f"  This is a one-time operation (~50-60 min for 10K, ~60 min for 115K)")
    print(f"{'='*60}\n")

    # Build m00d args
    import argparse
    m00d_args = argparse.Namespace(
        SANITY=False,
        FULL=True,
        subset=getattr(args, "subset", None),
        no_wandb=True,
    )

    # Import and run m00d download
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from m00d_download_subset import download_subset
    download_subset(m00d_args)

    # Verify download succeeded
    if not manifest.exists():
        print(f"FATAL: Download failed — {manifest} not created")
        sys.exit(1)

    print(f"\nDownload complete: {local_data}")
    return local_data
