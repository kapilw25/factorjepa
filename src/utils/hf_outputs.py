"""
Upload/download compute outputs + POC/val data to HuggingFace Hub.
Repo: anonymousML123/factorjepa-outputs (public, gated, auto-created on first upload).

USAGE:
    # Upload/download: Compute outputs (embeddings, metrics, plots from outputs/full/)
    python -u src/utils/hf_outputs.py upload outputs/full 2>&1 | tee logs/hf_upload.log
    python -u src/utils/hf_outputs.py download outputs/full 2>&1 | tee logs/hf_download.log
    
    # Upload/download: poc 10K (10 TARs, 10.5GB) + val 1K (1 TAR, 0.9GB) + JSON manifests
    python -u src/utils/hf_outputs.py upload-data 2>&1 | tee logs/upload_poc_val.log    # ~15 min upload
    python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/download_poc_val.log # ~3 min measured
    
FULL/POC/SANITY usage:
    python -u src/utils/hf_outputs.py upload outputs 2>&1 | tee logs/hf_upload_outputs.log                                                                    
    python -u src/utils/hf_outputs.py download outputs 2>&1 | tee logs/hf_download_outputs.log 
    
"""
import os
import sys
import time
from pathlib import Path

# Enable Rust-based HF transfer (1.5-3x faster per file).
# Safe here — upload_folder/snapshot_download are single-call APIs, no worker conflict.
# (m00d disables this because its 8 parallel workers + hf_transfer = CDN throttling)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

HF_OUTPUTS_REPO = "anonymousML123/factorjepa-outputs"


def _get_token():
    """Load HF_TOKEN from .env."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # check env directly
    return os.getenv("HF_TOKEN")


def _ensure_repo(token):
    """Auto-create repo if it doesn't exist. Public + gated access."""
    from huggingface_hub import HfApi, repo_exists
    api = HfApi(token=token)

    if repo_exists(HF_OUTPUTS_REPO, repo_type="dataset", token=token):
        return

    print(f"Creating HF repo: {HF_OUTPUTS_REPO}")
    api.create_repo(
        repo_id=HF_OUTPUTS_REPO,
        repo_type="dataset",
        private=False,
    )
    # Set gated access
    api.update_repo_settings(
        repo_id=HF_OUTPUTS_REPO,
        repo_type="dataset",
        gated="auto",
    )
    print(f"Created: https://huggingface.co/datasets/{HF_OUTPUTS_REPO} (public, gated)")


def upload_outputs(output_dir: str, subfolder: str = None):
    """Upload output directory to HF. Uses upload_folder with built-in dedup.

    Args:
        output_dir: Local path (e.g., "outputs/full")
        subfolder: HF path prefix (default: basename of output_dir, e.g., "full")
    """
    from huggingface_hub import HfApi

    token = _get_token()
    if not token:
        print("SKIP upload: HF_TOKEN not found in .env")
        return

    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"SKIP upload: {output_dir} does not exist")
        return

    _ensure_repo(token)

    if subfolder is None:
        subfolder = str(output_path)  # "outputs/full", "outputs/poc" — mirrors local layout

    api = HfApi(token=token)

    # Upload with dedup — HF only uploads files whose hash changed
    print(f"Uploading {output_dir} → {HF_OUTPUTS_REPO}/{subfolder}/")
    t0 = time.time()

    api.upload_folder(
        folder_path=str(output_path),
        repo_id=HF_OUTPUTS_REPO,
        repo_type="dataset",
        path_in_repo=subfolder,
        allow_patterns=["*.npy", "*.npz", "*.json", "*.csv", "*.png", "*.pdf", "*.tex", "*.pt"],
        ignore_patterns=[".*", "tmp_*"],  # skip checkpoints and temp dirs
    )

    elapsed = time.time() - t0
    print(f"Upload complete: {elapsed:.0f}s → https://huggingface.co/datasets/{HF_OUTPUTS_REPO}")


def download_outputs(output_dir: str, subfolder: str = None):
    """Download outputs from HF to local directory. Prints progress.

    Args:
        output_dir: Local destination (e.g., "outputs/full")
        subfolder: HF path prefix (default: basename of output_dir)
    """
    from huggingface_hub import snapshot_download, repo_exists

    token = _get_token()
    if not token:
        print("SKIP download: HF_TOKEN not found in .env")
        return False

    if not repo_exists(HF_OUTPUTS_REPO, repo_type="dataset", token=token):
        print(f"SKIP download: repo {HF_OUTPUTS_REPO} does not exist")
        return False

    if subfolder is None:
        subfolder = str(Path(output_dir))  # "outputs/full" — mirrors HF layout

    print(f"Downloading {HF_OUTPUTS_REPO}/{subfolder}/ → {output_dir}")
    t0 = time.time()

    snapshot_download(
        repo_id=HF_OUTPUTS_REPO,
        repo_type="dataset",
        local_dir=".",  # download to project root, HF preserves subfolder structure
        allow_patterns=[f"{subfolder}/*"],
        token=token,
    )

    elapsed = time.time() - t0
    print(f"Download complete: {elapsed:.0f}s")
    return True


def upload_after_step(output_dir: str):
    """Call at the end of each GPU script to upload new outputs. Non-fatal on failure."""
    try:
        upload_outputs(output_dir)
    except Exception as e:
        print(f"  HF upload failed (non-fatal): {e}")


def upload_data():
    """Upload POC + val data to HF for reproducibility across GPU instances.

    Uploads:
      data/subset_10k.json + data/subset_10k_local/*.tar  → factorjepa-outputs/data/subset_10k_local/
      data/val_1k.json + data/val_1k_local/*.tar           → factorjepa-outputs/data/val_1k_local/
    """
    from huggingface_hub import HfApi

    token = _get_token()
    if not token:
        print("SKIP: HF_TOKEN not found")
        return

    _ensure_repo(token)
    api = HfApi(token=token)

    uploads = [
        ("data/subset_10k.json", "data/subset_10k.json"),
        ("data/val_1k.json", "data/val_1k.json"),
        ("data/subset_10k_local", "data/subset_10k_local"),
        ("data/val_1k_local", "data/val_1k_local"),
    ]

    for local_path, repo_path in uploads:
        p = Path(local_path)
        if not p.exists():
            print(f"  SKIP: {local_path} not found")
            continue

        if p.is_file():
            print(f"  Uploading {local_path} → {repo_path}")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=repo_path,
                repo_id=HF_OUTPUTS_REPO,
                repo_type="dataset",
            )
        elif p.is_dir():
            print(f"  Uploading {local_path}/ → {repo_path}/ ({len(list(p.glob('*')))} files)")
            api.upload_folder(
                folder_path=str(p),
                repo_id=HF_OUTPUTS_REPO,
                repo_type="dataset",
                path_in_repo=repo_path,
            )

    print(f"Data upload complete → https://huggingface.co/datasets/{HF_OUTPUTS_REPO}")


def download_data():
    """Download POC + val data from HF to local data/ directory."""
    from huggingface_hub import snapshot_download, repo_exists

    token = _get_token()
    if not token:
        print("SKIP: HF_TOKEN not found")
        return False

    if not repo_exists(HF_OUTPUTS_REPO, repo_type="dataset", token=token):
        print(f"SKIP: {HF_OUTPUTS_REPO} does not exist")
        return False

    print(f"Downloading data from {HF_OUTPUTS_REPO}/data/ → data/")
    t0 = time.time()

    snapshot_download(
        repo_id=HF_OUTPUTS_REPO,
        repo_type="dataset",
        local_dir=".",
        allow_patterns=["data/*.json", "data/subset_10k_local/*", "data/val_1k_local/*"],
        token=token,
    )

    print(f"Data download complete: {time.time() - t0:.0f}s")
    return True


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -u src/utils/hf_outputs.py upload <output_dir>")
        print("  python -u src/utils/hf_outputs.py download <output_dir>")
        print("  python -u src/utils/hf_outputs.py upload-data")
        print("  python -u src/utils/hf_outputs.py download-data")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "upload" and len(sys.argv) >= 3:
        upload_outputs(sys.argv[2])
    elif cmd == "download" and len(sys.argv) >= 3:
        download_outputs(sys.argv[2])
    elif cmd == "upload-data":
        upload_data()
    elif cmd == "download-data":
        download_data()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
