"""
Upload/download compute outputs + POC/val data to HuggingFace Hub.
Repo: anonymousML123/factorjepa-outputs (public, gated, auto-created on first upload).

USAGE:
    # FULL/POC/SANITY usage: from outputs/
    python -u src/utils/hf_outputs.py upload outputs 2>&1 | tee logs/hf_upload_outputs.log
    python -u src/utils/hf_outputs.py download outputs 2>&1 | tee logs/hf_download_outputs.log

    # Upload/download: from outputs/full/ ONLY
    python -u src/utils/hf_outputs.py upload outputs/full 2>&1 | tee logs/hf_upload.log
    python -u src/utils/hf_outputs.py download outputs/full 2>&1 | tee logs/hf_download.log

    # Upload/download: from @data/
    # poc 10K (10 TARs, 10.5GB) + val 1K (1 TAR, 0.9GB) + JSON manifests
    python -u src/utils/hf_outputs.py upload-data 2>&1 | tee logs/upload_poc_val.log    # ~15 min upload
    python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/download_poc_val.log # ~3 min measured



"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from huggingface_hub import HfApi, repo_exists, snapshot_download
from utils.progress import make_pbar

# Enable Rust-based HF transfer (1.5-3x faster per file).
# Safe here — upload_folder/snapshot_download are single-call APIs, no worker conflict.
# (m00d disables this because its 8 parallel workers + hf_transfer = CDN throttling)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

HF_OUTPUTS_REPO = "anonymousML123/factorjepa-outputs"

_UPLOAD_EXTENSIONS = {"*.npy", "*.npz", "*.json", "*.csv", "*.png", "*.pdf", "*.tex", "*.pt"}

# Large regeneratable m11 subdirs — mirror .gitignore lines 80-84. These are
# deliberately EXCLUDED from HF upload because they are cheap to re-compute
# locally via Step B m11 (~10 min factor-gen + ~30-60 min per-clip plots) and
# together add ~58 GB per POC run (D_L=18G + D_A=16G + D_I=21G + per_clip_verify=2.9G).
# Downstream m09c reads from the LOCAL regenerated copies — never needs HF-hosted ones.
_UPLOAD_SKIP_PATTERNS = [
    "**/m11_factor_datasets/D_L/**",                  # gitignore:82
    "**/m11_factor_datasets/D_A/**",                  # gitignore:83
    "**/m11_factor_datasets/D_I/**",                  # gitignore:84
    "**/m11_factor_datasets/m11_per_clip_verify/**",  # gitignore:80
    "**/m10_sam_segment/masks/**",                    # 12,309 .npz mask files (~7 GB) — regeneratable from m10_sam_segment.py
    "**/m10_sam_segment/m10_overlay_verify/**",       # 598 .png overlays (~700 MB) — visual debug only
]

_CHECKPOINT_AGE_THRESHOLD = 120  # seconds — skip checkpoints modified within this window


def _get_token():
    """Load HF_TOKEN from .env."""
    if load_dotenv is not None:
        load_dotenv()
    return os.getenv("HF_TOKEN")


def _ensure_repo(token):
    """Auto-create repo if it doesn't exist. Public + gated access."""
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


def _fmt_size(nbytes: int) -> str:
    """Format bytes as human-readable string."""
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.1f} GB"
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:.1f} MB"
    if nbytes >= 1e3:
        return f"{nbytes / 1e3:.0f} KB"
    return f"{nbytes} B"


def _list_local_files(output_path: Path, extensions: set,
                      skip_patterns: list = None) -> list:
    """List local files matching allowed extensions, sorted by path.

    skip_patterns: list of gitignore-style globs (e.g., `**/m11_factor_datasets/D_L/**`).
    A file is skipped if ANY of its parent-dir path fragments match a skip pattern.
    Preview listing is kept in sync with what upload_folder() actually sends so
    "Uploading N files (X GB)" is accurate.
    """
    files = []
    for ext in extensions:
        # rglob needs the full glob pattern (e.g. "*.npy"); previously ext.lstrip("*")
        # stripped the wildcard → rglob(".npy") literal-matched nothing → preview
        # always showed "0 files (0 GB)" even as upload_folder sent 64 GB.
        files.extend(output_path.rglob(ext))
    files = sorted(set(files))
    if not skip_patterns:
        return files
    # Extract the unique mid-path fragments (e.g., "m11_factor_datasets/D_L")
    # from gitignore globs like "**/m11_factor_datasets/D_L/**". Simple substring
    # match on the POSIX relative path is enough — no need to emulate glob recursion.
    fragments = []
    for pat in skip_patterns:
        frag = pat.strip("/")
        frag = frag[len("**/"):] if frag.startswith("**/") else frag
        frag = frag[:-len("/**")] if frag.endswith("/**") else frag
        fragments.append(frag)
    filtered = []
    for f in files:
        rel = f.relative_to(output_path).as_posix()
        if any(f"/{frag}/" in f"/{rel}/" for frag in fragments):
            continue
        filtered.append(f)
    return filtered


def _list_remote_files(api, subfolder: str) -> list:
    """List remote files under subfolder on HF, returns list of (path, size_bytes).

    huggingface_hub list_repo_tree() returns RepoFile (has .path + .size + .blob_id)
    and RepoFolder (has .path + .tree_id). Discriminate via isinstance — old code
    used hasattr('rpath') which silently no-op'd after the API renamed rpath → path,
    so _mirror_cleanup deleted nothing for months and HF accumulated orphans.
    """
    from huggingface_hub.hf_api import RepoFile
    files = []
    for item in api.list_repo_tree(
            HF_OUTPUTS_REPO, path_in_repo=subfolder, repo_type="dataset",
            recursive=True):
        if isinstance(item, RepoFile):
            size = getattr(item, 'size', 0) or 0
            files.append((item.path, size))
    return sorted(files)


def _mirror_cleanup(api, local_path: Path, subfolder: str):
    """Delete remote HF files not in the current local upload set (true rsync --delete mirror).

    A remote file is "stale" if EITHER:
      (a) it does not exist on local disk at all (renamed/deleted file), OR
      (b) it exists locally but matches _UPLOAD_SKIP_PATTERNS (we deliberately don't
          want to back it up — masks/overlays/factor-tubes that are regeneratable).

    Prevents HF from accumulating orphans across iterations. Bug history: the old
    filter used hasattr(item, 'rpath') which became False after huggingface_hub
    renamed rpath → path → 0 stale files always → silent no-op for months → 73 GB
    of stale checkpoints + every renamed iter11/ path remained on HF forever.
    """
    from huggingface_hub.hf_api import RepoFile
    try:
        # 1. List all files on HF under this subfolder (recursive)
        remote_paths = []
        for item in api.list_repo_tree(
                HF_OUTPUTS_REPO, path_in_repo=subfolder, repo_type="dataset",
                recursive=True):
            if isinstance(item, RepoFile):
                remote_paths.append(item.path)

        if not remote_paths:
            print(f"  Mirror: no remote files under '{subfolder}' (fresh repo or empty subfolder)")
            return

        # 2. Compute the local UPLOAD SET (passes ext + skip filters) as a set of
        #    repo-relative posix paths matching how upload_folder names them on HF.
        local_files = _list_local_files(local_path, _UPLOAD_EXTENSIONS,
                                        skip_patterns=_UPLOAD_SKIP_PATTERNS)
        upload_set = {
            f"{subfolder}/{f.relative_to(local_path).as_posix()}"
            for f in local_files
        }

        # 3. Anything on remote not in upload_set is stale → delete.
        stale = sorted(p for p in remote_paths if p not in upload_set)

        if not stale:
            print(f"  Mirror: HF in sync ({len(remote_paths)} files on remote, "
                  f"all match local upload set, 0 stale)")
            return

        # BATCHED delete via api.delete_files() — single commit per chunk of 1000
        # operations. Was per-file api.delete_file() which made N HTTP round-trips
        # (11,029 deletes ≈ 30-60 min). Batched: ~12 commits total ≈ 30-60 sec.
        # Per HF docs, create_commit (which delete_files wraps) is the recommended
        # bulk-delete pattern; concurrent commits to the same branch race on
        # parent_commit so we serialize chunks. (Did not parallelize across chunks
        # — would create CommitConflictError on same-branch races.)
        CHUNK = 1000
        n_chunks = (len(stale) + CHUNK - 1) // CHUNK
        print(f"  Mirror cleanup: {len(remote_paths)} remote, "
              f"{len(upload_set)} local upload-set, "
              f"{len(stale)} stale → deleting in {n_chunks} batched commit(s) of ≤{CHUNK} ops each...")
        t_del = time.time()
        for i in range(0, len(stale), CHUNK):
            chunk = stale[i:i + CHUNK]
            batch_idx = i // CHUNK + 1
            t_chunk = time.time()
            try:
                api.delete_files(
                    repo_id=HF_OUTPUTS_REPO,
                    delete_patterns=chunk,
                    repo_type="dataset",
                    commit_message=f"Mirror cleanup batch {batch_idx}/{n_chunks}: "
                                   f"delete {len(chunk)} stale files",
                )
                dt = time.time() - t_chunk
                # Sample first 3 + last 3 paths per batch (full list would be 11K lines)
                sample = (chunk[:3] + ["..."] + chunk[-3:]) if len(chunk) > 6 else chunk
                print(f"    batch {batch_idx}/{n_chunks}: DEL {len(chunk)} files in {dt:.1f}s")
                for p in sample:
                    print(f"      {p}")
            except Exception as e:
                print(f"    batch {batch_idx}/{n_chunks} FAILED: {e}")
                print(f"      (chunk had {len(chunk)} paths; first 3: {chunk[:3]})")
        print(f"  Mirror cleanup: done ({len(stale)} files in {n_chunks} commits, {time.time() - t_del:.1f}s)")
    except Exception as e:
        print(f"FATAL: mirror cleanup failed ({e})")
        print("  Stale files on HF will be re-downloaded to other machines.")
        print("  Fix the error above, then re-run upload.")
        sys.exit(1)


def _stale_checkpoint_ignores(output_path: Path) -> list:
    """Return ignore patterns for checkpoint files currently being written.

    Checkpoints older than 60s are safe to upload (no active writer).
    Checkpoints younger than 60s are actively being written → skip to avoid LFS errors.
    """
    now = time.time()
    ignore = []
    for ckpt in output_path.rglob(".*checkpoint*"):
        age = now - ckpt.stat().st_mtime
        if age < _CHECKPOINT_AGE_THRESHOLD:
            # Relative to output_path for ignore_patterns
            rel = str(ckpt.relative_to(output_path))
            ignore.append(rel)
            print(f"  Skip active checkpoint: {rel} (modified {age:.0f}s ago)")
        else:
            print(f"  Include checkpoint: {ckpt.name} (idle {age:.0f}s)")
    return ignore


def upload_outputs(output_dir: str, subfolder: str = None):
    """Upload output directory to HF. Uses upload_folder with built-in dedup.

    Args:
        output_dir: Local path (e.g., "outputs/full")
        subfolder: HF path prefix (default: basename of output_dir, e.g., "full")
    """
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

    # Mirror: delete remote files not present locally before uploading.
    # Without this, HF accumulates stale files (old checkpoints, renamed metrics,
    # deleted plots). Download then pulls them all back → OOM (73GB incident).
    _mirror_cleanup(api, output_path, subfolder)

    # List files that will be uploaded (preview list filtered by the same
    # skip patterns that upload_folder will apply — so "N files (X GB)" is accurate).
    local_files = _list_local_files(output_path, _UPLOAD_EXTENSIONS,
                                    skip_patterns=_UPLOAD_SKIP_PATTERNS)
    total_bytes = sum(f.stat().st_size for f in local_files)
    print(f"Uploading {output_dir} → {HF_OUTPUTS_REPO}/{subfolder}/")
    print(f"  Skipping {len(_UPLOAD_SKIP_PATTERNS)} large regeneratable dirs "
          f"(see _UPLOAD_SKIP_PATTERNS in hf_outputs.py): "
          f"{', '.join(p.strip('/').split('/')[-2] for p in _UPLOAD_SKIP_PATTERNS)}")
    print(f"  {len(local_files)} files ({_fmt_size(total_bytes)}):")
    for f in local_files:
        rel = f.relative_to(output_path)
        print(f"    {rel} ({_fmt_size(f.stat().st_size)})")

    t0 = time.time()

    api.upload_folder(
        folder_path=str(output_path),
        repo_id=HF_OUTPUTS_REPO,
        repo_type="dataset",
        path_in_repo=subfolder,
        allow_patterns=list(_UPLOAD_EXTENSIONS),
        ignore_patterns=(["tmp_*"] + _stale_checkpoint_ignores(output_path)
                         + _UPLOAD_SKIP_PATTERNS),
    )

    elapsed = time.time() - t0
    print(f"Upload complete: {elapsed:.0f}s → https://huggingface.co/datasets/{HF_OUTPUTS_REPO}")


def download_outputs(output_dir: str, subfolder: str = None):
    """Download outputs from HF to local directory. Prints per-file list + progress.

    Args:
        output_dir: Local destination (e.g., "outputs/full")
        subfolder: HF path prefix (default: basename of output_dir)
    """
    token = _get_token()
    if not token:
        print("SKIP download: HF_TOKEN not found in .env")
        return False

    if not repo_exists(HF_OUTPUTS_REPO, repo_type="dataset", token=token):
        print(f"SKIP download: repo {HF_OUTPUTS_REPO} does not exist")
        return False

    if subfolder is None:
        subfolder = str(Path(output_dir))  # "outputs/full" — mirrors HF layout

    # Pre-list remote files so user sees what's coming
    api = HfApi(token=token)
    remote_files = _list_remote_files(api, subfolder)
    total_bytes = sum(size for _, size in remote_files)
    print(f"Downloading {HF_OUTPUTS_REPO}/{subfolder}/ → {output_dir}")
    print(f"  {len(remote_files)} files ({_fmt_size(total_bytes)}) on HF:")
    for rpath, size in remote_files:
        rel = rpath[len(subfolder):].lstrip("/") if rpath.startswith(subfolder) else rpath
        print(f"    {rel} ({_fmt_size(size)})")

    # Snapshot before download: track what files exist
    output_path = Path(output_dir)
    before = set()
    if output_path.exists():
        before = {str(p.relative_to(output_path)): p.stat().st_size
                  for p in output_path.rglob("*") if p.is_file()}

    t0 = time.time()

    snapshot_download(
        repo_id=HF_OUTPUTS_REPO,
        repo_type="dataset",
        local_dir=".",  # download to project root, HF preserves subfolder structure
        allow_patterns=[f"{subfolder}/*"],
        token=token,
        max_workers=16,
    )

    elapsed = time.time() - t0

    # Diff: what changed
    after = {}
    if output_path.exists():
        after = {str(p.relative_to(output_path)): p.stat().st_size
                 for p in output_path.rglob("*") if p.is_file()}

    new_files = set(after.keys()) - set(before.keys())
    updated_files = {f for f in set(after.keys()) & set(before.keys())
                     if after[f] != before.get(f, 0)}
    unchanged = len(after) - len(new_files) - len(updated_files)

    print(f"\nDownload complete: {elapsed:.0f}s")
    print(f"  Files: {len(new_files)} new, {len(updated_files)} updated, {unchanged} unchanged")
    total_bytes = sum(after[f] for f in new_files | updated_files)
    print(f"  Downloaded: {_fmt_size(total_bytes)}")
    if new_files:
        for f in sorted(new_files)[:10]:
            print(f"  NEW  {f} ({_fmt_size(after[f])})")
        if len(new_files) > 10:
            print(f"  ... and {len(new_files) - 10} more new files")
    if updated_files:
        for f in sorted(updated_files)[:10]:
            print(f"  UPD  {f} ({_fmt_size(after[f])})")
        if len(updated_files) > 10:
            print(f"  ... and {len(updated_files) - 10} more updated files")

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
    token = _get_token()
    if not token:
        print("SKIP: HF_TOKEN not found")
        return

    _ensure_repo(token)
    api = HfApi(token=token)

    uploads = [
        ("data/subset_10k.json", "data/subset_10k.json"),
        ("data/val_1k.json", "data/val_1k.json"),
        ("data/val_500.json", "data/val_500.json"),
        ("data/test_500.json", "data/test_500.json"),
        ("data/sanity_100_dense.json", "data/sanity_100_dense.json"),
        ("data/subset_10k_local", "data/subset_10k_local"),
        ("data/val_1k_local", "data/val_1k_local"),
        ("data/full_local/tags.json", "data/full_local/tags.json"),
        # iter10 Option C: paired-bootstrap eval pool (N=10K, disjoint from train+val).
        ("data/eval_10k.json", "data/eval_10k.json"),
        ("data/eval_10k_local", "data/eval_10k_local"),
    ]

    pbar = make_pbar(total=len(uploads), desc="upload_data", unit="item")
    for local_path, repo_path in uploads:
        p = Path(local_path)
        if not p.exists():
            print(f"  SKIP: {local_path} not found")
            pbar.update(1)
            continue

        if p.is_file():
            print(f"  Uploading {local_path} → {repo_path} ({_fmt_size(p.stat().st_size)})")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=repo_path,
                repo_id=HF_OUTPUTS_REPO,
                repo_type="dataset",
            )
        elif p.is_dir():
            n_files = len(list(p.glob('*')))
            dir_size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
            print(f"  Uploading {local_path}/ → {repo_path}/ ({n_files} files, {_fmt_size(dir_size)})")
            api.upload_folder(
                folder_path=str(p),
                repo_id=HF_OUTPUTS_REPO,
                repo_type="dataset",
                path_in_repo=repo_path,
            )
        pbar.update(1)
    pbar.close()

    print(f"Data upload complete → https://huggingface.co/datasets/{HF_OUTPUTS_REPO}")


def download_data():
    """Download POC + val data from HF to local data/ directory."""
    token = _get_token()
    if not token:
        print("SKIP: HF_TOKEN not found")
        return False

    if not repo_exists(HF_OUTPUTS_REPO, repo_type="dataset", token=token):
        print(f"SKIP: {HF_OUTPUTS_REPO} does not exist")
        return False

    # Pre-list remote data files
    api = HfApi(token=token)
    remote_files = _list_remote_files(api, "data")
    total_bytes = sum(size for _, size in remote_files)
    print(f"Downloading data from {HF_OUTPUTS_REPO}/data/ → data/")
    print(f"  {len(remote_files)} files ({_fmt_size(total_bytes)}) on HF:")
    for rpath, size in remote_files:
        rel = rpath[len("data"):].lstrip("/") if rpath.startswith("data") else rpath
        print(f"    {rel} ({_fmt_size(size)})")

    t0 = time.time()

    snapshot_download(
        repo_id=HF_OUTPUTS_REPO,
        repo_type="dataset",
        local_dir=".",
        allow_patterns=["data/*.json", "data/subset_10k_local/*", "data/val_1k_local/*",
                        "data/full_local/tags.json", "data/eval_10k_local/*"],
        token=token,
        max_workers=16,
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
