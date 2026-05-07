"""
Upload/download compute outputs + POC/val data to HuggingFace Hub.
Repo: anonymousML123/factorjepa-outputs (public, gated, auto-created on first upload).

USAGE:
    # FULL/POC/SANITY usage: from outputs/
    python -u src/utils/hf_outputs.py upload outputs 2>&1 | tee logs/hf_upload_outputs.log
    python -u src/utils/hf_outputs.py download outputs 2>&1 | tee logs/hf_download_outputs.log

    # Upload/download: from outputs/full/ ONLY
    python -u src/utils/hf_outputs.py upload outputs/full 2>&1 | tee logs/hf_upload.log
    HF_HUB_ENABLE_HF_TRANSFER=1 python -u src/utils/hf_outputs.py upload outputs/full  2>&1 | tee logs/hf_upload_outputs_full.log
    python -u src/utils/hf_outputs.py download outputs/full 2>&1 | tee logs/hf_download.log

    # Upload/download: from @data/{eval_10k_local/ , full_local/ , subset_10k_local/ , val_1k_local/ }
    python -u src/utils/hf_outputs.py upload-data 2>&1 | tee logs/upload_poc_val.log    # ~15 min upload
    python -u src/utils/hf_outputs.py download-data 2>&1 | tee logs/download_poc_val_v1.log # ~3 min measured



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


def _discover_data_uploads(data_root: Path) -> list:
    """Discover everything under data_root that should be uploaded — dynamic.

    iter13 v12+ (2026-05-06): replaces the hardcoded uploads list.
    Rules (NO hardcoded subfolder names):
      - every *.json directly under data_root             → upload as file
      - every subdirectory of data_root                   → upload as folder (bundle)

    Returns: [(local_path: str, repo_path: str), ...].
    """
    if not data_root.is_dir():
        return []
    pairs: list = []
    for f in sorted(data_root.glob("*.json")):
        pairs.append((str(f), str(f)))
    for d in sorted(data_root.iterdir()):
        if d.is_dir():
            pairs.append((str(d), str(d)))
    return pairs


def _pre_upload_pack_outputs(data_root: Path) -> None:
    """Pack m10/m11 raw per-clip outputs → TAR shards before HF upload.

    iter13 v13 FIX-25 (2026-05-07): single source of truth for tar packing.
    Compute scripts (m10, m11) produce raw .npz / .npy only. This helper
    converts them into HF-uploadable tar shards (HF 10k-file repo cap) and
    DELETES the raw sources (`keep_source=False`) so disk stays clean.

    Packs four shard families per local-data subdir (dynamic discovery — NO
    hardcoded list of subdir names):
      <subdir>/m10_sam_segment/masks/*.npz       → masks-{shard:05d}.tar
      <subdir>/m11_factor_datasets/D_L/*.npy     → D_L-{shard:05d}.tar
      <subdir>/m11_factor_datasets/D_A/*.npy     → D_A-{shard:05d}.tar
      <subdir>/m11_factor_datasets/D_I/*.npy     → D_I-{shard:05d}.tar

    Size-driven (m00d-style) — rolls a new shard whenever adding the next file
    would exceed `pipeline.yaml.data.max_tar_shard_gb`. Auto-scales 10K → 115K.

    NOTE on cleanup: raw files are deleted after a successful pack. If m09c
    needs them on the same machine after this runs, re-download via HF (the
    download path auto-unpacks via _post_download_unpack_masks).
    """
    from utils.tar_shard import pack_dir_to_shards
    from utils.config import get_pipeline_config
    if not data_root.is_dir():
        return
    max_shard_gb = get_pipeline_config()["data"]["max_tar_shard_gb"]

    # (raw_subdir, shard_template_relative_to_parent) — packed in this order so
    # that m10's masks/ goes first (m11 already finished, doesn't need them).
    pack_specs = [
        ("m10_sam_segment", "masks", "masks-{shard:05d}.tar"),
        ("m11_factor_datasets", "D_L", "D_L-{shard:05d}.tar"),
        ("m11_factor_datasets", "D_A", "D_A-{shard:05d}.tar"),
        ("m11_factor_datasets", "D_I", "D_I-{shard:05d}.tar"),
    ]

    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        for parent_name, raw_subdir, shard_pattern in pack_specs:
            raw_dir = d / parent_name / raw_subdir
            if not raw_dir.is_dir() or not any(raw_dir.iterdir()):
                continue
            shard_template = str(d / parent_name / shard_pattern)
            print(f"\n  [hf_outputs] pre-upload pack: {raw_dir} "
                  f"(cap={max_shard_gb:.2f} GB/shard, raws DELETED after pack)")
            pack_dir_to_shards(
                input_dir=raw_dir,
                shard_template=shard_template,
                max_shard_size_gb=max_shard_gb,
                keep_source=False,   # FIX-25: clean disk after pack
                force=False,
            )


def upload_data(data_root: Path = None):
    """Upload data_root/ contents to HF — files via discovery, no hardcoded subfolder list.

    iter13 v12+ (2026-05-06): dynamically iterates data_root for *.json files +
    subdir bundles. m10 masks/ subfolders get pre-packed into TAR shards
    (HF 10k-file cap workaround). m10/m11 outputs are CO-LOCATED inside each
    local-data subdir (data_root/<subdir>/m10_sam_segment/, m11_factor_datasets/)
    so they ride along automatically.

    Args:
      data_root: directory to upload. Defaults to "data" (project convention).
    """
    if data_root is None:
        data_root = Path("data")
    data_root = Path(data_root)

    token = _get_token()
    if not token:
        print("SKIP: HF_TOKEN not found")
        return

    _ensure_repo(token)
    api = HfApi(token=token)

    # Pre-upload: TAR-shard m10 masks + m11 D_L/D_A/D_I raws across every
    # <subdir>/ under data_root (HF 10k-file repo cap). Dynamic discovery — no
    # hardcoded subdir list. iter13 v13 FIX-25 (2026-05-07): packs ALL FOUR
    # raw dirs and DELETES sources (`keep_source=False`) so post-upload disk
    # contains only tars. m09c on the same machine would need to either read
    # tars or re-download from HF (auto-unpacked via _post_download_unpack_masks).
    _pre_upload_pack_outputs(data_root)

    uploads = _discover_data_uploads(data_root)
    if not uploads:
        print(f"SKIP: no *.json or subdirs found under {data_root}/")
        return
    print(f"Discovered {len(uploads)} upload items under {data_root}/")

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
                # iter13 v13 FIX-25 (2026-05-07): skip raw m10/m11 per-clip files
                # (already packed into tars by _pre_upload_pack_outputs above —
                # raw dirs are empty post-pack, but glob-skip is still cheaper
                # than walking them). Subset-*.tar inputs ride along as-is.
                ignore_patterns=[
                    "**/m10_sam_segment/masks/**",
                    "**/m10_sam_segment/m10_overlay_verify/**",
                    "**/m11_factor_datasets/D_L/**",
                    "**/m11_factor_datasets/D_A/**",
                    "**/m11_factor_datasets/D_I/**",
                    "**/m11_factor_datasets/m11_per_clip_verify/**",
                ],
            )
        pbar.update(1)
    pbar.close()

    print(f"Data upload complete → https://huggingface.co/datasets/{HF_OUTPUTS_REPO}")


def _delete_shards_after_unpack(shards: list, label: str) -> None:
    """Reclaim disk by deleting tar shards after a successful unpack.

    iter13 v13 FIX-28 (2026-05-08): post-download `data/` dir was holding BOTH
    the unpacked raws AND the source tars (~10 GB redundancy at FULL scale).
    Now: tars get deleted as soon as their unpack returns successfully. Tars
    are regenerable via re-download from HF, so deletion is safe.
    """
    if not shards:
        return
    freed_bytes = sum(t.stat().st_size for t in shards if t.is_file())
    n_deleted = 0
    for tar_path in shards:
        if tar_path.is_file():
            tar_path.unlink()
            n_deleted += 1
    print(f"  [hf_outputs] cleaned up {n_deleted} {label} "
          f"({freed_bytes / 1e9:.2f} GB freed)")


def _post_download_unpack_masks(data_root: Path) -> None:
    """Unpack m10/m11 TAR shards back into per-clip files after HF download.

    iter13 v13 FIX-25 (2026-05-07): mirror of _pre_upload_pack_outputs.
    Discovers shards via Path.iterdir() — NO hardcoded subdir list — and
    extracts them back into the dirs that m10/m11 read from at runtime:
      <subdir>/m10_sam_segment/masks-*.tar      → masks/*.npz
      <subdir>/m11_factor_datasets/D_L-*.tar    → D_L/*.npy
      <subdir>/m11_factor_datasets/D_A-*.tar    → D_A/*.npy
      <subdir>/m11_factor_datasets/D_I-*.tar    → D_I/*.npy
    Skips already-extracted files to allow incremental restore.

    iter13 v13 FIX-28 (2026-05-08): after each shard family unpacks
    successfully, delete its tars to reclaim disk (~10 GB at FULL scale).
    Tars are regenerable via re-download — safe to drop. Failures during
    unpack raise out of `unpack_shards_to_dir` BEFORE the deletion call,
    so tars are preserved on error (defensive — no data loss on partial unpack).
    """
    from utils.tar_shard import unpack_shards_to_dir
    if not data_root.is_dir():
        return
    for d in sorted(data_root.iterdir()):
        if not d.is_dir():
            continue
        # m10 masks
        seg_dir = d / "m10_sam_segment"
        masks_shards = list(seg_dir.glob("masks-*.tar")) if seg_dir.is_dir() else []
        if masks_shards:
            masks_dir = seg_dir / "masks"
            print(f"\n  [hf_outputs] post-download unpack: {seg_dir}/masks-*.tar → {masks_dir}/")
            unpack_shards_to_dir(
                shards_glob=str(seg_dir / "masks-*.tar"),
                output_dir=masks_dir,
                skip_existing=True,
            )
            _delete_shards_after_unpack(masks_shards, "masks-*.tar")
        # m11 factor shards (D_L / D_A / D_I)
        m11_dir = d / "m11_factor_datasets"
        if m11_dir.is_dir():
            for factor in ("D_L", "D_A", "D_I"):
                shards = list(m11_dir.glob(f"{factor}-*.tar"))
                if not shards:
                    continue
                out_dir = m11_dir / factor
                print(f"\n  [hf_outputs] post-download unpack: "
                      f"{m11_dir}/{factor}-*.tar → {out_dir}/")
                unpack_shards_to_dir(
                    shards_glob=str(m11_dir / f"{factor}-*.tar"),
                    output_dir=out_dir,
                    skip_existing=True,
                )
                _delete_shards_after_unpack(shards, f"{factor}-*.tar")


def download_data(data_root: Path = None):
    """Download data_root/ from HF — dynamic, no hardcoded subfolder allow-list.

    iter13 v12+ (2026-05-06): pulls everything under <repo>/data_root/, then
    unpacks any m10 masks-*.tar shards back into masks/ (for runtime random
    access by m11 + m09c).
    """
    if data_root is None:
        data_root = Path("data")
    data_root = Path(data_root)

    token = _get_token()
    if not token:
        print("SKIP: HF_TOKEN not found")
        return False

    if not repo_exists(HF_OUTPUTS_REPO, repo_type="dataset", token=token):
        print(f"SKIP: {HF_OUTPUTS_REPO} does not exist")
        return False

    subfolder = str(data_root)
    api = HfApi(token=token)
    remote_files = _list_remote_files(api, subfolder)
    total_bytes = sum(size for _, size in remote_files)
    print(f"Downloading {HF_OUTPUTS_REPO}/{subfolder}/ → {data_root}/")
    print(f"  {len(remote_files)} files ({_fmt_size(total_bytes)}) on HF:")
    for rpath, size in remote_files:
        rel = rpath[len(subfolder):].lstrip("/") if rpath.startswith(subfolder) else rpath
        print(f"    {rel} ({_fmt_size(size)})")

    t0 = time.time()

    snapshot_download(
        repo_id=HF_OUTPUTS_REPO,
        repo_type="dataset",
        local_dir=".",
        allow_patterns=[f"{subfolder}/*"],
        token=token,
        max_workers=16,
    )

    # iter13 v13 FIX-25: restore m10 masks/*.npz + m11 D_L/D_A/D_I/*.npy from
    # the four tar-shard families (HF only stored the shards, raws were deleted
    # pre-upload). Discovery is dynamic — no hardcoded subdir list.
    _post_download_unpack_masks(data_root)

    print(f"Data download complete: {time.time() - t0:.0f}s")
    return True


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -u src/utils/hf_outputs.py upload <output_dir>")
        print("  python -u src/utils/hf_outputs.py download <output_dir>")
        print("  python -u src/utils/hf_outputs.py upload-data [data_dir]      # default 'data'")
        print("  python -u src/utils/hf_outputs.py download-data [data_dir]    # default 'data'")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "upload" and len(sys.argv) >= 3:
        upload_outputs(sys.argv[2])
    elif cmd == "download" and len(sys.argv) >= 3:
        download_outputs(sys.argv[2])
    elif cmd == "upload-data":
        # iter13 v12+ (2026-05-06): optional positional <data_dir> override.
        # Default = "data" (project convention). No more hardcoded subfolder list.
        upload_data(Path(sys.argv[2]) if len(sys.argv) >= 3 else None)
    elif cmd == "download-data":
        download_data(Path(sys.argv[2]) if len(sys.argv) >= 3 else None)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
