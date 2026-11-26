"""
Upload/download compute outputs + POC/val data to HuggingFace Hub.
Repo: anon-org/factorjepa-outputs (public, gated, auto-created on first upload).

USAGE:
    # clean MIRROR [delete file on HF, if doesnt exist on disk]
    HF_UPLOAD_MODE=reuse python -u src/utils/hf_outputs.py upload outputs/poc/ 2>&1 | tee logs/upload_outputs_poc_$(date +%Y%m%d_%H%M%S).log
    # only ADDITIVE (token-safe, no-delete — reads HF_TOKEN from .env). USE THIS, not the bare `hf` CLI below,
    python -u src/utils/hf_outputs.py upload-additive outputs/ 2>&1 | tee logs/upload_large_outputs_pocNfull_$(date +%Y%m%d_%H%M%S).log
    # VERIFY UPLOAD
    python -u src/utils/hf_outputs.py verify outputs/poc 2>&1 | tee logs/verify_outputs_poc_$(date +%Y%m%d_%H%M%S).log

    # resolves the consistency-check crash (plain-HTTP path, no broken validator)
    python -u src/utils/hf_outputs.py download-data outputs/poc 2>&1 | tee logs/download_outputs_poc_$(date +%Y%m%d_%H%M%S).log
    # LIGHT refresh (ALL light artifacts — json/csv/png/pdf/… — EXCLUDING the heavy .pt/.npy/.npz/.tar;
python -u src/utils/hf_outputs.py download-data outputs/ \
--exclude '*.pt' --exclude '*.npy' --exclude '*.npz' --exclude 'tmp_*' --exclude '*.tar' \
2>&1 | tee logs/download_outputs_light_$(date +%Y%m%d_%H%M%S).log

    # Upload/download: from @data/{eval_10k_local/ , full_local/ , subset_10k_local/ , val_1k_local/ }
    python -u src/utils/hf_outputs.py upload-data 2>&1 | tee logs/upload_data_$(date +%Y%m%d_%H%M%S).log
    python -u src/utils/hf_outputs.py upload-data data/subset_10k_local 2>&1 | tee logs/upload_data_subset_10k_$(date +%Y%m%d_%H%M%S).log
    
    python -u src/utils/hf_outputs.py download-data data/eval_10k_local 2>&1 | tee logs/download_data_eval_10k_local_$(date +%Y%m%d_%H%M%S).log
    python -u src/utils/hf_outputs.py download-data data/subset_10k_local 2>&1 | tee logs/download_data_subset_10k_local_$(date +%Y%m%d_%H%M%S).log


"""
import fnmatch
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# iter16 (2026-05-22): env vars MUST be set BEFORE the huggingface_hub import —
# huggingface_hub reads HF_HUB_DISABLE_XET at import time and caches it.
# Xet stays DISABLED for outputs transfers: our 2026-05-22 incident ("File size
# mismatch: expected 14418227 vs downloaded 14446279" on .m04d_checkpoint.npz)
# plus STILL-OPEN silent-corruption reports (huggingface_hub#3643: correct-size
# blobs with wrong SHA-256; xet-core#581: elevated failure rates) — paper-critical
# ckpts ride the slower-but-verified plain HTTP path. Throughput is recovered by
# SHARD-LEVEL parallelism instead: data.max_tar_shard_gb_full caps shards small
# enough that snapshot_download(max_workers=16) fetches them concurrently.
# (iter18 2026-06-06: dropped HF_HUB_ENABLE_HF_TRANSFER=1 — hf_transfer was
# REMOVED in huggingface_hub 1.x; the flag was a silent no-op + FutureWarning.)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# iter18 (2026-06-13): raise the per-request READ timeout (huggingface_hub default = 10 s,
# far too short for a multi-GB .pt over a slow/flaky CDN → httpx.ReadTimeout /
# RemoteProtocolError "peer closed connection" aborts the WHOLE snapshot_download). The
# constant is read at huggingface_hub IMPORT time, so it MUST be set before the import below.
# (gold standard: huggingface_hub #1903 "snapshot_download timeout will not resume", #2822.)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from huggingface_hub import HfApi, repo_exists, snapshot_download
from huggingface_hub.errors import BadRequestError   # iter18 06-07: LFS churn-retry
from utils.progress import make_pbar
from utils.hf_upload_mode import resolve_upload_mode_interactive, delete_repo_folder_scoped

# iter18 (2026-06-13): transient network errors a flaky CDN raises mid-download (huggingface_hub
# 1.x uses httpx; older paths use requests). The download loop RESUMES on these — snapshot_download
# caches completed files + resumes partials — instead of crashing; FAIL LOUD only after the budget.
# httpx.TimeoutException = Read/Connect/PoolTimeout · NetworkError = Connect/ReadError ·
# RemoteProtocolError = "peer closed connection without sending complete message body".
try:
    import httpx as _httpx
    _HTTPX_NET_ERRS = (_httpx.TimeoutException, _httpx.NetworkError, _httpx.RemoteProtocolError)
except ImportError:
    _HTTPX_NET_ERRS = ()
try:
    import requests as _requests
    _RQ_NET_ERRS = (_requests.exceptions.ConnectionError, _requests.exceptions.Timeout,
                    _requests.exceptions.ChunkedEncodingError)
except ImportError:
    _RQ_NET_ERRS = ()
_TRANSIENT_NET_ERRS = _HTTPX_NET_ERRS + _RQ_NET_ERRS + (ConnectionError, TimeoutError)
_DOWNLOAD_ATTEMPTS = 6   # bounded resume-retry for snapshot_download over a flaky CDN

# iter16 (2026-05-20): moved to configs/pipeline.yaml > hf_repos.outputs per
# CLAUDE.md "No hardcoded values in Python". To fork for a new org, change yaml.
from utils.config import get_pipeline_config as _get_pcfg
from utils.data_paths import artifact  # iter18 W4: canonical artifact names (pipeline.yaml)
HF_OUTPUTS_REPO = _get_pcfg()["hf_repos"]["outputs"]   # was "anon-org/factorjepa-outputs" literal

# Upload policy (single source for preview + mirror-cleanup + upload_folder):
# the LIGHT science (json/csv/png/pdf/tex) + ONLY the trained `student_encoder.pt`.
# Heavy + regeneratable/duplicate artifacts are DROPPED — *.npy / *.npz (probe
# features, re-extractable at eval) and every OTHER *.pt (m09*_ckpt_best.pt =
# full resume-state that duplicates the encoder; probe*.pt heads = re-trainable in
# minutes). At vitG this cut outputs/ from ~248 GB → ~tens of GB. NOTE: the glob is
# `*student_encoder.pt` (not the bare filename) so it matches at ANY depth in BOTH
# pathlib.rglob (_list_local_files) AND huggingface_hub fnmatch (allow_patterns).
# iter18 (2026-06-06, user directive "upload everything"): the light set now carries EVERY
# result artifact — all .pt (ckpt_best with its PREDICTOR for stage-8/8b evals, motion_aux
# head, probe heads), .npy/.npz metric arrays, .jsonl logs — so a mid-run box death loses
# nothing a FINISHED run would keep. Still excluded (and why):
#   · _UPLOAD_SKIP_PATTERNS — regenerable caches (m12_frame_cache ~141G, masks, D_L/D_A/D_I)
#   · files modified <120s — _stale_checkpoint_ignores age guard (active writer; the file
#     uploads on the NEXT cycle, so nothing is ever permanently excluded)
# iter18 (2026-06-07, user order): the *ckpt_latest/step/stage* TRANSIENT excludes are
# GONE — the 45-min backup now mirrors EVERY file under outputs/, including the resume
# anchors (~346G at POC peak), so a box migration needs no extra upload. HF dedups
# unchanged files; only the ckpts that actually changed re-transfer each cycle.
_UPLOAD_EXTENSIONS = {"*.json", "*.csv", "*.png", "*.pdf", "*.tex",
                      "*.pt", "*.npy", "*.npz", "*.jsonl"}

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
    "**/m12_frame_cache/**",                          # ~141 GB decoded-frame cache — regeneratable via src/utils/eval_frame_cache.py; NEVER upload
]

_CHECKPOINT_AGE_THRESHOLD = 120  # seconds — skip checkpoints modified within this window

# iter18 W7 (PLR2004): semantic named constants — definitions, not configuration.
_GB, _MB, _KB = 1e9, 1e6, 1e3            # byte-unit boundaries for _fmt_size
_PROC_NET_TX_COL = 10                     # /proc/net/dev: ≥10 fields → col 9 = tx bytes
_RATE_ACTIVE = 1e6                        # B/s — heartbeat "actively moving" floor
_LIST_PREVIEW_N = 10                      # files shown before "... and N more"
_SAMPLE_EDGE_N = 3                        # head/tail items in long-list samples
_MIN_CLI_ARGS = 3                         # `hf_outputs.py <cmd> <dir>`


def _get_token():
    """HF_TOKEN from the environment, else parsed DIRECTLY from the repo-root .env.

    iter18 (2026-06-07): was load_dotenv()-only — python-dotenv is an OPTIONAL dep, so
    under a watch shell whose `python` lacks it, every 45-min auto-backup silently
    printed "HF_TOKEN not found" for 20 h (last real backup 06-06 10:09). The explicit
    parse below works from ANY python/cwd; dotenv (when present) stays as a supplement."""
    if os.getenv("HF_TOKEN"):
        return os.getenv("HF_TOKEN")
    if load_dotenv is not None:
        load_dotenv()
        if os.getenv("HF_TOKEN"):
            return os.getenv("HF_TOKEN")
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN=") or line.startswith("export HF_TOKEN="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                # iter18 2026-06-07: MUST match python-dotenv — a whitespace-preceded inline
                # `# comment` is NOT part of the value. The .env line is
                # `HF_TOKEN=hf_xxx # username: anon-org`; without this strip the fallback
                # returned the 64-char "hf_xxx # username: ..." → HF 401 (Invalid username/password)
                # on every auto-backup whose python lacked dotenv. HF tokens carry no whitespace/#,
                # so the first field after dropping the comment is the token.
                fields = val.split("#", 1)[0].split()
                return fields[0] if fields else None
    return None


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
    if nbytes >= _GB:
        return f"{nbytes / _GB:.1f} GB"
    if nbytes >= _MB:
        return f"{nbytes / _MB:.1f} MB"
    if nbytes >= _KB:
        return f"{nbytes / _KB:.0f} KB"
    return f"{nbytes} B"


def _host_tx_bytes() -> int:
    """Cumulative host network TX bytes (Linux /proc) — a movement proxy for the
    upload, since on a transfer box the dominant outbound traffic IS the upload."""
    try:
        total = 0
        with open("/proc/self/net/dev") as fh:
            for line in fh.readlines()[2:]:          # skip 2 header rows
                parts = line.split()
                if len(parts) >= _PROC_NET_TX_COL:
                    total += int(parts[9])           # column 9 = tx bytes
        return total
    except Exception:
        return 0


def _self_read_bytes() -> int:
    """Cumulative bytes THIS process read from block devices (Linux /proc/self/io).
    Climbs during LFS sha256 HASHING of large files (a pure disk-read phase) even
    while network TX is flat — this is what lets the heartbeat tell 'hashing
    (working)' apart from a genuine stall. (Fully page-cached reads don't hit the
    block device so won't increment, but multi-GB .pt files spill cache → they do.)"""
    try:
        with open("/proc/self/io") as fh:
            for line in fh:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
        return 0
    except Exception:
        return 0


class _UploadHeartbeat:
    """Pulse elapsed + phase every `interval`s while a SILENT huggingface_hub
    upload_folder() runs. It emits no per-file output in a piped/non-TTY context
    (`2>&1 | tee`) → a multi-GB .pt upload looks 'stuck' for 10-20 min across TWO
    phases: (1) LFS HASHING (disk-read-bound, network TX flat) then (2) SENDING
    (network-bound). The heartbeat reports BOTH disk-read and TX rates + infers
    the phase, so neither phase looks frozen. Daemon thread; stops on context exit.
    TX is host-namespace level (movement proxy); read is this-process-exact."""

    def __init__(self, label: str, interval: float = None):
        # iter18 H6: None → pipeline.yaml streaming.heartbeat_interval_s.
        if interval is None:
            interval = _get_pcfg()["streaming"]["heartbeat_interval_s"]
        self.label, self.interval = label, interval
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self._t0 = self._t_last = time.time()
        self._tx0 = self._tx_last = _host_tx_bytes()
        self._rd0 = self._rd_last = _self_read_bytes()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            now = time.time()
            tx, rd = _host_tx_bytes(), _self_read_bytes()
            dt = max(now - self._t_last, 1e-9)
            tx_rate = (tx - self._tx_last) / dt
            rd_rate = (rd - self._rd_last) / dt
            phase = ("SENDING" if tx_rate > _RATE_ACTIVE and tx_rate >= rd_rate
                     else "HASHING" if rd_rate > _RATE_ACTIVE
                     else "waiting")           # neither moving → HF commit/verify or net stall
            print(f"  [{self.label}] {phase} · {now - self._t0:.0f}s elapsed · "
                  f"read +{_fmt_size(int(rd_rate))}/s · sent +{_fmt_size(int(tx_rate))}/s · "
                  f"tot {_fmt_size(rd - self._rd0)} read / {_fmt_size(tx - self._tx0)} sent",
                  flush=True)
            self._tx_last, self._rd_last, self._t_last = tx, rd, now

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=2)
        return False


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
    from huggingface_hub.errors import RemoteEntryNotFoundError
    try:
        # 1. List all files on HF under this subfolder (recursive).
        #    A missing subfolder (fresh repo, or post-purge state where we manually
        #    wiped `outputs/` before re-uploading) raises RemoteEntryNotFoundError
        #    because HF has no folder-exists API — folders only exist implicitly
        #    via files. Semantically that's the same as "no remote files to clean
        #    up", so we early-return and let the upload create the subfolder fresh.
        remote_paths = []
        try:
            for item in api.list_repo_tree(
                    HF_OUTPUTS_REPO, path_in_repo=subfolder, repo_type="dataset",
                    recursive=True):
                if isinstance(item, RepoFile):
                    remote_paths.append(item.path)
        except RemoteEntryNotFoundError:
            print(f"  Mirror: subfolder '{subfolder}' does not exist on HF "
                  f"(fresh repo or post-purge) — skipping cleanup, upload will create it")
            return

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
        # iter18 (2026-06-04): PROTECT `_full-*.tar` full-fidelity snapshots (upload-full) — they are
        # deliberately absent from the light upload set (and deleted locally post-upload), so the
        # light mirror must never purge them. upload_outputs_full does its own scoped remote purge.
        # iter18 (2026-06-06): also protect `_full-manifest.json` — verify-full reads it;
        # a light mirror purge deleting it would silently break A4/B3 verification.
        stale = sorted(p for p in remote_paths
                       if p not in upload_set
                       and not (Path(p).name.startswith("_full-") and p.endswith(".tar"))
                       and Path(p).name != artifact("full_manifest"))

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
                sample = (chunk[:_SAMPLE_EDGE_N] + ["..."] + chunk[-_SAMPLE_EDGE_N:]) if len(chunk) > 2 * _SAMPLE_EDGE_N else chunk
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
    # iter18 (2026-06-06): also age-guard the big *ckpt*.pt files — a ckpt_best promotion
    # writes ~7G over ~30s and must not be tarred/uploaded mid-write.
    cands = list(output_path.rglob(".*checkpoint*")) + list(output_path.rglob("*ckpt*.pt"))
    for ckpt in cands:
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
        # iter18 (2026-06-07): FAIL LOUD — the silent rc-0 skip let the watcher report
        # "✅ backup done" through 20 h of no-token no-ops.
        print("FATAL upload: HF_TOKEN not found (env or repo-root .env)")
        sys.exit(1)

    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"SKIP upload: {output_dir} does not exist")
        return

    # iter18 (2026-06-07): HF platform pre-flight (docs/hub/repositories-recommendations:
    # max 10,000 files/folder · 100,000 files/repo · ~50-100 files/commit recommended).
    # upload_folder = ONE commit; a tree beyond the folder cap fails at the platform wall
    # MID-transfer — FAIL LOUD here instead, pointing at the tar-packing channel
    # (upload-full), which is HF's own recommended strategy for many-file trees.
    _HF_MAX_FILES_PER_FOLDER = 10_000   # platform constant, cited above
    per_folder = {}
    for f in output_path.rglob("*"):
        if f.is_file():
            per_folder[f.parent] = per_folder.get(f.parent, 0) + 1
    n_files = sum(per_folder.values())
    worst_dir, worst_n = max(per_folder.items(), key=lambda kv: kv[1],
                             default=(output_path, 0))
    print(f"  [hf-preflight] {n_files} files · busiest folder = {worst_n} "
          f"({worst_dir}) · HF cap {_HF_MAX_FILES_PER_FOLDER}/folder")
    if worst_n > _HF_MAX_FILES_PER_FOLDER:
        print(f"FATAL upload: {worst_dir} holds {worst_n} files > HF's "
              f"{_HF_MAX_FILES_PER_FOLDER}/folder cap — pack this tree via `upload-full` "
              f"(_full-*.tar shards) instead.")
        sys.exit(1)

    _ensure_repo(token)

    if subfolder is None:
        subfolder = str(output_path)  # "outputs/full", "outputs/poc" — mirrors local layout

    api = HfApi(token=token)

    # iter17 (2026-05-27): ask delete-then-reupload vs reuse (HF_UPLOAD_MODE env
    # bypass for tmux). 'delete' = scoped wipe of the remote subfolder first (clean
    # slate); 'reuse' = the existing mirror-cleanup (remove only stale remote files).
    mode = resolve_upload_mode_interactive(
        api, HF_OUTPUTS_REPO, "dataset", subfolder, f"outputs upload: {output_dir}")
    if mode == "delete":
        delete_repo_folder_scoped(api, HF_OUTPUTS_REPO, "dataset", subfolder)
    else:
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
    print("  Policy (iter18 2026-06-07, user order): EVERY artifact matching "
          f"{sorted(_UPLOAD_EXTENSIONS)} incl. resume ckpts — only *.tmp temps, <120s "
          "active-writer files (re-tried next cycle) and regenerable caches stay out")
    print(f"  Skipping {len(_UPLOAD_SKIP_PATTERNS)} large regeneratable dirs "
          f"(see _UPLOAD_SKIP_PATTERNS in hf_outputs.py): "
          f"{', '.join(p.strip('/').split('/')[-2] for p in _UPLOAD_SKIP_PATTERNS)}")
    print(f"  {len(local_files)} files ({_fmt_size(total_bytes)}):")
    for f in local_files:
        rel = f.relative_to(output_path)
        print(f"    {rel} ({_fmt_size(f.stat().st_size)})")

    t0 = time.time()

    # upload_folder is SILENT in a piped context (no per-file output) → the
    # heartbeat thread pulses elapsed + throughput every 30s so the multi-GB
    # .pt phase doesn't look stuck. Keeps upload_folder's dedup/batching/mirror.
    #
    # iter18 (2026-06-07) LIVE-CHURN RETRY: a training arm finalizing DURING the
    # ~10-min hash phase renames/removes its transients (06-07 06:57: full_ft promoted
    # student_best.pt mid-upload → "is not a file on the local file system" killed the
    # whole 350G backup). upload_folder re-lists the tree on each attempt, so a bounded
    # retry sees the post-finalize stable tree. One churn event per arm → attempt 2
    # virtually always lands; final strike = real error, FAIL LOUD.
    # 06-07 SECOND churn costume: live arms RE-RENDER their plot PNGs every val cycle
    # (atomic replace) → a file modified between hash-upload and commit makes the commit
    # reference an LFS blob that was never uploaded → BadRequestError "LFS pointer
    # pointed to a file that does not exist". Retry is CHEAP: everything already
    # transferred is deduped server-side; only the churned files re-upload.
    _CHURN_ATTEMPTS = 4
    for _attempt in range(1, _CHURN_ATTEMPTS + 1):
        try:
            with _UploadHeartbeat(f"upload {subfolder}"):
                api.upload_folder(
                    folder_path=str(output_path),
                    repo_id=HF_OUTPUTS_REPO,
                    repo_type="dataset",
                    path_in_repo=subfolder,
                    allow_patterns=list(_UPLOAD_EXTENSIONS),
                    # iter18 (2026-06-07, user order): EVERY artifact incl. resume
                    # ckpts — only atomic-write temps, the <120s active-writer guard
                    # (re-tried next cycle), and regenerable caches stay out.
                    ignore_patterns=(["tmp_*", "*.tmp"]
                                     + _stale_checkpoint_ignores(output_path)
                                     + _UPLOAD_SKIP_PATTERNS),
                )
            break
        except (ValueError, BadRequestError) as e:
            churn = ("is not a file" in str(e)          # file DELETED mid-upload (finalize)
                     or "LFS pointer" in str(e))        # file MODIFIED mid-upload (plot re-render)
            if not churn or _attempt == _CHURN_ATTEMPTS:
                raise
            print(f"  [churn-retry {_attempt}/{_CHURN_ATTEMPTS}] live-training churn mid-upload "
                  f"({type(e).__name__}): {e} — re-listing the tree and retrying "
                  f"(already-transferred blobs dedup, retry is cheap)")

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
        print("FATAL download: HF_TOKEN not found (env or repo-root .env)")
        sys.exit(1)

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

    # Snapshot before download: track what files exist. NOTE: must init as dict —
    # `set()` crashed `.keys()` at line ~528 when output_dir didn't pre-exist
    # (fresh box / scratch CWD), i.e. exactly the 4×-box C1 path.
    output_path = Path(output_dir)
    before = {}
    if output_path.exists():
        before = {str(p.relative_to(output_path)): p.stat().st_size
                  for p in output_path.rglob("*") if p.is_file()}

    t0 = time.time()

    # iter18 (2026-06-13): bounded RESUME-retry. snapshot_download caches completed files and
    # resumes partials, so a transient CDN drop (ReadTimeout / RemoteProtocolError on a multi-GB
    # .pt) re-attempts and skips what's already done — instead of crashing the whole pull. The
    # raised HF_HUB_DOWNLOAD_TIMEOUT (above) reduces how often this fires. FAIL LOUD only after
    # the attempt budget. (gold standard: huggingface_hub #1903 / #2822.)
    for _attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            snapshot_download(
                repo_id=HF_OUTPUTS_REPO,
                repo_type="dataset",
                local_dir=".",  # download to project root, HF preserves subfolder structure
                allow_patterns=[f"{subfolder}/*"],
                token=token,
                max_workers=16,
            )
            break
        except _TRANSIENT_NET_ERRS as e:
            if _attempt == _DOWNLOAD_ATTEMPTS:
                print(f"FATAL download: {_DOWNLOAD_ATTEMPTS} attempts exhausted on transient "
                      f"network error ({type(e).__name__}: {str(e)[:160]})")
                raise
            wait = min(60, 5 * _attempt)
            print(f"  [download retry {_attempt}/{_DOWNLOAD_ATTEMPTS}] {type(e).__name__}: "
                  f"{str(e)[:120]} — resuming (done files skipped) in {wait}s…", flush=True)
            time.sleep(wait)

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
        if len(new_files) > _LIST_PREVIEW_N:
            print(f"  ... and {len(new_files) - _LIST_PREVIEW_N} more new files")
    if updated_files:
        for f in sorted(updated_files)[:_LIST_PREVIEW_N]:
            print(f"  UPD  {f} ({_fmt_size(after[f])})")
        if len(updated_files) > _LIST_PREVIEW_N:
            print(f"  ... and {len(updated_files) - _LIST_PREVIEW_N} more updated files")

    # iter18 (2026-06-04): auto-unpack any full-fidelity `_full-*.tar` shards (upload-full) back
    # into their directories. No-op when none were downloaded.
    _post_download_unpack_full(output_path)

    return True


# ── FULL-FIDELITY outputs snapshot (iter18 multi-box migration, 2026-06-04) ──
# Why: the light `upload` policy deliberately drops m09*_ckpt_best.pt / motion_aux_head.pt /
# *.npy — but migrating a half-finished POC to a 2×/4× box needs EVERY file (m09a_ckpt_best.pt
# is every arm's --init-from-ckpt; m09c_ckpt_best.pt feeds eval Stage 8/8b). upload-full packs
# each directory's DIRECT files — ALL extensions, nothing skipped — into `_full-*.tar` shards
# written inside that directory (few large HF objects instead of thousands of small ones), then
# uploads ONLY the shards into the SAME outputs/<mode> prefix. `download` / `download-full` pulls
# them and auto-unpacks (then deletes the tars). Local shards are deleted after upload; the
# original files stay on disk. The light mirror_cleanup PROTECTS remote `_full-*.tar`.
# iter18 H5: template lives in configs/pipeline.yaml (data.full_shard_template);
# the glob is DERIVED from it (single source — no second literal).
_FULL_SHARD_TEMPLATE = _get_pcfg()["data"]["full_shard_template"]
_FULL_SHARD_GLOB = re.sub(r"\{shard:[^}]*\}", "*", _FULL_SHARD_TEMPLATE)


def _pack_outputs_full(output_path: Path, max_shard_gb: float) -> list:
    """Pack EVERY directory's direct files into per-dir `_full-*.tar` shards. Returns shard paths."""
    from utils.tar_shard import pack_dir_to_shards
    # Purge leftovers from a prior run FIRST — otherwise they'd be packed tar-in-tar.
    # iter18 (2026-06-06): ALSO purge stale `_full-manifest.json` files — the 09:51 run
    # packed the PREVIOUS upload's manifest into junk shards at outputs/ + outputs/sanity/
    # (header said 45 files, manifest said 43 — the 2 ghosts). A fresh manifest is
    # rewritten right after packing, so deleting stale ones here is always safe.
    leftovers = (list(output_path.rglob(_FULL_SHARD_GLOB))
                 + list(output_path.rglob(artifact("full_manifest"))))
    if leftovers:
        print(f"  [upload-full] purging {len(leftovers)} leftover shard/manifest file(s) from a prior run")
        for t in leftovers:
            t.unlink()
    shards = []
    dirs = [output_path] + sorted(p for p in output_path.rglob("*") if p.is_dir())
    for d in dirs:
        if not any(f.is_file() for f in d.iterdir()):
            continue
        pack_dir_to_shards(d, str(d / _FULL_SHARD_TEMPLATE), max_shard_gb,
                           keep_source=True, force=True)
        shards.extend(sorted(d.glob(_FULL_SHARD_GLOB)))
    return shards


def _purge_remote_full_shards(api, subfolder: str) -> None:
    """Scoped remote delete of stale `_full-*.tar` under subfolder (shard COUNT can shrink between
    snapshots — a stale _full-00003.tar would otherwise unpack stale files on the next box)."""
    from huggingface_hub.hf_api import RepoFile
    from huggingface_hub.errors import RemoteEntryNotFoundError
    try:
        old = [item.path for item in api.list_repo_tree(
                   HF_OUTPUTS_REPO, path_in_repo=subfolder, repo_type="dataset", recursive=True)
               if isinstance(item, RepoFile)
               and Path(item.path).name.startswith("_full-") and item.path.endswith(".tar")]
    except RemoteEntryNotFoundError:
        return
    if not old:
        return
    print(f"  [upload-full] purging {len(old)} previous remote {_FULL_SHARD_GLOB} (fresh snapshot)")
    api.delete_files(repo_id=HF_OUTPUTS_REPO, delete_patterns=old, repo_type="dataset",
                     commit_message=f"upload-full: purge {len(old)} stale _full shards")


def upload_outputs_full(output_dir: str, subfolder: str = None):
    """FULL-FIDELITY upload — every file under output_dir, no skips (see block comment above)."""
    token = _get_token()
    if not token:
        print("FATAL upload-full: HF_TOKEN not found (env or repo-root .env)")
        sys.exit(1)
    _ensure_repo(token)
    output_path = Path(output_dir)
    if not output_path.is_dir():
        print(f"SKIP upload-full: {output_dir} is not a directory")
        return False
    if subfolder is None:
        subfolder = str(output_path)            # "outputs/poc" — mirrors HF layout
    max_shard_gb = _get_pcfg()["data"]["max_tar_shard_gb_full"]

    n_files = sum(1 for p in output_path.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in output_path.rglob("*") if p.is_file())
    print(f"upload-full: {output_dir} → {HF_OUTPUTS_REPO}/{subfolder}/")
    print(f"  Policy: EVERYTHING — {n_files} files ({_fmt_size(total)}), all extensions, "
          f"packed per-dir into {_FULL_SHARD_GLOB} (cap {max_shard_gb:g} GB/shard)")

    # iter18 (2026-06-08, user order): ASK delete-vs-reuse BEFORE the (slow) pack — same as
    # `upload`/`upload-data`. This gate USED to sit AFTER _pack_outputs_full, so a 225 G tree
    # packed for ~hours before the prompt ever showed (it looked like upload-full never asked).
    #   1=delete → scoped-wipe the remote folder first → what lands on HF is EXACTLY this
    #              snapshot's shards+manifest, no stale strays (incl. loose files from a prior
    #              light `upload`; download's skip-existing unpack would otherwise keep an OLD
    #              copy and throw away the fresh one — the 2026-06-06 dl_test bug);
    #   2=reuse  → refresh the _full-*.tar shards only; other remote files survive.
    # HF_UPLOAD_MODE=delete|reuse bypasses the question for tmux/overnight runs.
    api = HfApi(token=token)
    mode = resolve_upload_mode_interactive(
        api, HF_OUTPUTS_REPO, "dataset", subfolder, f"outputs upload-full: {output_dir}")
    if mode == "delete":
        delete_repo_folder_scoped(api, HF_OUTPUTS_REPO, "dataset", subfolder)
    else:
        _purge_remote_full_shards(api, subfolder)

    shards = _pack_outputs_full(output_path, max_shard_gb)
    shard_bytes = sum(t.stat().st_size for t in shards)
    print(f"  Packed {len(shards)} shard(s) ({_fmt_size(shard_bytes)}) across "
          f"{len({t.parent for t in shards})} dir(s)")

    # Per-snapshot manifest (every packed file + every shard, with sizes) — written at pack
    # time so verify-full can later prove file-by-file that NOTHING was skipped (runbook §0.A7).
    mname = artifact("full_manifest")
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": subfolder,
        "files": [{"path": str(p.relative_to(output_path)), "size": p.stat().st_size}
                  for p in sorted(output_path.rglob("*"))
                  if p.is_file() and not fnmatch.fnmatch(p.name, _FULL_SHARD_GLOB)
                  and p.name != mname],
        "shards": [{"path": str(t.relative_to(output_path)), "size": t.stat().st_size}
                   for t in sorted(shards)],
    }
    (output_path / mname).write_text(json.dumps(manifest, indent=1))
    print(f"  Manifest: {len(manifest['files'])} files + {len(manifest['shards'])} shards "
          f"→ {output_path / mname}")

    t0 = time.time()
    with _UploadHeartbeat(f"upload-full {subfolder}"):
        api.upload_folder(
            folder_path=str(output_path),
            repo_id=HF_OUTPUTS_REPO,
            repo_type="dataset",
            path_in_repo=subfolder,
            allow_patterns=[f"*{_FULL_SHARD_GLOB}", mname],   # leading * → matches at any depth
        )
    print(f"upload-full complete: {time.time() - t0:.0f}s "
          f"→ https://huggingface.co/datasets/{HF_OUTPUTS_REPO}")
    # Reclaim local disk — the ORIGINAL files stay; only the duplicate shards go.
    _delete_shards_after_unpack(shards, f"{_FULL_SHARD_GLOB} (local, post-upload)")
    return True


def _post_download_unpack_full(output_path: Path) -> None:
    """Unpack downloaded `_full-*.tar` shards back into their own directories, then delete them."""
    from utils.tar_shard import unpack_shards_to_dir
    if not output_path.is_dir():
        return
    parents = sorted({t.parent for t in output_path.rglob(_FULL_SHARD_GLOB)})
    for d in parents:
        shards = sorted(d.glob(_FULL_SHARD_GLOB))
        print(f"\n  [hf_outputs] post-download unpack: {d}/{_FULL_SHARD_GLOB} → {d}/")
        unpack_shards_to_dir(shards_glob=str(d / _FULL_SHARD_GLOB), output_dir=d,
                             skip_existing=True)
        _delete_shards_after_unpack(shards, _FULL_SHARD_GLOB)


def verify_outputs(output_dir: str, subfolder: str = None):
    """Verify LOOSE files: every file under HF/<subfolder> is on local disk at the right size (and
    flag local-only extras). This is the disk-vs-HF check for prefixes pushed with plain `upload` /
    `upload-large-folder` / `download-data` — UNLIKE `verify-full`, which checks an `upload-full`
    tar-shard snapshot via `_full-manifest.json` (absent here → that's why verify-full 404s).

    FAIL LOUD (exit 1) if any HF file is missing on disk OR genuinely short. Size note: list_repo_tree's
    size is unreliable for some xet/LFS blobs (the resolve CDN serves a different byte count than the tree
    reports — huggingface_hub#3643), so a disk file LARGER than HF's listed size is the real blob (accepted);
    only a disk file SMALLER than listed (a true truncation) fails.
    """
    token = _get_token()
    if not token:
        print("FATAL verify: HF_TOKEN not found (env or repo-root .env)")
        sys.exit(1)
    if not repo_exists(HF_OUTPUTS_REPO, repo_type="dataset", token=token):
        print(f"FATAL verify: repo {HF_OUTPUTS_REPO} does not exist")
        sys.exit(1)
    output_path = Path(output_dir)
    if subfolder is None:
        subfolder = str(output_path)
    api = HfApi(token=token)
    remote = {(p[len(subfolder) + 1:] if p.startswith(subfolder) else p): s
              for p, s in _list_remote_files(api, subfolder)}
    local = {str(f.relative_to(output_path)): f.stat().st_size
             for f in output_path.rglob("*") if f.is_file() and not f.name.endswith(".part")}
    rset, lset = set(remote), set(local)

    missing_disk = sorted(rset - lset)                               # on HF, NOT on disk → incomplete pull
    extra_disk   = sorted(lset - rset)                              # on disk, NOT on HF → local-only (unbacked)
    short  = sorted(p for p in rset & lset if local[p] < remote[p])   # disk SMALLER than HF → truncated → FAIL
    larger = sorted(p for p in rset & lset if local[p] > remote[p])   # disk LARGER → xet stale metadata → OK
    matched = len(rset & lset) - len(short) - len(larger)

    print(f"verify (loose, disk vs HF): {HF_OUTPUTS_REPO}/{subfolder}  <->  {output_path}/")
    print(f"  HF files   : {len(remote):>5}  ({sum(remote.values()) / _GB:7.1f} GB listed)")
    print(f"  disk files : {len(local):>5}  ({sum(local.values()) / _GB:7.1f} GB)")
    print(f"  matched    : {matched}")
    for label, items in [("on HF but MISSING on disk (incomplete pull)", missing_disk),
                         ("on disk SMALLER than HF (truncated)", short),
                         ("disk LARGER than HF-listed (xet stale size — OK)", larger),
                         ("on disk but NOT on HF (local-only, not backed up)", extra_disk)]:
        if items:
            print(f"  {label}: {len(items)}")
            for p in items[:_LIST_PREVIEW_N]:
                d, h = local.get(p), remote.get(p)
                extra = f"  (disk {d} vs HF {h})" if d is not None and h is not None else ""
                print(f"       - {p}{extra}")
            if len(items) > _LIST_PREVIEW_N:
                print(f"       ... and {len(items) - _LIST_PREVIEW_N} more")

    if missing_disk or short:
        print(f"VERIFY: FAIL - {len(missing_disk)} missing + {len(short)} truncated. "
              f"Re-run `download-data {output_dir}` to fetch/resume them.")
        sys.exit(1)
    print(f"VERIFY: PASS - all {len(remote)} HF files present on disk at full size "
          f"({len(larger)} xet-stale-size accepted, {len(extra_disk)} local-only).")


def verify_outputs_full(output_dir: str, subfolder: str = None):
    """File-by-file proof that the last upload-full snapshot is COMPLETE (runbook §0.A7).

    Checks, FAIL LOUD (sys.exit 1) on any gap:
      1. every local file under output_dir appears in the uploaded manifest, size-matched
         (a file missing here was created/changed AFTER the upload → snapshot is stale);
      2. every shard the manifest promised exists on HF byte-identical
         (a missing/short shard = interrupted transfer).
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.hf_api import RepoFile
    token = _get_token()
    if not token:
        print("FATAL verify-full: HF_TOKEN not found in .env")
        sys.exit(1)
    output_path = Path(output_dir)
    if subfolder is None:
        subfolder = str(output_path)
    mname = artifact("full_manifest")

    mpath = hf_hub_download(HF_OUTPUTS_REPO, f"{subfolder}/{mname}",
                            repo_type="dataset", token=token)
    man = json.loads(Path(mpath).read_text())
    mfiles = {f["path"]: f["size"] for f in man["files"]}
    mshards = {s["path"]: s["size"] for s in man["shards"]}

    local = {str(p.relative_to(output_path)): p.stat().st_size
             for p in output_path.rglob("*")
             if p.is_file() and not fnmatch.fnmatch(p.name, _FULL_SHARD_GLOB)
             and p.name != mname}

    api = HfApi(token=token)
    remote = {item.path[len(subfolder) + 1:]: item.size
              for item in api.list_repo_tree(HF_OUTPUTS_REPO, path_in_repo=subfolder,
                                             repo_type="dataset", recursive=True)
              if isinstance(item, RepoFile)}

    not_uploaded = sorted(set(local) - set(mfiles))
    size_changed = sorted(p for p in set(local) & set(mfiles) if local[p] != mfiles[p])
    gone_local   = sorted(set(mfiles) - set(local))   # uploaded then deleted locally — info only
    shard_missing = sorted(p for p in mshards if p not in remote)
    shard_size_bad = sorted(p for p in mshards if p in remote and remote[p] != mshards[p])

    print(f"verify-full: {output_dir} vs {HF_OUTPUTS_REPO}/{subfolder} "
          f"(manifest {man['created_utc']})")
    print(f"  local files            : {len(local)}")
    print(f"  manifest files         : {len(mfiles)}  · shards: {len(mshards)}")
    print(f"  remote shards found    : {sum(1 for p in mshards if p in remote)}/{len(mshards)}")
    for label, items in [("NOT IN UPLOAD (created/changed after snapshot)", not_uploaded),
                         ("SIZE CHANGED since snapshot", size_changed),
                         ("shard MISSING on HF", shard_missing),
                         ("shard SIZE MISMATCH on HF", shard_size_bad)]:
        if items:
            print(f"  ❌ {label}: {len(items)}")
            for p in items[:_LIST_PREVIEW_N]:
                print(f"       − {p}")
    if gone_local:
        print(f"  · uploaded-but-since-deleted locally (info): {len(gone_local)}")

    if not_uploaded or size_changed or shard_missing or shard_size_bad:
        print("VERIFY-FULL: FAIL — re-run upload-full, then verify-full again.")
        sys.exit(1)
    print(f"VERIFY-FULL: PASS — all {len(local)} local files are in the snapshot; "
          f"all {len(mshards)} shards on HF byte-identical.")


def upload_after_step(output_dir: str):
    """Call at the end of each GPU script to upload new outputs. Non-fatal on failure."""
    try:
        upload_outputs(output_dir)
    except Exception as e:
        print(f"  HF upload failed (non-fatal): {e}")


def _discover_data_uploads(data_root: Path) -> list:
    """Discover everything under data_root that should be uploaded — dynamic.

    iter13 v12+ (2026-05-06): replaces the hardcoded uploads list.
    iter16 (2026-05-22): generalized to discover ANY top-level file extension
    (was *.json only). Without this, invoking `upload-data data/full_local`
    directly silently SKIPPED the 116 × subset-*.tar shards (~120 GB) at the
    data/full_local/ root level — they're neither *.json nor subdirs, so the
    old globs missed them. Only `upload-data` (no arg, parent dir) covered
    them by treating data/full_local/ as a subdir → upload_folder grabbed all
    files inside. The asymmetric behavior was a UX trap. Now ALL top-level
    entries are discovered uniformly — files via upload_file, subdirs via
    upload_folder. Banned ignore_patterns (m10/m11 regenerables) still apply
    inside upload_folder calls per the existing list at upload_data:572-586.

    Rules:
      - every FILE  directly under data_root → upload as file (upload_file)
      - every SUBDIR        under data_root → upload as folder (upload_folder)
      - EXCEPT a top-level dir named "legacy" → SKIPPED (retired artifacts)

    Returns: [(local_path: str, repo_path: str), ...].
    """
    if not data_root.is_dir():
        return []
    pairs: list = []
    for entry in sorted(data_root.iterdir()):
        # iter17 (2026-05-27): never sync retired artifacts to HF. `legacy/` is
        # the retire-don't-delete sink (CLAUDE.md "Retiring files: mv to sibling
        # legacy/ subdir — never rm") — keeping those files LOCAL is the whole
        # point; pushing them re-cruft the dataset (1.6 GB of worker-scratch
        # overlay plots at data/full_local). Skip any top-level dir named "legacy".
        if entry.is_dir() and entry.name == "legacy":
            print(f"  SKIP (retired, not synced to HF): {entry}")
            continue
        if entry.is_file() or entry.is_dir():
            pairs.append((str(entry), str(entry)))
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
        ("m10_sam_segment", "masks", artifact("masks_shard_template")),
        ("m11_factor_datasets", "D_L", artifact("dl_shard_template")),
        ("m11_factor_datasets", "D_A", artifact("da_shard_template")),
        ("m11_factor_datasets", "D_I", artifact("di_shard_template")),
    ]

    def _pack_one_subset(d: Path) -> None:
        """Pack the m10/m11 raw families inside ONE subset dir `d` — i.e. the
        dir that DIRECTLY contains m10_sam_segment/ and m11_factor_datasets/.
        No-op when a family has no raws; pack_dir_to_shards is idempotent."""
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

    # data_root may be EITHER the top-level `data/` (subset dirs are children,
    # m10/m11 are grandchildren) OR a single subset dir like data/full_local
    # (m10/m11 are DIRECT children — the form the iter15_v2 runbook passes).
    # Pack data_root itself AND each child so BOTH granularities work; a given
    # raw dir lives at exactly one level, so there is no double-pack.
    # iter17 FIX (2026-05-27): exact mirror of the _post_download_unpack_masks
    # fix (line ~696). The prior child-only loop silently no-op'd when handed a
    # subset dir directly → `upload-data data/full_local` left masks/ unpacked
    # and upload_folder pushed 114,572 raw .npz individually, hitting HTTP 429 on
    # POST .../objects/verify. See logs/upload_data_full_local_20260527_183618.log.
    _pack_one_subset(data_root)
    for d in sorted(data_root.iterdir()):
        if d.is_dir():
            _pack_one_subset(d)


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

    # iter17 (2026-05-27): ask delete-then-reupload vs reuse BEFORE any (slow) pack.
    # 'delete' scoped-wipes data_root's remote prefix (clean slate); 'reuse' uploads
    # over the existing remote. HF_UPLOAD_MODE env bypass for non-interactive runs.
    mode = resolve_upload_mode_interactive(
        api, HF_OUTPUTS_REPO, "dataset", str(data_root), f"data upload: {data_root}")
    if mode == "delete":
        delete_repo_folder_scoped(api, HF_OUTPUTS_REPO, "dataset", str(data_root))

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
                # iter15 Phase A (2026-05-15): also skip reproducibles regenerated
                # on every run_train.sh launch from eval_10k.json + motion_features
                # (probe_train_subset.py + eval_subset.py). These ride free on the
                # next instance via the bootstrap — no need to occupy HF storage.
                # iter17 FIX (2026-05-27): every entry needs BOTH a top-level
                # (`X/**`) AND a nested (`**/X/**`) form. `upload-data
                # data/full_local` makes each subdir its OWN upload_folder root,
                # so masks/ sits at depth-0 (needs `masks/**`); `upload-data`
                # (data/) makes full_local/ the root, so masks/ sits at depth-1
                # (needs `**/masks/**`). The prior single
                # `**/m10_sam_segment/masks/**` form matched NEITHER depth
                # (verified via huggingface_hub.filter_repo_objects — the leading
                # `**/` requires a segment BEFORE the named dir, but the named dir
                # IS the first relative segment) → the ignore list was dead and
                # only the pack-empties-the-dir mechanism kept raws off HF.
                # masks/D_L/D_A/D_I are packed+deleted above (belt-and-suspenders
                # here); m10_overlay_verify/m11_per_clip_verify are NOT packed —
                # these patterns are the ONLY thing keeping those regenerables out.
                ignore_patterns=[
                    "masks/**", "**/masks/**",
                    "m10_overlay_verify/**", "**/m10_overlay_verify/**",
                    "D_L/**", "**/D_L/**",
                    "D_A/**", "**/D_A/**",
                    "D_I/**", "**/D_I/**",
                    "m11_per_clip_verify/**", "**/m11_per_clip_verify/**",
                    # iter16 M9 (2026-05-21): corpus-agnostic split filenames
                    # (no "eval_10k_" prefix — derived from probe Stage 1).
                    # eval_10k_poc.json + eval_10k_sanity.json retired to
                    # data/eval_10k_local/legacy/ in the same pass.
                    artifact("train_split"), f"**/{artifact('train_split')}",
                    artifact("val_split"), f"**/{artifact('val_split')}",
                    artifact("test_split"), f"**/{artifact('test_split')}",
                    # iter20 (2026-07-19): three data/ subtrees that must NEVER be re-uploaded — each is
                    # either regenerable in minutes or ALREADY published elsewhere, and uploading them
                    # burns the HF 1000-req/5min api quota (10x HTTP 429 on lfs/objects/verify).
                    #   vlm/          TempCompass + LLaVA-Video clips — m18_vlm_data re-downloads them
                    #                 from HF on demand; they are third-party datasets, not our outputs.
                    #   youtube_demo/ third-party YouTube footage — not ours to republish publicly.
                    #   demo_src/     DenseWorld shards — already published as anon-org/denseworld-115k.
                    "vlm/**", "**/vlm/**",
                    "youtube_demo/**", "**/youtube_demo/**",
                    "demo_src/**", "**/demo_src/**",
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

    def _unpack_one_subset(d: Path) -> None:
        """Unpack the m10/m11 tar families inside ONE subset dir `d` — i.e. the
        dir that DIRECTLY contains m10_sam_segment/ and m11_factor_datasets/.
        No-op when neither family has tars; skip_existing makes it idempotent.
        """
        # m10 masks
        seg_dir = d / "m10_sam_segment"
        masks_shards = list(seg_dir.glob(artifact("masks_shard_glob"))) if seg_dir.is_dir() else []
        if masks_shards:
            masks_dir = seg_dir / "masks"
            print(f"\n  [hf_outputs] post-download unpack: {seg_dir}/masks-*.tar → {masks_dir}/")
            unpack_shards_to_dir(
                shards_glob=str(seg_dir / artifact("masks_shard_glob")),
                output_dir=masks_dir,
                skip_existing=True,
            )
            _delete_shards_after_unpack(masks_shards, artifact("masks_shard_glob"))
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

    # data_root may be EITHER the top-level `data/` (subset dirs are children,
    # m10/m11 are grandchildren) OR a single subset dir like data/eval_10k_local
    # (m10/m11 are DIRECT children — the form the iter15_v2 runbook passes).
    # Unpack data_root itself AND each child so BOTH granularities work; a given
    # tar lives at exactly one level, so there is no double-unpack.
    # iter17 FIX (2026-05-26): the prior child-only loop silently no-op'd when
    # handed a subset dir directly — masks/ D_L/ D_A/ D_I/ were left EMPTY while
    # masks-*.tar / D_*-*.tar sat un-extracted alongside. See
    # logs/iter15_v2_data_eval_10k_local_20260526_061931.log (zero unpack lines).
    _unpack_one_subset(data_root)
    for d in sorted(data_root.iterdir()):
        if d.is_dir():
            _unpack_one_subset(d)


def _download_one_file(rpath: str, base_url: str, headers: dict,
                       expected_size: int) -> tuple:
    """Worker: GET one HF file via plain HTTP, stream to disk. Idempotent.

    iter16 (2026-05-22): per-file helper used by `download_data`'s ThreadPool.
    Bypasses huggingface_hub entirely (which has a broken Xet validator for
    some files). Returns (rpath, status, error_or_None) where status is one
    of "skipped", "downloaded", "failed".

    Idempotent skip — iter18 (2026-06-06): the local file must match the REMOTE
    size to be skipped. The old "exists + non-empty" rule silently kept a stale
    or truncated local copy when the remote file changed size (same trap class
    as the dl_test loose-file incident). Size mismatch → re-download.

    iter19 (2026-06-19): bounded retry + HTTP Range RESUME + xet size-metadata tolerance.
    The OLD single-GET path `unlink()`ed the WHOLE partial on any mid-stream drop, so an
    ~18 GB ckpt that dropped at 13 GB threw away all 13 GB (`ChunkedEncodingError:
    IncompleteRead`). Now each file streams to a `<file>.part`, and a TRANSIENT drop (network
    error OR a 429/5xx CDN hiccup) retries up to `_DOWNLOAD_ATTEMPTS` times, each attempt
    RESUMING from the bytes on disk via `Range: bytes=<have>-` (200 = server ignored Range →
    restart; 416 = our offset is past EOF → restart clean), then atomically `os.replace`s the
    `.part` onto the final path.

    Completeness is judged by the SERVER's own `Content-Length`, NOT by list_repo_tree's
    `expected_size`. The list size is UNRELIABLE for xet/LFS blobs — the resolve CDN can serve
    a different byte count than the tree reports (huggingface_hub#3643, xet-core#592). The old
    strict `got == expected_size` gate turned that metadata mismatch into a hard failure that
    killed the whole batch over a throwaway `.npz` whose listed size was stale. Now a clean full
    transfer is ACCEPTED even if its total differs from the listed size (returned as a "downloaded"
    note); only a genuine short read (< the server's Content-Length) retries. Persistent 4xx, or a
    blob the CDN keeps 500-ing on, still fail per-file (no client can fetch bytes the server won't
    serve — that blob must be re-uploaded).
    """
    import requests
    local = Path(rpath)
    if local.exists() and local.stat().st_size == expected_size:
        return (rpath, "skipped", None)
    local.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(local) + ".part")
    url = f"{base_url}/{rpath}"
    last_err = None
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        have = part.stat().st_size if part.exists() else 0
        try:
            req_headers = dict(headers)
            mode = "ab" if have else "wb"
            if have:
                req_headers["Range"] = f"bytes={have}-"
            with requests.get(url, headers=req_headers, stream=True,
                              timeout=600, allow_redirects=True) as r:
                if have and r.status_code == 200:       # server ignored Range → restart from 0
                    have, mode = 0, "wb"
                r.raise_for_status()
                resp_len = r.headers.get("Content-Length")   # bytes THIS response promises
                written = 0
                with open(part, mode) as f:
                    # 32 MB chunks — fewer syscalls per file, ~4× faster on 1 GB+ files.
                    for chunk in r.iter_content(chunk_size=32 << 20):
                        if chunk:
                            f.write(chunk)
                            written += len(chunk)
            if resp_len is not None and written != int(resp_len):   # genuine short read → resume
                last_err = RuntimeError(f"short read {written}/{resp_len} this attempt")
                if attempt < _DOWNLOAD_ATTEMPTS:
                    time.sleep(min(30, 3 * attempt))
                continue
            total = part.stat().st_size
            note = (f"got {total}B but HF listed {expected_size}B — accepted "
                    f"(xet size-metadata mismatch)") if (expected_size and total != expected_size) else None
            os.replace(part, local)                     # atomic publish: server delivered all it promised
            return (rpath, "downloaded", note)
        except _TRANSIENT_NET_ERRS as e:                # keep .part → next attempt resumes from `have`
            last_err = e
            if attempt < _DOWNLOAD_ATTEMPTS:
                time.sleep(min(30, 3 * attempt))
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 416:                             # offset past EOF → drop .part, restart clean
                if part.exists():
                    part.unlink()
                last_err = e
            elif code in (429, 500, 502, 503, 504) and attempt < _DOWNLOAD_ATTEMPTS:
                last_err = e                            # transient server/CDN hiccup → retry (keep .part)
                time.sleep(min(30, 3 * attempt))
            else:                                       # 4xx, or exhausted-5xx → terminal
                if part.exists():
                    part.unlink()
                return (rpath, "failed", f"HTTP {code}: {str(e)[:100]}")
        except Exception as e:                          # truly non-transient → give up now
            if part.exists():
                part.unlink()
            return (rpath, "failed", f"{type(e).__name__}: {str(e)[:120]}")
    return (rpath, "failed", f"{type(last_err).__name__}: {str(last_err)[:100]} "
            f"(after {_DOWNLOAD_ATTEMPTS} attempts)")


def download_data(data_root: Path = None, exts: tuple = None, exclude: tuple = None):
    """Download data_root/ from HF — parallel plain HTTP. Bypasses HF lib.

    iter16 (2026-05-22): rewritten to skip huggingface_hub's download stack
    entirely. The HF lib's consistency-validator + Xet integration is broken
    for some uploaded files (poisoned expected_size metadata) — the validator
    rejects the correct bytes that the HTTP layer just delivered.

    Approach: list remote files via HfApi, then 8-way ThreadPoolExecutor fans
    out `requests.get` per file. Same wire-level approach as `wget` — proven
    to work for the Xet-poisoned checkpoint earlier in this session.
    Single-threaded raw HTTP measured ~8 MB/s on this network (single TCP
    stream + small chunks); 8-way parallel reaches ~50-80 MB/s.

    Idempotent: existing non-empty files at the same path are skipped.
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
    # LIGHT filters — the 8-way plain-HTTP loop below is NON-FATAL per file, so the handful of server-side
    # xet-corrupt blobs that crash `hf download` / snapshot_download (500/416 on a zero-hash cas-bridge URL)
    # are reported 'failed' + skipped instead of aborting the whole batch. Two independently-usable filters:
    #   exts    = ALLOWLIST by extension (only these download)
    #   exclude = DROP by FILENAME glob (everything ELSE downloads) — keeps EVERY light artifact
    #             (png/pdf/tex/txt/…), not just the extensions you remembered to allowlist. Matched on the
    #             basename so 'tmp_*', '*.pt', '*.tar' hit at any depth.
    if exts:
        remote_files = [(p, s) for p, s in remote_files if p.lower().endswith(exts)]
        print(f"  [--ext] allowlist → {len(remote_files)} files matching {exts}")
    if exclude:
        _n0 = len(remote_files)
        remote_files = [(p, s) for p, s in remote_files
                        if not any(fnmatch.fnmatch(Path(p).name, pat) for pat in exclude)]
        print(f"  [--exclude] dropped {_n0 - len(remote_files)} heavy file(s) matching {list(exclude)} "
              f"→ {len(remote_files)} light artifacts")
    total_bytes = sum(size for _, size in remote_files)
    print(f"Downloading {HF_OUTPUTS_REPO}/{subfolder}/ → {data_root}/")
    print(f"  {len(remote_files)} files ({_fmt_size(total_bytes)}) on HF "
          f"— 8-way parallel HTTP GET")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    base_url = f"https://huggingface.co/datasets/{HF_OUTPUTS_REPO}/resolve/main"
    headers = {"Authorization": f"Bearer {token}"}

    t0 = time.time()
    skipped = 0
    downloaded = 0
    fails = []
    pbar = make_pbar(total=len(remote_files), desc="download_data", unit="file")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(_download_one_file, rpath, base_url, headers, expected): rpath
            for rpath, expected in remote_files
        }
        for fut in as_completed(futures):
            rpath, status, err = fut.result()
            if status == "skipped":
                skipped += 1
            elif status == "downloaded":
                downloaded += 1
                if err:                                  # accepted-with-note (xet size mismatch)
                    print(f"  ⚠️  {rpath}: {err}")
            else:
                print(f"  ❌ {rpath}: {err}")
                fails.append((rpath, err))
            pbar.update(1)
    pbar.close()

    # iter13 v13 FIX-25: restore m10 masks/*.npz + m11 D_L/D_A/D_I/*.npy from
    # the four tar-shard families (HF only stored the shards, raws were deleted
    # pre-upload). Discovery is dynamic — no hardcoded subdir list.
    _post_download_unpack_masks(data_root)

    elapsed = time.time() - t0
    print(f"\nData download complete: {elapsed:.0f}s")
    print(f"  skipped (already on disk): {skipped:>5} / {len(remote_files)}")
    print(f"  newly downloaded         : {downloaded:>5} / {len(remote_files)}")
    print(f"  failed                   : {len(fails):>5} / {len(remote_files)}")
    if fails:
        print("\nFailed files (re-run download-data to retry — idempotent):")
        for rpath, err in fails:
            print(f"  ❌ {rpath} — {err}")
        return False
    return True


# ── CLI ──────────────────────────────────────────────────────────────

def upload_additive(subfolder: str):
    """ADDITIVE upload (NEVER deletes remote) of every local file under <subfolder> — the token-safe, in-repo
    replacement for the raw `hf upload-large-folder --include <subfolder>/** --exclude "**/.*"`. Fixes two
    things permanently:
      • AUTH — reads the token from .env via _get_token() and hands it to HfApi, so it can't hit the raw-CLI
        401 (the bare `hf` CLI does NOT read .env; it needs HF_TOKEN already exported / a prior `hf auth login`).
      • SAFETY — unlike `upload` (which MIRRORS: deletes remote files absent locally → catastrophic with a
        LIGHT local copy that has no .pt), this only uploads/updates local files; remote-only heavy checkpoints
        stay untouched.
    folder_path="." + allow_patterns=["<subfolder>/**"] preserves the on-repo path prefix (same as the raw
    `. --include`). Returns True on success, False if the token is missing."""
    token = _get_token()
    if not token:
        print("FATAL upload-additive: HF_TOKEN not found (env or repo-root .env)")
        return False
    sub = str(subfolder).rstrip("/")
    print(f"upload-additive (no-delete, auth from .env): {sub}/**  ->  {HF_OUTPUTS_REPO}")
    HfApi(token=token).upload_large_folder(
        repo_id=HF_OUTPUTS_REPO, repo_type="dataset", folder_path=".",
        # ignore dotfile scratch (.m12f_ckpt / .probe_*) AND the transient per-clip decode dirs. A LIVE
        # eval job creates+deletes tmp_decode_*/tmp*/<clip>.mp4 between the upload's path-scan and its
        # per-file stat() → FileNotFoundError crash (2026-07-09). These are regenerable scratch, never
        # meant for HF, so excluding them both fixes the race AND keeps the upload complete.
        allow_patterns=[f"{sub}/**"],
        # + regenerable DEMO scaffolding — NEVER back these up (they rebuild from the render/model), and
        # uploading them 429s the HF 1000-req/5min api quota. frames_* = per-frame PNG scaffolding
        # (utils/demo_frames rebuilds) · m15_* = m15 decoder feature caches (train_tokens.npz 6 GB +
        # decoder.pt, abandoned track) · m16/clips = source footage (m16 --stage pool re-samples). These
        # paths exist only under outputs/demo, so poc/full uploads are unaffected. ~16 GB + ~640 LFS objects cut.
        # pov_ood/ holds THIRD-PARTY YouTube footage used as out-of-domain probe material. It is NOT ours to
        # redistribute — excluded so a backup never republishes copyrighted video to the public HF repo.
        ignore_patterns=["**/.*", "**/tmp_decode_*/**", "**/tmp_decode_*",
                         "**/frames_*/**", "**/frames_*",
                         "**/m15_*/**", "**/m16/clips/**",
                         "**/pov_ood/**", "**/youtube_demo/**"],
    )
    print(f"upload-additive complete: {sub}/** committed additively (remote-only files preserved)")
    return True


def delete_remote(patterns):
    """DELETE remote files matching glob PATTERNS from the HF outputs repo in ONE batched commit
    (api.delete_files → create_commit; a handful of api calls, fits the rate limit). Use to prune files an
    earlier upload pushed BEFORE they were added to ignore_patterns — e.g. the frames_* PNG scaffolding a
    stalled upload_large_folder run committed. Previews the match count first (list_repo_files + fnmatch, same
    matching HF uses) so you see exactly what goes. Token-safe (auth from .env). Returns True on success.

    e.g.  python -u src/utils/hf_outputs.py delete "**/frames_*/**" "**/m15_*/**" "**/m16/clips/**"
    """
    import fnmatch
    token = _get_token()
    if not token:
        print("FATAL delete: HF_TOKEN not found (env or repo-root .env)")
        return False
    pats = list(patterns)
    if not pats:
        print("FATAL delete: no patterns given (e.g. \"**/frames_*/**\")")
        return False
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=HF_OUTPUTS_REPO, repo_type="dataset")
    matched = sorted(f for f in files if any(fnmatch.fnmatch(f, p) for p in pats))
    print(f"delete: {len(matched)}/{len(files)} remote files match {pats}")
    for f in matched[:6]:
        print(f"   - {f}")
    if len(matched) > 6:
        print(f"   ... +{len(matched) - 6} more")
    if not matched:
        print("nothing to delete (no remote files match)")
        return True
    api.delete_files(repo_id=HF_OUTPUTS_REPO, repo_type="dataset", delete_patterns=pats)
    print(f"delete complete: {len(matched)} remote files pruned in one commit")
    return True


def squash_history():
    """Collapse the HF dataset repo's ENTIRE commit history into ONE commit (super_squash_history). HF warns
    once a repo passes a few hundred commits that push/list operations slow down (this repo hit 848). The op is
    IRREVERSIBLE — it rewrites remote history, so old commit SHAs are gone — BUT it keeps the CURRENT file tree
    byte-for-byte, so no DATA is lost, only the history. Only run it when the tree is ALREADY fully uploaded
    (e.g. right before teardown, after upload-additive reports committed N/N). Returns True on success, False if
    the token is missing."""
    token = _get_token()
    if not token:
        print("FATAL squash: HF_TOKEN not found (env or repo-root .env)")
        return False
    print(f"super_squash_history: collapsing ALL commits of {HF_OUTPUTS_REPO} into one "
          f"(IRREVERSIBLE — old SHAs gone, current file tree preserved byte-for-byte)")
    HfApi(token=token).super_squash_history(repo_id=HF_OUTPUTS_REPO, repo_type="dataset")
    print(f"squash complete: {HF_OUTPUTS_REPO} history collapsed to a single commit (file tree unchanged)")
    return True


if __name__ == "__main__":
    if len(sys.argv) < _MIN_CLI_ARGS - 1:
        print("Usage:")
        print("  python -u src/utils/hf_outputs.py upload <output_dir>")
        print("  python -u src/utils/hf_outputs.py upload-additive <output_dir>  # ADDITIVE no-delete + auth from .env (safe for a LIGHT local copy; use instead of bare `hf`)")
        print("  python -u src/utils/hf_outputs.py delete <glob>...              # prune remote files by glob (e.g. \"**/frames_*/**\"); previews then deletes in ONE commit")
        print("  python -u src/utils/hf_outputs.py download <output_dir>")
        print("  python -u src/utils/hf_outputs.py upload-full <output_dir>    # EVERY file (ckpts/npy too), per-dir _full-*.tar")
        print("  python -u src/utils/hf_outputs.py download-full <output_dir>  # pulls + auto-unpacks _full-*.tar")
        print("  python -u src/utils/hf_outputs.py upload-data [data_dir]      # default 'data'")
        print("  python -u src/utils/hf_outputs.py download-data [data_dir]    # default 'data'")
        print("  python -u src/utils/hf_outputs.py verify <output_dir>         # LOOSE files: disk vs HF (path+size)")
        print("  python -u src/utils/hf_outputs.py squash                       # collapse HF repo history to 1 commit (IRREVERSIBLE; run ONLY after a complete upload)")
        sys.exit(1)

    cmd = sys.argv[1]

    # iter18 (2026-06-06): FAIL LOUD at the process level — every verb's False
    # return becomes exit 1. Before this, `download-data` could print "failed: 3"
    # and still exit 0, letting a `| tee` runbook chain sail past missing files.
    rc = None
    if cmd == "upload" and len(sys.argv) >= _MIN_CLI_ARGS:
        rc = upload_outputs(sys.argv[2])
    elif cmd == "upload-additive" and len(sys.argv) >= _MIN_CLI_ARGS:
        rc = upload_additive(sys.argv[2])   # token-safe, no-delete (the raw-`hf` 401 fix)
    elif cmd == "delete" and len(sys.argv) >= _MIN_CLI_ARGS:
        rc = delete_remote(sys.argv[2:])    # prune remote files by glob (e.g. frames_* pushed pre-exclude)
    elif cmd == "download" and len(sys.argv) >= _MIN_CLI_ARGS:
        rc = download_outputs(sys.argv[2])
    elif cmd == "upload-full" and len(sys.argv) >= _MIN_CLI_ARGS:
        rc = upload_outputs_full(sys.argv[2])
    elif cmd == "download-full" and len(sys.argv) >= _MIN_CLI_ARGS:
        rc = download_outputs(sys.argv[2])   # same prefix — download_outputs auto-unpacks _full shards
    elif cmd == "squash":
        rc = squash_history()                # collapse HF dataset commit history to 1 commit (IRREVERSIBLE, tree preserved)
    elif cmd == "verify" and len(sys.argv) >= _MIN_CLI_ARGS:
        verify_outputs(sys.argv[2])          # LOOSE files disk-vs-HF; exits 1 itself on any gap
    elif cmd == "verify-full" and len(sys.argv) >= _MIN_CLI_ARGS:
        verify_outputs_full(sys.argv[2])     # exits 1 itself on any gap
    elif cmd == "upload-data":
        # iter13 v12+ (2026-05-06): optional positional <data_dir> override.
        # Default = "data" (project convention). No more hardcoded subfolder list.
        rc = upload_data(Path(sys.argv[2]) if len(sys.argv) >= _MIN_CLI_ARGS else None)
    elif cmd == "download-data":
        # optional LIGHT filters (plain-HTTP path is non-fatal per file, so a few server-side xet-corrupt
        # blobs are reported + skipped, not crashing the batch like `hf download`):
        #   --ext json,csv,png        ALLOWLIST by extension
        #   --exclude '*.pt' '*.npy'  DROP by filename glob (everything else downloads) — repeatable,
        #                             also accepts --exclude=PAT and a comma-joined --exclude a,b,c
        _exts = None
        _exclude = []
        _pos = []
        _it = iter(sys.argv[2:])
        for _tok in _it:
            if _tok == "--ext":
                _exts = tuple("." + e.strip().lstrip(".").lower() for e in next(_it, "").split(",") if e.strip())
            elif _tok.startswith("--ext="):
                _exts = tuple("." + e.strip().lstrip(".").lower()
                              for e in _tok.split("=", 1)[1].split(",") if e.strip())
            elif _tok == "--exclude":
                _exclude += [g.strip() for g in next(_it, "").split(",") if g.strip()]
            elif _tok.startswith("--exclude="):
                _exclude += [g.strip() for g in _tok.split("=", 1)[1].split(",") if g.strip()]
            else:
                _pos.append(_tok)
        rc = download_data(Path(_pos[0]) if _pos else None,
                           exts=(_exts or None), exclude=(tuple(_exclude) or None))
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
    if rc is False:
        sys.exit(1)
