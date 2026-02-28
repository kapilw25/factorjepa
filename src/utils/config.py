"""
Common configuration for WalkIndia-200k pipeline.
"""
import os
import re
import sys
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
UTILS_DIR = SRC_DIR / "utils"
DATA_DIR = SRC_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
CLIPS_DIR = DATA_DIR / "clips"
SHARDS_DIR = DATA_DIR / "shards"
OUTPUTS_DIR = SRC_DIR / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Input data (canonical copies in src/utils/)
YT_VIDEOS_JSON = UTILS_DIR / "YT_videos_raw.json"
TAG_TAXONOMY_JSON = UTILS_DIR / "tag_taxonomy.json"

# Ensure directories exist
for d in [VIDEOS_DIR, CLIPS_DIR, SHARDS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# HuggingFace dataset config
HF_DATASET_REPO = "anonymousML123/walkindia-200k"

# HuggingFace private split repos (50GB limit per repo)
HF_DATASET_REPO_PART1 = "anonymousML123/walkindia-200k-part1"
HF_DATASET_REPO_PART2 = "anonymousML123/walkindia-200k-part2"
HF_DATASET_REPOS = [HF_DATASET_REPO_PART1, HF_DATASET_REPO_PART2]
HF_REPO_SIZE_LIMIT_GB = 300

# Re-encoding config (CRF 28 on 480p shrinks ~40-50%)
REENCODE_CRF = 28


def ensure_clips_exist() -> bool:
    """
    Check if clips exist locally. If not, download from HuggingFace (both split repos).
    Returns True if clips are available, False otherwise.
    """
    # Check if clips already exist (recursive search for hierarchical structure)
    clip_count = len(list(CLIPS_DIR.rglob("*.mp4"))) if CLIPS_DIR.exists() else 0

    if clip_count > 0:
        print(f"Found {clip_count} clips locally in {CLIPS_DIR}")
        return True

    # Auto-download from HuggingFace (try split repos, then fallback to single repo)
    repos_to_try = HF_DATASET_REPOS + [HF_DATASET_REPO]

    try:
        import shutil
        from huggingface_hub import snapshot_download

        for repo in repos_to_try:
            print(f"Downloading from HuggingFace: {repo}")
            try:
                local_dir = snapshot_download(
                    repo_id=repo,
                    repo_type="dataset",
                    local_dir=DATA_DIR / "hf_download",
                    allow_patterns=["clips/**/*.mp4"],
                )

                # Mirror entire clips tree (hierarchical structure)
                hf_clips_dir = Path(local_dir) / "clips"
                if hf_clips_dir.exists():
                    shutil.copytree(hf_clips_dir, CLIPS_DIR, dirs_exist_ok=True)

                # Cleanup temp download
                shutil.rmtree(DATA_DIR / "hf_download", ignore_errors=True)
                print(f"  Downloaded clips from {repo}")

            except Exception as e:
                print(f"  Skipping {repo}: {e}")
                continue

        # Verify
        clip_count = len(list(CLIPS_DIR.rglob("*.mp4")))
        print(f"Total clips available: {clip_count}")
        return clip_count > 0

    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False

# PySceneDetect config
CLIP_MIN_DURATION = 4.0  # seconds
CLIP_MAX_DURATION = 10.0  # seconds (professor spec: min 4, max 10)

# V-JEPA config (ViT-G 384: 1B params, strongest V-JEPA variant for best embeddings)
VJEPA_MODEL_ID = "facebook/vjepa2-vitg-fpc64-384"
VJEPA_FRAMES_PER_CLIP = 64
VJEPA_EMBEDDING_DIM = 1408  # ViT-G hidden dimension

# Qwen3-VL config
QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# FAISS config
FAISS_K_NEIGHBORS = 6  # includes self

# POC subset config
SUBSET_FILE = PROJECT_ROOT / "data" / "subset_10k.json"
OUTPUTS_POC_DIR = SRC_DIR / "outputs_poc"
BAKEOFF_DIR = DATA_DIR / "bakeoff"

# VLM bake-off config
VLM_MODELS = {
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    "videollama": "DAMO-NLP-SG/VideoLLaMA3-7B",
    "keye": "Keye-VL/Keye-VL-1.5-8B-Chat",
}
BAKEOFF_CLIP_COUNT = 2500

# Output files
EMBEDDINGS_FILE = OUTPUTS_DIR / "embeddings.npy"
TAGS_FILE = OUTPUTS_DIR / "tags.json"
UMAP_PLOT_PNG = OUTPUTS_DIR / "m08_umap.png"
UMAP_PLOT_PDF = OUTPUTS_DIR / "m08_umap.pdf"
METRICS_FILE = OUTPUTS_DIR / "m06_metrics.json"


# Ensure POC directories exist
for d in [OUTPUTS_POC_DIR, BAKEOFF_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SUBSET / POC UTILITIES
# =============================================================================

def load_subset(subset_path: str = None) -> set:
    """
    Load subset clip keys from JSON file.

    Args:
        subset_path: Path to subset JSON (e.g., data/subset_10k.json).
                     If None, returns empty set (= no filtering, full mode).

    Returns:
        Set of clip keys (e.g., {"goa/walking/04YKvC8kAgI/04YKvC8kAgI-000.mp4", ...})
        Empty set means no filtering (full mode).
    """
    if subset_path is None:
        return set()

    import json
    p = Path(subset_path)
    if not p.exists():
        print(f"ERROR: Subset file not found: {p}")
        sys.exit(1)

    with open(p) as f:
        data = json.load(f)

    keys = set(data["clip_keys"])
    print(f"[POC] Loaded subset: {len(keys):,} clip keys from {p.name}")
    return keys


def get_output_dir(subset_path: str = None) -> Path:
    """
    Return output directory based on mode.
    POC mode (--subset) → outputs_poc/
    Full mode           → outputs/
    """
    if subset_path:
        return OUTPUTS_POC_DIR
    return OUTPUTS_DIR


def add_subset_arg(parser):
    """Add --subset argument to any argparse parser (shared across m04-m08)."""
    parser.add_argument("--subset", type=str, default=None,
                        help="Path to subset JSON (e.g., data/subset_10k.json) for POC mode")


# =============================================================================
# UTILITY FUNCTIONS (avoid redundancy across modules)
# =============================================================================

def check_gpu():
    """Check if CUDA GPU is available. Exit if not."""
    try:
        import torch
    except ImportError:
        print("ERROR: torch not installed")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU not available.")
        print("This script requires Nvidia GPU. No CPU fallback.")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")


def get_all_clips() -> list:
    """
    Get all video clips from CLIPS_DIR (recursive).
    Returns list of Path objects.
    """
    if not CLIPS_DIR.exists():
        return []
    return sorted(CLIPS_DIR.rglob("*.mp4"))


def get_all_videos() -> list:
    """
    Get all videos from VIDEOS_DIR.
    Returns list of Path objects.
    """
    return list(VIDEOS_DIR.glob("*.mp4"))


def get_video_duration(video_path) -> float:
    """Get video duration in seconds using ffprobe."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def sanitize_name(name: str) -> str:
    """Sanitize name for use as directory name (lowercase, underscores)."""
    name = name.lower().strip()
    name = re.sub(r'[,\s]+', '_', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    name = re.sub(r'_+', '_', name)
    return name.strip('_')


def build_video_section_map() -> dict:
    """Build video_id → section path mapping from YT_videos_raw.json.

    Returns dict like: {"qABnYGIilHE": "tier1/bangalore/drive", ...}
    """
    import json

    if not YT_VIDEOS_JSON.exists():
        print(f"WARNING: {YT_VIDEOS_JSON} not found, cannot build section map")
        return {}

    with open(YT_VIDEOS_JSON, 'r') as f:
        data = json.load(f)

    mapping = {}

    # Drive tours (tier1)
    for city, vids in data.get("drive_tours", {}).items():
        for v in vids:
            if v.get("id"):
                mapping[v["id"]] = f"tier1/{city}/drive"

    # Drone views (tier1)
    for city, vids in data.get("drone_views", {}).items():
        for v in vids:
            if v.get("id"):
                mapping[v["id"]] = f"tier1/{city}/drone"

    # Walking tours (tier1 + goa)
    for city, vids in data.get("walking_tours", {}).items():
        for v in vids:
            if v.get("id"):
                if city == "goa":
                    mapping[v["id"]] = "goa/walking"
                else:
                    mapping[v["id"]] = f"tier1/{city}/walking"

    # Tier 2
    for city, city_data in data.get("tier2_cities", {}).items():
        for tour_type in ["drive", "walking", "drone", "rain"]:
            for v in city_data.get(tour_type, []):
                if v.get("id"):
                    mapping[v["id"]] = f"tier2/{city}/{tour_type}"

    # Monuments
    for m in data.get("monuments", []):
        monument_name = sanitize_name(m.get("name", "unknown"))
        city = m.get("city", "")
        if city:
            dir_name = f"{monument_name}_{sanitize_name(city)}"
        else:
            dir_name = monument_name
        for tour_type in ["walking_tours", "drive_tours", "drone_views"]:
            for v in m.get(tour_type, []):
                if v.get("id"):
                    mapping[v["id"]] = f"monuments/{dir_name}"

    return mapping


# Processed-video tracking (for m02 with clip-count verification)
PROCESSED_VIDEOS_FILE = CLIPS_DIR / ".processed.json"


def get_processed_video_ids() -> dict:
    """Get dict of {video_id: clip_count} already processed by m02."""
    import json
    if not PROCESSED_VIDEOS_FILE.exists():
        # Migrate from old .processed.txt if it exists
        old_file = CLIPS_DIR / ".processed.txt"
        if old_file.exists():
            ids = set(old_file.read_text().strip().split('\n')) - {''}
            migrated = {vid_id: -1 for vid_id in ids}  # -1 = unknown count
            with open(PROCESSED_VIDEOS_FILE, 'w') as f:
                json.dump(migrated, f)
            old_file.unlink()
            print(f"Migrated {len(migrated)} entries from .processed.txt → .processed.json")
            return migrated
        return {}
    with open(PROCESSED_VIDEOS_FILE) as f:
        return json.load(f)


def mark_video_processed(video_id: str, clip_count: int):
    """Record that a video has been processed with its clip count."""
    import json
    PROCESSED_VIDEOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    processed = get_processed_video_ids()
    processed[video_id] = clip_count
    with open(PROCESSED_VIDEOS_FILE, 'w') as f:
        json.dump(processed, f)


def load_embeddings_and_tags() -> tuple:
    """
    Load embeddings and tags, verify alignment.
    Returns (embeddings, tags) or exits on error.
    """
    import json
    import numpy as np

    # Load embeddings
    if not EMBEDDINGS_FILE.exists():
        print(f"ERROR: Embeddings not found: {EMBEDDINGS_FILE}")
        print("Run m05_vjepa_embed.py first")
        sys.exit(1)

    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load tags
    if not TAGS_FILE.exists():
        print(f"ERROR: Tags not found: {TAGS_FILE}")
        print("Run m04_vlm_tag.py first")
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

    return embeddings, tags


# =============================================================================
# BATCH PROCESSING CONFIG (optimized for A100-40GB)
# =============================================================================
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 8
RAM_CACHE_DIR = Path("/dev/shm/video_clips_cache")


def setup_ram_cache(clip_paths: list, use_cache: bool = True, cache_subdir: str = "clips") -> tuple:
    """
    Copy clips to RAM (/dev/shm) for faster I/O.

    Args:
        clip_paths: List of clip paths
        use_cache: Whether to use RAM cache
        cache_subdir: Subdirectory name in /dev/shm

    Returns:
        Tuple of (cached_paths, cache_enabled)
    """
    import shutil
    from tqdm import tqdm

    cache_dir = Path(f"/dev/shm/{cache_subdir}_cache")

    if not use_cache:
        return clip_paths, False

    # Check available RAM in /dev/shm
    try:
        shm_stat = os.statvfs("/dev/shm")
        available_gb = (shm_stat.f_bavail * shm_stat.f_frsize) / (1024**3)

        # Estimate clip size (assume ~2MB per clip average)
        estimated_size_gb = len(clip_paths) * 2 / 1024

        if available_gb < estimated_size_gb + 2:  # Keep 2GB buffer
            print(f"RAM cache: Skipping (need {estimated_size_gb:.1f}GB, have {available_gb:.1f}GB)")
            return clip_paths, False

    except Exception as e:
        print(f"RAM cache: Skipping ({e})")
        return clip_paths, False

    print(f"\n=== Setting up RAM cache ===")
    print(f"Copying {len(clip_paths)} clips to /dev/shm (~{estimated_size_gb:.1f}GB)")

    # Clean up old cache
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_paths = []
    for src_path in tqdm(clip_paths, desc="Caching to RAM", unit="clip"):
        src_path = Path(src_path)
        # Use relative path from CLIPS_DIR with __ separator for hierarchical dirs
        try:
            rel = src_path.relative_to(CLIPS_DIR)
            cache_name = str(rel).replace(os.sep, "__")
        except ValueError:
            cache_name = f"{src_path.parent.name}__{src_path.name}"
        dst_path = cache_dir / cache_name
        shutil.copy2(src_path, dst_path)
        cached_paths.append(dst_path)

    print(f"RAM cache ready: {cache_dir}")
    return cached_paths, True


def cleanup_ram_cache(cache_subdir: str = "clips"):
    """Remove RAM cache after processing."""
    import shutil
    cache_dir = Path(f"/dev/shm/{cache_subdir}_cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print("RAM cache cleaned up")


def get_deduplicated_clips() -> list:
    """
    Get clip paths from m04's deduplicated output (embeddings.paths.npy).
    Falls back to all clips if embeddings not available.

    Returns:
        List of Path objects for clips
    """
    import numpy as np

    paths_file = EMBEDDINGS_FILE.with_suffix('.paths.npy')

    if paths_file.exists():
        clip_paths = np.load(paths_file, allow_pickle=True).tolist()
        print(f"Loaded {len(clip_paths)} deduplicated clips from {paths_file.name}")
        return [Path(p) for p in clip_paths]
    else:
        print(f"WARNING: {paths_file.name} not found. Run m05_vjepa_embed.py first.")
        print("Falling back to all clips (may cause misalignment with embeddings)")
        return get_all_clips()


def restore_original_path(cache_path: Path, clips_dir: Path = None) -> str:
    """Convert RAM cache path back to original path (supports hierarchical dirs)."""
    if clips_dir is None:
        clips_dir = CLIPS_DIR
    cache_name = cache_path.name
    # Replace __ back to path separators (hierarchical structure)
    rel_path = cache_name.replace("__", os.sep)
    restored = clips_dir / rel_path
    if restored.exists():
        return str(restored)
    # Fallback: old-style single __ split
    parts = cache_name.split("__", 1)
    if len(parts) == 2:
        return str(clips_dir / parts[0] / parts[1])
    return str(cache_path)


def check_output_exists(output_paths: list, description: str = "output") -> bool:
    """
    Check if output files exist and prompt user for action.

    Args:
        output_paths: List of Path objects to check
        description: Description of the output for display

    Returns:
        True if should process (delete and re-run)
        False if should skip (use cached files)
    """
    import shutil

    # Convert to Path objects and check existence
    existing = []
    for p in output_paths:
        p = Path(p)
        if p.exists():
            existing.append(p)
        elif p.is_dir() or (p.parent.exists() and any(p.parent.iterdir())):
            # Check if it's a directory pattern
            if p.is_dir():
                existing.append(p)

    if not existing:
        return True  # No existing files, proceed with processing

    # Display existing files
    print(f"\n{'='*50}")
    print(f"Found existing {description}:")
    for p in existing[:5]:  # Show max 5
        if p.is_dir():
            count = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {p}/ ({count} files)")
        else:
            print(f"  {p}")
    if len(existing) > 5:
        print(f"  ... and {len(existing) - 5} more")
    print(f"{'='*50}")

    # Prompt user
    print("\nOptions:")
    print("  [1] Delete existing and re-run")
    print("  [2] Use cached files (skip processing)")

    while True:
        try:
            choice = input("\nEnter choice (1 or 2): ").strip()
            if choice == "1":
                # Delete existing files
                for p in existing:
                    if p.is_dir():
                        shutil.rmtree(p)
                        print(f"Deleted: {p}/")
                    else:
                        p.unlink()
                        print(f"Deleted: {p}")
                return True  # Proceed with processing
            elif choice == "2":
                print("Using cached files, skipping processing.")
                return False  # Skip processing
            else:
                print("Invalid choice. Enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(1)
        except EOFError:
            # Non-interactive mode - default to using cache
            print("Non-interactive mode: using cached files.")
            return False
