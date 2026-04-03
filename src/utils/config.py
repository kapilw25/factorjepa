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
LOGS_DIR = PROJECT_ROOT / "logs"

# Consolidated output directories (all under outputs/)
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
OUTPUTS_DIR = OUTPUTS_ROOT / "full"
OUTPUTS_DATA_PREP_DIR = OUTPUTS_ROOT / "data_prep"
OUTPUTS_PROFILE_DIR = OUTPUTS_ROOT / "profile"

# Input data (canonical copies in src/utils/)
YT_VIDEOS_JSON = UTILS_DIR / "YT_videos_raw.json"
TAG_TAXONOMY_JSON = UTILS_DIR / "tag_taxonomy.json"

# Ensure directories exist
for d in [VIDEOS_DIR, CLIPS_DIR, SHARDS_DIR, OUTPUTS_DIR, OUTPUTS_DATA_PREP_DIR,
          OUTPUTS_PROFILE_DIR, LOGS_DIR]:
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

# ── Pipeline config (clip limits from YAML, not hardcoded) ───────────
PIPELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"
_pipeline_cfg = None

def get_pipeline_config() -> dict:
    """Load configs/pipeline.yaml (cached). Single source of truth for clip limits."""
    global _pipeline_cfg
    if _pipeline_cfg is None:
        import yaml
        with open(PIPELINE_CONFIG_PATH) as f:
            _pipeline_cfg = yaml.safe_load(f)
    return _pipeline_cfg


def get_sanity_clip_limit(module: str) -> int:
    """Get SANITY clip limit for a module from configs/pipeline.yaml."""
    cfg = get_pipeline_config()
    return cfg["sanity"].get(module, cfg["sanity"]["default"])


def get_total_clips(local_data: str = None, subset_file: str = None) -> int:
    """Discover total clip count from data source. Never hardcode."""
    import json
    if subset_file:
        keys = load_subset(subset_file)
        return len(keys)
    if local_data:
        manifest = Path(local_data) / "manifest.json"
        if manifest.exists():
            data = json.load(open(manifest))
            return data.get("n", data.get("n_clips", data.get("total_clips", 0)))
    return 0  # caller must handle 0


# FAISS config (after get_pipeline_config is defined)
FAISS_K_NEIGHBORS = get_pipeline_config()["eval"]["faiss_k_neighbors"]

# POC subset config
SUBSET_FILE = PROJECT_ROOT / "data" / "subset_10k.json"
OUTPUTS_SANITY_DIR = OUTPUTS_ROOT / "sanity"
OUTPUTS_POC_DIR = OUTPUTS_ROOT / "poc"
BAKEOFF_DIR = DATA_DIR / "bakeoff"

# VLM bake-off config
VLM_MODELS = {
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    "videollama": "DAMO-NLP-SG/VideoLLaMA3-7B",
    "llava": "llava-hf/LLaVA-NeXT-Video-7B-hf",
}
BAKEOFF_CLIP_COUNT = get_pipeline_config()["bakeoff"]["clips"]

# Output files
EMBEDDINGS_FILE = OUTPUTS_DIR / "embeddings.npy"
TAGS_FILE = OUTPUTS_DIR / "tags.json"
UMAP_PLOT_PNG = OUTPUTS_DIR / "m08_umap.png"
UMAP_PLOT_PDF = OUTPUTS_DIR / "m08_umap.pdf"
METRICS_FILE = OUTPUTS_DIR / "m06_metrics.json"


# Encoder registry (baselines + V-JEPA). suffix="" = backward compat for vjepa.
ENCODER_REGISTRY = {
    "vjepa":          {"model_id": VJEPA_MODEL_ID,                       "dim": 1408, "type": "video",          "suffix": ""},
    "random":         {"model_id": None,                                  "dim": 1408, "type": "synthetic",      "suffix": "_random"},
    "dinov2":         {"model_id": "facebook/dinov2-giant",              "dim": 1536, "type": "image",           "suffix": "_dinov2"},
    "clip":           {"model_id": "openai/clip-vit-large-patch14",      "dim": 768,  "type": "image",           "suffix": "_clip"},
    "vjepa_shuffled": {"model_id": VJEPA_MODEL_ID,                       "dim": 1408, "type": "video_shuffled",  "suffix": "_vjepa_shuffled"},
    "vjepa_adapted":  {"model_id": None,                                  "dim": 1408, "type": "video_adapted",   "suffix": "_vjepa_adapted"},
}


def get_encoder_info(encoder: str) -> dict:
    """Get encoder info from registry, with dynamic fallback for unregistered encoders.

    Supports Ch10 ablation variants like vjepa_lambda0_01 without pre-registration.
    """
    if encoder in ENCODER_REGISTRY:
        return ENCODER_REGISTRY[encoder]
    # Dynamic fallback: assume V-JEPA dim (1408), infer suffix from name
    return {"model_id": None, "dim": 1408, "type": "video_adapted", "suffix": f"_{encoder}"}


def get_encoder_files(encoder: str, output_dir: Path) -> dict:
    """Return {embeddings, paths, metrics, knn_indices, umap_2d} paths for an encoder."""
    sfx = get_encoder_info(encoder)["suffix"]
    return {
        "embeddings":  output_dir / f"embeddings{sfx}.npy",
        "paths":       output_dir / f"embeddings{sfx}.paths.npy",
        "metrics":     output_dir / f"m06_metrics{sfx}.json",
        "knn_indices": output_dir / f"knn_indices{sfx}.npy",
        "umap_2d":     output_dir / f"umap_2d{sfx}.npy",
    }


def add_encoder_arg(parser):
    """Add --encoder argument to m05b/m06/m07/m08 parsers.

    Accepts registered encoders + any custom name (for Ch10 adapted variants).
    """
    parser.add_argument("--encoder", default="vjepa",
                        help=f"Encoder name (registered: {', '.join(ENCODER_REGISTRY.keys())}; "
                             "or any custom name for adapted variants)")


# Ensure POC/sanity/bakeoff directories exist
for d in [OUTPUTS_SANITY_DIR, OUTPUTS_POC_DIR, BAKEOFF_DIR]:
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


def get_output_dir(subset_path: str = None, sanity: bool = False) -> Path:
    """
    Return output directory based on mode.
    SANITY mode          → outputs_sanity/
    POC mode (--subset)  → outputs_poc/
    Full mode            → outputs/
    """
    if sanity:
        return OUTPUTS_SANITY_DIR
    if subset_path:
        return OUTPUTS_POC_DIR
    return OUTPUTS_DIR


def add_subset_arg(parser):
    """Add --subset argument to any argparse parser (shared across m04-m08)."""
    parser.add_argument("--subset", type=str, default=None,
                        help="Path to subset JSON (e.g., data/subset_10k.json) for POC mode")


def add_local_data_arg(parser):
    """Add --local-data argument for pre-downloaded local WebDataset shards."""
    parser.add_argument("--local-data", type=str, default=None,
                        help="Local WebDataset dir (from m00d_download_subset.py). "
                             "Bypasses HF streaming for 100%% hit rate.")


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
DEFAULT_BATCH_SIZE = get_pipeline_config()["gpu"]["default_batch_size"]
DEFAULT_NUM_WORKERS = get_pipeline_config()["gpu"]["default_num_workers"]
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

    # Auto-use cached (no interactive prompt — use `rm` to force re-run)
    # See shell script docstring for cache control commands
    print("Using cached files, skipping processing.")
    return False


# ═════════════════════════════════════════════════════════════════════════
# CLI — shell scripts call these instead of inline python -c "..."
# ═════════════════════════════════════════════════════════════════════════

def _get_nested(d: dict, key_path: str):
    """Traverse nested dict by dot-separated key path. e.g. 'optimization.max_epochs.full'."""
    for key in key_path.split("."):
        d = d[key]
    return d


if __name__ == "__main__":
    import json

    usage = """Usage:
  python -u src/utils/config.py get-yaml <yaml_path> <key_path>
  python -u src/utils/config.py get-json <json_path> <key>

Examples:
  python -u src/utils/config.py get-yaml configs/pretrain/vitg16_indian.yaml optimization.max_epochs.full
  python -u src/utils/config.py get-yaml configs/pipeline.yaml verify.sanity_min_clips
  python -u src/utils/config.py get-json outputs/full/ablation_winner.json winner_lambda
  python -u src/utils/config.py get-json outputs/full/m09_lambda0_001/training_summary.json epochs
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "get-yaml":
        if len(sys.argv) != 4:
            print("Usage: get-yaml <yaml_path> <key_path>")
            sys.exit(1)
        import yaml
        with open(sys.argv[2]) as f:
            cfg = yaml.safe_load(f)
        print(_get_nested(cfg, sys.argv[3]))

    elif cmd == "get-json":
        if len(sys.argv) != 4:
            print("Usage: get-json <json_path> <key>")
            sys.exit(1)
        with open(sys.argv[2]) as f:
            data = json.load(f)
        print(data[sys.argv[3]])

    else:
        print(f"Unknown command: {cmd}")
        print(usage)
        sys.exit(1)
