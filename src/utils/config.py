"""
Common configuration for WalkIndia-50 POC.
"""
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = SRC_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
CLIPS_DIR = DATA_DIR / "clips"
OUTPUTS_DIR = SRC_DIR / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for d in [VIDEOS_DIR, CLIPS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# HuggingFace dataset config
HF_DATASET_REPO = "anonymousML123/walkindia-50-clips"


def ensure_clips_exist() -> bool:
    """
    Check if clips exist locally. If not, download from HuggingFace.
    Returns True if clips are available, False otherwise.
    """
    # Check if clips already exist
    clip_dirs = [d for d in CLIPS_DIR.iterdir() if d.is_dir()] if CLIPS_DIR.exists() else []
    clip_count = sum(len(list(d.glob("*.mp4"))) for d in clip_dirs)

    if clip_count > 0:
        print(f"Found {clip_count} clips locally in {CLIPS_DIR}")
        return True

    # Auto-download from HuggingFace
    print(f"No clips found locally. Downloading from HuggingFace: {HF_DATASET_REPO}")
    try:
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            local_dir=DATA_DIR / "hf_download",
            allow_patterns=["clips/**/*.mp4"],
        )

        # Move clips to expected location
        hf_clips_dir = Path(local_dir) / "clips"
        if hf_clips_dir.exists():
            import shutil
            for scene_dir in hf_clips_dir.iterdir():
                if scene_dir.is_dir():
                    dest_dir = CLIPS_DIR / scene_dir.name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for clip in scene_dir.glob("*.mp4"):
                        shutil.copy2(clip, dest_dir / clip.name)

            # Cleanup temp download
            shutil.rmtree(DATA_DIR / "hf_download", ignore_errors=True)

        # Verify
        clip_dirs = [d for d in CLIPS_DIR.iterdir() if d.is_dir()]
        clip_count = sum(len(list(d.glob("*.mp4"))) for d in clip_dirs)
        print(f"Downloaded {clip_count} clips from HuggingFace")
        return clip_count > 0

    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"ERROR downloading from HuggingFace: {e}")
        return False

# Video URLs from @WalkinginIndia YouTube channel
# 3 different scenarios (~10 min each)
VIDEO_URLS = {
    "temple": "https://www.youtube.com/watch?v=ufV-7oGcxps",   # Amritsar - Golden Temple & Bazaar
    "metro": "https://www.youtube.com/watch?v=fGID1n2j5Qs",    # Delhi Metro - Dilli Haat INA Station
    "hilltown": "https://www.youtube.com/watch?v=wB5kuvzA5JI", # Mussoorie - Camel's Back Road
}

# PySceneDetect config
CLIP_MIN_DURATION = 4.0  # seconds
CLIP_MAX_DURATION = 5.0  # seconds

# V-JEPA config
VJEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
VJEPA_FRAMES_PER_CLIP = 64
VJEPA_EMBEDDING_DIM = 768

# Qwen3-VL config
QWEN_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# FAISS config
FAISS_K_NEIGHBORS = 6  # includes self

# Output files
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
TAGS_FILE = DATA_DIR / "tags.json"
UMAP_PLOT_PNG = OUTPUTS_DIR / "poc_umap.png"
UMAP_PLOT_PDF = OUTPUTS_DIR / "poc_umap.pdf"
METRICS_FILE = OUTPUTS_DIR / "metrics.json"


# =============================================================================
# UTILITY FUNCTIONS (avoid redundancy across modules)
# =============================================================================

def check_gpu():
    """Check if CUDA GPU is available. Exit if not."""
    import sys
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
    Get all video clips from CLIPS_DIR.
    Returns list of Path objects.
    """
    clip_dirs = [d for d in CLIPS_DIR.iterdir() if d.is_dir()]
    all_clips = []
    for clip_dir in clip_dirs:
        all_clips.extend(list(clip_dir.glob("*.mp4")))
    return all_clips


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


def load_embeddings_and_tags() -> tuple:
    """
    Load embeddings and tags, verify alignment.
    Returns (embeddings, tags) or exits on error.
    """
    import sys
    import json
    import numpy as np

    # Load embeddings
    if not EMBEDDINGS_FILE.exists():
        print(f"ERROR: Embeddings not found: {EMBEDDINGS_FILE}")
        print("Run m03_vjepa_embed.py first")
        sys.exit(1)

    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load tags
    if not TAGS_FILE.exists():
        print(f"ERROR: Tags not found: {TAGS_FILE}")
        print("Run m04_qwen_tag.py first")
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
