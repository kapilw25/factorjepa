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
