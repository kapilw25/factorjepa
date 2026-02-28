"""HuggingFace shared utilities: auth, README generation, metadata upload.
Used by m03_pack_shards.py (TAR upload) and m04_vlm_tag.py (metadata upload).
"""
import os
from pathlib import Path


def _setup_hf_env():
    """Override HF cache paths for local machine (prevents .env GPU paths on Mac)."""
    local_cache = Path.home() / ".cache" / "huggingface"
    local_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(local_cache)
    os.environ["HF_HUB_CACHE"] = str(local_cache / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(local_cache / "transformers")


def _get_token() -> str:
    """Load HF_TOKEN from .env."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.getenv("HF_TOKEN")


def generate_readme(num_clips: int, num_videos: int, total_gb: float,
                    num_shards: int = 0) -> str:
    """Generate README.md content. No ffprobe — uses pre-computed stats."""
    if num_clips > 100000:
        size_cat = "100K<n<1M"
    elif num_clips > 10000:
        size_cat = "10K<n<100K"
    elif num_clips > 1000:
        size_cat = "1K<n<10K"
    else:
        size_cat = "n<1K"

    readme = f"""---
license: cc-by-4.0
task_categories:
  - video-classification
  - feature-extraction
language:
  - en
tags:
  - video
  - indian-streets
  - walking-tour
  - driving-tour
  - drone-view
  - v-jepa
  - scene-detection
  - webdataset
size_categories:
  - {size_cat}
configs:
  - config_name: default
    data_files:
      - split: train
        path: "data/train-*.tar"
dataset_info:
  config_name: default
  features:
    - name: mp4
      dtype: video
    - name: json
      struct:
        - name: video_id
          dtype: string
        - name: section
          dtype: string
        - name: tier
          dtype: string
        - name: city
          dtype: string
        - name: tour_type
          dtype: string
        - name: duration_sec
          dtype: float64
        - name: size_mb
          dtype: float64
        - name: source_file
          dtype: string
    - name: __key__
      dtype: string
    - name: __url__
      dtype: string
---

# WalkIndia-200K Video Clips

~{num_clips // 1000}K video clips (4-10s each) from {num_videos} Indian street videos (walking tours, driving tours, drone views) across 21+ cities for evaluating video foundation models on non-Western urban scenes.

## Dataset Description

| Property | Value |
|----------|-------|
| Format | WebDataset (TAR shards) |
| Total Clips | {num_clips:,} |
| Shards | {num_shards} x ~1 GB TAR files |
| Source Videos | {num_videos} |
| Duration Range | 4.0s - 10.0s |
| Mean Duration | ~8.6s |
| Total Size | {total_gb:.1f} GB |
| Total Hours | ~277 hours source footage |
| Cities | 21+ (6 tier-1, 15 tier-2, Goa, monuments) |

## Format

WebDataset TAR shards with paired mp4 + json files:

```
data/
├── train-00000.tar
│   ├── 000000.mp4    # video clip
│   ├── 000000.json   # metadata (video_id, section, duration, etc.)
│   ├── 000001.mp4
│   ├── 000001.json
│   └── ...
├── train-00001.tar
└── ...
```

### Loading

```python
from datasets import load_dataset

# Streaming (recommended — no local download)
ds = load_dataset("anonymousML123/walkindia-200k", streaming=True)
for sample in ds["train"]:
    video = sample["mp4"]
    metadata = sample["json"]
```

## Processing Pipeline

1. **Download**: `yt-dlp` at 480p ({num_videos} videos from YouTube)
2. **Scene Detection**: `PySceneDetect` ContentDetector (threshold=15.0)
3. **Greedy Split**: Scene-aware splitting, 4-10s clips, libx264 CRF 28

## Metadata Fields

Each clip's JSON sidecar contains:

| Field | Description |
|-------|-------------|
| `video_id` | YouTube video ID |
| `section` | Geographic section (e.g., `tier1/mumbai/drive`) |
| `tier` | City tier (`tier1`, `tier2`, `goa`, `monuments`) |
| `city` | City name |
| `tour_type` | Tour type (`walking`, `drive`, `drone`, `rain`) |
| `duration_sec` | Clip duration in seconds |
| `size_mb` | File size in MB |
| `source_file` | Original clip filename |

## Intended Use

- Evaluate V-JEPA 2 video embeddings on Indian urban scenes
- Test geographic transfer of video foundation models
- Research on non-Western video understanding

## Citation

```
@dataset{{walkindia200k,
  title={{WalkIndia-200K: Indian Street Video Clips for Video Foundation Model Evaluation}},
  author={{Anonymous}},
  year={{2026}},
  url={{https://huggingface.co/datasets/anonymousML123/walkindia-200k}}
}}
```

## License

CC-BY-4.0 (Creative Commons Attribution 4.0)

Original videos sourced from YouTube walking/driving/drone tour channels.
"""
    return readme


def upload_readme(repo_id: str, token: str, readme_content: str) -> bool:
    """Upload README.md to the dataset."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)

    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print("Uploaded: README.md")
        return True
    except Exception as e:
        print(f"README upload error: {e}")
        return False


def upload_metadata_only(clips_dir: Path, repo_id: str, token: str, leaf_dirs: set = None) -> int:
    """Upload only metadata.jsonl files to existing HF repo (no clips re-upload).
    Traverses hierarchical directory structure recursively."""
    from huggingface_hub import HfApi
    api = HfApi(token=token)

    count = 0
    for metadata_file in clips_dir.rglob("metadata.jsonl"):
        rel_dir = str(metadata_file.parent.relative_to(clips_dir))
        if leaf_dirs is not None and rel_dir not in leaf_dirs:
            continue
        rel = metadata_file.relative_to(clips_dir)
        path_in_repo = f"clips/{rel}"
        api.upload_file(
            path_or_fileobj=str(metadata_file),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        count += 1

    return count


def upload_metadata(clips_dir: Path, repo_id: str) -> int:
    """Upload only metadata.jsonl files to existing HF repo (no clips re-upload).
    Called from m04_vlm_tag.py after tagging on GPU server."""
    _setup_hf_env()
    token = _get_token()
    if not token:
        print("WARNING: HF_TOKEN not found, skipping metadata upload")
        return 0

    count = upload_metadata_only(clips_dir, repo_id, token)
    print(f"\n=== METADATA UPLOAD COMPLETE ===")
    print(f"Uploaded: {count} metadata.jsonl files")
    print(f"Dataset: https://huggingface.co/datasets/{repo_id}")
    return count
