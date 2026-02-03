"""
Upload video clips to HuggingFace dataset for GPU server access.
CPU-only script (M1 compatible). Auto-uses upload_large_folder() for >10K clips.

USAGE:
    python -u src/m02b_upload_hf.py --SANITY 2>&1 | tee logs/m02b_upload_hf_sanity.log
    python -u src/m02b_upload_hf.py --FULL 2>&1 | tee logs/m02b_upload_hf_full.log
"""
import argparse
import os
import sys
from pathlib import Path

# Set LOCAL cache paths BEFORE importing huggingface_hub
# This overrides GPU server paths in .env that don't exist on M1 Mac
LOCAL_HF_CACHE = Path.home() / ".cache" / "huggingface"
LOCAL_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(LOCAL_HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(LOCAL_HF_CACHE / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(LOCAL_HF_CACHE / "transformers")

from dotenv import load_dotenv

# Load only HF_TOKEN from .env (cache paths already set above)
load_dotenv()

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import CLIPS_DIR, PROJECT_ROOT, HF_DATASET_REPO, get_all_clips, get_video_duration

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("ERROR: huggingface_hub not found. Install with: pip install huggingface_hub")
    sys.exit(1)

# HuggingFace config
HF_TOKEN = os.getenv("HF_TOKEN")


def generate_readme(clips: list, clips_dir: Path) -> str:
    """Generate README.md content for the dataset."""
    import statistics

    # Count clips per scene type
    scene_counts = {}
    durations = []

    for clip in clips:
        scene_type = clip.parent.name
        scene_counts[scene_type] = scene_counts.get(scene_type, 0) + 1

        dur = get_video_duration(clip)
        if dur > 0:
            durations.append(dur)

    # Calculate stats
    min_dur = min(durations) if durations else 0
    max_dur = max(durations) if durations else 0
    mean_dur = statistics.mean(durations) if durations else 0

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
  - v-jepa
  - scene-detection
size_categories:
  - n<1K
---

# WalkIndia-50 Video Clips

Video clips from Indian street walking tours for evaluating video foundation models (V-JEPA 2) on non-Western urban scenes.

## Dataset Description

| Property | Value |
|----------|-------|
| Total Clips | {len(clips)} |
| Duration Range | {min_dur:.1f}s - {max_dur:.1f}s |
| Mean Duration | {mean_dur:.1f}s |
| Source | [@WalkinginIndia YouTube](https://www.youtube.com/@WalkinginIndia) |

### Scene Types

| Scene | Clips | Description |
|-------|-------|-------------|
"""

    for scene, count in sorted(scene_counts.items()):
        descriptions = {
            "temple": "Golden Temple, Amritsar - religious site & bazaar",
            "metro": "Delhi Metro - Dilli Haat INA Station underground",
            "hilltown": "Mussoorie - Camel's Back Road hill station",
        }
        desc = descriptions.get(scene, "Walking tour footage")
        readme += f"| {scene} | {count} | {desc} |\n"

    readme += """
## Processing Pipeline

1. **Download**: `yt-dlp` (10 min per video)
2. **Scene Detection**: `PySceneDetect` ContentDetector (threshold=27.0)
3. **Intelligent Re-split**: Long clips split using:
   - ContentDetector (threshold=15.0) for subtle changes
   - AdaptiveDetector for gradual transitions
   - Fixed 5s chunks as fallback

## Intended Use

- Evaluate V-JEPA 2 video embeddings on Indian urban scenes
- Test geographic transfer of video foundation models
- Research on non-Western video understanding

## Citation

If you use this dataset, please cite:

```
@dataset{walkindia50,
  title={WalkIndia-50: Indian Street Video Clips},
  author={Anonymous},
  year={2026},
  url={https://huggingface.co/datasets/anonymousML123/walkindia-50-clips}
}
```

## License

CC-BY-4.0 (Creative Commons Attribution 4.0)

Original videos from @WalkinginIndia YouTube channel.
"""

    return readme


def upload_readme(repo_id: str, token: str, readme_content: str) -> bool:
    """Upload README.md to the dataset."""
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


def upload_clips_to_hf(clips_dir: Path, repo_id: str, token: str, readme_content: str, use_large: bool = False) -> int:
    """
    Upload video clips to HuggingFace dataset.

    Args:
        clips_dir: Directory containing clip subdirectories
        repo_id: HuggingFace repo ID (username/repo-name)
        token: HuggingFace token
        readme_content: README.md content to include
        use_large: Use upload_large_folder() for 200K+ clips (resumable, incremental)

    Returns:
        Number of clips uploaded
    """
    import tempfile
    import shutil

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
        print(f"Repository: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Repo creation note: {e}")

    # Create temp directory with proper structure for upload
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write README.md
        readme_path = tmpdir / "README.md"
        readme_path.write_text(readme_content)
        print("Created: README.md")

        # Copy clips maintaining folder structure
        clips_dest = tmpdir / "clips"
        print(f"Preparing clips for upload...")

        clip_count = 0
        for scene_dir in clips_dir.iterdir():
            if not scene_dir.is_dir():
                continue

            dest_scene_dir = clips_dest / scene_dir.name
            dest_scene_dir.mkdir(parents=True, exist_ok=True)

            for clip in scene_dir.glob("*.mp4"):
                shutil.copy2(clip, dest_scene_dir / clip.name)
                clip_count += 1

        print(f"Prepared {clip_count} clips in {clips_dest}")

        if use_large:
            # For 200K+ clips: incremental commits, resumable, survives interrupts
            print(f"\nUsing upload_large_folder() for {clip_count} clips...")
            print("Features: incremental commits, resumable, survives interrupts")
            try:
                api.upload_large_folder(
                    folder_path=str(tmpdir),
                    repo_id=repo_id,
                    repo_type="dataset",
                    num_workers=4,  # Parallel uploads
                )
                print(f"Upload successful!")
                return clip_count
            except Exception as e:
                print(f"Upload error: {e}")
                return 0
        else:
            # For small datasets (<10K clips): single commit
            print(f"\nUploading folder to HuggingFace (this may take a few minutes)...")
            try:
                api.upload_folder(
                    folder_path=str(tmpdir),
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )
                print(f"Upload successful!")
                return clip_count
            except Exception as e:
                print(f"Upload error: {e}")
                return 0


def main():
    parser = argparse.ArgumentParser(description="Upload clips to HuggingFace")
    parser.add_argument("--SANITY", action="store_true", help="Upload 3 clips only")
    parser.add_argument("--FULL", action="store_true", help="Upload all clips")
    parser.add_argument("--repo", type=str, default=HF_DATASET_REPO, help="HuggingFace repo ID")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not found in .env")
        print("Add HF_TOKEN=hf_xxx to your .env file")
        sys.exit(1)

    # Find all clips
    all_clips = get_all_clips()
    if not all_clips:
        print(f"ERROR: No clips found in {CLIPS_DIR}")
        print("Run m02_scene_detect.py first")
        sys.exit(1)

    print(f"Found {len(all_clips)} clips")

    # Auto-detect large dataset threshold
    use_large = len(all_clips) > 10000
    if use_large:
        print(f"\nLarge dataset detected ({len(all_clips)} clips). Using upload_large_folder():")
        print("  - Incremental commits (survives interrupts)")
        print("  - Resumable uploads")
        print("  - Memory efficient streaming")

    # Limit for sanity mode
    if args.SANITY:
        all_clips = all_clips[:3]
        print(f"SANITY MODE: Uploading {len(all_clips)} clips")

    # Upload
    print(f"\nUploading to: {args.repo}")

    # Generate README
    print("\nGenerating README.md...")
    readme_content = generate_readme(all_clips, CLIPS_DIR)

    # Upload entire folder (clips + README)
    uploaded = upload_clips_to_hf(CLIPS_DIR, args.repo, HF_TOKEN, readme_content, use_large=use_large)

    print(f"\n=== UPLOAD COMPLETE ===")
    print(f"Uploaded: {uploaded} clips + README.md")
    print(f"Dataset: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
