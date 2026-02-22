"""
Convert tags.json → per-directory metadata.jsonl for HuggingFace dataset upload.
HF auto-loads metadata.jsonl alongside media files as dataset columns.
Supports hierarchical clip directory structure (tier1/city/type, etc.).
"""
import json
from collections import defaultdict
from pathlib import Path


def load_tags(tags_file: Path) -> list:
    """Load tags.json and return list of tag dicts."""
    with open(tags_file, 'r') as f:
        return json.load(f)


def group_tags_by_leaf_dir(tags: list, clips_dir: Path) -> dict:
    """Group tag entries by leaf directory (relative path from clips_dir)."""
    groups = defaultdict(list)
    for tag in tags:
        clip_path = Path(tag["clip_path"])
        try:
            rel_dir = str(clip_path.parent.relative_to(clips_dir))
        except ValueError:
            # Fallback: use parent directory name
            rel_dir = clip_path.parent.name
        groups[rel_dir].append(tag)
    return dict(groups)


def tag_to_hf_metadata(tag: dict) -> dict:
    """Convert a tag dict to HF metadata format (file_name + fields, no clip_path)."""
    clip_path = Path(tag["clip_path"])
    metadata = {"file_name": clip_path.name}
    for key, value in tag.items():
        if key != "clip_path":
            metadata[key] = value
    return metadata


def export_metadata_jsonl(tags: list, clips_dir: Path) -> dict:
    """
    Write metadata.jsonl files into each leaf directory under clips_dir.

    Args:
        tags: List of tag dicts (from tags.json)
        clips_dir: Root clips directory (CLIPS_DIR)

    Returns:
        Summary dict with stats
    """
    groups = group_tags_by_leaf_dir(tags, clips_dir)

    total_clips = 0
    total_dirs = 0
    skipped_dirs = 0

    for rel_dir, dir_tags in sorted(groups.items()):
        leaf_dir = clips_dir / rel_dir
        if not leaf_dir.exists():
            skipped_dirs += 1
            continue

        metadata_path = leaf_dir / "metadata.jsonl"
        with open(metadata_path, 'w') as f:
            for tag in dir_tags:
                metadata = tag_to_hf_metadata(tag)
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                total_clips += 1

        total_dirs += 1

    return {
        "total_clips": total_clips,
        "total_dirs": total_dirs,
        "skipped_dirs": skipped_dirs,
        "fields_per_clip": len(tag_to_hf_metadata(tags[0])) if tags else 0,
    }


def print_summary(stats: dict, clips_dir: Path):
    """Print export summary."""
    print(f"\n=== METADATA EXPORT SUMMARY ===")
    print(f"Directories with metadata.jsonl: {stats['total_dirs']}")
    print(f"Clips mapped:                    {stats['total_clips']}")
    print(f"Fields per clip:                 {stats['fields_per_clip']}")
    if stats['skipped_dirs'] > 0:
        print(f"Skipped (dir missing):           {stats['skipped_dirs']}")
    print(f"Output:                          {clips_dir}/**/metadata.jsonl")
