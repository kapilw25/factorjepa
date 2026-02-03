"""
Generate structured tags for video clips using Qwen3-VL-8B with batched GPU inference.
Skips tagging if tags.json exists (delete manually to retag).

USAGE:
    python -u src/m04_qwen_tag.py --SANITY 2>&1 | tee logs/m04_qwen_tag_sanity.log
    python -u src/m04_qwen_tag.py --FULL 2>&1 | tee logs/m04_qwen_tag_full.log
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Load HF_TOKEN from .env for authenticated downloads (faster, higher rate limits)
try:
    from dotenv import load_dotenv
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("HuggingFace: Authenticated with HF_TOKEN")
    else:
        print("WARNING: HF_TOKEN not found in .env - using unauthenticated requests")
except ImportError:
    print("WARNING: python-dotenv not installed - using unauthenticated requests")

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import (
    CLIPS_DIR, TAGS_FILE, QWEN_MODEL_ID, EMBEDDINGS_FILE,
    ensure_clips_exist, get_all_clips, check_gpu,
    DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS,
    setup_ram_cache, cleanup_ram_cache, get_deduplicated_clips, restore_original_path,
    check_output_exists
)

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    HAS_QWEN = True
except ImportError as e:
    HAS_QWEN = False
    print(f"WARNING: Qwen3-VL dependencies not available: {e}")
    print("Requires: transformers>=4.57.0")
    print("Install: pip install -U transformers qwen-vl-utils")
    print("Will use dummy tags for testing.")

# Structured tagging prompt
TAG_PROMPT = """Analyze this video clip and output ONLY a JSON object with these fields:

{
  "scene_type": "market|temple|junction|lane|highway|residential|commercial|metro|hilltown",
  "crowd_density": "low|med|high",
  "traffic_density": "low|med|high",
  "time_of_day": "morning|afternoon|evening|night",
  "weather": "clear|cloudy|rain|fog",
  "notable_objects": ["list", "of", "objects"]
}

Output ONLY the JSON, no explanation."""

# Batch config for Qwen3-VL (more VRAM-intensive than V-JEPA)
QWEN_DEFAULT_BATCH_SIZE = 4  # VLM uses more VRAM per sample
QWEN_DEFAULT_NUM_WORKERS = 4


class VideoClipDataset(Dataset):
    """PyTorch Dataset for video clips (for VLM tagging)."""

    def __init__(self, clip_paths: list):
        self.clip_paths = [Path(p) for p in clip_paths]

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        return {"path": str(self.clip_paths[idx]), "idx": idx}


def tag_batch_with_qwen(model, processor, batch_paths: list, device: str) -> list:
    """
    Generate structured tags for a batch of video clips using Qwen3-VL.

    Args:
        model: Qwen3-VL model
        processor: Qwen3-VL processor
        batch_paths: List of video paths
        device: Device to use

    Returns:
        List of tag dictionaries
    """
    # Process each video in the batch
    batch_tags = []

    for video_path in batch_paths:
        try:
            # Prepare messages for Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": str(video_path), "max_pixels": 360 * 420, "fps": 1.0},
                        {"type": "text", "text": TAG_PROMPT},
                    ],
                }
            ]

            # Process input
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            # Parse JSON from output
            tags = parse_json_output(output_text, video_path)
            batch_tags.append(tags)

        except Exception as e:
            # Fallback to dummy tag on error
            batch_tags.append(get_dummy_tag(video_path))

    return batch_tags


def parse_json_output(output_text: str, video_path) -> dict:
    """Parse JSON from model output."""
    try:
        start = output_text.find('{')
        end = output_text.rfind('}') + 1
        if start != -1 and end > start:
            json_str = output_text[start:end]
            tags = json.loads(json_str)
            tags["clip_path"] = str(video_path)
            return tags
    except json.JSONDecodeError:
        pass
    return get_dummy_tag(video_path)


def get_dummy_tag(video_path) -> dict:
    """Generate dummy tag based on folder name."""
    video_path = Path(video_path)
    parent_name = video_path.parent.name.lower()

    # Infer scene type from folder name
    scene_type = "lane"
    if "market" in parent_name:
        scene_type = "market"
    elif "temple" in parent_name:
        scene_type = "temple"
    elif "junction" in parent_name:
        scene_type = "junction"
    elif "metro" in parent_name:
        scene_type = "metro"
    elif "hilltown" in parent_name:
        scene_type = "hilltown"

    return {
        "clip_path": str(video_path),
        "scene_type": scene_type,
        "crowd_density": "med",
        "traffic_density": "med",
        "time_of_day": "afternoon",
        "weather": "clear",
        "notable_objects": ["pedestrian", "vehicle"]
    }


def generate_plot_from_tags(tags_file: Path):
    """Generate scene distribution plot from existing tags.json."""
    import matplotlib.pyplot as plt
    from utils.config import OUTPUTS_DIR

    if not tags_file.exists():
        print(f"ERROR: {tags_file} not found. Run tagging first.")
        sys.exit(1)

    with open(tags_file, 'r') as f:
        all_tags = json.load(f)

    print(f"Loaded {len(all_tags)} tags from {tags_file}")

    # Count scene types
    scene_counts = {}
    for t in all_tags:
        st = t.get("scene_type", "unknown")
        scene_counts[st] = scene_counts.get(st, 0) + 1

    print(f"\nScene type distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: -x[1]):
        print(f"  {scene}: {count}")

    # Generate plot
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    scenes = sorted(scene_counts.keys(), key=lambda x: scene_counts[x], reverse=True)
    counts = [scene_counts[s] for s in scenes]
    colors = plt.cm.tab10(range(len(scenes)))

    bars = ax.bar(scenes, counts, color=colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Scene Type')
    ax.set_ylabel('Number of Clips')
    ax.set_title(f'Qwen3-VL Scene Type Distribution (n={len(all_tags)} clips)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plot_png = OUTPUTS_DIR / "m04_scene_distribution.png"
    plot_pdf = OUTPUTS_DIR / "m04_scene_distribution.pdf"
    plt.savefig(plot_png, dpi=150)
    plt.savefig(plot_pdf)
    plt.close()

    print(f"\nSaved: {plot_png}")
    print(f"Saved: {plot_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Generate structured tags using Qwen3-VL (GPU-optimized)")
    parser.add_argument("--SANITY", action="store_true", help="Process 5 clips only")
    parser.add_argument("--FULL", action="store_true", help="Process all deduplicated clips")
    parser.add_argument("--dummy", action="store_true", help="Use dummy tags (no model)")
    parser.add_argument("--all-clips", action="store_true", help="Tag all clips (ignore deduplication)")
    parser.add_argument("--batch-size", type=int, default=QWEN_DEFAULT_BATCH_SIZE, help="Batch size (default: 4)")
    parser.add_argument("--num-workers", type=int, default=QWEN_DEFAULT_NUM_WORKERS, help="DataLoader workers (default: 4)")
    parser.add_argument("--no-ram-cache", action="store_true", help="Disable RAM cache (/dev/shm)")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    # Check if tags.json already exists
    if TAGS_FILE.exists():
        if not check_output_exists([TAGS_FILE], "tags"):
            generate_plot_from_tags(TAGS_FILE)
            return

    # Check GPU
    check_gpu()
    device = "cuda"

    # Ensure clips exist
    if not ensure_clips_exist():
        print(f"ERROR: No clips available. Run m02_scene_detect.py or check HuggingFace access.")
        sys.exit(1)

    # Get clips to tag (deduplicated from m03, or all if --all-clips)
    if args.all_clips:
        all_clips = get_all_clips()
        print(f"Using ALL clips: {len(all_clips)}")
    else:
        all_clips = get_deduplicated_clips()

    if not all_clips:
        print(f"ERROR: No clips found")
        sys.exit(1)

    # Limit clips for sanity mode
    if args.SANITY:
        all_clips = all_clips[:5]
        args.batch_size = min(args.batch_size, 2)
        args.num_workers = min(args.num_workers, 2)
        print(f"SANITY MODE: Processing {len(all_clips)} clips")
    else:
        print(f"FULL MODE: Processing {len(all_clips)} clips")

    # Setup RAM cache (copy clips to /dev/shm for faster I/O)
    use_ram_cache = not args.no_ram_cache and not args.SANITY
    clip_paths, ram_cache_enabled = setup_ram_cache(all_clips, use_cache=use_ram_cache, cache_subdir="qwen_clips")

    # Load model or use dummy
    model = None
    processor = None

    if not args.dummy and HAS_QWEN:
        print(f"\nLoading model: {QWEN_MODEL_ID}")
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print("Ensure transformers>=4.57.0: pip install -U transformers")
            print("Falling back to dummy tags")
            model = None
    else:
        print("\nUsing dummy tags (--dummy flag or missing dependencies)")

    # Print batch config
    print(f"\n=== Batch Processing Config ===")
    print(f"batch_size:  {args.batch_size}")
    print(f"num_workers: {args.num_workers}")
    print(f"ram_cache:   {'enabled' if ram_cache_enabled else 'disabled'}")

    # Create dataset and dataloader
    dataset = VideoClipDataset(clip_paths)

    def collate_fn(batch):
        return {"paths": [b["path"] for b in batch], "indices": [b["idx"] for b in batch]}

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Generate tags with progress bar
    all_tags = []
    start_time = time.time()

    pbar = tqdm(dataloader, desc="Tagging clips", unit="batch")
    for batch in pbar:
        batch_paths = batch["paths"]

        if model is not None:
            tags_batch = tag_batch_with_qwen(model, processor, batch_paths, device)
        else:
            tags_batch = [get_dummy_tag(p) for p in batch_paths]

        # Restore original paths if using RAM cache
        for tags in tags_batch:
            if ram_cache_enabled:
                orig_path = restore_original_path(Path(tags["clip_path"]), CLIPS_DIR)
                tags["clip_path"] = orig_path
            all_tags.append(tags)

        elapsed = time.time() - start_time
        throughput = len(all_tags) / elapsed if elapsed > 0 else 0
        pbar.set_postfix({
            "clips": len(all_tags),
            "clips/s": f"{throughput:.2f}",
            "scene": tags_batch[-1].get('scene_type', '?')[:8]
        })

    elapsed = time.time() - start_time

    # Cleanup RAM cache
    if ram_cache_enabled:
        cleanup_ram_cache(cache_subdir="qwen_clips")

    # Save tags
    with open(TAGS_FILE, 'w') as f:
        json.dump(all_tags, f, indent=2)

    # Summary
    print(f"\n=== TAGGING COMPLETE ===")
    print(f"Saved: {TAGS_FILE}")
    print(f"Total clips tagged: {len(all_tags)}")
    print(f"Time: {elapsed:.1f}s ({len(all_tags)/elapsed:.2f} clips/sec)")

    # Scene type distribution
    scene_counts = {}
    for t in all_tags:
        st = t.get("scene_type", "unknown")
        scene_counts[st] = scene_counts.get(st, 0) + 1

    print(f"\nScene type distribution:")
    for scene, count in sorted(scene_counts.items(), key=lambda x: -x[1]):
        print(f"  {scene}: {count}")

    # Generate scene distribution plot
    try:
        import matplotlib.pyplot as plt
        from utils.config import OUTPUTS_DIR

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        scenes = sorted(scene_counts.keys(), key=lambda x: scene_counts[x], reverse=True)
        counts = [scene_counts[s] for s in scenes]
        colors = plt.cm.tab10(range(len(scenes)))

        bars = ax.bar(scenes, counts, color=colors)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Scene Type')
        ax.set_ylabel('Number of Clips')
        ax.set_title(f'Qwen3-VL Scene Type Distribution (n={len(all_tags)} clips)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plot_png = OUTPUTS_DIR / "m04_scene_distribution.png"
        plot_pdf = OUTPUTS_DIR / "m04_scene_distribution.pdf"
        plt.savefig(plot_png, dpi=150)
        plt.savefig(plot_pdf)
        plt.close()
        print(f"\nSaved: {plot_png}")
    except ImportError:
        print("WARNING: matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
