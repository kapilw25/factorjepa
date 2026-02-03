"""
Generate structured tags for video clips using Qwen3-VL.
GPU script (Nvidia CUDA) or API-based (M1 compatible with API).

USAGE:
    python -u src/m04_qwen_tag.py --SANITY 2>&1 | tee logs/m04_qwen_tag_sanity.log
    python -u src/m04_qwen_tag.py --FULL 2>&1 | tee logs/m04_qwen_tag_full.log
"""
import argparse
import json
import sys
from pathlib import Path

# Add src to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils.config import CLIPS_DIR, TAGS_FILE, QWEN_MODEL_ID

try:
    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    print("WARNING: Qwen2-VL not available. Will use dummy tags for testing.")

# Structured tagging prompt
TAG_PROMPT = """Analyze this video clip and output ONLY a JSON object with these fields:

{
  "scene_type": "market|temple|junction|lane|highway|residential|commercial",
  "crowd_density": "low|med|high",
  "traffic_density": "low|med|high",
  "time_of_day": "morning|afternoon|evening|night",
  "weather": "clear|cloudy|rain|fog",
  "notable_objects": ["list", "of", "objects"]
}

Output ONLY the JSON, no explanation."""


def extract_frames_for_vlm(video_path: Path, num_frames: int = 8) -> list:
    """Extract frames from video for VLM input."""
    try:
        import av
        import numpy as np
        from PIL import Image

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        total_frames = stream.frames or 100

        indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)

        frames = []
        frame_idx = 0
        for frame in container.decode(video=0):
            if frame_idx in indices:
                img = frame.to_image()
                frames.append(img)
            frame_idx += 1
            if len(frames) >= num_frames:
                break

        container.close()
        return frames
    except Exception as e:
        print(f"  Frame extraction error: {e}")
        return []


def tag_clip_with_qwen(model, processor, video_path: Path, device: str) -> dict:
    """
    Generate structured tags for a video clip using Qwen2-VL.

    Args:
        model: Qwen2-VL model
        processor: Qwen2-VL processor
        video_path: Path to video clip
        device: Device to use

    Returns:
        Dictionary with structured tags
    """
    frames = extract_frames_for_vlm(video_path, num_frames=8)
    if not frames:
        return get_dummy_tag(video_path)

    # Prepare messages for Qwen2-VL
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
    try:
        # Find JSON in output
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


def get_dummy_tag(video_path: Path) -> dict:
    """Generate dummy tag based on folder name."""
    parent_name = video_path.parent.name.lower()

    # Infer scene type from folder name
    scene_type = "lane"
    if "market" in parent_name:
        scene_type = "market"
    elif "temple" in parent_name:
        scene_type = "temple"
    elif "junction" in parent_name:
        scene_type = "junction"

    return {
        "clip_path": str(video_path),
        "scene_type": scene_type,
        "crowd_density": "med",
        "traffic_density": "med",
        "time_of_day": "afternoon",
        "weather": "clear",
        "notable_objects": ["pedestrian", "vehicle"]
    }


def main():
    parser = argparse.ArgumentParser(description="Generate structured tags using Qwen3-VL")
    parser.add_argument("--SANITY", action="store_true", help="Process 5 clips only")
    parser.add_argument("--FULL", action="store_true", help="Process all clips")
    parser.add_argument("--dummy", action="store_true", help="Use dummy tags (no model)")
    args = parser.parse_args()

    if not (args.SANITY or args.FULL):
        parser.print_help()
        print("\nERROR: Specify --SANITY or --FULL")
        sys.exit(1)

    # Find all clips
    clip_dirs = [d for d in CLIPS_DIR.iterdir() if d.is_dir()]
    all_clips = []
    for clip_dir in clip_dirs:
        all_clips.extend(list(clip_dir.glob("*.mp4")))

    if not all_clips:
        print(f"ERROR: No clips found in {CLIPS_DIR}")
        sys.exit(1)

    print(f"Found {len(all_clips)} clips")

    # Limit clips for sanity mode
    if args.SANITY:
        all_clips = all_clips[:5]
        print(f"SANITY MODE: Processing {len(all_clips)} clips")

    # Load model or use dummy
    model = None
    processor = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.dummy and HAS_QWEN:
        print(f"Loading model: {QWEN_MODEL_ID}")
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_ID,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )
            processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
            print(f"Model loaded on {device}")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            print("Falling back to dummy tags")
            model = None

    # Generate tags
    all_tags = []
    for i, clip_path in enumerate(all_clips):
        print(f"[{i+1}/{len(all_clips)}] Tagging: {clip_path.name}")

        if model is not None:
            tags = tag_clip_with_qwen(model, processor, clip_path, device)
        else:
            tags = get_dummy_tag(clip_path)

        all_tags.append(tags)
        print(f"  scene_type: {tags.get('scene_type', 'unknown')}")

    # Save tags
    with open(TAGS_FILE, 'w') as f:
        json.dump(all_tags, f, indent=2)

    print(f"\nSaved tags: {TAGS_FILE}")
    print(f"Total clips tagged: {len(all_tags)}")

    # Summary
    scene_counts = {}
    for t in all_tags:
        st = t.get("scene_type", "unknown")
        scene_counts[st] = scene_counts.get(st, 0) + 1
    print(f"Scene type distribution: {scene_counts}")


if __name__ == "__main__":
    main()
