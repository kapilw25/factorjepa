"""
Object detection from scene images using Qwen3-VL-32B (8-bit).

Commands:
    python -u src/m01_scene_understanding.py --sanity 2>&1 | tee logs/m01_sanity.log
    python -u src/m01_scene_understanding.py --full --input_dir data/images 2>&1 | tee logs/m01_full.log
    python -u src/m01_scene_understanding.py --metadata_only --input_dir data/images 2>&1 | tee logs/m01_metadata.log
"""

import os
import sys
import json
import re
import gc
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables from .env (before any HF/API calls)
load_dotenv()

import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.io_handler import IOHandler
from utils.image_utils import find_image_pairs, resize_image

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-32B-Instruct"  # Use Qwen2.5-VL-32B (latest stable)
IMAGE_SIZE = (1024, 1024)
CONFIDENCE_THRESHOLD = 0.7
DEFAULT_INPUT_DIR = "Literature/Prev_work2/dataset/3DReasoningProject_images"


# ─────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────
class SceneUnderstanding:
    """Qwen2.5-VL-32B for object detection + scene description."""

    def __init__(self, model_name: str = MODEL_NAME, image_size: tuple = IMAGE_SIZE):
        assert torch.cuda.is_available(), "GPU required"
        self.device = "cuda"
        self.image_size = image_size

        print(f"[m01] Loading {model_name}...")

        # 8-bit quantization for A100-80GB (~40GB VRAM)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load model with quantization
        # Use Qwen2_5_VL (not Qwen2VL) for Qwen2.5-VL models
        from transformers import Qwen2_5_VLForConditionalGeneration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        print(f"[m01] Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def analyze_scene(self, isometric_path: str, topdown_path: str) -> Dict:
        """
        Analyze scene from isometric and top-down views.

        Args:
            isometric_path: Path to isometric view image
            topdown_path: Path to top-down view image

        Returns:
            Dict with objects, scene_type, layout_description
        """
        # Prepare images
        img_iso = resize_image(isometric_path, self.image_size)
        img_top = resize_image(topdown_path, self.image_size)

        # Construct message with two images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_iso},
                    {"type": "image", "image": img_top},
                    {"type": "text", "text": self._get_prompt()}
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False
            )

        # Decode only generated tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return self._parse_json_response(response)

    def _get_prompt(self) -> str:
        """Get the prompt for scene analysis."""
        return '''Analyze these two views of the SAME indoor/outdoor scene:
- Image 1: Isometric (3D perspective) view
- Image 2: Top-down (bird's eye) view

Your task: Identify ALL objects visible in BOTH images that could serve as navigation landmarks for a robot.

Output ONLY valid JSON with this exact structure:
{
  "objects": [
    {"name": "bench", "confidence": 0.95},
    {"name": "tree", "confidence": 0.92},
    {"name": "fountain", "confidence": 0.88}
  ],
  "scene_type": "outdoor_mall",
  "layout_description": "Open plaza with central fountain, benches along walkway, trees providing shade"
}

Rules:
- List objects that appear in BOTH views (cross-validated)
- Use simple, common object names (bench, table, chair, tree, fountain, kiosk, etc.)
- Confidence 0.0-1.0 based on visibility and certainty
- scene_type: outdoor_mall, office, retail, residential, park, street, or other
- layout_description: Brief navigation-relevant description (1-2 sentences)

Output ONLY the JSON, no other text.'''

    def _parse_json_response(self, text: str) -> Dict:
        """Extract JSON from model response."""
        # Try to find JSON block
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Return empty structure if parsing fails
        return {
            "objects": [],
            "scene_type": "unknown",
            "layout_description": "",
            "raw_response": text  # Keep raw for debugging
        }

    def filter_objects(self, objects: List[Dict], threshold: float = CONFIDENCE_THRESHOLD) -> List[Dict]:
        """Filter low-confidence objects and deduplicate."""
        seen = set()
        filtered = []
        for obj in objects:
            name = obj.get("name", "").lower().strip()
            confidence = obj.get("confidence", 0.0)

            if confidence >= threshold and name and name not in seen:
                seen.add(name)
                filtered.append(obj)

        return filtered

    def unload(self) -> None:
        """Free GPU memory for next model."""
        print("[m01] Unloading model...")
        del self.model
        del self.processor
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[m01] VRAM after unload: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Scene understanding using Qwen2.5-VL-32B")
    parser.add_argument("--sanity", action="store_true", help="Run on 1 sample only")
    parser.add_argument("--full", action="store_true", help="Run on all images")
    parser.add_argument("--metadata_only", action="store_true", help="Only list found image pairs")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR,
                        help="Directory containing image pairs")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for results")
    args = parser.parse_args()

    # Find image pairs
    print(f"[m01] Scanning: {args.input_dir}")
    pairs = find_image_pairs(args.input_dir)
    print(f"[m01] Found {len(pairs)} image pairs")

    if not pairs:
        print("[m01] ERROR: No image pairs found. Check input directory.")
        return

    # List pairs
    for i, (scene_id, paths) in enumerate(pairs.items()):
        print(f"  [{i+1}] {scene_id}")

    # Metadata only mode - just list pairs and exit
    if args.metadata_only:
        print("\n[m01] Metadata only mode - exiting without model inference")
        return

    # Initialize IO handler
    io = IOHandler(args.output_dir)

    # Determine samples to process
    if args.sanity:
        scene_ids = [list(pairs.keys())[0]]
        print(f"\n[m01] Sanity mode: processing 1 sample")
    elif args.full:
        scene_ids = list(pairs.keys())
        print(f"\n[m01] Full mode: processing {len(scene_ids)} samples")
    else:
        print("\n[m01] No mode specified. Use --sanity or --full")
        return

    # Load model
    analyzer = SceneUnderstanding()

    # Process each scene
    for i, scene_id in enumerate(scene_ids):
        paths = pairs[scene_id]
        print(f"\n[m01] [{i+1}/{len(scene_ids)}] Processing: {scene_id}")

        try:
            # Analyze scene
            scene_data = analyzer.analyze_scene(
                paths["isometric"],
                paths["topdown"]
            )

            # Filter objects
            if scene_data.get("objects"):
                original_count = len(scene_data["objects"])
                scene_data["objects"] = analyzer.filter_objects(scene_data["objects"])
                print(f"  Objects: {original_count} detected, {len(scene_data['objects'])} after filtering")

            # Save results
            io.save_scene(scene_id, paths, scene_data)
            print(f"  Scene type: {scene_data.get('scene_type', 'unknown')}")
            print(f"  Saved to: outputs/m01_scenes/{scene_id}.json")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Unload model
    analyzer.unload()
    print("\n[m01] Complete!")


if __name__ == "__main__":
    main()
