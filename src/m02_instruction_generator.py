"""
Robot navigation instruction generation using Llama-3.1-70B-Instruct (8-bit).

Commands:
    python -u src/m02_instruction_generator.py --sanity 2>&1 | tee logs/m02_sanity.log
    python -u src/m02_instruction_generator.py --full 2>&1 | tee logs/m02_full.log
"""

import os
import sys
import re
import gc
import argparse
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv

# Load environment variables from .env (before any HF/API calls)
load_dotenv()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.io_handler import IOHandler

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"
NUM_PER_LEVEL = 3
BANNED_TERMS: Set[str] = {"left", "right", "front", "back", "center", "middle", "north", "south", "east", "west"}


# ─────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────
class InstructionGenerator:
    """Llama-3.1-70B for generating robot navigation tasks."""

    def __init__(self, model_name: str = MODEL_NAME):
        assert torch.cuda.is_available(), "GPU required"

        print(f"[m02] Loading {model_name}...")

        # 8-bit quantization for A100-80GB (~70GB VRAM)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_memory={0: "75GiB", "cpu": "0GiB"}  # Force GPU-only, no CPU offload
        )

        print(f"[m02] Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def generate_instructions(self, scene_data: Dict, num_per_level: int = NUM_PER_LEVEL) -> Dict:
        """
        Generate robot navigation instructions based on scene analysis.

        Args:
            scene_data: Dict from m01 with objects, scene_type, layout_description
            num_per_level: Number of instructions per difficulty level

        Returns:
            Dict with level_1, level_2, level_3 instruction lists
        """
        # Extract object names
        objects = [obj["name"] for obj in scene_data.get("objects", [])]
        if not objects:
            print("  WARNING: No objects found in scene data")
            return {"level_1": [], "level_2": [], "level_3": []}

        objects_str = ", ".join(objects)

        # Build prompts
        system_prompt = self._get_system_prompt(objects_str)
        user_prompt = self._get_user_prompt(scene_data, num_per_level)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Decode only generated tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        return self._parse_instructions(response, objects)

    def _get_system_prompt(self, objects_str: str) -> str:
        """Get system prompt with valid objects."""
        return f"""You are a robot task instruction generator for indoor/outdoor navigation.

VALID_OBJECTS (use ONLY these): {objects_str}

RULES:
1. Use ONLY object names from VALID_OBJECTS for START and END locations
2. Format locations as: "Near <object>" (e.g., "Near bench", "Near fountain")
3. NEVER use directional terms: left, right, front, back, center, middle, north, south, east, west
4. Difficulty levels:
   - LEVEL_1: Simple navigation (point A to point B), INTERACT=none
   - LEVEL_2: Navigation with 1-2 object interactions (pick up, inspect, etc.)
   - LEVEL_3: Complex multi-step tasks with 3+ object interactions

OUTPUT FORMAT (strict, one instruction per line):
LEVEL_X | TASK: <task description> | START: Near <obj> | END: Near <obj> | INTERACT: <obj1, obj2> or none

Example valid outputs:
LEVEL_1 | TASK: Navigate from the bench area to the fountain | START: Near bench | END: Near fountain | INTERACT: none
LEVEL_2 | TASK: Pick up the package near the table and deliver it to the chair | START: Near table | END: Near chair | INTERACT: package, table
LEVEL_3 | TASK: Collect items from the shelf, check the display, and deliver to the counter | START: Near shelf | END: Near counter | INTERACT: shelf, display, counter, items"""

    def _get_user_prompt(self, scene_data: Dict, num_per_level: int) -> str:
        """Get user prompt with scene context."""
        scene_type = scene_data.get("scene_type", "unknown")
        layout = scene_data.get("layout_description", "No description available")

        return f"""Scene Type: {scene_type}
Layout: {layout}

Generate exactly {num_per_level} tasks for EACH difficulty level (total {num_per_level * 3} tasks).

Requirements:
- All START and END locations must use objects from VALID_OBJECTS
- Tasks should be realistic for the scene type
- Ensure variety in task types and object usage
- LEVEL_1: Pure navigation only
- LEVEL_2: Include 1-2 meaningful interactions
- LEVEL_3: Complex multi-step tasks

Generate the tasks now:"""

    def _parse_instructions(self, text: str, valid_objects: List[str]) -> Dict:
        """Parse and validate generated instructions."""
        result = {"level_1": [], "level_2": [], "level_3": []}

        # Normalize valid objects for comparison
        valid_objects_lower = {obj.lower().strip() for obj in valid_objects}

        # Pattern to match instruction lines
        pattern = r'LEVEL_(\d)\s*\|\s*TASK:\s*(.+?)\s*\|\s*START:\s*(.+?)\s*\|\s*END:\s*(.+?)\s*\|\s*INTERACT:\s*(.+?)(?:\n|$)'

        for match in re.finditer(pattern, text, re.IGNORECASE):
            level = int(match.group(1))
            task = match.group(2).strip()
            start = match.group(3).strip()
            end = match.group(4).strip()
            interact = match.group(5).strip()

            if level not in [1, 2, 3]:
                continue

            # Validate: check for banned terms
            full_text = f"{task} {start} {end}".lower()
            has_banned = any(term in full_text for term in BANNED_TERMS)
            if has_banned:
                continue

            # Extract object from "Near <object>" format
            start_obj = self._extract_object(start)
            end_obj = self._extract_object(end)

            # Parse interact objects
            if interact.lower() == "none":
                interact_list = []
            else:
                interact_list = [obj.strip() for obj in interact.split(",") if obj.strip()]

            instruction = {
                "task": task,
                "start": start,
                "end": end,
                "start_object": start_obj,
                "end_object": end_obj,
                "interact": interact_list
            }

            result[f"level_{level}"].append(instruction)

        return result

    def _extract_object(self, location: str) -> str:
        """Extract object name from 'Near <object>' format."""
        match = re.search(r'near\s+(.+)', location, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return location.strip()

    def unload(self) -> None:
        """Free GPU memory."""
        print("[m02] Unloading model...")
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[m02] VRAM after unload: {torch.cuda.memory_allocated() / 1e9:.1f} GB")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Instruction generation using Llama-3.1-70B")
    parser.add_argument("--sanity", action="store_true", help="Run on 1 sample only")
    parser.add_argument("--full", action="store_true", help="Run on all scenes from m01")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory (must contain m01_scenes/)")
    parser.add_argument("--num_per_level", type=int, default=NUM_PER_LEVEL,
                        help="Number of instructions per level")
    args = parser.parse_args()

    # Initialize IO handler
    io = IOHandler(args.output_dir)

    # Get available scenes from m01
    scene_ids = io.list_scenes()
    if not scene_ids:
        print("[m02] ERROR: No scenes found in outputs/m01_scenes/")
        print("[m02] Run m01_scene_understanding.py first")
        return

    print(f"[m02] Found {len(scene_ids)} scenes from m01")

    # Determine samples to process
    if args.sanity:
        scene_ids = [scene_ids[0]]
        print(f"[m02] Sanity mode: processing 1 sample")
    elif args.full:
        print(f"[m02] Full mode: processing {len(scene_ids)} samples")
    else:
        print("[m02] No mode specified. Use --sanity or --full")
        return

    # Load model
    generator = InstructionGenerator()

    # Process each scene
    for i, scene_id in enumerate(scene_ids):
        print(f"\n[m02] [{i+1}/{len(scene_ids)}] Processing: {scene_id}")

        try:
            # Load scene data from m01
            scene_data = io.load_scene(scene_id)
            objects = scene_data.get("objects", [])
            print(f"  Objects: {[obj['name'] for obj in objects]}")

            # Generate instructions
            instructions = generator.generate_instructions(scene_data, args.num_per_level)

            # Count generated instructions
            total = sum(len(instructions[f"level_{l}"]) for l in [1, 2, 3])
            print(f"  Generated: L1={len(instructions['level_1'])}, L2={len(instructions['level_2'])}, L3={len(instructions['level_3'])} (total={total})")

            # Save results
            io.save_instructions(scene_id, instructions)
            print(f"  Saved to: outputs/m02_instructions/{scene_id}.json")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Unload model
    generator.unload()
    print("\n[m02] Complete!")


if __name__ == "__main__":
    main()
