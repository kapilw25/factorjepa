"""
Convert existing .txt instructions to JSON format for m03 evaluation.

Commands:
    python -u src/m00_data_prep.py --sanity 2>&1 | tee logs/m00_sanity.log
    python -u src/m00_data_prep.py --full 2>&1 | tee logs/m00_full.log
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.io_handler import IOHandler
from utils.image_utils import find_image_pairs

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
IMAGE_DIR = "Literature/Prev_work2/dataset/3DReasoningProject_images"
INSTRUCTION_DIR = "Literature/Prev_work2/dataset/instructions"


# ─────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────
def parse_instruction_line(line: str) -> Tuple[int, Dict]:
    """
    Parse a single instruction line.

    Format: LEVEL_X | TASK: ... | START: ... | END: ... | INTERACT: ...

    Returns:
        Tuple of (level, instruction_dict)
    """
    line = line.strip()
    if not line:
        return None, None

    # Extract level
    level_match = re.match(r'LEVEL_(\d+)', line)
    if not level_match:
        return None, None
    level = int(level_match.group(1))

    # Extract fields using regex
    task_match = re.search(r'TASK:\s*([^|]+)', line)
    start_match = re.search(r'START:\s*([^|]+)', line)
    end_match = re.search(r'END:\s*([^|]+)', line)
    interact_match = re.search(r'INTERACT:\s*(.+)$', line)

    task = task_match.group(1).strip() if task_match else ""
    start = start_match.group(1).strip() if start_match else ""
    end = end_match.group(1).strip() if end_match else ""
    interact_str = interact_match.group(1).strip() if interact_match else "none"

    # Parse interact list
    if interact_str.lower() == "none":
        interact = []
    else:
        interact = [x.strip() for x in interact_str.split(",")]

    return level, {
        "task": task,
        "start": start,
        "end": end,
        "interact": interact
    }


def parse_instruction_file(filepath: str) -> Dict[str, List[Dict]]:
    """
    Parse instruction .txt file into structured dict.

    Returns:
        Dict with level_1, level_2, level_3 lists
    """
    instructions = {
        "level_1": [],
        "level_2": [],
        "level_3": []
    }

    with open(filepath, "r") as f:
        for line in f:
            level, instr = parse_instruction_line(line)
            if level and instr:
                instructions[f"level_{level}"].append(instr)

    return instructions


def find_matching_instruction_file(scene_id: str, instruction_dir: str) -> str:
    """Find instruction .txt file for a scene_id."""
    instruction_path = Path(instruction_dir)

    # Try exact match
    exact_match = instruction_path / f"{scene_id}.txt"
    if exact_match.exists():
        return str(exact_match)

    # List available files for debugging
    available = list(instruction_path.glob("*.txt"))
    return None


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Convert .txt instructions to JSON format")
    parser.add_argument("--sanity", action="store_true", help="Process 1 sample only")
    parser.add_argument("--full", action="store_true", help="Process all available samples")
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR, help="Image directory")
    parser.add_argument("--instruction_dir", type=str, default=INSTRUCTION_DIR, help="Instruction directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    # Initialize IO handler
    io = IOHandler(args.output_dir)

    # Find image pairs
    pairs = find_image_pairs(args.image_dir)
    print(f"[m00] Found {len(pairs)} image pairs")

    if not pairs:
        print("[m00] ERROR: No image pairs found")
        return

    # List available instruction files
    instruction_path = Path(args.instruction_dir)
    instruction_files = {f.stem: f for f in instruction_path.glob("*.txt")}
    print(f"[m00] Found {len(instruction_files)} instruction files")

    # Find matching pairs
    matched = []
    for scene_id in pairs.keys():
        if scene_id in instruction_files:
            matched.append(scene_id)

    print(f"[m00] Matched {len(matched)} scenes with instructions")

    if not matched:
        print("[m00] ERROR: No matching scenes found")
        print(f"[m00] Image scene_ids: {list(pairs.keys())[:5]}...")
        print(f"[m00] Instruction files: {list(instruction_files.keys())[:5]}...")
        return

    # Select samples
    if args.sanity:
        matched = [matched[0]]
        print(f"[m00] Sanity mode: processing 1 sample")
    elif args.full:
        print(f"[m00] Full mode: processing {len(matched)} samples")
    else:
        print("[m00] No mode specified. Use --sanity or --full")
        return

    # Process each matched scene
    for i, scene_id in enumerate(matched):
        print(f"\n[m00] [{i+1}/{len(matched)}] Processing: {scene_id}")

        # Get paths
        paths = pairs[scene_id]
        instruction_file = instruction_files[scene_id]

        # Parse instructions
        instructions = parse_instruction_file(str(instruction_file))
        counts = [len(instructions.get(f"level_{l}", [])) for l in [1, 2, 3]]
        print(f"  Instructions: L1={counts[0]}, L2={counts[1]}, L3={counts[2]}")

        # Create minimal scene data (since we don't have VLM output)
        scene_data = {
            "objects": [],  # Empty - we don't have VLM detection
            "scene_type": "outdoor",  # Default
            "layout_description": "Scene from existing dataset"
        }

        # Save to JSON format
        io.save_scene(scene_id, paths, scene_data)
        io.save_instructions(scene_id, instructions)

        print(f"  Saved: m01_scenes/{scene_id}.json, m02_instructions/{scene_id}.json")

    print(f"\n[m00] Complete! Processed {len(matched)} scenes")
    print(f"[m00] Ready for: python src/m03_evaluator.py --sanity")


if __name__ == "__main__":
    main()
