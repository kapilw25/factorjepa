"""
VLM-as-Judge evaluation using GPT-4o (baseline) and Prometheus-Vision-13B (scale).

Commands:
    python -u src/m03_evaluator.py --sanity 2>&1 | tee logs/m03_sanity.log
    python -u src/m03_evaluator.py --full 2>&1 | tee logs/m03_full.log
    python -u src/m03_evaluator.py --plot_only 2>&1 | tee logs/m03_plot.log
    python -u src/m03_evaluator.py --metadata_only 2>&1 | tee logs/m03_metadata.log
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables from .env (before any API calls)
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.io_handler import IOHandler
from utils.image_utils import encode_image_base64
from utils.plotting import EvaluationVisualizer

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
OPENAI_MODEL = "gpt-4o"
PROMETHEUS_MODEL = "prometheus-eval/prometheus-vision-13b-v1.0"

EVALUATION_RUBRIC = """You are a STRICT evaluator for robot navigation instructions. Be critical and demanding.
A score of 5 should be RARE and only given for perfect instructions. Most scores should be 1-3.

Rate each criterion from 1-5:

1. Object Accuracy (1-5): Are ALL mentioned objects (START, END, INTERACT) clearly identifiable?
   - 5: Every single object is clearly visible AND correctly named in BOTH views (RARE)
   - 4: All objects visible, minor naming variations acceptable
   - 3: Most objects visible, but 1 object is vague or generic (e.g., "near unknown")
   - 2: Multiple objects are vague, generic, or only partially visible
   - 1: Objects don't exist in scene OR completely wrong object names

2. Spatial Coherence (1-5): Is the navigation path physically logical and optimal?
   - 5: Perfect path with precise spatial references, no ambiguity (RARE)
   - 4: Logical path, minor spatial ambiguity
   - 3: Path is possible but inefficient or has unclear waypoints
   - 2: Path has significant spatial errors or impossible shortcuts
   - 1: Path is physically impossible or completely nonsensical

3. Task Clarity (1-5): Is the instruction unambiguous for a robot to parse?
   - 5: Machine-parseable, zero ambiguity, specific action verbs (RARE)
   - 4: Clear with minor interpretation needed
   - 3: Understandable but requires human-level reasoning
   - 2: Ambiguous wording, multiple interpretations possible
   - 1: Confusing, contradictory, or unparseable

4. Difficulty Alignment (1-5): Does complexity match the stated level?
   Level 1: Single object, simple navigation
   Level 2: Two objects, intermediate reasoning
   Level 3: Three+ objects, multi-step reasoning
   - 5: Perfect match with appropriate challenge (RARE)
   - 4: Mostly aligned, slight over/under complexity
   - 3: Somewhat misaligned
   - 2: Clearly wrong complexity for the level
   - 1: Completely misaligned (e.g., trivial task at Level 3)

5. Executability (1-5): Can a real robot with standard sensors execute this?
   - 5: Fully executable with clear start/end positions and actions (RARE)
   - 4: Executable with minor assumptions
   - 3: Partially executable, needs additional context
   - 2: Difficult to execute, vague positions or actions
   - 1: Impossible (e.g., "near unknown" positions, non-existent objects)

IMPORTANT: Be strict. Generic instructions like "Move to X | START: Near unknown | END: Near unknown" should score LOW (1-2) on most criteria.

Output ONLY valid JSON:
{"object_accuracy": X, "spatial_coherence": X, "task_clarity": X, "difficulty_alignment": X, "executability": X, "reasoning": "Brief critical explanation"}"""


# ─────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────
class InstructionEvaluator:
    """Evaluate instruction quality using VLM-as-Judge."""

    def __init__(self, use_openai: bool = True):
        """
        Initialize evaluator.

        Args:
            use_openai: If True, use OpenAI GPT-4o API. If False, use local Prometheus.
        """
        self.use_openai = use_openai

        if use_openai:
            # Check for API key
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            from openai import OpenAI
            self.client = OpenAI()
            self.model = OPENAI_MODEL
            print(f"[m03] Using OpenAI API: {self.model}")
        else:
            # Load Prometheus-Vision-13B locally
            self._load_prometheus()
            print(f"[m03] Using local Prometheus: {PROMETHEUS_MODEL}")

    def _load_prometheus(self) -> None:
        """Load Prometheus-Vision-13B model locally."""
        raise NotImplementedError("Prometheus local loading not implemented yet. Use --use_openai")

    def evaluate_instruction(
        self,
        isometric_path: str,
        topdown_path: str,
        instruction: Dict,
        level: int
    ) -> Dict:
        """
        Evaluate a single instruction against the scene images.

        Args:
            isometric_path: Path to isometric view image
            topdown_path: Path to top-down view image
            instruction: Dict with task, start, end, interact
            level: Difficulty level (1, 2, or 3)

        Returns:
            Dict with scores and reasoning
        """
        # Encode images
        img_iso_b64 = encode_image_base64(isometric_path)
        img_top_b64 = encode_image_base64(topdown_path)

        # Format instruction for evaluation
        instr_text = self._format_instruction(instruction, level)

        prompt = f"""You are evaluating a robot navigation instruction for the scene shown in these images.

Image 1: Isometric (3D perspective) view
Image 2: Top-down (bird's eye) view

INSTRUCTION TO EVALUATE (Level {level}):
{instr_text}

{EVALUATION_RUBRIC}"""

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_iso_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_top_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=500
        )

        return self._parse_scores(response.choices[0].message.content)

    def _format_instruction(self, instruction: Dict, level: int) -> str:
        """Format instruction dict as readable text."""
        if isinstance(instruction, dict):
            task = instruction.get("task", "N/A")
            start = instruction.get("start", "N/A")
            end = instruction.get("end", "N/A")
            interact = instruction.get("interact", [])
            interact_str = ", ".join(interact) if interact else "none"
            return f"LEVEL_{level} | TASK: {task} | START: {start} | END: {end} | INTERACT: {interact_str}"
        return str(instruction)

    def evaluate_scene(self, scene_id: str, io: IOHandler) -> Dict:
        """
        Evaluate all instructions for a scene.

        Args:
            scene_id: Scene identifier
            io: IOHandler instance

        Returns:
            Dict with scores for each level and averages
        """
        # Load scene and instructions
        scene_data = io.load_scene(scene_id)
        instructions = io.load_instructions(scene_id)

        isometric_path = scene_data["isometric_path"]
        topdown_path = scene_data["topdown_path"]

        results = {
            "scene_id": scene_id,
            "level_1": [],
            "level_2": [],
            "level_3": []
        }

        # Evaluate each instruction
        for instr in instructions:
            level = instr.get("level", 1)
            print(f"    Evaluating Level {level} instruction...")

            try:
                scores = self.evaluate_instruction(
                    isometric_path,
                    topdown_path,
                    instr,
                    level
                )
                scores["instruction"] = instr
                results[f"level_{level}"].append(scores)
            except Exception as e:
                print(f"    ERROR evaluating instruction: {e}")
                results[f"level_{level}"].append({
                    "error": str(e),
                    "instruction": instr
                })

        # Compute averages per level
        metrics = ["object_accuracy", "spatial_coherence", "task_clarity",
                   "difficulty_alignment", "executability"]

        for level in [1, 2, 3]:
            level_scores = results[f"level_{level}"]
            valid_scores = [s for s in level_scores if "error" not in s]

            if valid_scores:
                results[f"level_{level}_avg"] = {
                    metric: sum(s.get(metric, 0) for s in valid_scores) / len(valid_scores)
                    for metric in metrics
                }
                results[f"level_{level}_count"] = len(valid_scores)

        return results

    def _parse_scores(self, text: str) -> Dict:
        """Extract JSON scores from response."""
        # Try to find JSON block
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                scores = json.loads(match.group())
                # Validate score ranges
                for key in ["object_accuracy", "spatial_coherence", "task_clarity",
                           "difficulty_alignment", "executability"]:
                    if key in scores:
                        scores[key] = max(1, min(5, float(scores[key])))
                return scores
            except (json.JSONDecodeError, ValueError):
                pass

        # Return empty dict if parsing fails
        return {"raw_response": text, "parse_error": True}


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="VLM-as-Judge evaluation using GPT-4o")
    parser.add_argument("--sanity", action="store_true", help="Run on 1 sample only")
    parser.add_argument("--full", action="store_true", help="Run on all instructions from m02")
    parser.add_argument("--plot_only", action="store_true", help="Only generate plots from existing metrics.csv")
    parser.add_argument("--metadata_only", action="store_true", help="Only list available scenes")
    parser.add_argument("--no_plot", action="store_true", help="Skip plot generation after evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--use_prometheus", action="store_true",
                        help="Use local Prometheus instead of OpenAI (not implemented)")
    args = parser.parse_args()

    # Initialize IO handler
    io = IOHandler(args.output_dir)

    # Plot only mode - generate plots from existing metrics.csv
    if args.plot_only:
        print("[m03] Plot-only mode: generating research-grade plots...")
        try:
            viz = EvaluationVisualizer(args.output_dir)
            saved_files = viz.generate_all_plots()
            print(f"\n[m03] Generated {len(saved_files)} plots:")
            for f in saved_files:
                print(f"  • {f}")
        except FileNotFoundError as e:
            print(f"[m03] ERROR: {e}")
            print("[m03] Run evaluation first to generate metrics.csv")
        return

    # Get available scenes with instructions
    scene_ids = io.list_instructions()
    if not scene_ids:
        print("[m03] ERROR: No instructions found in outputs/m02_instructions/")
        print("[m03] Run m02_instruction_generator.py first")
        return

    print(f"[m03] Found {len(scene_ids)} scenes with instructions")

    # Metadata only mode
    if args.metadata_only:
        for scene_id in scene_ids:
            instructions = io.load_instructions(scene_id)
            counts = {}
            for instr in instructions:
                level = instr.get("level", "?")
                counts[level] = counts.get(level, 0) + 1
            print(f"  {scene_id}: {counts}")
        return

    # Determine samples to process
    if args.sanity:
        scene_ids = [scene_ids[0]]
        print(f"[m03] Sanity mode: processing 1 sample")
    elif args.full:
        print(f"[m03] Full mode: processing {len(scene_ids)} samples")
    else:
        print("[m03] No mode specified. Use --sanity, --full, --plot_only, or --metadata_only")
        return

    # Initialize evaluator
    use_openai = not args.use_prometheus
    evaluator = InstructionEvaluator(use_openai=use_openai)

    # Process each scene
    all_metrics = []
    for i, scene_id in enumerate(scene_ids):
        print(f"\n[m03] [{i+1}/{len(scene_ids)}] Evaluating: {scene_id}")

        try:
            # Evaluate all instructions for scene
            eval_results = evaluator.evaluate_scene(scene_id, io)

            # Save evaluation results
            io.save_evaluation(scene_id, eval_results)

            # Collect metrics for CSV
            metrics_row = io.flatten_metrics(scene_id, eval_results)
            all_metrics.append(metrics_row)

            # Print summary
            for level in [1, 2, 3]:
                avg = eval_results.get(f"level_{level}_avg", {})
                if avg:
                    obj_acc = avg.get("object_accuracy", "N/A")
                    print(f"    Level {level}: obj_acc={obj_acc:.2f}" if isinstance(obj_acc, float) else f"    Level {level}: obj_acc={obj_acc}")

            print(f"  Saved to: outputs/m03_evaluations/{scene_id}.json")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save aggregated metrics
    if all_metrics:
        io.save_metrics_csv(all_metrics)
        print(f"\n[m03] Metrics saved to: outputs/metrics.csv")

        # Generate plots unless --no_plot
        if not args.no_plot:
            print("\n[m03] Generating research-grade plots...")
            try:
                viz = EvaluationVisualizer(args.output_dir)
                saved_files = viz.generate_all_plots()
                print(f"[m03] Generated {len(saved_files)} plots in outputs/m03_plots/")
            except Exception as e:
                print(f"[m03] WARNING: Plot generation failed: {e}")

    print("\n[m03] Complete!")


if __name__ == "__main__":
    main()
