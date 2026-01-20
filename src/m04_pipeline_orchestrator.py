"""
Full pipeline orchestrator with sequential model loading for A100-80GB.

Commands:
    python -u src/m04_pipeline_orchestrator.py --sanity 2>&1 | tee logs/m04_sanity.log
    python -u src/m04_pipeline_orchestrator.py --full --input_dir data/images 2>&1 | tee logs/m04_full.log
    python -u src/m04_pipeline_orchestrator.py --eval_only 2>&1 | tee logs/m04_eval.log
    python -u src/m04_pipeline_orchestrator.py --skip_eval --full 2>&1 | tee logs/m04_no_eval.log

Memory Strategy:
    - Step 1: Load Qwen2.5-VL-32B (~40GB) → process → unload
    - Step 2: Load Llama-3.1-70B (~70GB) → process → unload
    - Step 3: GPT-4o API (no GPU needed)
"""

import os
import sys
import gc
import argparse
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables from .env (before any HF/API calls)
load_dotenv()

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.io_handler import IOHandler
from utils.image_utils import find_image_pairs

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR = "Literature/Prev_work2/dataset/3DReasoningProject_images"
DEFAULT_OUTPUT_DIR = "outputs"


# ─────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────
class PipelineOrchestrator:
    """Sequential loading: VLM → unload → LLM → unload → Eval (API)"""

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.io = IOHandler(output_dir)
        self.output_dir = output_dir

    def _print_gpu_status(self, label: str) -> None:
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[GPU] {label}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

    def run_step1_scene_understanding(
        self,
        pairs: dict,
        scene_ids: Optional[List[str]] = None
    ) -> None:
        """
        Step 1: Object detection with Qwen2.5-VL-32B

        Args:
            pairs: Dict of scene_id -> {isometric, topdown} paths
            scene_ids: Optional list of specific scenes to process
        """
        print("\n" + "=" * 60)
        print("[STEP 1] Object Detection with Qwen2.5-VL-32B")
        print("=" * 60)

        self._print_gpu_status("Before loading VLM")

        # Import and load model
        from m01_scene_understanding import SceneUnderstanding
        analyzer = SceneUnderstanding()

        self._print_gpu_status("After loading VLM")

        # Process scenes
        scene_ids = scene_ids or list(pairs.keys())
        for i, scene_id in enumerate(scene_ids):
            paths = pairs[scene_id]
            print(f"\n[m01] [{i+1}/{len(scene_ids)}] {scene_id}")

            try:
                scene_data = analyzer.analyze_scene(
                    paths["isometric"],
                    paths["topdown"]
                )
                scene_data["objects"] = analyzer.filter_objects(scene_data["objects"])

                # Save
                self.io.save_scene(scene_id, paths, scene_data)

                obj_count = len(scene_data.get("objects", []))
                scene_type = scene_data.get("scene_type", "unknown")
                print(f"  ✓ {obj_count} objects | {scene_type}")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")

        # Unload model
        analyzer.unload()
        self._print_gpu_status("After unloading VLM")

    def run_step2_instruction_generation(
        self,
        scene_ids: Optional[List[str]] = None,
        num_per_level: int = 3
    ) -> None:
        """
        Step 2: Instruction generation with Llama-3.1-70B

        Args:
            scene_ids: Optional list of specific scenes to process
            num_per_level: Number of instructions per difficulty level
        """
        print("\n" + "=" * 60)
        print("[STEP 2] Instruction Generation with Llama-3.1-70B")
        print("=" * 60)

        self._print_gpu_status("Before loading LLM")

        # Import and load model
        from m02_instruction_generator import InstructionGenerator
        generator = InstructionGenerator()

        self._print_gpu_status("After loading LLM")

        # Get scene IDs from m01 output
        available_scenes = self.io.list_scenes()
        if scene_ids:
            scene_ids = [s for s in scene_ids if s in available_scenes]
        else:
            scene_ids = available_scenes

        if not scene_ids:
            print("[m02] No scenes available. Run Step 1 first.")
            generator.unload()
            return

        # Process scenes
        for i, scene_id in enumerate(scene_ids):
            print(f"\n[m02] [{i+1}/{len(scene_ids)}] {scene_id}")

            try:
                scene_data = self.io.load_scene(scene_id)
                instructions = generator.generate_instructions(scene_data, num_per_level)

                # Save
                self.io.save_instructions(scene_id, instructions)

                counts = [len(instructions.get(f"level_{l}", [])) for l in [1, 2, 3]]
                print(f"  ✓ Generated: L1={counts[0]}, L2={counts[1]}, L3={counts[2]}")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")

        # Unload model
        generator.unload()
        self._print_gpu_status("After unloading LLM")

    def run_step3_evaluation(
        self,
        scene_ids: Optional[List[str]] = None
    ) -> None:
        """
        Step 3: VLM-as-Judge evaluation with GPT-4o API

        Args:
            scene_ids: Optional list of specific scenes to evaluate
        """
        print("\n" + "=" * 60)
        print("[STEP 3] Evaluation with GPT-4o API")
        print("=" * 60)

        # Check API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("[m03] ERROR: OPENAI_API_KEY not set")
            print("[m03] Skipping evaluation")
            return

        # Import evaluator (no GPU needed)
        from m03_evaluator import InstructionEvaluator
        evaluator = InstructionEvaluator(use_openai=True)

        # Get scene IDs from m02 output
        available_scenes = self.io.list_instructions()
        if scene_ids:
            scene_ids = [s for s in scene_ids if s in available_scenes]
        else:
            scene_ids = available_scenes

        if not scene_ids:
            print("[m03] No instructions available. Run Step 2 first.")
            return

        # Process scenes
        all_metrics = []
        for i, scene_id in enumerate(scene_ids):
            print(f"\n[m03] [{i+1}/{len(scene_ids)}] {scene_id}")

            try:
                eval_results = evaluator.evaluate_scene(scene_id, self.io)

                # Save
                self.io.save_evaluation(scene_id, eval_results)

                # Collect metrics
                all_metrics.append(self.io.flatten_metrics(scene_id, eval_results))

                # Print summary
                for level in [1, 2, 3]:
                    avg = eval_results.get(f"level_{level}_avg", {})
                    if avg:
                        obj_acc = avg.get("object_accuracy", 0)
                        exec_score = avg.get("executability", 0)
                        print(f"  L{level}: obj_acc={obj_acc:.1f}, exec={exec_score:.1f}")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")

        # Save aggregated metrics
        if all_metrics:
            self.io.save_metrics_csv(all_metrics)
            print(f"\n[m03] Metrics saved to: {self.output_dir}/metrics.csv")

    def run_full_pipeline(
        self,
        input_dir: str,
        scene_ids: Optional[List[str]] = None,
        skip_eval: bool = False,
        num_per_level: int = 3
    ) -> None:
        """
        Run complete pipeline: Scene Understanding → Instruction Gen → Evaluation

        Args:
            input_dir: Directory containing image pairs
            scene_ids: Optional list of specific scenes to process
            skip_eval: If True, skip Step 3 (evaluation)
            num_per_level: Number of instructions per level
        """
        print("\n" + "=" * 60)
        print("FULL PIPELINE - 3D Scene Understanding + Instruction Generation")
        print("=" * 60)

        # Find image pairs
        pairs = find_image_pairs(input_dir)
        print(f"\nFound {len(pairs)} image pairs in: {input_dir}")

        if not pairs:
            print("ERROR: No image pairs found")
            return

        # Filter to specific scenes if provided
        if scene_ids:
            pairs = {k: v for k, v in pairs.items() if k in scene_ids}
            print(f"Processing {len(pairs)} selected scenes")

        # Step 1: Scene Understanding
        self.run_step1_scene_understanding(pairs, list(pairs.keys()))

        # Force garbage collection between steps
        gc.collect()
        torch.cuda.empty_cache()

        # Step 2: Instruction Generation
        self.run_step2_instruction_generation(list(pairs.keys()), num_per_level)

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()

        # Step 3: Evaluation (optional)
        if not skip_eval:
            self.run_step3_evaluation(list(pairs.keys()))
        else:
            print("\n[STEP 3] Skipped (--skip_eval)")

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"\nOutputs saved to: {self.output_dir}/")
        print(f"  - m01_scenes/      : Scene analysis JSON files")
        print(f"  - m02_instructions/: Generated instructions JSON files")
        if not skip_eval:
            print(f"  - m03_evaluations/ : Evaluation scores JSON files")
            print(f"  - metrics.csv      : Aggregated metrics for analysis")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline orchestrator for 3D scene understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sanity check (1 sample, all steps)
  python -u src/m04_pipeline_orchestrator.py --sanity

  # Full run without evaluation
  python -u src/m04_pipeline_orchestrator.py --full --skip_eval

  # Full run with evaluation (requires OPENAI_API_KEY)
  export OPENAI_API_KEY="your-key"
  python -u src/m04_pipeline_orchestrator.py --full

  # Evaluation only (requires existing m01 and m02 outputs)
  python -u src/m04_pipeline_orchestrator.py --eval_only
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--sanity", action="store_true",
                           help="Run on 1 sample only (all steps)")
    mode_group.add_argument("--full", action="store_true",
                           help="Run on all images (all steps)")
    mode_group.add_argument("--eval_only", action="store_true",
                           help="Run evaluation only (Step 3)")

    # Options
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR,
                        help="Directory containing image pairs")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for results")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip evaluation step (Step 3)")
    parser.add_argument("--num_per_level", type=int, default=3,
                        help="Number of instructions per level")

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(args.output_dir)

    # Execute based on mode
    if args.sanity:
        pairs = find_image_pairs(args.input_dir)
        if pairs:
            first_scene = list(pairs.keys())[0]
            print(f"Sanity mode: processing only '{first_scene}'")
            orchestrator.run_full_pipeline(
                args.input_dir,
                scene_ids=[first_scene],
                skip_eval=args.skip_eval,
                num_per_level=args.num_per_level
            )
        else:
            print("ERROR: No image pairs found")

    elif args.full:
        orchestrator.run_full_pipeline(
            args.input_dir,
            skip_eval=args.skip_eval,
            num_per_level=args.num_per_level
        )

    elif args.eval_only:
        orchestrator.run_step3_evaluation()

    else:
        print("No mode specified. Use --sanity, --full, or --eval_only")
        print("Run with --help for more options")


if __name__ == "__main__":
    main()
