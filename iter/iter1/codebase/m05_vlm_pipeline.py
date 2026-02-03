"""
VLM End-to-End Pipeline (Top-Down Only - first-person removed, unsuitable for VLM navigation)

Usage:
    python src/m05_vlm_pipeline.py --scene FloorPlan301 --task "bed to desk"
    python src/m05_vlm_pipeline.py --scene FloorPlan301 --task "bed to desk" --llm
    python src/m05_vlm_pipeline.py --demo
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Local imports
from m04_vlm_agent import VLMAgent, load_scene_metadata
from m02_hybrid_judge import HybridJudge, AlgorithmicJudge, PathEvaluation
from utils.grid import GridConfig

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "images" / "ai2thor"


@dataclass
class VLMEvaluationResult:
    """Result from VLM pipeline evaluation (top-down only)."""
    scene_name: str
    task: str
    vlm_path: List[str]
    path_length: int
    optimal_length: int
    efficiency_score: float
    reached_goal: bool
    vlm_success: bool
    explanation: Optional[str] = None
    error: Optional[str] = None


def build_grid_from_metadata(metadata: Dict) -> Tuple[GridConfig, Dict[str, Tuple[int, int]]]:
    """
    Build GridConfig from AI2-THOR scene metadata.

    Returns:
        Tuple of (GridConfig, object_positions dict)
    """
    grid_size = tuple(metadata.get("grid_size", (10, 10)))
    bounds = metadata.get("grid_bounds", {})

    # Create grid
    grid = GridConfig(
        rows=grid_size[0],
        cols=grid_size[1],
        obstacles=set()  # Will be populated from objects if needed
    )

    # Map objects to grid positions
    object_positions = {}
    objects = metadata.get("objects", [])

    # Convert 3D positions to 2D grid coordinates
    min_x = bounds.get("min_x", 0)
    min_z = bounds.get("min_z", 0)
    step = 0.25  # AI2-THOR grid step size

    for obj in objects:
        obj_type = obj.get("type", obj.get("name", "unknown"))
        pos = obj.get("position", {})

        if pos:
            # Convert world coords to grid coords
            grid_row = int((pos.get("z", 0) - min_z) / step)
            grid_col = int((pos.get("x", 0) - min_x) / step)

            # Clamp to grid bounds
            grid_row = max(0, min(grid_row, grid_size[0] - 1))
            grid_col = max(0, min(grid_col, grid_size[1] - 1))

            object_positions[obj_type.lower()] = (grid_row, grid_col)

    return grid, object_positions


def run_vlm_evaluation(
    scene_dir: Path,
    start_object: str,
    goal_object: str,
    use_llm: bool = False,
    vlm_model: str = "gpt-4o"
) -> VLMEvaluationResult:
    """
    Run full VLM evaluation pipeline for a scene (top-down only).

    Args:
        scene_dir: Path to scene directory (e.g., data/images/ai2thor/FloorPlan301)
        start_object: Starting object name
        goal_object: Goal object name
        use_llm: Enable LLM explanations
        vlm_model: OpenAI model for VLM

    Returns:
        VLMEvaluationResult with all metrics
    """
    scene_name = scene_dir.name

    # Load metadata
    metadata = load_scene_metadata(scene_dir)
    if not metadata:
        return VLMEvaluationResult(
            scene_name=scene_name,
            task=f"{start_object} to {goal_object}",
            vlm_path=[],
            path_length=0,
            optimal_length=0,
            efficiency_score=0.0,
            reached_goal=False,
            vlm_success=False,
            error="Metadata not found"
        )

    # Build grid and get object positions
    grid, object_positions = build_grid_from_metadata(metadata)

    # Get start and goal positions
    start_pos = object_positions.get(start_object.lower())
    goal_pos = object_positions.get(goal_object.lower())

    # Warn if objects not found in AI2-THOR metadata
    if not start_pos:
        print(f"  Warning: '{start_object}' not in AI2-THOR metadata")
    if not goal_pos:
        print(f"  Warning: '{goal_object}' not in AI2-THOR metadata")
        print(f"  Available objects: {list(object_positions.keys())[:10]}...")

    # If objects not found, cannot evaluate properly
    if not start_pos or not goal_pos:
        return VLMEvaluationResult(
            scene_name=scene_name,
            task=f"{start_object} to {goal_object}",
            vlm_path=[],
            path_length=0,
            optimal_length=0,
            efficiency_score=0.0,
            reached_goal=False,
            vlm_success=False,
            error=f"Object not found in metadata: start={start_object}({start_pos}), goal={goal_object}({goal_pos})"
        )

    # Get image path (top-down only)
    image_path = scene_dir / "top_down.png"
    if not image_path.exists():
        return VLMEvaluationResult(
            scene_name=scene_name,
            task=f"{start_object} to {goal_object}",
            vlm_path=[],
            path_length=0,
            optimal_length=0,
            efficiency_score=0.0,
            reached_goal=False,
            vlm_success=False,
            error=f"Image not found: {image_path}"
        )

    # Call VLM agent
    vlm_agent = VLMAgent(model=vlm_model)
    vlm_response = vlm_agent.generate_path(
        image_path=image_path,
        start_object=start_object,
        goal_object=goal_object
    )

    if not vlm_response.success:
        return VLMEvaluationResult(
            scene_name=scene_name,
            task=f"{start_object} to {goal_object}",
            vlm_path=[],
            path_length=0,
            optimal_length=0,
            efficiency_score=0.0,
            reached_goal=False,
            vlm_success=False,
            error=vlm_response.error
        )

    # Evaluate with hybrid judge
    hybrid_judge = HybridJudge(grid, use_llm=use_llm)
    evaluation = hybrid_judge.evaluate(
        agent_path=vlm_response.path,
        start=start_pos,
        goal=goal_pos,
        task_description=f"Navigate from {start_object} to {goal_object}"
    )

    return VLMEvaluationResult(
        scene_name=scene_name,
        task=f"{start_object} to {goal_object}",
        vlm_path=vlm_response.path,
        path_length=evaluation.path_length,
        optimal_length=evaluation.optimal_length,
        efficiency_score=evaluation.efficiency_score,
        reached_goal=evaluation.reached_goal,
        vlm_success=True,
        explanation=evaluation.explanation
    )


def run_demo():
    """Run demo with available AI2-THOR scenes (top-down only)."""
    print("=" * 60)
    print("VLM PIPELINE DEMO (Top-Down Only)")
    print("=" * 60)

    if not DATA_DIR.exists():
        print(f"\nNo AI2-THOR images found at {DATA_DIR}")
        print("Run on GPU server first: python src/m03_ai2thor_capture.py --scenes 5")
        return

    # Find available scenes
    scenes = [d for d in DATA_DIR.iterdir() if d.is_dir() and (d / "top_down.png").exists()]

    if not scenes:
        print("\nNo scenes with images found.")
        print("Run on GPU server first: python src/m03_ai2thor_capture.py --scenes 5")
        return

    print(f"\nFound {len(scenes)} scene(s)")

    for scene_dir in scenes[:3]:  # Demo first 3 scenes
        print("\n" + "-" * 60)
        print(f"Scene: {scene_dir.name}")
        print("-" * 60)

        # Load metadata to find objects
        metadata = load_scene_metadata(scene_dir)
        if metadata and metadata.get("objects"):
            objects = metadata["objects"][:2]
            start = objects[0]["type"] if objects else "start"
            goal = objects[1]["type"] if len(objects) > 1 else "goal"
        else:
            start, goal = "start", "goal"

        print(f"Task: {start} -> {goal}")

        # Run evaluation (top-down only)
        result = run_vlm_evaluation(scene_dir, start, goal, use_llm=False)

        if result.vlm_success:
            print(f"  VLM Path: {result.vlm_path}")
            print(f"  Length: {result.path_length} (optimal: {result.optimal_length})")
            print(f"  Efficiency: {result.efficiency_score}/10")
        else:
            print(f"  Error: {result.error}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="VLM End-to-End Pipeline (Top-Down Only)"
    )
    parser.add_argument(
        "--scene", type=str,
        help="Scene name (e.g., FloorPlan301)"
    )
    parser.add_argument(
        "--task", type=str,
        help="Navigation task (e.g., 'bed to lamp')"
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Enable LLM explanations"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="OpenAI model for VLM (default: gpt-4o)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo with available scenes"
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.scene:
        parser.print_help()
        print("\n--- Running Demo ---")
        run_demo()
        return

    # Parse task
    if args.task:
        parts = args.task.lower().split(" to ")
        if len(parts) == 2:
            start_object = parts[0].strip()
            goal_object = parts[1].strip()
        else:
            start_object, goal_object = "start", "goal"
    else:
        start_object, goal_object = "start", "goal"

    # Get scene directory
    scene_dir = DATA_DIR / args.scene
    if not scene_dir.exists():
        print(f"Error: Scene not found: {scene_dir}")
        return

    print(f"\nVLM Pipeline (Top-Down)")
    print(f"  Scene: {args.scene}")
    print(f"  Task: {start_object} -> {goal_object}")
    print(f"  LLM: {'Enabled' if args.llm else 'Disabled'}")

    # Run evaluation
    result = run_vlm_evaluation(
        scene_dir=scene_dir,
        start_object=start_object,
        goal_object=goal_object,
        use_llm=args.llm,
        vlm_model=args.model
    )

    if result.vlm_success:
        print(f"\nResult:")
        print(f"  VLM Path: {result.vlm_path}")
        print(f"  Length: {result.path_length} (optimal: {result.optimal_length})")
        print(f"  Efficiency: {result.efficiency_score}/10")
        print(f"  Reached Goal: {result.reached_goal}")
        if result.explanation:
            print(f"  Explanation: {result.explanation}")
    else:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    main()
