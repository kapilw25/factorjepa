"""
Hybrid Judge: Algorithms + LLM

Usage:
    python src/m02_hybrid_judge.py --demo           # Demo (algo only)
    python src/m02_hybrid_judge.py --demo --llm     # Demo with LLM explanations
    python src/m02_hybrid_judge.py --test           # Run tests
    python src/m02_hybrid_judge.py --compare        # Compare algo vs expected
    python src/m02_hybrid_judge.py --vlm --image data/images/ai2thor/FloorPlan1/
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Local imports
from m01_shortest_path import ShortestPathFinder
from utils.grid import GridConfig, GridBuilder, grid_to_ascii, create_bedroom_grid, visualize_path


# Load environment
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
PROMPTS_FILE = PROJECT_ROOT / "data" / "step0" / "prompts.json"


def load_prompts() -> dict:
    """Load prompt templates from prompts.json."""
    if PROMPTS_FILE.exists():
        with open(PROMPTS_FILE) as f:
            return json.load(f)
    return {}


@dataclass
class PathEvaluation:
    """Evaluation result for a navigation path."""
    path: List[str]
    path_length: int
    optimal_path: List[str]
    optimal_length: int
    efficiency_score: float  # 0-10 scale
    reached_goal: bool
    collision_count: int
    explanation: Optional[str] = None  # LLM-generated explanation


class AlgorithmicJudge:
    """
    Pure algorithmic judge for path evaluation.
    No LLM calls - fast and deterministic.
    """

    def __init__(self, grid: GridConfig):
        self.grid = grid
        self.pathfinder = ShortestPathFinder(
            grid_size=grid.grid_size,
            obstacles=grid.obstacles
        )

    def evaluate(
        self,
        agent_path: List[str],
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> PathEvaluation:
        """
        Evaluate an agent's path against the optimal path.

        Args:
            agent_path: List of directions taken by agent
            start: Starting position
            goal: Goal position

        Returns:
            PathEvaluation with scores and metrics
        """
        # Compute optimal path
        optimal_path, optimal_length = self.pathfinder.a_star(start, goal)

        # Trace agent path and check for collisions/goal
        agent_positions = self._trace_path(start, agent_path)
        final_position = agent_positions[-1] if agent_positions else start
        reached_goal = (final_position == goal)

        # Count collisions
        collision_count = sum(
            1 for pos in agent_positions
            if pos in self.grid.obstacles
        )

        # Calculate efficiency score (0-10)
        agent_length = len(agent_path)
        if agent_length == 0:
            efficiency_score = 10.0 if reached_goal else 0.0
        elif optimal_length <= 0:
            efficiency_score = 0.0  # No valid path exists
        else:
            # Score = (optimal / agent) * 10, capped at 10
            ratio = optimal_length / agent_length
            efficiency_score = min(ratio * 10, 10.0)

            # Penalty for not reaching goal
            if not reached_goal:
                efficiency_score *= 0.5

            # Penalty for collisions
            efficiency_score -= collision_count * 2
            efficiency_score = max(efficiency_score, 0.0)

        return PathEvaluation(
            path=agent_path,
            path_length=agent_length,
            optimal_path=optimal_path,
            optimal_length=optimal_length,
            efficiency_score=round(efficiency_score, 2),
            reached_goal=reached_goal,
            collision_count=collision_count
        )

    def _trace_path(
        self,
        start: Tuple[int, int],
        path: List[str]
    ) -> List[Tuple[int, int]]:
        """Trace path from start position, returning all positions visited."""
        directions = ShortestPathFinder.DIRECTIONS
        positions = [start]
        current = start

        for direction in path:
            if direction in directions:
                dr, dc = directions[direction]
                current = (current[0] + dr, current[1] + dc)
                positions.append(current)

        return positions

    def compare_paths(
        self,
        path_a: List[str],
        path_b: List[str],
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Dict:
        """Compare two paths and return winner."""
        eval_a = self.evaluate(path_a, start, goal)
        eval_b = self.evaluate(path_b, start, goal)

        if eval_a.efficiency_score > eval_b.efficiency_score:
            winner = "Path A"
        elif eval_b.efficiency_score > eval_a.efficiency_score:
            winner = "Path B"
        else:
            winner = "Tie"

        return {
            "path_a": {
                "length": eval_a.path_length,
                "efficiency": eval_a.efficiency_score,
                "reached_goal": eval_a.reached_goal
            },
            "path_b": {
                "length": eval_b.path_length,
                "efficiency": eval_b.efficiency_score,
                "reached_goal": eval_b.reached_goal
            },
            "optimal_length": eval_a.optimal_length,
            "winner": winner
        }


class HybridJudge:
    """
    Hybrid judge combining algorithmic evaluation with LLM explanations.

    Algorithm handles:
    - Optimal path computation
    - Efficiency scoring
    - Collision detection

    LLM handles:
    - Natural language explanations
    - Path comparison reasoning
    """

    def __init__(self, grid: GridConfig, use_llm: bool = False):
        self.grid = grid
        self.algo_judge = AlgorithmicJudge(grid)
        self.use_llm = use_llm and OPENAI_API_KEY is not None
        self.prompts = load_prompts() if self.use_llm else {}

    def evaluate(
        self,
        agent_path: List[str],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        task_description: Optional[str] = None
    ) -> PathEvaluation:
        """Evaluate path with optional LLM explanation."""
        # Get algorithmic evaluation
        result = self.algo_judge.evaluate(agent_path, start, goal)

        # Add LLM explanation if enabled
        if self.use_llm:
            result.explanation = self._generate_explanation(
                result, start, goal, task_description
            )

        return result

    def _generate_explanation(
        self,
        eval_result: PathEvaluation,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        task_description: Optional[str],
        prompt_key: str = "point_based_judge"
    ) -> str:
        """Generate LLM explanation using template from prompts.json."""
        import requests

        # Try to get template from prompts.json
        template = self.prompts.get("prompts", {}).get(prompt_key, {}).get("template")

        if template:
            # Format template with actual values
            try:
                prompt = template.format(
                    task=task_description or f"Navigate from {start} to {goal}",
                    path_actions=" -> ".join(eval_result.path),
                    path_length=eval_result.path_length,
                    optimal_length=eval_result.optimal_length
                )
            except KeyError:
                # Template has different variables, use fallback
                prompt = self._fallback_prompt(eval_result, start, goal, task_description)
        else:
            # No template found, use fallback
            prompt = self._fallback_prompt(eval_result, start, goal, task_description)

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0.3
                }
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[LLM explanation unavailable: {e}]"

    def _fallback_prompt(
        self,
        eval_result: PathEvaluation,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        task_description: Optional[str]
    ) -> str:
        """Fallback prompt if template not found."""
        return f"""Evaluate this navigation path:

Task: {task_description or f"Navigate from {start} to {goal}"}

Agent's Path: {eval_result.path}
Agent Path Length: {eval_result.path_length}
Optimal Path: {eval_result.optimal_path}
Optimal Length: {eval_result.optimal_length}
Reached Goal: {eval_result.reached_goal}
Efficiency Score: {eval_result.efficiency_score}/10

Provide a brief (2-3 sentences) explanation of why this path received this score.
Focus on what the agent did well or poorly."""


# =============================================================================
# TESTS
# =============================================================================

def run_tests():
    """Run tests for hybrid judge with visualizations."""
    print("=" * 60)
    print("HYBRID JUDGE TESTS")
    print("=" * 60)
    print("\nLegend: S=Start, G=Goal, #=Obstacle, *=Path\n")

    all_passed = True

    # Test 1: Perfect path
    print("-" * 60)
    print("Test 1: Perfect (optimal) path")
    print("-" * 60)
    grid = create_bedroom_grid()
    judge = AlgorithmicJudge(grid)

    start = grid.objects["bed"]   # (4, 0)
    goal = grid.objects["lamp"]   # (0, 4)

    # Get optimal path first
    pf = ShortestPathFinder(grid.grid_size, grid.obstacles)
    optimal_path, _ = pf.a_star(start, goal)

    result = judge.evaluate(optimal_path, start, goal)

    print(f"Start: {start}, Goal: {goal}")
    print(f"Path: {optimal_path}")
    print(f"Efficiency: {result.efficiency_score}/10")
    print(f"Reached Goal: {result.reached_goal}")
    print(f"\nVisualization:")
    print(visualize_path(grid, start, goal, optimal_path))

    if result.efficiency_score == 10.0 and result.reached_goal:
        print("\nPASS: Perfect path gets score 10")
    else:
        print("\nFAIL: Expected score 10 for optimal path")
        all_passed = False

    # Test 2: Suboptimal path
    print("\n" + "-" * 60)
    print("Test 2: Suboptimal path (with backtracking)")
    print("-" * 60)
    suboptimal = ["North", "South", "North", "North", "North", "North", "East", "East", "East", "East"]

    result = judge.evaluate(suboptimal, start, goal)

    print(f"Path: {suboptimal}")
    print(f"Path Length: {result.path_length}")
    print(f"Optimal Length: {result.optimal_length}")
    print(f"Efficiency: {result.efficiency_score}/10")
    print(f"Reached Goal: {result.reached_goal}")
    print(f"\nVisualization:")
    print(visualize_path(grid, start, goal, suboptimal))

    if result.efficiency_score < 10.0:
        print("\nPASS: Suboptimal path gets lower score")
    else:
        print("\nFAIL: Should score less than optimal")
        all_passed = False

    # Test 3: Path comparison
    print("\n" + "-" * 60)
    print("Test 3: Path comparison (A vs B)")
    print("-" * 60)
    path_a = optimal_path
    path_b = suboptimal

    comparison = judge.compare_paths(path_a, path_b, start, goal)

    print(f"Path A: {path_a}")
    print(f"Path B: {path_b}")
    print(f"\nPath A efficiency: {comparison['path_a']['efficiency']}/10")
    print(f"Path B efficiency: {comparison['path_b']['efficiency']}/10")
    print(f"Winner: {comparison['winner']}")

    print(f"\nPath A Visualization:")
    print(visualize_path(grid, start, goal, path_a))
    print(f"\nPath B Visualization:")
    print(visualize_path(grid, start, goal, path_b))

    if comparison["winner"] == "Path A":
        print("\nPASS: Correct winner identified")
    else:
        print("\nFAIL: Wrong winner")
        all_passed = False

    # Test 4: Path with collisions
    print("\n" + "-" * 60)
    print("Test 4: Path through obstacle")
    print("-" * 60)
    # This path goes through obstacle at (2,2)
    collision_path = ["North", "North", "East", "East"]  # Ends at (2,2) which is obstacle

    result = judge.evaluate(collision_path, start, goal)

    print(f"Path: {collision_path}")
    print(f"Collisions: {result.collision_count}")
    print(f"Efficiency: {result.efficiency_score}/10")
    print(f"\nVisualization (path hits obstacle at #):")
    print(visualize_path(grid, start, goal, collision_path))

    if result.collision_count > 0 or result.efficiency_score < 5:
        print("\nPASS: Collision detected and penalized")
    else:
        print("\nFAIL: Should detect collision")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


def run_compare():
    """Compare algorithmic judge on test cases with visualizations."""
    print("=" * 60)
    print("ALGORITHM vs EXPECTED COMPARISON")
    print("=" * 60)
    print("\nLegend: S=Start, G=Goal, #=Obstacle, *=Path\n")

    # Load test cases
    test_cases_file = PROJECT_ROOT / "data" / "step0" / "test_cases.json"

    if test_cases_file.exists():
        with open(test_cases_file) as f:
            data = json.load(f)

        print(f"Using test cases from data/step0/test_cases.json")
        print(f"Total test cases: {len(data['test_cases'])}")

        for test in data["test_cases"]:
            print("\n" + "-" * 60)
            print(f"{test['id']}: {test['task']}")
            print("-" * 60)

            # Build grid from test case layout
            layout = test["room_layout"]
            grid_size = tuple(layout["grid_size"])

            # Get obstacles if defined
            obstacles = set()
            if "obstacles" in layout:
                obstacles = set(tuple(pos) for pos in layout["obstacles"])

            # Build grid
            grid = GridConfig(
                rows=grid_size[0],
                cols=grid_size[1],
                obstacles=obstacles
            )

            # Get start/goal from test case objects
            objects = layout["objects"]
            start_key = list(objects.keys())[0]  # First object is start
            goal_key = list(objects.keys())[1]   # Second object is goal
            start = tuple(objects[start_key]["position"])
            goal = tuple(objects[goal_key]["position"])

            print(f"Grid: {grid_size[0]}x{grid_size[1]}, Start: {start} ({start_key}), Goal: {goal} ({goal_key})")

            # Create judge for this grid
            algo_judge = AlgorithmicJudge(grid)

            path_a = test["path_a"]["actions"]
            path_b = test["path_b"]["actions"]

            comparison = algo_judge.compare_paths(path_a, path_b, start, goal)

            print(f"\nPath A ({test['path_a']['name']}): {path_a}")
            print(f"Path B ({test['path_b']['name']}): {path_b}")
            print(f"Optimal: {test['ground_truth']['optimal_path']} (len={test['ground_truth']['optimal_length']})")

            print(f"\nPath A Visualization:")
            print(visualize_path(grid, start, goal, path_a))

            print(f"\nPath B Visualization:")
            print(visualize_path(grid, start, goal, path_b))

            print(f"\nAlgorithmic Result:")
            print(f"  Path A efficiency: {comparison['path_a']['efficiency']}/10")
            print(f"  Path B efficiency: {comparison['path_b']['efficiency']}/10")
            print(f"  Winner: {comparison['winner']}")
            print(f"  Expected: {test['expected_winner']}")

            if comparison['winner'].lower().replace(" ", "") in test['expected_winner'].lower().replace(" ", ""):
                print("  MATCH")
            else:
                print("  MISMATCH")
    else:
        print("\nNo test cases found. Run with --test for basic tests.")


def run_demo(use_llm: bool = False):
    """Run demo evaluation with multiple paths."""
    print("=" * 60)
    print("HYBRID JUDGE DEMO" + (" (LLM Enabled)" if use_llm else ""))
    print("=" * 60)
    print("\nLegend: S=Start, G=Goal, #=Obstacle, *=Path\n")

    grid = create_bedroom_grid()
    start = grid.objects["bed"]   # (4, 0)
    goal = grid.objects["lamp"]   # (0, 4)

    print(f"Task: Navigate from bed {start} to lamp {goal}")
    print(f"Grid: 5x5 bedroom with obstacles at (2,2) and (2,3)")
    if use_llm:
        print(f"LLM: Enabled (using OpenAI API)")

    # Create judge
    hybrid_judge = HybridJudge(grid, use_llm=use_llm)

    # Demo paths
    demo_paths = [
        {
            "name": "Optimal Path",
            "path": ["North", "North", "North", "North", "East", "East", "East", "East"],
            "desc": "Direct route, no backtracking"
        },
        {
            "name": "Zigzag Path",
            "path": ["North", "East", "North", "East", "North", "East", "North", "East"],
            "desc": "Diagonal zigzag, same length"
        },
        {
            "name": "Inefficient Path",
            "path": ["North", "South", "North", "North", "North", "North", "East", "East", "East", "East"],
            "desc": "Backtracking (North then South)"
        },
        {
            "name": "Very Inefficient",
            "path": ["East", "West", "North", "South", "North", "North", "North", "North", "East", "East", "East", "East"],
            "desc": "Multiple backtracks"
        }
    ]

    for demo in demo_paths:
        print("\n" + "-" * 60)
        print(f"{demo['name']}: {demo['desc']}")
        print("-" * 60)

        result = hybrid_judge.evaluate(demo["path"], start, goal, "Go from bed to lamp")

        print(f"Path: {demo['path']}")
        print(f"Length: {result.path_length} (optimal: {result.optimal_length})")
        print(f"Efficiency: {result.efficiency_score}/10")
        print(f"Reached Goal: {result.reached_goal}")

        if result.explanation:
            print(f"\nLLM Explanation:")
            print(f"  {result.explanation}")

        print(f"\nVisualization:")
        print(visualize_path(grid, start, goal, demo["path"]))

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


# =============================================================================
# VLM MODE (Top-Down Only)
# =============================================================================

def run_vlm(image_path: str, task: str = None, use_llm: bool = False):
    """Run VLM-based evaluation on AI2-THOR scene (top-down only)."""
    from pathlib import Path

    scene_dir = Path(image_path)
    if not scene_dir.exists():
        print(f"Error: Path not found: {scene_dir}")
        return

    # Import VLM pipeline
    try:
        from m05_vlm_pipeline import run_vlm_evaluation, load_scene_metadata
    except ImportError as e:
        print(f"Error importing VLM pipeline: {e}")
        return

    print("=" * 60)
    print("VLM EVALUATION (Top-Down Only)")
    print("=" * 60)

    # Parse task or use defaults from metadata
    if task:
        parts = task.lower().split(" to ")
        if len(parts) == 2:
            start_object = parts[0].strip()
            goal_object = parts[1].strip()
        else:
            start_object, goal_object = "start", "goal"
    else:
        # Load from metadata
        metadata = load_scene_metadata(scene_dir)
        if metadata and metadata.get("objects"):
            objects = metadata["objects"][:2]
            start_object = objects[0]["type"] if objects else "start"
            goal_object = objects[1]["type"] if len(objects) > 1 else "goal"
        else:
            start_object, goal_object = "start", "goal"

    print(f"\nScene: {scene_dir.name}")
    print(f"Task: {start_object} -> {goal_object}")
    print(f"LLM: {'Enabled' if use_llm else 'Disabled'}")

    # Run evaluation (top-down only)
    result = run_vlm_evaluation(scene_dir, start_object, goal_object, use_llm)

    if result.vlm_success:
        print(f"\nResult:")
        print(f"  VLM Path: {result.vlm_path}")
        print(f"  Length: {result.path_length} (optimal: {result.optimal_length})")
        print(f"  Efficiency: {result.efficiency_score}/10")
        if result.explanation:
            print(f"  Explanation: {result.explanation}")
    else:
        print(f"\nError: {result.error}")

    print("\n" + "=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid Judge: Algorithms + LLM")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--compare", action="store_true", help="Compare algo vs LLM")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--llm", action="store_true", help="Enable LLM explanations")
    parser.add_argument("--vlm", action="store_true", help="Run VLM-based evaluation")
    parser.add_argument("--image", type=str, help="Path to scene directory for VLM mode")
    parser.add_argument("--task", type=str, help="Task description (e.g., 'bed to lamp')")

    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.compare:
        run_compare()
    elif args.vlm:
        if not args.image:
            print("Error: --vlm requires --image <scene_directory>")
            return
        run_vlm(args.image, args.task, use_llm=args.llm)
    elif args.demo:
        run_demo(use_llm=args.llm)
    else:
        parser.print_help()
        print("\n--- Quick Demo ---")
        run_demo(use_llm=False)


if __name__ == "__main__":
    main()
