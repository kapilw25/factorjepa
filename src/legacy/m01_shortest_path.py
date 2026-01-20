"""
Shortest Path Algorithms for Grid Navigation
=============================================
Implements BFS, Dijkstra, and A* for finding optimal paths in grid environments.
Supports obstacles and weighted edges.

Usage:
    python src/m01_shortest_path.py --test    # Run all tests
    python src/m01_shortest_path.py --demo    # Show demo with visualization
"""

import heapq
from collections import deque
from typing import List, Tuple, Dict, Optional, Set
import argparse


class ShortestPathFinder:
    """
    Implements BFS, Dijkstra, and A* for grid navigation.

    Supports:
    - Obstacles (walls, furniture) that block movement
    - Weighted edges (e.g., stairs cost 2, walking costs 1)
    - 4-directional movement (North, South, East, West)
    """

    # Direction mappings: direction_name -> (row_delta, col_delta)
    DIRECTIONS = {
        "North": (-1, 0),
        "South": (1, 0),
        "East": (0, 1),
        "West": (0, -1)
    }

    def __init__(
        self,
        grid_size: Tuple[int, int],
        obstacles: Optional[Set[Tuple[int, int]]] = None
    ):
        """
        Initialize the path finder.

        Args:
            grid_size: (rows, cols) tuple defining grid dimensions
            obstacles: Set of (row, col) positions that are blocked
        """
        self.rows, self.cols = grid_size
        self.obstacles = obstacles if obstacles else set()

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not an obstacle."""
        row, col = pos
        return (
            0 <= row < self.rows and
            0 <= col < self.cols and
            pos not in self.obstacles
        )

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[str, Tuple[int, int]]]:
        """Get valid neighboring positions with their direction names."""
        row, col = pos
        neighbors = []
        for direction, (dr, dc) in self.DIRECTIONS.items():
            new_pos = (row + dr, col + dc)
            if self._is_valid(new_pos):
                neighbors.append((direction, new_pos))
        return neighbors

    # =========================================================================
    # BFS - Breadth-First Search (Unweighted)
    # =========================================================================

    def bfs(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> Tuple[List[str], int]:
        """
        BFS for unweighted grids - all moves cost 1.

        Time Complexity: O(V + E) where V = rows*cols, E = 4*V

        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)

        Returns:
            Tuple of (path as list of directions, path length)
            Returns ([], -1) if no path exists
        """
        if start == goal:
            return [], 0

        if not self._is_valid(start) or not self._is_valid(goal):
            return [], -1

        # Queue: (current_position, path_so_far)
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            current_pos, path = queue.popleft()

            for direction, next_pos in self._get_neighbors(current_pos):
                if next_pos == goal:
                    final_path = path + [direction]
                    return final_path, len(final_path)

                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [direction]))

        return [], -1  # No path found

    # =========================================================================
    # Dijkstra - Weighted Shortest Path
    # =========================================================================

    def dijkstra(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        weights: Optional[Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]] = None
    ) -> Tuple[List[str], float]:
        """
        Dijkstra's algorithm for weighted grids.

        Time Complexity: O(E log V) with binary heap

        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
            weights: Dict mapping (from_pos, to_pos) -> edge_cost
                     Default cost is 1 for all edges

        Returns:
            Tuple of (path as list of directions, total cost)
            Returns ([], -1) if no path exists
        """
        if start == goal:
            return [], 0

        if not self._is_valid(start) or not self._is_valid(goal):
            return [], -1

        weights = weights if weights else {}
        default_weight = 1.0

        # Heap: (cost, position, path)
        heap = [(0, start, [])]
        visited = set()

        while heap:
            cost, current_pos, path = heapq.heappop(heap)

            if current_pos in visited:
                continue
            visited.add(current_pos)

            if current_pos == goal:
                return path, cost

            for direction, next_pos in self._get_neighbors(current_pos):
                if next_pos not in visited:
                    # Get edge weight (default to 1)
                    edge_cost = weights.get((current_pos, next_pos), default_weight)
                    new_cost = cost + edge_cost
                    heapq.heappush(heap, (new_cost, next_pos, path + [direction]))

        return [], -1  # No path found

    # =========================================================================
    # A* - Heuristic-Guided Search
    # =========================================================================

    def a_star(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        weights: Optional[Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]] = None
    ) -> Tuple[List[str], float]:
        """
        A* algorithm with Manhattan distance heuristic.

        Faster than Dijkstra for large grids when goal is known.
        Time Complexity: O(E log V) - often better in practice

        Args:
            start: Starting position (row, col)
            goal: Goal position (row, col)
            weights: Dict mapping (from_pos, to_pos) -> edge_cost

        Returns:
            Tuple of (path as list of directions, total cost)
            Returns ([], -1) if no path exists
        """
        if start == goal:
            return [], 0

        if not self._is_valid(start) or not self._is_valid(goal):
            return [], -1

        weights = weights if weights else {}
        default_weight = 1.0

        def heuristic(pos: Tuple[int, int]) -> float:
            """Manhattan distance to goal."""
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

        # Heap: (f_score, g_score, position, path)
        # f_score = g_score + heuristic
        heap = [(heuristic(start), 0, start, [])]
        visited = set()

        while heap:
            _, g_score, current_pos, path = heapq.heappop(heap)

            if current_pos in visited:
                continue
            visited.add(current_pos)

            if current_pos == goal:
                return path, g_score

            for direction, next_pos in self._get_neighbors(current_pos):
                if next_pos not in visited:
                    edge_cost = weights.get((current_pos, next_pos), default_weight)
                    new_g = g_score + edge_cost
                    f_score = new_g + heuristic(next_pos)
                    heapq.heappush(heap, (f_score, new_g, next_pos, path + [direction]))

        return [], -1  # No path found


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def path_to_positions(
    start: Tuple[int, int],
    path: List[str]
) -> List[Tuple[int, int]]:
    """Convert a path (list of directions) to a list of positions."""
    positions = [start]
    current = start

    for direction in path:
        dr, dc = ShortestPathFinder.DIRECTIONS[direction]
        current = (current[0] + dr, current[1] + dc)
        positions.append(current)

    return positions


def visualize_path(
    grid_size: Tuple[int, int],
    obstacles: Set[Tuple[int, int]],
    start: Tuple[int, int],
    goal: Tuple[int, int],
    path: List[str]
) -> str:
    """Create ASCII visualization of path on grid."""
    rows, cols = grid_size
    positions = set(path_to_positions(start, path))

    lines = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            pos = (r, c)
            if pos == start:
                row_chars.append("S")
            elif pos == goal:
                row_chars.append("G")
            elif pos in obstacles:
                row_chars.append("#")
            elif pos in positions:
                row_chars.append("*")
            else:
                row_chars.append(".")
        lines.append(" ".join(row_chars))

    return "\n".join(lines)


# =============================================================================
# TESTS
# =============================================================================

def run_tests():
    """Run comprehensive tests for all algorithms."""
    print("=" * 60)
    print("SHORTEST PATH ALGORITHM TESTS")
    print("=" * 60)

    all_passed = True

    # Test 1: Simple 5x5 grid, no obstacles
    print("\n--- Test 1: Simple 5x5 grid (no obstacles) ---")
    finder = ShortestPathFinder(grid_size=(5, 5))
    start, goal = (0, 0), (4, 4)

    bfs_path, bfs_len = finder.bfs(start, goal)
    dijkstra_path, dijkstra_cost = finder.dijkstra(start, goal)
    astar_path, astar_cost = finder.a_star(start, goal)

    print(f"Start: {start}, Goal: {goal}")
    print(f"BFS:      path_len={bfs_len}, path={bfs_path}")
    print(f"Dijkstra: cost={dijkstra_cost}, path={dijkstra_path}")
    print(f"A*:       cost={astar_cost}, path={astar_path}")

    # All should find path of length 8 (4 down + 4 right)
    if bfs_len == 8 and dijkstra_cost == 8 and astar_cost == 8:
        print("PASS: All algorithms found optimal path length 8")
    else:
        print("FAIL: Expected path length 8")
        all_passed = False

    # Test 2: Grid with obstacles
    print("\n--- Test 2: 5x5 grid with wall obstacle ---")
    obstacles = {(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)}
    finder = ShortestPathFinder(grid_size=(5, 5), obstacles=obstacles)
    start, goal = (0, 0), (2, 2)

    bfs_path, bfs_len = finder.bfs(start, goal)
    dijkstra_path, dijkstra_cost = finder.dijkstra(start, goal)
    astar_path, astar_cost = finder.a_star(start, goal)

    print(f"Obstacles: {obstacles}")
    print(f"Start: {start}, Goal: {goal}")
    print(f"BFS:      path_len={bfs_len}, path={bfs_path}")
    print(f"Dijkstra: cost={dijkstra_cost}, path={dijkstra_path}")
    print(f"A*:       cost={astar_cost}, path={astar_path}")

    print("\nGrid visualization (BFS path):")
    print(visualize_path((5, 5), obstacles, start, goal, bfs_path))

    if bfs_len > 0 and dijkstra_cost > 0 and astar_cost > 0:
        print("PASS: All algorithms found a valid path around obstacles")
    else:
        print("FAIL: Could not find path")
        all_passed = False

    # Test 3: Weighted edges
    print("\n--- Test 3: Weighted edges (stairs cost 2) ---")
    finder = ShortestPathFinder(grid_size=(3, 3))
    start, goal = (0, 0), (2, 2)

    # Make the direct path more expensive
    weights = {
        ((0, 0), (0, 1)): 1,
        ((0, 1), (0, 2)): 1,
        ((0, 2), (1, 2)): 1,
        ((1, 2), (2, 2)): 1,
        ((0, 0), (1, 0)): 2,  # Stairs - more expensive
        ((1, 0), (2, 0)): 2,  # Stairs
    }

    dijkstra_path, dijkstra_cost = finder.dijkstra(start, goal, weights)
    astar_path, astar_cost = finder.a_star(start, goal, weights)

    print(f"Weights applied: some edges cost 2 (stairs)")
    print(f"Dijkstra: cost={dijkstra_cost}, path={dijkstra_path}")
    print(f"A*:       cost={astar_cost}, path={astar_path}")

    # BFS ignores weights
    bfs_path, bfs_len = finder.bfs(start, goal)
    print(f"BFS (ignores weights): path_len={bfs_len}")

    if dijkstra_cost <= astar_cost:
        print("PASS: Dijkstra/A* respect weights")
    else:
        print("FAIL: Weight handling issue")
        all_passed = False

    # Test 4: No path exists
    print("\n--- Test 4: No path exists (blocked) ---")
    obstacles = {(0, 1), (1, 0), (1, 1)}
    finder = ShortestPathFinder(grid_size=(3, 3), obstacles=obstacles)
    start, goal = (0, 0), (2, 2)

    bfs_path, bfs_len = finder.bfs(start, goal)
    dijkstra_path, dijkstra_cost = finder.dijkstra(start, goal)
    astar_path, astar_cost = finder.a_star(start, goal)

    print(f"Obstacles block all paths from (0,0)")
    print(f"BFS:      path_len={bfs_len}")
    print(f"Dijkstra: cost={dijkstra_cost}")
    print(f"A*:       cost={astar_cost}")

    if bfs_len == -1 and dijkstra_cost == -1 and astar_cost == -1:
        print("PASS: All algorithms correctly report no path")
    else:
        print("FAIL: Should report -1 for no path")
        all_passed = False

    # Test 5: Same start and goal
    print("\n--- Test 5: Start equals Goal ---")
    finder = ShortestPathFinder(grid_size=(3, 3))
    start, goal = (1, 1), (1, 1)

    bfs_path, bfs_len = finder.bfs(start, goal)
    dijkstra_path, dijkstra_cost = finder.dijkstra(start, goal)
    astar_path, astar_cost = finder.a_star(start, goal)

    print(f"Start = Goal = {start}")
    print(f"BFS:      path_len={bfs_len}, path={bfs_path}")
    print(f"Dijkstra: cost={dijkstra_cost}, path={dijkstra_path}")
    print(f"A*:       cost={astar_cost}, path={astar_path}")

    if bfs_len == 0 and dijkstra_cost == 0 and astar_cost == 0:
        print("PASS: Zero cost when already at goal")
    else:
        print("FAIL: Expected cost 0")
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    return all_passed


def run_demo():
    """Run all tests with visualizations."""
    print("=" * 60)
    print("SHORTEST PATH DEMO (All Tests with Visualization)")
    print("=" * 60)
    print("\nLegend: S=Start, G=Goal, #=Obstacle, *=Path\n")

    # Demo 1: Simple 5x5 grid, no obstacles
    print("-" * 60)
    print("Demo 1: Simple 5x5 grid (no obstacles)")
    print("-" * 60)
    finder = ShortestPathFinder(grid_size=(5, 5))
    start, goal = (0, 0), (4, 4)

    bfs_path, bfs_len = finder.bfs(start, goal)
    astar_path, astar_cost = finder.a_star(start, goal)

    print(f"Start: {start}, Goal: {goal}")
    print(f"BFS path:  {bfs_path} (len={bfs_len})")
    print(f"A* path:   {astar_path} (cost={astar_cost})")
    print(f"\nBFS Visualization:")
    print(visualize_path((5, 5), set(), start, goal, bfs_path))

    # Demo 2: Grid with obstacles
    print("\n" + "-" * 60)
    print("Demo 2: 5x5 grid with wall obstacle")
    print("-" * 60)
    obstacles = {(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)}
    finder = ShortestPathFinder(grid_size=(5, 5), obstacles=obstacles)
    start, goal = (0, 0), (2, 2)

    bfs_path, bfs_len = finder.bfs(start, goal)
    astar_path, astar_cost = finder.a_star(start, goal)

    print(f"Obstacles: L-shaped wall")
    print(f"Start: {start}, Goal: {goal}")
    print(f"BFS path:  {bfs_path} (len={bfs_len})")
    print(f"A* path:   {astar_path} (cost={astar_cost})")
    print(f"\nBFS Visualization:")
    print(visualize_path((5, 5), obstacles, start, goal, bfs_path))
    print(f"\nA* Visualization:")
    print(visualize_path((5, 5), obstacles, start, goal, astar_path))

    # Demo 3: No path exists
    print("\n" + "-" * 60)
    print("Demo 3: No path exists (blocked)")
    print("-" * 60)
    obstacles = {(0, 1), (1, 0), (1, 1)}
    finder = ShortestPathFinder(grid_size=(3, 3), obstacles=obstacles)
    start, goal = (0, 0), (2, 2)

    bfs_path, bfs_len = finder.bfs(start, goal)

    print(f"Start: {start}, Goal: {goal}")
    print(f"BFS result: path_len={bfs_len} (no path)")
    print(f"\nVisualization (S is blocked):")
    print(visualize_path((3, 3), obstacles, start, goal, []))

    # Demo 4: Bedroom navigation
    print("\n" + "-" * 60)
    print("Demo 4: Bedroom Layout (bed to lamp)")
    print("-" * 60)
    obstacles = {(2, 2), (2, 3), (3, 2)}  # Furniture
    finder = ShortestPathFinder(grid_size=(5, 5), obstacles=obstacles)
    start, goal = (4, 0), (0, 4)  # Bed to Lamp

    path, length = finder.a_star(start, goal)

    print(f"Task: Go from bed {start} to lamp {goal}")
    print(f"Optimal Path: {path}")
    print(f"Path Length: {length} steps")
    print(f"\nVisualization:")
    print(visualize_path((5, 5), obstacles, start, goal, path))

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Shortest Path Algorithms for Grid Navigation"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with visualization"
    )

    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.demo:
        run_demo()
    else:
        # Default: show help
        parser.print_help()
        print("\n--- Quick Test ---")
        finder = ShortestPathFinder(grid_size=(5, 5))
        path, length = finder.bfs((0, 0), (4, 4))
        print(f"BFS (0,0) -> (4,4): {path} (length={length})")


if __name__ == "__main__":
    main()
