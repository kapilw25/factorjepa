"""
Grid Utilities for Navigation
==============================
Provides utilities for creating grids, managing obstacles, and defining edge weights.

Usage:
    from utils.grid import GridBuilder, GridConfig

    grid = GridBuilder(5, 5).add_obstacle(2, 2).build()
"""

import json
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass, field


@dataclass
class GridConfig:
    """Configuration for a navigation grid."""
    rows: int
    cols: int
    obstacles: Set[Tuple[int, int]] = field(default_factory=set)
    weights: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = field(default_factory=dict)
    objects: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # name -> position

    @property
    def grid_size(self) -> Tuple[int, int]:
        return (self.rows, self.cols)


class GridBuilder:
    """
    Builder class for creating navigation grids.

    Supports:
    - Obstacle placement (walls, furniture)
    - Named object placement (bed, lamp, door)
    - Edge weight configuration (stairs=2, normal=1)
    """

    def __init__(self, rows: int, cols: int):
        self.config = GridConfig(rows=rows, cols=cols)

    def add_obstacle(self, row: int, col: int) -> "GridBuilder":
        """Add a single obstacle at position."""
        self.config.obstacles.add((row, col))
        return self

    def add_obstacles(self, positions: List[Tuple[int, int]]) -> "GridBuilder":
        """Add multiple obstacles."""
        for pos in positions:
            self.config.obstacles.add(pos)
        return self

    def add_wall(self, start: Tuple[int, int], end: Tuple[int, int]) -> "GridBuilder":
        """Add a wall (line of obstacles) from start to end."""
        r1, c1 = start
        r2, c2 = end

        if r1 == r2:  # Horizontal wall
            for c in range(min(c1, c2), max(c1, c2) + 1):
                self.config.obstacles.add((r1, c))
        elif c1 == c2:  # Vertical wall
            for r in range(min(r1, r2), max(r1, r2) + 1):
                self.config.obstacles.add((r, c1))
        return self

    def add_object(self, name: str, row: int, col: int) -> "GridBuilder":
        """Add a named object (landmark) at position."""
        self.config.objects[name] = (row, col)
        return self

    def set_weight(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
        weight: float,
        bidirectional: bool = True
    ) -> "GridBuilder":
        """Set edge weight between two adjacent cells."""
        self.config.weights[(from_pos, to_pos)] = weight
        if bidirectional:
            self.config.weights[(to_pos, from_pos)] = weight
        return self

    def add_high_cost_region(self, positions: List[Tuple[int, int]], cost: float = 2.0) -> "GridBuilder":
        """Mark a region where movement costs more (e.g., stairs)."""
        for pos in positions:
            row, col = pos
            neighbors = [(row - 1, col), (row + 1, col), (row, col + 1), (row, col - 1)]
            for neighbor in neighbors:
                if 0 <= neighbor[0] < self.config.rows and 0 <= neighbor[1] < self.config.cols:
                    self.config.weights[(neighbor, pos)] = cost
        return self

    def build(self) -> GridConfig:
        """Return the configured grid."""
        return self.config


class ASCIIGridParser:
    """
    Parse ASCII grid representations into GridConfig.

    Legend:
        . = empty, # = obstacle, S = start, G = goal, [A-Z] = named objects
    """

    @staticmethod
    def parse(ascii_map: List[str]) -> GridConfig:
        """Parse ASCII grid into GridConfig."""
        rows = len(ascii_map)
        cols = max(len(row.replace(" ", "")) for row in ascii_map) if ascii_map else 0

        config = GridConfig(rows=rows, cols=cols)

        for r, row in enumerate(ascii_map):
            cells = row.replace(" ", "")
            for c, char in enumerate(cells):
                if char == "#":
                    config.obstacles.add((r, c))
                elif char == "S":
                    config.objects["start"] = (r, c)
                elif char == "G":
                    config.objects["goal"] = (r, c)
                elif char.isalpha() and char not in ["S", "G"]:
                    config.objects[char.lower()] = (r, c)

        return config


# Direction mappings
DIRECTIONS = {
    "North": (-1, 0),
    "South": (1, 0),
    "East": (0, 1),
    "West": (0, -1)
}


def path_to_positions(
    start: Tuple[int, int],
    path: List[str]
) -> List[Tuple[int, int]]:
    """Convert a path (list of directions) to a list of positions."""
    positions = [start]
    current = start

    for direction in path:
        if direction in DIRECTIONS:
            dr, dc = DIRECTIONS[direction]
            current = (current[0] + dr, current[1] + dc)
            positions.append(current)

    return positions


def visualize_path(
    config: GridConfig,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    path: List[str]
) -> str:
    """Visualize a path on the grid. S=Start, G=Goal, #=Obstacle, *=Path"""
    positions = set(path_to_positions(start, path))

    lines = []
    for r in range(config.rows):
        row_chars = []
        for c in range(config.cols):
            pos = (r, c)
            if pos == start:
                row_chars.append("S")
            elif pos == goal:
                row_chars.append("G")
            elif pos in config.obstacles:
                row_chars.append("#")
            elif pos in positions:
                row_chars.append("*")
            else:
                row_chars.append(".")
        lines.append(" ".join(row_chars))
    return "\n".join(lines)


def grid_to_ascii(
    config: GridConfig,
    path: Optional[List[Tuple[int, int]]] = None,
    show_objects: bool = True
) -> str:
    """Convert GridConfig to ASCII visualization."""
    path_set = set(path) if path else set()
    pos_to_name = {pos: name[0].upper() for name, pos in config.objects.items()}

    lines = []
    for r in range(config.rows):
        row_chars = []
        for c in range(config.cols):
            pos = (r, c)
            if show_objects and pos in pos_to_name:
                row_chars.append(pos_to_name[pos])
            elif pos in config.obstacles:
                row_chars.append("#")
            elif pos in path_set:
                row_chars.append("*")
            else:
                row_chars.append(".")
        lines.append(" ".join(row_chars))
    return "\n".join(lines)


# =============================================================================
# PRESET GRIDS
# =============================================================================

def create_bedroom_grid() -> GridConfig:
    """Create a sample bedroom grid."""
    return (
        GridBuilder(5, 5)
        .add_object("bed", 4, 0)
        .add_object("lamp", 0, 4)
        .add_object("door", 4, 4)
        .add_object("desk", 0, 0)
        .add_obstacles([(2, 2), (2, 3)])
        .build()
    )


def create_office_grid() -> GridConfig:
    """Create a sample office grid."""
    return (
        GridBuilder(6, 6)
        .add_object("door", 5, 0)
        .add_object("desk", 0, 5)
        .add_object("chair", 1, 4)
        .add_wall((2, 1), (2, 4))
        .build()
    )


def create_living_room_grid() -> GridConfig:
    """Create a sample living room grid."""
    return (
        GridBuilder(7, 7)
        .add_object("sofa", 3, 1)
        .add_object("tv", 3, 5)
        .add_object("door", 6, 3)
        .add_obstacles([(2, 2), (2, 3), (2, 4), (4, 2), (4, 3), (4, 4)])
        .build()
    )
