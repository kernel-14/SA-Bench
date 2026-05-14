"""
Handcrafted Sokoban levels for intervention experiments.

Creates three types of levels:
1. Agent-Shortcut: agent can choose between short and long path
2. Box-Shortcut: box can be pushed short or long route to target
3. Cutoff: corridor with box at entrance (needs planning to solve)
"""

import numpy as np
from typing import List, Tuple

from ..environment.sokoban import (
    E_WALL, E_EMPTY, E_BOX, E_AGENT, E_TARGET,
    E_BOX_ON_TARGET, E_AGENT_ON_TARGET,
)


def rotate_grid_90(grid: np.ndarray) -> np.ndarray:
    """Rotate grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)


def reflect_grid_vertical(grid: np.ndarray) -> np.ndarray:
    """Reflect grid vertically."""
    return np.flipud(grid)


def generate_variations(grid: np.ndarray, num_variations: int = 8) -> List[np.ndarray]:
    """Generate all rotations and reflections of a grid."""
    variations = []
    current = grid.copy()
    for _ in range(4):
        variations.append(current.copy())
        variations.append(reflect_grid_vertical(current))
        current = rotate_grid_90(current)
    # Return up to num_variations unique ones
    seen = set()
    unique = []
    for v in variations:
        key = v.tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique[:num_variations]


def create_agent_shortcut_levels(
    grid_size: int = 8,
    num_levels: int = 25,
    num_variations: int = 8,
) -> List[Tuple[np.ndarray, List[Tuple[int, int]], Tuple[int, int], List[Tuple[Tuple[int, int], str]]]]:
    """
    Create Agent-Shortcut levels.

    Each level has:
    - Boxes and targets in one region
    - Agent can follow a short or long path to that region
    - Short path is the optimal one

    Returns:
        list of (grid, short_route_squares, long_route_start, long_route_directions)
    """
    levels = []

    # Template: create levels with a corridor that can be bypassed
    for i in range(num_levels):
        grid = np.full((grid_size, grid_size), E_EMPTY, dtype=np.int32)

        # Add walls to create structure
        # Wall border
        grid[0, :] = E_WALL
        grid[-1, :] = E_WALL
        grid[:, 0] = E_WALL
        grid[:, -1] = E_WALL

        # Boxes and targets in upper-right region
        boxes_targets_region = [
            (1, 5), (1, 6), (2, 5), (2, 6),
            (3, 5), (3, 6), (4, 5), (4, 6),
        ]
        targets = [(1, 5), (2, 5), (3, 5), (4, 5)]
        boxes = [(1, 6), (2, 6), (3, 6), (4, 6)]

        for tr, tc in targets:
            grid[tr, tc] = E_TARGET
        for br, bc in boxes:
            grid[br, bc] = E_BOX

        # Create two paths from agent start (bottom-left) to region:
        # Short path: go up through column 3
        # Long path: detour right through column 7

        # Wall separating paths
        wall_col = 3 + (i % 3)  # vary wall column
        for r in range(1, grid_size - 1):
            if r != 2:  # Opening for short path
                grid[r, wall_col] = E_WALL

        # Add walls to make long path detour
        grid[1:grid_size-2, grid_size-2] = E_WALL
        # Opening at bottom for long path
        grid[grid_size-2, grid_size-2] = E_EMPTY

        # Agent start
        agent_pos = (grid_size - 2, grid_size - 2)
        grid[agent_pos[0], agent_pos[1]] = E_AGENT

        # Short route squares (positions on short path to avoid)
        short_route = []
        for r in range(agent_pos[0], 0, -1):
            if grid[r, wall_col - 1] == E_EMPTY:
                short_route.append((r, wall_col - 1))
            if r == 2:
                short_route.append((r, wall_col))
                break

        # Long route directions
        long_start = (agent_pos[0], agent_pos[1])
        long_dirs = [(long_start, "DOWN")]  # First step direction for long route

        levels.append((grid, short_route, long_start, long_dirs))

    # Generate variations
    all_levels = []
    for grid, sr, ls, ld in levels:
        variations = generate_variations(grid, num_variations)
        for var_grid in variations:
            # Find short route squares and long start in rotated grid
            # For simplicity, we recompute
            sr_v = []
            for r in range(grid_size):
                for c in range(grid_size):
                    if var_grid[r, c] in (E_AGENT, E_AGENT_ON_TARGET):
                        agent_r, agent_c = r, c
            # Compute short route as direct path
            sr_v = [(agent_r, agent_c)]
            all_levels.append((var_grid, sr_v, (agent_r, agent_c), [(agent_r, agent_c, "DOWN")]))

    return all_levels[:num_levels * num_variations]


def create_box_shortcut_levels(
    grid_size: int = 8,
    num_levels: int = 25,
    num_variations: int = 8,
) -> List[Tuple[np.ndarray, List[Tuple[int, int]], Tuple[int, int], List[Tuple[Tuple[int, int], str]]]]:
    """
    Create Box-Shortcut levels.

    Each level has:
    - Three boxes adjacent to targets
    - One box that can be pushed a short or long route
    - Short route is the optimal one

    Returns:
        list of (grid, short_route_squares, box_pos, long_route_directions)
    """
    levels = []

    for i in range(num_levels):
        grid = np.full((grid_size, grid_size), E_EMPTY, dtype=np.int32)

        # Wall border
        grid[0, :] = E_WALL
        grid[-1, :] = E_WALL
        grid[:, 0] = E_WALL
        grid[:, -1] = E_WALL

        # Three boxes already on targets
        # Top-left corner
        grid[1, 1] = E_TARGET
        grid[1, 2] = E_BOX_ON_TARGET
        grid[2, 1] = E_BOX_ON_TARGET
        grid[1, 3] = E_TARGET
        grid[2, 3] = E_BOX_ON_TARGET

        # One box that can go short or long route
        box_r, box_c = 4, 2  # Box position
        target_short_r, target_short_c = 4, 1  # Short route target (left)
        target_long_r, target_long_c = 6, 6  # Long route target (far)

        grid[box_r, box_c] = E_BOX
        grid[target_short_r, target_short_c] = E_TARGET
        grid[target_long_r, target_long_c] = E_TARGET

        # Walls to make long route necessary if short blocked
        # Wall blocking short route
        grid[box_r, box_c - 1] = E_WALL  # Block direct short
        # But there's a path around
        grid[box_r - 1, box_c - 1] = E_EMPTY

        # Long route path
        for r in range(box_r, target_long_r + 1):
            grid[r, target_long_c] = E_EMPTY
        grid[target_long_r, target_long_c] = E_TARGET

        # Agent
        agent_pos = (grid_size - 2, 2)
        grid[agent_pos[0], agent_pos[1]] = E_AGENT

        # Short route squares
        short_route = [(box_r, box_c - 1), (box_r - 1, box_c - 1)]
        long_dirs = [(box_r, box_c, "RIGHT")]

        levels.append((grid, short_route, (box_r, box_c), long_dirs))

    all_levels = []
    for grid, sr, bp, ld in levels:
        variations = generate_variations(grid, num_variations)
        for var_grid in variations:
            all_levels.append((var_grid, sr, bp, ld))

    return all_levels[:num_levels * num_variations]


def create_cutoff_levels(
    grid_size: int = 8,
    num_levels: int = 25,
    num_variations: int = 8,
) -> List[np.ndarray]:
    """
    Create Cutoff levels.

    Each level has a corridor with:
    - Box at entrance adjacent to target
    - Box and target at corridor end
    - Agent must not myopically push box at entrance onto target
    - Requires planning to solve

    Returns:
        list of grids
    """
    levels = []

    for i in range(num_levels):
        grid = np.full((grid_size, grid_size), E_EMPTY, dtype=np.int32)

        # Wall border
        grid[0, :] = E_WALL
        grid[-1, :] = E_WALL
        grid[:, 0] = E_WALL
        grid[:, -1] = E_WALL

        # Corridor length (varies)
        corridor_len = 3 + (i % 5)

        # Create corridor (vertical)
        corridor_col = 3
        for r in range(2, 2 + corridor_len):
            grid[r, corridor_col - 1] = E_WALL
            grid[r, corridor_col + 1] = E_WALL
            grid[r, corridor_col] = E_EMPTY

        # Box at corridor end
        end_r = 2 + corridor_len
        grid[end_r, corridor_col] = E_BOX
        grid[end_r + 1, corridor_col] = E_TARGET

        # Box at corridor entrance with target on entrance square
        ent_r = 1
        grid[ent_r, corridor_col] = E_TARGET
        grid[ent_r, corridor_col + 1] = E_BOX  # Box adjacent to target

        # Agent
        agent_pos = (ent_r, corridor_col + 2)
        grid[agent_pos[0], agent_pos[1]] = E_AGENT

        # Make sure agent can push box out of the way
        grid[ent_r, corridor_col + 2] = E_EMPTY
        grid[agent_pos[0], agent_pos[1]] = E_AGENT

        levels.append(grid)

    all_levels = []
    for grid in levels:
        variations = generate_variations(grid, num_variations)
        all_levels.extend(variations)

    return all_levels[:num_levels * num_variations]
