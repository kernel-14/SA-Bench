"""
Level Generation Utilities for Sokoban.

Provides functions to create:
- Agent-Shortcut levels (Section 6.1)
- Box-Shortcut levels (Section 6.1)
- Cutoff levels (Appendix B.3)
- Corridor length variant levels (Appendix A.3.2)
- Level transformations (rotation, reflection)
"""

import numpy as np
from typing import List, Tuple, Set
from src.sokoban import SquareType


def rotate_level(level: np.ndarray, k: int) -> np.ndarray:
    """
    Rotate a level by k * 90 degrees.
    
    The rotation needs to adjust directional aspects accordingly.
    For Sokoban, since the board is symbolic, we just rotate the grid.
    
    Args:
        level: (8, 8) integer array
        k: Number of 90-degree counterclockwise rotations
    
    Returns:
        Rotated level
    """
    result = np.rot90(level, k=k, axes=(0, 1)).copy()
    return result


def reflect_level(level: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Reflect a level horizontally (axis=0) or vertically (axis=1).
    
    Args:
        level: (8, 8) integer array
        axis: 0 for vertical flip, 1 for horizontal flip
    
    Returns:
        Reflected level
    """
    return np.flip(level, axis=axis).copy()


def generate_level_variants(level: np.ndarray) -> List[np.ndarray]:
    """
    Generate 8 variants of a level: original + 3 rotations, and their reflections.
    
    As described in Section 6.1: 25 levels of each type, 8 versions each by
    applying vertical reflection and 90°, 180°, 270° rotations.
    
    Returns:
        List of 8 level variants
    """
    variants = []
    # Original and rotations
    for k in range(4):
        variants.append(rotate_level(level, k))
    
    # Reflected and rotations of reflected
    reflected = reflect_level(level, axis=0)
    for k in range(4):
        variants.append(rotate_level(reflected, k))
    
    return variants


def create_agent_shortcut_level() -> np.ndarray:
    """
    Create an Agent-Shortcut level.
    
    In these levels, all boxes and targets are in one region of the board,
    and the agent can follow either a long or short path to this region.
    
    The agent defaults to the short path; we intervene to force the long path.
    
    Returns:
        (8, 8) integer array
    """
    level = np.full((8, 8), SquareType.EMPTY, dtype=np.int32)
    
    # Walls forming a boundary and corridor
    level[0, :] = SquareType.WALL
    level[-1, :] = SquareType.WALL
    level[:, 0] = SquareType.WALL
    level[:, -1] = SquareType.WALL
    
    # Create a region at the bottom with boxes and targets
    # Short path goes through bottom corridor, long path goes around
    
    # Boxes and targets at bottom-right region
    level[5, 5] = SquareType.BOX
    level[5, 6] = SquareType.BOX
    level[6, 5] = SquareType.BOX
    level[6, 6] = SquareType.BOX
    level[4, 5] = SquareType.TARGET
    level[4, 6] = SquareType.TARGET
    level[5, 4] = SquareType.TARGET
    level[6, 4] = SquareType.TARGET
    
    # Agent starts at top-left
    level[1, 1] = SquareType.AGENT
    
    # Wall to create short/long route choice
    # Short route: agent goes down-right
    # Long route: agent goes around top
    level[2, 3:6] = SquareType.WALL
    level[3, 3] = SquareType.WALL
    level[3, 6] = SquareType.WALL
    
    # Openings for short and long routes
    level[1, 3] = SquareType.EMPTY  # short route entrance
    
    return level


def create_box_shortcut_level() -> np.ndarray:
    """
    Create a Box-Shortcut level.
    
    Three boxes are adjacent to targets; one box can be pushed a long or short route.
    The agent defaults to pushing the box the short route.
    
    Returns:
        (8, 8) integer array
    """
    level = np.full((8, 8), SquareType.EMPTY, dtype=np.int32)
    
    # Walls
    level[0, :] = SquareType.WALL
    level[-1, :] = SquareType.WALL
    level[:, 0] = SquareType.WALL
    level[:, -1] = SquareType.WALL
    
    # Three boxes adjacent to targets (easy)
    level[2, 2] = SquareType.BOX_ON_TARGET
    level[2, 5] = SquareType.BOX_ON_TARGET
    level[5, 2] = SquareType.BOX_ON_TARGET
    level[2, 3] = SquareType.TARGET
    level[2, 6] = SquareType.TARGET
    level[6, 2] = SquareType.TARGET
    
    # The fourth box - can go short or long route
    level[5, 5] = SquareType.BOX
    
    # Short route target
    level[5, 6] = SquareType.TARGET
    
    # Long route target (further away)
    level[6, 5] = SquareType.TARGET
    
    # Also make the long route target reachable
    level[5, 4] = SquareType.EMPTY
    
    # Agent
    level[4, 4] = SquareType.AGENT
    
    return level


def create_cutoff_level(corridor_length: int = 4) -> np.ndarray:
    """
    Create a Cutoff level as described in Appendix B.3.
    
    Has a corridor with a target at the entrance and a box+target at the end.
    The agent must NOT myopically push the entrance box onto the target.
    
    Args:
        corridor_length: Length of the corridor (2, 6, 10, or 14)
    
    Returns:
        (8, 8) integer array
    """
    level = np.full((8, 8), SquareType.EMPTY, dtype=np.int32)
    
    # Walls
    level[0, :] = SquareType.WALL
    level[-1, :] = SquareType.WALL
    level[:, 0] = SquareType.WALL
    level[:, -1] = SquareType.WALL
    
    # Corridor location: column 3, from row 2 downward
    # Target at corridor entrance
    entrance_row = 2
    level[entrance_row, 3] = SquareType.TARGET
    
    # Box adjacent to entrance target
    level[entrance_row, 4] = SquareType.BOX
    
    # Agent starts near the entrance box
    level[entrance_row + 1, 4] = SquareType.AGENT
    
    # Corridor walls (to make it a corridor)
    # Walls to left and right of corridor column
    for r in range(entrance_row, min(entrance_row + corridor_length + 1, 7)):
        if r != entrance_row:
            level[r, 2] = SquareType.WALL
            level[r, 4] = SquareType.WALL
    
    # End of corridor: box and target
    end_row = min(entrance_row + corridor_length, 6)
    level[end_row, 3] = SquareType.BOX
    level[end_row + 1, 3] = SquareType.TARGET if end_row + 1 < 7 else SquareType.EMPTY
    
    if end_row + 1 < 7:
        level[end_row + 1, 3] = SquareType.TARGET
    
    # Make sure the agent has space to push the entrance box out of the way
    level[entrance_row + 1, 5] = SquareType.EMPTY
    level[entrance_row + 1, 6] = SquareType.EMPTY
    
    return level


def create_corridor_length_levels(base_level: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
    """
    Create variants of a level with different corridor lengths.
    
    As described in Appendix A.3.2: creates copies with corridor lengths
    2, 6, 10, and 14.
    
    Returns:
        List of levels with different corridor lengths
    """
    return [create_cutoff_level(L) for L in lengths]


def get_symmetries(level: np.ndarray) -> List[np.ndarray]:
    """Generate all 8 symmetries (4 rotations * 2 reflections)."""
    symmetries = []
    for rot in range(4):
        base = np.rot90(level, k=rot)
        symmetries.append(base.copy())
        symmetries.append(np.flip(base, axis=0).copy())
    return symmetries


def validate_level(level: np.ndarray) -> bool:
    """
    Validate that a level is well-formed.
    
    Checks:
    - Correct dimensions (8x8)
    - Has exactly one agent
    - Has equal number of boxes and targets
    - No overlapping entities
    """
    if level.shape != (8, 8):
        return False
    
    # Count agent positions
    agent_count = np.sum(level == SquareType.AGENT) + np.sum(level == SquareType.AGENT_ON_TARGET)
    if agent_count != 1:
        return False
    
    # Count boxes
    box_count = np.sum(level == SquareType.BOX) + np.sum(level == SquareType.BOX_ON_TARGET)
    
    # Count targets
    target_count = np.sum(level == SquareType.TARGET) + np.sum(level == SquareType.BOX_ON_TARGET) + np.sum(level == SquareType.AGENT_ON_TARGET)
    
    if box_count != target_count:
        return False
    
    return True
