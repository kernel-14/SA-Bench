"""
Planning-relevant concept definitions for Sokoban.

Two main concepts studied in the paper:
1. C_A (Agent Approach Direction): For each square, encodes whether the agent will
   move onto the square in the future, and if so, from which direction.
   
2. C_B (Box Push Direction): For each square, encodes whether a box will be pushed
   off the square in the future, and if so, in which direction.

Both concepts map each grid square to one of 5 classes:
  - UP: agent will approach/push from/to up direction
  - DOWN: agent will approach/push from/to down direction
  - LEFT: agent will approach/push from/to left direction
  - RIGHT: agent will approach/push from/to right direction
  - NEVER: agent will never approach/push this square again
"""

import numpy as np
from typing import List, Tuple, Optional
from enum import IntEnum


class ConceptClass(IntEnum):
    """Classes for planning-relevant concepts."""
    NEVER = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


# Number of concept classes
NUM_CLASSES = 5

# Direction to class mapping
DIRECTION_TO_CLASS = {
    (-1, 0): ConceptClass.UP,    # Moving up means coming from below
    (1, 0): ConceptClass.DOWN,   # Moving down means coming from above
    (0, -1): ConceptClass.LEFT,  # Moving left means coming from right
    (0, 1): ConceptClass.RIGHT,  # Moving right means coming from left
}

# Action to direction delta
ACTION_TO_DELTA = {
    1: (-1, 0),  # UP
    2: (1, 0),   # DOWN
    3: (0, -1),  # LEFT
    4: (0, 1),   # RIGHT
}

GRID_SIZE = 8


def compute_agent_approach_direction(
    trajectory: List[Tuple[int, int]],
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """
    Compute C_A (Agent Approach Direction) for each square.
    
    For each square, this concept encodes whether the agent will move onto
    the square in the future. If so, it encodes the direction from which
    the agent will move onto the square the NEXT time it moves onto it.
    
    Args:
        trajectory: List of (row, col) agent positions over the episode
        grid_size: Size of the grid
        
    Returns:
        Array of shape (grid_size, grid_size) with ConceptClass values
    """
    # Initialize all squares as NEVER
    concept = np.full((grid_size, grid_size), ConceptClass.NEVER, dtype=np.int32)
    
    # For each square, find the next time the agent moves onto it
    # and record the direction from which it approaches
    
    # Build a map: square -> (step, approach_direction)
    # We want the NEXT approach after the current time step
    # Since we compute this for the current state, we look at future trajectory
    
    if len(trajectory) < 2:
        return concept
    
    # Track which squares have been assigned (we want the FIRST future visit)
    assigned = set()
    
    for t in range(1, len(trajectory)):
        prev_pos = trajectory[t - 1]
        curr_pos = trajectory[t]
        
        if curr_pos == prev_pos:
            continue  # No movement (noop)
        
        # Agent moved from prev_pos to curr_pos
        dr = curr_pos[0] - prev_pos[0]
        dc = curr_pos[1] - prev_pos[1]
        
        # The approach direction is the direction the agent moved
        approach_class = DIRECTION_TO_CLASS.get((dr, dc), ConceptClass.NEVER)
        
        # Only assign if not yet assigned (first future visit)
        if curr_pos not in assigned:
            concept[curr_pos[0], curr_pos[1]] = approach_class
            assigned.add(curr_pos)
    
    return concept


def compute_box_push_direction(
    trajectory: List[Tuple[int, int]],
    box_trajectories: List[List[Tuple[int, int]]],
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """
    Compute C_B (Box Push Direction) for each square.
    
    For each square, this concept encodes whether a box will be pushed off
    the square in the future. If so, it encodes the direction in which the
    next box pushed off this square will be pushed.
    
    Args:
        trajectory: List of (row, col) agent positions over the episode
        box_trajectories: List of trajectories for each box (list of positions)
        grid_size: Size of the grid
        
    Returns:
        Array of shape (grid_size, grid_size) with ConceptClass values
    """
    # Initialize all squares as NEVER
    concept = np.full((grid_size, grid_size), ConceptClass.NEVER, dtype=np.int32)
    
    if len(trajectory) < 2:
        return concept
    
    # For each box, track when it moves and from where
    for box_traj in box_trajectories:
        assigned = set()
        
        for t in range(1, len(box_traj)):
            prev_pos = box_traj[t - 1]
            curr_pos = box_traj[t]
            
            if curr_pos == prev_pos:
                continue  # Box didn't move
            
            # Box moved from prev_pos to curr_pos
            dr = curr_pos[0] - prev_pos[0]
            dc = curr_pos[1] - prev_pos[1]
            
            # The push direction is the direction the box moved
            push_class = DIRECTION_TO_CLASS.get((dr, dc), ConceptClass.NEVER)
            
            # Only assign if not yet assigned (first future push from this square)
            if prev_pos not in assigned:
                concept[prev_pos[0], prev_pos[1]] = push_class
                assigned.add(prev_pos)
    
    return concept


def extract_concepts_from_episode(
    agent_positions: List[Tuple[int, int]],
    box_positions_per_step: List[List[Tuple[int, int]]],
    grid_size: int = GRID_SIZE,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract C_A and C_B concept labels for each step of an episode.
    
    For each time step t, we compute the concept labels based on the
    FUTURE trajectory (from step t onwards).
    
    Args:
        agent_positions: List of agent (row, col) positions at each step
        box_positions_per_step: List of lists of box positions at each step
        grid_size: Size of the grid
        
    Returns:
        (ca_labels, cb_labels): Lists of concept arrays for each step
        - ca_labels[t]: C_A labels at step t (grid_size x grid_size)
        - cb_labels[t]: C_B labels at step t (grid_size x grid_size)
    """
    T = len(agent_positions)
    ca_labels = []
    cb_labels = []
    
    for t in range(T):
        # Future trajectory from step t
        future_agent = agent_positions[t:]
        
        # Compute C_A
        ca = compute_agent_approach_direction(future_agent, grid_size)
        ca_labels.append(ca)
        
        # Compute C_B using box trajectories
        # We need to track each box's future trajectory
        if box_positions_per_step:
            num_boxes = len(box_positions_per_step[t])
            box_trajs = []
            
            for box_idx in range(num_boxes):
                box_traj = []
                for step in range(t, T):
                    if box_idx < len(box_positions_per_step[step]):
                        box_traj.append(box_positions_per_step[step][box_idx])
                    else:
                        break
                box_trajs.append(box_traj)
            
            cb = compute_box_push_direction(future_agent, box_trajs, grid_size)
        else:
            cb = np.full((grid_size, grid_size), ConceptClass.NEVER, dtype=np.int32)
        
        cb_labels.append(cb)
    
    return ca_labels, cb_labels
