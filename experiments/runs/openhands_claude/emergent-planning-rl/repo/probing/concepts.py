import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from config import CONCEPT_CLASSES, DIRECTION_TO_DELTA, ACTION_TO_DIRECTION, CELL_TYPES


NEVER = CONCEPT_CLASSES["NEVER"]
UP = CONCEPT_CLASSES["UP"]
DOWN = CONCEPT_CLASSES["DOWN"]
LEFT = CONCEPT_CLASSES["LEFT"]
RIGHT = CONCEPT_CLASSES["RIGHT"]

DIRECTION_TO_CLASS = {
    "UP": UP,
    "DOWN": DOWN,
    "LEFT": LEFT,
    "RIGHT": RIGHT,
}

AGENT = CELL_TYPES["AGENT"]
AGENT_ON_TARGET = CELL_TYPES["AGENT_ON_TARGET"]
BOX = CELL_TYPES["BOX"]
BOX_ON_TARGET = CELL_TYPES["BOX_ON_TARGET"]


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    next_obs: np.ndarray
    agent_pos: Tuple[int, int]
    next_agent_pos: Tuple[int, int]
    box_positions: List[Tuple[int, int]]
    next_box_positions: List[Tuple[int, int]]
    pushed_box: Optional[Tuple[int, int]]
    push_direction: Optional[str]
    step: int


def get_agent_position(grid: np.ndarray) -> Optional[Tuple[int, int]]:
    positions = list(zip(*np.where((grid == AGENT) | (grid == AGENT_ON_TARGET))))
    return positions[0] if positions else None


def get_box_positions(grid: np.ndarray) -> List[Tuple[int, int]]:
    return list(zip(*np.where((grid == BOX) | (grid == BOX_ON_TARGET))))


def compute_agent_approach_direction(
    trajectory: List[Transition],
    grid_size: int = 8,
) -> np.ndarray:
    """
    Compute C_A (Agent Approach Direction) labels for each transition.
    
    For each square (r, c) and each transition t, the label is:
      - The direction from which the agent will NEXT step onto (r, c)
        after transition t (i.e., the direction the agent moves to enter (r,c))
      - NEVER if the agent never steps onto (r, c) again in the episode
    
    Returns:
        labels: (T, H, W) integer array with CONCEPT_CLASSES values
    """
    T = len(trajectory)
    labels = np.full((T, grid_size, grid_size), NEVER, dtype=np.int64)

    # For each square, find the next time the agent steps onto it after each t
    # The agent steps onto (r, c) at step t if agent_pos[t] == (r, c) and
    # agent_pos[t-1] != (r, c) (i.e., the agent moved onto it)
    # The approach direction is the direction from agent_pos[t-1] to (r, c)

    # Build list of (step, square, approach_direction) for all agent moves
    agent_visits = []
    for t, trans in enumerate(trajectory):
        if t == 0:
            continue
        prev_pos = trajectory[t - 1].agent_pos
        curr_pos = trans.agent_pos
        if curr_pos != prev_pos:
            dr = curr_pos[0] - prev_pos[0]
            dc = curr_pos[1] - prev_pos[1]
            direction = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}.get(
                (dr, dc)
            )
            if direction:
                agent_visits.append((t, curr_pos, direction))

    # For each transition t and each square, find the next visit after t
    for t in range(T):
        for visit_t, square, direction in agent_visits:
            if visit_t > t:
                r, c = square
                if labels[t, r, c] == NEVER:
                    labels[t, r, c] = DIRECTION_TO_CLASS[direction]

    return labels


def compute_box_push_direction(
    trajectory: List[Transition],
    grid_size: int = 8,
) -> np.ndarray:
    """
    Compute C_B (Box Push Direction) labels for each transition.
    
    For each square (r, c) and each transition t, the label is:
      - The direction in which the next box pushed off (r, c) will be pushed
        after transition t
      - NEVER if no box is ever pushed off (r, c) again in the episode
    
    Returns:
        labels: (T, H, W) integer array with CONCEPT_CLASSES values
    """
    T = len(trajectory)
    labels = np.full((T, grid_size, grid_size), NEVER, dtype=np.int64)

    # Build list of (step, square, push_direction) for all box pushes
    box_pushes = []
    for t, trans in enumerate(trajectory):
        if trans.pushed_box is not None and trans.push_direction is not None:
            box_pushes.append((t, trans.pushed_box, trans.push_direction))

    for t in range(T):
        for push_t, square, direction in box_pushes:
            if push_t > t:
                r, c = square
                if labels[t, r, c] == NEVER:
                    labels[t, r, c] = DIRECTION_TO_CLASS[direction]

    return labels


def compute_agent_approach_binary(
    trajectory: List[Transition],
    grid_size: int = 8,
) -> np.ndarray:
    """
    Compute binary 'Agent Approach' concept (NEVER vs AGAIN).
    Simplified version of C_A without directional information.
    """
    T = len(trajectory)
    labels = np.zeros((T, grid_size, grid_size), dtype=np.int64)

    agent_visits = set()
    for t, trans in enumerate(trajectory):
        if t == 0:
            continue
        prev_pos = trajectory[t - 1].agent_pos
        curr_pos = trans.agent_pos
        if curr_pos != prev_pos:
            agent_visits.add((t, curr_pos))

    for t in range(T):
        for visit_t, square in agent_visits:
            if visit_t > t:
                r, c = square
                labels[t, r, c] = 1  # AGAIN

    return labels


def compute_box_push_binary(
    trajectory: List[Transition],
    grid_size: int = 8,
) -> np.ndarray:
    """
    Compute binary 'Box Push' concept (NEVER vs AGAIN).
    Simplified version of C_B without directional information.
    """
    T = len(trajectory)
    labels = np.zeros((T, grid_size, grid_size), dtype=np.int64)

    box_pushes = []
    for t, trans in enumerate(trajectory):
        if trans.pushed_box is not None:
            box_pushes.append((t, trans.pushed_box))

    for t in range(T):
        for push_t, square in box_pushes:
            if push_t > t:
                r, c = square
                labels[t, r, c] = 1  # AGAIN

    return labels


def compute_agent_exit_direction(
    trajectory: List[Transition],
    grid_size: int = 8,
) -> np.ndarray:
    """
    Compute 'Agent Exit Direction' concept (reversed asymmetry of C_A).
    For each square, the direction the agent moves OFF of it next.
    """
    T = len(trajectory)
    labels = np.full((T, grid_size, grid_size), NEVER, dtype=np.int64)

    agent_exits = []
    for t, trans in enumerate(trajectory):
        if t == 0:
            continue
        prev_pos = trajectory[t - 1].agent_pos
        curr_pos = trans.agent_pos
        if curr_pos != prev_pos:
            dr = curr_pos[0] - prev_pos[0]
            dc = curr_pos[1] - prev_pos[1]
            direction = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}.get(
                (dr, dc)
            )
            if direction:
                agent_exits.append((t, prev_pos, direction))

    for t in range(T):
        for exit_t, square, direction in agent_exits:
            if exit_t > t:
                r, c = square
                if labels[t, r, c] == NEVER:
                    labels[t, r, c] = DIRECTION_TO_CLASS[direction]

    return labels


def compute_box_approach_direction(
    trajectory: List[Transition],
    grid_size: int = 8,
) -> np.ndarray:
    """
    Compute 'Box Approach Direction' concept (reversed asymmetry of C_B).
    For each square, the direction from which the next box is pushed ONTO it.
    """
    T = len(trajectory)
    labels = np.full((T, grid_size, grid_size), NEVER, dtype=np.int64)

    box_arrivals = []
    for t, trans in enumerate(trajectory):
        if trans.pushed_box is not None and trans.push_direction is not None:
            dr, dc = DIRECTION_TO_DELTA[trans.push_direction]
            dest_r = trans.pushed_box[0] + dr
            dest_c = trans.pushed_box[1] + dc
            if 0 <= dest_r < grid_size and 0 <= dest_c < grid_size:
                box_arrivals.append((t, (dest_r, dest_c), trans.push_direction))

    for t in range(T):
        for arr_t, square, direction in box_arrivals:
            if arr_t > t:
                r, c = square
                if labels[t, r, c] == NEVER:
                    labels[t, r, c] = DIRECTION_TO_CLASS[direction]

    return labels


def compute_agent_approach_direction_n(
    trajectory: List[Transition],
    grid_size: int = 8,
    horizon: int = 16,
) -> np.ndarray:
    """
    Compute 'Agent Approach Direction N' concept for Mini PacMan.
    Tracks agent visits within the next N steps only.
    """
    T = len(trajectory)
    labels = np.full((T, grid_size, grid_size), NEVER, dtype=np.int64)

    agent_visits = []
    for t, trans in enumerate(trajectory):
        if t == 0:
            continue
        prev_pos = trajectory[t - 1].agent_pos
        curr_pos = trans.agent_pos
        if curr_pos != prev_pos:
            dr = curr_pos[0] - prev_pos[0]
            dc = curr_pos[1] - prev_pos[1]
            direction = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}.get(
                (dr, dc)
            )
            if direction:
                agent_visits.append((t, curr_pos, direction))

    for t in range(T):
        for visit_t, square, direction in agent_visits:
            if t < visit_t <= t + horizon:
                r, c = square
                if labels[t, r, c] == NEVER:
                    labels[t, r, c] = DIRECTION_TO_CLASS[direction]

    return labels


def build_trajectory_from_episode(
    obs_sequence: List[np.ndarray],
    action_sequence: List[int],
    grid_size: int = 8,
) -> List[Transition]:
    """
    Build a list of Transition objects from an episode's observations and actions.
    obs_sequence: list of (H, W, 7) symbolic observations
    action_sequence: list of integer actions
    """
    from environment.sokoban import parse_boxoban_level

    def obs_to_grid(obs: np.ndarray) -> np.ndarray:
        return obs.argmax(axis=-1).astype(np.int64)

    transitions = []
    T = len(action_sequence)

    for t in range(T):
        obs = obs_sequence[t]
        next_obs = obs_sequence[t + 1] if t + 1 < len(obs_sequence) else obs_sequence[t]
        grid = obs_to_grid(obs)
        next_grid = obs_to_grid(next_obs)

        agent_pos = get_agent_position(grid)
        next_agent_pos = get_agent_position(next_grid)
        box_positions = get_box_positions(grid)
        next_box_positions = get_box_positions(next_grid)

        pushed_box = None
        push_direction = None

        if agent_pos and next_agent_pos and agent_pos != next_agent_pos:
            dr = next_agent_pos[0] - agent_pos[0]
            dc = next_agent_pos[1] - agent_pos[1]
            direction = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}.get(
                (dr, dc)
            )
            if direction:
                box_dest_r = next_agent_pos[0] + dr
                box_dest_c = next_agent_pos[1] + dc
                if (next_agent_pos in box_positions and
                        next_agent_pos not in next_box_positions):
                    pushed_box = next_agent_pos
                    push_direction = direction

        transitions.append(Transition(
            obs=obs,
            action=action_sequence[t],
            next_obs=next_obs,
            agent_pos=agent_pos or (0, 0),
            next_agent_pos=next_agent_pos or (0, 0),
            box_positions=list(box_positions),
            next_box_positions=list(next_box_positions),
            pushed_box=pushed_box,
            push_direction=push_direction,
            step=t,
        ))

    return transitions
