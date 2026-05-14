"""
Concept labelers for planning-relevant concepts in Sokoban.

Defines the concepts described in Section 3.2:
- Agent Approach Direction (C_A): For each square, the direction from which
  the agent will next step onto that square (UP/DOWN/LEFT/RIGHT/NEVER).
- Box Push Direction (C_B): For each square, the direction in which the
  next box pushed off that square will be pushed (UP/DOWN/LEFT/RIGHT/NEVER).

These are behavior-dependent concepts: they depend on the agent's actions
over the remainder of an episode.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from ..environment.sokoban import (
    SokobanEnv, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
    ACTION_NOOP, ACTION_NAMES, E_BOX, E_BOX_ON_TARGET, E_AGENT, E_AGENT_ON_TARGET,
    E_WALL, E_EMPTY, E_TARGET, ACTION_DELTAS,
)


# Concept class indices
CLASS_UP = 0
CLASS_DOWN = 1
CLASS_LEFT = 2
CLASS_RIGHT = 3
CLASS_NEVER = 4

CLASS_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "NEVER"]

# Map from action to concept class
ACTION_TO_CLASS = {
    ACTION_UP: CLASS_UP,
    ACTION_DOWN: CLASS_DOWN,
    ACTION_LEFT: CLASS_LEFT,
    ACTION_RIGHT: CLASS_RIGHT,
}


class AgentApproachDirection:
    """
    Concept C_A: Agent Approach Direction.

    For a given square, encodes whether the agent will move onto that square
    again in the future. If so, encodes the direction FROM WHICH the agent
    will move onto the square the NEXT time it steps onto it.

    Classes: UP, DOWN, LEFT, RIGHT, NEVER
    """

    def __init__(self):
        self.class_names = CLASS_NAMES

    def compute_labels(
        self,
        agent_trajectory: List[Dict],
        grid_H: int = 8,
        grid_W: int = 8,
    ) -> Dict[str, np.ndarray]:
        """
        Compute concept labels for all time steps in a trajectory.

        Args:
            agent_trajectory: list of dicts with keys:
                'step': int
                'agent_pos': (r, c) agent position at start of step
                'action': int (action taken)
            grid_H, grid_W: grid dimensions

        Returns:
            labels_by_step: dict mapping step number to (H, W) array of class indices
        """
        num_steps = len(agent_trajectory)
        labels_by_step = {}

        # For each step t, compute the concept classes based on future behavior
        for t in range(num_steps):
            labels = np.full((grid_H, grid_W), CLASS_NEVER, dtype=np.int64)

            # Find the next time the agent steps onto each square
            # Track when each square is first stepped onto after step t
            # and from which direction
            first_approach = {}  # (r, c) -> direction_class

            for s in range(t, num_steps):
                step_data = agent_trajectory[s]
                action = step_data["action"]
                if action == ACTION_NOOP:
                    continue

                # The agent moves FROM agent_pos TO neighbor based on action
                old_r, old_c = step_data["agent_pos"]
                dr, dc = ACTION_DELTAS[action]
                new_r, new_c = old_r + dr, old_c + dc

                # The agent approaches (new_r, new_c) from the direction
                # opposite to its movement direction
                # If action is UP, the agent approaches from BELOW (DOWN)
                # If action is DOWN, the agent approaches from ABOVE (UP)
                # If action is LEFT, the agent approaches from RIGHT
                # If action is RIGHT, the agent approaches from LEFT
                opp_direction = {
                    ACTION_UP: CLASS_DOWN,
                    ACTION_DOWN: CLASS_UP,
                    ACTION_LEFT: CLASS_RIGHT,
                    ACTION_RIGHT: CLASS_LEFT,
                }[action]

                if (new_r, new_c) not in first_approach:
                    first_approach[(new_r, new_c)] = opp_direction

            for (r, c), d in first_approach.items():
                if 0 <= r < grid_H and 0 <= c < grid_W:
                    labels[r, c] = d

            labels_by_step[t] = labels

        return labels_by_step


class BoxPushDirection:
    """
    Concept C_B: Box Push Direction.

    For a given square, encodes whether a box will be pushed off that square
    again in the future. If so, encodes the direction in which the NEXT box
    pushed off this square will be pushed.

    Classes: UP, DOWN, LEFT, RIGHT, NEVER
    """

    def __init__(self):
        self.class_names = CLASS_NAMES

    def compute_labels(
        self,
        agent_trajectory: List[Dict],
        grid_H: int = 8,
        grid_W: int = 8,
    ) -> Dict[str, np.ndarray]:
        """
        Compute concept labels for all time steps in a trajectory.
        """
        num_steps = len(agent_trajectory)
        labels_by_step = {}

        for t in range(num_steps):
            labels = np.full((grid_H, grid_W), CLASS_NEVER, dtype=np.int64)

            # Find the next time a box is pushed off each square
            first_push = {}  # (r, c) -> direction_class

            for s in range(t, num_steps):
                step_data = agent_trajectory[s]
                action = step_data["action"]
                if action == ACTION_NOOP:
                    continue

                old_r, old_c = step_data["agent_pos"]
                dr, dc = ACTION_DELTAS[action]
                push_r, push_c = old_r + dr, old_c + dc

                # Check if the agent pushed a box
                # env_state_before and env_state_after from trajectory
                if "pushed_box" in step_data and step_data["pushed_box"]:
                    # A box was pushed from (push_r, push_c) to (push_r+dr, push_c+dc)
                    if (push_r, push_c) not in first_push:
                        first_push[(push_r, push_c)] = ACTION_TO_CLASS[action]

            for (r, c), d in first_push.items():
                if 0 <= r < grid_H and 0 <= c < grid_W:
                    labels[r, c] = d

            labels_by_step[t] = labels

        return labels_by_step


class ConceptLabeler:
    """
    Unified interface for computing concept labels from complete episode trajectories.

    Works by replaying episodes to determine future behavior.
    """

    def __init__(self, env: SokobanEnv, concept_type: str = "agent_approach"):
        """
        Args:
            env: Sokoban environment (used to replay)
            concept_type: "agent_approach" or "box_push"
        """
        self.env = env
        self.concept_type = concept_type

        if concept_type == "agent_approach":
            self.concept = AgentApproachDirection()
        elif concept_type == "box_push":
            self.concept = BoxPushDirection()
        else:
            raise ValueError(f"Unknown concept type: {concept_type}")

    def compute_episode_labels(
        self,
        actions: List[int],
        grid: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Replay an episode and compute concept labels for each step.

        Args:
            actions: list of actions taken throughout the episode
            grid: initial grid state

        Returns:
            labels_by_step: dict mapping step index to (H, W) concept labels
        """
        H, W = grid.shape
        self.env.load_level(grid)
        self.env._episode_max_steps = len(actions) + 10
        self.env.reset()

        # Replay the episode to build trajectory
        trajectory = []
        for t, action in enumerate(actions):
            agent_pos = self.env.get_agent_position()
            grid_before = self.env.get_grid()

            obs, reward, done, info = self.env.step(action)

            grid_after = self.env.get_grid()
            pushed_box = info.get("pushed_box", self._detect_box_push(grid_before, grid_after))

            trajectory.append({
                "step": t,
                "agent_pos": agent_pos,
                "action": action,
                "pushed_box": pushed_box,
            })

            if done:
                break

        return self.concept.compute_labels(trajectory, H, W)

    def _detect_box_push(
        self, grid_before: np.ndarray, grid_after: np.ndarray
    ) -> bool:
        """Detect if a box was pushed by comparing grids."""
        H, W = grid_before.shape
        for r in range(H):
            for c in range(W):
                before = grid_before[r, c]
                after = grid_after[r, c]
                # Box moved: was at (r,c), now (r,c) is different
                if (before == E_BOX or before == E_BOX_ON_TARGET) and \
                   (after != E_BOX and after != E_BOX_ON_TARGET):
                    return True
        return False
