"""
Sokoban environment for studying emergent planning in model-free RL agents.

Based on the Boxoban dataset (Guez et al., 2018) as used in
"An Investigation of Model-Free Planning" (Guez et al., 2019).

Observations are symbolic: 8x8x7 one-hot grids.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List


# Channel indices for the symbolic 7-channel representation
CH_WALL = 0
CH_EMPTY = 1
CH_BOX = 2
CH_AGENT = 3
CH_BOX_TARGET = 4
CH_AGENT_TARGET = 5
CH_EMPTY_TARGET = 6

# Grid element types
E_WALL = 0
E_EMPTY = 1
E_BOX = 2
E_AGENT = 3
E_TARGET = 4
E_BOX_ON_TARGET = 5
E_AGENT_ON_TARGET = 6

# Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_NOOP = 4

ACTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
    ACTION_NOOP: (0, 0),
}

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "NOOP"]


def grid_to_symbolic(grid: np.ndarray) -> np.ndarray:
    """Convert a 2D grid of element types to symbolic 8x8x7 one-hot observation."""
    H, W = grid.shape
    obs = np.zeros((H, W, 7), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            e = grid[r, c]
            if e == E_WALL:
                obs[r, c, CH_WALL] = 1
            elif e == E_EMPTY:
                obs[r, c, CH_EMPTY] = 1
            elif e == E_BOX:
                obs[r, c, CH_BOX] = 1
            elif e == E_AGENT:
                obs[r, c, CH_AGENT] = 1
            elif e == E_TARGET:
                obs[r, c, CH_EMPTY_TARGET] = 1
            elif e == E_BOX_ON_TARGET:
                obs[r, c, CH_BOX_TARGET] = 1
            elif e == E_AGENT_ON_TARGET:
                obs[r, c, CH_AGENT_TARGET] = 1
    return obs


def parse_boxoban_level(level_str: str) -> np.ndarray:
    """
    Parse a Boxoban level string into a 2D grid of element types.

    Character mapping:
        # = wall
        @ = agent on empty
        $ = box on empty
        . = empty target
        * = box on target
        + = agent on target
        (space) = empty
    """
    lines = [line for line in level_str.strip().split("\n") if line.strip()]
    if not lines:
        raise ValueError("Empty level string")
    H = len(lines)
    W = max(len(line) for line in lines)
    grid = np.zeros((H, W), dtype=np.int32)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch == "#":
                grid[r, c] = E_WALL
            elif ch == " ":
                grid[r, c] = E_EMPTY
            elif ch == "@":
                grid[r, c] = E_AGENT
            elif ch == "$":
                grid[r, c] = E_BOX
            elif ch == ".":
                grid[r, c] = E_TARGET
            elif ch == "*":
                grid[r, c] = E_BOX_ON_TARGET
            elif ch == "+":
                grid[r, c] = E_AGENT_ON_TARGET
            else:
                grid[r, c] = E_EMPTY
    return grid


class SokobanEnv:
    """
    Sokoban environment.

    The agent observes an 8x8x7 symbolic one-hot representation.
    Actions: UP(0), DOWN(1), LEFT(2), RIGHT(3), NOOP(4).

    Rewards:
        - -0.01 per step
        - +1 for pushing a box onto a target
        - -1 for pushing a box off a target
        - +10 for solving (all boxes on targets)
    """

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 120,
        min_steps: int = 115,
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.action_space_size = 5

        self._grid: Optional[np.ndarray] = None
        self._agent_pos: Optional[Tuple[int, int]] = None
        self._targets: List[Tuple[int, int]] = []
        self._boxes: List[Tuple[int, int]] = []
        self._step_count: int = 0
        self._episode_max_steps: int = max_steps
        self._done: bool = False
        self._total_reward: float = 0.0

    def load_level(self, grid: np.ndarray) -> None:
        """Load a level from a 2D grid of element types."""
        self._grid = grid.copy()
        self._targets = []
        self._boxes = []
        self._agent_pos = None
        H, W = grid.shape
        for r in range(H):
            for c in range(W):
                e = grid[r, c]
                if e == E_TARGET or e == E_BOX_ON_TARGET or e == E_AGENT_ON_TARGET:
                    self._targets.append((r, c))
                if e == E_BOX or e == E_BOX_ON_TARGET:
                    self._boxes.append((r, c))
                if e == E_AGENT or e == E_AGENT_ON_TARGET:
                    self._agent_pos = (r, c)
        self._step_count = 0
        self._episode_max_steps = np.random.randint(self.min_steps, self.max_steps + 1)
        self._done = False
        self._total_reward = 0.0

    def reset(self) -> np.ndarray:
        """Reset the environment. Must call load_level first."""
        self._step_count = 0
        self._episode_max_steps = np.random.randint(self.min_steps, self.max_steps + 1)
        self._done = False
        self._total_reward = 0.0
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        """Get the symbolic observation (8x8x7)."""
        return grid_to_symbolic(self._grid)

    def get_grid(self) -> np.ndarray:
        """Get the raw grid."""
        return self._grid.copy()

    def _element_at(self, r: int, c: int) -> int:
        """Get the element type at (r, c)."""
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            return self._grid[r, c]
        return E_WALL

    def _set_element(self, r: int, c: int, e: int) -> None:
        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
            self._grid[r, c] = e

    def _is_target(self, r: int, c: int) -> bool:
        return (r, c) in self._targets

    def _is_box(self, r: int, c: int) -> bool:
        e = self._element_at(r, c)
        return e == E_BOX or e == E_BOX_ON_TARGET

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute an action and return (obs, reward, done, info).

        Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=NOOP
        """
        if self._done:
            return self.get_obs(), 0.0, True, {"solved": self._is_solved()}

        if action == ACTION_NOOP:
            self._step_count += 1
            reward = -0.01
            self._total_reward += reward
            done = self._step_count >= self._episode_max_steps
            if done:
                self._done = True
            return self.get_obs(), reward, done, {"solved": self._is_solved()}

        dr, dc = ACTION_DELTAS[action]
        ar, ac = self._agent_pos
        nr, nc = ar + dr, ac + dc

        target_e = self._element_at(nr, nc)

        if target_e == E_WALL:
            # Walk into wall: no movement
            self._step_count += 1
            reward = -0.01
            self._total_reward += reward
            done = self._step_count >= self._episode_max_steps
            if done:
                self._done = True
            return self.get_obs(), reward, done, {"solved": self._is_solved()}

        if self._is_box(nr, nc):
            # Try to push the box
            br, bc = nr + dr, nc + dc
            beyond_e = self._element_at(br, bc)
            # Cannot push into wall or another box
            if beyond_e == E_WALL or self._is_box(br, bc):
                self._step_count += 1
                reward = -0.01
                self._total_reward += reward
                done = self._step_count >= self._episode_max_steps
                if done:
                    self._done = True
                return self.get_obs(), reward, done, {"solved": self._is_solved()}

            reward = -0.01

            # Move box off its current square
            was_on_target = (nr, nc) in self._targets
            if was_on_target:
                reward += -1.0  # pushed box off target
                self._set_element(nr, nc, E_TARGET)
            else:
                self._set_element(nr, nc, E_EMPTY)

            # Move box to new square
            will_be_on_target = (br, bc) in self._targets
            if will_be_on_target:
                reward += 1.0  # pushed box onto target
                self._set_element(br, bc, E_BOX_ON_TARGET)
            else:
                self._set_element(br, bc, E_BOX)

            # Update box list
            self._boxes = []
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if self._is_box(r, c):
                        self._boxes.append((r, c))

            # Move agent
            old_ar, old_ac = self._agent_pos
            old_on_target = (old_ar, old_ac) in self._targets
            if old_on_target:
                self._set_element(old_ar, old_ac, E_TARGET)
            else:
                self._set_element(old_ar, old_ac, E_EMPTY)

            new_on_target = (nr, nc) in self._targets
            if new_on_target:
                self._set_element(nr, nc, E_AGENT_ON_TARGET)
            else:
                self._set_element(nr, nc, E_AGENT)
            self._agent_pos = (nr, nc)

        else:
            # Move agent without pushing
            old_ar, old_ac = self._agent_pos
            old_on_target = (old_ar, old_ac) in self._targets
            if old_on_target:
                self._set_element(old_ar, old_ac, E_TARGET)
            else:
                self._set_element(old_ar, old_ac, E_EMPTY)

            new_on_target = (nr, nc) in self._targets
            if new_on_target:
                self._set_element(nr, nc, E_AGENT_ON_TARGET)
            else:
                self._set_element(nr, nc, E_AGENT)
            self._agent_pos = (nr, nc)

            reward = -0.01

        self._total_reward += reward
        self._step_count += 1

        solved = self._is_solved()
        if solved:
            reward += 10.0
            self._total_reward += 10.0
            self._done = True
            return self.get_obs(), reward, True, {"solved": True}

        done = self._step_count >= self._episode_max_steps
        if done:
            self._done = True
        return self.get_obs(), reward, done, {"solved": solved}

    def _is_solved(self) -> bool:
        """Check if all boxes are on targets."""
        for tr, tc in self._targets:
            e = self._element_at(tr, tc)
            if e != E_BOX_ON_TARGET and e != E_AGENT_ON_TARGET:
                # Target square must have a box or agent on it
                # Actually, all targets need boxes on them (agent on target
                # means the target is occupied by agent not box)
                if e != E_BOX_ON_TARGET:
                    return False
        # Double-check: all boxes on targets
        for br, bc in self._boxes:
            if not self._is_box(br, bc):
                continue
            if (br, bc) not in self._targets:
                return False
        return True

    def get_agent_position(self) -> Tuple[int, int]:
        return self._agent_pos

    def get_box_positions(self) -> List[Tuple[int, int]]:
        return self._boxes.copy()

    def get_target_positions(self) -> List[Tuple[int, int]]:
        return self._targets.copy()

    @property
    def done(self) -> bool:
        return self._done
