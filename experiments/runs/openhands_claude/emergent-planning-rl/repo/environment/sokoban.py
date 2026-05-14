import numpy as np
import random
from typing import Optional, Tuple, List, Dict
from copy import deepcopy

from config import CELL_TYPES, DIRECTION_TO_DELTA, ACTION_TO_DIRECTION


WALL = CELL_TYPES["WALL"]
EMPTY = CELL_TYPES["EMPTY"]
BOX = CELL_TYPES["BOX"]
AGENT = CELL_TYPES["AGENT"]
BOX_ON_TARGET = CELL_TYPES["BOX_ON_TARGET"]
AGENT_ON_TARGET = CELL_TYPES["AGENT_ON_TARGET"]
TARGET = CELL_TYPES["TARGET"]

NUM_CELL_TYPES = 7


class SokobanEnv:
    """
    Sokoban environment with symbolic 8x8x7 observations.
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=NOOP
    """

    def __init__(
        self,
        grid_size: int = 8,
        min_steps: int = 115,
        max_steps: int = 120,
        reward_step: float = -0.01,
        reward_box_on_target: float = 1.0,
        reward_box_off_target: float = -1.0,
        reward_solved: float = 10.0,
    ):
        self.grid_size = grid_size
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.reward_step = reward_step
        self.reward_box_on_target = reward_box_on_target
        self.reward_box_off_target = reward_box_off_target
        self.reward_solved = reward_solved

        self.grid: Optional[np.ndarray] = None
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.step_count: int = 0
        self.max_episode_steps: int = 0
        self.done: bool = False

    def reset(self, level: np.ndarray) -> np.ndarray:
        """
        Reset environment with a given level grid.
        level: (H, W) integer array with CELL_TYPES values.
        """
        self.grid = level.copy()
        self.step_count = 0
        self.max_episode_steps = random.randint(self.min_steps, self.max_steps)
        self.done = False

        agent_positions = list(zip(*np.where(
            (self.grid == AGENT) | (self.grid == AGENT_ON_TARGET)
        )))
        assert len(agent_positions) == 1, "Level must have exactly one agent"
        self.agent_pos = agent_positions[0]

        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert not self.done, "Episode is done, call reset()"
        assert 0 <= action <= 4

        reward = self.reward_step
        self.step_count += 1

        if action < 4:
            direction = ACTION_TO_DIRECTION[action]
            dr, dc = DIRECTION_TO_DELTA[direction]
            r, c = self.agent_pos
            nr, nc = r + dr, c + dc

            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                target_cell = self.grid[nr, nc]

                if target_cell in (BOX, BOX_ON_TARGET):
                    bnr, bnc = nr + dr, nc + dc
                    if 0 <= bnr < self.grid_size and 0 <= bnc < self.grid_size:
                        box_dest = self.grid[bnr, bnc]
                        if box_dest in (EMPTY, TARGET):
                            was_on_target = (target_cell == BOX_ON_TARGET)
                            will_be_on_target = (box_dest == TARGET)

                            if was_on_target:
                                reward += self.reward_box_off_target
                                self.grid[nr, nc] = TARGET
                            else:
                                self.grid[nr, nc] = EMPTY

                            if will_be_on_target:
                                reward += self.reward_box_on_target
                                self.grid[bnr, bnc] = BOX_ON_TARGET
                            else:
                                self.grid[bnr, bnc] = BOX

                            self._move_agent(r, c, nr, nc)
                    # else: box can't be pushed, no movement

                elif target_cell in (EMPTY, TARGET):
                    self._move_agent(r, c, nr, nc)
                # else: wall, no movement

        boxes_on_targets = np.sum(
            (self.grid == BOX_ON_TARGET) | (self.grid == AGENT_ON_TARGET)
        )
        num_boxes = np.sum(
            (self.grid == BOX) | (self.grid == BOX_ON_TARGET)
        )

        solved = (boxes_on_targets == num_boxes) and (num_boxes > 0)
        if solved:
            reward += self.reward_solved

        timeout = self.step_count >= self.max_episode_steps
        self.done = solved or timeout

        info = {
            "solved": solved,
            "timeout": timeout,
            "step_count": self.step_count,
            "boxes_on_targets": int(boxes_on_targets),
        }

        return self._get_obs(), reward, self.done, info

    def _move_agent(self, r: int, c: int, nr: int, nc: int):
        old_cell = self.grid[r, c]
        new_cell = self.grid[nr, nc]

        self.grid[r, c] = TARGET if old_cell == AGENT_ON_TARGET else EMPTY
        self.grid[nr, nc] = AGENT_ON_TARGET if new_cell == TARGET else AGENT
        self.agent_pos = (nr, nc)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.grid_size, self.grid_size, NUM_CELL_TYPES), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                obs[i, j, self.grid[i, j]] = 1.0
        return obs

    def get_grid(self) -> np.ndarray:
        return self.grid.copy()

    def clone(self) -> "SokobanEnv":
        env = SokobanEnv(
            self.grid_size, self.min_steps, self.max_steps,
            self.reward_step, self.reward_box_on_target,
            self.reward_box_off_target, self.reward_solved,
        )
        env.grid = self.grid.copy()
        env.agent_pos = self.agent_pos
        env.step_count = self.step_count
        env.max_episode_steps = self.max_episode_steps
        env.done = self.done
        return env

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        return (self.grid_size, self.grid_size, NUM_CELL_TYPES)

    @property
    def num_actions(self) -> int:
        return 5


def parse_boxoban_level(level_str: str) -> np.ndarray:
    """Parse a Boxoban level string into a grid array."""
    char_to_cell = {
        "#": WALL,
        " ": EMPTY,
        "$": BOX,
        "@": AGENT,
        "*": BOX_ON_TARGET,
        "+": AGENT_ON_TARGET,
        ".": TARGET,
    }

    lines = level_str.strip().split("\n")
    grid = np.zeros((8, 8), dtype=np.int64)

    for i, line in enumerate(lines[:8]):
        for j, ch in enumerate(line[:8]):
            grid[i, j] = char_to_cell.get(ch, EMPTY)

    return grid


def run_episode(
    env: SokobanEnv,
    level: np.ndarray,
    policy_fn,
    thinking_steps: int = 0,
    collect_trajectory: bool = False,
) -> Dict:
    """
    Run a single episode.
    policy_fn: callable(obs, hidden_state) -> (action, hidden_state, extras)
    Returns episode info and optionally the full trajectory.
    """
    obs = env.reset(level)
    hidden_state = None
    total_reward = 0.0
    trajectory = [] if collect_trajectory else None
    done = False

    for t in range(thinking_steps):
        action = 4  # NOOP
        obs_tensor, hidden_state, extras = policy_fn(obs, hidden_state, force_action=action)
        if collect_trajectory:
            trajectory.append({
                "obs": obs.copy(),
                "action": action,
                "hidden_state": extras.get("hidden_state_copy"),
                "step": t,
                "thinking": True,
            })

    while not done:
        obs_tensor, hidden_state, extras = policy_fn(obs, hidden_state)
        action = extras["action"]
        next_obs, reward, done, info = env.step(action)
        total_reward += reward

        if collect_trajectory:
            trajectory.append({
                "obs": obs.copy(),
                "action": action,
                "hidden_state": extras.get("hidden_state_copy"),
                "step": env.step_count,
                "thinking": False,
            })

        obs = next_obs

    return {
        "total_reward": total_reward,
        "solved": info["solved"],
        "step_count": info["step_count"],
        "trajectory": trajectory,
    }
