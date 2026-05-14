## environment.py
"""
Sokoban environment reproducing the symbolic observation and reward structure used in
"Interpreting Emergent Planning in Model-Free RL".

Observations: (8, 8, 7) one‑hot per cell.
Actions: 0=NOOP, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT.
Rewards: step penalty, box‑on/off‑target events, level solve bonus.
Episode ends when all boxes are on targets or a random maximum step count (115‑120) is reached.
"""

import gym
import gym.spaces as spaces
import numpy as np
import random
from typing import List, Tuple, Optional
import cv2

# ------------------------------------------------------------------------------
# Board constants
# ------------------------------------------------------------------------------
BOARD_SIZE = 8
NUM_CHANNELS = 7

# Grid state encoding (mirrors the one‑hot channel index)
WALL = 0
FLOOR = 1
BOX_ON_FLOOR = 2
AGENT_ON_FLOOR = 3
BOX_ON_TARGET = 4
AGENT_ON_TARGET = 5
TARGET = 6

# Action encoding
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

# Direction vectors (row, col) for actions
DIRECTION_VECTORS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

# Rendering tile colours (BGR for OpenCV)
TILE_COLORS_BGR = {
    WALL: (40, 40, 40),               # dark grey
    FLOOR: (200, 200, 200),           # light grey
    BOX_ON_FLOOR: (20, 70, 140),      # brown
    AGENT_ON_FLOOR: (0, 255, 255),    # yellow
    BOX_ON_TARGET: (20, 70, 140),     # brown
    AGENT_ON_TARGET: (0, 255, 255),   # yellow
    TARGET: (80, 80, 255),            # red tint
}


class SokobanEnv(gym.Env):
    """
    Sokoban environment with a symbolic observation of shape (8,8,7).
    The agent must push four boxes onto four targets while navigating an 8×8 grid
    with walls.  Episode length is uniformly sampled from `max_steps_range` and
    the reward structure matches that of the paper.
    """
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self,
                 level_strings: List[str],
                 max_steps_range: Tuple[int, int] = (115, 120),
                 step_penalty: float = -0.01,
                 box_on_target_reward: float = 1.0,
                 box_off_target_reward: float = -1.0,
                 level_solve_reward: float = 10.0,
                 num_boxes: int = 4,
                 num_targets: int = 4,
                 seed: Optional[int] = None):
        """
        Args:
            level_strings: list of 64‑character level strings (Boxoban format).
            max_steps_range: min and max episode length, sampled uniformly per episode.
            step_penalty: reward applied each step (usually negative).
            box_on_target_reward: reward when a box is pushed onto a target.
            box_off_target_reward: reward when a box is pushed off a target.
            level_solve_reward: reward when all boxes are on targets.
            num_boxes: number of boxes required to solve the level (default 4).
            num_targets: number of targets (default 4).
            seed: random seed for reproducibility.
        """
        super().__init__()
        if not level_strings:
            raise ValueError("level_strings must not be empty.")
        self.level_strings = level_strings
        self.max_steps_range = max_steps_range
        self.step_penalty = step_penalty
        self.box_on_target_reward = box_on_target_reward
        self.box_off_target_reward = box_off_target_reward
        self.level_solve_reward = level_solve_reward
        self.num_boxes = num_boxes
        self.num_targets = num_targets

        # Gym spaces
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS),
            dtype=np.float32
        )

        # Internal state
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        self.agent_pos = (0, 0)
        self.steps = 0
        self.episode_max_steps = 0
        self.done = False

        if seed is not None:
            self.seed(seed)

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set the random seed for the environment and return the seed(s).
        """
        super().seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """
        Reset the environment: pick a random level, parse it, reset counters,
        and return the initial symbolic observation.
        """
        level_str = random.choice(self.level_strings)
        self._parse_level(level_str)
        self.steps = 0
        self.episode_max_steps = random.randint(self.max_steps_range[0],
                                                self.max_steps_range[1])
        self.done = False
        return self.get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute a single environment step.

        Args:
            action: integer 0‑4 (NOOP, UP, DOWN, LEFT, RIGHT).

        Returns:
            observation, reward, done, info
        """
        if self.done:
            raise RuntimeError("step() called on a finished episode.")

        self.steps += 1
        reward = self.step_penalty

        if action != NOOP:
            dr, dc = DIRECTION_VECTORS[action]
            agent_r, agent_c = self.agent_pos
            new_r = agent_r + dr
            new_c = agent_c + dc

            # The board is walled, but keep bounds check for safety
            if 0 <= new_r < BOARD_SIZE and 0 <= new_c < BOARD_SIZE:
                target_state = self.grid[new_r, new_c]

                if target_state == WALL:
                    pass  # blocked, no movement
                elif target_state in (FLOOR, TARGET):
                    # Move agent to empty square
                    self._move_agent(new_r, new_c, target_is_floor=(target_state == FLOOR))
                elif target_state in (BOX_ON_FLOOR, BOX_ON_TARGET):
                    # Attempt to push a box
                    push_r = new_r + dr
                    push_c = new_c + dc
                    if 0 <= push_r < BOARD_SIZE and 0 <= push_c < BOARD_SIZE:
                        push_target_state = self.grid[push_r, push_c]
                        if push_target_state in (WALL, BOX_ON_FLOOR, BOX_ON_TARGET):
                            pass  # push blocked
                        elif push_target_state in (FLOOR, TARGET):
                            reward += self._push_box(new_r, new_c, push_r, push_c,
                                                     target_state,
                                                     push_target_state)
                            # Agent position is already updated inside _push_box
                        # else other states (agent) should not occur for push_dest
                # else target_state is agent (should not happen)

        # Termination check
        if np.sum(self.grid == BOX_ON_TARGET) == self.num_boxes:
            reward += self.level_solve_reward
            self.done = True
        elif self.steps >= self.episode_max_steps:
            self.done = True

        return self.get_observation(), reward, self.done, {}

    # ------------------------------------------------------------------
    # Observation and rendering
    # ------------------------------------------------------------------
    def get_observation(self) -> np.ndarray:
        """
        Return the current 8×8×7 one‑hot symbolic observation.
        """
        obs = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_CHANNELS), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                obs[r, c, self.grid[r, c]] = 1.0
        return obs

    # Alias used in the design document
    get_symbolic_representation = get_observation

    def render(self, mode: str = 'rgb_array') -> np.ndarray:
        """
        Render the board as an RGB image (numpy array) using a cell size of 12 pixels.
        """
        cell_size = 12
        img_width = BOARD_SIZE * cell_size
        img_height = BOARD_SIZE * cell_size
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                state = self.grid[r, c]
                colour = TILE_COLORS_BGR[state]
                top = r * cell_size
                left = c * cell_size

                # Fill cell
                cv2.rectangle(img, (left, top),
                              (left + cell_size - 1, top + cell_size - 1), colour, -1)
                # Grid line
                cv2.rectangle(img, (left, top),
                              (left + cell_size - 1, top + cell_size - 1), (50, 50, 50), 1)

                # Target markers
                if state == TARGET:
                    offset = cell_size // 4
                    cv2.line(img, (left, top), (left + cell_size - 1, top + cell_size - 1),
                             (0, 0, 255), 1)
                    cv2.line(img, (left + cell_size - 1, top), (left, top + cell_size - 1),
                             (0, 0, 255), 1)
                elif state in (BOX_ON_TARGET, AGENT_ON_TARGET):
                    centre = (left + cell_size // 2, top + cell_size // 2)
                    radius = cell_size // 5
                    cv2.circle(img, centre, radius, (0, 0, 255), 1)

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        """Clean up (no resources to release)."""
        pass

    # ------------------------------------------------------------------
    # Direct level setting (useful for handcrafted intervention levels)
    # ------------------------------------------------------------------
    def set_level(self, level_str: str) -> np.ndarray:
        """
        Directly set the environment to the given level string and reset episode counters.
        """
        self._parse_level(level_str)
        self.steps = 0
        self.episode_max_steps = random.randint(self.max_steps_range[0],
                                                self.max_steps_range[1])
        self.done = False
        return self.get_observation()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _parse_level(self, level_str: str) -> None:
        """
        Convert a 64‑character Boxoban level string into the internal grid.
        """
        if len(level_str) != BOARD_SIZE * BOARD_SIZE:
            raise ValueError(f"Level string must be exactly {BOARD_SIZE * BOARD_SIZE} chars.")
        char_to_state = {
            '#': WALL, ' ': FLOOR, '$': BOX_ON_FLOOR,
            '@': AGENT_ON_FLOOR, '*': BOX_ON_TARGET,
            '+': AGENT_ON_TARGET, '.': TARGET,
        }
        agent_found = False
        for idx, ch in enumerate(level_str):
            state = char_to_state.get(ch, FLOOR)
            r = idx // BOARD_SIZE
            c = idx % BOARD_SIZE
            self.grid[r, c] = state
            if state in (AGENT_ON_FLOOR, AGENT_ON_TARGET):
                if agent_found:
                    raise ValueError("Level string contains multiple agent positions.")
                self.agent_pos = (r, c)
                agent_found = True
        if not agent_found:
            raise ValueError("No agent starting position found in level string.")

    def _move_agent(self, new_r: int, new_c: int, target_is_floor: bool) -> None:
        """
        Move agent from current position to (new_r, new_c) where the target is
        either FLOOR or TARGET.  Updates grid and self.agent_pos.
        """
        old_r, old_c = self.agent_pos
        old_state = self.grid[old_r, old_c]
        self.grid[old_r, old_c] = FLOOR if old_state == AGENT_ON_FLOOR else TARGET
        self.grid[new_r, new_c] = AGENT_ON_FLOOR if target_is_floor else AGENT_ON_TARGET
        self.agent_pos = (new_r, new_c)

    def _push_box(self,
                  box_r: int, box_c: int,
                  push_r: int, push_c: int,
                  box_state: int,
                  push_dest_state: int) -> float:
        """
        Handle a successful box push.  The agent pushes the box at (box_r,box_c)
        to (push_r,push_c).  Updates the grid and returns the extra reward (beyond
        step penalty) obtained from the push event.
        """
        box_was_on_target = (box_state == BOX_ON_TARGET)
        dest_is_target = (push_dest_state == TARGET)

        # Move box
        self.grid[push_r, push_c] = BOX_ON_TARGET if dest_is_target else BOX_ON_FLOOR

        # Vacated square becomes target if box was on target, else floor
        self.grid[box_r, box_c] = TARGET if box_was_on_target else FLOOR

        # Move agent into the vacated square
        self.grid[box_r, box_c] = AGENT_ON_TARGET if box_was_on_target else AGENT_ON_FLOOR

        # Update old agent position
        old_r, old_c = self.agent_pos
        old_state = self.grid[old_r, old_c]
        self.grid[old_r, old_c] = FLOOR if old_state == AGENT_ON_FLOOR else TARGET

        self.agent_pos = (box_r, box_c)

        # Compute reward components
        extra = 0.0
        if dest_is_target:
            extra += self.box_on_target_reward
        if box_was_on_target:
            extra += self.box_off_target_reward
        return extra
