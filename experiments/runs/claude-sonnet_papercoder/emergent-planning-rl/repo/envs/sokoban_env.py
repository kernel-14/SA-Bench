## envs/sokoban_env.py
"""Sokoban environment for the emergent planning interpretability pipeline.

This module implements the core Sokoban environment that the DRC agent interacts
with during training and evaluation. It faithfully reproduces the exact dynamics,
reward structure, and observation format described in the paper (Sections 2.2, E.2)
and config.yaml.

The environment uses a component-based state representation:
- walls: fixed set of wall positions (immutable after reset)
- targets: fixed list of target positions (immutable after reset)
- boxes: mutable list of current box positions
- agent_pos: mutable current agent position

This separation allows efficient O(1) lookup for collision detection and
correct handling of overlapping entities (box-on-target, agent-on-target).

The symbolic observation x_t ∈ R^{8×8×7} is reconstructed from these
components at each step, matching the paper's description in Section 2.2.

Thinking steps (forced NO-OP at episode start) are supported via the
thinking_steps parameter. During thinking steps, the action is overridden
to NOOP but the agent's forward pass (and cell state update) still runs
normally — this is the mechanism by which extra test-time compute helps
(Section 5, Appendix A.3).

Example:
    >>> from envs.boxoban_loader import BoxobanLoader
    >>> loader = BoxobanLoader("data/boxoban", split="unfiltered_train")
    >>> env = SokobanEnv(loader, thinking_steps=0)
    >>> obs = env.reset()
    >>> obs.shape
    (8, 8, 7)
    >>> obs, reward, done, info = env.step(0)  # UP
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from envs.boxoban_loader import BoxobanLoader, CELL_CODES


# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

#: Action index to name mapping.
ACTION_NAMES: Dict[int, str] = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "NOOP",
}

#: Action index to (row_delta, col_delta) mapping.
#: UP decreases row index, DOWN increases row index (row 0 is top).
ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),   # UP: row decreases
    1: (+1, 0),   # DOWN: row increases
    2: (0, -1),   # LEFT: col decreases
    3: (0, +1),   # RIGHT: col increases
    4: (0, 0),    # NOOP: no movement
}

#: Action index to direction string (matches CLASS_NAMES in metrics.py).
ACTION_TO_DIR: Dict[int, str] = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "NOOP",
}

#: Total number of actions.
N_ACTIONS: int = 5

# ---------------------------------------------------------------------------
# Cell code constants (must match BoxobanLoader.CELL_CODES)
# ---------------------------------------------------------------------------

#: Integer cell codes for the 7 cell types in the symbolic observation.
#: Index k corresponds to channel k in the one-hot observation vector.
CELL_WALL: int = 0
CELL_EMPTY: int = 1
CELL_BOX: int = 2           # box on empty square
CELL_AGENT: int = 3         # agent on empty square
CELL_BOX_ON_TARGET: int = 4  # box on target square
CELL_AGENT_ON_TARGET: int = 5  # agent on target square
CELL_TARGET: int = 6        # target with nothing on it

#: Number of cell types = number of observation channels.
N_CELL_TYPES: int = 7

# ---------------------------------------------------------------------------
# Pixel rendering colors (RGB) for get_pixel_obs()
# ---------------------------------------------------------------------------

#: RGB color per cell type for pixel visualization.
CELL_COLORS: Dict[int, Tuple[int, int, int]] = {
    CELL_WALL: (70, 70, 70),           # dark gray
    CELL_EMPTY: (242, 232, 196),       # beige
    CELL_BOX: (150, 75, 0),            # brown
    CELL_AGENT: (0, 128, 0),           # green
    CELL_BOX_ON_TARGET: (255, 165, 0), # orange
    CELL_AGENT_ON_TARGET: (0, 200, 0), # bright green
    CELL_TARGET: (255, 0, 0),          # red
}

#: Pixels per grid cell for pixel rendering.
PIXELS_PER_CELL: int = 32

# ---------------------------------------------------------------------------
# Reward constants (from config.yaml env section)
# ---------------------------------------------------------------------------

REWARD_STEP: float = -0.01
REWARD_BOX_ON_TARGET: float = 1.0
REWARD_BOX_OFF_TARGET: float = -1.0
REWARD_SOLVED: float = 10.0

# ---------------------------------------------------------------------------
# Grid size constants (from config.yaml agent section)
# ---------------------------------------------------------------------------

GRID_H: int = 8
GRID_W: int = 8

# Episode length range (from config.yaml env section)
MAX_STEPS_MIN: int = 115
MAX_STEPS_MAX: int = 120


class SokobanEnv:
    """Sokoban environment with symbolic observations and exact paper dynamics.

    Implements the Sokoban environment described in paper Sections 2.2 and E.2.
    Uses a component-based state representation for efficient collision detection
    and correct handling of overlapping entities.

    The environment supports:
    - Standard training episodes (thinking_steps=0)
    - Thinking-step experiments (thinking_steps=5, forces NOOP at episode start)
    - Pinned levels for intervention experiments (level_idx parameter)
    - Out-of-distribution modifications (add_wall, remove_wall, remove_agent)
    - Deep copying for parallel intervention runs (clone method)

    Attributes:
        loader: BoxobanLoader providing level data.
        thinking_steps: Number of forced NOOP steps at episode start.
        level_idx: If set, always use this level index; else random.
        grid_size: Grid dimension (8).
        walls: Set of (row, col) wall positions (fixed per episode).
        targets: List of (row, col) target positions (fixed per episode).
        targets_set: Set version of targets for O(1) lookup.
        boxes: List of (row, col) current box positions (mutable).
        agent_pos: Current (row, col) agent position, or None for blind planning.
        step_count: Number of environment steps taken this episode.
        max_steps: Episode length limit (random in [115, 120]).
        thinking_step_count: Number of thinking steps taken so far.
    """

    def __init__(
        self,
        loader: BoxobanLoader,
        thinking_steps: int = 0,
        level_idx: Optional[int] = None,
    ) -> None:
        """Initialize the Sokoban environment.

        Sets up the environment with the given loader and parameters. The
        actual episode state is not initialized until reset() is called.

        Args:
            loader: BoxobanLoader providing level data. Shared reference
                (not copied) since the loader is read-only.
            thinking_steps: Number of forced NOOP steps at the start of each
                episode. During these steps, the action is overridden to NOOP
                but the agent's cell state is still updated. Default 0 for
                training; set to 5 for thinking-step experiments.
            level_idx: If provided, always use this specific level index from
                the loader. If None, a random level is selected at each reset().
                Used by intervention experiments to pin a specific level.
        """
        self.loader: BoxobanLoader = loader
        self.thinking_steps: int = thinking_steps
        self.level_idx: Optional[int] = level_idx
        self.grid_size: int = GRID_H

        # Episode state — initialized to empty values; populated by reset().
        self.walls: Set[Tuple[int, int]] = set()
        self.targets: List[Tuple[int, int]] = []
        self.targets_set: Set[Tuple[int, int]] = set()
        self.boxes: List[Tuple[int, int]] = []
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.step_count: int = 0
        self.max_steps: int = MAX_STEPS_MAX
        self.thinking_step_count: int = 0

    def reset(self) -> np.ndarray:
        """Reset the environment to a new episode.

        Loads a level (random or pinned), parses it to extract walls, targets,
        boxes, and agent position, and returns the initial symbolic observation.

        The max_steps for this episode is sampled uniformly from [115, 120]
        as specified in config.yaml (env.max_steps_min=115, env.max_steps_max=120)
        and paper Section E.2.

        Returns:
            Initial symbolic observation as np.ndarray of shape (8, 8, 7)
            with dtype float32. Each spatial position contains a 7-dimensional
            one-hot vector encoding the cell type.
        """
        # Load level grid.
        if self.level_idx is not None:
            level_grid: np.ndarray = self.loader.get_level(self.level_idx)
        else:
            level_grid = self.loader.get_random_level()

        # Parse level grid into component-based state.
        self._parse_level(level_grid)

        # Initialize episode counters.
        self.step_count = 0
        self.max_steps = random.randint(MAX_STEPS_MIN, MAX_STEPS_MAX)
        self.thinking_step_count = 0

        return self.get_symbolic_obs()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step.

        Handles thinking-step logic (forcing NOOP for the first thinking_steps
        steps), applies the action via _apply_action(), computes the total
        reward including the step penalty, and checks for episode termination.

        During thinking steps, the action is overridden to NOOP (action=4)
        but the step is otherwise processed normally. This allows the agent's
        cell state to be updated during thinking steps, which is the mechanism
        by which extra test-time compute improves performance (Section 5).

        Args:
            action: Integer action in {0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=NOOP}.

        Returns:
            Tuple of (obs, reward, done, info) where:
            - obs: np.ndarray of shape (8, 8, 7), the new symbolic observation.
            - reward: float, total reward for this step (step penalty + action reward).
            - done: bool, True if episode is terminated (solved or max_steps reached).
            - info: dict with keys 'solved', 'step_count', 'thinking_step',
              'agent_pos', 'prev_agent_pos', 'boxes', 'prev_boxes', 'action',
              'box_pushed', 'box_push_from'.
        """
        # Record state before action for info dict and concept labeling.
        prev_agent_pos: Optional[Tuple[int, int]] = self.agent_pos
        prev_boxes: List[Tuple[int, int]] = list(self.boxes)

        # Thinking step: override action to NOOP.
        is_thinking_step: bool = self.thinking_step_count < self.thinking_steps
        if is_thinking_step:
            action = 4  # NOOP
            self.thinking_step_count += 1

        # Apply action and get action-specific reward.
        action_reward, solved, box_pushed, box_push_from = self._apply_action(action)

        # Total reward = step penalty + action reward.
        reward: float = REWARD_STEP + action_reward

        # Increment step counter.
        self.step_count += 1

        # Check termination.
        done: bool = solved or (self.step_count >= self.max_steps)

        # Build info dict for concept labeling and analysis.
        info: Dict = {
            "solved": solved,
            "step_count": self.step_count,
            "thinking_step": is_thinking_step,
            "agent_pos": self.agent_pos,
            "prev_agent_pos": prev_agent_pos,
            "boxes": list(self.boxes),
            "prev_boxes": prev_boxes,
            "action": action,
            "box_pushed": box_pushed,
            "box_push_from": box_push_from,
        }

        return self.get_symbolic_obs(), reward, done, info

    def get_symbolic_obs(self) -> np.ndarray:
        """Construct the symbolic observation from the current environment state.

        Builds the 8×8×7 one-hot observation x_t ∈ R^{8×8×7} described in
        paper Section 2.2 and E.2. Each grid square is encoded as a 7-dimensional
        one-hot vector where the active index corresponds to the cell type.

        The cell type for each square is determined by checking the component-
        based state (walls, targets, boxes, agent_pos) in priority order:
        wall > agent_on_target > agent > box_on_target > box > target > empty.

        Returns:
            np.ndarray of shape (8, 8, 7) with dtype float32. Exactly one
            element per spatial position is 1.0; all others are 0.0.
        """
        obs: np.ndarray = np.zeros((GRID_H, GRID_W, N_CELL_TYPES), dtype=np.float32)

        boxes_set: Set[Tuple[int, int]] = set(self.boxes)

        for r in range(GRID_H):
            for c in range(GRID_W):
                pos: Tuple[int, int] = (r, c)

                if pos in self.walls:
                    cell_type = CELL_WALL
                elif self.agent_pos is not None and pos == self.agent_pos:
                    if pos in self.targets_set:
                        cell_type = CELL_AGENT_ON_TARGET
                    else:
                        cell_type = CELL_AGENT
                elif pos in boxes_set:
                    if pos in self.targets_set:
                        cell_type = CELL_BOX_ON_TARGET
                    else:
                        cell_type = CELL_BOX
                elif pos in self.targets_set:
                    cell_type = CELL_TARGET
                else:
                    cell_type = CELL_EMPTY

                obs[r, c, cell_type] = 1.0

        return obs

    def get_pixel_obs(self) -> np.ndarray:
        """Render the current board state as an RGB pixel image.

        Used by PlanVisualizer to render the Sokoban board background before
        overlaying plan arrows. Each grid cell is rendered as a
        PIXELS_PER_CELL × PIXELS_PER_CELL colored square.

        Returns:
            np.ndarray of shape (GRID_H * PIXELS_PER_CELL, GRID_W * PIXELS_PER_CELL, 3)
            with dtype uint8. RGB values in [0, 255].
        """
        img_h: int = GRID_H * PIXELS_PER_CELL
        img_w: int = GRID_W * PIXELS_PER_CELL
        img: np.ndarray = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        boxes_set: Set[Tuple[int, int]] = set(self.boxes)

        for r in range(GRID_H):
            for c in range(GRID_W):
                pos: Tuple[int, int] = (r, c)

                if pos in self.walls:
                    cell_type = CELL_WALL
                elif self.agent_pos is not None and pos == self.agent_pos:
                    if pos in self.targets_set:
                        cell_type = CELL_AGENT_ON_TARGET
                    else:
                        cell_type = CELL_AGENT
                elif pos in boxes_set:
                    if pos in self.targets_set:
                        cell_type = CELL_BOX_ON_TARGET
                    else:
                        cell_type = CELL_BOX
                elif pos in self.targets_set:
                    cell_type = CELL_TARGET
                else:
                    cell_type = CELL_EMPTY

                color: Tuple[int, int, int] = CELL_COLORS.get(
                    cell_type, (128, 128, 128)
                )

                # Fill the pixel block for this cell.
                r_start: int = r * PIXELS_PER_CELL
                r_end: int = r_start + PIXELS_PER_CELL
                c_start: int = c * PIXELS_PER_CELL
                c_end: int = c_start + PIXELS_PER_CELL

                img[r_start:r_end, c_start:c_end, 0] = color[0]
                img[r_start:r_end, c_start:c_end, 1] = color[1]
                img[r_start:r_end, c_start:c_end, 2] = color[2]

        return img

    def clone(self) -> "SokobanEnv":
        """Create a deep copy of the current environment state.

        Used by the intervention engine to run the same level multiple times
        with different interventions without side effects. The loader is shared
        (read-only reference), but all mutable state is deep-copied.

        Returns:
            A new SokobanEnv instance with identical state to self. Modifying
            the returned environment does not affect self.

        Note:
            Does NOT use copy.deepcopy on the loader to avoid copying all
            900k levels. Only the mutable episode state is copied.
        """
        new_env: SokobanEnv = SokobanEnv.__new__(SokobanEnv)

        # Shared read-only references.
        new_env.loader = self.loader
        new_env.thinking_steps = self.thinking_steps
        new_env.level_idx = self.level_idx
        new_env.grid_size = self.grid_size

        # Deep copy of mutable episode state.
        new_env.walls = set(self.walls)
        new_env.targets = list(self.targets)
        new_env.targets_set = set(self.targets_set)
        new_env.boxes = list(self.boxes)
        new_env.agent_pos = self.agent_pos  # tuple is immutable

        # Scalar state.
        new_env.step_count = self.step_count
        new_env.max_steps = self.max_steps
        new_env.thinking_step_count = self.thinking_step_count

        return new_env

    def add_wall(self, pos: Tuple[int, int]) -> None:
        """Add a wall at the given position (OOD experiment support).

        Used in Appendix A.2.8 (blocked-route planning) experiments where
        a wall is added mid-episode to block an obvious route.

        Args:
            pos: (row, col) position to add a wall. If the position contains
                a box or the agent, they are not moved — the caller is
                responsible for ensuring the position is empty.
        """
        self.walls.add(pos)

    def remove_wall(self, pos: Tuple[int, int]) -> None:
        """Remove a wall at the given position (OOD experiment support).

        Used in Appendix A.2.9 (new-route planning) experiments where a wall
        is removed mid-episode to open up a new optimal route.

        Args:
            pos: (row, col) position to remove a wall. If the position is not
                a wall, this is a no-op (uses set.discard).
        """
        self.walls.discard(pos)

    def remove_agent(self) -> None:
        """Remove the agent from the board (blind planning experiment support).

        Used in Appendix A.2.6 (blind planning) experiments where the agent
        is not present in the level. Sets agent_pos to None.

        After calling this, get_symbolic_obs() will not render the agent,
        and step() will process actions without moving the agent.
        """
        self.agent_pos = None

    def _is_solved(self) -> bool:
        """Check if all boxes are on targets (episode solved).

        Returns:
            True if every box position is in the targets set, False otherwise.
        """
        return all(box in self.targets_set for box in self.boxes)

    def _apply_action(
        self, action: int
    ) -> Tuple[float, bool, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Apply an action to the environment and compute the action reward.

        Implements the exact Sokoban transition dynamics from paper Section E.2:
        - Agent moves in the specified direction
        - If agent moves into a box, the box is pushed in the same direction
        - If push would move box into wall or another box, neither moves
        - Agent cannot push two adjacent boxes simultaneously
        - Agent cannot pull boxes

        Args:
            action: Integer action in {0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=NOOP}.

        Returns:
            Tuple of (action_reward, solved, box_pushed, box_push_from) where:
            - action_reward: float, reward from this action (excluding step penalty).
              Includes box-on-target (+1.0), box-off-target (-1.0), solve (+10.0).
            - solved: bool, True if the level is now solved.
            - box_pushed: Optional[Tuple[int,int]], the new position of the pushed
              box (if a box was pushed), or None.
            - box_push_from: Optional[Tuple[int,int]], the original position of the
              pushed box (if a box was pushed), or None.
        """
        # NOOP: no movement, no reward.
        if action == 4:
            return 0.0, False, None, None

        # Agent not present (blind planning): no movement.
        if self.agent_pos is None:
            return 0.0, False, None, None

        delta: Tuple[int, int] = ACTION_DELTAS[action]
        new_agent_r: int = self.agent_pos[0] + delta[0]
        new_agent_c: int = self.agent_pos[1] + delta[1]
        new_agent_pos: Tuple[int, int] = (new_agent_r, new_agent_c)

        # Check bounds.
        if not self._in_bounds(new_agent_pos):
            return 0.0, False, None, None

        # Check if new agent position is a wall.
        if new_agent_pos in self.walls:
            return 0.0, False, None, None

        action_reward: float = 0.0
        box_pushed: Optional[Tuple[int, int]] = None
        box_push_from: Optional[Tuple[int, int]] = None

        # Build a set for O(1) box lookup.
        boxes_set: Set[Tuple[int, int]] = set(self.boxes)

        # Check if agent is pushing a box.
        if new_agent_pos in boxes_set:
            # Compute where the box would be pushed to.
            new_box_r: int = new_agent_r + delta[0]
            new_box_c: int = new_agent_c + delta[1]
            new_box_pos: Tuple[int, int] = (new_box_r, new_box_c)

            # Check if box can be pushed (not into wall, not into another box,
            # not out of bounds). The paper states: "If the move an agent
            # attempts to perform would involve pushing a box into a non-empty
            # square — that is, a square containing either a wall or another
            # box — neither the box nor the agent moves."
            if not self._in_bounds(new_box_pos):
                return 0.0, False, None, None

            if new_box_pos in self.walls:
                return 0.0, False, None, None

            if new_box_pos in boxes_set:
                # Cannot push box into another box.
                return 0.0, False, None, None

            # Box can be pushed. Find its index in self.boxes.
            box_idx: int = self.boxes.index(new_agent_pos)
            box_push_from = new_agent_pos  # original box position

            # Check if box was on a target before being pushed off.
            if new_agent_pos in self.targets_set:
                action_reward += REWARD_BOX_OFF_TARGET  # -1.0

            # Move the box.
            self.boxes[box_idx] = new_box_pos
            box_pushed = new_box_pos

            # Check if box landed on a target.
            if new_box_pos in self.targets_set:
                action_reward += REWARD_BOX_ON_TARGET  # +1.0

        # Move the agent.
        self.agent_pos = new_agent_pos

        # Check if level is solved.
        solved: bool = self._is_solved()
        if solved:
            action_reward += REWARD_SOLVED  # +10.0

        return action_reward, solved, box_pushed, box_push_from

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within the 8×8 grid bounds.

        Args:
            pos: (row, col) position to check.

        Returns:
            True if 0 <= row < GRID_H and 0 <= col < GRID_W, False otherwise.
        """
        return 0 <= pos[0] < GRID_H and 0 <= pos[1] < GRID_W

    def _parse_level(self, level_grid: np.ndarray) -> None:
        """Parse a level grid into the component-based state representation.

        Extracts walls, targets, boxes, and agent position from the integer-
        coded grid produced by BoxobanLoader. Targets are permanent features
        of the level and include squares currently occupied by boxes or the
        agent (box_on_target, agent_on_target).

        Args:
            level_grid: np.ndarray of shape (8, 8) with dtype int8, containing
                integer cell codes from BoxobanLoader.CELL_CODES.
        """
        self.walls = set()
        self.targets = []
        self.boxes = []
        self.agent_pos = None

        for r in range(GRID_H):
            for c in range(GRID_W):
                code: int = int(level_grid[r, c])
                pos: Tuple[int, int] = (r, c)

                if code == CELL_CODES["#"]:  # wall
                    self.walls.add(pos)
                elif code == CELL_CODES[" "]:  # empty
                    pass
                elif code == CELL_CODES["$"]:  # box on empty
                    self.boxes.append(pos)
                elif code == CELL_CODES["@"]:  # agent on empty
                    self.agent_pos = pos
                elif code == CELL_CODES["*"]:  # box on target
                    self.boxes.append(pos)
                    self.targets.append(pos)
                elif code == CELL_CODES["+"]:  # agent on target
                    self.agent_pos = pos
                    self.targets.append(pos)
                elif code == CELL_CODES["."]:  # target (empty)
                    self.targets.append(pos)
                # Unknown codes are treated as empty (no action needed).

        # Build the targets set for O(1) lookup.
        self.targets_set = set(self.targets)
