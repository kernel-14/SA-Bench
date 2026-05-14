"""
Sokoban Environment Implementation.

Implements the Sokoban environment as described in the paper:
- 8x8 grid world
- Symbolic observation x_t in R^(8x8x7) (one-hot encoding of 7 square states)
- Agent moves up/down/left/right to push boxes onto targets
- Episodic, fully-observable, deterministic

Square states (7 channels):
  0: wall
  1: empty
  2: box on empty
  3: agent on empty
  4: box on target
  5: agent on target
  6: target (empty)
"""

import numpy as np
from enum import IntEnum
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


class SquareType(IntEnum):
    """Enumeration of square types for the symbolic Sokoban representation."""
    WALL = 0
    EMPTY = 1
    BOX = 2           # box on empty
    AGENT = 3         # agent on empty
    BOX_ON_TARGET = 4
    AGENT_ON_TARGET = 5
    TARGET = 6        # empty target


# Action space: 0=no-op, 1=up, 2=down, 3=left, 4=right
ACTION_SPACE = 5
ACTION_NAMES = ['NOOP', 'UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_DELTAS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class SokobanState:
    """Internal state representation of a Sokoban board."""
    walls: np.ndarray          # (8, 8) bool - wall positions
    targets: np.ndarray        # (8, 8) bool - target positions
    boxes: np.ndarray          # (8, 8) bool - box positions
    agent_pos: Tuple[int, int]  # (y, x) agent position
    num_boxes: int = 4
    num_targets: int = 4


class SokobanEnv:
    """
    Sokoban environment with symbolic observations.

    Observation: x_t in R^(8x8x7), one-hot encoding per square.
    Action space: 5 discrete actions (noop, up, down, left, right).
    """
    def __init__(self, max_episode_steps: Optional[int] = None, seed: int = 42):
        self.board_size = 8
        self.num_channels = 7
        self.max_episode_steps = max_episode_steps
        self.rng = np.random.RandomState(seed)
        self.state: Optional[SokobanState] = None
        self.steps = 0
        self.boxes_on_targets = 0
        self.total_boxes = 4
        self.total_targets = 4
        # Episode length randomly between 115-120 as described in the paper
        self._episode_length = 120

    def reset(self, level: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset the environment.
        
        Args:
            level: Optional (8, 8) integer array encoding the board layout.
                   If None, uses a default level.
        
        Returns:
            observation x_t in R^(8x8x7)
        """
        if level is not None:
            self._load_level(level)
        else:
            self._load_default_level()
        
        self.steps = 0
        self.boxes_on_targets = 0
        # Count boxes already on targets
        if self.state is not None:
            self.boxes_on_targets = np.sum(self.state.boxes & self.state.targets)
        self._episode_length = self.rng.randint(115, 121)
        return self._get_observation()

    def _load_level(self, level: np.ndarray):
        """Load a level from an integer array."""
        self.state = SokobanState(
            walls=np.zeros((8, 8), dtype=bool),
            targets=np.zeros((8, 8), dtype=bool),
            boxes=np.zeros((8, 8), dtype=bool),
            agent_pos=(0, 0),
            num_boxes=0,
            num_targets=0,
        )
        for y in range(8):
            for x in range(8):
                val = level[y, x]
                if val == SquareType.WALL:
                    self.state.walls[y, x] = True
                elif val == SquareType.TARGET:
                    self.state.targets[y, x] = True
                    self.state.num_targets += 1
                elif val == SquareType.BOX:
                    self.state.boxes[y, x] = True
                    self.state.num_boxes += 1
                elif val == SquareType.AGENT:
                    self.state.agent_pos = (y, x)
                elif val == SquareType.BOX_ON_TARGET:
                    self.state.boxes[y, x] = True
                    self.state.targets[y, x] = True
                    self.state.num_boxes += 1
                    self.state.num_targets += 1
                elif val == SquareType.AGENT_ON_TARGET:
                    self.state.agent_pos = (y, x)
                    self.state.targets[y, x] = True
                    self.state.num_targets += 1

    def _load_default_level(self):
        """Load a simple default level."""
        level = np.zeros((8, 8), dtype=np.int32)
        level[:, :] = SquareType.EMPTY
        level[0, :] = SquareType.WALL
        level[-1, :] = SquareType.WALL
        level[:, 0] = SquareType.WALL
        level[:, -1] = SquareType.WALL
        level[1, 1] = SquareType.AGENT
        level[2, 2] = SquareType.BOX
        level[2, 5] = SquareType.BOX
        level[4, 3] = SquareType.BOX
        level[5, 6] = SquareType.BOX
        level[6, 2] = SquareType.TARGET
        level[6, 5] = SquareType.TARGET
        level[4, 7] = SquareType.TARGET
        level[3, 6] = SquareType.TARGET
        self._load_level(level)

    def _get_observation(self) -> np.ndarray:
        """Generate the symbolic observation x_t in R^(8x8x7)."""
        obs = np.zeros((8, 8, 7), dtype=np.float32)
        ay, ax = self.state.agent_pos
        
        for y in range(8):
            for x in range(8):
                is_wall = self.state.walls[y, x]
                is_target = self.state.targets[y, x]
                is_box = self.state.boxes[y, x]
                is_agent = (y == ay and x == ax)
                
                if is_wall:
                    obs[y, x, SquareType.WALL] = 1.0
                elif is_agent and is_target:
                    obs[y, x, SquareType.AGENT_ON_TARGET] = 1.0
                elif is_agent:
                    obs[y, x, SquareType.AGENT] = 1.0
                elif is_box and is_target:
                    obs[y, x, SquareType.BOX_ON_TARGET] = 1.0
                elif is_box:
                    obs[y, x, SquareType.BOX] = 1.0
                elif is_target:
                    obs[y, x, SquareType.TARGET] = 1.0
                else:
                    obs[y, x, SquareType.EMPTY] = 1.0
        
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: 0=noop, 1=up, 2=down, 3=left, 4=right
        
        Returns:
            observation, reward, done, info
        """
        if action == 0:  # NOOP
            self.steps += 1
            done = self._is_done()
            return self._get_observation(), -0.01, done, {}
        
        dy, dx = ACTION_DELTAS[action]
        ay, ax = self.state.agent_pos
        ny, nx = ay + dy, ax + dx
        
        # Check bounds and walls
        if not (0 <= ny < 8 and 0 <= nx < 8):
            self.steps += 1
            done = self._is_done()
            return self._get_observation(), -0.01, done, {}
        
        if self.state.walls[ny, nx]:
            self.steps += 1
            done = self._is_done()
            return self._get_observation(), -0.01, done, {}
        
        reward = -0.01
        
        # Check if pushing a box
        if self.state.boxes[ny, nx]:
            by, bx = ny + dy, nx + dx
            
            # Check box destination
            if not (0 <= by < 8 and 0 <= bx < 8):
                self.steps += 1
                done = self._is_done()
                return self._get_observation(), reward, done, {}
            
            if self.state.walls[by, bx] or self.state.boxes[by, bx]:
                self.steps += 1
                done = self._is_done()
                return self._get_observation(), reward, done, {}
            
            # Move box
            was_on_target = self.state.targets[ny, nx]
            self.state.boxes[ny, nx] = False
            self.state.boxes[by, bx] = True
            
            # Update rewards
            if was_on_target:
                reward += -1.0  # pushed box off target
                self.boxes_on_targets -= 1
            
            if self.state.targets[by, bx]:
                reward += 1.0  # pushed box onto target
                self.boxes_on_targets += 1
            
            # Move agent
            self.state.agent_pos = (ny, nx)
        else:
            # Move agent (no box)
            self.state.agent_pos = (ny, nx)
        
        self.steps += 1
        
        # Check if all boxes on targets
        if self.boxes_on_targets == self.total_targets:
            reward += 10.0
        
        done = self._is_done()
        return self._get_observation(), reward, done, {}

    def _is_done(self) -> bool:
        """Check if the episode should end."""
        if self.boxes_on_targets == self.total_targets:
            return True
        if self.steps >= self._episode_length:
            return True
        if self.max_episode_steps is not None and self.steps >= self.max_episode_steps:
            return True
        return False

    def get_state_array(self) -> np.ndarray:
        """Get the board state as an integer array (8x8)."""
        board = np.zeros((8, 8), dtype=np.int32)
        ay, ax = self.state.agent_pos
        
        for y in range(8):
            for x in range(8):
                is_wall = self.state.walls[y, x]
                is_target = self.state.targets[y, x]
                is_box = self.state.boxes[y, x]
                is_agent = (y == ay and x == ax)
                
                if is_wall:
                    board[y, x] = SquareType.WALL
                elif is_agent and is_target:
                    board[y, x] = SquareType.AGENT_ON_TARGET
                elif is_agent:
                    board[y, x] = SquareType.AGENT
                elif is_box and is_target:
                    board[y, x] = SquareType.BOX_ON_TARGET
                elif is_box:
                    board[y, x] = SquareType.BOX
                elif is_target:
                    board[y, x] = SquareType.TARGET
                else:
                    board[y, x] = SquareType.EMPTY
        
        return board

    def render(self) -> str:
        """Render the board as an ASCII string."""
        board = self.get_state_array()
        char_map = {
            SquareType.WALL: '#',
            SquareType.EMPTY: ' ',
            SquareType.BOX: 'B',
            SquareType.AGENT: 'A',
            SquareType.BOX_ON_TARGET: 'b',
            SquareType.AGENT_ON_TARGET: 'a',
            SquareType.TARGET: '.',
        }
        lines = []
        for y in range(8):
            line = ''.join(char_map[board[y, x]] for x in range(8))
            lines.append(line)
        return '\n'.join(lines)

    def is_solved(self) -> bool:
        """Check if all boxes are on targets."""
        return self.boxes_on_targets == self.total_targets


def create_level_from_string(level_str: str) -> np.ndarray:
    """
    Create a level array from a string representation.
    
    Characters:
        # = wall
        ' ' = empty
        B = box
        A = agent
        . = target
        b = box on target
        a = agent on target
    """
    lines = [l for l in level_str.strip().split('\n') if l.strip()]
    level = np.zeros((8, 8), dtype=np.int32)
    level[:, :] = SquareType.EMPTY
    
    char_map = {
        '#': SquareType.WALL,
        ' ': SquareType.EMPTY,
        'B': SquareType.BOX,
        'A': SquareType.AGENT,
        '.': SquareType.TARGET,
        'b': SquareType.BOX_ON_TARGET,
        'a': SquareType.AGENT_ON_TARGET,
    }
    
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch in char_map:
                level[y, x] = char_map[ch]
    
    return level


def generate_level_from_state_array(state_array: np.ndarray) -> np.ndarray:
    """Convert a state array to a level that can be loaded."""
    return state_array.copy()
