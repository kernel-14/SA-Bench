"""
Sokoban environment implementation.
Based on the paper: "Interpreting Emergent Planning in Model-Free Reinforcement Learning"

The environment uses symbolic representations: x_t in R^{8x8x7}
Each square is one of 7 states:
  0: wall
  1: empty floor
  2: box on empty floor
  3: agent on empty floor
  4: box on target
  5: agent on target
  6: target (empty)
"""

import numpy as np
from typing import Tuple, Optional, List


# Action constants
ACTION_NOOP = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4

# Direction deltas: (row_delta, col_delta)
DIRECTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

# Square state constants
WALL = 0
EMPTY = 1
BOX = 2
AGENT = 3
BOX_ON_TARGET = 4
AGENT_ON_TARGET = 5
TARGET = 6

# Number of channels in symbolic representation
NUM_CHANNELS = 7

# Grid size
GRID_SIZE = 8


class SokobanEnv:
    """
    Sokoban environment with symbolic observations.
    
    The agent observes a symbolic representation x_t in R^{8x8x7}.
    Each square is represented as a 7-dimensional one-hot vector.
    
    Actions: 0=noop, 1=up, 2=down, 3=left, 4=right
    
    Rewards:
      -0.01 per step
      +1 when pushing a box onto a target
      -1 when pushing a box off a target
      +10 when all boxes are on targets (episode solved)
    """
    
    def __init__(self, max_steps: Optional[int] = None):
        """
        Args:
            max_steps: Maximum episode length. If None, sampled uniformly from [115, 120].
        """
        self.max_steps = max_steps
        self.grid = None
        self.agent_pos = None
        self.step_count = 0
        self._episode_max_steps = 0
        self.done = False
        
    def reset(self, level: np.ndarray) -> np.ndarray:
        """
        Reset environment with a given level.
        
        Args:
            level: 8x8 integer array with square states
            
        Returns:
            Initial symbolic observation
        """
        assert level.shape == (GRID_SIZE, GRID_SIZE), f"Level must be {GRID_SIZE}x{GRID_SIZE}"
        self.grid = level.copy().astype(np.int32)
        
        # Find agent position
        agent_positions = np.argwhere(
            (self.grid == AGENT) | (self.grid == AGENT_ON_TARGET)
        )
        assert len(agent_positions) == 1, "Level must have exactly one agent"
        self.agent_pos = tuple(agent_positions[0])
        
        self.step_count = 0
        self.done = False
        
        if self.max_steps is not None:
            self._episode_max_steps = self.max_steps
        else:
            self._episode_max_steps = np.random.randint(115, 121)
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer action (0=noop, 1=up, 2=down, 3=left, 4=right)
            
        Returns:
            (observation, reward, done, info)
        """
        assert not self.done, "Episode is done, call reset()"
        
        reward = -0.01
        
        if action != ACTION_NOOP:
            dr, dc = DIRECTION_DELTAS[action]
            r, c = self.agent_pos
            nr, nc = r + dr, c + dc
            
            # Check bounds
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                target_square = self.grid[nr, nc]
                
                if target_square == WALL:
                    # Can't move into wall
                    pass
                elif target_square in (BOX, BOX_ON_TARGET):
                    # Try to push box
                    bnr, bnc = nr + dr, nc + dc
                    if 0 <= bnr < GRID_SIZE and 0 <= bnc < GRID_SIZE:
                        box_dest = self.grid[bnr, bnc]
                        if box_dest not in (WALL, BOX, BOX_ON_TARGET):
                            # Push is valid
                            box_was_on_target = (target_square == BOX_ON_TARGET)
                            box_goes_to_target = (box_dest == TARGET)
                            
                            # Update box destination
                            if box_goes_to_target:
                                self.grid[bnr, bnc] = BOX_ON_TARGET
                                reward += 1.0
                            else:
                                self.grid[bnr, bnc] = BOX
                            
                            # Update box source (now agent moves here)
                            if box_was_on_target:
                                reward -= 1.0
                                self.grid[nr, nc] = AGENT_ON_TARGET
                            else:
                                self.grid[nr, nc] = AGENT
                            
                            # Update agent source
                            old_r, old_c = self.agent_pos
                            if self.grid[old_r, old_c] == AGENT_ON_TARGET:
                                self.grid[old_r, old_c] = TARGET
                            else:
                                self.grid[old_r, old_c] = EMPTY
                            
                            self.agent_pos = (nr, nc)
                else:
                    # Move agent to empty/target square
                    old_r, old_c = self.agent_pos
                    if self.grid[old_r, old_c] == AGENT_ON_TARGET:
                        self.grid[old_r, old_c] = TARGET
                    else:
                        self.grid[old_r, old_c] = EMPTY
                    
                    if target_square == TARGET:
                        self.grid[nr, nc] = AGENT_ON_TARGET
                    else:
                        self.grid[nr, nc] = AGENT
                    
                    self.agent_pos = (nr, nc)
        
        self.step_count += 1
        
        # Check win condition
        boxes_on_targets = np.sum(self.grid == BOX_ON_TARGET)
        total_targets = np.sum(
            (self.grid == BOX_ON_TARGET) | 
            (self.grid == TARGET) | 
            (self.grid == AGENT_ON_TARGET)
        )
        
        if boxes_on_targets == total_targets and total_targets > 0:
            reward += 10.0
            self.done = True
        elif self.step_count >= self._episode_max_steps:
            self.done = True
        
        info = {
            'solved': boxes_on_targets == total_targets and total_targets > 0,
            'step_count': self.step_count,
        }
        
        return self._get_observation(), reward, self.done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get symbolic observation as one-hot encoded array.
        
        Returns:
            Array of shape (8, 8, 7) with one-hot encoded square states
        """
        obs = np.zeros((GRID_SIZE, GRID_SIZE, NUM_CHANNELS), dtype=np.float32)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                obs[i, j, self.grid[i, j]] = 1.0
        return obs
    
    def get_grid(self) -> np.ndarray:
        """Return current grid state."""
        return self.grid.copy()
    
    def is_solved(self) -> bool:
        """Check if the level is solved."""
        boxes_on_targets = np.sum(self.grid == BOX_ON_TARGET)
        total_targets = np.sum(
            (self.grid == BOX_ON_TARGET) | 
            (self.grid == TARGET) | 
            (self.grid == AGENT_ON_TARGET)
        )
        return boxes_on_targets == total_targets and total_targets > 0
    
    def get_agent_pos(self) -> Tuple[int, int]:
        """Return current agent position (row, col)."""
        return self.agent_pos
    
    @staticmethod
    def obs_to_grid(obs: np.ndarray) -> np.ndarray:
        """Convert symbolic observation back to integer grid."""
        return np.argmax(obs, axis=-1).astype(np.int32)
