import numpy as np
import random
import copy
from typing import Any, Dict, List, Tuple, Optional, Union

# Import BaseEnvironment
from environments.base_environment import BaseEnvironment

# Import Config. If config.py is not yet available, a dummy Config class will be used.
try:
    from config import Config
except ImportError:
    class Config:
        def __init__(self, data: Dict = None):
            self._data = data if data is not None else {}
        def get(self, key: str, default: Any = None) -> Any:
            keys = key.split('.')
            current = self._data
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            return current
        def set(self, key: str, value: Any) -> None:
            keys = key.split('.')
            d = self._data
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        def save(self, output_path: str) -> None: pass
    print("Warning: Could not import 'Config' from 'config.py'. Using a dummy Config class.")

# Import LevelGenerator
try:
    from utils.level_generator import LevelGenerator
    # Ensure constants from LevelGenerator are available for internal mapping
    from utils.level_generator import WALL_IDX, EMPTY_IDX, BOX_ON_EMPTY_IDX, AGENT_ON_EMPTY_IDX, \
                                    BOX_ON_TARGET_IDX, AGENT_ON_TARGET_IDX, TARGET_EMPTY_IDX
except ImportError:
    print("Warning: Could not import 'LevelGenerator' or its constants. Using dummy class and local constants.")
    class LevelGenerator:
        def __init__(self, config: Config): pass
        def generate_sokoban_level(self, level_type: str, seed: Optional[int] = None, params: Dict = {}) -> np.ndarray:
            # Dummy implementation for a simple 8x8 empty level for testing
            h, w = self.config.get('environment.sokoban.grid_size', [8, 8])
            c = self.config.get('environment.sokoban.observation_channels', 7)
            level = np.zeros((h, w, c), dtype=np.uint8)
            # Default empty cells
            for r in range(h):
                for col in range(w):
                    level[r, col, EMPTY_IDX] = 1
            # Add walls around perimeter
            level[:, 0, :] = 0; level[:, 0, WALL_IDX] = 1
            level[:, w-1, :] = 0; level[:, w-1, WALL_IDX] = 1
            level[0, :, :] = 0; level[0, :, WALL_IDX] = 1
            level[h-1, :, :] = 0; level[h-1, :, WALL_IDX] = 1
            # Add a single agent, box, target for a minimal solvable state
            level[1, 1, :] = 0; level[1, 1, AGENT_ON_EMPTY_IDX] = 1
            level[2, 2, :] = 0; level[2, 2, BOX_ON_EMPTY_IDX] = 1
            level[3, 3, :] = 0; level[3, 3, TARGET_EMPTY_IDX] = 1
            return level

    # Define constants locally if not imported
    WALL_IDX = 0
    EMPTY_IDX = 1
    BOX_ON_EMPTY_IDX = 2
    AGENT_ON_EMPTY_IDX = 3
    BOX_ON_TARGET_IDX = 4
    AGENT_ON_TARGET_IDX = 5
    TARGET_EMPTY_IDX = 6


# Internal 2D representation codes for simpler logic (0-6 distinct integer values)
# These map directly to the one-hot channel indices for easier conversion,
# but internally represent combined states using single integer values.
# Example: If a cell has `AGENT_ON_EMPTY_IDX`, its internal_code will be `AGENT_ON_EMPTY_IDX`.
# If it has `BOX_ON_TARGET_IDX`, its internal_code will be `BOX_ON_TARGET_IDX`.
# This is a direct mapping as the one-hot indices represent distinct, non-overlapping states.

# Actions (mapping to deltas: (row_delta, col_delta))
ACTION_MAP: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1),   # Right
    4: (0, 0)    # No-op
}

class SokobanEnv(BaseEnvironment):
    """
    Implements the Sokoban environment, adhering to the BaseEnvironment interface.
    Manages game logic, state transitions, rewards, and provides symbolic observations.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the Sokoban environment.

        Args:
            config (Config): The configuration object for environment settings.
        """
        super().__init__(config)

        # Environment dimensions
        self.grid_height: int = self.config.get('environment.sokoban.grid_size')[0]
        self.grid_width: int = self.config.get('environment.sokoban.grid_size')[1]
        self.observation_channels: int = self.config.get('environment.sokoban.observation_channels')

        # Reward structure
        self.reward_step_penalty: float = self.config.get('environment.sokoban.reward_structure.step_penalty')
        self.reward_box_on_target: float = self.config.get('environment.sokoban.reward_structure.box_on_target')
        self.reward_box_off_target: float = self.config.get('environment.sokoban.reward_structure.box_off_target')
        self.reward_all_boxes_on_targets: float = self.config.get('environment.sokoban.reward_structure.all_boxes_on_targets')

        # Episode length
        self.episode_length_min: int = self.config.get('environment.sokoban.episode_length_min')
        self.episode_length_max: int = self.config.get('environment.sokoban.episode_length_max')

        # Level generator
        self._level_generator: LevelGenerator = LevelGenerator(config)

        # Action space
        self._action_space_size: int = len(ACTION_MAP)

        # Observation space shape (H, W, C)
        self._observation_space_shape: Tuple[int, ...] = (self.grid_height, self.grid_width, self.observation_channels)

        # Internal state variables (initialized by reset)
        self._current_state_2d: Optional[np.ndarray] = None  # 2D array representing current board state (int codes)
        self._agent_pos: Tuple[int, int] = (-1, -1)
        self._box_positions: List[Tuple[int, int]] = []
        self._target_positions: List[Tuple[int, int]] = []
        self._num_boxes_on_targets: int = 0
        self._total_steps_in_episode: int = 0
        self._max_episode_steps: int = 0

        # Store the current level's original configuration for reset consistency
        self._current_level_config: Optional[Dict[str, Any]] = None

    def _get_object_positions(self, state_2d: np.ndarray, object_type_code: int) -> List[Tuple[int, int]]:
        """
        Scans a 2D state array to find all occurrences of a given object type code.

        Args:
            state_2d (np.ndarray): The 2D NumPy array representing the environment state.
            object_type_code (int): The integer code of the object to find.

        Returns:
            List[Tuple[int, int]]: A list of (row, col) coordinates for all found objects.
        """
        positions = []
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if state_2d[r, c] == object_type_code:
                    positions.append((r, c))
        return positions

    def _parse_level_initial(self, symbolic_level: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], List[Tuple[int, int]], List[Tuple[int, int]], int]:
        """
        Converts a (H, W, 7) symbolic level into the environment's internal 2D state
        and extracts initial positions of agent, boxes, and targets.

        Args:
            symbolic_level (np.ndarray): The initial symbolic (H, W, 7) representation.

        Returns:
            Tuple[np.ndarray, Tuple[int, int], List[Tuple[int, int]], List[Tuple[int, int]], int]:
                - internal_2d_state: The HxW 2D numpy array with internal integer codes.
                - agent_pos: (row, col) of the agent.
                - box_positions: List of (row, col) for all boxes.
                - target_positions: List of (row, col) for all targets.
                - num_boxes_on_targets: Initial count of boxes on targets.
        """
        internal_2d_state = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        agent_pos: Tuple[int, int] = (-1, -1)
        box_positions: List[Tuple[int, int]] = []
        target_positions: List[Tuple[int, int]] = []
        num_boxes_on_targets: int = 0

        for r in range(self.grid_height):
            for c in range(self.grid_width):
                # Determine what's at (r,c) based on the one-hot vector
                one_hot_vec = symbolic_level[r, c]
                
                # Check for agent
                if one_hot_vec[AGENT_ON_EMPTY_IDX] == 1:
                    internal_2d_state[r, c] = AGENT_ON_EMPTY_IDX
                    agent_pos = (r, c)
                elif one_hot_vec[AGENT_ON_TARGET_IDX] == 1:
                    internal_2d_state[r, c] = AGENT_ON_TARGET_IDX
                    agent_pos = (r, c)
                    target_positions.append((r, c)) # This square is also a target
                
                # Check for box
                elif one_hot_vec[BOX_ON_EMPTY_IDX] == 1:
                    internal_2d_state[r, c] = BOX_ON_EMPTY_IDX
                    box_positions.append((r, c))
                elif one_hot_vec[BOX_ON_TARGET_IDX] == 1:
                    internal_2d_state[r, c] = BOX_ON_TARGET_IDX
                    box_positions.append((r, c))
                    target_positions.append((r, c)) # This square is also a target
                    num_boxes_on_targets += 1
                
                # Check for target (empty)
                elif one_hot_vec[TARGET_EMPTY_IDX] == 1:
                    internal_2d_state[r, c] = TARGET_EMPTY_IDX
                    target_positions.append((r, c))
                
                # Check for wall
                elif one_hot_vec[WALL_IDX] == 1:
                    internal_2d_state[r, c] = WALL_IDX
                
                # Default to empty
                elif one_hot_vec[EMPTY_IDX] == 1:
                    internal_2d_state[r, c] = EMPTY_IDX
                else:
                    # Should not happen if one-hot encoding is correct
                    raise ValueError(f"Ambiguous symbolic state at ({r},{c}): {one_hot_vec}")

        if agent_pos == (-1, -1):
            raise ValueError("No agent found in the initial level state.")
        if not box_positions:
            raise ValueError("No boxes found in the initial level state.")
        if not target_positions:
            raise ValueError("No targets found in the initial level state.")

        # Ensure unique target positions. Could be duplicates if box/agent started on target.
        target_positions = list(set(target_positions))
        
        return internal_2d_state, agent_pos, box_positions, target_positions, num_boxes_on_targets

    def _get_symbolic_observation(self, state_2d: np.ndarray) -> np.ndarray:
        """
        Converts the internal 2D integer state representation to the (H, W, C) symbolic one-hot observation.

        Args:
            state_2d (np.ndarray): The HxW 2D numpy array with internal integer codes.

        Returns:
            np.ndarray: The (H, W, C) symbolic one-hot observation.
        """
        observation = np.zeros((self.grid_height, self.grid_width, self.observation_channels), dtype=np.uint8)
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                val = state_2d[r, c]
                observation[r, c, val] = 1
        return observation

    def reset(self, level_config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state.

        Args:
            level_config (Optional[Dict[str, Any]]): Dictionary specifying the level type, seed, and
                                                    any generation parameters. If None, a random
                                                    level from the unfiltered training set is used.
                                                    Example: {'level_type': 'AgentShortcut', 'seed': 123, 'params': {'corridor_length': 5}}
                                                    Or: {'level_type': 'unfiltered_train', 'seed': 456}

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - The initial symbolic observation (H, W, C).
                - An info dictionary.
        """
        if level_config is None:
            # Default: use a random level from unfiltered_train
            level_type = 'unfiltered_train'
            level_seed = random.randint(0, 1_000_000) # Ensure reproducibility if seed is set
            self._current_level_config = {'level_type': level_type, 'seed': level_seed}
        else:
            self._current_level_config = level_config
            level_type = level_config.get('level_type', 'unfiltered_train')
            level_seed = level_config.get('seed')
            level_params = level_config.get('params', {})
            
        initial_symbolic_level = self._level_generator.generate_sokoban_level(level_type, level_seed, level_params)
        
        (self._current_state_2d,
         self._agent_pos,
         self._box_positions,
         self._target_positions,
         self._num_boxes_on_targets) = self._parse_level_initial(initial_symbolic_level)

        self._total_steps_in_episode = 0
        self._max_episode_steps = random.randint(self.episode_length_min, self.episode_length_max)

        observation = self._get_symbolic_observation(self._current_state_2d)
        info = {
            'agent_pos': self._agent_pos,
            'box_positions': list(self._box_positions), # Make copy to avoid external modification issues
            'target_positions': list(self._target_positions),
            'num_boxes_on_targets': self._num_boxes_on_targets,
            'level_config': copy.deepcopy(self._current_level_config)
        }
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Applies an action to the environment, advancing its state by one step.

        Args:
            action (int): The action to take (0:Up, 1:Down, 2:Left, 3:Right, 4:No-op).

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]:
                - next_observation (np.ndarray): Symbolic observation of the next state.
                - reward (float): Reward received.
                - done (bool): True if episode terminates.
                - info (Dict[str, Any]): Additional info including object positions.
        """
        self._total_steps_in_episode += 1
        current_agent_pos = self._agent_pos
        next_state_2d = np.copy(self._current_state_2d)
        reward: float = self.reward_step_penalty
        done: bool = False
        
        info_sim = self.simulate_future_state(self._current_state_2d, action)
        
        # Unpack simulation results to update actual environment state
        new_board_state = info_sim['next_state_2d']
        moved_agent_pos = info_sim['agent_moved_to']
        box_moved = info_sim['box_moved']
        box_moved_from = info_sim['box_moved_from']
        box_moved_to = info_sim['box_moved_to']
        
        self._current_state_2d = new_board_state
        if moved_agent_pos != current_agent_pos: # Agent successfully moved
             self._agent_pos = moved_agent_pos
        
        old_num_boxes_on_targets = self._num_boxes_on_targets
        
        if box_moved:
            # Update box positions based on the simulation outcome
            old_box_positions = self._box_positions.copy()
            self._box_positions = [] # Clear and rebuild
            for bp in old_box_positions:
                if bp == box_moved_from:
                    self._box_positions.append(box_moved_to)
                else:
                    self._box_positions.append(bp)
            
            # Recalculate boxes on targets for reward
            current_boxes_on_targets = 0
            for box_pos in self._box_positions:
                if box_pos in self._target_positions:
                    current_boxes_on_targets += 1
            self._num_boxes_on_targets = current_boxes_on_targets

            if self._num_boxes_on_targets > old_num_boxes_on_targets:
                reward += self.reward_box_on_target
            elif self._num_boxes_on_targets < old_num_boxes_on_targets:
                reward += self.reward_box_off_target
        
        # Check for episode termination
        if self._num_boxes_on_targets == len(self._box_positions): # All boxes on targets
            reward += self.reward_all_boxes_on_targets
            done = True
        elif self._total_steps_in_episode >= self._max_episode_steps:
            done = True
        
        next_observation = self._get_symbolic_observation(self._current_state_2d)
        
        info = {
            'agent_pos': self._agent_pos,
            'box_positions': list(self._box_positions),
            'target_positions': list(self._target_positions),
            'num_boxes_on_targets': self._num_boxes_on_targets,
            'box_moved_from': box_moved_from, # Pass along for ConceptLabeler
            'box_moved_to': box_moved_to,     # Pass along for ConceptLabeler
            'box_pushed_direction': info_sim['box_pushed_direction'], # Pass along for ConceptLabeler
            'agent_moved_direction': info_sim['agent_moved_direction'], # Pass along for ConceptLabeler
            'is_valid_move': info_sim['is_valid_move'],
            'level_config': copy.deepcopy(self._current_level_config)
        }
        
        return next_observation, reward, done, info

    def render(self) -> np.ndarray:
        """
        Generates a pixel-based (RGB) image representation of the current Sokoban board.
        (Placeholder - actual rendering will be handled by utils.visualization).

        Returns:
            np.ndarray: A dummy RGB NumPy array for now.
        """
        # This will eventually call utils.visualization.plot_sokoban_board
        # For now, return a placeholder image.
        rgb_image = np.zeros((self.grid_height * 32, self.grid_width * 32, 3), dtype=np.uint8) # e.g. 32x32 pixels per tile
        # Could add a simple visual indicator of agent/boxes
        if self._agent_pos != (-1, -1):
            r, c = self._agent_pos
            rgb_image[r*32:(r+1)*32, c*32:(c+1)*32] = [0, 255, 0] # Green for agent
        for br, bc in self._box_positions:
            rgb_image[br*32:(br+1)*32, bc*32:(bc+1)*32] = [255, 165, 0] # Orange for boxes
        for tr, tc in self._target_positions:
            rgb_image[tr*32:(tr+1)*32, tc*32:(tc+1)*32] = [255, 0, 0] # Red for targets
        return rgb_image

    def get_state(self) -> np.ndarray:
        """
        Returns a deep copy of the environment's internal 2D state representation.

        Returns:
            np.ndarray: A copy of the HxW 2D numpy array representing the current state.
        """
        if self._current_state_2d is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        return self._current_state_2d.copy()

    def set_state(self, state_2d: np.ndarray) -> None:
        """
        Sets the environment's internal 2D state to the provided state.
        Recalculates agent, box, and target positions based on the new state.

        Args:
            state_2d (np.ndarray): The HxW 2D numpy array with internal integer codes.
        """
        if state_2d.shape != (self.grid_height, self.grid_width):
            raise ValueError(f"Provided state_2d shape {state_2d.shape} does not match expected "
                             f"environment shape {(self.grid_height, self.grid_width)}.")
        
        self._current_state_2d = state_2d.copy()

        # Re-parse positions based on the new state
        self._agent_pos = (-1, -1)
        self._box_positions = []
        self._target_positions = []
        self._num_boxes_on_targets = 0

        for r in range(self.grid_height):
            for c in range(self.grid_width):
                val = self._current_state_2d[r, c]
                if val == AGENT_ON_EMPTY_IDX or val == AGENT_ON_TARGET_IDX:
                    self._agent_pos = (r, c)
                if val == BOX_ON_EMPTY_IDX or val == BOX_ON_TARGET_IDX:
                    self._box_positions.append((r, c))
                if val == TARGET_EMPTY_IDX or val == AGENT_ON_TARGET_IDX or val == BOX_ON_TARGET_IDX:
                    self._target_positions.append((r, c))
                if val == BOX_ON_TARGET_IDX:
                    self._num_boxes_on_targets += 1
        
        # Ensure unique target positions.
        self._target_positions = list(set(self._target_positions))

    def get_action_space_size(self) -> int:
        """
        Returns the total number of discrete actions available (Up, Down, Left, Right, No-op).

        Returns:
            int: The size of the action space.
        """
        return self._action_space_size

    def get_observation_space_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the symbolic observation tensor (H, W, C).

        Returns:
            Tuple[int, ...]: The shape of the observation space.
        """
        return self._observation_space_shape

    def simulate_future_state(self, current_raw_state: np.ndarray, action: int) -> Dict[str, Any]:
        """
        Deterministically computes the next state and relevant info resulting from an action,
        without altering the environment's actual internal state.
        This is crucial for ConceptLabeler to predict future dynamics.

        Args:
            current_raw_state (np.ndarray): The HxW 2D numpy array representing a hypothetical current state.
            action (int): The action to simulate.

        Returns:
            Dict[str, Any]: A dictionary containing simulation results:
                - 'next_state_2d': The HxW 2D numpy array of the resulting state.
                - 'agent_moved_to': New (r, c) of the agent, or old if no move.
                - 'box_moved': Boolean, True if a box was pushed.
                - 'box_moved_from': (r, c) of the box's original position, or None.
                - 'box_moved_to': (r, c) of the box's new position, or None.
                - 'box_pushed_direction': String ('UP', 'DOWN', 'LEFT', 'RIGHT'), or None.
                - 'agent_moved_direction': String ('UP', 'DOWN', 'LEFT', 'RIGHT'), or None.
                - 'is_valid_move': True if the action resulted in any movement (agent or box).
        """
        sim_state_2d = np.copy(current_raw_state)
        
        # Find agent position in the provided raw state
        sim_agent_pos = (-1, -1)
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                if sim_state_2d[r, c] == AGENT_ON_EMPTY_IDX or sim_state_2d[r, c] == AGENT_ON_TARGET_IDX:
                    sim_agent_pos = (r, c)
                    break
            if sim_agent_pos != (-1, -1):
                break
        
        if sim_agent_pos == (-1, -1):
            raise ValueError("Agent not found in the state provided for simulation.")

        dr, dc = ACTION_MAP[action]
        next_agent_r, next_agent_c = sim_agent_pos[0] + dr, sim_agent_pos[1] + dc
        
        # Default simulation info
        sim_info: Dict[str, Any] = {
            'next_state_2d': sim_state_2d,
            'agent_moved_to': sim_agent_pos, # Assume no move initially
            'box_moved': False,
            'box_moved_from': None,
            'box_moved_to': None,
            'box_pushed_direction': None,
            'agent_moved_direction': None,
            'is_valid_move': False
        }

        if action == 4: # No-op
            return sim_info

        # Check boundaries
        if not (0 <= next_agent_r < self.grid_height and 0 <= next_agent_c < self.grid_width):
            return sim_info # Agent hits boundary, no move

        target_positions = self._get_object_positions(sim_state_2d, TARGET_EMPTY_IDX) + \
                           self._get_object_positions(sim_state_2d, AGENT_ON_TARGET_IDX) + \
                           self._get_object_positions(sim_state_2d, BOX_ON_TARGET_IDX)
        target_positions = list(set(target_positions)) # Ensure unique target positions for lookup

        # Get what's at the next agent position
        next_cell_content = sim_state_2d[next_agent_r, next_agent_c]

        # Case 1: Agent moves into a Wall
        if next_cell_content == WALL_IDX:
            return sim_info # Invalid move, no change

        # Case 2: Agent moves into a Box
        if next_cell_content == BOX_ON_EMPTY_IDX or next_cell_content == BOX_ON_TARGET_IDX:
            next_box_r, next_box_c = next_agent_r + dr, next_agent_c + dc

            # Check if box can be pushed
            if not (0 <= next_box_r < self.grid_height and 0 <= next_box_c < self.grid_width):
                return sim_info # Box pushed out of bounds, invalid

            next_box_cell_content = sim_state_2d[next_box_r, next_box_c]

            # Box pushed into another box or wall
            if next_box_cell_content == WALL_IDX or \
               next_box_cell_content == BOX_ON_EMPTY_IDX or \
               next_box_cell_content == BOX_ON_TARGET_IDX:
                return sim_info # Invalid box push, no change
            
            # Valid box push: Update sim_state_2d
            
            # 1. Clear agent's old position
            if sim_agent_pos in target_positions:
                sim_state_2d[sim_agent_pos] = TARGET_EMPTY_IDX
            else:
                sim_state_2d[sim_agent_pos] = EMPTY_IDX
            
            # 2. Move box to new position
            if (next_box_r, next_box_c) in target_positions:
                sim_state_2d[next_box_r, next_box_c] = BOX_ON_TARGET_IDX
            else:
                sim_state_2d[next_box_r, next_box_c] = BOX_ON_EMPTY_IDX
            
            # 3. Move agent to box's old position
            if (next_agent_r, next_agent_c) in target_positions:
                sim_state_2d[next_agent_r, next_agent_c] = AGENT_ON_TARGET_IDX
            else:
                sim_state_2d[next_agent_r, next_agent_c] = AGENT_ON_EMPTY_IDX

            sim_info['next_state_2d'] = sim_state_2d
            sim_info['agent_moved_to'] = (next_agent_r, next_agent_c)
            sim_info['box_moved'] = True
            sim_info['box_moved_from'] = (next_agent_r, next_agent_c) # Box was at agent's new pos
            sim_info['box_moved_to'] = (next_box_r, next_box_c)
            sim_info['box_pushed_direction'] = {0:'UP', 1:'DOWN', 2:'LEFT', 3:'RIGHT'}[action]
            sim_info['agent_moved_direction'] = {0:'UP', 1:'DOWN', 2:'LEFT', 3:'RIGHT'}[action]
            sim_info['is_valid_move'] = True
            return sim_info

        # Case 3: Agent moves into an Empty square or Target
        if next_cell_content == EMPTY_IDX or next_cell_content == TARGET_EMPTY_IDX:
            
            # Clear agent's old position
            if sim_agent_pos in target_positions:
                sim_state_2d[sim_agent_pos] = TARGET_EMPTY_IDX
            else:
                sim_state_2d[sim_agent_pos] = EMPTY_IDX
            
            # Place agent at new position
            if (next_agent_r, next_agent_c) in target_positions:
                sim_state_2d[next_agent_r, next_agent_c] = AGENT_ON_TARGET_IDX
            else:
                sim_state_2d[next_agent_r, next_agent_c] = AGENT_ON_EMPTY_IDX

            sim_info['next_state_2d'] = sim_state_2d
            sim_info['agent_moved_to'] = (next_agent_r, next_agent_c)
            sim_info['agent_moved_direction'] = {0:'UP', 1:'DOWN', 2:'LEFT', 3:'RIGHT'}[action]
            sim_info['is_valid_move'] = True
            return sim_info
        
        # Fallback for unexpected content, should not be reached with proper state encoding
        return sim_info


if __name__ == '__main__':
    print("--- Testing SokobanEnv ---")

    # Dummy Config for testing
    dummy_config_data = {
        'environment': {
            'name': 'Sokoban',
            'sokoban': {
                'grid_size': [8, 8],
                'observation_channels': 7,
                'episode_length_min': 10,
                'episode_length_max': 20,
                'reward_structure': {
                    'step_penalty': -0.01,
                    'box_on_target': 1.0,
                    'box_off_target': -1.0,
                    'all_boxes_on_targets': 10.0
                },
                'boxoban_paths': {
                    'unfiltered_train': 'data/boxoban/unfiltered_train.txt' # This path won't be used if LevelGenerator is dummy
                }
            }
        },
        'paths': {
            'data_dir': 'temp_data'
        }
    }
    dummy_config = Config(dummy_config_data)

    # Initialize environment
    env = SokobanEnv(dummy_config)
    print(f"Action space size: {env.get_action_space_size()}")
    print(f"Observation space shape: {env.get_observation_space_shape()}")

    # Test reset with default level
    obs, info = env.reset()
    print("\n--- Initial State (Symbolic Obs) ---")
    print(obs.shape)
    print(f"Agent Pos: {info['agent_pos']}, Boxes: {info['box_positions']}, Targets: {info['target_positions']}")
    print(f"Boxes on targets: {info['num_boxes_on_targets']}")
    
    # Test rendering (dummy)
    # import matplotlib.pyplot as plt
    # plt.imshow(env.render())
    # plt.title("Initial Render (Dummy)")
    # plt.show()

    # Test step - move agent up (action 0)
    print("\n--- Stepping (Action: Up) ---")
    next_obs, reward, done, info = env.step(0)
    print(f"Reward: {reward}, Done: {done}")
    print(f"Agent Pos: {info['agent_pos']}, Boxes: {info['box_positions']}")
    print(f"Box moved from: {info['box_moved_from']}, to: {info['box_moved_to']}, dir: {info['box_pushed_direction']}")
    print(f"Agent moved dir: {info['agent_moved_direction']}, Is valid move: {info['is_valid_move']}")
    # plt.imshow(env.render())
    # plt.title("After 1st Step Render (Dummy)")
    # plt.show()

    # Test step - move agent right (action 3), assuming agent is at (0,1) and box at (1,1) in dummy level
    # If the dummy level provides the following initial 2D state:
    # 0 0 0 0 0 0 0 0
    # 0 3 1 1 1 1 1 0  (Agent at (1,1))
    # 0 1 2 1 1 1 1 0  (Box at (2,2))
    # 0 1 1 4 1 1 1 0  (Target at (3,3))
    # 0 1 1 1 1 1 1 0
    # 0 1 1 1 1 1 1 0
    # 0 1 1 1 1 1 1 0
    # 0 0 0 0 0 0 0 0
    
    # Let's simulate a push. Agent at (1,1), Box at (2,2).
    # If agent moves UP (action 0), next_agent_r,c = (0,1). This is a WALL. Should be invalid.
    # If agent moves RIGHT (action 3), next_agent_r,c = (1,2). This is EMPTY. Agent moves.
    # If agent moves DOWN (action 1), next_agent_r,c = (2,1). This is an EMPTY space for our minimal level.
    # Let's try to set up a push scenario.
    print("\n--- Manual state setup for push scenario ---")
    test_state_2d = np.zeros((8, 8), dtype=np.uint8)
    # Walls around perimeter
    test_state_2d[0, :] = WALL_IDX
    test_state_2d[7, :] = WALL_IDX
    test_state_2d[:, 0] = WALL_IDX
    test_state_2d[:, 7] = WALL_IDX
    # Empty in middle
    test_state_2d[1:7, 1:7] = EMPTY_IDX
    
    # Agent, box, target
    test_state_2d[3, 3] = AGENT_ON_EMPTY_IDX # Agent
    test_state_2d[3, 4] = BOX_ON_EMPTY_IDX   # Box to its right
    test_state_2d[3, 6] = TARGET_EMPTY_IDX   # Target further right
    
    env.set_state(test_state_2d)
    print(f"Agent Pos: {env._agent_pos}, Box Pos: {env._box_positions}, Target Pos: {env._target_positions}")
    print(f"Current internal 2D state:\n{env.get_state()}")
    
    # Simulate moving right (action 3) - agent pushes box right
    print("\n--- Simulating push (Action: Right) ---")
    sim_info = env.simulate_future_state(env.get_state(), 3)
    print(f"Simulated next 2D state:\n{sim_info['next_state_2d']}")
    print(f"Sim info: Agent moved to {sim_info['agent_moved_to']}, Box moved from {sim_info['box_moved_from']} to {sim_info['box_moved_to']}")
    
    # Apply the step
    next_obs, reward, done, info = env.step(3)
    print(f"Reward after push: {reward}, Done: {done}")
    print(f"Agent Pos: {info['agent_pos']}, Boxes: {info['box_positions']}")
    print(f"Box moved from: {info['box_moved_from']}, to: {info['box_moved_to']}, dir: {info['box_pushed_direction']}")
    print(f"Agent moved dir: {info['agent_moved_direction']}, Is valid move: {info['is_valid_move']}")
    print(f"Current internal 2D state after step:\n{env.get_state()}")


    # Test completing level
    print("\n--- Testing level completion ---")
    completion_state_2d = np.copy(test_state_2d)
    completion_state_2d[3, 4] = EMPTY_IDX # Clear box from previous spot
    completion_state_2d[3, 5] = AGENT_ON_EMPTY_IDX # Agent at (3,5)
    completion_state_2d[3, 6] = BOX_ON_TARGET_IDX # Box already on target at (3,6)
    env.set_state(completion_state_2d)
    print(f"Current internal 2D state:\n{env.get_state()}")
    print(f"Agent Pos: {env._agent_pos}, Box Pos: {env._box_positions}, Target Pos: {env._target_positions}, Boxes on targets: {env._num_boxes_on_targets}")

    # No-op to trigger end of episode by boxes_on_targets
    _, final_reward, final_done, final_info = env.step(4)
    print(f"Final reward: {final_reward}, Final done: {final_done}, Final boxes on targets: {final_info['num_boxes_on_targets']}")
    assert final_done == True
    assert final_reward > 0 # Should include all_boxes_on_targets reward

    print("\n--- SokobanEnv testing complete ---")

