import abc
import numpy as np
import random
import copy
import heapq
from typing import Any, Dict, List, Tuple, Optional, Callable, Union

# Import BaseEnvironment
from environments.base_environment import BaseEnvironment

# Attempt to import Config from config.py
try:
    from config import Config
except ImportError:
    # Dummy Config class for self-testing or if config.py is not yet available
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


# --- Mini PacMan Specific Constants (for 14 channels) ---
WALL_MP_IDX: int = 0
FOOD_MP_IDX: int = 1
PILL_MP_IDX: int = 2
AGENT_MP_IDX: int = 3
# Ghost channels: MAX_GHOSTS * GHOST_CHANNELS_PER_GHOST starting from GHOST_BASE_MP_IDX
GHOST_BASE_MP_IDX: int = 4
GHOST_CHANNELS_PER_GHOST: int = 2
MAX_GHOSTS: int = 5 # Assuming 5 ghosts max based on 10 channels for ghosts

# Ghost State Codes (for the 'edible status' channel)
GHOST_STATE_NORMAL: int = 0
GHOST_STATE_EDIBLE: int = 1
GHOST_STATE_FLASHING: int = 2 # When edible_timer is low (e.g., last 2 turns)

# Action Mapping for Mini PacMan (Up, Down, Left, Right, No-op)
ACTION_MAP_MP: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1),   # Right
    4: (0, 0)    # No-op
}

# Colors for rendering (RGBA or RGB)
COLOR_MAP_MP: Dict[str, Tuple[int, int, int]] = {
    "wall": (0, 0, 0),          # Black
    "empty": (200, 200, 200),   # Light Grey
    "food": (255, 255, 0),      # Yellow
    "pill": (255, 100, 0),      # Orange
    "agent": (0, 255, 0),       # Green
    "ghost_normal": (255, 0, 0),    # Red
    "ghost_edible": (0, 0, 255),    # Blue
    "ghost_flashing": (100, 100, 255) # Light Blue
}


class Ghost:
    """
    Represents a single ghost in the Mini PacMan environment.
    """
    def __init__(self, ghost_id: int, pos: Tuple[int, int], edible_duration: int) -> None:
        self.id: int = ghost_id
        self.pos: Tuple[int, int] = pos
        self.initial_pos: Tuple[int, int] = pos # For respawning
        self.edible_timer: int = 0
        self.edible_duration: int = edible_duration # Max duration for being edible

    def is_edible(self) -> bool:
        return self.edible_timer > 0

    def is_flashing(self) -> bool:
        # Paper says "flashing on their final two turns of being edible"
        return 0 < self.edible_timer <= 2

    def decrement_edible_timer(self) -> None:
        if self.edible_timer > 0:
            self.edible_timer -= 1

    def make_edible(self) -> None:
        self.edible_timer = self.edible_duration

    def reset(self) -> None:
        self.pos = self.initial_pos
        self.edible_timer = 0


class MiniPacManEnv(BaseEnvironment):
    """
    Implements the Mini PacMan environment, adhering to the BaseEnvironment interface.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the Mini PacMan environment.

        Args:
            config (Config): The configuration object for environment settings.
        """
        super().__init__(config)

        self.grid_height: int = self.config.get('environment.mini_pacman.grid_size')[0]
        self.grid_width: int = self.config.get('environment.mini_pacman.grid_size')[1]
        self.observation_channels: int = self.config.get('environment.mini_pacman.observation_channels')
        
        # Reward structure
        self.reward_step_penalty: float = self.config.get('environment.sokoban.reward_structure.step_penalty', -0.01) # Using Sokoban default if not specified
        self.reward_food: float = self.config.get('environment.mini_pacman.reward_structure.food')
        self.reward_pill: float = self.config.get('environment.mini_pacman.reward_structure.pill')
        self.reward_ghost: float = self.config.get('environment.mini_pacman.reward_structure.ghost')

        # Maze generation parameters
        self.primm_wall_removal_prob: float = self.config.get('environment.mini_pacman.maze_generation.primm_wall_removal_prob')
        self.initial_ghosts_poisson_lambda: float = self.config.get('environment.mini_pacman.maze_generation.initial_ghosts_poisson_lambda')
        
        # Ghost AI parameters
        self.ghost_a_star_heuristic: str = self.config.get('environment.mini_pacman.ghost_a_star_heuristic')
        self.pill_edible_duration: int = 20 # Hardcoded from paper description (section H.1)

        # Episode and level parameters
        self.max_episode_steps_per_level: int = self.config.get('environment.mini_pacman.episode_max_steps')
        
        # Action space
        self._action_space_size: int = len(ACTION_MAP_MP)
        self._observation_space_shape: Tuple[int, ...] = (self.grid_height, self.grid_width, self.observation_channels)

        # Internal state variables (initialized by reset)
        self.maze: np.ndarray # 2D array: 0 for wall, 1 for path
        self.food_grid: np.ndarray # Boolean grid for food presence
        self.pill_grid: np.ndarray # Boolean grid for pill presence
        self.agent_pos: Tuple[int, int]
        self.ghosts: List[Ghost]
        self.score: float
        self.steps_taken_current_level: int
        self.total_steps_in_episode: int
        self.current_level_num: int # 0-indexed, increments when all food eaten
        self.initial_food_count: int

        self.current_seed: Optional[int] = None # Stores seed for current episode for reproducibility
        self.level_rand: random.Random # Per-level random generator for ghost counts and placement
        self.maze_rand: random.Random # Per-maze random generator for maze generation

    def reset(self, level_seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment for a new episode. Generates a new maze, places elements,
        and initializes all state variables.

        Args:
            level_seed (Optional[int]): Seed for level generation/selection for reproducibility.
                                        If None, a random seed is generated.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - The initial symbolic observation of the environment.
                - An info dictionary (currently empty).
        """
        self.current_seed = level_seed if level_seed is not None else random.randint(0, 2**32 - 1)
        # Use two separate random number generators to avoid interaction issues
        self.maze_rand = random.Random(self.current_seed)
        self.level_rand = random.Random(self.current_seed + 1) # A slightly different seed for element placement

        self._generate_maze(self.maze_rand)
        self._place_initial_elements(self.level_rand)

        self.score = 0.0
        self.steps_taken_current_level = 0
        self.total_steps_in_episode = 0
        self.current_level_num = 0

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Applies the chosen action to the environment, advancing its state by one step.

        Args:
            action (int): The action taken by the agent (0:Up, 1:Down, 2:Left, 3:Right, 4:No-op).

        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]:
                - next_observation (np.ndarray): The observation of the environment after taking the action.
                - reward (float): The reward received from taking the action.
                - done (bool): True if the episode has terminated, False otherwise.
                - info (Dict[str, Any]): An info dictionary (currently empty).
        """
        if not (0 <= action < self._action_space_size):
            raise ValueError(f"Invalid action: {action}. Must be between 0 and {self._action_space_size - 1}.")

        reward: float = self.reward_step_penalty
        done: bool = False

        self.steps_taken_current_level += 1
        self.total_steps_in_episode += 1

        # 1. Agent Movement
        dr, dc = ACTION_MAP_MP[action]
        new_agent_r, new_agent_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        if self._is_valid_pos(new_agent_r, new_agent_c) and self.maze[new_agent_r, new_agent_c] == 1: # 1 means path
            self.agent_pos = (new_agent_r, new_agent_c)
        # Else: agent stays in place, implicitly handled by not updating agent_pos

        # 2. Eating Food and Pills
        if self.food_grid[self.agent_pos]:
            reward += self.reward_food
            self.food_grid[self.agent_pos] = False

        if self.pill_grid[self.agent_pos]:
            reward += self.reward_pill
            self.pill_grid[self.agent_pos] = False
            for ghost in self.ghosts:
                ghost.make_edible()

        # 3. Ghost Movement and Edible Timer Decrement
        self._move_ghosts()
        for ghost in self.ghosts:
            ghost.decrement_edible_timer()

        # 4. Collisions
        collision_reward, collision_done = self._check_collisions()
        reward += collision_reward
        if collision_done:
            done = True

        # 5. Level Completion & Episode Termination
        if np.sum(self.food_grid) == 0: # All food eaten
            # Clear eaten ghosts and respawn
            eaten_ghosts_count = sum(1 for ghost in self.ghosts if not ghost.is_edible() and ghost.pos == self.agent_pos) # Check if any ghosts were eaten this step
            self.ghosts = [ghost for ghost in self.ghosts if not (not ghost.is_edible() and ghost.pos == self.agent_pos)] # Remove eaten ghosts
            
            self._advance_level()
        elif self.steps_taken_current_level >= self.max_episode_steps_per_level:
            done = True # Level timeout, episode ends

        return self._get_observation(), reward, done, {}

    def render(self) -> np.ndarray:
        """
        Generates a pixel-based (RGB) image representation of the current Mini PacMan board.
        """
        cell_size = 20 # pixels per grid cell
        img = np.zeros((self.grid_height * cell_size, self.grid_width * cell_size, 3), dtype=np.uint8)

        # Draw maze walls and empty spaces
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                color = COLOR_MAP_MP["wall"] if self.maze[r, c] == 0 else COLOR_MAP_MP["empty"]
                img[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = color
        
        # Draw food
        food_coords = np.argwhere(self.food_grid)
        for r, c in food_coords:
            center_x, center_y = c * cell_size + cell_size // 2, r * cell_size + cell_size // 2
            # Draw a small yellow dot
            radius = cell_size // 4
            for x_offset in range(-radius, radius + 1):
                for y_offset in range(-radius, radius + 1):
                    if x_offset**2 + y_offset**2 <= radius**2:
                        img[center_y + y_offset, center_x + x_offset] = COLOR_MAP_MP["food"]

        # Draw pills
        pill_coords = np.argwhere(self.pill_grid)
        for r, c in pill_coords:
            center_x, center_y = c * cell_size + cell_size // 2, r * cell_size + cell_size // 2
            # Draw an orange circle
            radius = cell_size // 2 - 2
            for x_offset in range(-radius, radius + 1):
                for y_offset in range(-radius, radius + 1):
                    if x_offset**2 + y_offset**2 <= radius**2:
                        img[center_y + y_offset, center_x + x_offset] = COLOR_MAP_MP["pill"]

        # Draw agent
        r, c = self.agent_pos
        img[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = COLOR_MAP_MP["agent"]

        # Draw ghosts
        for ghost in self.ghosts:
            r, c = ghost.pos
            if ghost.is_edible():
                color = COLOR_MAP_MP["ghost_flashing"] if ghost.is_flashing() else COLOR_MAP_MP["ghost_edible"]
            else:
                color = COLOR_MAP_MP["ghost_normal"]
            img[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = color

        return img

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a deep copy of the environment's internal state.
        This state is a dictionary containing all necessary information to fully
        reconstruct the environment.
        """
        state_dict: Dict[str, Any] = {
            'maze': self.maze.copy(),
            'food_grid': self.food_grid.copy(),
            'pill_grid': self.pill_grid.copy(),
            'agent_pos': self.agent_pos,
            'ghosts': [copy.deepcopy(g) for g in self.ghosts], # Deep copy ghost objects
            'score': self.score,
            'steps_taken_current_level': self.steps_taken_current_level,
            'total_steps_in_episode': self.total_steps_in_episode,
            'current_level_num': self.current_level_num,
            'initial_food_count': self.initial_food_count,
            'current_seed': self.current_seed,
        }
        return state_dict

    def set_state(self, state_dict: Dict[str, Any]) -> None:
        """
        Sets the environment's internal state from a provided state dictionary.

        Args:
            state_dict (Dict[str, Any]): A dictionary representing a valid environment state.
        """
        self.maze = state_dict['maze'].copy()
        self.food_grid = state_dict['food_grid'].copy()
        self.pill_grid = state_dict['pill_grid'].copy()
        self.agent_pos = state_dict['agent_pos']
        self.ghosts = [copy.deepcopy(g) for g in state_dict['ghosts']]
        self.score = state_dict['score']
        self.steps_taken_current_level = state_dict['steps_taken_current_level']
        self.total_steps_in_episode = state_dict['total_steps_in_episode']
        self.current_level_num = state_dict['current_level_num']
        self.initial_food_count = state_dict['initial_food_count']
        self.current_seed = state_dict['current_seed']
        
        # Re-initialize random generators if seed changes, but ensure they are for this state
        self.maze_rand = random.Random(self.current_seed)
        self.level_rand = random.Random(self.current_seed + 1)


    def get_action_space_size(self) -> int:
        """
        Returns the total number of discrete actions available (Up, Down, Left, Right, No-op).
        """
        return self._action_space_size

    def get_observation_space_shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the symbolic observation tensor (H, W, C).
        """
        return self._observation_space_shape

    def simulate_future_state(self, current_internal_state: Dict[str, Any], action: int) -> Dict[str, Any]:
        """
        Calculates the next internal state that would result from taking 'action'
        in 'current_internal_state', without altering the actual environment's internal state.

        Args:
            current_internal_state (Dict[str, Any]): A snapshot of the internal state dictionary.
            action (int): The action to simulate.

        Returns:
            Dict[str, Any]: The internal state dictionary that would result from the simulation.
        """
        original_env_state = self.get_state()  # Save current state of the live environment
        self.set_state(current_internal_state)  # Load the hypothetical state

        # Perform the step logic. _dry_run_step is designed to return the next internal state directly.
        # This helper avoids altering total_steps_in_episode as it's for simulation not actual progression.
        _ = self._dry_run_step(action)
        next_state_dict = self.get_state() # Get the new state after dry run

        self.set_state(original_env_state)  # Restore the original state of the live environment
        return next_state_dict

    def _dry_run_step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Performs a single step simulation without altering global episode counters
        like total_steps_in_episode. Used by simulate_future_state.
        Returns: next_observation, reward, done, info
        """
        reward: float = self.reward_step_penalty
        done: bool = False
        
        # Save current counts, will restore after dry run to not affect total_steps_in_episode
        original_total_steps_in_episode = self.total_steps_in_episode
        self.total_steps_in_episode = 0 # Temporarily zero out for dry run, only steps_taken_current_level matters

        # 1. Agent Movement
        dr, dc = ACTION_MAP_MP[action]
        new_agent_r, new_agent_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        if self._is_valid_pos(new_agent_r, new_agent_c) and self.maze[new_agent_r, new_agent_c] == 1:
            self.agent_pos = (new_agent_r, new_agent_c)

        # 2. Eating Food and Pills
        if self.food_grid[self.agent_pos]:
            reward += self.reward_food
            self.food_grid[self.agent_pos] = False

        if self.pill_grid[self.agent_pos]:
            reward += self.reward_pill
            self.pill_grid[self.agent_pos] = False
            for ghost in self.ghosts:
                ghost.make_edible()

        # 3. Ghost Movement and Edible Timer Decrement
        self._move_ghosts()
        for ghost in self.ghosts:
            ghost.decrement_edible_timer()

        # 4. Collisions
        collision_reward, collision_done = self._check_collisions()
        reward += collision_reward
        if collision_done:
            done = True

        # 5. Level Completion & Episode Termination (simplified for dry run, no actual advance_level)
        if np.sum(self.food_grid) == 0:
            done = True # If all food eaten in simulation, this level would end
        elif self.steps_taken_current_level + 1 >= self.max_episode_steps_per_level: # +1 for the simulated step
             done = True

        next_observation = self._get_observation()
        info = {} # No extra info for simulation

        self.total_steps_in_episode = original_total_steps_in_episode # Restore original value

        return next_observation, reward, done, info


    def _is_valid_pos(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_height and 0 <= c < self.grid_width

    def _generate_maze(self, maze_rand_gen: random.Random) -> None:
        """
        Generates a maze using a modified Primm's algorithm and applies wall removal.
        """
        # Initialize grid with walls
        self.maze = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8) # 0 for wall, 1 for path

        # Start Primm's algorithm from a random point (must be odd coordinates for a clean maze)
        start_r, start_c = (maze_rand_gen.randrange(1, self.grid_height - 1) // 2) * 2 + 1, \
                           (maze_rand_gen.randrange(1, self.grid_width - 1) // 2) * 2 + 1
        
        self.maze[start_r, start_c] = 1 # Mark as path

        # List of walls to be broken
        frontier = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]: # Neighbors 2 units away
            nr, nc = start_r + dr, start_c + dc
            if self._is_valid_pos(nr, nc):
                frontier.append(((start_r + dr // 2, start_c + dc // 2), (nr, nc))) # (wall_to_break, new_path)

        while frontier:
            wall_idx = maze_rand_gen.randrange(len(frontier))
            (wall_r, wall_c), (next_r, next_c) = frontier.pop(wall_idx)

            if self.maze[next_r, next_c] == 0: # If next cell is a wall (unvisited)
                self.maze[wall_r, wall_c] = 1 # Break the wall
                self.maze[next_r, next_c] = 1 # Make the new cell a path

                # Add new cell's neighbors to frontier
                for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                    nnr, nnc = next_r + dr, next_c + dc
                    if self._is_valid_pos(nnr, nnc) and self.maze[nnr, nnc] == 0:
                        frontier.append(((next_r + dr // 2, next_c + dc // 2), (nnr, nnc)))
        
        # Wall removal (creating loops/shortcuts)
        for r in range(1, self.grid_height - 1):
            for c in range(1, self.grid_width - 1):
                if self.maze[r, c] == 0: # If it's a wall
                    # Check if it's between two path cells (horizontally or vertically)
                    is_between_paths = False
                    if self.maze[r-1, c] == 1 and self.maze[r+1, c] == 1: # Vertical
                        is_between_paths = True
                    if self.maze[r, c-1] == 1 and self.maze[r, c+1] == 1: # Horizontal
                        is_between_paths = True
                    
                    if is_between_paths and maze_rand_gen.random() < self.primm_wall_removal_prob:
                        self.maze[r, c] = 1 # Remove the wall

    def _place_initial_elements(self, level_rand_gen: random.Random) -> None:
        """
        Places agent, food, pills, and ghosts on the generated maze.
        """
        all_path_coords = [(r, c) for r in range(self.grid_height)
                           for c in range(self.grid_width) if self.maze[r, c] == 1]
        level_rand_gen.shuffle(all_path_coords) # Randomize available path cells

        self.food_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        self.pill_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        
        # Place all food
        for r, c in all_path_coords:
            self.food_grid[r, c] = True
        
        self.initial_food_count = np.sum(self.food_grid)

        # Place 4 pills
        pill_count = 0
        placed_elements_pos = set()
        while pill_count < 4 and all_path_coords:
            pos = all_path_coords.pop(0)
            if pos not in placed_elements_pos:
                self.pill_grid[pos] = True
                placed_elements_pos.add(pos)
                pill_count += 1
        
        # Place agent
        if not all_path_coords: # Should not happen if maze is large enough
             raise RuntimeError("No available path cells for agent placement.")
        self.agent_pos = all_path_coords.pop(0)
        placed_elements_pos.add(self.agent_pos)

        # Place ghosts
        self.ghosts = []
        num_ghosts = max(1, self.level_rand.poisson(self.initial_ghosts_poisson_lambda)) # Min 1 ghost
        
        # Ensure num_ghosts does not exceed MAX_GHOSTS or available spots
        num_ghosts = min(num_ghosts, MAX_GHOSTS, len(all_path_coords)) 

        for i in range(num_ghosts):
            if not all_path_coords:
                break # No more free spots
            ghost_pos = all_path_coords.pop(0)
            self.ghosts.append(Ghost(i, ghost_pos, self.pill_edible_duration))
            placed_elements_pos.add(ghost_pos)
        
        # Ensure food/pills are not under agent/ghosts initially
        self.food_grid[self.agent_pos] = False
        self.pill_grid[self.agent_pos] = False
        for ghost in self.ghosts:
            self.food_grid[ghost.pos] = False
            self.pill_grid[ghost.pos] = False


    def _get_observation(self) -> np.ndarray:
        """
        Converts the internal game state into the symbolic (H, W, 14) one-hot observation.
        """
        observation = np.zeros((self.grid_height, self.grid_width, self.observation_channels), dtype=np.uint8)

        # Walls
        wall_coords = np.argwhere(self.maze == 0)
        observation[wall_coords[:, 0], wall_coords[:, 1], WALL_MP_IDX] = 1

        # Food
        food_coords = np.argwhere(self.food_grid)
        observation[food_coords[:, 0], food_coords[:, 1], FOOD_MP_IDX] = 1

        # Pills
        pill_coords = np.argwhere(self.pill_grid)
        observation[pill_coords[:, 0], pill_coords[:, 1], PILL_MP_IDX] = 1

        # Agent
        observation[self.agent_pos[0], self.agent_pos[1], AGENT_MP_IDX] = 1

        # Ghosts
        for ghost in self.ghosts:
            ghost_pos_channel = GHOST_BASE_MP_IDX + ghost.id * GHOST_CHANNELS_PER_GHOST
            ghost_state_channel = ghost_pos_channel + 1
            
            observation[ghost.pos[0], ghost.pos[1], ghost_pos_channel] = 1
            
            if ghost.is_flashing():
                observation[ghost.pos[0], ghost.pos[1], ghost_state_channel] = GHOST_STATE_FLASHING
            elif ghost.is_edible():
                observation[ghost.pos[0], ghost.pos[1], ghost_state_channel] = GHOST_STATE_EDIBLE
            else:
                observation[ghost.pos[0], ghost.pos[1], ghost_state_channel] = GHOST_STATE_NORMAL
        
        return observation

    def _move_ghosts(self) -> None:
        """
        Moves each ghost based on its state (chasing or fleeing) using A* search.
        """
        for ghost in self.ghosts:
            target_pos: Tuple[int, int]
            if ghost.is_edible():
                # Flee: target a random distant corner.
                corners = [(1, 1), (1, self.grid_width - 2),
                           (self.grid_height - 2, 1), (self.grid_height - 2, self.grid_width - 2)]
                
                # Filter to only path cells
                available_corners = [c for c in corners if self._is_valid_pos(c[0],c[1]) and self.maze[c] == 1]
                if available_corners:
                    # Choose corner furthest from agent
                    target_pos = max(available_corners, key=lambda p: self._manhattan_distance(p, self.agent_pos))
                else: # Fallback if no corners are path cells
                    target_pos = (self.level_rand.randrange(self.grid_height), self.level_rand.randrange(self.grid_width))
            else:
                # Chase: target the agent's position
                target_pos = self.agent_pos
            
            path = self._find_path_a_star(ghost.pos, target_pos, self.maze, self.ghost_a_star_heuristic)
            if path and len(path) > 1: # Path[0] is current pos, Path[1] is next step
                ghost.pos = path[1]
            # If no path or path is just current pos, ghost stays still

    def _find_path_a_star(self, start: Tuple[int, int], target: Tuple[int, int],
                           maze: np.ndarray, heuristic_name: str) -> List[Tuple[int, int]]:
        """
        Finds a path from start to target in the maze using A* search.

        Args:
            start (Tuple[int, int]): Starting (r, c) coordinates.
            target (Tuple[int, int]): Target (r, c) coordinates.
            maze (np.ndarray): 2D NumPy array representing the maze (0=wall, 1=path).
            heuristic_name (str): Name of the heuristic function ('manhattan').

        Returns:
            List[Tuple[int, int]]: A list of (r, c) tuples representing the path,
                                   or an empty list if no path exists.
        """
        if not (self._is_valid_pos(start[0], start[1]) and self.maze[start] == 1 and
                self._is_valid_pos(target[0], target[1]) and self.maze[target] == 1):
            return [] # Start or target is invalid/wall

        open_set = [] # Priority queue (f_score, (r,c))
        heapq.heappush(open_set, (0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        f_score: Dict[Tuple[int, int], float] = {start: self._manhattan_distance(start, target)}

        while open_set:
            current_f_score, current_pos = heapq.heappop(open_set)

            if current_pos == target:
                # Reconstruct path
                path = []
                while current_pos in came_from:
                    path.append(current_pos)
                    current_pos = came_from[current_pos]
                path.append(start)
                return path[::-1] # Reverse to get path from start to target

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # 4-directional movement
                neighbor_r, neighbor_c = current_pos[0] + dr, current_pos[1] + dc
                neighbor_pos = (neighbor_r, neighbor_c)

                if self._is_valid_pos(neighbor_r, neighbor_c) and self.maze[neighbor_r, neighbor_c] == 1:
                    tentative_g_score = g_score.get(current_pos, float('inf')) + 1 # Cost of 1 for each step

                    if tentative_g_score < g_score.get(neighbor_pos, float('inf')):
                        came_from[neighbor_pos] = current_pos
                        g_score[neighbor_pos] = tentative_g_score
                        f_score[neighbor_pos] = tentative_g_score + self._manhattan_distance(neighbor_pos, target)
                        heapq.heappush(open_set, (f_score[neighbor_pos], neighbor_pos))
        return [] # No path found

    def _manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculates Manhattan distance between two points."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


    def _check_collisions(self) -> Tuple[float, bool]:
        """
        Checks for agent-ghost collisions and applies rewards/termination logic.

        Returns:
            Tuple[float, bool]: (collision_reward, done_flag)
        """
        collision_reward: float = 0.0
        done_flag: bool = False
        
        # Identify ghosts that will be removed to avoid iterating over modified list
        ghosts_to_remove: List[Ghost] = []

        for ghost in self.ghosts:
            if self.agent_pos == ghost.pos:
                if ghost.is_edible():
                    collision_reward += self.reward_ghost
                    ghosts_to_remove.append(ghost)
                else:
                    done_flag = True # Agent eaten
                    break # No need to check other ghosts, episode ends

        # Remove eaten ghosts and respawn them to their initial position
        for eaten_ghost in ghosts_to_remove:
            eaten_ghost.reset() # Reset to initial position, not despawn entirely
            # Add back to ghosts list or ensure they are still there but reset
            # For simplicity, we are not removing them from the list, just resetting their state and position.
            # The paper implies respawn, so resetting state and position is appropriate.

        return collision_reward, done_flag

    def _advance_level(self) -> None:
        """
        Advances to the next level within an episode: respawns food, pills, and ghosts.
        """
        self.current_level_num += 1
        self.steps_taken_current_level = 0
        
        # Respawn all food and pills in current maze
        self.food_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        self.pill_grid = np.zeros((self.grid_height, self.grid_width), dtype=bool)
        all_path_coords = [(r, c) for r in range(self.grid_height)
                           for c in range(self.grid_width) if self.maze[r, c] == 1]
        
        for r, c in all_path_coords:
            self.food_grid[r, c] = True
        
        pill_count = 0
        level_rand_copy = random.Random(self.level_rand.getstate()) # Use a copy to re-shuffle
        level_rand_copy.shuffle(all_path_coords)
        placed_elements_pos = {self.agent_pos} # Agent should not be covered by food/pills

        while pill_count < 4 and all_path_coords: # Ensure 4 pills are placed
            pos = all_path_coords.pop(0)
            if pos not in placed_elements_pos:
                self.pill_grid[pos] = True
                placed_elements_pos.add(pos)
                pill_count += 1
        
        # Reset ghosts and respawn new ones
        self.ghosts = []
        # "number of ghosts at each subsequent level then increases by the floor of a 0.25 plus
        # the level number times a number drawn from Unif[0, 2]"
        additional_ghosts = np.floor(0.25 + self.current_level_num * self.level_rand.uniform(0, 2))
        num_ghosts = max(1, int(len(self.ghosts) + additional_ghosts)) # Start with previous count + additional
        num_ghosts = min(num_ghosts, MAX_GHOSTS, len(all_path_coords) - len(placed_elements_pos)) # Don't exceed MAX_GHOSTS or available spots
        
        for i in range(num_ghosts):
            if not all_path_coords:
                break
            ghost_pos = all_path_coords.pop(0)
            self.ghosts.append(Ghost(i, ghost_pos, self.pill_edible_duration))
            placed_elements_pos.add(ghost_pos)
        
        # Ensure food/pills are not under agent/ghosts after respawn
        self.food_grid[self.agent_pos] = False
        self.pill_grid[self.agent_pos] = False
        for ghost in self.ghosts:
            self.food_grid[ghost.pos] = False
            self.pill_grid[ghost.pos] = False


if __name__ == '__main__':
    print("--- Testing MiniPacManEnv ---")

    # Dummy Config for testing
    dummy_config_data = {
        'environment': {
            'name': 'MiniPacMan',
            'mini_pacman': {
                'grid_size': [13, 13],
                'observation_channels': 14,
                'maze_generation': {
                    'primm_wall_removal_prob': 0.1, # Reduced for simpler maze initially
                    'initial_ghosts_poisson_lambda': 1.0
                },
                'reward_structure': {
                    'food': 1.0,
                    'pill': 2.0,
                    'ghost': 5.0
                },
                'episode_max_steps': 100 # Short levels for testing
            },
            'sokoban': { # Need sokoban reward for step_penalty if using it as default
                'reward_structure': {
                    'step_penalty': -0.01
                }
            }
        },
        'paths': {
            'data_dir': 'temp_data'
        }
    }
    dummy_config = Config(dummy_config_data)

    env = MiniPacManEnv(dummy_config)
    print(f"Action space size: {env.get_action_space_size()}")
    print(f"Observation space shape: {env.get_observation_space_shape()}")

    obs, info = env.reset(level_seed=123)
    print(f"\n--- Initial State (Symbolic Obs) ---")
    print(f"Obs shape: {obs.shape}")
    print(f"Agent Pos: {env.agent_pos}")
    print(f"Num Ghosts: {len(env.ghosts)}, Ghost 0 Pos: {env.ghosts[0].pos if env.ghosts else 'N/A'}")
    print(f"Initial Food Count: {env.initial_food_count}, Current Food Count: {np.sum(env.food_grid)}")
    print(f"Pill Grid sum: {np.sum(env.pill_grid)}")
    
    # Simple rendering test
    # import matplotlib.pyplot as plt
    # plt.imshow(env.render())
    # plt.title("Initial Mini PacMan State")
    # plt.show()

    total_reward = 0.0
    done = False
    max_test_steps = 20
    step_count = 0

    print("\n--- Running simulation steps ---")
    while not done and step_count < max_test_steps:
        action = random.choice([0, 1, 2, 3]) # Random walk for agent
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        print(f"Step {step_count}: Action={action}, AgentPos={env.agent_pos}, Reward={reward:.2f}, Score={env.score:.2f}, Done={done}, Food Left={np.sum(env.food_grid)}")
        
        # Check if ghost became edible
        for g_idx, ghost in enumerate(env.ghosts):
            if ghost.is_edible():
                print(f"  Ghost {g_idx} edible! Timer: {ghost.edible_timer}, Flashing: {ghost.is_flashing()}")
        
        if step_count % 5 == 0 or done:
            # plt.imshow(env.render())
            # plt.title(f"Mini PacMan State at Step {step_count}")
            # plt.show()
            pass
            
    print(f"\n--- Simulation finished ---")
    print(f"Total steps: {step_count}, Final score: {total_reward:.2f}, Done: {done}")

    # Test get_state and set_state
    print("\n--- Testing get_state and set_state ---")
    initial_state_snap = env.get_state()
    
    # Advance state further
    env.step(0)
    env.step(1)
    print(f"Env agent pos after advancing: {env.agent_pos}")

    env.set_state(initial_state_snap)
    print(f"Env agent pos after restoring: {env.agent_pos}")
    assert env.agent_pos == initial_state_snap['agent_pos']
    assert np.array_equal(env.food_grid, initial_state_snap['food_grid'])
    assert len(env.ghosts) == len(initial_state_snap['ghosts'])

    # Test simulate_future_state
    print("\n--- Testing simulate_future_state ---")
    current_state_for_sim = env.get_state()
    sim_action = 0 # Try moving UP
    
    # Simulate a move from the restored state
    next_sim_state = env.simulate_future_state(current_state_for_sim, sim_action)
    print(f"Original agent pos: {current_state_for_sim['agent_pos']}")
    print(f"Simulated next agent pos with action {sim_action}: {next_sim_state['agent_pos']}")
    
    # Check that the live environment state was not changed
    assert env.agent_pos == current_state_for_sim['agent_pos']
    print("Live environment state remains unchanged after simulation (as expected).")

    # Verify a ghost path with A*
    print("\n--- Testing A* pathfinding ---")
    if env.ghosts:
        test_ghost = env.ghosts[0]
        # Choose a random non-wall target
        target_path_cells = [(r, c) for r in range(env.grid_height)
                             for c in range(env.grid_width) if env.maze[r,c] == 1 and (r,c) != test_ghost.pos]
        if target_path_cells:
            a_star_target = random.choice(target_path_cells)
            path = env._find_path_a_star(test_ghost.pos, a_star_target, env.maze, env.ghost_a_star_heuristic)
            print(f"Ghost {test_ghost.id} at {test_ghost.pos} to target {a_star_target}, path length: {len(path) if path else 'No path'}")
            if path:
                print(f"Path: {path}")
                assert path[0] == test_ghost.pos
                assert path[-1] == a_star_target
                # Validate path segments are valid moves
                for i in range(len(path) - 1):
                    p1, p2 = path[i], path[i+1]
                    assert (abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])) == 1 # Adjacent
                    assert env.maze[p2] == 1 # Not a wall
        else:
            print("No valid path cells for A* target test.")
    else:
        print("No ghosts to test A* pathfinding.")

    print("\n--- MiniPacManEnv testing complete ---")

