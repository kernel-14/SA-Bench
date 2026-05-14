# sokoban_environment.py
from typing import Tuple, List, Dict, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

class SokobanEnvironment(gym.Env):
    """
    SokobanEnvironment implements an 8x8 Sokoban grid-world environment following
    the Gymnasium API. It provides initialization, state transitions, and rendering.
    """

    def __init__(self, config: dict):
        """
        Constructor for the Sokoban environment.

        Args:
            config (dict): The configuration dictionary derived from config.yaml.
        """
        self.grid_size = config['environment']['grid_size']
        self.max_steps = config['environment']['max_steps']
        self.rewards = config['environment']['rewards']
        
        # Defining spaces for Gymnasium
        self.action_space = spaces.Discrete(5)  # [UP, DOWN, LEFT, RIGHT, STAY]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.grid_size, self.grid_size, 7), dtype=np.float32
        )
        
        # State variables
        self.agent_position = None
        self.box_positions = None
        self.target_positions = None
        self.wall_positions = None
        self.num_steps = 0

        # Internal state
        self.state = np.zeros((self.grid_size, self.grid_size, 7), dtype=np.float32)
    
    def reset(self) -> dict:
        """
        Resets the environment to a new Sokoban instance.
        
        Returns:
            dict: The initial observation state (8x8x7 tensor).
        """
        # Example level initialization - random placement of walls, boxes, agent
        self.state.fill(0)  # Clear the states
        self.agent_position = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        self.box_positions = [self._random_empty_position() for _ in range(4)]
        self.target_positions = [self._random_empty_position() for _ in range(4)]
        self.wall_positions = [self._random_empty_position() for _ in range(10)]

        # Populate the state grid
        self._update_state()
        self.num_steps = 0

        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Executes an action and updates the environment's state.

        Args:
            action (int): One of five possible discrete actions (UP=0, DOWN=1, LEFT=2, RIGHT=3, STAY=4).

        Returns:
            Tuple[np.ndarray, float, bool, dict]: 
                - Updated state (8x8x7 tensor).
                - Reward for this step (float).
                - Whether the environment is done (bool).
                - Additional metadata (dict).
        """
        if action not in range(5):
            raise ValueError("Invalid action. Must be in range [0, 4].")
        
        initial_step_reward = self.rewards['step_penalty']
        box_push_reward = 0
        done = False

        agent_x, agent_y = self.agent_position
        delta = self._action_to_delta(action)

        if delta is not None:
            new_agent_x = agent_x + delta[0]
            new_agent_y = agent_y + delta[1]

            # Check for boundaries
            if (
                0 <= new_agent_x < self.grid_size and
                0 <= new_agent_y < self.grid_size
            ):
                if self._is_wall(new_agent_x, new_agent_y):
                    # Agent cannot pass through walls
                    pass
                elif self._is_box(new_agent_x, new_agent_y):
                    # Try pushing the box
                    box_push_x = new_agent_x + delta[0]
                    box_push_y = new_agent_y + delta[1]
                    
                    if (
                        0 <= box_push_x < self.grid_size and
                        0 <= box_push_y < self.grid_size and
                        not self._is_wall(box_push_x, box_push_y) and
                        not self._is_box(box_push_x, box_push_y)
                    ):
                        # Move the box
                        self.box_positions.remove((new_agent_x, new_agent_y))
                        self.box_positions.append((box_push_x, box_push_y))
                        
                        # Calculate rewards for box push actions
                        if self._is_target(new_agent_x, new_agent_y):
                            box_push_reward += self.rewards['box_off_target_penalty']
                        if self._is_target(box_push_x, box_push_y):
                            box_push_reward += self.rewards['target_reward']
                        
                        # Update agent position
                        self.agent_position = (new_agent_x, new_agent_y)
                else:
                    # Valid empty move
                    self.agent_position = (new_agent_x, new_agent_y)
        
        # Increment step count and evaluate termination conditions
        self.num_steps += 1
        if self.num_steps >= self.max_steps:
            done = True

        # Check winning condition (all boxes on targets)
        if all(pos in self.target_positions for pos in self.box_positions):
            done = True
            box_push_reward += self.rewards['solve_level_reward']
        
        # Update environment state
        self._update_state()

        return self.state, initial_step_reward + box_push_reward, done, {}

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Provides a visual representation of the current state.

        Args:
            mode (str): Specifies 'human' for display or 'rgb_array' for pixel data.

        Returns:
            Optional[np.ndarray]: RGB array if mode='rgb_array', None otherwise.
        """
        if mode not in ['human', 'rgb_array']:
            raise ValueError("Render mode must be 'human' or 'rgb_array'.")

        pixels = self._generate_pixel_representation()
        if mode == 'human':
            plt.imshow(pixels)
            plt.axis("off")
            plt.show()
        elif mode == 'rgb_array':
            return pixels

    def _is_wall(self, x: int, y: int) -> bool:
        return (x, y) in self.wall_positions

    def _is_box(self, x: int, y: int) -> bool:
        return (x, y) in self.box_positions

    def _is_target(self, x: int, y: int) -> bool:
        return (x, y) in self.target_positions

    def _random_empty_position(self) -> Tuple[int, int]:
        """
        Finds a random empty position that is not occupied by walls, boxes, targets, or the agent.

        Returns:
            Tuple[int, int]: A valid random position.
        """
        while True:
            pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if pos not in self.wall_positions and pos not in self.box_positions and pos not in self.target_positions and pos != self.agent_position:
                return pos

    def _action_to_delta(self, action: int) -> Optional[Tuple[int, int]]:
        """
        Maps an action to a delta movement.

        Args:
            action (int): The action to be taken.

        Returns:
            Optional[Tuple[int, int]]: Delta movement as (dx, dy), or None if the action is `STAY`.
        """
        deltas = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),   # RIGHT
            4: None      # STAY
        }
        return deltas[action]

    def _update_state(self):
        """
        Updates the internal symbolic representation of the board.
        """
        self.state.fill(0)  # Clear the state
        
        # Set walls
        for x, y in self.wall_positions:
            self.state[x, y, 0] = 1  # Wall
        
        # Set boxes
        for x, y in self.box_positions:
            self.state[x, y, 1] = 1  # Box
        
        # Set targets
        for x, y in self.target_positions:
            self.state[x, y, 2] = 1  # Target
        
        # Set the agent
        x, y = self.agent_position
        self.state[x, y, 3] = 1  # Agent

    def _generate_pixel_representation(self) -> np.ndarray:
        """
        Generates a pixel-based visualization of the current environment state.

        Returns:
            np.ndarray: RGB array of shape (8, 8, 3).
        """
        # Example mapping (color scheme can be adjusted for better aesthetics)
        color_mapping = {
            0: np.array([0, 0, 0]),       # Wall = Black
            1: np.array([255, 128, 0]),   # Box = Orange
            2: np.array([128, 128, 255]), # Target = Blue
            3: np.array([128, 255, 128]), # Agent = Green
        }
        pixels = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.state[x, y, 0]:  # Wall
                    pixels[x, y] = color_mapping[0]
                elif self.state[x, y, 1]:  # Box
                    pixels[x, y] = color_mapping[1]
                elif self.state[x, y, 2]:  # Target
                    pixels[x, y] = color_mapping[2]
                elif self.state[x, y, 3]:  # Agent
                    pixels[x, y] = color_mapping[3]
        
        return pixels
