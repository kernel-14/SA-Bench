```python
import numpy as np
import torch
import math
from typing import Any, Dict, Tuple, List

# Assume Config class is available from config.py
from config import Config

# --- MOCK ISAAC LAB API ---
# This section simulates the behavior of Isaac Lab as we cannot directly
# integrate with it without a proper installation and environment.
# In a real setup, these would be replaced by actual Isaac Lab API calls.

class MockIsaacLabEnv:
    """
    A mock class to simulate an Isaac Lab environment for a specific robot.
    It returns random data with correct dimensions.
    """
    def __init__(self, robot_type: str, delta_t: float, obs_raw_dim: int, action_dim: int, device: str = "cpu"):
        """
        Initializes the mock Isaac Lab environment.

        Args:
            robot_type: The type of robot.
            delta_t: The simulation time step.
            obs_raw_dim: The total dimension of the raw observation space.
            action_dim: The dimension of the action space.
            device: The device to use ("cpu" or "cuda").
        """
        self.robot_type = robot_type
        self.delta_t = delta_t
        self.obs_raw_dim = obs_raw_dim
        self.action_dim = action_dim
        self.device = device
        self.current_raw_state = None
        self.current_sim_step = 0
        self.max_sim_steps_per_episode = 500 # Example episode length for mock

        print(f"MockIsaacLabEnv initialized for {robot_type} on {device} with raw obs dim: {obs_raw_dim}.")

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Simulates resetting the environment.
        Returns a random initial raw state and an empty info dict.
        """
        print(f"MockIsaacLabEnv: Resetting environment for {self.robot_type}.")
        # Generate a random raw state for all components. These will then be sliced.
        self.current_raw_state = np.random.rand(self.obs_raw_dim).astype(np.float32) * 2 - 1
        self.current_sim_step = 0
        return self.current_raw_state, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Simulates stepping the environment with a given action.
        Returns a random next raw state, a random reward, a random done flag, and an empty info dict.
        """
        if self.current_raw_state is None:
            raise RuntimeError("Environment must be reset before calling step().")

        self.current_sim_step += 1
        
        # Simulate next raw state based on current_raw_state and action (simplified for mock)
        # For a real simulator, this would be physics-based. Here, it's just random noise around current state.
        raw_next_state = self.current_raw_state + np.random.randn(self.obs_raw_dim).astype(np.float32) * 0.1
        raw_next_state = np.clip(raw_next_state, -1, 1) # Keep values within a range
        
        # Simulate reward - a simple random positive value
        reward = np.random.rand() * 0.1 

        # Simulate done flag - terminate randomly or after max steps
        done = (self.current_sim_step >= self.max_sim_steps_per_episode) or (np.random.rand() < 0.005) # 0.5% chance to terminate early

        # Simulate info
        info = {"episode_len": self.current_sim_step}

        self.current_raw_state = raw_next_state
        return raw_next_state, reward, done, info

    def render(self, mode: str = 'human'):
        """Mock render method."""
        print("MockIsaacLabEnv: Render called (no visual output).")

# --- END MOCK ISAAC LAB API ---


class Environment:
    """
    Interface to the Isaac Lab simulation environment, abstracting environment interactions
    and providing formatted observations, actions, and reward calculations.
    """

    def __init__(self, robot_type: str, config: Config):
        """
        Initializes the Isaac Lab environment and sets up robot-specific configurations.

        Args:
            robot_type: The type of robot ("ANYmal D" or "Unitree G1").
            config: The global configuration object.
        """
        self.robot_type = robot_type
        self.config = config
        self._delta_t = config.environment.delta_t
        self.device = config.global.device

        # --- Define Space Mappings and Dimensions ---
        # These mappings translate between Isaac Lab's raw state and the paper's defined spaces.
        # Format: slice(start_idx, end_idx)
        
        self._obs_wm_dim: int = 0
        self._priv_dim: int = 0
        self._action_dim: int = 0
        self._obs_policy_dim: int = 0

        self._obs_wm_slices: Dict[str, slice] = {}
        self._priv_info_slices: Dict[str, slice] = {}
        # Action space is simpler, usually contiguous and directly corresponds to _action_dim
        self._obs_policy_slices: Dict[str, slice] = {}

        # Default joint positions for r_q_d reward term (q_0 in paper)
        self._default_joint_positions: np.ndarray

        # The raw Isaac Lab observation space will be a concatenation of all
        # components needed for obs_wm and priv_info in a consistent order.
        # This is a conceptual mapping for the mock.
        self._isaac_lab_raw_obs_slices: Dict[str, slice] = {}
        self._isaac_lab_raw_obs_dim: int = 0


        if self.robot_type == "ANYmal D":
            # --- Isaac Lab Raw State Slices (Conceptual for Mock) ---
            # This order must be consistent for _extract_obs_wm and _extract_priv_info
            current_raw_idx = 0
            self._isaac_lab_raw_obs_slices['base_lin_vel'] = slice(current_raw_idx, current_raw_idx + 3)
            current_raw_idx += 3
            self._isaac_lab_raw_obs_slices['base_ang_vel'] = slice(current_raw_idx, current_raw_idx + 3)
            current_raw_idx += 3
            self._isaac_lab_raw_obs_slices['projected_gravity'] = slice(current_raw_idx, current_raw_idx + 3)
            current_raw_idx += 3
            self._isaac_lab_raw_obs_slices['joint_positions'] = slice(current_raw_idx, current_raw_idx + 12)
            current_raw_idx += 12
            self._isaac_lab_raw_obs_slices['joint_velocities'] = slice(current_raw_idx, current_raw_idx + 12)
            current_raw_idx += 12
            self._isaac_lab_raw_obs_slices['joint_torques'] = slice(current_raw_idx, current_raw_idx + 12)
            current_raw_idx += 12
            self._isaac_lab_raw_obs_slices['knee_contact'] = slice(current_raw_idx, current_raw_idx + 4)
            current_raw_idx += 4
            self._isaac_lab_raw_obs_slices['foot_contact'] = slice(current_raw_idx, current_raw_idx + 4)
            current_raw_idx += 4
            self._isaac_lab_raw_obs_dim = current_raw_idx # Total raw observation dimension for mock
            
            # --- World Model Observation (Table S2) ---
            self._obs_wm_slices['base_lin_vel'] = self._isaac_lab_raw_obs_slices['base_lin_vel']
            self._obs_wm_slices['base_ang_vel'] = self._isaac_lab_raw_obs_slices['base_ang_vel']
            self._obs_wm_slices['projected_gravity'] = self._isaac_lab_raw_obs_slices['projected_gravity']
            self._obs_wm_slices['joint_positions'] = self._isaac_lab_raw_obs_slices['joint_positions']
            self._obs_wm_slices['joint_velocities'] = self._isaac_lab_raw_obs_slices['joint_velocities']
            self._obs_wm_slices['joint_torques'] = self._isaac_lab_raw_obs_slices['joint_torques']
            self._obs_wm_dim = sum(s.stop - s.start for s in self._obs_wm_slices.values())
            
            # --- Privileged Information (Table S3) ---
            self._priv_info_slices['knee_contact'] = self._isaac_lab_raw_obs_slices['knee_contact']
            self._priv_info_slices['foot_contact'] = self._isaac_lab_raw_obs_slices['foot_contact']
            self._priv_dim = sum(s.stop - s.start for s in self._priv_info_slices.values())

            # --- Action Space (Table S4) ---
            self._action_dim = 12

            # --- Policy Observation (Table S5) ---
            # These are slices within the _constructed_ obs_policy array
            current_policy_idx = 0
            self._obs_policy_slices['base_lin_vel'] = slice(current_policy_idx, current_policy_idx + 3)
            current_policy_idx += 3
            self._obs_policy_slices['base_ang_vel'] = slice(current_policy_idx, current_policy_idx + 3)
            current_policy_idx += 3
            self._obs_policy_slices['projected_gravity'] = slice(current_policy_idx, current_policy_idx + 3)
            current_policy_idx += 3
            self._obs_policy_slices['velocity_command'] = slice(current_policy_idx, current_policy_idx + 3)
            current_policy_idx += 3
            self._obs_policy_slices['joint_positions'] = slice(current_policy_idx, current_policy_idx + 12)
            current_policy_idx += 12
            self._obs_policy_slices['joint_velocities'] = slice(current_policy_idx, current_policy_idx + 12)
            current_policy_idx += 12
            self._obs_policy_slices['last_actions'] = slice(current_policy_idx, current_policy_idx + 12)
            current_policy_idx += 12
            self._obs_policy_dim = current_policy_idx

            # Default joint positions (example values for ANYmal D's 12 joints, approx standing pose)
            self._default_joint_positions = np.array([
                0.0, -0.4, 0.8,  # Front Right leg (Hip, Thigh, Calf)
                0.0, -0.4, 0.8,  # Front Left leg
                0.0, -0.4, 0.8,  # Rear Right leg
                0.0, -0.4, 0.8   # Rear Left leg
            ], dtype=np.float32)

            self._mock_isaac_lab_env = MockIsaacLabEnv(
                robot_type, self._delta_t, self._isaac_lab_raw_obs_dim, self._action_dim, self.device
            )

        elif self.robot_type == "Unitree G1":
            # --- Isaac Lab Raw State Slices (Conceptual for Mock) ---
            current_raw_idx = 0
            self._isaac_lab_raw_obs_slices['base_lin_vel'] = slice(current_raw_idx, current_raw_idx + 3)
            current_raw_idx += 3
            self._isaac_lab_raw_obs_slices['base_ang_vel'] = slice(current_raw_idx, current_raw_idx + 3)
            current_raw_idx += 3
            self._isaac_lab_raw_obs_slices['projected_gravity'] = slice(current_raw_idx, current_raw_idx + 3)
            current_raw_idx += 3
            self._isaac_lab_raw_obs_slices['joint_positions'] = slice(current_raw_idx, current_raw_idx + 29)
            current_raw_idx += 29
            self._isaac_lab_raw_obs_slices['joint_velocities'] = slice(current_raw_idx, current_raw_idx + 29)
            current_raw_idx += 29
            self._isaac_lab_raw_obs_slices['joint_torques'] = slice(current_raw_idx, current_raw_idx + 29)
            current_raw_idx += 29
            self._isaac_lab_raw_obs_slices['body_contact'] = slice(current_raw_idx, current_raw_idx + 26)
            current_raw_idx += 26
            self._isaac_lab_raw_obs_slices['foot_height'] = slice(current_raw_idx, current_raw_idx + 2)
            current_raw_idx += 2
            self._isaac_lab_raw_obs_slices['foot_velocity'] = slice(current_raw_idx, current_raw_idx + 2)
            current_raw_idx += 2
            self._isaac_lab_raw_obs_dim = current_raw_idx
            
            # --- World Model Observation (Table S2) ---
            self._obs_wm_slices['base_lin_vel'] = self._isaac_lab_raw_obs_slices['base_lin_vel']
            self._obs_wm_slices['base_ang_vel'] = self._isaac_lab_raw_obs_slices['base_ang_vel']
            self._obs_wm_slices['projected_gravity'] = self._isaac_lab_raw_obs_slices['projected_gravity']
            self._obs_wm_slices['joint_positions'] = self._isaac_lab_raw_obs_slices['joint_positions']
            self._obs_wm_slices['joint_velocities'] = self._isaac_lab_raw_obs_slices['joint_velocities']
            self._obs_wm_slices['joint_torques'] = self._isaac_lab_raw_obs_slices['joint_torques']
            self._obs_wm_dim = sum(s.stop - s.start for s in self._obs_wm_slices.values())

            # --- Privileged Information (Table S3) ---
            self._priv_info_slices['body_contact'] = self._isaac_lab_raw_obs_slices['body_contact']
            self._priv_info_slices['foot_height'] = self._isaac_lab_raw_obs_slices['foot_height']
            self._priv_info_slices['foot_velocity'] = self._isaac_lab_raw_obs_slices['foot_velocity']
            self._priv_dim = sum(s.stop - s.start for s in self._priv_info_slices.values())

            # --- Action Space (Table S4) ---
            self._action_dim = 29

            # --- Policy Observation (Table S5) ---
            current_policy_idx = 0
            self._obs_policy_slices['base_lin_vel'] = slice(current_policy_idx, current_policy_idx + 3)
            current_policy_idx += 3
            self._obs_policy_slices['base_ang_vel'] = slice(current_policy_idx, current_policy_idx + 3)
            current_policy_idx += 3
            self._obs_policy_slices['projected_gravity'] = slice(current_policy_idx, current_policy_idx + 3)
            current_policy_idx += 3
            self._obs_policy_slices['velocity_command'] = slice(current_policy_idx, current_policy_idx + 3)
            current_policy_idx += 3
            self._obs_policy_slices['joint_positions'] = slice(current_policy_idx, current_policy_idx + 29)
            current_policy_idx += 29
            self._obs_policy_slices['joint_velocities'] = slice(current_policy_idx, current_policy_idx + 29)
            current_policy_idx += 29
            self._obs_policy_slices['last_actions'] = slice(current_policy_idx, current_policy_idx + 29)
            current_policy_idx += 29
            self._obs_policy_dim = current_policy_idx

            # Default joint positions (example values for Unitree G1's 29 joints)
            self._default_joint_positions = np.zeros(29, dtype=np.float32) # Placeholder for G1
            
            self._mock_isaac_lab_env = MockIsaacLabEnv(
                robot_type, self._delta_t, self._isaac_lab_raw_obs_dim, self._action_dim, self.device
            )
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")

        # Store reward weights
        # Robot type in config is e.g. "ANYmal D", but key in config.rewards is "ANYmal_D"
        reward_key = self.robot_type.replace(" ", "_")
        self._reward_weights = self.config.rewards.get(reward_key)
        if not self._reward_weights:
             raise ValueError(f"Reward weights not found for robot type: {self.robot_type} in config.rewards.")
        self._reward_temperatures_vxy = self.config.rewards.reward_temperature_vxy
        self._reward_temperatures_wz = self.config.rewards.reward_temperature_wz

        # Internal state for policy observation construction
        self._last_actions = np.zeros(self._action_dim, dtype=np.float32)
        # linear x,y and angular z commands (total 3 dimensions for command_vel)
        self._command_vel = np.zeros(3, dtype=np.float32) 

    def _extract_obs_wm(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Helper to extract world model observations from a raw Isaac Lab state.
        In a real Isaac Lab, this would involve accessing specific buffers.
        For the mock, it relies on the pre-defined _isaac_lab_raw_obs_slices order.
        """
        wm_components = []
        wm_components.append(raw_state[self._isaac_lab_raw_obs_slices['base_lin_vel']])
        wm_components.append(raw_state[self._isaac_lab_raw_obs_slices['base_ang_vel']])
        wm_components.append(raw_state[self._isaac_lab_raw_obs_slices['projected_gravity']])
        wm_components.append(raw_state[self._isaac_lab_raw_obs_slices['joint_positions']])
        wm_components.append(raw_state[self._isaac_lab_raw_obs_slices['joint_velocities']])
        wm_components.append(raw_state[self._isaac_lab_raw_obs_slices['joint_torques']])
        return np.concatenate(wm_components).astype(np.float32)


    def _extract_priv_info(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Helper to extract privileged information from a raw Isaac Lab state.
        For the mock, it relies on the pre-defined _isaac_lab_raw_obs_slices order.
        """
        priv_components = []
        if self.robot_type == "ANYmal D":
            priv_components.append(raw_state[self._isaac_lab_raw_obs_slices['knee_contact']])
            priv_components.append(raw_state[self._isaac_lab_raw_obs_slices['foot_contact']])
        elif self.robot_type == "Unitree G1":
            priv_components.append(raw_state[self._isaac_lab_raw_obs_slices['body_contact']])
            priv_components.append(raw_state[self._isaac_lab_raw_obs_slices['foot_height']])
            priv_components.append(raw_state[self._isaac_lab_raw_obs_slices['foot_velocity']])
        return np.concatenate(priv_components).astype(np.float32)

    def _construct_obs_policy(self, raw_state: np.ndarray, command_vel: np.ndarray) -> np.ndarray:
        """
        Helper to construct policy observations from a raw Isaac Lab state, command, and last actions.
        """
        obs_policy_components = []
        obs_policy_components.append(raw_state[self._isaac_lab_raw_obs_slices['base_lin_vel']])
        obs_policy_components.append(raw_state[self._isaac_lab_raw_obs_slices['base_ang_vel']])
        obs_policy_components.append(raw_state[self._isaac_lab_raw_obs_slices['projected_gravity']])
        obs_policy_components.append(command_vel) # The current command velocity
        obs_policy_components.append(raw_state[self._isaac_lab_raw_obs_slices['joint_positions']])
        obs_policy_components.append(raw_state[self._isaac_lab_raw_obs_slices['joint_velocities']])
        obs_policy_components.append(self._last_actions) # The action taken in the previous step

        return np.concatenate(obs_policy_components).astype(np.float32)


    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Resets the simulation environment.

        Returns:
            A tuple containing:
            - obs_wm: World model observation (np.ndarray).
            - obs_policy: Policy observation (np.ndarray).
            - priv_info: Privileged information (np.ndarray).
            - command_vel: The current velocity command (np.ndarray).
        """
        raw_state, _ = self._mock_isaac_lab_env.reset()

        # Generate a