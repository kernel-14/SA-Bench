
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict

from config import GlobalConfig

class TrajectoryDataset(Dataset):
    """
    Dataset for RWM training, constructed by sliding a window of size M+N over collected trajectories.
    Generates historical observation-action sequences and corresponding future targets.
    """
    def __init__(self, cfg: GlobalConfig, observations: np.ndarray, actions: np.ndarray, privileged_info: np.ndarray):
        self.cfg = cfg
        self.M = cfg.rwm_config.history_horizon_M
        self.N = cfg.rwm_config.forecast_horizon_N
        self.total_sequence_length = self.M + self.N

        self.observations = torch.from_numpy(observations).float()
        self.actions = torch.from_numpy(actions).float()
        self.privileged_info = torch.from_numpy(privileged_info).float()

        # Ensure all inputs have the same number of time steps
        assert self.observations.shape[0] == self.actions.shape[0] == self.privileged_info.shape[0]
        self.num_total_steps = self.observations.shape[0]

        # Determine the number of valid windows
        self.num_windows = self.num_total_steps - self.total_sequence_length + 1
        if self.num_windows < 1:
            raise ValueError(f"Not enough data to form a single window. Need at least {self.total_sequence_length} steps, but got {self.num_total_steps}.")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a window of data for RWM training.
        A window consists of:
        - observations_history: (M, obs_dim)
        - actions_history: (M, action_dim)
        - actions_forecast: (N, action_dim) (actions from M to M+N-1)
        - observations_target: (N, obs_dim) (observations from M to M+N-1)
        - privileged_info_target: (N, priv_info_dim) (privileged info from M to M+N-1)
        """
        start_idx = idx
        end_idx = idx + self.total_sequence_length

        # History for RWM input
        observations_history = self.observations[start_idx : start_idx + self.M] # (M, obs_dim)
        actions_history = self.actions[start_idx : start_idx + self.M]           # (M, action_dim)

        # Actions for forecast (from M to M+N-1 relative to window start)
        actions_forecast = self.actions[start_idx + self.M : end_idx]         # (N, action_dim)

        # Targets for loss calculation (observations and privileged info from M to M+N-1)
        observations_target = self.observations[start_idx + self.M : end_idx] # (N, obs_dim)
        privileged_info_target = self.privileged_info[start_idx + self.M : end_idx] # (N, priv_info_dim)

        return {
            "observations_history": observations_history,
            "actions_history": actions_history,
            "actions_forecast": actions_forecast,
            "observations_target": observations_target,
            "privileged_info_target": privileged_info_target,
        }

class ReplayBuffer:
    """
    A simple replay buffer to store environment transitions.
    Used for both RWM training data and MBPO-PPO policy updates.
    """
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, priv_info_dim: int, policy_obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.priv_info_dim = priv_info_dim
        self.policy_obs_dim = policy_obs_dim

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.privileged_info = np.zeros((capacity, priv_info_dim), dtype=np.float32)
        self.policy_observations = np.zeros((capacity, policy_obs_dim), dtype=np.float32) # For PPO updates

        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool, priv_info: np.ndarray, policy_obs: np.ndarray):
        """
        Adds a single transition to the replay buffer.
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.privileged_info[self.ptr] = priv_info
        self.policy_observations[self.ptr] = policy_obs

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of transitions from the replay buffer.
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": torch.from_numpy(self.observations[indices]).float(),
            "actions": torch.from_numpy(self.actions[indices]).float(),
            "rewards": torch.from_numpy(self.rewards[indices]).float(),
            "next_observations": torch.from_numpy(self.next_observations[indices]).float(),
            "dones": torch.from_numpy(self.dones[indices]).float(),
            "privileged_info": torch.from_numpy(self.privileged_info[indices]).float(),
            "policy_observations": torch.from_numpy(self.policy_observations[indices]).float(),
        }

    def get_all_rwm_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns all collected data suitable for RWM TrajectoryDataset creation.
        """
        return self.observations[:self.size], self.actions[:self.size], self.privileged_info[:self.size]

    def get_all_ppo_data(self) -> Dict[str, torch.Tensor]:
        """
        Returns all collected data suitable for PPO updates.
        """
        return {
            "policy_observations": torch.from_numpy(self.policy_observations[:self.size]).float(),
            "actions": torch.from_numpy(self.actions[:self.size]).float(),
            "rewards": torch.from_numpy(self.rewards[:self.size]).float(),
            "dones": torch.from_numpy(self.dones[:self.size]).float(),
        }


# --- Mock Environment for demonstration ---
# Since we don't have a real environment or simulator, we create a mock one
# to generate dummy data for the replay buffer.
class MockRobotEnvironment:
    """
    A mock environment to simulate robot interactions and generate data.
    """
    def __init__(self, cfg: GlobalConfig):
        self.cfg = cfg
        self.obs_dim = cfg.rwm_obs_dim
        self.action_dim = cfg.rwm_action_dim
        self.priv_info_dim = cfg.rwm_priv_info_dim
        self.policy_obs_dim = cfg.policy_obs_dim
        self.current_obs = self._reset_obs()
        self.current_priv_info = self._reset_priv_info()
        self.current_policy_obs = self._reset_policy_obs()
        self.steps = 0

    def _reset_obs(self):
        return np.random.rand(self.obs_dim).astype(np.float32)

    def _reset_priv_info(self):
        return np.random.rand(self.priv_info_dim).astype(np.float32)

    def _reset_policy_obs(self):
        return np.random.rand(self.policy_obs_dim).astype(np.float32)

    def reset(self):
        self.current_obs = self._reset_obs()
        self.current_priv_info = self._reset_priv_info()
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.current_policy_obs = self._construct_policy_obs(self.current_obs, self.last_action)
        self.steps = 0
        return self.current_obs, self.current_priv_info, self.current_policy_obs

    def _construct_policy_obs(self, rwm_obs: np.ndarray, last_action: np.ndarray) -> np.ndarray:
        """
        Constructs the policy observation from RWM observation and last action.
        This is a mock implementation based on Table S5.
        """
        policy_obs_slices = self.cfg.policy_obs_slices
        policy_obs_components = []

        # Base linear and angular velocities (from RWM obs 0:6)
        policy_obs_components.append(rwm_obs[0:6])
        # Projected gravity (from RWM obs 6:9)
        policy_obs_components.append(rwm_obs[6:9])
        # Velocity command (mocked for now as zeros/random)
        policy_obs_components.append(np.zeros(3, dtype=np.float32)) # or np.random.rand(3)
        # Joint positions (from RWM obs, slice depends on robot_type)
        policy_obs_components.append(rwm_obs[self.cfg.rwm_obs_slices['q_pos'][0]:self.cfg.rwm_obs_slices['q_pos'][1]])
        # Joint velocities (from RWM obs, slice depends on robot_type)
        policy_obs_components.append(rwm_obs[self.cfg.rwm_obs_slices['q_vel'][0]:self.cfg.rwm_obs_slices['q_vel'][1]])
        # Last actions
        policy_obs_components.append(last_action)

        return np.concatenate(policy_obs_components)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        """
        Simulates a step in the environment.
        Returns: next_obs, next_priv_info, reward, done, info
        """
        self.steps += 1
        next_obs = self._reset_obs() # Just random for now
        next_priv_info = self._reset_priv_info()
        
        # In a real environment, the actual reward would be calculated here.
        # For this mock, we just return a random reward.
        reward = np.random.rand()

        done = self.steps >= 100 # End episode after 100 steps for example

        self.last_action = action # Update last action
        next_policy_obs = self._construct_policy_obs(next_obs, self.last_action)
        
        info = {'policy_obs': next_policy_obs} # Pass next_policy_obs through info

        self.current_obs = next_obs
        self.current_priv_info = next_priv_info
        self.current_policy_obs = next_policy_obs
        
        return next_obs, next_priv_info, reward, done, info

