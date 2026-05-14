
import random
import numpy as np
import torch
from collections import deque
import gymnasium as gym

from config import Config

class ReplayBuffer:
    def __init__(self, capacity, observation_space_shape, action_dim, state_dtype=np.float32, action_dtype=np.float32):
        self.capacity = capacity
        self.observations = np.empty((capacity, *observation_space_shape), dtype=state_dtype)
        self.actions = np.empty((capacity, action_dim), dtype=action_dtype)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.next_observations = np.empty((capacity, *observation_space_shape), dtype=state_dtype)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, done, next_obs):
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.next_observations[self.idx] = next_obs
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.observations[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.dones[idxs],
            self.next_observations[idxs]
        )

    def __len__(self):
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, observation_space_shape, action_dim, alpha=Config.PROBABILITY_SMOOTHING_ALPHA, min_priority=Config.MINIMUM_PRIORITY, state_dtype=np.float32, action_dtype=np.float32):
        super().__init__(capacity, observation_space_shape, action_dim, state_dtype, action_dtype)
        self.alpha = alpha
        self.min_priority = min_priority
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = min_priority

    def add(self, obs, action, reward, done, next_obs):
        super().add(obs, action, reward, done, next_obs)
        self.priorities[self.idx - 1 if self.idx > 0 else self.capacity - 1] = self.max_priority

    def sample(self, batch_size, beta=0.4): # beta for importance sampling weight
        if self.size == 0:
            return None, None, None, None, None, None, None

        priorities = self.priorities[:self.size]
        # Scale priorities to [min_priority, max_priority]
        # priorities = priorities - priorities.min() + self.min_priority
        # priorities = priorities / priorities.sum()
        
        # Calculate probabilities from priorities
        probs = (priorities ** self.alpha) / (priorities[:self.size] ** self.alpha).sum()
        
        idxs = np.random.choice(self.size, batch_size, p=probs)

        # Compute importance-sampling weights
        total_priority = self.priorities[:self.size].sum()
        min_prob = (self.min_priority ** self.alpha) / total_priority
        max_weight = (1 / min_prob) ** beta
        
        weights = (1 / (self.size * probs[idxs])) ** beta
        weights = weights / max_weight # Normalize weights to [0, 1]

        return (
            self.observations[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.dones[idxs],
            self.next_observations[idxs],
            weights,
            idxs
        )

    def update_priorities(self, idxs, td_errors):
        self.priorities[idxs] = np.maximum(td_errors, self.min_priority)
        self.max_priority = max(self.max_priority, self.priorities.max())

def create_env(env_name, seed, discrete_action_space, is_visual=False):
    # This is a simplified environment creation.
    # Actual implementation might involve wrappers for action repeat, image preprocessing etc.
    env = gym.make(env_name)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class RewardScaler(nn.Module):
    def __init__(self, reward_range=Config.REWARD_RANGE, reward_bins=Config.REWARD_BINS):
        super().__init__()
        self.reward_range = reward_range
        self.reward_bins = reward_bins
        self.register_buffer('bins', torch.linspace(reward_range[0], reward_range[1], reward_bins))

    def two_hot_encode(self, rewards: torch.Tensor):
        """
        Creates a two-hot encoding of scalar rewards.
        The two-hot encoding locations are spaced at increasing non-uniform intervals,
        according to symexp(x) = sign(x) * (exp(abs(x)) - 1) (Hafner et al., 2023).
        
        This implementation assumes uniform binning for simplicity,
        but the paper mentions `symexp` for non-uniform intervals.
        A proper `symexp` implementation for bin edges would be more complex.
        For now, let's stick to uniform `linspace` for `bins` as a practical approximation.
        If rewards are outside the range, they are clipped to the nearest bin.
        """
        rewards = torch.clamp(rewards, self.reward_range[0], self.reward_range[1])
        
        # Determine which bin each reward falls into
        indices_low = torch.searchsorted(self.bins, rewards, right=False) - 1
        indices_high = torch.searchsorted(self.bins, rewards, right=True)

        # Handle edge cases where rewards are below min or above max bin
        indices_low = torch.clamp(indices_low, 0, self.reward_bins - 1)
        indices_high = torch.clamp(indices_high, 0, self.reward_bins - 1)
        
        # For rewards exactly on a bin boundary, searchsorted might give same low/high indices.
        # We need two distinct indices for two-hot.
        # If low == high, adjust high if possible, otherwise adjust low.
        mask = (indices_low == indices_high)
        if torch.any(mask):
            # Try to increment high if not at max
            adjust_high = (indices_high < self.reward_bins - 1) & mask
            indices_high[adjust_high] += 1
            # If still equal (i.e. high was already max), decrement low if not at min
            adjust_low = (indices_low > 0) & mask & (~adjust_high)
            indices_low[adjust_low] -= 1

        # Create one-hot vectors and sum them for two-hot
        two_hot = torch.zeros(rewards.shape[0], self.reward_bins, device=rewards.device)
        two_hot.scatter_(-1, indices_low.unsqueeze(-1), 1.0)
        two_hot.scatter_(-1, indices_high.unsqueeze(-1), 1.0)
        
        return two_hot

    def decode(self, two_hot_logits: torch.Tensor):
        """
        Decodes two-hot logits back to scalar rewards by taking the expectation
        over the bin values.
        """
        probs = torch.softmax(two_hot_logits, dim=-1)
        return torch.sum(probs * self.bins, dim=-1)




# TODO: Implement symexp for non-uniform binning described in paper
def symexp(x):
    """
    Symmetric exponential function: sign(x) * (exp(abs(x)) - 1).
    Used for non-uniform reward binning.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def symlog(x):
    """
    Symmetric logarithmic function: sign(x) * log(abs(x) + 1).
    Inverse of symexp.
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


class EnvironmentManager:
    def __init__(self, env_name, seed, discrete_action_space, is_visual=False):
        self.env_name = env_name
        self.seed = seed
        self.discrete_action_space = discrete_action_space
        self.is_visual = is_visual
        self.env = create_env(env_name, seed, discrete_action_space, is_visual)
        self.observation_space_shape = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n if discrete_action_space else self.env.action_space.shape[0]

        # For visual environments, stack frames
        if self.is_visual:
            self.frame_stack_queue = deque(maxlen=Config.FRAME_STACK)
            self.observation_space_shape = (Config.FRAME_STACK, *Config.IMAGE_SIZE) # New shape after stacking

    def _get_stacked_observation(self, obs):
        if self.is_visual:
            # Assume obs is an image. Convert to grayscale and resize if needed.
            # For simplicity, we'll just stack the raw observations.
            # In a real implementation, this would involve more sophisticated preprocessing.
            if len(self.frame_stack_queue) == 0:
                # Initialize with current observation repeated
                for _ in range(Config.FRAME_STACK):
                    self.frame_stack_queue.append(obs)
            else:
                self.frame_stack_queue.append(obs)
            return np.concatenate(self.frame_stack_queue, axis=0)
        return obs

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        if self.is_visual:
            self.frame_stack_queue.clear()
            for _ in range(Config.FRAME_STACK):
                self.frame_stack_queue.append(obs)
            obs = np.concatenate(self.frame_stack_queue, axis=0)
        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        if self.is_visual:
            self.frame_stack_queue.append(next_obs)
            next_obs = np.concatenate(self.frame_stack_queue, axis=0)
        return next_obs, reward, done, info

    def close(self):
        self.env.close()

