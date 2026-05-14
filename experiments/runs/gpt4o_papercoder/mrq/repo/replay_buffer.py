"""
replay_buffer.py
Manages experience replay, including storage, sampling with prioritization, 
and priority updates to support training for the MR.Q algorithm.
"""

import numpy as np
from typing import Dict, List, Any, Tuple


class ReplayBuffer:
    """
    Replay buffer for storing and providing batches of transitions for prioritized experience replay.
    This implementation supports both uniform and prioritized sampling.

    Attributes:
        capacity (int): Maximum buffer size (number of transitions).
    """

    def __init__(self, capacity: int = 1_000_000):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.buffer = [None] * capacity  # Circular buffer storage for transitions
        self.priorities = np.zeros(capacity, dtype=np.float32)  # Priority values for each transition
        self.write_index = 0  # Index to overwrite old transitions
        self.size = 0  # Current number of stored transitions
        self.epsilon = 1e-6  # Small constant to prevent zero probabilities
        self.alpha = 0.6  # Prioritization strength, configurable via `config.yaml`

    def add_sample(self, sample: Dict[str, Any]) -> None:
        """
        Add a new transition to the replay buffer.
        
        Args:
            sample (Dict[str, Any]): A transition sample containing:
                - `state`: The current state.
                - `action`: The action taken.
                - `reward`: The reward received.
                - `next_state`: The next state observed.
                - `terminal`: Whether the episode terminated.
        """
        # Add the sample to the buffer
        self.buffer[self.write_index] = sample

        # Assign the new sample a priority equal to the maximum priority in the buffer
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.write_index] = max_priority

        # Update write_index cyclically and buffer size
        self.write_index = (self.write_index + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample_batch(self, batch_size: int = 256) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions using proportional prioritization.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
                - Batch of sampled transitions.
                - Sampled indices corresponding to the transitions.
                - Importance sampling weights for the batch.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")

        if batch_size > self.size:
            raise ValueError("Batch size cannot exceed the current buffer size.")

        # Compute probabilities for proportional sampling
        scaled_priorities = self.priorities[:self.size] ** self.alpha
        sampling_probabilities = scaled_priorities / np.sum(scaled_priorities)

        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, replace=False, p=sampling_probabilities)

        # Retrieve the transitions
        transitions = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling (IS) weights for the batch
        is_weights = (1.0 / (self.size * sampling_probabilities[indices]))
        is_weights /= is_weights.max()  # Normalize the weights to range [0, 1]

        return transitions, indices, is_weights

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for specific transitions.

        Args:
            indices (List[int]): Indices of transitions to update.
            priorities (List[float]): Corresponding new priorities for the transitions.
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = max(self.epsilon, priority)  # Clip to lower bound

    def get_all(self) -> List[Dict[str, Any]]:
        """
        Fetch all stored transitions for debugging and visualization.
        
        Returns:
            List[Dict[str, Any]]: All transitions in the buffer.
        """
        return self.buffer[:self.size]

