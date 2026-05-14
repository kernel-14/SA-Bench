# replay_buffer.py

import numpy as np
from typing import List, Dict, Any


class ReplayBuffer:
    """
    Implements a replay buffer to store and manage real and synthetic transitions.
    This buffer supports FIFO storage and random sampling of transitions for RL training.
    """

    def __init__(self, size: int) -> None:
        """
        Initialize the replay buffer with a fixed capacity.

        Args:
            size (int): Maximum number of transitions the buffer can hold.
        """
        self.max_size = size
        self.buffer = [None] * size
        self.current_index = 0
        self.current_size = 0

    def store(self, transition: Dict[str, Any]) -> None:
        """
        Stores a transition in the buffer. Overwrites old transitions in FIFO order
        when the buffer’s capacity is exceeded.

        Args:
            transition (Dict[str, Any]): A transition with the following structure:
                {
                    "state": np.ndarray,
                    "action": np.ndarray,
                    "next_state": np.ndarray,
                    "reward": np.ndarray
                }
        """
        if not all(key in transition for key in ["state", "action", "next_state", "reward"]):
            raise ValueError("Transition must contain 'state', 'action', 'next_state', and 'reward' keys.")

        self.buffer[self.current_index] = transition
        self.current_index = (self.current_index + 1) % self.max_size  # Circular overwrite
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Samples a random batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            List[Dict[str, Any]]: A list of sampled transitions.

        Raises:
            ValueError: If there are fewer transitions than requested batch_size.
        """
        if batch_size > self.current_size:
            raise ValueError(f"Insufficient transitions in the buffer: {self.current_size} available, {batch_size} requested.")
        
        indices = np.random.choice(self.current_size, batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def clear(self) -> None:
        """
        Clears all transitions in the buffer. Resets the index and size.
        """
        self.buffer = [None] * self.max_size
        self.current_index = 0
        self.current_size = 0

    @staticmethod
    def mix_batches(real_buffer, synthetic_buffer, ratio: float) -> List[Dict[str, Any]]:
        """
        Mixes real and synthetic transitions based on the specified ratio.

        Args:
            real_buffer (ReplayBuffer): Buffer containing real transitions.
            synthetic_buffer (ReplayBuffer): Buffer containing synthetic transitions.
            ratio (float): Real-to-synthetic ratio. For example:
                - 0.5: 50% real, 50% synthetic transitions will be in the batch.

        Returns:
            List[Dict[str, Any]]: A mixed batch of transitions.

        Raises:
            ValueError: If the buffers do not have enough transitions to fulfill the batch demand.
        """
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("Ratio must be a value between 0.0 and 1.0.")

        # Define batch sizes for real and synthetic transitions
        batch_size = real_buffer.current_size
        real_batch_size = int(batch_size * ratio)
        synthetic_batch_size = batch_size - real_batch_size

        # Sample transitions from both buffers
        real_samples = real_buffer.sample(real_batch_size)
        synthetic_samples = synthetic_buffer.sample(synthetic_batch_size)

        # Combine the samples and shuffle
        mixed_samples = real_samples + synthetic_samples
        np.random.shuffle(mixed_samples)

        return mixed_samples
