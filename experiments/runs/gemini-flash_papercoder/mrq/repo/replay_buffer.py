from dataclasses import dataclass
import numpy as np
import torch
from typing import Tuple, List, Dict
import math

# Assuming Config is in the same directory or accessible via Python path
from config import Config


@dataclass(frozen=True)
class Experience:
    """A single experience transition stored in the replay buffer.

    Attributes:
        obs: The observation at the current timestep t.
        action: The action taken at timestep t.
        reward: The reward received after taking action in obs.
        next_obs: The observation at timestep t+1.
        done: A boolean indicating if the episode terminated at next_obs.
        priority: The priority of this transition for sampling.
    """
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool
    priority: float = 1.0


class SumTree:
    """A segment tree data structure for efficient priority-based sampling.

    It stores priorities at leaf nodes and their sums in parent nodes,
    allowing for O(log N) updates and proportional sampling.
    """

    def __init__(self, capacity: int):
        """Initializes the SumTree with a given capacity.

        Args:
            capacity: The maximum number of leaf nodes (experiences) to store.
        """
        self.capacity = capacity
        # Smallest power of 2 greater than or equal to capacity
        self.tree_base_size = self._get_next_power_of_2(capacity)
        # Total tree size: 2 * (smallest power of 2 >= capacity) - 1
        self.tree_size = self.tree_base_size * 2 - 1
        # Initialize tree nodes to zero
        self.tree = np.zeros(self.tree_size, dtype=np.float32)
        # The offset to access leaf nodes directly from data indices
        self.tree_offset = self.tree_base_size - 1

    def _get_next_power_of_2(self, n: int) -> int:
        """Returns the smallest power of 2 greater than or equal to n."""
        # For n=0, this results in 1, which is correct for 0 capacity (smallest power of 2 is 1)
        return 1 if n == 0 else 2**(n - 1).bit_length()

    def update(self, idx: int, priority: float):
        """Updates the priority of a leaf node and propagates the change up the tree.

        Args:
            idx: The original data index (0 to capacity-1).
            priority: The new priority value for this index.
        """
        tree_idx = idx + self.tree_offset
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value: float) -> Tuple[int, float]:
        """Retrieves a leaf node (data index and its priority) given a random value.

        Args:
            value: A random float in the range [0, total_sum).

        Returns:
            A tuple: (data_idx, priority_value)
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # If we reached a non-leaf node, but its children are outside the tree_size,
            # this implies an issue with tree structure or value (should not happen with valid input).
            if left_child_idx >= self.tree_size:
                data_idx = parent_idx - self.tree_offset
                return data_idx, self.tree[parent_idx]

            # Traverse down based on value
            if value <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                value -= self.tree[left_child_idx]
                parent_idx = right_child_idx

    @property
    def total_sum(self) -> float:
        """Returns the sum of all priorities (stored at the root of the tree)."""
        return self.tree[0]


class MinTree(SumTree):
    """A segment tree optimized for finding the minimum priority."""

    def update(self, idx: int, priority: float):
        """Updates the priority of a leaf node and propagates the change up the tree.
        Here, parent nodes store the minimum of their children.
        """
        tree_idx = idx + self.tree_offset
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            parent_idx = (tree_idx - 1) // 2
            # Handle potential padding nodes that might have 0 value if not explicitly updated
            # For this MinTree, we should take the minimum of its actual children.
            left_val = self.tree[2 * parent_idx + 1]
            right_val = self.tree[2 * parent_idx + 2] if (2 * parent_idx + 2) < self.tree_size else float('inf')
            self.tree[parent_idx] = min(left_val, right_val)
            tree_idx = parent_idx

    @property
    def min_val(self) -> float:
        """Returns the minimum priority in the tree (stored at the root)."""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer with multi-step sampling.

    Implements a buffer that stores single transitions and samples multi-step
    sequences with prioritization based on TD errors, and computes Importance
    Sampling (IS) weights. It uses a circular buffer for storage.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        action_dim: int,
        device: torch.device,
        alpha: float,
        min_priority_initial: float = 1.0,
        beta_initial: float = 1.0,
        td_error_epsilon: float = 1e-6,
    ):
        """Initializes the PrioritizedReplayBuffer.

        Args:
            capacity: The maximum number of transitions the buffer can store.
            obs_shape: A tuple representing the shape of the environment's observations.
            action_dim: An integer representing the dimension of the action space.
            device: The PyTorch device (e.g., 'cuda', 'cpu') to move sampled tensors to.
            alpha: The exponent parameter for PER, determining how much prioritization
                   influences sampling probability.
            min_priority_initial: The initial priority assigned to new experiences, and
                                  also the floor for all updated priorities. (e.g., 1.0 for LAP).
            beta_initial: The initial exponent for importance sampling weights calculation.
            td_error_epsilon: Small value added to absolute TD errors before exponentiation
                              to ensure non-zero priority and numerical stability.
        """
        self._capacity = capacity
        self._obs_shape = obs_shape
        self._action_dim = action_dim
        self._device = device
        self._alpha = alpha
        self._beta = beta_initial  # Beta for importance sampling, typically annealed.
        self._td_error_epsilon = td_error_epsilon

        self._storage_idx = 0  # Pointer to the next available slot for adding experience
        self._current_size = 0  # Current number of experiences in the buffer

        # max_priority is tracked to assign to new experiences, ensuring they are sampled at least once.
        # It's initialized to min_priority_initial as per LAP.
        self._max_priority = min_priority_initial
        # min_priority_floor is the absolute minimum any priority can be set to.
        self._min_priority_floor = min_priority_initial

        # Data storage using NumPy arrays for efficient memory usage and fast indexing
        self._obs_buffer = np.empty((capacity, *obs_shape), dtype=np.float32)
        self._action_buffer = np.empty((capacity, action_dim), dtype=np.float32)
        self._reward_buffer = np.empty(capacity, dtype=np.float32)
        self._next_obs_buffer = np.empty((capacity, *obs_shape), dtype=np.float32)
        self._done_buffer = np.empty(capacity, dtype=np.bool_)

        # Priority trees for sampling and IS weight calculation
        self._sum_tree = SumTree(capacity)
        self._min_tree = MinTree(capacity)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        """Adds a new single-step transition to the replay buffer.

        If the buffer is full, the oldest experience is overwritten.
        New experiences are assigned the current maximum priority to ensure they
        are sampled at least once.

        Args:
            obs: The observation at the current timestep.
            action: The action taken.
            reward: The reward received.
            next_obs: The observation at the next timestep.
            done: A boolean indicating if the episode terminated.
        """
        # Store data in the circular buffers at _storage_idx
        self._obs_buffer[self._storage_idx] = obs
        self._action_buffer[self._storage_idx] = action
        self._reward_buffer[self._storage_idx] = reward
        self._next_obs_buffer[self._storage_idx] = next_obs
        self._done_buffer[self._storage_idx] = done

        # Assign initial priority to the new experience.
        # This is `_max_priority` which means new samples are prioritized.
        # It is also floored by `_min_priority_initial` so that `_max_priority` is at least this.
        initial_priority = self._max_priority
        self._sum_tree.update(self._storage_idx, initial_priority)
        self._min_tree.update(self._storage_idx, initial_priority)

        # Update the overall maximum priority in the buffer
        self._max_priority = max(self._max_priority, initial_priority)

        # Move to the next storage index, wrapping around if capacity is reached
        self._storage_idx = (self._storage_idx + 1) % self._capacity
        # Increment current_size, capping at capacity
        self._current_size = min(self._current_size + 1, self._capacity)

    def sample(self, batch_size: int, k_step: int) -> Dict[str, torch.Tensor]:
        """Samples a batch of multi-step transitions from the buffer.

        Multi-step sequences are constructed starting from sampled indices.
        Handles episode termination within a sequence (subsequent rewards become 0).
        Calculates Importance Sampling (IS) weights.

        Args:
            batch_size: The number of sequences to sample.
            k_step: The length of the multi-step sequences to construct.

        Returns:
            A dictionary containing batched and processed tensors:
            - s_0: Initial observations (batch_size, *obs_shape).
            - actions_seq: Sequence of actions (batch_size, k_step, action_dim).
            - rewards_seq: Sequence of rewards (batch_size, k_step).
            - dones_seq: Sequence of done flags (batch_size, k_step).
            - final_obs_k: Observation at the end of the k_step sequence
                           (or terminal obs if episode ended) (batch_size, *obs_shape).
            - initial_indices: Original data indices in the buffer for sampled s_0 (np.ndarray).
            - is_weights: Importance sampling weights (batch_size,).
        """
        # We need at least k_step + 1 transitions to form a complete k_step sequence
        # (s_0, ..., s_k).
        if self._current_size < k_step + 1:
            raise ValueError(f"Not enough samples in buffer for k_step={k_step}. "
                             f"Current size: {self._current_size}. Required: {k_step + 1}.")

        sampled_indices: List[int] = []
        priorities: List[float] = []
        is_weights_list: List[float] = []

        total_priority_sum = self._sum_tree.total_sum
        min_priority_in_buffer = self._min_tree.min_val # This is self._min_priority_floor after init

        if total_priority_sum == 0: # Avoid division by zero if all priorities are 0 (e.g. empty buffer initially if not LAP style)
             raise ValueError("Total priority sum is zero, cannot sample.")

        for _ in range(batch_size):
            # Sample a random value proportional to priorities
            rand_val = np.random.uniform(0, total_priority_sum)
            data_idx, priority = self._sum_tree.get_leaf(rand_val)
            
            # Re-sample if the chosen `data_idx` leads to an incomplete sequence
            # An incomplete sequence would be one that extends beyond the _current_size
            # or into the next _storage_idx if the buffer is not full.
            # However, with _sum_tree initialized to 0 and only filled spots getting priorities,
            # and by enforcing _current_size for validity, we can simplify.
            # To ensure k-step sequence doesn't read future unfilled spots:
            # The start_idx `data_idx` must be such that `data_idx + k_step - 1`
            # is still within the `_current_size` (when buffer not yet full).
            # If buffer is full, `data_idx + k_step - 1` can wrap around.
            # A valid index for multi-step sampling is one where the sequence `(data_idx, ..., data_idx + k_step - 1)`
            # does not cross the `_storage_idx` (next write position) unless the buffer is full.
            # Simpler check: ensure the last element of the sequence is "before" _storage_idx
            # (conceptually for a continuous sequence)
            is_valid_sequence = False
            num_attempts = 0 # To prevent infinite loop for very small buffers
            while not is_valid_sequence and num_attempts < self._capacity:
                # Check for buffer not full:
                if self._current_size < self._capacity:
                    # If not full, indices past _current_size are empty.
                    # Sequence must not go past _current_size - 1.
                    if data_idx + k_step <= self._current_size:
                        is_valid_sequence = True
                else: # Buffer is full, all indices are valid, sequence can wrap around
                    is_valid_sequence = True
                
                if not is_valid_sequence:
                    rand_val = np.random.uniform(0, total_priority_sum)
                    data_idx, priority = self._sum_tree.get_leaf(rand_val)
                    num_attempts += 1
                    if num_attempts == self._capacity: # Should almost never happen if buffer has enough samples
                        raise RuntimeError("Failed to sample a valid k-step sequence after many attempts. "
                                           "Buffer might be too small or sampling logic flawed.")

            sampled_indices.append(data_idx)
            priorities.append(priority)

            # Calculate IS weight: (1 / (N * P_j))**beta
            # P_j = priority / total_priority_sum
            # IS weight = ( (self._current_size * P_j) )**(-self._beta)
            # The paper does not specify beta annealing, so beta_initial = 1.0 is used.
            P_j = priority / total_priority_sum
            is_weight = (P_j * self._current_size)**(-self._beta)
            is_weights_list.append(is_weight)

        # Normalize IS weights by dividing by max weight for stability (common practice in PER)
        # max_is_weight ensures no single weight becomes excessively large.
        max_is_weight = max(is_weights_list) if is_weights_list else 1.0
        is_weights_list = [w / max_is_weight for w in is_weights_list]
        
        # Prepare lists for collected multi-step data
        batch_s0: List[np.ndarray] = []
        batch_actions_seq: List[np.ndarray] = []
        batch_rewards_seq: List[np.ndarray] = []
        batch_dones_seq: List[np.ndarray] = []
        batch_final_obs_k: List[np.ndarray] = []

        for data_idx in sampled_indices:
            s0 = self._obs_buffer[data_idx]
            
            actions_k: List[np.ndarray] = []
            rewards_k: List[float] = []
            dones_k: List[bool] = []
            
            current_sequence_end_obs = None
            has_terminated_in_sequence = False
            
            # Loop to construct the k_step sequence
            for j in range(k_step):
                buffer_idx_t = (data_idx + j) % self._capacity
                
                if has_terminated_in_sequence:
                    # If episode already terminated, subsequent actions/rewards are 0, done is True
                    actions_k.append(np.zeros(self._action_dim, dtype=self._action_buffer.dtype))
                    rewards_k.append(0.0)
                    dones_k.append(True)
                else:
                    actions_k.append(self._action_buffer[buffer_idx_t])
                    rewards_k.append(self._reward_buffer[buffer_idx_t])
                    dones_k.append(self._done_buffer[buffer_idx_t])

                    if self._done_buffer[buffer_idx_t]:
                        # Episode terminates at buffer_idx_t.
                        # The next_obs for this transition becomes the sequence's final_obs_k.
                        current_sequence_end_obs = self._next_obs_buffer[buffer_idx_t]
                        has_terminated_in_sequence = True
            
            # Determine the final observation for the k_step sequence
            if not has_terminated_in_sequence:
                # If the episode didn't terminate, final_obs_k is the next_obs of the last step in the sequence
                current_sequence_end_obs = self._next_obs_buffer[(data_idx + k_step - 1) % self._capacity]

            # Append the constructed sequence elements to batch lists
            batch_s0.append(s0)
            batch_actions_seq.append(np.stack(actions_k, axis=0)) # (k_step, action_dim)
            batch_rewards_seq.append(np.stack(rewards_k, axis=0)) # (k_step,)
            batch_dones_seq.append(np.stack(dones_k, axis=0))     # (k_step,)
            batch_final_obs_k.append(current_sequence_end_obs)

        # Convert all NumPy arrays to PyTorch tensors and move them to the specified device
        s_0 = torch.tensor(np.stack(batch_s0, axis=0), dtype=torch.float32, device=self._device)
        actions_seq = torch.tensor(np.stack(batch_actions_seq, axis=0), dtype=torch.float32, device=self._device)
        rewards_seq = torch.tensor(np.stack(batch_rewards_seq, axis=0), dtype=torch.float32, device=self._device)
        dones_seq = torch.tensor(np.stack(batch_dones_seq, axis=0), dtype=torch.bool, device=self._device)
        final_obs_k = torch.tensor(np.stack(batch_final_obs_k, axis=0), dtype=torch.float32, device=self._device)
        
        # initial_indices remains a NumPy array as it's typically used for CPU-side indexing
        initial_indices = np.array(sampled_indices, dtype=np.int64)
        is_weights = torch.tensor(np.array(is_weights_list), dtype=torch.float32, device=self._device)

        return {
            "s_0": s_0,
            "actions_seq": actions_seq,  # (batch_size, k_step, action_dim)
            "rewards_seq": rewards_seq,  # (batch_size, k_step)
            "dones_seq": dones_seq,      # (batch_size, k_step)
            "final_obs_k": final_obs_k,  # (batch_size, *obs_shape)
            "initial_indices": initial_indices,
            "is_weights": is_weights,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Updates the priorities of previously sampled transitions based on new TD errors.

        Priorities are updated for the given indices using the absolute TD errors,
        exponentiated by alpha, and floored by `_min_priority_floor`.

        Args:
            indices: A NumPy array containing the data_idx of the transitions whose
                     priorities need to be updated.
            td_errors: A NumPy array of the absolute TD errors for these transitions.
        """
        # Calculate new priorities: (abs(TD_error) + epsilon)**alpha
        td_errors_processed = (np.abs(td_errors) + self._td_error_epsilon)**self._alpha
        
        # Ensure new priorities are at least _min_priority_floor
        new_priorities = np.maximum(td_errors_processed, self._min_priority_floor)

        for idx, new_p in zip(indices, new_priorities):
            self._sum_tree.update(idx, new_p)
            self._min_tree.update(idx, new_p)
            # Keep track of the highest priority observed for new experiences
            self._max_priority = max(self._max_priority, new_p)

    def size(self) -> int:
        """Returns the current number of transitions stored in the buffer."""
        return self._current_size

