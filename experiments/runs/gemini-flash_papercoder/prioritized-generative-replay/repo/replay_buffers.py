import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union

# Assuming Config class is available from config.py
from config import Config

class SumTree:
    """
    A binary tree data structure to efficiently store priorities and enable
    sampling proportional to priority and quick updates.
    """
    def __init__(self, capacity: int):
        """
        Initializes the SumTree.

        Args:
            capacity (int): The maximum number of leaf nodes (experiences).
        """
        self.capacity: int = capacity
        # The tree array stores sum of priorities. Tree size is 2*capacity - 1 for a complete binary tree.
        # The leaves (priorities of experiences) are at indices capacity-1 to 2*capacity-2.
        self.tree: np.ndarray = np.zeros(2 * capacity - 1, dtype=np.float32)
        # The data array stores the actual data index in the ReplayBuffer.
        self.data: np.ndarray = np.zeros(capacity, dtype=np.int64) # Store ReplayBuffer indices
        self.data_idx: int = 0  # Pointer to the next data entry to be written

    def _propagate(self, idx: int, change: float):
        """
        Propagate changes up the tree from leaf `idx`.

        Args:
            idx (int): Current node index in the tree.
            change (float): The amount of change to propagate.
        """
        parent_idx: int = (idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate(parent_idx, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """
        Recursively find the leaf node corresponding to a sampled value `s`.

        Args:
            idx (int): Current node index in the tree.
            s (float): Sampled value from [0, total_priority).

        Returns:
            int: The index of the leaf node within the `self.tree` array.
        """
        left_child_idx: int = 2 * idx + 1
        # right_child_idx: int = 2 * idx + 2 # Not strictly needed if left_child_idx condition is enough

        if left_child_idx >= len(self.tree): # Is a leaf node
            return idx

        if s <= self.tree[left_child_idx]:
            return self._retrieve(left_child_idx, s)
        else:
            return self._retrieve(left_child_idx + 1, s - self.tree[left_child_idx])

    @property
    def total_priority(self) -> float:
        """Returns the sum of all priorities (root of the tree)."""
        return self.tree[0]

    def add(self, priority: float, data_index: int):
        """
        Adds a new priority and associates it with a data_index.
        When a new experience is added to the ReplayBuffer, this method is called
        with its priority and the index where it was stored in the ReplayBuffer.

        Args:
            priority (float): The priority of the experience.
            data_index (int): The index in the ReplayBuffer where the experience is stored.
        """
        # Leaf node index for the new data within the `self.tree` array
        leaf_idx: int = self.data_idx + self.capacity - 1
        
        # Store the actual data index in our data array
        self.data[self.data_idx] = data_index
        
        # Calculate the change in priority if we are overwriting an existing leaf
        change: float = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        self._propagate(leaf_idx, change)

        self.data_idx = (self.data_idx + 1) % self.capacity

    def sample(self, s: float) -> Tuple[int, float, int]:
        """
        Samples an experience index based on a sampled value `s` from [0, total_priority).

        Args:
            s (float): A random value from [0, total_priority).

        Returns:
            Tuple[int, float, int]: (leaf_index_in_tree, priority_of_leaf, replay_buffer_index)
        """
        leaf_idx: int = self._retrieve(0, s)
        data_idx: int = leaf_idx - (self.capacity - 1) # Convert tree leaf index to data array index
        
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def update(self, leaf_idx: int, new_priority: float):
        """
        Updates the priority of a leaf node and propagates the change.

        Args:
            leaf_idx (int): The index of the leaf node in the `self.tree` array.
            new_priority (float): The new priority value.
        """
        change: float = new_priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = new_priority
        self._propagate(leaf_idx, change)


class ReplayBuffer:
    """
    A standard FIFO replay buffer for storing transitions.
    """
    def __init__(self, capacity: int, state_dim: Tuple[int, ...], action_dim: int,
                 pixel_based: bool, device: torch.device):
        """
        Initializes the ReplayBuffer.

        Args:
            capacity (int): The maximum number of transitions to store.
            state_dim (Tuple[int, ...]): The shape of a single state observation.
            action_dim (int): The dimension of the action space (e.g., 1 for discrete, N for continuous).
            pixel_based (bool): True if observations are pixel-based images (uint8), False for state vectors (float32).
            device (torch.device): The device (CPU/GPU) to move sampled tensors to.
        """
        self.capacity: int = capacity
        self.state_dim: Tuple[int, ...] = state_dim
        self.action_dim: int = action_dim
        self.pixel_based: bool = pixel_based
        self.device: torch.device = device

        # Determine storage dtype for states
        state_dtype: np.dtype = np.uint8 if pixel_based else np.float32

        # Pre-allocate NumPy arrays for efficiency
        self._states: np.ndarray = np.empty((capacity, *state_dim), dtype=state_dtype)
        self._actions: np.ndarray = np.empty((capacity, action_dim), dtype=np.float32)
        self._rewards: np.ndarray = np.empty(capacity, dtype=np.float32)
        self._next_states: np.ndarray = np.empty((capacity, *state_dim), dtype=state_dtype)
        self._dones: np.ndarray = np.empty(capacity, dtype=np.bool_)

        self._idx: int = 0  # Current insertion index
        self._size: int = 0 # Current number of elements in the buffer

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """
        Adds a single transition to the replay buffer.

        Args:
            state (np.ndarray): The current state observation.
            action (np.ndarray): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state observation.
            done (bool): Whether the episode terminated.
        """
        # Ensure correct data types before storing
        self._states[self._idx] = state.astype(self._states.dtype, copy=False)
        self._actions[self._idx] = action.astype(np.float32, copy=False)
        self._rewards[self._idx] = np.float32(reward)
        self._next_states[self._idx] = next_state.astype(self._next_states.dtype, copy=False)
        self._dones[self._idx] = np.bool_(done)

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing PyTorch tensors for
                                     'state', 'action', 'reward', 'next_state', 'done'.
        """
        if self._size == 0:
            raise IndexError("Replay buffer is empty, cannot sample.")

        indices: np.ndarray = np.random.choice(self._size, size=batch_size, replace=True)

        states: np.ndarray = self._states[indices]
        actions: np.ndarray = self._actions[indices]
        rewards: np.ndarray = self._rewards[indices]
        next_states: np.ndarray = self._next_states[indices]
        dones: np.ndarray = self._dones[indices]

        # Convert to torch tensors and move to device
        states_tensor: torch.Tensor = torch.as_tensor(states, device=self.device)
        actions_tensor: torch.Tensor = torch.as_tensor(actions, device=self.device)
        rewards_tensor: torch.Tensor = torch.as_tensor(rewards, device=self.device).unsqueeze(-1) # Ensure (batch_size, 1)
        next_states_tensor: torch.Tensor = torch.as_tensor(next_states, device=self.device)
        dones_tensor: torch.Tensor = torch.as_tensor(dones, device=self.device).unsqueeze(-1) # Ensure (batch_size, 1)

        # Pixel-specific preprocessing: normalize if pixel_based
        if self.pixel_based:
            states_tensor = states_tensor.float() / 255.0
            next_states_tensor = next_states_tensor.float() / 255.0

        return {
            'state': states_tensor,
            'action': actions_tensor,
            'reward': rewards_tensor,
            'next_state': next_states_tensor,
            'done': dones_tensor
        }

    def size(self) -> int:
        """Returns the current number of transitions in the buffer."""
        return self._size
    
    def get_all_transitions(self) -> Dict[str, np.ndarray]:
        """
        Returns all current transitions in the buffer as numpy arrays.
        Useful for operations like training a generative model on the entire D_real.
        """
        if self._size == 0:
            return {
                'state': np.empty((0, *self.state_dim), dtype=self._states.dtype),
                'action': np.empty((0, self.action_dim), dtype=np.float32),
                'reward': np.empty(0, dtype=np.float32),
                'next_state': np.empty((0, *self.state_dim), dtype=self._next_states.dtype),
                'done': np.empty(0, dtype=np.bool_)
            }
        
        return {
            'state': self._states[:self._size],
            'action': self._actions[:self._size],
            'reward': self._rewards[:self._size],
            'next_state': self._next_states[:self._size],
            'done': self._dones[:self._size]
        }


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Implements a Prioritized Experience Replay (PER) buffer using a SumTree.
    Inherits from ReplayBuffer to store the actual experience data.
    """
    def __init__(self, config: Config, capacity: int, state_dim: Tuple[int, ...], action_dim: int,
                 pixel_based: bool, device: torch.device):
        """
        Initializes the PrioritizedReplayBuffer.

        Args:
            config (Config): Configuration object to retrieve PER hyperparameters.
            capacity (int): The maximum number of transitions to store.
            state_dim (Tuple[int, ...]): The shape of a single state observation.
            action_dim (int): The dimension of the action space.
            pixel_based (bool): True if observations are pixel-based.
            device (torch.device): The device (CPU/GPU) for tensors.
        """
        super().__init__(capacity, state_dim, action_dim, pixel_based, device)

        self._alpha: float = config.get_hyperparam('replay_buffers.per_alpha')
        self._beta_start: float = config.get_hyperparam('replay_buffers.per_beta_start')
        self._beta_frames: int = config.get_hyperparam('replay_buffers.per_beta_frames')
        
        self._beta: float = self._beta_start # Current importance sampling exponent
        self._max_priority: float = 1.0  # Initial max priority for new experiences

        self._tree: SumTree = SumTree(capacity)
        self._epsilon: float = 1e-6 # Small epsilon to prevent zero priority

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool, priority: Optional[float] = None):
        """
        Adds a single transition to the prioritized replay buffer with a given priority.

        Args:
            state (np.ndarray): The current state observation.
            action (np.ndarray): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state observation.
            done (bool): Whether the episode terminated.
            priority (Optional[float]): The priority of the transition. If None, uses max_priority.
        """
        # Add experience to the base ReplayBuffer's storage
        # This will update self._idx and self._size
        super().add(state, action, reward, next_state, done)

        # Determine the priority for the new experience
        if priority is None:
            new_priority: float = self._max_priority
        else:
            new_priority = max(priority, self._epsilon) # Ensure priority is never zero

        # Add the priority to the SumTree, associating it with the index where the experience was stored
        # The `super().add` method places the experience at `(self._idx - 1 + self.capacity) % self.capacity`.
        # This is the `replay_buffer_index` for the SumTree.
        experience_rb_idx: int = (self._idx - 1 + self.capacity) % self.capacity
        self._tree.add(new_priority ** self._alpha, experience_rb_idx)


    def sample(self, batch_size: int, current_step: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Samples a batch of transitions from the buffer, prioritized by their TD-error,
        and provides importance sampling weights.

        Args:
            batch_size (int): The number of transitions to sample.
            current_step (int): The current training step for beta annealing.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
                - A dictionary containing PyTorch tensors for 'state', 'action', 'reward', 'next_state', 'done'.
                - A PyTorch tensor of indices (in the SumTree's leaf array) for updating priorities.
                - A PyTorch tensor of importance sampling weights.
        """
        if self._size == 0:
            raise IndexError("Replay buffer is empty, cannot sample.")

        # Anneal beta
        self._beta = min(1.0, self._beta_start + current_step * (1.0 - self._beta_start) / self._beta_frames)

        batch_indices_in_tree: np.ndarray = np.empty(batch_size, dtype=np.int64)
        batch_priorities: np.ndarray = np.empty(batch_size, dtype=np.float32)
        batch_rb_indices: np.ndarray = np.empty(batch_size, dtype=np.int64)

        # Sample values uniformly from [0, total_priority) for each sample
        segment_length: float = self._tree.total_priority / batch_size
        for i in range(batch_size):
            a: float = segment_length * i
            b: float = segment_length * (i + 1)
            s: float = np.random.uniform(a, b)
            
            tree_idx, priority, rb_idx = self._tree.sample(s)
            
            batch_indices_in_tree[i] = tree_idx
            batch_priorities[i] = priority
            batch_rb_indices[i] = rb_idx

        # Retrieve experiences from the base ReplayBuffer using the sampled indices
        states: np.ndarray = self._states[batch_rb_indices]
        actions: np.ndarray = self._actions[batch_rb_indices]
        rewards: np.ndarray = self._rewards[batch_rb_indices]
        next_states: np.ndarray = self._next_states[batch_rb_indices]
        dones: np.ndarray = self._dones[batch_rb_indices]

        # Calculate importance sampling (IS) weights
        # min_prob represents the probability of the most "unimportant" experience (lowest priority).
        # It's used for normalization to scale IS weights correctly.
        min_prob_val: float = np.min(batch_priorities) if len(batch_priorities) > 0 else 1.0
        if self._tree.total_priority == 0: # Avoid division by zero if tree is empty or all priorities are zero
            min_prob_norm: float = 1.0 
        else:
            min_prob_norm = min_prob_val / self._tree.total_priority

        # Calculate max_weight based on the lowest probability encountered in the batch
        # This normalization ensures that IS weights are never greater than 1,
        # preventing large updates from being scaled up excessively.
        max_weight: float = (self._size * min_prob_norm) ** (-self._beta) if min_prob_norm > 0 else 1.0
        
        probs: np.ndarray = batch_priorities / self._tree.total_priority
        is_weights: np.ndarray = (self._size * probs) ** (-self._beta)
        is_weights = is_weights / max_weight # Normalize for stability

        # Convert to torch tensors and move to device
        states_tensor: torch.Tensor = torch.as_tensor(states, device=self.device)
        actions_tensor: torch.Tensor = torch.as_tensor(actions, device=self.device)
        rewards_tensor: torch.Tensor = torch.as_tensor(rewards, device=self.device).unsqueeze(-1)
        next_states_tensor: torch.Tensor = torch.as_tensor(next_states, device=self.device)
        dones_tensor: torch.Tensor = torch.as_tensor(dones, device=self.device).unsqueeze(-1)
        is_weights_tensor: torch.Tensor = torch.as_tensor(is_weights, device=self.device).unsqueeze(-1)

        # Pixel-specific preprocessing: normalize if pixel_based
        if self.pixel_based:
            states_tensor = states_tensor.float() / 255.0
            next_states_tensor = next_states_tensor.float() / 255.0

        experience_batch: Dict[str, torch.Tensor] = {
            'state': states_tensor,
            'action': actions_tensor,
            'reward': rewards_tensor,
            'next_state': next_states_tensor,
            'done': dones_tensor
        }

        return experience_batch, torch.as_tensor(batch_indices_in_tree, device=self.device), is_weights_tensor

    def update_priorities(self, tree_indices: torch.Tensor, new_priorities: torch.Tensor):
        """
        Updates the priorities of sampled transitions in the SumTree.

        Args:
            tree_indices (torch.Tensor): A tensor of leaf indices in the SumTree where priorities need updating.
            new_priorities (torch.Tensor): A tensor of new priority values (e.g., absolute TD-errors).
        """
        # Ensure priorities are positive and apply alpha exponent
        new_priorities_np: np.ndarray = new_priorities.detach().cpu().numpy()
        new_priorities_np = (np.abs(new_priorities_np) + self._epsilon) ** self._alpha

        # Update _max_priority
        self._max_priority = max(self._max_priority, np.max(new_priorities_np))

        # Update the SumTree
        for tree_idx, prio in zip(tree_indices.cpu().numpy(), new_priorities_np):
            self._tree.update(int(tree_idx), float(prio))

