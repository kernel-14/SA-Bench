"""
MR.Q Replay Buffer with LAP (Loss-Adjusted Prioritized sampling)
Based on the paper's description (Fujimoto et al., 2020).
"""

import numpy as np
import torch


class SumTree:
    """SumTree data structure for prioritized experience replay."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class ReplayBuffer:
    """
    Replay buffer with prioritized experience replay (LAP).
    
    Uses the SumTree for efficient O(log N) sampling and updates.
    Priorities are set based on TD errors.
    """
    
    def __init__(self, state_dim, action_dim, capacity=int(1e6), 
                 alpha=0.4, min_priority=1.0, device='cpu'):
        """
        Args:
            state_dim: dimension of state space
            action_dim: dimension of action space (for discrete, use 1)
            capacity: max number of transitions
            alpha: prioritization exponent (0 = uniform, 1 = fully prioritized)
            min_priority: minimum priority for sampling
            device: torch device
        """
        self.capacity = capacity
        self.alpha = alpha
        self.min_priority = min_priority
        self.device = device
        
        self.tree = SumTree(capacity)
        
        # Pre-allocate numpy arrays for faster storage
        self.state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0
        
        # For computing average absolute reward
        self.total_abs_reward = 0.0
        self.num_rewards = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_state_buf[self.ptr] = next_state
        self.done_buf[self.ptr] = float(done)
        
        self.tree.add(self.max_priority ** self.alpha, self.ptr)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Track average absolute reward
        self.total_abs_reward += abs(reward)
        self.num_rewards += 1
    
    @property
    def avg_abs_reward(self):
        """Average absolute reward in buffer (used for reward scaling)."""
        if self.num_rewards == 0:
            return 1.0
        return self.total_abs_reward / self.num_rewards
    
    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions with priorities.
        
        Args:
            batch_size: number of transitions to sample
            beta: importance sampling exponent
        
        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        batch_size = min(batch_size, self.size)
        
        segment = self.tree.total() / batch_size
        
        states = np.zeros((batch_size,) + self.state_buf.shape[1:], dtype=np.float32)
        actions = np.zeros((batch_size,) + self.action_buf.shape[1:], dtype=np.float32)
        rewards = np.zeros(batch_size, dtype=np.float32)
        next_states = np.zeros((batch_size,) + self.next_state_buf.shape[1:], dtype=np.float32)
        dones = np.zeros(batch_size, dtype=np.float32)
        
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data_idx = self.tree.get(s)
            
            indices[i] = idx
            states[i] = self.state_buf[data_idx]
            actions[i] = self.action_buf[data_idx]
            rewards[i] = self.reward_buf[data_idx]
            next_states[i] = self.next_state_buf[data_idx]
            dones[i] = self.done_buf[data_idx]
            
            # Importance sampling weight
            prob = p / self.tree.total()
            weights[i] = (prob * self.size) ** (-beta)
        
        # Normalize weights
        weights = weights / weights.max()
        
        # Convert to tensors
        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).to(self.device),
            indices,
            torch.FloatTensor(weights).to(self.device)
        )
    
    def update_priorities(self, indices, td_errors, min_priority=None):
        """Update priorities based on absolute TD errors."""
        if min_priority is None:
            min_priority = self.min_priority
        
        for idx, td_error in zip(indices, np.abs(td_errors)):
            priority = max(td_error, min_priority)
            self.tree.update(idx, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)
    
    def sample_sequence(self, batch_size, seq_len):
        """
        Sample sequences of transitions for encoder unrolling.
        Returns sequences suitable for the encoder loss.
        
        Since we don't store full episode trajectories, we sample
        random starting points and collect subsequent transitions.
        """
        raise NotImplementedError(
            "For full MR.Q, a sequence-aware replay is needed. "
            "Use the MultiStepReplayBuffer instead."
        )


class MultiStepReplayBuffer(ReplayBuffer):
    """
    Replay buffer that stores episode data for multi-step returns
    and encoder unrolling.
    """
    
    def __init__(self, state_dim, action_dim, capacity=int(1e6),
                 n_step=3, encoder_horizon=5, alpha=0.4, 
                 min_priority=1.0, device='cpu'):
        super().__init__(state_dim, action_dim, capacity, 
                        alpha, min_priority, device)
        self.n_step = n_step
        self.encoder_horizon = encoder_horizon
        
        # Store episode segments for unrolling
        self.episode_buffer = []
        self.current_episode = []
    
    def add(self, state, action, reward, next_state, done):
        """Add transition and store episode data."""
        super().add(state, action, reward, next_state, done)
        self.current_episode.append((state, action, reward, next_state, done))
        if done:
            if len(self.current_episode) >= self.encoder_horizon + 1:
                self.episode_buffer.append(self.current_episode)
            self.current_episode = []
    
    def sample_encoder_batch(self, batch_size):
        """Sample sequences for encoder training with unrolling."""
        if len(self.episode_buffer) == 0:
            return None
        
        batch_size = min(batch_size, len(self.episode_buffer))
        
        # Sample episodes
        episode_indices = np.random.choice(len(self.episode_buffer), 
                                           batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for ep_idx in episode_indices:
            episode = self.episode_buffer[ep_idx]
            # Sample random start position
            max_start = len(episode) - self.encoder_horizon - 1
            if max_start < 0:
                max_start = 0
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            
            seq = episode[start:start + self.encoder_horizon + 1]
            for s, a, r, ns, d in seq:
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(ns)
                dones.append(d)
        
        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.FloatTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(np.array(rewards)).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(np.array(dones)).to(self.device),
        )
