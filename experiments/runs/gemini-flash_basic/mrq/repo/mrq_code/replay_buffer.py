import torch
import numpy as np
from collections import deque
import random

from mrq_code.config import MRQConfig

class ReplayBuffer:
    def __init__(self, capacity=MRQConfig.REPLAY_BUFFER_CAPACITY, alpha=MRQConfig.LAP_PROBABILITY_SMOOTHING_ALPHA):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.alpha = alpha
        self.priorities = deque(maxlen=capacity)
        self.min_priority = MRQConfig.MINIMUM_PRIORITY # Small epsilon to ensure every item has a chance to be sampled
        self.max_priority = self.min_priority

    def add(self, state, action, reward, next_state, done, error=None):
        # When a new experience is added, it gets max_priority
        if error is None:
            priority = self.max_priority
        else:
            priority = max(error.item(), self.min_priority) # Ensure priority is at least min_priority
        
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
        self.max_priority = max(self.max_priority, priority)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise IndexError("Not enough samples in replay buffer")

        # Calculate sampling probabilities
        scaled_priorities = np.array(self.priorities)**self.alpha
        sample_probs = scaled_priorities / np.sum(scaled_priorities)

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=sample_probs, replace=True)

        # Retrieve samples and their corresponding priorities/weights
        states, actions, rewards, next_states, dones = [], [], [], [], []
        weights = []

        # For importance sampling weights (beta is usually annealed, but here we keep it simple for now)
        # beta = MRQConfig.LAP_BETA # Assuming beta=1 for simplicity, as it's not in the config
        beta = 1.0 # Constant beta for now, can be added to config and annealed later if needed

        max_weight = (len(self.buffer) * self.max_priority)**(-beta) # This is not strictly max weight but a reference
        
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

            # Calculate importance sampling weight
            # weight = (len(self.buffer) * sample_probs[i])**(-beta)
            # weights.append(weight / max_weight) # Normalize weights
            prob_i = sample_probs[i]
            weight_i = (len(self.buffer) * prob_i)**(-beta)
            weights.append(weight_i)
        
        # Normalize weights
        weights = np.array(weights) / np.max(weights)
        
        states = torch.stack(states).float()
        actions = torch.stack(actions).float()
        rewards = torch.stack(rewards).float()
        next_states = torch.stack(next_states).float()
        dones = torch.stack(dones).float()
        weights = torch.tensor(weights).float()

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, errors):
        for i, error in zip(indices, errors):
            priority = max(error.item(), self.min_priority)
            self.priorities[i] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)
