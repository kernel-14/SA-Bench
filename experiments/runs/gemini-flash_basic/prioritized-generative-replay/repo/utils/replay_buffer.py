import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = {
            'states': torch.empty(capacity, obs_dim, dtype=torch.float32),
            'actions': torch.empty(capacity, action_dim, dtype=torch.float32),
            'next_states': torch.empty(capacity, obs_dim, dtype=torch.float32),
            'rewards': torch.empty(capacity, 1, dtype=torch.float32),
            'dones': torch.empty(capacity, 1, dtype=torch.float32),
        }
        self.position = 0
        self.size = 0

    def add(self, state, action, next_state, reward, done):
        self.buffer['states'][self.position] = torch.as_tensor(state, dtype=torch.float32)
        self.buffer['actions'][self.position] = torch.as_tensor(action, dtype=torch.float32)
        self.buffer['next_states'][self.position] = torch.as_tensor(next_state, dtype=torch.float32)
        self.buffer['rewards'][self.position] = torch.as_tensor(reward, dtype=torch.float32)
        self.buffer['dones'][self.position] = torch.as_tensor(done, dtype=torch.float32)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        batch = {
            key: self.buffer[key][indices] for key in self.buffer
        }
        return batch

    def __len__(self):
        return self.size
