import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(StateEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class StateActionEncoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, output_dim: int):
        super(StateActionEncoder, self).__init__()
        self.action_fc = nn.Linear(action_dim, 256)
        self.fc1 = nn.Linear(state_dim + 256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, output_dim)
        self.activation = nn.ELU()

    def forward(self, state, action):
        action_embedding = self.activation(self.action_fc(action))
        x = torch.cat([state, action_embedding], dim=-1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x