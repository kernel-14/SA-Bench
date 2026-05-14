import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class FeatureEncoder(nn.Module):
    def __init__(self, obs_dim, latent_dim=512):
        super().__init__()
        self.encoder = MLP(obs_dim, latent_dim)

    def forward(self, obs):
        return self.encoder(obs)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = MLP(obs_dim + action_dim, 1, hidden_dim=hidden_dim)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # For simplicity, assume a deterministic policy returning actions directly
        self.net = MLP(obs_dim, action_dim, hidden_dim=hidden_dim)

    def forward(self, obs):
        return self.net(obs)

class ForwardDynamicsModel(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.model = MLP(latent_dim + action_dim, latent_dim, hidden_dim=hidden_dim)

    def forward(self, latent_state, action):
        return self.model(torch.cat([latent_state, action], dim=-1))
