"""
FRE Decoder network.

Predicts reward r = eta(s) given state s and latent z.
Architecture: MLP with layers [512, 512, 512], raw state concatenated with z.
No embedding step for the observation state (per addendum).
"""

import torch
import torch.nn as nn


class FREDecoder(nn.Module):
    """
    Feedforward decoder that predicts scalar reward given (state, z).

    The raw state and z-vector are concatenated directly (no embedding step).
    Network: [state_dim + latent_dim] -> 512 -> 512 -> 512 -> 1
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        hidden_dims: list = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512]

        in_dim = state_dim + latent_dim
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (batch, state_dim) or (batch, K', state_dim)
            z:      (batch, latent_dim) or (batch, 1, latent_dim) broadcast
        Returns:
            rewards: (batch,) or (batch, K')
        """
        if states.dim() == 3:
            # states: (B, K', state_dim), z: (B, latent_dim)
            B, K, _ = states.shape
            z_expanded = z.unsqueeze(1).expand(B, K, -1)
            x = torch.cat([states, z_expanded], dim=-1)  # (B, K', state_dim+latent_dim)
            out = self.net(x).squeeze(-1)  # (B, K')
        else:
            x = torch.cat([states, z], dim=-1)
            out = self.net(x).squeeze(-1)
        return out
