"""DeepONet (Lu et al., 2021; Wang et al., 2021) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepONet(nn.Module):
    """Deep Operator Network.

    Architecture:
    - Branch net: processes input function at sensor points
    - Trunk net: processes query coordinates (x, t)

    Args:
        branch_input_dim: Number of sensor points / input function dims.
        trunk_input_dim: Dimension of query coordinates.
        hidden_dim: Hidden layer width.
        output_dim: Output dimension.
        n_layers: Number of hidden layers in branch and trunk nets.
    """

    def __init__(
        self,
        branch_input_dim: int,
        trunk_input_dim: int = 2,
        hidden_dim: int = 40,
        output_dim: int = 1,
        n_layers: int = 3,
    ):
        super().__init__()
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        branch_layers = []
        branch_layers.append(nn.Linear(branch_input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            branch_layers.extend([nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)])
        self.branch = nn.Sequential(*branch_layers)

        trunk_layers = []
        trunk_layers.append(nn.Linear(trunk_input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            trunk_layers.extend([nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)])
        self.trunk = nn.Sequential(*trunk_layers)

        self.output_bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            branch_input: (B, branch_input_dim) or (B, N_points, branch_input_dim)
            trunk_input: (M, trunk_input_dim) query coordinates
        Returns:
            (B, M, output_dim)
        """
        if branch_input.dim() == 3:
            branch_input = branch_input.reshape(branch_input.shape[0], -1)

        b = self.branch(branch_input)  # (B, hidden_dim)
        t = self.trunk(trunk_input)     # (M, hidden_dim)

        out = torch.einsum("bh,mh->bm", b, t)
        out = out.unsqueeze(-1) + self.output_bias
        return out
