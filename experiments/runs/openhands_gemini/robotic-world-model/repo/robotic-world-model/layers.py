
import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) module.
    Used for RWM heads and Policy/Value networks.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: list, activation: str = 'ReLU'):
        super().__init__()
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'ELU':
                layers.append(nn.ELU())
            elif activation == 'Tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# Placeholder for any other specific layers if they emerge during implementation.
