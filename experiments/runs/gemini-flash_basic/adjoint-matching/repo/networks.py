import torch
import torch.nn as nn

class VectorField(nn.Module):
    """Abstract base class for a vector field (v or epsilon)."""
    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward pass.")

class RewardModel(nn.Module):
    """Abstract base class for a reward model r(x)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward pass.")

# Placeholder for a concrete neural network implementation of a VectorField
# This would typically be a U-Net for images, or an MLP for simpler data.
class SimpleVectorField(VectorField):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim), # +1 for time t
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        # Ensure t is a tensor and broadcastable
        t_tensor = torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
        input_tensor = torch.cat([x, t_tensor], dim=-1)
        return self.network(input_tensor)

# Placeholder for a concrete neural network implementation of a RewardModel
class SimpleRewardModel(RewardModel):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output a single scalar reward
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1) # Ensure scalar output

