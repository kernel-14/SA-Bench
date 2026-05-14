import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LoRASBLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int, scaling_factor: float = 1.0):
        super(LoRASBLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.scaling_factor = scaling_factor

        # Fixed matrices B and A
        self.B = nn.Parameter(torch.randn(input_dim, rank), requires_grad=False)
        self.A = nn.Parameter(torch.randn(rank, output_dim), requires_grad=False)

        # Trainable matrix R
        self.R = nn.Parameter(torch.randn(rank, rank))

        # Initialize B, A, and R using truncated SVD
        self._initialize_parameters()

    def _initialize_parameters(self):
        with torch.no_grad():
            # Orthonormal initialization for B and A
            nn.init.orthogonal_(self.B)
            nn.init.orthogonal_(self.A)

            # Identity initialization for R
            nn.init.eye_(self.R)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the low-rank approximation
        low_rank_update = self.scaling_factor * torch.matmul(self.B, torch.matmul(self.R, self.A))
        return torch.matmul(x, low_rank_update)

class LoRASBModel(nn.Module):
    def __init__(self, base_model: nn.Module, rank: int, scaling_factor: float = 1.0):
        super(LoRASBModel, self).__init__()
        self.base_model = base_model
        self.rank = rank
        self.scaling_factor = scaling_factor

        # Replace layers in the base model with LoRA-SB layers
        self._replace_layers()

    def _replace_layers(self):
        for name, module in self.base_model.named_children():
            if isinstance(module, nn.Linear):
                setattr(self.base_model, name, LoRASBLayer(
                    input_dim=module.in_features,
                    output_dim=module.out_features,
                    rank=self.rank,
                    scaling_factor=self.scaling_factor
                ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)