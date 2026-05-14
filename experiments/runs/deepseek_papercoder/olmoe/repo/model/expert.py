# model/expert.py
"""
Defines a single Mixture-of-Experts (MoE) expert.

Each expert is a small SwiGLU feed‑forward network following the
OLMoE‑1B‑7B architecture (hidden_size = 2048, ffn_size = 1024,
truncated normal initialization with std = 0.02, truncation at ±3σ).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Expert(nn.Module):
    """A single fine‑grained expert in the MoE module.

    Forward pass:
        gate = silu(gate_proj(x))    # SiLU activation
        up   = up_proj(x)
        hidden = gate * up
        output = down_proj(hidden)

    Parameters:
        hidden_size:  Model dimension (input and output size).
        ffn_size:     Intermediate size of the expert's hidden layer.
        init_std:     Standard deviation for truncated normal weight init.
        init_trunc:   Truncation factor (e.g., 3 means truncate at ±3σ).
        bias:         Whether to use bias in linear layers (paper uses False).
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        ffn_size: int = 1024,
        init_std: float = 0.02,
        init_trunc: int = 3,
        bias: bool = False,
    ) -> None:
        super().__init__()
        # SwiGLU components
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=bias)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=bias)

        # Initialize weights with truncated normal distribution (Section 4.2.2)
        self._init_weights(init_std, init_trunc)

    def _init_weights(self, init_std: float, init_trunc: int) -> None:
        """Truncated normal initialization for all weight matrices."""
        with torch.no_grad():
            for module in [self.gate_proj, self.up_proj, self.down_proj]:
                # module.weight will always exist because bias=False only affects bias
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=init_std,
                    a=-init_trunc * init_std,
                    b=init_trunc * init_std,
                )
            # No bias initialization necessary because bias is disabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (num_tokens, hidden_size).
        Returns:
            Output tensor of the same shape.
        """
        gate = self.gate_proj(x)          # (num_tokens, ffn_size)
        up = self.up_proj(x)              # (num_tokens, ffn_size)

        # SwiGLU: element‑wise multiplication of SiLU(gate) and up
        hidden = F.silu(gate) * up        # (num_tokens, ffn_size)

        output = self.down_proj(hidden)  # (num_tokens, hidden_size)
        return output
