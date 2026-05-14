import torch
import torch.nn as nn
from typing import Callable, Optional, Union, List, Dict, Any

# Ensure utils is importable. Assuming it's in the same directory or PYTHONPATH.
try:
    from utils import get_activation_fn
except ImportError:
    # Fallback for testing or specific environment setups
    def get_activation_fn(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Dummy function for testing if utils is not available."""
        if name == "sigmoid":
            return torch.sigmoid
        elif name == "silu":
            return nn.SiLU()
        elif name == "identity":
            return lambda x: x
        elif name == "ns_sigmoid":
            return lambda x: 0.5 + 0.5 * torch.sigmoid(x)
        elif name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {name}")


class GatingModule(nn.Module):
    """
    Implements a flexible gating mechanism for attention layers, supporting various
    positions, granularities, sharing schemes, types (multiplicative/additive),
    and activation functions as described in the paper.

    The gating mechanism is formalized as: Y' = g(Y, X, W_theta, sigma) = Y (*) sigma(X @ W_theta + Bias_theta),
    where (*) is multiplication for multiplicative gating or addition for additive gating.

    Attributes:
        w_theta: Learnable parameters (W_theta and Bias_theta) for computing gating scores.
                 Can be a single nn.Linear or an nn.ModuleList of nn.Linear modules.
        activation_fn: The activation function applied to the raw gating scores.
        granularity: "elementwise" or "headwise".
        head_specific: True if each attention head has its own gating parameters.
        num_heads: Total number of attention heads.
        gating_type: "multiplicative" or "additive".
        input_dim: The feature dimension of the tensor (Y) to be modulated.
        score_input_dim: The feature dimension of the tensor (X) used to compute the gating scores.
    """

    def __init__(
        self,
        input_dim: int,
        score_input_dim: int,
        granularity: str,
        head_specific: bool,
        num_heads: int,
        activation_fn_name: str,
        gating_type: str,
    ):
        """
        Initializes the GatingModule.

        Args:
            input_dim: The feature dimension of the tensor (Y) that will be modulated.
                       e.g., config.head_dim for per-head, config.d_model for post-concat.
            score_input_dim: The feature dimension of the tensor (X) used to compute the gating scores.
                             e.g., config.head_dim or config.d_model.
            granularity: Specifies the granularity of gating scores ("elementwise" or "headwise").
            head_specific: If True, each attention head has its own W_theta parameters.
                           If False, W_theta is shared across heads.
            num_heads: Total number of attention heads.
            activation_fn_name: Name of the activation function ("sigmoid", "silu", "identity", "ns_sigmoid").
            gating_type: Type of gating ("multiplicative" or "additive").
        
        Raises:
            ValueError: If an unsupported granularity, gating type, or head configuration is provided.
        """
        super().__init__()

        self.granularity: str = granularity
        self.head_specific: bool = head_specific
        self.num_heads: int = num_heads
        self.gating_type: str = gating_type
        self.input_dim: int = input_dim
        self.score_input_dim: int = score_input_dim

        # Determine the output dimension of the linear projection for W_theta
        # This determines the shape of the raw gating scores
        output_dim_w_theta: int
        if self.granularity == "elementwise":
            output_dim_w_theta = self.input_dim  # Score has same dim as the modulated input features
        elif self.granularity == "headwise":
            if self.head_specific:
                output_dim_w_theta = 1  # A single scalar score for each head's output
            else:
                output_dim_w_theta = self.num_heads  # A scalar score for each head, produced by shared W_theta
        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}. Must be 'elementwise' or 'headwise'.")

        # Initialize learnable parameters (W_theta and Bias_theta)
        if self.head_specific:
            # Each head gets its own Linear layer
            self.w_theta = nn.ModuleList(
                [nn.Linear(self.score_input_dim, output_dim_w_theta) for _ in range(self.num_heads)]
            )
        else:
            # A single Linear layer shared across all heads
            self.w_theta = nn.Linear(self.score_input_dim, output_dim_w_theta)

        # Retrieve the activation function
        self.activation_fn: Callable[[torch.Tensor], torch.Tensor] = get_activation_fn(activation_fn_name)

        if self.gating_type not in ["multiplicative", "additive"]:
            raise ValueError(f"Unsupported gating type: {self.gating_type}. Must be 'multiplicative' or 'additive'.")

    def forward(
        self,
        modulated_input: torch.Tensor,
        score_computation_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies the gating mechanism to the modulated_input.

        Args:
            modulated_input: The tensor (Y) to be modulated.
                             Shape examples: (batch_size, seq_len, num_heads, head_dim) or (batch_size, seq_len, d_model).
            score_computation_input: The tensor (X) used to compute the gating scores.
                                     Shape will be prepared by the caller to match `score_input_dim` for the
                                     last dimension.
                                     e.g., (batch_size, seq_len, num_heads, head_dim) for head-specific G1 elementwise,
                                     or (batch_size, seq_len, d_model) for head-shared G5.

        Returns:
            The gated output tensor.
        """
        raw_gate_scores: torch.Tensor

        if self.head_specific:
            # For head-specific gating, score_computation_input is expected to have a head dimension.
            # Example: (batch_size, seq_len, num_heads, head_dim)
            if score_computation_input.dim() != 4:
                raise ValueError(
                    f"Expected score_computation_input with 4 dimensions (B, S, H, D) for head-specific gating, "
                    f"but got {score_computation_input.dim()} dimensions."
                )

            raw_gate_scores_list: List[torch.Tensor] = []
            for h_idx in range(self.num_heads):
                # Apply each head's linear projection (B, S, D) -> (B, S, output_dim_w_theta)
                raw_gate_scores_list.append(self.w_theta[h_idx](score_computation_input[:, :, h_idx, :]))

            # Stack the results along a new head dimension
            # If elementwise: (B, S, H, input_dim)
            # If headwise:    (B, S, H, 1)
            raw_gate_scores = torch.stack(raw_gate_scores_list, dim=-2)

        else:  # Head-shared gating
            # score_computation_input is expected to be (B, S, score_input_dim)
            # (Averaging over heads for e.g. G1 Head-Shared is done by the caller)
            if score_computation_input.dim() != 3:
                 raise ValueError(
                    f"Expected score_computation_input with 3 dimensions (B, S, D) for head-shared gating, "
                    f"but got {score_computation_input.dim()} dimensions."
                )
            
            # Apply the shared linear projection (B, S, score_input_dim) -> (B, S, output_dim_w_theta)
            raw_gate_scores = self.w_theta(score_computation_input)

            # Reshape raw_gate_scores for broadcasting if modulated_input has a head dimension.
            # This happens for G1, G2, G3, G4 where modulated_input is (B, S, H, D).
            if modulated_input.dim() == 4:
                if self.granularity == "elementwise":
                    # raw_gate_scores is (B, S, input_dim)
                    # Expand to (B, S, 1, input_dim) to broadcast across `num_heads` in modulated_input
                    raw_gate_scores = raw_gate_scores.unsqueeze(-2)
                elif self.granularity == "headwise":
                    # raw_gate_scores is (B, S, num_heads)
                    # Expand to (B, S, num_heads, 1) to broadcast across `head_dim` in modulated_input
                    raw_gate_scores = raw_gate_scores.unsqueeze(-1)

        # Apply the activation function
        gate_scores = self.activation_fn(raw_gate_scores)

        # Apply gating (multiplicative or additive)
        if self.gating_type == "multiplicative":
            gated_output = modulated_input * gate_scores
        else:  # Additive gating
            gated_output = modulated_input + gate_scores

        return gated_output

