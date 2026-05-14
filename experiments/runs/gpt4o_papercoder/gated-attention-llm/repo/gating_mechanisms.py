# gating_mechanisms.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class GatedAttention:
    """
    Implements various gating mechanisms as outlined in the paper, including
    elementwise/headwise gating, additive/multiplicative gating, and applications
    at different attention positions (e.g., G1 through G5).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the GatedAttention with configuration parameters.

        Args:
            config (dict): Configuration dictionary loaded from config.yaml.
        """
        self.gating_type = config["model"]["gating_config"].get("type", "multiplicative")
        self.granularity = config["model"]["gating_config"].get("granularity", "elementwise")
        self.activation_function = config["model"]["gating_config"].get("activation_function", "sigmoid")
        self.valid_positions = ["G1", "G2", "G3", "G4", "G5"]

        # Ensure only valid gating positions are used
        self.positions = config["model"].get("gated_attention_positions", [])
        if not all(pos in self.valid_positions for pos in self.positions):
            raise ValueError(f"Invalid gating positions specified. Valid positions: {self.valid_positions}")

        # Placeholder for learnable gating weights
        self.gating_weights = {}

    def initialize_gating_weights(self, input_dim: int, num_heads: int, gated_position: str):
        """
        Initializes the gating weights based on granularity and position.

        Args:
            input_dim (int): Input dimensionality of the gating mechanism.
            num_heads (int): Number of attention heads.
            gated_position (str): Position in the attention layer (e.g., G1, G2).
        """
        if gated_position not in self.valid_positions:
            raise ValueError(f"Gated position {gated_position} is not valid. Must be one of {self.valid_positions}.")

        # Gating weights differ based on granularity
        if self.granularity == "elementwise":
            self.gating_weights[gated_position] = nn.Parameter(
                torch.randn(input_dim), requires_grad=True
            )
        elif self.granularity == "headwise":
            self.gating_weights[gated_position] = nn.Parameter(
                torch.randn(num_heads), requires_grad=True
            )
        else:
            raise ValueError("Unsupported granularity. Use 'elementwise' or 'headwise'.")

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the activation function (sigmoid or SiLU) to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor.
        """
        if self.activation_function == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation_function == "SiLU":
            return F.silu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_function}")

    def apply_gating(self, input_tensor: torch.Tensor, position: str, granularity: str) -> torch.Tensor:
        """
        Applies the gating mechanism to the input tensor at the specified position.

        Args:
            input_tensor (torch.Tensor): Input tensor to apply gating on.
            position (str): Position in the attention pipeline (e.g., G1, G2).
            granularity (str): Granularity of the gating (e.g., "elementwise", "headwise").

        Returns:
            torch.Tensor: Output tensor after applying gating.
        """
        # Validate position and granularity
        if position not in self.positions:
            raise ValueError(f"Invalid gating position: {position}. Must be one of {self.positions}.")
        if granularity not in ["elementwise", "headwise"]:
            raise ValueError(f"Unsupported granularity: {granularity}. Use 'elementwise' or 'headwise'.")

        # Initialize weights if needed
        if position not in self.gating_weights:
            input_dim = input_tensor.size(-1)
            num_heads = input_tensor.size(1) if len(input_tensor.size()) > 2 else 1
            self.initialize_gating_weights(input_dim, num_heads, position)

        # Compute gating scores
        gating_score = self._apply_activation(
            torch.matmul(input_tensor, self.gating_weights[position])
        )

        # Adjust scores for granularity
        if granularity == "headwise":
            gating_score = gating_score.mean(dim=-1, keepdim=True)  # Aggregate scores per head

        # Apply gating based on type
        if self.gating_type == "multiplicative":
            gated_output = input_tensor * gating_score
        elif self.gating_type == "additive":
            gated_output = input_tensor + gating_score
        else:
            raise ValueError(f"Unsupported gating type: {self.gating_type}. Use 'multiplicative' or 'additive'.")

        return gated_output
