
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# Example of NS-sigmoid
class NS_Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return 0.5 + 0.5 * self.sigmoid(x)

class GatedMechanism(nn.Module):
    """
    Implements the gated mechanism as described in Section 2.2.
    Y' = g(Y, X, W_theta, sigma) = Y * sigma(X W_theta)  (Multiplicative Gating)
    Y' = Y + sigma(X W_theta)  (Additive Gating)
    Where Y is the input to be modulated, X is another input used to compute the gating scores.
    The paper primarily focuses on SDPA output gating (G1) where Y is the SDPA output
    and X is derived from the query. For other positions, X and Y vary.
    """
    def __init__(
        self,
        output_dim: int, # dimension of Y, also the dimension of the gate output
        gating_dim: int,  # dimension of X, the input to the gate linear layer
        gating_granularity: str = "elementwise", # "elementwise" or "headwise"
        head_specific: bool = True, # True for head-specific, False for head-shared
        gating_type: str = "multiplicative", # "multiplicative" or "additive"
        activation_function: str = "sigmoid", # "sigmoid", "SiLU", "identity", "ns_sigmoid"
        num_heads: Optional[int] = None, # Required for headwise/head-specific, or for elementwise head-shared (dk)
        head_dim: Optional[int] = None, # Required for elementwise head-shared (dk)
    ):
        super().__init__()
        self.gating_granularity = gating_granularity
        self.head_specific = head_specific
        self.gating_type = gating_type
        self.activation_function_name = activation_function
        self.num_heads = num_heads
        self.head_dim = head_dim

        if gating_granularity == "headwise":
            assert num_heads is not None, "num_heads must be provided for headwise gating."
            if head_specific:
                self.gate_linear = nn.Linear(gating_dim, num_heads)
            else:
                self.gate_linear = nn.Linear(gating_dim, 1) # Single scalar per position, shared across all heads
        elif gating_granularity == "elementwise":
            if head_specific:
                # Elementwise, head-specific: Output matches the total dimension of Y
                self.gate_linear = nn.Linear(gating_dim, output_dim)
            else:
                # Elementwise, head-shared: Score shape `n x dk` for G1 (Table 1).
                # This means the gate output has dimension `head_dim` (dk) and is shared across heads.
                assert head_dim is not None, "head_dim must be provided for elementwise head-shared gating."
                self.gate_linear = nn.Linear(gating_dim, head_dim)
        else:
            raise ValueError(f"Unsupported gating granularity: {gating_granularity}")

        self.activation_fn = self._get_activation_fn(activation_function)

    def _get_activation_fn(self, name: str):
        if name == "sigmoid":
            return nn.Sigmoid()
        elif name == "SiLU":
            return nn.SiLU()
        elif name == "identity":
            return lambda x: x
        elif name == "ns_sigmoid":
            return NS_Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, modulated_input: torch.Tensor, gate_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modulated_input (torch.Tensor): Y in the paper (e.g., SDPA output)
                                            Shape: (batch_size, seq_len, num_heads, head_dim) or (batch_size, seq_len, d_model)
            gate_input (torch.Tensor): X in the paper (e.g., query, or input_hidden_state)
                                       Shape: (batch_size, seq_len, gating_dim)
        Returns:
            torch.Tensor: Gated output Y'
        """
        if gate_input.dim() == 4:
            B, S, H, D_H = gate_input.shape
            gate_input_flat = gate_input.view(B, S, H * D_H)
        else:
            gate_input_flat = gate_input # Expecting (B, S, gating_dim)

        raw_gate_scores = self.gate_linear(gate_input_flat)
        gating_scores = self.activation_fn(raw_gate_scores)

        original_modulated_input_shape = modulated_input.shape
        reshaped_for_gating = modulated_input

        # If modulated_input is (B, S, D_total) but gating needs to be per-head, reshape it temporarily
        # This occurs when attention output is flattened (B, S, n_heads*head_dim) but gating is headwise,
        # or elementwise head-shared (where scores are dk per position).
        if modulated_input.dim() == 3 and self.num_heads is not None and self.head_dim is not None \
           and original_modulated_input_shape[-1] == self.num_heads * self.head_dim:
            reshaped_for_gating = modulated_input.view(
                original_modulated_input_shape[0],
                original_modulated_input_shape[1],
                self.num_heads,
                self.head_dim
            )

        # Reshape gating_scores to match reshaped_for_gating for broadcasting
        if self.gating_granularity == "headwise":
            if self.head_specific: # (B, S, H) -> (B, S, H, 1)
                gating_scores = gating_scores.unsqueeze(-1)
            else: # (B, S, 1) -> (B, S, 1, 1)
                gating_scores = gating_scores.unsqueeze(-1).unsqueeze(-1)
        elif self.gating_granularity == "elementwise":
            if self.head_specific: # (B, S, H*D_H) -> (B, S, H, D_H) if reshaped_for_gating is 4D
                if reshaped_for_gating.dim() == 4:
                    B_m, S_m, H_m, D_H_m = reshaped_for_gating.shape
                    gating_scores = gating_scores.view(B_m, S_m, H_m, D_H_m)
                # If reshaped_for_gating is 3D (B, S, D_model), and gating_scores is (B, S, D_model), no reshape needed.
            else: # Head-shared elementwise for G1-G4: raw_gate_scores is (B, S, head_dim). Reshape to (B, S, 1, head_dim)
                gating_scores = gating_scores.unsqueeze(-2) # (B, S, 1, head_dim)

        if self.gating_type == "multiplicative":
            gated_output = reshaped_for_gating * gating_scores
        elif self.gating_type == "additive":
            gated_output = reshaped_for_gating + gating_scores
        else:
            raise ValueError(f"Unsupported gating type: {self.gating_type}")

        # Reshape back to original if modulated_input was reshaped
        if reshaped_for_gating is not modulated_input:
            gated_output = gated_output.view(original_modulated_input_shape)

        return gated_output
