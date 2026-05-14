"""
Mixture-of-Experts (MoE) Layer for MoE-POT.

Implements the sparse MoE layer with:
- N_r routed experts (default 16), of which Top-K are selected per input (default K=4)
- N_s shared experts (default 2), always activated
- A router-gating network (CNN-based) that computes routing logits
- Load balancing loss based on coefficient of variation (CV) of expert importance

The final output is:
    z^{l+1}(x) = (1/N_s) * sum_i E_i^{(s)}(z_0) + sum_k w_k * E_{i_k}^{(r)}(z_0)

where w_k are the top-K normalized routing weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvExpert(nn.Module):
    """
    A single CNN-based expert network.

    Each expert is a small convolutional subnetwork that maps
    (B, H, W, C) -> (B, H, W, C) while preserving spatial dimensions.

    Architecture: LayerNorm -> Conv1x1 -> GELU -> DepthwiseConv3x3 -> GELU -> Conv1x1 + residual
    """

    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, mlp_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(mlp_dim, mlp_dim, kernel_size=3, padding=1, groups=mlp_dim)
        self.conv3 = nn.Conv2d(mlp_dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
        Returns:
            (B, H, W, C)
        """
        residual = x
        x = self.norm(x)
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x + residual


class RouterGatingNetwork(nn.Module):
    """
    CNN-based router-gating network that computes routing logits.

    Takes spatial features (B, H, W, C) and outputs routing logits (B, N_r).
    Uses global average pooling to aggregate spatial information before
    computing per-expert scores.
    """

    def __init__(self, dim: int, num_routed_experts: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.proj = nn.Linear(dim, num_routed_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)
        Returns:
            logits: (B, N_r)
        """
        x = self.norm(x)
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        # Global average pooling: (B, C, H, W) -> (B, C)
        x = x.mean(dim=(2, 3))
        # Project to routing logits: (B, N_r)
        logits = self.proj(x)
        return logits


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer combining shared and routed experts.

    Args:
        dim: Feature dimension.
        mlp_dim: Hidden dimension of each expert MLP.
        num_routed_experts: Number of routed experts (N_r), default 16.
        num_shared_experts: Number of shared experts (N_s), default 2.
        top_k: Number of routed experts to activate per input (K), default 4.
        balance_weight: Weight for the load balancing loss (w_bal), default 0.1.
    """

    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        num_routed_experts: int = 16,
        num_shared_experts: int = 2,
        top_k: int = 4,
        balance_weight: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.balance_weight = balance_weight

        # Shared experts (always activated)
        self.shared_experts = nn.ModuleList([
            ConvExpert(dim, mlp_dim) for _ in range(num_shared_experts)
        ])

        # Routed experts (top-K selected per input)
        self.routed_experts = nn.ModuleList([
            ConvExpert(dim, mlp_dim) for _ in range(num_routed_experts)
        ])

        # Router-gating network
        self.router = RouterGatingNetwork(dim, num_routed_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (B, H, W, C).

        Returns:
            output: (B, H, W, C)
            balance_loss: scalar load balancing loss
        """
        B, H, W, C = x.shape

        # Compute routing logits and weights
        logits = self.router(x)  # (B, N_r)
        weights = F.softmax(logits, dim=-1)  # (B, N_r)

        # Select top-K experts
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
        # topk_weights: (B, K), topk_indices: (B, K)

        # Normalize top-K weights so they sum to 1
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute shared expert outputs (always activated)
        shared_out = torch.zeros(B, H, W, C, device=x.device, dtype=x.dtype)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x)
        shared_out = shared_out / self.num_shared_experts

        # Compute routed expert outputs efficiently
        # Strategy: compute all expert outputs for unique selected experts,
        # then aggregate with routing weights
        routed_out = torch.zeros(B, H, W, C, device=x.device, dtype=x.dtype)

        # Get unique expert indices that are selected in this batch
        unique_experts = topk_indices.reshape(-1).unique()

        # Compute outputs for each selected expert
        expert_outputs = {}
        for expert_idx in unique_experts.tolist():
            expert_outputs[expert_idx] = self.routed_experts[expert_idx](x)
            # Shape: (B, H, W, C)

        # Aggregate: for each position k in top-K, add weighted expert output
        for k in range(self.top_k):
            # topk_indices[:, k]: (B,) - expert index for each sample at position k
            # topk_weights[:, k]: (B,) - weight for each sample at position k
            expert_idx_k = topk_indices[:, k]  # (B,)
            weight_k = topk_weights[:, k]  # (B,)

            # Group samples by expert index for efficient batching
            for expert_idx in unique_experts.tolist():
                # Find samples that selected this expert at position k
                mask = (expert_idx_k == expert_idx)  # (B,)
                if not mask.any():
                    continue

                # Get the expert output for selected samples
                expert_out = expert_outputs[expert_idx]  # (B, H, W, C)
                # Weight: (B,) -> (B, 1, 1, 1) for broadcasting
                w = weight_k[mask].reshape(-1, 1, 1, 1)
                routed_out[mask] = routed_out[mask] + w * expert_out[mask]

        # Final output: shared + routed
        output = shared_out + routed_out

        # Load balancing loss
        # Importance_i = sum_b w_{i,b}
        importance = weights.sum(dim=0)  # (N_r,)
        # CV^2 = Var / Mean^2
        mean_importance = importance.mean()
        var_importance = importance.var(unbiased=False)
        cv_squared = var_importance / (mean_importance ** 2 + 1e-8)
        balance_loss = self.balance_weight * cv_squared

        return output, balance_loss

    def get_routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the full softmax routing weights for interpretability analysis.

        Args:
            x: Input features (B, H, W, C).

        Returns:
            weights: (B, N_r) full softmax distribution over all routed experts.
        """
        logits = self.router(x)
        weights = F.softmax(logits, dim=-1)
        return weights
