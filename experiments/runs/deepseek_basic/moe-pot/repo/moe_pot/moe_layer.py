"""
Mixture of Experts Layer for MoE-POT.

Implements the core MoE architecture with:
- 16 routed experts (top-K=4 dynamically selected per input)
- 2 shared experts (always activated)
- Router-gating network (CNN-based) for expert selection
- Load balancing loss using coefficient of variation (CV)

Based on: DeepSeekMoE [8], Switch Transformer [11], Shazeer et al. [51]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ExpertCNN(nn.Module):
    """
    A single expert implemented as a convolutional subnetwork.
    Following the paper: experts are CNNs to preserve spatial information.
    Uses a simple two-convolution residual block design.
    """
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + identity


class RouterGating(nn.Module):
    """
    Router-gating network for expert selection.
    Takes spatial features and produces routing logits for each routed expert.
    
    The router is implemented as a CNN-based network to preserve spatial 
    information, as described in Appendix B.2.
    """
    def __init__(self, dim: int, num_routed_experts: int = 16, top_k: int = 4):
        super().__init__()
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        
        # CNN-based router: spatial average pooling + linear projection
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_routed_experts),
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] spatial features
        
        Returns:
            topk_weights: [B, top_k] normalized routing weights for selected experts
            topk_indices: [B, top_k] indices of selected experts
        """
        # Compute routing logits
        logits = self.router(x)  # [B, N_r]
        
        # Softmax over experts
        routing_weights = F.softmax(logits, dim=-1)  # [B, N_r]
        
        # Top-K selection
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize the top-k weights to sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        return topk_weights, topk_indices


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer for PDE operator learning.
    
    Architecture:
    - N_s = 2 shared experts (always activated for every input)
    - N_r = 16 routed experts (top-K = 4 selected dynamically)
    
    Final output = (1/N_s) * sum(shared_experts(x)) + sum(w_k * routed_expert_k(x))
    
    As described in Section 4, Equation (6):
        z^{l+1}(x) = 1/N_s * Σ E_i^{l(s)}(z_0^l(x)) + Σ w_k^l * E_{i_k}^{l(r)}(z_0^l(x))
    """
    
    def __init__(
        self,
        dim: int,
        num_routed_experts: int = 16,
        num_shared_experts: int = 2,
        top_k: int = 4,
        expert_kernel_size: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        
        # Shared experts (always activated)
        self.shared_experts = nn.ModuleList([
            ExpertCNN(dim, expert_kernel_size)
            for _ in range(num_shared_experts)
        ])
        
        # Routed experts (selected by router)
        self.routed_experts = nn.ModuleList([
            ExpertCNN(dim, expert_kernel_size)
            for _ in range(num_routed_experts)
        ])
        
        # Router-gating network
        self.router = RouterGating(dim, num_routed_experts, top_k)
        
        # For interpretability analysis: store routing weights
        self.routing_weights = None
        self.routing_indices = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] spatial feature map
            
        Returns:
            out: [B, C, H, W] output after MoE computation
        """
        B, C, H, W = x.shape
        
        # 1. Compute shared expert outputs (always activated)
        shared_out = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x)
        shared_out = shared_out / self.num_shared_experts
        
        # 2. Compute routing weights and select routed experts
        topk_weights, topk_indices = self.router(x)  # [B, K], [B, K]
        
        # Store for interpretability and load balancing
        self.routing_weights = topk_weights.detach()
        self.routing_indices = topk_indices.detach()
        
        # 3. Compute routed expert outputs (sparse activation)
        routed_out = torch.zeros_like(x)
        
        # Process each selected routed expert
        for k in range(self.top_k):
            # For each sample in the batch, get the k-th selected expert index
            expert_indices = topk_indices[:, k]  # [B]
            expert_weights = topk_weights[:, k]  # [B]
            
            # We need to compute each expert's output for the samples that selected it
            # For efficiency, we group samples by expert index
            for expert_idx in range(self.num_routed_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    selected_x = x[mask]
                    expert_out = self.routed_experts[expert_idx](selected_x)
                    # Apply weight: shape [N_selected, 1, 1, 1]
                    weight = expert_weights[mask].view(-1, 1, 1, 1)
                    routed_out[mask] = routed_out[mask] + weight * expert_out
        
        # 4. Combine shared and routed outputs
        out = shared_out + routed_out
        
        return out
    
    def get_load_balancing_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the load balancing loss for this layer.
        
        As described in Section 4:
        - Importance_i = Σ w_{i,b} over batch
        - L_balance = w_bal * CV({Importance_i})^2
        
        Uses coefficient of variation (CV) squared to encourage uniform expert usage.
        
        Args:
            x: [B, C, H, W] input to recompute routing weights
            
        Returns:
            loss: scalar load balancing loss
        """
        # Get routing weights (full softmax, not just top-K)
        logits = self.router.router(x)  # [B, N_r]
        routing_weights = F.softmax(logits, dim=-1)  # [B, N_r]
        
        # Compute importance per expert
        importance = routing_weights.sum(dim=0)  # [N_r]
        
        # Coefficient of variation: CV = std / mean
        mean_imp = importance.mean()
        std_imp = importance.std()
        
        # Avoid division by zero
        cv = std_imp / (mean_imp + 1e-8)
        
        # CV squared as loss
        loss = cv ** 2
        
        return loss
