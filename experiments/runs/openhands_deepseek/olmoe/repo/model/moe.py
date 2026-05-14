import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .layers import SwiGLUMLP


class Router(nn.Module):
    """Learned linear router for token-to-expert assignment."""
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, d_model))
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.weight, std=0.02, a=-0.06, b=0.06)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer with dropless token choice routing.
    
    Implements Equation 1:
        MoE_module(x) = sum_{i in Top-k(r(x))} softmax(r(x))_i * E_i(x)
    
    Uses dropless routing (Gale et al., 2022): every token gets processed by exactly k experts.
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int = 64,
        num_activated_experts: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_activated_experts = num_activated_experts
        self.ffn_dim = ffn_dim

        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList([
            SwiGLUMLP(d_model, ffn_dim, dropout) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor (B, T, d_model)
        Returns:
            output: (B, T, d_model)
            router_logits: (B, T, num_experts) for loss computation
            router_probs: (B, T, num_experts) softmax probabilities
        """
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, d_model)

        router_logits = self.router(x_flat)  # (B*T, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection: get indices of k experts with highest routing probabilities
        topk_probs, topk_indices = torch.topk(router_probs, self.num_activated_experts, dim=-1)
        # Re-normalize probabilities over selected experts
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (topk_indices == expert_idx).any(dim=-1)  # (B*T,)
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[expert_idx](expert_input)

                # Find positions and corresponding routing weights for each token
                token_indices = torch.where(expert_mask)[0]
                idx_in_topk = (topk_indices[token_indices] == expert_idx).float().argmax(dim=-1)
                routing_weights = topk_probs[token_indices, idx_in_topk]

                output[token_indices] += expert_output * routing_weights.unsqueeze(-1)

        output = output.view(B, T, C)
        return output, router_logits.view(B, T, -1), router_probs.view(B, T, -1)


def compute_load_balancing_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    num_activated_experts: int = 8,
) -> torch.Tensor:
    """
    Compute load balancing loss (Equation 3).
    
    L_LB = N_E * sum_{i=1}^{N_E} f_i * P_i
    
    where:
        f_i = fraction of tokens dispatched to expert i (based on top-k selection)
        P_i = average softmax probability assigned to expert i across all tokens
    
    Both f_i and P_i are computed per batch.
    
    Args:
        router_logits: raw router logits (B, T, num_experts)
        num_experts: total number of experts N_E
        num_activated_experts: k, number of top-k experts activated per token
    """
    B, T, N = router_logits.shape
    router_probs = torch.softmax(router_logits, dim=-1)

    # f_i: fraction of tokens where expert i is among the top-k selected experts
    _, topk_indices = torch.topk(router_logits, num_activated_experts, dim=-1)  # (B, T, k)
    # Create a mask of which experts are selected
    selected_mask = torch.zeros_like(router_logits)  # (B, T, N)
    selected_mask.scatter_(-1, topk_indices, 1.0)
    # f_i = fraction of tokens routed to expert i
    f_i = selected_mask.mean(dim=(0, 1))  # (N,)

    # P_i: average router probability allocated to expert i
    P_i = router_probs.mean(dim=(0, 1))  # (N,)

    # L_LB = N_E * sum(f_i * P_i)
    load_balancing_loss = num_experts * (f_i * P_i).sum()

    return load_balancing_loss


def compute_router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute router z-loss (Equation 4).
    
    L_RZ(x) = (1 / B) * sum_{i=1}^{B} (log sum_{j=1}^{N_E} exp(x_j^{(i)}))^2
    
    Penalizes large logits into the router to improve stability.
    """
    # router_logits: (B, T, num_experts)
    B, T, N = router_logits.shape

    # Compute log-sum-exp over experts for each token
    logsumexp = torch.logsumexp(router_logits, dim=-1)  # (B, T)
    # Square and average over batch and sequence
    z_loss = (logsumexp ** 2).mean()

    return z_loss
