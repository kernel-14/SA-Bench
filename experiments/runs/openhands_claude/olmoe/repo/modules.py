"""MoE-specific modules for OLMoE.

Implements:
- Expert FFN (SwiGLU)
- Router (linear layer with Top-k token choice)
- Load balancing loss (Shazeer et al., 2017)
- Router Z-loss (Zoph et al., 2022)
- MoE module (dropless token choice routing)
- Transformer block with MoE FFN
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Attention, RMSNorm, SwiGLU


class ExpertFFN(nn.Module):
    """Single expert feed-forward network with SwiGLU activation.

    For OLMoE-1B-7B: ffn_dim=1024, d_model=2048.
    SwiGLU uses two parallel projections (gate + value) then merges.
    """

    def __init__(self, d_model: int, ffn_dim: int, use_bias: bool = False):
        super().__init__()
        # SwiGLU: project to 2*ffn_dim, split, apply silu gate
        self.gate_up_proj = nn.Linear(d_model, 2 * ffn_dim, bias=use_bias)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=use_bias)
        self.act = SwiGLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_up_proj(x)))


class Router(nn.Module):
    """Learned linear router mapping token representations to expert logits.

    Implements token choice routing: each token selects its top-k experts.
    The router is a simple linear layer (no bias) from d_model -> n_experts.
    """

    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch * seq_len, d_model) or (n_tokens, d_model)
        Returns:
            logits: (n_tokens, n_experts)
        """
        return self.linear(x)


def load_balance_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Load balancing auxiliary loss (Shazeer et al., 2017, Eq. 3 in paper).

    L_LB = N_E * sum_{i=1}^{N_E} f_i * P_i

    where:
        f_i = fraction of tokens routed to expert i (from hard top-k selection)
        P_i = mean routing probability assigned to expert i (soft, differentiable)

    Args:
        router_probs: (n_tokens, n_experts) softmax probabilities from router
        expert_indices: (n_tokens, k) top-k expert indices selected per token
        n_experts: total number of experts N_E
    Returns:
        scalar loss
    """
    n_tokens = router_probs.shape[0]

    # f_i: fraction of tokens routed to expert i (non-differentiable)
    # Build one-hot mask over all selected experts
    expert_mask = torch.zeros(n_tokens, n_experts, device=router_probs.device, dtype=router_probs.dtype)
    expert_mask.scatter_(1, expert_indices, 1.0)
    # f_i = mean over tokens of whether expert i was selected
    f = expert_mask.mean(dim=0)  # (n_experts,)

    # P_i: mean routing probability for expert i (differentiable)
    P = router_probs.mean(dim=0)  # (n_experts,)

    loss = n_experts * (f * P).sum()
    return loss


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Router Z-loss for stability (Zoph et al., 2022, Eq. 4 in paper).

    L_RZ(x) = (1/B) * sum_{i=1}^{B} (log sum_{j=1}^{N_E} exp(x_j^(i)))^2

    Penalizes large logits entering the router to prevent numeric overflow.

    Args:
        router_logits: (n_tokens, n_experts) raw logits before softmax
    Returns:
        scalar loss
    """
    log_sum_exp = torch.logsumexp(router_logits, dim=-1)  # (n_tokens,)
    loss = (log_sum_exp ** 2).mean()
    return loss


class MoEModule(nn.Module):
    """Sparse Mixture-of-Experts module with dropless token choice routing.

    Implements Equation 1 from the paper:
        MoE(x) = sum_{i in Top-k(r(x))} softmax(r(x))_i * E_i(x)

    Uses dropless routing (MegaBlocks, Gale et al. 2022): all tokens are
    processed, no token dropping. Each token selects k experts.

    For OLMoE-1B-7B: n_experts=64, k=8, ffn_dim=1024.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        n_experts_per_token: int,
        ffn_dim: int,
        use_bias: bool = False,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.k = n_experts_per_token

        self.router = Router(d_model, n_experts)
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model, ffn_dim, use_bias=use_bias) for _ in range(n_experts)]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_router_info: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: (batch, seq_len, d_model)
            return_router_info: if True, return routing metadata for analysis
        Returns:
            output: (batch, seq_len, d_model)
            aux_losses: dict with 'load_balance_loss' and 'router_z_loss'
        """
        bsz, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (n_tokens, d_model)
        n_tokens = x_flat.shape[0]

        # Router forward: compute logits and probabilities
        router_logits = self.router(x_flat)  # (n_tokens, n_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (n_tokens, n_experts)

        # Top-k selection (token choice routing)
        topk_probs, topk_indices = torch.topk(router_probs, self.k, dim=-1)
        # topk_probs: (n_tokens, k), topk_indices: (n_tokens, k)

        # Renormalize top-k probabilities (softmax over selected experts)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Compute auxiliary losses
        lb_loss = load_balance_loss(router_probs, topk_indices, self.n_experts)
        rz_loss = router_z_loss(router_logits)

        # Dispatch tokens to experts (dropless: process all tokens)
        output = torch.zeros_like(x_flat)

        # Group tokens by expert for efficient batched computation
        for expert_idx in range(self.n_experts):
            # Find which tokens selected this expert and at which position
            token_mask = (topk_indices == expert_idx)  # (n_tokens, k)
            token_positions = token_mask.nonzero(as_tuple=False)  # (n_selected, 2)

            if token_positions.shape[0] == 0:
                continue

            token_ids = token_positions[:, 0]  # which tokens
            slot_ids = token_positions[:, 1]   # which of the k slots

            expert_input = x_flat[token_ids]  # (n_selected, d_model)
            expert_output = self.experts[expert_idx](expert_input)  # (n_selected, d_model)

            # Weight by routing probability
            weights = topk_probs[token_ids, slot_ids].unsqueeze(-1)  # (n_selected, 1)
            output.index_add_(0, token_ids, expert_output * weights)

        output = output.view(bsz, seq_len, d_model)

        aux_losses = {
            "load_balance_loss": lb_loss,
            "router_z_loss": rz_loss,
        }

        if return_router_info:
            aux_losses["router_logits"] = router_logits.detach()
            aux_losses["router_probs"] = router_probs.detach()
            aux_losses["topk_indices"] = topk_indices.detach()
            aux_losses["topk_probs"] = topk_probs.detach()

        return output, aux_losses


class TransformerBlock(nn.Module):
    """Single transformer block with attention + MoE FFN.

    Pre-norm architecture (norm before attention and FFN).
    Every layer uses MoE (not every 6th like OpenMoE).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_experts: int,
        n_experts_per_token: int,
        ffn_dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_qk_norm: bool = True,
        norm_eps: float = 1e-5,
        use_bias: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.attn = Attention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            use_qk_norm=use_qk_norm,
            norm_eps=norm_eps,
            use_bias=use_bias,
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        self.moe = MoEModule(
            d_model=d_model,
            n_experts=n_experts,
            n_experts_per_token=n_experts_per_token,
            ffn_dim=ffn_dim,
            use_bias=use_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_router_info: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        # Attention with pre-norm and residual
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)

        # MoE FFN with pre-norm and residual
        moe_out, aux_losses = self.moe(self.ffn_norm(x), return_router_info=return_router_info)
        x = x + moe_out

        return x, aux_losses
