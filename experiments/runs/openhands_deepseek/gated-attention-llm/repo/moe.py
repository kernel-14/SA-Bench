"""
Mixture of Experts components following the paper's setup:
- Fine-grained experts (Dai et al., 2024 - DeepSeekMoE)
- Top-k softmax gating
- Global-batch Load Balancing Loss (Qiu et al., 2025)
- Z-loss (Zoph et al., 2022 - ST-MoE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from gated_attention import GatedMLP


class MoEGate(nn.Module):
    """
    MoE router with top-k softmax gating, load balancing loss, and Z-loss.

    The paper uses:
    - 128 total experts with top-8 softmax gating
    - Fine-grained experts
    - Global-batch LBL (Qiu et al., 2025)
    - Z-loss (Zoph et al., 2022)
    """
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        n_active: int,
        z_loss_coef: float = 0.001,
        lbl_coef: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_active = n_active
        self.z_loss_coef = z_loss_coef
        self.lbl_coef = lbl_coef

        self.router = nn.Linear(d_model, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            dispatch_weights: (batch_size * seq_len, n_active) normalized weights
            combine_weights: (batch_size * seq_len, n_active, 1)
            expert_indices: (batch_size * seq_len, n_active)
            aux_loss: scalar
        """
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*S, D)

        # Router logits
        logits = self.router(x_flat)  # (B*S, n_experts)

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, self.n_active, dim=-1)  # (B*S, n_active)

        # Softmax over selected experts
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # (B*S, n_active)

        # Z-loss: encourages router logits to be small
        z_loss = self.z_loss_coef * torch.mean(torch.square(logits))

        # Load balancing loss (global-batch LBL from Qiu et al., 2025)
        # Mean dispatch weight per expert across the batch
        mask = torch.zeros_like(logits).scatter_(
            -1, top_k_indices, top_k_weights
        )  # (B*S, n_experts)
        fraction_per_expert = mask.sum(dim=0) / (B * S)  # (n_experts,)
        load_balancing_loss = self.lbl_coef * self.n_experts * torch.sum(
            fraction_per_expert * fraction_per_expert
        )

        aux_loss = z_loss + load_balancing_loss

        return top_k_weights, top_k_indices, aux_loss


class MoELayer(nn.Module):
    """
    Mixture-of-Experts FFN layer.

    Uses fine-grained experts where each expert is a smaller FFN.
    The paper uses 128 experts, each being a SwiGLU MLP.
    """
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        n_active: int,
        expert_intermediate_dim: int,
        z_loss_coef: float = 0.001,
        lbl_coef: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.n_active = n_active

        self.gate = MoEGate(
            d_model, n_experts, n_active, z_loss_coef, lbl_coef
        )

        # Create experts
        self.experts = nn.ModuleList([
            GatedMLP(d_model, expert_intermediate_dim)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: scalar
        """
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*S, D)

        weights, indices, aux_loss = self.gate(x)  # weights: (B*S, n_active)

        # Sparse computation: compute only activated experts
        output = torch.zeros_like(x_flat)

        # For each position, compute weighted sum of expert outputs
        for expert_idx in range(self.n_active):
            expert_ids = indices[:, expert_idx]  # (B*S,)
            expert_weights = weights[:, expert_idx]  # (B*S,)

            # For each unique expert, process its assigned tokens
            unique_experts = torch.unique(expert_ids)

            for exp_id in unique_experts:
                mask = expert_ids == exp_id
                if mask.sum() == 0:
                    continue
                token_inputs = x_flat[mask]
                expert_out = self.experts[exp_id](token_inputs)
                output[mask] += expert_out * expert_weights[mask].unsqueeze(-1)

        output = output.reshape(B, S, D)
        return output, aux_loss


class MoETransformerLayer(nn.Module):
    """
    Single transformer layer with MoE FFN and gated attention.
    """
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_experts: int,
        n_active: int,
        expert_intermediate_dim: int,
        norm_eps: float = 1e-6,
        use_sandwich_norm: bool = False,
        z_loss_coef: float = 0.001,
        lbl_coef: float = 0.01,
    ):
        super().__init__()
        from gated_attention import RMSNorm

        self.attention = attention
        self.moe = MoELayer(
            d_model, n_experts, n_active, expert_intermediate_dim,
            z_loss_coef, lbl_coef
        )

        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.use_sandwich_norm = use_sandwich_norm
        if use_sandwich_norm:
            self.norm_attn = RMSNorm(d_model, eps=norm_eps)
            self.norm_ffn = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x: torch.Tensor, attention_mask=None):
        # Attention sublayer
        if self.use_sandwich_norm:
            residual = x
            x = self.norm1(x)
            attn_out = self.attention(x, attention_mask)
            attn_out = self.norm_attn(attn_out)
            x = residual + attn_out
        else:
            x = x + self.attention(self.norm1(x), attention_mask)

        # FFN sublayer
        if self.use_sandwich_norm:
            residual = x
            x = self.norm2(x)
            ffn_out, aux_loss = self.moe(x)
            ffn_out = self.norm_ffn(ffn_out)
            x = residual + ffn_out
        else:
            residual = x
            x = self.norm2(x)
            ffn_out, aux_loss = self.moe(x)
            x = residual + ffn_out

        return x, aux_loss
