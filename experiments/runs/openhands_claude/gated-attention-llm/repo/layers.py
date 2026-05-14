"""
Transformer layer components:
  - SwiGLU FFN
  - MoE layer (DeepSeekMoE-style fine-grained experts, top-k softmax routing,
    Z-loss, global-batch load-balancing loss)
  - TransformerBlock (pre-norm, optional sandwich norm)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FFNConfig, MoEConfig, TransformerBlockConfig, GatingConfig
from modules import GatedMultiHeadAttention, RMSNorm


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU FFN (Shazeer, 2020): FFN(x) = (SiLU(x W1) ⊙ (x W3)) W2."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        if self.training and self.dropout > 0:
            hidden = F.dropout(hidden, p=self.dropout)
        return self.down_proj(hidden)


# ---------------------------------------------------------------------------
# MoE Expert
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    """Single SwiGLU expert for MoE."""

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# MoE Layer (DeepSeekMoE-style fine-grained experts, top-k softmax routing)
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """Mixture-of-Experts layer as used in the 15A2B model (Sec. 3.1).

    Routing: top-k softmax gating (Dai et al., 2024 / DeepSeekMoE).
    Auxiliary losses:
      - Z-loss (Zoph et al., 2022): penalises large router logits.
      - Load-balancing loss (global-batch LBL, Qiu et al., 2025).
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_tok

        # Router
        self.router = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

        # Expert pool
        self.experts = nn.ModuleList(
            [Expert(cfg.d_model, cfg.expert_d_ffn) for _ in range(cfg.num_experts)]
        )

        # Optional shared expert (always active, not routed)
        self.shared_expert = (
            SwiGLUFFN(cfg.d_model, cfg.expert_d_ffn * cfg.num_shared_experts)
            if cfg.num_shared_experts > 0
            else None
        )

    def _z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Z-loss (Zoph et al., 2022): L_z = (1/n) * sum(log(sum(exp(x_i)))^2)."""
        log_z = torch.logsumexp(router_logits, dim=-1)  # (batch * seq_len,)
        return self.cfg.z_loss_coeff * (log_z ** 2).mean()

    def _lb_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Load-balancing loss (global-batch LBL, Qiu et al., 2025).

        L_lb = num_experts * sum_i(f_i * P_i)
        where f_i = fraction of tokens routed to expert i,
              P_i = mean router probability for expert i.
        """
        num_experts = self.num_experts
        # Expert load: fraction of tokens assigned to each expert
        expert_mask = F.one_hot(expert_indices, num_classes=num_experts).float()
        # expert_mask: (num_tokens, top_k, num_experts)
        tokens_per_expert = expert_mask.sum(dim=(0, 1)) / (num_tokens * self.top_k)
        # Mean router probability per expert
        mean_prob = router_probs.mean(dim=0)  # (num_experts,)
        lb = self.cfg.lb_loss_coeff * num_experts * (tokens_per_expert * mean_prob).sum()
        return lb

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            aux_losses: dict with 'z_loss' and 'lb_loss'
        """
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)
        num_tokens = x_flat.shape[0]

        # Router logits and probabilities
        router_logits = self.router(x_flat)  # (num_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Renormalise top-k weights
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Auxiliary losses
        z_loss = self._z_loss(router_logits)
        lb_loss = self._lb_loss(router_probs, topk_indices, num_tokens)

        # Dispatch tokens to experts
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]   # (num_tokens,)
            weight = topk_weights[:, k]        # (num_tokens,)
            for e in range(self.num_experts):
                token_mask = expert_idx == e
                if not token_mask.any():
                    continue
                expert_input = x_flat[token_mask]
                expert_out = self.experts[e](expert_input)
                output[token_mask] += weight[token_mask].unsqueeze(-1) * expert_out

        # Add shared expert contribution
        if self.shared_expert is not None:
            output = output + self.shared_expert(x_flat)

        output = output.view(batch, seq_len, d_model)
        return output, {"z_loss": z_loss, "lb_loss": lb_loss}


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional sandwich norm (Ding et al., 2021).

    Sandwich norm applies RMSNorm to attention/FFN outputs *before* the
    residual addition, preventing large activations from entering the residual
    stream (Sec. 4.3 / Table 2 row 7).
    """

    def __init__(self, cfg: TransformerBlockConfig):
        super().__init__()
        d = cfg.d_model
        eps = cfg.norm_eps

        self.attn_norm = RMSNorm(d, eps=eps)
        self.ffn_norm = RMSNorm(d, eps=eps)

        self.attn = GatedMultiHeadAttention(
            d_model=d,
            num_heads=cfg.attention.num_heads,
            num_kv_heads=cfg.attention.num_kv_heads,
            head_dim=cfg.attention.head_dim,
            max_seq_len=cfg.attention.max_seq_len,
            rope_base=cfg.attention.rope_base,
            dropout=cfg.attention.dropout,
            gating_cfg=cfg.attention.gating,
            norm_eps=eps,
        )

        self.use_moe = cfg.use_moe
        if cfg.use_moe:
            self.ffn = MoELayer(cfg.moe)
        else:
            self.ffn = SwiGLUFFN(d, cfg.ffn.d_ffn, cfg.ffn.dropout)

        # Sandwich norm: extra norms applied to sub-layer outputs before residual
        self.sandwich_norm = cfg.sandwich_norm
        if cfg.sandwich_norm:
            self.attn_post_norm = RMSNorm(d, eps=eps)
            self.ffn_post_norm = RMSNorm(d, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple], dict[str, torch.Tensor]]:
        """
        Returns:
            x:              updated hidden states (batch, seq_len, d_model)
            new_cache:      updated KV cache (or None)
            aux_losses:     dict of auxiliary losses (empty for dense models)
        """
        # Attention sub-layer
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, new_cache, _ = self.attn(
            x_norm, attention_mask=attention_mask,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        if self.sandwich_norm:
            attn_out = self.attn_post_norm(attn_out)
        x = residual + attn_out

        # FFN sub-layer
        residual = x
        x_norm = self.ffn_norm(x)
        aux_losses: dict[str, torch.Tensor] = {}
        if self.use_moe:
            ffn_out, aux_losses = self.ffn(x_norm)
        else:
            ffn_out = self.ffn(x_norm)
        if self.sandwich_norm:
            ffn_out = self.ffn_post_norm(ffn_out)
        x = residual + ffn_out

        return x, new_cache, aux_losses
