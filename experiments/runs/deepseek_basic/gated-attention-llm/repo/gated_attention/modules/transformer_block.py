"""
Transformer blocks with gated attention and FFN.

Implements the decoder blocks used in the paper's MoE and dense models.
Supports:
  - Standard dense FFN (SwiGLU)
  - Mixture of Experts (MoE) FFN with fine-grained experts
  - Gated attention (via GatedAttention)
  - Pre-norm architecture with optional sandwich norm
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating import GatedAttention, GatedAttentionConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020).

    FFN(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))


class MoEFFN(nn.Module):
    """Mixture of Experts FFN with fine-grained experts.

    Implements the MoE architecture described in Sec 3.1:
      - 128 total experts with top-8 softmax gating
      - Fine-grained experts (Dai et al., 2024)
      - Z-loss (Zoph et al., 2022) for load balancing
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 128,
        top_k: int = 8,
        dropout: float = 0.0,
        use_z_loss: bool = True,
        z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_z_loss = use_z_loss
        self.z_loss_coef = z_loss_coef

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Expert parameters
        self.gate_proj = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
        self.up_proj = nn.Parameter(torch.empty(num_experts, d_model, d_ff))
        self.down_proj = nn.Parameter(torch.empty(num_experts, d_ff, d_model))

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        std = math.sqrt(2.0 / (5 * self.d_model))
        for proj in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(proj, std=std)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        batch, seq_len, d_model = x.shape

        # Compute router logits
        router_logits = self.router(x)  # (batch, seq_len, num_experts)

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # normalized over top-k

        # Z-loss for load balancing
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.use_z_loss and self.training:
            # Z-loss: mean(log(sum(exp(router_logits), dim=-1))^2)
            z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
            aux_loss = self.z_loss_coef * z_loss

        # Compute expert outputs
        output = torch.zeros(batch, seq_len, d_model, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_idx = top_k_indices[..., k]  # (batch, seq_len)
            weight = top_k_weights[..., k:k+1]  # (batch, seq_len, 1)

            # Gather parameters for selected expert
            gate_w = self.gate_proj[expert_idx]    # (batch, seq_len, d_model, d_ff)
            up_w = self.up_proj[expert_idx]
            down_w = self.down_proj[expert_idx]

            # Compute expert output
            x_gate = torch.einsum("bsi,bsio->bso", x, gate_w)
            x_up = torch.einsum("bsi,bsio->bso", x, up_w)
            expert_out = F.silu(x_gate) * x_up
            expert_out = torch.einsum("bsi,bsio->bso", expert_out, down_w)

            output = output + weight * expert_out

        return self.dropout(output), {"router_z_loss": aux_loss}


class GatedDecoderBlock(nn.Module):
    """Pre-norm decoder block with gated attention and FFN.

    Layout:
      x -> RMSNorm -> GatedAttention -> residual (+)
      x -> RMSNorm -> FFN (dense or MoE) -> residual (+)

    Optional sandwich norm (Ding et al., 2021):
      Normalizes attention/FFN outputs before adding to residual stream.
    """

    def __init__(
        self,
        d_model: int,
        attn_config: Optional[GatedAttentionConfig] = None,
        d_ff: int = None,
        ffn_type: str = "swiglu",
        moe_num_experts: int = 128,
        moe_top_k: int = 8,
        dropout: float = 0.0,
        use_sandwich_norm: bool = False,
        sandwich_eps: float = 1e-6,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_sandwich_norm = use_sandwich_norm

        if d_ff is None:
            d_ff = 4 * d_model

        # Attention
        self.attn_norm = RMSNorm(d_model)
        self.attn = GatedAttention(attn_config) if attn_config is not None else None

        # FFN
        self.ffn_norm = RMSNorm(d_model)
        if ffn_type == "swiglu":
            self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
        elif ffn_type == "moe":
            self.ffn = MoEFFN(d_model, d_ff, moe_num_experts, moe_top_k, dropout)
        else:
            raise ValueError(f"Unknown FFN type: {ffn_type}")

        # Optional sandwich norms
        if use_sandwich_norm:
            self.attn_sandwich_norm = RMSNorm(d_model, eps=sandwich_eps)
            self.ffn_sandwich_norm = RMSNorm(d_model, eps=sandwich_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # Attention sub-layer
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        attn_output, attn_weights, past_kv = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        if self.use_sandwich_norm:
            attn_output = self.attn_sandwich_norm(attn_output)
        hidden_states = residual + attn_output

        # FFN sub-layer
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        if isinstance(ffn_output, tuple):
            ffn_output, aux_losses = ffn_output
        else:
            aux_losses = {}
        if self.use_sandwich_norm:
            ffn_output = self.ffn_sandwich_norm(ffn_output)
        hidden_states = residual + ffn_output

        return hidden_states, attn_weights, past_kv, aux_losses
