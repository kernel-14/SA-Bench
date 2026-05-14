"""
Modality-specific Mixture-of-Experts (MMoE) for NaViL.

The paper introduces modality-specific MoEs for both attention (MHA-MMoE) and
FFN (FFN-MMoE) layers. This allows different processing of visual and linguistic tokens
while maintaining unified global attention computation.

From the paper:
  x_{i,m}^{l'} = x_{i,m}^{l-1} + MHA-MMoE(RMSNorm(x_{i,m}^{l-1}))
  x_{i,m}^l   = x_{i,m}^{l'} + FFN-MMoE(RMSNorm(x_{i,m}^{l'}))

where m ∈ {visual, linguistic}

MHA-MMoE uses modality-specific Q, K, V, O projection matrices:
  MHA-MMoE(x_{i,m}) = (softmax(QK^T/sqrt(d)) V) W_O^m
  Q_{i,m} = x_{i,m} W_Q^m, K_{i,m} = x_{i,m} W_K^m, V_{i,m} = x_{i,m} W_V^m

FFN-MMoE uses modality-specific gate, up, down matrices:
  FFN-MMoE(x_{i,m}) = (SiLU(x_{i,m} W_gate^m) ⊙ x_{i,m} W_up^m) W_down^m

The number of activated experts is set to 1 to maintain consistent inference costs.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from .norm import get_rms_norm as RMSNorm

import torch
import torch.nn as nn
import torch.nn.functional as F


# Modality indices
VISUAL_MODALITY = 0
LINGUISTIC_MODALITY = 1


@dataclass
class MMoEConfig:
    """Configuration for Modality-specific MoE."""
    hidden_size: int = 2048
    num_heads: int = 16
    head_dim: int = 128
    intermediate_size: int = 8192  # FFN intermediate size
    num_modalities: int = 2        # visual + linguistic
    norm_eps: float = 1e-6


class ModalitySpecificAttention(nn.Module):
    """
    MHA-MMoE: Modality-specific multi-head attention.
    
    Each modality has its own Q, K, V, O projection matrices.
    Global attention is computed across all tokens (visual + linguistic) together.
    
    This addresses the feature scale mismatch between visual and language modalities
    that occurs when using only FFN experts.
    """

    def __init__(self, config: MMoEConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = self.head_dim ** -0.5

        # Modality-specific projection matrices
        # W_Q^m, W_K^m, W_V^m, W_O^m for each modality m
        self.q_projs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
            for _ in range(config.num_modalities)
        ])
        self.k_projs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
            for _ in range(config.num_modalities)
        ])
        self.v_projs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
            for _ in range(config.num_modalities)
        ])
        self.o_projs = nn.ModuleList([
            nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
            for _ in range(config.num_modalities)
        ])

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_fn=None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (B, seq_len, hidden_size) - mixed visual + linguistic tokens
            modality_ids: (B, seq_len) - 0 for visual, 1 for linguistic
            attention_mask: (B, 1, seq_len, seq_len) or None
            position_ids: (B, seq_len) for RoPE
            past_key_value: cached (K, V) for inference
            rope_fn: function to apply RoPE to (q, k, position_ids)
        Returns:
            output: (B, seq_len, hidden_size)
            new_past_key_value: updated KV cache
        """
        B, seq_len, _ = x.shape

        # Apply modality-specific projections
        # For efficiency, process each modality separately then combine
        q = torch.zeros(B, seq_len, self.num_heads * self.head_dim, device=x.device, dtype=x.dtype)
        k = torch.zeros(B, seq_len, self.num_heads * self.head_dim, device=x.device, dtype=x.dtype)
        v = torch.zeros(B, seq_len, self.num_heads * self.head_dim, device=x.device, dtype=x.dtype)

        for m in range(len(self.q_projs)):
            mask = (modality_ids == m)  # (B, seq_len)
            if mask.any():
                # Apply modality-specific projections to tokens of this modality
                # We need to handle variable-length sequences per modality
                # Use masked scatter for efficiency
                x_m = x * mask.unsqueeze(-1).float()
                q = q + self.q_projs[m](x_m) * mask.unsqueeze(-1).float()
                k = k + self.k_projs[m](x_m) * mask.unsqueeze(-1).float()
                v = v + self.v_projs[m](x_m) * mask.unsqueeze(-1).float()

        # Reshape for multi-head attention
        q = q.reshape(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if rope_fn is not None and position_ids is not None:
            q, k = rope_fn(q, k, position_ids)

        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_past_key_value = (k, v)

        # Unified global attention across all tokens (visual + linguistic)
        # This is the key design: global attention with modality-specific projections
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, seq_len, head_dim)

        attn_output = attn_output.transpose(1, 2).reshape(B, seq_len, self.num_heads * self.head_dim)

        # Apply modality-specific output projections
        output = torch.zeros(B, seq_len, self.hidden_size, device=x.device, dtype=x.dtype)
        for m in range(len(self.o_projs)):
            mask = (modality_ids == m)
            if mask.any():
                out_m = self.o_projs[m](attn_output) * mask.unsqueeze(-1).float()
                output = output + out_m

        return output, new_past_key_value


class ModalitySpecificFFN(nn.Module):
    """
    FFN-MMoE: Modality-specific feed-forward network.
    
    Each modality has its own gate, up, and down projection matrices.
    Uses SiLU (SwiGLU) activation.
    
    FFN-MMoE(x_{i,m}) = (SiLU(x_{i,m} W_gate^m) ⊙ x_{i,m} W_up^m) W_down^m
    
    The number of activated experts is 1 (each token uses exactly one expert
    based on its modality), maintaining consistent inference costs.
    """

    def __init__(self, config: MMoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Modality-specific FFN weights: W_gate^m, W_up^m, W_down^m
        self.gate_projs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            for _ in range(config.num_modalities)
        ])
        self.up_projs = nn.ModuleList([
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            for _ in range(config.num_modalities)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
            for _ in range(config.num_modalities)
        ])

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, hidden_size)
            modality_ids: (B, seq_len) - 0 for visual, 1 for linguistic
        Returns:
            output: (B, seq_len, hidden_size)
        """
        B, seq_len, _ = x.shape
        output = torch.zeros_like(x)

        for m in range(len(self.gate_projs)):
            mask = (modality_ids == m)  # (B, seq_len)
            if not mask.any():
                continue

            # Apply modality-specific FFN
            # FFN-MMoE(x_{i,m}) = (SiLU(x W_gate^m) ⊙ x W_up^m) W_down^m
            gate = F.silu(self.gate_projs[m](x))  # (B, seq_len, intermediate_size)
            up = self.up_projs[m](x)               # (B, seq_len, intermediate_size)
            hidden = gate * up                      # element-wise product
            out_m = self.down_projs[m](hidden)      # (B, seq_len, hidden_size)

            # Apply mask: only update tokens of this modality
            output = output + out_m * mask.unsqueeze(-1).float()

        return output


class ModalitySpecificMoE(nn.Module):
    """
    Full Modality-specific MoE layer combining MHA-MMoE and FFN-MMoE.
    
    This is the core building block of the MoE-extended LLM in NaViL.
    
    From the paper:
      x_{i,m}^{l'} = x_{i,m}^{l-1} + MHA-MMoE(RMSNorm(x_{i,m}^{l-1}))
      x_{i,m}^l   = x_{i,m}^{l'} + FFN-MMoE(RMSNorm(x_{i,m}^{l'}))
    """

    def __init__(self, config: MMoEConfig):
        super().__init__()
        self.config = config

        self.norm1 = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = ModalitySpecificAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn = ModalitySpecificFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_fn=None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (B, seq_len, hidden_size)
            modality_ids: (B, seq_len) - 0 for visual, 1 for linguistic
            attention_mask: optional attention mask
            position_ids: for RoPE
            past_key_value: KV cache
            rope_fn: RoPE function
        Returns:
            output: (B, seq_len, hidden_size)
            new_past_key_value
        """
        # MHA-MMoE with residual
        residual = x
        x_norm = self.norm1(x)
        attn_out, new_pkv = self.attn(
            x_norm,
            modality_ids=modality_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rope_fn=rope_fn,
        )
        x = residual + attn_out

        # FFN-MMoE with residual
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm, modality_ids=modality_ids)
        x = residual + ffn_out

        return x, new_pkv
