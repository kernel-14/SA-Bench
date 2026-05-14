"""
Transformer model with gated attention.

Implements a standard pre-norm transformer decoder with:
  - Gated multi-head attention (GQA supported)
  - SwiGLU FFN
  - RMSNorm
  - RoPE positional embeddings

This matches the architecture used in the paper's experiments (Qwen2.5-style).
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gated_attention import (
    GatedMultiHeadAttention,
    GatingActivation,
    GatingGranularity,
    GatingPosition,
    GatingType,
)


@dataclass
class TransformerConfig:
    """Configuration for the gated transformer model.
    
    Default values approximate the 1.7B dense model from the paper.
    """
    # Model dimensions
    d_model: int = 2048
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 8  # GQA
    head_dim: int = 128
    ffn_intermediate_dim: int = 11008  # SwiGLU intermediate
    vocab_size: int = 151936  # Qwen2.5 tokenizer
    max_seq_len: int = 4096
    
    # Regularization
    dropout: float = 0.0
    
    # RoPE
    rope_base: float = 10000.0
    
    # Gating configuration
    # Set gating_position=None for baseline (no gating)
    gating_position: Optional[str] = "sdpa_output"  # G1 by default
    gating_granularity: str = "elementwise"
    head_specific: bool = True
    gating_type: str = "multiplicative"
    gating_activation: str = "sigmoid"
    
    # Whether to use sandwich norm (for training stability ablation)
    use_sandwich_norm: bool = False
    
    def get_gating_position(self) -> Optional[GatingPosition]:
        if self.gating_position is None:
            return None
        mapping = {
            "sdpa_output": GatingPosition.G1_SDPA_OUTPUT,
            "value": GatingPosition.G2_VALUE,
            "key": GatingPosition.G3_KEY,
            "query": GatingPosition.G4_QUERY,
            "dense_output": GatingPosition.G5_DENSE_OUTPUT,
        }
        return mapping[self.gating_position]
    
    def get_gating_granularity(self) -> GatingGranularity:
        return GatingGranularity(self.gating_granularity)
    
    def get_gating_type(self) -> GatingType:
        return GatingType(self.gating_type)
    
    def get_gating_activation(self) -> GatingActivation:
        return GatingActivation(self.gating_activation)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.
    
    FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    
    Used as the standard FFN in the paper's models.
    """
    
    def __init__(self, d_model: int, intermediate_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        if self.dropout > 0.0 and self.training:
            hidden = F.dropout(hidden, p=self.dropout)
        return self.down_proj(hidden)


class GatedTransformerLayer(nn.Module):
    """Single transformer decoder layer with gated attention.
    
    Pre-norm architecture:
      x = x + Attention(RMSNorm(x))
      x = x + FFN(RMSNorm(x))
    
    With optional sandwich norm (Ding et al., 2021):
      x = x + RMSNorm(Attention(RMSNorm(x)))
      x = x + RMSNorm(FFN(RMSNorm(x)))
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Pre-norm
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        
        # Sandwich norm (optional, for training stability)
        self.use_sandwich_norm = config.use_sandwich_norm
        if config.use_sandwich_norm:
            self.attn_post_norm = RMSNorm(config.d_model)
            self.ffn_post_norm = RMSNorm(config.d_model)
        
        # Gated attention
        self.attn = GatedMultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dropout=config.dropout,
            gating_position=config.get_gating_position(),
            gating_granularity=config.get_gating_granularity(),
            head_specific=config.head_specific,
            gating_type=config.get_gating_type(),
            gating_activation=config.get_gating_activation(),
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base,
        )
        
        # FFN
        self.ffn = SwiGLUFFN(
            d_model=config.d_model,
            intermediate_dim=config.ffn_intermediate_dim,
            dropout=config.dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch, seq, d_model)
            attention_mask: Optional attention mask
            past_key_value: Optional KV cache
            use_cache: Whether to return KV cache
        
        Returns:
            x: Updated hidden states (batch, seq, d_model)
            past_key_value: Updated KV cache if use_cache=True
        """
        # Attention with pre-norm
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, new_past_kv = self.attn(
            x_norm,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if self.use_sandwich_norm:
            attn_out = self.attn_post_norm(attn_out)
        x = residual + attn_out
        
        # FFN with pre-norm
        residual = x
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        if self.use_sandwich_norm:
            ffn_out = self.ffn_post_norm(ffn_out)
        x = residual + ffn_out
        
        return x, new_past_kv


class GatedTransformerModel(nn.Module):
    """Full transformer decoder model with gated attention.
    
    Architecture:
      - Token embedding
      - N transformer layers with gated attention
      - Final RMSNorm
      - LM head (tied with embedding)
    
    This implements the 1.7B dense model architecture from the paper.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GatedTransformerLayer(config)
            for _ in range(config.num_layers)
        ])
        
        # Final norm
        self.norm = RMSNorm(config.d_model)
        
        # LM head (tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # weight tying
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            input_ids: (batch, seq)
            attention_mask: Optional (batch, seq) or (batch, 1, seq, seq)
            past_key_values: Optional list of KV caches per layer
            use_cache: Whether to return KV caches
            labels: Optional (batch, seq) for computing loss
        
        Returns:
            dict with keys: logits, loss (if labels provided), past_key_values
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embed_tokens(input_ids)
        
        # Process through transformer layers
        new_past_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_past_kv = layer(
                x,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(new_past_kv)
        
        # Final norm
        x = self.norm(x)
        
        # LM head
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": new_past_key_values,
        }
    
    def get_num_params(self) -> dict:
        """Return parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        gate_params = sum(
            layer.attn.get_num_gate_params()
            for layer in self.layers
        )
        return {
            "total": total,
            "gate_params": gate_params,
            "non_gate_params": total - gate_params,
        }
