"""
Full transformer model supporting both dense and MoE architectures
with configurable gated attention variants as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from config import ModelConfig
from gated_attention import (
    build_gated_attention,
    RMSNorm,
    GatedMLP,
)
from moe import MoETransformerLayer, MoELayer


class DenseTransformerLayer(nn.Module):
    """
    Single transformer layer for dense models with gated attention.
    """
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int,
        norm_eps: float = 1e-6,
        use_sandwich_norm: bool = False,
    ):
        super().__init__()
        self.attention = attention
        self.ffn = GatedMLP(d_model, d_ff)

        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.norm2 = RMSNorm(d_model, eps=norm_eps)
        self.use_sandwich_norm = use_sandwich_norm
        if use_sandwich_norm:
            self.norm_attn = RMSNorm(d_model, eps=norm_eps)
            self.norm_ffn = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        if self.use_sandwich_norm:
            residual = x
            x_norm = self.norm1(x)
            attn_out = self.attention(x_norm, attention_mask)
            attn_out = self.norm_attn(attn_out)
            x = residual + attn_out

            residual = x
            x_norm = self.norm2(x)
            ffn_out = self.ffn(x_norm)
            ffn_out = self.norm_ffn(ffn_out)
            x = residual + ffn_out
        else:
            x = x + self.attention(self.norm1(x), attention_mask)
            x = x + self.ffn(self.norm2(x))

        return x


class Transformer(nn.Module):
    """
    Full transformer model supporting both dense and MoE architectures,
    with configurable gated attention and training stabilizers.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Build layers
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            attn = build_gated_attention(config)

            if config.model_type == "moe":
                expert_intermediate = (
                    config.expert_intermediate_dim
                    if config.expert_intermediate_dim
                    else config.d_ff // 4
                )
                layer = MoETransformerLayer(
                    attention=attn,
                    d_model=config.d_model,
                    n_experts=config.n_experts,
                    n_active=config.n_active_experts,
                    expert_intermediate_dim=expert_intermediate,
                    norm_eps=config.rms_norm_eps,
                    use_sandwich_norm=config.use_sandwich_norm,
                )
            else:
                layer = DenseTransformerLayer(
                    attention=attn,
                    d_model=config.d_model,
                    d_ff=config.d_ff,
                    norm_eps=config.rms_norm_eps,
                    use_sandwich_norm=config.use_sandwich_norm,
                )
            self.layers.append(layer)

        # Final norm
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: embed and lm_head share weights
        self.lm_head.weight = self.embed_tokens.weight

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: optional padding mask (batch, seq_len)
            labels: optional target labels for loss computation

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar if labels provided, else None
        """
        B, S = input_ids.shape

        x = self.embed_tokens(input_ids)

        # Create causal mask
        causal_mask = self._create_causal_mask(S, x.device)

        # Combine with padding mask if provided
        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :] == 0
            pad_mask = pad_mask.to(device=x.device, dtype=x.dtype)
            pad_mask = pad_mask * float("-inf")
            causal_mask = causal_mask + pad_mask

        total_aux_loss = 0.0

        for layer in self.layers:
            if self.model_type == "moe":
                x, aux_loss = layer(x, causal_mask)
                total_aux_loss = total_aux_loss + aux_loss
            else:
                x = layer(x, causal_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            if self.model_type == "moe":
                loss = loss + total_aux_loss

        return logits, loss


def get_parameter_count(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_model(config: ModelConfig) -> Transformer:
    """Create a transformer model from config."""
    return Transformer(config)
