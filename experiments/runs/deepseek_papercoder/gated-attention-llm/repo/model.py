## model.py
"""
Core transformer model components for reproducing the Gated Attention LLM architecture.

This module implements:
    - RMSNorm (root mean square layer normalization)
    - SwiGLU (feed‑forward network with SiLU gating)
    - TransformerBlock (a single decoder layer with optional sandwich normalization)
    - GPTModel (full decoder‑only language model with embedding, transformer layers, and LM head)

The design strictly follows the paper "Gated Attention for Large Language Models" and relies on
the `GatedAttention` module (from `gated_attention.py`) for all attention and gating variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# The GatedAttention module is expected to handle all gating logic, RoPE, and SDPA internally.
# It is imported here from the companion file.
from gated_attention import GatedAttention


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Args:
        hidden_size: Model dimension (d_model).
        eps: Small constant for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to the last dimension of the input tensor.

        Args:
            x: Tensor of shape (..., hidden_size).

        Returns:
            Normalized tensor of the same shape.
        """
        ms = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(ms + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):
    """
    SwiGLU feed‑forward network (Shazeer, 2020).

    Args:
        hidden_size: Input and output dimension.
        intermediate_size: Hidden dimension of the up‑projection and gating.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)  # gate
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)  # up
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: SwiGLU(x) = (xW₂ ⊙ SiLU(xW₁)) W₃

        Args:
            x: Tensor of shape (batch, seq_len, hidden_size).

        Returns:
            Tensor of shape (batch, seq_len, hidden_size).
        """
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """
    Single decoder block of a pre‑norm transformer, optionally with
    sandwich normalization (Ding et al., 2021) for training stability.

    Args:
        config: A dictionary with model‑specific hyperparameters (the 'model' sub‑dict
                from the global configuration). Must contain at least:
                - hidden_size: int
                - intermediate_size: int (already adjusted if gating is enabled)
                - rms_norm_eps: float
                - use_sandwich_norm: bool
                - (and all keys required by GatedAttention)
    """

    def __init__(self, config: Dict):
        super().__init__()
        hidden_size = config["hidden_size"]
        eps = config.get("rms_norm_eps", 1e-5)

        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.attn = GatedAttention(config)                   # handles attention + optional gating
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = SwiGLU(hidden_size, config["intermediate_size"])

        # Optional sandwich normalization (Table 2 row 7, Appendix A.5)
        self.use_sandwich = config.get("use_sandwich_norm", False)
        if self.use_sandwich:
            self.sandwich_attn_norm = RMSNorm(hidden_size, eps=eps)
            self.sandwich_ffn_norm = RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder block.

        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_size).
            attention_mask: Optional attention mask (passed directly to GatedAttention).
                            If None, a causal mask is applied internally.

        Returns:
            Output tensor of the same shape as hidden_states.
        """
        # Pre‑norm + attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask=attention_mask)

        if self.use_sandwich:
            attn_output = self.sandwich_attn_norm(attn_output)

        hidden_states = residual + attn_output

        # Pre‑norm + FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_output = self.mlp(hidden_states)

        if self.use_sandwich:
            ffn_output = self.sandwich_ffn_norm(ffn_output)

        hidden_states = residual + ffn_output
        return hidden_states


class GPTModel(nn.Module):
    """
    Full decoder‑only transformer model for causal language modelling.

    Args:
        config: A dictionary with model‑specific hyperparameters (the 'model' sub‑dict
                from the global configuration). Must contain at least:
                - num_layers: int
                - hidden_size: int
                - vocab_size: int
                - intermediate_size: int (already adjusted for gating if needed)
                - rms_norm_eps: float
                - (and all keys required by GatedAttention)
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Stack of transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(num_layers)]
        )

        # Final norm before language model head
        eps = config.get("rms_norm_eps", 1e-5)
        self.norm = RMSNorm(hidden_size, eps=eps)

        # LM head (no bias, optionally weight‑tied with embedding)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Weight tying (common practice, default True)
        if config.get("tie_word_embeddings", True):
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Default initialisation for linear layers, embeddings, and RMSNorm weights.

        Args:
            module: A sub‑module of the model.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the language model.

        Args:
            input_ids: Long tensor of shape (batch, seq_len).
            attention_mask: Optional boolean tensor of shape (batch, seq_len); True means
                            "attend to this position". This is passed to all attention layers.
                            If None, a causal mask is applied internally by each GatedAttention.
            labels: Optional long tensor of shape (batch, seq_len) for language modelling loss.
                    If provided, the loss is computed as the cross‑entropy of next‑token
                    predictions.

        Returns:
            A tuple (logits, loss) where:
                - logits: tensor of shape (batch, seq_len, vocab_size)
                - loss: scalar tensor if labels is provided, else None
        """
        B, T = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)  # (B, T, hidden_size)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)          # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Shift logits and labels for next‑token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,                        # typical ignore index for padding
            )

        return logits, loss

