
"""Full GPT and nGPT model implementations.

GPT: Standard decoder-only Transformer with RMSNorm and residual connections.
nGPT: Normalized Transformer with hypersphere representation learning.

Both models are autoregressive with causal masking.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from modules import (
    NormalizedEmbedding,
    ScaledParameter,
    GPTBlock,
    nGPTBlock,
)


class GPT(nn.Module):
    """Standard GPT decoder-only Transformer with SwiGLU and RoPE.

    Architecture matches the baseline described in the paper (Section 2.2.1, 2.3.1, 2.4.1):
    - Token embeddings (learned, unconstrained)
    - RoPE applied to Q and K in each attention layer
    - L layers of GPTBlock (RMSNorm + ATTN + RMSNorm + SwiGLU MLP + residual)
    - Final RMSNorm
    - Output projection (tied with input embeddings)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_mlp: int,
        max_seq_len: int = 4096,
        rope_base: int = 10000,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_mlp, max_seq_len, rope_base)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        for layer in self.layers:
            nn.init.normal_(
                layer.attn.W_o.weight,
                std=0.02 / math.sqrt(2 * self.n_layers)
            )
            nn.init.normal_(
                layer.mlp.W_o.weight,
                std=0.02 / math.sqrt(2 * self.n_layers)
            )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        h = self.token_embed(idx)  # (B, T, d_model)

        # Causal mask: (1, 1, T, T) for broadcasting over batch and heads
        causal_mask = torch.triu(
            torch.ones(1, 1, T, T, device=idx.device) * float('-inf'), diagonal=1
        )

        for layer in self.layers:
            h = layer(h, causal_mask)

        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits


class nGPT(nn.Module):
    """Normalized GPT with representation learning on the hypersphere.

    Key characteristics:
    - All embeddings and weight matrices are L2-normalized along embedding dimension
    - No RMSNorm/LayerNorm anywhere
    - Hidden state updates via LERP (or SLERP) with per-dimension eigen learning rates
    - QK normalization and scaling
    - Logit scaling by trainable s_z
    - No weight decay needed
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_mlp: int,
        max_seq_len: int = 4096,
        rope_base: int = 10000,
        use_qk_norm: bool = True,
        alpha_A_init: float = 0.05,
        alpha_A_scale: float = 1.0,
        alpha_M_init: float = 0.05,
        alpha_M_scale: float = 1.0,
        s_qk_init: float = 1.0,
        s_qk_scale: float = 1.0,
        s_u_init: float = 1.0,
        s_u_scale: float = 1.0,
        s_v_init: float = 1.0,
        s_v_scale: float = 1.0,
        s_z_init: float = 1.0,
        s_z_scale: float = 1.0,
        use_lerp: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Normalized input and output embeddings (Section 2.1)
        self.E_input = NormalizedEmbedding(vocab_size, d_model)
        self.E_output = NormalizedEmbedding(vocab_size, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            nGPTBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_mlp=d_mlp,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                use_qk_norm=use_qk_norm,
                alpha_A_init=alpha_A_init,
                alpha_A_scale=alpha_A_scale,
                alpha_M_init=alpha_M_init,
                alpha_M_scale=alpha_M_scale,
                s_qk_init=s_qk_init,
                s_qk_scale=s_qk_scale,
                s_u_init=s_u_init,
                s_u_scale=s_u_scale,
                s_v_init=s_v_init,
                s_v_scale=s_v_scale,
                use_lerp=use_lerp,
            )
            for _ in range(n_layers)
        ])

        # Logit scaling factor (Section 2.1, eq 3)
        self.s_z = ScaledParameter(vocab_size, s_z_init, s_z_scale)

        self._init_weights()

    def _init_weights(self):
        """Initialize all parameters. For nGPT, matrices are normalized afterward,
        so exact initialization is less critical (Section A.6).
        """
        pass  # NormalizedLinear/Embedding handle their own initialization

    def normalize_all_weights(self):
        """Normalize all weight matrices and embeddings (Section 2.6, step 2).

        Called after each training step to keep all vectors on the hypersphere.
        """
        self.E_input.normalize_weights()
        self.E_output.normalize_weights()
        for layer in self.layers:
            layer.normalize_weights()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape

        # Input embeddings (already normalized by NormalizedEmbedding)
        h = self.E_input(idx)  # (B, T, d_model)

        # Causal mask: (1, 1, T, T) for broadcasting
        causal_mask = torch.triu(
            torch.ones(1, 1, T, T, device=idx.device) * float('-inf'), diagonal=1
        )

        # Transformer layers
        for layer in self.layers:
            h = layer(h, causal_mask)

        # Output logits: z_i = E_output @ h_i (eq 1), then scaled element-wise by s_z (eq 3)
        # Both h and E_output rows are normalized → dot products in [-1, 1]
        E_out_norm = F.normalize(self.E_output.weight, p=2, dim=1, eps=1e-12)
        logits = h @ E_out_norm.T  # (B, T, vocab)
        logits = logits * self.s_z()

        return logits


def create_model(config, use_ngpt: bool = True):
    """Factory function to create GPT or nGPT model from config."""
    if use_ngpt:
        ngpt_cfg = config.ngpt
        return nGPT(
            vocab_size=config.model.vocab_size,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            d_mlp=config.model.d_mlp,
            max_seq_len=config.model.max_seq_len,
            rope_base=config.model.rope_base,
            use_qk_norm=ngpt_cfg.qk_norm,
            alpha_A_init=ngpt_cfg.alpha_A_init,
            alpha_A_scale=ngpt_cfg.alpha_A_scale,
            alpha_M_init=ngpt_cfg.alpha_M_init,
            alpha_M_scale=ngpt_cfg.alpha_M_scale,
            s_qk_init=ngpt_cfg.s_qk_init,
            s_qk_scale=ngpt_cfg.s_qk_scale,
            s_u_init=ngpt_cfg.s_u_init,
            s_u_scale=ngpt_cfg.s_u_scale,
            s_v_init=ngpt_cfg.s_v_init,
            s_v_scale=ngpt_cfg.s_v_scale,
            s_z_init=ngpt_cfg.s_z_init,
            s_z_scale=ngpt_cfg.s_z_scale,
            use_lerp=ngpt_cfg.use_lerp,
        )
    else:
        return GPT(
            vocab_size=config.model.vocab_size,
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            d_mlp=config.model.d_mlp,
            max_seq_len=config.model.max_seq_len,
        )
