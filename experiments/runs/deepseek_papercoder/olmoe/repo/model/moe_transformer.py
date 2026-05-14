# model/moe_transformer.py
"""
OLMoE‑1B‑7B decoder-only transformer with dropless Mixture‑of‑Experts layers.

This module implements the complete model architecture described in the
OLMoE paper (Sections 2, 4; Appendix B), including:
  • RMSNorm (parametric) for pre‑attention, pre‑MoE, and QK‑Norm.
  • Rotary Position Embeddings (RoPE) with θ = 10,000.
  • 16 transformer blocks, each containing multi‑head self‑attention with
    optional QK‑Norm, followed by a dropless token‑choice MoE layer
    (64 small SwiGLU experts, top‑k=8).
  • Final RMSNorm and an un‑tied linear output head.

The forward pass returns both the logits and the raw router logits from every
MoE layer to allow auxiliary loss (load balancing, router z‑loss) computation.
All linear layers are bias‑free and initialised with a truncated normal
distribution (µ=0, σ=0.02, truncated at ±0.06) as described in the paper.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# The MoE layer is implemented in a separate module.
from model.moe_layer import MoELayer


# ----------------------------------------------------------------------
# RMSNorm – lightweight implementation following LLaMA / OLMoE
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation.

    Args:
        dim:         Feature dimension to normalise.
        eps:         Small constant for numerical stability.
        bias:        Whether to include a bias term (paper uses False).
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = x * rms * self.weight
        if self.bias is not None:
            output += self.bias
        return output


# ----------------------------------------------------------------------
# Rotary Position Embedding (RoPE) – pre‑computed cos / sin caching
# ----------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding with a configurable base frequency.

    Args:
        dim:                    Head dimension (must be even).
        max_position_embeddings:Maximum sequence length to pre‑compute.
        theta:                  Base frequency (default 10000).
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 4096,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even.")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._max_seq_len_cached = max_position_embeddings
        t = torch.arange(self._max_seq_len_cached, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate to form full head_dim: [cos, sin] each of size dim//2
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
            k: Key tensor of shape (batch, num_heads, seq_len, head_dim).

        Returns:
            Tuple of (q_rotated, k_rotated).
        """
        seq_len = q.shape[-2]
        cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates the first half of the hidden dimensions with the second."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


# ----------------------------------------------------------------------
# Transformer Block – Attention + MoE
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """
    A single transformer block containing:
        - Pre‑attention RMSNorm
        - Multi‑head self‑attention (with optional QK‑Norm and RoPE)
        - Residual connection
        - Pre‑MoE RMSNorm
        - Dropless token‑choice MoE layer
        - Residual connection

    Returns the hidden states and the MoE router logits for this layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rms_norm_eps: float,
        qk_norm: bool,
        rope_theta: float,
        max_seq_length: int,
        init_std: float,
        init_truncation: int,
        moe_config: Dict,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qk_norm_enabled = qk_norm

        # -------- Attention ---------
        self.attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # QK‑Norm: RMSNorm applied per head
        if self.qk_norm_enabled:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # Rotary embeddings (applied to q and k after QK‑Norm)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_seq_length,
            theta=rope_theta,
        )

        # -------- MoE ---------
        self.moe_norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Build config dict for MoELayer
        moe_cfg_for_moe = {
            "hidden_size": hidden_size,
            "num_experts": moe_config["num_experts"],
            "top_k": moe_config["top_k"],
            "expert_ffn_size": moe_config["expert_ffn_size"],
            "init_std": init_std,
            "init_truncation": init_truncation,
        }
        self.moe = MoELayer(config=moe_cfg_for_moe, use_megablocks=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ---- Attention ----
        residual = x
        x_norm = self.attn_norm(x)
        batch_size, seq_len, _ = x_norm.shape

        # QKV projections and head reshaping
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # QK‑Norm (per‑head RMSNorm)
        if self.qk_norm_enabled:
            # Reshape to (batch*num_heads, seq_len, head_dim) for RMSNorm
            q = q.reshape(-1, seq_len, self.head_dim)
            q = self.q_norm(q)
            q = q.view(batch_size, self.num_heads, seq_len, self.head_dim)

            k = k.reshape(-1, seq_len, self.head_dim)
            k = self.k_norm(k)
            k = k.view(batch_size, self.num_heads, seq_len, self.head_dim)

        # Rotary Position Embeddings
        q, k = self.rotary_emb(q, k)

        # Attention with causal mask
        if attention_mask is not None:
            # Build combined causal + padding mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            # attention_mask is (batch, seq_len) with True for valid tokens
            padding_mask = ~attention_mask[:, None, None, :]
            # Combine: mask out both future and padding tokens
            attn_bias = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_bias = attn_bias | padding_mask
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        # Reshape back and apply output projection
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        attn_output = self.o_proj(attn_output)
        x = residual + attn_output

        # ---- MoE ----
        residual = x
        x_norm = self.moe_norm(x)
        moe_output, router_logits = self.moe(x_norm)
        x = residual + moe_output

        return x, router_logits


# ----------------------------------------------------------------------
# MoETransformer – the full model
# ----------------------------------------------------------------------
class MoETransformer(nn.Module):
    """
    OLMoE‑1B‑7B Mixture‑of‑Experts language model.

    Arguments:
        config: Dictionary containing all model hyperparameters (see config.yaml).

    Forward returns:
        logits:          (batch_size, seq_len, vocab_size)
        router_logits:   List of tensors, each (batch_size*seq_len, num_experts)
                         one per MoE layer.
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()

        # --- Extract config values with defaults ---
        self.hidden_size: int = config["hidden_size"]
        self.num_layers: int = config["num_layers"]
        self.vocab_size: int = config["vocab_size"]
        self.max_seq_length: int = config["max_sequence_length"]
        self.rms_norm_eps: float = config["rms_norm_eps"]
        self.qk_norm: bool = config["qk_norm"]
        self.rope_theta: float = config["rope_theta"]
        self.init_std: float = config.get("init_std", 0.02)
        self.init_truncation: int = config.get("init_truncation", 3)
        num_heads: int = config["num_attention_heads"]

        moe_config: Dict = config["moe"]

        # --- Token embeddings (no weight tying) ---
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # --- Transformer blocks ---
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=self.hidden_size,
                    num_heads=num_heads,
                    rms_norm_eps=self.rms_norm_eps,
                    qk_norm=self.qk_norm,
                    rope_theta=self.rope_theta,
                    max_seq_length=self.max_seq_length,
                    init_std=self.init_std,
                    init_truncation=self.init_truncation,
                    moe_config=moe_config,
                )
                for _ in range(self.num_layers)
            ]
        )

        # --- Final norm and output projection ---
        self.final_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # --- Weight initialization ---
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Apply truncated normal initialisation to all linear and embedding
        weights, and ones to RMSNorm scales.
        """
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=self.init_std,
                a=-self.init_truncation * self.init_std,
                b=self.init_truncation * self.init_std,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=self.init_std,
                a=-self.init_truncation * self.init_std,
                b=self.init_truncation * self.init_std,
            )

        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            input_ids:      Token indices (batch_size, seq_len).
            attention_mask: Optional mask with True for valid tokens,
                            shape (batch_size, seq_len).

        Returns:
            logits:         Next‑token prediction logits.
            all_router_logits: List of raw router logits from each MoE layer.
        """
        # Embed and scale (common in many Transformer implementations)
        x = self.embed_tokens(input_ids) * math.sqrt(self.hidden_size)

        all_router_logits: List[torch.Tensor] = []
        for layer in self.layers:
            x, router_logits = layer(x, attention_mask)
            all_router_logits.append(router_logits)

        x = self.final_norm(x)
        logits = self.output_proj(x)

        return logits, all_router_logits


# ----------------------------------------------------------------------
# Quick sanity test (executed only when the module is run directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import yaml

    # Load a minimal config (relying on defaults for missing keys is dangerous)
    with open("config.yaml", "r") as f:
        full_cfg = yaml.safe_load(f)
    model_cfg = full_cfg["model"]
    # Ensure init fields are present
    model_cfg.setdefault("init_std", 0.02)
    model_cfg.setdefault("init_truncation", 3)
    model_cfg.setdefault("rope_theta", 10000.0)

    model = MoETransformer(model_cfg)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy forward
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, model_cfg["vocab_size"], (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    logits, router_logits = model(input_ids, attention_mask)
    print(f"Logits shape: {logits.shape}")
    print(f"Number of MoE layers: {len(router_logits)}")
    print(f"Router logits shape (layer 0): {router_logits[0].shape}")
