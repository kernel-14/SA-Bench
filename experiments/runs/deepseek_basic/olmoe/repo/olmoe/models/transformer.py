"""OLMoE Transformer model.

Full decoder-only transformer with MoE layers, RoPE, QK-Norm, and RMSNorm.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import OLMoEMoE
from .configuration import OLMoEConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Used in OLMoE-1B-7B instead of non-parametric LayerNorm.
    Includes learnable affine parameters that ARE weight-decayed.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2023).

    OLMoE uses RoPE with theta=10000.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency bands
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("freqs", freqs)

        # Precompute full position embeddings
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.freqs)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(
        self, x: torch.Tensor, seq_len: int, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos, sin for positions [offset, offset + seq_len]."""
        cos = self.cos_cached[offset:offset + seq_len].to(x.device)
        sin = self.sin_cached[offset:offset + seq_len].to(x.device)
        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    # q, k: (batch, num_heads, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Split into two halves for rotation
    q_rot = q.float() * cos + rotate_half(q.float()) * sin
    k_rot = k.float() * cos + rotate_half(k.float()) * sin

    return q_rot.to(q.dtype), k_rot.to(k.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last half of the tensor dimensions."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention with QK-Norm.

    OLMoE uses full attention (not MQA/GQA) with QK-Norm for stability.
    """

    def __init__(self, config: OLMoEConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.hidden_size = config.d_model

        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)

        # QK-Norm (for stability, as per §4.2.5)
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.layer_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.layer_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Rotary embedding
        self.rotary = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        offset = position_ids[0, 0].item() if position_ids is not None else 0
        cos, sin = self.rotary(x, seq_len, offset)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    """Single transformer block with MoE FFN (or dense FFN)."""

    def __init__(self, config: OLMoEConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-attention RMSNorm
        self.attn_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)

        # Self-attention
        self.attention = Attention(config)

        # Pre-FFN RMSNorm
        self.ffn_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)

        # Decide whether this layer uses MoE
        use_moe = config.moe_layers == "every"  # All layers use MoE

        if use_moe:
            self.moe = OLMoEMoE(
                hidden_size=config.d_model,
                ffn_dim=config.moe_ffn_dim,
                num_experts=config.moe_num_experts,
                num_activated=config.moe_num_activated,
                dropout=config.dropout,
                lb_loss_weight=config.load_balancing_loss_weight,
                rz_loss_weight=config.router_z_loss_weight,
            )
        else:
            # Dense FFN (not used in OLMoE-1B-7B but kept for flexibility)
            self.moe = None
            from .moe import SwiGLU, Expert
            self.ffn = Expert(
                config.d_model,
                config.d_model * 4,  # Standard expansion
                SwiGLU(),
                config.dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-norm + attention
        residual = x
        x = self.attn_norm(x)
        x = self.attention(x, attention_mask, position_ids)
        x = residual + x

        # Pre-norm + FFN/MoE
        residual = x
        x = self.ffn_norm(x)

        aux_loss = None
        if self.moe is not None:
            x, aux_loss = self.moe(x)
        else:
            x = self.ffn(x)

        x = residual + x

        return x, aux_loss


class OLMoEModel(nn.Module):
    """OLMoE decoder-only transformer model.

    This is the full OLMoE-1B-7B model with:
    - 16 transformer layers
    - MoE in every layer (64 experts, 8 activated)
    - RMSNorm, QK-Norm, RoPE
    - Truncated normal initialization
    """

    def __init__(self, config: OLMoEConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final output norm
        self.output_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)

        # Output projection (no weight tying in OLMoE-1B-7B)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights with truncated normal distribution.

        As per §4.2.2: truncated normal init with std=0.02, truncation at ±3*std.
        """
        if isinstance(module, nn.Linear):
            if self.config.init_dist == "truncated_normal":
                # Truncated normal: clip at ±init_trunc * std
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=self.config.init_std,
                    a=-self.config.init_trunc * self.config.init_std,
                    b=self.config.init_trunc * self.config.init_std,
                )
            else:
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> dict:
        """Forward pass of OLMoE.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Optional causal mask
            position_ids: Optional position IDs
            labels: Optional labels for computing cross-entropy loss

        Returns:
            dict with keys: logits, ce_loss, aux_loss, total_loss
        """
        bsz, seq_len = input_ids.shape

        # Create causal attention mask if not provided
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
                diagonal=1,
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Get position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Token embeddings
        x = self.token_embeddings(input_ids)

        # Pass through transformer layers
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, aux_loss = layer(x, attention_mask, position_ids)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

        # Output norm and LM head
        x = self.output_norm(x)
        logits = self.lm_head(x)

        # Compute cross-entropy loss if labels provided
        ce_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        # Total loss = CE + α*L_LB + β*L_RZ (Equation 2)
        total_loss = None
        if ce_loss is not None:
            total_loss = ce_loss + total_aux_loss

        return {
            "logits": logits,
            "ce_loss": ce_loss,
            "aux_loss": total_aux_loss,
            "total_loss": total_loss,
        }

    def get_router_logits(
        self, x: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Get router logits for a specific layer (for analysis)."""
        layer = self.layers[layer_idx]
        if layer.moe is not None:
            x_norm = layer.ffn_norm(x)
            router_logits = layer.moe.router(x_norm)
            return router_logits
        raise ValueError(f"Layer {layer_idx} is not an MoE layer")

    def get_top_k_experts(
        self, x: torch.Tensor, layer_idx: int, k: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k expert indices for analysis (routing visualization).

        Returns:
            top_k_indices: (batch*seq, k) expert indices
            top_k_probs: (batch*seq, k) routing probabilities
        """
        router_logits = self.get_router_logits(x, layer_idx)
        router_probs = F.softmax(router_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        return top_k_indices, top_k_probs

    @property
    def num_active_params(self) -> int:
        """Count active parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def create_olmoe_model() -> OLMoEModel:
    """Create the OLMoE-1B-7B model with default configuration.

    Returns:
        OLMoEModel instance configured as OLMoE-1B-7B
    """
    config = OLMoEConfig()
    model = OLMoEModel(config)
    return model
