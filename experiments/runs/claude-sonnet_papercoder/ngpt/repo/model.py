## Code: model.py

```python
## model.py
"""GPT and nGPT decoder-only Transformer implementations.

This module provides both the baseline GPT model and the normalized nGPT model
as described in "nGPT: Normalized Transformer with Representation Learning on
the Hypersphere" (Loshchilov et al.).

Key classes:
    - RoPEEmbedding: Rotary Position Embeddings (shared by both models).
    - GPTAttention / GPTMLP / GPTLayer / GPTModel: Standard baseline.
    - nGPTAttention / nGPTMLP / nGPTLayer / nGPTModel: Normalized variant.

Architecture differences (paper Table 1):
    GPT:  h ← h + ATTN(RMSNorm(h))   |  nGPT: hA = Norm(ATTN(h))
          h ← h + MLP(RMSNorm(h))    |         h  = Norm(h + αA(hA − h))
          Final: h ← RMSNorm(h)      |         hM = Norm(MLP(h))
                                     |         h  = Norm(h + αM(hM − h))

All configuration values are sourced from Config (config.py), which is
populated from config.yaml. No values are hardcoded in this file.

Typical usage:
    from config import Config
    from model import GPTModel, nGPTModel

    config = Config.ngpt_500m(context_length=4096)
    model = nGPTModel(config)
    logits, loss = model(tokens, targets)
    model.normalize_all_weights()  # called after optimizer.step()
"""

import math
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import Config
from ngpt_components import HypersphereUpdate
from ngpt_components import NormEmbedding
from ngpt_components import NormLinear
from ngpt_components import ScaledParameter


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RoPEEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for query and key tensors.

    Implements the RoPE formulation from Su et al. (2024), "RoFormer: Enhanced
    Transformer with Rotary Position Embedding." Used by both GPTAttention and
    nGPTAttention.

    RoPE encodes absolute position information into the query and key vectors
    by rotating pairs of dimensions. The rotation angle for dimension pair i
    is theta_i * position, where theta_i = 1 / (base^(2i/d_k)).

    The cos/sin caches are precomputed up to max_seq_len and registered as
    non-trainable buffers so they move to the correct device with .to(device).

    Attributes:
        d_k: Per-head key/query dimension.
        max_seq_len: Maximum sequence length for precomputed cache.
        base: RoPE base frequency (config.yaml architecture.rope_base: 10000).
        cos_cache: Precomputed cosine values, shape (max_seq_len, d_k // 2).
        sin_cache: Precomputed sine values, shape (max_seq_len, d_k // 2).
    """

    def __init__(
        self,
        d_k: int,
        max_seq_len: int,
        base: int = 10000,
    ) -> None:
        """Initialize RoPEEmbedding and precompute cos/sin caches.

        Args:
            d_k: Per-head key/query dimension. Must be even.
            max_seq_len: Maximum sequence length. Cache is precomputed for
                positions [0, max_seq_len). For length extrapolation beyond
                this value, the cache will be extended dynamically.
            base: Base for the inverse frequency computation. Paper default
                is 10000 (config.yaml architecture.rope_base: 10000).

        Raises:
            ValueError: If d_k is odd (RoPE requires pairs of dimensions).
        """
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(
                f"d_k must be even for RoPE (requires dimension pairs), "
                f"got d_k={d_k}."
            )

        self.d_k: int = d_k
        self.max_seq_len: int = max_seq_len
        self.base: int = base

        # Precompute and register cos/sin caches as buffers
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Precompute and register cos/sin caches for positions [0, seq_len).

        Args:
            seq_len: Number of positions to precompute.
        """
        # Inverse frequencies: theta_i = 1 / (base^(2i/d_k))
        # Shape: (d_k // 2,)
        half_d: int = self.d_k // 2
        i_indices = torch.arange(0, half_d, dtype=torch.float32)
        inv_freq: Tensor = 1.0 / (
            self.base ** (i_indices * 2.0 / self.d_k)
        )  # shape: (d_k // 2,)

        # Position indices: [0, 1, ..., seq_len - 1]
        # Shape: (seq_len,)
        positions = torch.arange(0, seq_len, dtype=torch.float32)

        # Outer product: freqs[t, i] = positions[t] * inv_freq[i]
        # Shape: (seq_len, d_k // 2)
        freqs: Tensor = torch.outer(positions, inv_freq)

        # Precompute cos and sin
        # Shape: (seq_len, d_k // 2)
        cos_cache: Tensor = torch.cos(freqs)
        sin_cache: Tensor = torch.sin(freqs)

        # Register as buffers (not parameters — not trained, but move with .to())
        self.register_buffer("cos_cache", cos_cache, persistent=True)
        self.register_buffer("sin_cache", sin_cache, persistent=True)

        self.max_seq_len = seq_len

    def _extend_cache_if_needed(self, seq_len: int) -> None:
        """Extend the cos/sin cache if seq_len exceeds the current cache size.

        This supports length extrapolation beyond the training context length,
        as described in Appendix A.8 of the paper.

        Args:
            seq_len: Required sequence length.
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            # Move new buffers to the same device as existing ones
            device = self.cos_cache.device  # type: ignore[attr-defined]
            self.cos_cache = self.cos_cache.to(device)  # type: ignore[attr-defined]
            self.sin_cache = self.sin_cache.to(device)  # type: ignore[attr-defined]

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        """Apply rotary position embeddings to query or key tensor.

        Applies the RoPE rotation to each position in the sequence. The
        rotation for position t rotates dimension pairs (2i, 2i+1) by angle
        theta_i * t.

        Rotation formula for each dimension pair (x1, x2) at position t:
            x1_rot = x1 * cos(theta_i * t) - x2 * sin(theta_i * t)
            x2_rot = x1 * sin(theta_i * t) + x2 * cos(theta_i * t)

        Args:
            x: Query or key tensor of shape (batch, n_heads, seq_len, d_k).
            offset: Starting position offset for KV-cache inference. Defaults
                to 0 for training.

        Returns:
            Rotated tensor of the same shape as x.
        """
        # x shape: (batch, n_heads, seq_len, d_k)
        seq_len: int = x.shape[2]

        # Extend cache if needed (for length extrapolation)
        self._extend_cache_if_needed(offset + seq_len)

        # Slice the relevant positions from the cache
        # Shape: (seq_len, d_k // 2)
        cos: Tensor = self.cos_cache[offset : offset + seq_len]  # type: ignore[index]
        sin: Tensor = self.sin_cache[offset : offset + seq_len]  # type: ignore[index]

        # Reshape for broadcasting over (batch, n_heads, seq_len, d_k // 2)
        # cos/sin: (1, 1, seq_len, d_k // 2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split x into two halves along the last dimension
        # x1, x2: (batch, n_heads, seq_len, d_k // 2)
        half_d: int = self.d_k // 2
        x1: Tensor = x[..., :half_d]
        x2: Tensor = x[..., half_d:]

        # Apply rotation
        x1_rot: Tensor = x1 * cos - x2 * sin
        x2_rot: Tensor = x1 * sin + x2 * cos

        # Concatenate back along last dimension
        return torch.cat([x1_rot, x2_rot], dim=-1)


# ---------------------------------------------------------------------------
# Causal mask helper
# ---------------------------------------------------------------------------

def _build_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Build an additive causal attention mask.

    Creates a mask where future positions are set to -inf, preventing
    attention to future tokens. This implements the causal masking described
    in Section 2.3.1: M[i,j] = 0 if j <= i, -inf if j > i.

    Note: In practice, F.scaled_dot_product_attention with is_causal=True
    is more efficient and is used instead of this explicit mask. This function
    is provided for reference and fallback use.

    Args:
        seq_len: Sequence length.
        device: Target device for the mask tensor.

    Returns:
        Additive mask tensor of shape (seq_len, seq_len) with 0.0 for
        allowed positions and -inf for masked (future) positions.
    """
    # Upper triangular mask (excluding diagonal): True where j > i
    mask: Tensor = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )
    return mask


# ---------------------------------------------------------------------------
# GPT Baseline Components
# ---------------------------------------------------------------------------

class GPTAttention(nn.Module):
    """Standard multi-head self-attention for the GPT baseline.

    Implements the attention block from Section 2.3.1:
        h_A ← ATTN(RMSNorm(h))

    Features:
        - Pre-norm: RMSNorm applied to input before projection.
        - Unconstrained weight matrices (Wq, Wk, Wv, Wo as nn.Linear).
        - RoPE applied to q and k.
        - Softmax scale: 1/sqrt(d_k) (standard).
        - Causal masking via F.scaled_dot_product_attention(is_causal=True).

    Attributes:
        d_model: Model embedding dimension.
        n_heads: Number of attention heads.
        d_k: Per-head key/query dimension (d_model // n_heads).
        Wq: Query projection, shape (n_heads * d_k, d_model).
        Wk: Key projection, shape (n_heads * d_k, d_model).
        Wv: Value projection, shape (n_heads * d_k, d_model).
        Wo: Output projection, shape (d_model, n_heads * d_k).
        norm: RMSNorm applied to input.
        rope: Rotary position embeddings.
        softmax_scale: Attention scale factor (1/sqrt(d_k)).
    """

    def __init__(self, config: Config) -> None:
        """Initialize GPTAttention.

        Args:
            config: Experiment configuration. Key fields:
                - config.d_model: Model dimension.
                - config.n_heads: Number of attention heads.
                - config.d_k: Per-head dimension.
                - config.context_length: Max sequence length for RoPE cache.
                - config.rope_base: RoPE base frequency (10000).
                - config.bias: Whether to use bias (False per paper).
        """
        super().__init__()

        self.d_model: int = config.d_model
        self.n_heads: int = config.n_heads
        self.d_k: int = config.d_k

        # Projection matrices — unconstrained in GPT baseline
        self.Wq: nn.Linear = nn.Linear(
            config.d_model, config.n_heads * config.d_k, bias=config.bias
        )
        self.Wk: nn.Linear = nn.Linear(
            config.d_model, config.n_heads * config.d_k, bias=config.bias
        )
        self.Wv: nn.Linear = nn.Linear(
            config.d_model, config.n_heads * config.d_k, bias=config.bias
        )
        self.Wo: nn.Linear = nn.Linear(
            config.n_heads * config.d_k, config.d_model, bias=config.bias
        )

        # Pre-norm: RMSNorm applied to input before projection
        self.norm: nn.RMSNorm = nn.RMSNorm(config.d_model)

        # Rotary position embeddings (applied per head to d_k-dim vectors)
        self.rope: RoPEEmbedding = RoPEEmbedding(
            d_k=config.d_k,
            max_seq_len=config.context_length,
            base=config.rope_base,
        )

        # Standard softmax scale: 1/sqrt(d_k)
        self.softmax_scale: float = 1.0 / math.sqrt(config.d_k)

    def forward(
        self,
        h: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute multi-head self-attention with pre-norm.

        Args:
            h: Hidden state tensor of shape (batch_size, seq_len, d_model).
            mask: Optional additive attention mask. If None, causal masking
                is applied via is_causal=True in scaled_dot_product_attention.

        Returns:
            Attention output tensor of shape (batch_size, seq_len, d_model).
            The residual addition (h + output) is performed in GPTLayer.
        """
        batch_size: int = h.shape[0]
        seq_len: int = h.shape[1]

        # Pre-norm: normalize input before projection
        h_norm: Tensor = self.norm(h)  # (B, T, d_model)

        # Project to queries, keys, values
        q: Tensor = self.Wq(h_norm)  # (B, T, n_heads * d_k)
        k: Tensor = self.Wk(h_norm)  # (B, T, n_heads * d_k)
        v: Tensor = self.Wv(h_norm)  # (B, T, n_heads * d_k)

        # Reshape to (B, n_heads, T, d_k) for multi-head processing
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE to queries and keys
        q = self.rope(q)  # (B, n_heads, T, d_k)
        k = self.rope(k)  # (B, n_heads, T, d_k)

        # Scaled dot-product attention with causal masking
        # F.scaled_dot_product_attention handles flash attention when available
        attn_out: Tensor = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.softmax_scale,
        )  # (B, n_heads, T, d_k)

        # Reshape back to (B, T, n_heads * d_k)
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.d_k
        )

        # Output projection
        output: Tensor = self.Wo(attn_out)  # (B, T, d_model)

        return output


class GPTMLP(nn.Module):
    """Standard SwiGLU MLP for the GPT baseline.

    Implements the MLP block from Section 2.4.1:
        h_M ← MLP(RMSNorm(h))

    Features:
        - Pre-norm: RMSNorm applied to input.
        - SwiGLU activation: u * SiLU(v) (Shazeer, 2020).
        - Unconstrained weight matrices.

    Attributes:
        d_model: Model embedding dimension.
        d_mlp: MLP hidden dimension (4 * d_model per paper Table 2).
        norm: RMSNorm applied to input.
        Wu: Gate-up projection, shape (d_mlp, d_model).
        Wv: Gate projection, shape (d_mlp, d_model).
        Wo: Down projection, shape (d_model, d_mlp).
    """

    def __init__(self, config: Config) -> None:
        """Initialize GPTMLP.

        Args:
            config: Experiment configuration. Key fields:
                - config.d_model: Model dimension.
                - config.d_mlp: MLP hidden dimension (4 * d_model).
                - config.bias: Whether to use bias (False per paper).
        """
        super().__init__()

        self.d_model: int = config.d_model
        self.d_mlp: int = config.d_mlp

        # Pre-norm
        self.norm: nn.RMSNorm = nn.RMSNorm(config.d_model)

        # SwiGLU projections — unconstrained in GPT baseline
        self.Wu: nn.Linear = nn.Linear(config.d_model, config.d_mlp, bias=config.bias)
        self.Wv: nn.Linear = nn.Linear(config.d_model, config.d_mlp, bias=config.bias)
        self.Wo: nn.Linear = nn.Linear(config.d_mlp, config.d_model, bias=config.bias)

    def _swiglu(self, u: Tensor, v: Tensor) -> Tensor:
        """Compute SwiGLU activation: u * SiLU(v).

        Args:
            u: Gate-up tensor of shape (..., d_mlp).
            v: Gate tensor of shape (..., d_mlp).

        Returns:
            Gated activation tensor of shape (..., d_mlp).
        """
        return u * F.silu(v)

    def forward(self, h: Tensor) -> Tensor:
        """Compute SwiGLU MLP with pre-norm.

        Args:
            h: Hidden state tensor of shape (batch_size, seq_len, d_model).

        Returns:
            MLP output tensor of shape (batch_size, seq_len, d_model).
            The residual addition (h + output) is performed in GPTLayer.
        """
        # Pre-norm
        h_norm: Tensor = self.norm(h)  # (B, T, d_model)

        # SwiGLU projections
        u: Tensor = self.Wu(h_norm)  # (B, T, d_mlp)
        v: Tensor = self.Wv(h_norm)  # (B, T, d_mlp)

        # Gated activation
        gate: Tensor = self._swiglu(u, v)  # (B, T, d_mlp)

        # Down projection
        output: Tensor = self.Wo(gate)  # (B, T, d_model)

        return output


class GPTLayer(nn.Module):
    """One GPT decoder layer with pre-norm residual connections.

    Implements the baseline Transformer layer from Section 2.2.1:
        h ← h + ATTN(RMSNorm(h))
        h ← h + MLP(RMSNorm(h))

    The RMSNorm operations are encapsulated inside GPTAttention and GPTMLP
    respectively (pre-norm architecture). GPTLayer itself only handles the
    residual additions.

    Attributes:
        attention: GPTAttention block.
        mlp: GPTMLP block.
    """

    def __init__(self, config: Config) -> None:
        """Initialize GPTLayer.

        Args:
            config: Experiment configuration passed to sub-modules.
        """
        super().__init__()

        self.attention: GPTAttention = GPTAttention(config)
        self.mlp: GPTMLP = GPTMLP(config)

    def forward(
        self,
        h: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply one GPT decoder layer.

        Args:
            h: Hidden state tensor of shape (batch_size, seq_len, d_model).
            mask: Optional attention mask (passed to attention block).

        Returns:
            Updated hidden state tensor of shape (batch_size, seq_len, d_model).
        """
        # Residual connection: h + ATTN(RMSNorm(h))
        h = h + self.attention(h, mask)

        # Residual connection: h + MLP(RMSNorm(h))
        h = h + self.mlp(h)

        return h


class GPTModel(nn.Module):
    """Baseline decoder-only GPT model.

    Implements the full GPT architecture as described in Section 2 of the
    paper. Serves as the baseline for comparison with nGPT.

    Architecture:
        - Separate input and output embedding matrices (not tied).
        - L layers of GPTLayer (pre-norm residual connections).
        - Final RMSNorm after the last layer.
        - Cross-entropy loss for next-token prediction.

    Parameter count (paper Table 2):
        - 0.5B model: 468.2M parameters
        - 1B model: 1025.7M parameters

    Attributes:
        config: Experiment configuration.
        E_input: Input embedding matrix, shape (vocab_size, d_model).
        E_output: Output projection (logit head), shape (vocab_size, d_model).
        layers: ModuleList of GPTLayer instances.
        final_norm: RMSNorm applied after the last layer.
    """

    def __init__(self, config: Config) -> None:
        """Initialize GPTModel.

        Args:
            config: Experiment configuration. Key fields:
                - config.vocab_size: Vocabulary size (32000 or 50257).
                - config.d_model: Model dimension.
                - config.n_layers: Number of transformer layers.
                - config.n_heads: Number of attention heads.
                - config.d_k: Per-head dimension.
                - config.d_mlp: MLP hidden dimension.
        """
        super().__init__()

        self.config: Config = config

        # Input embedding: maps token IDs to d_model-dim vectors
        self.E_input: nn.Embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )

        # Output projection (logit head): maps d_model to vocab_size
        # NOT tied to E_input — separate learnable matrix per paper Section 2.1
        self.E_output: nn.Linear = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        # Transformer layers
        self.layers: nn.ModuleList = nn.ModuleList(
            [GPTLayer(config) for _ in range(config.n_layers)]
        )

        # Final normalization after the last layer (paper Table 1)
        self.final_norm: nn.RMSNorm = nn.RMSNorm(config.d_model)

        # Initialize weights following Radford et al. (2018) / Appendix A.6
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights following Appendix A.6.

        GPT initialization (config.yaml initialization.gpt):
            - All nn.Linear and nn.Embedding: N(0, 0.02)
            - Output projection matrices (Wo in attention, Wo in MLP):
              N(0, 0.02 / sqrt(2 * n_layers))
              This follows Radford et al. (2018) to prevent residual stream
              growth with depth.
        """
        std_base: float = 0.02  # config.yaml initialization.gpt.std
        std_output: float = 0.02 / math.sqrt(2.0 * self.config.n_layers)

        for module_name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std_base)
            elif isinstance(module, nn.Linear):
                # Identify output projection matrices by name convention:
                # "attention.Wo" and "mlp.Wo" are the output projections
                is_output_proj: bool = (
                    module_name.endswith("attention.Wo")
                    or module_name.endswith("mlp.Wo")
                )
                if is_output_proj:
                    nn.init.normal_(module.weight, mean=0.0, std=std_output)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std_base)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass through the GPT model.

        Args:
            tokens: Input token indices of shape (batch_size, seq_len).
                Values must be in [0, vocab_size).
            targets: Optional target token indices of shape
                (batch_size, seq_len) for computing cross-entropy loss.
                If None, only logits are returned.

        Returns:
            A tuple (logits, loss) where:
                - logits: Unnormalized token probabilities of shape
                  (batch_size, seq_len, vocab_size).
                - loss: Scalar cross-entropy loss if targets is not None,
                  otherwise None.
        """
        # Token embeddings: (B, T) → (B, T, d_model)
        h: Tensor = self.E_input(tokens)

        # Apply transformer layers
        for layer in self.layers:
            h = layer(h)

        # Final normalization (paper Table 1: "Final: h ← RMSNorm(h)")
        h = self.final_norm(h)  # (B, T, d_model)

        # Compute logits: (B, T, d_model) → (B, T, vocab_size)
        logits: Tensor = self.E_output(h)

        # Compute loss if targets provided
        loss: Optional[Tensor] = None
        if targets is not None:
            # Flatten for cross-entropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return logits, loss

    def get_num_params(self) -> int:
        """Count total trainable parameters.

        Returns:
            Total number of trainable parameters. Expected values per
            paper Table 2: 468.2M (0.5B model), 1025.7M (1B model).
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizer(
        self,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Configure AdamW optimizer with parameter group separation.

        Separates parameters into two groups:
            - Decayed: 2D+ tensors (weight matrices, embeddings) — apply
              weight_decay to regularize.
            - Non-decayed: 1D tensors (biases, RMSNorm scale parameters) —
              no weight decay.

        This follows the standard practice for GPT training (paper Table 3:
        weight_decay=0.1 for GPT).

        Args:
            lr: Initial learning rate. The only tuned hyperparameter per
                paper Appendix A.7.
            weight_decay: Weight decay coefficient (0.1 for GPT per
                config.yaml training.gpt.weight_decay).
            betas: Adam momentum coefficients. Paper default: (0.9, 0.95)
                per config.yaml training.gpt.betas.

        Returns:
            Configured AdamW optimizer with two parameter groups.
        """
        # Separate parameters by dimensionality
        decayed_params: List[Tensor] = []
        non_decayed_params: List[Tensor] = []

        seen_params: set = set()

        for module in self.modules():
            for param_name, param in module.named_parameters(recurse=False):
                if id(param) in seen_params:
                    continue
                seen_params.add(id(param))

                if not param.requires_grad:
                    continue

                if param.ndim >= 2:
                    # Weight matrices and embeddings: apply weight decay
                    decayed_params.append(param)
                else:
                    # Biases, RMSNorm scale/bias (1D): no weight decay
                    non_decayed_params.append(param)

        param_groups = [
            {"params": decayed_params, "weight_decay": weight_decay},
            {"params": non_decayed_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
        )

        return optimizer


# ---------------------------------------------------------------------------
# nGPT Components
# ---------------------------------------------------------------------------

class nGPTAttention(nn.Module):
    """Normalized multi-head self-attention for nGPT.

    Implements the attention block from Section 2.3.2:
        hA = Norm(ATTN(h))

    Key differences from GPTAttention:
        1. No input RMSNorm (removed per paper Table 1).
        2. All weight matrices (Wq, Wk, Wv, Wo) are NormLinear — each
           embedding-dimension vector has unit L2 norm.
        3. QK normalization: q and k are normalized after RoPE, then scaled
           by learnable sqk (Equations 15, 16).
        4. Softmax scale: sqrt(d_k) instead of 1/