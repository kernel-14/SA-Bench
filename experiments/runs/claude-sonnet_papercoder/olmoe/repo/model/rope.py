## model/rope.py
"""Rotary Position Embedding (RoPE) for OLMoE.

Implements Rotary Position Embedding as described in:
  Su et al. (2023). "RoFormer: Enhanced Transformer with Rotary Position Embedding."
  https://arxiv.org/abs/2104.09864

Used in OLMoE-1B-7B to encode positional information in query and key
projections of every attention layer. Applied AFTER QK-Norm and BEFORE
the attention score computation.

Configuration values (from config.yaml):
  model.rope_theta: 10000.0
  model.max_seq_len: 4096
  model.hidden_dim: 2048
  model.num_heads: 16
  -> head_dim = hidden_dim // num_heads = 128  (dim parameter)

Key design decisions:
  - cos/sin caches are registered as buffers (not parameters): they move
    with the model to GPU automatically but are not optimized.
  - Cache is precomputed up to max_seq_len=4096 and sliced at runtime.
  - apply_rotary is a @staticmethod for clean usage in model/attention.py.
  - Computation is done in float32 for numerical stability; cos/sin are
    cast to match input dtype (BF16 in training) before multiplication.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the input tensor by splitting it in half and interleaving with sign flip.

    This implements the core rotation operation used in RoPE. For a vector
    x = [x1, x2, ..., x_{d/2}, x_{d/2+1}, ..., x_d], it returns:
        [-x_{d/2+1}, ..., -x_d, x_1, ..., x_{d/2}]

    This corresponds to the 2D rotation matrix [[cos, -sin], [sin, cos]]
    applied to each consecutive pair (x_{2i-1}, x_{2i}) when combined with
    the standard RoPE formula: x_rotated = x * cos + rotate_half(x) * sin.

    Args:
        x: Input tensor of shape (..., dim) where dim is even.

    Returns:
        Rotated tensor of the same shape as x.
    """
    half_dim: int = x.shape[-1] // 2
    # First half: x[..., :half_dim]  shape (..., dim/2)
    x1: Tensor = x[..., :half_dim]
    # Second half: x[..., half_dim:] shape (..., dim/2)
    x2: Tensor = x[..., half_dim:]
    # Concatenate [-x2, x1] along last dimension: shape (..., dim)
    return torch.cat([-x2, x1], dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with precomputed cos/sin caches.

    Precomputes cosine and sine caches for all positions up to max_seq_len,
    then slices them at runtime for efficiency. The caches are registered as
    PyTorch buffers so they are automatically moved to the correct device
    when the model is moved (e.g., .to('cuda')) and are included in
    state_dict() for checkpoint compatibility.

    The rotation is applied to query and key tensors via apply_rotary(),
    which is a static method for clean usage in model/attention.py.

    Attributes:
        dim: Head dimension (hidden_dim // num_heads = 128 for OLMoE-1B-7B).
        max_seq_len: Maximum sequence length for the precomputed cache (4096).
        theta: RoPE base frequency (10000.0 for OLMoE-1B-7B).
        cos_cache: Precomputed cosine values, shape (max_seq_len, dim).
        sin_cache: Precomputed sine values, shape (max_seq_len, dim).

    Example:
        >>> rope = RotaryEmbedding(dim=128, max_seq_len=4096, theta=10000.0)
        >>> cos, sin = rope(seq_len=512)
        >>> cos.shape
        torch.Size([512, 128])
        >>> q = torch.randn(2, 16, 512, 128)  # (batch, heads, seq, head_dim)
        >>> k = torch.randn(2, 16, 512, 128)
        >>> q_rot, k_rot = RotaryEmbedding.apply_rotary(q, k, cos, sin)
        >>> q_rot.shape
        torch.Size([2, 16, 512, 128])
    """

    def __init__(
        self,
        dim: int = 128,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
    ) -> None:
        """Initialize RotaryEmbedding with precomputed cos/sin caches.

        Computes inverse frequencies using the standard RoPE formula:
            inv_freq[i] = 1.0 / (theta ** (2i / dim))  for i in [0, dim/2)

        Then builds position-frequency outer products and caches cos/sin.

        Args:
            dim: Head dimension to apply RoPE over. For OLMoE-1B-7B:
                 dim = hidden_dim // num_heads = 2048 // 16 = 128.
                 (from config.yaml: model.hidden_dim=2048, model.num_heads=16)
            max_seq_len: Maximum sequence length for the precomputed cache.
                         (from config.yaml: model.max_seq_len = 4096)
            theta: RoPE base frequency controlling the range of rotations.
                   Larger theta -> slower rotation -> longer effective context.
                   (from config.yaml: model.rope_theta = 10000.0)

        Raises:
            ValueError: If dim is not a positive even integer, max_seq_len <= 0,
                        or theta <= 0.
        """
        super().__init__()

        if dim <= 0 or dim % 2 != 0:
            raise ValueError(
                f"dim must be a positive even integer, got {dim}. "
                f"For OLMoE-1B-7B: dim = hidden_dim // num_heads = 2048 // 16 = 128."
            )
        if max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be a positive integer, got {max_seq_len}."
            )
        if theta <= 0:
            raise ValueError(
                f"theta must be positive, got {theta}."
            )

        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        self.theta: float = theta

        # -----------------------------------------------------------------------
        # Step 1: Compute inverse frequencies.
        # inv_freq[i] = 1.0 / (theta ** (2i / dim)) for i in [0, dim/2)
        # Shape: (dim/2,) = (64,) for OLMoE-1B-7B
        # Computed in float32 for numerical precision.
        # -----------------------------------------------------------------------
        # arange(0, dim, 2) produces [0, 2, 4, ..., dim-2], shape (dim/2,)
        exponents: Tensor = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        inv_freq: Tensor = 1.0 / (theta ** exponents)
        # Shape: (dim/2,) = (64,)

        # -----------------------------------------------------------------------
        # Step 2: Compute position-frequency outer product.
        # positions: [0, 1, 2, ..., max_seq_len-1], shape (max_seq_len,)
        # outer product: shape (max_seq_len, dim/2) = (4096, 64)
        # freqs[pos, i] = pos * inv_freq[i] = angle for position pos, pair i
        # -----------------------------------------------------------------------
        positions: Tensor = torch.arange(0, max_seq_len, dtype=torch.float32)
        freqs: Tensor = torch.outer(positions, inv_freq)
        # Shape: (max_seq_len, dim/2) = (4096, 64)

        # -----------------------------------------------------------------------
        # Step 3: Build full-dimension embedding by concatenating freqs with itself.
        # This matches the standard LLaMA/OLMo RoPE formulation where the first
        # dim/2 and last dim/2 positions use the same angles.
        # emb shape: (max_seq_len, dim) = (4096, 128)
        # -----------------------------------------------------------------------
        emb: Tensor = torch.cat([freqs, freqs], dim=-1)
        # Shape: (max_seq_len, dim) = (4096, 128)

        # -----------------------------------------------------------------------
        # Step 4: Compute and register cos/sin caches as buffers.
        # Buffers are:
        #   - Moved to the correct device with .to(device) / .cuda()
        #   - Saved/loaded with state_dict() for checkpoint compatibility
        #   - NOT included in model.parameters() (no gradient, no optimization)
        # -----------------------------------------------------------------------
        cos_cache: Tensor = emb.cos()  # (max_seq_len, dim) = (4096, 128)
        sin_cache: Tensor = emb.sin()  # (max_seq_len, dim) = (4096, 128)

        self.register_buffer("cos_cache", cos_cache, persistent=True)
        self.register_buffer("sin_cache", sin_cache, persistent=True)

    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Return precomputed cos and sin tensors sliced to the requested length.

        Slicing is O(1) and avoids recomputation on every forward pass.
        The returned tensors are used by apply_rotary() in model/attention.py.

        Args:
            seq_len: The sequence length to return embeddings for.
                     Must be <= self.max_seq_len (4096 for OLMoE-1B-7B).

        Returns:
            Tuple of (cos, sin) tensors, each of shape (seq_len, dim).
            These are slices of the precomputed caches.

        Raises:
            AssertionError: If seq_len > self.max_seq_len.

        Example:
            >>> rope = RotaryEmbedding(dim=128, max_seq_len=4096)
            >>> cos, sin = rope(512)
            >>> cos.shape, sin.shape
            (torch.Size([512, 128]), torch.Size([512, 128]))
        """
        assert seq_len <= self.max_seq_len, (
            f"seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len}). "
            f"Increase max_seq_len when constructing RotaryEmbedding."
        )
        # Slice the precomputed caches to the requested sequence length.
        # cos_cache and sin_cache have shape (max_seq_len, dim).
        # Sliced result has shape (seq_len, dim).
        cos: Tensor = self.cos_cache[:seq_len]  # (seq_len, dim)
        sin: Tensor = self.sin_cache[:seq_len]  # (seq_len, dim)
        return cos, sin

    @staticmethod
    def apply_rotary(
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply rotary position embeddings to query and key tensors.

        Implements the RoPE rotation:
            q_rotated = q * cos + rotate_half(q) * sin
            k_rotated = k * cos + rotate_half(k) * sin

        This encodes absolute position information into Q and K such that
        the dot product Q·K^T naturally captures relative position differences.

        The cos/sin tensors from forward() have shape (seq_len, dim). They are
        reshaped to (1, 1, seq_len, dim) for broadcasting over the batch and
        head dimensions of Q and K.

        Dtype handling: cos/sin caches are stored in float32 (for precision
        during precomputation). Q and K are in BF16 during training
        (pretraining.bf16=true in config.yaml). The cos/sin are cast to match
        the input dtype before multiplication to avoid dtype mismatch errors
        and maintain BF16 throughput.

        Placement in attention (from model/attention.py call order):
            1. Project Q, K via linear layers
            2. Reshape to (batch, heads, seq, head_dim)
            3. Apply QK-Norm: q = q_norm(q), k = k_norm(k)  [if use_qk_norm]
            4. Apply RoPE: q, k = apply_rotary(q, k, cos, sin)  ← this method
            5. Compute attention scores

        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
               For OLMoE-1B-7B: (B, 16, S, 128).
            k: Key tensor of shape (batch, num_heads, seq_len, head_dim).
               For OLMoE-1B-7B: (B, 16, S, 128).
            cos: Cosine values of shape (seq_len, head_dim) from forward().
                 Will be reshaped to (1, 1, seq_len, head_dim) for broadcasting.
            sin: Sine values of shape (seq_len, head_dim) from forward().
                 Will be reshaped to (1, 1, seq_len, head_dim) for broadcasting.

        Returns:
            Tuple of (q_rotated, k_rotated), each with the same shape and
            dtype as the corresponding input tensor.

        Example:
            >>> rope = RotaryEmbedding(dim=128, max_seq_len=4096)
            >>> cos, sin = rope(seq_len=512)
            >>> q = torch.randn(2, 16, 512, 128)
            >>> k = torch.randn(2, 16, 512, 128)
            >>> q_rot, k_rot = RotaryEmbedding.apply_rotary(q, k, cos, sin)
            >>> q_rot.shape
            torch.Size([2, 16, 512, 128])
        """
        # -----------------------------------------------------------------------
        # Reshape cos/sin for broadcasting over batch and head dimensions.
        # Input cos/sin shape: (seq_len, head_dim)
        # After unsqueeze: (1, 1, seq_len, head_dim)
        # This broadcasts correctly against q/k of shape (B, H, S, D).
        # -----------------------------------------------------------------------
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

        # -----------------------------------------------------------------------
        # Cast cos/sin to match the input dtype.
        # During BF16 training, q and k are BF16 tensors. The cos/sin caches
        # are stored in float32. Casting here avoids dtype mismatch errors and
        # ensures the rotation runs in BF16 for throughput efficiency.
        # -----------------------------------------------------------------------
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)

        # -----------------------------------------------------------------------
        # Apply the rotary transformation:
        #   x_rotated = x * cos + rotate_half(x) * sin
        #
        # This implements the 2D rotation matrix [[cos, -sin], [sin, cos]]
        # applied to each consecutive pair of dimensions in Q and K.
        # -----------------------------------------------------------------------
        q_rotated: Tensor = q * cos + _rotate_half(q) * sin
        k_rotated: Tensor = k * cos + _rotate_half(k) * sin

        return q_rotated, k_rotated

    def extra_repr(self) -> str:
        """Return extra representation string for printing the module.

        Returns:
            String showing dim, max_seq_len, and theta values.
        """
        return (
            f"dim={self.dim}, "
            f"max_seq_len={self.max_seq_len}, "
            f"theta={self.theta}"
        )


def _cast_value(value: object, type_hint: object) -> object:
    """Helper to cast a value to a given type hint string.

    This is a minimal helper used only within this module for any
    local type coercion needs. The main cast helper lives in config.py.

    Args:
        value: The value to cast.
        type_hint: The target type (as a type object or string annotation).

    Returns:
        The value, potentially cast to the target type.
    """
    # For this module we don't need complex casting; return as-is.
    return value
