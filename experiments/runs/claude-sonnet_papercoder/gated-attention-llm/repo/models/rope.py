## models/rope.py
"""Rotary Position Embedding (RoPE) with YaRN context extension support.

This module implements RoPE as described in Su et al. (2024) "RoFormer: Enhanced
Transformer with Rotary Position Embedding", with support for context length
extension via base frequency scaling and YaRN interpolation (Peng et al., 2023).

Used by GatedMultiHeadAttention to apply position-dependent rotations to query
and key tensors before scaled dot-product attention.

Config values used:
    model.d_k: Per-head dimension (128 for MoE, 64 for dense)
    model.max_seq_len: Maximum sequence length (4096)
    rope.base: RoPE base frequency (10000.0)
    rope.extended_base: Extended base for long-context training (1_000_000.0)
    rope.extended_seq_len: Sequence length for continued training (32768)
    yarn.max_seq_len: Final context length after YaRN (131072)
    yarn.scale: YaRN frequency scaling factor (8.0)
    yarn.beta_fast: High-frequency threshold divisor (32)
    yarn.beta_slow: Low-frequency threshold divisor (1)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding with YaRN context extension.

    Precomputes complex exponential rotation factors for all positions up to
    max_seq_len and applies them to query/key tensors during attention.

    Attributes:
        d_k: Per-head dimension.
        max_seq_len: Maximum sequence length supported by current freqs_cis.
        base: Current RoPE base frequency.
        yarn_config: Optional YaRN configuration dict.
        freqs_cis: Precomputed complex rotation factors, shape [max_seq_len, d_k//2].
    """

    def __init__(
        self,
        d_k: int,
        max_seq_len: int,
        base: float = 10000.0,
        yarn_config: Optional[dict] = None,
    ) -> None:
        """Initialize RoPE embedding and precompute frequency cache.

        Args:
            d_k: Per-head dimension. Must be even.
            max_seq_len: Maximum sequence length (from model.max_seq_len, default 4096).
            base: RoPE base frequency (from rope.base, default 10000.0).
            yarn_config: Optional dict with YaRN parameters:
                - scale (float): Frequency scaling factor (yarn.scale = 8.0)
                - beta_fast (int): High-frequency threshold divisor (yarn.beta_fast = 32)
                - beta_slow (int): Low-frequency threshold divisor (yarn.beta_slow = 1)
                - max_seq_len (int): Target context length (yarn.max_seq_len = 131072)
        """
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")

        self.d_k: int = d_k
        self.max_seq_len: int = max_seq_len
        self.base: float = base
        self.yarn_config: Optional[dict] = yarn_config

        # Precompute and register as buffer (non-learnable, device-aware)
        freqs_cis = self._compute_freqs()
        self.register_buffer("freqs_cis", freqs_cis, persistent=True)

    def _compute_freqs(self) -> torch.Tensor:
        """Precompute complex exponential rotation factors using standard RoPE.

        Computes theta_i = 1 / (base ^ (2i / d_k)) for i in [0, d_k//2),
        then builds the outer product with positions [0, max_seq_len) and
        converts to complex exponentials via torch.polar.

        Returns:
            freqs_cis: Complex tensor of shape [max_seq_len, d_k//2], dtype complex64.
                Entry [p, i] = exp(j * p * theta_i).
        """
        # Compute inverse frequencies: theta_i = 1 / (base ^ (2i / d_k))
        # Shape: [d_k // 2]
        exponents = torch.arange(0, self.d_k, 2, dtype=torch.float32) / self.d_k
        theta = 1.0 / (self.base ** exponents)

        # Compute position-frequency outer product
        # positions shape: [max_seq_len], theta shape: [d_k//2]
        # freqs shape: [max_seq_len, d_k//2], entry [p, i] = p * theta_i
        positions = torch.arange(0, self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, theta)

        # Convert to complex exponentials: exp(j * freqs)
        # torch.polar(magnitude, angle) = magnitude * exp(j * angle)
        # Shape: [max_seq_len, d_k//2], dtype: complex64
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis

    def _compute_freqs_yarn(
        self,
        new_max_len: int,
        yarn_config: dict,
        original_max_len: int,
    ) -> torch.Tensor:
        """Compute YaRN-modified frequency cache for context length extension.

        YaRN (Peng et al., 2023) interpolates between original and scaled
        frequencies based on the wavelength of each frequency component relative
        to the original and target context lengths.

        For each frequency dimension i:
            - High-frequency dims (small wavelength): keep original frequency
            - Low-frequency dims (large wavelength): scale down by yarn_config['scale']
            - Mid-range dims: linear blend between original and scaled

        Args:
            new_max_len: Target context length (e.g., 131072 from yarn.max_seq_len).
            yarn_config: Dict with keys 'scale', 'beta_fast', 'beta_slow'.
            original_max_len: The max_seq_len before YaRN extension (e.g., 32768).

        Returns:
            freqs_cis: Complex tensor of shape [new_max_len, d_k//2], dtype complex64.
        """
        scale: float = float(yarn_config.get("scale", 8.0))
        beta_fast: int = int(yarn_config.get("beta_fast", 32))
        beta_slow: int = int(yarn_config.get("beta_slow", 1))

        # Compute standard inverse frequencies
        exponents = torch.arange(0, self.d_k, 2, dtype=torch.float32) / self.d_k
        theta = 1.0 / (self.base ** exponents)  # shape: [d_k//2]

        # Compute wavelength for each frequency dimension
        # lambda_i = 2 * pi / theta_i
        wavelengths = 2.0 * math.pi / theta  # shape: [d_k//2]

        # Compute wavelength thresholds based on original context length
        # low_threshold: wavelengths above this get full interpolation (low-freq)
        # high_threshold: wavelengths below this get no interpolation (high-freq)
        low_threshold: float = float(original_max_len) / float(beta_slow)
        high_threshold: float = float(original_max_len) / float(beta_fast)

        # Compute per-dimension interpolation factor alpha_i in [0, 1]
        # alpha_i = 1.0: keep original frequency (high-freq, small wavelength)
        # alpha_i = 0.0: use scaled frequency (low-freq, large wavelength)
        alpha = torch.zeros_like(theta)

        # High-frequency region: wavelength < high_threshold → alpha = 1
        high_freq_mask = wavelengths < high_threshold
        alpha[high_freq_mask] = 1.0

        # Low-frequency region: wavelength > low_threshold → alpha = 0
        # (already 0 from initialization)

        # Mid-range region: linear ramp
        mid_mask = (~high_freq_mask) & (wavelengths <= low_threshold)
        if mid_mask.any():
            alpha[mid_mask] = (wavelengths[mid_mask] - high_threshold) / (
                low_threshold - high_threshold
            )

        # Compute scaled (interpolated) frequency
        theta_scaled = theta / scale

        # Blend: theta_yarn_i = alpha_i * theta_i + (1 - alpha_i) * theta_scaled_i
        theta_yarn = alpha * theta + (1.0 - alpha) * theta_scaled  # shape: [d_k//2]

        # Build frequency matrix for new_max_len positions
        positions = torch.arange(0, new_max_len, dtype=torch.float32)
        freqs = torch.outer(positions, theta_yarn)  # shape: [new_max_len, d_k//2]

        # Convert to complex exponentials
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis

    def apply_rotary(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply RoPE rotations to an input tensor (Q or K).

        Applies the precomputed complex rotations to the input tensor by:
        1. Reshaping to complex view (pairing consecutive real dimensions)
        2. Multiplying by the appropriate rotation factors
        3. Converting back to real representation

        Args:
            x: Input tensor of shape [batch, seq, num_heads, d_k].
                May be bfloat16 in training; internally cast to float32 for
                complex arithmetic.
            position_ids: Optional integer tensor of shape [batch, seq] or [seq]
                specifying position indices into freqs_cis. If None, defaults to
                sequential positions [0, 1, ..., seq_len-1].

        Returns:
            Tensor of same shape and dtype as x, with RoPE applied.
        """
        batch_size, seq_len, num_heads, d_k = x.shape
        original_dtype = x.dtype

        # Default to sequential positions if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)
            # Shape: [seq_len] → broadcast over batch

        # Index into frequency cache using position_ids
        # freqs_cis shape: [max_seq_len, d_k//2]
        if position_ids.dim() == 1:
            # Shape: [seq_len, d_k//2]
            freqs = self.freqs_cis[position_ids]
            # Unsqueeze for batch and head broadcasting: [1, seq_len, 1, d_k//2]
            freqs = freqs.unsqueeze(0).unsqueeze(2)
        else:
            # position_ids shape: [batch, seq_len]
            # freqs shape: [batch, seq_len, d_k//2]
            freqs = self.freqs_cis[position_ids]
            # Unsqueeze for head broadcasting: [batch, seq_len, 1, d_k//2]
            freqs = freqs.unsqueeze(2)

        # Cast to float32 for complex arithmetic (bfloat16 not supported by view_as_complex)
        x_float = x.float()

        # Reshape to pair consecutive dimensions for complex view
        # [batch, seq, num_heads, d_k] → [batch, seq, num_heads, d_k//2, 2]
        x_reshaped = x_float.reshape(batch_size, seq_len, num_heads, d_k // 2, 2)

        # View as complex: [batch, seq, num_heads, d_k//2] of dtype complex64
        x_complex = torch.view_as_complex(x_reshaped)

        # Apply rotation via complex multiplication
        # freqs broadcasts over batch and num_heads dimensions
        x_rotated = x_complex * freqs  # shape: [batch, seq, num_heads, d_k//2]

        # Convert back to real: [batch, seq, num_heads, d_k//2, 2]
        x_real = torch.view_as_real(x_rotated)

        # Flatten last two dims: [batch, seq, num_heads, d_k]
        x_out = x_real.flatten(3)

        # Cast back to original dtype (e.g., bfloat16)
        return x_out.to(original_dtype)

    def extend_context(
        self,
        new_max_len: int,
        new_base: float,
        yarn_config: dict,
    ) -> None:
        """Extend context length support by updating the frequency cache in-place.

        Implements the two-phase context extension from Section 4.4:
            Phase 1: Update RoPE base (10k → 1M) for continued training on 32k data.
            Phase 2: Apply YaRN scaling to reach 128k without further training.

        This method modifies self.freqs_cis in-place. After calling this method,
        the model supports sequences up to new_max_len positions.

        Args:
            new_max_len: Target context length (e.g., 131072 from yarn.max_seq_len).
            new_base: New RoPE base frequency (e.g., 1_000_000.0 from rope.extended_base).
            yarn_config: Dict with YaRN parameters:
                - scale (float): Frequency scaling factor (8.0)
                - beta_fast (int): High-frequency threshold divisor (32)
                - beta_slow (int): Low-frequency threshold divisor (1)
                - extended_seq_len (int, optional): Intermediate seq len for phase 1 (32768)

        Note:
            This operation is destructive — the original freqs_cis is replaced.
            The device of the new buffer matches the current freqs_cis device.
        """
        device = self.freqs_cis.device

        # Phase 1: Update base frequency for continued training
        # Paper Sec 4.4: "increase the RoPE base from 10k to 1M"
        # and "continue training on data with a sequence length of 32k"
        self.base = new_base

        # Determine intermediate sequence length for phase 1
        # Use extended_seq_len from yarn_config if provided, else use new_max_len
        intermediate_seq_len: int = int(
            yarn_config.get("extended_seq_len", new_max_len)
        )

        # Update max_seq_len to intermediate length for phase 1
        self.max_seq_len = intermediate_seq_len

        # Recompute standard freqs with new base and intermediate seq_len
        # This is used for continued training on 32k data
        phase1_freqs = self._compute_freqs()

        if new_max_len <= intermediate_seq_len:
            # No YaRN needed — just update with new base and seq_len
            self.register_buffer(
                "freqs_cis",
                phase1_freqs.to(device),
                persistent=True,
            )
            return

        # Phase 2: Apply YaRN scaling to reach new_max_len (e.g., 128k)
        # Paper Sec 4.4: "use YaRN to extend the context length to 128k"
        original_max_len_for_yarn = intermediate_seq_len
        yarn_freqs = self._compute_freqs_yarn(
            new_max_len=new_max_len,
            yarn_config=yarn_config,
            original_max_len=original_max_len_for_yarn,
        )

        # Update max_seq_len to final target length
        self.max_seq_len = new_max_len
        self.yarn_config = yarn_config

        # Register updated buffer, ensuring it's on the correct device
        self.register_buffer(
            "freqs_cis",
            yarn_freqs.to(device),
            persistent=True,
        )
