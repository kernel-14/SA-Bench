## models/transformer/attention.py
"""Block-wise causal attention for the NFIG Transformer.

Implements the frequency-band-aware attention mask and multi-head scaled
dot-product attention described in Section 3.2 of the NFIG paper.

The autoregressive factorization:
    p(T_1, T_2, ..., T_n) = Π_i p(T_i | T_1, T_2, ..., T_{i-1})

is enforced by a block lower-triangular attention mask at the band level:
  - Tokens within band i attend to ALL tokens in bands 0..i (full within-band
    attention + full attention to all prior bands).
  - Tokens in band i CANNOT attend to tokens in bands i+1..n-1.

This is the key structural difference from standard token-level causal masking
used in raster-scan AR models (e.g., GPT). Within each frequency band, all
tokens are predicted jointly (parallel prediction), not sequentially.

Config values used (config.yaml nfig section):
    hidden_dim:    1024   (D)
    num_heads:     16     (H)
    scale_factors: [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    total_tokens:  680    (sum of s_i^2)

Derived:
    head_dim = hidden_dim // num_heads = 64
    scale    = head_dim ** -0.5 = 0.125
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BlockwiseCausalAttention(nn.Module):
    """Multi-head attention with a static block-wise causal mask.

    Enforces the NFIG autoregressive constraint: tokens in frequency band i
    can attend to all tokens in bands 0..i but not to bands i+1..n-1.
    Within each band, attention is fully bidirectional (all-to-all).

    The attention mask is NOT stored as a buffer inside this class — it is
    constructed via build_causal_mask() and managed at the NFIGTransformer
    level, then passed into forward() as the attn_mask argument. This matches
    the design specification exactly.

    Architecture:
        x [B, T, D]
          → qkv_proj: Linear(D, 3D, bias=False) → Q, K, V each [B, T, D]
          → reshape to [B, H, T, head_dim]
          → F.scaled_dot_product_attention with block-wise causal mask
          → reshape to [B, T, D]
          → out_proj: Linear(D, D, bias=False)
          → [B, T, D]

    where T = total_tokens = 680, D = hidden_dim = 1024, H = num_heads = 16,
    head_dim = D // H = 64.

    Attributes:
        num_heads: Number of attention heads (H = 16 from config.nfig.num_heads).
        head_dim: Dimension per attention head (D // H = 64).
        scale: Scaling factor for dot-product attention (head_dim ** -0.5 = 0.125).
        qkv_proj: Fused QKV linear projection, Linear(hidden_dim, 3*hidden_dim).
        out_proj: Output linear projection, Linear(hidden_dim, hidden_dim).
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 16,
    ) -> None:
        """Initialize BlockwiseCausalAttention.

        Args:
            hidden_dim: Total hidden dimension D of the transformer.
                From config.nfig.hidden_dim = 1024.
                Must be divisible by num_heads.
            num_heads: Number of attention heads H.
                From config.nfig.num_heads = 16.
                Must be a positive integer that divides hidden_dim evenly.

        Raises:
            ValueError: If hidden_dim is not divisible by num_heads.
            ValueError: If hidden_dim or num_heads are not positive integers.
        """
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be a positive integer, got {hidden_dim}."
            )
        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}."
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}. "
                f"Got hidden_dim % num_heads = {hidden_dim % num_heads}."
            )

        # Store configuration as attributes (per design specification).
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_dim // num_heads  # 1024 // 16 = 64
        self.scale: float = self.head_dim ** -0.5     # 1 / sqrt(64) = 0.125

        # Store hidden_dim for use in forward().
        self._hidden_dim: int = hidden_dim

        # --- Fused QKV projection ---
        # Maps [B, T, D] → [B, T, 3D] in a single matrix multiply.
        # bias=False: standard practice for transformer attention projections;
        # reduces parameter count and avoids bias-induced attention asymmetry.
        self.qkv_proj: nn.Linear = nn.Linear(
            hidden_dim, 3 * hidden_dim, bias=False
        )

        # --- Output projection ---
        # Maps [B, T, D] → [B, T, D] after multi-head concatenation.
        # bias=False: consistent with qkv_proj and standard ViT practice.
        self.out_proj: nn.Linear = nn.Linear(
            hidden_dim, hidden_dim, bias=False
        )

        # Initialize weights following standard transformer initialization.
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize QKV and output projection weights.

        Uses scaled normal initialization for QKV (standard for transformers)
        and zero-initialized output projection (residual branch starts as
        near-identity, stabilizing early training).

        The output projection zero-init is a common trick from GPT-2 and
        subsequent work: each residual branch starts as a no-op, so the
        model initially behaves like a shallower network and gradually
        learns to use the full depth.
        """
        # QKV projection: scaled normal initialization.
        # std = 0.02 is the standard GPT-style initialization.
        nn.init.normal_(self.qkv_proj.weight, mean=0.0, std=0.02)

        # Output projection: zero initialization for residual branch stability.
        # The transformer block adds this output to the residual stream;
        # starting at zero means the block initially passes through unchanged.
        nn.init.zeros_(self.out_proj.weight)

    def build_causal_mask(
        self,
        scale_factors: List[int],
        device: torch.device,
    ) -> Tensor:
        """Build the static block-wise causal attention mask.

        Constructs a boolean mask of shape (total_tokens, total_tokens) where
        entry [i, j] is True if token i is allowed to attend to token j.

        The mask enforces the NFIG autoregressive constraint:
            mask[i, j] = True  iff  band(j) <= band(i)

        This creates a block lower-triangular structure at the band level:
          - Diagonal blocks (same band): all True — full within-band attention.
          - Lower-triangular blocks (earlier band): all True — attend to all
            tokens from all previous frequency bands.
          - Upper-triangular blocks (later band): all False — no future band
            attention (causal constraint).

        Token layout (for default scale_factors = [1,2,3,4,5,6,8,10,13,16]):
            band 0:  1 token   (positions   0..0)
            band 1:  4 tokens  (positions   1..4)
            band 2:  9 tokens  (positions   5..13)
            band 3: 16 tokens  (positions  14..29)
            band 4: 25 tokens  (positions  30..54)
            band 5: 36 tokens  (positions  55..90)
            band 6: 64 tokens  (positions  91..154)
            band 7: 100 tokens (positions 155..254)
            band 8: 169 tokens (positions 255..423)
            band 9: 256 tokens (positions 424..679)
            Total: 680 tokens

        Args:
            scale_factors: List of n integer scale factors defining the token
                grid size for each frequency band. scale_factors[i] = s_i
                gives band i a token grid of s_i × s_i tokens.
                From config.nfig.scale_factors = [1,2,3,4,5,6,8,10,13,16].
                Must be non-empty and contain positive integers.
            device: Target device for the returned mask tensor.
                Should match the device of the input tensors in forward().

        Returns:
            Boolean tensor of shape (total_tokens, total_tokens) on the given
            device. Entry [i, j] is True if token i can attend to token j.
            total_tokens = sum(s**2 for s in scale_factors) = 680 for default.

        Raises:
            ValueError: If scale_factors is empty or contains non-positive values.

        Example:
            >>> attn = BlockwiseCausalAttention(hidden_dim=1024, num_heads=16)
            >>> mask = attn.build_causal_mask([1, 2, 3], torch.device('cpu'))
            >>> mask.shape
            torch.Size([14, 14])  # 1 + 4 + 9 = 14 tokens
            >>> # Band 0 (pos 0) can only attend to itself
            >>> mask[0, 0].item()
            True
            >>> mask[0, 1].item()  # Band 0 cannot attend to band 1
            False
            >>> # Band 1 (pos 1..4) can attend to band 0 and itself
            >>> mask[1, 0].item()
            True
            >>> mask[1, 1].item()
            True
            >>> mask[1, 5].item()  # Band 1 cannot attend to band 2
            False
        """
        if not scale_factors:
            raise ValueError(
                "scale_factors must be a non-empty list. "
                "Got an empty list."
            )
        if any(s <= 0 for s in scale_factors):
            raise ValueError(
                f"All scale_factors must be positive integers. "
                f"Got: {scale_factors}"
            )

        # Compute per-band token counts: band_sizes[i] = scale_factors[i]^2
        band_sizes: List[int] = [s * s for s in scale_factors]
        total_tokens: int = sum(band_sizes)
        n_bands: int = len(band_sizes)

        # Compute [start, end) index ranges for each band in the flat sequence.
        # band_ranges[i] = (start_idx, end_idx) for band i.
        band_ranges: List[tuple] = []
        cumulative: int = 0
        for size in band_sizes:
            band_ranges.append((cumulative, cumulative + size))
            cumulative += size

        # Initialize mask as all-False (no attention allowed by default).
        # Shape: (total_tokens, total_tokens), dtype=bool.
        # Constructed on CPU first for efficiency, then moved to target device.
        mask: Tensor = torch.zeros(
            total_tokens, total_tokens,
            dtype=torch.bool,
            device=device,
        )

        # Fill in the allowed attention regions.
        # For each query band b_q and each key band b_k:
        #   If b_k <= b_q: allow all tokens in b_q to attend to all tokens in b_k.
        for b_q in range(n_bands):
            q_start: int
            q_end: int
            q_start, q_end = band_ranges[b_q]

            for b_k in range(n_bands):
                # Causal constraint: only attend to current and past bands.
                if b_k <= b_q:
                    k_start: int
                    k_end: int
                    k_start, k_end = band_ranges[b_k]

                    # Set the entire block [q_start:q_end, k_start:k_end] to True.
                    # This allows all tokens in band b_q to attend to all tokens
                    # in band b_k (full bidirectional attention within and across
                    # past bands).
                    mask[q_start:q_end, k_start:k_end] = True

        return mask

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """Apply multi-head attention with the block-wise causal mask.

        Implements scaled dot-product attention with the NFIG frequency-band
        causal constraint. Uses PyTorch's F.scaled_dot_product_attention for
        numerical stability and potential flash attention acceleration on H100.

        The attention mask is passed in from NFIGTransformer (which stores it
        as a registered buffer) rather than being stored inside this class.
        This matches the design specification.

        Computation:
            1. Fused QKV projection: x → Q, K, V each of shape (B, T, D)
            2. Reshape to multi-head format: (B, H, T, head_dim)
            3. Scaled dot-product attention with block-wise causal mask
            4. Reshape back to (B, T, D)
            5. Output projection: (B, T, D) → (B, T, D)

        Args:
            x: Input token sequence of shape (B, T, D) where:
                - B: batch size
                - T: total_tokens = 680 (config.nfig.total_tokens)
                - D: hidden_dim = 1024 (config.nfig.hidden_dim)
                Values are in an unconstrained real range (transformer hidden states).
            attn_mask: Block-wise causal attention mask of shape (T, T), dtype bool.
                Entry [i, j] = True means token i is allowed to attend to token j.
                Constructed by build_causal_mask() and stored in NFIGTransformer.
                PyTorch's F.scaled_dot_product_attention broadcasts this over
                the batch (B) and head (H) dimensions automatically.
                Must be on the same device as x.

        Returns:
            Attended output tensor of shape (B, T, D).
            Same shape as input x. Gradients flow through all operations.

        Raises:
            RuntimeError: If x.shape[-1] != hidden_dim (dimension mismatch).
            RuntimeError: If attn_mask.shape != (T, T) where T = x.shape[1].
        """
        B: int
        T: int
        D: int
        B, T, D = x.shape

        # Validate input dimensions.
        if D != self._hidden_dim:
            raise RuntimeError(
                f"Input hidden dimension D={D} does not match "
                f"expected hidden_dim={self._hidden_dim}. "
                f"Input shape: {tuple(x.shape)}."
            )

        if attn_mask.shape != (T, T):
            raise RuntimeError(
                f"attn_mask shape {tuple(attn_mask.shape)} does not match "
                f"expected (T, T) = ({T}, {T}) where T = x.shape[1]. "
                "Ensure the mask was built with the same scale_factors as the "
                "token sequence length."
            )

        # ------------------------------------------------------------------ #
        # Step 1: Fused QKV projection
        # ------------------------------------------------------------------ #
        # x: (B, T, D) → qkv: (B, T, 3D)
        qkv: Tensor = self.qkv_proj(x)

        # Split into Q, K, V along the last dimension.
        # Each: (B, T, D)
        q: Tensor
        k: Tensor
        v: Tensor
        q, k, v = qkv.chunk(3, dim=-1)

        # ------------------------------------------------------------------ #
        # Step 2: Reshape to multi-head format
        # ------------------------------------------------------------------ #
        # (B, T, D) → (B, T, H, head_dim) → (B, H, T, head_dim)
        # contiguous() ensures memory layout is compatible with view().
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Each: (B, H, T, head_dim) = (B, 16, 680, 64)

        # ------------------------------------------------------------------ #
        # Step 3: Scaled dot-product attention with block-wise causal mask
        # ------------------------------------------------------------------ #
        # F.scaled_dot_product_attention:
        #   - Accepts attn_mask of shape (T, T) and broadcasts over (B, H).
        #   - Boolean mask: True = attend, False = mask to -inf before softmax.
        #   - Uses flash attention when available (H100 hardware per config).
        #   - scale parameter overrides the default sqrt(head_dim) scaling.
        #
        # attn_mask: (T, T) bool → broadcast to (B, H, T, T) internally.
        # The scale self.scale = head_dim ** -0.5 = 0.125 is passed explicitly
        # to ensure consistent behavior regardless of PyTorch version defaults.
        attended: Tensor = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,   # (T, T) bool, broadcast over (B, H)
            dropout_p=0.0,         # No dropout in attention (config.nfig.dropout = 0.0)
            is_causal=False,       # We provide our own custom mask; not standard causal
            scale=self.scale,      # head_dim ** -0.5 = 0.125
        )
        # attended: (B, H, T, head_dim) = (B, 16, 680, 64)

        # ------------------------------------------------------------------ #
        # Step 4: Reshape back to sequence format
        # ------------------------------------------------------------------ #
        # (B, H, T, head_dim) → (B, T, H, head_dim) → (B, T, D)
        # contiguous() is required before view() after transpose().
        attended = attended.transpose(1, 2).contiguous()  # (B, T, H, head_dim)
        attended = attended.view(B, T, self._hidden_dim)   # (B, T, D)

        # ------------------------------------------------------------------ #
        # Step 5: Output projection
        # ------------------------------------------------------------------ #
        # (B, T, D) → (B, T, D)
        out: Tensor = self.out_proj(attended)

        return out

    def extra_repr(self) -> str:
        """Return a human-readable string with key attention configuration.

        Returns:
            String describing the attention module's key dimensions.
        """
        return (
            f"hidden_dim={self._hidden_dim}, "
            f"num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, "
            f"scale={self.scale:.6f}"
        )
