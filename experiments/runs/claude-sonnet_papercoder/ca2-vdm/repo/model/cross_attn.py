## model/cross_attn.py
"""Visual-text cross attention module for Ca2-VDM.

This module implements CrossAttention, the visual-text cross attention layer
used in T2V (text-to-video) generation. It bridges the T5 text encoder output
with the visual token stream inside each DiTBlock, following the PixArt-α /
DiT-style architecture.

This module is only instantiated when config.use_cross_attn == True
(i.e., task == 't2v'). For video prediction on SkyTimelapse, this module
is entirely absent from the model.

Paper: Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal
Generation and Cache Sharing (Sec. 3.2).

Configuration references (config.yaml):
    model.model_dim:      1152   (visual hidden dimension, dim)
    model.context_dim:    1024   (T5-Large output dimension, context_dim)
    model.num_heads:      16
    model.dropout:        0.0
    model.use_cross_attn: true   (only True for task='t2v')
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Multi-head cross attention from visual tokens to T5 text features.

    Each visual token (spatial grid position) attends to the T5 text token
    sequence. The L (frame) dimension is treated as a batch dimension so that
    every frame independently conditions on the same text embedding. This
    matches the PixArt-α / DiT convention for video generation.

    Projection dimensions:
        W_Q: dim       → dim        (visual queries)
        W_K: context_dim → dim      (text keys, projected to visual space)
        W_V: context_dim → dim      (text values, projected to visual space)
        W_O: dim       → dim        (output projection)

    Attention map shape per frame: (N, S) where N = H*W = 1024 and S is the
    T5 token sequence length.

    Attributes:
        dim: Visual hidden dimension. From config.yaml: ``model.model_dim: 1152``.
        context_dim: T5 text encoder output dimension. From config.yaml:
                     ``model.context_dim: 1024``.
        num_heads: Number of attention heads. From config.yaml:
                   ``model.num_heads: 16``.
        head_dim: Per-head dimension. ``dim // num_heads = 72``.
        scale: Dot-product scaling factor. ``head_dim ** -0.5``.
        W_Q: Query projection ``Linear(dim, dim, bias=False)``.
        W_K: Key projection ``Linear(context_dim, dim, bias=False)``.
        W_V: Value projection ``Linear(context_dim, dim, bias=False)``.
        W_O: Output projection ``Linear(dim, dim, bias=False)``.
        attn_dropout: Dropout applied to attention weights during training.

    Example::

        cross_attn = CrossAttention(dim=1152, context_dim=1024, num_heads=16)
        # x: visual tokens, shape (B*L, N, dim) where N = H*W = 1024
        x = torch.randn(4 * 8, 1024, 1152)
        # context: T5 text features, shape (B*L, S, context_dim)
        context = torch.randn(4 * 8, 77, 1024)
        # mask: T5 attention mask, shape (B*L, S), 1=valid, 0=padding
        mask = torch.ones(4 * 8, 77, dtype=torch.long)
        out = cross_attn(x, context, mask)
        # out: (32, 1024, 1152) — same shape as x
    """

    def __init__(
        self,
        dim: int = 1152,
        context_dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the cross attention module.

        Args:
            dim: Visual hidden dimension. Must be a positive integer divisible
                 by ``num_heads``. From config.yaml: ``model.model_dim: 1152``.
            context_dim: T5 text encoder output dimension. Must be a positive
                         integer. From config.yaml:
                         ``model.context_dim: 1024`` (T5-Large).
            num_heads: Number of attention heads. Must be a positive integer
                       that divides ``dim`` evenly. From config.yaml:
                       ``model.num_heads: 16``.
            dropout: Dropout probability applied to attention weights during
                     training. Set to 0.0 at inference. From config.yaml:
                     ``model.dropout: 0.0``.

        Raises:
            ValueError: If ``dim`` is not divisible by ``num_heads``.
            ValueError: If ``dim``, ``context_dim``, or ``num_heads`` is not
                        a positive integer.
            ValueError: If ``dropout`` is not in ``[0.0, 1.0)``.
        """
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}.")
        if context_dim <= 0:
            raise ValueError(
                f"context_dim must be a positive integer, got {context_dim}."
            )
        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}."
            )
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}). "
                f"Got head_dim = {dim / num_heads:.2f} (non-integer)."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0), got {dropout}."
            )

        self.dim: int = dim
        self.context_dim: int = context_dim
        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        # Precomputed scaling factor: 1 / sqrt(head_dim).
        # Matches the paper's attention formula: QK^T / sqrt(C').
        self.scale: float = self.head_dim ** -0.5

        # Query projection: visual tokens → query space.
        # bias=False follows PixArt-α / DiT convention.
        self.W_Q: nn.Linear = nn.Linear(dim, dim, bias=False)

        # Key and value projections: text features → visual space.
        # W_K and W_V project from context_dim (1024 for T5-Large) to dim (1152).
        self.W_K: nn.Linear = nn.Linear(context_dim, dim, bias=False)
        self.W_V: nn.Linear = nn.Linear(context_dim, dim, bias=False)

        # Output projection: attended features → visual space.
        self.W_O: nn.Linear = nn.Linear(dim, dim, bias=False)

        # Attention weight dropout (applied only during training).
        self.attn_dropout: nn.Dropout = nn.Dropout(p=dropout)
        self._dropout_p: float = dropout

        # Initialise weights following Xavier uniform convention.
        self._init_weights()

    # -----------------------------------------------------------------------
    # Weight initialisation
    # -----------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialise projection weights with Xavier uniform distribution.

        Follows the Open-Sora / PixArt-α initialisation convention:
        - Xavier uniform for all projection weight matrices.
        - No bias terms to initialise (bias=False for all projections).

        Xavier uniform is appropriate for attention projections as it
        maintains variance across layers during forward and backward passes.
        """
        for module in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.xavier_uniform_(module.weight)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape a projected tensor into multi-head format.

        Args:
            x: Tensor of shape ``(B, N, dim)`` where ``B`` is the effective
               batch size (may include the frame dimension folded in by the
               calling ``DiTBlock``), and ``N`` is the number of tokens.

        Returns:
            Tensor of shape ``(B, num_heads, N, head_dim)``.
        """
        B, N, _ = x.shape
        # (B, N, dim) → (B, N, num_heads, head_dim)
        x = x.reshape(B, N, self.num_heads, self.head_dim)
        # (B, N, num_heads, head_dim) → (B, num_heads, N, head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of ``_split_heads``: merge multi-head format back to ``dim``.

        Args:
            x: Tensor of shape ``(B, num_heads, N, head_dim)``.

        Returns:
            Tensor of shape ``(B, N, dim)``.
        """
        B, _, N, _ = x.shape
        # (B, num_heads, N, head_dim) → (B, N, num_heads, head_dim)
        x = x.transpose(1, 2)
        # contiguous() required before reshape after transpose.
        x = x.contiguous()
        # (B, N, num_heads, head_dim) → (B, N, dim)
        return x.reshape(B, N, self.dim)

    def _build_additive_mask(
        self,
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert a boolean/integer attention mask to an additive float mask.

        The T5 tokenizer returns ``attention_mask`` where ``1 = valid token``
        and ``0 = padding token``. This method converts it to an additive mask
        suitable for addition to attention logits before softmax:
        - Valid tokens (mask == 1): additive value = 0.0 (no effect).
        - Padding tokens (mask == 0): additive value = -inf (masked out).

        Args:
            mask: Integer or boolean tensor of shape ``(B, S)`` where
                  ``B`` is the effective batch size and ``S`` is the text
                  sequence length. Values: ``1 = valid``, ``0 = padding``.
            dtype: Target dtype for the additive mask (float32, float16,
                   bfloat16). Should match the attention logits dtype to
                   avoid implicit casts.

        Returns:
            Additive float mask of shape ``(B, 1, 1, S)`` with values in
            ``{0.0, -inf}``. The shape is broadcastable over the
            ``(B, num_heads, N, S)`` attention score tensor.
        """
        # Convert to float and invert: 1 → 0.0 (keep), 0 → -inf (mask out).
        # Using a large negative value instead of -inf avoids NaN in softmax
        # when all tokens in a sequence are masked (degenerate case).
        additive_mask: torch.Tensor = torch.zeros_like(mask, dtype=dtype)
        additive_mask = additive_mask.masked_fill(mask == 0, float("-inf"))

        # Expand to (B, 1, 1, S) for broadcasting over (B, num_heads, N, S).
        return additive_mask.unsqueeze(1).unsqueeze(2)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute visual-text cross attention.

        Each visual token attends to the full T5 text token sequence.
        Padding tokens in the text sequence are masked out via the attention
        mask to prevent visual tokens from attending to meaningless padding.

        The calling ``DiTBlock`` is responsible for:
        1. Reshaping ``x`` from ``(B, L, H*W, dim)`` to ``(B*L, H*W, dim)``
           before calling this method (treating frames as batch elements).
        2. Expanding ``context`` and ``mask`` from ``(B, S, context_dim)``
           and ``(B, S)`` to ``(B*L, S, context_dim)`` and ``(B*L, S)``
           via ``repeat_interleave(L, dim=0)`` so each frame sees the same
           text conditioning.
        3. Reshaping the output back to ``(B, L, H*W, dim)`` after this call.

        Args:
            x: Visual token features of shape ``(B, N, dim)`` where:
               - ``B`` is the effective batch size (may be ``B_orig * L``
                 when frames are folded into the batch dimension).
               - ``N = H*W = 1024`` spatial tokens (32×32 latent grid after
                 8× VAE downsampling of 256×256 input).
               - ``dim = 1152`` (config.yaml: ``model.model_dim``).
            context: T5 text encoder output of shape ``(B, S, context_dim)``
               where:
               - ``S`` is the T5 token sequence length (up to 77 or 128).
               - ``context_dim = 1024`` (config.yaml: ``model.context_dim``
                 for T5-Large).
            mask: Optional T5 attention mask of shape ``(B, S)``.
               Values: ``1 = valid token``, ``0 = padding token``.
               This is the ``attention_mask`` output from the T5 tokenizer.
               If ``None``, all text tokens are treated as valid (no masking).

        Returns:
            Attended visual features of shape ``(B, N, dim)`` — same shape
            as input ``x``. The calling ``DiTBlock`` applies the residual
            connection: ``x = x + gate * CrossAttn(norm(x), context, mask)``.

        Raises:
            ValueError: If ``x`` does not have exactly 3 dimensions.
            ValueError: If ``context`` does not have exactly 3 dimensions.
            ValueError: If ``x`` and ``context`` have incompatible batch sizes.
            ValueError: If ``context``'s last dimension does not match
                        ``self.context_dim``.
            ValueError: If ``mask`` shape is incompatible with ``context``.
        """
        # ── Input validation ─────────────────────────────────────────────────
        if x.ndim != 3:
            raise ValueError(
                f"x must be a 3-D tensor (B, N, dim), "
                f"got shape {tuple(x.shape)}."
            )
        if context.ndim != 3:
            raise ValueError(
                f"context must be a 3-D tensor (B, S, context_dim), "
                f"got shape {tuple(context.shape)}."
            )

        batch_size: int = x.shape[0]
        num_tokens: int = x.shape[1]  # N = H*W = 1024
        seq_len: int = context.shape[1]  # S = text sequence length

        if context.shape[0] != batch_size:
            raise ValueError(
                f"x and context must have the same batch size. "
                f"Got x.shape[0]={batch_size}, context.shape[0]={context.shape[0]}."
            )
        if context.shape[2] != self.context_dim:
            raise ValueError(
                f"context last dimension must be context_dim={self.context_dim}, "
                f"got {context.shape[2]}."
            )
        if mask is not None:
            if mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"mask shape must be ({batch_size}, {seq_len}), "
                    f"got {tuple(mask.shape)}."
                )

        # ── Step 1: Compute Q, K, V projections ──────────────────────────────
        # Q: visual tokens → query space.
        # (B, N, dim) → (B, N, dim)
        Q: torch.Tensor = self.W_Q(x)

        # K, V: text features → visual space.
        # (B, S, context_dim) → (B, S, dim)
        K: torch.Tensor = self.W_K(context)
        V: torch.Tensor = self.W_V(context)

        # ── Step 2: Reshape for multi-head attention ──────────────────────────
        # Q: (B, num_heads, N, head_dim)
        # K: (B, num_heads, S, head_dim)
        # V: (B, num_heads, S, head_dim)
        Q_heads: torch.Tensor = self._split_heads(Q)
        K_heads: torch.Tensor = self._split_heads(K)
        V_heads: torch.Tensor = self._split_heads(V)

        # ── Step 3: Build additive attention mask ─────────────────────────────
        # Convert T5 padding mask (1=valid, 0=padding) to additive float mask
        # of shape (B, 1, 1, S) for broadcasting over (B, num_heads, N, S).
        attn_mask: Optional[torch.Tensor] = None
        if mask is not None:
            attn_mask = self._build_additive_mask(mask, dtype=x.dtype)
            # attn_mask: (B, 1, 1, S) — broadcasts over (B, num_heads, N, S)

        # ── Step 4: Scaled dot-product attention ─────────────────────────────
        # Use F.scaled_dot_product_attention (PyTorch 2.0+) for efficiency.
        # Falls back to manual implementation for older PyTorch versions.
        dropout_p: float = self._dropout_p if self.training else 0.0

        _use_sdpa: bool = hasattr(F, "scaled_dot_product_attention")

        if _use_sdpa:
            # F.scaled_dot_product_attention handles scaling (1/sqrt(head_dim))
            # internally via the ``scale`` argument (PyTorch >= 2.0).
            # attn_mask is an additive float mask: 0.0 for valid, -inf for padding.
            attn_out: torch.Tensor = F.scaled_dot_product_attention(
                Q_heads,
                K_heads,
                V_heads,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                scale=self.scale,
            )
            # attn_out: (B, num_heads, N, head_dim)
        else:
            # Manual implementation for PyTorch < 2.0 compatibility.
            # attn_scores: (B, num_heads, N, S)
            attn_scores: torch.Tensor = (
                torch.matmul(Q_heads, K_heads.transpose(-2, -1)) * self.scale
            )

            # Apply additive padding mask before softmax.
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask

            # Softmax over text sequence dimension (S).
            attn_weights: torch.Tensor = F.softmax(attn_scores, dim=-1)

            # Apply dropout to attention weights (training only).
            attn_weights = self.attn_dropout(attn_weights)

            # Weighted sum of values: (B, num_heads, N, head_dim)
            attn_out = torch.matmul(attn_weights, V_heads)

        # ── Step 5: Merge heads and apply output projection ───────────────────
        # (B, num_heads, N, head_dim) → (B, N, dim)
        attn_out = self._merge_heads(attn_out)

        # Output projection: (B, N, dim) → (B, N, dim)
        output: torch.Tensor = self.W_O(attn_out)

        # ── Return ────────────────────────────────────────────────────────────
        # output: (B, N, dim) — same shape as input x.
        # The calling DiTBlock applies the residual connection and gate scaling.
        return output
