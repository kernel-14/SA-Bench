## model/prefix_spatial_attn.py
"""Prefix-Enhanced Spatial Attention module for Ca2-VDM.

This module implements PrefixEnhancedSpatialAttention, which augments each
frame's spatial self-attention with P' clean prefix frames concatenated along
the spatial dimension. This strengthens the guidance from conditional frames
without the quadratic cost of full temporal attention.

Paper: Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal
Generation and Cache Sharing (Sec. 3.2, Eq. 4).

Key equation from the paper:
    K̄(i) = W^K [h_0^{P-P'}, ..., h_0^{P-1}, h_t^i]   if i >= P
    K̄(i) = W^K [h_0^i, ..., h_0^i]                     if i < P  (self-repeat P'+1 times)
    Attention map shape per frame: (HW) × ((P'+1) × HW)

Configuration references (config.yaml):
    ca2vdm.prefix_spatial_len: 3   (P' = 3)
    model.model_dim:            1152
    model.num_heads:            16
    model.dropout:              0.0
    video_prediction.chunk_len: 8
    t2v.chunk_len:              16
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixEnhancedSpatialAttention(nn.Module):
    """Prefix-enhanced spatial attention for Ca2-VDM Transformer blocks.

    Each frame's spatial attention is augmented by concatenating P' clean
    prefix frames along the spatial (HW) dimension before computing keys and
    values. This gives every frame access to rich conditioning context from
    the most recent clean prefix frames, improving temporal consistency.

    The L dimension (number of frames) is treated as a batch dimension
    throughout — each frame's spatial attention is computed independently.
    There is no causal mask; causality is enforced by only using clean prefix
    frames as the prefix source.

    Three operational modes:

    **Training** (``spatial_kv_cache=None``, ``P`` is an int):
        Full sequence ``x`` of shape ``(L, HW, dim)`` is processed.
        Prefix frames ``x[P-P':P]`` are used to augment denoising target
        frames ``x[P:]``. Clean prefix frames ``x[:P]`` self-repeat.

    **Inference denoising stage** (``spatial_kv_cache`` is a Tensor,
    ``P=None``):
        Only the current ``l``-frame chunk ``x`` of shape ``(l, HW, dim)``
        is processed. The spatial KV cache (last P' generated frames) is
        used as the prefix source.

    **Inference cache writing stage** (``spatial_kv_cache=None``,
    ``P=l``):
        The denoised chunk is processed with self-repeat (all frames treated
        as clean prefix). Returns the last P' frames as the new spatial KV
        cache for the next AR step.

    Attributes:
        dim: Model hidden dimension. From config.yaml: ``model.model_dim: 1152``.
        num_heads: Number of attention heads. From config.yaml:
                   ``model.num_heads: 16``.
        prefix_len: Sub-prefix length P'. From config.yaml:
                    ``ca2vdm.prefix_spatial_len: 3``.
        head_dim: Per-head dimension. ``dim // num_heads = 72``.
        scale: Dot-product scaling factor. ``head_dim ** -0.5``.
        W_Q: Query projection ``Linear(dim, dim)``.
        W_K: Key projection ``Linear(dim, dim)``.
        W_V: Value projection ``Linear(dim, dim)``.
        W_O: Output projection ``Linear(dim, dim)``.
        attn_dropout: Dropout applied to attention weights during training.

    Example (training)::

        attn = PrefixEnhancedSpatialAttention(dim=1152, num_heads=16, prefix_len=3)
        x = torch.randn(33, 1024, 1152)   # (L, HW, dim)
        out, spatial_kv = attn(x, P=25, spatial_kv_cache=None)
        # out: (33, 1024, 1152)
        # spatial_kv: (3, 1024, 1152)  — last P' clean prefix frames

    Example (inference denoising)::

        attn = PrefixEnhancedSpatialAttention(dim=1152, num_heads=16, prefix_len=3)
        x_chunk = torch.randn(8, 1024, 1152)    # (l, HW, dim)
        cache = torch.randn(3, 1024, 1152)       # (P', HW, dim)
        out, spatial_kv = attn(x_chunk, P=None, spatial_kv_cache=cache)
        # out: (8, 1024, 1152)
        # spatial_kv: (3, 1024, 1152)  — cache returned unchanged
    """

    def __init__(
        self,
        dim: int = 1152,
        num_heads: int = 16,
        prefix_len: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the prefix-enhanced spatial attention module.

        Args:
            dim: Model hidden dimension. Must be a positive integer divisible
                 by ``num_heads``. From config.yaml: ``model.model_dim: 1152``.
            num_heads: Number of attention heads. Must be a positive integer
                       that divides ``dim`` evenly. From config.yaml:
                       ``model.num_heads: 16``.
            prefix_len: Sub-prefix length P'. Number of clean prefix frames
                        to concatenate spatially. From config.yaml:
                        ``ca2vdm.prefix_spatial_len: 3``. Must satisfy
                        ``prefix_len < chunk_len`` (paper: P' < l).
            dropout: Dropout probability applied to attention weights during
                     training. From config.yaml: ``model.dropout: 0.0``.

        Raises:
            ValueError: If ``dim`` is not divisible by ``num_heads``.
            ValueError: If ``dim``, ``num_heads``, or ``prefix_len`` is not
                        a positive integer.
            ValueError: If ``dropout`` is not in ``[0.0, 1.0)``.
        """
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}.")
        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, got {num_heads}."
            )
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}). "
                f"Got head_dim = {dim / num_heads:.2f} (non-integer)."
            )
        if prefix_len <= 0:
            raise ValueError(
                f"prefix_len must be a positive integer, got {prefix_len}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0), got {dropout}."
            )

        self.dim: int = dim
        self.num_heads: int = num_heads
        self.prefix_len: int = prefix_len
        self.head_dim: int = dim // num_heads
        self.scale: float = self.head_dim ** -0.5

        # Linear projections: all dim → dim.
        # W_K and W_V are applied to the concatenated (P'+1)*HW tokens,
        # but since nn.Linear operates on the last dimension only, this is
        # equivalent to projecting each token independently.
        self.W_Q: nn.Linear = nn.Linear(dim, dim, bias=True)
        self.W_K: nn.Linear = nn.Linear(dim, dim, bias=True)
        self.W_V: nn.Linear = nn.Linear(dim, dim, bias=True)
        self.W_O: nn.Linear = nn.Linear(dim, dim, bias=True)

        # Attention weight dropout (applied only during training).
        self.attn_dropout: nn.Dropout = nn.Dropout(p=dropout)
        self._dropout_p: float = dropout

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_prefix_kv(
        self,
        x: torch.Tensor,
        P: Optional[int],
        cache: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build prefix-enhanced key and value inputs for all frames.

        Constructs the ``(L, (P'+1)*HW, dim)`` input tensor for W_K and W_V
        by concatenating P' prefix frames with each frame's own tokens along
        the spatial (HW) dimension.

        Three cases are handled:

        **Case 1 — Training** (``cache is None``, ``P`` is an int):
            For denoising target frames (``i >= P``): concatenate the last P'
            clean prefix frames with the current frame's tokens.
            For clean prefix frames (``i < P``): self-repeat the current
            frame's tokens P'+1 times.

        **Case 2 — Inference denoising** (``cache`` is a Tensor):
            For all l chunk frames: concatenate the cached P' frames with the
            current frame's tokens.

        **Case 3 — Inference cache writing** (``cache is None``, ``P == L``):
            All frames are treated as clean prefix → self-repeat P'+1 times.
            This is handled by Case 1 with ``P == L`` (no denoising target
            frames), so only the clean prefix branch executes.

        Args:
            x: Input hidden features.
               - Training / cache writing: shape ``(L, HW, dim)``.
               - Inference denoising: shape ``(l, HW, dim)``.
            P: Prefix boundary index. Number of clean prefix frames in ``x``.
               - Training: integer in ``[1, L]``.
               - Inference denoising: ``None`` (cache is used instead).
               - Inference cache writing: ``P = L`` (all frames are clean).
            cache: Spatial KV cache from the most recent generated chunk.
               - Inference denoising: shape ``(P', HW, dim)``.
               - Training / cache writing: ``None``.

        Returns:
            A tuple ``(K_input, V_input)`` where both tensors have shape
            ``(L, (P'+1)*HW, dim)``. These are passed to ``W_K`` and ``W_V``
            respectively.

        Raises:
            ValueError: If both ``P`` and ``cache`` are ``None``.
            ValueError: If ``cache`` shape does not match ``(P', HW, dim)``.
        """
        if P is None and cache is None:
            raise ValueError(
                "Either P (training/cache-writing) or cache (inference "
                "denoising) must be provided to _build_prefix_kv."
            )

        num_frames: int = x.shape[0]
        hw: int = x.shape[1]
        p_prime: int = self.prefix_len

        if cache is not None:
            # ── Case 2: Inference denoising stage ────────────────────────────
            # cache: (P', HW, dim) — last P' frames of most recent chunk.
            # x:     (l, HW, dim) — current denoising target chunk.
            if cache.shape != (p_prime, hw, self.dim):
                raise ValueError(
                    f"cache shape must be ({p_prime}, {hw}, {self.dim}), "
                    f"got {tuple(cache.shape)}."
                )

            # Expand cache to match the chunk batch dimension.
            # cache_expanded: (l, P', HW, dim)
            cache_expanded: torch.Tensor = cache.unsqueeze(0).expand(
                num_frames, -1, -1, -1
            )
            # x_expanded: (l, 1, HW, dim)
            x_expanded: torch.Tensor = x.unsqueeze(1)

            # Concatenate along the prefix dimension: (l, P'+1, HW, dim)
            kv_input: torch.Tensor = torch.cat(
                [cache_expanded, x_expanded], dim=1
            )
            # Reshape to (l, (P'+1)*HW, dim) for linear projection.
            kv_input = kv_input.reshape(num_frames, (p_prime + 1) * hw, self.dim)

            K_input: torch.Tensor = self.W_K(kv_input)
            V_input: torch.Tensor = self.W_V(kv_input)
            return K_input, V_input

        # ── Cases 1 & 3: Training or cache writing (cache is None) ───────────
        # P is guaranteed to be an int here (validated above).
        assert P is not None  # for type checker

        # Clamp P to valid range [0, num_frames].
        P_clamped: int = max(0, min(P, num_frames))

        # ── Build KV input for clean prefix frames (i < P_clamped) ──────────
        # Each clean prefix frame self-repeats P'+1 times along spatial dim.
        # prefix_part: (P_clamped, HW, dim)
        # After self-repeat: (P_clamped, P'+1, HW, dim) → (P_clamped, (P'+1)*HW, dim)
        if P_clamped > 0:
            prefix_part: torch.Tensor = x[:P_clamped]  # (P_clamped, HW, dim)
            # Expand each frame P'+1 times: unsqueeze → (P_clamped, 1, HW, dim)
            # expand → (P_clamped, P'+1, HW, dim)
            prefix_kv_input: torch.Tensor = (
                prefix_part.unsqueeze(1)
                .expand(-1, p_prime + 1, -1, -1)
                .reshape(P_clamped, (p_prime + 1) * hw, self.dim)
            )
        else:
            prefix_kv_input = x.new_empty(0, (p_prime + 1) * hw, self.dim)

        # ── Build KV input for denoising target frames (i >= P_clamped) ─────
        num_target: int = num_frames - P_clamped

        if num_target > 0:
            target_part: torch.Tensor = x[P_clamped:]  # (num_target, HW, dim)

            # Determine which prefix frames to use for augmentation.
            # Ideally: x[P_clamped - P' : P_clamped] (last P' clean frames).
            # Edge case: if P_clamped < P', we have fewer than P' clean frames.
            # In that case, repeat available frames to fill P' slots.
            if P_clamped >= p_prime:
                # Normal case: use the last P' clean prefix frames.
                aug_frames: torch.Tensor = x[P_clamped - p_prime : P_clamped]
                # aug_frames: (P', HW, dim)
            elif P_clamped > 0:
                # Edge case: fewer than P' clean frames available.
                # Repeat the available frames to fill P' slots.
                available: torch.Tensor = x[:P_clamped]  # (P_clamped, HW, dim)
                # Compute how many times to repeat and trim.
                repeat_times: int = (p_prime + P_clamped - 1) // P_clamped
                aug_frames = available.repeat(repeat_times, 1, 1)[:p_prime]
                # aug_frames: (P', HW, dim)
            else:
                # P_clamped == 0: no clean prefix frames at all.
                # Fall back to self-repeat of the first target frame.
                aug_frames = target_part[:1].expand(p_prime, -1, -1)
                # aug_frames: (P', HW, dim)

            # Expand aug_frames to match target batch dimension.
            # aug_expanded: (num_target, P', HW, dim)
            aug_expanded: torch.Tensor = aug_frames.unsqueeze(0).expand(
                num_target, -1, -1, -1
            )
            # target_expanded: (num_target, 1, HW, dim)
            target_expanded: torch.Tensor = target_part.unsqueeze(1)

            # Concatenate: (num_target, P'+1, HW, dim)
            target_kv_input: torch.Tensor = torch.cat(
                [aug_expanded, target_expanded], dim=1
            )
            # Reshape: (num_target, (P'+1)*HW, dim)
            target_kv_input = target_kv_input.reshape(
                num_target, (p_prime + 1) * hw, self.dim
            )
        else:
            target_kv_input = x.new_empty(0, (p_prime + 1) * hw, self.dim)

        # ── Concatenate prefix and target KV inputs along frame dimension ────
        # full_kv_input: (L, (P'+1)*HW, dim)
        if P_clamped > 0 and num_target > 0:
            full_kv_input: torch.Tensor = torch.cat(
                [prefix_kv_input, target_kv_input], dim=0
            )
        elif P_clamped > 0:
            full_kv_input = prefix_kv_input
        else:
            full_kv_input = target_kv_input

        # Apply W_K and W_V projections.
        K_input = self.W_K(full_kv_input)   # (L, (P'+1)*HW, dim)
        V_input = self.W_V(full_kv_input)   # (L, (P'+1)*HW, dim)

        return K_input, V_input

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape a projected tensor into multi-head format.

        Args:
            x: Tensor of shape ``(L, N, dim)`` where ``N`` is the number of
               tokens (either ``HW`` for queries or ``(P'+1)*HW`` for keys/values).

        Returns:
            Tensor of shape ``(L, num_heads, N, head_dim)``.
        """
        L, N, _ = x.shape
        # (L, N, dim) → (L, N, num_heads, head_dim)
        x = x.reshape(L, N, self.num_heads, self.head_dim)
        # (L, N, num_heads, head_dim) → (L, num_heads, N, head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of ``_split_heads``: merge multi-head format back to ``dim``.

        Args:
            x: Tensor of shape ``(L, num_heads, HW, head_dim)``.

        Returns:
            Tensor of shape ``(L, HW, dim)``.
        """
        L, _, hw, _ = x.shape
        # (L, num_heads, HW, head_dim) → (L, HW, num_heads, head_dim)
        x = x.transpose(1, 2)
        # contiguous() required before reshape after transpose.
        x = x.contiguous()
        # (L, HW, num_heads, head_dim) → (L, HW, dim)
        return x.reshape(L, hw, self.dim)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        P: Optional[int] = None,
        spatial_kv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute prefix-enhanced spatial attention.

        Handles all three operational modes (training, inference denoising,
        inference cache writing) through the ``P`` and ``spatial_kv_cache``
        arguments.

        Args:
            x: Input hidden features. Shape depends on mode:
               - Training: ``(L, HW, dim)`` where ``L = P + (L-P)`` is the
                 full clip length (clean prefix + denoising target).
               - Inference denoising: ``(l, HW, dim)`` — current chunk only.
               - Inference cache writing: ``(l, HW, dim)`` — denoised chunk.
               ``HW = (resolution / vae_downsample)^2 = 32^2 = 1024``
               (config.yaml: ``evaluation.resolution: 256``,
               ``model.vae_downsample: 8``).
            P: Prefix boundary index (number of clean prefix frames in ``x``).
               - Training: integer in ``[1, L]``.
               - Inference denoising: ``None`` (use ``spatial_kv_cache``).
               - Inference cache writing: ``P = l`` (all frames are clean).
               Must be ``None`` when ``spatial_kv_cache`` is provided.
            spatial_kv_cache: Cached hidden features from the last P' frames
               of the most recently generated chunk.
               - Inference denoising: shape ``(P', HW, dim)`` where
                 ``P' = prefix_len = 3`` (config.yaml:
                 ``ca2vdm.prefix_spatial_len: 3``).
               - Training / cache writing: ``None``.

        Returns:
            A tuple ``(output, spatial_kv_for_cache)`` where:
            - ``output``: Attended output of shape ``(L, HW, dim)`` (same
              shape as input ``x``). The calling ``DiTBlock`` applies the
              residual connection.
            - ``spatial_kv_for_cache``: Hidden features to store as the
              spatial KV cache for the next AR step:
              - Training: ``x[P - P' : P]`` of shape ``(P', HW, dim)``
                (last P' clean prefix frames' raw features).
              - Inference denoising: ``spatial_kv_cache`` unchanged, shape
                ``(P', HW, dim)`` (no update during denoising).
              - Inference cache writing: ``x[l - P' : l]`` of shape
                ``(P', HW, dim)`` (last P' frames of denoised chunk).

        Raises:
            ValueError: If ``x`` does not have exactly 3 dimensions.
            ValueError: If both ``P`` and ``spatial_kv_cache`` are ``None``.
            ValueError: If ``P`` is provided alongside ``spatial_kv_cache``
                        (ambiguous mode).
            ValueError: If ``spatial_kv_cache`` shape is incompatible.
        """
        if x.ndim != 3:
            raise ValueError(
                f"x must be a 3-D tensor (L, HW, dim), "
                f"got shape {tuple(x.shape)}."
            )

        if P is None and spatial_kv_cache is None:
            raise ValueError(
                "Either P (training/cache-writing mode) or spatial_kv_cache "
                "(inference denoising mode) must be provided."
            )

        num_frames: int = x.shape[0]
        hw: int = x.shape[1]
        p_prime: int = self.prefix_len

        # ── Step 1: Compute queries ───────────────────────────────────────────
        # Q: (L, HW, dim) → split heads → (L, num_heads, HW, head_dim)
        Q_raw: torch.Tensor = self.W_Q(x)  # (L, HW, dim)
        Q_heads: torch.Tensor = self._split_heads(Q_raw)
        # Q_heads: (L, num_heads, HW, head_dim)

        # ── Step 2: Build prefix-enhanced K and V ────────────────────────────
        # K_raw, V_raw: (L, (P'+1)*HW, dim)
        K_raw: torch.Tensor
        V_raw: torch.Tensor
        K_raw, V_raw = self._build_prefix_kv(x, P, spatial_kv_cache)

        # Split into multi-head format.
        # K_heads, V_heads: (L, num_heads, (P'+1)*HW, head_dim)
        K_heads: torch.Tensor = self._split_heads(K_raw)
        V_heads: torch.Tensor = self._split_heads(V_raw)

        # ── Step 3: Scaled dot-product attention ─────────────────────────────
        # L is treated as the batch dimension; each frame attends independently.
        # Attention scores: (L, num_heads, HW, (P'+1)*HW)
        # No causal mask — each frame attends freely to all (P'+1)*HW tokens.
        dropout_p: float = self._dropout_p if self.training else 0.0

        _use_sdpa: bool = hasattr(F, "scaled_dot_product_attention")

        if _use_sdpa:
            # F.scaled_dot_product_attention handles scaling internally.
            # No attn_mask needed (full attention over prefix-enhanced tokens).
            attn_out: torch.Tensor = F.scaled_dot_product_attention(
                Q_heads,
                K_heads,
                V_heads,
                attn_mask=None,
                dropout_p=dropout_p,
                scale=self.scale,
            )
            # attn_out: (L, num_heads, HW, head_dim)
        else:
            # Manual implementation for PyTorch < 2.0 compatibility.
            # attn_weights: (L, num_heads, HW, (P'+1)*HW)
            attn_weights: torch.Tensor = (
                torch.matmul(Q_heads, K_heads.transpose(-2, -1)) * self.scale
            )
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            # Weighted sum: (L, num_heads, HW, head_dim)
            attn_out = torch.matmul(attn_weights, V_heads)

        # ── Step 4: Merge heads and apply output projection ──────────────────
        # (L, num_heads, HW, head_dim) → (L, HW, dim)
        attn_out = self._merge_heads(attn_out)
        # Output projection: (L, HW, dim) → (L, HW, dim)
        output: torch.Tensor = self.W_O(attn_out)

        # ── Step 5: Compute spatial KV cache for next AR step ────────────────
        # The spatial KV cache stores the raw hidden features (before W_K/W_V
        # projection) of the last P' clean prefix frames. These will be used
        # as the prefix source in the next AR step's denoising stage.
        spatial_kv_for_cache: torch.Tensor

        if spatial_kv_cache is not None:
            # Inference denoising stage: return existing cache unchanged.
            # The cache is only updated in the cache writing stage.
            spatial_kv_for_cache = spatial_kv_cache
            # spatial_kv_for_cache: (P', HW, dim) — unchanged

        else:
            # Training or inference cache writing stage.
            # P is guaranteed to be an int here.
            assert P is not None  # for type checker
            P_clamped: int = max(0, min(P, num_frames))

            if P_clamped >= p_prime:
                # Normal case: return the last P' clean prefix frames.
                # x[P_clamped - P' : P_clamped]: (P', HW, dim)
                spatial_kv_for_cache = x[P_clamped - p_prime : P_clamped].detach().clone()
            elif P_clamped > 0:
                # Edge case: fewer than P' clean frames available.
                # Repeat available frames to fill P' slots.
                available: torch.Tensor = x[:P_clamped]  # (P_clamped, HW, dim)
                repeat_times: int = (p_prime + P_clamped - 1) // P_clamped
                spatial_kv_for_cache = (
                    available.repeat(repeat_times, 1, 1)[:p_prime].detach().clone()
                )
                # spatial_kv_for_cache: (P', HW, dim)
            else:
                # P_clamped == 0: no clean prefix frames.
                # Use the first frame of x as a fallback (repeated P' times).
                spatial_kv_for_cache = (
                    x[:1].expand(p_prime, -1, -1).detach().clone()
                )
                # spatial_kv_for_cache: (P', HW, dim)

        return output, spatial_kv_for_cache
