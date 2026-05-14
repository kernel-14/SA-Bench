## model/causal_temporal_attn.py
"""Causal Temporal Attention module for Ca2-VDM.

This module implements CausalTemporalAttention, the core architectural
innovation of Ca2-VDM that replaces bidirectional temporal attention with a
causally-masked variant. The causal mask ensures each frame only attends to
its prefix frames, enabling KV-cache precomputation and reuse across all
autoregressive steps and denoising timesteps (cache sharing).

Paper: Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal
Generation and Cache Sharing (Sec. 3.2, 3.3).

Key equations from the paper:
    CausalAttn(Q, K, V) = Softmax(QK^T / sqrt(C') + M) V
    where M_{i,j} = -inf if i < j else 0  (lower-triangular mask)

Configuration references (config.yaml):
    model.model_dim:  1152
    model.num_heads:  16
    model.dropout:    0.0
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTemporalAttention(nn.Module):
    """Causal multi-head temporal attention with KV-cache support.

    Replaces bidirectional temporal attention in each Transformer block with
    a causally-masked variant. The causal mask (lower-triangular) ensures
    frame ``i`` only attends to frames ``0..i``, making it possible to
    precompute and cache the keys and values of clean prefix frames once and
    reuse them across all 100 denoising steps without recomputation.

    This module is stateless with respect to caching — all KV-cache queue
    management is handled externally by ``inference.kv_cache.KVCacheQueue``
    and ``inference.ar_inference.ARInference``. The module simply accepts an
    optional ``kv_cache`` argument and returns the current input's ``(K, V)``
    pair for the caller to store or discard.

    Input layout convention:
        The spatial dimensions ``H × W`` are folded into the batch dimension
        by the calling ``DiTBlock`` before this module is invoked. This module
        receives and returns tensors of shape ``(B * H * W, L, dim)``.

    Attributes:
        dim: Model hidden dimension. From config.yaml: ``model.model_dim: 1152``.
        num_heads: Number of attention heads. From config.yaml:
                   ``model.num_heads: 16``.
        head_dim: Per-head dimension. ``dim // num_heads = 72``.
        scale: Dot-product scaling factor. ``head_dim ** -0.5``.
        W_Q: Query projection ``Linear(dim, dim)``.
        W_K: Key projection ``Linear(dim, dim)``.
        W_V: Value projection ``Linear(dim, dim)``.
        W_O: Output projection ``Linear(dim, dim)``.
        attn_dropout: Dropout applied to attention weights during training.

    Example (training)::

        attn = CausalTemporalAttention(dim=1152, num_heads=16)
        x = torch.randn(32 * 32, 33, 1152)   # (B*H*W, L, dim)
        tpe = torch.randn(33, 1152)            # (L, dim)
        out, (K, V) = attn(x, tpe=tpe, kv_cache=None)
        # out: (1024, 33, 1152)
        # K, V: (1024, 33, 1152) — current input's KVs

    Example (inference denoising stage)::

        attn = CausalTemporalAttention(dim=1152, num_heads=16)
        x_chunk = torch.randn(1024, 8, 1152)   # (B*H*W, l, dim)
        tpe_chunk = torch.randn(8, 1152)        # (l, dim)
        K_cache = torch.randn(1024, 25, 1152)  # (B*H*W, P_k, dim)
        V_cache = torch.randn(1024, 25, 1152)
        out, (K, V) = attn(x_chunk, tpe=tpe_chunk,
                           kv_cache=(K_cache, V_cache))
        # out: (1024, 8, 1152)
        # K, V: (1024, 8, 1152) — current chunk's KVs (caller discards during denoising)
    """

    def __init__(
        self,
        dim: int = 1152,
        num_heads: int = 16,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the causal temporal attention module.

        Args:
            dim: Model hidden dimension. Must be a positive integer divisible
                 by ``num_heads``. From config.yaml: ``model.model_dim: 1152``.
            num_heads: Number of attention heads. Must be a positive integer
                       that divides ``dim`` evenly. From config.yaml:
                       ``model.num_heads: 16``.
            dropout: Dropout probability applied to attention weights during
                     training. Set to 0.0 at inference. From config.yaml:
                     ``model.dropout: 0.0``.

        Raises:
            ValueError: If ``dim`` is not divisible by ``num_heads``.
            ValueError: If ``dim`` or ``num_heads`` is not a positive integer.
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
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0), got {dropout}."
            )

        self.dim: int = dim
        self.num_heads: int = num_heads
        self.head_dim: int = dim // num_heads
        # Precomputed scaling factor: 1 / sqrt(head_dim).
        # Stored as a float attribute (not a buffer) since it's a scalar constant.
        self.scale: float = self.head_dim ** -0.5

        # Linear projections: all dim → dim.
        # bias=True follows standard Transformer practice and Open-Sora v1.0.
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

    def _build_causal_mask(
        self,
        query_len: int,
        key_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build an additive causal attention mask.

        For the **square training case** (``query_len == key_len == L``):
            ``M[i, j] = -inf if j > i else 0``
            Shape: ``(L, L)`` — standard lower-triangular causal mask.

        For the **rectangular inference case** (``query_len == l``,
        ``key_len == P_k + l``):
            The query frames (current chunk) can attend freely to all
            ``P_k`` cached frames (left block = zeros) but cannot attend to
            future frames within the current chunk (right block = upper-tri).
            Shape: ``(l, P_k + l)``

            Concretely:
            - Left block ``(l, P_k)``: all zeros — chunk attends to all cache.
            - Right block ``(l, l)``: upper-triangular ``-inf`` — causal within chunk.
            - ``M[i, j] = -inf if j > (P_k + i) else 0``

        Paper (Sec. 3.2): "M ∈ R^{L×L} is a lower triangular attention mask
        with M_{i,j} = -∞ if i < j else 0."

        Args:
            query_len: Number of query frames (``L`` during training, ``l``
                       during inference denoising/cache-writing).
            key_len: Number of key/value frames (``L`` during training,
                     ``P_k + l`` during inference denoising).
            device: Target device for the mask tensor.
            dtype: Compute dtype for the mask (float32, float16, bfloat16).
                   Using the model's compute dtype avoids implicit casts in
                   the attention computation.

        Returns:
            Additive float mask of shape ``(query_len, key_len)`` with values
            in ``{0.0, -inf}``. Shape is broadcastable over the batch and
            head dimensions in the attention computation.
        """
        # Determine the number of cached (prefix) frames.
        # P_k = key_len - query_len (0 during training, >= 0 during inference).
        p_k: int = key_len - query_len

        if p_k < 0:
            raise ValueError(
                f"key_len ({key_len}) must be >= query_len ({query_len}). "
                f"Got P_k = {p_k} < 0."
            )

        # Build the causal mask for the current chunk (right block).
        # Shape: (query_len, query_len) — upper-triangular -inf.
        # torch.triu with diagonal=1 sets elements above the main diagonal to 1.
        causal_block: torch.Tensor = torch.triu(
            torch.ones(query_len, query_len, device=device, dtype=dtype),
            diagonal=1,
        ) * float("-inf")
        # causal_block[i, j] = -inf if j > i else 0.0

        if p_k == 0:
            # Square case (training or cache-writing with no prior cache):
            # mask is just the causal block.
            return causal_block  # (query_len, query_len)

        # Rectangular case (inference denoising stage with P_k > 0):
        # Left block: all zeros — chunk frames attend freely to all cached frames.
        zeros_block: torch.Tensor = torch.zeros(
            query_len, p_k, device=device, dtype=dtype
        )
        # Concatenate: [zeros_block | causal_block] along key dimension.
        mask: torch.Tensor = torch.cat([zeros_block, causal_block], dim=1)
        # mask: (query_len, P_k + query_len) = (query_len, key_len)
        return mask

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape a projected tensor into multi-head format.

        Splits the ``dim``-dimensional feature axis into ``num_heads`` heads,
        each of dimension ``head_dim``.

        Args:
            x: Tensor of shape ``(B, L, dim)`` where ``B = batch * H * W``.

        Returns:
            Tensor of shape ``(B, num_heads, L, head_dim)``.
        """
        B, L, _ = x.shape
        # Reshape: (B, L, dim) → (B, L, num_heads, head_dim)
        x = x.reshape(B, L, self.num_heads, self.head_dim)
        # Transpose: (B, L, num_heads, head_dim) → (B, num_heads, L, head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of ``_split_heads``: merge multi-head format back to ``dim``.

        Args:
            x: Tensor of shape ``(B, num_heads, L, head_dim)``.

        Returns:
            Tensor of shape ``(B, L, dim)``.
        """
        B, _, L, _ = x.shape
        # Transpose: (B, num_heads, L, head_dim) → (B, L, num_heads, head_dim)
        x = x.transpose(1, 2)
        # contiguous() is required before reshape when the tensor is non-contiguous
        # after transpose (common in PyTorch).
        x = x.contiguous()
        # Reshape: (B, L, num_heads, head_dim) → (B, L, dim)
        return x.reshape(B, L, self.dim)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        tpe: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute causal temporal attention with optional KV-cache.

        Handles three distinct operational modes:

        **Training (``kv_cache=None``, full sequence):**
            - ``x``: ``(B*H*W, L, dim)`` — full training clip (prefix + target).
            - Applies full causal mask of shape ``(L, L)``.
            - Returns current ``(K, V)`` for potential use by the caller.

        **Inference denoising stage (``kv_cache=(K_cache, V_cache)``):**
            - ``x``: ``(B*H*W, l, dim)`` — only the current ``l``-frame noisy chunk.
            - ``K_cache``, ``V_cache``: ``(B*H*W, P_k, dim)`` — cached clean KVs.
            - Concatenates cache with current KVs: full key/value length = ``P_k + l``.
            - Applies rectangular mask of shape ``(l, P_k + l)``.
            - Returns current chunk's ``(K, V)`` — caller discards these during
              denoising (they are noisy and timestep-dependent).

        **Inference cache writing stage (``kv_cache=None``, clean chunk):**
            - ``x``: ``(B*H*W, l, dim)`` — denoised clean chunk (``tEmb(0)``).
            - Applies causal mask of shape ``(l, l)``.
            - Returns current chunk's ``(K, V)`` — caller stores these in the
              ``KVCacheQueue`` for use in subsequent AR steps.

        Args:
            x: Input tensor of shape ``(B*H*W, L, dim)`` where ``B*H*W`` is
               the effective batch size (spatial dimensions folded in by the
               calling ``DiTBlock``). ``L`` is the sequence length:
               - Training: full clip length ``l_train`` (33 or 65).
               - Inference: chunk length ``l`` (8 or 16).
            tpe: Optional temporal positional embeddings of shape ``(L, dim)``.
                 Broadcast over the batch dimension ``B*H*W``. Added to ``x``
                 before computing Q, K, V so that TPE is baked into the cached
                 K and V features (consistent with Cyclic-TPE design).
                 If ``None``, no positional information is added.
            kv_cache: Optional tuple ``(K_cache, V_cache)`` of cached clean
                      key and value tensors, each of shape
                      ``(B*H*W, P_k, dim)``. If provided, the cached KVs are
                      prepended to the current KVs before attention.
                      If ``None``, no cache is used (training or cache writing).

        Returns:
            A tuple ``(output, (K_current, V_current))`` where:
            - ``output``: Attended output of shape ``(B*H*W, L, dim)``.
            - ``K_current``: Keys computed from the current input ``x`` only
              (before cache concatenation), shape ``(B*H*W, L, dim)``.
            - ``V_current``: Values computed from the current input ``x`` only
              (before cache concatenation), shape ``(B*H*W, L, dim)``.

            The caller is responsible for deciding whether to store
            ``(K_current, V_current)`` in the KV-cache queue.

        Raises:
            ValueError: If ``x`` does not have exactly 3 dimensions.
            ValueError: If ``tpe`` shape does not match ``(L, dim)``.
            ValueError: If ``kv_cache`` tensors have incompatible shapes.
        """
        if x.ndim != 3:
            raise ValueError(
                f"x must be a 3-D tensor (B*H*W, L, dim), got shape {tuple(x.shape)}."
            )

        b_hw, seq_len, _ = x.shape

        # ── Step 1: Validate and add temporal positional embeddings ──────────
        if tpe is not None:
            if tpe.shape != (seq_len, self.dim):
                raise ValueError(
                    f"tpe shape must be ({seq_len}, {self.dim}), "
                    f"got {tuple(tpe.shape)}."
                )
            # Add TPE before projection: bakes positional info into K and V,
            # which is essential for the Cyclic-TPE cache sharing mechanism.
            # tpe: (L, dim) broadcasts over batch dimension (B*H*W).
            x = x + tpe.unsqueeze(0)  # (B*H*W, L, dim)

        # ── Step 2: Compute Q, K, V from current input ───────────────────────
        # All projections: (B*H*W, L, dim) → (B*H*W, L, dim)
        Q: torch.Tensor = self.W_Q(x)
        K_current: torch.Tensor = self.W_K(x)
        V_current: torch.Tensor = self.W_V(x)
        # K_current and V_current are the "current input's KVs" that will be
        # returned to the caller for optional caching.

        # ── Step 3: KV-cache concatenation (inference denoising stage only) ──
        if kv_cache is not None:
            K_cache, V_cache = kv_cache

            # Validate cache shapes.
            if K_cache.ndim != 3 or V_cache.ndim != 3:
                raise ValueError(
                    f"kv_cache tensors must be 3-D (B*H*W, P_k, dim), "
                    f"got K_cache shape {tuple(K_cache.shape)}, "
                    f"V_cache shape {tuple(V_cache.shape)}."
                )
            if K_cache.shape[0] != b_hw or V_cache.shape[0] != b_hw:
                raise ValueError(
                    f"kv_cache batch dimension must match x batch dimension "
                    f"({b_hw}), got K_cache.shape[0]={K_cache.shape[0]}, "
                    f"V_cache.shape[0]={V_cache.shape[0]}."
                )
            if K_cache.shape[2] != self.dim or V_cache.shape[2] != self.dim:
                raise ValueError(
                    f"kv_cache feature dimension must be {self.dim}, "
                    f"got K_cache.shape[2]={K_cache.shape[2]}, "
                    f"V_cache.shape[2]={V_cache.shape[2]}."
                )

            # Prepend cached clean KVs to current (noisy) KVs.
            # K_cache: (B*H*W, P_k, dim), K_current: (B*H*W, l, dim)
            # K_full:  (B*H*W, P_k + l, dim)
            K_full: torch.Tensor = torch.cat([K_cache, K_current], dim=1)
            V_full: torch.Tensor = torch.cat([V_cache, V_current], dim=1)
        else:
            # Training or cache writing: no cache, use current KVs only.
            K_full = K_current
            V_full = V_current

        # ── Step 4: Split into multi-head format ─────────────────────────────
        # Q:      (B*H*W, L_q, dim)     → (B*H*W, num_heads, L_q, head_dim)
        # K_full: (B*H*W, L_kv, dim)    → (B*H*W, num_heads, L_kv, head_dim)
        # V_full: (B*H*W, L_kv, dim)    → (B*H*W, num_heads, L_kv, head_dim)
        # where L_q = seq_len, L_kv = P_k + seq_len (or seq_len if no cache)
        Q_heads: torch.Tensor = self._split_heads(Q)
        K_heads: torch.Tensor = self._split_heads(K_full)
        V_heads: torch.Tensor = self._split_heads(V_full)

        key_len: int = K_full.shape[1]  # P_k + seq_len or seq_len

        # ── Step 5: Build causal attention mask ──────────────────────────────
        mask: torch.Tensor = self._build_causal_mask(
            query_len=seq_len,
            key_len=key_len,
            device=x.device,
            dtype=x.dtype,
        )
        # mask: (seq_len, key_len) — broadcastable over (B*H*W, num_heads, ...)

        # ── Step 6: Scaled dot-product attention ─────────────────────────────
        # Use F.scaled_dot_product_attention (PyTorch 2.0+) for efficiency.
        # It handles the scaling (1/sqrt(head_dim)) internally.
        # The attn_mask argument accepts additive float masks with -inf values.
        dropout_p: float = self._dropout_p if self.training else 0.0

        # Check if F.scaled_dot_product_attention is available (PyTorch >= 2.0).
        _use_sdpa: bool = hasattr(F, "scaled_dot_product_attention")

        if _use_sdpa:
            # F.scaled_dot_product_attention expects attn_mask to be broadcastable
            # to (B, num_heads, L_q, L_kv). Our mask is (L_q, L_kv) which
            # broadcasts correctly over the first two dimensions.
            attn_out: torch.Tensor = F.scaled_dot_product_attention(
                Q_heads,
                K_heads,
                V_heads,
                attn_mask=mask,
                dropout_p=dropout_p,
                scale=self.scale,
            )
            # attn_out: (B*H*W, num_heads, seq_len, head_dim)
        else:
            # Manual implementation for PyTorch < 2.0 compatibility.
            # attn_weights: (B*H*W, num_heads, seq_len, key_len)
            attn_weights: torch.Tensor = (
                torch.matmul(Q_heads, K_heads.transpose(-2, -1)) * self.scale
            )
            # Add causal mask (broadcasts over batch and head dims).
            attn_weights = attn_weights + mask
            # Softmax over key dimension.
            attn_weights = F.softmax(attn_weights, dim=-1)
            # Apply dropout to attention weights (training only).
            attn_weights = self.attn_dropout(attn_weights)
            # Weighted sum of values.
            attn_out = torch.matmul(attn_weights, V_heads)
            # attn_out: (B*H*W, num_heads, seq_len, head_dim)

        # ── Step 7: Merge heads and apply output projection ──────────────────
        # (B*H*W, num_heads, seq_len, head_dim) → (B*H*W, seq_len, dim)
        attn_out = self._merge_heads(attn_out)
        # Output projection: (B*H*W, seq_len, dim) → (B*H*W, seq_len, dim)
        output: torch.Tensor = self.W_O(attn_out)

        # ── Step 8: Return output and current input's KVs ────────────────────
        # K_current and V_current are the KVs computed from the current input x
        # (before any cache concatenation). The caller decides whether to store
        # them in the KVCacheQueue:
        # - Denoising stage: caller discards (noisy, timestep-dependent).
        # - Cache writing stage: caller stores in KVCacheQueue.
        return output, (K_current, V_current)
