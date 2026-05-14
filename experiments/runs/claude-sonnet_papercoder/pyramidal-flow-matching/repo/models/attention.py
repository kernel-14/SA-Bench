## models/attention.py
"""Blockwise causal attention for Pyramidal Flow Matching.

Implements the BlockwiseCausalAttention module used inside every MMDiTBlock.
Enforces the paper's autoregressive video generation constraint (Section 3.4):
tokens within the same frame attend bidirectionally, but tokens in frame i
cannot attend to tokens in any frame j > i (causal across frames).

The ablation study (Appendix C.2) confirms this design is critical:
bidirectional attention across frames causes temporal incoherence.

Supports three attention backends in priority order:
    1. Flash Attention (flash_attn) — most memory-efficient
    2. xFormers memory-efficient attention — supports arbitrary masks
    3. PyTorch scaled_dot_product_attention — universal fallback

Usage:
    from models.attention import BlockwiseCausalAttention

    attn = BlockwiseCausalAttention(num_heads=16, head_dim=72)

    # Build causal mask from frame token counts
    mask = BlockwiseCausalAttention.build_causal_mask(
        frame_token_counts=[256, 256, 256]
    )

    # Self-attention with causal mask
    out = attn(x=visual_tokens, context=None, attn_mask=mask)

    # Cross-attention (text conditioning, no causal mask)
    out = attn(x=visual_tokens, context=text_embeddings, attn_mask=None)
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.logging import get_logger

## ---------------------------------------------------------------------------
## Module-level logger
## ---------------------------------------------------------------------------
logger = get_logger(__name__)

## ---------------------------------------------------------------------------
## Backend availability detection (module-level, done once at import time)
## ---------------------------------------------------------------------------
_FLASH_ATTN_AVAILABLE: bool = False
_XFORMERS_AVAILABLE: bool = False

try:
    import flash_attn  # type: ignore[import]
    from flash_attn import flash_attn_func  # type: ignore[import]
    _FLASH_ATTN_AVAILABLE = True
    logger.info("Flash Attention available — will use as primary backend.")
except ImportError:
    logger.info(
        "flash_attn not available. "
        "Install with: pip install flash-attn==2.5.8"
    )

try:
    import xformers.ops as xops  # type: ignore[import]
    _XFORMERS_AVAILABLE = True
    logger.info("xFormers available — will use as secondary backend.")
except ImportError:
    logger.info(
        "xformers not available. "
        "Install with: pip install xformers==0.0.23"
    )

if not _FLASH_ATTN_AVAILABLE and not _XFORMERS_AVAILABLE:
    logger.info(
        "Using PyTorch scaled_dot_product_attention as attention backend."
    )


class BlockwiseCausalAttention(nn.Module):
    """Multi-head attention with blockwise causal masking for video generation.

    Implements the attention mechanism described in Section 3.4 of the paper:
    "blockwise causal attention is adopted in each transformer layer, ensuring
    that each token cannot attend to its subsequent frames."

    The 'blockwise' structure means:
    - Within a frame: fully bidirectional (all tokens attend to each other)
    - Across frames: strictly causal (frame i can attend to frames 0..i only)

    Supports both self-attention (visual tokens attending to themselves) and
    cross-attention (visual tokens attending to text context tokens). The
    causal mask is only applied during self-attention; cross-attention is
    always fully bidirectional.

    Attributes:
        num_heads: Number of attention heads (16 from config).
        head_dim: Dimension per attention head (72 from config).
        embed_dim: Total embedding dimension = num_heads * head_dim (1152).
        context_dim: Dimension of context (text) embeddings for cross-attention.
        dropout: Dropout probability for attention weights.
        use_flash_attn: Whether flash-attn backend is available and active.
        use_xformers: Whether xFormers backend is available and active.
        q_proj: Linear projection for queries.
        k_proj: Linear projection for keys.
        v_proj: Linear projection for values.
        out_proj: Linear projection for output.
    """

    def __init__(
        self,
        num_heads: int = 16,
        head_dim: int = 72,
        context_dim: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        """Initializes BlockwiseCausalAttention.

        Args:
            num_heads: Number of attention heads. From config.model.num_heads.
                Defaults to 16 (SD3 Medium / MM-DiT standard).
            head_dim: Dimension per attention head. From config.model.head_dim.
                Defaults to 72 (= 1152 // 16).
            context_dim: Dimension of context embeddings for cross-attention.
                Set to T5-XXL embedding dimension (4096) for text conditioning.
                Defaults to 4096.
            dropout: Dropout probability applied to attention weights during
                training. Defaults to 0.0 (no dropout).

        Raises:
            ValueError: If num_heads <= 0 or head_dim <= 0.
        """
        super().__init__()

        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be positive, got num_heads={num_heads}."
            )
        if head_dim <= 0:
            raise ValueError(
                f"head_dim must be positive, got head_dim={head_dim}."
            )

        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.embed_dim: int = num_heads * head_dim
        self.context_dim: int = context_dim
        self.dropout: float = dropout
        self.scale: float = math.sqrt(head_dim)  # 1/sqrt(d_k) scaling factor

        # ----------------------------------------------------------------
        # Self-attention projections (Q, K, V from visual tokens)
        # ----------------------------------------------------------------
        # Q always comes from the visual stream (x)
        self.q_proj: nn.Linear = nn.Linear(
            self.embed_dim, self.embed_dim, bias=False
        )
        # K and V come from x (self-attn) or context (cross-attn)
        # For self-attention: k_proj and v_proj accept embed_dim
        # For cross-attention: k_proj and v_proj accept context_dim
        # We implement this by having two sets of K/V projections and
        # selecting based on whether context is provided.
        self.k_proj: nn.Linear = nn.Linear(
            self.embed_dim, self.embed_dim, bias=False
        )
        self.v_proj: nn.Linear = nn.Linear(
            self.embed_dim, self.embed_dim, bias=False
        )

        # Cross-attention K/V projections (accept context_dim input)
        self.k_proj_context: nn.Linear = nn.Linear(
            self.context_dim, self.embed_dim, bias=False
        )
        self.v_proj_context: nn.Linear = nn.Linear(
            self.context_dim, self.embed_dim, bias=False
        )

        # ----------------------------------------------------------------
        # Output projection
        # ----------------------------------------------------------------
        self.out_proj: nn.Linear = nn.Linear(
            self.embed_dim, self.embed_dim, bias=False
        )

        # ----------------------------------------------------------------
        # Attention dropout (applied to attention weights, not output)
        # ----------------------------------------------------------------
        self.attn_drop: nn.Dropout = nn.Dropout(p=dropout)

        # ----------------------------------------------------------------
        # Backend selection (determined at module-level import time)
        # ----------------------------------------------------------------
        # Flash-attn is preferred but has limited mask support.
        # For blockwise causal masks, we use xFormers or SDPA.
        # Flash-attn is used only for unconstrained attention (mask=None).
        self.use_flash_attn: bool = _FLASH_ATTN_AVAILABLE
        self.use_xformers: bool = _XFORMERS_AVAILABLE and not _FLASH_ATTN_AVAILABLE

        # ----------------------------------------------------------------
        # Weight initialization (following SD3 / DiT conventions)
        # ----------------------------------------------------------------
        self._init_weights()

        logger.debug(
            "BlockwiseCausalAttention initialized: num_heads=%d, head_dim=%d, "
            "embed_dim=%d, context_dim=%d, dropout=%.3f, "
            "backend=%s",
            self.num_heads,
            self.head_dim,
            self.embed_dim,
            self.context_dim,
            self.dropout,
            "flash_attn" if self.use_flash_attn else
            ("xformers" if self.use_xformers else "sdpa"),
        )

    def _init_weights(self) -> None:
        """Initializes projection weights using Xavier uniform initialization.

        Follows the DiT / SD3 convention of Xavier uniform for attention
        projections, which helps with training stability at 2B parameter scale.
        """
        for module in [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.k_proj_context,
            self.v_proj_context,
            self.out_proj,
        ]:
            nn.init.xavier_uniform_(module.weight)

    @staticmethod
    def build_causal_mask(
        frame_token_counts: List[int],
        sample_token_counts: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Constructs the blockwise causal attention mask.

        Builds a boolean tensor where mask[i, j] = True means token i
        CAN attend to token j. The blockwise causal structure is:
        - Same frame: bidirectional (True in both directions)
        - Earlier frame: token i can see token j (True), but not vice versa
        - Later frame: blocked (False)

        When sample_token_counts is provided (Patch n' Pack), cross-sample
        attention is also blocked: tokens from different packed samples
        cannot attend to each other, regardless of frame ordering.

        Args:
            frame_token_counts: List of token counts per frame within a
                single sample (or the full packed sequence if
                sample_token_counts is None). Example: [256, 256, 256]
                for 3 frames of 256 tokens each.
                For packed sequences, this is the concatenated list across
                all samples: [256, 256, 128, 128] for two samples with
                2 frames each.
            sample_token_counts: Optional list of total token counts per
                packed sample. Used to block cross-sample attention in
                Patch n' Pack batches. Example: [512, 256] means the first
                512 tokens belong to sample 0 and the next 256 to sample 1.
                If None, all tokens are treated as belonging to one sample.
            device: Target device for the output mask tensor. Defaults to
                CPU (moved to GPU in forward()).

        Returns:
            Boolean tensor of shape [seq_len, seq_len] where seq_len =
            sum(frame_token_counts). True = can attend, False = blocked.
            The tensor is on the specified device (or CPU if device=None).

        Example:
            >>> # 3 frames, 4 tokens each
            >>> mask = BlockwiseCausalAttention.build_causal_mask([4, 4, 4])
            >>> mask.shape
            torch.Size([12, 12])
            >>> # Frame 0 tokens (0-3) can only attend to frame 0
            >>> mask[0, 4].item()  # token 0 attending to frame 1 token
            False
            >>> # Frame 1 tokens (4-7) can attend to frames 0 and 1
            >>> mask[4, 0].item()  # token 4 attending to frame 0 token
            True
            >>> # Within-frame: bidirectional
            >>> mask[0, 3].item()  # token 0 attending to token 3 (same frame)
            True
        """
        if device is None:
            device = torch.device("cpu")

        seq_len: int = sum(frame_token_counts)

        if seq_len == 0:
            return torch.ones(0, 0, dtype=torch.bool, device=device)

        # ----------------------------------------------------------------
        # Step 1: Build frame_id tensor [seq_len]
        # Maps each token position to its frame index.
        # ----------------------------------------------------------------
        # frame_token_counts = [n0, n1, n2, ...]
        # frame_ids = [0,0,...,0, 1,1,...,1, 2,2,...,2, ...]
        #              |---n0---|  |---n1---|  |---n2---|
        frame_ids: Tensor = torch.repeat_interleave(
            torch.arange(
                len(frame_token_counts),
                dtype=torch.long,
                device=device,
            ),
            torch.tensor(frame_token_counts, dtype=torch.long, device=device),
        )  # [seq_len]

        # ----------------------------------------------------------------
        # Step 2: Build causal mask from frame_ids
        # mask[i, j] = True iff frame_id[j] <= frame_id[i]
        # ----------------------------------------------------------------
        # Broadcast: frame_ids[i] >= frame_ids[j]
        # frame_ids.unsqueeze(1): [seq_len, 1]
        # frame_ids.unsqueeze(0): [1, seq_len]
        causal_mask: Tensor = (
            frame_ids.unsqueeze(1) >= frame_ids.unsqueeze(0)
        )  # [seq_len, seq_len]

        # ----------------------------------------------------------------
        # Step 3: Apply cross-sample blocking (Patch n' Pack)
        # ----------------------------------------------------------------
        if sample_token_counts is not None and len(sample_token_counts) > 1:
            total_from_samples: int = sum(sample_token_counts)
            if total_from_samples != seq_len:
                logger.warning(
                    "sum(sample_token_counts)=%d != sum(frame_token_counts)=%d. "
                    "Cross-sample blocking may be incorrect. "
                    "Ensure sample_token_counts sums to the same total as "
                    "frame_token_counts.",
                    total_from_samples,
                    seq_len,
                )

            # Build sample_id tensor [seq_len]
            sample_ids: Tensor = torch.repeat_interleave(
                torch.arange(
                    len(sample_token_counts),
                    dtype=torch.long,
                    device=device,
                ),
                torch.tensor(
                    sample_token_counts, dtype=torch.long, device=device
                ),
            )  # [seq_len]

            # Cross-sample mask: True iff same sample
            # sample_ids[i] == sample_ids[j]
            same_sample_mask: Tensor = (
                sample_ids.unsqueeze(1) == sample_ids.unsqueeze(0)
            )  # [seq_len, seq_len]

            # Combined mask: causal AND same sample
            causal_mask = causal_mask & same_sample_mask

        return causal_mask  # [seq_len, seq_len], dtype=bool

    def _bool_mask_to_additive(
        self,
        bool_mask: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        """Converts a boolean attention mask to an additive float mask.

        PyTorch's scaled_dot_product_attention and xFormers expect an
        additive mask where:
        - 0.0 means "can attend" (no penalty)
        - -inf means "cannot attend" (effectively zero after softmax)

        Args:
            bool_mask: Boolean tensor [seq_len, seq_len] where True = can
                attend. From build_causal_mask().
            dtype: Target float dtype (e.g., torch.bfloat16 during training).
            device: Target device.

        Returns:
            Float tensor [seq_len, seq_len] with 0.0 where bool_mask is
            True and -inf where bool_mask is False.
        """
        # Start with zeros (all positions can attend)
        additive_mask: Tensor = torch.zeros(
            bool_mask.shape,
            dtype=dtype,
            device=device,
        )
        # Set blocked positions to -inf
        additive_mask = additive_mask.masked_fill(
            ~bool_mask.to(device=device),
            float("-inf"),
        )
        return additive_mask

    def _attention_flash(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        """Computes attention using Flash Attention (no mask support).

        Flash Attention is used only when attn_mask is None (unconstrained
        attention). For blockwise causal attention, we fall back to SDPA.

        Flash-attn expects inputs in shape [B, seq_len, num_heads, head_dim].

        Args:
            q: Query tensor [B, num_heads, seq_len, head_dim].
            k: Key tensor [B, num_heads, k_len, head_dim].
            v: Value tensor [B, num_heads, v_len, head_dim].

        Returns:
            Output tensor [B, num_heads, seq_len, head_dim].
        """
        # Flash-attn expects [B, seq_len, num_heads, head_dim]
        # Our tensors are [B, num_heads, seq_len, head_dim] — transpose
        q_fa: Tensor = q.transpose(1, 2)  # [B, seq_len, num_heads, head_dim]
        k_fa: Tensor = k.transpose(1, 2)
        v_fa: Tensor = v.transpose(1, 2)

        # flash_attn_func signature:
        # flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
        dropout_p: float = self.dropout if self.training else 0.0
        out_fa: Tensor = flash_attn_func(  # type: ignore[name-defined]
            q_fa,
            k_fa,
            v_fa,
            dropout_p=dropout_p,
            softmax_scale=1.0 / self.scale,
            causal=False,  # We handle causality via explicit mask
        )  # [B, seq_len, num_heads, head_dim]

        # Transpose back to [B, num_heads, seq_len, head_dim]
        return out_fa.transpose(1, 2)

    def _attention_xformers(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes attention using xFormers memory-efficient attention.

        xFormers supports arbitrary attention biases (additive masks),
        making it suitable for the blockwise causal mask.

        xFormers expects inputs in shape [B, seq_len, num_heads, head_dim].

        Args:
            q: Query tensor [B, num_heads, seq_len, head_dim].
            k: Key tensor [B, num_heads, k_len, head_dim].
            v: Value tensor [B, num_heads, v_len, head_dim].
            attn_bias: Optional additive attention bias [seq_len, k_len]
                or [B, num_heads, seq_len, k_len]. 0.0 = attend, -inf = block.

        Returns:
            Output tensor [B, num_heads, seq_len, head_dim].
        """
        # xFormers expects [B, seq_len, num_heads, head_dim]
        q_xf: Tensor = q.transpose(1, 2)
        k_xf: Tensor = k.transpose(1, 2)
        v_xf: Tensor = v.transpose(1, 2)

        # xFormers attn_bias must be [B, num_heads, seq_len, k_len] or broadcastable
        # If attn_bias is [seq_len, k_len], expand to [1, 1, seq_len, k_len]
        xf_bias: Optional[Tensor] = None
        if attn_bias is not None:
            if attn_bias.dim() == 2:
                xf_bias = attn_bias.unsqueeze(0).unsqueeze(0)
            else:
                xf_bias = attn_bias

        scale: float = 1.0 / self.scale
        out_xf: Tensor = xops.memory_efficient_attention(  # type: ignore[name-defined]
            q_xf,
            k_xf,
            v_xf,
            attn_bias=xf_bias,
            scale=scale,
            p=self.dropout if self.training else 0.0,
        )  # [B, seq_len, num_heads, head_dim]

        return out_xf.transpose(1, 2)

    def _attention_sdpa(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes attention using PyTorch scaled_dot_product_attention.

        Universal fallback that works with any mask shape and dtype.
        Supports bfloat16 natively on modern hardware.

        Args:
            q: Query tensor [B, num_heads, seq_len, head_dim].
            k: Key tensor [B, num_heads, k_len, head_dim].
            v: Value tensor [B, num_heads, v_len, head_dim].
            attn_mask: Optional additive float mask [seq_len, k_len] or
                [B, num_heads, seq_len, k_len]. 0.0 = attend, -inf = block.
                Can also be a bool mask (SDPA handles both).

        Returns:
            Output tensor [B, num_heads, seq_len, head_dim].
        """
        dropout_p: float = self.dropout if self.training else 0.0

        out: Tensor = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,  # We provide explicit mask
            scale=1.0 / self.scale,
        )  # [B, num_heads, seq_len, head_dim]

        return out

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Computes blockwise causal attention.

        Supports both self-attention (context=None) and cross-attention
        (context provided for text conditioning). The causal mask is only
        meaningful for self-attention; cross-attention ignores it.

        Args:
            x: Visual token sequence of shape [B, seq_len, embed_dim].
                Contains packed latent tokens from all frames at the current
                pyramid stage. embed_dim = num_heads * head_dim = 1152.
            context: Optional text context tensor of shape
                [B, context_seq_len, context_dim] for cross-attention.
                When None, performs self-attention on x.
                When provided, Q comes from x and K/V come from context.
                context_dim must match self.context_dim (4096 for T5-XXL).
            attn_mask: Optional boolean attention mask of shape
                [seq_len, seq_len] from build_causal_mask(). True = can
                attend, False = blocked. Only applied during self-attention
                (when context is None). Pass None for image-only batches
                (single frame, no causal constraint needed) or for
                cross-attention.

        Returns:
            Output tensor of shape [B, seq_len, embed_dim] containing
            the attended and projected visual token representations.

        Raises:
            ValueError: If x has wrong number of dimensions or embed_dim
                does not match self.embed_dim.
            ValueError: If context is provided but its last dimension does
                not match self.context_dim.

        Example:
            >>> attn = BlockwiseCausalAttention(num_heads=16, head_dim=72)
            >>> x = torch.randn(2, 768, 1152)  # B=2, 3 frames * 256 tokens
            >>> mask = BlockwiseCausalAttention.build_causal_mask([256, 256, 256])
            >>> out = attn(x, attn_mask=mask)
            >>> out.shape
            torch.Size([2, 768, 1152])
        """
        # ----------------------------------------------------------------
        # Input validation
        # ----------------------------------------------------------------
        if x.dim() != 3:
            raise ValueError(
                f"x must be 3D [B, seq_len, embed_dim], got shape {tuple(x.shape)}."
            )

        batch_size: int = x.shape[0]
        seq_len: int = x.shape[1]
        input_dim: int = x.shape[2]

        if input_dim != self.embed_dim:
            raise ValueError(
                f"x.shape[-1]={input_dim} does not match "
                f"self.embed_dim={self.embed_dim}. "
                f"Ensure x has the correct embedding dimension."
            )

        if context is not None:
            if context.dim() != 3:
                raise ValueError(
                    f"context must be 3D [B, context_len, context_dim], "
                    f"got shape {tuple(context.shape)}."
                )
            context_input_dim: int = context.shape[2]
            if context_input_dim != self.context_dim:
                raise ValueError(
                    f"context.shape[-1]={context_input_dim} does not match "
                    f"self.context_dim={self.context_dim}. "
                    f"Ensure context has the correct dimension (4096 for T5-XXL)."
                )

        # ----------------------------------------------------------------
        # Step 1: Compute Q, K, V projections
        # ----------------------------------------------------------------
        # Q always comes from x (visual tokens)
        q: Tensor = self.q_proj(x)  # [B, seq_len, embed_dim]

        if context is None:
            # Self-attention: K and V from x
            k: Tensor = self.k_proj(x)   # [B, seq_len, embed_dim]
            v: Tensor = self.v_proj(x)   # [B, seq_len, embed_dim]
            k_len: int = seq_len
        else:
            # Cross-attention: K and V from context (text embeddings)
            k = self.k_proj_context(context)  # [B, context_len, embed_dim]
            v = self.v_proj_context(context)  # [B, context_len, embed_dim]
            k_len = context.shape[1]

        # ----------------------------------------------------------------
        # Step 2: Reshape to multi-head format
        # [B, seq_len, embed_dim] -> [B, num_heads, seq_len, head_dim]
        # ----------------------------------------------------------------
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, num_heads, seq_len, head_dim]

        k = k.reshape(batch_size, k_len, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)  # [B, num_heads, k_len, head_dim]

        v = v.reshape(batch_size, k_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # [B, num_heads, k_len, head_dim]

        # ----------------------------------------------------------------
        # Step 3: Prepare attention mask
        # Only apply causal mask for self-attention (context is None).
        # Cross-attention is always fully bidirectional.
        # ----------------------------------------------------------------
        effective_mask: Optional[Tensor] = None

        if attn_mask is not None and context is None:
            # Convert bool mask to additive float mask for SDPA / xFormers
            # bool_mask: [seq_len, seq_len], True = can attend
            # additive_mask: [seq_len, seq_len], 0.0 = attend, -inf = block
            effective_mask = self._bool_mask_to_additive(
                bool_mask=attn_mask,
                dtype=q.dtype,
                device=q.device,
            )  # [seq_len, seq_len]

            # Expand to [1, 1, seq_len, seq_len] for broadcasting over
            # batch and head dimensions in SDPA
            effective_mask = effective_mask.unsqueeze(0).unsqueeze(0)
            # [1, 1, seq_len, seq_len]

        # ----------------------------------------------------------------
        # Step 4: Compute attention using the best available backend
        # ----------------------------------------------------------------
        # Backend selection logic:
        # - Flash-attn: only when no mask (mask=None), most memory-efficient
        # - xFormers: when mask is needed, supports arbitrary biases
        # - SDPA: universal fallback
        #
        # Note: Flash-attn's causal=True implements standard lower-triangular
        # causality (token-level), not blockwise causality. So we cannot use
        # flash-attn's built-in causal mode for our blockwise mask.
        # We use flash-attn only for unconstrained attention (mask=None).

        if self.use_flash_attn and effective_mask is None:
            # Flash-attn path: no mask, most efficient
            out: Tensor = self._attention_flash(q, k, v)
        elif self.use_xformers:
            # xFormers path: supports additive mask
            out = self._attention_xformers(q, k, v, attn_bias=effective_mask)
        else:
            # PyTorch SDPA fallback: universal, supports bool and float masks
            out = self._attention_sdpa(q, k, v, attn_mask=effective_mask)

        # out: [B, num_heads, seq_len, head_dim]

        # ----------------------------------------------------------------
        # Step 5: Reshape output and apply output projection
        # [B, num_heads, seq_len, head_dim] -> [B, seq_len, embed_dim]
        # ----------------------------------------------------------------
        out = out.transpose(1, 2)  # [B, seq_len, num_heads, head_dim]
        out = out.reshape(batch_size, seq_len, self.embed_dim)
        # [B, seq_len, embed_dim]

        out = self.out_proj(out)  # [B, seq_len, embed_dim]

        return out

    def extra_repr(self) -> str:
        """Returns a string representation of the module's configuration.

        Used by PyTorch's print(module) for debugging.

        Returns:
            String describing key hyperparameters.
        """
        backend: str = (
            "flash_attn" if self.use_flash_attn else
            ("xformers" if self.use_xformers else "sdpa")
        )
        return (
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"embed_dim={self.embed_dim}, context_dim={self.context_dim}, "
            f"dropout={self.dropout}, backend={backend}"
        )
