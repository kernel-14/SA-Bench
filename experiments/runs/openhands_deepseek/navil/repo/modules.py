"""Compound modules for NaViL.

Modules: VisualEncoderLayer, VisualEncoder, MoEDecoderLayer,
MoEDecoder (LLM with MoE), MHA-MMoE, FFN-MMoE, MLPProjector,
PixelShuffleConnector.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (
    MultiHeadAttention,
    ModalityMultiHeadAttention,
    ModalitySwiGLUFFN,
    PatchEmbedding,
    RMSNorm,
    SwiGLUFFN,
)


class TransformerLayer(nn.Module):
    """Standard transformer decoder layer (used in visual encoder)."""

    def __init__(self, dim: int, n_heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, dropout, use_rope=True, rope_type="2d")
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, mlp_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis_h: torch.Tensor,
        freqs_cis_w: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            freqs_cis_h=freqs_cis_h,
            freqs_cis_w=freqs_cis_w,
            height=height,
            width=width,
        )
        x = x + self.ffn(self.norm2(x))
        return x


class VisualEncoder(nn.Module):
    """Bidirectional visual encoder.

    Stack of TransformerLayers with 2D RoPE and bidirectional attention.
    Architecture uses parameters matching the LLM block style.

    Config: depth d, width w, MLP width, n_heads.
    Parameter count approx: 12 * d * w^2.
    """

    def __init__(
        self,
        depth: int,
        width: int,
        mlp_width: int,
        n_heads: int,
        patch_size: int = 16,
        max_image_size: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        self.patch_size = patch_size
        self.max_image_size = max_image_size

        self.patch_embed = PatchEmbedding(patch_size, 3, width)

        max_h = max_image_size // patch_size
        max_w = max_image_size // patch_size
        from layers import precompute_freqs_cis_2d
        head_dim = width // n_heads
        self.freqs_cis_h, self.freqs_cis_w = precompute_freqs_cis_2d(
            head_dim, max_h, max_w,
        )
        self.register_buffer("_freqs_cis_h", self.freqs_cis_h, persistent=False)
        self.register_buffer("_freqs_cis_w", self.freqs_cis_w, persistent=False)

        self.layers = nn.ModuleList([
            TransformerLayer(width, n_heads, mlp_width, dropout)
            for _ in range(depth)
        ])

        self.final_norm = RMSNorm(width)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Encode image to visual tokens.

        Args:
            x: [batch, 3, H, W]

        Returns:
            tokens: [batch, h*w, width]
            h, w: feature map height and width
        """
        x, h, w = self.patch_embed(x)
        freqs_cis_h = self._freqs_cis_h.to(x.device)
        freqs_cis_w = self._freqs_cis_w.to(x.device)
        for layer in self.layers:
            x = layer(x, freqs_cis_h, freqs_cis_w, h, w)
        return self.final_norm(x), h, w


class PixelShuffleDownsample(nn.Module):
    """Downsample visual tokens via PixelShuffle.

    Reduces spatial resolution by combining patches.
    """

    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, int, int]:
        """Downsample by pixel shuffle.

        Args:
            x: [batch, h*w, dim]
            h, w: spatial dimensions

        Returns:
            x: [batch, h'*w', dim * scale^2]
            h', w': new spatial dimensions
        """
        batch, seq_len, dim = x.shape
        x = x.view(batch, h, w, dim).permute(0, 3, 1, 2)  # [B, C, H, W]
        x = F.pixel_unshuffle(x, self.scale)
        _, new_dim, new_h, new_w = x.shape
        x = x.view(batch, new_dim, new_h * new_w).transpose(1, 2)
        return x, new_h, new_w


class MLPProjector(nn.Module):
    """MLP projector to map visual features to LLM embedding space."""

    def __init__(self, visual_dim: int, llm_dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = int(visual_dim * hidden_mult)
        self.fc1 = nn.Linear(visual_dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, llm_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x))
        return self.fc2(x)


class Connector(nn.Module):
    """Connector: PixelShuffle + MLP projection.

    Downsamples encoded image embeddings through PixelShuffle and
    projects them to the LLM's feature space via MLP.
    """

    def __init__(
        self,
        visual_dim: int,
        llm_dim: int,
        pixel_shuffle_scale: int = 2,
        mlp_hidden_mult: int = 4,
    ):
        super().__init__()
        self.downsample = PixelShuffleDownsample(pixel_shuffle_scale)
        self.projector = MLPProjector(
            visual_dim * (pixel_shuffle_scale ** 2),
            llm_dim,
            mlp_hidden_mult,
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x, _, _ = self.downsample(x, h, w)
        return self.projector(x)


class MoEDecoderLayer(nn.Module):
    """MoE-extended decoder layer with modality-specific experts.

    As described in Section 3.2.2:
    - MHA-MMoE: modality-specific Q,K,V,O projections, unified global attention
    - FFN-MMoE: modality-specific gate, up, down with SiLU activation

    x' = x + MHA-MMoE(RMSNorm(x))
    x^l = x' + FFN-MMoE(RMSNorm(x'))
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = ModalityMultiHeadAttention(dim, n_heads, dropout, use_rope=True, rope_type="1d")
        self.norm2 = RMSNorm(dim)
        self.ffn = ModalitySwiGLUFFN(dim, mlp_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        modality_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, dim]
            modality_mask: [batch, seq_len] True = visual, False = text
            freqs_cis: precomputed rotary frequencies
            attn_mask: optional attention mask
        """
        x = x + self.attn(
            self.norm1(x),
            modality_mask,
            freqs_cis=freqs_cis,
            attn_mask=attn_mask,
        )
        x = x + self.ffn(self.norm2(x), modality_mask)
        return x


class MoEDecoder(nn.Module):
    """MoE-extended LLM (decoder-only).

    Stack of MoEDecoderLayers. Supports from-scratch or pre-trained initialization.
    Uses causal attention with 1D RoPE.
    """

    def __init__(
        self,
        depth: int,
        dim: int,
        n_heads: int,
        mlp_dim: int,
        vocab_size: int,
        max_seq_len: int = 16384,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.max_seq_len = max_seq_len

        from layers import precompute_freqs_cis
        freqs_cis = precompute_freqs_cis(dim // n_heads, max_seq_len)
        self.register_buffer("_freqs_cis", freqs_cis, persistent=False)

        self.token_embedding = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            MoEDecoderLayer(dim, n_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None,
        visual_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through MoE decoder.

        This is a simplified interface. In practice, embeddings are constructed
        externally and passed through the decoder layers. See model.py for full pipeline.
        """
        freqs_cis = self._freqs_cis.to(input_ids.device)

        x = self.token_embedding(input_ids)

        batch, seq_len, _ = x.shape

        if modality_mask is None:
            modality_mask = torch.zeros(batch, seq_len, dtype=torch.bool, device=x.device)

        if attention_mask is not None:
            attn_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * float("-inf")
            attn_mask = attn_mask.to(x.dtype)
        else:
            from layers import create_causal_mask
            attn_mask = create_causal_mask(seq_len, x.device, x.dtype)

        for layer in self.layers:
            x = layer(x, modality_mask, freqs_cis, attn_mask)

        x = self.final_norm(x)
        return self.lm_head(x)
