
import torch
import torch.nn as nn
from typing import Optional

from layers import RMSNorm, PatchEmbedding, MHA_MMoE, FFN_MMoE

class TransformerBlock(nn.Module):
    """
    A single transformer block, used in both visual encoder and LLM.
    Can be configured for MoE and different attention types.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int,
                 is_moe: bool = False, is_bidirectional_attn: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = MHA_MMoE(embed_dim, num_heads, is_bidirectional=is_bidirectional_attn) if is_moe else \
                    nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) # Placeholder for non-MoE MHA
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = FFN_MMoE(embed_dim, mlp_dim) if is_moe else \
                   nn.Sequential(
                       nn.Linear(embed_dim, mlp_dim),
                       nn.SiLU(),
                       nn.Linear(mlp_dim, embed_dim)
                   ) # Placeholder for non-MoE FFN

        self.is_moe = is_moe
        self.is_bidirectional_attn = is_bidirectional_attn

    def forward(self, x: torch.Tensor, modality_indicator: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                rope_pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Self-attention block
        if self.is_moe:
            assert modality_indicator is not None, "modality_indicator must be provided for MoE attention"
            # MHA-MMoE expects modality_indicator to select experts
            x_norm = self.norm1(x)
            attn_output = self.attn(x_norm, modality_indicator, attn_mask, rope_pos_bias)
            x = x + attn_output
        else:
            # Standard MultiheadAttention
            x_norm = self.norm1(x)
            # For standard MultiheadAttention, need to adapt modality_indicator handling
            # or ensure this path is only for unimodal parts.
            # Assuming for now it's mostly in context of MMoE in NaViL, or
            # this will be used where modality_indicator is not relevant (e.g., standard Visual Encoder FFN)
            if isinstance(self.attn, nn.MultiheadAttention):
                # attn_mask for MultiheadAttention is typically (L, S) or (B*H, L, S) for transformers
                # For causal, upper triangle is filled with -inf
                if not self.is_bidirectional_attn and attn_mask is None: # Causal attention
                    seq_len = x_norm.shape[1]
                    subsequent_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
                    attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=subsequent_mask)
                else:
                    attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
            else: # Fallback for non-moe but custom attn
                attn_output = self.attn(x_norm, modality_indicator, attn_mask, rope_pos_bias) # Still pass modality just in case

            x = x + attn_output

        # FFN block
        if self.is_moe:
            assert modality_indicator is not None, "modality_indicator must be provided for MoE FFN"
            # FFN-MMoE expects modality_indicator to select experts
            x_norm = self.norm2(x)
            ffn_output = self.ffn(x_norm, modality_indicator)
            x = x + ffn_output
        else:
            # Standard FFN
            x_norm = self.norm2(x)
            ffn_output = self.ffn(x_norm)
            x = x + ffn_output

        return x

class VisualEncoder(nn.Module):
    """
    Visual Encoder as described in the paper.
    Consists of a Patch Embedding Layer and a series of Transformer Blocks.
    Uses bi-directional attention and 2D-RoPE (applied in MHA_MMoE for simplicity).
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int,
                 embed_dim: int, depth: int, num_heads: int, mlp_dim: int,
                 stride: int = 16):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim, stride)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, is_bidirectional_attn=True, is_moe=False) # Visual encoder blocks are not MoE
            for _ in range(depth)
        ])
        self.norm = RMSNorm(embed_dim)

    def forward(self, x: torch.Tensor,
                spatial_pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.patch_embed(x) # (B, N_patches, embed_dim)

        # For visual encoder, modality_indicator is always 'visual' (e.g., all ones)
        # We don't need modality_indicator if the visual encoder itself doesn't use MoE.
        # The paper states visual encoder is a series of transformer layers, and MoE is injected into LLM.
        # So, the transformer blocks in visual encoder will use standard (non-MMoE) MHA and FFN.
        # If 2D-RoPE is applied, it will be handled within the MHA layer.

        for block in self.blocks:
            # Assuming 2D-RoPE is implicitly handled if passed as rope_pos_bias
            x = block(x, modality_indicator=None, attn_mask=None, rope_pos_bias=spatial_pos_bias)
        
        x = self.norm(x)
        return x

class MLPProjector(nn.Module):
    """
    Connector C that downsamples encoded image embeddings and projects them to LLM's feature space.
    The paper mentions pixel shuffle and then MLP.
    For simplicity, we'll assume a direct MLP projection after potential resizing if needed.
    A full pixel shuffle implementation would be more complex and usually used for upsampling.
    Given "downsamples the encoded image embeddings through pixel shuffle [15]", this sounds like
    an error in interpretation or a very specific use of pixel shuffle for downsampling.
    Pixel shuffle typically upsamples, e.g., from (B, C*r*r, H, W) to (B, C, H*r, W*r).
    A more common approach for downsampling and projection would be pooling + MLP or Conv + MLP.
    For now, let's assume a simple MLP that projects from visual_embed_dim to llm_embed_dim.
    The "downsamples" could refer to reducing sequence length, or simply changing feature dimension.
    Given the overall architecture, a simple linear projection is a reasonable interpretation
    for the "projects them to the LLM's feature space by a MLP."
    """
    def __init__(self, visual_embed_dim: int, llm_embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(visual_embed_dim, llm_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_patches, visual_embed_dim)
        return self.proj(x) # (B, N_patches, llm_embed_dim)

class NaViLLLMTypicalBlock(nn.Module):
    """
    A single LLM block with MoE.
    Uses causal attention and 1D-RoPE (applied in MHA_MMoE for simplicity).
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        # LLM blocks use MoE and causal attention
        self.block = TransformerBlock(embed_dim, num_heads, mlp_dim,
                                      is_moe=True, is_bidirectional_attn=False)

    def forward(self, x: torch.Tensor, modality_indicator: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                rope_pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.block(x, modality_indicator, attn_mask, rope_pos_bias)
