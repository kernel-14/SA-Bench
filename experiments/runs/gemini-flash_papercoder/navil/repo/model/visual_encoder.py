import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import Config
from model.components import RMSNorm, PatchEmbed
from utils import apply_rope_2d, logger # Assuming apply_rope_2d is in utils.py and logger for logging


class Mlp(nn.Module):
    """
    A standard Multilayer Perceptron (MLP) block for the transformer.
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU, drop: float = 0.0):
        """
        Initializes the MLP block.

        Args:
            in_features: Number of input features.
            hidden_features: Number of hidden features. Defaults to in_features.
            out_features: Number of output features. Defaults to in_features.
            act_layer: Activation function to use (e.g., nn.GELU).
            drop: Dropout rate.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x: Input tensor of shape (B, N, C).

        Returns:
            Output tensor of shape (B, N, C).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VisualAttention(nn.Module):
    """
    Multi-Head Self-Attention module for the Visual Encoder, incorporating 2D-RoPE.
    """
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        """
        Initializes the VisualAttention module.

        Args:
            dim: Input and output feature dimension.
            num_heads: Number of attention heads.
            qkv_bias: If True, add bias to QKV projections.
            attn_drop: Dropout rate for attention weights.
            proj_drop: Dropout rate for output projection.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim # Store head_dim for RoPE calculation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, patch_H: int, patch_W: int, rope_theta: float) -> torch.Tensor:
        """
        Forward pass of the VisualAttention.

        Args:
            x: Input tensor of shape (B, N, C), where B is batch size, N is number of tokens, C is embedding dimension.
            patch_H: Height of the patch grid (number of patches vertically).
            patch_W: Width of the patch grid (number of patches horizontally).
            rope_theta: Hyperparameter for 2D RoPE frequency calculation.

        Returns:
            Output tensor of shape (B, N, C).
        """
        B, N, C = x.shape
        # Project to QKV and split into heads
        # qkv: (B, N, C*3) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # q, k, v: (B, num_heads, N, head_dim)

        # Apply 2D-RoPE to Q and K
        q, k = apply_rope_2d(q, k, patch_H, patch_W, self.head_dim, rope_theta)

        # Compute attention scores
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Compute weighted sum of values
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # Reshape and concatenate heads

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VisualEncoderBlock(nn.Module):
    """
    A single Transformer block for the Visual Encoder.
    Consists of Multi-Head Self-Attention with 2D-RoPE and a Feed-Forward Network.
    """
    def __init__(self, dim: int, num_heads: int, mlp_dim: int,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 act_layer: nn.Module = nn.GELU):
        """
        Initializes a VisualEncoderBlock.

        Args:
            dim: Input and output feature dimension.
            num_heads: Number of attention heads.
            mlp_dim: Hidden dimension of the MLP.
            qkv_bias: If True, add bias to QKV projections in attention.
            drop: Dropout rate for MLP.
            attn_drop: Dropout rate for attention weights.
            act_layer: Activation function for MLP.
        """
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = VisualAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = RMSNorm(dim)
        
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, patch_H: int, patch_W: int, rope_theta: float) -> torch.Tensor:
        """
        Forward pass of the VisualEncoderBlock.

        Args:
            x: Input tensor of shape (B, N, C).
            patch_H: Height of the patch grid.
            patch_W: Width of the patch grid.
            rope_theta: Hyperparameter for 2D RoPE frequency calculation.

        Returns:
            Output tensor of shape (B, N, C).
        """
        x = x + self.attn(self.norm1(x), patch_H, patch_W, rope_theta)
        x = x + self.mlp(self.norm2(x))
        return x


class VisualEncoder(nn.Module):
    """
    The Visual Encoder module for NaViL.
    Transforms raw images into visual token embeddings using patch embedding
    followed by a stack of Transformer blocks with bidirectional attention and 2D-RoPE.
    """
    def __init__(self, config: Config):
        """
        Initializes the VisualEncoder.

        Args:
            config: The global configuration object.
        """
        super().__init__()
        visual_encoder_config = config.model_architecture.visual_encoder

        self.depth = visual_encoder_config.depth
        self.width = visual_encoder_config.width  # Hidden dimension (embed_dim)
        self.mlp_width = visual_encoder_config.mlp_width
        self.num_attention_heads = visual_encoder_config.num_attention_heads
        self.patch_embedding_stride = visual_encoder_config.patch_embedding_stride
        
        # RoPE theta value from common config
        self.rope_theta = config.get("common.rope_theta", 10000.0) # Default if not in config

        # 1. Patch Embedding Layer
        self.patch_embedding = PatchEmbed(config)

        # 2. Transformer Layers
        self.transformer_layers = nn.ModuleList([
            VisualEncoderBlock(
                dim=self.width,
                num_heads=self.num_attention_heads,
                mlp_dim=self.mlp_width, # Use explicit mlp_width from config
                qkv_bias=True, # Common for visual transformers
                drop=0.0, # No dropout rates specified in paper for VE, assume 0.0
                attn_drop=0.0, # No dropout rates specified in paper for VE, assume 0.0
            )
            for _ in range(self.depth)
        ])

        # 3. Final normalization layer
        self.norm = RMSNorm(self.width)
        logger.info(f"VisualEncoder initialized with depth={self.depth}, width={self.width}, "
                    f"num_heads={self.num_attention_heads}, mlp_width={self.mlp_width}.")

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Forward pass of the VisualEncoder.

        Args:
            images: Input image tensor of shape (B, C, H_img, W_img).
                    Assumed to be padded (in `dataset/collate_fn.py`)
                    so that H_img and W_img are multiples of
                    `patch_embedding_stride`.

        Returns:
            A tuple:
                - Processed visual token embeddings of shape (B, N_patches, embed_dim).
                - Height of the patch grid (H_out).
                - Width of the patch grid (W_out).
        """
        B, C, H_img, W_img = images.shape
        x = self.patch_embedding(images) # x: (B, N_patches, embed_dim)
        
        N_patches = x.shape[1]
        
        # Infer H_out, W_out from original image dimensions and patch_embedding_stride.
        # This relies on the assumption that images are correctly padded upstream
        # so H_img and W_img are already multiples of patch_embedding_stride.
        patch_H = H_img // self.patch_embedding_stride
        patch_W = W_img // self.patch_embedding_stride
        
        # A sanity check for consistency, useful for debugging if padding is incorrect
        if N_patches != patch_H * patch_W:
            logger.warning(
                f"Mismatch between inferred patch grid dimensions ({patch_H}x{patch_W}={patch_H*patch_W}) "
                f"and actual number of patches ({N_patches}) after patch embedding. "
                "This might indicate an issue with input image padding or the patch embedding layer's calculation. "
                "Proceeding with inferred patch_H, patch_W for RoPE application."
            )
            # In a real scenario, this might need more robust error handling or recalculation,
            # but per instructions, we proceed assuming logical consistency or for debugging.

        logger.debug(f"After patch embedding: x shape {x.shape}, inferred patch_H={patch_H}, patch_W={patch_W}")

        for layer_idx, blk in enumerate(self.transformer_layers):
            x = blk(x, patch_H, patch_W, self.rope_theta)
            logger.debug(f"After VisualEncoderBlock {layer_idx}: x shape {x.shape}")

        x = self.norm(x)
        logger.debug(f"After final RMSNorm: x shape {x.shape}")

        return x, patch_H, patch_W

