"""
NaViL Connector Module.

The connector C downsamples the encoded image embeddings through pixel shuffle 
and projects them to the LLM's feature space by an MLP.

Based on the paper (Section 3.1):
    "C is the connector which downsamples the encoded image embeddings 
     through pixel shuffle [15] and projects them to the LLM's feature 
     space by a MLP."

Reference: InternVL paper (Chen et al., 2023) for pixel shuffle details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelShuffleDownsample(nn.Module):
    """
    Pixel shuffle downsampling that rearranges spatial dimensions.
    Following InternVL's approach: given feature map with spatial dimensions
    (H, W), applies pixel unshuffle to reduce spatial resolution while 
    increasing channels.
    """
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C) token sequence
            h, w: original spatial height and width
        Returns:
            (B, H'*W', C * scale^2) downsampled tokens
        """
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, h, w)  # (B, C, H, W)
        x = F.pixel_unshuffle(x, self.scale_factor)  # (B, C*r^2, H/r, W/r)
        _, C2, h2, w2 = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', C*r^2)
        return x, h2, w2


class Connector(nn.Module):
    """
    Connector that downsamples visual features via pixel shuffle 
    and projects them to the LLM's hidden size.
    
    Architecture: PixelShuffle(downscale) -> LayerNorm -> MLP -> LLM hidden size
    """
    def __init__(
        self,
        visual_hidden_size: int = 1472,
        llm_hidden_size: int = 2048,
        downsample_ratio: int = 2,
        mlp_hidden_multiplier: int = 4,
    ):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        
        # After pixel shuffle: channels become visual_hidden_size * downsample_ratio^2
        intermediate_size = visual_hidden_size * (downsample_ratio ** 2)
        
        self.pixel_shuffle = PixelShuffleDownsample(scale_factor=downsample_ratio)
        self.norm = nn.LayerNorm(intermediate_size)
        
        # MLP projector
        self.mlp = nn.Sequential(
            nn.Linear(intermediate_size, llm_hidden_size * mlp_hidden_multiplier),
            nn.GELU(),
            nn.Linear(llm_hidden_size * mlp_hidden_multiplier, llm_hidden_size),
        )
        
    def forward(self, visual_features: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            visual_features: (B, H*W, visual_hidden_size) from visual encoder
            h, w: spatial dimensions of the feature map
        Returns:
            (B, H'*W', llm_hidden_size) projected visual tokens
        """
        x, h_out, w_out = self.pixel_shuffle(visual_features, h, w)
        x = self.norm(x)
        x = self.mlp(x)
        return x, h_out, w_out


class SimpleConnector(nn.Module):
    """
    A simpler connector variant that just uses an MLP without pixel shuffle.
    Useful for cases where d=0 (no visual encoder, just patch embedding).
    """
    def __init__(
        self,
        visual_hidden_size: int = 2048,
        llm_hidden_size: int = 2048,
        mlp_hidden_multiplier: int = 4,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(visual_hidden_size, llm_hidden_size * mlp_hidden_multiplier),
            nn.GELU(),
            nn.Linear(llm_hidden_size * mlp_hidden_multiplier, llm_hidden_size),
        )
        
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(visual_features)
