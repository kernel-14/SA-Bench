import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union

# Placeholder for Config type hint to avoid circular import with config.py
# In a real project, this would be 'from config import Config'
Config = Any

# --- Helper / Mock Hiera-like Backbone (Non-Functional for this file) ---
# In a full implementation, this would be replaced by an actual Hiera model
# imported from a dedicated library (e.g., from Facebook Research's Hiera repo).
# This mock is for demonstration of the ImageEncoder's integration logic.

# Define typical output channel dimensions for Hiera variants at different stages.
# This is a simplification; actual Hiera configs are much more detailed.
_HIERA_CONFIGS = {
    "Hiera-T": {
        "out_channels": [96, 192, 384, 768],  # Channels for strides 4, 8, 16, 32
        "patch_size": 16,
        "global_attn_blocks": [5, 7, 9], # Example, not used in mock functional
    },
    "Hiera-S": {
        "out_channels": [96, 192, 384, 768],
        "patch_size": 16,
        "global_attn_blocks": [7, 10, 13],
    },
    "Hiera-B+": {
        "out_channels": [128, 256, 512, 1024], # Channels for strides 4, 8, 16, 32
        "patch_size": 16,
        "global_attn_blocks": [12, 16, 20],
    },
    "Hiera-L": {
        "out_channels": [192, 384, 768, 1536],
        "patch_size": 16,
        "global_attn_blocks": [23, 33, 43],
    },
}

class _HieraLikeBackbone(nn.Module):
    """
    A non-functional mock to simulate a Hiera-based image encoder backbone.
    In a real scenario, this would be replaced by the actual Hiera implementation.
    This mock produces dummy feature maps with correct dimensions.
    """
    def __init__(self, hiera_type: str):
        super().__init__()
        if hiera_type not in _HIERA_CONFIGS:
            raise ValueError(f"Unknown Hiera type: {hiera_type}. Supported: {list(_HIERA_CONFIGS.keys())}")
        
        self.config = _HIERA_CONFIGS[hiera_type]
        self.output_channels = self.config["out_channels"]
        self.patch_size = self.config["patch_size"]
        # self.global_attn_blocks = self.config["global_attn_blocks"] # Not used in mock functional

        print(f"Initialized mock Hiera-like backbone for {hiera_type}")
        print(f"  Expected output channels (strides 4,8,16,32): {self.output_channels}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simulates the output of a Hiera backbone.
        Returns a dictionary of feature maps at different strides.
        """
        batch_size, _, H, W = x.shape
        
        # Simulate features at different strides
        # C1: Stride 4
        C1_H, C1_W = H // 4, W // 4
        C1_feats = torch.randn(batch_size, self.output_channels[0], C1_H, C1_W, device=x.device)
        
        # C2: Stride 8
        C2_H, C2_W = H // 8, W // 8
        C2_feats = torch.randn(batch_size, self.output_channels[1], C2_H, C2_W, device=x.device)
        
        # C3: Stride 16
        C3_H, C3_W = H // 16, W // 16
        C3_feats = torch.randn(batch_size, self.output_channels[2], C3_H, C3_W, device=x.device)
        
        # C4: Stride 32
        C4_H, C4_W = H // 32, W // 32
        C4_feats = torch.randn(batch_size, self.output_channels[3], C4_H, C4_W, device=x.device)
        
        return {
            "C1": C1_feats, # Stride 4
            "C2": C2_feats, # Stride 8
            "C3": C3_feats, # Stride 16
            "C4": C4_feats, # Stride 32
        }

    def load_pretrained_weights(self, path: str) -> None:
        """
        Mocks loading pretrained weights. In a real scenario, this would load
        a state_dict into the actual Hiera backbone.
        """
        if os.path.exists(path):
            print(f"MOCK: Loading pretrained weights from {path} (simulated).")
            # In a real scenario:
            # state_dict = torch.load(path, map_location='cpu')
            # self.load_state_dict(state_dict, strict=False) # strict=False to handle potential FPN additions
        else:
            print(f"MOCK WARNING: Pretrained weights file not found at {path}. Skipping load.")


class ImageEncoder(nn.Module):
    """
    ImageEncoder module for SAM2, utilizing a Hiera-based backbone and an FPN.
    It extracts multi-scale features, fuses higher-level features, and outputs
    features for the MaskDecoder and MemoryAttention modules.
    """

    def __init__(self, config: Config):
        """
        Initializes the ImageEncoder.

        Args:
            config (Config): The global configuration object.
        """
        super().__init__()
        self._config = config

        hiera_type = self._config.get("model.image_encoder.type", "Hiera-B+")
        pretrained_path = self._config.get("model.image_encoder.pretrained_path")
        freeze_during_finetune = self._config.get("model.image_encoder.freeze_during_finetune", True)
        
        # The FPN's output channel dimension should match the hidden_dim
        # expected by MemoryAttention and MaskDecoder for consistency.
        output_feature_dim = self._config.get("model.memory_attention.hidden_dim", 256)

        # 1. Hiera Backbone Instantiation
        # Using the mock backbone. In a real project, replace _HieraLikeBackbone
        # with the actual Hiera model class.
        self.hiera_backbone = _HieraLikeBackbone(hiera_type)
        
        # 2. Pre-trained Weights Loading
        if pretrained_path:
            self.hiera_backbone.load_pretrained_weights(pretrained_path)

        # Get the output channel dimensions from the mock backbone config
        # These are for strides 4, 8, 16, 32 respectively.
        C1_channels, C2_channels, C3_channels, C4_channels = self.hiera_backbone.output_channels

        # 3. Feature Pyramid Network (FPN) Implementation
        # As per paper: "fuse the stride 16 and 32 features from Stages 3 and 4"
        # and "stride 4 and 8 features from Stages 1 and 2 are not used in the memory attention
        # but are added to the upsampling layers in the mask decoder"

        # FPN layers for C3 (stride 16) and C4 (stride 32)
        # Lateral convolutions for C3 and C4 to unify channel dimensions before fusion
        self.conv_c3_lateral = nn.Conv2d(C3_channels, output_feature_dim, kernel_size=1)
        self.conv_c4_lateral = nn.Conv2d(C4_channels, output_feature_dim, kernel_size=1)

        # Output convolution after fusion
        self.conv_fused_out = nn.Conv2d(output_feature_dim, output_feature_dim, kernel_size=3, padding=1)

        # 4. Freezing Parameters
        if freeze_during_finetune:
            for param in self.hiera_backbone.parameters():
                param.requires_grad = False
            # FPN layers might also be frozen or fine-tuned.
            # For simplicity, freezing only backbone as per usual pretrain then finetune flow.
            print(f"Image encoder backbone parameters frozen: {freeze_during_finetune}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Performs the forward pass through the ImageEncoder.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            List[torch.Tensor]: A list containing the multi-scale features:
                                [C1_feats (stride 4), C2_feats (stride 8), image_embedding (fused stride 16/32)].
        """
        # 1. Hiera Feature Extraction
        hiera_outputs = self.hiera_backbone(x)
        C1_feats = hiera_outputs["C1"] # Stride 4
        C2_feats = hiera_outputs["C2"] # Stride 8
        C3_feats = hiera_outputs["C3"] # Stride 16
        C4_feats = hiera_outputs["C4"] # Stride 32

        # 2. FPN Fusion (C3 and C4)
        # Apply lateral convolutions
        c3_lateral = self.conv_c3_lateral(C3_feats)
        c4_lateral = self.conv_c4_lateral(C4_feats)

        # Upsample C4 to C3's resolution and add
        c4_upsampled = F.interpolate(c4_lateral, size=c3_lateral.shape[2:], mode="nearest")
        fused_c3_c4 = c3_lateral + c4_upsampled

        # Apply final convolution to the fused features
        image_embedding = self.conv_fused_out(fused_c3_c4) # This is the primary input for MemoryAttention/MaskDecoder

        # Return the multi-scale features as specified: C1, C2, and the fused image_embedding
        return [C1_feats, C2_feats, image_embedding]

