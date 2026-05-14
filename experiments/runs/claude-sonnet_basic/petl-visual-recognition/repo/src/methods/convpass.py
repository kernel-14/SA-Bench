"""
ConvPass: Convolutional Bypasses are Better Vision Transformer Adapters.
Jie and Deng, 2022.

ConvPass inserts a convolutional bottleneck module in parallel with MSA and MLP blocks.
h5 = ConvPass1(h2) + h5
h9 = ConvPass2(h7) + h9
"""

import torch
import torch.nn as nn
import math


class ConvPassModule(nn.Module):
    """
    Convolutional bypass module.
    ConvPass(h) = s * W_up * sigma(Conv2d(sigma(W_down * h)))
    """
    
    def __init__(self, dim, bottleneck_dim, scale_factor=1.0, kernel_size=3, 
                 xavier_init=False, act_layer=nn.GELU):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.conv = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=kernel_size, 
                              padding=kernel_size // 2, groups=bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        self.act = act_layer()
        self.scale = scale_factor
        
        if xavier_init:
            nn.init.xavier_uniform_(self.down_proj.weight)
            nn.init.xavier_uniform_(self.up_proj.weight)
        else:
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
        
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        # x: (B, N, D) where N = 1 + num_patches
        B, N, D = x.shape
        
        # Down projection
        h = self.act(self.down_proj(x))  # (B, N, r)
        
        # Reshape for 2D convolution (skip cls token)
        cls_token = h[:, :1, :]  # (B, 1, r)
        patch_tokens = h[:, 1:, :]  # (B, N-1, r)
        
        # Compute spatial dimensions
        num_patches = N - 1
        H = W = int(math.sqrt(num_patches))
        
        # Reshape to 2D
        patch_tokens = patch_tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, r, H, W)
        
        # Apply 2D convolution
        patch_tokens = self.act(self.conv(patch_tokens))  # (B, r, H, W)
        
        # Reshape back
        patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(B, num_patches, -1)  # (B, N-1, r)
        
        # Concatenate cls token back
        h = torch.cat([cls_token, patch_tokens], dim=1)  # (B, N, r)
        
        # Up projection
        return self.scale * self.up_proj(h)


class ConvPassBlock(nn.Module):
    """ViT Block with ConvPass modules in parallel with MSA and MLP."""
    
    def __init__(self, block, bottleneck_dim, scale_factor=1.0, xavier_init=False):
        super().__init__()
        self.block = block
        dim = block.norm1.normalized_shape[0]
        self.convpass1 = ConvPassModule(dim, bottleneck_dim, scale_factor, xavier_init=xavier_init)
        self.convpass2 = ConvPassModule(dim, bottleneck_dim, scale_factor, xavier_init=xavier_init)
    
    def _attn_output(self, x):
        """Get attention output, handling different timm versions."""
        attn_out = self.block.attn(x)
        if hasattr(self.block, 'ls1'):
            attn_out = self.block.ls1(attn_out)
        if hasattr(self.block, 'drop_path1'):
            attn_out = self.block.drop_path1(attn_out)
        elif hasattr(self.block, 'drop_path'):
            attn_out = self.block.drop_path(attn_out)
        return attn_out
    
    def _mlp_output(self, x):
        """Get MLP output, handling different timm versions."""
        mlp_out = self.block.mlp(x)
        if hasattr(self.block, 'ls2'):
            mlp_out = self.block.ls2(mlp_out)
        if hasattr(self.block, 'drop_path2'):
            mlp_out = self.block.drop_path2(mlp_out)
        elif hasattr(self.block, 'drop_path'):
            mlp_out = self.block.drop_path(mlp_out)
        return mlp_out
    
    def forward(self, x):
        # MSA block with parallel ConvPass
        h2 = self.block.norm1(x)
        h5_attn = self._attn_output(h2)
        h5_conv = self.convpass1(h2)  # Parallel ConvPass
        x = x + h5_attn + h5_conv
        
        # MLP block with parallel ConvPass
        h7 = self.block.norm2(x)
        h9_mlp = self._mlp_output(h7)
        h9_conv = self.convpass2(h7)  # Parallel ConvPass
        x = x + h9_mlp + h9_conv
        
        return x


def apply_convpass(model, bottleneck_dim=8, scale_factor=1.0, xavier_init=False, **kwargs):
    """
    Apply ConvPass to a ViT model.
    
    Args:
        model: ViT model (timm)
        bottleneck_dim: Bottleneck dimension for ConvPass
        scale_factor: Scale factor for ConvPass output
        xavier_init: Whether to use Xavier initialization
        **kwargs: Additional arguments (unused)
    
    Returns:
        model with ConvPass applied
    """
    new_blocks = nn.ModuleList()
    for block in model.blocks:
        new_blocks.append(ConvPassBlock(block, bottleneck_dim, scale_factor, xavier_init))
    model.blocks = new_blocks
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze ConvPass parameters
    for name, param in model.named_parameters():
        if 'convpass' in name:
            param.requires_grad = True
    
    # Always keep head trainable
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
    
    return model
