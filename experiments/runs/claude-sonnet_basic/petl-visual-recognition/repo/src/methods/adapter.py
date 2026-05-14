"""
Adapter-based PEFT methods:
- Houlsby Adapter: Inserts adapters after MSA and MLP blocks (sequential)
- Pfeiffer Adapter: Inserts adapter only after MLP block (sequential)

Houlsby et al., 2019; Pfeiffer et al., 2021.
"""

import torch
import torch.nn as nn
import math


class Adapter(nn.Module):
    """
    Bottleneck adapter module.
    Adapter(h) = s * W_up * sigma(W_down * h) + h
    """
    
    def __init__(self, dim, bottleneck_dim, scale_factor=1.0, act_layer=nn.GELU):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.act = act_layer()
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        self.scale = scale_factor
        
        # Initialize with near-zero weights for stable training
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        return self.scale * self.up_proj(self.act(self.down_proj(x))) + x


def _block_attn_output(block, x):
    """Get attention output from a timm ViT block, handling version differences."""
    attn_out = block.attn(block.norm1(x))
    if hasattr(block, 'ls1'):
        attn_out = block.ls1(attn_out)
    if hasattr(block, 'drop_path1'):
        attn_out = block.drop_path1(attn_out)
    elif hasattr(block, 'drop_path'):
        attn_out = block.drop_path(attn_out)
    return attn_out


def _block_mlp_output(block, x):
    """Get MLP output from a timm ViT block, handling version differences."""
    mlp_out = block.mlp(block.norm2(x))
    if hasattr(block, 'ls2'):
        mlp_out = block.ls2(mlp_out)
    if hasattr(block, 'drop_path2'):
        mlp_out = block.drop_path2(mlp_out)
    elif hasattr(block, 'drop_path'):
        mlp_out = block.drop_path(mlp_out)
    return mlp_out


class HoulsbyAdapterBlock(nn.Module):
    """ViT Block with Houlsby Adapters after MSA and MLP."""
    
    def __init__(self, block, bottleneck_dim, scale_factor=1.0):
        super().__init__()
        self.block = block
        dim = block.norm1.normalized_shape[0]
        self.adapter1 = Adapter(dim, bottleneck_dim, scale_factor)
        self.adapter2 = Adapter(dim, bottleneck_dim, scale_factor)
    
    def forward(self, x):
        # MSA block with adapter
        h5 = _block_attn_output(self.block, x)
        h5 = self.adapter1(h5)  # Adapter after MSA
        x = x + h5
        
        # MLP block with adapter
        h9 = _block_mlp_output(self.block, x)
        h9 = self.adapter2(h9)  # Adapter after MLP
        x = x + h9
        
        return x


class PfeiferAdapterBlock(nn.Module):
    """ViT Block with Pfeiffer Adapter only after MLP."""
    
    def __init__(self, block, bottleneck_dim, scale_factor=1.0):
        super().__init__()
        self.block = block
        dim = block.norm1.normalized_shape[0]
        self.adapter = Adapter(dim, bottleneck_dim, scale_factor)
    
    def forward(self, x):
        # MSA block (no adapter)
        x = x + _block_attn_output(self.block, x)
        
        # MLP block with adapter
        h9 = _block_mlp_output(self.block, x)
        h9 = self.adapter(h9)  # Adapter after MLP
        x = x + h9
        
        return x


def apply_adapter(model, bottleneck_dim=8, scale_factor=1.0, adapter_type='houlsby', **kwargs):
    """
    Apply Adapter to a ViT model.
    
    Args:
        model: ViT model (timm)
        bottleneck_dim: Bottleneck dimension for adapter
        scale_factor: Scale factor for adapter output
        adapter_type: 'houlsby' or 'pfeiffer'
        **kwargs: Additional arguments (unused)
    
    Returns:
        model with adapters applied
    """
    new_blocks = nn.ModuleList()
    for block in model.blocks:
        if adapter_type == 'houlsby':
            new_blocks.append(HoulsbyAdapterBlock(block, bottleneck_dim, scale_factor))
        else:  # pfeiffer
            new_blocks.append(PfeiferAdapterBlock(block, bottleneck_dim, scale_factor))
    model.blocks = new_blocks
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze adapter parameters
    for name, param in model.named_parameters():
        if 'adapter' in name:
            param.requires_grad = True
    
    # Always keep head trainable
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
    
    return model
