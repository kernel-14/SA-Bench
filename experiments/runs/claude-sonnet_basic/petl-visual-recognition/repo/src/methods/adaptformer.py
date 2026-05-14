"""
AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition.
Chen et al., 2022.

AdaptFormer inserts an Adapter in parallel with the MLP block.
h9 = h9 + Adapter(h7)
"""

import torch
import torch.nn as nn
import math


class AdaptFormerAdapter(nn.Module):
    """
    AdaptFormer adapter module.
    Parallel adapter: h9 = h9 + s * W_up * sigma(W_down * h7)
    """
    
    def __init__(self, dim, bottleneck_dim, scale_factor=0.1, act_layer=nn.GELU):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck_dim)
        self.act = act_layer()
        self.up_proj = nn.Linear(bottleneck_dim, dim)
        self.scale = scale_factor
        
        # Initialize
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        return self.scale * self.up_proj(self.act(self.down_proj(x)))


class AdaptFormerBlock(nn.Module):
    """ViT Block with AdaptFormer (parallel adapter with MLP)."""
    
    def __init__(self, block, bottleneck_dim, scale_factor=0.1):
        super().__init__()
        self.block = block
        dim = block.norm1.normalized_shape[0]
        self.adapter = AdaptFormerAdapter(dim, bottleneck_dim, scale_factor)
    
    def _attn_output(self, x):
        """Get attention output, handling different timm versions."""
        attn_out = self.block.attn(self.block.norm1(x))
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
        # MSA block (no adapter)
        x = x + self._attn_output(x)
        
        # MLP block with parallel adapter
        h7 = self.block.norm2(x)
        h9_mlp = self._mlp_output(h7)
        h9_adapter = self.adapter(h7)  # Parallel adapter
        x = x + h9_mlp + h9_adapter
        
        return x


def apply_adaptformer(model, bottleneck_dim=8, scale_factor=0.1, **kwargs):
    """
    Apply AdaptFormer to a ViT model.
    
    Args:
        model: ViT model (timm)
        bottleneck_dim: Bottleneck dimension for adapter
        scale_factor: Scale factor for adapter output
        **kwargs: Additional arguments (unused)
    
    Returns:
        model with AdaptFormer applied
    """
    new_blocks = nn.ModuleList()
    for block in model.blocks:
        new_blocks.append(AdaptFormerBlock(block, bottleneck_dim, scale_factor))
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
