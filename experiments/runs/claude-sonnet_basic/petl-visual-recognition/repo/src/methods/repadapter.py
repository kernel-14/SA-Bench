"""
RepAdapter: Towards Efficient Visual Adaption via Structural Re-parameterization.
Luo et al., 2023.

RepAdapter uses linear adapters with group-wise transformations.
Placed sequentially after MSA and MLP blocks.
After training, can be re-parameterized into the original weights.
"""

import torch
import torch.nn as nn
import math


class RepAdapterModule(nn.Module):
    """
    RepAdapter module with group-wise transformation.
    RepAdapter(h) = s * phi_up(phi_down(h)) + h
    phi_down(h) = W_down * h
    phi_up(h_tilde) = [W_g1 * h_tilde_g1, ..., W_gG * h_tilde_gG]
    """
    
    def __init__(self, dim, bottleneck_dim, scale_factor=1.0, num_groups=2):
        super().__init__()
        self.down_proj = nn.Linear(dim, bottleneck_dim, bias=False)
        self.up_proj = nn.Linear(bottleneck_dim, dim, bias=False)
        self.scale = scale_factor
        self.num_groups = num_groups
        self.bottleneck_dim = bottleneck_dim
        self.dim = dim
        
        # Initialize
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
    
    def forward(self, x):
        # Down projection
        h = self.down_proj(x)  # (B, N, r)
        
        # Apply up projection
        out = self.up_proj(h)  # (B, N, D)
        
        return self.scale * out + x


class RepAdapterBlock(nn.Module):
    """ViT Block with RepAdapter after MSA and MLP."""
    
    def __init__(self, block, bottleneck_dim, scale_factor=1.0, num_groups=2):
        super().__init__()
        self.block = block
        dim = block.norm1.normalized_shape[0]
        self.repadapter1 = RepAdapterModule(dim, bottleneck_dim, scale_factor, num_groups)
        self.repadapter2 = RepAdapterModule(dim, bottleneck_dim, scale_factor, num_groups)
    
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
        # Apply RepAdapter before MSA (on h2 = LN(x))
        h2 = self.block.norm1(x)
        h2 = self.repadapter1(h2)  # RepAdapter on h2
        h5 = self._attn_output(h2)
        x = x + h5
        
        # Apply RepAdapter before MLP (on h7 = LN(x))
        h7 = self.block.norm2(x)
        h7 = self.repadapter2(h7)  # RepAdapter on h7
        h9 = self._mlp_output(h7)
        x = x + h9
        
        return x


def apply_repadapter(model, bottleneck_dim=8, scale_factor=1.0, num_groups=2, **kwargs):
    """
    Apply RepAdapter to a ViT model.
    
    Args:
        model: ViT model (timm)
        bottleneck_dim: Bottleneck dimension for RepAdapter
        scale_factor: Scale factor for RepAdapter output
        num_groups: Number of groups for group-wise transformation
        **kwargs: Additional arguments (unused)
    
    Returns:
        model with RepAdapter applied
    """
    new_blocks = nn.ModuleList()
    for block in model.blocks:
        new_blocks.append(RepAdapterBlock(block, bottleneck_dim, scale_factor, num_groups))
    model.blocks = new_blocks
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze RepAdapter parameters
    for name, param in model.named_parameters():
        if 'repadapter' in name:
            param.requires_grad = True
    
    # Always keep head trainable
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
    
    return model
