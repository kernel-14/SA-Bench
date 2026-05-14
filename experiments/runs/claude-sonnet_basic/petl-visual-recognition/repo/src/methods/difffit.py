"""
DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning.
Xie et al., 2023.

DiffFit tunes:
- Bias terms (like BitFit)
- LayerNorm parameters
- Learnable scale factors gamma after MSA and MLP blocks
"""

import torch
import torch.nn as nn


class DiffFitBlock(nn.Module):
    """Wrapper around ViT Block that adds DiffFit scale factors."""
    
    def __init__(self, block):
        super().__init__()
        self.block = block
        dim = block.norm1.normalized_shape[0]
        # Learnable scale factors after MSA and MLP blocks
        self.gamma1 = nn.Parameter(torch.ones(dim))
        self.gamma2 = nn.Parameter(torch.ones(dim))
    
    def _attn_output(self, x):
        """Get attention output, handling different timm versions."""
        attn_out = self.block.attn(self.block.norm1(x))
        # Apply layer scale if present
        if hasattr(self.block, 'ls1'):
            attn_out = self.block.ls1(attn_out)
        # Apply drop path if present
        if hasattr(self.block, 'drop_path1'):
            attn_out = self.block.drop_path1(attn_out)
        elif hasattr(self.block, 'drop_path'):
            attn_out = self.block.drop_path(attn_out)
        return attn_out
    
    def _mlp_output(self, x):
        """Get MLP output, handling different timm versions."""
        mlp_out = self.block.mlp(self.block.norm2(x))
        # Apply layer scale if present
        if hasattr(self.block, 'ls2'):
            mlp_out = self.block.ls2(mlp_out)
        # Apply drop path if present
        if hasattr(self.block, 'drop_path2'):
            mlp_out = self.block.drop_path2(mlp_out)
        elif hasattr(self.block, 'drop_path'):
            mlp_out = self.block.drop_path(mlp_out)
        return mlp_out
    
    def forward(self, x):
        # MSA block with scale factor
        x = x + self.gamma1 * self._attn_output(x)
        # MLP block with scale factor
        x = x + self.gamma2 * self._mlp_output(x)
        return x


def apply_difffit(model, **kwargs):
    """
    Apply DiffFit to a ViT model.
    Tunes bias terms, LayerNorm params, and adds learnable scale factors.
    
    Args:
        model: ViT model (timm)
        **kwargs: Additional arguments (unused)
    
    Returns:
        model with DiffFit applied
    """
    # Replace blocks with DiffFit blocks
    new_blocks = nn.ModuleList()
    for block in model.blocks:
        new_blocks.append(DiffFitBlock(block))
    model.blocks = new_blocks
    
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze bias terms
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True
    
    # Unfreeze LayerNorm parameters
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = True
    
    # Unfreeze gamma scale factors
    for name, param in model.named_parameters():
        if 'gamma1' in name or 'gamma2' in name:
            param.requires_grad = True
    
    # Always keep head trainable
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
    
    return model
