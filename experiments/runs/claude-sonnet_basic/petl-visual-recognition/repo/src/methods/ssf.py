"""
SSF: Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning.
Lian et al., 2022.

SSF modulates intermediate features with learnable scale and shift factors.
Applied to features h2, h3, h5, h7, h8, h9 in the ViT block.
"""

import torch
import torch.nn as nn
import math


class SSFLayer(nn.Module):
    """Scale and Shift Feature layer."""
    
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        # x: (B, N, D) or (B, D)
        return x * self.scale + self.shift


class SSFBlock(nn.Module):
    """ViT Block with SSF applied to intermediate features."""
    
    def __init__(self, block):
        super().__init__()
        self.block = block
        dim = block.norm1.normalized_shape[0]
        
        # Get MLP hidden dim
        mlp_dim = block.mlp.fc1.out_features
        
        # SSF for LN1 output (h2 in paper notation)
        self.ssf_ln1 = SSFLayer(dim)
        # SSF for QKV output (h3 in paper notation)
        self.ssf_qkv = SSFLayer(3 * dim)
        # SSF for attention projection output (h5 in paper notation)
        self.ssf_proj = SSFLayer(dim)
        # SSF for LN2 output (h7 in paper notation)
        self.ssf_ln2 = SSFLayer(dim)
        # SSF for first FC output (h8 in paper notation)
        self.ssf_fc1 = SSFLayer(mlp_dim)
        # SSF for second FC output (h9 in paper notation)
        self.ssf_fc2 = SSFLayer(dim)
    
    def _attn_with_ssf(self, x):
        """Attention with SSF applied to QKV and projection."""
        B, N, C = x.shape
        
        # QKV projection with SSF
        qkv = self.block.attn.qkv(x)
        qkv = self.ssf_qkv(qkv)  # Apply SSF to QKV
        
        qkv = qkv.reshape(B, N, 3, self.block.attn.num_heads, C // self.block.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.block.attn.scale
        attn = attn.softmax(dim=-1)
        attn = self.block.attn.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.block.attn.proj(x)
        x = self.ssf_proj(x)  # Apply SSF to projection output
        x = self.block.attn.proj_drop(x)
        return x
    
    def _mlp_with_ssf(self, x):
        """MLP with SSF applied to intermediate features."""
        x = self.block.mlp.fc1(x)
        x = self.ssf_fc1(x)  # Apply SSF after first FC
        x = self.block.mlp.act(x)
        x = self.block.mlp.drop1(x)
        x = self.block.mlp.fc2(x)
        x = self.ssf_fc2(x)  # Apply SSF after second FC
        x = self.block.mlp.drop2(x)
        return x
    
    def forward(self, x):
        # LN1 + SSF
        h2 = self.ssf_ln1(self.block.norm1(x))
        # Attention with SSF
        h5 = self._attn_with_ssf(h2)
        # Apply layer scale if present
        if hasattr(self.block, 'ls1'):
            h5 = self.block.ls1(h5)
        # Apply drop path if present
        if hasattr(self.block, 'drop_path1'):
            h5 = self.block.drop_path1(h5)
        elif hasattr(self.block, 'drop_path'):
            h5 = self.block.drop_path(h5)
        x = x + h5
        
        # LN2 + SSF
        h7 = self.ssf_ln2(self.block.norm2(x))
        # MLP with SSF
        h9 = self._mlp_with_ssf(h7)
        # Apply layer scale if present
        if hasattr(self.block, 'ls2'):
            h9 = self.block.ls2(h9)
        # Apply drop path if present
        if hasattr(self.block, 'drop_path2'):
            h9 = self.block.drop_path2(h9)
        elif hasattr(self.block, 'drop_path'):
            h9 = self.block.drop_path(h9)
        x = x + h9
        
        return x


def apply_ssf(model, **kwargs):
    """
    Apply SSF to a ViT model.
    
    Args:
        model: ViT model (timm)
        **kwargs: Additional arguments (unused)
    
    Returns:
        model with SSF applied
    """
    # Replace blocks with SSF blocks
    new_blocks = nn.ModuleList()
    for block in model.blocks:
        new_blocks.append(SSFBlock(block))
    model.blocks = new_blocks
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze SSF parameters
    for name, param in model.named_parameters():
        if 'ssf_' in name:
            param.requires_grad = True
    
    # Always keep head trainable
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
    
    return model
