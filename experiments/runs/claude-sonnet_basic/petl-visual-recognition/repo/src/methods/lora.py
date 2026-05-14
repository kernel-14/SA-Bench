"""
LoRA: Low-Rank Adaptation of Large Language Models.
Hu et al., 2021.

LoRA applies low-rank decomposition to approximate weight updates.
Applied to Q and V projection weights in MSA blocks.
h3 = LoRA(h2) + h3
LoRA(h2) = [W_down^Q * W_up^Q * h2, 0, W_down^V * W_up^V * h2]
"""

import torch
import torch.nn as nn
import math


class LoRABlock(nn.Module):
    """ViT Block with LoRA applied to attention Q and V projections."""
    
    def __init__(self, block, rank=4):
        super().__init__()
        self.block = block
        dim = block.attn.qkv.in_features
        
        # LoRA for Q and V projections
        self.lora_q_down = nn.Linear(dim, rank, bias=False)
        self.lora_q_up = nn.Linear(rank, dim, bias=False)
        self.lora_v_down = nn.Linear(dim, rank, bias=False)
        self.lora_v_up = nn.Linear(rank, dim, bias=False)
        
        # Initialize: A with kaiming, B with zeros (so delta_W = 0 initially)
        nn.init.kaiming_uniform_(self.lora_q_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_up.weight)
        nn.init.kaiming_uniform_(self.lora_v_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_up.weight)
    
    def _attn_with_lora(self, x):
        """Attention with LoRA applied to Q and V."""
        B, N, C = x.shape
        
        # Original QKV projection
        qkv = self.block.attn.qkv(x)
        
        # LoRA updates for Q and V
        delta_q = self.lora_q_up(self.lora_q_down(x))
        delta_v = self.lora_v_up(self.lora_v_down(x))
        
        # Add LoRA updates to Q and V parts of QKV
        qkv = qkv.reshape(B, N, 3, C)
        qkv[:, :, 0, :] = qkv[:, :, 0, :] + delta_q  # Q
        qkv[:, :, 2, :] = qkv[:, :, 2, :] + delta_v  # V
        qkv = qkv.reshape(B, N, 3 * C)
        
        qkv = qkv.reshape(B, N, 3, self.block.attn.num_heads, C // self.block.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.block.attn.scale
        attn = attn.softmax(dim=-1)
        attn = self.block.attn.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.block.attn.proj(x)
        x = self.block.attn.proj_drop(x)
        return x
    
    def forward(self, x):
        # MSA block with LoRA
        attn_out = self._attn_with_lora(self.block.norm1(x))
        if hasattr(self.block, 'ls1'):
            attn_out = self.block.ls1(attn_out)
        if hasattr(self.block, 'drop_path1'):
            attn_out = self.block.drop_path1(attn_out)
        elif hasattr(self.block, 'drop_path'):
            attn_out = self.block.drop_path(attn_out)
        x = x + attn_out
        
        # MLP block (unchanged)
        mlp_out = self.block.mlp(self.block.norm2(x))
        if hasattr(self.block, 'ls2'):
            mlp_out = self.block.ls2(mlp_out)
        if hasattr(self.block, 'drop_path2'):
            mlp_out = self.block.drop_path2(mlp_out)
        elif hasattr(self.block, 'drop_path'):
            mlp_out = self.block.drop_path(mlp_out)
        x = x + mlp_out
        
        return x


def apply_lora(model, rank=4, **kwargs):
    """
    Apply LoRA to a ViT model.
    
    Args:
        model: ViT model (timm)
        rank: Rank for LoRA decomposition
        **kwargs: Additional arguments (unused)
    
    Returns:
        model with LoRA applied
    """
    new_blocks = nn.ModuleList()
    for block in model.blocks:
        new_blocks.append(LoRABlock(block, rank))
    model.blocks = new_blocks
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
    
    # Always keep head trainable
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
    
    return model
