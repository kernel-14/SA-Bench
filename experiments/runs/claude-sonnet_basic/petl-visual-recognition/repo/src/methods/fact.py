"""
FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer.
Jie and Deng, 2023.

FacT extends LoRA to tensor decomposition across all transformer layers.
Two variants:
- FacT_TT: Tensor-Train decomposition
- FacT_TK: Tucker decomposition
"""

import torch
import torch.nn as nn
import math


class FacTTT(nn.Module):
    """
    FacT with Tensor-Train (TT) decomposition.
    Delta_W_FacT = s * Sigma x_2 U^T x_3 V^T
    where U in R^{D x r}, V in R^{D x r}, Sigma in R^{12L x r x r}
    """
    
    def __init__(self, num_layers, dim, rank, scale_factor=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.rank = rank
        self.scale = scale_factor
        
        # Shared factors across layers
        self.U = nn.Parameter(torch.zeros(dim, rank))
        self.V = nn.Parameter(torch.zeros(dim, rank))
        # Layer-specific core tensor (12 matrices per layer: Q, K, V, O, W1, W2 x 2 for up/down)
        # Simplified: 6 weight matrices per layer (Q, K, V, O, MLP_fc1, MLP_fc2)
        self.Sigma = nn.Parameter(torch.zeros(12 * num_layers, rank, rank))
        
        # Initialize
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.zeros_(self.Sigma)
    
    def get_delta_w(self, layer_idx, matrix_idx):
        """Get weight update for a specific layer and matrix."""
        sigma_idx = layer_idx * 12 + matrix_idx
        sigma = self.Sigma[sigma_idx]  # (r, r)
        # Delta_W = U * sigma * V^T
        delta_w = self.scale * (self.U @ sigma @ self.V.T)  # (D, D)
        return delta_w


class FacTTK(nn.Module):
    """
    FacT with Tucker (TK) decomposition.
    Delta_W_FacT = s * A x_1 B^T x_2 U^T x_3 V^T
    where U in R^{D x r}, V in R^{D x r}, B in R^{12L x r}, A in R^{r x r x r}
    """
    
    def __init__(self, num_layers, dim, rank, scale_factor=1.0):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.rank = rank
        self.scale = scale_factor
        
        # Shared factors
        self.U = nn.Parameter(torch.zeros(dim, rank))
        self.V = nn.Parameter(torch.zeros(dim, rank))
        self.B = nn.Parameter(torch.zeros(12 * num_layers, rank))
        self.A = nn.Parameter(torch.zeros(rank, rank, rank))
        
        # Initialize
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.zeros_(self.B)
        nn.init.zeros_(self.A)
    
    def get_delta_w(self, layer_idx, matrix_idx):
        """Get weight update for a specific layer and matrix."""
        b_idx = layer_idx * 12 + matrix_idx
        b = self.B[b_idx]  # (r,)
        # Tucker: Delta_W = U * (b^T * A) * V^T
        # Simplified Tucker product
        core = torch.einsum('r,rij->ij', b, self.A)  # (r, r)
        delta_w = self.scale * (self.U @ core @ self.V.T)  # (D, D)
        return delta_w


class FacTBlock(nn.Module):
    """ViT Block with FacT weight updates."""
    
    def __init__(self, block, layer_idx, fact_module):
        super().__init__()
        self.block = block
        self.layer_idx = layer_idx
        self.fact = fact_module
    
    def forward(self, x):
        # Get weight updates for this layer
        # Matrix indices: 0=Q, 1=K, 2=V, 3=O, 4=MLP_fc1, 5=MLP_fc2
        # (simplified to 6 matrices per layer, using indices 0-5)
        
        # Apply FacT to attention
        h2 = self.block.norm1(x)
        
        # Compute attention with FacT updates
        B_size, N, C = h2.shape
        
        # Get delta weights
        delta_q = self.fact.get_delta_w(self.layer_idx, 0)
        delta_v = self.fact.get_delta_w(self.layer_idx, 2)
        
        # Original QKV
        qkv = self.block.attn.qkv(h2)
        
        # Add FacT updates to Q and V
        qkv_reshaped = qkv.reshape(B_size, N, 3, C)
        qkv_reshaped[:, :, 0, :] = qkv_reshaped[:, :, 0, :] + h2 @ delta_q.T
        qkv_reshaped[:, :, 2, :] = qkv_reshaped[:, :, 2, :] + h2 @ delta_v.T
        qkv = qkv_reshaped.reshape(B_size, N, 3 * C)
        
        qkv = qkv.reshape(B_size, N, 3, self.block.attn.num_heads, C // self.block.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.block.attn.scale
        attn = attn.softmax(dim=-1)
        attn = self.block.attn.attn_drop(attn)
        
        h4 = (attn @ v).transpose(1, 2).reshape(B_size, N, C)
        
        # Output projection with FacT update
        delta_o = self.fact.get_delta_w(self.layer_idx, 3)
        h5 = self.block.attn.proj(h4) + h4 @ delta_o.T
        h5 = self.block.attn.proj_drop(h5)
        
        x = x + self.block.drop_path1(self.block.ls1(h5))
        
        # MLP with FacT updates
        h7 = self.block.norm2(x)
        
        delta_fc1 = self.fact.get_delta_w(self.layer_idx, 4)
        delta_fc2 = self.fact.get_delta_w(self.layer_idx, 5)
        
        # MLP forward with FacT
        h8 = self.block.mlp.fc1(h7) + h7 @ delta_fc1[:self.block.mlp.fc1.out_features, :self.block.mlp.fc1.in_features].T
        h8 = self.block.mlp.act(h8)
        h8 = self.block.mlp.drop1(h8)
        h9 = self.block.mlp.fc2(h8) + h8 @ delta_fc2[:self.block.mlp.fc2.out_features, :self.block.mlp.fc2.in_features].T
        h9 = self.block.mlp.drop2(h9)
        
        x = x + self.block.drop_path2(self.block.ls2(h9))
        
        return x


def apply_fact(model, rank=8, scale_factor=1.0, fact_type='tt', **kwargs):
    """
    Apply FacT to a ViT model.
    
    Args:
        model: ViT model (timm)
        rank: Rank for tensor decomposition
        scale_factor: Scale factor for FacT output
        fact_type: 'tt' for Tensor-Train or 'tk' for Tucker
        **kwargs: Additional arguments (unused)
    
    Returns:
        model with FacT applied
    """
    num_layers = len(model.blocks)
    dim = model.blocks[0].norm1.normalized_shape[0]
    
    # Create FacT module
    if fact_type == 'tt':
        fact_module = FacTTT(num_layers, dim, rank, scale_factor)
    else:  # tk
        fact_module = FacTTK(num_layers, dim, rank, scale_factor)
    
    # Store FacT module in model
    model.fact_module = fact_module
    
    # Replace blocks with FacT blocks
    new_blocks = nn.ModuleList()
    for i, block in enumerate(model.blocks):
        new_blocks.append(FacTBlock(block, i, fact_module))
    model.blocks = new_blocks
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze FacT parameters
    for param in model.fact_module.parameters():
        param.requires_grad = True
    
    # Always keep head trainable
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
    
    return model
