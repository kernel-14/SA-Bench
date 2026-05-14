"""Direct Selective Tuning PEFT methods: BitFit, LayerNorm, DiffFit.

These methods selectively update a subset of backbone parameters directly.
They introduce no additional inference latency.
"""

import torch
import torch.nn as nn


class BitFit(nn.Module):
    """BitFit: Fine-tune only the bias terms of the pre-trained model.
    
    Updates bias terms in:
    - Patch embedding projection
    - Q/K/V projections in MSA
    - FC layer in MSA (attention output projection)
    - Two FC layers in MLP
    - Two LN blocks per layer
    
    Parameters: 0.102M (fixed)
    """
    
    def __init__(self):
        super().__init__()
    
    def apply(self, vit_model):
        """Configure the ViT model for BitFit training.
        
        Freezes all parameters except bias terms.
        """
        for name, param in vit_model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return vit_model
    
    def get_trainable_params(self, vit_model=None):
        if vit_model is None:
            return 0.102  # Fixed for ViT-B/16
        return sum(p.numel() for p in vit_model.parameters() if p.requires_grad) / 1e6
    
    def forward(self, x, vit_model):
        return vit_model(x)


class LayerNorm(nn.Module):
    """LayerNorm Tuning: Fine-tune only the LN block parameters.
    
    Updates the two LN blocks in each Transformer layer:
    - LN before MSA (weight and bias)
    - LN before MLP (weight and bias)
    
    Also updates the final LN before the head.
    
    Each LN has 2*D trainable parameters (weight + bias).
    For ViT-B/16 with 12 layers: ~38K parameters = ~0.04% of total.
    
    Parameters: 0.038M (fixed)
    """
    
    def __init__(self):
        super().__init__()
    
    def apply(self, vit_model):
        """Freeze all parameters except LN weights and biases."""
        for name, param in vit_model.named_parameters():
            if 'norm' in name.lower() or 'ln' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        return vit_model
    
    def get_trainable_params(self, vit_model=None):
        if vit_model is None:
            return 0.038  # Fixed for ViT-B/16
        return sum(p.numel() for p in vit_model.parameters() if p.requires_grad) / 1e6
    
    def forward(self, x, vit_model):
        return vit_model(x)


class DiffFit(nn.Module):
    """DiffFit: Combines BitFit + LayerNorm tuning + learnable scale factors.
    
    Updates:
    - All bias terms (from BitFit)
    - All LN parameters (from LayerNorm)
    - Learnable scale factors γ₁, γ₂ after MSA and MLP blocks:
        h5 = γ₁ · h5
        h9 = γ₂ · h9
    
    Parameters: 0.140M (fixed)
    """
    
    def __init__(self, embed_dim=768, num_layers=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Learnable scale factors for MSA output (h5) and MLP output (h9)
        self.msa_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, embed_dim))
            for _ in range(num_layers)
        ])
        self.mlp_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, 1, embed_dim))
            for _ in range(num_layers)
        ])
    
    def apply(self, vit_model):
        """Freeze backbone, enable bias/LN tuning, add scale factors."""
        for name, param in vit_model.named_parameters():
            if 'bias' in name or 'norm' in name.lower() or 'ln' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
        return vit_model
    
    def forward(self, x, vit_model):
        """Forward pass with DiffFit modifications.
        
        Inserts scale factors after MSA and MLP blocks.
        """
        B = x.shape[0]
        
        x = vit_model.patch_embed(x)
        cls_token = vit_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit_model.pos_embed
        
        for i, block in enumerate(vit_model.blocks):
            # MSA block
            z_norm = block.norm1(x)
            z_msa = block.attn(z_norm)
            x = x + z_msa
            
            # Apply MSA scale: h5 = γ₁ · h5
            x = self.msa_scales[i] * x
            
            # MLP block
            z_norm2 = block.norm2(x)
            z_mlp = block.mlp(z_norm2)
            x = x + z_mlp
            
            # Apply MLP scale: h9 = γ₂ · h9
            x = self.mlp_scales[i] * x
        
        cls_out = x[:, 0, :]
        if hasattr(vit_model, 'head'):
            return vit_model.head(cls_out)
        return cls_out
    
    def get_trainable_params(self, vit_model=None):
        if vit_model is None:
            return 0.140  # Fixed for ViT-B/16
        backbone_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
        own_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (backbone_params + own_params) / 1e6
