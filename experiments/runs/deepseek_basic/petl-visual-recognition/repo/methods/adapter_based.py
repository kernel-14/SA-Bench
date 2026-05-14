"""Adapter-based PEFT methods.

Includes: Houlsby Adapter, Pfeiffer Adapter, AdaptFormer, ConvPass, RepAdapter

Each adapter is a lightweight bottleneck-structured module inserted into
the Transformer layers to adapt features for downstream tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BottleneckAdapter(nn.Module):
    """Standard bottleneck adapter: down-project -> activate -> up-project + residual.
    
    Adapter(h) = s * W_up * σ(W_down * h) + h
    
    where σ is GELU activation, s is scaling factor, r << D is bottleneck dimension.
    """
    
    def __init__(self, embed_dim=768, bottleneck_dim=16, scale=1.0, use_residual=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale
        self.use_residual = use_residual
        
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.act = nn.GELU()
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        residual = x
        out = self.down_proj(x)
        out = self.act(out)
        out = self.up_proj(out)
        out = self.scale * out
        if self.use_residual:
            out = out + residual
        return out


class ConvPassAdapter(nn.Module):
    """Convolutional bypass adapter for visual inductive bias.
    
    Convpass(h) = s * W_up * σ(Conv2d(σ(W_down * h)))
    
    Uses 1x1 -> 3x3 conv -> 1x1 convolutions to capture local spatial info.
    """
    
    def __init__(self, embed_dim=768, bottleneck_dim=16, scale=1.0, 
                 kernel_size=3, num_patches=197):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale
        
        # 1x1 conv for down projection
        self.down_proj = nn.Conv2d(embed_dim, bottleneck_dim, kernel_size=1)
        # 3x3 conv for spatial processing
        self.conv = nn.Conv2d(bottleneck_dim, bottleneck_dim, 
                              kernel_size=kernel_size, padding=kernel_size//2)
        # 1x1 conv for up projection
        self.up_proj = nn.Conv2d(bottleneck_dim, embed_dim, kernel_size=1)
        self.act = nn.GELU()
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.down_proj, self.conv, self.up_proj]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """x: [B, N+1, D] where N+1 includes class token and patches."""
        residual = x
        B, T, D = x.shape
        
        # For ConvPass, we operate on patches only (skip class token for conv)
        # The class token interacts through the 1x1 convs
        # Reshape to spatial for conv: [B, D, H, W]
        # We need to handle this carefully; for simplicity, treat entire sequence
        # as spatial (assumes square patches)
        H = W = int(math.sqrt(T))
        
        # If T is not a perfect square (includes class token), handle separately
        if T != H * W:
            # Separate class token
            cls_token = x[:, :1, :]
            patches = x[:, 1:, :]
            H_p = W_p = int(math.sqrt(T - 1))
            patches = patches.transpose(1, 2).reshape(B, D, H_p, W_p)
            
            # Process patches
            patches_out = self.down_proj(patches)
            patches_out = self.act(patches_out)
            patches_out = self.conv(patches_out)
            patches_out = self.act(patches_out)
            patches_out = self.up_proj(patches_out)
            patches_out = patches_out.reshape(B, D, -1).transpose(1, 2)
            
            out = torch.cat([torch.zeros_like(cls_token), patches_out], dim=1)
        else:
            x_spatial = x.transpose(1, 2).reshape(B, D, H, W)
            out = self.down_proj(x_spatial)
            out = self.act(out)
            out = self.conv(out)
            out = self.act(out)
            out = self.up_proj(out)
            out = out.reshape(B, D, -1).transpose(1, 2)
        
        return self.scale * out + residual


class RepAdapterModule(nn.Module):
    """Linear adapter with group-wise transformation for re-parameterization.
    
    RepAdapter(h) = s * φ_up(φ_down(h)) + h
    φ_down(h) = W_down * h
    φ_up(h̃) = [W_g1 * h̃_g1, ..., W_gG * h̃_gG]
    
    No nonlinearity - can be re-parameterized into backbone after training.
    """
    
    def __init__(self, embed_dim=768, bottleneck_dim=16, scale=1.0, num_groups=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale
        self.num_groups = num_groups
        
        assert bottleneck_dim % num_groups == 0, "bottleneck_dim must be divisible by num_groups"
        assert embed_dim % num_groups == 0, "embed_dim must be divisible by num_groups"
        
        group_bn_dim = bottleneck_dim // num_groups
        group_emb_dim = embed_dim // num_groups
        
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        
        # Group-wise up projections
        self.up_projs = nn.ModuleList([
            nn.Linear(group_bn_dim, group_emb_dim, bias=False)
            for _ in range(num_groups)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        for proj in self.up_projs:
            nn.init.kaiming_uniform_(proj.weight, a=math.sqrt(5))
    
    def forward(self, x):
        residual = x
        out = self.down_proj(x)  # [B, N, bottleneck_dim]
        
        # Split into groups
        group_size = self.bottleneck_dim // self.num_groups
        groups = torch.chunk(out, self.num_groups, dim=-1)
        
        # Apply group-wise up projections
        up_groups = []
        for i, g in enumerate(groups):
            up_groups.append(self.up_projs[i](g))
        
        out = torch.cat(up_groups, dim=-1)
        return self.scale * out + residual


class HoulsbyAdapter(nn.Module):
    """Houlsby Adapter: Two adapters per Transformer layer.
    
    Adapter1 after MSA block (h5): h5 = Adapter1(h5)
    Adapter2 after MLP block (h9): h9 = Adapter2(h9)
    
    Hyperparameters: adapter_scale in [0.01, 0.1, 1, 10], 
                     adapter_bottleneck in [4, 8, 16, 32]
    Parameters: 0.165M ~ 1.198M
    """
    
    def __init__(self, embed_dim=768, num_layers=12, bottleneck_dim=16, scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale
        
        # Two adapters per layer
        self.msa_adapters = nn.ModuleList([
            BottleneckAdapter(embed_dim, bottleneck_dim, scale) 
            for _ in range(num_layers)
        ])
        self.mlp_adapters = nn.ModuleList([
            BottleneckAdapter(embed_dim, bottleneck_dim, scale) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x, vit_model):
        """Apply adapters to each Transformer layer."""
        B = x.shape[0]
        
        x = vit_model.patch_embed(x)
        cls_token = vit_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit_model.pos_embed
        
        for i, block in enumerate(vit_model.blocks):
            # LayerNorm 1
            z_norm = block.norm1(x)
            # MSA
            z_msa = block.attn(z_norm)
            # Residual + adapter after MSA
            x = x + z_msa
            x = self.msa_adapters[i](x)  # h5 = Adapter1(h5)
            
            # LayerNorm 2
            z_norm2 = block.norm2(x)
            # MLP
            z_mlp = block.mlp(z_norm2)
            # Residual + adapter after MLP
            x = x + z_mlp
            x = self.mlp_adapters[i](x)  # h9 = Adapter2(h9)
        
        cls_out = x[:, 0, :]
        if hasattr(vit_model, 'head'):
            return vit_model.head(cls_out)
        return cls_out
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


class PfeifferAdapter(nn.Module):
    """Pfeiffer Adapter: One adapter per layer, only after MLP.
    
    h9 = Adapter(h9)
    
    More efficient than Houlsby (only one adapter per layer).
    
    Hyperparameters: same as Houlsby
    Parameters: 0.082M ~ 0.599M
    """
    
    def __init__(self, embed_dim=768, num_layers=12, bottleneck_dim=16, scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale
        
        self.adapters = nn.ModuleList([
            BottleneckAdapter(embed_dim, bottleneck_dim, scale)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, vit_model):
        B = x.shape[0]
        
        x = vit_model.patch_embed(x)
        cls_token = vit_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit_model.pos_embed
        
        for i, block in enumerate(vit_model.blocks):
            # MSA block
            z_norm = block.norm1(x)
            x = x + block.attn(z_norm)
            
            # MLP block
            z_norm2 = block.norm2(x)
            z_mlp = block.mlp(z_norm2)
            x = x + z_mlp
            
            # Adapter after MLP
            x = self.adapters[i](x)
        
        cls_out = x[:, 0, :]
        if hasattr(vit_model, 'head'):
            return vit_model.head(cls_out)
        return cls_out
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


class AdaptFormer(nn.Module):
    """AdaptFormer: Adapter in parallel with MLP block.
    
    h9 = h9 + Adapter(h7)
    
    Domain-specific features from the adapter complement domain-agnostic 
    features from the original MLP block.
    
    Hyperparameters: adapter_scale in [0.05, 0.1, 0.2],
                     adapter_bottleneck in [4, 16, 32]
    Parameters: 0.082M ~ 0.599M
    """
    
    def __init__(self, embed_dim=768, num_layers=12, bottleneck_dim=16, scale=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale
        
        # Parallel adapters (one per layer, placed parallel to MLP)
        self.adapters = nn.ModuleList([
            BottleneckAdapter(embed_dim, bottleneck_dim, scale, use_residual=False)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, vit_model):
        B = x.shape[0]
        
        x = vit_model.patch_embed(x)
        cls_token = vit_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit_model.pos_embed
        
        for i, block in enumerate(vit_model.blocks):
            # MSA block
            z_norm = block.norm1(x)
            x = x + block.attn(z_norm)
            
            # MLP block with parallel adapter
            z_norm2 = block.norm2(x)
            z_mlp = block.mlp(z_norm2)
            
            # Adapter takes h7 (input to MLP, after LN)
            adapter_out = self.adapters[i](z_norm2)
            
            # h9 = h9 + Adapter(h7)
            x = x + z_mlp + adapter_out
        
        cls_out = x[:, 0, :]
        if hasattr(vit_model, 'head'):
            return vit_model.head(cls_out)
        return cls_out
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


class ConvPass(nn.Module):
    """ConvPass: Convolutional bypasses parallel to MSA and MLP.
    
    h5 = Convpass1(h2) + h5
    h9 = Convpass2(h7) + h9
    
    Uses 1x1 -> 3x3 -> 1x1 convolutions for visual inductive bias.
    
    Hyperparameters: adapter_scale in [0.01, 0.1, 1, 10, 100],
                     adapter_bottleneck in [8, 16],
                     xavier_init in [True, False]
    Parameters: 0.327M ~ 0.664M
    """
    
    def __init__(self, embed_dim=768, num_layers=12, bottleneck_dim=16, scale=1.0,
                 xavier_init=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale
        
        self.msa_convpass = nn.ModuleList([
            ConvPassAdapter(embed_dim, bottleneck_dim, scale)
            for _ in range(num_layers)
        ])
        self.mlp_convpass = nn.ModuleList([
            ConvPassAdapter(embed_dim, bottleneck_dim, scale)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, vit_model):
        B = x.shape[0]
        
        x = vit_model.patch_embed(x)
        cls_token = vit_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit_model.pos_embed
        
        for i, block in enumerate(vit_model.blocks):
            # h2: after first LN
            h2 = block.norm1(x)
            
            # MSA
            h4 = block.attn(h2)
            
            # h5 = h4 + x + Convpass1(h2)
            h5 = h4 + x + self.msa_convpass[i](h2)
            
            # h6: after second LN
            h6 = block.norm2(h5)
            
            # MLP output
            z_mlp = block.mlp(h6)
            
            # h9 = z_mlp + h5 + Convpass2(h6)
            x = z_mlp + h5 + self.mlp_convpass[i](h6)
        
        cls_out = x[:, 0, :]
        if hasattr(vit_model, 'head'):
            return vit_model.head(cls_out)
        return cls_out
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


class RepAdapter(nn.Module):
    """RepAdapter: Linear adapters with group-wise transformation.
    
    h5 = RepAdapter1(h2)
    h7 = RepAdapter2(h7)
    
    Sequential placement allows re-parameterization into backbone after training.
    
    Hyperparameters: adapter_scale in [0.1, 0.5, 1, 5, 10],
                     adapter_bottleneck in [8, 16, 32]
    Parameters: 0.239M ~ 0.903M
    """
    
    def __init__(self, embed_dim=768, num_layers=12, bottleneck_dim=16, scale=1.0,
                 num_groups=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale
        self.num_groups = num_groups
        
        # RepAdapter after MSA (processes h2)
        self.msa_adapters = nn.ModuleList([
            RepAdapterModule(embed_dim, bottleneck_dim, scale, num_groups)
            for _ in range(num_layers)
        ])
        
        # RepAdapter before MLP (processes h7)
        self.mlp_adapters = nn.ModuleList([
            RepAdapterModule(embed_dim, bottleneck_dim, scale, num_groups)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, vit_model):
        B = x.shape[0]
        
        x = vit_model.patch_embed(x)
        cls_token = vit_model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + vit_model.pos_embed
        
        for i, block in enumerate(vit_model.blocks):
            # h2: after first LN
            h2 = block.norm1(x)
            
            # RepAdapter after MSA: h5 = RepAdapter1(h2)
            h5_adapted = self.msa_adapters[i](h2)
            
            # MSA
            h4 = block.attn(h2)
            x = h4 + h5_adapted  # RepAdapter replaces the residual connection
            
            # h7: after second LN (before MLP)
            h6 = block.norm2(x)
            
            # RepAdapter: h7 = RepAdapter2(h7)
            # Actually h7 in the paper notation is the MLP input
            # RepAdapter2 processes h7 (after LN but before MLP)
            z_adapted = self.mlp_adapters[i](h6)
            
            # MLP
            z_mlp = block.mlp(h6)
            x = z_mlp + z_adapted  # RepAdapter replaces the residual
        
        cls_out = x[:, 0, :]
        if hasattr(vit_model, 'head'):
            return vit_model.head(cls_out)
        return cls_out
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
