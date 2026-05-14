"""
MoE-POT: Mixture-of-Experts Pre-training Operator Transformer.

Main model architecture combining:
1. Patchification + Temporal Aggregation (input encoding)
2. N blocks of [FourierLayer + MoELayer]
3. Output projection

Based on the DPOT [15] architecture with MoE layers replacing dense FFNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from .patch_embed import PatchEmbed, TemporalAggregation
from .fourier_layer import FourierLayer
from .moe_layer import MoELayer


class MoEPOTBlock(nn.Module):
    """
    A single block of MoE-POT, containing:
    - Fourier Layer (multi-head spectral mixing)
    - MoE Layer (sparse expert computation)
    
    As shown in Figure 3, each block consists of a Fourier layer followed by a MoE layer.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mode: int = 32,
        num_routed_experts: int = 16,
        num_shared_experts: int = 2,
        top_k: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.fourier = FourierLayer(dim, num_heads=num_heads, mode=mode)
        self.moe = MoELayer(
            dim=dim,
            num_routed_experts=num_routed_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
        )
        
        # Layer norms for stability
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map
            
        Returns:
            out: [B, C, H, W] processed features
        """
        # Fourier layer with residual connection
        B, C, H, W = x.shape
        
        # LayerNorm (applied in channel-first format)
        x_norm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        fourier_out = self.fourier(x_norm)
        x = x + self.dropout(fourier_out)
        
        # MoE layer with residual connection
        x_norm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        moe_out = self.moe(x_norm)
        x = x + self.dropout(moe_out)
        
        return x
    
    def get_load_balancing_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss for this block's MoE layer."""
        return self.moe.get_load_balancing_loss(x)


class MoEPOT(nn.Module):
    """
    Mixture-of-Experts Pre-training Operator Transformer.
    
    Full model architecture as described in Section 4 and Figure 3:
    
    1. PatchEmbed: spatial downsampling via convolution
    2. TemporalAggregation: cross-timestep feature fusion
    3. N x MoEPOTBlock: alternating Fourier + MoE layers
    4. Output projection: back to original resolution and channels
    
    Model configurations (Table 5):
    - Tiny:  dim=512,  mlp_dim=512,  layers=4, heads=4, size=30M,  activated=17M
    - Small: dim=1024, mlp_dim=1024, layers=6, heads=8, size=166M, activated=90M
    - Medium:dim=1024, mlp_dim=2048, layers=8, heads=8, size=489M, activated=288M
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_size: int = 128,
        patch_size: int = 8,
        T: int = 10,
        dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 4,
        mode: int = 32,
        num_routed_experts: int = 16,
        num_shared_experts: int = 2,
        top_k: int = 4,
        dropout: float = 0.0,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_size = spatial_size
        self.patch_size = patch_size
        self.T = T
        self.dim = dim
        self.num_layers = num_layers
        
        # Effective spatial size after patchification
        self.effective_size = spatial_size // patch_size
        
        # 1. Patch embedding (spatial downsampling)
        self.patch_embed = PatchEmbed(in_channels, dim, patch_size=patch_size)
        
        # 2. Positional encoding (learnable)
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, dim, self.effective_size, self.effective_size) * 0.02
            )
        else:
            self.pos_encoding = None
        
        # 3. Temporal aggregation
        self.temporal_agg = TemporalAggregation(dim, T=T)
        
        # 4. MoE-POT blocks
        self.blocks = nn.ModuleList([
            MoEPOTBlock(
                dim=dim,
                num_heads=num_heads,
                mode=mode,
                num_routed_experts=num_routed_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # 5. Output projection: back to spatial resolution and channels
        # Decoder: upsample back to original resolution
        self.output_proj = nn.Sequential(
            nn.ConvTranspose2d(
                dim, dim // 2,
                kernel_size=patch_size, stride=patch_size
            ),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, out_channels, kernel_size=3, padding=1),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
                
    def forward(
        self, 
        x: torch.Tensor, 
        return_routing: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of MoE-POT.
        
        Args:
            x: [B, T, C_in, H, W] input spatiotemporal tensor
                - B: batch size
                - T: number of input timesteps (10)
                - C_in: input channels
                - H, W: spatial dimensions (128)
            return_routing: if True, return routing weights from all blocks
            
        Returns:
            out: [B, C_out, H, W] predicted next frame
            routing_info (optional): list of routing weight tensors per block
        """
        B, T, C_in, H, W = x.shape
        
        # 1. Patchify each timestep
        # Process each timestep independently through patch embed
        patches = []
        for t in range(T):
            x_t = x[:, t]  # [B, C_in, H, W]
            p_t = self.patch_embed(x_t)  # [B, dim, H/P, W/P]
            patches.append(p_t)
        
        # Stack: [B, T, dim, H/P, W/P]
        patches = torch.stack(patches, dim=1)
        
        # Add positional encoding to each timestep
        if self.pos_encoding is not None:
            patches = patches + self.pos_encoding.unsqueeze(1)
        
        # 2. Temporal aggregation
        z = self.temporal_agg(patches)  # [B, dim, H/P, W/P]
        
        # 3. Process through MoE-POT blocks
        routing_info = []
        for block in self.blocks:
            z = block(z)
            if return_routing:
                routing_info.append({
                    'weights': block.moe.routing_weights,
                    'indices': block.moe.routing_indices,
                })
        
        # 4. Output projection (decode back to spatial resolution and channels)
        out = self.output_proj(z)  # [B, C_out, H, W]
        
        if return_routing:
            return out, routing_info
        return out
    
    def get_load_balancing_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the total load balancing loss across all blocks.
        
        As described in Section 4:
        L_balance = Σ_l w_bal · CV({Importance_i^l})^2
        """
        B, T, C_in, H, W = x.shape
        
        # Patchify and aggregate to get features at each block
        patches = []
        for t in range(T):
            x_t = x[:, t]
            p_t = self.patch_embed(x_t)
            patches.append(p_t)
        patches = torch.stack(patches, dim=1)
        
        if self.pos_encoding is not None:
            patches = patches + self.pos_encoding.unsqueeze(1)
        
        z = self.temporal_agg(patches)
        
        total_loss = 0.0
        for block in self.blocks:
            # Pass through block and collect load balancing loss
            z = block(z)
            total_loss += block.moe.get_load_balancing_loss(z)
            
        return total_loss / len(self.blocks)
    
    def compute_l2_relative_error(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute L2 Relative Error (L2RE), the primary evaluation metric.
        
        L2RE = ||pred - target||_2 / ||target||_2
        
        As used in all experiments (Tables 1-4, etc.)
        """
        diff = pred - target
        rel_error = torch.norm(diff) / (torch.norm(target) + 1e-8)
        return rel_error


def create_moe_pot_tiny(**kwargs) -> MoEPOT:
    """Create MoE-POT-Tiny: 30M total params, 17M activated."""
    defaults = dict(
        dim=512,
        num_heads=4,
        num_layers=4,
        num_routed_experts=16,
        num_shared_experts=2,
        top_k=4,
        patch_size=8,
    )
    defaults.update(kwargs)
    return MoEPOT(**defaults)


def create_moe_pot_small(**kwargs) -> MoEPOT:
    """Create MoE-POT-Small: 166M total params, 90M activated."""
    defaults = dict(
        dim=1024,
        num_heads=8,
        num_layers=6,
        num_routed_experts=16,
        num_shared_experts=2,
        top_k=4,
        patch_size=8,
    )
    defaults.update(kwargs)
    return MoEPOT(**defaults)


def create_moe_pot_medium(**kwargs) -> MoEPOT:
    """Create MoE-POT-Medium: 489M total params, 288M activated."""
    defaults = dict(
        dim=1024,
        num_heads=8,
        num_layers=8,
        num_routed_experts=16,
        num_shared_experts=2,
        top_k=4,
        patch_size=8,
    )
    defaults.update(kwargs)
    return MoEPOT(**defaults)
