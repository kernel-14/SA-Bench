import torch
import torch.nn as nn
from typing import List, Tuple

from layers import Patchification, TemporalAggregation
from modules import Block


class MoEPOT(nn.Module):
    """Mixture-of-Experts Pre-training Operator Transformer.

    Processes spatiotemporal PDE data through:
    1. Patchification + positional encoding
    2. Temporal aggregation (Fourier-inspired)
    3. N blocks of (Multi-head Fourier layer + MoE layer)
    4. Decoder to predict next frame

    Input: (B, C_in, H, W, T)
    Output: (B, C_out, H, W)
    """

    def __init__(self, in_channels: int, out_channels: int, dim: int, mlp_dim: int,
                 num_layers: int, num_heads: int, num_routed_experts: int = 16,
                 num_shared_experts: int = 2, top_k: int = 4, patch_size: int = 8,
                 fourier_modes: int = 16, spatial_resolution: int = 128,
                 expert_kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.patch_size = patch_size
        self.spatial_resolution = spatial_resolution
        self.num_timesteps_in = 10  # default, overridden by training

        # Input encoding
        self.patchify = Patchification(in_channels, dim, patch_size)

        # Temporal aggregation
        self.temporal_agg = TemporalAggregation(dim, self.num_timesteps_in)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, num_routed_experts, num_shared_experts,
                  top_k, fourier_modes, expert_kernel_size)
            for _ in range(num_layers)
        ])

        # Decoder: project back to original resolution and output channels
        h_out = spatial_resolution // patch_size
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(mlp_dim, out_channels, kernel_size=3, padding=1),
        )

    def set_num_timesteps(self, T: int):
        """Update temporal aggregation for different number of input timesteps."""
        self.num_timesteps_in = T
        self.temporal_agg = TemporalAggregation(self.dim, T).to(
            next(self.parameters()).device)

    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            u: (B, C, H, W, T) input spatiotemporal data

        Returns:
            pred: (B, C_out, H, W) predicted next frame
            total_lb_loss: scalar load balancing loss summed over layers
        """
        B, C, H, W, T = u.shape

        # Patchification: (B, d, H/P, W/P, T)
        z_p = self.patchify(u)

        # Temporal aggregation: (B, d, H/P, W/P)
        z_agg = self.temporal_agg(z_p)

        # Blocks
        z = z_agg
        total_lb_loss = torch.tensor(0.0, device=u.device)
        for block in self.blocks:
            z, lb_loss = block(z)
            total_lb_loss = total_lb_loss + lb_loss

        # Decode: (B, d, H/P, W/P) -> (B, C_out, H, W)
        pred = self.decoder(z)

        return pred, total_lb_loss

    def autoregressive_rollout(self, u_init: torch.Tensor,
                                num_rollout_steps: int) -> List[torch.Tensor]:
        """Perform auto-regressive rollout prediction.

        Args:
            u_init: (B, C, H, W, T) initial T frames
            num_rollout_steps: number of future frames to predict

        Returns:
            predictions: list of predicted frames, each (B, C, H, W)
        """
        predictions = []
        u_current = u_init
        for _ in range(num_rollout_steps):
            pred, _ = self.forward(u_current)
            predictions.append(pred)
            # Shift input window: drop oldest frame, append prediction
            u_current = torch.cat([
                u_current[..., 1:],
                pred.unsqueeze(-1)
            ], dim=-1)
        return predictions
