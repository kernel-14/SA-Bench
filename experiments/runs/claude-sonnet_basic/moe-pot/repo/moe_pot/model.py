"""
MoE-POT: Mixture-of-Experts Pre-training Operator Transformer.

Main model architecture implementing:
1. Patchification layer with positional embeddings
2. Temporal aggregation layer with Fourier feature constants
3. N blocks, each containing:
   - Multi-head Fourier layer
   - MoE layer (shared + routed experts)
4. Output projection head

The model auto-regressively predicts the next PDE frame from T previous frames.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .fourier_layer import FourierLayer
from .moe_layer import MoELayer


class PatchEmbedding(nn.Module):
    """
    Patchification layer following Vision Transformer (ViT) style.

    Applies a convolutional embedding with kernel_size=P and stride=P,
    partitioning the spatial domain into non-overlapping P×P patches.
    Each patch is mapped to a d-dimensional embedding vector.

    Also adds learnable positional encodings that incorporate (x, y, t) coordinates.

    Args:
        in_channels: Number of input channels (C).
        embed_dim: Output embedding dimension (d).
        patch_size: Patch size P.
        max_timesteps: Maximum number of timesteps T.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 8,
        max_timesteps: int = 20,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Convolutional patch embedding: C -> embed_dim
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Learnable positional encoding: maps (x, y, t) -> embed_dim
        # W_p in R^{n x 3} where n = embed_dim
        self.pos_embed = nn.Linear(3, embed_dim)

        self.max_timesteps = max_timesteps

    def forward(self, u: torch.Tensor, t: int) -> torch.Tensor:
        """
        Args:
            u: Input frame (B, C, H, W).
            t: Timestep index (0-indexed).

        Returns:
            Z_p: Patch embeddings (B, H/P, W/P, embed_dim).
        """
        B, C, H, W = u.shape
        P = self.patch_size

        # Compute positional encodings
        # Create grid of (x, y) coordinates normalized to [0, 1]
        h_patches = H // P
        w_patches = W // P
        device = u.device

        # Grid coordinates: (h_patches, w_patches, 2)
        ys = torch.linspace(0, 1, h_patches, device=device)
        xs = torch.linspace(0, 1, w_patches, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        # Normalize timestep
        t_norm = t / max(self.max_timesteps - 1, 1)
        t_coord = torch.full((h_patches, w_patches, 1), t_norm, device=device)

        # Concatenate (x, y, t): (h_patches, w_patches, 3)
        coords = torch.stack([grid_x, grid_y, t_coord.squeeze(-1)], dim=-1)
        # Positional encoding: (h_patches, w_patches, embed_dim)
        pos = self.pos_embed(coords)

        # Patch embedding: (B, embed_dim, h_patches, w_patches)
        z = self.patch_embed(u)
        # Convert to (B, h_patches, w_patches, embed_dim)
        z = z.permute(0, 2, 3, 1)

        # Add positional encoding (broadcast over batch)
        z = z + pos.unsqueeze(0)

        return z


class TemporalAggregation(nn.Module):
    """
    Temporal aggregation layer that combines T frames into a single representation.

    For each spatial location, applies a learnable MLP W_t combined with
    Fourier feature constants gamma:

        z_agg = sum_t W_t * z_p^t * exp(-i * gamma * t)

    In practice, we implement this as a weighted sum with learnable weights
    and Fourier feature modulation.

    Args:
        embed_dim: Feature dimension.
        num_timesteps: Number of input timesteps T.
    """

    def __init__(self, embed_dim: int, num_timesteps: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps

        # Learnable temporal weights W_t: (T, embed_dim, embed_dim)
        # Implemented as a linear layer applied per timestep
        self.temporal_mlp = nn.Linear(embed_dim * num_timesteps, embed_dim)

        # Fourier feature constants gamma: (embed_dim,)
        # These are fixed (not learned) frequency constants
        self.register_buffer(
            "gamma",
            torch.randn(embed_dim) * 2 * math.pi
        )

    def forward(self, z_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            z_list: List of T patch embeddings, each (B, H, W, embed_dim).

        Returns:
            z_agg: Aggregated features (B, H, W, embed_dim).
        """
        T = len(z_list)
        B, H, W, C = z_list[0].shape

        # Apply Fourier modulation: z_p^t * exp(-i * gamma * t)
        # We use the real part: z_p^t * cos(gamma * t)
        modulated = []
        for t, z in enumerate(z_list):
            # gamma: (C,), t: scalar
            phase = torch.cos(self.gamma * t)  # (C,)
            modulated.append(z * phase)

        # Concatenate along channel dimension: (B, H, W, T*C)
        z_cat = torch.cat(modulated, dim=-1)

        # Apply temporal MLP: (B, H, W, T*C) -> (B, H, W, C)
        z_agg = self.temporal_mlp(z_cat)

        return z_agg


class MoEPOTBlock(nn.Module):
    """
    A single MoE-POT block containing:
    1. Multi-head Fourier layer
    2. MoE layer (shared + routed experts)

    Args:
        dim: Feature dimension.
        mlp_dim: MLP hidden dimension for experts.
        num_heads: Number of Fourier heads.
        modes: Number of Fourier modes.
        num_routed_experts: Number of routed experts.
        num_shared_experts: Number of shared experts.
        top_k: Number of experts to activate per input.
        balance_weight: Load balancing loss weight.
    """

    def __init__(
        self,
        dim: int,
        mlp_dim: int,
        num_heads: int = 4,
        modes: int = 16,
        num_routed_experts: int = 16,
        num_shared_experts: int = 2,
        top_k: int = 4,
        balance_weight: float = 0.1,
    ):
        super().__init__()
        self.fourier_layer = FourierLayer(
            dim=dim,
            num_heads=num_heads,
            modes=modes,
        )
        self.moe_layer = MoELayer(
            dim=dim,
            mlp_dim=mlp_dim,
            num_routed_experts=num_routed_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            balance_weight=balance_weight,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, H, W, C)

        Returns:
            output: (B, H, W, C)
            balance_loss: scalar
        """
        # Fourier layer (with residual connection inside)
        x = self.fourier_layer(x)
        # MoE layer
        x, balance_loss = self.moe_layer(x)
        return x, balance_loss

    def get_routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get routing weights for interpretability analysis."""
        # First pass through Fourier layer
        x_fourier = self.fourier_layer(x)
        return self.moe_layer.get_routing_weights(x_fourier)


class MoEPOT(nn.Module):
    """
    MoE-POT: Mixture-of-Experts Pre-training Operator Transformer.

    Architecture:
    1. Patchification + positional encoding (per timestep)
    2. Temporal aggregation
    3. N MoE-POT blocks (Fourier layer + MoE layer)
    4. Output projection

    The model takes T input frames and predicts the next frame.

    Args:
        in_channels: Number of input channels (C).
        out_channels: Number of output channels (defaults to in_channels).
        embed_dim: Embedding/feature dimension (attention_dim in paper).
        mlp_dim: MLP hidden dimension for experts.
        num_layers: Number of MoE-POT blocks (N).
        num_heads: Number of Fourier heads.
        modes: Number of Fourier modes.
        patch_size: Spatial patch size P.
        num_timesteps: Number of input timesteps T.
        num_routed_experts: Number of routed experts per block.
        num_shared_experts: Number of shared experts per block.
        top_k: Number of routed experts to activate per input.
        balance_weight: Load balancing loss weight.
        max_channels: Maximum number of channels across all datasets (for padding).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: Optional[int] = None,
        embed_dim: int = 512,
        mlp_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        modes: int = 16,
        patch_size: int = 8,
        num_timesteps: int = 10,
        num_routed_experts: int = 16,
        num_shared_experts: int = 2,
        top_k: int = 4,
        balance_weight: float = 0.1,
        max_channels: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        self.patch_size = patch_size

        # Patchification layer
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            max_timesteps=num_timesteps + 10,  # some buffer
        )

        # Temporal aggregation
        self.temporal_agg = TemporalAggregation(
            embed_dim=embed_dim,
            num_timesteps=num_timesteps,
        )

        # MoE-POT blocks
        self.blocks = nn.ModuleList([
            MoEPOTBlock(
                dim=embed_dim,
                mlp_dim=mlp_dim,
                num_heads=num_heads,
                modes=modes,
                num_routed_experts=num_routed_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                balance_weight=balance_weight,
            )
            for _ in range(num_layers)
        ])

        # Output projection: patch features -> pixel space
        # Upsample from (H/P, W/P, embed_dim) to (H, W, out_channels)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.ConvTranspose2d(
            embed_dim, self.out_channels,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(
        self,
        u_seq: torch.Tensor,
        noise_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Auto-regressive forward pass: predict next frame from T previous frames.

        Args:
            u_seq: Input sequence (B, T, C, H, W) - T frames.
            noise_scale: Scale for denoising pre-training noise injection.

        Returns:
            pred: Predicted next frame (B, C, H, W).
            total_balance_loss: Sum of load balancing losses across all blocks.
        """
        B, T, C, H, W = u_seq.shape

        # Inject noise during pre-training
        if noise_scale > 0 and self.training:
            norm = u_seq.norm(dim=(2, 3, 4), keepdim=True)
            noise = torch.randn_like(u_seq) * noise_scale * norm
            u_seq = u_seq + noise

        # Step 1: Patchification for each timestep
        z_patches = []
        for t in range(T):
            z_t = self.patch_embed(u_seq[:, t], t)  # (B, H/P, W/P, embed_dim)
            z_patches.append(z_t)

        # Step 2: Temporal aggregation
        z = self.temporal_agg(z_patches)  # (B, H/P, W/P, embed_dim)

        # Step 3: MoE-POT blocks
        total_balance_loss = torch.tensor(0.0, device=u_seq.device)
        for block in self.blocks:
            z, balance_loss = block(z)
            total_balance_loss = total_balance_loss + balance_loss

        # Step 4: Output projection
        z = self.output_norm(z)
        # (B, H/P, W/P, embed_dim) -> (B, embed_dim, H/P, W/P)
        z = z.permute(0, 3, 1, 2)
        # Upsample to original resolution: (B, out_channels, H, W)
        pred = self.output_proj(z)

        return pred, total_balance_loss

    def get_routing_weights_all_blocks(
        self, u_seq: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get routing weights from all blocks for interpretability analysis.

        Runs a single forward pass and collects routing weights at each block.
        The routing weights are computed from the input to each MoE layer
        (i.e., after the Fourier layer in each block).

        Args:
            u_seq: Input sequence (B, T, C, H, W).

        Returns:
            routing_weights: List of (B, N_r) tensors, one per block.
        """
        B, T, C, H, W = u_seq.shape

        # Patchification
        z_patches = []
        for t in range(T):
            z_t = self.patch_embed(u_seq[:, t], t)
            z_patches.append(z_t)

        # Temporal aggregation
        z = self.temporal_agg(z_patches)

        # Collect routing weights from each block during a single forward pass
        routing_weights = []
        for block in self.blocks:
            # Pass through Fourier layer
            z_fourier = block.fourier_layer(z)
            # Get routing weights from MoE layer (before expert computation)
            rw = block.moe_layer.get_routing_weights(z_fourier)
            routing_weights.append(rw)
            # Complete the MoE layer forward pass
            z_moe, _ = block.moe_layer(z_fourier)
            z = z_moe

        return routing_weights

    def compute_loss(
        self,
        u_seq: torch.Tensor,
        u_target: torch.Tensor,
        noise_scale: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the full training loss including prediction loss and balance loss.

        Loss = ||G_w(u^{<t} + eps) - u^t||_2^2 + sum_l L_balance^l

        Args:
            u_seq: Input sequence (B, T, C, H, W).
            u_target: Target next frame (B, C, H, W).
            noise_scale: Noise injection scale epsilon.

        Returns:
            total_loss: Combined loss.
            pred_loss: Prediction MSE loss.
            balance_loss: Load balancing loss.
        """
        pred, balance_loss = self.forward(u_seq, noise_scale=noise_scale)

        # L2 prediction loss
        pred_loss = F.mse_loss(pred, u_target)

        total_loss = pred_loss + balance_loss

        return total_loss, pred_loss, balance_loss


def create_moe_pot_tiny(in_channels: int = 4, **kwargs) -> MoEPOT:
    """Create MoE-POT-Tiny (30M total, 17M activated)."""
    return MoEPOT(
        in_channels=in_channels,
        embed_dim=512,
        mlp_dim=512,
        num_layers=4,
        num_heads=4,
        num_routed_experts=16,
        num_shared_experts=2,
        top_k=4,
        **kwargs,
    )


def create_moe_pot_small(in_channels: int = 4, **kwargs) -> MoEPOT:
    """Create MoE-POT-Small (166M total, 90M activated)."""
    return MoEPOT(
        in_channels=in_channels,
        embed_dim=1024,
        mlp_dim=1024,
        num_layers=6,
        num_heads=8,
        num_routed_experts=16,
        num_shared_experts=2,
        top_k=4,
        **kwargs,
    )


def create_moe_pot_medium(in_channels: int = 4, **kwargs) -> MoEPOT:
    """Create MoE-POT-Medium (489M total, 288M activated)."""
    return MoEPOT(
        in_channels=in_channels,
        embed_dim=1024,
        mlp_dim=2048,
        num_layers=8,
        num_heads=8,
        num_routed_experts=16,
        num_shared_experts=2,
        top_k=4,
        **kwargs,
    )
