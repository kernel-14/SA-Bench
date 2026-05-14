## model.py
"""MoE‑POT neural operator architecture.

Implements the Mixture‑of‑Experts Pre‑training Operator Transformer as
described in the paper. The model consists of:
    * PatchEmbed with learnable coordinate‑based positional encodings.
    * TemporalAgg – Fourier‑feature‑based aggregation over input frames.
    * N blocks, each containing a multi‑head FourierLayer and a
      sparsely‑gated MoEBlock (shared + top‑K routed experts).
    * A transposed convolution decoder.

All shape assumptions follow the global config, e.g. input shape
(B, T_in, H=128, W=128, C_max=5), T_in=10, patch_size=8, etc.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# We only import the Config class for type hinting, not for module logic.
# The actual construction receives a Config instance.
from config import Config


class PatchEmbed(nn.Module):
    """Per‑frame spatial patchification with coordinate‑based positional encoding.

    For each time step ``t``, the input ``u^t`` of shape ``(C_in, H, W)`` is
    combined with a learned positional encoding ``p^t`` and then projected
    into ``patches`` using a strided convolution.

    Args:
        in_channels (int): number of input channels (max_channels, e.g. 5).
        attention_dim (int): output feature dimension.
        patch_size (int): kernel size and stride of the convolution.
        H (int): spatial height (128).
        W (int): spatial width (128).
        T_in (int): number of input time frames (10).
    """

    def __init__(
        self,
        in_channels: int,
        attention_dim: int,
        patch_size: int,
        H: int,
        W: int,
        T_in: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.attention_dim = attention_dim
        self.patch_size = patch_size
        self.H = H
        self.W = W
        self.T_in = T_in

        # Learnable linear mapping from (x, y, t) coordinates to channel‑wise
        # positional embeddings.  W_p : (in_channels, 3)
        self.W_p = nn.Parameter(torch.randn(in_channels, 3) * 0.01)

        # Convolution for patch embedding
        self.conv = nn.Conv2d(
            in_channels,
            attention_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # Pre‑compute static spatial coordinate grids (normalised to [-1, 1]).
        # We use meshgrid to obtain shape (H, W) for x and y.
        xs = torch.linspace(-1.0, 1.0, W)
        ys = torch.linspace(-1.0, 1.0, H)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W) each
        self.register_buffer('grid_x', grid_x.float().unsqueeze(-1))  # (H, W, 1)
        self.register_buffer('grid_y', grid_y.float().unsqueeze(-1))  # (H, W, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward embedding.

        Args:
            x: Tensor of shape ``(B, T, H, W, C)`` with raw input frames.

        Returns:
            Tensor of shape ``(B, T, H', W', attention_dim)`` where
            ``H' = H // patch_size``, ``W' = W // patch_size``.
        """
        B, T, H, W, _ = x.shape
        assert H == self.H and W == self.W, "Input resolution mismatch."

        # Pre‑compute the time scalar normalised to [-1, 1] for each t.
        t_vals = torch.linspace(-1.0, 1.0, T, device=x.device)
        # Spatial coordinates remain static, just expand to (H, W, 3):
        #   channel 0 = X, channel 1 = Y, channel 2 = t (broadcast per frame later)
        coords_spatial = torch.cat([self.grid_x, self.grid_y], dim=-1)  # (H, W, 2)

        frame_features = []
        for t_idx in range(T):
            # Build coordinate tensor of shape (H, W, 3)
            t_coord = torch.full((H, W, 1), t_vals[t_idx], device=x.device)
            grid_t = torch.cat([coords_spatial, t_coord], dim=-1)  # (H, W, 3)

            # Compute positional encoding: p = grid @ W_p^T   (H,W,3) @ (3,in_channels) -> (H,W,in_channels)
            pos_enc = torch.matmul(grid_t, self.W_p.t())  # (H, W, in_channels)

            # Add positional encoding to the current frame
            u_t = x[:, t_idx]  # (B, H, W, in_channels)
            u_t = u_t + pos_enc.unsqueeze(0)  # broadcast over batch

            # Permute to (B, in_channels, H, W) for convolution
            u_t = u_t.permute(0, 3, 1, 2).contiguous()
            feat = self.conv(u_t)  # (B, attention_dim, H', W')

            # Bring back to (B, H', W', attention_dim) then stack across T
            feat = feat.permute(0, 2, 3, 1).unsqueeze(1)  # (B, 1, H', W', attention_dim)
            frame_features.append(feat)

        # Concatenate along time dimension -> (B, T, H', W', attention_dim)
        return torch.cat(frame_features, dim=1)


class TemporalAgg(nn.Module):
    """Fourier‑feature based temporal aggregation.

    Combines the ``T`` embedded frames into a single feature map
    ``z_agg`` using a learnable per‑time‑step weighting modulated by
    a learned frequency vector ``γ``.

    Args:
        attention_dim (int): dimension of the frame features.
        T_in (int): number of input time frames.
    """

    def __init__(self, attention_dim: int, T_in: int) -> None:
        super().__init__()
        self.attention_dim = attention_dim
        self.T_in = T_in

        # Learnable Fourier frequency vector (one frequency per channel)
        self.gamma = nn.Parameter(torch.randn(attention_dim) * 0.01)

        # Per‑timestep linear layer: takes concatenation of real/imag
        # modulated features (2 * attention_dim) and outputs attention_dim.
        self.W_t = nn.ModuleList(
            nn.Linear(2 * attention_dim, attention_dim, bias=False)
            for _ in range(T_in)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward aggregation.

        Args:
            x: Tensor of shape ``(B, T, H', W', attention_dim)``.

        Returns:
            Aggregated tensor of shape ``(B, H', W', attention_dim)``.
        """
        B, T, Hp, Wp, C = x.shape

        agg = 0.0
        for t in range(T):
            x_t = x[:, t]  # (B, H', W', C)
            # Element‑wise cosine and sine modulation
            cos_term = torch.cos(self.gamma * t).view(1, 1, 1, C)
            sin_term = torch.sin(self.gamma * t).view(1, 1, 1, C)
            real_part = x_t * cos_term
            imag_part = x_t * sin_term
            concat = torch.cat([real_part, imag_part], dim=-1)  # (B, H', W', 2C)
            # Apply per‑timestep linear
            out_t = self.W_t[t](concat)  # (B, H', W', C)
            agg = agg + out_t

        return agg


class FourierLayer(nn.Module):
    """Multi‑head frequency‑domain mixing layer.

    Splits the channel dimension into ``n_heads`` groups, applies a
    learnable transformation in the Fourier domain independently per
    head, and concatenates the results.

    Args:
        n_heads (int): number of attention heads.
        attention_dim (int): total feature dimension (must be divisible
            by ``n_heads``).
    """

    def __init__(self, n_heads: int, attention_dim: int) -> None:
        super().__init__()
        if attention_dim % n_heads != 0:
            raise ValueError(
                f"attention_dim ({attention_dim}) must be divisible by "
                f"n_heads ({n_heads})."
            )
        self.n_heads = n_heads
        self.head_dim = attention_dim // n_heads

        # Each head has its own small MLP operating on real‑imag stacked
        # frequency coefficients: input (2 * head_dim), output (2 * head_dim).
        self.head_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * self.head_dim, 2 * self.head_dim),
                nn.GELU(),
                nn.Linear(2 * self.head_dim, 2 * self.head_dim),
            )
            for _ in range(n_heads)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward Fourier mixing.

        Args:
            x: Tensor of shape ``(B, H', W', attention_dim)``.

        Returns:
            Transformed tensor of the same shape.
        """
        B, H, W, C = x.shape
        assert C == self.n_heads * self.head_dim, "Channel dimension mismatch."

        # Reshape to (B, H, W, n_heads, head_dim)
        x = x.view(B, H, W, self.n_heads, self.head_dim)
        # Permute for head‑wise processing: (B, n_heads, H, W, head_dim)
        x = x.permute(0, 3, 1, 2, 4).contiguous()

        # Collect outputs for each head
        head_outputs = []
        for h in range(self.n_heads):
            z_h = x[:, h]  # (B, H, W, head_dim)

            # 2D real FFT: last two spatial dims are H, W
            F_h = torch.fft.rfft2(z_h, dim=(-2, -1))  # complex, shape (B, H, W//2+1, head_dim)

            # Convert to real‑imag stacked tensor
            F_real = torch.view_as_real(F_h)  # (B, H, W//2+1, head_dim, 2)
            # Merge last two dims: (B, H, W//2+1, 2*head_dim)
            B_h, H_h, W_half, _, _ = F_real.shape
            F_stacked = F_real.reshape(B_h, H_h, W_half, 2 * self.head_dim)

            # Apply per‑head MLP (point‑wise across spatial/frequency locations)
            F_stacked = F_stacked.view(-1, 2 * self.head_dim)  # flatten spatial dims
            F_out = self.head_mlps[h](F_stacked)
            F_out = F_out.view(B_h, H_h, W_half, 2 * self.head_dim)

            # Reshape back to complex representation
            F_out = F_out.view(B_h, H_h, W_half, self.head_dim, 2)
            F_out = torch.view_as_complex(F_out.contiguous())  # (B, H, W//2+1, head_dim)

            # Inverse FFT
            z_out = torch.fft.irfft2(
                F_out, s=(H, W), dim=(-2, -1)
            )  # (B, H, W, head_dim)
            head_outputs.append(z_out)

        # Concatenate heads -> (B, H, W, n_heads*head_dim) = (B, H, W, attention_dim)
        out = torch.cat(head_outputs, dim=-1)
        return out


class ExpertCNN(nn.Module):
    """A single expert implemented as a two‑layer CNN.

    The input and output feature maps have the same shape.
    ``GELU`` activation is used between convolutions.

    Args:
        attention_dim (int): number of input/output channels.
        mlp_dim (int): hidden dimension.
    """

    def __init__(self, attention_dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(attention_dim, mlp_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mlp_dim, attention_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the expert to a feature map.

        Args:
            x: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of same shape.
        """
        return self.net(x)


class Router(nn.Module):
    """Light‑weight CNN that produces routing logits for the MoE.

    Architecture: two strided convolutions with ReLU, global average
    pool, and a final linear projection.

    Args:
        attention_dim (int): input feature dimension.
        n_routed (int): number of routed experts (output logits).
    """

    def __init__(self, attention_dim: int, n_routed: int) -> None:
        super().__init__()
        mid_dim = attention_dim // 2
        low_dim = attention_dim // 4

        self.conv1 = nn.Conv2d(attention_dim, mid_dim, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(mid_dim, low_dim, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(low_dim, n_routed)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Produce routing logits.

        Args:
            x: Feature map of shape ``(B, H', W', attention_dim)``.

        Returns:
            Logits tensor of shape ``(B, n_routed)``.
        """
        # Input is expected in channels‑last format from the block,
        # but this router operates on channels‑first (N,C,H,W).
        # We'll first permute.
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, attention_dim, H', W')
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class MoEBlock(nn.Module):
    """Mixture‑of‑Experts block with shared and top‑K routed experts.

    Args:
        attention_dim (int): feature dimension.
        mlp_dim (int): hidden dimension inside each expert CNN.
        n_shared (int): number of shared experts (always active).
        n_routed (int): number of routed experts (top‑K selected).
        top_k (int): number of routed experts to activate per sample.
        bal_weight (float): weight for the load‑balancing auxiliary loss.
    """

    def __init__(
        self,
        attention_dim: int,
        mlp_dim: int,
        n_shared: int = 2,
        n_routed: int = 16,
        top_k: int = 4,
        bal_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention_dim = attention_dim
        self.n_routed = n_routed
        self.top_k = top_k
        self.bal_weight = bal_weight

        # Router (logits for each routed expert)
        self.router = Router(attention_dim, n_routed)

        # Shared experts
        self.shared_experts = nn.ModuleList(
            ExpertCNN(attention_dim, mlp_dim) for _ in range(n_shared)
        )
        # Routed experts
        self.routed_experts = nn.ModuleList(
            ExpertCNN(attention_dim, mlp_dim) for _ in range(n_routed)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Process the feature map and return output + auxiliary loss.

        Args:
            x: Feature map of shape ``(B, H', W', attention_dim)``.

        Returns:
            Tuple ``(output, balance_loss)`` where
            ``output`` has the same shape as ``x``.
        """
        B, H, W, C = x.shape

        # 1. Router – produce logits and full softmax weights
        logits = self.router(x)  # (B, n_routed)
        w_full = F.softmax(logits, dim=-1)  # routing probabilities before top‑K

        # 2. Top‑K masking (non‑renormalised)
        topk_vals, topk_idx = torch.topk(w_full, self.top_k, dim=-1)
        mask = torch.zeros_like(w_full).scatter_(-1, topk_idx, 1.0)
        w_routed = w_full * mask

        # 3. Shared experts (average their outputs)
        # Convert x to channels‑first for conv experts
        x_cf = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        shared_out = 0.0
        for exp in self.shared_experts:
            shared_out = shared_out + exp(x_cf)
        shared_out = shared_out / len(self.shared_experts)  # (B, C, H, W)

        # 4. Routed experts – evaluate all, then weighted sum
        # Stack all expert outputs: (B, n_routed, C, H, W)
        expert_outputs = []
        for exp in self.routed_experts:
            expert_outputs.append(exp(x_cf))
        all_experts = torch.stack(expert_outputs, dim=1)  # (B, n_routed, C, H, W)

        # Weighted sum (broadcast w_routed over spatial/channel dims)
        routed_out = torch.sum(
            w_routed.view(B, self.n_routed, 1, 1, 1) * all_experts, dim=1
        )  # (B, C, H, W)

        # Combine and convert back to channels‑last
        output = (shared_out + routed_out).permute(0, 2, 3, 1)  # (B, H, W, C)

        # 5. Load‑balancing loss (coefficient of variation squared)
        importance = w_full.sum(dim=0)  # (n_routed,)
        eps = 1e-8
        imp_mean = importance.mean()
        imp_std = torch.std(importance, unbiased=False)  # empirical std
        cv = imp_std / (imp_mean + eps)
        balance_loss = self.bal_weight * (cv ** 2)

        return output, balance_loss


class Model(nn.Module):
    """The complete MoE‑POT neural operator.

    Assembles the patch embedding, temporal aggregation, multiple
    Fourier+MoE blocks, and a decoder into an autoregressive prediction
    model.

    Args:
        config (Config): configuration dataclass containing all
            hyperparameters.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        # Input embedding
        self.patch_embed = PatchEmbed(
            in_channels=config.max_channels,
            attention_dim=config.attention_dim,
            patch_size=config.patch_size,
            H=config.spatial_resolution[0],
            W=config.spatial_resolution[1],
            T_in=config.input_frames,
        )

        # Temporal aggregation
        self.temporal_agg = TemporalAgg(
            attention_dim=config.attention_dim,
            T_in=config.input_frames,
        )

        # Fourier layers & MoE blocks
        self.fourier_layers = nn.ModuleList(
            FourierLayer(config.n_heads, config.attention_dim)
            for _ in range(config.n_blocks)
        )
        self.moe_blocks = nn.ModuleList(
            MoEBlock(
                attention_dim=config.attention_dim,
                mlp_dim=config.mlp_dim,
                n_shared=config.shared_experts,
                n_routed=config.routed_experts,
                top_k=config.top_k,
                bal_weight=config.pretrain_load_balance_weight,  # pre‑train weight
            )
            for _ in range(config.n_blocks)
        )

        # Decoder: transposed conv back to original resolution and max_channels
        self.decoder = nn.ConvTranspose2d(
            config.attention_dim,
            config.max_channels,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returning the predicted next frame and total
        load‑balance loss.

        Args:
            x: Input tensor of shape
                ``(B, T_in, H, W, max_channels)``.

        Returns:
            Tuple ``(prediction, balance_loss)``.
            ``prediction``: next frame of shape ``(B, max_channels, H, W)``.
            ``balance_loss``: scalar sum of per‑MoE‑block balancing losses.
        """
        # 1. Patch embedding + positional encoding
        x = self.patch_embed(x)  # (B, T, H', W', attention_dim)

        # 2. Temporal aggregation
        x = self.temporal_agg(x)  # (B, H', W', attention_dim)

        # 3. Blocks
        total_bal_loss = 0.0
        for fourier_layer, moe_block in zip(self.fourier_layers, self.moe_blocks):
            x = fourier_layer(x)  # still (B, H', W', attention_dim)
            x, bal = moe_block(x)  # output same shape, bal is scalar
            total_bal_loss = total_bal_loss + bal

        # 4. Decoder
        # Convert channels‑last to channels‑first for ConvTranspose2d
        x_cf = x.permute(0, 3, 1, 2).contiguous()   # (B, attention_dim, H', W')
        out = self.decoder(x_cf)                     # (B, max_channels, H, W)

        return out, total_bal_loss

    def freeze_router(self) -> None:
        """Set ``requires_grad=False`` for all router parameters.

        Used during fine‑tuning so that only experts and other layers
        are updated.
        """
        for block in self.moe_blocks:
            for param in block.router.parameters():
                param.requires_grad_(False)

    def unfreeze_router(self) -> None:
        """Re‑enable gradient tracking for all router parameters."""
        for block in self.moe_blocks:
            for param in block.router.parameters():
                param.requires_grad_(True)


# Export the main class for use in other modules
__all__ = ["Model", "MoEBlock", "FourierLayer", "ExpertCNN", "Router",
           "PatchEmbed", "TemporalAgg"]
